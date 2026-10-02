//! `asphodel import`: a copy of Hermes' `state.db` and the private manifest
//! into a corpus.
//!
//! It reads only the rows and columns it needs, checks the schema it was
//! written against and fails loudly on a mismatch, and never selects the system
//! prompt or `api_content`. Each turn becomes a prefetch at the user message's
//! time and a `sync_turn` at the final assistant reply's time, with tool rows
//! skipped.
//!
//! Only rows with `active = 1 OR compacted = 1` are read, the predicate Hermes
//! uses for search, in timestamp order. So compacted turns replay, while rows
//! with `active=0, compacted=0` (the originals of a tail a compaction carried
//! forward, which have a live clone, and turns a rewind discarded) never do.
//! Carried-tail clones keep their original timestamps under later ids, so
//! timestamp order replays them when they were said. Hermes' summary row
//! (`_compressed_summary=1`) is skipped, and a clear is emitted at its time.
//! Cron sessions get prefetch only, and subagent sessions (`parent_session_id`
//! set) are skipped.

use std::path::Path;

use anyhow::{Context as _, bail};
use jiff::Timestamp;
use regex::Regex;
use rusqlite::{Connection, OpenFlags, OptionalExtension as _};
use serde_json::Value;

use super::corpus::{self, Counts, Event, Header, Owner};
use super::manifest::{Manifest, MemoryBlock};
use super::scenario::Author;
use super::timeline::SessionClass;
use crate::cli::ImportArgs;

/// The Hermes schema versions the importer was written against. A
/// `state.db` at any other version is refused rather than guessed at.
pub const HERMES_SCHEMA_VERSIONS: [i64; 1] = [31];

/// The columns the importer reads, and nothing else, per table, with the
/// type each is declared with in Hermes' DDL.
const REQUIRED_COLUMNS: [(&str, &[(&str, &str)]); 3] = [
    (
        "sessions",
        &[
            ("id", "TEXT"),
            ("source", "TEXT"),
            ("parent_session_id", "TEXT"),
            ("started_at", "REAL"),
        ],
    ),
    (
        "messages",
        &[
            ("id", "INTEGER"),
            ("session_id", "TEXT"),
            ("role", "TEXT"),
            ("content", "TEXT"),
            ("timestamp", "REAL"),
            ("active", "INTEGER"),
            ("compacted", "INTEGER"),
            ("_compressed_summary", "INTEGER"),
            ("tool_calls", "TEXT"),
        ],
    ),
    ("schema_version", &[("version", "INTEGER")]),
];

/// The gateway puts a channel's recent history and this marker in front of the
/// user text on backfill.
const NEW_MESSAGE_MARKER: &str = "[New message]";

/// How Hermes stores multimodal content: this prefix and a JSON parts
/// list.
const MULTIMODAL_PREFIX: &str = "\u{0}json:";

/// The session `source` Hermes gives cron runs.
const CRON_SOURCE: &str = "cron";

/// Runs the command: exit 0 with the counts on stdout, 2 when refused.
pub fn run(args: ImportArgs) -> anyhow::Result<()> {
    match execute(&args) {
        Ok(()) => Ok(()),
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::exit(2)
        }
    }
}

fn execute(args: &ImportArgs) -> anyhow::Result<()> {
    let dir = super::private_dir(args.replay_dir.as_deref())?;
    // Both inputs are real history, so both stay inside the private dir,
    // checked before either is read.
    let manifest_path = super::inside_private(&dir, &args.manifest, "the manifest")?;
    let state_db = super::inside_private(&dir, &args.state_db, "the state.db copy")?;
    let manifest = super::manifest::load(&manifest_path)?;
    let out = match &args.out {
        Some(path) => super::inside_private(&dir, path, "the corpus")?,
        None => {
            let corpora = dir.join("corpus");
            super::refuse_symlink(&corpora)?;
            std::fs::create_dir_all(&corpora)
                .with_context(|| format!("creating {}", corpora.display()))?;
            let stem = args
                .state_db
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .filter(|stem| !stem.is_empty())
                .unwrap_or_else(|| "history".into());
            corpora.join(format!("{stem}.jsonl"))
        }
    };
    super::refuse_symlink(&out)?;

    let conn = open_read_only(&state_db)?;
    let schema_version = check_schema(&conn, &state_db)?;
    let (header, events, counts) = import(&conn, &manifest, schema_version)?;
    if args.dry_run {
        println!("{}", serde_json::to_string_pretty(&counts)?);
        return Ok(());
    }
    corpus::write(&out, &header, &events)
        .with_context(|| format!("writing the corpus to {}", out.display()))?;
    println!("{}", serde_json::to_string_pretty(&counts)?);
    eprintln!("wrote {}", out.display());
    Ok(())
}

/// Opens the copy read-only, never following a symlink, so the import can't
/// write to it and can't be pointed at anything else.
fn open_read_only(path: &Path) -> anyhow::Result<Connection> {
    Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_NOFOLLOW,
    )
    .with_context(|| format!("opening {} read-only", path.display()))
}

/// The schema the importer was written against: every column it reads must be
/// there with the type it was declared with, and the schema version must be one
/// it knows. Everything wrong is listed at once.
fn check_schema(conn: &Connection, path: &Path) -> anyhow::Result<i64> {
    let mut problems = Vec::new();
    for (table, columns) in REQUIRED_COLUMNS {
        let present: Vec<(String, String)> =
            match conn.prepare(&format!("PRAGMA table_info({table})")) {
                Ok(mut statement) => statement
                    .query_map([], |row| Ok((row.get(1)?, row.get(2)?)))?
                    .collect::<Result<_, _>>()?,
                Err(error) => {
                    problems.push(format!("the table {table} can't be read: {error}"));
                    continue;
                }
            };
        if present.is_empty() {
            problems.push(format!("the table {table} is missing"));
            continue;
        }
        for (column, declared) in columns {
            match present.iter().find(|(name, _)| name == column) {
                None => problems.push(format!("the column {table}.{column} is missing")),
                Some((_, found)) if !found.eq_ignore_ascii_case(declared) => {
                    problems.push(format!(
                        "the column {table}.{column} is declared {found:?}; the importer was written against {declared}"
                    ));
                }
                Some(_) => {}
            }
        }
    }
    let version: Option<i64> = if problems.iter().any(|p| p.contains("schema_version")) {
        None
    } else {
        conn.query_row("SELECT MAX(version) FROM schema_version", [], |row| {
            row.get(0)
        })
        .optional()?
        .flatten()
    };
    let version = match version {
        Some(version) if HERMES_SCHEMA_VERSIONS.contains(&version) => Some(version),
        Some(version) => {
            problems.push(format!(
                "the schema version is {version}; the importer was written against {}",
                HERMES_SCHEMA_VERSIONS
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            None
        }
        None => {
            problems.push("the schema_version table has no version".into());
            None
        }
    };
    if !problems.is_empty() {
        bail!(
            "{} isn't the Hermes state.db the importer was written against:\n{}",
            path.display(),
            problems.join("\n")
        );
    }
    Ok(version.expect("a version without problems"))
}

struct SessionRow {
    id: String,
    source: String,
    parent: Option<String>,
}

struct MessageRow {
    role: String,
    content: Option<String>,
    at: f64,
    summary: bool,
    tool_calls: Option<String>,
}

/// A turn being assembled: the user row and the final assistant reply so
/// far.
struct Open {
    user: MessageRow,
    assistant: Option<MessageRow>,
}

/// An event with the order it was found in, so equal times sort stably
/// and the output is the same on every import.
struct Found {
    at: Timestamp,
    seq: u64,
    event: Event,
}

fn import(
    conn: &Connection,
    manifest: &Manifest,
    schema_version: i64,
) -> anyhow::Result<(Header, Vec<Event>, Counts)> {
    let prefix = Regex::new(r"^\[([^\]\n]+)\] ").expect("the prefix regex parses");
    let block = manifest.memory_block();
    let mut counts = Counts::default();
    let mut found: Vec<Found> = Vec::new();
    let mut seq = 0u64;

    let sessions: Vec<SessionRow> = conn
        .prepare("SELECT id, source, parent_session_id FROM sessions ORDER BY started_at, id")?
        .query_map([], |row| {
            Ok(SessionRow {
                id: row.get(0)?,
                source: row.get(1)?,
                parent: row.get(2)?,
            })
        })?
        .collect::<Result<_, _>>()?;

    let mut skipped = conn.prepare(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?1 AND active = 0 AND compacted = 0",
    )?;
    let mut statement = conn.prepare(
        "SELECT role, content, timestamp, _compressed_summary, tool_calls
         FROM messages WHERE session_id = ?1 AND (active = 1 OR compacted = 1)
         ORDER BY timestamp, id",
    )?;
    for session in &sessions {
        if session.parent.is_some() {
            counts.subagent_sessions_skipped += 1;
            continue;
        }
        let class = if session.source == CRON_SOURCE {
            counts.cron_sessions += 1;
            SessionClass::Cron
        } else {
            counts.primary_sessions += 1;
            SessionClass::Primary
        };
        let inactive: i64 = skipped.query_row([&session.id], |row| row.get(0))?;
        counts.inactive_rows_skipped += u64::try_from(inactive).unwrap_or_default();
        let rows: Vec<MessageRow> = statement
            .query_map([&session.id], |row| {
                Ok(MessageRow {
                    role: row.get(0)?,
                    content: row.get(1)?,
                    at: row.get(2)?,
                    summary: row.get::<_, i64>(3)? != 0,
                    tool_calls: row.get(4)?,
                })
            })?
            .collect::<Result<_, _>>()?;

        let mut open: Option<Open> = None;
        let mut previous_query: Option<String> = None;
        let emit = |open: Option<Open>,
                    counts: &mut Counts,
                    found: &mut Vec<Found>,
                    previous_query: &mut Option<String>,
                    seq: &mut u64|
         -> anyhow::Result<()> {
            let Some(turn) = open else {
                return Ok(());
            };
            let (user_text, author) = user_text(
                turn.user.content.as_deref().unwrap_or(""),
                manifest,
                &block,
                &prefix,
                counts,
            )?;
            let at = epoch(turn.user.at)?;
            let platform = Some(session.source.clone());
            *seq += 1;
            found.push(Found {
                at,
                seq: *seq,
                event: Event::Prefetch {
                    at,
                    session: session.id.clone(),
                    class,
                    query: user_text.clone(),
                    previous_query: previous_query.take(),
                    platform: platform.clone(),
                },
            });
            *previous_query = Some(user_text.clone());
            match (&turn.assistant, class) {
                (Some(assistant), SessionClass::Primary) => {
                    let (assistant_text, _) =
                        plain_text(assistant.content.as_deref().unwrap_or(""), &block, counts)?;
                    let reply_at = epoch(assistant.at)?.max(at);
                    if author.is_some() {
                        counts.non_owner_turns += 1;
                    }
                    counts.turns += 1;
                    *seq += 1;
                    found.push(Found {
                        at: reply_at,
                        seq: *seq,
                        event: Event::Sync {
                            at: reply_at,
                            session: session.id.clone(),
                            message_at: at,
                            user: user_text,
                            assistant: assistant_text,
                            author,
                            platform,
                        },
                    });
                }
                _ => counts.prefetch_only_turns += 1,
            }
            Ok(())
        };

        for row in rows {
            if row.summary {
                // Hermes replaced the turns before here with this summary,
                // and the session's context started over.
                emit(
                    open.take(),
                    &mut counts,
                    &mut found,
                    &mut previous_query,
                    &mut seq,
                )?;
                let at = epoch(row.at)?;
                counts.compactions += 1;
                seq += 1;
                found.push(Found {
                    at,
                    seq,
                    event: Event::Clear {
                        at,
                        session: session.id.clone(),
                    },
                });
                counts.summary_rows_skipped += 1;
                continue;
            }
            match row.role.as_str() {
                "user" => {
                    emit(
                        open.take(),
                        &mut counts,
                        &mut found,
                        &mut previous_query,
                        &mut seq,
                    )?;
                    open = Some(Open {
                        user: row,
                        assistant: None,
                    });
                }
                "assistant" => {
                    if has_tool_calls(row.tool_calls.as_deref()) {
                        counts.tool_rows_skipped += 1;
                        continue;
                    }
                    if let Some(open) = &mut open {
                        open.assistant = Some(row);
                    }
                }
                _ => counts.tool_rows_skipped += 1,
            }
        }
        emit(
            open.take(),
            &mut counts,
            &mut found,
            &mut previous_query,
            &mut seq,
        )?;
    }

    found.sort_by_key(|found| (found.at, found.seq));
    let events = found.into_iter().map(|found| found.event).collect();
    let header = Header {
        corpus: corpus::VERSION,
        bank: manifest.bank.clone(),
        timezone: manifest.timezone.clone(),
        owner: Owner {
            name: manifest.owner.name.clone(),
            platform_ids: manifest.owner.platform_ids.clone(),
        },
        assistant: manifest.assistant.clone(),
        models: manifest.models.clone(),
        hermes_schema_version: schema_version,
        counts: counts.clone(),
    };
    Ok((header, events, counts))
}

/// Whether an assistant row only calls tools: `tool_calls` is a non-empty
/// JSON array.
fn has_tool_calls(tool_calls: Option<&str>) -> bool {
    match tool_calls {
        None => false,
        Some(text) => match serde_json::from_str::<Value>(text) {
            Ok(Value::Array(calls)) => !calls.is_empty(),
            Ok(Value::Null) => false,
            Ok(_) => true,
            Err(_) => !text.trim().is_empty(),
        },
    }
}

/// The user message as `sync_turn` would send it: multimodal content reduced to
/// its text with the memory block cut out, backfill stripped, and the speaker
/// picked from the `[Name] ` prefix, which stays in the text.
fn user_text(
    content: &str,
    manifest: &Manifest,
    block: &MemoryBlock,
    prefix: &Regex,
    counts: &mut Counts,
) -> anyhow::Result<(String, Option<Author>)> {
    let (text, _) = plain_text(content, block, counts)?;
    let text = match text.rfind(NEW_MESSAGE_MARKER) {
        Some(index) => {
            counts.backfills_stripped += 1;
            text[index + NEW_MESSAGE_MARKER.len()..]
                .trim_start()
                .to_string()
        }
        None => text,
    };
    let author = prefix
        .captures(&text)
        .and_then(|captures| manifest.speaker(captures.get(1)?.as_str()))
        .map(|speaker| Author {
            id: speaker.id.clone(),
            name: Some(speaker.name.clone()),
        });
    Ok((text, author))
}

/// A row's text: plain content as it is, multimodal content reduced to
/// its text parts, in both cases with the memory block cut out. Says
/// whether it was multimodal.
fn plain_text(
    content: &str,
    block: &MemoryBlock,
    counts: &mut Counts,
) -> anyhow::Result<(String, bool)> {
    let (text, multimodal) = match content.strip_prefix(MULTIMODAL_PREFIX) {
        Some(json) => {
            counts.multimodal_rows += 1;
            // serde's message can quote the row, so only `trace` sees it.
            let parts: Value = serde_json::from_str(json).map_err(|error| {
                tracing::trace!(%error, "a multimodal row's parts aren't JSON");
                anyhow::anyhow!("a multimodal row's parts aren't JSON")
            })?;
            let Value::Array(parts) = parts else {
                bail!("a multimodal row's parts aren't a list");
            };
            let mut texts = Vec::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            texts.push(text.to_string());
                        }
                    }
                    _ => counts.image_parts_dropped += 1,
                }
            }
            (texts.join("\n"), true)
        }
        None => (content.to_string(), false),
    };
    Ok((strip_memory_block(&text, block, counts), multimodal))
}

/// Cuts every fenced memory block out of `text`. A start with no end cuts
/// to the end of the text, since injected memory must never become source
/// text (ADR 0002).
fn strip_memory_block(text: &str, block: &MemoryBlock, counts: &mut Counts) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(&block.start) {
        counts.memory_blocks_stripped += 1;
        out.push_str(&rest[..start]);
        let after = &rest[start + block.start.len()..];
        match after.find(&block.end) {
            Some(end) => rest = &after[end + block.end.len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// Hermes' epoch-seconds float as an instant, to the microsecond.
fn epoch(seconds: f64) -> anyhow::Result<Timestamp> {
    if !seconds.is_finite() {
        bail!("a message timestamp isn't a number");
    }
    let micros = (seconds * 1_000_000.0).round() as i64;
    Timestamp::from_microsecond(micros).context("a message timestamp is out of range")
}
