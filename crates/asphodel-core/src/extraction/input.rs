//! Assembling call 1's input for a leased chunk (TIM-92, "Inputs" and
//! "Entity resolution").

use std::collections::{BTreeMap, BTreeSet};

use jiff::Timestamp;
use jiff::ToSpan;
use jiff::civil::Date;
use jiff::tz::TimeZone;
use rusqlite::{Connection, OptionalExtension};
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

use super::{
    CALENDAR_DAYS, CANDIDATE_MEMORIES, CONTEXT_CHARS, CONTEXT_TURNS, Call1Input, Candidate,
    ENTITY_CANDIDATE_CAP, EntityKind, ExtractError, InContextEntry, InContextMemory,
    PREVIOUS_CHUNK_CHARS, SpeakerRef,
};
use crate::config::Tuning;
use crate::ingest::TURN_SEPARATOR;
use crate::queue::{Lease, SourceKind};
use crate::store::strength::StrengthLoader;
use crate::store::timestamp;
use crate::system_prompt::BlockEntry;

/// What the checks and the commit need beyond the input itself.
pub(super) struct Unit {
    pub bank_id: i64,
    pub chunk_id: i64,
    pub source_id: i64,
    pub tz: TimeZone,
    pub ingested_at: Timestamp,
    /// The document's id, for a document chunk: a neighbour from an earlier
    /// version of the same document isn't mentioned again (TIM-92).
    pub document_id: Option<String>,
    /// The turn number accesses from this chunk carry: for a turn, the
    /// bank's counter just after it was counted; for a document, the
    /// counter when it was ingested.
    pub turn: i64,
    /// The highest entity rowid when the input was assembled. Rowids are
    /// never reused, so an entity above it was created after call 1's input
    /// was read (TIM-92: commit reuses only those).
    pub entity_boundary: i64,
    /// Whether the speaker is the owner. False for a document.
    pub owner_speaking: bool,
    /// Candidate handle to entity rowid.
    pub candidates: BTreeMap<String, i64>,
    /// In-context handle to memory rowid and public id.
    pub in_context: BTreeMap<String, (i64, Uuid)>,
    /// Entry handle to the memories the entry cites, rowid and public id.
    pub entries: BTreeMap<String, Vec<(i64, Uuid)>>,
}

impl Unit {
    /// The entities call 1 was shown, which a proposed new entity never
    /// overrides (TIM-92).
    pub fn seen(&self) -> BTreeSet<i64> {
        self.candidates.values().copied().collect()
    }
}

struct Source {
    chunk: Uuid,
    text: String,
    start_offset: usize,
    source_id: i64,
    kind: SourceKind,
    session_id: Option<String>,
    document_id: Option<String>,
    message_at: Option<i64>,
    observed_at: Timestamp,
    reference_date: Option<String>,
    reference_date_exact: bool,
    timezone: String,
    author_id: Option<String>,
    platform: Option<String>,
    source_text: Option<String>,
    ingested_at: Timestamp,
}

pub(super) fn assemble(
    conn: &Connection,
    tuning: &Tuning,
    now: Timestamp,
    lease: &Lease,
    in_context: &[Uuid],
    entries: &[BlockEntry],
) -> Result<(Call1Input, Unit), ExtractError> {
    let bank_id = lease.bank_id();
    let chunk_id = lease.chunk_id();
    let source = conn.query_row(
        "SELECT c.uuid, c.text, c.start_offset, s.id, s.kind, s.session_id, s.message_at,
                s.observed_at, s.reference_date, s.reference_date_exact, s.timezone, s.author_id,
                s.platform, s.text, s.ingested_at, s.document_id
         FROM chunks c JOIN sources s ON s.id = c.source_id
         WHERE c.id = ?1",
        [chunk_id],
        |row| {
            Ok(Source {
                chunk: parse_uuid(&row.get::<_, String>(0)?),
                text: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                start_offset: usize::try_from(row.get::<_, i64>(2)?).unwrap_or(0),
                source_id: row.get(3)?,
                kind: if row.get::<_, String>(4)? == "turn" {
                    SourceKind::Turn
                } else {
                    SourceKind::Document
                },
                session_id: row.get(5)?,
                message_at: row.get(6)?,
                observed_at: timestamp(row.get(7)?),
                reference_date: row.get(8)?,
                reference_date_exact: row.get(9)?,
                timezone: row.get(10)?,
                author_id: row.get(11)?,
                platform: row.get(12)?,
                source_text: row.get(13)?,
                ingested_at: timestamp(row.get(14)?),
                document_id: row.get(15)?,
            })
        },
    )?;
    let tz = TimeZone::get(&source.timezone).unwrap_or(TimeZone::UTC);
    let is_turn = source.kind == SourceKind::Turn;

    let reference_date = if is_turn {
        Some(source.observed_at.to_zoned(tz.clone()).date())
    } else if source.reference_date_exact {
        source
            .reference_date
            .as_deref()
            .and_then(|date| date.parse::<Date>().ok())
    } else {
        None
    };
    let calendar = reference_date.map(calendar).unwrap_or_default();

    let reply_start = is_turn.then(|| {
        source.source_text.as_deref().unwrap_or("").chars().count() + TURN_SEPARATOR.chars().count()
    });

    let context = if is_turn {
        earlier_turns(conn, bank_id, &source)?
    } else {
        text_before(&source)
    };

    let user = seeded(conn, bank_id, "user")?;
    let assistant = seeded(conn, bank_id, "assistant")?;
    let speaker = if is_turn {
        speaker_entity(conn, bank_id, user, &source)?
    } else {
        None
    };

    let mut always = vec![user, assistant];
    if let Some(speaker) = speaker
        && !always.contains(&speaker)
    {
        always.push(speaker);
    }
    let mut passages: Vec<&str> = vec![source.text.as_str()];
    passages.extend(context.iter().map(String::as_str));
    let found = found_entities(conn, bank_id, &passages, &always)?;

    let mut loader: Option<StrengthLoader> = None;
    let mut candidates = Vec::new();
    let mut handles = BTreeMap::new();
    for (index, entity_id) in always.iter().chain(found.iter()).enumerate() {
        let handle = format!("e{}", index + 1);
        let candidate = candidate(conn, tuning, now, bank_id, *entity_id, &handle, &mut loader)?;
        handles.insert(handle, *entity_id);
        candidates.push(candidate);
    }

    let speaker_ref = match speaker {
        Some(entity_id) => {
            let index = always
                .iter()
                .position(|id| *id == entity_id)
                .expect("the speaker is a candidate");
            let candidate = &candidates[index];
            Some(SpeakerRef {
                handle: candidate.handle.clone(),
                entity: candidate.entity,
                name: candidate.name.clone(),
                owner: entity_id == user,
            })
        }
        None => None,
    };

    let mut in_context_memories = Vec::new();
    let mut in_context_handles = BTreeMap::new();
    if is_turn {
        let mut seen = BTreeSet::new();
        for memory in in_context {
            if !seen.insert(*memory) {
                continue;
            }
            let found: Option<(i64, String)> = conn
                .query_row(
                    "SELECT id, content FROM memories
                     WHERE uuid = ?1 AND bank_id = ?2 AND hidden_at IS NULL",
                    (memory.to_string(), bank_id),
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?;
            if let Some((memory_id, content)) = found {
                let handle = format!("m{}", in_context_memories.len() + 1);
                in_context_handles.insert(handle.clone(), (memory_id, *memory));
                in_context_memories.push(InContextMemory {
                    handle,
                    memory: *memory,
                    content,
                });
            }
        }
    }

    // An entry is shown only when every memory it cites is: one citing a
    // memory forgotten since can't be credited (TIM-95, decision 6).
    let mut in_context_entries = Vec::new();
    let mut entry_handles = BTreeMap::new();
    if is_turn {
        let by_memory: BTreeMap<Uuid, (&String, i64)> = in_context_handles
            .iter()
            .map(|(handle, (id, memory))| (*memory, (handle, *id)))
            .collect();
        for entry in entries {
            let cited: Option<Vec<(&String, i64, Uuid)>> = entry
                .cites
                .iter()
                .map(|memory| {
                    by_memory
                        .get(memory)
                        .map(|(handle, id)| (*handle, *id, *memory))
                })
                .collect();
            let Some(cited) = cited.filter(|cited| !cited.is_empty()) else {
                continue;
            };
            let handle = format!("n{}", in_context_entries.len() + 1);
            entry_handles.insert(
                handle.clone(),
                cited.iter().map(|(_, id, memory)| (*id, *memory)).collect(),
            );
            in_context_entries.push(InContextEntry {
                handle,
                entry: entry.entry,
                text: entry.text.clone(),
                cites: cited
                    .iter()
                    .map(|(handle, _, _)| (*handle).clone())
                    .collect(),
            });
        }
    }

    let turn = turn_number(conn, bank_id, source.source_id, is_turn)?;
    let entity_boundary: i64 =
        conn.query_row("SELECT COALESCE(MAX(id), 0) FROM entities", [], |row| {
            row.get(0)
        })?;
    let unit = Unit {
        bank_id,
        chunk_id,
        source_id: source.source_id,
        tz,
        ingested_at: source.ingested_at,
        document_id: source.document_id.clone(),
        turn,
        entity_boundary,
        owner_speaking: speaker_ref.as_ref().is_some_and(|speaker| speaker.owner),
        candidates: handles,
        in_context: in_context_handles,
        entries: entry_handles,
    };
    let input = Call1Input {
        chunk: source.chunk,
        source_kind: source.kind,
        text: source.text,
        reply_start,
        observed_at: source.observed_at,
        timezone: source.timezone,
        reference_date,
        calendar,
        speaker: speaker_ref,
        context,
        candidates,
        in_context: in_context_memories,
        entries: in_context_entries,
    };
    Ok((input, unit))
}

fn calendar(reference: Date) -> Vec<Date> {
    (-CALENDAR_DAYS..=CALENDAR_DAYS)
        .filter_map(|offset| reference.checked_add(offset.days()).ok())
        .collect()
}

/// Up to [`CONTEXT_TURNS`] earlier turns of the session that still have
/// their text, oldest first, clipped to [`CONTEXT_CHARS`].
fn earlier_turns(
    conn: &Connection,
    bank_id: i64,
    source: &Source,
) -> Result<Vec<String>, rusqlite::Error> {
    let (Some(session_id), Some(message_at)) = (&source.session_id, source.message_at) else {
        return Ok(Vec::new());
    };
    let mut statement = conn.prepare_cached(
        "SELECT text, reply FROM sources
         WHERE bank_id = ?1 AND kind = 'turn' AND session_id = ?2 AND message_at < ?3
           AND text IS NOT NULL
         ORDER BY message_at DESC, ingested_at DESC, id DESC
         LIMIT ?4",
    )?;
    let mut turns: Vec<String> = statement
        .query_map(
            (bank_id, session_id, message_at, CONTEXT_TURNS as i64),
            |row| {
                let text: String = row.get(0)?;
                let reply: Option<String> = row.get(1)?;
                Ok([
                    text.as_str(),
                    TURN_SEPARATOR,
                    reply.as_deref().unwrap_or(""),
                ]
                .concat())
            },
        )?
        .collect::<Result<_, _>>()?;
    turns.reverse();
    Ok(clip(turns, CONTEXT_CHARS))
}

/// Takes characters off the start of the oldest passages until the total is
/// at most `cap`, leaving out any passage clipped to nothing.
fn clip(passages: Vec<String>, cap: usize) -> Vec<String> {
    let total: usize = passages.iter().map(|p| p.chars().count()).sum();
    let mut excess = total.saturating_sub(cap);
    passages
        .into_iter()
        .filter_map(|passage| {
            if excess == 0 {
                return Some(passage);
            }
            let length = passage.chars().count();
            let cut = excess.min(length);
            excess -= cut;
            let kept: String = passage.chars().skip(cut).collect();
            (!kept.is_empty()).then_some(kept)
        })
        .collect()
}

/// The [`PREVIOUS_CHUNK_CHARS`] of the document before the chunk.
fn text_before(source: &Source) -> Vec<String> {
    let Some(text) = &source.source_text else {
        return Vec::new();
    };
    let start = source.start_offset;
    let from = start.saturating_sub(PREVIOUS_CHUNK_CHARS);
    let before: String = text.chars().skip(from).take(start - from).collect();
    if before.is_empty() {
        Vec::new()
    } else {
        vec![before]
    }
}

fn seeded(conn: &Connection, bank_id: i64, which: &str) -> Result<i64, rusqlite::Error> {
    conn.query_row(
        "SELECT id FROM entities WHERE bank_id = ?1 AND seeded = ?2",
        (bank_id, which),
        |row| row.get(0),
    )
}

/// The turn's speaker, resolved as ingest resolved it (TIM-94, decision 1):
/// no author is the owner, anyone else through `speaker_ids`.
fn speaker_entity(
    conn: &Connection,
    bank_id: i64,
    user: i64,
    source: &Source,
) -> Result<Option<i64>, rusqlite::Error> {
    let Some(author_id) = &source.author_id else {
        return Ok(Some(user));
    };
    let platform_id = match &source.platform {
        Some(platform) => format!("{platform}:{author_id}"),
        None => author_id.clone(),
    };
    let found: Option<i64> = conn
        .query_row(
            "SELECT entity_id FROM speaker_ids WHERE bank_id = ?1 AND platform_id = ?2",
            (bank_id, platform_id.trim()),
            |row| row.get(0),
        )
        .optional()?;
    match found {
        Some(entity_id) => Ok(Some(survivor(conn, entity_id)?)),
        None => {
            tracing::warn!("a turn's speaker has no speaker id; extracting without a speaker");
            Ok(None)
        }
    }
}

/// Follows `merged_into` to the entity that survived any merges. Bounded, so
/// a corrupt cycle can't hang extraction.
pub(super) fn survivor(conn: &Connection, mut entity_id: i64) -> Result<i64, rusqlite::Error> {
    for _ in 0..64 {
        let merged_into: Option<i64> = conn.query_row(
            "SELECT merged_into FROM entities WHERE id = ?1",
            [entity_id],
            |row| row.get(0),
        )?;
        match merged_into {
            Some(next) if next != entity_id => entity_id = next,
            _ => break,
        }
    }
    Ok(entity_id)
}

/// The alias FTS's tokenizer, as `entity_aliases_fts` declares it in the
/// version 1 migration. The passages are searched with the same one, so a
/// name matches exactly when SQLite would match it.
const ALIAS_TOKENIZER: &str = "unicode61 remove_diacritics 2";

/// Entities other than `always` whose aliases appear in a passage, as their
/// surviving entities, ranked by how many memories link to them and capped
/// at [`ENTITY_CANDIDATE_CAP`].
///
/// SQLite does all the tokenizing and folding, so matching is exactly the
/// alias FTS's: Latin diacritics and case are folded, everything else is
/// kept. The passages go into an in-memory FTS table of their own (NFC
/// first, so precomposed and decomposed spellings agree). Its vocabulary,
/// the terms as SQLite folded them, is the query that finds aliases sharing
/// any term. Each alias found then has to match a passage as a whole phrase,
/// in that same table, so a term shared with one word of an alias never
/// matches the rest of it loosely.
fn found_entities(
    conn: &Connection,
    bank_id: i64,
    passages: &[&str],
    always: &[i64],
) -> Result<Vec<i64>, rusqlite::Error> {
    let found = entities_named(conn, bank_id, passages, always)?;

    let mut ranked = Vec::with_capacity(found.len());
    for entity_id in found {
        let links: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memory_entities WHERE entity_id = ?1",
            [entity_id],
            |row| row.get(0),
        )?;
        ranked.push((links, entity_id));
    }
    ranked.sort_by(|(left_links, left_id), (right_links, right_id)| {
        right_links.cmp(left_links).then(left_id.cmp(right_id))
    });
    Ok(ranked
        .into_iter()
        .take(ENTITY_CANDIDATE_CAP)
        .map(|(_, entity_id)| entity_id)
        .collect())
}

/// The entities of `bank_id` with an alias that appears whole in any of
/// `passages`, matched with the alias FTS's own tokenizer, each resolved to
/// the entity it was merged into and leaving out `exclude`. Retrieval's
/// entity arm and the recall tool's `entity` parameter use this too
/// ("Retrieval and ranking", TIM-93, decision 1).
pub(crate) fn entities_named(
    conn: &Connection,
    bank_id: i64,
    passages: &[&str],
    exclude: &[i64],
) -> Result<BTreeSet<i64>, rusqlite::Error> {
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS temp.extraction_passages
           USING fts5(text, tokenize = '{ALIAS_TOKENIZER}');
         CREATE VIRTUAL TABLE IF NOT EXISTS temp.extraction_passage_terms
           USING fts5vocab('temp', 'extraction_passages', 'row');
         DELETE FROM temp.extraction_passages;"
    ))?;
    let found = named_entities(conn, bank_id, passages, exclude);
    // The passages are memory content: don't leave them behind, even after
    // an error. The temp store is in memory (`temp_store = MEMORY`).
    conn.execute("DELETE FROM temp.extraction_passages", [])?;
    found
}

fn named_entities(
    conn: &Connection,
    bank_id: i64,
    passages: &[&str],
    always: &[i64],
) -> Result<BTreeSet<i64>, rusqlite::Error> {
    let mut insert =
        conn.prepare_cached("INSERT INTO temp.extraction_passages (text) VALUES (?1)")?;
    for passage in passages {
        insert.execute([passage.nfc().collect::<String>()])?;
    }
    let mut terms = conn.prepare_cached("SELECT term FROM temp.extraction_passage_terms")?;
    let terms: Vec<String> = terms
        .query_map([], |row| row.get(0))?
        .collect::<Result<_, _>>()?;
    if terms.is_empty() {
        return Ok(BTreeSet::new());
    }
    let query = terms
        .iter()
        .map(|term| phrase(term))
        .collect::<Vec<_>>()
        .join(" OR ");
    let mut statement = conn.prepare_cached(
        "SELECT a.entity_id, a.alias FROM entity_aliases_fts f
         JOIN entity_aliases a ON a.id = f.rowid
         WHERE entity_aliases_fts MATCH ?1 AND a.bank_id = ?2",
    )?;
    let hits: Vec<(i64, String)> = statement
        .query_map((query, bank_id), |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<Result<_, _>>()?;

    let mut whole = conn.prepare_cached(
        "SELECT 1 FROM temp.extraction_passages WHERE extraction_passages MATCH ?1 LIMIT 1",
    )?;
    let mut found = BTreeSet::new();
    for (entity_id, alias) in hits {
        // An alias with no letters or digits has no terms, and an empty
        // phrase is a syntax error, so it can't be named.
        if !alias.chars().any(char::is_alphanumeric) {
            continue;
        }
        let alias: String = alias.nfc().collect();
        if whole.exists([phrase(&alias)])? {
            let entity_id = survivor(conn, entity_id)?;
            if !always.contains(&entity_id) {
                found.insert(entity_id);
            }
        }
    }
    Ok(found)
}

/// `text` as one FTS5 phrase: in double quotes, with any double quote
/// doubled, so nothing in it is read as query syntax.
pub(crate) fn phrase(text: &str) -> String {
    format!("\"{}\"", text.replace('"', "\"\""))
}

fn candidate(
    conn: &Connection,
    tuning: &Tuning,
    now: Timestamp,
    bank_id: i64,
    entity_id: i64,
    handle: &str,
    loader: &mut Option<StrengthLoader>,
) -> Result<Candidate, rusqlite::Error> {
    let (uuid, name, kind): (String, String, String) = conn.query_row(
        "SELECT uuid, name, kind FROM entities WHERE id = ?1",
        [entity_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    let mut statement =
        conn.prepare_cached("SELECT alias FROM entity_aliases WHERE entity_id = ?1 ORDER BY id")?;
    let aliases: Vec<String> = statement
        .query_map([entity_id], |row| row.get(0))?
        .collect::<Result<_, _>>()?;
    Ok(Candidate {
        handle: handle.to_owned(),
        entity: parse_uuid(&uuid),
        name,
        kind: EntityKind::parse(&kind).unwrap_or(EntityKind::Thing),
        aliases,
        memories: strongest_memories(conn, tuning, now, bank_id, entity_id, loader)?,
    })
}

/// The sentences of up to [`CANDIDATE_MEMORIES`] memories linked to the
/// entity, strongest now first, leaving out hidden and retracted ones.
/// Strength is the full TIM-91 strength, inherited accesses and window
/// closes included ([`StrengthLoader`]); ties go to the older memory.
fn strongest_memories(
    conn: &Connection,
    tuning: &Tuning,
    now: Timestamp,
    bank_id: i64,
    entity_id: i64,
    loader: &mut Option<StrengthLoader>,
) -> Result<Vec<String>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT m.id, m.content
         FROM memory_entities me JOIN memories m ON m.id = me.memory_id
         WHERE me.entity_id = ?1 AND m.hidden_at IS NULL AND m.invalidated_at IS NULL
         ORDER BY m.id",
    )?;
    let linked: Vec<(i64, String)> = statement
        .query_map([entity_id], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<Result<_, _>>()?;
    if linked.is_empty() {
        return Ok(Vec::new());
    }
    let loader = match loader {
        Some(loader) => loader,
        None => loader.insert(StrengthLoader::new(
            conn,
            bank_id,
            tuning.clock.quiet_rate,
            now,
        )?),
    };

    let mut scored = Vec::with_capacity(linked.len());
    for (memory_id, content) in linked {
        let value = loader.strength(conn, memory_id)?.value;
        scored.push((value, memory_id, content));
    }
    scored.sort_by(|(left, left_id, _), (right, right_id, _)| {
        right.total_cmp(left).then(left_id.cmp(right_id))
    });
    Ok(scored
        .into_iter()
        .take(CANDIDATE_MEMORIES)
        .map(|(_, _, content)| content)
        .collect())
}

/// The turn number for accesses from this source. Ingest counts every turn,
/// tombstones included, in rowid order, so a turn's number is how many turn
/// sources the bank has up to it, and a document's is how many it had before
/// it.
fn turn_number(
    conn: &Connection,
    bank_id: i64,
    source_id: i64,
    is_turn: bool,
) -> Result<i64, rusqlite::Error> {
    let sql = if is_turn {
        "SELECT COUNT(*) FROM sources WHERE bank_id = ?1 AND kind = 'turn' AND id <= ?2"
    } else {
        "SELECT COUNT(*) FROM sources WHERE bank_id = ?1 AND kind = 'turn' AND id < ?2"
    };
    conn.query_row(sql, (bank_id, source_id), |row| row.get(0))
}

fn parse_uuid(text: &str) -> Uuid {
    text.parse().expect("a stored uuid parses")
}
