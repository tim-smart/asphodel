//! Assembling call 1's input for a leased chunk (TIM-92, "Inputs" and
//! "Entity resolution").

use std::collections::{BTreeMap, BTreeSet};

use jiff::Timestamp;
use jiff::ToSpan;
use jiff::civil::Date;
use jiff::tz::TimeZone;
use rusqlite::{Connection, OptionalExtension};
use uuid::Uuid;

use super::{
    CALENDAR_DAYS, CANDIDATE_MEMORIES, CONTEXT_CHARS, CONTEXT_TURNS, Call1Input, Candidate,
    ENTITY_CANDIDATE_CAP, EntityKind, ExtractError, InContextMemory, PREVIOUS_CHUNK_CHARS,
    SpeakerRef,
};
use crate::config::Tuning;
use crate::constants::{SIGNIFICANCE_KEPT, Significance};
use crate::ingest::TURN_SEPARATOR;
use crate::queue::{Lease, SourceKind};
use crate::store::timestamp;
use crate::strength::{Access, AccessKind, BankTime, strength};

/// What the checks and the commit need beyond the input itself.
pub(super) struct Unit {
    pub bank_id: i64,
    pub chunk_id: i64,
    pub source_id: i64,
    pub tz: TimeZone,
    pub ingested_at: Timestamp,
    /// The turn number accesses from this chunk carry: for a turn, the
    /// bank's counter just after it was counted; for a document, the
    /// counter when it was ingested.
    pub turn: i64,
    /// Whether the speaker is the owner. False for a document.
    pub owner_speaking: bool,
    /// Candidate handle to entity rowid.
    pub candidates: BTreeMap<String, i64>,
    /// In-context handle to memory rowid and public id.
    pub in_context: BTreeMap<String, (i64, Uuid)>,
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
) -> Result<(Call1Input, Unit), ExtractError> {
    let bank_id = lease.bank_id();
    let chunk_id = lease.chunk_id();
    let source = conn.query_row(
        "SELECT c.uuid, c.text, c.start_offset, s.id, s.kind, s.session_id, s.message_at,
                s.observed_at, s.reference_date, s.reference_date_exact, s.timezone, s.author_id,
                s.platform, s.text, s.ingested_at
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

    let mut bank_time: Option<BankTime> = None;
    let mut candidates = Vec::new();
    let mut handles = BTreeMap::new();
    for (index, entity_id) in always.iter().chain(found.iter()).enumerate() {
        let handle = format!("e{}", index + 1);
        let candidate = candidate(
            conn,
            tuning,
            now,
            bank_id,
            *entity_id,
            &handle,
            &mut bank_time,
        )?;
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

    let turn = turn_number(conn, bank_id, source.source_id, is_turn)?;
    let unit = Unit {
        bank_id,
        chunk_id,
        source_id: source.source_id,
        tz,
        ingested_at: source.ingested_at,
        turn,
        owner_speaking: speaker_ref.as_ref().is_some_and(|speaker| speaker.owner),
        candidates: handles,
        in_context: in_context_handles,
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

/// Lowercase runs of letters and digits, the way the alias FTS tokenises
/// closely enough to confirm its matches.
pub(super) fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Entities other than `always` whose aliases appear in a passage, as their
/// surviving entities, ranked by how many memories link to them and capped
/// at [`ENTITY_CANDIDATE_CAP`].
fn found_entities(
    conn: &Connection,
    bank_id: i64,
    passages: &[&str],
    always: &[i64],
) -> Result<Vec<i64>, rusqlite::Error> {
    let passage_words: Vec<Vec<String>> = passages.iter().map(|p| words(p)).collect();
    let terms: BTreeSet<&str> = passage_words.iter().flatten().map(String::as_str).collect();
    if terms.is_empty() {
        return Ok(Vec::new());
    }
    // Every term is letters and digits only, so quoting is enough.
    let query = terms
        .iter()
        .map(|term| format!("\"{term}\""))
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

    let mut found = BTreeSet::new();
    for (entity_id, alias) in hits {
        let alias_words = words(&alias);
        let named = !alias_words.is_empty()
            && passage_words.iter().any(|passage| {
                passage
                    .windows(alias_words.len())
                    .any(|window| window == alias_words.as_slice())
            });
        if named {
            let entity_id = survivor(conn, entity_id)?;
            if !always.contains(&entity_id) {
                found.insert(entity_id);
            }
        }
    }

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

fn candidate(
    conn: &Connection,
    tuning: &Tuning,
    now: Timestamp,
    bank_id: i64,
    entity_id: i64,
    handle: &str,
    bank_time: &mut Option<BankTime>,
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
        memories: strongest_memories(conn, tuning, now, bank_id, entity_id, bank_time)?,
    })
}

/// The sentences of up to [`CANDIDATE_MEMORIES`] memories linked to the
/// entity, strongest now first, leaving out hidden and retracted ones.
///
/// Strength here is a memory's own accesses and significance on the bank's
/// clock. It doesn't follow supersession or restart at a window's close,
/// which only matters for ranking three example sentences.
fn strongest_memories(
    conn: &Connection,
    tuning: &Tuning,
    now: Timestamp,
    bank_id: i64,
    entity_id: i64,
    bank_time: &mut Option<BankTime>,
) -> Result<Vec<String>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT m.id, m.content, m.significance, m.owner_significance
         FROM memory_entities me JOIN memories m ON m.id = me.memory_id
         WHERE me.entity_id = ?1 AND m.hidden_at IS NULL AND m.invalidated_at IS NULL
         ORDER BY m.id",
    )?;
    let linked: Vec<(i64, String, String, Option<String>)> = statement
        .query_map([entity_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?
        .collect::<Result<_, _>>()?;
    if linked.is_empty() {
        return Ok(Vec::new());
    }
    if bank_time.is_none() {
        let mut turns = conn.prepare_cached(
            "SELECT message_at FROM sources WHERE bank_id = ?1 AND kind = 'turn'",
        )?;
        let at: Vec<Timestamp> = turns
            .query_map([bank_id], |row| row.get::<_, i64>(0))?
            .map(|micros| micros.map(timestamp))
            .collect::<Result<_, _>>()?;
        *bank_time = Some(BankTime::new(&at, tuning.clock.quiet_rate));
    }
    let bank_time = bank_time.as_ref().expect("built above");

    let mut accesses = conn.prepare_cached("SELECT kind, at FROM accesses WHERE memory_id = ?1")?;
    let mut scored = Vec::with_capacity(linked.len());
    for (memory_id, content, level, owner) in linked {
        let log: Vec<Access> = accesses
            .query_map([memory_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?
            .filter_map(|row| {
                let (kind, at) = row.ok()?;
                Some(Access {
                    kind: access_kind(&kind)?,
                    at: timestamp(at),
                })
            })
            .collect();
        let significance = significance_value(owner.as_deref().unwrap_or(&level));
        let value = strength(significance, &log, None, bank_time, now).value;
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

fn access_kind(kind: &str) -> Option<AccessKind> {
    match kind {
        "created" => Some(AccessKind::Created),
        "used" => Some(AccessKind::Used),
        "mentioned_again" => Some(AccessKind::MentionedAgain),
        "confirmed" => Some(AccessKind::Confirmed),
        _ => None,
    }
}

/// A stored significance level, or `kept`, as its value.
fn significance_value(level: &str) -> f64 {
    match level {
        "kept" => SIGNIFICANCE_KEPT,
        "trivial" => Significance::Trivial.value(),
        "minor" => Significance::Minor.value(),
        "notable" => Significance::Notable.value(),
        "major" => Significance::Major.value(),
        _ => Significance::Critical.value(),
    }
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
