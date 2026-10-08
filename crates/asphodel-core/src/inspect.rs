//! What `memory show`, `entity show` and `model show` print.
//!
//! These are for the operator, through the CLI; the agent gets no new tool.
//! They read only. A view holds content, since showing it is the point, so
//! nothing here is logged above `trace`.
//!
//! **Projected dates.** Strength runs on bank time, which only runs at full
//! speed for a day after each turn. So a fade or purge date is
//! given as the bank days until it, assuming the memory isn't used again,
//! and as the earliest world date it can happen: the date it would be if
//! bank time ran at full speed from now. A quieter bank reaches it later.

use std::collections::BTreeSet;

use jiff::Timestamp;
use jiff::tz::TimeZone;
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;

use crate::config::{PurgePause, Tuning};
use crate::constants::{S, TAU};
use crate::entities::{EntityError, resolve, uuid_of};
use crate::erase::REDACTION_MASK;
use crate::extraction::survivor;
use crate::ingest::find_bank;
use crate::mental_models::{Model, find_model, model};
use crate::store::strength::{StrengthLoader, window, world_time};
use crate::store::{Store, StoreError, timestamp};
use crate::strength::{
    Phase, WorldTime, chain, chain_head, never_purged, projected_below, projected_below_after,
    unit_end,
};

mod browse;

pub use browse::{
    BankOverview, ChunkCounts, ChunkState, ChunkView, DEFAULT_LIST, Fading, MAX_LIST, MemoryPage,
    MemoryQuery, MemorySort, MemoryStatus, MemorySummary, SourceCounts, SourceDetail, SourcePage,
    SourceQuery, SourceSummary, SourceVersion, StatusCounts,
};
pub(crate) use browse::{banks, memories, source, sources};

/// The most linked memories `entity show` lists, newest first.
pub const ENTITY_MEMORIES: usize = 100;

#[derive(Debug, thiserror::Error)]
pub enum InspectError {
    #[error("unknown bank")]
    UnknownBank,

    #[error("no such memory in the bank")]
    UnknownMemory,

    #[error("unknown model")]
    UnknownModel,

    #[error("no such source in the bank")]
    UnknownSource,

    #[error("{reason}")]
    InvalidQuery { reason: &'static str },

    #[error(transparent)]
    Entity(#[from] EntityError),

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for InspectError {
    fn from(error: rusqlite::Error) -> Self {
        InspectError::Store(StoreError::Sqlite(error))
    }
}

/// One row of the edit log. `details` holds ids, times and counts.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EditEntry {
    pub id: Uuid,
    pub kind: String,
    pub at: Timestamp,
    pub memory: Option<Uuid>,
    pub entity: Option<Uuid>,
    pub details: Value,
}

/// `memory show`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemoryView {
    pub id: Uuid,
    pub sentence: String,
    pub kind: String,
    pub phase: Option<Phase>,
    pub window: WindowView,
    pub observed_at: Timestamp,
    pub created_at: Timestamp,
    /// Set from the moment forget is called until the erase runs.
    pub hidden_at: Option<Timestamp>,
    /// Set when a later memory retracted it.
    pub retracted_at: Option<Timestamp>,
    pub significance: SignificanceView,
    pub source: SourceView,
    /// Its own accesses and those it inherits, oldest first.
    pub accesses: Vec<AccessEntry>,
    pub edits: Vec<EditEntry>,
    /// The claims absorbed into it as repeats, newest first. Its own, not
    /// its predecessors'.
    pub restatements: Vec<RestatementEntry>,
    pub chain: ChainView,
    pub entities: Vec<LinkedEntity>,
    pub strength: StrengthView,
    pub purge: PurgeView,
    pub projection: Projection,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WindowView {
    pub valid_from: Option<WorldTime>,
    pub valid_until: Option<WorldTime>,
    pub until_event: Option<String>,
    pub window_confidence: String,
    pub due_at: Option<WorldTime>,
    pub volatility: Option<String>,
    pub recurrence: Option<String>,
    pub recurrence_rrule: Option<String>,
    /// The first occurrence, with its recorded precision.
    pub recurrence_start: Option<WorldTime>,
    pub timezone: String,
}

/// Both significance fields: the level extraction gave, and the owner's
/// setting over it, which keep, unkeep and `memory significance` write.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SignificanceView {
    pub extracted: String,
    pub owner: Option<String>,
    /// The one strength uses, and its value.
    pub effective: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SourceView {
    pub source: Uuid,
    pub kind: String,
    pub session_id: Option<String>,
    pub document_id: Option<String>,
    pub message_at: Option<Timestamp>,
    pub chunk: Uuid,
    /// Character offsets into the chunk.
    pub start: i64,
    pub end: i64,
    /// The passage the memory was extracted from, while it's kept.
    pub passage: Option<String>,
    /// Why there's no passage.
    pub gone: Option<Gone>,
    /// The secret-scan kinds that fired on the source, redacted before it
    /// was stored.
    pub secret_kinds: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum Gone {
    /// The nightly sweep deleted the text at its 90-day horizon.
    Swept { at: Option<Timestamp> },
    /// A forget masked the passage.
    Redacted,
    /// The turn asked to forget, so it was never stored.
    ForgetRequested,
    /// The owner removed the turn, or the document with every version of it.
    Removed,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AccessEntry {
    pub kind: String,
    pub at: Timestamp,
    pub turn: i64,
    pub source: Option<Uuid>,
    /// The predecessor it's inherited from, along `superseded_by`.
    pub inherited_from: Option<Uuid>,
}

/// A claim a repeat absorbed into the memory: the sentence call 1 wrote,
/// when it was said, and its label.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RestatementEntry {
    pub sentence: String,
    pub observed_at: Timestamp,
    pub label: String,
}

/// The supersession chain the memory is in.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChainView {
    pub head: Uuid,
    /// Oldest first.
    pub members: Vec<ChainMember>,
    /// The memory that ended this one's window, if any.
    pub ended_by: Option<Uuid>,
    /// The memories whose windows this one ended.
    pub ends: Vec<Uuid>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChainMember {
    pub id: Uuid,
    pub superseded_by: Option<Uuid>,
    pub retracted: bool,
    pub hidden: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LinkedEntity {
    pub id: Uuid,
    pub name: String,
    pub surface_form: Option<String>,
}

/// Strength now and its parts: `significance_boost + max(recent_use,
/// lasting_floor)`. −∞ shows as `null`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StrengthView {
    pub value: f64,
    /// S·significance.
    pub significance_boost: f64,
    pub recent_use: f64,
    pub lasting_floor: f64,
    /// Accesses counted as separate occasions for the floor.
    pub occasions: u32,
    /// τ: recall leaves out anything below it.
    pub threshold: f64,
    pub recallable: bool,
}

/// Purge, read on the chain's head as the sweep reads it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PurgeView {
    pub head: Uuid,
    pub head_strength: f64,
    /// δ, the margin below τ: `None` never purges.
    pub delta: Option<f64>,
    /// τ − δ.
    pub line: Option<f64>,
    /// What's holding a purge back now, if anything.
    pub guards: Vec<Guard>,
    pub eligible_now: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "guard", rename_all = "snake_case")]
pub enum Guard {
    /// δ is unset, so nothing is ever purged.
    PurgeDisabled,
    /// The deletion settings changed and no one has acknowledged them.
    PurgePaused,
    /// Forget hid the chain; its erase removes it instead.
    Forgotten,
    /// The head's start or end date hasn't passed yet.
    DateAhead { until: Timestamp },
    /// A task is held until its overdue window ends.
    OverdueTask { until: Timestamp },
    /// The head is above the purge line.
    Strength,
    /// The head's lasting strength, from its occasions and significance,
    /// keeps it from purge for good.
    Lasting,
}

/// Fade and purge dates if the memory isn't used again. Each is the bank
/// days from now, and the earliest world date it can come: the one it
/// would be if bank time ran at full speed from now.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Projection {
    pub basis: &'static str,
    /// When strength falls below τ and recall stops finding it.
    pub fade: Option<Projected>,
    /// When the chain can be purged: its head below τ − δ, not held by its
    /// lasting strength, and its guards past. `None` when it never can, or purge is off.
    pub purge: Option<Projected>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Projected {
    pub bank_days: f64,
    /// The earliest world date, at full speed.
    pub earliest_at: Timestamp,
}

const BASIS: &str =
    "if it isn't used again; bank days from now, and the earliest world date at full speed";

fn parse(text: &str) -> Uuid {
    text.parse().expect("a stored uuid parses")
}

fn later(now: Timestamp, days: f64) -> Timestamp {
    now.checked_add(jiff::SignedDuration::from_secs_f64(days * 86_400.0))
        .unwrap_or(Timestamp::MAX)
}

/// The edit log rows `filter` selects, oldest first.
fn edits(conn: &Connection, filter: &str, id: i64) -> Result<Vec<EditEntry>, rusqlite::Error> {
    let mut statement = conn.prepare(&format!(
        "SELECT e.uuid, e.kind, e.at, m.uuid, n.uuid, e.details FROM edits e
         LEFT JOIN memories m ON m.id = e.memory_id
         LEFT JOIN entities n ON n.id = e.entity_id
         WHERE {filter} ORDER BY e.id"
    ))?;
    statement
        .query_map([id], |row| {
            Ok(EditEntry {
                id: parse(&row.get::<_, String>(0)?),
                kind: row.get(1)?,
                at: timestamp(row.get(2)?),
                memory: row.get::<_, Option<String>>(3)?.map(|u| parse(&u)),
                entity: row.get::<_, Option<String>>(4)?.map(|u| parse(&u)),
                details: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
            })
        })?
        .collect()
}

/// `memory show <id>`.
pub(crate) fn memory(
    store: &Store,
    tuning: &Tuning,
    pause: &PurgePause,
    bank: &str,
    id: &str,
) -> Result<MemoryView, InspectError> {
    let now = store.now();
    let conn = store.connection();
    let (bank_id, _) = find_bank(&conn, bank)?.ok_or(InspectError::UnknownBank)?;
    let uuid = id
        .trim()
        .parse::<Uuid>()
        .map_err(|_| InspectError::UnknownMemory)?;
    let memory_id: i64 = conn
        .query_row(
            "SELECT id FROM memories WHERE uuid = ?1 AND bank_id = ?2",
            (uuid.to_string(), bank_id),
            |row| row.get(0),
        )
        .optional()?
        .ok_or(InspectError::UnknownMemory)?;

    let row = conn.query_row(
        "SELECT m.content, m.kind, m.significance, m.owner_significance, m.observed_at,
                m.created_at, m.hidden_at, m.invalidated_at, m.valid_from, m.valid_from_precision,
                m.valid_until, m.valid_until_precision, m.until_event, m.window_confidence,
                m.due_at, m.due_at_precision, m.volatility, m.recurrence_text, m.recurrence_rrule,
                m.source_start, m.source_end, m.ended_by,
                c.uuid, c.text, c.tombstoned_at,
                s.uuid, s.kind, s.session_id, s.document_id, s.message_at, s.timezone,
                s.secret_kinds, s.tombstoned_at, s.tombstone_reason, s.removed_at,
                m.recurrence_start, m.recurrence_start_precision
         FROM memories m JOIN chunks c ON c.id = m.chunk_id JOIN sources s ON s.id = c.source_id
         WHERE m.id = ?1",
        [memory_id],
        |row| {
            Ok(MemoryRow {
                content: row.get(0)?,
                kind: row.get(1)?,
                significance: row.get(2)?,
                owner: row.get(3)?,
                observed_at: timestamp(row.get(4)?),
                created_at: timestamp(row.get(5)?),
                hidden_at: row.get::<_, Option<i64>>(6)?.map(timestamp),
                invalidated_at: row.get::<_, Option<i64>>(7)?.map(timestamp),
                valid_from: world_time(row.get(8)?, row.get(9)?),
                valid_until: world_time(row.get(10)?, row.get(11)?),
                until_event: row.get(12)?,
                window_confidence: row.get(13)?,
                due_at: world_time(row.get(14)?, row.get(15)?),
                volatility: row.get(16)?,
                recurrence_text: row.get(17)?,
                recurrence_rrule: row.get(18)?,
                recurrence_start: world_time(row.get(35)?, row.get(36)?),
                start: row.get(19)?,
                end: row.get(20)?,
                ended_by: row.get(21)?,
                chunk: parse(&row.get::<_, String>(22)?),
                chunk_text: row.get(23)?,
                chunk_tombstoned_at: row.get::<_, Option<i64>>(24)?.map(timestamp),
                source: parse(&row.get::<_, String>(25)?),
                source_kind: row.get(26)?,
                session_id: row.get(27)?,
                document_id: row.get(28)?,
                message_at: row.get::<_, Option<i64>>(29)?.map(timestamp),
                timezone: row.get(30)?,
                secret_kinds: row.get(31)?,
                source_tombstoned_at: row.get::<_, Option<i64>>(32)?.map(timestamp),
                tombstone_reason: row.get(33)?,
                removed: row.get::<_, Option<i64>>(34)?.is_some(),
            })
        },
    )?;

    let tz = TimeZone::get(&row.timezone).unwrap_or(TimeZone::UTC);
    let phase = window(&conn, memory_id)?.map(|(window, tz)| window.phase(&tz, now));
    let loader = StrengthLoader::new(&conn, bank_id, tuning, now)?;
    let links = loader.links().to_vec();
    let inputs = loader.inputs(&conn, memory_id)?;
    let strength = loader.strength(&conn, memory_id)?;
    let effective = row.owner.clone().unwrap_or(row.significance.clone());

    let Outlook {
        fade,
        head,
        head_strength,
        line,
        guards,
        purge,
    } = outlook(&conn, tuning, pause, &loader, memory_id, now)?;
    let delta = tuning.purge.delta;
    let members = chain(&links, memory_id);

    let mut chain_members = Vec::new();
    {
        let mut statement = conn.prepare_cached(
            "SELECT m.uuid, n.uuid, m.invalidated_at IS NOT NULL, m.hidden_at IS NOT NULL
             FROM memories m LEFT JOIN memories n ON n.id = m.superseded_by WHERE m.id = ?1",
        )?;
        for member in &members {
            chain_members.push(statement.query_row([member], |row| {
                Ok(ChainMember {
                    id: parse(&row.get::<_, String>(0)?),
                    superseded_by: row.get::<_, Option<String>>(1)?.map(|u| parse(&u)),
                    retracted: row.get(2)?,
                    hidden: row.get(3)?,
                })
            })?);
        }
    }
    let ends: Vec<Uuid> = {
        let mut statement =
            conn.prepare_cached("SELECT uuid FROM memories WHERE ended_by = ?1 ORDER BY id")?;
        statement
            .query_map([memory_id], |row| row.get::<_, String>(0))?
            .map(|uuid| uuid.map(|u| parse(&u)))
            .collect::<Result<_, _>>()?
    };
    let ended_by = match row.ended_by {
        Some(id) => Some(conn.query_row(
            "SELECT uuid FROM memories WHERE id = ?1",
            [id],
            |row| row.get::<_, String>(0).map(|u| parse(&u)),
        )?),
        None => None,
    };

    let mut accesses = Vec::new();
    {
        let mut statement = conn.prepare_cached(
            "SELECT a.kind, a.at, a.turn, s.uuid, m.uuid FROM accesses a
             JOIN memories m ON m.id = a.memory_id
             LEFT JOIN sources s ON s.id = a.source_id
             WHERE a.memory_id = ?1",
        )?;
        for member in crate::strength::inherits_from(&links, memory_id) {
            for access in statement.query_map([member], |row| {
                let owner = parse(&row.get::<_, String>(4)?);
                Ok(AccessEntry {
                    kind: row.get(0)?,
                    at: timestamp(row.get(1)?),
                    turn: row.get(2)?,
                    source: row.get::<_, Option<String>>(3)?.map(|u| parse(&u)),
                    inherited_from: (owner != uuid).then_some(owner),
                })
            })? {
                accesses.push(access?);
            }
        }
    }
    accesses.sort_by_key(|access| (access.at, access.turn));

    let entities: Vec<LinkedEntity> = {
        let mut statement = conn.prepare_cached(
            "SELECT e.uuid, e.name, me.surface_form FROM memory_entities me
             JOIN entities e ON e.id = me.entity_id WHERE me.memory_id = ?1 ORDER BY e.id",
        )?;
        statement
            .query_map([memory_id], |row| {
                Ok(LinkedEntity {
                    id: parse(&row.get::<_, String>(0)?),
                    name: row.get(1)?,
                    surface_form: row.get(2)?,
                })
            })?
            .collect::<Result<_, _>>()?
    };

    let restatements: Vec<RestatementEntry> = {
        let mut statement = conn.prepare_cached(
            "SELECT json_extract(claim, '$.sentence'), observed_at, label FROM restatements
             WHERE memory_id = ?1 ORDER BY observed_at DESC, id DESC",
        )?;
        statement
            .query_map([memory_id], |row| {
                Ok(RestatementEntry {
                    sentence: row.get(0)?,
                    observed_at: timestamp(row.get(1)?),
                    label: row.get(2)?,
                })
            })?
            .collect::<Result<_, _>>()?
    };

    let (passage, gone) = passage(&row);
    Ok(MemoryView {
        id: uuid,
        sentence: row.content.clone(),
        kind: row.kind.clone(),
        phase,
        window: WindowView {
            valid_from: row.valid_from,
            valid_until: row.valid_until,
            until_event: row.until_event.clone(),
            window_confidence: row.window_confidence.clone(),
            due_at: row.due_at,
            volatility: row.volatility.clone(),
            recurrence: row.recurrence_text.clone(),
            recurrence_rrule: row.recurrence_rrule.clone(),
            recurrence_start: row.recurrence_start,
            timezone: tz.iana_name().unwrap_or(&row.timezone).to_string(),
        },
        observed_at: row.observed_at,
        created_at: row.created_at,
        hidden_at: row.hidden_at,
        retracted_at: row.invalidated_at,
        significance: SignificanceView {
            extracted: row.significance.clone(),
            owner: row.owner.clone(),
            value: inputs.significance,
            effective,
        },
        source: SourceView {
            source: row.source,
            kind: row.source_kind.clone(),
            session_id: row.session_id.clone(),
            document_id: row.document_id.clone(),
            message_at: row.message_at,
            chunk: row.chunk,
            start: row.start,
            end: row.end,
            passage,
            gone,
            secret_kinds: row
                .secret_kinds
                .as_deref()
                .and_then(|kinds| serde_json::from_str(kinds).ok())
                .unwrap_or_default(),
        },
        accesses,
        edits: edits(&conn, "e.memory_id = ?1", memory_id)?,
        restatements,
        chain: ChainView {
            head: parse(&conn.query_row(
                "SELECT uuid FROM memories WHERE id = ?1",
                [head],
                |row| row.get::<_, String>(0),
            )?),
            members: chain_members,
            ended_by,
            ends,
        },
        entities,
        strength: StrengthView {
            value: strength.value,
            significance_boost: S * inputs.significance,
            recent_use: strength.recent_use,
            lasting_floor: strength.lasting_floor,
            occasions: strength.occasions,
            threshold: TAU,
            recallable: strength.value >= TAU,
        },
        purge: PurgeView {
            head: parse(&conn.query_row(
                "SELECT uuid FROM memories WHERE id = ?1",
                [head],
                |row| row.get::<_, String>(0),
            )?),
            head_strength,
            delta,
            line,
            eligible_now: guards.is_empty(),
            guards,
        },
        projection: Projection {
            basis: BASIS,
            fade,
            purge,
        },
    })
}

/// A memory's fade and purge, if it isn't used again: what `memory show`
/// projects, and what each row of the memory list carries.
struct Outlook {
    fade: Option<Projected>,
    /// The chain's head, which purge reads.
    head: i64,
    head_strength: f64,
    line: Option<f64>,
    guards: Vec<Guard>,
    purge: Option<Projected>,
}

fn outlook(
    conn: &Connection,
    tuning: &Tuning,
    pause: &PurgePause,
    loader: &StrengthLoader,
    memory_id: i64,
    now: Timestamp,
) -> Result<Outlook, rusqlite::Error> {
    let inputs = loader.inputs(conn, memory_id)?;
    let fade = projected_below(
        inputs.significance,
        &inputs.accesses,
        inputs.close,
        loader.bank_time(),
        now,
        TAU,
    )
    .map(|days| Projected {
        bank_days: days,
        earliest_at: later(now, days),
    });

    // Purge reads the chain's head.
    let links = loader.links();
    let head = chain_head(links, memory_id);
    let members = chain(links, memory_id);
    let head_parts = loader.strength(conn, head)?;
    let head_strength = head_parts.value;
    let lasting = never_purged(head_parts.lasting);
    let delta = tuning.purge.delta;
    let line = delta.map(|delta| TAU - delta);
    let mut guards = Vec::new();
    let mut held_until: Option<Timestamp> = None;
    if delta.is_none() {
        guards.push(Guard::PurgeDisabled);
    }
    if *pause != PurgePause::Running {
        guards.push(Guard::PurgePaused);
    }
    let hidden = {
        let mut statement = conn.prepare_cached("SELECT hidden_at FROM memories WHERE id = ?1")?;
        let mut hidden = false;
        for member in &members {
            hidden |= statement
                .query_row([member], |row| row.get::<_, Option<i64>>(0))?
                .is_some();
        }
        hidden
    };
    if hidden {
        guards.push(Guard::Forgotten);
    }
    if let Some((head_window, head_tz)) = window(conn, head)? {
        let date_until = [head_window.valid_from, head_window.valid_until]
            .into_iter()
            .flatten()
            .map(|time| unit_end(time, &head_tz))
            .max();
        if let Some(until) = date_until.filter(|until| *until > now) {
            guards.push(Guard::DateAhead { until });
            held_until = Some(until);
        }
        if let Some(until) = head_window
            .overdue_until(&head_tz, tuning.agenda.overdue_days)
            .filter(|until| now < *until)
        {
            guards.push(Guard::OverdueTask { until });
            held_until = Some(held_until.map_or(until, |held| held.max(until)));
        }
    }
    if line.is_some_and(|line| head_strength >= line) {
        guards.push(Guard::Strength);
    }
    if lasting {
        guards.push(Guard::Lasting);
    }
    // The first time the guards have cleared and the head is below the
    // line together: a window closing as its date guard clears restarts
    // recent use, so the head can be below the line now and above it then.
    let purge = match line {
        Some(line) if !hidden && !lasting => {
            let head_inputs = loader.inputs(conn, head)?;
            projected_below_after(
                head_inputs.significance,
                &head_inputs.accesses,
                head_inputs.close,
                loader.bank_time(),
                now,
                held_until.map_or(now, |held| held.max(now)),
                line,
            )
            .map(|days| Projected {
                bank_days: days,
                earliest_at: later(now, days),
            })
        }
        _ => None,
    };
    Ok(Outlook {
        fade,
        head,
        head_strength,
        line,
        guards,
        purge,
    })
}

struct MemoryRow {
    content: String,
    kind: String,
    significance: String,
    owner: Option<String>,
    observed_at: Timestamp,
    created_at: Timestamp,
    hidden_at: Option<Timestamp>,
    invalidated_at: Option<Timestamp>,
    valid_from: Option<WorldTime>,
    valid_until: Option<WorldTime>,
    until_event: Option<String>,
    window_confidence: String,
    due_at: Option<WorldTime>,
    volatility: Option<String>,
    recurrence_text: Option<String>,
    recurrence_rrule: Option<String>,
    recurrence_start: Option<WorldTime>,
    start: i64,
    end: i64,
    ended_by: Option<i64>,
    chunk: Uuid,
    chunk_text: Option<String>,
    chunk_tombstoned_at: Option<Timestamp>,
    source: Uuid,
    source_kind: String,
    session_id: Option<String>,
    document_id: Option<String>,
    message_at: Option<Timestamp>,
    timezone: String,
    secret_kinds: Option<String>,
    source_tombstoned_at: Option<Timestamp>,
    tombstone_reason: Option<String>,
    removed: bool,
}

/// The memory's passage, or why it's gone.
fn passage(row: &MemoryRow) -> (Option<String>, Option<Gone>) {
    let Some(text) = &row.chunk_text else {
        let gone = if row.removed {
            Gone::Removed
        } else if row.tombstone_reason.as_deref() == Some("forget_requested") {
            Gone::ForgetRequested
        } else {
            Gone::Swept {
                at: row.chunk_tombstoned_at.or(row.source_tombstoned_at),
            }
        };
        return (None, Some(gone));
    };
    let start = usize::try_from(row.start).unwrap_or(0);
    let end = usize::try_from(row.end).unwrap_or(0);
    let passage: String = text
        .chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect();
    if !passage.is_empty()
        && passage
            .chars()
            .all(|c| c == REDACTION_MASK || c.is_whitespace())
    {
        return (None, Some(Gone::Redacted));
    }
    (Some(passage), None)
}

/// An entity in a bank, including seeded and merged entities.
/// Use its id with [`crate::Service::show_entity`] for the full view.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EntitySummary {
    pub id: Uuid,
    pub name: String,
    pub kind: String,
    pub merged_into: Option<Uuid>,
}

pub(crate) fn entities(store: &Store, bank: &str) -> Result<Vec<EntitySummary>, InspectError> {
    let conn = store.connection();
    let (bank_id, _) = find_bank(&conn, bank)?.ok_or(InspectError::UnknownBank)?;
    let mut statement = conn.prepare(
        "SELECT e.uuid, e.name, e.kind, target.uuid
         FROM entities e LEFT JOIN entities target ON target.id = e.merged_into
         WHERE e.bank_id = ?1 ORDER BY e.name, e.uuid",
    )?;
    Ok(statement
        .query_map([bank_id], |row| {
            Ok(EntitySummary {
                id: parse(&row.get::<_, String>(0)?),
                name: row.get(1)?,
                kind: row.get(2)?,
                merged_into: row.get::<_, Option<String>>(3)?.map(|id| parse(&id)),
            })
        })?
        .collect::<Result<_, _>>()?)
}

/// `entity show`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EntityView {
    pub id: Uuid,
    pub name: String,
    pub kind: String,
    /// `user` or `assistant` for the entities every bank starts with.
    pub seeded: Option<String>,
    pub merged_into: Option<Uuid>,
    /// The entity at the end of the merge chain, when it's merged.
    pub survivor: Option<Uuid>,
    /// Entities merged into this one.
    pub merged_from: Vec<Uuid>,
    pub aliases: Vec<String>,
    pub speaker_ids: Vec<String>,
    /// Memories linked to it or to an entity merged into it.
    pub memory_count: usize,
    /// The newest [`ENTITY_MEMORIES`] of them.
    pub memories: Vec<EntityMemory>,
    /// Models whose entity filter is this entity.
    pub models: Vec<String>,
    pub edits: Vec<EditEntry>,
    pub created_at: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EntityMemory {
    pub id: Uuid,
    pub sentence: String,
    pub kind: String,
    pub surface_form: Option<String>,
    /// The merged entity the link is on, when it isn't this one.
    pub via: Option<Uuid>,
}

/// `entity show <id|name>`.
pub(crate) fn entity(
    store: &Store,
    bank: &str,
    reference: &str,
) -> Result<EntityView, InspectError> {
    let conn = store.connection();
    let (bank_id, _) = find_bank(&conn, bank)?.ok_or(InspectError::UnknownBank)?;
    let found = resolve(&conn, bank_id, reference)?;
    let (name, kind, created_at): (String, String, i64) = conn.query_row(
        "SELECT name, kind, created_at FROM entities WHERE id = ?1",
        [found.id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    )?;
    let merged_into = found.merged_into.map(|id| uuid_of(&conn, id)).transpose()?;
    let last = survivor(&conn, found.id)?;
    let survivor = (last != found.id && Some(last) != found.merged_into)
        .then(|| uuid_of(&conn, last))
        .transpose()?;
    let strings = |sql: &str| -> Result<Vec<String>, rusqlite::Error> {
        let mut statement = conn.prepare(sql)?;
        statement
            .query_map([found.id], |row| row.get(0))?
            .collect::<Result<_, _>>()
    };
    let merged_from = strings("SELECT uuid FROM entities WHERE merged_into = ?1 ORDER BY id")?
        .iter()
        .map(|u| parse(u))
        .collect();
    let aliases = strings("SELECT alias FROM entity_aliases WHERE entity_id = ?1 ORDER BY id")?;
    let speaker_ids =
        strings("SELECT platform_id FROM speaker_ids WHERE entity_id = ?1 ORDER BY id")?;
    let models = strings("SELECT name FROM mental_models WHERE filter_entity_id = ?1 ORDER BY id")?;
    let link_filter = "FROM memory_entities me
         JOIN memories m ON m.id = me.memory_id
         JOIN entities e ON e.id = me.entity_id
         WHERE m.hidden_at IS NULL AND (e.id = ?1 OR e.merged_into = ?1)";
    let memory_count: i64 = conn.query_row(
        &format!("SELECT COUNT(DISTINCT me.memory_id) {link_filter}"),
        [found.id],
        |row| row.get(0),
    )?;
    let memories = {
        let mut statement = conn.prepare(&format!(
            "SELECT m.uuid, m.content, m.kind, me.surface_form, e.id, e.uuid {link_filter}
             ORDER BY m.id DESC LIMIT ?2"
        ))?;
        let mut seen = BTreeSet::new();
        let mut memories = Vec::new();
        for row in statement.query_map((found.id, ENTITY_MEMORIES as i64), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, String>(5)?,
            ))
        })? {
            let (id, sentence, kind, surface_form, entity, entity_uuid) = row?;
            if !seen.insert(id.clone()) {
                continue;
            }
            memories.push(EntityMemory {
                id: parse(&id),
                sentence,
                kind,
                surface_form,
                via: (entity != found.id).then(|| parse(&entity_uuid)),
            });
        }
        memories
    };
    Ok(EntityView {
        id: found.uuid,
        name,
        kind,
        seeded: found.seeded.clone(),
        merged_into,
        survivor,
        merged_from,
        aliases,
        speaker_ids,
        memory_count: usize::try_from(memory_count).unwrap_or(0),
        memories,
        models,
        edits: edits(&conn, "e.entity_id = ?1", found.id)?,
        created_at: timestamp(created_at),
    })
}

/// `model show`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelView {
    #[serde(flatten)]
    pub model: Model,
    /// The entity filter's id.
    pub entity_id: Option<Uuid>,
    pub refresh_requested_at: Option<Timestamp>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    /// Each memory the answer cites, with its status.
    pub cited: Vec<CitedMemory>,
    /// Whether the block shows the model, as the block is laid out now:
    /// it's enabled, has an answer, every memory the answer cites is
    /// current, and at least its first sentence fits what the agenda and
    /// the older models leave of the budget.
    pub renders: bool,
    /// The answer as the block shows it: whole, or cut at a sentence or line end.
    /// `None` when the block leaves the model out.
    pub shown_answer: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CitedMemory {
    pub id: Uuid,
    pub sentence: String,
    pub status: CitedStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CitedStatus {
    Current,
    Ended,
    Retracted,
    Forgotten,
}

/// `model show <name>`.
pub(crate) fn model_view(
    store: &Store,
    tuning: &Tuning,
    bank: &str,
    name: &str,
) -> Result<ModelView, InspectError> {
    let now = store.now();
    let conn = store.connection();
    let (bank_id, timezone) = find_bank(&conn, bank)?.ok_or(InspectError::UnknownBank)?;
    let row = find_model(&conn, bank_id, name)?.ok_or(InspectError::UnknownModel)?;
    let shown = model(&conn, &row)?;
    let (created_at, updated_at): (i64, i64) = conn.query_row(
        "SELECT created_at, updated_at FROM mental_models WHERE id = ?1",
        [row.id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let mut statement = conn.prepare(
        "SELECT m.uuid, m.content, m.invalidated_at IS NOT NULL, m.hidden_at IS NOT NULL,
                m.valid_until, m.valid_until_precision, s.timezone
         FROM mental_model_cites c
         JOIN memories m ON m.id = c.memory_id
         JOIN chunks k ON k.id = m.chunk_id JOIN sources s ON s.id = k.source_id
         WHERE c.model_id = ?1 ORDER BY c.rowid",
    )?;
    let cited: Vec<CitedMemory> = statement
        .query_map([row.id], |row| {
            let retracted: bool = row.get(2)?;
            let hidden: bool = row.get(3)?;
            let until = world_time(row.get(4)?, row.get(5)?);
            let tz = TimeZone::get(&row.get::<_, String>(6)?).unwrap_or(TimeZone::UTC);
            let status = if hidden {
                CitedStatus::Forgotten
            } else if retracted {
                CitedStatus::Retracted
            } else if crate::agenda::has_ended(until, &tz, now) {
                CitedStatus::Ended
            } else {
                CitedStatus::Current
            };
            Ok(CitedMemory {
                id: parse(&row.get::<_, String>(0)?),
                sentence: row.get(1)?,
                status,
            })
        })?
        .collect::<Result<_, _>>()?;
    // What the block would show now, from the same layout it's built by.
    let tz = TimeZone::get(&timezone).unwrap_or(TimeZone::UTC);
    let shown_answer = crate::system_prompt::lay_out(&conn, tuning, bank_id, &tz, now)?
        .models
        .remove(&row.id);
    Ok(ModelView {
        model: shown,
        entity_id: row.entity_id.map(|id| uuid_of(&conn, id)).transpose()?,
        refresh_requested_at: row.refresh_requested_at,
        created_at: timestamp(created_at),
        updated_at: timestamp(updated_at),
        cited,
        renders: shown_answer.is_some(),
        shown_answer,
    })
}

/// The first instant the memory's strength fell below τ, or `None` when
/// it's at or above τ now and always was. Between two accesses (and after a
/// window close that restarted recent use) strength only falls, so each
/// stretch is checked at its end and the first that ends below τ is
/// bisected to the minute, the precision of `memory show`'s projection.
pub(crate) fn faded_at(
    store: &Store,
    tuning: &Tuning,
    bank: &str,
    id: Uuid,
) -> Result<Option<Timestamp>, InspectError> {
    let now = store.now();
    let conn = store.connection();
    let (bank_id, _) = find_bank(&conn, bank)?.ok_or(InspectError::UnknownBank)?;
    let memory_id: i64 = conn
        .query_row(
            "SELECT id FROM memories WHERE uuid = ?1 AND bank_id = ?2",
            (id.to_string(), bank_id),
            |row| row.get(0),
        )
        .optional()?
        .ok_or(InspectError::UnknownMemory)?;
    let loader = StrengthLoader::new(&conn, bank_id, tuning, now)?;
    let inputs = loader.inputs(&conn, memory_id)?;
    let bank_time = loader.bank_time();
    let value = |at: Timestamp| {
        crate::strength::strength(
            inputs.significance,
            &inputs.accesses,
            inputs.close,
            bank_time,
            at,
        )
        .value
    };

    let mut points: Vec<Timestamp> = inputs
        .accesses
        .iter()
        .map(|access| access.at)
        .filter(|at| *at <= now)
        .collect();
    if let Some(close) = inputs.close {
        let restart = close.closes_at.max(close.known_at);
        if restart <= now {
            points.push(restart);
        }
    }
    points.sort();
    points.dedup();

    let minute = jiff::SignedDuration::from_mins(1);
    let microsecond = jiff::SignedDuration::from_micros(1);
    for (index, &start) in points.iter().enumerate() {
        let next = points.get(index + 1).copied();
        // Just before the next access, which lifts strength again; or now.
        let end = match next {
            Some(next) => next.checked_sub(microsecond).unwrap_or(next),
            None => now,
        };
        if end <= start || value(end) >= TAU {
            continue;
        }
        let (mut above, mut below) = (start, end);
        while below.duration_since(above) > minute {
            let middle = above
                .checked_add(below.duration_since(above) / 2)
                .unwrap_or(below);
            if value(middle) < TAU {
                below = middle;
            } else {
                above = middle;
            }
        }
        return Ok(Some(below));
    }
    Ok(None)
}

/// Every live memory's strength now: neither forgotten nor retracted, in
/// rowid order.
pub(crate) fn strengths(
    store: &Store,
    tuning: &Tuning,
    bank: &str,
) -> Result<Vec<(Uuid, f64)>, InspectError> {
    let now = store.now();
    let conn = store.connection();
    let (bank_id, _) = find_bank(&conn, bank)?.ok_or(InspectError::UnknownBank)?;
    let loader = StrengthLoader::new(&conn, bank_id, tuning, now)?;
    let mut statement = conn.prepare_cached(
        "SELECT id, uuid FROM memories
         WHERE bank_id = ?1 AND hidden_at IS NULL AND invalidated_at IS NULL
         ORDER BY id",
    )?;
    let rows: Vec<(i64, String)> = statement
        .query_map([bank_id], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<Result<_, _>>()?;
    rows.into_iter()
        .map(|(memory_id, uuid)| Ok((parse(&uuid), loader.strength(&conn, memory_id)?.value)))
        .collect()
}
