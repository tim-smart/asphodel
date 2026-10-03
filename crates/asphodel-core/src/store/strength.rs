//! Loading what the strength model needs about a memory, so every caller
//! computes the same strength the pure functions in [`crate::strength`]
//! define.
//!
//! A memory's strength takes:
//!
//! - its significance: the owner's setting when there is one (kept is
//!   [`SIGNIFICANCE_KEPT`]), otherwise the level extraction gave;
//! - its own accesses and those it inherits along `superseded_by`, never
//!   along `ended_by` ([`inherits_from`]);
//! - its window's close, when it has one, with the end known at the
//!   `observed_at` of the `ended_by` memory or else its own (ADR 0003);
//! - the bank's clock ([`BankTime`]), from the bank's turns.

use jiff::Timestamp;
use jiff::tz::TimeZone;
use rusqlite::{Connection, OptionalExtension};

use super::timestamp;
use crate::constants::{SIGNIFICANCE_KEPT, Significance};
use crate::strength::{
    Access, AccessKind, BankTime, Kind, Link, Strength, TimePrecision, Window, WindowClose,
    WorldTime, inherits_from, strength,
};

/// One bank's strength inputs that every memory shares: its clock, its
/// supersession links and the instant strength is taken at. Build it once
/// per bank and operation.
pub(crate) struct StrengthLoader {
    bank_time: BankTime,
    links: Vec<Link>,
    now: Timestamp,
}

impl StrengthLoader {
    /// `quiet_rate` is `Tuning::clock.quiet_rate`.
    pub(crate) fn new(
        conn: &Connection,
        bank_id: i64,
        quiet_rate: f64,
        now: Timestamp,
    ) -> Result<Self, rusqlite::Error> {
        // Tombstoned turns still happened, so they keep bank time running.
        let mut turns = conn.prepare_cached(
            "SELECT message_at FROM sources WHERE bank_id = ?1 AND kind = 'turn'",
        )?;
        let turns: Vec<Timestamp> = turns
            .query_map([bank_id], |row| row.get::<_, i64>(0))?
            .map(|micros| micros.map(timestamp))
            .collect::<Result<_, _>>()?;
        let mut links = conn.prepare_cached(
            "SELECT id, superseded_by, ended_by FROM memories
             WHERE bank_id = ?1 AND superseded_by IS NOT NULL",
        )?;
        let links: Vec<Link> = links
            .query_map([bank_id], |row| {
                Ok(Link {
                    id: row.get(0)?,
                    superseded_by: row.get(1)?,
                    ended_by: row.get(2)?,
                })
            })?
            .collect::<Result<_, _>>()?;
        Ok(Self {
            bank_time: BankTime::new(&turns, quiet_rate),
            links,
            now,
        })
    }

    /// The strength of the memory with rowid `memory_id` at the loader's
    /// `now`.
    pub(crate) fn strength(
        &self,
        conn: &Connection,
        memory_id: i64,
    ) -> Result<Strength, rusqlite::Error> {
        let inputs = self.inputs(conn, memory_id)?;
        Ok(strength(
            inputs.significance,
            &inputs.accesses,
            inputs.close,
            &self.bank_time,
            self.now,
        ))
    }

    /// The bank's clock, as loaded.
    pub(crate) fn bank_time(&self) -> &BankTime {
        &self.bank_time
    }

    /// What the strength of the memory with rowid `memory_id` is computed
    /// from, so `memory show` can project it forward.
    pub(crate) fn inputs(
        &self,
        conn: &Connection,
        memory_id: i64,
    ) -> Result<Inputs, rusqlite::Error> {
        let memory = conn.query_row(
            "SELECT m.significance, m.owner_significance, m.kind, m.observed_at,
                    m.valid_from, m.valid_from_precision, m.valid_until, m.valid_until_precision,
                    m.due_at, m.due_at_precision, m.ended_by, s.timezone
             FROM memories m JOIN chunks c ON c.id = m.chunk_id JOIN sources s ON s.id = c.source_id
             WHERE m.id = ?1",
            [memory_id],
            |row| {
                Ok(Memory {
                    significance: row.get(0)?,
                    owner_significance: row.get(1)?,
                    kind: row.get(2)?,
                    observed_at: timestamp(row.get(3)?),
                    valid_from: world_time(row.get(4)?, row.get(5)?),
                    valid_until: world_time(row.get(6)?, row.get(7)?),
                    due_at: world_time(row.get(8)?, row.get(9)?),
                    ended_by: row.get(10)?,
                    timezone: row.get(11)?,
                })
            },
        )?;

        let significance = significance_value(
            memory
                .owner_significance
                .as_deref()
                .unwrap_or(&memory.significance),
        );

        let accesses = self.accesses(conn, memory_id)?;

        let close = match memory_kind(&memory.kind) {
            Some(kind) => {
                let window = Window {
                    kind,
                    valid_from: memory.valid_from,
                    valid_until: memory.valid_until,
                    due_at: memory.due_at,
                };
                let tz = TimeZone::get(&memory.timezone).unwrap_or(TimeZone::UTC);
                match window.closes_at(&tz) {
                    Some(closes_at) => {
                        let known_at = match memory.ended_by {
                            Some(ended_by) => conn
                                .query_row(
                                    "SELECT observed_at FROM memories WHERE id = ?1",
                                    [ended_by],
                                    |row| row.get::<_, i64>(0),
                                )
                                .optional()?
                                .map(timestamp)
                                .unwrap_or(memory.observed_at),
                            None => memory.observed_at,
                        };
                        Some(WindowClose {
                            closes_at,
                            known_at,
                        })
                    }
                    None => None,
                }
            }
            None => None,
        };

        Ok(Inputs {
            significance,
            accesses,
            close,
        })
    }
}

/// A memory's strength inputs: its significance value, the accesses it
/// counts and its window's close.
pub(crate) struct Inputs {
    pub significance: f64,
    pub accesses: Vec<Access>,
    pub close: Option<WindowClose>,
}

impl StrengthLoader {
    /// The accesses the memory with rowid `memory_id` counts: its own and
    /// those it inherits along `superseded_by`, in no particular order.
    pub(crate) fn accesses(
        &self,
        conn: &Connection,
        memory_id: i64,
    ) -> Result<Vec<Access>, rusqlite::Error> {
        let mut accesses = Vec::new();
        let mut statement =
            conn.prepare_cached("SELECT kind, at FROM accesses WHERE memory_id = ?1")?;
        for id in inherits_from(&self.links, memory_id) {
            let rows = statement.query_map([id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?;
            for row in rows {
                let (kind, at) = row?;
                if let Some(kind) = access_kind(&kind) {
                    accesses.push(Access {
                        kind,
                        at: timestamp(at),
                    });
                }
            }
        }
        Ok(accesses)
    }

    /// The bank's supersession links, as loaded.
    pub(crate) fn links(&self) -> &[Link] {
        &self.links
    }
}

struct Memory {
    significance: String,
    owner_significance: Option<String>,
    kind: String,
    observed_at: Timestamp,
    valid_from: Option<WorldTime>,
    valid_until: Option<WorldTime>,
    due_at: Option<WorldTime>,
    ended_by: Option<i64>,
    timezone: String,
}

pub(crate) fn world_time(at: Option<i64>, precision: Option<String>) -> Option<WorldTime> {
    Some(WorldTime {
        at: timestamp(at?),
        precision: time_precision(precision.as_deref()?)?,
    })
}

fn time_precision(text: &str) -> Option<TimePrecision> {
    match text {
        "year" => Some(TimePrecision::Year),
        "month" => Some(TimePrecision::Month),
        "day" => Some(TimePrecision::Day),
        "hour" => Some(TimePrecision::Hour),
        "minute" => Some(TimePrecision::Minute),
        _ => None,
    }
}

pub(crate) fn memory_kind(text: &str) -> Option<Kind> {
    match text {
        "fact" => Some(Kind::Fact),
        "event" => Some(Kind::Event),
        "state" => Some(Kind::State),
        "task" => Some(Kind::Task),
        "recurring" => Some(Kind::Recurring),
        _ => None,
    }
}

fn access_kind(text: &str) -> Option<AccessKind> {
    match text {
        "created" => Some(AccessKind::Created),
        "used" => Some(AccessKind::Used),
        "mentioned_again" => Some(AccessKind::MentionedAgain),
        "confirmed" => Some(AccessKind::Confirmed),
        _ => None,
    }
}

/// A stored significance level, or `kept`, as its value.
pub(crate) fn significance_value(level: &str) -> f64 {
    match level {
        "kept" => SIGNIFICANCE_KEPT,
        "trivial" => Significance::Trivial.value(),
        "minor" => Significance::Minor.value(),
        "notable" => Significance::Notable.value(),
        "major" => Significance::Major.value(),
        _ => Significance::Critical.value(),
    }
}

/// The memory's validity window and its source's timezone, which purge's
/// guards read on a chain's head (ADR 0008). `None` for an unknown kind.
pub(crate) fn window(
    conn: &Connection,
    memory_id: i64,
) -> Result<Option<(Window, TimeZone)>, rusqlite::Error> {
    let (kind, valid_from, valid_until, due_at, timezone) = conn.query_row(
        "SELECT m.kind, m.valid_from, m.valid_from_precision, m.valid_until,
                m.valid_until_precision, m.due_at, m.due_at_precision, s.timezone
         FROM memories m JOIN chunks c ON c.id = m.chunk_id JOIN sources s ON s.id = c.source_id
         WHERE m.id = ?1",
        [memory_id],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                world_time(row.get(1)?, row.get(2)?),
                world_time(row.get(3)?, row.get(4)?),
                world_time(row.get(5)?, row.get(6)?),
                row.get::<_, String>(7)?,
            ))
        },
    )?;
    Ok(memory_kind(&kind).map(|kind| {
        (
            Window {
                kind,
                valid_from,
                valid_until,
                due_at,
            },
            TimeZone::get(&timezone).unwrap_or(TimeZone::UTC),
        )
    }))
}
