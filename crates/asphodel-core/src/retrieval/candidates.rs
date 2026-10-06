//! Cleaning up the retrievers' hits before fusion.
//!
//! - A retracted or hidden hit is dropped. Retracted memories are kept for
//!   audit only, and a hidden one is waiting to be erased.
//! - Any other hit is replaced by the head of its supersession chain, as
//!   reconciliation does, so a refined memory shows its latest version. A
//!   chain whose head is retracted or hidden shows nothing.
//! - Strength is computed in-process for each head, with no cached column.
//! - The caller's filter then decides: injection drops heads below τ and
//!   those already in context, and explicit recall applies its parameters.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use jiff::Timestamp;
use jiff::tz::TimeZone;
use rusqlite::Connection;
use uuid::Uuid;

use crate::config::Tuning;
use crate::constants::Volatility;
use crate::store::strength::{StrengthLoader, memory_kind, world_time};
use crate::store::timestamp;
use crate::strength::{AccessKind, Chains, Kind, Phase, Window, state_confidence, strength};

/// A memory that survived clean-up, with what ranking and rendering need.
#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub id: i64,
    pub uuid: Uuid,
    pub content: String,
    pub window: Window,
    pub low_confidence: bool,
    pub until_event: Option<String>,
    pub recurrence_text: Option<String>,
    pub rrule: Option<String>,
    pub observed_at: Timestamp,
    /// Its source's timezone, which its window's units are in.
    pub tz: TimeZone,
    pub phase: Phase,
    /// Strength's value at the loader's instant. −∞ with no accesses.
    pub strength: f64,
    /// 1.0 for anything but a state.
    pub state_confidence: f64,
    /// When the state was last said to hold: `observed_at`, or a later
    /// `mentioned_again` or `confirmed`, as state confidence ages it.
    pub last_observed: Timestamp,
    /// A state's volatility; `None` for anything else, or a state with none.
    pub volatility: Option<Volatility>,
    pub kept: bool,
    /// The level strength uses: the owner's setting over the extracted
    /// one, `trivial` to `critical`, or `kept`.
    pub significance: String,
}

/// Turns hits into candidates for one recall, caching each head.
pub(crate) struct Cleanup<'a> {
    conn: &'a Connection,
    chains: Chains,
    strength: StrengthLoader,
    now: Timestamp,
    /// Whether a hit's own row is retracted or hidden, by rowid.
    dropped_hits: BTreeMap<i64, bool>,
    /// Each head's candidate and whether `keep` admitted it, or `None`
    /// when it's dropped.
    heads: BTreeMap<i64, Option<(Candidate, bool)>>,
    keep: &'a dyn Fn(&Candidate) -> bool,
}

impl<'a> Cleanup<'a> {
    pub(crate) fn new(
        conn: &'a Connection,
        bank_id: i64,
        tuning: &Tuning,
        now: Timestamp,
        keep: &'a dyn Fn(&Candidate) -> bool,
    ) -> Result<Self, rusqlite::Error> {
        let strength = StrengthLoader::new(conn, bank_id, tuning, now)?;
        Ok(Self {
            conn,
            chains: Chains::new(strength.links()),
            strength,
            now,
            dropped_hits: BTreeMap::new(),
            heads: BTreeMap::new(),
            keep,
        })
    }

    /// Cleans up one retriever's hits, best first: each hit becomes its
    /// chain's head, or nothing, and each head is listed once.
    pub(crate) fn list(&mut self, hits: &[i64]) -> Result<Vec<i64>, rusqlite::Error> {
        let mut ranked = Vec::new();
        for &hit in hits {
            if let Some(head) = self.shown(hit)?
                && !ranked.contains(&head)
            {
                ranked.push(head);
            }
        }
        Ok(ranked)
    }

    /// [`Cleanup::list`] for hits carrying `T` along: each head listed once,
    /// with what its best hit carried.
    pub(crate) fn list_with<T: Copy>(
        &mut self,
        hits: &[(i64, T)],
    ) -> Result<Vec<(i64, T)>, rusqlite::Error> {
        let mut ranked: Vec<(i64, T)> = Vec::new();
        for &(hit, carried) in hits {
            if let Some(head) = self.shown(hit)?
                && !ranked.iter().any(|(listed, _)| *listed == head)
            {
                ranked.push((head, carried));
            }
        }
        Ok(ranked)
    }

    /// `head` and every memory in its chain before it.
    pub(crate) fn members(&self, head: i64) -> std::collections::BTreeSet<i64> {
        crate::strength::inherits_from(self.strength.links(), head)
    }

    /// The candidates for `ids`, in that order. Every id must have come out
    /// of [`Cleanup::list`].
    pub(crate) fn take(&mut self, ids: &[i64]) -> Vec<Candidate> {
        ids.iter()
            .filter_map(|id| match self.heads.get(id) {
                Some(Some((candidate, true))) => Some(candidate.clone()),
                _ => None,
            })
            .collect()
    }

    /// The heads of `hits` that `keep` refused, best first, each once.
    /// Every hit must have been through [`Cleanup::list`].
    pub(crate) fn refused(&mut self, hits: &[i64]) -> Vec<i64> {
        let mut ranked = Vec::new();
        for &hit in hits {
            if self.dropped_hits.get(&hit) != Some(&false) {
                continue;
            }
            let head = self.chains.head(hit);
            if matches!(self.heads.get(&head), Some(Some((_, false)))) && !ranked.contains(&head) {
                ranked.push(head);
            }
        }
        ranked
    }

    /// The refused candidates for `ids`, in that order. Every id must have
    /// come out of [`Cleanup::refused`].
    pub(crate) fn take_refused(&self, ids: &[i64]) -> Vec<Candidate> {
        ids.iter()
            .filter_map(|id| match self.heads.get(id) {
                Some(Some((candidate, false))) => Some(candidate.clone()),
                _ => None,
            })
            .collect()
    }

    fn shown(&mut self, hit: i64) -> Result<Option<i64>, rusqlite::Error> {
        let dropped = match self.dropped_hits.entry(hit) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => *entry.insert(self.conn.query_row(
                "SELECT invalidated_at IS NOT NULL OR hidden_at IS NOT NULL
                 FROM memories WHERE id = ?1",
                [hit],
                |row| row.get(0),
            )?),
        };
        if dropped {
            return Ok(None);
        }
        let head = self.chains.head(hit);
        if let Entry::Vacant(entry) = self.heads.entry(head) {
            let candidate = load(self.conn, &self.strength, self.now, head)?.map(|candidate| {
                let admitted = (self.keep)(&candidate);
                (candidate, admitted)
            });
            entry.insert(candidate);
        }
        Ok(match &self.heads[&head] {
            Some((candidate, true)) => Some(candidate.id),
            _ => None,
        })
    }
}

/// The memory with rowid `id` as a candidate, or `None` when it's retracted
/// or hidden.
fn load(
    conn: &Connection,
    loader: &StrengthLoader,
    now: Timestamp,
    id: i64,
) -> Result<Option<Candidate>, rusqlite::Error> {
    let row = conn.query_row(
        "SELECT m.uuid, m.content, m.kind, m.observed_at,
                m.valid_from, m.valid_from_precision, m.valid_until, m.valid_until_precision,
                m.due_at, m.due_at_precision, m.window_confidence, m.until_event,
                m.recurrence_text, m.volatility, m.owner_significance,
                m.invalidated_at IS NOT NULL OR m.hidden_at IS NOT NULL, s.timezone, m.recurrence_rrule,
                m.significance
         FROM memories m JOIN chunks c ON c.id = m.chunk_id JOIN sources s ON s.id = c.source_id
         WHERE m.id = ?1",
        [id],
        |row| {
            Ok(Row {
                uuid: row.get(0)?,
                content: row.get(1)?,
                kind: row.get(2)?,
                observed_at: timestamp(row.get(3)?),
                window: (
                    world_time(row.get(4)?, row.get(5)?),
                    world_time(row.get(6)?, row.get(7)?),
                    world_time(row.get(8)?, row.get(9)?),
                ),
                window_confidence: row.get(10)?,
                until_event: row.get(11)?,
                recurrence_text: row.get(12)?,
                volatility: row.get(13)?,
                owner_significance: row.get(14)?,
                dropped: row.get(15)?,
                timezone: row.get(16)?,
                rrule: row.get(17)?,
                significance: row.get(18)?,
            })
        },
    )?;
    if row.dropped {
        return Ok(None);
    }
    let kind = memory_kind(&row.kind).unwrap_or(Kind::Fact);
    let (valid_from, valid_until, due_at) = row.window;
    let window = Window {
        kind,
        valid_from,
        valid_until,
        due_at,
    };
    let tz = TimeZone::get(&row.timezone).unwrap_or(TimeZone::UTC);
    let inputs = loader.inputs(conn, id)?;
    let strength = strength(
        inputs.significance,
        &inputs.accesses,
        inputs.close,
        loader.bank_time(),
        now,
    )
    .value;

    let volatility = row.volatility.as_deref().and_then(volatility);
    let (state_confidence, last_observed) = match volatility {
        Some(_) => {
            let accesses = &inputs.accesses;
            let last_observed = accesses
                .iter()
                .filter(|a| a.at <= now)
                .filter(|a| matches!(a.kind, AccessKind::MentionedAgain | AccessKind::Confirmed))
                .map(|a| a.at)
                .fold(row.observed_at, Timestamp::max);
            (
                state_confidence(volatility, row.observed_at, accesses, now),
                last_observed,
            )
        }
        None => (1.0, row.observed_at),
    };

    Ok(Some(Candidate {
        id,
        uuid: row.uuid.parse().expect("a stored memory uuid parses"),
        content: row.content,
        phase: window.phase(&tz, now),
        window,
        low_confidence: row.window_confidence == "low",
        until_event: row.until_event,
        recurrence_text: row.recurrence_text,
        rrule: row.rrule,
        observed_at: row.observed_at,
        tz,
        strength,
        state_confidence,
        last_observed,
        volatility,
        kept: row.owner_significance.as_deref() == Some("kept"),
        significance: row.owner_significance.unwrap_or(row.significance),
    }))
}

struct Row {
    uuid: String,
    content: String,
    kind: String,
    observed_at: Timestamp,
    window: (
        Option<crate::strength::WorldTime>,
        Option<crate::strength::WorldTime>,
        Option<crate::strength::WorldTime>,
    ),
    window_confidence: String,
    until_event: Option<String>,
    recurrence_text: Option<String>,
    rrule: Option<String>,
    volatility: Option<String>,
    owner_significance: Option<String>,
    significance: String,
    dropped: bool,
    timezone: String,
}

fn volatility(text: &str) -> Option<Volatility> {
    match text {
        "hours" => Some(Volatility::Hours),
        "days" => Some(Volatility::Days),
        "weeks" => Some(Volatility::Weeks),
        "months" => Some(Volatility::Months),
        "years" => Some(Volatility::Years),
        _ => None,
    }
}
