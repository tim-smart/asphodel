//! The agenda.
//!
//! Three groups, built by query with no LLM, and judged by bank-local date
//! so the clock alone changes them only at midnight:
//!
//! - **Dated lines**, chosen by world time alone: events starting and
//!   tasks due from today to `agenda.horizon_days` ahead, tasks overdue
//!   since up to `agenda.overdue_days` ago, and recurring memories with a
//!   period longer than a week whose next occurrence falls in the horizon.
//!   Both ends are inclusive. They're never gated on τ, so a minor
//!   appointment can't fade out on the day it matters. In date order, at
//!   most `agenda.dated_lines`; over the cap, faded items fold into a count
//!   first, then the least significant, then the furthest from today.
//! - **Routines**: recurring memories with a period of a week or less, or
//!   with no usable RRULE, at or above τ, strongest first and then by
//!   significance, at most `agenda.routines`.
//! - **Undated tasks**: open tasks with no due date, gated and ranked the
//!   same way, at most `agenda.undated_tasks`.
//!
//! Only current heads count: nothing retracted, forgotten, refined into a
//! newer version, or ended ([`has_ended`]: a stated end holds through its
//! unit, and an ending still ahead hasn't ended anything). Building it never
//! writes an access, and its
//! items count as in context once a session's block lists them.

use chrono::TimeZone as _;
use jiff::civil::Date;
use jiff::tz::TimeZone;
use jiff::{Timestamp, ToSpan};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::{SignificanceTuning, Tuning};
use crate::constants::TAU;
use crate::retrieval::candidates::Cleanup;
use crate::retrieval::format;
use crate::store::strength::{StrengthLoader, memory_kind, significance_value, world_time};
use crate::store::timestamp;
use crate::strength::{Kind, WorldTime, unit_end};

/// The bank's agenda now.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Agenda {
    /// In date order.
    pub dated: Vec<Uuid>,
    /// Dated items over the cap, shown only as a count.
    pub folded: usize,
    pub routines: Vec<Uuid>,
    pub undated_tasks: Vec<Uuid>,
}

impl Agenda {
    /// Every memory the agenda lists, in the order the block shows them.
    pub fn listed(&self) -> Vec<Uuid> {
        self.dated
            .iter()
            .chain(&self.routines)
            .chain(&self.undated_tasks)
            .copied()
            .collect()
    }
}

/// The agenda with each group's rendered lines, for the block. The lines
/// are in the same order as the agenda's ids.
pub(crate) struct Built {
    pub agenda: Agenda,
    pub dated: Vec<String>,
    pub routines: Vec<String>,
    pub undated_tasks: Vec<String>,
    /// Indexes into `dated`, in the order they'd fold next: faded first,
    /// then the least significant, then the furthest from today. The block
    /// folds in this order when the agenda alone is over its budget.
    pub dated_fold: Vec<usize>,
}

/// Whether a memory's stated end has passed: the end of `valid_until`'s
/// unit, in its source's timezone, as `Window::closes_at` has it. A stored
/// time is the start of its unit, so an end of 1 October holds through 1
/// October. A memory with no stated end hasn't ended, whatever else it has
/// (a point event in the past is past, not ended), and neither has one
/// whose recorded ending is still ahead.
pub(crate) fn has_ended(valid_until: Option<WorldTime>, tz: &TimeZone, now: Timestamp) -> bool {
    valid_until.is_some_and(|until| unit_end(until, tz) <= now)
}

struct Row {
    id: i64,
    uuid: Uuid,
    kind: Kind,
    level: f64,
    valid_from: Option<Timestamp>,
    valid_until: Option<WorldTime>,
    due_at: Option<Timestamp>,
    rrule: Option<String>,
    recurrence_start: Option<Timestamp>,
    timezone: String,
}

/// A dated item: when it falls, and the local date that places it.
struct Dated {
    row: usize,
    at: Timestamp,
    date: Date,
}

/// Builds the bank's agenda at `now`, in the bank's timezone `tz`.
pub(crate) fn build(
    conn: &Connection,
    tuning: &Tuning,
    bank_id: i64,
    tz: &TimeZone,
    now: Timestamp,
) -> Result<Built, rusqlite::Error> {
    let settings = &tuning.agenda;
    let today = now.to_zoned(tz.clone()).date();
    let horizon = today
        .checked_add(i64::from(settings.horizon_days).days())
        .unwrap_or(Date::MAX);
    let overdue_from = today
        .checked_sub(i64::from(settings.overdue_days).days())
        .unwrap_or(Date::MIN);
    let start_of_today = today
        .to_zoned(tz.clone())
        .map(|zoned| zoned.timestamp())
        .unwrap_or(now);
    let local = |at: Timestamp| at.to_zoned(tz.clone()).date();

    let rows = rows(conn, bank_id, &tuning.strength.significance)?;
    let mut dated: Vec<Dated> = Vec::new();
    let mut routines: Vec<usize> = Vec::new();
    let mut undated: Vec<usize> = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let source_tz = TimeZone::get(&row.timezone).unwrap_or(TimeZone::UTC);
        if has_ended(row.valid_until, &source_tz, now) {
            continue;
        }
        match row.kind {
            Kind::Event => {
                if let Some(from) = row.valid_from {
                    let date = local(from);
                    if today <= date && date <= horizon {
                        dated.push(Dated {
                            row: index,
                            at: from,
                            date,
                        });
                    }
                }
            }
            Kind::Task => match row.due_at {
                Some(due) => {
                    let date = local(due);
                    if overdue_from <= date && date <= horizon {
                        dated.push(Dated {
                            row: index,
                            at: due,
                            date,
                        });
                    }
                }
                None => undated.push(index),
            },
            Kind::Recurring => {
                let long = row
                    .rrule
                    .as_deref()
                    .and_then(period_within_a_week)
                    .is_some_and(|short| !short);
                if !long {
                    routines.push(index);
                    continue;
                }
                if let Some(next) = next_occurrence(row, start_of_today) {
                    let date = local(next);
                    if date <= horizon {
                        dated.push(Dated {
                            row: index,
                            at: next,
                            date,
                        });
                    }
                }
            }
            Kind::Fact | Kind::State => {}
        }
    }

    let loader = StrengthLoader::new(conn, bank_id, tuning, now)?;
    let strength =
        |row: &Row| -> Result<f64, rusqlite::Error> { Ok(loader.strength(conn, row.id)?.value) };

    dated.sort_by(|a, b| a.at.cmp(&b.at).then(rows[a.row].id.cmp(&rows[b.row].id)));
    // Fold faded items first, then the least significant, then the furthest
    // from today.
    let mut order: Vec<(bool, f64, i64, usize)> = Vec::with_capacity(dated.len());
    for (position, item) in dated.iter().enumerate() {
        let row = &rows[item.row];
        let distance = i64::from((item.date - today).get_days().abs());
        order.push((strength(row)? >= TAU, row.level, -distance, position));
    }
    order.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.total_cmp(&b.1))
            .then(a.2.cmp(&b.2))
            .then(b.3.cmp(&a.3))
    });
    let order: Vec<usize> = order.into_iter().map(|item| item.3).collect();
    let excess = dated.len().saturating_sub(settings.dated_lines as usize);
    let folded = excess;
    let mut fold: Vec<usize> = order[..excess].to_vec();
    fold.sort_unstable();
    for position in fold.iter().rev() {
        dated.remove(*position);
    }
    // Where each remaining item now sits, in the order it would fold next.
    let dated_fold: Vec<usize> = order[excess..]
        .iter()
        .map(|position| position - fold.iter().filter(|folded| *folded < position).count())
        .collect();

    let ranked = |indices: Vec<usize>, cap: u32| -> Result<Vec<usize>, rusqlite::Error> {
        let mut gated = Vec::new();
        for index in indices {
            let value = strength(&rows[index])?;
            if value >= TAU {
                gated.push((index, value));
            }
        }
        gated.sort_by(|(a, a_strength), (b, b_strength)| {
            b_strength
                .total_cmp(a_strength)
                .then(rows[*b].level.total_cmp(&rows[*a].level))
                .then(rows[*a].id.cmp(&rows[*b].id))
        });
        gated.truncate(cap as usize);
        Ok(gated.into_iter().map(|(index, _)| index).collect())
    };
    let routines = ranked(routines, settings.routines)?;
    let undated = ranked(undated, settings.undated_tasks)?;

    let dated: Vec<usize> = dated.into_iter().map(|item| item.row).collect();
    let all: Vec<i64> = dated
        .iter()
        .chain(&routines)
        .chain(&undated)
        .map(|index| rows[*index].id)
        .collect();
    let keep_all = |_: &crate::retrieval::candidates::Candidate| true;
    let mut cleanup = Cleanup::new(conn, bank_id, tuning, now, &keep_all)?;
    cleanup.list(&all)?;
    let mut lines = |indices: &[usize]| -> Vec<String> {
        let ids: Vec<i64> = indices.iter().map(|index| rows[*index].id).collect();
        cleanup
            .take(&ids)
            .iter()
            .map(|candidate| format::line(candidate, now))
            .collect()
    };
    let built = Built {
        dated_fold,
        dated: lines(&dated),
        routines: lines(&routines),
        undated_tasks: lines(&undated),
        agenda: Agenda {
            dated: dated.iter().map(|index| rows[*index].uuid).collect(),
            folded,
            routines: routines.iter().map(|index| rows[*index].uuid).collect(),
            undated_tasks: undated.iter().map(|index| rows[*index].uuid).collect(),
        },
    };
    Ok(built)
}

fn rows(
    conn: &Connection,
    bank_id: i64,
    significance: &SignificanceTuning,
) -> Result<Vec<Row>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT m.id, m.uuid, m.kind, COALESCE(m.owner_significance, m.significance),
                m.valid_from, m.valid_until, m.due_at, m.recurrence_rrule, m.recurrence_start,
                s.timezone, m.valid_until_precision
         FROM memories m JOIN chunks c ON c.id = m.chunk_id JOIN sources s ON s.id = c.source_id
         WHERE m.bank_id = ?1 AND m.kind IN ('event', 'task', 'recurring')
           AND m.invalidated_at IS NULL AND m.hidden_at IS NULL
           AND m.superseded_by IS NULL
         ORDER BY m.id",
    )?;
    statement
        .query_map([bank_id], |row| {
            let uuid: String = row.get(1)?;
            let kind: String = row.get(2)?;
            let level: String = row.get(3)?;
            Ok(Row {
                id: row.get(0)?,
                uuid: uuid.parse().unwrap_or_default(),
                kind: memory_kind(&kind).unwrap_or(Kind::Fact),
                level: significance_value(&level, significance),
                valid_from: row.get::<_, Option<i64>>(4)?.map(timestamp),
                valid_until: world_time(row.get(5)?, row.get(10)?),
                due_at: row.get::<_, Option<i64>>(6)?.map(timestamp),
                rrule: row.get(7)?,
                recurrence_start: row.get::<_, Option<i64>>(8)?.map(timestamp),
                timezone: row.get(9)?,
            })
        })?
        .collect()
}

/// Whether an RRULE repeats at least weekly; `None` when its frequency
/// can't be read, which makes it a routine like one with no RRULE.
pub(crate) fn period_within_a_week(rule: &str) -> Option<bool> {
    let rule = rule.trim();
    let rule = rule.strip_prefix("RRULE:").unwrap_or(rule);
    let mut freq = None;
    let mut interval: u32 = 1;
    for part in rule.split(';') {
        let (key, value) = part.split_once('=')?;
        match key.trim().to_ascii_uppercase().as_str() {
            "FREQ" => freq = Some(value.trim().to_ascii_uppercase()),
            "INTERVAL" => interval = value.trim().parse().ok()?,
            _ => {}
        }
    }
    match freq?.as_str() {
        "SECONDLY" | "MINUTELY" | "HOURLY" => Some(true),
        "DAILY" => Some(interval <= 7),
        "WEEKLY" => Some(interval <= 1),
        "MONTHLY" | "YEARLY" => Some(false),
        _ => None,
    }
}

/// The first occurrence of a recurring memory at or after `from`, in its
/// source's timezone, as extraction checked it.
fn next_occurrence(row: &Row, from: Timestamp) -> Option<Timestamp> {
    let rule = row.rrule.as_deref()?.trim();
    let rule = rule.strip_prefix("RRULE:").unwrap_or(rule);
    if rule.is_empty() || rule.contains(['\n', '\r', ':']) {
        return None;
    }
    let tz = TimeZone::get(&row.timezone).ok()?;
    let start = row.recurrence_start?.to_zoned(tz).datetime();
    let text = format!(
        "DTSTART;TZID={}:{}\nRRULE:{rule}",
        row.timezone,
        start.strftime("%Y%m%dT%H%M%S")
    );
    let set = text.parse::<rrule::RRuleSet>().ok()?;
    // A second early, so an occurrence at midnight counts whether `after`
    // is inclusive or not.
    let after = rrule::Tz::UTC
        .timestamp_opt(from.as_second() - 1, 0)
        .single()?;
    let next = set.after(after).all(1).dates.into_iter().next()?;
    Timestamp::from_second(next.timestamp()).ok()
}
