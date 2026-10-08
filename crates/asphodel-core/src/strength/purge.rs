//! Purge eligibility: a chain is purged once its head's strength falls δ
//! (`purge.delta`, 0.15 by default) below the recall threshold τ, unless
//! its lasting strength holds it at or above `τ − NEVER_PURGED_MARGIN`.

use jiff::Timestamp;
use jiff::tz::TimeZone;

use super::Strength;
use super::window::{Window, unit_end};
use crate::config::Tuning;
use crate::constants::{NEVER_PURGED_MARGIN, TAU};

/// The settings purge reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PurgeRule {
    /// δ: purge once strength falls this far below τ. `None` never purges.
    pub delta: Option<f64>,

    /// The task guard, `agenda.overdue_days`: see [`Window::overdue_until`].
    pub overdue_days: u32,
}

impl PurgeRule {
    pub fn from_tuning(tuning: &Tuning) -> Self {
        Self {
            delta: tuning.purge.delta,
            overdue_days: tuning.agenda.overdue_days,
        }
    }
}

/// Whether `lasting` strength keeps a memory from purge for good.
pub fn never_purged(lasting: f64) -> bool {
    lasting >= TAU - NEVER_PURGED_MARGIN
}

/// Whether a chain may be purged at `now`, read on its head: the head's
/// strength (which includes what it inherited) and the head's window. A
/// retracted predecessor's dates hold nothing back.
///
/// Purged when the head's strength is below `τ − δ`, unless:
///
/// - δ is `None`;
/// - the head is [`never_purged`];
/// - the head's `valid_from` or `valid_until` unit hasn't ended yet, so a
///   day-precision appointment is held until that day ends in the source's
///   timezone;
/// - the head is a task before [`Window::overdue_until`], capped at the
///   end of its closing unit for a completed task.
///
/// Recurring memories and undated open tasks have no guard.
pub fn purge_eligible(
    rule: &PurgeRule,
    head_strength: &Strength,
    head: &Window,
    tz: &TimeZone,
    now: Timestamp,
) -> bool {
    let Some(delta) = rule.delta else {
        return false;
    };
    let date_ahead = [head.valid_from, head.valid_until]
        .into_iter()
        .flatten()
        .any(|time| unit_end(time, tz) > now);
    let overdue_held = head
        .overdue_until(tz, rule.overdue_days)
        .is_some_and(|until| now < until);
    !date_ahead
        && !overdue_held
        && !never_purged(head_strength.lasting)
        && head_strength.value < TAU - delta
}
