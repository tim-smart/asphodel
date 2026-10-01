//! When refreshes run (TIM-95 amendment, decision 1; ADR 0007).
//!
//! - A triggering write marks the models it concerns with
//!   `refresh_requested_at`, kept from the first trigger since their last
//!   refresh. The debounce is per bank: a requested model is due
//!   `refresh_debounce_minutes` after the bank's last trigger, and never
//!   later than `refresh_max_delay_minutes` after its first.
//! - A model is refreshed at most every [`MIN_REFRESH_INTERVAL`], and a
//!   failed refresh waits as long before it's tried again.
//! - Every bank has a sweep at `sweep_time` bank-local each day, which
//!   checks every enabled model; the fingerprint skip keeps it cheap.
//!
//! The bank's last trigger and last sweep live in memory: after a restart
//! the first trigger stands in for the last, and the next sweep is the next
//! `sweep_time` after the daemon started. Everything runs on the service's
//! clock.

use std::collections::HashMap;
use std::sync::Mutex;

use jiff::civil::Time;
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};

use super::{MIN_REFRESH_INTERVAL, ModelRow};
use crate::config::MentalModelsTuning;

/// Each bank's last trigger and last sweep.
#[derive(Debug)]
pub(crate) struct Schedule {
    started: Timestamp,
    banks: Mutex<HashMap<i64, Bank>>,
}

#[derive(Debug, Clone, Copy)]
struct Bank {
    last_trigger: Option<Timestamp>,
    last_sweep: Timestamp,
}

impl Schedule {
    /// A schedule for a service started at `started`.
    pub(crate) fn new(started: Timestamp) -> Self {
        Self {
            started,
            banks: Mutex::default(),
        }
    }

    fn with<T>(&self, bank_id: i64, f: impl FnOnce(&mut Bank) -> T) -> T {
        let mut banks = self.banks.lock().unwrap_or_else(|e| e.into_inner());
        let bank = banks.entry(bank_id).or_insert(Bank {
            last_trigger: None,
            last_sweep: self.started,
        });
        f(bank)
    }

    /// A triggering write landed in the bank at `now`.
    pub(crate) fn triggered(&self, bank_id: i64, now: Timestamp) {
        self.with(bank_id, |bank| {
            bank.last_trigger = Some(bank.last_trigger.map_or(now, |last| last.max(now)));
        });
    }

    pub(crate) fn last_trigger(&self, bank_id: i64) -> Option<Timestamp> {
        self.with(bank_id, |bank| bank.last_trigger)
    }

    /// When the bank's next sweep is due.
    pub(crate) fn next_sweep(&self, bank_id: i64, tz: &TimeZone, at: Time) -> Timestamp {
        let last = self.with(bank_id, |bank| bank.last_sweep);
        next_local(last, tz, at)
    }

    /// The bank's sweep ran at `now`.
    pub(crate) fn swept(&self, bank_id: i64, now: Timestamp) {
        self.with(bank_id, |bank| bank.last_sweep = now);
    }
}

/// The first instant after `after` whose local time in `tz` is `at`. A
/// time skipped by a daylight-saving change runs at the instant after the
/// gap.
pub(crate) fn next_local(after: Timestamp, tz: &TimeZone, at: Time) -> Timestamp {
    let mut date = after.to_zoned(tz.clone()).date();
    for _ in 0..3 {
        if let Ok(zoned) = date.to_datetime(at).to_zoned(tz.clone())
            && zoned.timestamp() > after
        {
            return zoned.timestamp();
        }
        match date.tomorrow() {
            Ok(next) => date = next,
            Err(_) => break,
        }
    }
    Timestamp::MAX
}

/// The earliest a refresh of `model` may run: [`MIN_REFRESH_INTERVAL`]
/// after its last refresh, and as long after a failure still standing.
pub(crate) fn not_before(model: &ModelRow) -> Option<Timestamp> {
    let after = |at: Option<Timestamp>| at.and_then(|at| at.checked_add(MIN_REFRESH_INTERVAL).ok());
    let refreshed = after(model.last_refreshed_at);
    let failed = if model.last_error.is_some() {
        after(model.last_error_at)
    } else {
        None
    };
    refreshed.max(failed)
}

/// When a requested refresh of `model` is due, given the bank's last
/// trigger; `None` when nothing has requested one.
pub(crate) fn due(
    model: &ModelRow,
    last_trigger: Option<Timestamp>,
    tuning: &MentalModelsTuning,
) -> Option<Timestamp> {
    let first = model.refresh_requested_at?;
    let last = last_trigger.filter(|last| *last >= first).unwrap_or(first);
    let debounce = SignedDuration::from_mins(i64::from(tuning.refresh_debounce_minutes));
    let cap = SignedDuration::from_mins(i64::from(tuning.refresh_max_delay_minutes));
    let quiet = last.checked_add(debounce).unwrap_or(Timestamp::MAX);
    let capped = first.checked_add(cap).unwrap_or(Timestamp::MAX);
    let debounced = quiet.min(capped);
    Some(match not_before(model) {
        Some(floor) => debounced.max(floor),
        None => debounced,
    })
}
