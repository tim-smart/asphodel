//! Bank time: the clock strength runs on.

use jiff::Timestamp;

use super::MICROS_PER_DAY;

/// How far ahead [`BankTime::at_full_speed_from`] runs at full speed: two
/// centuries, past anything a projection reports.
pub const FULL_SPEED_HORIZON_DAYS: f64 = 200.0 * 365.25;
use crate::constants::FULL_SPEED_WINDOW;

/// One bank's clock for strength, built from when its turns happened.
///
/// Bank time runs at full speed inside the union of `[turn, turn + 24 h)`
/// over every turn, and at the quiet rate everywhere else. It never runs
/// faster than world time, however many turns there are.
///
/// The turns are the bank's `sources` of kind `turn`, at `message_at`, and
/// include tombstoned ones: a swept or forgotten turn still happened. Ingested
/// documents don't count. A turn replayed from the plugin's spool after an
/// outage is placed by `message_at`, while its memories' `created` accesses
/// carry the ingest time, so they may start ageing at the quiet rate. That's
/// at most a day of bank time and isn't corrected for.
#[derive(Debug, Clone)]
pub struct BankTime {
    /// Disjoint full-speed windows in microseconds, sorted.
    windows: Vec<(i64, i64)>,
    /// `full[k]`: full-speed microseconds in `windows[..k]`.
    full: Vec<i64>,
    quiet_rate: f64,
}

impl BankTime {
    /// `turns` in any order; duplicates are harmless. `quiet_rate` is
    /// `Tuning::clock.quiet_rate`, in (0, 1].
    pub fn new(turns: &[Timestamp], quiet_rate: f64) -> Self {
        let span = FULL_SPEED_WINDOW.as_micros() as i64;
        let mut starts: Vec<i64> = turns.iter().map(|t| t.as_microsecond()).collect();
        starts.sort_unstable();

        let mut windows: Vec<(i64, i64)> = Vec::new();
        for start in starts {
            let end = start.saturating_add(span);
            match windows.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => windows.push((start, end)),
            }
        }
        let mut full = Vec::with_capacity(windows.len() + 1);
        full.push(0);
        for (start, end) in &windows {
            full.push(full.last().unwrap() + (end - start));
        }

        Self {
            windows,
            full,
            quiet_rate,
        }
    }

    /// Bank days from `from` to `to`, or 0 when `to` isn't later. Additive:
    /// `elapsed(a, b) + elapsed(b, c) == elapsed(a, c)`.
    pub fn elapsed_days(&self, from: Timestamp, to: Timestamp) -> f64 {
        let (from, to) = (from.as_microsecond(), to.as_microsecond());
        if to <= from {
            return 0.0;
        }
        let full = (self.full_before(to) - self.full_before(from)) as f64;
        let quiet = (to - from) as f64 - full;
        (full + self.quiet_rate * quiet) / MICROS_PER_DAY
    }

    /// This clock with bank time running at full speed from `from` for
    /// [`FULL_SPEED_HORIZON_DAYS`]: what it would be if a turn came at
    /// least once a day. `memory show` projects fade and purge dates on it,
    /// so a bank day after `from` is a world day, the earliest one.
    pub fn at_full_speed_from(&self, from: Timestamp) -> Self {
        let start = from.as_microsecond();
        let span = (FULL_SPEED_HORIZON_DAYS * MICROS_PER_DAY) as i64;
        let mut windows = self.windows.clone();
        windows.push((start, start.saturating_add(span)));
        windows.sort_unstable();
        let mut merged: Vec<(i64, i64)> = Vec::with_capacity(windows.len());
        for (start, end) in windows {
            match merged.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => merged.push((start, end)),
            }
        }
        let mut full = Vec::with_capacity(merged.len() + 1);
        full.push(0);
        for (start, end) in &merged {
            full.push(full.last().unwrap() + (end - start));
        }
        Self {
            windows: merged,
            full,
            quiet_rate: self.quiet_rate,
        }
    }

    /// Full-speed microseconds before `at`.
    fn full_before(&self, at: i64) -> i64 {
        let k = self.windows.partition_point(|&(start, _)| start <= at);
        let mut full = self.full[k];
        if k > 0 {
            let (_, end) = self.windows[k - 1];
            full -= (end - at).max(0);
        }
        full
    }
}
