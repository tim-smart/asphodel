//! The strength model, phase and state confidence as pure functions.
//!
//! Everything here is computed from stored facts and a `now` the caller
//! passes in. Nothing reads a clock or the store, and nothing computed here is
//! ever stored:
//!
//! ```text
//! strength      = S·significance + max(recent_use, lasting_floor)
//! recent_use    = ln Σ_j w_j · age_j^(−d_j)        age in bank days, min 0.01
//! d_j           = min(D_MAX, a + c·e^(m_{j−1}))     m = recent_use just before access j
//! lasting_floor = τ − g·ln(n0) + g·ln(n)            n = occasions ≥ 3 world days apart
//! ```
//!
//! Strength runs on bank time ([`BankTime`]); windows, phase, due dates and
//! state confidence run on world time in the source's timezone.

mod bank_time;
mod chain;
mod confidence;
mod purge;
mod window;

use jiff::Timestamp;

use crate::config::AccessWeightsTuning;
use crate::constants::{
    A, C, D_MAX, FLOOR_SPACING_DAYS, G, MIN_ACCESS_AGE_DAYS, N0, S, TAU, WEIGHT_CONFIRMED,
    WEIGHT_CREATED, WEIGHT_MENTIONED_AGAIN, WEIGHT_WINDOW_CLOSE,
};

pub use crate::constants::RECENTLY_PAST_DAYS;
pub use bank_time::{BankTime, FULL_SPEED_HORIZON_DAYS};
pub use chain::{Chains, Link, chain, chain_head, inherits_from};
pub use confidence::state_confidence;
pub use purge::{PurgeRule, never_purged, purge_eligible};
pub(crate) use window::task_due_is_eligible;
pub use window::{Kind, Phase, TimePrecision, Window, WorldTime, unit_end};

const MICROS_PER_DAY: f64 = 86_400_000_000.0;

/// World days from `from` to `to`, negative when `to` is earlier.
fn world_days(from: Timestamp, to: Timestamp) -> f64 {
    (to.as_microsecond() - from.as_microsecond()) as f64 / MICROS_PER_DAY
}

/// The four kinds of access, the only things that strengthen a memory:
/// being returned by a search, injected or viewed never does. The kind sets
/// an access's weight in recent use; the lasting floor ignores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessKind {
    Created,
    Used,
    MentionedAgain,
    Confirmed,
}

impl AccessKind {
    /// The weight of this kind in recent use: `used` is tuned, the rest are
    /// fixed.
    pub const fn weight(self, weights: &AccessWeightsTuning) -> f64 {
        match self {
            AccessKind::Created => WEIGHT_CREATED,
            AccessKind::Used => weights.used,
            AccessKind::MentionedAgain => WEIGHT_MENTIONED_AGAIN,
            AccessKind::Confirmed => WEIGHT_CONFIRMED,
        }
    }
}

/// One row of the access log: its kind, when it happened, in world time,
/// and its kind's weight in recent use, set when the row is loaded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Access {
    pub kind: AccessKind,
    pub at: Timestamp,
    pub weight: f64,
}

/// A validity window's close, which restarts recent use. The lasting floor
/// still counts every access.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowClose {
    /// When the window closed: [`Window::closes_at`], so already the end of
    /// the closing time's unit in the source's timezone.
    pub closes_at: Timestamp,

    /// When the end became known: the `observed_at` of the `ended_by` memory
    /// for a late-reported end, or the memory's own `observed_at` when the
    /// end was stated from the start.
    pub known_at: Timestamp,
}

/// A memory's strength and its parts at one instant. Never stored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Strength {
    /// `S·significance + max(recent_use, lasting_floor)`.
    pub value: f64,
    pub recent_use: f64,
    pub lasting_floor: f64,
    /// `S·significance + lasting_floor`: what strength falls to without
    /// use, and never below.
    pub lasting: f64,
    /// n: the accesses counted as separate occasions for the floor.
    pub occasions: u32,
}

/// A memory's strength at `now`.
///
/// - `significance` is the value, not the level, so kept is
///   [`SIGNIFICANCE_KEPT`](crate::constants::SIGNIFICANCE_KEPT).
/// - `accesses` is the memory's own log plus everything it inherits along
///   `superseded_by` (see [`inherits_from`]), in any order. Accesses after
///   `now` are ignored.
/// - Recent use ages each access in bank days from `bank_time`, never younger
///   than `MIN_ACCESS_AGE_DAYS`. An access made while the memory is already
///   fresh fades faster: its `d` grows with the recent use just before it
///   (Pavlik & Anderson), capped at `D_MAX`. The first has `d = a`.
///   Weighted accesses are ordered by time ascending, then weight descending
///   (`f64::total_cmp`), including the synthetic access. Heaviest-first ties
///   keep the result independent of input order without letting lighter tied
///   accesses increase the heaviest access's decay. Equal-time, equal-weight
///   entries are interchangeable.
/// - Once `close` is given and `max(closes_at, known_at)` has passed, recent
///   use starts over. It counts one synthetic access of weight
///   `WEIGHT_WINDOW_CLOSE` at that instant, plus the accesses strictly after
///   `closes_at`.
/// - The lasting floor counts every access, closed window or not. In time
///   order, an access is a new occasion when it's at least
///   `FLOOR_SPACING_DAYS` world days after the last occasion. The synthetic
///   access is a recency boost, not an access, and is never an occasion.
///
/// Without accesses, recent use, the floor and the value are all −∞.
pub fn strength(
    significance: f64,
    accesses: &[Access],
    close: Option<WindowClose>,
    bank_time: &BankTime,
    now: Timestamp,
) -> Strength {
    let mut accesses: Vec<Access> = accesses.iter().filter(|a| a.at <= now).copied().collect();
    accesses.sort_by_key(|a| a.at);

    let occasions = occasions(&accesses);
    let lasting_floor = if occasions == 0 {
        f64::NEG_INFINITY
    } else {
        TAU - G * N0.ln() + G * f64::from(occasions).ln()
    };

    let restart = close
        .map(|c| (c.closes_at, c.closes_at.max(c.known_at)))
        .filter(|&(_, restart)| restart <= now);
    let mut counted: Vec<(f64, Timestamp)> = match restart {
        None => accesses.iter().map(|a| (a.weight, a.at)).collect(),
        Some((closes_at, restart)) => {
            let mut counted: Vec<_> = accesses
                .iter()
                .filter(|a| a.at > closes_at)
                .map(|a| (a.weight, a.at))
                .collect();
            counted.push((WEIGHT_WINDOW_CLOSE, restart));
            counted
        }
    };
    counted.sort_by(|(left_weight, left_at), (right_weight, right_at)| {
        left_at
            .cmp(right_at)
            .then_with(|| right_weight.total_cmp(left_weight))
    });
    let recent_use = recent_use(&counted, bank_time, now);

    Strength {
        value: S * significance + recent_use.max(lasting_floor),
        recent_use,
        lasting_floor,
        lasting: S * significance + lasting_floor,
        occasions,
    }
}

/// Separate occasions among accesses sorted by time.
fn occasions(sorted: &[Access]) -> u32 {
    let mut count = 0;
    let mut last: Option<Timestamp> = None;
    for access in sorted {
        if last.is_none_or(|last| world_days(last, access.at) >= FLOOR_SPACING_DAYS) {
            count += 1;
            last = Some(access.at);
        }
    }
    count
}

/// `ln Σ w_j · age_j^(−d_j)` over weighted accesses sorted by time ascending,
/// then weight descending.
///
/// Each `d_j` needs the recent use at the time of access j. Sum newest-first
/// and stop once the decay cap is reached: all remaining terms are positive
/// and cannot change `d_j`. Frequently used memories reach the cap quickly;
/// cold histories still take quadratic work in the worst case.
fn recent_use(sorted: &[(f64, Timestamp)], bank_time: &BankTime, now: Timestamp) -> f64 {
    let age =
        |from: Timestamp, to: Timestamp| bank_time.elapsed_days(from, to).max(MIN_ACCESS_AGE_DAYS);
    let mut decays: Vec<f64> = Vec::with_capacity(sorted.len());
    for (j, &(_, at)) in sorted.iter().enumerate() {
        // e^m, where m is recent use just before this access: the sum
        // itself. It's 0 before the first access, so that one's d is a.
        let mut before = 0.0;
        for (&(weight, earlier), &d) in sorted[..j].iter().zip(&decays).rev() {
            before += weight * age(earlier, at).powf(-d);
            if before >= (D_MAX - A) / C {
                break;
            }
        }
        decays.push((A + C * before).min(D_MAX));
    }
    sorted
        .iter()
        .zip(&decays)
        .map(|(&(weight, at), &d)| weight * age(at, now).powf(-d))
        .sum::<f64>()
        .ln()
}

/// When strength first falls below `threshold` if nothing uses the memory
/// again and bank time runs at full speed from `now`: the bank days from
/// `now`, which at full speed are also world days, so `now` plus them is
/// the earliest world time it can happen. `Some(0.0)` when it's already
/// below, `None` when it never falls below, as for a kept memory or one
/// whose lasting floor holds it up.
///
/// Recent use only falls without accesses, so strength falls too, except
/// at a window's close still ahead, where recent use starts over: the
/// stretch before it is searched first.
pub fn projected_below(
    significance: f64,
    accesses: &[Access],
    close: Option<WindowClose>,
    bank_time: &BankTime,
    now: Timestamp,
    threshold: f64,
) -> Option<f64> {
    projected_below_after(
        significance,
        accesses,
        close,
        bank_time,
        now,
        now,
        threshold,
    )
}

/// [`projected_below`], but the first time at or after `after` that
/// strength is below `threshold`: purge's projection, which can't come
/// before its guards clear, and must still find strength below the line
/// then, after any window close that restarted recent use. Still in bank
/// days from `now`.
pub fn projected_below_after(
    significance: f64,
    accesses: &[Access],
    close: Option<WindowClose>,
    bank_time: &BankTime,
    now: Timestamp,
    after: Timestamp,
    threshold: f64,
) -> Option<f64> {
    let clock = bank_time.at_full_speed_from(now);
    let at = |days: f64| {
        now.checked_add(jiff::SignedDuration::from_secs_f64(days * 86_400.0))
            .unwrap_or(Timestamp::MAX)
    };
    let value = |days: f64| strength(significance, accesses, close, &clock, at(days)).value;
    let below = |days: f64| value(days) < threshold;
    let start = world_days(now, after).max(0.0);
    if below(start) {
        return Some(start);
    }
    // Precise to about a minute.
    const PRECISION: f64 = 1.0 / 1440.0;
    let search = |mut low: f64, mut high: f64| {
        while high - low > PRECISION {
            let middle = (low + high) / 2.0;
            if below(middle) {
                high = middle;
            } else {
                low = middle;
            }
        }
        high
    };
    let mut from = start;
    if let Some(close) = close {
        let restart = close.closes_at.max(close.known_at);
        let until = world_days(now, restart);
        if until > start {
            let before = (until - PRECISION).max(start);
            if below(before) {
                return Some(search(start, before));
            }
            from = until;
        }
    }
    if S * significance + lasting_floor_of(accesses) >= threshold {
        return None;
    }
    let mut high = from.max(1.0);
    while !below(high) {
        if high >= FULL_SPEED_HORIZON_DAYS {
            return None;
        }
        high = (high * 2.0).min(FULL_SPEED_HORIZON_DAYS);
    }
    Some(search(from, high))
}

/// The lasting floor of `accesses`, which never falls.
fn lasting_floor_of(accesses: &[Access]) -> f64 {
    let mut sorted = accesses.to_vec();
    sorted.sort_by_key(|a| a.at);
    let occasions = occasions(&sorted);
    if occasions == 0 {
        f64::NEG_INFINITY
    } else {
        TAU - G * N0.ln() + G * f64::from(occasions).ln()
    }
}
