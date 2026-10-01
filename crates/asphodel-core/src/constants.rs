//! Settings fixed in code (ADR 0009).
//!
//! Every constant the strength model was calibrated with lives here, apart
//! from the quiet-time rate, which is [`Tuning`](crate::config::Tuning). None
//! of these ever appears in a config struct: strength is never stored, so
//! editing one changes every memory at once. Changing a value here is a code
//! change and an ADR.
//!
//! Durations on bank time are in bank days, and durations on world time are
//! in world days, matching the formula in "Strength model: decay,
//! reinforcement and significance" (TIM-91).

use std::time::Duration;

use serde::{Deserialize, Serialize};

// Strength (TIM-91):
//
//   strength      = S·significance + max(recent_use, lasting_floor)
//   recent_use    = ln Σ w_j · age_j^(−d_j)
//   d_j           = min(D_MAX, a + c·e^(m_{j−1}))
//   lasting_floor = τ − g·ln(n0) + g·ln(n)

/// S: how much significance lifts strength.
pub const S: f64 = 2.5;

/// τ: the recall threshold. It is also the boundary between faded and
/// fading, by definition.
pub const TAU: f64 = -0.7;

/// a: the base fading rate of each access (Pavlik & Anderson).
pub const A: f64 = 0.35;

/// c: how much an access made while the memory is fresh fades faster.
pub const C: f64 = 0.2;

/// The cap on an access's fading rate d.
pub const D_MAX: f64 = 2.0;

/// g: how much each separate occasion raises the lasting floor.
pub const G: f64 = 0.8;

/// n0: separate occasions that take a memory at significance 0 to
/// permanence. It places the lasting floor relative to τ.
pub const N0: f64 = 18.0;

/// The youngest age an access counts at, in bank days, so a fresh access
/// doesn't divide by zero.
pub const MIN_ACCESS_AGE_DAYS: f64 = 0.01;

/// The floor spacing: accesses count as separate occasions only when they
/// are at least this many world days apart (ADR 0003).
pub const FLOOR_SPACING_DAYS: f64 = 3.0;

/// The weight of a `created` access in recent use.
pub const WEIGHT_CREATED: f64 = 1.0;

/// The weight of a `used` access in recent use. It's the lowest because it's
/// the only kind the injection loop can generate.
pub const WEIGHT_USED: f64 = 1.0;

/// The weight of a `mentioned_again` access in recent use.
pub const WEIGHT_MENTIONED_AGAIN: f64 = 1.5;

/// The weight of a `confirmed` access in recent use.
pub const WEIGHT_CONFIRMED: f64 = 2.0;

/// The weight of the synthetic access that restarts recent use when a
/// validity window closes.
pub const WEIGHT_WINDOW_CLOSE: f64 = 1.0;

/// Bank time runs at full speed for this long after any turn in the bank,
/// and at the tuned quiet-time rate otherwise (ADR 0004).
pub const FULL_SPEED_WINDOW: Duration = Duration::from_secs(24 * 60 * 60);

/// World days after its window closes that a memory is recently past rather
/// than long past. It's the end of the ramp in "Retrieval and ranking"
/// (TIM-93, decision 6) after which the full phase penalty applies. Only
/// rendering and the phase label use it; ranking works from the days since
/// the close.
pub const RECENTLY_PAST_DAYS: f64 = 30.0;

/// A kept memory's significance. It never fades.
pub const SIGNIFICANCE_KEPT: f64 = 1.0;

/// How much a memory matters on its own terms, judged at extraction. Each
/// level maps to a fixed significance value; kept is
/// [`SIGNIFICANCE_KEPT`], above every level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Significance {
    Trivial,
    Minor,
    Notable,
    Major,
    Critical,
}

impl Significance {
    /// Every level, lowest first.
    pub const ALL: [Significance; 5] = [
        Significance::Trivial,
        Significance::Minor,
        Significance::Notable,
        Significance::Major,
        Significance::Critical,
    ];

    /// The significance value this level stands for.
    pub const fn value(self) -> f64 {
        match self {
            Significance::Trivial => 0.1,
            Significance::Minor => 0.3,
            Significance::Notable => 0.5,
            Significance::Major => 0.7,
            Significance::Critical => 0.9,
        }
    }
}

/// How quickly a state is expected to go stale.
///
/// State confidence is `1 / (1 + (age / T)²)` on world time, where T is
/// [`Volatility::rate_days`]. Confidence only lowers ranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Volatility {
    Hours,
    Days,
    Weeks,
    Months,
    Years,
}

impl Volatility {
    /// Every level, fastest first.
    pub const ALL: [Volatility; 5] = [
        Volatility::Hours,
        Volatility::Days,
        Volatility::Weeks,
        Volatility::Months,
        Volatility::Years,
    ];

    /// T, the age in world days at which confidence is a coin flip: 3 hours,
    /// 3 days, 3 weeks, 3 months or 5 years. Months and years are Gregorian
    /// averages.
    pub const fn rate_days(self) -> f64 {
        match self {
            Volatility::Hours => 3.0 / 24.0,
            Volatility::Days => 3.0,
            Volatility::Weeks => 21.0,
            Volatility::Months => 3.0 * 365.2425 / 12.0,
            Volatility::Years => 5.0 * 365.2425,
        }
    }
}

/// The reranker deadline in prefetch. Past it, the reranker is skipped and
/// RRF order is used. It's the first link in a chain of timeouts: 1.5 s in
/// the daemon, 3 s in the plugin and 8 s in Hermes.
pub const RERANKER_DEADLINE: Duration = Duration::from_millis(1500);

// Extraction (TIM-92, TIM-98):

/// The size a document section is split down to when it's too long, in
/// characters ("at about 3,000 characters").
pub const CHUNK_CHARS: usize = 3_000;

/// How many failed attempts mark a chunk `failed`. A failed chunk leaves the
/// queue and is surfaced by `asphodel chunks --failed` instead of retried.
pub const CHUNK_RETRY_CAP: u32 = 5;

/// The snapshot of the fixed constants that `GET /v1/config` shows as a
/// read-only section.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FixedConstants {
    pub s: f64,
    pub tau: f64,
    pub a: f64,
    pub c: f64,
    pub d_max: f64,
    pub g: f64,
    pub n0: f64,
    pub min_access_age_days: f64,
    pub floor_spacing_days: f64,
    pub access_weights: AccessWeights,
    pub window_close_weight: f64,
    pub full_speed_window_hours: f64,
    pub recently_past_days: f64,
    pub significance: SignificanceValues,
    pub volatility_rate_days: VolatilityRates,
    pub reranker_deadline_ms: u128,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AccessWeights {
    pub created: f64,
    pub used: f64,
    pub mentioned_again: f64,
    pub confirmed: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SignificanceValues {
    pub trivial: f64,
    pub minor: f64,
    pub notable: f64,
    pub major: f64,
    pub critical: f64,
    pub kept: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VolatilityRates {
    pub hours: f64,
    pub days: f64,
    pub weeks: f64,
    pub months: f64,
    pub years: f64,
}

impl FixedConstants {
    /// The constants this build was compiled with.
    pub fn current() -> Self {
        Self {
            s: S,
            tau: TAU,
            a: A,
            c: C,
            d_max: D_MAX,
            g: G,
            n0: N0,
            min_access_age_days: MIN_ACCESS_AGE_DAYS,
            floor_spacing_days: FLOOR_SPACING_DAYS,
            access_weights: AccessWeights {
                created: WEIGHT_CREATED,
                used: WEIGHT_USED,
                mentioned_again: WEIGHT_MENTIONED_AGAIN,
                confirmed: WEIGHT_CONFIRMED,
            },
            window_close_weight: WEIGHT_WINDOW_CLOSE,
            full_speed_window_hours: FULL_SPEED_WINDOW.as_secs_f64() / 3600.0,
            recently_past_days: RECENTLY_PAST_DAYS,
            significance: SignificanceValues {
                trivial: Significance::Trivial.value(),
                minor: Significance::Minor.value(),
                notable: Significance::Notable.value(),
                major: Significance::Major.value(),
                critical: Significance::Critical.value(),
                kept: SIGNIFICANCE_KEPT,
            },
            volatility_rate_days: VolatilityRates {
                hours: Volatility::Hours.rate_days(),
                days: Volatility::Days.rate_days(),
                weeks: Volatility::Weeks.rate_days(),
                months: Volatility::Months.rate_days(),
                years: Volatility::Years.rate_days(),
            },
            reranker_deadline_ms: RERANKER_DEADLINE.as_millis(),
        }
    }
}
