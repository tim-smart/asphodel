//! State confidence.

use jiff::Timestamp;

use super::{Access, AccessKind, world_days};
use crate::constants::Volatility;

/// How likely a state still holds: `1 / (1 + (age / T)²)`, a log-logistic
/// curve that's flat while the state is young, a coin flip at T, and falls
/// as 1/t² after. T is [`Volatility::rate_days`].
///
/// Age is in world days from the later of `observed_at` and the last
/// `mentioned_again` or `confirmed` access at or before `now`, and never
/// negative. `used` never resets it, since the assistant repeating a state is
/// no evidence it still holds. 1.0 when `volatility` is `None`.
///
/// Confidence only lowers ranking; it never gates anything.
pub fn state_confidence(
    volatility: Option<Volatility>,
    observed_at: Timestamp,
    accesses: &[Access],
    now: Timestamp,
) -> f64 {
    let Some(volatility) = volatility else {
        return 1.0;
    };
    let anchor = accesses
        .iter()
        .filter(|a| a.at <= now)
        .filter(|a| matches!(a.kind, AccessKind::MentionedAgain | AccessKind::Confirmed))
        .map(|a| a.at)
        .fold(observed_at, Timestamp::max);
    let age = world_days(anchor, now).max(0.0);
    1.0 / (1.0 + (age / volatility.rate_days()).powi(2))
}
