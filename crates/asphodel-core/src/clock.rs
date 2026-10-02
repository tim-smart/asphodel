//! The clock the service layer runs on.
//!
//! Memory runs on bank time and truth on world time (ADR 0004), and both are
//! derived from one source of "now" that the caller hands in. `serve` passes
//! [`SystemClock`]; `replay` passes a [`SimulatedClock`] and advances it by
//! hand. No other code reads the system time or asks the database for it.
//! `clippy.toml` denies the direct calls workspace-wide.

use std::fmt;
use std::sync::Mutex;

use jiff::{SignedDuration, Timestamp};

/// A source of the current world time.
///
/// Implementations must be cheap to call and safe to share across threads,
/// because every request and every sweep asks for the time.
pub trait Clock: Send + Sync {
    /// The current instant in world time.
    fn now(&self) -> Timestamp;
}

/// The clock `asphodel serve` runs on. It is the only place in the workspace
/// allowed to read the system time.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    #[allow(clippy::disallowed_methods)]
    fn now(&self) -> Timestamp {
        Timestamp::now()
    }
}

/// A clock that only moves when told to.
///
/// The replay harness drives it from its event queue, so extraction
/// completions, sweeps and idle timeouts all happen at the simulated time
/// their events were scheduled for. Advancing it never blocks.
pub struct SimulatedClock {
    now: Mutex<Timestamp>,
}

impl SimulatedClock {
    /// A clock stopped at `start`.
    pub fn new(start: Timestamp) -> Self {
        Self {
            now: Mutex::new(start),
        }
    }

    /// Moves the clock forward by `by`. Panics if `by` is negative, because
    /// replay is a discrete-event simulation and events never run backwards.
    pub fn advance(&self, by: SignedDuration) {
        assert!(
            !by.is_negative(),
            "a simulated clock only moves forward, got {by:?}"
        );
        let mut now = self.now.lock().unwrap_or_else(|e| e.into_inner());
        *now = now
            .checked_add(by)
            .expect("advancing the simulated clock overflowed the timestamp range");
    }

    /// Sets the clock to `to`. Panics if `to` is earlier than the current time.
    pub fn set(&self, to: Timestamp) {
        let mut now = self.now.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            to >= *now,
            "a simulated clock only moves forward: tried to set {to} while at {now}"
        );
        *now = to;
    }
}

impl Clock for SimulatedClock {
    fn now(&self) -> Timestamp {
        *self.now.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl fmt::Debug for SimulatedClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimulatedClock")
            .field("now", &self.now())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;

    use super::*;

    fn start() -> Timestamp {
        "2026-01-01T00:00:00.123456789Z".parse().unwrap()
    }

    #[test]
    fn advance_accumulates_exact_durations() {
        let clock = SimulatedClock::new(start());
        clock.advance(SignedDuration::from_secs(60));
        assert_eq!(
            clock.now(),
            "2026-01-01T00:01:00.123456789Z".parse().unwrap()
        );

        clock.advance(SignedDuration::from_nanos(1));
        assert_eq!(
            clock.now(),
            "2026-01-01T00:01:00.123456790Z".parse().unwrap()
        );
    }

    #[test]
    fn set_moves_forward_and_advance_uses_the_new_time() {
        let clock = SimulatedClock::new(start());
        let later = "2026-02-01T12:00:00Z".parse().unwrap();
        clock.set(later);
        assert_eq!(clock.now(), later);

        clock.advance(SignedDuration::from_secs(1));
        assert_eq!(clock.now(), "2026-02-01T12:00:01Z".parse().unwrap());
    }

    #[test]
    fn zero_advance_and_equal_set_leave_time_unchanged() {
        let clock = SimulatedClock::new(start());
        clock.advance(SignedDuration::ZERO);
        assert_eq!(clock.now(), start());
        clock.set(start());
        assert_eq!(clock.now(), start());
    }

    #[test]
    fn negative_advance_panics_without_changing_time() {
        let clock = SimulatedClock::new(start());
        assert!(catch_unwind(|| clock.advance(SignedDuration::from_nanos(-1))).is_err());
        assert_eq!(clock.now(), start());

        clock.advance(SignedDuration::from_secs(1));
        assert_eq!(
            clock.now(),
            "2026-01-01T00:00:01.123456789Z".parse().unwrap()
        );
    }

    #[test]
    fn backward_set_panics_without_changing_time() {
        let clock = SimulatedClock::new(start());
        let earlier = "2026-01-01T00:00:00.123456788Z".parse().unwrap();
        assert!(catch_unwind(|| clock.set(earlier)).is_err());
        assert_eq!(clock.now(), start());

        let later = "2026-01-01T00:00:00.123456790Z".parse().unwrap();
        clock.set(later);
        assert_eq!(clock.now(), later);
        clock.advance(SignedDuration::from_nanos(1));
        assert_eq!(
            clock.now(),
            "2026-01-01T00:00:00.123456791Z".parse().unwrap()
        );
    }
}
