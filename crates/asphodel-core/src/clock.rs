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
