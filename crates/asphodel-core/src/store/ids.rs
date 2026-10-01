//! Public ids: UUIDv7, timed by the store's clock (TIM-90).
//!
//! `uuid::Uuid::now_v7` reads the system clock, which ADR 0004 forbids, so
//! ids are built from a [`Timestamp`] the caller took from the [`Clock`]
//! (`clippy.toml` denies the direct call). A shared `ContextV7` keeps ids
//! minted in the same instant ordered.
//!
//! [`Clock`]: crate::clock::Clock

use std::sync::Mutex;

use jiff::Timestamp;
use uuid::{ContextV7, Uuid};

/// Mints ids. The context keeps a counter for ids minted in the same
/// instant, so it sits behind a mutex to be shared across threads.
#[derive(Debug)]
pub struct IdSource {
    context: Mutex<ContextV7>,
}

impl Default for IdSource {
    fn default() -> Self {
        Self::new()
    }
}

impl IdSource {
    pub fn new() -> Self {
        Self {
            context: Mutex::new(ContextV7::new()),
        }
    }

    /// A UUIDv7 whose time part is `now`. A time before 1970 is clamped to
    /// the epoch, since UUIDv7 has no room for it.
    pub fn next(&self, now: Timestamp) -> Uuid {
        let seconds = u64::try_from(now.as_second()).unwrap_or(0);
        let nanos = u32::try_from(now.subsec_nanosecond()).unwrap_or(0);
        let context = self
            .context
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Uuid::new_v7(uuid::Timestamp::from_unix(&*context, seconds, nanos))
    }
}
