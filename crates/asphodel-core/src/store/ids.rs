//! Public ids: UUIDv7, timed by the store's clock (TIM-90), or derived
//! UUIDv5s for a replay (TIM-96, decision 4).
//!
//! `uuid::Uuid::now_v7` reads the system clock, which ADR 0004 forbids, so
//! timed ids are built from a [`Timestamp`] the caller took from the
//! [`Clock`] (`clippy.toml` denies the direct call). A shared `ContextV7`
//! keeps ids minted in the same instant ordered.
//!
//! A replay opens its store with deterministic ids. A memory's id is then
//! UUIDv5 of (source id, `<chunk position>:<claim index in call 1's reply>`).
//! An entity's is UUIDv5 of (creating source id,
//! `entity:<chunk position>:<name>`), where the proposed name is composed
//! to NFC, trimmed and lowercased: the exact key `resolve_proposals`
//! dedups on within a commit, without a second normalization. Neither key
//! depends on other store contents; `entity:` keeps them disjoint. A
//! source's id is derived from its key, and a chunk's from its source and
//! position, so recorded `used` ids, reconcile targets and citations point
//! at the same rows on every run. Ids with no natural
//! parent (recall ids, edits, banks, models) count up under a fixed
//! namespace, which is the same on every run of the same timeline.
//!
//! [`Clock`]: crate::clock::Clock

use std::sync::Mutex;

use jiff::Timestamp;
use uuid::{ContextV7, Uuid};

/// The parent of every derived id that has no parent of its own.
pub const NAMESPACE: Uuid = Uuid::from_u128(0x6a2f_1d3e_8c4b_4f7a_9e1d_2b3c_4d5e_6f70);

/// UUIDv5 of `name` under `parent`: how a store with deterministic ids
/// mints an id for a row that belongs to another.
pub fn derived(parent: Uuid, name: &str) -> Uuid {
    Uuid::new_v5(&parent, name.as_bytes())
}

/// Mints ids. Behind mutexes to be shared across threads: the timed
/// context keeps a counter for ids minted in the same instant, and the
/// derived counter must never hand out the same number twice.
#[derive(Debug)]
pub enum IdSource {
    /// Production: UUIDv7 from the clock.
    Timed(Mutex<ContextV7>),
    /// Replay: UUIDv5 from parents and names, and a counter under
    /// [`NAMESPACE`] for the rest.
    Derived(Mutex<u64>),
}

impl Default for IdSource {
    fn default() -> Self {
        Self::timed()
    }
}

impl IdSource {
    pub fn timed() -> Self {
        Self::Timed(Mutex::new(ContextV7::new()))
    }

    pub fn counting() -> Self {
        Self::Derived(Mutex::new(0))
    }

    /// An id for a row with no natural parent. Timed: a UUIDv7 whose time
    /// part is `now`, with a time before 1970 clamped to the epoch, since
    /// UUIDv7 has no room for it. Derived: the next number under
    /// [`NAMESPACE`].
    pub fn next(&self, now: Timestamp) -> Uuid {
        match self {
            Self::Timed(context) => {
                let seconds = u64::try_from(now.as_second()).unwrap_or(0);
                let nanos = u32::try_from(now.subsec_nanosecond()).unwrap_or(0);
                let context = context
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                Uuid::new_v7(uuid::Timestamp::from_unix(&*context, seconds, nanos))
            }
            Self::Derived(counter) => {
                let mut counter = counter
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                *counter += 1;
                derived(NAMESPACE, &counter.to_string())
            }
        }
    }

    /// An id for a row that belongs to `parent`, distinguished by `name`.
    /// Timed: the same as [`IdSource::next`]. Derived: [`derived`].
    pub fn derived(&self, parent: Uuid, name: &str, now: Timestamp) -> Uuid {
        match self {
            Self::Timed(_) => self.next(now),
            Self::Derived(_) => derived(parent, name),
        }
    }
}
