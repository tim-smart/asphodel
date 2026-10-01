//! Per-session state for injection ("API surface and Hermes transport",
//! TIM-94, decision 6). It lives in daemon memory, keyed by bank and Hermes
//! session, so a restart loses it; the worst case is one repeated
//! injection.
//!
//! - **Pending injections.** `prefetch` holds what it injected under its
//!   `recall_id`. The turn that echoes that id commits the set to the
//!   session's in-context set. A turn with a missing or different id
//!   discards it, which covers a prefetch Hermes timed out on and trivial
//!   prompts it never prefetched.
//! - **The in-context set** is what the agent can already see this session:
//!   committed injections and recall-tool results (and, once it exists, the
//!   agenda). Injection skips it, and extraction gets it to judge `used`
//!   (ADR 0001). It's cleared on compaction.
//! - **Idle timeout.** A session untouched for
//!   [`IN_CONTEXT_IDLE_TIMEOUT`] on the service's clock is dropped.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use jiff::{SignedDuration, Timestamp};
use uuid::Uuid;

use crate::constants::IN_CONTEXT_IDLE_TIMEOUT;

/// Every live session's state.
#[derive(Debug, Default)]
pub(crate) struct Sessions {
    inner: Mutex<HashMap<Key, Session>>,
}

/// A bank's rowid and a Hermes session id.
type Key = (i64, String);

#[derive(Debug)]
struct Session {
    /// In the order the memories came into context, each once.
    in_context: Vec<Uuid>,
    pending: Option<Pending>,
    touched_at: Timestamp,
}

#[derive(Debug)]
struct Pending {
    recall_id: Uuid,
    memories: Vec<Uuid>,
}

impl Sessions {
    /// The session's in-context set, oldest first.
    pub(crate) fn in_context(&self, bank_id: i64, session_id: &str, now: Timestamp) -> Vec<Uuid> {
        let mut sessions = self.live(now);
        match sessions.get_mut(&(bank_id, session_id.to_owned())) {
            Some(session) => {
                session.touched_at = now;
                session.in_context.clone()
            }
            None => Vec::new(),
        }
    }

    /// Holds `memories` as the session's pending injection under
    /// `recall_id`, replacing any earlier one that no turn committed.
    pub(crate) fn hold(
        &self,
        bank_id: i64,
        session_id: &str,
        recall_id: Uuid,
        memories: Vec<Uuid>,
        now: Timestamp,
    ) {
        let mut sessions = self.live(now);
        let session = session(&mut sessions, bank_id, session_id, now);
        session.pending = Some(Pending {
            recall_id,
            memories,
        });
    }

    /// A turn arrived echoing `recall_id`: commits the pending injection
    /// when the ids match, and discards it otherwise.
    pub(crate) fn turn(
        &self,
        bank_id: i64,
        session_id: &str,
        recall_id: Option<&str>,
        now: Timestamp,
    ) {
        let mut sessions = self.live(now);
        let session = session(&mut sessions, bank_id, session_id, now);
        let Some(pending) = session.pending.take() else {
            return;
        };
        let matches = recall_id
            .and_then(|id| id.trim().parse::<Uuid>().ok())
            .is_some_and(|id| id == pending.recall_id);
        if matches {
            add(&mut session.in_context, &pending.memories);
        } else {
            tracing::debug!(
                recall = %pending.recall_id,
                "a turn didn't echo the pending injection's recall id, so it's discarded"
            );
        }
    }

    /// Adds `memories` to the session's in-context set straight away, as
    /// for recall-tool results, which the agent sees as soon as they return.
    pub(crate) fn add(&self, bank_id: i64, session_id: &str, memories: &[Uuid], now: Timestamp) {
        let mut sessions = self.live(now);
        let session = session(&mut sessions, bank_id, session_id, now);
        add(&mut session.in_context, memories);
    }

    /// Clears the session's in-context set and pending injection, as Hermes
    /// asks on compaction, reset or rewind.
    pub(crate) fn clear(&self, bank_id: i64, session_id: &str, now: Timestamp) {
        let mut sessions = self.live(now);
        sessions.remove(&(bank_id, session_id.to_owned()));
    }

    /// The map, with every session idle past the timeout dropped.
    fn live(&self, now: Timestamp) -> MutexGuard<'_, HashMap<Key, Session>> {
        let mut sessions = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let timeout = SignedDuration::try_from(IN_CONTEXT_IDLE_TIMEOUT)
            .expect("the idle timeout fits a signed duration");
        sessions.retain(|_, session| {
            session
                .touched_at
                .checked_add(timeout)
                .is_ok_and(|expires| now < expires)
        });
        sessions
    }
}

fn session<'a>(
    sessions: &'a mut HashMap<Key, Session>,
    bank_id: i64,
    session_id: &str,
    now: Timestamp,
) -> &'a mut Session {
    let session = sessions
        .entry((bank_id, session_id.to_owned()))
        .or_insert_with(|| Session {
            in_context: Vec::new(),
            pending: None,
            touched_at: now,
        });
    session.touched_at = now;
    session
}

fn add(set: &mut Vec<Uuid>, memories: &[Uuid]) {
    for memory in memories {
        if !set.contains(memory) {
            set.push(*memory);
        }
    }
}
