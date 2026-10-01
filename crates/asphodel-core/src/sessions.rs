//! Per-session state for injection ("API surface and Hermes transport",
//! TIM-94, decision 6). It lives in daemon memory, keyed by bank and Hermes
//! session, so a restart loses it; the worst case is one repeated
//! injection.
//!
//! - **Pending injections.** `prefetch` holds what it injected under its
//!   `recall_id`, and the turn that echoes that id commits that set to the
//!   session's in-context set. A turn's sync can arrive after the next
//!   prefetch (TIM-99), so several can be pending at once, and a turn with
//!   a missing or unknown id changes nothing: it can't be matched to any of
//!   them. A set no turn acknowledges never enters the in-context set, which
//!   covers a prefetch Hermes timed out on; it waits until
//!   [`PENDING_PER_SESSION`] newer ones push it out, or the session idles.
//! - **The in-context set** is what the agent can already see this session:
//!   committed injections and recall-tool results (and, once it exists, the
//!   agenda). Injection skips it, and it's cleared on compaction. Extraction
//!   judges `used` (ADR 0001) against the set as each turn's sync left it,
//!   which ingest stores with the turn ([`Sessions::after_turn`]), never
//!   against the session as it is when the worker gets there.
//! - **Idle timeout.** A session untouched for
//!   `sessions.in_context_idle_days` on the service's clock is dropped. It's
//!   garbage collection only.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use jiff::{SignedDuration, Timestamp};
use uuid::Uuid;

/// The pending injections a session keeps; holding another drops the
/// oldest. Syncs trail prefetches by a turn or so, so a few is plenty.
const PENDING_PER_SESSION: usize = 4;

/// Every live session's state.
#[derive(Debug)]
pub(crate) struct Sessions {
    inner: Mutex<HashMap<Key, Session>>,
    idle: SignedDuration,
}

/// A bank's rowid and a Hermes session id.
type Key = (i64, String);

#[derive(Debug)]
struct Session {
    /// In the order the memories came into context, each once.
    in_context: Vec<Uuid>,
    /// Oldest first.
    pending: Vec<Pending>,
    touched_at: Timestamp,
}

#[derive(Debug)]
struct Pending {
    recall_id: Uuid,
    memories: Vec<Uuid>,
}

impl Sessions {
    /// No sessions yet, each to be dropped after `idle_days` untouched.
    pub(crate) fn new(idle_days: u32) -> Self {
        Self {
            inner: Mutex::default(),
            idle: SignedDuration::from_hours(24 * i64::from(idle_days)),
        }
    }

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

    /// Holds `memories` as a pending injection under `recall_id`, alongside
    /// any others still waiting for their turn.
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
        session.pending.push(Pending {
            recall_id,
            memories,
        });
        if session.pending.len() > PENDING_PER_SESSION {
            let dropped = session.pending.remove(0);
            tracing::debug!(
                recall = %dropped.recall_id,
                "no turn acknowledged a pending injection, so it's dropped"
            );
        }
    }

    /// The in-context set [`Sessions::turn`] would leave for a turn
    /// echoing `recall_id`, without changing anything: the set now, plus the
    /// pending injection held under that id. Ingest stores it with the turn,
    /// so extraction sees what the agent saw when it wrote the reply.
    pub(crate) fn after_turn(
        &self,
        bank_id: i64,
        session_id: &str,
        recall_id: Option<&str>,
        now: Timestamp,
    ) -> Vec<Uuid> {
        let sessions = self.live(now);
        let Some(session) = sessions.get(&(bank_id, session_id.to_owned())) else {
            return Vec::new();
        };
        let mut set = session.in_context.clone();
        let recall_id = recall_id.and_then(|id| id.trim().parse::<Uuid>().ok());
        if let Some(pending) = session
            .pending
            .iter()
            .find(|pending| Some(pending.recall_id) == recall_id)
        {
            add(&mut set, &pending.memories);
        }
        set
    }

    /// A turn arrived echoing `recall_id`: commits the pending injection
    /// held under that id. A missing or unknown id changes nothing.
    pub(crate) fn turn(
        &self,
        bank_id: i64,
        session_id: &str,
        recall_id: Option<&str>,
        now: Timestamp,
    ) {
        let mut sessions = self.live(now);
        let session = session(&mut sessions, bank_id, session_id, now);
        let Some(recall_id) = recall_id.and_then(|id| id.trim().parse::<Uuid>().ok()) else {
            return;
        };
        if let Some(index) = session
            .pending
            .iter()
            .position(|pending| pending.recall_id == recall_id)
        {
            let pending = session.pending.remove(index);
            add(&mut session.in_context, &pending.memories);
        }
    }

    /// Adds `memories` to the session's in-context set straight away, as
    /// for recall-tool results, which the agent sees as soon as they return.
    pub(crate) fn add(&self, bank_id: i64, session_id: &str, memories: &[Uuid], now: Timestamp) {
        let mut sessions = self.live(now);
        let session = session(&mut sessions, bank_id, session_id, now);
        add(&mut session.in_context, memories);
    }

    /// Clears the session's in-context set and pending injections, as Hermes
    /// asks on compaction, reset or rewind.
    pub(crate) fn clear(&self, bank_id: i64, session_id: &str, now: Timestamp) {
        let mut sessions = self.live(now);
        sessions.remove(&(bank_id, session_id.to_owned()));
    }

    /// The map, with every session idle past the timeout dropped.
    fn live(&self, now: Timestamp) -> MutexGuard<'_, HashMap<Key, Session>> {
        let mut sessions = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let idle = self.idle;
        sessions.retain(|_, session| {
            session
                .touched_at
                .checked_add(idle)
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
            pending: Vec::new(),
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
