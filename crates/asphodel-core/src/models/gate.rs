//! The daemon's one way to the LLM: at most `[llm] concurrency` calls in
//! flight across every bank's extraction and every mental model refresh,
//! and one hold shared by all of them.
//!
//! When any call comes back [`LlmError::UsageLimited`] or
//! [`LlmError::RateLimited`], every call that starts before the limit
//! lifts fails the same way without reaching the LLM. Extraction holds the
//! queue on either without counting anything, so a burst of 429s across
//! the chunks in flight never spends their retry caps on a fault that
//! isn't theirs.

use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

use jiff::{SignedDuration, Timestamp};

use super::llm::{LlmClient, LlmError, LlmRequest, LlmResponse};
use crate::clock::Clock;

/// An [`LlmClient`] that limits the calls in flight and shares holds.
pub struct LlmGate {
    inner: Arc<dyn LlmClient>,
    clock: Arc<dyn Clock>,
    limit: usize,
    state: Mutex<State>,
    freed: Condvar,
}

#[derive(Default)]
struct State {
    in_flight: usize,
    hold: Option<Hold>,
}

/// Until when every call holds, and why.
#[derive(Clone, Copy)]
struct Hold {
    until: Timestamp,
    /// A usage limit rather than a rate limit.
    usage: bool,
}

impl LlmGate {
    /// Lets at most `limit` calls through to `inner` at once; at least one.
    /// Holds run on `clock`.
    pub fn new(inner: Arc<dyn LlmClient>, limit: usize, clock: Arc<dyn Clock>) -> Self {
        Self {
            inner,
            clock,
            limit: limit.max(1),
            state: Mutex::new(State::default()),
            freed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The error a call gets while a hold is on, or `None` once it's over.
    fn held(&self, state: &mut State) -> Option<LlmError> {
        let hold = state.hold?;
        let now = self.clock.now();
        if now >= hold.until {
            state.hold = None;
            return None;
        }
        Some(if hold.usage {
            LlmError::UsageLimited {
                resets_at: hold.until,
            }
        } else {
            let left = now.duration_until(hold.until);
            LlmError::RateLimited {
                retry_after: Duration::try_from(left).unwrap_or(Duration::ZERO),
            }
        })
    }

    /// Waits for a free slot, unless a hold is on. The slot is freed when
    /// the returned guard drops, even if the call panics.
    fn enter(&self) -> Result<Slot<'_>, LlmError> {
        let mut state = self.lock();
        loop {
            if let Some(error) = self.held(&mut state) {
                return Err(error);
            }
            if state.in_flight < self.limit {
                state.in_flight += 1;
                return Ok(Slot(self));
            }
            state = self
                .freed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// Starts a hold when a call hit a limit, unless a longer one is on.
    fn note(&self, result: &Result<LlmResponse, LlmError>) {
        let hold = match result {
            Err(LlmError::UsageLimited { resets_at }) => Hold {
                until: *resets_at,
                usage: true,
            },
            Err(LlmError::RateLimited { retry_after }) => {
                let wait = SignedDuration::try_from(*retry_after).unwrap_or(SignedDuration::MAX);
                Hold {
                    until: self.clock.now().checked_add(wait).unwrap_or(Timestamp::MAX),
                    usage: false,
                }
            }
            _ => return,
        };
        let mut state = self.lock();
        if state.hold.is_none_or(|current| hold.until > current.until) {
            tracing::warn!(until = %hold.until, usage = hold.usage, "every LLM call holds");
            state.hold = Some(hold);
        }
    }
}

/// A call's slot, freed on drop.
struct Slot<'a>(&'a LlmGate);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.lock().in_flight -= 1;
        self.0.freed.notify_all();
    }
}

impl LlmClient for LlmGate {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn reasoning_effort(&self) -> Option<&str> {
        self.inner.reasoning_effort()
    }

    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        let _slot = self.enter()?;
        let result = self.inner.complete(request);
        self.note(&result);
        result
    }

    fn complete_identified(
        &self,
        request: &LlmRequest,
        identities: &[(String, uuid::Uuid)],
    ) -> Result<LlmResponse, LlmError> {
        let _slot = self.enter()?;
        let result = self.inner.complete_identified(request, identities);
        self.note(&result);
        result
    }
}
