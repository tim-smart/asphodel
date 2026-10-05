//! Retrying a failed LLM call within the call.
//!
//! [`LlmError::retry`] says why a call failed. When the provider is down
//! (unreachable, overloaded or limiting), the same request succeeds once it
//! recovers, so the daemon retries it for as long as that takes and counts
//! nothing. When the request itself may be the cause, such as a reply that
//! always runs past the timeout, each caller decides: extraction and
//! refresh count the failure as they always have, replay tries a few times,
//! and translate a few times within a budget. Everything else goes back at
//! once. The waits back off exponentially with jitter, a stopped
//! [`Sleeper`] ends the call as [`LlmError::Stopped`], and a call waiting
//! to retry is listed on a [`RetryBoard`] until it returns.

use std::collections::BTreeMap;
use std::hash::{BuildHasher, RandomState};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use jiff::Timestamp;
use serde::Serialize;

use super::llm::{LlmClient, LlmError, LlmRequest, LlmResponse, Retry};
use crate::clock::{Clock, Sleeper};

/// How many attempts a call gets for each kind of failure, and how long it
/// waits between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// The backoff after the first failed attempt. It doubles with each
    /// failed attempt after, whatever the kind, up to `max_wait`, and each
    /// wait is jittered down to no less than half of it.
    pub first_wait: Duration,
    pub max_wait: Duration,
    /// Attempts in all for a call failing because the provider is down,
    /// the first included; `None` retries until it recovers.
    pub provider_attempts: Option<u32>,
    /// Attempts in all for a call failing in a way the request may cause,
    /// the first included; 1 returns the failure at once. The provider
    /// being down in between uses none of them.
    pub request_attempts: u32,
    /// No new attempt starts once this long has passed since the first
    /// began, checked before and after each wait; `None` sets no limit.
    pub budget: Option<Duration>,
}

impl RetryPolicy {
    /// Extraction and refresh: the provider being down is retried until it
    /// recovers, and a failure the request may cause is returned for the
    /// caller to count, toward a chunk's retry cap or a model's
    /// `last_error`.
    pub fn daemon() -> Self {
        Self {
            first_wait: Duration::from_secs(1),
            max_wait: Duration::from_secs(60),
            provider_attempts: None,
            request_attempts: 1,
            budget: None,
        }
    }

    /// Replay's live calls: as [`Self::daemon`], but with no chunk retry
    /// cap behind it, a failure the request may cause gets five attempts
    /// before the run fails.
    pub fn replay() -> Self {
        Self {
            request_attempts: 5,
            ..Self::daemon()
        }
    }

    /// Translate, which an operator waits on: three attempts for either
    /// kind of failure, waiting 1s then 2s, within 30s.
    pub fn translate() -> Self {
        Self {
            first_wait: Duration::from_secs(1),
            max_wait: Duration::from_secs(4),
            provider_attempts: Some(3),
            request_attempts: 3,
            budget: Some(Duration::from_secs(30)),
        }
    }

    /// The backoff after `failures` failed attempts, before jitter.
    fn backoff(&self, failures: u32) -> Duration {
        let doublings = failures.saturating_sub(1).min(16);
        self.first_wait
            .saturating_mul(1 << doublings)
            .min(self.max_wait)
    }
}

/// A call waiting to retry, as `status` lists it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LlmRetrying {
    /// Who made the call: a bank's name for its extraction, `refresh`.
    pub caller: String,
    /// When its first attempt failed.
    pub since: Timestamp,
    /// The attempts that have failed so far.
    pub attempts: u32,
    /// The last failure, never the request or the reply.
    pub error: String,
}

/// The calls waiting to retry, shared by every [`LlmRetry`] reporting to
/// it and read by `status`.
#[derive(Debug, Clone, Default)]
pub struct RetryBoard {
    calls: Arc<Mutex<BTreeMap<u64, LlmRetrying>>>,
    next: Arc<AtomicU64>,
}

impl RetryBoard {
    /// Every call waiting to retry, oldest first.
    pub fn retrying(&self) -> Vec<LlmRetrying> {
        let mut calls: Vec<_> = self.lock().values().cloned().collect();
        calls.sort_by_key(|call| call.since);
        calls
    }

    fn lock(&self) -> MutexGuard<'_, BTreeMap<u64, LlmRetrying>> {
        self.calls
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// A call's entry on a board, removed when the call returns, even if it
/// panics.
struct Listed<'a> {
    board: &'a RetryBoard,
    id: u64,
}

impl Drop for Listed<'_> {
    fn drop(&mut self) {
        self.board.lock().remove(&self.id);
    }
}

/// An [`LlmClient`] that retries its inner client's failures under a
/// [`RetryPolicy`].
pub struct LlmRetry {
    inner: Arc<dyn LlmClient>,
    policy: RetryPolicy,
    clock: Arc<dyn Clock>,
    sleeper: Arc<dyn Sleeper>,
    board: Option<(RetryBoard, String)>,
}

impl LlmRetry {
    /// Retries `inner` under `policy`, measuring time on `clock` and
    /// waiting on `sleeper`.
    pub fn new(
        inner: Arc<dyn LlmClient>,
        policy: RetryPolicy,
        clock: Arc<dyn Clock>,
        sleeper: Arc<dyn Sleeper>,
    ) -> Self {
        Self {
            inner,
            policy,
            clock,
            sleeper,
            board: None,
        }
    }

    /// Lists each call waiting to retry on `board`, as made by `caller`.
    pub fn reporting(mut self, board: RetryBoard, caller: &str) -> Self {
        self.board = Some((board, caller.to_string()));
        self
    }

    /// Whether the budget, if any, has run out since `started`, or would
    /// before `wait` is over.
    fn past_budget(&self, started: Timestamp, wait: Duration) -> bool {
        let Some(budget) = self.policy.budget else {
            return false;
        };
        let elapsed =
            Duration::try_from(started.duration_until(self.clock.now())).unwrap_or(Duration::ZERO);
        wait >= budget.saturating_sub(elapsed)
    }

    fn call(
        &self,
        attempt_once: impl Fn() -> Result<LlmResponse, LlmError>,
    ) -> Result<LlmResponse, LlmError> {
        let started = self.clock.now();
        let (mut provider, mut request) = (0, 0);
        let mut listed: Option<(Listed<'_>, Timestamp)> = None;
        loop {
            let error = match attempt_once() {
                Ok(response) => return Ok(response),
                Err(error) => error,
            };
            let spent = match error.retry() {
                Retry::Never => return Err(error),
                Retry::ProviderDown => {
                    provider += 1;
                    self.policy
                        .provider_attempts
                        .is_some_and(|attempts| provider >= attempts)
                }
                Retry::MaybeTheRequest => {
                    request += 1;
                    request >= self.policy.request_attempts
                }
            };
            if spent {
                return Err(error);
            }
            let failures = provider + request;
            let wait = jittered(self.policy.backoff(failures));
            if self.past_budget(started, wait) {
                return Err(error);
            }
            if self.sleeper.stopped() {
                return Err(LlmError::Stopped);
            }
            if let Some((board, caller)) = &self.board {
                let (listed, since) = listed.get_or_insert_with(|| {
                    let id = board.next.fetch_add(1, Ordering::Relaxed);
                    (Listed { board, id }, self.clock.now())
                });
                let retrying = LlmRetrying {
                    caller: caller.clone(),
                    since: *since,
                    attempts: failures,
                    error: error.to_string(),
                };
                board.lock().insert(listed.id, retrying);
            }
            tracing::warn!(
                attempt = failures,
                retry = ?error.retry(),
                wait_ms = u64::try_from(wait.as_millis()).unwrap_or(u64::MAX),
                %error,
                "an LLM call failed and will be retried"
            );
            self.sleeper.sleep(wait);
            if self.sleeper.stopped() {
                return Err(LlmError::Stopped);
            }
            if self.past_budget(started, Duration::ZERO) {
                return Err(error);
            }
        }
    }
}

/// `backoff` less a random part of its second half, so callers that failed
/// together don't retry in lockstep.
fn jittered(backoff: Duration) -> Duration {
    let half = backoff / 2;
    let fraction = RandomState::new().hash_one(backoff) as f64 / u64::MAX as f64;
    half + half.mul_f64(fraction)
}

impl LlmClient for LlmRetry {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn reasoning_effort(&self) -> Option<&str> {
        self.inner.reasoning_effort()
    }

    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        self.call(|| self.inner.complete(request))
    }

    fn complete_identified(
        &self,
        request: &LlmRequest,
        identities: &[(String, uuid::Uuid)],
    ) -> Result<LlmResponse, LlmError> {
        self.call(|| self.inner.complete_identified(request, identities))
    }

    fn skips_write(&self, request: &LlmRequest, identities: &[(String, uuid::Uuid)]) -> bool {
        self.inner.skips_write(request, identities)
    }
}
