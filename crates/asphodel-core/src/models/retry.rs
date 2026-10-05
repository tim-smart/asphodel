//! Retrying a transient LLM failure within the call.
//!
//! A transport error, a timeout, a 408, a 429 without `Retry-After`, a 5xx
//! or a transient backend code ([`LlmError::is_retryable`]) is tried again
//! after a short, jittered backoff, so one blip doesn't become a counted
//! failure on a chunk or a model. Everything else, holds included, goes
//! back at once. The daemon puts [`LlmRetry`] under its gate, so the gate
//! only sees the final result, and replay puts it round its live client.

use std::hash::{BuildHasher, RandomState};
use std::sync::Arc;
use std::time::Duration;

use super::llm::{LlmClient, LlmError, LlmRequest, LlmResponse};
use crate::clock::{Clock, Sleeper};

/// How many attempts a call gets and how long it waits between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Attempts in all, the first included; at least one is made.
    pub attempts: u32,
    /// The backoff before the second attempt. It doubles before each one
    /// after, up to `max_wait`, and each wait is jittered down to no less
    /// than half of it.
    pub first_wait: Duration,
    pub max_wait: Duration,
    /// No new attempt starts once this long has passed since the first
    /// began, so a call that timed out isn't made to wait for another
    /// timeout.
    pub budget: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            attempts: 3,
            first_wait: Duration::from_secs(1),
            max_wait: Duration::from_secs(4),
            budget: Duration::from_secs(30),
        }
    }
}

impl RetryPolicy {
    /// The backoff before attempt `attempt + 1`, before jitter.
    fn backoff(&self, attempt: u32) -> Duration {
        let doublings = attempt.saturating_sub(1).min(16);
        self.first_wait
            .saturating_mul(1 << doublings)
            .min(self.max_wait)
    }
}

/// An [`LlmClient`] that retries its inner client's transient failures.
pub struct LlmRetry {
    inner: Arc<dyn LlmClient>,
    policy: RetryPolicy,
    clock: Arc<dyn Clock>,
    sleeper: Arc<dyn Sleeper>,
}

impl LlmRetry {
    /// Retries `inner` under `policy`, measuring the budget on `clock` and
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
        }
    }

    fn call(
        &self,
        attempt_once: impl Fn() -> Result<LlmResponse, LlmError>,
    ) -> Result<LlmResponse, LlmError> {
        let started = self.clock.now();
        let mut attempt = 1;
        loop {
            let error = match attempt_once() {
                Ok(response) => return Ok(response),
                Err(error) => error,
            };
            if !error.is_retryable() || attempt >= self.policy.attempts {
                return Err(error);
            }
            let elapsed = started.duration_until(self.clock.now());
            if Duration::try_from(elapsed).unwrap_or(Duration::ZERO) >= self.policy.budget {
                return Err(error);
            }
            let wait = jittered(self.policy.backoff(attempt));
            tracing::warn!(
                attempt,
                attempts = self.policy.attempts,
                wait_ms = u64::try_from(wait.as_millis()).unwrap_or(u64::MAX),
                %error,
                "an LLM call failed and will be retried"
            );
            self.sleeper.sleep(wait);
            attempt += 1;
        }
    }
}

/// `backoff` less a random part of its second half.
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
