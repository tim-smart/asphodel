//! The service layer the HTTP handlers and the replay harness both call.
//!
//! Handlers stay thin: anything that would be skipped by replay if it lived
//! in a handler belongs here instead (TIM-96, decision 3).

use std::sync::Arc;

use jiff::Timestamp;
use serde::Serialize;

use crate::clock::Clock;

/// One running store: the daemon's banks, models and jobs, driven by a clock.
///
/// Later stages add the SQLite store, the models and the extraction queue
/// behind this type. For now it carries the clock and answers health checks.
pub struct Service {
    clock: Arc<dyn Clock>,
}

impl Service {
    /// Builds a service on the given clock.
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self { clock }
    }

    /// The clock this service runs on.
    pub fn clock(&self) -> &dyn Clock {
        self.clock.as_ref()
    }

    /// The current world time, as the service sees it.
    pub fn now(&self) -> Timestamp {
        self.clock.now()
    }

    /// What `/v1/health` reports. The daemon is not ready while models load
    /// and migrations run; with neither in place yet it is ready at once.
    pub fn health(&self) -> Health {
        Health {
            version: crate::VERSION,
            ready: true,
            now: self.now(),
        }
    }
}

impl std::fmt::Debug for Service {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Service").field("now", &self.now()).finish()
    }
}

/// The health response. The plugin compares `version`'s major against the
/// one it was written for and warns on a mismatch (TIM-94, decision 2).
#[derive(Debug, Clone, Serialize)]
pub struct Health {
    pub version: &'static str,
    pub ready: bool,
    pub now: Timestamp,
}
