//! The service layer the HTTP handlers and the replay harness both call.
//!
//! Handlers stay thin: anything that would be skipped by replay if it lived
//! in a handler belongs here instead (TIM-96, decision 3).

use std::path::PathBuf;
use std::sync::Arc;

use jiff::Timestamp;
use serde::Serialize;

use crate::clock::Clock;
use crate::config::Tuning;
use crate::store::bank::{Bank, BankError, BankIdentity, ModelIds};
use crate::store::{Store, StoreError};

/// One running store: the daemon's banks, models and jobs, driven by a clock.
///
/// Later stages add the models and the extraction queue behind this type.
/// `serve` always builds it on a store; [`Service::new`] still builds one
/// without, where nothing is persisted and bank calls fail.
pub struct Service {
    clock: Arc<dyn Clock>,
    store: Option<Store>,
    tuning: Tuning,
}

impl Service {
    /// Builds a service on the given clock and no store.
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            store: None,
            tuning: Tuning::default(),
        }
    }

    /// Builds a service on an open store and the tuning it runs under.
    pub fn open(clock: Arc<dyn Clock>, store: Store, tuning: Tuning) -> Self {
        Self {
            clock,
            store: Some(store),
            tuning,
        }
    }

    /// The clock this service runs on.
    pub fn clock(&self) -> &dyn Clock {
        self.clock.as_ref()
    }

    /// The current world time, as the service sees it.
    pub fn now(&self) -> Timestamp {
        self.clock.now()
    }

    /// The store, when the service has one.
    pub fn store(&self) -> Option<&Store> {
        self.store.as_ref()
    }

    /// The tuning the service runs under.
    pub fn tuning(&self) -> &Tuning {
        &self.tuning
    }

    /// Creates a bank or merges `identity` into it (TIM-94, decision 7).
    pub fn ensure_bank(
        &self,
        name: &str,
        identity: &BankIdentity,
        models: &ModelIds,
    ) -> Result<Bank, BankError> {
        let store = self.store.as_ref().ok_or(BankError::NoStore)?;
        crate::store::bank::ensure(
            store,
            name,
            identity,
            models,
            self.tuning.mental_models.profile_max_tokens,
        )
    }

    /// The periodic store upkeep. `serve` calls it on a timer and the replay
    /// harness after advancing its clock, so it runs on this service's clock
    /// either way (TIM-96, decision 3). It runs whether or not purge is
    /// paused: deleting an expired pre-migration copy is ADR 0010's bound on
    /// how long forgotten content survives, not a purge.
    pub fn housekeeping(&self) -> Result<Housekeeping, StoreError> {
        let copies_removed = match &self.store {
            Some(store) => store.expire_copies()?,
            None => Vec::new(),
        };
        Ok(Housekeeping { copies_removed })
    }

    /// What `/v1/health` reports. The daemon is not ready while models load
    /// and migrations run; both happen before the service is built, so once
    /// it exists it is ready.
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
        f.debug_struct("Service")
            .field("now", &self.now())
            .field("store", &self.store)
            .finish_non_exhaustive()
    }
}

/// What one [`Service::housekeeping`] pass did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Housekeeping {
    /// The pre-migration copies it deleted.
    pub copies_removed: Vec<PathBuf>,
}

/// The health response. The plugin compares `version`'s major against the
/// one it was written for and warns on a mismatch (TIM-94, decision 2).
#[derive(Debug, Clone, Serialize)]
pub struct Health {
    pub version: &'static str,
    pub ready: bool,
    pub now: Timestamp,
}
