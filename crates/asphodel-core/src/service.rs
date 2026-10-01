//! The service layer the HTTP handlers and the replay harness both call.
//!
//! Handlers stay thin: anything that would be skipped by replay if it lived
//! in a handler belongs here instead (TIM-96, decision 3).

use std::path::PathBuf;
use std::sync::Arc;

use jiff::Timestamp;
use serde::Serialize;

use crate::clock::Clock;
use crate::config::{ConfigError, Tuning};
use crate::models::Models;
use crate::store::bank::{Bank, BankError, BankIdentity, ModelIds};
use crate::store::{Store, StoreError};

/// One running store: the daemon's banks, models and jobs, driven by a clock.
///
/// Later stages add the extraction queue behind this type.
pub struct Service {
    clock: Arc<dyn Clock>,
    store: Store,
    tuning: Tuning,
    /// `None` only for a service built with [`Service::open`], which the
    /// existing callers use; [`Service::with_models`] always sets it.
    models: Option<Models>,
}

/// Why a service couldn't be built on a store and models.
#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    /// The tuning has no floor for a loaded model (ADR 0009).
    #[error(transparent)]
    Config(#[from] ConfigError),
}

impl Service {
    /// Builds a service on an open store and the tuning it runs under,
    /// without models. Retained for existing callers; `serve` and replay
    /// use [`Service::with_models`].
    pub fn open(clock: Arc<dyn Clock>, store: Store, tuning: Tuning) -> Self {
        Self {
            clock,
            store,
            tuning,
            models: None,
        }
    }

    /// Builds a service on an open store, the tuning it runs under and the
    /// models it serves with. The tuning must have a floor for each model's
    /// exact id, or the service doesn't open: a missing gate floor would
    /// flood injection, and a missing reconcile floor would skip
    /// reconciliation (ADR 0009). The check lives here, not in `serve`, so
    /// the replay harness gets the same refusal (TIM-96, decision 3).
    pub fn with_models(
        clock: Arc<dyn Clock>,
        store: Store,
        tuning: Tuning,
        models: Models,
    ) -> Result<Self, OpenError> {
        let ids = models.ids();
        tuning.check_floors(&ids.embedding, &ids.reranker)?;
        Ok(Self {
            clock,
            store,
            tuning,
            models: Some(models),
        })
    }

    /// The clock this service runs on.
    pub fn clock(&self) -> &dyn Clock {
        self.clock.as_ref()
    }

    /// The current world time, as the service sees it.
    pub fn now(&self) -> Timestamp {
        self.clock.now()
    }

    /// The store this service runs on. Always returns `Some`; the optional
    /// return type is retained for compatibility with existing callers.
    pub fn store(&self) -> Option<&Store> {
        Some(&self.store)
    }

    /// The tuning the service runs under.
    pub fn tuning(&self) -> &Tuning {
        &self.tuning
    }

    /// The models the service serves with, when it was built with them.
    pub fn models(&self) -> Option<&Models> {
        self.models.as_ref()
    }

    /// Creates a bank or merges `identity` into it (TIM-94, decision 7),
    /// recording `models` on creation. Retained for existing callers;
    /// [`Service::ensure_bank_with_models`] records the loaded models.
    pub fn ensure_bank(
        &self,
        name: &str,
        identity: &BankIdentity,
        models: &ModelIds,
    ) -> Result<Bank, BankError> {
        crate::store::bank::ensure(
            &self.store,
            name,
            identity,
            models,
            self.tuning.mental_models.profile_max_tokens,
        )
    }

    /// Creates a bank or merges `identity` into it. A new bank records the
    /// ids of the models this service was built with (TIM-94, decision 4);
    /// a merge leaves the recorded ids alone, since a change goes through
    /// `asphodel reembed` (TIM-99).
    pub fn ensure_bank_with_models(
        &self,
        name: &str,
        identity: &BankIdentity,
    ) -> Result<Bank, BankError> {
        let models = self.models.as_ref().ok_or(BankError::NoModels)?;
        self.ensure_bank(name, identity, &models.ids())
    }

    /// The periodic store upkeep. `serve` calls it on a timer and the replay
    /// harness after advancing its clock, so it runs on this service's clock
    /// either way (TIM-96, decision 3). It runs whether or not purge is
    /// paused: deleting an expired pre-migration copy is ADR 0010's bound on
    /// how long forgotten content survives, not a purge.
    ///
    /// The result says when the next pass is due, so a caller can wake at a
    /// copy's deadline instead of finding it on a later poll.
    pub fn housekeeping(&self) -> Result<Housekeeping, StoreError> {
        let copies_removed = self.store.expire_copies()?;
        let next_due = self.store.next_copy_expiry()?;
        Ok(Housekeeping {
            copies_removed,
            next_due,
        })
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
            .field("models", &self.models)
            .finish_non_exhaustive()
    }
}

/// What one [`Service::housekeeping`] pass did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Housekeeping {
    /// The pre-migration copies it deleted.
    pub copies_removed: Vec<PathBuf>,
    /// When the next pass has work, on the service's clock: the earliest
    /// deadline of a copy still on disk. `None` when nothing is pending.
    pub next_due: Option<Timestamp>,
}

/// The health response. The plugin compares `version`'s major against the
/// one it was written for and warns on a mismatch (TIM-94, decision 2).
#[derive(Debug, Clone, Serialize)]
pub struct Health {
    pub version: &'static str,
    pub ready: bool,
    pub now: Timestamp,
}
