//! The service layer the HTTP handlers and the replay harness both call.
//!
//! Handlers stay thin: anything that would be skipped by replay if it lived
//! in a handler belongs here instead (TIM-96, decision 3).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use jiff::Timestamp;
use serde::Serialize;
use uuid::Uuid;

use crate::clock::Clock;
use crate::config::{ConfigError, Tuning};
use crate::constants::RERANKER_DEADLINE;
use crate::extraction::{Call1Input, Call2Input, ExtractError, Extracted};
use crate::ingest::{Document, IngestError, Ingested, Outcome, Turn};
use crate::keep::{KeepError, Kept, Unkept};
use crate::models::{LlmClient, Models};
use crate::queue::{
    ChunkError, ChunkList, FailedChunk, Failure, Lease, Leases, QueueError, Retried, SourceKind,
};
use crate::retrieval::{Permit, Prefetch, PrefetchRequest, Recall, RecallError, RecallRequest};
use crate::sessions::Sessions;
use crate::store::bank::{Bank, BankError, BankIdentity, ModelIds};
use crate::store::{Store, StoreError};

/// One running store: the daemon's banks, models, extraction queue and jobs,
/// driven by a clock.
pub struct Service {
    clock: Arc<dyn Clock>,
    store: Store,
    tuning: Tuning,
    /// `None` only for a service built with [`Service::open`], which the
    /// existing callers use; [`Service::with_models`] always sets it.
    models: Option<Models>,
    /// The extraction queue's leases, one per bank at most. They live here
    /// rather than in the store so a restart releases them.
    leases: Leases,
    /// Pending injections and in-context sets, per Hermes session. In
    /// memory, so a restart costs at most one repeated injection.
    sessions: Sessions,
    /// The right to run the reranker, one inference at a time, shared by
    /// prefetch and explicit recall so a timed-out call never queues work
    /// behind the one running.
    reranker_permit: Arc<Permit>,
    /// The reranker deadline: [`RERANKER_DEADLINE`], fixed in code
    /// (ADR 0009), unless [`Service::with_reranker_deadline`] set another.
    reranker_deadline: Duration,
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
        let sessions = Sessions::new(tuning.sessions.in_context_idle_days);
        Self {
            clock,
            store,
            tuning,
            models: None,
            leases: Leases::default(),
            sessions,
            reranker_permit: Arc::default(),
            reranker_deadline: RERANKER_DEADLINE,
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
        let sessions = Sessions::new(tuning.sessions.in_context_idle_days);
        Ok(Self {
            clock,
            store,
            tuning,
            models: Some(models),
            leases: Leases::default(),
            sessions,
            reranker_permit: Arc::default(),
            reranker_deadline: RERANKER_DEADLINE,
        })
    }

    /// The same service with the reranker deadline set to
    /// `deadline`. The deadline is fixed in code (ADR 0009); this is for
    /// tests of the fallback and for the bench, which shouldn't wait 1.5 s
    /// per slow call.
    pub fn with_reranker_deadline(mut self, deadline: Duration) -> Self {
        self.reranker_deadline = deadline;
        self
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

    /// Ingests a turn into `bank`: scans it for secrets, stores it as a
    /// source and queues it as one chunk, or stores only a tombstone when it
    /// asked to forget ([`crate::ingest`]).
    ///
    /// A new turn also settles its prefetch: echoing a pending injection's
    /// `recall_id` commits it to the session's in-context set (TIM-94,
    /// decision 6). A duplicate, such as a resend from the plugin's spool,
    /// settles nothing, since the turn it repeats already did.
    pub fn ingest_turn(&self, bank: &str, turn: &Turn) -> Result<Ingested, IngestError> {
        let ingested = crate::ingest::ingest_turn(&self.store, bank, turn)?;
        if ingested.outcome == Outcome::Duplicate {
            return Ok(ingested);
        }
        let bank_id = {
            let conn = self.store.connection();
            crate::ingest::find_bank(&conn, bank)?.map(|(bank_id, _)| bank_id)
        };
        if let Some(bank_id) = bank_id {
            self.sessions.turn(
                bank_id,
                &turn.session_id,
                turn.recall_id.as_deref(),
                self.now(),
            );
        }
        Ok(ingested)
    }

    /// Ingests a document into `bank`: scans it for secrets, stores it as a
    /// source and queues each chunk no earlier version of it had.
    pub fn ingest_document(
        &self,
        bank: &str,
        document: &Document,
    ) -> Result<Ingested, IngestError> {
        crate::ingest::ingest_document(&self.store, bank, document)
    }

    /// The head of `bank`'s extraction queue, or `None` when the queue is
    /// empty or the bank's worker already holds a lease ([`crate::queue`]).
    pub fn claim_chunk(&self, bank: &str) -> Result<Option<Lease>, QueueError> {
        crate::queue::claim(&self.store, &self.leases, bank)
    }

    /// Marks a leased chunk extracted and takes it off the queue.
    pub fn complete_chunk(&self, lease: Lease) -> Result<(), QueueError> {
        crate::queue::complete(&self.store, &self.leases, lease)
    }

    /// Counts a failed attempt on a leased chunk. At
    /// [`CHUNK_RETRY_CAP`](crate::constants::CHUNK_RETRY_CAP) the chunk is
    /// marked failed.
    pub fn fail_chunk(&self, lease: Lease, error: ChunkError) -> Result<Failure, QueueError> {
        crate::queue::fail(&self.store, &self.leases, lease, error)
    }

    /// What call 1 would be given for the leased chunk ([`crate::extraction`]).
    /// `in_context` is the session's in-context set, the public ids of the
    /// memories the agent can already see; ids that aren't visible memories
    /// of the chunk's bank are left out, and a document gets none.
    pub fn call1_input(
        &self,
        lease: &Lease,
        in_context: &[Uuid],
    ) -> Result<Call1Input, ExtractError> {
        crate::extraction::call1_input(&self.store, &self.leases, &self.tuning, lease, in_context)
    }

    /// What call 2 would be given for the leased chunk if call 1 replied
    /// `call1_reply`, or `None` when call 2 wouldn't run
    /// ([`crate::extraction`]). Reads only; the lease is still held
    /// afterwards. `in_context` is as for [`Service::call1_input`], since the
    /// reply's `used_injected_ids` refer to its handles.
    pub fn call2_input(
        &self,
        lease: &Lease,
        call1_reply: &serde_json::Value,
        in_context: &[Uuid],
    ) -> Result<Option<Call2Input>, ExtractError> {
        let models = self.models.as_ref().ok_or(ExtractError::NoModels)?;
        crate::extraction::call2_input(
            &self.store,
            &self.leases,
            &self.tuning,
            models.embedder.as_ref(),
            lease,
            call1_reply,
            in_context,
        )
    }

    /// Runs call 1 on the leased chunk with `llm`, reconciles the claims
    /// with call 2 when they land near something stored, and commits,
    /// marking the chunk extracted. A failure writes nothing but the queue's
    /// count of the attempt, so the chunk is retried in place; an LLM that
    /// can't be used at all holds the queue without counting
    /// ([`ExtractError::Held`]).
    pub fn extract_chunk(
        &self,
        lease: Lease,
        llm: &dyn LlmClient,
        in_context: &[Uuid],
    ) -> Result<Extracted, ExtractError> {
        let models = self.models.as_ref().ok_or(ExtractError::NoModels)?;
        crate::extraction::extract(
            &self.store,
            &self.leases,
            &self.tuning,
            models.embedder.as_ref(),
            lease,
            llm,
            in_context,
        )
    }

    /// Prefetch: recalls for the user's message in `bank` and returns the
    /// injection, holding it as the session's pending set until the turn
    /// that echoes its `recall_id` ([`crate::retrieval`]).
    pub fn prefetch(&self, bank: &str, request: &PrefetchRequest) -> Result<Prefetch, RecallError> {
        crate::retrieval::prefetch(&self.retrieval()?, bank, request)
    }

    /// Explicit recall, as the `memory_recall` tool asks for it
    /// ([`crate::retrieval`]). The results join the session's in-context
    /// set when the request names a session.
    pub fn recall(&self, bank: &str, request: &RecallRequest) -> Result<Recall, RecallError> {
        crate::retrieval::recall(&self.retrieval()?, bank, request)
    }

    /// The session's in-context set: the public ids of the memories the
    /// agent can already see, oldest first. It's what
    /// [`Service::call1_input`] and [`Service::extract_chunk`] take.
    pub fn in_context(&self, bank: &str, session_id: &str) -> Result<Vec<Uuid>, RecallError> {
        let bank_id = self.bank_id(bank)?;
        Ok(self.sessions.in_context(bank_id, session_id, self.now()))
    }

    /// Clears the session's in-context set and pending injection, as Hermes
    /// asks on compaction, reset or rewind (`sessions/{id}/clear`).
    pub fn clear_session(&self, bank: &str, session_id: &str) -> Result<(), RecallError> {
        let bank_id = self.bank_id(bank)?;
        self.sessions.clear(bank_id, session_id, self.now());
        Ok(())
    }

    fn retrieval(&self) -> Result<crate::retrieval::Context<'_>, RecallError> {
        Ok(crate::retrieval::Context {
            store: &self.store,
            tuning: &self.tuning,
            models: self.models.as_ref().ok_or(RecallError::NoModels)?,
            sessions: &self.sessions,
            permit: &self.reranker_permit,
            deadline: self.reranker_deadline,
        })
    }

    fn bank_id(&self, bank: &str) -> Result<i64, RecallError> {
        let conn = self.store.connection();
        let (bank_id, _) =
            crate::ingest::find_bank(&conn, bank)?.ok_or(RecallError::UnknownBank)?;
        Ok(bank_id)
    }

    /// Chunks waiting or in flight in `bank`, not counting failed ones.
    pub fn queue_depth(&self, bank: &str) -> Result<usize, QueueError> {
        crate::queue::depth(&self.store, bank)
    }

    /// `bank`'s failed chunks, oldest failure first.
    pub fn failed_chunks(&self, bank: &str) -> Result<Vec<FailedChunk>, QueueError> {
        crate::queue::failed(&self.store, bank)
    }

    /// `bank`'s queue in the order it runs, and its failed chunks, as
    /// `chunks` lists them. With `failed_only` the queue is left out.
    pub fn chunks(&self, bank: &str, failed_only: bool) -> Result<ChunkList, QueueError> {
        let queued = if failed_only {
            Vec::new()
        } else {
            crate::queue::queued(&self.store, &self.leases, bank)?
        };
        Ok(ChunkList {
            queued,
            failed: crate::queue::failed(&self.store, bank)?,
        })
    }

    /// Puts `bank`'s failed chunks back on its queue: the ones `chunks`
    /// names, or all of them when it's `None` (`chunks --failed --retry`).
    pub fn retry_chunks(&self, bank: &str, chunks: Option<&[Uuid]>) -> Result<Retried, QueueError> {
        crate::queue::retry(&self.store, bank, chunks)
    }

    /// Takes the head of `bank`'s queue and extracts it with `llm`, giving
    /// call 1 the in-context set of the turn's session. `Ok(None)` when the
    /// queue is empty or the bank's worker already holds a lease. This is
    /// the step the daemon's per-bank worker repeats, and the replay harness
    /// runs it the same way (TIM-96, decision 3).
    pub fn extract_next(
        &self,
        bank: &str,
        llm: &dyn LlmClient,
    ) -> Result<Option<Extracted>, ExtractError> {
        let Some(lease) = self.claim_chunk(bank)? else {
            return Ok(None);
        };
        let in_context = match lease.source_kind {
            SourceKind::Document => Vec::new(),
            SourceKind::Turn => {
                let session: Option<String> = {
                    let conn = self.store.connection();
                    conn.query_row(
                        "SELECT session_id FROM sources WHERE uuid = ?1",
                        [lease.source.to_string()],
                        |row| row.get(0),
                    )
                    .map_err(StoreError::Sqlite)?
                };
                match session {
                    Some(session) => {
                        self.sessions
                            .in_context(lease.bank_id(), &session, self.now())
                    }
                    None => Vec::new(),
                }
            }
        };
        self.extract_chunk(lease, llm, &in_context).map(Some)
    }

    /// The names of every bank in the store, so the daemon can start a
    /// worker for each.
    pub fn bank_names(&self) -> Result<Vec<String>, StoreError> {
        let conn = self.store.connection();
        let mut statement = conn.prepare("SELECT name FROM banks ORDER BY id")?;
        let names = statement
            .query_map([], |row| row.get(0))?
            .collect::<Result<_, _>>()?;
        Ok(names)
    }

    /// Keeps the memories `ids` names in `bank`, so they never fade
    /// (`memory_keep`, TIM-94, decision 9).
    pub fn keep(&self, bank: &str, ids: &[String]) -> Result<Kept, KeepError> {
        crate::keep::keep(&self.store, bank, ids)
    }

    /// Hands the memories `ids` names in `bank` back to the significance
    /// extraction gave them (`memory_unkeep`).
    pub fn unkeep(&self, bank: &str, ids: &[String]) -> Result<Unkept, KeepError> {
        crate::keep::unkeep(&self.store, bank, ids)
    }

    /// Checkpoints the WAL into the database file, as the daemon does on
    /// SIGTERM once the chunk in flight is done (TIM-94, decision 3).
    pub fn checkpoint(&self) -> Result<(), StoreError> {
        self.store.checkpoint()
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

    /// What `/v1/health` reports once the service exists. The daemon is not
    /// ready while migrations run and models load; both happen before the
    /// service is built, and the daemon answers with [`Health::starting`]
    /// until then.
    pub fn health(&self) -> Health {
        Health {
            version: crate::VERSION,
            ready: true,
            now: self.now(),
        }
    }
}

impl Health {
    /// What `/v1/health` reports before the service exists: while the store
    /// opens and migrates and the models load.
    pub fn starting(now: Timestamp) -> Self {
        Self {
            version: crate::VERSION,
            ready: false,
            now,
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
