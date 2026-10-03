//! The service layer the HTTP handlers and the replay harness both call.
//!
//! Handlers stay thin: anything that would be skipped by replay if it lived
//! in a handler belongs here instead.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use rusqlite::OptionalExtension;
use serde::Serialize;
use uuid::Uuid;

use crate::agenda::Agenda;
use crate::clock::Clock;
use crate::config::{ConfigError, DeletionInputs, PurgePause, Tuning};
use crate::constants::RERANKER_DEADLINE;
use crate::entities::{
    AliasRemoval, AliasRemoved, EntityError, LinkEdited, LinkRequest, MergeRequest, Merged,
    Unmerged,
};
use crate::erase::{
    Aftermath, BankDeleteError, BankDeleted, Erased, ForgetError, ForgetRequest, Forgotten,
};
use crate::extraction::{Call1Input, Call2Input, Committed, ExtractError, Extracted};
use crate::ingest::{Document, IngestError, Ingested, Outcome, Turn};
use crate::inspect::{EntityView, InspectError, MemoryView, ModelView};
use crate::keep::{KeepError, Kept, SignificanceSet, Unkept};
use crate::mental_models::{
    Model, ModelEdit, ModelError, ModelSpec, Outcome as RefreshOutcome, RefreshInput, RefreshRun,
    Refreshes, Schedule,
};
use crate::models::{Embedder, LlmClient, Models};
use crate::operations::{Audit, AuditError, AuditList, Backup, BackupError, Status};
use crate::queue::{
    ChunkError, ChunkList, FailedChunk, Failure, Lease, Leases, QueueError, Retried, SourceKind,
};
use crate::reembed::{ReembedError, ReembedStatus};
use crate::retrieval::{Permit, Prefetch, PrefetchRequest, Recall, RecallError, RecallRequest};
use crate::sessions::Sessions;
use crate::store::bank::{Bank, BankError, BankIdentity, ModelIds};
use crate::store::{Store, StoreError};
use crate::sweep::{PurgeError, PurgePlan, SweepSchedule, Sweeps};
use crate::system_prompt::{Block, BlockEntry, Blocks};

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
    /// The reranker deadline: [`RERANKER_DEADLINE`], fixed in code, unless
    /// [`Service::with_reranker_deadline`] set another.
    reranker_deadline: Duration,
    /// Each bank's last refresh trigger and last daily sweep. In memory;
    /// the requests themselves are in the store.
    schedule: Schedule,
    /// Each bank's system prompt block, rebuilt lazily once cleared.
    blocks: Blocks,
    /// Held while a refresh runs, so the timer and `model refresh` never
    /// refresh at once.
    refreshing: Mutex<()>,
    /// Whether purge and the source sweep run, from the stored deletion
    /// fingerprint at startup and any ack since.
    purge: Mutex<PurgePause>,
    /// Each bank's last nightly sweep.
    sweeps: SweepSchedule,
    /// Held while the sweep runs, so two callers never sweep at once.
    sweeping: Mutex<()>,
    /// Embedding models carried besides the models' own, for banks a
    /// re-embed hasn't moved yet.
    previous_embedders: Vec<Arc<dyn Embedder>>,
    /// The banks whose re-embed is running, and why each one's last run
    /// failed.
    reembeds: Mutex<ReembedRuns>,
}

/// A chunk claimed by [`Service::next_extraction`], with what call 1 is
/// shown besides its text: the session's in-context set as the turn stored
/// it, and the block entries it held.
#[derive(Debug)]
pub struct Claimed {
    pub lease: Lease,
    pub in_context: Vec<Uuid>,
    pub entries: Vec<BlockEntry>,
}

/// What [`Service`] keeps in memory about re-embeds.
#[derive(Debug, Default)]
struct ReembedRuns {
    running: BTreeSet<i64>,
    failed: BTreeMap<i64, String>,
}

/// Why a service couldn't be built on a store and models.
#[derive(Debug, thiserror::Error)]
pub enum OpenError {
    /// The tuning has no floor for a loaded model.
    #[error(transparent)]
    Config(#[from] ConfigError),
}

impl Service {
    /// Builds a service on an open store and the tuning it runs under,
    /// without models. Retained for existing callers; `serve` and replay
    /// use [`Service::with_models`].
    pub fn open(clock: Arc<dyn Clock>, store: Store, tuning: Tuning) -> Self {
        let sessions = Sessions::new(tuning.sessions.in_context_idle_days);
        let started = clock.now();
        let schedule = Schedule::new(started);
        Self {
            clock,
            store,
            tuning,
            models: None,
            leases: Leases::default(),
            sessions,
            reranker_permit: Arc::default(),
            reranker_deadline: RERANKER_DEADLINE,
            schedule,
            blocks: Blocks::default(),
            refreshing: Mutex::new(()),
            purge: Mutex::new(PurgePause::Running),
            sweeps: SweepSchedule::new(started),
            sweeping: Mutex::new(()),
            previous_embedders: Vec::new(),
            reembeds: Mutex::default(),
        }
    }

    /// Builds a service on an open store, the tuning it runs under and the
    /// models it serves with. The tuning must have a floor for each model's
    /// exact id, or the service doesn't open: a missing gate floor would
    /// flood injection, and a missing reconcile floor would skip
    /// reconciliation. The check lives here, not in `serve`, so the replay
    /// harness gets the same refusal.
    pub fn with_models(
        clock: Arc<dyn Clock>,
        store: Store,
        tuning: Tuning,
        models: Models,
    ) -> Result<Self, OpenError> {
        let ids = models.ids();
        tuning.check_floors(&ids.embedding, &ids.reranker)?;
        let sessions = Sessions::new(tuning.sessions.in_context_idle_days);
        let started = clock.now();
        let schedule = Schedule::new(started);
        Ok(Self {
            clock,
            store,
            tuning,
            models: Some(models),
            leases: Leases::default(),
            sessions,
            reranker_permit: Arc::default(),
            reranker_deadline: RERANKER_DEADLINE,
            schedule,
            blocks: Blocks::default(),
            refreshing: Mutex::new(()),
            purge: Mutex::new(PurgePause::Running),
            sweeps: SweepSchedule::new(started),
            sweeping: Mutex::new(()),
            previous_embedders: Vec::new(),
            reembeds: Mutex::default(),
        })
    }

    /// The same service, also carrying `embedder`, a model banks may have
    /// recorded before the daemon's model changed. Each such bank is served
    /// with it until `asphodel reembed` swaps the bank to the daemon's
    /// model. It needs a reconcile floor like the daemon's own.
    pub fn with_previous_embedder(
        mut self,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Self, OpenError> {
        let reranker = self
            .models
            .as_ref()
            .map(|models| models.reranker.model_id().to_string());
        if let Some(reranker) = reranker {
            self.tuning.check_floors(embedder.model_id(), &reranker)?;
        }
        self.previous_embedders.push(embedder);
        Ok(self)
    }

    /// The banks recorded under an embedding model this service doesn't
    /// carry, with that model. Recall and extraction are refused for them
    /// until a re-embed moves them, so `serve` warns about each at startup
    /// and `status` asks for attention.
    pub fn banks_without_their_model(&self) -> Result<Vec<(String, String)>, StoreError> {
        let Some(models) = &self.models else {
            return Ok(Vec::new());
        };
        let conn = self.store.connection();
        let mut statement = conn.prepare("SELECT name, embedding_model FROM banks ORDER BY id")?;
        let banks: Vec<(String, String)> = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<_, _>>()?;
        Ok(banks
            .into_iter()
            .filter(|(_, recorded)| {
                models.embedder.model_id() != recorded
                    && !self
                        .previous_embedders
                        .iter()
                        .any(|embedder| embedder.model_id() == recorded)
            })
            .collect())
    }

    /// The embedder `bank_id` is served with: the model it recorded, until
    /// a re-embed swaps it. Refused when this service doesn't carry that
    /// model, so no vector is written or compared under another.
    fn bank_embedder<'a>(
        &'a self,
        models: &'a Models,
        bank_id: i64,
    ) -> Result<&'a dyn Embedder, ExtractError> {
        let recorded = {
            let conn = self.store.connection();
            crate::reembed::recorded_model(&conn, bank_id).map_err(StoreError::Sqlite)?
        };
        crate::models::serving(models, &self.previous_embedders, &recorded)
            .ok_or(ExtractError::ModelUnavailable { model: recorded })
    }

    /// The same service with the reranker deadline set to
    /// `deadline`. The deadline is fixed in code; this is for
    /// tests of the fallback and for the bench, which shouldn't wait 1.5 s
    /// per slow call.
    pub fn with_reranker_deadline(mut self, deadline: Duration) -> Self {
        self.reranker_deadline = deadline;
        self
    }

    /// The same service with the purge state `serve` read from the store at
    /// startup ([`Store::check_fingerprint`]). A service built without it
    /// runs purge, which is what replay wants: replay is where δ is changed
    /// on purpose, so it never pauses. While purge runs, the store also
    /// keeps the deletion inputs behind its fingerprint, so `purge plan` can
    /// name what changes later.
    pub fn with_purge_pause(self, pause: PurgePause) -> Self {
        if pause == PurgePause::Running
            && let Err(error) = self
                .store
                .record_deletion_inputs(&DeletionInputs::new(&self.tuning))
        {
            tracing::warn!(%error, "recording the deletion inputs failed");
        }
        *self.purge.lock().unwrap_or_else(|e| e.into_inner()) = pause;
        self
    }

    /// Whether purge and the source sweep are running or paused.
    pub fn purge_pause(&self) -> PurgePause {
        self.purge.lock().unwrap_or_else(|e| e.into_inner()).clone()
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

    /// Creates a bank or merges `identity` into it,
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
    /// ids of the models this service was built with;
    /// a merge leaves the recorded ids alone, since a change goes through
    /// `asphodel reembed`.
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
    /// `recall_id` commits it to the session's in-context set. The set as the
    /// turn leaves it is stored with the turn,
    /// in the same transaction, and extraction reads it from there rather
    /// than from the session, which may since have been cleared, added to
    /// or lost to a restart. A duplicate, such as a resend
    /// from the plugin's spool, settles nothing, since the turn it repeats
    /// already did.
    pub fn ingest_turn(&self, bank: &str, turn: &Turn) -> Result<Ingested, IngestError> {
        let bank_id = {
            let conn = self.store.connection();
            crate::ingest::find_bank(&conn, bank)?.map(|(bank_id, _)| bank_id)
        };
        let entries = match bank_id {
            Some(bank_id) => {
                self.restore_block(bank_id, &turn.session_id)?;
                let conn = self.store.connection();
                crate::system_prompt::mapped_entries(
                    &conn,
                    bank_id,
                    &turn.session_id,
                    self.now(),
                    self.mapping_expiry(),
                )
                .map_err(StoreError::Sqlite)?
            }
            None => Vec::new(),
        };
        let in_context = bank_id.map_or_else(Vec::new, |bank_id| {
            self.sessions.after_turn(
                bank_id,
                &turn.session_id,
                turn.recall_id.as_deref(),
                self.now(),
            )
        });
        let ingested = crate::ingest::ingest_turn(&self.store, bank, turn, &in_context, &entries)?;
        if ingested.outcome == Outcome::Duplicate {
            return Ok(ingested);
        }
        if let Some(bank_id) = bank_id {
            self.sessions.turn(
                bank_id,
                &turn.session_id,
                turn.recall_id.as_deref(),
                self.now(),
            );
            let conn = self.store.connection();
            crate::system_prompt::touch_session(&conn, bank_id, &turn.session_id, self.now())
                .map_err(StoreError::Sqlite)?;
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

    /// The first chunk in `bank`'s extraction queue that isn't out already,
    /// or `None` when there's none or the bank already has `[llm]
    /// concurrency` leases out ([`crate::queue`]).
    pub fn claim_chunk(&self, bank: &str) -> Result<Option<Lease>, QueueError> {
        crate::queue::claim(
            &self.store,
            &self.leases,
            bank,
            self.tuning.llm.concurrency as usize,
        )
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
            self.bank_embedder(models, lease.bank_id())?,
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
        self.extract_with_entries(lease, llm, in_context, &[])
    }

    /// [`Service::extract_chunk`] with the entries of the block the turn's
    /// session held, which call 1 is shown with the memories they cite.
    fn extract_with_entries(
        &self,
        lease: Lease,
        llm: &dyn LlmClient,
        in_context: &[Uuid],
        entries: &[BlockEntry],
    ) -> Result<Extracted, ExtractError> {
        let models = self.models.as_ref().ok_or(ExtractError::NoModels)?;
        let bank_id = lease.bank_id();
        let watermark = {
            let conn = self.store.connection();
            crate::mental_models::watermark(&conn, bank_id).map_err(StoreError::Sqlite)?
        };
        let extracted = crate::extraction::extract(
            &self.store,
            &self.leases,
            &self.tuning,
            self.bank_embedder(models, bank_id)?,
            lease,
            llm,
            in_context,
            entries,
        )?;
        self.after_writes(bank_id, watermark, &extracted.memories)?;
        Ok(extracted)
    }

    /// Prefetch: recalls for the user's message in `bank` and returns the
    /// injection, holding it as the session's pending set until the turn
    /// that echoes its `recall_id` ([`crate::retrieval`]).
    pub fn prefetch(&self, bank: &str, request: &PrefetchRequest) -> Result<Prefetch, RecallError> {
        self.scored_prefetch(bank, request)
            .map(|scored| scored.prefetch)
    }

    /// [`Service::prefetch`], also returning what the gate was shown: the
    /// query and every reranked candidate with its logit, for replay's
    /// labelling material.
    pub fn scored_prefetch(
        &self,
        bank: &str,
        request: &PrefetchRequest,
    ) -> Result<crate::retrieval::ScoredPrefetch, RecallError> {
        if let Ok(bank_id) = self.bank_id(bank) {
            if let Some(block) = request.block_id {
                self.map_held_block(bank_id, &request.session_id, block)?;
            }
            self.restore_block(bank_id, &request.session_id)?;
        }
        crate::retrieval::scored_prefetch(&self.retrieval()?, bank, request)
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
        self.restore_block(bank_id, session_id)?;
        Ok(self.sessions.in_context(bank_id, session_id, self.now()))
    }

    /// Clears the session's in-context set and pending injection, as Hermes
    /// asks on compaction, reset or rewind (`sessions/{id}/clear`).
    pub fn clear_session(&self, bank: &str, session_id: &str) -> Result<(), RecallError> {
        let bank_id = self.bank_id(bank)?;
        self.sessions.clear(bank_id, session_id, self.now());
        let conn = self.store.connection();
        crate::system_prompt::unmap_session(&conn, bank_id, session_id)
            .map_err(StoreError::Sqlite)?;
        Ok(())
    }

    fn retrieval(&self) -> Result<crate::retrieval::Context<'_>, RecallError> {
        Ok(crate::retrieval::Context {
            store: &self.store,
            tuning: &self.tuning,
            models: self.models.as_ref().ok_or(RecallError::NoModels)?,
            previous: &self.previous_embedders,
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
    /// call 1 the in-context set stored with the turn when it was ingested,
    /// never the session's set now. `Ok(None)` when the
    /// queue is empty or the bank's worker already holds a lease. This is
    /// the step the daemon's per-bank worker repeats, and the replay harness
    /// runs it the same way.
    pub fn extract_next(
        &self,
        bank: &str,
        llm: &dyn LlmClient,
    ) -> Result<Option<Extracted>, ExtractError> {
        let Some(Claimed {
            lease,
            in_context,
            entries,
        }) = self.next_extraction(bank)?
        else {
            return Ok(None);
        };
        self.extract_leased(lease, llm, &in_context, &entries)
            .map(Some)
    }

    /// The first half of [`Service::extract_next`]: claims the head of
    /// `bank`'s queue and reads the in-context set and block entries stored
    /// with it. The replay harness takes the lease from here so it can
    /// script the LLM's replies against what call 1 and call 2 are shown,
    /// then finishes with [`Service::extract_leased`].
    pub fn next_extraction(&self, bank: &str) -> Result<Option<Claimed>, ExtractError> {
        let Some(lease) = self.claim_chunk(bank)? else {
            return Ok(None);
        };
        let (in_context, entries) = match lease.source_kind {
            SourceKind::Document => (Vec::new(), Vec::new()),
            SourceKind::Turn => {
                let stored: Option<(String, Option<String>)> = {
                    let conn = self.store.connection();
                    conn.query_row(
                        "SELECT t.memories, e.entries FROM turn_in_context t
                         JOIN sources s ON s.id = t.source_id
                         LEFT JOIN turn_entries e ON e.source_id = t.source_id
                         WHERE s.uuid = ?1",
                        [lease.source.to_string()],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )
                    .optional()
                    .map_err(StoreError::Sqlite)?
                };
                // No row is an empty set, including for a turn ingested
                // before version 5: the session can't stand in for it. No
                // entries is none, including for a turn before version 6.
                match stored {
                    Some((memories, entries)) => (
                        serde_json::from_str(&memories).unwrap_or_else(|error| {
                            tracing::warn!(
                                source = %lease.source,
                                %error,
                                "a turn's stored in-context set doesn't parse; extracting without it"
                            );
                            Vec::new()
                        }),
                        entries
                            .and_then(|entries| serde_json::from_str(&entries).ok())
                            .unwrap_or_default(),
                    ),
                    None => (Vec::new(), Vec::new()),
                }
            }
        };
        Ok(Some(Claimed {
            lease,
            in_context,
            entries,
        }))
    }

    /// The first half of [`Service::extract_leased`]: runs the LLM calls
    /// and plans the chunk, holding its lease, without writing anything.
    /// Replay runs it when the worker claims the chunk and
    /// [`Service::commit_extraction`] a latency later.
    pub fn prepare_extraction(
        &self,
        lease: Lease,
        llm: &dyn LlmClient,
        in_context: &[Uuid],
        entries: &[BlockEntry],
    ) -> Result<crate::extraction::Prepared, ExtractError> {
        let models = self.models.as_ref().ok_or(ExtractError::NoModels)?;
        let bank_id = lease.bank_id();
        crate::extraction::prepare(
            &self.store,
            &self.leases,
            &self.tuning,
            self.bank_embedder(models, bank_id)?,
            lease,
            llm,
            in_context,
            entries,
        )
    }

    /// The candidate lists call 2 was shown for a prepared chunk, with each
    /// neighbour's similarity to its claim, for replay's labelling material.
    /// Reads only.
    pub fn call2_lists(
        &self,
        prepared: &crate::extraction::Prepared,
    ) -> Result<Vec<crate::extraction::Call2List>, StoreError> {
        crate::extraction::call2_lists(&self.store, prepared)
    }

    /// The second half of [`Service::extract_leased`]: commits a prepared
    /// chunk, then runs what follows writes. It waits for every chunk of
    /// the bank handed out before this one to commit or fail first. A chunk
    /// that another commit made stale since its search
    /// ([`crate::extraction`]) is [`ExtractError::Stale`], with nothing
    /// written or counted; [`Service::try_commit_extraction`] hands it back
    /// to reconcile again instead.
    pub fn commit_extraction(
        &self,
        prepared: crate::extraction::Prepared,
    ) -> Result<Extracted, ExtractError> {
        match self.try_commit_extraction(prepared)? {
            Committed::Extracted(extracted) => Ok(extracted),
            Committed::Stale(_) => Err(ExtractError::Stale),
        }
    }

    /// [`Service::commit_extraction`], handing a stale chunk back with its
    /// lease still held, for [`Service::redo_extraction`].
    pub fn try_commit_extraction(
        &self,
        prepared: crate::extraction::Prepared,
    ) -> Result<Committed, ExtractError> {
        let bank_id = prepared.lease().bank_id();
        let watermark = {
            let conn = self.store.connection();
            crate::mental_models::watermark(&conn, bank_id).map_err(StoreError::Sqlite)?
        };
        let committed = crate::extraction::commit_prepared(&self.store, &self.leases, prepared)?;
        if let Committed::Extracted(extracted) = &committed {
            self.after_writes(bank_id, watermark, &extracted.memories)?;
        }
        Ok(committed)
    }

    /// What call 2 would be given if `prepared` searched again now, or
    /// `None` when it wouldn't run. Reads only: replay scripts the redo's
    /// call 2 from it.
    pub fn redo_input(
        &self,
        prepared: &crate::extraction::Prepared,
    ) -> Result<Option<Call2Input>, ExtractError> {
        Ok(crate::extraction::redo_input(&self.store, prepared)?)
    }

    /// Searches and runs call 2 again with `llm` for a chunk
    /// [`Service::try_commit_extraction`] found stale, keeping its call 1.
    /// The redo isn't counted against the chunk.
    pub fn redo_extraction(
        &self,
        prepared: crate::extraction::Prepared,
        llm: &dyn LlmClient,
    ) -> Result<crate::extraction::Prepared, ExtractError> {
        crate::extraction::redo(&self.store, &self.leases, llm, prepared)
    }

    /// The second half of [`Service::extract_next`]: extracts a chunk
    /// claimed by [`Service::next_extraction`] with the in-context set and
    /// entries it returned.
    pub fn extract_leased(
        &self,
        lease: Lease,
        llm: &dyn LlmClient,
        in_context: &[Uuid],
        entries: &[BlockEntry],
    ) -> Result<Extracted, ExtractError> {
        self.extract_with_entries(lease, llm, in_context, entries)
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
    /// (`memory_keep`).
    pub fn keep(&self, bank: &str, ids: &[String]) -> Result<Kept, KeepError> {
        let watermark = self.watermark_for(bank)?;
        let kept = crate::keep::keep(&self.store, bank, ids)?;
        if let Some((bank_id, watermark)) = watermark {
            self.after_writes(bank_id, watermark, &[])?;
        }
        Ok(kept)
    }

    /// Hands the memories `ids` names in `bank` back to the significance
    /// extraction gave them (`memory_unkeep`).
    pub fn unkeep(&self, bank: &str, ids: &[String]) -> Result<Unkept, KeepError> {
        let watermark = self.watermark_for(bank)?;
        let unkept = crate::keep::unkeep(&self.store, bank, ids)?;
        if let Some((bank_id, watermark)) = watermark {
            self.after_writes(bank_id, watermark, &[])?;
        }
        Ok(unkept)
    }

    /// Checkpoints the WAL into the database file, as the daemon does on
    /// SIGTERM once the chunk in flight is done.
    pub fn checkpoint(&self) -> Result<(), StoreError> {
        self.store.checkpoint()
    }

    /// The periodic store upkeep. `serve` calls it on a timer and the replay
    /// harness after advancing its clock, so it runs on this service's clock
    /// either way. It runs whether or not purge is
    /// paused: deleting a pre-migration copy 7 days after its migration is
    /// what bounds how long forgotten content survives, not a purge.
    ///
    /// The result says when the next pass is due, so a caller can wake at a
    /// copy's deadline instead of finding it on a later poll.
    pub fn housekeeping(&self) -> Result<Housekeeping, StoreError> {
        let copies_removed = self.store.expire_copies()?;
        {
            let conn = self.store.connection();
            crate::system_prompt::expire_mappings(&conn, self.now(), self.mapping_expiry())?;
        }
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

/// Backup, status and the audit lists. Restore is offline, under
/// the data-dir lock, so it isn't here: [`crate::operations::restore`].
impl Service {
    /// `POST /v1/backup`: an online backup of the store, checked, in a file
    /// already unlinked from the data dir ([`crate::operations`]).
    pub fn backup(&self) -> Result<Backup, BackupError> {
        crate::operations::take_backup(&self.store)
    }

    /// Records that a backup's stream completed now, for `status`.
    pub fn record_backup(&self) -> Result<(), StoreError> {
        crate::operations::record_backup(&self.store, self.now())
    }

    /// `GET /v1/status`: what an operator alerts on, and what needs
    /// attention now.
    pub fn status(&self) -> Result<Status, StoreError> {
        let mut status = crate::operations::status(
            &self.store,
            self.purge_pause(),
            self.tuning.deletion_fingerprint(),
        )?;
        for (bank, model) in self.banks_without_their_model()? {
            status.attention.push(format!(
                "{bank}: its recorded embedding model {model} isn't loaded, so recall and extraction are refused; run `asphodel reembed --bank {bank}`"
            ));
        }
        Ok(status)
    }

    /// `GET /v1/banks/{bank}/{purges,forgets,sweeps,recalls}`, newest first.
    pub fn audit(
        &self,
        bank: &str,
        list: AuditList,
        limit: Option<usize>,
    ) -> Result<Audit, AuditError> {
        crate::operations::audit(&self.store, bank, list, limit)
    }
}

/// Inspection and correction. The views read only; each correction is one
/// logged edit, and refreshes the models it changes what they see.
impl Service {
    /// `memory show`: the memory, both significance fields, its passage or
    /// why it's gone, its accesses, edits and chain, its strength in parts,
    /// what holds back a purge, and projected fade and purge dates.
    pub fn show_memory(&self, bank: &str, id: &str) -> Result<MemoryView, InspectError> {
        crate::inspect::memory(&self.store, &self.tuning, &self.purge_pause(), bank, id)
    }

    /// `entity show`: an entity by id, `user`, `assistant`, name or alias.
    pub fn show_entity(&self, bank: &str, entity: &str) -> Result<EntityView, InspectError> {
        crate::inspect::entity(&self.store, bank, entity)
    }

    /// `model show [--entry]`: a model with each entry's citations.
    pub fn show_model(
        &self,
        bank: &str,
        name: &str,
        entry: Option<&str>,
    ) -> Result<ModelView, InspectError> {
        crate::inspect::model_view(&self.store, bank, name, entry)
    }

    /// The first instant the memory's strength fell below τ, from its
    /// access log and the bank's clock as they stand now, to the minute.
    /// `None` when it hasn't faded yet. It's what a replay's `faded_at`
    /// probe reads.
    pub fn faded_at(&self, bank: &str, id: Uuid) -> Result<Option<Timestamp>, InspectError> {
        crate::inspect::faded_at(&self.store, &self.tuning, bank, id)
    }

    /// Every live memory's strength now, by id: neither forgotten nor
    /// retracted. The replay report counts bands from it.
    pub fn strengths(&self, bank: &str) -> Result<Vec<(Uuid, f64)>, InspectError> {
        crate::inspect::strengths(&self.store, &self.tuning, bank)
    }

    /// `memory significance <id> <level|clear>`: the owner's significance,
    /// the field keep and unkeep write.
    pub fn set_significance(
        &self,
        bank: &str,
        id: &str,
        level: Option<&str>,
    ) -> Result<SignificanceSet, KeepError> {
        let watermark = self.watermark_for(bank)?;
        let set = crate::keep::set_significance(&self.store, bank, id, level)?;
        if let Some((bank_id, watermark)) = watermark {
            self.after_writes(bank_id, watermark, &[])?;
        }
        Ok(set)
    }

    /// `entity merge <from> <into>`.
    pub fn merge_entities(
        &self,
        bank: &str,
        request: &MergeRequest,
    ) -> Result<Merged, EntityError> {
        let (bank_id, merged, models) = crate::entities::merge(&self.store, bank, request)?;
        self.corrected(bank_id, &models)?;
        Ok(merged)
    }

    /// `entity unmerge <edit-id>`.
    pub fn unmerge_entity(&self, bank: &str, edit: &str) -> Result<Unmerged, EntityError> {
        let (bank_id, unmerged, models) = crate::entities::unmerge(&self.store, bank, edit)?;
        self.corrected(bank_id, &models)?;
        Ok(unmerged)
    }

    /// `entity alias rm [--relink-to]`.
    pub fn remove_alias(
        &self,
        bank: &str,
        request: &AliasRemoval,
    ) -> Result<AliasRemoved, EntityError> {
        let (bank_id, removed, models) = crate::entities::remove_alias(&self.store, bank, request)?;
        self.corrected(bank_id, &models)?;
        Ok(removed)
    }

    /// `entity link <memory> <entity>`.
    pub fn link_entity(
        &self,
        bank: &str,
        request: &LinkRequest,
    ) -> Result<LinkEdited, EntityError> {
        let (bank_id, linked, models) =
            crate::entities::edit_link(&self.store, bank, request, true)?;
        self.corrected(bank_id, &models)?;
        Ok(linked)
    }

    /// `entity unlink <memory> <entity>`.
    pub fn unlink_entity(
        &self,
        bank: &str,
        request: &LinkRequest,
    ) -> Result<LinkEdited, EntityError> {
        let (bank_id, unlinked, models) =
            crate::entities::edit_link(&self.store, bank, request, false)?;
        self.corrected(bank_id, &models)?;
        Ok(unlinked)
    }

    /// Requests a refresh of the models a correction changed, and clears
    /// the bank's block.
    fn corrected(&self, bank_id: i64, models: &BTreeSet<i64>) -> Result<(), StoreError> {
        if !models.is_empty() {
            let now = self.now();
            {
                let conn = self.store.connection();
                crate::mental_models::request(&conn, &self.schedule, models, now)?;
            }
            self.schedule.triggered(bank_id, now);
        }
        self.blocks.invalidate(bank_id);
        Ok(())
    }
}

/// Re-embedding and bank deletion, the daemon jobs that change a bank at
/// scale.
impl Service {
    /// How long a re-embed's swap or a bank deletion waits for the bank's
    /// chunk in flight: longer than any one extraction takes.
    const HOLD_WAIT: Duration = Duration::from_secs(600);

    /// `POST /v1/banks/{bank}/reembed`: records a job moving the bank to the
    /// daemon's embedding model, unless it's already on it, and says where
    /// it stands. [`Service::run_reembed`] runs it.
    pub fn start_reembed(&self, bank: &str) -> Result<ReembedStatus, ReembedError> {
        let models = self.models.as_ref().ok_or(ReembedError::NoModels)?;
        if crate::reembed::start(&self.store, bank, models.embedder.model_id())? {
            let bank_id = crate::reembed::bank(&self.store.connection(), bank)?.0;
            self.reembed_runs().failed.remove(&bank_id);
        }
        self.reembed_status(bank)
    }

    /// `GET /v1/banks/{bank}/reembed`.
    pub fn reembed_status(&self, bank: &str) -> Result<ReembedStatus, ReembedError> {
        let models = self.models.as_ref().ok_or(ReembedError::NoModels)?;
        let bank_id = crate::reembed::bank(&self.store.connection(), bank)?.0;
        let (running, failed) = {
            let runs = self.reembed_runs();
            (
                runs.running.contains(&bank_id),
                runs.failed.get(&bank_id).cloned(),
            )
        };
        crate::reembed::status(
            &self.store,
            bank,
            models.embedder.model_id(),
            running,
            failed,
        )
    }

    /// Runs the bank's recorded re-embed to the end, from where it last
    /// stopped, and swaps it in. A run already going for the bank is left
    /// to it. Blocks for as long as it takes: the daemon calls it on a
    /// blocking thread, and at startup for every job a restart stopped.
    pub fn run_reembed(&self, bank: &str) -> Result<ReembedStatus, ReembedError> {
        let models = self.models.as_ref().ok_or(ReembedError::NoModels)?;
        let bank_id = crate::reembed::bank(&self.store.connection(), bank)?.0;
        if !self.reembed_runs().running.insert(bank_id) {
            return self.reembed_status(bank);
        }
        let result = self.reembed_until_swapped(models, bank_id);
        {
            let mut runs = self.reembed_runs();
            runs.running.remove(&bank_id);
            match &result {
                Ok(_) => {
                    runs.failed.remove(&bank_id);
                }
                Err(error) => {
                    runs.failed.insert(bank_id, error.to_string());
                }
            }
        }
        if let Err(error) = &result {
            tracing::warn!(bank_id, %error, "a re-embed stopped; it resumes from where it got to");
        }
        result?;
        self.reembed_status(bank)
    }

    fn reembed_until_swapped(&self, models: &Models, bank_id: i64) -> Result<(), ReembedError> {
        let embedder = models.embedder.as_ref();
        while crate::reembed::step(&self.store, bank_id, embedder)? > 0 {}
        let hold = self
            .leases
            .hold(bank_id, Self::HOLD_WAIT)
            .ok_or(ReembedError::Busy)?;
        if let Some(memories) = crate::reembed::swap(&self.store, &hold, bank_id, embedder)? {
            tracing::info!(
                bank_id,
                memories,
                model = embedder.model_id(),
                "re-embedded a bank"
            );
        }
        Ok(())
    }

    /// The banks with a re-embed recorded, which the daemon resumes at
    /// startup.
    pub fn pending_reembeds(&self) -> Result<Vec<String>, StoreError> {
        crate::reembed::pending(&self.store)
    }

    fn reembed_runs(&self) -> std::sync::MutexGuard<'_, ReembedRuns> {
        self.reembeds.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `bank delete <bank> --confirm <bank>`: erases the bank through the
    /// erase path and removes everything it holds, writing a daemon-wide
    /// `bank_deleted` row with counts. `confirm` must repeat the name. A
    /// live plugin recreates the bank empty on `initialize`, so disable it
    /// first.
    pub fn delete_bank(&self, bank: &str, confirm: &str) -> Result<BankDeleted, BankDeleteError> {
        if bank.trim() != confirm.trim() {
            return Err(BankDeleteError::NotConfirmed);
        }
        let bank_id = {
            let conn = self.store.connection();
            crate::ingest::find_bank(&conn, bank)?
                .ok_or(BankDeleteError::UnknownBank)?
                .0
        };
        let _hold = self
            .leases
            .hold(bank_id, Self::HOLD_WAIT)
            .ok_or(BankDeleteError::Busy)?;
        let (bank_id, deleted) = crate::erase::delete_bank(&self.store, bank)?;
        self.sessions.forget_bank(bank_id);
        self.blocks.invalidate(bank_id);
        {
            let mut runs = self.reembed_runs();
            runs.failed.remove(&bank_id);
        }
        Ok(deleted)
    }
}

/// Forget, the erase path and the nightly sweep.
impl Service {
    /// `memory_forget`: hides each named memory's whole chain at once and
    /// queues its erase behind the chunks already queued
    /// ([`crate::erase`]). Forget never pauses.
    pub fn forget(&self, bank: &str, ids: &[String]) -> Result<Forgotten, ForgetError> {
        self.forget_request(
            bank,
            &ForgetRequest {
                ids: ids.to_vec(),
                session_id: None,
            },
        )
    }

    /// [`Service::forget`] with the request as the plugin sends it: the
    /// session its request turn will arrive in, so the audit row can be
    /// linked to that turn when it's ingested.
    pub fn forget_request(
        &self,
        bank: &str,
        request: &ForgetRequest,
    ) -> Result<Forgotten, ForgetError> {
        let (bank_id, forgotten, aftermath) = crate::erase::forget(&self.store, bank, request)?;
        self.settle(bank_id, aftermath)?;
        Ok(forgotten)
    }

    /// Runs every erase that's ready, in every bank, behind the same
    /// barrier as [`Service::erase_next`]. Housekeeping calls it, so a ready
    /// erase runs even when there's no LLM and so no worker.
    pub fn run_erases(&self) -> Result<Vec<Erased>, StoreError> {
        let mut erased = Vec::new();
        for (_, bank, _) in self.bank_zones()? {
            loop {
                match self.erase_next(&bank) {
                    Ok(Some(one)) => erased.push(one),
                    Ok(None) | Err(QueueError::UnknownBank) => break,
                    Err(QueueError::Store(error)) => return Err(error),
                    Err(error @ QueueError::NotHeld { .. }) => {
                        tracing::warn!(%error, "an erase failed");
                        break;
                    }
                }
            }
        }
        Ok(erased)
    }

    /// Runs the erase at the head of `bank`'s queue, once every chunk queued
    /// before it has been extracted or has failed. `None` when there's none
    /// to run yet. The bank's worker calls it before each
    /// [`Service::extract_next`].
    pub fn erase_next(&self, bank: &str) -> Result<Option<Erased>, QueueError> {
        let Some((bank_id, erased, aftermath)) = crate::erase::erase_next(&self.store, bank)?
        else {
            return Ok(None);
        };
        self.settle(bank_id, aftermath)?;
        Ok(Some(erased))
    }

    /// Runs every bank's nightly sweep that's due now: purge, then the
    /// source, failed-chunk and recall-log sweep ([`crate::sweep`]). The
    /// daemon calls it before [`Service::run_refreshes`], so a model citing
    /// a purged memory refreshes once.
    pub fn run_sweeps(&self) -> Result<Sweeps, StoreError> {
        let _sweeping = self.sweeping.lock().unwrap_or_else(|e| e.into_inner());
        let pause = self.purge_pause();
        let mut settled = Vec::new();
        let result = crate::sweep::run(
            &self.store,
            &self.tuning,
            &self.sweeps,
            &pause,
            &self.bank_zones()?,
            &mut settled,
        );
        // What committed is settled even when a later step failed.
        let settling = settled
            .into_iter()
            .try_for_each(|(bank_id, aftermath)| self.settle(bank_id, aftermath));
        let sweeps = result?;
        settling?;
        Ok(sweeps)
    }

    /// Purge's first phase, as the nightly sweep runs it: the head of every
    /// chain eligible now, by bank. Nothing is deleted. Exposed so tests can
    /// change a chain between the phases.
    #[doc(hidden)]
    pub fn purge_candidates(&self) -> Result<Vec<(String, Uuid)>, StoreError> {
        let now = self.now();
        let mut found = Vec::new();
        if self.purge_pause() != PurgePause::Running {
            return Ok(found);
        }
        for (bank_id, bank, _) in self.bank_zones()? {
            for head in crate::sweep::candidates(&self.store, &self.tuning, bank_id, now)? {
                let uuid: String = self.store.connection().query_row(
                    "SELECT uuid FROM memories WHERE id = ?1",
                    [head],
                    |row| row.get(0),
                )?;
                found.push((bank.clone(), uuid.parse().expect("a stored uuid parses")));
            }
        }
        Ok(found)
    }

    /// Purge's second phase for one chain: decides again, inside its
    /// transaction, and purges it only if it's still eligible. `None` when
    /// nothing was purged.
    #[doc(hidden)]
    pub fn purge_chain(&self, bank: &str, head: Uuid) -> Result<Option<Erased>, StoreError> {
        let found: Option<(i64, i64)> = {
            let conn = self.store.connection();
            conn.query_row(
                "SELECT m.bank_id, m.id FROM memories m JOIN banks b ON b.id = m.bank_id
                 WHERE b.name = ?1 AND m.uuid = ?2",
                (bank.trim(), head.to_string()),
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?
        };
        let Some((bank_id, head)) = found else {
            return Ok(None);
        };
        let purged = crate::sweep::purge_chain(
            &self.store,
            &self.tuning,
            &self.purge_pause(),
            bank_id,
            head,
            self.now(),
        )?;
        let Some((memories, aftermath)) = purged else {
            return Ok(None);
        };
        self.settle(bank_id, aftermath)?;
        Ok(Some(Erased {
            reason: crate::erase::EraseReason::Purge,
            memories,
        }))
    }

    /// `purge plan`: which fingerprinted values changed and what the sweep
    /// would delete now. It deletes nothing, and can run at any time.
    pub fn purge_plan(&self) -> Result<PurgePlan, StoreError> {
        let banks: Vec<i64> = self
            .bank_zones()?
            .into_iter()
            .map(|(bank_id, _, _)| bank_id)
            .collect();
        crate::sweep::plan(&self.store, &self.tuning, &self.purge_pause(), &banks)
    }

    /// `purge ack --hash`: acknowledges this daemon's deletion fingerprint,
    /// which `hash` must quote. Purging resumes at the next sweep, and the
    /// ack is stored, so it holds after a restart.
    pub fn purge_ack(&self, hash: &str) -> Result<(), PurgeError> {
        crate::sweep::ack(&self.store, &self.tuning, hash)?;
        *self.purge.lock().unwrap_or_else(|e| e.into_inner()) = PurgePause::Running;
        Ok(())
    }

    /// What a forget or an erase leaves in memory: live sessions lose the
    /// memories, models whose entries went are requested for a refresh, and
    /// the bank's block is cleared.
    fn settle(&self, bank_id: i64, aftermath: Aftermath) -> Result<(), StoreError> {
        let now = self.now();
        if !aftermath.scrub.is_empty() {
            self.sessions.scrub(bank_id, &aftermath.scrub, now);
        }
        if !aftermath.models.is_empty() {
            {
                let conn = self.store.connection();
                crate::mental_models::request(&conn, &self.schedule, &aftermath.models, now)?;
            }
            self.schedule.triggered(bank_id, now);
        }
        self.blocks.invalidate(bank_id);
        Ok(())
    }
}

/// Mental models, the agenda and the system prompt block.
impl Service {
    /// The model surface, `model create`. The enabled
    /// models' `max_tokens` must fit `mental_models.budget`. A new enabled
    /// model is refreshed after the debounce, like any owner edit.
    pub fn create_model(&self, bank: &str, spec: &ModelSpec) -> Result<Model, ModelError> {
        let (bank_id, _) = self.model_bank(bank)?;
        let row = crate::mental_models::create(
            &self.store,
            bank_id,
            spec,
            self.tuning.mental_models.budget,
        )?;
        if row.enabled {
            self.trigger(bank_id, row.id)?;
        }
        self.blocks.invalidate(bank_id);
        let conn = self.store.connection();
        Ok(crate::mental_models::model(&conn, &row)?)
    }

    /// An owner edit to a model (`model edit`). Changing the question, the
    /// filters or `max_tokens`, or enabling it, triggers a refresh;
    /// resizing and enabling are checked against the budget.
    pub fn edit_model(
        &self,
        bank: &str,
        name: &str,
        edit: &ModelEdit,
    ) -> Result<Model, ModelError> {
        let (bank_id, _) = self.model_bank(bank)?;
        let edited = crate::mental_models::edit(
            &self.store,
            bank_id,
            name,
            edit,
            self.tuning.mental_models.budget,
        )?;
        if edited.triggers {
            self.trigger(bank_id, edited.row.id)?;
        }
        self.blocks.invalidate(bank_id);
        let conn = self.store.connection();
        Ok(crate::mental_models::model(&conn, &edited.row)?)
    }

    /// Every model of the bank with its entries (`model list`).
    pub fn list_models(&self, bank: &str) -> Result<Vec<Model>, ModelError> {
        let (bank_id, _) = self.model_bank(bank)?;
        let conn = self.store.connection();
        crate::mental_models::load_models(&conn, bank_id)?
            .iter()
            .map(|row| crate::mental_models::model(&conn, row).map_err(ModelError::from))
            .collect()
    }

    /// What the next refresh of the model would send the LLM, and its
    /// fingerprint. Reads only, and writes no recall row.
    pub fn refresh_input(&self, bank: &str, name: &str) -> Result<RefreshInput, ModelError> {
        let row = self.model_row(bank, name)?;
        crate::mental_models::refresh_input(&self.retrieval()?, &row)
    }

    /// One refresh now, outside the schedule (`model refresh [--force]`).
    /// Without `force` it's skipped when the fingerprint matches the last
    /// completed refresh's.
    pub fn refresh_model(
        &self,
        bank: &str,
        name: &str,
        llm: &dyn LlmClient,
        force: bool,
    ) -> Result<RefreshOutcome, ModelError> {
        let _refreshing = self.refreshing.lock().unwrap_or_else(|e| e.into_inner());
        let row = self.model_row(bank, name)?;
        let outcome =
            crate::mental_models::refresh(&self.retrieval()?, &self.schedule, &row, llm, force)?;
        if matches!(outcome, RefreshOutcome::Applied(_)) {
            self.blocks.invalidate(row.bank_id);
        }
        Ok(outcome)
    }

    /// Runs every refresh due now, across banks, and says when the next is
    /// due: a model whose debounce or retry has come round, and every
    /// enabled model of a bank whose daily sweep has. `serve` calls it from
    /// a timer and the replay harness after advancing its clock. A model that
    /// fails to refresh doesn't stop the rest.
    pub fn run_refreshes(&self, llm: &dyn LlmClient) -> Result<Refreshes, ModelError> {
        let _refreshing = self.refreshing.lock().unwrap_or_else(|e| e.into_inner());
        let now = self.now();
        let tuning = &self.tuning.mental_models;
        let cx = self.retrieval()?;
        let mut ran = Vec::new();
        for (bank_id, bank, tz) in self.bank_zones()? {
            let sweep = self.schedule.next_sweep(bank_id, &tz, tuning.sweep_time) <= now;
            let models = {
                let conn = self.store.connection();
                crate::mental_models::load_models(&conn, bank_id)?
            };
            for model in models.into_iter().filter(|model| model.enabled) {
                let requested = self
                    .refresh_due(&model, bank_id)
                    .is_some_and(|due| due <= now);
                let run = requested
                    || (sweep && {
                        let held = crate::mental_models::schedule::not_before(&model)
                            .is_some_and(|floor| floor > now)
                            || self.schedule.held_until(model.id, now).is_some();
                        if held {
                            // Refreshed too recently: the sweep's check
                            // waits for the interval instead.
                            let conn = self.store.connection();
                            crate::mental_models::request(
                                &conn,
                                &self.schedule,
                                &[model.id].into(),
                                now,
                            )?;
                        }
                        !held
                    });
                if !run {
                    continue;
                }
                match crate::mental_models::refresh(&cx, &self.schedule, &model, llm, false) {
                    Ok(outcome) => {
                        if matches!(outcome, RefreshOutcome::Applied(_)) {
                            self.blocks.invalidate(bank_id);
                        }
                        ran.push(RefreshRun {
                            bank: bank.clone(),
                            model: model.name.clone(),
                            outcome,
                        });
                    }
                    Err(error) => {
                        tracing::warn!(bank = %bank, model = %model.uuid, %error,
                            "a mental model refresh failed in the store");
                    }
                }
            }
            if sweep {
                self.schedule.swept(bank_id, now);
            }
        }
        Ok(Refreshes {
            ran,
            next_due: self.next_refresh_due()?,
        })
    }

    /// When a requested refresh of `model` is due: on the schedule, and no
    /// earlier than a hold the LLM put on it lifts.
    fn refresh_due(
        &self,
        model: &crate::mental_models::ModelRow,
        bank_id: i64,
    ) -> Option<Timestamp> {
        let last_trigger = self.schedule.last_trigger(bank_id);
        let due =
            crate::mental_models::schedule::due(model, last_trigger, &self.tuning.mental_models)?;
        Some(match self.schedule.held_until(model.id, self.now()) {
            Some(until) => due.max(until),
            None => due,
        })
    }

    /// The earliest requested refresh or daily sweep still ahead.
    fn next_refresh_due(&self) -> Result<Option<Timestamp>, ModelError> {
        let tuning = &self.tuning.mental_models;
        let mut next: Option<Timestamp> = None;
        let mut earliest = |at: Timestamp| next = Some(next.map_or(at, |next| next.min(at)));
        for (bank_id, _, tz) in self.bank_zones()? {
            let models = {
                let conn = self.store.connection();
                crate::mental_models::load_models(&conn, bank_id)?
            };
            let mut any = false;
            for model in models.iter().filter(|model| model.enabled) {
                any = true;
                if let Some(due) = self.refresh_due(model, bank_id) {
                    earliest(due);
                }
            }
            if any {
                earliest(self.schedule.next_sweep(bank_id, &tz, tuning.sweep_time));
            }
        }
        Ok(next)
    }

    /// The bank's agenda now.
    pub fn agenda(&self, bank: &str) -> Result<Agenda, ModelError> {
        let (bank_id, tz) = self.model_bank(bank)?;
        let conn = self.store.connection();
        Ok(crate::agenda::build(&conn, &self.tuning, bank_id, &tz, self.now())?.agenda)
    }

    /// The bank's system prompt block, `GET /v1/banks/{bank}/system-prompt`:
    /// the cached one, or a new one built from the agenda and each enabled
    /// model's entries as of its last completed refresh. It never calls the
    /// LLM. With a session id, the daemon records which block the session
    /// holds, and what the block lists or cites joins the session's
    /// in-context set.
    pub fn system_prompt(&self, bank: &str, session: Option<&str>) -> Result<Block, ModelError> {
        let (bank_id, tz) = self.model_bank(bank)?;
        let now = self.now();
        let today = now.to_zoned(tz.clone()).date();
        let block = match self.blocks.get(bank_id, today) {
            Ok(block) => block,
            Err(generation) => {
                let block = crate::system_prompt::build(&self.store, &self.tuning, bank_id, &tz)?;
                self.blocks.put(bank_id, today, block.clone(), generation);
                block
            }
        };
        if let Some(session) = session.map(str::trim).filter(|session| !session.is_empty()) {
            {
                let conn = self.store.connection();
                crate::system_prompt::map_session(&conn, bank_id, session, &block, now)?;
            }
            self.sessions
                .add(bank_id, session, &block.in_context(), now);
        }
        Ok(block)
    }

    fn mapping_expiry(&self) -> SignedDuration {
        SignedDuration::from_hours(24 * i64::from(self.tuning.sessions.mapping_expiry_days))
    }

    /// Puts what the session's block listed and cited back into its
    /// in-context set, from the stored mapping, which outlives a restart.
    fn restore_block(&self, bank_id: i64, session_id: &str) -> Result<(), StoreError> {
        let now = self.now();
        let ids = {
            let conn = self.store.connection();
            crate::system_prompt::mapped(&conn, bank_id, session_id, now, self.mapping_expiry())?
                .unwrap_or_default()
        };
        if !ids.is_empty() {
            self.sessions.add(bank_id, session_id, &ids, now);
        }
        Ok(())
    }

    /// The block-id fallback: maps the session to the block the plugin
    /// holds, unless it already has a live mapping ([`crate::system_prompt::map_held_block`]).
    fn map_held_block(
        &self,
        bank_id: i64,
        session_id: &str,
        block: Uuid,
    ) -> Result<(), StoreError> {
        let now = self.now();
        let conn = self.store.connection();
        let expiry = self.mapping_expiry();
        if crate::system_prompt::mapped(&conn, bank_id, session_id, now, expiry)?.is_none()
            && !crate::system_prompt::map_held_block(&conn, bank_id, session_id, block, now)?
        {
            tracing::debug!(%block, "a prefetch named a block this bank never built");
        }
        Ok(())
    }

    /// The bank's rowid and edit-log high-water mark, when it exists.
    fn watermark_for(&self, bank: &str) -> Result<Option<(i64, i64)>, StoreError> {
        let conn = self.store.connection();
        let Some((bank_id, _)) = crate::ingest::find_bank(&conn, bank)? else {
            return Ok(None);
        };
        Ok(Some((
            bank_id,
            crate::mental_models::watermark(&conn, bank_id)?,
        )))
    }

    /// Triggers refreshes and clears the block for what was written to the
    /// bank since `watermark` ([`crate::mental_models::effects`]).
    fn after_writes(
        &self,
        bank_id: i64,
        watermark: i64,
        created: &[Uuid],
    ) -> Result<(), StoreError> {
        let now = self.now();
        let effects = {
            let conn = self.store.connection();
            let effects = crate::mental_models::effects(
                &conn,
                bank_id,
                watermark,
                created,
                self.tuning.mental_models.trigger_level,
            )?;
            crate::mental_models::request(&conn, &self.schedule, &effects.triggered, now)?;
            effects
        };
        if !effects.triggered.is_empty() {
            self.schedule.triggered(bank_id, now);
        }
        if effects.invalidates {
            self.blocks.invalidate(bank_id);
        }
        Ok(())
    }

    fn trigger(&self, bank_id: i64, model_id: i64) -> Result<(), StoreError> {
        let now = self.now();
        {
            let conn = self.store.connection();
            crate::mental_models::request(&conn, &self.schedule, &[model_id].into(), now)?;
        }
        self.schedule.triggered(bank_id, now);
        Ok(())
    }

    fn model_bank(&self, bank: &str) -> Result<(i64, TimeZone), ModelError> {
        let conn = self.store.connection();
        let (bank_id, timezone) =
            crate::ingest::find_bank(&conn, bank)?.ok_or(ModelError::UnknownBank)?;
        Ok((bank_id, TimeZone::get(&timezone).unwrap_or(TimeZone::UTC)))
    }

    fn model_row(
        &self,
        bank: &str,
        name: &str,
    ) -> Result<crate::mental_models::ModelRow, ModelError> {
        let (bank_id, _) = self.model_bank(bank)?;
        let conn = self.store.connection();
        crate::mental_models::find_model(&conn, bank_id, name)?.ok_or(ModelError::UnknownModel)
    }

    /// Every bank's rowid, name and timezone.
    fn bank_zones(&self) -> Result<Vec<(i64, String, TimeZone)>, StoreError> {
        let conn = self.store.connection();
        let mut statement = conn.prepare("SELECT id, name, timezone FROM banks ORDER BY id")?;
        let banks = statement
            .query_map([], |row| {
                let timezone: String = row.get(2)?;
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    TimeZone::get(&timezone).unwrap_or(TimeZone::UTC),
                ))
            })?
            .collect::<Result<_, _>>()?;
        Ok(banks)
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
/// one it was written for and warns on a mismatch.
#[derive(Debug, Clone, Serialize)]
pub struct Health {
    pub version: &'static str,
    pub ready: bool,
    pub now: Timestamp,
}
