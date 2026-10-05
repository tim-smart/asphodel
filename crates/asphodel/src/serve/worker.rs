//! The extraction workers: one per bank, each running up to `[llm]
//! concurrency` chunks at once, so banks don't wait on each other. At the
//! default of 1 a bank's chunks run one at a time in queue order.
//!
//! A worker claims the heads of its bank's queue while it has room, and
//! extracts each on a blocking thread. The chunks commit in the order they
//! were claimed, and one that another's commit made stale reconciles again
//! ([`asphodel_core::extraction`]). When the queue is empty, or the bank has
//! as many chunks out as it may, the worker sleeps until a chunk finishes or
//! an ingest or a retry wakes it. A held queue (no login, a usage limit, a
//! 429 that said when to retry) waits without counting anything, and a
//! counted failure waits a little before the retry when the LLM might
//! recover, so a dead endpoint doesn't burn through the retry cap in a
//! second.
//!
//! A forget's erase waits on the same queue, behind the chunks queued
//! before it, and nothing queued after it is claimed until it has run, so
//! each step runs a due erase first.
//!
//! On shutdown a worker finishes the chunks in flight and stops before
//! claiming another. Nothing queued is lost, since the queue is in SQLite.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use asphodel_core::Service;
use asphodel_core::extraction::{ExtractError, Extracted};
use asphodel_core::models::{LlmClient, LlmError};
use asphodel_core::queue::{Failure, QueueError};
use asphodel_core::service::Claimed;
use tokio::sync::{Notify, watch};
use tokio::task::JoinSet;
use tokio::time::Instant;
use tracing::{debug, info, warn};

/// How long a held queue waits before trying again, unless the LLM said
/// when its usage window resets.
const HELD_WAIT: Duration = Duration::from_secs(60);

/// The first wait after a counted failure the LLM might recover from. It
/// doubles with each failure of the same chunk, up to [`RETRY_WAIT_MAX`].
const RETRY_WAIT: Duration = Duration::from_secs(1);
const RETRY_WAIT_MAX: Duration = Duration::from_secs(60);

/// The wait after an error that isn't the chunk's, such as the store
/// failing to claim.
const ERROR_WAIT: Duration = Duration::from_secs(5);

/// The running workers, by bank name.
pub(crate) struct Workers {
    service: Arc<Service>,
    llm: Arc<dyn LlmClient>,
    stop: watch::Receiver<bool>,
    banks: Registry,
    tasks: Mutex<JoinSet<()>>,
}

/// Each running worker's wake-up, by bank name. A worker whose bank is
/// gone takes itself out, so a bank created again under the name gets a
/// new one.
type Registry = Arc<Mutex<BTreeMap<String, Arc<Notify>>>>;

impl Workers {
    /// Starts a worker for each of `banks`, the banks in the store, since
    /// the queue may hold chunks from before a restart.
    pub(crate) fn start(
        service: Arc<Service>,
        llm: Arc<dyn LlmClient>,
        banks: &[String],
        stop: watch::Receiver<bool>,
    ) -> Arc<Self> {
        let workers = Arc::new(Self {
            service,
            llm,
            stop,
            banks: Registry::default(),
            tasks: Mutex::new(JoinSet::new()),
        });
        for bank in banks {
            workers.wake(bank);
        }
        workers
    }

    /// Wakes `bank`'s worker, starting it if it isn't running. Called after
    /// anything that queues a chunk. A wake while the worker is busy isn't
    /// lost: it looks at the queue again before it next sleeps.
    pub(crate) fn wake(&self, bank: &str) {
        if *self.stop.borrow() {
            return;
        }
        let bank = bank.trim().to_string();
        let mut banks = self
            .banks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(notify) = banks.get(&bank) {
            notify.notify_one();
            return;
        }
        let notify = Arc::new(Notify::new());
        banks.insert(bank.clone(), Arc::clone(&notify));
        let worker = Worker {
            service: Arc::clone(&self.service),
            llm: Arc::clone(&self.llm),
            bank,
            notify,
            stop: self.stop.clone(),
            registry: Arc::clone(&self.banks),
        };
        self.tasks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .spawn(worker.run());
    }

    /// Waits for every worker to finish its chunk in flight and stop. The
    /// caller has already signalled the stop.
    pub(crate) async fn join(&self) {
        let mut tasks = std::mem::take(
            &mut *self
                .tasks
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
        while let Some(result) = tasks.join_next().await {
            if let Err(error) = result {
                warn!(%error, "an extraction worker panicked");
            }
        }
    }
}

struct Worker {
    service: Arc<Service>,
    llm: Arc<dyn LlmClient>,
    bank: String,
    notify: Arc<Notify>,
    stop: watch::Receiver<bool>,
    registry: Registry,
}

/// What one blocking claim step did.
enum Step {
    /// An erase ran, deleting this many memories.
    Erased(usize),
    EraseFailed(QueueError),
    /// No erase was due, so the head chunk was claimed, or not.
    Claimed(Result<Option<Claimed>, ExtractError>),
}

/// What a worker does after a chunk's extraction ends.
enum Next {
    /// Claim again at once.
    Continue,
    /// Claim nothing for this long, or until stopped.
    Wait(Duration),
    /// The bank is gone.
    Exit,
}

type Extraction = Result<Extracted, ExtractError>;

impl Worker {
    async fn run(self) {
        debug!(bank = %self.bank, "extraction worker started");
        let limit = self.service.tuning().llm.concurrency.max(1) as usize;
        let mut in_flight: JoinSet<Extraction> = JoinSet::new();
        // No claims before this, after a held queue or a counted failure.
        let mut resume: Option<Instant> = None;
        // The queue had nothing to claim at the last look.
        let mut drained = false;
        let mut exit = false;
        loop {
            let stopping = *self.stop.borrow() || exit;
            if stopping && in_flight.is_empty() {
                break;
            }
            let waiting = resume.is_some_and(|at| Instant::now() < at);
            if !stopping && !waiting && !drained && in_flight.len() < limit {
                match self.claim().await {
                    Ok(Step::Erased(memories)) => {
                        info!(bank = %self.bank, memories, "erased a forgotten chain");
                        continue;
                    }
                    Ok(Step::EraseFailed(QueueError::UnknownBank)) => exit = true,
                    Ok(Step::EraseFailed(error)) => {
                        warn!(bank = %self.bank, %error, "an erase failed");
                        resume = Some(after(ERROR_WAIT));
                    }
                    Ok(Step::Claimed(Ok(Some(claimed)))) => {
                        let service = Arc::clone(&self.service);
                        let llm = Arc::clone(&self.llm);
                        // The blocking extraction always runs to the end:
                        // shutdown waits for the chunks in flight rather than
                        // abandoning them.
                        in_flight.spawn_blocking(move || {
                            let Claimed { lease, in_context } = claimed;
                            service.extract_chunk(lease, llm.as_ref(), &in_context)
                        });
                        continue;
                    }
                    Ok(Step::Claimed(Ok(None))) => drained = true,
                    Ok(Step::Claimed(Err(error))) => match self.next(Err(error)) {
                        Next::Continue => {}
                        Next::Wait(wait) => resume = Some(after(wait)),
                        Next::Exit => exit = true,
                    },
                    Err(error) => {
                        warn!(bank = %self.bank, %error, "an extraction step panicked");
                        resume = Some(after(ERROR_WAIT));
                    }
                }
                continue;
            }
            let sleep_until = resume.filter(|_| waiting);
            tokio::select! {
                Some(done) = in_flight.join_next(), if !in_flight.is_empty() => {
                    drained = false;
                    let next = match done {
                        Ok(result) => self.next(result),
                        Err(error) => {
                            warn!(bank = %self.bank, %error, "an extraction step panicked");
                            Next::Wait(ERROR_WAIT)
                        }
                    };
                    match next {
                        Next::Continue => {}
                        Next::Wait(wait) => {
                            let at = after(wait);
                            resume = Some(resume.map_or(at, |resume| resume.max(at)));
                        }
                        Next::Exit => exit = true,
                    }
                }
                _ = self.notify.notified(), if !stopping => drained = false,
                _ = tokio::time::sleep_until(sleep_until.unwrap_or_else(Instant::now)),
                    if !stopping && sleep_until.is_some() => resume = None,
                _ = wait_for_stop(self.stop.clone()), if !stopping => {}
            }
        }
        if exit {
            let mut banks = self
                .registry
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if banks
                .get(&self.bank)
                .is_some_and(|notify| Arc::ptr_eq(notify, &self.notify))
            {
                banks.remove(&self.bank);
            }
        }
        debug!(bank = %self.bank, "extraction worker stopped");
    }

    /// Runs a due erase, or claims the next chunk, on a blocking thread.
    async fn claim(&self) -> Result<Step, tokio::task::JoinError> {
        let service = Arc::clone(&self.service);
        let bank = self.bank.clone();
        tokio::task::spawn_blocking(move || match service.erase_next(&bank) {
            Ok(Some(erased)) => Step::Erased(erased.memories.len()),
            Ok(None) => Step::Claimed(service.next_extraction(&bank)),
            Err(error) => Step::EraseFailed(error),
        })
        .await
    }

    fn next(&self, result: Extraction) -> Next {
        let error = match result {
            Ok(extracted) => {
                info!(
                    bank = %self.bank,
                    chunk = %extracted.chunk,
                    memories = extracted.memories.len(),
                    used = extracted.used.len(),
                    "chunk extracted"
                );
                return Next::Continue;
            }
            Err(error) => error,
        };
        match &error {
            ExtractError::Held { error } => {
                let wait = match error {
                    LlmError::UsageLimited { resets_at } => {
                        let now = self.service.now();
                        Duration::try_from(now.duration_until(*resets_at))
                            .unwrap_or(Duration::ZERO)
                            .max(RETRY_WAIT)
                    }
                    LlmError::RateLimited { retry_after } => (*retry_after).max(RETRY_WAIT),
                    _ => HELD_WAIT,
                };
                Next::Wait(wait)
            }
            ExtractError::Call1 { error, failure } | ExtractError::Call2 { error, failure } => {
                if error.is_retryable() {
                    Next::Wait(retry_wait(*failure))
                } else {
                    Next::Continue
                }
            }
            ExtractError::InvalidReply { .. } => Next::Continue,
            // The document went while the chunk was out; it's off the queue.
            ExtractError::SourceRemoved => Next::Continue,
            ExtractError::Search { failure, .. }
            | ExtractError::Embedding { failure, .. }
            | ExtractError::Commit { failure, .. } => {
                warn!(bank = %self.bank, %error, "extraction failed");
                Next::Wait(retry_wait(*failure))
            }
            ExtractError::Queue(QueueError::UnknownBank) => Next::Exit,
            // Nothing was counted: the queue holds until a re-embed moves
            // the bank to a model the daemon carries, which wakes it.
            ExtractError::ModelUnavailable { .. } => {
                warn!(bank = %self.bank, %error, "extraction waits for a re-embed");
                Next::Wait(HELD_WAIT)
            }
            _ => {
                warn!(bank = %self.bank, %error, "extraction failed");
                Next::Wait(ERROR_WAIT)
            }
        }
    }
}

/// `wait` from now, or a year from now when that's further than the clock
/// can say.
fn after(wait: Duration) -> Instant {
    let now = Instant::now();
    now.checked_add(wait)
        .unwrap_or_else(|| now + Duration::from_secs(365 * 24 * 60 * 60))
}

/// Resolves once `stop` is set.
async fn wait_for_stop(mut stop: watch::Receiver<bool>) {
    let _ = stop.wait_for(|stop| *stop).await;
}

/// The wait before retrying a chunk after a counted failure: none once it
/// has failed for good, since the next chunk is a different one.
fn retry_wait(failure: Failure) -> Duration {
    match failure {
        Failure::Failed => Duration::ZERO,
        Failure::Retry { error_count } => {
            let doublings = error_count.saturating_sub(1).min(16);
            RETRY_WAIT
                .saturating_mul(1 << doublings)
                .min(RETRY_WAIT_MAX)
        }
    }
}
