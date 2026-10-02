//! The extraction workers: one per bank, so each bank's chunks run one at a
//! time in queue order (TIM-92), and banks don't wait on each other.
//!
//! A worker takes the head of its bank's queue, extracts it on a blocking
//! thread and moves on to the next. When the queue is empty it sleeps until
//! an ingest or a retry wakes it. A held queue (no login, a usage limit)
//! waits without counting anything, and a counted failure waits a little
//! before the retry when the LLM might recover, so a dead endpoint doesn't
//! burn through the retry cap in a second.
//!
//! A forget's erase waits on the same queue, behind the chunks queued
//! before it (ADR 0010), so each step runs a due erase first.
//!
//! On shutdown a worker finishes the chunk in flight and stops before
//! claiming another (TIM-94, decision 3). Nothing queued is lost, since the
//! queue is in SQLite.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use asphodel_core::Service;
use asphodel_core::extraction::ExtractError;
use asphodel_core::models::{LlmClient, LlmError};
use asphodel_core::queue::{Failure, QueueError};
use tokio::sync::{Notify, watch};
use tokio::task::JoinSet;
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
    banks: Mutex<BTreeMap<String, Arc<Notify>>>,
    tasks: Mutex<JoinSet<()>>,
}

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
            banks: Mutex::default(),
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
}

/// What one blocking step did.
enum Step {
    /// An erase ran, deleting this many memories.
    Erased(usize),
    EraseFailed(QueueError),
    /// No erase was due, so the head chunk was extracted, or not.
    Extracted(Result<Option<asphodel_core::extraction::Extracted>, ExtractError>),
}

/// What a worker does after one step.
enum Next {
    /// Take the next chunk at once.
    Continue,
    /// Sleep until woken: the queue is empty.
    Idle,
    /// Sleep this long, or until stopped.
    Wait(Duration),
    /// The bank is gone.
    Exit,
}

impl Worker {
    async fn run(mut self) {
        debug!(bank = %self.bank, "extraction worker started");
        loop {
            if *self.stop.borrow() {
                break;
            }
            let service = Arc::clone(&self.service);
            let llm = Arc::clone(&self.llm);
            let bank = self.bank.clone();
            // The blocking step always runs to the end: shutdown waits for
            // the chunk in flight rather than abandoning it.
            let step = tokio::task::spawn_blocking(move || match service.erase_next(&bank) {
                Ok(Some(erased)) => Step::Erased(erased.memories.len()),
                Ok(None) => Step::Extracted(service.extract_next(&bank, llm.as_ref())),
                Err(error) => Step::EraseFailed(error),
            })
            .await;
            let next = match step {
                Ok(Step::Erased(memories)) => {
                    info!(bank = %self.bank, memories, "erased a forgotten chain");
                    Next::Continue
                }
                Ok(Step::EraseFailed(QueueError::UnknownBank)) => Next::Exit,
                Ok(Step::EraseFailed(error)) => {
                    warn!(bank = %self.bank, %error, "an erase failed");
                    Next::Wait(ERROR_WAIT)
                }
                Ok(Step::Extracted(result)) => self.next(result),
                Err(error) => {
                    warn!(bank = %self.bank, %error, "an extraction step panicked");
                    Next::Wait(ERROR_WAIT)
                }
            };
            match next {
                Next::Continue => {}
                Next::Exit => break,
                Next::Idle => {
                    tokio::select! {
                        _ = self.notify.notified() => {}
                        _ = self.stop.wait_for(|stop| *stop) => break,
                    }
                }
                Next::Wait(wait) => {
                    tokio::select! {
                        _ = tokio::time::sleep(wait) => {}
                        _ = self.stop.wait_for(|stop| *stop) => break,
                    }
                }
            }
        }
        debug!(bank = %self.bank, "extraction worker stopped");
    }

    fn next(
        &self,
        result: Result<Option<asphodel_core::extraction::Extracted>, ExtractError>,
    ) -> Next {
        let error = match result {
            Ok(Some(extracted)) => {
                info!(
                    bank = %self.bank,
                    chunk = %extracted.chunk,
                    memories = extracted.memories.len(),
                    used = extracted.used.len(),
                    "chunk extracted"
                );
                return Next::Continue;
            }
            Ok(None) => return Next::Idle,
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
            ExtractError::Search { failure, .. }
            | ExtractError::Embedding { failure, .. }
            | ExtractError::Commit { failure, .. } => {
                warn!(bank = %self.bank, %error, "extraction failed");
                Next::Wait(retry_wait(*failure))
            }
            ExtractError::Queue(QueueError::UnknownBank) => Next::Exit,
            _ => {
                warn!(bank = %self.bank, %error, "extraction failed");
                Next::Wait(ERROR_WAIT)
            }
        }
    }
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
