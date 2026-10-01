//! Running the reranker under the deadline (TIM-93, decision 8, as amended
//! by TIM-109).
//!
//! The deadline is real latency, not world or bank time, so it's measured
//! with [`Instant`] rather than the service's clock: a replay that runs
//! years of bank time in minutes still has to answer Hermes in time. The
//! reranker runs on a thread of its own so the deadline can cut it off; a
//! late answer is dropped when it arrives.
//!
//! Inference can't be cancelled, and the ONNX reranker runs one call at a
//! time, so one call holds a [`Permit`] for as long as its inference runs.
//! Another caller waits for the permit only until its own deadline and then
//! falls back without starting anything, so timed-out calls never queue
//! work behind the one running.

use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::models::Reranker;

/// The right to run the reranker: one inference at a time. A service holds
/// one for its reranker, shared by prefetch and explicit recall.
#[derive(Debug, Default)]
pub(crate) struct Permit {
    busy: Mutex<bool>,
    freed: Condvar,
}

impl Permit {
    /// Takes the permit, waiting for it until `deadline` at most.
    fn take(self: &Arc<Self>, deadline: Instant) -> Option<Held> {
        let mut busy = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        while *busy {
            let remaining = deadline.checked_duration_since(Instant::now())?;
            busy = self
                .freed
                .wait_timeout(busy, remaining)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        *busy = true;
        Some(Held(Arc::clone(self)))
    }
}

/// A taken permit, given back when dropped.
struct Held(Arc<Permit>);

impl Drop for Held {
    fn drop(&mut self) {
        *self.0.busy.lock().unwrap_or_else(|e| e.into_inner()) = false;
        self.0.freed.notify_one();
    }
}

/// The reranker's logits for `documents` against `query`, in the
/// documents' order, or `None` when it fails, when `deadline` passes first,
/// or when another call holds `permit` until then.
pub(super) fn logits(
    reranker: &Arc<dyn Reranker>,
    permit: &Arc<Permit>,
    query: &str,
    documents: Vec<String>,
    deadline: Instant,
) -> Option<Vec<f32>> {
    if documents.is_empty() {
        return Some(Vec::new());
    }
    let expected = documents.len();
    let Some(held) = permit.take(deadline) else {
        tracing::debug!(
            candidates = expected,
            "the reranker was busy until the deadline"
        );
        return None;
    };
    let remaining = deadline.checked_duration_since(Instant::now())?;
    let (send, receive) = mpsc::sync_channel(1);
    let reranker = Arc::clone(reranker);
    let query = query.to_owned();
    std::thread::Builder::new()
        .name("asphodel-rerank".into())
        .spawn(move || {
            let documents: Vec<&str> = documents.iter().map(String::as_str).collect();
            let result = reranker.rerank(&query, &documents);
            // Free the reranker before answering, so a caller that got the
            // answer finds it free.
            drop(held);
            // The receiver is gone once the deadline has passed.
            let _ = send.send(result);
        })
        .ok()?;
    let Ok(result) = receive.recv_timeout(remaining) else {
        tracing::debug!(candidates = expected, "the reranker missed its deadline");
        return None;
    };
    match result {
        Ok(logits) if logits.len() == expected => Some(logits),
        Ok(logits) => {
            tracing::warn!(
                expected,
                got = logits.len(),
                "the reranker returned the wrong number of logits"
            );
            None
        }
        Err(error) => {
            tracing::warn!(%error, "the reranker failed");
            None
        }
    }
}
