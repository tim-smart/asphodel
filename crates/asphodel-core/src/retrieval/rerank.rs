//! Running the reranker under the deadline (TIM-93, decision 8, as amended
//! by TIM-109).
//!
//! The deadline is real latency, not world or bank time, so it's measured
//! with [`Instant`] rather than the service's clock: a replay that runs
//! years of bank time in minutes still has to answer Hermes in time. The
//! reranker runs on a thread of its own so the deadline can cut it off; a
//! late answer is dropped when it arrives.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Instant;

use crate::models::Reranker;

/// The reranker's logits for `documents` against `query`, in the
/// documents' order, or `None` when it fails or `deadline` passes first.
pub(super) fn logits(
    reranker: &Arc<dyn Reranker>,
    query: &str,
    documents: Vec<String>,
    deadline: Instant,
) -> Option<Vec<f32>> {
    if documents.is_empty() {
        return Some(Vec::new());
    }
    let expected = documents.len();
    let remaining = deadline.checked_duration_since(Instant::now())?;
    let (send, receive) = mpsc::sync_channel(1);
    let reranker = Arc::clone(reranker);
    let query = query.to_owned();
    std::thread::Builder::new()
        .name("asphodel-rerank".into())
        .spawn(move || {
            let documents: Vec<&str> = documents.iter().map(String::as_str).collect();
            // The receiver is gone once the deadline has passed.
            let _ = send.send(reranker.rerank(&query, &documents));
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
