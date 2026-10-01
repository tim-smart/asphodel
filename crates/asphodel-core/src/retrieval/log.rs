//! The recall log (TIM-90): one `recalls` row per recall, with the query,
//! the memories that came back, whether each was injected, the latency, the
//! session and the bank's turn counter. It's for detecting `used`, for the
//! replay harness and for "why did it bring that up?". Nothing here writes
//! an access: being recalled never counts towards strength (ADR 0001).

use jiff::Timestamp;
use rusqlite::Connection;
use uuid::Uuid;

use crate::store::micros;

/// `recalls.kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RecallKind {
    Prefetch,
    Tool,
}

impl RecallKind {
    fn as_str(self) -> &'static str {
        match self {
            RecallKind::Prefetch => "prefetch",
            RecallKind::Tool => "tool",
        }
    }
}

/// One `recall_results` row: a memory that came back, at its rank.
pub(super) struct Logged {
    pub memory_id: i64,
    /// The final score, or `None` when the reranker was skipped. It's not
    /// the reranker's logit: calibrating the gate floor (TIM-93, decision 9)
    /// recomputes logits from the logged query and memories, which the
    /// deterministic reranker reproduces exactly. Add a logit column only if
    /// calibration finds that impractical.
    pub score: Option<f64>,
    pub injected: bool,
}

/// One recall to log. `results` are in rank order, best first.
pub(super) struct Entry<'a> {
    pub uuid: Uuid,
    pub bank_id: i64,
    pub kind: RecallKind,
    pub session_id: Option<&'a str>,
    /// The query that ran, a short follow-up's borrowed context included.
    pub query: &'a str,
    pub latency_ms: u64,
    pub at: Timestamp,
    pub results: &'a [Logged],
}

pub(super) fn write(conn: &mut Connection, entry: &Entry<'_>) -> Result<(), rusqlite::Error> {
    let tx = conn.transaction()?;
    let turn: i64 = tx.query_row(
        "SELECT turns FROM banks WHERE id = ?1",
        [entry.bank_id],
        |row| row.get(0),
    )?;
    tx.execute(
        "INSERT INTO recalls (uuid, bank_id, kind, session_id, turn, query, latency_ms, at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        (
            entry.uuid.to_string(),
            entry.bank_id,
            entry.kind.as_str(),
            entry.session_id,
            turn,
            entry.query,
            i64::try_from(entry.latency_ms).unwrap_or(i64::MAX),
            micros(entry.at),
        ),
    )?;
    let recall_id = tx.last_insert_rowid();
    {
        let mut insert = tx.prepare_cached(
            "INSERT INTO recall_results (recall_id, memory_id, rank, score, injected)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for (index, result) in entry.results.iter().enumerate() {
            insert.execute((
                recall_id,
                result.memory_id,
                i64::try_from(index + 1).unwrap_or(i64::MAX),
                result.score,
                result.injected,
            ))?;
        }
    }
    tx.commit()
}
