//! The durable extraction queue, in the `extraction_queue` table so nothing
//! queued is lost on SIGTERM or a crash (TIM-94, decision 3).
//!
//! Each bank has one worker (TIM-92): while a bank has a [`Lease`] out, it
//! hands out no other. Chunks run with turns ahead of document chunks, then
//! in `observed_at` order, ties broken by the later `ingested_at` and then by
//! rowid (TIM-92). A failed attempt is counted on the chunk and the chunk is
//! retried in place, so nothing behind it overtakes it; at
//! [`CHUNK_RETRY_CAP`] it's marked failed, leaves the queue and is surfaced
//! instead of retried.
//!
//! Leases live in memory, not the store. A restart releases them, and the
//! chunk that was in flight is handed out first again without counting an
//! error, while the error count itself is stored, so the cap holds across
//! restarts. Dropping a lease without completing or failing it releases it
//! too, so a worker that panics doesn't stall its bank.
//!
//! Only `chunk` jobs are handed out here. The `erase` jobs that wait behind
//! them belong to the erase path (ADR 0010).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use jiff::Timestamp;
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::constants::CHUNK_RETRY_CAP;
use crate::ingest::find_bank;
use crate::store::{Store, StoreError, micros, timestamp};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    Turn,
    Document,
}

/// The leases out, by bank rowid: the queue row each bank's worker holds.
#[derive(Debug, Clone, Default)]
pub(crate) struct Leases(Arc<Mutex<BTreeMap<i64, i64>>>);

impl Leases {
    fn lock(&self) -> MutexGuard<'_, BTreeMap<i64, i64>> {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn release(&self, bank_id: i64, queue_id: i64) {
        let mut held = self.lock();
        if held.get(&bank_id) == Some(&queue_id) {
            held.remove(&bank_id);
        }
    }

    /// Takes the bank's lease without a chunk, waiting up to `wait` for its
    /// worker to finish the chunk in flight, so no extraction runs on the
    /// bank until the hold drops. A re-embed's swap and a bank deletion
    /// hold it (ADR 0010). `None` when the wait ran out.
    pub(crate) fn hold(&self, bank_id: i64, wait: Duration) -> Option<BankHold> {
        let started = Instant::now();
        loop {
            {
                let mut held = self.lock();
                if let std::collections::btree_map::Entry::Vacant(entry) = held.entry(bank_id) {
                    entry.insert(HOLD);
                    return Some(BankHold {
                        leases: self.clone(),
                        bank_id,
                    });
                }
            }
            if started.elapsed() >= wait {
                return None;
            }
            std::thread::sleep(HOLD_POLL);
        }
    }
}

/// The queue row a [`BankHold`] stands in for. Queue rowids start at 1.
const HOLD: i64 = 0;

/// How often [`Leases::hold`] looks again while a chunk is in flight.
const HOLD_POLL: Duration = Duration::from_millis(50);

/// A bank's lease held without a chunk; dropping it releases the bank.
#[derive(Debug)]
pub(crate) struct BankHold {
    leases: Leases,
    bank_id: i64,
}

impl Drop for BankHold {
    fn drop(&mut self) {
        self.leases.release(self.bank_id, HOLD);
    }
}

/// A chunk handed to its bank's one worker. Complete it with
/// [`complete_chunk`] or fail it with [`fail_chunk`]; dropping it releases
/// the bank without changing the chunk.
#[derive(Debug)]
pub struct Lease {
    pub chunk: Uuid,
    pub source: Uuid,
    pub source_kind: SourceKind,
    /// The chunk's position in its source.
    pub position: u32,
    pub observed_at: Timestamp,
    /// Failed attempts so far, across restarts.
    pub error_count: u32,
    bank_id: i64,
    queue_id: i64,
    chunk_id: i64,
    leases: Leases,
}

impl Lease {
    /// The bank's rowid.
    pub(crate) fn bank_id(&self) -> i64 {
        self.bank_id
    }

    /// The chunk's rowid.
    pub(crate) fn chunk_id(&self) -> i64 {
        self.chunk_id
    }
}

impl PartialEq for Lease {
    fn eq(&self, other: &Self) -> bool {
        self.queue_id == other.queue_id && self.chunk == other.chunk
    }
}

impl Eq for Lease {}

impl Drop for Lease {
    fn drop(&mut self) {
        self.leases.release(self.bank_id, self.queue_id);
    }
}

/// Why an attempt failed: the error kind and HTTP status, never the response
/// (ADR 0010). `kind` is static so it can't carry content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ChunkError {
    pub kind: &'static str,
    pub status: Option<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Failure {
    /// The chunk stays at its place in the queue and is retried.
    Retry { error_count: u32 },
    /// The cap was reached: the chunk is marked failed and leaves the queue.
    Failed,
}

/// A chunk that reached the retry cap, as `chunks --failed` lists it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FailedChunk {
    pub chunk: Uuid,
    pub source: Uuid,
    pub error_count: u32,
    pub error_kind: String,
    pub status: Option<u16>,
    pub failed_at: Timestamp,
}

#[derive(Debug, thiserror::Error)]
pub enum QueueError {
    #[error("unknown bank")]
    UnknownBank,
    /// The lease isn't the one its bank's worker holds in this service, such
    /// as a lease from before a restart or from another store.
    #[error("the lease on chunk {chunk} is no longer held")]
    NotHeld { chunk: Uuid },
    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for QueueError {
    fn from(error: rusqlite::Error) -> Self {
        QueueError::Store(StoreError::Sqlite(error))
    }
}

fn bank_id(conn: &rusqlite::Connection, bank: &str) -> Result<i64, QueueError> {
    let (bank_id, _) = find_bank(conn, bank)?.ok_or(QueueError::UnknownBank)?;
    Ok(bank_id)
}

/// The head of the bank's queue, or `None` when the queue is empty or the
/// bank's worker already holds a lease.
pub(crate) fn claim(
    store: &Store,
    leases: &Leases,
    bank: &str,
) -> Result<Option<Lease>, QueueError> {
    let conn = store.connection();
    let bank_id = bank_id(&conn, bank)?;
    let mut held = leases.lock();
    if held.contains_key(&bank_id) {
        return Ok(None);
    }
    let head = conn
        .query_row(
            "SELECT q.id, c.id, c.uuid, s.uuid, s.kind, c.position, q.observed_at, c.error_count
             FROM extraction_queue q
             JOIN chunks c ON c.id = q.chunk_id
             JOIN sources s ON s.id = c.source_id
             WHERE q.bank_id = ?1 AND q.kind = 'chunk'
             ORDER BY q.priority, q.observed_at, s.ingested_at, q.id
             LIMIT 1",
            [bank_id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                ))
            },
        )
        .optional()?;
    let Some((queue_id, chunk_id, chunk, source, kind, position, observed_at, error_count)) = head
    else {
        return Ok(None);
    };
    held.insert(bank_id, queue_id);
    Ok(Some(Lease {
        chunk: chunk.parse().expect("a stored uuid parses"),
        source: source.parse().expect("a stored uuid parses"),
        source_kind: if kind == "turn" {
            SourceKind::Turn
        } else {
            SourceKind::Document
        },
        position: u32::try_from(position).unwrap_or(u32::MAX),
        observed_at: timestamp(observed_at),
        error_count: u32::try_from(error_count).unwrap_or(u32::MAX),
        bank_id,
        queue_id,
        chunk_id,
        leases: leases.clone(),
    }))
}

/// Checks that `lease` is the one its bank's worker holds in this service.
/// Rowids alone don't identify a lease: after a restart the same chunk is
/// claimed again under the same rowids, and another store's rowids can
/// coincide. So the lease must also come from this service's registry.
pub(crate) fn check_held(leases: &Leases, lease: &Lease) -> Result<(), QueueError> {
    if Arc::ptr_eq(&leases.0, &lease.leases.0)
        && leases.lock().get(&lease.bank_id) == Some(&lease.queue_id)
    {
        Ok(())
    } else {
        Err(QueueError::NotHeld { chunk: lease.chunk })
    }
}

/// Marks the chunk extracted, drops call 1's saved output (ADR 0008) and
/// takes it off the queue. The lease is released when it drops.
pub(crate) fn complete(store: &Store, leases: &Leases, lease: Lease) -> Result<(), QueueError> {
    check_held(leases, &lease)?;
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    finish(&tx, store.now(), &lease)?;
    tx.commit()?;
    Ok(())
}

/// [`complete`]'s writes, inside the caller's transaction, so extraction
/// commits its memories together with `extracted_at` (TIM-92). A turn's
/// stored in-context set goes too: it was only kept for this extraction.
/// The caller has checked the lease is held.
pub(crate) fn finish(
    tx: &rusqlite::Transaction<'_>,
    now: Timestamp,
    lease: &Lease,
) -> Result<(), rusqlite::Error> {
    tx.execute(
        "UPDATE chunks SET extracted_at = ?1, call1_output = NULL WHERE id = ?2",
        (micros(now), lease.chunk_id),
    )?;
    if lease.source_kind == SourceKind::Turn {
        tx.execute(
            "DELETE FROM turn_in_context
             WHERE source_id = (SELECT source_id FROM chunks WHERE id = ?1)",
            [lease.chunk_id],
        )?;
        tx.execute(
            "DELETE FROM turn_entries
             WHERE source_id = (SELECT source_id FROM chunks WHERE id = ?1)",
            [lease.chunk_id],
        )?;
    }
    tx.execute(
        "DELETE FROM extraction_queue WHERE id = ?1",
        [lease.queue_id],
    )?;
    Ok(())
}

/// Counts a failed attempt. Below the cap the chunk stays at the head of
/// its bank's queue; at the cap it's marked failed and leaves the queue.
pub(crate) fn fail(
    store: &Store,
    leases: &Leases,
    lease: Lease,
    error: ChunkError,
) -> Result<Failure, QueueError> {
    check_held(leases, &lease)?;
    let now = micros(store.now());
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let error_count: i64 = tx.query_row(
        "UPDATE chunks SET error_count = error_count + 1, last_error_kind = ?1,
                           last_error_status = ?2
         WHERE id = ?3
         RETURNING error_count",
        (error.kind, error.status, lease.chunk_id),
        |row| row.get(0),
    )?;
    let error_count = u32::try_from(error_count).unwrap_or(u32::MAX);
    let failure = if error_count >= CHUNK_RETRY_CAP {
        tx.execute(
            "UPDATE chunks SET failed_at = ?1 WHERE id = ?2",
            (now, lease.chunk_id),
        )?;
        tx.execute(
            "DELETE FROM extraction_queue WHERE id = ?1",
            [lease.queue_id],
        )?;
        Failure::Failed
    } else {
        tx.execute(
            "UPDATE extraction_queue SET attempts = attempts + 1 WHERE id = ?1",
            [lease.queue_id],
        )?;
        Failure::Retry { error_count }
    };
    tx.commit()?;
    tracing::warn!(
        chunk = %lease.chunk,
        kind = error.kind,
        status = error.status,
        error_count,
        failed = failure == Failure::Failed,
        "a chunk's extraction failed"
    );
    Ok(failure)
}

/// Chunks waiting or in flight in `bank`, not counting failed ones.
pub(crate) fn depth(store: &Store, bank: &str) -> Result<usize, QueueError> {
    let conn = store.connection();
    let bank_id = bank_id(&conn, bank)?;
    let depth: i64 = conn.query_row(
        "SELECT COUNT(*) FROM extraction_queue WHERE bank_id = ?1 AND kind = 'chunk'",
        [bank_id],
        |row| row.get(0),
    )?;
    Ok(usize::try_from(depth).unwrap_or(0))
}

/// The bank's failed chunks, oldest failure first.
pub(crate) fn failed(store: &Store, bank: &str) -> Result<Vec<FailedChunk>, QueueError> {
    let conn = store.connection();
    let bank_id = bank_id(&conn, bank)?;
    let mut statement = conn.prepare(
        "SELECT c.uuid, s.uuid, c.error_count, c.last_error_kind, c.last_error_status, c.failed_at
         FROM chunks c JOIN sources s ON s.id = c.source_id
         WHERE c.bank_id = ?1 AND c.failed_at IS NOT NULL
         ORDER BY c.failed_at, c.id",
    )?;
    let rows = statement.query_map([bank_id], |row| {
        Ok(FailedChunk {
            chunk: row
                .get::<_, String>(0)?
                .parse()
                .expect("a stored uuid parses"),
            source: row
                .get::<_, String>(1)?
                .parse()
                .expect("a stored uuid parses"),
            error_count: u32::try_from(row.get::<_, i64>(2)?).unwrap_or(u32::MAX),
            error_kind: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
            status: row.get(4)?,
            failed_at: timestamp(row.get(5)?),
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// A chunk on the queue, waiting or in flight, as `chunks` lists it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QueuedChunk {
    pub chunk: Uuid,
    pub source: Uuid,
    pub source_kind: SourceKind,
    /// The chunk's position in its source.
    pub position: u32,
    pub observed_at: Timestamp,
    /// Failed attempts so far, below the retry cap.
    pub error_count: u32,
    /// Whether the bank's worker holds it now.
    pub in_flight: bool,
}

/// What `chunks` lists: the bank's queue in the order it runs, and its
/// failed chunks, oldest failure first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChunkList {
    pub queued: Vec<QueuedChunk>,
    pub failed: Vec<FailedChunk>,
}

/// What `chunks/retry` takes: the failed chunks to put back on the queue,
/// or every failed chunk in the bank when `chunks` is absent.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RetryRequest {
    #[serde(default)]
    pub chunks: Option<Vec<Uuid>>,
}

/// What a retry did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Retried {
    /// The failed chunks put back on the queue.
    pub retried: Vec<Uuid>,
    /// Asked-for chunks that aren't failed chunks of the bank.
    pub unknown: Vec<Uuid>,
}

/// The bank's queue in the order [`claim`] hands it out.
pub(crate) fn queued(
    store: &Store,
    leases: &Leases,
    bank: &str,
) -> Result<Vec<QueuedChunk>, QueueError> {
    let conn = store.connection();
    let bank_id = bank_id(&conn, bank)?;
    let held = leases.lock().get(&bank_id).copied();
    let mut statement = conn.prepare(
        "SELECT q.id, c.uuid, s.uuid, s.kind, c.position, q.observed_at, c.error_count
         FROM extraction_queue q
         JOIN chunks c ON c.id = q.chunk_id
         JOIN sources s ON s.id = c.source_id
         WHERE q.bank_id = ?1 AND q.kind = 'chunk'
         ORDER BY q.priority, q.observed_at, s.ingested_at, q.id",
    )?;
    let rows = statement.query_map([bank_id], |row| {
        let queue_id: i64 = row.get(0)?;
        Ok(QueuedChunk {
            chunk: row
                .get::<_, String>(1)?
                .parse()
                .expect("a stored uuid parses"),
            source: row
                .get::<_, String>(2)?
                .parse()
                .expect("a stored uuid parses"),
            source_kind: if row.get::<_, String>(3)? == "turn" {
                SourceKind::Turn
            } else {
                SourceKind::Document
            },
            position: u32::try_from(row.get::<_, i64>(4)?).unwrap_or(u32::MAX),
            observed_at: timestamp(row.get(5)?),
            error_count: u32::try_from(row.get::<_, i64>(6)?).unwrap_or(u32::MAX),
            in_flight: held == Some(queue_id),
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Puts failed chunks back on their bank's queue with a fresh error count,
/// at the place their source's time gives them. `chunks` names the ones to
/// retry; `None` retries every failed chunk in the bank. A chunk whose text
/// is gone, swept or erased, can't be extracted again and counts as
/// unknown. Call 1's saved reply, if any, stays, so a chunk that failed in
/// call 2 resumes there.
pub(crate) fn retry(
    store: &Store,
    bank: &str,
    chunks: Option<&[Uuid]>,
) -> Result<Retried, QueueError> {
    let now = micros(store.now());
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let bank_id = bank_id(&tx, bank)?;
    let failed: Vec<(i64, Uuid, String, i64)> = {
        let mut statement = tx.prepare(
            "SELECT c.id, c.uuid, s.kind, s.observed_at
             FROM chunks c JOIN sources s ON s.id = c.source_id
             WHERE c.bank_id = ?1 AND c.failed_at IS NOT NULL AND c.text IS NOT NULL
               AND c.tombstoned_at IS NULL
             ORDER BY c.failed_at, c.id",
        )?;
        let rows = statement.query_map([bank_id], |row| {
            Ok((
                row.get(0)?,
                row.get::<_, String>(1)?
                    .parse()
                    .expect("a stored uuid parses"),
                row.get(2)?,
                row.get(3)?,
            ))
        })?;
        rows.collect::<Result<_, _>>()?
    };
    let wanted: Option<std::collections::BTreeSet<Uuid>> =
        chunks.map(|chunks| chunks.iter().copied().collect());
    let mut retried = Vec::new();
    for (chunk_id, chunk, kind, observed_at) in failed {
        if wanted
            .as_ref()
            .is_some_and(|wanted| !wanted.contains(&chunk))
        {
            continue;
        }
        let priority = if kind == "turn" {
            crate::ingest::PRIORITY_TURN
        } else {
            crate::ingest::PRIORITY_DOCUMENT
        };
        tx.execute(
            "UPDATE chunks SET failed_at = NULL, error_count = 0 WHERE id = ?1",
            [chunk_id],
        )?;
        tx.execute(
            "INSERT INTO extraction_queue (bank_id, kind, chunk_id, priority, observed_at, enqueued_at)
             VALUES (?1, 'chunk', ?2, ?3, ?4, ?5)",
            (bank_id, chunk_id, priority, observed_at, now),
        )?;
        retried.push(chunk);
    }
    tx.commit()?;
    let unknown = match chunks {
        None => Vec::new(),
        Some(chunks) => {
            let mut seen = std::collections::BTreeSet::new();
            chunks
                .iter()
                .copied()
                .filter(|chunk| !retried.contains(chunk) && seen.insert(*chunk))
                .collect()
        }
    };
    if !retried.is_empty() {
        tracing::info!(count = retried.len(), "failed chunks put back on the queue");
    }
    Ok(Retried { retried, unknown })
}
