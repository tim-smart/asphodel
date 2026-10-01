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

use jiff::Timestamp;
use rusqlite::OptionalExtension;
use serde::Serialize;
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
fn check_held(leases: &Leases, lease: &Lease) -> Result<(), QueueError> {
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
    tx.execute(
        "UPDATE chunks SET extracted_at = ?1, call1_output = NULL WHERE id = ?2",
        (micros(store.now()), lease.chunk_id),
    )?;
    tx.execute(
        "DELETE FROM extraction_queue WHERE id = ?1",
        [lease.queue_id],
    )?;
    tx.commit()?;
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
