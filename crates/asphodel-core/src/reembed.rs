//! Re-embedding a bank (ADR 0010, "Consequences").
//!
//! A bank records the embedding model it was created under, and is served
//! with that model, not the daemon's, until a re-embed swaps it. So the daemon
//! carries both models during a change
//! ([`Service::with_previous_embedder`]).
//!
//! `asphodel reembed --bank` is a daemon job. It embeds the bank's
//! memories with the daemon's model, in rowid order and a batch at a time,
//! into the `reembed_vectors` side table, and records the last rowid it
//! reached, so a job a restart stopped resumes there. Extraction carries on
//! meanwhile with the recorded model, and its new memories come later in
//! rowid order, so the job reaches them too. The swap holds the bank's
//! lease, so no chunk is in flight, embeds whatever the job hasn't reached
//! yet, and in one transaction replaces the bank's vectors with the side
//! table's, records the new model, and deletes the job and its side rows.
//! A memory erased during the job takes its staged vector with it in the
//! erase's transaction, and a batch embedded while its memories were
//! erased, or its bank deleted, stages only what's still there.
//!
//! The job and its side table are the schema's version 10 migration.
//!
//! A bank whose recorded model the daemon doesn't carry is refused recall
//! and extraction rather than served with another model, whose vectors
//! aren't comparable with its own. A re-embed is how it recovers: it needs
//! only the daemon's model, and once it swaps the bank is served again.
//!
//! The new model's reconcile floor has to be in the tuning before the
//! deploy, calibrated in replay; the daemon won't start without it.
//!
//! [`Service::with_previous_embedder`]: crate::Service::with_previous_embedder

use jiff::Timestamp;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::ingest::find_bank;
use crate::models::{Embedder, ModelError};
use crate::queue::BankHold;
use crate::store::{Store, StoreError, VectorIndex, micros, timestamp};

/// How many memories one step embeds.
pub const REEMBED_BATCH: usize = 32;

/// The edit kind the swap writes.
pub const EDIT_REEMBEDDED: &str = "reembedded";

/// Where a bank's re-embed stands, as `GET /v1/banks/{bank}/reembed`
/// shows it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReembedStatus {
    pub bank: String,
    /// The model the bank is served with until the swap.
    pub recorded_model: String,
    /// The daemon's embedding model, which a re-embed moves the bank to.
    pub model: String,
    pub state: ReembedState,
    /// Memories embedded with `model` so far.
    pub embedded: u64,
    /// Memories in the bank.
    pub memories: u64,
    pub started_at: Option<Timestamp>,
    /// Why the last run stopped, when it failed. Model errors only, never
    /// content.
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReembedState {
    /// The bank is on the daemon's model and no job is waiting.
    Current,
    /// A job is recorded and waiting to run, or to resume.
    Pending,
    /// The daemon is running the job.
    Running,
    /// The last run failed; the job resumes from where it stopped.
    Failed,
}

#[derive(Debug, thiserror::Error)]
pub enum ReembedError {
    #[error("unknown bank")]
    UnknownBank,

    #[error("no models are loaded, so nothing can be re-embedded")]
    NoModels,

    #[error("the bank's extraction didn't finish its chunk in time; try again")]
    Busy,

    #[error("embedding failed: {error}")]
    Model { error: ModelError },

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for ReembedError {
    fn from(error: rusqlite::Error) -> Self {
        ReembedError::Store(StoreError::Sqlite(error))
    }
}

/// The bank's rowid, name and recorded embedding model.
pub(crate) fn bank(conn: &Connection, bank: &str) -> Result<(i64, String), ReembedError> {
    let (bank_id, _) = find_bank(conn, bank)?.ok_or(ReembedError::UnknownBank)?;
    Ok((bank_id, recorded_model(conn, bank_id)?))
}

/// The embedding model the bank recorded, which it's served with.
pub(crate) fn recorded_model(conn: &Connection, bank_id: i64) -> Result<String, rusqlite::Error> {
    conn.query_row(
        "SELECT embedding_model FROM banks WHERE id = ?1",
        [bank_id],
        |row| row.get(0),
    )
}

/// The recorded job: the model it embeds with, how far it got, how many
/// it embedded and when it started.
fn job(conn: &Connection, bank_id: i64) -> Result<Option<Job>, rusqlite::Error> {
    conn.query_row(
        "SELECT model, cursor, embedded, started_at FROM reembeds WHERE bank_id = ?1",
        [bank_id],
        |row| {
            Ok(Job {
                model: row.get(0)?,
                cursor: row.get(1)?,
                embedded: row.get(2)?,
                started_at: timestamp(row.get(3)?),
            })
        },
    )
    .optional()
}

struct Job {
    model: String,
    cursor: i64,
    embedded: i64,
    started_at: Timestamp,
}

/// Where the bank's re-embed stands. `running` and `error` are the
/// service's, which keeps them in memory.
pub(crate) fn status(
    store: &Store,
    bank_name: &str,
    model: &str,
    running: bool,
    error: Option<String>,
) -> Result<ReembedStatus, ReembedError> {
    let conn = store.connection();
    let (bank_id, recorded) = bank(&conn, bank_name)?;
    let memories: i64 = conn.query_row(
        "SELECT COUNT(*) FROM memories WHERE bank_id = ?1",
        [bank_id],
        |row| row.get(0),
    )?;
    let job = job(&conn, bank_id)?;
    let state = match (&job, running, &error) {
        (_, true, _) => ReembedState::Running,
        (None, false, _) => ReembedState::Current,
        (Some(_), false, Some(_)) => ReembedState::Failed,
        (Some(_), false, None) => ReembedState::Pending,
    };
    Ok(ReembedStatus {
        bank: bank_name.trim().to_string(),
        recorded_model: recorded,
        model: model.to_string(),
        state,
        embedded: job
            .as_ref()
            .map_or(0, |job| u64::try_from(job.embedded).unwrap_or(0)),
        memories: u64::try_from(memories).unwrap_or(0),
        started_at: job.as_ref().map(|job| job.started_at),
        error: job.and(error),
    })
}

/// Records a job moving the bank to `model`, unless the bank is already
/// on it. A job recorded for another model, from a deploy since replaced,
/// starts over. Returns whether there's a job to run.
pub(crate) fn start(store: &Store, bank_name: &str, model: &str) -> Result<bool, ReembedError> {
    let now = micros(store.now());
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, recorded) = bank(&tx, bank_name)?;
    let existing = job(&tx, bank_id)?;
    let wanted = match existing {
        Some(job) if job.model == model => true,
        Some(_) => {
            tx.execute("DELETE FROM reembeds WHERE bank_id = ?1", [bank_id])?;
            tx.execute("DELETE FROM reembed_vectors WHERE bank_id = ?1", [bank_id])?;
            recorded != model
        }
        None => recorded != model,
    };
    if wanted {
        tx.execute(
            "INSERT INTO reembeds (bank_id, model, started_at, updated_at)
             VALUES (?1, ?2, ?3, ?3)
             ON CONFLICT (bank_id) DO NOTHING",
            (bank_id, model, now),
        )?;
    }
    tx.commit()?;
    Ok(wanted)
}

/// The banks with a recorded job, which the daemon resumes at startup.
pub(crate) fn pending(store: &Store) -> Result<Vec<String>, StoreError> {
    let conn = store.connection();
    let mut statement = conn
        .prepare("SELECT b.name FROM reembeds r JOIN banks b ON b.id = r.bank_id ORDER BY r.id")?;
    let names = statement
        .query_map([], |row| row.get(0))?
        .collect::<Result<_, _>>()?;
    Ok(names)
}

/// Embeds the next batch after the job's cursor into the side table.
/// Returns how many it embedded: 0 once it has caught up, or when there's
/// no job. The store isn't held while the model runs.
pub(crate) fn step(
    store: &Store,
    bank_id: i64,
    embedder: &dyn Embedder,
) -> Result<usize, ReembedError> {
    let batch: Vec<(i64, String)> = {
        let conn = store.connection();
        let Some(job) = job(&conn, bank_id)? else {
            return Ok(0);
        };
        if job.model != embedder.model_id() {
            return Ok(0);
        }
        next_batch(&conn, bank_id, job.cursor, REEMBED_BATCH)?
    };
    if batch.is_empty() {
        return Ok(0);
    }
    let vectors = embed(store, embedder, &batch)?;
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    // The store wasn't held while the model ran: a bank deletion may have
    // removed the job, and an erase the memories. A batch whose job is gone
    // is dropped, and only memories still in the bank are staged.
    let still_running = job(&tx, bank_id)?.is_some_and(|job| job.model == embedder.model_id());
    if !still_running {
        return Ok(0);
    }
    for ((memory_id, _), vector) in batch.iter().zip(&vectors) {
        stage(&tx, bank_id, *memory_id, vector)?;
    }
    let cursor = batch.last().map_or(0, |(id, _)| *id);
    tx.execute(
        "UPDATE reembeds SET cursor = ?2, embedded = embedded + ?3, updated_at = ?4
         WHERE bank_id = ?1",
        (bank_id, cursor, batch.len() as i64, micros(store.now())),
    )?;
    tx.commit()?;
    Ok(batch.len())
}

/// The bank's memories after `cursor`, in rowid order, with their sentences.
fn next_batch(
    conn: &Connection,
    bank_id: i64,
    cursor: i64,
    limit: usize,
) -> Result<Vec<(i64, String)>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT id, content FROM memories WHERE bank_id = ?1 AND id > ?2 ORDER BY id LIMIT ?3",
    )?;
    statement
        .query_map((bank_id, cursor, limit as i64), |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?
        .collect()
}

/// One vector per memory, each fitting the index.
fn embed(
    store: &Store,
    embedder: &dyn Embedder,
    batch: &[(i64, String)],
) -> Result<Vec<Vec<f32>>, ReembedError> {
    let texts: Vec<&str> = batch.iter().map(|(_, text)| text.as_str()).collect();
    let vectors = embedder
        .embed(&texts)
        .map_err(|error| ReembedError::Model { error })?;
    let width = store.vectors().dimensions();
    if vectors.len() != texts.len() || vectors.iter().any(|vector| vector.len() != width) {
        return Err(ReembedError::Model {
            error: ModelError::Inference {
                model: embedder.model_id().to_owned(),
                reason: format!(
                    "returned vectors that don't fit the index: one {width}-wide vector per text"
                ),
            },
        });
    }
    Ok(vectors)
}

fn stage(
    conn: &Connection,
    bank_id: i64,
    memory_id: i64,
    vector: &[f32],
) -> Result<(), rusqlite::Error> {
    let bytes: Vec<u8> = vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    // Only a memory still in the bank: one erased while the model ran
    // leaves nothing staged behind.
    conn.execute(
        "INSERT INTO reembed_vectors (bank_id, memory_id, embedding)
         SELECT ?1, ?2, ?3 WHERE EXISTS (SELECT 1 FROM memories WHERE id = ?2 AND bank_id = ?1)
         ON CONFLICT (memory_id) DO UPDATE SET embedding = excluded.embedding",
        (bank_id, memory_id, bytes),
    )?;
    Ok(())
}

/// The swap, under the bank's hold: embeds what the job hasn't reached,
/// then replaces the bank's vectors, records the model and deletes the job
/// and its side rows, in one transaction. Returns how many memories the
/// bank now has vectors for under the new model, or `None` when there was
/// no job for this model.
pub(crate) fn swap(
    store: &Store,
    _hold: &BankHold,
    bank_id: i64,
    embedder: &dyn Embedder,
) -> Result<Option<usize>, ReembedError> {
    // What arrived after the last step: few, since the hold stops
    // extraction. Embedded before the transaction, outside the store lock.
    let (from, tail) = {
        let conn = store.connection();
        let Some(job) = job(&conn, bank_id)? else {
            return Ok(None);
        };
        if job.model != embedder.model_id() {
            return Ok(None);
        }
        let recorded = recorded_model(&conn, bank_id)?;
        let mut statement = conn.prepare(
            "SELECT m.id, m.content FROM memories m
             WHERE m.bank_id = ?1
               AND NOT EXISTS (SELECT 1 FROM reembed_vectors r WHERE r.memory_id = m.id)
             ORDER BY m.id",
        )?;
        let tail: Vec<(i64, String)> = statement
            .query_map([bank_id], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<_, _>>()?;
        (recorded, tail)
    };
    let mut tail_vectors = Vec::with_capacity(tail.len());
    for batch in tail.chunks(REEMBED_BATCH) {
        tail_vectors.extend(embed(store, embedder, batch)?);
    }

    let mut conn = store.connection();
    let tx = conn.transaction()?;
    if !job(&tx, bank_id)?.is_some_and(|job| job.model == embedder.model_id()) {
        return Ok(None);
    }
    for ((memory_id, _), vector) in tail.iter().zip(&tail_vectors) {
        stage(&tx, bank_id, *memory_id, vector)?;
    }
    let staged: Vec<(i64, Vec<u8>)> = {
        let mut statement = tx.prepare(
            "SELECT r.memory_id, r.embedding FROM reembed_vectors r
             JOIN memories m ON m.id = r.memory_id AND m.bank_id = r.bank_id
             WHERE r.bank_id = ?1 ORDER BY r.memory_id",
        )?;
        statement
            .query_map([bank_id], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<_, _>>()?
    };
    let vectors = store.vectors();
    for (memory_id, bytes) in &staged {
        let vector: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        vectors
            .upsert(&tx, bank_id, *memory_id, &vector)
            .map_err(|error| match error {
                crate::store::VectorError::Sqlite(error) => {
                    ReembedError::Store(StoreError::Sqlite(error))
                }
                other => ReembedError::Model {
                    error: ModelError::Inference {
                        model: embedder.model_id().to_owned(),
                        reason: other.to_string(),
                    },
                },
            })?;
    }
    let now = micros(store.now());
    tx.execute(
        "UPDATE banks SET embedding_model = ?2, updated_at = ?3 WHERE id = ?1",
        (bank_id, embedder.model_id(), now),
    )?;
    tx.execute("DELETE FROM reembed_vectors WHERE bank_id = ?1", [bank_id])?;
    tx.execute("DELETE FROM reembeds WHERE bank_id = ?1", [bank_id])?;
    crate::store::bank::log_edit(
        &tx,
        store,
        bank_id,
        EDIT_REEMBEDDED,
        None,
        &serde_json::json!({
            "from": from,
            "to": embedder.model_id(),
            "memories": staged.len(),
        })
        .to_string(),
    )?;
    tx.commit()?;
    Ok(Some(staged.len()))
}
