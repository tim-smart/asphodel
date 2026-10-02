//! The shadow table of purged rows (TIM-97, decision 7). Replay keeps the
//! content and embedding of every memory purge removed, only inside
//! `ASPHODEL_REPLAY_DIR`, and counts after the run how many memories
//! created later would have reconciled against one: the purged-then-re-
//! mentioned rate, which sets δ before the first production purge.

use std::path::Path;

use jiff::Timestamp;
use rusqlite::Connection;
use uuid::Uuid;

use super::report::ReMentioned;

/// A purged memory, as it was.
#[derive(Debug, Clone)]
pub struct ShadowRow {
    pub memory: Uuid,
    pub content: String,
    pub embedding: Vec<f32>,
    pub purged_at: Timestamp,
}

/// A memory extraction created during the run.
#[derive(Debug, Clone)]
pub struct Created {
    pub memory: Uuid,
    pub content: String,
    pub embedding: Vec<f32>,
    pub created_at: Timestamp,
}

/// Writes the rows to `shadow.db` under the replay dir, replacing any
/// earlier run's.
pub fn write(path: &Path, rows: &[ShadowRow]) -> rusqlite::Result<()> {
    let mut conn = Connection::open(path)?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS purged (
             memory TEXT NOT NULL,
             content TEXT NOT NULL,
             embedding TEXT NOT NULL,
             purged_at TEXT NOT NULL
         );
         DELETE FROM purged;",
    )?;
    let tx = conn.transaction()?;
    {
        let mut insert = tx.prepare(
            "INSERT INTO purged (memory, content, embedding, purged_at) VALUES (?1, ?2, ?3, ?4)",
        )?;
        for row in rows {
            insert.execute((
                row.memory.to_string(),
                &row.content,
                serde_json::to_string(&row.embedding).expect("floats serialise"),
                row.purged_at.to_string(),
            ))?;
        }
    }
    tx.commit()
}

/// Each memory created after time t whose nearest shadow row, purged before
/// t, is at or above `floor` counts as purged then re-mentioned.
pub fn re_mentioned(created: &[Created], shadow: &[ShadowRow], floor: f64) -> ReMentioned {
    let re_mentioned = created
        .iter()
        .filter(|memory| {
            shadow
                .iter()
                .filter(|row| row.purged_at < memory.created_at)
                .map(|row| cosine(&row.embedding, &memory.embedding))
                .fold(None, |best: Option<f64>, score| {
                    Some(best.map_or(score, |best| best.max(score)))
                })
                .is_some_and(|best| best >= floor)
        })
        .count() as u64;
    let purged = shadow.len() as u64;
    ReMentioned {
        purged,
        re_mentioned,
        rate: if purged == 0 {
            0.0
        } else {
            re_mentioned as f64 / purged as f64
        },
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let scale = norm(a) * norm(b);
    if scale == 0.0 { 0.0 } else { dot / scale }
}
