//! The three retrievers, each a ranked list of memory rowids within one
//! bank, best first. They find; cleaning up the hits is
//! [`super::candidates`].

use std::collections::BTreeSet;

use rusqlite::Connection;

use crate::extraction::{entities_named, phrase};
use crate::store::{SqliteVec, VectorError, VectorIndex};

/// Vector search over memory content: the `limit` nearest memories to
/// `vector`, nearest first.
pub(super) fn vector(
    conn: &Connection,
    bank_id: i64,
    vector: &[f32],
    limit: usize,
) -> Result<Vec<i64>, rusqlite::Error> {
    let hits = SqliteVec
        .nearest(conn, bank_id, vector, limit)
        .map_err(sqlite_error)?;
    Ok(hits.into_iter().map(|hit| hit.memory_id).collect())
}

/// BM25 over memory content within the bank, best first: any of `text`'s
/// words, each as a quoted phrase so nothing is read as syntax.
/// Reconciliation searches its claims the same way.
pub(crate) fn bm25(
    conn: &Connection,
    bank_id: i64,
    text: &str,
    limit: usize,
) -> Result<Vec<i64>, rusqlite::Error> {
    let words: BTreeSet<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect();
    if words.is_empty() {
        return Ok(Vec::new());
    }
    let query = words
        .iter()
        .map(|word| phrase(word))
        .collect::<Vec<_>>()
        .join(" OR ");
    let mut statement = conn.prepare_cached(
        "SELECT m.id FROM memories_fts f JOIN memories m ON m.id = f.rowid
         WHERE memories_fts MATCH ?1 AND m.bank_id = ?2
         ORDER BY f.rank, m.id
         LIMIT ?3",
    )?;
    statement
        .query_map((query, bank_id, sql_limit(limit)), |row| row.get(0))?
        .collect()
}

/// The entity arm: the entities whose aliases appear in `query`, found with
/// the alias FTS extraction uses, leaving out the seeded `user` and
/// `assistant` (nearly every memory links to `user`, so the arm would be a
/// full scan). Their memories rank by cosine to `vector`, the newer
/// `observed_at` winning ties.
pub(super) fn entity(
    conn: &Connection,
    bank_id: i64,
    query: &str,
    vector: &[f32],
    limit: usize,
) -> Result<Vec<i64>, rusqlite::Error> {
    let seeded = seeded(conn, bank_id)?;
    let entities = entities_named(conn, bank_id, &[query], &seeded)?;
    linked(conn, &entities, vector, limit)
}

/// The memories linked to any of `entities`, by cosine to `vector`, newer
/// `observed_at` first on a tie, then the lower rowid.
pub(super) fn linked(
    conn: &Connection,
    entities: &BTreeSet<i64>,
    vector: &[f32],
    limit: usize,
) -> Result<Vec<i64>, rusqlite::Error> {
    if entities.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    // sqlite-vec reads a float vector as little-endian f32 bytes.
    let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ids = entities
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let mut statement = conn.prepare(&format!(
        "SELECT DISTINCT m.id, vec_distance_cosine(v.embedding, ?1) AS distance, m.observed_at
         FROM memory_entities me
         JOIN memories m ON m.id = me.memory_id
         JOIN memory_vectors v ON v.memory_id = m.id
         WHERE me.entity_id IN ({ids})
            OR me.entity_id IN (SELECT id FROM entities WHERE merged_into IN ({ids}))
         ORDER BY distance, m.observed_at DESC, m.id
         LIMIT ?2"
    ))?;
    statement
        .query_map((bytes, sql_limit(limit)), |row| row.get(0))?
        .collect()
}

/// The bank's seeded `user` and `assistant` entities.
pub(super) fn seeded(conn: &Connection, bank_id: i64) -> Result<Vec<i64>, rusqlite::Error> {
    let mut statement =
        conn.prepare_cached("SELECT id FROM entities WHERE bank_id = ?1 AND seeded IS NOT NULL")?;
    statement.query_map([bank_id], |row| row.get(0))?.collect()
}

fn sql_limit(limit: usize) -> i64 {
    i64::try_from(limit).unwrap_or(i64::MAX)
}

fn sqlite_error(error: VectorError) -> rusqlite::Error {
    match error {
        VectorError::Sqlite(error) => error,
        other => rusqlite::Error::ToSqlConversionFailure(Box::new(other)),
    }
}
