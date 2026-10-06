//! The three retrievers, each a ranked list of memory rowids within one
//! bank, best first. They find; cleaning up the hits is
//! [`super::candidates`].

use std::collections::BTreeSet;

use rusqlite::Connection;

use crate::constants::RESTATED_FETCH;
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
    let Some(query) = match_any(text) else {
        return Ok(Vec::new());
    };
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

/// An FTS5 query matching any of `text`'s words, each as a quoted phrase so
/// nothing is read as syntax, or `None` when it has none.
fn match_any(text: &str) -> Option<String> {
    let words: BTreeSet<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect();
    if words.is_empty() {
        return None;
    }
    let query = words.iter().map(|word| phrase(word));
    Some(query.collect::<Vec<_>>().join(" OR "))
}

/// Which of a memory's sentences a hit matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Sentence {
    /// The memory's own.
    Memory,
    /// A restatement absorbed into it, by rowid.
    Restatement(i64),
}

/// One hit of an arm explicit recall runs: the memory, which may be any
/// version in its chain, and the sentence of it that matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Hit {
    pub memory: i64,
    pub sentence: Sentence,
}

/// `ids` as hits on their own sentences, for the arms that search nothing
/// else.
pub(super) fn own(ids: Vec<i64>) -> Vec<Hit> {
    let hit = |memory| Hit {
        memory,
        sentence: Sentence::Memory,
    };
    ids.into_iter().map(hit).collect()
}

/// Vector search over memory and restatement sentences together, for
/// explicit recall: the [`RESTATED_FETCH`] × `limit` nearest sentences of
/// either kind, nearest first, collapsed to each memory's nearest, then the
/// first `limit` memories. A memory restated many times takes one place,
/// and its restatements can't fill the arm. A memory's own sentence wins a
/// tie with a restatement, then the lower rowid.
pub(super) fn vector_restated(
    conn: &Connection,
    bank_id: i64,
    vector: &[f32],
    limit: usize,
) -> Result<Vec<Hit>, rusqlite::Error> {
    let fetch = limit.saturating_mul(RESTATED_FETCH);
    let memories = SqliteVec
        .nearest(conn, bank_id, vector, fetch)
        .map_err(sqlite_error)?;
    let restatements = SqliteVec
        .nearest_restatements(conn, bank_id, vector, fetch)
        .map_err(sqlite_error)?;
    let mut rows: Vec<(f32, u8, i64, Hit)> = memories
        .into_iter()
        .map(|hit| {
            let own = Hit {
                memory: hit.memory_id,
                sentence: Sentence::Memory,
            };
            (hit.distance, 0, hit.memory_id, own)
        })
        .chain(restatements.into_iter().map(|hit| {
            let restated = Hit {
                memory: hit.memory_id,
                sentence: Sentence::Restatement(hit.restatement_id),
            };
            (hit.distance, 1, hit.restatement_id, restated)
        }))
        .collect();
    rows.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then(left.1.cmp(&right.1))
            .then(left.2.cmp(&right.2))
    });
    rows.truncate(fetch);
    Ok(collapse(rows.into_iter().map(|row| row.3), limit))
}

/// BM25 over memory and restatement sentences together, for explicit
/// recall: one query over `recall_fts` (schema version 20), so the two
/// kinds' ranks compare,
/// the [`RESTATED_FETCH`] × `limit` best sentences collapsed to each
/// memory's best, then the first `limit` memories. Words are matched as
/// [`bm25`] matches them. A memory's own sentence wins a tie with a
/// restatement.
pub(super) fn bm25_restated(
    conn: &Connection,
    bank_id: i64,
    text: &str,
    limit: usize,
) -> Result<Vec<Hit>, rusqlite::Error> {
    let Some(query) = match_any(text) else {
        return Ok(Vec::new());
    };
    let fetch = limit.saturating_mul(RESTATED_FETCH);
    let mut statement = conn.prepare_cached(
        "SELECT r.id, COALESCE(m.id, r.memory_id)
         FROM recall_fts f
         LEFT JOIN memories m ON (f.rowid & 1) = 0 AND m.id = f.rowid / 2
         LEFT JOIN restatements r ON (f.rowid & 1) = 1 AND r.id = (f.rowid - 1) / 2
         LEFT JOIN memories rm ON rm.id = r.memory_id
         WHERE recall_fts MATCH ?1 AND COALESCE(m.bank_id, rm.bank_id) = ?2
         ORDER BY f.rank, f.rowid & 1, f.rowid
         LIMIT ?3",
    )?;
    let rows = statement.query_map((query, bank_id, sql_limit(fetch)), |row| {
        let restatement: Option<i64> = row.get(0)?;
        Ok(Hit {
            memory: row.get(1)?,
            sentence: restatement.map_or(Sentence::Memory, Sentence::Restatement),
        })
    })?;
    let rows: Vec<Hit> = rows.collect::<Result<_, _>>()?;
    Ok(collapse(rows.into_iter(), limit))
}

/// Each memory's first hit, in order, up to `limit` memories.
fn collapse(hits: impl Iterator<Item = Hit>, limit: usize) -> Vec<Hit> {
    let mut seen = BTreeSet::new();
    hits.filter(|hit| seen.insert(hit.memory))
        .take(limit)
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
