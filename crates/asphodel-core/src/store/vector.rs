//! Vector search behind a trait, with a flat sqlite-vec implementation.
//!
//! "Rust storage and search stack" (TIM-89) picked sqlite-vec's exact,
//! brute-force `vec0` tables: at tens to hundreds of thousands of memories a
//! scan is tens of milliseconds, and the table commits and rolls back with
//! the enclosing transaction. The trait is the hedge the same decision asked
//! for: if sqlite-vec goes unmaintained, a scan over a BLOB column replaces
//! it without touching callers. Every operation takes the connection it
//! should run on, so a vector write is part of the memory's transaction.

use rusqlite::Connection;

/// bge-small-en-v1.5's output size, and the width of the `vec0` table
/// (TIM-89). Changing models means a re-embed and a migration.
pub const EMBEDDING_DIMENSIONS: usize = 384;

/// The most neighbours sqlite-vec 0.1.9 returns from one KNN query
/// (`SQLITE_VEC_VEC0_K_MAX`); it refuses a larger `k`.
pub const KNN_K_MAX: usize = 4096;

/// One hit from a nearest-neighbour search.
#[derive(Debug, Clone, PartialEq)]
pub struct Neighbour {
    /// The memory's rowid.
    pub memory_id: i64,
    /// Cosine distance: 0 is identical, 1 is orthogonal.
    pub distance: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum VectorError {
    #[error("vector has {got} dimensions, the index takes {expected}")]
    Dimensions { expected: usize, got: usize },

    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
}

/// A nearest-neighbour index over memory vectors, keyed by memory rowid and
/// partitioned by bank.
pub trait VectorIndex: Send + Sync {
    /// The width every vector must have.
    fn dimensions(&self) -> usize;

    /// Stores `vector` for `memory_id`, replacing any vector it had.
    fn upsert(
        &self,
        conn: &Connection,
        bank_id: i64,
        memory_id: i64,
        vector: &[f32],
    ) -> Result<(), VectorError>;

    /// Removes `memory_id`'s vector. Removing a memory that has none is
    /// not an error.
    fn remove(&self, conn: &Connection, memory_id: i64) -> Result<(), VectorError>;

    /// The `k` nearest memories to `query` within `bank_id`, nearest first.
    /// Any `k` is allowed: an index with a limit of its own answers a larger
    /// one some other way.
    fn nearest(
        &self,
        conn: &Connection,
        bank_id: i64,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<Neighbour>, VectorError>;
}

/// The `memory_vectors` vec0 table from the schema.
#[derive(Debug, Clone, Copy, Default)]
pub struct SqliteVec;

impl SqliteVec {
    fn check(&self, vector: &[f32]) -> Result<Vec<u8>, VectorError> {
        if vector.len() != EMBEDDING_DIMENSIONS {
            return Err(VectorError::Dimensions {
                expected: EMBEDDING_DIMENSIONS,
                got: vector.len(),
            });
        }
        // sqlite-vec reads a float vector as little-endian f32 bytes.
        Ok(vector
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect())
    }
}

impl VectorIndex for SqliteVec {
    fn dimensions(&self) -> usize {
        EMBEDDING_DIMENSIONS
    }

    fn upsert(
        &self,
        conn: &Connection,
        bank_id: i64,
        memory_id: i64,
        vector: &[f32],
    ) -> Result<(), VectorError> {
        let bytes = self.check(vector)?;
        // vec0 can't update a primary or partition key, so replace the row.
        conn.execute(
            "DELETE FROM memory_vectors WHERE memory_id = ?1",
            [memory_id],
        )?;
        conn.execute(
            "INSERT INTO memory_vectors (memory_id, bank_id, embedding) VALUES (?1, ?2, ?3)",
            (memory_id, bank_id, bytes),
        )?;
        Ok(())
    }

    fn remove(&self, conn: &Connection, memory_id: i64) -> Result<(), VectorError> {
        conn.execute(
            "DELETE FROM memory_vectors WHERE memory_id = ?1",
            [memory_id],
        )?;
        Ok(())
    }

    fn nearest(
        &self,
        conn: &Connection,
        bank_id: i64,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<Neighbour>, VectorError> {
        let bytes = self.check(query)?;
        if k == 0 {
            return Ok(Vec::new());
        }
        // Past sqlite-vec's KNN limit, an exact scan of the bank's vectors
        // in the same metric (the TIM-108 re-review). It's the brute force
        // the KNN query does anyway, without the cap.
        let sql = if k <= KNN_K_MAX {
            "SELECT memory_id, distance FROM memory_vectors
             WHERE bank_id = ?1 AND embedding MATCH ?2 AND k = ?3
             ORDER BY distance"
        } else {
            "SELECT memory_id, vec_distance_cosine(embedding, ?2) AS distance
             FROM memory_vectors
             WHERE bank_id = ?1
             ORDER BY distance, memory_id
             LIMIT ?3"
        };
        let k = i64::try_from(k).unwrap_or(i64::MAX);
        let mut statement = conn.prepare_cached(sql)?;
        let rows = statement.query_map((bank_id, bytes, k), |row| {
            Ok(Neighbour {
                memory_id: row.get(0)?,
                distance: row.get::<_, f64>(1)? as f32,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }
}
