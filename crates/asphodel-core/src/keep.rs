//! Keep and unkeep: the owner's significance setting (TIM-94, decision 9).
//!
//! Keeping a memory sets its `owner_significance` to `kept`, as high as
//! significance goes, so it never fades. Unkeeping hands it back to the
//! significance extraction gave it. Both take memory ids only, never text
//! (TIM-93), and write the same field `memory significance` will (ADR 0010).
//! Each change is a logged edit with ids and levels, never content.
//!
//! An id that isn't a memory of the bank, or names one that's been
//! forgotten, is reported as unknown rather than failing the call, so a tool
//! call with one stale id still keeps the rest.

use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::extraction::EDIT_KEPT;
use crate::ingest::find_bank;
use crate::store::bank::log_memory_edit;
use crate::store::{Store, StoreError, micros};

/// The most ids one call takes (TIM-94, decision 9).
pub const MAX_IDS: usize = 50;

/// The edit kind unkeep writes.
pub const EDIT_UNKEPT: &str = "memory_unkept";

/// What `keep` and `unkeep` take.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MemoryIds {
    /// Memory ids, at most [`MAX_IDS`].
    pub ids: Vec<String>,
}

/// What `keep` returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Kept {
    pub kept: Vec<Uuid>,
    pub unknown: Vec<String>,
}

/// What `unkeep` returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Unkept {
    pub unkept: Vec<Uuid>,
    pub unknown: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum KeepError {
    #[error("unknown bank")]
    UnknownBank,

    #[error("at most {MAX_IDS} ids per call, got {given}")]
    TooMany { given: usize },

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for KeepError {
    fn from(error: rusqlite::Error) -> Self {
        KeepError::Store(StoreError::Sqlite(error))
    }
}

/// Keeps each memory `ids` names in `bank`.
pub(crate) fn keep(store: &Store, bank: &str, ids: &[String]) -> Result<Kept, KeepError> {
    let (kept, unknown) = apply(store, bank, ids, Change::Keep)?;
    Ok(Kept { kept, unknown })
}

/// Hands each memory `ids` names in `bank` back to its extracted
/// significance. A memory that wasn't kept is left as it is and still
/// listed as unkept, so a repeat is harmless.
pub(crate) fn unkeep(store: &Store, bank: &str, ids: &[String]) -> Result<Unkept, KeepError> {
    let (unkept, unknown) = apply(store, bank, ids, Change::Unkeep)?;
    Ok(Unkept { unkept, unknown })
}

#[derive(Clone, Copy)]
enum Change {
    Keep,
    Unkeep,
}

fn apply(
    store: &Store,
    bank: &str,
    ids: &[String],
    change: Change,
) -> Result<(Vec<Uuid>, Vec<String>), KeepError> {
    if ids.len() > MAX_IDS {
        return Err(KeepError::TooMany { given: ids.len() });
    }
    let now = micros(store.now());
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, _) = find_bank(&tx, bank)?.ok_or(KeepError::UnknownBank)?;
    let mut done = Vec::new();
    let mut unknown = Vec::new();
    for id in ids {
        let Ok(uuid) = id.trim().parse::<Uuid>() else {
            unknown.push(id.clone());
            continue;
        };
        if done.contains(&uuid) {
            continue;
        }
        let found: Option<(i64, Option<String>)> = tx
            .query_row(
                "SELECT id, owner_significance FROM memories
                 WHERE uuid = ?1 AND bank_id = ?2 AND hidden_at IS NULL",
                (uuid.to_string(), bank_id),
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        let Some((memory_id, owner)) = found else {
            unknown.push(id.clone());
            continue;
        };
        match change {
            Change::Keep if owner.as_deref() != Some("kept") => {
                tx.execute(
                    "UPDATE memories SET owner_significance = 'kept', updated_at = ?2
                     WHERE id = ?1",
                    (memory_id, now),
                )?;
                let details = match &owner {
                    Some(level) => format!("{{\"from\":\"{level}\"}}"),
                    None => "{\"from\":null}".to_string(),
                };
                log_memory_edit(&tx, store, bank_id, EDIT_KEPT, memory_id, &details)?;
            }
            Change::Unkeep if owner.as_deref() == Some("kept") => {
                tx.execute(
                    "UPDATE memories SET owner_significance = NULL, updated_at = ?2
                     WHERE id = ?1",
                    (memory_id, now),
                )?;
                log_memory_edit(&tx, store, bank_id, EDIT_UNKEPT, memory_id, "{}")?;
            }
            Change::Keep | Change::Unkeep => {}
        }
        done.push(uuid);
    }
    tx.commit()?;
    Ok((done, unknown))
}
