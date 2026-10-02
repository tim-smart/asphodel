//! Keep and unkeep: the owner's significance setting.
//!
//! Keeping a memory sets its `owner_significance` to `kept`, as high as
//! significance goes, so it never fades. Unkeeping hands it back to the
//! significance extraction gave it. Both take memory ids only, never text,
//! and write the same field `memory significance` will (ADR 0010).
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

/// The most ids one call takes.
pub const MAX_IDS: usize = 50;

/// The edit kind unkeep writes.
pub const EDIT_UNKEPT: &str = "memory_unkept";

/// The edit kind `memory significance` writes (ADR 0010).
pub const EDIT_SIGNIFICANCE_SET: &str = "owner_significance_set";

/// The levels `memory significance` takes: extraction's five and `kept`.
pub const OWNER_LEVELS: [&str; 6] = ["trivial", "minor", "notable", "major", "critical", "kept"];

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

/// What `PUT /v1/banks/{bank}/memories/{id}/significance` takes: a level,
/// or `null` to clear the owner's setting.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SignificanceRequest {
    pub level: Option<String>,
}

/// What `memory significance` did. `from` and `to` are the owner's
/// setting before and after; `extracted` is the level beneath it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SignificanceSet {
    pub memory: Uuid,
    pub extracted: String,
    pub from: Option<String>,
    pub to: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum KeepError {
    #[error("unknown bank")]
    UnknownBank,

    #[error("no such memory in the bank")]
    UnknownMemory,

    #[error(
        "unknown significance level; give trivial, minor, notable, major, critical, kept or clear"
    )]
    InvalidLevel,

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

/// Sets the owner's significance on one memory, or clears it with `None`,
/// handing the memory back to the level extraction gave. The same field keep
/// and unkeep write; a change is a logged edit of
/// levels only.
pub(crate) fn set_significance(
    store: &Store,
    bank: &str,
    id: &str,
    level: Option<&str>,
) -> Result<SignificanceSet, KeepError> {
    let to = match level.map(str::trim) {
        None | Some("clear") => None,
        Some(level) => Some(
            OWNER_LEVELS
                .into_iter()
                .find(|known| *known == level)
                .ok_or(KeepError::InvalidLevel)?,
        ),
    };
    let now = micros(store.now());
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, _) = find_bank(&tx, bank)?.ok_or(KeepError::UnknownBank)?;
    let uuid = id
        .trim()
        .parse::<Uuid>()
        .map_err(|_| KeepError::UnknownMemory)?;
    let (memory_id, extracted, from): (i64, String, Option<String>) = tx
        .query_row(
            "SELECT id, significance, owner_significance FROM memories
             WHERE uuid = ?1 AND bank_id = ?2 AND hidden_at IS NULL",
            (uuid.to_string(), bank_id),
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?
        .ok_or(KeepError::UnknownMemory)?;
    if from.as_deref() != to {
        tx.execute(
            "UPDATE memories SET owner_significance = ?2, updated_at = ?3 WHERE id = ?1",
            (memory_id, to, now),
        )?;
        let details = serde_json::json!({ "from": from, "to": to }).to_string();
        log_memory_edit(
            &tx,
            store,
            bank_id,
            EDIT_SIGNIFICANCE_SET,
            memory_id,
            &details,
        )?;
    }
    tx.commit()?;
    Ok(SignificanceSet {
        memory: uuid,
        extracted,
        from,
        to: to.map(str::to_string),
    })
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
