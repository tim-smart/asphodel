//! Retract: the owner says a memory never held.
//!
//! It's a denial, as call 2's `denies` label makes one, but with no
//! successor: the memory is invalidated and `superseded_by` stays empty.
//! Recall, the agenda, refresh inputs, the prompt block and reconciliation
//! all leave out an invalidated memory, so it drops out of each at once.
//! Whatever the memory ended is open again, since the ending never
//! happened either. The memory itself stays, shown as retracted, keeps its
//! accesses, and is purged once it fades like any other.
//!
//! Only a chain's head can be retracted, as only a head can be translated:
//! an older version was already replaced. Each change is a logged edit of
//! ids, never content. There's no undo yet; the edit row holds what one
//! would need.

use jiff::Timestamp;
use rusqlite::OptionalExtension;
use serde::Serialize;
use uuid::Uuid;

use crate::erase::bank_links;
use crate::extraction::{EDIT_END_CLEARED, EDIT_RETRACTED};
use crate::ingest::find_bank;
use crate::store::bank::log_memory_edit;
use crate::store::{Store, StoreError, micros};
use crate::strength::chain_head;

/// What `retract` returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Retracted {
    pub memory: Uuid,
    pub retracted_at: Timestamp,
    /// The memories it had ended, open again.
    pub reopened: Vec<Uuid>,
}

#[derive(Debug, thiserror::Error)]
pub enum RetractError {
    #[error("unknown bank")]
    UnknownBank,

    #[error("no such memory in the bank")]
    UnknownMemory,

    #[error("the memory was superseded by {head}; retract that instead")]
    Superseded { head: Uuid },

    #[error("the memory is already retracted")]
    AlreadyRetracted,

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for RetractError {
    fn from(error: rusqlite::Error) -> Self {
        RetractError::Store(StoreError::Sqlite(error))
    }
}

/// Retracts the memory `id` names in `bank`. Returns the bank's rowid too,
/// for the service to settle what it changed.
pub(crate) fn retract(
    store: &Store,
    bank: &str,
    id: &str,
) -> Result<(i64, Retracted), RetractError> {
    let now = store.now();
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, _) = find_bank(&tx, bank)?.ok_or(RetractError::UnknownBank)?;
    let uuid = id
        .trim()
        .parse::<Uuid>()
        .map_err(|_| RetractError::UnknownMemory)?;
    let (memory_id, superseded, invalidated): (i64, bool, bool) = tx
        .query_row(
            "SELECT id, superseded_by IS NOT NULL, invalidated_at IS NOT NULL FROM memories
             WHERE uuid = ?1 AND bank_id = ?2 AND hidden_at IS NULL",
            (uuid.to_string(), bank_id),
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?
        .ok_or(RetractError::UnknownMemory)?;
    if invalidated {
        return Err(RetractError::AlreadyRetracted);
    }
    if superseded {
        let head = chain_head(&bank_links(&tx, bank_id)?, memory_id);
        let head: String =
            tx.query_row("SELECT uuid FROM memories WHERE id = ?1", [head], |row| {
                row.get(0)
            })?;
        return Err(RetractError::Superseded {
            head: head.parse().expect("a stored uuid parses"),
        });
    }

    tx.execute(
        "UPDATE memories SET invalidated_at = ?2, updated_at = ?2 WHERE id = ?1",
        (memory_id, micros(now)),
    )?;
    let ended: Vec<(i64, String)> = {
        let mut statement =
            tx.prepare("SELECT id, uuid FROM memories WHERE ended_by = ?1 ORDER BY id")?;
        statement
            .query_map([memory_id], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<_, _>>()?
    };
    let mut reopened = Vec::with_capacity(ended.len());
    for (ended_id, ended_uuid) in ended {
        tx.execute(
            "UPDATE memories SET valid_until = NULL, valid_until_precision = NULL,
                    ended_by = NULL, updated_at = ?2
             WHERE id = ?1",
            (ended_id, micros(now)),
        )?;
        log_memory_edit(
            &tx,
            store,
            bank_id,
            EDIT_END_CLEARED,
            ended_id,
            &serde_json::json!({ "ended_by": memory_id, "by": "owner" }).to_string(),
        )?;
        reopened.push(ended_uuid.parse().expect("a stored uuid parses"));
    }
    log_memory_edit(
        &tx,
        store,
        bank_id,
        EDIT_RETRACTED,
        memory_id,
        &serde_json::json!({
            "by": "owner",
            "invalidated_at": micros(now),
            "denied": true,
            "reopened": reopened,
        })
        .to_string(),
    )?;
    tx.commit()?;
    tracing::info!(reopened = reopened.len(), "the owner retracted a memory");
    Ok((
        bank_id,
        Retracted {
            memory: uuid,
            retracted_at: now,
            reopened,
        },
    ))
}
