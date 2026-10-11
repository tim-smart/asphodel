//! End and reopen: the owner says a memory has stopped holding, or that it
//! still holds after all.
//!
//! Ending is what call 2's `ends` does, with no memory to point at: the
//! window closes a minute before now and `ended_by` stays empty, so the
//! memory has ended by the time the call returns. The agenda, the open
//! tasks reconciliation shows and the status filter all read a closed
//! window as ended. The memory keeps its accesses and fades like any other.
//!
//! Reopening undoes an ending, the owner's or extraction's. An owner's
//! ending is undone to the window the memory had before it; one by another
//! memory is cleared, as a denial of that memory clears it. A window that
//! closed on its own, with no ending, isn't reopened.
//!
//! Only a chain's head can be ended or reopened, as only a head can be
//! retracted. Each change is a logged edit of ids and times, never content.

use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use rusqlite::{OptionalExtension, Transaction};
use serde::Serialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::agenda::has_ended;
use crate::extraction::{EDIT_END_CLEARED, EDIT_END_REPOINTED, EDIT_ENDED};
use crate::ingest::find_bank;
use crate::retract::superseded_head;
use crate::store::bank::log_memory_edit;
use crate::store::strength::world_time;
use crate::store::{Store, StoreError, micros};

/// What `end` returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Ended {
    pub memory: Uuid,
    /// When the window now closes.
    pub ended_at: Timestamp,
}

/// What `reopen` returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Reopened {
    pub memory: Uuid,
}

#[derive(Debug, thiserror::Error)]
pub enum EndError {
    #[error("unknown bank")]
    UnknownBank,

    #[error("no such memory in the bank")]
    UnknownMemory,

    #[error("the memory was superseded by {head}; use that instead")]
    Superseded { head: Uuid },

    #[error("the memory is retracted")]
    Retracted,

    #[error("the memory has already ended")]
    AlreadyEnded,

    #[error("nothing ended the memory")]
    NotEnded,

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for EndError {
    fn from(error: rusqlite::Error) -> Self {
        EndError::Store(StoreError::Sqlite(error))
    }
}

/// A chain head the owner can end or reopen.
struct Head {
    id: i64,
    uuid: Uuid,
    ended_by: Option<i64>,
    valid_until: Option<i64>,
    precision: Option<String>,
    tz: TimeZone,
}

impl Head {
    fn has_ended(&self, now: Timestamp) -> bool {
        self.ended_by.is_some()
            || has_ended(
                world_time(self.valid_until, self.precision.clone()),
                &self.tz,
                now,
            )
    }
}

fn head(tx: &Transaction<'_>, bank: &str, id: &str) -> Result<(i64, Head), EndError> {
    let (bank_id, _) = find_bank(tx, bank)?.ok_or(EndError::UnknownBank)?;
    let uuid = id
        .trim()
        .parse::<Uuid>()
        .map_err(|_| EndError::UnknownMemory)?;
    let (head, superseded, invalidated) = tx
        .query_row(
            "SELECT m.id, m.ended_by, m.valid_until, m.valid_until_precision, s.timezone,
                    m.superseded_by IS NOT NULL, m.invalidated_at IS NOT NULL
             FROM memories m JOIN chunks c ON c.id = m.chunk_id JOIN sources s ON s.id = c.source_id
             WHERE m.uuid = ?1 AND m.bank_id = ?2 AND m.hidden_at IS NULL",
            (uuid.to_string(), bank_id),
            |row| {
                let tz: String = row.get(4)?;
                Ok((
                    Head {
                        id: row.get(0)?,
                        uuid,
                        ended_by: row.get(1)?,
                        valid_until: row.get(2)?,
                        precision: row.get(3)?,
                        tz: TimeZone::get(&tz).unwrap_or(TimeZone::UTC),
                    },
                    row.get::<_, bool>(5)?,
                    row.get::<_, bool>(6)?,
                ))
            },
        )
        .optional()?
        .ok_or(EndError::UnknownMemory)?;
    if invalidated {
        return Err(EndError::Retracted);
    }
    if superseded {
        let head = superseded_head(tx, bank_id, head.id)?;
        return Err(EndError::Superseded { head });
    }
    Ok((bank_id, head))
}

/// Ends the memory `id` names in `bank`. Returns the bank's rowid too, for
/// the service to settle what it changed.
pub(crate) fn end(store: &Store, bank: &str, id: &str) -> Result<(i64, Ended), EndError> {
    let now = store.now();
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, head) = head(&tx, bank, id)?;
    if head.has_ended(now) {
        return Err(EndError::AlreadyEnded);
    }
    // A minute's window that closes now: ended as soon as this returns.
    let until = now - SignedDuration::from_mins(1);
    tx.execute(
        "UPDATE memories SET valid_until = ?2, valid_until_precision = 'minute', updated_at = ?3
         WHERE id = ?1",
        (head.id, micros(until), micros(now)),
    )?;
    let previous = head
        .valid_until
        .map(|at| json!({ "valid_until": at, "precision": head.precision }));
    log_memory_edit(
        &tx,
        store,
        bank_id,
        EDIT_ENDED,
        head.id,
        &json!({
            "by": "owner",
            "valid_until": micros(until),
            "precision": "minute",
            "previous": previous,
        })
        .to_string(),
    )?;
    tx.commit()?;
    tracing::info!("the owner ended a memory");
    Ok((
        bank_id,
        Ended {
            memory: head.uuid,
            ended_at: now,
        },
    ))
}

/// Reopens the memory `id` names in `bank`, undoing the latest ending.
pub(crate) fn reopen(store: &Store, bank: &str, id: &str) -> Result<(i64, Reopened), EndError> {
    let now = store.now();
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, head) = head(&tx, bank, id)?;
    let (valid_until, precision) = match head.ended_by {
        Some(_) => (None, None),
        None => {
            let previous = owner_ending(&tx, head.id)?.ok_or(EndError::NotEnded)?;
            (
                previous["valid_until"].as_i64(),
                previous["precision"].as_str().map(str::to_owned),
            )
        }
    };
    tx.execute(
        "UPDATE memories SET valid_until = ?2, valid_until_precision = ?3, ended_by = NULL,
                updated_at = ?4
         WHERE id = ?1",
        (head.id, valid_until, &precision, micros(now)),
    )?;
    log_memory_edit(
        &tx,
        store,
        bank_id,
        EDIT_END_CLEARED,
        head.id,
        &json!({ "by": "owner", "ended_by": head.ended_by }).to_string(),
    )?;
    tx.commit()?;
    tracing::info!("the owner reopened a memory");
    Ok((bank_id, Reopened { memory: head.uuid }))
}

/// The window before the owner's ending, when the latest change to the
/// memory's end is one: `{valid_until, precision}`, both null for a memory
/// that had no end.
fn owner_ending(tx: &Transaction<'_>, memory: i64) -> Result<Option<Value>, rusqlite::Error> {
    let latest: Option<(String, String)> = tx
        .query_row(
            "SELECT kind, details FROM edits
             WHERE memory_id = ?1 AND kind IN (?2, ?3, ?4)
             ORDER BY id DESC LIMIT 1",
            (memory, EDIT_ENDED, EDIT_END_CLEARED, EDIT_END_REPOINTED),
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((kind, details)) = latest else {
        return Ok(None);
    };
    let details: Value = serde_json::from_str(&details).unwrap_or(Value::Null);
    if kind != EDIT_ENDED || details["by"] != "owner" {
        return Ok(None);
    }
    Ok(Some(match details["previous"].clone() {
        Value::Null => json!({ "valid_until": null, "precision": null }),
        previous => previous,
    }))
}
