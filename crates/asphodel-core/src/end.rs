//! End and reopen: the owner says a memory has stopped holding, or that it
//! still holds after all.
//!
//! Ending is what call 2's `ends` does, with no memory to point at: the
//! window closes a minute before now and `ended_by` stays empty, so the
//! memory has ended by the time the call returns. The agenda, the open
//! tasks reconciliation shows and the status filter all read a closed
//! window as ended, and call 2 is shown the memory as ended, so a later
//! claim can't relabel it. A memory whose window has already closed, such as
//! a past event, can't be ended. The memory keeps its accesses and fades
//! like any other.
//!
//! Reopening undoes an ending, the owner's or extraction's. An owner's
//! ending is undone to the window the memory had before it, found on any
//! version of the memory, since a translation carries the window over; it's
//! refused when that window has closed since. An ending by another memory is
//! cleared, as a denial of that memory clears it. A window that closed on
//! its own, with no ending, isn't reopened.
//!
//! Only a chain's head can be ended or reopened, as only a head can be
//! retracted. Each change is a logged edit of ids and times, never content.

use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::agenda::has_ended;
use crate::extraction::{EDIT_END_CLEARED, EDIT_END_REPOINTED, EDIT_ENDED};
use crate::ingest::find_bank;
use crate::retract::superseded_head;
use crate::store::bank::log_memory_edit;
use crate::store::strength::{memory_kind, world_time};
use crate::store::{Store, StoreError, micros, timestamp};
use crate::strength::{Kind, Window};

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

    #[error("the memory has already ended or its window has closed")]
    AlreadyEnded,

    #[error("nothing ended the memory")]
    NotEnded,

    #[error("the window the memory had before it was ended has closed since")]
    StillEnded,

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for EndError {
    fn from(error: rusqlite::Error) -> Self {
        EndError::Store(StoreError::Sqlite(error))
    }
}

/// A memory the owner might end or reopen.
struct Row {
    id: i64,
    uuid: Uuid,
    ended_by: Option<i64>,
    valid_until: Option<i64>,
    precision: Option<String>,
    window: Window,
    tz: TimeZone,
    superseded: bool,
    invalidated: bool,
}

impl Row {
    /// Nothing ended it and its window hasn't closed, so it can be ended.
    fn open(&self, now: Timestamp) -> bool {
        self.ended_by.is_none()
            && !self
                .window
                .closes_at(&self.tz)
                .is_some_and(|closes| closes <= now)
    }
}

fn load(conn: &Connection, id: i64) -> Result<Option<Row>, rusqlite::Error> {
    conn.query_row(
        "SELECT m.uuid, m.ended_by, m.valid_until, m.valid_until_precision, s.timezone,
                m.superseded_by IS NOT NULL, m.invalidated_at IS NOT NULL, m.kind,
                m.valid_from, m.valid_from_precision, m.due_at, m.due_at_precision,
                m.observed_at
         FROM memories m JOIN chunks c ON c.id = m.chunk_id JOIN sources s ON s.id = c.source_id
         WHERE m.id = ?1 AND m.hidden_at IS NULL",
        [id],
        |row| {
            let tz: String = row.get(4)?;
            let valid_until = row.get(2)?;
            let precision: Option<String> = row.get(3)?;
            Ok(Row {
                id,
                uuid: row
                    .get::<_, String>(0)?
                    .parse()
                    .expect("a stored uuid parses"),
                ended_by: row.get(1)?,
                window: Window {
                    kind: memory_kind(&row.get::<_, String>(7)?).unwrap_or(Kind::Fact),
                    valid_from: world_time(row.get(8)?, row.get(9)?),
                    valid_until: world_time(valid_until, precision.clone()),
                    due_at: world_time(row.get(10)?, row.get(11)?),
                    observed_at: timestamp(row.get(12)?),
                },
                valid_until,
                precision,
                tz: TimeZone::get(&tz).unwrap_or(TimeZone::UTC),
                superseded: row.get(5)?,
                invalidated: row.get(6)?,
            })
        },
    )
    .optional()
}

fn head(conn: &Connection, bank: &str, id: &str) -> Result<(i64, Row), EndError> {
    let (bank_id, _) = find_bank(conn, bank)?.ok_or(EndError::UnknownBank)?;
    let uuid = id
        .trim()
        .parse::<Uuid>()
        .map_err(|_| EndError::UnknownMemory)?;
    let id: Option<i64> = conn
        .query_row(
            "SELECT id FROM memories WHERE uuid = ?1 AND bank_id = ?2",
            (uuid.to_string(), bank_id),
            |row| row.get(0),
        )
        .optional()?;
    let row = match id {
        Some(id) => load(conn, id)?,
        None => None,
    };
    let row = row.ok_or(EndError::UnknownMemory)?;
    if row.invalidated {
        return Err(EndError::Retracted);
    }
    if row.superseded {
        let head = superseded_head(conn, bank_id, row.id)?;
        return Err(EndError::Superseded { head });
    }
    Ok((bank_id, row))
}

/// Ends the memory `id` names in `bank`. Returns the bank's rowid too, for
/// the service to settle what it changed.
pub(crate) fn end(store: &Store, bank: &str, id: &str) -> Result<(i64, Ended), EndError> {
    let now = store.now();
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, head) = head(&tx, bank, id)?;
    if !head.open(now) {
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
    let (valid_until, precision) = restored(&tx, &head, now)??;
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

/// What the owner can do to the memory with rowid `id` at `now`: end it,
/// reopen it, both false for one retracted, superseded or forgotten.
pub(crate) fn actions(
    conn: &Connection,
    id: i64,
    now: Timestamp,
) -> Result<(bool, bool), rusqlite::Error> {
    let Some(row) = load(conn, id)? else {
        return Ok((false, false));
    };
    if row.invalidated || row.superseded {
        return Ok((false, false));
    }
    Ok((row.open(now), restored(conn, &row, now)?.is_ok()))
}

/// The window reopening gives `row`: none after another memory's ending,
/// the previous one after the owner's, or why it can't be reopened.
#[allow(clippy::type_complexity)]
fn restored(
    conn: &Connection,
    row: &Row,
    now: Timestamp,
) -> Result<Result<(Option<i64>, Option<String>), EndError>, rusqlite::Error> {
    if row.ended_by.is_some() {
        return Ok(Ok((None, None)));
    }
    let Some(previous) = owner_ending(conn, row.id, row.valid_until)? else {
        return Ok(Err(EndError::NotEnded));
    };
    let valid_until = previous["valid_until"].as_i64();
    let precision = previous["precision"].as_str().map(str::to_owned);
    if has_ended(world_time(valid_until, precision.clone()), &row.tz, now) {
        return Ok(Err(EndError::StillEnded));
    }
    Ok(Ok((valid_until, precision)))
}

/// Whether the owner's ending is what closes the memory with rowid `id`:
/// the latest change to the end of it or any earlier version is the
/// owner's, and set the `valid_until` it has now. Returns the window before
/// it, `{valid_until, precision}`, both null for a memory that had no end.
pub(crate) fn owner_ending(
    conn: &Connection,
    id: i64,
    valid_until: Option<i64>,
) -> Result<Option<Value>, rusqlite::Error> {
    let latest: Option<(String, String)> = conn
        .query_row(
            "WITH RECURSIVE versions(id) AS (
                 SELECT ?1
                 UNION SELECT m.id FROM memories m JOIN versions v ON m.superseded_by = v.id
             )
             SELECT kind, details FROM edits
             WHERE memory_id IN versions AND kind IN (?2, ?3, ?4)
             ORDER BY id DESC LIMIT 1",
            (id, EDIT_ENDED, EDIT_END_CLEARED, EDIT_END_REPOINTED),
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((kind, details)) = latest else {
        return Ok(None);
    };
    let details: Value = serde_json::from_str(&details).unwrap_or(Value::Null);
    if kind != EDIT_ENDED
        || details["by"] != "owner"
        || details["valid_until"].as_i64() != valid_until
    {
        return Ok(None);
    }
    Ok(Some(match details["previous"].clone() {
        Value::Null => json!({ "valid_until": null, "precision": null }),
        previous => previous,
    }))
}
