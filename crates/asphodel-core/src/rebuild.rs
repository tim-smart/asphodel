//! Rebuilding a bank's memories from the sources it holds.
//!
//! `asphodel rebuild --bank` extracts a bank's memories again under the
//! daemon's current templates, in place, through the bank's own extraction
//! queue. Under the bank's hold, one transaction ([`start`]) records the
//! rebuild, deletes the memories it replaces and queues every chunk that
//! still has its text, oldest first, at turn priority so time alone orders
//! turns and documents. The workers then extract them as they extract
//! anything, each chunk as of its source's time ([`rerun`]). While chunks
//! of the rebuild are queued the bank's recall, injection and agenda are
//! off, and a chunk queued after the start isn't handed out
//! ([`crate::queue`]). When the last one leaves the queue, [`complete`]
//! carries over what the rerun can't derive and the bank is served again.
//! The nightly sweep stays off for the bank until [`ack`].
//!
//! What a rebuild never touches:
//!
//! - A chunk a forget masked. It's never sent to the LLM again, as the
//!   chunk extracted or as the context of another chunk the rebuild runs
//!   ([`masked`]), and the memories resting on it stay as they are. A
//!   forget made while the rebuild runs counts from the moment it hides its
//!   chain, before its erase masks anything ([`withheld`]): the chunks it
//!   will mask aren't extracted again, and the rebuild counts them. A
//!   chunk already out with a worker writes nothing, whether the forget
//!   reached it or its context, and is read again ([`forgot`]). This is
//!   about the rebuild's own chunks. A turn that arrives later is
//!   extracted as any turn is, earlier turns of its session and all.
//! - A memory the owner touched: its significance or `kept` was set, the
//!   owner retracted, ended or reopened it, it was translated, or an entity
//!   was linked or unlinked by hand. It keeps its id, with the rest of its
//!   chain and the memories it ended, and a claim extracted again over its
//!   passage is dropped.
//!
//! A rebuilt memory gets a new id. `created`, `mentioned_again` and
//! `confirmed` accesses and restatements come back from the rerun, on their
//! sources' dates. `used` accesses, pending credits and mentions in chunks
//! that aren't run again are held in `rebuild_accesses` by passage, and go to
//! the new memory quoting an overlapping passage of the same chunk.

use std::collections::{BTreeMap, BTreeSet};

use jiff::{SignedDuration, Timestamp};
use rusqlite::{Connection, OptionalExtension, Transaction};
use serde::Serialize;

use crate::erase::REDACTION_MASK;
use crate::ingest::{PRIORITY_TURN, find_bank};
use crate::mental_models::Schedule;
use crate::operations::META_LAST_BACKUP;
use crate::queue::BankHold;
use crate::store::bank::log_edit;
use crate::store::{Store, StoreError, VectorError, VectorIndex, micros, timestamp};

/// The edit kind a rebuild's start writes, with its counts.
pub const EDIT_REBUILD_STARTED: &str = "rebuild_started";

/// The edit kind a rebuild's last chunk writes, with what was carried over.
pub const EDIT_REBUILD_COMPLETED: &str = "rebuild_completed";

/// The edit kind `rebuild ack` writes.
pub const EDIT_REBUILD_ACKED: &str = "rebuild_acked";

/// How old the last backup may be when a rebuild starts. A backup is the
/// only way back to the memories a rebuild replaces.
pub const BACKUP_MAX_AGE: SignedDuration = SignedDuration::from_hours(24);

/// What a rebuild would do, `asphodel rebuild --bank` without `--confirm`.
/// Counts only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RebuildPlan {
    pub bank: String,
    /// Sources with a chunk to run again.
    pub sources: usize,
    /// Chunks to run again: one call 1 each, and a call 2 for most.
    pub chunks: usize,
    pub skipped: Skipped,
    /// Memories deleted at the start and extracted again.
    pub memories_replaced: usize,
    /// Memories left as they are.
    pub memories_kept: usize,
    /// Accesses and pending credits held by passage for the new memories.
    pub accesses_carried: usize,
    /// Mental models whose answers are blanked until the rebuild ends.
    pub models: usize,
    pub last_backup_at: Option<Timestamp>,
    /// Why the rebuild can't start now. Empty when it can.
    pub blockers: Vec<Blocker>,
}

/// What a rebuild leaves out, by reason.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Skipped {
    /// Chunks whose text the sweep cleared.
    pub swept: usize,
    /// Chunks of a removed document or turn.
    pub removed: usize,
    /// Chunks a forget masked.
    pub masked: usize,
    /// Turns that asked for a forget, stored as a key only.
    pub forget_requests: usize,
}

/// Why a rebuild can't start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Blocker {
    /// A forget or a purge is waiting in the bank's queue.
    ErasePending,
    /// The bank is moving to another embedding model.
    ReembedPending,
    /// The daemon doesn't carry the bank's embedding model.
    ModelUnavailable,
    /// No backup completed in the last [`BACKUP_MAX_AGE`].
    BackupStale,
    /// The bank's last rebuild is still running or unacknowledged.
    Unacknowledged,
}

impl std::fmt::Display for Blocker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Blocker::ErasePending => "a forget or a purge is waiting in the bank's queue",
            Blocker::ReembedPending => "a re-embed of the bank is recorded",
            Blocker::ModelUnavailable => "the daemon doesn't carry the bank's embedding model",
            Blocker::BackupStale => "no backup completed in the last 24 hours",
            Blocker::Unacknowledged => {
                "the bank's last rebuild is still running or waiting for `asphodel rebuild ack`"
            }
        })
    }
}

/// Where a bank's rebuild stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RebuildState {
    /// Never rebuilt, or the last rebuild was acknowledged.
    Idle,
    /// Chunks of the rebuild are still queued: recall is off for the bank.
    Running,
    /// Every chunk has run. The sweep waits for `asphodel rebuild ack`.
    Complete,
}

/// `GET /v1/banks/{bank}/rebuild`: the bank's latest rebuild. The counts are
/// zero for a bank never rebuilt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RebuildStatus {
    pub bank: String,
    pub state: RebuildState,
    pub started_at: Option<Timestamp>,
    pub completed_at: Option<Timestamp>,
    pub acked_at: Option<Timestamp>,
    /// Chunks the rebuild queued.
    pub chunks: usize,
    /// Of those, the ones still queued.
    pub remaining: usize,
    /// Chunks of the bank that reached the retry cap since the start. Their
    /// memories are gone until `asphodel chunks --failed --retry` runs them.
    pub failed: usize,
    pub replaced: usize,
    pub kept: usize,
    /// Accesses, credits and passages moved to a new memory at the end.
    pub carried: usize,
    /// Those no new memory's passage overlapped.
    pub dropped: usize,
    /// Chunks the rebuild queued and didn't extract again, because a forget
    /// made after it started reached them. Not failures: they can't be
    /// retried, and whatever else they held wasn't rebuilt.
    pub withheld: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum RebuildError {
    #[error("unknown bank")]
    UnknownBank,
    /// `--confirm` didn't repeat the bank's name.
    #[error("the confirmation doesn't repeat the bank's name")]
    NotConfirmed,
    #[error("the rebuild can't start: {}", .blockers.iter().map(Blocker::to_string).collect::<Vec<_>>().join("; "))]
    Blocked { blockers: Vec<Blocker> },
    /// The bank's chunk in flight didn't finish in time.
    #[error("the bank is busy extracting; try again")]
    Busy,
    /// `rebuild ack` for a bank with no rebuild waiting for one.
    #[error("the bank has no rebuild to acknowledge")]
    NothingToAck,
    /// `rebuild ack` while chunks of the rebuild are still queued.
    #[error("the rebuild is still running")]
    StillRunning,
    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for RebuildError {
    fn from(error: rusqlite::Error) -> Self {
        RebuildError::Store(StoreError::Sqlite(error))
    }
}

/// A bank's `rebuilds` row.
#[derive(Debug, Clone)]
struct Row {
    started_at: i64,
    queue_boundary: i64,
    memory_boundary: i64,
    chunks: i64,
    replaced: i64,
    kept: i64,
    carried: i64,
    dropped: i64,
    forget_generation: i64,
    withheld: i64,
    completed_at: Option<i64>,
    acked_at: Option<i64>,
}

fn row(conn: &Connection, bank_id: i64) -> Result<Option<Row>, rusqlite::Error> {
    conn.query_row(
        "SELECT started_at, queue_boundary, memory_boundary, chunks, replaced, kept, carried,
                dropped, completed_at, acked_at, forget_generation, withheld
         FROM rebuilds WHERE bank_id = ?1",
        [bank_id],
        |row| {
            Ok(Row {
                started_at: row.get(0)?,
                queue_boundary: row.get(1)?,
                memory_boundary: row.get(2)?,
                chunks: row.get(3)?,
                replaced: row.get(4)?,
                kept: row.get(5)?,
                carried: row.get(6)?,
                dropped: row.get(7)?,
                completed_at: row.get(8)?,
                acked_at: row.get(9)?,
                forget_generation: row.get(10)?,
                withheld: row.get(11)?,
            })
        },
    )
    .optional()
}

/// Whether chunks of a rebuild of the bank are still queued, so its recall,
/// injection and agenda are off.
pub(crate) fn running(conn: &Connection, bank_id: i64) -> Result<bool, rusqlite::Error> {
    conn.query_row(
        "SELECT EXISTS (SELECT 1 FROM rebuilds WHERE bank_id = ?1 AND completed_at IS NULL)",
        [bank_id],
        |row| row.get(0),
    )
}

/// The banks whose nightly sweep waits for `rebuild ack`: a rebuild changes
/// what the sweep would delete, so the operator looks first.
pub(crate) fn held(conn: &Connection) -> Result<BTreeSet<i64>, rusqlite::Error> {
    let mut statement =
        conn.prepare_cached("SELECT bank_id FROM rebuilds WHERE acked_at IS NULL")?;
    statement.query_map([], |row| row.get(0))?.collect()
}

/// SQL for "a forget masked some of chunk `c`", with [`REDACTION_MASK`] as
/// parameter `mask`. A rebuild never shows such a chunk to the LLM again,
/// as the chunk extracted or as the context of another it runs.
pub(crate) fn masked(mask: usize) -> String {
    format!(
        "(EXISTS (SELECT 1 FROM chunk_redactions r WHERE r.chunk_id = c.id)
          OR instr(COALESCE(c.text, ''), ?{mask}) > 0)"
    )
}

/// Whether a chunk a rebuild runs again must not be extracted: a forget
/// masked some of it, or a forget still waiting on the queue will
/// ([`crate::erase::awaited`]). `false` for a chunk ingested after the
/// bank's rebuild started, and for a bank never rebuilt.
///
/// Extraction asks before call 1. A forget that arrives after that moves
/// the bank's forget generation ([`forgot`]), which sends the chunk back
/// to be asked again ([`Rerun::stale`]).
pub(crate) fn withheld(conn: &Connection, chunk_id: i64) -> Result<bool, rusqlite::Error> {
    let masked = masked(2);
    let rerun: Option<(i64, bool)> = conn
        .query_row(
            &format!(
                "SELECT c.bank_id, {masked} FROM chunks c
                 JOIN sources s ON s.id = c.source_id
                 JOIN rebuilds r ON r.bank_id = c.bank_id
                 WHERE c.id = ?1 AND s.ingested_at <= r.started_at"
            ),
            (chunk_id, REDACTION_MASK.to_string()),
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    match rerun {
        None => Ok(false),
        Some((_, true)) => Ok(true),
        Some((bank_id, false)) => Ok(crate::erase::awaited(conn, bank_id)?
            .chunks
            .contains(&chunk_id)),
    }
}

/// Counts a chunk the rebuild took off the queue unextracted because it was
/// [`withheld`], inside the transaction that takes it off.
pub(crate) fn count_withheld(tx: &Transaction<'_>, bank_id: i64) -> Result<(), rusqlite::Error> {
    tx.execute(
        "UPDATE rebuilds SET withheld = withheld + 1 WHERE bank_id = ?1",
        [bank_id],
    )?;
    Ok(())
}

/// Called inside the transaction of a forget that hid a chain of the bank,
/// or of a source's removal. What the chunks a rebuild runs again may be
/// shown has changed, so the forget generation moves, and a chunk that read
/// its text and context before now writes nothing ([`Rerun::stale`]).
/// Call 1's saved replies go too: one of them may have been written from a
/// context that included what's now forgotten, and a retry would resume
/// from it. Nothing happens for a bank never rebuilt.
pub(crate) fn forgot(tx: &Transaction<'_>, bank_id: i64) -> Result<(), rusqlite::Error> {
    let rebuilt = tx.execute(
        "UPDATE rebuilds SET forget_generation = forget_generation + 1 WHERE bank_id = ?1",
        [bank_id],
    )?;
    if rebuilt > 0 {
        tx.execute(
            "UPDATE chunks SET call1_output = NULL
             WHERE bank_id = ?1 AND call1_output IS NOT NULL
               AND EXISTS (SELECT 1 FROM sources s JOIN rebuilds r ON r.bank_id = s.bank_id
                           WHERE s.id = chunks.source_id AND s.ingested_at <= r.started_at)",
            [bank_id],
        )?;
    }
    Ok(())
}

/// What extraction needs to know about a chunk a rebuild runs again.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Rerun {
    /// Keys the ids of the memories the chunk makes, so they differ from
    /// the ids the chunk's first extraction gave.
    pub salt: i64,
    /// The passages of the chunk that memories the rebuild kept rest on. A
    /// claim over one of them is dropped.
    pub kept: Vec<(usize, usize)>,
    /// The rebuild is still running: the chunk is read as of its source's
    /// time.
    pub running: bool,
    /// The bank's forget generation when the chunk's text and context were
    /// read.
    pub forget_generation: i64,
}

impl Rerun {
    /// Whether a forget has hidden something, or a source was removed,
    /// since the chunk's text and context were read ([`forgot`]). What the
    /// LLM made of them may then rest on forgotten text, in the chunk or in
    /// its context, so none of it is written or sent on to call 2. The
    /// chunk is read again from the start.
    pub(crate) fn stale(&self, conn: &Connection, bank_id: i64) -> Result<bool, rusqlite::Error> {
        let now: Option<i64> = conn
            .query_row(
                "SELECT forget_generation FROM rebuilds WHERE bank_id = ?1",
                [bank_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(now != Some(self.forget_generation))
    }
}

/// The rebuild context of a chunk whose source was ingested at
/// `ingested_at`: `None` unless the bank was rebuilt after that.
pub(crate) fn rerun(
    conn: &Connection,
    bank_id: i64,
    chunk_id: i64,
    ingested_at: Timestamp,
) -> Result<Option<Rerun>, rusqlite::Error> {
    let Some(row) = row(conn, bank_id)? else {
        return Ok(None);
    };
    if micros(ingested_at) > row.started_at {
        return Ok(None);
    }
    let mut statement = conn.prepare_cached(
        "SELECT source_start, source_end FROM memories WHERE chunk_id = ?1 AND id <= ?2",
    )?;
    let kept = statement
        .query_map((chunk_id, row.memory_boundary), |row| {
            Ok((span(row.get(0)?), span(row.get(1)?)))
        })?
        .collect::<Result<_, _>>()?;
    Ok(Some(Rerun {
        salt: row.started_at,
        kept,
        running: row.completed_at.is_none(),
        forget_generation: row.forget_generation,
    }))
}

fn span(offset: i64) -> usize {
    usize::try_from(offset).unwrap_or(0)
}

fn count(value: i64) -> usize {
    usize::try_from(value).unwrap_or(0)
}

/// A chunk of the bank as the survey reads it.
struct Chunk {
    id: i64,
    source_id: i64,
}

/// What the rerun can't derive again for one replaced memory.
struct Carry {
    chunk_id: i64,
    start: i64,
    end: i64,
    kind: &'static str,
    at: Option<i64>,
    turn: Option<i64>,
    source_id: Option<i64>,
    mention: Option<(i64, i64, i64)>,
}

/// What a rebuild of the bank would do now.
struct Survey {
    /// The chunks to run again, in the order they're queued.
    chunks: Vec<Chunk>,
    skipped: Skipped,
    replaced: BTreeSet<i64>,
    kept: usize,
    carries: Vec<Carry>,
}

/// Whether an edit is the owner's own, by its kind and details. An ending,
/// a reopened end and a retraction are the owner's when their details say
/// so, and a refinement is when it's a translation.
const OWNER_EDITS: &str = "e.kind IN ('owner_significance_set', 'memory_kept', 'memory_unkept',
                'entity_linked', 'entity_unlinked')
     OR (e.kind IN ('memory_ended', 'end_cleared', 'memory_retracted')
         AND json_extract(e.details, '$.by') = 'owner')
     OR (e.kind = 'memory_refined' AND json_type(e.details, '$.translated_to') IS NOT NULL)";

fn survey(conn: &Connection, bank_id: i64) -> Result<Survey, rusqlite::Error> {
    let mut skipped = Skipped::default();
    let mut chunks = Vec::new();
    {
        let masked = masked(2);
        let mut statement = conn.prepare(&format!(
            "SELECT c.id, c.source_id, c.text IS NULL OR c.tombstoned_at IS NOT NULL
                      OR s.tombstoned_at IS NOT NULL,
                    s.removed_at IS NOT NULL,
                    {masked}
             FROM chunks c JOIN sources s ON s.id = c.source_id
             WHERE c.bank_id = ?1
             ORDER BY s.observed_at, s.ingested_at, c.id"
        ))?;
        let mut rows = statement.query((bank_id, REDACTION_MASK.to_string()))?;
        while let Some(row) = rows.next()? {
            let (gone, removed, masked): (bool, bool, bool) =
                (row.get(2)?, row.get(3)?, row.get(4)?);
            if removed {
                skipped.removed += 1;
            } else if masked {
                skipped.masked += 1;
            } else if gone {
                skipped.swept += 1;
            } else {
                chunks.push(Chunk {
                    id: row.get(0)?,
                    source_id: row.get(1)?,
                });
            }
        }
    }
    skipped.forget_requests = count(conn.query_row(
        "SELECT COUNT(*) FROM sources
         WHERE bank_id = ?1 AND tombstone_reason = 'forget_requested'",
        [bank_id],
        |row| row.get(0),
    )?);
    let rerun: BTreeSet<i64> = chunks.iter().map(|chunk| chunk.id).collect();
    let rerun_sources: BTreeSet<i64> = chunks.iter().map(|chunk| chunk.source_id).collect();

    // Every memory with its chunk, its links and whether it stays on its
    // own account: the owner touched it, a forget is waiting on it, or its
    // chunk isn't run again.
    struct Memory {
        id: i64,
        chunk_id: i64,
        start: i64,
        end: i64,
        superseded_by: Option<i64>,
        ended_by: Option<i64>,
        stays: bool,
    }
    let memories: Vec<Memory> = {
        let mut statement = conn.prepare(&format!(
            "SELECT m.id, m.chunk_id, m.source_start, m.source_end, m.superseded_by, m.ended_by,
                    m.owner_significance IS NOT NULL OR m.hidden_at IS NOT NULL
                      OR EXISTS (SELECT 1 FROM edits e WHERE e.memory_id = m.id AND ({OWNER_EDITS}))
             FROM memories m WHERE m.bank_id = ?1 ORDER BY m.id"
        ))?;
        statement
            .query_map([bank_id], |row| {
                let chunk_id: i64 = row.get(1)?;
                Ok(Memory {
                    id: row.get(0)?,
                    chunk_id,
                    start: row.get(2)?,
                    end: row.get(3)?,
                    superseded_by: row.get(4)?,
                    ended_by: row.get(5)?,
                    stays: row.get::<_, bool>(6)? || !rerun.contains(&chunk_id),
                })
            })?
            .collect::<Result<_, _>>()?
    };

    // A chain stays whole, and so does what a kept memory ended: the rerun
    // drops the claim that ended it, so nothing would end it again.
    let mut kept: BTreeSet<i64> = memories.iter().filter(|m| m.stays).map(|m| m.id).collect();
    loop {
        let before = kept.len();
        for memory in &memories {
            let linked = memory.superseded_by.is_some_and(|id| kept.contains(&id))
                || memory.ended_by.is_some_and(|id| kept.contains(&id));
            if linked {
                kept.insert(memory.id);
            }
            if kept.contains(&memory.id)
                && let Some(successor) = memory.superseded_by
            {
                kept.insert(successor);
            }
        }
        if kept.len() == before {
            break;
        }
    }

    let mut carries = Vec::new();
    let mut replaced = BTreeSet::new();
    let mut accesses = conn.prepare(
        "SELECT kind, at, turn, source_id FROM accesses WHERE memory_id = ?1 AND kind != 'created'",
    )?;
    let mut credits = conn.prepare("SELECT turn, at FROM pending_credits WHERE memory_id = ?1")?;
    let mut mentions = conn.prepare(
        "SELECT p.chunk_id, c.source_id, p.start_offset, p.end_offset
         FROM mention_passages p JOIN chunks c ON c.id = p.chunk_id WHERE p.memory_id = ?1",
    )?;
    for memory in memories.iter().filter(|m| !kept.contains(&m.id)) {
        replaced.insert(memory.id);
        let carry = |kind, at, turn, source_id, mention| Carry {
            chunk_id: memory.chunk_id,
            start: memory.start,
            end: memory.end,
            kind,
            at,
            turn,
            source_id,
            mention,
        };
        // The mentions in chunks that aren't run again: their passages are
        // carried, and so are the accesses their sources gave.
        let mut left_sources = BTreeSet::new();
        for row in mentions.query_map([memory.id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })? {
            let (chunk, source, start, end) = row?;
            if !rerun.contains(&chunk) {
                left_sources.insert(source);
                carries.push(carry(
                    "passage",
                    None,
                    None,
                    None,
                    Some((chunk, start, end)),
                ));
            }
        }
        for row in accesses.query_map([memory.id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, Option<i64>>(3)?,
            ))
        })? {
            let (kind, at, turn, source) = row?;
            let kind = match kind.as_str() {
                "used" => "used",
                // A mention the rerun gives again unless its source isn't
                // run again.
                mention => {
                    let derived = source.is_some_and(|source| {
                        rerun_sources.contains(&source) && !left_sources.contains(&source)
                    });
                    if derived {
                        continue;
                    }
                    if mention == "confirmed" {
                        "confirmed"
                    } else {
                        "mentioned_again"
                    }
                }
            };
            carries.push(carry(kind, Some(at), Some(turn), source, None));
        }
        for row in credits.query_map([memory.id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
        })? {
            let (turn, at) = row?;
            carries.push(carry("pending", Some(at), Some(turn), None, None));
        }
    }

    Ok(Survey {
        chunks,
        skipped,
        kept: kept.len(),
        replaced,
        carries,
    })
}

/// Why the bank's rebuild can't start now. `serving` is whether the daemon
/// carries the bank's embedding model.
fn blockers(
    conn: &Connection,
    bank_id: i64,
    now: Timestamp,
    serving: bool,
) -> Result<(Vec<Blocker>, Option<Timestamp>), rusqlite::Error> {
    let mut blockers = Vec::new();
    let exists = |sql: &str| -> Result<bool, rusqlite::Error> {
        conn.query_row(&format!("SELECT EXISTS ({sql})"), [bank_id], |row| {
            row.get(0)
        })
    };
    if exists("SELECT 1 FROM extraction_queue WHERE bank_id = ?1 AND kind = 'erase'")?
        || exists("SELECT 1 FROM memories WHERE bank_id = ?1 AND hidden_at IS NOT NULL")?
    {
        blockers.push(Blocker::ErasePending);
    }
    if exists("SELECT 1 FROM reembeds WHERE bank_id = ?1")? {
        blockers.push(Blocker::ReembedPending);
    }
    if !serving {
        blockers.push(Blocker::ModelUnavailable);
    }
    let last_backup_at: Option<Timestamp> = conn
        .query_row(
            "SELECT value FROM store_meta WHERE key = ?1",
            [META_LAST_BACKUP],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .and_then(|at| at.parse().ok());
    if !last_backup_at.is_some_and(|at| at <= now && now.duration_since(at) <= BACKUP_MAX_AGE) {
        blockers.push(Blocker::BackupStale);
    }
    if exists("SELECT 1 FROM rebuilds WHERE bank_id = ?1 AND acked_at IS NULL")? {
        blockers.push(Blocker::Unacknowledged);
    }
    Ok((blockers, last_backup_at))
}

fn find(conn: &Connection, bank: &str) -> Result<(i64, String), RebuildError> {
    let (bank_id, _) = find_bank(conn, bank)?.ok_or(RebuildError::UnknownBank)?;
    let name = conn.query_row("SELECT name FROM banks WHERE id = ?1", [bank_id], |row| {
        row.get(0)
    })?;
    Ok((bank_id, name))
}

/// The dry run. It writes nothing.
pub(crate) fn plan(store: &Store, bank: &str, serving: bool) -> Result<RebuildPlan, RebuildError> {
    let conn = store.connection();
    let (bank_id, name) = find(&conn, bank)?;
    let survey = survey(&conn, bank_id)?;
    let (blockers, last_backup_at) = blockers(&conn, bank_id, store.now(), serving)?;
    let models = count(conn.query_row(
        "SELECT COUNT(*) FROM mental_models WHERE bank_id = ?1 AND answer IS NOT NULL",
        [bank_id],
        |row| row.get(0),
    )?);
    Ok(RebuildPlan {
        bank: name,
        sources: survey
            .chunks
            .iter()
            .map(|chunk| chunk.source_id)
            .collect::<BTreeSet<_>>()
            .len(),
        chunks: survey.chunks.len(),
        skipped: survey.skipped,
        memories_replaced: survey.replaced.len(),
        memories_kept: survey.kept,
        accesses_carried: survey
            .carries
            .iter()
            .filter(|carry| carry.kind != "passage")
            .count(),
        models,
        last_backup_at,
        blockers,
    })
}

/// Starts the bank's rebuild in one transaction, under the bank's hold so
/// no extraction is in flight. A crash before it commits leaves the bank as
/// it was. A crash after leaves the rebuild in the store, where the workers
/// pick it up.
pub(crate) fn start(
    store: &Store,
    _hold: &BankHold,
    bank: &str,
    serving: bool,
) -> Result<i64, RebuildError> {
    let now = store.now();
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, _) = find(&tx, bank)?;
    let (blockers, _) = blockers(&tx, bank_id, now, serving)?;
    if !blockers.is_empty() {
        return Err(RebuildError::Blocked { blockers });
    }
    let survey = survey(&tx, bank_id)?;

    tx.execute("DELETE FROM rebuilds WHERE bank_id = ?1", [bank_id])?;
    tx.execute("DELETE FROM rebuild_accesses WHERE bank_id = ?1", [bank_id])?;
    {
        let mut insert = tx.prepare(
            "INSERT INTO rebuild_accesses (bank_id, chunk_id, start_offset, end_offset, kind, at,
                                           turn, source_id, mention_chunk, mention_start,
                                           mention_end)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        )?;
        for carry in &survey.carries {
            let mention = match carry.mention {
                Some((chunk, start, end)) => (Some(chunk), Some(start), Some(end)),
                None => (None, None, None),
            };
            insert.execute(rusqlite::params![
                bank_id,
                carry.chunk_id,
                carry.start,
                carry.end,
                carry.kind,
                carry.at,
                carry.turn,
                carry.source_id,
                mention.0,
                mention.1,
                mention.2,
            ])?;
        }
    }

    // An answer restates the memories it cites, so every model's goes: each
    // is refreshed once when the rebuild ends.
    tx.execute(
        "DELETE FROM mental_model_cites
         WHERE model_id IN (SELECT id FROM mental_models WHERE bank_id = ?1)",
        [bank_id],
    )?;
    // The fingerprint goes too. A model whose memories were all kept would
    // otherwise look unchanged to its refresh and stay blank.
    tx.execute(
        "UPDATE mental_models SET answer = NULL, last_fingerprint = NULL WHERE bank_id = ?1",
        [bank_id],
    )?;
    // What sessions were shown names memories that are going.
    tx.execute("DELETE FROM session_blocks WHERE bank_id = ?1", [bank_id])?;
    tx.execute("DELETE FROM prompt_blocks WHERE bank_id = ?1", [bank_id])?;

    let memory_boundary: i64 = tx.query_row(
        "SELECT COALESCE(MAX(seq), 0) FROM sqlite_sequence WHERE name = 'memories'",
        [],
        |row| row.get(0),
    )?;
    let vectors = store.vectors();
    for memory in &survey.replaced {
        vectors.remove(&tx, *memory).map_err(|error| match error {
            VectorError::Sqlite(error) => error,
            other => rusqlite::Error::ToSqlConversionFailure(Box::new(other)),
        })?;
        tx.execute("DELETE FROM memories WHERE id = ?1", [memory])?;
    }

    {
        // A kept memory's restatements from these chunks come back with the
        // rerun, as its mentions do.
        let mut restatements = tx.prepare("DELETE FROM restatements WHERE chunk_id = ?1")?;
        let mut unqueue =
            tx.prepare("DELETE FROM extraction_queue WHERE kind = 'chunk' AND chunk_id = ?1")?;
        let mut reset = tx.prepare(
            "UPDATE chunks SET extracted_at = NULL, call1_output = NULL, failed_at = NULL,
                               error_count = 0, last_error_kind = NULL, last_error_status = NULL
             WHERE id = ?1",
        )?;
        let mut queue = tx.prepare(
            "INSERT INTO extraction_queue (bank_id, kind, chunk_id, priority, observed_at,
                                           enqueued_at)
             SELECT ?1, 'chunk', c.id, ?3, s.observed_at, ?4
             FROM chunks c JOIN sources s ON s.id = c.source_id WHERE c.id = ?2",
        )?;
        for chunk in &survey.chunks {
            restatements.execute([chunk.id])?;
            unqueue.execute([chunk.id])?;
            reset.execute([chunk.id])?;
            queue.execute((bank_id, chunk.id, PRIORITY_TURN, micros(now)))?;
        }
    }
    // Documents waiting from before run in time order with the rest.
    tx.execute(
        "UPDATE extraction_queue SET priority = ?2 WHERE bank_id = ?1 AND kind = 'chunk'",
        (bank_id, PRIORITY_TURN),
    )?;
    let queue_boundary: i64 = tx.query_row(
        "SELECT COALESCE(MAX(seq), 0) FROM sqlite_sequence WHERE name = 'extraction_queue'",
        [],
        |row| row.get(0),
    )?;

    tx.execute(
        "INSERT INTO rebuilds (bank_id, started_at, queue_boundary, memory_boundary, chunks,
                               replaced, kept)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        (
            bank_id,
            micros(now),
            queue_boundary,
            memory_boundary,
            survey.chunks.len() as i64,
            survey.replaced.len() as i64,
            survey.kept as i64,
        ),
    )?;
    let details = serde_json::json!({
        "chunks": survey.chunks.len(),
        "skipped": survey.skipped,
        "replaced": survey.replaced.len(),
        "kept": survey.kept,
    });
    log_edit(
        &tx,
        store,
        bank_id,
        EDIT_REBUILD_STARTED,
        None,
        &details.to_string(),
    )?;
    tx.commit()?;
    tracing::info!(
        bank_id,
        chunks = survey.chunks.len(),
        replaced = survey.replaced.len(),
        kept = survey.kept,
        "started a rebuild"
    );
    Ok(bank_id)
}

/// Ends the bank's rebuild once none of its chunks is queued: carries the
/// held accesses to the new memories, deletes the entities nothing rests on
/// any more, and turns the bank's recall back on. `None` while chunks are
/// still queued, or when no rebuild is running. Returns the bank's models,
/// which the service refreshes.
pub(crate) fn complete(
    store: &Store,
    schedule: &Schedule,
    bank_id: i64,
) -> Result<Option<BTreeSet<i64>>, StoreError> {
    let now = store.now();
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let Some(row) = row(&tx, bank_id)?.filter(|row| row.completed_at.is_none()) else {
        return Ok(None);
    };
    let queued: bool = tx.query_row(
        "SELECT EXISTS (SELECT 1 FROM extraction_queue
                        WHERE bank_id = ?1 AND kind = 'chunk' AND id <= ?2)",
        (bank_id, row.queue_boundary),
        |row| row.get(0),
    )?;
    if queued {
        return Ok(None);
    }

    let (carried, dropped) = carry(&tx, bank_id, row.memory_boundary)?;
    let entities: BTreeSet<i64> = {
        let mut statement = tx.prepare("SELECT id FROM entities WHERE bank_id = ?1")?;
        statement
            .query_map([bank_id], |row| row.get(0))?
            .collect::<Result<_, _>>()?
    };
    crate::erase::delete_orphans(&tx, &entities)?;
    tx.execute(
        "UPDATE rebuilds SET completed_at = ?2, carried = ?3, dropped = ?4 WHERE bank_id = ?1",
        (bank_id, micros(now), carried, dropped),
    )?;
    let details = serde_json::json!({
        "carried": carried,
        "dropped": dropped,
        "withheld": row.withheld,
    });
    log_edit(
        &tx,
        store,
        bank_id,
        EDIT_REBUILD_COMPLETED,
        None,
        &details.to_string(),
    )?;
    let models: BTreeSet<i64> = {
        let mut statement = tx.prepare("SELECT id FROM mental_models WHERE bank_id = ?1")?;
        statement
            .query_map([bank_id], |row| row.get(0))?
            .collect::<Result<_, _>>()?
    };
    // Requested with the completion, so a restart between the two can't
    // leave a completed rebuild whose models nothing refreshes.
    crate::mental_models::request(&tx, schedule, &models, now, false)?;
    tx.commit()?;
    tracing::info!(bank_id, carried, dropped, "a rebuild completed");
    Ok(Some(models))
}

/// Moves each held row to the memory the rerun made over an overlapping
/// passage of the same chunk, the one overlapping most, and deletes the
/// rows. Returns how many moved and how many found no memory.
fn carry(
    tx: &Transaction<'_>,
    bank_id: i64,
    memory_boundary: i64,
) -> Result<(i64, i64), rusqlite::Error> {
    type Held = (
        i64,
        i64,
        i64,
        String,
        Option<i64>,
        Option<i64>,
        Option<i64>,
        Option<(i64, i64, i64)>,
    );
    let rows: Vec<Held> = {
        let mut statement = tx.prepare(
            "SELECT chunk_id, start_offset, end_offset, kind, at, turn, source_id, mention_chunk,
                    mention_start, mention_end
             FROM rebuild_accesses WHERE bank_id = ?1 ORDER BY id",
        )?;
        statement
            .query_map([bank_id], |row| {
                let mention: Option<i64> = row.get(7)?;
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    match mention {
                        Some(chunk) => Some((chunk, row.get(8)?, row.get(9)?)),
                        None => None,
                    },
                ))
            })?
            .collect::<Result<_, _>>()?
    };
    let mut target = tx.prepare(
        "SELECT id FROM memories
         WHERE bank_id = ?1 AND chunk_id = ?2 AND id > ?5
           AND source_start < ?4 AND source_end > ?3
         ORDER BY MIN(source_end, ?4) - MAX(source_start, ?3) DESC, id
         LIMIT 1",
    )?;
    let (mut carried, mut dropped) = (0, 0);
    let mut targets: BTreeMap<(i64, i64, i64), Option<i64>> = BTreeMap::new();
    for (chunk, start, end, kind, at, turn, source, mention) in rows {
        let memory = match targets.get(&(chunk, start, end)) {
            Some(memory) => *memory,
            None => {
                let memory: Option<i64> = target
                    .query_row((bank_id, chunk, start, end, memory_boundary), |row| {
                        row.get(0)
                    })
                    .optional()?;
                targets.insert((chunk, start, end), memory);
                memory
            }
        };
        let Some(memory) = memory else {
            dropped += 1;
            continue;
        };
        carried += 1;
        match (kind.as_str(), mention) {
            ("passage", Some((chunk, start, end))) => {
                tx.execute(
                    "INSERT OR IGNORE INTO mention_passages (memory_id, chunk_id, start_offset,
                                                             end_offset)
                     VALUES (?1, ?2, ?3, ?4)",
                    (memory, chunk, start, end),
                )?;
            }
            ("pending", _) => {
                tx.execute(
                    "INSERT OR IGNORE INTO pending_credits (bank_id, memory_id, turn, at)
                     VALUES (?1, ?2, ?3, ?4)",
                    (bank_id, memory, turn.unwrap_or(0), at.unwrap_or(0)),
                )?;
            }
            (kind, _) => crate::extraction::carry_access(
                tx,
                bank_id,
                memory,
                kind,
                at.unwrap_or(0),
                turn.unwrap_or(0),
                source,
            )?,
        }
    }
    tx.execute("DELETE FROM rebuild_accesses WHERE bank_id = ?1", [bank_id])?;
    Ok((carried, dropped))
}

/// `rebuild ack`: the operator has looked at what the sweep would delete
/// after the rebuild (`purge plan`), and the bank's sweep runs again.
pub(crate) fn ack(store: &Store, bank: &str) -> Result<(), RebuildError> {
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, _) = find(&tx, bank)?;
    let Some(row) = row(&tx, bank_id)?.filter(|row| row.acked_at.is_none()) else {
        return Err(RebuildError::NothingToAck);
    };
    if row.completed_at.is_none() {
        return Err(RebuildError::StillRunning);
    }
    tx.execute(
        "UPDATE rebuilds SET acked_at = ?2 WHERE bank_id = ?1",
        (bank_id, micros(store.now())),
    )?;
    log_edit(&tx, store, bank_id, EDIT_REBUILD_ACKED, None, "{}")?;
    tx.commit()?;
    Ok(())
}

/// Where the bank's latest rebuild stands.
pub(crate) fn status(store: &Store, bank: &str) -> Result<RebuildStatus, RebuildError> {
    let conn = store.connection();
    let (bank_id, name) = find(&conn, bank)?;
    let Some(row) = row(&conn, bank_id)? else {
        return Ok(RebuildStatus {
            bank: name,
            state: RebuildState::Idle,
            started_at: None,
            completed_at: None,
            acked_at: None,
            chunks: 0,
            remaining: 0,
            failed: 0,
            replaced: 0,
            kept: 0,
            carried: 0,
            dropped: 0,
            withheld: 0,
        });
    };
    let remaining: i64 = conn.query_row(
        "SELECT COUNT(*) FROM extraction_queue
         WHERE bank_id = ?1 AND kind = 'chunk' AND id <= ?2",
        (bank_id, row.queue_boundary),
        |row| row.get(0),
    )?;
    let failed: i64 = conn.query_row(
        "SELECT COUNT(*) FROM chunks WHERE bank_id = ?1 AND failed_at >= ?2",
        (bank_id, row.started_at),
        |row| row.get(0),
    )?;
    Ok(RebuildStatus {
        bank: name,
        state: match (row.completed_at, row.acked_at) {
            (None, _) => RebuildState::Running,
            (Some(_), None) => RebuildState::Complete,
            (Some(_), Some(_)) => RebuildState::Idle,
        },
        started_at: Some(timestamp(row.started_at)),
        completed_at: row.completed_at.map(timestamp),
        acked_at: row.acked_at.map(timestamp),
        chunks: count(row.chunks),
        remaining: count(remaining),
        failed: count(failed),
        replaced: count(row.replaced),
        kept: count(row.kept),
        carried: count(row.carried),
        dropped: count(row.dropped),
        withheld: count(row.withheld),
    })
}

/// The names of the banks whose rebuild is still running, which the daemon
/// checks at startup in case it stopped between the last chunk and the end.
pub(crate) fn unfinished(conn: &Connection) -> Result<Vec<i64>, rusqlite::Error> {
    let mut statement =
        conn.prepare("SELECT bank_id FROM rebuilds WHERE completed_at IS NULL ORDER BY id")?;
    statement.query_map([], |row| row.get(0))?.collect()
}
