//! Forget, and the one erase path forget and purge share.
//!
//! **Chains.** Both act on whole supersession chains: every memory joined
//! along `superseded_by`, whichever version was named. `ended_by` isn't a
//! chain link, so a memory another one ended stays, keeps its `valid_until`
//! and loses only the pointer.
//!
//! **Forget splits in two.** Everything that can be undone
//! happens when it's called: the chain is hidden (`hidden_at`), which keeps
//! it out of recall, injection, the agenda, refresh inputs and `used`
//! credit; a model citing it has its answer blanked; recall rows naming it are
//! deleted; and it's scrubbed from every stored in-context set. The service
//! scrubs the live sessions and clears the block. Deleting the rows,
//! redacting the passages and the tombstone wait on the bank's queue as an
//! `erase` job, behind every chunk queued before it. Those chunks reconcile
//! against the hidden memory, so a new version of it joins the chain
//! (hidden as it's committed) and a mention leaves its passage in
//! `mention_passages`, credited or not, and the erase takes both. Forget
//! never pauses. It writes a `forget` audit row there and then, which the
//! request turn is linked to when it's ingested ([`crate::ingest`]).
//!
//! **The erase** takes a chain and a reason. In common it deletes the
//! memory rows (their vectors, FTS rows, entity links, accesses, recall
//! results and citations go with them, and `ended_by` and `superseded_by`
//! pointing in are cleared), blanks the answer of every model citing the
//! chain for a refresh, deletes orphan entities, and writes one edit row of ids, never
//! content. Forget also redacts every passage the chain rests on or was
//! mentioned in, deletes the chain's recall rows and scrubs it from stored
//! in-context sets again, for what joined it after the forget. Purge also
//! records each memory's chunk and span in its edit row, and never redacts.
//!
//! **The tombstone.** A redacted passage is masked character for
//! character, so the spans of other memories in the same chunk still hold.
//! The source keeps its key and content hash, and the chunk its hash, so
//! sending the same turn or document again is a duplicate and a later
//! version can't bring the passage back for extraction.
//! A document's other versions hold the passage in their own text, so it's
//! masked wherever it appears verbatim in them, and the masks are recorded
//! in `chunk_redactions` for a version ingested later to apply to its text.

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::{Connection, OptionalExtension, Transaction};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::ingest::{TURN_SEPARATOR, find_bank};
use crate::keep::MAX_IDS;
use crate::queue::QueueError;
use crate::store::bank::log_edit;
use crate::store::{Store, StoreError, VectorIndex, micros};
use crate::strength::{Link, chain};

/// The edit kind forget writes when it's called: the audit row, which the
/// turn that asked for the forget is linked to once it's ingested.
pub const EDIT_FORGET: &str = "forget";

/// The edit kind a forget's erase writes.
pub const EDIT_FORGOTTEN: &str = "forgotten";

/// The edit kind a purge writes.
pub const EDIT_PURGED: &str = "purged";

/// What each character of a redacted passage becomes. Masking character
/// for character keeps every other span in the chunk where it was.
pub const REDACTION_MASK: char = '\u{2588}';

/// What `memory_forget` and `POST /v1/banks/{bank}/forget` take. `session_id`
/// is the Hermes session whose `sync_turn` will
/// carry the request turn, so the audit row can be linked to it;
/// the CLI sends none.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ForgetRequest {
    pub ids: Vec<String>,
    #[serde(default)]
    pub session_id: Option<String>,
}

/// What `forget` returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Forgotten {
    /// Every memory the erase will remove: the whole chain of each named
    /// memory, oldest first.
    pub forgotten: Vec<Uuid>,
    /// The ids that aren't visible memories of the bank, including ones
    /// already forgotten.
    pub unknown: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ForgetError {
    #[error("unknown bank")]
    UnknownBank,

    #[error("at most {MAX_IDS} ids per call, got {given}")]
    TooMany { given: usize },

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for ForgetError {
    fn from(error: rusqlite::Error) -> Self {
        ForgetError::Store(StoreError::Sqlite(error))
    }
}

/// The edit kind removing a document writes: the sources removed and how
/// much went with them, never the document id or its text.
pub const EDIT_DOCUMENT_REMOVED: &str = "document_removed";

/// What `POST /v1/banks/{bank}/documents/remove` takes: the document id
/// exactly as it was ingested. It's never a path segment, since a client
/// normalizes `.` and `..` out of a path before sending it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoveDocumentRequest {
    pub document_id: String,
}

/// What `remove_document` returns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DocumentRemoved {
    /// The id removed, as given, so a client can check it's the one the
    /// owner confirmed.
    pub document_id: String,
    /// Every version of the document, oldest first.
    pub sources: Vec<Uuid>,
    /// Every memory forgotten with it: the whole chain of each memory that
    /// rested on its chunks.
    pub forgotten: Vec<Uuid>,
    /// Chunks taken off the queue before extraction. A chunk in flight
    /// stays, and its commit is refused.
    pub dequeued: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum RemoveDocumentError {
    #[error("unknown bank")]
    UnknownBank,

    #[error("the bank has no such document, or it was removed already")]
    UnknownDocument,

    #[error("give the id of the document to remove")]
    EmptyDocumentId,

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for RemoveDocumentError {
    fn from(error: rusqlite::Error) -> Self {
        RemoveDocumentError::Store(StoreError::Sqlite(error))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EraseReason {
    Forget,
    Purge,
}

impl EraseReason {
    fn as_str(self) -> &'static str {
        match self {
            EraseReason::Forget => "forget",
            EraseReason::Purge => "purge",
        }
    }
}

/// One erase that ran.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Erased {
    pub reason: EraseReason,
    /// Every memory it deleted: the chain as it stood when the erase ran,
    /// including what queued chunks added to it after the forget.
    pub memories: BTreeSet<Uuid>,
}

/// What a forget or an erase leaves the service to do in memory: refresh
/// the models whose answers were blanked, clear the bank's block, and take
/// the memories out of live sessions.
#[derive(Debug, Default)]
pub(crate) struct Aftermath {
    pub models: BTreeSet<i64>,
    pub scrub: BTreeSet<Uuid>,
}

/// Forget's immediate part: hides each named memory's chain and queues its
/// erase. Ids that aren't visible memories of the bank are unknown.
pub(crate) fn forget(
    store: &Store,
    bank: &str,
    request: &ForgetRequest,
) -> Result<(i64, Forgotten, Aftermath), ForgetError> {
    let ids = &request.ids;
    let session = request
        .session_id
        .as_deref()
        .map(str::trim)
        .filter(|session| !session.is_empty());
    if ids.len() > MAX_IDS {
        return Err(ForgetError::TooMany { given: ids.len() });
    }
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, _) = find_bank(&tx, bank)?.ok_or(ForgetError::UnknownBank)?;

    let mut named = Vec::new();
    let mut unknown = Vec::new();
    for id in ids {
        let found: Option<i64> = match id.trim().parse::<Uuid>() {
            Ok(uuid) => tx
                .query_row(
                    "SELECT id FROM memories
                     WHERE uuid = ?1 AND bank_id = ?2 AND hidden_at IS NULL",
                    (uuid.to_string(), bank_id),
                    |row| row.get(0),
                )
                .optional()?,
            Err(_) => None,
        };
        match found {
            Some(memory_id) => named.push(memory_id),
            None => unknown.push(id.clone()),
        }
    }

    let (uuids, members, aftermath) = hide(&tx, store, bank_id, &named, session)?;
    tx.commit()?;
    if !members.is_empty() {
        tracing::info!(
            memories = members.len(),
            "forgot a chain; its erase is queued"
        );
    }
    Ok((
        bank_id,
        Forgotten {
            forgotten: uuids,
            unknown,
        },
        aftermath,
    ))
}

/// Removes every version of the document `document_id` names: forgets the
/// memories resting on its chunks as [`forget`] would, takes its chunks
/// that are waiting off the queue, and clears each version's text now. The
/// sources keep their keys, so sending a version again is a duplicate. A
/// chunk on a lease in `out` (queue rowids) is left to its worker, whose
/// commit is refused because the source is removed. Memories the document
/// only mentioned again are left alone.
pub(crate) fn remove_document(
    store: &Store,
    out: &BTreeSet<i64>,
    bank: &str,
    document_id: &str,
) -> Result<(i64, DocumentRemoved, Aftermath), RemoveDocumentError> {
    if document_id.is_empty() {
        return Err(RemoveDocumentError::EmptyDocumentId);
    }
    let now = micros(store.now());
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, _) = find_bank(&tx, bank)?.ok_or(RemoveDocumentError::UnknownBank)?;
    let sources: Vec<(i64, String)> = {
        let mut statement = tx.prepare(
            "SELECT id, uuid FROM sources
             WHERE bank_id = ?1 AND kind = 'document' AND document_id = ?2
               AND removed_at IS NULL
             ORDER BY ingested_at, id",
        )?;
        statement
            .query_map((bank_id, document_id), |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<_, _>>()?
    };
    if sources.is_empty() {
        return Err(RemoveDocumentError::UnknownDocument);
    }

    let mut dequeued = 0;
    let mut named = Vec::new();
    {
        let mut queued = tx.prepare_cached(
            "SELECT q.id FROM extraction_queue q JOIN chunks c ON c.id = q.chunk_id
             WHERE c.source_id = ?1",
        )?;
        let mut resting = tx.prepare_cached(
            "SELECT m.id FROM memories m JOIN chunks c ON c.id = m.chunk_id
             WHERE c.source_id = ?1 AND m.hidden_at IS NULL ORDER BY m.id",
        )?;
        for (source, _) in &sources {
            let jobs: Vec<i64> = queued
                .query_map([source], |row| row.get(0))?
                .collect::<Result<_, _>>()?;
            for job in jobs.into_iter().filter(|job| !out.contains(job)) {
                tx.execute("DELETE FROM extraction_queue WHERE id = ?1", [job])?;
                dequeued += 1;
            }
            for memory in resting.query_map([source], |row| row.get::<_, i64>(0))? {
                named.push(memory?);
            }
        }
    }
    let (forgotten, members, aftermath) = hide(&tx, store, bank_id, &named, None)?;

    for (source, _) in &sources {
        tx.execute(
            "UPDATE chunks SET text = NULL, call1_output = NULL, failed_at = NULL,
                    tombstoned_at = COALESCE(tombstoned_at, ?2)
             WHERE source_id = ?1",
            (source, now),
        )?;
        tx.execute(
            "UPDATE sources SET text = NULL, reply = NULL,
                    tombstoned_at = COALESCE(tombstoned_at, ?2), removed_at = ?2
             WHERE id = ?1",
            (source, now),
        )?;
    }
    let uuids: Vec<Uuid> = sources
        .iter()
        .map(|(_, uuid)| uuid.parse().expect("a stored uuid parses"))
        .collect();
    log_edit(
        &tx,
        store,
        bank_id,
        EDIT_DOCUMENT_REMOVED,
        None,
        &serde_json::json!({
            "sources": uuids,
            "memories": members.len(),
            "dequeued": dequeued,
        })
        .to_string(),
    )?;
    tx.commit()?;
    tracing::info!(
        sources = uuids.len(),
        memories = members.len(),
        dequeued,
        "removed a document"
    );
    Ok((
        bank_id,
        DocumentRemoved {
            document_id: document_id.to_string(),
            sources: uuids,
            forgotten,
            dequeued,
        },
        aftermath,
    ))
}

/// Forget's immediate part for the memories with rowids `named`, inside the
/// caller's transaction: hides their whole chains, drops what cites them,
/// and queues their erase. Returns the chains' ids and rowids, and what the
/// service settles.
fn hide(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    named: &[i64],
    session: Option<&str>,
) -> Result<(Vec<Uuid>, BTreeSet<i64>, Aftermath), rusqlite::Error> {
    let now = micros(store.now());
    let links = bank_links(tx, bank_id)?;
    let members: BTreeSet<i64> = named.iter().flat_map(|&id| chain(&links, id)).collect();
    let uuids = uuids_of(tx, &members)?;
    let mut aftermath = Aftermath::default();
    if !members.is_empty() {
        let mut hide = tx.prepare_cached(
            "UPDATE memories SET hidden_at = ?2, updated_at = ?2
             WHERE id = ?1 AND hidden_at IS NULL",
        )?;
        for member in &members {
            hide.execute((member, now))?;
        }
        drop(hide);
        aftermath.models = blank_answers(tx, &members)?;
        delete_recalls(tx, &members)?;
        let scrub: BTreeSet<Uuid> = uuids.iter().copied().collect();
        scrub_stored(tx, bank_id, &scrub)?;
        aftermath.scrub = scrub;
        // The audit row, written now so a crash before the
        // erase still leaves it. `after` is the session's latest turn so
        // far: the request turn is the next one, whatever the clock says.
        let after: Option<i64> = match session {
            Some(session) => Some(tx.query_row(
                "SELECT COALESCE(MAX(id), 0) FROM sources
                 WHERE bank_id = ?1 AND kind = 'turn' AND session_id = ?2",
                (bank_id, session),
                |row| row.get(0),
            )?),
            None => None,
        };
        log_edit(
            tx,
            store,
            bank_id,
            EDIT_FORGET,
            None,
            &serde_json::json!({
                "memories": uuids,
                "session_id": session,
                "after": after,
                "request": null,
            })
            .to_string(),
        )?;
        tx.execute(
            "INSERT INTO extraction_queue (bank_id, kind, memory_ids, reason, priority,
                                           observed_at, enqueued_at)
             VALUES (?1, 'erase', ?2, 'forget', 0, ?3, ?3)",
            (
                bank_id,
                serde_json::to_string(&members).expect("rowids serialise"),
                now,
            ),
        )?;
    }
    Ok((uuids, members, aftermath))
}

/// Runs the erase at the head of the bank's queue once every chunk queued
/// before it has left the queue, extracted or failed. `None` when there's
/// no erase, or a chunk queued before it is still waiting or in flight.
pub(crate) fn erase_next(
    store: &Store,
    bank: &str,
) -> Result<Option<(i64, Erased, Aftermath)>, QueueError> {
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, _) = find_bank(&tx, bank)?.ok_or(QueueError::UnknownBank)?;
    let job: Option<(i64, String, String)> = tx
        .query_row(
            "SELECT id, memory_ids, reason FROM extraction_queue
             WHERE bank_id = ?1 AND kind = 'erase'
             ORDER BY id LIMIT 1",
            [bank_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    let Some((job_id, memory_ids, reason)) = job else {
        return Ok(None);
    };
    let waiting = tx
        .query_row(
            "SELECT 1 FROM extraction_queue
             WHERE bank_id = ?1 AND kind = 'chunk' AND id < ?2 LIMIT 1",
            (bank_id, job_id),
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    if waiting {
        return Ok(None);
    }
    let reason = if reason == "purge" {
        EraseReason::Purge
    } else {
        EraseReason::Forget
    };
    let named: Vec<i64> = serde_json::from_str(&memory_ids).unwrap_or_default();
    let links = bank_links(&tx, bank_id)?;
    let mut members = BTreeSet::new();
    for id in named {
        let exists = tx
            .query_row("SELECT 1 FROM memories WHERE id = ?1", [id], |_| Ok(()))
            .optional()?
            .is_some();
        if exists {
            members.extend(chain(&links, id));
        }
    }
    let (memories, aftermath) = erase_chain(&tx, store, bank_id, &members, reason)?;
    tx.execute("DELETE FROM extraction_queue WHERE id = ?1", [job_id])?;
    tx.commit()?;
    tracing::info!(
        memories = memories.len(),
        reason = reason.as_str(),
        "erased a chain"
    );
    Ok(Some((bank_id, Erased { reason, memories }, aftermath)))
}

/// The erase both reasons share, inside the caller's transaction:
/// `members` is a whole chain, or several.
pub(crate) fn erase_chain(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    members: &BTreeSet<i64>,
    reason: EraseReason,
) -> Result<(BTreeSet<Uuid>, Aftermath), rusqlite::Error> {
    let mut aftermath = Aftermath::default();
    if members.is_empty() {
        return Ok((BTreeSet::new(), aftermath));
    }
    let rows = passages(tx, members)?;
    let uuids: BTreeSet<Uuid> = rows.iter().map(|row| row.memory).collect();

    let mut entities = BTreeSet::new();
    {
        let mut linked =
            tx.prepare_cached("SELECT entity_id FROM memory_entities WHERE memory_id = ?1")?;
        for member in members {
            for entity in linked.query_map([member], |row| row.get::<_, i64>(0))? {
                entities.insert(entity?);
            }
        }
    }

    aftermath.models = blank_answers(tx, members)?;
    let mut redacted = Redacted::default();
    if reason == EraseReason::Forget {
        redacted = redact_chain(tx, members, &rows)?;
        delete_recalls(tx, members)?;
        scrub_stored(tx, bank_id, &uuids)?;
        aftermath.scrub = uuids.clone();
    }

    let vectors = store.vectors();
    for member in members {
        vectors.remove(tx, *member).map_err(|error| match error {
            crate::store::VectorError::Sqlite(error) => error,
            other => rusqlite::Error::ToSqlConversionFailure(Box::new(other)),
        })?;
        tx.execute("DELETE FROM memories WHERE id = ?1", [member])?;
        // A re-embed may have staged its vector too.
        tx.execute("DELETE FROM reembed_vectors WHERE memory_id = ?1", [member])?;
    }
    delete_orphans(tx, &entities)?;

    let details = match reason {
        EraseReason::Forget => serde_json::json!({
            "memories": uuids,
            "legacy_mentions": redacted.legacy_mentions,
            "whole_source": redacted.whole_source,
        }),
        EraseReason::Purge => serde_json::json!({
            "memories": rows
                .iter()
                .map(|row| serde_json::json!({
                    "memory": row.memory,
                    "chunk": row.chunk,
                    "start": row.start,
                    "end": row.end,
                }))
                .collect::<Vec<_>>(),
        }),
    };
    let kind = match reason {
        EraseReason::Forget => EDIT_FORGOTTEN,
        EraseReason::Purge => EDIT_PURGED,
    };
    log_edit(tx, store, bank_id, kind, None, &details.to_string())?;
    Ok((uuids, aftermath))
}

/// Every `superseded_by` link of the bank.
pub(crate) fn bank_links(conn: &Connection, bank_id: i64) -> Result<Vec<Link>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT id, superseded_by, ended_by FROM memories
         WHERE bank_id = ?1 AND superseded_by IS NOT NULL",
    )?;
    statement
        .query_map([bank_id], |row| {
            Ok(Link {
                id: row.get(0)?,
                superseded_by: row.get(1)?,
                ended_by: row.get(2)?,
            })
        })?
        .collect()
}

fn uuids_of(conn: &Connection, members: &BTreeSet<i64>) -> Result<Vec<Uuid>, rusqlite::Error> {
    let mut statement = conn.prepare_cached("SELECT uuid FROM memories WHERE id = ?1")?;
    let mut uuids = Vec::with_capacity(members.len());
    for member in members {
        let uuid: String = statement.query_row([member], |row| row.get(0))?;
        uuids.push(uuid.parse().expect("a stored uuid parses"));
    }
    Ok(uuids)
}

/// A memory's own passage.
struct Passage {
    memory: Uuid,
    chunk_id: i64,
    chunk: Uuid,
    start: usize,
    end: usize,
}

fn passages(conn: &Connection, members: &BTreeSet<i64>) -> Result<Vec<Passage>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT m.uuid, m.chunk_id, c.uuid, m.source_start, m.source_end
         FROM memories m JOIN chunks c ON c.id = m.chunk_id WHERE m.id = ?1",
    )?;
    let mut rows = Vec::with_capacity(members.len());
    for member in members {
        rows.push(statement.query_row([member], |row| {
            Ok(Passage {
                memory: row
                    .get::<_, String>(0)?
                    .parse()
                    .expect("a stored uuid parses"),
                chunk_id: row.get(1)?,
                chunk: row
                    .get::<_, String>(2)?
                    .parse()
                    .expect("a stored uuid parses"),
                start: usize::try_from(row.get::<_, i64>(3)?).unwrap_or(0),
                end: usize::try_from(row.get::<_, i64>(4)?).unwrap_or(0),
            })
        })?);
    }
    Ok(rows)
}

/// Blanks the answer of every model citing any of `members`, deletes its
/// citations, and returns the models: the answer restates what it cites,
/// so it goes with the memory, and the model refreshes.
fn blank_answers(
    conn: &Connection,
    members: &BTreeSet<i64>,
) -> Result<BTreeSet<i64>, rusqlite::Error> {
    let mut citing =
        conn.prepare_cached("SELECT model_id FROM mental_model_cites WHERE memory_id = ?1")?;
    let mut models = BTreeSet::new();
    for member in members {
        for model in citing.query_map([member], |row| row.get::<_, i64>(0))? {
            models.insert(model?);
        }
    }
    for model in &models {
        conn.execute(
            "UPDATE mental_models SET answer = NULL WHERE id = ?1",
            [model],
        )?;
        conn.execute(
            "DELETE FROM mental_model_cites WHERE model_id = ?1",
            [model],
        )?;
    }
    Ok(models)
}

/// Deletes every recall row whose results name any of `members`: the
/// query is the user's message, so it can restate what's forgotten.
fn delete_recalls(conn: &Connection, members: &BTreeSet<i64>) -> Result<(), rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "DELETE FROM recalls WHERE id IN (SELECT recall_id FROM recall_results WHERE memory_id = ?1)",
    )?;
    for member in members {
        statement.execute([member])?;
    }
    Ok(())
}

/// Takes `memories` out of every stored in-context set of the bank: queued
/// and failed turns' sets, built blocks, and session mappings (schema
/// versions 5 and 6).
fn scrub_stored(
    conn: &Connection,
    bank_id: i64,
    memories: &BTreeSet<Uuid>,
) -> Result<(), rusqlite::Error> {
    let keep_ids = |json: &str| -> Option<String> {
        let ids: Vec<Uuid> = serde_json::from_str(json).ok()?;
        let kept: Vec<&Uuid> = ids.iter().filter(|id| !memories.contains(id)).collect();
        (kept.len() != ids.len()).then(|| serde_json::to_string(&kept).expect("ids serialise"))
    };
    let tables = [
        (
            "SELECT t.id, t.memories FROM turn_in_context t
             JOIN sources s ON s.id = t.source_id WHERE s.bank_id = ?1",
            "UPDATE turn_in_context SET memories = ?2 WHERE id = ?1",
        ),
        (
            "SELECT id, in_context FROM prompt_blocks WHERE bank_id = ?1",
            "UPDATE prompt_blocks SET in_context = ?2 WHERE id = ?1",
        ),
        (
            "SELECT rowid, cited FROM session_blocks WHERE bank_id = ?1",
            "UPDATE session_blocks SET cited = ?2 WHERE rowid = ?1",
        ),
    ];
    for (select, update) in tables {
        let rows: Vec<(i64, String)> = {
            let mut statement = conn.prepare_cached(select)?;
            statement
                .query_map([bank_id], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<_, _>>()?
        };
        for (id, json) in rows {
            if let Some(kept) = keep_ids(&json) {
                conn.execute(update, (id, kept))?;
            }
        }
    }
    Ok(())
}

/// What a forget's redaction did beyond the chain's recorded passages.
#[derive(Debug, Default)]
struct Redacted {
    /// Mentions stored before version 7, which have no span.
    legacy_mentions: usize,
    /// Documents masked whole for one, because no exact passage was found.
    whole_source: usize,
}

/// Masks every passage the chain rests on or was restated in: the members'
/// own spans and their recorded mention passages. A mention from before
/// version 7 has no span, so its source is masked more widely rather than
/// not at all: a turn except what surviving memories
/// rest on, and a document at the chain's own passages wherever they appear
/// verbatim, or else whole, again except surviving passages.
fn redact_chain(
    tx: &Transaction<'_>,
    members: &BTreeSet<i64>,
    rows: &[Passage],
) -> Result<Redacted, rusqlite::Error> {
    let mut spans: BTreeMap<i64, Vec<(usize, usize)>> = BTreeMap::new();
    for row in rows {
        spans
            .entry(row.chunk_id)
            .or_default()
            .push((row.start, row.end));
    }
    let mut legacy_sources: Vec<i64> = Vec::new();
    {
        let mut mentions = tx.prepare_cached(
            "SELECT chunk_id, start_offset, end_offset FROM mention_passages WHERE memory_id = ?1",
        )?;
        let mut legacy = tx.prepare_cached(
            "SELECT a.source_id FROM accesses a
             WHERE a.memory_id = ?1 AND a.kind IN ('mentioned_again', 'confirmed')
               AND a.source_id IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM mention_passages p
                               JOIN chunks c ON c.id = p.chunk_id
                               WHERE p.memory_id = a.memory_id AND c.source_id = a.source_id)",
        )?;
        for member in members {
            for row in mentions.query_map([member], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })? {
                let (chunk, start, end) = row?;
                spans
                    .entry(chunk)
                    .or_default()
                    .push((offset(start), offset(end)));
            }
            for source in legacy.query_map([member], |row| row.get::<_, i64>(0))? {
                legacy_sources.push(source?);
            }
        }
    }

    let mut redacted = Redacted {
        legacy_mentions: legacy_sources.len(),
        whole_source: 0,
    };
    if !legacy_sources.is_empty() {
        // The chain's own passages, read before anything is masked.
        let own: Vec<String> = rows
            .iter()
            .filter_map(|row| {
                chunk_text(tx, row.chunk_id)
                    .ok()
                    .flatten()
                    .map(|text| slice(&text, row.start, row.end))
            })
            .filter(|passage| !passage.trim().is_empty())
            .collect();
        legacy_sources.sort_unstable();
        legacy_sources.dedup();
        for source in legacy_sources {
            let kind: String =
                tx.query_row("SELECT kind FROM sources WHERE id = ?1", [source], |row| {
                    row.get(0)
                })?;
            let chunks = chunks_of(tx, source)?;
            let found: Vec<(i64, Vec<(usize, usize)>)> = if kind == "document" {
                chunks
                    .iter()
                    .map(|(chunk, text)| {
                        let hits = own
                            .iter()
                            .flat_map(|passage| occurrences(text, passage))
                            .collect();
                        (*chunk, hits)
                    })
                    .filter(|(_, hits): &(i64, Vec<(usize, usize)>)| !hits.is_empty())
                    .collect()
            } else {
                Vec::new()
            };
            if !found.is_empty() {
                for (chunk, hits) in found {
                    spans.entry(chunk).or_default().extend(hits);
                }
                continue;
            }
            if kind == "document" {
                redacted.whole_source += 1;
            }
            for (chunk, text) in &chunks {
                let keep = surviving(tx, *chunk, members)?;
                spans
                    .entry(*chunk)
                    .or_default()
                    .extend(complement(text.chars().count(), &keep));
            }
        }
    }
    redact(tx, &spans)?;
    Ok(redacted)
}

fn offset(value: i64) -> usize {
    usize::try_from(value).unwrap_or(0)
}

fn chunk_text(conn: &Connection, chunk: i64) -> Result<Option<String>, rusqlite::Error> {
    conn.query_row("SELECT text FROM chunks WHERE id = ?1", [chunk], |row| {
        row.get(0)
    })
}

/// A source's chunks that still have text.
fn chunks_of(conn: &Connection, source: i64) -> Result<Vec<(i64, String)>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT id, text FROM chunks WHERE source_id = ?1 AND text IS NOT NULL ORDER BY position",
    )?;
    statement
        .query_map([source], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect()
}

/// The own passages of memories outside `members` that rest on `chunk`.
fn surviving(
    conn: &Connection,
    chunk: i64,
    members: &BTreeSet<i64>,
) -> Result<Vec<(usize, usize)>, rusqlite::Error> {
    let mut statement = conn
        .prepare_cached("SELECT id, source_start, source_end FROM memories WHERE chunk_id = ?1")?;
    let rows: Vec<(i64, i64, i64)> = statement
        .query_map([chunk], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
        .collect::<Result<_, _>>()?;
    Ok(rows
        .into_iter()
        .filter(|(id, _, _)| !members.contains(id))
        .map(|(_, start, end)| (offset(start), offset(end)))
        .collect())
}

/// Every span of `0..len` outside `keep`.
fn complement(len: usize, keep: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut keep = keep.to_vec();
    keep.sort_unstable();
    let mut spans = Vec::new();
    let mut at = 0;
    for (start, end) in keep {
        if start > at {
            spans.push((at, start.min(len)));
        }
        at = at.max(end);
    }
    if at < len {
        spans.push((at, len));
    }
    spans
}

/// The characters `start..end` of `text`.
fn slice(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

/// Where `needle` appears in `text`, in characters, overlapping included.
/// A mask on either side matches any character: an earlier forget masked
/// part of the same passage there, or in the text the needle was read from,
/// and that mustn't hide the rest of it. At
/// least one character that isn't a mask or whitespace has to match, so a
/// run of masks finds nothing. Matching only ever widens what's masked.
fn occurrences(text: &str, needle: &str) -> Vec<(usize, usize)> {
    let needle: Vec<char> = needle.chars().collect();
    if !needle
        .iter()
        .any(|c| *c != REDACTION_MASK && !c.is_whitespace())
    {
        return Vec::new();
    }
    let text: Vec<char> = text.chars().collect();
    if needle.len() > text.len() {
        return Vec::new();
    }
    (0..=text.len() - needle.len())
        .filter(|&start| {
            let mut anchored = false;
            for (n, t) in needle.iter().zip(&text[start..]) {
                if *n == REDACTION_MASK || *t == REDACTION_MASK {
                    continue;
                }
                if n != t {
                    return false;
                }
                anchored |= !n.is_whitespace();
            }
            anchored
        })
        .map(|start| (start, start + needle.len()))
        .collect()
}

/// Character spans, start inclusive and end exclusive.
type Spans = Vec<(usize, usize)>;

/// What one redaction masks: chunk-relative spans by chunk rowid, and spans
/// of each source's `text` and `reply` by source rowid.
#[derive(Debug, Default)]
struct Masks {
    chunks: BTreeMap<i64, Spans>,
    sources: BTreeMap<i64, (Spans, Spans)>,
}

/// Masks `spans` (chunk-relative characters, by chunk rowid) in each
/// chunk's text and in its source, and records them in `chunk_redactions`
/// so a document version ingested later masks the same characters. A
/// turn's chunk is the message, the separator and the reply, so a span maps
/// into `text` or `reply`; a document's chunk starts at its `start_offset`
/// in the source text. Text already swept is left as it is. Call 1's saved
/// reply goes too, since it quotes the chunk.
///
/// A document's other versions hold the same passages in their own text:
/// in chunks that changed, and in sections they share with this one, which
/// have no chunk row of their own. Each passage is masked wherever it
/// appears in them too ([`occurrences`]).
///
/// Every target is read before anything is masked, and each is masked once
/// with the union of its spans. Masking one
/// passage first would hide a longer one that contains it from the search
/// in the other versions.
fn redact(
    conn: &Connection,
    spans: &BTreeMap<i64, Vec<(usize, usize)>>,
) -> Result<(), rusqlite::Error> {
    let mut masks = Masks::default();
    let mut passages: BTreeMap<(i64, String), Vec<String>> = BTreeMap::new();
    for (&chunk_id, spans) in spans {
        let (source_id, start_offset, text): (i64, i64, Option<String>) = conn.query_row(
            "SELECT source_id, start_offset, text FROM chunks WHERE id = ?1",
            [chunk_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        masks
            .chunks
            .entry(chunk_id)
            .or_default()
            .extend_from_slice(spans);
        let (kind, document_id, source_text): (String, Option<String>, Option<String>) = conn
            .query_row(
                "SELECT kind, document_id, text FROM sources WHERE id = ?1",
                [source_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
        let (in_text, in_reply) = masks.sources.entry(source_id).or_default();
        if kind == "turn" {
            let message = source_text
                .as_deref()
                .map_or(0, |text| text.chars().count());
            let reply_start = message + TURN_SEPARATOR.chars().count();
            in_text.extend(
                spans
                    .iter()
                    .filter(|(start, _)| *start < message)
                    .map(|&(start, end)| (start, end.min(message))),
            );
            in_reply.extend(
                spans
                    .iter()
                    .filter(|(_, end)| *end > reply_start)
                    .map(|&(start, end)| (start.max(reply_start) - reply_start, end - reply_start)),
            );
        } else {
            let offset = offset(start_offset);
            in_text.extend(
                spans
                    .iter()
                    .map(|&(start, end)| (start + offset, end + offset)),
            );
        }
        if let (Some(document_id), Some(text)) = (document_id, text) {
            passages
                .entry((source_id, document_id))
                .or_default()
                .extend(
                    spans
                        .iter()
                        .map(|&(start, end)| slice(&text, start, end))
                        .filter(|passage| !passage.trim().is_empty()),
                );
        }
    }
    for ((source_id, document_id), passages) in &passages {
        versions(conn, *source_id, document_id, passages, &mut masks)?;
    }

    for (chunk_id, spans) in &masks.chunks {
        let text: Option<String> = chunk_text(conn, *chunk_id)?;
        mask_chunk(conn, *chunk_id, text.as_deref(), spans)?;
    }
    for (source_id, (in_text, in_reply)) in &masks.sources {
        let (text, reply): (Option<String>, Option<String>) = conn.query_row(
            "SELECT text, reply FROM sources WHERE id = ?1",
            [source_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        conn.execute(
            "UPDATE sources SET text = ?2, reply = ?3 WHERE id = ?1",
            (
                source_id,
                text.map(|text| mask(&text, in_text)),
                reply.map(|reply| mask(&reply, in_reply)),
            ),
        )?;
    }
    Ok(())
}

/// Masks `spans` in a chunk's text, drops call 1's saved reply, and records
/// the spans.
fn mask_chunk(
    conn: &Connection,
    chunk_id: i64,
    text: Option<&str>,
    spans: &[(usize, usize)],
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "UPDATE chunks SET text = ?2, call1_output = NULL WHERE id = ?1",
        (chunk_id, text.map(|text| mask(text, spans))),
    )?;
    let stored: Option<String> = conn
        .query_row(
            "SELECT spans FROM chunk_redactions WHERE chunk_id = ?1",
            [chunk_id],
            |row| row.get(0),
        )
        .optional()?;
    let mut all: Vec<(usize, usize)> = stored
        .and_then(|stored| serde_json::from_str(&stored).ok())
        .unwrap_or_default();
    all.extend_from_slice(spans);
    all.sort_unstable();
    all.dedup();
    conn.execute(
        "INSERT INTO chunk_redactions (chunk_id, spans) VALUES (?1, ?2)
         ON CONFLICT (chunk_id) DO UPDATE SET spans = excluded.spans",
        (
            chunk_id,
            serde_json::to_string(&all).expect("spans serialise"),
        ),
    )?;
    Ok(())
}

/// Adds to `masks` every place `passages` appear in the other versions of a
/// document, read as they are before this redaction masks anything: their
/// source text and their chunks.
fn versions(
    conn: &Connection,
    source_id: i64,
    document_id: &str,
    passages: &[String],
    masks: &mut Masks,
) -> Result<(), rusqlite::Error> {
    let others: Vec<(i64, Option<String>)> = {
        let mut statement = conn.prepare_cached(
            "SELECT id, text FROM sources
             WHERE bank_id = (SELECT bank_id FROM sources WHERE id = ?1)
               AND kind = 'document' AND document_id = ?2 AND id != ?1",
        )?;
        statement
            .query_map((source_id, document_id), |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?
            .collect::<Result<_, _>>()?
    };
    let hits = |text: &str| -> Vec<(usize, usize)> {
        passages
            .iter()
            .flat_map(|passage| occurrences(text, passage))
            .collect()
    };
    for (version, text) in others {
        if let Some(text) = text {
            let found = hits(&text);
            if !found.is_empty() {
                masks.sources.entry(version).or_default().0.extend(found);
            }
        }
        for (chunk, text) in chunks_of(conn, version)? {
            let found = hits(&text);
            if !found.is_empty() {
                masks.chunks.entry(chunk).or_default().extend(found);
            }
        }
    }
    Ok(())
}

/// The masks recorded for chunks of a document with `content_hash`, from
/// any of its versions: what a version sharing that chunk masks in its own
/// text when it's ingested (schema version 8).
pub(crate) fn recorded_redactions(
    conn: &Connection,
    bank_id: i64,
    document_id: &str,
    content_hash: &str,
) -> Result<Vec<(usize, usize)>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT r.spans FROM chunk_redactions r
         JOIN chunks c ON c.id = r.chunk_id
         JOIN sources s ON s.id = c.source_id
         WHERE c.bank_id = ?1 AND c.content_hash = ?2
           AND s.kind = 'document' AND s.document_id = ?3",
    )?;
    let mut all = Vec::new();
    for spans in statement.query_map((bank_id, content_hash, document_id), |row| {
        row.get::<_, String>(0)
    })? {
        let spans: Vec<(usize, usize)> = serde_json::from_str(&spans?).unwrap_or_default();
        all.extend(spans);
    }
    Ok(all)
}

/// `text` with every character inside `spans` replaced by
/// [`REDACTION_MASK`], except line breaks, which keep a document's lines.
pub(crate) fn mask(text: &str, spans: &[(usize, usize)]) -> String {
    text.chars()
        .enumerate()
        .map(|(index, c)| {
            if c != '\n'
                && spans
                    .iter()
                    .any(|&(start, end)| start <= index && index < end)
            {
                REDACTION_MASK
            } else {
                c
            }
        })
        .collect()
}

/// Deletes those of `entities` nothing rests on any more, with their
/// aliases. Kept: the seeded `user` and `assistant`, a merge tombstone
/// (`merged_into` set) and an entity another was merged into, one a mental
/// model's filter names, and a speaker, whose platform
/// id resolves to it.
fn delete_orphans(conn: &Connection, entities: &BTreeSet<i64>) -> Result<(), rusqlite::Error> {
    let mut orphan = conn.prepare_cached(
        "SELECT 1 FROM entities e
         WHERE e.id = ?1 AND e.seeded IS NULL AND e.merged_into IS NULL
           AND NOT EXISTS (SELECT 1 FROM memory_entities WHERE entity_id = e.id)
           AND NOT EXISTS (SELECT 1 FROM entities x WHERE x.merged_into = e.id)
           AND NOT EXISTS (SELECT 1 FROM mental_models WHERE filter_entity_id = e.id)
           AND NOT EXISTS (SELECT 1 FROM speaker_ids WHERE entity_id = e.id)",
    )?;
    for &entity in entities {
        if orphan.query_row([entity], |_| Ok(())).optional()?.is_none() {
            continue;
        }
        conn.execute("DELETE FROM entity_aliases WHERE entity_id = ?1", [entity])?;
        conn.execute("DELETE FROM entities WHERE id = ?1", [entity])?;
    }
    Ok(())
}

/// The daemon-wide edit kind a bank deletion writes.
pub const EDIT_BANK_DELETED: &str = "bank_deleted";

/// What `bank delete` did: counts only, as its `bank_deleted` row holds
/// them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BankDeleted {
    pub bank: Uuid,
    pub name: String,
    pub memories: usize,
    pub entities: usize,
    pub sources: usize,
    pub chunks: usize,
    pub edits: usize,
    pub models: usize,
    pub recalls: usize,
    pub sessions: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum BankDeleteError {
    #[error("unknown bank")]
    UnknownBank,

    #[error("--confirm must repeat the bank's name")]
    NotConfirmed,

    #[error("the bank's extraction didn't finish its chunk in time; try again")]
    Busy,

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for BankDeleteError {
    fn from(error: rusqlite::Error) -> Self {
        BankDeleteError::Store(StoreError::Sqlite(error))
    }
}

/// Deletes a bank. Every memory goes
/// through the erase path as one purge, which blanks the model answers
/// citing them and drops their vectors; then everything else the bank holds goes
/// too, its tombstones, edit rows and session mappings included, and one
/// daemon-wide `bank_deleted` row records the counts. The caller holds the
/// bank's lease, so no chunk is in flight.
pub(crate) fn delete_bank(
    store: &Store,
    bank: &str,
) -> Result<(i64, BankDeleted), BankDeleteError> {
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, _) = find_bank(&tx, bank)?.ok_or(BankDeleteError::UnknownBank)?;
    let (uuid, name): (String, String) = tx.query_row(
        "SELECT uuid, name FROM banks WHERE id = ?1",
        [bank_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let count = |sql: &str| -> Result<usize, rusqlite::Error> {
        tx.query_row(sql, [bank_id], |row| row.get::<_, i64>(0))
            .map(|count| usize::try_from(count).unwrap_or(0))
    };
    let entities = count("SELECT COUNT(*) FROM entities WHERE bank_id = ?1")?;
    let sources = count("SELECT COUNT(*) FROM sources WHERE bank_id = ?1")?;
    let chunks = count("SELECT COUNT(*) FROM chunks WHERE bank_id = ?1")?;
    let models = count("SELECT COUNT(*) FROM mental_models WHERE bank_id = ?1")?;
    let recalls = count("SELECT COUNT(*) FROM recalls WHERE bank_id = ?1")?;
    let sessions = count("SELECT COUNT(*) FROM session_blocks WHERE bank_id = ?1")?;

    let members: BTreeSet<i64> = {
        let mut statement = tx.prepare("SELECT id FROM memories WHERE bank_id = ?1")?;
        statement
            .query_map([bank_id], |row| row.get(0))?
            .collect::<Result<_, _>>()?
    };
    let (memories, _) = erase_chain(&tx, store, bank_id, &members, EraseReason::Purge)?;
    let edits = count("SELECT COUNT(*) FROM edits WHERE bank_id = ?1")?;

    for sql in [
        "DELETE FROM reembed_vectors WHERE bank_id = ?1",
        "DELETE FROM reembeds WHERE bank_id = ?1",
        "DELETE FROM mental_models WHERE bank_id = ?1",
        "DELETE FROM recalls WHERE bank_id = ?1",
        "DELETE FROM session_blocks WHERE bank_id = ?1",
        "DELETE FROM prompt_blocks WHERE bank_id = ?1",
        "DELETE FROM sweep_runs WHERE bank_id = ?1",
        "DELETE FROM sweep_progress WHERE bank_id = ?1",
        "DELETE FROM extraction_queue WHERE bank_id = ?1",
        "DELETE FROM accesses WHERE bank_id = ?1",
        "DELETE FROM pending_credits WHERE bank_id = ?1",
        "DELETE FROM mention_passages
         WHERE chunk_id IN (SELECT id FROM chunks WHERE bank_id = ?1)",
        "DELETE FROM chunks WHERE bank_id = ?1",
        "DELETE FROM sources WHERE bank_id = ?1",
        "DELETE FROM speaker_ids WHERE bank_id = ?1",
        "DELETE FROM edits WHERE bank_id = ?1",
        "DELETE FROM memory_entities
         WHERE entity_id IN (SELECT id FROM entities WHERE bank_id = ?1)",
        "DELETE FROM entity_aliases WHERE bank_id = ?1",
        "UPDATE entities SET merged_into = NULL WHERE bank_id = ?1",
        "DELETE FROM entities WHERE bank_id = ?1",
        "DELETE FROM banks WHERE id = ?1",
    ] {
        tx.execute(sql, [bank_id])?;
    }

    let deleted = BankDeleted {
        bank: uuid.parse().expect("a stored uuid parses"),
        name,
        memories: memories.len(),
        entities,
        sources,
        chunks,
        edits,
        models,
        recalls,
        sessions,
    };
    tx.execute(
        "INSERT INTO edits (uuid, bank_id, kind, details, at) VALUES (?1, NULL, ?2, ?3, ?4)",
        (
            store.new_id().to_string(),
            EDIT_BANK_DELETED,
            serde_json::to_string(&deleted).expect("counts serialise"),
            micros(store.now()),
        ),
    )?;
    tx.commit()?;
    tracing::info!(bank = %deleted.bank, memories = deleted.memories, "deleted a bank");
    Ok((bank_id, deleted))
}
