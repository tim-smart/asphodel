//! Forget, and the one erase path forget and purge share ("Deletion
//! policy", TIM-97, decisions 2 and 5; ADR 0008; ADR 0010, "Forgetting";
//! "Erase path, forget, purge and the nightly sweep", TIM-112).
//!
//! **Chains.** Both act on whole supersession chains: every memory joined
//! along `superseded_by`, whichever version was named. `ended_by` isn't a
//! chain link, so a memory another one ended stays, keeps its `valid_until`
//! and loses only the pointer.
//!
//! **Forget splits in two** (ADR 0010). Everything that can be undone
//! happens when it's called: the chain is hidden (`hidden_at`), which keeps
//! it out of recall, injection, the agenda, refresh inputs and `used`
//! credit; model entries citing it are dropped; recall rows naming it are
//! deleted; and it's scrubbed from every stored in-context set. The service
//! scrubs the live sessions and clears the block. Deleting the rows,
//! redacting the passages and the tombstone wait on the bank's queue as an
//! `erase` job, behind every chunk queued before it. Those chunks reconcile
//! against the hidden memory, so a new version of it joins the chain
//! (hidden as it's committed) and a mention leaves an access with its
//! spans, and the erase takes both. Forget never pauses.
//!
//! **The erase** takes a chain and a reason. In common it deletes the
//! memory rows (their vectors, FTS rows, entity links, accesses, recall
//! results and citations go with them, and `ended_by` and `superseded_by`
//! pointing in are cleared), drops model entries citing the chain for a
//! refresh, deletes orphan entities, and writes one edit row of ids, never
//! content. Forget also redacts every passage the chain rests on or was
//! mentioned in, deletes the chain's recall rows and scrubs it from stored
//! in-context sets again, for what joined it after the forget. Purge also
//! records each memory's chunk and span in its edit row, and never redacts.
//!
//! **The tombstone.** A redacted passage is masked character for
//! character, so the spans of other memories in the same chunk still hold.
//! The source keeps its key and content hash, and the chunk its hash, so
//! sending the same turn or document again is a duplicate and a later
//! version can't bring the passage back (ADR 0002, TIM-92).

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::{Connection, OptionalExtension, Transaction};
use serde::Serialize;
use uuid::Uuid;

use crate::ingest::{TURN_SEPARATOR, find_bank};
use crate::keep::MAX_IDS;
use crate::queue::QueueError;
use crate::store::bank::log_edit;
use crate::store::{Store, StoreError, VectorIndex, micros};
use crate::strength::{Link, chain};
use crate::system_prompt::BlockEntry;

/// The edit kind a forget's erase writes.
pub const EDIT_FORGOTTEN: &str = "forgotten";

/// The edit kind a purge writes.
pub const EDIT_PURGED: &str = "purged";

/// What each character of a redacted passage becomes. Masking character
/// for character keeps every other span in the chunk where it was.
pub const REDACTION_MASK: char = '\u{2588}';

/// What `forget` returns (TIM-94, decision 9, as amended by TIM-97).
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
/// the models whose entries went, clear the bank's block, and take the
/// memories out of live sessions.
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
    ids: &[String],
) -> Result<(i64, Forgotten, Aftermath), ForgetError> {
    if ids.len() > MAX_IDS {
        return Err(ForgetError::TooMany { given: ids.len() });
    }
    let now = micros(store.now());
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

    let links = bank_links(&tx, bank_id)?;
    let members: BTreeSet<i64> = named.iter().flat_map(|&id| chain(&links, id)).collect();
    let uuids = uuids_of(&tx, &members)?;
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
        aftermath.models = drop_entries(&tx, &members)?;
        delete_recalls(&tx, &members)?;
        let scrub: BTreeSet<Uuid> = uuids.iter().copied().collect();
        scrub_stored(&tx, bank_id, &scrub)?;
        aftermath.scrub = scrub;
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

    aftermath.models = drop_entries(tx, members)?;
    if reason == EraseReason::Forget {
        let mut spans: BTreeMap<i64, Vec<(usize, usize)>> = BTreeMap::new();
        for row in &rows {
            spans
                .entry(row.chunk_id)
                .or_default()
                .push((row.start, row.end));
        }
        let mut mentions = tx.prepare_cached(
            "SELECT spans FROM accesses WHERE memory_id = ?1 AND spans IS NOT NULL",
        )?;
        for member in members {
            for stored in mentions.query_map([member], |row| row.get::<_, String>(0))? {
                let stored: Vec<(i64, usize, usize)> =
                    serde_json::from_str(&stored?).unwrap_or_default();
                for (chunk, start, end) in stored {
                    spans.entry(chunk).or_default().push((start, end));
                }
            }
        }
        drop(mentions);
        redact(tx, &spans)?;
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
    }
    delete_orphans(tx, &entities)?;

    let details = match reason {
        EraseReason::Forget => serde_json::json!({ "memories": uuids }),
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

/// Deletes the model entries citing any of `members` and returns their
/// models: code drops an entry when a memory it cites goes (TIM-95,
/// decision 6), and the model refreshes.
fn drop_entries(
    conn: &Connection,
    members: &BTreeSet<i64>,
) -> Result<BTreeSet<i64>, rusqlite::Error> {
    let mut citing = conn.prepare_cached(
        "SELECT e.id, e.model_id FROM mental_model_citations c
         JOIN mental_model_entries e ON e.id = c.entry_id
         WHERE c.memory_id = ?1",
    )?;
    let mut entries = BTreeSet::new();
    let mut models = BTreeSet::new();
    for member in members {
        for row in citing.query_map([member], |row| Ok((row.get::<_, i64>(0)?, row.get(1)?)))? {
            let (entry, model) = row?;
            entries.insert(entry);
            models.insert(model);
        }
    }
    for entry in entries {
        conn.execute("DELETE FROM mental_model_entries WHERE id = ?1", [entry])?;
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

/// Rewrites a stored JSON set without the forgotten memories, or `None`
/// when nothing in it changes.
type Keep<'a> = &'a dyn Fn(&str) -> Option<String>;

/// Takes `memories` out of every stored in-context set of the bank: queued
/// and failed turns' sets and entries, built blocks, and session mappings
/// (schema versions 5 and 6). An entry citing any of them goes whole, since
/// its text restates the memory.
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
    let keep_entries = |json: &str| -> Option<String> {
        let entries: Vec<BlockEntry> = serde_json::from_str(json).ok()?;
        let kept: Vec<&BlockEntry> = entries
            .iter()
            .filter(|entry| !entry.cites.iter().any(|id| memories.contains(id)))
            .collect();
        (kept.len() != entries.len())
            .then(|| serde_json::to_string(&kept).expect("entries serialise"))
    };
    let tables: [(&str, &str, Keep<'_>); 5] = [
        (
            "SELECT t.id, t.memories FROM turn_in_context t
             JOIN sources s ON s.id = t.source_id WHERE s.bank_id = ?1",
            "UPDATE turn_in_context SET memories = ?2 WHERE id = ?1",
            &keep_ids,
        ),
        (
            "SELECT t.id, t.entries FROM turn_entries t
             JOIN sources s ON s.id = t.source_id WHERE s.bank_id = ?1",
            "UPDATE turn_entries SET entries = ?2 WHERE id = ?1",
            &keep_entries,
        ),
        (
            "SELECT id, in_context FROM prompt_blocks WHERE bank_id = ?1",
            "UPDATE prompt_blocks SET in_context = ?2 WHERE id = ?1",
            &keep_ids,
        ),
        (
            "SELECT id, entries FROM prompt_blocks WHERE bank_id = ?1",
            "UPDATE prompt_blocks SET entries = ?2 WHERE id = ?1",
            &keep_entries,
        ),
        (
            "SELECT rowid, cited FROM session_blocks WHERE bank_id = ?1",
            "UPDATE session_blocks SET cited = ?2 WHERE rowid = ?1",
            &keep_ids,
        ),
    ];
    for (select, update, keep) in tables {
        let rows: Vec<(i64, String)> = {
            let mut statement = conn.prepare_cached(select)?;
            statement
                .query_map([bank_id], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<Result<_, _>>()?
        };
        for (id, json) in rows {
            if let Some(kept) = keep(&json) {
                conn.execute(update, (id, kept))?;
            }
        }
    }
    Ok(())
}

/// Masks `spans` (chunk-relative characters, by chunk rowid) in each
/// chunk's text and in its source. A turn's chunk is the message, the
/// separator and the reply, so a span maps into `text` or `reply`; a
/// document's chunk starts at its `start_offset` in the source text. Text
/// already swept is left as it is. Call 1's saved reply goes too, since it
/// quotes the chunk.
fn redact(
    conn: &Connection,
    spans: &BTreeMap<i64, Vec<(usize, usize)>>,
) -> Result<(), rusqlite::Error> {
    for (&chunk_id, spans) in spans {
        let (source_id, start_offset, text): (i64, i64, Option<String>) = conn.query_row(
            "SELECT source_id, start_offset, text FROM chunks WHERE id = ?1",
            [chunk_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        conn.execute(
            "UPDATE chunks SET text = ?2, call1_output = NULL WHERE id = ?1",
            (chunk_id, text.map(|text| mask(&text, spans))),
        )?;

        let (kind, source_text, reply): (String, Option<String>, Option<String>) = conn.query_row(
            "SELECT kind, text, reply FROM sources WHERE id = ?1",
            [source_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        let (source_text, reply) = if kind == "turn" {
            let message = source_text
                .as_deref()
                .map_or(0, |text| text.chars().count());
            let reply_start = message + TURN_SEPARATOR.chars().count();
            let in_message: Vec<(usize, usize)> = spans
                .iter()
                .filter(|(start, _)| *start < message)
                .map(|&(start, end)| (start, end.min(message)))
                .collect();
            let in_reply: Vec<(usize, usize)> = spans
                .iter()
                .filter(|(_, end)| *end > reply_start)
                .map(|&(start, end)| (start.max(reply_start) - reply_start, end - reply_start))
                .collect();
            (
                source_text.map(|text| mask(&text, &in_message)),
                reply.map(|reply| mask(&reply, &in_reply)),
            )
        } else {
            let offset = usize::try_from(start_offset).unwrap_or(0);
            let shifted: Vec<(usize, usize)> = spans
                .iter()
                .map(|&(start, end)| (start + offset, end + offset))
                .collect();
            (source_text.map(|text| mask(&text, &shifted)), reply)
        };
        conn.execute(
            "UPDATE sources SET text = ?2, reply = ?3 WHERE id = ?1",
            (source_id, source_text, reply),
        )?;
    }
    Ok(())
}

/// `text` with every character inside `spans` replaced by
/// [`REDACTION_MASK`], except line breaks, which keep a document's lines.
fn mask(text: &str, spans: &[(usize, usize)]) -> String {
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
/// model's filter names (TIM-97, decision 4), and a speaker, whose platform
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
