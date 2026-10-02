//! Committing a checked reply: one transaction for everything the chunk
//! produced and its `extracted_at` (TIM-92).

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::{OptionalExtension, Transaction};
use uuid::Uuid;

use super::claims::{Checked, Link, NewMemory, Stamp, is_pronoun};
use super::input::{Unit, survivor};
use super::reconcile::{Edit, Fate, Neighbour, Plan, end_at};
use super::{
    Call1Input, EDIT_END_CLEARED, EDIT_END_REPOINTED, EDIT_ENDED, EDIT_KEPT, EDIT_REFINED,
    EDIT_RETRACTED, EDIT_SIGNIFICANCE_RAISED, EntityKind, Extracted,
};
use crate::constants::{
    Significance, Volatility, WEIGHT_CONFIRMED, WEIGHT_CREATED, WEIGHT_MENTIONED_AGAIN, WEIGHT_USED,
};
use crate::queue::{self, Lease};
use crate::store::bank::{add_alias, log_edit, log_memory_edit};
use crate::store::{Store, StoreError, VectorError, VectorIndex, micros};

/// A new memory's row, as the edits on its neighbours need it.
struct Written {
    id: i64,
    end: (Stamp, bool),
}

#[allow(clippy::too_many_arguments)]
pub(super) fn commit(
    store: &Store,
    lease: &Lease,
    input: &Call1Input,
    unit: &Unit,
    checked: &Checked,
    vectors: &[Vec<f32>],
    plan: &Plan,
    neighbours: &[Neighbour],
) -> Result<Extracted, StoreError> {
    let now = store.now();
    let mut conn = store.connection();
    let tx = conn.transaction()?;

    let (proposed, entities_created) = resolve_proposals(&tx, store, unit, checked, plan)?;

    let mut memories = Vec::with_capacity(checked.memories.len());
    let mut written: BTreeMap<usize, Written> = BTreeMap::new();
    for (index, ((memory, vector), fate)) in checked
        .memories
        .iter()
        .zip(vectors)
        .zip(&plan.fates)
        .enumerate()
    {
        let ended = match fate {
            Fate::Absorbed => continue,
            Fate::New => None,
            Fate::NewEnded {
                by,
                until,
                low_confidence,
            } => Some((*by, *until, *low_confidence)),
        };
        let uuid = store.new_id();
        let memory_id = insert_memory(&tx, store, unit, input, memory, uuid, ended)?;
        written.insert(
            index,
            Written {
                id: memory_id,
                // Where a neighbour this claim ends, or a memory whose ender
                // it replaces, ends (TIM-92).
                end: match memory.valid_from {
                    Some(stamp) => (stamp, memory.low_confidence),
                    None => end_at(None, input.observed_at, &unit.tz),
                },
            },
        );
        // The caller checked every vector's width, so only SQLite can fail
        // here.
        store
            .vectors()
            .upsert(&tx, unit.bank_id, memory_id, vector)
            .map_err(|error| match error {
                VectorError::Sqlite(error) => StoreError::Sqlite(error),
                other => {
                    StoreError::Sqlite(rusqlite::Error::ToSqlConversionFailure(Box::new(other)))
                }
            })?;
        link_entities(&tx, store, unit, memory_id, &memory.links, &proposed)?;
        // TIM-92: the created access carries the source's ingested_at, never
        // the time extraction ran.
        insert_access(&tx, unit, memory_id, "created")?;
        memories.push(uuid);
    }

    let by_id: BTreeMap<i64, &Neighbour> = neighbours.iter().map(|n| (n.id, n)).collect();
    for &(index, neighbour, edit) in &plan.edits {
        let by = &written[&index];
        match edit {
            Edit::Ends => end(&tx, store, unit, neighbour, by.id, by.end, EDIT_ENDED)?,
            Edit::Retracts | Edit::Denies => {
                tx.execute(
                    "UPDATE memories SET invalidated_at = ?2, superseded_by = ?3, updated_at = ?4
                     WHERE id = ?1",
                    (neighbour, micros(input.observed_at), by.id, micros(now)),
                )?;
                log_memory_edit(
                    &tx,
                    store,
                    unit.bank_id,
                    EDIT_RETRACTED,
                    neighbour,
                    &format!(
                        "{{\"superseded_by\":{},\"invalidated_at\":{},\"denied\":{}}}",
                        by.id,
                        micros(input.observed_at),
                        edit == Edit::Denies
                    ),
                )?;
                reopen(&tx, store, unit, neighbour, by, edit == Edit::Denies)?;
            }
            Edit::Refines => {
                tx.execute(
                    "UPDATE memories SET superseded_by = ?2, updated_at = ?3 WHERE id = ?1",
                    (neighbour, by.id, micros(now)),
                )?;
                log_memory_edit(
                    &tx,
                    store,
                    unit.bank_id,
                    EDIT_REFINED,
                    neighbour,
                    &format!("{{\"superseded_by\":{}}}", by.id),
                )?;
                reopen(&tx, store, unit, neighbour, by, false)?;
                // TIM-95 decision 6: a citation of a refined memory moves to
                // the head of its chain, where its accesses are inherited.
                tx.execute(
                    "UPDATE OR IGNORE mental_model_citations SET memory_id = ?2
                     WHERE memory_id = ?1",
                    (neighbour, by.id),
                )?;
                tx.execute(
                    "DELETE FROM mental_model_citations WHERE memory_id = ?1",
                    [neighbour],
                )?;
            }
        }
        // A model citing it refreshes (TIM-95, "refreshes follow
        // conversations"): the service reads this edit back and triggers
        // it, with the bank's debounce ([`crate::mental_models::effects`]).

        // A new version of a forgotten memory joins a chain waiting to be
        // erased (ADR 0010), so it's hidden from the moment it's committed.
        // An ending isn't a chain link, so what it creates stays.
        if edit != Edit::Ends {
            tx.execute(
                "UPDATE memories
                 SET hidden_at = (SELECT hidden_at FROM memories WHERE id = ?2)
                 WHERE id = ?1
                   AND (SELECT hidden_at FROM memories WHERE id = ?2) IS NOT NULL",
                (by.id, neighbour),
            )?;
        }
    }

    for (&neighbour, &label) in &plan.accesses {
        insert_access(&tx, unit, neighbour, label.as_str())?;
    }
    for (&neighbour, spans) in &plan.mention_spans {
        record_passages(&tx, unit, neighbour, spans)?;
    }
    for (&neighbour, &significance) in &plan.raises {
        let raised = tx.execute(
            "UPDATE memories SET significance = ?2, updated_at = ?3
             WHERE id = ?1 AND owner_significance IS NULL",
            (neighbour, level(significance), micros(now)),
        )?;
        if raised > 0 {
            log_memory_edit(
                &tx,
                store,
                unit.bank_id,
                EDIT_SIGNIFICANCE_RAISED,
                neighbour,
                &format!(
                    "{{\"from\":\"{}\",\"to\":\"{}\"}}",
                    by_id.get(&neighbour).map_or("", |n| level(n.significance)),
                    level(significance)
                ),
            )?;
        }
    }
    for &neighbour in &plan.keeps {
        let kept = tx.execute(
            "UPDATE memories SET owner_significance = 'kept', updated_at = ?2
             WHERE id = ?1 AND owner_significance IS NOT 'kept'",
            (neighbour, micros(now)),
        )?;
        if kept > 0 {
            log_memory_edit(&tx, store, unit.bank_id, EDIT_KEPT, neighbour, "{}")?;
        }
    }

    // At most one access per memory per turn and source, keeping the
    // strongest (TIM-90). `used` weighs least, so an access already in this
    // turn always stays.
    for (memory_id, _) in &checked.used {
        insert_access(&tx, unit, *memory_id, "used")?;
    }

    queue::finish(&tx, now, lease)?;
    tx.commit()?;

    Ok(Extracted {
        chunk: input.chunk,
        memories,
        used: checked.used.iter().map(|(_, uuid)| *uuid).collect(),
        entities_created,
        dropped: checked.dropped.clone(),
    })
}

/// The entity each proposed name resolves to, keyed by its lowercase name,
/// and the entities created for them. A name call 1 proposed twice is one
/// entity. At commit, code repeats the exact alias lookup and links to an
/// entity created after call 1's input was read. It never links to one that
/// existed before, whether call 1 saw it and chose not to use it or never
/// compared it at all, such as one the candidate cap left out (TIM-92).
fn resolve_proposals(
    tx: &Transaction<'_>,
    store: &Store,
    unit: &Unit,
    checked: &Checked,
    plan: &Plan,
) -> Result<(BTreeMap<String, i64>, Vec<Uuid>), rusqlite::Error> {
    let seen = unit.seen();
    let mut resolved = BTreeMap::new();
    let mut created = Vec::new();
    // A claim that became accesses or nothing links nothing, so it proposes
    // no entity either.
    for (memory, _) in checked
        .memories
        .iter()
        .zip(&plan.fates)
        .filter(|(_, fate)| **fate != Fate::Absorbed)
    {
        for link in &memory.links {
            let Link::Proposed { name, kind, .. } = link else {
                continue;
            };
            let key = name.to_lowercase();
            if resolved.contains_key(&key) {
                continue;
            }
            let entity_id = match created_meanwhile(tx, unit, name, &seen)? {
                Some(entity_id) => entity_id,
                None => {
                    let (entity_id, uuid) = create_entity(tx, store, unit.bank_id, name, *kind)?;
                    created.push(uuid);
                    entity_id
                }
            };
            resolved.insert(key, entity_id);
        }
    }
    Ok((resolved, created))
}

/// The first entity, as its survivor, with `name` as an alias that was
/// created after the input's entity boundary and that call 1 wasn't shown.
fn created_meanwhile(
    tx: &Transaction<'_>,
    unit: &Unit,
    name: &str,
    seen: &BTreeSet<i64>,
) -> Result<Option<i64>, rusqlite::Error> {
    let mut statement = tx.prepare_cached(
        "SELECT entity_id FROM entity_aliases
         WHERE bank_id = ?1 AND alias = ?2 COLLATE NOCASE
         ORDER BY id",
    )?;
    let hits: Vec<i64> = statement
        .query_map((unit.bank_id, name), |row| row.get(0))?
        .collect::<Result<_, _>>()?;
    for entity_id in hits {
        let entity_id = survivor(tx, entity_id)?;
        if entity_id > unit.entity_boundary && !seen.contains(&entity_id) {
            return Ok(Some(entity_id));
        }
    }
    Ok(None)
}

fn create_entity(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    name: &str,
    kind: EntityKind,
) -> Result<(i64, Uuid), rusqlite::Error> {
    let uuid = store.new_id();
    let now = micros(store.now());
    tx.execute(
        "INSERT INTO entities (uuid, bank_id, name, kind, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
        (uuid.to_string(), bank_id, name, kind.as_str(), now),
    )?;
    let entity_id = tx.last_insert_rowid();
    log_edit(tx, store, bank_id, "entity_created", Some(entity_id), "{}")?;
    add_alias(tx, store, bank_id, entity_id, name)?;
    Ok((entity_id, uuid))
}

/// Inserts a new memory. `ended` is the newer neighbour that already ends
/// it, with where it ends and whether that's a guess: an older claim
/// arriving after a newer one (TIM-92).
fn insert_memory(
    tx: &Transaction<'_>,
    store: &Store,
    unit: &Unit,
    input: &Call1Input,
    memory: &NewMemory,
    uuid: Uuid,
    ended: Option<(i64, Stamp, bool)>,
) -> Result<i64, rusqlite::Error> {
    let now = micros(store.now());
    let at = |stamp: Option<Stamp>| stamp.map(|stamp| micros(stamp.at));
    let precision = |stamp: Option<Stamp>| stamp.map(|stamp| stamp.precision.as_str());
    let (valid_until, ended_by, low) = match ended {
        Some((by, until, low)) => (Some(until), Some(by), memory.low_confidence || low),
        None => (memory.valid_until, None, memory.low_confidence),
    };
    tx.execute(
        "INSERT INTO memories (uuid, bank_id, content, kind, significance, owner_significance,
                               chunk_id, source_start, source_end, observed_at, valid_from,
                               valid_from_precision, valid_until, valid_until_precision,
                               until_event, window_confidence, due_at, due_at_precision,
                               volatility, recurrence_text, recurrence_rrule, recurrence_start,
                               recurrence_start_precision, ended_by, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18,
                 ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?25)",
        rusqlite::params![
            uuid.to_string(),
            unit.bank_id,
            memory.content,
            memory.kind.as_str(),
            level(memory.significance),
            memory.kept.then_some("kept"),
            unit.chunk_id,
            memory.start as i64,
            memory.end as i64,
            micros(input.observed_at),
            at(memory.valid_from),
            precision(memory.valid_from),
            at(valid_until),
            precision(valid_until),
            memory.until_event,
            if low { "low" } else { "high" },
            at(memory.due_at),
            precision(memory.due_at),
            memory.volatility.map(volatility),
            memory.recurrence_text,
            memory.recurrence_rrule,
            at(memory.recurrence_start),
            precision(memory.recurrence_start),
            ended_by,
            now,
        ],
    )?;
    Ok(tx.last_insert_rowid())
}

/// Links a memory to its entities, keeping each link's surface form. A new
/// surface form becomes an alias in a logged edit, so a mislink can be undone
/// (TIM-92); a pronoun never does. A known entity merged into another since
/// call 1 read its input is linked as the entity that survived (ADR 0010).
fn link_entities(
    tx: &Transaction<'_>,
    store: &Store,
    unit: &Unit,
    memory_id: i64,
    links: &[Link],
    proposed: &BTreeMap<String, i64>,
) -> Result<(), rusqlite::Error> {
    for link in links {
        let (entity_id, surface_form) = match link {
            Link::Known {
                entity,
                surface_form,
            } => (survivor(tx, *entity)?, surface_form),
            Link::Proposed {
                name, surface_form, ..
            } => (proposed[&name.to_lowercase()], surface_form),
        };
        tx.execute(
            "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id, surface_form)
             VALUES (?1, ?2, ?3)",
            (memory_id, entity_id, surface_form),
        )?;
        if let Some(surface_form) = surface_form
            && !is_pronoun(surface_form)
            && !has_alias(tx, entity_id, surface_form)?
        {
            add_alias(tx, store, unit.bank_id, entity_id, surface_form)?;
        }
    }
    Ok(())
}

fn has_alias(tx: &Transaction<'_>, entity_id: i64, alias: &str) -> Result<bool, rusqlite::Error> {
    Ok(tx
        .query_row(
            "SELECT 1 FROM entity_aliases WHERE entity_id = ?1 AND alias = ?2",
            (entity_id, alias),
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

/// An access at the source's ingested_at and turn number, keeping the
/// strongest kind: an access already there stays unless this one weighs
/// more. A memory has at most one access per turn (TIM-90), and one per
/// document: a document carries the number of the turn before it, and its
/// mention is a separate, independent one (CONTEXT.md, "Mentioned again").
fn insert_access(
    tx: &Transaction<'_>,
    unit: &Unit,
    memory_id: i64,
    kind: &str,
) -> Result<(), rusqlite::Error> {
    // TIM-97 decision 4: a memory erased since the chunk's input was read
    // (purged, or forgotten from a live in-context set) takes no access.
    if !exists(tx, memory_id)? {
        return Ok(());
    }
    // ?3 is the document's source, or NULL for a turn.
    let existing: Option<(i64, String)> = tx
        .query_row(
            "SELECT a.id, a.kind FROM accesses a LEFT JOIN sources s ON s.id = a.source_id
             WHERE a.memory_id = ?1 AND a.turn = ?2
               AND (CASE WHEN ?3 IS NULL THEN s.kind IS NOT 'document'
                         ELSE a.source_id = ?3 END)",
            (
                memory_id,
                unit.turn,
                unit.document_id.is_some().then_some(unit.source_id),
            ),
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    match existing {
        None => {
            tx.execute(
                "INSERT INTO accesses (bank_id, memory_id, kind, at, turn, source_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                (
                    unit.bank_id,
                    memory_id,
                    kind,
                    micros(unit.ingested_at),
                    unit.turn,
                    unit.source_id,
                ),
            )?;
        }
        Some((id, old)) if weight(kind) > weight(&old) => {
            tx.execute(
                "UPDATE accesses SET kind = ?2, at = ?3 WHERE id = ?1",
                (id, kind, micros(unit.ingested_at)),
            )?;
        }
        Some(_) => {}
    }
    Ok(())
}

/// Records where the chunk restated `memory_id` without making a memory of
/// it, so a forget can redact those passages (schema version 8). Kept
/// apart from strength: a later version of the same document repeating
/// itself is no access, but its passage still goes when the memory does.
fn record_passages(
    tx: &Transaction<'_>,
    unit: &Unit,
    memory_id: i64,
    spans: &[(usize, usize)],
) -> Result<(), rusqlite::Error> {
    if !exists(tx, memory_id)? {
        return Ok(());
    }
    let mut statement = tx.prepare_cached(
        "INSERT OR IGNORE INTO mention_passages (memory_id, chunk_id, start_offset, end_offset)
         VALUES (?1, ?2, ?3, ?4)",
    )?;
    for &(start, end) in spans {
        statement.execute((memory_id, unit.chunk_id, start as i64, end as i64))?;
    }
    Ok(())
}

fn exists(tx: &Transaction<'_>, memory_id: i64) -> Result<bool, rusqlite::Error> {
    Ok(tx
        .query_row("SELECT 1 FROM memories WHERE id = ?1", [memory_id], |_| {
            Ok(())
        })
        .optional()?
        .is_some())
}

fn weight(kind: &str) -> f64 {
    match kind {
        "confirmed" => WEIGHT_CONFIRMED,
        "mentioned_again" => WEIGHT_MENTIONED_AGAIN,
        "created" => WEIGHT_CREATED,
        _ => WEIGHT_USED,
    }
}

/// Ends `neighbour` by `by`: `valid_until` where `by` starts, or the day it
/// was said with low confidence (TIM-92). A guessed end lowers the ended
/// memory's window confidence. The edit is logged as `kind`.
fn end(
    tx: &Transaction<'_>,
    store: &Store,
    unit: &Unit,
    neighbour: i64,
    by: i64,
    (until, low): (Stamp, bool),
    kind: &str,
) -> Result<(), rusqlite::Error> {
    tx.execute(
        "UPDATE memories SET valid_until = ?2, valid_until_precision = ?3, ended_by = ?4,
                window_confidence = CASE WHEN ?5 THEN 'low' ELSE window_confidence END,
                updated_at = ?6
         WHERE id = ?1",
        (
            neighbour,
            micros(until.at),
            until.precision.as_str(),
            by,
            low,
            micros(store.now()),
        ),
    )?;
    log_memory_edit(
        tx,
        store,
        unit.bank_id,
        kind,
        neighbour,
        &format!(
            "{{\"ended_by\":{by},\"valid_until\":{},\"precision\":\"{}\"}}",
            micros(until.at),
            until.precision.as_str()
        ),
    )
}

/// The memories `superseded` had ended, now that a claim supersedes it
/// (TIM-92, "Reopening", as amended by TIM-108). After a `retracts` or
/// `refines` their end follows the successor, which still ended them. After
/// a `denies` (`denied`) the ending never happened, so they're open again.
/// The label decides, never the kinds. Either way the edit is logged.
fn reopen(
    tx: &Transaction<'_>,
    store: &Store,
    unit: &Unit,
    superseded: i64,
    successor: &Written,
    denied: bool,
) -> Result<(), rusqlite::Error> {
    let mut statement = tx.prepare_cached("SELECT id FROM memories WHERE ended_by = ?1")?;
    let ended: Vec<i64> = statement
        .query_map([superseded], |row| row.get(0))?
        .collect::<Result<_, _>>()?;
    for memory in ended {
        if denied {
            tx.execute(
                "UPDATE memories SET valid_until = NULL, valid_until_precision = NULL,
                        ended_by = NULL, updated_at = ?2
                 WHERE id = ?1",
                (memory, micros(store.now())),
            )?;
            log_memory_edit(
                tx,
                store,
                unit.bank_id,
                EDIT_END_CLEARED,
                memory,
                &format!(
                    "{{\"ended_by\":{superseded},\"denied_by\":{}}}",
                    successor.id
                ),
            )?;
        } else {
            end(
                tx,
                store,
                unit,
                memory,
                successor.id,
                successor.end,
                EDIT_END_REPOINTED,
            )?;
        }
    }
    Ok(())
}

fn level(significance: Significance) -> &'static str {
    match significance {
        Significance::Trivial => "trivial",
        Significance::Minor => "minor",
        Significance::Notable => "notable",
        Significance::Major => "major",
        Significance::Critical => "critical",
    }
}

fn volatility(volatility: Volatility) -> &'static str {
    match volatility {
        Volatility::Hours => "hours",
        Volatility::Days => "days",
        Volatility::Weeks => "weeks",
        Volatility::Months => "months",
        Volatility::Years => "years",
    }
}
