//! Committing a checked reply: one transaction for everything the chunk
//! produced and its `extracted_at`.

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::{OptionalExtension, Transaction};
use uuid::Uuid;

use super::claims::{Checked, Kind as ClaimKind, Link, NewMemory, Precision, Stamp, is_pronoun};
use super::input::{Unit, survivor};
use super::reconcile::{Edit, Fate, Neighbour, Plan, Restated, end_at};
use super::{
    CALL2_VERSION, Call1Input, EDIT_END_CLEARED, EDIT_END_REPOINTED, EDIT_ENDED, EDIT_KEPT,
    EDIT_REFINED, EDIT_RETRACTED, EDIT_SIGNIFICANCE_RAISED, EntityKind, Extracted, KindMismatch,
};
use crate::constants::{
    Significance, Volatility, WEIGHT_CONFIRMED, WEIGHT_CREATED, WEIGHT_MENTIONED_AGAIN, WEIGHT_USED,
};
use crate::queue::{self, Lease};
use crate::store::bank::{add_alias, log_edit, log_memory_edit};
use crate::store::{Store, StoreError, VectorError, VectorIndex, micros, timestamp};
use crate::strength::Kind;

/// A new memory's row, as the edits on its neighbours need it.
struct Written {
    id: i64,
    uuid: Uuid,
    end: (Stamp, bool),
    said_at: Stamp,
}

impl Written {
    /// Task endings use when the claim was said, not its event time.
    fn end_for(&self, task: bool) -> (Stamp, bool) {
        if task {
            (self.said_at, false)
        } else {
            self.end
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn commit(
    tx: &Transaction<'_>,
    store: &Store,
    lease: &Lease,
    input: &Call1Input,
    unit: &Unit,
    checked: &Checked,
    vectors: &[Vec<f32>],
    plan: &Plan,
    neighbours: &[Neighbour],
    corroborate_used: bool,
    call2_model: Option<&str>,
) -> Result<Extracted, StoreError> {
    let now = store.now();
    let (proposed, entities_created) = resolve_proposals(tx, store, lease, unit, checked, plan)?;

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
        // Keyed by the source and the claim's ordinal,
        // written with the chunk's position so a document's chunks can't
        // collide.
        let uuid = store.derived_id(
            lease.source,
            &format!("{}:{}", lease.position, memory.claim),
        );
        // A correction of the same kind with no window is silent about dates,
        // not a request to erase them, and so is a repeat promoted for its
        // significance. Keep the first such neighbour's whole window; never
        // combine windows or inherit another kind's fields. A refinement
        // call 2 labelled itself keeps its own window.
        let mut memory = memory.clone();
        if memory.due_at.is_none()
            && memory.supplied_valid_from().is_none()
            && memory.valid_until.is_none()
            && memory.until_event.is_none()
            && let Some(neighbour) = plan.edits.iter().find_map(|&(claim, id, edit)| {
                let promoted = || plan.promoted.get(&index).is_some_and(|ns| ns.contains(&id));
                (claim == index && (edit == Edit::Retracts || edit == Edit::Refines && promoted()))
                    .then(|| neighbours.iter().find(|neighbour| neighbour.id == id))
                    .flatten()
            })
            && matches!(
                (memory.kind, neighbour.kind),
                (ClaimKind::Fact, Kind::Fact)
                    | (ClaimKind::Event, Kind::Event)
                    | (ClaimKind::State, Kind::State)
                    | (ClaimKind::Task, Kind::Task)
                    | (ClaimKind::Recurring, Kind::Recurring)
            )
        {
            memory.due_at = neighbour.due_at;
            memory.valid_from = neighbour.valid_from;
            memory.valid_from_defaulted = false;
            memory.valid_until = neighbour.valid_until;
            memory.until_event = neighbour.until_event.clone();
            memory.low_confidence = neighbour.low_confidence;
        }
        let memory_id = insert_memory(tx, store, unit, input, &memory, uuid, ended)?;
        written.insert(
            index,
            Written {
                id: memory_id,
                uuid,
                // Where a neighbour this claim ends, or a memory whose ender
                // it replaces, ends.
                end: match memory.valid_from {
                    Some(stamp) => (stamp, memory.low_confidence),
                    None => end_at(None, input.observed_at, &unit.tz),
                },
                said_at: Stamp {
                    at: input.observed_at,
                    precision: Precision::Minute,
                },
            },
        );
        // The caller checked every vector's width, so only SQLite can fail
        // here.
        store
            .vectors()
            .upsert(tx, unit.bank_id, memory_id, vector)
            .map_err(|error| match error {
                VectorError::Sqlite(error) => StoreError::Sqlite(error),
                other => {
                    StoreError::Sqlite(rusqlite::Error::ToSqlConversionFailure(Box::new(other)))
                }
            })?;
        link_entities(tx, store, unit, memory_id, &memory.links, &proposed)?;
        // The created access carries the source's access time, never
        // the time extraction ran.
        insert_access(tx, unit, memory_id, "created")?;
        memories.push(uuid);
    }

    let by_id: BTreeMap<i64, &Neighbour> = neighbours.iter().map(|n| (n.id, n)).collect();
    for &(index, neighbour, edit) in &plan.edits {
        let by = &written[&index];
        match edit {
            Edit::Ends => end(
                tx,
                store,
                unit,
                neighbour,
                by.id,
                by.end_for(by_id[&neighbour].kind == crate::strength::Kind::Task),
                EDIT_ENDED,
            )?,
            Edit::Retracts | Edit::Denies => {
                tx.execute(
                    "UPDATE memories SET invalidated_at = ?2, superseded_by = ?3, updated_at = ?4
                     WHERE id = ?1",
                    (neighbour, micros(input.observed_at), by.id, micros(now)),
                )?;
                log_memory_edit(
                    tx,
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
                reopen(tx, store, unit, neighbour, by, edit)?;
            }
            Edit::Refines => {
                tx.execute(
                    "UPDATE memories SET superseded_by = ?2, updated_at = ?3 WHERE id = ?1",
                    (neighbour, by.id, micros(now)),
                )?;
                log_memory_edit(
                    tx,
                    store,
                    unit.bank_id,
                    EDIT_REFINED,
                    neighbour,
                    &format!("{{\"superseded_by\":{}}}", by.id),
                )?;
                reopen(tx, store, unit, neighbour, by, edit)?;
                // A citation of a refined memory moves to
                // the head of its chain, where its accesses are inherited.
                tx.execute(
                    "UPDATE OR IGNORE mental_model_cites SET memory_id = ?2
                     WHERE memory_id = ?1",
                    (neighbour, by.id),
                )?;
                tx.execute(
                    "DELETE FROM mental_model_cites WHERE memory_id = ?1",
                    [neighbour],
                )?;
            }
        }
        // A model citing it refreshes: the service reads this edit back and
        // triggers
        // it, with the bank's debounce ([`crate::mental_models::effects`]).

        // A new version of a forgotten memory joins a chain waiting to be
        // erased, so it's hidden from the moment it's committed.
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
        insert_access(tx, unit, neighbour, label.as_str())?;
    }
    for (&neighbour, spans) in &plan.mention_spans {
        record_passages(tx, unit, neighbour, spans)?;
    }
    let mut restatements = 0;
    for restated in &plan.restatements {
        let memory = &checked.memories[restated.claim];
        if record_restatement(tx, unit, input, memory, restated, call2_model)? {
            restatements += 1;
        }
    }
    for (&neighbour, &significance) in &plan.raises {
        let raised = tx.execute(
            "UPDATE memories SET significance = ?2, updated_at = ?3
             WHERE id = ?1 AND owner_significance IS NULL",
            (neighbour, level(significance), micros(now)),
        )?;
        if raised > 0 {
            log_memory_edit(
                tx,
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
            log_memory_edit(tx, store, unit.bank_id, EDIT_KEPT, neighbour, "{}")?;
        }
    }

    // At most one access per memory per turn and source, keeping the
    // strongest. `used` weighs least, so an access already in this
    // turn always stays. With `strength.corroborate_used` on, a chain's
    // first credited turn is held as pending and writes nothing.
    for (memory_id, _) in &checked.used {
        if corroborate_used && !credited_before(tx, unit, *memory_id)? {
            hold_credit(tx, unit, *memory_id)?;
        } else {
            insert_access(tx, unit, *memory_id, "used")?;
        }
    }

    queue::finish(tx, now, lease)?;

    Ok(Extracted {
        chunk: input.chunk,
        memories,
        used: checked.used.iter().map(|(_, uuid)| *uuid).collect(),
        entities_created,
        dropped: checked.dropped.clone(),
        promoted: plan
            .promoted
            .keys()
            .map(|index| written[index].uuid)
            .collect(),
        kind_mismatches: plan
            .mismatches
            .iter()
            .map(|mismatch| {
                let claim = &checked.memories[mismatch.claim];
                let neighbour = by_id[&mismatch.neighbour];
                KindMismatch {
                    claim: claim.claim,
                    memory: written.get(&mismatch.claim).map(|written| written.uuid),
                    neighbour: neighbour.uuid,
                    claim_kind: claim.kind.stored(),
                    neighbour_kind: neighbour.kind,
                    cause: mismatch.cause,
                    older: mismatch.older,
                    rejected: mismatch.rejected,
                }
            })
            .collect(),
        restatements,
    })
}

/// The entity each proposed name resolves to, keyed by its lowercase name,
/// and the entities created for them. A name call 1 proposed twice is one
/// entity. At commit, code repeats the exact alias lookup and links to an
/// entity created after call 1's input was read. It never links to one that
/// existed before, whether call 1 saw it and chose not to use it or never
/// compared it at all, such as one the candidate cap left out.
fn resolve_proposals(
    tx: &Transaction<'_>,
    store: &Store,
    lease: &Lease,
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
                    let (entity_id, uuid) =
                        create_entity(tx, store, unit.bank_id, lease, &key, name, *kind)?;
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

/// Creates an entity keyed by its creating source, chunk position and the
/// exact dedup key from `resolve_proposals`.
/// The proposed name is already NFC and trimmed; the key lowercases it.
/// No second normalization or lookup of existing ids affects replay ids.
fn create_entity(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    lease: &Lease,
    key: &str,
    name: &str,
    kind: EntityKind,
) -> Result<(i64, Uuid), rusqlite::Error> {
    let uuid = store.derived_id(lease.source, &format!("entity:{}:{key}", lease.position));
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
/// arriving after a newer one.
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
/// surface form becomes an alias in a logged edit, so a mislink can be
/// undone; a pronoun never does. A known entity merged into another since
/// call 1 read its input is linked as the entity that survived.
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

/// An access at the source's access time and turn number, keeping the
/// strongest kind: an access already there stays unless this one weighs
/// more. A memory has at most one access per turn, and one per
/// document: a document carries the number of the turn before it, and its
/// mention is a separate, independent one (CONTEXT.md, "Mentioned again").
fn insert_access(
    tx: &Transaction<'_>,
    unit: &Unit,
    memory_id: i64,
    kind: &str,
) -> Result<(), rusqlite::Error> {
    // A memory erased since the chunk's input was read
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
                    micros(unit.access_at),
                    unit.turn,
                    unit.source_id,
                ),
            )?;
        }
        Some((id, old)) if weight(kind) > weight(&old) => {
            tx.execute(
                "UPDATE accesses SET kind = ?2, at = ?3 WHERE id = ?1",
                (id, kind, micros(unit.access_at)),
            )?;
        }
        Some(_) => {}
    }
    Ok(())
}

/// Whether `memory_id`'s chain was credited `used` in another turn: a
/// pending credit, or a `used` access, on the memory or on anything it
/// inherits from along `superseded_by`.
fn credited_before(
    tx: &Transaction<'_>,
    unit: &Unit,
    memory_id: i64,
) -> Result<bool, rusqlite::Error> {
    tx.query_row(
        "WITH RECURSIVE chain(id) AS (
           SELECT ?1
           UNION
           SELECT m.id FROM memories m JOIN chain c ON m.superseded_by = c.id
         )
         SELECT EXISTS (SELECT 1 FROM pending_credits
                        WHERE memory_id IN (SELECT id FROM chain) AND turn <> ?2)
             OR EXISTS (SELECT 1 FROM accesses
                        WHERE memory_id IN (SELECT id FROM chain)
                          AND kind = 'used' AND turn <> ?2)",
        (memory_id, unit.turn),
        |row| row.get(0),
    )
}

/// Holds a first `used` credit for `memory_id` at the unit's turn. It
/// writes no access, so strength never counts it.
fn hold_credit(tx: &Transaction<'_>, unit: &Unit, memory_id: i64) -> Result<(), rusqlite::Error> {
    if !exists(tx, memory_id)? {
        return Ok(());
    }
    tx.execute(
        "INSERT OR IGNORE INTO pending_credits (bank_id, memory_id, turn, at)
         VALUES (?1, ?2, ?3, ?4)",
        (unit.bank_id, memory_id, unit.turn, micros(unit.access_at)),
    )?;
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

/// Keeps what an absorbed claim said on the memory it restated (schema
/// version 19): call 1's checked claim, call 2's label and the label after
/// the guards, and the passage it was taken from, so a forget masking that
/// passage can delete it. Returns whether a row was written: a memory
/// erased since the chunk's input was read takes none, as it takes no
/// access.
fn record_restatement(
    tx: &Transaction<'_>,
    unit: &Unit,
    input: &Call1Input,
    memory: &NewMemory,
    restated: &Restated,
    call2_model: Option<&str>,
) -> Result<bool, rusqlite::Error> {
    if !exists(tx, restated.neighbour)? {
        return Ok(false);
    }
    let claim = claim_json(tx, memory)?;
    tx.execute(
        "INSERT INTO restatements (memory_id, chunk_id, start_offset, end_offset, claim, label,
                                   outcome, observed_at, call2_version, model)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        rusqlite::params![
            restated.neighbour,
            unit.chunk_id,
            memory.start as i64,
            memory.end as i64,
            claim.to_string(),
            restated.label.as_str(),
            restated.outcome.as_str(),
            micros(input.observed_at),
            CALL2_VERSION,
            call2_model.unwrap_or_default(),
        ],
    )?;
    Ok(true)
}

/// A checked claim as JSON, as [`insert_memory`] would have written it,
/// with its resolved window: an undated event's start is the day it was
/// said, flagged `valid_from_defaulted`. A known entity is named by its id,
/// a proposed one by its name and kind.
fn claim_json(
    tx: &Transaction<'_>,
    memory: &NewMemory,
) -> Result<serde_json::Value, rusqlite::Error> {
    let stamp = |stamp: Option<Stamp>| {
        stamp.map(|stamp| {
            serde_json::json!({
                "at": stamp.at.to_string(),
                "precision": stamp.precision.as_str(),
            })
        })
    };
    let mut links = Vec::with_capacity(memory.links.len());
    for link in &memory.links {
        links.push(match link {
            Link::Known {
                entity,
                surface_form,
            } => {
                let uuid: Option<String> = tx
                    .query_row("SELECT uuid FROM entities WHERE id = ?1", [entity], |row| {
                        row.get(0)
                    })
                    .optional()?;
                serde_json::json!({"entity": uuid, "surface_form": surface_form})
            }
            Link::Proposed {
                name,
                kind,
                surface_form,
            } => serde_json::json!({
                "new_name": name,
                "new_kind": kind.as_str(),
                "surface_form": surface_form,
            }),
        });
    }
    Ok(serde_json::json!({
        "sentence": memory.content,
        "kind": memory.kind.as_str(),
        "significance": level(memory.significance),
        "valid_from": stamp(memory.valid_from),
        "valid_from_defaulted": memory.valid_from_defaulted,
        "valid_until": stamp(memory.valid_until),
        "due_at": stamp(memory.due_at),
        "until_event": memory.until_event,
        "window_confidence": if memory.low_confidence { "low" } else { "high" },
        "volatility": memory.volatility.map(volatility),
        "recurrence_text": memory.recurrence_text,
        "recurrence_rrule": memory.recurrence_rrule,
        "recurrence_start": stamp(memory.recurrence_start),
        "entities": links,
    }))
}

fn exists(tx: &Transaction<'_>, memory_id: i64) -> Result<bool, rusqlite::Error> {
    Ok(tx
        .query_row("SELECT 1 FROM memories WHERE id = ?1", [memory_id], |_| {
            Ok(())
        })
        .optional()?
        .is_some())
}

/// Orders kinds within one turn, so the strongest access stays. It reads
/// the fixed weights, never `strength.access_weights`: tuning `used` changes
/// how much a `used` access counts, not which access a turn keeps.
fn weight(kind: &str) -> f64 {
    match kind {
        "confirmed" => WEIGHT_CONFIRMED,
        "mentioned_again" => WEIGHT_MENTIONED_AGAIN,
        "created" => WEIGHT_CREATED,
        _ => WEIGHT_USED,
    }
}

/// Ends `neighbour` by `by` at the caller-selected time. A guessed end lowers the ended
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

/// The memories `superseded` had ended, now that a claim supersedes it. After a `retracts` or
/// `refines` their end follows the successor, which still ended them, except
/// task refinements preserve the original closing boundary and task corrections
/// use the successor observation time. States keep event-time endings. After
/// a `denies` (`denied`) the ending never happened, so they're open again.
/// The label decides whether to reopen. Either way the edit is logged.
fn reopen(
    tx: &Transaction<'_>,
    store: &Store,
    unit: &Unit,
    superseded: i64,
    successor: &Written,
    edit: Edit,
) -> Result<(), rusqlite::Error> {
    let mut statement = tx.prepare_cached(
        "SELECT id, kind = 'task', valid_until, valid_until_precision
         FROM memories WHERE ended_by = ?1",
    )?;
    let ended: Vec<(i64, bool, i64, String)> = statement
        .query_map([superseded], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?
        .collect::<Result<_, _>>()?;
    for (memory, task, until, precision) in ended {
        if edit == Edit::Denies {
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
            let end_at = if task && edit == Edit::Refines {
                (
                    Stamp {
                        at: timestamp(until),
                        precision: Precision::parse(&precision).expect("a stored precision parses"),
                    },
                    false,
                )
            } else {
                successor.end_for(task)
            };
            end(
                tx,
                store,
                unit,
                memory,
                successor.id,
                end_at,
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
