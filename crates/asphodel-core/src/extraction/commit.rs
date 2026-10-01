//! Committing a checked reply: one transaction for everything the chunk
//! produced and its `extracted_at` (TIM-92).

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::{OptionalExtension, Transaction};
use uuid::Uuid;

use super::claims::{Checked, Link, NewMemory, Stamp, is_pronoun};
use super::input::{Unit, survivor};
use super::{Call1Input, EntityKind, Extracted};
use crate::constants::{Significance, Volatility};
use crate::queue::{self, Lease};
use crate::store::bank::{add_alias, log_edit};
use crate::store::{Store, StoreError, VectorError, VectorIndex, micros};

pub(super) fn commit(
    store: &Store,
    lease: &Lease,
    input: &Call1Input,
    unit: &Unit,
    checked: &Checked,
    vectors: &[Vec<f32>],
) -> Result<Extracted, StoreError> {
    let now = store.now();
    let mut conn = store.connection();
    let tx = conn.transaction()?;

    let (proposed, entities_created) = resolve_proposals(&tx, store, unit, checked)?;

    let mut memories = Vec::with_capacity(checked.memories.len());
    for (memory, vector) in checked.memories.iter().zip(vectors) {
        let uuid = store.new_id();
        let memory_id = insert_memory(&tx, store, unit, input, memory, uuid)?;
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

    // At most one access per memory per turn, keeping the strongest
    // (TIM-90). `used` weighs least, so an access already in this turn
    // always stays.
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
) -> Result<(BTreeMap<String, i64>, Vec<Uuid>), rusqlite::Error> {
    let seen = unit.seen();
    let mut resolved = BTreeMap::new();
    let mut created = Vec::new();
    for memory in &checked.memories {
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

fn insert_memory(
    tx: &Transaction<'_>,
    store: &Store,
    unit: &Unit,
    input: &Call1Input,
    memory: &NewMemory,
    uuid: Uuid,
) -> Result<i64, rusqlite::Error> {
    let now = micros(store.now());
    let at = |stamp: Option<Stamp>| stamp.map(|stamp| micros(stamp.at));
    let precision = |stamp: Option<Stamp>| stamp.map(|stamp| stamp.precision.as_str());
    tx.execute(
        "INSERT INTO memories (uuid, bank_id, content, kind, significance, owner_significance,
                               chunk_id, source_start, source_end, observed_at, valid_from,
                               valid_from_precision, valid_until, valid_until_precision,
                               until_event, window_confidence, due_at, due_at_precision,
                               volatility, recurrence_text, recurrence_rrule, recurrence_start,
                               recurrence_start_precision, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18,
                 ?19, ?20, ?21, ?22, ?23, ?24, ?24)",
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
            at(memory.valid_until),
            precision(memory.valid_until),
            memory.until_event,
            if memory.low_confidence { "low" } else { "high" },
            at(memory.due_at),
            precision(memory.due_at),
            memory.volatility.map(volatility),
            memory.recurrence_text,
            memory.recurrence_rrule,
            at(memory.recurrence_start),
            precision(memory.recurrence_start),
            now,
        ],
    )?;
    Ok(tx.last_insert_rowid())
}

/// Links a memory to its entities, keeping each link's surface form. A new
/// surface form becomes an alias in a logged edit, so a mislink can be undone
/// (TIM-92); a pronoun never does.
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
            } => (*entity, surface_form),
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

/// An access at the source's ingested_at and turn number. An access the
/// memory already has in that turn stays (`UNIQUE (memory_id, turn)`).
fn insert_access(
    tx: &Transaction<'_>,
    unit: &Unit,
    memory_id: i64,
    kind: &str,
) -> Result<(), rusqlite::Error> {
    tx.execute(
        "INSERT OR IGNORE INTO accesses (bank_id, memory_id, kind, at, turn, source_id)
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
