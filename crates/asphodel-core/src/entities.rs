//! Entity correction ("Operations: backup, inspection, correction and
//! forgetting", TIM-99, decision 5; ADR 0010, "Inspection and correction").
//!
//! **Merges keep the entity row.** `entity merge <from> <into>` sets
//! `from.merged_into` and moves `from`'s aliases and links to `into` in one
//! logged edit. An alias or link `into` already has stays on `from`, where
//! it resolves through `merged_into` like any other, so nothing is deleted
//! and an unmerge has nothing to recreate. Mental model filters on `from`
//! are repointed and the models refreshed. A chunk whose call 1 ran before
//! the merge links through `merged_into` at commit. `user` and `assistant`
//! can only ever be `into`.
//!
//! **Unmerge** takes the merge's edit id and moves back exactly what it
//! moved, provided `into` hasn't been merged again since. What was linked
//! to `into` after the merge stays there.
//!
//! **Aliases and links.** `entity alias rm --relink-to` removes a wrong
//! alias and moves the links whose surface form it was to the right entity,
//! which gets the alias. `entity link` and `entity unlink` edit one link.
//! None of these touch a memory's sentence, kind or window.
//!
//! Every edit row holds rowids and counts, never names or aliases, and no
//! error carries an entity's name (ADR 0010, "Logging").

use std::collections::BTreeSet;

use rusqlite::{Connection, OptionalExtension, Transaction};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::extraction::survivor;
use crate::ingest::find_bank;
use crate::store::bank::{add_alias, log_edit};
use crate::store::{Store, StoreError, micros, nfc};

/// The edit kind a merge writes, on `from`.
pub const EDIT_ENTITY_MERGED: &str = "entity_merged";

/// The edit kind an unmerge writes, on `from`.
pub const EDIT_ENTITY_UNMERGED: &str = "entity_unmerged";

/// The edit kind `entity alias rm` writes.
pub const EDIT_ALIAS_REMOVED: &str = "alias_removed";

/// The edit kind `entity link` writes, on the memory and the entity.
pub const EDIT_ENTITY_LINKED: &str = "entity_linked";

/// The edit kind `entity unlink` writes, on the memory and the entity.
pub const EDIT_ENTITY_UNLINKED: &str = "entity_unlinked";

/// The edit kind `--relink-to` writes for the links it moves.
pub const EDIT_LINKS_MOVED: &str = "links_moved";

/// What `POST /v1/banks/{bank}/entities/merge` takes. Each side is an
/// entity id, `user` or `assistant`, or a name or alias that names exactly
/// one entity of the bank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeRequest {
    pub from: String,
    pub into: String,
}

/// What a merge did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Merged {
    /// The edit row, which `entity unmerge` takes.
    pub edit: Uuid,
    pub from: Uuid,
    pub into: Uuid,
    pub aliases_moved: usize,
    pub links_moved: usize,
    pub models_repointed: usize,
}

/// What an unmerge did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Unmerged {
    pub edit: Uuid,
    /// The merge it reversed.
    pub merge: Uuid,
    pub from: Uuid,
    pub into: Uuid,
    pub aliases_moved: usize,
    pub links_moved: usize,
    pub models_repointed: usize,
}

/// What `POST /v1/banks/{bank}/entities/alias/remove` takes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AliasRemoval {
    pub entity: String,
    pub alias: String,
    /// The entity the links that used the alias as their surface form move
    /// to. It gets the alias.
    #[serde(default)]
    pub relink_to: Option<String>,
}

/// What an alias removal did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AliasRemoved {
    pub entity: Uuid,
    pub relinked_to: Option<Uuid>,
    pub links_moved: usize,
}

/// What `POST /v1/banks/{bank}/entities/link` and `.../unlink` take.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkRequest {
    pub memory: String,
    pub entity: String,
}

/// What a link or unlink did. `changed` is false when the link was already
/// there, or already gone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LinkEdited {
    pub memory: Uuid,
    pub entity: Uuid,
    pub changed: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum EntityError {
    #[error("unknown bank")]
    UnknownBank,

    #[error("no entity of the bank has that id, name or alias")]
    UnknownEntity,

    #[error("the name matches {} entities; give one of their ids: {}", .matches.len(), ids(.matches))]
    Ambiguous { matches: Vec<Uuid> },

    #[error("no such memory in the bank")]
    UnknownMemory,

    #[error("the entity has no such alias")]
    UnknownAlias,

    #[error("no entity merge in the bank has that edit id")]
    UnknownMerge,

    #[error("user and assistant can only be merged into, never merged")]
    SeededFrom,

    #[error("an entity can't be merged into itself")]
    SameEntity,

    #[error("that entity was already merged into {into}")]
    AlreadyMerged { into: Uuid },

    #[error("the target was merged into {into}; merge into that instead")]
    TargetMerged { into: Uuid },

    #[error("that merge was already undone")]
    AlreadyUnmerged,

    #[error("the target has since been merged into {into}; unmerge that first")]
    MergedSince { into: Uuid },

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for EntityError {
    fn from(error: rusqlite::Error) -> Self {
        EntityError::Store(StoreError::Sqlite(error))
    }
}

fn ids(uuids: &[Uuid]) -> String {
    uuids
        .iter()
        .map(Uuid::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// An entity row as correction reads it.
#[derive(Debug, Clone)]
pub(crate) struct EntityRow {
    pub id: i64,
    pub uuid: Uuid,
    pub seeded: Option<String>,
    pub merged_into: Option<i64>,
}

fn entity_row(conn: &Connection, id: i64) -> Result<EntityRow, rusqlite::Error> {
    conn.query_row(
        "SELECT id, uuid, seeded, merged_into FROM entities WHERE id = ?1",
        [id],
        |row| {
            Ok(EntityRow {
                id: row.get(0)?,
                uuid: parse(&row.get::<_, String>(1)?),
                seeded: row.get(2)?,
                merged_into: row.get(3)?,
            })
        },
    )
}

fn parse(text: &str) -> Uuid {
    text.parse().expect("a stored uuid parses")
}

pub(crate) fn uuid_of(conn: &Connection, entity_id: i64) -> Result<Uuid, rusqlite::Error> {
    conn.query_row(
        "SELECT uuid FROM entities WHERE id = ?1",
        [entity_id],
        |row| row.get::<_, String>(0),
    )
    .map(|uuid| parse(&uuid))
}

/// The entity `reference` names in the bank: its id, `user` or
/// `assistant`, or a name or alias, compared without case. A name several
/// entities share is ambiguous, unless only one of them hasn't been merged:
/// a merged entity keeps the aliases its survivor already had.
pub(crate) fn resolve(
    conn: &Connection,
    bank_id: i64,
    reference: &str,
) -> Result<EntityRow, EntityError> {
    let reference = reference.trim();
    if let Ok(uuid) = reference.parse::<Uuid>() {
        let id: Option<i64> = conn
            .query_row(
                "SELECT id FROM entities WHERE uuid = ?1 AND bank_id = ?2",
                (uuid.to_string(), bank_id),
                |row| row.get(0),
            )
            .optional()?;
        return Ok(entity_row(conn, id.ok_or(EntityError::UnknownEntity)?)?);
    }
    if reference == "user" || reference == "assistant" {
        let id: i64 = conn.query_row(
            "SELECT id FROM entities WHERE bank_id = ?1 AND seeded = ?2",
            (bank_id, reference),
            |row| row.get(0),
        )?;
        return Ok(entity_row(conn, id)?);
    }
    let name = nfc(reference);
    let mut statement = conn.prepare_cached(
        "SELECT DISTINCT e.id FROM entities e
         LEFT JOIN entity_aliases a ON a.entity_id = e.id
         WHERE e.bank_id = ?1 AND (e.name = ?2 COLLATE NOCASE OR a.alias = ?2 COLLATE NOCASE)
         ORDER BY e.id",
    )?;
    let hits: Vec<i64> = statement
        .query_map((bank_id, &name), |row| row.get(0))?
        .collect::<Result<_, _>>()?;
    let rows: Vec<EntityRow> = hits
        .into_iter()
        .map(|id| entity_row(conn, id))
        .collect::<Result<_, _>>()?;
    let unmerged: Vec<&EntityRow> = rows.iter().filter(|r| r.merged_into.is_none()).collect();
    match (rows.len(), unmerged.len()) {
        (0, _) => Err(EntityError::UnknownEntity),
        (1, _) => Ok(rows[0].clone()),
        (_, 1) => Ok(unmerged[0].clone()),
        (_, 0) => Err(EntityError::Ambiguous {
            matches: rows.iter().map(|r| r.uuid).collect(),
        }),
        _ => Err(EntityError::Ambiguous {
            matches: unmerged.iter().map(|r| r.uuid).collect(),
        }),
    }
}

/// Logs an edit and returns its id.
fn log(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    kind: &str,
    memory_id: Option<i64>,
    entity_id: Option<i64>,
    details: &serde_json::Value,
) -> Result<Uuid, rusqlite::Error> {
    let uuid = store.new_id();
    tx.execute(
        "INSERT INTO edits (uuid, bank_id, kind, memory_id, entity_id, details, at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        (
            uuid.to_string(),
            bank_id,
            kind,
            memory_id,
            entity_id,
            details.to_string(),
            micros(store.now()),
        ),
    )?;
    Ok(uuid)
}

fn bank(tx: &Connection, bank: &str) -> Result<i64, EntityError> {
    Ok(find_bank(tx, bank)?.ok_or(EntityError::UnknownBank)?.0)
}

/// The models whose entity filter is any of `entities`.
fn filtering(
    conn: &Connection,
    bank_id: i64,
    entities: &[i64],
) -> Result<BTreeSet<i64>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT id FROM mental_models WHERE bank_id = ?1 AND filter_entity_id = ?2",
    )?;
    let mut models = BTreeSet::new();
    for entity in entities {
        for model in statement.query_map((bank_id, entity), |row| row.get::<_, i64>(0))? {
            models.insert(model?);
        }
    }
    Ok(models)
}

fn json_ids(value: &serde_json::Value, key: &str) -> Vec<i64> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .map(|ids| ids.iter().filter_map(serde_json::Value::as_i64).collect())
        .unwrap_or_default()
}

/// Merges `from` into `into`, returning what it did and the models to
/// refresh: those whose filter was repointed and those already on `into`,
/// which now see `from`'s memories.
pub(crate) fn merge(
    store: &Store,
    bank_name: &str,
    request: &MergeRequest,
) -> Result<(i64, Merged, BTreeSet<i64>), EntityError> {
    let now = micros(store.now());
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let bank_id = bank(&tx, bank_name)?;
    let from = resolve(&tx, bank_id, &request.from)?;
    let into = resolve(&tx, bank_id, &request.into)?;
    if from.id == into.id {
        return Err(EntityError::SameEntity);
    }
    if from.seeded.is_some() {
        return Err(EntityError::SeededFrom);
    }
    if let Some(merged) = from.merged_into {
        return Err(EntityError::AlreadyMerged {
            into: uuid_of(&tx, merged)?,
        });
    }
    if into.merged_into.is_some() {
        return Err(EntityError::TargetMerged {
            into: uuid_of(&tx, survivor(&tx, into.id)?)?,
        });
    }

    let aliases: Vec<i64> = {
        let mut statement = tx.prepare(
            "SELECT a.id FROM entity_aliases a
             WHERE a.entity_id = ?1
               AND NOT EXISTS (SELECT 1 FROM entity_aliases b
                               WHERE b.entity_id = ?2 AND b.alias = a.alias)
             ORDER BY a.id",
        )?;
        statement
            .query_map((from.id, into.id), |row| row.get(0))?
            .collect::<Result<_, _>>()?
    };
    for alias in &aliases {
        tx.execute(
            "UPDATE entity_aliases SET entity_id = ?2 WHERE id = ?1",
            (alias, into.id),
        )?;
    }
    let links: Vec<i64> = {
        let mut statement = tx.prepare(
            "SELECT memory_id FROM memory_entities
             WHERE entity_id = ?1
               AND memory_id NOT IN (SELECT memory_id FROM memory_entities WHERE entity_id = ?2)
             ORDER BY memory_id",
        )?;
        statement
            .query_map((from.id, into.id), |row| row.get(0))?
            .collect::<Result<_, _>>()?
    };
    for memory in &links {
        tx.execute(
            "UPDATE memory_entities SET entity_id = ?3 WHERE memory_id = ?1 AND entity_id = ?2",
            (memory, from.id, into.id),
        )?;
    }
    let repointed = filtering(&tx, bank_id, &[from.id])?;
    for model in &repointed {
        tx.execute(
            "UPDATE mental_models SET filter_entity_id = ?2, updated_at = ?3 WHERE id = ?1",
            (model, into.id, now),
        )?;
    }
    tx.execute(
        "UPDATE entities SET merged_into = ?2, updated_at = ?3 WHERE id = ?1",
        (from.id, into.id, now),
    )?;
    let edit = log(
        &tx,
        store,
        bank_id,
        EDIT_ENTITY_MERGED,
        None,
        Some(from.id),
        &serde_json::json!({
            "into": into.id,
            "aliases": aliases,
            "links": links,
            "models": repointed,
        }),
    )?;
    let mut refresh = repointed.clone();
    refresh.extend(filtering(&tx, bank_id, &[into.id])?);
    tx.commit()?;
    tracing::info!(%edit, aliases = aliases.len(), links = links.len(),
        models = repointed.len(), "merged an entity");
    Ok((
        bank_id,
        Merged {
            edit,
            from: from.uuid,
            into: into.uuid,
            aliases_moved: aliases.len(),
            links_moved: links.len(),
            models_repointed: repointed.len(),
        },
        refresh,
    ))
}

/// Reverses the merge `edit` names, and returns the models to refresh.
pub(crate) fn unmerge(
    store: &Store,
    bank_name: &str,
    edit: &str,
) -> Result<(i64, Unmerged, BTreeSet<i64>), EntityError> {
    let now = micros(store.now());
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let bank_id = bank(&tx, bank_name)?;
    let Ok(edit) = edit.trim().parse::<Uuid>() else {
        return Err(EntityError::UnknownMerge);
    };
    let found: Option<(i64, Option<i64>, String)> = tx
        .query_row(
            "SELECT id, entity_id, details FROM edits
             WHERE uuid = ?1 AND bank_id = ?2 AND kind = ?3",
            (edit.to_string(), bank_id, EDIT_ENTITY_MERGED),
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    let Some((merge_id, Some(from_id), details)) = found else {
        return Err(EntityError::UnknownMerge);
    };
    let details: serde_json::Value = serde_json::from_str(&details).unwrap_or_default();
    let into_id = details
        .get("into")
        .and_then(serde_json::Value::as_i64)
        .ok_or(EntityError::UnknownMerge)?;
    let undone = tx
        .query_row(
            "SELECT 1 FROM edits
             WHERE bank_id = ?1 AND kind = ?2 AND json_extract(details, '$.merge') = ?3",
            (bank_id, EDIT_ENTITY_UNMERGED, merge_id),
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    let from = entity_row(&tx, from_id)?;
    if undone || from.merged_into != Some(into_id) {
        return Err(EntityError::AlreadyUnmerged);
    }
    let into = entity_row(&tx, into_id)?;
    if let Some(merged) = into.merged_into {
        return Err(EntityError::MergedSince {
            into: uuid_of(&tx, merged)?,
        });
    }

    let mut aliases_moved = 0;
    for alias in json_ids(&details, "aliases") {
        aliases_moved += tx.execute(
            "UPDATE entity_aliases SET entity_id = ?2 WHERE id = ?1 AND entity_id = ?3",
            (alias, from.id, into.id),
        )?;
    }
    let mut links_moved = 0;
    for memory in json_ids(&details, "links") {
        links_moved += tx.execute(
            "UPDATE OR IGNORE memory_entities SET entity_id = ?2
             WHERE memory_id = ?1 AND entity_id = ?3",
            (memory, from.id, into.id),
        )?;
    }
    let mut repointed = BTreeSet::new();
    for model in json_ids(&details, "models") {
        if tx.execute(
            "UPDATE mental_models SET filter_entity_id = ?2, updated_at = ?4
             WHERE id = ?1 AND filter_entity_id = ?3",
            (model, from.id, into.id, now),
        )? == 1
        {
            repointed.insert(model);
        }
    }
    tx.execute(
        "UPDATE entities SET merged_into = NULL, updated_at = ?2 WHERE id = ?1",
        (from.id, now),
    )?;
    let unmerge = log(
        &tx,
        store,
        bank_id,
        EDIT_ENTITY_UNMERGED,
        None,
        Some(from.id),
        &serde_json::json!({
            "merge": merge_id,
            "into": into.id,
            "aliases": aliases_moved,
            "links": links_moved,
            "models": repointed,
        }),
    )?;
    let mut refresh = repointed.clone();
    refresh.extend(filtering(&tx, bank_id, &[into.id])?);
    tx.commit()?;
    tracing::info!(edit = %unmerge, merge = %edit, "unmerged an entity");
    Ok((
        bank_id,
        Unmerged {
            edit: unmerge,
            merge: edit,
            from: from.uuid,
            into: into.uuid,
            aliases_moved,
            links_moved,
            models_repointed: repointed.len(),
        },
        refresh,
    ))
}

/// Removes an alias from an entity. With `relink_to`, the links whose
/// surface form was the alias move to that entity, which gets the alias.
pub(crate) fn remove_alias(
    store: &Store,
    bank_name: &str,
    request: &AliasRemoval,
) -> Result<(i64, AliasRemoved, BTreeSet<i64>), EntityError> {
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let bank_id = bank(&tx, bank_name)?;
    let entity = resolve(&tx, bank_id, &request.entity)?;
    let alias = nfc(request.alias.trim());
    let found: Option<(i64, String)> = tx
        .query_row(
            "SELECT id, alias FROM entity_aliases
             WHERE entity_id = ?1 AND alias = ?2
             UNION ALL
             SELECT id, alias FROM entity_aliases
             WHERE entity_id = ?1 AND alias = ?2 COLLATE NOCASE
             LIMIT 1",
            (entity.id, &alias),
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((alias_id, alias)) = found else {
        return Err(EntityError::UnknownAlias);
    };
    let target = match &request.relink_to {
        Some(reference) => {
            let target = resolve(&tx, bank_id, reference)?;
            let target = entity_row(&tx, survivor(&tx, target.id)?)?;
            if target.id == entity.id {
                return Err(EntityError::SameEntity);
            }
            Some(target)
        }
        None => None,
    };

    tx.execute("DELETE FROM entity_aliases WHERE id = ?1", [alias_id])?;
    log_edit(
        &tx,
        store,
        bank_id,
        EDIT_ALIAS_REMOVED,
        Some(entity.id),
        &serde_json::json!({
            "alias_id": alias_id,
            "relink_to": target.as_ref().map(|t| t.id),
        })
        .to_string(),
    )?;

    let mut moved = Vec::new();
    let mut refresh = BTreeSet::new();
    if let Some(target) = &target {
        let memories: Vec<i64> = {
            let mut statement = tx.prepare(
                "SELECT memory_id FROM memory_entities
                 WHERE entity_id = ?1 AND surface_form = ?2 COLLATE NOCASE
                 ORDER BY memory_id",
            )?;
            statement
                .query_map((entity.id, &alias), |row| row.get(0))?
                .collect::<Result<_, _>>()?
        };
        for memory in memories {
            let already = tx
                .query_row(
                    "SELECT 1 FROM memory_entities WHERE memory_id = ?1 AND entity_id = ?2",
                    (memory, target.id),
                    |_| Ok(()),
                )
                .optional()?
                .is_some();
            if already {
                tx.execute(
                    "DELETE FROM memory_entities WHERE memory_id = ?1 AND entity_id = ?2",
                    (memory, entity.id),
                )?;
            } else {
                tx.execute(
                    "UPDATE memory_entities SET entity_id = ?3
                     WHERE memory_id = ?1 AND entity_id = ?2",
                    (memory, entity.id, target.id),
                )?;
            }
            moved.push(memory);
        }
        if !moved.is_empty() {
            log_edit(
                &tx,
                store,
                bank_id,
                EDIT_LINKS_MOVED,
                Some(entity.id),
                &serde_json::json!({ "to": target.id, "memories": moved }).to_string(),
            )?;
        }
        add_alias(&tx, store, bank_id, target.id, &alias)?;
        refresh = filtering(&tx, bank_id, &[entity.id, target.id])?;
    }
    tx.commit()?;
    Ok((
        bank_id,
        AliasRemoved {
            entity: entity.uuid,
            relinked_to: target.map(|t| t.uuid),
            links_moved: moved.len(),
        },
        refresh,
    ))
}

/// Links a memory to an entity, or unlinks it, as one logged edit. The
/// entity is taken as the one that survived any merge.
pub(crate) fn edit_link(
    store: &Store,
    bank_name: &str,
    request: &LinkRequest,
    link: bool,
) -> Result<(i64, LinkEdited, BTreeSet<i64>), EntityError> {
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let bank_id = bank(&tx, bank_name)?;
    let Ok(memory) = request.memory.trim().parse::<Uuid>() else {
        return Err(EntityError::UnknownMemory);
    };
    let memory_id: i64 = tx
        .query_row(
            "SELECT id FROM memories WHERE uuid = ?1 AND bank_id = ?2 AND hidden_at IS NULL",
            (memory.to_string(), bank_id),
            |row| row.get(0),
        )
        .optional()?
        .ok_or(EntityError::UnknownMemory)?;
    let entity = resolve(&tx, bank_id, &request.entity)?;
    let entity = entity_row(&tx, survivor(&tx, entity.id)?)?;
    let changed = if link {
        tx.execute(
            "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?1, ?2)",
            (memory_id, entity.id),
        )?
    } else {
        tx.execute(
            "DELETE FROM memory_entities
             WHERE memory_id = ?1
               AND (entity_id = ?2
                    OR entity_id IN (SELECT id FROM entities WHERE merged_into = ?2))",
            (memory_id, entity.id),
        )?
    } > 0;
    if changed {
        log(
            &tx,
            store,
            bank_id,
            if link {
                EDIT_ENTITY_LINKED
            } else {
                EDIT_ENTITY_UNLINKED
            },
            Some(memory_id),
            Some(entity.id),
            &serde_json::json!({}),
        )?;
    }
    let refresh = if changed {
        filtering(&tx, bank_id, &[entity.id])?
    } else {
        BTreeSet::new()
    };
    tx.commit()?;
    Ok((
        bank_id,
        LinkEdited {
            memory,
            entity: entity.uuid,
            changed,
        },
        refresh,
    ))
}
