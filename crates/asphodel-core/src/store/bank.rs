//! Bank create-or-merge.
//!
//! `PUT /v1/banks/{bank}` and `asphodel bank create|config` both land here.
//! Fields present are set, aliases are added, and absent fields are left
//! alone, so many Hermes instances calling it can't undo a change made
//! through the CLI. Renaming adds an alias and never removes one. Creating a
//! bank seeds the `user` and `assistant` entities with the names as their
//! first aliases, and the "User profile" mental model; the merge
//! path never seeds, so an owner who deletes the profile doesn't get it back
//! on the next Hermes start.

use jiff::tz::TimeZone;
use rusqlite::{OptionalExtension, Transaction};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{Store, StoreError, micros, nfc};

/// The timezone a bank gets when none was given at creation.
pub const DEFAULT_TIMEZONE: &str = "UTC";

/// The seeded profile's question.
pub const PROFILE_NAME: &str = "User profile";
pub const PROFILE_QUESTION: &str = "Who is the user: their preferences, important people, work and home, \
     the platforms they use, and how they like to be helped. Not upcoming events, tasks or routines.";

/// The seeded profile's stored filters: facts, plus states of
/// volatility weeks or slower. A memory with no volatility passes, and there
/// is no entity filter, since the profile is about the whole bank. The default
/// profile also admits non-routine recurring memories at selection time.
pub const PROFILE_FILTER_KINDS: &str = r#"["fact","state"]"#;
pub const PROFILE_MIN_VOLATILITY: &str = "weeks";

/// What `initialize` and `bank config` send, as the body of
/// `PUT /v1/banks/{bank}`. Every field is optional; an absent one is left as
/// it is.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct BankIdentity {
    pub owner_name: Option<String>,
    /// Platform ids of the owner, such as `discord:1234`. Each becomes the
    /// `user` entity's speaker id, the only thing a turn's speaker is
    /// resolved through, and an alias of it.
    pub owner_platform_ids: Vec<String>,
    pub assistant_name: Option<String>,
    /// An IANA timezone name, the default for sources without one.
    pub timezone: Option<String>,
}

/// The models a bank is served with, recorded at creation. A later change goes
/// through `asphodel reembed`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelIds {
    pub embedding: String,
    pub reranker: String,
}

/// A bank as the API shows it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Bank {
    pub id: Uuid,
    pub name: String,
    pub owner_name: Option<String>,
    pub assistant_name: Option<String>,
    pub timezone: String,
    pub embedding_model: String,
    pub reranker_model: String,
    /// Whether this call created the bank.
    pub created: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum BankError {
    #[error("a bank name must not be empty")]
    EmptyName,

    #[error("unknown timezone")]
    InvalidTimezone,

    /// The service was built without models, so there are no ids to record.
    #[error("no models are loaded, so a bank can't record them")]
    NoModels,

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for BankError {
    fn from(error: rusqlite::Error) -> Self {
        BankError::Store(StoreError::Sqlite(error))
    }
}

/// Creates `name` or merges `identity` into it. Idempotent: the same call
/// twice gives the same bank, entities and aliases.
pub fn ensure(
    store: &Store,
    name: &str,
    identity: &BankIdentity,
    models: &ModelIds,
    profile_max_tokens: u32,
) -> Result<Bank, BankError> {
    let name = name.trim();
    if name.is_empty() {
        return Err(BankError::EmptyName);
    }
    if let Some(timezone) = &identity.timezone
        && TimeZone::get(timezone).is_err()
    {
        return Err(BankError::InvalidTimezone);
    }

    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let existing = tx
        .query_row(
            "SELECT id, uuid FROM banks WHERE name = ?1",
            [name],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()?;
    let (bank_id, created) = match existing {
        Some((bank_id, _)) => {
            merge(&tx, store, bank_id, identity)?;
            (bank_id, false)
        }
        None => (
            create(&tx, store, name, identity, models, profile_max_tokens)?,
            true,
        ),
    };
    let bank = tx.query_row(
        "SELECT uuid, name, owner_name, assistant_name, timezone, embedding_model, reranker_model
         FROM banks WHERE id = ?1",
        [bank_id],
        |row| {
            Ok(Bank {
                id: row
                    .get::<_, String>(0)?
                    .parse()
                    .expect("a stored bank uuid parses"),
                name: row.get(1)?,
                owner_name: row.get(2)?,
                assistant_name: row.get(3)?,
                timezone: row.get(4)?,
                embedding_model: row.get(5)?,
                reranker_model: row.get(6)?,
                created,
            })
        },
    )?;
    tx.commit()?;
    if created {
        tracing::info!(bank = %bank.id, "created a bank");
    }
    Ok(bank)
}

fn create(
    tx: &Transaction<'_>,
    store: &Store,
    name: &str,
    identity: &BankIdentity,
    models: &ModelIds,
    profile_max_tokens: u32,
) -> Result<i64, BankError> {
    let now = micros(store.now());
    tx.execute(
        "INSERT INTO banks (uuid, name, owner_name, assistant_name, timezone, embedding_model,
                            reranker_model, turns, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, ?8, ?8)",
        (
            store.new_id().to_string(),
            name,
            &identity.owner_name,
            &identity.assistant_name,
            identity.timezone.as_deref().unwrap_or(DEFAULT_TIMEZONE),
            &models.embedding,
            &models.reranker,
            now,
        ),
    )?;
    let bank_id = tx.last_insert_rowid();

    let user = seed_entity(
        tx,
        store,
        bank_id,
        "user",
        identity.owner_name.as_deref().unwrap_or("user"),
        "person",
    )?;
    if let Some(owner) = &identity.owner_name {
        add_alias(tx, store, bank_id, user, owner)?;
    }
    for platform_id in &identity.owner_platform_ids {
        add_alias(tx, store, bank_id, user, platform_id)?;
        set_speaker_id(tx, store, bank_id, platform_id, user)?;
    }

    let assistant = seed_entity(
        tx,
        store,
        bank_id,
        "assistant",
        identity.assistant_name.as_deref().unwrap_or("assistant"),
        "thing",
    )?;
    if let Some(assistant_name) = &identity.assistant_name {
        add_alias(tx, store, bank_id, assistant, assistant_name)?;
    }

    tx.execute(
        "INSERT INTO mental_models (uuid, bank_id, name, question, max_tokens, filter_kinds,
                                    filter_min_volatility, enabled, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 1, ?8, ?8)",
        (
            store.new_id().to_string(),
            bank_id,
            PROFILE_NAME,
            PROFILE_QUESTION,
            profile_max_tokens,
            PROFILE_FILTER_KINDS,
            PROFILE_MIN_VOLATILITY,
            now,
        ),
    )?;
    log_edit(tx, store, bank_id, "bank_created", None, "{}")?;
    Ok(bank_id)
}

fn merge(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    identity: &BankIdentity,
) -> Result<(), BankError> {
    let now = micros(store.now());
    let user = seeded_entity(tx, bank_id, "user")?;
    let assistant = seeded_entity(tx, bank_id, "assistant")?;

    if let Some(owner) = &identity.owner_name {
        tx.execute(
            "UPDATE banks SET owner_name = ?1, updated_at = ?2 WHERE id = ?3",
            (owner, now, bank_id),
        )?;
        rename_entity(tx, store, bank_id, user, owner)?;
    }
    for platform_id in &identity.owner_platform_ids {
        add_alias(tx, store, bank_id, user, platform_id)?;
        set_speaker_id(tx, store, bank_id, platform_id, user)?;
    }
    if let Some(assistant_name) = &identity.assistant_name {
        tx.execute(
            "UPDATE banks SET assistant_name = ?1, updated_at = ?2 WHERE id = ?3",
            (assistant_name, now, bank_id),
        )?;
        rename_entity(tx, store, bank_id, assistant, assistant_name)?;
    }
    if let Some(timezone) = &identity.timezone {
        tx.execute(
            "UPDATE banks SET timezone = ?1, updated_at = ?2 WHERE id = ?3",
            (timezone, now, bank_id),
        )?;
    }
    Ok(())
}

fn seed_entity(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    seeded: &str,
    name: &str,
    kind: &str,
) -> Result<i64, BankError> {
    let now = micros(store.now());
    tx.execute(
        "INSERT INTO entities (uuid, bank_id, name, kind, seeded, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
        (
            store.new_id().to_string(),
            bank_id,
            nfc(name),
            kind,
            seeded,
            now,
        ),
    )?;
    Ok(tx.last_insert_rowid())
}

fn seeded_entity(tx: &Transaction<'_>, bank_id: i64, seeded: &str) -> Result<i64, BankError> {
    Ok(tx.query_row(
        "SELECT id FROM entities WHERE bank_id = ?1 AND seeded = ?2",
        (bank_id, seeded),
        |row| row.get(0),
    )?)
}

/// Gives a seeded entity a new canonical name. The old name stays as an
/// alias, and the new one is added as one: a rename never removes.
fn rename_entity(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    entity_id: i64,
    name: &str,
) -> Result<(), BankError> {
    let name = nfc(name);
    let name = name.as_str();
    let current: String = tx.query_row(
        "SELECT name FROM entities WHERE id = ?1",
        [entity_id],
        |row| row.get(0),
    )?;
    if current != name {
        add_alias(tx, store, bank_id, entity_id, &current)?;
        tx.execute(
            "UPDATE entities SET name = ?1, updated_at = ?2 WHERE id = ?3",
            (name, micros(store.now()), entity_id),
        )?;
    }
    Ok(add_alias(tx, store, bank_id, entity_id, name)?)
}

/// Adds an alias, logging it as an edit so a mislink can be undone. An alias
/// the entity already has is left alone and not logged.
pub(crate) fn add_alias(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    entity_id: i64,
    alias: &str,
) -> Result<(), rusqlite::Error> {
    // Composed, so the alias FTS indexes it as extraction searches it.
    let alias = nfc(alias.trim());
    if alias.is_empty() {
        return Ok(());
    }
    let inserted = tx.execute(
        "INSERT OR IGNORE INTO entity_aliases (bank_id, entity_id, alias, created_at)
         VALUES (?1, ?2, ?3, ?4)",
        (bank_id, entity_id, alias, micros(store.now())),
    )?;
    if inserted == 1 {
        let alias_id = tx.last_insert_rowid();
        log_edit(
            tx,
            store,
            bank_id,
            "alias_added",
            Some(entity_id),
            &format!("{{\"alias_id\":{alias_id}}}"),
        )?;
    }
    Ok(())
}

/// Maps a platform id to the speaker entity it identifies. Bank config
/// asserts the owner's ids, so it takes an id over from whichever entity
/// held it, such as one ingest created when the owner spoke before the id
/// was configured. A change is logged as an edit, by rowid only.
pub(crate) fn set_speaker_id(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    platform_id: &str,
    entity_id: i64,
) -> Result<(), rusqlite::Error> {
    let platform_id = platform_id.trim();
    if platform_id.is_empty() {
        return Ok(());
    }
    let now = micros(store.now());
    let changed = tx.execute(
        "INSERT INTO speaker_ids (bank_id, platform_id, entity_id, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?4)
         ON CONFLICT (bank_id, platform_id) DO UPDATE
           SET entity_id = excluded.entity_id, updated_at = excluded.updated_at
           WHERE entity_id != excluded.entity_id",
        (bank_id, platform_id, entity_id, now),
    )?;
    if changed == 1 {
        let speaker_id: i64 = tx.query_row(
            "SELECT id FROM speaker_ids WHERE bank_id = ?1 AND platform_id = ?2",
            (bank_id, platform_id),
            |row| row.get(0),
        )?;
        log_edit(
            tx,
            store,
            bank_id,
            "speaker_id_set",
            Some(entity_id),
            &format!("{{\"speaker_id\":{speaker_id}}}"),
        )?;
    }
    Ok(())
}

pub(crate) fn log_edit(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    kind: &str,
    entity_id: Option<i64>,
    details: &str,
) -> Result<(), rusqlite::Error> {
    tx.execute(
        "INSERT INTO edits (uuid, bank_id, kind, entity_id, details, at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        (
            store.new_id().to_string(),
            bank_id,
            kind,
            entity_id,
            details,
            micros(store.now()),
        ),
    )?;
    Ok(())
}

/// Logs an edit to a memory's metadata.
/// `details` holds ids, times and levels, never content.
pub(crate) fn log_memory_edit(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    kind: &str,
    memory_id: i64,
    details: &str,
) -> Result<(), rusqlite::Error> {
    tx.execute(
        "INSERT INTO edits (uuid, bank_id, kind, memory_id, details, at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        (
            store.new_id().to_string(),
            bank_id,
            kind,
            memory_id,
            details,
            micros(store.now()),
        ),
    )?;
    Ok(())
}
