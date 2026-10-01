//! Ingest: receiving turns and documents and queueing them for extraction.
//!
//! Ingest scans the input for secrets, stores it as a source, splits it into
//! chunks and queues the chunks. It never calls a model; extraction takes
//! chunks off the queue ([`crate::queue`]). The decisions it implements:
//!
//! - **Sources are kept verbatim** after the secret scan (ADR 0002). The
//!   stored text is the clean turn, never Hermes' `api_content`.
//! - **Ingest is idempotent** ("What is a memory record?", TIM-90). The key
//!   is bank, session, message time and content hash for a turn, and bank,
//!   document id and content hash for a document. A conflicting ingest does
//!   nothing. The hash is taken after redaction, so nothing stored is
//!   derived from a secret.
//! - **Edited documents** are matched by chunk hash: a chunk seen in any
//!   earlier version of the same document id is skipped, even once its text
//!   has been swept or erased, because the hash is the tombstone (TIM-92).
//! - **The turn that asks to forget** is stored as a tombstone only: its key
//!   and nothing else. It's never chunked or queued, and its recall row is
//!   deleted (ADR 0010).
//! - **Speakers** (TIM-94, decision 1) are resolved by `<platform>:<id>`
//!   through `speaker_ids`, never through aliases. The owner's platform ids
//!   map to the seeded `user`, a turn with no author is the owner's, and
//!   anyone else becomes a `person` entity with a speaker id of their own.
//!   A display name is only ever an alias, so it can't capture an identity.
//!
//! Everything for one ingest commits in one transaction.

use std::collections::BTreeSet;

use jiff::Timestamp;
use jiff::civil::Date;
use jiff::tz::TimeZone;
use rusqlite::{OptionalExtension, Transaction};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::chunking::{chunk_hash, hex, split_document};
use crate::secrets::{SecretKind, scan};
use crate::store::bank::{add_alias, log_edit, set_speaker_id};
use crate::store::{Store, StoreError, micros, nfc};

/// Hermes' `turn_author` (TIM-94, decision 1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnAuthor {
    /// The platform's id for the speaker, such as a Discord user id. The
    /// speaker is resolved by the speaker id `<platform>:<id>`, the form the
    /// owner's platform ids take in bank config.
    pub id: String,
    pub name: Option<String>,
    #[serde(default)]
    pub is_bot: bool,
}

/// What `sync_turn` sends (TIM-94, decision 5 as amended by TIM-96 and
/// TIM-99). The plugin has already stripped any backfilled channel history
/// before `[New message]`; the `[Name] ` prefix in shared threads stays and
/// is stored as sent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    pub session_id: String,
    /// The user message's time. It's the turn's `observed_at` and part of
    /// its key.
    pub message_at: Timestamp,
    /// `hermes_time.get_timezone_name()`. The bank's default when absent.
    pub timezone: Option<String>,
    /// The clean user message.
    pub user_text: String,
    pub assistant_text: String,
    /// `None` on the CLI, TUI and Hermes UI, where the turn is the owner's.
    pub author: Option<TurnAuthor>,
    pub platform: Option<String>,
    /// The `recall_id` the plugin echoes from `prefetch`.
    pub recall_id: Option<String>,
    /// The turn called `memory_forget` (ADR 0010).
    #[serde(default)]
    pub forget_requested: bool,
}

/// A document sent through the API or `asphodel ingest`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    /// `--id`. Re-ingesting the same id with new text is an edit.
    pub document_id: String,
    /// Plain text or markdown.
    pub text: String,
    /// `--date`. The document's `observed_at` is the start of this day in
    /// its timezone.
    pub reference_date: Date,
    /// `false` with `--inexact` (TIM-92).
    pub reference_date_exact: bool,
    /// The bank's default when absent.
    pub timezone: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// A new source, stored and its chunks queued.
    Stored,
    /// The key was already there, so nothing changed.
    Duplicate,
    /// A `forget_requested` turn, stored as a tombstone only.
    Tombstone,
}

/// Whose words a turn holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Speaker {
    /// The seeded `user` for the owner, otherwise an entity of their own.
    pub entity: Uuid,
    pub owner: bool,
}

/// What one ingest did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Ingested {
    /// The source's public id; the existing one on a duplicate.
    pub source: Uuid,
    pub outcome: Outcome,
    /// Chunks queued for extraction by this call.
    pub chunks_queued: usize,
    /// Chunks of an edited document already seen in an earlier version, and
    /// so not stored or queued again.
    pub chunks_skipped: usize,
    /// The secret kinds that fired on what this call stored.
    pub secret_kinds: BTreeSet<SecretKind>,
    /// The speaker of a stored turn; `None` for documents, duplicates and
    /// tombstones.
    pub speaker: Option<Speaker>,
}

impl Ingested {
    fn nothing(source: Uuid, outcome: Outcome) -> Self {
        Self {
            source,
            outcome,
            chunks_queued: 0,
            chunks_skipped: 0,
            secret_kinds: BTreeSet::new(),
            speaker: None,
        }
    }
}

/// Why an ingest was refused. Nothing is stored when it is, and no variant
/// carries content (ADR 0010).
#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("unknown bank")]
    UnknownBank,
    #[error("unknown timezone")]
    InvalidTimezone,
    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for IngestError {
    fn from(error: rusqlite::Error) -> Self {
        IngestError::Store(StoreError::Sqlite(error))
    }
}

/// Queue priorities: turns go ahead of document chunks (TIM-92).
pub(crate) const PRIORITY_TURN: i64 = 0;
pub(crate) const PRIORITY_DOCUMENT: i64 = 1;

/// The separator between the user message and the reply in a turn's chunk.
/// The message is characters `0..len(user)` of the chunk and the reply
/// starts after this.
pub const TURN_SEPARATOR: &str = "\n\n";

/// Ingests a turn into `bank`.
pub fn ingest_turn(store: &Store, bank: &str, turn: &Turn) -> Result<Ingested, IngestError> {
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, bank_timezone) = find_bank(&tx, bank)?.ok_or(IngestError::UnknownBank)?;
    let timezone = resolve_timezone(turn.timezone.as_deref(), &bank_timezone)?;

    let user = scan(&turn.user_text);
    let reply = scan(&turn.assistant_text);
    let content_hash = turn_hash(&user.text, &reply.text);
    let message_at = micros(turn.message_at);
    let now = micros(store.now());

    let existing: Option<String> = tx
        .query_row(
            "SELECT uuid FROM sources
             WHERE bank_id = ?1 AND kind = 'turn' AND session_id = ?2 AND message_at = ?3
               AND content_hash = ?4",
            (bank_id, &turn.session_id, message_at, &content_hash),
            |row| row.get(0),
        )
        .optional()?;
    if let Some(existing) = existing {
        return Ok(Ingested::nothing(parse_uuid(&existing), Outcome::Duplicate));
    }

    let source = store.new_id();
    if turn.forget_requested {
        // ADR 0010: only the key, from the start. No text, no provenance
        // beyond the key, nothing to extract.
        tx.execute(
            "INSERT INTO sources (uuid, bank_id, kind, session_id, message_at, content_hash,
                                  observed_at, timezone, ingested_at, tombstoned_at,
                                  tombstone_reason)
             VALUES (?1, ?2, 'turn', ?3, ?4, ?5, ?4, ?6, ?7, ?7, 'forget_requested')",
            (
                source.to_string(),
                bank_id,
                &turn.session_id,
                message_at,
                &content_hash,
                &timezone,
                now,
            ),
        )?;
        if let Some(recall_id) = &turn.recall_id {
            tx.execute(
                "DELETE FROM recalls WHERE bank_id = ?1 AND uuid = ?2",
                (bank_id, recall_id),
            )?;
        }
        count_turn(&tx, bank_id, message_at)?;
        tx.commit()?;
        tracing::debug!(%source, "stored a forget request as a tombstone");
        return Ok(Ingested::nothing(source, Outcome::Tombstone));
    }

    let speaker = resolve_speaker(&tx, store, bank_id, turn)?;
    let secret_kinds: BTreeSet<SecretKind> = user.kinds.union(&reply.kinds).copied().collect();
    let author = turn.author.as_ref();
    tx.execute(
        "INSERT INTO sources (uuid, bank_id, kind, session_id, message_at, content_hash, platform,
                              author_id, author_name, author_is_bot, observed_at, timezone, text,
                              reply, secret_kinds, recall_id, ingested_at)
         VALUES (?1, ?2, 'turn', ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?4, ?10, ?11, ?12, ?13, ?14, ?15)",
        rusqlite::params![
            source.to_string(),
            bank_id,
            &turn.session_id,
            message_at,
            &content_hash,
            &turn.platform,
            author.map(|a| &a.id),
            author.and_then(|a| a.name.as_ref()),
            author.is_some_and(|a| a.is_bot),
            &timezone,
            &user.text,
            &reply.text,
            kinds_json(&secret_kinds),
            &turn.recall_id,
            now,
        ],
    )?;
    let source_id = tx.last_insert_rowid();

    let text = [user.text.as_str(), TURN_SEPARATOR, reply.text.as_str()].concat();
    let chunk = NewChunk {
        position: 0,
        heading_path: &[],
        start: 0,
        end: text.chars().count(),
        text: &text,
    };
    insert_chunk(
        &tx,
        store,
        bank_id,
        source_id,
        &chunk,
        PRIORITY_TURN,
        message_at,
    )?;
    count_turn(&tx, bank_id, message_at)?;
    tx.commit()?;
    if !secret_kinds.is_empty() {
        tracing::info!(%source, kinds = ?secret_kinds, "redacted secrets from a turn");
    }
    Ok(Ingested {
        source,
        outcome: Outcome::Stored,
        chunks_queued: 1,
        chunks_skipped: 0,
        secret_kinds,
        speaker: Some(speaker),
    })
}

/// Ingests a document into `bank`.
pub fn ingest_document(
    store: &Store,
    bank: &str,
    document: &Document,
) -> Result<Ingested, IngestError> {
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let (bank_id, bank_timezone) = find_bank(&tx, bank)?.ok_or(IngestError::UnknownBank)?;
    let timezone = resolve_timezone(document.timezone.as_deref(), &bank_timezone)?;
    let zone = TimeZone::get(&timezone).map_err(|_| IngestError::InvalidTimezone)?;
    // TIM-90: a time is stored as the start of its unit in the source's
    // timezone, so the reference date is that day's midnight there.
    let observed_at = micros(
        document
            .reference_date
            .to_zoned(zone)
            .map_err(|_| IngestError::InvalidTimezone)?
            .timestamp(),
    );

    let scanned = scan(&document.text);
    let content_hash = hex(&Sha256::digest(scanned.text.as_bytes()));
    let now = micros(store.now());

    let existing: Option<String> = tx
        .query_row(
            "SELECT uuid FROM sources
             WHERE bank_id = ?1 AND kind = 'document' AND document_id = ?2 AND content_hash = ?3",
            (bank_id, &document.document_id, &content_hash),
            |row| row.get(0),
        )
        .optional()?;
    if let Some(existing) = existing {
        return Ok(Ingested::nothing(parse_uuid(&existing), Outcome::Duplicate));
    }

    let source = store.new_id();
    tx.execute(
        "INSERT INTO sources (uuid, bank_id, kind, document_id, content_hash, observed_at,
                              reference_date, reference_date_exact, timezone, text, secret_kinds,
                              ingested_at)
         VALUES (?1, ?2, 'document', ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        rusqlite::params![
            source.to_string(),
            bank_id,
            &document.document_id,
            &content_hash,
            observed_at,
            document.reference_date.to_string(),
            document.reference_date_exact,
            &timezone,
            &scanned.text,
            kinds_json(&scanned.kinds),
            now,
        ],
    )?;
    let source_id = tx.last_insert_rowid();

    let mut queued = 0;
    let mut skipped = 0;
    for (position, chunk) in split_document(&scanned.text).iter().enumerate() {
        let new = NewChunk {
            position,
            heading_path: &chunk.heading_path,
            start: chunk.start,
            end: chunk.end,
            text: &chunk.text,
        };
        if seen_before(&tx, bank_id, &document.document_id, source_id, &new.hash())? {
            skipped += 1;
            continue;
        }
        insert_chunk(
            &tx,
            store,
            bank_id,
            source_id,
            &new,
            PRIORITY_DOCUMENT,
            observed_at,
        )?;
        queued += 1;
    }
    tx.commit()?;
    if !scanned.kinds.is_empty() {
        tracing::info!(%source, kinds = ?scanned.kinds, "redacted secrets from a document");
    }
    Ok(Ingested {
        source,
        outcome: Outcome::Stored,
        chunks_queued: queued,
        chunks_skipped: skipped,
        secret_kinds: scanned.kinds,
        speaker: None,
    })
}

/// The bank's rowid and default timezone, or `None` when there's no such
/// bank.
pub(crate) fn find_bank(
    conn: &rusqlite::Connection,
    bank: &str,
) -> Result<Option<(i64, String)>, rusqlite::Error> {
    conn.query_row(
        "SELECT id, timezone FROM banks WHERE name = ?1",
        [bank.trim()],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
    .optional()
}

fn resolve_timezone(given: Option<&str>, bank_default: &str) -> Result<String, IngestError> {
    let timezone = given.unwrap_or(bank_default);
    TimeZone::get(timezone).map_err(|_| IngestError::InvalidTimezone)?;
    Ok(timezone.to_owned())
}

/// The turn's content hash: the redacted message and reply, each
/// length-prefixed so moving text between them changes the hash.
fn turn_hash(user: &str, reply: &str) -> String {
    let mut hash = Sha256::new();
    for part in [user, reply] {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part.as_bytes());
    }
    hex(&hash.finalize())
}

/// The kinds as the `secret_kinds` column holds them: a JSON array of names,
/// or NULL when none fired.
fn kinds_json(kinds: &BTreeSet<SecretKind>) -> Option<String> {
    (!kinds.is_empty()).then(|| {
        serde_json::to_string(&kinds.iter().map(|k| k.as_str()).collect::<Vec<_>>())
            .expect("a list of names serialises")
    })
}

fn parse_uuid(text: &str) -> Uuid {
    text.parse().expect("a stored uuid parses")
}

/// Advances the bank's turn counter and the start of bank time's latest
/// full-speed window (ADR 0004). Documents don't count.
fn count_turn(tx: &Transaction<'_>, bank_id: i64, message_at: i64) -> Result<(), IngestError> {
    tx.execute(
        "UPDATE banks SET turns = turns + 1,
                          last_turn_at = MAX(COALESCE(last_turn_at, ?2), ?2)
         WHERE id = ?1",
        (bank_id, message_at),
    )?;
    Ok(())
}

/// Whether a chunk with this hash was stored for an earlier version of the
/// document, whatever has happened to its text since. The version being
/// ingested (`source_id`) doesn't count, so a section repeated within it is
/// stored and queued each time (TIM-92 skips earlier versions only).
fn seen_before(
    tx: &Transaction<'_>,
    bank_id: i64,
    document_id: &str,
    source_id: i64,
    hash: &str,
) -> Result<bool, IngestError> {
    Ok(tx
        .query_row(
            "SELECT 1 FROM chunks c JOIN sources s ON s.id = c.source_id
             WHERE c.bank_id = ?1 AND c.content_hash = ?2
               AND s.kind = 'document' AND s.document_id = ?3 AND s.id != ?4
             LIMIT 1",
            (bank_id, hash, document_id, source_id),
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

struct NewChunk<'a> {
    position: usize,
    heading_path: &'a [String],
    start: usize,
    end: usize,
    text: &'a str,
}

impl NewChunk<'_> {
    fn hash(&self) -> String {
        chunk_hash(self.heading_path, self.text)
    }
}

/// Stores a chunk and queues it.
fn insert_chunk(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    source_id: i64,
    chunk: &NewChunk<'_>,
    priority: i64,
    observed_at: i64,
) -> Result<(), IngestError> {
    let heading_path = (!chunk.heading_path.is_empty())
        .then(|| serde_json::to_string(chunk.heading_path).expect("headings serialise"));
    tx.execute(
        "INSERT INTO chunks (uuid, bank_id, source_id, position, heading_path, content_hash,
                             start_offset, end_offset, text)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        rusqlite::params![
            store.new_id().to_string(),
            bank_id,
            source_id,
            chunk.position as i64,
            heading_path,
            chunk.hash(),
            chunk.start as i64,
            chunk.end as i64,
            chunk.text,
        ],
    )?;
    let chunk_id = tx.last_insert_rowid();
    tx.execute(
        "INSERT INTO extraction_queue (bank_id, kind, chunk_id, priority, observed_at, enqueued_at)
         VALUES (?1, 'chunk', ?2, ?3, ?4, ?5)",
        (
            bank_id,
            chunk_id,
            priority,
            observed_at,
            micros(store.now()),
        ),
    )?;
    Ok(())
}

/// Resolves a turn's speaker, creating an entity for a speaker seen for the
/// first time.
fn resolve_speaker(
    tx: &Transaction<'_>,
    store: &Store,
    bank_id: i64,
    turn: &Turn,
) -> Result<Speaker, IngestError> {
    let Some(author) = &turn.author else {
        // The CLI, TUI and Hermes UI send no author: the owner's turn.
        return seeded_user(tx, bank_id);
    };
    let platform_id = match &turn.platform {
        Some(platform) => format!("{platform}:{}", author.id),
        None => author.id.clone(),
    };
    // Only `speaker_ids` identifies a speaker. Aliases are free text, and a
    // display name shaped like a platform id must not capture that id.
    let found: Option<i64> = tx
        .query_row(
            "SELECT entity_id FROM speaker_ids WHERE bank_id = ?1 AND platform_id = ?2",
            (bank_id, platform_id.trim()),
            |row| row.get(0),
        )
        .optional()?;
    if let Some(entity_id) = found {
        let (uuid, seeded) = surviving_entity(tx, entity_id)?;
        return Ok(Speaker {
            entity: parse_uuid(&uuid),
            owner: seeded.as_deref() == Some("user"),
        });
    }

    // Composed, as every entity name and alias is stored (schema version 3).
    let name = nfc(author
        .name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or(&platform_id));
    let name = name.as_str();
    let uuid = store.new_id();
    let now = micros(store.now());
    tx.execute(
        "INSERT INTO entities (uuid, bank_id, name, kind, created_at, updated_at)
         VALUES (?1, ?2, ?3, 'person', ?4, ?4)",
        (uuid.to_string(), bank_id, name, now),
    )?;
    let entity_id = tx.last_insert_rowid();
    log_edit(tx, store, bank_id, "entity_created", Some(entity_id), "{}")?;
    set_speaker_id(tx, store, bank_id, &platform_id, entity_id)?;
    // Both go in as aliases too, for entity search; neither is an identity.
    add_alias(tx, store, bank_id, entity_id, &platform_id)?;
    if name != platform_id {
        add_alias(tx, store, bank_id, entity_id, name)?;
    }
    Ok(Speaker {
        entity: uuid,
        owner: false,
    })
}

fn seeded_user(tx: &Transaction<'_>, bank_id: i64) -> Result<Speaker, IngestError> {
    let uuid: String = tx.query_row(
        "SELECT uuid FROM entities WHERE bank_id = ?1 AND seeded = 'user'",
        [bank_id],
        |row| row.get(0),
    )?;
    Ok(Speaker {
        entity: parse_uuid(&uuid),
        owner: true,
    })
}

/// Follows `merged_into` to the entity that survived any merges (ADR 0010).
/// The walk is bounded, so a corrupt cycle can't hang ingest.
fn surviving_entity(
    tx: &Transaction<'_>,
    mut entity_id: i64,
) -> Result<(String, Option<String>), IngestError> {
    for _ in 0..MAX_MERGE_DEPTH {
        let (uuid, seeded, merged_into): (String, Option<String>, Option<i64>) = tx.query_row(
            "SELECT uuid, seeded, merged_into FROM entities WHERE id = ?1",
            [entity_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        match merged_into {
            Some(next) if next != entity_id => entity_id = next,
            _ => return Ok((uuid, seeded)),
        }
    }
    tracing::error!(
        entity_id,
        "an entity's merge chain doesn't end; treating it as the speaker"
    );
    let (uuid, seeded) = tx.query_row(
        "SELECT uuid, seeded FROM entities WHERE id = ?1",
        [entity_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok((uuid, seeded))
}

const MAX_MERGE_DEPTH: usize = 64;
