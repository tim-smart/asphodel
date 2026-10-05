//! Translating a memory into `[llm] language`.
//!
//! `[llm] language` only shapes what extraction writes from now on, so a
//! store that held memories in another language before it was set keeps
//! them, and an English query can't reach them. `asphodel memory translate`
//! fixes one memory at a time, named by the operator.
//!
//! A memory's sentence never changes, so a translation is a new memory that
//! supersedes the one named, the way a refinement does: the same chain, so
//! it inherits every access and with them the strength. It copies
//! everything but the sentence: kind, window, significance and the owner's
//! setting, the chunk and span it came from, and its entity links. It gets
//! no access of its own, is embedded with the bank's embedder, and the
//! supersession is logged as `memory_refined`, with `translated_to`, so the
//! models citing it refresh. This is the one place the LLM writes a sentence
//! into an existing chain without the owner having said anything new.
//!
//! The LLM is shown the sentence alone, never the passage: the passage may
//! already be redacted or swept, and the span is provenance, not input.
//!
//! Naming an id doesn't make a request safe to repeat, so the chain does.
//! The memory has to be its chain's head, checked before the call and again
//! in the transaction that commits, which is the only write: a request
//! repeated after it committed, or overtaken by extraction while the LLM
//! answered, is refused with the head it now has. A head whose predecessor
//! was translated into the language, or a sentence the LLM hands back
//! unchanged, writes nothing. Nothing is held while the LLM answers.
//! Afterwards the bank is held through embedding and commit, after waiting
//! at most five seconds for in-flight extractions or another hold. A busy
//! bank is refused without writing, so the operator can retry.

use rusqlite::{OptionalExtension, Transaction};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::extraction::{EDIT_END_REPOINTED, EDIT_REFINED};
use crate::ingest::find_bank;
use crate::models::{Embedder, LlmClient, LlmError, LlmRequest, ModelError, Template};
use crate::store::bank::log_memory_edit;
use crate::store::{Store, StoreError, VectorError, VectorIndex, micros};

/// The prompt's name in [`Template`].
pub const TRANSLATE_TEMPLATE: &str = "translate";

/// The prompt's version. Bump it with any change to its wording.
pub const TRANSLATE_VERSION: u32 = 1;

/// What `translate_memory` did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Translation {
    /// `to`, in `language`, now supersedes `from`.
    Translated {
        from: Uuid,
        to: Uuid,
        language: String,
    },
    /// `memory` is already in `language`: a translation, or a sentence the
    /// LLM returned unchanged. Nothing was written.
    AlreadyInLanguage { memory: Uuid, language: String },
}

#[derive(Debug, thiserror::Error)]
pub enum TranslateError {
    #[error("unknown bank")]
    UnknownBank,

    #[error("no such memory in the bank")]
    UnknownMemory,

    #[error("[llm] language isn't set, so there's nothing to translate into")]
    LanguageUnset,

    #[error("the bank is busy; retry the translation")]
    Busy,

    #[error("the memory has been superseded by {head}; name that one instead")]
    Superseded { head: Uuid },

    #[error("the LLM's reply had no sentence")]
    BadReply,

    /// The bank isn't recorded under a model this daemon carries, or a
    /// re-embed swapped its model while the translation was embedded.
    #[error("the bank's embedding model {model} isn't the one this translation was embedded with")]
    ModelUnavailable { model: String },

    #[error("the daemon has no models loaded")]
    NoModels,

    #[error(transparent)]
    Llm(#[from] LlmError),

    #[error(transparent)]
    Embed(#[from] ModelError),

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for TranslateError {
    fn from(error: rusqlite::Error) -> Self {
        TranslateError::Store(StoreError::Sqlite(error))
    }
}

/// The memory named, as the translation needs it.
pub(crate) struct Named {
    pub(crate) bank_id: i64,
    pub(crate) id: i64,
    pub(crate) uuid: Uuid,
    pub(crate) content: String,
}

/// What [`named`] found.
pub(crate) enum Found {
    /// A head to translate.
    Head(Named),
    /// A translation into the language already.
    Translated(Translation),
}

/// Finds the memory `id` names in `bank` and checks it's a head, and whether
/// it's already a translation into `language`.
pub(crate) fn named(
    store: &Store,
    bank: &str,
    id: &str,
    language: &str,
) -> Result<Found, TranslateError> {
    let conn = store.connection();
    let (bank_id, _) = find_bank(&conn, bank)?.ok_or(TranslateError::UnknownBank)?;
    let uuid = id
        .trim()
        .parse::<Uuid>()
        .map_err(|_| TranslateError::UnknownMemory)?;
    let (memory_id, content): (i64, String) = conn
        .query_row(
            "SELECT id, content FROM memories
             WHERE uuid = ?1 AND bank_id = ?2 AND hidden_at IS NULL",
            (uuid.to_string(), bank_id),
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?
        .ok_or(TranslateError::UnknownMemory)?;
    check_head(&conn, memory_id, uuid)?;
    let translated: bool = conn.query_row(
        "SELECT EXISTS (
           SELECT 1 FROM memories p JOIN edits e ON e.memory_id = p.id
           WHERE p.superseded_by = ?1 AND e.kind = ?2
             AND json_extract(e.details, '$.translated_to') = ?3)",
        (memory_id, EDIT_REFINED, language),
        |row| row.get(0),
    )?;
    if translated {
        return Ok(Found::Translated(Translation::AlreadyInLanguage {
            memory: uuid,
            language: language.to_owned(),
        }));
    }
    Ok(Found::Head(Named {
        bank_id,
        id: memory_id,
        uuid,
        content,
    }))
}

/// Refuses a memory that isn't its chain's head, naming the head.
fn check_head(
    conn: &rusqlite::Connection,
    memory_id: i64,
    uuid: Uuid,
) -> Result<(), TranslateError> {
    // Bounded, should the links ever form a cycle.
    let head: String = conn.query_row(
        "WITH RECURSIVE up(id, uuid, next, depth) AS (
           SELECT id, uuid, superseded_by, 0 FROM memories WHERE id = ?1
           UNION ALL
           SELECT m.id, m.uuid, m.superseded_by, up.depth + 1
           FROM memories m JOIN up ON m.id = up.next
           WHERE up.depth < 10000)
         SELECT uuid FROM up ORDER BY depth DESC LIMIT 1",
        [memory_id],
        |row| row.get(0),
    )?;
    let head = head.parse::<Uuid>().unwrap_or(uuid);
    if head != uuid {
        return Err(TranslateError::Superseded { head });
    }
    Ok(())
}

/// Asks `llm` for `content` in `language`. `None` when it came back
/// unchanged.
pub(crate) fn ask(
    llm: &dyn LlmClient,
    content: &str,
    language: &str,
) -> Result<Option<String>, TranslateError> {
    let reply = llm.complete(&request(content, language))?;
    let Ok(Reply { sentence }) = serde_json::from_value::<Reply>(reply.json) else {
        return Err(TranslateError::BadReply);
    };
    let sentence = sentence.trim();
    if sentence.is_empty() {
        return Err(TranslateError::BadReply);
    }
    Ok((sentence != content.trim()).then(|| sentence.to_owned()))
}

/// Writes `sentence` as the memory's new head, embedded by `embedder`, if
/// the memory is still the head and the bank still recorded under the
/// embedder's model. Returns the new head's id.
pub(crate) fn commit(
    store: &Store,
    named: &Named,
    sentence: &str,
    language: &str,
    embedder: &dyn Embedder,
) -> Result<Uuid, TranslateError> {
    let vector = embedder
        .embed(&[sentence])?
        .pop()
        .ok_or_else(|| ModelError::Inference {
            model: embedder.model_id().to_owned(),
            reason: "no vector for the sentence".to_owned(),
        })?;
    let now = micros(store.now());
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let visible: bool = tx.query_row(
        "SELECT EXISTS (SELECT 1 FROM memories WHERE id = ?1 AND hidden_at IS NULL)",
        [named.id],
        |row| row.get(0),
    )?;
    if !visible {
        return Err(TranslateError::UnknownMemory);
    }
    check_head(&tx, named.id, named.uuid)?;
    let recorded = crate::reembed::recorded_model(&tx, named.bank_id)?;
    if recorded != embedder.model_id() {
        return Err(TranslateError::ModelUnavailable { model: recorded });
    }

    let uuid = store.derived_id(named.uuid, &format!("translate:{language}"));
    tx.execute(
        "INSERT INTO memories (uuid, bank_id, content, kind, significance, owner_significance,
                               chunk_id, source_start, source_end, observed_at,
                               valid_from, valid_from_precision, valid_until,
                               valid_until_precision, until_event, window_confidence,
                               due_at, due_at_precision, volatility, recurrence_text,
                               recurrence_rrule, recurrence_start, recurrence_start_precision,
                               invalidated_at, ended_by, created_at, updated_at)
         SELECT ?2, bank_id, ?3, kind, significance, owner_significance,
                chunk_id, source_start, source_end, observed_at,
                valid_from, valid_from_precision, valid_until,
                valid_until_precision, until_event, window_confidence,
                due_at, due_at_precision, volatility, recurrence_text,
                recurrence_rrule, recurrence_start, recurrence_start_precision,
                invalidated_at, ended_by, ?4, ?4
         FROM memories WHERE id = ?1",
        (named.id, uuid.to_string(), sentence, now),
    )?;
    let head = tx.last_insert_rowid();
    tx.execute(
        "INSERT INTO memory_entities (memory_id, entity_id, surface_form)
         SELECT ?2, entity_id, surface_form FROM memory_entities WHERE memory_id = ?1",
        (named.id, head),
    )?;
    store
        .vectors()
        .upsert(&tx, named.bank_id, head, &vector)
        .map_err(|error| match error {
            VectorError::Sqlite(error) => StoreError::Sqlite(error),
            other => StoreError::Sqlite(rusqlite::Error::ToSqlConversionFailure(Box::new(other))),
        })?;

    tx.execute(
        "UPDATE memories SET superseded_by = ?2, updated_at = ?3 WHERE id = ?1",
        (named.id, head, now),
    )?;
    let details = json!({ "superseded_by": head, "translated_to": language }).to_string();
    log_memory_edit(&tx, store, named.bank_id, EDIT_REFINED, named.id, &details)?;
    repoint_ends(&tx, store, named, head)?;
    // A citation moves to the head, where its accesses are inherited.
    tx.execute(
        "UPDATE OR IGNORE mental_model_cites SET memory_id = ?2 WHERE memory_id = ?1",
        (named.id, head),
    )?;
    tx.execute(
        "DELETE FROM mental_model_cites WHERE memory_id = ?1",
        [named.id],
    )?;
    tx.commit()?;
    Ok(uuid)
}

/// The memories the translated one ended are ended by the translation now,
/// at the same time, since it says the same thing.
fn repoint_ends(
    tx: &Transaction<'_>,
    store: &Store,
    named: &Named,
    head: i64,
) -> Result<(), rusqlite::Error> {
    let mut statement = tx.prepare_cached(
        "SELECT id, valid_until, valid_until_precision FROM memories WHERE ended_by = ?1",
    )?;
    let ended: Vec<(i64, Option<i64>, Option<String>)> = statement
        .query_map([named.id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .collect::<Result<_, _>>()?;
    for (memory, until, precision) in ended {
        tx.execute(
            "UPDATE memories SET ended_by = ?2, updated_at = ?3 WHERE id = ?1",
            (memory, head, micros(store.now())),
        )?;
        let details =
            json!({ "ended_by": head, "valid_until": until, "precision": precision }).to_string();
        log_memory_edit(
            tx,
            store,
            named.bank_id,
            EDIT_END_REPOINTED,
            memory,
            &details,
        )?;
    }
    Ok(())
}

const SYSTEM: &str = "You translate one memory into {language}. A memory is a single sentence \
that makes sense on its own. Keep its meaning exactly: dates, times, numbers and amounts stay as \
they are, and nothing is added, explained or left out. Write names of people, places and things \
the way they're usually written in {language}. If the sentence is already in {language}, return \
it unchanged.";

fn request(content: &str, language: &str) -> LlmRequest {
    LlmRequest {
        template: Template {
            name: TRANSLATE_TEMPLATE.into(),
            version: TRANSLATE_VERSION,
            guidance: None,
        },
        system: SYSTEM.replace("{language}", language),
        user: content.to_owned(),
        schema_name: "translation".into(),
        schema: schema(),
        max_tokens: None,
    }
}

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "sentence": {"type": "string"},
        },
        "required": ["sentence"],
        "additionalProperties": false,
    })
}

#[derive(Deserialize)]
struct Reply {
    sentence: String,
}
