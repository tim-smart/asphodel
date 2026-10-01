//! Mental models (ADR 0007; "Mental models: synthesized documents over
//! memories", TIM-95, as amended by "refreshes follow conversations").
//!
//! A model answers a standing question, such as "who is the user?", with
//! entries: one sentence each, citing the memories it rests on. Only the
//! owner defines models, through the CLI or API, and there are no hand
//! edits. A refresh is one retrieval for the question and one LLM call that
//! returns edits to the entries ([`refresh`]); code applies them and refuses
//! any that cite outside the refresh's input. The memories always win: an
//! entry citing a retracted or ended memory isn't rendered, and one citing a
//! memory that left the input is dropped at the next refresh.
//!
//! A refresh is triggered by a write a model would care about, debounced
//! per bank, at most every [`MIN_REFRESH_INTERVAL`] per model, and once a
//! day besides at `mental_models.sweep_time` bank-local ([`schedule`]). It
//! never runs inside a request. Nothing here ever writes an access, embeds
//! an entry or ingests one (decision 10).

mod refresh;
pub(crate) mod schedule;

use std::collections::BTreeSet;

use jiff::{SignedDuration, Timestamp};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Deserializer, Serialize};
use uuid::Uuid;

use crate::constants::{Significance, Volatility};
use crate::retrieval::RecallError;
use crate::store::strength::memory_kind;
use crate::store::{Store, StoreError, micros, timestamp};
use crate::strength::Kind;

pub(crate) use refresh::{refresh, refresh_input};
pub(crate) use schedule::Schedule;

/// The refresh call's template name and version, which replay's cassette
/// keys include (TIM-96, decision 4).
pub const REFRESH_TEMPLATE: &str = "refresh_model";
pub const REFRESH_VERSION: u32 = 1;

/// The least time between two refreshes of one model, and the wait before
/// a failed refresh is tried again (TIM-95 amendment, decision 1). Fixed
/// in code until the replay harness says otherwise.
pub const MIN_REFRESH_INTERVAL: SignedDuration = SignedDuration::from_mins(30);

/// What `model create` takes (TIM-95, decision 2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub question: String,
    /// The kinds the model takes. Empty means every kind.
    #[serde(default)]
    pub kinds: Vec<Kind>,
    /// A name or alias, resolved through the alias FTS as recall resolves
    /// `entity`. When it matches several entities, the oldest is used.
    #[serde(default)]
    pub entity: Option<String>,
    /// States less durable than this are left out. Other kinds, and states
    /// with no volatility, pass.
    #[serde(default)]
    pub min_volatility: Option<Volatility>,
    pub max_tokens: u32,
    #[serde(default = "enabled")]
    pub enabled: bool,
}

fn enabled() -> bool {
    true
}

/// An owner edit (`model edit`). `None` leaves a field as it is.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelEdit {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<String>,
    /// The kinds the model takes; an empty list means every kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kinds: Option<Vec<Kind>>,
    /// `Some(None)`, sent as `null`, clears it.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present"
    )]
    pub min_volatility: Option<Option<Volatility>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

/// A field that's present, `null` included.
fn present<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

/// A model as `model list` shows it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub id: Uuid,
    pub name: String,
    pub question: String,
    pub kinds: Vec<Kind>,
    /// The entity filter's canonical name.
    pub entity: Option<String>,
    pub min_volatility: Option<Volatility>,
    pub max_tokens: u32,
    pub enabled: bool,
    /// In position order, as stored. An entry citing an ended or retracted
    /// memory stays stored until the next refresh; only rendering drops it
    /// (TIM-95, decision 6).
    pub entries: Vec<Entry>,
    pub last_refreshed_at: Option<Timestamp>,
    pub last_error: Option<FailureKind>,
    pub last_error_at: Option<Timestamp>,
}

/// One sentence citing the memories it rests on. Code assigns `id`, and it
/// survives edits (TIM-95, decision 5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub id: Uuid,
    pub text: String,
    /// In the order the refresh gave them.
    pub cites: Vec<Uuid>,
}

/// What one refresh sends the LLM. Handles are short ids local to the
/// call, `m1`, `m2`, … for memories in score order and `e1`, `e2`, … for
/// entries in position order, as call 1 and call 2 use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefreshInput {
    pub question: String,
    pub max_tokens: u32,
    /// The selection (TIM-95, decision 3).
    pub memories: Vec<InputMemory>,
    /// The entries whose citations are all in the selection. The others
    /// are dropped by the refresh before the call.
    pub entries: Vec<InputEntry>,
    /// The hash of the selection, the question, the filters and
    /// `max_tokens` (TIM-95, decision 7).
    pub fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputMemory {
    pub handle: String,
    pub memory: Uuid,
    pub sentence: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputEntry {
    pub handle: String,
    pub entry: Uuid,
    pub text: String,
    /// Memory handles.
    pub cites: Vec<String>,
}

/// What a refresh did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", content = "detail", rename_all = "snake_case")]
pub enum Outcome {
    /// The fingerprint matched the last completed refresh's: no LLM call.
    Unchanged,
    Applied(Applied),
    /// Nothing changed. The error is recorded on the model, and the
    /// refresh is tried again once [`MIN_REFRESH_INTERVAL`] has passed.
    Failed(FailureKind),
}

/// The reply's operations, as code applied them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Applied {
    pub added: Vec<Uuid>,
    pub edited: Vec<Uuid>,
    pub removed: Vec<Uuid>,
    pub rejected: Vec<Rejected>,
    /// Entries dropped because a memory they cite left the input: fading,
    /// retraction or ending takes an entry out of the model (ADR 0007).
    pub dropped: Vec<Uuid>,
    /// Entries trimmed to fit `max_tokens`, lowest-ranked first.
    pub trimmed: Vec<Uuid>,
}

/// An operation code refused, by its index in the reply.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rejected {
    pub index: usize,
    pub reason: RejectReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectReason {
    /// A citation that isn't a memory handle in this refresh's input.
    CitesOutsideInput,
    /// An add or edit that leaves the entry citing nothing.
    NoCitations,
    /// An edit or remove naming no current entry.
    UnknownEntry,
    /// An add or edit with no text.
    EmptyText,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// The LLM call failed.
    Llm,
    /// The reply didn't parse as operations.
    Malformed,
    /// The retrieval for the question failed, such as the embedder
    /// erroring, so there was nothing to send.
    Retrieval,
}

impl FailureKind {
    fn as_str(self) -> &'static str {
        match self {
            FailureKind::Llm => "llm",
            FailureKind::Malformed => "malformed",
            FailureKind::Retrieval => "retrieval",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "llm" => Some(FailureKind::Llm),
            "malformed" => Some(FailureKind::Malformed),
            "retrieval" => Some(FailureKind::Retrieval),
            _ => None,
        }
    }
}

/// What [`crate::Service::run_refreshes`] did.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Refreshes {
    pub ran: Vec<RefreshRun>,
    /// The earliest debounce deadline, retry or daily sweep still ahead, on
    /// the service's clock.
    pub next_due: Option<Timestamp>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefreshRun {
    pub bank: String,
    pub model: String,
    pub outcome: Outcome,
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("no such bank")]
    UnknownBank,
    #[error("no such model")]
    UnknownModel,
    #[error("a model with that name exists")]
    DuplicateName,
    /// The enabled models' `max_tokens` would sum to `requested`
    /// (TIM-95, decision 8).
    #[error("{requested} tokens is over the {budget}-token budget for mental models")]
    OverBudget { requested: u32, budget: u32 },
    #[error("no entity matches")]
    UnknownEntity,
    #[error("{reason}")]
    Invalid { reason: &'static str },
    /// The refresh's retrieval failed.
    #[error(transparent)]
    Retrieval(RecallError),
    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for ModelError {
    fn from(error: rusqlite::Error) -> Self {
        ModelError::Store(StoreError::Sqlite(error))
    }
}

impl From<RecallError> for ModelError {
    fn from(error: RecallError) -> Self {
        match error {
            RecallError::UnknownBank => ModelError::UnknownBank,
            RecallError::Store(error) => ModelError::Store(error),
            other => ModelError::Retrieval(other),
        }
    }
}

/// A `mental_models` row.
#[derive(Debug, Clone)]
pub(crate) struct ModelRow {
    pub id: i64,
    pub uuid: Uuid,
    pub bank_id: i64,
    pub name: String,
    pub question: String,
    pub kinds: Vec<Kind>,
    pub entity_id: Option<i64>,
    pub min_volatility: Option<Volatility>,
    pub max_tokens: u32,
    pub enabled: bool,
    pub last_fingerprint: Option<String>,
    pub last_refreshed_at: Option<Timestamp>,
    pub refresh_requested_at: Option<Timestamp>,
    pub last_error: Option<FailureKind>,
    pub last_error_at: Option<Timestamp>,
}

impl ModelRow {
    /// Whether the model's own filters let a memory of `kind` and
    /// `volatility` through. The entity filter is checked apart, since it
    /// needs the memory's links.
    pub(crate) fn admits(&self, kind: Kind, volatility: Option<Volatility>) -> bool {
        let kind_passes = self.kinds.is_empty() || self.kinds.contains(&kind);
        let volatility_passes = match (self.min_volatility, kind, volatility) {
            (Some(min), Kind::State, Some(volatility)) => volatility >= min,
            _ => true,
        };
        kind_passes && volatility_passes
    }
}

/// An entry as stored, with its citations' rowids and public ids.
#[derive(Debug, Clone)]
pub(crate) struct StoredEntry {
    pub id: i64,
    pub uuid: Uuid,
    pub text: String,
    pub cites: Vec<(i64, Uuid)>,
}

const MODEL_COLUMNS: &str = "id, uuid, bank_id, name, question, filter_kinds, filter_entity_id,
     filter_min_volatility, max_tokens, enabled, last_fingerprint, last_refreshed_at,
     refresh_requested_at, last_error_kind, last_error_at";

fn model_row(row: &rusqlite::Row<'_>) -> Result<ModelRow, rusqlite::Error> {
    let uuid: String = row.get(1)?;
    let kinds: Option<String> = row.get(5)?;
    let min_volatility: Option<String> = row.get(7)?;
    let max_tokens: i64 = row.get(8)?;
    let error: Option<String> = row.get(13)?;
    Ok(ModelRow {
        id: row.get(0)?,
        uuid: uuid.parse().unwrap_or_default(),
        bank_id: row.get(2)?,
        name: row.get(3)?,
        question: row.get(4)?,
        kinds: kinds
            .as_deref()
            .and_then(|kinds| serde_json::from_str::<Vec<String>>(kinds).ok())
            .unwrap_or_default()
            .iter()
            .filter_map(|kind| memory_kind(kind))
            .collect(),
        entity_id: row.get(6)?,
        min_volatility: min_volatility.as_deref().and_then(volatility),
        max_tokens: u32::try_from(max_tokens).unwrap_or(0),
        enabled: row.get(9)?,
        last_fingerprint: row.get(10)?,
        last_refreshed_at: row.get::<_, Option<i64>>(11)?.map(timestamp),
        refresh_requested_at: row.get::<_, Option<i64>>(12)?.map(timestamp),
        last_error: error.as_deref().and_then(FailureKind::parse),
        last_error_at: row.get::<_, Option<i64>>(14)?.map(timestamp),
    })
}

pub(crate) fn volatility(text: &str) -> Option<Volatility> {
    Volatility::ALL
        .into_iter()
        .find(|level| volatility_str(*level) == text)
}

pub(crate) fn volatility_str(level: Volatility) -> &'static str {
    match level {
        Volatility::Hours => "hours",
        Volatility::Days => "days",
        Volatility::Weeks => "weeks",
        Volatility::Months => "months",
        Volatility::Years => "years",
    }
}

fn kind_str(kind: Kind) -> &'static str {
    match kind {
        Kind::Fact => "fact",
        Kind::Event => "event",
        Kind::State => "state",
        Kind::Task => "task",
        Kind::Recurring => "recurring",
    }
}

fn kinds_json(kinds: &[Kind]) -> Option<String> {
    if kinds.is_empty() {
        return None;
    }
    let mut names: Vec<&str> = kinds.iter().map(|kind| kind_str(*kind)).collect();
    names.sort_unstable();
    names.dedup();
    serde_json::to_string(&names).ok()
}

/// Every model of the bank, oldest first.
pub(crate) fn load_models(
    conn: &Connection,
    bank_id: i64,
) -> Result<Vec<ModelRow>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(&format!(
        "SELECT {MODEL_COLUMNS} FROM mental_models WHERE bank_id = ?1 ORDER BY id"
    ))?;
    statement
        .query_map([bank_id], model_row)?
        .collect::<Result<_, _>>()
}

pub(crate) fn find_model(
    conn: &Connection,
    bank_id: i64,
    name: &str,
) -> Result<Option<ModelRow>, rusqlite::Error> {
    conn.query_row(
        &format!("SELECT {MODEL_COLUMNS} FROM mental_models WHERE bank_id = ?1 AND name = ?2"),
        (bank_id, name.trim()),
        model_row,
    )
    .optional()
}

/// The model's entries in position order, with their citations in the
/// order they were written.
pub(crate) fn load_entries(
    conn: &Connection,
    model_id: i64,
) -> Result<Vec<StoredEntry>, rusqlite::Error> {
    let mut entries = conn.prepare_cached(
        "SELECT id, uuid, text FROM mental_model_entries WHERE model_id = ?1
         ORDER BY position, id",
    )?;
    let mut entries: Vec<StoredEntry> = entries
        .query_map([model_id], |row| {
            let uuid: String = row.get(1)?;
            Ok(StoredEntry {
                id: row.get(0)?,
                uuid: uuid.parse().unwrap_or_default(),
                text: row.get(2)?,
                cites: Vec::new(),
            })
        })?
        .collect::<Result<_, _>>()?;
    let mut cites = conn.prepare_cached(
        "SELECT c.memory_id, m.uuid FROM mental_model_citations c
         JOIN memories m ON m.id = c.memory_id
         WHERE c.entry_id = ?1 ORDER BY c.rowid",
    )?;
    for entry in &mut entries {
        entry.cites = cites
            .query_map([entry.id], |row| {
                let uuid: String = row.get(1)?;
                Ok((row.get(0)?, uuid.parse().unwrap_or_default()))
            })?
            .collect::<Result<_, _>>()?;
    }
    Ok(entries)
}

/// A model as the API shows it.
pub(crate) fn model(conn: &Connection, row: &ModelRow) -> Result<Model, rusqlite::Error> {
    let entity = match row.entity_id {
        Some(entity) => conn
            .query_row("SELECT name FROM entities WHERE id = ?1", [entity], |row| {
                row.get(0)
            })
            .optional()?,
        None => None,
    };
    let entries = load_entries(conn, row.id)?
        .into_iter()
        .map(|entry| Entry {
            id: entry.uuid,
            text: entry.text,
            cites: entry.cites.into_iter().map(|(_, uuid)| uuid).collect(),
        })
        .collect();
    Ok(Model {
        id: row.uuid,
        name: row.name.clone(),
        question: row.question.clone(),
        kinds: row.kinds.clone(),
        entity,
        min_volatility: row.min_volatility,
        max_tokens: row.max_tokens,
        enabled: row.enabled,
        entries,
        last_refreshed_at: row.last_refreshed_at,
        last_error: row.last_error,
        last_error_at: row.last_error_at,
    })
}

/// Checks the enabled models' `max_tokens` against `budget`, with the
/// model `id` (or a new one, when `None`) at `max_tokens` and `enabled`.
/// A disabled model isn't rendered, so it doesn't count until it's enabled.
fn check_budget(
    models: &[ModelRow],
    id: Option<i64>,
    max_tokens: u32,
    enabled: bool,
    budget: u32,
) -> Result<(), ModelError> {
    let others: u32 = models
        .iter()
        .filter(|model| Some(model.id) != id && model.enabled)
        .map(|model| model.max_tokens)
        .fold(0, u32::saturating_add);
    let requested = if enabled {
        others.saturating_add(max_tokens)
    } else {
        others
    };
    if enabled && requested > budget {
        return Err(ModelError::OverBudget { requested, budget });
    }
    Ok(())
}

/// Creates a model (`model create`). Refused past the budget.
pub(crate) fn create(
    store: &Store,
    bank_id: i64,
    spec: &ModelSpec,
    budget: u32,
) -> Result<ModelRow, ModelError> {
    let name = spec.name.trim();
    if name.is_empty() {
        return Err(ModelError::Invalid {
            reason: "a model needs a name",
        });
    }
    if spec.question.trim().is_empty() {
        return Err(ModelError::Invalid {
            reason: "a model needs a question",
        });
    }
    if spec.max_tokens == 0 {
        return Err(ModelError::Invalid {
            reason: "max_tokens must be at least 1",
        });
    }
    let now = micros(store.now());
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    if find_model(&tx, bank_id, name)?.is_some() {
        return Err(ModelError::DuplicateName);
    }
    let models = load_models(&tx, bank_id)?;
    check_budget(&models, None, spec.max_tokens, spec.enabled, budget)?;
    let entity = match spec.entity.as_deref().map(str::trim) {
        Some(entity) if !entity.is_empty() => Some(
            crate::extraction::entities_named(&tx, bank_id, &[entity], &[])?
                .into_iter()
                .min()
                .ok_or(ModelError::UnknownEntity)?,
        ),
        _ => None,
    };
    tx.execute(
        "INSERT INTO mental_models (uuid, bank_id, name, question, filter_kinds, filter_entity_id,
                                    filter_min_volatility, max_tokens, enabled, created_at,
                                    updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?10)",
        (
            store.new_id().to_string(),
            bank_id,
            name,
            spec.question.trim(),
            kinds_json(&spec.kinds),
            entity,
            spec.min_volatility.map(volatility_str),
            spec.max_tokens,
            spec.enabled,
            now,
        ),
    )?;
    let row = find_model(&tx, bank_id, name)?.ok_or(ModelError::UnknownModel)?;
    tx.commit()?;
    Ok(row)
}

/// What an owner edit changed.
pub(crate) struct Edited {
    pub row: ModelRow,
    /// The question, filters or `max_tokens` changed, or the model was
    /// enabled: a triggering write (TIM-95 amendment, decision 1).
    pub triggers: bool,
}

/// Applies an owner edit (`model edit`). Resizing and enabling are checked
/// against the budget.
pub(crate) fn edit(
    store: &Store,
    bank_id: i64,
    name: &str,
    edit: &ModelEdit,
    budget: u32,
) -> Result<Edited, ModelError> {
    let now = micros(store.now());
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let current = find_model(&tx, bank_id, name)?.ok_or(ModelError::UnknownModel)?;
    let question = match &edit.question {
        Some(question) if question.trim().is_empty() => {
            return Err(ModelError::Invalid {
                reason: "a model needs a question",
            });
        }
        Some(question) => question.trim().to_owned(),
        None => current.question.clone(),
    };
    let max_tokens = edit.max_tokens.unwrap_or(current.max_tokens);
    if max_tokens == 0 {
        return Err(ModelError::Invalid {
            reason: "max_tokens must be at least 1",
        });
    }
    let enabled = edit.enabled.unwrap_or(current.enabled);
    let kinds = edit.kinds.clone().unwrap_or_else(|| current.kinds.clone());
    let min_volatility = edit.min_volatility.unwrap_or(current.min_volatility);
    if max_tokens != current.max_tokens || (enabled && !current.enabled) {
        let models = load_models(&tx, bank_id)?;
        check_budget(&models, Some(current.id), max_tokens, enabled, budget)?;
    }
    let kinds_changed = kinds_json(&kinds) != kinds_json(&current.kinds);
    let triggers = enabled
        && (question != current.question
            || kinds_changed
            || min_volatility != current.min_volatility
            || max_tokens != current.max_tokens
            || !current.enabled);
    tx.execute(
        "UPDATE mental_models SET question = ?2, filter_kinds = ?3, filter_min_volatility = ?4,
                                  max_tokens = ?5, enabled = ?6, updated_at = ?7
         WHERE id = ?1",
        (
            current.id,
            &question,
            kinds_json(&kinds),
            min_volatility.map(volatility_str),
            max_tokens,
            enabled,
            now,
        ),
    )?;
    let row = find_model(&tx, bank_id, &current.name)?.ok_or(ModelError::UnknownModel)?;
    tx.commit()?;
    Ok(Edited { row, triggers })
}

/// A memory as the trigger check sees it.
struct Written {
    id: i64,
    kind: Kind,
    volatility: Option<Volatility>,
    /// The owner's setting if there is one, `kept` as the highest level.
    level: Option<Significance>,
    kept: bool,
}

fn written(conn: &Connection, memory_id: i64) -> Result<Option<Written>, rusqlite::Error> {
    conn.query_row(
        "SELECT kind, volatility, significance, owner_significance FROM memories
         WHERE id = ?1 AND invalidated_at IS NULL AND hidden_at IS NULL",
        [memory_id],
        |row| {
            let kind: String = row.get(0)?;
            let volatility_text: Option<String> = row.get(1)?;
            let extracted: String = row.get(2)?;
            let owner: Option<String> = row.get(3)?;
            let level = owner.as_deref().unwrap_or(&extracted).to_owned();
            Ok(Written {
                id: memory_id,
                kind: memory_kind(&kind).unwrap_or(Kind::Fact),
                volatility: volatility_text.as_deref().and_then(volatility),
                level: significance(&level),
                kept: owner.as_deref() == Some("kept"),
            })
        },
    )
    .optional()
}

/// A stored significance level; `None` for `kept`, which is above them all.
fn significance(text: &str) -> Option<Significance> {
    match text {
        "trivial" => Some(Significance::Trivial),
        "minor" => Some(Significance::Minor),
        "notable" => Some(Significance::Notable),
        "major" => Some(Significance::Major),
        "critical" => Some(Significance::Critical),
        _ => None,
    }
}

/// Whether `memory` is linked to `entity`, or to an entity merged into it.
fn linked(conn: &Connection, memory: i64, entity: i64) -> Result<bool, rusqlite::Error> {
    conn.query_row(
        "SELECT EXISTS (SELECT 1 FROM memory_entities me JOIN entities e ON e.id = me.entity_id
                        WHERE me.memory_id = ?1 AND (e.id = ?2 OR e.merged_into = ?2))",
        (memory, entity),
        |row| row.get(0),
    )
}

/// What a batch of writes means for models and the block.
#[derive(Debug, Default)]
pub(crate) struct Effects {
    /// Models to refresh.
    pub triggered: BTreeSet<i64>,
    /// The block's content may have changed.
    pub invalidates: bool,
}

/// Whether a memory of `kind` can be on the agenda.
fn agenda_kind(kind: Kind) -> bool {
    matches!(kind, Kind::Event | Kind::Task | Kind::Recurring)
}

/// The effects of everything written to `bank_id` since the edit log's
/// `watermark`, plus `created`, the memories extraction just wrote. The
/// check is code only, with no LLM call and no retrieval (TIM-95
/// amendment, decision 1):
///
/// - a new memory, or one whose significance went up, at `trigger_level`
///   or above that passes a model's filters triggers that model;
/// - a kept memory triggers every model whose filters it passes;
/// - the retraction, ending or refinement of a memory a model cites, or the
///   reopening of one, triggers that model.
///
/// The block changes when a memory the agenda would list is written,
/// ended, retracted, kept or unkept, which is a check on its kind, and when
/// a cited memory is ended or retracted, so the next fetch can't serve an
/// entry that rendering would drop.
pub(crate) fn effects(
    conn: &Connection,
    bank_id: i64,
    watermark: i64,
    created: &[Uuid],
    trigger_level: Significance,
) -> Result<Effects, rusqlite::Error> {
    let mut effects = Effects::default();
    let models: Vec<ModelRow> = load_models(conn, bank_id)?
        .into_iter()
        .filter(|model| model.enabled)
        .collect();

    let mut statement = conn.prepare_cached(
        "SELECT kind, memory_id FROM edits
         WHERE bank_id = ?1 AND id > ?2 AND memory_id IS NOT NULL ORDER BY id",
    )?;
    let edits: Vec<(String, i64)> = statement
        .query_map((bank_id, watermark), |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<Result<_, _>>()?;

    let mut checked: Vec<(i64, bool)> = Vec::new();
    for uuid in created {
        if let Some(id) = conn
            .query_row(
                "SELECT id FROM memories WHERE uuid = ?1",
                [uuid.to_string()],
                |row| row.get(0),
            )
            .optional()?
        {
            checked.push((id, false));
        }
    }
    let mut cited_changes = Vec::new();
    for (kind, memory) in &edits {
        match kind.as_str() {
            crate::extraction::EDIT_SIGNIFICANCE_RAISED => checked.push((*memory, false)),
            crate::extraction::EDIT_KEPT => checked.push((*memory, true)),
            crate::keep::EDIT_UNKEPT => {
                if let Some(memory) = written(conn, *memory)? {
                    effects.invalidates |= agenda_kind(memory.kind);
                }
            }
            crate::extraction::EDIT_ENDED
            | crate::extraction::EDIT_RETRACTED
            | crate::extraction::EDIT_REFINED
            | crate::extraction::EDIT_END_REPOINTED
            | crate::extraction::EDIT_END_CLEARED => {
                effects.invalidates = true;
                cited_changes.push(*memory);
            }
            _ => {}
        }
    }

    for (memory, kept) in checked {
        let Some(memory) = written(conn, memory)? else {
            continue;
        };
        effects.invalidates |= agenda_kind(memory.kind);
        let significant =
            kept || memory.kept || memory.level.is_some_and(|level| level >= trigger_level);
        if !significant {
            continue;
        }
        for model in &models {
            if !model.admits(memory.kind, memory.volatility) {
                continue;
            }
            if let Some(entity) = model.entity_id
                && !linked(conn, memory.id, entity)?
            {
                continue;
            }
            effects.triggered.insert(model.id);
        }
    }

    if !cited_changes.is_empty() {
        let mut citing = conn.prepare_cached(
            "SELECT DISTINCT e.model_id FROM mental_model_citations c
             JOIN mental_model_entries e ON e.id = c.entry_id
             WHERE c.memory_id = ?1
                OR c.memory_id = (SELECT superseded_by FROM memories WHERE id = ?1)",
        )?;
        for memory in cited_changes {
            for model in citing.query_map([memory], |row| row.get::<_, i64>(0))? {
                let model = model?;
                if models.iter().any(|enabled| enabled.id == model) {
                    effects.triggered.insert(model);
                }
            }
        }
    }
    Ok(effects)
}

/// The edit log's high-water mark for `bank_id`, taken before a write so
/// [`effects`] can find what it did.
pub(crate) fn watermark(conn: &Connection, bank_id: i64) -> Result<i64, rusqlite::Error> {
    conn.query_row(
        "SELECT COALESCE(MAX(id), 0) FROM edits WHERE bank_id = ?1",
        [bank_id],
        |row| row.get(0),
    )
}

/// Marks `models` for a refresh. The first trigger since the last refresh
/// is kept: the debounce caps the wait from it.
pub(crate) fn request(
    conn: &Connection,
    models: &BTreeSet<i64>,
    now: Timestamp,
) -> Result<(), rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "UPDATE mental_models SET refresh_requested_at = COALESCE(refresh_requested_at, ?2)
         WHERE id = ?1 AND enabled = 1",
    )?;
    for model in models {
        statement.execute((model, micros(now)))?;
    }
    Ok(())
}
