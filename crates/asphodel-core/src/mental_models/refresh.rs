//! One refresh: one retrieval for the model's question, then one LLM call
//! that returns edits to its entries.
//!
//! 1. **Selection.** The question runs through the recall pipeline with
//!    injection's weighting over current memories at or above τ that pass
//!    the model's filters: the top `input_budget` by score, plus every
//!    memory the model cites that still qualifies, up to
//!    `input_budget_with_cited`.
//! 2. **Fingerprint.** The selection's memory ids and sentence hashes, the
//!    question, the filters and `max_tokens` are hashed. Unless forced, a
//!    refresh whose hash matches the last completed one's stops here, with
//!    no LLM call.
//! 3. **Drops.** An entry citing a memory that isn't in the selection is
//!    dropped before the call: fading, retraction and ending take entries
//!    out of the model, and the LLM can't keep them.
//! 4. **The call.** The LLM gets the current entries and the selection by
//!    handle, and replies with operations: add, edit or remove. With
//!    nothing selected there's nothing to ask, and no call.
//! 5. **Apply.** Code applies them in order, copies untouched entries byte
//!    for byte, and rejects an operation citing outside the selection,
//!    citing nothing, or naming no current entry. A reply that doesn't
//!    parse changes nothing and records the error.
//! 6. **Trim.** Over `max_tokens`, the lowest-ranked entries go, ranked by
//!    the best score among each one's cited memories.
//!
//! The prompt and reply are never written to disk (ADR 0007). The
//! retrieval writes one `refresh` row to the recall log, and nothing writes
//! an access.

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::OptionalExtension;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::schedule::Schedule;
use super::{
    Applied, FailureKind, InputEntry, InputMemory, ModelError, ModelRow, Outcome, REFRESH_TEMPLATE,
    REFRESH_VERSION, RefreshInput, RejectReason, Rejected, StoredEntry, load_entries,
    volatility_str,
};
use crate::constants::TAU;
use crate::models::{LlmClient, LlmError, LlmRequest, Template};
use crate::retrieval::candidates::Candidate;
use crate::retrieval::{Context, Selected, estimate_tokens, linked_memories};
use crate::store::micros;
use crate::strength::Phase;

/// What a refresh works from.
struct Selection {
    selected: Vec<Selected>,
    /// Memory rowid to handle.
    handles: BTreeMap<i64, String>,
    /// The entries whose citations are all in the selection, in order.
    survivors: Vec<StoredEntry>,
    dropped: Vec<StoredEntry>,
    input: RefreshInput,
}

fn select(cx: &Context<'_>, model: &ModelRow, log: bool) -> Result<Selection, ModelError> {
    let (entries, linked) = {
        let conn = cx.store.connection();
        let entries = load_entries(&conn, model.id)?;
        let linked = match model.entity_id {
            Some(entity) => Some(linked_memories(&conn, model.bank_id, entity)?),
            None => None,
        };
        (entries, linked)
    };
    let mut cited: Vec<i64> = Vec::new();
    for entry in &entries {
        for (memory, _) in &entry.cites {
            if !cited.contains(memory) {
                cited.push(*memory);
            }
        }
    }
    let keep = |candidate: &Candidate| {
        // Only current memories: nothing retracted (clean-up drops those),
        // ended or long past.
        let current = match candidate.phase {
            Phase::LongPast => false,
            Phase::RecentlyPast => candidate.window.valid_until.is_none(),
            Phase::Upcoming | Phase::Current | Phase::Overdue => true,
        };
        candidate.strength >= TAU
            && current
            && model.admits(candidate.window.kind, candidate.volatility)
            && linked
                .as_ref()
                .is_none_or(|linked| linked.contains(&candidate.id))
    };
    let tuning = &cx.tuning.mental_models;
    let selected = crate::retrieval::select(
        cx,
        model.bank_id,
        &model.question,
        &keep,
        &cited,
        tuning.input_budget as usize,
        tuning.input_budget_with_cited as usize,
        log,
    )?;

    let handles: BTreeMap<i64, String> = selected
        .iter()
        .enumerate()
        .map(|(index, item)| (item.candidate.id, format!("m{}", index + 1)))
        .collect();
    let (survivors, dropped): (Vec<StoredEntry>, Vec<StoredEntry>) =
        entries.into_iter().partition(|entry| {
            !entry.cites.is_empty()
                && entry
                    .cites
                    .iter()
                    .all(|(memory, _)| handles.contains_key(memory))
        });
    let input = RefreshInput {
        question: model.question.clone(),
        max_tokens: model.max_tokens,
        memories: selected
            .iter()
            .map(|item| InputMemory {
                handle: handles[&item.candidate.id].clone(),
                memory: item.candidate.uuid,
                sentence: item.candidate.content.clone(),
            })
            .collect(),
        entries: survivors
            .iter()
            .enumerate()
            .map(|(index, entry)| InputEntry {
                handle: format!("e{}", index + 1),
                entry: entry.uuid,
                text: entry.text.clone(),
                cites: entry
                    .cites
                    .iter()
                    .map(|(memory, _)| handles[memory].clone())
                    .collect(),
            })
            .collect(),
        fingerprint: fingerprint(model, &selected),
    };
    Ok(Selection {
        selected,
        handles,
        survivors,
        dropped,
        input,
    })
}

/// The selection's memory ids and sentence hashes, in id order so a change
/// of score alone doesn't count, with the question, the filters and
/// `max_tokens`.
fn fingerprint(model: &ModelRow, selected: &[Selected]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!(
        "question:{}\nkinds:{:?}\nentity:{:?}\nmin_volatility:{:?}\nmax_tokens:{}\n",
        model.question,
        super::kinds_json(&model.kinds),
        model.entity_id,
        model.min_volatility.map(volatility_str),
        model.max_tokens,
    ));
    let mut memories: Vec<(Uuid, String)> = selected
        .iter()
        .map(|item| {
            (
                item.candidate.uuid,
                format!("{:x}", Sha256::digest(item.candidate.content.as_bytes())),
            )
        })
        .collect();
    memories.sort();
    for (uuid, sentence) in memories {
        hasher.update(format!("memory:{uuid}:{sentence}\n"));
    }
    format!("{:x}", hasher.finalize())
}

/// What the next refresh of `model` would send, without writing anything.
pub(crate) fn refresh_input(
    cx: &Context<'_>,
    model: &ModelRow,
) -> Result<RefreshInput, ModelError> {
    Ok(select(cx, model, false)?.input)
}

/// Refreshes `model` with `llm`. Unless `force`, it's skipped when the
/// fingerprint matches the last completed refresh's. A failed retrieval,
/// an LLM failure or a malformed reply is `Ok(Outcome::Failed)`, recorded
/// on the model so the schedule waits before trying again. An LLM held by
/// a limit is `Ok(Outcome::Held)`: nothing is recorded as failed, and the
/// refresh stays requested until the hold lifts.
pub(crate) fn refresh(
    cx: &Context<'_>,
    schedule: &Schedule,
    model: &ModelRow,
    llm: &dyn LlmClient,
    force: bool,
) -> Result<Outcome, ModelError> {
    // Taken before the inputs are selected: a request made after this may
    // have written something the selection doesn't hold.
    let started = Started {
        schedule,
        model: model.id,
        generation: schedule.generation(model.id),
    };
    // A failed retrieval is a failed refresh like any other: recorded, and
    // tried again once the interval has passed, never on every timer pass.
    let selection = match select(cx, model, true) {
        Ok(selection) => selection,
        Err(ModelError::Retrieval(error)) => {
            tracing::warn!(model = %model.uuid, %error, "a mental model refresh's retrieval failed");
            return failed(cx, model, FailureKind::Retrieval);
        }
        Err(error) => return Err(error),
    };
    if !force && model.last_fingerprint.as_deref() == Some(selection.input.fingerprint.as_str()) {
        let conn = cx.store.connection();
        conn.execute(
            "UPDATE mental_models SET last_error_kind = NULL, last_error_at = NULL,
                    refresh_requested_at = CASE WHEN ?2 THEN NULL ELSE refresh_requested_at END
             WHERE id = ?1",
            (model.id, started.unchanged()),
        )?;
        return Ok(Outcome::Unchanged);
    }

    if selection.input.memories.is_empty() {
        // Nothing qualifies, so every entry has already been dropped and
        // there's nothing to ask: an empty model costs no LLM call.
        let applied = Applied {
            dropped: selection.dropped.iter().map(|entry| entry.uuid).collect(),
            ..Applied::default()
        };
        write(cx, model, &selection, &[], started)?;
        return Ok(Outcome::Applied(applied));
    }
    let request = request(&selection.input);
    let identities: Vec<(String, Uuid)> = selection
        .input
        .memories
        .iter()
        .map(|memory| (memory.handle.clone(), memory.memory))
        .chain(
            selection
                .input
                .entries
                .iter()
                .map(|entry| (entry.handle.clone(), entry.entry)),
        )
        .collect();
    let reply = match llm.complete_identified(&request, &identities) {
        Ok(response) => response.json,
        Err(error) => {
            if let Some(until) = held_until(&error, cx.store.now()) {
                tracing::info!(model = %model.uuid, %until, "a mental model refresh waits for the LLM's hold");
                return held(cx, schedule, model, until);
            }
            tracing::warn!(model = %model.uuid, %error, "a mental model refresh failed");
            return failed(cx, model, FailureKind::Llm);
        }
    };
    let Ok(reply) = serde_json::from_value::<Reply>(reply) else {
        tracing::warn!(model = %model.uuid, "a mental model refresh got a malformed reply");
        return failed(cx, model, FailureKind::Malformed);
    };

    let mut applied = Applied {
        dropped: selection.dropped.iter().map(|entry| entry.uuid).collect(),
        ..Applied::default()
    };
    let mut drafts = apply(cx, &selection, reply, &mut applied);
    trim(
        &selection.selected,
        model.max_tokens,
        &mut drafts,
        &mut applied,
    );
    write(cx, model, &selection, &drafts, started)?;
    Ok(Outcome::Applied(applied))
}

/// The model's generation when a refresh started.
#[derive(Clone, Copy)]
struct Started<'a> {
    schedule: &'a Schedule,
    model: i64,
    generation: u64,
}

impl Started<'_> {
    /// Whether no request has been made since the refresh started, so it
    /// may clear the one it ran for. Called while holding the store's
    /// connection, as every request is made.
    fn unchanged(&self) -> bool {
        self.schedule.generation(self.model) == self.generation
    }
}

/// When a limit the LLM reported lifts, or `None` for any other error.
fn held_until(error: &LlmError, now: jiff::Timestamp) -> Option<jiff::Timestamp> {
    match error {
        LlmError::UsageLimited { resets_at } => Some(*resets_at),
        LlmError::RateLimited { retry_after } => Some(
            jiff::SignedDuration::try_from(*retry_after)
                .ok()
                .and_then(|wait| now.checked_add(wait).ok())
                .unwrap_or(jiff::Timestamp::MAX),
        ),
        _ => None,
    }
}

/// Keeps the refresh requested and due again at `until`, without recording
/// a failure: the call never reached the LLM.
fn held(
    cx: &Context<'_>,
    schedule: &Schedule,
    model: &ModelRow,
    until: jiff::Timestamp,
) -> Result<Outcome, ModelError> {
    let now = micros(cx.store.now());
    cx.store.connection().execute(
        "UPDATE mental_models SET refresh_requested_at = COALESCE(refresh_requested_at, ?2)
         WHERE id = ?1",
        (model.id, now),
    )?;
    schedule.held(model.id, until);
    Ok(Outcome::Held { until })
}

fn failed(cx: &Context<'_>, model: &ModelRow, kind: FailureKind) -> Result<Outcome, ModelError> {
    let now = micros(cx.store.now());
    cx.store.connection().execute(
        "UPDATE mental_models SET last_error_kind = ?2, last_error_at = ?3,
                refresh_requested_at = COALESCE(refresh_requested_at, ?3)
         WHERE id = ?1",
        (model.id, kind.as_str(), now),
    )?;
    Ok(Outcome::Failed(kind))
}

const SYSTEM: &str = "You keep a mental model: a short list of entries that answers a standing \
question about the user, built only from their memories. Each entry is one plain sentence of \
about 25 words at most, and cites by handle every memory it rests on. Say nothing the cited \
memories don't support, and use names, not pronouns.

Reply with operations on the current entries:
- add: a new entry, with its text and cites;
- edit: an entry whose wording or citations should change, by its handle, with its new text \
and its full list of cites;
- remove: an entry that no longer answers the question or is no longer supported.

Leave an entry that's still right alone: don't edit it to reword it. Cite only the memory handles \
listed (m1, m2, ...); every entry must cite at least one. Keep all the entries together within the \
token budget, about four characters to a token, and put what matters most first. Reply with no \
operations when nothing needs to change.";

fn request(input: &RefreshInput) -> LlmRequest {
    let mut user = format!(
        "Question: {}\nToken budget: {}\n\nCurrent entries:\n",
        input.question, input.max_tokens
    );
    if input.entries.is_empty() {
        user.push_str("(none)\n");
    }
    for entry in &input.entries {
        user.push_str(&format!(
            "{}: {} [cites {}]\n",
            entry.handle,
            entry.text,
            entry.cites.join(", ")
        ));
    }
    user.push_str("\nMemories:\n");
    if input.memories.is_empty() {
        user.push_str("(none)\n");
    }
    for memory in &input.memories {
        user.push_str(&format!("{}: {}\n", memory.handle, memory.sentence));
    }
    LlmRequest {
        template: Template {
            name: REFRESH_TEMPLATE.into(),
            version: REFRESH_VERSION,
        },
        system: SYSTEM.to_owned(),
        user,
        schema_name: "mental_model_edits".into(),
        schema: schema(),
        max_tokens: None,
    }
}

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["add", "edit", "remove"]},
                        "entry": {"type": ["string", "null"]},
                        "text": {"type": ["string", "null"]},
                        "cites": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["op", "entry", "text", "cites"],
                    "additionalProperties": false,
                },
            },
        },
        "required": ["operations"],
        "additionalProperties": false,
    })
}

#[derive(Deserialize)]
struct Reply {
    operations: Vec<Operation>,
}

#[derive(Deserialize)]
struct Operation {
    op: Op,
    #[serde(default)]
    entry: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    cites: Vec<String>,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Op {
    Add,
    Edit,
    Remove,
}

/// An entry as the refresh leaves it.
struct Draft {
    /// The stored row, for an entry that was there before.
    stored: Option<i64>,
    uuid: Uuid,
    text: String,
    cites: Vec<i64>,
    edited: bool,
}

fn apply(
    cx: &Context<'_>,
    selection: &Selection,
    reply: Reply,
    applied: &mut Applied,
) -> Vec<Draft> {
    let memories: BTreeMap<&str, i64> = selection
        .handles
        .iter()
        .map(|(memory, handle)| (handle.as_str(), *memory))
        .collect();
    let mut drafts: Vec<Option<Draft>> = selection
        .survivors
        .iter()
        .map(|entry| {
            Some(Draft {
                stored: Some(entry.id),
                uuid: entry.uuid,
                text: entry.text.clone(),
                cites: entry.cites.iter().map(|(memory, _)| *memory).collect(),
                edited: false,
            })
        })
        .collect();
    let entry_handles: BTreeMap<String, usize> = selection
        .input
        .entries
        .iter()
        .enumerate()
        .map(|(index, entry)| (entry.handle.clone(), index))
        .collect();

    let cites = |handles: &[String]| -> Result<Vec<i64>, RejectReason> {
        if handles.is_empty() {
            return Err(RejectReason::NoCitations);
        }
        let mut ids = Vec::new();
        for handle in handles {
            let id = memories
                .get(handle.trim())
                .ok_or(RejectReason::CitesOutsideInput)?;
            if !ids.contains(id) {
                ids.push(*id);
            }
        }
        Ok(ids)
    };
    let text = |text: &Option<String>| -> Result<String, RejectReason> {
        match text.as_deref().map(str::trim) {
            Some(text) if !text.is_empty() => Ok(text.to_owned()),
            _ => Err(RejectReason::EmptyText),
        }
    };

    for (index, operation) in reply.operations.into_iter().enumerate() {
        let outcome: Result<(), RejectReason> = (|| {
            match operation.op {
                Op::Add => {
                    let text = text(&operation.text)?;
                    let cites = cites(&operation.cites)?;
                    let uuid = cx.store.new_id();
                    drafts.push(Some(Draft {
                        stored: None,
                        uuid,
                        text,
                        cites,
                        edited: false,
                    }));
                    applied.added.push(uuid);
                }
                Op::Edit => {
                    let position = operation
                        .entry
                        .as_deref()
                        .and_then(|handle| entry_handles.get(handle.trim()))
                        .copied()
                        .filter(|position| drafts[*position].is_some())
                        .ok_or(RejectReason::UnknownEntry)?;
                    let text = text(&operation.text)?;
                    let cites = cites(&operation.cites)?;
                    let draft = drafts[position].as_mut().expect("checked above");
                    draft.text = text;
                    draft.cites = cites;
                    draft.edited = true;
                    if !applied.edited.contains(&draft.uuid) {
                        applied.edited.push(draft.uuid);
                    }
                }
                Op::Remove => {
                    let position = operation
                        .entry
                        .as_deref()
                        .and_then(|handle| entry_handles.get(handle.trim()))
                        .copied()
                        .filter(|position| drafts[*position].is_some())
                        .ok_or(RejectReason::UnknownEntry)?;
                    let draft = drafts[position].take().expect("checked above");
                    applied.edited.retain(|uuid| *uuid != draft.uuid);
                    applied.removed.push(draft.uuid);
                }
            }
            Ok(())
        })();
        if let Err(reason) = outcome {
            applied.rejected.push(Rejected { index, reason });
        }
    }
    drafts.into_iter().flatten().collect()
}

/// Drops the lowest-ranked entries until the rest fit `max_tokens`. An
/// entry ranks by the best score among its cited memories; of two equal,
/// the later one goes.
fn trim(selected: &[Selected], max_tokens: u32, drafts: &mut Vec<Draft>, applied: &mut Applied) {
    let scores: BTreeMap<i64, f64> = selected
        .iter()
        .map(|item| (item.candidate.id, item.score))
        .collect();
    let rank = |draft: &Draft| {
        draft
            .cites
            .iter()
            .filter_map(|memory| scores.get(memory))
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let tokens =
        |drafts: &[Draft]| -> usize { drafts.iter().map(|d| estimate_tokens(&d.text)).sum() };
    while !drafts.is_empty() && tokens(drafts) > max_tokens as usize {
        let mut lowest = 0;
        for (index, draft) in drafts.iter().enumerate() {
            if rank(draft) <= rank(&drafts[lowest]) {
                lowest = index;
            }
        }
        let draft = drafts.remove(lowest);
        applied.edited.retain(|uuid| *uuid != draft.uuid);
        applied.trimmed.push(draft.uuid);
    }
}

/// Whether every memory `draft` cites is still there and visible, and the
/// entry it edits, if any, still exists.
fn still_standing(tx: &rusqlite::Transaction<'_>, draft: &Draft) -> Result<bool, rusqlite::Error> {
    for memory in &draft.cites {
        let visible: Option<bool> = tx
            .query_row(
                "SELECT hidden_at IS NULL FROM memories WHERE id = ?1",
                [memory],
                |row| row.get(0),
            )
            .optional()?;
        if visible != Some(true) {
            return Ok(false);
        }
    }
    match draft.stored {
        Some(entry) => Ok(tx
            .query_row(
                "SELECT 1 FROM mental_model_entries WHERE id = ?1",
                [entry],
                |_| Ok(()),
            )
            .optional()?
            .is_some()),
        None => Ok(true),
    }
}

fn write(
    cx: &Context<'_>,
    model: &ModelRow,
    selection: &Selection,
    drafts: &[Draft],
    started: Started<'_>,
) -> Result<(), ModelError> {
    let now = micros(cx.store.now());
    let mut conn = cx.store.connection();
    let tx = conn.transaction()?;
    // The LLM answered from a selection made before its call. A memory
    // forgotten or erased since can't be cited again, and an entry a forget
    // dropped meanwhile can't be edited back: such a
    // draft goes, as one failing the citation check does.
    let drafts: Vec<&Draft> = {
        let mut standing = Vec::with_capacity(drafts.len());
        for draft in drafts {
            if still_standing(&tx, draft)? {
                standing.push(draft);
            }
        }
        standing
    };
    let kept: BTreeSet<i64> = drafts.iter().filter_map(|draft| draft.stored).collect();
    for entry in selection.survivors.iter().chain(&selection.dropped) {
        if !kept.contains(&entry.id) {
            tx.execute("DELETE FROM mental_model_entries WHERE id = ?1", [entry.id])?;
        }
    }
    for (position, draft) in drafts.iter().enumerate() {
        let entry = match draft.stored {
            Some(entry) => {
                tx.execute(
                    "UPDATE mental_model_entries SET position = ?2 WHERE id = ?1",
                    (entry, position as i64),
                )?;
                if !draft.edited {
                    continue;
                }
                tx.execute(
                    "UPDATE mental_model_entries SET text = ?2, updated_at = ?3 WHERE id = ?1",
                    (entry, &draft.text, now),
                )?;
                tx.execute(
                    "DELETE FROM mental_model_citations WHERE entry_id = ?1",
                    [entry],
                )?;
                entry
            }
            None => {
                tx.execute(
                    "INSERT INTO mental_model_entries (uuid, model_id, position, text, created_at,
                                                       updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
                    (
                        draft.uuid.to_string(),
                        model.id,
                        position as i64,
                        &draft.text,
                        now,
                    ),
                )?;
                tx.last_insert_rowid()
            }
        };
        for memory in &draft.cites {
            tx.execute(
                "INSERT OR IGNORE INTO mental_model_citations (entry_id, memory_id) VALUES (?1, ?2)",
                (entry, memory),
            )?;
        }
    }
    tx.execute(
        "UPDATE mental_models SET last_fingerprint = ?2, last_refreshed_at = ?3,
                last_error_kind = NULL, last_error_at = NULL,
                refresh_requested_at = CASE WHEN ?4 THEN NULL ELSE refresh_requested_at END,
                updated_at = ?3
         WHERE id = ?1",
        (
            model.id,
            &selection.input.fingerprint,
            now,
            started.unchanged(),
        ),
    )?;
    tx.commit()?;
    Ok(())
}
