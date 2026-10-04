//! One refresh: a plan, one retrieval per facet, then one LLM call that
//! writes the whole summary.
//!
//! 1. **Plan.** The facets of the model's question, each a heading and a
//!    recall query. The seeded profile's question has them built in
//!    ([`PROFILE_FACETS`](crate::store::bank::PROFILE_FACETS)). Any other
//!    question is planned by one LLM call holding only the question and the
//!    language, so it replays by key. The plan is stored on the model in
//!    its own write before anything is recalled, so a failed or held write
//!    doesn't pay for it again, and it's made again only once the question
//!    changes. Whichever plan it is, a refresh recalls by its first
//!    `max_facets` facets, so the limit holds for plans made before it was
//!    lowered too. A bank whose embedder isn't served fails the refresh
//!    before the plan call, as its retrieval would.
//! 2. **Selection.** Each facet's query runs through the recall pipeline
//!    with injection's weighting over current memories at or above τ that
//!    pass the model's filters, taking its best `facet_budget`. Reranker
//!    logits for different queries aren't comparable, so the input is
//!    filled by taking each facet's next best in turn, skipping memories
//!    already in, up to `input_budget`; then every memory the model cites
//!    that still qualifies, best first, up to `input_budget_with_cited`.
//!    The facets share one reranker deadline: a facet that misses it scores
//!    on strength alone, and the refresh goes on.
//! 3. **Fingerprint.** The selection's memory ids and sentence hashes, the
//!    question, the plan, the filters and `max_tokens` are hashed. Unless
//!    forced, a refresh whose hash matches the last completed one's stops
//!    here, with no write call.
//! 4. **Drops.** An entry citing a memory that isn't in the selection is
//!    dropped before the call: fading, retraction and ending take entries
//!    out of the model, and the LLM can't keep them.
//! 5. **The write.** The LLM gets the question, the facets, the selection
//!    by handle under the facet that first found each memory, and the
//!    remaining entries as the previous summary. It replies with the whole
//!    summary: sections of sentences, each citing handles. With nothing
//!    selected there's nothing to ask, and no call.
//! 6. **Apply.** Code refuses a sentence citing outside the selection,
//!    citing nothing, or with no text. A sentence whose text matches an
//!    entry keeps that entry's id, and one citing exactly what an entry
//!    cited keeps its id under new words, but citations and section always
//!    come from the write. Every other sentence is new, and an entry the
//!    write leaves out is removed. A reply that doesn't parse changes
//!    nothing and records the error.
//! 7. **Trim.** Over `max_tokens`, headings included, the lowest-ranked
//!    entries go, ranked by the best score among each one's cited memories.
//!
//! The prompts and replies are never written to disk, so forget has nothing
//! to scrub there. Each facet's retrieval writes one `refresh` row to the
//! recall log, with its query, and nothing writes an access.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use rusqlite::OptionalExtension;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::schedule::Schedule;
use super::{
    Applied, Facet, FailureKind, InputEntry, InputMemory, ModelError, ModelRow, Outcome,
    PLAN_TEMPLATE, PLAN_VERSION, RefreshInput, RejectReason, Rejected, StoredEntry, StoredPlan,
    WRITE_TEMPLATE, WRITE_VERSION, load_entries, volatility_str,
};
use crate::constants::TAU;
use crate::models::{LlmClient, LlmError, LlmRequest, Template};
use crate::retrieval::candidates::Candidate;
use crate::retrieval::{
    Context, REFRESH_RERANK_DEADLINE, Selected, estimate_tokens, linked_memories,
};
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

fn select(
    cx: &Context<'_>,
    model: &ModelRow,
    facets: Vec<Facet>,
    log: bool,
) -> Result<Selection, ModelError> {
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
            && model.admits(
                candidate.window.kind,
                candidate.volatility,
                candidate.rrule.as_deref(),
            )
            && linked
                .as_ref()
                .is_none_or(|linked| linked.contains(&candidate.id))
    };
    let tuning = &cx.tuning.mental_models;
    let facet_budget = tuning.facet_budget as usize;
    // Each facet's best, and past them the cited memories it scored.
    let queries: Vec<&str> = facets.iter().map(|facet| facet.query.as_str()).collect();
    let found = crate::retrieval::select(
        cx,
        model.bank_id,
        &queries,
        &keep,
        &cited,
        facet_budget,
        facet_budget + cited.len(),
        log,
        Instant::now() + REFRESH_RERANK_DEADLINE,
    )?;
    let mut ranked: Vec<Vec<Selected>> = Vec::with_capacity(facets.len());
    let mut extras: Vec<(usize, Selected)> = Vec::new();
    for (index, mut found) in found.into_iter().enumerate() {
        extras.extend(
            found
                .split_off(found.len().min(facet_budget))
                .into_iter()
                .map(|item| (index, item)),
        );
        ranked.push(found);
    }
    let (selected, found_by) = interleave(
        ranked,
        extras,
        &cited,
        tuning.input_budget as usize,
        tuning.input_budget_with_cited as usize,
    );

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
        language: cx.tuning.llm.language.clone(),
        max_tokens: model.max_tokens,
        memories: selected
            .iter()
            .zip(&found_by)
            .map(|(item, facet)| InputMemory {
                handle: handles[&item.candidate.id].clone(),
                memory: item.candidate.uuid,
                sentence: item.candidate.content.clone(),
                facet: facets
                    .get(*facet)
                    .map(|facet| facet.heading.clone())
                    .unwrap_or_default(),
            })
            .collect(),
        entries: survivors
            .iter()
            .map(|entry| InputEntry {
                entry: entry.uuid,
                section: entry.section.clone(),
                text: entry.text.clone(),
                cites: entry
                    .cites
                    .iter()
                    .map(|(memory, _)| handles[memory].clone())
                    .collect(),
            })
            .collect(),
        fingerprint: fingerprint(model, &facets, &selected),
        facets,
    };
    Ok(Selection {
        selected,
        handles,
        survivors,
        dropped,
        input,
    })
}

/// The selection from each facet's best, in rank order: the first of each
/// facet in turn, then the second of each, and so on, skipping a memory
/// already in, until there are `budget`. Then the memories in `cited` that
/// any facet scored, best first, until there are `with_cited`. A memory
/// keeps the best score any facet gave it, which trimming ranks by. Returns
/// the selection with the index of the facet that first found each memory,
/// or, for one the cited fill took, the facet that scored it best.
/// `extras` are the cited memories each facet scored past its best.
fn interleave(
    ranked: Vec<Vec<Selected>>,
    extras: Vec<(usize, Selected)>,
    cited: &[i64],
    budget: usize,
    with_cited: usize,
) -> (Vec<Selected>, Vec<usize>) {
    let mut selected: Vec<Selected> = Vec::new();
    let mut found_by: Vec<usize> = Vec::new();
    let mut position: BTreeMap<i64, usize> = BTreeMap::new();
    let mut rest: Vec<(usize, Selected)> = extras;
    let longest = ranked.iter().map(Vec::len).max().unwrap_or(0);
    let mut columns: Vec<std::vec::IntoIter<Selected>> =
        ranked.into_iter().map(Vec::into_iter).collect();
    for _ in 0..longest {
        for (facet, column) in columns.iter_mut().enumerate() {
            let Some(item) = column.next() else {
                continue;
            };
            if let Some(&at) = position.get(&item.candidate.id) {
                if item.score > selected[at].score {
                    selected[at].score = item.score;
                }
            } else if selected.len() < budget {
                position.insert(item.candidate.id, selected.len());
                selected.push(item);
                found_by.push(facet);
            } else {
                rest.push((facet, item));
            }
        }
    }

    // A memory left out above keeps its best score for the cited fill.
    let mut best: BTreeMap<i64, (usize, Selected)> = BTreeMap::new();
    for (facet, item) in rest {
        if let Some(&at) = position.get(&item.candidate.id) {
            if item.score > selected[at].score {
                selected[at].score = item.score;
            }
            continue;
        }
        if !cited.contains(&item.candidate.id) {
            continue;
        }
        match best.get(&item.candidate.id) {
            Some((_, kept)) if kept.score >= item.score => {}
            _ => {
                best.insert(item.candidate.id, (facet, item));
            }
        }
    }
    let mut fill: Vec<(usize, Selected)> = best.into_values().collect();
    fill.sort_by(|(_, left), (_, right)| right.score.total_cmp(&left.score));
    for (facet, item) in fill {
        if selected.len() >= with_cited {
            break;
        }
        selected.push(item);
        found_by.push(facet);
    }
    (selected, found_by)
}

/// The selection's memory ids and sentence hashes, in id order so a change
/// of score alone doesn't count, with the question, the plan, the filters
/// and `max_tokens`.
fn fingerprint(model: &ModelRow, facets: &[Facet], selected: &[Selected]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!(
        "question:{}\nkinds:{:?}\nentity:{:?}\nmin_volatility:{:?}\nmax_tokens:{}\n",
        model.question,
        super::kinds_json(&model.kinds),
        model.entity_id,
        model.min_volatility.map(volatility_str),
        model.max_tokens,
    ));
    for facet in facets {
        hasher.update(format!("facet:{}\t{}\n", facet.heading, facet.query));
    }
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

/// What the next refresh of `model` would send, without writing anything
/// or calling the LLM. A question that still needs planning is shown
/// recalled as one facet, its own.
pub(crate) fn refresh_input(
    cx: &Context<'_>,
    model: &ModelRow,
) -> Result<RefreshInput, ModelError> {
    let facets = model
        .facets(cx.tuning.mental_models.max_facets as usize)
        .unwrap_or_else(|| {
            vec![Facet {
                heading: model.name.clone(),
                query: model.question.clone(),
            }]
        });
    Ok(select(cx, model, facets, false)?.input)
}

/// Refreshes `model` with `llm`. Unless `force`, the write is skipped when
/// the fingerprint matches the last completed refresh's. A failed
/// retrieval, an LLM failure or a malformed reply is `Ok(Outcome::Failed)`,
/// recorded on the model so the schedule waits before trying again. An LLM
/// held by a limit is `Ok(Outcome::Held)`: nothing is recorded as failed,
/// and the refresh stays requested until the hold lifts.
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
    let facets = match model.facets(cx.tuning.mental_models.max_facets as usize) {
        Some(facets) => facets,
        None => {
            // A plan is no use if nothing can be recalled for it.
            match cx.check_serving(model.bank_id).map_err(ModelError::from) {
                Ok(()) => {}
                Err(ModelError::Retrieval(error)) => {
                    tracing::warn!(model = %model.uuid, %error, "a mental model refresh's retrieval failed");
                    return failed(cx, model, FailureKind::Retrieval);
                }
                Err(error) => return Err(error),
            }
            match plan(cx, schedule, model, llm)? {
                Planned::Facets(facets) => facets,
                Planned::Stopped(outcome) => return Ok(outcome),
            }
        }
    };
    // A failed retrieval is a failed refresh like any other: recorded, and
    // tried again once the interval has passed, never on every timer pass.
    let selection = match select(cx, model, facets, true) {
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
        // there's nothing to ask: an empty model costs no write call.
        let applied = Applied {
            dropped: selection.dropped.iter().map(|entry| entry.uuid).collect(),
            ..Applied::default()
        };
        write(cx, model, &selection, &[], started)?;
        return Ok(Outcome::Applied(applied));
    }
    let request = write_request(&selection.input);
    let identities: Vec<(String, Uuid)> = selection
        .input
        .memories
        .iter()
        .map(|memory| (memory.handle.clone(), memory.memory))
        .collect();
    let reply = match llm.complete_identified(&request, &identities) {
        Ok(response) => response.json,
        Err(error) => return llm_failed(cx, schedule, model, &error),
    };
    let Ok(reply) = serde_json::from_value::<Written>(reply) else {
        tracing::warn!(model = %model.uuid, "a mental model refresh got a malformed summary");
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

/// A plan or write call that failed: held when a limit says when to come
/// back, failed otherwise.
fn llm_failed(
    cx: &Context<'_>,
    schedule: &Schedule,
    model: &ModelRow,
    error: &LlmError,
) -> Result<Outcome, ModelError> {
    if let Some(until) = held_until(error, cx.store.now()) {
        tracing::info!(model = %model.uuid, %until, "a mental model refresh waits for the LLM's hold");
        return held(cx, schedule, model, until);
    }
    tracing::warn!(model = %model.uuid, %error, "a mental model refresh failed");
    failed(cx, model, FailureKind::Llm)
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

// The plan.

/// What planning came to: the facets, or the outcome the refresh stops at.
enum Planned {
    Facets(Vec<Facet>),
    Stopped(Outcome),
}

const PLAN_SYSTEM: &str = "You plan how to answer a standing question about the user from their \
memories. Split the question into the separate things it asks about, its facets, so that each can \
be looked up on its own. For each facet give a heading of a few words, which the answer's section \
for it will be written under, and a query: one plain sentence describing the memories that would \
answer that part, for a search over the user's memories. Give one facet per part of the question, \
in the order the question asks them, and no more than six. {language_rule}";

fn plan_system(language: Option<&str>) -> String {
    let rule = match language {
        None => "Write the headings and queries in the language of the question.".to_owned(),
        Some(language) => format!(
            "Write the headings and queries in {}, translating the question if it's in \
             another language.",
            language.trim()
        ),
    };
    PLAN_SYSTEM.replace("{language_rule}", &rule)
}

/// The plan request: the question and the language, and nothing that
/// differs between runs or stores, so the cassette answers it by key.
fn plan_request(question: &str, language: Option<&str>) -> LlmRequest {
    LlmRequest {
        template: Template {
            name: PLAN_TEMPLATE.into(),
            version: PLAN_VERSION,
            guidance: None,
        },
        system: plan_system(language),
        user: format!("Question: {question}\n"),
        schema_name: "mental_model_plan".into(),
        schema: json!({
            "type": "object",
            "properties": {
                "facets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "query": {"type": "string"},
                        },
                        "required": ["heading", "query"],
                        "additionalProperties": false,
                    },
                },
            },
            "required": ["facets"],
            "additionalProperties": false,
        }),
        max_tokens: None,
    }
}

#[derive(Deserialize)]
struct PlanReply {
    facets: Vec<Facet>,
}

/// Plans `model`'s question and stores the plan on it.
fn plan(
    cx: &Context<'_>,
    schedule: &Schedule,
    model: &ModelRow,
    llm: &dyn LlmClient,
) -> Result<Planned, ModelError> {
    let request = plan_request(&model.question, cx.tuning.llm.language.as_deref());
    let reply = match llm.complete(&request) {
        Ok(response) => response.json,
        Err(error) => return Ok(Planned::Stopped(llm_failed(cx, schedule, model, &error)?)),
    };
    // A facet needs a heading and a query, and a heading names one section.
    let mut facets: Vec<Facet> = Vec::new();
    let replied = serde_json::from_value::<PlanReply>(reply)
        .map(|reply| reply.facets)
        .unwrap_or_default();
    for facet in replied {
        let heading = facet.heading.trim();
        let query = facet.query.trim();
        if heading.is_empty()
            || query.is_empty()
            || facets.iter().any(|kept| kept.heading == heading)
        {
            continue;
        }
        facets.push(Facet {
            heading: heading.to_owned(),
            query: query.to_owned(),
        });
    }
    facets.truncate(cx.tuning.mental_models.max_facets as usize);
    if facets.is_empty() {
        tracing::warn!(model = %model.uuid, "a mental model refresh got a malformed plan");
        return Ok(Planned::Stopped(failed(cx, model, FailureKind::Malformed)?));
    }
    let plan = StoredPlan {
        question: model.question.clone(),
        facets,
    };
    cx.store.connection().execute(
        "UPDATE mental_models SET plan = ?2 WHERE id = ?1",
        (
            model.id,
            serde_json::to_string(&plan).expect("a plan serialises"),
        ),
    )?;
    Ok(Planned::Facets(plan.facets))
}

// The write.

const WRITE_SYSTEM: &str = "You write a mental model: a short summary that answers a standing \
question about the user, built only from their memories. Write it in sections, one for each facet \
listed, in that order and under the facet's heading, and leave out a section the memories say \
nothing about. A section is a few plain sentences that read as a paragraph. Each sentence is an \
entry of about 25 words at most, says nothing the memories it cites don't support, and cites by \
handle every memory it rests on. Use names, not pronouns. {language_rule}

The previous summary is there to keep the wording steady. Restate a sentence word for word when \
the memories listed still support it, reword it when they say something new, and leave it out \
when they no longer support it or it no longer answers the question: anything you leave out is \
removed. Cite only the memory handles listed (m1, m2, ...); every sentence must cite at least one. \
Keep the whole summary, headings included, within the token budget, about four characters to a \
token, and put what matters most first in each section.";

/// The write's system prompt, with the language rule for `language`.
fn write_system(language: Option<&str>) -> String {
    let rule = match language {
        None => "Write the entries in the language of the memories they cite.".to_owned(),
        Some(language) => format!(
            "Write every entry in {}, translating if the memories are in another language.",
            language.trim()
        ),
    };
    WRITE_SYSTEM.replace("{language_rule}", &rule)
}

/// The write request. Its first line is the question, which recorded-
/// refresh carry-over in replay matches records on.
fn write_request(input: &RefreshInput) -> LlmRequest {
    let mut user = format!(
        "Question: {}\nToken budget: {}\n\nSections:\n",
        input.question, input.max_tokens
    );
    for facet in &input.facets {
        user.push_str(&format!("- {}\n", facet.heading));
    }

    user.push_str("\nPrevious summary:\n");
    if input.entries.is_empty() {
        user.push_str("(none)\n");
    }
    let mut heading: Option<&Option<String>> = None;
    for entry in &input.entries {
        if heading != Some(&entry.section) {
            match &entry.section {
                Some(section) => user.push_str(&format!("### {section}\n")),
                None => user.push_str("### (no heading)\n"),
            }
            heading = Some(&entry.section);
        }
        user.push_str(&format!(
            "- {} [cites {}]\n",
            entry.text,
            entry.cites.join(", ")
        ));
    }

    user.push_str("\nMemories, under the facet that found each:\n");
    for facet in &input.facets {
        user.push_str(&format!("### {}\n", facet.heading));
        let mut any = false;
        for memory in input.memories.iter().filter(|m| m.facet == facet.heading) {
            user.push_str(&format!("{}: {}\n", memory.handle, memory.sentence));
            any = true;
        }
        if !any {
            user.push_str("(none)\n");
        }
    }
    LlmRequest {
        template: Template {
            name: WRITE_TEMPLATE.into(),
            version: WRITE_VERSION,
            guidance: None,
        },
        system: write_system(input.language.as_deref()),
        user,
        schema_name: "mental_model_summary".into(),
        schema: write_schema(),
        max_tokens: None,
    }
}

fn write_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "sentences": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "cites": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["text", "cites"],
                                "additionalProperties": false,
                            },
                        },
                    },
                    "required": ["heading", "sentences"],
                    "additionalProperties": false,
                },
            },
        },
        "required": ["sections"],
        "additionalProperties": false,
    })
}

#[derive(Deserialize)]
struct Written {
    sections: Vec<WrittenSection>,
}

#[derive(Deserialize)]
struct WrittenSection {
    #[serde(default)]
    heading: Option<String>,
    sentences: Vec<WrittenSentence>,
}

#[derive(Deserialize)]
struct WrittenSentence {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    cites: Vec<String>,
}

/// An entry as the refresh leaves it.
struct Draft {
    /// The stored row, for an entry that was there before.
    stored: Option<i64>,
    uuid: Uuid,
    section: Option<String>,
    text: String,
    cites: Vec<i64>,
    /// The text, citations or section differ from the stored row's.
    changed: bool,
}

/// A sentence that passed the citation checks.
struct Accepted {
    section: Option<String>,
    text: String,
    cites: Vec<i64>,
}

fn apply(
    cx: &Context<'_>,
    selection: &Selection,
    reply: Written,
    applied: &mut Applied,
) -> Vec<Draft> {
    let memories: BTreeMap<&str, i64> = selection
        .handles
        .iter()
        .map(|(memory, handle)| (handle.as_str(), *memory))
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

    let mut accepted: Vec<Accepted> = Vec::new();
    let mut index = 0;
    for section in reply.sections {
        let heading = section
            .heading
            .as_deref()
            .map(str::trim)
            .filter(|heading| !heading.is_empty())
            .map(str::to_owned);
        for sentence in section.sentences {
            let outcome = (|| {
                let text = match sentence.text.as_deref().map(str::trim) {
                    Some(text) if !text.is_empty() => text.to_owned(),
                    _ => return Err(RejectReason::EmptyText),
                };
                Ok(Accepted {
                    section: heading.clone(),
                    text,
                    cites: cites(&sentence.cites)?,
                })
            })();
            match outcome {
                Ok(sentence) => accepted.push(sentence),
                Err(reason) => applied.rejected.push(Rejected { index, reason }),
            }
            index += 1;
        }
    }

    // Which entry each sentence restates: the same text first, then, among
    // the rest, the same citations.
    let survivors = &selection.survivors;
    let mut claimed: Vec<Option<usize>> = vec![None; accepted.len()];
    let mut taken = vec![false; survivors.len()];
    let same_cites = |entry: &StoredEntry, cites: &[i64]| {
        let stored: BTreeSet<i64> = entry.cites.iter().map(|(memory, _)| *memory).collect();
        stored == cites.iter().copied().collect()
    };
    for matches in [
        &(|entry: &StoredEntry, sentence: &Accepted| entry.text == sentence.text)
            as &dyn Fn(&StoredEntry, &Accepted) -> bool,
        &|entry: &StoredEntry, sentence: &Accepted| same_cites(entry, &sentence.cites),
    ] {
        for (sentence, claim) in accepted.iter().zip(claimed.iter_mut()) {
            if claim.is_some() {
                continue;
            }
            if let Some(found) =
                (0..survivors.len()).find(|&at| !taken[at] && matches(&survivors[at], sentence))
            {
                taken[found] = true;
                *claim = Some(found);
            }
        }
    }

    let mut drafts = Vec::with_capacity(accepted.len());
    for (sentence, claim) in accepted.into_iter().zip(claimed) {
        match claim {
            Some(at) => {
                let entry = &survivors[at];
                let changed = entry.text != sentence.text
                    || entry.section != sentence.section
                    || !same_cites(entry, &sentence.cites);
                if changed {
                    applied.edited.push(entry.uuid);
                }
                drafts.push(Draft {
                    stored: Some(entry.id),
                    uuid: entry.uuid,
                    section: sentence.section,
                    text: sentence.text,
                    cites: sentence.cites,
                    changed,
                });
            }
            None => {
                let uuid = cx.store.new_id();
                applied.added.push(uuid);
                drafts.push(Draft {
                    stored: None,
                    uuid,
                    section: sentence.section,
                    text: sentence.text,
                    cites: sentence.cites,
                    changed: true,
                });
            }
        }
    }
    for (entry, taken) in survivors.iter().zip(taken) {
        if !taken {
            applied.removed.push(entry.uuid);
        }
    }
    drafts
}

/// Drops the lowest-ranked entries until the rest, with a heading line for
/// each section they leave, fit `max_tokens`. An entry ranks by the best
/// score among its cited memories; of two equal, the later one goes.
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
    let tokens = |drafts: &[Draft]| -> usize {
        let headings: BTreeSet<&str> = drafts
            .iter()
            .filter_map(|draft| draft.section.as_deref())
            .collect();
        drafts
            .iter()
            .map(|d| estimate_tokens(&d.text))
            .sum::<usize>()
            + headings
                .iter()
                .map(|heading| estimate_tokens(&crate::system_prompt::heading_line(heading)))
                .sum::<usize>()
    };
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
/// entry it restates, if any, still exists.
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
    // dropped meanwhile can't be restated: such a draft goes, as one failing
    // the citation check does.
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
                if !draft.changed {
                    continue;
                }
                tx.execute(
                    "UPDATE mental_model_entries SET text = ?2, section = ?3, updated_at = ?4
                     WHERE id = ?1",
                    (entry, &draft.text, &draft.section, now),
                )?;
                tx.execute(
                    "DELETE FROM mental_model_citations WHERE entry_id = ?1",
                    [entry],
                )?;
                entry
            }
            None => {
                tx.execute(
                    "INSERT INTO mental_model_entries (uuid, model_id, position, text, section,
                                                       created_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
                    (
                        draft.uuid.to_string(),
                        model.id,
                        position as i64,
                        &draft.text,
                        &draft.section,
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
