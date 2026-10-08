//! One refresh: a plan, one retrieval per facet, then one LLM call that
//! writes the whole answer.
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
//! 3. **Fingerprint.** The selection's memory ids, sentence hashes and
//!    significance levels, the ids of states whose confidence is below 0.9,
//!    the question, the plan, the filters and `max_tokens` are hashed.
//!    Crossing the confidence threshold changes the hash, but another day
//!    below it does not. Unless forced, a refresh whose hash matches the
//!    last completed one's stops here, with no write call.
//! 4. **The write.** The LLM gets the question, the facets, the selection
//!    by handle and significance under the facet that first found each
//!    memory, and the stored answer as the previous one. It's asked to
//!    prefer the more significant memories when the budget is tight, and,
//!    for the seeded profile, to keep only what will still hold in months.
//!    Possibly stale states carry their absolute observed date in the
//!    memory's timezone; the write is asked to include it in the prose. It
//!    replies with the whole answer, a heading and a paragraph or list per
//!    section, and the handles of every memory it rests on. A memory that
//!    left the selection isn't listed, so the reply can't cite it, and the
//!    prompt says that what the listed memories no longer support goes. With nothing selected there's
//!    nothing to ask: the answer is cleared, with no call.
//! 5. **Apply.** Code joins the sections into the stored text. A section
//!    with no text is left out. A reply with no text, no citations or a
//!    citation outside the selection changes nothing and is recorded like
//!    one that doesn't parse. Code can check the citations, not the prose.
//! 6. **Trim.** Over `max_tokens`, measured on the stored text with its
//!    headings, sentences or lines go from the end ([`Answer::trim_to`]).
//! 7. **Store.** The answer and its citations replace the old ones whole,
//!    unless a memory it cites was forgotten while it was written, or a
//!    forget blanked the answer meanwhile: then nothing is stored, and the
//!    refresh the forget requested writes it again.
//!
//! The prompts and replies are never written to disk, so forget has nothing
//! to scrub there. Each facet's retrieval writes one `refresh` row to the
//! recall log, with its query, and nothing writes an access.

use std::collections::BTreeMap;
use std::time::Instant;

use rusqlite::OptionalExtension;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::schedule::Schedule;
use super::{
    Answer, Applied, Facet, FailureKind, InputMemory, ModelError, ModelRow, Outcome, PLAN_TEMPLATE,
    PLAN_VERSION, RefreshInput, RejectReason, Rejected, ScoredFacet, StoredPlan, WRITE_TEMPLATE,
    WRITE_VERSION, load_cites, volatility_str,
};
use crate::constants::{STATE_AGE_SHOWN_BELOW, TAU};
use crate::models::{LlmClient, LlmError, LlmRequest, Template};
use crate::retrieval::candidates::Candidate;
use crate::retrieval::{
    Context, REFRESH_RERANK_DEADLINE, Selected, estimate_tokens, linked_memories,
};
use crate::store::micros;
use crate::strength::{Kind, Phase};

/// What a refresh works from.
struct Selection {
    /// Memory rowid to handle.
    handles: BTreeMap<i64, String>,
    input: RefreshInput,
    /// Each facet's whole scored pool, when they were asked for.
    scored: Option<Vec<ScoredFacet>>,
}

fn select(
    cx: &Context<'_>,
    model: &ModelRow,
    facets: Vec<Facet>,
    log: bool,
    pools: bool,
) -> Result<Selection, ModelError> {
    let (previous, cited, linked) = {
        let conn = cx.store.connection();
        let previous: Option<String> = conn.query_row(
            "SELECT answer FROM mental_models WHERE id = ?1",
            [model.id],
            |row| row.get(0),
        )?;
        let cited: Vec<i64> = load_cites(&conn, model.id)?
            .into_iter()
            .map(|(memory, _)| memory)
            .collect();
        let linked = match model.entity_id {
            Some(entity) => Some(linked_memories(&conn, model.bank_id, entity)?),
            None => None,
        };
        (previous, cited, linked)
    };
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
    // Each facet's best, and past them the cited memories it scored. A
    // replay may retrieve a facet by another query; the plan keeps its own.
    let queries: Vec<&str> = facets
        .iter()
        .map(|facet| {
            cx.refresh_queries
                .get(&facet.heading)
                .unwrap_or(&facet.query)
                .as_str()
        })
        .collect();
    let found = crate::retrieval::select(
        cx,
        model.bank_id,
        &queries,
        &keep,
        &cited,
        facet_budget,
        facet_budget + cited.len(),
        log,
        pools,
        Instant::now() + REFRESH_RERANK_DEADLINE,
    )?;
    let mut ranked: Vec<Vec<Selected>> = Vec::with_capacity(facets.len());
    let mut extras: Vec<(usize, Selected)> = Vec::new();
    let mut scored: Vec<ScoredFacet> = Vec::new();
    for (index, selection) in found.into_iter().enumerate() {
        if let (Some(pool), Some(facet)) = (selection.pool, facets.get(index)) {
            scored.push(ScoredFacet {
                heading: facet.heading.clone(),
                query: facet.query.clone(),
                pool,
            });
        }
        let mut found = selection.selected;
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
                sentence: write_sentence(&item.candidate),
                facet: facets
                    .get(*facet)
                    .map(|facet| facet.heading.clone())
                    .unwrap_or_default(),
                significance: item.candidate.significance.clone(),
            })
            .collect(),
        previous,
        fingerprint: fingerprint(model, &facets, &selected),
        facets,
    };
    let scored = pools.then(|| {
        let by_memory: BTreeMap<Uuid, &str> = input
            .memories
            .iter()
            .map(|memory| (memory.memory, memory.handle.as_str()))
            .collect();
        for facet in &mut scored {
            for candidate in &mut facet.pool.candidates {
                candidate.input = by_memory.get(&candidate.memory).map(|h| (*h).to_owned());
            }
        }
        scored
    });
    Ok(Selection {
        handles,
        input,
        scored,
    })
}

/// The selection from each facet's best, in rank order: the first of each
/// facet in turn, then the second of each, and so on, skipping a memory
/// already in, until there are `budget`. Then the memories in `cited` that
/// any facet scored, best first, until there are `with_cited`. A memory
/// keeps the best score any facet gave it, which the cited fill ranks by.
/// Returns the selection with the index of the facet that first found each memory,
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

fn stale_state(candidate: &Candidate) -> bool {
    candidate.window.kind == Kind::State && candidate.state_confidence < STATE_AGE_SHOWN_BELOW
}

/// Dates belong in the write input, not annotations on the stored answer.
fn write_sentence(candidate: &Candidate) -> String {
    if stale_state(candidate) {
        let observed = candidate
            .last_observed
            .to_zoned(candidate.tz.clone())
            .strftime("%a %-d %b %Y");
        format!("{} [observed {observed}, may be stale]", candidate.content)
    } else {
        candidate.content.clone()
    }
}

/// Hash the selection's ids, sentence hashes, significance levels and
/// stale-state ids in id order, with the question, plan, filters and token
/// budget. Score changes alone do not count.
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
    let mut memories: Vec<(Uuid, String, &str)> = selected
        .iter()
        .map(|item| {
            (
                item.candidate.uuid,
                format!("{:x}", Sha256::digest(item.candidate.content.as_bytes())),
                item.candidate.significance.as_str(),
            )
        })
        .collect();
    memories.sort();
    for (uuid, sentence, significance) in memories {
        hasher.update(format!("memory:{uuid}:{sentence}:{significance}\n"));
    }
    // Only threshold crossings count, not another day of staleness.
    let mut stale: Vec<Uuid> = selected
        .iter()
        .filter(|item| stale_state(&item.candidate))
        .map(|item| item.candidate.uuid)
        .collect();
    stale.sort();
    for uuid in stale {
        hasher.update(format!("stale:{uuid}\n"));
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
    Ok(select(cx, model, facets, false, false)?.input)
}

/// Refreshes `model` with `llm`. Unless `force`, the write is skipped when
/// the fingerprint matches the last completed refresh's. A failed
/// retrieval, an LLM failure or a malformed reply is `Ok(Outcome::Failed)`,
/// recorded on the model so the schedule waits before trying again. An LLM
/// held by a limit is `Ok(Outcome::Held)`: nothing is recorded as failed,
/// and the refresh stays requested until the hold lifts. With `scored`,
/// each facet's whole scored pool is put there once the selection is made,
/// whatever the refresh goes on to do.
pub(crate) fn refresh(
    cx: &Context<'_>,
    schedule: &Schedule,
    model: &ModelRow,
    llm: &dyn LlmClient,
    force: bool,
    scored: Option<&mut Vec<ScoredFacet>>,
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
    let selection = match select(cx, model, facets, true, scored.is_some()) {
        Ok(mut selection) => {
            if let (Some(scored), Some(pools)) = (scored, selection.scored.take()) {
                *scored = pools;
            }
            selection
        }
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
                    refresh_requested_at = CASE WHEN ?2 THEN NULL ELSE refresh_requested_at END,
                    refresh_urgent = CASE WHEN ?2 THEN 0 ELSE refresh_urgent END
             WHERE id = ?1",
            (model.id, started.unchanged()),
        )?;
        return Ok(Outcome::Unchanged);
    }

    if selection.input.memories.is_empty() {
        // Nothing qualifies, so there's nothing for an answer to rest on
        // and nothing to ask: the answer is cleared, with no write call.
        store(cx, model, &selection, None, started)?;
        return Ok(Outcome::Applied(Applied::default()));
    }
    let request = write_request(
        &selection.input,
        model.name == crate::store::bank::PROFILE_NAME,
    );
    let identities: Vec<(String, Uuid)> = selection
        .input
        .memories
        .iter()
        .map(|memory| (memory.handle.clone(), memory.memory))
        .collect();
    if llm.skips_write(&request, &identities) {
        // Replay's `--refresh off`: the write counts as made and changes
        // nothing, so the schedule runs as it would have.
        settle(cx, model, &selection, started)?;
        return Ok(Outcome::Applied(Applied::default()));
    }
    let reply = match llm.complete_identified(&request, &identities) {
        Ok(response) => response.json,
        Err(error) => return llm_failed(cx, schedule, model, &error),
    };
    let Some((mut written, applied)) = serde_json::from_value::<Written>(reply)
        .ok()
        .and_then(|reply| apply(&selection, reply))
    else {
        tracing::warn!(model = %model.uuid, "a mental model refresh got a malformed answer");
        return failed(cx, model, FailureKind::Malformed);
    };
    let max_tokens = model.max_tokens as usize;
    let trimmed = written
        .answer
        .trim_to(|answer| estimate_tokens(&answer.text()) <= max_tokens);
    let stored = store(cx, model, &selection, Some(&written), started)?;
    Ok(Outcome::Applied(Applied {
        written: stored,
        trimmed,
        ..applied
    }))
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

/// When a limit the LLM reported lifts, or `None` for any other error. A
/// call the daemon's shutdown stopped is held too, due again at once: the
/// next start makes it.
fn held_until(error: &LlmError, now: jiff::Timestamp) -> Option<jiff::Timestamp> {
    match error {
        LlmError::Stopped => Some(now),
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

const WRITE_SYSTEM: &str = "You write a mental model: a summary of the user's memories that \
answers a standing question about them. An AI assistant reads it at the start of every \
conversation, instead of reading the memories themselves, to understand the user and help them \
well. Give the assistant everything the memories say that answers the question, and nothing the \
question rules out, with the specifics that make a fact usable. Don't generalise a specific away, \
and merge memories that say the same thing.

Write a section for each facet listed, in that order and under the facet's heading, with what \
matters most first, and leave out a facet the memories say nothing about. Write a section as one \
paragraph of connected prose when its facts bear on each other, or as a list, one fact per line \
starting with \"- \", when they stand alone. Don't mix the two in a section, and don't split one \
fact over lines. Say nothing the memories you cite don't support, and use names, not \
pronouns, for the people in it. When a memory is marked with an observed date and may be stale, \
write its fact with that absolute date (for example, 'as of 1 Sep'), not as an unqualified \
current fact or a relative age. {language_rule}

Each memory is listed with its significance: trivial, minor, notable, major or critical, or kept \
when the user asked for it to be remembered. When the budget is tight, keep the more significant \
memories and leave out the less significant ones.

The previous answer keeps the wording steady; it is not a limit. Restate what the memories listed \
still support, reword what they change, add what they support that it left out, and leave out \
what they no longer support or what no longer answers the question: anything you leave out is \
gone. Cite by handle every memory the answer rests on, in the cites list for the whole answer \
and never in the text, and only the handles listed (m1, m2, ...). Use the token budget, headings included, at about four \
characters to a token, for what the memories support: leaving out a supported fact that answers \
the question is worse than a longer answer, but don't pad and don't go over.\
{profile_rule}";

/// The seeded profile's durability rule. Other models keep whatever time
/// scale their question asks about, such as upcoming trips. It comes last,
/// after the previous answer's paragraph, so restating what a listed memory
/// still supports doesn't carry a message over from the previous answer. A
/// message's memory supports only a lasting fact it states outright; a
/// relationship or trait read into it, or a previous sentence only such a
/// reading would support, isn't cited to it.
const PROFILE_RULE: &str = "

This answer is the user's profile: it holds what will still be true about them in months. A \
message the user sent, received, quoted or forwarded, an order, a booking, or the details of a \
one-off purchase or trip is not profile material, whatever its significance and even when a \
heading seems to fit it. Leave out its wording, quoted or paraphrased, and the one-off matter it \
was about, and don't recast that matter as a habit or trait of the user.

A message's memory may support only a lasting fact it states in so many words, such as a person's \
name, their relationship to the user or a birthday, or a preference the user states in it. When \
another listed memory states the same fact, write it from that memory and cite that one; \
otherwise write the fact on its own, without the message's occasion, and cite the message's \
memory for it. Nothing else rests on it: not a relationship the memory doesn't name, and nothing \
read into who the message is to or from, its tone or what it's about. Don't cite a memory that \
holds nothing lasting, and drop what it supported from the previous answer even though the memory \
is still listed. When only such a reading of a message would support a sentence of the previous \
answer, drop the sentence rather than cite the message for it.

Recurring personal dates, such as birthdays and anniversaries, belong with the people they are \
about. Say how each person is related to the user when a memory says so.";

/// The write's system prompt, with the language rule for `language`, and
/// the durability rule when it writes the seeded profile.
fn write_system(language: Option<&str>, profile: bool) -> String {
    let rule = match language {
        None => "Write in the language of the memories you cite.".to_owned(),
        Some(language) => format!(
            "Write in {}, translating if the memories are in another language.",
            language.trim()
        ),
    };
    WRITE_SYSTEM
        .replace("{language_rule}", &rule)
        .replace("{profile_rule}", if profile { PROFILE_RULE } else { "" })
}

/// The write request. Its first line is the question, which recorded-
/// refresh carry-over in replay matches records on. `profile` is whether
/// it writes the seeded profile.
fn write_request(input: &RefreshInput, profile: bool) -> LlmRequest {
    let mut user = format!(
        "Question: {}\nToken budget: {}\n\nSections:\n",
        input.question, input.max_tokens
    );
    for facet in &input.facets {
        user.push_str(&format!("- {}\n", facet.heading));
    }

    user.push_str("\nPrevious answer:\n");
    // Its headings a level under the facets' below, which are what the
    // reply's sections are written under.
    match &input.previous {
        Some(previous) => {
            user.push_str(&Answer::parse(previous).render("####"));
            user.push('\n');
        }
        None => user.push_str("(none)\n"),
    }

    user.push_str("\nMemories, under the facet that found each:\n");
    for facet in &input.facets {
        user.push_str(&format!("### {}\n", facet.heading));
        let mut any = false;
        for memory in input.memories.iter().filter(|m| m.facet == facet.heading) {
            user.push_str(&format!(
                "{} ({}): {}\n",
                memory.handle, memory.significance, memory.sentence
            ));
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
        system: write_system(input.language.as_deref(), profile),
        user,
        schema_name: "mental_model_answer".into(),
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
                        "text": {"type": "string"},
                    },
                    "required": ["heading", "text"],
                    "additionalProperties": false,
                },
            },
            "cites": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["sections", "cites"],
        "additionalProperties": false,
    })
}

/// The write's reply. A first-version reply, sentence by sentence, has no
/// `text` or `cites`, and doesn't parse.
#[derive(Deserialize)]
struct Written {
    sections: Vec<WrittenSection>,
    cites: Vec<String>,
}

#[derive(Deserialize)]
struct WrittenSection {
    #[serde(default)]
    heading: Option<String>,
    text: String,
}

/// A reply that passed the checks: its answer and the memories it cites,
/// by rowid, each once in the order given.
struct Accepted {
    answer: Answer,
    cites: Vec<i64>,
}

/// The reply as an answer, or `None` when it has no text, cites nothing or
/// cites a handle that isn't in the selection.
fn apply(selection: &Selection, reply: Written) -> Option<(Accepted, Applied)> {
    let memories: BTreeMap<&str, i64> = selection
        .handles
        .iter()
        .map(|(memory, handle)| (handle.as_str(), *memory))
        .collect();
    let mut cites: Vec<i64> = Vec::new();
    for handle in &reply.cites {
        let memory = *memories.get(handle.trim())?;
        if !cites.contains(&memory) {
            cites.push(memory);
        }
    }
    if cites.is_empty() {
        return None;
    }
    let mut applied = Applied::default();
    for (index, section) in reply.sections.iter().enumerate() {
        if section.text.trim().is_empty() {
            applied.rejected.push(Rejected {
                index,
                reason: RejectReason::EmptyText,
            });
        }
    }
    let answer = Answer::new(
        reply
            .sections
            .into_iter()
            .map(|section| (section.heading, section.text)),
    );
    if answer.is_empty() {
        return None;
    }
    Some((Accepted { answer, cites }, applied))
}

/// Whether a reply made from `selection` can still be stored: no memory it
/// cites was forgotten or erased while it was written, and no forget has
/// blanked the answer it was shown.
fn still_standing(
    tx: &rusqlite::Transaction<'_>,
    model: &ModelRow,
    selection: &Selection,
    cites: &[i64],
) -> Result<bool, rusqlite::Error> {
    for memory in cites {
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
    let answer: Option<Option<String>> = tx
        .query_row(
            "SELECT answer FROM mental_models WHERE id = ?1",
            [model.id],
            |row| row.get(0),
        )
        .optional()?;
    Ok(answer.is_some_and(|answer| answer == selection.input.previous))
}

/// Stores `written`, or clears the answer when it's `None` or trimmed to
/// nothing, and records the refresh as completed. Returns whether an
/// answer was stored. A reply that no longer stands
/// ([`still_standing`]) stores nothing and leaves the refresh requested.
fn store(
    cx: &Context<'_>,
    model: &ModelRow,
    selection: &Selection,
    written: Option<&Accepted>,
    started: Started<'_>,
) -> Result<bool, ModelError> {
    let mut conn = cx.store.connection();
    let tx = conn.transaction()?;
    let written = written.filter(|written| !written.answer.is_empty());
    if let Some(written) = written
        && !still_standing(&tx, model, selection, &written.cites)?
    {
        return Ok(false);
    }
    tx.execute(
        "UPDATE mental_models SET answer = ?2 WHERE id = ?1",
        (model.id, written.map(|written| written.answer.text())),
    )?;
    tx.execute(
        "DELETE FROM mental_model_cites WHERE model_id = ?1",
        [model.id],
    )?;
    for memory in written.iter().flat_map(|written| &written.cites) {
        tx.execute(
            "INSERT OR IGNORE INTO mental_model_cites (model_id, memory_id) VALUES (?1, ?2)",
            (model.id, memory),
        )?;
    }
    completed(&tx, cx, model, selection, started)?;
    tx.commit()?;
    Ok(written.is_some())
}

/// Records the refresh as completed without touching the answer.
fn settle(
    cx: &Context<'_>,
    model: &ModelRow,
    selection: &Selection,
    started: Started<'_>,
) -> Result<(), ModelError> {
    let mut conn = cx.store.connection();
    let tx = conn.transaction()?;
    completed(&tx, cx, model, selection, started)?;
    tx.commit()?;
    Ok(())
}

/// The refresh's fingerprint and time, with the error and, unless a request
/// came in while it ran, the request cleared.
fn completed(
    tx: &rusqlite::Transaction<'_>,
    cx: &Context<'_>,
    model: &ModelRow,
    selection: &Selection,
    started: Started<'_>,
) -> Result<(), rusqlite::Error> {
    tx.execute(
        "UPDATE mental_models SET last_fingerprint = ?2, last_refreshed_at = ?3,
                last_error_kind = NULL, last_error_at = NULL,
                refresh_requested_at = CASE WHEN ?4 THEN NULL ELSE refresh_requested_at END,
                refresh_urgent = CASE WHEN ?4 THEN 0 ELSE refresh_urgent END,
                updated_at = ?3
         WHERE id = ?1",
        (
            model.id,
            &selection.input.fingerprint,
            micros(cx.store.now()),
            started.unchanged(),
        ),
    )?;
    Ok(())
}
