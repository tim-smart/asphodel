//! Retrieval: prefetch's injection and the recall tool.
//!
//! Both run one pipeline:
//!
//! 1. **Retrievers.** Vector search and BM25 over memory content, and an
//!    entity arm over the memories linked to entities named in the query,
//!    [`CANDIDATES_PER_ARM`] hits each ([`arms`]).
//! 2. **Clean-up.** Retracted and hidden hits drop out, the rest become the
//!    heads of their supersession chains, and strength is computed for each
//!    ([`candidates`]). Injection drops heads below τ and those already in
//!    context here, so they never take places in the fused list; explicit
//!    recall applies its filters.
//! 3. **Fusion.** Unweighted RRF ([`fuse`]); an empty arm contributes
//!    nothing.
//! 4. **Reranking.** The top [`RERANKED`] go to the reranker, under
//!    [`RERANKER_DEADLINE`](crate::constants::RERANKER_DEADLINE) measured
//!    from the start of the request. Past it, explicit recall keeps RRF
//!    order and prefetch injects nothing: without the logit there's no
//!    relevance to gate on, and a wrong injection is replayed for the rest
//!    of the session. Prefetch reranks against the message, or with
//!    `injection.rerank_query = "conversation"` against
//!    [`conversation_query`].
//! 5. **Score.** [`score`]: relevance (the reranker logit), plus w_s times
//!    strength, plus the clamped log of state confidence, plus the phase
//!    term. w_s is `ranking.w_s_inject` or `ranking.w_s_recall`, and the
//!    phase term counts in injection, and in recall only when the caller
//!    filters by phase.
//!
//! Injection then gates on the reranker floor for the loaded model and
//! takes at most `injection.cap` memories in about `injection.token_budget`
//! tokens. Prefetch puts the session's agenda update, if it has one, ahead
//! of it, under its own budget
//! ([`agenda_update`](crate::system_prompt::agenda_update)). Every recall
//! writes one row to the recall log ([`log`]) and none writes an access: being recalled never strengthens a memory. "Now" is
//! always the service's clock.
//!
//! Recall and prefetch each work out their result in one function with no
//! side effects, then log it and update the session. [`explain`] calls the
//! same two functions and returns their working instead, so it can't drift
//! from either ([`explain`](self::explain)).

mod arms;
pub(crate) mod candidates;
mod explain;
pub(crate) mod format;
mod log;
mod rerank;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, MutexGuard};
use std::time::{Duration, Instant};

use jiff::tz::TimeZone;
use jiff::{Span, Timestamp, ToSpan};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub(crate) use arms::bm25;
pub use explain::{
    Arm, ArmRank, Cut, Explain, ExplainInjection, ExplainMode, ExplainRecall, ExplainRequest,
    Explained, ExplainedInjection, ScoreParts, StageLatency,
};
pub(crate) use rerank::Permit;

use crate::config::{RankingTuning, RerankQuery, Tuning};
use crate::constants::{
    CANDIDATES_PER_ARM, CONFIDENCE_TERM_MIN, ENDED_GRACE_DAYS, OVERDUE_FULL_DAYS,
    OVERDUE_ZERO_DAYS, RECALL_LIMIT_DEFAULT, RECALL_LIMIT_MAX, RECENTLY_PAST_DAYS,
    RERANK_CONTEXT_CHARS, RERANKED, RRF_K, SHORT_FOLLOW_UP_WORDS, TAU, UPCOMING_BONUS_DAYS,
};
use crate::models::{ModelError, Models};
use crate::sessions::Sessions;
use crate::store::{Store, StoreError};
use crate::strength::{Kind, Phase, TimePrecision, Window, WorldTime, unit_end};
use candidates::{Candidate, Cleanup};
use log::{Entry, Logged, RecallKind};

const MICROS_PER_DAY: f64 = 24.0 * 60.0 * 60.0 * 1_000_000.0;

/// What `prefetch` sends.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PrefetchRequest {
    pub session_id: String,
    /// The user's current message.
    pub query: String,
    /// The plugin's last prefetch query for the session, which a short
    /// follow-up borrows. The plugin drops it on `memory_forget`.
    #[serde(default)]
    pub previous_query: Option<String>,
    /// The assistant's reply to the previous message, which the
    /// conversation query starts from.
    #[serde(default)]
    pub previous_reply: Option<String>,
    /// The block `system_prompt_block()` returned, when Hermes gave no
    /// session id then. The plugin sends it with the session's first
    /// prefetch, and the daemon maps the session to that block unless it
    /// already holds one.
    #[serde(default)]
    pub block_id: Option<Uuid>,
}

/// What `prefetch` returns.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Prefetch {
    /// The recall log row's id. `sync_turn` echoes it to commit the
    /// injection to the session's in-context set.
    pub recall_id: Uuid,
    /// The injection, or empty when nothing passed the gate or the
    /// reranker missed its deadline.
    pub text: String,
    /// The memories injected, in the order they're listed.
    pub injected: Vec<Uuid>,
    /// Whether the reranker answered in time. When it didn't, nothing is
    /// injected.
    pub reranked: bool,
}

/// What a `happened`/`said` range applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum On {
    /// The validity window. A low-confidence window is widened by one unit
    /// of its precision at each end, and either end can be open. A fact
    /// with no end matches only through a stated start inside the range.
    #[default]
    Happened,
    /// `observed_at`, exactly.
    Said,
}

/// The recall tool's phase filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseFilter {
    Upcoming,
    /// Recently or long past.
    Past,
    /// Current, including an open task that's overdue.
    Current,
    #[default]
    Any,
}

impl PhaseFilter {
    pub(crate) fn admits(self, phase: Phase) -> bool {
        match self {
            PhaseFilter::Upcoming => phase == Phase::Upcoming,
            PhaseFilter::Past => matches!(phase, Phase::RecentlyPast | Phase::LongPast),
            PhaseFilter::Current => matches!(phase, Phase::Current | Phase::Overdue),
            PhaseFilter::Any => true,
        }
    }
}

/// `memory_recall`'s parameters, plus the session
/// whose in-context set the results join.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RecallRequest {
    #[serde(default)]
    pub session_id: Option<String>,
    pub query: String,
    #[serde(default)]
    pub from: Option<Timestamp>,
    #[serde(default)]
    pub to: Option<Timestamp>,
    #[serde(default)]
    pub on: On,
    #[serde(default)]
    pub phase: PhaseFilter,
    /// Empty means every kind.
    #[serde(default)]
    pub kinds: Vec<Kind>,
    /// A name or alias, resolved through the alias FTS; when it names
    /// several entities, recall takes the union.
    #[serde(default)]
    pub entity: Option<String>,
    /// [`RECALL_LIMIT_DEFAULT`] when absent, and never more than
    /// [`RECALL_LIMIT_MAX`].
    #[serde(default)]
    pub limit: Option<usize>,
}

/// What an explicit recall returns.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Recall {
    pub recall_id: Uuid,
    pub results: Vec<Recalled>,
    /// Plain lines for the recall tool, with ids and source-local dates.
    /// Structured results remain available to CLI and dashboard callers.
    pub text: String,
    /// Whether the reranker answered in time. When it didn't, the results
    /// are in RRF order.
    pub reranked: bool,
}

/// One recall-tool result.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Recalled {
    pub id: Uuid,
    pub sentence: String,
    pub kind: Kind,
    pub window: RecalledWindow,
    pub phase: Phase,
    pub observed_at: Timestamp,
    pub strength: Band,
    pub kept: bool,
}

/// A result's validity window as stored: each time is the instant its unit
/// starts in the source's timezone.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecalledWindow {
    pub valid_from: Option<WorldTime>,
    pub valid_until: Option<WorldTime>,
    pub until_event: Option<String>,
    pub due_at: Option<WorldTime>,
    pub recurrence: Option<String>,
    /// Extraction was unsure of the dates.
    pub uncertain: bool,
}

/// A strength band: faded below τ, fading up to
/// `recall.strong_cutoff`, strong above it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Band {
    Strong,
    Fading,
    Faded,
}

/// The band of `strength` under `strong_cutoff`.
pub fn band(strength: f64, strong_cutoff: f64) -> Band {
    if strength > strong_cutoff {
        Band::Strong
    } else if strength >= TAU {
        Band::Fading
    } else {
        Band::Faded
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RecallError {
    #[error("unknown bank")]
    UnknownBank,

    /// The service was built without models.
    #[error("no models are loaded, so nothing can be recalled")]
    NoModels,

    /// `from` is after `to`.
    #[error("the range's start is after its end")]
    InvertedRange,

    /// Embedding the query failed.
    #[error("embedding the query failed: {error}")]
    Model { error: ModelError },

    /// The bank's recorded embedding model isn't loaded, so its vectors
    /// can't be searched. `asphodel reembed --bank` moves it to
    /// the daemon's model.
    #[error(
        "the bank records embedding model {model}, which this daemon doesn't carry; run `asphodel reembed --bank` to move it"
    )]
    ModelUnavailable { model: String },

    /// A re-embed swapped the bank's model while the query was embedded,
    /// and again while it was embedded once more with the new model. The
    /// query was never searched against vectors of another model.
    #[error("the bank's embedding model changed while the query was embedded; try again")]
    ModelChanged,

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for RecallError {
    fn from(error: rusqlite::Error) -> Self {
        RecallError::Store(StoreError::Sqlite(error))
    }
}

/// Reciprocal rank fusion of ranked lists, best first, each id once: a hit
/// at rank r of a list scores 1 / ([`RRF_K`] + r), and an id's score is the
/// sum over the lists. A repeat within one list counts once, an empty list
/// contributes nothing, and ties go to the lower id, so the order is stable.
/// It knows nothing about memories: reconciliation calls it without any
/// ranking, and recall ranks afterwards.
pub fn fuse(lists: &[&[i64]]) -> Vec<i64> {
    let mut scores: BTreeMap<i64, f64> = BTreeMap::new();
    for list in lists {
        let mut seen = BTreeSet::new();
        let mut rank = 0usize;
        for id in list.iter() {
            if !seen.insert(*id) {
                continue;
            }
            rank += 1;
            *scores.entry(*id).or_default() += 1.0 / (RRF_K + rank as f64);
        }
    }
    let mut fused: Vec<(i64, f64)> = scores.into_iter().collect();
    fused.sort_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then(left_id.cmp(right_id))
    });
    fused.into_iter().map(|(id, _)| id).collect()
}

/// The retrieval score:
///
/// ```text
/// relevance + w_s·strength + max(−3, ln(state_confidence)) + phase_term
/// ```
///
/// `relevance` is the reranker logit divided by the reranker's relevance
/// scale ([`Reranking::relevance`]). `state_confidence` is 1.0 for anything
/// but a state, so its term is 0.
pub fn score(
    relevance: f64,
    w_s: f64,
    strength: f64,
    state_confidence: f64,
    phase_term: f64,
) -> f64 {
    relevance + w_s * strength + state_confidence.ln().max(CONFIDENCE_TERM_MIN) + phase_term
}

/// The phase term at `now`, on world time in the
/// window's timezone. It never excludes anything.
///
/// | Phase | Term |
/// |---|---|
/// | Upcoming | 0 until 7 days before the start, then rising linearly to the full bonus at it |
/// | Current | 0 |
/// | Overdue | The full bonus until 14 days past due, then falling linearly to 0 at 60 |
/// | Ended up to 7 days ago | 0 |
/// | Ended 7 to 30 days ago | The penalty, rising linearly to its full value |
/// | Ended more than 30 days ago | The full penalty |
///
/// A low-confidence window halves the term in both directions.
pub fn phase_term(
    window: &Window,
    low_confidence: bool,
    tz: &TimeZone,
    now: Timestamp,
    ranking: &RankingTuning,
) -> f64 {
    let term = match window.phase(tz, now) {
        Phase::Upcoming => match window.valid_from {
            Some(from) => {
                let days = days_between(now, from.at).max(0.0);
                if days >= UPCOMING_BONUS_DAYS {
                    0.0
                } else {
                    ranking.phase_bonus * (1.0 - days / UPCOMING_BONUS_DAYS)
                }
            }
            None => 0.0,
        },
        Phase::Current => 0.0,
        Phase::Overdue => match window.overdue_from(tz) {
            Some(from) => {
                let days = days_between(from, now).max(0.0);
                if days <= OVERDUE_FULL_DAYS {
                    ranking.phase_bonus
                } else if days >= OVERDUE_ZERO_DAYS {
                    0.0
                } else {
                    ranking.phase_bonus * (OVERDUE_ZERO_DAYS - days)
                        / (OVERDUE_ZERO_DAYS - OVERDUE_FULL_DAYS)
                }
            }
            None => 0.0,
        },
        Phase::RecentlyPast | Phase::LongPast => match window.closes_at(tz) {
            Some(closed) => {
                let days = days_between(closed, now).max(0.0);
                if days <= ENDED_GRACE_DAYS {
                    0.0
                } else if days >= RECENTLY_PAST_DAYS {
                    -ranking.phase_penalty
                } else {
                    -ranking.phase_penalty * (days - ENDED_GRACE_DAYS)
                        / (RECENTLY_PAST_DAYS - ENDED_GRACE_DAYS)
                }
            }
            None => 0.0,
        },
    };
    if low_confidence { term / 2.0 } else { term }
}

/// The note Hermes' Discord gateway puts in front of a turn's message,
/// up to the message id.
const DISCORD_NOTE_START: &str = "[Triggering message id: `";

/// The user's message as prefetch recalls for it: without the note Hermes'
/// Discord gateway puts in front of it, naming the triggering message's id,
/// and without the `[Name] ` speaker prefix of a shared thread, which
/// follows the note. Nothing else is touched, and either can be missing.
/// A message that is only these cleans to nothing.
pub fn clean_query(raw: &str) -> String {
    let mut text = raw.trim_start();
    if let Some(rest) = text.strip_prefix(DISCORD_NOTE_START)
        && let Some(end) = rest.find(']')
        && !rest[..end].contains('\n')
    {
        text = rest[end + 1..].trim_start();
    }
    // As the replay importer reads the speaker: `^\[([^\]\n]+)\] `.
    if let Some(rest) = text.strip_prefix('[')
        && let Some(end) = rest.find(']')
        && end > 0
        && !rest[..end].contains('\n')
        && let Some(message) = rest[end + 1..].strip_prefix(' ')
    {
        text = message;
    }
    text.trim().to_owned()
}

/// The query prefetch runs: the message itself, or, when it's a short
/// follow-up of fewer than [`SHORT_FOLLOW_UP_WORDS`] words split on
/// whitespace ("yes, book it"), the previous prefetch query and then the
/// message, on two lines. The
/// recall log stores this query.
pub fn effective_query(message: &str, previous: Option<&str>) -> String {
    let message = message.trim();
    match previous.map(str::trim) {
        Some(previous)
            if !previous.is_empty()
                && message.split_whitespace().count() < SHORT_FOLLOW_UP_WORDS =>
        {
            format!("{previous}\n{message}")
        }
        _ => message.to_owned(),
    }
}

/// The query a conversation-wide rerank scores against: the current
/// message, the start of the previous message, then the start of the
/// assistant's reply to it, one per line, leaving out any that's empty.
/// Each context start is at most [`RERANK_CONTEXT_CHARS`] characters. The
/// message is kept whole here, but the model may truncate the pair.
pub fn conversation_query(message: &str, previous: Option<&str>, reply: Option<&str>) -> String {
    let context = [previous, reply]
        .into_iter()
        .flatten()
        .map(|text| start_of(text.trim(), RERANK_CONTEXT_CHARS));
    [message.trim()]
        .into_iter()
        .chain(context)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// At most `max` characters from the start of `text`, cut back to the last
/// whitespace when it has to be cut and there's any.
fn start_of(text: &str, max: usize) -> &str {
    let Some((end, _)) = text.char_indices().nth(max) else {
        return text;
    };
    let cut = &text[..end];
    match cut.rfind(char::is_whitespace) {
        Some(space) => cut[..space].trim_end(),
        None => cut,
    }
}

/// About how many tokens `text` costs: a quarter of its characters, rounded
/// up. It's what the injection budget counts.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

/// World days from `from` to `to`, negative when `to` is earlier.
fn days_between(from: Timestamp, to: Timestamp) -> f64 {
    (to.as_microsecond() - from.as_microsecond()) as f64 / MICROS_PER_DAY
}

/// What a service hands the pipeline.
pub(crate) struct Context<'a> {
    pub store: &'a Store,
    pub tuning: &'a Tuning,
    pub models: &'a Models,
    /// The loaded reranker's floor and relevance scale.
    pub reranking: Reranking,
    /// Embedding models the daemon carries besides its own, for banks a
    /// re-embed hasn't moved yet.
    pub previous: &'a [Arc<dyn crate::models::Embedder>],
    pub sessions: &'a Sessions,
    /// The service's permit to run its reranker.
    pub permit: &'a Arc<Permit>,
    /// [`RERANKER_DEADLINE`](crate::constants::RERANKER_DEADLINE) unless a
    /// test or bench set another.
    pub deadline: Duration,
    /// Replay only: refresh facet heading to the query its retrieval runs
    /// instead of the plan's.
    pub refresh_queries: &'a BTreeMap<String, String>,
}

/// The gate floor and relevance scale for the loaded reranker, looked up
/// by its exact model id once, when the service opens. There's no fallback
/// for either.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Reranking {
    /// The injection gate, compared with the raw logit.
    pub floor: f64,
    /// What the logit is divided by to give relevance.
    pub scale: f64,
}

impl Reranking {
    /// The floor and scale `tuning` gives `reranker_model`, or `None` when
    /// either is missing.
    pub fn for_model(tuning: &Tuning, reranker_model: &str) -> Option<Self> {
        Some(Self {
            floor: *tuning.injection.reranker_floors.get(reranker_model)?,
            scale: *tuning.ranking.relevance_scales.get(reranker_model)?,
        })
    }

    /// The relevance term of the score for a reranker logit.
    pub fn relevance(&self, logit: f64) -> f64 {
        logit / self.scale
    }
}

/// How many times a query is embedded before a recall gives up on a bank
/// whose model keeps changing under it: once, and once more after a swap.
const QUERY_EMBED_ATTEMPTS: usize = 2;

impl Context<'_> {
    /// Whether a query for `bank_id` could be embedded now: the bank's
    /// recorded embedding model is one the daemon serves. A refresh checks
    /// this before spending an LLM call on a plan it couldn't recall for.
    pub(crate) fn check_serving(&self, bank_id: i64) -> Result<(), RecallError> {
        let recorded = {
            let conn = self.store.connection();
            crate::reembed::recorded_model(&conn, bank_id)?
        };
        match crate::models::serving(self.models, self.previous, &recorded) {
            Some(_) => Ok(()),
            None => Err(RecallError::ModelUnavailable { model: recorded }),
        }
    }

    /// [`Context::embed_query`] for several queries in one embedder call.
    pub(crate) fn embed_queries(
        &self,
        bank_id: i64,
        queries: &[&str],
    ) -> Result<(Vec<Vec<f32>>, MutexGuard<'_, Connection>), RecallError> {
        for _ in 0..QUERY_EMBED_ATTEMPTS {
            let recorded = {
                let conn = self.store.connection();
                crate::reembed::recorded_model(&conn, bank_id)?
            };
            let embedder = crate::models::serving(self.models, self.previous, &recorded)
                .ok_or_else(|| RecallError::ModelUnavailable {
                    model: recorded.clone(),
                })?;
            let vectors = embedder
                .embed(queries)
                .map_err(|error| RecallError::Model { error })?;
            let conn = self.store.connection();
            if crate::reembed::recorded_model(&conn, bank_id)? == recorded {
                return Ok((vectors, conn));
            }
            tracing::debug!(
                bank_id,
                "a re-embed swapped the bank while its queries were embedded"
            );
        }
        Err(RecallError::ModelChanged)
    }

    /// Embeds `query` with the model `bank_id` is served with, and returns
    /// the vector together with the store's connection, held, under which
    /// that model is still the one the bank records. The model
    /// isn't run under the connection, so a re-embed's swap can land while
    /// it runs. That's caught when the connection is taken back, and the
    /// query embedded again with the new model, up to
    /// [`QUERY_EMBED_ATTEMPTS`]. A swap writes through the same connection,
    /// so none can come between the check and the vector search the caller
    /// runs on it. Refused when the daemon doesn't carry the bank's model.
    pub(crate) fn embed_query(
        &self,
        bank_id: i64,
        query: &str,
    ) -> Result<(Vec<f32>, MutexGuard<'_, Connection>), RecallError> {
        for _ in 0..QUERY_EMBED_ATTEMPTS {
            let recorded = {
                let conn = self.store.connection();
                crate::reembed::recorded_model(&conn, bank_id)?
            };
            let embedder = crate::models::serving(self.models, self.previous, &recorded)
                .ok_or_else(|| RecallError::ModelUnavailable {
                    model: recorded.clone(),
                })?;
            let vector = embedder
                .embed(&[query])
                .map_err(|error| RecallError::Model { error })?
                .pop()
                .unwrap_or_default();
            let conn = self.store.connection();
            if crate::reembed::recorded_model(&conn, bank_id)? == recorded {
                return Ok((vector, conn));
            }
            tracing::debug!(
                bank_id,
                "a re-embed swapped the bank while its query was embedded"
            );
        }
        Err(RecallError::ModelChanged)
    }
}

/// A prefetch's candidate as the gate saw it, before the gate: the
/// reranker logit the floor compares, or `None` when the reranker missed
/// its deadline. Replay's labelling material lists these.
#[derive(Debug, Clone, PartialEq)]
pub struct GateCandidate {
    pub memory: Uuid,
    pub sentence: String,
    pub logit: Option<f64>,
}

/// A prefetch with what the gate was shown: the query it recalled for,
/// after a short follow-up borrowed the previous one, and every reranked
/// candidate in ranked order.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredPrefetch {
    pub prefetch: Prefetch,
    /// The query the retrievers searched, which the recall log stores.
    pub query: String,
    /// The query the reranker scored against: `query`, or the conversation
    /// query.
    pub rerank_query: String,
    /// The message as it was sent, before [`clean_query`].
    pub raw_query: String,
    pub candidates: Vec<GateCandidate>,
}

/// Prefetch: recalls for the current message and injects what passes the
/// gate, after the session's agenda update if it has one, holding both as
/// the session's pending set. The candidates the gate was shown come back
/// beside it.
pub(crate) fn scored_prefetch(
    cx: &Context<'_>,
    bank: &str,
    request: &PrefetchRequest,
) -> Result<ScoredPrefetch, RecallError> {
    let started = Instant::now();
    let now = cx.store.now();
    let (bank_id, bank_tz) = find_bank(cx.store, bank)?;
    let in_context: BTreeSet<Uuid> = cx
        .sessions
        .in_context(bank_id, &request.session_id, now)
        .into_iter()
        .collect();
    let run = inject(
        cx,
        bank_id,
        &bank_tz,
        now,
        started,
        &Conversation {
            message: &request.query,
            previous_query: request.previous_query.as_deref(),
            previous_reply: request.previous_reply.as_deref(),
        },
        &in_context,
        Aside::Nothing,
    )?;
    // What this prefetch injects needn't be listed again.
    let mut seen = in_context.clone();
    seen.extend(&run.injected);
    let update = agenda_update(cx, bank_id, &bank_tz, now, &request.session_id, &seen)?;

    let logged: Vec<Logged> = run
        .ranked
        .iter()
        .zip(&run.cuts)
        .map(|(item, cut)| Logged {
            memory_id: item.candidate.id,
            score: item.score(),
            injected: cut.is_none(),
        })
        .collect();
    let shown = run
        .ranked
        .iter()
        .map(|item| GateCandidate {
            memory: item.candidate.uuid,
            sentence: item.candidate.content.clone(),
            logit: item.logit,
        })
        .collect();

    let recall_id = cx.store.new_id();
    log::write(
        &mut cx.store.connection(),
        &Entry {
            uuid: recall_id,
            bank_id,
            kind: RecallKind::Prefetch,
            session_id: Some(&request.session_id),
            query: &run.query,
            raw_query: Some(&request.query),
            latency_ms: elapsed_ms(started),
            at: now,
            results: &logged,
        },
    )?;
    let mut held = run.injected.clone();
    let text = match &update {
        Some(update) => {
            held.extend(&update.memories);
            if run.text.is_empty() {
                update.text.clone()
            } else {
                format!("{}\n\n{}", update.text, run.text)
            }
        }
        None => run.text,
    };
    cx.sessions.hold(
        bank_id,
        &request.session_id,
        recall_id,
        held,
        update.as_ref().map(|update| update.date),
        now,
    );
    tracing::debug!(
        bank = bank_id,
        recall = %recall_id,
        candidates = logged.len(),
        injected = run.injected.len(),
        agenda_update = update.as_ref().map(|update| update.memories.len()),
        reranked = run.reranked,
        "prefetched"
    );
    Ok(ScoredPrefetch {
        prefetch: Prefetch {
            recall_id,
            text,
            injected: run.injected,
            reranked: run.reranked,
        },
        query: run.query,
        rerank_query: run.rerank_query,
        raw_query: request.query.clone(),
        candidates: shown,
    })
}

/// The session's agenda update, when its block is mapped: the agenda is
/// current as of the later of the day the block was built for and the date
/// of the last update the session committed. A session with no mapping
/// gets none.
fn agenda_update(
    cx: &Context<'_>,
    bank_id: i64,
    bank_tz: &TimeZone,
    now: Timestamp,
    session_id: &str,
    in_context: &BTreeSet<Uuid>,
) -> Result<Option<crate::system_prompt::AgendaUpdate>, RecallError> {
    let expiry =
        jiff::SignedDuration::from_hours(24 * i64::from(cx.tuning.sessions.mapping_expiry_days));
    let conn = cx.store.connection();
    let Some(built_at) =
        crate::system_prompt::mapped_built_at(&conn, bank_id, session_id, now, expiry)?
    else {
        return Ok(None);
    };
    let built_for = built_at.to_zoned(bank_tz.clone()).date();
    let as_of = built_for.max(
        cx.sessions
            .agenda_date(bank_id, session_id, now)
            .unwrap_or(built_for),
    );
    Ok(crate::system_prompt::agenda_update(
        &conn, cx.tuning, bank_id, bank_tz, now, as_of, in_context,
    )?)
}

/// Explicit recall: the same pipeline with no τ gate, no in-context skip,
/// the recall weight, and the phase term only when the caller filters by
/// phase. Ended memories come back; retracted ones never do. The results
/// join the session's in-context set when a session is given.
pub(crate) fn recall(
    cx: &Context<'_>,
    bank: &str,
    request: &RecallRequest,
) -> Result<Recall, RecallError> {
    let started = Instant::now();
    let now = cx.store.now();
    let run = recall_run(cx, bank, now, started, request, Aside::Nothing)?;
    let mut ranked = run.ranked;
    ranked.truncate(run.limit);

    let strong_cutoff = cx.tuning.recall.strong_cutoff;
    let text = if ranked.is_empty() {
        "No memories recalled.".to_owned()
    } else {
        ranked
            .iter()
            .map(|item| format::recall_line(&item.candidate, now, strong_cutoff))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let logged: Vec<Logged> = ranked
        .iter()
        .map(|item| Logged {
            memory_id: item.candidate.id,
            score: item.score(),
            injected: false,
        })
        .collect();
    let results: Vec<Recalled> = ranked
        .into_iter()
        .map(|item| {
            let c = item.candidate;
            Recalled {
                id: c.uuid,
                sentence: c.content,
                kind: c.window.kind,
                window: RecalledWindow {
                    valid_from: c.window.valid_from,
                    valid_until: c.window.valid_until,
                    until_event: c.until_event,
                    due_at: c.window.due_at,
                    recurrence: c.recurrence_text,
                    uncertain: c.low_confidence,
                },
                phase: c.phase,
                observed_at: c.observed_at,
                strength: band(c.strength, strong_cutoff),
                kept: c.kept,
            }
        })
        .collect();

    let recall_id = cx.store.new_id();
    log::write(
        &mut cx.store.connection(),
        &Entry {
            uuid: recall_id,
            bank_id: run.bank_id,
            kind: RecallKind::Tool,
            session_id: request.session_id.as_deref(),
            query: &run.query,
            raw_query: None,
            latency_ms: elapsed_ms(started),
            at: now,
            results: &logged,
        },
    )?;
    if let Some(session_id) = &request.session_id {
        let ids: Vec<Uuid> = results.iter().map(|r| r.id).collect();
        cx.sessions.add(run.bank_id, session_id, &ids, now);
    }
    tracing::debug!(
        bank = run.bank_id,
        recall = %recall_id,
        results = results.len(),
        reranked = run.reranked,
        "recalled"
    );
    Ok(Recall {
        recall_id,
        results,
        text,
        reranked: run.reranked,
    })
}

/// Explain: [`recall`] or [`scored_prefetch`]'s pipeline with its working
/// shown and none of their side effects ([`explain`](self::explain)). An
/// injection runs as for a session with nothing in context.
pub(crate) fn explain(
    cx: &Context<'_>,
    bank: &str,
    request: &ExplainRequest,
) -> Result<Explain, RecallError> {
    let started = Instant::now();
    let now = cx.store.now();
    let strong_cutoff = cx.tuning.recall.strong_cutoff;
    match request {
        ExplainRequest::Recall(filters) => {
            let request = RecallRequest {
                session_id: None,
                query: filters.query.clone(),
                from: filters.from,
                to: filters.to,
                on: filters.on,
                phase: filters.phase,
                kinds: filters.kinds.clone(),
                entity: filters.entity.clone(),
                limit: filters.limit,
            };
            let run = recall_run(cx, bank, now, started, &request, Aside::Overflow)?;
            let candidates = run
                .ranked
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    let reason = (index >= run.limit).then_some(Cut::OverLimit);
                    explained_ranked(item, &run.gathered.arms, strong_cutoff, reason)
                })
                .chain(explained_overflow(&run.gathered, strong_cutoff))
                .collect();
            Ok(Explain {
                mode: ExplainMode::Recall,
                rerank_query: run.query.clone(),
                query: run.query,
                reranked: run.reranked,
                latency: run.gathered.latency(run.rerank, started),
                candidates,
                injection: None,
            })
        }
        ExplainRequest::Injection(conversation) => {
            let (bank_id, bank_tz) = find_bank(cx.store, bank)?;
            let run = inject(
                cx,
                bank_id,
                &bank_tz,
                now,
                started,
                &Conversation {
                    message: &conversation.query,
                    previous_query: conversation.previous_query.as_deref(),
                    previous_reply: conversation.previous_reply.as_deref(),
                },
                &BTreeSet::new(),
                Aside::OverflowAndRefused,
            )?;
            let gathered = &run.gathered;
            let mut candidates: Vec<Explained> = run
                .ranked
                .iter()
                .zip(&run.cuts)
                .map(|(item, cut)| explained_ranked(item, &gathered.arms, strong_cutoff, *cut))
                .chain(explained_overflow(gathered, strong_cutoff))
                .collect();
            candidates.extend(gathered.refused_candidates.iter().map(|candidate| {
                explained(
                    candidate,
                    arm_ranks(candidate.id, &gathered.refused, false),
                    None,
                    None,
                    None,
                    strong_cutoff,
                    Some(Cut::BelowTau),
                )
            }));
            Ok(Explain {
                mode: ExplainMode::Injection,
                query: run.query,
                rerank_query: run.rerank_query,
                reranked: run.reranked,
                latency: gathered.latency(run.rerank, started),
                candidates,
                injection: Some(ExplainedInjection {
                    tokens: estimate_tokens(&run.text),
                    text: run.text,
                    injected: run.injected,
                    floor: cx.reranking.floor,
                    cap: cx.tuning.injection.cap as usize,
                    token_budget: cx.tuning.injection.token_budget as usize,
                }),
            })
        }
    }
}

/// What a prefetch recalls for, as the plugin sends it.
struct Conversation<'a> {
    /// The user's message, before [`clean_query`].
    message: &'a str,
    previous_query: Option<&'a str>,
    previous_reply: Option<&'a str>,
}

/// An injection worked out, before anything is logged or held.
struct Injection {
    query: String,
    rerank_query: String,
    gathered: Gathered,
    /// Every reranked candidate, in ranked order.
    ranked: Vec<Ranked>,
    /// Why each of `ranked` wasn't injected, or `None` when it was.
    cuts: Vec<Option<Cut>>,
    text: String,
    injected: Vec<Uuid>,
    reranked: bool,
    rerank: Duration,
}

/// The injection for `conversation`, skipping the memories in
/// `in_context`, with no side effects. `aside` is what explain wants kept
/// besides.
#[allow(clippy::too_many_arguments)]
fn inject(
    cx: &Context<'_>,
    bank_id: i64,
    bank_tz: &TimeZone,
    now: Timestamp,
    started: Instant,
    conversation: &Conversation<'_>,
    in_context: &BTreeSet<Uuid>,
    aside: Aside,
) -> Result<Injection, RecallError> {
    let deadline = started + cx.deadline;
    let message = clean_query(conversation.message);
    let previous = conversation.previous_query.map(clean_query);
    let query = effective_query(&message, previous.as_deref());
    let rerank_query = match cx.tuning.injection.rerank_query {
        RerankQuery::Message => query.clone(),
        RerankQuery::Conversation => {
            conversation_query(&message, previous.as_deref(), conversation.previous_reply)
        }
    };

    let keep =
        |candidate: &Candidate| candidate.strength >= TAU && !in_context.contains(&candidate.uuid);
    let mut gathered = gather(cx, bank_id, &query, now, &keep, None, aside)?;
    let found = std::mem::take(&mut gathered.candidates);
    let documents = found.iter().map(|c| c.content.clone()).collect();
    let reranking = Instant::now();
    let logits = rerank::logits(
        &cx.models.reranker,
        cx.permit,
        &rerank_query,
        documents,
        deadline,
    );
    let rerank = reranking.elapsed();
    let reranked = logits.is_some();
    let ranking = &cx.tuning.ranking;
    let ranked = rank(found, logits, |candidate, logit| {
        let phase = phase_term(
            &candidate.window,
            candidate.low_confidence,
            &candidate.tz,
            now,
            ranking,
        );
        score_parts(
            cx.reranking.relevance(logit),
            ranking.w_s_inject,
            candidate.strength,
            candidate.state_confidence,
            phase,
        )
    });

    // The floor gates on the raw logit, not on relevance, so the floors
    // and the logits in the labelling material stay comparable.
    let floor = cx.reranking.floor;
    let cap = cx.tuning.injection.cap as usize;
    let budget = cx.tuning.injection.token_budget as usize;
    let header = format::header(now, bank_tz);
    let mut lines = Vec::new();
    let mut injected = Vec::new();
    let mut cuts = Vec::with_capacity(ranked.len());
    for item in &ranked {
        // Past the deadline, or when the reranker fails, there's no logit,
        // so nothing passes the gate and nothing is injected; the candidates
        // are still logged.
        let cut = match item.logit {
            None => Some(Cut::NotReranked),
            Some(logit) if logit < floor => Some(Cut::UnderFloor),
            Some(_) if injected.len() >= cap => Some(Cut::OverCap),
            Some(_) => {
                let mut with = lines.clone();
                with.push(format::line(&item.candidate, now));
                if estimate_tokens(&format::block(&header, &with)) <= budget {
                    lines = with;
                    injected.push(item.candidate.uuid);
                    None
                } else {
                    Some(Cut::OverBudget)
                }
            }
        };
        cuts.push(cut);
    }
    let text = if lines.is_empty() {
        String::new()
    } else {
        format::block(&header, &lines)
    };
    Ok(Injection {
        query,
        rerank_query,
        gathered,
        ranked,
        cuts,
        text,
        injected,
        reranked,
        rerank,
    })
}

/// An explicit recall worked out, before anything is logged or joins a
/// session.
struct RecallRun {
    bank_id: i64,
    query: String,
    gathered: Gathered,
    /// Every reranked candidate, in ranked order, past the limit too.
    ranked: Vec<Ranked>,
    limit: usize,
    reranked: bool,
    rerank: Duration,
}

/// Explicit recall's pipeline for `request`, ignoring its session, with no
/// side effects. `aside` is what explain wants kept besides.
fn recall_run(
    cx: &Context<'_>,
    bank: &str,
    now: Timestamp,
    started: Instant,
    request: &RecallRequest,
    aside: Aside,
) -> Result<RecallRun, RecallError> {
    let deadline = started + cx.deadline;
    if let (Some(from), Some(to)) = (request.from, request.to)
        && from > to
    {
        return Err(RecallError::InvertedRange);
    }
    let query = request.query.trim().to_owned();
    let (bank_id, _) = find_bank(cx.store, bank)?;
    let limit = request
        .limit
        .unwrap_or(RECALL_LIMIT_DEFAULT)
        .clamp(1, RECALL_LIMIT_MAX);

    let linked = match &request.entity {
        Some(entity) => Some(entity_memories(cx.store, bank_id, entity)?),
        None => None,
    };
    let keep = |candidate: &Candidate| {
        (request.kinds.is_empty() || request.kinds.contains(&candidate.window.kind))
            && linked
                .as_ref()
                .is_none_or(|ids| ids.contains(&candidate.id))
            && request.phase.admits(candidate.phase)
            && in_range(candidate, request)
    };
    let mut gathered = gather(cx, bank_id, &query, now, &keep, linked.as_ref(), aside)?;
    let found = std::mem::take(&mut gathered.candidates);
    let documents = found.iter().map(|c| c.content.clone()).collect();
    let reranking = Instant::now();
    let logits = rerank::logits(&cx.models.reranker, cx.permit, &query, documents, deadline);
    let rerank = reranking.elapsed();
    let reranked = logits.is_some();
    let ranking = &cx.tuning.ranking;
    let with_phase = request.phase != PhaseFilter::Any;
    let ranked = rank(found, logits, |candidate, logit| {
        let phase = if with_phase {
            phase_term(
                &candidate.window,
                candidate.low_confidence,
                &candidate.tz,
                now,
                ranking,
            )
        } else {
            0.0
        };
        score_parts(
            cx.reranking.relevance(logit),
            ranking.w_s_recall,
            candidate.strength,
            candidate.state_confidence,
            phase,
        )
    });
    Ok(RecallRun {
        bank_id,
        query,
        gathered,
        ranked,
        limit,
        reranked,
        rerank,
    })
}

/// [`score`] and the terms it sums.
fn score_parts(
    relevance: f64,
    w_s: f64,
    strength: f64,
    state_confidence: f64,
    phase_term: f64,
) -> ScoreParts {
    ScoreParts {
        relevance,
        w_s,
        strength_term: w_s * strength,
        confidence_term: state_confidence.ln().max(CONFIDENCE_TERM_MIN),
        phase_term,
        total: score(relevance, w_s, strength, state_confidence, phase_term),
    }
}

/// A ranked candidate as explain shows it.
fn explained_ranked(
    item: &Ranked,
    arms: &[Vec<i64>; 3],
    strong_cutoff: f64,
    reason: Option<Cut>,
) -> Explained {
    explained(
        &item.candidate,
        arm_ranks(item.candidate.id, arms, true),
        Some(item.rrf_rank),
        item.logit,
        item.parts,
        strong_cutoff,
        reason,
    )
}

/// The fused candidates past the rerank pool as explain shows them: with
/// their arm and RRF ranks, and nothing from the reranker.
fn explained_overflow(
    gathered: &Gathered,
    strong_cutoff: f64,
) -> impl Iterator<Item = Explained> + '_ {
    gathered
        .overflow
        .iter()
        .enumerate()
        .map(move |(index, candidate)| {
            explained(
                candidate,
                arm_ranks(candidate.id, &gathered.arms, true),
                Some(RERANKED + index + 1),
                None,
                None,
                strong_cutoff,
                Some(Cut::OutsideRerankPool),
            )
        })
}

fn explained(
    candidate: &Candidate,
    arms: Vec<ArmRank>,
    rrf_rank: Option<usize>,
    logit: Option<f64>,
    score: Option<ScoreParts>,
    strong_cutoff: f64,
    reason: Option<Cut>,
) -> Explained {
    Explained {
        id: candidate.uuid,
        sentence: candidate.content.clone(),
        kind: candidate.window.kind,
        phase: candidate.phase,
        arms,
        rrf_rank,
        logit,
        score,
        strength: band(candidate.strength, strong_cutoff),
        kept: candidate.kept,
        included: reason.is_none(),
        reason,
    }
}

/// The arms whose list holds `id`, with its 1-based place in each when
/// `ranked`.
fn arm_ranks(id: i64, lists: &[Vec<i64>; 3], ranked: bool) -> Vec<ArmRank> {
    [Arm::Vector, Arm::Bm25, Arm::Entity]
        .into_iter()
        .zip(lists)
        .filter_map(|(arm, list)| {
            let place = list.iter().position(|listed| *listed == id)?;
            Some(ArmRank {
                arm,
                rank: ranked.then_some(place + 1),
            })
        })
        .collect()
}

/// How long a refresh waits for the reranker, across all of its facets. A
/// refresh is a daemon job and never runs inside a request, so it can wait
/// its turn behind prefetch; a facet reranked past this scores on strength
/// alone, and the refresh goes on.
pub(crate) const REFRESH_RERANK_DEADLINE: Duration = Duration::from_secs(60);

/// A memory a mental model's refresh selected, with its score.
pub(crate) struct Selected {
    pub candidate: Candidate,
    pub score: f64,
}

/// What took a facet's candidate: the facet's budget, the cited fill past
/// it, a memory that asked for the refresh found relevant past it, or
/// neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Taken {
    Budget,
    Cited,
    Triggered,
    Cut,
}

/// A facet's candidate as a refresh's selection scored it, before the
/// facet's budget cut it: the raw reranker logit, or `None` when the facet
/// missed the reranker and scored on strength alone. Replay's labelling
/// material lists these.
#[derive(Debug, Clone, PartialEq)]
pub struct FacetCandidate {
    pub memory: Uuid,
    pub sentence: String,
    pub logit: Option<f64>,
    pub strength: f64,
    pub score: f64,
    /// Whether the model cites the memory now.
    pub cited: bool,
    pub taken: Taken,
    /// The handle the memory reached the write under, under whichever facet
    /// took it, or `None` when it isn't in the selection.
    pub input: Option<String>,
}

/// One facet's whole pool, in score order, and the query the reranker
/// scored it against.
#[derive(Debug, Clone, PartialEq)]
pub struct FacetPool {
    pub rerank_query: String,
    /// Whether the reranker scored the facet.
    pub reranked: bool,
    pub candidates: Vec<FacetCandidate>,
}

/// One facet's selection, and its whole pool when it was asked for.
pub(crate) struct FacetSelection {
    pub selected: Vec<Selected>,
    pub pool: Option<FacetPool>,
    /// The triggers the reranker found relevant to the facet. Each is in
    /// `selected`.
    pub related: Vec<i64>,
}

/// A refresh's selection, one facet per query: each query runs through the
/// pipeline with injection's weighting, over the memories `keep` admits.
/// Every fused candidate is reranked and scored, the best `budget` are
/// taken, and then the memories in `cited` that `keep` still admits, best
/// first, until there are `with_cited`.
///
/// `triggers` are the memories that asked for the refresh. Each one `keep`
/// admits is reranked against every facet with a query, found by its
/// retrievers or not, and one whose logit reaches the injection floor is
/// relevant to the facet: it's taken even past the budget, and listed in
/// `related`. One the retrievers didn't find that isn't relevant is dropped,
/// so judging it changes nothing else the facet takes or reports.
///
/// The queries are embedded in one call and their candidates cleaned up
/// in one pass over the bank, so a facet costs its search and its rerank.
/// Each result is in score order, and the recall log gets one `refresh` row
/// per query when `log` is set, with the query in `query` and `raw_query`
/// alike. The facets share `deadline`: one whose rerank misses it scores
/// every candidate as if its relevance were 0, so strength decides. With
/// `pools`, each facet's whole scored pool comes back too.
#[allow(clippy::too_many_arguments)]
pub(crate) fn select(
    cx: &Context<'_>,
    bank_id: i64,
    queries: &[&str],
    keep: &dyn Fn(&Candidate) -> bool,
    cited: &[i64],
    triggers: &[i64],
    budget: usize,
    with_cited: usize,
    log: bool,
    pools: bool,
    deadline: Instant,
) -> Result<Vec<FacetSelection>, RecallError> {
    let started = Instant::now();
    let now = cx.store.now();
    let queries: Vec<&str> = queries.iter().map(|query| query.trim()).collect();
    let asked: Vec<&str> = queries
        .iter()
        .copied()
        .filter(|query| !query.is_empty())
        .collect();
    // The vector searches run on the connection the queries' model was
    // checked under, so a re-embed's swap can't come between them.
    let (mut vectors, conn) = if asked.is_empty() {
        (Vec::new(), cx.store.connection())
    } else {
        cx.embed_queries(bank_id, &asked)?
    };
    vectors.reverse();
    let (found, trigger_heads) = {
        let conn = conn;
        let mut cleanup = Cleanup::new(&conn, bank_id, cx.tuning, now, keep)?;
        let cited_heads: BTreeSet<i64> = cleanup.list(cited)?.into_iter().collect();
        let trigger_heads: BTreeSet<i64> = cleanup.list(triggers)?.into_iter().collect();
        let mut found = Vec::with_capacity(queries.len());
        for query in &queries {
            let vector = if query.is_empty() {
                None
            } else {
                vectors.pop()
            };
            let mut ids = match &vector {
                Some(vector) => {
                    let lists = [
                        cleanup.list(&arms::vector(&conn, bank_id, vector, CANDIDATES_PER_ARM)?)?,
                        cleanup.list(&arms::bm25(&conn, bank_id, query, CANDIDATES_PER_ARM)?)?,
                        cleanup.list(&arms::entity(
                            &conn,
                            bank_id,
                            query,
                            vector,
                            CANDIDATES_PER_ARM,
                        )?)?,
                    ];
                    fuse(&[
                        lists[0].as_slice(),
                        lists[1].as_slice(),
                        lists[2].as_slice(),
                    ])
                }
                None => Vec::new(),
            };
            let searched = vector.is_some();
            for id in &cited_heads {
                if !ids.contains(id) {
                    ids.push(*id);
                }
            }
            // Only a facet with a query can judge a trigger.
            let mut judged = BTreeSet::new();
            for id in trigger_heads.iter().filter(|_| searched) {
                if !ids.contains(id) {
                    ids.push(*id);
                    judged.insert(*id);
                }
            }
            found.push((cleanup.take(&ids), cited_heads.clone(), judged));
        }
        (found, trigger_heads)
    };

    let mut selections = Vec::with_capacity(queries.len());
    for (query, (found, cited, judged)) in queries.iter().zip(found) {
        let selection = rank_facet(
            cx,
            query,
            found,
            &cited,
            &judged,
            &trigger_heads,
            budget,
            with_cited,
            pools,
            now,
            deadline,
        );
        if log {
            let logged: Vec<Logged> = selection
                .selected
                .iter()
                .map(|item| Logged {
                    memory_id: item.candidate.id,
                    score: Some(item.score),
                    injected: false,
                })
                .collect();
            log::write(
                &mut cx.store.connection(),
                &Entry {
                    uuid: cx.store.new_id(),
                    bank_id,
                    kind: RecallKind::Refresh,
                    session_id: None,
                    query,
                    raw_query: Some(query),
                    latency_ms: elapsed_ms(started),
                    at: now,
                    results: &logged,
                },
            )?;
        }
        selections.push(selection);
    }
    Ok(selections)
}

/// One facet's candidates reranked for `query` and scored, best first:
/// the best `budget`, then the `triggers` relevant to the facet, then the
/// memories in `cited`, best first, until there are `with_cited`. A
/// candidate in `judged`, there only to be judged, is dropped unless it's
/// relevant. With `pool`, every candidate scored comes back too, with what
/// took it.
#[allow(clippy::too_many_arguments)]
fn rank_facet(
    cx: &Context<'_>,
    query: &str,
    found: Vec<Candidate>,
    cited: &BTreeSet<i64>,
    judged: &BTreeSet<i64>,
    triggers: &BTreeSet<i64>,
    budget: usize,
    with_cited: usize,
    pool: bool,
    now: Timestamp,
    deadline: Instant,
) -> FacetSelection {
    let logits = if query.is_empty() {
        None
    } else {
        let documents = found.iter().map(|c| c.content.clone()).collect();
        rerank::logits(&cx.models.reranker, cx.permit, query, documents, deadline)
    };
    let ranking = &cx.tuning.ranking;
    let mut scored: Vec<(usize, Option<f64>, Selected)> = found
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| {
            let raw = logits
                .as_ref()
                .and_then(|logits| logits.get(index))
                .map(|logit| f64::from(*logit));
            let logit = raw.unwrap_or(0.0);
            let phase = phase_term(
                &candidate.window,
                candidate.low_confidence,
                &candidate.tz,
                now,
                ranking,
            );
            let score = score(
                cx.reranking.relevance(logit),
                ranking.w_s_inject,
                candidate.strength,
                candidate.state_confidence,
                phase,
            );
            (index, raw, Selected { candidate, score })
        })
        .collect();
    scored.sort_by(|(left_index, _, left), (right_index, _, right)| {
        right
            .score
            .total_cmp(&left.score)
            .then(left_index.cmp(right_index))
    });
    // A trigger is relevant to the facet where injection would find it
    // relevant to a message: the floor gates on the raw logit.
    let relevant = |id: i64, logit: Option<f64>| {
        triggers.contains(&id) && logit.is_some_and(|logit| logit >= cx.reranking.floor)
    };
    // A trigger the retrievers didn't find goes unless it's relevant.
    scored.retain(|(_, logit, item)| {
        !judged.contains(&item.candidate.id) || relevant(item.candidate.id, *logit)
    });
    let room = with_cited.saturating_sub(budget.min(scored.len()));
    let mut taken = Vec::with_capacity(scored.len());
    let mut related = Vec::new();
    let mut cited_taken = 0;
    for (place, (_, logit, item)) in scored.iter().enumerate() {
        let id = item.candidate.id;
        let relevant = relevant(id, *logit);
        if relevant {
            related.push(id);
        }
        taken.push(if place < budget {
            Taken::Budget
        } else if relevant {
            Taken::Triggered
        } else if cited.contains(&id) && cited_taken < room {
            cited_taken += 1;
            Taken::Cited
        } else {
            Taken::Cut
        });
    }
    let pool = pool.then(|| FacetPool {
        rerank_query: query.to_owned(),
        reranked: logits.is_some(),
        candidates: scored
            .iter()
            .zip(&taken)
            .map(|((_, logit, item), taken)| FacetCandidate {
                memory: item.candidate.uuid,
                sentence: item.candidate.content.clone(),
                logit: *logit,
                strength: item.candidate.strength,
                score: item.score,
                cited: cited.contains(&item.candidate.id),
                taken: *taken,
                input: None,
            })
            .collect(),
    });
    let selected = scored
        .into_iter()
        .zip(taken)
        .filter(|(_, taken)| *taken != Taken::Cut)
        .map(|((_, _, item), _)| item)
        .collect();
    FacetSelection {
        selected,
        pool,
        related,
    }
}

/// The memories linked to `entity` or to an entity merged into it, with the
/// head of each one's supersession chain, as a model's entity filter
/// admits them.
pub(crate) fn linked_memories(
    conn: &Connection,
    bank_id: i64,
    entity: i64,
) -> Result<BTreeSet<i64>, rusqlite::Error> {
    let mut statement = conn.prepare_cached(
        "SELECT DISTINCT me.memory_id FROM memory_entities me
         JOIN memories m ON m.id = me.memory_id
         JOIN entities e ON e.id = me.entity_id
         WHERE m.bank_id = ?1 AND (e.id = ?2 OR e.merged_into = ?2)",
    )?;
    let direct: Vec<i64> = statement
        .query_map((bank_id, entity), |row| row.get(0))?
        .collect::<Result<_, _>>()?;
    let mut links = conn.prepare_cached(
        "SELECT id, superseded_by, ended_by FROM memories
         WHERE bank_id = ?1 AND superseded_by IS NOT NULL",
    )?;
    let links: Vec<crate::strength::Link> = links
        .query_map([bank_id], |row| {
            Ok(crate::strength::Link {
                id: row.get(0)?,
                superseded_by: row.get(1)?,
                ended_by: row.get(2)?,
            })
        })?
        .collect::<Result<_, _>>()?;
    let mut chains = crate::strength::Chains::new(&links);
    let mut ids: BTreeSet<i64> = direct.iter().copied().collect();
    ids.extend(direct.iter().map(|id| chains.head(*id)));
    Ok(ids)
}

/// A candidate in its final place.
struct Ranked {
    candidate: Candidate,
    /// Its 1-based place in the fused list.
    rrf_rank: usize,
    logit: Option<f64>,
    /// `None` when the reranker was skipped.
    parts: Option<ScoreParts>,
}

impl Ranked {
    fn score(&self) -> Option<f64> {
        self.parts.map(|parts| parts.total)
    }
}

/// Orders the fused candidates: by score when there are logits, the RRF
/// order breaking ties, and in RRF order otherwise.
fn rank(
    found: Vec<Candidate>,
    logits: Option<Vec<f32>>,
    score: impl Fn(&Candidate, f64) -> ScoreParts,
) -> Vec<Ranked> {
    let Some(logits) = logits else {
        return found
            .into_iter()
            .enumerate()
            .map(|(index, candidate)| Ranked {
                candidate,
                rrf_rank: index + 1,
                logit: None,
                parts: None,
            })
            .collect();
    };
    let mut ranked: Vec<Ranked> = found
        .into_iter()
        .zip(logits)
        .enumerate()
        .map(|(index, (candidate, logit))| {
            let logit = f64::from(logit);
            let parts = score(&candidate, logit);
            Ranked {
                candidate,
                rrf_rank: index + 1,
                logit: Some(logit),
                parts: Some(parts),
            }
        })
        .collect();
    ranked.sort_by(|left, right| {
        let left_score = left.score().unwrap_or(f64::NEG_INFINITY);
        let right_score = right.score().unwrap_or(f64::NEG_INFINITY);
        right_score
            .total_cmp(&left_score)
            .then(left.rrf_rank.cmp(&right.rrf_rank))
    });
    ranked
}

/// What [`gather`] keeps aside for explain besides the candidates it
/// passes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Aside {
    Nothing,
    /// The fused candidates past the top [`RERANKED`].
    Overflow,
    /// The overflow, and the heads the filter refused.
    OverflowAndRefused,
}

/// Steps 1 to 3 and what explain shows of them.
#[derive(Default)]
struct Gathered {
    /// The top [`RERANKED`] fused candidates, in RRF order.
    candidates: Vec<Candidate>,
    /// The fused candidates past them, in RRF order, when asked for. They
    /// never reach the reranker.
    overflow: Vec<Candidate>,
    /// Each arm's list as fusion took it: vector, BM25, entity.
    arms: [Vec<i64>; 3],
    /// The heads each arm found that the filter refused, when asked for.
    refused: [Vec<i64>; 3],
    /// Their candidates, in RRF order over `refused`.
    refused_candidates: Vec<Candidate>,
    embed: Duration,
    retrieve: Duration,
}

impl Gathered {
    fn latency(&self, rerank: Duration, started: Instant) -> StageLatency {
        let ms = |duration: Duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
        StageLatency {
            embed_ms: ms(self.embed),
            retrieve_ms: ms(self.retrieve),
            rerank_ms: ms(rerank),
            total_ms: elapsed_ms(started),
        }
    }
}

/// Steps 1 to 3: the retrievers, clean-up and fusion, cut to the top
/// [`RERANKED`] in RRF order. `linked`, when given, also seeds the entity
/// arm with the memories of the recall tool's `entity`. `aside` says what
/// else to keep for explain.
fn gather(
    cx: &Context<'_>,
    bank_id: i64,
    query: &str,
    now: Timestamp,
    keep: &dyn Fn(&Candidate) -> bool,
    linked: Option<&BTreeSet<i64>>,
    aside: Aside,
) -> Result<Gathered, RecallError> {
    if query.is_empty() {
        return Ok(Gathered::default());
    }
    let started = Instant::now();
    // The vector search runs on the connection the query's model was
    // checked under, so a re-embed's swap can't come between them.
    let (vector, conn) = cx.embed_query(bank_id, query)?;
    let embed = started.elapsed();
    let mut cleanup = Cleanup::new(&conn, bank_id, cx.tuning, now, keep)?;
    let vector_hits = arms::vector(&conn, bank_id, &vector, CANDIDATES_PER_ARM)?;
    let bm25_hits = arms::bm25(&conn, bank_id, query, CANDIDATES_PER_ARM)?;
    let entity_hits = match linked {
        // The filter already names the entities; their memories by cosine
        // are the entity arm.
        Some(ids) => entity_arm_over(&conn, ids, &vector)?,
        None => arms::entity(&conn, bank_id, query, &vector, CANDIDATES_PER_ARM)?,
    };
    let hits = [vector_hits, bm25_hits, entity_hits];
    let lists = [
        cleanup.list(&hits[0])?,
        cleanup.list(&hits[1])?,
        cleanup.list(&hits[2])?,
    ];
    let mut fused = fuse(&[
        lists[0].as_slice(),
        lists[1].as_slice(),
        lists[2].as_slice(),
    ]);
    let overflow = match aside {
        Aside::Nothing => Vec::new(),
        Aside::Overflow | Aside::OverflowAndRefused => {
            cleanup.take(fused.get(RERANKED..).unwrap_or_default())
        }
    };
    fused.truncate(RERANKED);
    let candidates = cleanup.take(&fused);
    let (refused, refused_candidates) = if aside == Aside::OverflowAndRefused {
        let refused = hits.map(|hits| cleanup.refused(&hits));
        let order = fuse(&[
            refused[0].as_slice(),
            refused[1].as_slice(),
            refused[2].as_slice(),
        ]);
        let candidates = cleanup.take_refused(&order);
        (refused, candidates)
    } else {
        Default::default()
    };
    Ok(Gathered {
        candidates,
        overflow,
        arms: lists,
        refused,
        refused_candidates,
        embed,
        retrieve: started.elapsed().saturating_sub(embed),
    })
}

/// The entity arm when the caller already resolved the entities: the
/// memories in `ids`, by cosine to `vector`.
fn entity_arm_over(
    conn: &Connection,
    ids: &BTreeSet<i64>,
    vector: &[f32],
) -> Result<Vec<i64>, rusqlite::Error> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let list = ids
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut statement = conn.prepare(&format!(
        "SELECT m.id FROM memories m JOIN memory_vectors v ON v.memory_id = m.id
         WHERE m.id IN ({list})
         ORDER BY vec_distance_cosine(v.embedding, ?1), m.observed_at DESC, m.id
         LIMIT ?2"
    ))?;
    statement
        .query_map((bytes, CANDIDATES_PER_ARM as i64), |row| row.get(0))?
        .collect()
}

/// The memories linked to any entity `name` resolves to through the alias
/// FTS, seeded entities included, with every version of each in their
/// chains so a refined head still matches. Empty when it names none.
fn entity_memories(store: &Store, bank_id: i64, name: &str) -> Result<BTreeSet<i64>, RecallError> {
    let conn = store.connection();
    let entities = crate::extraction::entities_named(&conn, bank_id, &[name], &[])?;
    if entities.is_empty() {
        return Ok(BTreeSet::new());
    }
    let list = entities
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let mut statement = conn.prepare(&format!(
        "SELECT DISTINCT me.memory_id FROM memory_entities me JOIN memories m ON m.id = me.memory_id
         WHERE m.bank_id = ?1
           AND (me.entity_id IN ({list})
                OR me.entity_id IN (SELECT id FROM entities WHERE merged_into IN ({list})))"
    ))?;
    let direct: Vec<i64> = statement
        .query_map([bank_id], |row| row.get(0))?
        .collect::<Result<_, _>>()?;
    // A refined head carries its predecessors' links in spirit: count it
    // as linked when any version is.
    let mut links = conn.prepare_cached(
        "SELECT id, superseded_by, ended_by FROM memories
         WHERE bank_id = ?1 AND superseded_by IS NOT NULL",
    )?;
    let links: Vec<crate::strength::Link> = links
        .query_map([bank_id], |row| {
            Ok(crate::strength::Link {
                id: row.get(0)?,
                superseded_by: row.get(1)?,
                ended_by: row.get(2)?,
            })
        })?
        .collect::<Result<_, _>>()?;
    let mut chains = crate::strength::Chains::new(&links);
    let mut ids: BTreeSet<i64> = direct.iter().copied().collect();
    ids.extend(direct.iter().map(|id| chains.head(*id)));
    Ok(ids)
}

/// Whether `candidate` falls in the request's `from`/`to` range. With no range,
/// everything does.
fn in_range(candidate: &Candidate, request: &RecallRequest) -> bool {
    if request.from.is_none() && request.to.is_none() {
        return true;
    }
    let from = request.from.unwrap_or(Timestamp::MIN);
    let to = request.to.unwrap_or(Timestamp::MAX);
    match request.on {
        On::Said => from <= candidate.observed_at && candidate.observed_at <= to,
        On::Happened => {
            let window = &candidate.window;
            let tz = &candidate.tz;
            let widen = candidate.low_confidence;
            if window.kind == Kind::Fact && window.valid_until.is_none() {
                // A fact with no end holds from its start on; only a stated
                // start in the range places it there. An ended fact has a
                // bounded window and is matched like anything else.
                let Some(start) = window.valid_from else {
                    return false;
                };
                let (begins, ends) = unit(start, tz, widen);
                return begins <= to && ends > from;
            }
            // Either end can be open (CONTEXT.md): with no stated start,
            // there's no lower bound.
            let begins = window
                .valid_from
                .map(|start| unit(start, tz, widen).0)
                .unwrap_or(Timestamp::MIN);
            let ends = match window.closes_at(tz) {
                Some(closes) if widen => {
                    let precision = window
                        .valid_until
                        .or(window.valid_from)
                        .map(|time| time.precision)
                        .unwrap_or(TimePrecision::Day);
                    shift(closes, precision, tz, 1)
                }
                Some(closes) => closes,
                None => Timestamp::MAX,
            };
            begins <= to && ends > from
        }
    }
}

/// The span of `time`'s unit, `[start, end)`, widened by one unit at each
/// end when `widen`.
fn unit(time: WorldTime, tz: &TimeZone, widen: bool) -> (Timestamp, Timestamp) {
    let end = unit_end(time, tz);
    if widen {
        (
            shift(time.at, time.precision, tz, -1),
            shift(end, time.precision, tz, 1),
        )
    } else {
        (time.at, end)
    }
}

/// `at` moved by `units` of `precision` in `tz`, saturating.
fn shift(at: Timestamp, precision: TimePrecision, tz: &TimeZone, units: i64) -> Timestamp {
    let span: Span = match precision {
        TimePrecision::Year => units.years(),
        TimePrecision::Month => units.months(),
        TimePrecision::Day => units.days(),
        TimePrecision::Hour => units.hours(),
        TimePrecision::Minute => units.minutes(),
    };
    at.to_zoned(tz.clone())
        .checked_add(span)
        .map(|zoned| zoned.timestamp())
        .unwrap_or(if units < 0 {
            Timestamp::MIN
        } else {
            Timestamp::MAX
        })
}

fn find_bank(store: &Store, bank: &str) -> Result<(i64, TimeZone), RecallError> {
    let conn = store.connection();
    let (bank_id, timezone) =
        crate::ingest::find_bank(&conn, bank)?.ok_or(RecallError::UnknownBank)?;
    Ok((bank_id, TimeZone::get(&timezone).unwrap_or(TimeZone::UTC)))
}

fn elapsed_ms(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}
