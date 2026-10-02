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
//!    of the session.
//! 5. **Score.** [`score`]: relevance (the reranker logit), plus w_s times
//!    strength, plus the clamped log of state confidence, plus the phase
//!    term. w_s is `ranking.w_s_inject` or `ranking.w_s_recall`, and the
//!    phase term counts in injection, and in recall only when the caller
//!    filters by phase.
//!
//! Injection then gates on the reranker floor for the loaded model and
//! takes at most `injection.cap` memories in about `injection.token_budget`
//! tokens. Every recall writes one row to the recall log ([`log`]) and none
//! writes an access (ADR 0001). "Now" is always the service's clock.

mod arms;
pub(crate) mod candidates;
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
pub(crate) use rerank::Permit;

use crate::config::{RankingTuning, Tuning};
use crate::constants::{
    CANDIDATES_PER_ARM, CONFIDENCE_TERM_MIN, ENDED_GRACE_DAYS, OVERDUE_FULL_DAYS,
    OVERDUE_ZERO_DAYS, RECALL_LIMIT_DEFAULT, RECALL_LIMIT_MAX, RECENTLY_PAST_DAYS, RERANKED, RRF_K,
    SHORT_FOLLOW_UP_WORDS, TAU, UPCOMING_BONUS_DAYS,
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
    fn admits(self, phase: Phase) -> bool {
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
    /// can't be searched (ADR 0010). `asphodel reembed --bank` moves it to
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
/// `state_confidence` is 1.0 for anything but a state, so its term is 0.
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
    /// Embedding models the daemon carries besides its own, for banks a
    /// re-embed hasn't moved yet (ADR 0010).
    pub previous: &'a [Arc<dyn crate::models::Embedder>],
    pub sessions: &'a Sessions,
    /// The service's permit to run its reranker.
    pub permit: &'a Arc<Permit>,
    /// [`RERANKER_DEADLINE`](crate::constants::RERANKER_DEADLINE) unless a
    /// test or bench set another.
    pub deadline: Duration,
}

/// How many times a query is embedded before a recall gives up on a bank
/// whose model keeps changing under it: once, and once more after a swap.
const QUERY_EMBED_ATTEMPTS: usize = 2;

impl Context<'_> {
    /// Embeds `query` with the model `bank_id` is served with, and returns
    /// the vector together with the store's connection, held, under which
    /// that model is still the one the bank records (ADR 0010). The model
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
    pub query: String,
    pub candidates: Vec<GateCandidate>,
}

/// Prefetch: recalls for the current message and injects what passes the
/// gate, holding the injection as the session's pending set. The
/// candidates the gate was shown come back beside it.
pub(crate) fn scored_prefetch(
    cx: &Context<'_>,
    bank: &str,
    request: &PrefetchRequest,
) -> Result<ScoredPrefetch, RecallError> {
    let started = Instant::now();
    let deadline = started + cx.deadline;
    let now = cx.store.now();
    let query = effective_query(&request.query, request.previous_query.as_deref());
    let (bank_id, bank_tz) = find_bank(cx.store, bank)?;
    let in_context: BTreeSet<Uuid> = cx
        .sessions
        .in_context(bank_id, &request.session_id, now)
        .into_iter()
        .collect();

    let keep =
        |candidate: &Candidate| candidate.strength >= TAU && !in_context.contains(&candidate.uuid);
    let found = gather(cx, bank_id, &query, now, &keep, None)?;
    let documents = found.iter().map(|c| c.content.clone()).collect();
    let logits = rerank::logits(&cx.models.reranker, cx.permit, &query, documents, deadline);
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
        score(
            logit,
            ranking.w_s_inject,
            candidate.strength,
            candidate.state_confidence,
            phase,
        )
    });

    // There's no fallback floor (ADR 0009); a service built with models
    // always has one, and a missing one injects nothing.
    let floor = cx
        .tuning
        .injection
        .reranker_floors
        .get(cx.models.reranker.model_id())
        .copied()
        .unwrap_or(f64::INFINITY);
    let cap = cx.tuning.injection.cap as usize;
    let budget = cx.tuning.injection.token_budget as usize;
    let header = format::header(now, &bank_tz);
    let mut lines = Vec::new();
    let mut injected = Vec::new();
    let mut logged = Vec::with_capacity(ranked.len());
    let mut shown = Vec::with_capacity(ranked.len());
    for item in &ranked {
        // Past the deadline, or when the reranker fails, there's no logit,
        // so nothing passes the gate and nothing is injected; the candidates
        // are still logged.
        let passes = item.logit.is_some_and(|logit| logit >= floor);
        let mut take = false;
        if passes && injected.len() < cap {
            let line = format::line(&item.candidate, now);
            let mut with = lines.clone();
            with.push(line);
            if estimate_tokens(&format::block(&header, &with)) <= budget {
                lines = with;
                take = true;
            }
        }
        if take {
            injected.push(item.candidate.uuid);
        }
        logged.push(Logged {
            memory_id: item.candidate.id,
            score: item.score,
            injected: take,
        });
        shown.push(GateCandidate {
            memory: item.candidate.uuid,
            sentence: item.candidate.content.clone(),
            logit: item.logit,
        });
    }
    let text = if lines.is_empty() {
        String::new()
    } else {
        format::block(&header, &lines)
    };

    let recall_id = cx.store.new_id();
    log::write(
        &mut cx.store.connection(),
        &Entry {
            uuid: recall_id,
            bank_id,
            kind: RecallKind::Prefetch,
            session_id: Some(&request.session_id),
            query: &query,
            latency_ms: elapsed_ms(started),
            at: now,
            results: &logged,
        },
    )?;
    cx.sessions.hold(
        bank_id,
        &request.session_id,
        recall_id,
        injected.clone(),
        now,
    );
    tracing::debug!(
        bank = bank_id,
        recall = %recall_id,
        candidates = logged.len(),
        injected = injected.len(),
        reranked,
        "prefetched"
    );
    Ok(ScoredPrefetch {
        prefetch: Prefetch {
            recall_id,
            text,
            injected,
            reranked,
        },
        query,
        candidates: shown,
    })
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
    let deadline = started + cx.deadline;
    let now = cx.store.now();
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
    let found = gather(cx, bank_id, &query, now, &keep, linked.as_ref())?;
    let documents = found.iter().map(|c| c.content.clone()).collect();
    let logits = rerank::logits(&cx.models.reranker, cx.permit, &query, documents, deadline);
    let reranked = logits.is_some();
    let ranking = &cx.tuning.ranking;
    let with_phase = request.phase != PhaseFilter::Any;
    let mut ranked = rank(found, logits, |candidate, logit| {
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
        score(
            logit,
            ranking.w_s_recall,
            candidate.strength,
            candidate.state_confidence,
            phase,
        )
    });
    ranked.truncate(limit);

    let strong_cutoff = cx.tuning.recall.strong_cutoff;
    let logged: Vec<Logged> = ranked
        .iter()
        .map(|item| Logged {
            memory_id: item.candidate.id,
            score: item.score,
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
            bank_id,
            kind: RecallKind::Tool,
            session_id: request.session_id.as_deref(),
            query: &query,
            latency_ms: elapsed_ms(started),
            at: now,
            results: &logged,
        },
    )?;
    if let Some(session_id) = &request.session_id {
        let ids: Vec<Uuid> = results.iter().map(|r| r.id).collect();
        cx.sessions.add(bank_id, session_id, &ids, now);
    }
    tracing::debug!(
        bank = bank_id,
        recall = %recall_id,
        results = results.len(),
        reranked,
        "recalled"
    );
    Ok(Recall {
        recall_id,
        results,
        reranked,
    })
}

/// How long a refresh waits for the reranker. A refresh is a daemon job
/// and never runs inside a request, so it can wait its
/// turn behind prefetch; past this it scores on strength alone.
const REFRESH_RERANK_DEADLINE: Duration = Duration::from_secs(60);

/// A memory a mental model's refresh selected, with its score.
pub(crate) struct Selected {
    pub candidate: Candidate,
    pub score: f64,
}

/// A refresh's selection: the model's question runs
/// through the pipeline with injection's weighting, over the memories
/// `keep` admits. Every fused candidate is reranked and scored, the best
/// `budget` are taken, and then the memories in `cited` that `keep` still
/// admits, best first, until there are `with_cited`. Keeping cited
/// memories stops one that slips from 60th to 61st from leaving and coming
/// back on alternate refreshes.
///
/// The result is in score order, and the recall log gets one `refresh` row
/// when `log` is set. A missed reranker scores every candidate as if its
/// relevance were 0, so strength decides.
#[allow(clippy::too_many_arguments)]
pub(crate) fn select(
    cx: &Context<'_>,
    bank_id: i64,
    question: &str,
    keep: &dyn Fn(&Candidate) -> bool,
    cited: &[i64],
    budget: usize,
    with_cited: usize,
    log: bool,
) -> Result<Vec<Selected>, RecallError> {
    let started = Instant::now();
    let now = cx.store.now();
    let query = question.trim();
    // The vector search runs on the connection the query's model was
    // checked under, so a re-embed's swap can't come between them.
    let (vector, conn) = if query.is_empty() {
        (None, cx.store.connection())
    } else {
        let (vector, conn) = cx.embed_query(bank_id, query)?;
        (Some(vector), conn)
    };
    let (found, cited) = {
        let conn = conn;
        let mut cleanup = Cleanup::new(&conn, bank_id, cx.tuning.clock.quiet_rate, now, keep)?;
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
        let cited: BTreeSet<i64> = cleanup.list(cited)?.into_iter().collect();
        for id in &cited {
            if !ids.contains(id) {
                ids.push(*id);
            }
        }
        (cleanup.take(&ids), cited)
    };

    let logits = if query.is_empty() {
        None
    } else {
        let documents = found.iter().map(|c| c.content.clone()).collect();
        rerank::logits(
            &cx.models.reranker,
            cx.permit,
            query,
            documents,
            Instant::now() + REFRESH_RERANK_DEADLINE,
        )
    };
    let ranking = &cx.tuning.ranking;
    let mut scored: Vec<(usize, Selected)> = found
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| {
            let logit = logits
                .as_ref()
                .and_then(|logits| logits.get(index))
                .map_or(0.0, |logit| f64::from(*logit));
            let phase = phase_term(
                &candidate.window,
                candidate.low_confidence,
                &candidate.tz,
                now,
                ranking,
            );
            let score = score(
                logit,
                ranking.w_s_inject,
                candidate.strength,
                candidate.state_confidence,
                phase,
            );
            (index, Selected { candidate, score })
        })
        .collect();
    scored.sort_by(|(left_index, left), (right_index, right)| {
        right
            .score
            .total_cmp(&left.score)
            .then(left_index.cmp(right_index))
    });
    let mut selected = Vec::new();
    let mut extras = Vec::new();
    for (_, item) in scored {
        if selected.len() < budget {
            selected.push(item);
        } else if cited.contains(&item.candidate.id) {
            extras.push(item);
        }
    }
    let room = with_cited.saturating_sub(selected.len());
    selected.extend(extras.into_iter().take(room));

    if log {
        let logged: Vec<Logged> = selected
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
                latency_ms: elapsed_ms(started),
                at: now,
                results: &logged,
            },
        )?;
    }
    Ok(selected)
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
    logit: Option<f64>,
    /// `None` when the reranker was skipped.
    score: Option<f64>,
}

/// Orders the fused candidates: by score when there are logits, the RRF
/// order breaking ties, and in RRF order otherwise.
fn rank(
    found: Vec<Candidate>,
    logits: Option<Vec<f32>>,
    score: impl Fn(&Candidate, f64) -> f64,
) -> Vec<Ranked> {
    let Some(logits) = logits else {
        return found
            .into_iter()
            .map(|candidate| Ranked {
                candidate,
                logit: None,
                score: None,
            })
            .collect();
    };
    let mut ranked: Vec<(usize, Ranked)> = found
        .into_iter()
        .zip(logits)
        .enumerate()
        .map(|(index, (candidate, logit))| {
            let logit = f64::from(logit);
            let score = score(&candidate, logit);
            (
                index,
                Ranked {
                    candidate,
                    logit: Some(logit),
                    score: Some(score),
                },
            )
        })
        .collect();
    ranked.sort_by(|(left_index, left), (right_index, right)| {
        let left_score = left.score.unwrap_or(f64::NEG_INFINITY);
        let right_score = right.score.unwrap_or(f64::NEG_INFINITY);
        right_score
            .total_cmp(&left_score)
            .then(left_index.cmp(right_index))
    });
    ranked.into_iter().map(|(_, ranked)| ranked).collect()
}

/// Steps 1 to 3: the retrievers, clean-up and fusion, cut to the top
/// [`RERANKED`] in RRF order. `linked`, when given, also seeds the entity
/// arm with the memories of the recall tool's `entity`.
fn gather(
    cx: &Context<'_>,
    bank_id: i64,
    query: &str,
    now: Timestamp,
    keep: &dyn Fn(&Candidate) -> bool,
    linked: Option<&BTreeSet<i64>>,
) -> Result<Vec<Candidate>, RecallError> {
    if query.is_empty() {
        return Ok(Vec::new());
    }
    // The vector search runs on the connection the query's model was
    // checked under, so a re-embed's swap can't come between them.
    let (vector, conn) = cx.embed_query(bank_id, query)?;
    let mut cleanup = Cleanup::new(&conn, bank_id, cx.tuning.clock.quiet_rate, now, keep)?;
    let vector_hits = arms::vector(&conn, bank_id, &vector, CANDIDATES_PER_ARM)?;
    let bm25_hits = arms::bm25(&conn, bank_id, query, CANDIDATES_PER_ARM)?;
    let entity_hits = match linked {
        // The filter already names the entities; their memories by cosine
        // are the entity arm.
        Some(ids) => entity_arm_over(&conn, ids, &vector)?,
        None => arms::entity(&conn, bank_id, query, &vector, CANDIDATES_PER_ARM)?,
    };
    let lists = [
        cleanup.list(&vector_hits)?,
        cleanup.list(&bm25_hits)?,
        cleanup.list(&entity_hits)?,
    ];
    let mut fused = fuse(&[
        lists[0].as_slice(),
        lists[1].as_slice(),
        lists[2].as_slice(),
    ]);
    fused.truncate(RERANKED);
    Ok(cleanup.take(&fused))
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
