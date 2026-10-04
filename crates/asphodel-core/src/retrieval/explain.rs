//! Explain: one query through the recall or the injection pipeline, with
//! the working shown and no side effects. It's what the dashboard's Recall
//! page calls, through `POST /v1/banks/{bank}/recall/explain`.
//!
//! It runs the pipeline [`recall`](super::recall) and
//! [`scored_prefetch`](super::scored_prefetch) run, through the same code,
//! so it can't drift from them. It writes no recall row and no access, and
//! it reads and writes no session state: there's no session, so in
//! injection mode nothing counts as already in context.
//!
//! For the same bank state and query, recall mode includes the memories
//! `/recall` returns in the same order, and injection mode injects what a
//! prefetch for a session with nothing in context would.

use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{Band, On, PhaseFilter};
use crate::strength::{Kind, Phase};

/// What explain takes, tagged by `mode`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ExplainRequest {
    Recall(ExplainRecall),
    Injection(ExplainInjection),
}

/// Recall mode: [`RecallRequest`](super::RecallRequest)'s query and
/// filters, without a session.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ExplainRecall {
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
    #[serde(default)]
    pub entity: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Injection mode: [`PrefetchRequest`](super::PrefetchRequest)'s message
/// and conversation, without a session or a block.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ExplainInjection {
    /// The user's message, cleaned as prefetch cleans it.
    pub query: String,
    #[serde(default)]
    pub previous_query: Option<String>,
    #[serde(default)]
    pub previous_reply: Option<String>,
}

/// What explain returns.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Explain {
    pub mode: ExplainMode,
    /// The query the retrievers searched: for injection, after cleaning and
    /// a short follow-up's borrowing.
    pub query: String,
    /// The query the reranker scored against.
    pub rerank_query: String,
    /// Whether the reranker answered within the deadline.
    pub reranked: bool,
    pub latency: StageLatency,
    /// The reranked candidates in their final order, then, in injection
    /// mode, the heads the arms found below τ, in RRF order among
    /// themselves. Recall mode doesn't list what its filters dropped.
    pub candidates: Vec<Explained>,
    /// Injection mode only.
    pub injection: Option<ExplainedInjection>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExplainMode {
    Recall,
    Injection,
}

/// How long each stage took, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StageLatency {
    /// Embedding the query.
    pub embed_ms: u64,
    /// The arms, clean-up and fusion.
    pub retrieve_ms: u64,
    /// Waiting for the reranker, up to the deadline.
    pub rerank_ms: u64,
    pub total_ms: u64,
}

/// One candidate and what the pipeline made of it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Explained {
    pub id: Uuid,
    pub sentence: String,
    pub kind: Kind,
    pub phase: Phase,
    /// The arms that found it, in the order vector, BM25, entity.
    pub arms: Vec<ArmRank>,
    /// Its 1-based place in the fused list; `None` below τ.
    pub rrf_rank: Option<usize>,
    /// The raw reranker logit; `None` below τ or when the reranker missed.
    pub logit: Option<f64>,
    /// `None` wherever `logit` is.
    pub score: Option<ScoreParts>,
    pub strength: Band,
    pub kept: bool,
    /// Returned by recall, or injected.
    pub included: bool,
    /// Why it wasn't included; `None` when it was.
    pub reason: Option<Cut>,
}

/// An arm that found a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ArmRank {
    pub arm: Arm,
    /// Its 1-based place in the arm's list as fusion took it, after
    /// clean-up and the mode's filter; `None` below τ, where it never
    /// reached fusion.
    pub rank: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Vector,
    Bm25,
    Entity,
}

/// The parts of [`score`](super::score), which sum to `total`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ScoreParts {
    /// The logit divided by the reranker's relevance scale.
    pub relevance: f64,
    /// `ranking.w_s_inject` or `ranking.w_s_recall`.
    pub w_s: f64,
    /// w_s × strength.
    pub strength_term: f64,
    /// The clamped log of state confidence.
    pub confidence_term: f64,
    /// 0 in recall mode unless it filters by phase.
    pub phase_term: f64,
    pub total: f64,
}

/// Why a candidate wasn't included.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cut {
    /// Injection: strength below τ, so it never reached fusion.
    BelowTau,
    /// Injection: the reranker missed its deadline, so nothing passes.
    NotReranked,
    /// Injection: the logit is under the loaded reranker's floor.
    UnderFloor,
    /// Injection: `injection.cap` memories were already taken.
    OverCap,
    /// Injection: its line would take the block past
    /// `injection.token_budget`.
    OverBudget,
    /// Recall: ranked past the limit.
    OverLimit,
}

/// What injection would put in the prompt.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExplainedInjection {
    /// The block exactly as prefetch returns it; empty when nothing passed.
    pub text: String,
    /// [`estimate_tokens`](super::estimate_tokens) of `text`.
    pub tokens: usize,
    /// In the order they're listed.
    pub injected: Vec<Uuid>,
    /// The raw-logit gate for the loaded reranker.
    pub floor: f64,
    pub cap: usize,
    pub token_budget: usize,
}
