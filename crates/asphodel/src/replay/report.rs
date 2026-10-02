//! The JSON report (TIM-96, decision 7; `docs/replay.md`). A plain serde
//! value with keys in struct order and nothing from the wall clock, so two
//! runs of the same scenario compare byte for byte.

use asphodel_core::Tuning;
use jiff::Timestamp;
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Serialize)]
pub struct Report {
    /// `scripted` for a scenario; `live`, `replay` or `fast` for real
    /// history.
    pub kind: &'static str,
    pub scenario: String,
    pub group: &'static str,
    pub version: &'static str,
    pub git_sha: Option<&'static str>,
    /// The resolved tuning, every layer applied.
    pub tuning: Tuning,
    pub flags: Flags,
    pub probes: Vec<ProbeResult>,
    pub purges_per_day: Vec<DayCount>,
    pub purged_then_re_mentioned: ReMentioned,
    pub fade_outs_per_week: Vec<WeekCount>,
    pub bands_per_week: Vec<WeekBands>,
    pub extraction_lag: Lag,
    pub refresh_calls_per_day: Vec<DayCount>,
    pub llm: LlmCounts,
}

#[derive(Debug, Serialize)]
pub struct Flags {
    pub latency_ms: u64,
    pub until: Option<Timestamp>,
    /// How refreshes were answered: `scripted` (no edits) for a scenario.
    pub refresh: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ProbeResult {
    pub id: String,
    pub at: Timestamp,
    pub kind: &'static str,
    pub passed: bool,
    /// What the probe saw, in a shape per kind.
    pub observed: Value,
}

#[derive(Debug, Serialize)]
pub struct DayCount {
    /// A bank-local date.
    pub day: String,
    pub count: u64,
    /// The same number under the name the scripted tests read for purges.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub purged: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct WeekCount {
    /// An ISO week, such as `2026-W03`.
    pub week: String,
    pub count: u64,
}

#[derive(Debug, Serialize)]
pub struct WeekBands {
    pub week: String,
    pub strong: u64,
    pub fading: u64,
    pub faded: u64,
}

/// The shadow table's verdict (TIM-97, decision 7): purged rows, memories
/// created after a purge whose nearest shadow row is at or above the
/// reconcile floor, and the ratio.
#[derive(Debug, Serialize)]
pub struct ReMentioned {
    pub purged: u64,
    pub re_mentioned: u64,
    pub rate: f64,
}

/// Simulated time from a source's sync to its extraction's completion.
#[derive(Debug, Serialize)]
pub struct Lag {
    pub samples: u64,
    pub p50_ms: u64,
    pub p95_ms: u64,
}

/// Where each LLM reply came from.
#[derive(Debug, Serialize)]
pub struct LlmCounts {
    pub scripted: u64,
    pub cache: u64,
    pub top_up: u64,
    pub live: u64,
}
