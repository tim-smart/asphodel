//! The JSON report (`docs/replay.md`). A plain serde value with keys in struct
//! order and nothing from the wall clock, so two runs of the same scenario
//! compare byte for byte.
//!
//! [`Aggregate`] is the one thing that leaves the private directory: its only
//! string values are probe ids. Everything else is a number, a boolean or null,
//! and the hashes travel as byte arrays.

use std::collections::BTreeMap;

use asphodel_core::Tuning;
use asphodel_core::extraction::{CALL1_VERSION, guidance_hash};
use jiff::Timestamp;
use jiff::civil::Date;
use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Serialize)]
pub struct Report {
    /// `scripted` for a scenario; `live`, `replay` or `fast` for real
    /// history.
    pub kind: &'static str,
    /// The scenario's name, or the corpus file's stem.
    pub scenario: String,
    /// `ci` or `models` for a scenario; `fake` or `models` for real
    /// history, by the models the run used.
    pub group: &'static str,
    pub version: &'static str,
    pub git_sha: Option<&'static str>,
    /// SHA-256 of the corpus file; null for a scenario.
    pub corpus_hash: Option<String>,
    /// SHA-256 of the cassette as it stood when the run started; null for a
    /// scenario.
    pub cassette_hash: Option<String>,
    /// The resolved tuning, every layer applied.
    pub tuning: Tuning,
    /// Which call 1 prompt the run extracted with.
    pub call1: Call1,
    pub flags: Flags,
    pub probes: Vec<ProbeResult>,
    pub purges_per_day: Vec<DayCount>,
    pub purged_then_re_mentioned: ReMentioned,
    pub fade_outs_per_week: Vec<WeekCount>,
    pub bands_per_week: Vec<WeekBands>,
    pub extraction_lag: Lag,
    pub refresh_calls_per_day: Vec<DayCount>,
    pub injected_tokens: InjectedTokens,
    pub injection_usage: InjectionUsage,
    /// The tokens the mental models' entries hold, sampled daily.
    pub profile_tokens: Percentiles,
    pub call2_rate: Call2Rate,
    pub agenda_lines_per_day: Vec<DayCount>,
    /// Memories created, by the significance extraction gave them.
    pub significance_histogram: BTreeMap<String, u64>,
    /// Memories created, by kind.
    pub kind_histogram: BTreeMap<String, u64>,
    /// Every memory the run created, with when it faded and whether it was
    /// purged, so the A/B diff can name the ones that differ.
    pub memories: Vec<MemoryOutcome>,
    /// Where each LLM reply came from, and what the run identifies by:
    /// everything that differs between a `live` run and the `replay` of
    /// its cassette sits here.
    pub llm: LlmCounts,
}

/// Call 1's template version and the hash of the `[extraction] guidance`
/// its prompt carries, null without any: what a recording of call 1 is
/// keyed to, besides the model and language.
#[derive(Debug, Clone, Serialize)]
pub struct Call1 {
    pub version: u32,
    pub guidance_hash: Option<String>,
}

impl Call1 {
    pub fn of(tuning: &Tuning) -> Self {
        Self {
            version: CALL1_VERSION,
            guidance_hash: guidance_hash(tuning.extraction.guidance.as_deref()),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Flags {
    pub latency_ms: u64,
    pub until: Option<Timestamp>,
    /// How refreshes were answered: `scripted` (no edits) for a scenario;
    /// `request` (by request key) for `live` and `replay`; `live`,
    /// `recorded` or `off` for `fast`.
    pub refresh: &'static str,
    /// The recording mode of a real-history run; null for a scenario.
    pub mode: Option<&'static str>,
    pub no_cache: bool,
    pub self_test: bool,
    /// `--prime-concurrency`: how many call 1s at a time a primed `fast`
    /// run recorded before its simulation; null when it wasn't primed.
    pub prime_concurrency: Option<usize>,
    /// The pinned ONNX Runtime intra-op thread count, when real models ran.
    pub onnx_threads: Option<usize>,
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

/// The shadow table's verdict: purged rows, memories created after a purge
/// whose nearest shadow row is at or above the reconcile floor, and the ratio.
#[derive(Debug, Clone, Serialize)]
pub struct ReMentioned {
    pub purged: u64,
    pub re_mentioned: u64,
    pub rate: f64,
}

/// Simulated time from a source's sync to its extraction's completion.
#[derive(Debug, Clone, Serialize)]
pub struct Lag {
    pub samples: u64,
    pub p50_ms: u64,
    pub p95_ms: u64,
}

/// A distribution of whole numbers: how many, and the median and 95th
/// percentile.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Percentiles {
    pub samples: u64,
    pub p50: u64,
    pub p95: u64,
}

impl Percentiles {
    /// Nearest-rank percentiles of `values`, which it sorts.
    pub fn of(values: &mut [u64]) -> Self {
        values.sort_unstable();
        Self {
            samples: values.len() as u64,
            p50: percentile(values, 0.5),
            p95: percentile(values, 0.95),
        }
    }
}

/// The value at the nearest rank of `p` in sorted `values`, or 0 when
/// empty.
pub fn percentile(values: &[u64], p: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let index = ((values.len() as f64 - 1.0) * p).round() as usize;
    values[index.min(values.len() - 1)]
}

/// Injected tokens: per turn session, per turn, and cron prefetches apart so
/// they don't skew the percentiles.
#[derive(Debug, Default, Serialize)]
pub struct InjectedTokens {
    /// Every session with at least one synced turn, in session id order.
    pub sessions: Vec<SessionTokens>,
    /// Over the prefetches of synced turns.
    pub per_turn: Percentiles,
    pub cron: CronTokens,
}

#[derive(Debug, Serialize)]
pub struct SessionTokens {
    pub session: String,
    pub prefetches: u64,
    pub tokens: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct CronTokens {
    pub prefetches: u64,
    pub tokens: u64,
}

/// How often a chunk's claims landed near something stored and call 2 ran.
#[derive(Debug, Default, Serialize)]
pub struct Call2Rate {
    pub chunks: u64,
    pub call2: u64,
    pub rate: f64,
    /// Commits that found their chunk stale and reconciled it again, and
    /// that per chunk. Only above `[llm] concurrency = 1`, where a chunk
    /// can be.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redos: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redo_rate: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct MemoryOutcome {
    pub id: Uuid,
    pub created_at: Timestamp,
    /// The first instant strength fell below τ, by the access log at the
    /// end of the run; null when it hadn't by then, or the memory is gone.
    pub faded_at: Option<Timestamp>,
    pub purged_at: Option<Timestamp>,
}

/// Where each LLM reply came from, and what else only the run's mode
/// decides.
#[derive(Debug, Default, Clone, Serialize)]
pub struct LlmCounts {
    pub scripted: u64,
    pub cache: u64,
    pub top_up: u64,
    pub live: u64,
    /// Call 1s the prime recorded before the simulation, counted here
    /// alone: not as live calls or misses.
    pub primed: u64,
    /// Calls the cassette couldn't answer. A `replay` run stops at the
    /// first; `live` and `fast` call the LLM instead.
    pub misses: u64,
    pub used_verdicts: UsedVerdicts,
    /// The measured round trip of each live call, in `live` mode.
    pub latency_ms: Percentiles,
}

/// Where each `used` verdict came from: the recording, a top-up call, the live
/// call that judged the whole chunk, or nowhere.
#[derive(Debug, Default, Clone, Serialize)]
pub struct UsedVerdicts {
    pub recorded: u64,
    pub top_up: u64,
    pub live: u64,
    pub none: u64,
}

/// Per-memory outcomes across committed extraction chunks. Unjudged
/// memories do not enter the fraction's denominator.
#[derive(Debug, Default, Clone, Serialize)]
pub struct InjectionUsage {
    pub used: u64,
    pub not_used: u64,
    pub unjudged: u64,
    /// Used / (used + not_used), or 0.0 when nothing was judged.
    pub used_fraction: f64,
}

// The aggregate export.

/// What may leave the private directory: probe ids and numbers. No field is a
/// string but a probe's id; dates are days since the epoch and weeks are two
/// integers; hashes are byte arrays.
#[derive(Debug, Serialize)]
pub struct Aggregate {
    pub mode: Mode,
    pub corpus_hash: Option<Vec<u8>>,
    pub cassette_hash: Option<Vec<u8>>,
    pub git_sha: Option<Vec<u8>>,
    pub call1: AggregateCall1,
    pub latency_ms: u64,
    pub prime_concurrency: Option<u64>,
    pub probes: Vec<ProbeOutcome>,
    pub probes_passed: u64,
    pub probes_failed: u64,
    pub purges_per_day: Vec<EpochDayCount>,
    pub purged_then_re_mentioned: ReMentioned,
    pub fade_outs_per_week: Vec<WeekNumberCount>,
    pub bands_per_week: Vec<WeekNumberBands>,
    pub extraction_lag: Lag,
    pub refresh_calls_per_day: Vec<EpochDayCount>,
    pub injected_tokens: AggregateTokens,
    pub injection_usage: InjectionUsage,
    pub profile_tokens: Percentiles,
    pub call2_rate: Call2Rate,
    pub agenda_lines_per_day: Vec<EpochDayCount>,
    pub significance_histogram: BTreeMap<String, u64>,
    pub kind_histogram: BTreeMap<String, u64>,
    pub memories: MemoryCounts,
    pub llm: LlmCounts,
}

/// [`Call1`] with the hash as bytes.
#[derive(Debug, Serialize)]
pub struct AggregateCall1 {
    pub version: u32,
    pub guidance_hash: Option<Vec<u8>>,
}

/// The run's kind as flags, since an enum name would be a string.
#[derive(Debug, Default, Serialize)]
pub struct Mode {
    pub scripted: bool,
    pub live: bool,
    pub replay: bool,
    pub fast: bool,
}

#[derive(Debug, Serialize)]
pub struct ProbeOutcome {
    pub id: String,
    pub passed: bool,
}

#[derive(Debug, Serialize)]
pub struct EpochDayCount {
    /// Days since 1970-01-01, of the bank-local date.
    pub epoch_day: i64,
    pub count: u64,
}

#[derive(Debug, Serialize)]
pub struct WeekNumberCount {
    pub iso_year: i64,
    pub iso_week: i64,
    pub count: u64,
}

#[derive(Debug, Serialize)]
pub struct WeekNumberBands {
    pub iso_year: i64,
    pub iso_week: i64,
    pub strong: u64,
    pub fading: u64,
    pub faded: u64,
}

#[derive(Debug, Serialize)]
pub struct AggregateTokens {
    pub sessions: u64,
    /// Over the sessions' totals.
    pub per_session: Percentiles,
    pub per_turn: Percentiles,
    pub cron: CronTokens,
}

#[derive(Debug, Serialize)]
pub struct MemoryCounts {
    pub created: u64,
    pub faded: u64,
    pub purged: u64,
}

impl Aggregate {
    pub fn from_report(report: &Report) -> Self {
        let mode = Mode {
            scripted: report.kind == "scripted",
            live: report.kind == "live",
            replay: report.kind == "replay",
            fast: report.kind == "fast",
        };
        let day = |count: &DayCount| EpochDayCount {
            epoch_day: epoch_day(&count.day),
            count: count.count,
        };
        let mut per_session: Vec<u64> = report
            .injected_tokens
            .sessions
            .iter()
            .map(|session| session.tokens)
            .collect();
        Self {
            mode,
            corpus_hash: report.corpus_hash.as_deref().map(hex_bytes),
            cassette_hash: report.cassette_hash.as_deref().map(hex_bytes),
            git_sha: report.git_sha.map(hex_bytes),
            call1: AggregateCall1 {
                version: report.call1.version,
                guidance_hash: report.call1.guidance_hash.as_deref().map(hex_bytes),
            },
            latency_ms: report.flags.latency_ms,
            prime_concurrency: report.flags.prime_concurrency.map(|n| n as u64),
            probes: report
                .probes
                .iter()
                .map(|probe| ProbeOutcome {
                    id: probe.id.clone(),
                    passed: probe.passed,
                })
                .collect(),
            probes_passed: report.probes.iter().filter(|probe| probe.passed).count() as u64,
            probes_failed: report.probes.iter().filter(|probe| !probe.passed).count() as u64,
            purges_per_day: report.purges_per_day.iter().map(day).collect(),
            purged_then_re_mentioned: report.purged_then_re_mentioned.clone(),
            fade_outs_per_week: report
                .fade_outs_per_week
                .iter()
                .map(|week| {
                    let (iso_year, iso_week) = iso_week(&week.week);
                    WeekNumberCount {
                        iso_year,
                        iso_week,
                        count: week.count,
                    }
                })
                .collect(),
            bands_per_week: report
                .bands_per_week
                .iter()
                .map(|week| {
                    let (iso_year, iso_week) = iso_week(&week.week);
                    WeekNumberBands {
                        iso_year,
                        iso_week,
                        strong: week.strong,
                        fading: week.fading,
                        faded: week.faded,
                    }
                })
                .collect(),
            extraction_lag: report.extraction_lag.clone(),
            refresh_calls_per_day: report.refresh_calls_per_day.iter().map(day).collect(),
            injected_tokens: AggregateTokens {
                sessions: report.injected_tokens.sessions.len() as u64,
                per_session: Percentiles::of(&mut per_session),
                per_turn: report.injected_tokens.per_turn.clone(),
                cron: CronTokens {
                    prefetches: report.injected_tokens.cron.prefetches,
                    tokens: report.injected_tokens.cron.tokens,
                },
            },
            profile_tokens: report.profile_tokens.clone(),
            injection_usage: report.injection_usage.clone(),
            call2_rate: Call2Rate {
                chunks: report.call2_rate.chunks,
                call2: report.call2_rate.call2,
                rate: report.call2_rate.rate,
                redos: report.call2_rate.redos,
                redo_rate: report.call2_rate.redo_rate,
            },
            agenda_lines_per_day: report.agenda_lines_per_day.iter().map(day).collect(),
            significance_histogram: report.significance_histogram.clone(),
            kind_histogram: report.kind_histogram.clone(),
            memories: MemoryCounts {
                created: report.memories.len() as u64,
                faded: report
                    .memories
                    .iter()
                    .filter(|memory| memory.faded_at.is_some())
                    .count() as u64,
                purged: report
                    .memories
                    .iter()
                    .filter(|memory| memory.purged_at.is_some())
                    .count() as u64,
            },
            llm: report.llm.clone(),
        }
    }
}

/// Days since the epoch of a `YYYY-MM-DD` date; 0 when it doesn't parse.
fn epoch_day(day: &str) -> i64 {
    day.parse::<Date>()
        .ok()
        .and_then(|date| {
            date.since((jiff::Unit::Day, Date::constant(1970, 1, 1)))
                .ok()
        })
        .map(|span| span.get_days().into())
        .unwrap_or(0)
}

/// The year and week of a `YYYY-Www` ISO week; zeros when it doesn't parse.
fn iso_week(week: &str) -> (i64, i64) {
    let Some((year, week)) = week.split_once("-W") else {
        return (0, 0);
    };
    (year.parse().unwrap_or(0), week.parse().unwrap_or(0))
}

/// The bytes of a hex string; a string that isn't hex gives an empty array
/// rather than a string.
fn hex_bytes(hex: &str) -> Vec<u8> {
    let digits = hex.as_bytes();
    if !digits.len().is_multiple_of(2) {
        return Vec::new();
    }
    digits
        .chunks(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).unwrap_or("zz");
            u8::from_str_radix(text, 16)
        })
        .collect::<Result<Vec<u8>, _>>()
        .unwrap_or_default()
}
