//! `Tuning`: the daemon-wide settings with code defaults, read from an
//! optional TOML file.
//!
//! Every section and every key is optional; a missing one keeps its code
//! default. Unknown keys and out-of-range values are errors, and the daemon
//! refuses to start on one. A bank never overrides a tuning value. The replay
//! overrides file has exactly this shape, and replay layers code defaults,
//! then the production file, then the overrides ([`Tuning::from_layers`]).

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use jiff::civil::Time;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::constants::{Significance, TAU};

/// The daemon-wide tuning. `Default` gives the code defaults.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Tuning {
    pub clock: ClockTuning,
    pub purge: PurgeTuning,
    pub ranking: RankingTuning,
    pub recall: RecallTuning,
    pub injection: InjectionTuning,
    pub reconcile: ReconcileTuning,
    pub agenda: AgendaTuning,
    pub mental_models: MentalModelsTuning,
    pub sessions: SessionsTuning,
    pub llm: LlmTuning,
}

/// `[clock]`: bank time (ADR 0004).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClockTuning {
    /// The speed of bank time while a bank is quiet, as a fraction of full
    /// speed. The one strength constant the replay harness may tune.
    pub quiet_rate: f64,
}

impl Default for ClockTuning {
    fn default() -> Self {
        Self { quiet_rate: 0.1 }
    }
}

/// `[purge]`: the deletion policy (ADR 0008).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PurgeTuning {
    /// δ: a memory is purged once its strength falls this far below τ.
    /// `None` means never purge, written `delta = "never"` in TOML.
    #[serde(with = "delta")]
    pub delta: Option<f64>,

    /// Days after ingest before the sweep deletes the text of a source no
    /// memory rests on, and of failed chunks and recall rows.
    pub source_horizon_days: u32,
}

impl Default for PurgeTuning {
    fn default() -> Self {
        Self {
            delta: Some(1.0),
            source_horizon_days: 90,
        }
    }
}

/// `[ranking]`: the weights in the retrieval score:
///
/// ```text
/// score = relevance + w_s·strength + max(−3, ln(state_confidence)) + phase_term
/// ```
///
/// The defaults are opening values, to be tuned on the replay harness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RankingTuning {
    /// w_s in injection.
    pub w_s_inject: f64,

    /// w_s in explicit recall, at most `w_s_inject`: recall answers the
    /// question asked, so strength counts for less.
    pub w_s_recall: f64,

    /// The full phase bonus, for something starting now or overdue.
    pub phase_bonus: f64,

    /// The full phase penalty, for something ended a month or more ago. It's
    /// subtracted, so it's given as a positive number.
    pub phase_penalty: f64,
}

impl Default for RankingTuning {
    fn default() -> Self {
        Self {
            w_s_inject: 0.5,
            w_s_recall: 0.2,
            phase_bonus: 1.0,
            phase_penalty: 1.0,
        }
    }
}

/// `[recall]`: explicit recall.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RecallTuning {
    /// The strength above which a result is in the strong band. The faded
    /// band ends at τ by definition and isn't tunable.
    pub strong_cutoff: f64,
}

impl Default for RecallTuning {
    fn default() -> Self {
        Self { strong_cutoff: 0.3 }
    }
}

/// `[injection]`: relevance injection in prefetch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InjectionTuning {
    /// At most this many memories per injection.
    pub cap: u32,

    /// At most about this many tokens per injection.
    pub token_budget: u32,

    /// The relevance gate on the reranker logit, keyed by the exact reranker
    /// model string, quantisation included.
    pub reranker_floors: BTreeMap<String, f64>,
}

impl Default for InjectionTuning {
    fn default() -> Self {
        Self {
            cap: 8,
            token_budget: 600,
            reranker_floors: BTreeMap::new(),
        }
    }
}

/// `[reconcile]`: neighbour search in extraction.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReconcileTuning {
    /// The cosine floor that decides whether call 2 runs, keyed by the exact
    /// embedding model string, quantisation included.
    pub embedding_floors: BTreeMap<String, f64>,
}

/// `[agenda]`: the list in `system_prompt_block()` chosen by world time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgendaTuning {
    /// Events starting, and tasks due, within this many days are listed.
    pub horizon_days: u32,

    /// Overdue tasks are listed for this many days past their due date. It's
    /// also the guard that holds a task back from purge, so it's part of the
    /// deletion fingerprint.
    pub overdue_days: u32,

    /// The cap on dated lines.
    pub dated_lines: u32,

    /// The cap on routines.
    pub routines: u32,

    /// The cap on undated open tasks.
    pub undated_tasks: u32,
}

impl Default for AgendaTuning {
    fn default() -> Self {
        Self {
            horizon_days: 7,
            overdue_days: 30,
            dated_lines: 15,
            routines: 4,
            undated_tasks: 5,
        }
    }
}

/// `[mental_models]` (ADR 0007).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MentalModelsTuning {
    /// Tokens shared by every enabled model in `system_prompt_block()`.
    pub budget: u32,

    /// `max_tokens` for the seeded "User profile".
    pub profile_max_tokens: u32,

    /// The lowest significance of a new or changed memory that triggers a
    /// refresh.
    pub trigger_level: Significance,

    /// A refresh runs this many minutes after the last trigger...
    pub refresh_debounce_minutes: u32,

    /// ...and at most this many minutes after the first.
    pub refresh_max_delay_minutes: u32,

    /// The daily sweep's bank-local time.
    pub sweep_time: Time,

    /// Memories a refresh selects by score.
    pub input_budget: u32,

    /// The input cap once the memories the model cites now are added.
    pub input_budget_with_cited: u32,
}

impl Default for MentalModelsTuning {
    fn default() -> Self {
        Self {
            budget: 800,
            profile_max_tokens: 500,
            trigger_level: Significance::Notable,
            refresh_debounce_minutes: 5,
            refresh_max_delay_minutes: 30,
            sweep_time: Time::constant(4, 0, 0, 0),
            input_budget: 60,
            input_budget_with_cited: 70,
        }
    }
}

/// `[sessions]`: the daemon's session to block mapping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SessionsTuning {
    /// A mapping expires after this many days without a turn.
    pub mapping_expiry_days: u32,

    /// A session's in-context set and pending injection are dropped after
    /// this many days without a prefetch, recall or turn, on the service's
    /// clock. It's garbage collection only: compaction, reset and rewind
    /// already clear the set, and a thread resumed days later is the same
    /// Hermes session, so expiring early would re-inject what Hermes still
    /// holds.
    pub in_context_idle_days: u32,
}

impl Default for SessionsTuning {
    fn default() -> Self {
        Self {
            mapping_expiry_days: 30,
            in_context_idle_days: 7,
        }
    }
}

/// `[llm] auth`. `api_key` is the default and stays fully supported;
/// `chatgpt` is the subscription over the Codex backend, whose login lives
/// in a token file under the data dir and never in the tuning file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmAuth {
    #[default]
    ApiKey,
    Chatgpt,
}

/// `[llm]`: the one LLM for extraction, reconciliation and refresh. Its API
/// key is a secret and comes from the environment only; so does the
/// subscription's token file.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmTuning {
    /// How the LLM is authenticated: an API key (the default) or a ChatGPT
    /// subscription.
    pub auth: LlmAuth,

    /// The exact model string. Required in both modes: calibration runs
    /// against one model, and the subscription doesn't choose it.
    pub model: Option<String>,

    /// The base URL of the LLM's HTTP API.
    pub endpoint: Option<String>,

    /// The reasoning effort sent with every request, such as `"low"`.
    /// Unset leaves it to the backend's default. Part of what calibration
    /// is pinned to, like the model.
    pub reasoning_effort: Option<String>,
}

/// One TOML layer of tuning, named for error messages.
#[derive(Debug, Clone, Copy)]
pub struct Layer<'a> {
    /// Where the text came from, usually a path.
    pub origin: &'a str,
    pub text: &'a str,
}

/// Why a tuning couldn't be loaded. Any of these stops the daemon starting.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    // The underlying errors go in the message rather than as a `source`, so
    // anyhow's cause chain doesn't print them twice.
    #[error("reading {}: {error}", path.display())]
    Read {
        path: PathBuf,
        error: std::io::Error,
    },

    #[error("{origin}: {error}")]
    Parse {
        origin: String,
        error: toml::de::Error,
    },

    #[error("invalid tuning: {}", list(.0))]
    Invalid(Vec<InvalidValue>),
}

/// One out-of-range or missing value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidValue {
    /// The dotted key, such as `purge.delta`.
    pub key: String,
    pub reason: String,
}

impl fmt::Display for InvalidValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}` {}", self.key, self.reason)
    }
}

fn list(values: &[InvalidValue]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ")
}

impl Tuning {
    /// Reads the tuning file at `path`, or the code defaults when there's
    /// none, and validates the result.
    pub fn load(path: Option<&Path>) -> Result<Self, ConfigError> {
        let Some(path) = path else {
            return Ok(Self::default());
        };
        let text = std::fs::read_to_string(path).map_err(|error| ConfigError::Read {
            path: path.to_owned(),
            error,
        })?;
        Self::from_layers(&[Layer {
            origin: &path.display().to_string(),
            text: &text,
        }])
    }

    /// Parses one TOML document over the code defaults and validates it.
    pub fn from_toml(text: &str) -> Result<Self, ConfigError> {
        Self::from_layers(&[Layer {
            origin: "tuning",
            text,
        }])
    }

    /// Layers TOML documents over the code defaults, later ones winning key
    /// by key, then validates the result. Each layer must have the shape of
    /// `Tuning` on its own; range checks run once, on the merged result, so a
    /// later layer can fix a combination an earlier one left invalid.
    pub fn from_layers(layers: &[Layer<'_>]) -> Result<Self, ConfigError> {
        let mut merged = toml::Table::new();
        for layer in layers {
            let parse = |error| ConfigError::Parse {
                origin: layer.origin.to_owned(),
                error,
            };
            let table: toml::Table = toml::from_str(layer.text).map_err(parse)?;
            // Catch unknown keys and wrong types against this layer's name.
            toml::from_str::<Tuning>(layer.text).map_err(parse)?;
            merge(&mut merged, table);
        }
        let tuning: Tuning = merged.try_into().map_err(|error| ConfigError::Parse {
            origin: "merged tuning".to_owned(),
            error,
        })?;
        tuning.validate()?;
        Ok(tuning)
    }

    /// Checks every value is in range.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut errors = Vec::new();
        let mut fail = |key: &str, reason: String| {
            errors.push(InvalidValue {
                key: key.to_owned(),
                reason,
            })
        };

        let quiet_rate = self.clock.quiet_rate;
        if !(quiet_rate > 0.0 && quiet_rate <= 1.0) {
            fail(
                "clock.quiet_rate",
                format!("must be above 0 and at most 1, got {quiet_rate}"),
            );
        }

        if let Some(delta) = self.purge.delta
            && !(delta.is_finite() && delta >= 0.0)
        {
            fail(
                "purge.delta",
                format!("must be \"never\" or a number of at least 0, got {delta}"),
            );
        }
        if self.purge.source_horizon_days == 0 {
            fail("purge.source_horizon_days", "must be at least 1".into());
        }

        let ranking = &self.ranking;
        for (key, value) in [
            ("ranking.w_s_inject", ranking.w_s_inject),
            ("ranking.w_s_recall", ranking.w_s_recall),
            ("ranking.phase_bonus", ranking.phase_bonus),
            ("ranking.phase_penalty", ranking.phase_penalty),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                fail(key, format!("must be a number of at least 0, got {value}"));
            }
        }
        if ranking.w_s_recall > ranking.w_s_inject {
            fail(
                "ranking.w_s_recall",
                format!(
                    "must be at most ranking.w_s_inject ({}), got {}",
                    ranking.w_s_inject, ranking.w_s_recall
                ),
            );
        }

        let strong_cutoff = self.recall.strong_cutoff;
        if !(strong_cutoff.is_finite() && strong_cutoff > TAU) {
            fail(
                "recall.strong_cutoff",
                format!("must be above τ ({TAU}), got {strong_cutoff}"),
            );
        }

        if self.injection.cap == 0 {
            fail("injection.cap", "must be at least 1".into());
        }
        if self.injection.token_budget == 0 {
            fail("injection.token_budget", "must be at least 1".into());
        }
        for (model, floor) in &self.injection.reranker_floors {
            if model.is_empty() {
                fail("injection.reranker_floors", "has an empty model key".into());
            }
            if !floor.is_finite() {
                fail(
                    &format!("injection.reranker_floors.\"{model}\""),
                    format!("must be a finite logit, got {floor}"),
                );
            }
        }
        for (model, floor) in &self.reconcile.embedding_floors {
            if model.is_empty() {
                fail(
                    "reconcile.embedding_floors",
                    "has an empty model key".into(),
                );
            }
            if !(-1.0..=1.0).contains(floor) {
                fail(
                    &format!("reconcile.embedding_floors.\"{model}\""),
                    format!("must be a cosine between -1 and 1, got {floor}"),
                );
            }
        }

        let agenda = &self.agenda;
        for (key, value) in [
            ("agenda.horizon_days", agenda.horizon_days),
            ("agenda.overdue_days", agenda.overdue_days),
            ("agenda.dated_lines", agenda.dated_lines),
        ] {
            if value == 0 {
                fail(key, "must be at least 1".into());
            }
        }

        let models = &self.mental_models;
        for (key, value) in [
            ("mental_models.budget", models.budget),
            (
                "mental_models.profile_max_tokens",
                models.profile_max_tokens,
            ),
            (
                "mental_models.refresh_debounce_minutes",
                models.refresh_debounce_minutes,
            ),
            ("mental_models.input_budget", models.input_budget),
        ] {
            if value == 0 {
                fail(key, "must be at least 1".into());
            }
        }
        if models.profile_max_tokens > models.budget {
            fail(
                "mental_models.profile_max_tokens",
                format!(
                    "must fit in mental_models.budget ({}), got {}",
                    models.budget, models.profile_max_tokens
                ),
            );
        }
        if models.refresh_max_delay_minutes < models.refresh_debounce_minutes {
            fail(
                "mental_models.refresh_max_delay_minutes",
                format!(
                    "must be at least mental_models.refresh_debounce_minutes ({}), got {}",
                    models.refresh_debounce_minutes, models.refresh_max_delay_minutes
                ),
            );
        }
        if models.input_budget_with_cited < models.input_budget {
            fail(
                "mental_models.input_budget_with_cited",
                format!(
                    "must be at least mental_models.input_budget ({}), got {}",
                    models.input_budget, models.input_budget_with_cited
                ),
            );
        }

        if self.sessions.mapping_expiry_days == 0 {
            fail("sessions.mapping_expiry_days", "must be at least 1".into());
        }
        if self.sessions.in_context_idle_days == 0 {
            fail("sessions.in_context_idle_days", "must be at least 1".into());
        }

        if let Some(model) = &self.llm.model
            && model.trim().is_empty()
        {
            fail("llm.model", "must not be empty".into());
        }
        if let Some(endpoint) = &self.llm.endpoint {
            // Keep the raw spelling: URL parsing silently repairs missing
            // slashes and surrounding whitespace. Validation must not do so.
            let explicit_scheme =
                endpoint.starts_with("http://") || endpoint.starts_with("https://");
            let valid = explicit_scheme
                && endpoint.trim() == endpoint
                && url::Url::parse(endpoint).is_ok_and(|url| {
                    url.has_host()
                        && url.port() != Some(0)
                        && url.username().is_empty()
                        && url.password().is_none()
                        && url.query().is_none()
                        && url.fragment().is_none()
                });
            if !valid {
                // Neither the raw URL nor parser errors belong in startup
                // diagnostics: userinfo and queries can contain secrets.
                fail(
                    "llm.endpoint",
                    "must be an http:// or https:// URL with a host and a nonzero port, without credentials, query, fragment or surrounding whitespace".into(),
                );
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Invalid(errors))
        }
    }

    /// Checks there's a floor for each model the daemon runs. Floors are
    /// keyed by the exact model string, and there's no fallback: a missing
    /// gate floor would flood injection, and a missing reconcile floor would
    /// skip reconciliation.
    pub fn check_floors(
        &self,
        embedding_model: &str,
        reranker_model: &str,
    ) -> Result<(), ConfigError> {
        let mut errors = Vec::new();
        if !self
            .reconcile
            .embedding_floors
            .contains_key(embedding_model)
        {
            errors.push(InvalidValue {
                key: format!("reconcile.embedding_floors.\"{embedding_model}\""),
                reason: "is missing for the configured embedding model".into(),
            });
        }
        if !self.injection.reranker_floors.contains_key(reranker_model) {
            errors.push(InvalidValue {
                key: format!("injection.reranker_floors.\"{reranker_model}\""),
                reason: "is missing for the configured reranker model".into(),
            });
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Invalid(errors))
        }
    }
}

/// Merges `layer` into `base`, recursing into tables so a layer only
/// replaces the keys it names.
fn merge(base: &mut toml::Table, layer: toml::Table) {
    for (key, value) in layer {
        match (base.get_mut(&key), value) {
            (Some(toml::Value::Table(base)), toml::Value::Table(layer)) => merge(base, layer),
            (_, value) => {
                base.insert(key, value);
            }
        }
    }
}

/// `purge.delta`: a number of at least 0, or `"never"` for never purge. TOML
/// has no null, so the string stands in for it; JSON shows `null`.
mod delta {
    use super::*;

    const NEVER: &str = "never";

    pub fn serialize<S: Serializer>(delta: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error> {
        match delta {
            Some(delta) => serializer.serialize_f64(*delta),
            // Serde has no null-support query, and both TOML and JSON are
            // human-readable. Identify TOML serializers, including the
            // document serializer's strategy probe, to use the explicit sentinel
            // rather than silently omitting the key and restoring the default.
            None if std::any::type_name::<S>().starts_with("toml::") => {
                serializer.serialize_str(NEVER)
            }
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<f64>, D::Error> {
        struct Visitor;

        impl serde::de::Visitor<'_> for Visitor {
            type Value = Option<f64>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a number of at least 0, or \"{NEVER}\"")
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                Ok(Some(v))
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(Some(v as f64))
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(Some(v as f64))
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                if v == NEVER {
                    Ok(None)
                } else {
                    Err(E::invalid_value(serde::de::Unexpected::Str(v), &self))
                }
            }

            fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(None)
            }
        }

        deserializer.deserialize_any(Visitor)
    }
}
