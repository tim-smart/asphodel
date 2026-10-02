//! The scenario file, as `docs/replay.md` gives it (TIM-96, decisions 2 and
//! 5).
//!
//! The enums for kinds, significance, outcomes, bands and phases are the
//! production ones, so the file's words are exactly the API's. Only the
//! time precision is defined here, since extraction keeps its own private.
//! [`check`] is what the loader refuses before a run; the engine refuses
//! the rest as it goes, since whether a target is a neighbour or a `used`
//! memory is in context is only known then.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Context as _;
use asphodel_core::constants::{Significance, Volatility};
use asphodel_core::extraction::Label;
use asphodel_core::retrieval::Band;
use asphodel_core::strength::{Kind, Phase};
use jiff::civil::Date;
use jiff::{SignedDuration, Span, SpanRelativeTo, Timestamp};
use serde::{Deserialize, Serialize};

/// The prefix of the sessions the prefetch probes run on, `probe:<id>`.
/// No scenario session may start with it, so a probe never touches a
/// scenario session's pending injections or idle timeout.
pub const PROBE_SESSION_PREFIX: &str = "probe:";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub name: String,
    pub group: Group,
    #[serde(default)]
    pub description: Option<String>,
    /// The simulated extraction latency, such as `10m`; `0s` when absent.
    /// `--latency` overrides it.
    #[serde(default)]
    pub latency: Option<String>,
    #[serde(default)]
    pub bank: Option<BankSection>,
    /// A layer in the shape of `Tuning`, above `--config` and below
    /// `--overrides`.
    #[serde(default)]
    pub tuning: Option<toml::Table>,
    /// Mental models created in the bank before the first event, so the
    /// refresh schedule has something to run.
    #[serde(default, rename = "model")]
    pub models: Vec<ModelSection>,
    #[serde(default, rename = "turn")]
    pub turns: Vec<Turn>,
    #[serde(default)]
    pub chatter: Vec<Chatter>,
    #[serde(default, rename = "document")]
    pub documents: Vec<Document>,
    #[serde(default, rename = "clear")]
    pub clears: Vec<Clear>,
    #[serde(default, rename = "probe")]
    pub probes: Vec<Probe>,
}

/// `ci` runs on the fake models in CI; `models` needs the real ones in
/// `ASPHODEL_MODEL_DIR`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Group {
    Ci,
    Models,
}

impl Group {
    pub fn as_str(self) -> &'static str {
        match self {
            Group::Ci => "ci",
            Group::Models => "models",
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BankSection {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub timezone: Option<String>,
    #[serde(default)]
    pub owner: Option<String>,
    #[serde(default)]
    pub assistant: Option<String>,
    #[serde(default)]
    pub owner_platform_ids: Vec<String>,
}

/// A mental model to create before the run. The scripted LLM answers its
/// refreshes with no edits, so its entries stay empty; the schedule, the
/// triggers and the calls per day are what a scenario can watch. A
/// real-history manifest's `[[model]]` has the same shape, and the corpus
/// header carries it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSection {
    pub name: String,
    pub question: String,
    #[serde(default)]
    pub kinds: Vec<Kind>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

fn default_max_tokens() -> u32 {
    500
}

/// One user message and the assistant's reply. Prefetch runs at `at`,
/// `sync_turn` at `reply_at` (or `at`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Turn {
    pub at: Timestamp,
    #[serde(default)]
    pub reply_at: Option<Timestamp>,
    pub session: String,
    pub user: String,
    pub assistant: String,
    #[serde(default)]
    pub author: Option<Author>,
    #[serde(default)]
    pub platform: Option<String>,
    #[serde(default, rename = "claim")]
    pub claims: Vec<Claim>,
    /// Labels of memories the reply relied on. Each must be in the
    /// session's in-context set at the turn, or the run is a scenario
    /// error.
    #[serde(default)]
    pub used: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Author {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
}

/// A run of turns with nothing to extract, to keep the bank in
/// conversation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Chatter {
    pub from: Timestamp,
    /// A duration such as `1d` or `12h`.
    pub every: String,
    pub count: u32,
    #[serde(default)]
    pub session: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Document {
    pub at: Timestamp,
    pub id: String,
    pub text: String,
    pub reference_date: Date,
    #[serde(default)]
    pub timezone: Option<String>,
    #[serde(default, rename = "claim")]
    pub claims: Vec<Claim>,
}

/// Clears a session's in-context set and pending injection.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Clear {
    pub at: Timestamp,
    pub session: String,
}

/// What extraction would have found: call 1's reply, and the outcomes
/// call 2 gives it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Claim {
    /// Names the memory the claim creates. Unique across the scenario.
    #[serde(default)]
    pub label: Option<String>,
    pub content: String,
    pub quote: String,
    pub kind: Kind,
    pub significance: Significance,
    #[serde(default)]
    pub remember_this: bool,
    #[serde(default)]
    pub changes_something: bool,
    #[serde(default)]
    pub valid_from: Option<When>,
    #[serde(default)]
    pub valid_until: Option<When>,
    #[serde(default)]
    pub low_confidence: bool,
    #[serde(default)]
    pub until_event: Option<String>,
    #[serde(default)]
    pub due_at: Option<When>,
    #[serde(default)]
    pub volatility: Option<Volatility>,
    #[serde(default)]
    pub recurrence_text: Option<String>,
    #[serde(default)]
    pub recurrence_rrule: Option<String>,
    #[serde(default)]
    pub recurrence_start: Option<When>,
    #[serde(default)]
    pub reconcile: Vec<Outcome>,
}

impl Claim {
    /// How an error names the claim: its label, or its ordinal in the
    /// event's list.
    pub fn name(&self, ordinal: usize) -> String {
        match &self.label {
            Some(label) => format!("{label:?}"),
            None => format!("claim {}", ordinal + 1),
        }
    }
}

/// A time as call 1 returns it: local to the source's timezone, at a
/// precision.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct When {
    pub at: String,
    pub precision: Precision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    Year,
    Month,
    Day,
    Hour,
    Minute,
}

impl Precision {
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::Year => "year",
            Precision::Month => "month",
            Precision::Day => "day",
            Precision::Hour => "hour",
            Precision::Minute => "minute",
        }
    }
}

/// One of the claim's outcomes against an existing memory.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    pub memory: String,
    pub outcome: Label,
}

impl Outcome {
    /// Whether the outcome absorbs the claim into the memory rather than
    /// creating one.
    pub fn absorbs(&self) -> bool {
        matches!(self.outcome, Label::MentionedAgain | Label::Confirmed)
    }
}

/// A time and an expectation. Never changes the run.
#[derive(Debug, Clone, Deserialize)]
pub struct Probe {
    /// `p<n>` in file order when absent.
    #[serde(default)]
    pub id: Option<String>,
    pub at: Timestamp,
    #[serde(flatten)]
    pub check: Check,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Check {
    Band {
        memory: String,
        band: Band,
    },
    /// The first instant strength fell below τ, inclusive at both ends.
    FadedAt {
        memory: String,
        between: [Timestamp; 2],
    },
    Exists {
        memory: String,
        #[serde(default)]
        memory_kind: Option<Kind>,
        #[serde(default)]
        ended: Option<bool>,
        #[serde(default)]
        retracted: Option<bool>,
        /// Whether it's the head of its supersession chain.
        #[serde(default)]
        head: Option<bool>,
        #[serde(default)]
        phase: Option<Phase>,
    },
    Absent {
        memory: String,
    },
    AgendaHas {
        memory: String,
    },
    AgendaLacks {
        memory: String,
    },
    RecallFinds {
        memory: String,
        query: String,
    },
    RecallLacks {
        memory: String,
        query: String,
    },
    /// Group `models` only.
    Injects {
        memory: String,
        query: String,
    },
    NotInjects {
        memory: String,
        query: String,
    },
    ProfileHas {
        model: String,
        memory: String,
    },
    ProfileLacks {
        model: String,
        memory: String,
    },
}

impl Check {
    pub fn memory(&self) -> &str {
        match self {
            Check::Band { memory, .. }
            | Check::FadedAt { memory, .. }
            | Check::Exists { memory, .. }
            | Check::Absent { memory }
            | Check::AgendaHas { memory }
            | Check::AgendaLacks { memory }
            | Check::RecallFinds { memory, .. }
            | Check::RecallLacks { memory, .. }
            | Check::Injects { memory, .. }
            | Check::NotInjects { memory, .. }
            | Check::ProfileHas { memory, .. }
            | Check::ProfileLacks { memory, .. } => memory,
        }
    }

    /// The probe's kind as the file and the report name it.
    pub fn kind(&self) -> &'static str {
        match self {
            Check::Band { .. } => "band",
            Check::FadedAt { .. } => "faded_at",
            Check::Exists { .. } => "exists",
            Check::Absent { .. } => "absent",
            Check::AgendaHas { .. } => "agenda_has",
            Check::AgendaLacks { .. } => "agenda_lacks",
            Check::RecallFinds { .. } => "recall_finds",
            Check::RecallLacks { .. } => "recall_lacks",
            Check::Injects { .. } => "injects",
            Check::NotInjects { .. } => "not_injects",
            Check::ProfileHas { .. } => "profile_has",
            Check::ProfileLacks { .. } => "profile_lacks",
        }
    }

    /// Whether the probe needs the real models.
    pub fn needs_models(&self) -> bool {
        matches!(self, Check::Injects { .. } | Check::NotInjects { .. })
    }
}

impl Probe {
    /// The probe's id: its own, or `p<n>` for its position in file order.
    pub fn id(&self, index: usize) -> String {
        self.id.clone().unwrap_or_else(|| format!("p{}", index + 1))
    }
}

impl Scenario {
    /// The earliest event or probe, where the clock starts.
    pub fn earliest(&self) -> Option<Timestamp> {
        self.times().min()
    }

    /// The latest event or probe, where the run ends unless `--until`
    /// extends it.
    pub fn latest(&self) -> Option<Timestamp> {
        self.times().max()
    }

    fn times(&self) -> impl Iterator<Item = Timestamp> + '_ {
        let turns = self
            .turns
            .iter()
            .flat_map(|turn| [turn.at, turn.reply_at.unwrap_or(turn.at)]);
        let chatter = self.chatter.iter().flat_map(|chatter| {
            let every = duration(&chatter.every).unwrap_or(SignedDuration::ZERO);
            let last = chatter
                .from
                .checked_add(every * i64::from(chatter.count.saturating_sub(1)) as i32)
                .unwrap_or(chatter.from);
            [chatter.from, last]
        });
        turns
            .chain(chatter)
            .chain(self.documents.iter().map(|document| document.at))
            .chain(self.clears.iter().map(|clear| clear.at))
            .chain(self.probes.iter().map(|probe| probe.at))
    }
}

/// A duration as the file writes it: `1d`, `12h`, `10m`, `0s`, with days
/// as 24 hours.
pub fn duration(text: &str) -> Result<SignedDuration, String> {
    let span: Span = text.parse().map_err(|error| format!("{error}"))?;
    span.to_duration(SpanRelativeTo::days_are_24_hours())
        .map_err(|error| format!("{error}"))
}

/// Reads and checks a scenario file.
pub fn load(path: &Path) -> anyhow::Result<Scenario> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading the scenario {}", path.display()))?;
    let scenario: Scenario =
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let errors = check(&scenario);
    if !errors.is_empty() {
        anyhow::bail!("{}:\n{}", path.display(), errors.join("\n"));
    }
    Ok(scenario)
}

/// The checks the loader makes before a run, each as a sentence naming
/// the label at fault. Empty when the scenario is well formed.
pub fn check(scenario: &Scenario) -> Vec<String> {
    let mut errors = Vec::new();
    // The default report path is `reports/<name>.json` under the private
    // dir, so the name is one filename component.
    let mut components = Path::new(&scenario.name).components();
    let one_component = matches!(
        (components.next(), components.next()),
        (Some(std::path::Component::Normal(_)), None)
    );
    if !one_component || scenario.name.contains(['/', '\\', '\0']) {
        errors.push(format!(
            "the scenario's name {:?} isn't a single filename component",
            scenario.name
        ));
    }
    let sessions = scenario
        .turns
        .iter()
        .map(|turn| turn.session.as_str())
        .chain(
            scenario
                .chatter
                .iter()
                .map(|chatter| chatter.session.as_deref().unwrap_or("chatter")),
        )
        .chain(scenario.clears.iter().map(|clear| clear.session.as_str()));
    let reserved: BTreeSet<&str> = sessions
        .filter(|session| session.starts_with(PROBE_SESSION_PREFIX))
        .collect();
    for session in reserved {
        errors.push(format!(
            "the session {session:?} starts with {PROBE_SESSION_PREFIX:?}, which is reserved for probes"
        ));
    }
    let mut probe_ids = BTreeSet::new();
    for (index, probe) in scenario.probes.iter().enumerate() {
        let id = probe.id(index);
        if !probe_ids.insert(id.clone()) {
            errors.push(format!("the probe id {id:?} is used twice"));
        }
    }
    // Every labelled claim, with when its event happens.
    let mut labels: Vec<(String, Timestamp)> = Vec::new();
    let mut claims: Vec<(Timestamp, &Claim)> = Vec::new();
    for turn in &scenario.turns {
        let chunk = format!("{}\n{}", turn.user, turn.assistant);
        for claim in &turn.claims {
            if !turn.user.contains(&claim.quote) && !turn.assistant.contains(&claim.quote) {
                errors.push(format!(
                    "the quote {:?} isn't in the turn at {}: {chunk:?}",
                    claim.quote, turn.at
                ));
            }
            claims.push((turn.at, claim));
        }
        for used in &turn.used {
            if !labels_before(&claims, used, turn.at) {
                errors.push(format!(
                    "the turn at {} uses {used:?}, which no earlier claim labels",
                    turn.at
                ));
            }
        }
    }
    for document in &scenario.documents {
        for claim in &document.claims {
            if !document.text.contains(&claim.quote) {
                errors.push(format!(
                    "the quote {:?} isn't in document {}",
                    claim.quote, document.id
                ));
            }
            claims.push((document.at, claim));
        }
    }
    claims.sort_by_key(|(at, _)| *at);
    for (at, claim) in &claims {
        if let Some(label) = &claim.label {
            if labels.iter().any(|(known, _)| known == label) {
                errors.push(format!("the label {label:?} is used twice"));
            }
            if !claim.reconcile.is_empty() && claim.reconcile.iter().all(Outcome::absorbs) {
                errors.push(format!(
                    "the claim {label:?} is absorbed by its outcomes, so its label names nothing"
                ));
            }
            labels.push((label.clone(), *at));
        }
        for outcome in &claim.reconcile {
            let earlier = labels
                .iter()
                .any(|(known, when)| known == &outcome.memory && when < at);
            if !earlier {
                errors.push(format!(
                    "the claim at {at} reconciles against {:?}, which no earlier claim labels",
                    outcome.memory
                ));
            }
        }
    }
    for chatter in &scenario.chatter {
        if chatter.count == 0 {
            errors.push(format!("the chatter from {} has count 0", chatter.from));
        }
        if duration(&chatter.every).is_err() {
            errors.push(format!(
                "the chatter from {} has an interval that doesn't parse: {:?}",
                chatter.from, chatter.every
            ));
        }
    }
    if let Some(latency) = &scenario.latency
        && duration(latency).is_err()
    {
        errors.push(format!("the latency doesn't parse: {latency:?}"));
    }
    for (index, probe) in scenario.probes.iter().enumerate() {
        let id = probe.id(index);
        let memory = probe.check.memory();
        match labels.iter().find(|(known, _)| known == memory) {
            None => errors.push(format!(
                "probe {id} names {memory:?}, which no claim labels"
            )),
            Some((_, created)) => {
                if !matches!(probe.check, Check::Absent { .. }) && probe.at < *created {
                    errors.push(format!(
                        "probe {id} at {} is before {memory:?} is created at {created}",
                        probe.at
                    ));
                }
            }
        }
        if let Check::FadedAt { between, .. } = &probe.check {
            if between[0] > between[1] {
                errors.push(format!("probe {id} has its range backwards"));
            }
            if probe.at < between[1] {
                errors.push(format!(
                    "probe {id} at {} is before the end of its range {}",
                    probe.at, between[1]
                ));
            }
        }
        if probe.check.needs_models() && scenario.group == Group::Ci {
            errors.push(format!(
                "probe {id} needs the real models but the group is ci"
            ));
        }
    }
    errors
}

fn labels_before(claims: &[(Timestamp, &Claim)], label: &str, at: Timestamp) -> bool {
    claims
        .iter()
        .any(|(when, claim)| *when < at && claim.label.as_deref() == Some(label))
}
