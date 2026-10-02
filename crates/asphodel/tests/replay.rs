//! The replay harness on scripted scenarios, checked against "Replay
//! harness: simulated-clock replay of recorded sessions" (TIM-96, the
//! resolution and its TIM-97 and TIM-98 amendments), "Deletion policy"
//! (TIM-97, decision 7), the lifetimes in "Strength model" (TIM-91), and
//! ADRs 0004 and 0008. The contract these tests pin is `docs/replay.md`.
//!
//! `asphodel replay` is a stub, so every test that runs it is ignored until
//! TIM-116 lands. They drive the binary as a process, as `serve_http.rs`
//! does, and read the JSON report, so nothing here depends on how the
//! engine is laid out inside. The scenario files under `scenarios/` are the
//! fixtures; [`contract`] holds their shape as serde types, and the tests
//! that run now parse every checked-in scenario with it and check that
//! labels resolve. To activate: move [`contract`] into the crate as the
//! scenario loader, import it here, and drop the `ignore` attributes.
//!
//! Where TIM-96 leaves a detail open, `docs/replay.md` proposes one and
//! says so. Those are the places to argue with.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

/// The checked-in scenarios.
const SCENARIOS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../scenarios");

/// The first set (TIM-116), all in group `ci`.
const FIRST_SET: [&str; 6] = [
    "lifetimes",
    "purge-table",
    "maya-to-mia",
    "rescheduled-appointment",
    "three-week-holiday",
    "extraction-latency",
];

fn scenario(name: &str) -> PathBuf {
    Path::new(SCENARIOS).join(format!("{name}.toml"))
}

/// The scenario file's shape, as `docs/replay.md` gives it.
///
/// The enums for kinds, significance, outcomes, bands and phases are the
/// production ones, so the file's words are exactly the API's. Only the
/// time precision is defined here, since extraction keeps its own private.
mod contract {
    #![allow(dead_code)]

    use asphodel_core::constants::{Significance, Volatility};
    use asphodel_core::extraction::Label;
    use asphodel_core::retrieval::Band;
    use asphodel_core::strength::{Kind, Phase};
    use jiff::Timestamp;
    use jiff::civil::Date;
    use serde::Deserialize;

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

    #[derive(Debug, Deserialize)]
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

    #[derive(Debug, Deserialize)]
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
    #[derive(Debug, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub struct Clear {
        pub at: Timestamp,
        pub session: String,
    }

    /// What extraction would have found: call 1's reply, and the outcomes
    /// call 2 gives it.
    #[derive(Debug, Deserialize)]
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

    /// A time as call 1 returns it: local to the source's timezone, at a
    /// precision.
    #[derive(Debug, Deserialize)]
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

    /// One of the claim's outcomes against an existing memory.
    #[derive(Debug, Deserialize)]
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
    #[derive(Debug, Deserialize)]
    pub struct Probe {
        /// `p<n>` in file order when absent.
        #[serde(default)]
        pub id: Option<String>,
        pub at: Timestamp,
        #[serde(flatten)]
        pub check: Check,
    }

    #[derive(Debug, Deserialize)]
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

        /// Whether the probe needs the real models.
        pub fn needs_models(&self) -> bool {
            matches!(self, Check::Injects { .. } | Check::NotInjects { .. })
        }
    }

    /// The checks the loader makes before a run, each as a sentence naming
    /// the label at fault. Empty when the scenario is well formed.
    pub fn check(scenario: &Scenario) -> Vec<String> {
        let mut errors = Vec::new();
        // Every labelled claim, with when its event happens.
        let mut labels: Vec<(String, Timestamp)> = Vec::new();
        let mut claims: Vec<(Timestamp, &str, &Claim)> = Vec::new();
        for turn in &scenario.turns {
            let chunk = format!("{}\n{}", turn.user, turn.assistant);
            for claim in &turn.claims {
                if !turn.user.contains(&claim.quote) && !turn.assistant.contains(&claim.quote) {
                    errors.push(format!(
                        "the quote {:?} isn't in the turn at {}: {chunk:?}",
                        claim.quote, turn.at
                    ));
                }
                claims.push((turn.at, "turn", claim));
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
                claims.push((document.at, "document", claim));
            }
        }
        claims.sort_by_key(|(at, _, _)| *at);
        for (at, _, claim) in &claims {
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
            if chatter.every.parse::<jiff::Span>().is_err() {
                errors.push(format!(
                    "the chatter from {} has an interval that doesn't parse: {:?}",
                    chatter.from, chatter.every
                ));
            }
        }
        if let Some(latency) = &scenario.latency
            && latency.parse::<jiff::Span>().is_err()
        {
            errors.push(format!("the latency doesn't parse: {latency:?}"));
        }
        for (index, probe) in scenario.probes.iter().enumerate() {
            let id = probe
                .id
                .clone()
                .unwrap_or_else(|| format!("p{}", index + 1));
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

    fn labels_before(claims: &[(Timestamp, &str, &Claim)], label: &str, at: Timestamp) -> bool {
        claims
            .iter()
            .any(|(when, _, claim)| *when < at && claim.label.as_deref() == Some(label))
    }
}

// Test support.

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-replay-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    /// The private directory a run writes under.
    fn replay_dir(&self) -> PathBuf {
        let path = self.path("private");
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn file(&self, name: &str, text: &str) -> PathBuf {
        let path = self.path(name);
        fs::write(&path, text).unwrap();
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// One `asphodel replay` run: the process output and the report it wrote,
/// if it wrote one.
struct Run {
    output: Output,
    report_path: PathBuf,
    report: Option<Value>,
}

impl Run {
    fn code(&self) -> Option<i32> {
        self.output.status.code()
    }

    fn stderr(&self) -> String {
        String::from_utf8_lossy(&self.output.stderr).into_owned()
    }

    fn report(&self) -> &Value {
        self.report.as_ref().unwrap_or_else(|| {
            panic!(
                "no report at {}: {}",
                self.report_path.display(),
                self.stderr()
            )
        })
    }

    fn probes(&self) -> Vec<&Value> {
        self.report()["probes"]
            .as_array()
            .expect("the report lists its probes")
            .iter()
            .collect()
    }

    fn probe(&self, id: &str) -> &Value {
        self.probes()
            .into_iter()
            .find(|probe| probe["id"] == id)
            .unwrap_or_else(|| panic!("no probe {id} in the report"))
    }

    /// Exit 0 and every probe passed, or the failed probes in the message.
    fn assert_passed(&self) {
        let failed: Vec<String> = self
            .probes()
            .iter()
            .filter(|probe| probe["passed"] != true)
            .map(|probe| probe.to_string())
            .collect();
        assert!(
            self.code() == Some(0) && failed.is_empty(),
            "exit {:?}, failed probes: {failed:#?}, stderr: {}",
            self.code(),
            self.stderr()
        );
    }

    /// Exit 2 with `word` in the message and no report.
    fn assert_refused(&self, word: &str) {
        assert_eq!(self.code(), Some(2), "stderr: {}", self.stderr());
        assert!(
            self.stderr().contains(word),
            "stderr should name {word:?}: {}",
            self.stderr()
        );
        assert!(self.report.is_none(), "a refused run writes no report");
    }
}

/// Runs `asphodel replay` on `scenario` with `ASPHODEL_REPLAY_DIR` set to
/// the test's private dir, `--report` to a fresh file, and `extra` after.
fn replay(dir: &TestDir, scenario: &Path, extra: &[&str]) -> Run {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let report_path = dir.path(&format!(
        "report-{}.json",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut command = Command::new(env!("CARGO_BIN_EXE_asphodel"));
    command
        .arg("replay")
        .arg("--scenario")
        .arg(scenario)
        .arg("--report")
        .arg(&report_path)
        .args(extra)
        .env("ASPHODEL_REPLAY_DIR", dir.replay_dir())
        .env_remove("ASPHODEL_MODEL_DIR");
    let output = command.output().unwrap();
    let report = fs::read(&report_path)
        .ok()
        .map(|bytes| serde_json::from_slice(&bytes).expect("the report is JSON"));
    Run {
        output,
        report_path,
        report,
    }
}

/// The sum of `purges_per_day`.
fn purges(report: &Value) -> u64 {
    report["purges_per_day"]
        .as_array()
        .expect("the report counts purges per day")
        .iter()
        .map(|day| day["purged"].as_u64().unwrap())
        .sum()
}

/// A scenario written by a test, for the scenario-error cases.
fn inline(dir: &TestDir, name: &str, body: &str) -> PathBuf {
    dir.file(
        &format!("{name}.toml"),
        &format!("name = \"{name}\"\ngroup = \"ci\"\n{body}"),
    )
}

/// A turn that creates one minor fact, `Tim lives in Auckland.`, labelled
/// `home`.
const HOME_TURN: &str = r#"
[[turn]]
at = "2026-01-05T09:00:00Z"
session = "s1"
user = "I live in Auckland."
assistant = "Noted."

[[turn.claim]]
label = "home"
content = "Tim lives in Auckland."
quote = "I live in Auckland"
kind = "fact"
significance = "minor"
"#;

// The fixtures, checked now.

#[test]
fn every_checked_in_scenario_parses_and_resolves_its_labels() {
    let mut seen = 0;
    for entry in fs::read_dir(SCENARIOS).expect("the scenarios directory exists") {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|ext| ext != "toml") {
            continue;
        }
        seen += 1;
        let text = fs::read_to_string(&path).unwrap();
        let scenario: contract::Scenario =
            toml::from_str(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        assert_eq!(
            Some(scenario.name.as_str()),
            path.file_stem().and_then(|stem| stem.to_str()),
            "{}: the name is the file stem",
            path.display()
        );
        let errors = contract::check(&scenario);
        assert!(
            errors.is_empty(),
            "{}:\n{}",
            path.display(),
            errors.join("\n")
        );
        assert!(
            !scenario.probes.is_empty(),
            "{}: a scenario without probes checks nothing",
            path.display()
        );
    }
    assert!(
        seen >= FIRST_SET.len(),
        "only {seen} scenarios are checked in"
    );
}

#[test]
fn the_first_set_is_checked_in_and_runs_in_ci() {
    for name in FIRST_SET {
        let text =
            fs::read_to_string(scenario(name)).unwrap_or_else(|error| panic!("{name}: {error}"));
        let scenario: contract::Scenario = toml::from_str(&text).unwrap();
        assert_eq!(scenario.group, contract::Group::Ci, "{name}");
    }
}

#[test]
fn the_contract_rejects_what_the_loader_would() {
    let text = format!(
        "{HOME_TURN}
[[turn]]
at = \"2026-01-06T09:00:00Z\"
session = \"s1\"
user = \"I live in Auckland, as I said.\"
assistant = \"You did.\"
used = [\"nowhere\"]

[[turn.claim]]
label = \"home-again\"
content = \"Tim lives in Auckland.\"
quote = \"I live in Auckland\"
kind = \"fact\"
significance = \"minor\"
reconcile = [{{ memory = \"home\", outcome = \"mentioned_again\" }}]

[[turn.claim]]
label = \"later\"
content = \"Tim lives in Wellington.\"
quote = \"not in the turn\"
kind = \"fact\"
significance = \"minor\"
reconcile = [{{ memory = \"future\", outcome = \"refines\" }}]

[[probe]]
at = \"2026-01-01T00:00:00Z\"
kind = \"band\"
memory = \"home\"
band = \"strong\"

[[probe]]
id = \"ghost\"
at = \"2026-02-01T00:00:00Z\"
kind = \"absent\"
memory = \"nobody\"

[[probe]]
id = \"backwards\"
at = \"2026-01-07T00:00:00Z\"
kind = \"faded_at\"
memory = \"home\"
between = [\"2026-02-01T00:00:00Z\", \"2026-01-20T00:00:00Z\"]
"
    );
    let scenario: contract::Scenario =
        toml::from_str(&format!("name = \"bad\"\ngroup = \"ci\"\n{text}")).unwrap();
    let errors = contract::check(&scenario);
    let expect = |needle: &str| {
        assert!(
            errors.iter().any(|error| error.contains(needle)),
            "no error mentions {needle:?}: {errors:#?}"
        );
    };
    expect("uses \"nowhere\"");
    expect("\"home-again\" is absorbed");
    expect("reconciles against \"future\"");
    expect("the quote \"not in the turn\"");
    expect("probe p1 at 2026-01-01T00:00:00Z is before \"home\"");
    expect("probe ghost names \"nobody\"");
    expect("probe backwards has its range backwards");
    expect("probe backwards at 2026-01-07T00:00:00Z is before the end of its range");
}

#[test]
fn an_unknown_scenario_field_doesnt_parse() {
    let error = toml::from_str::<contract::Scenario>(
        "name = \"x\"\ngroup = \"ci\"\nstart = \"2026-01-01T00:00:00Z\"\n",
    )
    .unwrap_err();
    assert!(error.to_string().contains("start"), "{error}");
    let error =
        toml::from_str::<contract::Scenario>("name = \"x\"\ngroup = \"nightly\"\n").unwrap_err();
    assert!(error.to_string().contains("nightly"), "{error}");
}

// The first set, through `asphodel replay`.

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn lifetimes_reproduce_tim_91_within_tolerance() {
    let dir = TestDir::new();
    let run = replay(&dir, &scenario("lifetimes"), &[]);
    run.assert_passed();
    for id in [
        "trivial-fades",
        "minor-fades",
        "notable-fades",
        "major-fades",
        "critical-fades",
    ] {
        let probe = run.probe(id);
        assert_eq!(probe["kind"], "faded_at", "{probe}");
        assert!(probe["observed"]["faded_at"].is_string(), "{probe}");
    }
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn the_purge_table_reproduces_adr_0008_and_the_shadow_table_counts_re_mentions() {
    let dir = TestDir::new();
    let run = replay(&dir, &scenario("purge-table"), &[]);
    run.assert_passed();
    let report = run.report();
    assert_eq!(purges(report), 3, "{}", report["purges_per_day"]);
    let shadow = &report["purged_then_re_mentioned"];
    assert_eq!(shadow["purged"], 3, "{shadow}");
    assert_eq!(shadow["re_mentioned"], 1, "{shadow}");
    let rate = shadow["rate"].as_f64().expect("a rate");
    assert!((rate - 1.0 / 3.0).abs() < 1e-3, "{shadow}");
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn maya_corrected_to_mia_keeps_her_strength() {
    let dir = TestDir::new();
    let run = replay(&dir, &scenario("maya-to-mia"), &[]);
    run.assert_passed();
    // Mia's id is the head Maya's chain points at: the observed ids agree.
    let mia = &run.probe("mia-is-the-head")["observed"];
    let maya = &run.probe("maya-is-retracted")["observed"];
    assert!(
        mia["id"].is_string() && maya["id"].is_string(),
        "{mia} {maya}"
    );
    assert_ne!(mia["id"], maya["id"]);
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn a_rescheduled_appointment_moves_on_the_agenda() {
    let dir = TestDir::new();
    replay(&dir, &scenario("rescheduled-appointment"), &[]).assert_passed();
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn a_three_week_holiday_runs_on_bank_time() {
    let dir = TestDir::new();
    replay(&dir, &scenario("three-week-holiday"), &[]).assert_passed();
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn extraction_latency_is_modelled_in_simulated_time() {
    let dir = TestDir::new();
    let run = replay(&dir, &scenario("extraction-latency"), &[]);
    run.assert_passed();
    assert_eq!(run.report()["flags"]["latency_ms"], 600_000);

    // With no latency the first turn's memory exists at 09:05, so the
    // first probe fails, the run exits 1, and the report still says so.
    let run = replay(&dir, &scenario("extraction-latency"), &["--latency", "0s"]);
    assert_eq!(run.code(), Some(1), "stderr: {}", run.stderr());
    assert_eq!(run.report()["flags"]["latency_ms"], 0);
    assert_eq!(run.probe("first-turn-not-yet-extracted")["passed"], false);
    assert_eq!(
        run.probe("second-turn-waits-for-the-worker")["passed"],
        false
    );
    assert_eq!(
        run.probe("first-turn-extracted-after-its-latency")["passed"],
        true
    );
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn until_keeps_sweeping_past_the_last_turn() {
    let dir = TestDir::new();
    let run = replay(&dir, &scenario("extraction-latency"), &[]);
    run.assert_passed();
    assert_eq!(purges(run.report()), 0);
    assert!(run.report()["flags"]["until"].is_null());

    // A year of nightly sweeps purges both trivial memories.
    let run = replay(
        &dir,
        &scenario("extraction-latency"),
        &["--until", "2027-01-05T09:00:00Z"],
    );
    run.assert_passed();
    assert_eq!(run.report()["flags"]["until"], "2027-01-05T09:00:00Z");
    assert_eq!(
        purges(run.report()),
        2,
        "{}",
        run.report()["purges_per_day"]
    );
}

// The report.

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn the_report_names_its_run_and_defaults_its_path() {
    let dir = TestDir::new();
    let run = replay(&dir, &scenario("extraction-latency"), &[]);
    run.assert_passed();
    let report = run.report();
    assert_eq!(report["kind"], "scripted");
    assert_eq!(report["scenario"], "extraction-latency");
    assert_eq!(report["group"], "ci");
    assert_eq!(report["version"], env!("CARGO_PKG_VERSION"));
    assert!(report.get("git_sha").is_some(), "{report}");
    assert_eq!(report["llm"]["live"], 0);
    assert_eq!(report["llm"]["cache"], 0);
    assert!(report["llm"]["scripted"].as_u64().is_some_and(|n| n >= 2));
    // The fake floors are in the resolved tuning, under the scenario's
    // own layer.
    assert_eq!(report["tuning"]["clock"]["quiet_rate"], 1.0);
    assert_eq!(
        report["tuning"]["reconcile"]["embedding_floors"]["fake-embedder:v1"],
        0.5
    );

    // Without --report, the report lands in the private dir.
    let output = Command::new(env!("CARGO_BIN_EXE_asphodel"))
        .arg("replay")
        .arg("--scenario")
        .arg(scenario("extraction-latency"))
        .env("ASPHODEL_REPLAY_DIR", dir.replay_dir())
        .env_remove("ASPHODEL_MODEL_DIR")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let default = dir.replay_dir().join("reports/extraction-latency.json");
    assert!(default.is_file(), "no report at {}", default.display());
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn probes_without_an_id_are_numbered_in_file_order() {
    let dir = TestDir::new();
    let path = inline(
        &dir,
        "unnamed",
        &format!(
            "{HOME_TURN}
[[probe]]
at = \"2026-01-05T10:00:00Z\"
kind = \"exists\"
memory = \"home\"

[[probe]]
at = \"2026-01-05T10:00:00Z\"
kind = \"band\"
memory = \"home\"
band = \"strong\"
"
        ),
    );
    let run = replay(&dir, &path, &[]);
    run.assert_passed();
    let ids: Vec<&Value> = run.probes().iter().map(|probe| &probe["id"]).collect();
    assert_eq!(ids, [&Value::from("p1"), &Value::from("p2")]);
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn replay_is_deterministic() {
    let first = TestDir::new();
    let second = TestDir::new();
    let a = replay(&first, &scenario("maya-to-mia"), &[]);
    let b = replay(&second, &scenario("maya-to-mia"), &[]);
    a.assert_passed();
    b.assert_passed();
    assert_eq!(
        fs::read(&a.report_path).unwrap(),
        fs::read(&b.report_path).unwrap(),
        "two runs of the same scenario in different private dirs differ"
    );
}

// Overrides and layering (TIM-98 amendment).

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn overrides_layer_over_the_production_file_in_the_shape_of_tuning() {
    let dir = TestDir::new();
    let config = dir.file("production.toml", "[clock]\nquiet_rate = 0.2\n");
    let overrides = dir.file("overrides.toml", "[clock]\nquiet_rate = 0.5\n");
    let path = scenario("extraction-latency");
    let path = path.to_str().unwrap();

    // The scenario's own [tuning] (1.0) sits above the production file.
    let run = replay(
        &dir,
        Path::new(path),
        &["--config", config.to_str().unwrap()],
    );
    run.assert_passed();
    assert_eq!(run.report()["tuning"]["clock"]["quiet_rate"], 1.0);

    // --overrides sits above the scenario.
    let run = replay(
        &dir,
        Path::new(path),
        &[
            "--config",
            config.to_str().unwrap(),
            "--overrides",
            overrides.to_str().unwrap(),
        ],
    );
    run.assert_passed();
    assert_eq!(run.report()["tuning"]["clock"]["quiet_rate"], 0.5);
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn an_overrides_file_with_an_unknown_key_is_refused_before_the_run() {
    let dir = TestDir::new();
    let overrides = dir.file("overrides.toml", "[clock]\nspeed = 2.0\n");
    let run = replay(
        &dir,
        &scenario("extraction-latency"),
        &["--overrides", overrides.to_str().unwrap()],
    );
    run.assert_refused("speed");
}

// Privacy and the private directory (TIM-96, decisions 3 and 8).

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn replay_refuses_to_run_without_a_private_directory() {
    let dir = TestDir::new();
    let report = dir.path("report.json");
    let output = Command::new(env!("CARGO_BIN_EXE_asphodel"))
        .arg("replay")
        .arg("--scenario")
        .arg(scenario("extraction-latency"))
        .arg("--report")
        .arg(&report)
        .env_remove("ASPHODEL_REPLAY_DIR")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("ASPHODEL_REPLAY_DIR"), "{stderr}");
    assert!(!report.exists());
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn replay_refuses_a_private_directory_inside_a_git_working_tree() {
    let dir = TestDir::new();
    fs::create_dir_all(dir.path("repo/.git")).unwrap();
    let private = dir.path("repo/private");
    fs::create_dir_all(&private).unwrap();
    let run = replay(
        &dir,
        &scenario("extraction-latency"),
        &["--replay-dir", private.to_str().unwrap()],
    );
    run.assert_refused("git");
    assert!(
        fs::read_dir(&private).unwrap().next().is_none(),
        "nothing is written inside the working tree"
    );
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn replay_refuses_a_serve_data_dir() {
    let dir = TestDir::new();
    // A data dir is recognised by the store file at its top level; replay's
    // own store lives under <replay dir>/store.
    fs::write(dir.replay_dir().join("asphodel.db"), b"").unwrap();
    let run = replay(&dir, &scenario("extraction-latency"), &[]);
    run.assert_refused("serve");
}

// Scenario errors: the engine never guesses (docs/replay.md, "Claims").

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn a_scripted_outcome_whose_target_isnt_a_neighbour_is_a_scenario_error() {
    let dir = TestDir::new();
    let path = inline(
        &dir,
        "bad-target",
        &format!(
            "{HOME_TURN}
[[turn]]
at = \"2026-01-06T09:00:00Z\"
session = \"s1\"
user = \"The weather is lovely today.\"
assistant = \"Enjoy it.\"

[[turn.claim]]
label = \"weather\"
content = \"The weather was lovely on 2026-01-06.\"
quote = \"The weather is lovely today\"
kind = \"event\"
significance = \"trivial\"
reconcile = [{{ memory = \"home\", outcome = \"mentioned_again\" }}]

[[probe]]
at = \"2026-01-07T09:00:00Z\"
kind = \"exists\"
memory = \"home\"
"
        ),
    );
    let run = replay(&dir, &path, &[]);
    run.assert_refused("weather");
    assert!(run.stderr().contains("home"), "{}", run.stderr());
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn a_used_memory_that_isnt_in_context_is_a_scenario_error() {
    let dir = TestDir::new();
    // A new session whose query shares no word with the memory: nothing is
    // injected, so nothing is in context to use.
    let path = inline(
        &dir,
        "bad-used",
        &format!(
            "{HOME_TURN}
[[turn]]
at = \"2026-01-06T09:00:00Z\"
session = \"s2\"
user = \"Tell me a joke.\"
assistant = \"Why did the chicken cross the road?\"
used = [\"home\"]

[[probe]]
at = \"2026-01-07T09:00:00Z\"
kind = \"exists\"
memory = \"home\"
"
        ),
    );
    replay(&dir, &path, &[]).assert_refused("home");
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn a_label_on_an_absorbed_claim_is_a_scenario_error() {
    let dir = TestDir::new();
    let path = inline(
        &dir,
        "bad-label",
        &format!(
            "{HOME_TURN}
[[turn]]
at = \"2026-01-06T09:00:00Z\"
session = \"s1\"
user = \"I live in Auckland, as I said.\"
assistant = \"You did.\"

[[turn.claim]]
label = \"home-again\"
content = \"Tim lives in Auckland.\"
quote = \"I live in Auckland\"
kind = \"fact\"
significance = \"minor\"
reconcile = [{{ memory = \"home\", outcome = \"mentioned_again\" }}]

[[probe]]
at = \"2026-01-07T09:00:00Z\"
kind = \"exists\"
memory = \"home\"
"
        ),
    );
    replay(&dir, &path, &[]).assert_refused("home-again");
}

#[test]
#[ignore = "needs TIM-116: asphodel replay"]
fn a_probe_naming_an_unknown_label_is_refused_before_the_run() {
    let dir = TestDir::new();
    let path = inline(
        &dir,
        "bad-probe",
        &format!(
            "{HOME_TURN}
[[probe]]
id = \"ghost\"
at = \"2026-01-07T09:00:00Z\"
kind = \"exists\"
memory = \"nobody\"
"
        ),
    );
    let run = replay(&dir, &path, &[]);
    run.assert_refused("nobody");
    assert!(
        !dir.replay_dir().join("store").exists(),
        "a scenario refused at load opens no store"
    );
}
