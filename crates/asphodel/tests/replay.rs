//! The replay harness on scripted scenarios, checked against "Replay
//! harness: simulated-clock replay of recorded sessions" (TIM-96, the
//! resolution and its TIM-97 and TIM-98 amendments), "Deletion policy"
//! (TIM-97, decision 7), the lifetimes in "Strength model" (TIM-91), and
//! ADRs 0004 and 0008. The contract these tests pin is `docs/replay.md`.
//!
//! The tests drive the binary as a process, as `serve_http.rs` does, and
//! read the JSON report, so nothing here depends on how the engine is laid
//! out inside. The scenario files under `scenarios/` are the fixtures; the
//! command's own loader (`asphodel::replay::scenario`) parses every
//! checked-in one here and checks that labels resolve.
//!
//! Where TIM-96 leaves a detail open, `docs/replay.md` proposes one and
//! says so. Those are the places to argue with.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::Value;

/// The checked-in scenarios.
const SCENARIOS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../scenarios");

fn scenario(name: &str) -> PathBuf {
    Path::new(SCENARIOS).join(format!("{name}.toml"))
}

use asphodel::replay::scenario as contract;

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
    assert!(seen > 0, "no scenarios are checked in");
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
}

// The first set, through `asphodel replay`.

#[test]
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
fn the_purge_table_reproduces_adr_0008_and_the_shadow_table_counts_re_mentions() {
    let dir = TestDir::new();
    let run = replay(&dir, &scenario("purge-table"), &[]);
    run.assert_passed();
    let report = run.report();
    // ADR 0008 protects the reinforced memories and the major one, not
    // the single minor/notable mentions. Both purge before the run ends.
    assert_eq!(purges(report), 5, "{}", report["purges_per_day"]);
    let shadow = &report["purged_then_re_mentioned"];
    assert_eq!(shadow["purged"], 5, "{shadow}");
    assert_eq!(shadow["re_mentioned"], 1, "{shadow}");
    let rate = shadow["rate"].as_f64().expect("a rate");
    assert!((rate - 0.2).abs() < 1e-3, "{shadow}");
}

#[test]
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
fn a_rescheduled_appointment_moves_on_the_agenda() {
    let dir = TestDir::new();
    replay(&dir, &scenario("rescheduled-appointment"), &[]).assert_passed();
}

#[test]
fn a_three_week_holiday_runs_on_bank_time() {
    let dir = TestDir::new();
    replay(&dir, &scenario("three-week-holiday"), &[]).assert_passed();
}

#[test]
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

#[test]
fn scheduled_model_refreshes_are_counted_on_their_bank_local_day() {
    let dir = TestDir::new();
    let path = inline(
        &dir,
        "scheduled-refreshes",
        &format!(
            r#"
[bank]
timezone = "Pacific/Auckland"

[[model]]
name = "profile"
question = "Who is the user?"
kinds = ["fact"]
max_tokens = 100

{HOME_TURN}

[[probe]]
at = "2026-01-05T09:00:00Z"
kind = "exists"
memory = "home"
"#
        ),
    );
    let before = replay(&dir, &path, &["--until", "2026-01-05T09:04:59Z"]);
    before.assert_passed();
    assert_eq!(
        before.report()["refresh_calls_per_day"],
        serde_json::json!([])
    );

    let debounced = replay(&dir, &path, &["--until", "2026-01-05T09:05:00Z"]);
    debounced.assert_passed();
    assert_eq!(debounced.report()["flags"]["refresh"], "scripted");
    assert_eq!(
        debounced.report()["refresh_calls_per_day"],
        serde_json::json!([{ "day": "2026-01-05", "count": 1 }])
    );

    let daily_path = dir.file(
        "daily-refresh.toml",
        &format!(
            r#"{}
# A minor write changes the inputs without requesting a notable-write
# refresh. The daily sweep must pick it up at 04:00 bank-local.
[[turn]]
at = "2026-01-05T14:00:00Z"
session = "s1"
user = "My bicycle is blue."
assistant = "Noted."

[[turn.claim]]
label = "bicycle"
content = "Tim's bicycle is blue."
quote = "My bicycle is blue"
kind = "fact"
significance = "minor"
"#,
            fs::read_to_string(&path).unwrap()
        ),
    );
    let before_sweep = replay(&dir, &daily_path, &["--until", "2026-01-05T14:59:59Z"]);
    before_sweep.assert_passed();
    assert_eq!(
        before_sweep.report()["refresh_calls_per_day"],
        debounced.report()["refresh_calls_per_day"]
    );
    let swept = replay(&dir, &daily_path, &["--until", "2026-01-05T15:00:00Z"]);
    swept.assert_passed();
    // ADR 0007's seeded "User profile" also refreshes at the sweep.
    // Only the explicit model had a creation-triggered debounce earlier.
    assert_eq!(
        swept.report()["refresh_calls_per_day"],
        serde_json::json!([
            { "day": "2026-01-05", "count": 1 },
            { "day": "2026-01-06", "count": 2 }
        ])
    );
    assert_eq!(swept.report()["llm"]["live"], 0);
    let unchanged = replay(&dir, &daily_path, &["--until", "2026-01-06T15:00:00Z"]);
    unchanged.assert_passed();
    assert_eq!(
        unchanged.report()["refresh_calls_per_day"],
        swept.report()["refresh_calls_per_day"],
        "an unchanged fingerprint skips the next day's LLM calls"
    );
}

#[test]
fn a_multi_chunk_document_routes_claims_by_quote_and_has_distinct_stable_ids() {
    let dir = TestDir::new();
    let path = inline(
        &dir,
        "multi-chunk",
        r##"
latency = "10m"

[[document]]
at = "2026-01-05T09:00:00Z"
id = "notes"
reference_date = "2026-01-05"
text = "# Home\nTim lives in Auckland.\n\n# Transport\nTim's bicycle is blue."

# Deliberately reverse document order: assignment must follow the quote,
# not the ordinal in the scenario's claim list.
[[document.claim]]
label = "bicycle"
content = "Tim's bicycle is blue."
quote = "Tim's bicycle is blue"
kind = "fact"
significance = "minor"

[[document.claim]]
label = "home"
content = "Tim lives in Auckland."
quote = "Tim lives in Auckland"
kind = "fact"
significance = "minor"

[[probe]]
at = "2026-01-05T09:05:00Z"
kind = "absent"
memory = "home"

[[probe]]
at = "2026-01-05T09:05:00Z"
kind = "absent"
memory = "bicycle"

[[probe]]
id = "home-extracted"
at = "2026-01-05T09:10:00Z"
kind = "exists"
memory = "home"
head = true

[[probe]]
id = "bicycle-extracted"
at = "2026-01-05T09:10:00Z"
kind = "exists"
memory = "bicycle"
head = true
"##,
    );
    let run = replay(&dir, &path, &[]);
    run.assert_passed();
    let home = &run.probe("home-extracted")["observed"]["id"];
    let bicycle = &run.probe("bicycle-extracted")["observed"]["id"];
    assert!(home.is_string() && bicycle.is_string(), "{home} {bicycle}");
    assert_ne!(
        home, bicycle,
        "the first claim in each chunk needs its own id"
    );
    assert_eq!(run.report()["extraction_lag"]["samples"], 2);
    assert_eq!(run.report()["extraction_lag"]["p50_ms"], 600_000);
    assert_eq!(run.report()["extraction_lag"]["p95_ms"], 600_000);

    let other = TestDir::new();
    let repeated = replay(&other, &path, &[]);
    repeated.assert_passed();
    assert_eq!(
        fs::read(&run.report_path).unwrap(),
        fs::read(&repeated.report_path).unwrap()
    );
}

/// TIM-116 review, finding 4: the simulated worker claims the head of the
/// queue, which production orders by priority (turns before documents),
/// not in arrival order. The turn at 09:05 goes before the document's
/// second chunk, which arrived at 09:00.
#[test]
fn a_turn_between_a_documents_chunks_costs_the_rest_another_latency() {
    let dir = TestDir::new();
    let path = inline(
        &dir,
        "interrupted-document",
        r##"
latency = "10m"

[[document]]
at = "2026-01-05T09:00:00Z"
id = "notes"
reference_date = "2026-01-05"
text = "# Home\nTim lives in Auckland.\n\n# Transport\nTim's bicycle is blue."

[[document.claim]]
label = "bicycle"
content = "Tim's bicycle is blue."
quote = "Tim's bicycle is blue"
kind = "fact"
significance = "minor"

[[document.claim]]
label = "home"
content = "Tim lives in Auckland."
quote = "Tim lives in Auckland"
kind = "fact"
significance = "minor"

[[turn]]
at = "2026-01-05T09:05:00Z"
session = "s1"
user = "My favourite tea is Earl Grey."
assistant = "Noted."

[[turn.claim]]
label = "tea"
content = "Tim's favourite tea is Earl Grey."
quote = "My favourite tea is Earl Grey"
kind = "fact"
significance = "minor"

[[probe]]
at = "2026-01-05T09:15:00Z"
kind = "exists"
memory = "home"

[[probe]]
at = "2026-01-05T09:15:00Z"
kind = "absent"
memory = "bicycle"

[[probe]]
at = "2026-01-05T09:15:00Z"
kind = "absent"
memory = "tea"

[[probe]]
at = "2026-01-05T09:25:00Z"
kind = "exists"
memory = "tea"

[[probe]]
at = "2026-01-05T09:25:00Z"
kind = "absent"
memory = "bicycle"

[[probe]]
at = "2026-01-05T09:35:00Z"
kind = "exists"
memory = "bicycle"
"##,
    );
    let run = replay(&dir, &path, &[]);
    run.assert_passed();
    assert_eq!(run.report()["extraction_lag"]["samples"], 3);
    assert_eq!(run.report()["extraction_lag"]["p95_ms"], 1_800_000);
}

// The report.

#[test]
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

// Overrides and layering (TIM-98 amendment).

#[test]
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

// Privacy and the private directory (TIM-96, decisions 3 and 8).

#[test]
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

// The TIM-116 review findings on `ceef0c5`, accepted as regressions. Each
// names the finding it pins. They fail until the fixes land, except where
// noted.

/// Finding 1: replay resets `<replay dir>/store` without checking it made
/// it. A store dir replay didn't create, such as a stopped daemon's data
/// dir named `store`, is refused before anything in it is touched.
#[test]
fn replay_refuses_to_reset_a_store_it_didnt_create() {
    let dir = TestDir::new();
    let store = dir.replay_dir().join("store");
    fs::create_dir_all(&store).unwrap();
    fs::write(store.join("asphodel.db"), b"").unwrap();
    let sentinel = store.join("sentinel");
    fs::write(&sentinel, b"a daemon's file").unwrap();
    let run = replay(&dir, &scenario("extraction-latency"), &[]);
    run.assert_refused("store");
    assert!(
        sentinel.is_file(),
        "the sentinel in the unowned store was removed"
    );
}

/// Finding 2: the store lock is dropped before the reset, so two replays
/// on one private dir can race. The guard this pins: a replay holds its
/// private dir for its whole run, so a second replay on the same dir is
/// refused and the first finishes untouched. It passes today through the
/// store lock; the race between the lock's release and the reset has no
/// black-box reproduction and is fixed by control flow.
#[test]
fn a_second_replay_on_the_same_private_dir_is_refused_while_the_first_runs() {
    let dir = TestDir::new();
    let first_report = dir.path("first.json");
    let mut first = Command::new(env!("CARGO_BIN_EXE_asphodel"))
        .arg("replay")
        .arg("--scenario")
        .arg(scenario("three-week-holiday"))
        .arg("--report")
        .arg(&first_report)
        .arg("--until")
        .arg("2032-01-01T00:00:00Z")
        .env("ASPHODEL_REPLAY_DIR", dir.replay_dir())
        .env_remove("ASPHODEL_MODEL_DIR")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let store_db = dir.replay_dir().join("store/asphodel.db");
    let started = Instant::now();
    while !store_db.exists() {
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the first replay never opened its store"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let second = replay(&dir, &scenario("extraction-latency"), &[]);
    second.assert_refused("another replay");
    let status = first.wait().unwrap();
    assert!(status.success(), "the first replay was disturbed: {status}");
    let report: Value = serde_json::from_slice(&fs::read(&first_report).unwrap()).unwrap();
    assert!(
        report["probes"]
            .as_array()
            .unwrap()
            .iter()
            .all(|probe| probe["passed"] == true),
        "{}",
        report["probes"]
    );
}

/// Finding 3: the default report path joins the scenario's name unchecked,
/// so a name with a path in it writes outside the private dir. A name is
/// one filename component.
#[test]
fn a_scenario_name_that_isnt_a_filename_is_refused() {
    let dir = TestDir::new();
    let path = dir.file(
        "escaping.toml",
        &format!("name = \"../../escaped\"\ngroup = \"ci\"\n{HOME_TURN}"),
    );
    let output = Command::new(env!("CARGO_BIN_EXE_asphodel"))
        .arg("replay")
        .arg("--scenario")
        .arg(&path)
        .env("ASPHODEL_REPLAY_DIR", dir.replay_dir())
        .env_remove("ASPHODEL_MODEL_DIR")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("name"), "{stderr}");
    assert!(
        !dir.path("escaped.json").exists(),
        "the report escaped the private dir"
    );
}

/// Finding 3: `--report` is checked like the private dir. A destination
/// inside a git working tree is refused.
#[test]
fn a_report_path_inside_a_git_working_tree_is_refused() {
    let dir = TestDir::new();
    fs::create_dir_all(dir.path("repo/.git")).unwrap();
    let report = dir.path("repo/report.json");
    let run = replay(
        &dir,
        &scenario("extraction-latency"),
        &["--report", report.to_str().unwrap()],
    );
    run.assert_refused("git");
    assert!(!report.exists(), "the report was written inside the tree");
}

/// Finding 3: an existing symlink at the report path would write through
/// to wherever it points. It's refused and left as it was.
#[test]
fn a_report_path_that_is_a_symlink_is_refused() {
    let dir = TestDir::new();
    let target = dir.path("elsewhere.json");
    fs::write(&target, b"untouched").unwrap();
    let link = dir.path("report.json");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let run = replay(
        &dir,
        &scenario("extraction-latency"),
        &["--report", link.to_str().unwrap()],
    );
    run.assert_refused("symlink");
    assert_eq!(fs::read(&target).unwrap(), b"untouched");
}

/// Reserved report destinations must be rejected before resetting a
/// previously populated store. The second scenario has different memories,
/// so a reset cannot hide behind reproducing the first run's contents.
fn assert_reserved_report(destination: &str) {
    use std::os::unix::fs::MetadataExt as _;

    let dir = TestDir::new();
    let seed = inline(
        &dir,
        "reserved-report-seed",
        &format!(
            "{HOME_TURN}\n[[probe]]\nat = \"2026-01-05T09:01:00Z\"\nkind = \"exists\"\nmemory = \"home\"\n"
        ),
    );
    replay(&dir, &seed, &[]).assert_passed();
    let private = dir.replay_dir();
    let db = private.join("store/asphodel.db");
    let before = fs::read(&db).unwrap();
    let lock = private.join("lock");
    let lock_inode = fs::metadata(&lock).unwrap().ino();
    let sentinel = private.join("store/nested/pre-run-contents");
    fs::create_dir_all(sentinel.parent().unwrap()).unwrap();
    fs::write(&sentinel, b"keep the pre-run store, including descendants").unwrap();
    let marker = private.join("store/replay-store");
    let marker_before = fs::read(&marker).unwrap();

    let target = private.join(destination);
    let shadow_sentinel = destination.starts_with("shadow.db");
    if shadow_sentinel {
        fs::write(&target, b"keep the pre-run shadow file").unwrap();
    }
    let run = replay(
        &dir,
        &scenario("extraction-latency"),
        &["--report", target.to_str().unwrap()],
    );

    // Check preservation even when the exit code or diagnostic is wrong.
    assert!(
        fs::read(&db).unwrap() == before,
        "report destination {destination:?} changed the pre-run database"
    );
    assert_eq!(
        fs::read(&sentinel).unwrap(),
        b"keep the pre-run store, including descendants",
        "report destination {destination:?} reset the store"
    );
    assert_eq!(fs::read(&marker).unwrap(), marker_before);
    assert_eq!(
        fs::metadata(&lock).unwrap().ino(),
        lock_inode,
        "report destination {destination:?} replaced the private lock inode"
    );
    if shadow_sentinel {
        assert_eq!(
            fs::read(&target).unwrap(),
            b"keep the pre-run shadow file",
            "report destination {destination:?} overwrote the shadow sentinel"
        );
    }
    run.assert_refused("reserved");
}

#[test]
fn a_report_over_the_replay_lock_is_refused() {
    assert_reserved_report("lock");
}

#[test]
fn a_report_over_the_shadow_table_is_refused() {
    assert_reserved_report("shadow.db");
}

#[test]
fn a_report_over_the_shadow_wal_is_refused() {
    assert_reserved_report("shadow.db-wal");
}

#[test]
fn a_report_inside_the_store_is_refused() {
    assert_reserved_report("store/asphodel.db");
}

/// Finding 5: the run ends at the last event, so a completion scheduled
/// after it never happens and the report says success with nothing
/// extracted. Accepted work is drained before the run ends.
#[test]
fn accepted_sources_are_extracted_even_after_the_last_probe() {
    let dir = TestDir::new();
    let path = inline(&dir, "trailing", &format!("latency = \"10m\"\n{HOME_TURN}"));
    let run = replay(&dir, &path, &[]);
    assert_eq!(run.code(), Some(0), "stderr: {}", run.stderr());
    let report = run.report();
    assert_eq!(report["extraction_lag"]["samples"], 1, "{report}");
    assert_eq!(report["extraction_lag"]["p50_ms"], 600_000, "{report}");
    assert!(
        report["llm"]["scripted"].as_u64().is_some_and(|n| n >= 1),
        "{report}"
    );
}

/// Finding 6: two probes with one id would share a probe session and, for
/// `injects`, could evict each other's pending injections. Ids are unique
/// and the loader says so.
#[test]
fn duplicate_probe_ids_are_refused_before_the_run() {
    let dir = TestDir::new();
    let path = inline(
        &dir,
        "twins",
        &format!(
            "{HOME_TURN}
[[probe]]
id = \"twin\"
at = \"2026-01-05T10:00:00Z\"
kind = \"exists\"
memory = \"home\"

[[probe]]
id = \"twin\"
at = \"2026-01-05T11:00:00Z\"
kind = \"exists\"
memory = \"home\"
"
        ),
    );
    let run = replay(&dir, &path, &[]);
    run.assert_refused("twin");
}

/// Finding 6: probe sessions are `probe:<id>`, and no scenario session may
/// start with `probe:`, so an `injects` probe can never touch a scenario
/// session's pending injection or idle timeout.
#[test]
fn a_scenario_session_in_the_probe_namespace_is_refused() {
    let dir = TestDir::new();
    let path = inline(
        &dir,
        "reserved",
        &HOME_TURN.replace("session = \"s1\"", "session = \"probe:p1\""),
    );
    let run = replay(&dir, &path, &[]);
    run.assert_refused("probe:");
}

/// Finding 7: TIM-96 decision 4 keys a memory id by its source and claim
/// ordinal. The source is the one ingest minted for the turn's key, and the
/// ordinal is the claim's chunk position and index in call 1's reply, so a
/// document's chunks can't collide.
#[test]
fn memory_ids_are_uuidv5_of_the_source_and_claim_ordinal() {
    let dir = TestDir::new();
    let run = replay(&dir, &scenario("extraction-latency"), &[]);
    run.assert_passed();
    let observed = &run.probe("first-turn-extracted-after-its-latency")["observed"]["id"];
    let observed: uuid::Uuid = observed.as_str().unwrap().parse().unwrap();
    // The first bank in a fresh store, session `s1`, the first turn's time.
    let at: jiff::Timestamp = "2026-01-05T09:00:00Z".parse().unwrap();
    let source = uuid::Uuid::new_v5(
        &asphodel_core::store::ids::NAMESPACE,
        format!("1:turn:s1:{}", at.as_microsecond()).as_bytes(),
    );
    let expected = uuid::Uuid::new_v5(&source, b"0:0");
    assert_eq!(
        observed, expected,
        "the memory id isn't keyed by its source"
    );
}
