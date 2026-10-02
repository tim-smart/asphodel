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
    assert_eq!(purges(report), 3, "{}", report["purges_per_day"]);
    let shadow = &report["purged_then_re_mentioned"];
    assert_eq!(shadow["purged"], 3, "{shadow}");
    assert_eq!(shadow["re_mentioned"], 1, "{shadow}");
    let rate = shadow["rate"].as_f64().expect("a rate");
    assert!((rate - 1.0 / 3.0).abs() < 1e-3, "{shadow}");
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

#[test]
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

#[test]
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
