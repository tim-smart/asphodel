//! The replay harness on scripted scenarios, including deletion policy
//! and memory lifetimes. The contract these tests pin is `docs/replay.md`.
//!
//! The tests drive the binary as a process, as `serve_http.rs` does, and
//! read the JSON report, so nothing here depends on how the engine is laid
//! out inside. The scenario files under `scenarios/` are the fixtures, and
//! every checked-in one runs here with its probes.

mod support;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use support::TestDir;

/// The checked-in scenarios.
const SCENARIOS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../scenarios");

fn scenario(name: &str) -> PathBuf {
    Path::new(SCENARIOS).join(format!("{name}.toml"))
}

/// One `asphodel replay` run: the process output and the report it wrote,
/// if it wrote one.
struct Run {
    scenario: PathBuf,
    output: Output,
    report_path: PathBuf,
    report: Option<Value>,
}

impl Run {
    fn code(&self) -> Option<i32> {
        self.output.status.code()
    }

    fn stderr(&self) -> String {
        support::stderr(&self.output)
    }

    fn report(&self) -> &Value {
        self.report.as_ref().unwrap_or_else(|| {
            panic!(
                "{}: no report at {}: {}",
                self.scenario.display(),
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
            "{}: exit {:?}, failed probes: {failed:#?}, stderr: {}",
            self.scenario.display(),
            self.code(),
            self.stderr()
        );
    }

    /// Exit 2 with `word` in the message and no report.
    fn assert_refused(&self, word: &str) {
        support::assert_refused(&self.output, word);
        assert!(self.report.is_none(), "a refused run writes no report");
    }
}

/// `asphodel replay --scenario <scenario>` with `ASPHODEL_REPLAY_DIR` set
/// to the test's private dir.
fn command(dir: &TestDir, scenario: &Path) -> Command {
    let mut command = support::asphodel(dir);
    command.arg("replay").arg("--scenario").arg(scenario);
    command
}

/// Runs `command` and reads the report it was to write at `report_path`.
fn run(scenario: &Path, mut command: Command, report_path: PathBuf) -> Run {
    let output = command.output().unwrap();
    let report = fs::read(&report_path)
        .ok()
        .map(|bytes| serde_json::from_slice(&bytes).expect("the report is JSON"));
    Run {
        scenario: scenario.to_path_buf(),
        output,
        report_path,
        report,
    }
}

/// Runs `asphodel replay` on `scenario` with `--report` to a fresh file
/// and `extra` after.
fn replay(dir: &TestDir, scenario: &Path, extra: &[&str]) -> Run {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let report_path = dir.path(&format!(
        "report-{}.json",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut replay = command(dir, scenario);
    replay.arg("--report").arg(&report_path).args(extra);
    run(scenario, replay, report_path)
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

/// A scenario written by a test.
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

// The fixtures.

/// Every checked-in scenario loads, runs with no LLM, writes its report to
/// the default path under its own name and passes its probes, so a new
/// scenario file is covered by adding it. They run side by side, each in a
/// private dir of its own.
#[test]
fn every_checked_in_scenario_passes_its_probes() {
    let paths: Vec<PathBuf> = fs::read_dir(SCENARIOS)
        .expect("the scenarios directory exists")
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "toml"))
        .collect();
    assert!(!paths.is_empty(), "no scenarios are checked in");
    std::thread::scope(|scope| {
        for path in &paths {
            scope.spawn(move || {
                let dir = TestDir::new();
                let stem = path.file_stem().unwrap().to_str().unwrap();
                let report_path = dir.private().join(format!("reports/{stem}.json"));
                let run = run(path, command(&dir, path), report_path);
                run.assert_passed();
                let report = run.report();
                assert_eq!(report["scenario"], stem, "the name is the file stem");
                assert!(
                    !run.probes().is_empty(),
                    "{stem}: a scenario without probes checks nothing"
                );
                assert_eq!(report["llm"]["live"], 0, "{stem}");
                if stem == "purge-table" {
                    // Purge spares the reinforced memories and the major
                    // one, not the single minor/notable mentions, and one
                    // is said again.
                    let shadow = &report["purged_then_re_mentioned"];
                    assert_eq!(purges(report), 5, "{}", report["purges_per_day"]);
                    assert_eq!(shadow["purged"], 5, "{shadow}");
                    assert_eq!(shadow["re_mentioned"], 1, "{shadow}");
                }
                // Refinements rejected across kinds are counted by what made
                // them refinements; `promoted` counts only those that landed.
                let rejected = |explicit: u64, date_promoted: u64, weight_promoted: u64| {
                    json!({
                        "explicit": explicit,
                        "date_promoted": date_promoted,
                        "weight_promoted": weight_promoted,
                    })
                };
                let call2 = &report["call2_rate"];
                if stem == "relationship-outweighs-repeat" {
                    assert_eq!(call2["promoted"], 0, "{stem}");
                    assert_eq!(call2["refines_rejected"], rejected(0, 0, 1), "{stem}");
                }
                if stem == "preference-survives-mislabels" {
                    // The weightier repeat of the refinement.
                    assert_eq!(call2["promoted"], 1, "{stem}");
                    assert_eq!(call2["refines_rejected"], rejected(2, 1, 0), "{stem}");
                }
            });
        }
    });
}

/// Usage is counted per injected memory, not per turn or LLM call, in the
/// report and the aggregate export alike. Unjudged memories are excluded
/// from the fraction; zero judged is reported as 0.0.
#[test]
fn injection_usage_counts_mixed_verdicts_in_report_and_aggregate() {
    let dir = TestDir::new();
    fn usage(used: u64, not_used: u64, fraction: f64) -> Value {
        json!({ "used": used, "not_used": not_used, "unjudged": 0, "used_fraction": fraction })
    }
    for (path, expected) in [
        (scenario("injection-usage"), usage(1, 3, 0.25)),
        (inline(&dir, "no-injection", HOME_TURN), usage(0, 0, 0.0)),
    ] {
        let aggregate = dir.path("aggregate.json");
        let run = replay(&dir, &path, &["--aggregate", aggregate.to_str().unwrap()]);
        run.assert_passed();
        let export: Value = serde_json::from_slice(&fs::read(aggregate).unwrap()).unwrap();
        let path = path.display();
        assert_eq!(run.report()["injection_usage"], expected, "{path}");
        assert_eq!(export["injection_usage"], expected, "{path}");
    }
}

/// The private census identifies each incompatible decision, while both
/// exports count it even in the control arm where it is allowed to land.
#[test]
fn refinement_census_and_exports_agree_with_both_kind_guard_settings() {
    for (cause, outcome, significance, date, older) in [
        ("explicit", "refines", "minor", "", false),
        ("explicit", "refines", "minor", "", true),
        (
            "date_promoted",
            "confirmed",
            "minor",
            "valid_from = { at = \"2026-01-20\", precision = \"day\" }",
            false,
        ),
        ("weight_promoted", "mentioned_again", "major", "", false),
    ] {
        for guard in [true, false] {
            let dir = TestDir::new();
            // An accepted older refinement edits the existing head without
            // creating a memory; a rejected one becomes an independent head.
            let creates_memory = !older || guard;
            let label = if creates_memory {
                "label = \"visit\""
            } else {
                ""
            };
            let claim_probe = if creates_memory {
                r#"[[probe]]
id = "claim"
at = "2026-01-06T10:00:00Z"
kind = "exists"
memory = "visit""#
            } else {
                ""
            };
            let source = if older {
                r#"[[document]]
at = "2026-01-06T09:00:00Z"
id = "old-notes"
reference_date = "2026-01-01"
text = "Tim lived in Auckland."

[[document.claim]]"#
            } else {
                r#"[[turn]]
at = "2026-01-06T09:00:00Z"
session = "s1"
user = "Tim lived in Auckland."
assistant = "Noted."

[[turn.claim]]"#
            };
            let path = inline(
                &dir,
                "refinement-census",
                &format!(
                    r#"{HOME_TURN}
{source}
{label}
content = "Tim lived in Auckland."
quote = "Tim lived in Auckland"
kind = "event"
significance = "{significance}"
{date}
reconcile = [{{ memory = "home", outcome = "{outcome}" }}]

[[probe]]
id = "original"
at = "2026-01-06T10:00:00Z"
kind = "exists"
memory = "home"

{claim_probe}
"#
                ),
            );
            let overrides = dir.file(
                "guard.toml",
                &format!("[reconcile]\nkind_guard = {guard}\n"),
            );
            let aggregate = dir.path("aggregate.json");
            let run = replay(
                &dir,
                &path,
                &[
                    "--overrides",
                    overrides.to_str().unwrap(),
                    "--aggregate",
                    aggregate.to_str().unwrap(),
                ],
            );
            run.assert_passed();
            let report = run.report();
            let rows = report["kind_mismatches"].as_array().unwrap();
            assert_eq!(rows.len(), 1, "{cause}, older={older}, guard={guard}");
            let row = &rows[0];
            assert!(uuid::Uuid::parse_str(row["chunk"].as_str().unwrap()).is_ok());
            assert_eq!(row["claim"], 0);
            if creates_memory {
                assert_eq!(row["memory"], run.probe("claim")["observed"]["id"]);
            } else {
                assert!(row["memory"].is_null());
            }
            assert_eq!(row["neighbour"], run.probe("original")["observed"]["id"]);
            assert_eq!(row["claim_kind"], "event");
            assert_eq!(row["neighbour_kind"], "fact");
            assert_eq!(row["cause"], cause);
            assert_eq!(row["older"], older);
            assert_eq!(row["rejected"], guard);

            let mut across = json!({ "explicit": 0, "date_promoted": 0, "weight_promoted": 0 });
            across[cause] = json!(1);
            let rejected = if guard {
                across.clone()
            } else {
                json!({ "explicit": 0, "date_promoted": 0, "weight_promoted": 0 })
            };
            let export: Value = serde_json::from_slice(&fs::read(aggregate).unwrap()).unwrap();
            for counters in [&report["call2_rate"], &export["call2_rate"]] {
                assert_eq!(counters["refines_across_kinds"], across);
                assert_eq!(counters["refines_rejected"], rejected);
            }
        }
    }
}

/// Every restatement a run writes is counted, in the report and the
/// aggregate export alike; a run that absorbs nothing writes none.
#[test]
fn restatements_written_are_counted_in_report_and_aggregate() {
    let dir = TestDir::new();
    for (path, written) in [
        (scenario("absorbed-restatement"), 2),
        (inline(&dir, "nothing-absorbed", HOME_TURN), 0),
    ] {
        let aggregate = dir.path("aggregate.json");
        let run = replay(&dir, &path, &["--aggregate", aggregate.to_str().unwrap()]);
        run.assert_passed();
        let export: Value = serde_json::from_slice(&fs::read(aggregate).unwrap()).unwrap();
        let path = path.display();
        assert_eq!(run.report()["restatements"]["written"], written, "{path}");
        assert_eq!(export["restatements"]["written"], written, "{path}");
    }
}

/// Each loader error, as a sentence naming the label at fault.
const LOADER_ERRORS: &str = r#"
[[turn]]
at = "2026-01-06T09:00:00Z"
session = "s1"
user = "I live in Auckland, as I said."
assistant = "You did."
used = ["nowhere"]

[[turn.claim]]
content = "Tim lives in Auckland."
quote = "I live in Auckland"
kind = "fact"
significance = "minor"
reconcile = [{ memory = "home", outcome = "mentioned_again" }]

[[turn.claim]]
label = "later"
content = "Tim lives in Wellington."
quote = "not in the turn"
kind = "fact"
significance = "minor"
reconcile = [{ memory = "future", outcome = "refines" }]

[[probe]]
at = "2026-01-01T00:00:00Z"
kind = "band"
memory = "home"
band = "strong"

[[probe]]
id = "ghost"
at = "2026-02-01T00:00:00Z"
kind = "absent"
memory = "nobody"

[[probe]]
id = "backwards"
at = "2026-01-07T00:00:00Z"
kind = "faded_at"
memory = "home"
between = ["2026-02-01T00:00:00Z", "2026-01-20T00:00:00Z"]
"#;

/// A second weather turn whose unlabelled claim reconciles against `home`,
/// which isn't among the neighbours call 2 is shown for it. An absorbing
/// outcome can't label a new memory, so the loader lets it through.
const NOT_A_NEIGHBOUR: &str = r#"
[[turn]]
at = "2026-01-05T10:00:00Z"
session = "s1"
user = "The weather is lovely today."
assistant = "Enjoy it."

[[turn.claim]]
label = "weather"
content = "The weather was lovely on 2026-01-05."
quote = "The weather is lovely today"
kind = "event"
significance = "trivial"

[[turn]]
at = "2026-01-06T09:00:00Z"
session = "s1"
user = "The weather is lovely today."
assistant = "Enjoy it."

[[turn.claim]]
content = "The weather was lovely on 2026-01-06."
quote = "The weather is lovely today"
kind = "event"
significance = "trivial"
reconcile = [{ memory = "home", outcome = "mentioned_again" }]
"#;

/// A labelled restatement of `home` that matters no more than it, so the
/// repeat absorbs it and the label names nothing. The loader can't tell
/// (code may make a repeat a new memory), so the run refuses it.
const ABSORBED_LABEL: &str = r#"
[[turn]]
at = "2026-01-06T09:00:00Z"
session = "s1"
user = "I live in Auckland, as I said."
assistant = "You did."

[[turn.claim]]
label = "home-again"
content = "Tim lives in Auckland."
quote = "I live in Auckland"
kind = "fact"
significance = "minor"
reconcile = [{ memory = "home", outcome = "mentioned_again" }]
"#;

/// A new session whose query shares no word with the memory: nothing is
/// injected, so `home` isn't in context to use.
const USED_OUT_OF_CONTEXT: &str = r#"
[[turn]]
at = "2026-01-06T09:00:00Z"
session = "s2"
user = "Tell me a joke."
assistant = "Why did the chicken cross the road?"
used = ["home"]
"#;

/// Two probes with one id would share a probe session and, for `injects`,
/// could evict each other's pending injections.
const TWIN_PROBES: &str = r#"
[[probe]]
id = "twin"
at = "2026-01-05T10:00:00Z"
kind = "exists"
memory = "home"

[[probe]]
id = "twin"
at = "2026-01-05T11:00:00Z"
kind = "exists"
memory = "home"
"#;

/// The engine never guesses (docs/replay.md, "Claims"): a scenario that
/// doesn't hold together is refused with exit 2, naming what's wrong, and
/// writes no report, including to the default path its name gives.
#[test]
fn an_invalid_scenario_is_refused_without_a_report() {
    let dir = TestDir::new();
    let concurrency = dir.file("concurrency.toml", "[llm]\nconcurrency = 2\n");
    let pooled = ["--overrides", concurrency.to_str().unwrap()];
    let with_home = |body: &str| format!("{HOME_TURN}{body}");
    let loader = [
        "nowhere",
        "future",
        "not in the turn",
        "p1",
        "ghost",
        "backwards",
    ];
    // Probe sessions are `probe:<id>`, so no scenario session may start
    // with `probe:`. The default report path is `reports/<name>.json`, so
    // the name is one filename component. A pooled run's call 2 replies
    // are scripted against what a serial run shows.
    let reserved = HOME_TURN.replace("session = \"s1\"", "session = \"probe:p1\"");
    let cases: [(&str, String, &[&str], &[&str]); 9] = [
        ("loader", with_home(LOADER_ERRORS), &[], &loader),
        (
            "unknown-field",
            "start = \"2026-01-01T00:00:00Z\"\n".into(),
            &[],
            &["start"],
        ),
        ("twins", with_home(TWIN_PROBES), &[], &["twin"]),
        ("reserved", reserved, &[], &["probe:"]),
        ("../../escaped", HOME_TURN.into(), &[], &["name"]),
        (
            "not-a-neighbour",
            with_home(NOT_A_NEIGHBOUR),
            &[],
            &["home", "neighbour"],
        ),
        (
            "absorbed-label",
            with_home(ABSORBED_LABEL),
            &[],
            &["home-again"],
        ),
        (
            "used-out-of-context",
            with_home(USED_OUT_OF_CONTEXT),
            &[],
            &["home"],
        ),
        ("pooled", HOME_TURN.into(), &pooled, &["concurrency"]),
    ];
    for (name, body, extra, words) in cases {
        let file = name.trim_start_matches("../../");
        let path = dir.file(
            &format!("{file}.toml"),
            &format!("name = \"{name}\"\ngroup = \"ci\"\n{body}"),
        );
        let output = command(&dir, &path).args(extra).output().unwrap();
        let stderr = support::stderr(&output);
        assert_eq!(output.status.code(), Some(2), "{name}: {stderr}");
        for word in words {
            assert!(
                stderr.contains(word),
                "{name} should name {word:?}: {stderr}"
            );
        }
        assert!(
            !dir.private().join(format!("reports/{file}.json")).exists()
                && !dir.path("escaped.json").exists(),
            "{name} wrote a report"
        );
    }
}

#[test]
fn a_repeated_errand_gains_a_date_and_leaves_the_agenda_after_its_event() {
    use std::sync::Arc;

    use asphodel_core::clock::SimulatedClock;
    use asphodel_core::config::Tuning;
    use asphodel_core::service::Service;
    use asphodel_core::store::{OpenOptions, Store};

    let dir = TestDir::new();
    let run = replay(&dir, &scenario("reminder-adds-date"), &[]);
    let old = run.probe("undated-errand-on-agenda")["observed"]["id"]
        .as_str()
        .expect("the original errand exists");
    let clock = Arc::new(SimulatedClock::new("2026-02-10T13:59:00Z".parse().unwrap()));
    let store = Store::open(
        &dir.path("private/store"),
        OpenOptions::default(),
        clock.clone(),
    )
    .unwrap();
    let service = Service::open(clock.clone(), store, Tuning::default());
    let original = service.show_memory("main", old).unwrap();
    let head = service
        .show_memory("main", &original.chain.head.to_string())
        .unwrap();
    assert_eq!(
        head.window.due_at.map(|time| time.at),
        Some("2026-02-10T14:00:00Z".parse().unwrap()),
        "the repeated reminder must leave a dated head"
    );
    assert_eq!(
        head.window.valid_until.map(|time| time.at),
        Some("2026-02-10T14:00:00Z".parse().unwrap())
    );
    // The dated head is actionable before its event, then leaves the agenda.
    assert!(service.agenda("main").unwrap().listed().contains(&head.id));
    clock.set("2026-02-10T14:01:00Z".parse().unwrap());
    assert!(!service.agenda("main").unwrap().listed().contains(&head.id));
    run.assert_passed();
}

#[test]
fn p18_memory_show_has_no_overdue_task_guard_after_completion() {
    use std::sync::Arc;

    use asphodel_core::clock::SimulatedClock;
    use asphodel_core::config::Tuning;
    use asphodel_core::inspect::Guard;
    use asphodel_core::service::Service;
    use asphodel_core::store::{OpenOptions, Store};

    let dir = TestDir::new();
    let run = replay(&dir, &scenario("p18"), &[]);
    // Inspect even if the replay probes fail, so this guard is checked
    // independently of the closure and agenda assertions.
    let id = run.probe("p18-2")["observed"]["id"]
        .as_str()
        .expect("p18-2 resolves its memory");
    let clock = Arc::new(SimulatedClock::new("2026-01-12T09:00:00Z".parse().unwrap()));
    let store = Store::open(
        &dir.path("private/store"),
        OpenOptions::default(),
        clock.clone(),
    )
    .unwrap();
    let service = Service::open(clock, store, Tuning::default());
    // This is the same view used by `memory show`, after the final correction closes.
    let view = service.show_memory("main", id).unwrap();
    assert!(
        !view
            .purge
            .guards
            .iter()
            .any(|guard| matches!(guard, Guard::OverdueTask { .. })),
        "p18: {:?}",
        view.purge.guards
    );
}

/// With no latency the first turn's memory exists at 09:05, so the probes
/// timed for the scenario's own latency fail, the run exits 1, and the
/// report still says so.
#[test]
fn extraction_latency_is_modelled_in_simulated_time() {
    let dir = TestDir::new();
    let run = replay(&dir, &scenario("extraction-latency"), &["--latency", "0s"]);
    assert_eq!(run.code(), Some(1), "stderr: {}", run.stderr());
    for (id, passed) in [
        ("first-turn-not-yet-extracted", false),
        ("second-turn-waits-for-the-worker", false),
        ("first-turn-extracted-after-its-latency", true),
    ] {
        assert_eq!(run.probe(id)["passed"], passed, "{id}");
    }
}

#[test]
fn until_keeps_sweeping_past_the_last_turn() {
    let dir = TestDir::new();
    let run = replay(&dir, &scenario("extraction-latency"), &[]);
    run.assert_passed();
    assert_eq!(purges(run.report()), 0);

    // A year of nightly sweeps purges both trivial memories.
    let until = ["--until", "2027-01-05T09:00:00Z"];
    let run = replay(&dir, &scenario("extraction-latency"), &until);
    run.assert_passed();
    let report = run.report();
    assert_eq!(purges(report), 2, "{}", report["purges_per_day"]);
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
    // The report of a passing run until `until`.
    let until = |path: &Path, until: &str| {
        let run = replay(&dir, path, &["--until", until]);
        run.assert_passed();
        run.report().clone()
    };
    let refreshes = |path: &Path, at: &str| until(path, at)["refresh_calls_per_day"].clone();
    assert_eq!(refreshes(&path, "2026-01-05T09:04:59Z"), json!([]));
    let debounced = refreshes(&path, "2026-01-05T09:05:00Z");
    assert_eq!(debounced, json!([{ "day": "2026-01-05", "count": 1 }]));

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
    assert_eq!(refreshes(&daily_path, "2026-01-05T14:59:59Z"), debounced);
    let swept = until(&daily_path, "2026-01-05T15:00:00Z");
    // The seeded "User profile" also refreshes at the sweep.
    // Only the explicit model had a creation-triggered debounce earlier.
    assert_eq!(
        swept["refresh_calls_per_day"],
        json!([
            { "day": "2026-01-05", "count": 1 },
            { "day": "2026-01-06", "count": 2 }
        ])
    );
    assert_eq!(swept["llm"]["live"], 0);
    assert_eq!(
        refreshes(&daily_path, "2026-01-06T15:00:00Z"),
        swept["refresh_calls_per_day"],
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

    let other = TestDir::new();
    let repeated = replay(&other, &path, &[]);
    repeated.assert_passed();
    assert_eq!(
        fs::read(&run.report_path).unwrap(),
        fs::read(&repeated.report_path).unwrap()
    );
}

// Overrides and layering.

#[test]
fn overrides_layer_over_the_production_file_in_the_shape_of_tuning() {
    let dir = TestDir::new();
    let config = dir.file("production.toml", "[clock]\nquiet_rate = 0.2\n");
    let overrides = dir.file("overrides.toml", "[clock]\nquiet_rate = 0.5\n");
    let path = scenario("extraction-latency");

    let config = ["--config", config.to_str().unwrap()];
    let quiet_rate = |extra: &[&str]| {
        let run = replay(&dir, &path, extra);
        run.assert_passed();
        run.report()["tuning"]["clock"]["quiet_rate"].clone()
    };
    // The scenario's own [tuning] (1.0) sits above the production file,
    // and --overrides above the scenario.
    assert_eq!(quiet_rate(&config), 1.0);
    let overrides = ["--overrides", overrides.to_str().unwrap()];
    assert_eq!(quiet_rate(&[config, overrides].concat()), 0.5);
}

// Privacy and the private directory.

/// Replay writes only under a private dir it owns. No private dir, one
/// inside a git working tree, a serve data dir (recognised by the store
/// file at its top level), and a store dir replay didn't create, such as a
/// stopped daemon's data dir named `store`, are each refused before
/// anything is written or reset.
#[test]
fn replay_refuses_a_private_dir_it_doesnt_own() {
    let path = scenario("extraction-latency");

    let unset = TestDir::new();
    let report = unset.path("report.json");
    let output = command(&unset, &path)
        .env_remove("ASPHODEL_REPLAY_DIR")
        .arg("--report")
        .arg(&report)
        .output()
        .unwrap();
    support::assert_refused(&output, "ASPHODEL_REPLAY_DIR");
    assert!(!report.exists());

    let in_git = TestDir::new();
    fs::create_dir_all(in_git.path("repo/.git")).unwrap();
    let private = in_git.path("repo/private");
    fs::create_dir_all(&private).unwrap();
    replay(&in_git, &path, &["--replay-dir", private.to_str().unwrap()]).assert_refused("git");
    assert!(
        fs::read_dir(&private).unwrap().next().is_none(),
        "nothing is written inside the working tree"
    );

    let data_dir = TestDir::new();
    data_dir.private_file("asphodel.db", "");
    replay(&data_dir, &path, &[]).assert_refused("serve");

    let unowned = TestDir::new();
    unowned.private_file("store/asphodel.db", "");
    let sentinel = unowned.private_file("store/sentinel", "a daemon's file");
    replay(&unowned, &path, &[]).assert_refused("store");
    assert!(
        sentinel.is_file(),
        "the sentinel in the unowned store was removed"
    );
}

/// A replay holds its private dir for its whole run, so a second replay
/// on the same dir is refused and the first finishes untouched. This
/// exercises lock contention; a race between lock release and store reset
/// has no black-box reproduction and must be prevented by control flow.
#[test]
fn a_second_replay_on_the_same_private_dir_is_refused_while_the_first_runs() {
    let dir = TestDir::new();
    let first_report = dir.path("first.json");
    let mut first = command(&dir, &scenario("three-week-holiday"))
        .arg("--report")
        .arg(&first_report)
        .args(["--until", "2032-01-01T00:00:00Z"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let store_db = dir.private().join("store/asphodel.db");
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
    let probes = report["probes"].as_array().unwrap();
    assert!(
        probes.iter().all(|probe| probe["passed"] == true),
        "{probes:?}"
    );
}

/// `--report` is checked like the private dir, before a run resets
/// anything. A destination inside a git working tree, an existing symlink
/// that would write through to wherever it points, and a file replay keeps
/// in the private dir (the lock, the shadow table and its WAL, the store)
/// are each refused and left as they were. The run over replay's own files
/// follows a different scenario, so a reset can't hide behind reproducing
/// the first run's contents.
#[test]
fn a_report_path_replay_cant_write_safely_is_refused() {
    use std::os::unix::fs::MetadataExt as _;

    let dir = TestDir::new();
    let path = scenario("extraction-latency");
    fs::create_dir_all(dir.path("repo/.git")).unwrap();
    let in_git = dir.path("repo/report.json");
    replay(&dir, &path, &["--report", in_git.to_str().unwrap()]).assert_refused("git");
    assert!(!in_git.exists(), "the report was written inside the tree");

    let target = dir.file("elsewhere.json", "untouched");
    let link = dir.path("link.json");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    replay(&dir, &path, &["--report", link.to_str().unwrap()]).assert_refused("symlink");
    assert_eq!(fs::read(&target).unwrap(), b"untouched");

    for destination in ["lock", "shadow.db", "shadow.db-wal", "store/asphodel.db"] {
        let dir = TestDir::new();
        let seed = inline(
            &dir,
            "reserved-report-seed",
            &format!(
                "{HOME_TURN}\n[[probe]]\nat = \"2026-01-05T09:01:00Z\"\nkind = \"exists\"\nmemory = \"home\"\n"
            ),
        );
        replay(&dir, &seed, &[]).assert_passed();
        dir.private_file("store/nested/pre-run-contents", "keep the pre-run store");
        let shadow = destination.starts_with("shadow.db");
        if shadow {
            dir.private_file(destination, "keep the pre-run shadow file");
        }
        // Everything the run must leave as it was: the store's database,
        // its marker and the sentinel inside it, any shadow file there, and
        // the lock's inode.
        let private = dir.private();
        let snapshot = || {
            let files = [
                "store/asphodel.db",
                "store/replay-store",
                "store/nested/pre-run-contents",
            ];
            let mut bytes: Vec<Vec<u8>> = files
                .iter()
                .map(|file| fs::read(private.join(file)).unwrap())
                .collect();
            if shadow {
                bytes.push(fs::read(private.join(destination)).unwrap());
            }
            (bytes, fs::metadata(private.join("lock")).unwrap().ino())
        };
        let before = snapshot();
        let target = private.join(destination);
        let run = replay(&dir, &path, &["--report", target.to_str().unwrap()]);
        // Check preservation even when the exit code or diagnostic is wrong.
        assert!(
            snapshot() == before,
            "report destination {destination:?} changed the pre-run private dir"
        );
        run.assert_refused("reserved");
    }
}

/// Accepted work is drained before the run ends, including completions
/// scheduled after the last event, so extraction appears in the report.
#[test]
fn accepted_sources_are_extracted_even_after_the_last_probe() {
    let dir = TestDir::new();
    let path = inline(&dir, "trailing", &format!("latency = \"10m\"\n{HOME_TURN}"));
    let run = replay(&dir, &path, &[]);
    assert_eq!(run.code(), Some(0), "stderr: {}", run.stderr());
    let report = run.report();
    assert_eq!(report["extraction_lag"]["samples"], 1, "{report}");
    assert_eq!(report["extraction_lag"]["p50_ms"], 600_000, "{report}");
}
