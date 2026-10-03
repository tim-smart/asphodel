//! Real-history replay on a synthetic history: LLM recording modes,
//! deterministic reports, private aggregate exports and A/B diffs.
//!
//! The history is `support::hermes::small_history`, imported with
//! `asphodel import`. `live` runs on a scripted stand-in for the LLM
//! (`ASPHODEL_LLM_SCRIPT`) and records the cassette; `replay` and `fast`
//! run with no LLM available at all, so anything they need must come from
//! the cassette.

mod support;

use std::collections::BTreeSet;
use std::fs;

use serde_json::Value;
use sha2::{Digest, Sha256};
use support::hermes::{self, StateDb, epoch};
use support::{
    HOME_QUESTION, PASSING_PROBES, PROBES_WITH_A_FAILURE, TestDir, add_entry, asphodel, assert_ok,
    assert_refused, cassette_records, imported_small_history, imported_with_a_model, judge_script,
    live_script, record, replay_history, stderr, universal_script, write_cassette,
};

/// What a run simulated, without what identifies the run: the mode
/// (`kind`), the flags it was invoked with, where its LLM replies came
/// from (`llm`) and the cassette hash it started from. A `live` run and a
/// `replay` of its cassette differ in exactly those, so everything else in
/// their reports must match.
fn simulation(report: &Value) -> Value {
    let mut report = report.clone();
    let object = report.as_object_mut().expect("the report is an object");
    for key in ["kind", "flags", "llm", "cassette_hash"] {
        object.remove(key);
    }
    report
}

// Recording modes.

/// `live` calls the LLM and records; `replay` of that cassette needs no
/// LLM and simulates the same run.
#[test]
fn replay_of_a_live_cassette_simulates_the_same_run_without_an_llm() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let live = record(&dir, &corpus);
    let replay = replay_history(&dir, &corpus, "replay", PASSING_PROBES, "replay", None, &[]);
    assert_ok(&replay.output);

    let live = live.report();
    let replay = replay.report();
    assert_eq!(live["kind"], "live");
    assert_eq!(replay["kind"], "replay");
    assert!(
        live["llm"]["live"].as_u64().is_some_and(|n| n > 0),
        "{live}"
    );
    assert_eq!(replay["llm"]["live"], 0, "{replay}");
    assert!(
        replay["llm"]["cache"].as_u64().is_some_and(|n| n > 0),
        "{replay}"
    );
    assert_eq!(simulation(&live), simulation(&replay));
}

/// With the same corpus, cassette and overrides,
/// `replay` writes a byte-identical report.
#[test]
fn repeated_replay_runs_write_byte_identical_reports() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let first = replay_history(&dir, &corpus, "replay", PASSING_PROBES, "first", None, &[]);
    let second = replay_history(&dir, &corpus, "replay", PASSING_PROBES, "second", None, &[]);
    assert_ok(&first.output);
    assert_ok(&second.output);
    assert_eq!(
        first.report_bytes(),
        second.report_bytes(),
        "two replays of the same configuration differ"
    );
}

/// Replay takes `[llm] concurrency` from the overrides like the daemon,
/// with up to that many chunks out at once in simulated time. The small
/// history never needs call 2, so a run at 5 replays the cassette recorded
/// at 1 without a miss and ends with the same memories.
#[test]
fn replay_at_concurrency_five_ends_with_the_memories_of_a_serial_run() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let serial = record(&dir, &corpus).report();
    let five = dir.private_file("five.toml", "[llm]\nconcurrency = 5\n");
    let pooled = replay_history(
        &dir,
        &corpus,
        "replay",
        PASSING_PROBES,
        "five",
        None,
        &["--overrides", five.to_str().unwrap()],
    );
    assert_ok(&pooled.output);
    let pooled = pooled.report();
    assert_eq!(pooled["llm"]["misses"], 0, "{pooled}");
    assert_eq!(pooled["memories"], serial["memories"], "{pooled}");
}

/// The small history with the home turn said again ten minutes later in
/// the same session, imported to `corpus/twice.jsonl`.
fn imported_with_home_twice(dir: &TestDir) -> std::path::PathBuf {
    let state_db = dir.private_path("state-twice.db");
    let db: StateDb = hermes::small_history(&state_db);
    db.turn(
        "s-main",
        epoch("2026-01-05T09:10:00Z"),
        &format!("{}, still.", hermes::HOME_QUOTE),
        "Noted again.",
    );
    drop(db);
    let corpus = dir.private_path("corpus/twice.jsonl");
    assert_ok(&support::import(dir, &state_db, &corpus));
    corpus
}

/// A script whose every step answers any call: call 1 with the home
/// claim, call 2 labelling it a mention of the one neighbour, and a
/// refresh with no edits. Neither parser refuses the others' fields.
fn mention_script(dir: &TestDir) -> std::path::PathBuf {
    let mut claim = support::home_claim();
    claim["claim"] = serde_json::json!("c1");
    claim["labels"] = serde_json::json!([{"neighbour": "n1", "label": "mentioned_again"}]);
    support::script_answering_everything(dir, "mention-script", vec![claim], Vec::new())
}

/// Two chunks out at once in simulated time, with an hour's latency: the
/// home turn and its repeat ten minutes later.
fn pooled_flags(dir: &TestDir) -> Vec<String> {
    let pooled = dir.private_file("pooled.toml", "[llm]\nconcurrency = 2\n");
    vec![
        "--overrides".into(),
        pooled.to_str().unwrap().into(),
        "--latency".into(),
        "1h".into(),
    ]
}

fn is_call2(record: &Value) -> bool {
    record["template"]["name"] == "reconcile_claims"
}

/// At concurrency 2 the repeat is claimed before the first home turn
/// commits, so neither sees the other at its search. The repeat's commit
/// is stale: it runs call 2 again through the cassette, live in `live`,
/// and becomes a mention of the one memory. `replay` of that cassette
/// simulates the same run.
#[test]
fn a_stale_commit_redoes_call_2_through_the_cassette_and_replays() {
    let dir = TestDir::new();
    let corpus = imported_with_home_twice(&dir);
    let script = mention_script(&dir);
    let flags = pooled_flags(&dir);
    let flags: Vec<&str> = flags.iter().map(String::as_str).collect();

    let live = replay_history(
        &dir,
        &corpus,
        "live",
        PASSING_PROBES,
        "live",
        Some(&script),
        &flags,
    );
    assert_ok(&live.output);
    let live = live.report();
    assert_eq!(live["call2_rate"]["redos"], 1, "{live}");
    assert_eq!(live["call2_rate"]["call2"], 1, "{live}");
    assert_eq!(
        live["memories"].as_array().unwrap().len(),
        1,
        "a mention, not a copy: {live}"
    );
    let call2: Vec<Value> = cassette_records(&dir)
        .into_iter()
        .filter(is_call2)
        .collect();
    assert_eq!(call2.len(), 1, "the redo's call 2 is recorded");

    let replay = replay_history(
        &dir,
        &corpus,
        "replay",
        PASSING_PROBES,
        "replay",
        None,
        &flags,
    );
    assert_ok(&replay.output);
    let replay = replay.report();
    assert_eq!(replay["llm"]["misses"], 0, "{replay}");
    assert_eq!(simulation(&live), simulation(&replay));
}

/// A redo takes as long as the call 2 that answers it. With every call 1
/// recorded at 20 minutes and the redo's call 2 at 10, the repeat (synced
/// ten minutes after the home turn, so out with it) completes at 30
/// minutes, finds itself stale and commits after its redo, 30 minutes
/// after its sync: never at the completion it had before the redo.
#[test]
fn a_redo_is_charged_the_latency_of_its_call_2() {
    let dir = TestDir::new();
    let corpus = imported_with_home_twice(&dir);
    let script = mention_script(&dir);
    let flags = pooled_flags(&dir);
    let flags: Vec<&str> = flags.iter().map(String::as_str).collect();
    let live = replay_history(
        &dir,
        &corpus,
        "live",
        PASSING_PROBES,
        "live",
        Some(&script),
        &flags,
    );
    assert_ok(&live.output);
    let minutes = |n: u64| n * 60 * 1000;
    let timed: Vec<Value> = cassette_records(&dir)
        .into_iter()
        .map(|mut record| {
            record["latency_ms"] = if is_call2(&record) {
                minutes(10).into()
            } else {
                minutes(20).into()
            };
            record
        })
        .collect();
    write_cassette(&dir, &timed);

    // Latency from the cassette: no --latency.
    let replay = replay_history(
        &dir,
        &corpus,
        "replay",
        PASSING_PROBES,
        "timed",
        None,
        &flags[..2],
    );
    assert_ok(&replay.output);
    let report = replay.report();
    assert_eq!(report["call2_rate"]["redos"], 1, "{report}");
    assert_eq!(report["extraction_lag"]["p50_ms"], minutes(20), "{report}");
    assert_eq!(
        report["extraction_lag"]["p95_ms"],
        minutes(30),
        "the repeat commits after its redo's call 2: {report}"
    );
}

/// Without the redo's call 2 in the cassette, `replay` stops on the miss
/// and `fast` answers it from the LLM and records it.
#[test]
fn a_redo_that_misses_stops_replay_and_is_answered_live_in_fast() {
    let dir = TestDir::new();
    let corpus = imported_with_home_twice(&dir);
    let script = mention_script(&dir);
    let flags = pooled_flags(&dir);
    let flags: Vec<&str> = flags.iter().map(String::as_str).collect();
    let live = replay_history(
        &dir,
        &corpus,
        "live",
        PASSING_PROBES,
        "live",
        Some(&script),
        &flags,
    );
    assert_ok(&live.output);
    let without: Vec<Value> = cassette_records(&dir)
        .into_iter()
        .filter(|record| !is_call2(record))
        .collect();
    write_cassette(&dir, &without);

    let replay = replay_history(
        &dir,
        &corpus,
        "replay",
        PASSING_PROBES,
        "replay",
        None,
        &flags,
    );
    assert_refused(&replay.output, "miss");
    assert!(
        !replay.report_path.exists(),
        "a failed run writes no report"
    );

    let fast = replay_history(
        &dir,
        &corpus,
        "fast",
        PASSING_PROBES,
        "fast",
        Some(&script),
        &flags,
    );
    assert_ok(&fast.output);
    let fast = fast.report();
    assert_eq!(fast["call2_rate"]["redos"], 1, "{fast}");
    assert_eq!(fast["llm"]["misses"], 1, "{fast}");
    assert_eq!(fast["llm"]["live"], 1, "{fast}");
    assert_eq!(
        cassette_records(&dir).into_iter().filter(is_call2).count(),
        1,
        "fast records the call it answered live"
    );
}

/// `replay` fails on a miss.
#[test]
fn replay_fails_on_a_cassette_miss() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    fs::write(dir.private_path("cassettes/main.jsonl"), b"").unwrap();
    let run = replay_history(&dir, &corpus, "replay", PASSING_PROBES, "miss", None, &[]);
    assert_refused(&run.output, "miss");
    assert!(!run.report_path.exists(), "a failed run writes no report");
}

/// `used` verdicts are cached per (reply, sentence)
/// pair, and in `fast` the pairs nobody has judged get one short top-up
/// call per chunk, recorded like any other call, so the next `fast` run
/// needs none.
///
/// The recording is made with the reranker gate shut, so nothing is
/// injected and no pair is judged. `fast` then runs with the gate open, so
/// the home memory is injected into later turns as a pair nobody has
/// judged.
#[test]
fn fast_tops_up_unjudged_pairs_once_per_chunk_and_records_them() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let shut = dir.private_file(
        "shut-gate.toml",
        "[injection.reranker_floors]\n\"fake-reranker:v1\" = 1000000.0\n",
    );
    let script = live_script(&dir);
    let live = replay_history(
        &dir,
        &corpus,
        "live",
        PASSING_PROBES,
        "live",
        Some(&script),
        &["--overrides", shut.to_str().unwrap()],
    );
    assert_ok(&live.output);
    let judged = |records: &[Value]| -> Vec<Value> {
        records
            .iter()
            .filter(|record| record["template"]["name"] == "judge_used")
            .cloned()
            .collect()
    };
    assert!(judged(&cassette_records(&dir)).is_empty());

    let judge = judge_script(&dir);
    let first = replay_history(
        &dir,
        &corpus,
        "fast",
        PASSING_PROBES,
        "fast-first",
        Some(&judge),
        &[],
    );
    assert_ok(&first.output);
    let report = first.report();
    let top_ups = judged(&cassette_records(&dir));
    assert!(
        !top_ups.is_empty(),
        "an injected memory nobody judged gets a top-up: {report}"
    );
    assert_eq!(report["llm"]["top_up"], top_ups.len(), "{report}");
    let chunks: BTreeSet<String> = top_ups
        .iter()
        .map(|record| record["chunk"].to_string())
        .collect();
    assert_eq!(
        chunks.len(),
        top_ups.len(),
        "one top-up per chunk: {top_ups:#?}"
    );
    assert!(
        report["llm"]["used_verdicts"]["top_up"]
            .as_u64()
            .is_some_and(|n| n > 0),
        "{report}"
    );

    let cassette = fs::read(dir.private_path("cassettes/main.jsonl")).unwrap();
    let second = replay_history(
        &dir,
        &corpus,
        "fast",
        PASSING_PROBES,
        "fast-second",
        None,
        &[],
    );
    assert_ok(&second.output);
    let report = second.report();
    assert_eq!(report["llm"]["top_up"], 0, "{report}");
    assert_eq!(report["llm"]["misses"], 0, "{report}");
    assert_eq!(
        fs::read(dir.private_path("cassettes/main.jsonl")).unwrap(),
        cassette,
        "a fast run with nothing to judge records nothing"
    );
}

/// The small history with one more turn at the end, imported to
/// `corpus/extra.jsonl` beside the original.
fn imported_with_an_extra_turn(dir: &TestDir) -> std::path::PathBuf {
    let state_db = dir.private_path("state-extra.db");
    let db: StateDb = hermes::small_history(&state_db);
    db.turn(
        "s-later",
        epoch("2026-01-11T09:00:00Z"),
        "One more question.",
        "One more answer.",
    );
    drop(db);
    let corpus = dir.private_path("corpus/extra.jsonl");
    assert_ok(&support::import(dir, &state_db, &corpus));
    corpus
}

/// A chunk with no recorded claims is a miss in `fast`: with an LLM it's
/// called live and counted.
#[test]
fn a_fast_claims_miss_calls_live_with_an_llm_and_counts_it() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let extra = imported_with_an_extra_turn(&dir);
    let script = live_script(&dir);
    let run = replay_history(
        &dir,
        &extra,
        "fast",
        PASSING_PROBES,
        "fast-extra",
        Some(&script),
        &[],
    );
    assert_ok(&run.output);
    let report = run.report();
    assert_eq!(report["llm"]["misses"], 1, "{report}");
    assert_eq!(report["llm"]["live"], 1, "{report}");
}

// The determinism self-test.

/// `fast` on its own recording needs no LLM, counts zero misses and passes
/// the self-test.
#[test]
fn the_self_test_passes_for_fast_with_zero_misses() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let run = replay_history(
        &dir,
        &corpus,
        "fast",
        PASSING_PROBES,
        "self-test",
        None,
        &["--self-test"],
    );
    assert_ok(&run.output);
    assert_eq!(run.report()["llm"]["misses"], 0);
}

// The report.

/// Injected tokens per session and per turn, with cron reported apart so
/// it doesn't skew the percentiles; purges per day next to the
/// purged-then-re-mentioned rate.
#[test]
fn the_report_counts_injected_tokens_with_cron_apart_and_purges() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let run = replay_history(&dir, &corpus, "replay", PASSING_PROBES, "replay", None, &[]);
    assert_ok(&run.output);
    let report = run.report();
    let injected = &report["injected_tokens"];

    let sessions: BTreeSet<&str> = injected["sessions"]
        .as_array()
        .unwrap_or_else(|| panic!("injected tokens per session: {report}"))
        .iter()
        .map(|session| session["session"].as_str().unwrap())
        .collect();
    assert_eq!(
        sessions,
        BTreeSet::from(["s-later", "s-main"]),
        "cron and subagent sessions aren't turn sessions"
    );
    assert!(injected["per_turn"]["p50"].is_number(), "{report}");
    assert!(injected["per_turn"]["p95"].is_number(), "{report}");
    assert_eq!(injected["cron"]["prefetches"], 1, "{report}");

    assert!(report["purges_per_day"].is_array(), "{report}");
    assert!(
        report["purged_then_re_mentioned"]["rate"].is_number(),
        "{report}"
    );
}

// Privacy.

/// The aggregate export is the only thing that leaves the private dir, and
/// its only string values are probe ids. A failed
/// probe is included, since that's where a sentence would most likely
/// leak.
#[test]
fn the_aggregate_export_holds_no_strings_but_probe_ids() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let aggregate = dir.path("aggregate.json");
    let run = replay_history(
        &dir,
        &corpus,
        "replay",
        PROBES_WITH_A_FAILURE,
        "replay",
        None,
        &["--aggregate", aggregate.to_str().unwrap()],
    );
    assert_eq!(
        run.output.status.code(),
        Some(1),
        "p003 fails: {}",
        stderr(&run.output)
    );
    let text = fs::read_to_string(&aggregate).expect("the aggregate export is written");
    let export: Value = serde_json::from_str(&text).expect("the aggregate export is JSON");

    let probe_ids = BTreeSet::from(["p001", "p002", "p003"]);
    let mut strings = Vec::new();
    let mut numbers = 0;
    let mut pending = vec![&export];
    while let Some(value) = pending.pop() {
        match value {
            Value::String(text) => strings.push(text.as_str()),
            Value::Number(_) => numbers += 1,
            Value::Array(items) => pending.extend(items),
            Value::Object(fields) => pending.extend(fields.values()),
            Value::Null | Value::Bool(_) => {}
        }
    }
    for text in &strings {
        assert!(
            probe_ids.contains(text),
            "the aggregate export holds the string {text:?}"
        );
    }
    let seen: BTreeSet<&str> = strings.into_iter().collect();
    assert_eq!(seen, probe_ids, "every probe is in the export");
    assert!(numbers > 0, "the export carries numbers: {text}");

    // Keys are the type's, never content: nothing from the history, the
    // manifest or the probes file appears anywhere in the file.
    for private in [
        "Auckland", "harbour", "curry", "weekend", "Tim", "Sam", "Hermes", "Pacific", "discord",
        "s-main", "s-later", "s-cron", "lives in",
    ] {
        assert!(!text.contains(private), "the export holds {private:?}");
    }
}

/// A real-history report and the cassette never leave the private dir.
#[test]
fn real_history_material_is_refused_outside_the_private_dir() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let script = support::live_script(&dir);
    let probes = dir.private_file("probes.toml", PASSING_PROBES);

    let report = dir.path("report.json");
    let output = asphodel(&dir)
        .arg("replay")
        .arg("--corpus")
        .arg(&corpus)
        .arg("--mode")
        .arg("live")
        .arg("--cassette")
        .arg(dir.private_path("cassettes/main.jsonl"))
        .arg("--probes")
        .arg(&probes)
        .arg("--report")
        .arg(&report)
        .env("ASPHODEL_LLM_SCRIPT", &script)
        .output()
        .unwrap();
    assert_refused(&output, "private");
    assert!(!report.exists());

    let cassette = dir.path("cassette.jsonl");
    let output = asphodel(&dir)
        .arg("replay")
        .arg("--corpus")
        .arg(&corpus)
        .arg("--mode")
        .arg("live")
        .arg("--cassette")
        .arg(&cassette)
        .arg("--probes")
        .arg(&probes)
        .arg("--report")
        .arg(dir.private_path("reports/live.json"))
        .env("ASPHODEL_LLM_SCRIPT", &script)
        .output()
        .unwrap();
    assert_refused(&output, "private");
    assert!(!cassette.exists());
}

// The A/B diff.

/// The diff refuses to compare runs with a different corpus unless forced,
/// and compares runs on the same one.
#[test]
fn the_diff_refuses_runs_on_different_corpora_unless_forced() {
    let a_dir = TestDir::new();
    let a_corpus = imported_small_history(&a_dir);
    record(&a_dir, &a_corpus);
    let a = replay_history(&a_dir, &a_corpus, "replay", PASSING_PROBES, "a", None, &[]);
    assert_ok(&a.output);
    let a_again = replay_history(
        &a_dir,
        &a_corpus,
        "replay",
        PASSING_PROBES,
        "a-again",
        None,
        &[],
    );
    assert_ok(&a_again.output);

    // The same history with one more turn.
    let b_dir = TestDir::new();
    let state_db = b_dir.private_path("state.db");
    let db: StateDb = hermes::small_history(&state_db);
    db.turn(
        "s-later",
        epoch("2026-01-11T09:00:00Z"),
        "One more question.",
        "One more answer.",
    );
    drop(db);
    let b_corpus = b_dir.private_path("corpus/main.jsonl");
    assert_ok(&support::import(&b_dir, &state_db, &b_corpus));
    record(&b_dir, &b_corpus);
    let b = replay_history(&b_dir, &b_corpus, "replay", PASSING_PROBES, "b", None, &[]);
    assert_ok(&b.output);
    assert_ne!(
        sha(&fs::read(&a_corpus).unwrap()),
        sha(&fs::read(&b_corpus).unwrap())
    );

    let diff = |left: &std::path::Path, right: &std::path::Path, force: bool| {
        let mut command = asphodel(&a_dir);
        command.arg("report").arg("diff").arg(left).arg(right);
        if force {
            command.arg("--force");
        }
        command.output().unwrap()
    };
    assert_ok(&diff(&a.report_path, &a_again.report_path, false));
    assert_refused(&diff(&a.report_path, &b.report_path, false), "corpus");
    assert_ok(&diff(&a.report_path, &b.report_path, true));
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The diff lists a memory that faded in one run and not the other by id,
/// and numbers that differ beyond the tolerance by path, leaving out those
/// within it. The two reports are one run's report
/// edited by hand, so the corpus and cassette match and only the edited
/// values differ.
#[test]
fn the_diff_lists_fates_by_id_and_numbers_beyond_the_tolerance_by_path() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let run = replay_history(&dir, &corpus, "replay", PASSING_PROBES, "replay", None, &[]);
    assert_ok(&run.output);
    let report = run.report();
    let id = report["memories"][0]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("the report lists each memory by id: {report}"))
        .to_string();

    let mut a = report.clone();
    let mut b = report;
    a["memories"][0]["faded_at"] = Value::from("2026-03-01T00:00:00Z");
    b["memories"][0]["faded_at"] = Value::Null;
    a["memories"][0]["purged_at"] = Value::Null;
    b["memories"][0]["purged_at"] = Value::Null;
    a["injected_tokens"]["per_turn"]["p50"] = Value::from(100.0);
    b["injected_tokens"]["per_turn"]["p50"] = Value::from(100.000_000_01);
    a["injected_tokens"]["per_turn"]["p95"] = Value::from(100.0);
    b["injected_tokens"]["per_turn"]["p95"] = Value::from(150.0);
    let a_path = dir.private_file("reports/a.json", &a.to_string());
    let b_path = dir.private_file("reports/b.json", &b.to_string());

    let output = asphodel(&dir)
        .arg("report")
        .arg("diff")
        .arg(&a_path)
        .arg(&b_path)
        .output()
        .unwrap();
    assert_ok(&output);
    let diff: Value = serde_json::from_slice(&output.stdout).expect("the diff is JSON");
    assert_eq!(
        diff["memories"]["faded_only_in_a"],
        serde_json::json!([id]),
        "{diff:#}"
    );
    assert_eq!(
        diff["memories"]["faded_only_in_b"],
        serde_json::json!([]),
        "{diff:#}"
    );
    let paths: Vec<&str> = diff["numbers"]
        .as_array()
        .unwrap_or_else(|| panic!("the numbers that differ: {diff:#}"))
        .iter()
        .map(|number| number["path"].as_str().unwrap())
        .collect();
    assert!(paths.contains(&"injected_tokens.per_turn.p95"), "{paths:?}");
    assert!(
        !paths.contains(&"injected_tokens.per_turn.p50"),
        "a difference within the tolerance isn't listed: {paths:?}"
    );
}

// Refreshes. Every bank is seeded with "User profile"
// (ADR 0007); the corpus here adds `home` from the manifest, within the
// budget the profile leaves. The live stand-in answers every call with a
// reply call 1 and a refresh can both read, so the order of calls doesn't
// matter.

/// Both models have an entry citing the home memory a day in.
const MODELS_CITE_HOME: &str = r#"
[[probe]]
id = "p001"
at = "2026-01-06T12:00:00Z"
kind = "profile_has"
model = "home"
memory = "lives in Auckland"

[[probe]]
id = "p002"
at = "2026-01-06T12:00:00Z"
kind = "profile_has"
model = "User profile"
memory = "lives in Auckland"
"#;

/// `home` has no entry citing the home memory, at any of three days.
const HOME_LACKS_HOME: &str = r#"
[[probe]]
id = "p001"
at = "2026-01-06T12:00:00Z"
kind = "profile_lacks"
model = "home"
memory = "lives in Auckland"

[[probe]]
id = "p002"
at = "2026-01-10T12:00:00Z"
kind = "profile_lacks"
model = "home"
memory = "lives in Auckland"
"#;

const HOME_HAS_HOME: &str = r#"
[[probe]]
id = "p001"
at = "2026-01-06T12:00:00Z"
kind = "profile_has"
model = "home"
memory = "lives in Auckland"
"#;

fn is_refresh(record: &Value) -> bool {
    record["template"]["name"] == "refresh_model"
}

/// Whether a refresh record is `home`'s, by the question line its request
/// starts with, which is how `--refresh recorded` tells models apart.
fn is_home_refresh(record: &Value) -> bool {
    is_refresh(record)
        && record["request"]["user"]
            .as_str()
            .is_some_and(|user| user.starts_with(&format!("Question: {HOME_QUESTION}\n")))
}

/// A `live` run on the corpus with a manifest model; returns its report.
fn record_with_models(dir: &TestDir, corpus: &std::path::Path) -> Value {
    let script = universal_script(dir);
    let run = replay_history(
        dir,
        corpus,
        "live",
        MODELS_CITE_HOME,
        "live",
        Some(&script),
        &[],
    );
    assert_ok(&run.output);
    run.report()
}

fn refresh_calls(report: &Value) -> u64 {
    report["refresh_calls_per_day"]
        .as_array()
        .unwrap_or_else(|| panic!("refresh calls per day: {report}"))
        .iter()
        .map(|day| day["count"].as_u64().unwrap())
        .sum()
}

/// The manifest's model reaches the bank and refreshes beside the seeded
/// profile in `live`, and both refreshes are recorded.
#[test]
fn live_refreshes_the_seeded_profile_and_the_manifest_model_and_records_both() {
    let dir = TestDir::new();
    let corpus = imported_with_a_model(&dir);
    let report = record_with_models(&dir, &corpus);
    assert!(refresh_calls(&report) > 0, "{report}");

    let records = cassette_records(&dir);
    assert!(records.iter().any(is_home_refresh), "no refresh of home");
    assert!(
        records
            .iter()
            .any(|record| is_refresh(record) && !is_home_refresh(record)),
        "no refresh of the seeded profile"
    );
}

/// `--refresh off` answers every refresh with no edits, so `home` stays
/// empty, but triggers are counted by code, so refresh calls per day match
/// the recording.
#[test]
fn fast_refresh_off_makes_no_edits_but_counts_every_trigger() {
    let dir = TestDir::new();
    let corpus = imported_with_a_model(&dir);
    let live = record_with_models(&dir, &corpus);
    assert!(refresh_calls(&live) > 0, "{live}");
    let run = replay_history(
        &dir,
        &corpus,
        "fast",
        HOME_LACKS_HOME,
        "fast-off",
        None,
        &["--refresh", "off"],
    );
    assert_ok(&run.output);
    let report = run.report();
    assert_eq!(report["flags"]["refresh"], "off", "{report}");
    assert_eq!(report["llm"]["misses"], 0, "{report}");
    assert_eq!(
        report["refresh_calls_per_day"], live["refresh_calls_per_day"],
        "{report}"
    );
}

/// The cassette with `home`'s recorded refreshes replaced by two: one at
/// the first recorded refresh's time answering `near`, and one 200 days
/// later answering `far`, written first so file order can't pick it.
fn with_near_and_far_home_refreshes(dir: &TestDir, near: Vec<Value>, far: Vec<Value>) {
    let records = cassette_records(dir);
    let first = records
        .iter()
        .find(|record| is_home_refresh(record))
        .expect("the recording refreshed home")
        .clone();
    let at: jiff::Timestamp = first["at"].as_str().unwrap().parse().unwrap();
    let later = at + jiff::SignedDuration::from_hours(24 * 200);
    let copy = |at: jiff::Timestamp, operations: Vec<Value>, tag: &str| {
        let mut record = first.clone();
        record["at"] = Value::from(at.to_string());
        record["key"] = Value::from(format!("{}-{tag}", first["key"].as_str().unwrap()));
        record["response"]["json"] = serde_json::json!({ "operations": operations });
        record
    };
    let mut kept: Vec<Value> = records
        .into_iter()
        .filter(|record| !is_home_refresh(record))
        .collect();
    kept.push(copy(later, far, "far"));
    kept.push(copy(at, near, "near"));
    write_cassette(dir, &kept);
}

/// `--refresh recorded` substitutes the recorded refresh of the same model
/// nearest in simulated time, whichever way round the two are.
#[test]
fn fast_refresh_recorded_substitutes_the_nearest_recorded_refresh() {
    let dir = TestDir::new();
    let corpus = imported_with_a_model(&dir);
    record_with_models(&dir, &corpus);

    with_near_and_far_home_refreshes(
        &dir,
        vec![add_entry("Tim lives in Auckland.", &["m1"])],
        vec![],
    );
    let near_adds = replay_history(
        &dir,
        &corpus,
        "fast",
        HOME_HAS_HOME,
        "near-adds",
        None,
        &["--refresh", "recorded"],
    );
    assert_ok(&near_adds.output);
    assert_eq!(near_adds.report()["flags"]["refresh"], "recorded");

    with_near_and_far_home_refreshes(
        &dir,
        vec![],
        vec![add_entry("Tim lives in Auckland.", &["m1"])],
    );
    let far_adds = replay_history(
        &dir,
        &corpus,
        "fast",
        HOME_LACKS_HOME,
        "far-adds",
        None,
        &["--refresh", "recorded"],
    );
    assert_ok(&far_adds.output);
}

/// `--refresh live` calls the LLM for a refresh the cassette has no record
/// of, records it, and counts it as live.
#[test]
fn fast_refresh_live_calls_the_llm_for_an_unrecorded_refresh() {
    let dir = TestDir::new();
    let corpus = imported_with_a_model(&dir);
    record_with_models(&dir, &corpus);
    let unrefreshed: Vec<Value> = cassette_records(&dir)
        .into_iter()
        .filter(|record| !is_refresh(record))
        .collect();
    write_cassette(&dir, &unrefreshed);

    let script = universal_script(&dir);
    let run = replay_history(
        &dir,
        &corpus,
        "fast",
        HOME_HAS_HOME,
        "fast-live",
        Some(&script),
        &["--refresh", "live"],
    );
    assert_ok(&run.output);
    let report = run.report();
    let calls = refresh_calls(&report);
    assert!(calls > 0, "{report}");
    assert_eq!(report["llm"]["live"], calls, "{report}");
    assert!(cassette_records(&dir).iter().any(is_home_refresh));
}

// Input validation, replay timing and cassette substitution.

/// Exit 0 with every probe passed, or the failed probes in the message.
fn assert_probes_pass(run: &support::Run) {
    let report = run.report();
    let failed: Vec<&Value> = report["probes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|probe| probe["passed"] != true)
        .collect();
    assert!(failed.is_empty(), "failed probes: {failed:#?}");
    assert_ok(&run.output);
}

/// Exit 2, `sentinel` in neither stream, and `words` in stderr.
fn assert_refused_without(output: &std::process::Output, sentinel: &str, words: &[&str]) {
    let out = String::from_utf8_lossy(&output.stdout);
    let err = stderr(output);
    assert_eq!(output.status.code(), Some(2), "stderr: {err}");
    assert!(!out.contains(sentinel), "stdout echoes the input");
    assert!(!err.contains(sentinel), "stderr echoes the input: {err}");
    for word in words {
        assert!(err.contains(word), "stderr should name {word:?}: {err}");
    }
}

/// (ADR 0010, "Logging"): A probes file that doesn't parse is
/// refused naming the file and the line, never quoting it.
#[test]
fn a_malformed_probes_file_is_refused_without_quoting_it() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let sentinel = "SENTINEL-PROBE-QUERY-7a2";
    let probes = format!("{PASSING_PROBES}\n[[probe]]\nid = \"p003\"\nquery = {sentinel}\n");
    let line = probes
        .lines()
        .position(|line| line.contains(sentinel))
        .unwrap()
        + 1;
    let run = replay_history(&dir, &corpus, "replay", &probes, "bad-probes", None, &[]);
    assert_refused_without(
        &run.output,
        sentinel,
        &["probes.toml", &format!("line {line}")],
    );
}

/// A probe whose regex doesn't compile is refused naming the
/// probe, never the pattern.
#[test]
fn a_probe_regex_that_doesnt_compile_is_refused_without_quoting_it() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let sentinel = "SENTINEL-PROBE-PATTERN-((";
    let probes = format!(
        "[[probe]]\nid = \"p001\"\nat = \"2026-01-05T12:00:00Z\"\nkind = \"exists\"\nmemory = \"{sentinel}\"\n"
    );
    let run = replay_history(&dir, &corpus, "replay", &probes, "bad-regex", None, &[]);
    assert_refused_without(&run.output, "SENTINEL-PROBE-PATTERN", &["p001"]);
}

/// A corpus line that doesn't parse is refused naming the line,
/// never quoting it.
#[test]
fn a_malformed_corpus_line_is_refused_without_quoting_it() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let sentinel = "SENTINEL-CORPUS-VALUE-c40";
    let mut lines: Vec<Value> = fs::read_to_string(&corpus)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let line = lines
        .iter()
        .position(|line| line["event"] == "prefetch")
        .unwrap();
    lines[line]["class"] = Value::from(sentinel);
    let text: String = lines.iter().map(|line| format!("{line}\n")).collect();
    fs::write(&corpus, text).unwrap();
    let run = replay_history(
        &dir,
        &corpus,
        "replay",
        PASSING_PROBES,
        "bad-corpus",
        None,
        &[],
    );
    assert_refused_without(&run.output, sentinel, &[&format!("line {}", line + 1)]);
}

/// With nonzero live latency, a prefetch between a turn's sync
/// and its completion doesn't see the turn's memory, and one after does.
/// Both prefetches ask the same question from sessions of their own; the
/// probes check the store at the same two moments.
#[test]
fn a_prefetch_before_a_live_completion_doesnt_see_the_turn() {
    let dir = TestDir::new();
    let state_db = dir.private_path("state.db");
    let db = StateDb::create(&state_db);
    let t = epoch("2026-01-05T09:00:00Z");
    for (session, at) in [("s1", t), ("s2", t + 31.0), ("s3", t + 40.0)] {
        db.session(session, "discord", Some("discord:1"), None, at);
    }
    db.turn(
        "s1",
        t,
        &format!("{}, near the harbour.", hermes::HOME_QUOTE),
        "Noted.",
    );
    // Synced at t+30 with a 3 s call 1: complete at t+33.
    for (session, at) in [("s2", t + 31.0), ("s3", t + 40.0)] {
        db.message(support::hermes::Message {
            session,
            role: "user",
            content: "Tim lives in Auckland near the harbour?",
            at,
            ..Default::default()
        });
        db.message(support::hermes::Message {
            session,
            role: "assistant",
            content: "Let me think.",
            at: at + 1.0,
            ..Default::default()
        });
    }
    drop(db);
    let corpus = dir.private_path("corpus/main.jsonl");
    assert_ok(&support::import(&dir, &state_db, &corpus));

    let probes = r#"
[[probe]]
id = "p001"
at = "2026-01-05T09:00:31Z"
kind = "absent"
memory = "lives in Auckland"

[[probe]]
id = "p002"
at = "2026-01-05T09:00:40Z"
kind = "exists"
memory = "lives in Auckland"
"#;
    let script = support::delayed_script(&dir, 3000);
    let run = replay_history(&dir, &corpus, "live", probes, "live", Some(&script), &[]);
    let report = run.report();
    let tokens = |session: &str| -> u64 {
        report["injected_tokens"]["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["session"] == session)
            .unwrap_or_else(|| panic!("no {session} in {report}"))["tokens"]
            .as_u64()
            .unwrap()
    };
    assert!(
        tokens("s3") > 0,
        "the later prefetch sees the memory: {report}"
    );
    assert_eq!(tokens("s2"), 0, "the earlier prefetch can't: {report}");
    assert_ok(&run.output);
}

/// `--no-cache` re-records from scratch, so re-recording leaves
/// one record per call, and the replay's lag is one call's latency, not
/// the sum over every recording.
#[test]
fn re_recording_with_no_cache_starts_a_fresh_cassette() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let script = support::delayed_script(&dir, 300);
    let first = replay_history(
        &dir,
        &corpus,
        "live",
        PASSING_PROBES,
        "live-1",
        Some(&script),
        &[],
    );
    assert_ok(&first.output);
    let recorded = cassette_records(&dir).len();
    let mut last = None;
    for n in 2..=3 {
        let run = replay_history(
            &dir,
            &corpus,
            "live",
            PASSING_PROBES,
            &format!("live-{n}"),
            Some(&script),
            &["--no-cache"],
        );
        assert_ok(&run.output);
        last = Some(run.report());
    }
    assert_eq!(
        cassette_records(&dir).len(),
        recorded,
        "a re-recording replaces the cassette"
    );
    let replay = replay_history(&dir, &corpus, "replay", PASSING_PROBES, "replay", None, &[]);
    assert_ok(&replay.output);
    assert_eq!(
        replay.report()["extraction_lag"],
        last.unwrap()["extraction_lag"]
    );
}

/// A chunk's latency comes from the records that answered it,
/// not from every record tagged with the chunk. Copies of each record
/// under another model, ten times slower, sit first in the cassette; the
/// replay answers from the original model's and keeps its lag.
#[test]
fn replay_lag_counts_only_the_records_that_answered() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let script = support::delayed_script(&dir, 300);
    let live = replay_history(
        &dir,
        &corpus,
        "live",
        PASSING_PROBES,
        "live",
        Some(&script),
        &[],
    );
    assert_ok(&live.output);
    let records = cassette_records(&dir);
    let mut mixed: Vec<Value> = records
        .iter()
        .map(|record| {
            let mut copy = record.clone();
            copy["model"] = Value::from("other-model");
            copy["key"] = Value::from(format!("{}-other", record["key"].as_str().unwrap()));
            copy["latency_ms"] = Value::from(record["latency_ms"].as_u64().unwrap() * 10);
            copy
        })
        .collect();
    mixed.extend(records);
    write_cassette(&dir, &mixed);

    let replay = replay_history(&dir, &corpus, "replay", PASSING_PROBES, "replay", None, &[]);
    assert_ok(&replay.output);
    assert_eq!(
        replay.report()["extraction_lag"],
        live.report()["extraction_lag"]
    );
}

/// The manifest for the identity tests: `home` takes `kinds`.
fn manifest_with_home_kinds(kinds: &str) -> String {
    format!(
        "{}\n[[model]]\nname = \"home\"\nquestion = \"{HOME_QUESTION}\"\nkinds = {kinds}\nmax_tokens = 100\n",
        hermes::MANIFEST
    )
}

const PASSPORT_QUOTE: &str = "I need to renew my passport";

/// A history where a passport task (A) is remembered on day 0 and the home
/// fact (B) on day 3, imported twice: `home` filtered to facts and tasks
/// (`corpus/both.jsonl`), and to facts only (`corpus/facts.jsonl`), so A
/// is out of its input. The chunks are the same in both.
fn identity_corpora(dir: &TestDir) -> (std::path::PathBuf, std::path::PathBuf) {
    let state_db = dir.private_path("state.db");
    let db = StateDb::create(&state_db);
    let t = epoch("2026-01-05T09:00:00Z");
    db.session("s1", "discord", Some("discord:1"), None, t);
    db.turn("s1", t, &format!("{PASSPORT_QUOTE} soon."), "Noted.");
    db.turn(
        "s1",
        t + 3.0 * 86_400.0,
        &format!("{}, near the harbour.", hermes::HOME_QUOTE),
        "Noted.",
    );
    db.turn("s1", t + 5.0 * 86_400.0, "Anything else?", "No.");
    drop(db);
    let both = dir.private_path("corpus/both.jsonl");
    let facts = dir.private_path("corpus/facts.jsonl");
    for (corpus, kinds) in [(&both, r#"["fact", "task"]"#), (&facts, r#"["fact"]"#)] {
        assert_ok(&support::import_with(
            dir,
            &state_db,
            corpus,
            &manifest_with_home_kinds(kinds),
            &[],
        ));
    }
    (both, facts)
}

/// The memories a refresh request lists, in handle order.
fn listed_memories(record: &Value) -> Vec<String> {
    let user = record["request"]["user"].as_str().unwrap();
    let memories = user.split("\nMemories:\n").nth(1).unwrap_or("");
    memories
        .lines()
        .filter(|line| line.starts_with('m'))
        .map(str::to_string)
        .collect()
}

/// Records the identity history with every refresh adding an entry that
/// cites `m1`, then keeps only `home`'s refresh recorded when the passport
/// task was its one memory, so `m1` meant A. Checks the fixture: `home`'s
/// later recorded refresh lists the home fact as `m1`, so a substitution
/// by handle name would cite B.
fn record_with_a_meaning_m1(dir: &TestDir, corpus: &std::path::Path) {
    let script = support::script_answering_everything(
        dir,
        "identity-script",
        vec![
            support::claim("Tim needs to renew his passport.", PASSPORT_QUOTE, "task"),
            support::home_claim(),
        ],
        vec![add_entry("An entry citing m1.", &["m1"])],
    );
    let live = replay_history(dir, corpus, "live", "", "live", Some(&script), &[]);
    assert_ok(&live.output);
    let records = cassette_records(dir);
    let homes: Vec<&Value> = records.iter().filter(|r| is_home_refresh(r)).collect();
    let only_a = homes
        .iter()
        .find(|record| {
            let listed = listed_memories(record);
            listed.len() == 1 && listed[0].contains("passport")
        })
        .expect("home refreshed when the passport task was its one memory");
    assert!(
        homes.iter().any(|record| {
            let listed = listed_memories(record);
            listed.len() == 2 && listed[0].starts_with("m1: ") && listed[0].contains("Auckland")
        }),
        "fixture: with both memories, home lists the home fact as m1: {:#?}",
        homes.iter().map(|r| listed_memories(r)).collect::<Vec<_>>()
    );
    let kept: Vec<Value> = records
        .iter()
        .filter(|record| !is_home_refresh(record) || record == only_a)
        .cloned()
        .collect();
    write_cassette(dir, &kept);
}

/// A substituted refresh cites the memory
/// it cited when recorded, not whichever memory holds its handle now.
#[test]
fn a_substituted_refresh_cites_the_memory_it_was_recorded_with() {
    let dir = TestDir::new();
    let (both, _) = identity_corpora(&dir);
    record_with_a_meaning_m1(&dir, &both);
    let probes = r#"
[[probe]]
id = "p001"
at = "2026-01-09T12:00:00Z"
kind = "profile_has"
model = "home"
memory = "renew his passport"

[[probe]]
id = "p002"
at = "2026-01-09T12:00:00Z"
kind = "profile_lacks"
model = "home"
memory = "lives in Auckland"
"#;
    let run = replay_history(
        &dir,
        &both,
        "fast",
        probes,
        "fast",
        None,
        &["--refresh", "recorded"],
    );
    assert_probes_pass(&run);
}

/// When the memory a substituted refresh cited isn't in the
/// current input, the operation is dropped, even though its handle names
/// another memory now.
#[test]
fn a_substituted_refresh_citing_an_absent_memory_is_dropped() {
    let dir = TestDir::new();
    let (both, facts) = identity_corpora(&dir);
    record_with_a_meaning_m1(&dir, &both);
    let probes = r#"
[[probe]]
id = "p001"
at = "2026-01-09T12:00:00Z"
kind = "profile_lacks"
model = "home"
memory = "lives in Auckland"
"#;
    let run = replay_history(
        &dir,
        &facts,
        "fast",
        probes,
        "fast",
        None,
        &["--refresh", "recorded"],
    );
    assert_probes_pass(&run);
}

/// Nearest-refresh substitution only considers records of the
/// run's own LLM model. A record under another model at the very time of
/// the refresh loses to the run's own model's record 200 days off.
#[test]
fn a_refresh_recorded_under_another_model_is_never_substituted() {
    let dir = TestDir::new();
    let corpus = imported_with_a_model(&dir);
    record_with_models(&dir, &corpus);
    let records = cassette_records(&dir);
    let first = records
        .iter()
        .find(|record| is_home_refresh(record))
        .expect("the recording refreshed home")
        .clone();
    let at: jiff::Timestamp = first["at"].as_str().unwrap().parse().unwrap();
    let mut other = first.clone();
    other["model"] = Value::from("other-model");
    other["key"] = Value::from(format!("{}-other", first["key"].as_str().unwrap()));
    other["response"]["json"] =
        serde_json::json!({ "operations": [add_entry("Tim lives in Auckland.", &["m1"])] });
    let mut far = first.clone();
    far["at"] = Value::from((at + jiff::SignedDuration::from_hours(24 * 200)).to_string());
    far["key"] = Value::from(format!("{}-far", first["key"].as_str().unwrap()));
    far["response"]["json"] = serde_json::json!({ "operations": [] });
    // The other model's record goes first, so the run's model, which
    // `replay` and `fast` take from the last record, stays the original.
    let mut kept = vec![other];
    kept.extend(
        records
            .into_iter()
            .filter(|record| !is_home_refresh(record)),
    );
    kept.push(far);
    write_cassette(&dir, &kept);

    let run = replay_history(
        &dir,
        &corpus,
        "fast",
        HOME_LACKS_HOME,
        "fast",
        None,
        &["--refresh", "recorded"],
    );
    assert_probes_pass(&run);
}

// A substituted refresh's edit or remove
// whose recorded entry is absent from this run is dropped, even though its
// handle names another entry now.

const LIBRARY_QUOTE: &str = "I work at the library";

/// The identity history with a library fact on day 7, so `home` refreshes
/// again after the home fact, imported as `both` (facts and tasks) and
/// `facts` (facts only, so the passport task and any entry citing it never
/// exist).
fn entry_corpora(dir: &TestDir) -> (std::path::PathBuf, std::path::PathBuf) {
    let state_db = dir.private_path("state.db");
    let db = StateDb::create(&state_db);
    let t = epoch("2026-01-05T09:00:00Z");
    db.session("s1", "discord", Some("discord:1"), None, t);
    db.turn("s1", t, &format!("{PASSPORT_QUOTE} soon."), "Noted.");
    db.turn(
        "s1",
        t + 3.0 * 86_400.0,
        &format!("{}, near the harbour.", hermes::HOME_QUOTE),
        "Noted.",
    );
    db.turn(
        "s1",
        t + 7.0 * 86_400.0,
        &format!("{LIBRARY_QUOTE} now."),
        "Noted.",
    );
    drop(db);
    let both = dir.private_path("corpus/both.jsonl");
    let facts = dir.private_path("corpus/facts.jsonl");
    for (corpus, kinds) in [(&both, r#"["fact", "task"]"#), (&facts, r#"["fact"]"#)] {
        assert_ok(&support::import_with(
            dir,
            &state_db,
            corpus,
            &manifest_with_home_kinds(kinds),
            &[],
        ));
    }
    (both, facts)
}

/// The current entries a refresh request lists, in handle order.
fn listed_entries(record: &Value) -> Vec<String> {
    let user = record["request"]["user"].as_str().unwrap();
    let entries = user
        .split("\nCurrent entries:\n")
        .nth(1)
        .and_then(|rest| rest.split("\n\nMemories:\n").next())
        .unwrap_or("");
    entries
        .lines()
        .filter(|line| line.starts_with('e'))
        .map(str::to_string)
        .collect()
}

/// Records the entry history on `both`, where `home`'s refresh after the
/// home fact sees entry A (citing the passport) as `e1`. Then rewrites the
/// cassette so `home` has two refreshes, both copies of that one and so
/// both recorded with `e1` meaning A and `m1` the home fact: R_add at its
/// own time, adding an entry citing `m1`, and R_target at the time of the
/// refresh after the library fact, answering `target`.
///
/// On `facts`, R_add makes entry B citing the home fact, `home`'s only
/// entry and so `e1`, and R_target then reaches a run where its `e1`
/// meant A, which never exists, while `e1` names B.
fn record_with_e1_meaning_a(dir: &TestDir, both: &std::path::Path, target: Vec<Value>) {
    let script = support::script_answering_everything(
        dir,
        "entry-script",
        vec![
            support::claim("Tim needs to renew his passport.", PASSPORT_QUOTE, "task"),
            support::home_claim(),
            support::claim("Tim works at the library.", LIBRARY_QUOTE, "fact"),
        ],
        vec![add_entry("An entry citing m1.", &["m1"])],
    );
    let live = replay_history(
        dir,
        both,
        "live",
        "",
        "live",
        Some(&script),
        &["--until", "2026-01-14T09:00:00Z"],
    );
    assert_ok(&live.output);
    let records = cassette_records(dir);
    let homes: Vec<&Value> = records.iter().filter(|r| is_home_refresh(r)).collect();
    let after_home = homes
        .iter()
        .find(|record| {
            let listed = listed_memories(record);
            listed.len() == 2 && listed[0].starts_with("m1: ") && listed[0].contains("Auckland")
        })
        .expect("home refreshed with the home fact as m1 and the passport");
    let entries = listed_entries(after_home);
    assert!(
        entries.len() == 1 && entries[0].starts_with("e1: ") && entries[0].ends_with("[cites m2]"),
        "fixture: e1 is A, citing the passport: {entries:?}"
    );
    assert!(
        after_home["identities"]
            .as_array()
            .is_some_and(|ids| ids.iter().any(|pair| pair[0] == "e1")),
        "fixture: the record keeps what e1 meant"
    );
    let after_library = homes
        .iter()
        .find(|record| {
            listed_memories(record)
                .iter()
                .any(|line| line.contains("library"))
        })
        .expect("home refreshed after the library fact");

    let copy = |at: &Value, operations: Vec<Value>, tag: &str| {
        let mut record = (*after_home).clone();
        record["at"] = at.clone();
        record["key"] = Value::from(format!("{}-{tag}", after_home["key"].as_str().unwrap()));
        record["response"]["json"] = serde_json::json!({ "operations": operations });
        record
    };
    let r_add = copy(
        &after_home["at"],
        vec![add_entry("Tim lives in Auckland.", &["m1"])],
        "add",
    );
    let r_target = copy(&after_library["at"], target, "target");
    let mut kept: Vec<Value> = records
        .iter()
        .filter(|record| !is_home_refresh(record))
        .cloned()
        .collect();
    kept.push(r_target);
    kept.push(r_add);
    write_cassette(dir, &kept);
}

/// B, citing the home fact, is in `home` the day after the library fact.
const B_REMAINS: &str = r#"
[[probe]]
id = "p001"
at = "2026-01-13T12:00:00Z"
kind = "profile_has"
model = "home"
memory = "lives in Auckland"
"#;

/// A `fast --refresh recorded` run on `facts` with R_target answering
/// `target`.
fn substituted(target: Vec<Value>, name: &str) -> (TestDir, support::Run) {
    let dir = TestDir::new();
    let (both, facts) = entry_corpora(&dir);
    record_with_e1_meaning_a(&dir, &both, target);
    let run = replay_history(
        &dir,
        &facts,
        "fast",
        B_REMAINS,
        name,
        None,
        &["--refresh", "recorded", "--until", "2026-01-14T09:00:00Z"],
    );
    (dir, run)
}

#[test]
fn a_substituted_edit_whose_entry_is_absent_is_dropped() {
    let (_control_dir, control) = substituted(vec![], "control");
    assert_probes_pass(&control);
    let control = control.report();
    assert!(
        control["profile_tokens"]["p95"]
            .as_u64()
            .is_some_and(|tokens| tokens > 0),
        "fixture: B renders: {control}"
    );

    let (_dir, run) = substituted(
        vec![serde_json::json!({
            "op": "edit",
            "entry": "e1",
            "text": "Tim lives in Auckland, in a tall blue house above the harbour with a view over the whole city and the gulf.",
            "cites": ["m1"],
        })],
        "edit",
    );
    assert_probes_pass(&run);
    assert_eq!(
        run.report()["profile_tokens"],
        control["profile_tokens"],
        "B's text is unchanged"
    );
}
