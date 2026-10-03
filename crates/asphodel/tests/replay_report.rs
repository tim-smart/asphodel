//! The HTML report, labelling material and precision curve contracts,
//! exercised on synthetic history only.
//!
//! - `asphodel report html R [--out FILE]` writes one page holding the
//!   report's numbers, every asset inlined. Without `--out`
//!   it goes beside the report, as `R` with the extension `html`.
//! - `asphodel replay... --labelling FILE` writes the labelling material
//!   with recall candidates at 50 sampled turns, scored with the
//!   reranker logit the gate floor compares, and call 2's candidate lists,
//!   scored with the cosine similarity the reconcile floor compares.
//! - `asphodel report precision --labels L --material M` prints the
//!   precision curve for each list as JSON.
//!
//! Everything these read or write is derived from history, so all of it
//! stays under the private dir and errors never quote it.

mod support;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Output;

use serde_json::{Value, json};
use support::hermes::{self, StateDb, epoch};
use support::{
    PASSING_PROBES, TestDir, asphodel, assert_ok, assert_refused, claim, imported_small_history,
    record, replay_history, stderr, stdout,
};

/// Exit 2, `sentinel` in neither stream, and `words` in stderr.
fn assert_refused_without(output: &Output, sentinel: &str, words: &[&str]) {
    let out = stdout(output);
    let err = stderr(output);
    assert_eq!(output.status.code(), Some(2), "stderr: {err}");
    assert!(!out.contains(sentinel), "stdout echoes the input: {out}");
    assert!(!err.contains(sentinel), "stderr echoes the input: {err}");
    for word in words {
        assert!(err.contains(word), "stderr should name {word:?}: {err}");
    }
}

/// What a run simulated, without what identifies the run.
fn simulation(report: &Value) -> Value {
    let mut report = report.clone();
    let object = report.as_object_mut().expect("the report is an object");
    for key in ["kind", "flags", "llm", "cassette_hash"] {
        object.remove(key);
    }
    report
}

// The HTML report.

fn report_html(dir: &TestDir, report: &Path, extra: &[&str]) -> Output {
    asphodel(dir)
        .args(["report", "html"])
        .arg(report)
        .args(extra)
        .output()
        .unwrap()
}

/// A recorded report with a distinct number in every section the page
/// shows, so finding the number on the page means the section is there.
fn report_with_distinct_numbers(dir: &TestDir) -> (PathBuf, Value) {
    let corpus = imported_small_history(dir);
    let mut report = record(dir, &corpus).report();
    let set = |report: &mut Value, pointer: &str, value: Value| {
        *report
            .pointer_mut(pointer)
            .unwrap_or_else(|| panic!("the report has {pointer}")) = value;
    };
    set(&mut report, "/injected_tokens/per_turn/p50", json!(71001));
    set(&mut report, "/injected_tokens/per_turn/p95", json!(71002));
    set(&mut report, "/injected_tokens/cron/tokens", json!(71003));
    set(
        &mut report,
        "/injected_tokens/sessions/0/tokens",
        json!(71004),
    );
    set(&mut report, "/profile_tokens/p95", json!(71005));
    set(&mut report, "/extraction_lag/p95_ms", json!(71006));
    set(&mut report, "/call2_rate/call2", json!(71007));
    set(&mut report, "/llm/cache", json!(71008));
    set(&mut report, "/llm/live", json!(71009));
    set(&mut report, "/llm/misses", json!(71010));
    set(&mut report, "/llm/used_verdicts/top_up", json!(71011));
    set(
        &mut report,
        "/purged_then_re_mentioned/purged",
        json!(71012),
    );
    set(&mut report, "/kind_histogram", json!({ "fact": 71013 }));
    set(
        &mut report,
        "/significance_histogram",
        json!({ "notable": 71014 }),
    );
    set(
        &mut report,
        "/bands_per_week",
        json!([{ "week": "2026-W02", "strong": 71015, "fading": 71016, "faded": 71017 }]),
    );
    set(
        &mut report,
        "/fade_outs_per_week",
        json!([{ "week": "2026-W03", "count": 71018 }]),
    );
    set(
        &mut report,
        "/purges_per_day",
        json!([{ "day": "2026-01-09", "count": 71019, "purged": 71019 }]),
    );
    set(
        &mut report,
        "/refresh_calls_per_day",
        json!([{ "day": "2026-01-10", "count": 71020 }]),
    );
    set(
        &mut report,
        "/agenda_lines_per_day",
        json!([{ "day": "2026-01-11", "count": 71021 }]),
    );
    let path = dir.private_file(
        "reports/patched.json",
        &serde_json::to_string_pretty(&report).unwrap(),
    );
    (path, report)
}

/// `number` as written, plain or with thousands separators.
fn shows_number(page: &str, number: u64) -> bool {
    let plain = number.to_string();
    let grouped = format!("{},{:03}", number / 1000, number % 1000);
    page.contains(&plain) || page.contains(&grouped)
}

/// Every reference on the page that would load something from elsewhere:
/// a `src`, `href` or similar attribute, or a CSS `url(`, whose value is
/// neither a fragment nor a `data:` URL, plus `@import` and the script
/// APIs that open connections.
fn external_references(page: &str) -> Vec<String> {
    let lower = page.to_lowercase();
    let mut found = Vec::new();
    for attribute in [
        "src=",
        "href=",
        "srcset=",
        "action=",
        "formaction=",
        "poster=",
        " data=",
        "url(",
    ] {
        let mut rest = lower.as_str();
        while let Some(at) = rest.find(attribute) {
            rest = &rest[at + attribute.len()..];
            let value = rest
                .trim_start()
                .trim_start_matches(['"', '\''])
                .trim_start();
            let end = value
                .find(|c: char| c == '"' || c == '\'' || c == ')' || c == '>' || c.is_whitespace())
                .unwrap_or(value.len());
            let value = &value[..end];
            if !(value.is_empty() || value.starts_with('#') || value.starts_with("data:")) {
                found.push(format!("{attribute}{value}"));
            }
        }
    }
    for api in [
        "@import",
        "fetch(",
        "xmlhttprequest",
        "websocket",
        "eventsource",
        "sendbeacon",
    ] {
        if lower.contains(api) {
            found.push(api.into());
        }
    }
    found
}

/// The page shows the report's numbers, its identity and its probes, and
/// loads nothing from anywhere: a page full of Tim's history never
/// fetches from a CDN.
#[test]
fn the_html_report_shows_the_reports_numbers_with_every_asset_inlined() {
    let dir = TestDir::new();
    let (path, report) = report_with_distinct_numbers(&dir);
    let output = report_html(&dir, &path, &[]);
    assert_ok(&output);

    let page_path = path.with_extension("html");
    let page = fs::read_to_string(&page_path)
        .unwrap_or_else(|_| panic!("no page at {}: {}", page_path.display(), stderr(&output)));
    assert!(
        page.trim_start()
            .to_lowercase()
            .starts_with("<!doctype html>"),
        "the page is an HTML document"
    );
    for number in 71001..=71021 {
        assert!(shows_number(&page, number), "the page lacks {number}");
    }
    for key in ["corpus_hash", "cassette_hash", "git_sha"] {
        let value = report[key].as_str().unwrap();
        assert!(page.contains(value), "the page lacks the {key} {value}");
    }
    for probe in ["p001", "p002"] {
        assert!(page.contains(probe), "the page lacks probe {probe}");
    }
    assert_eq!(external_references(&page), Vec::<String>::new());
}

/// The report read and the page written are both history, so both stay
/// in the private dir: a path outside it, or a symlink inside it pointing
/// out, is refused and nothing is written.
#[test]
fn the_html_report_is_refused_outside_the_private_dir() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let recorded = record(&dir, &corpus).report_path;

    let outside = dir.path("outside.json");
    fs::copy(&recorded, &outside).unwrap();
    assert_refused(&report_html(&dir, &outside, &[]), "private");
    assert!(!outside.with_extension("html").exists());

    let linked = dir.private().join("reports/linked.json");
    std::os::unix::fs::symlink(&outside, &linked).unwrap();
    assert_refused(&report_html(&dir, &linked, &[]), "private");
    assert!(!dir.path("outside.html").exists());
    assert!(!linked.with_extension("html").exists());

    let page = dir.path("page.html");
    let output = report_html(&dir, &recorded, &["--out", page.to_str().unwrap()]);
    assert_refused(&output, "private");
    assert!(!page.exists());

    let target = dir.path("target.html");
    let linked_page = dir.private().join("reports/page.html");
    std::os::unix::fs::symlink(&target, &linked_page).unwrap();
    let output = report_html(&dir, &recorded, &["--out", linked_page.to_str().unwrap()]);
    assert_refused(&output, "private");
    assert!(!target.exists());
}

// The labelling material.

/// More synced turns than the 50 the material samples, one session each so
/// no turn's candidates are hidden as already in context. The first two
/// say where Tim lives, so the second's claim lands on the first's memory
/// and call 2 runs. The rest ask a question that either shares a word with
/// that memory ("Auckland") or none, so the material holds candidates on
/// both sides of the fake reranker's gate floor (0.0). The questions come
/// as Hermes hands them over in a shared Discord thread: with the speaker's
/// `[Name] ` prefix, and every other one after the message-id note.
const TURNS: usize = 60;

/// The note Hermes' Discord gateway puts in front of a turn's message, with
/// a synthetic message id.
const DISCORD_NOTE: &str = "[Triggering message id: `100000000000000001` \u{2014} use as \
                            `message_id` for reply/react/pin via the discord tools.]";

fn labelling_history(dir: &TestDir) -> PathBuf {
    let state_db = dir.private_path("state.db");
    let db = StateDb::create(&state_db);
    let start = epoch("2026-01-05T09:00:00Z");
    for turn in 0..TURNS {
        let session = format!("s{turn:02}");
        let at = start + 3600.0 * turn as f64;
        db.session(&session, "discord", Some("discord:1"), None, at);
        let user = match turn {
            0 => format!("{}, near the harbour.", hermes::HOME_QUOTE),
            1 => format!("{} still.", hermes::HOME_QUOTE),
            n if n % 2 == 0 => format!("[Sam] Is Auckland sunny today, question {n}?"),
            n => format!("{DISCORD_NOTE}\n\n[Sam] What is on the radio, question {n}?"),
        };
        db.turn(&session, at, &user, "Noted.");
    }
    drop(db);
    let corpus = dir.private_path("corpus/main.jsonl");
    assert_ok(&support::import(dir, &state_db, &corpus));
    corpus
}

/// A stand-in whose every step answers any call: call 1 claims the home
/// sentence (kept only where the turn quotes it), call 2 labels it
/// `mentioned_again` on the first neighbour, and a refresh makes no edits.
fn labelling_script(dir: &TestDir) -> PathBuf {
    let mut home = claim(hermes::HOME_SENTENCE, hermes::HOME_QUOTE, "fact");
    home["claim"] = json!("c1");
    home["labels"] = json!([{ "neighbour": "n1", "label": "mentioned_again" }]);
    let reply = json!({
        "claims": [home],
        "used_injected_ids": [],
        "operations": []
    });
    let steps: Vec<Value> = (0..256).map(|_| json!({ "reply": reply })).collect();
    let path = dir.path("labelling-script.json");
    fs::write(&path, serde_json::to_vec(&steps).unwrap()).unwrap();
    path
}

/// The fake reranker's logit: one per word the query and the document
/// share, less a half.
fn fake_logit(query: &str, document: &str) -> f64 {
    let words = |text: &str| -> BTreeSet<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .map(str::to_lowercase)
            .collect()
    };
    words(query).intersection(&words(document)).count() as f64 - 0.5
}

fn candidates(sample: &Value) -> &Vec<Value> {
    sample["candidates"]
        .as_array()
        .expect("every sample lists its candidates")
}

/// Every candidate carries an id unique in the material, the memory's id,
/// a score and the sentence.
fn assert_candidate_shape(candidate: &Value, ids: &mut BTreeSet<String>) {
    let id = candidate["id"].as_str().expect("a candidate has an id");
    assert!(ids.insert(id.into()), "candidate id {id} repeats");
    let memory = candidate["memory"]
        .as_str()
        .expect("a candidate names its memory");
    assert!(
        uuid::Uuid::parse_str(memory).is_ok(),
        "the memory id {memory} is a UUID"
    );
    assert!(candidate["score"].is_number(), "{candidate}");
    assert!(candidate["sentence"].is_string(), "{candidate}");
}

/// Replay's `--labelling` writes recall candidates at 50 sampled turns and
/// call 2's candidate lists, with ids, scores and sentences. Recall
/// candidates are scored with the reranker logit the gate floor compares,
/// including those the gate turned away, since a floor can't be calibrated
/// from what it already let through; call 2's with the similarity the
/// reconcile floor compares. Each sample shows the raw query and the
/// cleaned one the reranker scored. Writing it doesn't change the run, the
/// same run writes the same bytes, and `report precision` reads it.
#[test]
fn labelling_material_holds_scored_candidates_at_50_turns_and_call2_lists() {
    let dir = TestDir::new();
    let corpus = labelling_history(&dir);
    let script = labelling_script(&dir);
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
    let plain = replay_history(&dir, &corpus, "replay", PASSING_PROBES, "plain", None, &[]);
    assert_ok(&plain.output);

    let material_path = dir.private_path("labelling/material.json");
    let labelled = replay_history(
        &dir,
        &corpus,
        "replay",
        PASSING_PROBES,
        "labelled",
        None,
        &["--labelling", material_path.to_str().unwrap()],
    );
    assert_ok(&labelled.output);
    assert_eq!(
        simulation(&plain.report()),
        simulation(&labelled.report()),
        "writing the material changes nothing the run simulated"
    );
    let bytes = fs::read(&material_path).expect("the material is written");
    let material: Value = serde_json::from_slice(&bytes).expect("the material is JSON");

    let again_path = dir.private_path("labelling/again.json");
    let again = replay_history(
        &dir,
        &corpus,
        "replay",
        PASSING_PROBES,
        "again",
        None,
        &["--labelling", again_path.to_str().unwrap()],
    );
    assert_ok(&again.output);
    assert_eq!(
        fs::read(&again_path).unwrap(),
        bytes,
        "the same run samples the same turns"
    );

    let first_memory = labelled.report()["memories"][0]["id"]
        .as_str()
        .expect("the history makes a memory")
        .to_string();
    let mut ids = BTreeSet::new();

    let recall = material["recall"].as_array().expect("a recall list");
    assert_eq!(recall.len(), 50, "50 of the {TURNS} turns are sampled");
    let turns: BTreeSet<(String, String)> = recall
        .iter()
        .map(|sample| {
            assert!(sample["sample"].is_string(), "{sample}");
            (
                sample["session"]
                    .as_str()
                    .expect("a sample's session")
                    .into(),
                sample["at"].as_str().expect("a sample's time").into(),
            )
        })
        .collect();
    assert_eq!(turns.len(), 50, "each sample is a different turn");
    let mut below_floor = 0;
    let mut on_home = 0;
    let mut noted = 0;
    for sample in recall {
        let query = sample["query"].as_str().expect("a sample's query");
        let raw = sample["raw_query"].as_str().expect("a sample's raw query");
        if raw.starts_with("[Sam] ") || raw.starts_with(DISCORD_NOTE) {
            let message = raw.rsplit_once("[Sam] ").unwrap().1;
            assert_eq!(query, message, "the cleaned query for {raw:?}");
            noted += usize::from(raw.starts_with(DISCORD_NOTE));
        } else {
            assert_eq!(query, raw, "a turn with no prefix is its own query");
        }
        for candidate in candidates(sample) {
            assert_candidate_shape(candidate, &mut ids);
            let sentence = candidate["sentence"].as_str().unwrap();
            let score = candidate["score"].as_f64().unwrap();
            assert_eq!(
                score,
                fake_logit(query, sentence),
                "a recall candidate's score is the reranker logit: {candidate} for {query:?}"
            );
            if score < 0.0 {
                below_floor += 1;
            }
            if candidate["memory"] == first_memory.as_str() {
                assert_eq!(sentence, hermes::HOME_SENTENCE);
                on_home += 1;
            }
        }
    }
    assert!(noted > 0, "a sampled turn carries the message-id note");
    assert!(on_home > 0, "the home memory is a recall candidate");
    assert!(
        below_floor > 0,
        "candidates the gate turned away are in the material"
    );

    let call2 = material["call2"].as_array().expect("a call 2 list");
    let home = call2
        .iter()
        .find(|sample| sample["claim"] == hermes::HOME_SENTENCE)
        .expect("the restated claim's call 2 list is in the material");
    assert!(
        home["sample"].is_string() && home["at"].is_string(),
        "{home}"
    );
    for sample in call2 {
        for candidate in candidates(sample) {
            assert_candidate_shape(candidate, &mut ids);
        }
    }
    let neighbour = candidates(home)
        .iter()
        .find(|candidate| candidate["memory"] == first_memory.as_str())
        .expect("the claim's call 2 list holds the memory it restates");
    assert_eq!(neighbour["sentence"], hermes::HOME_SENTENCE);
    let similarity = neighbour["score"].as_f64().unwrap();
    assert!(
        (similarity - 1.0).abs() < 1e-6,
        "an identical sentence's similarity is 1: {neighbour}"
    );

    // Label every candidate, relevant when it's the home memory, and the
    // curve comes out of the material as written.
    let labels: String = ids
        .iter()
        .map(|id| {
            let relevant = recall
                .iter()
                .chain(call2)
                .flat_map(candidates)
                .any(|candidate| {
                    candidate["id"] == id.as_str() && candidate["memory"] == first_memory.as_str()
                });
            format!("{id:?} = {relevant}\n")
        })
        .collect();
    let labels_path = dir.private_file("labelling/labels.toml", &labels);
    let output = precision(&dir, &labels_path, &material_path);
    assert_ok(&output);
    let curve: Value = serde_json::from_str(&stdout(&output)).expect("the curve is JSON");
    assert_eq!(curve["recall"]["unlabelled"], 0, "{curve}");
    assert!(
        !curve["recall"]["curve"].as_array().unwrap().is_empty(),
        "{curve}"
    );
}

/// The material is written after the run, so a `--labelling` path that is
/// also one of the run's inputs or outputs would replace that file with
/// material: a paid cassette, the corpus, the probes, the report or the
/// aggregate export. Each is refused before the run, naming the flag, and
/// the file keeps its bytes. The cassette is also given by another
/// spelling of its path, which resolves to the same file.
#[test]
fn labelling_material_is_refused_over_the_runs_own_files() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let cassette = dir.private_path("cassettes/main.jsonl");
    let probes = dir.private_path("probes.toml");
    let report = dir.private_file("reports/collide.json", "prior report\n");
    let aggregate = dir.private_file("exports/aggregate.json", "prior aggregate\n");
    let cassette_alias = dir.private_path("labelling/../cassettes/main.jsonl");

    let cases: [(&str, &Path); 6] = [
        ("the corpus", &corpus),
        ("the cassette", &cassette),
        ("the cassette by another spelling", &cassette_alias),
        ("the probes file", &probes),
        ("the report", &report),
        ("the aggregate export", &aggregate),
    ];
    for (what, path) in cases {
        let before: Vec<(PathBuf, Vec<u8>)> = [&corpus, &cassette, &report, &aggregate]
            .into_iter()
            .map(|file| (file.clone(), fs::read(file).unwrap()))
            .collect();
        let run = replay_history(
            &dir,
            &corpus,
            "replay",
            PASSING_PROBES,
            "collide",
            None,
            &[
                "--aggregate",
                aggregate.to_str().unwrap(),
                "--labelling",
                path.to_str().unwrap(),
            ],
        );
        assert_eq!(
            run.output.status.code(),
            Some(2),
            "--labelling over {what} is refused: {}",
            stderr(&run.output)
        );
        assert!(
            stderr(&run.output).contains("--labelling")
                || stderr(&run.output).contains("labelling material"),
            "the refusal of {what} names the labelling flag: {}",
            stderr(&run.output)
        );
        for (file, bytes) in before {
            assert_eq!(
                fs::read(&file).unwrap(),
                bytes,
                "--labelling over {what} left {} as it was",
                file.display()
            );
        }
        assert_eq!(
            fs::read_to_string(&probes).unwrap(),
            PASSING_PROBES,
            "--labelling over {what} left the probes as they were"
        );
    }
}

/// A claim that changes something is shown its nearest memories whatever
/// their similarity, and BM25
/// neighbours are never held to the vector floor at all. So call 2's lists
/// can hold candidates scoring below the reconcile floor, and the material
/// keeps them as shown, with their real similarity: the curve is precision
/// over what call 2 saw, not a prediction of what another floor keeps.
#[test]
fn call2_material_keeps_a_flagged_claims_neighbour_below_the_floor() {
    let dir = TestDir::new();
    let state_db = dir.private_path("state.db");
    let db = StateDb::create(&state_db);
    let start = epoch("2026-01-05T09:00:00Z");
    let cat_quote = "we adopted a kitten called Miso";
    db.session("s1", "discord", Some("discord:1"), None, start);
    db.turn(
        "s1",
        start,
        &format!("{}, near the harbour.", hermes::HOME_QUOTE),
        "Noted.",
    );
    db.session("s2", "discord", Some("discord:1"), None, start + 3600.0);
    db.turn(
        "s2",
        start + 3600.0,
        &format!("Big news: {cat_quote} yesterday."),
        "Congratulations.",
    );
    drop(db);
    let corpus = dir.private_path("corpus/main.jsonl");
    assert_ok(&support::import(&dir, &state_db, &corpus));

    // One reply for every call: each claim survives only in the turn that
    // quotes it, the kitten changes something, and call 2 labels nothing.
    let mut home = claim(hermes::HOME_SENTENCE, hermes::HOME_QUOTE, "fact");
    home["claim"] = json!("c1");
    home["labels"] = json!([]);
    let mut cat = claim("Tim adopted a kitten called Miso.", cat_quote, "fact");
    cat["changes_something"] = json!(true);
    cat["claim"] = json!("c1");
    cat["labels"] = json!([]);
    let reply = json!({ "claims": [home, cat], "used_injected_ids": [], "operations": [] });
    let steps: Vec<Value> = (0..64).map(|_| json!({ "reply": reply })).collect();
    let script = dir.path("flagged-script.json");
    fs::write(&script, serde_json::to_vec(&steps).unwrap()).unwrap();

    let material_path = dir.private_path("labelling/material.json");
    let run = replay_history(
        &dir,
        &corpus,
        "live",
        PASSING_PROBES,
        "live",
        Some(&script),
        &["--labelling", material_path.to_str().unwrap()],
    );
    assert_ok(&run.output);
    let report = run.report();
    let floors = report["tuning"]["reconcile"]["embedding_floors"]
        .as_object()
        .expect("the report embeds the reconcile floors");
    assert_eq!(floors.len(), 1, "{floors:?}");
    let floor = floors.values().next().unwrap().as_f64().unwrap();
    let home_memory = report["memories"][0]["id"].as_str().unwrap().to_string();

    let material: Value =
        serde_json::from_slice(&fs::read(&material_path).expect("the material is written"))
            .unwrap();
    let flagged = material["call2"]
        .as_array()
        .unwrap()
        .iter()
        .find(|sample| sample["claim"] == "Tim adopted a kitten called Miso.")
        .expect("the flagged claim's call 2 list is in the material");
    let neighbour = candidates(flagged)
        .iter()
        .find(|candidate| candidate["memory"] == home_memory.as_str())
        .expect("call 2 was shown the home memory for the flagged claim");
    let score = neighbour["score"].as_f64().unwrap();
    assert!(
        score < floor,
        "the shown neighbour scores {score}, below the reconcile floor {floor}"
    );
}

// The precision curve.

fn precision(dir: &TestDir, labels: &Path, material: &Path) -> Output {
    asphodel(dir)
        .args(["report", "precision", "--labels"])
        .arg(labels)
        .arg("--material")
        .arg(material)
        .output()
        .unwrap()
}

/// Words the hand-written material holds that must never be printed.
const MATERIAL_SENTINEL: &str = "SENTINEL-MATERIAL-TEXT-4c1b";

/// A small material, written by hand in the shape `--labelling` writes.
/// Two recall samples and two call 2 lists; one recall candidate is left
/// unlabelled.
fn hand_material() -> Value {
    let memory = |n: u32| format!("00000000-0000-4000-8000-{n:012}");
    let candidate = |id: &str, n: u32, score: f64| {
        json!({
            "id": id,
            "memory": memory(n),
            "score": score,
            "sentence": format!("{MATERIAL_SENTINEL} sentence {n}")
        })
    };
    json!({
        "version": 1,
        "recall": [
            {
                "sample": "r01",
                "at": "2026-01-05T09:00:00Z",
                "session": "s1",
                "query": format!("{MATERIAL_SENTINEL} query one"),
                "candidates": [
                    candidate("r01.1", 1, 2.5),
                    candidate("r01.2", 2, 0.5),
                    candidate("r01.3", 3, -0.5)
                ]
            },
            {
                "sample": "r02",
                "at": "2026-01-06T09:00:00Z",
                "session": "s2",
                "query": format!("{MATERIAL_SENTINEL} query two"),
                "candidates": [
                    candidate("r02.1", 4, 1.5),
                    candidate("r02.2", 5, 0.5),
                    candidate("r02.3", 6, -1.5)
                ]
            }
        ],
        "call2": [
            {
                "sample": "c01",
                "at": "2026-01-05T09:00:30Z",
                "claim": format!("{MATERIAL_SENTINEL} claim one"),
                "candidates": [
                    candidate("c01.1", 1, 0.9),
                    candidate("c01.2", 2, 0.6),
                    candidate("c01.3", 3, 0.4)
                ]
            },
            {
                "sample": "c02",
                "at": "2026-01-06T09:00:30Z",
                "claim": format!("{MATERIAL_SENTINEL} claim two"),
                "candidates": [candidate("c02.1", 4, 0.6)]
            }
        ]
    })
}

/// Labels are a TOML table of candidate id to whether the candidate is
/// relevant: for recall, worth injecting for the query; for call 2, about
/// the same thing as the claim.
const HAND_LABELS: &str = r#"
"r01.1" = true
"r01.2" = false
"r01.3" = false
"r02.1" = true
"r02.2" = true
"c01.1" = true
"c01.2" = false
"c01.3" = false
"c02.1" = true
"#;

/// The curve, worked by hand. At each floor, the distinct scores of the
/// labelled candidates in ascending order, a candidate is kept when its
/// score is at or above the floor, as the gate and call 2 keep one, and
/// precision is the relevant fraction of those kept.
///
/// Recall (r02.3 unlabelled): scores 2.5 T, 0.5 F, -0.5 F, 1.5 T, 0.5 T.
/// - floor -0.5: 5 kept, 3 relevant, 0.6
/// - floor 0.5: 4 kept, 3 relevant, 0.75
/// - floor 1.5: 2 kept, 2 relevant, 1.0
/// - floor 2.5: 1 kept, 1 relevant, 1.0
///
/// Call 2: 0.9 T, 0.6 F, 0.4 F, 0.6 T.
/// - floor 0.4: 4 kept, 2 relevant, 0.5
/// - floor 0.6: 3 kept, 2 relevant, 2/3
/// - floor 0.9: 1 kept, 1 relevant, 1.0
#[test]
fn the_precision_curve_matches_a_hand_computed_example() {
    let dir = TestDir::new();
    let material = dir.private_file(
        "labelling/material.json",
        &serde_json::to_string_pretty(&hand_material()).unwrap(),
    );
    let labels = dir.private_file("labelling/labels.toml", HAND_LABELS);
    let output = precision(&dir, &labels, &material);
    assert_ok(&output);
    let out = stdout(&output);
    let curve: Value = serde_json::from_str(&out).expect("the curve is JSON on stdout");

    let points = |list: &str| -> Vec<(f64, u64, u64, f64)> {
        curve[list]["curve"]
            .as_array()
            .unwrap_or_else(|| panic!("a {list} curve: {curve}"))
            .iter()
            .map(|point| {
                (
                    point["floor"].as_f64().unwrap(),
                    point["kept"].as_u64().unwrap(),
                    point["relevant"].as_u64().unwrap(),
                    point["precision"].as_f64().unwrap(),
                )
            })
            .collect()
    };
    let close = |got: Vec<(f64, u64, u64, f64)>, want: &[(f64, u64, u64, f64)]| {
        assert_eq!(got.len(), want.len(), "{got:?}");
        for (got, want) in got.iter().zip(want) {
            assert!(
                (got.0 - want.0).abs() < 1e-9
                    && got.1 == want.1
                    && got.2 == want.2
                    && (got.3 - want.3).abs() < 1e-9,
                "{got:?} should be {want:?}"
            );
        }
    };
    close(
        points("recall"),
        &[
            (-0.5, 5, 3, 0.6),
            (0.5, 4, 3, 0.75),
            (1.5, 2, 2, 1.0),
            (2.5, 1, 1, 1.0),
        ],
    );
    close(
        points("call2"),
        &[(0.4, 4, 2, 0.5), (0.6, 3, 2, 2.0 / 3.0), (0.9, 1, 1, 1.0)],
    );
    assert_eq!(curve["recall"]["labelled"], 5, "{curve}");
    assert_eq!(curve["recall"]["unlabelled"], 1, "{curve}");
    assert_eq!(curve["call2"]["labelled"], 4, "{curve}");
    assert_eq!(curve["call2"]["unlabelled"], 0, "{curve}");

    // The curve is numbers: nothing of the material is printed.
    assert!(!out.contains(MATERIAL_SENTINEL), "{out}");
    assert!(!out.contains("00000000-0000-4000-8000"), "{out}");
}

/// A label naming no candidate in the material is refused, without
/// echoing the key, which is whatever was typed into the labels file.
#[test]
fn a_label_for_no_candidate_is_refused_without_quoting_it() {
    let dir = TestDir::new();
    let material = dir.private_file(
        "labelling/material.json",
        &serde_json::to_string_pretty(&hand_material()).unwrap(),
    );
    let sentinel = "SENTINEL-LABEL-KEY-31f0";
    let labels = dir.private_file(
        "labelling/labels.toml",
        &format!("{HAND_LABELS}\"{sentinel}\" = true\n"),
    );
    let output = precision(&dir, &labels, &material);
    assert_refused_without(&output, sentinel, &["labels.toml"]);
    assert!(!stdout(&output).contains(MATERIAL_SENTINEL));
}

/// A labels file or material that doesn't parse is refused naming the
/// file and the line, never quoting it.
#[test]
fn malformed_labels_and_material_are_refused_without_quoting_them() {
    let dir = TestDir::new();
    let material = dir.private_file(
        "labelling/material.json",
        &serde_json::to_string_pretty(&hand_material()).unwrap(),
    );
    let sentinel = "SENTINEL-LABEL-VALUE-8e2a";
    let labels_text = format!("{HAND_LABELS}\"r02.3\" = {sentinel}\n");
    let line = labels_text
        .lines()
        .position(|line| line.contains(sentinel))
        .unwrap()
        + 1;
    let labels = dir.private_file("labelling/labels.toml", &labels_text);
    assert_refused_without(
        &precision(&dir, &labels, &material),
        sentinel,
        &["labels.toml", &format!("line {line}")],
    );

    let labels = dir.private_file("labelling/good.toml", HAND_LABELS);
    let material_text = serde_json::to_string_pretty(&hand_material())
        .unwrap()
        .replacen(
            "\"score\": 2.5",
            &format!("\"score\": {MATERIAL_SENTINEL}"),
            1,
        );
    let line = material_text
        .lines()
        .position(|line| line.contains(&format!("\"score\": {MATERIAL_SENTINEL}")))
        .unwrap()
        + 1;
    let material = dir.private_file("labelling/broken.json", &material_text);
    assert_refused_without(
        &precision(&dir, &labels, &material),
        MATERIAL_SENTINEL,
        &["broken.json", &format!("line {line}")],
    );
}
