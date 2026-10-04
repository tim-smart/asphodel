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
//!   precision curve for each list as JSON, with how many labels matched a
//!   candidate, how many found nothing and how many candidates have none,
//!   and for recall how many relevant candidates rank in the top 8 of their
//!   sample by logit.
//! - Labels are keyed by what they judge: a recall label by the cleaned
//!   query and the memory, a call 2 label by the claim's chunk and ordinal
//!   and the neighbour. So labels written against one run's material score
//!   the same judgements in another run's. A file of candidate ids, the old
//!   form, is still read against the material it was written for, and
//!   `--convert` rewrites it in the keyed form.
//! - `--labels L` beside `--labelling` makes sampling prefer turns whose
//!   queries already have labels.
//! - `asphodel report rescore --material M --corpus C --rerank-query
//!   message|conversation --out M2` scores each recall sample's candidates
//!   again against the query the mode gives, keeping every id, and writes
//!   them in logit order. It compares rerankers' queries on fixed pools;
//!   it says nothing about prefetch's final ranking.
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
/// `mentioned_again` on the first neighbour, and a refresh writes nothing.
fn labelling_script(dir: &TestDir) -> PathBuf {
    let mut home = claim(hermes::HOME_SENTENCE, hermes::HOME_QUOTE, "fact");
    home["claim"] = json!("c1");
    home["labels"] = json!([{ "neighbour": "n1", "label": "mentioned_again" }]);
    let reply = json!({
        "claims": [home],
        "used_injected_ids": [],
        "sections": []
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

/// Tim's home turn, then a session whose second message leans on the
/// first exchange: only the assistant's reply names Auckland.
const ASKED: &str = "Can you check the weather for me?";
const ANSWERED: &str = "Auckland is sunny today.";
const LEANING: &str = "great, should I bring a jacket when I go out later";

fn conversation_history(dir: &TestDir) -> PathBuf {
    history_asking(dir, Some(ANSWERED))
}

/// [`conversation_history`], with `answer` as the reply to [`ASKED`], or
/// with no reply before [`LEANING`].
fn history_asking(dir: &TestDir, answer: Option<&str>) -> PathBuf {
    let state_db = dir.private_path("state.db");
    let db = StateDb::create(&state_db);
    let start = epoch("2026-01-05T09:00:00Z");
    db.session("s1", "discord", Some("discord:1"), None, start);
    db.turn(
        "s1",
        start,
        &format!("{}, near the harbour.", hermes::HOME_QUOTE),
        "Noted.",
    );
    let later = start + 3600.0;
    db.session("s2", "discord", Some("discord:1"), None, later);
    match answer {
        Some(answer) => db.turn("s2", later, ASKED, answer),
        None => db.message(hermes::Message {
            session: "s2",
            role: "user",
            content: ASKED,
            at: later,
            ..hermes::Message::default()
        }),
    }
    db.turn("s2", later + 300.0, LEANING, "Yes, a light one.");
    drop(db);
    let corpus = dir.private_path("corpus/main.jsonl");
    assert_ok(&support::import(dir, &state_db, &corpus));
    corpus
}

/// The material of a live run on `corpus` with the labelling stand-in and
/// `extra` flags.
fn material_of(dir: &TestDir, corpus: &Path, name: &str, extra: &[&str]) -> (Value, Value) {
    let script = labelling_script(dir);
    let path = dir.private_path(&format!("labelling/{name}.json"));
    let mut flags = vec!["--labelling", path.to_str().unwrap()];
    flags.extend_from_slice(extra);
    let run = replay_history(
        dir,
        corpus,
        "live",
        PASSING_PROBES,
        name,
        Some(&script),
        &flags,
    );
    assert_ok(&run.output);
    let material = serde_json::from_slice(&fs::read(&path).expect("the material is written"))
        .expect("the material is JSON");
    (run.report(), material)
}

/// The sample for the turn whose message is `message`.
fn sample_for<'a>(material: &'a Value, message: &str) -> &'a Value {
    material["recall"]
        .as_array()
        .expect("a recall list")
        .iter()
        .find(|sample| sample["raw_query"] == message)
        .unwrap_or_else(|| panic!("a sample for {message:?}: {material}"))
}

/// Each sample records both queries: `query`, the message the vector and
/// BM25 arms searched, and `rerank_query`, what the reranker scored
/// against. In explicit message mode they're the same.
#[test]
fn in_message_mode_the_material_records_the_message_as_the_rerank_query() {
    let dir = TestDir::new();
    let corpus = conversation_history(&dir);
    let overrides = dir.private_file("message.toml", "[injection]\nrerank_query = \"message\"\n");
    let (report, material) = material_of(
        &dir,
        &corpus,
        "message",
        &["--overrides", overrides.to_str().unwrap()],
    );
    assert_eq!(report["tuning"]["injection"]["rerank_query"], "message");
    for sample in material["recall"].as_array().unwrap() {
        assert_eq!(sample["rerank_query"], sample["query"], "{sample}");
    }
    let leaning = sample_for(&material, LEANING);
    assert_eq!(leaning["query"], LEANING);
}

/// By default replay reranks against the message, the previous message,
/// and the start of the assistant's reply to it from the corpus. The material
/// records that query beside the message, and scores candidates against it, so the existing
/// labels, keyed by memory, can be read against either run.
#[test]
fn by_default_replay_reranks_against_the_conversation_and_records_both_queries() {
    let dir = TestDir::new();
    let corpus = conversation_history(&dir);
    let (report, material) = material_of(&dir, &corpus, "conversation", &[]);
    assert_eq!(
        report["tuning"]["injection"]["rerank_query"], "conversation",
        "the report says which query the reranker scored against"
    );

    let leaning = sample_for(&material, LEANING);
    assert_eq!(leaning["query"], LEANING, "the arms search the message");
    let rerank_query = leaning["rerank_query"]
        .as_str()
        .expect("the sample records the rerank query");
    assert_eq!(rerank_query, format!("{LEANING}\n{ASKED}\n{ANSWERED}"));

    let home = candidates(leaning)
        .iter()
        .find(|candidate| candidate["sentence"] == hermes::HOME_SENTENCE)
        .expect("the home memory is a candidate");
    let score = home["score"].as_f64().unwrap();
    assert_eq!(score, fake_logit(rerank_query, hermes::HOME_SENTENCE));
    assert!(
        score > fake_logit(LEANING, hermes::HOME_SENTENCE),
        "the reply's Auckland lifts the home memory: {home}"
    );

    // The first message of a session has no conversation before it.
    let asked = sample_for(&material, ASKED);
    assert_eq!(asked["rerank_query"], ASKED);
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
    let reply = json!({ "claims": [home, cat], "used_injected_ids": [], "sections": [] });
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

/// The curve's points for `list`, each (floor, kept, relevant, precision),
/// match `want`.
fn assert_points(curve: &Value, list: &str, want: &[(f64, u64, u64, f64)]) {
    let got: Vec<(f64, u64, u64, f64)> = curve[list]["curve"]
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
        .collect();
    assert_eq!(got.len(), want.len(), "{list}: {got:?}");
    for (got, want) in got.iter().zip(want) {
        assert!(
            (got.0 - want.0).abs() < 1e-9
                && got.1 == want.1
                && got.2 == want.2
                && (got.3 - want.3).abs() < 1e-9,
            "{list}: {got:?} should be {want:?}"
        );
    }
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

    assert_points(
        &curve,
        "recall",
        &[
            (-0.5, 5, 3, 0.6),
            (0.5, 4, 3, 0.75),
            (1.5, 2, 2, 1.0),
            (2.5, 1, 1, 1.0),
        ],
    );
    assert_points(
        &curve,
        "call2",
        &[(0.4, 4, 2, 0.5), (0.6, 3, 2, 2.0 / 3.0), (0.9, 1, 1, 1.0)],
    );
    assert_eq!(curve["recall"]["labelled"], 5, "{curve}");
    assert_eq!(curve["recall"]["unlabelled"], 1, "{curve}");
    assert_eq!(curve["call2"]["labelled"], 4, "{curve}");
    assert_eq!(curve["call2"]["unlabelled"], 0, "{curve}");
    // Labels in the old form, read against the material they name, all
    // match.
    for (list, matched) in [("recall", 5), ("call2", 4)] {
        assert_eq!(curve[list]["matched"], matched, "{list}: {curve}");
        assert_eq!(curve[list]["unmatched"], 0, "{list}: {curve}");
    }

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

// Labels that survive a re-record.

/// A chunk id in the hand-written material.
fn hand_chunk(n: u32) -> String {
    format!("00000000-0000-4000-9000-{n:012}")
}

/// Another run's material over the same corpus, in the shape a keyed
/// `--labelling` writes: call 2 samples name the claim's chunk and
/// ordinal. It holds the same judgements as [`hand_material`] under new
/// sample numbers and in a new order, with one recall pair and one call 2
/// pair gone, and new candidates no label judges: a memory under a query it
/// wasn't labelled for, and a neighbour of another claim in the same chunk.
fn rerecorded_material() -> Value {
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
        "version": 2,
        "recall": [
            {
                "sample": "r01",
                "at": "2026-01-06T09:00:00Z",
                "session": "s2",
                "query": format!("{MATERIAL_SENTINEL} query two"),
                "candidates": [
                    candidate("r01.1", 5, 0.7),
                    candidate("r01.2", 4, 1.2),
                    candidate("r01.3", 7, -1.0)
                ]
            },
            {
                "sample": "r02",
                "at": "2026-01-05T09:00:00Z",
                "session": "s1",
                "query": format!("{MATERIAL_SENTINEL} query one"),
                "candidates": [
                    candidate("r02.1", 3, -0.2),
                    candidate("r02.2", 1, 2.0)
                ]
            },
            {
                "sample": "r03",
                "at": "2026-01-07T09:00:00Z",
                "session": "s3",
                "query": format!("{MATERIAL_SENTINEL} query three"),
                "candidates": [candidate("r03.1", 1, 0.1)]
            }
        ],
        "call2": [
            {
                "sample": "c01",
                "at": "2026-01-06T09:00:30Z",
                "chunk": hand_chunk(2),
                "ordinal": 1,
                "claim": format!("{MATERIAL_SENTINEL} claim two"),
                "candidates": [candidate("c01.1", 4, 0.55)]
            },
            {
                "sample": "c02",
                "at": "2026-01-05T09:00:30Z",
                "chunk": hand_chunk(1),
                "ordinal": 0,
                "claim": format!("{MATERIAL_SENTINEL} claim one, worded anew"),
                "candidates": [
                    candidate("c02.1", 2, 0.65),
                    candidate("c02.2", 1, 0.85)
                ]
            },
            {
                "sample": "c03",
                "at": "2026-01-05T09:00:30Z",
                "chunk": hand_chunk(1),
                "ordinal": 1,
                "claim": format!("{MATERIAL_SENTINEL} claim three"),
                "candidates": [candidate("c03.1", 3, 0.5)]
            }
        ]
    })
}

/// [`HAND_LABELS`] keyed by what they judge, with the chunks and ordinals
/// [`hand_material`]'s claims had: claim one is ordinal 0 of chunk 1, claim
/// two ordinal 1 of chunk 2.
fn keyed_hand_labels() -> String {
    let memory = |n: u32| format!("00000000-0000-4000-8000-{n:012}");
    let query = |text: &str| format!("{MATERIAL_SENTINEL} query {text}");
    let recall = [
        ("one", 1, true),
        ("one", 2, false),
        ("one", 3, false),
        ("two", 4, true),
        ("two", 5, true),
    ]
    .map(|(text, n, relevant)| {
        json!({ "query": query(text), "memory": memory(n), "relevant": relevant })
    });
    let call2 = [
        (1, 0, 1, true),
        (1, 0, 2, false),
        (1, 0, 3, false),
        (2, 1, 4, true),
    ]
    .map(|(chunk, ordinal, n, relevant)| {
        json!({
            "chunk": hand_chunk(chunk),
            "ordinal": ordinal,
            "memory": memory(n),
            "relevant": relevant
        })
    });
    toml::to_string(&json!({ "recall": recall, "call2": call2 })).unwrap()
}

/// The recall curve of [`HAND_LABELS`]' judgements over
/// [`rerecorded_material`], worked by hand.
///
/// Matched (4): query one with memories 1 and 3, query two with 4 and 5.
/// Unmatched (1): query one with memory 2, which the run no longer shows.
/// Unlabelled (2): memory 7 under query two, and memory 1 under query
/// three, labelled only for query one.
/// Scores: 0.7 T, 1.2 T, -0.2 F, 2.0 T.
/// - floor -0.2: 4 kept, 3 relevant, 0.75
/// - floor 0.7: 3 kept, 3 relevant, 1.0
/// - floor 1.2: 2 kept, 2 relevant, 1.0
/// - floor 2.0: 1 kept, 1 relevant, 1.0
fn assert_rerecorded_recall(curve: &Value) {
    assert_eq!(curve["recall"]["matched"], 4, "{curve}");
    assert_eq!(curve["recall"]["unmatched"], 1, "{curve}");
    assert_eq!(curve["recall"]["labelled"], 4, "{curve}");
    assert_eq!(curve["recall"]["unlabelled"], 2, "{curve}");
    assert_points(
        curve,
        "recall",
        &[
            (-0.2, 4, 3, 0.75),
            (0.7, 3, 3, 1.0),
            (1.2, 2, 2, 1.0),
            (2.0, 1, 1, 1.0),
        ],
    );
}

/// Labels keyed by query and memory, and by the claim's chunk and ordinal
/// and the neighbour, score the same judgements in a re-recorded run's
/// material whatever its candidates are numbered. A pair the new run
/// doesn't show is counted as unmatched rather than refused, and the same
/// memory under another query, or under another claim of the same chunk,
/// is unlabelled. The claim's wording isn't part of the key. Nothing of
/// the labels or the material is printed.
///
/// Call 2, worked by hand. Matched (3): chunk 2 ordinal 1 with memory 4,
/// chunk 1 ordinal 0 with memories 1 and 2. Unmatched (1): chunk 1 ordinal
/// 0 with memory 3. Unlabelled (1): memory 3 under chunk 1 ordinal 1.
/// Scores: 0.55 T, 0.65 F, 0.85 T.
/// - floor 0.55: 3 kept, 2 relevant, 2/3
/// - floor 0.65: 2 kept, 1 relevant, 0.5
/// - floor 0.85: 1 kept, 1 relevant, 1.0
#[test]
fn keyed_labels_score_the_same_judgements_in_a_rerecorded_runs_material() {
    let dir = TestDir::new();
    let material = dir.private_file(
        "labelling/rerecorded.json",
        &serde_json::to_string_pretty(&rerecorded_material()).unwrap(),
    );
    let labels = dir.private_file("labelling/keyed.toml", &keyed_hand_labels());
    let output = precision(&dir, &labels, &material);
    assert_ok(&output);
    let out = stdout(&output);
    let curve: Value = serde_json::from_str(&out).expect("the curve is JSON on stdout");

    assert_rerecorded_recall(&curve);
    assert_eq!(curve["call2"]["matched"], 3, "{curve}");
    assert_eq!(curve["call2"]["unmatched"], 1, "{curve}");
    assert_eq!(curve["call2"]["labelled"], 3, "{curve}");
    assert_eq!(curve["call2"]["unlabelled"], 1, "{curve}");
    assert_points(
        &curve,
        "call2",
        &[
            (0.55, 3, 2, 2.0 / 3.0),
            (0.65, 2, 1, 0.5),
            (0.85, 1, 1, 1.0),
        ],
    );

    assert!(!out.contains(MATERIAL_SENTINEL), "{out}");
    assert!(!out.contains("00000000-0000-4000"), "{out}");
    assert!(!stderr(&output).contains(MATERIAL_SENTINEL));
}

/// Labels in the old form, candidate ids, carry over once converted:
/// `--convert` reads them against the material they were written for, a
/// material from before keys were recorded, and writes them keyed by query
/// and memory. The converted file then scores the re-recorded material's
/// recall candidates as if it had been written keyed.
#[test]
fn old_id_labels_convert_to_keyed_labels_that_score_another_run() {
    let dir = TestDir::new();
    let old_material = dir.private_file(
        "labelling/material.json",
        &serde_json::to_string_pretty(&hand_material()).unwrap(),
    );
    let old_labels = dir.private_file("labelling/labels.toml", HAND_LABELS);
    let converted = dir.private_path("labelling/converted.toml");
    let output = asphodel(&dir)
        .args(["report", "precision", "--labels"])
        .arg(&old_labels)
        .arg("--material")
        .arg(&old_material)
        .arg("--convert")
        .arg(&converted)
        .output()
        .unwrap();
    assert_ok(&output);
    assert!(converted.exists(), "the keyed labels are written");
    assert!(!stdout(&output).contains(MATERIAL_SENTINEL));

    let material = dir.private_file(
        "labelling/rerecorded.json",
        &serde_json::to_string_pretty(&rerecorded_material()).unwrap(),
    );
    let output = precision(&dir, &converted, &material);
    assert_ok(&output);
    let curve: Value = serde_json::from_str(&stdout(&output)).expect("the curve is JSON");
    assert_rerecorded_recall(&curve);
}

/// A conversion can keep nothing: every call 2 label on material from
/// before keys were recorded is dropped, and so are recall labels judging
/// one query and memory both ways. Each loss is counted, and the file it
/// writes is still keyed labels, which `replay --labels` accepts rather
/// than asking for another conversion.
#[test]
fn a_conversion_that_keeps_nothing_counts_its_losses_and_stays_keyed() {
    let dir = TestDir::new();
    // The second recall sample asks the first's query, and its first
    // candidate is the first's memory, so r01.1 and r02.1 judge one key.
    let mut material = hand_material();
    material["recall"][1]["query"] = material["recall"][0]["query"].clone();
    material["recall"][1]["candidates"][0]["memory"] =
        material["recall"][0]["candidates"][0]["memory"].clone();
    let old_material = dir.private_file(
        "labelling/material.json",
        &serde_json::to_string_pretty(&material).unwrap(),
    );
    let old_labels = dir.private_file(
        "labelling/labels.toml",
        "\"r01.1\" = true\n\"r02.1\" = false\n\"c01.1\" = true\n\"c01.2\" = false\n\"c02.1\" = true\n",
    );
    let converted = dir.private_path("labelling/converted.toml");
    let output = asphodel(&dir)
        .args(["report", "precision", "--labels"])
        .arg(&old_labels)
        .arg("--material")
        .arg(&old_material)
        .arg("--convert")
        .arg(&converted)
        .output()
        .unwrap();
    assert_ok(&output);
    let curve: Value = serde_json::from_str(&stdout(&output)).expect("the curve is JSON");
    assert_eq!(
        curve["converted"],
        json!({ "recall": 0, "call2": 0, "dropped_call2": 3, "conflicting": 2 }),
        "{curve}"
    );

    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let material_path = dir.private_path("labelling/next.json");
    let run = replay_history(
        &dir,
        &corpus,
        "replay",
        PASSING_PROBES,
        "next",
        None,
        &[
            "--labelling",
            material_path.to_str().unwrap(),
            "--labels",
            converted.to_str().unwrap(),
        ],
    );
    assert_ok(&run.output);
    assert!(material_path.exists(), "the material is written");
}

/// The query of a turn of [`labelling_history`], as cleaned for the
/// reranker.
fn labelling_query(turn: usize) -> String {
    match turn {
        0 | 1 => unreachable!("the home turns aren't questions"),
        n if n % 2 == 0 => format!("Is Auckland sunny today, question {n}?"),
        n => format!("What is on the radio, question {n}?"),
    }
}

/// A re-recorded run carries labels over: one run's material is labelled
/// by key, plus a label for a turn its even spread skipped, and a second
/// run given those labels with `--labels` samples the labelled turns in
/// preference, the skipped one included, still 50 of them. Every
/// candidate the second run shows is labelled; recall labels it no longer
/// shows are unmatched, and every call 2 label matches, keyed by the
/// chunk and ordinal the material records for each claim.
#[test]
fn sampling_prefers_labelled_queries_so_labels_carry_over() {
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
    let first_path = dir.private_path("labelling/first.json");
    let first = replay_history(
        &dir,
        &corpus,
        "replay",
        PASSING_PROBES,
        "first",
        None,
        &["--labelling", first_path.to_str().unwrap()],
    );
    assert_ok(&first.output);
    let report = first.report();
    let home = report["memories"][0]["id"]
        .as_str()
        .expect("the history makes a memory");
    let first_material: Value = serde_json::from_slice(&fs::read(&first_path).unwrap()).unwrap();
    let queries = |material: &Value| -> BTreeSet<String> {
        material["recall"]
            .as_array()
            .expect("a recall list")
            .iter()
            .map(|sample| sample["query"].as_str().unwrap().to_string())
            .collect()
    };
    let first_queries = queries(&first_material);
    let skipped = (2..TURNS)
        .map(labelling_query)
        .find(|query| !first_queries.contains(query))
        .expect("the even spread skips a question turn");

    let mut recall: Vec<Value> = first_material["recall"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|sample| {
            candidates(sample).iter().map(|candidate| {
                json!({
                    "query": sample["query"],
                    "memory": candidate["memory"],
                    "relevant": candidate["memory"] == home
                })
            })
        })
        .collect();
    recall.push(json!({ "query": skipped, "memory": home, "relevant": false }));
    let call2: Vec<Value> = first_material["call2"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|sample| {
            let chunk = sample["chunk"].as_str().expect("a call 2 sample's chunk");
            assert!(uuid::Uuid::parse_str(chunk).is_ok(), "{sample}");
            let ordinal = sample["ordinal"]
                .as_u64()
                .expect("a call 2 sample's claim ordinal");
            candidates(sample).iter().map(move |candidate| {
                json!({
                    "chunk": chunk,
                    "ordinal": ordinal,
                    "memory": candidate["memory"],
                    "relevant": candidate["memory"] == home
                })
            })
        })
        .collect();
    assert!(!call2.is_empty(), "call 2 ran: {first_material}");
    let (recall_labels, call2_labels) = (recall.len(), call2.len());
    let labels = dir.private_file(
        "labelling/keyed.toml",
        &toml::to_string(&json!({ "recall": recall, "call2": call2 })).unwrap(),
    );

    let second_path = dir.private_path("labelling/second.json");
    let second = replay_history(
        &dir,
        &corpus,
        "replay",
        PASSING_PROBES,
        "second",
        None,
        &[
            "--labelling",
            second_path.to_str().unwrap(),
            "--labels",
            labels.to_str().unwrap(),
        ],
    );
    assert_ok(&second.output);
    let second_material: Value = serde_json::from_slice(&fs::read(&second_path).unwrap()).unwrap();
    let second_queries = queries(&second_material);
    assert_eq!(
        second_material["recall"].as_array().unwrap().len(),
        50,
        "still 50 sampled turns"
    );
    assert!(
        second_queries.contains(&skipped),
        "the labelled turn the first run skipped is sampled"
    );

    let output = precision(&dir, &labels, &second_path);
    assert_ok(&output);
    let curve: Value = serde_json::from_str(&stdout(&output)).expect("the curve is JSON");
    assert_eq!(curve["recall"]["unlabelled"], 0, "{curve}");
    assert!(curve["recall"]["matched"].as_u64().unwrap() > 0, "{curve}");
    assert_eq!(
        curve["recall"]["matched"].as_u64().unwrap()
            + curve["recall"]["unmatched"].as_u64().unwrap(),
        recall_labels as u64,
        "every recall label either matched or found nothing: {curve}"
    );
    assert_eq!(curve["call2"]["matched"], call2_labels as u64, "{curve}");
    assert_eq!(curve["call2"]["unmatched"], 0, "{curve}");
    assert_eq!(curve["call2"]["unlabelled"], 0, "{curve}");
}

// Top 8 by logit.

/// A material whose list order differs from its logit order, with one
/// sample longer than 8, one shorter, and one of nine tied logits.
///
/// - r01: r01.1 at -9 (relevant, listed first but ranked last), r01.2 to
///   r01.9 at 8 down to 1 (r01.5 relevant, ranked 4th), r01.10 at 0.5
///   (relevant, ranked 9th): 1 of 3 in the top 8.
/// - r02: r02.3 at 3 (unlabelled), r02.1 at 1, r02.2 at 0 (relevant): 1 of 1.
/// - r03: nine at 0, r03.8 and r03.9 relevant. Ties keep list order, so
///   r03.8 is 8th and r03.9 9th: 1 of 2.
fn top8_material() -> (Value, String) {
    let memory = |n: usize| format!("00000000-0000-4000-8000-{n:012}");
    let candidate = |id: String, n: usize, score: f64| {
        json!({
            "id": id,
            "memory": memory(n),
            "score": score,
            "sentence": format!("{MATERIAL_SENTINEL} sentence {n}")
        })
    };
    let sample = |id: &str, candidates: Vec<Value>| {
        json!({
            "sample": id,
            "at": "2026-01-05T09:00:00Z",
            "session": "s1",
            "query": format!("{MATERIAL_SENTINEL} query"),
            "candidates": candidates
        })
    };
    let mut r01 = vec![candidate("r01.1".into(), 1, -9.0)];
    for n in 2..=9 {
        r01.push(candidate(format!("r01.{n}"), n, (10 - n) as f64));
    }
    r01.push(candidate("r01.10".into(), 10, 0.5));
    let r02 = vec![
        candidate("r02.1".into(), 21, 1.0),
        candidate("r02.2".into(), 22, 0.0),
        candidate("r02.3".into(), 23, 3.0),
    ];
    let r03 = (1..=9)
        .map(|n| candidate(format!("r03.{n}"), 30 + n, 0.0))
        .collect();
    let material = json!({
        "version": 1,
        "recall": [sample("r01", r01), sample("r02", r02), sample("r03", r03)],
        "call2": []
    });
    let relevant = ["r01.1", "r01.5", "r01.10", "r02.2", "r03.8", "r03.9"];
    let labels = material["recall"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(candidates)
        .filter_map(|candidate| {
            let id = candidate["id"].as_str().unwrap();
            (id != "r02.3").then(|| format!("{id:?} = {}\n", relevant.contains(&id)))
        })
        .collect();
    (material, labels)
}

/// `report precision` counts, for recall, the relevant labelled candidates
/// ranked in the top 8 of their sample by logit, ties in list order, out
/// of every relevant labelled candidate. Unlabelled candidates take places
/// but aren't counted.
#[test]
fn precision_counts_the_relevant_candidates_in_each_samples_top_8_by_logit() {
    let dir = TestDir::new();
    let (material, labels) = top8_material();
    let material = dir.private_file(
        "labelling/top8.json",
        &serde_json::to_string_pretty(&material).unwrap(),
    );
    let labels = dir.private_file("labelling/top8.toml", &labels);
    let output = precision(&dir, &labels, &material);
    assert_ok(&output);
    let out = stdout(&output);
    let curve: Value = serde_json::from_str(&out).expect("the curve is JSON on stdout");
    assert_eq!(
        curve["recall"]["top8"],
        json!({ "found": 3, "relevant": 6 }),
        "{curve}"
    );
    assert_eq!(curve["recall"]["labelled"], 21, "{curve}");
    assert_eq!(curve["recall"]["unlabelled"], 1, "{curve}");
    assert!(!out.contains(MATERIAL_SENTINEL), "{out}");
}

// Rescoring fixed pools.

fn rescore(dir: &TestDir, material: &Path, corpus: &Path, mode: &str, out: &Path) -> Output {
    asphodel(dir)
        .args(["report", "rescore", "--material"])
        .arg(material)
        .arg("--corpus")
        .arg(corpus)
        .args(["--rerank-query", mode])
        .arg("--out")
        .arg(out)
        .output()
        .unwrap()
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("the file is written")).expect("it is JSON")
}

/// `candidates` by logit, highest first, ties in the order given.
fn logit_order(candidates: &[Value]) -> Value {
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| {
        b["score"]
            .as_f64()
            .unwrap()
            .total_cmp(&a["score"].as_f64().unwrap())
    });
    Value::Array(sorted)
}

/// Asserts `after` is `before` rescored: the same sample, the same
/// candidates by id with their memories and sentences, each scored with the
/// fake reranker against `after`'s rerank query, in logit order. Returns
/// that query.
fn assert_rescored<'a>(before: &Value, after: &'a Value) -> &'a str {
    for key in ["sample", "at", "session", "query", "raw_query"] {
        assert_eq!(after[key], before[key], "{key} is kept: {after}");
    }
    let rerank_query = after["rerank_query"]
        .as_str()
        .unwrap_or_else(|| panic!("a rescored sample records its rerank query: {after}"));
    let expected: Vec<Value> = candidates(before)
        .iter()
        .map(|candidate| {
            let mut candidate = candidate.clone();
            let sentence = candidate["sentence"].as_str().unwrap();
            candidate["score"] = json!(fake_logit(rerank_query, sentence));
            candidate
        })
        .collect();
    assert_eq!(after["candidates"], logit_order(&expected), "{after}");
    rerank_query
}

/// In message mode, rescoring reranks against the query a message-mode
/// replay did, so every score comes back as it was, now in logit order.
/// Sample and candidate ids are kept, so labels written for the material
/// apply to the rescored one, and call 2's lists are copied unchanged.
#[test]
fn rescoring_by_the_message_reproduces_the_replay_logits_in_logit_order() {
    let dir = TestDir::new();
    let corpus = conversation_history(&dir);
    let overrides = dir.private_file("message.toml", "[injection]\nrerank_query = \"message\"\n");
    let (_, material) = material_of(
        &dir,
        &corpus,
        "message",
        &["--overrides", overrides.to_str().unwrap()],
    );
    let out = dir.private_path("labelling/rescored-message.json");
    assert_ok(&rescore(
        &dir,
        &dir.private_path("labelling/message.json"),
        &corpus,
        "message",
        &out,
    ));
    let rescored = read_json(&out);
    assert_eq!(rescored["version"], material["version"]);
    assert_eq!(rescored["call2"], material["call2"]);
    let before = material["recall"].as_array().unwrap();
    let after = rescored["recall"].as_array().unwrap();
    assert_eq!(after.len(), before.len());
    for (before, after) in before.iter().zip(after) {
        let rerank_query = assert_rescored(before, after);
        assert_eq!(rerank_query, before["rerank_query"], "{after}");
    }
}

/// In conversation mode, rescoring takes the previous message and the
/// reply from the corpus, as a conversation replay would, and scores the
/// same pools against them. The same input rescores to the same bytes.
#[test]
fn rescoring_by_the_conversation_scores_the_same_pools_against_it() {
    let dir = TestDir::new();
    let corpus = conversation_history(&dir);
    let (_, material) = material_of(&dir, &corpus, "message", &[]);
    let input = dir.private_path("labelling/message.json");
    let out = dir.private_path("labelling/rescored.json");
    assert_ok(&rescore(&dir, &input, &corpus, "conversation", &out));
    let rescored = read_json(&out);
    assert_eq!(rescored["call2"], material["call2"]);
    let before = material["recall"].as_array().unwrap();
    let after = rescored["recall"].as_array().unwrap();
    assert_eq!(after.len(), before.len());
    for (before, after) in before.iter().zip(after) {
        assert_rescored(before, after);
    }
    assert_eq!(
        sample_for(&rescored, LEANING)["rerank_query"],
        format!("{LEANING}\n{ASKED}\n{ANSWERED}")
    );
    assert_eq!(sample_for(&rescored, ASKED)["rerank_query"], ASKED);
    let home = candidates(sample_for(&rescored, LEANING))
        .iter()
        .find(|candidate| candidate["sentence"] == hermes::HOME_SENTENCE)
        .expect("the home memory is still a candidate");
    assert!(
        home["score"].as_f64().unwrap() > fake_logit(LEANING, hermes::HOME_SENTENCE),
        "the reply's Auckland lifts the home memory: {home}"
    );

    let again = dir.private_path("labelling/again.json");
    assert_ok(&rescore(&dir, &input, &corpus, "conversation", &again));
    assert_eq!(fs::read(&again).unwrap(), fs::read(&out).unwrap());
}

/// A previous message Hermes never answered gives the conversation no
/// reply, in a conversation replay and in a rescore alike.
#[test]
fn an_unanswered_previous_message_gives_the_conversation_no_reply() {
    let dir = TestDir::new();
    let corpus = history_asking(&dir, None);
    let overrides = dir.private_file(
        "conversation.toml",
        "[injection]\nrerank_query = \"conversation\"\n",
    );
    let (_, replayed) = material_of(
        &dir,
        &corpus,
        "conversation",
        &["--overrides", overrides.to_str().unwrap()],
    );
    let expected = format!("{LEANING}\n{ASKED}");
    assert_eq!(sample_for(&replayed, LEANING)["rerank_query"], expected);

    let (_, _) = material_of(&dir, &corpus, "message", &[]);
    let out = dir.private_path("labelling/rescored.json");
    assert_ok(&rescore(
        &dir,
        &dir.private_path("labelling/message.json"),
        &corpus,
        "conversation",
        &out,
    ));
    assert_eq!(
        sample_for(&read_json(&out), LEANING)["rerank_query"],
        expected
    );
}

/// [`hand_material`]'s first sample alone: Tim's home turn in session s1,
/// which [`conversation_history`] holds.
fn matching_material() -> Value {
    let mut material = hand_material();
    material["recall"].as_array_mut().unwrap().truncate(1);
    material
}

/// A sample with no prefetch in the corpus at its session and time is
/// refused, naming the sample and quoting nothing, and nothing is written.
#[test]
fn a_sample_missing_from_the_corpus_is_refused_without_quoting_it() {
    let dir = TestDir::new();
    let corpus = conversation_history(&dir);
    // r02 is in session s2 a day later, where the corpus has no prefetch.
    let material = dir.private_file(
        "labelling/hand.json",
        &serde_json::to_string_pretty(&hand_material()).unwrap(),
    );
    let out = dir.private_path("labelling/rescored.json");
    for mode in ["message", "conversation"] {
        let output = rescore(&dir, &material, &corpus, mode, &out);
        assert_refused_without(&output, MATERIAL_SENTINEL, &["r02", "corpus"]);
        for text in [hermes::HOME_QUOTE, ASKED, ANSWERED, LEANING] {
            assert!(!stderr(&output).contains(text), "{}", stderr(&output));
            assert!(!stdout(&output).contains(text), "{}", stdout(&output));
        }
        assert!(!out.exists(), "nothing is written in {mode} mode");
    }
}

/// The material, the corpus and the output are history, so each must be
/// inside the private dir, and the output may not replace an input.
#[test]
fn rescoring_is_refused_outside_the_private_dir_and_over_its_inputs() {
    let dir = TestDir::new();
    let corpus = conversation_history(&dir);
    let text = serde_json::to_string_pretty(&matching_material()).unwrap();
    let material = dir.private_file("labelling/hand.json", &text);
    let out = dir.private_path("labelling/rescored.json");
    assert_ok(&rescore(&dir, &material, &corpus, "message", &out));
    fs::remove_file(&out).unwrap();

    let outside_material = dir.path("material.json");
    fs::write(&outside_material, &text).unwrap();
    let output = rescore(&dir, &outside_material, &corpus, "message", &out);
    assert_refused(&output, "private");
    assert!(!out.exists());

    let outside_corpus = dir.path("corpus.jsonl");
    fs::copy(&corpus, &outside_corpus).unwrap();
    let output = rescore(&dir, &material, &outside_corpus, "message", &out);
    assert_refused(&output, "private");
    assert!(!out.exists());

    let outside_out = dir.path("rescored.json");
    let output = rescore(&dir, &material, &corpus, "message", &outside_out);
    assert_refused(&output, "private");
    assert!(!outside_out.exists());

    let corpus_bytes = fs::read(&corpus).unwrap();
    for input in [&material, &corpus] {
        let output = rescore(&dir, &material, &corpus, "message", input);
        assert_refused(&output, "--out");
    }
    assert_eq!(fs::read(&material).unwrap(), text.as_bytes());
    assert_eq!(fs::read(&corpus).unwrap(), corpus_bytes);
}

/// Material that doesn't parse is refused naming the file and the line,
/// never quoting it.
#[test]
fn malformed_material_is_refused_by_rescore_without_quoting_it() {
    let dir = TestDir::new();
    let corpus = conversation_history(&dir);
    let text = serde_json::to_string_pretty(&matching_material())
        .unwrap()
        .replacen(
            "\"score\": 2.5",
            &format!("\"score\": {MATERIAL_SENTINEL}"),
            1,
        );
    let line = text
        .lines()
        .position(|line| line.contains(&format!("\"score\": {MATERIAL_SENTINEL}")))
        .unwrap()
        + 1;
    let material = dir.private_file("labelling/broken.json", &text);
    let out = dir.private_path("labelling/rescored.json");
    assert_refused_without(
        &rescore(&dir, &material, &corpus, "conversation", &out),
        MATERIAL_SENTINEL,
        &["broken.json", &format!("line {line}")],
    );
    assert!(!out.exists());
}

// Top 8 with labels keyed by query and memory.

/// A version 2 material for top 8 under keyed labels, and those labels.
/// Recall labels are keyed by the cleaned query and the memory, so a label
/// reaches the memory in every sample with that query, and none with
/// another.
///
/// - r01, query A: memory 1 at -9 (relevant, listed first but ranked
///   last), memories 2 to 9 at 8 down to 1 (5 relevant, ranked 4th), memory
///   10 at 0.5 (relevant, ranked 9th): 1 of 3 in the top 8.
/// - r02, query A again: memory 11 at 3 (unlabelled), memory 5 at 2 (its
///   query A label makes it relevant here too): 1 of 1.
/// - r03, query B: memory 1 at 5 (labelled only under query A, so
///   unlabelled here, but it takes 1st place), then memories 21 to 29 tied
///   at 0 in list order (27 and 28 relevant, the rest unlabelled): 27 is
///   8th and 28 9th, so 1 of 2.
/// - One label, query B with memory 99, judges nothing the material shows:
///   unmatched, and not counted.
///
/// So 3 found of 6 relevant; 13 candidates labelled and 9 not; 12 labels
/// matched and 1 unmatched.
fn keyed_top8_fixture() -> (Value, String) {
    let memory = |n: u32| format!("00000000-0000-4000-8000-{n:012}");
    let query = |name: &str| format!("{MATERIAL_SENTINEL} query {name}");
    let candidate = |id: String, n: u32, score: f64| {
        json!({
            "id": id,
            "memory": memory(n),
            "score": score,
            "sentence": format!("{MATERIAL_SENTINEL} sentence {n}")
        })
    };
    let sample = |id: &str, at: &str, name: &str, candidates: Vec<Value>| {
        json!({
            "sample": id,
            "at": at,
            "session": format!("s-{id}"),
            "query": query(name),
            "candidates": candidates
        })
    };
    let mut r01 = vec![candidate("r01.1".into(), 1, -9.0)];
    for n in 2..=9 {
        r01.push(candidate(format!("r01.{n}"), n, f64::from(10 - n)));
    }
    r01.push(candidate("r01.10".into(), 10, 0.5));
    let r02 = vec![
        candidate("r02.1".into(), 11, 3.0),
        candidate("r02.2".into(), 5, 2.0),
    ];
    let mut r03 = vec![candidate("r03.1".into(), 1, 5.0)];
    for n in 21..=29 {
        r03.push(candidate(format!("r03.{}", n - 19), n, 0.0));
    }
    let material = json!({
        "version": 2,
        "recall": [
            sample("r01", "2026-01-05T09:00:00Z", "A", r01),
            sample("r02", "2026-01-06T09:00:00Z", "A", r02),
            sample("r03", "2026-01-07T09:00:00Z", "B", r03)
        ],
        "call2": []
    });
    let label = |name: &str, n: u32, relevant: bool| json!({ "query": query(name), "memory": memory(n), "relevant": relevant });
    let mut recall: Vec<Value> = (1..=10)
        .map(|n| label("A", n, [1, 5, 10].contains(&n)))
        .collect();
    recall.extend([
        label("B", 27, true),
        label("B", 28, true),
        label("B", 99, true),
    ]);
    let labels = toml::to_string(&json!({ "recall": recall })).unwrap();
    (material, labels)
}

/// The same judgements as candidate ids: every candidate a keyed label
/// reaches, under its id in `material`.
fn id_labels_for(material: &Value, keyed: &str) -> String {
    let keyed: Value = toml::from_str(keyed).unwrap();
    let judged: Vec<(&Value, &Value, bool)> = keyed["recall"]
        .as_array()
        .unwrap()
        .iter()
        .map(|label| {
            (
                &label["query"],
                &label["memory"],
                label["relevant"].as_bool().unwrap(),
            )
        })
        .collect();
    let mut labels = String::new();
    for sample in material["recall"].as_array().unwrap() {
        for candidate in candidates(sample) {
            if let Some((_, _, relevant)) = judged.iter().find(|(query, memory, _)| {
                **query == sample["query"] && **memory == candidate["memory"]
            }) {
                labels.push_str(&format!(
                    "{:?} = {relevant}\n",
                    candidate["id"].as_str().unwrap()
                ));
            }
        }
    }
    labels
}

/// Top 8 is the same for keyed labels as for candidate ids: each sample's
/// candidates by logit, highest first and ties in list order, unlabelled
/// candidates taking places but never counting, and relevant counting the
/// labelled-relevant candidates the material shows. Keyed labels keep their
/// matching: a label reaches its memory under its query in any sample, a
/// label judging nothing shown is unmatched, and the two paths give the
/// same curve.
#[test]
fn keyed_labels_and_candidate_ids_count_the_same_top_8() {
    let dir = TestDir::new();
    let (material, keyed) = keyed_top8_fixture();
    let ids = id_labels_for(&material, &keyed);
    let material_path = dir.private_file(
        "labelling/keyed-top8.json",
        &serde_json::to_string_pretty(&material).unwrap(),
    );
    let keyed_path = dir.private_file("labelling/keyed-top8.toml", &keyed);
    let ids_path = dir.private_file("labelling/id-top8.toml", &ids);

    let output = precision(&dir, &keyed_path, &material_path);
    assert_ok(&output);
    let out = stdout(&output);
    let by_key: Value = serde_json::from_str(&out).expect("the curve is JSON on stdout");
    assert_eq!(
        by_key["recall"]["top8"],
        json!({ "found": 3, "relevant": 6 }),
        "{by_key}"
    );
    assert_eq!(by_key["recall"]["labelled"], 13, "{by_key}");
    assert_eq!(by_key["recall"]["unlabelled"], 9, "{by_key}");
    assert_eq!(by_key["recall"]["matched"], 12, "{by_key}");
    assert_eq!(by_key["recall"]["unmatched"], 1, "{by_key}");
    assert!(!out.contains(MATERIAL_SENTINEL), "{out}");
    assert!(!out.contains("00000000-0000-4000"), "{out}");

    let output = precision(&dir, &ids_path, &material_path);
    assert_ok(&output);
    let by_id: Value = serde_json::from_str(&stdout(&output)).expect("the curve is JSON");
    assert_eq!(by_id["recall"]["top8"], by_key["recall"]["top8"], "{by_id}");
    assert_eq!(by_id["recall"]["labelled"], 13, "{by_id}");
    assert_eq!(by_id["recall"]["unlabelled"], 9, "{by_id}");
    assert_eq!(
        by_id["recall"]["curve"], by_key["recall"]["curve"],
        "{by_id}"
    );
}

/// Rescoring keeps each sample's query, which keyed labels match on, so a
/// keyed label reaches the same candidates in the rescored material as in
/// the material it was written for, in either mode.
#[test]
fn keyed_labels_reach_the_same_candidates_after_a_rescore() {
    let dir = TestDir::new();
    let corpus = conversation_history(&dir);
    let (_, material) = material_of(&dir, &corpus, "message", &[]);
    let recall: Vec<Value> = material["recall"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|sample| {
            candidates(sample).iter().map(move |candidate| {
                json!({
                    "query": sample["query"],
                    "memory": candidate["memory"],
                    "relevant": candidate["sentence"] == hermes::HOME_SENTENCE
                })
            })
        })
        .collect();
    assert!(!recall.is_empty(), "the run shows candidates: {material}");
    let labels = dir.private_file(
        "labelling/keyed.toml",
        &toml::to_string(&json!({ "recall": recall })).unwrap(),
    );

    let input = dir.private_path("labelling/message.json");
    let output = precision(&dir, &labels, &input);
    assert_ok(&output);
    let before: Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_eq!(before["recall"]["unmatched"], 0, "{before}");
    assert_eq!(before["recall"]["unlabelled"], 0, "{before}");
    for mode in ["message", "conversation"] {
        let out = dir.private_path(&format!("labelling/rescored-{mode}.json"));
        assert_ok(&rescore(&dir, &input, &corpus, mode, &out));
        let output = precision(&dir, &labels, &out);
        assert_ok(&output);
        let after: Value = serde_json::from_str(&stdout(&output)).unwrap();
        for key in ["labelled", "unlabelled", "matched", "unmatched"] {
            assert_eq!(
                after["recall"][key], before["recall"][key],
                "{mode} {key}: {after}"
            );
        }
        assert_eq!(
            after["recall"]["top8"]["relevant"], before["recall"]["top8"]["relevant"],
            "{mode}: {after}"
        );
    }
}
