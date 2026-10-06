//! The HTML report, labelling material and precision curve contracts,
//! exercised on synthetic history only.
//!
//! - `asphodel report html R [--out FILE]` writes one page holding the
//!   report's numbers, every asset inlined. Without `--out`
//!   it goes beside the report, as `R` with the extension `html`.
//! - `asphodel replay... --labelling FILE` writes the labelling material
//!   with recall candidates at sampled turns, scored with the
//!   reranker logit the gate floor compares, and call 2's candidate lists,
//!   scored with the cosine similarity the reconcile floor compares.
//! - It also writes, for a sample of refreshes, each facet's whole pool
//!   before the facet budget cut it: every candidate's reranker logit, what
//!   took or cut it and the writer handle it reached, if any.
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
//! - `--refresh-queries Q`, a TOML table of facet heading to another
//!   query, scores each refresh sample of those facets again against its
//!   query, keeping the sample's `query` and every candidate with what the
//!   refresh recorded of it. Without it the refresh samples stay as they
//!   were.
//! - `asphodel replay ... --refresh-queries Q`, the same table, has every
//!   refresh facet with a heading in it retrieve and rerank with that
//!   query instead of its own, recording it as the sample's `rerank_query`
//!   and keeping its `query` as the label key. The report holds the file's
//!   SHA-256 as `refresh_queries_hash`, null without it.
//!
//! Everything these read or write is derived from history, so all of it
//! stays under the private dir and errors never quote it.

mod support;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Output;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use support::hermes::{self, StateDb, start};
use support::{
    HOME_QUESTION, PASSING_PROBES, TestDir, asphodel, assert_ok, assert_refused,
    assert_refused_without, claim, home_claim, import_history, import_with, imported_small_history,
    model_manifest, overrides, probe_in, read_json, record, replay_history, replay_history_to,
    script_answering_everything, simulation, stderr, stdout,
};

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
    for (pointer, value) in [
        ("/injected_tokens/per_turn/p50", json!(71001)),
        ("/injected_tokens/per_turn/p95", json!(71002)),
        ("/injected_tokens/cron/tokens", json!(71003)),
        ("/injected_tokens/sessions/0/tokens", json!(71004)),
        ("/profile_tokens/p95", json!(71005)),
        ("/extraction_lag/p95_ms", json!(71006)),
        ("/call2_rate/call2", json!(71007)),
        ("/llm/cache", json!(71008)),
        ("/llm/live", json!(71009)),
        ("/llm/misses", json!(71010)),
        ("/llm/used_verdicts/top_up", json!(71011)),
        ("/purged_then_re_mentioned/purged", json!(71012)),
        ("/kind_histogram", json!({ "fact": 71013 })),
        ("/significance_histogram", json!({ "notable": 71014 })),
        (
            "/bands_per_week",
            json!([{ "week": "2026-W02", "strong": 71015, "fading": 71016, "faded": 71017 }]),
        ),
        (
            "/fade_outs_per_week",
            json!([{ "week": "2026-W03", "count": 71018 }]),
        ),
        (
            "/purges_per_day",
            json!([{ "day": "2026-01-09", "count": 71019, "purged": 71019 }]),
        ),
        (
            "/refresh_calls_per_day",
            json!([{ "day": "2026-01-10", "count": 71020 }]),
        ),
        (
            "/agenda_lines_per_day",
            json!([{ "day": "2026-01-11", "count": 71021 }]),
        ),
    ] {
        *report
            .pointer_mut(pointer)
            .unwrap_or_else(|| panic!("the report has {pointer}")) = value;
    }
    (dir.private_json("reports/patched.json", &report), report)
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
    let link = |target: &Path, name: &str| {
        let link = dir.private().join(name);
        std::os::unix::fs::symlink(target, &link).unwrap();
        link
    };

    let outside = dir.path("outside.json");
    fs::copy(&recorded, &outside).unwrap();
    assert_refused(&report_html(&dir, &outside, &[]), "private");
    assert!(!outside.with_extension("html").exists());

    let linked = link(&outside, "reports/linked.json");
    assert_refused(&report_html(&dir, &linked, &[]), "private");
    assert!(!linked.with_extension("html").exists());
    assert!(!outside.with_extension("html").exists());

    let page = dir.path("page.html");
    let linked_page = link(&dir.path("target.html"), "reports/page.html");
    for out in [&page, &linked_page] {
        let output = report_html(&dir, &recorded, &["--out", out.to_str().unwrap()]);
        assert_refused(&output, "private");
    }
    assert!(!page.exists() && !dir.path("target.html").exists());
}

// The labelling material.

/// More synced turns than the material samples, one session each so
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
    import_history(dir, |path| {
        let db = StateDb::create(path);
        for turn in 0..TURNS {
            let session = format!("s{turn:02}");
            let at = start() + 3600.0 * turn as f64;
            db.owner_session(&session, at);
            let user = match turn {
                0 => format!("{}, near the harbour.", hermes::HOME_QUOTE),
                1 => format!("{} still.", hermes::HOME_QUOTE),
                n if n % 2 == 0 => format!("[Sam] {}", labelling_query(n)),
                n => format!("{DISCORD_NOTE}\n\n[Sam] {}", labelling_query(n)),
            };
            db.turn(&session, at, &user, "Noted.");
        }
        db
    })
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

/// A stand-in whose every step answers any call: call 1 claims the home
/// sentence (kept only where the turn quotes it), call 2 labels it
/// `mentioned_again` on the first neighbour, and a refresh writes one
/// sentence citing `m1`.
fn labelling_script(dir: &TestDir) -> PathBuf {
    let mut home = home_claim();
    home["claim"] = json!("c1");
    home["labels"] = json!([{ "neighbour": "n1", "label": "mentioned_again" }]);
    script_answering_everything(dir, "labelling-script", vec![home], vec![])
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

/// Every candidate of every sample in `list`.
fn all_candidates<'a>(material: &'a Value, list: &str) -> impl Iterator<Item = &'a Value> {
    material[list]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(candidates)
}

/// A `replay` of the recorded cassette writing its material to
/// `labelling/<name>.json`, with `extra` flags; returns the run and the
/// material's path.
fn labelled_replay(
    dir: &TestDir,
    corpus: &Path,
    name: &str,
    extra: &[&str],
) -> (support::Run, PathBuf) {
    let path = dir.private_path(&format!("labelling/{name}.json"));
    let mut flags = vec!["--labelling", path.to_str().unwrap()];
    flags.extend(extra);
    let run = replay_history(dir, corpus, "replay", PASSING_PROBES, None, &flags);
    assert_ok(&run.output);
    (run, path)
}

/// Replay's `--labelling` writes recall candidates at a sample of the turns
/// and call 2's candidate lists, with ids, scores and sentences. Recall
/// candidates are scored with the reranker logit the gate floor compares,
/// including those the gate turned away, since a floor can't be calibrated
/// from what it already let through; call 2's with the similarity the
/// reconcile floor compares. Each sample shows the raw query and the
/// cleaned one the reranker scored. Writing it doesn't change the run, the
/// same run writes the same bytes, and `report precision` reads it.
#[test]
fn labelling_material_holds_scored_candidates_at_sampled_turns_and_call2_lists() {
    let dir = TestDir::new();
    let corpus = labelling_history(&dir);
    let script = labelling_script(&dir);
    replay_history(&dir, &corpus, "live", PASSING_PROBES, Some(&script), &[]).ok();
    let plain = replay_history(&dir, &corpus, "replay", PASSING_PROBES, None, &[]).ok();

    let (labelled, material_path) = labelled_replay(&dir, &corpus, "material", &[]);
    let report = labelled.report();
    assert_eq!(
        simulation(&plain),
        simulation(&report),
        "writing the material changes nothing the run simulated"
    );
    let bytes = fs::read(&material_path).expect("the material is written");
    let material: Value = serde_json::from_slice(&bytes).expect("the material is JSON");
    let (_, again) = labelled_replay(&dir, &corpus, "again", &[]);
    assert_eq!(
        fs::read(again).unwrap(),
        bytes,
        "the same run samples the same turns"
    );

    let first_memory = report["memories"][0]["id"]
        .as_str()
        .expect("the history makes a memory");
    let mut ids = BTreeSet::new();

    let recall = material["recall"].as_array().expect("a recall list");
    assert!(recall.len() < TURNS, "the turns are sampled");
    let turns: BTreeSet<(&str, &str)> = recall
        .iter()
        .map(|sample| {
            let session = sample["session"].as_str().expect("a sample's session");
            (session, sample["at"].as_str().expect("a sample's time"))
        })
        .collect();
    assert_eq!(turns.len(), recall.len(), "each sample is a different turn");
    let (mut below_floor, mut on_home, mut noted) = (0, 0, 0);
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
            let sentence = candidate["sentence"].as_str().unwrap();
            let score = candidate["score"].as_f64().unwrap();
            assert_eq!(
                score,
                fake_logit(query, sentence),
                "a recall candidate's score is the reranker logit: {candidate} for {query:?}"
            );
            below_floor += usize::from(score < 0.0);
            if candidate["memory"] == first_memory {
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
    for candidate in all_candidates(&material, "recall").chain(all_candidates(&material, "call2")) {
        let id = candidate["id"].as_str().unwrap().to_string();
        assert!(ids.insert(id), "{candidate}");
    }
    let neighbour = candidates(home)
        .iter()
        .find(|candidate| candidate["memory"] == first_memory)
        .expect("the claim's call 2 list holds the memory it restates");
    assert_eq!(neighbour["sentence"], hermes::HOME_SENTENCE);
    let similarity = neighbour["score"].as_f64().unwrap();
    assert!(
        (similarity - 1.0).abs() < 1e-6,
        "an identical sentence's similarity is 1: {neighbour}"
    );

    // Label every candidate, relevant when it's the home memory, and the
    // curve comes out of the material as written.
    let labels: String = all_candidates(&material, "recall")
        .chain(all_candidates(&material, "call2"))
        .map(|candidate| {
            format!(
                "{} = {}\n",
                candidate["id"],
                candidate["memory"] == first_memory
            )
        })
        .collect();
    let labels_path = dir.private_file("labelling/labels.toml", &labels);
    let curve = curve(&dir, &labels_path, &material_path);
    assert_eq!(curve["recall"]["unlabelled"], 0, "{curve}");
    assert!(
        !curve["recall"]["curve"].as_array().unwrap().is_empty(),
        "{curve}"
    );
}

// Refresh calibration material.

/// What Tim says, one turn each, and the memory each makes. The first four
/// share "family" with the profile's People facet; the rest share no word
/// with any profile facet's query, so the fake reranker scores them below
/// zero under every facet.
const REFRESH_FACTS: [(&str, &str); 12] = [
    (
        "I cook pasta for family every Friday",
        "Tim cooks pasta for family every Friday.",
    ),
    (
        "I drive family north each summer",
        "Tim drives family north each summer.",
    ),
    (
        "I keep a family recipe book",
        "Tim keeps a family recipe book.",
    ),
    (
        "I play chess with family on Sundays",
        "Tim plays chess with family on Sundays.",
    ),
    (
        "I measured a fridge at sixty centimetres",
        "Tim measured a fridge at sixty centimetres.",
    ),
    (
        "I set a webhook timeout at thirty seconds",
        "Tim set a webhook timeout at thirty seconds.",
    ),
    ("I own a blue kayak", "Tim owns a blue kayak."),
    ("I bought a cordless drill", "Tim bought a cordless drill."),
    (
        "I ran a marathon last October",
        "Tim ran a marathon last October.",
    ),
    (
        "I grow tomatoes on a balcony",
        "Tim grows tomatoes on a balcony.",
    ),
    ("I read maps for fun", "Tim reads maps for fun."),
    ("I collect old stamps", "Tim collects old stamps."),
];

/// One session per fact, an hour apart.
fn refresh_history(dir: &TestDir) -> PathBuf {
    refresh_history_of(dir, &REFRESH_FACTS, hermes::MANIFEST)
}

/// One session per fact of `facts`, an hour apart, imported with
/// `manifest`.
fn refresh_history_of(dir: &TestDir, facts: &[(&str, &str)], manifest: &str) -> PathBuf {
    let state_db = dir.private_path("state.db");
    let db = StateDb::create(&state_db);
    for (turn, (quote, _)) in facts.iter().enumerate() {
        let session = format!("s{turn:02}");
        let at = start() + 3600.0 * turn as f64;
        db.owner_session(&session, at);
        db.turn(&session, at, &format!("By the way, {quote}."), "Noted.");
    }
    drop(db);
    let corpus = dir.private_path("corpus/main.jsonl");
    assert_ok(&import_with(dir, &state_db, &corpus, manifest, &[]));
    corpus
}

/// Every call 1 claims every fact, kept only in the turn that quotes it,
/// and call 2 labels nothing.
fn refresh_script(dir: &TestDir) -> PathBuf {
    let claims = REFRESH_FACTS
        .iter()
        .map(|(quote, sentence)| {
            let mut claim = claim(sentence, quote, "fact");
            claim["claim"] = json!("c1");
            claim["labels"] = json!([]);
            claim
        })
        .collect();
    script_answering_everything(dir, "refresh-script", claims, vec![])
}

/// A probe the facts pass, since the home memory isn't among them.
const REFRESH_PROBES: &str = r#"
[[probe]]
id = "p001"
at = "2026-01-06T12:00:00Z"
kind = "exists"
memory = "family recipe book"
"#;

/// A `replay` of the recorded cassette with `flags`, writing its material
/// to `labelling/<name>.json` when `name` is given; returns the report and
/// the material's path.
fn refresh_replay(
    dir: &TestDir,
    corpus: &Path,
    name: Option<&str>,
    flags: &[&str],
) -> (Value, PathBuf) {
    let path = dir.private_path(&format!("labelling/{}.json", name.unwrap_or("none")));
    let mut flags = flags.to_vec();
    if name.is_some() {
        flags.extend(["--labelling", path.to_str().unwrap()]);
    }
    let run = replay_history(dir, corpus, "replay", REFRESH_PROBES, None, &flags);
    (run.ok(), path)
}

/// The refresh samples of `material`, by refresh: its time and model.
fn refreshes(material: &Value) -> BTreeMap<(String, String), Vec<&Value>> {
    let mut by_refresh = BTreeMap::new();
    for sample in material["refresh"]
        .as_array()
        .expect("a refresh list beside recall and call 2")
    {
        let at = sample["at"].as_str().expect("a refresh sample's time");
        let model = sample["model"].as_str().expect("a refresh sample's model");
        by_refresh
            .entry((at.to_string(), model.to_string()))
            .or_insert_with(Vec::new)
            .push(sample);
    }
    by_refresh
}

/// `--labelling` also writes, for a sample of refreshes, every facet's whole
/// pool before it was cut to `facet_budget`: each candidate's raw reranker
/// logit against the query the facet scored, whether the facet's budget took
/// it or cut it, and the handle it reached the writer under. So a memory the
/// writer never saw is still in the material, scored, which is what a
/// refresh floor is calibrated from. Writing it doesn't change the run, the
/// same run writes the same bytes, and `report precision` reads facet labels
/// keyed by the facet's query and the memory.
#[test]
fn labelling_material_holds_each_refresh_facets_whole_scored_pool() {
    let dir = TestDir::new();
    let corpus = refresh_history(&dir);
    let script = refresh_script(&dir);
    let budget = overrides(&dir, "[mental_models]\nfacet_budget = 2\n");
    let tuned = ["--overrides", budget.as_str()];
    replay_history(&dir, &corpus, "live", REFRESH_PROBES, Some(&script), &tuned).ok();
    let (plain, _) = refresh_replay(&dir, &corpus, None, &tuned);

    let (labelled, material_path) = refresh_replay(&dir, &corpus, Some("material"), &tuned);
    assert_eq!(
        simulation(&plain),
        simulation(&labelled),
        "writing the material changes nothing the run simulated"
    );
    let bytes = fs::read(&material_path).expect("the material is written");
    let material: Value = serde_json::from_slice(&bytes).expect("the material is JSON");
    let (_, again) = refresh_replay(&dir, &corpus, Some("again"), &tuned);
    assert_eq!(
        fs::read(again).unwrap(),
        bytes,
        "the same run samples the same refreshes"
    );

    let by_refresh = refreshes(&material);
    assert!(!by_refresh.is_empty(), "a refresh is sampled: {material}");
    let mut excluded = None;
    let mut below_zero = 0;
    for samples in by_refresh.values() {
        let mut indexes: Vec<u64> = samples
            .iter()
            .map(|sample| sample["facet_index"].as_u64().expect("a facet index"))
            .collect();
        indexes.sort_unstable();
        let every_facet: Vec<u64> = (0..samples.len() as u64).collect();
        assert_eq!(
            indexes, every_facet,
            "every facet of a refresh: {samples:?}"
        );

        // A handle names one memory in a refresh, and a memory found under
        // several facets keeps its handle.
        let mut memory_of = BTreeMap::new();
        let mut handle_of = BTreeMap::new();
        for sample in samples {
            assert!(sample["facet"].is_string(), "{sample}");
            assert_eq!(
                sample["reranked"], true,
                "the fake reranker answers: {sample}"
            );
            let rerank_query = sample["rerank_query"].as_str().expect("a rerank query");
            let pool = candidates(sample);
            let taken = pool.iter().filter(|c| c["taken"] == "budget").count();
            assert!(taken <= 2, "the facet budget is 2: {sample}");
            for candidate in pool {
                let sentence = candidate["sentence"].as_str().unwrap();
                let logit = candidate["logit"]
                    .as_f64()
                    .unwrap_or_else(|| panic!("a reranked candidate's logit: {candidate}"));
                assert_eq!(logit, fake_logit(rerank_query, sentence), "{candidate}");
                below_zero += usize::from(logit < 0.0);
                let taken = candidate["taken"]
                    .as_str()
                    .expect("what took the candidate");
                assert!(["budget", "cited", "cut"].contains(&taken), "{candidate}");
                let memory = candidate["memory"].as_str().unwrap();
                match candidate["input"].as_str() {
                    Some(handle) => {
                        let named = memory_of.entry(handle.to_string()).or_insert(memory);
                        assert_eq!(*named, memory, "{handle} names one memory: {samples:?}");
                        let kept = handle_of.entry(memory.to_string()).or_insert(handle);
                        assert_eq!(*kept, handle, "{memory} keeps its handle: {samples:?}");
                    }
                    None if taken == "cut" => {
                        excluded.get_or_insert((sample["query"].clone(), memory, logit));
                    }
                    None => {}
                }
            }
        }
    }
    assert!(
        below_zero > 0,
        "memories sharing nothing with a facet are scored"
    );
    let (query, memory, logit) =
        excluded.expect("a cut candidate the writer never saw is in the material");

    let label = json!({ "query": query, "memory": memory, "relevant": true });
    let labels = dir.private_file(
        "labelling/facets.toml",
        &toml::to_string(&json!({ "facet": [label] })).unwrap(),
    );
    let curve = curve(&dir, &labels, &material_path);
    assert_eq!(curve["refresh"]["matched"], 1, "{curve}");
    let points = curve["refresh"]["curve"]
        .as_array()
        .unwrap_or_else(|| panic!("a refresh curve: {curve}"));
    assert_eq!(points.len(), 1, "{curve}");
    assert_eq!(points[0]["floor"].as_f64(), Some(logit), "{curve}");
    assert_eq!(points[0]["precision"].as_f64(), Some(1.0), "{curve}");
}

/// The labelled material of a `live` run on `facts` with `manifest`, and
/// the corpus.
fn refresh_material(dir: &TestDir, facts: &[(&str, &str)], manifest: &str) -> (Value, PathBuf) {
    let corpus = refresh_history_of(dir, facts, manifest);
    let script = refresh_script(dir);
    replay_history(dir, &corpus, "live", REFRESH_PROBES, Some(&script), &[]).ok();
    let (_, path) = refresh_replay(dir, &corpus, Some("material"), &[]);
    (read_json(&path), corpus)
}

/// The models of `material`'s sampled refreshes, one per refresh.
fn sampled_models(material: &Value) -> Vec<String> {
    refreshes(material)
        .into_keys()
        .map(|(_, model)| model)
        .collect()
}

/// The refresh material samples every refresh when there are 10 or fewer,
/// the other models' as well as the seeded profile's. When there are more,
/// it takes 10, the profile's first: with a profile that refreshes more
/// than 10 times, the other model that refreshes beside it isn't sampled.
/// `report rescore` leaves the refresh samples as they were.
#[test]
fn refresh_samples_take_ten_refreshes_the_profiles_first() {
    let manifest = model_manifest(HOME_QUESTION);

    let dir = TestDir::new();
    let (material, corpus) = refresh_material(&dir, &REFRESH_FACTS[..3], &manifest);
    let few = sampled_models(&material);
    assert!(few.len() <= 10, "{few:?}");
    let models: BTreeSet<&str> = few.iter().map(String::as_str).collect();
    assert_eq!(models.len(), 2, "both models are sampled: {few:?}");
    assert!(models.contains("home"), "{few:?}");

    let input = dir.private_path("labelling/material.json");
    let out = dir.private_path("labelling/rescored.json");
    assert_ok(&rescore(&dir, &input, &corpus, "message", &out));
    assert_eq!(read_json(&out)["refresh"], material["refresh"]);

    let dir = TestDir::new();
    let (material, _) = refresh_material(&dir, &REFRESH_FACTS, &manifest);
    let many = sampled_models(&material);
    assert_eq!(many.len(), 10, "{many:?}");
    assert!(!many.iter().any(|model| model == "home"), "{many:?}");
}

/// A refresh candidate with its logit left out: what the refresh recorded
/// of it, which a rescore keeps.
fn recorded(candidate: &Value) -> Value {
    let mut candidate = candidate.clone();
    candidate.as_object_mut().unwrap().remove("logit");
    candidate
}

/// `report rescore --refresh-queries` scores the fixed refresh pools again
/// against another query for a facet, named by its heading: every sample of
/// that facet records the query as `rerank_query`, and each candidate's
/// logit is the fake reranker's for it. The sample keeps its `query`, so
/// facet labels match the same candidates, and keeps every candidate with
/// what the refresh recorded of it: its strength, score, rank, citation,
/// what took it and its handle. Samples of other facets stay as they were,
/// and the same rescore writes the same bytes.
#[test]
fn refresh_queries_rescore_the_fixed_pools_keeping_label_keys_and_selection() {
    let dir = TestDir::new();
    let (material, corpus) = refresh_material(&dir, &REFRESH_FACTS[..3], hermes::MANIFEST);
    let samples = material["refresh"].as_array().expect("a refresh list");
    let heading = samples.first().expect("a refresh is sampled")["facet"]
        .as_str()
        .unwrap()
        .to_string();
    let alternative = "Tim pasta family";
    let queries = toml::to_string(&json!({ heading.as_str(): alternative })).unwrap();
    let queries = dir.private_file("labelling/refresh-queries.toml", &queries);
    let flags = ["--refresh-queries", queries.to_str().unwrap()];

    let input = dir.private_path("labelling/material.json");
    let out = dir.private_path("labelling/rescored.json");
    assert_ok(&rescore_with(
        &dir, &input, &corpus, "message", &out, &flags,
    ));
    let rescored = read_json(&out);
    let after = rescored["refresh"].as_array().expect("a refresh list");
    assert_eq!(after.len(), samples.len());

    let mut changed = 0;
    for (before, after) in samples.iter().zip(after) {
        if before["facet"] != heading.as_str() {
            assert_eq!(after, before, "another facet's sample is as it was");
            continue;
        }
        for key in [
            "sample",
            "at",
            "model",
            "facet",
            "facet_index",
            "query",
            "reranked",
        ] {
            assert_eq!(after[key], before[key], "{key} is kept: {after}");
        }
        assert_eq!(after["rerank_query"], alternative, "{after}");
        let by_id = |sample: &Value| -> BTreeMap<String, Value> {
            candidates(sample)
                .iter()
                .map(|c| (c["id"].as_str().unwrap().to_string(), c.clone()))
                .collect()
        };
        let (was, now) = (by_id(before), by_id(after));
        assert_eq!(
            was.keys().collect::<Vec<_>>(),
            now.keys().collect::<Vec<_>>(),
            "the same candidates: {after}"
        );
        for (id, candidate) in &now {
            assert_eq!(recorded(candidate), recorded(&was[id]), "{candidate}");
            let sentence = candidate["sentence"].as_str().unwrap();
            let logit = candidate["logit"].as_f64().expect("a rescored logit");
            assert_eq!(logit, fake_logit(alternative, sentence), "{candidate}");
            changed += usize::from(Some(logit) != was[id]["logit"].as_f64());
        }
    }
    assert!(changed > 0, "the other query moves some logit");

    let again = dir.private_path("labelling/rescored-again.json");
    assert_ok(&rescore_with(
        &dir, &input, &corpus, "message", &again, &flags,
    ));
    assert_eq!(fs::read(&again).unwrap(), fs::read(&out).unwrap());

    // Facet labels keyed by the stored query reach the same candidates.
    let keys: BTreeSet<(&str, &str)> = samples
        .iter()
        .flat_map(|sample| {
            let query = sample["query"].as_str().unwrap();
            candidates(sample)
                .iter()
                .map(move |candidate| (query, candidate["memory"].as_str().unwrap()))
        })
        .collect();
    let labels: Vec<Value> = keys
        .into_iter()
        .map(|(query, memory)| json!({ "query": query, "memory": memory, "relevant": true }))
        .collect();
    let labels = dir.private_file(
        "labelling/facets.toml",
        &toml::to_string(&json!({ "facet": labels })).unwrap(),
    );
    let (before, after) = (curve(&dir, &labels, &input), curve(&dir, &labels, &out));
    for key in ["labelled", "unlabelled", "matched", "unmatched"] {
        assert_eq!(after["refresh"][key], before["refresh"][key], "{key}");
    }
    assert_eq!(before["refresh"]["unlabelled"], 0, "{before}");
}

// Refresh queries in a replay.

/// More memories than a retrieval arm returns (100), every one sharing
/// several words with one profile facet's query, so they fill that facet's
/// pool and [`TARGET`], which shares none, is left out of it.
const FILLERS: usize = 110;

/// The memory a mapped query pulls into the pool: Tim's quote and its
/// sentence.
const TARGET: (&str, &str) = ("I own a blue kayak", "Tim owns a blue kayak.");

/// The query the facet is mapped to, sharing words with [`TARGET`] and
/// none with a filler.
const MAPPED_QUERY: &str = "blue kayak";

/// Filler `n`: Tim's quote and its sentence.
fn filler(n: usize) -> (String, String) {
    (
        format!("family friends and colleagues include Zed{n:03}"),
        format!("Tim's family friends and colleagues include Zed{n:03}."),
    )
}

/// Ten fillers a turn, one session each, an hour apart. Insert the target
/// before the last filler turn so it exists during a sampled refresh.
fn pool_history(dir: &TestDir) -> PathBuf {
    import_history(dir, |path| {
        let db = StateDb::create(path);
        let fillers: Vec<(String, String)> = (0..FILLERS).map(filler).collect();
        let mut turns: Vec<String> = fillers
            .chunks(10)
            .map(|chunk| {
                let quotes: Vec<&str> = chunk.iter().map(|(quote, _)| quote.as_str()).collect();
                format!("By the way, {}.", quotes.join(", and "))
            })
            .collect();
        turns.insert(turns.len() - 1, format!("By the way, {}.", TARGET.0));
        for (turn, user) in turns.iter().enumerate() {
            let session = format!("s{turn:02}");
            let at = start() + 3600.0 * turn as f64;
            db.owner_session(&session, at);
            db.turn(&session, at, user, "Noted.");
        }
        db
    })
}

/// Every call 1 claims every filler and the target, each kept only in the
/// turn that quotes it, and call 2 labels nothing.
fn pool_script(dir: &TestDir) -> PathBuf {
    let mut facts: Vec<(String, String)> = (0..FILLERS).map(filler).collect();
    facts.push((TARGET.0.to_string(), TARGET.1.to_string()));
    let claims = facts
        .iter()
        .map(|(quote, sentence)| {
            let mut claim = claim(sentence, quote, "fact");
            claim["claim"] = json!("c1");
            claim["labels"] = json!([]);
            claim
        })
        .collect();
    script_answering_everything(dir, "pool-script", claims, vec![])
}

/// A probe the target passes, which names it.
const POOL_PROBES: &str = r#"
[[probe]]
id = "p001"
at = "2026-01-06T12:00:00Z"
kind = "exists"
memory = "blue kayak"
"#;

/// A `fast` run with every refresh write skipped, so it calls no LLM,
/// writing its material to `labelling/<name>.json` and its report to
/// `reports/<name>.json`, with `extra` flags.
fn pool_run(dir: &TestDir, corpus: &Path, name: &str, extra: &[&str]) -> support::Run {
    let material = dir.private_path(&format!("labelling/{name}.json"));
    let mut flags = vec![
        "--refresh",
        "off",
        "--labelling",
        material.to_str().unwrap(),
    ];
    flags.extend(extra);
    replay_history_to(dir, corpus, "fast", POOL_PROBES, name, None, &flags)
}

/// A refresh sample's candidates as retrieval and reranking found them,
/// leaving out what the selection across facets made of them: what took
/// each, its handle and its citation can change when another facet's pool
/// does.
fn retrieved(sample: &Value) -> Vec<Value> {
    candidates(sample)
        .iter()
        .map(|candidate| {
            let keys = ["memory", "sentence", "logit", "strength", "score", "rank"];
            keys.iter()
                .map(|key| (key.to_string(), candidate[key].clone()))
                .collect::<serde_json::Map<_, _>>()
                .into()
        })
        .collect()
}

/// `material`'s refresh samples by refresh and facet.
fn by_facet(material: &Value) -> BTreeMap<(String, String, u64), &Value> {
    material["refresh"]
        .as_array()
        .expect("a refresh list")
        .iter()
        .map(|sample| {
            let key = (
                sample["at"].as_str().unwrap().to_string(),
                sample["model"].as_str().unwrap().to_string(),
                sample["facet_index"].as_u64().unwrap(),
            );
            (key, sample)
        })
        .collect()
}

/// Whether `memory` is in `sample`'s pool.
fn pools(sample: &Value, memory: &str) -> bool {
    candidates(sample).iter().any(|c| c["memory"] == memory)
}

/// `replay --refresh-queries` retrieves with the mapped query, not only
/// reranks with it: the facet whose own query leaves the target out of its
/// pool (the fillers fill it) finds the target under a query that names it.
/// Its samples record that query as `rerank_query`, scored with it, and
/// keep their `query`, so a facet label keyed by it reaches the target in
/// this run's material and finds nothing in the run without the option.
/// The other facets retrieve and rerank as they did. The report holds the
/// file's SHA-256; a run without the option holds none and writes the same
/// material as before.
#[test]
fn refresh_queries_in_a_replay_change_the_retrieved_pool_keeping_label_keys() {
    let dir = TestDir::new();
    let corpus = pool_history(&dir);
    let script = pool_script(&dir);
    replay_history(&dir, &corpus, "live", POOL_PROBES, Some(&script), &[]).ok();

    let base_run = pool_run(&dir, &corpus, "base", &[]);
    let base_report = base_run.ok();
    assert_eq!(base_report["refresh_queries_hash"], Value::Null);
    let base_path = dir.private_path("labelling/base.json");
    let base = read_json(&base_path);
    pool_run(&dir, &corpus, "again", &[]).ok();
    assert_eq!(
        fs::read(dir.private_path("labelling/again.json")).unwrap(),
        fs::read(&base_path).unwrap(),
        "without the option the material is the same"
    );

    let target = probe_in(&base_report, "p001")["observed"]["id"]
        .as_str()
        .expect("the probe finds the target")
        .to_string();
    let base_samples = by_facet(&base);
    let heading = base_samples
        .values()
        .map(|sample| sample["facet"].as_str().unwrap())
        .find(|heading| {
            base_samples
                .values()
                .filter(|sample| sample["facet"] == *heading)
                .all(|sample| !pools(sample, &target))
        })
        .expect("a facet whose own query leaves the target out of its pool")
        .to_string();

    let text = toml::to_string(&json!({ heading.as_str(): MAPPED_QUERY })).unwrap();
    let queries = dir.private_file("refresh-queries.toml", &text);
    let flags = ["--refresh-queries", queries.to_str().unwrap()];
    let mapped_report = pool_run(&dir, &corpus, "mapped", &flags).ok();
    let hash = format!("{:x}", Sha256::digest(text.as_bytes()));
    assert_eq!(mapped_report["refresh_queries_hash"], hash.as_str());
    let mapped = read_json(&dir.private_path("labelling/mapped.json"));
    let mapped_samples = by_facet(&mapped);
    assert_eq!(
        mapped_samples.keys().collect::<Vec<_>>(),
        base_samples.keys().collect::<Vec<_>>(),
        "the same refreshes and facets are sampled"
    );

    let mut found = 0;
    for (key, after) in &mapped_samples {
        let before = base_samples[key];
        for field in ["facet", "query", "reranked"] {
            assert_eq!(after[field], before[field], "{field} is kept: {after}");
        }
        if after["facet"] != heading.as_str() {
            assert_eq!(after["rerank_query"], before["rerank_query"], "{after}");
            assert_eq!(retrieved(after), retrieved(before), "{after}");
            continue;
        }
        assert_eq!(after["rerank_query"], MAPPED_QUERY, "{after}");
        for candidate in candidates(after) {
            let sentence = candidate["sentence"].as_str().unwrap();
            let logit = candidate["logit"].as_f64().expect("a logit");
            assert_eq!(logit, fake_logit(MAPPED_QUERY, sentence), "{candidate}");
        }
        if pools(after, &target) {
            assert!(
                !pools(before, &target),
                "the base pool excludes the target at the same sampled refresh: {key:?}"
            );
            found += 1;
        }
    }
    assert!(
        found > 0,
        "the mapped query brings the target into the pool"
    );

    // A facet label keyed by the facet's own query reaches the target only
    // where the mapped query retrieved it.
    let query = mapped_samples
        .values()
        .find(|sample| sample["facet"] == heading.as_str())
        .unwrap()["query"]
        .clone();
    let label = json!({ "query": query, "memory": target, "relevant": true });
    let labels = dir.private_file(
        "labelling/facets.toml",
        &toml::to_string(&json!({ "facet": [label] })).unwrap(),
    );
    let mapped_path = dir.private_path("labelling/mapped.json");
    assert_eq!(curve(&dir, &labels, &mapped_path)["refresh"]["matched"], 1);
    assert_eq!(curve(&dir, &labels, &base_path)["refresh"]["unmatched"], 1);
}

/// The refresh queries file is history: one outside the private dir is
/// refused, and so is one naming a heading no refresh facet has or giving
/// a blank query, each before the run, naming the file and quoting
/// nothing of it, with no report or material written.
#[test]
fn refresh_queries_in_a_replay_are_refused_before_the_run() {
    let dir = TestDir::new();
    let corpus = refresh_history(&dir);
    let script = refresh_script(&dir);
    replay_history(&dir, &corpus, "live", REFRESH_PROBES, Some(&script), &[]).ok();
    let (_, material) = refresh_replay(&dir, &corpus, Some("material"), &[]);
    let heading = read_json(&material)["refresh"][0]["facet"]
        .as_str()
        .expect("a refresh is sampled")
        .to_string();

    let sentinel = "SENTINEL-REFRESH-QUERY-6d0e";
    let cases = [
        (
            "unknown",
            format!("\"{sentinel} heading\" = \"{sentinel}\"\n"),
        ),
        (
            "blank",
            toml::to_string(&json!({ heading.as_str(): "   " })).unwrap(),
        ),
    ];
    for (name, text) in cases {
        let file = format!("refresh-queries-{name}.toml");
        let queries = dir.private_file(&file, &text);
        let path = dir.private_path(&format!("labelling/{name}.json"));
        let flags = [
            "--refresh-queries",
            queries.to_str().unwrap(),
            "--labelling",
            path.to_str().unwrap(),
        ];
        let run = replay_history_to(&dir, &corpus, "fast", REFRESH_PROBES, name, None, &flags);
        assert_refused_without(&run.output, sentinel, &[&file]);
        assert!(!run.report_path.exists(), "{name}: no report");
        assert!(!path.exists(), "{name}: no material");
    }

    let outside = dir.file("refresh-queries.toml", "\"People\" = \"Who?\"\n");
    let flags = ["--refresh-queries", outside.to_str().unwrap()];
    let run = replay_history_to(
        &dir,
        &corpus,
        "fast",
        REFRESH_PROBES,
        "outside",
        None,
        &flags,
    );
    assert_refused(&run.output, "private");
    assert!(!run.report_path.exists());
}

// Refresh curves, worked by hand.

/// A refresh candidate in the hand-written material.
fn refresh_candidate(
    n: u32,
    logit: Option<f64>,
    taken: &str,
    cited: bool,
    input: Option<&str>,
) -> Value {
    let sentence = format!("{MATERIAL_SENTINEL} sentence {n}");
    json!({
        "memory": memory(n), "sentence": sentence, "logit": logit, "strength": 0.5,
        "score": 0.5, "cited": cited, "taken": taken, "input": input
    })
}

/// A refresh sample of the profile at 04:00 on `day` of January 2026, for
/// the facet asking `name`, its candidates numbered and ranked in order.
fn refresh_sample(
    sample: &str,
    day: u8,
    facet_index: u32,
    name: &str,
    candidates: Vec<Value>,
) -> Value {
    let reranked = candidates.iter().any(|c| !c["logit"].is_null());
    let candidates: Vec<Value> = candidates
        .into_iter()
        .enumerate()
        .map(|(index, mut candidate)| {
            candidate["id"] = json!(format!("{sample}.{}", index + 1));
            candidate["rank"] = json!(index + 1);
            candidate
        })
        .collect();
    json!({
        "sample": sample,
        "at": format!("2026-01-{day:02}T04:00:00Z"),
        "model": "User profile",
        "facet": format!("{MATERIAL_SENTINEL} heading {name}"),
        "facet_index": facet_index,
        "query": query(name),
        "rerank_query": query(name),
        "reranked": reranked,
        "candidates": candidates
    })
}

/// Two refreshes of two facets each, "people" and "home".
///
/// - Refresh 1: people shows memory 1 at 2.5 and 2 at 0.5, both in its
///   budget, and 3 at -1.5, cut; home shows 3 at 1.5 in its budget and 4 at
///   -0.5, taken by the cited fill. The model cites 2, 3 and 4.
/// - Refresh 2: the reranker missed people, which shows 1 and 5 with no
///   logit, both in its budget; home shows 4 at 0.5. The model cites 1.
fn hand_refresh_material() -> Value {
    let c = refresh_candidate;
    json!({
        "version": 3,
        "recall": [],
        "call2": [],
        "refresh": [
            refresh_sample("f1", 5, 0, "people", vec![
                c(1, Some(2.5), "budget", false, Some("m1")),
                c(2, Some(0.5), "budget", true, Some("m2")),
                c(3, Some(-1.5), "cut", true, Some("m3"))
            ]),
            refresh_sample("f2", 5, 1, "home", vec![
                c(3, Some(1.5), "budget", true, Some("m3")),
                c(4, Some(-0.5), "cited", true, Some("m4"))
            ]),
            refresh_sample("f3", 6, 0, "people", vec![
                c(1, None, "budget", true, Some("m1")),
                c(5, None, "budget", false, Some("m2"))
            ]),
            refresh_sample("f4", 6, 1, "home", vec![c(4, Some(0.5), "budget", false, Some("m3"))])
        ]
    })
}

/// Facet labels for [`hand_refresh_material`]: under people, 1 and 5 are
/// relevant and 2 and 3 aren't; under home, 3 is and 4 isn't. One more,
/// under a query no sample asks, judges nothing shown.
fn hand_facet_labels() -> String {
    let labels = [
        ("people", 1, true),
        ("people", 2, false),
        ("people", 3, false),
        ("people", 5, true),
        ("home", 3, true),
        ("home", 4, false),
        ("gone", 9, true),
    ]
    .map(|(name, n, relevant)| {
        json!({ "query": query(name), "memory": memory(n), "relevant": relevant })
    });
    toml::to_string(&json!({ "facet": labels })).unwrap()
}

/// `curve[list][index]` as a curve of its own, for the helpers.
fn nested(curve: &Value, list: &str, index: usize) -> Value {
    json!({ "nested": curve["refresh"][list][index] })
}

/// The refresh section, worked by hand from [`hand_refresh_material`] and
/// [`hand_facet_labels`]. A candidate with no logit is labelled but never
/// kept, never counts toward an input size and never retains a citation.
///
/// Pooled: 8 candidates labelled; 6 labels matched and "gone" unmatched.
/// Logits 2.5 T, 0.5 F, -1.5 F, 1.5 T, -0.5 F, 0.5 F, and two null T.
/// - floor -1.5: 6 kept, 2 relevant
/// - floor -0.5: 5 kept, 2 relevant, 0.4
/// - floor 0.5: 4 kept, 2 relevant, 0.5
/// - floor 1.5: 2 kept, 2 relevant, 1.0
/// - floor 2.5: 1 kept, 1 relevant, 1.0
///
/// People (first in f1, facet 0): 5 labelled, 4 matched, 0 unmatched;
/// 2.5 T, 0.5 F, -1.5 F and two null T: floors -1.5 (3, 1), 0.5 (2, 1) and
/// 2.5 (1, 1). Home (first in f2, facet 1): 3 labelled, 2 matched; 1.5 T,
/// -0.5 F, 0.5 F: floors -0.5 (3, 1), 0.5 (2, 1) and 1.5 (1, 1). The label
/// under "gone" belongs to no facet.
///
/// Inputs, at the pooled floors: f1's budget took 2.5 and 0.5, f2's 1.5
/// (the cited fill's -0.5 isn't its budget's), f3's two nulls, f4's 0.5.
///
/// Cited: refresh 1 cites 2 (best 0.5), 3 (best 1.5, under home though
/// people cut it) and 4 (-0.5); refresh 2 cites 1, with no logit. Of 4:
/// 3 retained at -1.5 and -0.5, 2 at 0.5, 1 at 1.5, none at 2.5.
#[test]
fn the_refresh_curves_match_a_hand_computed_example() {
    let dir = TestDir::new();
    let material = dir.private_json("labelling/refresh.json", &hand_refresh_material());
    let labels = dir.private_file("labelling/facets.toml", &hand_facet_labels());
    let curve = curve(&dir, &labels, &material);

    assert_counts(&curve, "refresh", [8, 0, 6, 1]);
    assert_points(
        &curve,
        "refresh",
        &[
            (-1.5, 6, 2, 2.0 / 6.0),
            (-0.5, 5, 2, 0.4),
            (0.5, 4, 2, 0.5),
            (1.5, 2, 2, 1.0),
            (2.5, 1, 1, 1.0),
        ],
    );

    let facets = curve["refresh"]["facets"].as_array().expect("facet curves");
    assert_eq!(facets.len(), 2, "{curve}");
    for (index, sample, facet_index) in [(0, "f1", 0), (1, "f2", 1)] {
        assert_eq!(facets[index]["sample"], sample, "{curve}");
        assert_eq!(facets[index]["facet_index"], facet_index, "{curve}");
    }
    let people = nested(&curve, "facets", 0);
    assert_counts(&people, "nested", [5, 0, 4, 0]);
    assert_points(
        &people,
        "nested",
        &[(-1.5, 3, 1, 1.0 / 3.0), (0.5, 2, 1, 0.5), (2.5, 1, 1, 1.0)],
    );
    let home = nested(&curve, "facets", 1);
    assert_counts(&home, "nested", [3, 0, 2, 0]);
    assert_points(
        &home,
        "nested",
        &[(-0.5, 3, 1, 1.0 / 3.0), (0.5, 2, 1, 0.5), (1.5, 1, 1, 1.0)],
    );

    let floors = [-1.5, -0.5, 0.5, 1.5, 2.5];
    let inputs = |sample: &str, budget: u64, sizes: [u64; 5]| {
        let sizes: Vec<Value> = floors
            .iter()
            .zip(sizes)
            .map(|(floor, size)| json!({ "floor": floor, "size": size }))
            .collect();
        json!({ "sample": sample, "budget": budget, "sizes": sizes })
    };
    assert_eq!(
        curve["refresh"]["inputs"],
        json!([
            inputs("f1", 2, [2, 2, 2, 1, 1]),
            inputs("f2", 1, [1, 1, 1, 1, 0]),
            inputs("f3", 2, [0, 0, 0, 0, 0]),
            inputs("f4", 1, [1, 1, 1, 0, 0])
        ]),
        "{curve}"
    );
    let cited: Vec<Value> = floors
        .iter()
        .zip([3, 3, 2, 1, 0])
        .map(|(floor, retained)| {
            let share = f64::from(retained) / 4.0;
            json!({ "floor": floor, "cited": 4, "retained": retained, "share": share })
        })
        .collect();
    assert_eq!(curve["refresh"]["cited"], json!(cited), "{curve}");
}

/// Input sizes and cited retention are read at the floors facet labels
/// give, so with no facet label matching anything there are none: each
/// sample still counts what its budget took. Facet labels are keyed only:
/// a refresh candidate's id in a file of candidate ids is refused, naming
/// the labels file and quoting nothing.
#[test]
fn refresh_floors_come_only_from_keyed_facet_labels() {
    let dir = TestDir::new();
    let material = dir.private_json("labelling/refresh.json", &hand_refresh_material());
    let label = json!({ "query": query("gone"), "memory": memory(9), "relevant": true });
    let unmatched = toml::to_string(&json!({ "facet": [label] })).unwrap();
    let labels = dir.private_file("labelling/unmatched.toml", &unmatched);
    let curve = curve(&dir, &labels, &material);
    assert_counts(&curve, "refresh", [0, 8, 0, 1]);
    assert_points(&curve, "refresh", &[]);
    let budgets: Vec<(Value, Value)> = curve["refresh"]["inputs"]
        .as_array()
        .expect("a sample's inputs")
        .iter()
        .map(|sample| (sample["budget"].clone(), sample["sizes"].clone()))
        .collect();
    let none = json!([]);
    assert_eq!(
        budgets,
        [2, 1, 2, 1].map(|budget| (json!(budget), none.clone())),
        "{curve}"
    );
    assert_eq!(curve["refresh"]["cited"], none, "{curve}");

    let labels = dir.private_file("labelling/ids.toml", "\"f1.1\" = true\n");
    let output = precision(&dir, &labels, &material, &[]);
    assert_refused_without(&output, MATERIAL_SENTINEL, &["ids.toml"]);
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
    import_history(dir, |path| {
        let db = hermes::one_session(path);
        db.home_turn("s1", start());
        let later = start() + 3600.0;
        db.owner_session("s2", later);
        match answer {
            Some(answer) => db.turn("s2", later, ASKED, answer),
            None => db.message(hermes::Message {
                session: "s2",
                content: ASKED,
                at: later,
                ..hermes::Message::default()
            }),
        }
        db.turn("s2", later + 300.0, LEANING, "Yes, a light one.");
        db
    })
}

/// The report and material of a live run on `corpus` with `script` and
/// `extra` flags, the material written to `labelling/<name>.json`.
fn material_of(
    dir: &TestDir,
    corpus: &Path,
    script: &Path,
    name: &str,
    extra: &[&str],
) -> (Value, Value) {
    let path = dir.private_path(&format!("labelling/{name}.json"));
    let mut flags = vec!["--labelling", path.to_str().unwrap()];
    flags.extend_from_slice(extra);
    let report = replay_history(dir, corpus, "live", PASSING_PROBES, Some(script), &flags).ok();
    (report, read_json(&path))
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
    let cassette = support::cassette_path(&dir);
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
        let flags = [
            "--aggregate",
            aggregate.to_str().unwrap(),
            "--labelling",
            path.to_str().unwrap(),
        ];
        let run = replay_history_to(
            &dir,
            &corpus,
            "replay",
            PASSING_PROBES,
            "collide",
            None,
            &flags,
        );
        let err = stderr(&run.output);
        assert_eq!(
            run.output.status.code(),
            Some(2),
            "--labelling over {what} is refused: {err}"
        );
        assert!(
            err.contains("--labelling") || err.contains("labelling material"),
            "the refusal of {what} names the labelling flag: {err}"
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
    let cat_quote = "we adopted a kitten called Miso";
    let corpus = import_history(&dir, |path| {
        let db = hermes::one_session(path);
        db.home_turn("s1", start());
        let later = start() + 3600.0;
        db.owner_session("s2", later);
        let news = format!("Big news: {cat_quote} yesterday.");
        db.turn("s2", later, &news, "Congratulations.");
        db
    });

    // One reply for every call: each claim survives only in the turn that
    // quotes it, the kitten changes something, and call 2 labels nothing.
    let mut cat = claim("Tim adopted a kitten called Miso.", cat_quote, "fact");
    cat["changes_something"] = json!(true);
    let claims = [home_claim(), cat].map(|mut claim| {
        claim["claim"] = json!("c1");
        claim["labels"] = json!([]);
        claim
    });
    let script = script_answering_everything(&dir, "flagged-script", claims.into(), vec![]);
    let (report, material) = material_of(&dir, &corpus, &script, "material", &[]);
    let floors = report["tuning"]["reconcile"]["embedding_floors"]
        .as_object()
        .expect("the report embeds the reconcile floors");
    assert_eq!(floors.len(), 1, "{floors:?}");
    let floor = floors.values().next().unwrap().as_f64().unwrap();
    let home_memory = report["memories"][0]["id"].as_str().unwrap();
    let flagged = material["call2"]
        .as_array()
        .unwrap()
        .iter()
        .find(|sample| sample["claim"] == "Tim adopted a kitten called Miso.")
        .expect("the flagged claim's call 2 list is in the material");
    let neighbour = candidates(flagged)
        .iter()
        .find(|candidate| candidate["memory"] == home_memory)
        .expect("call 2 was shown the home memory for the flagged claim");
    let score = neighbour["score"].as_f64().unwrap();
    assert!(
        score < floor,
        "the shown neighbour scores {score}, below the reconcile floor {floor}"
    );
}

// The precision curve.

fn precision(dir: &TestDir, labels: &Path, material: &Path, extra: &[&str]) -> Output {
    asphodel(dir)
        .args(["report", "precision", "--labels"])
        .arg(labels)
        .arg("--material")
        .arg(material)
        .args(extra)
        .output()
        .unwrap()
}

/// The curve `report precision` prints for `labels` over `material`. The
/// curve is numbers: nothing of the labels or the material is printed.
fn curve(dir: &TestDir, labels: &Path, material: &Path) -> Value {
    let output = precision(dir, labels, material, &[]);
    assert_ok(&output);
    let out = stdout(&output);
    assert!(!out.contains(MATERIAL_SENTINEL), "{out}");
    assert!(!out.contains("00000000-0000-4000"), "{out}");
    assert!(!stderr(&output).contains(MATERIAL_SENTINEL));
    serde_json::from_str(&out).expect("the curve is JSON on stdout")
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

/// `list`'s label counts in `curve`: labelled, unlabelled, matched and
/// unmatched.
fn assert_counts(curve: &Value, list: &str, want: [u64; 4]) {
    let got = ["labelled", "unlabelled", "matched", "unmatched"].map(|key| &curve[list][key]);
    assert_eq!(got, want.map(Value::from).each_ref(), "{list}: {curve}");
}

/// Words the hand-written material holds that must never be printed.
const MATERIAL_SENTINEL: &str = "SENTINEL-MATERIAL-TEXT-4c1b";

/// A memory id in the hand-written material.
fn memory(n: u32) -> String {
    format!("00000000-0000-4000-8000-{n:012}")
}

/// A chunk id in the hand-written material.
fn hand_chunk(n: u32) -> String {
    format!("00000000-0000-4000-9000-{n:012}")
}

/// A query in the hand-written material.
fn query(name: &str) -> String {
    format!("{MATERIAL_SENTINEL} query {name}")
}

fn candidate(id: &str, n: u32, score: f64) -> Value {
    let sentence = format!("{MATERIAL_SENTINEL} sentence {n}");
    json!({ "id": id, "memory": memory(n), "score": score, "sentence": sentence })
}

/// A recall sample at 09:00 on `day` of January 2026, asking `query`.
fn recall_sample(sample: &str, day: u8, session: &str, name: &str, candidates: &[Value]) -> Value {
    json!({
        "sample": sample,
        "at": format!("2026-01-{day:02}T09:00:00Z"),
        "session": session,
        "query": query(name),
        "candidates": candidates
    })
}

/// A call 2 sample at 09:00:30 on `day` of January 2026.
fn call2_sample(sample: &str, day: u8, claim: &str, candidates: &[Value]) -> Value {
    json!({
        "sample": sample,
        "at": format!("2026-01-{day:02}T09:00:30Z"),
        "claim": format!("{MATERIAL_SENTINEL} claim {claim}"),
        "candidates": candidates
    })
}

/// A small material, written by hand in the shape `--labelling` writes.
/// Two recall samples and two call 2 lists; one recall candidate is left
/// unlabelled.
fn hand_material() -> Value {
    let c = candidate;
    json!({
        "version": 1,
        "recall": [
            recall_sample("r01", 5, "s1", "one", &[
                c("r01.1", 1, 2.5), c("r01.2", 2, 0.5), c("r01.3", 3, -0.5)
            ]),
            recall_sample("r02", 6, "s2", "two", &[
                c("r02.1", 4, 1.5), c("r02.2", 5, 0.5), c("r02.3", 6, -1.5)
            ])
        ],
        "call2": [
            call2_sample("c01", 5, "one", &[
                c("c01.1", 1, 0.9), c("c01.2", 2, 0.6), c("c01.3", 3, 0.4)
            ]),
            call2_sample("c02", 6, "two", &[c("c02.1", 4, 0.6)])
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
///
/// Labels in the old form, read against the material they name, all
/// match.
#[test]
fn the_precision_curve_matches_a_hand_computed_example() {
    let dir = TestDir::new();
    let material = dir.private_json("labelling/material.json", &hand_material());
    let labels = dir.private_file("labelling/labels.toml", HAND_LABELS);
    let curve = curve(&dir, &labels, &material);
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
    assert_counts(&curve, "recall", [5, 1, 5, 0]);
    assert_counts(&curve, "call2", [4, 0, 4, 0]);
}

/// `material` with its first `"score": 2.5` replaced by the sentinel, and
/// the line it lands on.
fn broken(material: &Value) -> (String, usize) {
    let score = format!("\"score\": {MATERIAL_SENTINEL}");
    let text =
        serde_json::to_string_pretty(material)
            .unwrap()
            .replacen("\"score\": 2.5", &score, 1);
    let line = text.lines().position(|line| line.contains(&score)).unwrap();
    (text, line + 1)
}

/// A labels file or material that doesn't parse is refused naming the file
/// and the line, and a label naming no candidate in the material naming
/// the labels file, never quoting either: a label's key is whatever was
/// typed into the labels file.
#[test]
fn bad_labels_and_material_are_refused_without_quoting_them() {
    let dir = TestDir::new();
    let material = dir.private_json("labelling/material.json", &hand_material());
    let sentinel = "SENTINEL-LABEL-KEY-31f0";
    let labels = format!("{HAND_LABELS}\"{sentinel}\" = true\n");
    let labels = dir.private_file("labelling/labels.toml", &labels);
    let output = precision(&dir, &labels, &material, &[]);
    assert_refused_without(&output, sentinel, &["labels.toml"]);
    assert!(!stdout(&output).contains(MATERIAL_SENTINEL));

    let sentinel = "SENTINEL-LABEL-VALUE-8e2a";
    let labels_text = format!("{HAND_LABELS}\"r02.3\" = {sentinel}\n");
    let line = format!("line {}", labels_text.lines().count());
    let labels = dir.private_file("labelling/labels.toml", &labels_text);
    let output = precision(&dir, &labels, &material, &[]);
    assert_refused_without(&output, sentinel, &["labels.toml", &line]);

    let labels = dir.private_file("labelling/good.toml", HAND_LABELS);
    let (text, line) = broken(&hand_material());
    let material = dir.private_file("labelling/broken.json", &text);
    let output = precision(&dir, &labels, &material, &[]);
    let line = format!("line {line}");
    assert_refused_without(&output, MATERIAL_SENTINEL, &["broken.json", &line]);
}

// Labels that survive a re-record.

/// [`call2_sample`] keyed by the claim's chunk and ordinal.
fn keyed(mut sample: Value, chunk: u32, ordinal: u32) -> Value {
    sample["chunk"] = hand_chunk(chunk).into();
    sample["ordinal"] = ordinal.into();
    sample
}

/// Another run's material over the same corpus, in the shape a keyed
/// `--labelling` writes: call 2 samples name the claim's chunk and
/// ordinal. It holds the same judgements as [`hand_material`] under new
/// sample numbers and in a new order, with one recall pair and one call 2
/// pair gone, and new candidates no label judges: a memory under a query it
/// wasn't labelled for, and a neighbour of another claim in the same chunk.
fn rerecorded_material() -> Value {
    let c = candidate;
    json!({
        "version": 2,
        "recall": [
            recall_sample("r01", 6, "s2", "two", &[
                c("r01.1", 5, 0.7), c("r01.2", 4, 1.2), c("r01.3", 7, -1.0)
            ]),
            recall_sample("r02", 5, "s1", "one", &[c("r02.1", 3, -0.2), c("r02.2", 1, 2.0)]),
            recall_sample("r03", 7, "s3", "three", &[c("r03.1", 1, 0.1)])
        ],
        "call2": [
            keyed(call2_sample("c01", 6, "two", &[c("c01.1", 4, 0.55)]), 2, 1),
            keyed(call2_sample("c02", 5, "one, worded anew", &[
                c("c02.1", 2, 0.65), c("c02.2", 1, 0.85)
            ]), 1, 0),
            keyed(call2_sample("c03", 5, "three", &[c("c03.1", 3, 0.5)]), 1, 1)
        ]
    })
}

/// [`HAND_LABELS`] keyed by what they judge, with the chunks and ordinals
/// [`hand_material`]'s claims had: claim one is ordinal 0 of chunk 1, claim
/// two ordinal 1 of chunk 2.
fn keyed_hand_labels() -> String {
    let recall = [
        ("one", 1, true),
        ("one", 2, false),
        ("one", 3, false),
        ("two", 4, true),
        ("two", 5, true),
    ]
    .map(|(name, n, relevant)| {
        json!({ "query": query(name), "memory": memory(n), "relevant": relevant })
    });
    let call2 = [
        (1, 0, 1, true),
        (1, 0, 2, false),
        (1, 0, 3, false),
        (2, 1, 4, true),
    ]
    .map(|(chunk, ordinal, n, relevant)| {
        let chunk = hand_chunk(chunk);
        json!({ "chunk": chunk, "ordinal": ordinal, "memory": memory(n), "relevant": relevant })
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
    assert_counts(curve, "recall", [4, 2, 4, 1]);
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
    let material = dir.private_json("labelling/rerecorded.json", &rerecorded_material());
    let labels = dir.private_file("labelling/keyed.toml", &keyed_hand_labels());
    let curve = curve(&dir, &labels, &material);
    assert_rerecorded_recall(&curve);
    assert_counts(&curve, "call2", [3, 1, 3, 1]);
    assert_points(
        &curve,
        "call2",
        &[
            (0.55, 3, 2, 2.0 / 3.0),
            (0.65, 2, 1, 0.5),
            (0.85, 1, 1, 1.0),
        ],
    );
}

/// `report precision --convert` of `labels` against `material`, written
/// to `labelling/converted.toml`; returns what it prints and that path.
fn convert(dir: &TestDir, labels: &str, material: &Value) -> (Value, PathBuf) {
    let old_material = dir.private_json("labelling/material.json", material);
    let old_labels = dir.private_file("labelling/labels.toml", labels);
    let converted = dir.private_path("labelling/converted.toml");
    let flags = ["--convert", converted.to_str().unwrap()];
    let output = precision(dir, &old_labels, &old_material, &flags);
    assert_ok(&output);
    assert!(!stdout(&output).contains(MATERIAL_SENTINEL));
    let printed = serde_json::from_str(&stdout(&output)).expect("the curve is JSON");
    (printed, converted)
}

/// Labels in the old form, candidate ids, carry over once converted:
/// `--convert` reads them against the material they were written for, a
/// material from before keys were recorded, and writes them keyed by query
/// and memory. The converted file then scores the re-recorded material's
/// recall candidates as if it had been written keyed.
#[test]
fn old_id_labels_convert_to_keyed_labels_that_score_another_run() {
    let dir = TestDir::new();
    let (_, converted) = convert(&dir, HAND_LABELS, &hand_material());
    assert!(converted.exists(), "the keyed labels are written");
    let material = dir.private_json("labelling/rerecorded.json", &rerecorded_material());
    assert_rerecorded_recall(&curve(&dir, &converted, &material));
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
    let labels = "\"r01.1\" = true\n\"r02.1\" = false\n\"c01.1\" = true\n\"c01.2\" = false\n\"c02.1\" = true\n";
    let (printed, converted) = convert(&dir, labels, &material);
    assert_eq!(
        printed["converted"],
        json!({ "recall": 0, "call2": 0, "dropped_call2": 3, "conflicting": 2 }),
        "{printed}"
    );

    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let flags = ["--labels", converted.to_str().unwrap()];
    let (_, material_path) = labelled_replay(&dir, &corpus, "next", &flags);
    assert!(material_path.exists(), "the material is written");
}

/// Keyed labels for every candidate of `material`'s `list`, relevant when
/// `relevant` says so: recall labels by query and memory, call 2 labels by
/// the claim's chunk and ordinal and the memory.
fn keyed_labels(material: &Value, list: &str, relevant: impl Fn(&Value) -> bool) -> Vec<Value> {
    let mut labels = Vec::new();
    for sample in material[list].as_array().unwrap() {
        for candidate in candidates(sample) {
            let mut label =
                json!({ "memory": candidate["memory"], "relevant": relevant(candidate) });
            if list == "recall" {
                label["query"] = sample["query"].clone();
            } else {
                let chunk = sample["chunk"].as_str().expect("a call 2 sample's chunk");
                assert!(uuid::Uuid::parse_str(chunk).is_ok(), "{sample}");
                let ordinal = &sample["ordinal"];
                assert!(
                    ordinal.is_u64(),
                    "a call 2 sample's claim ordinal: {sample}"
                );
                label["chunk"] = chunk.into();
                label["ordinal"] = ordinal.clone();
            }
            labels.push(label);
        }
    }
    labels
}

/// A re-recorded run carries labels over: one run's material is labelled
/// by key, plus a label for a turn its even spread skipped, and a second
/// run given those labels with `--labels` samples the labelled turns in
/// preference, the skipped one included, still as many of them. Every
/// candidate the second run shows is labelled; recall labels it no longer
/// shows are unmatched, and every call 2 label matches, keyed by the
/// chunk and ordinal the material records for each claim.
#[test]
fn sampling_prefers_labelled_queries_so_labels_carry_over() {
    let dir = TestDir::new();
    let corpus = labelling_history(&dir);
    let script = labelling_script(&dir);
    replay_history(&dir, &corpus, "live", PASSING_PROBES, Some(&script), &[]).ok();
    let (first, first_path) = labelled_replay(&dir, &corpus, "first", &[]);
    let report = first.report();
    let home = &report["memories"][0]["id"];
    assert!(home.is_string(), "the history makes a memory");
    let first_material = read_json(&first_path);
    let sampled = first_material["recall"].as_array().unwrap().len();
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

    let is_home = |candidate: &Value| candidate["memory"] == *home;
    let mut recall = keyed_labels(&first_material, "recall", is_home);
    recall.push(json!({ "query": skipped, "memory": home, "relevant": false }));
    let call2 = keyed_labels(&first_material, "call2", is_home);
    assert!(!call2.is_empty(), "call 2 ran: {first_material}");
    let labels = dir.private_file(
        "labelling/keyed.toml",
        &toml::to_string(&json!({ "recall": recall, "call2": call2 })).unwrap(),
    );

    let flags = ["--labels", labels.to_str().unwrap()];
    let (_, second_path) = labelled_replay(&dir, &corpus, "second", &flags);
    let second_material = read_json(&second_path);
    assert_eq!(
        second_material["recall"].as_array().unwrap().len(),
        sampled,
        "as many sampled turns"
    );
    assert!(
        queries(&second_material).contains(&skipped),
        "the labelled turn the first run skipped is sampled"
    );

    let curve = curve(&dir, &labels, &second_path);
    let recall_curve = &curve["recall"];
    assert_eq!(recall_curve["unlabelled"], 0, "{curve}");
    let matched = recall_curve["matched"].as_u64().unwrap();
    assert!(matched > 0, "{curve}");
    assert_eq!(
        matched + recall_curve["unmatched"].as_u64().unwrap(),
        recall.len() as u64,
        "every recall label either matched or found nothing: {curve}"
    );
    assert_eq!(curve["call2"]["matched"], call2.len(), "{curve}");
    assert_eq!(curve["call2"]["unmatched"], 0, "{curve}");
    assert_eq!(curve["call2"]["unlabelled"], 0, "{curve}");
}

// Rescoring fixed pools.

fn rescore(dir: &TestDir, material: &Path, corpus: &Path, mode: &str, out: &Path) -> Output {
    rescore_with(dir, material, corpus, mode, out, &[])
}

/// [`rescore`] with `extra` flags.
fn rescore_with(
    dir: &TestDir,
    material: &Path,
    corpus: &Path,
    mode: &str,
    out: &Path,
    extra: &[&str],
) -> Output {
    asphodel(dir)
        .args(["report", "rescore", "--material"])
        .arg(material)
        .arg("--corpus")
        .arg(corpus)
        .args(["--rerank-query", mode])
        .arg("--out")
        .arg(out)
        .args(extra)
        .output()
        .unwrap()
}

/// Asserts `after` is `before` rescored: the same sample, the same
/// candidates by id with their memories and sentences, each scored with the
/// fake reranker against `after`'s rerank query, in logit order (highest
/// first, ties in the order given). Returns that query.
fn assert_rescored<'a>(before: &Value, after: &'a Value) -> &'a str {
    for key in ["sample", "at", "session", "query", "raw_query"] {
        assert_eq!(after[key], before[key], "{key} is kept: {after}");
    }
    let rerank_query = after["rerank_query"]
        .as_str()
        .unwrap_or_else(|| panic!("a rescored sample records its rerank query: {after}"));
    let mut expected: Vec<Value> = candidates(before)
        .iter()
        .map(|candidate| {
            let mut candidate = candidate.clone();
            let sentence = candidate["sentence"].as_str().unwrap();
            candidate["score"] = json!(fake_logit(rerank_query, sentence));
            candidate
        })
        .collect();
    let score = |candidate: &Value| candidate["score"].as_f64().unwrap();
    expected.sort_by(|a, b| score(b).total_cmp(&score(a)));
    assert_eq!(after["candidates"], Value::Array(expected), "{after}");
    rerank_query
}

/// Overrides setting `[injection] rerank_query` to `mode`.
fn rerank_query(dir: &TestDir, mode: &str) -> String {
    overrides(dir, &format!("[injection]\nrerank_query = \"{mode}\"\n"))
}

/// Each sample records both queries: `query`, the message the vector and
/// BM25 arms searched, and `rerank_query`, what the reranker scored
/// against. In message mode they're the same. In conversation mode the
/// reranker scores the message, the previous message and the start of the
/// assistant's reply to it from the corpus, so the reply's Auckland lifts
/// the home memory; a session's first message has nothing before it.
///
/// `report rescore` scores the same pools again against the query a mode
/// gives, keeping each sample, its candidates by id and the call 2 lists,
/// in logit order, and the same bytes each time: by the message it gives
/// back the message replay's scores, by the conversation the conversation
/// replay's query. Keyed labels match on the query, so they reach the same
/// candidates in the rescored material. It compares rerankers' queries on
/// fixed pools; it says nothing about prefetch's final ranking.
#[test]
fn the_rerank_query_mode_sets_what_replay_and_rescore_score_against() {
    let dir = TestDir::new();
    let corpus = conversation_history(&dir);
    let script = labelling_script(&dir);
    let materials = ["message", "conversation"].map(|mode| {
        let flags = ["--overrides", &rerank_query(&dir, mode)];
        material_of(&dir, &corpus, &script, mode, &flags).1
    });
    let [message, conversation] = &materials;
    for sample in message["recall"].as_array().unwrap() {
        assert_eq!(sample["rerank_query"], sample["query"], "{sample}");
    }
    let conversed = format!("{LEANING}\n{ASKED}\n{ANSWERED}");
    let leaning = sample_for(conversation, LEANING);
    assert_eq!(leaning["query"], LEANING, "the arms search the message");
    assert_eq!(leaning["rerank_query"], conversed.as_str());
    assert_eq!(sample_for(conversation, ASKED)["rerank_query"], ASKED);
    let home_score = |material: &Value| {
        candidates(sample_for(material, LEANING))
            .iter()
            .find(|candidate| candidate["sentence"] == hermes::HOME_SENTENCE)
            .expect("the home memory is a candidate")["score"]
            .as_f64()
            .unwrap()
    };
    assert_eq!(
        home_score(conversation),
        fake_logit(&conversed, hermes::HOME_SENTENCE)
    );
    assert!(home_score(conversation) > home_score(message));

    let recall = keyed_labels(message, "recall", |candidate| {
        candidate["sentence"] == hermes::HOME_SENTENCE
    });
    assert!(!recall.is_empty(), "the run shows candidates: {message}");
    let labels = dir.private_file(
        "labelling/keyed.toml",
        &toml::to_string(&json!({ "recall": recall })).unwrap(),
    );
    let input = dir.private_path("labelling/message.json");
    let before = curve(&dir, &labels, &input);
    assert_eq!(before["recall"]["unmatched"], 0, "{before}");
    assert_eq!(before["recall"]["unlabelled"], 0, "{before}");

    for (mode, replayed) in [("message", message), ("conversation", conversation)] {
        let out = dir.private_path(&format!("labelling/rescored-{mode}.json"));
        assert_ok(&rescore(&dir, &input, &corpus, mode, &out));
        let rescored = read_json(&out);
        assert_eq!(rescored["call2"], message["call2"], "{mode}");
        let before_samples = message["recall"].as_array().unwrap();
        let after_samples = rescored["recall"].as_array().unwrap();
        assert_eq!(after_samples.len(), before_samples.len(), "{mode}");
        for (before, after) in before_samples.iter().zip(after_samples) {
            let rerank_query = assert_rescored(before, after);
            let raw = before["raw_query"].as_str().unwrap();
            let replayed = &sample_for(replayed, raw)["rerank_query"];
            assert_eq!(rerank_query, replayed, "{mode}");
        }
        let again = dir.private_path(&format!("labelling/again-{mode}.json"));
        assert_ok(&rescore(&dir, &input, &corpus, mode, &again));
        assert_eq!(fs::read(&again).unwrap(), fs::read(&out).unwrap(), "{mode}");

        let after = curve(&dir, &labels, &out);
        for key in ["labelled", "unlabelled", "matched", "unmatched"] {
            assert_eq!(after["recall"][key], before["recall"][key], "{mode} {key}");
        }
        assert_eq!(
            after["recall"]["top8"]["relevant"], before["recall"]["top8"]["relevant"],
            "{mode}"
        );
    }
}

/// A previous message Hermes never answered gives the conversation no
/// reply, in a conversation replay and in a rescore alike.
#[test]
fn an_unanswered_previous_message_gives_the_conversation_no_reply() {
    let dir = TestDir::new();
    let corpus = history_asking(&dir, None);
    let script = labelling_script(&dir);
    let flags = ["--overrides", &rerank_query(&dir, "conversation")];
    let (_, replayed) = material_of(&dir, &corpus, &script, "conversation", &flags);
    let expected = format!("{LEANING}\n{ASKED}");
    assert_eq!(sample_for(&replayed, LEANING)["rerank_query"], expected);

    material_of(&dir, &corpus, &script, "message", &[]);
    let input = dir.private_path("labelling/message.json");
    let out = dir.private_path("labelling/rescored.json");
    assert_ok(&rescore(&dir, &input, &corpus, "conversation", &out));
    let rescored = read_json(&out);
    assert_eq!(sample_for(&rescored, LEANING)["rerank_query"], expected);
}

/// [`hand_material`]'s first sample alone: Tim's home turn in session s1,
/// which [`conversation_history`] holds.
fn matching_material() -> Value {
    let mut material = hand_material();
    material["recall"].as_array_mut().unwrap().truncate(1);
    material
}

/// A sample with no prefetch in the corpus at its session and time is
/// refused naming the sample, and material that doesn't parse naming the
/// file and the line, quoting nothing, and nothing is written.
#[test]
fn bad_material_is_refused_by_rescore_without_quoting_it() {
    let dir = TestDir::new();
    let corpus = conversation_history(&dir);
    let out = dir.private_path("labelling/rescored.json");
    // r02 is in session s2 a day later, where the corpus has no prefetch.
    let material = dir.private_json("labelling/hand.json", &hand_material());
    for mode in ["message", "conversation"] {
        let output = rescore(&dir, &material, &corpus, mode, &out);
        assert_refused_without(&output, MATERIAL_SENTINEL, &["r02", "corpus"]);
        for text in [hermes::HOME_QUOTE, ASKED, ANSWERED, LEANING] {
            assert!(!stderr(&output).contains(text), "{}", stderr(&output));
            assert!(!stdout(&output).contains(text), "{}", stdout(&output));
        }
        assert!(!out.exists(), "nothing is written in {mode} mode");
    }

    let (text, line) = broken(&matching_material());
    let material = dir.private_file("labelling/broken.json", &text);
    let output = rescore(&dir, &material, &corpus, "conversation", &out);
    let line = format!("line {line}");
    assert_refused_without(&output, MATERIAL_SENTINEL, &["broken.json", &line]);
    assert!(!out.exists());
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

    let outside_material = dir.file("material.json", &text);
    let outside_corpus = dir.path("corpus.jsonl");
    fs::copy(&corpus, &outside_corpus).unwrap();
    let outside_out = dir.path("rescored.json");
    for (material, corpus, out) in [
        (&outside_material, &corpus, &out),
        (&material, &outside_corpus, &out),
        (&material, &corpus, &outside_out),
    ] {
        assert_refused(&rescore(&dir, material, corpus, "message", out), "private");
        assert!(!out.exists());
    }

    let corpus_bytes = fs::read(&corpus).unwrap();
    for input in [&material, &corpus] {
        let output = rescore(&dir, &material, &corpus, "message", input);
        assert_refused(&output, "--out");
    }
    assert_eq!(fs::read(&material).unwrap(), text.as_bytes());
    assert_eq!(fs::read(&corpus).unwrap(), corpus_bytes);
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
    let mut r01 = vec![candidate("r01.1", 1, -9.0)];
    for n in 2..=9 {
        r01.push(candidate(&format!("r01.{n}"), n, f64::from(10 - n)));
    }
    r01.push(candidate("r01.10", 10, 0.5));
    let r02 = [candidate("r02.1", 11, 3.0), candidate("r02.2", 5, 2.0)];
    let mut r03 = vec![candidate("r03.1", 1, 5.0)];
    for n in 21..=29 {
        r03.push(candidate(&format!("r03.{}", n - 19), n, 0.0));
    }
    let material = json!({
        "version": 2,
        "recall": [
            recall_sample("r01", 5, "s-r01", "A", &r01),
            recall_sample("r02", 6, "s-r02", "A", &r02),
            recall_sample("r03", 7, "s-r03", "B", &r03)
        ],
        "call2": []
    });
    let label = |name: &str, n: u32, relevant: bool| json!({ "query": query(name), "memory": memory(n), "relevant": relevant });
    let mut recall: Vec<Value> = (1..=10)
        .map(|n| label("A", n, [1, 5, 10].contains(&n)))
        .collect();
    recall.extend([27, 28, 99].map(|n| label("B", n, true)));
    let labels = toml::to_string(&json!({ "recall": recall })).unwrap();
    (material, labels)
}

/// The same judgements as candidate ids: every candidate a keyed label
/// reaches, under its id in `material`.
fn id_labels_for(material: &Value, keyed: &str) -> String {
    let keyed: Value = toml::from_str(keyed).unwrap();
    let mut labels = String::new();
    for sample in material["recall"].as_array().unwrap() {
        for candidate in candidates(sample) {
            let judged = keyed["recall"].as_array().unwrap().iter().find(|label| {
                label["query"] == sample["query"] && label["memory"] == candidate["memory"]
            });
            if let Some(label) = judged {
                labels.push_str(&format!("{} = {}\n", candidate["id"], label["relevant"]));
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
    let material = dir.private_json("labelling/keyed-top8.json", &material);
    let keyed = dir.private_file("labelling/keyed-top8.toml", &keyed);
    let ids = dir.private_file("labelling/id-top8.toml", &ids);

    let by_key = curve(&dir, &keyed, &material);
    let top8 = &by_key["recall"]["top8"];
    assert_eq!(*top8, json!({ "found": 3, "relevant": 6 }), "{by_key}");
    assert_counts(&by_key, "recall", [13, 9, 12, 1]);

    let by_id = curve(&dir, &ids, &material);
    assert_eq!(by_id["recall"]["top8"], *top8, "{by_id}");
    assert_eq!(by_id["recall"]["labelled"], 13, "{by_id}");
    assert_eq!(by_id["recall"]["unlabelled"], 9, "{by_id}");
    let curves = [&by_id, &by_key].map(|curve| &curve["recall"]["curve"]);
    assert_eq!(curves[0], curves[1], "{by_id}");
}
