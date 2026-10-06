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
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use asphodel_core::config::Secret;
use asphodel_core::models::{ChatgptTokens, TokenStore};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use support::hermes::{self, StateDb, epoch, start};
use support::{
    HOME_MEMORY, PASSING_PROBES, Run, TestDir, asphodel, assert_ok, assert_refused,
    assert_refused_without, cassette_bytes, cassette_path, cassette_records, claim, home_claim,
    import_history, imported_small_history, imported_with_a_model, live_script, overrides, probe,
    probe_in, record, replay_history, said, script_answering_everything, script_steps, simulation,
    stderr, universal_script, write_cassette, write_reply,
};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// A probe at noon on `day` of January 2026.
fn probe_on(id: &str, day: u8, kind: &str, fields: &str) -> String {
    probe(id, &format!("2026-01-{day:02}T12:00:00Z"), kind, fields)
}

/// The id of the memory probe `id` observed.
fn observed_id(report: &Value, id: &str) -> Value {
    probe_in(report, id)["observed"]["id"].clone()
}

fn is_call1(record: &Value) -> bool {
    record["template"]["name"] == "extract_claims"
}

fn is_call2(record: &Value) -> bool {
    record["template"]["name"] == "reconcile_claims"
}

/// Whether a record is a refresh's write. A plan is recorded apart, under
/// `plan_model`, and replays by its key like any other call.
fn is_refresh(record: &Value) -> bool {
    record["template"]["name"] == "write_model"
}

// Probe identity and unresolved-memory regressions, through the CLI report.

/// A regex no memory's sentence matches.
const UNSEEN: &str = "never recorded wording";

/// A probe naming a `memory_id` that exists resolves by it, whatever its
/// regex matches, and reports whether the regex agrees; one naming none,
/// or an id that doesn't exist, resolves by the regex. A probe whose memory
/// resolves to nothing fails whatever its kind, even one that expects an
/// absence, and fails the run. The grounding ids come from a `live` run's
/// report, not from UUID derivation or the store.
#[test]
fn probes_resolve_by_id_then_regex_and_an_unresolved_probe_fails() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let harbour = claim("Tim lives near the harbour.", "near the harbour", "fact");
    let script =
        script_answering_everything(&dir, "resolution", vec![home_claim(), harbour], vec![]);
    let grounding = probe_on("home", 8, "exists", HOME_MEMORY)
        + &probe_on("harbour", 8, "exists", "memory = \"near the harbour\"");
    let report = replay_history(&dir, &corpus, "live", &grounding, Some(&script), &[]).ok();
    let (home, harbour) = (
        observed_id(&report, "home"),
        observed_id(&report, "harbour"),
    );
    assert!(home.is_string() && harbour.is_string(), "{report}");
    assert_ne!(home, harbour, "the fixture must create distinct memories");

    let anchored = |pattern: &str| format!("memory_id = {harbour}\nmemory = \"{pattern}\"");
    let unknown = "memory_id = \"00000000-0000-0000-0000-000000000000\"";
    // (probe, its fields, resolved by id rather than regex, the regex agrees)
    let resolved = [
        ("id-beats-regex", anchored("lives in Auckland"), true, false),
        ("id-only", anchored(UNSEEN), true, false),
        ("id-and-regex", anchored("near the harbour"), true, true),
        (
            "unknown-id",
            format!("{unknown}\n{HOME_MEMORY}"),
            false,
            true,
        ),
        ("regex-only", HOME_MEMORY.into(), false, true),
    ];
    let unresolved = [
        ("agenda_lacks", ""),
        ("recall_lacks", "query = \"Tim lives in Auckland\""),
        ("profile_lacks", "model = \"User profile\""),
        ("absent", ""),
    ];
    let mut probes = String::new();
    for (id, fields, ..) in &resolved {
        probes += &probe_on(id, 8, "exists", fields);
    }
    for (kind, fields) in unresolved {
        probes += &probe_on(kind, 8, kind, &format!("memory = \"{UNSEEN}\"\n{fields}"));
    }
    let run = replay_history(&dir, &corpus, "replay", &probes, None, &[]);
    assert_eq!(
        run.output.status.code(),
        Some(1),
        "the unresolved probes fail the run: {}",
        stderr(&run.output)
    );
    let report = run.report();
    for (id, _, by_id, agrees) in resolved {
        let probe = probe_in(&report, id);
        let (memory, by) = if by_id {
            (&harbour, "id")
        } else {
            (&home, "regex")
        };
        let observed = &probe["observed"];
        assert_eq!(
            (
                &probe["passed"],
                &observed["id"],
                &observed["resolved_by"],
                &observed["regex_matches"]
            ),
            (&json!(true), memory, &json!(by), &json!(agrees)),
            "{probe}"
        );
    }
    for (kind, _) in unresolved {
        let probe = probe_in(&report, kind);
        assert_eq!(
            (&probe["passed"], &probe["observed"]["resolved"]),
            (&json!(false), &json!(false)),
            "{probe}"
        );
    }
}

#[test]
fn probe_resolution_absent_passes_after_a_created_memory_is_purged() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let mut claim = home_claim();
    claim["significance"] = json!("trivial");
    let script = script_answering_everything(&dir, "purge", vec![claim], vec![]);
    let quiet = overrides(&dir, "[clock]\nquiet_rate = 1.0\n");
    let flags = ["--overrides", &quiet];
    let grounding = replay_history(&dir, &corpus, "live", PASSING_PROBES, Some(&script), &flags);
    let home = observed_id(&grounding.ok(), "p001");
    let probes = probe("purged", "2026-11-05T12:00:00Z", "absent", HOME_MEMORY);
    let report = replay_history(&dir, &corpus, "replay", &probes, None, &flags).ok();
    let memory = report["memories"]
        .as_array()
        .unwrap()
        .iter()
        .find(|memory| memory["id"] == home)
        .expect("the created list retains home");
    assert!(memory["purged_at"].is_string(), "{memory}");
    let probe = probe_in(&report, "purged");
    assert_eq!(probe["passed"], true, "{probe}");
    assert_eq!(probe["observed"]["id"], home, "{probe}");
    assert_eq!(probe["observed"]["present"], false, "{probe}");
}

// Recording modes.

/// `live` calls the LLM, records, and reports the hash of the completed
/// cassette. `replay` of that cassette needs no LLM, simulates the same
/// run, and writes a byte-identical report each time.
#[test]
fn replay_of_a_live_cassette_simulates_the_same_run_without_an_llm() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let live = record(&dir, &corpus).report();
    let cassette = cassette_bytes(&dir);
    assert!(!cassette.is_empty(), "the live run must record calls");
    assert_eq!(live["cassette_hash"], sha(&cassette));

    let [first, second] =
        [(); 2].map(|()| replay_history(&dir, &corpus, "replay", PASSING_PROBES, None, &[]));
    let replay = first.ok();
    assert_ok(&second.output);
    assert!(live["llm"]["live"].as_u64().unwrap() > 0, "{live}");
    assert_eq!(replay["llm"]["live"], 0, "{replay}");
    assert!(replay["llm"]["cache"].as_u64().unwrap() > 0, "{replay}");
    assert_eq!(simulation(&live), simulation(&replay));
    assert_eq!(
        first.report_bytes(),
        second.report_bytes(),
        "two replays of the same configuration differ"
    );
}

/// One server-sent event of `kind` carrying `data`.
fn sse(kind: &str, mut data: Value) -> String {
    data["type"] = json!(kind);
    format!("event: {kind}\ndata: {data}\n\n")
}

/// A loopback server answering its `n`th request, counting from 0, with
/// `answer(n)`, a whole HTTP response. Returns its URL and the number of
/// requests it has answered.
fn loopback(answer: impl Fn(usize) -> String + Send + 'static) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let requests = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&requests);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let line = line.trim_end().to_ascii_lowercase();
                if line.is_empty() {
                    break;
                }
                if let Some(value) = line.strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
            }
            reader.read_exact(&mut vec![0; length]).unwrap();
            let response = answer(counted.fetch_add(1, Ordering::SeqCst));
            let _ = stream.write_all(response.as_bytes());
        }
    });
    (url, requests)
}

/// A 200 response with `body` of `content_type`.
fn ok(content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// The Codex stream answering with `reply`.
fn codex_answer(reply: &Value) -> String {
    let text = reply.to_string();
    let item = json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]});
    let body = sse("response.output_item.done", json!({ "item": item }))
        + &sse(
            "response.completed",
            json!({"response": {"id": "resp_1", "status": "completed"}}),
        );
    ok("text/event-stream", &body)
}

/// A loopback stand-in for the Codex backend. Its first `failures`
/// requests stream `response.failed` with `code`; every later one streams
/// `reply` as the answer. Returns its URL and the number of requests it has
/// answered.
fn codex_failing(code: &str, failures: usize, reply: &Value) -> (String, Arc<AtomicUsize>) {
    let failed = ok(
        "text/event-stream",
        &sse(
            "response.failed",
            json!({"response": {"id": "resp_f", "status": "failed", "error": {"code": code, "message": "busy"}}}),
        ),
    );
    let answered = codex_answer(reply);
    loopback(move |n| {
        if n < failures {
            failed.clone()
        } else {
            answered.clone()
        }
    })
}

/// A Codex stand-in that rejects its first request's login with a 401 and
/// streams `reply` to every later one.
fn codex_unauthorized_once(reply: &Value) -> (String, Arc<AtomicUsize>) {
    let answered = codex_answer(reply);
    loopback(move |n| {
        if n == 0 {
            "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into()
        } else {
            answered.clone()
        }
    })
}

/// A loopback OAuth issuer that answers every refresh with fresh tokens
/// for the account [`logged_in`] uses. Returns its URL and the number of
/// refreshes it has answered.
fn issuer_refreshing() -> (String, Arc<AtomicUsize>) {
    let tokens = json!({
        "id_token": jwt(json!({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-test"}})),
        "access_token": jwt(json!({"sub": "refreshed", "exp": 4_102_444_800_i64})),
        "refresh_token": "rt-refreshed",
    })
    .to_string();
    loopback(move |_| ok("application/json", &tokens))
}

/// A JWT-shaped token with `claims`; nothing checks the signature.
fn jwt(claims: Value) -> String {
    let header = base64url(br#"{"alg":"RS256","typ":"JWT"}"#);
    let payload = base64url(claims.to_string().as_bytes());
    format!("{header}.{payload}.{}", base64url(b"signature"))
}

fn base64url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let mut buffer = [0u8; 3];
        buffer[..chunk.len()].copy_from_slice(chunk);
        let n = u32::from_be_bytes([0, buffer[0], buffer[1], buffer[2]]);
        for i in 0..chunk.len() + 1 {
            out.push(ALPHABET[((n >> (18 - 6 * i)) & 0x3f) as usize] as char);
        }
    }
    out
}

/// A ChatGPT login in `token_dir` whose access token outlives the test, so
/// no refresh is attempted.
fn logged_in(token_dir: &Path) {
    fs::create_dir_all(token_dir).unwrap();
    let auth = json!({"chatgpt_account_id": "acct-test"});
    let tokens = ChatgptTokens {
        access_token: Secret::new(jwt(json!({"sub": "test", "exp": 4_102_444_800_i64}))),
        refresh_token: Secret::new("rt-test"),
        id_token: Secret::new(jwt(json!({ "https://api.openai.com/auth": auth }))),
        account_id: "acct-test".into(),
        last_refresh: "2026-01-01T00:00:00Z".parse().unwrap(),
    };
    TokenStore::open(token_dir).save(&tokens).unwrap();
}

/// A `live` run whose LLM is the Codex stub `codex_failing` starts:
/// its output and report path, and the stub's request count.
fn live_against_codex(dir: &TestDir, code: &str, failures: usize) -> (Run, u64) {
    let corpus = imported_small_history(dir);
    let reply = support::reply_to_everything(vec![home_claim()], vec![]);
    let (url, requests) = codex_failing(code, failures, &reply);
    let token_dir = dir.private_path("tokens");
    logged_in(&token_dir);
    let llm = format!("[llm]\nauth = \"chatgpt\"\nendpoint = \"{url}\"\nmodel = \"gpt-test\"\n");
    let probes = dir.private_file("probes.toml", PASSING_PROBES);
    let report_path = dir.private_path(&format!("reports/{code}.json"));
    let output = asphodel(dir)
        .arg("replay")
        .arg("--corpus")
        .arg(&corpus)
        .args(["--mode", "live"])
        .arg("--cassette")
        .arg(cassette_path(dir))
        .arg("--probes")
        .arg(&probes)
        .arg("--report")
        .arg(&report_path)
        .args(["--overrides", &overrides(dir, &llm)])
        .arg("--token-dir")
        .arg(&token_dir)
        .env("ASPHODEL_LLM_RETRY_WAIT_MS", "0")
        .output()
        .unwrap();
    let run = Run {
        output,
        report_path,
    };
    (run, requests.load(Ordering::SeqCst) as u64)
}

/// The ChatGPT backend reports overload inside the stream, as a
/// `response.failed` event, not as an HTTP status. The provider is down,
/// so a `live` run retries that call until it answers, however many
/// attempts that takes, waiting nothing here (`ASPHODEL_LLM_RETRY_WAIT_MS`),
/// and completes with every call answered.
#[test]
fn a_live_call_the_backend_fails_as_overloaded_is_retried_until_it_answers() {
    let dir = TestDir::new();
    let (run, requests) = live_against_codex(&dir, "server_is_overloaded", 8);
    let report = run.ok();
    assert!(requests > 8, "{requests} requests");
    assert_eq!(report["llm"]["live"], requests - 8, "{report}");
}

/// A `server_error` may be the request's own doing, and replay has no
/// chunk retry cap to stop a request that always fails, so a `live` run
/// tries it a few times and then fails rather than retrying forever.
#[test]
fn a_live_call_the_backend_always_fails_with_server_error_fails_the_run() {
    let dir = TestDir::new();
    let (run, requests) = live_against_codex(&dir, "server_error", usize::MAX);
    assert!(!run.output.status.success(), "{}", stderr(&run.output));
    assert!((2..=10).contains(&requests), "{requests} requests");
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
    let five = overrides(&dir, "[llm]\nconcurrency = 5\n");
    let flags = ["--overrides", &five];
    let pooled = replay_history(&dir, &corpus, "replay", PASSING_PROBES, None, &flags).ok();
    assert_eq!(pooled["llm"]["misses"], 0, "{pooled}");
    assert_eq!(pooled["memories"], serial["memories"], "{pooled}");
}

/// A script whose every step answers any call: call 1 with the home
/// claim, call 2 labelling it a mention of the one neighbour, and a
/// refresh writing one sentence citing `m1`.
fn mention_script(dir: &TestDir) -> PathBuf {
    let mut claim = home_claim();
    claim["claim"] = json!("c1");
    claim["labels"] = json!([{"neighbour": "n1", "label": "mentioned_again"}]);
    script_answering_everything(dir, "mention-script", vec![claim], vec![])
}

/// The small history with the home turn said again ten minutes later in
/// the same session, recorded `live` with `probes` and [`mention_script`],
/// two chunks out at once in simulated time with an hour's latency.
/// Returns the corpus, the overrides that pool the chunks, and the live
/// report.
fn record_pooled(dir: &TestDir, probes: &str) -> (PathBuf, String, Value) {
    let corpus = import_history(dir, |path| {
        let db = hermes::small_history(path);
        let again = format!("{}, still.", hermes::HOME_QUOTE);
        db.turn(
            "s-main",
            epoch("2026-01-05T09:10:00Z"),
            &again,
            "Noted again.",
        );
        db
    });
    let pooled = overrides(dir, "[llm]\nconcurrency = 2\n");
    let flags = ["--overrides", &pooled, "--latency", "1h"];
    let script = mention_script(dir);
    let live = replay_history(dir, &corpus, "live", probes, Some(&script), &flags).ok();
    (corpus, pooled, live)
}

/// At concurrency 2 the repeat is claimed before the first home turn
/// commits, so neither sees the other at its search. The repeat's commit
/// is stale: it runs call 2 again through the cassette, live in `live`,
/// and becomes a mention of the one memory. `replay` of that cassette
/// simulates the same run.
///
/// Under `--latency 1h` the redo is charged the hour again. The home
/// turn syncs at 09:00:30 and commits at 10:00:30; the repeat syncs at
/// 09:10:30, completes at 10:10:30, and its redo commits at 11:10:30, two
/// hours after its sync. Until then the store doesn't hold its mention.
#[test]
fn a_stale_commit_redoes_call_2_through_the_cassette_and_replays() {
    let dir = TestDir::new();
    let strong = format!("{HOME_MEMORY}\nband = \"strong\"");
    let probes = format!(
        "{PASSING_PROBES}{}{}",
        probe("p003", "2026-01-05T11:10:29Z", "band", &strong),
        probe("p004", "2026-01-05T11:10:30Z", "band", &strong)
    );
    let (corpus, pooled, live) = record_pooled(&dir, &probes);
    assert_eq!(live["call2_rate"]["redos"], 1, "{live}");
    assert_eq!(live["call2_rate"]["call2"], 1, "{live}");
    let hours = |n: u64| n * 60 * 60 * 1000;
    assert_eq!(live["extraction_lag"]["p50_ms"], hours(1), "{live}");
    assert_eq!(
        live["extraction_lag"]["p95_ms"],
        hours(2),
        "the redo is charged the constant latency again: {live}"
    );
    let strength = |id| {
        probe_in(&live, id)["observed"]["strength"]
            .as_f64()
            .unwrap()
    };
    assert!(
        strength("p004") > strength("p003"),
        "the mention lands at 11:10:30, not before: {live}"
    );
    assert_eq!(
        live["memories"].as_array().unwrap().len(),
        1,
        "a mention, not a copy: {live}"
    );
    let call2 = cassette_records(&dir).into_iter().filter(is_call2).count();
    assert_eq!(call2, 1, "the redo's call 2 is recorded");

    let flags = ["--overrides", &pooled, "--latency", "1h"];
    let replay = replay_history(&dir, &corpus, "replay", &probes, None, &flags).ok();
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
    let (corpus, pooled, _) = record_pooled(&dir, PASSING_PROBES);
    let minutes = |n: u64| n * 60 * 1000;
    let timed: Vec<Value> = cassette_records(&dir)
        .into_iter()
        .map(|mut record| {
            let latency = if is_call2(&record) { 10 } else { 20 };
            record["latency_ms"] = minutes(latency).into();
            record
        })
        .collect();
    write_cassette(&dir, &timed);

    // Latency from the cassette: no --latency.
    let flags = ["--overrides", &pooled];
    let report = replay_history(&dir, &corpus, "replay", PASSING_PROBES, None, &flags).ok();
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
    let (corpus, pooled, _) = record_pooled(&dir, PASSING_PROBES);
    let without: Vec<Value> = cassette_records(&dir)
        .into_iter()
        .filter(|record| !is_call2(record))
        .collect();
    write_cassette(&dir, &without);
    let flags = ["--overrides", &pooled, "--latency", "1h"];

    let replay = replay_history(&dir, &corpus, "replay", PASSING_PROBES, None, &flags);
    assert_refused(&replay.output, "miss");
    assert!(
        !replay.report_path.exists(),
        "a failed run writes no report"
    );

    let script = mention_script(&dir);
    let fast = replay_history(&dir, &corpus, "fast", PASSING_PROBES, Some(&script), &flags).ok();
    assert_eq!(fast["call2_rate"]["redos"], 1, "{fast}");
    assert_eq!(fast["llm"]["misses"], 1, "{fast}");
    assert_eq!(fast["llm"]["live"], 1, "{fast}");
    assert_eq!(
        cassette_records(&dir).into_iter().filter(is_call2).count(),
        1,
        "fast records the call it answered live"
    );
}

/// Overrides that shut the reranker gate, so nothing is injected.
const SHUT_GATE: &str = "[injection.reranker_floors]\n\"fake-reranker:v1\" = 1000000.0\n";

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
    let shut = overrides(&dir, SHUT_GATE);
    let script = live_script(&dir);
    let flags = ["--overrides", &shut];
    replay_history(&dir, &corpus, "live", PASSING_PROBES, Some(&script), &flags).ok();
    let judged = || -> Vec<Value> {
        cassette_records(&dir)
            .into_iter()
            .filter(|record| record["template"]["name"] == "judge_used")
            .collect()
    };
    assert!(judged().is_empty());

    let judge = support::script(&dir, "judge-script", json!({ "used": [] }));
    let report = replay_history(&dir, &corpus, "fast", PASSING_PROBES, Some(&judge), &[]).ok();
    let top_ups = judged();
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
    let verdicts = &report["llm"]["used_verdicts"]["top_up"];
    assert!(verdicts.as_u64().unwrap() > 0, "{report}");

    let cassette = cassette_bytes(&dir);
    let report = replay_history(&dir, &corpus, "fast", PASSING_PROBES, None, &[]).ok();
    assert_eq!(report["llm"]["top_up"], 0, "{report}");
    assert_eq!(report["llm"]["misses"], 0, "{report}");
    assert_eq!(
        cassette_bytes(&dir),
        cassette,
        "a fast run with nothing to judge records nothing"
    );
}

/// A recorded call 1 reply naming a handle its record doesn't list, as one
/// naming a mental model entry (`n1`) could before call 1 credited `used`
/// by memory alone, may have relied on any listed memory through it. In
/// `fast` its claims are still reused and the memories it names are still
/// used, but its other pairs are judged by a top-up rather than read as not
/// used. A reply naming only listed handles leaves nothing to judge.
#[test]
fn fast_tops_up_the_pairs_a_reply_naming_an_unlisted_handle_leaves_open() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let script = live_script(&dir);
    replay_history(&dir, &corpus, "live", PASSING_PROBES, Some(&script), &[]).ok();
    let report = replay_history(&dir, &corpus, "fast", PASSING_PROBES, None, &[]).ok();
    assert_eq!(report["llm"]["top_up"], 0, "{report}");

    // The first reply with memories in context also names its first one.
    let mut records = cassette_records(&dir);
    let mut named = None;
    let mut open = BTreeSet::new();
    for record in records.iter_mut().filter(|record| is_call1(record)) {
        let shown: Vec<Value> = record["in_context"].as_array().unwrap().clone();
        if shown.is_empty() {
            continue;
        }
        let mut used = vec![json!("n1")];
        if named.is_none() {
            used.push(shown[0]["handle"].clone());
            named = Some((record["chunk"].clone(), shown[0]["sentence"].clone()));
        }
        if shown.len() >= used.len() {
            open.insert(record["chunk"].to_string());
        }
        record["response"]["json"]["used_injected_ids"] = json!(used);
    }
    let (named_chunk, named_sentence) = named.expect("a later turn has the home memory in context");
    assert!(!open.is_empty(), "some reply leaves a memory unnamed");
    write_cassette(&dir, &records);

    let judge = support::script(&dir, "judge-script", json!({ "used": [] }));
    let report = replay_history(&dir, &corpus, "fast", PASSING_PROBES, Some(&judge), &[]).ok();
    let top_ups: Vec<Value> = cassette_records(&dir)
        .into_iter()
        .filter(|record| record["template"]["name"] == "judge_used")
        .collect();
    let judged: BTreeSet<String> = top_ups
        .iter()
        .map(|record| record["chunk"].to_string())
        .collect();
    assert_eq!(judged, open, "{report}");
    assert_eq!(
        report["llm"]["live"], report["llm"]["top_up"],
        "only top-ups call the LLM; claims are reused: {report}"
    );
    for record in top_ups
        .iter()
        .filter(|record| record["chunk"] == named_chunk)
    {
        let sentences: Vec<&Value> = record["in_context"]
            .as_array()
            .unwrap()
            .iter()
            .map(|memory| &memory["sentence"])
            .collect();
        assert!(
            !sentences.contains(&&named_sentence),
            "the memory the reply named was judged again: {record}"
        );
    }
}

/// The `[extraction] guidance` the guided runs use.
const GUIDANCE: &str = "Skip routine checks.";

/// Call 1 claims recorded under other rules came from another prompt, so
/// `fast` doesn't reuse them: claims and refreshes recorded without
/// `[llm] language` (even with `--refresh recorded`), claims recorded
/// without `[extraction] guidance`, and claims recorded under another
/// version of call 1's template. Each is a miss answered live and recorded
/// under the run's rules, and the next `fast` run under them reuses
/// everything without recording more. The report carries the guidance's
/// hash.
#[test]
fn fast_doesnt_reuse_claims_recorded_under_other_rules() {
    let call1 = |records: &[Value]| records.iter().filter(|r| is_call1(r)).count() as u64;
    for case in ["language", "guidance", "template version"] {
        let dir = TestDir::new();
        let corpus = imported_small_history(&dir);
        record(&dir, &corpus);
        let mut records = cassette_records(&dir);
        let recorded = call1(&records);
        let refreshes = records.iter().filter(|r| is_refresh(r)).count() as u64;
        assert!(recorded > 0, "the fixture must record call 1");
        assert!(refreshes > 0, "the fixture must record a refresh");
        let (toml, misses) = match case {
            "language" => (
                "[llm]\nlanguage = \"English\"\n".into(),
                recorded + refreshes,
            ),
            "guidance" => (
                format!("[extraction]\nguidance = \"{GUIDANCE}\"\n"),
                recorded,
            ),
            _ => {
                // As if recorded before the template changed: another
                // version, and so another request key.
                for record in records.iter_mut().filter(|record| is_call1(record)) {
                    let other = record["template"]["version"].as_u64().unwrap() + 1;
                    record["template"]["version"] = other.into();
                    record["request"]["template"]["version"] = other.into();
                    record["key"] = format!("{}-old", record["key"].as_str().unwrap()).into();
                }
                write_cassette(&dir, &records);
                (String::new(), recorded)
            }
        };
        let overrides = overrides(&dir, &toml);
        let flags = ["--overrides", &overrides, "--refresh", "recorded"];
        let script = live_script(&dir);
        let report =
            replay_history(&dir, &corpus, "fast", PASSING_PROBES, Some(&script), &flags).ok();
        assert_eq!(report["llm"]["misses"], misses, "{case}: {report}");
        assert_eq!(report["llm"]["live"], misses, "{case}: {report}");
        assert_eq!(call1(&cassette_records(&dir)), 2 * recorded, "{case}");
        if case == "guidance" {
            let hash = &report["call1"]["guidance_hash"];
            assert_eq!(*hash, sha(GUIDANCE.as_bytes()), "{report}");
        }
        let cassette = cassette_bytes(&dir);

        let second = replay_history(&dir, &corpus, "fast", PASSING_PROBES, None, &flags).ok();
        assert_eq!(second["llm"]["misses"], 0, "{case}");
        assert_eq!(
            cassette_bytes(&dir),
            cassette,
            "{case}: claims and refreshes recorded under the run's rules are reused"
        );
    }
}

// The determinism self-test.

/// `fast` on its own recording needs no LLM, counts zero misses and passes
/// the self-test.
#[test]
fn the_self_test_passes_for_fast_with_zero_misses() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let flags = ["--self-test"];
    let report = replay_history(&dir, &corpus, "fast", PASSING_PROBES, None, &flags).ok();
    assert_eq!(report["llm"]["misses"], 0);
}

// Priming call 1.

/// A `fast` run kept to call 1 alone, with `extra` flags: the reranker
/// gate shut, so nothing is injected and no pair needs a top-up, and
/// refreshes writing nothing.
fn fast_call1_only(dir: &TestDir, corpus: &Path, script: Option<&Path>, extra: &[&str]) -> Run {
    let shut = overrides(dir, SHUT_GATE);
    let mut flags = vec!["--overrides", &shut, "--refresh", "off"];
    flags.extend(extra);
    replay_history(dir, corpus, "fast", PASSING_PROBES, script, &flags)
}

/// The chunks of the cassette's call 1 records, in file order.
fn call1_chunks(dir: &TestDir) -> Vec<String> {
    cassette_records(dir)
        .iter()
        .filter(|record| is_call1(record))
        .map(|record| record["chunk"].to_string())
        .collect()
}

/// The chunks a serial `live` run on the small history records call 1 for.
fn serial_call1_chunks() -> Vec<String> {
    let serial = TestDir::new();
    record(&serial, &imported_small_history(&serial));
    call1_chunks(&serial)
}

/// Priming relies on reusing claims by chunk, which only `fast` does, so
/// `--prime-concurrency`, bare or with a value, is refused in `live`,
/// `replay` and scenario runs. In `fast` the bare flag primes, and the
/// report says how many calls at a time.
#[test]
fn prime_concurrency_is_fast_only_and_primes_when_bare() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let script = live_script(&dir);
    for prime in [&["--prime-concurrency"][..], &["--prime-concurrency", "4"]] {
        for mode in ["live", "replay"] {
            let run = replay_history(&dir, &corpus, mode, PASSING_PROBES, Some(&script), prime);
            assert_refused(&run.output, "fast");
            assert!(!run.report_path.exists(), "a refused run writes no report");
        }
        let output = asphodel(&dir)
            .args(["replay", "--scenario", "unused.toml"])
            .args(prime)
            .output()
            .unwrap();
        assert_refused(&output, "--scenario");
    }

    let report = fast_call1_only(&dir, &corpus, Some(&script), &["--prime-concurrency"]).ok();
    assert!(report["flags"]["prime_concurrency"].is_u64(), "{report}");
    assert!(report["llm"]["primed"].as_u64().unwrap() > 0, "{report}");
    assert_eq!(report["llm"]["misses"], 0, "{report}");
}

/// `--prime-concurrency` records call 1 for every chunk a serial run
/// records it for, once each and in the serial run's order however the
/// calls finish, before the simulation starts. The simulation then has no
/// miss, and the report says the run was primed and how many calls the
/// prime made. A later unprimed `fast` run reuses the primed claims with
/// no LLM at all, and its report says it wasn't primed.
#[test]
fn a_primed_fast_run_records_call_1_for_every_chunk_before_simulating() {
    let expected = serial_call1_chunks();
    assert!(expected.len() > 1, "the fixture needs several chunks");

    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    // The first called answers last: step `i` takes `(8 - i) × 100` ms,
    // and from the eighth on no time at all.
    let reply = support::reply_to_everything(vec![home_claim()], vec![]);
    let steps: Vec<Value> = (0..64u64)
        .map(|i| json!({ "reply": reply, "delay_ms": 800u64.saturating_sub(i * 100) }))
        .collect();
    let script = script_steps(&dir, "backwards-script", &steps);
    let prime = ["--prime-concurrency", "4"];
    let report = fast_call1_only(&dir, &corpus, Some(&script), &prime).ok();
    assert_eq!(report["flags"]["prime_concurrency"], 4, "{report}");
    assert_eq!(report["llm"]["primed"], expected.len(), "{report}");
    assert_eq!(report["llm"]["misses"], 0, "{report}");
    assert_eq!(call1_chunks(&dir), expected);

    let report = fast_call1_only(&dir, &corpus, None, &[]).ok();
    assert!(report["flags"]["prime_concurrency"].is_null(), "{report}");
    assert_eq!(report["llm"]["primed"], 0, "{report}");
    assert_eq!(report["llm"]["misses"], 0, "{report}");
}

/// The prime skips chunks whose claims `fast` would already reuse, so
/// priming a recorded history calls nothing and leaves the cassette as it
/// was.
#[test]
fn priming_a_recorded_history_calls_nothing() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let cassette = cassette_bytes(&dir);
    let script = live_script(&dir);
    let prime = ["--prime-concurrency", "4"];
    let report = fast_call1_only(&dir, &corpus, Some(&script), &prime).ok();
    assert_eq!(report["llm"]["primed"], 0, "{report}");
    assert_eq!(report["llm"]["misses"], 0, "{report}");
    assert_eq!(cassette_bytes(&dir), cassette, "nothing needed priming");
}

/// A failed prime keeps successful in-flight replies, writes no report,
/// and resumes by calling only chunks still absent from the cassette.
#[test]
fn a_failed_prime_preserves_replies_and_resumes_only_missing_chunks() {
    let expected: BTreeSet<String> = serial_call1_chunks().into_iter().collect();
    assert!(expected.len() > 2, "the fixture needs unstarted chunks too");

    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let reply = support::reply_to_everything(vec![home_claim()], vec![]);
    // Both workers start before the refusal; the successful reply arrives
    // afterwards, so recovery must also preserve work still in flight.
    let steps = [
        json!({"reply": reply, "delay_ms": 500}),
        json!({"fail": "refused", "delay_ms": 100}),
    ];
    let script = script_steps(&dir, "partial-failure", &steps);
    let prime = ["--prime-concurrency", "2"];
    let failed = fast_call1_only(&dir, &corpus, Some(&script), &prime);
    assert_refused(&failed.output, "priming call 1 failed");
    assert!(
        !failed.report_path.exists(),
        "a failed prime writes no report"
    );
    let preserved = cassette_records(&dir);
    assert_eq!(preserved.len(), 1, "{preserved:?}");
    assert_eq!(preserved[0]["response"]["json"], reply);
    let completed = call1_chunks(&dir);
    assert_eq!(completed.len(), 1);
    assert!(expected.contains(&completed[0]));
    let before = cassette_bytes(&dir);

    // Exactly enough replies for the missing chunks. Any redundant call
    // exhausts the script and fails instead of silently duplicating work.
    let missing = expected.len() - completed.len();
    let script = script_steps(&dir, "resume", &vec![json!({"reply": reply}); missing]);
    let report = fast_call1_only(&dir, &corpus, Some(&script), &prime).ok();
    assert_eq!(report["llm"]["primed"], missing, "{report}");
    assert_eq!(report["llm"]["misses"], 0, "{report}");
    assert!(cassette_bytes(&dir).starts_with(&before));
    let chunks = call1_chunks(&dir);
    assert_eq!(chunks.len(), expected.len(), "no duplicate recordings");
    assert_eq!(chunks.into_iter().collect::<BTreeSet<_>>(), expected);
}

/// The prime makes its calls `--prime-concurrency` at a time in wall-clock
/// time, where an unprimed run makes them one after another. With every
/// call taking `DELAY_MS`, priming the small history at 4 saves at least
/// half of the wait the serial calls beyond the first batch add.
#[test]
fn priming_makes_the_call_1s_in_parallel() {
    const DELAY_MS: u64 = 3000;
    let timed = |prime: &[&str]| {
        let dir = TestDir::new();
        let corpus = imported_small_history(&dir);
        let script = support::delayed_script(&dir, DELAY_MS);
        let started = Instant::now();
        let report = fast_call1_only(&dir, &corpus, Some(&script), prime).ok();
        (started.elapsed(), report)
    };
    let (unprimed_elapsed, unprimed) = timed(&[]);
    let calls = unprimed["llm"]["live"].as_u64().unwrap();
    assert!(calls > 1, "the fixture needs several chunks");
    let (primed_elapsed, primed) = timed(&["--prime-concurrency", "4"]);
    assert_eq!(primed["llm"]["primed"], calls);

    let batches = calls.div_ceil(4);
    let saved = Duration::from_millis((calls - batches) * DELAY_MS / 2);
    assert!(
        unprimed_elapsed.saturating_sub(primed_elapsed) >= saved,
        "unprimed took {unprimed_elapsed:?} and primed {primed_elapsed:?} for {calls} calls"
    );
}

// The report.

/// Injected tokens per session, with cron reported apart so it doesn't
/// skew the percentiles: cron and subagent sessions aren't turn sessions.
#[test]
fn the_report_counts_injected_tokens_with_cron_apart() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let report = replay_history(&dir, &corpus, "replay", PASSING_PROBES, None, &[]).ok();
    let injected = &report["injected_tokens"];
    let sessions: BTreeSet<&str> = injected["sessions"]
        .as_array()
        .unwrap_or_else(|| panic!("injected tokens per session: {report}"))
        .iter()
        .map(|session| session["session"].as_str().unwrap())
        .collect();
    assert_eq!(sessions, BTreeSet::from(["s-later", "s-main"]));
    assert_eq!(injected["cron"]["prefetches"], 1, "{report}");
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
    // p003 fails: the memory isn't absent.
    let probes = PASSING_PROBES.to_owned() + &probe_on("p003", 9, "absent", HOME_MEMORY);
    let flags = ["--aggregate", aggregate.to_str().unwrap()];
    let run = replay_history(&dir, &corpus, "replay", &probes, None, &flags);
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
    let script = live_script(&dir);
    let probes = dir.private_file("probes.toml", PASSING_PROBES);
    let (report, cassette) = (dir.path("report.json"), dir.path("cassette.jsonl"));
    for (report, cassette, outside) in [
        (&report, &cassette_path(&dir), &report),
        (&dir.private_path("reports/live.json"), &cassette, &cassette),
    ] {
        let output = asphodel(&dir)
            .arg("replay")
            .arg("--corpus")
            .arg(&corpus)
            .args(["--mode", "live"])
            .arg("--cassette")
            .arg(cassette)
            .arg("--probes")
            .arg(&probes)
            .arg("--report")
            .arg(report)
            .env("ASPHODEL_LLM_SCRIPT", &script)
            .output()
            .unwrap();
        assert_refused(&output, "private");
        assert!(!outside.exists());
    }
}

// The A/B diff.

/// The diff refuses to compare runs with a different corpus unless forced,
/// and compares runs on the same one.
#[test]
fn the_diff_refuses_runs_on_different_corpora_unless_forced() {
    let replayed = |dir: &TestDir, corpus: &Path| {
        let run = replay_history(dir, corpus, "replay", PASSING_PROBES, None, &[]);
        assert_ok(&run.output);
        run.report_path
    };
    let a_dir = TestDir::new();
    let a_corpus = imported_small_history(&a_dir);
    record(&a_dir, &a_corpus);
    let [a, a_again] = [(); 2].map(|()| replayed(&a_dir, &a_corpus));

    // The small history with one more turn at the end.
    let b_dir = TestDir::new();
    let b_corpus = import_history(&b_dir, |path| {
        let db = hermes::small_history(path);
        let at = epoch("2026-01-11T09:00:00Z");
        db.turn("s-later", at, "One more question.", "One more answer.");
        db
    });
    record(&b_dir, &b_corpus);
    let b = replayed(&b_dir, &b_corpus);

    let diff = |right: &Path, force: &[&str]| {
        let mut command = asphodel(&a_dir);
        command.args(["report", "diff"]).arg(&a).arg(right);
        command.args(force).output().unwrap()
    };
    assert_ok(&diff(&a_again, &[]));
    assert_refused(&diff(&b, &[]), "corpus");
    assert_ok(&diff(&b, &["--force"]));
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
    let report = replay_history(&dir, &corpus, "replay", PASSING_PROBES, None, &[]).ok();
    let id = report["memories"][0]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("the report lists each memory by id: {report}"))
        .to_string();

    let (mut a, mut b) = (report.clone(), report);
    for (pointer, in_a, in_b) in [
        (
            "/memories/0/faded_at",
            json!("2026-03-01T00:00:00Z"),
            json!(null),
        ),
        ("/memories/0/purged_at", json!(null), json!(null)),
        (
            "/injected_tokens/per_turn/p50",
            json!(100.0),
            json!(100.000_000_01),
        ),
        ("/injected_tokens/per_turn/p95", json!(100.0), json!(150.0)),
    ] {
        *a.pointer_mut(pointer).unwrap() = in_a;
        *b.pointer_mut(pointer).unwrap() = in_b;
    }
    let a_path = dir.private_json("reports/a.json", &a);
    let b_path = dir.private_json("reports/b.json", &b);

    let output = asphodel(&dir)
        .args(["report", "diff"])
        .arg(&a_path)
        .arg(&b_path)
        .output()
        .unwrap();
    assert_ok(&output);
    let diff: Value = serde_json::from_slice(&output.stdout).expect("the diff is JSON");
    assert_eq!(diff["memories"]["faded_only_in_a"], json!([id]), "{diff:#}");
    assert_eq!(diff["memories"]["faded_only_in_b"], json!([]), "{diff:#}");
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

// Refreshes. Every bank is seeded with "User profile";
// the corpus here adds `home` from the manifest, within the
// budget the profile leaves. The live stand-in answers every call with a
// reply call 1 and a refresh can both read, so the order of calls doesn't
// matter.

/// The probe fields for `home`'s entry citing the home memory.
const HOME_IN_HOME: &str = "model = \"home\"\nmemory = \"lives in Auckland\"";

/// `home` has no entry citing the home memory, at either of two days.
fn home_lacks_home() -> String {
    probe_on("p001", 6, "profile_lacks", HOME_IN_HOME)
        + &probe_on("p002", 10, "profile_lacks", HOME_IN_HOME)
}

fn home_has_home() -> String {
    probe_on("p001", 6, "profile_has", HOME_IN_HOME)
}

/// A `live` run on the corpus with a manifest model, in which both models
/// refresh and cite the home memory; returns its report.
fn record_with_models(dir: &TestDir, corpus: &Path) -> Value {
    let script = universal_script(dir);
    let in_profile = format!("model = \"User profile\"\n{HOME_MEMORY}");
    let probes = home_has_home() + &probe_on("p002", 6, "profile_has", &in_profile);
    replay_history(dir, corpus, "live", &probes, Some(&script), &[]).ok()
}

fn refresh_calls(report: &Value) -> u64 {
    report["refresh_calls_per_day"]
        .as_array()
        .unwrap_or_else(|| panic!("refresh calls per day: {report}"))
        .iter()
        .map(|day| day["count"].as_u64().unwrap())
        .sum()
}

/// `--refresh off` answers every write with nothing, so `home` stays
/// empty, but triggers are counted by code, so refresh calls per day match
/// the recording.
#[test]
fn fast_refresh_off_makes_no_edits_but_counts_every_trigger() {
    let dir = TestDir::new();
    let corpus = imported_with_a_model(&dir);
    let live = record_with_models(&dir, &corpus);
    assert!(refresh_calls(&live) > 0, "{live}");
    let flags = ["--refresh", "off"];
    let report = replay_history(&dir, &corpus, "fast", &home_lacks_home(), None, &flags).ok();
    assert_eq!(report["llm"]["misses"], 0, "{report}");
    assert_eq!(
        report["refresh_calls_per_day"], live["refresh_calls_per_day"],
        "{report}"
    );
}

/// A copy of refresh `record` `days` later, under a key of its own,
/// writing `sentences`.
fn refresh_copy(record: &Value, days: i64, sentences: &[Value], tag: &str) -> Value {
    let at: jiff::Timestamp = record["at"].as_str().unwrap().parse().unwrap();
    let mut copy = record.clone();
    copy["at"] = Value::from((at + jiff::SignedDuration::from_hours(24 * days)).to_string());
    copy["key"] = Value::from(format!("{}-{tag}", record["key"].as_str().unwrap()));
    copy["response"]["json"] = write_reply(sentences);
    copy
}

/// A `fast --refresh recorded` run whose probes all pass; returns its
/// report.
fn recorded_refreshes(dir: &TestDir, corpus: &Path, probes: &str, extra: &[&str]) -> Value {
    let mut flags = vec!["--refresh", "recorded"];
    flags.extend(extra);
    let run = replay_history(dir, corpus, "fast", probes, None, &flags);
    let report = run.report();
    let failed: Vec<&Value> = report["probes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|probe| probe["passed"] != true)
        .collect();
    assert!(failed.is_empty(), "failed probes: {failed:#?}");
    assert_ok(&run.output);
    report
}

/// `--refresh recorded` substitutes the recorded refresh nearest in
/// simulated time, whichever way round the two are. Every recorded
/// refresh is replaced by two copies: one 200 days later, written first so
/// file order can't pick it, and one at the refresh's own time.
#[test]
fn fast_refresh_recorded_substitutes_the_nearest_recorded_refresh() {
    let dir = TestDir::new();
    let corpus = imported_with_a_model(&dir);
    record_with_models(&dir, &corpus);
    let (refreshes, others): (Vec<Value>, Vec<Value>) =
        cassette_records(&dir).into_iter().partition(is_refresh);
    assert!(!refreshes.is_empty(), "the recording refreshed");
    let adds = [said("Tim lives in Auckland.", &["m1"])];
    for (near, far, probes) in [
        (&adds[..], &[][..], home_has_home()),
        (&[][..], &adds[..], home_lacks_home()),
    ] {
        let mut kept = others.clone();
        kept.extend(refreshes.iter().map(|r| refresh_copy(r, 200, far, "far")));
        kept.extend(refreshes.iter().map(|r| refresh_copy(r, 0, near, "near")));
        write_cassette(&dir, &kept);
        recorded_refreshes(&dir, &corpus, &probes, &[]);
    }
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
    let flags = ["--refresh", "live"];
    let probes = home_has_home();
    let report = replay_history(&dir, &corpus, "fast", &probes, Some(&script), &flags).ok();
    let calls = refresh_calls(&report);
    assert!(calls > 0, "{report}");
    assert_eq!(report["llm"]["live"], calls, "{report}");
    assert!(cassette_records(&dir).iter().any(is_refresh));
}

// Input validation, replay timing and cassette substitution.

/// Content is logged only at `trace`: a probes file or a corpus line that
/// doesn't parse is refused naming the file and the line, and a probe
/// whose regex doesn't compile naming the probe, never quoting either.
#[test]
fn malformed_inputs_are_refused_without_quoting_them() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);

    let sentinel = "SENTINEL-PROBE-QUERY-7a2";
    let probes = format!("{PASSING_PROBES}\n[[probe]]\nid = \"p003\"\nquery = {sentinel}\n");
    let line = probes.lines().position(|line| line.contains(sentinel));
    let line = format!("line {}", line.unwrap() + 1);
    let run = replay_history(&dir, &corpus, "replay", &probes, None, &[]);
    assert_refused_without(&run.output, sentinel, &["probes.toml", &line]);

    let probes = probe_on(
        "p001",
        5,
        "exists",
        "memory = \"SENTINEL-PROBE-PATTERN-((\"",
    );
    let run = replay_history(&dir, &corpus, "replay", &probes, None, &[]);
    assert_refused_without(&run.output, "SENTINEL-PROBE-PATTERN", &["p001"]);

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
    let run = replay_history(&dir, &corpus, "replay", PASSING_PROBES, None, &[]);
    assert_refused_without(&run.output, sentinel, &[&format!("line {}", line + 1)]);
}

/// With nonzero live latency, a prefetch between a turn's sync
/// and its completion doesn't see the turn's memory, and one after does.
/// Both prefetches ask the same question from sessions of their own; the
/// probes check the store at the same two moments.
#[test]
fn a_prefetch_before_a_live_completion_doesnt_see_the_turn() {
    let dir = TestDir::new();
    let t = start();
    let corpus = import_history(&dir, |path| {
        let db = StateDb::create(path);
        db.owner_session("s1", t);
        db.home_turn("s1", t);
        // Synced at t+30 with a 3 s call 1: complete at t+33.
        for (session, at) in [("s2", t + 31.0), ("s3", t + 40.0)] {
            db.owner_session(session, at);
            for (role, content, at) in [
                ("user", "Tim lives in Auckland near the harbour?", at),
                ("assistant", "Let me think.", at + 1.0),
            ] {
                db.message(hermes::Message {
                    session,
                    role,
                    content,
                    at,
                    ..Default::default()
                });
            }
        }
        db
    });

    let probes = probe("p001", "2026-01-05T09:00:31Z", "absent", HOME_MEMORY)
        + &probe("p002", "2026-01-05T09:00:40Z", "exists", HOME_MEMORY);
    let script = support::delayed_script(&dir, 3000);
    let run = replay_history(&dir, &corpus, "live", &probes, Some(&script), &[]);
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
    let early = probe_in(&report, "p001");
    assert_eq!(early["passed"], false, "{early}");
    assert_eq!(early["observed"]["resolved"], false, "{early}");
    let later = probe_in(&report, "p002");
    assert_eq!(later["passed"], true, "{later}");
    assert_eq!(
        run.output.status.code(),
        Some(1),
        "the pre-creation probe is unresolved: {}",
        stderr(&run.output)
    );
}

/// `--no-cache` re-records from scratch, so re-recording leaves
/// one record per call, and the replay's lag is one call's latency, not
/// the sum over every recording.
#[test]
fn re_recording_with_no_cache_starts_a_fresh_cassette() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let script = support::delayed_script(&dir, 300);
    replay_history(&dir, &corpus, "live", PASSING_PROBES, Some(&script), &[]).ok();
    let recorded = cassette_records(&dir).len();
    let mut last = Value::Null;
    for _ in 0..2 {
        let flags = ["--no-cache"];
        last = replay_history(&dir, &corpus, "live", PASSING_PROBES, Some(&script), &flags).ok();
    }
    assert_eq!(
        cassette_records(&dir).len(),
        recorded,
        "a re-recording replaces the cassette"
    );
    let replay = replay_history(&dir, &corpus, "replay", PASSING_PROBES, None, &[]).ok();
    assert_eq!(replay["extraction_lag"], last["extraction_lag"]);
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
    let live = replay_history(&dir, &corpus, "live", PASSING_PROBES, Some(&script), &[]).ok();
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

    let replay = replay_history(&dir, &corpus, "replay", PASSING_PROBES, None, &[]).ok();
    assert_eq!(replay["extraction_lag"], live["extraction_lag"]);
}

const PASSPORT_QUOTE: &str = "I need to renew my passport";

/// A history where a passport task (A) is remembered on day 0 and the home
/// fact (B) on day 3, imported twice: `home` filtered to facts and tasks
/// (`corpus/both.jsonl`), and to facts only (`corpus/facts.jsonl`), so A is
/// out of its input. The chunks are the same in both.
fn identity_corpora(dir: &TestDir) -> (PathBuf, PathBuf) {
    let state_db = dir.private_path("state.db");
    let db = hermes::one_session(&state_db);
    let t = start();
    db.turn("s1", t, &format!("{PASSPORT_QUOTE} soon."), "Noted.");
    db.home_turn("s1", t + 3.0 * hermes::DAY);
    db.turn("s1", t + 5.0 * hermes::DAY, "Anything else?", "No.");
    drop(db);
    let corpora = [("both", r#"["fact", "task"]"#), ("facts", r#"["fact"]"#)].map(|(name, kinds)| {
        let corpus = dir.private_path(&format!("corpus/{name}.jsonl"));
        let manifest = format!(
            "{}\n[[model]]\nname = \"home\"\nquestion = \"{}\"\nkinds = {kinds}\nmax_tokens = 100\n",
            hermes::MANIFEST,
            support::HOME_QUESTION
        );
        assert_ok(&support::import_with(dir, &state_db, &corpus, &manifest, &[]));
        corpus
    });
    let [both, facts] = corpora;
    (both, facts)
}

/// A `live` run of the identity history on `both` with every write
/// answering one sentence citing `m1`; returns the cassette and the ids
/// of A and B, resolved by probes.
fn record_identities(dir: &TestDir, both: &Path) -> (Vec<Value>, [Value; 2]) {
    let script = script_answering_everything(
        dir,
        "identity-script",
        vec![
            claim("Tim needs to renew his passport.", PASSPORT_QUOTE, "task"),
            home_claim(),
        ],
        vec![said("Tim lives in Auckland.", &["m1"])],
    );
    let probes = probe_on("a", 5, "exists", "memory = \"renew his passport\"")
        + &probe_on("b", 8, "exists", HOME_MEMORY);
    let report = replay_history(dir, both, "live", &probes, Some(&script), &[]).report();
    let ids = ["a", "b"].map(|probe| observed_id(&report, probe));
    assert!(ids.iter().all(Value::is_string), "{report}");
    (cassette_records(dir), ids)
}

/// The handles a write record's request used, each with the memory it
/// stood for, in handle order.
fn identities(record: &Value) -> Vec<(String, Value)> {
    let mut pairs: Vec<(String, Value)> = record["identities"]
        .as_array()
        .map(|pairs| {
            pairs
                .iter()
                .map(|pair| (pair[0].as_str().unwrap().to_string(), pair[1].clone()))
                .collect()
        })
        .unwrap_or_default();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    pairs
}

/// A substituted write is carried over whole, by what each handle stood
/// for when recorded, not by the handle's name now, and only when every
/// memory it cites is in this run's input: one that isn't would leave the
/// text resting on something the LLM never saw here.
///
/// The cassette keeps one of `home`'s writes (`home` is the only model
/// shown the passport task A), rewritten to cite `cites`, and drops the
/// rest:
/// - the one recorded when A was its one memory, citing `m1`, which meant
///   A. On `both` it cites A, not the home fact B that `m1` names there
///   now; on `facts`, where A isn't in the input, it isn't written.
/// - the one recorded with B as `m1` and A as `m2`, citing both. On
///   `both` it's written; on `facts` it isn't, though B is there.
#[test]
fn a_substituted_refresh_cites_the_memories_it_was_recorded_with_or_isnt_written() {
    let dir = TestDir::new();
    let (both, facts) = identity_corpora(&dir);
    let (records, [a, b]) = record_identities(&dir, &both);
    let is_home =
        |record: &Value| is_refresh(record) && identities(record).iter().any(|(_, id)| *id == a);
    let keep_only = |handles: &[(String, Value)], cites: &[&str]| {
        let reply = write_reply(&[said("Tim has a passport and a home.", cites)]);
        let kept: Vec<Value> = records
            .iter()
            .filter(|record| !is_home(record) || identities(record) == handles)
            .map(|record| {
                let mut record = record.clone();
                if is_home(&record) {
                    record["response"]["json"] = reply.clone();
                }
                record
            })
            .collect();
        assert!(
            kept.iter().any(is_home),
            "fixture: home wrote with {handles:?}"
        );
        write_cassette(&dir, &kept);
    };
    let lacks_b = probe_on("p001", 9, "profile_lacks", HOME_IN_HOME);
    let has_b = probe_on("p001", 9, "profile_has", HOME_IN_HOME);
    let has_a = probe_on(
        "p002",
        9,
        "profile_has",
        "model = \"home\"\nmemory = \"renew his passport\"",
    );

    keep_only(&[("m1".into(), a.clone())], &["m1"]);
    recorded_refreshes(&dir, &both, &format!("{lacks_b}{has_a}"), &[]);
    recorded_refreshes(&dir, &facts, &lacks_b, &[]);

    keep_only(
        &[("m1".into(), b.clone()), ("m2".into(), a.clone())],
        &["m1", "m2"],
    );
    recorded_refreshes(&dir, &both, &format!("{has_b}{has_a}"), &[]);
    recorded_refreshes(&dir, &facts, &lacks_b, &[]);
}

/// Nearest-refresh substitution only considers records of the run's own
/// LLM model. A record under another model at the very time of each
/// refresh loses to the run's own model's record 200 days off.
#[test]
fn a_refresh_recorded_under_another_model_is_never_substituted() {
    let dir = TestDir::new();
    let corpus = imported_with_a_model(&dir);
    record_with_models(&dir, &corpus);
    let (refreshes, others): (Vec<Value>, Vec<Value>) =
        cassette_records(&dir).into_iter().partition(is_refresh);
    let adds = [said("Tim lives in Auckland.", &["m1"])];
    // The other model's records go first, so the run's model, which
    // `replay` and `fast` take from the last record, stays the original.
    let mut kept: Vec<Value> = refreshes
        .iter()
        .map(|record| {
            let mut other = refresh_copy(record, 0, &adds, "other");
            other["model"] = Value::from("other-model");
            other
        })
        .collect();
    kept.extend(others);
    kept.extend(refreshes.iter().map(|r| refresh_copy(r, 200, &[], "far")));
    write_cassette(&dir, &kept);
    recorded_refreshes(&dir, &corpus, &home_lacks_home(), &[]);
}

// Attempt budgets.

/// An allocations file: `stages` as `(id, cap)` and `runs` as
/// `(id, stage, cap)`, a run with no cap of its own taking from its
/// stage's alone.
fn allocations(stages: &[(&str, u64)], runs: &[(&str, &str, Option<u64>)]) -> String {
    let mut toml = String::new();
    for (id, cap) in stages {
        toml += &format!("[[stage]]\nid = \"{id}\"\ncap = {cap}\n\n");
    }
    for (id, stage, cap) in runs {
        toml += &format!("[[run]]\nid = \"{id}\"\nstage = \"{stage}\"\n");
        if let Some(cap) = cap {
            toml += &format!("cap = {cap}\n");
        }
        toml += "\n";
    }
    toml
}

/// `asphodel replay-budget init` of `allocations` into the ledger `name`
/// under the private dir: the ledger's path and the command's output.
fn budget_init(dir: &TestDir, name: &str, allocations: &str) -> (PathBuf, std::process::Output) {
    let file = dir.private_file(&format!("{name}.allocations.toml"), allocations);
    let ledger = dir.private_path(name);
    let output = asphodel(dir)
        .args(["replay-budget", "init", "--allocations"])
        .arg(&file)
        .arg("--ledger")
        .arg(&ledger)
        .output()
        .unwrap();
    (ledger, output)
}

/// `asphodel replay-budget show` of `ledger`: its stdout as text, which
/// must parse as JSON.
fn budget_show(dir: &TestDir, ledger: &Path) -> (String, Value) {
    let output = asphodel(dir)
        .args(["replay-budget", "show", "--ledger"])
        .arg(ledger)
        .output()
        .unwrap();
    assert_ok(&output);
    let text = support::stdout(&output);
    let shown = serde_json::from_str(&text).unwrap_or_else(|error| panic!("{error}: {text}"));
    (text, shown)
}

/// The path beside `ledger` with `suffix` added to its name.
fn beside(ledger: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{suffix}", ledger.display()))
}

/// A run of `mode` on `corpus` under `ledger` as `run`, whose LLM is a
/// Codex stub failing its first `failures` requests as overloaded, with
/// the reranker gate shut, `tuning` added to the overrides and `extra`
/// flags: the command, to run or spawn, and the stub's request count,
/// which is every backend attempt the run makes.
fn budgeted(
    dir: &TestDir,
    corpus: &Path,
    mode: &str,
    ledger: (&Path, &str),
    failures: usize,
    tuning: &str,
    extra: &[&str],
) -> (std::process::Command, Arc<AtomicUsize>) {
    let reply = support::reply_to_everything(vec![home_claim()], vec![]);
    let (url, requests) = codex_failing("server_is_overloaded", failures, &reply);
    let command = budgeted_against(dir, corpus, mode, ledger, &url, tuning, extra);
    (command, requests)
}

/// [`budgeted`] against the Codex stand-in at `url`, logged in with a
/// token that isn't due for refresh.
fn budgeted_against(
    dir: &TestDir,
    corpus: &Path,
    mode: &str,
    (ledger, run): (&Path, &str),
    url: &str,
    tuning: &str,
    extra: &[&str],
) -> std::process::Command {
    let token_dir = dir.private_path("tokens");
    logged_in(&token_dir);
    let llm = format!("[llm]\nauth = \"chatgpt\"\nendpoint = \"{url}\"\nmodel = \"gpt-test\"\n");
    let probes = dir.private_file("probes.toml", PASSING_PROBES);
    let mut command = asphodel(dir);
    command
        .arg("replay")
        .arg("--corpus")
        .arg(corpus)
        .args(["--mode", mode])
        .arg("--cassette")
        .arg(cassette_path(dir))
        .arg("--probes")
        .arg(&probes)
        .args([
            "--overrides",
            &overrides(dir, &format!("{llm}{SHUT_GATE}{tuning}")),
        ])
        .arg("--token-dir")
        .arg(&token_dir)
        .arg("--attempt-budget")
        .arg(ledger)
        .args(["--budget-run", run])
        .args(extra)
        .env("ASPHODEL_LLM_RETRY_WAIT_MS", "0");
    command
}

/// Priming 4 chunks at a time, with no refreshes, in `fast`.
const PRIMED: [&str; 4] = ["--refresh", "off", "--prime-concurrency", "4"];

/// Every backend attempt a budgeted run makes takes a slot from the
/// ledger first, retries and priming workers included. A run capped at 3
/// that wants more makes exactly 3 requests, though 4 calls go out at once
/// and the first fails and is retried. Running out stops the run with
/// exit 2 and keeps the 2 replies that succeeded. `show` counts all 3
/// attempts against the run, its stage and call 1's template, and names
/// no path, since the ledger binds a run to its cassette's.
#[test]
fn a_budgeted_run_makes_exactly_its_cap_in_attempts_across_priming_and_retries() {
    assert!(
        serial_call1_chunks().len() > 3,
        "the run must want more than its cap"
    );
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let budget = allocations(&[("s1", 3)], &[("r1", "s1", Some(3))]);
    let (ledger, init) = budget_init(&dir, "ledger.json", &budget);
    assert_ok(&init);

    let (mut run, requests) = budgeted(&dir, &corpus, "fast", (&ledger, "r1"), 1, "", &PRIMED);
    let output = run.output().unwrap();
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert_eq!(requests.load(Ordering::SeqCst), 3);
    assert_eq!(call1_chunks(&dir).len(), 2, "the successful replies stay");

    let (text, shown) = budget_show(&dir, &ledger);
    assert_eq!(shown["runs"]["r1"]["consumed"], 3, "{shown}");
    assert_eq!(
        shown["runs"]["r1"]["by_template"]["extract_claims"], 3,
        "{shown}"
    );
    assert_eq!(shown["stages"]["s1"]["consumed"], 3, "{shown}");
    let root = dir.private().parent().unwrap().to_str().unwrap().to_owned();
    assert!(!text.contains(&root), "show names a path: {text}");
}

/// A budget belongs to the ledger, not to an invocation. Run again, an
/// exhausted run makes no request; a second `init` can't replace the
/// ledger; and two runs started together in one stage make, between them,
/// exactly the stage's cap however they interleave.
#[test]
fn a_resumed_or_concurrent_run_cannot_spend_past_its_ledger() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let budget = allocations(
        &[("s1", 2), ("s2", 3)],
        &[("r1", "s1", Some(2)), ("a", "s2", None), ("b", "s2", None)],
    );
    let (ledger, init) = budget_init(&dir, "ledger.json", &budget);
    assert_ok(&init);

    for spent in [2, 0] {
        let (mut run, requests) = budgeted(&dir, &corpus, "fast", (&ledger, "r1"), 0, "", &PRIMED);
        let output = run.output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
        assert_eq!(requests.load(Ordering::SeqCst), spent);
    }

    let (_, again) = budget_init(&dir, "ledger.json", &budget);
    assert_eq!(again.status.code(), Some(2), "{}", stderr(&again));
    let (_, shown) = budget_show(&dir, &ledger);
    assert_eq!(shown["runs"]["r1"]["consumed"], 2, "{shown}");

    // Each in a directory of its own, wanting every chunk, so each wants
    // more than the whole stage holds and both run out.
    let (a_dir, b_dir) = (TestDir::new(), TestDir::new());
    let (a_corpus, b_corpus) = (
        imported_small_history(&a_dir),
        imported_small_history(&b_dir),
    );
    let (mut a, a_requests) = budgeted(&a_dir, &a_corpus, "fast", (&ledger, "a"), 0, "", &PRIMED);
    let (mut b, b_requests) = budgeted(&b_dir, &b_corpus, "fast", (&ledger, "b"), 0, "", &PRIMED);
    let (a, b) = (a.spawn().unwrap(), b.spawn().unwrap());
    for output in [a.wait_with_output().unwrap(), b.wait_with_output().unwrap()] {
        assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    }
    let total = a_requests.load(Ordering::SeqCst) + b_requests.load(Ordering::SeqCst);
    assert_eq!(total, 3);
    let (_, shown) = budget_show(&dir, &ledger);
    assert_eq!(shown["stages"]["s2"]["consumed"], 3, "{shown}");
}

/// A run is bound to what it first ran with: run again with other
/// overrides, it is refused before calling anything, though its budget
/// has room. A call the cassette answers never takes a slot, so a `live`
/// run of a fully recorded history completes under a cap of 0.
#[test]
fn a_budgeted_run_is_bound_to_its_first_configuration_and_cache_hits_are_free() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let budget = allocations(
        &[("s1", 1000)],
        &[("recorder", "s1", Some(1000)), ("cached", "s1", Some(0))],
    );
    let (ledger, init) = budget_init(&dir, "ledger.json", &budget);
    assert_ok(&init);

    let (mut run, requests) = budgeted(&dir, &corpus, "live", (&ledger, "recorder"), 0, "", &[]);
    assert_ok(&run.output().unwrap());
    let recorded = requests.load(Ordering::SeqCst);
    assert!(recorded > 0);

    let (mut run, requests) = budgeted(&dir, &corpus, "live", (&ledger, "cached"), 0, "", &[]);
    assert_ok(&run.output().unwrap());
    assert_eq!(requests.load(Ordering::SeqCst), 0);

    // Without its cassette the run would have to call again.
    fs::remove_file(cassette_path(&dir)).unwrap();
    let other = "[clock]\nquiet_rate = 0.2\n";
    let (mut run, requests) = budgeted(&dir, &corpus, "live", (&ledger, "recorder"), 0, other, &[]);
    let output = run.output().unwrap();
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert_eq!(requests.load(Ordering::SeqCst), 0);
    let (_, shown) = budget_show(&dir, &ledger);
    assert_eq!(shown["runs"]["recorder"]["consumed"], recorded, "{shown}");
}

/// A ledger that can't be trusted stops a run before any request: one
/// that's missing, one without its lock file, one that doesn't parse, one
/// whose stage count disagrees with its runs', and one in a directory a
/// commit can't be written to. Nothing falls back to calling unbudgeted.
#[test]
fn a_ledger_that_cannot_be_trusted_dispatches_nothing() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let budget = allocations(&[("s1", 100)], &[("r1", "s1", None)]);
    let fresh = |name: &str| {
        let (ledger, init) = budget_init(&dir, name, &budget);
        assert_ok(&init);
        ledger
    };

    let missing = dir.private_path("missing.json");
    let unlocked = fresh("unlocked.json");
    fs::remove_file(beside(&unlocked, ".lock")).unwrap();
    let malformed = fresh("malformed.json");
    fs::write(&malformed, "{").unwrap();
    let disagreeing = fresh("disagreeing.json");
    let mut counts: Value = serde_json::from_slice(&fs::read(&disagreeing).unwrap()).unwrap();
    counts["stages"]["s1"]["consumed"] = json!(1);
    fs::write(&disagreeing, counts.to_string()).unwrap();
    let mut ledgers = vec![missing, unlocked, malformed, disagreeing];

    let read_only = fresh("read-only/ledger.json");
    let parent = read_only.parent().unwrap().to_owned();
    set_mode(&parent, 0o555);
    // Root writes anyway, so the case means nothing there.
    let writable = fs::write(parent.join("probe"), "").is_ok();
    if !writable {
        ledgers.push(read_only);
    }

    for ledger in &ledgers {
        let (mut run, requests) = budgeted(&dir, &corpus, "live", (ledger, "r1"), 0, "", &[]);
        let output = run.output().unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{}: {}",
            ledger.display(),
            stderr(&output)
        );
        assert_eq!(requests.load(Ordering::SeqCst), 0, "{}", ledger.display());
    }
    set_mode(&parent, 0o755);
}

fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
}

/// A slot is one post to the model. With ChatGPT auth, a 401 makes the
/// client refresh the login and post again within the one call; that post
/// takes a slot of its own, and the refresh, which goes to the issuer
/// rather than the model, takes none. So a run capped at 1 whose first
/// post gets a 401 refreshes the login, sends the model nothing more, and
/// stops with exit 2, and `show` counts the 1 attempt against the run,
/// its stage and call 1's template.
#[test]
fn a_post_after_a_401_refresh_takes_a_slot_of_its_own() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let budget = allocations(&[("s1", 1)], &[("r1", "s1", Some(1))]);
    let (ledger, init) = budget_init(&dir, "ledger.json", &budget);
    assert_ok(&init);
    let reply = support::reply_to_everything(vec![home_claim()], vec![]);
    let (model, requests) = codex_unauthorized_once(&reply);
    let (issuer, refreshes) = issuer_refreshing();

    let mut run = budgeted_against(&dir, &corpus, "live", (&ledger, "r1"), &model, "", &[]);
    let output = run.env("ASPHODEL_LLM_ISSUER", &issuer).output().unwrap();
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert_eq!(
        refreshes.load(Ordering::SeqCst),
        1,
        "the 401 refreshes the login"
    );
    assert_eq!(
        requests.load(Ordering::SeqCst),
        1,
        "the post after the refresh needs a slot of its own"
    );

    let (_, shown) = budget_show(&dir, &ledger);
    assert_eq!(shown["runs"]["r1"]["consumed"], 1, "{shown}");
    assert_eq!(shown["stages"]["s1"]["consumed"], 1, "{shown}");
    assert_eq!(
        shown["runs"]["r1"]["by_template"],
        json!({"extract_claims": 1}),
        "{shown}"
    );
}

/// The ledger only changes by a complete commit replacing it, and a call
/// goes out only after its commit lands. So a commit interrupted before it
/// landed leaves the last one standing: a stale `.tmp` beside the ledger,
/// here claiming the whole cap spent, changes nothing, and the run spends
/// its cap from the ledger's own count.
#[test]
fn an_interrupted_commit_leaves_the_last_ledger_standing() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let budget = allocations(&[("s1", 2)], &[("r1", "s1", Some(2))]);
    let (ledger, init) = budget_init(&dir, "ledger.json", &budget);
    assert_ok(&init);
    let mut stale: Value = serde_json::from_slice(&fs::read(&ledger).unwrap()).unwrap();
    stale["runs"]["r1"]["consumed"] = json!(2);
    stale["runs"]["r1"]["by_template"] = json!({"extract_claims": 2});
    stale["stages"]["s1"]["consumed"] = json!(2);
    fs::write(beside(&ledger, ".tmp"), stale.to_string()).unwrap();

    let (mut run, requests) = budgeted(&dir, &corpus, "fast", (&ledger, "r1"), 0, "", &PRIMED);
    let output = run.output().unwrap();
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert_eq!(requests.load(Ordering::SeqCst), 2);
    let (_, shown) = budget_show(&dir, &ledger);
    assert_eq!(shown["runs"]["r1"]["consumed"], 2, "{shown}");
}
