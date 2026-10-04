//! Numbers-only comparison of call 1 cassettes through the CLI.
//! These fixtures are synthetic; no corpus, LLM or imported history is needed.
mod support;

use std::fs;
use std::process::Output;

use serde_json::{Value, json};
use support::{TestDir, asphodel, assert_ok, claim};

const SOURCE_A: &str = "11111111-1111-4111-8111-111111111111";
const SOURCE_B: &str = "22222222-2222-4222-8222-222222222222";
const SECRET: &str = "SYNTHETIC_CLAIM_TEXT_MUST_NOT_ESCAPE";

fn call1(source: &str, position: u32, kinds: &[&str], primed: bool) -> Value {
    let template = json!({"name": "extract_claims", "version": 5});
    let claims = Vec::from_iter(kinds.iter().map(|kind| claim(SECRET, SECRET, kind)));
    json!({
        // Different request keys/context are expected for serial and primed calls.
        "key": format!("{source}-{position}-{primed}"), "model": "synthetic-model",
        "template": template, "at": "2026-01-01T00:00:00Z", "latency_ms": 1,
        "chunk": {"source": source, "position": position}, "primed": primed,
        "in_context": if primed {json!([])} else {json!([{"handle": "m1", "sentence": "opaque-hash"}])},
        "request": {"template": template, "system": SECRET, "user": SECRET,
            "schema_name": "claims", "schema": {}, "max_tokens": 100},
        "response": {"json": {"claims": claims, "used_injected_ids": []},
            "usage": null, "latency": {"secs": 0, "nanos": 1000000}}
    })
}

fn cassette(dir: &TestDir, name: &str, records: &[Value]) {
    let text: String = records.iter().map(|record| format!("{record}\n")).collect();
    fs::write(dir.private_path(name), text).unwrap();
}

const SERIAL: &str = "cassettes/serial.jsonl";
const PRIMED: &str = "cassettes/primed.jsonl";

/// Both cassettes' bytes.
fn cassettes(dir: &TestDir) -> [Vec<u8>; 2] {
    [SERIAL, PRIMED].map(|name| fs::read(dir.private_path(name)).unwrap())
}

/// Compares the two cassettes, which it leaves as they were.
fn compare(dir: &TestDir) -> Output {
    let before = cassettes(dir);
    let output = compare_paths(dir, SERIAL, PRIMED);
    assert_eq!(cassettes(dir), before);
    output
}

fn compare_paths(dir: &TestDir, serial: &str, primed: &str) -> Output {
    asphodel(dir)
        .args(["report", "claims-agreement"])
        .args(["--serial", serial, "--primed", primed])
        .output()
        .unwrap()
}

fn report(output: &Output) -> Value {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    for text in [&stdout, &stderr] {
        assert!(!text.contains(SECRET), "claim or prompt text escaped");
        assert!(
            !text.contains(SOURCE_A) && !text.contains(SOURCE_B),
            "chunk identities escaped"
        );
    }
    assert_ok(output);
    serde_json::from_slice(&output.stdout).expect("one JSON report on stdout")
}

/// Chunks are compared by identity, whatever order each cassette holds
/// them in: a chunk agrees on its claim count, and on its multiset of
/// kinds, never on the claims' text or order. Chunks only one side holds
/// are counted apart and left out of the totals, and records other than
/// call 1's, or without a chunk, are ignored. An empty cassette, or one of
/// blank lines, compares nothing.
#[test]
fn compares_chunks_by_identity_and_counts_kind_multisets_not_text_or_order() {
    let mut reordered = call1(SOURCE_A, 0, &["preference", "fact", "fact"], true);
    reordered["response"]["json"]["claims"][0]["content"] = json!("different synthetic wording");
    let mut top_up = call1(SOURCE_A, 8, &["fact"], false);
    top_up["template"]["name"] = json!("judge_used");
    top_up["request"]["template"]["name"] = json!("judge_used");
    top_up["response"]["json"] = json!({"used": []});
    let mut without_chunk = call1(SOURCE_A, 9, &["fact"], false);
    without_chunk["chunk"] = Value::Null;
    let nothing_compared = |serial: u64, primed: u64| {
        json!({
            "chunks": {
                "serial": serial, "primed": primed, "compared": 0,
                "serial_only": serial, "primed_only": primed
            },
            "agreement": {"claim_count": 0, "kind_multiset": 0},
            "claims": {"serial": 0, "primed": 0},
            "kinds": {"serial": {}, "primed": {}}
        })
    };
    let cases = [
        (
            vec![
                call1(SOURCE_A, 0, &["fact", "fact", "preference"], false),
                call1(SOURCE_A, 1, &["fact", "fact", "preference"], false),
                call1(SOURCE_A, 2, &["fact"], false),
                call1(SOURCE_A, 3, &[], false),
            ],
            vec![
                call1(SOURCE_A, 3, &[], true),
                call1(SOURCE_A, 2, &["fact", "preference"], true),
                reordered,
                // Equal count and equal kind SET, but unequal kind multiplicities.
                call1(SOURCE_A, 1, &["fact", "preference", "preference"], true),
            ],
            json!({
                "chunks": {"serial": 4, "primed": 4, "compared": 4, "serial_only": 0, "primed_only": 0},
                "agreement": {"claim_count": 3, "kind_multiset": 2},
                "claims": {"serial": 7, "primed": 8},
                "kinds": {"serial": {"fact": 5, "preference": 2}, "primed": {"fact": 4, "preference": 4}}
            }),
        ),
        (
            vec![
                call1(SOURCE_A, 0, &["fact"], false),
                call1(SOURCE_A, 1, &["preference"], false),
                top_up,
                without_chunk,
            ],
            vec![
                call1(SOURCE_B, 1, &["fact", "fact"], true),
                call1(SOURCE_A, 0, &["fact"], true),
                call1(SOURCE_A, 2, &["preference"], true),
            ],
            json!({
                "chunks": {"serial": 2, "primed": 3, "compared": 1, "serial_only": 1, "primed_only": 2},
                "agreement": {"claim_count": 1, "kind_multiset": 1},
                "claims": {"serial": 1, "primed": 1},
                "kinds": {"serial": {"fact": 1}, "primed": {"fact": 1}}
            }),
        ),
        (
            vec![call1(SOURCE_A, 0, &["fact"], false)],
            vec![call1(SOURCE_B, 0, &["task"], true)],
            nothing_compared(1, 1),
        ),
        (vec![], vec![], nothing_compared(0, 0)),
    ];
    for (serial, primed, expected) in cases {
        let dir = TestDir::new();
        cassette(&dir, SERIAL, &serial);
        cassette(&dir, PRIMED, &primed);
        if serial.is_empty() {
            fs::write(dir.private_path(SERIAL), b"\n \t\n").unwrap();
        }
        assert_eq!(report(&compare(&dir)), expected);
    }
}

fn refused(output: &Output, reason: &str, dir: &TestDir) {
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty(), "no partial report on failure");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(reason), "{stderr}");
    for forbidden in [SECRET, SOURCE_A, SOURCE_B, dir.path("").to_str().unwrap()] {
        assert!(
            !stderr.contains(forbidden),
            "private input escaped: {stderr}"
        );
    }
}

/// A cassette that can't be compared is refused with no partial report
/// and nothing of its input in the error: a chunk recorded twice on either
/// side, even identically; records whose extraction metadata differs,
/// within a cassette or between the two; a record whose request names
/// another template than the record; and a record or claim that doesn't
/// parse.
#[test]
fn refuses_cassettes_it_cant_compare_without_exposing_their_input() {
    const INCOMPATIBLE: &str = "incompatible extraction metadata";
    let line = |record: &Value| format!("{record}\n");
    let fact = || call1(SOURCE_A, 0, &["fact"], false);
    let bad = |field: &str| match field {
        "version" => json!(999),
        _ => json!(SECRET),
    };
    let mut cases: Vec<(String, String, &str)> = Vec::new();
    let mut either_side = |text: String, reason: &'static str| {
        cases.push((text.clone(), String::new(), reason));
        cases.push((String::new(), text, reason));
    };
    for second in [fact(), call1(SOURCE_A, 0, &["preference"], true)] {
        either_side(line(&fact()) + &line(&second), "duplicate call 1 chunk");
    }
    let mut incompatible = Vec::new();
    for field in ["version", "guidance", "model", "language"] {
        let mut changed = call1(SOURCE_B, 1, &["fact"], true);
        if field == "model" || field == "language" {
            changed[field] = bad(field);
        } else {
            changed["template"][field] = bad(field);
            changed["request"]["template"][field] = bad(field);
        }
        either_side(line(&fact()) + &line(&changed), INCOMPATIBLE);
        incompatible.push(changed);
    }
    for field in ["name", "version", "guidance"] {
        let mut record = fact();
        record["request"]["template"][field] = bad(field);
        either_side(line(&record), "inconsistent extraction metadata");
    }
    either_side(format!("{{{SECRET}\n"), "invalid");
    either_side(line(&json!({"private": SECRET})), "invalid");
    let mut claims = fact();
    claims["response"]["json"]["claims"] = json!(SECRET);
    either_side(line(&claims), "invalid claims");
    for kind in [Some(json!(SECRET)), None, Some(json!({"private": SECRET}))] {
        let mut record = fact();
        let claim = record["response"]["json"]["claims"][0]
            .as_object_mut()
            .unwrap();
        match kind {
            Some(kind) => claim.insert("kind".into(), kind),
            None => claim.remove("kind"),
        };
        either_side(line(&record), "invalid claim kind");
    }
    for changed in incompatible {
        cases.push((line(&fact()), line(&changed), INCOMPATIBLE));
    }
    for (serial, primed, reason) in cases {
        let dir = TestDir::new();
        fs::write(dir.private_path(SERIAL), serial).unwrap();
        fs::write(dir.private_path(PRIMED), primed).unwrap();
        refused(&compare(&dir), reason, &dir);
    }
}

#[test]
fn refuses_paths_outside_private_directory_without_reading_or_changing_them() {
    let dir = TestDir::new();
    cassette(&dir, SERIAL, &[]);
    cassette(&dir, PRIMED, &[]);
    let outside = dir.file(SECRET, SECRET);
    let before = cassettes(&dir);
    std::os::unix::fs::symlink(&outside, dir.private_path("cassettes/escape.jsonl")).unwrap();
    let outside_path = outside.to_str().unwrap().to_owned();
    for path in [
        outside_path,
        format!("../{SECRET}"),
        "cassettes/escape.jsonl".into(),
    ] {
        for (serial, primed) in [(path.as_str(), PRIMED), (SERIAL, path.as_str())] {
            refused(
                &compare_paths(&dir, serial, primed),
                "inside the private replay directory",
                &dir,
            );
            assert_eq!(fs::read(&outside).unwrap(), SECRET.as_bytes());
            assert_eq!(cassettes(&dir), before);
        }
    }
}
