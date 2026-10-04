//! Numbers-only comparison of call 1 cassettes through the CLI.
//! These fixtures are synthetic; no corpus, LLM or imported history is needed.
mod support;

use std::fs;
use std::process::Output;

use asphodel::replay::cassette::Record;
use serde_json::{Value, json};
use support::{TestDir, asphodel, assert_ok};

const SOURCE_A: &str = "11111111-1111-4111-8111-111111111111";
const SOURCE_B: &str = "22222222-2222-4222-8222-222222222222";
const SECRET: &str = "SYNTHETIC_CLAIM_TEXT_MUST_NOT_ESCAPE";

fn call1(source: &str, position: u32, kinds: &[&str], primed: bool) -> Value {
    let template = json!({"name": "extract_claims", "version": 5});
    let claims: Vec<Value> = kinds
        .iter()
        .map(|kind| {
            json!({
                "content": SECRET, "quote": SECRET, "kind": kind,
                "significance": "notable", "remember_this": false,
                "changes_something": false, "valid_from": null, "valid_until": null,
                "window_confidence": "high", "until_event": null, "due_at": null,
                "volatility": null, "recurrence_text": null, "recurrence_rrule": null,
                "recurrence_start": null, "entities": []
            })
        })
        .collect();
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
    let lines: Vec<String> = records
        .iter()
        .map(|record| {
            // Check that hand-written fixtures are real cassette records.
            serde_json::from_value::<Record>(record.clone()).expect("valid synthetic record");
            serde_json::to_string(record).unwrap()
        })
        .collect();
    fs::write(
        dir.private_path(name),
        format!(
            "{}
",
            lines.join(
                "
"
            )
        ),
    )
    .unwrap();
}

fn compare(dir: &TestDir) -> Output {
    asphodel(dir)
        .args([
            "report",
            "claims-agreement",
            "--serial",
            "cassettes/serial.jsonl",
            "--primed",
            "cassettes/primed.jsonl",
        ])
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

#[test]
fn compares_chunks_by_identity_and_counts_kind_multisets_not_text_or_order() {
    let dir = TestDir::new();
    cassette(
        &dir,
        "cassettes/serial.jsonl",
        &[
            call1(SOURCE_A, 0, &["fact", "fact", "preference"], false),
            call1(SOURCE_A, 1, &["fact", "fact", "preference"], false),
            call1(SOURCE_A, 2, &["fact"], false),
            call1(SOURCE_A, 3, &[], false),
        ],
    );
    let mut reordered = call1(SOURCE_A, 0, &["preference", "fact", "fact"], true);
    reordered["response"]["json"]["claims"][0]["content"] = json!("different synthetic wording");
    cassette(
        &dir,
        "cassettes/primed.jsonl",
        &[
            call1(SOURCE_A, 3, &[], true),
            call1(SOURCE_A, 2, &["fact", "preference"], true),
            reordered,
            // Equal count and equal kind SET, but unequal kind multiplicities.
            call1(SOURCE_A, 1, &["fact", "preference", "preference"], true),
        ],
    );
    assert_eq!(
        report(&compare(&dir)),
        json!({
            "chunks": {"serial": 4, "primed": 4, "compared": 4, "serial_only": 0, "primed_only": 0},
            "agreement": {"claim_count": 3, "kind_multiset": 2},
            "claims": {"serial": 7, "primed": 8},
            "kinds": {"serial": {"fact": 5, "preference": 2}, "primed": {"fact": 4, "preference": 4}}
        })
    );
}

#[test]
fn unmatched_chunks_are_counted_separately_and_non_call1_records_are_ignored() {
    let dir = TestDir::new();
    let mut top_up = call1(SOURCE_A, 8, &["fact"], false);
    top_up["template"]["name"] = json!("judge_used");
    top_up["request"]["template"]["name"] = json!("judge_used");
    top_up["response"]["json"] = json!({"used": []});
    let mut without_chunk = call1(SOURCE_A, 9, &["fact"], false);
    without_chunk["chunk"] = Value::Null;
    cassette(
        &dir,
        "cassettes/serial.jsonl",
        &[
            call1(SOURCE_A, 0, &["fact"], false),
            call1(SOURCE_A, 1, &["preference"], false),
            top_up,
            without_chunk,
        ],
    );
    cassette(
        &dir,
        "cassettes/primed.jsonl",
        &[
            call1(SOURCE_B, 1, &["fact", "fact"], true),
            call1(SOURCE_A, 0, &["fact"], true),
            call1(SOURCE_A, 2, &["preference"], true),
        ],
    );
    assert_eq!(
        report(&compare(&dir)),
        json!({
            "chunks": {"serial": 2, "primed": 3, "compared": 1, "serial_only": 1, "primed_only": 2},
            "agreement": {"claim_count": 1, "kind_multiset": 1},
            // Totals use only the intersection, not unmatched chunks.
            "claims": {"serial": 1, "primed": 1},
            "kinds": {"serial": {"fact": 1}, "primed": {"fact": 1}}
        })
    );
}
