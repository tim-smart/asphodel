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
    let before: Vec<_> = ["cassettes/serial.jsonl", "cassettes/primed.jsonl"]
        .map(|name| fs::read(dir.private_path(name)).unwrap())
        .into();
    let output = compare_paths(dir, "cassettes/serial.jsonl", "cassettes/primed.jsonl");
    for (name, bytes) in ["cassettes/serial.jsonl", "cassettes/primed.jsonl"]
        .into_iter()
        .zip(before)
    {
        assert_eq!(fs::read(dir.private_path(name)).unwrap(), bytes);
    }
    output
}

fn compare_paths(dir: &TestDir, serial: &str, primed: &str) -> Output {
    asphodel(dir)
        .args([
            "report",
            "claims-agreement",
            "--serial",
            serial,
            "--primed",
            primed,
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

#[test]
fn rejects_duplicate_chunks_even_when_records_are_identical() {
    for side in ["serial", "primed"] {
        for identical in [true, false] {
            let dir = TestDir::new();
            let first = call1(SOURCE_A, 0, &["fact"], false);
            let second = if identical {
                first.clone()
            } else {
                call1(SOURCE_A, 0, &["preference"], true)
            };
            cassette(&dir, "cassettes/serial.jsonl", &[]);
            cassette(&dir, "cassettes/primed.jsonl", &[]);
            cassette(&dir, &format!("cassettes/{side}.jsonl"), &[first, second]);
            refused(&compare(&dir), "duplicate call 1 chunk", &dir);
        }
    }
}

#[test]
fn rejects_incompatible_metadata_within_and_between_cassettes() {
    for field in ["version", "guidance", "model", "language"] {
        for location in ["serial", "primed", "between"] {
            let dir = TestDir::new();
            let first = call1(SOURCE_A, 0, &["fact"], false);
            let mut changed = call1(SOURCE_B, 1, &["fact"], true);
            match field {
                "version" => {
                    changed["template"][field] = json!(999);
                    changed["request"]["template"][field] = json!(999);
                }
                "guidance" => {
                    changed["template"][field] = json!(SECRET);
                    changed["request"]["template"][field] = json!(SECRET);
                }
                _ => changed[field] = json!(SECRET),
            }
            let (serial, primed) = match location {
                "serial" => (vec![first, changed], vec![]),
                "primed" => (vec![], vec![first, changed]),
                _ => (vec![first], vec![changed]),
            };
            cassette(&dir, "cassettes/serial.jsonl", &serial);
            cassette(&dir, "cassettes/primed.jsonl", &primed);
            refused(&compare(&dir), "incompatible extraction metadata", &dir);
        }
    }
}

#[test]
fn rejects_record_request_template_mismatches_without_exposing_values() {
    for field in ["name", "version", "guidance"] {
        let dir = TestDir::new();
        let mut record = call1(SOURCE_A, 0, &["fact"], false);
        record["request"]["template"][field] = if field == "version" {
            json!(999)
        } else {
            json!(SECRET)
        };
        cassette(&dir, "cassettes/serial.jsonl", &[record]);
        cassette(&dir, "cassettes/primed.jsonl", &[]);
        refused(&compare(&dir), "inconsistent extraction metadata", &dir);
    }
}

#[test]
fn malformed_records_and_claims_fail_without_exporting_input_text() {
    for side in ["serial", "primed"] {
        for case in [
            "json",
            "record",
            "claims",
            "unknown_kind",
            "missing_kind",
            "kind_type",
        ] {
            let dir = TestDir::new();
            cassette(&dir, "cassettes/serial.jsonl", &[]);
            cassette(&dir, "cassettes/primed.jsonl", &[]);
            let mut record = call1(SOURCE_A, 0, &["fact"], false);
            let (bytes, reason) = match case {
                "json" => (format!("{{{SECRET}"), "invalid"),
                "record" => (json!({"private": SECRET}).to_string(), "invalid"),
                "claims" => {
                    record["response"]["json"]["claims"] = json!(SECRET);
                    (record.to_string(), "invalid claims")
                }
                _ => {
                    let claim = &mut record["response"]["json"]["claims"][0];
                    match case {
                        "unknown_kind" => claim["kind"] = json!(SECRET),
                        "missing_kind" => {
                            claim.as_object_mut().unwrap().remove("kind");
                        }
                        _ => claim["kind"] = json!({"private": SECRET}),
                    }
                    (record.to_string(), "invalid claim kind")
                }
            };
            fs::write(dir.private_path(&format!("cassettes/{side}.jsonl")), bytes).unwrap();
            refused(&compare(&dir), reason, &dir);
        }
    }
}

#[test]
fn empty_and_disjoint_cassettes_report_zero_agreement_and_no_claim_totals() {
    for disjoint in [false, true] {
        let dir = TestDir::new();
        let serial = if disjoint {
            vec![call1(SOURCE_A, 0, &["fact"], false)]
        } else {
            vec![]
        };
        let primed = if disjoint {
            vec![call1(SOURCE_B, 0, &["task"], true)]
        } else {
            vec![]
        };
        cassette(&dir, "cassettes/serial.jsonl", &serial);
        cassette(&dir, "cassettes/primed.jsonl", &primed);
        if !disjoint {
            // Whitespace-only lines are also an empty cassette.
            fs::write(dir.private_path("cassettes/serial.jsonl"), b"\n \t\n").unwrap();
        }
        let n = usize::from(disjoint);
        assert_eq!(
            report(&compare(&dir)),
            json!({
                "chunks": {"serial": n, "primed": n, "compared": 0, "serial_only": n, "primed_only": n},
                "agreement": {"claim_count": 0, "kind_multiset": 0},
                "claims": {"serial": 0, "primed": 0},
                "kinds": {"serial": {}, "primed": {}}
            })
        );
    }
}

#[test]
fn refuses_paths_outside_private_directory_without_reading_or_changing_them() {
    let dir = TestDir::new();
    cassette(&dir, "cassettes/serial.jsonl", &[]);
    cassette(&dir, "cassettes/primed.jsonl", &[]);
    let outside = dir.path(SECRET);
    fs::write(&outside, SECRET).unwrap();
    let before_serial = fs::read(dir.private_path("cassettes/serial.jsonl")).unwrap();
    let before_primed = fs::read(dir.private_path("cassettes/primed.jsonl")).unwrap();
    let mut paths = vec![outside.to_str().unwrap().to_owned(), format!("../{SECRET}")];
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&outside, dir.private_path("cassettes/escape.jsonl")).unwrap();
        paths.push("cassettes/escape.jsonl".to_owned());
    }
    for path in paths {
        for serial_side in [true, false] {
            let output = if serial_side {
                compare_paths(&dir, &path, "cassettes/primed.jsonl")
            } else {
                compare_paths(&dir, "cassettes/serial.jsonl", &path)
            };
            refused(&output, "inside the private replay directory", &dir);
            assert_eq!(fs::read(&outside).unwrap(), SECRET.as_bytes());
            assert_eq!(
                fs::read(dir.private_path("cassettes/serial.jsonl")).unwrap(),
                before_serial
            );
            assert_eq!(
                fs::read(dir.private_path("cassettes/primed.jsonl")).unwrap(),
                before_primed
            );
        }
    }
}
