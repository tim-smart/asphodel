//! `asphodel report diff A B`: the A/B diff of two replay reports. It refuses
//! runs on a different corpus or cassette unless `--force`, compares numbers
//! with a tolerance (the last float digits can differ across machines), and
//! names the memories that faded or were purged in one run and not the other.
//! The diff is printed as JSON; it holds ids and numbers, and the reports stay
//! where they are.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{Context as _, bail};
use serde_json::{Map, Value, json};

use crate::cli::DiffArgs;

/// Relative tolerance for floats, and absolute for values near zero.
const TOLERANCE: f64 = 1e-6;

/// Runs the command: the diff on stdout and exit 0, or exit 2 when the
/// reports can't be compared.
pub fn run(args: DiffArgs) -> anyhow::Result<()> {
    match execute(&args) {
        Ok(diff) => {
            println!("{}", serde_json::to_string_pretty(&diff)?);
            Ok(())
        }
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::exit(2)
        }
    }
}

fn execute(args: &DiffArgs) -> anyhow::Result<Value> {
    let a = load(&args.a)?;
    let b = load(&args.b)?;
    for key in ["corpus_hash", "cassette_hash"] {
        if a.get(key) != b.get(key) && !args.force {
            bail!(
                "the runs have a different {}: {} has {} and {} has {}; pass --force to compare them anyway",
                key.trim_end_matches("_hash"),
                args.a.display(),
                describe(a.get(key)),
                args.b.display(),
                describe(b.get(key))
            );
        }
    }
    Ok(diff(&a, &b))
}

fn load(path: &Path) -> anyhow::Result<Map<String, Value>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("{} isn't a JSON report", path.display()))?;
    match value {
        Value::Object(map) => Ok(map),
        _ => bail!("{} isn't a replay report", path.display()),
    }
}

fn describe(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Null) | None => "none".into(),
        Some(other) => other.to_string(),
    }
}

/// The diff: the identities side by side, probes whose result changed,
/// numbers that differ beyond the tolerance by path, and the memories
/// whose fate differs.
fn diff(a: &Map<String, Value>, b: &Map<String, Value>) -> Value {
    let identity = |report: &Map<String, Value>| {
        json!({
            "kind": report.get("kind"),
            "git_sha": report.get("git_sha"),
            "corpus_hash": report.get("corpus_hash"),
            "cassette_hash": report.get("cassette_hash"),
            "refresh_queries_hash": report.get("refresh_queries_hash"),
        })
    };

    let probes = |report: &Map<String, Value>| -> BTreeMap<String, bool> {
        report
            .get("probes")
            .and_then(Value::as_array)
            .map(|probes| {
                probes
                    .iter()
                    .filter_map(|probe| {
                        Some((
                            probe.get("id")?.as_str()?.to_string(),
                            probe.get("passed")?.as_bool()?,
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let probes_a = probes(a);
    let probes_b = probes(b);
    let probe_ids: BTreeSet<&String> = probes_a.keys().chain(probes_b.keys()).collect();
    let changed_probes: Vec<Value> = probe_ids
        .into_iter()
        .filter(|id| probes_a.get(*id) != probes_b.get(*id))
        .map(|id| json!({ "id": id, "a": probes_a.get(id), "b": probes_b.get(id) }))
        .collect();

    let mut numbers = Vec::new();
    for key in [
        "purged_then_re_mentioned",
        "extraction_lag",
        "injected_tokens",
        "injection_usage",
        "profile_tokens",
        "call2_rate",
        "significance_histogram",
        "kind_histogram",
        "llm",
    ] {
        compare_numbers(
            key,
            a.get(key).unwrap_or(&Value::Null),
            b.get(key).unwrap_or(&Value::Null),
            &mut numbers,
        );
    }
    for key in [
        "purges_per_day",
        "fade_outs_per_week",
        "bands_per_week",
        "refresh_calls_per_day",
        "agenda_lines_per_day",
    ] {
        compare_series(key, a.get(key), b.get(key), &mut numbers);
    }

    let memories = |report: &Map<String, Value>| -> BTreeMap<String, (bool, bool)> {
        report
            .get("memories")
            .and_then(Value::as_array)
            .map(|memories| {
                memories
                    .iter()
                    .filter_map(|memory| {
                        Some((
                            memory.get("id")?.as_str()?.to_string(),
                            (
                                !memory.get("faded_at")?.is_null(),
                                !memory.get("purged_at")?.is_null(),
                            ),
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let memories_a = memories(a);
    let memories_b = memories(b);
    let only = |left: &BTreeMap<String, (bool, bool)>,
                right: &BTreeMap<String, (bool, bool)>,
                pick: fn(&(bool, bool)) -> bool|
     -> Vec<String> {
        left.iter()
            .filter(|(id, fate)| pick(fate) && !right.get(*id).is_some_and(pick))
            .map(|(id, _)| id.clone())
            .collect()
    };
    let faded = |fate: &(bool, bool)| fate.0;
    let purged = |fate: &(bool, bool)| fate.1;

    json!({
        "a": identity(a),
        "b": identity(b),
        "probes": { "changed": changed_probes },
        "numbers": numbers,
        "memories": {
            "created_in_a": memories_a.len(),
            "created_in_b": memories_b.len(),
            "faded_only_in_a": only(&memories_a, &memories_b, faded),
            "faded_only_in_b": only(&memories_b, &memories_a, faded),
            "purged_only_in_a": only(&memories_a, &memories_b, purged),
            "purged_only_in_b": only(&memories_b, &memories_a, purged),
            "only_in_a": memories_a.keys().filter(|id| !memories_b.contains_key(*id)).collect::<Vec<_>>(),
            "only_in_b": memories_b.keys().filter(|id| !memories_a.contains_key(*id)).collect::<Vec<_>>(),
        },
    })
}

/// Walks two values together and records every number that differs
/// beyond the tolerance, by dotted path.
fn compare_numbers(path: &str, a: &Value, b: &Value, out: &mut Vec<Value>) {
    match (a, b) {
        (Value::Object(a), Value::Object(b)) => {
            let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
            for key in keys {
                compare_numbers(
                    &format!("{path}.{key}"),
                    a.get(key).unwrap_or(&Value::Null),
                    b.get(key).unwrap_or(&Value::Null),
                    out,
                );
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                compare_numbers(&format!("{path}[{index}]"), a, b, out);
            }
            if a.len() != b.len() {
                out.push(json!({ "path": format!("{path}.length"), "a": a.len(), "b": b.len() }));
            }
        }
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap_or(0.0), y.as_f64().unwrap_or(0.0));
            if !within(x, y) {
                out.push(json!({ "path": path, "a": x, "b": y, "delta": y - x }));
            }
        }
        (Value::Null, Value::Null) => {}
        (Value::Number(_), _) | (_, Value::Number(_)) => {
            out.push(json!({ "path": path, "a": a, "b": b }));
        }
        _ => {}
    }
}

/// Compares two series keyed by their `day` or `week`, so a day present in
/// one run and not the other shows as a difference.
fn compare_series(key: &str, a: Option<&Value>, b: Option<&Value>, out: &mut Vec<Value>) {
    let index = |series: Option<&Value>| -> BTreeMap<String, Value> {
        series
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| {
                        let label = item
                            .get("day")
                            .or_else(|| item.get("week"))?
                            .as_str()?
                            .to_string();
                        Some((label, item.clone()))
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let a = index(a);
    let b = index(b);
    let labels: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
    for label in labels {
        compare_numbers(
            &format!("{key}[{label}]"),
            a.get(label).unwrap_or(&Value::Null),
            b.get(label).unwrap_or(&Value::Null),
            out,
        );
    }
}

fn within(x: f64, y: f64) -> bool {
    let scale = x.abs().max(y.abs()).max(1.0);
    (x - y).abs() <= TOLERANCE * scale
}
