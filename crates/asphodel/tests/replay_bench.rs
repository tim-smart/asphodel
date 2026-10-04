//! `asphodel bench`: concurrent prefetches over HTTP against a daemon
//! started on a copy of the replayed store, to see how the reranker
//! deadline behaves under contention. The store is the one a `live` run
//! on the synthetic
//! history leaves under the private dir.

mod support;

use std::fs;

use support::{
    TestDir, asphodel, assert_ok, assert_refused, imported_small_history, read_json, record, stderr,
};

/// The bench never touches the store it copies, takes an explicit config
/// without one in the environment, and reports latency percentiles and the
/// reranked fraction for each concurrency level.
#[test]
fn bench_runs_on_a_copy_of_the_replayed_store() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let db = dir.private().join("store/asphodel.db");
    let before = fs::read(&db).expect("the live run left a replayed store");
    let modified = fs::metadata(&db).unwrap().modified().unwrap();

    let config = dir.private_file("bench.toml", "[purge]\ndelta = \"never\"\n");
    let report = dir.private_path("reports/bench.json");
    let output = asphodel(&dir)
        .arg("bench")
        .arg("--corpus")
        .arg(&corpus)
        .arg("--config")
        .arg(&config)
        .args(["--concurrency", "1", "--concurrency", "4"])
        .args(["--requests", "8"])
        .args(["--listen", "127.0.0.1:0"])
        .arg("--report")
        .arg(&report)
        .output()
        .unwrap();
    assert_ok(&output);

    assert!(fs::read(&db).unwrap() == before, "bench changed the store");
    assert_eq!(fs::metadata(&db).unwrap().modified().unwrap(), modified);

    let report = read_json(&report);
    let levels = report["levels"]
        .as_array()
        .unwrap_or_else(|| panic!("one entry per concurrency level: {report}"));
    let concurrency: Vec<&serde_json::Value> =
        levels.iter().map(|level| &level["concurrency"]).collect();
    assert_eq!(concurrency, [1, 4]);
    for level in levels {
        for field in ["p50_ms", "p95_ms", "p99_ms"] {
            assert!(level[field].is_number(), "{field}: {level}");
        }
        let reranked = level["reranked_fraction"].as_f64().expect("a fraction");
        assert!((0.0..=1.0).contains(&reranked), "{level}");
    }
}

/// The bench daemon listens on loopback only, checked before anything
/// else; it runs only on a copy of a store replay made, never creating
/// one; and a daemon that fails to open the copy after binding is reported
/// as a refusal rather than left waiting for health.
#[test]
fn bench_refuses_what_it_cant_run_on() {
    let dir = TestDir::new();
    let output = asphodel(&dir)
        .args(["bench", "--listen", "0.0.0.0:0"])
        .output();
    assert_refused(&output.unwrap(), "loopback");

    let corpus = imported_small_history(&dir);
    let bench = || {
        asphodel(&dir)
            .arg("bench")
            .arg("--corpus")
            .arg(&corpus)
            .args(["--requests", "1", "--concurrency", "1"])
            .output()
            .unwrap()
    };
    let output = bench();
    assert_refused(&output, "store");
    assert!(
        !dir.private().join("store").exists(),
        "bench created a store: {}",
        stderr(&output)
    );

    record(&dir, &corpus);
    // The replay marker is still present, but the copied store isn't a
    // database.
    dir.private_file("store/asphodel.db", "not a SQLite database");
    let output = bench();
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
}
