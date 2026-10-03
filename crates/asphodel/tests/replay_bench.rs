//! `asphodel bench`: concurrent prefetches over HTTP against a daemon
//! started on a copy of the replayed store, to see how the reranker
//! deadline behaves under contention. The store is the one a `live` run
//! on the synthetic
//! history leaves under the private dir.

mod support;

use std::fs;

use support::{
    TestDir, asphodel, assert_ok, assert_refused, imported_small_history, record, stderr,
};

/// The bench never touches the store it copies, and reports latency
/// percentiles and the reranked fraction for each concurrency level.
#[test]
fn bench_runs_on_a_copy_of_the_replayed_store() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    record(&dir, &corpus);
    let db = dir.private().join("store/asphodel.db");
    let before = fs::read(&db).expect("the live run left a replayed store");
    let modified = fs::metadata(&db).unwrap().modified().unwrap();

    let report = dir.private_path("reports/bench.json");
    let output = asphodel(&dir)
        .arg("bench")
        .arg("--corpus")
        .arg(&corpus)
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

    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(&report).expect("bench writes its report")).unwrap();
    let levels = report["levels"]
        .as_array()
        .unwrap_or_else(|| panic!("one entry per concurrency level: {report}"));
    let concurrency: Vec<u64> = levels
        .iter()
        .map(|level| level["concurrency"].as_u64().unwrap())
        .collect();
    assert_eq!(concurrency, [1, 4]);
    for level in levels {
        for field in ["p50_ms", "p95_ms", "p99_ms"] {
            assert!(level[field].is_number(), "{field}: {level}");
        }
        let reranked = level["reranked_fraction"]
            .as_f64()
            .unwrap_or_else(|| panic!("the reranked fraction: {level}"));
        assert!((0.0..=1.0).contains(&reranked), "{level}");
    }
}

/// The bench daemon listens on loopback only. The
/// address is checked before anything else, so no store is needed.
#[test]
fn bench_refuses_a_listen_address_off_loopback() {
    let dir = TestDir::new();
    let output = asphodel(&dir)
        .arg("bench")
        .args(["--listen", "0.0.0.0:0"])
        .output()
        .unwrap();
    assert_refused(&output, "loopback");
}

/// The bench runs only on a copy of a store replay made, never on any
/// other.
#[test]
fn bench_refuses_a_private_dir_without_a_replayed_store() {
    let dir = TestDir::new();
    let corpus = imported_small_history(&dir);
    let output = asphodel(&dir)
        .arg("bench")
        .arg("--corpus")
        .arg(&corpus)
        .output()
        .unwrap();
    assert_refused(&output, "store");
    assert!(
        !dir.private().join("store").exists(),
        "bench created a store: {}",
        stderr(&output)
    );
}
