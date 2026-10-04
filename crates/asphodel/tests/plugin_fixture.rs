//! The plugin's backfill tests read a Hermes `state.db` checked in under
//! `plugin/tests/fixtures/`, with the manifest it imports with and the
//! counts `asphodel import --dry-run` prints for it. All three come from
//! here: [`hermes::backfill_history`] is the one source, and these tests
//! fail when a checked-in file drifts from it or the importer's counts
//! change. `ASPHODEL_WRITE_FIXTURES=1` rewrites them.

mod support;

use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::Connection;
use rusqlite::types::Value as SqlValue;
use serde_json::Value;
use support::hermes;
use support::{TestDir, assert_ok, import_with, stdout};

const WRITE_ENV: &str = "ASPHODEL_WRITE_FIXTURES";

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../plugin/tests/fixtures")
}

fn writing() -> bool {
    std::env::var(WRITE_ENV).is_ok_and(|value| value == "1")
}

/// Every table's DDL and rows, in rowid order.
fn dump(path: &Path) -> Vec<(String, String, Vec<Vec<SqlValue>>)> {
    let conn = Connection::open(path).unwrap();
    let tables: Vec<(String, String)> = conn
        .prepare("SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name")
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    tables
        .into_iter()
        .map(|(name, sql)| {
            let mut statement = conn
                .prepare(&format!("SELECT * FROM \"{name}\" ORDER BY rowid"))
                .unwrap();
            let width = statement.column_count();
            let rows = statement
                .query_map([], |row| {
                    (0..width).map(|i| row.get::<_, SqlValue>(i)).collect()
                })
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap();
            (name, sql, rows)
        })
        .collect()
}

/// The checked-in `state.db` holds what [`hermes::backfill_history`]
/// builds, table for table, and the manifest is the importer tests' own.
#[test]
fn the_checked_in_state_db_is_backfill_history() {
    let dir = TestDir::new();
    let built = dir.path("state.db");
    drop(hermes::backfill_history(&built));
    let state_db = fixtures().join("state.db");
    let manifest = fixtures().join("manifest.toml");
    if writing() {
        fs::create_dir_all(fixtures()).unwrap();
        fs::copy(&built, &state_db).unwrap();
        fs::write(&manifest, hermes::MANIFEST.trim_start()).unwrap();
    }
    assert!(
        dump(&state_db) == dump(&built),
        "{} isn't backfill_history; rerun with {WRITE_ENV}=1",
        state_db.display()
    );
    assert_eq!(
        fs::read_to_string(&manifest).unwrap(),
        hermes::MANIFEST.trim_start(),
        "rerun with {WRITE_ENV}=1"
    );
}

/// `import-counts.json` is what `asphodel import --dry-run` prints for the
/// checked-in `state.db` and manifest, the counts the backfill must match.
#[test]
fn the_checked_in_counts_are_the_importers_dry_run() {
    let dir = TestDir::new();
    let state_db = dir.private_path("state.db");
    drop(hermes::backfill_history(&state_db));
    let corpus = dir.private_path("corpus/main.jsonl");
    let output = import_with(&dir, &state_db, &corpus, hermes::MANIFEST, &["--dry-run"]);
    assert_ok(&output);
    let printed: Value = serde_json::from_str(&stdout(&output)).unwrap();
    let counts = fixtures().join("import-counts.json");
    if writing() {
        fs::write(
            &counts,
            serde_json::to_string_pretty(&printed).unwrap() + "\n",
        )
        .unwrap();
    }
    let checked_in: Value = serde_json::from_str(&fs::read_to_string(&counts).unwrap()).unwrap();
    assert_eq!(checked_in, printed, "rerun with {WRITE_ENV}=1");
}
