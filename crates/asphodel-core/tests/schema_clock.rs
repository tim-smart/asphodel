//! "No timestamp comes from SQLite's clock" (TIM-103). "What is a memory
//! record?" (TIM-90) put it as: the schema never takes a timestamp from the
//! database clock, and every time comes through a clock the replay harness
//! can swap out.
//!
//! `clippy.toml` already denies the Rust wall-clock calls. This closes the
//! other door: SQL that asks SQLite for the time, whether as a column default
//! or inside a statement. It scans every `.rs` and `.sql` file in the
//! workspace outside `tests/` directories, so it holds whichever file the
//! store keeps its schema and migrations in, and comments count too.

use std::fs;
use std::path::{Path, PathBuf};

/// SQL that reads SQLite's clock, matched case-insensitively.
const CLOCK_READS: &[&str] = &[
    "current_timestamp",
    "current_time",
    "current_date",
    // datetime('now'), strftime('%s', 'now'), julianday('now'), unixepoch('now')
    "'now'",
    "unixepoch()",
    "julianday()",
];

/// Clock reads that are only unambiguous in SQL files, because jiff has
/// methods of the same name (`Zoned::datetime()`, `Zoned::date()`,
/// `Zoned::time()`).
const SQL_ONLY_CLOCK_READS: &[&str] = &["datetime()", "date()", "time()"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// Every `.rs` and `.sql` file under `dir`, skipping build output, the git
/// dir and `tests/` directories (which hold this file).
fn source_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            if matches!(
                entry.file_name().to_str(),
                Some("target" | ".git" | "tests")
            ) {
                continue;
            }
            source_files(&path, out);
        } else if matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("rs" | "sql")
        ) {
            out.push(path);
        }
    }
}

#[test]
fn no_sql_reads_sqlites_clock() {
    let root = workspace_root();
    let mut files = Vec::new();
    source_files(&root, &mut files);
    assert!(
        files
            .iter()
            .any(|path| path.ends_with("crates/asphodel-core/src/lib.rs")),
        "the scan from {} didn't reach the core crate",
        root.display()
    );

    let mut hits = Vec::new();
    for path in files {
        let is_sql = path.extension().is_some_and(|ext| ext == "sql");
        let text = fs::read_to_string(&path).unwrap().to_lowercase();
        for (index, line) in text.lines().enumerate() {
            let sql_only = is_sql.then_some(SQL_ONLY_CLOCK_READS);
            for pattern in CLOCK_READS.iter().chain(sql_only.into_iter().flatten()) {
                if line.contains(pattern) {
                    hits.push(format!(
                        "{}:{}: {pattern}",
                        path.strip_prefix(&root).unwrap().display(),
                        index + 1
                    ));
                }
            }
        }
    }
    assert!(
        hits.is_empty(),
        "SQL reads SQLite's clock; take the time from the Clock instead:\n{}",
        hits.join("\n")
    );
}
