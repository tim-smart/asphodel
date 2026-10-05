//! Schema migrations and the pre-migration copy.
//!
//! The schema version is SQLite's `user_version`. The pending migrations
//! run in one transaction that also records one row in `migrations` and
//! bumps the version, so a crash mid-way leaves the store at the old version
//! with no half-applied schema. A fresh store is created at the current
//! version in one step, with one row from version 0. Before the first pending migration the daemon copies
//! the database into the data dir, keyed by the version it came from. The
//! copy is integrity-checked like backup's, and never overwritten for the
//! same version, so a migration that crash-loops can't replace the clean
//! copy with a damaged one. It is deleted seven days after the migration
//! that follows it completes, by [`expire_copies`] at open and from the
//! service's housekeeping while the daemon runs.

use std::path::{Path, PathBuf};

use jiff::{SignedDuration, Timestamp};
use rusqlite::{Connection, OpenFlags, backup::Backup};

use super::{DB_FILE, StoreError, micros, timestamp};
use crate::clock::Clock;

/// The schema version this binary writes.
pub const SCHEMA_VERSION: u32 = 16;

/// How long a pre-migration copy is kept after its migration completes.
pub const PRE_MIGRATION_COPY_TTL: SignedDuration = SignedDuration::from_hours(7 * 24);

/// Every migration, in order: the version it brings the store to and the
/// SQL that does it.
const MIGRATIONS: &[(u32, &str)] = &[
    (1, include_str!("../../migrations/0001_initial.sql")),
    (2, include_str!("../../migrations/0002_speaker_ids.sql")),
    (
        3,
        include_str!("../../migrations/0003_composed_aliases.sql"),
    ),
    (
        4,
        include_str!("../../migrations/0004_access_per_source.sql"),
    ),
    (5, include_str!("../../migrations/0005_turn_in_context.sql")),
    (6, include_str!("../../migrations/0006_prompt_blocks.sql")),
    (7, include_str!("../../migrations/0007_access_spans.sql")),
    (
        8,
        include_str!("../../migrations/0008_mention_passages.sql"),
    ),
    (9, include_str!("../../migrations/0009_sweep_progress.sql")),
    (10, include_str!("../../migrations/0010_reembed.sql")),
    (11, include_str!("../../migrations/0011_raw_query.sql")),
    (
        12,
        include_str!("../../migrations/0012_document_removal.sql"),
    ),
    (13, include_str!("../../migrations/0013_model_plans.sql")),
    (
        14,
        include_str!("../../migrations/0014_no_entry_snapshots.sql"),
    ),
    (15, include_str!("../../migrations/0015_model_answers.sql")),
    (16, include_str!("../../migrations/0016_refresh_urgent.sql")),
];

/// The columns a migration adds, by version. Every migration is safe to run
/// again over a store that already has what it adds, and SQLite has no
/// `ADD COLUMN IF NOT EXISTS`, so a migration whose columns are all already
/// there is skipped. A column whose table a later migration dropped counts
/// as there.
const ADDED_COLUMNS: &[(u32, &str, &str)] = &[
    (11, "recalls", "raw_query"),
    (12, "sources", "removed_at"),
    (13, "mental_models", "plan"),
    (13, "mental_model_entries", "section"),
    (15, "mental_models", "answer"),
    (16, "mental_models", "refresh_urgent"),
];

/// The columns a migration drops, by version, after its SQL runs. SQLite
/// has no `DROP COLUMN IF EXISTS` either, so each is dropped only while
/// it's there.
const DROPPED_COLUMNS: &[(u32, &str, &str)] = &[(14, "prompt_blocks", "entries")];

/// A migration's step in code.
type Conversion = fn(&Connection) -> Result<(), rusqlite::Error>;

/// What a migration does in code, by version, after its SQL runs. Each is
/// safe to run again, and does nothing once what it converts is gone.
const CONVERSIONS: &[(u32, Conversion)] = &[(15, crate::mental_models::entries_to_answers)];

/// What one open applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Applied {
    pub from: u32,
    pub to: u32,
    /// The pre-migration copy, when there was a database to copy.
    pub copy: Option<PathBuf>,
}

/// The schema version of an open database. A fresh database is at 0.
pub fn version(conn: &Connection) -> Result<u32, rusqlite::Error> {
    conn.query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
        .map(|version| u32::try_from(version).unwrap_or(0))
}

/// Where the copy taken before migrating from `from` lives.
pub fn copy_path(dir: &Path, from: u32) -> PathBuf {
    dir.join(format!("{DB_FILE}.pre-migration-v{from}"))
}

/// Copies the database before migrating from `from`, unless a copy for that
/// version is already there, in which case it is kept as it is. The copy is
/// written to a temporary name, integrity-checked and renamed, so a crash or
/// a bad copy can't leave a file under the final name. An existing copy that
/// fails the check refuses the migration rather than being replaced: the
/// never-overwrite rule is what protects a crash-looping migration.
pub fn take_copy(conn: &Connection, dir: &Path, from: u32) -> Result<PathBuf, StoreError> {
    let path = copy_path(dir, from);
    if path.exists() {
        let existing = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        check_integrity(&existing, &path)?;
        tracing::info!(copy = %path.display(), "keeping the existing pre-migration copy");
        return Ok(path);
    }
    let partial = path.with_extension(format!("v{from}.partial"));
    let _ = std::fs::remove_file(&partial);
    {
        let mut target = Connection::open(&partial)?;
        Backup::new(conn, &mut target)?.run_to_completion(
            256,
            std::time::Duration::from_millis(5),
            None,
        )?;
        // The backup carries the live database's WAL flag. A copy in
        // rollback mode is one self-contained file, and reading it later
        // leaves no `-wal` or `-shm` beside it.
        target.pragma_update(None, "journal_mode", "DELETE")?;
        if let Err(error) = check_integrity(&target, &path) {
            drop(target);
            let _ = std::fs::remove_file(&partial);
            return Err(error);
        }
    }
    std::fs::rename(&partial, &path).map_err(StoreError::io(format!(
        "renaming pre-migration copy to {}",
        path.display()
    )))?;
    tracing::info!(copy = %path.display(), from, "took the pre-migration copy");
    Ok(path)
}

/// Runs `PRAGMA integrity_check` on a copy, naming `path` on failure. A file
/// SQLite can't read as a database fails the same way as a damaged one.
fn check_integrity(copy: &Connection, path: &Path) -> Result<(), StoreError> {
    integrity_problems(copy).map_err(|detail| StoreError::CorruptCopy {
        path: path.to_owned(),
        detail,
    })
}

/// `PRAGMA integrity_check`, with what it found when that isn't `ok`. A
/// file SQLite can't read as a database fails with the reading error.
/// Backup and restore check their copies with it too.
pub(crate) fn integrity_problems(conn: &Connection) -> Result<(), String> {
    let mut statement = conn
        .prepare("PRAGMA integrity_check")
        .map_err(|error| error.to_string())?;
    let problems = statement
        .query_map([], |row| row.get::<_, String>(0))
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .map_err(|error| error.to_string())?;
    match problems.as_slice() {
        [ok] if ok == "ok" => Ok(()),
        _ => Err(problems.join("; ")),
    }
}

/// Applies every migration past the current version in one transaction,
/// with one `migrations` row from the version found to the version reached.
/// Times come from `clock`.
pub fn apply(
    conn: &mut Connection,
    clock: &dyn Clock,
    copy: Option<PathBuf>,
) -> Result<Applied, rusqlite::Error> {
    let from = version(conn)?;
    let mut current = from;
    let started_at = clock.now();
    let tx = conn.transaction()?;
    for (to, sql) in MIGRATIONS.iter().copied() {
        if to <= current {
            continue;
        }
        let mut added = false;
        for (_, table, column) in ADDED_COLUMNS.iter().filter(|(version, ..)| *version == to) {
            added = !has_table(&tx, table)? || has_column(&tx, table, column)?;
            if !added {
                break;
            }
        }
        if !added {
            tx.execute_batch(sql)?;
        }
        for (_, table, column) in DROPPED_COLUMNS
            .iter()
            .filter(|(version, ..)| *version == to)
        {
            if has_column(&tx, table, column)? {
                tx.execute_batch(&format!("ALTER TABLE {table} DROP COLUMN {column}"))?;
            }
        }
        for (_, convert) in CONVERSIONS.iter().filter(|(version, _)| *version == to) {
            convert(&tx)?;
        }
        current = to;
    }
    if current > from {
        tx.execute(
            "INSERT INTO migrations (from_version, to_version, binary_version, started_at, completed_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            (
                from,
                current,
                crate::VERSION,
                micros(started_at),
                micros(clock.now()),
            ),
        )?;
        tx.pragma_update(None, "user_version", current)?;
    }
    tx.commit()?;
    Ok(Applied {
        from,
        to: current,
        copy,
    })
}

/// Whether the database has `table`.
fn has_table(conn: &Connection, table: &str) -> Result<bool, rusqlite::Error> {
    conn.query_row(
        "SELECT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
        [table],
        |row| row.get(0),
    )
}

/// Whether `table` has `column`.
fn has_column(conn: &Connection, table: &str, column: &str) -> Result<bool, rusqlite::Error> {
    conn.query_row(
        "SELECT COUNT(*) FROM pragma_table_info(?1) WHERE name = ?2",
        (table, column),
        |row| row.get::<_, i64>(0),
    )
    .map(|count| count > 0)
}

/// Deletes pre-migration copies whose migration completed more than
/// [`PRE_MIGRATION_COPY_TTL`] ago. Returns what it removed.
pub fn expire_copies(
    conn: &Connection,
    dir: &Path,
    now: Timestamp,
) -> Result<Vec<PathBuf>, StoreError> {
    let mut removed = Vec::new();
    for (path, expires_at) in copy_deadlines(conn, dir)? {
        if now < expires_at {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => removed.push(path),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(StoreError::io(format!(
                    "deleting pre-migration copy {}",
                    path.display()
                ))(error));
            }
        }
    }
    Ok(removed)
}

/// When the next pre-migration copy still on disk is due for deletion, if
/// any is. A copy that is already gone has no deadline, so a deleted copy
/// can't keep the daemon waking for it.
pub fn next_copy_expiry(conn: &Connection, dir: &Path) -> Result<Option<Timestamp>, StoreError> {
    Ok(copy_deadlines(conn, dir)?
        .into_iter()
        .filter(|(path, _)| path.exists())
        .map(|(_, expires_at)| expires_at)
        .min())
}

/// Every pre-migration copy still on disk: the version it was taken from,
/// its path and when it's deleted. `asphodel status` reports them.
pub fn copies(conn: &Connection, dir: &Path) -> Result<Vec<(u32, PathBuf, Timestamp)>, StoreError> {
    Ok(copy_rows(conn, dir)?
        .into_iter()
        .filter(|(_, path, _)| path.exists())
        .collect())
}

/// Every copy a completed migration has a row for, with when it expires.
fn copy_deadlines(conn: &Connection, dir: &Path) -> Result<Vec<(PathBuf, Timestamp)>, StoreError> {
    Ok(copy_rows(conn, dir)?
        .into_iter()
        .map(|(_, path, expires_at)| (path, expires_at))
        .collect())
}

/// Every copy a completed migration has a row for: the version it was
/// taken from, where it lives and when it expires.
fn copy_rows(conn: &Connection, dir: &Path) -> Result<Vec<(u32, PathBuf, Timestamp)>, StoreError> {
    let mut statement = conn.prepare(
        "SELECT from_version, completed_at FROM migrations WHERE from_version > 0 ORDER BY to_version",
    )?;
    let rows = statement.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?;
    let mut copies = Vec::new();
    for row in rows {
        let (from, completed_at) = row?;
        let from = u32::try_from(from).unwrap_or(0);
        let expires_at = timestamp(completed_at)
            .checked_add(PRE_MIGRATION_COPY_TTL)
            .unwrap_or(Timestamp::MAX);
        copies.push((from, copy_path(dir, from), expires_at));
    }
    Ok(copies)
}
