//! Schema migrations and the pre-migration copy (ADR 0010).
//!
//! The schema version is SQLite's `user_version`. Each migration runs in
//! one transaction that also records a row in `migrations` and bumps the
//! version, so a crash mid-way leaves the store at the old version with no
//! half-applied schema. Before the first pending migration the daemon copies
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
pub const SCHEMA_VERSION: u32 = 1;

/// How long a pre-migration copy is kept after its migration completes.
pub const PRE_MIGRATION_COPY_TTL: SignedDuration = SignedDuration::from_hours(7 * 24);

/// Every migration, in order: the version it brings the store to and the
/// SQL that does it.
const MIGRATIONS: &[(u32, &str)] = &[(1, include_str!("../../migrations/0001_initial.sql"))];

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
    let corrupt = |detail: String| StoreError::CorruptCopy {
        path: path.to_owned(),
        detail,
    };
    let mut statement = copy
        .prepare("PRAGMA integrity_check")
        .map_err(|error| corrupt(error.to_string()))?;
    let problems = statement
        .query_map([], |row| row.get::<_, String>(0))
        .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
        .map_err(|error| corrupt(error.to_string()))?;
    match problems.as_slice() {
        [ok] if ok == "ok" => Ok(()),
        _ => Err(corrupt(problems.join("; "))),
    }
}

/// Applies every migration past the current version, each in its own
/// transaction with its `migrations` row. Times come from `clock`.
pub fn apply(
    conn: &mut Connection,
    clock: &dyn Clock,
    copy: Option<PathBuf>,
) -> Result<Applied, rusqlite::Error> {
    let from = version(conn)?;
    let mut current = from;
    for (to, sql) in MIGRATIONS.iter().copied() {
        if to <= current {
            continue;
        }
        let started_at = clock.now();
        let tx = conn.transaction()?;
        tx.execute_batch(sql)?;
        tx.execute(
            "INSERT INTO migrations (from_version, to_version, binary_version, started_at, completed_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            (
                current,
                to,
                crate::VERSION,
                micros(started_at),
                micros(clock.now()),
            ),
        )?;
        tx.pragma_update(None, "user_version", to)?;
        tx.commit()?;
        current = to;
    }
    Ok(Applied {
        from,
        to: current,
        copy,
    })
}

/// Deletes pre-migration copies whose migration completed more than
/// [`PRE_MIGRATION_COPY_TTL`] ago. Returns what it removed.
pub fn expire_copies(
    conn: &Connection,
    dir: &Path,
    now: Timestamp,
) -> Result<Vec<PathBuf>, StoreError> {
    let mut statement = conn.prepare(
        "SELECT from_version, completed_at FROM migrations WHERE from_version > 0 ORDER BY to_version",
    )?;
    let rows = statement.query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)))?;
    let mut removed = Vec::new();
    for row in rows {
        let (from, completed_at) = row?;
        let from = u32::try_from(from).unwrap_or(0);
        let expires_at = timestamp(completed_at)
            .checked_add(PRE_MIGRATION_COPY_TTL)
            .unwrap_or(Timestamp::MAX);
        if now < expires_at {
            continue;
        }
        let path = copy_path(dir, from);
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
