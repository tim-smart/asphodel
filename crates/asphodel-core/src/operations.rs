//! Backup, restore, status and the audit lists ("Operations: backup,
//! restore, status and audit lists", TIM-114; TIM-99, decisions 1, 6 and 7;
//! ADR 0010).
//!
//! - **Backup** takes SQLite's online backup into a temporary file in the
//!   data dir, writes the backup time into the copy, checks it with
//!   `PRAGMA integrity_check`, and hands it over open and already unlinked,
//!   so nothing is left behind however the stream ends.
//! - **Restore** is offline. It holds the data-dir lock, checks the copy's
//!   integrity and that its schema version isn't newer than the binary's,
//!   moves the current database and its WAL aside, and copies the backup in
//!   with a daemon-wide `restored` edit row. A deletion fingerprint that
//!   differs from the binary's pauses purge at the next start (ADR 0009).
//! - **Status** gathers what an operator alerts on, and says what needs
//!   attention. A store with no backup yet doesn't: Asphodel has no backup
//!   schedule of its own.
//! - **The audit lists** read the edit log, the sweep runs and the recall
//!   log. Only the recall list carries content: its queries.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use jiff::Timestamp;
use rusqlite::{Connection, OpenFlags, OptionalExtension, backup};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::clock::Clock;
use crate::config::{Fingerprint, PurgePause};
use crate::erase::{EDIT_FORGET, EDIT_PURGED};
use crate::store::fs::check_data_dir;
use crate::store::ids::IdSource;
use crate::store::migrations::{self, SCHEMA_VERSION, integrity_problems};
use crate::store::{
    DB_FILE, DataDirLock, OpenOptions, Store, StoreError, micros, prepare_dir, register_extensions,
    timestamp,
};

/// The edit kind restore writes: daemon-wide, with the backup time, the
/// restore time and the binary version.
pub const EDIT_RESTORED: &str = "restored";

/// The `store_meta` key of when `POST /v1/backup` last completed, in the
/// live store.
pub const META_LAST_BACKUP: &str = "last_backup_at";

/// The `store_meta` key of when a copy was taken, written into the copy
/// itself, since the stream reaches a restore through pipes.
pub const META_BACKED_UP_AT: &str = "backed_up_at";

/// The response header holding a backup's SHA-256, as lowercase hex.
pub const SHA256_HEADER: &str = "asphodel-sha256";

/// The response header holding a backup's length in bytes.
pub const LENGTH_HEADER: &str = "asphodel-length";

/// The response header holding when the copy was taken (RFC 3339).
pub const BACKED_UP_AT_HEADER: &str = "asphodel-backed-up-at";

/// The most rows one audit list returns.
pub const MAX_LIST: usize = 1000;

/// How many rows an audit list returns when not told.
pub const DEFAULT_LIST: usize = 50;

/// A checked copy of the store, ready to stream. The file is already
/// unlinked from the data dir; dropping it frees the space.
#[derive(Debug)]
pub struct Backup {
    pub file: File,
    /// The copy's SHA-256, as lowercase hex.
    pub sha256: String,
    pub length: u64,
    pub backed_up_at: Timestamp,
}

#[derive(Debug, thiserror::Error)]
pub enum BackupError {
    #[error("the backup copy failed its integrity check ({detail})")]
    Corrupt { detail: String },

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for BackupError {
    fn from(error: rusqlite::Error) -> Self {
        BackupError::Store(StoreError::Sqlite(error))
    }
}

impl From<std::io::Error> for BackupError {
    fn from(error: std::io::Error) -> Self {
        BackupError::Store(StoreError::Io {
            context: "writing the backup copy".to_owned(),
            error,
        })
    }
}

/// Takes the online backup of `store` into a temporary file in its data
/// dir and checks it (ADR 0010).
pub(crate) fn take_backup(store: &Store) -> Result<Backup, BackupError> {
    let dir = store.dir();
    let (path, file) =
        crate::models::write::create_temp(&dir.join(format!("{DB_FILE}.backup")), 0o600)?;
    drop(file);
    let taken = copy_and_check(store, &path);
    let opened = taken.and_then(|backed_up_at| {
        let (sha256, length) = digest(&path)?;
        let file = File::open(&path)?;
        Ok((file, sha256, length, backed_up_at))
    });
    // Unlinked either way: an open file keeps its contents until it's
    // closed, and a failed copy has nothing worth keeping.
    let _ = std::fs::remove_file(&path);
    let (file, sha256, length, backed_up_at) = opened?;
    Ok(Backup {
        file,
        sha256,
        length,
        backed_up_at,
    })
}

/// Copies the live database into `path` through its own connection, so the
/// copy never holds the store's, and checks it.
fn copy_and_check(store: &Store, path: &Path) -> Result<Timestamp, BackupError> {
    let source = Connection::open_with_flags(
        store.dir().join(DB_FILE),
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    source.busy_timeout(std::time::Duration::from_secs(5))?;
    let mut target = Connection::open(path)?;
    // One step of every page copies under one read transaction, so the
    // copy is a consistent snapshot and writers carry on meanwhile (WAL). A
    // busy or locked source is retried.
    {
        let copy = backup::Backup::new(&source, &mut target)?;
        while copy.step(-1)? != backup::StepResult::Done {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
    drop(source);
    let backed_up_at = store.now();
    target.execute(
        "INSERT INTO store_meta (key, value, updated_at) VALUES (?1, ?2, ?3)
         ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (
            META_BACKED_UP_AT,
            backed_up_at.to_string(),
            micros(backed_up_at),
        ),
    )?;
    // A copy in rollback mode is one self-contained file.
    target.pragma_update(None, "journal_mode", "DELETE")?;
    integrity_problems(&target).map_err(|detail| BackupError::Corrupt { detail })?;
    Ok(backed_up_at)
}

/// A new file beside `path` for writing a copy into, created exclusively
/// with mode 0600 under a fresh name, as `asphodel backup --out` writes
/// before it renames.
pub fn temp_file_beside(path: &Path) -> std::io::Result<(PathBuf, File)> {
    crate::models::write::create_temp(path, 0o600)
}

/// The SHA-256 of the file at `path`, as lowercase hex, and its length.
fn digest(path: &Path) -> std::io::Result<(String, u64)> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0; 64 * 1024];
    let mut length = 0;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        length += read as u64;
    }
    let hash = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok((hash, length))
}

/// Records that `POST /v1/backup` completed at `at`, for `status`. It says
/// nothing about whether the stream reached its destination.
pub(crate) fn record_backup(store: &Store, at: Timestamp) -> Result<(), StoreError> {
    store.connection().execute(
        "INSERT INTO store_meta (key, value, updated_at) VALUES (?1, ?2, ?3)
         ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (META_LAST_BACKUP, at.to_string(), micros(at)),
    )?;
    Ok(())
}

/// Runs `PRAGMA integrity_check` on the database file at `path`, read-only,
/// as `asphodel backup --out <file>` checks what it wrote.
pub fn check_copy(path: &Path) -> Result<(), String> {
    register_extensions();
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|error| error.to_string())?;
    integrity_problems(&conn)
}

/// What a restore did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Restored {
    /// When the copy was taken, when it says.
    pub backed_up_at: Option<Timestamp>,
    pub restored_at: Timestamp,
    /// The copy's schema version. An older one migrates when the daemon
    /// next starts, with a pre-migration copy first.
    pub schema_version: u32,
    /// Where the database it replaced went, with its WAL beside it under
    /// the same name. `None` when the data dir had no database.
    pub moved_aside: Option<PathBuf>,
}

#[derive(Debug, thiserror::Error)]
pub enum RestoreError {
    #[error("{} failed its integrity check ({detail})", path.display())]
    Corrupt { path: PathBuf, detail: String },

    #[error("{} isn't an Asphodel store: it has no schema version", path.display())]
    NotAStore { path: PathBuf },

    #[error(
        "{} is schema version {found}, newer than this binary's {supported}; restore it with a newer asphodel",
        path.display()
    )]
    NewerSchema {
        path: PathBuf,
        found: u32,
        supported: u32,
    },

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for RestoreError {
    fn from(error: rusqlite::Error) -> Self {
        RestoreError::Store(StoreError::Sqlite(error))
    }
}

/// `asphodel restore <backup> --data-dir <dir>`: offline, under the
/// data-dir lock, so it refuses while a daemon runs. Nothing in the data
/// dir moves until the copy has passed its checks and been written beside
/// the database with its `restored` row.
pub fn restore(
    backup: &Path,
    dir: &Path,
    options: OpenOptions,
    clock: &dyn Clock,
) -> Result<Restored, RestoreError> {
    prepare_dir(dir)?;
    check_data_dir(dir, options.allow_network_fs)?;
    let _lock = DataDirLock::acquire(dir)?;
    register_extensions();

    let (schema_version, backed_up_at) = inspect(backup)?;
    let live = dir.join(DB_FILE);
    let (staged, file) =
        crate::models::write::create_temp(&dir.join(format!("{DB_FILE}.restore")), 0o600)
            .map_err(io(format!("creating a file in {}", dir.display())))?;
    let restored_at = clock.now();
    let staging = stage(
        backup,
        &staged,
        file,
        backed_up_at,
        restored_at,
        schema_version,
    );
    let installed = staging.and_then(|()| install(dir, &live, &staged, restored_at));
    if installed.is_err() {
        let _ = std::fs::remove_file(&staged);
    }
    let moved_aside = installed?;
    tracing::info!(
        dir = %dir.display(),
        schema_version,
        moved_aside = moved_aside.as_ref().map(|path| path.display().to_string()),
        "restored the store"
    );
    Ok(Restored {
        backed_up_at,
        restored_at,
        schema_version,
        moved_aside,
    })
}

/// Checks the copy at `path` and reads its schema version and backup time.
fn inspect(path: &Path) -> Result<(u32, Option<Timestamp>), RestoreError> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    integrity_problems(&conn).map_err(|detail| RestoreError::Corrupt {
        path: path.to_owned(),
        detail,
    })?;
    let found = migrations::version(&conn)?;
    if found == 0 {
        return Err(RestoreError::NotAStore {
            path: path.to_owned(),
        });
    }
    if found > SCHEMA_VERSION {
        return Err(RestoreError::NewerSchema {
            path: path.to_owned(),
            found,
            supported: SCHEMA_VERSION,
        });
    }
    let backed_up_at: Option<String> = conn
        .query_row(
            "SELECT value FROM store_meta WHERE key = ?1",
            [META_BACKED_UP_AT],
            |row| row.get(0),
        )
        .optional()?;
    Ok((found, backed_up_at.and_then(|at| at.parse().ok())))
}

/// Copies the backup into `staged` and writes the `restored` row there, so
/// the row lands with the database or not at all.
fn stage(
    backup: &Path,
    staged: &Path,
    mut file: File,
    backed_up_at: Option<Timestamp>,
    restored_at: Timestamp,
    schema_version: u32,
) -> Result<(), RestoreError> {
    let copying = io(format!("copying {}", backup.display()));
    let mut source = File::open(backup).map_err(&copying)?;
    std::io::copy(&mut source, &mut file).map_err(&copying)?;
    file.sync_all().map_err(&copying)?;
    drop(file);

    let conn = Connection::open(staged)?;
    // The backup time belongs to the copy; the live store has its own.
    conn.execute("DELETE FROM store_meta WHERE key = ?1", [META_BACKED_UP_AT])?;
    conn.execute(
        "INSERT INTO edits (uuid, bank_id, kind, details, at) VALUES (?1, NULL, ?2, ?3, ?4)",
        (
            IdSource::new().next(restored_at).to_string(),
            EDIT_RESTORED,
            serde_json::json!({
                "backed_up_at": backed_up_at,
                "restored_at": restored_at,
                "binary_version": crate::VERSION,
                "schema_version": schema_version,
            })
            .to_string(),
            micros(restored_at),
        ),
    )?;
    conn.pragma_update(None, "journal_mode", "DELETE")?;
    drop(conn);
    File::open(staged)
        .and_then(|file| file.sync_all())
        .map_err(io(format!("syncing {}", staged.display())))?;
    Ok(())
}

/// Moves the live database and its WAL files aside, under one name so the
/// old store still opens with its WAL, and renames `staged` into place. A
/// rename that fails puts back what had moved.
fn install(
    dir: &Path,
    live: &Path,
    staged: &Path,
    restored_at: Timestamp,
) -> Result<Option<PathBuf>, RestoreError> {
    let aside = if live.exists() {
        Some(aside_name(dir, restored_at))
    } else {
        None
    };
    let mut moves = Vec::new();
    if let Some(aside) = &aside {
        moves.push((live.to_owned(), aside.clone()));
        for suffix in ["-wal", "-shm"] {
            let from = with_suffix(live, suffix);
            if from.exists() {
                moves.push((from, with_suffix(aside, suffix)));
            }
        }
    }
    moves.push((staged.to_owned(), live.to_owned()));
    let mut done: Vec<&(PathBuf, PathBuf)> = Vec::new();
    for step in &moves {
        if let Err(error) = std::fs::rename(&step.0, &step.1) {
            for (from, to) in done.into_iter().rev() {
                let _ = std::fs::rename(to, from);
            }
            return Err(io(format!(
                "moving {} to {}",
                step.0.display(),
                step.1.display()
            ))(error)
            .into());
        }
        done.push(step);
    }
    File::open(dir)
        .and_then(|dir| dir.sync_all())
        .map_err(io(format!("syncing {}", dir.display())))?;
    Ok(aside)
}

/// `asphodel.db.before-restore-<time>`, with a counter when a restore in
/// the same second already took that name.
fn aside_name(dir: &Path, restored_at: Timestamp) -> PathBuf {
    let stamp = restored_at.strftime("%Y%m%dT%H%M%SZ").to_string();
    let base = dir.join(format!("{DB_FILE}.before-restore-{stamp}"));
    let taken = |path: &Path| {
        ["", "-wal", "-shm"]
            .iter()
            .any(|suffix| std::fs::symlink_metadata(with_suffix(path, suffix)).is_ok())
    };
    if !taken(&base) {
        return base;
    }
    (1..)
        .map(|n| with_suffix(&base, &format!(".{n}")))
        .find(|path| !taken(path))
        .expect("an unused name")
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn io(context: String) -> impl Fn(std::io::Error) -> StoreError {
    move |error| StoreError::Io {
        context: context.clone(),
        error,
    }
}

/// What `GET /v1/status` and `asphodel status` report (ADR 0010).
#[derive(Debug, Clone, Serialize)]
pub struct Status {
    pub version: &'static str,
    /// What an operator should look at, one line each. Empty when nothing
    /// needs it; `asphodel status` exits non-zero otherwise.
    pub attention: Vec<String>,
    /// Whether purge runs, with the stored fingerprint when it's paused.
    pub purge: PurgePause,
    /// The running daemon's deletion fingerprint, the hash an ack quotes.
    pub deletion_fingerprint: Fingerprint,
    pub banks: BTreeMap<String, BankStatus>,
    /// The latest sweep run of any bank.
    pub last_sweep: Option<SweepRun>,
    /// The newest pre-migration copy still in the data dir.
    pub pre_migration_copy: Option<PreMigrationCopy>,
    /// When `POST /v1/backup` last completed, which says nothing about
    /// whether the stream reached its destination.
    pub last_backup_at: Option<Timestamp>,
}

/// One bank's queue and failures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BankStatus {
    /// Chunks waiting or in flight.
    pub queued: usize,
    pub failed_chunks: usize,
    /// Enabled mental models whose last refresh failed.
    pub failed_refreshes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PreMigrationCopy {
    pub from_version: u32,
    pub path: PathBuf,
    pub expires_at: Timestamp,
}

/// One bank's nightly sweep: counts only, with the fingerprint and δ it ran
/// under.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SweepRun {
    pub bank: String,
    pub started_at: Timestamp,
    pub completed_at: Timestamp,
    pub fingerprint: String,
    pub delta: Option<f64>,
    pub purged_memories: u64,
    pub swept_sources: u64,
    pub swept_chunks: u64,
    pub swept_failed_chunks: u64,
    pub swept_recalls: u64,
}

/// Builds the status under `purge` and the running daemon's fingerprint.
/// The queue and failure counts are the ones `chunks` and `model list`
/// show.
pub(crate) fn status(
    store: &Store,
    purge: PurgePause,
    current: Fingerprint,
) -> Result<Status, StoreError> {
    let conn = store.connection();
    let mut attention = Vec::new();
    let mut statuses = BTreeMap::new();
    let mut banks = conn.prepare(
        "SELECT b.name,
                (SELECT COUNT(*) FROM extraction_queue q WHERE q.bank_id = b.id AND q.kind = 'chunk'),
                (SELECT COUNT(*) FROM chunks c WHERE c.bank_id = b.id AND c.failed_at IS NOT NULL),
                (SELECT COUNT(*) FROM mental_models m
                 WHERE m.bank_id = b.id AND m.enabled = 1 AND m.last_error_kind IS NOT NULL)
         FROM banks b ORDER BY b.id",
    )?;
    let count = |value: i64| usize::try_from(value).unwrap_or(0);
    let banks = banks
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                count(row.get(1)?),
                count(row.get(2)?),
                count(row.get(3)?),
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    for (name, queued, failed_chunks, failed_refreshes) in banks {
        if failed_chunks > 0 {
            attention.push(format!(
                "{name}: {failed_chunks} failed chunks; see `asphodel chunks --bank {name} --failed`"
            ));
        }
        if failed_refreshes > 0 {
            attention.push(format!(
                "{name}: {failed_refreshes} mental models failed their last refresh; see `asphodel model list --bank {name}`"
            ));
        }
        statuses.insert(
            name,
            BankStatus {
                queued,
                failed_chunks,
                failed_refreshes,
            },
        );
    }
    if let PurgePause::Paused { stored } = &purge {
        attention.push(format!(
            "purge is paused: the deletion fingerprint changed from {stored} to {current}; \
             see `asphodel purge plan`, then `asphodel purge ack --hash {current}`",
            stored = stored.as_str(),
            current = current.as_str(),
        ));
    }
    let last_sweep = sweep_runs(&conn, None, 1)?.into_iter().next();
    let pre_migration_copy = migrations::copies(&conn, store.dir())?
        .into_iter()
        .max_by_key(|(from, _, _)| *from)
        .map(|(from_version, path, expires_at)| PreMigrationCopy {
            from_version,
            path,
            expires_at,
        });
    let last_backup_at: Option<String> = conn
        .query_row(
            "SELECT value FROM store_meta WHERE key = ?1",
            [META_LAST_BACKUP],
            |row| row.get(0),
        )
        .optional()?;
    Ok(Status {
        version: crate::VERSION,
        attention,
        purge,
        deletion_fingerprint: current,
        banks: statuses,
        last_sweep,
        pre_migration_copy,
        last_backup_at: last_backup_at.and_then(|at| at.parse().ok()),
    })
}

/// The sweep runs, newest first: one bank's, or every bank's.
fn sweep_runs(
    conn: &Connection,
    bank_id: Option<i64>,
    limit: usize,
) -> Result<Vec<SweepRun>, rusqlite::Error> {
    let mut statement = conn.prepare(
        "SELECT b.name, r.started_at, r.completed_at, r.fingerprint, r.delta, r.purged_memories,
                r.swept_sources, r.swept_chunks, r.swept_failed_chunks, r.swept_recalls
         FROM sweep_runs r JOIN banks b ON b.id = r.bank_id
         WHERE ?1 IS NULL OR r.bank_id = ?1
         ORDER BY r.completed_at DESC, r.id DESC
         LIMIT ?2",
    )?;
    let count = |value: i64| u64::try_from(value).unwrap_or(0);
    statement
        .query_map((bank_id, limit as i64), |row| {
            Ok(SweepRun {
                bank: row.get(0)?,
                started_at: timestamp(row.get(1)?),
                completed_at: timestamp(row.get(2)?),
                fingerprint: row.get(3)?,
                delta: row.get(4)?,
                purged_memories: count(row.get(5)?),
                swept_sources: count(row.get(6)?),
                swept_chunks: count(row.get(7)?),
                swept_failed_chunks: count(row.get(8)?),
                swept_recalls: count(row.get(9)?),
            })
        })?
        .collect()
}

#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("unknown bank")]
    UnknownBank,

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for AuditError {
    fn from(error: rusqlite::Error) -> Self {
        AuditError::Store(StoreError::Sqlite(error))
    }
}

/// The audit lists (TIM-99, decision 7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditList {
    Purges,
    Forgets,
    Sweeps,
    Recalls,
}

impl AuditList {
    pub fn as_str(self) -> &'static str {
        match self {
            AuditList::Purges => "purges",
            AuditList::Forgets => "forgets",
            AuditList::Sweeps => "sweeps",
            AuditList::Recalls => "recalls",
        }
    }
}

/// One purge: the memories it deleted. Counts and ids only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PurgeRow {
    pub edit: Uuid,
    pub at: Timestamp,
    pub memories: Vec<Uuid>,
}

/// One forget: the chain it hid, and the key of the turn that asked for it
/// once that turn arrives. `pending` while the erase waits behind the
/// chunks queued before it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgetRow {
    pub edit: Uuid,
    pub at: Timestamp,
    pub memories: Vec<Uuid>,
    pub session_id: Option<String>,
    pub request: serde_json::Value,
    pub pending: bool,
}

/// One recall and what came back. The query is content, and the sweep
/// clears it at the 90-day horizon.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RecallRow {
    pub id: Uuid,
    pub kind: String,
    pub session_id: Option<String>,
    pub at: Timestamp,
    pub query: Option<String>,
    pub latency_ms: i64,
    pub swept_at: Option<Timestamp>,
    pub results: Vec<Uuid>,
}

/// One audit list, as `GET /v1/banks/{bank}/{list}` returns it: the rows
/// under the list's own name, newest first.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Audit {
    Purges { purges: Vec<PurgeRow> },
    Forgets { forgets: Vec<ForgetRow> },
    Sweeps { sweeps: Vec<SweepRun> },
    Recalls { recalls: Vec<RecallRow> },
}

/// Reads one audit list of `bank`, newest first, at most `limit` rows
/// ([`DEFAULT_LIST`] when not given, never more than [`MAX_LIST`]).
pub(crate) fn audit(
    store: &Store,
    bank: &str,
    list: AuditList,
    limit: Option<usize>,
) -> Result<Audit, AuditError> {
    let conn = store.connection();
    let (bank_id, _) = crate::ingest::find_bank(&conn, bank)?.ok_or(AuditError::UnknownBank)?;
    let conn = &*conn;
    let limit = limit.unwrap_or(DEFAULT_LIST).clamp(1, MAX_LIST);
    Ok(match list {
        AuditList::Purges => Audit::Purges {
            purges: edits(conn, bank_id, EDIT_PURGED, limit)?
                .into_iter()
                .map(|(edit, at, details)| PurgeRow {
                    edit,
                    at,
                    memories: details["memories"]
                        .as_array()
                        .map_or(&[][..], Vec::as_slice)
                        .iter()
                        .filter_map(|row| row["memory"].as_str()?.parse().ok())
                        .collect(),
                })
                .collect(),
        },
        AuditList::Forgets => {
            let mut forgets = Vec::new();
            for (edit, at, details) in edits(conn, bank_id, EDIT_FORGET, limit)? {
                let memories: Vec<Uuid> = details["memories"]
                    .as_array()
                    .map_or(&[][..], Vec::as_slice)
                    .iter()
                    .filter_map(|id| id.as_str()?.parse().ok())
                    .collect();
                let mut remaining =
                    conn.prepare_cached("SELECT 1 FROM memories WHERE uuid = ?1")?;
                let mut pending = false;
                for memory in &memories {
                    pending |= remaining.exists([memory.to_string()])?;
                }
                forgets.push(ForgetRow {
                    edit,
                    at,
                    memories,
                    session_id: details["session_id"].as_str().map(str::to_owned),
                    request: details["request"].clone(),
                    pending,
                });
            }
            Audit::Forgets { forgets }
        }
        AuditList::Sweeps => Audit::Sweeps {
            sweeps: sweep_runs(conn, Some(bank_id), limit)?,
        },
        AuditList::Recalls => Audit::Recalls {
            recalls: recalls(conn, bank_id, limit)?,
        },
    })
}

/// The bank's edit rows of `kind`, newest first, with their details.
fn edits(
    conn: &Connection,
    bank_id: i64,
    kind: &str,
    limit: usize,
) -> Result<Vec<(Uuid, Timestamp, serde_json::Value)>, rusqlite::Error> {
    let mut statement = conn.prepare(
        "SELECT uuid, at, details FROM edits WHERE bank_id = ?1 AND kind = ?2
         ORDER BY at DESC, id DESC LIMIT ?3",
    )?;
    statement
        .query_map((bank_id, kind, limit as i64), |row| {
            let details: String = row.get(2)?;
            Ok((
                row.get::<_, String>(0)?
                    .parse()
                    .expect("a stored uuid parses"),
                timestamp(row.get(1)?),
                serde_json::from_str(&details).unwrap_or(serde_json::Value::Null),
            ))
        })?
        .collect()
}

fn recalls(
    conn: &Connection,
    bank_id: i64,
    limit: usize,
) -> Result<Vec<RecallRow>, rusqlite::Error> {
    let mut statement = conn.prepare(
        "SELECT id, uuid, kind, session_id, at, query, latency_ms, swept_at FROM recalls
         WHERE bank_id = ?1 ORDER BY at DESC, id DESC LIMIT ?2",
    )?;
    let mut results = conn.prepare(
        "SELECT m.uuid FROM recall_results r JOIN memories m ON m.id = r.memory_id
         WHERE r.recall_id = ?1 ORDER BY r.rank",
    )?;
    let rows = statement
        .query_map((bank_id, limit as i64), |row| {
            Ok((
                row.get::<_, i64>(0)?,
                RecallRow {
                    id: row
                        .get::<_, String>(1)?
                        .parse()
                        .expect("a stored uuid parses"),
                    kind: row.get(2)?,
                    session_id: row.get(3)?,
                    at: timestamp(row.get(4)?),
                    query: row.get(5)?,
                    latency_ms: row.get(6)?,
                    swept_at: row.get::<_, Option<i64>>(7)?.map(timestamp),
                    results: Vec::new(),
                },
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let mut recalls = Vec::with_capacity(rows.len());
    for (id, mut row) in rows {
        row.results = results
            .query_map([id], |row| row.get::<_, String>(0))?
            .map(|uuid| Ok(uuid?.parse().expect("a stored uuid parses")))
            .collect::<Result<_, rusqlite::Error>>()?;
        recalls.push(row);
    }
    Ok(recalls)
}
