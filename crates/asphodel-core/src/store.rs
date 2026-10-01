//! The embedded store: one SQLite database under the data dir.
//!
//! "Rust storage and search stack" (TIM-89) chose rusqlite in WAL mode,
//! FTS5 for text and sqlite-vec's flat vec0 tables for vectors, all in one
//! file so records, text index and vectors commit together. "API surface and
//! Hermes transport" (TIM-94, decision 4) adds the data-dir lock and the
//! network-filesystem refusal. [`Store::open`] does, in order:
//!
//! 1. check the data dir is a directory, creating it when it is missing;
//! 2. refuse a network filesystem unless [`OpenOptions::allow_network_fs`];
//! 3. take the exclusive lock, so one daemon owns a data dir at a time;
//! 4. open [`DB_FILE`] in WAL mode with foreign keys on;
//! 5. take the pre-migration copy and run any pending migrations (ADR 0010);
//! 6. delete pre-migration copies older than seven days.
//!
//! The service repeats step 6 through [`Store::expire_copies`], waking at
//! the deadline [`Store::next_copy_expiry`] gives.
//!
//! Every timestamp written here comes from the [`Clock`] the store was
//! opened with; no SQL reads SQLite's clock (TIM-90).

pub mod bank;
pub mod fs;
mod ids;
mod lock;
pub mod migrations;
mod time;
pub mod vector;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, Once};

use jiff::Timestamp;
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::clock::Clock;
use crate::config::{Fingerprint, PurgePause};

pub use fs::FilesystemKind;
pub use lock::DataDirLock;
pub use migrations::{Applied, SCHEMA_VERSION};
pub use time::{micros, timestamp};
pub use vector::{EMBEDDING_DIMENSIONS, Neighbour, SqliteVec, VectorError, VectorIndex};

/// The database file under the data dir.
pub const DB_FILE: &str = "asphodel.db";

/// The lock file under the data dir. It holds the pid of the daemon that
/// has it, for operators; the lock itself is `flock`, so the kernel releases
/// it when the process dies.
pub const LOCK_FILE: &str = "lock";

/// The `store_meta` key of the stored deletion fingerprint (ADR 0009).
pub const META_DELETION_FINGERPRINT: &str = "deletion_fingerprint";

/// How to open a store.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenOptions {
    /// `--allow-network-fs`: run on NFS, SMB/CIFS, CephFS or FUSE anyway.
    pub allow_network_fs: bool,
}

/// Why a store couldn't be opened or used. Every variant that stops the
/// daemon names the data dir, and none of them carries content.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("data dir {} is not a directory", dir.display())]
    NotADirectory { dir: PathBuf },

    #[error(
        "data dir {} is locked by another asphodel process (pid {holder}); one daemon owns a data dir at a time",
        dir.display()
    )]
    Locked { dir: PathBuf, holder: String },

    #[error(
        "data dir {} is on {kind}; SQLite's locking is not safe on a network filesystem, pass --allow-network-fs to run anyway",
        dir.display()
    )]
    NetworkFilesystem { dir: PathBuf, kind: FilesystemKind },

    #[error(
        "the store in {} is schema version {found}, newer than this binary's {supported}",
        dir.display()
    )]
    NewerSchema {
        dir: PathBuf,
        found: u32,
        supported: u32,
    },

    #[error("migrating the store in {} from version {from}: {error}", dir.display())]
    Migration {
        dir: PathBuf,
        from: u32,
        error: rusqlite::Error,
    },

    #[error(
        "pre-migration copy {} failed its integrity check ({detail}); it is never overwritten, so move it aside to migrate",
        path.display()
    )]
    CorruptCopy { path: PathBuf, detail: String },

    #[error("{context}: {error}")]
    Io {
        context: String,
        error: std::io::Error,
    },

    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
}

impl StoreError {
    fn io(context: impl Into<String>) -> impl FnOnce(std::io::Error) -> Self {
        let context = context.into();
        move |error| StoreError::Io { context, error }
    }
}

/// One open store: the connection, the lock and the clock it writes times
/// from. The connection is behind a mutex; readers never block each other in
/// WAL mode, and a reader pool is a later stage's concern.
pub struct Store {
    dir: PathBuf,
    conn: Mutex<Connection>,
    clock: Arc<dyn Clock>,
    ids: ids::IdSource,
    applied: Option<Applied>,
    filesystem: FilesystemKind,
    _lock: DataDirLock,
}

impl Store {
    /// Opens the store under `dir`. See the module docs for the order of
    /// checks.
    pub fn open(
        dir: &Path,
        options: OpenOptions,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, StoreError> {
        prepare_dir(dir)?;
        let filesystem = fs::check_data_dir(dir, options.allow_network_fs)?;
        let lock = DataDirLock::acquire(dir)?;

        register_extensions();
        let path = dir.join(DB_FILE);
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let mut conn = Connection::open_with_flags(&path, flags)?;
        configure(&conn)?;

        let found = migrations::version(&conn)?;
        if found > SCHEMA_VERSION {
            return Err(StoreError::NewerSchema {
                dir: dir.to_owned(),
                found,
                supported: SCHEMA_VERSION,
            });
        }
        let applied = if found < SCHEMA_VERSION {
            let copy = if found > 0 {
                Some(migrations::take_copy(&conn, dir, found)?)
            } else {
                None
            };
            let applied = migrations::apply(&mut conn, clock.as_ref(), copy).map_err(|error| {
                StoreError::Migration {
                    dir: dir.to_owned(),
                    from: found,
                    error,
                }
            })?;
            info!(
                dir = %dir.display(),
                from = applied.from,
                to = applied.to,
                copy = applied.copy.as_ref().map(|path| path.display().to_string()),
                "migrated the store"
            );
            Some(applied)
        } else {
            None
        };
        expire_copies(&conn, dir, clock.now())?;

        info!(
            dir = %dir.display(),
            schema_version = SCHEMA_VERSION,
            filesystem = %filesystem,
            "opened the store"
        );
        Ok(Self {
            dir: dir.to_owned(),
            conn: Mutex::new(conn),
            clock,
            ids: ids::IdSource::new(),
            applied,
            filesystem,
            _lock: lock,
        })
    }

    /// The data dir.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The migration this open ran, if any.
    pub fn applied(&self) -> Option<&Applied> {
        self.applied.as_ref()
    }

    /// What the data dir sits on.
    pub fn filesystem(&self) -> FilesystemKind {
        self.filesystem
    }

    /// The current world time, from the store's clock.
    pub fn now(&self) -> Timestamp {
        self.clock.now()
    }

    /// The clock the store writes times from.
    pub fn clock(&self) -> &dyn Clock {
        self.clock.as_ref()
    }

    /// A fresh UUIDv7 for a row's public id, timed by the store's clock.
    pub fn new_id(&self) -> Uuid {
        self.ids.next(self.clock.now())
    }

    /// The connection, for the service layer. Held for one operation at a
    /// time; never across an await.
    pub fn connection(&self) -> MutexGuard<'_, Connection> {
        self.conn
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The vector index over memory content.
    pub fn vectors(&self) -> SqliteVec {
        SqliteVec
    }

    /// The schema version the open database is at.
    pub fn schema_version(&self) -> Result<u32, StoreError> {
        Ok(migrations::version(&self.connection())?)
    }

    /// Compares the binary's deletion fingerprint with the stored one (ADR
    /// 0009). On first start the fingerprint is just recorded. A different
    /// stored value pauses purge until an operator acknowledges it.
    pub fn check_fingerprint(&self, current: &Fingerprint) -> Result<PurgePause, StoreError> {
        let conn = self.connection();
        let stored: Option<String> = conn
            .query_row(
                "SELECT value FROM store_meta WHERE key = ?1",
                [META_DELETION_FINGERPRINT],
                |row| row.get(0),
            )
            .optional()?;
        match stored {
            None => {
                conn.execute(
                    "INSERT INTO store_meta (key, value, updated_at) VALUES (?1, ?2, ?3)",
                    (
                        META_DELETION_FINGERPRINT,
                        current.as_str(),
                        micros(self.clock.now()),
                    ),
                )?;
                info!("recorded the deletion fingerprint on first start");
                Ok(PurgePause::Running)
            }
            Some(stored) if stored == current.as_str() => Ok(PurgePause::Running),
            Some(stored) => {
                warn!(
                    "the deletion fingerprint changed; purge and the sweep are paused until acknowledged"
                );
                Ok(PurgePause::Paused {
                    stored: Fingerprint::from_stored(stored),
                })
            }
        }
    }

    /// Deletes the pre-migration copies past their seven days on the store's
    /// clock (ADR 0010). Open does this once; the service's housekeeping does
    /// it while the daemon runs, so the bound holds without a restart.
    pub fn expire_copies(&self) -> Result<Vec<PathBuf>, StoreError> {
        expire_copies(&self.connection(), &self.dir, self.clock.now())
    }

    /// When the next pre-migration copy on disk is due for deletion, so the
    /// daemon can wake for it rather than wait for its next poll.
    pub fn next_copy_expiry(&self) -> Result<Option<Timestamp>, StoreError> {
        migrations::next_copy_expiry(&self.connection(), &self.dir)
    }

    /// Checkpoints the WAL into the database file, as the daemon does on
    /// SIGTERM (TIM-94, decision 3).
    pub fn checkpoint(&self) -> Result<(), StoreError> {
        self.connection()
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(())
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        if let Err(error) = self.checkpoint() {
            debug!(%error, "checkpoint on close failed");
        }
    }
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("dir", &self.dir)
            .field("filesystem", &self.filesystem)
            .finish_non_exhaustive()
    }
}

/// [`migrations::expire_copies`], logging each copy it deletes.
fn expire_copies(
    conn: &Connection,
    dir: &Path,
    now: Timestamp,
) -> Result<Vec<PathBuf>, StoreError> {
    let removed = migrations::expire_copies(conn, dir, now)?;
    for path in &removed {
        info!(copy = %path.display(), "deleted an expired pre-migration copy");
    }
    Ok(removed)
}

/// Makes sure `dir` is a directory, creating it when nothing is there. A
/// file at the path is refused and left alone.
fn prepare_dir(dir: &Path) -> Result<(), StoreError> {
    match std::fs::metadata(dir) {
        Ok(metadata) if metadata.is_dir() => Ok(()),
        Ok(_) => Err(StoreError::NotADirectory {
            dir: dir.to_owned(),
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(dir).map_err(StoreError::io(format!(
                "creating data dir {}",
                dir.display()
            )))?;
            info!(dir = %dir.display(), "created the data dir");
            Ok(())
        }
        Err(error) => Err(StoreError::io(format!(
            "inspecting data dir {}",
            dir.display()
        ))(error)),
    }
}

/// Registers sqlite-vec with SQLite once, before any connection opens, so
/// every connection has `vec0` (TIM-89).
fn register_extensions() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // SAFETY: `sqlite3_vec_init` is sqlite-vec's extension entry point,
        // with the `sqlite3_loadext_entry` signature SQLite expects. It is
        // registered once, before any connection exists.
        unsafe {
            let entry: unsafe extern "C" fn() = sqlite_vec::sqlite3_vec_init;
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
                unsafe extern "C" fn(),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut std::os::raw::c_char,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> std::os::raw::c_int,
            >(entry)));
        }
    });
}

/// Per-connection settings: WAL with `synchronous=NORMAL` (TIM-89), foreign
/// keys on, and a busy timeout so a checkpoint never fails a request.
fn configure(conn: &Connection) -> Result<(), StoreError> {
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;
         PRAGMA temp_store = MEMORY;",
    )?;
    Ok(())
}
