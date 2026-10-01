//! The exclusive lock on the data dir (TIM-94, decision 4).
//!
//! It is an advisory `flock` on [`LOCK_FILE`](super::LOCK_FILE), so the
//! kernel releases it when the holder exits however it exits. A crashed
//! daemon never leaves a lock behind, and a supervisor's restart gets
//! straight in. The file's contents are the holder's pid, for an operator
//! reading the refusal; they are not what the lock rests on.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use super::{LOCK_FILE, StoreError};

/// The held lock. Dropping it releases the data dir.
#[derive(Debug)]
pub struct DataDirLock {
    path: PathBuf,
    file: File,
}

impl DataDirLock {
    /// Takes the lock, or reports who holds it.
    pub fn acquire(dir: &Path) -> Result<Self, StoreError> {
        let path = dir.join(LOCK_FILE);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(StoreError::io(format!(
                "opening lock file {}",
                path.display()
            )))?;
        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                let mut holder = String::new();
                let _ = file.read_to_string(&mut holder);
                let holder = holder.trim();
                return Err(StoreError::Locked {
                    dir: dir.to_owned(),
                    holder: if holder.is_empty() {
                        "unknown".to_owned()
                    } else {
                        holder.to_owned()
                    },
                });
            }
            Err(TryLockError::Error(error)) => {
                return Err(StoreError::io(format!("locking {}", path.display()))(error));
            }
        }
        // Best effort: the pid is a courtesy, and a failure to write it
        // must not stop a daemon that holds the lock.
        let _ = file
            .set_len(0)
            .and_then(|()| file.rewind())
            .and_then(|()| writeln!(file, "{}", std::process::id()));
        Ok(Self { path, file })
    }

    /// The lock file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for DataDirLock {
    fn drop(&mut self) {
        // Clear the pid so a reader doesn't mistake a dead one for a holder.
        // The kernel drops the lock itself when the file closes.
        let _ = self.file.set_len(0);
    }
}
