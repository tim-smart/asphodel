//! Replacing a file through a temp file and a rename, without trusting
//! whatever already sits next to it.

use std::fs::{File, OpenOptions, Permissions};
use std::io::{ErrorKind, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// How many taken temp names to step past before giving up. Each try uses
/// a new name, so only a directory filling up with planted names gets here.
const ATTEMPTS: usize = 64;

/// Writes `bytes` to `path` with permissions `mode`, through a new temp
/// file in the same directory and a rename, so a reader sees the old file
/// or the new one and never a partial write.
///
/// The temp file is created exclusively under a fresh name, so it never
/// follows a symlink or reuses a file someone else put there. A name that
/// is taken is left alone and the next one is tried. On failure only the
/// temp file this call created is removed. A symlink at `path` itself is
/// replaced by the rename, not followed.
pub(crate) fn replace_file(path: &Path, bytes: &[u8], mode: u32) -> std::io::Result<()> {
    let (temp, mut file) = create_temp(path, mode)?;
    let written = file
        .set_permissions(Permissions::from_mode(mode))
        .and_then(|()| file.write_all(bytes))
        .and_then(|()| file.sync_all())
        .and_then(|()| std::fs::rename(&temp, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    written
}

/// A new file next to `path`, named from it, the pid and a process-wide
/// counter. The mode is set at creation (less the umask) so the file is
/// never readable beyond `mode`, even before [`replace_file`] fixes it
/// exactly.
pub(crate) fn create_temp(path: &Path, mode: u32) -> std::io::Result<(PathBuf, File)> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut last = None;
    for _ in 0..ATTEMPTS {
        let mut temp = path.as_os_str().to_owned();
        temp.push(format!(
            ".part-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let temp = PathBuf::from(temp);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(mode)
            .open(&temp)
        {
            Ok(file) => return Ok((temp, file)),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => last = Some(error),
            Err(error) => return Err(error),
        }
    }
    Err(last.expect("at least one attempt"))
}
