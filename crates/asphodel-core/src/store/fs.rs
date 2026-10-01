//! Which filesystem the data dir sits on.
//!
//! SQLite's locking and WAL need a local filesystem: `flock` and shared
//! memory are unreliable over NFS and SMB, and FUSE mounts make no promises.
//! So the daemon calls `statfs` at startup and refuses NFS, SMB/CIFS, CephFS
//! and FUSE unless given `--allow-network-fs` (TIM-94, decision 4).
//!
//! [`classify`] is a pure function of the filesystem magic so the refusal
//! can be tested without mounting anything.

use std::fmt;
use std::io;
use std::path::Path;

use serde::Serialize;

use super::StoreError;

/// What a data dir sits on, as far as the refusal cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FilesystemKind {
    /// Anything the daemon is happy to run on.
    Local,
    Nfs,
    /// SMB, CIFS and SMB2 mounts.
    Smb,
    Ceph,
    Fuse,
}

impl FilesystemKind {
    /// Whether the daemon refuses this kind without `--allow-network-fs`.
    pub fn is_network(self) -> bool {
        !matches!(self, FilesystemKind::Local)
    }
}

impl fmt::Display for FilesystemKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            FilesystemKind::Local => "a local filesystem",
            FilesystemKind::Nfs => "NFS",
            FilesystemKind::Smb => "SMB/CIFS",
            FilesystemKind::Ceph => "CephFS",
            FilesystemKind::Fuse => "FUSE",
        })
    }
}

/// Linux `f_type` magics, from `include/uapi/linux/magic.h` and the NFS,
/// CIFS and Ceph headers.
pub mod magic {
    pub const NFS: u32 = 0x6969;
    pub const SMB: u32 = 0x517B;
    pub const CIFS: u32 = 0xFF53_4D42;
    pub const SMB2: u32 = 0xFE53_4D42;
    pub const CEPH: u32 = 0x00C3_6400;
    pub const FUSE: u32 = 0x6573_5546;
}

/// Classifies a Linux `statfs` `f_type` magic. Anything not on the refusal
/// list is local, which is the safe direction for an unknown magic: an
/// unexpected refusal would stop the daemon, and the operator can't fix a
/// magic number.
pub fn classify(f_type: u32) -> FilesystemKind {
    match f_type {
        magic::NFS => FilesystemKind::Nfs,
        magic::SMB | magic::CIFS | magic::SMB2 => FilesystemKind::Smb,
        magic::CEPH => FilesystemKind::Ceph,
        magic::FUSE => FilesystemKind::Fuse,
        _ => FilesystemKind::Local,
    }
}

/// Classifies a filesystem type name, as `statfs` reports it on macOS and
/// the BSDs (`f_fstypename`).
pub fn classify_name(name: &str) -> FilesystemKind {
    let name = name.to_ascii_lowercase();
    if name == "nfs" || name.starts_with("nfs") {
        FilesystemKind::Nfs
    } else if name == "smbfs" || name == "cifs" || name.starts_with("smb") {
        FilesystemKind::Smb
    } else if name == "ceph" || name == "cephfs" {
        FilesystemKind::Ceph
    } else if name.contains("fuse") {
        FilesystemKind::Fuse
    } else {
        FilesystemKind::Local
    }
}

/// What `path` sits on, by `statfs`.
pub fn filesystem_kind(path: &Path) -> io::Result<FilesystemKind> {
    platform::filesystem_kind(path)
}

/// Checks the data dir's filesystem and refuses a network one unless
/// `allow_network_fs`.
pub fn check_data_dir(dir: &Path, allow_network_fs: bool) -> Result<FilesystemKind, StoreError> {
    let kind = filesystem_kind(dir).map_err(StoreError::io(format!(
        "checking the filesystem of {}",
        dir.display()
    )))?;
    if kind.is_network() {
        if allow_network_fs {
            tracing::warn!(
                dir = %dir.display(),
                filesystem = %kind,
                "running on a network filesystem because --allow-network-fs was given"
            );
        } else {
            return Err(StoreError::NetworkFilesystem {
                dir: dir.to_owned(),
                kind,
            });
        }
    }
    Ok(kind)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod platform {
    use std::ffi::CString;
    use std::io;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    use super::{FilesystemKind, classify};

    pub fn filesystem_kind(path: &Path) -> io::Result<FilesystemKind> {
        let path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains a NUL byte"))?;
        let mut buf = std::mem::MaybeUninit::<libc::statfs>::uninit();
        // SAFETY: `path` is a valid NUL-terminated string and `buf` is a
        // correctly sized out-parameter that `statfs` fills on success.
        let rc = unsafe { libc::statfs(path.as_ptr(), buf.as_mut_ptr()) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `statfs` returned 0, so `buf` is initialised.
        let buf = unsafe { buf.assume_init() };
        // `f_type` is a signed word on glibc. Going through `u64` keeps the
        // low 32 bits whether the field is 32 or 64 bits wide, so a magic
        // above `i32::MAX` such as CIFS compares correctly on both.
        let magic = buf.f_type as u64 as u32;
        Ok(classify(magic))
    }
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd"
))]
mod platform {
    use std::ffi::{CStr, CString};
    use std::io;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    use super::{FilesystemKind, classify_name};

    pub fn filesystem_kind(path: &Path) -> io::Result<FilesystemKind> {
        let path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains a NUL byte"))?;
        let mut buf = std::mem::MaybeUninit::<libc::statfs>::uninit();
        // SAFETY: as on Linux; `statfs` fills `buf` on success.
        let rc = unsafe { libc::statfs(path.as_ptr(), buf.as_mut_ptr()) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `statfs` returned 0, so `buf` is initialised, and
        // `f_fstypename` is a NUL-terminated name within its array.
        let buf = unsafe { buf.assume_init() };
        let name = unsafe { CStr::from_ptr(buf.f_fstypename.as_ptr()) };
        Ok(classify_name(&name.to_string_lossy()))
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd"
)))]
mod platform {
    use std::io;
    use std::path::Path;

    use super::FilesystemKind;

    /// No `statfs` here, so nothing can be refused.
    pub fn filesystem_kind(_path: &Path) -> io::Result<FilesystemKind> {
        Ok(FilesystemKind::Local)
    }
}
