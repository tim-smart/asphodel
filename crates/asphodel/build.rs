//! Records the git SHA the binary was built from, so every replay report can
//! embed it (TIM-96, decision 6). `ASPHODEL_GIT_SHA` in the build
//! environment wins, so a build outside a checkout can still say what it is;
//! otherwise `git rev-parse HEAD` in the crate's checkout. Without either the
//! report's `git_sha` is null.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=ASPHODEL_GIT_SHA");
    if std::env::var_os("ASPHODEL_GIT_SHA").is_some() {
        return;
    }
    let Some(manifest_dir) = std::env::var_os("CARGO_MANIFEST_DIR") else {
        return;
    };
    let git = |args: &[&str]| -> Option<String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(&manifest_dir)
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    };
    // Rebuild when HEAD moves: the HEAD file, and the ref it points at.
    if let Some(head) = git(&["rev-parse", "--git-path", "HEAD"]) {
        let head_path = Path::new(&manifest_dir).join(&head);
        if head_path.is_file() {
            println!("cargo:rerun-if-changed={}", head_path.display());
            if let Ok(content) = std::fs::read_to_string(&head_path)
                && let Some(reference) = content.strip_prefix("ref: ")
                && let Some(path) = git(&["rev-parse", "--git-path", reference.trim()])
            {
                let path = Path::new(&manifest_dir).join(path);
                if path.is_file() {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
    }
    if let Some(sha) = git(&["rev-parse", "HEAD"])
        && sha.len() == 40
        && sha.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        println!("cargo:rustc-env=ASPHODEL_GIT_SHA={sha}");
    }
}
