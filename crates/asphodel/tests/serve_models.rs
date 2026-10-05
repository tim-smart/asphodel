//! `asphodel serve` and the models, run as a process.
//!
//! The daemon loads the models from the model dir
//! before it is ready, never downloads, fails fast on a missing file, and
//! refuses to start without a floor for each loaded model. These tests see
//! only what an operator sees: exit codes, stderr and the model dir.
//!
//! The daemon can't load the real models on a CI machine. The empty model
//! dir test leaves `ASPHODEL_MODELS` unset, so it exercises real-model
//! loading validation without model artifacts or ONNX Runtime; the floor
//! test runs on the fakes (`ASPHODEL_MODELS=fake`, environment only).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-serve-models-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Runs `asphodel serve` with its own data dir under `dir`, after `edit`,
/// and a clean environment, so the caller's `ASPHODEL_*` variables can't
/// leak in. It must refuse to start; returns its stderr.
fn refused(dir: &TestDir, edit: impl FnOnce(&mut Command) -> &mut Command) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_asphodel"));
    command
        .env_clear()
        .arg("serve")
        .arg("--data-dir")
        .arg(dir.0.join("data"))
        .args(["--listen", "127.0.0.1:0"]);
    let output = edit(&mut command).stdin(Stdio::null()).output().unwrap();
    assert!(!output.status.success());
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn an_empty_model_dir_stops_startup_naming_the_missing_file() {
    // Never a download: an empty dir is still empty afterwards, and stderr
    // names the file in the model dir: `--model-dir`, else an absolute
    // `XDG_CACHE_HOME`, else `$HOME/.cache`. A relative `XDG_CACHE_HOME` is
    // ignored, as the XDG spec says.
    let dir = TestDir::new();
    let models = dir.0.join("models");
    std::fs::create_dir_all(&models).unwrap();
    let (home, xdg) = (dir.0.join("home"), dir.0.join("xdg"));
    let under_home = home.join(".cache/asphodel/models");
    let cases = [
        (Some(&models), Some(xdg.as_path()), &models),
        (None, Some(&xdg), &xdg.join("asphodel/models")),
        (None, None, &under_home),
        (None, Some(Path::new("cache")), &under_home),
    ];
    for (flag, xdg, expected) in cases {
        let stderr = refused(&dir, |c| {
            c.env("HOME", &home);
            if let Some(xdg) = xdg {
                c.env("XDG_CACHE_HOME", xdg);
            }
            match flag {
                Some(models) => c.arg("--model-dir").arg(models),
                None => c,
            }
        });
        let missing = format!("{}/", expected.display());
        assert!(stderr.contains(&missing), "{expected:?}: {stderr}");
        assert!(stderr.contains("model.onnx"), "{stderr}");
    }
    assert_eq!(std::fs::read_dir(&models).unwrap().count(), 0);
}

#[test]
fn a_floor_for_the_real_models_does_not_cover_the_fakes() {
    // A missing floor for a configured model stops the daemon,
    // whichever models are configured. Floors are keyed by the exact model
    // string, so a production tuning file doesn't make a fake daemon start,
    // and the switch can't hide a missing floor.
    let dir = TestDir::new();
    let tuning = dir.0.join("tuning.toml");
    std::fs::write(
        &tuning,
        "[injection.reranker_floors]\n\"jina-reranker-v1-turbo-en:int8\" = -1.5\n\
         [ranking.relevance_scales]\n\"jina-reranker-v1-turbo-en:int8\" = 1.0\n\
         [reconcile.embedding_floors]\n\"bge-small-en-v1.5:int8\" = 0.82\n",
    )
    .unwrap();
    let stderr = refused(&dir, |c| {
        c.env("ASPHODEL_MODELS", "fake")
            .arg("--config")
            .arg(&tuning)
    });
    // The fake models' ids (`asphodel_core::models`).
    for missing in [
        "reconcile.embedding_floors.\"fake-embedder:v1\"",
        "injection.reranker_floors.\"fake-reranker:v1\"",
        "ranking.relevance_scales.\"fake-reranker:v1\"",
    ] {
        assert!(stderr.contains(missing), "{missing}: {stderr}");
    }
}
