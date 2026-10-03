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

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

/// The fake models' ids (`asphodel_core::models`).
const FAKE_EMBEDDER: &str = "fake-embedder:v1";
const FAKE_RERANKER: &str = "fake-reranker:v1";

/// `asphodel` with a clean environment, so the caller's `ASPHODEL_*`
/// variables can't leak in.
fn asphodel() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_asphodel"));
    command.env_clear();
    command
}

/// `asphodel serve` with its own data dir under `dir`.
fn serve_in(dir: &TestDir) -> Command {
    let mut command = asphodel();
    command
        .arg("serve")
        .arg("--data-dir")
        .arg(dir.0.join("data"));
    command
}

fn run(command: &mut Command) -> Output {
    command.stdin(Stdio::null()).output().unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

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

    fn file(&self, name: &str, text: &str) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, text).unwrap();
        path
    }

    /// An empty model dir: the daemon must not fill it.
    fn empty_models(&self) -> PathBuf {
        let path = self.0.join("models");
        std::fs::create_dir_all(&path).unwrap();
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn an_empty_model_dir_stops_startup_naming_the_missing_file() {
    // Never a download: an empty dir is still empty afterwards, and stderr
    // names the file and the command that fills it.
    let dir = TestDir::new();
    let models = dir.empty_models();
    let output = run(serve_in(&dir)
        .arg("--model-dir")
        .arg(&models)
        .args(["--listen", "127.0.0.1:0"]));
    assert!(!output.status.success());
    let stderr = stderr(&output);
    assert!(stderr.contains("model.onnx"), "{stderr}");
    assert!(stderr.contains("asphodel models fetch"), "{stderr}");
    assert!(stderr.contains(models.to_str().unwrap()), "{stderr}");
    assert_eq!(std::fs::read_dir(&models).unwrap().count(), 0);
    // And the daemon never became ready: its listener answers 503 while the
    // models load, and "asphodel listening" is logged only once they have.
    assert!(!stderr.contains("asphodel listening"), "{stderr}");
}

#[test]
fn a_floor_for_the_real_models_does_not_cover_the_fakes() {
    // A missing floor for a configured model stops the daemon,
    // whichever models are configured. Floors are keyed by the exact model
    // string, so a production tuning file doesn't make a fake daemon start,
    // and the switch can't hide a missing floor.
    let dir = TestDir::new();
    let tuning = dir.file(
        "tuning.toml",
        "[injection.reranker_floors]\n\"jina-reranker-v1-turbo-en:int8\" = -1.5\n\
         [ranking.relevance_scales]\n\"jina-reranker-v1-turbo-en:int8\" = 1.0\n\
         [reconcile.embedding_floors]\n\"bge-small-en-v1.5:int8\" = 0.82\n",
    );
    let output = run(serve_in(&dir)
        .env("ASPHODEL_MODELS", "fake")
        .arg("--config")
        .arg(&tuning)
        .args(["--listen", "127.0.0.1:0"]));
    assert!(!output.status.success());
    let stderr = stderr(&output);
    assert!(
        stderr.contains(&format!("reconcile.embedding_floors.\"{FAKE_EMBEDDER}\"")),
        "{stderr}"
    );
    assert!(
        stderr.contains(&format!("injection.reranker_floors.\"{FAKE_RERANKER}\"")),
        "{stderr}"
    );
    assert!(
        stderr.contains(&format!("ranking.relevance_scales.\"{FAKE_RERANKER}\"")),
        "{stderr}"
    );
}
