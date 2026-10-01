//! `asphodel serve` and the models, run as a process.
//!
//! "Models: local embeddings, reranker and the OpenAI-compatible LLM
//! client" (TIM-105), from "API surface and Hermes transport" (TIM-94,
//! decision 4) and ADR 0009: the daemon loads the models from the model dir
//! before it listens, never downloads, fails fast on a missing file, and
//! refuses to start without a floor for each loaded model. These tests see
//! only what an operator sees: flags, exit codes, stderr and the resolved
//! config line.
//!
//! The daemon can't load the real models on a CI machine, so these tests
//! start it on the fakes with `ASPHODEL_MODELS=fake`, environment only and
//! hidden from `--help`, which `serve` honours with a warning and shows in
//! the resolved config. The process tests in `serve_config.rs`,
//! `serve_store.rs` and `llm_login.rs` set the same variable and a floor for
//! each fake.
//!
//! Two tests stay ignored until `serve` loads the ONNX models when the
//! variable is unset: today it warns and starts without models, so they
//! would hang waiting for a refusal instead of failing.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Duration;

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

    /// A tuning file with a floor for each fake model.
    fn floors_for_fakes(&self) -> PathBuf {
        self.file(
            "tuning.toml",
            &format!(
                "[injection.reranker_floors]\n\"{FAKE_RERANKER}\" = 0.0\n\
                 [reconcile.embedding_floors]\n\"{FAKE_EMBEDDER}\" = 0.5\n"
            ),
        )
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A running daemon, killed on drop.
struct Daemon {
    child: Child,
    log: String,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Starts the daemon on an ephemeral loopback port and collects its log up
/// to the "listening" line. Panics if it exits or stalls first.
fn start(command: &mut Command) -> Daemon {
    let mut child = command
        .args(["--listen", "127.0.0.1:0"])
        .env("ASPHODEL_LOG", "trace")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stderr = child.stderr.take().unwrap();
    let (lines, received) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            if lines.send(line).is_err() {
                break;
            }
        }
    });

    let mut log = String::new();
    loop {
        match received.recv_timeout(Duration::from_secs(10)) {
            Ok(line) => {
                log.push_str(&line);
                log.push('\n');
                if line.contains("asphodel listening") {
                    return Daemon { child, log };
                }
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("the daemon never started listening:\n{log}");
            }
        }
    }
}

/// The JSON of the "resolved config" log line.
fn resolved_config(log: &str) -> serde_json::Value {
    let line = log
        .lines()
        .find(|line| line.contains("resolved config"))
        .unwrap_or_else(|| panic!("no resolved config line in:\n{log}"));
    let start = line.find("config=").expect("a config field") + "config=".len();
    let text = &line[start..];
    let mut stream = serde_json::Deserializer::from_str(text).into_iter::<serde_json::Value>();
    stream.next().unwrap().unwrap()
}

fn help_line(help: &str, flag: &str) -> String {
    help.lines()
        .find(|line| line.trim_start().starts_with(flag))
        .unwrap_or_else(|| panic!("no {flag} in:\n{help}"))
        .to_string()
}

#[test]
fn models_fetch_takes_the_model_dir_from_its_variable() {
    // TIM-94, decision 4: `asphodel models fetch` fills the model dir, and
    // `ASPHODEL_MODEL_DIR` overrides where that is.
    let output = run(asphodel().args(["models", "fetch", "--help"]));
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    let line = help_line(&help, "--model-dir");
    assert!(line.contains("ASPHODEL_MODEL_DIR"), "{line}");
}

#[test]
fn onnx_threads_is_a_deployment_flag_with_a_variable() {
    // TIM-98 lists ONNX threads under deployment, and TIM-96 pins it in
    // replay.
    let output = run(asphodel().args(["serve", "--help"]));
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    let line = help_line(&help, "--onnx-threads");
    assert!(line.contains("ASPHODEL_ONNX_THREADS"), "{line}");

    // The fake-models switch is for tests and never advertised.
    assert!(!help.contains("ASPHODEL_MODELS"), "{help}");
    assert!(!help.to_lowercase().contains("fake"), "{help}");
}

#[test]
#[ignore = "needs serve to load the models from the model dir when ASPHODEL_MODELS is unset"]
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
    // And the daemon never listened.
    assert!(!stderr.contains("asphodel listening"), "{stderr}");
}

#[test]
#[ignore = "needs serve to load the models from the model dir when ASPHODEL_MODELS is unset"]
fn a_model_dir_that_does_not_exist_stops_startup_without_creating_it() {
    let dir = TestDir::new();
    let models = dir.0.join("missing-models");
    let output = run(serve_in(&dir)
        .arg("--model-dir")
        .arg(&models)
        .args(["--listen", "127.0.0.1:0"]));
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("asphodel models fetch"),
        "{}",
        stderr(&output)
    );
    assert!(!models.exists(), "the daemon created the model dir");
}

#[test]
fn fake_models_still_need_floors() {
    // ADR 0009: a missing floor for a configured model stops the daemon,
    // whichever models are configured.
    let dir = TestDir::new();
    let output = run(serve_in(&dir)
        .env("ASPHODEL_MODELS", "fake")
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
}

#[test]
fn fake_models_with_floors_start_and_show_in_the_resolved_config() {
    let dir = TestDir::new();
    let tuning = dir.floors_for_fakes();
    let daemon = start(
        serve_in(&dir)
            .env("ASPHODEL_MODELS", "fake")
            .arg("--config")
            .arg(&tuning),
    );
    let config = resolved_config(&daemon.log);
    assert_eq!(config["models"]["embedding"], FAKE_EMBEDDER);
    assert_eq!(config["models"]["reranker"], FAKE_RERANKER);
    // An operator reading the log sees that this daemon isn't on real models.
    assert!(daemon.log.contains("fake"), "{}", daemon.log);
}

#[test]
fn a_floor_for_the_real_models_does_not_cover_the_fakes() {
    // Floors are keyed by the exact model string. A production tuning file
    // doesn't make a fake daemon start, so the switch can't hide a missing
    // floor.
    let dir = TestDir::new();
    let tuning = dir.file(
        "tuning.toml",
        "[injection.reranker_floors]\n\"jina-reranker-v1-turbo-en:int8\" = -1.5\n\
         [reconcile.embedding_floors]\n\"bge-small-en-v1.5:int8\" = 0.82\n",
    );
    let output = run(serve_in(&dir)
        .env("ASPHODEL_MODELS", "fake")
        .arg("--config")
        .arg(&tuning)
        .args(["--listen", "127.0.0.1:0"]));
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains(FAKE_EMBEDDER),
        "{}",
        stderr(&output)
    );
}

#[test]
fn the_llm_key_never_reaches_the_resolved_config_or_the_log() {
    let dir = TestDir::new();
    let tuning = dir.floors_for_fakes();
    let daemon = start(
        serve_in(&dir)
            .env("ASPHODEL_MODELS", "fake")
            .env("ASPHODEL_LLM_API_KEY", "sk-live-41b2e8-secret")
            .arg("--config")
            .arg(&tuning),
    );
    let config = resolved_config(&daemon.log);
    assert_eq!(config["deployment"]["llm_api_key"], "[redacted]");
    assert!(!daemon.log.contains("41b2e8"), "{}", daemon.log);
}
