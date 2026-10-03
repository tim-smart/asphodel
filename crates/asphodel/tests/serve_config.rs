//! `asphodel serve`'s deployment flags and tuning file, run as a process.
//!
//! Each deployment flag has an `ASPHODEL_*` environment variable,
//! secrets come from the environment only, and an unknown key or
//! out-of-range value in the tuning file stops the daemon starting.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Duration;

const TOKEN: &str = "tok-7f3a9c-secret";
const LLM_KEY: &str = "sk-live-41b2e8-secret";

/// A floor for each fake model. The daemon loads its models before it is
/// ready and refuses a model without floors, so every daemon
/// here that is meant to start runs on the fakes with these.
const FLOORS_FOR_FAKES: &str = "[injection.reranker_floors]\n\"fake-reranker:v1\" = 0.0\n\
                                [ranking.relevance_scales]\n\"fake-reranker:v1\" = 1.0\n\
                                [reconcile.embedding_floors]\n\"fake-embedder:v1\" = 0.5\n";

/// `asphodel serve` with a clean environment, so the caller's `ASPHODEL_*`
/// variables can't leak in. It runs on the fake models, because the real
/// ones aren't on a CI machine.
fn serve() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_asphodel"));
    command
        .env_clear()
        .env("ASPHODEL_MODELS", "fake")
        .arg("serve");
    command
}

/// `asphodel serve` with its own data dir under `dir`. Every daemon that
/// is meant to start gets one, so `--data-dir` can become required without
/// touching these tests. Precedence tests that set `ASPHODEL_DATA_DIR`
/// use `serve()` instead, because the flag would win over the variable.
fn serve_in(dir: &TestDir) -> Command {
    let mut command = serve();
    command.arg("--data-dir").arg(dir.0.join("data"));
    command
}

/// [`serve_in`] with a tuning file holding only the floors for the fakes,
/// for a daemon whose test doesn't need a tuning file of its own.
fn serve_with_floors(dir: &TestDir) -> Command {
    let mut command = serve_in(dir);
    command
        .arg("--config")
        .arg(dir.with_floors("floors.toml", ""));
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
            "asphodel-serve-config-{}-{}",
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

    /// A tuning file holding `text` and the floors for the fakes.
    fn with_floors(&self, name: &str, text: &str) -> PathBuf {
        self.file(name, &format!("{FLOORS_FOR_FAKES}{text}"))
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
                panic!("the daemon never became ready:\n{log}");
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

#[test]
fn secrets_have_no_flag() {
    let output = run(serve().arg("--help"));
    let help = String::from_utf8_lossy(&output.stdout).to_lowercase();
    assert!(!help.contains("--token"), "{help}");
    assert!(!help.contains("api-key"), "{help}");
    assert!(!help.contains("--llm-key"), "{help}");

    for args in [
        ["--token", TOKEN],
        ["--llm-api-key", LLM_KEY],
        ["--api-key", LLM_KEY],
    ] {
        let output = run(serve().args(args));
        assert!(!output.status.success(), "{args:?} was accepted");
    }
}

#[test]
fn an_unknown_tuning_key_stops_startup() {
    let dir = TestDir::new();
    let path = dir.file("tuning.toml", "[clock]\nquiet_rat = 0.2\n");
    let output = run(serve_in(&dir).arg("--config").arg(&path));
    assert!(!output.status.success());
    assert!(stderr(&output).contains("quiet_rat"), "{}", stderr(&output));
}

#[test]
fn an_out_of_range_tuning_value_stops_startup() {
    let dir = TestDir::new();
    let path = dir.file("tuning.toml", "[purge]\ndelta = -1.0\n");
    let output = run(serve_in(&dir).arg("--config").arg(&path));
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("purge.delta"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn a_missing_tuning_file_stops_startup() {
    let dir = TestDir::new();
    let output = run(serve_in(&dir)
        .arg("--config")
        .arg(dir.0.join("missing.toml")));
    assert!(!output.status.success());
}

#[test]
fn the_config_flag_wins_over_its_variable() {
    let dir = TestDir::new();
    let bad = dir.file("bad.toml", "[injection]\ncap = 0\n");
    let good = dir.with_floors("good.toml", "[clock]\nquiet_rate = 0.3\n");
    let daemon = start(
        serve_in(&dir)
            .env("ASPHODEL_CONFIG", &bad)
            .arg("--config")
            .arg(&good),
    );
    let config = resolved_config(&daemon.log);
    assert_eq!(config["tuning"]["clock"]["quiet_rate"], 0.3);
}

#[test]
fn deployment_values_come_from_the_environment() {
    let dir = TestDir::new();
    let tuning = dir.with_floors("tuning.toml", "[purge]\nsource_horizon_days = 60\n");
    let data = dir.0.join("data");
    let models = dir.0.join("models");
    let daemon = start(
        serve()
            .env("ASPHODEL_CONFIG", &tuning)
            .env("ASPHODEL_DATA_DIR", &data)
            .env("ASPHODEL_MODEL_DIR", &models)
            .env("ASPHODEL_ALLOW_NETWORK_FS", "true"),
    );
    let config = resolved_config(&daemon.log);
    let deployment = &config["deployment"];
    assert_eq!(deployment["data_dir"], data.to_str().unwrap());
    assert_eq!(deployment["model_dir"], models.to_str().unwrap());
    assert_eq!(deployment["allow_network_fs"], true);
    assert_eq!(config["tuning"]["purge"]["source_horizon_days"], 60);
}

#[test]
fn listen_comes_from_its_variable_and_the_flag_wins() {
    // An unparseable address in the variable must lose to the flag.
    let dir = TestDir::new();
    let daemon = start(serve_with_floors(&dir).env("ASPHODEL_LISTEN", "not-an-address"));
    let config = resolved_config(&daemon.log);
    assert_eq!(config["deployment"]["listen"], "127.0.0.1:0");

    let output = run(serve_with_floors(&dir).env("ASPHODEL_LISTEN", "not-an-address"));
    assert!(!output.status.success(), "ASPHODEL_LISTEN was ignored");
}

#[test]
fn secrets_are_read_from_the_environment_and_never_logged() {
    let dir = TestDir::new();
    let daemon = start(
        serve_with_floors(&dir)
            .env("ASPHODEL_TOKEN", TOKEN)
            .env("ASPHODEL_LLM_API_KEY", LLM_KEY),
    );
    assert!(!daemon.log.contains(TOKEN), "{}", daemon.log);
    assert!(!daemon.log.contains(LLM_KEY), "{}", daemon.log);
    assert!(!daemon.log.contains("7f3a9c"));
    assert!(!daemon.log.contains("41b2e8"));

    let config = resolved_config(&daemon.log);
    assert_eq!(config["deployment"]["token"], "[redacted]", "{config}");
    assert_eq!(
        config["deployment"]["llm_api_key"], "[redacted]",
        "{config}"
    );
}

#[test]
fn off_loopback_needs_a_token_from_the_environment() {
    let dir = TestDir::new();
    for env in [None, Some("")] {
        let mut command = serve_with_floors(&dir);
        command.args(["--listen", "0.0.0.0:0"]);
        if let Some(value) = env {
            command.env("ASPHODEL_TOKEN", value);
        }
        let output = run(&mut command);
        assert!(
            !output.status.success(),
            "started without a token ({env:?})"
        );
        assert!(
            stderr(&output).contains("ASPHODEL_TOKEN"),
            "{}",
            stderr(&output)
        );
        // Refused before binding, so nothing off this machine ever reached it.
        assert!(
            !stderr(&output).contains("asphodel starting"),
            "the daemon bound before refusing:\n{}",
            stderr(&output)
        );
    }
}

/// Runs a daemon that is expected to refuse and exit on its own. Its
/// stderr goes to `log_path` so a pipe can't fill while waiting, and the
/// guard kills and reaps the child on every exit path, including a timeout
/// or an assertion panic, so a daemon that wrongly starts can't hang the
/// suite. Returns `None` when it was still running at the deadline.
fn run_bounded(command: &mut Command, log_path: &Path) -> (Option<ExitStatus>, String) {
    let child = command
        .args(["--listen", "127.0.0.1:0"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(std::fs::File::create(log_path).unwrap())
        .spawn()
        .unwrap();
    let mut daemon = Daemon {
        child,
        log: String::new(),
    };
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = daemon.child.try_wait().unwrap() {
            break Some(status);
        }
        if std::time::Instant::now() >= deadline {
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    drop(daemon);
    (status, std::fs::read_to_string(log_path).unwrap())
}

#[test]
fn a_malformed_llm_endpoint_stops_startup() {
    let dir = TestDir::new();
    let path = dir.with_floors("tuning.toml", "[llm]\nendpoint = \"http://\"\n");
    let (status, log) = run_bounded(
        serve_in(&dir).arg("--config").arg(&path),
        &dir.0.join("stderr.log"),
    );
    let status = status.unwrap_or_else(|| {
        panic!("invalid endpoint did not stop startup within 5 seconds:\n{log}")
    });
    assert!(
        !status.success(),
        "invalid endpoint exited successfully:\n{log}"
    );
    assert!(log.contains("llm.endpoint"), "{log}");
}

#[test]
fn serve_needs_a_data_dir_from_the_flag_or_the_environment() {
    // The store lives under `--data-dir`. A daemon with
    // neither the flag nor `ASPHODEL_DATA_DIR` has nowhere to persist, so it
    // must refuse to start rather than answer ready with no store.
    let dir = TestDir::new();
    let (status, log) = run_bounded(&mut serve(), &dir.0.join("stderr.log"));
    let status = status
        .unwrap_or_else(|| panic!("a daemon with no data dir kept running for 5 seconds:\n{log}"));
    assert!(
        !status.success(),
        "a daemon with no data dir started:\n{log}"
    );
    assert!(
        log.contains("--data-dir") || log.contains("ASPHODEL_DATA_DIR"),
        "the refusal doesn't name the missing setting:\n{log}"
    );
    // Refused before binding, let alone becoming ready.
    assert!(
        !log.contains("asphodel starting"),
        "the daemon bound before refusing:\n{log}"
    );
    assert!(
        !log.contains("asphodel listening"),
        "the daemon became ready before refusing:\n{log}"
    );
}
