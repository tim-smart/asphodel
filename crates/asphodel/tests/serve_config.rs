//! `asphodel serve`'s deployment flags and tuning file, run as a process.
//!
//! Each deployment flag has an `ASPHODEL_*` environment variable,
//! secrets come from the environment only, and an unknown key or
//! out-of-range value in the tuning file stops the daemon starting. The
//! tests read what the daemon resolved from `GET /v1/config`.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::Value;

const TOKEN: &str = "tok-7f3a9c-secret";
const LLM_KEY: &str = "sk-live-41b2e8-secret";

/// A floor for each fake model. The daemon refuses a model without floors,
/// so every daemon here that is meant to start runs on the fakes with these.
const FLOORS_FOR_FAKES: &str = "[injection.reranker_floors]\n\"fake-reranker:v1\" = 0.0\n\
                                [ranking.relevance_scales]\n\"fake-reranker:v1\" = 1.0\n\
                                [reconcile.embedding_floors]\n\"fake-embedder:v1\" = 0.5\n";

/// `asphodel serve` on the fake models (the real ones aren't on a CI machine)
/// with a clean environment, so the caller's `ASPHODEL_*` can't leak in.
fn serve() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_asphodel"));
    command
        .env_clear()
        .env("ASPHODEL_MODELS", "fake")
        .arg("serve");
    command
}

/// `asphodel serve` with its own data dir under `dir`. Tests that set
/// `ASPHODEL_DATA_DIR` use `serve()`, since the flag would win.
fn serve_in(dir: &TestDir) -> Command {
    let mut command = serve();
    command.arg("--data-dir").arg(dir.0.join("data"));
    command
}

/// [`serve_in`] with a tuning file holding only the floors for the fakes.
fn serve_with_floors(dir: &TestDir) -> Command {
    let mut command = serve_in(dir);
    command
        .arg("--config")
        .arg(dir.with_floors("floors.toml", ""));
    command
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

/// A loopback address on a port that is free now.
fn loopback() -> String {
    let free = TcpListener::bind("127.0.0.1:0").unwrap();
    free.local_addr().unwrap().to_string()
}

/// A running daemon, killed on drop.
struct Daemon {
    child: Child,
    /// The loopback address it listens on.
    addr: String,
    /// Reads its stderr to the end.
    log: Option<JoinHandle<String>>,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Daemon {
    /// `GET <path>` with `token` if given: the status and the body.
    fn get(&self, path: &str, token: Option<&str>) -> std::io::Result<(u16, String)> {
        let mut stream = TcpStream::connect(&self.addr)?;
        let auth = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
        write!(stream, "GET {path} HTTP/1.0\r\n{auth}\r\n")?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        let (head, body) = response.split_once("\r\n\r\n").unwrap_or((&response, ""));
        let status = head.split_whitespace().nth(1).and_then(|s| s.parse().ok());
        Ok((status.unwrap_or(0), body.to_string()))
    }

    /// `GET /v1/config`, which must answer 200, with `token` if given.
    fn config(&self, token: Option<&str>) -> Value {
        let (status, body) = self.get("/v1/config", token).unwrap();
        assert_eq!(status, 200, "{body}");
        serde_json::from_str(&body).unwrap()
    }

    /// Kills the daemon and returns its whole log.
    fn log(&mut self) -> String {
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.log.take().unwrap().join().unwrap()
    }
}

/// Starts the daemon listening on `addr` and waits until `/v1/health`
/// answers 200. Panics with its log if it exits or stalls first.
fn start(command: &mut Command, addr: &str) -> Daemon {
    let mut child = command
        .env("ASPHODEL_LOG", "trace")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let log = std::thread::spawn(move || {
        let mut log = String::new();
        let _ = stderr.read_to_string(&mut log);
        log
    });
    // Dropped, so killed, if it panics here.
    let mut daemon = Daemon {
        child,
        addr: addr.to_string(),
        log: Some(log),
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while !matches!(daemon.get("/v1/health", None), Ok((200, _))) {
        if Instant::now() >= deadline {
            panic!("the daemon never became ready:\n{}", daemon.log());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    daemon
}

/// Runs a daemon that is expected to refuse and exit on its own, naming
/// `name` (an empty one checks nothing). Stderr goes to a file so a pipe
/// can't fill while waiting, and a daemon still running at the deadline is
/// killed so one that wrongly starts can't hang the suite.
fn refused(dir: &TestDir, command: &mut Command, name: &str) {
    let log_path = dir.0.join("stderr.log");
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(std::fs::File::create(&log_path).unwrap())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        match child.try_wait().unwrap() {
            None if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            status => break status,
        }
    };
    let _ = child.kill();
    let _ = child.wait();
    let log = std::fs::read_to_string(log_path).unwrap();
    assert!(
        status.is_some_and(|status| !status.success()),
        "it started ({status:?}):\n{log}"
    );
    assert!(log.contains(name), "{name}:\n{log}");
}

#[test]
fn bad_settings_and_secret_flags_stop_startup() {
    let dir = TestDir::new();
    let loopback = ["--listen", "127.0.0.1:0"];
    let with_config = |name, text| {
        let mut command = serve_in(&dir);
        command.arg("--config").arg(dir.with_floors(name, text));
        command
    };
    // Secrets have no flag, so they never show in a process list.
    for flag in ["--token", "--llm-api-key", "--api-key"] {
        refused(&dir, serve().args([flag, TOKEN]), "");
    }
    let key = "[clock]\nquiet_rat = 0.2\n";
    refused(&dir, &mut with_config("key.toml", key), "quiet_rat");
    let range = "[purge]\ndelta = -1.0\n";
    refused(&dir, &mut with_config("range.toml", range), "purge.delta");
    let missing = dir.0.join("missing.toml");
    refused(&dir, serve_in(&dir).arg("--config").arg(missing), "");
    let endpoint = "[llm]\nendpoint = \"http://\"\n";
    let mut command = with_config("endpoint.toml", endpoint);
    refused(&dir, command.args(loopback), "llm.endpoint");
    // The store lives under the data dir: without one there's nowhere to
    // persist.
    refused(&dir, serve().args(loopback), "--data-dir");
    // The variable is read: an unparseable address in it is refused.
    let mut command = serve_with_floors(&dir);
    refused(&dir, command.env("ASPHODEL_LISTEN", "not-an-address"), "");
    // Off loopback needs a token, from the environment.
    let off_loopback = || {
        let mut command = serve_with_floors(&dir);
        command.args(["--listen", "0.0.0.0:0"]);
        command
    };
    refused(&dir, &mut off_loopback(), "ASPHODEL_TOKEN");
    let mut empty_token = off_loopback();
    empty_token.env("ASPHODEL_TOKEN", "");
    refused(&dir, &mut empty_token, "ASPHODEL_TOKEN");
}

#[test]
fn deployment_values_come_from_the_environment_and_flags_win() {
    let dir = TestDir::new();
    let tuning = dir.with_floors("tuning.toml", "[purge]\nsource_horizon_days = 60\n");
    let data = dir.0.join("data");
    let models = dir.0.join("models");
    let addr = loopback();
    let daemon = start(
        serve()
            .env("ASPHODEL_CONFIG", &tuning)
            .env("ASPHODEL_DATA_DIR", &data)
            .env("ASPHODEL_MODEL_DIR", &models)
            .env("ASPHODEL_ALLOW_NETWORK_FS", "true")
            .env("ASPHODEL_LISTEN", &addr),
        &addr,
    );
    let config = daemon.config(None);
    let deployment = &config["deployment"];
    assert_eq!(deployment["data_dir"], data.to_str().unwrap());
    assert_eq!(deployment["model_dir"], models.to_str().unwrap());
    assert_eq!(deployment["allow_network_fs"], true);
    assert_eq!(config["tuning"]["purge"]["source_horizon_days"], 60);
    drop(daemon);

    let bad = dir.file("bad.toml", "[injection]\ncap = 0\n");
    let good = dir.with_floors("good.toml", "[clock]\nquiet_rate = 0.3\n");
    let addr = loopback();
    let daemon = start(
        serve_in(&dir)
            .env("ASPHODEL_CONFIG", &bad)
            .env("ASPHODEL_LISTEN", "not-an-address")
            .arg("--config")
            .arg(&good)
            .args(["--listen", &addr]),
        &addr,
    );
    let config = daemon.config(None);
    assert_eq!(config["tuning"]["clock"]["quiet_rate"], 0.3);
    assert_eq!(config["deployment"]["listen"], addr);
}

#[test]
fn secrets_are_read_from_the_environment_and_never_logged() {
    let dir = TestDir::new();
    let addr = loopback();
    let mut daemon = start(
        serve_with_floors(&dir)
            .args(["--listen", &addr])
            .env("ASPHODEL_TOKEN", TOKEN)
            .env("ASPHODEL_LLM_API_KEY", LLM_KEY),
        &addr,
    );
    let config = daemon.config(Some(TOKEN));
    for secret in ["token", "llm_api_key"] {
        assert_eq!(config["deployment"][secret], "[redacted]", "{config}");
    }
    let log = daemon.log();
    for secret in ["7f3a9c", "41b2e8"] {
        assert!(!config.to_string().contains(secret), "{config}");
        assert!(!log.contains(secret), "{log}");
    }
}
