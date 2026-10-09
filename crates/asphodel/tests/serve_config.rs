//! `asphodel serve`'s deployment flags and tuning file, run as a process.
//!
//! Each deployment flag has an `ASPHODEL_*` environment variable,
//! secrets come from the environment or the data dir, never a flag, and an
//! unknown key or out-of-range value in the tuning file stops the daemon
//! starting. Without a tuning file, first-run setup writes one into the data
//! dir. The tests read what the daemon resolved from `GET /v1/config`.

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

/// A port free now on every interface, as `--listen` takes it, and the
/// loopback address that reaches it.
fn every_interface() -> (String, String) {
    let addr = loopback();
    let port = addr.rsplit_once(':').unwrap().1;
    (format!("0.0.0.0:{port}"), addr)
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
        self.send("GET", path, token, None)
    }

    /// `<method> <path>` with `token` if given and a JSON body: the status
    /// and the body.
    fn send(
        &self,
        method: &str,
        path: &str,
        token: Option<&str>,
        body: Option<&Value>,
    ) -> std::io::Result<(u16, String)> {
        let mut stream = TcpStream::connect(&self.addr)?;
        let auth = token.map_or(String::new(), |t| format!("Authorization: Bearer {t}\r\n"));
        let body = body.map_or(String::new(), Value::to_string);
        write!(
            stream,
            "{method} {path} HTTP/1.0\r\n{auth}Content-Type: application/json\r\n\
             Content-Length: {}\r\n\r\n{body}",
            body.len()
        )?;
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

    /// `GET /v1/health`'s body, which must answer 200.
    fn health(&self) -> Value {
        let (status, body) = self.get("/v1/health", None).unwrap();
        assert_eq!(status, 200, "{body}");
        serde_json::from_str(&body).unwrap()
    }

    /// Waits until `/v1/health` says the daemon is ready, not just set up.
    /// It answers 503 while the store opens after setup.
    fn wait_ready(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(10);
        let ready = |daemon: &Self| {
            daemon
                .get("/v1/health", None)
                .is_ok_and(|(status, body)| status == 200 && body.contains("\"ready\":true"))
        };
        while !ready(self) {
            if Instant::now() >= deadline {
                panic!("the daemon never became ready:\n{}", self.log());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// `POST /v1/setup` with `code` as the bearer token.
    fn setup(&self, code: Option<&str>, body: &Value) -> (u16, Value) {
        let (status, body) = self.send("POST", "/v1/setup", code, Some(body)).unwrap();
        (status, serde_json::from_str(&body).unwrap_or(Value::Null))
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

/// The setup code the daemon wrote into `data`.
fn setup_code(data: &std::path::Path) -> String {
    std::fs::read_to_string(data.join("setup-code"))
        .unwrap()
        .trim()
        .to_string()
}

#[test]
fn first_run_setup_writes_the_config_into_the_data_dir_and_survives_restarts() {
    let dir = TestDir::new();
    let data = dir.0.join("data");
    // Off loopback with no token anywhere: setup still starts, guarded by
    // its code, and makes the token.
    let serve_off_loopback = || {
        let (listen, addr) = every_interface();
        let mut command = serve_in(&dir);
        command.args(["--listen", &listen]);
        (command, addr)
    };
    let (mut command, addr) = serve_off_loopback();
    let mut daemon = start(&mut command, &addr);
    let health = daemon.health();
    assert_eq!(health["setup"], true, "{health}");
    assert_eq!(health["ready"], false, "{health}");
    assert_eq!(daemon.get("/v1/config", None).unwrap().0, 503);
    let (status, body) = daemon.get("/v1/setup", None).unwrap();
    assert_eq!(status, 200, "{body}");
    let state: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(state["needed"], true, "{state}");
    assert_eq!(state["makes_token"], true, "{state}");

    let llm = serde_json::json!({"llm": {
        "endpoint": "https://llm.example.com/v1",
        "model": "model-under-test",
        "api_key": LLM_KEY,
    }});
    // Without the code, or with a wrong one, nothing is written.
    assert_eq!(daemon.setup(None, &llm).0, 401);
    assert_eq!(daemon.setup(Some("not-the-code"), &llm).0, 401);
    // A bad value is refused, and setup still waits.
    let code = setup_code(&data);
    let bad = serde_json::json!({"llm": {"endpoint": "ftp://nowhere", "model": "m"}});
    assert_eq!(daemon.setup(Some(&code), &bad).0, 400);
    assert!(!data.join("asphodel.toml").exists());

    let (status, done) = daemon.setup(Some(&code), &llm);
    assert_eq!(status, 200, "{done}");
    let token = done["token"]
        .as_str()
        .expect("setup makes a token")
        .to_string();
    daemon.wait_ready();
    // Setup happens once: the code is gone and a second call is refused.
    assert!(!data.join("setup-code").exists());
    assert_eq!(daemon.setup(Some(&code), &llm).0, 409);
    {
        use std::os::unix::fs::PermissionsExt;
        let secrets = std::fs::metadata(data.join("secrets.toml")).unwrap();
        assert_eq!(
            secrets.permissions().mode() & 0o077,
            0,
            "secrets.toml is private"
        );
    }
    assert_eq!(daemon.get("/v1/config", None).unwrap().0, 401);
    let config = daemon.config(Some(&token));
    assert_eq!(config["tuning"]["llm"]["model"], "model-under-test");
    assert_eq!(config["deployment"]["llm_api_key"], "[redacted]");
    let bank = serde_json::json!({});
    let (status, body) = daemon
        .send("PUT", "/v1/banks/persisted", Some(&token), Some(&bank))
        .unwrap();
    assert_eq!(status, 201, "{body}");
    let log = daemon.log();
    for secret in [LLM_KEY, &token, &code] {
        assert!(!log.contains(secret), "a secret was logged:\n{log}");
    }

    // A restart with nothing but the data dir reads what setup wrote: the
    // tuning, the token and the store.
    let (mut command, addr) = serve_off_loopback();
    let mut daemon = start(&mut command, &addr);
    assert_eq!(daemon.health()["setup"], false);
    daemon.wait_ready();
    assert_eq!(daemon.get("/v1/config", None).unwrap().0, 401);
    let config = daemon.config(Some(&token));
    assert_eq!(config["tuning"]["llm"]["model"], "model-under-test");
    assert_eq!(config["llm"]["logged_in"], true, "{config}");
    let (status, body) = daemon.get("/v1/banks", Some(&token)).unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(body.contains("persisted"), "{body}");
    drop(daemon);

    // The environment still wins over the data dir's token.
    let (mut command, addr) = serve_off_loopback();
    let daemon = start(command.env("ASPHODEL_TOKEN", TOKEN), &addr);
    assert_eq!(daemon.get("/v1/config", Some(&token)).unwrap().0, 401);
    daemon.config(Some(TOKEN));
    drop(daemon);

    // And `--config` still wins over the data dir's tuning file.
    let tuning = dir.with_floors("explicit.toml", "[clock]\nquiet_rate = 0.3\n");
    let (mut command, addr) = serve_off_loopback();
    let daemon = start(command.arg("--config").arg(&tuning), &addr);
    let config = daemon.config(Some(&token));
    assert_eq!(config["tuning"]["clock"]["quiet_rate"], 0.3);
    assert_eq!(config["tuning"]["llm"]["model"], Value::Null);
    assert_eq!(config["deployment"]["config"], tuning.to_str().unwrap());
}

#[test]
fn a_pending_setup_keeps_its_code_across_restarts_owns_its_data_dir_and_takes_the_environment_token()
 {
    let dir = TestDir::new();
    let data = dir.0.join("data");
    let addr = loopback();
    let mut daemon = start(serve_in(&dir).args(["--listen", &addr]), &addr);
    assert_eq!(daemon.health()["setup"], true);
    let code = setup_code(&data);
    daemon.log();

    // A restart before setup is done waits again, on the same code.
    let addr = loopback();
    let daemon = start(
        serve_in(&dir)
            .args(["--listen", &addr])
            .env("ASPHODEL_TOKEN", TOKEN),
        &addr,
    );
    assert_eq!(daemon.health()["setup"], true);
    assert_eq!(setup_code(&data), code);
    // Waiting for setup, the daemon owns the data dir: a second one on it
    // refuses to start, so it can't take the code or write the files.
    let (listen, _) = every_interface();
    refused(
        &dir,
        serve_in(&dir).args(["--listen", &listen]),
        "locked by another asphodel process",
    );
    assert_eq!(setup_code(&data), code);
    assert!(!data.join("asphodel.toml").exists());
    assert!(!data.join("secrets.toml").exists());
    // `ASPHODEL_TOKEN` authorises setup too, and setup leaves the LLM for
    // later and makes no token when one is already configured.
    let (status, done) = daemon.setup(Some(TOKEN), &serde_json::json!({}));
    assert_eq!(status, 200, "{done}");
    assert_eq!(done["token"], Value::Null);
    let mut daemon = daemon;
    daemon.wait_ready();
    let config = daemon.config(Some(TOKEN));
    assert_eq!(config["llm"], Value::Null, "{config}");
    assert!(!data.join("secrets.toml").exists());
}
