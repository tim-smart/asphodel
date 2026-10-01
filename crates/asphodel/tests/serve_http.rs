//! The daemon over HTTP, and the CLI as its client, run as processes.
//!
//! "HTTP API and the CLI as its client" (TIM-110), from "API surface and
//! Hermes transport" (TIM-94, decisions 2, 3 and 10) and ADR 0006: the
//! routes under `/v1`, the bearer token off loopback, `/v1/health` answering
//! 503 until the store and models are ready, SIGTERM finishing the chunk in
//! flight and checkpointing the WAL, and every CLI subcommand reaching the
//! daemon over `--url`.
//!
//! The daemon runs on the fake models (`ASPHODEL_MODELS=fake`) and a
//! scripted fake LLM (`ASPHODEL_LLM_SCRIPT`), both environment only. The
//! script plays one step per LLM call, in order, across the whole daemon,
//! so each test scripts exactly the calls its chunks make.
//! `ASPHODEL_STARTUP_GATE` holds startup after the bind so the 503 can be
//! seen, and `--listen 127.0.0.1:0` logs the port the system picked. The
//! HTTP client here is a few lines over std, so the tests need no new
//! dependencies.

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// How long a daemon gets to bind, to become ready, or to stop.
const STARTUP: Duration = Duration::from_secs(10);

/// How long a condition polled over HTTP gets to hold.
const SETTLE: Duration = Duration::from_secs(10);

const TOKEN: &str = "test-bearer-token-5a1d";

/// The document every extraction test ingests, and the one claim its
/// scripted call 1 reply makes about it.
const NOTES: &str = "# Notes\n\nI live in Auckland and I drink tea every morning.\n";
const SENTENCE: &str = "Tim lives in Auckland.";

/// A call 1 reply with one fact quoted from [`NOTES`].
fn auckland_reply() -> Value {
    json!({
        "claims": [{
            "content": SENTENCE,
            "kind": "fact",
            "quote": "I live in Auckland",
            "significance": "notable",
            "remember_this": false,
            "changes_something": false,
            "valid_from": null,
            "valid_until": null,
            "window_confidence": "high",
            "until_event": null,
            "due_at": null,
            "volatility": null,
            "recurrence_text": null,
            "recurrence_rrule": null,
            "recurrence_start": null,
            "entities": []
        }],
        "used_injected_ids": []
    })
}

/// A call 1 reply with nothing in it.
fn empty_reply() -> Value {
    json!({"claims": [], "used_injected_ids": []})
}

/// A temporary directory removed even when an assertion unwinds. Its path is
/// short, so a Unix socket inside it fits in `sun_path`.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-http-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn data_dir(&self) -> PathBuf {
        let path = self.path("data");
        fs::create_dir_all(&path).unwrap();
        path
    }

    /// A tuning file with a floor for each fake model.
    fn floors_for_fakes(&self) -> PathBuf {
        let path = self.path("tuning.toml");
        fs::write(
            &path,
            "[injection.reranker_floors]\n\"fake-reranker:v1\" = 0.0\n\
             [reconcile.embedding_floors]\n\"fake-embedder:v1\" = 0.5\n",
        )
        .unwrap();
        path
    }

    /// Writes `steps` as the LLM script and returns its path.
    fn script(&self, name: &str, steps: &[Value]) -> PathBuf {
        let path = self.path(name);
        fs::write(&path, serde_json::to_string(steps).unwrap()).unwrap();
        path
    }

    fn file(&self, name: &str, text: &str) -> PathBuf {
        let path = self.path(name);
        fs::write(&path, text).unwrap();
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Where a daemon is.
#[derive(Debug, Clone)]
enum Addr {
    Tcp(String),
    Unix(PathBuf),
}

impl Addr {
    /// The `--url` the CLI takes for it.
    fn url(&self) -> String {
        match self {
            Addr::Tcp(addr) => format!("http://{addr}"),
            Addr::Unix(path) => format!("unix:{}", path.display()),
        }
    }
}

/// One HTTP reply.
#[derive(Debug)]
struct Reply {
    status: u16,
    /// The header block, lowercased.
    headers: String,
    body: String,
}

impl Reply {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body)
            .unwrap_or_else(|error| panic!("not JSON ({error}): {} {}", self.status, self.body))
    }
}

/// One request on its own connection, with `Connection: close`, so the
/// reply ends where the stream does.
fn request(
    addr: &Addr,
    method: &str,
    path: &str,
    token: Option<&str>,
    body: Option<&Value>,
) -> std::io::Result<Reply> {
    let body = body.map(|body| serde_json::to_string(body).unwrap());
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: asphodel\r\nConnection: close\r\n");
    if let Some(token) = token {
        head.push_str(&format!("Authorization: Bearer {token}\r\n"));
    }
    if let Some(body) = &body {
        head.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            body.len()
        ));
    }
    head.push_str("\r\n");
    let mut bytes = head.into_bytes();
    bytes.extend_from_slice(body.unwrap_or_default().as_bytes());

    let mut response = Vec::new();
    match addr {
        Addr::Tcp(addr) => {
            let mut stream = TcpStream::connect(addr)?;
            stream.set_read_timeout(Some(SETTLE))?;
            stream.write_all(&bytes)?;
            stream.read_to_end(&mut response)?;
        }
        Addr::Unix(path) => {
            let mut stream = UnixStream::connect(path)?;
            stream.set_read_timeout(Some(SETTLE))?;
            stream.write_all(&bytes)?;
            stream.read_to_end(&mut response)?;
        }
    }
    let response = String::from_utf8(response).expect("a UTF-8 reply");
    let (head, body) = response
        .split_once("\r\n\r\n")
        .unwrap_or_else(|| panic!("no header block in:\n{response}"));
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("no status line in:\n{response}"));
    let headers = head.to_lowercase();
    let body = if headers.contains("transfer-encoding: chunked") {
        dechunk(body)
    } else {
        body.to_string()
    };
    Ok(Reply {
        status,
        headers,
        body,
    })
}

fn dechunk(mut body: &str) -> String {
    let mut out = String::new();
    loop {
        let (size, rest) = body.split_once("\r\n").expect("a chunk size line");
        let size = usize::from_str_radix(size.trim(), 16).expect("a hex chunk size");
        if size == 0 {
            return out;
        }
        out.push_str(&rest[..size]);
        body = &rest[size + 2..];
    }
}

/// How to start a daemon.
struct Serve<'a> {
    dir: &'a TestDir,
    data_dir: PathBuf,
    listen: String,
    script: Option<PathBuf>,
    gate: Option<PathBuf>,
    token: Option<&'a str>,
}

impl<'a> Serve<'a> {
    /// A daemon on a Unix socket in `dir`, on the fakes, with no LLM.
    fn new(dir: &'a TestDir) -> Self {
        Self {
            dir,
            data_dir: dir.data_dir(),
            listen: format!("unix:{}", dir.path("d.sock").display()),
            script: None,
            gate: None,
            token: None,
        }
    }

    fn listen(mut self, listen: &str) -> Self {
        self.listen = listen.to_string();
        self
    }

    fn script(mut self, steps: &[Value]) -> Self {
        self.script = Some(self.dir.script("script.json", steps));
        self
    }

    fn gate(mut self, gate: &Path) -> Self {
        self.gate = Some(gate.to_owned());
        self
    }

    fn token(mut self, token: &'a str) -> Self {
        self.token = Some(token);
        self
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_asphodel"));
        command
            .env_clear()
            .env("ASPHODEL_MODELS", "fake")
            .env("ASPHODEL_LOG", "info")
            .arg("serve")
            .arg("--config")
            .arg(self.dir.floors_for_fakes())
            .arg("--data-dir")
            .arg(&self.data_dir)
            .arg("--listen")
            .arg(&self.listen);
        if let Some(script) = &self.script {
            command.env("ASPHODEL_LLM_SCRIPT", script);
        }
        if let Some(gate) = &self.gate {
            command.env("ASPHODEL_STARTUP_GATE", gate);
        }
        if let Some(token) = self.token {
            command.env("ASPHODEL_TOKEN", token);
        }
        command
    }

    /// Starts the daemon and returns once it has bound, which the
    /// "asphodel starting" line reports with the address.
    fn bind(self) -> Daemon {
        let mut child = self
            .command()
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
        let mut daemon = Daemon {
            child,
            addr: Addr::Unix(PathBuf::new()),
            token: self.token.map(str::to_string),
            data_dir: self.data_dir.clone(),
            log: String::new(),
            lines: received,
        };
        let line = daemon.wait_for_line("asphodel starting");
        daemon.addr = match self.listen.strip_prefix("unix:") {
            Some(path) => Addr::Unix(PathBuf::from(path)),
            None => {
                let bound = line
                    .split_whitespace()
                    .find_map(|field| field.strip_prefix("listen="))
                    .unwrap_or_else(|| panic!("no listen= in: {line}"));
                // A daemon bound to every interface is reached on loopback.
                Addr::Tcp(bound.replace("0.0.0.0", "127.0.0.1"))
            }
        };
        daemon
    }

    /// Starts the daemon and waits until `/v1/health` answers 200.
    fn ready(self) -> Daemon {
        let mut daemon = self.bind();
        daemon.wait_ready();
        daemon
    }
}

/// A running daemon, killed on drop.
struct Daemon {
    child: Child,
    addr: Addr,
    token: Option<String>,
    data_dir: PathBuf,
    /// Its stderr so far.
    log: String,
    lines: mpsc::Receiver<String>,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Daemon {
    /// Reads the log until a line contains `pattern`, and returns it.
    fn wait_for_line(&mut self, pattern: &str) -> String {
        let deadline = Instant::now() + STARTUP;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(line) => {
                    self.log.push_str(&line);
                    self.log.push('\n');
                    if line.contains(pattern) {
                        return line;
                    }
                }
                Err(_) => panic!("no {pattern:?} line in the log:\n{}", self.log),
            }
        }
    }

    /// Appends every log line that has arrived, up to the end of stderr if
    /// the daemon has exited.
    fn drain(&mut self) {
        while let Ok(line) = self.lines.recv_timeout(Duration::from_millis(500)) {
            self.log.push_str(&line);
            self.log.push('\n');
        }
    }

    /// A request with the daemon's token, if it has one.
    fn send(&self, method: &str, path: &str, body: Option<&Value>) -> Reply {
        request(&self.addr, method, path, self.token.as_deref(), body)
            .unwrap_or_else(|error| panic!("{method} {path}: {error}"))
    }

    fn get(&self, path: &str) -> Reply {
        self.send("GET", path, None)
    }

    fn post(&self, path: &str, body: &Value) -> Reply {
        self.send("POST", path, Some(body))
    }

    fn put(&self, path: &str, body: &Value) -> Reply {
        self.send("PUT", path, Some(body))
    }

    /// The JSON body of a reply that must be 2xx.
    fn ok(&self, reply: Reply) -> Value {
        assert!(
            (200..300).contains(&reply.status),
            "{} {}\n{}",
            reply.status,
            reply.body,
            self.log
        );
        reply.json()
    }

    fn wait_ready(&mut self) {
        let deadline = Instant::now() + STARTUP;
        loop {
            let last = match request(&self.addr, "GET", "/v1/health", None, None) {
                Ok(reply) if reply.status == 200 => break,
                Ok(reply) => format!("{} {}", reply.status, reply.body),
                Err(error) => format!("connect: {error}"),
            };
            if Instant::now() >= deadline {
                self.drain();
                panic!("never ready; last health was {last}\n{}", self.log);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        self.wait_for_line("asphodel listening");
    }

    /// Polls `read` until `holds` accepts its value, and returns that value.
    fn wait_until(
        &mut self,
        what: &str,
        read: impl Fn(&Self) -> Value,
        holds: impl Fn(&Value) -> bool,
    ) -> Value {
        let deadline = Instant::now() + SETTLE;
        loop {
            let value = read(self);
            if holds(&value) {
                return value;
            }
            if Instant::now() >= deadline {
                self.drain();
                panic!("{what} never held; last saw {value}\n{}", self.log);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn create_bank(&self, bank: &str) {
        let reply = self.put(
            &format!("/v1/banks/{bank}"),
            &json!({"owner_name": "Tim", "assistant_name": "Ash", "timezone": "Pacific/Auckland"}),
        );
        assert_eq!(reply.status, 201, "{}", reply.body);
    }

    fn ingest_notes(&self, bank: &str, id: &str) -> Value {
        self.ok(self.post(
            &format!("/v1/banks/{bank}/documents"),
            &json!({
                "document_id": id,
                "text": NOTES,
                "reference_date": "2026-09-30",
                "reference_date_exact": true,
                "timezone": null
            }),
        ))
    }

    fn recall(&self, bank: &str, query: &str) -> Value {
        self.ok(self.post(
            &format!("/v1/banks/{bank}/recall"),
            &json!({"query": query}),
        ))
    }

    fn chunks(&self, bank: &str) -> Value {
        self.ok(self.get(&format!("/v1/banks/{bank}/chunks")))
    }

    /// Waits until `bank`'s queue is empty and nothing has failed.
    fn wait_extracted(&mut self, bank: &str) {
        self.wait_until(
            "an empty queue",
            |daemon| daemon.chunks(bank),
            |chunks| chunks["queued"] == json!([]) && chunks["failed"] == json!([]),
        );
    }

    /// The id of the memory recalled for [`SENTENCE`], once extraction has
    /// written it.
    fn wait_for_memory(&mut self, bank: &str) -> String {
        let recall = self.wait_until(
            "a recalled memory",
            |daemon| daemon.recall(bank, "where does Tim live? Auckland"),
            |recall| recall["results"][0]["sentence"] == SENTENCE,
        );
        recall["results"][0]["id"].as_str().unwrap().to_string()
    }

    /// SIGTERM, as a supervisor sends it.
    fn sigterm(&self) {
        let signalled = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status()
            .unwrap();
        assert!(signalled.success(), "kill -TERM failed");
    }

    /// Waits for the daemon to exit, then the status and the whole log.
    fn wait_exit(&mut self) -> ExitStatus {
        let deadline = Instant::now() + STARTUP;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                self.drain();
                return status;
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                self.drain();
                panic!("the daemon did not stop in time:\n{}", self.log);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

/// Where `pattern` first appears in `log`, which must contain it.
fn position(log: &str, pattern: &str) -> usize {
    log.find(pattern)
        .unwrap_or_else(|| panic!("no {pattern:?} in the log:\n{log}"))
}

// Health and readiness (TIM-94, decision 3).

#[test]
fn health_is_503_until_the_store_has_migrated_and_the_models_have_loaded() {
    let dir = TestDir::new();
    let gate = dir.path("gate");
    let mut daemon = Serve::new(&dir).gate(&gate).bind();

    // Bound but held before the store opens: no migration has run yet.
    let health = daemon.get("/v1/health");
    assert_eq!(health.status, 503, "{}", health.body);
    let body = health.json();
    assert_eq!(body["ready"], false);
    assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
    assert!(
        !daemon.data_dir.join("asphodel.db").exists(),
        "the store was opened before the daemon was ready"
    );

    // Every other route refuses rather than racing the startup.
    for (method, path) in [("GET", "/v1/config"), ("GET", "/v1/banks/main/chunks")] {
        let reply = daemon.send(method, path, None);
        assert_eq!(reply.status, 503, "{method} {path}: {}", reply.body);
        assert!(reply.json()["error"].is_string());
    }
    let reply = daemon.put("/v1/banks/main", &json!({}));
    assert_eq!(reply.status, 503, "{}", reply.body);
    assert!(!daemon.log.contains("asphodel listening"), "{}", daemon.log);

    fs::write(&gate, "").unwrap();
    daemon.wait_ready();
    let health = daemon.get("/v1/health");
    assert_eq!(health.status, 200);
    let body = health.json();
    assert_eq!(body["ready"], true);
    assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
    assert!(daemon.data_dir.join("asphodel.db").exists());
}

#[test]
fn a_daemon_that_fails_to_start_after_binding_exits_and_removes_its_socket() {
    // A second daemon on a locked data dir binds its own socket, then
    // fails to open the store. It must exit non-zero, never become ready,
    // and leave no socket behind.
    let dir = TestDir::new();
    let first = Serve::new(&dir).ready();
    let socket = dir.path("second.sock");
    let output = Serve::new(&dir)
        .listen(&format!("unix:{}", socket.display()))
        .command()
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let log = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{log}");
    assert!(log.contains("asphodel starting"), "{log}");
    assert!(!log.contains("asphodel listening"), "{log}");
    assert!(log.contains("locked"), "{log}");
    assert!(
        !socket.exists(),
        "the failed daemon left {}",
        socket.display()
    );
    assert_eq!(first.get("/v1/health").status, 200);
}

// The bearer token (TIM-94, decision 2).

#[test]
fn off_loopback_every_route_but_health_needs_the_bearer_token() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir).listen("0.0.0.0:0").token(TOKEN).bind();
    daemon.wait_ready();
    let addr = daemon.addr.clone();

    // Health stays open, so a readiness probe needs no secret.
    let health = request(&addr, "GET", "/v1/health", None, None).unwrap();
    assert_eq!(health.status, 200, "{}", health.body);

    for token in [None, Some("wrong-token"), Some("")] {
        for (method, path, body) in [
            ("GET", "/v1/config", None),
            ("PUT", "/v1/banks/main", Some(json!({}))),
            ("GET", "/v1/banks/main/chunks", None),
            (
                "POST",
                "/v1/banks/main/recall",
                Some(json!({"query": "anything"})),
            ),
        ] {
            let reply = request(&addr, method, path, token, body.as_ref()).unwrap();
            assert_eq!(reply.status, 401, "{method} {path} with {token:?}");
            assert!(
                reply.headers.contains("www-authenticate: bearer"),
                "{}",
                reply.headers
            );
            assert!(reply.json()["error"].is_string());
        }
    }

    let reply = request(
        &addr,
        "PUT",
        "/v1/banks/main",
        Some(TOKEN),
        Some(&json!({})),
    )
    .unwrap();
    assert_eq!(reply.status, 201, "{}", reply.body);
    let config = request(&addr, "GET", "/v1/config", Some(TOKEN), None).unwrap();
    assert_eq!(config.status, 200, "{}", config.body);
    assert_eq!(config.json()["deployment"]["token"], "[redacted]");
    assert!(!config.body.contains(TOKEN));

    daemon.sigterm();
    daemon.wait_exit();
    assert!(!daemon.log.contains(TOKEN), "{}", daemon.log);
}

#[test]
fn a_token_set_on_loopback_is_checked_too() {
    let dir = TestDir::new();
    let daemon = Serve::new(&dir).listen("127.0.0.1:0").token(TOKEN).ready();
    let reply = request(&daemon.addr, "GET", "/v1/config", None, None).unwrap();
    assert_eq!(reply.status, 401);
    let reply = request(&daemon.addr, "GET", "/v1/config", Some(TOKEN), None).unwrap();
    assert_eq!(reply.status, 200);
}

// The routes, driven with the fake models and LLM.

#[test]
fn put_bank_creates_then_merges_without_undoing_fields() {
    let dir = TestDir::new();
    let daemon = Serve::new(&dir).ready();

    let created = daemon.put(
        "/v1/banks/main",
        &json!({"owner_name": "Tim", "assistant_name": "Ash", "timezone": "Pacific/Auckland"}),
    );
    assert_eq!(created.status, 201, "{}", created.body);
    let created = created.json();
    assert_eq!(created["created"], true);
    assert_eq!(created["embedding_model"], "fake-embedder:v1");
    assert_eq!(created["reranker_model"], "fake-reranker:v1");

    // A second instance sending only some fields changes only those.
    let merged = daemon.put(
        "/v1/banks/main",
        &json!({"owner_platform_ids": ["discord:1234"]}),
    );
    assert_eq!(merged.status, 200, "{}", merged.body);
    let merged = merged.json();
    assert_eq!(merged["created"], false);
    assert_eq!(merged["id"], created["id"]);
    assert_eq!(merged["owner_name"], "Tim");
    assert_eq!(merged["assistant_name"], "Ash");
    assert_eq!(merged["timezone"], "Pacific/Auckland");

    let bad = daemon.put("/v1/banks/main", &json!({"timezone": "Not/AZone"}));
    assert_eq!(bad.status, 400, "{}", bad.body);
}

#[test]
fn a_document_is_extracted_by_the_worker_recalled_kept_and_unkept() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .script(&[json!({"reply": auckland_reply()})])
        .ready();
    daemon.create_bank("main");

    let ingested = daemon.ingest_notes("main", "notes.md");
    assert_eq!(ingested["outcome"], "stored");
    assert_eq!(ingested["chunks_queued"], 1);
    let id = daemon.wait_for_memory("main");
    daemon.wait_extracted("main");

    // The same document again changes nothing and calls no LLM.
    let again = daemon.ingest_notes("main", "notes.md");
    assert_eq!(again["outcome"], "duplicate");
    assert_eq!(again["source"], ingested["source"]);

    let recall = daemon.recall("main", "Auckland");
    assert_eq!(recall["results"][0]["kept"], false);
    assert_eq!(recall["results"][0]["kind"], "fact");
    assert!(recall["recall_id"].is_string());

    let unknown = "01a0f958-0000-7000-8000-000000000000";
    let kept = daemon.ok(daemon.post(
        "/v1/banks/main/keep",
        &json!({"ids": [id, unknown, "not-an-id"]}),
    ));
    assert_eq!(kept["kept"], json!([id]));
    assert_eq!(kept["unknown"], json!([unknown, "not-an-id"]));
    assert_eq!(
        daemon.recall("main", "Auckland")["results"][0]["kept"],
        true
    );

    let unkept = daemon.ok(daemon.post("/v1/banks/main/unkeep", &json!({"ids": [id]})));
    assert_eq!(unkept["unkept"], json!([id]));
    assert_eq!(unkept["unknown"], json!([]));
    assert_eq!(
        daemon.recall("main", "Auckland")["results"][0]["kept"],
        false
    );

    // At most 50 ids per call (TIM-94, decision 9).
    let ids: Vec<String> = (0..51).map(|_| id.clone()).collect();
    let too_many = daemon.post("/v1/banks/main/keep", &json!({ "ids": ids }));
    assert_eq!(too_many.status, 400, "{}", too_many.body);

    let config = daemon.ok(daemon.get("/v1/config"));
    assert_eq!(config["fake_llm"], true);
    assert_eq!(config["models"]["fake"], true);
}

#[test]
fn a_turn_commits_its_prefetch_and_clearing_the_session_forgets_it() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .script(&[
            json!({"reply": auckland_reply()}),
            json!({"reply": empty_reply()}),
        ])
        .ready();
    daemon.create_bank("main");
    daemon.ingest_notes("main", "notes.md");
    let id = daemon.wait_for_memory("main");
    daemon.wait_extracted("main");

    let prefetch = |daemon: &Daemon| {
        daemon.ok(daemon.post(
            "/v1/banks/main/prefetch",
            &json!({"session_id": "s1", "query": "where do I live, Auckland?"}),
        ))
    };
    let first = prefetch(&daemon);
    assert_eq!(first["injected"], json!([id]), "{first}");
    assert!(first["text"].as_str().unwrap().contains(SENTENCE));

    // The turn echoes the recall id, so the injection joins the session's
    // in-context set and isn't injected again.
    let turn = json!({
        "session_id": "s1",
        "message_at": "2026-10-01T10:00:00Z",
        "user_text": "where do I live, Auckland?",
        "assistant_text": "You live in Auckland.",
        "platform": "cli",
        "recall_id": first["recall_id"],
    });
    let ingested = daemon.ok(daemon.post("/v1/banks/main/turns", &turn));
    assert_eq!(ingested["outcome"], "stored");
    assert_eq!(ingested["speaker"]["owner"], true);
    assert_eq!(prefetch(&daemon)["injected"], json!([]));

    // A resend from the plugin's spool is a duplicate.
    let resent = daemon.ok(daemon.post("/v1/banks/main/turns", &turn));
    assert_eq!(resent["outcome"], "duplicate");
    daemon.wait_extracted("main");

    let cleared = daemon.send("POST", "/v1/banks/main/sessions/s1/clear", None);
    assert_eq!(cleared.status, 204, "{}", cleared.body);
    assert_eq!(prefetch(&daemon)["injected"], json!([id]));
}

#[test]
fn failed_chunks_are_listed_and_retried() {
    let dir = TestDir::new();
    // Five failures reach the retry cap; the retry then succeeds.
    let mut steps = vec![json!({"fail": "no_content"}); 5];
    steps.push(json!({"reply": auckland_reply()}));
    let mut daemon = Serve::new(&dir).script(&steps).ready();
    daemon.create_bank("main");
    daemon.ingest_notes("main", "notes.md");

    let failed = daemon.wait_until(
        "a failed chunk",
        |daemon| daemon.ok(daemon.get("/v1/banks/main/chunks?failed=true")),
        |chunks| chunks["failed"].as_array().is_some_and(|f| f.len() == 1),
    );
    assert_eq!(
        failed["queued"],
        json!([]),
        "the filter lists failed chunks only"
    );
    let chunk = &failed["failed"][0];
    assert_eq!(chunk["error_count"], 5);
    assert_eq!(chunk["error_kind"], "llm_no_content");
    let chunk_id = chunk["chunk"].as_str().unwrap().to_string();
    assert_eq!(daemon.chunks("main")["failed"][0]["chunk"], chunk_id);

    let unknown = "01a0f958-0000-7000-8000-000000000000";
    let retried = daemon.ok(daemon.post(
        "/v1/banks/main/chunks/retry",
        &json!({"chunks": [chunk_id, unknown]}),
    ));
    assert_eq!(retried["retried"], json!([chunk_id]));
    assert_eq!(retried["unknown"], json!([unknown]));

    daemon.wait_for_memory("main");
    daemon.wait_extracted("main");
}

#[test]
fn errors_are_json_with_the_right_status() {
    let dir = TestDir::new();
    let daemon = Serve::new(&dir).ready();
    daemon.create_bank("main");

    for (method, path, body, status) in [
        (
            "POST",
            "/v1/banks/nope/recall",
            Some(json!({"query": "x"})),
            404,
        ),
        ("GET", "/v1/banks/nope/chunks", None, 404),
        ("POST", "/v1/banks/nope/keep", Some(json!({"ids": []})), 404),
        (
            "POST",
            "/v1/banks/main/recall",
            Some(json!({"nope": 1})),
            422,
        ),
        (
            "POST",
            "/v1/banks/main/recall",
            Some(
                json!({"query": "x", "from": "2026-10-02T00:00:00Z", "to": "2026-10-01T00:00:00Z"}),
            ),
            400,
        ),
        ("GET", "/v1/nowhere", None, 404),
    ] {
        let reply = daemon.send(method, path, body.as_ref());
        assert_eq!(reply.status, status, "{method} {path}: {}", reply.body);
        assert!(
            reply.json()["error"].is_string(),
            "{method} {path}: {}",
            reply.body
        );
    }
}

// SIGTERM (TIM-94, decision 3).

#[test]
fn sigterm_refuses_ingest_finishes_the_chunk_in_flight_and_checkpoints() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .listen("127.0.0.1:0")
        .script(&[json!({"reply": auckland_reply(), "delay_ms": 3000})])
        .ready();
    daemon.create_bank("main");
    daemon.ingest_notes("main", "notes.md");
    daemon.wait_until(
        "a chunk in flight",
        |daemon| daemon.chunks("main"),
        |chunks| chunks["queued"][0]["in_flight"] == true,
    );

    daemon.sigterm();
    daemon.wait_for_line("received SIGTERM");
    // The listener is closing: new ingest is refused or never connects.
    let late = request(
        &daemon.addr,
        "POST",
        "/v1/banks/main/documents",
        None,
        Some(&json!({
            "document_id": "late.md",
            "text": "Late text.",
            "reference_date": "2026-09-30",
            "reference_date_exact": true
        })),
    );
    if let Ok(reply) = &late {
        assert!(
            !(200..300).contains(&reply.status),
            "late ingest was accepted"
        );
    }

    let status = daemon.wait_exit();
    assert!(status.success(), "{status}\n{}", daemon.log);
    let log = daemon.log.clone();
    let signal = position(&log, "received SIGTERM");
    let extracted = position(&log, "chunk extracted");
    let checkpointed = position(&log, "checkpointed the WAL");
    let stopped = position(&log, "asphodel stopped");
    assert!(
        signal < extracted && extracted < checkpointed && checkpointed < stopped,
        "{log}"
    );
    let wal = daemon.data_dir.join("asphodel.db-wal");
    assert!(
        !fs::metadata(&wal).is_ok_and(|wal| wal.len() > 0),
        "the WAL wasn't checkpointed"
    );
    drop(daemon);

    // The chunk was committed, not abandoned: a restart has nothing queued
    // and recalls the memory, and the late document was never stored.
    let mut restarted = Serve::new(&dir).listen("127.0.0.1:0").ready();
    restarted.wait_for_memory("main");
    let chunks = restarted.chunks("main");
    assert_eq!(chunks["queued"], json!([]));
    assert_eq!(chunks["failed"], json!([]));
    let late = restarted.ok(restarted.post(
        "/v1/banks/main/documents",
        &json!({
            "document_id": "late.md",
            "text": "Late text.",
            "reference_date": "2026-09-30",
            "reference_date_exact": true
        }),
    ));
    assert_eq!(
        late["outcome"], "stored",
        "late.md was stored during the drain"
    );
}

#[test]
fn chunks_queued_before_a_restart_are_extracted_after_it() {
    // No LLM on the first run, so the chunk waits on the queue; the
    // restarted daemon's worker picks it up without a new ingest.
    let dir = TestDir::new();
    let mut first = Serve::new(&dir).ready();
    first.create_bank("main");
    first.ingest_notes("main", "notes.md");
    assert_eq!(first.chunks("main")["queued"].as_array().unwrap().len(), 1);
    first.sigterm();
    assert!(first.wait_exit().success());
    drop(first);

    let mut second = Serve::new(&dir)
        .script(&[json!({"reply": auckland_reply()})])
        .ready();
    second.wait_for_memory("main");
    second.wait_extracted("main");
}

// The CLI as a client (ADR 0006, ADR 0010).

/// `asphodel` with a clean environment and `ASPHODEL_URL` pointing at
/// `daemon`, without its token.
fn cli(daemon: &Daemon) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_asphodel"));
    command
        .env_clear()
        .env("ASPHODEL_URL", daemon.addr.url())
        .stdin(Stdio::null());
    command
}

fn run(command: &mut Command) -> Output {
    command.output().unwrap()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn succeeded(output: Output) -> String {
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        stdout(&output),
        stderr(&output)
    );
    stdout(&output)
}

#[test]
fn the_cli_drives_the_daemon_over_a_unix_socket() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .script(&[json!({"reply": auckland_reply()})])
        .ready();
    let notes = dir.file("notes.md", NOTES);

    let out = succeeded(run(cli(&daemon).args([
        "bank",
        "create",
        "main",
        "--owner-name",
        "Tim",
        "--timezone",
        "Pacific/Auckland",
    ])));
    assert!(out.contains("created bank main"), "{out}");
    let out = succeeded(run(cli(&daemon).args([
        "bank",
        "config",
        "main",
        "--assistant-name",
        "Ash",
        "--json",
    ])));
    let bank: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(bank["created"], false);
    assert_eq!(bank["owner_name"], "Tim");
    assert_eq!(bank["assistant_name"], "Ash");

    // The document id defaults to the file's name.
    let out = succeeded(run(cli(&daemon).arg("ingest").arg(&notes).args([
        "--bank",
        "main",
        "--date",
        "2026-09-30",
    ])));
    assert!(out.contains("ingested notes.md into main"), "{out}");
    assert!(out.contains("1 chunks queued"), "{out}");
    let id = daemon.wait_for_memory("main");

    let out = succeeded(run(
        cli(&daemon).args(["recall", "--bank", "main", "Auckland"])
    ));
    assert!(out.contains(&id) && out.contains(SENTENCE), "{out}");
    let out = succeeded(run(cli(&daemon).args([
        "recall", "--bank", "main", "--kind", "fact", "--json", "Auckland",
    ])));
    let recall: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(recall["results"][0]["id"], id.as_str());
    let out =
        succeeded(run(cli(&daemon).args([
            "recall", "--bank", "main", "--kind", "event", "Auckland",
        ])));
    assert!(out.contains("nothing recalled"), "{out}");

    let out = succeeded(run(cli(&daemon).args(["keep", "--bank", "main", &id])));
    assert!(out.contains(&format!("kept {id}")), "{out}");
    assert_eq!(
        daemon.recall("main", "Auckland")["results"][0]["kept"],
        true
    );
    let out = succeeded(run(cli(&daemon).args(["unkeep", "--bank", "main", &id])));
    assert!(out.contains(&format!("unkept {id}")), "{out}");
    assert_eq!(
        daemon.recall("main", "Auckland")["results"][0]["kept"],
        false
    );

    // An unknown id is reported and fails the command, after the rest.
    let unknown = "01a0f958-0000-7000-8000-000000000000";
    let output = run(cli(&daemon).args(["keep", "--bank", "main", &id, unknown]));
    assert!(!output.status.success());
    assert!(stdout(&output).contains(&format!("kept {id}")));
    assert!(stderr(&output).contains(unknown), "{}", stderr(&output));

    let out = succeeded(run(cli(&daemon).args(["chunks", "--bank", "main"])));
    assert!(
        out.contains("0 queued") && out.contains("0 failed"),
        "{out}"
    );
}

#[test]
fn the_cli_lists_and_retries_failed_chunks() {
    let dir = TestDir::new();
    let mut steps = vec![json!({"fail": "status", "status": 400}); 5];
    steps.push(json!({"reply": auckland_reply()}));
    let mut daemon = Serve::new(&dir).script(&steps).ready();
    daemon.create_bank("main");
    daemon.ingest_notes("main", "notes.md");
    let failed = daemon.wait_until(
        "a failed chunk",
        |daemon| daemon.chunks("main"),
        |chunks| chunks["failed"].as_array().is_some_and(|f| f.len() == 1),
    );
    let chunk = failed["failed"][0]["chunk"].as_str().unwrap().to_string();

    let out = succeeded(run(
        cli(&daemon).args(["chunks", "--bank", "main", "--failed"])
    ));
    assert!(out.contains("1 failed") && out.contains(&chunk), "{out}");
    assert!(out.contains("HTTP 400"), "{out}");
    assert!(!out.contains("queued"), "{out}");

    let output = run(cli(&daemon).args(["chunks", "--bank", "main", "--retry"]));
    assert!(
        !output.status.success(),
        "--retry without --failed was accepted"
    );

    let out = succeeded(run(
        cli(&daemon).args(["chunks", "--bank", "main", "--failed", "--retry"])
    ));
    assert!(out.contains("1 put back on the queue"), "{out}");
    daemon.wait_for_memory("main");
    daemon.wait_extracted("main");
}

#[test]
fn the_cli_reaches_a_tcp_daemon_with_the_token_from_the_environment() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir).listen("0.0.0.0:0").token(TOKEN).bind();
    daemon.wait_ready();
    daemon.create_bank("main");

    let output = run(cli(&daemon).args(["chunks", "--bank", "main"]));
    assert!(!output.status.success());
    assert!(stderr(&output).contains("401"), "{}", stderr(&output));
    assert!(
        stderr(&output).contains("ASPHODEL_TOKEN"),
        "{}",
        stderr(&output)
    );

    let out = succeeded(run(cli(&daemon)
        .env("ASPHODEL_TOKEN", TOKEN)
        .args(["chunks", "--bank", "main"])));
    assert!(out.contains("0 queued"), "{out}");

    // --url wins over ASPHODEL_URL.
    let out = succeeded(run(cli(&daemon)
        .env("ASPHODEL_URL", "unix:/nonexistent/asphodel.sock")
        .env("ASPHODEL_TOKEN", TOKEN)
        .args(["chunks", "--bank", "main", "--url", &daemon.addr.url()])));
    assert!(out.contains("0 queued"), "{out}");

    // The token has no flag, so it never shows in a process list.
    let output = run(cli(&daemon).args(["chunks", "--bank", "main", "--token", TOKEN]));
    assert!(!output.status.success(), "--token was accepted");
}

#[test]
fn the_cli_names_a_daemon_it_cannot_reach_or_a_url_it_cannot_use() {
    let dir = TestDir::new();
    let socket = dir.path("nobody.sock");
    let url = format!("unix:{}", socket.display());
    let output = run(Command::new(env!("CARGO_BIN_EXE_asphodel"))
        .env_clear()
        .args(["recall", "--bank", "main", "anything", "--url", &url]));
    assert!(!output.status.success());
    let message = stderr(&output);
    assert!(message.contains("can't reach the daemon"), "{message}");
    assert!(message.contains(&url), "{message}");

    for url in ["https://127.0.0.1:7720", "ftp://127.0.0.1", "unix:"] {
        let output = run(Command::new(env!("CARGO_BIN_EXE_asphodel"))
            .env_clear()
            .args(["chunks", "--bank", "main", "--url", url]));
        assert!(!output.status.success(), "{url} was accepted");
        assert!(stderr(&output).contains("--url"), "{}", stderr(&output));
    }
}
