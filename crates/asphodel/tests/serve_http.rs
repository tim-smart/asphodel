//! The daemon over HTTP, and the CLI as its client, run as processes.
//!
//! Under ADR 0006, the HTTP contract includes routes under `/v1`,
//! the bearer token off loopback, `/v1/health` answering
//! 503 until the store and models are ready, SIGTERM finishing the chunk in
//! flight and checkpointing the WAL, and every CLI subcommand reaching the
//! daemon over `--url`. The tests also cover the model routes,
//! `/system-prompt`, `/agenda` and `asphodel model`; `forget`,
//! `/v1/purge/plan`, `/v1/purge/ack`, `asphodel forget` and
//! `asphodel purge plan|ack`; and `/v1/backup`,
//! `/v1/status` and the audit lists, `asphodel backup`, `status` and the
//! list commands, and the offline `asphodel restore`.
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
use std::net::{TcpListener, TcpStream};
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

    /// A tuning file with a floor for each fake model, and `extra` TOML.
    fn floors_for_fakes(&self, extra: &str) -> PathBuf {
        let path = self.path("tuning.toml");
        fs::write(
            &path,
            format!(
                "[injection.reranker_floors]\n\"fake-reranker:v1\" = 0.0\n\
                 [reconcile.embedding_floors]\n\"fake-embedder:v1\" = 0.5\n{extra}"
            ),
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
    /// More tuning TOML, after the floors.
    tuning: &'a str,
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
            tuning: "",
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

    fn tuning(mut self, extra: &'a str) -> Self {
        self.tuning = extra;
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
            .arg(self.dir.floors_for_fakes(self.tuning))
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

    fn ingest_document(&self, bank: &str, id: &str, text: &str) -> Value {
        self.ok(self.post(
            &format!("/v1/banks/{bank}/documents"),
            &json!({
                "document_id": id,
                "text": text,
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

// Health and readiness.

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

// The bearer token.

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
            ("POST", "/v1/backup", None),
            ("GET", "/v1/status", None),
            ("GET", "/v1/banks/main/purges", None),
            ("GET", "/v1/banks/main/forgets", None),
            ("GET", "/v1/banks/main/sweeps", None),
            ("GET", "/v1/banks/main/recalls", None),
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
        // No LLM is configured, so nothing can be refreshed.
        (
            "POST",
            "/v1/banks/main/models/User%20profile/refresh",
            Some(Value::Null),
            503,
        ),
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

// SIGTERM.

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

// Concurrent extraction: a pool of leases per bank behind `[llm] concurrency`.

/// `[llm] concurrency` set to `n`.
fn pool(n: u32) -> String {
    format!("[llm]\nconcurrency = {n}\n")
}

/// Queues `documents` in bank `main` on a daemon with no LLM, then stops
/// it, so the daemon a test starts next finds the whole queue at once.
fn queue_without_an_llm(dir: &TestDir, tuning: &str, documents: &[(String, String)]) {
    let mut daemon = Serve::new(dir).tuning(tuning).ready();
    daemon.create_bank("main");
    for (id, text) in documents {
        daemon.ingest_document("main", id, text);
    }
    assert_eq!(
        daemon.chunks("main")["queued"].as_array().unwrap().len(),
        documents.len()
    );
    daemon.sigterm();
    assert!(daemon.wait_exit().success(), "{}", daemon.log);
}

/// `n` one-chunk documents that share no sentence.
fn distinct_documents(n: usize) -> Vec<(String, String)> {
    (0..n)
        .map(|i| {
            (
                format!("day-{i}.md"),
                format!("# Day {i}\n\nNothing much happened on day {i}.\n"),
            )
        })
        .collect()
}

#[test]
fn five_leases_extract_in_about_a_fifth_of_the_serial_time() {
    // Ten chunks whose call 1 takes a second and finds nothing, so call 2
    // never runs: 10 s one at a time, about 2 s five at a time, plus about
    // a second to start.
    let dir = TestDir::new();
    let tuning = pool(5);
    queue_without_an_llm(&dir, &tuning, &distinct_documents(10));

    let steps = vec![json!({"reply": empty_reply(), "delay_ms": 1000}); 10];
    let started = Instant::now();
    let mut daemon = Serve::new(&dir).tuning(&tuning).script(&steps).ready();
    daemon.wait_extracted("main");
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(5),
        "took {elapsed:?} against 10 s serial\n{}",
        daemon.log
    );
}

#[test]
fn two_chunks_in_flight_stating_one_fact_make_one_memory_and_a_mention() {
    // Both chunks run call 1 before either commits, and neither finds a
    // neighbour. The second to commit finds a memory made since its search
    // at or above the floor for its claim, so it searches again and runs
    // call 2, which labels the claim a mention: one memory plus one access,
    // never a second copy (ADR 0005). Each call 1 takes 3 s, so both
    // finishing within 5 s of the start means they ran together.
    let dir = TestDir::new();
    let tuning = pool(2);
    queue_without_an_llm(
        &dir,
        &tuning,
        &[
            ("notes.md".to_string(), NOTES.to_string()),
            (
                "more-notes.md".to_string(),
                "# More notes\n\nI live in Auckland, near the harbour.\n".to_string(),
            ),
        ],
    );

    let mention = json!({"claims": [{
        "claim": "c1",
        "labels": [{"neighbour": "n1", "label": "mentioned_again"}]
    }]});
    let started = Instant::now();
    let mut daemon = Serve::new(&dir)
        .tuning(&tuning)
        .script(&[
            json!({"reply": auckland_reply(), "delay_ms": 3000}),
            json!({"reply": auckland_reply(), "delay_ms": 3000}),
            json!({"reply": mention}),
        ])
        .ready();
    daemon.wait_extracted("main");
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_millis(5500),
        "took {elapsed:?}: the chunks ran one after the other\n{}",
        daemon.log
    );

    let recall = daemon.recall("main", "where does Tim live? Auckland");
    let ids: Vec<&str> = recall["results"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|result| result["sentence"] == SENTENCE)
        .map(|result| result["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids.len(), 1, "one memory, not a copy per chunk: {recall}");
    let memory = daemon.ok(daemon.get(&format!("/v1/banks/main/memories/{}", ids[0])));
    let kinds: Vec<&str> = memory["accesses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|access| access["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["created", "mentioned_again"], "{memory}");
}

/// Eight chunks on a pool of two, where the first LLM call hits `limit`,
/// a hold of about six seconds. Every call after it waits for the hold
/// to end, so between the calls already in flight finishing and the hold
/// ending nothing is extracted, and no chunk counts a failure.
fn a_limit_pauses_every_caller_and_counts_nothing(limit: Value) {
    let dir = TestDir::new();
    let tuning = pool(2);
    queue_without_an_llm(&dir, &tuning, &distinct_documents(8));

    // One call may have started before the limit came back; the rest run
    // after the hold. 8 more replies: every chunk, the limited one again.
    let mut steps = vec![limit, json!({"reply": empty_reply(), "delay_ms": 1000})];
    steps.extend(vec![json!({"reply": empty_reply(), "delay_ms": 500}); 7]);
    let mut daemon = Serve::new(&dir).tuning(&tuning).script(&steps).ready();

    let snapshot = |daemon: &Daemon| {
        let chunks = daemon.chunks("main");
        let queued = chunks["queued"].as_array().unwrap().clone();
        for chunk in &queued {
            assert_eq!(
                chunk["error_count"], 0,
                "a hold isn't the chunk's failure: {chunks}"
            );
        }
        assert_eq!(chunks["failed"], json!([]), "{chunks}");
        queued.len()
    };
    std::thread::sleep(Duration::from_millis(1500));
    let early = snapshot(&daemon);
    std::thread::sleep(Duration::from_millis(1500));
    let later = snapshot(&daemon);
    assert_eq!(
        early, later,
        "chunks were extracted while the hold was on\n{}",
        daemon.log
    );
    assert!(later >= 7, "{later} chunks still queued");

    daemon.wait_extracted("main");
}

#[test]
fn a_usage_limit_pauses_every_caller_and_counts_no_failure() {
    use asphodel_core::{Clock, SystemClock};
    // The daemon runs on the system clock, so the reset is in its time.
    let resets_at = SystemClock.now() + jiff::SignedDuration::from_secs(6);
    a_limit_pauses_every_caller_and_counts_nothing(
        json!({"fail": "usage_limited", "resets_at": resets_at.to_string()}),
    );
}

#[test]
fn a_429_with_retry_after_pauses_every_caller_and_counts_no_failure() {
    a_limit_pauses_every_caller_and_counts_nothing(
        json!({"fail": "status", "status": 429, "retry_after_secs": 6}),
    );
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

// Mental models and the system prompt block.

/// A refresh reply adding one entry citing the first memory in its input.
fn adds_entry(text: &str) -> Value {
    json!({"operations": [{"op": "add", "entry": null, "text": text, "cites": ["m1"]}]})
}

const ENTRY: &str = "Tim lives in Auckland.";

#[test]
fn models_are_created_listed_edited_and_refreshed_over_http() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .script(&[
            json!({"reply": auckland_reply()}),
            json!({"reply": adds_entry(ENTRY)}),
        ])
        .ready();
    daemon.create_bank("main");
    daemon.ingest_notes("main", "notes.md");
    let memory = daemon.wait_for_memory("main");
    daemon.wait_extracted("main");

    // Only "User profile" is seeded, empty and never refreshed.
    let models = daemon.ok(daemon.get("/v1/banks/main/models"));
    assert_eq!(models.as_array().unwrap().len(), 1);
    assert_eq!(models[0]["name"], "User profile");
    assert_eq!(models[0]["entries"], json!([]));
    assert_eq!(models[0]["last_refreshed_at"], Value::Null);

    // The profile takes 500 of the 800 tokens.
    let created = daemon.post(
        "/v1/banks/main/models",
        &json!({"name": "Plans", "question": "Where is Tim going?", "kinds": ["event"],
                "max_tokens": 300}),
    );
    assert_eq!(created.status, 201, "{}", created.body);
    let plans = created.json();
    assert_eq!(plans["enabled"], true);
    assert_eq!(plans["kinds"], json!(["event"]));
    let over = daemon.post(
        "/v1/banks/main/models",
        &json!({"name": "Big", "question": "Anything?", "max_tokens": 1}),
    );
    assert_eq!(over.status, 422, "{}", over.body);
    assert!(over.json()["error"].as_str().unwrap().contains("budget"));
    let duplicate = daemon.post(
        "/v1/banks/main/models",
        &json!({"name": "Plans", "question": "Again?", "max_tokens": 1}),
    );
    assert_eq!(duplicate.status, 409, "{}", duplicate.body);

    let edited = daemon.ok(daemon.send(
        "PATCH",
        "/v1/banks/main/models/Plans",
        Some(&json!({"enabled": false, "min_volatility": "weeks"})),
    ));
    assert_eq!(edited["enabled"], false);
    assert_eq!(edited["min_volatility"], "weeks");
    assert_eq!(edited["question"], "Where is Tim going?");
    let cleared = daemon.ok(daemon.send(
        "PATCH",
        "/v1/banks/main/models/Plans",
        Some(&json!({"min_volatility": null})),
    ));
    assert_eq!(cleared["min_volatility"], Value::Null);

    // A forced refresh calls the LLM; the next one finds nothing changed.
    let refreshed = daemon.ok(daemon.post(
        "/v1/banks/main/models/User%20profile/refresh?force=true",
        &Value::Null,
    ));
    assert_eq!(refreshed["outcome"], "applied", "{refreshed}");
    assert_eq!(refreshed["detail"]["added"].as_array().unwrap().len(), 1);
    let unchanged =
        daemon.ok(daemon.post("/v1/banks/main/models/User%20profile/refresh", &Value::Null));
    assert_eq!(unchanged["outcome"], "unchanged", "{unchanged}");

    let profile = &daemon.ok(daemon.get("/v1/banks/main/models"))[0];
    assert_eq!(profile["entries"][0]["text"], ENTRY);
    assert_eq!(profile["entries"][0]["cites"], json!([memory]));
    assert!(profile["last_refreshed_at"].is_string());

    // The block holds the entry and the pointer line, and a session's fetch
    // puts the cited memory in context, so prefetch doesn't inject it.
    let block = daemon.ok(daemon.get("/v1/banks/main/system-prompt?session_id=s1"));
    let text = block["text"].as_str().unwrap();
    assert!(
        text.contains("User profile") && text.contains(ENTRY),
        "{text}"
    );
    assert!(text.contains("memory_recall"), "{text}");
    assert!(!text.contains("Plans"), "a disabled model was rendered");
    assert_eq!(block["cited"], json!([memory]));
    assert_eq!(
        daemon.ok(daemon.get("/v1/banks/main/system-prompt"))["id"],
        block["id"],
        "the cached block was rebuilt"
    );
    let prefetch = daemon.ok(daemon.post(
        "/v1/banks/main/prefetch",
        &json!({"session_id": "s1", "query": "where does Tim live? Auckland"}),
    ));
    assert!(
        !prefetch["injected"]
            .as_array()
            .unwrap()
            .contains(&json!(memory)),
        "{prefetch}"
    );
    let elsewhere = daemon.ok(daemon.post(
        "/v1/banks/main/prefetch",
        &json!({"session_id": "s2", "query": "where does Tim live? Auckland"}),
    ));
    assert_eq!(elsewhere["injected"], json!([memory]));

    let agenda = daemon.ok(daemon.get("/v1/banks/main/agenda"));
    assert_eq!(
        agenda,
        json!({"dated": [], "folded": 0, "routines": [], "undated_tasks": []})
    );
}

// Forget and the purge pause (ADR 0009; ADR 0010).

#[test]
fn forget_erases_over_http_and_the_cli() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .script(&[json!({"reply": auckland_reply()})])
        .ready();
    daemon.create_bank("main");
    let ingested = daemon.ingest_notes("main", "notes.md");
    let id = daemon.wait_for_memory("main");
    daemon.wait_extracted("main");

    // Nothing was queued before the forget, so the erase runs at once.
    let out = succeeded(run(cli(&daemon).args(["forget", "--bank", "main", &id])));
    assert!(out.contains(&format!("forgot {id}")), "{out}");
    let recall = daemon.recall("main", "where does Tim live? Auckland");
    assert!(
        !recall["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|result| result["id"] == id.as_str()),
        "{recall}"
    );

    // The key and hash are the tombstone: the same document brings nothing
    // back and queues nothing.
    let again = daemon.ingest_notes("main", "notes.md");
    assert_eq!(again["outcome"], "duplicate");
    assert_eq!(again["source"], ingested["source"]);
    assert_eq!(daemon.chunks("main")["queued"], json!([]));

    // A forgotten id is unknown from then on, over HTTP and the CLI.
    let forgotten =
        daemon.ok(daemon.post("/v1/banks/main/forget", &json!({"ids": [id, "not-an-id"]})));
    assert_eq!(forgotten["forgotten"], json!([]));
    assert_eq!(forgotten["unknown"], json!([id, "not-an-id"]));
    let output = run(cli(&daemon).args(["forget", "--bank", "main", &id]));
    assert!(!output.status.success());
    assert!(stderr(&output).contains(&id), "{}", stderr(&output));

    let ids: Vec<String> = (0..51).map(|_| id.clone()).collect();
    let too_many = daemon.post("/v1/banks/main/forget", &json!({ "ids": ids }));
    assert_eq!(too_many.status, 400, "{}", too_many.body);
    let no_bank = daemon.post("/v1/banks/nobody/forget", &json!({"ids": [id]}));
    assert_eq!(no_bank.status, 404, "{}", no_bank.body);
}

#[test]
fn a_changed_fingerprint_pauses_purge_until_the_cli_acks_the_running_hash() {
    let dir = TestDir::new();
    let mut first = Serve::new(&dir).ready();
    first.create_bank("main");
    let config = first.ok(first.get("/v1/config"));
    assert_eq!(config["purge"]["state"], "running");
    let stored = config["deletion_fingerprint"].as_str().unwrap().to_string();
    let plan = first.ok(first.get("/v1/purge/plan"));
    assert_eq!(plan["pause"]["state"], "running");
    assert_eq!(plan["changed"], json!([]));
    first.sigterm();
    assert!(first.wait_exit().success());
    drop(first);

    let delta = "[purge]\ndelta = 0.5\n";
    let mut second = Serve::new(&dir).tuning(delta).ready();
    let config = second.ok(second.get("/v1/config"));
    let current = config["deletion_fingerprint"].as_str().unwrap().to_string();
    assert_ne!(current, stored);
    assert_eq!(
        config["purge"],
        json!({"state": "paused", "stored": stored})
    );

    let out = succeeded(run(cli(&second).args(["purge", "plan"])));
    assert!(out.contains("purge: paused"), "{out}");
    assert!(out.contains("changed: purge.delta"), "{out}");
    assert!(out.contains(&current), "{out}");
    let plan: Value = serde_json::from_str(&succeeded(run(
        cli(&second).args(["purge", "plan", "--json"])
    )))
    .unwrap();
    assert_eq!(plan["current"], current.as_str());
    assert_eq!(plan["changed"], json!(["purge.delta"]));

    // Only the hash the running daemon computed is accepted.
    let output = run(cli(&second).args(["purge", "ack", "--hash", "nope"]));
    assert!(!output.status.success());
    assert!(stderr(&output).contains("409"), "{}", stderr(&output));
    let stale = second.post("/v1/purge/ack", &json!({"hash": stored}));
    assert_eq!(stale.status, 409, "{}", stale.body);
    assert_eq!(
        second.ok(second.get("/v1/config"))["purge"]["state"],
        "paused"
    );

    let out = succeeded(run(cli(&second).args(["purge", "ack", "--hash", &current])));
    assert!(out.contains("acknowledged"), "{out}");
    assert_eq!(
        second.ok(second.get("/v1/config"))["purge"]["state"],
        "running"
    );
    assert_eq!(
        second.ok(second.get("/v1/purge/plan"))["changed"],
        json!([])
    );
    second.sigterm();
    assert!(second.wait_exit().success());
    drop(second);

    // The ack is stored, so it holds after a restart.
    let third = Serve::new(&dir).tuning(delta).ready();
    assert_eq!(
        third.ok(third.get("/v1/config"))["purge"]["state"],
        "running"
    );
}

#[test]
fn a_ready_erase_runs_after_a_restart_without_an_llm() {
    // The forget arrives while a chunk queued
    // before it is in flight, so its erase waits. SIGTERM lets that chunk
    // finish and stops the worker before the erase runs. Restarted with no
    // LLM there's no worker, but the erase is ready and must still run.
    let dir = TestDir::new();
    let mut first = Serve::new(&dir)
        .script(&[
            json!({"reply": auckland_reply()}),
            json!({"reply": empty_reply(), "delay_ms": 3000}),
        ])
        .ready();
    first.create_bank("main");
    first.ingest_notes("main", "notes.md");
    let id = first.wait_for_memory("main");
    first.wait_extracted("main");
    first.ok(first.post(
        "/v1/banks/main/documents",
        &json!({
            "document_id": "other.md",
            "text": "# Other\n\nNothing to remember.\n",
            "reference_date": "2026-09-30",
            "reference_date_exact": true,
            "timezone": null
        }),
    ));
    first.wait_until(
        "the second document in flight",
        |daemon| daemon.chunks("main"),
        |chunks| chunks["queued"][0]["in_flight"] == true,
    );
    let forgotten = first.ok(first.post("/v1/banks/main/forget", &json!({"ids": [id]})));
    assert_eq!(forgotten["forgotten"], json!([id]));
    first.sigterm();
    assert!(first.wait_exit().success());
    assert!(!first.log.contains("erased a chain"), "{}", first.log);
    drop(first);

    let mut second = Serve::new(&dir).ready();
    second.wait_for_line("erased a chain");
}

// Backup, restore, status and the audit lists (ADR 0010,
// "Backup and restore" and "Sweeps, pauses and failures").
//
// The contract these tests pin, beyond what the ADR says:
//
// - `POST /v1/backup` answers 200 with the copy as its body and the
// copy's SHA-256 (lowercase hex) and length in bytes in
// `SHA256_HEADER` and `LENGTH_HEADER`, and leaves no temporary file in
// the data dir. It needs the bearer token like every route but health.
// - `asphodel backup --out <file|->` fails, and leaves nothing at `<file>`,
// when the stream is cut short, its length or hash doesn't match the
// headers, or, for a file, the copy fails `PRAGMA integrity_check`.
// - `asphodel restore <file> --data-dir <dir>` keeps the old database (and
// its WAL) in the data dir under another name, and writes a daemon-wide
// `restored` edit row whose details hold `backed_up_at`, `restored_at`
// and `binary_version`. The backup time has to travel inside the copy,
// since the stream reaches the restore through pipes.
// - `GET /v1/status` holds `attention` (an array, empty when nothing needs
// it), `last_backup_at`, `last_sweep`, `pre_migration_copy` and
// `banks.<bank>.{queued, failed_chunks, failed_refreshes}`. `asphodel
// status` prints it, and exits non-zero whenever `attention` isn't empty,
// with `--json` too.
// - `GET /v1/banks/{bank}/{purges,forgets,sweeps,recalls}` answer
// `{"<list>": [...]}`, and `asphodel <list> --bank <bank>` prints them.

/// The response header holding the backup's SHA-256, as lowercase hex.
const SHA256_HEADER: &str = "asphodel-sha256";

/// The response header holding the backup's length in bytes.
const LENGTH_HEADER: &str = "asphodel-length";

/// Every SQLite database file starts with this.
const SQLITE_MAGIC: &[u8] = b"SQLite format 3\0";

/// A reply read as bytes, for the backup stream, which isn't UTF-8.
struct RawReply {
    status: u16,
    /// The header block, lowercased.
    headers: String,
    /// The body, dechunked.
    body: Vec<u8>,
}

impl RawReply {
    fn header(&self, name: &str) -> Option<&str> {
        header(&self.headers, name)
    }
}

/// The value of `name` in a lowercased header block.
fn header<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
    headers.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == name).then(|| value.trim())
    })
}

/// A bodiless request on its own connection, read as bytes.
fn request_raw(addr: &Addr, method: &str, path: &str) -> RawReply {
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: asphodel\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
    );
    let mut response = Vec::new();
    match addr {
        Addr::Tcp(addr) => {
            let mut stream = TcpStream::connect(addr).unwrap();
            stream.set_read_timeout(Some(SETTLE)).unwrap();
            stream.write_all(head.as_bytes()).unwrap();
            stream.read_to_end(&mut response).unwrap();
        }
        Addr::Unix(path) => {
            let mut stream = UnixStream::connect(path).unwrap();
            stream.set_read_timeout(Some(SETTLE)).unwrap();
            stream.write_all(head.as_bytes()).unwrap();
            stream.read_to_end(&mut response).unwrap();
        }
    }
    let split = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("a header block");
    let headers = String::from_utf8_lossy(&response[..split]).to_lowercase();
    let status = headers
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("no status line in:\n{headers}"));
    let body = &response[split + 4..];
    let body = if headers.contains("transfer-encoding: chunked") {
        dechunk_bytes(body)
    } else {
        body.to_vec()
    };
    RawReply {
        status,
        headers,
        body,
    }
}

fn dechunk_bytes(mut body: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let line = body
            .windows(2)
            .position(|window| window == b"\r\n")
            .expect("a chunk size line");
        let size = std::str::from_utf8(&body[..line]).unwrap();
        let size = usize::from_str_radix(size.trim(), 16).expect("a hex chunk size");
        if size == 0 {
            return out;
        }
        let rest = &body[line + 2..];
        out.extend_from_slice(&rest[..size]);
        body = &rest[size + 2..];
    }
}

/// The SHA-256 of `bytes` as lowercase hex, from coreutils, so the tests
/// need no hashing dependency.
fn sha256(bytes: &[u8]) -> String {
    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    stdout(&output)
        .split_whitespace()
        .next()
        .unwrap()
        .to_string()
}

/// Serves one reply on a loopback port, as a daemon whose backup went wrong
/// would: it reads the request, writes `head` and then `body`, and closes
/// the connection. Returns the `--url` that reaches it.
fn serve_once(head: String, body: Vec<u8>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let Ok((stream, _)) = listener.accept() else {
            return;
        };
        stream.set_read_timeout(Some(SETTLE)).unwrap();
        // Read the whole request first, so closing with it unread can't
        // reset the connection before the client reads the reply.
        let mut reader = BufReader::new(stream);
        let mut length = 0;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                break;
            }
            if let Some(value) = header(&line.to_lowercase(), "content-length") {
                length = value.parse().unwrap_or(0);
            }
        }
        let mut request_body = vec![0; length];
        let _ = reader.read_exact(&mut request_body);
        let mut stream = reader.into_inner();
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(&body);
        let _ = stream.shutdown(std::net::Shutdown::Write);
    });
    url
}

/// A 200 head carrying `backup`'s headers, with `replace` swapped in and a
/// `Content-Length` of `length`. The framing headers aren't carried over:
/// the body the fake serves is framed by its own length.
fn backup_head(backup: &RawReply, replace: &[(&str, String)], length: usize) -> String {
    let mut head = String::from("HTTP/1.1 200 OK\r\n");
    for line in backup.headers.lines().skip(1) {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if ["content-length", "transfer-encoding", "connection", "date"].contains(&name) {
            continue;
        }
        let value = replace
            .iter()
            .find(|(replaced, _)| *replaced == name)
            .map_or(value.trim(), |(_, value)| value.as_str());
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str(&format!(
        "content-length: {length}\r\nconnection: close\r\n\r\n"
    ));
    head
}

/// `asphodel` with a clean environment and `ASPHODEL_URL` set to `url`.
fn cli_at(url: &str) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_asphodel"));
    command
        .env_clear()
        .env("ASPHODEL_URL", url)
        .stdin(Stdio::null());
    command
}

/// `asphodel restore <backup> --data-dir <data>`, which runs offline.
fn restore(backup: &Path, data: &Path) -> Output {
    run(Command::new(env!("CARGO_BIN_EXE_asphodel"))
        .env_clear()
        .stdin(Stdio::null())
        .arg("restore")
        .arg(backup)
        .arg("--data-dir")
        .arg(data))
}

/// The details of every daemon-wide `restored` edit row in the store under
/// `data`, which no daemon may hold.
fn restored_rows(data: &Path) -> Vec<Value> {
    use asphodel_core::store::{OpenOptions, Store};
    use asphodel_core::{Clock, SystemClock};
    use std::sync::Arc;

    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    let store = Store::open(data, OpenOptions::default(), clock).unwrap();
    let conn = store.connection();
    let mut statement = conn
        .prepare("SELECT details FROM edits WHERE kind = 'restored' AND bank_id IS NULL")
        .unwrap();
    statement
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .map(|details| serde_json::from_str(&details.unwrap()).unwrap())
        .collect()
}

/// The names in `dir`, sorted.
fn names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Whether the file at `path` is a SQLite database.
fn is_sqlite(path: &Path) -> bool {
    fs::read(path).is_ok_and(|bytes| bytes.starts_with(SQLITE_MAGIC))
}

/// What a running daemon keeps in its data dir: the database, its WAL
/// files and the lock.
const LIVE_FILES: &[&str] = &["asphodel.db", "asphodel.db-shm", "asphodel.db-wal", "lock"];

#[test]
fn backup_streams_a_checked_copy_with_its_hash_and_length() {
    let dir = TestDir::new();
    let daemon = Serve::new(&dir).ready();
    daemon.create_bank("main");

    let backup = request_raw(&daemon.addr, "POST", "/v1/backup");
    assert_eq!(
        backup.status,
        200,
        "{}",
        String::from_utf8_lossy(&backup.body)
    );
    assert!(backup.body.starts_with(SQLITE_MAGIC), "not a SQLite file");
    let length = backup.body.len().to_string();
    assert_eq!(backup.header(LENGTH_HEADER), Some(length.as_str()));
    let hash = sha256(&backup.body);
    assert_eq!(backup.header(SHA256_HEADER), Some(hash.as_str()));

    // The temporary file the online backup wrote is gone once it's sent.
    let deadline = Instant::now() + SETTLE;
    loop {
        let left: Vec<String> = names(&daemon.data_dir)
            .into_iter()
            .filter(|name| !LIVE_FILES.contains(&name.as_str()))
            .collect();
        if left.is_empty() {
            break;
        }
        assert!(Instant::now() < deadline, "left in the data dir: {left:?}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_backup_restores_offline_and_writes_a_restored_edit_row() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .script(&[json!({"reply": auckland_reply()})])
        .ready();
    daemon.create_bank("main");
    daemon.ingest_notes("main", "notes.md");
    let id = daemon.wait_for_memory("main");
    daemon.wait_extracted("main");

    let file = dir.path("backup.db");
    let out = succeeded(run(cli(&daemon).arg("backup").arg("--out").arg(&file)));
    assert!(is_sqlite(&file), "{out}");
    let piped = run(cli(&daemon).args(["backup", "--out", "-"]));
    assert!(piped.status.success(), "{}", stderr(&piped));
    assert!(piped.stdout.starts_with(SQLITE_MAGIC), "not a SQLite file");

    // Forgotten after the backup, so a restore brings it back (ADR 0010,
    // "Consequences"). SIGKILL leaves the forget in the WAL: the restore
    // has to move the WAL aside with the database, or SQLite would replay
    // the old store's frames onto the restored one.
    let forgotten = daemon.ok(daemon.post("/v1/banks/main/forget", &json!({"ids": [id]})));
    assert_eq!(forgotten["forgotten"], json!([id]));
    let data = daemon.data_dir.clone();
    drop(daemon);

    let before = names(&data);
    let output = restore(&file, &data);
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        stdout(&output),
        stderr(&output)
    );
    // The old database is moved aside, not deleted.
    let aside: Vec<String> = names(&data)
        .into_iter()
        .filter(|name| !LIVE_FILES.contains(&name.as_str()) && !before.contains(name))
        .collect();
    assert!(
        aside.iter().any(|name| is_sqlite(&data.join(name))),
        "no old database kept in {:?}",
        names(&data)
    );

    let mut daemon = Serve::new(&dir).ready();
    let recall = daemon.recall("main", "where does Tim live? Auckland");
    assert_eq!(recall["results"][0]["id"], id.as_str(), "{recall}");
    assert_eq!(recall["results"][0]["sentence"], SENTENCE);
    daemon.sigterm();
    assert!(daemon.wait_exit().success(), "{}", daemon.log);
    drop(daemon);

    // One daemon-wide `restored` row, with the backup time, the restore time
    // and the binary version.
    let rows = restored_rows(&data);
    assert_eq!(rows.len(), 1, "{rows:?}");
    let details = &rows[0];
    assert_eq!(details["binary_version"], env!("CARGO_PKG_VERSION"));
    assert!(!details["backed_up_at"].is_null(), "{details}");
    assert!(!details["restored_at"].is_null(), "{details}");
}

/// Every file in `dir` with its bytes, sorted by name. The lock file's
/// bytes are left out: they're the pid of whoever last took the lock, a
/// courtesy for operators that a refused restore rewrites too.
fn snapshot(dir: &Path) -> Vec<(String, Vec<u8>)> {
    names(dir)
        .into_iter()
        .map(|name| {
            let bytes = if name == "lock" {
                Vec::new()
            } else {
                fs::read(dir.join(&name)).unwrap()
            };
            (name, bytes)
        })
        .collect()
}

#[test]
fn restore_refuses_while_the_lock_is_held_and_a_damaged_newer_or_foreign_copy() {
    let dir = TestDir::new();
    let daemon = Serve::new(&dir).ready();
    daemon.create_bank("main");
    let file = dir.path("backup.db");
    succeeded(run(cli(&daemon).arg("backup").arg("--out").arg(&file)));

    // The daemon holds the data-dir lock, so the restore can't run beside it.
    let before = names(&daemon.data_dir);
    let output = restore(&file, &daemon.data_dir);
    assert!(!output.status.success(), "restored under a running daemon");
    assert!(stderr(&output).contains("locked"), "{}", stderr(&output));
    assert_eq!(names(&daemon.data_dir), before);
    assert_eq!(daemon.get("/v1/health").status, 200);
    let data = daemon.data_dir.clone();
    // SIGKILL leaves the WAL as it was, so the snapshot holds a live WAL.
    drop(daemon);

    // Each copy is refused before anything in the data dir moves or
    // changes, and no staged or moved-aside file is left.
    let full = fs::read(&file).unwrap();
    // `user_version` is the big-endian u32 at offset 60 of the header.
    let mut newer = full.clone();
    let version = asphodel_core::store::SCHEMA_VERSION + 1;
    newer[60..64].copy_from_slice(&version.to_be_bytes());
    let cases = [
        ("a newer schema", newer, "schema version"),
        (
            "a cut-short copy",
            full[..full.len() / 2].to_vec(),
            "integrity check",
        ),
        // An empty file is a valid, empty SQLite database.
        (
            "a database that isn't a store",
            Vec::new(),
            "isn't an Asphodel store",
        ),
    ];
    let before = snapshot(&data);
    for (case, bytes, message) in cases {
        let copy = dir.path("copy.db");
        fs::write(&copy, bytes).unwrap();
        let output = restore(&copy, &data);
        assert!(!output.status.success(), "restored {case}");
        assert!(
            stderr(&output).contains(message),
            "{case}: {}",
            stderr(&output)
        );
        assert!(
            snapshot(&data) == before,
            "{case} changed {:?}",
            names(&data)
        );
    }
}

#[test]
fn the_cli_rejects_a_truncated_or_damaged_backup_stream() {
    let dir = TestDir::new();
    let daemon = Serve::new(&dir).ready();
    daemon.create_bank("main");
    let backup = request_raw(&daemon.addr, "POST", "/v1/backup");
    assert_eq!(backup.status, 200);
    let full = backup.body.clone();
    let half = full[..full.len() / 2].to_vec();

    let backup_to = |url: &str, out: &str| run(cli_at(url).args(["backup", "--out", out]));
    let out = dir.path("out.db");
    let out_arg = out.to_str().unwrap();

    // The control: the daemon's own reply, replayed, is accepted, so the
    // failures below come from what was changed and not from the fake.
    let url = serve_once(backup_head(&backup, &[], full.len()), full.clone());
    succeeded(backup_to(&url, out_arg));
    assert_eq!(fs::read(&out).unwrap(), full);
    fs::remove_file(&out).unwrap();

    let mut flipped = full.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xff;
    let cases = [
        (
            "a complete-looking reply shorter than its length header",
            backup_head(&backup, &[], half.len()),
            half.clone(),
        ),
        (
            "a byte that doesn't match the hash",
            backup_head(&backup, &[], full.len()),
            flipped,
        ),
        (
            // The headers match the body, so only the integrity check of a
            // file target can catch it.
            "a cut-short copy whose headers match it",
            backup_head(
                &backup,
                &[
                    (SHA256_HEADER, sha256(&half)),
                    (LENGTH_HEADER, half.len().to_string()),
                ],
                half.len(),
            ),
            half.clone(),
        ),
    ];
    for (case, head, body) in cases {
        let url = serve_once(head, body);
        let output = backup_to(&url, out_arg);
        assert!(!output.status.success(), "accepted {case}");
        assert!(!out.exists(), "left a file behind after {case}");
    }

    // Writing to stdout, a short stream fails the command too, so a pipe
    // into storage can tell.
    let url = serve_once(backup_head(&backup, &[], half.len()), half);
    let output = backup_to(&url, "-");
    assert!(
        !output.status.success(),
        "accepted a short stream on stdout"
    );
}

#[test]
fn status_needs_attention_while_a_chunk_has_failed() {
    let dir = TestDir::new();
    let mut steps = vec![json!({"fail": "no_content"}); 5];
    steps.push(json!({"reply": auckland_reply()}));
    let mut daemon = Serve::new(&dir).script(&steps).ready();
    daemon.create_bank("main");

    let status = daemon.ok(daemon.get("/v1/status"));
    assert_eq!(status["attention"], json!([]), "{status}");
    assert_eq!(status["last_backup_at"], Value::Null, "{status}");
    succeeded(run(cli(&daemon).arg("status")));

    // A completed backup is reported.
    succeeded(run(cli(&daemon)
        .arg("backup")
        .arg("--out")
        .arg(dir.path("backup.db"))));
    let status = daemon.ok(daemon.get("/v1/status"));
    assert!(status["last_backup_at"].is_string(), "{status}");

    daemon.ingest_notes("main", "notes.md");
    daemon.wait_until(
        "a failed chunk",
        |daemon| daemon.chunks("main"),
        |chunks| chunks["failed"].as_array().is_some_and(|f| f.len() == 1),
    );
    let status = daemon.ok(daemon.get("/v1/status"));
    assert_eq!(status["banks"]["main"]["failed_chunks"], 1, "{status}");
    assert_ne!(status["attention"], json!([]), "{status}");
    let output = run(cli(&daemon).arg("status"));
    assert!(!output.status.success(), "{}", stdout(&output));
    let output = run(cli(&daemon).args(["status", "--json"]));
    assert!(!output.status.success(), "{}", stdout(&output));
    let printed: Value = serde_json::from_str(&stdout(&output)).unwrap();
    assert_ne!(printed["attention"], json!([]), "{printed}");

    succeeded(run(
        cli(&daemon).args(["chunks", "--bank", "main", "--failed", "--retry"])
    ));
    daemon.wait_for_memory("main");
    daemon.wait_extracted("main");
    succeeded(run(cli(&daemon).arg("status")));
}

#[test]
fn the_audit_lists_hold_no_content_except_recalls() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .script(&[json!({"reply": auckland_reply()})])
        .ready();
    daemon.create_bank("main");
    daemon.ingest_notes("main", "notes.md");
    let id = daemon.wait_for_memory("main");
    daemon.wait_extracted("main");

    // `wait_for_memory` recalled with this query, and recalls keep theirs.
    let query = "where does Tim live? Auckland";
    let recalls = daemon.ok(daemon.get("/v1/banks/main/recalls"));
    assert!(
        recalls["recalls"].as_array().is_some_and(|r| !r.is_empty()),
        "{recalls}"
    );
    assert!(recalls.to_string().contains(query), "{recalls}");
    let out = succeeded(run(cli(&daemon).args(["recalls", "--bank", "main"])));
    assert!(out.contains(query), "{out}");

    daemon.ok(daemon.post("/v1/banks/main/forget", &json!({"ids": [id]})));
    let forgets = daemon.ok(daemon.get("/v1/banks/main/forgets"));
    assert_eq!(
        forgets["forgets"].as_array().map(Vec::len),
        Some(1),
        "{forgets}"
    );
    let out = succeeded(run(cli(&daemon).args(["forgets", "--bank", "main"])));
    for listed in [forgets.to_string(), out] {
        assert!(listed.contains(&id), "{listed}");
        assert!(
            !listed.contains("Auckland") && !listed.contains(SENTENCE),
            "{listed}"
        );
    }

    let purges = daemon.ok(daemon.get("/v1/banks/main/purges"));
    assert_eq!(purges["purges"], json!([]), "{purges}");
    let sweeps = daemon.ok(daemon.get("/v1/banks/main/sweeps"));
    assert!(sweeps["sweeps"].is_array(), "{sweeps}");
    for list in ["purges", "sweeps"] {
        succeeded(run(cli(&daemon).args([list, "--bank", "main"])));
    }
    let unknown = daemon.get("/v1/banks/nobody/forgets");
    assert_eq!(unknown.status, 404, "{}", unknown.body);
}

#[test]
fn a_restored_store_keeps_its_fingerprint_and_pauses_purge_under_another() {
    // The stored fingerprint travels in the copy: a backup taken under
    // `purge.delta = 0.5`, restored over a store whose fingerprint matches
    // this daemon's, pauses purge at the next start (ADR 0009, ADR 0010).
    let source = TestDir::new();
    let delta = Serve::new(&source).tuning("[purge]\ndelta = 0.5\n").ready();
    delta.create_bank("main");
    let backed_up = delta.ok(delta.get("/v1/config"))["deletion_fingerprint"]
        .as_str()
        .unwrap()
        .to_string();
    let file = source.path("backup.db");
    succeeded(run(cli(&delta).arg("backup").arg("--out").arg(&file)));
    drop(delta);

    let dir = TestDir::new();
    let mut first = Serve::new(&dir).ready();
    let current = first.ok(first.get("/v1/config"))["deletion_fingerprint"]
        .as_str()
        .unwrap()
        .to_string();
    assert_ne!(current, backed_up);
    assert_eq!(first.ok(first.get("/v1/status"))["attention"], json!([]));
    first.sigterm();
    assert!(first.wait_exit().success());
    let data = first.data_dir.clone();
    drop(first);

    let output = restore(&file, &data);
    assert!(output.status.success(), "{}", stderr(&output));
    let second = Serve::new(&dir).ready();
    let status = second.ok(second.get("/v1/status"));
    assert_eq!(
        status["purge"],
        json!({"state": "paused", "stored": backed_up}),
        "{status}"
    );
    assert_eq!(status["deletion_fingerprint"], current.as_str());
    assert_ne!(status["attention"], json!([]), "{status}");
    let output = run(cli(&second).arg("status"));
    assert!(!output.status.success(), "{}", stdout(&output));
    let out = stdout(&output);
    assert!(out.contains(&backed_up) && out.contains(&current), "{out}");
}

/// SQL that undoes the latest registered migration, `SCHEMA_VERSION`'s, by
/// dropping the tables and indexes it creates. It only handles a migration
/// that creates tables and indexes and nothing else, and fails loudly on any
/// other, so this test is extended when such a migration lands rather than
/// downgrading to a schema no older binary wrote.
fn undo_latest_migration() -> String {
    use asphodel_core::store::SCHEMA_VERSION;

    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../asphodel-core/migrations");
    let mut files: Vec<(u32, PathBuf)> = fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            let version = name.split('_').next()?.parse().ok()?;
            Some((version, path))
        })
        .collect();
    files.sort();
    let (version, path) = files.last().expect("migrations exist");
    assert_eq!(
        *version,
        SCHEMA_VERSION,
        "the newest migration file, {}, is the latest registered one",
        path.display()
    );
    let sql = fs::read_to_string(path).unwrap();
    let mut undo = Vec::new();
    for statement in sql
        .lines()
        .filter(|line| !line.trim_start().starts_with("--"))
        .collect::<Vec<_>>()
        .join("\n")
        .split(';')
        .map(str::trim)
        .filter(|statement| !statement.is_empty())
    {
        let words: Vec<&str> = statement.split_whitespace().collect();
        let upper: Vec<String> = words.iter().map(|word| word.to_uppercase()).collect();
        let name = |at: usize| {
            let at = if upper.get(at..at + 3) == Some(&["IF".into(), "NOT".into(), "EXISTS".into()])
            {
                at + 3
            } else {
                at
            };
            words[at].trim_end_matches('(').to_string()
        };
        match upper.iter().map(String::as_str).take(3).collect::<Vec<_>>()[..] {
            ["CREATE", "TABLE", ..] => undo.push(format!("DROP TABLE IF EXISTS {};", name(2))),
            ["CREATE", "VIRTUAL", "TABLE"] => {
                undo.push(format!("DROP TABLE IF EXISTS {};", name(3)))
            }
            ["CREATE", "INDEX", ..] | ["CREATE", "UNIQUE", "INDEX"] => {
                let at = if upper[1] == "UNIQUE" { 3 } else { 2 };
                undo.push(format!("DROP INDEX IF EXISTS {};", name(at)))
            }
            _ => panic!(
                "{} does more than create tables and indexes; extend this downgrade for it",
                path.display()
            ),
        }
    }
    assert!(!undo.is_empty(), "{} creates nothing", path.display());
    undo.join("\n")
}

#[test]
fn an_older_backup_restores_into_a_new_data_dir_and_migrates_with_a_copy() {
    use asphodel_core::store::SCHEMA_VERSION;

    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .script(&[json!({"reply": auckland_reply()})])
        .ready();
    daemon.create_bank("main");
    daemon.ingest_notes("main", "notes.md");
    let id = daemon.wait_for_memory("main");
    daemon.wait_extracted("main");
    let file = dir.path("backup.db");
    succeeded(run(cli(&daemon).arg("backup").arg("--out").arg(&file)));
    drop(daemon);

    // The copy put back one schema version, as the binary before the latest
    // registered migration would have left a fresh store: without what that
    // migration creates, and with one migration row from 0 to the version
    // before it.
    let older = SCHEMA_VERSION - 1;
    let older_file = dir.path("older.db");
    fs::copy(&file, &older_file).unwrap();
    let conn = rusqlite::Connection::open(&older_file).unwrap();
    let latest: u32 = conn
        .query_row("SELECT MAX(to_version) FROM migrations", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(
        latest, SCHEMA_VERSION,
        "the backup is at the latest registered migration"
    );
    conn.execute_batch(&undo_latest_migration()).unwrap();
    conn.execute(
        "UPDATE migrations SET to_version = ?1 WHERE to_version = ?2",
        (older, SCHEMA_VERSION),
    )
    .unwrap();
    conn.pragma_update(None, "user_version", older).unwrap();
    drop(conn);

    // The data dir doesn't exist yet, nor does its parent.
    let data = dir.path("volume").join("data");
    let output = restore(&older_file, &data);
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(stdout(&output).contains("migrates"), "{}", stdout(&output));

    let mut serve = Serve::new(&dir);
    serve.data_dir = data.clone();
    let mut daemon = serve.ready();
    let recall = daemon.recall("main", "where does Tim live? Auckland");
    assert_eq!(recall["results"][0]["id"], id.as_str(), "{recall}");

    // The migration took a pre-migration copy, which status reports and
    // which needs no attention.
    let status = daemon.ok(daemon.get("/v1/status"));
    let copy = &status["pre_migration_copy"];
    assert_eq!(copy["from_version"], older, "{status}");
    assert_eq!(
        copy["path"],
        json!(data.join(format!("asphodel.db.pre-migration-v{older}"))),
        "{status}"
    );
    assert!(copy["expires_at"].is_string(), "{status}");
    assert_eq!(status["attention"], json!([]), "{status}");
    let out = succeeded(run(cli(&daemon).arg("status")));
    assert!(
        out.contains(&format!("pre-migration copy: from schema version {older}")),
        "{out}"
    );
    daemon.sigterm();
    assert!(daemon.wait_exit().success(), "{}", daemon.log);
    drop(daemon);

    // The `restored` row survived the migration, once.
    let rows = restored_rows(&data);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert_eq!(rows[0]["schema_version"], older);
}
