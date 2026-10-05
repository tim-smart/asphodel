//! The daemon over HTTP, and the CLI as its client, run as processes.
//!
//! The HTTP contract: routes under `/v1`, the bearer token off loopback,
//! `/v1/health` answering 503 until the store and models are ready, SIGTERM
//! finishing the chunks in flight, the data-dir lock and the Unix socket,
//! and every CLI subcommand reaching the daemon over `--url`: models and the
//! system prompt, forget and the purge pause, backup and the offline
//! restore, status and the audit lists, and the dashboard's routes and page.
//!
//! The daemon runs on the fake models (`ASPHODEL_MODELS=fake`) and a
//! scripted fake LLM (`ASPHODEL_LLM_SCRIPT`), both environment only. The
//! script plays one step per LLM call, in order, across the whole daemon,
//! so each test scripts exactly the calls its chunks make.
//! `ASPHODEL_STARTUP_GATE` holds startup after the bind so the 503 can be
//! seen. A TCP daemon gets a port the test found free, so its address is
//! known before it starts, and readiness is `/v1/health` over HTTP, never a
//! log line. The HTTP client here is a few lines over std, so the tests need
//! no new dependencies.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use regex::Regex;
use serde_json::{Value, json};

/// How long a daemon gets to bind, to become ready or to stop, and a
/// condition polled over HTTP gets to hold.
const TIMEOUT: Duration = Duration::from_secs(10);

const TOKEN: &str = "test-bearer-token-5a1d";

/// An id no memory or chunk has.
const UNKNOWN: &str = "01a0f958-0000-7000-8000-000000000000";

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
            "window_confidence": "high",
            "entities": []
        }],
        "used_injected_ids": []
    })
}

/// A call 1 reply with nothing in it.
fn empty_reply() -> Value {
    json!({"claims": [], "used_injected_ids": []})
}

/// A script step answering `reply` after `delay_ms`.
fn step(reply: Value, delay_ms: u64) -> Value {
    json!({"reply": reply, "delay_ms": delay_ms})
}

/// A script step answering [`auckland_reply`] at once.
fn auckland() -> Value {
    step(auckland_reply(), 0)
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

    fn file(&self, name: &str, text: &str) -> PathBuf {
        let path = self.path(name);
        fs::write(&path, text).unwrap();
        path
    }

    /// A tuning file with a floor for each fake model, and `extra` TOML.
    fn floors_for_fakes(&self, extra: &str) -> PathBuf {
        self.file(
            "tuning.toml",
            &format!(
                "[injection.reranker_floors]\n\"fake-reranker:v1\" = 0.0\n\
                 [ranking.relevance_scales]\n\"fake-reranker:v1\" = 1.0\n\
                 [reconcile.embedding_floors]\n\"fake-embedder:v1\" = 0.5\n{extra}"
            ),
        )
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
    /// The body as bytes (the backup stream isn't UTF-8) and as text.
    bytes: Vec<u8>,
    body: String,
}

impl Reply {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body)
            .unwrap_or_else(|error| panic!("not JSON ({error}): {} {}", self.status, self.body))
    }

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

/// One HTTP/1.0 request on its own connection, so the reply is never
/// chunked and ends where the stream does.
fn request(
    addr: &Addr,
    method: &str,
    path: &str,
    token: Option<&str>,
    body: Option<&Value>,
) -> std::io::Result<Reply> {
    let body = body.map(Value::to_string);
    let mut head = format!("{method} {path} HTTP/1.0\r\nHost: asphodel\r\n");
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
    head.push_str(body.as_deref().unwrap_or_default());

    let mut response = Vec::new();
    match addr {
        Addr::Tcp(addr) => {
            let mut stream = TcpStream::connect(addr)?;
            stream.set_read_timeout(Some(TIMEOUT))?;
            stream.write_all(head.as_bytes())?;
            stream.read_to_end(&mut response)?;
        }
        Addr::Unix(path) => {
            let mut stream = UnixStream::connect(path)?;
            stream.set_read_timeout(Some(TIMEOUT))?;
            stream.write_all(head.as_bytes())?;
            stream.read_to_end(&mut response)?;
        }
    }
    // A daemon closing its listener may accept and drop a connection.
    let split = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| std::io::Error::other("no header block"))?;
    let headers = String::from_utf8_lossy(&response[..split]).to_lowercase();
    let status = headers
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("no status line in:\n{headers}"));
    let bytes = response[split + 4..].to_vec();
    Ok(Reply {
        status,
        headers,
        body: String::from_utf8_lossy(&bytes).into_owned(),
        bytes,
    })
}

/// `asphodel` with a clean environment, so the caller's `ASPHODEL_*`
/// variables can't leak in.
fn asphodel() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_asphodel"));
    command.env_clear().stdin(Stdio::null());
    command
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

    fn data_dir(mut self, data_dir: &Path) -> Self {
        self.data_dir = data_dir.to_owned();
        self
    }

    /// Port 0 is swapped for a port free now, so the address is known up
    /// front.
    fn listen(mut self, listen: &str) -> Self {
        self.listen = match listen.strip_suffix(":0") {
            Some(host) => {
                let free = TcpListener::bind((host, 0)).unwrap().local_addr().unwrap();
                format!("{host}:{}", free.port())
            }
            None => listen.to_string(),
        };
        self
    }

    fn socket(self, socket: &Path) -> Self {
        self.listen(&format!("unix:{}", socket.display()))
    }

    fn script(mut self, steps: &[Value]) -> Self {
        let steps = serde_json::to_string(steps).unwrap();
        self.script = Some(self.dir.file("script.json", &steps));
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
        let mut command = asphodel();
        command
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

    /// Starts the daemon and returns once `/v1/health` answers at all.
    fn bind(self) -> Daemon {
        let mut child = self
            .command()
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
        let addr = match self.listen.strip_prefix("unix:") {
            Some(path) => Addr::Unix(PathBuf::from(path)),
            // A daemon bound to every interface is reached on loopback.
            None => Addr::Tcp(self.listen.replace("0.0.0.0", "127.0.0.1")),
        };
        let mut daemon = Daemon {
            child,
            addr,
            token: self.token.map(str::to_string),
            data_dir: self.data_dir.clone(),
            log: String::new(),
            lines: received,
        };
        daemon.wait_health("bound", |status| status.is_some());
        daemon
    }

    /// Starts the daemon and waits until `/v1/health` answers 200.
    fn ready(self) -> Daemon {
        let mut daemon = self.bind();
        daemon.wait_ready();
        daemon
    }
}

/// Waits for `child` to exit, and kills it at the deadline.
fn wait_or_kill(child: &mut Child) -> Option<ExitStatus> {
    let deadline = Instant::now() + TIMEOUT;
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().unwrap() {
            return Some(status);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let _ = child.kill();
    let _ = child.wait();
    None
}

/// Runs a daemon that should refuse and exit on its own, and kills it at
/// the deadline, so one that wrongly starts can't hang the suite.
fn exits(mut command: Command) -> ExitStatus {
    let mut child = command
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_or_kill(&mut child).expect("the daemon kept running")
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

    fn get_ok(&self, path: &str) -> Value {
        self.ok(self.get(path))
    }

    fn post_ok(&self, path: &str, body: &Value) -> Value {
        self.ok(self.post(path, body))
    }

    fn wait_ready(&mut self) {
        self.wait_health("ready", |status| status == Some(200));
    }

    /// Polls `/v1/health` until `holds` accepts its status, `None` when
    /// nothing answered.
    fn wait_health(&mut self, what: &str, holds: impl Fn(Option<u16>) -> bool) {
        let deadline = Instant::now() + TIMEOUT;
        loop {
            let health = request(&self.addr, "GET", "/v1/health", None, None)
                .map(|reply| (reply.status, reply.body));
            if holds(health.as_ref().ok().map(|(status, _)| *status)) {
                return;
            }
            if Instant::now() >= deadline {
                self.drain();
                panic!("never {what}; last health was {health:?}\n{}", self.log);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Polls `read` until `holds` accepts its value, and returns that value.
    fn wait_until(
        &mut self,
        what: &str,
        read: impl Fn(&Self) -> Value,
        holds: impl Fn(&Value) -> bool,
    ) -> Value {
        let deadline = Instant::now() + TIMEOUT;
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
        let reply = self.send(
            "PUT",
            &format!("/v1/banks/{bank}"),
            Some(&json!({"owner_name": "Tim", "assistant_name": "Ash", "timezone": "Pacific/Auckland"})),
        );
        assert_eq!(reply.status, 201, "{}", reply.body);
    }

    /// Ingests [`NOTES`] as `notes.md` in bank `main`.
    fn ingest_notes(&self) -> Value {
        self.ingest_document("main", "notes.md", NOTES)
    }

    fn ingest_document(&self, bank: &str, id: &str, text: &str) -> Value {
        self.post_ok(
            &format!("/v1/banks/{bank}/documents"),
            &json!({
                "document_id": id,
                "text": text,
                "reference_date": "2026-09-30",
                "reference_date_exact": true,
                "timezone": null
            }),
        )
    }

    fn recall(&self, bank: &str, query: &str) -> Value {
        self.post_ok(
            &format!("/v1/banks/{bank}/recall"),
            &json!({"query": query}),
        )
    }

    fn chunks(&self, bank: &str) -> Value {
        self.get_ok(&format!("/v1/banks/{bank}/chunks"))
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

    /// Creates bank `main`, ingests [`NOTES`] and waits until its memory is
    /// extracted and the queue is empty. Returns the memory's id.
    fn seed_notes(&mut self) -> String {
        self.create_bank("main");
        self.ingest_notes();
        let id = self.wait_for_memory("main");
        self.wait_extracted("main");
        id
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
        let status = wait_or_kill(&mut self.child);
        self.drain();
        status.unwrap_or_else(|| panic!("the daemon did not stop in time:\n{}", self.log))
    }

    /// SIGTERM, and a clean exit.
    fn stop(mut self) {
        self.sigterm();
        let status = self.wait_exit();
        assert!(status.success(), "{status}\n{}", self.log);
    }
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
    assert_eq!(health.json()["ready"], false);
    assert_eq!(health.json()["version"], env!("CARGO_PKG_VERSION"));
    assert!(
        !daemon.data_dir.join("asphodel.db").exists(),
        "the store was opened before the daemon was ready"
    );

    // Every other route refuses rather than racing the startup.
    for (method, path, body) in [
        ("GET", "/v1/config", None),
        ("GET", "/v1/banks/main/chunks", None),
        ("PUT", "/v1/banks/main", Some(json!({}))),
    ] {
        let reply = daemon.send(method, path, body.as_ref());
        assert_eq!(reply.status, 503, "{method} {path}: {}", reply.body);
        assert!(reply.json()["error"].is_string());
    }

    fs::write(&gate, "").unwrap();
    daemon.wait_ready();
    let health = daemon.get_ok("/v1/health");
    assert_eq!(health["ready"], true);
    assert_eq!(health["version"], env!("CARGO_PKG_VERSION"));
    assert!(daemon.data_dir.join("asphodel.db").exists());
}

#[test]
fn a_second_daemon_on_a_locked_data_dir_exits_and_removes_its_socket() {
    // The second daemon binds its own socket, answers 503, then fails to
    // open the store. It must exit non-zero, leave no socket behind, and
    // leave the first daemon's lock and files as they were.
    let dir = TestDir::new();
    let first = Serve::new(&dir).ready();
    let before = names(&first.data_dir);
    let gate = dir.path("gate");
    let socket = dir.path("second.sock");
    let mut second = Serve::new(&dir).socket(&socket).gate(&gate).bind();
    assert_eq!(second.get("/v1/health").status, 503);
    fs::write(&gate, "").unwrap();
    assert!(!second.wait_exit().success(), "{}", second.log);
    assert!(!socket.exists(), "the failed daemon left {socket:?}");

    // Refusing didn't release the lock: a third daemon is refused too.
    let third = Serve::new(&dir).socket(&dir.path("third.sock"));
    assert!(!exits(third.command()).success());
    assert_eq!(names(&first.data_dir), before);
    assert_eq!(first.get("/v1/health").status, 200);
}

/// The device and inode at `path`, not following a symlink.
fn identity(path: &Path) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::symlink_metadata(path).unwrap();
    (metadata.dev(), metadata.ino())
}

#[test]
fn serve_refuses_a_socket_path_or_data_dir_it_does_not_own_and_leaves_it_be() {
    // A regular file, a symlink to a live socket and a live listener at the
    // socket path, and a regular file as the data dir: each refused, and
    // each still there, untouched, afterwards.
    let dir = TestDir::new();
    let file = dir.file("file", "irreplaceable contents");
    let live = dir.path("live.sock");
    let _listener = UnixListener::bind(&live).unwrap();
    let link = dir.path("link.sock");
    std::os::unix::fs::symlink(&live, &link).unwrap();
    let before = [&file, &live, &link].map(|path| identity(path));

    for (socket, data_dir) in [
        (&file, dir.data_dir()),
        (&link, dir.data_dir()),
        (&live, dir.data_dir()),
        (&dir.path("d.sock"), file.clone()),
    ] {
        let serve = Serve::new(&dir).socket(socket).data_dir(&data_dir);
        assert!(!exits(serve.command()).success(), "{socket:?} {data_dir:?}");
    }
    assert_eq!([&file, &live, &link].map(|path| identity(path)), before);
    assert_eq!(fs::read(&file).unwrap(), b"irreplaceable contents");
    assert_eq!(fs::read_link(&link).unwrap(), live);
    UnixStream::connect(&link).expect("the live socket still accepts");
}

#[test]
fn a_stale_socket_is_reused_and_only_the_daemons_own_socket_is_removed() {
    // A socket nobody listens on is recovered. At exit the daemon removes
    // the socket it bound, but not one that replaced it meanwhile.
    let dir = TestDir::new();
    let socket = dir.path("d.sock");
    drop(UnixListener::bind(&socket).unwrap());
    Serve::new(&dir).ready().stop();
    assert!(!socket.exists(), "the daemon left its socket");

    let daemon = Serve::new(&dir).ready();
    fs::remove_file(&socket).unwrap();
    let _replacement = UnixListener::bind(&socket).unwrap();
    daemon.stop();
    UnixStream::connect(&socket).expect("the replacement still accepts");
}

// The bearer token.

#[test]
fn off_loopback_every_route_but_health_needs_the_bearer_token() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir).listen("0.0.0.0:0").token(TOKEN).ready();
    let addr = daemon.addr.clone();

    // Health stays open, so a readiness probe needs no secret, and so does
    // the dashboard page, which holds nothing.
    let health = request(&addr, "GET", "/v1/health", None, None).unwrap();
    assert_eq!(health.status, 200, "{}", health.body);
    let page = request(&addr, "GET", "/dashboard", None, None).unwrap();
    assert_eq!(page.status, 200, "{}", page.body);
    assert!(page.headers.contains("content-type: text/html"));
    assert!(!page.body.contains(TOKEN));

    let source = format!("/v1/banks/main/sources/{UNKNOWN}");
    let retract = format!("/v1/banks/main/memories/{UNKNOWN}/retract");
    let (empty, query) = (json!({}), json!({"query": "anything"}));
    for token in [None, Some("wrong-token"), Some("")] {
        for (method, path, body) in [
            ("GET", "/v1/config", None),
            ("PUT", "/v1/banks/main", Some(&empty)),
            ("GET", "/v1/banks/main/chunks", None),
            ("POST", "/v1/banks/main/recall", Some(&query)),
            ("POST", "/v1/backup", None),
            ("GET", "/v1/status", None),
            ("GET", "/v1/banks/main/purges", None),
            ("GET", "/v1/banks/main/forgets", None),
            ("GET", "/v1/banks/main/sweeps", None),
            ("GET", "/v1/banks/main/recalls", None),
            ("GET", "/v1/banks", None),
            ("GET", "/v1/banks/main/memories", None),
            ("GET", "/v1/banks/main/sources", None),
            ("GET", source.as_str(), None),
            ("POST", retract.as_str(), None),
            ("POST", "/v1/banks/main/documents/remove", None),
            ("POST", "/v1/banks/main/recall/explain", Some(&query)),
        ] {
            let reply = request(&addr, method, path, token, body).unwrap();
            assert_eq!(reply.status, 401, "{method} {path} with {token:?}");
            assert!(
                reply.headers.contains("www-authenticate: bearer"),
                "{}",
                reply.headers
            );
            assert!(reply.json()["error"].is_string());
        }
    }

    let reply = daemon.send("PUT", "/v1/banks/main", Some(&empty));
    assert_eq!(reply.status, 201, "{}", reply.body);
    let config = daemon.get("/v1/config");
    assert_eq!(config.status, 200, "{}", config.body);
    assert_eq!(config.json()["deployment"]["token"], "[redacted]");
    assert!(!config.body.contains(TOKEN));

    daemon.sigterm();
    daemon.wait_exit();
    assert!(!daemon.log.contains(TOKEN), "{}", daemon.log);
}

// The routes, driven with the fake models and LLM.

#[test]
fn a_turn_commits_its_prefetch_and_clearing_the_session_forgets_it() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .script(&[auckland(), step(empty_reply(), 0)])
        .ready();
    let id = daemon.seed_notes();

    let prefetch = |daemon: &Daemon| {
        daemon.post_ok(
            "/v1/banks/main/prefetch",
            &json!({"session_id": "s1", "query": "where do I live, Auckland?"}),
        )
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
    let ingested = daemon.post_ok("/v1/banks/main/turns", &turn);
    assert_eq!(ingested["outcome"], "stored");
    assert_eq!(ingested["speaker"]["owner"], true);
    assert_eq!(prefetch(&daemon)["injected"], json!([]));

    // A resend from the plugin's spool is a duplicate.
    let resent = daemon.post_ok("/v1/banks/main/turns", &turn);
    assert_eq!(resent["outcome"], "duplicate");
    daemon.wait_extracted("main");

    let cleared = daemon.send("POST", "/v1/banks/main/sessions/s1/clear", None);
    assert_eq!(cleared.status, 204, "{}", cleared.body);
    assert_eq!(prefetch(&daemon)["injected"], json!([id]));
}

/// The prefetch route takes the assistant's last reply as `previous_reply`,
/// which reranking against the conversation reads. A request without it
/// still works, and reranks against the previous message and the message.
#[test]
fn prefetch_takes_the_previous_reply_for_the_conversation_query() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .tuning("[injection]\nrerank_query = \"conversation\"\n")
        .script(&[auckland()])
        .ready();
    let id = daemon.seed_notes();

    let message = "great, should I bring a jacket when I go out later";
    let asked = "Can you check the weather for me?";
    let without = daemon.post_ok(
        "/v1/banks/main/prefetch",
        &json!({"session_id": "s1", "query": message, "previous_query": asked}),
    );
    assert_eq!(without["injected"], json!([]), "{without}");

    let with = daemon.post_ok(
        "/v1/banks/main/prefetch",
        &json!({
            "session_id": "s2",
            "query": message,
            "previous_query": asked,
            "previous_reply": "Auckland is sunny today.",
        }),
    );
    assert_eq!(with["injected"], json!([id]), "{with}");
}

#[test]
fn errors_are_json_with_the_right_status() {
    let dir = TestDir::new();
    let daemon = Serve::new(&dir).ready();
    daemon.create_bank("main");

    let (query, unknown_field) = (json!({"query": "x"}), json!({"nope": 1}));
    let backwards =
        json!({"query": "x", "from": "2026-10-02T00:00:00Z", "to": "2026-10-01T00:00:00Z"});
    let refresh = "/v1/banks/main/models/User%20profile/refresh";
    for (method, path, body, status) in [
        ("POST", "/v1/banks/nope/recall", Some(&query), 404),
        ("POST", "/v1/banks/main/recall", Some(&unknown_field), 422),
        ("POST", "/v1/banks/main/recall", Some(&backwards), 400),
        ("GET", "/v1/nowhere", None, 404),
        // No LLM is configured, so nothing can be refreshed.
        ("POST", refresh, Some(&Value::Null), 503),
    ] {
        let reply = daemon.send(method, path, body);
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
fn sigterm_refuses_ingest_and_finishes_the_chunk_in_flight() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .listen("127.0.0.1:0")
        .script(&[step(auckland_reply(), 3000)])
        .ready();
    daemon.create_bank("main");
    daemon.ingest_notes();
    daemon.wait_until(
        "a chunk in flight",
        |daemon| daemon.chunks("main"),
        |chunks| chunks["queued"][0]["in_flight"] == true,
    );

    daemon.sigterm();
    daemon.wait_health("draining", |status| status != Some(200));
    // The listener is closing: new ingest is refused or never connects.
    let late = json!({
        "document_id": "late.md",
        "text": "Late text.",
        "reference_date": "2026-09-30",
        "reference_date_exact": true
    });
    let path = "/v1/banks/main/documents";
    if let Ok(reply) = request(&daemon.addr, "POST", path, None, Some(&late)) {
        assert!(
            !(200..300).contains(&reply.status),
            "late ingest was accepted"
        );
    }

    let status = daemon.wait_exit();
    assert!(status.success(), "{status}\n{}", daemon.log);
    drop(daemon);

    // The chunk was committed, not abandoned: a restart has nothing queued
    // and recalls the memory, and the late document was never stored.
    let mut restarted = Serve::new(&dir).listen("127.0.0.1:0").ready();
    restarted.wait_for_memory("main");
    let chunks = restarted.chunks("main");
    assert_eq!(chunks["queued"], json!([]));
    assert_eq!(chunks["failed"], json!([]));
    let late = restarted.post_ok(path, &late);
    assert_eq!(
        late["outcome"], "stored",
        "late.md was stored during the drain"
    );
}

// Concurrent extraction: a pool of leases per bank behind `[llm] concurrency`.

/// `[llm] concurrency` set to `n`.
fn pool(n: u32) -> String {
    format!("[llm]\nconcurrency = {n}\n")
}

/// Queues `documents` in bank `main` on a daemon with no LLM, then stops
/// it, so the daemon a test starts next finds the whole queue at once.
fn queue_without_an_llm(dir: &TestDir, tuning: &str, documents: &[(String, String)]) {
    let daemon = Serve::new(dir).tuning(tuning).ready();
    daemon.create_bank("main");
    for (id, text) in documents {
        daemon.ingest_document("main", id, text);
    }
    assert_eq!(
        daemon.chunks("main")["queued"].as_array().unwrap().len(),
        documents.len()
    );
    daemon.stop();
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

/// A daemon on a pool of two holding two of three queued chunks in flight,
/// each call taking `delay_ms` and finding nothing.
fn two_of_three_in_flight(dir: &TestDir, delay_ms: u64) -> Daemon {
    queue_without_an_llm(dir, &pool(2), &distinct_documents(3));
    let steps = vec![step(empty_reply(), delay_ms); 3];
    let mut daemon = Serve::new(dir).tuning(&pool(2)).script(&steps).ready();
    daemon.wait_until(
        "two chunks in flight",
        |daemon| daemon.chunks("main"),
        |chunks| {
            let queued = chunks["queued"].as_array().unwrap();
            queued.iter().filter(|c| c["in_flight"] == true).count() == 2
        },
    );
    daemon
}

/// The kinds of `memory`'s accesses, oldest first.
fn access_kinds(memory: &Value) -> Vec<&str> {
    let accesses = memory["accesses"].as_array().unwrap();
    accesses
        .iter()
        .map(|a| a["kind"].as_str().unwrap())
        .collect()
}

/// An OpenAI-compatible endpoint on loopback whose every call finds nothing
/// after `hold`. Returns its `[llm] endpoint` and the most calls it ever had
/// open at once.
fn holding_llm(hold: Duration) -> (String, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    let (open, most) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let content = empty_reply().to_string();
    let body = json!({"choices": [{"message": {"content": content}}]}).to_string();
    let reply = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
         content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    let counted = Arc::clone(&most);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            let (open, most, reply) = (Arc::clone(&open), Arc::clone(&counted), reply.clone());
            std::thread::spawn(move || {
                let mut stream = read_request(stream);
                most.fetch_max(open.fetch_add(1, Ordering::SeqCst) + 1, Ordering::SeqCst);
                std::thread::sleep(hold);
                // Closed before the client sees the reply, so a call the gate
                // lets in next is never counted beside this one.
                open.fetch_sub(1, Ordering::SeqCst);
                let _ = stream.write_all(reply.as_bytes());
            });
        }
    });
    (endpoint, most)
}

#[test]
fn a_banks_leases_call_together_and_llm_concurrency_caps_calls_across_banks() {
    // Two banks on a pool of three each hold six leases out at once, but
    // the gate lets three calls reach the LLM: three open together, so one
    // bank ran two at once, and never four.
    let (endpoint, most) = holding_llm(Duration::from_millis(1000));
    let tuning = format!("{}endpoint = \"{endpoint}\"\nmodel = \"stub\"\n", pool(3));
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir).tuning(&tuning).ready();
    let banks = ["main", "other"];
    for bank in banks {
        daemon.create_bank(bank);
        for (id, text) in distinct_documents(3) {
            daemon.ingest_document(bank, &id, &text);
        }
    }
    daemon.wait_until(
        "six leases out",
        |daemon| {
            let leases = banks.map(|bank| {
                let chunks = daemon.chunks(bank);
                let queued = chunks["queued"].as_array().unwrap();
                queued.iter().filter(|c| c["in_flight"] == true).count()
            });
            json!(leases)
        },
        |leases| leases == &json!([3, 3]),
    );
    for bank in banks {
        daemon.wait_extracted(bank);
    }
    assert_eq!(most.load(Ordering::SeqCst), 3);
}

#[test]
fn two_chunks_in_flight_stating_one_fact_make_one_memory_and_a_mention() {
    // Both chunks run call 1 before either commits, and neither finds a
    // neighbour. The second to commit finds a memory made since its search
    // at or above the floor for its claim, so it searches again and runs
    // call 2, which labels the claim a mention: one memory plus one access,
    // never a second copy. Each call 1 takes 3 s, so both
    // finishing within 5 s of the start means they ran together.
    let dir = TestDir::new();
    let tuning = pool(2);
    let more = "# More notes\n\nI live in Auckland, near the harbour.\n";
    let documents = [("notes.md", NOTES), ("more-notes.md", more)];
    let documents = documents.map(|(id, text)| (id.to_string(), text.to_string()));
    queue_without_an_llm(&dir, &tuning, &documents);

    let mention = json!({"claims": [{
        "claim": "c1",
        "labels": [{"neighbour": "n1", "label": "mentioned_again"}]
    }]});
    let started = Instant::now();
    let mut daemon = Serve::new(&dir)
        .tuning(&tuning)
        .script(&[
            step(auckland_reply(), 3000),
            step(auckland_reply(), 3000),
            step(mention, 0),
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
    let results = recall["results"].as_array().unwrap();
    let ids: Vec<_> = results
        .iter()
        .filter(|r| r["sentence"] == SENTENCE)
        .collect();
    assert_eq!(ids.len(), 1, "one memory, not a copy per chunk: {recall}");
    let id = ids[0]["id"].as_str().unwrap();
    let memory = daemon.get_ok(&format!("/v1/banks/main/memories/{id}"));
    assert_eq!(
        access_kinds(&memory),
        ["created", "mentioned_again"],
        "{memory}"
    );
}

#[test]
fn sigterm_finishes_every_chunk_in_flight_before_stopping() {
    let dir = TestDir::new();
    two_of_three_in_flight(&dir, 2000).stop();

    // Both chunks in flight committed: only the one never claimed is still
    // queued, with nothing counted.
    let restarted = Serve::new(&dir).tuning(&pool(2)).ready();
    let chunks = restarted.chunks("main");
    let queued = chunks["queued"].as_array().unwrap();
    assert_eq!(queued.len(), 1, "{chunks}");
    assert_eq!(queued[0]["error_count"], 0, "{chunks}");
    assert_eq!(chunks["failed"], json!([]), "{chunks}");
}

#[test]
fn deleting_a_bank_waits_for_its_chunks_in_flight_and_stops_its_worker() {
    let dir = TestDir::new();
    let mut daemon = two_of_three_in_flight(&dir, 1500);

    let deleted = daemon.send("DELETE", "/v1/banks/main?confirm=main", None);
    assert_eq!(deleted.status, 200, "{}\n{}", deleted.body, daemon.log);
    assert_eq!(daemon.get("/v1/banks/main/chunks").status, 404);

    // The hold drained the pool without handing out the third chunk, and
    // the old worker stopped with its bank, so the bank made again under
    // the name gets a new one, which takes the script's last step: had the
    // third chunk taken it, this one would fail.
    daemon.create_bank("main");
    daemon.ingest_document("main", "again.md", "# Again\n\nA new start.\n");
    daemon.wait_extracted("main");
}

#[test]
fn a_usage_limit_or_a_429_pauses_every_caller_and_counts_no_failure() {
    // Eight chunks on a pool of two, where the first LLM call hits the
    // limit, a hold of about six seconds. Every call after it waits for the
    // hold to end, so between the calls already in flight finishing and the
    // hold ending nothing is extracted, and no chunk counts a failure.
    use asphodel_core::{Clock, SystemClock};
    // The daemon runs on the system clock, so the reset is in its time.
    let resets_at = SystemClock.now() + jiff::SignedDuration::from_secs(6);
    for limit in [
        json!({"fail": "usage_limited", "resets_at": resets_at.to_string()}),
        json!({"fail": "status", "status": 429, "retry_after_secs": 6}),
    ] {
        let dir = TestDir::new();
        let tuning = pool(2);
        queue_without_an_llm(&dir, &tuning, &distinct_documents(8));

        // One call may have started before the limit came back; the rest
        // run after the hold. 8 more replies: every chunk, the limited one
        // again.
        let mut steps = vec![limit, step(empty_reply(), 1000)];
        steps.extend(vec![step(empty_reply(), 500); 7]);
        let mut daemon = Serve::new(&dir).tuning(&tuning).script(&steps).ready();

        let snapshot = |daemon: &Daemon| {
            let chunks = daemon.chunks("main");
            let queued = chunks["queued"].as_array().unwrap();
            let counted = queued.iter().any(|chunk| chunk["error_count"] != 0);
            assert!(!counted, "a hold isn't the chunk's failure: {chunks}");
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
}

// The CLI as a client. Every operator command is an HTTP client of the daemon.

/// `asphodel` with `ASPHODEL_URL` set to `url`.
fn cli_at(url: &str) -> Command {
    let mut command = asphodel();
    command.env("ASPHODEL_URL", url);
    command
}

/// [`cli_at`] `daemon`, without its token.
fn cli(daemon: &Daemon) -> Command {
    cli_at(&daemon.addr.url())
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

/// The JSON a successful `--json` command printed.
fn json_out(output: Output) -> Value {
    serde_json::from_str(&succeeded(output)).expect("--json prints JSON")
}

/// `asphodel <args> --json` against `daemon`, which must succeed. `args`
/// are split on whitespace.
fn cli_json(daemon: &Daemon, args: &str) -> Value {
    json_out(run(cli(daemon).args(args.split_whitespace()).arg("--json")))
}

/// [`cli_json`] for a command that must fail, and still print JSON.
fn cli_json_fails(daemon: &Daemon, args: &str) -> Value {
    let output = run(cli(daemon).args(args.split_whitespace()).arg("--json"));
    assert!(!output.status.success(), "{}", stdout(&output));
    serde_json::from_str(&stdout(&output)).unwrap()
}

#[test]
fn the_cli_drives_the_daemon_over_a_unix_socket() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir).script(&[auckland()]).ready();
    let notes = dir.file("notes.md", NOTES);

    let bank = cli_json(
        &daemon,
        "bank create main --owner-name Tim --timezone Pacific/Auckland",
    );
    assert_eq!(bank["created"], true, "{bank}");
    let bank = cli_json(&daemon, "bank config main --assistant-name Ash");
    assert_eq!(bank["created"], false);
    assert_eq!(bank["owner_name"], "Tim");
    assert_eq!(bank["assistant_name"], "Ash");

    // The document id defaults to the file's name.
    let ingest = ["--bank", "main", "--date", "2026-09-30", "--json"];
    let ingested = json_out(run(cli(&daemon).arg("ingest").arg(&notes).args(ingest)));
    assert_eq!(ingested["chunks_queued"], 1, "{ingested}");
    let sources = daemon.get_ok("/v1/banks/main/sources?kind=document");
    assert_eq!(
        sources["sources"][0]["document_id"], "notes.md",
        "{sources}"
    );
    let id = daemon.wait_for_memory("main");

    let recall = cli_json(&daemon, "recall --bank main --kind fact Auckland");
    assert_eq!(recall["results"][0]["id"], id.as_str());
    assert_eq!(recall["results"][0]["sentence"], SENTENCE);
    let recall = cli_json(&daemon, "recall --bank main --kind event Auckland");
    assert_eq!(recall["results"], json!([]), "{recall}");

    let kept = cli_json(&daemon, &format!("keep --bank main {id}"));
    assert_eq!(kept["kept"], json!([id]));
    let recalled = daemon.recall("main", "Auckland");
    assert_eq!(recalled["results"][0]["kept"], true);
    let unkept = cli_json(&daemon, &format!("unkeep --bank main {id}"));
    assert_eq!(unkept["unkept"], json!([id]));
    let recalled = daemon.recall("main", "Auckland");
    assert_eq!(recalled["results"][0]["kept"], false);

    // An unknown id fails the command, after the rest are done.
    let kept = cli_json_fails(&daemon, &format!("keep --bank main {id} {UNKNOWN}"));
    assert_eq!(kept["kept"], json!([id]), "{kept}");
    assert_eq!(kept["unknown"], json!([UNKNOWN]), "{kept}");

    let chunks = cli_json(&daemon, "chunks --bank main");
    assert_eq!(chunks["queued"], json!([]), "{chunks}");
    assert_eq!(chunks["failed"], json!([]), "{chunks}");
}

#[test]
fn the_cli_reaches_a_tcp_daemon_with_the_token_from_the_environment() {
    // A token set on loopback is checked too.
    let dir = TestDir::new();
    let daemon = Serve::new(&dir).listen("127.0.0.1:0").token(TOKEN).ready();
    daemon.create_bank("main");
    let chunks = ["chunks", "--bank", "main", "--json"];

    assert!(!run(cli(&daemon).args(chunks)).status.success(), "no token");
    let listed = json_out(run(cli(&daemon).env("ASPHODEL_TOKEN", TOKEN).args(chunks)));
    assert_eq!(listed["queued"], json!([]), "{listed}");

    // --url wins over ASPHODEL_URL.
    succeeded(run(cli(&daemon)
        .env("ASPHODEL_URL", "unix:/nonexistent/asphodel.sock")
        .env("ASPHODEL_TOKEN", TOKEN)
        .args(chunks)
        .args(["--url", &daemon.addr.url()])));

    // The token has no flag, so it never shows in a process list.
    let output = run(cli(&daemon).args(chunks).args(["--token", TOKEN]));
    assert!(!output.status.success(), "--token was accepted");
}

// Mental models and the system prompt block.

const ENTRY: &str = "Tim's home is in Auckland.";
const SECOND: &str = "Tim's home is in New Zealand.";

/// Whether the line after `heading`'s own line in `text` is the section's
/// paragraph: [`ENTRY`] then [`SECOND`]. A heading may be marked up, as
/// `### Home`.
fn paragraph_after(text: &str, heading: &str) -> bool {
    let mut lines = text.lines();
    lines
        .by_ref()
        .find(|line| line.trim().trim_start_matches('#').trim() == heading);
    let paragraph = lines.next().unwrap_or_default();
    paragraph
        .find(ENTRY)
        .zip(paragraph.find(SECOND))
        .is_some_and(|(first, second)| first < second)
}

#[test]
fn models_are_created_listed_edited_and_refreshed_over_http() {
    // The refresh writes two sentences under "Home", citing the first
    // memory in its input.
    let writes = json!({
        "sections": [{"heading": "Home", "text": format!("{ENTRY} {SECOND}")}],
        "cites": ["m1"],
    });
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .script(&[auckland(), step(writes, 0)])
        .ready();
    let memory = daemon.seed_notes();

    // Only "User profile" is seeded, empty and never refreshed.
    let models = daemon.get_ok("/v1/banks/main/models")["models"].clone();
    assert_eq!(models.as_array().unwrap().len(), 1);
    assert_eq!(models[0]["name"], "User profile");
    assert_eq!(models[0]["answer"], Value::Null);
    assert_eq!(models[0]["cites"], json!([]));
    assert_eq!(models[0]["last_refreshed_at"], Value::Null);

    // Plans takes what the profile leaves of the budget, so nothing more fits.
    let budget = &daemon.get_ok("/v1/config")["tuning"]["mental_models"]["budget"];
    let left = budget.as_u64().unwrap() - models[0]["max_tokens"].as_u64().unwrap();
    let create = |name: &str, question: &str, max_tokens: u64| {
        let model = json!({"name": name, "question": question, "max_tokens": max_tokens});
        daemon.post("/v1/banks/main/models", &model)
    };
    let created = daemon.post(
        "/v1/banks/main/models",
        &json!({"name": "Plans", "question": "Where is Tim going?", "kinds": ["event"],
                "max_tokens": left}),
    );
    assert_eq!(created.status, 201, "{}", created.body);
    let plans = created.json();
    assert_eq!(plans["enabled"], true);
    assert_eq!(plans["kinds"], json!(["event"]));
    let over = create("Big", "Anything?", 1);
    assert_eq!(over.status, 422, "{}", over.body);
    let duplicate = create("Plans", "Again?", 1);
    assert_eq!(duplicate.status, 409, "{}", duplicate.body);

    let patch =
        |edit: Value| daemon.ok(daemon.send("PATCH", "/v1/banks/main/models/Plans", Some(&edit)));
    let edited = patch(json!({"enabled": false, "min_volatility": "weeks"}));
    assert_eq!(edited["enabled"], false);
    assert_eq!(edited["min_volatility"], "weeks");
    assert_eq!(edited["question"], "Where is Tim going?");
    let cleared = patch(json!({"min_volatility": null}));
    assert_eq!(cleared["min_volatility"], Value::Null);

    // A forced refresh calls the LLM; the next one finds nothing changed.
    let refresh = "/v1/banks/main/models/User%20profile/refresh";
    let refreshed = daemon.post_ok(&format!("{refresh}?force=true"), &Value::Null);
    assert_eq!(refreshed["outcome"], "applied", "{refreshed}");
    assert_eq!(refreshed["detail"]["written"], true, "{refreshed}");
    let unchanged = daemon.post_ok(refresh, &Value::Null);
    assert_eq!(unchanged["outcome"], "unchanged", "{unchanged}");

    let profile = &daemon.get_ok("/v1/banks/main/models")["models"][0];
    let answer = format!("### Home\n{ENTRY} {SECOND}");
    assert_eq!(profile["answer"], answer.as_str(), "{profile}");
    assert_eq!(profile["cites"], json!([memory]));
    assert!(profile["last_refreshed_at"].is_string());
    let shown = daemon.get_ok("/v1/banks/main/models/User%20profile");
    assert_eq!(shown["answer"], answer.as_str(), "{shown}");

    // `model list` shows the section's paragraph under its heading,
    // without the cited memories. `model show` adds the memories
    // the answer cites.
    let model = ["model", "show", "--bank", "main", "User profile"];
    let list = ["model", "list", "--bank", "main"];
    for args in [&model[..], &list[..]] {
        let text = succeeded(run(cli(&daemon).args(args)));
        assert!(paragraph_after(&text, "Home"), "{args:?}:\n{text}");
    }
    let listed = succeeded(run(cli(&daemon).args(list)));
    assert!(!listed.contains(memory.as_str()), "{listed}");
    let detail = succeeded(run(cli(&daemon).args(model)));
    assert!(detail.contains(memory.as_str()), "{detail}");

    // The block shows the answer, and a session's fetch puts the cited
    // memory in context, so prefetch doesn't inject it.
    let block = daemon.get_ok("/v1/banks/main/system-prompt?session_id=s1");
    let text = block["text"].as_str().unwrap();
    assert!(text.contains("User profile"), "{text}");
    let (_, output) = text.split_once("\nOutput:\n").expect("model output");
    assert_eq!(output.trim(), format!("{ENTRY} {SECOND}"), "{text}");
    assert!(!text.contains("Plans"), "a disabled model was rendered");
    assert_eq!(block["cited"], json!([memory]));
    assert_eq!(
        daemon.get_ok("/v1/banks/main/system-prompt")["id"],
        block["id"],
        "the cached block was rebuilt"
    );
    let prefetch = |session: &str| {
        let query = json!({"session_id": session, "query": "where does Tim live? Auckland"});
        daemon.post_ok("/v1/banks/main/prefetch", &query)
    };
    let fetched = prefetch("s1");
    let injected = fetched["injected"].as_array().unwrap();
    assert!(!injected.contains(&json!(memory)), "{fetched}");
    assert_eq!(prefetch("s2")["injected"], json!([memory]));

    let agenda = daemon.get_ok("/v1/banks/main/agenda");
    assert_eq!(
        agenda,
        json!({"dated": [], "folded": 0, "routines": [], "undated_tasks": []})
    );
}

/// The dashboard shows the enabled models' `max_tokens` against the budget,
/// so the list carries the budget beside the models.
#[test]
fn the_model_list_carries_the_budget_and_enabling_past_it_changes_nothing() {
    let dir = TestDir::new();
    let daemon = Serve::new(&dir)
        .tuning("[mental_models]\nbudget = 800\nprofile_max_tokens = 500\n")
        .ready();
    daemon.create_bank("main");
    // The profile takes 500 of the 800 tokens; a disabled model doesn't count.
    let created = daemon.post(
        "/v1/banks/main/models",
        &json!({"name": "Plans", "question": "Where is Tim going?", "max_tokens": 301,
                "enabled": false}),
    );
    assert_eq!(created.status, 201, "{}", created.body);

    let listed = daemon.get_ok("/v1/banks/main/models");
    assert_eq!(listed["budget"], 800, "{listed}");
    let enabled = |listed: &Value| -> Vec<(String, bool)> {
        listed["models"]
            .as_array()
            .unwrap_or_else(|| panic!("no models array: {listed}"))
            .iter()
            .map(|m| (m["name"].as_str().unwrap().to_owned(), m["enabled"] == true))
            .collect()
    };
    let before = vec![
        ("User profile".to_owned(), true),
        ("Plans".to_owned(), false),
    ];
    assert_eq!(enabled(&listed), before);

    let refused = daemon.send(
        "PATCH",
        "/v1/banks/main/models/Plans",
        Some(&json!({"enabled": true})),
    );
    assert_eq!(refused.status, 422, "{}", refused.body);
    assert_eq!(
        refused.json()["error"],
        "801 tokens is over the 800-token budget for mental models"
    );
    assert_eq!(enabled(&daemon.get_ok("/v1/banks/main/models")), before);
}

/// The dashboard shows the block the daemon has cached without building
/// one, so looking writes nothing.
#[test]
fn the_cached_system_prompt_is_null_until_a_fetch_builds_it() {
    let dir = TestDir::new();
    let daemon = Serve::new(&dir).ready();
    daemon.create_bank("main");
    let cached = || daemon.get_ok("/v1/banks/main/system-prompt/cached");

    assert_eq!(cached(), json!({"block": null}));
    assert_eq!(cached(), json!({"block": null}), "looking built a block");
    let block = daemon.get_ok("/v1/banks/main/system-prompt");
    assert_eq!(cached(), json!({ "block": block }));
    assert_eq!(
        daemon.get("/v1/banks/nope/system-prompt/cached").status,
        404
    );
}

#[test]
fn a_refresh_held_by_an_extraction_limit_answers_held_over_http_and_the_cli() {
    // A memory is extracted, so the profile has something to refresh; then
    // the next LLM call hits a usage limit, so the gate holds every call.
    // Whichever call met the limit, a refresh is then held until the reset,
    // not failed, over HTTP and the CLI alike: the script is spent, so a
    // call that reached the LLM would fail.
    use asphodel_core::{Clock, SystemClock};
    let resets_at = jiff::Timestamp::from_second(SystemClock.now().as_second() + 3600).unwrap();
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .script(&[
            auckland(),
            json!({"fail": "usage_limited", "resets_at": resets_at.to_string()}),
        ])
        .ready();
    daemon.seed_notes();
    daemon.ingest_document("main", "later.md", "# Later\n\nNothing much happened.\n");

    let held = daemon.wait_until(
        "a held refresh",
        |daemon| {
            let refresh = "/v1/banks/main/models/User%20profile/refresh?force=true";
            daemon.post_ok(refresh, &Value::Null)
        },
        |refreshed| refreshed["outcome"] == "held",
    );
    let until: jiff::Timestamp = held["detail"]["until"].as_str().unwrap().parse().unwrap();
    assert_eq!(until, resets_at, "{held}");

    let refresh = [
        "model",
        "refresh",
        "--bank",
        "main",
        "User profile",
        "--force",
    ];
    let json = json_out(run(cli(&daemon).args(refresh).arg("--json")));
    assert_eq!(json["outcome"], "held", "{json}");
}

// Forget and the purge pause.

#[test]
fn forget_erases_over_http_and_the_cli() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir).script(&[auckland()]).ready();
    daemon.create_bank("main");
    let ingested = daemon.ingest_notes();
    let id = daemon.wait_for_memory("main");
    daemon.wait_extracted("main");

    // Nothing was queued before the forget, so the erase runs at once.
    let forgotten = cli_json(&daemon, &format!("forget --bank main {id}"));
    assert_eq!(forgotten["forgotten"], json!([id]), "{forgotten}");
    let recall = daemon.recall("main", "where does Tim live? Auckland");
    let results = recall["results"].as_array().unwrap();
    assert!(!results.iter().any(|r| r["id"] == id.as_str()), "{recall}");

    // The key and hash are the tombstone: the same document brings nothing
    // back and queues nothing.
    let again = daemon.ingest_notes();
    assert_eq!(again["outcome"], "duplicate");
    assert_eq!(again["source"], ingested["source"]);
    assert_eq!(daemon.chunks("main")["queued"], json!([]));

    // A forgotten id is unknown from then on, over HTTP and the CLI.
    let forgotten = daemon.post_ok("/v1/banks/main/forget", &json!({"ids": [id, "not-an-id"]}));
    assert_eq!(forgotten["forgotten"], json!([]));
    assert_eq!(forgotten["unknown"], json!([id, "not-an-id"]));
    let printed = cli_json_fails(&daemon, &format!("forget --bank main {id}"));
    assert_eq!(printed["unknown"], json!([id]), "{printed}");

    let ids: Vec<String> = (0..51).map(|_| id.clone()).collect();
    let too_many = daemon.post("/v1/banks/main/forget", &json!({ "ids": ids }));
    assert_eq!(too_many.status, 400, "{}", too_many.body);
    let no_bank = daemon.post("/v1/banks/nobody/forget", &json!({"ids": [id]}));
    assert_eq!(no_bank.status, 404, "{}", no_bank.body);
}

#[test]
fn a_changed_fingerprint_pauses_purge_until_the_cli_acks_the_running_hash() {
    let dir = TestDir::new();
    let first = Serve::new(&dir).ready();
    first.create_bank("main");
    let config = first.get_ok("/v1/config");
    assert_eq!(config["purge"]["state"], "running");
    let stored = config["deletion_fingerprint"].as_str().unwrap().to_string();
    let plan = first.get_ok("/v1/purge/plan");
    assert_eq!(plan["pause"]["state"], "running");
    assert_eq!(plan["changed"], json!([]));
    first.stop();

    let delta = "[purge]\ndelta = 0.5\n";
    let second = Serve::new(&dir).tuning(delta).ready();
    let config = second.get_ok("/v1/config");
    let current = config["deletion_fingerprint"].as_str().unwrap().to_string();
    assert_ne!(current, stored);
    assert_eq!(
        config["purge"],
        json!({"state": "paused", "stored": stored})
    );

    let plan = cli_json(&second, "purge plan");
    assert_eq!(plan["pause"]["state"], "paused", "{plan}");
    assert_eq!(plan["current"], current.as_str());
    assert_eq!(plan["changed"], json!(["purge.delta"]));

    // Only the hash the running daemon computed is accepted.
    let output = run(cli(&second).args(["purge", "ack", "--hash", "nope"]));
    assert!(!output.status.success());
    let stale = second.post("/v1/purge/ack", &json!({"hash": stored}));
    assert_eq!(stale.status, 409, "{}", stale.body);
    assert_eq!(second.get_ok("/v1/config")["purge"]["state"], "paused");

    succeeded(run(cli(&second).args(["purge", "ack", "--hash", &current])));
    assert_eq!(second.get_ok("/v1/config")["purge"]["state"], "running");
    assert_eq!(second.get_ok("/v1/purge/plan")["changed"], json!([]));
    second.stop();

    // The ack is stored, so it holds after a restart.
    let third = Serve::new(&dir).tuning(delta).ready();
    assert_eq!(third.get_ok("/v1/config")["purge"]["state"], "running");
}

#[test]
fn a_ready_erase_runs_after_a_restart_without_an_llm() {
    // The forget arrives while a chunk queued
    // before it is in flight, so its erase waits. SIGTERM lets that chunk
    // finish and stops the worker before the erase runs. Restarted with no
    // LLM there's no worker, but the erase is ready and must still run.
    let dir = TestDir::new();
    let mut first = Serve::new(&dir)
        .script(&[auckland(), step(empty_reply(), 3000)])
        .ready();
    let id = first.seed_notes();
    first.ingest_document("main", "other.md", "# Other\n\nNothing to remember.\n");
    first.wait_until(
        "the second document in flight",
        |daemon| daemon.chunks("main"),
        |chunks| chunks["queued"][0]["in_flight"] == true,
    );
    let forgotten = first.post_ok("/v1/banks/main/forget", &json!({"ids": [id]}));
    assert_eq!(forgotten["forgotten"], json!([id]));
    // Hidden, but its erase hasn't run.
    let memory = format!("/v1/banks/main/memories/{id}");
    assert_eq!(first.get(&memory).status, 200);
    first.stop();

    let mut second = Serve::new(&dir).ready();
    second.wait_until(
        "the erase",
        |daemon| json!(daemon.get(&memory).status),
        |status| status == 404,
    );
}

// Backup, restore, status and the audit lists.
//
// `POST /v1/backup` streams the copy with its SHA-256 and length in
// headers. `asphodel backup --out <file|->` fails, and leaves nothing at
// `<file>`, when the stream is cut short, doesn't match its headers, or,
// for a file, fails `PRAGMA integrity_check`. `asphodel restore` runs
// offline and keeps the old database (and its WAL) aside. `asphodel status`
// exits non-zero whenever `attention` isn't empty, with `--json` too.

/// The response header holding the backup's SHA-256, as lowercase hex.
const SHA256_HEADER: &str = "asphodel-sha256";

/// The response header holding the backup's length in bytes.
const LENGTH_HEADER: &str = "asphodel-length";

/// Every SQLite database file starts with this.
const SQLITE_MAGIC: &[u8] = b"SQLite format 3\0";

/// The SHA-256 of `bytes` as lowercase hex.
fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// Serves one reply on a loopback port, as a daemon whose backup went wrong
/// would: `backup`'s headers with `replace` swapped in, then `body`, and
/// closes the connection. The framing headers aren't carried over: the body
/// is framed by its own length. Returns the `--url` that reaches it.
fn serve_once(backup: &Reply, replace: &[(&str, String)], body: &[u8]) -> String {
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
    let length = body.len();
    head.push_str(&format!(
        "content-length: {length}\r\nconnection: close\r\n\r\n"
    ));
    let mut reply = head.into_bytes();
    reply.extend_from_slice(body);

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let Ok((stream, _)) = listener.accept() else {
            return;
        };
        let mut stream = read_request(stream);
        let _ = stream.write_all(&reply);
        let _ = stream.shutdown(std::net::Shutdown::Write);
    });
    url
}

/// Reads a whole request off `stream`, so closing it with the request
/// unread can't reset the connection before the client reads the reply.
fn read_request(stream: TcpStream) -> TcpStream {
    stream.set_read_timeout(Some(TIMEOUT)).unwrap();
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
    let _ = reader.read_exact(&mut vec![0; length]);
    reader.into_inner()
}

/// `asphodel restore <backup> --data-dir <data>`, which runs offline.
fn restore(backup: &Path, data: &Path) -> Output {
    run(asphodel()
        .arg("restore")
        .arg(backup)
        .arg("--data-dir")
        .arg(data))
}

/// `asphodel backup --out <file>` against `daemon`, which must succeed.
fn backup_to_file(daemon: &Daemon, file: &Path) -> String {
    succeeded(run(cli(daemon).arg("backup").arg("--out").arg(file)))
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

/// The names in `dir` beyond what a running daemon keeps there: the
/// database, its WAL files and the lock.
fn extra_names(dir: &Path) -> Vec<String> {
    let live = ["asphodel.db", "asphodel.db-shm", "asphodel.db-wal", "lock"];
    let mut names = names(dir);
    names.retain(|name| !live.contains(&name.as_str()));
    names
}

/// Whether the file at `path` is a SQLite database.
fn is_sqlite(path: &Path) -> bool {
    fs::read(path).is_ok_and(|bytes| bytes.starts_with(SQLITE_MAGIC))
}

#[test]
fn backup_streams_a_checked_copy_with_its_hash_and_length() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir).ready();
    daemon.create_bank("main");

    let backup = daemon.send("POST", "/v1/backup", None);
    assert_eq!(backup.status, 200, "{}", backup.body);
    assert!(backup.bytes.starts_with(SQLITE_MAGIC), "not a SQLite file");
    let length = backup.bytes.len().to_string();
    assert_eq!(backup.header(LENGTH_HEADER), Some(length.as_str()));
    let hash = sha256(&backup.bytes);
    assert_eq!(backup.header(SHA256_HEADER), Some(hash.as_str()));

    // The temporary file the online backup wrote is gone once it's sent.
    let data = daemon.data_dir.clone();
    daemon.wait_until(
        "nothing left in the data dir",
        |_| json!(extra_names(&data)),
        |left| left == &json!([]),
    );
}

#[test]
fn a_backup_restores_offline_and_moves_the_old_database_aside() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir).script(&[auckland()]).ready();
    let id = daemon.seed_notes();

    let file = dir.path("backup.db");
    let out = backup_to_file(&daemon, &file);
    assert!(is_sqlite(&file), "{out}");
    let piped = run(cli(&daemon).args(["backup", "--out", "-"]));
    assert!(piped.status.success(), "{}", stderr(&piped));
    assert!(piped.stdout.starts_with(SQLITE_MAGIC), "not a SQLite file");

    // Forgotten after the backup, so a restore brings it back: forget never
    // reaches backups taken before it. SIGKILL leaves the forget in the WAL:
    // the restore has to move the WAL aside with the database, or SQLite
    // would replay the old store's frames onto the restored one.
    let forgotten = daemon.post_ok("/v1/banks/main/forget", &json!({"ids": [id]}));
    assert_eq!(forgotten["forgotten"], json!([id]));
    let data = daemon.data_dir.clone();
    drop(daemon);

    let before = names(&data);
    succeeded(restore(&file, &data));
    // The old database is moved aside, not deleted.
    let mut aside = extra_names(&data);
    aside.retain(|name| !before.contains(name));
    assert!(
        aside.iter().any(|name| is_sqlite(&data.join(name))),
        "no old database kept in {:?}",
        names(&data)
    );

    let daemon = Serve::new(&dir).ready();
    let recall = daemon.recall("main", "where does Tim live? Auckland");
    assert_eq!(recall["results"][0]["id"], id.as_str(), "{recall}");
    assert_eq!(recall["results"][0]["sentence"], SENTENCE);
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
    backup_to_file(&daemon, &file);

    // The daemon holds the data-dir lock, so the restore can't run beside it.
    let data = daemon.data_dir.clone();
    let before = names(&data);
    let output = restore(&file, &data);
    assert!(!output.status.success(), "restored under a running daemon");
    assert_eq!(names(&data), before);
    assert_eq!(daemon.get("/v1/health").status, 200);
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
        ("a newer schema", newer),
        ("a cut-short copy", full[..full.len() / 2].to_vec()),
        // An empty file is a valid, empty SQLite database.
        ("a database that isn't a store", Vec::new()),
    ];
    let before = snapshot(&data);
    for (case, bytes) in cases {
        let copy = dir.path("copy.db");
        fs::write(&copy, bytes).unwrap();
        let output = restore(&copy, &data);
        assert!(!output.status.success(), "restored {case}");
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
    let backup = daemon.send("POST", "/v1/backup", None);
    assert_eq!(backup.status, 200);
    let full = backup.bytes.clone();
    let half = full[..full.len() / 2].to_vec();

    let backup_to = |url: &str, out: &str| run(cli_at(url).args(["backup", "--out", out]));
    let out = dir.path("out.db");
    let out_arg = out.to_str().unwrap();

    // The control: the daemon's own reply, replayed, is accepted, so the
    // failures below come from what was changed and not from the fake.
    succeeded(backup_to(&serve_once(&backup, &[], &full), out_arg));
    assert_eq!(fs::read(&out).unwrap(), full);
    fs::remove_file(&out).unwrap();

    let mut flipped = full.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xff;
    let matching = [
        (SHA256_HEADER, sha256(&half)),
        (LENGTH_HEADER, half.len().to_string()),
    ];
    for (case, replace, body) in [
        (
            "a complete-looking reply shorter than its length header",
            &[][..],
            &half,
        ),
        ("a byte that doesn't match the hash", &[], &flipped),
        // The headers match the body, so only the integrity check of a file
        // target can catch it.
        ("a cut-short copy whose headers match it", &matching, &half),
    ] {
        let output = backup_to(&serve_once(&backup, replace, body), out_arg);
        assert!(!output.status.success(), "accepted {case}");
        assert!(!out.exists(), "left a file behind after {case}");
    }

    // Writing to stdout, a short stream fails the command too, so a pipe
    // into storage can tell.
    let output = backup_to(&serve_once(&backup, &[], &half), "-");
    assert!(
        !output.status.success(),
        "accepted a short stream on stdout"
    );
}

#[test]
fn a_failed_chunk_is_listed_needs_attention_and_is_retried() {
    // Every call fails, however many tries the chunk gets, so it fails for
    // good; the daemon restarted with a working script then retries it.
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir)
        .script(&vec![json!({"fail": "no_content"}); 32])
        .ready();
    daemon.create_bank("main");

    let status = daemon.get_ok("/v1/status");
    assert_eq!(status["attention"], json!([]), "{status}");
    assert_eq!(status["last_backup_at"], Value::Null, "{status}");
    succeeded(run(cli(&daemon).arg("status")));

    // A completed backup is reported.
    backup_to_file(&daemon, &dir.path("backup.db"));
    let status = daemon.get_ok("/v1/status");
    assert!(status["last_backup_at"].is_string(), "{status}");

    daemon.ingest_notes();
    let failed = daemon.wait_until(
        "a failed chunk",
        |daemon| daemon.get_ok("/v1/banks/main/chunks?failed=true"),
        |chunks| chunks["failed"].as_array().is_some_and(|f| f.len() == 1),
    );
    assert_eq!(
        failed["queued"],
        json!([]),
        "the filter lists failed chunks only"
    );
    assert_eq!(failed["failed"][0]["error_kind"], "llm_no_content");
    let chunk = failed["failed"][0]["chunk"].clone();
    assert_eq!(daemon.chunks("main")["failed"][0]["chunk"], chunk);

    let status = daemon.get_ok("/v1/status");
    assert_eq!(status["banks"]["main"]["failed_chunks"], 1, "{status}");
    assert_ne!(status["attention"], json!([]), "{status}");
    let output = run(cli(&daemon).arg("status"));
    assert!(!output.status.success(), "{}", stdout(&output));
    let printed = cli_json_fails(&daemon, "status");
    assert_ne!(printed["attention"], json!([]), "{printed}");

    let retried = daemon.post_ok("/v1/banks/main/chunks/retry", &json!({"chunks": [UNKNOWN]}));
    assert_eq!(retried["retried"], json!([]));
    assert_eq!(retried["unknown"], json!([UNKNOWN]));
    daemon.stop();

    let mut daemon = Serve::new(&dir).script(&[auckland()]).ready();
    let retried = cli_json(&daemon, "chunks --bank main --failed --retry");
    assert_eq!(retried["retried"]["retried"], json!([chunk]), "{retried}");
    daemon.wait_for_memory("main");
    daemon.wait_extracted("main");
    succeeded(run(cli(&daemon).arg("status")));
}

#[test]
fn the_audit_lists_hold_no_content_except_recalls() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir).script(&[auckland()]).ready();
    let id = daemon.seed_notes();

    // `wait_for_memory` recalled with this query, and recalls keep theirs.
    let recalls = cli_json(&daemon, "recalls --bank main");
    assert!(
        recalls["recalls"].as_array().is_some_and(|r| !r.is_empty()),
        "{recalls}"
    );
    let listed = recalls.to_string();
    assert!(listed.contains("where does Tim live? Auckland"), "{listed}");

    daemon.post_ok("/v1/banks/main/forget", &json!({"ids": [id]}));
    let forgets = cli_json(&daemon, "forgets --bank main");
    assert_eq!(
        forgets["forgets"].as_array().map(Vec::len),
        Some(1),
        "{forgets}"
    );
    let listed = forgets.to_string();
    assert!(listed.contains(&id), "{listed}");
    assert!(!listed.contains("Auckland"), "{listed}");

    let purges = cli_json(&daemon, "purges --bank main");
    assert_eq!(purges["purges"], json!([]), "{purges}");
    let sweeps = cli_json(&daemon, "sweeps --bank main");
    assert!(sweeps["sweeps"].is_array(), "{sweeps}");
    let unknown = daemon.get("/v1/banks/nobody/forgets");
    assert_eq!(unknown.status, 404, "{}", unknown.body);
}

#[test]
fn a_restored_store_keeps_its_fingerprint_and_pauses_purge_under_another() {
    // The stored fingerprint travels in the copy: a backup taken under
    // `purge.delta = 0.5`, restored over a store whose fingerprint matches
    // this daemon's, pauses purge at the next start.
    let source = TestDir::new();
    let delta = Serve::new(&source).tuning("[purge]\ndelta = 0.5\n").ready();
    delta.create_bank("main");
    let fingerprint = |daemon: &Daemon| {
        let config = daemon.get_ok("/v1/config");
        config["deletion_fingerprint"].as_str().unwrap().to_string()
    };
    let backed_up = fingerprint(&delta);
    let file = source.path("backup.db");
    backup_to_file(&delta, &file);
    drop(delta);

    let dir = TestDir::new();
    let first = Serve::new(&dir).ready();
    let current = fingerprint(&first);
    assert_ne!(current, backed_up);
    assert_eq!(first.get_ok("/v1/status")["attention"], json!([]));
    let data = first.data_dir.clone();
    first.stop();

    succeeded(restore(&file, &data));
    let second = Serve::new(&dir).ready();
    let status = second.get_ok("/v1/status");
    assert_eq!(
        status["purge"],
        json!({"state": "paused", "stored": backed_up}),
        "{status}"
    );
    assert_eq!(status["deletion_fingerprint"], current.as_str());
    assert_ne!(status["attention"], json!([]), "{status}");
    let output = run(cli(&second).arg("status"));
    assert!(!output.status.success(), "{}", stdout(&output));
}

/// SQL that undoes the latest registered migration, `SCHEMA_VERSION`'s, by
/// dropping the tables, indexes and added columns it creates. It only
/// handles SQL that creates or drops tables and indexes or adds columns,
/// plus the runner changes below, and fails loudly on any other. Extend
/// this test when such a migration lands rather than downgrading to a
/// schema no older binary wrote.
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
    let statements = |path: &Path| -> Vec<String> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter(|line| !line.trim_start().starts_with("--"))
            .collect::<Vec<_>>()
            .join("\n")
            .split(';')
            .map(str::trim)
            .filter(|statement| !statement.is_empty())
            .map(str::to_string)
            .collect()
    };
    // A dropped table comes back as the earlier migration that made it
    // created it.
    let created = |table: &str| -> String {
        files
            .iter()
            .filter(|(earlier, _)| earlier < version)
            .flat_map(|(_, path)| statements(path))
            .find(|statement| {
                let words: Vec<&str> = statement.split_whitespace().collect();
                words.len() > 5
                    && words[..5]
                        .iter()
                        .map(|word| word.to_uppercase())
                        .eq(["CREATE", "TABLE", "IF", "NOT", "EXISTS"])
                    && words[5].trim_end_matches('(') == table
            })
            .map(|statement| format!("{statement};"))
            .unwrap_or_else(|| panic!("no earlier migration creates {table}"))
    };
    let mut undo = Vec::new();
    for statement in statements(path) {
        let statement = statement.as_str();
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
            ["ALTER", "TABLE", ..] if upper.get(3..5) == Some(&["ADD".into(), "COLUMN".into()]) => {
                undo.push(format!("ALTER TABLE {} DROP COLUMN {};", name(2), name(5)))
            }
            ["DROP", "TABLE", "IF"] if upper.get(3) == Some(&"EXISTS".into()) => {
                undo.push(created(words[4]))
            }
            _ => panic!(
                "{} does more than create or drop tables and indexes; extend this downgrade for it",
                path.display()
            ),
        }
    }
    // The runner drops these columns itself (`DROPPED_COLUMNS`), so the SQL
    // doesn't show them.
    if *version == 14 {
        undo.push(
            "ALTER TABLE prompt_blocks ADD COLUMN entries TEXT NOT NULL DEFAULT '[]';".into(),
        );
    }
    // Version 15 drops the entry tables in the runner after copying their
    // answers and cites. Put back the v14 schema, including v13's section
    // column and both indexes; these drops do not appear in the SQL.
    if *version == 15 {
        undo.push(
            "CREATE TABLE mental_model_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT NOT NULL UNIQUE,
                model_id INTEGER NOT NULL REFERENCES mental_models(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                text TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                section TEXT
            );
            CREATE INDEX mental_model_entries_model ON mental_model_entries(model_id, position);
            CREATE TABLE mental_model_citations (
                entry_id INTEGER NOT NULL REFERENCES mental_model_entries(id) ON DELETE CASCADE,
                memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                PRIMARY KEY (entry_id, memory_id)
            );
            CREATE INDEX mental_model_citations_memory ON mental_model_citations(memory_id);"
                .into(),
        );
    }
    assert!(!undo.is_empty(), "{} creates nothing", path.display());
    undo.join("\n")
}

#[test]
fn an_older_backup_restores_into_a_new_data_dir_and_migrates_with_a_copy() {
    use asphodel_core::store::SCHEMA_VERSION;

    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir).script(&[auckland()]).ready();
    let id = daemon.seed_notes();
    let file = dir.path("backup.db");
    backup_to_file(&daemon, &file);
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
    succeeded(restore(&older_file, &data));

    let daemon = Serve::new(&dir).data_dir(&data).ready();
    let recall = daemon.recall("main", "where does Tim live? Auckland");
    assert_eq!(recall["results"][0]["id"], id.as_str(), "{recall}");

    // The migration took a pre-migration copy, which status reports and
    // which needs no attention.
    let status = daemon.get_ok("/v1/status");
    let copy = status["pre_migration_copy"].clone();
    assert_eq!(copy["from_version"], older, "{status}");
    let path = PathBuf::from(copy["path"].as_str().unwrap());
    assert!(path.exists(), "{status}");
    assert_eq!(status["attention"], json!([]), "{status}");
    succeeded(run(cli(&daemon).arg("status")));
    daemon.stop();

    // The copy is deleted at its deadline. Moved to `LEAD` from now, a
    // daemon started before it keeps the copy at open, so the deletion has
    // to come from a wake at the deadline itself, not the next hourly poll.
    // `GRACE` is scheduler latency, not a polling interval.
    const LEAD: Duration = Duration::from_secs(5);
    const GRACE: Duration = Duration::from_secs(5);
    let expires_at: jiff::Timestamp = copy["expires_at"].as_str().unwrap().parse().unwrap();
    let deadline = Instant::now() + LEAD;
    {
        use asphodel_core::store::{OpenOptions, Store};
        use asphodel_core::{Clock, SystemClock};
        let shift = expires_at.duration_since(SystemClock.now())
            - jiff::SignedDuration::try_from(LEAD).unwrap();
        let clock: Arc<dyn Clock> = Arc::new(SystemClock);
        let store = Store::open(&data, OpenOptions::default(), clock).unwrap();
        let moved = store
            .connection()
            .execute(
                "UPDATE migrations SET completed_at = completed_at - ?1 WHERE from_version = ?2",
                (i64::try_from(shift.as_micros()).unwrap(), older),
            )
            .unwrap();
        assert_eq!(moved, 1);
    }
    let mut daemon = Serve::new(&dir).data_dir(&data).ready();
    assert!(
        Instant::now() < deadline,
        "the daemon took longer than {LEAD:?} to start, so this run can't tell an open-time \
         deletion from a timed one"
    );
    assert!(path.exists(), "the copy was deleted before its deadline");
    daemon.wait_until(
        "the copy's deletion",
        |_| json!(path.exists()),
        |exists| exists == false,
    );
    assert!(
        Instant::now() < deadline + GRACE,
        "deleted too long past its deadline"
    );
    assert_eq!(
        daemon.get_ok("/v1/status")["pre_migration_copy"],
        Value::Null
    );
}

// The dashboard.
//
// None of the browse routes, nor `GET .../memories/{memory}`, writes an
// access or a recall row. A document id is never a path segment: a client
// normalizes `.` and `..` out of a path, so the id it confirmed wouldn't be
// the one it sent. The page, its modules and its fonts are served without
// the token, byte for byte as they are in `assets/dashboard/`: there is no
// build step. `app.js` is the module `tests/dashboard` drives.

impl Daemon {
    /// `POST /v1/banks/{bank}/documents/remove` for `document`.
    fn remove_document(&self, bank: &str, document: &str) -> Reply {
        self.post(
            &format!("/v1/banks/{bank}/documents/remove"),
            &json!({ "document_id": document }),
        )
    }

    /// The memories `query` lists in `bank`.
    fn memories(&self, bank: &str, query: &str) -> Vec<Value> {
        let listed = self.get_ok(&format!("/v1/banks/{bank}/memories?{query}"));
        listed["memories"]
            .as_array()
            .unwrap_or_else(|| panic!("no memories array: {listed}"))
            .clone()
    }
}

#[test]
fn the_dashboard_routes_browse_without_writing_anything() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir).script(&[auckland()]).ready();
    daemon.create_bank("main");
    let ingested = daemon.ingest_notes();
    daemon.wait_extracted("main");

    let banks = daemon.get_ok("/v1/banks");
    assert_eq!(banks["banks"][0]["name"], "main", "{banks}");

    let live = daemon.memories("main", "status=live&sort=fade");
    assert_eq!(live.len(), 1, "{live:?}");
    assert_eq!(live[0]["sentence"], SENTENCE);
    assert_eq!(live[0]["status"], "live");
    assert!(live[0]["fade"]["bank_days"].is_number(), "{}", live[0]);
    assert!(live[0]["fade"]["earliest_at"].is_string(), "{}", live[0]);
    let id = live[0]["id"].as_str().unwrap().to_string();
    assert_eq!(daemon.memories("main", "q=Auckland")[0]["id"], id.as_str());
    assert!(daemon.memories("main", "q=Berlin").is_empty());
    assert!(daemon.memories("main", "status=retracted").is_empty());

    let sources = daemon.get_ok("/v1/banks/main/sources?kind=document");
    assert_eq!(sources["sources"][0]["id"], ingested["source"], "{sources}");
    assert_eq!(sources["sources"][0]["document_id"], "notes.md");
    let source = ingested["source"].as_str().unwrap();
    let shown = daemon.get_ok(&format!("/v1/banks/main/sources/{source}"));
    assert_eq!(shown["text"], NOTES, "{shown}");
    assert_eq!(shown["chunks"][0]["memories"], json!([id]), "{shown}");

    // Explaining a recall or an injection runs the pipeline without using
    // anything either.
    let explain = |mode: &str, query: &str| {
        let body = json!({"mode": mode, "query": query});
        daemon.post_ok("/v1/banks/main/recall/explain", &body)
    };
    let recall = explain("recall", "Auckland");
    assert_eq!(recall["candidates"][0]["id"], id.as_str(), "{recall}");
    assert_eq!(recall["candidates"][0]["included"], true, "{recall}");
    let injection = explain("injection", "Where does Tim live? Auckland?");
    assert_eq!(
        injection["injection"]["injected"],
        json!([id]),
        "{injection}"
    );

    // Looking is not using: no access, no recall.
    let memory = daemon.get_ok(&format!("/v1/banks/main/memories/{id}"));
    assert_eq!(access_kinds(&memory), ["created"], "{memory}");
    let recalls = daemon.get_ok("/v1/banks/main/recalls");
    assert_eq!(recalls["recalls"], json!([]), "{recalls}");
}

#[test]
fn retract_and_document_removal_answer_over_http() {
    let dir = TestDir::new();
    let mut daemon = Serve::new(&dir).script(&[auckland()]).ready();
    daemon.create_bank("main");
    let ingested = daemon.ingest_notes();
    let source = ingested["source"].as_str().unwrap().to_string();
    daemon.wait_extracted("main");
    let id = daemon.memories("main", "status=live")[0]["id"]
        .as_str()
        .unwrap()
        .to_string();

    let retract = |bank: &str, id: &str| {
        daemon.post(
            &format!("/v1/banks/{bank}/memories/{id}/retract"),
            &json!({}),
        )
    };
    let retracted = daemon.ok(retract("main", &id));
    assert_eq!(retracted["memory"], id.as_str(), "{retracted}");
    assert!(retracted["retracted_at"].is_string(), "{retracted}");
    assert_eq!(retracted["reopened"], json!([]), "{retracted}");
    let again = retract("main", &id);
    assert_eq!(again.status, 409, "{}", again.body);
    assert!(again.json()["error"].is_string());
    for (bank, id) in [("main", UNKNOWN), ("nobody", id.as_str())] {
        let reply = retract(bank, id);
        assert_eq!(reply.status, 404, "{bank} {id}: {}", reply.body);
    }
    let retracted = daemon.memories("main", "status=retracted");
    assert_eq!(retracted.len(), 1, "{retracted:?}");
    assert_eq!(retracted[0]["id"], id.as_str());
    assert!(daemon.memories("main", "status=live").is_empty());

    let removed = daemon.ok(daemon.remove_document("main", "notes.md"));
    assert_eq!(removed["document_id"], "notes.md", "{removed}");
    assert_eq!(removed["sources"], json!([source]), "{removed}");
    assert_eq!(removed["forgotten"], json!([id]), "{removed}");
    assert_eq!(removed["dequeued"], 0, "{removed}");
    let reply = daemon.remove_document("main", "other.md");
    assert_eq!(reply.status, 404, "{}", reply.body);

    // The daemon runs the erase; the memory is gone and the source stays
    // as a tombstone, so the same document brings nothing back.
    daemon.wait_until(
        "the erase",
        |daemon| json!(daemon.get(&format!("/v1/banks/main/memories/{id}")).status),
        |status| status == 404,
    );
    let shown = daemon.get_ok(&format!("/v1/banks/main/sources/{source}"));
    assert_eq!(shown["text"], Value::Null, "{shown}");
    assert_eq!(shown["gone"]["reason"], "removed", "{shown}");
    let again = daemon.ingest_notes();
    assert_eq!(again["outcome"], "duplicate");
    assert_eq!(daemon.chunks("main")["queued"], json!([]));
}

/// GETs `path` without the token, checks it's served as `content_type`,
/// byte for byte as the file under `assets/`, and returns the reply.
fn served_unbuilt(addr: &Addr, path: &str, content_type: &str) -> Reply {
    let reply = request(addr, "GET", path, None, None).unwrap();
    assert_eq!(reply.status, 200, "{path}: {}", reply.body);
    assert!(
        reply
            .headers
            .contains(&format!("content-type: {content_type}")),
        "{path}: {}",
        reply.headers
    );
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    let file = assets.join(path.trim_start_matches('/'));
    let on_disk =
        fs::read(&file).unwrap_or_else(|error| panic!("{path} isn't {}: {error}", file.display()));
    assert!(
        reply.bytes == on_disk,
        "{path} differs from {}",
        file.display()
    );
    reply
}

#[test]
fn the_dashboard_assets_are_served_unbuilt_without_the_token() {
    let dir = TestDir::new();
    let daemon = Serve::new(&dir).listen("0.0.0.0:0").token(TOKEN).ready();
    let addr = daemon.addr.clone();

    let page = request(&addr, "GET", "/dashboard", None, None).unwrap();
    let script = attribute(&page.body, "script", "type=\"module\"", "src")
        .unwrap_or_else(|| panic!("no <script type=\"module\" src=...>: {}", page.body));
    assert!(
        script.starts_with("/dashboard/"),
        "{script} must resolve the same from /dashboard and /dashboard/"
    );

    // The module script and every module it imports as "./name.js".
    let mut pending = vec![script];
    let mut served = Vec::new();
    while let Some(path) = pending.pop() {
        if served.contains(&path) {
            continue;
        }
        let reply = served_unbuilt(&addr, &path, "text/javascript");
        let base = &path[..=path.rfind('/').unwrap()];
        let imports = relative_imports(&reply.body);
        pending.extend(imports.iter().map(|import| format!("{base}{import}")));
        served.push(path);
    }
    assert!(
        served.iter().any(|path| path == "/dashboard/app.js"),
        "{served:?}"
    );

    // The serif is self-hosted, under a strict font policy: fonts from the
    // daemon and nowhere else.
    let policy = page
        .header("content-security-policy")
        .unwrap_or_else(|| panic!("no CSP on the page: {}", page.headers));
    let directive = |name: &str| {
        policy
            .split(';')
            .map(str::trim)
            .find_map(|part| part.strip_prefix(name))
            .map(str::trim)
    };
    assert_eq!(directive("default-src "), Some("'none'"), "{policy}");
    assert_eq!(directive("script-src "), Some("'self'"), "{policy}");
    assert_eq!(directive("font-src "), Some("'self'"), "{policy}");

    let sheet = attribute(&page.body, "link", "rel=\"stylesheet\"", "href")
        .unwrap_or_else(|| panic!("no <link rel=\"stylesheet\" href=...>: {}", page.body));
    let css = request(&addr, "GET", &sheet, None, None).unwrap();
    assert_eq!(css.status, 200, "{sheet}: {}", css.body);
    let (faces, rest) = font_faces(&css.body);
    assert!(
        !faces.is_empty(),
        "{sheet} declares no @font-face: the serif must be self-hosted"
    );
    for (family, urls) in &faces {
        assert!(!urls.is_empty(), "@font-face {family} has no url()");
        assert!(
            rest.contains(family.as_str()),
            "{family} is declared but no font-family uses it"
        );
        for url in urls {
            assert!(!url.contains(".."), "{url} climbs out of /dashboard/");
            let path = if url.starts_with('/') {
                url.clone()
            } else {
                let base = &sheet[..=sheet.rfind('/').unwrap()];
                format!("{base}{}", url.trim_start_matches("./"))
            };
            assert!(
                path.starts_with("/dashboard/"),
                "{url} isn't served by the daemon"
            );
            served_unbuilt(&addr, &path, "font/woff2");
        }
    }
}

/// The `name` attribute of the first `<tag>` in `html` that holds `marker`.
fn attribute(html: &str, tag: &str, marker: &str, name: &str) -> Option<String> {
    let tags = Regex::new(&format!("<{tag}\\b[^>]*>")).unwrap();
    let tag = tags
        .find_iter(html)
        .map(|found| found.as_str())
        .find(|tag| tag.contains(marker))?;
    let value = Regex::new(&format!(r#"\s{name}="([^"]*)""#)).unwrap();
    Some(value.captures(tag)?[1].to_string())
}

/// The `./name.js` modules a module imports, by name.
fn relative_imports(module: &str) -> Vec<String> {
    let import = Regex::new(r#""\./([^"]+\.js)""#).unwrap();
    let imports = import.captures_iter(module);
    imports.map(|found| found[1].to_string()).collect()
}

/// Each `@font-face` in `css` as its family and `url()`s, and the rest of
/// the sheet with the faces taken out.
fn font_faces(css: &str) -> (Vec<(String, Vec<String>)>, String) {
    let face = Regex::new(r"@font-face[^}]*\}?").unwrap();
    let family = Regex::new(r#"font-family:\s*['"]?([^;'"]*)"#).unwrap();
    let url = Regex::new(r#"url\(\s*['"]?([^)'"]*)"#).unwrap();
    let faces = face
        .find_iter(css)
        .map(|found| {
            let found = found.as_str();
            let family = family
                .captures(found)
                .map_or("", |c| c.get(1).unwrap().as_str());
            let urls = url.captures_iter(found).map(|c| c[1].trim().to_string());
            (family.trim().to_string(), urls.collect())
        })
        .collect();
    (faces, face.replace_all(css, "").into_owned())
}

#[test]
fn a_document_is_removed_by_the_exact_id_in_the_body() {
    // Each id is its own document, over HTTP and the CLI. Sent as a path,
    // `folder/../victim` would reach the daemon as `victim`, and `..` not
    // at all.
    let dir = TestDir::new();
    let daemon = Serve::new(&dir).ready();
    daemon.create_bank("main");
    let mut sources = BTreeMap::new();
    for (id, text) in [
        ("victim", "# Victim\n\nThe victim stays.\n"),
        ("folder/../victim", "# Folder\n\nThis one goes.\n"),
        ("..", "# Dots\n\nSo does this one.\n"),
    ] {
        let ingested = daemon.ingest_document("main", id, text);
        sources.insert(id, ingested["source"].as_str().unwrap().to_string());
    }

    let over_http = daemon.ok(daemon.remove_document("main", ".."));
    let by_cli = cli_json(&daemon, "document remove --bank main folder/../victim");
    for (id, removed) in [("..", over_http), ("folder/../victim", by_cli)] {
        assert_eq!(removed["document_id"], id, "{removed}");
        assert_eq!(removed["sources"], json!([sources[id]]), "{removed}");
        assert_eq!(removed["dequeued"], 1, "{removed}");
    }
    let victim = format!("/v1/banks/main/sources/{}", sources["victim"]);
    let shown = daemon.get_ok(&victim);
    assert_eq!(shown["gone"], Value::Null, "{shown}");
    assert_eq!(shown["chunks"][0]["state"], "queued", "{shown}");

    // Nothing removes a document by its id in the path.
    let reply = daemon.send("DELETE", "/v1/banks/main/documents/victim", None);
    assert_eq!(reply.status, 404, "{}", reply.body);
    assert_eq!(daemon.get_ok(&victim)["gone"], Value::Null);

    for body in [json!({}), json!({ "document_id": "" })] {
        let reply = daemon.post("/v1/banks/main/documents/remove", &body);
        assert!(
            matches!(reply.status, 400 | 422),
            "{body}: {} {}",
            reply.status,
            reply.body
        );
    }
    let reply = daemon.remove_document("nobody", "victim");
    assert_eq!(reply.status, 404, "{}", reply.body);

    // The CLI removes `victim` itself, and fails once it's gone.
    let remove = ["document", "remove", "--bank", "main", "victim"];
    succeeded(run(cli(&daemon).args(remove)));
    assert_eq!(daemon.get_ok(&victim)["gone"]["reason"], "removed");
    assert!(!run(cli(&daemon).args(remove)).status.success());
}
