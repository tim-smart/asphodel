//! `asphodel llm login`, the setup wizard's login, and `asphodel serve` in
//! `chatgpt` mode, run as a process.
//!
//! The owner logs in once with the device-code flow, from the CLI or from
//! the setup wizard; the daemon reads the token file under the data dir and
//! refreshes it itself. The stub issuer doubles as the Codex backend, so a
//! test can see which credentials the daemon's LLM calls carry. These tests see only what an operator sees: flags,
//! exit codes, stderr, stdout and the HTTP API.
//!
//! The login talks to the issuer over HTTPS. To test the command without
//! the network, a hidden `ASPHODEL_LLM_ISSUER` variable, environment only
//! and never in `--help`, points the flow at a loopback stub, in the same
//! spirit as `ASPHODEL_MODELS=fake`. `ASPHODEL_LLM_LOGIN_WAIT_MS`, also
//! hidden, shortens how long the daemon waits for approval, so an expired
//! login doesn't take 15 minutes.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// `asphodel <args>` with a clean environment, so the caller's
/// `ASPHODEL_*` variables can't leak in.
fn asphodel(args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_asphodel"));
    command.env_clear().args(args);
    command
}

/// `asphodel llm login` on `data` against `issuer`.
fn login(data: &Path, issuer: &str) -> Command {
    let mut command = asphodel(&["llm", "login", "--data-dir"]);
    command.arg(data).env("ASPHODEL_LLM_ISSUER", issuer);
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
            "asphodel-llm-login-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn data(&self) -> PathBuf {
        let path = self.0.join("data");
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    /// Floors for the fakes plus the subscription mode.
    fn chatgpt_tuning(&self) -> PathBuf {
        let path = self.0.join("tuning.toml");
        let tuning = "[llm]\nauth = \"chatgpt\"\nmodel = \"gpt-5.1\"\n\
                      [injection.reranker_floors]\n\"fake-reranker:v1\" = 0.0\n\
                      [ranking.relevance_scales]\n\"fake-reranker:v1\" = 1.0\n\
                      [reconcile.embedding_floors]\n\"fake-embedder:v1\" = 0.5\n";
        std::fs::write(&path, tuning).unwrap();
        path
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
    /// `GET <path>`: the status and the body.
    fn get(&self, path: &str) -> std::io::Result<(u16, String)> {
        self.request("GET", path, None)
    }

    /// `<method> <path>` with a JSON body if given: the status and the body.
    fn request(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
    ) -> std::io::Result<(u16, String)> {
        let mut stream = TcpStream::connect(&self.addr)?;
        let body = body.map_or(String::new(), Value::to_string);
        write!(
            stream,
            "{method} {path} HTTP/1.0\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\n\r\n{body}",
            body.len()
        )?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        let (head, body) = response.split_once("\r\n\r\n").unwrap_or((&response, ""));
        let status = head.split_whitespace().nth(1).and_then(|s| s.parse().ok());
        Ok((status.unwrap_or(0), body.to_string()))
    }

    /// [`Daemon::request`] with the body read as JSON.
    fn send(&self, method: &str, path: &str, body: Option<&Value>) -> (u16, Value) {
        let (status, body) = self.request(method, path, body).unwrap();
        (status, serde_json::from_str(&body).unwrap_or(Value::Null))
    }

    /// `GET /v1/config`, which must answer 200.
    fn config(&self) -> Value {
        let (status, body) = self.get("/v1/config").unwrap();
        assert_eq!(status, 200, "{body}");
        serde_json::from_str(&body).unwrap()
    }

    /// `GET` [`LOGIN`], which must answer 200.
    fn login(&self) -> Value {
        let (status, body) = self.send("GET", LOGIN, None);
        assert_eq!(status, 200, "{body}");
        body
    }

    /// Polls [`LOGIN`] until the login is in `state`, and returns it.
    fn wait_login(&mut self, state: &str) -> Value {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let login = self.login();
            if login["state"] == state {
                return login;
            }
            if Instant::now() >= deadline {
                panic!("the login never became {state}: {login}\n{}", self.log());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Ingests [`NOTES`] into a new bank `main`, and waits until extraction
    /// has made its memory and recall finds it.
    fn extract_notes(&mut self) {
        let bank =
            json!({"owner_name": "Tim", "assistant_name": "Ash", "timezone": "Pacific/Auckland"});
        let (status, body) = self.send("PUT", "/v1/banks/main", Some(&bank));
        assert_eq!(status, 201, "{body}");
        let document = json!({
            "document_id": "notes.md",
            "text": NOTES,
            "reference_date": "2026-09-30",
            "reference_date_exact": true,
            "timezone": null
        });
        let (status, body) = self.send("POST", "/v1/banks/main/documents", Some(&document));
        assert!((200..300).contains(&status), "{status}: {body}");
        let query = json!({"query": "where does Tim live? Auckland"});
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let (status, recall) = self.send("POST", "/v1/banks/main/recall", Some(&query));
            assert_eq!(status, 200, "{recall}");
            if recall["results"][0]["sentence"] == SENTENCE {
                return;
            }
            if Instant::now() >= deadline {
                panic!("extraction never made the memory: {recall}\n{}", self.log());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Waits until `/v1/health` says the daemon is ready, not just set up.
    /// It answers 503 while the store opens after setup.
    fn wait_ready(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(10);
        let ready = |daemon: &Self| {
            daemon
                .get("/v1/health")
                .is_ok_and(|(status, body)| status == 200 && body.contains("\"ready\":true"))
        };
        while !ready(self) {
            if Instant::now() >= deadline {
                panic!("the daemon never became ready:\n{}", self.log());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Kills the daemon and returns its whole log.
    fn log(&mut self) -> String {
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.log.take().unwrap().join().unwrap()
    }
}

/// `asphodel serve` in `chatgpt` mode on `data` and the fake models,
/// listening on `listen`.
fn serve(dir: &TestDir, data: &Path, listen: &str) -> Command {
    let mut command = asphodel(&["serve", "--data-dir"]);
    command
        .arg(data)
        .arg("--config")
        .arg(dir.chatgpt_tuning())
        .args(["--listen", listen])
        .env("ASPHODEL_MODELS", "fake");
    command
}

/// Starts [`serve`] on a free loopback port with `issuer` as the issuer,
/// and waits until `/v1/health` answers 200.
fn start(dir: &TestDir, data: &Path, issuer: &str) -> Daemon {
    launch(|listen| serve(dir, data, listen), issuer)
}

/// Starts `asphodel serve` on `data` with no tuning file, so it waits for
/// first-run setup, with `issuer` as the issuer and `env` set.
fn start_setup(data: &Path, issuer: &str, env: &[(&str, &str)]) -> Daemon {
    launch(
        |listen| {
            let mut command = asphodel(&["serve", "--data-dir"]);
            command
                .arg(data)
                .args(["--listen", listen])
                .env("ASPHODEL_MODELS", "fake")
                .envs(env.iter().copied());
            command
        },
        issuer,
    )
}

/// Runs the daemon `command` makes for a free loopback port, with `issuer`
/// as the issuer, and waits until `/v1/health` answers 200.
fn launch(command: impl FnOnce(&str) -> Command, issuer: &str) -> Daemon {
    let addr = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let addr = addr.to_string();
    let mut child = command(&addr)
        .env("ASPHODEL_LOG", "trace")
        .env("ASPHODEL_LLM_ISSUER", issuer)
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
        addr,
        log: Some(log),
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while !matches!(daemon.get("/v1/health"), Ok((200, _))) {
        if Instant::now() >= deadline {
            panic!("the daemon never became ready:\n{}", daemon.log());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    daemon
}

fn base64url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let mut buffer = [0u8; 3];
        buffer[..chunk.len()].copy_from_slice(chunk);
        let n = u32::from_be_bytes([0, buffer[0], buffer[1], buffer[2]]);
        for i in 0..chunk.len() + 1 {
            out.push(ALPHABET[((n >> (18 - 6 * i)) & 0x3f) as usize] as char);
        }
    }
    out
}

fn jwt(claims: Value) -> String {
    let header = base64url(br#"{"alg":"RS256","typ":"JWT"}"#);
    let payload = base64url(claims.to_string().as_bytes());
    format!("{header}.{payload}.{}", base64url(b"signature"))
}

/// How the stub issuer answers a poll for a device code.
#[derive(Debug, Clone, Copy)]
enum Poll {
    /// 403, as the issuer answers until the owner approves.
    Pending,
    Approved,
    /// 400: the issuer won't take the code.
    Refused,
}

/// Starts a loopback issuer (usercode, two pending polls, approval,
/// exchange) and returns its URL.
fn issuer() -> String {
    let issuer = stub_issuer(|_, poll| {
        if poll <= 2 {
            Poll::Pending
        } else {
            Poll::Approved
        }
    });
    issuer.url.clone()
}

/// Starts a loopback issuer. The `n`th code asked for (from 1) is
/// `ABCD-000n`, and `poll(n, p)` answers the `p`th poll for it (from 1);
/// `poll(n, 0)` answers the request for the code itself, where `Refused`
/// hands out none. `POST /responses` answers every LLM call with
/// [`auckland_reply`].
fn stub_issuer(poll: impl Fn(usize, usize) -> Poll + Send + Sync + 'static) -> Arc<Issuer> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let issuer = Arc::new(Issuer {
        url: format!("http://{}", listener.local_addr().unwrap()),
        poll: Box::new(poll),
        codes: AtomicUsize::new(0),
        polls: Mutex::new(HashMap::new()),
        llm_calls: Mutex::new(Vec::new()),
    });
    std::thread::spawn({
        let issuer = Arc::clone(&issuer);
        move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let issuer = Arc::clone(&issuer);
                std::thread::spawn(move || issuer.answer(stream));
            }
        }
    });
    issuer
}

struct Issuer {
    url: String,
    poll: Box<dyn Fn(usize, usize) -> Poll + Send + Sync>,
    /// Codes asked for so far.
    codes: AtomicUsize,
    /// Polls so far for each code.
    polls: Mutex<HashMap<usize, usize>>,
    /// The headers of each LLM call, names lowercased.
    llm_calls: Mutex<Vec<HashMap<String, String>>>,
}

/// The access token the stub's exchange hands out.
fn access_token() -> String {
    jwt(json!({"sub": "first", "exp": 4_102_444_800i64}))
}

/// The document the token-use test ingests, and the claim the stub's LLM
/// makes about it.
const NOTES: &str = "# Notes\n\nI live in Auckland and I drink tea every morning.\n";
const SENTENCE: &str = "Tim lives in Auckland.";

/// An extraction reply with one fact quoted from [`NOTES`].
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

/// [`auckland_reply`] as the Codex backend streams it.
fn completion() -> String {
    let events = [
        (
            "response.output_item.done",
            json!({"item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": auckland_reply().to_string()}]
            }}),
        ),
        (
            "response.completed",
            json!({"response": {
                "status": "completed",
                "usage": {"input_tokens": 41, "output_tokens": 7, "total_tokens": 48}
            }}),
        ),
    ];
    events
        .into_iter()
        .map(|(kind, mut data)| {
            data["type"] = json!(kind);
            format!("event: {kind}\ndata: {data}\n\n")
        })
        .collect()
}

impl Issuer {
    fn answer(&self, mut stream: TcpStream) {
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let lines = reader.by_ref().lines().map_while(Result::ok);
        let head: Vec<String> = lines.take_while(|line| !line.is_empty()).collect();
        let Some(path) = head.first().and_then(|line| line.split_whitespace().nth(1)) else {
            return;
        };
        let length = head.iter().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().ok())?
        });
        let mut body = vec![0; length.unwrap_or(0)];
        reader.read_exact(&mut body).unwrap();
        if path.ends_with("/responses") {
            let headers = head.iter().skip(1).filter_map(|line| {
                let (key, value) = line.split_once(':')?;
                Some((key.trim().to_ascii_lowercase(), value.trim().to_string()))
            });
            self.llm_calls.lock().unwrap().push(headers.collect());
            let body = completion();
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            return;
        }
        let (status, body) = match path {
            "/api/accounts/deviceauth/usercode" => {
                let n = self.codes.fetch_add(1, Ordering::SeqCst) + 1;
                match (self.poll)(n, 0) {
                    Poll::Refused => (503, json!({})),
                    Poll::Pending | Poll::Approved => (
                        200,
                        json!({"device_auth_id": format!("dev_{n}"), "user_code": format!("ABCD-{n:04}"), "interval": "0"}),
                    ),
                }
            }
            "/api/accounts/deviceauth/token" => {
                let sent: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                let n = sent["device_auth_id"]
                    .as_str()
                    .and_then(|id| id.strip_prefix("dev_"))
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0);
                let polls = {
                    let mut polls = self.polls.lock().unwrap();
                    let count = polls.entry(n).or_insert(0);
                    *count += 1;
                    *count
                };
                match (self.poll)(n, polls) {
                    Poll::Pending => {
                        // The daemon polls without a pause at interval 0.
                        std::thread::sleep(Duration::from_millis(20));
                        (403, json!({}))
                    }
                    Poll::Refused => (400, json!({})),
                    Poll::Approved => (
                        200,
                        json!({"authorization_code": "code_9", "code_challenge": "chal", "code_verifier": "verif"}),
                    ),
                }
            }
            "/oauth/token" => (
                200,
                json!({
                    "id_token": jwt(json!({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_7f3a9c"}})),
                    "access_token": access_token(),
                    "refresh_token": "rt-first"
                }),
            ),
            _ => (404, json!({})),
        };
        let body = body.to_string();
        let _ = write!(
            stream,
            "HTTP/1.1 {status} Status\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
    }
}

/// The wizard's ChatGPT login: `POST` starts one, `GET` reads its state.
const LOGIN: &str = "/v1/setup/llm-login";

#[test]
fn a_login_writes_a_private_token_file_the_daemon_reads_and_never_shows_a_token() {
    // The daemon starts without a login; extraction waits for one. After
    // the device-code login it reads the token file and is logged in. The
    // prompt names the URL and the code, and no token is ever shown or
    // logged.
    let issuer = issuer();
    let dir = TestDir::new();
    let data = dir.data();
    let daemon = start(&dir, &data, &issuer);
    let config = daemon.config();
    assert_eq!(config["llm"]["auth"], "chatgpt");
    assert_eq!(config["llm"]["logged_in"], false);
    let token_file = PathBuf::from(config["llm"]["token_file"].as_str().unwrap());
    assert_eq!(token_file.parent(), Some(data.as_path()));
    drop(daemon);

    let output = run(&mut login(&data, &issuer));
    assert!(output.status.success(), "{}", stderr(&output));
    let shown = String::from_utf8_lossy(&output.stdout).into_owned() + &stderr(&output);
    assert!(shown.contains(&format!("{issuer}/codex/device")), "{shown}");
    assert!(shown.contains("ABCD-0001"), "{shown}");
    let mode = std::fs::metadata(&token_file).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o600);

    let mut daemon = start(&dir, &data, &issuer);
    let config = daemon.config();
    assert_eq!(config["llm"]["logged_in"], true);
    // Not in the prompt, the log or the config.
    let everywhere = format!("{shown}\n{}\n{config}", daemon.log());
    for forbidden in ["rt-first", "eyJ", "acct_7f3a9c"] {
        assert!(
            !everywhere.contains(forbidden),
            "{forbidden}:\n{everywhere}"
        );
    }
}

#[test]
fn llm_login_never_reads_the_codex_cli_credentials() {
    // Sharing the Codex CLI's token chain would log one of the two out:
    // refresh tokens are single-use. A login with no reachable issuer must
    // fail rather than fall back to ~/.codex/auth.json.
    let dir = TestDir::new();
    let home = dir.0.join("home");
    std::fs::create_dir_all(home.join(".codex")).unwrap();
    std::fs::write(
        home.join(".codex").join("auth.json"),
        json!({"tokens": {"access_token": "cli-access", "refresh_token": "cli-refresh", "account_id": "acct_cli"}})
            .to_string(),
    )
    .unwrap();
    // A port that was bound and released, so the issuer is unreachable.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let unreachable = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let output = run(login(&dir.data(), &unreachable).env("HOME", &home));
    assert!(!output.status.success());
    let files = std::fs::read_dir(dir.data()).unwrap().count();
    assert_eq!(files, 0, "a token file appeared without a login");
    let stderr = stderr(&output);
    assert!(!stderr.contains("cli-refresh"), "{stderr}");
}

#[test]
fn serve_refuses_a_key_together_with_chatgpt_mode() {
    let dir = TestDir::new();
    let mut command = serve(&dir, &dir.data(), "127.0.0.1:0");
    let output = run(command.env("ASPHODEL_LLM_API_KEY", "sk-live-41b2e8-secret"));
    assert!(!output.status.success());
    let stderr = stderr(&output);
    assert!(stderr.contains("ASPHODEL_LLM_API_KEY"), "{stderr}");
    assert!(!stderr.contains("41b2e8"), "{stderr}");
}

#[test]
fn the_setup_wizard_logs_in_and_the_running_daemon_uses_the_login() {
    // The wizard starts the login before setup is finished, and finishing
    // doesn't wait for it. Its state stays readable afterwards, and once the
    // owner approves, the running daemon is logged in without a restart.
    // Starting again while a login waits hands back the same code, so a
    // reloaded wizard can show it; once setup is done, only
    // `asphodel llm login` starts one. The same daemon's LLM calls then
    // carry the new login. No token is ever shown or logged.
    let approved = Arc::new(AtomicBool::new(false));
    let issuer = stub_issuer({
        let approved = Arc::clone(&approved);
        move |_, _| match approved.load(Ordering::SeqCst) {
            true => Poll::Approved,
            false => Poll::Pending,
        }
    });
    let dir = TestDir::new();
    let mut daemon = start_setup(&dir.data(), &issuer.url, &[]);
    assert_eq!(daemon.login()["state"], "none");

    let (status, started) = daemon.send("POST", LOGIN, None);
    assert_eq!(status, 200, "{started}");
    assert_eq!(started["state"], "pending");
    assert_eq!(
        started["verification_url"],
        format!("{}/codex/device", issuer.url)
    );
    assert_eq!(started["user_code"], "ABCD-0001");
    assert_eq!(daemon.send("POST", LOGIN, None), (200, started.clone()));

    let chatgpt = json!({"llm": {"auth": "chatgpt", "model": "gpt-5.1", "endpoint": issuer.url}});
    let (status, done) = daemon.send("POST", "/v1/setup", Some(&chatgpt));
    assert_eq!(status, 200, "{done}");
    daemon.wait_ready();
    // Read without a token, the state never carries the code.
    let pending = daemon.login();
    assert_eq!(pending, json!({"state": "pending"}));
    assert_eq!(daemon.config()["llm"]["logged_in"], false);

    approved.store(true, Ordering::SeqCst);
    let login = daemon.wait_login("done");
    let config = daemon.config();
    assert_eq!(config["llm"]["logged_in"], true, "{config}");
    assert_eq!(daemon.send("POST", LOGIN, None).0, 409);

    daemon.extract_notes();
    let calls = issuer.llm_calls.lock().unwrap().clone();
    assert!(!calls.is_empty());
    for call in &calls {
        let bearer = format!("Bearer {}", access_token());
        assert_eq!(call.get("authorization"), Some(&bearer));
        let account = call.get("chatgpt-account-id").map(String::as_str);
        assert_eq!(account, Some("acct_7f3a9c"));
    }

    let everywhere = format!("{started}\n{login}\n{config}\n{}", daemon.log());
    for forbidden in ["rt-first", "eyJ", "acct_7f3a9c"] {
        assert!(
            !everywhere.contains(forbidden),
            "{forbidden}:\n{everywhere}"
        );
    }
}

#[test]
fn a_failed_or_expired_wizard_login_is_shown_and_can_be_retried() {
    // The issuer hands out no first code and refuses the second at its
    // first poll; nobody approves the third before the wait runs out, and
    // the fourth goes through. Each retry is a new code, and a failure says
    // why.
    let issuer = stub_issuer(|code, poll| match (code, poll) {
        (1, 0) => Poll::Refused,
        (2, 0) => Poll::Pending,
        (2, _) => Poll::Refused,
        (3, _) => Poll::Pending,
        _ => Poll::Approved,
    });
    let dir = TestDir::new();
    let wait = [("ASPHODEL_LLM_LOGIN_WAIT_MS", "200")];
    let mut daemon = start_setup(&dir.data(), &issuer.url, &wait);
    let (status, refused) = daemon.send("POST", LOGIN, None);
    assert_eq!(status, 502, "{refused}");
    let login = daemon.login();
    assert_eq!(login["state"], "failed", "{login}");
    assert!(login["error"].is_string(), "{login}");
    let mut codes = Vec::new();
    for state in ["failed", "expired", "done"] {
        let (status, started) = daemon.send("POST", LOGIN, None);
        assert_eq!(status, 200, "{started}");
        codes.push(started["user_code"].clone());
        let login = daemon.wait_login(state);
        if state == "failed" {
            assert!(login["error"].is_string(), "{login}");
        }
    }
    assert_eq!(codes, ["ABCD-0002", "ABCD-0003", "ABCD-0004"]);
}

#[test]
fn sigterm_stops_the_daemon_promptly_while_a_wizard_login_waits() {
    // A login nobody approves waits 15 minutes for the owner; a supervisor
    // stopping the daemon meanwhile mustn't wait with it.
    let issuer = stub_issuer(|_, _| Poll::Pending);
    let dir = TestDir::new();
    let mut daemon = start_setup(&dir.data(), &issuer.url, &[]);
    let (status, started) = daemon.send("POST", LOGIN, None);
    assert_eq!(status, 200, "{started}");

    let signalled = Command::new("kill")
        .args(["-TERM", &daemon.child.id().to_string()])
        .status()
        .unwrap();
    assert!(signalled.success(), "kill -TERM failed");
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        match daemon.child.try_wait().unwrap() {
            Some(status) => break status,
            None if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            None => panic!("the daemon didn't stop:\n{}", daemon.log()),
        }
    };
    assert!(status.success(), "{status}\n{}", daemon.log());
}
