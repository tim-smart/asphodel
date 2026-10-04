//! `asphodel llm login` and `asphodel serve` in `chatgpt` mode, run as a
//! process.
//!
//! The owner logs in once with the device-code flow; the daemon reads the
//! token file under the data dir and refreshes it itself. These tests see
//! only what an operator sees: flags, exit codes, stderr, stdout and
//! `GET /v1/config`.
//!
//! The login talks to the issuer over HTTPS. To test the command without
//! the network, a hidden `ASPHODEL_LLM_ISSUER` variable, environment only
//! and never in `--help`, points the flow at a loopback stub, in the same
//! spirit as `ASPHODEL_MODELS=fake`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
        let mut stream = TcpStream::connect(&self.addr)?;
        write!(stream, "GET {path} HTTP/1.0\r\n\r\n")?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        let (head, body) = response.split_once("\r\n\r\n").unwrap_or((&response, ""));
        let status = head.split_whitespace().nth(1).and_then(|s| s.parse().ok());
        Ok((status.unwrap_or(0), body.to_string()))
    }

    /// `GET /v1/config`, which must answer 200.
    fn config(&self) -> Value {
        let (status, body) = self.get("/v1/config").unwrap();
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
    let addr = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let addr = addr.to_string();
    let mut child = serve(dir, data, &addr)
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

/// Starts a loopback issuer (usercode, two pending polls, approval,
/// exchange) and returns its URL.
fn issuer() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let polls = Arc::new(AtomicUsize::new(0));
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            let polls = Arc::clone(&polls);
            std::thread::spawn(move || answer(stream, &polls));
        }
    });
    url
}

fn answer(mut stream: TcpStream, polls: &AtomicUsize) {
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
    reader
        .read_exact(&mut vec![0; length.unwrap_or(0)])
        .unwrap();
    let (status, body) = match path {
        "/api/accounts/deviceauth/usercode" => (
            200,
            json!({"device_auth_id": "dev_123", "user_code": "ABCD-EFGH", "interval": "0"}),
        ),
        "/api/accounts/deviceauth/token" if polls.fetch_add(1, Ordering::SeqCst) < 2 => {
            (403, json!({}))
        }
        "/api/accounts/deviceauth/token" => (
            200,
            json!({"authorization_code": "code_9", "code_challenge": "chal", "code_verifier": "verif"}),
        ),
        "/oauth/token" => (
            200,
            json!({
                "id_token": jwt(json!({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_7f3a9c"}})),
                "access_token": jwt(json!({"sub": "first", "exp": 4_102_444_800i64})),
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
    assert!(shown.contains("ABCD-EFGH"), "{shown}");
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
