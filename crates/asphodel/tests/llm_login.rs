//! `asphodel llm login` and `asphodel serve` in `chatgpt` mode, run as a
//! process (the scope addition to TIM-105, comment `01a0f6c8`).
//!
//! The owner logs in once with the device-code flow; the daemon reads the
//! token file under the data dir and refreshes it itself. These tests see
//! only what an operator sees: flags, exit codes, stderr, stdout and the
//! resolved config line.
//!
//! The login talks to the issuer over HTTPS. To test the command without
//! the network, a hidden `ASPHODEL_LLM_ISSUER` variable, environment only
//! and never in `--help`, points the flow at a loopback stub, in the same
//! spirit as `ASPHODEL_MODELS=fake`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};

const FAKE_EMBEDDER: &str = "fake-embedder:v1";
const FAKE_RERANKER: &str = "fake-reranker:v1";
const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const TOKEN_FILE: &str = "llm-tokens.json";

/// `asphodel` with a clean environment, so the caller's `ASPHODEL_*`
/// variables can't leak in.
fn asphodel() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_asphodel"));
    command.env_clear();
    command
}

fn run(command: &mut Command) -> Output {
    command.stdin(Stdio::null()).output().unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
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

    fn file(&self, name: &str, text: &str) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, text).unwrap();
        path
    }

    /// Floors for the fakes plus the subscription mode.
    fn chatgpt_tuning(&self) -> PathBuf {
        self.file(
            "tuning.toml",
            &format!(
                "[llm]\nauth = \"chatgpt\"\nmodel = \"gpt-5.1\"\n\
                 [injection.reranker_floors]\n\"{FAKE_RERANKER}\" = 0.0\n\
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

/// Starts the daemon on the fake models and an ephemeral loopback port and
/// collects its log up to the "listening" line.
fn start(command: &mut Command) -> Daemon {
    let mut child = command
        .args(["--listen", "127.0.0.1:0"])
        .env("ASPHODEL_MODELS", "fake")
        .env("ASPHODEL_LOG", "trace")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stderr = child.stderr.take().unwrap();
    let (lines, received) = std::sync::mpsc::channel();
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
fn resolved_config(log: &str) -> Value {
    let line = log
        .lines()
        .find(|line| line.contains("resolved config"))
        .unwrap_or_else(|| panic!("no resolved config line in:\n{log}"));
    let start = line.find("config=").expect("a config field") + "config=".len();
    let mut stream = serde_json::Deserializer::from_str(&line[start..]).into_iter::<Value>();
    stream.next().unwrap().unwrap()
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
    format!(
        "{}.{}.{}",
        base64url(br#"{"alg":"RS256","typ":"JWT"}"#),
        base64url(claims.to_string().as_bytes()),
        base64url(b"signature")
    )
}

/// A loopback issuer: usercode, two pending polls, approval, exchange.
struct Issuer {
    url: String,
    requests: Arc<Mutex<Vec<(String, String)>>>,
}

impl Issuer {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&requests);
        let polls = Arc::new(AtomicUsize::new(0));
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let log = Arc::clone(&log);
                let polls = Arc::clone(&polls);
                std::thread::spawn(move || answer(stream, &log, &polls));
            }
        });
        Self {
            url: format!("http://127.0.0.1:{port}"),
            requests,
        }
    }

    fn paths(&self) -> Vec<String> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .map(|(path, _)| path.clone())
            .collect()
    }
}

fn answer(mut stream: TcpStream, log: &Mutex<Vec<(String, String)>>, polls: &AtomicUsize) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let path = line
        .split_whitespace()
        .nth(1)
        .unwrap_or_default()
        .to_string();
    let mut length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        if let Some((key, value)) = line.split_once(':')
            && key.trim().eq_ignore_ascii_case("content-length")
        {
            length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    log.lock()
        .unwrap()
        .push((path.clone(), String::from_utf8_lossy(&body).into_owned()));
    let (status, body) = match path.as_str() {
        "/api/accounts/deviceauth/usercode" => (
            200,
            json!({"device_auth_id": "dev_123", "user_code": "ABCD-EFGH", "interval": "0"}).to_string(),
        ),
        "/api/accounts/deviceauth/token" if polls.fetch_add(1, Ordering::SeqCst) < 2 => {
            (403, "{}".into())
        }
        "/api/accounts/deviceauth/token" => (
            200,
            json!({"authorization_code": "code_9", "code_challenge": "chal", "code_verifier": "verif"})
                .to_string(),
        ),
        "/oauth/token" => (
            200,
            json!({
                "id_token": jwt(json!({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_7f3a9c"}})),
                "access_token": jwt(json!({"sub": "first", "exp": 4_102_444_800i64})),
                "refresh_token": "rt-first"
            })
            .to_string(),
        ),
        _ => (404, "{}".into()),
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {status} Status\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.flush();
}

#[test]
fn llm_login_takes_the_data_dir_and_hides_the_issuer_override() {
    let output = run(asphodel().args(["llm", "login", "--help"]));
    assert!(output.status.success(), "{}", stderr(&output));
    let help = stdout(&output);
    let line = help
        .lines()
        .find(|line| line.trim_start().starts_with("--data-dir"))
        .unwrap_or_else(|| panic!("no --data-dir in:\n{help}"));
    assert!(line.contains("ASPHODEL_DATA_DIR"), "{line}");
    assert!(!help.contains("ASPHODEL_LLM_ISSUER"), "{help}");
    // Logging in never goes through the daemon: no --url, no token.
    assert!(!help.contains("--url"), "{help}");
}

#[test]
fn llm_login_shows_the_code_and_writes_the_token_file() {
    let issuer = Issuer::start();
    let dir = TestDir::new();
    let data = dir.data();
    let output = run(asphodel()
        .args(["llm", "login", "--data-dir"])
        .arg(&data)
        .env("ASPHODEL_LLM_ISSUER", &issuer.url));
    assert!(output.status.success(), "{}", stderr(&output));

    // The prompt names the URL and the code, and nothing secret.
    let shown = format!("{}{}", stdout(&output), stderr(&output));
    assert!(
        shown.contains(&format!("{}/codex/device", issuer.url)),
        "{shown}"
    );
    assert!(shown.contains("ABCD-EFGH"), "{shown}");
    assert!(!shown.contains("rt-first"), "{shown}");
    assert!(!shown.contains("eyJ"), "{shown}");

    let token_file = data.join(TOKEN_FILE);
    assert!(token_file.exists());
    assert_eq!(
        std::fs::metadata(&token_file).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let saved: Value =
        serde_json::from_str(&std::fs::read_to_string(&token_file).unwrap()).unwrap();
    assert_eq!(saved["refresh_token"], "rt-first");
    assert_eq!(saved["account_id"], "acct_7f3a9c");

    assert_eq!(
        issuer.paths(),
        [
            "/api/accounts/deviceauth/usercode",
            "/api/accounts/deviceauth/token",
            "/api/accounts/deviceauth/token",
            "/api/accounts/deviceauth/token",
            "/oauth/token",
        ]
    );
    let requests = issuer.requests.lock().unwrap().clone();
    assert_eq!(
        serde_json::from_str::<Value>(&requests[0].1).unwrap(),
        json!({"client_id": CLIENT_ID})
    );
    assert!(
        requests[4].1.contains("grant_type=authorization_code"),
        "{}",
        requests[4].1
    );
    assert!(
        requests[4].1.contains("code_verifier=verif"),
        "{}",
        requests[4].1
    );
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
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let output = run(asphodel()
        .args(["llm", "login", "--data-dir"])
        .arg(dir.data())
        .env("HOME", &home)
        .env("ASPHODEL_LLM_ISSUER", format!("http://127.0.0.1:{port}")));
    assert!(!output.status.success());
    assert!(
        !dir.data().join(TOKEN_FILE).exists(),
        "a token file appeared without a login"
    );
    assert!(
        !stderr(&output).contains("cli-refresh"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn serve_in_chatgpt_mode_starts_logged_out_and_says_so() {
    // The daemon starts without a login; extraction waits for one. The
    // resolved config shows the mode, the token file and the state, and
    // never a token.
    let dir = TestDir::new();
    let tuning = dir.chatgpt_tuning();
    let daemon = start(
        asphodel()
            .args(["serve", "--data-dir"])
            .arg(dir.data())
            .arg("--config")
            .arg(&tuning),
    );
    let config = resolved_config(&daemon.log);
    assert_eq!(config["llm"]["auth"], "chatgpt");
    assert_eq!(config["llm"]["logged_in"], false);
    assert_eq!(
        config["llm"]["token_file"],
        dir.data().join(TOKEN_FILE).to_str().unwrap()
    );
    assert_eq!(config["tuning"]["llm"]["auth"], "chatgpt");
    assert_eq!(
        config["tuning"]["llm"]["endpoint"],
        Value::Null,
        "the default isn't written back"
    );
    assert!(daemon.log.contains("asphodel llm login"), "{}", daemon.log);
}

#[test]
fn serve_refuses_a_key_together_with_chatgpt_mode() {
    let dir = TestDir::new();
    let tuning = dir.chatgpt_tuning();
    let output = run(asphodel()
        .args(["serve", "--data-dir"])
        .arg(dir.data())
        .arg("--config")
        .arg(&tuning)
        .args(["--listen", "127.0.0.1:0"])
        .env("ASPHODEL_MODELS", "fake")
        .env("ASPHODEL_LLM_API_KEY", "sk-live-41b2e8-secret"));
    assert!(!output.status.success());
    let stderr = stderr(&output);
    assert!(stderr.contains("ASPHODEL_LLM_API_KEY"), "{stderr}");
    assert!(stderr.contains("chatgpt"), "{stderr}");
    assert!(!stderr.contains("41b2e8"), "{stderr}");
}

#[test]
fn serve_in_chatgpt_mode_shows_logged_in_after_a_login_and_never_the_tokens() {
    let dir = TestDir::new();
    let data = dir.data();
    std::fs::write(
        data.join(TOKEN_FILE),
        json!({
            "access_token": jwt(json!({"sub": "fresh", "exp": 4_102_444_800i64})),
            "refresh_token": "rt-one",
            "id_token": jwt(json!({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_7f3a9c"}})),
            "account_id": "acct_7f3a9c",
            "last_refresh": "2026-03-02T09:00:00Z"
        })
        .to_string(),
    )
    .unwrap();
    let tuning = dir.chatgpt_tuning();
    let daemon = start(
        asphodel()
            .args(["serve", "--data-dir"])
            .arg(&data)
            .arg("--config")
            .arg(&tuning),
    );
    let config = resolved_config(&daemon.log);
    assert_eq!(config["llm"]["logged_in"], true);
    for forbidden in ["rt-one", "eyJ", "acct_7f3a9c"] {
        assert!(
            !daemon.log.contains(forbidden),
            "{forbidden} in log:\n{}",
            daemon.log
        );
    }
}
