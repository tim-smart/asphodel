//! The ChatGPT/Codex subscription mode of the LLM client, checked against
//! the scope addition to "Models: local embeddings, reranker and the
//! OpenAI-compatible LLM client" (TIM-105, comment `01a0f6c8`), "Replay
//! harness" (TIM-96, decisions 4 and 8) and ADR 0009.
//!
//! The protocol was read from `openai/codex` at `6b4daaf` (2026-10-01),
//! because the Codex backend is undocumented:
//!
//! - `codex-rs/login/src/device_code_auth.rs`: the device-code login.
//!   `POST {issuer}/api/accounts/deviceauth/usercode` with `{client_id}`
//!   gives `{device_auth_id, user_code, interval}`; the user opens
//!   `{issuer}/codex/device`; `POST {issuer}/api/accounts/deviceauth/token`
//!   with `{device_auth_id, user_code}` answers 403 or 404 while pending and
//!   `{authorization_code, code_challenge, code_verifier}` once approved;
//!   the code is exchanged at `POST {issuer}/oauth/token`, form-encoded,
//!   with `grant_type=authorization_code`, `client_id`, `code`,
//!   `redirect_uri={issuer}/deviceauth/callback` and `code_verifier`, for
//!   `{id_token, access_token, refresh_token}`.
//! - `codex-rs/login/src/auth/manager.rs` and `oauth/client.rs`: the
//!   client id `app_EMoamEEZ73f0CkXaXp7hrann`, the issuer
//!   `https://auth.openai.com`, and refresh as a JSON `POST
//!   {issuer}/oauth/token` with `{grant_type: "refresh_token", client_id,
//!   refresh_token}` answering `{id_token?, access_token?, refresh_token?}`.
//!   The access token is refreshed when its JWT `exp` is within 5 minutes,
//!   or when the last refresh is more than 8 days old.
//! - `codex-rs/login/src/token_data.rs`: the account id is the
//!   `chatgpt_account_id` claim under `https://api.openai.com/auth` in the
//!   id token.
//! - `codex-rs/model-provider-info/src/lib.rs`: the backend base URL
//!   `https://chatgpt.com/backend-api/codex`, and `codex-api/src/endpoint/
//!   responses.rs`: `POST /responses` with `Accept: text/event-stream`.
//! - `codex-rs/codex-api/src/common.rs`: the request body (`model`,
//!   `instructions`, `input`, `tools`, `tool_choice`, `parallel_tool_calls`,
//!   `store`, `stream`, `include`, `text.format` with `type: json_schema`,
//!   `strict`, `schema` and `name`). There is no `temperature` and no
//!   `max_tokens`.
//! - `codex-rs/codex-api/src/sse/responses.rs`: the events that matter are
//!   `response.output_item.done` (a `message` item with `output_text`
//!   content), `response.output_text.delta`, `response.completed` (with
//!   `response.usage.input_tokens` and `output_tokens`), `response.failed`
//!   and `response.incomplete`.
//! - `codex-rs/codex-api/src/api_bridge.rs`: a 429 whose body is
//!   `{"error": {"type": "usage_limit_reached", "resets_at": <unix
//!   seconds>, ...}}` is a usage limit, not a rate limit; the reset is an
//!   absolute time, also sent as the `x-codex-primary-reset-at` header.
//! - `codex-rs/login/src/auth/default_client.rs`: requests carry an
//!   `originator` header and a `User-Agent` built from it.
//!
//! The code under test doesn't exist yet. [`contract`] holds the proposed
//! additions to `asphodel_core::models` with `todo!()` bodies, so this file
//! compiles and every test that needs them is ignored. The types that
//! already exist (`LlmRequest`, `LlmResponse`, `LlmUsage`, `Template`,
//! `Secret`, `Tuning`, `Deployment`, the clock) are used from the crate.
//! Where the contract supersedes a crate type (`LlmSettings`, `LlmError`,
//! `LlmClient`), it is the same type with more variants or fields; on
//! activation the variants merge into the crate's type and the contract is
//! deleted.
//!
//! Nothing here touches the network. [`StubServer`] plays the Codex backend
//! and the auth issuer on loopback. The one exception is the ignored
//! [`real_backend_answers_a_structured_request`], which runs only when a
//! token file written by `asphodel llm login` is present.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use asphodel_core::clock::{Clock, SimulatedClock};
use asphodel_core::config::{Deployment, Secret, Tuning};
use asphodel_core::models::{LlmRequest, LlmUsage, Template};
use jiff::{SignedDuration, Timestamp};
use serde_json::{Value, json};

use contract::*;

/// The proposed additions to `asphodel_core::models` for the subscription
/// mode. Everything the API-key mode has stays as it is; this adds a second
/// auth mode behind the same `LlmClient` trait, so `FakeLlm` and the
/// cassette wrapper don't change (TIM-96, decision 4).
#[allow(dead_code, unused_variables)]
mod contract {
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Duration;

    use asphodel_core::clock::Clock;
    use asphodel_core::config::{Deployment, Secret, Tuning};
    use asphodel_core::models::{LlmRequest, LlmResponse};
    use jiff::Timestamp;
    use serde::Serialize;

    /// `[llm] auth`. `api_key` is the default and stays fully supported;
    /// `chatgpt` is the subscription, over the undocumented Codex backend.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum LlmAuth {
        #[default]
        ApiKey,
        Chatgpt,
    }

    /// The proposed `tuning.llm.auth`, parsed from `auth = "api_key" |
    /// "chatgpt"` with `api_key` as the default. Any other string is a
    /// config error.
    pub fn llm_auth(tuning: &Tuning) -> LlmAuth {
        todo!()
    }

    /// The Codex backend. The default endpoint in `chatgpt` mode; `llm.endpoint`
    /// still overrides it.
    pub const CODEX_ENDPOINT: &str = "https://chatgpt.com/backend-api/codex";

    /// The OAuth issuer for login and refresh.
    pub const AUTH_ISSUER: &str = "https://auth.openai.com";

    /// Codex's public OAuth client id.
    pub const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

    /// The `originator` header. Codex sends `codex_cli_rs`; Asphodel sends
    /// its own name, and the ignored real-backend test is what tells us
    /// whether the backend accepts it.
    pub const ORIGINATOR: &str = "asphodel";

    /// The token file under the data dir.
    pub const TOKEN_FILE: &str = "llm-tokens.json";

    /// Refresh the access token when its `exp` is this close.
    pub const REFRESH_WINDOW: Duration = Duration::from_secs(5 * 60);

    /// Where the LLM is and how to talk to it. Supersedes the crate's
    /// `LlmSettings` by adding `auth`.
    #[derive(Debug, Clone)]
    pub struct LlmSettings {
        pub auth: LlmAuth,
        /// The base URL. In `chatgpt` mode it defaults to [`CODEX_ENDPOINT`].
        pub endpoint: String,
        /// The exact model string, sent as `model` in both modes. Still
        /// pinned in `chatgpt` mode: calibration runs against one model
        /// (ADR 0009).
        pub model: String,
        /// `api_key` mode only.
        pub api_key: Option<Secret>,
        pub timeout: Duration,
    }

    impl LlmSettings {
        pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

        /// `api_key` mode behaves as today. In `chatgpt` mode the endpoint
        /// defaults to [`CODEX_ENDPOINT`], `llm.model` is still required,
        /// and an `ASPHODEL_LLM_API_KEY` that is also set is
        /// [`LlmError::Conflicting`] rather than silently ignored.
        pub fn from_config(
            tuning: &Tuning,
            deployment: &Deployment,
        ) -> Result<Option<Self>, LlmError> {
            todo!()
        }
    }

    /// Why a call failed. Supersedes the crate's `LlmError` by adding the
    /// last four variants. No variant carries a token, the prompt or the
    /// reply (TIM-96, decision 8).
    #[derive(Debug, thiserror::Error)]
    pub enum LlmError {
        #[error("the LLM isn't configured: {missing} is not set")]
        NotConfigured { missing: &'static str },

        #[error("LLM transport: {reason}")]
        Transport { reason: String },

        #[error("the LLM didn't answer within the timeout")]
        Timeout,

        #[error("the LLM answered HTTP {status}")]
        Status { status: u16 },

        #[error("the LLM returned no content")]
        NoContent,

        #[error("the LLM returned {bytes} bytes that aren't JSON")]
        NotJson { bytes: usize },

        #[error("the LLM refused the request")]
        Refused,

        /// Two settings that can't both hold, named by key.
        #[error("{first} and {second} are both set; unset one")]
        Conflicting {
            first: &'static str,
            second: &'static str,
        },

        /// No token file, or a refreshed credential the backend still
        /// rejects. The queue can't proceed until the owner logs in again.
        #[error("the ChatGPT login is missing or no longer valid: run `asphodel llm login`")]
        LoginRequired,

        /// The subscription's usage window is spent. Not a failure: the
        /// extraction queue holds until `resets_at` (world time).
        #[error("the ChatGPT usage limit is reached until {resets_at}")]
        UsageLimited { resets_at: Timestamp },

        /// The backend reported a failed or incomplete response. Only the
        /// code is kept.
        #[error("the LLM backend failed the request: {code}")]
        Backend { code: String },
    }

    impl LlmError {
        /// Transport errors, timeouts, 408, 429 and 5xx. `UsageLimited` is
        /// not retryable: it's deferred to `resets_at` instead.
        pub fn is_retryable(&self) -> bool {
            todo!()
        }
    }

    /// Same as the crate's trait, over the superset error.
    pub trait LlmClient: Send + Sync {
        fn model(&self) -> &str;
        fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError>;
    }

    // Tokens.

    /// What a login leaves behind. Every token is a [`Secret`]: none of
    /// them show in `Debug`, logs or serialised config.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ChatgptTokens {
        pub access_token: Secret,
        pub refresh_token: Secret,
        pub id_token: Secret,
        /// From the id token's `https://api.openai.com/auth.chatgpt_account_id`
        /// claim; sent as the `chatgpt-account-id` header.
        pub account_id: String,
        /// When these tokens were obtained or last refreshed.
        pub last_refresh: Timestamp,
    }

    impl ChatgptTokens {
        /// Builds the record from an exchange or refresh reply, reading the
        /// account id out of the id token.
        pub fn from_reply(
            id_token: &str,
            access_token: &str,
            refresh_token: &str,
            now: Timestamp,
        ) -> Result<Self, TokenError> {
            todo!()
        }

        /// The access token's `exp` claim, if it has one.
        pub fn access_expires_at(&self) -> Option<Timestamp> {
            todo!()
        }
    }

    #[derive(Debug, thiserror::Error)]
    pub enum TokenError {
        #[error("{}: {error}", path.display())]
        Io {
            path: PathBuf,
            #[source]
            error: std::io::Error,
        },

        #[error("{} isn't a token file: run `asphodel llm login`", path.display())]
        Malformed { path: PathBuf },

        #[error("the id token is not a JWT")]
        InvalidJwt,

        #[error("the id token has no chatgpt_account_id claim")]
        MissingAccountId,
    }

    /// The token file: `<data dir>/llm-tokens.json`, mode 0600, written
    /// through a temp file and a rename. `asphodel llm login` writes it,
    /// and the daemon reads it before every refresh, so a login while the
    /// daemon runs takes effect without a restart. The daemon serialises
    /// its own refreshes with a lock; the CLI never refreshes.
    pub struct TokenStore {
        path: PathBuf,
    }

    impl TokenStore {
        /// Never creates anything.
        pub fn open(data_dir: &Path) -> Self {
            todo!()
        }

        pub fn path(&self) -> &Path {
            todo!()
        }

        /// Re-reads the file every time. `Ok(None)` when it doesn't exist.
        pub fn load(&self) -> Result<Option<ChatgptTokens>, TokenError> {
            todo!()
        }

        pub fn save(&self, tokens: &ChatgptTokens) -> Result<(), TokenError> {
            todo!()
        }

        /// Removes the file. Removing a file that isn't there is not an
        /// error.
        pub fn clear(&self) -> Result<(), TokenError> {
            todo!()
        }

        /// What the resolved config shows: the path and whether a login is
        /// present. Never a token.
        pub fn status(&self) -> LlmStatus {
            todo!()
        }
    }

    /// The `llm` section of the resolved config.
    #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
    pub struct LlmStatus {
        pub auth: LlmAuth,
        pub token_file: Option<PathBuf>,
        pub logged_in: bool,
    }

    // Login.

    /// What the owner has to do: open the URL and enter the code.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct DeviceCode {
        pub verification_url: String,
        pub user_code: String,
        pub interval: Duration,
    }

    #[derive(Debug, thiserror::Error)]
    pub enum LoginError {
        #[error("login transport: {reason}")]
        Transport { reason: String },

        #[error("the issuer answered HTTP {status} at {step}")]
        Status { status: u16, step: &'static str },

        #[error("the login wasn't approved within 15 minutes")]
        Expired,

        #[error(transparent)]
        Token(#[from] TokenError),
    }

    /// `asphodel llm login`: the device-code flow, headless. `show` is
    /// called once with the URL and code; the function then polls every
    /// `interval` until the owner approves, exchanges the code for tokens,
    /// and saves them. It never reads `~/.codex/auth.json`: refresh tokens
    /// are single-use, so sharing a token chain with the Codex CLI would
    /// log one of them out.
    pub fn device_code_login(
        issuer: &str,
        store: &TokenStore,
        clock: &dyn Clock,
        show: &mut dyn FnMut(&DeviceCode),
    ) -> Result<ChatgptTokens, LoginError> {
        todo!()
    }

    // The client.

    /// The subscription client: the Responses API on the Codex backend,
    /// streamed, with a bearer access token from the [`TokenStore`].
    pub struct CodexResponses {}

    impl CodexResponses {
        /// `clock` decides when the access token is due for refresh
        /// (ADR 0004: nothing reads the wall clock).
        pub fn new(settings: LlmSettings, store: TokenStore, clock: Arc<dyn Clock>) -> Self {
            todo!()
        }

        /// The issuer for refresh. Defaults to [`AUTH_ISSUER`]; tests point
        /// it at a stub.
        pub fn with_issuer(self, issuer: &str) -> Self {
            todo!()
        }

        /// Refresh calls so far, for tests.
        pub fn refreshes(&self) -> usize {
            todo!()
        }
    }

    impl LlmClient for CodexResponses {
        fn model(&self) -> &str {
            todo!()
        }

        /// The sequence: load the token file; refresh first if the access
        /// token expires within [`REFRESH_WINDOW`]; post; on 401 refresh
        /// once (serialised, persisted before the retry) and retry once; a
        /// second 401, or a refresh the issuer rejects, is
        /// [`LlmError::LoginRequired`].
        fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
            todo!()
        }
    }
}

// Fixtures.

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-llm-chatgpt-{}-{}",
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
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn start() -> Timestamp {
    "2026-03-02T09:00:00Z".parse().unwrap()
}

fn clock() -> Arc<SimulatedClock> {
    Arc::new(SimulatedClock::new(start()))
}

fn deployment(llm_api_key: Option<&str>) -> Deployment {
    Deployment {
        listen: "127.0.0.1:7720".into(),
        data_dir: Some("/var/lib/asphodel".into()),
        config: None,
        allow_network_fs: false,
        model_dir: None,
        token: None,
        llm_api_key: llm_api_key.map(Secret::new),
    }
}

fn request() -> LlmRequest {
    LlmRequest {
        template: Template {
            name: "extract".into(),
            version: 3,
        },
        system: "You extract memories.".into(),
        user: "Tim said: I moved to Wellington in March.".into(),
        schema_name: "claims".into(),
        schema: json!({
            "type": "object",
            "properties": { "claims": { "type": "array", "items": { "type": "string" } } },
            "required": ["claims"],
            "additionalProperties": false
        }),
        max_tokens: Some(512),
    }
}

const ACCOUNT_ID: &str = "acct_7f3a9c";

/// URL-safe base64 without padding, as JWTs use.
fn base64url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let mut buffer = [0u8; 3];
        buffer[..chunk.len()].copy_from_slice(chunk);
        let n = u32::from_be_bytes([0, buffer[0], buffer[1], buffer[2]]);
        let count = chunk.len() + 1;
        for i in 0..count {
            let index = (n >> (18 - 6 * i)) & 0x3f;
            out.push(ALPHABET[index as usize] as char);
        }
    }
    out
}

/// An unsigned JWT with `claims` as its payload. The signature is never
/// checked client-side, so any bytes do.
fn jwt(claims: Value) -> String {
    format!(
        "{}.{}.{}",
        base64url(br#"{"alg":"RS256","typ":"JWT"}"#),
        base64url(claims.to_string().as_bytes()),
        base64url(b"signature")
    )
}

/// An access token that expires at `exp`.
fn access_token(label: &str, exp: Timestamp) -> String {
    jwt(json!({ "sub": label, "exp": exp.as_second() }))
}

/// An id token carrying the account id, as the issuer mints it.
fn id_token(account_id: &str) -> String {
    jwt(json!({
        "email": "tim@example.test",
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
            "chatgpt_plan_type": "plus"
        }
    }))
}

fn tokens(
    access_label: &str,
    refresh_label: &str,
    exp: Timestamp,
    now: Timestamp,
) -> ChatgptTokens {
    ChatgptTokens {
        access_token: Secret::new(access_token(access_label, exp)),
        refresh_token: Secret::new(format!("rt-{refresh_label}")),
        id_token: Secret::new(id_token(ACCOUNT_ID)),
        account_id: ACCOUNT_ID.into(),
        last_refresh: now,
    }
}

/// A store holding tokens whose access token is good for an hour.
fn logged_in_store(dir: &TestDir) -> TokenStore {
    let store = TokenStore::open(&dir.data());
    let exp = start()
        .checked_add(SignedDuration::from_secs(3600))
        .unwrap();
    store.save(&tokens("fresh", "one", exp, start())).unwrap();
    store
}

/// A store whose access token expired an hour ago.
fn expired_store(dir: &TestDir) -> TokenStore {
    let store = TokenStore::open(&dir.data());
    let exp = start()
        .checked_sub(SignedDuration::from_secs(3600))
        .unwrap();
    store.save(&tokens("stale", "one", exp, start())).unwrap();
    store
}

fn chatgpt_settings(endpoint: &str) -> LlmSettings {
    LlmSettings {
        auth: LlmAuth::Chatgpt,
        endpoint: endpoint.to_string(),
        model: "gpt-5.1".into(),
        api_key: None,
        timeout: Duration::from_secs(5),
    }
}

fn mode(bits: u32) -> String {
    format!("{bits:o}")
}

fn file_mode(path: &Path) -> u32 {
    std::fs::metadata(path).unwrap().permissions().mode() & 0o777
}

/// An SSE body: `response.output_item.done` with `text`, then
/// `response.completed` with usage.
fn sse_completion(text: &str) -> String {
    sse(&[
        (
            "response.created",
            json!({"type": "response.created", "response": {"id": "resp_1"}}),
        ),
        (
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}]
                }
            }),
        ),
        (
            "response.completed",
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "usage": {"input_tokens": 41, "output_tokens": 7, "total_tokens": 48}
                }
            }),
        ),
    ])
}

fn sse(events: &[(&str, Value)]) -> String {
    events
        .iter()
        .map(|(kind, data)| format!("event: {kind}\ndata: {data}\n\n"))
        .collect()
}

/// A loopback HTTP/1.1 server driven by a handler. Each connection is
/// answered and closed; every request is recorded in order.
struct StubServer {
    url: String,
    requests: Arc<Mutex<Vec<StubRequest>>>,
}

#[derive(Debug, Clone)]
struct StubRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: String,
}

impl StubRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    fn json(&self) -> Value {
        serde_json::from_str(&self.body).expect("a JSON body")
    }

    fn form(&self) -> BTreeMap<String, String> {
        self.body
            .split('&')
            .filter(|pair| !pair.is_empty())
            .map(|pair| {
                let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
                (percent_decode(key), percent_decode(value))
            })
            .collect()
    }

    fn bearer(&self) -> Option<&str> {
        self.header("authorization")?.strip_prefix("Bearer ")
    }
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("00");
                out.push(u8::from_str_radix(hex, 16).unwrap_or(b'?'));
                i += 2;
            }
            byte => out.push(byte),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[derive(Debug, Clone)]
struct StubResponse {
    status: u16,
    content_type: &'static str,
    headers: Vec<(String, String)>,
    body: String,
    /// Held before answering.
    delay: Duration,
}

impl StubResponse {
    fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            content_type: "application/json",
            headers: Vec::new(),
            body: value.to_string(),
            delay: Duration::ZERO,
        }
    }

    fn stream(body: String) -> Self {
        Self {
            status: 200,
            content_type: "text/event-stream",
            headers: Vec::new(),
            body,
            delay: Duration::ZERO,
        }
    }

    fn status(status: u16) -> Self {
        Self::json(status, json!({}))
    }

    fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    fn after(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }
}

type Handler = dyn Fn(&StubRequest) -> StubResponse + Send + Sync;

impl StubServer {
    fn start(handler: impl Fn(&StubRequest) -> StubResponse + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&requests);
        let handler: Arc<Handler> = Arc::new(handler);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let handler = Arc::clone(&handler);
                let log = Arc::clone(&log);
                std::thread::spawn(move || serve_one(stream, handler.as_ref(), &log));
            }
        });
        Self {
            url: format!("http://127.0.0.1:{port}"),
            requests,
        }
    }

    /// The Codex backend answering every request with `response`.
    fn backend(response: StubResponse) -> Self {
        Self::start(move |_| response.clone())
    }

    fn requests(&self) -> Vec<StubRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn paths(&self) -> Vec<String> {
        self.requests().into_iter().map(|r| r.path).collect()
    }

    fn only_request(&self) -> StubRequest {
        let requests = self.requests();
        assert_eq!(requests.len(), 1, "{requests:?}");
        requests.into_iter().next().unwrap()
    }
}

fn serve_one(mut stream: TcpStream, handler: &Handler, log: &Mutex<Vec<StubRequest>>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    let mut headers = Vec::new();
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
        if let Some((key, value)) = line.split_once(':') {
            let (key, value) = (key.trim().to_string(), value.trim().to_string());
            if key.eq_ignore_ascii_case("content-length") {
                length = value.parse().unwrap_or(0);
            }
            headers.push((key, value));
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    let request = StubRequest {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    };
    let response = handler(&request);
    log.lock().unwrap().push(request);
    std::thread::sleep(response.delay);
    let mut extra = String::new();
    for (name, value) in &response.headers {
        extra.push_str(&format!("{name}: {value}\r\n"));
    }
    let _ = write!(
        stream,
        "HTTP/1.1 {} Status\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n{extra}\r\n{}",
        response.status,
        response.content_type,
        response.body.len(),
        response.body
    );
    let _ = stream.flush();
}

/// A stub that is both the backend (`/responses`) and the issuer
/// (`/oauth/token`, `/api/accounts/deviceauth/*`), scripted by path and
/// call count.
type Script = dyn Fn(&StubRequest, usize) -> StubResponse + Send + Sync;

struct Scripted {
    calls: AtomicUsize,
    script: Box<Script>,
}

impl Scripted {
    fn server(
        script: impl Fn(&StubRequest, usize) -> StubResponse + Send + Sync + 'static,
    ) -> StubServer {
        let scripted = Arc::new(Scripted {
            calls: AtomicUsize::new(0),
            script: Box::new(script),
        });
        StubServer::start(move |request| {
            let n = scripted.calls.fetch_add(1, Ordering::SeqCst);
            (scripted.script)(request, n)
        })
    }
}

/// An issuer refresh reply rotating to `access`/`refresh`.
fn refresh_reply(access: &str, refresh: &str, exp: Timestamp) -> StubResponse {
    StubResponse::json(
        200,
        json!({
            "id_token": id_token(ACCOUNT_ID),
            "access_token": access_token(access, exp),
            "refresh_token": format!("rt-{refresh}"),
        }),
    )
}

/// A client on `server` for both the backend and the issuer.
fn client(server: &StubServer, store: TokenStore, clock: Arc<SimulatedClock>) -> CodexResponses {
    CodexResponses::new(chatgpt_settings(&server.url), store, clock).with_issuer(&server.url)
}

// What runs now.

#[test]
fn the_logical_request_carries_no_auth_and_no_wire_format() {
    // TIM-96, decision 4: the cassette records the logical request, so a
    // cassette recorded in one auth mode replays in the other. The request
    // type has exactly the six logical fields and nothing from the wire.
    let value = serde_json::to_value(request()).unwrap();
    let keys: Vec<_> = value.as_object().unwrap().keys().cloned().collect();
    assert_eq!(
        keys,
        [
            "max_tokens",
            "schema",
            "schema_name",
            "system",
            "template",
            "user"
        ]
    );
    let text = value.to_string();
    for forbidden in ["Bearer", "api_key", "access_token", "endpoint", "auth"] {
        assert!(!text.contains(forbidden), "{forbidden} in {text}");
    }
}

#[test]
fn a_secret_never_shows_in_debug_or_json() {
    let secret = Secret::new("rt-41b2e8-secret");
    assert_eq!(format!("{secret:?}"), "[redacted]");
    assert_eq!(serde_json::to_value(&secret).unwrap(), "[redacted]");
    assert_eq!(secret.expose(), "rt-41b2e8-secret");
}

#[test]
fn the_jwt_helper_round_trips_its_claims() {
    // The test's own JWT builder, checked once so the token tests can
    // trust it. The payload is the second dot-separated part.
    let token = id_token(ACCOUNT_ID);
    let payload = token.split('.').nth(1).unwrap();
    assert!(!payload.contains('='), "{payload}");
    // Decode by hand through the same alphabet.
    let decoded = {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        let mut bits = 0u32;
        let mut count = 0;
        let mut out = Vec::new();
        for byte in payload.bytes() {
            let value = ALPHABET.iter().position(|c| *c == byte).unwrap() as u32;
            bits = (bits << 6) | value;
            count += 6;
            if count >= 8 {
                count -= 8;
                out.push(((bits >> count) & 0xff) as u8);
            }
        }
        String::from_utf8(out).unwrap()
    };
    let claims: Value = serde_json::from_str(&decoded).unwrap();
    assert_eq!(
        claims["https://api.openai.com/auth"]["chatgpt_account_id"],
        ACCOUNT_ID
    );
}

#[test]
fn the_stub_server_streams_events_and_decodes_forms() {
    let server = StubServer::backend(StubResponse::stream(sse_completion("{\"ok\":true}")));
    let mut stream = TcpStream::connect(server.url.trim_start_matches("http://")).unwrap();
    let body = "grant_type=refresh_token&refresh_token=rt%2Done";
    write!(
        stream,
        "POST /oauth/token HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut reply = String::new();
    stream.read_to_string(&mut reply).unwrap();
    assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
    assert!(reply.contains("Content-Type: text/event-stream"), "{reply}");
    assert!(reply.contains("event: response.completed\n"), "{reply}");
    let form = server.only_request().form();
    assert_eq!(form["grant_type"], "refresh_token");
    assert_eq!(form["refresh_token"], "rt-one");
}

// Config: defaults and conflicts (ADR 0009).

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn the_auth_mode_defaults_to_api_key() {
    let tuning = Tuning::from_toml(
        "[llm]\nmodel = \"some-model\"\nendpoint = \"http://llm.internal:8080/v1\"\n",
    )
    .unwrap();
    assert_eq!(llm_auth(&tuning), LlmAuth::ApiKey);
    assert_eq!(llm_auth(&Tuning::default()), LlmAuth::ApiKey);

    let settings = LlmSettings::from_config(&tuning, &deployment(Some("sk-live-41b2e8-secret")))
        .unwrap()
        .expect("configured");
    assert_eq!(settings.auth, LlmAuth::ApiKey);
    assert_eq!(settings.endpoint, "http://llm.internal:8080/v1");
    assert_eq!(
        settings.api_key.as_ref().map(Secret::expose),
        Some("sk-live-41b2e8-secret")
    );
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn chatgpt_mode_defaults_to_the_codex_backend_and_still_pins_the_model() {
    let tuning = Tuning::from_toml("[llm]\nauth = \"chatgpt\"\nmodel = \"gpt-5.1\"\n").unwrap();
    assert_eq!(llm_auth(&tuning), LlmAuth::Chatgpt);
    let settings = LlmSettings::from_config(&tuning, &deployment(None))
        .unwrap()
        .expect("configured");
    assert_eq!(settings.auth, LlmAuth::Chatgpt);
    assert_eq!(settings.endpoint, CODEX_ENDPOINT);
    assert_eq!(settings.model, "gpt-5.1");
    assert!(settings.api_key.is_none());

    // The model stays required: calibration runs against one model, and
    // the subscription doesn't choose it for us (ADR 0009).
    let no_model = Tuning::from_toml("[llm]\nauth = \"chatgpt\"\n").unwrap();
    let error = LlmSettings::from_config(&no_model, &deployment(None)).unwrap_err();
    assert!(
        matches!(
            error,
            LlmError::NotConfigured {
                missing: "llm.model"
            }
        ),
        "{error:?}"
    );
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn an_explicit_endpoint_overrides_the_codex_default() {
    let tuning = Tuning::from_toml(
        "[llm]\nauth = \"chatgpt\"\nmodel = \"gpt-5.1\"\nendpoint = \"https://proxy.internal/codex\"\n",
    )
    .unwrap();
    let settings = LlmSettings::from_config(&tuning, &deployment(None))
        .unwrap()
        .expect("configured");
    assert_eq!(settings.endpoint, "https://proxy.internal/codex");
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn a_key_set_together_with_chatgpt_mode_is_a_config_error() {
    // Silently ignoring the key would hide a misconfiguration: the operator
    // thinks they're on the key, and the subscription is being billed.
    let tuning = Tuning::from_toml("[llm]\nauth = \"chatgpt\"\nmodel = \"gpt-5.1\"\n").unwrap();
    let error =
        LlmSettings::from_config(&tuning, &deployment(Some("sk-live-41b2e8-secret"))).unwrap_err();
    assert!(
        matches!(
            error,
            LlmError::Conflicting {
                first: "llm.auth = \"chatgpt\"",
                second: "ASPHODEL_LLM_API_KEY"
            }
        ),
        "{error:?}"
    );
    assert!(!error.is_retryable());
    assert!(!format!("{error}").contains("41b2e8"), "{error}");
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn an_unknown_auth_mode_is_rejected() {
    let error = Tuning::from_toml("[llm]\nauth = \"oauth\"\nmodel = \"gpt-5.1\"\n").unwrap_err();
    assert!(error.to_string().contains("auth"), "{error}");
}

// The token file.

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn tokens_are_built_from_a_login_reply_with_the_account_id_from_the_id_token() {
    let exp = start()
        .checked_add(SignedDuration::from_secs(3600))
        .unwrap();
    let tokens = ChatgptTokens::from_reply(
        &id_token(ACCOUNT_ID),
        &access_token("fresh", exp),
        "rt-one",
        start(),
    )
    .unwrap();
    assert_eq!(tokens.account_id, ACCOUNT_ID);
    assert_eq!(tokens.refresh_token.expose(), "rt-one");
    assert_eq!(tokens.last_refresh, start());
    assert_eq!(tokens.access_expires_at(), Some(exp));

    let no_claim = jwt(json!({"email": "tim@example.test"}));
    let error =
        ChatgptTokens::from_reply(&no_claim, &access_token("fresh", exp), "rt-one", start())
            .unwrap_err();
    assert!(matches!(error, TokenError::MissingAccountId), "{error:?}");

    let error = ChatgptTokens::from_reply("not.a.jwt.at.all", "x", "y", start()).unwrap_err();
    assert!(matches!(error, TokenError::InvalidJwt), "{error:?}");
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn the_token_file_lives_under_the_data_dir_with_mode_0600() {
    let dir = TestDir::new();
    let data = dir.data();
    let store = TokenStore::open(&data);
    assert_eq!(store.path(), data.join(TOKEN_FILE));
    assert_eq!(store.load().unwrap(), None);
    assert!(!store.path().exists(), "open created the file");

    let exp = start()
        .checked_add(SignedDuration::from_secs(3600))
        .unwrap();
    let saved = tokens("fresh", "one", exp, start());
    store.save(&saved).unwrap();
    assert_eq!(
        file_mode(store.path()),
        0o600,
        "mode {}",
        mode(file_mode(store.path()))
    );
    assert_eq!(store.load().unwrap(), Some(saved.clone()));

    // No temp file is left behind, and the file holds the raw tokens (it
    // is the one place they may be written).
    let entries: Vec<_> = std::fs::read_dir(&data)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    assert_eq!(entries, [TOKEN_FILE]);
    let text = std::fs::read_to_string(store.path()).unwrap();
    assert!(text.contains("rt-one"), "{text}");
    assert!(!text.contains("[redacted]"), "{text}");

    store.clear().unwrap();
    assert!(!store.path().exists());
    store.clear().unwrap();
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn a_malformed_token_file_names_the_path_and_the_fix() {
    let dir = TestDir::new();
    let store = TokenStore::open(&dir.data());
    std::fs::write(store.path(), "{\"access_token\": 42}").unwrap();
    let error = store.load().unwrap_err();
    assert!(
        matches!(&error, TokenError::Malformed { path } if path == store.path()),
        "{error:?}"
    );
    assert!(error.to_string().contains("asphodel llm login"), "{error}");
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn the_status_shows_the_path_and_whether_a_login_is_present_and_nothing_else() {
    let dir = TestDir::new();
    let store = TokenStore::open(&dir.data());
    let status = store.status();
    assert_eq!(status.auth, LlmAuth::Chatgpt);
    assert_eq!(status.token_file.as_deref(), Some(store.path()));
    assert!(!status.logged_in);

    let store = logged_in_store(&dir);
    let status = store.status();
    assert!(status.logged_in);
    let text = serde_json::to_string(&status).unwrap();
    assert!(text.contains("\"auth\":\"chatgpt\""), "{text}");
    assert!(text.contains("\"logged_in\":true"), "{text}");
    for forbidden in ["rt-one", "fresh", "eyJ", ACCOUNT_ID] {
        assert!(!text.contains(forbidden), "{forbidden} in {text}");
    }
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn debug_output_never_holds_a_token() {
    let dir = TestDir::new();
    let store = logged_in_store(&dir);
    let tokens = store.load().unwrap().unwrap();
    let shown = format!("{tokens:?}");
    assert!(shown.contains(ACCOUNT_ID), "{shown}");
    for forbidden in ["rt-one", "fresh", "eyJ"] {
        assert!(!shown.contains(forbidden), "{forbidden} in {shown}");
    }
    let settings = chatgpt_settings(CODEX_ENDPOINT);
    assert!(
        !format!("{settings:?}").contains("[redacted]"),
        "no key in chatgpt mode"
    );
}

// Login: the device-code flow against a stub issuer.

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn device_code_login_polls_exchanges_and_saves_tokens() {
    let exp = start()
        .checked_add(SignedDuration::from_secs(3600))
        .unwrap();
    let polls = Arc::new(AtomicUsize::new(0));
    let issuer = {
        let polls = Arc::clone(&polls);
        StubServer::start(move |request| match request.path.as_str() {
            "/api/accounts/deviceauth/usercode" => StubResponse::json(
                200,
                json!({"device_auth_id": "dev_123", "user_code": "ABCD-EFGH", "interval": "0"}),
            ),
            "/api/accounts/deviceauth/token" => {
                if polls.fetch_add(1, Ordering::SeqCst) < 2 {
                    StubResponse::status(403)
                } else {
                    StubResponse::json(
                        200,
                        json!({
                            "authorization_code": "code_9",
                            "code_challenge": "chal",
                            "code_verifier": "verif"
                        }),
                    )
                }
            }
            "/oauth/token" => StubResponse::json(
                200,
                json!({
                    "id_token": id_token(ACCOUNT_ID),
                    "access_token": access_token("first", exp),
                    "refresh_token": "rt-first"
                }),
            ),
            other => panic!("unexpected path {other}"),
        })
    };
    let dir = TestDir::new();
    let store = TokenStore::open(&dir.data());
    let clock = clock();
    let mut shown = Vec::new();

    let tokens = device_code_login(&issuer.url, &store, clock.as_ref(), &mut |code| {
        shown.push(code.clone())
    })
    .unwrap();

    // The owner saw the URL and code exactly once.
    assert_eq!(
        shown,
        [DeviceCode {
            verification_url: format!("{}/codex/device", issuer.url),
            user_code: "ABCD-EFGH".into(),
            interval: Duration::ZERO,
        }]
    );
    assert_eq!(tokens.account_id, ACCOUNT_ID);
    assert_eq!(tokens.refresh_token.expose(), "rt-first");
    assert_eq!(tokens.last_refresh, start());
    assert_eq!(store.load().unwrap(), Some(tokens));
    assert_eq!(file_mode(store.path()), 0o600);

    // The wire, as codex-rs sends it.
    let requests = issuer.requests();
    assert_eq!(
        requests.iter().map(|r| r.path.as_str()).collect::<Vec<_>>(),
        [
            "/api/accounts/deviceauth/usercode",
            "/api/accounts/deviceauth/token",
            "/api/accounts/deviceauth/token",
            "/api/accounts/deviceauth/token",
            "/oauth/token",
        ]
    );
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].json(), json!({"client_id": CLIENT_ID}));
    assert_eq!(
        requests[1].json(),
        json!({"device_auth_id": "dev_123", "user_code": "ABCD-EFGH"})
    );
    let exchange = &requests[4];
    assert!(
        exchange
            .header("content-type")
            .is_some_and(|value| value.starts_with("application/x-www-form-urlencoded")),
        "{:?}",
        exchange.headers
    );
    let form = exchange.form();
    assert_eq!(form["grant_type"], "authorization_code");
    assert_eq!(form["client_id"], CLIENT_ID);
    assert_eq!(form["code"], "code_9");
    assert_eq!(form["code_verifier"], "verif");
    assert_eq!(
        form["redirect_uri"],
        format!("{}/deviceauth/callback", issuer.url)
    );
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn a_login_the_issuer_refuses_is_an_error_without_a_token_file() {
    let issuer = StubServer::start(|request| match request.path.as_str() {
        "/api/accounts/deviceauth/usercode" => StubResponse::json(
            200,
            json!({"device_auth_id": "dev_123", "user_code": "ABCD-EFGH", "interval": "0"}),
        ),
        // Anything but 403/404 while polling is a failure, not "pending".
        "/api/accounts/deviceauth/token" => StubResponse::status(400),
        other => panic!("unexpected path {other}"),
    });
    let dir = TestDir::new();
    let store = TokenStore::open(&dir.data());
    let error = device_code_login(&issuer.url, &store, clock().as_ref(), &mut |_| {}).unwrap_err();
    assert!(
        matches!(
            error,
            LoginError::Status {
                status: 400,
                step: "deviceauth/token"
            }
        ),
        "{error:?}"
    );
    assert!(!store.path().exists());
}

// The Responses wire format.

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn the_client_streams_a_structured_responses_request_with_the_codex_headers() {
    let backend = StubServer::backend(StubResponse::stream(sse_completion(
        "{\"claims\":[\"Tim moved to Wellington in March 2026.\"]}",
    )));
    let dir = TestDir::new();
    let client = client(&backend, logged_in_store(&dir), clock());
    assert_eq!(client.model(), "gpt-5.1");

    let response = client.complete(&request()).unwrap();
    assert_eq!(
        response.json,
        json!({"claims": ["Tim moved to Wellington in March 2026."]})
    );
    assert_eq!(
        response.usage,
        Some(LlmUsage {
            input_tokens: 41,
            output_tokens: 7
        })
    );
    assert!(response.latency > Duration::ZERO);
    assert_eq!(client.refreshes(), 0);

    let sent = backend.only_request();
    assert_eq!(sent.method, "POST");
    assert_eq!(sent.path, "/responses");
    assert_eq!(
        sent.bearer(),
        Some(
            access_token(
                "fresh",
                start()
                    .checked_add(SignedDuration::from_secs(3600))
                    .unwrap()
            )
            .as_str()
        )
    );
    assert_eq!(sent.header("chatgpt-account-id"), Some(ACCOUNT_ID));
    assert_eq!(sent.header("originator"), Some(ORIGINATOR));
    assert!(
        sent.header("user-agent")
            .is_some_and(|ua| ua.starts_with(ORIGINATOR)),
        "{:?}",
        sent.headers
    );
    assert!(
        sent.header("accept")
            .is_some_and(|accept| accept.contains("text/event-stream")),
        "{:?}",
        sent.headers
    );
    assert!(
        sent.header("content-type")
            .is_some_and(|value| value.starts_with("application/json")),
        "{:?}",
        sent.headers
    );

    let body = sent.json();
    assert_eq!(body["model"], "gpt-5.1");
    assert_eq!(body["instructions"], "You extract memories.");
    assert_eq!(
        body["input"],
        json!([{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Tim said: I moved to Wellington in March."}]
        }])
    );
    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], true);
    assert_eq!(body["tools"], json!([]));
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["parallel_tool_calls"], false);
    assert_eq!(body["include"], json!([]));
    assert_eq!(body["text"]["format"]["type"], "json_schema");
    assert_eq!(body["text"]["format"]["name"], "claims");
    assert_eq!(body["text"]["format"]["strict"], true);
    assert_eq!(body["text"]["format"]["schema"], request().schema);
    // GPT-5 rejects temperature, and the Responses API has no max_tokens;
    // neither is sent. Chat Completions' response_format isn't either.
    for absent in ["temperature", "max_tokens", "response_format", "messages"] {
        assert!(body.get(absent).is_none(), "{absent} in {body}");
    }
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn deltas_are_reassembled_when_no_done_item_carries_the_text() {
    let backend = StubServer::backend(StubResponse::stream(sse(&[
        (
            "response.output_text.delta",
            json!({"type": "response.output_text.delta", "delta": "{\"cla"}),
        ),
        (
            "response.output_text.delta",
            json!({"type": "response.output_text.delta", "delta": "ims\":[]}"}),
        ),
        (
            "response.completed",
            json!({"type": "response.completed", "response": {"id": "resp_2", "status": "completed"}}),
        ),
    ])));
    let dir = TestDir::new();
    let response = client(&backend, logged_in_store(&dir), clock())
        .complete(&request())
        .unwrap();
    assert_eq!(response.json, json!({"claims": []}));
    assert_eq!(response.usage, None, "no usage in the completed event");
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn fenced_json_in_the_streamed_text_is_unwrapped() {
    let backend = StubServer::backend(StubResponse::stream(sse_completion(
        "```json\n{\"claims\": []}\n```",
    )));
    let dir = TestDir::new();
    let response = client(&backend, logged_in_store(&dir), clock())
        .complete(&request())
        .unwrap();
    assert_eq!(response.json, json!({"claims": []}));
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn streamed_text_that_is_not_json_is_an_error_that_carries_only_its_size() {
    let content = "Sure! Here are Tim's claims: he moved to Wellington.";
    let backend = StubServer::backend(StubResponse::stream(sse_completion(content)));
    let dir = TestDir::new();
    let error = client(&backend, logged_in_store(&dir), clock())
        .complete(&request())
        .unwrap_err();
    assert!(
        matches!(error, LlmError::NotJson { bytes } if bytes == content.len()),
        "{error:?}"
    );
    assert!(!format!("{error} {error:?}").contains("Wellington"));
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn a_failed_or_incomplete_response_keeps_only_its_code() {
    let backend = StubServer::backend(StubResponse::stream(sse(&[(
        "response.failed",
        json!({
            "type": "response.failed",
            "response": {
                "id": "resp_3",
                "status": "failed",
                "error": {"code": "server_error", "message": "Tim's claims could not be processed"}
            }
        }),
    )])));
    let dir = TestDir::new();
    let error = client(&backend, logged_in_store(&dir), clock())
        .complete(&request())
        .unwrap_err();
    assert!(
        matches!(&error, LlmError::Backend { code } if code == "server_error"),
        "{error:?}"
    );
    assert!(!format!("{error} {error:?}").contains("Tim"), "{error}");

    let backend = StubServer::backend(StubResponse::stream(sse(&[(
        "response.incomplete",
        json!({
            "type": "response.incomplete",
            "response": {"id": "resp_4", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        }),
    )])));
    let error = client(&backend, logged_in_store(&dir), clock())
        .complete(&request())
        .unwrap_err();
    assert!(
        matches!(&error, LlmError::Backend { code } if code == "max_output_tokens"),
        "{error:?}"
    );

    // A stream that ends without completing.
    let backend = StubServer::backend(StubResponse::stream(sse(&[(
        "response.created",
        json!({"type": "response.created", "response": {"id": "resp_5"}}),
    )])));
    let error = client(&backend, logged_in_store(&dir), clock())
        .complete(&request())
        .unwrap_err();
    assert!(matches!(error, LlmError::NoContent), "{error:?}");
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn without_a_token_file_the_client_asks_for_a_login_before_any_request() {
    let backend = StubServer::backend(StubResponse::stream(sse_completion("{}")));
    let dir = TestDir::new();
    let store = TokenStore::open(&dir.data());
    let error = client(&backend, store, clock())
        .complete(&request())
        .unwrap_err();
    assert!(matches!(error, LlmError::LoginRequired), "{error:?}");
    assert!(error.to_string().contains("asphodel llm login"), "{error}");
    assert!(!error.is_retryable());
    assert!(
        backend.requests().is_empty(),
        "a request went out without a login"
    );
}

// Refresh.

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn an_expired_access_token_is_refreshed_first_and_the_rotation_is_persisted_before_the_request() {
    let dir = TestDir::new();
    let store = expired_store(&dir);
    let token_path = store.path().to_path_buf();
    let exp = start()
        .checked_add(SignedDuration::from_secs(3600))
        .unwrap();
    // The backend checks, at the moment the request arrives, that the file
    // already holds the rotated refresh token.
    let file_at_request = Arc::new(Mutex::new(String::new()));
    let seen = Arc::clone(&file_at_request);
    let server = Scripted::server(move |request, _| match request.path.as_str() {
        "/oauth/token" => refresh_reply("rotated", "two", exp),
        "/responses" => {
            *seen.lock().unwrap() = std::fs::read_to_string(&token_path).unwrap();
            StubResponse::stream(sse_completion("{\"claims\":[]}"))
        }
        other => panic!("unexpected path {other}"),
    });
    let clock = clock();
    let client = client(&server, store, Arc::clone(&clock));

    let response = client.complete(&request()).unwrap();
    assert_eq!(response.json, json!({"claims": []}));
    assert_eq!(client.refreshes(), 1);
    assert_eq!(server.paths(), ["/oauth/token", "/responses"]);

    // The refresh, as codex-rs sends it: JSON, with the client id.
    let refresh = &server.requests()[0];
    assert!(
        refresh
            .header("content-type")
            .is_some_and(|value| value.starts_with("application/json")),
        "{:?}",
        refresh.headers
    );
    assert_eq!(
        refresh.json(),
        json!({"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": "rt-one"})
    );
    assert_eq!(refresh.header("authorization"), None);

    // Persisted before the retried request went out, and the request used
    // the new access token.
    let at_request = file_at_request.lock().unwrap().clone();
    assert!(at_request.contains("rt-two"), "{at_request}");
    assert!(!at_request.contains("rt-one"), "{at_request}");
    assert_eq!(
        server.requests()[1].bearer(),
        Some(access_token("rotated", exp).as_str())
    );

    let saved = TokenStore::open(&dir.data()).load().unwrap().unwrap();
    assert_eq!(saved.refresh_token.expose(), "rt-two");
    assert_eq!(saved.account_id, ACCOUNT_ID);
    assert_eq!(saved.last_refresh, clock.now());
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn a_token_inside_the_refresh_window_counts_as_expired() {
    // codex-rs refreshes when exp is within 5 minutes.
    let dir = TestDir::new();
    let store = TokenStore::open(&dir.data());
    let soon = start()
        .checked_add(SignedDuration::from_secs(4 * 60))
        .unwrap();
    store.save(&tokens("soon", "one", soon, start())).unwrap();
    let later = start()
        .checked_add(SignedDuration::from_secs(3600))
        .unwrap();
    let server = Scripted::server(move |request, _| match request.path.as_str() {
        "/oauth/token" => refresh_reply("rotated", "two", later),
        _ => StubResponse::stream(sse_completion("{}")),
    });
    client(&server, store, clock())
        .complete(&request())
        .unwrap();
    assert_eq!(server.paths(), ["/oauth/token", "/responses"]);
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn a_401_triggers_one_refresh_and_one_retry() {
    let dir = TestDir::new();
    let store = logged_in_store(&dir);
    let exp = start()
        .checked_add(SignedDuration::from_secs(3600))
        .unwrap();
    let server = Scripted::server(move |request, n| match (request.path.as_str(), n) {
        ("/responses", 0) => StubResponse::status(401),
        ("/oauth/token", _) => refresh_reply("rotated", "two", exp),
        ("/responses", _) => StubResponse::stream(sse_completion("{\"claims\":[]}")),
        other => panic!("unexpected {other:?}"),
    });
    let client = client(&server, store, clock());
    let response = client.complete(&request()).unwrap();
    assert_eq!(response.json, json!({"claims": []}));
    assert_eq!(client.refreshes(), 1);
    assert_eq!(server.paths(), ["/responses", "/oauth/token", "/responses"]);
    assert_eq!(
        server.requests()[2].bearer(),
        Some(access_token("rotated", exp).as_str())
    );
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn a_401_after_a_refresh_asks_for_a_login_and_stops() {
    let dir = TestDir::new();
    let store = logged_in_store(&dir);
    let exp = start()
        .checked_add(SignedDuration::from_secs(3600))
        .unwrap();
    let server = Scripted::server(move |request, _| match request.path.as_str() {
        "/responses" => StubResponse::status(401),
        "/oauth/token" => refresh_reply("rotated", "two", exp),
        other => panic!("unexpected path {other}"),
    });
    let client = client(&server, store, clock());
    let error = client.complete(&request()).unwrap_err();
    assert!(matches!(error, LlmError::LoginRequired), "{error:?}");
    assert!(error.to_string().contains("asphodel llm login"), "{error}");
    assert!(!error.is_retryable());
    // Exactly one refresh and one retry, never a loop.
    assert_eq!(server.paths(), ["/responses", "/oauth/token", "/responses"]);
    assert_eq!(client.refreshes(), 1);
    // The rotated tokens were still persisted: the refresh itself succeeded.
    let saved = TokenStore::open(&dir.data()).load().unwrap().unwrap();
    assert_eq!(saved.refresh_token.expose(), "rt-two");
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn a_refresh_the_issuer_rejects_asks_for_a_login_and_keeps_the_file() {
    let dir = TestDir::new();
    let store = expired_store(&dir);
    let server = Scripted::server(|request, _| match request.path.as_str() {
        "/oauth/token" => StubResponse::json(
            400,
            json!({"error": "invalid_grant", "error_description": "refresh token rt-one is revoked"}),
        ),
        other => panic!("unexpected path {other}"),
    });
    let client = client(&server, store, clock());
    let error = client.complete(&request()).unwrap_err();
    assert!(matches!(error, LlmError::LoginRequired), "{error:?}");
    assert!(!format!("{error:?}").contains("rt-one"), "{error:?}");
    assert_eq!(server.paths(), ["/oauth/token"]);
    // The old file stays, so the owner can see they were logged in and the
    // next login overwrites it.
    assert!(TokenStore::open(&dir.data()).path().exists());
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn a_new_login_written_while_running_is_picked_up_without_a_restart() {
    // Re-login after a failed credential: `asphodel llm login` writes the
    // file; the daemon's next call reads it instead of refreshing.
    let dir = TestDir::new();
    let store = expired_store(&dir);
    let data = dir.data();
    let exp = start()
        .checked_add(SignedDuration::from_secs(3600))
        .unwrap();
    let server = Scripted::server(|request, _| match request.path.as_str() {
        "/oauth/token" => StubResponse::status(401),
        "/responses" => StubResponse::stream(sse_completion("{}")),
        other => panic!("unexpected path {other}"),
    });
    let client = client(&server, store, clock());
    assert!(matches!(
        client.complete(&request()),
        Err(LlmError::LoginRequired)
    ));

    // The CLI logs in: a fresh file, written by another TokenStore.
    TokenStore::open(&data)
        .save(&tokens("relogin", "three", exp, start()))
        .unwrap();

    client.complete(&request()).unwrap();
    assert_eq!(server.paths(), ["/oauth/token", "/responses"]);
    assert_eq!(
        server.requests()[1].bearer(),
        Some(access_token("relogin", exp).as_str())
    );
    assert_eq!(
        client.refreshes(),
        1,
        "no second refresh: the new file was used as is"
    );
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn concurrent_calls_share_one_refresh() {
    // Refresh tokens are single-use. Two threads with the same expired
    // token must not both refresh: the second would present a token the
    // first already spent and log the daemon out.
    let dir = TestDir::new();
    let store = expired_store(&dir);
    let exp = start()
        .checked_add(SignedDuration::from_secs(3600))
        .unwrap();
    let server = Scripted::server(move |request, _| match request.path.as_str() {
        "/oauth/token" => refresh_reply("rotated", "two", exp).after(Duration::from_millis(300)),
        "/responses" => StubResponse::stream(sse_completion("{}")),
        other => panic!("unexpected path {other}"),
    });
    let client = Arc::new(client(&server, store, clock()));
    let workers: Vec<_> = (0..3)
        .map(|_| {
            let client = Arc::clone(&client);
            std::thread::spawn(move || client.complete(&request()).map(|r| r.json))
        })
        .collect();
    for worker in workers {
        assert_eq!(worker.join().unwrap().unwrap(), json!({}));
    }
    let paths = server.paths();
    assert_eq!(
        paths.iter().filter(|p| *p == "/oauth/token").count(),
        1,
        "{paths:?}"
    );
    assert_eq!(
        paths.iter().filter(|p| *p == "/responses").count(),
        3,
        "{paths:?}"
    );
    assert_eq!(client.refreshes(), 1);
    for request in server.requests().iter().filter(|r| r.path == "/responses") {
        assert_eq!(
            request.bearer(),
            Some(access_token("rotated", exp).as_str())
        );
    }
}

// Usage limits (reset-aware 429).

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn a_usage_limit_is_deferred_to_its_reset_time_not_retried() {
    let resets_at: Timestamp = "2026-03-02T14:30:00Z".parse().unwrap();
    let backend = StubServer::backend(
        StubResponse::json(
            429,
            json!({
                "error": {
                    "type": "usage_limit_reached",
                    "message": "You've hit your usage limit.",
                    "plan_type": "plus",
                    "resets_at": resets_at.as_second()
                }
            }),
        )
        .with_header(
            "x-codex-primary-reset-at",
            &resets_at.as_second().to_string(),
        ),
    );
    let dir = TestDir::new();
    let error = client(&backend, logged_in_store(&dir), clock())
        .complete(&request())
        .unwrap_err();
    assert!(
        matches!(error, LlmError::UsageLimited { resets_at: at } if at == resets_at),
        "{error:?}"
    );
    assert!(!error.is_retryable(), "deferral, not a retry");
    assert!(
        error.to_string().contains("2026-03-02T14:30:00Z"),
        "{error}"
    );
    assert_eq!(backend.requests().len(), 1);
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn the_reset_header_fills_in_when_the_body_has_no_reset() {
    let backend = StubServer::backend(
        StubResponse::json(429, json!({"error": {"type": "usage_limit_reached"}}))
            .with_header("x-codex-primary-reset-at", "1772462400"),
    );
    let dir = TestDir::new();
    let error = client(&backend, logged_in_store(&dir), clock())
        .complete(&request())
        .unwrap_err();
    let expected = Timestamp::from_second(1_772_462_400).unwrap();
    assert!(
        matches!(error, LlmError::UsageLimited { resets_at } if resets_at == expected),
        "{error:?}"
    );
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn a_plain_429_stays_a_retryable_status() {
    // A rate limit without a usage window is the old behaviour: retry.
    let backend = StubServer::backend(StubResponse::json(
        429,
        json!({"error": {"type": "rate_limit_exceeded", "message": "slow down"}}),
    ));
    let dir = TestDir::new();
    let error = client(&backend, logged_in_store(&dir), clock())
        .complete(&request())
        .unwrap_err();
    assert!(
        matches!(error, LlmError::Status { status: 429 }),
        "{error:?}"
    );
    assert!(error.is_retryable());
}

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn other_statuses_map_as_in_api_key_mode() {
    for (status, retryable) in [(400, false), (403, false), (500, true), (503, true)] {
        let backend = StubServer::backend(StubResponse::status(status));
        let dir = TestDir::new();
        let error = client(&backend, logged_in_store(&dir), clock())
            .complete(&request())
            .unwrap_err();
        assert!(
            matches!(error, LlmError::Status { status: got } if got == status),
            "{status}: {error:?}"
        );
        assert_eq!(error.is_retryable(), retryable, "{status}");
        assert_eq!(backend.requests().len(), 1, "{status}: retried");
    }
}

// Secret hygiene across the whole client.

#[test]
#[ignore = "needs the chatgpt auth mode"]
fn no_error_or_response_carries_a_token() {
    let dir = TestDir::new();
    let store = expired_store(&dir);
    let exp = start()
        .checked_add(SignedDuration::from_secs(3600))
        .unwrap();
    let server = Scripted::server(move |request, n| match (request.path.as_str(), n) {
        ("/oauth/token", _) => refresh_reply("rotated", "two", exp),
        ("/responses", 1) => StubResponse::status(503),
        ("/responses", _) => StubResponse::stream(sse_completion("{\"ok\":true}")),
        other => panic!("unexpected {other:?}"),
    });
    let client = client(&server, store, clock());
    let error = client.complete(&request()).unwrap_err();
    let response = client.complete(&request()).unwrap();
    let shown = format!(
        "{error} {error:?} {:?} {}",
        response,
        serde_json::to_string(&response).unwrap()
    );
    for forbidden in ["rt-one", "rt-two", "rotated", "stale", "eyJ", "Bearer"] {
        assert!(!shown.contains(forbidden), "{forbidden} in {shown}");
    }
}

// The real backend. Ignored: it runs only with a token file from
// `asphodel llm login`.

#[test]
#[ignore = "needs a ChatGPT login: run `asphodel llm login --data-dir <dir>` and set ASPHODEL_DATA_DIR"]
fn real_backend_answers_a_structured_request() {
    let data_dir = std::env::var_os("ASPHODEL_DATA_DIR")
        .map(PathBuf::from)
        .expect("ASPHODEL_DATA_DIR");
    let store = TokenStore::open(&data_dir);
    assert!(
        store.load().expect("a readable token file").is_some(),
        "no token file at {}: run `asphodel llm login`",
        store.path().display()
    );
    let model = std::env::var("ASPHODEL_LLM_MODEL").unwrap_or_else(|_| "gpt-5.1".into());
    let settings = LlmSettings {
        auth: LlmAuth::Chatgpt,
        endpoint: CODEX_ENDPOINT.into(),
        model,
        api_key: None,
        timeout: Duration::from_secs(120),
    };
    let client = CodexResponses::new(settings, store, Arc::new(asphodel_core::SystemClock));

    let mut request = request();
    request.system =
        "Answer with JSON matching the schema. The claims are the facts stated.".into();
    request.user = "The user said: my sister Maya lives in Auckland.".into();
    let response = client
        .complete(&request)
        .unwrap_or_else(|error| panic!("{error}"));
    let claims = response.json["claims"].as_array().expect("claims array");
    assert!(!claims.is_empty(), "{}", response.json);
    assert!(claims.iter().all(Value::is_string), "{}", response.json);
    assert!(
        response
            .usage
            .is_some_and(|u| u.input_tokens > 0 && u.output_tokens > 0)
    );
    assert!(response.latency > Duration::ZERO);

    // Whatever refresh happened, the file still parses and keeps the
    // account id.
    let saved = TokenStore::open(&data_dir).load().unwrap().unwrap();
    assert!(!saved.account_id.is_empty());
}
