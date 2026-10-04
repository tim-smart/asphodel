//! The ChatGPT/Codex subscription mode of the LLM client.
//!
//! A subscription authenticates with ChatGPT OAuth tokens and talks to the
//! Codex backend's Responses API, streamed. The protocol was read from
//! `openai/codex` at `6b4daaf` on 2026-10-01, because the backend is
//! undocumented; see `docs/models.md` for the parts that matter and what is
//! still unverified against the real backend.
//!
//! Asphodel has its own login (`asphodel llm login`, the device-code flow)
//! and its own token file under the data dir. It never reads
//! `~/.codex/auth.json`: refresh tokens are single-use, so sharing a token
//! chain with the Codex CLI would log one of the two out.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use jiff::{SignedDuration, Timestamp};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::llm::{LlmClient, LlmError, LlmRequest, LlmResponse, LlmSettings, LlmUsage, unfence};
use crate::clock::Clock;
use crate::config::{LlmAuth, Secret};

/// The Codex backend: the default endpoint in `chatgpt` mode. `llm.endpoint`
/// still overrides it.
pub const CODEX_ENDPOINT: &str = "https://chatgpt.com/backend-api/codex";

/// The OAuth issuer for login and refresh.
pub const AUTH_ISSUER: &str = "https://auth.openai.com";

/// Codex's public OAuth client id.
pub const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// The `originator` header. Codex sends `codex_cli_rs`; Asphodel sends its
/// own name. Whether the backend accepts it is unverified until the ignored
/// real-backend test runs.
pub const ORIGINATOR: &str = "asphodel";

/// The token file under the data dir.
pub const TOKEN_FILE: &str = "llm-tokens.json";

/// The token store's lock file, next to [`TOKEN_FILE`]. It holds nothing;
/// an advisory `flock` on it serialises every write of the token file.
pub const TOKEN_LOCK_FILE: &str = "llm-tokens.lock";

/// Refresh the access token when its `exp` is this close (codex-rs uses the
/// same window).
pub const REFRESH_WINDOW: Duration = Duration::from_secs(5 * 60);

/// Without an `exp` claim, refresh once the last refresh is this old.
const REFRESH_AFTER: Duration = Duration::from_secs(8 * 24 * 60 * 60);

/// How long a login waits for the owner's approval.
const LOGIN_WAIT: Duration = Duration::from_secs(15 * 60);

/// Per-request timeout for the issuer.
const ISSUER_TIMEOUT: Duration = Duration::from_secs(30);

/// Replies past this size aren't read.
const REPLY_LIMIT: u64 = 16 * 1024 * 1024;

// Tokens.

/// What a login leaves behind. Every token is a [`Secret`]: none of them
/// show in `Debug`, logs or serialised config.
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
        let account_id = account_id_from(id_token)?;
        Ok(Self {
            access_token: Secret::new(access_token),
            refresh_token: Secret::new(refresh_token),
            id_token: Secret::new(id_token),
            account_id,
            last_refresh: now,
        })
    }

    /// The access token's `exp` claim, if it has one.
    pub fn access_expires_at(&self) -> Option<Timestamp> {
        let claims = jwt_claims(self.access_token.expose()).ok()?;
        let exp = claims.get("exp")?.as_i64()?;
        Timestamp::from_second(exp).ok()
    }

    /// Whether the client should refresh before using the access token:
    /// `exp` within [`REFRESH_WINDOW`], or no `exp` and a last refresh
    /// older than eight days.
    fn due_for_refresh(&self, now: Timestamp) -> bool {
        match self.access_expires_at() {
            Some(exp) => {
                now.duration_until(exp)
                    <= SignedDuration::try_from(REFRESH_WINDOW).unwrap_or_default()
            }
            None => {
                self.last_refresh.duration_until(now)
                    >= SignedDuration::try_from(REFRESH_AFTER).unwrap_or_default()
            }
        }
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

impl From<TokenError> for LlmError {
    fn from(error: TokenError) -> Self {
        match error {
            TokenError::Io { path, error } => LlmError::Transport {
                reason: format!("token file {}: {error}", path.display()),
            },
            TokenError::Malformed { .. }
            | TokenError::InvalidJwt
            | TokenError::MissingAccountId => LlmError::LoginRequired,
        }
    }
}

/// The on-disk shape. The one place tokens are written in the clear.
#[derive(Serialize, Deserialize)]
struct TokenFile {
    access_token: String,
    refresh_token: String,
    id_token: String,
    account_id: String,
    last_refresh: Timestamp,
}

/// The token file: `<data dir>/llm-tokens.json`, mode 0600, written through
/// a new temp file and a rename. `asphodel llm login` writes it, and the
/// daemon reads it before every call and every refresh, so a login while
/// the daemon runs takes effect without a restart.
///
/// Every write goes through [`TokenLock`], an exclusive `flock` on
/// `<data dir>/llm-tokens.lock`. A refresh holds it from its re-read of the
/// file through the issuer exchange to its save, so a login or a clear,
/// from this process or another, waits for the refresh and then replaces
/// what it wrote instead of being overwritten by it. Two clients on one
/// store share a refresh the same way: the second finds the rotated tokens
/// on its re-read. Reads take no lock; the rename makes them atomic.
pub struct TokenStore {
    path: PathBuf,
    lock_path: PathBuf,
}

impl TokenStore {
    /// Never creates anything.
    pub fn open(data_dir: &Path) -> Self {
        Self {
            path: data_dir.join(TOKEN_FILE),
            lock_path: data_dir.join(TOKEN_LOCK_FILE),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Re-reads the file every time. `Ok(None)` when it doesn't exist.
    pub fn load(&self) -> Result<Option<ChatgptTokens>, TokenError> {
        let text = match std::fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(TokenError::Io {
                    path: self.path.clone(),
                    error,
                });
            }
        };
        let file: TokenFile = serde_json::from_str(&text).map_err(|_| TokenError::Malformed {
            path: self.path.clone(),
        })?;
        Ok(Some(ChatgptTokens {
            access_token: Secret::new(file.access_token),
            refresh_token: Secret::new(file.refresh_token),
            id_token: Secret::new(file.id_token),
            account_id: file.account_id,
            last_refresh: file.last_refresh,
        }))
    }

    /// Takes the store's lock, waiting for whoever holds it. Hold it only
    /// across work that must not interleave with another write; never
    /// across a wait for the owner, such as a device-code approval.
    pub fn lock(&self) -> Result<TokenLock<'_>, TokenError> {
        use std::os::unix::fs::OpenOptionsExt;
        let io = |error| TokenError::Io {
            path: self.lock_path.clone(),
            error,
        };
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(&self.lock_path)
            .map_err(io)?;
        file.lock().map_err(io)?;
        Ok(TokenLock {
            store: self,
            _file: file,
        })
    }

    /// Replaces the file under the lock, after any refresh in flight.
    pub fn save(&self, tokens: &ChatgptTokens) -> Result<(), TokenError> {
        self.lock()?.save(tokens)
    }

    /// Removes the file under the lock, after any refresh in flight, so the
    /// refresh can't write the tokens back. Removing a file that isn't
    /// there, or from a data dir that doesn't exist, is not an error.
    pub fn clear(&self) -> Result<(), TokenError> {
        match self.lock() {
            Ok(lock) => lock.clear(),
            Err(TokenError::Io { error, .. }) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    /// What the resolved config shows: the path and whether a login is
    /// present. Never a token.
    pub fn status(&self) -> LlmStatus {
        LlmStatus {
            auth: LlmAuth::Chatgpt,
            token_file: Some(self.path.clone()),
            logged_in: matches!(self.load(), Ok(Some(_))),
        }
    }
}

/// The held lock on a [`TokenStore`]. Its writes don't lock again, so a
/// refresh can read, exchange and save under one hold. Dropping it closes
/// the file, which releases the lock.
pub struct TokenLock<'a> {
    store: &'a TokenStore,
    _file: std::fs::File,
}

impl TokenLock<'_> {
    pub fn load(&self) -> Result<Option<ChatgptTokens>, TokenError> {
        self.store.load()
    }

    pub fn save(&self, tokens: &ChatgptTokens) -> Result<(), TokenError> {
        let file = TokenFile {
            access_token: tokens.access_token.expose().to_string(),
            refresh_token: tokens.refresh_token.expose().to_string(),
            id_token: tokens.id_token.expose().to_string(),
            account_id: tokens.account_id.clone(),
            last_refresh: tokens.last_refresh,
        };
        let text = serde_json::to_string_pretty(&file).expect("a token file serialises");
        super::write::replace_file(&self.store.path, text.as_bytes(), 0o600).map_err(|error| {
            TokenError::Io {
                path: self.store.path.clone(),
                error,
            }
        })
    }

    pub fn clear(&self) -> Result<(), TokenError> {
        match std::fs::remove_file(&self.store.path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(TokenError::Io {
                path: self.store.path.clone(),
                error,
            }),
        }
    }
}

/// The `llm` section of the resolved config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LlmStatus {
    pub auth: LlmAuth,
    pub token_file: Option<PathBuf>,
    pub logged_in: bool,
}

impl LlmStatus {
    /// `api_key` mode: logged in means a key is set.
    pub fn api_key(key_present: bool) -> Self {
        Self {
            auth: LlmAuth::ApiKey,
            token_file: None,
            logged_in: key_present,
        }
    }
}

// JWTs. Only the payload is read; signatures are the issuer's business.

fn base64url_decode(text: &str) -> Option<Vec<u8>> {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut bits = 0u32;
    let mut count = 0;
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    for byte in text.bytes() {
        if byte == b'=' {
            break;
        }
        let value = ALPHABET.iter().position(|c| *c == byte)? as u32;
        bits = (bits << 6) | value;
        count += 6;
        if count >= 8 {
            count -= 8;
            out.push(((bits >> count) & 0xff) as u8);
        }
    }
    Some(out)
}

fn jwt_claims(jwt: &str) -> Result<Value, TokenError> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 || parts.iter().any(|part| part.is_empty()) {
        return Err(TokenError::InvalidJwt);
    }
    let payload = base64url_decode(parts[1]).ok_or(TokenError::InvalidJwt)?;
    serde_json::from_slice(&payload).map_err(|_| TokenError::InvalidJwt)
}

fn account_id_from(id_token: &str) -> Result<String, TokenError> {
    let claims = jwt_claims(id_token)?;
    claims
        .get("https://api.openai.com/auth")
        .and_then(|auth| auth.get("chatgpt_account_id"))
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .map(str::to_string)
        .ok_or(TokenError::MissingAccountId)
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

    #[error("the issuer's reply couldn't be read at {step}")]
    InvalidReply { step: &'static str },

    #[error(transparent)]
    Token(#[from] TokenError),
}

fn issuer_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(ISSUER_TIMEOUT))
        .http_status_as_error(false)
        .build()
        .into()
}

fn login_transport(error: ureq::Error) -> LoginError {
    LoginError::Transport {
        reason: error.to_string(),
    }
}

fn read_login_body(
    mut response: ureq::http::Response<ureq::Body>,
) -> Result<(u16, String), LoginError> {
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .with_config()
        .limit(REPLY_LIMIT)
        .read_to_string()
        .map_err(login_transport)?;
    Ok((status, text))
}

fn parse_login_reply(text: &str, step: &'static str) -> Result<Value, LoginError> {
    let value = if text.trim().is_empty() {
        Value::Null
    } else {
        serde_json::from_str(text).map_err(|_| LoginError::InvalidReply { step })?
    };
    Ok(value)
}

fn string_field(value: &Value, key: &str, step: &'static str) -> Result<String, LoginError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or(LoginError::InvalidReply { step })
}

/// `asphodel llm login`: the device-code flow, headless. `show` is called
/// once with the URL and code; the function then polls every `interval`
/// until the owner approves, exchanges the code for tokens, and saves them.
pub fn device_code_login(
    issuer: &str,
    store: &TokenStore,
    clock: &dyn Clock,
    show: &mut dyn FnMut(&DeviceCode),
) -> Result<ChatgptTokens, LoginError> {
    let base = issuer.trim_end_matches('/');
    let api = format!("{base}/api/accounts");
    let agent = issuer_agent();

    // 1. Ask for a code.
    let step = "deviceauth/usercode";
    let response = agent
        .post(format!("{api}/deviceauth/usercode"))
        .send_json(json!({ "client_id": CLIENT_ID }))
        .map_err(login_transport)?;
    let (status, text) = read_login_body(response)?;
    if !(200..300).contains(&status) {
        return Err(LoginError::Status { status, step });
    }
    let reply = parse_login_reply(&text, step)?;
    let device_auth_id = string_field(&reply, "device_auth_id", step)?;
    let user_code = reply
        .get("user_code")
        .or_else(|| reply.get("usercode"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or(LoginError::InvalidReply { step })?;
    let interval = match reply.get("interval") {
        Some(Value::String(text)) => text.trim().parse::<u64>().unwrap_or(5),
        Some(Value::Number(number)) => number.as_u64().unwrap_or(5),
        _ => 5,
    };
    let code = DeviceCode {
        verification_url: format!("{base}/codex/device"),
        user_code,
        interval: Duration::from_secs(interval),
    };
    show(&code);

    // 2. Poll until the owner approves.
    let step = "deviceauth/token";
    let started = Instant::now();
    let approval = loop {
        let response = agent
            .post(format!("{api}/deviceauth/token"))
            .send_json(json!({
                "device_auth_id": device_auth_id,
                "user_code": code.user_code,
            }))
            .map_err(login_transport)?;
        let (status, text) = read_login_body(response)?;
        if (200..300).contains(&status) {
            break parse_login_reply(&text, step)?;
        }
        if status == 403 || status == 404 {
            if started.elapsed() >= LOGIN_WAIT {
                return Err(LoginError::Expired);
            }
            std::thread::sleep(code.interval);
            continue;
        }
        return Err(LoginError::Status { status, step });
    };
    let authorization_code = string_field(&approval, "authorization_code", step)?;
    let code_verifier = string_field(&approval, "code_verifier", step)?;

    // 3. Exchange the code for tokens. Form-encoded, as codex-rs does.
    let step = "oauth/token";
    let redirect_uri = format!("{base}/deviceauth/callback");
    let response = agent
        .post(format!("{base}/oauth/token"))
        .send_form([
            ("grant_type", "authorization_code"),
            ("client_id", CLIENT_ID),
            ("code", authorization_code.as_str()),
            ("redirect_uri", redirect_uri.as_str()),
            ("code_verifier", code_verifier.as_str()),
        ])
        .map_err(login_transport)?;
    let (status, text) = read_login_body(response)?;
    if !(200..300).contains(&status) {
        return Err(LoginError::Status { status, step });
    }
    let reply = parse_login_reply(&text, step)?;
    let tokens = ChatgptTokens::from_reply(
        &string_field(&reply, "id_token", step)?,
        &string_field(&reply, "access_token", step)?,
        &string_field(&reply, "refresh_token", step)?,
        clock.now(),
    )?;
    store.save(&tokens)?;
    Ok(tokens)
}

// The client.

/// The subscription client: the Responses API on the Codex backend,
/// streamed, with a bearer access token from the [`TokenStore`].
pub struct CodexResponses {
    settings: LlmSettings,
    store: TokenStore,
    clock: Arc<dyn Clock>,
    issuer: String,
    agent: ureq::Agent,
    refreshes: AtomicUsize,
}

/// What a single post came back with, before the refresh-and-retry logic.
enum Post {
    Unauthorized,
    Failed(LlmError),
}

impl From<LlmError> for Post {
    fn from(error: LlmError) -> Self {
        Post::Failed(error)
    }
}

impl CodexResponses {
    /// `clock` decides when the access token is due for refresh: nothing
    /// reads the wall clock.
    pub fn new(settings: LlmSettings, store: TokenStore, clock: Arc<dyn Clock>) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(settings.timeout))
            .http_status_as_error(false)
            .build();
        Self {
            settings,
            store,
            clock,
            issuer: AUTH_ISSUER.to_string(),
            agent: config.into(),
            refreshes: AtomicUsize::new(0),
        }
    }

    /// The issuer for refresh. Defaults to [`AUTH_ISSUER`]; tests point it
    /// at a stub.
    pub fn with_issuer(mut self, issuer: &str) -> Self {
        self.issuer = issuer.trim_end_matches('/').to_string();
        self
    }

    /// Refresh calls so far.
    pub fn refreshes(&self) -> usize {
        self.refreshes.load(Ordering::SeqCst)
    }

    fn user_agent() -> String {
        format!("{ORIGINATOR}/{}", crate::VERSION)
    }

    fn body(&self, request: &LlmRequest) -> Value {
        let mut body = json!({
            "model": self.settings.model,
            "instructions": request.system,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": request.user}],
            }],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "store": false,
            "stream": true,
            "include": [],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name,
                    "strict": true,
                    "schema": request.schema,
                },
            },
        });
        if let Some(effort) = &self.settings.reasoning_effort {
            body["reasoning"] = json!({ "effort": effort });
        }
        body
    }

    /// One streamed request with `tokens`.
    fn post(&self, tokens: &ChatgptTokens, request: &LlmRequest) -> Result<LlmResponse, Post> {
        let started = Instant::now();
        let mut response = self
            .agent
            .post(format!(
                "{}/responses",
                self.settings.endpoint.trim_end_matches('/')
            ))
            .header(
                "Authorization",
                &format!("Bearer {}", tokens.access_token.expose()),
            )
            .header("chatgpt-account-id", &tokens.account_id)
            .header("originator", ORIGINATOR)
            .header("User-Agent", &Self::user_agent())
            .header("Accept", "text/event-stream")
            .header("Content-Type", "application/json")
            .send_json(self.body(request))
            .map_err(super::llm::transport)?;
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let reset_header = response
            .headers()
            .get("x-codex-primary-reset-at")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.trim().parse::<i64>().ok());
        match status {
            200..=299 => {}
            401 => return Err(Post::Unauthorized),
            429 => {
                let text = response
                    .body_mut()
                    .with_config()
                    .limit(REPLY_LIMIT)
                    .read_to_string()
                    .map_err(super::llm::transport)?;
                if let Some(resets_at) = usage_limit(&text, reset_header) {
                    return Err(LlmError::UsageLimited { resets_at }.into());
                }
                return Err(super::llm::status_error(status, &headers, self.clock.now()).into());
            }
            _ => return Err(LlmError::Status { status }.into()),
        }
        let reader = response
            .body_mut()
            .with_config()
            .limit(REPLY_LIMIT)
            .reader();
        let (json, usage) = parse_stream_reader(reader)?;
        let latency = started.elapsed();
        Ok(LlmResponse {
            json,
            usage,
            latency,
        })
    }

    /// Refreshes `stale`, unless the file already holds something newer (a
    /// refresh by another thread, client or process, or a new login), which
    /// is used instead. A file cleared since the caller loaded it counts as
    /// a logout: refresh returns LoginRequired and never recreates it.
    /// A refresh token is single-use, so the store's lock
    /// is held from that re-read to the save: no one else spends the same
    /// token, and no login or clear lands in between to be overwritten. The
    /// rotated tokens are saved before this returns, so a crash after the
    /// refresh never loses the only valid refresh token.
    fn refresh(&self, stale: &ChatgptTokens) -> Result<ChatgptTokens, LlmError> {
        let lock = self.store.lock()?;
        let Some(current) = lock.load()? else {
            return Err(LlmError::LoginRequired);
        };
        if current.access_token != stale.access_token {
            return Ok(current);
        }
        self.refreshes.fetch_add(1, Ordering::SeqCst);
        let mut response = self
            .agent
            .post(format!("{}/oauth/token", self.issuer))
            .send_json(json!({
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": stale.refresh_token.expose(),
            }))
            .map_err(super::llm::transport)?;
        let status = response.status().as_u16();
        let text = response
            .body_mut()
            .with_config()
            .limit(REPLY_LIMIT)
            .read_to_string()
            .map_err(super::llm::transport)?;
        if !(200..300).contains(&status) {
            // Transient issuer failures don't invalidate the credential.
            return Err(match status {
                408 | 429 | 500..=599 => LlmError::Status { status },
                _ => LlmError::LoginRequired,
            });
        }
        let reply: Value = serde_json::from_str(&text).map_err(|_| LlmError::LoginRequired)?;
        let field = |key: &str| reply.get(key).and_then(Value::as_str).map(str::to_string);
        let id_token = field("id_token");
        let account_id = match &id_token {
            Some(id_token) => account_id_from(id_token)?,
            None => stale.account_id.clone(),
        };
        let rotated = ChatgptTokens {
            access_token: field("access_token")
                .map_or_else(|| stale.access_token.clone(), Secret::new),
            refresh_token: field("refresh_token")
                .map_or_else(|| stale.refresh_token.clone(), Secret::new),
            id_token: id_token.map_or_else(|| stale.id_token.clone(), Secret::new),
            account_id,
            last_refresh: self.clock.now(),
        };
        lock.save(&rotated)?;
        Ok(rotated)
    }
}

impl LlmClient for CodexResponses {
    fn model(&self) -> &str {
        &self.settings.model
    }

    fn reasoning_effort(&self) -> Option<&str> {
        self.settings.reasoning_effort.as_deref()
    }

    /// Load the token file; refresh first if the access token is due; post;
    /// on 401 refresh once and retry once. A second 401, or a refresh the
    /// issuer rejects, is [`LlmError::LoginRequired`].
    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        let mut tokens = self.store.load()?.ok_or(LlmError::LoginRequired)?;
        if tokens.due_for_refresh(self.clock.now()) {
            tokens = self.refresh(&tokens)?;
        }
        match self.post(&tokens, request) {
            Ok(response) => Ok(response),
            Err(Post::Failed(error)) => Err(error),
            Err(Post::Unauthorized) => {
                let tokens = self.refresh(&tokens)?;
                match self.post(&tokens, request) {
                    Ok(response) => Ok(response),
                    Err(Post::Failed(error)) => Err(error),
                    Err(Post::Unauthorized) => Err(LlmError::LoginRequired),
                }
            }
        }
    }
}

/// A 429 body of `{"error": {"type": "usage_limit_reached", "resets_at":
/// <unix seconds>}}` is a usage limit; the header is the fallback.
fn usage_limit(body: &str, reset_header: Option<i64>) -> Option<Timestamp> {
    let value: Value = serde_json::from_str(body).ok()?;
    let error = value.get("error")?;
    if error.get("type").and_then(Value::as_str) != Some("usage_limit_reached") {
        return None;
    }
    let seconds = error
        .get("resets_at")
        .and_then(Value::as_i64)
        .or(reset_header)?;
    Timestamp::from_second(seconds).ok()
}

/// The error and incomplete-reason codes the Codex backend sends, from
/// `codex-rs` at `6b4daaf`. They are the only backend strings an error
/// carries: the field is the backend's to fill, so a value outside this
/// list could hold anything it was sent, a token or the prompt included.
const BACKEND_CODES: &[&str] = &[
    "server_error",
    "rate_limit_exceeded",
    "slow_down",
    "server_is_overloaded",
    "context_length_exceeded",
    "insufficient_quota",
    "usage_not_included",
    "invalid_prompt",
    "cyber_policy",
    "bio_policy",
    "misalignment_policy_violation",
    "max_output_tokens",
    "content_filter",
    "interrupted",
];

/// A known code as itself, any other as `unknown`, and none as `missing`
/// (the event's own fixed name).
fn backend_code(code: Option<&str>, missing: &'static str) -> String {
    let code = match code {
        None => missing,
        Some(code) => BACKEND_CODES
            .iter()
            .copied()
            .find(|known| *known == code)
            .unwrap_or("unknown"),
    };
    code.to_string()
}

fn parse_stream_reader(reader: impl Read) -> Result<(Value, Option<LlmUsage>), LlmError> {
    // Bound even a single unterminated line, not just well-formed SSE blocks.
    let mut reader = BufReader::new(reader.take(REPLY_LIMIT + 1));
    let mut bytes = 0u64;
    let mut done_text: Option<String> = None;
    let mut deltas = String::new();
    let mut completed = false;
    let mut usage = None;
    loop {
        let mut data = Vec::new();
        let mut eof = false;
        loop {
            let mut line = String::new();
            let count = reader
                .read_line(&mut line)
                .map_err(|error| super::llm::transport(error.into()))?;
            bytes += count as u64;
            if bytes > REPLY_LIMIT {
                return Err(LlmError::Transport {
                    reason: "SSE reply exceeds size limit".into(),
                });
            }
            if count == 0 {
                eof = true;
                break;
            }
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                break;
            }
            if let Some(value) = line.strip_prefix("data:") {
                data.push(value.trim_start().to_string());
            }
        }
        if data.is_empty() {
            if eof {
                break;
            }
            continue;
        }
        let Ok(event) = serde_json::from_str::<Value>(&data.join("\n")) else {
            if eof {
                break;
            }
            continue;
        };
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "response.output_item.done" => {
                let item = &event["item"];
                if item.get("type").and_then(Value::as_str) == Some("message")
                    && let Some(content) = item.get("content").and_then(Value::as_array)
                {
                    let text: String = content
                        .iter()
                        .filter(|part| {
                            part.get("type").and_then(Value::as_str) == Some("output_text")
                        })
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect();
                    done_text = Some(text);
                }
            }
            "response.output_text.delta" => {
                if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                    deltas.push_str(delta);
                }
            }
            "response.completed" => {
                completed = true;
                let response = &event["response"];
                if let (Some(input), Some(output)) = (
                    response["usage"]["input_tokens"].as_u64(),
                    response["usage"]["output_tokens"].as_u64(),
                ) {
                    usage = Some(LlmUsage {
                        input_tokens: input,
                        output_tokens: output,
                    });
                }
            }
            "response.failed" => {
                let code = backend_code(event["response"]["error"]["code"].as_str(), "failed");
                return Err(LlmError::Backend { code });
            }
            "response.incomplete" => {
                let code = backend_code(
                    event["response"]["incomplete_details"]["reason"].as_str(),
                    "incomplete",
                );
                return Err(LlmError::Backend { code });
            }
            "error" => {
                let code = backend_code(
                    event["error"]["code"]
                        .as_str()
                        .or_else(|| event["code"].as_str()),
                    "error",
                );
                return Err(LlmError::Backend { code });
            }
            _ => {}
        }
        if completed || eof {
            break;
        }
    }
    if !completed {
        return Err(LlmError::NoContent);
    }
    let text = match done_text {
        Some(text) => text,
        None if !deltas.is_empty() => deltas,
        None => return Err(LlmError::NoContent),
    };
    let content = unfence(&text);
    let json = serde_json::from_str(content).map_err(|_| LlmError::NotJson {
        bytes: content.len(),
    })?;
    Ok((json, usage))
}
