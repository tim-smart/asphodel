//! The ChatGPT/Codex subscription client follows the authentication and
//! transport contracts of the `chatgpt` auth mode.
//!
//! The protocol follows `openai/codex`; the Codex backend is undocumented.
//! The tests pin the wire as these files have it:
//!
//! - `codex-rs/login/src/device_code_auth.rs`: the device-code login, its
//!   polling (403 or 404 while pending) and the form-encoded code exchange.
//! - `codex-rs/login/src/auth/manager.rs` and `oauth/client.rs`: the client
//!   id, the issuer `https://auth.openai.com`, and the JSON refresh. The
//!   access token is refreshed when its JWT `exp` is within 5 minutes, or
//!   when the last refresh is more than 8 days old.
//! - `codex-rs/login/src/token_data.rs`: the account id is the
//!   `chatgpt_account_id` claim under `https://api.openai.com/auth` in the
//!   id token.
//! - `codex-rs/model-provider-info/src/lib.rs` and `codex-api/src/endpoint/
//!   responses.rs`: `POST {endpoint}/responses` with `Accept:
//!   text/event-stream`.
//! - `codex-rs/codex-api/src/common.rs`: the request body. There is no
//!   `temperature` and no `max_tokens`.
//! - `codex-rs/codex-api/src/sse/responses.rs`: the stream events.
//! - `codex-rs/codex-api/src/api_bridge.rs`: a 429 with
//!   `usage_limit_reached` is a usage limit, not a rate limit; its reset is
//!   an absolute time, also sent as the `x-codex-primary-reset-at` header.
//! - `codex-rs/login/src/auth/default_client.rs`: the `originator` header
//!   and the `User-Agent` built from it.
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
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use asphodel_core::clock::SimulatedClock;
use asphodel_core::config::Secret;
use asphodel_core::models::*;
use jiff::{SignedDuration, Timestamp};
use serde_json::{Value, json};

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

fn an_hour_from_start() -> Timestamp {
    start() + SignedDuration::from_hours(1)
}

fn clock() -> Arc<SimulatedClock> {
    Arc::new(SimulatedClock::new(start()))
}

fn request() -> LlmRequest {
    LlmRequest {
        template: Template {
            name: "extract".into(),
            version: 3,
            guidance: None,
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
        for i in 0..=chunk.len() {
            out.push(ALPHABET[((n >> (18 - 6 * i)) & 0x3f) as usize] as char);
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

/// The access token `label` good for an hour, as the backend sees it.
fn bearer(label: &str) -> String {
    access_token(label, an_hour_from_start())
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

fn tokens(access_label: &str, refresh_label: &str, exp: Timestamp) -> ChatgptTokens {
    ChatgptTokens {
        access_token: Secret::new(access_token(access_label, exp)),
        refresh_token: Secret::new(format!("rt-{refresh_label}")),
        id_token: Secret::new(id_token(ACCOUNT_ID)),
        account_id: ACCOUNT_ID.into(),
        last_refresh: start(),
    }
}

/// A store holding the access token `label` that expires at `exp`, with
/// the refresh token `rt-one`.
fn store_with(dir: &TestDir, label: &str, exp: Timestamp) -> TokenStore {
    let store = TokenStore::open(&dir.data());
    store.save(&tokens(label, "one", exp)).unwrap();
    store
}

fn logged_in_store(dir: &TestDir) -> TokenStore {
    store_with(dir, "current", an_hour_from_start())
}

fn expired_store(dir: &TestDir) -> TokenStore {
    store_with(dir, "stale", start() - SignedDuration::from_hours(1))
}

/// What `asphodel llm login` saves: a different session from the one the
/// daemon is refreshing.
fn relogin_tokens() -> ChatgptTokens {
    tokens("relogin", "three", an_hour_from_start())
}

fn chatgpt_settings(endpoint: &str) -> LlmSettings {
    LlmSettings {
        auth: LlmAuth::Chatgpt,
        endpoint: endpoint.to_string(),
        model: "gpt-5.1".into(),
        reasoning_effort: None,
        api_key: None,
        timeout: Duration::from_secs(5),
    }
}

/// A client on `server` for both the backend and the issuer.
fn client(server: &StubServer, store: TokenStore) -> CodexResponses {
    client_with(server, store, |_| {})
}

fn client_with(
    server: &StubServer,
    store: TokenStore,
    edit: impl FnOnce(&mut LlmSettings),
) -> CodexResponses {
    let mut settings = chatgpt_settings(&server.url);
    edit(&mut settings);
    CodexResponses::new(settings, store, clock()).with_issuer(&server.url)
}

fn file_mode(path: &Path) -> u32 {
    std::fs::metadata(path).unwrap().permissions().mode() & 0o777
}

/// An SSE body. Each event's data carries its kind as `type`.
fn sse(events: &[(&str, Value)]) -> String {
    events
        .iter()
        .map(|(kind, data)| {
            let mut data = data.clone();
            data["type"] = json!(kind);
            format!("event: {kind}\ndata: {data}\n\n")
        })
        .collect()
}

/// A streamed reply: `response.output_item.done` with `text`, then
/// `response.completed` with usage.
fn completion(text: &str) -> StubResponse {
    StubResponse::stream(sse(&[
        ("response.created", json!({"response": {"id": "resp_1"}})),
        (
            "response.output_item.done",
            json!({"item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}]
            }}),
        ),
        (
            "response.completed",
            json!({"response": {
                "id": "resp_1",
                "status": "completed",
                "usage": {"input_tokens": 41, "output_tokens": 7, "total_tokens": 48}
            }}),
        ),
    ]))
}

/// The usage [`completion`] reports.
const USAGE: Option<LlmUsage> = Some(LlmUsage {
    input_tokens: 41,
    output_tokens: 7,
});

/// The three ways a stream ends in a backend error, each carrying `code`
/// where the client reads it.
fn terminal_events(code: &str) -> [(&'static str, String); 3] {
    let event = |kind: &'static str, data: Value| (kind, sse(&[(kind, data)]));
    [
        event(
            "response.failed",
            json!({"response": {"id": "resp_f", "status": "failed", "error": {"code": code, "message": "failed"}}}),
        ),
        event(
            "response.incomplete",
            json!({"response": {"id": "resp_i", "status": "incomplete", "incomplete_details": {"reason": code}}}),
        ),
        event(
            "error",
            json!({"code": code, "message": "error", "param": null}),
        ),
    ]
}

// The stub server.

/// A loopback HTTP/1.1 server driven by a handler, which gets each request
/// and the number of requests before it. Each connection is answered and
/// closed; every request is recorded in order.
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

    fn assert_header_starts_with(&self, name: &str, prefix: &str) {
        let value = self.header(name).unwrap_or_default();
        assert!(value.starts_with(prefix), "{name}: {:?}", self.headers);
    }

    fn json(&self) -> Value {
        serde_json::from_str(&self.body).expect("a JSON body")
    }

    fn form(&self) -> BTreeMap<String, String> {
        url::form_urlencoded::parse(self.body.as_bytes())
            .into_owned()
            .collect()
    }

    fn bearer(&self) -> Option<&str> {
        self.header("authorization")?.strip_prefix("Bearer ")
    }
}

#[derive(Debug, Clone)]
struct StubResponse {
    status: u16,
    content_type: &'static str,
    headers: Vec<(String, String)>,
    body: String,
    /// When nonzero, the body is sent chunked and the connection is held
    /// open this long after it before the terminating chunk, as a backend
    /// that has finished a response but not closed the stream would.
    linger: Duration,
}

impl StubResponse {
    fn new(status: u16, content_type: &'static str, body: String) -> Self {
        Self {
            status,
            content_type,
            headers: Vec::new(),
            body,
            linger: Duration::ZERO,
        }
    }

    fn json(status: u16, value: Value) -> Self {
        Self::new(status, "application/json", value.to_string())
    }

    /// A non-JSON reply, as a proxy or a load balancer sends one.
    fn text(status: u16, body: &str) -> Self {
        Self::new(status, "text/plain", body.to_string())
    }

    fn stream(body: String) -> Self {
        Self::new(200, "text/event-stream", body)
    }

    fn status(status: u16) -> Self {
        Self::json(status, json!({}))
    }

    fn lingering(mut self, linger: Duration) -> Self {
        self.linger = linger;
        self
    }

    fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }
}

impl StubServer {
    fn start(
        handler: impl Fn(&StubRequest, usize) -> StubResponse + Send + Sync + 'static,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&requests);
        let handler = Arc::new(handler);
        let calls = Arc::new(AtomicUsize::new(0));
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let (handler, log, calls) =
                    (Arc::clone(&handler), Arc::clone(&log), Arc::clone(&calls));
                std::thread::spawn(move || {
                    serve_one(
                        stream,
                        |request| handler(request, calls.fetch_add(1, Ordering::SeqCst)),
                        &log,
                    )
                });
            }
        });
        Self {
            url: format!("http://127.0.0.1:{port}"),
            requests,
        }
    }

    /// The Codex backend answering every request with `response`.
    fn backend(response: StubResponse) -> Self {
        Self::start(move |_, _| response.clone())
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

fn serve_one(
    mut stream: TcpStream,
    handler: impl FnOnce(&StubRequest) -> StubResponse,
    log: &Mutex<Vec<StubRequest>>,
) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut lines = reader.by_ref().lines().map_while(Result::ok);
    let Some(line) = lines.next() else { return };
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    let headers = lines
        .take_while(|line| !line.is_empty())
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            Some((key.trim().to_string(), value.trim().to_string()))
        })
        .collect();
    let mut request = StubRequest {
        method,
        path,
        headers,
        body: String::new(),
    };
    let length = request
        .header("content-length")
        .map_or(0, |v| v.parse().unwrap());
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    request.body = String::from_utf8_lossy(&body).into_owned();
    let response = handler(&request);
    log.lock().unwrap().push(request);
    let (status, body) = (response.status, &response.body);
    let mut head = format!(
        "HTTP/1.1 {status} Status\r\nContent-Type: {}\r\nConnection: close\r\n",
        response.content_type
    );
    for (name, value) in &response.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    if response.linger > Duration::ZERO {
        let _ = write!(
            stream,
            "{head}Transfer-Encoding: chunked\r\n\r\n{:x}\r\n{body}\r\n",
            body.len()
        );
        let _ = stream.flush();
        std::thread::sleep(response.linger);
        let _ = write!(stream, "0\r\n\r\n");
    } else {
        let _ = write!(stream, "{head}Content-Length: {}\r\n\r\n{body}", body.len());
    }
    let _ = stream.flush();
}

/// A server that is both the issuer (`/oauth/token`) and the backend
/// (`/responses`).
fn codex(
    oauth: impl Fn() -> StubResponse + Send + Sync + 'static,
    responses: impl Fn() -> StubResponse + Send + Sync + 'static,
) -> StubServer {
    StubServer::start(move |request, _| match request.path.as_str() {
        "/oauth/token" => oauth(),
        "/responses" => responses(),
        other => panic!("unexpected path {other}"),
    })
}

/// An issuer refresh reply rotating to the access token `rotated` and the
/// refresh token `rt-two`.
fn rotation() -> StubResponse {
    StubResponse::json(
        200,
        json!({
            "id_token": id_token(ACCOUNT_ID),
            "access_token": bearer("rotated"),
            "refresh_token": "rt-two",
        }),
    )
}

/// An issuer that approves a device-code login on the first poll, unless
/// `script` answers a request (by path and call count) first.
fn issuer(
    script: impl Fn(&str, usize) -> Option<StubResponse> + Send + Sync + 'static,
) -> StubServer {
    StubServer::start(move |request, n| {
        script(&request.path, n).unwrap_or_else(|| match request.path.as_str() {
            "/api/accounts/deviceauth/usercode" => StubResponse::json(
                200,
                json!({"device_auth_id": "dev_123", "user_code": "ABCD-EFGH", "interval": "0"}),
            ),
            "/api/accounts/deviceauth/token" => StubResponse::json(
                200,
                json!({
                    "authorization_code": "code_9",
                    "code_challenge": "chal",
                    "code_verifier": "verif"
                }),
            ),
            "/oauth/token" => login_reply(&id_token(ACCOUNT_ID)),
            other => panic!("unexpected path {other}"),
        })
    })
}

/// The issuer's code exchange reply, with `id_token` as given.
fn login_reply(id_token: &str) -> StubResponse {
    StubResponse::json(
        200,
        json!({
            "id_token": id_token,
            "access_token": bearer("first"),
            "refresh_token": "rt-first"
        }),
    )
}

// Login: the device-code flow against a stub issuer, and the token file.

#[test]
fn device_code_login_polls_exchanges_and_saves_tokens() {
    // Pending, as the issuer answers it: 403 or 404 with a plain text
    // body. codex-rs checks the status before parsing JSON.
    let issuer = issuer(|path, n| match (path, n) {
        ("/api/accounts/deviceauth/token", 1) => Some(StubResponse::text(403, "Forbidden")),
        ("/api/accounts/deviceauth/token", 2) => Some(StubResponse::text(404, "Not Found")),
        _ => None,
    });
    let dir = TestDir::new();
    let data = dir.data();
    let store = TokenStore::open(&data);
    assert_eq!(store.load().unwrap(), None);
    assert!(
        std::fs::read_dir(&data).unwrap().next().is_none(),
        "open created a file"
    );
    let mut shown = Vec::new();

    let tokens = device_code_login(&issuer.url, &store, clock().as_ref(), &mut |code| {
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
    let requests = issuer.requests();
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].json(), json!({"client_id": CLIENT_ID}));
    assert_eq!(
        requests[1].json(),
        json!({"device_auth_id": "dev_123", "user_code": "ABCD-EFGH"})
    );
    let exchange = &requests[4];
    exchange.assert_header_starts_with("content-type", "application/x-www-form-urlencoded");
    let form = exchange.form();
    assert_eq!(form["grant_type"], "authorization_code");
    assert_eq!(form["client_id"], CLIENT_ID);
    assert_eq!(form["code"], "code_9");
    assert_eq!(form["code_verifier"], "verif");
    assert_eq!(
        form["redirect_uri"],
        format!("{}/deviceauth/callback", issuer.url)
    );

    // A clear removes the file, and clearing again is fine.
    store.clear().unwrap();
    assert_eq!(store.load().unwrap(), None);
    store.clear().unwrap();
}

#[test]
fn a_refused_login_saves_nothing() {
    // A proxy's text error at any step is a status error, not "the reply
    // couldn't be read". An id token without the account id, which the
    // backend needs on every request, or that isn't a JWT, can't be saved.
    let refuse = |failing: &'static str, response: StubResponse| {
        let issuer = issuer(move |path, _| (path == failing).then(|| response.clone()));
        let dir = TestDir::new();
        let store = TokenStore::open(&dir.data());
        let error =
            device_code_login(&issuer.url, &store, clock().as_ref(), &mut |_| {}).unwrap_err();
        assert_eq!(store.load().unwrap(), None, "{failing}");
        error
    };
    for (path, code) in [
        ("/api/accounts/deviceauth/usercode", 503),
        ("/api/accounts/deviceauth/token", 400),
        ("/oauth/token", 500),
    ] {
        let error = refuse(path, StubResponse::text(code, "upstream error"));
        assert!(
            matches!(error, LoginError::Status { status, .. } if status == code),
            "{path}: {error:?}"
        );
    }
    let no_account = login_reply(&jwt(json!({"email": "tim@example.test"})));
    let error = refuse("/oauth/token", no_account);
    assert!(
        matches!(error, LoginError::Token(TokenError::MissingAccountId)),
        "{error:?}"
    );
    let error = refuse("/oauth/token", login_reply("not.a.jwt.at.all"));
    assert!(
        matches!(error, LoginError::Token(TokenError::InvalidJwt)),
        "{error:?}"
    );
}

// The Responses wire format.

#[test]
fn the_client_streams_a_structured_responses_request_with_the_codex_headers() {
    let backend = StubServer::backend(completion(
        "{\"claims\":[\"Tim moved to Wellington in March 2026.\"]}",
    ));
    let dir = TestDir::new();
    let client = client(&backend, logged_in_store(&dir));
    assert_eq!(client.model(), "gpt-5.1");

    let response = client.complete(&request()).unwrap();
    assert_eq!(
        response.json,
        json!({"claims": ["Tim moved to Wellington in March 2026."]})
    );
    assert_eq!(response.usage, USAGE);
    assert!(response.latency > Duration::ZERO);
    assert_eq!(client.refreshes(), 0);

    let sent = backend.only_request();
    assert_eq!(sent.method, "POST");
    assert_eq!(sent.path, "/responses");
    assert_eq!(sent.bearer(), Some(bearer("current").as_str()));
    assert_eq!(sent.header("chatgpt-account-id"), Some(ACCOUNT_ID));
    assert_eq!(sent.header("originator"), Some(ORIGINATOR));
    sent.assert_header_starts_with("user-agent", ORIGINATOR);
    sent.assert_header_starts_with("accept", "text/event-stream");
    sent.assert_header_starts_with("content-type", "application/json");

    // GPT-5 rejects temperature, and the Responses API has no max_tokens;
    // neither is sent. Chat Completions' response_format and messages
    // aren't either, and no reasoning effort was configured.
    assert_eq!(
        sent.json(),
        json!({
            "model": "gpt-5.1",
            "instructions": "You extract memories.",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Tim said: I moved to Wellington in March."}]
            }],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "store": false,
            "stream": true,
            "include": [],
            "text": {"format": {
                "type": "json_schema",
                "name": "claims",
                "strict": true,
                "schema": request().schema
            }}
        })
    );

    // A configured reasoning effort reaches the request.
    let backend = StubServer::backend(completion("{}"));
    client_with(&backend, logged_in_store(&dir), |settings| {
        settings.reasoning_effort = Some("low".into())
    })
    .complete(&request())
    .unwrap();
    assert_eq!(
        backend.only_request().json()["reasoning"],
        json!({"effort": "low"})
    );
}

#[test]
fn a_stream_ends_in_its_reply_or_a_fixed_backend_code() {
    // Deltas are reassembled when no done item carries the text; a stream
    // without `response.completed` has no reply; each terminal event passes
    // a known code through. A stream that completes and then stays open is
    // answered at completion, as codex-rs does
    // (`codex-rs/codex-api/src/sse/responses.rs`), or a good answer becomes
    // a timeout and a duplicate request.
    let deltas = StubResponse::stream(sse(&[
        ("response.output_text.delta", json!({"delta": "{\"cla"})),
        ("response.output_text.delta", json!({"delta": "ims\":[]}"})),
        (
            "response.completed",
            json!({"response": {"id": "resp_2", "status": "completed"}}),
        ),
    ]));
    let created = ("response.created", json!({"response": {"id": "resp_1"}}));
    let unfinished = StubResponse::stream(sse(&[created]));
    let open = completion("{\"claims\":[\"kept\"]}").lingering(Duration::from_secs(3));
    let kept = json!({"claims": ["kept"]});
    let mut cases = vec![
        ("deltas", deltas, Ok((json!({"claims": []}), None))),
        ("no completed event", unfinished, Err("no content")),
        ("open after completion", open, Ok((kept, USAGE))),
    ];
    for (shape, body) in terminal_events("context_length_exceeded") {
        let response = StubResponse::stream(body);
        cases.push((shape, response, Err("context_length_exceeded")));
    }
    for (shape, response, expected) in cases {
        let backend = StubServer::backend(response);
        let dir = TestDir::new();
        let result = client_with(&backend, logged_in_store(&dir), |settings| {
            settings.timeout = Duration::from_secs(1)
        })
        .complete(&request());
        match (result, expected) {
            (Ok(response), Ok((json, usage))) => {
                assert_eq!((response.json, response.usage), (json, usage), "{shape}");
                assert!(
                    response.latency < Duration::from_secs(2),
                    "{shape}: waited for the close"
                );
            }
            (Err(LlmError::NoContent), Err("no content")) => {}
            (Err(LlmError::Backend { code }), Err(expected)) if code == expected => {}
            (result, expected) => panic!("{shape}: {result:?}, expected {expected:?}"),
        }
        assert_eq!(backend.requests().len(), 1, "{shape}: a duplicate request");
    }
}

// Refresh.

#[test]
fn a_due_access_token_is_refreshed_and_the_rotation_persisted_before_the_request() {
    // Due means expired, or, as codex-rs has it, expiring within 5 minutes.
    for exp in [
        start() - SignedDuration::from_hours(1),
        start() + SignedDuration::from_mins(4),
    ] {
        let dir = TestDir::new();
        let store = store_with(&dir, "stale", exp);
        let token_path = store.path().to_path_buf();
        // The backend checks, at the moment the request arrives, that the
        // file already holds the rotated refresh token.
        let file_at_request = Arc::new(Mutex::new(String::new()));
        let seen = Arc::clone(&file_at_request);
        let server = codex(rotation, move || {
            *seen.lock().unwrap() = std::fs::read_to_string(&token_path).unwrap();
            completion("{\"claims\":[]}")
        });
        let client = client(&server, store);

        let response = client.complete(&request()).unwrap();
        assert_eq!(response.json, json!({"claims": []}));
        assert_eq!(client.refreshes(), 1);
        assert_eq!(server.paths(), ["/oauth/token", "/responses"], "{exp}");

        // The refresh, as codex-rs sends it: JSON, with the client id.
        let refresh = &server.requests()[0];
        refresh.assert_header_starts_with("content-type", "application/json");
        assert_eq!(
            refresh.json(),
            json!({"grant_type": "refresh_token", "client_id": CLIENT_ID, "refresh_token": "rt-one"})
        );
        assert_eq!(refresh.header("authorization"), None);

        // Persisted before the retried request went out, and the request
        // used the new access token.
        let at_request = file_at_request.lock().unwrap().clone();
        assert!(at_request.contains("rt-two"), "{at_request}");
        assert!(!at_request.contains("rt-one"), "{at_request}");
        assert_eq!(
            server.requests()[1].bearer(),
            Some(bearer("rotated").as_str())
        );

        let saved = TokenStore::open(&dir.data()).load().unwrap().unwrap();
        assert_eq!(saved.refresh_token.expose(), "rt-two");
        assert_eq!(saved.account_id, ACCOUNT_ID);
        assert_eq!(saved.last_refresh, start());
    }
}

#[test]
fn a_failed_credential_asks_for_a_login_and_a_transient_issuer_error_does_not() {
    // Only a login helps when there is no token file, when the issuer
    // rejects the refresh (400 `invalid_grant`, 401), or when the backend
    // answers 401 again after a refresh (one refresh and one retry, never a
    // loop). An issuer that is down, overloaded or rate limiting has not
    // rejected the credential: that's retryable, and the credential stays
    // for the next attempt. Nothing goes to the backend without a login.
    let ok = || completion("{}");
    let revoked = StubResponse::json(
        400,
        json!({"error": "invalid_grant", "error_description": "refresh token rt-one is revoked"}),
    );
    let unknown = StubResponse::json(401, json!({"error": "invalid_client"}));
    let refresh = || vec!["/oauth/token"];
    let retried = vec!["/responses", "/oauth/token", "/responses"];
    let kept = Some("rt-one");
    // (case, issuer, backend, the status if transient, requests, the
    // refresh token left in the file)
    let mut cases = vec![
        ("no token file", rotation(), ok(), None, vec![], None),
        ("invalid_grant", revoked, ok(), None, refresh(), kept),
        ("invalid_client", unknown, ok(), None, refresh(), kept),
        (
            "401 after a refresh",
            rotation(),
            StubResponse::status(401),
            None,
            retried,
            Some("rt-two"),
        ),
    ];
    for status in [503, 502, 429] {
        let down = StubResponse::text(status, "Service Unavailable");
        cases.push(("transient", down, ok(), Some(status), refresh(), kept));
    }
    for (case, issuer, backend, transient, paths, saved) in cases {
        let dir = TestDir::new();
        let store = match case {
            "no token file" => TokenStore::open(&dir.data()),
            "401 after a refresh" => logged_in_store(&dir),
            _ => expired_store(&dir),
        };
        let server = codex(move || issuer.clone(), move || backend.clone());
        let error = client(&server, store).complete(&request()).unwrap_err();
        match transient {
            None => assert!(
                matches!(error, LlmError::LoginRequired),
                "{case}: {error:?}"
            ),
            Some(status) => assert!(
                matches!(error, LlmError::Status { status: got } if got == status),
                "{case}: {error:?}"
            ),
        }
        assert_eq!(error.is_retryable(), transient.is_some(), "{case}");
        assert!(!format!("{error:?}").contains("rt-one"), "{error:?}");
        assert_eq!(server.paths(), paths, "{case}");
        let file = TokenStore::open(&dir.data()).load().unwrap();
        assert_eq!(
            file.as_ref().map(|tokens| tokens.refresh_token.expose()),
            saved,
            "{case}"
        );
    }
}

#[test]
fn a_new_login_written_while_running_is_picked_up_without_a_restart() {
    // Re-login after a failed credential: `asphodel llm login` writes the
    // file; the daemon's next call reads it instead of refreshing.
    let dir = TestDir::new();
    let server = codex(|| StubResponse::status(401), || completion("{}"));
    let client = client(&server, expired_store(&dir));
    assert!(matches!(
        client.complete(&request()),
        Err(LlmError::LoginRequired)
    ));

    // The CLI logs in: a fresh file, written by another TokenStore.
    TokenStore::open(&dir.data())
        .save(&relogin_tokens())
        .unwrap();

    client.complete(&request()).unwrap();
    assert_eq!(server.paths(), ["/oauth/token", "/responses"]);
    assert_eq!(
        server.requests()[1].bearer(),
        Some(bearer("relogin").as_str())
    );
    assert_eq!(
        client.refreshes(),
        1,
        "no second refresh: the new file was used as is"
    );
}

// Limits on a 429.

#[test]
fn a_429_is_a_usage_limit_a_rate_limit_hold_or_a_retryable_status() {
    // A usage limit needs the `usage_limit_reached` type and a reset, from
    // the body or else the `x-codex-primary-reset-at` header; it defers to
    // the reset. A Retry-After holds for that long, a date on the client's
    // clock. Neither is retried. Any other 429 is a plain, retryable status.
    enum Expected {
        Usage(Timestamp),
        Hold(Duration),
        Status,
    }
    use Expected::*;
    let resets_at: Timestamp = "2026-03-02T14:30:00Z".parse().unwrap();
    let header_reset: Timestamp = "2026-03-02T15:00:00Z".parse().unwrap();
    let reset_header = |response: StubResponse| {
        response.with_header(
            "x-codex-primary-reset-at",
            &header_reset.as_second().to_string(),
        )
    };
    let usage = |body: Value| StubResponse::json(429, json!({"error": body}));
    let rate = || usage(json!({"type": "rate_limit_exceeded", "message": "slow down"}));
    let until = jiff::fmt::rfc2822::DateTimePrinter::new()
        .timestamp_to_rfc9110_string(&(start() + SignedDuration::from_mins(5)))
        .unwrap();
    let limit = || usage(json!({"type": "usage_limit_reached"}));
    let body_reset = usage(json!({
        "type": "usage_limit_reached",
        "message": "You've hit your usage limit.",
        "plan_type": "plus",
        "resets_at": resets_at.as_second()
    }));
    let not_json = StubResponse::text(429, "Too Many Requests");
    let secs = Duration::from_secs;
    let after_12s = rate().with_header("Retry-After", "12");
    let after_date = rate().with_header("Retry-After", &until);
    let cases = [
        ("body reset", reset_header(body_reset), Usage(resets_at)),
        ("header reset", reset_header(limit()), Usage(header_reset)),
        ("no reset", limit(), Status),
        ("rate limit with a reset", reset_header(rate()), Status),
        ("not json", reset_header(not_json), Status),
        ("plain", rate(), Status),
        ("Retry-After seconds", after_12s, Hold(secs(12))),
        ("Retry-After date", after_date, Hold(secs(300))),
    ];
    for (case, response, expected) in cases {
        let backend = StubServer::backend(response);
        let dir = TestDir::new();
        let error = client(&backend, logged_in_store(&dir))
            .complete(&request())
            .unwrap_err();
        let matched = match expected {
            Usage(at) => matches!(error, LlmError::UsageLimited { resets_at } if resets_at == at),
            Hold(hold) => {
                matches!(error, LlmError::RateLimited { retry_after } if retry_after == hold)
            }
            Status => matches!(error, LlmError::Status { status: 429 }),
        };
        assert!(matched, "{case}: {error:?}");
        assert_eq!(
            error.is_retryable(),
            matches!(expected, Status),
            "{case}: {error:?}"
        );
        assert_eq!(backend.requests().len(), 1, "{case}");
    }
}

// Secret hygiene across the whole client.

#[test]
fn no_error_or_response_carries_a_token() {
    let dir = TestDir::new();
    let server = StubServer::start(move |request, n| match (request.path.as_str(), n) {
        ("/oauth/token", _) => rotation(),
        ("/responses", 1) => StubResponse::status(503),
        ("/responses", _) => completion("{\"ok\":true}"),
        other => panic!("unexpected {other:?}"),
    });
    let client = client(&server, expired_store(&dir));
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

#[test]
fn a_backend_code_reflecting_a_secret_or_the_prompt_never_reaches_the_error() {
    // The backend echoes what it was sent into the field the client keeps.
    // Whatever arrives there, only a fixed code may come out.
    type Reflect = fn(&StubRequest) -> String;
    let reflections: [(&str, Reflect); 3] = [
        ("the bearer token", |sent| {
            sent.bearer().unwrap_or_default().to_string()
        }),
        ("the prompt", |sent| {
            sent.json()["input"][0]["content"][0]["text"]
                .as_str()
                .unwrap_or_default()
                .to_string()
        }),
        ("a forged log line", |_| {
            "server_error\nForged: logged in as admin".to_string()
        }),
    ];
    let user = request().user;
    let bearer = bearer("current");
    for (shape, (label, _)) in terminal_events("").iter().enumerate() {
        for (name, reflect) in reflections {
            let backend = StubServer::start(move |sent, _| {
                StubResponse::stream(terminal_events(&reflect(sent))[shape].1.clone())
            });
            let dir = TestDir::new();
            let error = client(&backend, logged_in_store(&dir))
                .complete(&request())
                .unwrap_err();
            let shown = format!("{error} {error:?}");
            for forbidden in [
                bearer.as_str(),
                "eyJ",
                user.as_str(),
                "Wellington",
                "Forged",
            ] {
                assert!(
                    !shown.contains(forbidden),
                    "{label} reflecting {name}: {forbidden:?} in {shown}"
                );
            }
            assert!(
                matches!(&error, LlmError::Backend { code } if code == "unknown"),
                "{label} reflecting {name}: {error:?}"
            );
        }
    }
}

// Refreshes against concurrent clients and other processes.

/// Holds a stub's reply until the test opens it, and tells the test when a
/// request has arrived. The tests synchronise on the arrival, not on a
/// sleep, so "a refresh is in flight" is a fact when they act on it.
struct Gate {
    state: Mutex<GateState>,
    changed: Condvar,
}

#[derive(Default)]
struct GateState {
    arrived: usize,
    open: bool,
}

impl Gate {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(GateState::default()),
            changed: Condvar::new(),
        })
    }

    /// Called by the stub: counts the arrival, then waits until the gate
    /// opens. The wait is bounded so a failing test can't hang the suite.
    fn pass(&self) {
        let mut state = self.state.lock().unwrap();
        state.arrived += 1;
        self.changed.notify_all();
        let _ = self
            .changed
            .wait_timeout_while(state, Duration::from_secs(10), |state| !state.open)
            .unwrap();
    }

    /// Whether `count` requests have arrived within `timeout`.
    fn arrived(&self, count: usize, timeout: Duration) -> bool {
        let state = self.state.lock().unwrap();
        let (state, _) = self
            .changed
            .wait_timeout_while(state, timeout, |state| state.arrived < count)
            .unwrap();
        state.arrived >= count
    }

    fn open(&self) {
        self.state.lock().unwrap().open = true;
        self.changed.notify_all();
    }
}

/// How long a token-file write gets to finish while a refresh is held at
/// the issuer. Returning inside it means the write didn't wait for the
/// refresh. A process needs longer than a thread just to start.
const THREAD_GRACE: Duration = Duration::from_millis(300);
const PROCESS_GRACE: Duration = Duration::from_secs(1);

/// A server whose refresh waits at `gate` before rotating, and whose
/// `/responses` always completes.
fn gated_server(gate: &Arc<Gate>) -> StubServer {
    let gate = Arc::clone(gate);
    codex(
        move || {
            gate.pass();
            rotation()
        },
        || completion("{}"),
    )
}

#[test]
fn two_clients_on_one_store_share_one_refresh() {
    // Refresh tokens are single-use, so the lock has to belong to the token
    // store, not to one `CodexResponses`.
    let dir = TestDir::new();
    expired_store(&dir);
    let gate = Gate::new();
    let server = gated_server(&gate);
    let workers: Vec<_> = (0..2)
        .map(|_| {
            let client = client(&server, TokenStore::open(&dir.data()));
            std::thread::spawn(move || client.complete(&request()).map(|r| r.json))
        })
        .collect();
    assert!(
        gate.arrived(1, Duration::from_secs(5)),
        "no refresh reached the issuer"
    );
    let second_refresh = gate.arrived(2, THREAD_GRACE);
    gate.open();
    let results: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();

    assert!(!second_refresh, "a second refresh reached the issuer");
    for result in results {
        assert_eq!(result.unwrap(), json!({}));
    }
    let mut paths = server.paths();
    paths.sort();
    assert_eq!(paths, ["/oauth/token", "/responses", "/responses"]);
    for request in server.requests().iter().filter(|r| r.path == "/responses") {
        assert_eq!(request.bearer(), Some(bearer("rotated").as_str()));
    }
}

/// Selects [`token_store_in_a_child_process`] and what it does.
const CHILD_OP_ENV: &str = "ASPHODEL_TEST_TOKEN_STORE_OP";
const CHILD_DATA_ENV: &str = "ASPHODEL_TEST_TOKEN_STORE_DATA";

/// Not a test on its own: the cross-process tests re-run this binary with
/// only this selected, so a `TokenStore` in another process writes the
/// file, as `asphodel llm login` does. Without the variables it does
/// nothing.
#[test]
fn token_store_in_a_child_process() {
    let Ok(op) = std::env::var(CHILD_OP_ENV) else {
        return;
    };
    let store = TokenStore::open(&PathBuf::from(std::env::var_os(CHILD_DATA_ENV).unwrap()));
    match op.as_str() {
        "save" => store.save(&relogin_tokens()).unwrap(),
        "clear" => store.clear().unwrap(),
        other => panic!("unknown op {other}"),
    }
}

/// A child process running `op` (`save` or `clear`) on the token store in
/// `data`, killed on drop.
struct ChildProcess(Child);

impl ChildProcess {
    fn token_store(op: &str, data: &Path) -> Self {
        let child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "token_store_in_a_child_process",
                "--test-threads=1",
            ])
            .env(CHILD_OP_ENV, op)
            .env(CHILD_DATA_ENV, data)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        Self(child)
    }

    /// Its exit status, if it exits within `timeout`.
    fn exited_within(&mut self, timeout: Duration) -> Option<ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                return Some(status);
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for ChildProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn a_save_or_clear_from_another_process_waits_for_a_refresh_and_wins() {
    for op in ["save", "clear"] {
        // Hold a refresh at the issuer, run `op` from another process, then
        // let the refresh finish.
        let dir = TestDir::new();
        let gate = Gate::new();
        let server = gated_server(&gate);
        let client = client(&server, expired_store(&dir));
        let worker = std::thread::spawn(move || client.complete(&request()));
        assert!(
            gate.arrived(1, Duration::from_secs(5)),
            "the refresh never reached the issuer"
        );
        let mut child = ChildProcess::token_store(op, &dir.data());
        let during_refresh = child.exited_within(PROCESS_GRACE);
        gate.open();
        let status = during_refresh
            .or_else(|| child.exited_within(Duration::from_secs(10)))
            .unwrap_or_else(|| panic!("the child's {op} never finished"));
        assert!(status.success(), "the child's {op} failed: {status}");
        worker.join().unwrap().unwrap();

        assert!(
            during_refresh.is_none(),
            "the other process's {op} didn't wait for the refresh in flight"
        );
        let file = TokenStore::open(&dir.data()).load().unwrap();
        assert_eq!(
            file.as_ref().map(|tokens| tokens.refresh_token.expose()),
            (op == "save").then_some("rt-three"),
            "the refresh overwrote the other process's {op}"
        );
    }
}

#[test]
fn a_clear_before_a_401_is_not_undone_by_the_refresh() {
    let dir = TestDir::new();
    let store = logged_in_store(&dir);
    let token_path = store.path().to_path_buf();
    let gate = Gate::new();
    let held = Arc::clone(&gate);
    let server = codex(rotation, move || {
        held.pass();
        StubResponse::status(401)
    });
    let client = client(&server, store);
    let worker = std::thread::spawn(move || (client.complete(&request()), client.refreshes()));
    assert!(
        gate.arrived(1, Duration::from_secs(5)),
        "the Responses request never reached the backend"
    );
    let mut child = ChildProcess::token_store("clear", &dir.data());
    let status = child.exited_within(Duration::from_secs(5));
    let cleared = !token_path.exists();
    gate.open();
    let (result, refreshes) = worker.join().unwrap();

    assert!(status.expect("the child clear never finished").success());
    assert!(cleared, "the child clear left the token file behind");
    assert!(matches!(result, Err(LlmError::LoginRequired)), "{result:?}");
    assert_eq!(refreshes, 0, "a cleared login must not be refreshed");
    assert_eq!(server.paths(), ["/responses"]);
    assert!(!token_path.exists(), "the refresh recreated the token file");
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
    let mut settings = chatgpt_settings(CODEX_ENDPOINT);
    settings.model = std::env::var("ASPHODEL_LLM_MODEL").unwrap_or_else(|_| "gpt-5.1".into());
    settings.reasoning_effort = std::env::var("ASPHODEL_LLM_REASONING_EFFORT").ok();
    settings.timeout = Duration::from_secs(120);
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
