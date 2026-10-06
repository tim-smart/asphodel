//! Local embedding and reranking models and the OpenAI-compatible LLM client
//! follow their loading and transport contracts.
//!
//! Nothing here touches the network. The OpenAI-compatible client is tested
//! against [`StubServer`], a loopback HTTP/1.1 server in this file, and the
//! model loader against files written into a temp dir. The real-model tests
//! are ignored and run only when the real models are in
//! `ASPHODEL_MODEL_DIR` (or the XDG cache).

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use asphodel_core::clock::{Clock, SimulatedClock, Sleeper};
use asphodel_core::config::{ConfigError, Deployment, Secret, Tuning};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::vector::EMBEDDING_DIMENSIONS;
use asphodel_core::store::{OpenOptions, Store};
use jiff::{SignedDuration, Timestamp};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use asphodel_core::models::*;
use asphodel_core::{OpenError, Service};

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-models-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn models(&self) -> ModelDir {
        ModelDir::at(self.join("models"))
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

fn clock() -> Arc<dyn Clock> {
    Arc::new(SimulatedClock::new(start()))
}

/// A service on the fake models over a fresh store in `dir`.
fn open_service(dir: &TestDir, tuning: Tuning) -> Result<Service, OpenError> {
    let store = Store::open(&dir.join("data"), OpenOptions::default(), clock()).unwrap();
    Service::with_models(clock(), store, tuning, Models::fake())
}

/// A floor for each fake model, and a relevance scale for `scaled`.
fn tuning(scaled: &str) -> Tuning {
    Tuning::from_toml(&format!(
        "[injection.reranker_floors]\n\"{}\" = 0.0\n\
         [reconcile.embedding_floors]\n\"{}\" = 0.5\n\
         [ranking.relevance_scales]\n\"{scaled}\" = 1.0\n",
        FakeReranker::MODEL_ID,
        FakeEmbedder::MODEL_ID,
    ))
    .unwrap()
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

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// A manifest of two small models with known bytes, for the fetch tests.
/// The real manifest's checksums can't be met offline, so `fetch_models`
/// takes the specs it fills.
struct Canned {
    specs: Vec<ModelSpec>,
    bytes: BTreeMap<String, Vec<u8>>,
}

fn canned() -> Canned {
    let mut bytes = BTreeMap::new();
    let specs = [
        (EMBEDDING_MODEL_ID, "bge-small-en-v1.5-int8"),
        (RERANKER_MODEL_ID, "jina-reranker-v1-turbo-en-int8"),
    ]
    .map(|(id, dir)| {
        let files = MODEL_FILES.iter().map(|name| {
            let url = format!("https://models.example/{dir}/{name}");
            let content = format!("{id} {name} bytes").into_bytes();
            let sha256 = sha256_hex(&content);
            bytes.insert(url.clone(), content);
            let name = name.to_string();
            ModelFile { name, url, sha256 }
        });
        let (id, dir) = (id.to_string(), dir.to_string());
        ModelSpec {
            id,
            dir,
            files: files.collect(),
        }
    })
    .to_vec();
    Canned { specs, bytes }
}

/// Serves canned bytes by URL and counts calls.
#[derive(Default)]
struct MapFetcher {
    bytes: BTreeMap<String, Vec<u8>>,
    calls: AtomicUsize,
    /// URLs that fail instead of answering.
    failing: Vec<String>,
}

impl MapFetcher {
    fn new(bytes: BTreeMap<String, Vec<u8>>) -> Self {
        Self {
            bytes,
            ..Self::default()
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }
}

impl Fetcher for MapFetcher {
    fn fetch(&self, url: &str) -> Result<Vec<u8>, FetchError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.failing.iter().any(|failing| failing == url) {
            return Err(FetchError::Status(503));
        }
        self.bytes.get(url).cloned().ok_or(FetchError::Status(404))
    }
}

fn request() -> LlmRequest {
    LlmRequest {
        template: Template {
            name: "extract".into(),
            version: 3,
            guidance: None,
            attempt: None,
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

/// A loopback HTTP/1.1 server with one scripted response. It records every
/// request it got. Each connection is answered and closed.
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
}

#[derive(Debug, Clone)]
struct StubResponse {
    status: u16,
    /// Headers sent besides the content type, length and connection.
    headers: Vec<(String, String)>,
    body: String,
    /// Held before answering, for the timeout test.
    delay: Duration,
}

impl StubResponse {
    fn new(status: u16, body: Value) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: body.to_string(),
            delay: Duration::ZERO,
        }
    }

    fn json(value: Value) -> Self {
        Self::new(200, value)
    }

    fn status(status: u16) -> Self {
        Self::new(status, json!({}))
    }

    fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// An OpenAI-style completion whose content is `content`.
    fn completion(content: &str) -> Self {
        Self::json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "some-model:q4_K_M",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 41, "completion_tokens": 7, "total_tokens": 48 }
        }))
    }
}

impl StubServer {
    fn start(response: StubResponse) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&requests);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let response = response.clone();
                let log = Arc::clone(&log);
                std::thread::spawn(move || serve_one(stream, response, &log));
            }
        });
        Self {
            url: format!("http://127.0.0.1:{port}"),
            requests,
        }
    }

    fn only_request(&self) -> StubRequest {
        let requests = self.requests.lock().unwrap().clone();
        assert_eq!(requests.len(), 1, "{requests:?}");
        requests.into_iter().next().unwrap()
    }

    fn settings(&self, api_key: Option<&str>) -> LlmSettings {
        LlmSettings {
            auth: LlmAuth::ApiKey,
            endpoint: format!("{}/v1", self.url),
            model: "some-model:q4_K_M".into(),
            reasoning_effort: None,
            api_key: api_key.map(Secret::new),
            timeout: Duration::from_secs(5),
        }
    }
}

fn serve_one(mut stream: TcpStream, response: StubResponse, log: &Mutex<Vec<StubRequest>>) {
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
    log.lock().unwrap().push(request);
    std::thread::sleep(response.delay);
    let extra: String = response
        .headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect();
    let _ = write!(
        stream,
        "HTTP/1.1 {} Status\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n{}",
        response.status,
        response.body.len(),
        response.body
    );
    let _ = stream.flush();
}

/// One call to a client on a fresh stub answering `response`.
fn complete(response: StubResponse) -> Result<LlmResponse, LlmError> {
    complete_with(&StubServer::start(response), |_| {})
}

/// One call to a client on `server`, with its settings edited.
fn complete_with(
    server: &StubServer,
    edit: impl FnOnce(&mut LlmSettings),
) -> Result<LlmResponse, LlmError> {
    let mut settings = server.settings(None);
    edit(&mut settings);
    OpenAiCompatible::new(settings).complete(&request())
}

#[test]
fn fetch_fills_the_dir_then_skips_what_is_right_and_replaces_what_is_not() {
    let dir = TestDir::new();
    let models = dir.models();
    let canned = canned();
    let fetcher = MapFetcher::new(canned.bytes.clone());

    let report = fetch_models(&models, &canned.specs, &fetcher).unwrap();
    assert_eq!((report.fetched.len(), report.skipped.len()), (10, 0));
    for spec in &canned.specs {
        for file in &spec.files {
            let written = std::fs::read(models.file(spec, &file.name)).unwrap();
            assert_eq!(sha256_hex(&written), file.sha256, "{}", file.url);
        }
        // Nothing else was left behind: no temp files.
        let entries = std::fs::read_dir(models.path().join(&spec.dir)).unwrap();
        assert_eq!(entries.count(), MODEL_FILES.len());
    }

    let report = fetch_models(&models, &canned.specs, &fetcher).unwrap();
    assert_eq!(fetcher.calls(), 10, "a second run fetched again");
    assert_eq!((report.fetched.len(), report.skipped.len()), (0, 10));

    let spec = &canned.specs[1];
    let path = models.file(spec, "model.onnx");
    std::fs::write(&path, b"truncated").unwrap();
    let report = fetch_models(&models, &canned.specs, &fetcher).unwrap();
    assert_eq!(report.fetched, std::slice::from_ref(&path));
    assert_eq!(report.skipped.len(), 9);
    assert_eq!(
        sha256_hex(&std::fs::read(&path).unwrap()),
        spec.files[0].sha256
    );
}

#[test]
fn fetch_stops_at_a_bad_file_keeps_what_came_before_and_resumes() {
    let canned = canned();
    // The third file of the first model either has bytes that don't match
    // the manifest or a source that is down.
    let failing = canned.specs[0].files[2].url.clone();
    let mut tampered = canned.bytes.clone();
    tampered.insert(failing.clone(), b"not the model".to_vec());
    type Expected = fn(&FetchFailure, &str) -> bool;
    let checksum: Expected =
        |error, failing| matches!(error, FetchFailure::Checksum { url } if url == failing);
    let down: Expected = |error, failing| matches!(error, FetchFailure::Fetch { url, error: FetchError::Status(503) } if url == failing);
    for (bytes, source_down, expected) in [
        (tampered, false, checksum),
        (canned.bytes.clone(), true, down),
    ] {
        let dir = TestDir::new();
        let models = dir.models();
        let mut fetcher = MapFetcher::new(bytes);
        if source_down {
            fetcher.failing.push(failing.clone());
        }

        let error = fetch_models(&models, &canned.specs, &fetcher).unwrap_err();
        assert!(expected(&error, &failing), "{error:?}");
        assert_eq!(fetcher.calls(), 3, "{error:?}");
        // The two files before it stay; the bad bytes were never written,
        // and no temp file was left behind.
        let mut entries: Vec<_> = std::fs::read_dir(models.path().join(&canned.specs[0].dir))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        entries.sort();
        let mut kept = MODEL_FILES[..2].to_vec();
        kept.sort();
        assert_eq!(entries, kept, "{error:?}");

        // Once the source recovers, the next run picks up where it stopped.
        fetcher.bytes = canned.bytes.clone();
        fetcher.failing.clear();
        let report = fetch_models(&models, &canned.specs, &fetcher).unwrap();
        assert_eq!((report.fetched.len(), report.skipped.len()), (8, 2));
    }
}

// Loading: never a download, and a bad file fails fast.

#[test]
fn loading_fails_fast_on_a_missing_or_corrupt_file() {
    let dir = TestDir::new();
    let models = dir.models();
    let manifest = manifest();
    let expected = models.file(&manifest[0], "model.onnx");

    let error = Models::load(&models, &ModelOptions::default()).unwrap_err();
    assert!(
        matches!(&error, ModelError::MissingFile { model, path } if model == EMBEDDING_MODEL_ID && *path == expected),
        "{error:?}"
    );
    // Fail fast means fail without side effects: nothing was created.
    assert!(!models.path().exists(), "the loader created the model dir");

    // Every file present, none with the manifest's bytes. The loader checks
    // checksums before building a session, so the error is ours and names
    // the file, not an ONNX Runtime message about a bad protobuf.
    for spec in &manifest {
        for file in &spec.files {
            let path = models.file(spec, &file.name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, format!("garbage for {}", file.name)).unwrap();
        }
    }
    let error = Models::load(&models, &ModelOptions::default()).unwrap_err();
    assert!(
        matches!(&error, ModelError::Checksum { model, path } if model == EMBEDDING_MODEL_ID && *path == expected),
        "{error:?}"
    );
}

// The service: recorded model ids and the floor check at startup.

#[test]
fn the_service_opens_only_with_floors_for_the_exact_loaded_model_ids() {
    // A missing floor or relevance scale for a loaded model stops the
    // daemon. Each is keyed by the exact model string, quantisation
    // included, with no fallback. The check lives in the service so replay
    // gets it too.
    let embedding_floor = format!("reconcile.embedding_floors.\"{}\"", FakeEmbedder::MODEL_ID);
    let reranker_floor = format!("injection.reranker_floors.\"{}\"", FakeReranker::MODEL_ID);
    let scale = format!("ranking.relevance_scales.\"{}\"", FakeReranker::MODEL_ID);
    for (tuning, missing) in [
        (
            Tuning::default(),
            vec![embedding_floor, reranker_floor, scale.clone()],
        ),
        (tuning("fake-reranker"), vec![scale.clone()]),
        (tuning("fake-reranker:v2"), vec![scale.clone()]),
        (tuning("FAKE-RERANKER:V1"), vec![scale]),
        (tuning(FakeReranker::MODEL_ID), vec![]),
    ] {
        let dir = TestDir::new();
        let keys: Vec<String> = match open_service(&dir, tuning) {
            Ok(_) => Vec::new(),
            Err(OpenError::Config(ConfigError::Invalid(errors))) => {
                errors.into_iter().map(|error| error.key).collect()
            }
            Err(error) => panic!("{error}"),
        };
        assert_eq!(keys, missing);
    }
}

#[test]
fn a_new_bank_records_the_loaded_models() {
    // A bank records the loaded embedding and reranker model ids, not the
    // caller's, and a merge leaves them alone: a change goes through
    // `asphodel reembed`, never through bank config.
    let dir = TestDir::new();
    let service = open_service(&dir, tuning(FakeReranker::MODEL_ID)).unwrap();
    let merged = BankIdentity {
        owner_name: Some("Tim".into()),
        ..BankIdentity::default()
    };
    for (identity, created) in [(BankIdentity::default(), true), (merged, false)] {
        let bank = service.ensure_bank_with_models("tim", &identity).unwrap();
        assert_eq!(bank.created, created);
        assert_eq!(bank.embedding_model, FakeEmbedder::MODEL_ID);
        assert_eq!(bank.reranker_model, FakeReranker::MODEL_ID);
    }
}

#[test]
fn llm_settings_come_from_the_tuning_file_and_the_environment() {
    let resolve = |toml: &str, key: Option<&str>| {
        LlmSettings::from_config(&Tuning::from_toml(toml).unwrap(), &deployment(key))
    };
    let configured = |toml: &str, key| resolve(toml, key).unwrap().expect("configured");
    let api = "[llm]\nmodel = \"some-model:q4_K_M\"\nendpoint = \"http://llm.internal:8080/v1\"\n";
    let settings = configured(api, Some("sk-live-41b2e8-secret"));
    // Without `auth`, the mode is the API key one.
    assert_eq!(settings.auth, LlmAuth::ApiKey);
    assert_eq!(settings.endpoint, "http://llm.internal:8080/v1");
    assert_eq!(settings.model, "some-model:q4_K_M");
    assert_eq!(
        settings.api_key.as_ref().map(Secret::expose),
        Some("sk-live-41b2e8-secret")
    );
    // A local endpoint needs no key.
    assert!(configured(api, None).api_key.is_none());

    // The subscription mode needs no endpoint, takes one that is given, and
    // carries a reasoning effort.
    let subscription = "[llm]\nauth = \"chatgpt\"\nmodel = \"gpt-5.1\"\n";
    let chatgpt = configured(&format!("{subscription}reasoning_effort = \"low\"\n"), None);
    assert_eq!(chatgpt.auth, LlmAuth::Chatgpt);
    assert_eq!(chatgpt.model, "gpt-5.1");
    assert_eq!(chatgpt.reasoning_effort.as_deref(), Some("low"));
    assert!(chatgpt.api_key.is_none());
    let proxy = format!("{subscription}endpoint = \"https://proxy.internal/codex\"\n");
    assert_eq!(
        configured(&proxy, None).endpoint,
        "https://proxy.internal/codex"
    );

    // Nothing set is no LLM; half set is an error. The model stays required
    // in the subscription mode too: calibration runs against one model.
    assert!(resolve("", None).unwrap().is_none());
    for (toml, missing) in [
        ("[llm]\nmodel = \"some-model\"\n", "llm.endpoint"),
        ("[llm]\nendpoint = \"http://llm.internal\"\n", "llm.model"),
        ("[llm]\nauth = \"chatgpt\"\n", "llm.model"),
    ] {
        let error = resolve(toml, None).unwrap_err();
        assert!(
            matches!(error, LlmError::NotConfigured { missing: got } if got == missing),
            "{toml}: {error:?}"
        );
    }
}

// The OpenAI-compatible client, against the loopback stub.

#[test]
fn the_client_posts_a_structured_chat_completion() {
    let server = StubServer::start(StubResponse::completion(
        "{\"claims\":[\"Tim moved to Wellington in March 2026.\"]}",
    ));
    let client = OpenAiCompatible::new(server.settings(Some("sk-live-41b2e8-secret")));
    assert_eq!(client.model(), "some-model:q4_K_M");

    let response = client.complete(&request()).unwrap();
    let usage = LlmUsage {
        input_tokens: 41,
        output_tokens: 7,
    };
    let claims = json!({"claims": ["Tim moved to Wellington in March 2026."]});
    assert_eq!((response.json, response.usage), (claims, Some(usage)));
    assert!(response.latency > Duration::ZERO);

    let sent = server.only_request();
    assert_eq!(sent.method, "POST");
    assert_eq!(sent.path, "/v1/chat/completions");
    assert_eq!(
        sent.header("authorization"),
        Some("Bearer sk-live-41b2e8-secret")
    );
    let content_type = sent.header("content-type").unwrap_or_default();
    assert!(
        content_type.starts_with("application/json"),
        "{content_type}"
    );
    // Not streamed.
    assert_eq!(
        serde_json::from_str::<Value>(&sent.body).unwrap(),
        json!({
            "model": "some-model:q4_K_M",
            "messages": [
                {"role": "system", "content": "You extract memories."},
                {"role": "user", "content": "Tim said: I moved to Wellington in March."}
            ],
            "temperature": 0,
            "max_tokens": 512,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "claims", "strict": true, "schema": request().schema}
            }
        })
    );

    // A trailing slash on the endpoint doesn't double the path.
    let server = StubServer::start(StubResponse::completion("{}"));
    complete_with(&server, |settings| settings.endpoint.push('/')).unwrap();
    assert_eq!(server.only_request().path, "/v1/chat/completions");
}

#[test]
fn the_client_answers_at_temperature_zero_so_it_skips_sending_a_request_again() {
    // A refresh sends a write a second time, marked as an attempt, hoping
    // for a different reply. At temperature 0 the same request gets the
    // same reply, so the client skips it, and only it.
    let server = StubServer::start(StubResponse::completion("{}"));
    let client = OpenAiCompatible::new(server.settings(None));
    let mut again = serde_json::to_value(request()).unwrap();
    again["template"]["attempt"] = json!(1);
    let again: LlmRequest = serde_json::from_value(again).unwrap();
    assert!(!client.skips_write(&request(), &[]));
    assert!(client.skips_write(&again, &[]));
    assert!(server.requests.lock().unwrap().is_empty());
}

#[test]
fn reply_content_is_json_fenced_or_not_and_anything_else_is_an_error() {
    for content in [
        "{\"a\":1}",
        "```json\n{\"a\":1}\n```",
        "```\n{\"a\":1}\n```",
        "  ```json\n{\"a\":1}\n```  ",
        "```{\"a\":1}```",
    ] {
        let response = complete(StubResponse::completion(content)).unwrap();
        assert_eq!(response.json, json!({"a": 1}), "{content:?}");
    }

    let prose = "Sure! Here are Tim's claims: he moved to Wellington.";
    let not_json = complete(StubResponse::completion(prose)).unwrap_err();
    assert!(
        matches!(not_json, LlmError::NotJson { bytes } if bytes == prose.len()),
        "{not_json:?}"
    );
    // No content in logs: the reply is Tim's data.
    let shown = format!("{not_json} {not_json:?}");
    assert!(!shown.contains("Wellington"), "{shown}");

    let no_choices = complete(StubResponse::json(json!({"choices": []}))).unwrap_err();
    assert!(matches!(no_choices, LlmError::NoContent), "{no_choices:?}");
    let refusal = json!({"role": "assistant", "content": null, "refusal": "I can't help."});
    let reply = json!({"choices": [{"message": refusal}]});
    let refused = complete(StubResponse::json(reply)).unwrap_err();
    assert!(matches!(refused, LlmError::Refused), "{refused:?}");
    for error in [not_json, no_choices, refused] {
        assert_eq!(error.retry(), Retry::Never, "{error:?}");
    }
}

/// `Retry-After` as an HTTP date, the header's other form (RFC 9110).
fn retry_after_date(from_now: SignedDuration) -> String {
    let at = asphodel_core::clock::SystemClock.now() + from_now;
    jiff::fmt::rfc2822::DateTimePrinter::new()
        .timestamp_to_rfc9110_string(&at)
        .unwrap()
}

#[test]
fn an_http_error_is_retried_or_held_or_neither() {
    // A hold says when to come back, so every caller waits that long and
    // nothing is counted: a deferral, not a retry. A Retry-After that's
    // neither seconds nor a date counts like none, and on any status but
    // 429 it changes nothing. A 429 without one, 502, 503 and 504 say the
    // provider is down; 408, 500 and any other 5xx may be the request's
    // own doing.
    enum Expected {
        Fatal,
        Down,
        Request,
        Hold(RangeInclusive<Duration>),
    }
    use Expected::*;
    let secs = Duration::from_secs;
    let after = |value: &str| Some(value.to_string());
    let at = |mins| Some(retry_after_date(SignedDuration::from_mins(mins)));
    let cases = [
        (400, None, Fatal),
        (401, None, Fatal),
        (404, None, Fatal),
        (408, None, Request),
        (429, None, Down),
        (500, None, Request),
        (501, None, Request),
        (502, None, Down),
        (503, None, Down),
        (504, None, Down),
        (505, None, Request),
        (429, after("30"), Hold(secs(30)..=secs(30))),
        (429, after("soon"), Down),
        (503, after("30"), Down),
        (429, at(10), Hold(secs(8 * 60)..=secs(10 * 60))),
        (429, at(-10), Hold(Duration::ZERO..=Duration::ZERO)),
    ];
    for (status, retry_after, expected) in cases {
        let mut response = StubResponse::status(status);
        if let Some(value) = &retry_after {
            response = response.with_header("Retry-After", value);
        }
        let error = complete(response).unwrap_err();
        let case = format!("{status} {retry_after:?}: {error:?}");
        if let Hold(range) = &expected {
            assert!(
                matches!(&error, LlmError::RateLimited { retry_after } if range.contains(retry_after)),
                "{case}"
            );
        } else {
            assert!(
                matches!(error, LlmError::Status { status: got } if got == status),
                "{case}"
            );
        }
        let retry = match expected {
            Fatal | Hold(_) => Retry::Never,
            Down => Retry::ProviderDown,
            Request => Retry::MaybeTheRequest,
        };
        assert_eq!(error.retry(), retry, "{case}");
    }
}

#[test]
fn a_slow_endpoint_may_be_the_requests_fault_and_a_dead_one_is_the_providers() {
    let mut slow = StubResponse::completion("{}");
    slow.delay = Duration::from_secs(3);
    let server = StubServer::start(slow);
    let error = complete_with(&server, |settings| {
        settings.timeout = Duration::from_millis(200)
    })
    .unwrap_err();
    assert!(matches!(error, LlmError::Timeout), "{error:?}");
    // A reply that reliably runs past the timeout is the request's doing.
    assert_eq!(error.retry(), Retry::MaybeTheRequest);

    // Bind and drop, so the port is closed.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let closed = listener.local_addr().unwrap();
    drop(listener);
    let error = complete_with(&server, |settings| {
        settings.endpoint = format!("http://{closed}/v1")
    })
    .unwrap_err();
    assert!(matches!(error, LlmError::Transport { .. }), "{error:?}");
    assert_eq!(error.retry(), Retry::ProviderDown);
}

// Retrying a transient error within the call.

/// One call's outcome, and how long the call takes on the simulated clock.
type Step = (SignedDuration, Result<Value, LlmError>);

/// A client that plays one [`Step`] per call, so a test can fail a call
/// with any [`LlmError`] and make it slow.
struct Steps {
    clock: Arc<SimulatedClock>,
    steps: Mutex<std::collections::VecDeque<Step>>,
    calls: AtomicUsize,
}

impl LlmClient for Steps {
    fn model(&self) -> &str {
        "steps"
    }

    fn complete(&self, _: &LlmRequest) -> Result<LlmResponse, LlmError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let (takes, outcome) = self.steps.lock().unwrap().pop_front().expect("a step");
        self.clock.advance(takes);
        outcome.map(|json| LlmResponse {
            json,
            usage: None,
            latency: Duration::ZERO,
        })
    }
}

/// Records each wait instead of sleeping, and moves the clock on by it.
struct Waits {
    clock: Arc<SimulatedClock>,
    waits: Mutex<Vec<Duration>>,
    elapsed: Option<Duration>,
}

impl Sleeper for Waits {
    fn sleep(&self, wait: Duration) {
        self.waits.lock().unwrap().push(wait);
        self.clock
            .advance(SignedDuration::try_from(self.elapsed.unwrap_or(wait)).unwrap());
    }
}

/// A bounded policy, the shape translate has: four attempts at most for
/// either kind of failure, waiting 1s, 2s, then 3s (capped from 4s) before
/// them, and none started 30s after the first.
fn retry_policy() -> RetryPolicy {
    RetryPolicy {
        first_wait: Duration::from_secs(1),
        max_wait: Duration::from_secs(3),
        provider_attempts: Some(4),
        request_attempts: 4,
        budget: Some(Duration::from_secs(30)),
    }
}

/// The shape extraction and refresh have: the provider being down is
/// retried until it recovers, backing off from 1s to 60s, and a failure
/// the request may cause is returned at once for the caller to count.
fn daemon_policy() -> RetryPolicy {
    RetryPolicy {
        first_wait: Duration::from_secs(1),
        max_wait: Duration::from_secs(60),
        provider_attempts: None,
        request_attempts: 1,
        budget: None,
    }
}

/// One call through [`LlmRetry`] to a client playing `steps`: its result,
/// how many calls reached the client, and the waits between them.
fn retried(steps: Vec<Step>) -> (Result<LlmResponse, LlmError>, usize, Vec<Duration>) {
    retried_with_sleep(steps, None)
}

/// Optionally simulate scheduler oversleep instead of the requested wait.
fn retried_with_sleep(
    steps: Vec<Step>,
    elapsed: Option<Duration>,
) -> (Result<LlmResponse, LlmError>, usize, Vec<Duration>) {
    retried_under(retry_policy(), steps, elapsed)
}

/// [`retried_with_sleep`] under `policy`.
fn retried_under(
    policy: RetryPolicy,
    steps: Vec<Step>,
    elapsed: Option<Duration>,
) -> (Result<LlmResponse, LlmError>, usize, Vec<Duration>) {
    let clock = Arc::new(SimulatedClock::new(start()));
    let inner = Arc::new(Steps {
        clock: clock.clone(),
        steps: Mutex::new(steps.into()),
        calls: AtomicUsize::new(0),
    });
    let waits = Arc::new(Waits {
        clock: clock.clone(),
        waits: Mutex::new(Vec::new()),
        elapsed,
    });
    let retry = LlmRetry::new(inner.clone(), policy, clock, waits.clone());
    let result = retry.complete(&request());
    let waits = waits.waits.lock().unwrap().clone();
    (result, inner.calls.load(Ordering::SeqCst), waits)
}

fn quick(outcome: Result<Value, LlmError>) -> Step {
    (SignedDuration::ZERO, outcome)
}

/// Each wait is jittered down to no less than half its backoff: 1s, 2s,
/// then 3s once doubling passes the cap.
fn assert_backed_off(waits: &[Duration], case: &str) {
    let secs = Duration::from_secs;
    let backoffs = [secs(1), secs(2), secs(3)];
    assert!(waits.len() <= backoffs.len(), "{case}: {waits:?}");
    for (wait, backoff) in waits.iter().zip(backoffs) {
        assert!(
            *wait >= backoff / 2 && *wait <= backoff,
            "{case}: waited {wait:?} for a {backoff:?} backoff"
        );
    }
}

#[test]
fn a_transient_error_is_retried_within_the_call_and_nothing_else_is() {
    let ok = || Ok(json!({"claims": []}));
    let status = |status| Err(LlmError::Status { status });
    let backend = |code: &str| {
        Err(LlmError::Backend {
            code: code.to_string(),
        })
    };

    // Transient failures short of the cap end in the reply.
    let transient = [
        vec![quick(Err(LlmError::Timeout)), quick(ok())],
        vec![quick(status(429)), quick(status(408)), quick(ok())],
        vec![
            quick(Err(LlmError::Transport {
                reason: "connection reset".into(),
            })),
            quick(status(502)),
            quick(backend("server_error")),
            quick(ok()),
        ],
    ];
    for steps in transient {
        let attempts = steps.len();
        let (result, calls, waits) = retried(steps);
        let case = format!("{attempts} attempts");
        assert_eq!(result.unwrap().json, json!({"claims": []}), "{case}");
        assert_eq!((calls, waits.len()), (attempts, attempts - 1), "{case}");
        assert_backed_off(&waits, &case);
    }

    // Exhausting the attempts returns the last error.
    let (result, calls, waits) = retried(vec![
        quick(status(502)),
        quick(status(503)),
        quick(status(504)),
        quick(status(503)),
        quick(ok()),
    ]);
    let error = result.unwrap_err();
    assert!(
        matches!(error, LlmError::Status { status: 503 }),
        "{error:?}"
    );
    assert_eq!((calls, waits.len()), (4, 3));
    assert_backed_off(&waits, "exhausted");

    // Anything else is returned at once: a wrong reply, a hold the gate
    // shares, a login, a deterministic backend failure.
    let resets_at = start() + SignedDuration::from_hours(1);
    let fatal: Vec<(&str, Result<Value, LlmError>)> = vec![
        ("not json", Err(LlmError::NotJson { bytes: 3 })),
        ("no content", Err(LlmError::NoContent)),
        ("refused", Err(LlmError::Refused)),
        ("login", Err(LlmError::LoginRequired)),
        ("usage", Err(LlmError::UsageLimited { resets_at })),
        (
            "rate",
            Err(LlmError::RateLimited {
                retry_after: Duration::from_secs(5),
            }),
        ),
        ("bad request", status(400)),
        ("max_output_tokens", backend("max_output_tokens")),
    ];
    for (case, outcome) in fatal {
        let expected = format!("{:?}", outcome.as_ref().unwrap_err());
        let (result, calls, waits) = retried(vec![quick(outcome), quick(ok())]);
        let error = result.unwrap_err();
        assert_eq!(format!("{error:?}"), expected, "{case}");
        assert_eq!((calls, waits), (1, vec![]), "{case}");
    }

    // A call that ends past the budget isn't retried, however it failed: a
    // stalled endpoint doesn't hold a call for minutes. Nor is one whose
    // backoff, at least 0.5s here, would start the next attempt past it,
    // and it doesn't wait for nothing. One that ends well inside it is.
    let slow = SignedDuration::from_secs(31);
    let (result, calls, _) = retried(vec![(slow, Err(LlmError::Timeout)), quick(ok())]);
    assert!(matches!(result, Err(LlmError::Timeout)), "{result:?}");
    assert_eq!(calls, 1);
    let edge = SignedDuration::from_millis(29_800);
    let (result, calls, waits) = retried(vec![(edge, Err(LlmError::Timeout)), quick(ok())]);
    assert!(matches!(result, Err(LlmError::Timeout)), "{result:?}");
    assert_eq!((calls, waits), (1, vec![]));
    let inside = SignedDuration::from_secs(20);
    let (result, calls, _) = retried(vec![(inside, Err(LlmError::Timeout)), quick(ok())]);
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(calls, 2);

    // The planned wait fits, but scheduler oversleep reaches or passes
    // the cutoff. Return the last error without starting another call.
    for sleep_secs in [10, 11] {
        let (result, calls, waits) = retried_with_sleep(
            vec![(inside, status(503)), quick(ok())],
            Some(Duration::from_secs(sleep_secs)),
        );
        assert!(
            matches!(result, Err(LlmError::Status { status: 503 })),
            "slept {sleep_secs}s: {result:?}"
        );
        assert_eq!((calls, waits.len()), (1, 1), "slept {sleep_secs}s");
        assert_backed_off(&waits, "oversleep");
    }
}

#[test]
fn the_provider_being_down_is_retried_until_it_recovers_and_a_request_fault_as_the_caller_allows() {
    let ok = || Ok(json!({"claims": []}));
    let status = |status| Err(LlmError::Status { status });
    let backend = |code: &str| {
        Err(LlmError::Backend {
            code: code.to_string(),
        })
    };

    // Unreachable, overloaded or limiting: retried past any attempt count,
    // the backoff doubling from 1s to its 60s cap.
    let down = vec![
        quick(Err(LlmError::Transport {
            reason: "connection refused".into(),
        })),
        quick(status(502)),
        quick(status(503)),
        quick(status(504)),
        quick(status(429)),
        quick(backend("server_is_overloaded")),
        quick(backend("slow_down")),
        quick(backend("rate_limit_exceeded")),
        quick(status(503)),
        quick(status(503)),
        quick(status(503)),
        quick(ok()),
    ];
    let failures = down.len() - 1;
    let (result, calls, waits) = retried_under(daemon_policy(), down, None);
    assert_eq!(result.unwrap().json, json!({"claims": []}));
    assert_eq!((calls, waits.len()), (failures + 1, failures));
    for (doublings, wait) in waits.iter().enumerate() {
        let backoff = Duration::from_secs(1 << doublings.min(16)).min(Duration::from_secs(60));
        assert!(
            *wait >= backoff / 2 && *wait <= backoff,
            "waited {wait:?} for a {backoff:?} backoff"
        );
    }

    // A failure the request may cause goes back at once when the caller
    // counts it itself, as extraction and refresh do.
    let request_faults: Vec<(&str, Result<Value, LlmError>)> = vec![
        ("timeout", Err(LlmError::Timeout)),
        ("408", status(408)),
        ("500", status(500)),
        ("501", status(501)),
        ("server_error", backend("server_error")),
        ("interrupted", backend("interrupted")),
    ];
    for (case, outcome) in request_faults {
        let expected = format!("{:?}", outcome.as_ref().unwrap_err());
        let (result, calls, waits) =
            retried_under(daemon_policy(), vec![quick(outcome), quick(ok())], None);
        assert_eq!(format!("{:?}", result.unwrap_err()), expected, "{case}");
        assert_eq!((calls, waits), (1, vec![]), "{case}");
    }

    // A caller with no count of its own, as replay, retries them a few
    // times. The provider being down in between uses up none of those.
    let replay = RetryPolicy {
        request_attempts: 5,
        ..daemon_policy()
    };
    let faults = |n| {
        (0..n)
            .map(|_| quick(backend("server_error")))
            .collect::<Vec<Step>>()
    };
    let mut recovers = faults(4);
    recovers.push(quick(ok()));
    let (result, calls, _) = retried_under(replay, recovers, None);
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(calls, 5);
    let mut never = faults(5);
    never.push(quick(ok()));
    let (result, calls, _) = retried_under(replay, never, None);
    assert!(
        matches!(result, Err(LlmError::Backend { .. })),
        "{result:?}"
    );
    assert_eq!(calls, 5);
    let mut mixed: Vec<Step> = (0..6).map(|_| quick(status(503))).collect();
    mixed.extend(faults(4));
    mixed.push(quick(ok()));
    let (result, calls, _) = retried_under(replay, mixed, None);
    assert!(result.is_ok(), "{result:?}");
    assert_eq!(calls, 11);
}

/// Records each wait and then reports itself stopped, as the daemon's
/// sleeper does once shutdown wakes it.
#[derive(Default)]
struct Stopping {
    waits: AtomicUsize,
}

impl Sleeper for Stopping {
    fn sleep(&self, _: Duration) {
        self.waits.fetch_add(1, Ordering::SeqCst);
    }

    fn stopped(&self) -> bool {
        self.waits.load(Ordering::SeqCst) > 0
    }
}

#[test]
fn a_retry_its_sleeper_stops_gives_up_as_stopped() {
    // Shutdown wakes a retry in the middle of an outage. It makes no more
    // attempts and returns `Stopped`, not the outage's error, so nothing
    // counts it as a failure.
    let clock = Arc::new(SimulatedClock::new(start()));
    let steps = vec![
        quick(Err(LlmError::Status { status: 503 })),
        quick(Ok(json!({}))),
    ];
    let inner = Arc::new(Steps {
        clock: clock.clone(),
        steps: Mutex::new(steps.into()),
        calls: AtomicUsize::new(0),
    });
    let sleeper = Arc::new(Stopping::default());
    let retry = LlmRetry::new(inner.clone(), daemon_policy(), clock, sleeper.clone());
    let result = retry.complete(&request());
    assert!(matches!(result, Err(LlmError::Stopped)), "{result:?}");
    assert_eq!(LlmError::Stopped.retry(), Retry::Never);
    assert_eq!(inner.calls.load(Ordering::SeqCst), 1);
    assert_eq!(sleeper.waits.load(Ordering::SeqCst), 1);
}

/// What `status` says while a retry waits: who is retrying, since when and
/// how many attempts have failed, and how many lines need attention.
type Seen = (Vec<(String, Timestamp, u32)>, usize);

/// Reads the service's status each time the retry waits, then moves the
/// clock on by the wait.
struct Watching {
    service: Arc<Service>,
    clock: Arc<SimulatedClock>,
    seen: Mutex<Vec<Seen>>,
}

fn seen(service: &Service) -> Seen {
    let status = service.status().unwrap();
    let retrying = status
        .llm_retrying
        .iter()
        .map(|call| (call.caller.clone(), call.since, call.attempts))
        .collect();
    (retrying, status.attention.len())
}

impl Sleeper for Watching {
    fn sleep(&self, wait: Duration) {
        self.seen.lock().unwrap().push(seen(&self.service));
        self.clock.advance(SignedDuration::try_from(wait).unwrap());
    }
}

#[test]
fn a_call_retrying_through_an_outage_shows_in_status_until_it_ends() {
    // A long outage must not look like a quiet queue: `status` names the
    // caller, when its retrying began and the attempts so far, and once it
    // has gone on for minutes, needs attention.
    let dir = TestDir::new();
    let clock = Arc::new(SimulatedClock::new(start()));
    let store = Store::open(&dir.join("data"), OpenOptions::default(), clock.clone()).unwrap();
    let tuning = tuning(FakeReranker::MODEL_ID);
    let service = Service::with_models(clock.clone(), store, tuning, Models::fake()).unwrap();
    let service = Arc::new(service);
    let down = || quick(Err(LlmError::Status { status: 503 }));
    let inner = Arc::new(Steps {
        clock: clock.clone(),
        steps: Mutex::new(vec![down(), down(), quick(Ok(json!({})))].into()),
        calls: AtomicUsize::new(0),
    });
    let watching = Arc::new(Watching {
        service: Arc::clone(&service),
        clock: clock.clone(),
        seen: Mutex::new(Vec::new()),
    });
    let policy = RetryPolicy {
        first_wait: Duration::from_secs(20 * 60),
        max_wait: Duration::from_secs(20 * 60),
        ..daemon_policy()
    };
    let before = seen(&service);
    assert_eq!(before.0, vec![]);

    let retry = LlmRetry::new(inner, policy, clock.clone(), watching.clone())
        .reporting(service.llm_retries(), "main");
    retry.complete(&request()).unwrap();

    let seen_while = watching.seen.lock().unwrap().clone();
    let main = |attempts| vec![("main".to_string(), start(), attempts)];
    assert_eq!(seen_while.len(), 2);
    assert_eq!(seen_while[0], (main(1), before.1));
    // Ten minutes or more on, it needs attention.
    assert_eq!(seen_while[1].0, main(2));
    assert!(seen_while[1].1 > before.1, "{seen_while:?}");
    assert_eq!(seen(&service), before);
}

// The real models. Ignored: they run only when the models are present.

/// The real models on this machine, from the dir the daemon would resolve.
fn real_models() -> Models {
    let env = |name: &str| std::env::var_os(name).map(PathBuf::from);
    let dir = ModelDir::resolve(
        env("ASPHODEL_MODEL_DIR").as_deref(),
        env("XDG_CACHE_HOME").as_deref(),
        env("HOME").as_deref(),
    )
    .unwrap();
    let options = ModelOptions {
        threads: std::num::NonZeroUsize::new(1),
    };
    Models::load(&dir, &options)
        .unwrap_or_else(|error| panic!("loading from {}: {error}", dir.path().display()))
}

#[test]
#[ignore = "needs the real models: run `asphodel models fetch`, or set ASPHODEL_MODEL_DIR"]
fn real_models_embed_and_rerank() {
    let models = real_models();
    assert_eq!(models.embedder.model_id(), EMBEDDING_MODEL_ID);
    assert_eq!(models.reranker.model_id(), RERANKER_MODEL_ID);
    assert_eq!(models.embedder.dimensions(), EMBEDDING_DIMENSIONS);

    let texts = [
        "The cat sat on the mat.",
        "A cat is sitting on a mat.",
        "The quarterly tax filing deadline is in April.",
    ];
    let vectors = models.embedder.embed(&texts).unwrap();
    assert_eq!(vectors.len(), 3);
    let cosine = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    for vector in &vectors {
        assert_eq!(vector.len(), EMBEDDING_DIMENSIONS);
        let norm = cosine(vector, vector).sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "norm {norm}");
    }
    let near = cosine(&vectors[0], &vectors[1]);
    let far = cosine(&vectors[0], &vectors[2]);
    assert!(near > far, "near {near} far {far}");
    // bge-small scores even unrelated pairs at 0.6 or more, so the
    // check is on the gap, not an absolute.
    assert!(near - far > 0.1, "near {near} far {far}");
    // Deterministic across calls and batches.
    let again = models.embedder.embed(&[texts[0]]).unwrap();
    for (a, b) in vectors[0].iter().zip(&again[0]) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
    assert!(models.embedder.embed(&[]).unwrap().is_empty());

    // Independent reference: Python onnxruntime 1.27.1, Xenova L-6 int8,
    // revision a09144355adeed5f58c8ed011d209bf8ee5a1fec. Keep each batch
    // intact and in this order: dynamic int8 activation ranges depend on
    // the other pairs in the batch, so scoring pairs separately differs.
    let batches: [(&str, &[&str], &[f32]); 2] = [
        ("where did the cat sit", &texts, &[5.9225, 3.2397, -11.1688]),
        (
            "How many people live in Berlin?",
            &[
                "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
                "Berlin is well known for its museums.",
            ],
            &[8.3994, -4.6122],
        ),
    ];
    for (batch, (query, documents, expected)) in batches.iter().enumerate() {
        let scores = models.reranker.rerank(query, documents).unwrap();
        assert_eq!(
            scores.len(),
            expected.len(),
            "batch {batch}: one logit per document"
        );
        for (document, (score, reference)) in scores.iter().zip(*expected).enumerate() {
            assert!(
                (score - reference).abs() <= 1e-3,
                "batch {batch}, document {document}: logit {score}, reference {reference}"
            );
        }
    }
    assert!(
        models
            .reranker
            .rerank(batches[0].0, &[])
            .unwrap()
            .is_empty()
    );
}

/// The real reranker scores a pair of at most 512 tokens, truncating the
/// longer side first and from its end, and its tokenizer makes each CJK
/// character a token. So a query whose context runs past the budget loses
/// whatever comes after it, and what comes first survives. Prefetch's
/// conversation query has to fit that; `TruncatingReranker` in
/// `tests/retrieval.rs` models it.
#[test]
#[ignore = "needs the real models: run `asphodel models fetch`, or set ASPHODEL_MODEL_DIR"]
fn real_reranker_truncates_a_long_query_from_its_end() {
    let models = real_models();
    let documents = ["Tim takes a pottery class on Tuesdays."];
    let message = "when is my pottery class";
    let context = "東京".repeat(300);
    let score = |query: &str| models.reranker.rerank(query, &documents).unwrap()[0];

    assert_eq!(
        score(&format!("{context}\n{message}")),
        score(&context),
        "a message after 600 CJK tokens never reaches the model"
    );
    let first = score(&format!("{message}\n{context}"));
    assert!(
        first > score(&context) + 1.0,
        "a message before the context does: {first}"
    );
}
