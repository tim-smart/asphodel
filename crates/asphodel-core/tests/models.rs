//! The local models and the LLM client, checked against "Models: local
//! embeddings, reranker and the OpenAI-compatible LLM client" (TIM-105) and
//! the decisions it rests on: "Rust storage and search stack" (TIM-89),
//! "Retrieval and ranking" (TIM-93, decision 3), "API surface and Hermes
//! transport" (TIM-94, decision 4, as amended by TIM-99), "Replay harness"
//! (TIM-96, decision 4), "Configuration surface" (TIM-98) and ADR 0009.
//!
//! Nothing here touches the network. The OpenAI-compatible client is tested
//! against [`StubServer`], a loopback HTTP/1.1 server in this file, and the
//! model loader against files written into a temp dir. The one exception is
//! [`real_models_embed_and_rerank`], which is ignored and runs only when the
//! real models are in `ASPHODEL_MODEL_DIR` (or the XDG cache).

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use asphodel_core::clock::{Clock, SimulatedClock};
use asphodel_core::config::{ConfigError, Deployment, Secret, Tuning};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::vector::EMBEDDING_DIMENSIONS;
use asphodel_core::store::{OpenOptions, Store};
use jiff::Timestamp;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use asphodel_core::models::*;
use asphodel_core::{OpenError, Service};

// Fixtures.

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

fn open_store(dir: &TestDir) -> Store {
    Store::open(&dir.join("data"), OpenOptions::default(), clock()).unwrap()
}

/// `[llm]` plus a floor for each fake model, so a service opens on the
/// fakes.
fn tuning_for_fakes() -> Tuning {
    Tuning::from_toml(&format!(
        "[injection.reranker_floors]\n\"{}\" = 0.0\n\
         [reconcile.embedding_floors]\n\"{}\" = 0.5\n",
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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
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
    let mut specs = Vec::new();
    for (id, dir) in [
        (EMBEDDING_MODEL_ID, "bge-small-en-v1.5-int8"),
        (RERANKER_MODEL_ID, "jina-reranker-v1-turbo-en-int8"),
    ] {
        let files = MODEL_FILES
            .iter()
            .map(|name| {
                let url = format!("https://models.example/{dir}/{name}");
                let content = format!("{id} {name} bytes").into_bytes();
                let sha256 = sha256_hex(&content);
                bytes.insert(url.clone(), content);
                ModelFile {
                    name: (*name).to_string(),
                    url,
                    sha256,
                }
            })
            .collect();
        specs.push(ModelSpec {
            id: id.to_string(),
            dir: dir.to_string(),
            files,
        });
    }
    Canned { specs, bytes }
}

/// Serves canned bytes by URL and counts calls.
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
            calls: AtomicUsize::new(0),
            failing: Vec::new(),
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

    fn json(&self) -> Value {
        serde_json::from_str(&self.body).expect("a JSON body")
    }
}

#[derive(Debug, Clone)]
struct StubResponse {
    status: u16,
    body: String,
    /// Held before answering, for the timeout test.
    delay: Duration,
}

impl StubResponse {
    fn json(value: Value) -> Self {
        Self {
            status: 200,
            body: value.to_string(),
            delay: Duration::ZERO,
        }
    }

    fn status(status: u16) -> Self {
        Self {
            status,
            body: "{}".into(),
            delay: Duration::ZERO,
        }
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

    fn requests(&self) -> Vec<StubRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn only_request(&self) -> StubRequest {
        let requests = self.requests();
        assert_eq!(requests.len(), 1, "{requests:?}");
        requests.into_iter().next().unwrap()
    }

    fn settings(&self, api_key: Option<&str>) -> LlmSettings {
        LlmSettings {
            auth: LlmAuth::ApiKey,
            endpoint: format!("{}/v1", self.url),
            model: "some-model:q4_K_M".into(),
            api_key: api_key.map(Secret::new),
            timeout: Duration::from_secs(5),
        }
    }
}

fn serve_one(mut stream: TcpStream, response: StubResponse, log: &Mutex<Vec<StubRequest>>) {
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
    log.lock().unwrap().push(StubRequest {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    });
    std::thread::sleep(response.delay);
    let reason = match response.status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        408 => "Request Timeout",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Status",
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        response.status,
        response.body.len(),
        response.body
    );
    let _ = stream.flush();
}

// What runs now: the floors and the LLM settings that already exist.

#[test]
fn the_default_tuning_has_no_floor_for_either_model() {
    // ADR 0009: floors come only from the precision curve on Tim's labels,
    // so there's no code default, and a daemon on the default tuning stops.
    let error = Tuning::default()
        .check_floors(EMBEDDING_MODEL_ID, RERANKER_MODEL_ID)
        .unwrap_err();
    let ConfigError::Invalid(errors) = error else {
        panic!("{error}");
    };
    let keys: Vec<_> = errors.iter().map(|e| e.key.as_str()).collect();
    assert_eq!(
        keys,
        [
            "reconcile.embedding_floors.\"bge-small-en-v1.5:int8\"",
            "injection.reranker_floors.\"jina-reranker-v1-turbo-en:int8\"",
        ]
    );
}

#[test]
fn floors_for_the_exact_model_strings_satisfy_the_check() {
    let tuning = Tuning::from_toml(
        "[injection.reranker_floors]\n\"jina-reranker-v1-turbo-en:int8\" = -1.5\n\
         [reconcile.embedding_floors]\n\"bge-small-en-v1.5:int8\" = 0.82\n",
    )
    .unwrap();
    tuning
        .check_floors(EMBEDDING_MODEL_ID, RERANKER_MODEL_ID)
        .unwrap();
    // A floor under a different quantisation of the same model doesn't
    // count: int8 and fp32 give different scores (TIM-98).
    let fp32 = Tuning::from_toml(
        "[injection.reranker_floors]\n\"jina-reranker-v1-turbo-en:fp32\" = -1.5\n\
         [reconcile.embedding_floors]\n\"bge-small-en-v1.5:fp32\" = 0.82\n",
    )
    .unwrap();
    assert!(
        fp32.check_floors(EMBEDDING_MODEL_ID, RERANKER_MODEL_ID)
            .is_err()
    );
}

#[test]
fn the_embedding_width_matches_the_vector_table() {
    // bge-small-en-v1.5 is 384 wide (TIM-89), and so is the vec0 table.
    assert_eq!(EMBEDDING_DIMENSIONS, 384);
}

#[test]
fn the_llm_endpoint_and_model_come_from_the_tuning_file_and_the_key_from_the_environment() {
    // ADR 0009: the API key is a secret, environment only, and never in the
    // tuning file.
    let tuning = Tuning::from_toml(
        "[llm]\nmodel = \"some-model:q4_K_M\"\nendpoint = \"http://llm.internal:8080/v1\"\n",
    )
    .unwrap();
    assert_eq!(tuning.llm.model.as_deref(), Some("some-model:q4_K_M"));
    assert_eq!(
        tuning.llm.endpoint.as_deref(),
        Some("http://llm.internal:8080/v1")
    );
    assert!(Tuning::from_toml("[llm]\napi_key = \"sk-1\"\n").is_err());

    let deployment = deployment(Some("sk-live-41b2e8-secret"));
    let key = deployment.llm_api_key.as_ref().unwrap();
    assert_eq!(key.expose(), "sk-live-41b2e8-secret");
    assert_eq!(format!("{key:?}"), "[redacted]");
    assert_eq!(
        serde_json::to_value(&deployment).unwrap()["llm_api_key"],
        "[redacted]"
    );
}

#[test]
fn the_stub_server_answers_a_request() {
    // The test harness itself, so a failure in the client tests points at
    // the client and not at this file's HTTP server.
    let server = StubServer::start(StubResponse::completion("{\"ok\":true}"));
    let mut stream = TcpStream::connect(server.url.trim_start_matches("http://")).unwrap();
    let body = "{\"a\":1}";
    write!(
        stream,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer k\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut reply = String::new();
    stream.read_to_string(&mut reply).unwrap();
    assert!(reply.starts_with("HTTP/1.1 200 OK\r\n"), "{reply}");
    assert!(reply.contains("\"choices\""), "{reply}");

    let request = server.only_request();
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(request.header("authorization"), Some("Bearer k"));
    assert_eq!(request.json(), json!({"a": 1}));
}

// The model dir (TIM-94, decision 4; TIM-98 deployment).

#[test]
fn the_override_wins_over_the_xdg_cache() {
    let dir = ModelDir::resolve(
        Some(Path::new("/srv/models")),
        Some(Path::new("/home/tim/.cache")),
        Some(Path::new("/home/tim")),
    )
    .unwrap();
    assert_eq!(dir.path(), Path::new("/srv/models"));
}

#[test]
fn the_default_is_the_xdg_cache() {
    let dir = ModelDir::resolve(
        None,
        Some(Path::new("/var/cache/tim")),
        Some(Path::new("/home/tim")),
    )
    .unwrap();
    assert_eq!(dir.path(), Path::new("/var/cache/tim/asphodel/models"));
}

#[test]
fn without_xdg_cache_home_the_default_is_under_home() {
    let dir = ModelDir::resolve(None, None, Some(Path::new("/home/tim"))).unwrap();
    assert_eq!(dir.path(), Path::new("/home/tim/.cache/asphodel/models"));

    // The XDG spec: a relative XDG_CACHE_HOME is invalid and ignored.
    let dir =
        ModelDir::resolve(None, Some(Path::new("cache")), Some(Path::new("/home/tim"))).unwrap();
    assert_eq!(dir.path(), Path::new("/home/tim/.cache/asphodel/models"));
}

#[test]
fn no_dir_at_all_is_an_error() {
    let error = ModelDir::resolve(None, None, None).unwrap_err();
    assert!(matches!(error, ModelError::NoModelDir), "{error:?}");
    assert!(error.to_string().contains("ASPHODEL_MODEL_DIR"), "{error}");
}

#[test]
fn files_live_under_the_models_dir_name() {
    let dir = ModelDir::at("/srv/models");
    let spec = &canned().specs[0];
    assert_eq!(
        dir.file(spec, "model.onnx"),
        Path::new("/srv/models/bge-small-en-v1.5-int8/model.onnx")
    );
}

// The manifest.

#[test]
fn the_manifest_names_the_two_models_and_their_five_files() {
    let manifest = manifest();
    let ids: Vec<_> = manifest.iter().map(|spec| spec.id.as_str()).collect();
    assert_eq!(ids, [EMBEDDING_MODEL_ID, RERANKER_MODEL_ID]);
    for spec in &manifest {
        let names: Vec<_> = spec.files.iter().map(|file| file.name.as_str()).collect();
        assert_eq!(names, MODEL_FILES, "{}", spec.id);
        assert!(!spec.dir.contains(':'), "{}: {}", spec.id, spec.dir);
        assert!(!spec.dir.contains('/'), "{}: {}", spec.id, spec.dir);
        for file in &spec.files {
            assert!(
                file.url.starts_with("https://huggingface.co/"),
                "{}: {}",
                spec.id,
                file.url
            );
            assert_eq!(file.sha256.len(), 64, "{}: {}", spec.id, file.name);
            assert!(
                file.sha256
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{}: {}",
                spec.id,
                file.sha256
            );
        }
    }
    // The two models don't share a directory.
    assert_ne!(manifest[0].dir, manifest[1].dir);
}

// `asphodel models fetch` (TIM-94, decision 4).

#[test]
fn fetch_fills_an_empty_dir_and_checks_every_file() {
    let dir = TestDir::new();
    let models = dir.models();
    let canned = canned();
    let fetcher = MapFetcher::new(canned.bytes.clone());

    let report = fetch_models(&models, &canned.specs, &fetcher).unwrap();

    assert_eq!(fetcher.calls(), 10);
    assert_eq!(report.fetched.len(), 10);
    assert!(report.skipped.is_empty());
    for spec in &canned.specs {
        for file in &spec.files {
            let path = models.file(spec, &file.name);
            assert!(report.fetched.contains(&path), "{}", path.display());
            let written = std::fs::read(&path).unwrap();
            assert_eq!(sha256_hex(&written), file.sha256, "{}", path.display());
        }
    }
    // Nothing else was left behind: no temp files.
    for spec in &canned.specs {
        let entries: Vec<_> = std::fs::read_dir(models.path().join(&spec.dir))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(entries.len(), MODEL_FILES.len(), "{entries:?}");
    }
}

#[test]
fn fetch_skips_files_that_are_already_right() {
    let dir = TestDir::new();
    let models = dir.models();
    let canned = canned();
    let fetcher = MapFetcher::new(canned.bytes.clone());
    fetch_models(&models, &canned.specs, &fetcher).unwrap();
    assert_eq!(fetcher.calls(), 10);

    let report = fetch_models(&models, &canned.specs, &fetcher).unwrap();
    assert_eq!(fetcher.calls(), 10, "a second run fetched again");
    assert!(report.fetched.is_empty());
    assert_eq!(report.skipped.len(), 10);
}

#[test]
fn fetch_replaces_a_file_whose_bytes_are_wrong() {
    let dir = TestDir::new();
    let models = dir.models();
    let canned = canned();
    let fetcher = MapFetcher::new(canned.bytes.clone());
    fetch_models(&models, &canned.specs, &fetcher).unwrap();

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
fn fetch_refuses_bytes_that_do_not_match_the_manifest() {
    let dir = TestDir::new();
    let models = dir.models();
    let canned = canned();
    let mut bytes = canned.bytes.clone();
    let tampered = canned.specs[0].files[0].url.clone();
    bytes.insert(tampered.clone(), b"not the model".to_vec());
    let fetcher = MapFetcher::new(bytes);

    let error = fetch_models(&models, &canned.specs, &fetcher).unwrap_err();
    assert!(
        matches!(&error, FetchFailure::Checksum { url } if *url == tampered),
        "{error:?}"
    );
    // The bad bytes were never written, and nothing after them was fetched.
    let path = models.file(&canned.specs[0], "model.onnx");
    assert!(!path.exists(), "{} was written", path.display());
    assert_eq!(fetcher.calls(), 1);
    let entries = std::fs::read_dir(models.path().join(&canned.specs[0].dir))
        .map(|entries| entries.count())
        .unwrap_or(0);
    assert_eq!(entries, 0, "a temp file was left behind");
}

#[test]
fn fetch_stops_at_the_first_failure_and_keeps_what_it_wrote() {
    let dir = TestDir::new();
    let models = dir.models();
    let canned = canned();
    let mut fetcher = MapFetcher::new(canned.bytes.clone());
    // The third file of the first model fails.
    let failing = canned.specs[0].files[2].url.clone();
    fetcher.failing.push(failing.clone());

    let error = fetch_models(&models, &canned.specs, &fetcher).unwrap_err();
    assert!(
        matches!(&error, FetchFailure::Fetch { url, error: FetchError::Status(503) } if *url == failing),
        "{error:?}"
    );
    assert_eq!(fetcher.calls(), 3);
    assert!(models.file(&canned.specs[0], "model.onnx").exists());
    assert!(models.file(&canned.specs[0], "tokenizer.json").exists());
    assert!(!models.file(&canned.specs[0], "config.json").exists());

    // Once the source recovers, the next run picks up where it stopped.
    fetcher.failing.clear();
    let report = fetch_models(&models, &canned.specs, &fetcher).unwrap();
    assert_eq!(report.skipped.len(), 2);
    assert_eq!(report.fetched.len(), 8);
}

#[test]
fn fetch_never_writes_through_a_planted_temp_symlink() {
    // Today the temp file is `<file>.part-<pid>`. The fix moves to fresh
    // names created exclusively, so a symlink planted here must simply be
    // left alone.
    let dir = TestDir::new();
    let models = dir.models();
    let canned = canned();
    let spec = &canned.specs[0];
    let target = models.file(spec, &spec.files[0].name);
    std::fs::create_dir_all(target.parent().unwrap()).unwrap();
    let victim = dir.join("victim.txt");
    std::fs::write(&victim, "irreplaceable contents").unwrap();
    let mut planted = target.as_os_str().to_owned();
    planted.push(format!(".part-{}", std::process::id()));
    let planted = PathBuf::from(planted);
    std::os::unix::fs::symlink(&victim, &planted).unwrap();

    fetch_models(
        &models,
        &canned.specs,
        &MapFetcher::new(canned.bytes.clone()),
    )
    .unwrap();

    assert_eq!(
        std::fs::read_to_string(&victim).unwrap(),
        "irreplaceable contents",
        "the model bytes were written through the symlink"
    );
    let installed = std::fs::symlink_metadata(&target).unwrap();
    assert!(
        installed.file_type().is_file(),
        "{} is not a regular file: {:?}",
        target.display(),
        installed.file_type()
    );
    assert_eq!(
        sha256_hex(&std::fs::read(&target).unwrap()),
        spec.files[0].sha256
    );
    assert!(
        std::fs::symlink_metadata(&planted)
            .unwrap()
            .file_type()
            .is_symlink(),
        "the planted symlink was removed or replaced"
    );
}

// Loading: never a download, and a missing file fails fast.

#[test]
fn an_empty_model_dir_fails_fast_naming_the_first_missing_file() {
    let dir = TestDir::new();
    let models = dir.models();
    let spec = &manifest()[0];

    let error = Models::load(&models, &ModelOptions::default()).unwrap_err();
    let expected = models.file(spec, "model.onnx");
    assert!(
        matches!(&error, ModelError::MissingFile { model, path } if model == EMBEDDING_MODEL_ID && *path == expected),
        "{error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains(&expected.display().to_string()),
        "{message}"
    );
    assert!(message.contains("asphodel models fetch"), "{message}");
    // Fail fast means fail without side effects: nothing was created.
    assert!(!models.path().exists(), "the loader created the model dir");
}

#[test]
fn a_missing_reranker_file_is_named_even_when_the_embedder_is_complete() {
    // A full embedding model dir and a reranker dir short of one tokenizer
    // file. The checksum of the embedding files can't be met here, so the
    // test uses the real manifest only for the paths and expects the loader
    // to report the missing file before any checksum: presence is checked
    // for every file first, so an operator sees the whole problem's shape.
    let dir = TestDir::new();
    let models = dir.models();
    let manifest = manifest();
    for file in &manifest[0].files {
        let path = models.file(&manifest[0], &file.name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"placeholder").unwrap();
    }
    for file in &manifest[1].files {
        if file.name == "special_tokens_map.json" {
            continue;
        }
        let path = models.file(&manifest[1], &file.name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"placeholder").unwrap();
    }

    let error = Models::load(&models, &ModelOptions::default()).unwrap_err();
    let expected = models.file(&manifest[1], "special_tokens_map.json");
    assert!(
        matches!(&error, ModelError::MissingFile { model, path } if model == RERANKER_MODEL_ID && *path == expected),
        "{error:?}"
    );
}

#[test]
fn a_corrupt_file_fails_before_onnx_runtime_is_touched() {
    // Every file present, none with the manifest's bytes. The loader checks
    // checksums before building a session, so the error is ours and names
    // the file, not an ONNX Runtime message about a bad protobuf.
    let dir = TestDir::new();
    let models = dir.models();
    let manifest = manifest();
    for spec in &manifest {
        for file in &spec.files {
            let path = models.file(spec, &file.name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, format!("garbage for {}", file.name)).unwrap();
        }
    }

    let error = Models::load(&models, &ModelOptions::default()).unwrap_err();
    let expected = models.file(&manifest[0], "model.onnx");
    assert!(
        matches!(&error, ModelError::Checksum { model, path } if model == EMBEDDING_MODEL_ID && *path == expected),
        "{error:?}"
    );
    assert!(
        error.to_string().contains("asphodel models fetch"),
        "{error}"
    );
}

// The fakes (TIM-96, decision 2).

#[test]
fn the_fakes_have_their_own_ids() {
    let models = Models::fake();
    assert_eq!(models.embedder.model_id(), FakeEmbedder::MODEL_ID);
    assert_eq!(models.reranker.model_id(), FakeReranker::MODEL_ID);
    assert_ne!(models.embedder.model_id(), EMBEDDING_MODEL_ID);
    assert_ne!(models.reranker.model_id(), RERANKER_MODEL_ID);
    let ids = models.ids();
    assert_eq!(ids.embedding, FakeEmbedder::MODEL_ID);
    assert_eq!(ids.reranker, FakeReranker::MODEL_ID);
}

#[test]
fn the_fake_embedder_is_deterministic_unit_length_and_384_wide() {
    let embedder = FakeEmbedder;
    assert_eq!(embedder.dimensions(), EMBEDDING_DIMENSIONS);

    let texts = [
        "Tim moved to Wellington in March.",
        "Maya's birthday is on 14 June.",
        "",
    ];
    let first = embedder.embed(&texts).unwrap();
    assert_eq!(first.len(), 3);
    for vector in &first {
        assert_eq!(vector.len(), EMBEDDING_DIMENSIONS);
        assert!((norm(vector) - 1.0).abs() < 1e-5, "norm {}", norm(vector));
    }
    // Twice, and from another instance, gives the same bytes.
    let again = FakeEmbedder.embed(&texts).unwrap();
    assert_eq!(first, again);
    // One at a time matches the batch: no cross-text state.
    let alone = embedder.embed(&[texts[1]]).unwrap();
    assert_eq!(alone[0], first[1]);
    assert!(embedder.embed(&[]).unwrap().is_empty());
}

#[test]
fn the_fake_embedder_puts_texts_that_share_words_closer() {
    let embedder = FakeEmbedder;
    let vectors = embedder
        .embed(&[
            "red car parked outside",
            "red car parked inside",
            "Outside parked car red",
            "quarterly tax filing deadline",
        ])
        .unwrap();
    let same = cosine(&vectors[0], &vectors[0]);
    let near = cosine(&vectors[0], &vectors[1]);
    let far = cosine(&vectors[0], &vectors[3]);
    assert!((same - 1.0).abs() < 1e-5, "{same}");
    assert!(near > far, "near {near} far {far}");
    assert!(near > 0.5, "three of four words shared: {near}");
    // A bag of words: order and case don't matter.
    assert!(
        (cosine(&vectors[0], &vectors[2]) - 1.0).abs() < 1e-5,
        "{}",
        cosine(&vectors[0], &vectors[2])
    );
}

#[test]
fn the_fake_reranker_scores_by_query_words_in_the_document() {
    let reranker = FakeReranker;
    let query = "when is Maya's birthday";
    let documents = [
        "Maya's birthday is on 14 June.",
        "Tim moved to Wellington in March.",
        "Maya is Tim's sister.",
    ];
    let scores = reranker.rerank(query, &documents).unwrap();
    assert_eq!(scores.len(), 3, "one logit per document, in order");
    assert!(scores[0] > scores[2], "{scores:?}");
    assert!(scores[2] > scores[1], "{scores:?}");
    assert!(
        scores[1] < 0.0,
        "nothing shared scores below zero: {scores:?}"
    );
    assert_eq!(scores, FakeReranker.rerank(query, &documents).unwrap());
    assert!(reranker.rerank(query, &[]).unwrap().is_empty());
}

#[test]
fn the_fakes_are_shareable_trait_objects() {
    fn takes(embedder: Arc<dyn Embedder>, reranker: Arc<dyn Reranker>) -> (String, String) {
        (
            embedder.model_id().to_string(),
            reranker.model_id().to_string(),
        )
    }
    let models = Models::fake();
    let ids = takes(Arc::clone(&models.embedder), Arc::clone(&models.reranker));
    assert_eq!(
        ids,
        (FakeEmbedder::MODEL_ID.into(), FakeReranker::MODEL_ID.into())
    );
    // Used from another thread, as the extraction worker will.
    let handle = std::thread::spawn(move || models.embedder.embed(&["from a thread"]).unwrap());
    assert_eq!(handle.join().unwrap()[0].len(), EMBEDDING_DIMENSIONS);
}

// The service: recorded model ids and the floor check at startup.

#[test]
fn a_missing_floor_for_a_loaded_model_stops_the_service_opening() {
    // ADR 0009: a missing floor for a configured model stops the daemon.
    // The check lives in the service so replay gets it too.
    let dir = TestDir::new();
    let error = Service::with_models(clock(), open_store(&dir), Tuning::default(), Models::fake())
        .expect_err("opened without floors");
    let OpenError::Config(ConfigError::Invalid(errors)) = error else {
        panic!("{error}");
    };
    let keys: Vec<_> = errors.iter().map(|e| e.key.clone()).collect();
    assert_eq!(
        keys,
        [
            format!("reconcile.embedding_floors.\"{}\"", FakeEmbedder::MODEL_ID),
            format!("injection.reranker_floors.\"{}\"", FakeReranker::MODEL_ID),
        ]
    );
}

#[test]
fn floors_for_the_loaded_models_let_the_service_open() {
    let dir = TestDir::new();
    let service = Service::with_models(
        clock(),
        open_store(&dir),
        tuning_for_fakes(),
        Models::fake(),
    )
    .unwrap();
    assert_eq!(
        service.models().unwrap().ids().embedding,
        FakeEmbedder::MODEL_ID
    );
    assert!(service.health().ready);
}

#[test]
fn a_new_bank_records_the_loaded_models() {
    // TIM-94, decision 4: a bank records its embedding and reranker model
    // ids. They come from what's loaded, not from the caller.
    let dir = TestDir::new();
    let service = Service::with_models(
        clock(),
        open_store(&dir),
        tuning_for_fakes(),
        Models::fake(),
    )
    .unwrap();
    let bank = service
        .ensure_bank_with_models("tim", &BankIdentity::default())
        .unwrap();
    assert!(bank.created);
    assert_eq!(bank.embedding_model, FakeEmbedder::MODEL_ID);
    assert_eq!(bank.reranker_model, FakeReranker::MODEL_ID);

    // A merge leaves the recorded ids alone: a change goes through
    // `asphodel reembed` (TIM-99), never through bank config.
    let again = service
        .ensure_bank_with_models(
            "tim",
            &BankIdentity {
                owner_name: Some("Tim".into()),
                ..BankIdentity::default()
            },
        )
        .unwrap();
    assert!(!again.created);
    assert_eq!(again.embedding_model, FakeEmbedder::MODEL_ID);
    assert_eq!(again.reranker_model, FakeReranker::MODEL_ID);
}

// LLM settings (ADR 0009).

#[test]
fn llm_settings_come_from_the_tuning_file_and_the_environment() {
    let tuning = Tuning::from_toml(
        "[llm]\nmodel = \"some-model:q4_K_M\"\nendpoint = \"http://llm.internal:8080/v1\"\n",
    )
    .unwrap();
    let settings = LlmSettings::from_config(&tuning, &deployment(Some("sk-live-41b2e8-secret")))
        .unwrap()
        .expect("configured");
    assert_eq!(settings.endpoint, "http://llm.internal:8080/v1");
    assert_eq!(settings.model, "some-model:q4_K_M");
    assert_eq!(
        settings.api_key.as_ref().map(Secret::expose),
        Some("sk-live-41b2e8-secret")
    );
    assert_eq!(settings.timeout, LlmSettings::DEFAULT_TIMEOUT);
    // The key never shows in Debug output.
    assert!(!format!("{settings:?}").contains("41b2e8"), "{settings:?}");

    // A local endpoint needs no key.
    let local = LlmSettings::from_config(&tuning, &deployment(None))
        .unwrap()
        .expect("configured");
    assert!(local.api_key.is_none());
}

#[test]
fn llm_settings_are_absent_when_nothing_is_set_and_an_error_when_half_set() {
    assert!(
        LlmSettings::from_config(&Tuning::default(), &deployment(None))
            .unwrap()
            .is_none()
    );

    let only_model = Tuning::from_toml("[llm]\nmodel = \"some-model\"\n").unwrap();
    let error = LlmSettings::from_config(&only_model, &deployment(None)).unwrap_err();
    assert!(
        matches!(
            error,
            LlmError::NotConfigured {
                missing: "llm.endpoint"
            }
        ),
        "{error:?}"
    );

    let only_endpoint =
        Tuning::from_toml("[llm]\nendpoint = \"http://llm.internal:8080/v1\"\n").unwrap();
    let error = LlmSettings::from_config(&only_endpoint, &deployment(None)).unwrap_err();
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

// The OpenAI-compatible client, against the loopback stub.

#[test]
fn the_client_posts_a_structured_chat_completion() {
    let server = StubServer::start(StubResponse::completion(
        "{\"claims\":[\"Tim moved to Wellington in March 2026.\"]}",
    ));
    let client = OpenAiCompatible::new(server.settings(Some("sk-live-41b2e8-secret")));
    assert_eq!(client.model(), "some-model:q4_K_M");

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

    let sent = server.only_request();
    assert_eq!(sent.method, "POST");
    assert_eq!(sent.path, "/v1/chat/completions");
    assert_eq!(
        sent.header("authorization"),
        Some("Bearer sk-live-41b2e8-secret")
    );
    assert!(
        sent.header("content-type")
            .is_some_and(|value| value.starts_with("application/json")),
        "{:?}",
        sent.headers
    );

    let body = sent.json();
    assert_eq!(body["model"], "some-model:q4_K_M");
    assert_eq!(
        body["messages"],
        json!([
            {"role": "system", "content": "You extract memories."},
            {"role": "user", "content": "Tim said: I moved to Wellington in March."}
        ])
    );
    assert_eq!(body["temperature"], 0);
    assert_eq!(body["max_tokens"], 512);
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["response_format"]["json_schema"]["name"], "claims");
    assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    assert_eq!(
        body["response_format"]["json_schema"]["schema"],
        request().schema
    );
    assert!(
        body.get("stream")
            .is_none_or(|stream| stream == &json!(false)),
        "{body}"
    );
}

#[test]
fn without_a_key_there_is_no_authorization_header_and_no_max_tokens_when_unset() {
    let server = StubServer::start(StubResponse::completion("{}"));
    let client = OpenAiCompatible::new(server.settings(None));
    let mut request = request();
    request.max_tokens = None;
    client.complete(&request).unwrap();

    let sent = server.only_request();
    assert_eq!(sent.header("authorization"), None, "{:?}", sent.headers);
    assert!(sent.json().get("max_tokens").is_none(), "{}", sent.body);
}

#[test]
fn a_trailing_slash_on_the_endpoint_does_not_double_the_path() {
    let server = StubServer::start(StubResponse::completion("{}"));
    let mut settings = server.settings(None);
    settings.endpoint.push('/');
    OpenAiCompatible::new(settings)
        .complete(&request())
        .unwrap();
    assert_eq!(server.only_request().path, "/v1/chat/completions");
}

#[test]
fn fenced_json_in_the_content_is_unwrapped() {
    // Local models often fence their output even under json_schema.
    let server = StubServer::start(StubResponse::completion("```json\n{\"claims\": []}\n```"));
    let response = OpenAiCompatible::new(server.settings(None))
        .complete(&request())
        .unwrap();
    assert_eq!(response.json, json!({"claims": []}));
}

#[test]
fn content_that_is_not_json_is_an_error_that_carries_only_its_size() {
    let content = "Sure! Here are Tim's claims: he moved to Wellington.";
    let server = StubServer::start(StubResponse::completion(content));
    let error = OpenAiCompatible::new(server.settings(None))
        .complete(&request())
        .unwrap_err();
    assert!(
        matches!(error, LlmError::NotJson { bytes } if bytes == content.len()),
        "{error:?}"
    );
    assert!(!error.is_retryable());
    // TIM-96, decision 8: no content in logs. The reply is Tim's data.
    let shown = format!("{error} {error:?}");
    assert!(!shown.contains("Wellington"), "{shown}");
}

#[test]
fn no_choices_and_a_refusal_are_errors() {
    let server = StubServer::start(StubResponse::json(json!({
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0}
    })));
    let error = OpenAiCompatible::new(server.settings(None))
        .complete(&request())
        .unwrap_err();
    assert!(matches!(error, LlmError::NoContent), "{error:?}");
    assert!(!error.is_retryable());

    let server = StubServer::start(StubResponse::json(json!({
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": null, "refusal": "I can't help with that."},
            "finish_reason": "stop"
        }]
    })));
    let error = OpenAiCompatible::new(server.settings(None))
        .complete(&request())
        .unwrap_err();
    assert!(matches!(error, LlmError::Refused), "{error:?}");
    assert!(!error.is_retryable());
}

#[test]
fn missing_usage_is_none_not_zero() {
    let server = StubServer::start(StubResponse::json(json!({
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "{\"claims\":[]}"},
            "finish_reason": "stop"
        }]
    })));
    let response = OpenAiCompatible::new(server.settings(None))
        .complete(&request())
        .unwrap();
    assert_eq!(response.usage, None);
}

#[test]
fn http_statuses_map_to_retryable_or_not() {
    for (status, retryable) in [
        (400, false),
        (401, false),
        (404, false),
        (408, true),
        (429, true),
        (500, true),
        (502, true),
        (503, true),
    ] {
        let server = StubServer::start(StubResponse::status(status));
        let error = OpenAiCompatible::new(server.settings(None))
            .complete(&request())
            .unwrap_err();
        assert!(
            matches!(error, LlmError::Status { status: got } if got == status),
            "{status}: {error:?}"
        );
        assert_eq!(error.is_retryable(), retryable, "{status}");
    }
}

#[test]
fn a_slow_endpoint_times_out() {
    let mut response = StubResponse::completion("{}");
    response.delay = Duration::from_secs(3);
    let server = StubServer::start(response);
    let mut settings = server.settings(None);
    settings.timeout = Duration::from_millis(200);
    let error = OpenAiCompatible::new(settings)
        .complete(&request())
        .unwrap_err();
    assert!(matches!(error, LlmError::Timeout), "{error:?}");
    assert!(error.is_retryable());
}

#[test]
fn a_dead_endpoint_is_a_retryable_transport_error() {
    // Bind and drop, so the port is closed.
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let settings = LlmSettings {
        auth: LlmAuth::ApiKey,
        endpoint: format!("http://127.0.0.1:{port}/v1"),
        model: "some-model".into(),
        api_key: None,
        timeout: Duration::from_secs(2),
    };
    let error = OpenAiCompatible::new(settings)
        .complete(&request())
        .unwrap_err();
    assert!(matches!(error, LlmError::Transport { .. }), "{error:?}");
    assert!(error.is_retryable());
}

#[test]
fn the_real_client_is_a_trait_object() {
    let server = StubServer::start(StubResponse::completion("{\"ok\":true}"));
    let client: Arc<dyn LlmClient> = Arc::new(OpenAiCompatible::new(server.settings(None)));
    let handle = {
        let client = Arc::clone(&client);
        std::thread::spawn(move || client.complete(&request()).unwrap())
    };
    assert_eq!(handle.join().unwrap().json, json!({"ok": true}));
}

// The fake LLM.

#[test]
fn the_fake_llm_replies_in_order_and_records_requests() {
    let fake = FakeLlm::scripted(
        "fake-llm",
        vec![json!({"claims": ["one"]}), json!({"claims": ["two"]})],
    );
    assert_eq!(fake.model(), "fake-llm");

    let first = request();
    let mut second = request();
    second.user = "Tim said: and Maya's birthday is 14 June.".into();

    assert_eq!(
        fake.complete(&first).unwrap().json,
        json!({"claims": ["one"]})
    );
    assert_eq!(
        fake.complete(&second).unwrap().json,
        json!({"claims": ["two"]})
    );
    let error = fake.complete(&first).unwrap_err();
    assert!(matches!(error, LlmError::NoContent), "{error:?}");

    assert_eq!(fake.requests(), vec![first, second.clone(), request()]);
    assert_eq!(fake.requests()[1].user, second.user);
}

#[test]
fn the_fake_llm_can_fail_every_call() {
    let fake = FakeLlm::failing("fake-llm", || LlmError::Status { status: 503 });
    let error = fake.complete(&request()).unwrap_err();
    assert!(
        matches!(error, LlmError::Status { status: 503 }),
        "{error:?}"
    );
    assert!(error.is_retryable());
    assert_eq!(fake.requests().len(), 1);
}

#[test]
fn the_fake_llm_is_a_shareable_trait_object() {
    let client: Arc<dyn LlmClient> = Arc::new(FakeLlm::scripted("fake-llm", vec![json!({})]));
    let handle = {
        let client = Arc::clone(&client);
        std::thread::spawn(move || client.complete(&request()).unwrap())
    };
    assert_eq!(handle.join().unwrap().json, json!({}));
}

#[test]
fn requests_and_responses_round_trip_through_json_for_the_cassette() {
    // TIM-96, decision 4: replay records calls. The types serialise so the
    // cassette can store them and key on the template and model.
    let request = request();
    let text = serde_json::to_string(&request).unwrap();
    assert_eq!(serde_json::from_str::<LlmRequest>(&text).unwrap(), request);
    assert_eq!(
        serde_json::to_value(&request).unwrap()["template"],
        json!({"name": "extract", "version": 3})
    );

    let response = LlmResponse {
        json: json!({"claims": []}),
        usage: Some(LlmUsage {
            input_tokens: 41,
            output_tokens: 7,
        }),
        latency: Duration::from_millis(850),
    };
    let text = serde_json::to_string(&response).unwrap();
    assert_eq!(
        serde_json::from_str::<LlmResponse>(&text).unwrap(),
        response
    );
}

// The real models. Ignored: they run only when the models are present.

/// Where the real models are on this machine, resolved as the daemon would.
fn real_model_dir() -> ModelDir {
    let env = |name: &str| std::env::var_os(name).map(PathBuf::from);
    ModelDir::resolve(
        env("ASPHODEL_MODEL_DIR").as_deref(),
        env("XDG_CACHE_HOME").as_deref(),
        env("HOME").as_deref(),
    )
    .unwrap()
}

#[test]
#[ignore = "needs the real models: run `asphodel models fetch`, or set ASPHODEL_MODEL_DIR"]
fn real_models_embed_and_rerank() {
    let dir = real_model_dir();
    let models = Models::load(
        &dir,
        &ModelOptions {
            threads: std::num::NonZeroUsize::new(1),
        },
    )
    .unwrap_or_else(|error| panic!("loading from {}: {error}", dir.path().display()));
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
    for vector in &vectors {
        assert_eq!(vector.len(), EMBEDDING_DIMENSIONS);
        assert!((norm(vector) - 1.0).abs() < 1e-3, "norm {}", norm(vector));
    }
    let near = cosine(&vectors[0], &vectors[1]);
    let far = cosine(&vectors[0], &vectors[2]);
    assert!(near > far, "near {near} far {far}");
    // bge-small scores even unrelated pairs at 0.6 or more (TIM-93), so the
    // check is on the gap, not an absolute.
    assert!(near - far > 0.1, "near {near} far {far}");
    // Deterministic across calls and batches.
    let again = models.embedder.embed(&[texts[0]]).unwrap();
    for (a, b) in vectors[0].iter().zip(&again[0]) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
    assert!(models.embedder.embed(&[]).unwrap().is_empty());

    let query = "where did the cat sit";
    let scores = models.reranker.rerank(query, &texts).unwrap();
    assert_eq!(
        scores.len(),
        3,
        "one logit per document, in the documents' order"
    );
    assert!(scores[0] > scores[2], "{scores:?}");
    assert!(scores[1] > scores[2], "{scores:?}");
    assert!(scores.iter().all(|score| score.is_finite()), "{scores:?}");
    assert!(models.reranker.rerank(query, &[]).unwrap().is_empty());
}
