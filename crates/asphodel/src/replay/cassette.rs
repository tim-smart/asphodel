//! The cassette of recorded LLM calls and the modes that read it.
//!
//! The cassette is JSON lines under the private dir, one record per call:
//! the request, the reply, the measured latency, the simulated time, and
//! for extraction calls which chunk it was and the handles call 1 was
//! shown with a hash of each sentence. A record's key is SHA-256 of the
//! model id, the template name and version, and the whole request, so
//! editing a prompt never hits a stale recording.
//!
//! - `live` answers from the cassette and calls and records on a miss.
//! - `replay` answers from the cassette and fails on a miss.
//! - `fast` reuses call 1's claims by chunk, `[llm] language` and the
//!   `[extraction] guidance` hash, and
//!   `used` verdicts by (reply hash, sentence hash) pair, judges the pairs
//!   nobody has judged with one short top-up call, and answers call 2 and
//!   refreshes by request key, calling the LLM on a miss when one is
//!   configured. `--refresh` says how refreshes are answered.
//!
//! A record carries prompt text, so the cassette never leaves the private
//! dir. Nothing here logs content.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context as _;
use asphodel_core::extraction::{CALL1_TEMPLATE, CALL1_VERSION, CALL2_TEMPLATE, Call1Input};
use asphodel_core::mental_models::REFRESH_TEMPLATE;
use asphodel_core::models::{LlmClient, LlmError, LlmRequest, LlmResponse, Template};
use asphodel_core::{Clock, SimulatedClock};
use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::corpus::{hex, sha256};
use super::report::{LlmCounts, Percentiles, UsedVerdicts};
use crate::cli::{RefreshMode, ReplayMode};

/// The top-up call's template: the reply and the new sentences only.
pub const JUDGE_TEMPLATE: &str = "judge_used";
pub const JUDGE_VERSION: u32 = 1;

const JUDGE_SYSTEM: &str = "You judge which remembered facts an assistant's reply actually relied on. \
The memories are listed with handles (m1, m2, ...). Reply with the handles of the memories the reply \
relied on, in `used`. Being shown a memory isn't using it: only a reply whose content rests on it counts. \
Reply with an empty list when none does.";

/// The model name a cassette with no live client answers under, until a
/// record says otherwise.
const UNRECORDED_MODEL: &str = "unrecorded";

/// How many times a live call is tried on a retryable error, and the wait
/// between tries, the worker's policy in small.
const ATTEMPTS: u32 = 3;
const RETRY_WAIT: Duration = Duration::from_millis(500);

/// One recorded call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub key: String,
    pub model: String,
    pub template: Template,
    /// Simulated time of the call.
    pub at: Timestamp,
    pub latency_ms: u64,
    #[serde(default)]
    pub chunk: Option<ChunkKey>,
    /// The in-context handles call 1 (or a top-up) was shown, each with
    /// the SHA-256 of its sentence.
    #[serde(default)]
    pub in_context: Vec<HandleHash>,
    /// The mental model entries call 1 was shown, each with the handles it
    /// cites.
    #[serde(default)]
    pub entries: Vec<EntryHandles>,
    /// SHA-256 of the assistant's reply in the chunk.
    #[serde(default)]
    pub reply_hash: Option<String>,
    /// For a refresh, the memory or entry each of its handles stood for,
    /// so a substituted reply can be carried over by identity.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub identities: Vec<(String, Uuid)>,
    /// The run's `[llm] language`, which call 1 claims and refresh entries
    /// are written in. A record from before the setting has none, as did
    /// the run that made it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub request: LlmRequest,
    pub response: LlmResponse,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChunkKey {
    pub source: Uuid,
    pub position: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandleHash {
    pub handle: String,
    pub sentence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryHandles {
    pub handle: String,
    pub cites: Vec<String>,
}

/// What the engine tells the recorder about the chunk being extracted,
/// before the calls for it.
#[derive(Debug, Clone)]
pub struct ChunkContext {
    pub key: ChunkKey,
    pub reply_text: String,
    pub reply_hash: String,
    pub in_context: Vec<InContext>,
    pub entries: Vec<EntryHandles>,
}

#[derive(Debug, Clone)]
pub struct InContext {
    pub handle: String,
    pub memory: Uuid,
    pub sentence: String,
    pub hash: String,
}

impl ChunkContext {
    pub fn new(key: ChunkKey, input: &Call1Input) -> Self {
        let reply_text: String = match input.reply_start {
            Some(start) => input.text.chars().skip(start).collect(),
            None => String::new(),
        };
        Self {
            key,
            reply_hash: sha256(&reply_text),
            reply_text,
            in_context: input
                .in_context
                .iter()
                .map(|memory| InContext {
                    handle: memory.handle.clone(),
                    memory: memory.memory,
                    sentence: memory.content.clone(),
                    hash: sha256(&memory.content),
                })
                .collect(),
            entries: input
                .entries
                .iter()
                .map(|entry| EntryHandles {
                    handle: entry.handle.clone(),
                    cites: entry.cites.clone(),
                })
                .collect(),
        }
    }
}

/// What `fast` reuses call 1's claims by: the chunk, the template version
/// and guidance hash, the model and the language.
type ClaimsKey = (ChunkKey, u32, Option<String>, String, Option<String>);

/// The records, indexed three ways.
#[derive(Default)]
struct Index {
    records: Vec<Record>,
    by_key: BTreeMap<String, usize>,
    /// Call 1 records by chunk, template version and guidance hash, model
    /// and language.
    claims: BTreeMap<ClaimsKey, usize>,
    /// `used` verdicts by (reply hash, sentence hash).
    pairs: BTreeMap<(String, String), bool>,
    /// Refresh records, for the nearest-in-time substitution.
    refreshes: Vec<usize>,
}

impl Index {
    fn insert(&mut self, record: Record) {
        let index = self.records.len();
        self.by_key.insert(record.key.clone(), index);
        if let Some(chunk) = &record.chunk
            && record.template.name == CALL1_TEMPLATE
        {
            self.claims.insert(
                (
                    chunk.clone(),
                    record.template.version,
                    record.template.guidance.clone(),
                    record.model.clone(),
                    record.language.clone(),
                ),
                index,
            );
        }
        if record.template.name == REFRESH_TEMPLATE {
            self.refreshes.push(index);
        }
        if let Some(reply_hash) = &record.reply_hash {
            let used = used_handles(&record);
            for memory in &record.in_context {
                let judged = used.contains(&memory.handle)
                    || record.entries.iter().any(|entry| {
                        used.contains(&entry.handle) && entry.cites.contains(&memory.handle)
                    });
                self.pairs
                    .insert((reply_hash.clone(), memory.sentence.clone()), judged);
            }
        }
        self.records.push(record);
    }
}

/// The handles a call 1 or top-up reply says were used.
fn used_handles(record: &Record) -> Vec<String> {
    let field = if record.template.name == JUDGE_TEMPLATE {
        "used"
    } else {
        "used_injected_ids"
    };
    record
        .response
        .json
        .get(field)
        .and_then(Value::as_array)
        .map(|handles| {
            handles
                .iter()
                .filter_map(Value::as_str)
                .map(|handle| handle.trim().to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// Running counts, and the measured latencies.
#[derive(Default)]
pub struct Counts {
    pub cache: u64,
    pub top_up: u64,
    pub live: u64,
    pub misses: u64,
    pub verdicts: UsedVerdicts,
    pub latencies_ms: Vec<u64>,
    /// Call 2 requests seen, however answered.
    pub call2: u64,
    /// Simulated times of refresh requests, however answered.
    pub refresh_times: Vec<Timestamp>,
}

/// The cassette-backed [`LlmClient`].
pub struct Recorder {
    path: PathBuf,
    mode: ReplayMode,
    refresh: RefreshMode,
    no_cache: bool,
    live: Option<Arc<dyn LlmClient>>,
    model: String,
    /// The run's `[llm] language`. Call 1 claims recorded in another
    /// language are never reused by chunk.
    language: Option<String>,
    /// The hash of the run's `[extraction] guidance`. Call 1 claims
    /// recorded under other guidance, or none, are never reused by chunk.
    guidance: Option<String>,
    clock: Arc<SimulatedClock>,
    index: Mutex<Index>,
    chunk: Mutex<Option<ChunkContext>>,
    counts: Mutex<Counts>,
    file: Mutex<Option<File>>,
    first_miss: Mutex<Option<String>>,
    /// The latency of the responses served since the last
    /// [`Recorder::enter`], in milliseconds: recorded for the records that
    /// answered, measured for live calls.
    served_ms: Mutex<u64>,
    /// SHA-256 of the file as it stood when opened.
    pub hash: String,
}

impl Recorder {
    /// Opens the cassette, reading what it holds. The file is created on
    /// the first record, never followed through a symlink. With
    /// `no_cache` it's emptied instead: a re-recording starts afresh, and
    /// the hash is of what the run could read, which is nothing.
    pub fn open(
        path: &Path,
        mode: ReplayMode,
        refresh: RefreshMode,
        no_cache: bool,
        live: Option<Arc<dyn LlmClient>>,
        language: Option<String>,
        clock: Arc<SimulatedClock>,
    ) -> anyhow::Result<Self> {
        super::refuse_symlink(path)?;
        let bytes = if no_cache {
            if path.exists() {
                File::create(path)
                    .with_context(|| format!("emptying the cassette {}", path.display()))?;
            }
            Vec::new()
        } else {
            match fs::read(path) {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("reading the cassette {}", path.display()));
                }
            }
        };
        let hash = hex(&Sha256::digest(&bytes));
        let text = std::str::from_utf8(&bytes)
            .with_context(|| format!("the cassette {} isn't UTF-8", path.display()))?;
        let mut index = Index::default();
        for (number, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let record: Record = serde_json::from_str(line).map_err(|error| {
                super::json_error(path, number + 1, "a cassette record", &error)
            })?;
            index.insert(record);
        }
        let model = match &live {
            // Recordings at another effort are another model's answers.
            Some(live) => match live.reasoning_effort() {
                Some(effort) => format!("{} reasoning={effort}", live.model()),
                None => live.model().to_string(),
            },
            None => index
                .records
                .last()
                .map(|record| record.model.clone())
                .unwrap_or_else(|| UNRECORDED_MODEL.into()),
        };
        Ok(Self {
            path: path.to_owned(),
            mode,
            refresh,
            no_cache,
            live,
            model,
            language,
            guidance: None,
            clock,
            index: Mutex::new(index),
            chunk: Mutex::new(None),
            counts: Mutex::new(Counts::default()),
            file: Mutex::new(None),
            first_miss: Mutex::new(None),
            served_ms: Mutex::new(0),
            hash,
        })
    }

    /// Tells the recorder which chunk the next calls are for.
    pub fn enter(&self, context: ChunkContext) {
        *lock(&self.served_ms) = 0;
        *lock(&self.chunk) = Some(context);
    }

    pub fn leave(&self) {
        *lock(&self.chunk) = None;
    }

    /// How long the calls for the chunk since [`Recorder::enter`] took: the
    /// recorded latency of each record that answered, and the measured latency
    /// of each live call, which is what its record will say. Only the responses
    /// actually served count, so older recordings of the chunk, other models'
    /// and top-ups that weren't needed never do.
    pub fn served_latency(&self) -> Duration {
        Duration::from_millis(*lock(&self.served_ms))
    }

    /// The first miss that couldn't be answered, if any.
    pub fn first_miss(&self) -> Option<String> {
        lock(&self.first_miss).clone()
    }

    pub fn with_counts<T>(&self, f: impl FnOnce(&mut Counts) -> T) -> T {
        f(&mut lock(&self.counts))
    }

    /// Simulated times of the refresh requests so far, drained.
    pub fn take_refresh_times(&self) -> Vec<Timestamp> {
        std::mem::take(&mut lock(&self.counts).refresh_times)
    }

    /// The counts as the report carries them.
    pub fn llm_counts(&self) -> LlmCounts {
        let counts = lock(&self.counts);
        let mut latencies = counts.latencies_ms.clone();
        LlmCounts {
            scripted: 0,
            cache: counts.cache,
            top_up: counts.top_up,
            live: counts.live,
            misses: counts.misses,
            used_verdicts: counts.verdicts.clone(),
            latency_ms: Percentiles::of(&mut latencies),
        }
    }

    /// Sets the run's `[extraction] guidance` hash, which `fast` reuses call
    /// 1's claims by.
    pub fn with_guidance(mut self, guidance: Option<String>) -> Self {
        self.guidance = guidance;
        self
    }

    /// `fast` mode's call 1: the chunk's recorded claims, the `used` verdicts
    /// the pair cache holds, and one top-up for the pairs it doesn't. `None`
    /// when the chunk has no record, which is a miss the caller answers by
    /// request key.
    pub fn compose_call1(&self, context: &ChunkContext) -> Result<Option<Value>, LlmError> {
        let (claims, mut used, unknown) = {
            let index = lock(&self.index);
            let key = (
                context.key.clone(),
                CALL1_VERSION,
                self.guidance.clone(),
                self.model.clone(),
                self.language.clone(),
            );
            let Some(&position) = index.claims.get(&key) else {
                return Ok(None);
            };
            let record = &index.records[position];
            *lock(&self.served_ms) += record.latency_ms;
            let claims = record
                .response
                .json
                .get("claims")
                .cloned()
                .unwrap_or_else(|| json!([]));
            let mut used = Vec::new();
            let mut unknown = Vec::new();
            for memory in &context.in_context {
                match index
                    .pairs
                    .get(&(context.reply_hash.clone(), memory.hash.clone()))
                {
                    Some(true) => used.push(memory.handle.clone()),
                    Some(false) => {}
                    None => unknown.push(memory.clone()),
                }
            }
            (claims, used, unknown)
        };
        let known = (context.in_context.len() - unknown.len()) as u64;
        {
            let mut counts = lock(&self.counts);
            counts.cache += 1;
            counts.verdicts.recorded += known;
        }
        if !unknown.is_empty() {
            used.extend(self.top_up(context, &unknown)?);
        }
        Ok(Some(json!({ "claims": claims, "used_injected_ids": used })))
    }

    /// One short call judging the reply against the sentences nobody has
    /// judged it against, recorded like any other.
    fn top_up(
        &self,
        context: &ChunkContext,
        unknown: &[InContext],
    ) -> Result<Vec<String>, LlmError> {
        let mut user = format!("Reply:\n{}\n\nMemories:\n", context.reply_text);
        for memory in unknown {
            user.push_str(&format!("{}: {}\n", memory.handle, memory.sentence));
        }
        let request = LlmRequest {
            template: Template {
                name: JUDGE_TEMPLATE.into(),
                version: JUDGE_VERSION,
                guidance: None,
            },
            system: JUDGE_SYSTEM.to_owned(),
            user,
            schema_name: "used_memories".into(),
            schema: json!({
                "type": "object",
                "properties": {
                    "used": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["used"],
                "additionalProperties": false
            }),
            max_tokens: Some(200),
        };
        let subset = ChunkContext {
            key: context.key.clone(),
            reply_text: context.reply_text.clone(),
            reply_hash: context.reply_hash.clone(),
            in_context: unknown.to_vec(),
            entries: Vec::new(),
        };
        let response = self.answer(&request, Some(&subset), &[])?;
        let judged: Vec<String> = response
            .json
            .get("used")
            .and_then(Value::as_array)
            .map(|handles| {
                handles
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|handle| handle.trim().to_string())
                    .filter(|handle| unknown.iter().any(|memory| &memory.handle == handle))
                    .collect()
            })
            .unwrap_or_default();
        let mut counts = lock(&self.counts);
        counts.top_up += 1;
        counts.verdicts.top_up += unknown.len() as u64;
        Ok(judged)
    }

    /// Answers a request from the cassette, or the live client on a miss
    /// when the mode allows, recording what it called. `context` is the
    /// chunk to tag a record with; the current one when `None`.
    fn answer(
        &self,
        request: &LlmRequest,
        context: Option<&ChunkContext>,
        identities: &[(String, Uuid)],
    ) -> Result<LlmResponse, LlmError> {
        let key = key_of(&self.model, request);
        if !self.no_cache {
            let index = lock(&self.index);
            if let Some(&position) = index.by_key.get(&key) {
                let record = &index.records[position];
                let response = record.response.clone();
                let latency_ms = record.latency_ms;
                drop(index);
                *lock(&self.served_ms) += latency_ms;
                lock(&self.counts).cache += 1;
                return Ok(response);
            }
        }
        lock(&self.counts).misses += 1;
        let live = match (&self.live, self.mode) {
            (Some(live), ReplayMode::Live | ReplayMode::Fast) => live,
            _ => {
                let message = format!(
                    "cassette miss: no recorded reply for a {} v{} call{}",
                    request.template.name,
                    request.template.version,
                    match context.or(lock(&self.chunk).as_ref()) {
                        Some(context) => format!(
                            " on chunk {} of source {}",
                            context.key.position, context.key.source
                        ),
                        None => String::new(),
                    }
                );
                let mut first = lock(&self.first_miss);
                if first.is_none() {
                    *first = Some(message.clone());
                }
                return Err(LlmError::Transport { reason: message });
            }
        };
        let response = call_with_retries(live.as_ref(), request)?;
        let latency_ms = u64::try_from(response.latency.as_millis()).unwrap_or(u64::MAX);
        *lock(&self.served_ms) += latency_ms;
        {
            let mut counts = lock(&self.counts);
            counts.live += 1;
            counts.latencies_ms.push(latency_ms);
        }
        let current = lock(&self.chunk);
        let context = context.or(current.as_ref());
        let record = Record {
            key,
            model: self.model.clone(),
            template: request.template.clone(),
            at: self.clock.now(),
            latency_ms,
            chunk: context.map(|context| context.key.clone()),
            in_context: context
                .filter(|_| request.template.name != CALL2_TEMPLATE)
                .map(|context| {
                    context
                        .in_context
                        .iter()
                        .map(|memory| HandleHash {
                            handle: memory.handle.clone(),
                            sentence: memory.hash.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            entries: context
                .filter(|_| request.template.name != CALL2_TEMPLATE)
                .map(|context| context.entries.clone())
                .unwrap_or_default(),
            reply_hash: context
                .filter(|_| request.template.name != CALL2_TEMPLATE)
                .map(|context| context.reply_hash.clone()),
            identities: identities.to_vec(),
            language: self.language.clone(),
            request: request.clone(),
            response: response.clone(),
        };
        drop(current);
        self.append(record).map_err(|error| LlmError::Transport {
            reason: format!("writing the cassette: {error}"),
        })?;
        Ok(response)
    }

    fn append(&self, record: Record) -> anyhow::Result<()> {
        let mut file = lock(&self.file);
        if file.is_none() {
            if let Some(parent) = self.path.parent() {
                fs::create_dir_all(parent)?;
            }
            super::refuse_symlink(&self.path)?;
            *file = Some(
                fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(&self.path)?,
            );
        }
        let mut line = serde_json::to_vec(&record)?;
        line.push(b'\n');
        file.as_mut()
            .expect("the cassette is open")
            .write_all(&line)?;
        lock(&self.index).insert(record);
        Ok(())
    }

    /// `fast` with `--refresh recorded`: the recorded refresh of the same
    /// mental model, by the question line the request starts with, nearest
    /// in simulated time, among those made with this run's LLM model, language, and
    /// the request's template version. Its operations are carried over by
    /// identity: each handle goes to the memory or entry it stood for when
    /// recorded, then to that one's handle now, and an operation whose
    /// entry or any cited memory isn't in this input is dropped, since
    /// it would say something the LLM never said about what's here.
    fn nearest_refresh(
        &self,
        request: &LlmRequest,
        identities: &[(String, Uuid)],
    ) -> Option<LlmResponse> {
        let question = request.user.lines().next().unwrap_or("").to_string();
        let now = self.clock.now();
        let index = lock(&self.index);
        let record = index
            .refreshes
            .iter()
            .map(|&position| &index.records[position])
            .filter(|record| {
                record.model == self.model
                    && record.language == self.language
                    && record.template.version == request.template.version
                    && record.request.user.lines().next().unwrap_or("") == question
            })
            .min_by_key(|record| {
                let distance = record.at.duration_since(now).abs();
                (distance, record.at)
            })?;
        Some(carry_over(record, identities))
    }
}

/// A recorded refresh's response with its handles carried over to
/// `identities` (see [`Recorder::nearest_refresh`]). A record without
/// identities, from before they were kept, carries nothing over.
fn carry_over(record: &Record, identities: &[(String, Uuid)]) -> LlmResponse {
    let meant: BTreeMap<&str, Uuid> = record
        .identities
        .iter()
        .map(|(handle, id)| (handle.as_str(), *id))
        .collect();
    let current: BTreeMap<Uuid, &str> = identities
        .iter()
        .map(|(handle, id)| (*id, handle.as_str()))
        .collect();
    let carry = |handle: &str| -> Option<String> {
        let id = meant.get(handle.trim())?;
        current.get(id).map(|handle| (*handle).to_string())
    };
    let operations = record
        .response
        .json
        .get("operations")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut kept = Vec::new();
    for mut operation in operations {
        if let Some(entry) = operation.get("entry").and_then(Value::as_str) {
            match carry(entry) {
                Some(handle) => operation["entry"] = Value::from(handle),
                None => continue,
            }
        }
        if let Some(cites) = operation.get("cites").and_then(Value::as_array) {
            let carried: Option<Vec<Value>> = cites
                .iter()
                .map(|cite| cite.as_str().and_then(carry).map(Value::from))
                .collect();
            match carried {
                Some(cites) => operation["cites"] = Value::Array(cites),
                None => continue,
            }
        }
        kept.push(operation);
    }
    let mut json = record.response.json.clone();
    json["operations"] = Value::Array(kept);
    LlmResponse {
        json,
        ..record.response.clone()
    }
}

impl LlmClient for Recorder {
    fn model(&self) -> &str {
        &self.model
    }

    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        self.complete_identified(request, &[])
    }

    fn complete_identified(
        &self,
        request: &LlmRequest,
        identities: &[(String, Uuid)],
    ) -> Result<LlmResponse, LlmError> {
        if request.template.name == REFRESH_TEMPLATE {
            lock(&self.counts).refresh_times.push(self.clock.now());
            if self.mode == ReplayMode::Fast {
                match self.refresh {
                    RefreshMode::Off => {
                        return Ok(LlmResponse {
                            json: json!({ "operations": [] }),
                            usage: None,
                            latency: Duration::ZERO,
                        });
                    }
                    RefreshMode::Recorded => {
                        if let Some(response) = self.nearest_refresh(request, identities) {
                            lock(&self.counts).cache += 1;
                            return Ok(response);
                        }
                    }
                    RefreshMode::Live => {}
                }
            }
        }
        if request.template.name == CALL2_TEMPLATE {
            lock(&self.counts).call2 += 1;
        }
        self.answer(request, None, identities)
    }
}

/// A client that answers its first request with a composed reply and the
/// rest through the recorder: `fast` mode's call 1, then call 2.
pub struct Chained<'a> {
    first: Mutex<Option<Value>>,
    rest: &'a Recorder,
}

impl<'a> Chained<'a> {
    pub fn new(first: Value, rest: &'a Recorder) -> Self {
        Self {
            first: Mutex::new(Some(first)),
            rest,
        }
    }
}

impl LlmClient for Chained<'_> {
    fn model(&self) -> &str {
        self.rest.model()
    }

    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        if request.template.name == CALL1_TEMPLATE
            && let Some(json) = lock(&self.first).take()
        {
            return Ok(LlmResponse {
                json,
                usage: None,
                latency: Duration::ZERO,
            });
        }
        self.rest.complete(request)
    }
}

/// The worker's retry policy in small: a retryable error is tried again a
/// few times after a short wait, anything else fails at once.
fn call_with_retries(live: &dyn LlmClient, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
    let mut attempt = 1;
    loop {
        match live.complete(request) {
            Ok(response) => return Ok(response),
            Err(error) if error.is_retryable() && attempt < ATTEMPTS => {
                attempt += 1;
                std::thread::sleep(RETRY_WAIT);
            }
            Err(error) => return Err(error),
        }
    }
}

/// SHA-256 of the model id, the template and the whole request.
pub fn key_of(model: &str, request: &LlmRequest) -> String {
    let material = serde_json::to_vec(&json!({
        "model": model,
        "template": request.template,
        "system": request.system,
        "user": request.user,
        "schema_name": request.schema_name,
        "schema": request.schema,
        "max_tokens": request.max_tokens,
    }))
    .expect("a request serialises");
    hex(&Sha256::digest(&material))
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
