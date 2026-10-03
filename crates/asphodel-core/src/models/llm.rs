//! The LLM client: one trait, an OpenAI-compatible implementation and a
//! deterministic fake.
//!
//! The trait is synchronous. The extraction worker is a thread per bank, and
//! replay's cassette wrapper needs no runtime. The daemon
//! calls it from `spawn_blocking`.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use jiff::Timestamp;

use super::chatgpt::CODEX_ENDPOINT;
use crate::clock::{Clock, SystemClock};
use crate::config::{Deployment, LLM_API_KEY_ENV, LlmAuth, Secret, Tuning};

/// Where the LLM is and how to talk to it. The endpoint and model come from
/// `[llm]` in the tuning file, and the key from `ASPHODEL_LLM_API_KEY`
/// (ADR 0009).
#[derive(Debug, Clone)]
pub struct LlmSettings {
    /// Which wire format and credentials: Chat Completions with a key, or
    /// the Codex backend's Responses API with a subscription login.
    pub auth: LlmAuth,
    /// The base URL, without the `/chat/completions` or `/responses` path.
    /// In `chatgpt` mode it defaults to [`CODEX_ENDPOINT`].
    pub endpoint: String,
    /// The exact model string, sent as `model`.
    pub model: String,
    /// Sent as the request's reasoning effort when set.
    pub reasoning_effort: Option<String>,
    /// Sent as a bearer token when set. Local endpoints have none.
    pub api_key: Option<Secret>,
    /// The whole-request timeout.
    pub timeout: Duration,
}

impl LlmSettings {
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

    /// `api_key` mode: `Ok(None)` when neither `llm.endpoint` nor
    /// `llm.model` is set, `Ok(Some)` when both are, and
    /// [`LlmError::NotConfigured`] naming the missing key when only one is.
    /// `chatgpt` mode: the endpoint defaults to the Codex backend,
    /// `llm.model` is still required, and a key that is also set is
    /// [`LlmError::Conflicting`] rather than silently ignored.
    pub fn from_config(tuning: &Tuning, deployment: &Deployment) -> Result<Option<Self>, LlmError> {
        let llm = &tuning.llm;
        match llm.auth {
            LlmAuth::ApiKey => match (&llm.endpoint, &llm.model) {
                (None, None) => Ok(None),
                (Some(_), None) => Err(LlmError::NotConfigured {
                    missing: "llm.model",
                }),
                (None, Some(_)) => Err(LlmError::NotConfigured {
                    missing: "llm.endpoint",
                }),
                (Some(endpoint), Some(model)) => Ok(Some(Self {
                    auth: LlmAuth::ApiKey,
                    endpoint: endpoint.clone(),
                    model: model.clone(),
                    reasoning_effort: llm.reasoning_effort.clone(),
                    api_key: deployment.llm_api_key.clone(),
                    timeout: Self::DEFAULT_TIMEOUT,
                })),
            },
            LlmAuth::Chatgpt => {
                if deployment.llm_api_key.is_some() {
                    return Err(LlmError::Conflicting {
                        first: "llm.auth = \"chatgpt\"",
                        second: LLM_API_KEY_ENV,
                    });
                }
                let model = llm.model.clone().ok_or(LlmError::NotConfigured {
                    missing: "llm.model",
                })?;
                Ok(Some(Self {
                    auth: LlmAuth::Chatgpt,
                    endpoint: llm
                        .endpoint
                        .clone()
                        .unwrap_or_else(|| CODEX_ENDPOINT.to_string()),
                    model,
                    reasoning_effort: llm.reasoning_effort.clone(),
                    api_key: None,
                    timeout: Self::DEFAULT_TIMEOUT,
                }))
            }
        }
    }
}

/// Which prompt built a request, and its version. Replay's cache keys
/// include it next to the model id, so editing a prompt never hits a stale
/// recording.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Template {
    pub name: String,
    pub version: u32,
}

/// One structured-output call: a system and a user message, and the JSON
/// schema the reply must satisfy. `Serialize` so a cassette can store and
/// key it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmRequest {
    pub template: Template,
    pub system: String,
    pub user: String,
    /// The schema's name in `response_format`.
    pub schema_name: String,
    pub schema: Value,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// A reply that parsed as JSON. `latency` is the measured round trip, which
/// replay records in `live` mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmResponse {
    pub json: Value,
    /// `None` when the endpoint reports no usage.
    pub usage: Option<LlmUsage>,
    pub latency: Duration,
}

/// Why a call failed. No variant carries the prompt or the reply: only
/// sizes and statuses.
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

    /// The reply had no choices, or a choice with no content.
    #[error("the LLM returned no content")]
    NoContent,

    /// The content wasn't JSON, fenced or bare.
    #[error("the LLM returned {bytes} bytes that aren't JSON")]
    NotJson { bytes: usize },

    /// The model refused (`message.refusal`).
    #[error("the LLM refused the request")]
    Refused,

    /// Two settings that can't both hold, named by key.
    #[error("{first} and {second} are both set; unset one")]
    Conflicting {
        first: &'static str,
        second: &'static str,
    },

    /// No token file, or a refreshed credential the backend still rejects.
    /// The queue can't proceed until the owner logs in again.
    #[error("the ChatGPT login is missing or no longer valid: run `asphodel llm login`")]
    LoginRequired,

    /// The subscription's usage window is spent. Not a failure: the
    /// extraction queue holds until `resets_at` (world time).
    #[error("the ChatGPT usage limit is reached until {resets_at}")]
    UsageLimited { resets_at: Timestamp },

    /// The endpoint answered 429 with a `Retry-After`. Not a failure: every
    /// caller holds for that long. A 429 without one is a [`Self::Status`].
    #[error("the LLM is rate limited for {}s", retry_after.as_secs())]
    RateLimited { retry_after: Duration },

    /// The backend reported a failed or incomplete response. Only the code
    /// is kept.
    #[error("the LLM backend failed the request: {code}")]
    Backend { code: String },
}

impl LlmError {
    /// Whether the caller's retry policy may try again: transport errors,
    /// timeouts, 408, 429 and 5xx. Never for a reply that came back and was
    /// wrong, and never for a usage limit or a 429 that said when to come
    /// back, which are deferred to then instead.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Transport { .. } | Self::Timeout => true,
            Self::Status { status } => matches!(status, 408 | 429 | 500..=599),
            Self::NotConfigured { .. }
            | Self::NoContent
            | Self::NotJson { .. }
            | Self::Refused
            | Self::Conflicting { .. }
            | Self::LoginRequired
            | Self::UsageLimited { .. }
            | Self::RateLimited { .. }
            | Self::Backend { .. } => false,
        }
    }
}

/// The boundary extraction, reconciliation and refresh call through, and
/// that replay's recording and cassette modes wrap.
pub trait LlmClient: Send + Sync {
    /// The model string sent with every request.
    fn model(&self) -> &str;

    /// The reasoning effort sent with every request, if any.
    fn reasoning_effort(&self) -> Option<&str> {
        None
    }

    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError>;

    /// [`LlmClient::complete`] for a request whose handles (`m1`, `e1`, …)
    /// stand for the memories and entries `identities` names. Handles are
    /// positional, so only the identities say which memory a recorded
    /// reply meant: replay's cassette keeps them, to carry a recorded
    /// refresh over to a run where the handles name other memories. Every other
    /// client ignores them.
    fn complete_identified(
        &self,
        request: &LlmRequest,
        identities: &[(String, uuid::Uuid)],
    ) -> Result<LlmResponse, LlmError> {
        let _ = identities;
        self.complete(request)
    }
}

/// The real client, for any OpenAI-compatible `chat/completions`.
pub struct OpenAiCompatible {
    settings: LlmSettings,
    agent: ureq::Agent,
}

impl OpenAiCompatible {
    /// Replies past this size aren't read: a structured reply is never
    /// megabytes, and an unbounded read is a way to exhaust memory.
    const REPLY_LIMIT: u64 = 16 * 1024 * 1024;

    pub fn new(settings: LlmSettings) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(settings.timeout))
            .http_status_as_error(false)
            .build();
        Self {
            settings,
            agent: config.into(),
        }
    }

    fn url(&self) -> String {
        format!(
            "{}/chat/completions",
            self.settings.endpoint.trim_end_matches('/')
        )
    }

    fn body(&self, request: &LlmRequest) -> Value {
        let mut body = json!({
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "strict": true,
                    "schema": request.schema,
                },
            },
        });
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        if let Some(effort) = &self.settings.reasoning_effort {
            body["reasoning_effort"] = json!(effort);
        }
        body
    }
}

impl LlmClient for OpenAiCompatible {
    fn model(&self) -> &str {
        &self.settings.model
    }

    fn reasoning_effort(&self) -> Option<&str> {
        self.settings.reasoning_effort.as_deref()
    }

    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        let started = Instant::now();
        let mut call = self
            .agent
            .post(self.url())
            .header("Content-Type", "application/json");
        if let Some(key) = &self.settings.api_key {
            call = call.header("Authorization", &format!("Bearer {}", key.expose()));
        }
        let mut response = call.send_json(self.body(request)).map_err(transport)?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            // A Retry-After date is the server's wall time, whatever clock
            // the caller runs on.
            return Err(status_error(status, response.headers(), SystemClock.now()));
        }
        let text = response
            .body_mut()
            .with_config()
            .limit(Self::REPLY_LIMIT)
            .read_to_string()
            .map_err(transport)?;
        let latency = started.elapsed();
        let reply: Value =
            serde_json::from_str(&text).map_err(|_| LlmError::NotJson { bytes: text.len() })?;
        let json = parse_content(&reply)?;
        Ok(LlmResponse {
            json,
            usage: parse_usage(&reply),
            latency,
        })
    }
}

/// A non-2xx status as an error: a 429 with a `Retry-After` is
/// [`LlmError::RateLimited`], anything else [`LlmError::Status`].
/// `Retry-After` is seconds or an HTTP date (RFC 9110), read against `now`;
/// a date already past holds for no time. A value that's neither counts
/// like no header.
pub(super) fn status_error(
    status: u16,
    headers: &ureq::http::HeaderMap,
    now: Timestamp,
) -> LlmError {
    let retry_after = headers
        .get("retry-after")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| retry_after(value.trim(), now));
    match (status, retry_after) {
        (429, Some(retry_after)) => LlmError::RateLimited { retry_after },
        _ => LlmError::Status { status },
    }
}

/// A `Retry-After` value as a wait from `now`, or `None` when it's neither
/// seconds nor an HTTP date.
fn retry_after(value: &str, now: Timestamp) -> Option<Duration> {
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let at = jiff::fmt::rfc2822::DateTimeParser::new()
        .parse_timestamp(value)
        .ok()?;
    Some(Duration::try_from(now.duration_until(at)).unwrap_or(Duration::ZERO))
}

/// Maps a ureq error. The message names the error kind, never the body.
pub(super) fn transport(error: ureq::Error) -> LlmError {
    match error {
        ureq::Error::Timeout(_) => LlmError::Timeout,
        ureq::Error::StatusCode(status) => LlmError::Status { status },
        other => LlmError::Transport {
            reason: other.to_string(),
        },
    }
}

/// `choices[0].message.content` as JSON, unwrapping a code fence if the
/// model put one round it.
fn parse_content(reply: &Value) -> Result<Value, LlmError> {
    let message = reply
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .ok_or(LlmError::NoContent)?;
    match message.get("content").and_then(Value::as_str) {
        Some(content) => {
            let content = unfence(content);
            serde_json::from_str(content).map_err(|_| LlmError::NotJson {
                bytes: content.len(),
            })
        }
        None => {
            if message
                .get("refusal")
                .and_then(Value::as_str)
                .is_some_and(|refusal| !refusal.is_empty())
            {
                Err(LlmError::Refused)
            } else {
                Err(LlmError::NoContent)
            }
        }
    }
}

/// Strips a ```json ... ``` fence, if there is one, and surrounding
/// whitespace.
pub(super) fn unfence(content: &str) -> &str {
    let content = content.trim();
    let Some(rest) = content.strip_prefix("```") else {
        return content;
    };
    let rest = rest.strip_suffix("```").unwrap_or(rest);
    // The opening fence may carry a language tag on its line.
    let rest = match rest.split_once('\n') {
        Some((tag, body)) if tag.trim().chars().all(char::is_alphanumeric) => body,
        _ => rest,
    };
    rest.trim()
}

fn parse_usage(reply: &Value) -> Option<LlmUsage> {
    let usage = reply.get("usage")?;
    Some(LlmUsage {
        input_tokens: usage.get("prompt_tokens")?.as_u64()?,
        output_tokens: usage.get("completion_tokens")?.as_u64()?,
    })
}

/// A deterministic client for tests: it hands out scripted replies in order
/// and records every request it was given.
pub struct FakeLlm {
    model: String,
    mode: Mode,
    requests: Mutex<Vec<LlmRequest>>,
}

enum Mode {
    Scripted(Mutex<VecDeque<Value>>),
    Failing(fn() -> LlmError),
    Script(Mutex<VecDeque<ScriptStep>>),
}

/// One step of a script for [`FakeLlm::from_script`]: a reply or a failure,
/// optionally after a delay. A script file is a JSON array of them:
///
/// ```json
/// [
///   {"reply": {"claims": [], "used": []}},
///   {"fail": "status", "status": 503},
///   {"fail": "usage_limited", "resets_at": "2026-10-02T00:00:00Z"},
///   {"fail": "status", "status": 429, "retry_after_secs": 30},
///   {"reply": {"claims": [], "used": []}, "delay_ms": 2000}
/// ]
/// ```
///
/// It's how integration tests drive the daemon's extraction worker
/// without an LLM (`ASPHODEL_LLM_SCRIPT`), and the delay lets them catch a
/// chunk in flight.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScriptStep {
    /// The reply's JSON. Exactly one of `reply` and `fail` is set.
    #[serde(default)]
    pub reply: Option<Value>,
    /// The failure to return instead.
    #[serde(default)]
    pub fail: Option<ScriptedFailure>,
    /// The HTTP status for `"fail": "status"`; 500 when absent.
    #[serde(default)]
    pub status: Option<u16>,
    /// When the usage window resets, for `"fail": "usage_limited"`.
    #[serde(default)]
    pub resets_at: Option<Timestamp>,
    /// The `Retry-After` of a `"fail": "status"` with status 429, which
    /// makes it [`LlmError::RateLimited`].
    #[serde(default)]
    pub retry_after_secs: Option<u64>,
    /// How long the call takes before it answers, in milliseconds.
    #[serde(default)]
    pub delay_ms: u64,
}

/// The [`LlmError`] a [`ScriptStep`] fails with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScriptedFailure {
    Transport,
    Timeout,
    Status,
    NoContent,
    NotJson,
    Refused,
    LoginRequired,
    UsageLimited,
}

/// Why a script didn't load. It names the step, never its content.
#[derive(Debug, thiserror::Error)]
pub enum ScriptError {
    #[error("the LLM script isn't a JSON array of steps: {reason}")]
    Parse { reason: String },
    #[error("step {step} of the LLM script needs exactly one of `reply` and `fail`")]
    Ambiguous { step: usize },
    #[error("step {step} of the LLM script fails with usage_limited but has no `resets_at`")]
    NoReset { step: usize },
}

impl ScriptStep {
    fn outcome(&self) -> Result<Value, LlmError> {
        if let Some(reply) = &self.reply {
            return Ok(reply.clone());
        }
        Err(match self.fail.unwrap_or(ScriptedFailure::NoContent) {
            ScriptedFailure::Transport => LlmError::Transport {
                reason: "scripted".into(),
            },
            ScriptedFailure::Timeout => LlmError::Timeout,
            ScriptedFailure::Status => match (self.status.unwrap_or(500), self.retry_after_secs) {
                (429, Some(seconds)) => LlmError::RateLimited {
                    retry_after: Duration::from_secs(seconds),
                },
                (status, _) => LlmError::Status { status },
            },
            ScriptedFailure::NoContent => LlmError::NoContent,
            ScriptedFailure::NotJson => LlmError::NotJson { bytes: 0 },
            ScriptedFailure::Refused => LlmError::Refused,
            ScriptedFailure::LoginRequired => LlmError::LoginRequired,
            ScriptedFailure::UsageLimited => LlmError::UsageLimited {
                resets_at: self
                    .resets_at
                    .expect("a usage_limited step was checked for resets_at"),
            },
        })
    }
}

impl FakeLlm {
    /// Replies with each JSON value in turn, then fails with
    /// [`LlmError::NoContent`] once they run out.
    pub fn scripted(model: &str, replies: Vec<Value>) -> Self {
        Self {
            model: model.to_string(),
            mode: Mode::Scripted(Mutex::new(replies.into())),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Plays `script`, a JSON array of [`ScriptStep`]s, one step per call,
    /// then fails with [`LlmError::NoContent`] once it runs out.
    pub fn from_script(model: &str, script: &str) -> Result<Self, ScriptError> {
        let steps: Vec<ScriptStep> =
            serde_json::from_str(script).map_err(|error| ScriptError::Parse {
                reason: error.to_string(),
            })?;
        for (index, step) in steps.iter().enumerate() {
            let step_number = index + 1;
            if step.reply.is_some() == step.fail.is_some() {
                return Err(ScriptError::Ambiguous { step: step_number });
            }
            if step.fail == Some(ScriptedFailure::UsageLimited) && step.resets_at.is_none() {
                return Err(ScriptError::NoReset { step: step_number });
            }
        }
        Ok(Self {
            model: model.to_string(),
            mode: Mode::Script(Mutex::new(steps.into())),
            requests: Mutex::new(Vec::new()),
        })
    }

    /// Fails every call with the error `make` builds.
    pub fn failing(model: &str, make: fn() -> LlmError) -> Self {
        Self {
            model: model.to_string(),
            mode: Mode::Failing(make),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Every request so far, in order.
    pub fn requests(&self) -> Vec<LlmRequest> {
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl LlmClient for FakeLlm {
    fn model(&self) -> &str {
        &self.model
    }

    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        self.requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        match &self.mode {
            Mode::Scripted(replies) => {
                let json = replies
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .pop_front()
                    .ok_or(LlmError::NoContent)?;
                Ok(LlmResponse {
                    json,
                    usage: None,
                    latency: Duration::ZERO,
                })
            }
            Mode::Failing(make) => Err(make()),
            Mode::Script(steps) => {
                let step = steps
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .pop_front()
                    .ok_or(LlmError::NoContent)?;
                if step.delay_ms > 0 {
                    std::thread::sleep(Duration::from_millis(step.delay_ms));
                }
                Ok(LlmResponse {
                    json: step.outcome()?,
                    usage: None,
                    latency: Duration::from_millis(step.delay_ms),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fences_are_stripped() {
        assert_eq!(unfence("{\"a\":1}"), "{\"a\":1}");
        assert_eq!(unfence("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(unfence("```\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(unfence("  ```json\n[1, 2]\n```  "), "[1, 2]");
        assert_eq!(unfence("```{\"a\":1}```"), "{\"a\":1}");
    }
}
