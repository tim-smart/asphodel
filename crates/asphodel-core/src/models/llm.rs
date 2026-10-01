//! The LLM client: one trait, an OpenAI-compatible implementation and a
//! deterministic fake.
//!
//! The trait is synchronous. The extraction worker is a thread per bank
//! (TIM-92), and replay's cassette wrapper needs no runtime. The daemon
//! calls it from `spawn_blocking`.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::{Deployment, Secret, Tuning};

/// Where the LLM is and how to talk to it. The endpoint and model come from
/// `[llm]` in the tuning file, and the key from `ASPHODEL_LLM_API_KEY`
/// (ADR 0009).
#[derive(Debug, Clone)]
pub struct LlmSettings {
    /// The base URL, without the `/chat/completions` path.
    pub endpoint: String,
    /// The exact model string, sent as `model`.
    pub model: String,
    /// Sent as a bearer token when set. Local endpoints have none.
    pub api_key: Option<Secret>,
    /// The whole-request timeout.
    pub timeout: Duration,
}

impl LlmSettings {
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

    /// `Ok(None)` when neither `llm.endpoint` nor `llm.model` is set,
    /// `Ok(Some)` when both are, and [`LlmError::NotConfigured`] naming the
    /// missing key when only one is.
    pub fn from_config(tuning: &Tuning, deployment: &Deployment) -> Result<Option<Self>, LlmError> {
        match (&tuning.llm.endpoint, &tuning.llm.model) {
            (None, None) => Ok(None),
            (Some(_), None) => Err(LlmError::NotConfigured {
                missing: "llm.model",
            }),
            (None, Some(_)) => Err(LlmError::NotConfigured {
                missing: "llm.endpoint",
            }),
            (Some(endpoint), Some(model)) => Ok(Some(Self {
                endpoint: endpoint.clone(),
                model: model.clone(),
                api_key: deployment.llm_api_key.clone(),
                timeout: Self::DEFAULT_TIMEOUT,
            })),
        }
    }
}

/// Which prompt built a request, and its version. Replay's cache keys
/// include it next to the model id, so editing a prompt never hits a stale
/// recording (TIM-96, decision 4).
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
/// replay records in `live` mode (TIM-96, decision 3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmResponse {
    pub json: Value,
    /// `None` when the endpoint reports no usage.
    pub usage: Option<LlmUsage>,
    pub latency: Duration,
}

/// Why a call failed. No variant carries the prompt or the reply: only
/// sizes and statuses (TIM-96, decision 8).
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
}

impl LlmError {
    /// Whether the caller's retry policy may try again: transport errors,
    /// timeouts, 408, 429 and 5xx. Never for a reply that came back and was
    /// wrong.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Transport { .. } | Self::Timeout => true,
            Self::Status { status } => matches!(status, 408 | 429 | 500..=599),
            Self::NotConfigured { .. } | Self::NoContent | Self::NotJson { .. } | Self::Refused => {
                false
            }
        }
    }
}

/// The boundary extraction, reconciliation and refresh call through, and
/// that replay's recording and cassette modes wrap (TIM-96, decision 4).
pub trait LlmClient: Send + Sync {
    /// The model string sent with every request.
    fn model(&self) -> &str;

    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError>;
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
        body
    }
}

impl LlmClient for OpenAiCompatible {
    fn model(&self) -> &str {
        &self.settings.model
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
            return Err(LlmError::Status { status });
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

/// Maps a ureq error. The message names the error kind, never the body.
fn transport(error: ureq::Error) -> LlmError {
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
fn unfence(content: &str) -> &str {
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

    #[test]
    fn retryable_statuses() {
        for (status, retryable) in [
            (400, false),
            (401, false),
            (408, true),
            (429, true),
            (503, true),
        ] {
            assert_eq!(
                LlmError::Status { status }.is_retryable(),
                retryable,
                "{status}"
            );
        }
    }
}
