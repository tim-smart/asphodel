//! Deployment settings: flags with an `ASPHODEL_*` environment variable each,
//! and secrets from the environment only.
//!
//! The binary parses the flags; this is the plain record of what it resolved,
//! so the resolved config can show it with the secrets redacted.

use std::fmt;
use std::path::PathBuf;

use serde::{Serialize, Serializer};

/// The bearer token clients must send. Mandatory off loopback.
pub const TOKEN_ENV: &str = "ASPHODEL_TOKEN";

/// The LLM's API key.
pub const LLM_API_KEY_ENV: &str = "ASPHODEL_LLM_API_KEY";

/// What `asphodel serve` was started with.
#[derive(Debug, Clone, Serialize)]
pub struct Deployment {
    /// `--listen` / `ASPHODEL_LISTEN`: `host:port` or `unix:/path`.
    pub listen: String,

    /// `--data-dir` / `ASPHODEL_DATA_DIR`.
    pub data_dir: Option<PathBuf>,

    /// `--config` / `ASPHODEL_CONFIG`: the tuning file.
    pub config: Option<PathBuf>,

    /// `--allow-network-fs` / `ASPHODEL_ALLOW_NETWORK_FS`.
    pub allow_network_fs: bool,

    /// `--model-dir` / `ASPHODEL_MODEL_DIR`: the embedding and reranker models.
    pub model_dir: Option<PathBuf>,

    /// `ASPHODEL_TOKEN`, environment only.
    pub token: Option<Secret>,

    /// `ASPHODEL_LLM_API_KEY`, environment only.
    pub llm_api_key: Option<Secret>,
}

/// A secret value. It never shows in `Debug` output, logs or serialised
/// config: all of them print `[redacted]`.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    const REDACTED: &'static str = "[redacted]";

    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Reads a secret from the environment. Unset, empty and non-UTF-8
    /// values all count as absent.
    pub fn from_env(name: &str) -> Option<Self> {
        std::env::var(name)
            .ok()
            .filter(|value| !value.is_empty())
            .map(Self)
    }

    /// The secret itself, for the one place that has to send or compare it.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(Self::REDACTED)
    }
}

impl Serialize for Secret {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(Self::REDACTED)
    }
}
