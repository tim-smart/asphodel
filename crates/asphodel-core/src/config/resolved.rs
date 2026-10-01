//! The resolved config: one struct for `GET /v1/config`, `asphodel config
//! show`, the startup log line and the replay report.

use serde::Serialize;

use crate::config::{Deployment, Fingerprint, Tuning};
use crate::constants::FixedConstants;
use crate::models::Models;

/// Everything the daemon is running with, safe to show: secrets serialise
/// as `[redacted]`.
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedConfig {
    pub version: &'static str,

    /// Set at build time from `ASPHODEL_GIT_SHA`, when the build provides it.
    pub git_sha: Option<&'static str>,

    pub tuning: Tuning,
    pub deployment: Deployment,

    /// The fixed constants, read-only.
    pub constants: FixedConstants,

    /// The fingerprint of this tuning's deletion inputs.
    pub deletion_fingerprint: Fingerprint,

    pub purge: PurgePause,

    /// The models the daemon serves with. `None` until they're loaded.
    pub models: Option<ModelsConfig>,
}

/// Whether purge and the sweep are running, given the fingerprint stored in
/// the store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum PurgePause {
    /// No store has been compared against yet.
    Unchecked,
    /// The stored fingerprint matches, or was just recorded on first start.
    Running,
    /// The stored fingerprint differs, so purge and the sweep wait for an
    /// operator to acknowledge the new one.
    Paused { stored: Fingerprint },
}

/// The models as the resolved config shows them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelsConfig {
    /// The exact embedding model string.
    pub embedding: String,
    /// The exact reranker model string.
    pub reranker: String,
    /// Whether these are the deterministic fakes rather than the ONNX
    /// models. An operator reading the log should never mistake one for
    /// the other.
    pub fake: bool,
    /// `--onnx-threads`, when set.
    pub onnx_threads: Option<usize>,
}

impl ModelsConfig {
    pub fn new(models: &Models, fake: bool, onnx_threads: Option<usize>) -> Self {
        let ids = models.ids();
        Self {
            embedding: ids.embedding,
            reranker: ids.reranker,
            fake,
            onnx_threads,
        }
    }
}

impl ResolvedConfig {
    /// Resolves a validated tuning and the deployment it runs under. The
    /// purge state starts unchecked; the store sets it at startup. The
    /// models are set once they're loaded.
    pub fn new(tuning: Tuning, deployment: Deployment) -> Self {
        Self {
            version: crate::VERSION,
            git_sha: option_env!("ASPHODEL_GIT_SHA"),
            deletion_fingerprint: tuning.deletion_fingerprint(),
            tuning,
            deployment,
            constants: FixedConstants::current(),
            purge: PurgePause::Unchecked,
            models: None,
        }
    }
}
