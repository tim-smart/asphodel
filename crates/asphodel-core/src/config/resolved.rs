//! The resolved config: one struct for `GET /v1/config`, `asphodel config
//! show`, the startup log line and the replay report.

use serde::Serialize;

use crate::config::{Deployment, Fingerprint, Tuning};
use crate::constants::FixedConstants;

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

impl ResolvedConfig {
    /// Resolves a validated tuning and the deployment it runs under. The
    /// purge state starts unchecked; the store sets it at startup.
    pub fn new(tuning: Tuning, deployment: Deployment) -> Self {
        Self {
            version: crate::VERSION,
            git_sha: option_env!("ASPHODEL_GIT_SHA"),
            deletion_fingerprint: tuning.deletion_fingerprint(),
            tuning,
            deployment,
            constants: FixedConstants::current(),
            purge: PurgePause::Unchecked,
        }
    }
}
