//! The configuration surface.
//!
//! Every setting sits in one of five places:
//!
//! - fixed in code, in [`crate::constants`];
//! - the daemon-wide [`Tuning`], read from an optional TOML file;
//! - [`Deployment`] flags and environment variables, with secrets from the
//!   environment only;
//! - the bank's identity, which lives in the bank row;
//! - flags on `asphodel replay` and `asphodel bench`.
//!
//! Tuning and deployment have separate precedence chains that never overlap:
//! tuning goes from code defaults to the file, and deployment from defaults
//! to the environment to flags. Changes take effect on restart only.

mod deployment;
mod fingerprint;
mod resolved;
mod tuning;

pub use deployment::{Deployment, LLM_API_KEY_ENV, Secret, TOKEN_ENV};
pub use fingerprint::{DeletionInputs, Fingerprint, deletion_fingerprint};
pub use resolved::{ModelsConfig, PurgePause, ResolvedConfig};
pub use tuning::{
    AgendaTuning, ClockTuning, ConfigError, ExtractionTuning, InjectionTuning, InvalidValue, Layer,
    LlmAuth, LlmTuning, MentalModelsTuning, PurgeTuning, RankingTuning, RecallTuning,
    ReconcileTuning, RerankQuery, SessionsTuning, SignificanceTuning, StrengthTuning, Tuning,
};
