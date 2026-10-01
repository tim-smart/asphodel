//! Asphodel's service layer.
//!
//! The daemon (`asphodel serve`) and the replay harness (`asphodel replay`)
//! both drive this crate in-process. Nothing in here reads the wall clock:
//! every operation takes its notion of "now" from the [`clock::Clock`] it was
//! given, so a replay can run years of bank time in minutes (ADR 0004).

pub mod chunking;
pub mod clock;
pub mod config;
pub mod constants;
pub mod extraction;
pub mod ingest;
pub mod logging;
pub mod models;
pub mod queue;
pub mod retrieval;
pub mod secrets;
pub mod service;
mod sessions;
pub mod store;
pub mod strength;

pub use clock::{Clock, SimulatedClock, SystemClock};
pub use config::{ResolvedConfig, Tuning};
pub use models::Models;
pub use service::{Health, Housekeeping, OpenError, Service};
pub use store::Store;

/// The version of this build, as reported by `/v1/health`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
