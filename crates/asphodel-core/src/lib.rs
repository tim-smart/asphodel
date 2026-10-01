//! Asphodel's service layer.
//!
//! The daemon (`asphodel serve`) and the replay harness (`asphodel replay`)
//! both drive this crate in-process. Nothing in here reads the wall clock:
//! every operation takes its notion of "now" from the [`clock::Clock`] it was
//! given, so a replay can run years of bank time in minutes (ADR 0004).

pub mod clock;
pub mod config;
pub mod constants;
pub mod logging;
pub mod service;

pub use clock::{Clock, SimulatedClock, SystemClock};
pub use config::{ResolvedConfig, Tuning};
pub use service::{Health, Service};

/// The version of this build, as reported by `/v1/health`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
