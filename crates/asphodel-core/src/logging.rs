//! Tracing setup, and the rule on what may be logged.
//!
//! Memory content is logged only at `trace`. Content means sentences,
//! source text, recall queries, entity names and aliases, and LLM bodies.
//! Anything at `debug` or above carries ids, counts, kinds and durations
//! only, and panic messages carry ids only. `docs/logging.md` has the full
//! rule.

use std::io::IsTerminal;

use tracing_subscriber::EnvFilter;

/// The environment variable that sets the log filter.
pub const FILTER_ENV: &str = "ASPHODEL_LOG";

/// The filter used when [`FILTER_ENV`] is unset.
pub const DEFAULT_FILTER: &str = "info";

/// Installs the global `tracing` subscriber.
///
/// The filter comes from `ASPHODEL_LOG`, falling back to `info`. Output goes
/// to stderr so a CLI's stdout stays parseable. Calling this twice is a
/// programming error and panics, which is what `tracing` does on a second
/// global subscriber.
pub fn init() {
    let filter =
        EnvFilter::try_from_env(FILTER_ENV).unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .with_target(true)
        .init();
}
