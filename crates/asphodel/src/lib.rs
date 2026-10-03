//! The `asphodel` binary's crate: the daemon under `serve`, a CLI client of it
//! under every other subcommand, and `replay`, the one exception, which opens
//! its own store under `ASPHODEL_REPLAY_DIR` and runs the service layer
//! in-process on a simulated clock.
//!
//! It's a library as well as a binary so the replay tests can read
//! scenarios with the same loader the command uses.

pub mod bench;
pub mod cli;
mod client;
mod listen;
pub mod replay;
mod serve;
