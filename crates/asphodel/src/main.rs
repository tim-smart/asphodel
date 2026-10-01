//! The `asphodel` binary: the daemon under `serve`, and a CLI client of it
//! under every other subcommand (ADR 0006, ADR 0010). `replay` is the one
//! exception, opening its own store under `ASPHODEL_REPLAY_DIR` (TIM-96).

mod cli;
mod listen;
mod serve;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();
    asphodel_core::logging::init();
    cli.run()
}
