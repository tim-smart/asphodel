//! The `asphodel` binary. Everything is in the library crate; this parses
//! the command line and runs it.

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let cli = asphodel::cli::Cli::parse();
    asphodel_core::logging::init();
    cli.run()
}
