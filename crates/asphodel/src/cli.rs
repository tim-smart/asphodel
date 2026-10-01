//! The subcommand tree from "API surface and Hermes transport" (TIM-94,
//! decision 10), plus `replay` and `bench` from the replay harness decision
//! (TIM-96). Only `serve` does anything yet; the rest are stubs that later
//! stages fill in.

use std::path::PathBuf;

use anyhow::bail;
use clap::{Args, Parser, Subcommand};

use crate::listen::Listen;

/// Brain-like memory for the Hermes agent.
#[derive(Debug, Parser)]
#[command(name = "asphodel", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the daemon.
    Serve(ServeArgs),

    /// Ingest a document into a bank.
    Ingest(IngestArgs),

    /// Create a bank or change its configuration.
    #[command(subcommand)]
    Bank(BankCommand),

    /// List extraction chunks, and retry the failed ones.
    Chunks(ChunksArgs),

    /// Recall memories for a query.
    Recall(RecallArgs),

    /// Erase memories, every version of them, and the passages they came from.
    Forget(IdsArgs),

    /// Mark memories as kept, so they never fade.
    Keep(IdsArgs),

    /// Hand kept memories back to the significance extraction gave them.
    Unkeep(IdsArgs),

    /// Fetch and manage the local models.
    #[command(subcommand)]
    Models(ModelsCommand),

    /// Replay recorded sessions on a simulated clock.
    Replay(ReplayArgs),

    /// Run concurrent prefetches against a daemon on a copy of a store.
    Bench(BenchArgs),
}

/// Flags shared by every subcommand that is an HTTP client of the daemon.
#[derive(Debug, Args)]
pub struct ClientArgs {
    /// Base URL of the daemon.
    #[arg(long, env = "ASPHODEL_URL", default_value = "http://127.0.0.1:7720")]
    pub url: String,

    /// Bearer token, required when the daemon listens off loopback.
    #[arg(long, env = "ASPHODEL_TOKEN", hide_env_values = true)]
    pub token: Option<String>,
}

/// Deployment flags (ADR 0009). Each has an `ASPHODEL_*` environment
/// variable, and a flag wins over its variable. The secrets, `ASPHODEL_TOKEN`
/// and `ASPHODEL_LLM_API_KEY`, have no flag: they come from the environment
/// only, so they never show up in a process list.
#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Address to listen on: `host:port`, or `unix:/path` for a socket.
    #[arg(long, env = "ASPHODEL_LISTEN", default_value_t = Listen::default())]
    pub listen: Listen,

    /// Directory holding the SQLite store and its lock. Required, with no
    /// default, so an unmounted volume can't become an empty store.
    #[arg(long, env = "ASPHODEL_DATA_DIR")]
    pub data_dir: PathBuf,

    /// The tuning file (TOML). Without one, the code defaults apply.
    #[arg(long, env = "ASPHODEL_CONFIG")]
    pub config: Option<PathBuf>,

    /// Run even when the data dir is on a network filesystem.
    #[arg(long, env = "ASPHODEL_ALLOW_NETWORK_FS")]
    pub allow_network_fs: bool,

    /// Directory holding the embedding and reranker models.
    #[arg(long, env = "ASPHODEL_MODEL_DIR")]
    pub model_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct IngestArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    /// The file to ingest, as plain text or markdown.
    pub file: PathBuf,

    /// The bank to ingest into.
    #[arg(long)]
    pub bank: String,

    /// The reference date relative times in the document resolve against.
    #[arg(long)]
    pub date: String,

    /// The reference date is approximate.
    #[arg(long)]
    pub inexact: bool,

    /// Source id to store the document under. Defaults to one derived from the file.
    #[arg(long)]
    pub id: Option<String>,
}

#[derive(Debug, Subcommand)]
pub enum BankCommand {
    /// Create a bank.
    Create {
        #[command(flatten)]
        client: ClientArgs,
        /// The bank's name.
        name: String,
    },
    /// Change a bank's owner, assistant or timezone.
    Config {
        #[command(flatten)]
        client: ClientArgs,
        /// The bank's name.
        name: String,
    },
}

#[derive(Debug, Args)]
pub struct ChunksArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    /// The bank whose chunks to list.
    #[arg(long)]
    pub bank: String,

    /// Only chunks whose extraction failed.
    #[arg(long)]
    pub failed: bool,

    /// Put the listed failed chunks back on the extraction queue.
    #[arg(long, requires = "failed")]
    pub retry: bool,
}

#[derive(Debug, Args)]
pub struct RecallArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    /// The bank to recall from.
    #[arg(long)]
    pub bank: String,

    /// What to recall.
    pub query: String,
}

#[derive(Debug, Args)]
pub struct IdsArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    /// The bank the memories live in.
    #[arg(long)]
    pub bank: String,

    /// Memory ids.
    #[arg(required = true)]
    pub ids: Vec<String>,
}

#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    /// Download the embedding and reranker models into the model dir.
    Fetch {
        /// Where the models go. Defaults to the XDG cache dir.
        #[arg(long, env = "ASPHODEL_MODEL_DIR")]
        model_dir: Option<PathBuf>,
    },
}

#[derive(Debug, Args)]
pub struct ReplayArgs {
    /// The private directory holding the corpus, cassettes and reports.
    #[arg(long, env = "ASPHODEL_REPLAY_DIR")]
    pub replay_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct BenchArgs {
    #[command(flatten)]
    pub client: ClientArgs,
}

impl Cli {
    pub fn run(self) -> anyhow::Result<()> {
        match self.command {
            Command::Serve(args) => serve(args),
            Command::Ingest(_) => stub("ingest"),
            Command::Bank(BankCommand::Create { .. }) => stub("bank create"),
            Command::Bank(BankCommand::Config { .. }) => stub("bank config"),
            Command::Chunks(_) => stub("chunks"),
            Command::Recall(_) => stub("recall"),
            Command::Forget(_) => stub("forget"),
            Command::Keep(_) => stub("keep"),
            Command::Unkeep(_) => stub("unkeep"),
            Command::Models(ModelsCommand::Fetch { .. }) => stub("models fetch"),
            Command::Replay(_) => stub("replay"),
            Command::Bench(_) => stub("bench"),
        }
    }
}

fn serve(args: ServeArgs) -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(crate::serve::run(args))
}

fn stub(name: &str) -> anyhow::Result<()> {
    bail!("`asphodel {name}` isn't implemented yet")
}
