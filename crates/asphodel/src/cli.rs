//! The subcommand tree from "API surface and Hermes transport" (TIM-94,
//! decision 10), plus `replay` and `bench` from the replay harness decision
//! (TIM-96). `serve` and `models fetch` do something; the rest are stubs
//! that later stages fill in.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::{Context, bail};
use asphodel_core::SystemClock;
use asphodel_core::models::{
    AUTH_ISSUER, DeviceCode, HttpFetcher, ModelDir, TokenStore, device_code_login, fetch_models,
    manifest,
};
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

    /// Log in to the LLM subscription.
    #[command(subcommand)]
    Llm(LlmCommand),

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

    /// Intra-op threads for ONNX Runtime. Unset leaves it to the runtime.
    #[arg(long, env = "ASPHODEL_ONNX_THREADS")]
    pub onnx_threads: Option<NonZeroUsize>,
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

/// `asphodel llm login` writes the token file under the data dir. It
/// never goes through the daemon and never reads the Codex CLI's
/// credentials. `ASPHODEL_LLM_ISSUER` (environment only, for tests) points
/// the flow at another issuer.
#[derive(Debug, Subcommand)]
pub enum LlmCommand {
    /// Log in to a ChatGPT subscription with a device code.
    Login {
        /// The daemon's data dir, where the token file goes.
        #[arg(long, env = "ASPHODEL_DATA_DIR")]
        data_dir: PathBuf,
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
            Command::Models(ModelsCommand::Fetch { model_dir }) => models_fetch(model_dir),
            Command::Llm(LlmCommand::Login { data_dir }) => llm_login(&data_dir),
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

/// Resolves the model dir as the daemon does: the flag or
/// `ASPHODEL_MODEL_DIR`, else the XDG cache, else under `HOME`.
pub(crate) fn resolve_model_dir(
    override_dir: Option<&std::path::Path>,
) -> anyhow::Result<ModelDir> {
    let env = |name: &str| std::env::var_os(name).map(PathBuf::from);
    Ok(ModelDir::resolve(
        override_dir,
        env("XDG_CACHE_HOME").as_deref(),
        env("HOME").as_deref(),
    )?)
}

/// `asphodel models fetch`: fills the model dir from the manifest, skipping
/// files already present with the right checksum (TIM-94, decision 4).
fn models_fetch(model_dir: Option<PathBuf>) -> anyhow::Result<()> {
    let dir = resolve_model_dir(model_dir.as_deref())?;
    let report = fetch_models(&dir, &manifest(), &HttpFetcher::new())
        .with_context(|| format!("filling {}", dir.path().display()))?;
    for path in &report.fetched {
        println!("fetched {}", path.display());
    }
    for path in &report.skipped {
        println!("kept    {}", path.display());
    }
    println!(
        "{} files in {}: {} fetched, {} already there",
        report.fetched.len() + report.skipped.len(),
        dir.path().display(),
        report.fetched.len(),
        report.skipped.len()
    );
    Ok(())
}

/// The issuer override, for tests against a loopback stub. Hidden from
/// `--help` on purpose.
const LLM_ISSUER_ENV: &str = "ASPHODEL_LLM_ISSUER";

/// `asphodel llm login`: the device-code flow, printing the URL and the
/// code and nothing else. The daemon picks the file up on its next call.
fn llm_login(data_dir: &std::path::Path) -> anyhow::Result<()> {
    let issuer = std::env::var(LLM_ISSUER_ENV).unwrap_or_else(|_| AUTH_ISSUER.to_string());
    let store = TokenStore::open(data_dir);
    let mut show = |code: &DeviceCode| {
        println!("Sign in to ChatGPT to let Asphodel use your subscription:");
        println!();
        println!("  1. Open {}", code.verification_url);
        println!("  2. Enter the code {}", code.user_code);
        println!();
        println!("The code expires in 15 minutes. Waiting for approval...");
    };
    device_code_login(&issuer, &store, &SystemClock, &mut show)
        .with_context(|| format!("logging in at {issuer}"))?;
    println!("Logged in. Tokens saved to {}", store.path().display());
    Ok(())
}

fn stub(name: &str) -> anyhow::Result<()> {
    bail!("`asphodel {name}` isn't implemented yet")
}
