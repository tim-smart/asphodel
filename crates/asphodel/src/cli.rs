//! The daemon and replay subcommands. `serve` runs the daemon; `ingest`,
//! `bank`, `chunks`, `recall`, `forget`, `keep`, `unkeep`, `memory`, `entity`,
//! `model`, `reembed`, `purge`, `backup`, `status` and the audit lists are HTTP
//! clients of it ([`crate::client`]); `models fetch` and `llm login` work on
//! files, and `restore` works on the data dir offline. `replay` runs scripted
//! scenarios and real-history corpora, `import` writes those corpora, `report
//! diff` compares runs, `report html` renders one, `report precision` turns
//! labelled material into a precision curve, and `bench` drives a daemon on a
//! copy of a replayed store.

use std::io::Write;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::{Context, bail};
use asphodel_core::SystemClock;
use asphodel_core::config::{Secret, TOKEN_ENV};
use asphodel_core::constants::Volatility;
use asphodel_core::entities::{AliasRemoval, LinkRequest, MergeRequest};
use asphodel_core::ingest::Document;
use asphodel_core::keep::{MemoryIds, SignificanceRequest};
use asphodel_core::mental_models::{ModelEdit, ModelSpec};
use asphodel_core::models::{
    AUTH_ISSUER, DeviceCode, HttpFetcher, ModelDir, TokenStore, device_code_login, fetch_models,
    manifest,
};
use asphodel_core::operations::{
    AuditList, BACKED_UP_AT_HEADER, LENGTH_HEADER, SHA256_HEADER, check_copy, temp_file_beside,
};
use asphodel_core::queue::RetryRequest;
use asphodel_core::retrieval::{On, PhaseFilter, RecallRequest};
use asphodel_core::store::OpenOptions;
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::strength::Kind;
use asphodel_core::sweep::PurgeAck;
use axum::http::HeaderMap;
use clap::{Args, Parser, Subcommand};
use jiff::Timestamp;
use jiff::civil::Date;
use serde::de::DeserializeOwned;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::client::{Client, segment};
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

    /// Show a memory, or set the owner's significance on it.
    #[command(subcommand)]
    Memory(MemoryCommand),

    /// Show and correct entities: merge, unmerge, aliases and links.
    #[command(subcommand)]
    Entity(EntityCommand),

    /// Define, list, show and refresh a bank's mental models.
    #[command(subcommand)]
    Model(ModelCommand),

    /// Re-embed a bank with the daemon's embedding model, then swap it in.
    /// The bank is served with its recorded model until the swap.
    Reembed(ReembedArgs),

    /// See and acknowledge a purge pause.
    #[command(subcommand)]
    Purge(PurgeCommand),

    /// Stream a checked copy of the store from the daemon to a file or
    /// stdout.
    Backup(BackupArgs),

    /// Replace the store with a backup. Offline: stop the daemon first.
    Restore(RestoreArgs),

    /// Show the queue, failures, the purge pause, the last sweep and backup.
    /// Exits non-zero when anything needs attention.
    Status(StatusArgs),

    /// List a bank's purges, newest first.
    Purges(ListArgs),

    /// List a bank's forgets, newest first.
    Forgets(ListArgs),

    /// List a bank's nightly sweeps, newest first.
    Sweeps(ListArgs),

    /// List a bank's recalls with their queries, newest first.
    Recalls(ListArgs),

    /// Fetch and manage the local models.
    #[command(subcommand)]
    Models(ModelsCommand),

    /// Log in to the LLM subscription.
    #[command(subcommand)]
    Llm(LlmCommand),

    /// Turn a copy of Hermes' `state.db` into a replay corpus.
    Import(ImportArgs),

    /// Replay recorded sessions on a simulated clock.
    Replay(ReplayArgs),

    /// Compare, render and calibrate from replay reports.
    #[command(subcommand)]
    Report(ReportCommand),

    /// Run concurrent prefetches against a daemon on a copy of a store.
    Bench(BenchArgs),
}

/// Flags shared by every subcommand that is an HTTP client of the daemon.
/// The bearer token, which the daemon requires off loopback, comes from
/// `ASPHODEL_TOKEN` only, like the daemon's, so it never shows up in a
/// process list.
#[derive(Debug, Args)]
pub struct ClientArgs {
    /// The daemon: `http://host:port`, or `unix:/path` for a socket.
    #[arg(long, env = "ASPHODEL_URL", default_value = "http://127.0.0.1:7720")]
    pub url: String,

    /// Print the daemon's JSON reply instead of a summary.
    #[arg(long)]
    pub json: bool,

    #[arg(skip = Secret::from_env(TOKEN_ENV).map(|token| token.expose().to_string()))]
    pub token: Option<String>,
}

/// Deployment flags. Each has an `ASPHODEL_*` environment variable, and a
/// flag wins over its variable. The secrets, `ASPHODEL_TOKEN` and
/// `ASPHODEL_LLM_API_KEY`, have no flag: they come from the environment only,
/// so they never show up in a process list.
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

    /// The reference date relative times in the document resolve against,
    /// as YYYY-MM-DD.
    #[arg(long)]
    pub date: Date,

    /// The reference date is approximate.
    #[arg(long)]
    pub inexact: bool,

    /// Document id to store it under. Ingesting the same id again with new
    /// text is an edit. Defaults to the file's name.
    #[arg(long)]
    pub id: Option<String>,
}

#[derive(Debug, Subcommand)]
pub enum BankCommand {
    /// Create a bank. If it already exists, the given fields are merged in.
    Create {
        #[command(flatten)]
        client: ClientArgs,
        /// The bank's name.
        name: String,
        #[command(flatten)]
        identity: IdentityArgs,
    },
    /// Change a bank's owner, assistant or timezone. Fields not given are
    /// left as they are, and a new name adds an alias without removing one.
    Config {
        #[command(flatten)]
        client: ClientArgs,
        /// The bank's name.
        name: String,
        #[command(flatten)]
        identity: IdentityArgs,
    },
    /// Delete a bank and everything in it, through the erase path. Disable
    /// the plugin first: its `initialize` creates the bank again, empty.
    Delete {
        #[command(flatten)]
        client: ClientArgs,
        /// The bank's name.
        name: String,
        /// The bank's name again.
        #[arg(long)]
        confirm: String,
    },
}

/// A bank's identity, as `PUT /v1/banks/{bank}` takes it.
#[derive(Debug, Args)]
pub struct IdentityArgs {
    /// The owner's name, an alias of the `user` entity.
    #[arg(long)]
    pub owner_name: Option<String>,

    /// One of the owner's platform ids, as `<platform>:<id>` (such as
    /// `discord:1234`). Repeat for several.
    #[arg(long = "owner-id")]
    pub owner_ids: Vec<String>,

    /// The assistant's name, an alias of the `assistant` entity.
    #[arg(long)]
    pub assistant_name: Option<String>,

    /// The IANA timezone for sources that don't give one.
    #[arg(long)]
    pub timezone: Option<String>,
}

impl IdentityArgs {
    fn identity(self) -> BankIdentity {
        BankIdentity {
            owner_name: self.owner_name,
            owner_platform_ids: self.owner_ids,
            assistant_name: self.assistant_name,
            timezone: self.timezone,
        }
    }
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

    /// Only memories from this time on (RFC 3339).
    #[arg(long)]
    pub from: Option<Timestamp>,

    /// Only memories up to this time (RFC 3339).
    #[arg(long)]
    pub to: Option<Timestamp>,

    /// What `--from` and `--to` apply to: when it happened, or when it was
    /// said.
    #[arg(long, value_parser = serde_value::<On>, default_value = "happened")]
    pub on: On,

    /// upcoming, past, current or any.
    #[arg(long, value_parser = serde_value::<PhaseFilter>, default_value = "any")]
    pub phase: PhaseFilter,

    /// Only this kind: fact, event, state, task or recurring. Repeat for
    /// several.
    #[arg(long = "kind", value_parser = serde_value::<Kind>)]
    pub kinds: Vec<Kind>,

    /// Only memories about this entity, by name or alias.
    #[arg(long)]
    pub entity: Option<String>,

    /// How many results, at most.
    #[arg(long)]
    pub limit: Option<usize>,
}

/// Parses a flag value through the enum's own JSON name, so the CLI and the
/// API spell it the same way.
fn serde_value<T: DeserializeOwned>(value: &str) -> Result<T, String> {
    serde_json::from_value(Value::String(value.to_string())).map_err(|error| error.to_string())
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

/// `asphodel purge`: purge and the sweep pause when the settings that
/// decide deletion change, until an operator acknowledges them.
#[derive(Debug, Subcommand)]
pub enum PurgeCommand {
    /// Show which settings changed and what the sweep would delete now. It
    /// deletes nothing.
    Plan {
        #[command(flatten)]
        client: ClientArgs,
    },

    /// Acknowledge the running daemon's deletion fingerprint, so purging
    /// resumes at the next sweep.
    Ack {
        #[command(flatten)]
        client: ClientArgs,

        /// The fingerprint `purge plan` shows as current.
        #[arg(long)]
        hash: String,
    },
}

/// `asphodel backup`. Asphodel has no destination, schedule or retention of
/// its own: a timer pipes `--out -` wherever it should go.
#[derive(Debug, Args)]
pub struct BackupArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    /// Where the copy goes: a file, or `-` for stdout. A file is written
    /// beside its final name and only renamed into place once its length,
    /// hash and integrity check pass.
    #[arg(long)]
    pub out: PathBuf,
}

/// `asphodel restore`: offline, under the data-dir lock.
#[derive(Debug, Args)]
pub struct RestoreArgs {
    /// The backup to restore.
    pub file: PathBuf,

    /// The data dir whose store it replaces. The current database is moved
    /// aside, not deleted.
    #[arg(long, env = "ASPHODEL_DATA_DIR")]
    pub data_dir: PathBuf,

    /// Run even when the data dir is on a network filesystem.
    #[arg(long, env = "ASPHODEL_ALLOW_NETWORK_FS")]
    pub allow_network_fs: bool,
}

#[derive(Debug, Args)]
pub struct StatusArgs {
    #[command(flatten)]
    pub client: ClientArgs,
}

/// The audit lists.
#[derive(Debug, Args)]
pub struct ListArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    /// The bank whose list to show.
    #[arg(long)]
    pub bank: String,

    /// How many rows, newest first. The daemon caps it at 1000.
    #[arg(long)]
    pub limit: Option<usize>,
}

/// `asphodel model`: only the owner defines models, through here or the API.
#[derive(Debug, Subcommand)]
pub enum ModelCommand {
    /// Define a model: a standing question its entries answer.
    Create(ModelCreateArgs),

    /// List the bank's models with their entries and citations.
    List(ModelListArgs),

    /// Change a model's question, filters, size, or whether it's enabled.
    Edit(ModelEditArgs),

    /// Refresh a model now. It's skipped when its inputs haven't changed,
    /// unless `--force`.
    Refresh(ModelRefreshArgs),

    /// Show a model's entries with the memories each cites and whether the
    /// block shows it.
    Show(ModelShowArgs),
}

#[derive(Debug, Args)]
pub struct ModelShowArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    #[arg(long)]
    pub bank: String,

    /// The model's name.
    pub name: String,

    /// Only this entry, by id.
    #[arg(long)]
    pub entry: Option<String>,
}

/// `asphodel memory`: a memory's metadata can be edited, never its sentence,
/// kind or window.
#[derive(Debug, Subcommand)]
pub enum MemoryCommand {
    /// Show a memory: both significance fields, its passage or why it's
    /// gone, its accesses, edits and chain, the secret-scan kinds, its
    /// strength in parts, what holds back a purge, and projected fade and
    /// purge dates.
    Show {
        #[command(flatten)]
        client: ClientArgs,
        #[arg(long)]
        bank: String,
        /// The memory's id.
        id: String,
    },

    /// Set the owner's significance on a memory: trivial, minor, notable,
    /// major, critical or kept, or `clear` to hand it back to the level
    /// extraction gave.
    Significance {
        #[command(flatten)]
        client: ClientArgs,
        #[arg(long)]
        bank: String,
        /// The memory's id.
        id: String,
        level: String,
    },
}

/// `asphodel entity`. An entity is named by its id, `user`,
/// `assistant`, or a name or alias only one entity of the bank has.
#[derive(Debug, Subcommand)]
pub enum EntityCommand {
    /// Show an entity: its aliases, merges, linked memories and edits.
    Show {
        #[command(flatten)]
        client: ClientArgs,
        #[arg(long)]
        bank: String,
        entity: String,
    },

    /// Merge one entity into another. `user` and `assistant` can only be
    /// merged into.
    Merge {
        #[command(flatten)]
        client: ClientArgs,
        #[arg(long)]
        bank: String,
        from: String,
        into: String,
    },

    /// Undo a merge, by the edit id `entity merge` printed.
    Unmerge {
        #[command(flatten)]
        client: ClientArgs,
        #[arg(long)]
        bank: String,
        edit: String,
    },

    /// Remove a wrong alias.
    #[command(subcommand)]
    Alias(AliasCommand),

    /// Link a memory to an entity.
    Link {
        #[command(flatten)]
        client: ClientArgs,
        #[arg(long)]
        bank: String,
        memory: String,
        entity: String,
    },

    /// Unlink a memory from an entity.
    Unlink {
        #[command(flatten)]
        client: ClientArgs,
        #[arg(long)]
        bank: String,
        memory: String,
        entity: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum AliasCommand {
    /// Remove an alias from an entity. With `--relink-to`, the links that
    /// named the entity by it move to that entity, which gets the alias.
    Rm {
        #[command(flatten)]
        client: ClientArgs,
        #[arg(long)]
        bank: String,
        entity: String,
        alias: String,
        #[arg(long)]
        relink_to: Option<String>,
    },
}

#[derive(Debug, Args)]
pub struct ReembedArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    /// The bank to re-embed.
    #[arg(long)]
    pub bank: String,

    /// Start the job and return, rather than follow it to the swap.
    #[arg(long)]
    pub no_wait: bool,
}

#[derive(Debug, Args)]
pub struct ModelCreateArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    #[arg(long)]
    pub bank: String,

    /// The model's name.
    pub name: String,

    /// The standing question its entries answer.
    #[arg(long)]
    pub question: String,

    /// The model's share of the prompt block, in tokens.
    #[arg(long)]
    pub max_tokens: u32,

    /// Only this kind: fact, event, state, task or recurring. Repeat for
    /// several; none means every kind.
    #[arg(long = "kind", value_parser = serde_value::<Kind>)]
    pub kinds: Vec<Kind>,

    /// Only memories about this entity, by name or alias.
    #[arg(long)]
    pub entity: Option<String>,

    /// Leave out states less durable than this: hours, days, weeks, months
    /// or years.
    #[arg(long, value_parser = serde_value::<Volatility>)]
    pub min_volatility: Option<Volatility>,

    /// Create it disabled: not refreshed and not rendered.
    #[arg(long)]
    pub disabled: bool,
}

#[derive(Debug, Args)]
pub struct ModelListArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    #[arg(long)]
    pub bank: String,
}

#[derive(Debug, Args)]
pub struct ModelEditArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    #[arg(long)]
    pub bank: String,

    /// The model's name.
    pub name: String,

    #[arg(long)]
    pub question: Option<String>,

    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// Only this kind. Repeat for several; replaces the model's kinds.
    #[arg(long = "kind", value_parser = serde_value::<Kind>, conflicts_with = "all_kinds")]
    pub kinds: Vec<Kind>,

    /// Take every kind.
    #[arg(long)]
    pub all_kinds: bool,

    /// hours, days, weeks, months or years, or `none` to clear it.
    #[arg(long)]
    pub min_volatility: Option<String>,

    #[arg(long, conflicts_with = "disable")]
    pub enable: bool,

    #[arg(long)]
    pub disable: bool,
}

#[derive(Debug, Args)]
pub struct ModelRefreshArgs {
    #[command(flatten)]
    pub client: ClientArgs,

    #[arg(long)]
    pub bank: String,

    /// The model's name.
    pub name: String,

    /// Refresh even when the model's inputs haven't changed.
    #[arg(long)]
    pub force: bool,
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

/// `asphodel replay`: a scripted scenario on a simulated clock
/// (`docs/replay.md`). Exit 0 when every probe passed, 1 when one failed (the
/// report is still written), 2 when the run was refused or failed.
#[derive(Debug, Args)]
pub struct ReplayArgs {
    /// The private directory holding the replayed store, the shadow table
    /// and the reports. Never inside a git working tree or a `serve` data
    /// dir.
    #[arg(long, env = "ASPHODEL_REPLAY_DIR")]
    pub replay_dir: Option<PathBuf>,

    /// The scenario file to run.
    #[arg(long, required_unless_present = "corpus", conflicts_with = "corpus")]
    pub scenario: Option<PathBuf>,

    /// The corpus `asphodel import` wrote, for a real-history run.
    #[arg(long)]
    pub corpus: Option<PathBuf>,

    /// Where real-history LLM replies come from.
    #[arg(long, value_enum, conflicts_with = "scenario")]
    pub mode: Option<ReplayMode>,

    /// The cassette of recorded LLM calls, under the private dir.
    #[arg(long, conflicts_with = "scenario")]
    pub cassette: Option<PathBuf>,

    /// Ignore every recorded call and record afresh (`live` only).
    #[arg(long, conflicts_with = "scenario")]
    pub no_cache: bool,

    /// How `fast` answers refreshes.
    #[arg(long, value_enum, conflicts_with = "scenario")]
    pub refresh: Option<RefreshMode>,

    /// The real-history probes file, under the private dir.
    #[arg(long, conflicts_with = "scenario")]
    pub probes: Option<PathBuf>,

    /// Run twice and compare the reports byte for byte.
    #[arg(long)]
    pub self_test: bool,

    /// Also write the aggregate export: probe ids and numbers only.
    #[arg(long)]
    pub aggregate: Option<PathBuf>,

    /// Also write the labelling material, under the private dir: recall
    /// candidates at 50 sampled turns and call 2's candidate lists.
    #[arg(long, conflicts_with = "scenario")]
    pub labelling: Option<PathBuf>,

    /// Where to write the JSON report; `<replay dir>/reports/<name>.json`
    /// by default. Given twice, the last one wins.
    #[arg(long, overrides_with = "report")]
    pub report: Option<PathBuf>,

    /// The production tuning file, layered over the code defaults.
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// A file in the shape of `Tuning`, layered over everything else.
    #[arg(long)]
    pub overrides: Option<PathBuf>,

    /// The simulated extraction latency, such as `10m`; overrides the
    /// scenario's own.
    #[arg(long)]
    pub latency: Option<String>,

    /// Keep running sweeps and refreshes past the last event until here.
    #[arg(long)]
    pub until: Option<jiff::Timestamp>,

    /// Where the real models are, for scenarios in group `models`.
    #[arg(long, env = "ASPHODEL_MODEL_DIR")]
    pub model_dir: Option<PathBuf>,

    /// ONNX Runtime's intra-op threads, pinned so a run is repeatable;
    /// 1 by default.
    #[arg(long, env = "ASPHODEL_ONNX_THREADS")]
    pub onnx_threads: Option<NonZeroUsize>,

    /// Where the ChatGPT login lives, for `live` and `fast` with
    /// `llm.auth = "chatgpt"`; the private dir by default.
    #[arg(long)]
    pub token_dir: Option<PathBuf>,
}

/// Where a real-history replay's LLM replies come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ReplayMode {
    /// Use the cassette, and call and record on a miss.
    Live,
    /// Use the cassette, and fail on a miss.
    Replay,
    /// Reuse claims by chunk and `used` verdicts by pair, topping up the
    /// rest.
    Fast,
}

/// How `fast` answers refreshes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum RefreshMode {
    Live,
    Recorded,
    Off,
}

/// `asphodel import`: a copy of Hermes' `state.db` and the private manifest
/// into a replay corpus.
#[derive(Debug, Args)]
pub struct ImportArgs {
    /// The private directory the corpus goes under.
    #[arg(long, env = "ASPHODEL_REPLAY_DIR")]
    pub replay_dir: Option<PathBuf>,

    /// A copy of Hermes' `state.db`.
    #[arg(long)]
    pub state_db: PathBuf,

    /// The private manifest: timezone, owner, assistant, speakers and the
    /// mental models to replay with.
    #[arg(long)]
    pub manifest: PathBuf,

    /// Where to write the corpus, under the private dir.
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Check the history and print the counts, writing nothing.
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Subcommand)]
pub enum ReportCommand {
    /// The A/B diff of two replay reports.
    Diff(DiffArgs),

    /// A replay report as one self-contained HTML page.
    Html(HtmlArgs),

    /// The precision curve of the labelled labelling material.
    Precision(PrecisionArgs),
}

/// `asphodel report html`: the page goes beside the report unless `--out` says
/// otherwise; both stay under the private dir.
#[derive(Debug, Args)]
pub struct HtmlArgs {
    /// The private directory the report and the page are in.
    #[arg(long, env = "ASPHODEL_REPLAY_DIR")]
    pub replay_dir: Option<PathBuf>,

    /// The JSON report.
    pub report: PathBuf,

    /// Where to write the page; the report's path with the extension
    /// `html` by default.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

/// `asphodel report precision`: the curve on stdout.
#[derive(Debug, Args)]
pub struct PrecisionArgs {
    /// The private directory the labels and the material are in.
    #[arg(long, env = "ASPHODEL_REPLAY_DIR")]
    pub replay_dir: Option<PathBuf>,

    /// The labels: a TOML table of candidate id to `true` or `false`.
    #[arg(long)]
    pub labels: PathBuf,

    /// The material `asphodel replay --labelling` wrote.
    #[arg(long)]
    pub material: PathBuf,
}

#[derive(Debug, Args)]
pub struct DiffArgs {
    pub a: PathBuf,
    pub b: PathBuf,

    /// Compare runs with a different corpus or cassette.
    #[arg(long)]
    pub force: bool,
}

/// `asphodel bench`: concurrent prefetches over HTTP against a daemon started
/// on a copy of the replayed store.
#[derive(Debug, Args)]
pub struct BenchArgs {
    /// The private directory whose replayed store is copied.
    #[arg(long, env = "ASPHODEL_REPLAY_DIR")]
    pub replay_dir: Option<PathBuf>,

    /// The corpus the prefetch queries are sampled from.
    #[arg(long)]
    pub corpus: Option<PathBuf>,

    /// A concurrency level to measure; given more than once, each is.
    #[arg(long)]
    pub concurrency: Vec<NonZeroUsize>,

    /// Prefetches per concurrency level.
    #[arg(long)]
    pub requests: Option<NonZeroUsize>,

    /// Where the bench daemon listens. Loopback only.
    #[arg(long)]
    pub listen: Option<Listen>,

    /// Where to write the JSON report, under the private dir.
    #[arg(long)]
    pub report: Option<PathBuf>,
}

impl Cli {
    pub fn run(self) -> anyhow::Result<()> {
        match self.command {
            Command::Serve(args) => serve(args),
            Command::Ingest(args) => ingest(args),
            Command::Bank(BankCommand::Create {
                client,
                name,
                identity,
            }) => bank(&client, &name, identity),
            Command::Bank(BankCommand::Config {
                client,
                name,
                identity,
            }) => bank(&client, &name, identity),
            Command::Bank(BankCommand::Delete {
                client,
                name,
                confirm,
            }) => bank_delete(&client, &name, &confirm),
            Command::Memory(command) => memory(command),
            Command::Entity(command) => entity(command),
            Command::Reembed(args) => reembed(args),
            Command::Chunks(args) => chunks(args),
            Command::Recall(args) => recall(args),
            Command::Forget(args) => by_ids(args, IdsAction::Forget),
            Command::Keep(args) => by_ids(args, IdsAction::Keep),
            Command::Unkeep(args) => by_ids(args, IdsAction::Unkeep),
            Command::Model(command) => model(command),
            Command::Purge(command) => purge(command),
            Command::Backup(args) => backup(args),
            Command::Restore(args) => restore(args),
            Command::Status(args) => status(args),
            Command::Purges(args) => audit(args, AuditList::Purges),
            Command::Forgets(args) => audit(args, AuditList::Forgets),
            Command::Sweeps(args) => audit(args, AuditList::Sweeps),
            Command::Recalls(args) => audit(args, AuditList::Recalls),
            Command::Models(ModelsCommand::Fetch { model_dir }) => models_fetch(model_dir),
            Command::Llm(LlmCommand::Login { data_dir }) => llm_login(&data_dir),
            Command::Import(args) => crate::replay::import::run(args),
            Command::Replay(args) => crate::replay::run(args),
            Command::Report(ReportCommand::Diff(args)) => crate::replay::diff::run(args),
            Command::Report(ReportCommand::Html(args)) => crate::replay::html::run(args),
            Command::Report(ReportCommand::Precision(args)) => crate::replay::labelling::run(args),
            Command::Bench(args) => crate::bench::run(args),
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
/// files already present with the right checksum.
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
pub(crate) const LLM_ISSUER_ENV: &str = "ASPHODEL_LLM_ISSUER";

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

/// Prints the daemon's reply as JSON, for `--json`.
fn print_json(value: &Value) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

fn count(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn list<'a>(value: &'a Value, key: &str) -> &'a [Value] {
    value
        .get(key)
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// `asphodel ingest`: sends a document to `POST /v1/banks/{bank}/documents`.
fn ingest(args: IngestArgs) -> anyhow::Result<()> {
    let client = Client::new(&args.client)?;
    let contents = std::fs::read_to_string(&args.file)
        .with_context(|| format!("reading {}", args.file.display()))?;
    let document_id = match args.id {
        Some(id) => id,
        None => args
            .file
            .file_name()
            .and_then(|name| name.to_str())
            .map(str::to_string)
            .with_context(|| {
                format!(
                    "{} has no file name to use as the document id: pass --id",
                    args.file.display()
                )
            })?,
    };
    let document = Document {
        document_id: document_id.clone(),
        text: contents,
        reference_date: args.date,
        reference_date_exact: !args.inexact,
        timezone: None,
    };
    let ingested: Value = client.post(
        &format!("/v1/banks/{}/documents", segment(&args.bank)),
        &document,
    )?;
    if args.client.json {
        return print_json(&ingested);
    }
    match text(&ingested, "outcome") {
        "duplicate" => println!(
            "{document_id} is already in {} unchanged (source {})",
            args.bank,
            text(&ingested, "source")
        ),
        _ => println!(
            "ingested {document_id} into {} as source {}: {} chunks queued, {} already seen",
            args.bank,
            text(&ingested, "source"),
            count(&ingested, "chunks_queued"),
            count(&ingested, "chunks_skipped"),
        ),
    }
    let secrets = list(&ingested, "secret_kinds");
    if !secrets.is_empty() {
        let kinds: Vec<&str> = secrets.iter().filter_map(Value::as_str).collect();
        println!("redacted before storing: {}", kinds.join(", "));
    }
    Ok(())
}

/// `asphodel bank create|config`: `PUT /v1/banks/{bank}`, which creates
/// the bank or merges the given fields into it.
fn bank(client_args: &ClientArgs, name: &str, identity: IdentityArgs) -> anyhow::Result<()> {
    let client = Client::new(client_args)?;
    let bank: Value = client.put(
        &format!("/v1/banks/{}", segment(name)),
        &identity.identity(),
    )?;
    if client_args.json {
        return print_json(&bank);
    }
    let created = bank.get("created").and_then(Value::as_bool) == Some(true);
    println!(
        "{} bank {}",
        if created { "created" } else { "updated" },
        text(&bank, "name")
    );
    for (label, key) in [
        ("owner", "owner_name"),
        ("assistant", "assistant_name"),
        ("timezone", "timezone"),
        ("embedding model", "embedding_model"),
        ("reranker model", "reranker_model"),
    ] {
        let value = text(&bank, key);
        if !value.is_empty() {
            println!("  {label}: {value}");
        }
    }
    Ok(())
}

/// `asphodel chunks`: the bank's queue and failed chunks, and with
/// `--failed --retry` puts the failed chunks it listed back on the queue.
fn chunks(args: ChunksArgs) -> anyhow::Result<()> {
    let client = Client::new(&args.client)?;
    let bank = segment(&args.bank);
    let path = if args.failed {
        format!("/v1/banks/{bank}/chunks?failed=true")
    } else {
        format!("/v1/banks/{bank}/chunks")
    };
    let listed: Value = client.get(&path)?;
    let retried: Option<Value> = if args.retry {
        // Only the chunks just listed: one that fails between the list
        // and the retry waits for the next run, rather than being retried
        // unseen.
        let chunks = list(&listed, "failed")
            .iter()
            .filter_map(|chunk| chunk.get("chunk")?.as_str()?.parse().ok())
            .collect();
        let request = RetryRequest {
            chunks: Some(chunks),
        };
        Some(client.post(&format!("/v1/banks/{bank}/chunks/retry"), &request)?)
    } else {
        None
    };
    if args.client.json {
        return print_json(&match retried {
            Some(retried) => serde_json::json!({ "listed": listed, "retried": retried }),
            None => listed,
        });
    }
    if !args.failed {
        let queued = list(&listed, "queued");
        println!("{} queued", queued.len());
        for chunk in queued {
            println!(
                "  {}  {} {} #{}  {} errors{}",
                text(chunk, "chunk"),
                text(chunk, "source_kind"),
                text(chunk, "source"),
                count(chunk, "position"),
                count(chunk, "error_count"),
                if chunk.get("in_flight").and_then(Value::as_bool) == Some(true) {
                    "  in flight"
                } else {
                    ""
                },
            );
        }
    }
    let failed = list(&listed, "failed");
    println!("{} failed", failed.len());
    for chunk in failed {
        let status = chunk
            .get("status")
            .and_then(Value::as_u64)
            .map(|status| format!(" (HTTP {status})"))
            .unwrap_or_default();
        println!(
            "  {}  source {}  {} errors, last {}{status}, failed {}",
            text(chunk, "chunk"),
            text(chunk, "source"),
            count(chunk, "error_count"),
            text(chunk, "error_kind"),
            text(chunk, "failed_at"),
        );
    }
    if let Some(retried) = retried {
        println!("{} put back on the queue", list(&retried, "retried").len());
        for chunk in list(&retried, "unknown") {
            println!(
                "  {} is no longer a failed chunk",
                chunk.as_str().unwrap_or("")
            );
        }
    }
    Ok(())
}

/// `asphodel recall`: `POST /v1/banks/{bank}/recall`, outside any session.
fn recall(args: RecallArgs) -> anyhow::Result<()> {
    let client = Client::new(&args.client)?;
    let request = RecallRequest {
        session_id: None,
        query: args.query,
        from: args.from,
        to: args.to,
        on: args.on,
        phase: args.phase,
        kinds: args.kinds,
        entity: args.entity,
        limit: args.limit,
    };
    let recall: Value = client.post(
        &format!("/v1/banks/{}/recall", segment(&args.bank)),
        &request,
    )?;
    if args.client.json {
        return print_json(&recall);
    }
    let results = list(&recall, "results");
    if results.is_empty() {
        println!("nothing recalled");
    }
    for result in results {
        let kept = if result.get("kept").and_then(Value::as_bool) == Some(true) {
            ", kept"
        } else {
            ""
        };
        println!(
            "{}  [{}, {}, {}{kept}]  {}",
            text(result, "id"),
            text(result, "kind"),
            text(result, "phase"),
            text(result, "strength"),
            text(result, "sentence"),
        );
    }
    if recall.get("reranked").and_then(Value::as_bool) == Some(false) {
        println!("(the reranker missed its deadline, so these are in fused order)");
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum IdsAction {
    Keep,
    Unkeep,
    Forget,
}

/// `asphodel keep|unkeep|forget`: `POST /v1/banks/{bank}/keep|unkeep|forget`.
/// Forget lists every version it erases. Ids the bank doesn't have make it
/// exit non-zero, after the rest are done.
fn by_ids(args: IdsArgs, action: IdsAction) -> anyhow::Result<()> {
    let client = Client::new(&args.client)?;
    let (route, done_key, verb) = match action {
        IdsAction::Keep => ("keep", "kept", "kept"),
        IdsAction::Unkeep => ("unkeep", "unkept", "unkept"),
        IdsAction::Forget => ("forget", "forgotten", "forgot"),
    };
    let reply: Value = client.post(
        &format!("/v1/banks/{}/{route}", segment(&args.bank)),
        &MemoryIds { ids: args.ids },
    )?;
    if args.client.json {
        print_json(&reply)?;
    } else {
        for id in list(&reply, done_key) {
            println!("{verb} {}", id.as_str().unwrap_or(""));
        }
    }
    let unknown: Vec<&str> = list(&reply, "unknown")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    if !unknown.is_empty() {
        bail!("no such memory in {}: {}", args.bank, unknown.join(", "));
    }
    Ok(())
}

/// `asphodel model create|list|edit|refresh` over
/// `/v1/banks/{bank}/models`.
fn model(command: ModelCommand) -> anyhow::Result<()> {
    match command {
        ModelCommand::Create(args) => {
            let client = Client::new(&args.client)?;
            let spec = ModelSpec {
                name: args.name,
                question: args.question,
                kinds: args.kinds,
                entity: args.entity,
                min_volatility: args.min_volatility,
                max_tokens: args.max_tokens,
                enabled: !args.disabled,
            };
            let model: Value =
                client.post(&format!("/v1/banks/{}/models", segment(&args.bank)), &spec)?;
            if args.client.json {
                return print_json(&model);
            }
            println!(
                "created {} ({} tokens); it's refreshed in a few minutes, or now with `asphodel model refresh`",
                text(&model, "name"),
                model.get("max_tokens").and_then(Value::as_u64).unwrap_or(0)
            );
            Ok(())
        }
        ModelCommand::List(args) => {
            let client = Client::new(&args.client)?;
            let models: Value = client.get(&format!("/v1/banks/{}/models", segment(&args.bank)))?;
            if args.client.json {
                return print_json(&models);
            }
            let models = models.as_array().cloned().unwrap_or_default();
            if models.is_empty() {
                println!("no models");
            }
            for model in &models {
                print_model(model);
            }
            Ok(())
        }
        ModelCommand::Edit(args) => {
            let client = Client::new(&args.client)?;
            let min_volatility = match args.min_volatility.as_deref() {
                None => None,
                Some("none") => Some(None),
                Some(level) => Some(Some(
                    serde_value::<Volatility>(level).map_err(anyhow::Error::msg)?,
                )),
            };
            let edit = ModelEdit {
                question: args.question,
                kinds: if args.all_kinds {
                    Some(Vec::new())
                } else if args.kinds.is_empty() {
                    None
                } else {
                    Some(args.kinds)
                },
                min_volatility,
                max_tokens: args.max_tokens,
                enabled: if args.enable {
                    Some(true)
                } else if args.disable {
                    Some(false)
                } else {
                    None
                },
            };
            let model: Value = client.patch(
                &format!(
                    "/v1/banks/{}/models/{}",
                    segment(&args.bank),
                    segment(&args.name)
                ),
                &edit,
            )?;
            if args.client.json {
                return print_json(&model);
            }
            print_model(&model);
            Ok(())
        }
        ModelCommand::Refresh(args) => {
            let client = Client::new(&args.client)?;
            let outcome: Value = client.post(
                &format!(
                    "/v1/banks/{}/models/{}/refresh?force={}",
                    segment(&args.bank),
                    segment(&args.name),
                    args.force
                ),
                &Value::Null,
            )?;
            if args.client.json {
                return print_json(&outcome);
            }
            let detail = &outcome["detail"];
            match outcome.get("outcome").and_then(Value::as_str) {
                Some("unchanged") => {
                    println!("unchanged: its inputs are the same as at the last refresh")
                }
                Some("applied") => {
                    let count = |key: &str| list(detail, key).len();
                    println!(
                        "refreshed: {} added, {} edited, {} removed, {} rejected, {} dropped, {} trimmed",
                        count("added"),
                        count("edited"),
                        count("removed"),
                        count("rejected"),
                        count("dropped"),
                        count("trimmed"),
                    );
                }
                Some("held") => println!(
                    "held: the refresh waits until {}",
                    detail["until"].as_str().unwrap_or("unknown")
                ),
                _ => bail!(
                    "the refresh failed ({}); it's tried again in 30 minutes",
                    detail.as_str().unwrap_or("unknown")
                ),
            }
            Ok(())
        }
        ModelCommand::Show(args) => {
            let client = Client::new(&args.client)?;
            let mut path = format!(
                "/v1/banks/{}/models/{}",
                segment(&args.bank),
                segment(&args.name)
            );
            if let Some(entry) = &args.entry {
                path.push_str(&format!("?entry={}", segment(entry)));
            }
            let view: Value = client.get(&path)?;
            if args.client.json {
                return print_json(&view);
            }
            print_model_view(&view);
            Ok(())
        }
    }
}

fn print_model_view(view: &Value) {
    let enabled = if view.get("enabled").and_then(Value::as_bool) == Some(false) {
        ", disabled"
    } else {
        ""
    };
    println!(
        "{}  {}  [{} tokens{enabled}]",
        text(view, "name"),
        text(view, "id"),
        count(view, "max_tokens"),
    );
    println!("  question: {}", text(view, "question"));
    let kinds: Vec<&str> = list(view, "kinds")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    if !kinds.is_empty() {
        println!("  kinds: {}", kinds.join(", "));
    }
    if let Some(entity) = view.get("entity").and_then(Value::as_str) {
        println!("  entity: {entity} ({})", text(view, "entity_id"));
    }
    if let Some(level) = view.get("min_volatility").and_then(Value::as_str) {
        println!("  min volatility: {level}");
    }
    match view.get("last_refreshed_at").and_then(Value::as_str) {
        Some(at) => println!("  last refreshed {at}"),
        None => println!("  never refreshed"),
    }
    if let Some(at) = view.get("refresh_requested_at").and_then(Value::as_str) {
        println!("  refresh requested {at}");
    }
    if let Some(error) = view.get("last_error").and_then(Value::as_str) {
        println!(
            "  the last refresh failed ({error}) at {}",
            text(view, "last_error_at")
        );
    }
    for entry in list(view, "entry_views") {
        let renders = if entry.get("renders").and_then(Value::as_bool) == Some(true) {
            ""
        } else {
            "  (not shown: a memory it cites isn't current)"
        };
        println!(
            "  - {}  {}{renders}",
            text(entry, "id"),
            text(entry, "text")
        );
        for cite in list(entry, "cites") {
            println!(
                "      cites {} [{}]  {}",
                text(cite, "id"),
                text(cite, "status"),
                text(cite, "sentence")
            );
        }
    }
}

fn print_model(model: &Value) {
    let enabled = if model.get("enabled").and_then(Value::as_bool) == Some(false) {
        ", disabled"
    } else {
        ""
    };
    println!(
        "{}  [{} tokens{enabled}]  {}",
        text(model, "name"),
        model.get("max_tokens").and_then(Value::as_u64).unwrap_or(0),
        text(model, "question"),
    );
    match model.get("last_refreshed_at").and_then(Value::as_str) {
        Some(at) => println!("  last refreshed {at}"),
        None => println!("  never refreshed"),
    }
    if let Some(error) = model.get("last_error").and_then(Value::as_str) {
        println!(
            "  the last refresh failed ({error}) at {}",
            text(model, "last_error_at")
        );
    }
    for entry in list(model, "entries") {
        let cites: Vec<&str> = list(entry, "cites")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        println!("  - {}  (cites {})", text(entry, "text"), cites.join(", "));
    }
}

/// `asphodel purge plan|ack` over `/v1/purge`.
fn purge(command: PurgeCommand) -> anyhow::Result<()> {
    match command {
        PurgeCommand::Plan { client: args } => {
            let client = Client::new(&args)?;
            let plan: Value = client.get("/v1/purge/plan")?;
            if args.json {
                return print_json(&plan);
            }
            let state = plan
                .get("pause")
                .and_then(|pause| pause.get("state"))
                .and_then(Value::as_str)
                .unwrap_or("");
            println!("purge: {state}");
            println!("current fingerprint: {}", text(&plan, "current"));
            let changed: Vec<&str> = list(&plan, "changed")
                .iter()
                .filter_map(Value::as_str)
                .collect();
            if !changed.is_empty() {
                println!("changed: {}", changed.join(", "));
            }
            println!(
                "the sweep would delete now: {} memories, the text of {} sources and {} failed chunks, and {} recall queries",
                count(&plan, "memories"),
                count(&plan, "sources"),
                count(&plan, "failed_chunks"),
                count(&plan, "recalls"),
            );
            Ok(())
        }
        PurgeCommand::Ack { client: args, hash } => {
            let client = Client::new(&args)?;
            let _: Value = client.post("/v1/purge/ack", &PurgeAck { hash })?;
            println!("acknowledged; purging resumes at the next sweep");
            Ok(())
        }
    }
}

/// `asphodel backup --out <file|->`: streams `POST /v1/backup` and checks
/// its length and SHA-256 against the headers. A file is written under a
/// temporary name beside it, integrity-checked, and renamed into place only
/// when everything passes, so a failed backup leaves nothing at `--out`.
/// Stdout can't be taken back, so there a mismatch only fails the command,
/// which a pipe into storage sees as the exit code.
fn backup(args: BackupArgs) -> anyhow::Result<()> {
    let client = Client::new(&args.client)?;
    if args.out.as_os_str() == "-" {
        let mut stdout = std::io::stdout().lock();
        let mut received = Received::default();
        let headers = client.download("/v1/backup", &mut |bytes| {
            received.add(bytes);
            stdout
                .write_all(bytes)
                .context("writing the backup to stdout")
        })?;
        stdout.flush().context("writing the backup to stdout")?;
        received.verify(&headers)?;
        return Ok(());
    }

    let (temp, mut file) = temp_file_beside(&args.out)
        .with_context(|| format!("creating a file beside {}", args.out.display()))?;
    let written = (|| {
        let mut received = Received::default();
        let headers = client.download("/v1/backup", &mut |bytes| {
            received.add(bytes);
            file.write_all(bytes)
                .with_context(|| format!("writing {}", temp.display()))
        })?;
        file.sync_all()
            .with_context(|| format!("writing {}", temp.display()))?;
        let summary = received.verify(&headers)?;
        check_copy(&temp)
            .map_err(|detail| anyhow::anyhow!("the backup failed its integrity check: {detail}"))?;
        std::fs::rename(&temp, &args.out)
            .with_context(|| format!("renaming the backup to {}", args.out.display()))?;
        Ok::<_, anyhow::Error>(summary)
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    let summary = written?;
    if args.client.json {
        return print_json(&serde_json::json!({
            "out": args.out,
            "length": summary.length,
            "sha256": summary.sha256,
            "backed_up_at": summary.backed_up_at,
        }));
    }
    println!(
        "wrote {}: {} bytes, sha256 {}{}",
        args.out.display(),
        summary.length,
        summary.sha256,
        summary
            .backed_up_at
            .map(|at| format!(", taken {at}"))
            .unwrap_or_default(),
    );
    Ok(())
}

/// What a backup stream delivered, hashed as it arrives.
#[derive(Default)]
struct Received {
    hasher: Sha256,
    length: u64,
}

/// A backup that matched its headers.
struct Verified {
    length: u64,
    sha256: String,
    backed_up_at: Option<String>,
}

impl Received {
    fn add(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
        self.length += bytes.len() as u64;
    }

    /// Checks the length and hash against the daemon's headers.
    fn verify(self, headers: &HeaderMap) -> anyhow::Result<Verified> {
        let header = |name: &str| {
            headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::trim)
                .with_context(|| format!("the daemon sent no {name} header"))
        };
        let expected: u64 = header(LENGTH_HEADER)?
            .parse()
            .with_context(|| format!("the {LENGTH_HEADER} header isn't a length"))?;
        let expected_hash = header(SHA256_HEADER)?.to_ascii_lowercase();
        if self.length != expected {
            bail!(
                "the backup stream was cut short: {} of {expected} bytes arrived",
                self.length
            );
        }
        let sha256: String = self
            .hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        if sha256 != expected_hash {
            bail!("the backup's SHA-256 is {sha256}, but the daemon sent {expected_hash}");
        }
        Ok(Verified {
            length: self.length,
            sha256,
            backed_up_at: header(BACKED_UP_AT_HEADER).ok().map(str::to_string),
        })
    }
}

/// `asphodel restore <file> --data-dir <dir>`: offline, so it refuses
/// while a daemon holds the lock. Purge pauses at the next start if the
/// restored store's deletion fingerprint differs from this binary's.
fn restore(args: RestoreArgs) -> anyhow::Result<()> {
    let options = OpenOptions {
        allow_network_fs: args.allow_network_fs,
        deterministic_ids: false,
    };
    let restored =
        asphodel_core::operations::restore(&args.file, &args.data_dir, options, &SystemClock)
            .with_context(|| {
                format!(
                    "restoring {} into {}",
                    args.file.display(),
                    args.data_dir.display()
                )
            })?;
    println!(
        "restored {} into {} (schema version {}, taken {})",
        args.file.display(),
        args.data_dir.display(),
        restored.schema_version,
        restored
            .backed_up_at
            .map_or_else(|| "at an unknown time".to_string(), |at| at.to_string()),
    );
    if let Some(aside) = &restored.moved_aside {
        println!("the database it replaced is at {}", aside.display());
    }
    if restored.schema_version < asphodel_core::store::SCHEMA_VERSION {
        println!(
            "the daemon migrates it from schema version {} when it starts",
            restored.schema_version
        );
    }
    Ok(())
}

/// `asphodel status`: `GET /v1/status`, printed, and a non-zero exit when
/// anything needs attention, with `--json` too.
fn status(args: StatusArgs) -> anyhow::Result<()> {
    let client = Client::new(&args.client)?;
    let status: Value = client.get("/v1/status")?;
    let attention: Vec<&str> = list(&status, "attention")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    if args.client.json {
        print_json(&status)?;
    } else {
        print_status(&status, &attention);
    }
    if !attention.is_empty() {
        bail!("{} things need attention", attention.len());
    }
    Ok(())
}

fn print_status(status: &Value, attention: &[&str]) {
    if attention.is_empty() {
        println!(
            "asphodel {}: nothing needs attention",
            text(status, "version")
        );
    } else {
        println!(
            "asphodel {}: {} things need attention",
            text(status, "version"),
            attention.len()
        );
        for item in attention {
            println!("  ! {item}");
        }
    }
    let current = text(status, "deletion_fingerprint");
    let purge = &status["purge"];
    match text(purge, "state") {
        "paused" => println!(
            "purge: paused; stored fingerprint {}, running daemon's {current}",
            text(purge, "stored")
        ),
        state => println!("purge: {state}; deletion fingerprint {current}"),
    }
    if let Some(banks) = status["banks"].as_object() {
        for (name, bank) in banks {
            println!(
                "bank {name}: {} queued, {} failed chunks, {} failed refreshes",
                count(bank, "queued"),
                count(bank, "failed_chunks"),
                count(bank, "failed_refreshes"),
            );
        }
    }
    match &status["last_sweep"] {
        Value::Null => println!("last sweep: never"),
        sweep => println!(
            "last sweep: {} (bank {}): {} memories purged; the text of {} sources, {} chunks and {} failed chunks swept, and {} recall queries",
            text(sweep, "completed_at"),
            text(sweep, "bank"),
            count(sweep, "purged_memories"),
            count(sweep, "swept_sources"),
            count(sweep, "swept_chunks"),
            count(sweep, "swept_failed_chunks"),
            count(sweep, "swept_recalls"),
        ),
    }
    match &status["pre_migration_copy"] {
        Value::Null => println!("pre-migration copy: none"),
        copy => println!(
            "pre-migration copy: from schema version {} at {}, deleted after {}",
            count(copy, "from_version"),
            text(copy, "path"),
            text(copy, "expires_at"),
        ),
    }
    match status["last_backup_at"].as_str() {
        Some(at) => println!("last backup: {at}"),
        None => println!("last backup: never"),
    }
}

/// `asphodel purges|forgets|sweeps|recalls --bank <bank>`.
fn audit(args: ListArgs, kind: AuditList) -> anyhow::Result<()> {
    let client = Client::new(&args.client)?;
    let name = kind.as_str();
    let mut path = format!("/v1/banks/{}/{name}", segment(&args.bank));
    if let Some(limit) = args.limit {
        path.push_str(&format!("?limit={limit}"));
    }
    let reply: Value = client.get(&path)?;
    if args.client.json {
        return print_json(&reply);
    }
    let rows = list(&reply, name);
    if rows.is_empty() {
        println!("no {name} in {}", args.bank);
    }
    let ids = |row: &Value, key: &str| {
        list(row, key)
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(", ")
    };
    for row in rows {
        match kind {
            AuditList::Purges => println!(
                "{}  purge {}: {} memories: {}",
                text(row, "at"),
                text(row, "edit"),
                list(row, "memories").len(),
                ids(row, "memories"),
            ),
            AuditList::Forgets => println!(
                "{}  forget {}: {}{}",
                text(row, "at"),
                text(row, "edit"),
                ids(row, "memories"),
                if row["pending"].as_bool() == Some(true) {
                    "  (erase pending)"
                } else {
                    ""
                },
            ),
            AuditList::Sweeps => println!(
                "{}  sweep: {} memories purged; the text of {} sources, {} chunks and {} failed chunks swept, and {} recall queries (fingerprint {}, delta {})",
                text(row, "completed_at"),
                count(row, "purged_memories"),
                count(row, "swept_sources"),
                count(row, "swept_chunks"),
                count(row, "swept_failed_chunks"),
                count(row, "swept_recalls"),
                text(row, "fingerprint"),
                row["delta"]
                    .as_f64()
                    .map_or_else(|| "none".to_string(), |delta| delta.to_string()),
            ),
            AuditList::Recalls => println!(
                "{}  {} {} ({} ms, {} results): {}",
                text(row, "at"),
                text(row, "kind"),
                text(row, "id"),
                count(row, "latency_ms"),
                list(row, "results").len(),
                row["query"].as_str().unwrap_or("[swept]"),
            ),
        }
    }
    Ok(())
}

/// A number from a view, or −∞ for the `null` JSON makes of it.
fn number(value: &Value, key: &str) -> String {
    match value.get(key).and_then(Value::as_f64) {
        Some(number) => format!("{number:.3}"),
        None => "−∞".to_string(),
    }
}

/// `asphodel bank delete <bank> --confirm <bank>`: `DELETE /v1/banks/{bank}`.
fn bank_delete(client_args: &ClientArgs, name: &str, confirm: &str) -> anyhow::Result<()> {
    if name.trim() != confirm.trim() {
        bail!("--confirm must repeat the bank's name");
    }
    let client = Client::new(client_args)?;
    let deleted: Value = client.delete(&format!(
        "/v1/banks/{}?confirm={}",
        segment(name),
        segment(confirm)
    ))?;
    if client_args.json {
        return print_json(&deleted);
    }
    println!(
        "deleted bank {} ({}): {} memories, {} entities, {} sources, {} chunks, {} edits, {} models, {} recalls, {} sessions",
        text(&deleted, "name"),
        text(&deleted, "bank"),
        count(&deleted, "memories"),
        count(&deleted, "entities"),
        count(&deleted, "sources"),
        count(&deleted, "chunks"),
        count(&deleted, "edits"),
        count(&deleted, "models"),
        count(&deleted, "recalls"),
        count(&deleted, "sessions"),
    );
    println!("a running plugin creates it again, empty, on its next `initialize`");
    Ok(())
}

/// `asphodel memory show|significance`.
fn memory(command: MemoryCommand) -> anyhow::Result<()> {
    match command {
        MemoryCommand::Show { client, bank, id } => {
            let json = client.json;
            let client = Client::new(&client)?;
            let view: Value = client.get(&format!(
                "/v1/banks/{}/memories/{}",
                segment(&bank),
                segment(&id)
            ))?;
            if json {
                return print_json(&view);
            }
            print_memory(&view);
            Ok(())
        }
        MemoryCommand::Significance {
            client,
            bank,
            id,
            level,
        } => {
            let json = client.json;
            let client = Client::new(&client)?;
            let level = (level.trim() != "clear").then(|| level.trim().to_string());
            let set: Value = client.put(
                &format!(
                    "/v1/banks/{}/memories/{}/significance",
                    segment(&bank),
                    segment(&id)
                ),
                &SignificanceRequest { level },
            )?;
            if json {
                return print_json(&set);
            }
            let shown = |key: &str| {
                set.get(key)
                    .and_then(Value::as_str)
                    .unwrap_or("none")
                    .to_string()
            };
            println!(
                "{}: owner's significance {} -> {} (extracted {})",
                text(&set, "memory"),
                shown("from"),
                shown("to"),
                text(&set, "extracted"),
            );
            Ok(())
        }
    }
}

fn print_memory(view: &Value) {
    let phase = view.get("phase").and_then(Value::as_str).unwrap_or("-");
    println!(
        "{}  [{}, {}]  {}",
        text(view, "id"),
        text(view, "kind"),
        phase,
        text(view, "sentence")
    );
    let significance = &view["significance"];
    println!(
        "  significance: extracted {}, owner {} (strength uses {}, {})",
        text(significance, "extracted"),
        significance
            .get("owner")
            .and_then(Value::as_str)
            .unwrap_or("none"),
        text(significance, "effective"),
        number(significance, "value"),
    );
    let window = &view["window"];
    let time = |key: &str| {
        window
            .get(key)
            .filter(|value| !value.is_null())
            .map(|value| format!("{} ({})", text(value, "at"), text(value, "precision")))
    };
    let mut parts = Vec::new();
    if let Some(from) = time("valid_from") {
        parts.push(format!("from {from}"));
    }
    if let Some(until) = time("valid_until") {
        parts.push(format!("until {until}"));
    }
    if let Some(event) = window.get("until_event").and_then(Value::as_str) {
        parts.push(format!("until \"{event}\""));
    }
    if let Some(due) = time("due_at") {
        parts.push(format!("due {due}"));
    }
    if let Some(volatility) = window.get("volatility").and_then(Value::as_str) {
        parts.push(format!("volatility {volatility}"));
    }
    if let Some(recurrence) = window.get("recurrence").and_then(Value::as_str) {
        parts.push(format!("recurs \"{recurrence}\""));
    }
    parts.push(format!(
        "{} confidence, {}",
        text(window, "window_confidence"),
        text(window, "timezone")
    ));
    println!("  window: {}", parts.join("; "));
    let mut times = vec![
        format!("observed {}", text(view, "observed_at")),
        format!("created {}", text(view, "created_at")),
    ];
    if let Some(at) = view.get("hidden_at").and_then(Value::as_str) {
        times.push(format!("forgotten {at}, erase pending"));
    }
    if let Some(at) = view.get("retracted_at").and_then(Value::as_str) {
        times.push(format!("retracted {at}"));
    }
    println!("  {}", times.join(", "));

    let source = &view["source"];
    println!(
        "  source: {} {}, chunk {} [{}..{}]",
        text(source, "kind"),
        text(source, "source"),
        text(source, "chunk"),
        count(source, "start"),
        count(source, "end"),
    );
    match source.get("passage").and_then(Value::as_str) {
        Some(passage) => println!("    \"{passage}\""),
        None => {
            let gone = &source["gone"];
            match text(gone, "reason") {
                "swept" => println!(
                    "    passage gone: swept at its 90-day horizon{}",
                    gone.get("at")
                        .and_then(Value::as_str)
                        .map(|at| format!(" ({at})"))
                        .unwrap_or_default()
                ),
                "redacted" => println!("    passage gone: redacted by a forget"),
                "forget_requested" => {
                    println!("    passage gone: the turn asked to forget and was never stored")
                }
                other => println!("    passage gone: {other}"),
            }
        }
    }
    let secrets: Vec<&str> = list(source, "secret_kinds")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    if !secrets.is_empty() {
        println!("    secret scan redacted: {}", secrets.join(", "));
    }

    let strength = &view["strength"];
    let recallable = if strength.get("recallable").and_then(Value::as_bool) == Some(true) {
        "recallable"
    } else {
        "below τ, so recall leaves it out"
    };
    println!(
        "  strength {} = significance {} + max(recent use {}, lasting floor {} over {} occasions); τ {}: {recallable}",
        number(strength, "value"),
        number(strength, "significance_boost"),
        number(strength, "recent_use"),
        number(strength, "lasting_floor"),
        count(strength, "occasions"),
        number(strength, "threshold"),
    );
    let purge = &view["purge"];
    let line = match purge.get("line").and_then(Value::as_f64) {
        Some(_) => format!("purge line τ−δ {}", number(purge, "line")),
        None => "purge is off (δ unset)".to_string(),
    };
    println!(
        "  purge: read on head {} at {}; {line}",
        text(purge, "head"),
        number(purge, "head_strength"),
    );
    let guards = list(purge, "guards");
    if guards.is_empty() {
        println!("    nothing holds it back: the next sweep purges the chain");
    }
    for guard in guards {
        match text(guard, "guard") {
            "purge_disabled" => println!("    held: δ is unset, so nothing is purged"),
            "purge_paused" => {
                println!("    held: purge is paused until `asphodel purge ack`")
            }
            "forgotten" => println!("    held: forgotten; its erase removes it"),
            "date_ahead" => println!("    held: a date ahead, until {}", text(guard, "until")),
            "overdue_task" => println!("    held: an overdue task, until {}", text(guard, "until")),
            "strength" => println!("    held: the head is above the purge line"),
            other => println!("    held: {other}"),
        }
    }
    let projection = &view["projection"];
    println!("  projected, {}:", text(projection, "basis"));
    let projected = |label: &str, key: &str, never: &str| match projection.get(key) {
        Some(when) if !when.is_null() => println!(
            "    {label} in {} bank days; earliest {} (at full speed)",
            number(when, "bank_days"),
            text(when, "earliest_at"),
        ),
        _ => println!("    {never}"),
    };
    projected("fades below τ", "fade", "never fades below τ");
    projected("purgeable", "purge", "never purged");

    let chain = &view["chain"];
    let members = list(chain, "members");
    if members.len() > 1 {
        println!("  chain, head {}:", text(chain, "head"));
        for member in members {
            let mut notes = Vec::new();
            if let Some(by) = member.get("superseded_by").and_then(Value::as_str) {
                notes.push(format!("superseded by {by}"));
            }
            if member.get("retracted").and_then(Value::as_bool) == Some(true) {
                notes.push("retracted".to_string());
            }
            if member.get("hidden").and_then(Value::as_bool) == Some(true) {
                notes.push("forgotten".to_string());
            }
            println!("    {}  {}", text(member, "id"), notes.join(", "));
        }
    }
    if let Some(by) = chain.get("ended_by").and_then(Value::as_str) {
        println!("  ended by {by}");
    }
    let ends: Vec<&str> = list(chain, "ends")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    if !ends.is_empty() {
        println!("  ends {}", ends.join(", "));
    }
    for entity in list(view, "entities") {
        let surface = entity
            .get("surface_form")
            .and_then(Value::as_str)
            .map(|form| format!(" as \"{form}\""))
            .unwrap_or_default();
        println!(
            "  about {} ({}){surface}",
            text(entity, "name"),
            text(entity, "id")
        );
    }
    let accesses = list(view, "accesses");
    println!("  accesses ({}):", accesses.len());
    for access in accesses {
        let inherited = access
            .get("inherited_from")
            .and_then(Value::as_str)
            .map(|from| format!(", inherited from {from}"))
            .unwrap_or_default();
        println!(
            "    {} {} turn {}{inherited}",
            text(access, "at"),
            text(access, "kind"),
            count(access, "turn"),
        );
    }
    print_edits(list(view, "edits"));
}

fn print_edits(edits: &[Value]) {
    println!("  edits ({}):", edits.len());
    for edit in edits {
        println!(
            "    {} {} {}  {}",
            text(edit, "at"),
            text(edit, "kind"),
            text(edit, "id"),
            edit.get("details")
                .map(Value::to_string)
                .unwrap_or_default(),
        );
    }
}

/// `asphodel entity show|merge|unmerge|alias rm|link|unlink`.
fn entity(command: EntityCommand) -> anyhow::Result<()> {
    match command {
        EntityCommand::Show {
            client,
            bank,
            entity,
        } => {
            let json = client.json;
            let client = Client::new(&client)?;
            let view: Value = client.get(&format!(
                "/v1/banks/{}/entities/{}",
                segment(&bank),
                segment(&entity)
            ))?;
            if json {
                return print_json(&view);
            }
            print_entity(&view);
            Ok(())
        }
        EntityCommand::Merge {
            client,
            bank,
            from,
            into,
        } => {
            let json = client.json;
            let client = Client::new(&client)?;
            let merged: Value = client.post(
                &format!("/v1/banks/{}/merges", segment(&bank)),
                &MergeRequest { from, into },
            )?;
            if json {
                return print_json(&merged);
            }
            println!(
                "merged {} into {}: {} aliases and {} links moved, {} models repointed",
                text(&merged, "from"),
                text(&merged, "into"),
                count(&merged, "aliases_moved"),
                count(&merged, "links_moved"),
                count(&merged, "models_repointed"),
            );
            println!(
                "undo with `asphodel entity unmerge --bank {bank} {}`",
                text(&merged, "edit")
            );
            Ok(())
        }
        EntityCommand::Unmerge { client, bank, edit } => {
            let json = client.json;
            let client = Client::new(&client)?;
            let unmerged: Value = client.post(
                &format!(
                    "/v1/banks/{}/merges/{}/undo",
                    segment(&bank),
                    segment(&edit)
                ),
                &Value::Null,
            )?;
            if json {
                return print_json(&unmerged);
            }
            println!(
                "unmerged {} from {}: {} aliases and {} links moved back, {} models repointed",
                text(&unmerged, "from"),
                text(&unmerged, "into"),
                count(&unmerged, "aliases_moved"),
                count(&unmerged, "links_moved"),
                count(&unmerged, "models_repointed"),
            );
            Ok(())
        }
        EntityCommand::Alias(AliasCommand::Rm {
            client,
            bank,
            entity,
            alias,
            relink_to,
        }) => {
            let json = client.json;
            let client = Client::new(&client)?;
            let removed: Value = client.post(
                &format!("/v1/banks/{}/aliases/remove", segment(&bank)),
                &AliasRemoval {
                    entity,
                    alias,
                    relink_to,
                },
            )?;
            if json {
                return print_json(&removed);
            }
            match removed.get("relinked_to").and_then(Value::as_str) {
                Some(target) => println!(
                    "removed the alias from {}; {} links moved to {target}, which has it now",
                    text(&removed, "entity"),
                    count(&removed, "links_moved"),
                ),
                None => println!("removed the alias from {}", text(&removed, "entity")),
            }
            Ok(())
        }
        EntityCommand::Link {
            client,
            bank,
            memory,
            entity,
        } => edit_link(client, &bank, memory, entity, true),
        EntityCommand::Unlink {
            client,
            bank,
            memory,
            entity,
        } => edit_link(client, &bank, memory, entity, false),
    }
}

fn edit_link(
    client_args: ClientArgs,
    bank: &str,
    memory: String,
    entity: String,
    link: bool,
) -> anyhow::Result<()> {
    let client = Client::new(&client_args)?;
    let route = if link { "links" } else { "links/remove" };
    let edited: Value = client.post(
        &format!("/v1/banks/{}/{route}", segment(bank)),
        &LinkRequest { memory, entity },
    )?;
    if client_args.json {
        return print_json(&edited);
    }
    let changed = edited.get("changed").and_then(Value::as_bool) == Some(true);
    let (done, already) = if link {
        ("linked", "already linked")
    } else {
        ("unlinked", "wasn't linked")
    };
    println!(
        "{} {} {}",
        text(&edited, "memory"),
        if changed { done } else { already },
        text(&edited, "entity"),
    );
    Ok(())
}

fn print_entity(view: &Value) {
    let seeded = view
        .get("seeded")
        .and_then(Value::as_str)
        .map(|seeded| format!(", the bank's {seeded}"))
        .unwrap_or_default();
    println!(
        "{}  {}  [{}{seeded}]",
        text(view, "id"),
        text(view, "name"),
        text(view, "kind")
    );
    if let Some(into) = view.get("merged_into").and_then(Value::as_str) {
        let survivor = view
            .get("survivor")
            .and_then(Value::as_str)
            .map(|survivor| format!(", which ends at {survivor}"))
            .unwrap_or_default();
        println!("  merged into {into}{survivor}");
    }
    let merged_from: Vec<&str> = list(view, "merged_from")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    if !merged_from.is_empty() {
        println!("  merged into it: {}", merged_from.join(", "));
    }
    let aliases: Vec<&str> = list(view, "aliases")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    println!("  aliases: {}", aliases.join(", "));
    let speakers: Vec<&str> = list(view, "speaker_ids")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    if !speakers.is_empty() {
        println!("  speaker ids: {}", speakers.join(", "));
    }
    let models: Vec<&str> = list(view, "models")
        .iter()
        .filter_map(Value::as_str)
        .collect();
    if !models.is_empty() {
        println!("  models filtering on it: {}", models.join(", "));
    }
    let memories = list(view, "memories");
    println!(
        "  memories: {} linked{}",
        count(view, "memory_count"),
        if (memories.len() as u64) < count(view, "memory_count") {
            format!(", the newest {} shown", memories.len())
        } else {
            String::new()
        }
    );
    for memory in memories {
        let surface = memory
            .get("surface_form")
            .and_then(Value::as_str)
            .map(|form| format!(" as \"{form}\""))
            .unwrap_or_default();
        let via = memory
            .get("via")
            .and_then(Value::as_str)
            .map(|via| format!(" via {via}"))
            .unwrap_or_default();
        println!(
            "    {}  [{}]  {}{surface}{via}",
            text(memory, "id"),
            text(memory, "kind"),
            text(memory, "sentence"),
        );
    }
    print_edits(list(view, "edits"));
}

/// How often `asphodel reembed` looks at the job while it follows it.
const REEMBED_POLL: std::time::Duration = std::time::Duration::from_secs(1);

/// `asphodel reembed --bank`: starts or resumes the daemon's job, and
/// follows it to the swap unless `--no-wait`.
fn reembed(args: ReembedArgs) -> anyhow::Result<()> {
    let client = Client::new(&args.client)?;
    let path = format!("/v1/banks/{}/reembed", segment(&args.bank));
    let mut status: Value = client.post(&path, &Value::Null)?;
    if text(&status, "state") == "current" {
        if args.client.json {
            return print_json(&status);
        }
        println!(
            "{} is already on {}; nothing to re-embed",
            args.bank,
            text(&status, "model")
        );
        return Ok(());
    }
    let from = text(&status, "recorded_model").to_string();
    if args.no_wait {
        if args.client.json {
            return print_json(&status);
        }
        println!(
            "re-embedding {} from {from} to {}; follow it with `asphodel reembed --bank {}`",
            args.bank,
            text(&status, "model"),
            args.bank
        );
        return Ok(());
    }
    if !args.client.json {
        println!(
            "re-embedding {} from {from} to {}: served with {from} until the swap",
            args.bank,
            text(&status, "model")
        );
    }
    let mut shown = u64::MAX;
    loop {
        match text(&status, "state") {
            "current" => break,
            "failed" => bail!(
                "the re-embed stopped ({}); run it again to resume from where it got to",
                text(&status, "error")
            ),
            _ => {}
        }
        let embedded = count(&status, "embedded");
        if embedded != shown && !args.client.json {
            println!(
                "  {embedded} of {} memories embedded",
                count(&status, "memories")
            );
            shown = embedded;
        }
        std::thread::sleep(REEMBED_POLL);
        status = client.get(&path)?;
    }
    if args.client.json {
        return print_json(&status);
    }
    println!(
        "swapped: {} is served with {} now",
        args.bank,
        text(&status, "recorded_model")
    );
    Ok(())
}
