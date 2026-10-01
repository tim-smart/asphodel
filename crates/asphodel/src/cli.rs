//! The subcommand tree from "API surface and Hermes transport" (TIM-94,
//! decision 10), plus `replay` and `bench` from the replay harness decision
//! (TIM-96). `serve` runs the daemon; `ingest`, `bank`, `chunks`, `recall`,
//! `keep`, `unkeep` and `model` are HTTP clients of it ([`crate::client`]);
//! `models fetch` and `llm login` work on files. `forget`, `replay` and `bench` are
//! stubs that later stages fill in.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::{Context, bail};
use asphodel_core::SystemClock;
use asphodel_core::config::{Secret, TOKEN_ENV};
use asphodel_core::constants::Volatility;
use asphodel_core::ingest::Document;
use asphodel_core::keep::MemoryIds;
use asphodel_core::mental_models::{ModelEdit, ModelSpec};
use asphodel_core::models::{
    AUTH_ISSUER, DeviceCode, HttpFetcher, ModelDir, TokenStore, device_code_login, fetch_models,
    manifest,
};
use asphodel_core::queue::RetryRequest;
use asphodel_core::retrieval::{On, PhaseFilter, RecallRequest};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::strength::Kind;
use clap::{Args, Parser, Subcommand};
use jiff::Timestamp;
use jiff::civil::Date;
use serde::de::DeserializeOwned;
use serde_json::Value;

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

    /// Define, list and refresh a bank's mental models.
    #[command(subcommand)]
    Model(ModelCommand),

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
}

/// A bank's identity (TIM-94, decision 7), as `PUT /v1/banks/{bank}` takes it.
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

/// `asphodel model`: only the owner defines models, through here or the
/// API (TIM-95, decision 2).
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
            Command::Chunks(args) => chunks(args),
            Command::Recall(args) => recall(args),
            Command::Forget(_) => stub("forget"),
            Command::Keep(args) => keep(args, Keep::Keep),
            Command::Unkeep(args) => keep(args, Keep::Unkeep),
            Command::Model(command) => model(command),
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
enum Keep {
    Keep,
    Unkeep,
}

/// `asphodel keep|unkeep`: `POST /v1/banks/{bank}/keep|unkeep`. Ids the
/// bank doesn't have make it exit non-zero, after the rest are done.
fn keep(args: IdsArgs, action: Keep) -> anyhow::Result<()> {
    let client = Client::new(&args.client)?;
    let (route, done_key, verb) = match action {
        Keep::Keep => ("keep", "kept", "kept"),
        Keep::Unkeep => ("unkeep", "unkept", "unkept"),
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
                _ => bail!(
                    "the refresh failed ({}); it's tried again in 30 minutes",
                    detail.as_str().unwrap_or("unknown")
                ),
            }
            Ok(())
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

fn stub(name: &str) -> anyhow::Result<()> {
    bail!("`asphodel {name}` isn't implemented yet")
}
