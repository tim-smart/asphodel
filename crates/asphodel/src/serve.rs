//! `asphodel serve`: the daemon.
//!
//! It binds the listen address first, so `/v1/health` can answer 503 while the
//! store opens and migrates and the models load, then builds the service layer
//! on the system clock and serves the HTTP API under `/v1` ([`api`]). The
//! handlers stay thin and call the same service functions the replay harness
//! does. Each bank's chunks are extracted by a worker of its own ([`worker`]).
//! With no tuning file anywhere, it waits for first-run setup ([`setup`])
//! before it opens the store.

use std::{
    fs::{self, Metadata},
    io::ErrorKind,
    os::unix::fs::{FileTypeExt, MetadataExt},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock, Weak},
    time::Duration,
};

use anyhow::{Context, bail};
use asphodel_core::clock::{Sleeper, StopSleeper, ThreadSleeper};
use asphodel_core::config::{
    Deployment, LLM_API_KEY_ENV, LlmAuth, ModelsConfig, Secret, TOKEN_ENV,
};
use asphodel_core::models::{
    CodexResponses, EMBEDDING_MODEL_ID, Embedder, FakeEmbedder, FakeEmbedderV2, FakeLlm,
    FakeReranker, LlmClient, LlmGate, LlmRetry, LlmSettings, LlmStatus, ModelOptions,
    OpenAiCompatible, RERANKER_MODEL_ID, Reranker, ResidentModels, RetryBoard, RetryPolicy,
    TokenStore,
};
use asphodel_core::store::{LockedDir, OpenOptions, Store};
use asphodel_core::{Clock, ResolvedConfig, Service, SystemClock, Tuning};
use tokio::net::{TcpListener, UnixListener, UnixStream};
use tokio::signal;
use tokio::sync::{oneshot, watch};
use tracing::{info, warn};

use crate::cli::ServeArgs;
use crate::listen::Listen;
use setup::{Pending, StartingFloors, StoredSecrets};
use worker::Workers;

mod api;
mod dashboard;
mod setup;
mod worker;

/// The longest the daemon waits between housekeeping passes. It normally
/// wakes at the next copy's deadline; this cap bounds how late that wake can
/// be after a host suspend, which tokio's monotonic timer doesn't count, and
/// is the retry delay after a failed pass.
const HOUSEKEEPING_INTERVAL: Duration = Duration::from_secs(60 * 60);

/// The longest the refresh timer sleeps. A trigger written meanwhile moves
/// the next refresh earlier than the timer knew, so it looks again at
/// least this often; the debounce is minutes, so a minute late is fine.
const REFRESH_INTERVAL: Duration = Duration::from_secs(60);

/// The longest the idle release waits between looks at the models.
const RELEASE_INTERVAL: Duration = Duration::from_secs(60);

/// The shortest wait between passes. The timer is monotonic and the
/// deadline is on the service's clock, so a wake can land just short of it;
/// this keeps that from becoming a run of near-zero waits.
const HOUSEKEEPING_MIN_WAIT: Duration = Duration::from_secs(1);

pub(crate) type Shared = Arc<App>;

/// What the HTTP handlers share. The service only exists once the store is open
/// and the models are loaded; until then `/v1/health` answers 503 and every
/// other route refuses.
pub(crate) struct App {
    /// The clock `/v1/health` reads before the service exists.
    clock: Arc<dyn Clock>,
    /// The bearer token clients must send, when one is configured. Set once
    /// the config is resolved: at startup, or once setup is done.
    token: OnceLock<Option<Secret>>,
    /// First-run setup, while it waits for the wizard.
    setup: setup::Slot,
    ready: OnceLock<Ready>,
    /// Set on SIGTERM or SIGINT: ingest stops, and so do the workers once
    /// their chunk in flight is done.
    stop: watch::Receiver<bool>,
}

/// The daemon once it's ready.
pub(crate) struct Ready {
    service: Arc<Service>,
    config: ResolvedConfig,
    /// `None` when no LLM is configured: chunks wait on the queue.
    workers: Option<Arc<Workers>>,
    /// The LLM translate calls, when one is configured: through the shared
    /// gate, on translate's own bounded retries.
    llm: Option<Arc<dyn LlmClient>>,
}

impl App {
    fn ready(&self) -> Option<&Ready> {
        self.ready.get()
    }

    fn draining(&self) -> bool {
        *self.stop.borrow()
    }

    fn setting_up(&self) -> bool {
        self.setup
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .is_some()
    }
}

/// What startup builds on a blocking thread before the daemon is ready.
struct Started {
    service: Service,
    config: ResolvedConfig,
    /// The gate every caller's LLM calls go through, when one is
    /// configured. Each caller retries above it on its own policy.
    llm: Option<Arc<dyn LlmClient>>,
    /// Every bank in the store, each of which gets a worker at once, since
    /// the queue may hold chunks from before a restart.
    banks: Vec<String>,
    /// The models the service runs on, for the idle release.
    models: Arc<ResidentModels>,
}

/// The bound listener, before axum takes it.
enum Bound {
    Tcp(TcpListener),
    Unix(UnixListener, UnixSocketCleanup),
}

pub async fn run(args: ServeArgs) -> anyhow::Result<()> {
    let (stop_tx, stop) = watch::channel(false);
    let signals = tokio::spawn({
        let stop_tx = stop_tx.clone();
        async move {
            shutdown_signal().await;
            let _ = stop_tx.send(true);
        }
    });
    let served = run_with(args, stop_tx, stop, None).await;
    signals.abort();
    served
}

/// The daemon under a caller's stop signal, telling `bound` the address it
/// listens on once it does. `run` adds the signal handlers; `asphodel bench`
/// runs this on a copy of a replayed store and stops it itself.
pub(crate) async fn run_with(
    args: ServeArgs,
    stop_tx: watch::Sender<bool>,
    stop: watch::Receiver<bool>,
    bound_tx: Option<tokio::sync::oneshot::Sender<String>>,
) -> anyhow::Result<()> {
    let config = resolve_config(&args)?;
    if let Some(config) = &config {
        require_token(&args.listen, config)?;
    }
    // The switches are read before anything binds, so a bad value stops the
    // daemon before it exists to a client.
    let models = models_switch()?;
    let script = llm_script()?;
    let retry_wait = retry_wait()?;
    let gate = startup_gate();
    let model_idle_minutes = args.model_idle_minutes;
    // No tuning file anywhere: setup waits for the wizard, guarded by a code
    // in the data dir. It takes the data dir's lock first and keeps it until
    // the store opens, so no other daemon can read the code or write the
    // files meanwhile.
    let (locked, pending, finished) = match &config {
        Some(_) => (None, None, None),
        None => {
            let locked = LockedDir::acquire(&args.data_dir, args.allow_network_fs)
                .with_context(|| format!("locking the data dir {}", args.data_dir.display()))?;
            let floors = models.starting_floors();
            let (pending, finished) = Pending::new(locked.dir(), !args.listen.is_local(), floors)?;
            (Some(locked), Some(pending), Some(finished))
        }
    };

    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    let app: Shared = Arc::new(App {
        clock: Arc::clone(&clock),
        token: OnceLock::new(),
        setup: Mutex::new(pending),
        ready: OnceLock::new(),
        stop: stop.clone(),
    });

    let (bound, address) = bind(&args.listen).await?;
    info!(
        version = asphodel_core::VERSION,
        listen = %address,
        "asphodel starting: /v1/health answers 503 until the store and models are ready"
    );
    if let Some(bound_tx) = bound_tx {
        let _ = bound_tx.send(address.clone());
    }
    let router = api::router(Arc::clone(&app));
    let (server, cleanup) = match bound {
        Bound::Tcp(listener) => (
            tokio::spawn(
                axum::serve(listener, router)
                    .with_graceful_shutdown(stopped(stop.clone()))
                    .into_future(),
            ),
            None,
        ),
        Bound::Unix(listener, cleanup) => (
            tokio::spawn(
                axum::serve(listener, router)
                    .with_graceful_shutdown(stopped(stop.clone()))
                    .into_future(),
            ),
            Some(cleanup),
        ),
    };

    let config = match (config, finished) {
        (Some(config), _) => Ok(Some(config)),
        (None, Some(finished)) => {
            warn!(
                listen = %address,
                code = %args.data_dir.join(setup::SETUP_CODE_FILE).display(),
                "no tuning file: open /dashboard to set up, with the code in the file named here"
            );
            await_setup(&args, finished, stop.clone()).await
        }
        (None, None) => unreachable!("setup is pending whenever there's no config"),
    };

    // The store opens, migrates and loads its models while the listener answers
    // 503: the lock, the filesystem check, the migrations and the floors all
    // have to pass before the daemon is ready.
    let started = match config {
        Ok(Some(config)) => {
            let _ = app.token.set(config.deployment.token.clone());
            let clock = Arc::clone(&clock);
            let stop = stop.clone();
            tokio::task::spawn_blocking(move || {
                start(&args, locked, config, clock, models, script, gate, &stop)
            })
            .await
            .map_err(|error| anyhow::anyhow!("startup panicked: {error}"))
            .and_then(|started| started)
        }
        Ok(None) => Ok(None),
        Err(error) => Err(error),
    };
    let started = match started {
        Ok(Some(started)) => started,
        Ok(None) | Err(_) => {
            let _ = stop_tx.send(true);
            let served = server.await;
            if let Some(cleanup) = cleanup {
                cleanup.remove()?;
            }
            served??;
            return started.map(|_| ());
        }
    };

    let service = Arc::new(started.service);
    tokio::spawn(housekeeping(Arc::downgrade(&service)));
    if model_idle_minutes > 0 {
        let idle = Duration::from_secs(u64::from(model_idle_minutes) * 60);
        tokio::spawn(release_idle_models(started.models, idle, stop.clone()));
    }
    let retries = Arc::new(Retries {
        clock: Arc::clone(&clock),
        sleeper: Arc::new(StopSleeper::default()),
        board: service.llm_retries(),
        wait: retry_wait,
    });
    // A call waiting out an outage wakes on the stop and gives up without
    // counting anything, so shutdown never waits for the provider.
    tokio::spawn({
        let retries = Arc::clone(&retries);
        let stop = stop.clone();
        async move {
            stopped(stop).await;
            retries.sleeper.stop();
        }
    });
    if let Some(gate) = &started.llm {
        tokio::spawn(refreshes(
            Arc::downgrade(&service),
            retries.client(gate, RetryPolicy::daemon(), Some("refresh")),
            stop.clone(),
        ));
    }
    let workers = match &started.llm {
        Some(gate) => Some(Workers::start(
            Arc::clone(&service),
            Arc::clone(gate),
            Arc::clone(&retries),
            &started.banks,
            stop.clone(),
        )),
        None => None,
    };
    let translate = started
        .llm
        .as_ref()
        .map(|gate| retries.client(gate, RetryPolicy::translate(), None));
    let _ = app.ready.set(Ready {
        service: Arc::clone(&service),
        config: started.config,
        workers: workers.clone(),
        llm: translate,
    });
    info!(version = asphodel_core::VERSION, listen = %address, "asphodel listening");
    // A re-embed a restart stopped resumes from where it got to.
    match service.pending_reembeds() {
        Ok(banks) => {
            for bank in banks {
                run_reembed(Arc::clone(&service), workers.clone(), bank);
            }
        }
        Err(error) => warn!(%error, "listing the re-embeds to resume failed"),
    }

    // SIGTERM: ingest is refused and the listener stops; each worker finishes
    // its chunk in flight; then the WAL is checkpointed. The queue is in
    // SQLite, so nothing waiting is lost.
    let served = server.await;
    let _ = stop_tx.send(true);
    if let Some(workers) = &workers {
        workers.join().await;
    }
    let checkpoint = tokio::task::spawn_blocking({
        let service = Arc::clone(&service);
        move || service.checkpoint()
    })
    .await?;
    match checkpoint {
        Ok(()) => info!("checkpointed the WAL"),
        Err(error) => warn!(%error, "checkpointing the WAL failed"),
    }
    if let Some(cleanup) = cleanup {
        cleanup.remove()?;
    }
    served??;
    info!("asphodel stopped");
    Ok(())
}

/// Runs a bank's re-embed to its swap on a blocking thread, then wakes the
/// bank's worker, which waited while the swap held its lease. A failure is
/// kept for `GET /v1/banks/{bank}/reembed`, and the job resumes on the next
/// `asphodel reembed` or restart.
pub(crate) fn run_reembed(service: Arc<Service>, workers: Option<Arc<Workers>>, bank: String) {
    tokio::task::spawn_blocking(move || {
        if let Err(error) = service.run_reembed(&bank) {
            warn!(%error, "a re-embed stopped");
        }
        if let Some(workers) = workers {
            workers.wake(&bank);
        }
    });
}

/// Waits for the wizard to finish setup, then resolves the config it wrote
/// exactly as a restart would. `Ok(None)` when a stop arrives first.
async fn await_setup(
    args: &ServeArgs,
    finished: oneshot::Receiver<()>,
    stop: watch::Receiver<bool>,
) -> anyhow::Result<Option<ResolvedConfig>> {
    tokio::select! {
        done = finished => {
            if done.is_err() {
                return Ok(None);
            }
        }
        () = stopped(stop) => return Ok(None),
    }
    let config = resolve_config(args)?.context("setup finished without a tuning file")?;
    require_token(&args.listen, &config)?;
    info!("setup is done: starting");
    Ok(Some(config))
}

/// Off loopback, every client must send a token, so the daemon refuses to
/// start without one.
fn require_token(listen: &Listen, config: &ResolvedConfig) -> anyhow::Result<()> {
    if !listen.is_local() && config.deployment.token.is_none() {
        bail!(
            "listening on {listen} is reachable off this machine, so a bearer token is required: \
             set {TOKEN_ENV}, or `token` in {} in the data dir",
            setup::SECRETS_FILE
        );
    }
    Ok(())
}

/// Resolves once a stop has been signalled.
async fn stopped(mut stop: watch::Receiver<bool>) {
    let _ = stop.wait_for(|stop| *stop).await;
}

/// Binds the listen address, and says where it ended up: with port 0 the
/// system picks the port, and the log line is how a test finds it.
async fn bind(listen: &Listen) -> anyhow::Result<(Bound, String)> {
    match listen {
        Listen::Tcp(addr) => {
            let listener = TcpListener::bind(addr)
                .await
                .with_context(|| format!("binding {addr}"))?;
            let address = listener.local_addr()?.to_string();
            Ok((Bound::Tcp(listener), address))
        }
        Listen::Unix(path) => {
            let (listener, cleanup) = bind_unix(path).await?;
            Ok((Bound::Unix(listener, cleanup), listen.to_string()))
        }
    }
}

/// Opens the store, loads the models and builds the service and the LLM
/// client, on a blocking thread. `locked` is the data dir setup already
/// locked, if it ran. `Ok(None)` when a stop arrived while the startup gate
/// held it.
#[allow(clippy::too_many_arguments)]
fn start(
    args: &ServeArgs,
    locked: Option<LockedDir>,
    mut config: ResolvedConfig,
    clock: Arc<dyn Clock>,
    models: ModelsSwitch,
    script: Option<String>,
    gate: Option<PathBuf>,
    stop: &watch::Receiver<bool>,
) -> anyhow::Result<Option<Started>> {
    if let Some(gate) = gate {
        warn!(
            "{STARTUP_GATE_ENV} is set: startup waits until {} exists",
            gate.display()
        );
        while !gate.exists() {
            if *stop.borrow() {
                return Ok(None);
            }
            std::thread::sleep(STARTUP_GATE_POLL);
        }
    }

    let options = OpenOptions {
        allow_network_fs: args.allow_network_fs,
        deterministic_ids: false,
    };
    let store = match locked {
        Some(locked) => Store::open_locked(locked, options, Arc::clone(&clock)),
        None => Store::open(&args.data_dir, options, Arc::clone(&clock)),
    }
    .with_context(|| format!("opening the store in {}", args.data_dir.display()))?;
    config.purge = store.check_fingerprint(&config.deletion_fingerprint)?;
    config.llm = llm_status(&config, &args.data_dir)?;
    // The models always sit behind their idle release; with
    // `--model-idle-minutes 0` nothing ever releases them.
    let carries_v1 = matches!(models, ModelsSwitch::FakeV2);
    let (resident, fake) = match models {
        ModelsSwitch::Fake => {
            warn!(
                "{MODELS_ENV}=fake: serving with the deterministic fake models, not the ONNX ones"
            );
            let resident = ResidentModels::new(
                || Ok(Arc::new(FakeEmbedder) as Arc<dyn Embedder>),
                || Ok(Arc::new(FakeReranker) as Arc<dyn Reranker>),
            )?;
            (resident, true)
        }
        ModelsSwitch::FakeV2 => {
            warn!(
                "{MODELS_ENV}=fake-v2: serving with the second fake embedder, carrying the first for banks recorded under it"
            );
            let resident = ResidentModels::new(
                || Ok(Arc::new(FakeEmbedderV2) as Arc<dyn Embedder>),
                || Ok(Arc::new(FakeReranker) as Arc<dyn Reranker>),
            )?;
            (resident, true)
        }
        ModelsSwitch::None => {
            let dir = crate::cli::resolve_model_dir(args.model_dir.as_deref())?;
            let options = ModelOptions {
                threads: args.onnx_threads,
            };
            let resident = ResidentModels::load(&dir, &options)
                .with_context(|| format!("loading the models in {}", dir.path().display()))?;
            (resident, false)
        }
    };
    let models = resident.models();
    config.models = Some(ModelsConfig::new(
        &models,
        fake,
        args.onnx_threads.map(std::num::NonZeroUsize::get),
        (args.model_idle_minutes > 0).then_some(args.model_idle_minutes),
    ));
    let mut service =
        Service::with_models(Arc::clone(&clock), store, config.tuning.clone(), models)?;
    if carries_v1 {
        service = service.with_previous_embedder(Arc::new(FakeEmbedder))?;
    }
    // Purge and the sweep run, or wait for an ack, as the store's deletion
    // fingerprint says. Forget never waits.
    let service = service.with_purge_pause(config.purge.clone());
    let llm = llm_client(&config, &args.data_dir, Arc::clone(&clock), script)?;
    config.fake_llm = llm.as_ref().is_some_and(|(_, fake)| *fake);
    // Extraction, refresh and translate share one gate: `[llm]
    // concurrency` calls in flight at most, and one hold when any of them
    // hits a limit. Each retries above it, so a call waiting to retry
    // frees its slot, and a hold set meanwhile stops its next attempt.
    let concurrency = config.tuning.llm.concurrency as usize;
    let llm = llm.map(|(llm, fake)| {
        let gate: Arc<dyn LlmClient> = Arc::new(LlmGate::new(llm, concurrency, clock));
        (gate, fake)
    });
    info!(
        config = %serde_json::to_string(&config)?,
        "resolved config"
    );
    for (bank, model) in service.banks_without_their_model()? {
        warn!(%bank, %model, "the bank records an embedding model this daemon doesn't carry, so its recall and extraction are refused; run `asphodel reembed --bank` to move it");
    }
    let banks = service.bank_names()?;
    Ok(Some(Started {
        service,
        config,
        llm: llm.map(|(llm, _)| llm),
        banks,
        models: Arc::new(resident),
    }))
}

/// `ASPHODEL_LLM_SCRIPT=<file>` runs extraction on a scripted fake LLM that
/// plays the file's steps in order ([`FakeLlm::from_script`]). It is for
/// integration tests, environment only, never in `--help`, and the resolved
/// config says so.
pub(crate) const LLM_SCRIPT_ENV: &str = "ASPHODEL_LLM_SCRIPT";

/// The model name the scripted LLM reports when `llm.model` isn't set.
pub(crate) const FAKE_LLM_MODEL: &str = "fake-llm";

/// Reads the LLM script, if there is one, before anything binds.
fn llm_script() -> anyhow::Result<Option<String>> {
    let Some(path) = std::env::var_os(LLM_SCRIPT_ENV) else {
        return Ok(None);
    };
    let script = fs::read_to_string(&path)
        .with_context(|| format!("reading {LLM_SCRIPT_ENV} {}", Path::new(&path).display()))?;
    Ok(Some(script))
}

/// `ASPHODEL_LLM_RETRY_WAIT_MS=<ms>` replaces every wait between the
/// attempts of an LLM call, in the daemon and in replay's live calls, so an
/// integration test can retry without sleeping. It is for integration
/// tests, environment only, never in `--help`.
pub(crate) const LLM_RETRY_WAIT_ENV: &str = "ASPHODEL_LLM_RETRY_WAIT_MS";

/// The wait [`LLM_RETRY_WAIT_ENV`] sets, if it's set.
fn retry_wait() -> anyhow::Result<Option<Duration>> {
    let Some(value) = std::env::var_os(LLM_RETRY_WAIT_ENV) else {
        return Ok(None);
    };
    let wait = value
        .to_str()
        .and_then(|value| value.parse().ok())
        .map(Duration::from_millis)
        .with_context(|| format!("{LLM_RETRY_WAIT_ENV} isn't a whole number of milliseconds"))?;
    warn!("{LLM_RETRY_WAIT_ENV} is set: LLM retries wait {wait:?}");
    Ok(Some(wait))
}

/// `policy` with every wait [`LLM_RETRY_WAIT_ENV`] set, if it's set.
fn waiting(mut policy: RetryPolicy, wait: Option<Duration>) -> RetryPolicy {
    if let Some(wait) = wait {
        policy.first_wait = wait;
        policy.max_wait = wait;
    }
    policy
}

/// Replay's live client: `llm` retrying on [`RetryPolicy::replay`], timed
/// on `clock`.
pub(crate) fn retrying(
    llm: Arc<dyn LlmClient>,
    clock: Arc<dyn Clock>,
) -> anyhow::Result<Arc<dyn LlmClient>> {
    let policy = waiting(RetryPolicy::replay(), retry_wait()?);
    Ok(Arc::new(LlmRetry::new(
        llm,
        policy,
        clock,
        Arc::new(ThreadSleeper),
    )))
}

/// What every daemon caller's retries share: the clock, the sleeper the
/// stop wakes, and the board `status` reads.
pub(crate) struct Retries {
    clock: Arc<dyn Clock>,
    sleeper: Arc<StopSleeper>,
    board: RetryBoard,
    wait: Option<Duration>,
}

impl Retries {
    /// `gate` retrying on `policy`, and listed on the board as `caller`
    /// while it waits to retry, if it has one.
    pub(crate) fn client(
        &self,
        gate: &Arc<dyn LlmClient>,
        policy: RetryPolicy,
        caller: Option<&str>,
    ) -> Arc<dyn LlmClient> {
        let sleeper: Arc<dyn Sleeper> = self.sleeper.clone();
        let retry = LlmRetry::new(
            Arc::clone(gate),
            waiting(policy, self.wait),
            Arc::clone(&self.clock),
            sleeper,
        );
        Arc::new(match caller {
            Some(caller) => retry.reporting(self.board.clone(), caller),
            None => retry,
        })
    }
}

/// `ASPHODEL_STARTUP_GATE=<path>` holds startup, after the listener is bound
/// and before the store opens, until the path exists. It lets an
/// integration test see `/v1/health` answer 503 while the daemon starts,
/// which the fake models otherwise make too quick to catch. Environment
/// only, never in `--help`.
const STARTUP_GATE_ENV: &str = "ASPHODEL_STARTUP_GATE";
const STARTUP_GATE_POLL: Duration = Duration::from_millis(50);

fn startup_gate() -> Option<PathBuf> {
    std::env::var_os(STARTUP_GATE_ENV).map(PathBuf::from)
}

/// The LLM extraction calls, and whether it's the scripted fake. `None`
/// when `[llm]` isn't configured and there's no script: the daemon serves,
/// and chunks wait on the queue.
fn llm_client(
    config: &ResolvedConfig,
    data_dir: &Path,
    clock: Arc<dyn Clock>,
    script: Option<String>,
) -> anyhow::Result<Option<(Arc<dyn LlmClient>, bool)>> {
    if let Some(script) = script {
        let model = config.tuning.llm.model.as_deref().unwrap_or(FAKE_LLM_MODEL);
        warn!("{LLM_SCRIPT_ENV} is set: extraction runs on a scripted fake LLM, not a real one");
        let llm = FakeLlm::from_script(model, &script)?;
        return Ok(Some((Arc::new(llm), true)));
    }
    let Some(settings) = LlmSettings::from_config(&config.tuning, &config.deployment)? else {
        warn!("[llm] isn't configured: chunks wait on the queue until it is");
        return Ok(None);
    };
    let llm: Arc<dyn LlmClient> = match settings.auth {
        LlmAuth::ApiKey => Arc::new(OpenAiCompatible::new(settings)),
        LlmAuth::Chatgpt => {
            let mut client = CodexResponses::new(settings, TokenStore::open(data_dir), clock);
            if let Ok(issuer) = std::env::var(crate::cli::LLM_ISSUER_ENV) {
                client = client.with_issuer(&issuer);
            }
            Arc::new(client)
        }
    };
    Ok(Some((llm, false)))
}

/// `ASPHODEL_MODELS=fake` runs the daemon on the deterministic fake models.
/// It is for tests, environment only, never in `--help`, and the resolved
/// config says so.
pub(crate) const MODELS_ENV: &str = "ASPHODEL_MODELS";

enum ModelsSwitch {
    None,
    Fake,
    /// The second fake embedder, carrying the first: a model change on
    /// fakes, for re-embed tests.
    FakeV2,
}

impl ModelsSwitch {
    /// The floors setup writes for the models this switch runs. For the real
    /// models: the reranker's gate floor is -8.5 from 380 labels over 10
    /// prefetches of the call 1 v5 replay (6 of 18 relevant kept at
    /// precision 0.21); its relevance scale is std(L-6 logits) / std(jina
    /// logits) over the 69-query eval pools; and the embedding floor is a
    /// placeholder still to be calibrated in replay.
    fn starting_floors(&self) -> StartingFloors {
        let fake_reranker = (FakeReranker::MODEL_ID, 0.0, 1.0);
        match self {
            ModelsSwitch::None => StartingFloors {
                embedding: vec![(EMBEDDING_MODEL_ID, 0.8)],
                reranker: (RERANKER_MODEL_ID, -8.5, 3.564211),
                uncalibrated: vec![EMBEDDING_MODEL_ID],
            },
            ModelsSwitch::Fake => StartingFloors {
                embedding: vec![(FakeEmbedder::MODEL_ID, 0.5)],
                reranker: fake_reranker,
                uncalibrated: Vec::new(),
            },
            ModelsSwitch::FakeV2 => StartingFloors {
                embedding: vec![
                    (FakeEmbedderV2::MODEL_ID, 0.5),
                    (FakeEmbedder::MODEL_ID, 0.5),
                ],
                reranker: fake_reranker,
                uncalibrated: Vec::new(),
            },
        }
    }
}

fn models_switch() -> anyhow::Result<ModelsSwitch> {
    match std::env::var_os(MODELS_ENV) {
        None => Ok(ModelsSwitch::None),
        Some(value) if value == "fake" => Ok(ModelsSwitch::Fake),
        Some(value) if value == "fake-v2" => Ok(ModelsSwitch::FakeV2),
        Some(value) => bail!(
            "{MODELS_ENV} is {:?}; the values are `fake` and `fake-v2`, for tests. Unset it to run the ONNX models",
            value
        ),
    }
}

/// Resolves the LLM settings so a misconfiguration stops the daemon: a
/// half-set `[llm]`, or a key set together with `auth = "chatgpt"`. In
/// `chatgpt` mode the token file under the data dir decides whether the
/// daemon is logged in; it can start logged out, and extraction waits for
/// `asphodel llm login`.
fn llm_status(config: &ResolvedConfig, data_dir: &Path) -> anyhow::Result<Option<LlmStatus>> {
    let Some(settings) = LlmSettings::from_config(&config.tuning, &config.deployment)? else {
        return Ok(None);
    };
    let status = match settings.auth {
        LlmAuth::ApiKey => LlmStatus::api_key(settings.api_key.is_some()),
        LlmAuth::Chatgpt => TokenStore::open(data_dir).status(),
    };
    if status.auth == LlmAuth::Chatgpt && !status.logged_in {
        warn!(
            "llm.auth is chatgpt but there is no login: run `asphodel llm login --data-dir {}`",
            data_dir.display()
        );
    }
    Ok(Some(status))
}

/// Loads the tuning file and records the deployment. The file is
/// `--config`, else the one in the data dir; `None` when neither is there,
/// and setup has to write one. Each secret comes from the environment, else
/// the data dir's secrets file. An unreadable file, an unknown key or an
/// out-of-range value stops the daemon starting.
fn resolve_config(args: &ServeArgs) -> anyhow::Result<Option<ResolvedConfig>> {
    let Some(config) = args
        .config
        .clone()
        .or_else(|| setup::data_dir_config(&args.data_dir))
    else {
        return Ok(None);
    };
    let tuning = Tuning::load(Some(&config))?;
    let stored = StoredSecrets::read(&args.data_dir)?;
    let deployment = Deployment {
        listen: args.listen.to_string(),
        data_dir: Some(args.data_dir.clone()),
        config: Some(config),
        allow_network_fs: args.allow_network_fs,
        model_dir: args.model_dir.clone(),
        token: setup::secret(TOKEN_ENV, stored.token()),
        llm_api_key: setup::secret(LLM_API_KEY_ENV, stored.llm_api_key()),
    };
    Ok(Some(ResolvedConfig::new(tuning, deployment)))
}

/// Binds a Unix socket, recovering only a stale socket (never a file,
/// symlink or live listener). The caller removes its socket after serving.
pub(crate) async fn bind_unix(path: &Path) -> anyhow::Result<(UnixListener, UnixSocketCleanup)> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            let file_type = metadata.file_type();
            if !file_type.is_socket() {
                let kind = if file_type.is_symlink() {
                    "a symlink"
                } else if file_type.is_file() {
                    "a regular file"
                } else if file_type.is_dir() {
                    "a directory"
                } else {
                    "a non-socket file"
                };
                bail!("refusing to bind {}: found {kind}", path.display());
            }
            match UnixStream::connect(path).await {
                Ok(_) => bail!("socket {} is already in use", path.display()),
                Err(error) if error.kind() == ErrorKind::ConnectionRefused => {
                    // Check identity again after the probe: a replacement must
                    // not be mistaken for the stale socket we inspected.
                    UnixSocketCleanup::new(path, &metadata).remove()?;
                }
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("checking socket {}", path.display()));
                }
            }
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("inspecting {}", path.display()));
        }
    }

    let listener = std::os::unix::net::UnixListener::bind(path)
        .with_context(|| format!("binding {}", path.display()))?;
    listener.set_nonblocking(true)?;
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("inspecting bound socket {}", path.display()))?;
    let mut cleanup = UnixSocketCleanup::new(path, &metadata);
    // Keep the bound socket alive through cleanup, even after axum drops its
    // listener, so its filesystem inode cannot be reused by a replacement.
    cleanup._listener = Some(listener.try_clone()?);
    Ok((UnixListener::from_std(listener)?, cleanup))
}

/// The filesystem identity of the socket we created, not just its name.
pub(crate) struct UnixSocketCleanup {
    path: PathBuf,
    device: u64,
    inode: u64,
    _listener: Option<std::os::unix::net::UnixListener>,
}

impl UnixSocketCleanup {
    fn new(path: &Path, metadata: &Metadata) -> Self {
        Self {
            path: path.to_owned(),
            device: metadata.dev(),
            inode: metadata.ino(),
            _listener: None,
        }
    }

    pub(crate) fn remove(self) -> anyhow::Result<()> {
        let metadata = match fs::symlink_metadata(&self.path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspecting socket {}", self.path.display()));
            }
        };
        if metadata.file_type().is_socket()
            && metadata.dev() == self.device
            && metadata.ino() == self.inode
        {
            // POSIX has no conditional unlink by inode. As with stale-socket
            // recovery, this assumes an operator-controlled socket directory;
            // the identity check protects replacements already at the path.
            fs::remove_file(&self.path)
                .with_context(|| format!("removing socket {}", self.path.display()))?;
        }
        Ok(())
    }
}

/// Runs ready erases, the nightly sweep when it's due, then
/// [`Service::housekeeping`], at
/// the earlier of the two `next_due`s, or after [`HOUSEKEEPING_INTERVAL`] if
/// that comes first, until the service is dropped. The sweep runs here so
/// it runs without an LLM too; the refresh timer also runs it first, so
/// the night's purge always comes before its refreshes. It holds the
/// service weakly, and never across a wait, so the store still closes, and
/// checkpoints, when `run` returns.
async fn housekeeping(service: Weak<Service>) {
    loop {
        let Some(service) = service.upgrade() else {
            return;
        };
        // The first pass runs at once: open deletes what has expired but
        // doesn't say when the next copy is due.
        let pass = tokio::task::spawn_blocking(move || {
            erase(&service);
            let sweep_due = sweep(&service);
            let result = service.housekeeping();
            (result, sweep_due, service.now())
        })
        .await;
        let wait = match pass {
            Ok((Ok(done), sweep_due, now)) => {
                let due = match (done.next_due, sweep_due) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
                due.map_or(HOUSEKEEPING_INTERVAL, |due| {
                    // A deadline already past converts to nothing: wait the
                    // minimum and run again.
                    Duration::try_from(now.duration_until(due))
                        .unwrap_or(Duration::ZERO)
                        .clamp(HOUSEKEEPING_MIN_WAIT, HOUSEKEEPING_INTERVAL)
                })
            }
            Ok((Err(error), _, _)) => {
                warn!(%error, "housekeeping failed");
                HOUSEKEEPING_INTERVAL
            }
            Err(error) => {
                warn!(%error, "housekeeping panicked");
                HOUSEKEEPING_INTERVAL
            }
        };
        tokio::time::sleep(wait).await;
    }
}

/// Releases each model that's gone `idle` unused, looking a quarter of
/// `idle` apart and at least once a minute, until the daemon stops. A model
/// released this way loads again on its next call.
async fn release_idle_models(
    models: Arc<ResidentModels>,
    idle: Duration,
    stop: watch::Receiver<bool>,
) {
    let every = (idle / 4).min(RELEASE_INTERVAL);
    loop {
        tokio::select! {
            () = tokio::time::sleep(every) => {}
            () = stopped(stop.clone()) => return,
        }
        let models = Arc::clone(&models);
        if let Err(error) = tokio::task::spawn_blocking(move || models.release_idle(idle)).await {
            warn!(%error, "releasing idle models panicked");
        }
    }
}

/// Runs every erase that's ready ([`Service::run_erases`]). The workers
/// run them too, but only when there's an LLM; this is how a forget queued
/// before a restart without one still completes. A failure is logged and
/// the next pass tries again.
fn erase(service: &Service) {
    match service.run_erases() {
        Ok(erased) => {
            for one in &erased {
                info!(memories = one.memories.len(), "erased a forgotten chain");
            }
        }
        Err(error) => warn!(%error, "running ready erases failed"),
    }
}

/// Runs any nightly sweep that's due ([`Service::run_sweeps`]) and says
/// when the next is. A failure is logged, and the sweep tries again at the
/// next pass.
fn sweep(service: &Service) -> Option<jiff::Timestamp> {
    match service.run_sweeps() {
        Ok(sweeps) => {
            for run in &sweeps.ran {
                info!(bank = %run.bank, purged = run.purged_memories,
                    sources = run.swept_sources, failed_chunks = run.swept_failed_chunks,
                    recalls = run.swept_recalls, "swept");
            }
            sweeps.next_due
        }
        Err(error) => {
            warn!(%error, "the nightly sweep failed");
            None
        }
    }
}

/// Runs [`Service::run_refreshes`] at each pass's `next_due`, or after
/// [`REFRESH_INTERVAL`] if that comes first, until the daemon stops or the
/// service is dropped. Refreshes run here, never inside a request. Like
/// [`housekeeping`], it holds the service weakly and never across a wait.
async fn refreshes(service: Weak<Service>, llm: Arc<dyn LlmClient>, stop: watch::Receiver<bool>) {
    loop {
        if *stop.borrow() {
            return;
        }
        let Some(service) = service.upgrade() else {
            return;
        };
        let llm = Arc::clone(&llm);
        let pass = tokio::task::spawn_blocking(move || {
            // The night's purge runs before its refreshes.
            sweep(&service);
            let result = service.run_refreshes(llm.as_ref());
            (result, service.now())
        })
        .await;
        let wait = match pass {
            Ok((Ok(done), now)) => {
                for run in &done.ran {
                    info!(bank = %run.bank, model = %run.model, outcome = ?run.outcome,
                        "refreshed a mental model");
                }
                done.next_due.map_or(REFRESH_INTERVAL, |due| {
                    Duration::try_from(now.duration_until(due))
                        .unwrap_or(Duration::ZERO)
                        .clamp(HOUSEKEEPING_MIN_WAIT, REFRESH_INTERVAL)
                })
            }
            Ok((Err(error), _)) => {
                warn!(%error, "refreshing mental models failed");
                REFRESH_INTERVAL
            }
            Err(error) => {
                warn!(%error, "refreshing mental models panicked");
                REFRESH_INTERVAL
            }
        };
        tokio::select! {
            () = tokio::time::sleep(wait) => {}
            () = stopped(stop.clone()) => return,
        }
    }
}

/// Resolves on SIGINT or SIGTERM. Either one stops the daemon as
/// [`run`] describes.
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = signal::ctrl_c().await {
            warn!(%error, "failed to listen for ctrl-c");
            std::future::pending::<()>().await;
        }
    };
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(error) => {
                warn!(%error, "failed to listen for SIGTERM");
                std::future::pending::<()>().await;
            }
        }
    };
    tokio::select! {
        _ = ctrl_c => info!("received SIGINT, shutting down"),
        _ = terminate => info!("received SIGTERM, shutting down"),
    }
}
