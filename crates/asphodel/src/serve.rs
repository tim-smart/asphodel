//! `asphodel serve`: the daemon.
//!
//! It builds the service layer on the system clock, binds the listen address
//! and serves the HTTP API under `/v1`. The handlers stay thin and call the
//! same service functions the replay harness does (TIM-96, decision 3). Only
//! `health` exists so far.

use std::{
    fs::{self, Metadata},
    io::ErrorKind,
    os::unix::fs::{FileTypeExt, MetadataExt},
    path::{Path, PathBuf},
    sync::{Arc, Weak},
    time::Duration,
};

use anyhow::{Context, bail};
use asphodel_core::config::{Deployment, LLM_API_KEY_ENV, Secret, TOKEN_ENV};
use asphodel_core::store::{OpenOptions, Store};
use asphodel_core::{Clock, Health, ResolvedConfig, Service, SystemClock, Tuning};
use axum::{Json, Router, extract::State, http::StatusCode, routing::get};
use tokio::net::{TcpListener, UnixListener, UnixStream};
use tokio::signal;
use tracing::{info, warn};

use crate::cli::ServeArgs;
use crate::listen::Listen;

type Shared = Arc<Service>;

/// The longest the daemon waits between housekeeping passes. It normally
/// wakes at the next copy's deadline (ADR 0010); this cap bounds how late
/// that wake can be after a host suspend, which tokio's monotonic timer
/// doesn't count, and is the retry delay after a failed pass.
const HOUSEKEEPING_INTERVAL: Duration = Duration::from_secs(60 * 60);

/// The shortest wait between passes. The timer is monotonic and the
/// deadline is on the service's clock, so a wake can land just short of it;
/// this keeps that from becoming a run of near-zero waits.
const HOUSEKEEPING_MIN_WAIT: Duration = Duration::from_secs(1);

#[cfg(test)]
mod tests;

pub async fn run(args: ServeArgs) -> anyhow::Result<()> {
    let mut config = resolve_config(&args)?;
    if !args.listen.is_local() && config.deployment.token.is_none() {
        bail!(
            "listening on {} is reachable off this machine, so a bearer token is required: \
             set {TOKEN_ENV}",
            args.listen
        );
    }

    // The store is opened before anything listens: the lock, the filesystem
    // check and the migrations all have to pass before the daemon exists to
    // a client (TIM-94, decision 4; ADR 0010).
    let clock: Arc<dyn Clock> = Arc::new(SystemClock);
    let options = OpenOptions {
        allow_network_fs: args.allow_network_fs,
    };
    let store = Store::open(&args.data_dir, options, Arc::clone(&clock))
        .with_context(|| format!("opening the store in {}", args.data_dir.display()))?;
    config.purge = store.check_fingerprint(&config.deletion_fingerprint)?;
    let service = Service::open(clock, store, config.tuning.clone());
    info!(
        config = %serde_json::to_string(&config)?,
        "resolved config"
    );
    let service: Shared = Arc::new(service);
    tokio::spawn(housekeeping(Arc::downgrade(&service)));
    let app = Router::new()
        .route("/v1/health", get(health))
        .with_state(service);

    info!(version = asphodel_core::VERSION, listen = %args.listen, "asphodel listening");
    match &args.listen {
        Listen::Tcp(addr) => {
            let listener = TcpListener::bind(addr)
                .await
                .with_context(|| format!("binding {addr}"))?;
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
        Listen::Unix(path) => {
            let (listener, cleanup) = bind_unix(path).await?;
            let result = axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await;
            cleanup.remove()?;
            result?;
        }
    }
    info!("asphodel stopped");
    Ok(())
}

/// Loads the tuning file and records the deployment. An unreadable file,
/// an unknown key or an out-of-range value stops the daemon starting.
fn resolve_config(args: &ServeArgs) -> anyhow::Result<ResolvedConfig> {
    let tuning = Tuning::load(args.config.as_deref())?;
    let deployment = Deployment {
        listen: args.listen.to_string(),
        data_dir: Some(args.data_dir.clone()),
        config: args.config.clone(),
        allow_network_fs: args.allow_network_fs,
        model_dir: args.model_dir.clone(),
        token: Secret::from_env(TOKEN_ENV),
        llm_api_key: Secret::from_env(LLM_API_KEY_ENV),
    };
    Ok(ResolvedConfig::new(tuning, deployment))
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

/// Runs [`Service::housekeeping`] at each pass's `next_due`, or after
/// [`HOUSEKEEPING_INTERVAL`] if that comes first, until the service is
/// dropped. It holds the service weakly, and never across a wait, so the
/// store still closes, and checkpoints, when `run` returns.
async fn housekeeping(service: Weak<Service>) {
    loop {
        let Some(service) = service.upgrade() else {
            return;
        };
        // The first pass runs at once: open deletes what has expired but
        // doesn't say when the next copy is due.
        let pass = tokio::task::spawn_blocking(move || {
            let result = service.housekeeping();
            (result, service.now())
        })
        .await;
        let wait = match pass {
            Ok((Ok(done), now)) => done.next_due.map_or(HOUSEKEEPING_INTERVAL, |due| {
                // A deadline already past converts to nothing: wait the
                // minimum and run again.
                Duration::try_from(now.duration_until(due))
                    .unwrap_or(Duration::ZERO)
                    .clamp(HOUSEKEEPING_MIN_WAIT, HOUSEKEEPING_INTERVAL)
            }),
            Ok((Err(error), _)) => {
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

/// `GET /v1/health`: 503 until the daemon is ready, then 200 with the version.
async fn health(State(service): State<Shared>) -> (StatusCode, Json<Health>) {
    let health = service.health();
    let status = if health.ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(health))
}

/// Resolves on SIGINT or SIGTERM. On SIGTERM the daemon stops accepting
/// ingest, finishes the chunk in flight and checkpoints the WAL (TIM-94,
/// decision 3). There is no ingest yet, so it is a graceful HTTP shutdown
/// and the store checkpoints as it closes.
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
