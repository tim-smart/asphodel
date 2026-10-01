//! `asphodel serve`: the daemon.
//!
//! It builds the service layer on the system clock, binds the listen address
//! and serves the HTTP API under `/v1`. The handlers stay thin and call the
//! same service functions the replay harness does (TIM-96, decision 3). Only
//! `health` exists so far.

use std::sync::Arc;

use anyhow::{Context, bail};
use asphodel_core::{Health, Service, SystemClock};
use axum::{Json, Router, extract::State, http::StatusCode, routing::get};
use tokio::net::{TcpListener, UnixListener};
use tokio::signal;
use tracing::{info, warn};

use crate::cli::ServeArgs;
use crate::listen::Listen;

type Shared = Arc<Service>;

pub async fn run(args: ServeArgs) -> anyhow::Result<()> {
    if !args.listen.is_local() && args.token.as_deref().unwrap_or("").is_empty() {
        bail!(
            "listening on {} is reachable off this machine, so a bearer token is required: \
             set ASPHODEL_TOKEN or pass --token",
            args.listen
        );
    }
    if args.data_dir.is_none() {
        warn!("no --data-dir given; the store arrives in a later stage, so nothing is persisted");
    }

    let service: Shared = Arc::new(Service::new(Arc::new(SystemClock)));
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
            if path.exists() {
                std::fs::remove_file(path)
                    .with_context(|| format!("removing stale socket {}", path.display()))?;
            }
            let listener =
                UnixListener::bind(path).with_context(|| format!("binding {}", path.display()))?;
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
    }
    info!("asphodel stopped");
    Ok(())
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
/// decision 3); with no store yet, that is just a graceful HTTP shutdown.
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
