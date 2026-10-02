//! `asphodel bench`: concurrent prefetches over HTTP against a daemon
//! started on a copy of the replayed store, to see how the reranker
//! deadline behaves under contention (TIM-96, decision 3; TIM-117).
//!
//! It never runs against the production daemon, because prefetch writes
//! the recall log and pending sets: the store is copied from
//! `<replay dir>/store`, which must carry replay's marker, into
//! `<replay dir>/bench/`, and the daemon on it listens on loopback only.
//! The queries are the corpus's prefetch queries, in order, so nothing
//! here comes from anywhere but the private dir. The report holds latency
//! percentiles and the reranked fraction per concurrency level, and goes
//! under the private dir.

use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context as _, anyhow, bail};
use asphodel_core::config::Layer;
use asphodel_core::retrieval::PrefetchRequest;
use asphodel_core::store::{DB_FILE, LOCK_FILE};
use asphodel_core::{Tuning, VERSION};
use serde::{Deserialize, Serialize};
use tokio::sync::{oneshot, watch};

use crate::cli::{BenchArgs, ClientArgs, ServeArgs};
use crate::client::Client;
use crate::listen::Listen;
use crate::replay::{self, STORE_MARKER, corpus};

/// The concurrency levels measured when none is given.
const DEFAULT_LEVELS: [usize; 3] = [1, 4, 16];

/// Prefetches per level when `--requests` isn't given.
const DEFAULT_REQUESTS: usize = 32;

/// How long the daemon may take to answer `/v1/health` with 200.
const READY_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Serialize)]
struct Report {
    kind: &'static str,
    version: &'static str,
    git_sha: Option<&'static str>,
    corpus_hash: String,
    requests: usize,
    queries: usize,
    levels: Vec<Level>,
}

#[derive(Debug, Serialize)]
struct Level {
    concurrency: usize,
    requests: usize,
    failed: usize,
    p50_ms: u64,
    p95_ms: u64,
    p99_ms: u64,
    max_ms: u64,
    /// The share of prefetches the reranker answered within its deadline.
    reranked_fraction: f64,
}

/// The one field of the prefetch reply the bench reads.
#[derive(Debug, Deserialize)]
struct Reranked {
    reranked: bool,
}

/// Runs the command: exit 0 with the report written, 2 when refused or
/// failed.
pub fn run(args: BenchArgs) -> anyhow::Result<()> {
    match execute(&args) {
        Ok(report) => {
            eprintln!("wrote {}", report.display());
            Ok(())
        }
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::exit(2)
        }
    }
}

fn execute(args: &BenchArgs) -> anyhow::Result<PathBuf> {
    let listen = args
        .listen
        .clone()
        .unwrap_or_else(|| "127.0.0.1:0".parse().expect("loopback parses"));
    if !listen.is_local() {
        bail!(
            "the bench daemon listens on loopback only, not {listen}: it serves a copy of real history"
        );
    }
    let dir = replay::private_dir(args.replay_dir.as_deref())?;
    let corpus_path = args
        .corpus
        .as_deref()
        .ok_or_else(|| anyhow!("--corpus is required: the prefetch queries are sampled from it"))?;
    let corpus_path = replay::inside_private(&dir, corpus_path, "the corpus")?;
    let corpus = corpus::load(&corpus_path)?;
    let queries: Vec<String> = corpus
        .events
        .iter()
        .filter_map(|event| match event {
            corpus::Event::Prefetch { query, .. } => Some(query.clone()),
            _ => None,
        })
        .collect();
    if queries.is_empty() {
        bail!(
            "the corpus {} has no prefetches to sample",
            corpus_path.display()
        );
    }
    let store = dir.join("store");
    if !store.join(STORE_MARKER).is_file() || !store.join(DB_FILE).is_file() {
        bail!(
            "{} holds no replayed store: bench runs only on a copy of a store `asphodel replay` made, so run a replay first",
            dir.display()
        );
    }
    let report_path = match &args.report {
        Some(path) => replay::inside_private(&dir, path, "the report")?,
        None => replay::default_report(&dir, "bench")?,
    };
    let levels: Vec<usize> = if args.concurrency.is_empty() {
        DEFAULT_LEVELS.to_vec()
    } else {
        args.concurrency.iter().map(|n| n.get()).collect()
    };
    let requests = args.requests.map_or(DEFAULT_REQUESTS, NonZeroUsize::get);

    let copy = copy_store(&dir, &store)?;
    let result = daemon_config(&copy).and_then(|config| {
        measure(
            &copy,
            config,
            listen,
            &corpus.header.bank,
            &queries,
            &levels,
            requests,
        )
    });
    let _ = fs::remove_dir_all(&copy);
    let levels = result?;

    let report = Report {
        kind: "bench",
        version: VERSION,
        git_sha: option_env!("ASPHODEL_GIT_SHA"),
        corpus_hash: corpus.hash,
        requests,
        queries: queries.len(),
        levels,
    };
    let mut json = serde_json::to_vec_pretty(&report)?;
    json.push(b'\n');
    replay::write_file(&report_path, &json)
        .with_context(|| format!("writing the bench report to {}", report_path.display()))?;
    Ok(report_path)
}

/// Copies the replayed store's database files into a fresh directory
/// under `<replay dir>/bench/`, reading the originals only.
fn copy_store(dir: &Path, store: &Path) -> anyhow::Result<PathBuf> {
    let bench = dir.join("bench");
    replay::refuse_symlink(&bench)?;
    fs::create_dir_all(&bench).with_context(|| format!("creating {}", bench.display()))?;
    let copy = bench.join(format!("store-{}", std::process::id()));
    if copy.exists() {
        fs::remove_dir_all(&copy).with_context(|| format!("clearing {}", copy.display()))?;
    }
    fs::create_dir(&copy).with_context(|| format!("creating {}", copy.display()))?;
    for suffix in ["", "-wal", "-shm"] {
        let name = format!("{DB_FILE}{suffix}");
        let from = store.join(&name);
        if from.is_file() {
            fs::copy(&from, copy.join(&name))
                .with_context(|| format!("copying {}", from.display()))?;
        }
    }
    let _ = LOCK_FILE;
    Ok(copy)
}

/// The daemon's tuning, resolved the way the replay resolved its own:
/// the code defaults, the fake floors when `ASPHODEL_MODELS=fake`, then
/// `ASPHODEL_CONFIG`; written under the copy so `serve` reads one file.
fn daemon_config(copy: &Path) -> anyhow::Result<PathBuf> {
    let fake = replay::fake_models_requested()?;
    let mut layers: Vec<(String, String)> = Vec::new();
    if fake {
        layers.push(("the fake floors".into(), replay::fake_floors()));
    }
    if let Some(path) = std::env::var_os("ASPHODEL_CONFIG") {
        let path = PathBuf::from(path);
        let text = fs::read_to_string(&path)
            .with_context(|| format!("reading ASPHODEL_CONFIG {}", path.display()))?;
        layers.push((path.display().to_string(), text));
    }
    let layers: Vec<Layer<'_>> = layers
        .iter()
        .map(|(origin, text)| Layer { origin, text })
        .collect();
    let tuning = Tuning::from_layers(&layers)?;
    let path = copy.join("bench-tuning.toml");
    fs::write(&path, toml::to_string(&tuning)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Starts the daemon on the copy, runs each level, and stops it.
fn measure(
    copy: &Path,
    config: PathBuf,
    listen: Listen,
    bank: &str,
    queries: &[String],
    levels: &[usize],
    requests: usize,
) -> anyhow::Result<Vec<Level>> {
    let (stop_tx, stop) = watch::channel(false);
    let (bound_tx, bound_rx) = oneshot::channel();
    let serve_args = ServeArgs {
        listen,
        data_dir: copy.to_owned(),
        config: Some(config),
        allow_network_fs: false,
        model_dir: std::env::var_os("ASPHODEL_MODEL_DIR").map(PathBuf::from),
        onnx_threads: None,
    };
    let daemon_stop = stop_tx.clone();
    let daemon = std::thread::spawn(move || -> anyhow::Result<()> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        runtime.block_on(crate::serve::run_with(
            serve_args,
            daemon_stop,
            stop,
            Some(bound_tx),
        ))
    });
    let address = match bound_rx.blocking_recv() {
        Ok(address) => address,
        Err(_) => {
            return Err(match daemon.join() {
                Ok(Err(error)) => error.context("the bench daemon didn't start"),
                _ => anyhow!("the bench daemon didn't start"),
            });
        }
    };
    let url = format!("http://{address}");
    let result = wait_ready(&url, &daemon).and_then(|()| {
        let mut out = Vec::new();
        for &concurrency in levels {
            out.push(level(&url, bank, queries, concurrency, requests)?);
        }
        Ok(out)
    });
    let _ = stop_tx.send(true);
    match daemon.join() {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            if result.is_ok() {
                return Err(error.context("the bench daemon failed"));
            }
        }
        Err(_) => bail!("the bench daemon panicked"),
    }
    result
}

fn client(url: &str) -> anyhow::Result<Client> {
    Client::new(&ClientArgs {
        url: url.to_string(),
        json: false,
        token: None,
    })
}

/// Polls `/v1/health` until it answers 200, or the daemon has stopped.
fn wait_ready(
    url: &str,
    daemon: &std::thread::JoinHandle<anyhow::Result<()>>,
) -> anyhow::Result<()> {
    let client = client(url)?;
    let started = Instant::now();
    loop {
        if client.get::<serde_json::Value>("/v1/health").is_ok() {
            return Ok(());
        }
        if daemon.is_finished() {
            bail!("the bench daemon at {url} stopped before it was ready");
        }
        if started.elapsed() > READY_TIMEOUT {
            bail!("the bench daemon at {url} didn't become ready in time");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// One concurrency level: `requests` prefetches shared by `concurrency`
/// threads, each with its own connection, on sessions of its own.
fn level(
    url: &str,
    bank: &str,
    queries: &[String],
    concurrency: usize,
    requests: usize,
) -> anyhow::Result<Level> {
    let next = Arc::new(AtomicUsize::new(0));
    let samples: Arc<Mutex<Vec<(u64, bool)>>> = Arc::new(Mutex::new(Vec::new()));
    let failed = Arc::new(AtomicUsize::new(0));
    let path = format!("/v1/banks/{bank}/prefetch");
    let mut workers = Vec::new();
    for worker in 0..concurrency {
        let next = Arc::clone(&next);
        let samples = Arc::clone(&samples);
        let failed = Arc::clone(&failed);
        let path = path.clone();
        let url = url.to_string();
        let queries = queries.to_vec();
        workers.push(std::thread::spawn(move || -> anyhow::Result<()> {
            let client = client(&url)?;
            loop {
                let index = next.fetch_add(1, Ordering::SeqCst);
                if index >= requests {
                    return Ok(());
                }
                let request = PrefetchRequest {
                    session_id: format!("bench:{concurrency}:{worker}"),
                    query: queries[index % queries.len()].clone(),
                    previous_query: None,
                    block_id: None,
                };
                let started = Instant::now();
                match client.post::<_, Reranked>(&path, &request) {
                    Ok(reply) => {
                        let elapsed =
                            u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                        samples
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .push((elapsed, reply.reranked));
                    }
                    Err(_) => {
                        failed.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        }));
    }
    for worker in workers {
        worker
            .join()
            .map_err(|_| anyhow!("a bench worker panicked"))??;
    }
    let samples = samples
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let mut latencies: Vec<u64> = samples.iter().map(|(ms, _)| *ms).collect();
    latencies.sort_unstable();
    let reranked = samples.iter().filter(|(_, reranked)| *reranked).count();
    let percentile = |p: f64| replay::report::percentile(&latencies, p);
    Ok(Level {
        concurrency,
        requests: samples.len(),
        failed: failed.load(Ordering::SeqCst),
        p50_ms: percentile(0.5),
        p95_ms: percentile(0.95),
        p99_ms: percentile(0.99),
        max_ms: latencies.last().copied().unwrap_or(0),
        reranked_fraction: if samples.is_empty() {
            0.0
        } else {
            reranked as f64 / samples.len() as f64
        },
    })
}
