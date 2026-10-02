//! `asphodel replay`: a scripted scenario on a simulated clock ("Replay
//! harness: simulated-clock replay of recorded sessions", TIM-96, with its
//! TIM-97 and TIM-98 amendments; `docs/replay.md`).
//!
//! The command opens its own store under the private replay dir with
//! deterministic ids, builds the service layer on the fake models (group
//! `ci`) or the real ones (group `models`), layers the tuning (code
//! defaults, the fake floors, `--config`, the scenario's `[tuning]`,
//! `--overrides`), and hands the scenario to the [`engine`]. The report
//! goes to `--report` or `<replay dir>/reports/<name>.json`, and the shadow
//! table of purged rows to `<replay dir>/shadow.db`.
//!
//! A run holds a lock on the private dir from before it touches the store
//! until the report is written, so two replays never share one. The store
//! carries a marker naming it replay's own; a `store` dir without one is
//! refused, never reset. Both outputs are checked before the run: neither
//! may already be a symlink, and a `--report` inside a git working tree is
//! refused.
//!
//! Exit 0 when every probe passed; 1 when one failed, with the report
//! written; 2 when the arguments or the scenario were refused, or the run
//! itself failed, with no report.

pub mod engine;
pub mod report;
pub mod scenario;
pub mod shadow;

use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, anyhow, bail};
use asphodel_core::config::Layer;
use asphodel_core::models::{FakeEmbedder, FakeReranker, ModelOptions};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{DB_FILE, DataDirLock, LOCK_FILE, OpenOptions, StoreError};
use asphodel_core::{Clock, Models, Service, SimulatedClock, Store, Tuning, VERSION};
use jiff::tz::TimeZone;
use tracing::info;

use crate::cli::ReplayArgs;
use engine::{Engine, Failure, Settings};
use report::{Flags, LlmCounts, Report};
use scenario::{Group, Scenario};

/// Floors for the deterministic fakes, the layer group `ci` runs under
/// (ADR 0009): the reranker gate open, the reconcile floor where the unit
/// tests put it.
fn fake_floors() -> String {
    format!(
        "[injection.reranker_floors]\n{:?} = 0.0\n[reconcile.embedding_floors]\n{:?} = 0.5\n",
        FakeReranker::MODEL_ID,
        FakeEmbedder::MODEL_ID
    )
}

/// The marker in `<replay dir>/store` that says replay created it, so a
/// run may reset it.
const STORE_MARKER: &str = "replay-store";

/// The shadow table under the replay dir.
const SHADOW_FILE: &str = "shadow.db";

/// Replay never skips the reranker (TIM-96, decision 3).
const NO_DEADLINE: Duration = Duration::from_secs(365 * 24 * 60 * 60);

/// Runs the command and exits with the documented code.
pub fn run(args: ReplayArgs) -> anyhow::Result<()> {
    match execute(&args) {
        Ok(Finished { failed: 0, report }) => {
            info!(report = %report.display(), "every probe passed");
            Ok(())
        }
        Ok(Finished { failed, report }) => {
            eprintln!(
                "{failed} probe(s) failed; the report is at {}",
                report.display()
            );
            std::process::exit(1)
        }
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::exit(2)
        }
    }
}

struct Finished {
    failed: usize,
    report: PathBuf,
}

fn execute(args: &ReplayArgs) -> anyhow::Result<Finished> {
    let scenario = scenario::load(&args.scenario)?;
    let dir = private_dir(args.replay_dir.as_deref())?;
    let _lock = lock_private_dir(&dir)?;
    let report_path = report_path(args.report.as_deref(), &dir, &scenario.name)?;
    let shadow_path = dir.join(SHADOW_FILE);
    refuse_symlink(&shadow_path)?;

    let (models, fake) = match scenario.group {
        Group::Ci => (Models::fake(), true),
        Group::Models => {
            let model_dir = crate::cli::resolve_model_dir(args.model_dir.as_deref())?;
            let models = Models::load(
                &model_dir,
                &ModelOptions {
                    threads: NonZeroUsize::new(1),
                },
            )
            .context("a scenario in group `models` needs the real models in ASPHODEL_MODEL_DIR")?;
            (models, false)
        }
    };
    let tuning = layered_tuning(args, &scenario, fake)?;
    let latency = match args.latency.as_deref().or(scenario.latency.as_deref()) {
        Some(text) => scenario::duration(text).map_err(|error| anyhow!("--latency: {error}"))?,
        None => jiff::SignedDuration::ZERO,
    };
    let start = scenario
        .earliest()
        .ok_or_else(|| anyhow!("a scenario needs at least one event or probe"))?;

    let bank = scenario.bank.as_ref();
    let bank_name = bank
        .and_then(|bank| bank.name.clone())
        .unwrap_or_else(|| "main".into());
    let timezone_name = bank
        .and_then(|bank| bank.timezone.clone())
        .unwrap_or_else(|| "UTC".into());
    let timezone = TimeZone::get(&timezone_name)
        .with_context(|| format!("the bank's timezone {timezone_name:?}"))?;

    let clock = Arc::new(SimulatedClock::new(start));
    let store = open_store(&dir, Arc::clone(&clock) as Arc<dyn Clock>)?;
    // Replay records the fingerprint on its own store and never pauses
    // purge (TIM-98 amendment).
    store.check_fingerprint(&tuning.deletion_fingerprint())?;
    let service = Service::with_models(
        Arc::clone(&clock) as Arc<dyn Clock>,
        store,
        tuning.clone(),
        models,
    )?
    .with_reranker_deadline(NO_DEADLINE);
    service.ensure_bank_with_models(
        &bank_name,
        &BankIdentity {
            owner_name: bank.and_then(|bank| bank.owner.clone()),
            owner_platform_ids: bank
                .map(|bank| bank.owner_platform_ids.clone())
                .unwrap_or_default(),
            assistant_name: bank.and_then(|bank| bank.assistant.clone()),
            timezone: Some(timezone_name.clone()),
        },
    )?;
    for model in &scenario.models {
        service.create_model(
            &bank_name,
            &asphodel_core::mental_models::ModelSpec {
                name: model.name.clone(),
                question: model.question.clone(),
                kinds: model.kinds.clone(),
                entity: None,
                min_volatility: None,
                max_tokens: model.max_tokens,
                enabled: true,
            },
        )?;
    }

    let settings = Settings {
        bank: bank_name.clone(),
        timezone,
        latency,
        until: args.until,
    };
    let engine =
        Engine::new(&service, Arc::clone(&clock), &scenario, &tuning, settings).map_err(failure)?;
    let outcome = engine.run().map_err(failure)?;

    let embedder = service
        .models()
        .map(|models| models.embedder.model_id().to_string())
        .unwrap_or_default();
    let floor = tuning
        .reconcile
        .embedding_floors
        .get(&embedder)
        .copied()
        .unwrap_or(1.0);
    shadow::write(&shadow_path, &outcome.shadow).context("writing the shadow table")?;
    let purged_then_re_mentioned = shadow::re_mentioned(&outcome.created, &outcome.shadow, floor);

    let failed = outcome.probes.iter().filter(|probe| !probe.passed).count();
    let report = Report {
        kind: "scripted",
        scenario: scenario.name.clone(),
        group: scenario.group.as_str(),
        version: VERSION,
        git_sha: option_env!("ASPHODEL_GIT_SHA"),
        tuning,
        flags: Flags {
            latency_ms: u64::try_from(latency.as_millis()).unwrap_or(0),
            until: args.until,
            refresh: "scripted",
        },
        probes: outcome.probes,
        purges_per_day: outcome.purges_per_day,
        purged_then_re_mentioned,
        fade_outs_per_week: outcome.fade_outs_per_week,
        bands_per_week: outcome.bands_per_week,
        extraction_lag: outcome.extraction_lag,
        refresh_calls_per_day: outcome.refresh_calls_per_day,
        llm: LlmCounts {
            scripted: outcome.llm_calls,
            cache: 0,
            top_up: 0,
            live: 0,
        },
    };
    let mut json = serde_json::to_vec_pretty(&report)?;
    json.push(b'\n');
    write_report(&report_path, &json)
        .with_context(|| format!("writing the report to {}", report_path.display()))?;
    Ok(Finished {
        failed,
        report: report_path,
    })
}

fn failure(failure: Failure) -> anyhow::Error {
    match failure {
        Failure::Scenario(message) => anyhow!("scenario error: {message}"),
        Failure::Internal(error) => error,
    }
}

/// The private directory (TIM-96, decision 8): given, not in a git working
/// tree, and not a `serve` data dir.
fn private_dir(given: Option<&Path>) -> anyhow::Result<PathBuf> {
    let Some(dir) = given else {
        bail!("replay needs a private directory: set ASPHODEL_REPLAY_DIR or pass --replay-dir");
    };
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let dir = fs::canonicalize(dir).with_context(|| format!("resolving {}", dir.display()))?;
    refuse_git_tree(&dir)?;
    if dir.join(DB_FILE).exists() {
        bail!(
            "{} holds {DB_FILE} at its top level, which makes it a `serve` data dir; replay opens only its own store under <replay dir>/store",
            dir.display()
        );
    }
    Ok(dir)
}

/// Refuses `dir` when it or an ancestor holds `.git`.
fn refuse_git_tree(dir: &Path) -> anyhow::Result<()> {
    let mut ancestor = Some(dir);
    while let Some(path) = ancestor {
        if path.join(".git").exists() {
            bail!(
                "{} is inside a git working tree ({}); replay never writes inside one",
                dir.display(),
                path.display()
            );
        }
        ancestor = path.parent();
    }
    Ok(())
}

/// Refuses an output path that is already a symlink, which would write
/// through to wherever it points.
fn refuse_symlink(path: &Path) -> anyhow::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => bail!(
            "{} is a symlink; replay never writes through one",
            path.display()
        ),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("checking {}", path.display())),
    }
}

/// The lock on the private dir, held for the whole run: from before the
/// store is reset until the report is written.
fn lock_private_dir(dir: &Path) -> anyhow::Result<DataDirLock> {
    refuse_symlink(&dir.join(LOCK_FILE))?;
    DataDirLock::acquire(dir).map_err(|error| match error {
        StoreError::Locked { holder, .. } => anyhow!(
            "another replay (pid {holder}) is running in {}; one replay owns a private dir at a time",
            dir.display()
        ),
        error => anyhow::Error::new(error).context(format!("locking {}", dir.display())),
    })
}

/// Where the report goes, checked before the run (TIM-96, decision 8):
/// `--report`, or `<replay dir>/reports/<name>.json`. A scripted report
/// may go outside the private dir, since it derives from a checked-in
/// fixture, but never inside a git working tree, through a symlink, or
/// over replay's private directory, lock, shadow table or store.
fn report_path(given: Option<&Path>, dir: &Path, name: &str) -> anyhow::Result<PathBuf> {
    let path = match given {
        Some(path) => path.to_owned(),
        None => {
            let reports = dir.join("reports");
            refuse_symlink(&reports)?;
            fs::create_dir_all(&reports)
                .with_context(|| format!("creating {}", reports.display()))?;
            reports.join(format!("{name}.json"))
        }
    };
    refuse_symlink(&path)?;
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow!("--report {} doesn't name a file", path.display()))?;
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let parent = fs::canonicalize(parent)
        .with_context(|| format!("resolving the report's directory {}", parent.display()))?;
    refuse_git_tree(&parent)?;
    let path = parent.join(file_name);
    if path == dir
        || path == dir.join(LOCK_FILE)
        || path.starts_with(dir.join("store"))
        || [
            SHADOW_FILE,
            "shadow.db-journal",
            "shadow.db-wal",
            "shadow.db-shm",
        ]
        .iter()
        .any(|name| path == dir.join(name))
    {
        bail!(
            "--report {} is a reserved replay destination",
            path.display()
        );
    }
    Ok(path)
}

/// Writes the report through a fresh file in the same directory and
/// renames it into place, so a symlink planted at the destination since
/// the check is replaced rather than followed.
fn write_report(path: &Path, json: &[u8]) -> anyhow::Result<()> {
    use std::io::Write as _;

    let file_name = path.file_name().unwrap_or_default().to_string_lossy();
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let result = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .and_then(|mut file| file.write_all(json).and_then(|()| file.sync_all()))
        .and_then(|()| fs::rename(&temporary, path));
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    Ok(result?)
}

/// A fresh store under `<replay dir>/store`, with deterministic ids. The
/// caller holds the private dir's lock. A run starts from nothing, so an
/// earlier run's store is reset first: only one carrying replay's marker,
/// and under the store's own lock, whose file is kept so its inode still
/// excludes anyone who opened it.
fn open_store(dir: &Path, clock: Arc<dyn Clock>) -> anyhow::Result<Store> {
    let store_dir = dir.join("store");
    refuse_symlink(&store_dir)?;
    let marked = if store_dir.exists() {
        reset_store(&store_dir)?
    } else {
        fs::create_dir(&store_dir).with_context(|| format!("creating {}", store_dir.display()))?;
        false
    };
    if !marked {
        // `create_new` never follows a symlink at the marker's path.
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(store_dir.join(STORE_MARKER))
            .with_context(|| format!("marking {}", store_dir.display()))?;
    }
    Ok(Store::open(
        &store_dir,
        OpenOptions {
            allow_network_fs: false,
            deterministic_ids: true,
        },
        clock,
    )?)
}

/// Empties an earlier run's store, keeping its lock file and marker, and
/// says whether it was marked. A `store` dir without the marker isn't
/// replay's and is left as it is, unless it's empty.
fn reset_store(store_dir: &Path) -> anyhow::Result<bool> {
    let marked =
        fs::symlink_metadata(store_dir.join(STORE_MARKER)).is_ok_and(|metadata| metadata.is_file());
    if !marked {
        let empty = fs::read_dir(store_dir)
            .with_context(|| format!("reading {}", store_dir.display()))?
            .next()
            .is_none();
        if !empty {
            bail!(
                "{} wasn't created by replay (no {STORE_MARKER} marker); replay never resets a store it didn't create",
                store_dir.display()
            );
        }
        return Ok(false);
    }
    let _lock = DataDirLock::acquire(store_dir).map_err(|error| match error {
        StoreError::Locked { holder, .. } => anyhow!(
            "the store {} is held by another process (pid {holder})",
            store_dir.display()
        ),
        error => anyhow::Error::new(error).context(format!("locking {}", store_dir.display())),
    })?;
    for entry in
        fs::read_dir(store_dir).with_context(|| format!("reading {}", store_dir.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        if name == LOCK_FILE || name == STORE_MARKER {
            continue;
        }
        let path = entry.path();
        let removed = if entry.file_type()?.is_dir() {
            fs::remove_dir_all(&path)
        } else {
            fs::remove_file(&path)
        };
        removed.with_context(|| format!("clearing {}", path.display()))?;
    }
    Ok(true)
}

/// Code defaults, the fake floors (group `ci`), `--config`, the scenario's
/// `[tuning]`, then `--overrides` (TIM-98 amendment).
fn layered_tuning(args: &ReplayArgs, scenario: &Scenario, fake: bool) -> anyhow::Result<Tuning> {
    let mut layers: Vec<(String, String)> = Vec::new();
    if fake {
        layers.push(("the fake floors".into(), fake_floors()));
    }
    if let Some(path) = &args.config {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading --config {}", path.display()))?;
        layers.push((path.display().to_string(), text));
    }
    if let Some(table) = &scenario.tuning {
        layers.push((
            format!("the scenario's [tuning] ({})", scenario.name),
            toml::to_string(table)?,
        ));
    }
    if let Some(path) = &args.overrides {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading --overrides {}", path.display()))?;
        layers.push((path.display().to_string(), text));
    }
    let layers: Vec<Layer<'_>> = layers
        .iter()
        .map(|(origin, text)| Layer { origin, text })
        .collect();
    Ok(Tuning::from_layers(&layers)?)
}
