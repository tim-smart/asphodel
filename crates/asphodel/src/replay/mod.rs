//! `asphodel replay`: a scripted scenario or a real-history corpus on a
//! simulated clock ("Replay harness: simulated-clock replay of recorded
//! sessions", TIM-96, with its TIM-97, TIM-98 and TIM-116 amendments;
//! `docs/replay.md`).
//!
//! The command opens its own store under the private replay dir with
//! deterministic ids, builds the service layer on the fake models or the
//! real ones, layers the tuning (code defaults, the fake floors,
//! `--config`, a scenario's `[tuning]`, `--overrides`), and hands the
//! timeline to the [`engine`]. A scenario answers the LLM from its claims;
//! a corpus answers it through the cassette ([`cassette`]), in `live`,
//! `replay` or `fast` mode ([`history`]). The report goes to `--report` or
//! under `<replay dir>/reports/`, and the shadow table of purged rows to
//! `<replay dir>/shadow.db`.
//!
//! A run holds a lock on the private dir from before it touches the store
//! until the report is written, so two replays never share one. The store
//! carries a marker naming it replay's own; a `store` dir without one is
//! refused, never reset. Both outputs are checked before the run: neither
//! may already be a symlink, and a `--report` inside a git working tree is
//! refused. Everything derived from real history is refused outside the
//! private dir; only the `--aggregate` export may leave it.
//!
//! `--self-test` runs the simulation twice under the one lock and requires
//! the two reports to be byte-identical (TIM-96, decision 3).
//!
//! Exit 0 when every probe passed; 1 when one failed, with the report
//! written; 2 when the arguments or the scenario were refused, or the run
//! itself failed, with no report.

pub mod cassette;
pub mod corpus;
pub mod diff;
pub mod engine;
pub mod history;
pub mod import;
pub mod manifest;
pub mod probes;
pub mod report;
pub mod scenario;
pub mod shadow;
pub mod timeline;

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
use engine::{Engine, Failure, Llm, Settings};
use report::{Aggregate, Flags, Report};
use scenario::Group;
use timeline::Timeline;

/// Floors for the deterministic fakes, the layer group `ci` runs under
/// (ADR 0009): the reranker gate open, the reconcile floor where the unit
/// tests put it.
pub(crate) fn fake_floors() -> String {
    format!(
        "[injection.reranker_floors]\n{:?} = 0.0\n[reconcile.embedding_floors]\n{:?} = 0.5\n",
        FakeReranker::MODEL_ID,
        FakeEmbedder::MODEL_ID
    )
}

/// The marker in `<replay dir>/store` that says replay created it, so a
/// run may reset it, and `bench` may copy it.
pub(crate) const STORE_MARKER: &str = "replay-store";

/// The shadow table under the replay dir.
pub(crate) const SHADOW_FILE: &str = "shadow.db";

/// Replay never skips the reranker (TIM-96, decision 3).
pub(crate) const NO_DEADLINE: Duration = Duration::from_secs(365 * 24 * 60 * 60);

/// Runs the command and exits with the documented code.
pub fn run(args: ReplayArgs) -> anyhow::Result<()> {
    let result = if args.corpus.is_some() {
        history::execute(&args)
    } else {
        execute(&args)
    };
    match result {
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

pub(crate) struct Finished {
    failed: usize,
    report: PathBuf,
}

/// A scripted scenario.
fn execute(args: &ReplayArgs) -> anyhow::Result<Finished> {
    let path = args
        .scenario
        .as_deref()
        .ok_or_else(|| anyhow!("--scenario is required"))?;
    if args.mode.is_some()
        || args.cassette.is_some()
        || args.probes.is_some()
        || args.refresh.is_some()
        || args.no_cache
    {
        bail!(
            "--mode, --cassette, --probes, --refresh and --no-cache go with --corpus, not --scenario"
        );
    }
    let scenario = scenario::load(path)?;
    let dir = private_dir(args.replay_dir.as_deref())?;
    let _lock = lock_private_dir(&dir)?;
    let report_path = report_path(args.report.as_deref(), &dir, &scenario.name)?;
    let aggregate_path = args
        .aggregate
        .as_deref()
        .map(|path| aggregate_path(path, &dir))
        .transpose()?;
    let shadow_path = dir.join(SHADOW_FILE);
    refuse_symlink(&shadow_path)?;

    let (models, fake) = match scenario.group {
        Group::Ci => (Models::fake(), true),
        Group::Models => (
            load_models(args.model_dir.as_deref(), NonZeroUsize::new(1)).context(
                "a scenario in group `models` needs the real models in ASPHODEL_MODEL_DIR",
            )?,
            false,
        ),
    };
    let tuning = layered_tuning(args, scenario.tuning.as_ref(), &scenario.name, fake)?;
    let latency = match args.latency.as_deref().or(scenario.latency.as_deref()) {
        Some(text) => scenario::duration(text).map_err(|error| anyhow!("--latency: {error}"))?,
        None => jiff::SignedDuration::ZERO,
    };
    let timeline = Timeline::from_scenario(&scenario)
        .map_err(|message| failure(Failure::Scenario(message)))?;
    let start = timeline
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
    let identity = BankIdentity {
        owner_name: bank.and_then(|bank| bank.owner.clone()),
        owner_platform_ids: bank
            .map(|bank| bank.owner_platform_ids.clone())
            .unwrap_or_default(),
        assistant_name: bank.and_then(|bank| bank.assistant.clone()),
        timezone: Some(timezone_name.clone()),
    };

    deliver(args, &report_path, aggregate_path.as_deref(), || {
        let clock = Arc::new(SimulatedClock::new(start));
        let store = open_store(&dir, Arc::clone(&clock) as Arc<dyn Clock>)?;
        // Replay records the fingerprint on its own store and never pauses
        // purge (TIM-98 amendment).
        store.check_fingerprint(&tuning.deletion_fingerprint())?;
        let service = Service::with_models(
            Arc::clone(&clock) as Arc<dyn Clock>,
            store,
            tuning.clone(),
            clone_models(&models),
        )?
        .with_reranker_deadline(NO_DEADLINE);
        service.ensure_bank_with_models(&bank_name, &identity)?;
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
            timezone: timezone.clone(),
            latency,
            latency_from_cassette: false,
            until: args.until,
        };
        let engine = Engine::new(
            &service,
            Arc::clone(&clock),
            &timeline,
            &tuning,
            settings,
            Llm::Scripted,
        )
        .map_err(failure)?;
        let outcome = engine.run().map_err(failure)?;
        let purged_then_re_mentioned = write_shadow(&service, &tuning, &shadow_path, &outcome)?;
        Ok(Report {
            kind: "scripted",
            scenario: scenario.name.clone(),
            group: scenario.group.as_str(),
            version: VERSION,
            git_sha: option_env!("ASPHODEL_GIT_SHA"),
            corpus_hash: None,
            cassette_hash: None,
            tuning: tuning.clone(),
            flags: Flags {
                latency_ms: u64::try_from(latency.as_millis()).unwrap_or(0),
                until: args.until,
                refresh: "scripted",
                mode: None,
                no_cache: false,
                self_test: args.self_test,
                onnx_threads: (!fake).then_some(1),
            },
            probes: outcome.probes,
            purges_per_day: outcome.purges_per_day,
            purged_then_re_mentioned,
            fade_outs_per_week: outcome.fade_outs_per_week,
            bands_per_week: outcome.bands_per_week,
            extraction_lag: outcome.extraction_lag,
            refresh_calls_per_day: outcome.refresh_calls_per_day,
            injected_tokens: outcome.injected_tokens,
            profile_tokens: outcome.profile_tokens,
            call2_rate: outcome.call2_rate,
            agenda_lines_per_day: outcome.agenda_lines_per_day,
            significance_histogram: outcome.significance_histogram,
            kind_histogram: outcome.kind_histogram,
            memories: outcome.memories,
            llm: outcome.llm,
        })
    })
}

/// Writes the shadow table and computes the purged-then-re-mentioned rate
/// (TIM-97, decision 7).
pub(crate) fn write_shadow(
    service: &Service,
    tuning: &Tuning,
    shadow_path: &Path,
    outcome: &engine::Outcome,
) -> anyhow::Result<report::ReMentioned> {
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
    shadow::write(shadow_path, &outcome.shadow).context("writing the shadow table")?;
    Ok(shadow::re_mentioned(
        &outcome.created,
        &outcome.shadow,
        floor,
    ))
}

/// Runs the simulation (twice under `--self-test`), then writes the report
/// and the aggregate export.
pub(crate) fn deliver(
    args: &ReplayArgs,
    report_path: &Path,
    aggregate_path: Option<&Path>,
    mut once: impl FnMut() -> anyhow::Result<Report>,
) -> anyhow::Result<Finished> {
    let report = once()?;
    let json = encode(&report)?;
    if args.self_test {
        let again = once()?;
        let second = encode(&again)?;
        if json != second {
            bail!(
                "the determinism self-test failed: two runs of the same configuration wrote different reports ({} and {} bytes)",
                json.len(),
                second.len()
            );
        }
        info!("the determinism self-test passed: two runs wrote byte-identical reports");
    }
    let failed = report.probes.iter().filter(|probe| !probe.passed).count();
    write_file(report_path, &json)
        .with_context(|| format!("writing the report to {}", report_path.display()))?;
    if let Some(path) = aggregate_path {
        let mut json = serde_json::to_vec_pretty(&Aggregate::from_report(&report))?;
        json.push(b'\n');
        write_file(path, &json)
            .with_context(|| format!("writing the aggregate export to {}", path.display()))?;
    }
    Ok(Finished {
        failed,
        report: report_path.to_owned(),
    })
}

fn encode(report: &Report) -> anyhow::Result<Vec<u8>> {
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    Ok(json)
}

pub(crate) fn failure(failure: Failure) -> anyhow::Error {
    match failure {
        Failure::Scenario(message) => anyhow!("scenario error: {message}"),
        Failure::Internal(error) => error,
    }
}

/// The real models, with the pinned thread count (TIM-96, decision 3).
pub(crate) fn load_models(
    model_dir: Option<&Path>,
    threads: Option<NonZeroUsize>,
) -> anyhow::Result<Models> {
    let model_dir = crate::cli::resolve_model_dir(model_dir)?;
    Ok(Models::load(&model_dir, &ModelOptions { threads })?)
}

/// A second handle on the models, for a second run under one process.
pub(crate) fn clone_models(models: &Models) -> Models {
    Models {
        embedder: Arc::clone(&models.embedder),
        reranker: Arc::clone(&models.reranker),
    }
}

/// The private directory (TIM-96, decision 8): given, not in a git working
/// tree, and not a `serve` data dir.
pub(crate) fn private_dir(given: Option<&Path>) -> anyhow::Result<PathBuf> {
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
pub(crate) fn refuse_git_tree(dir: &Path) -> anyhow::Result<()> {
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
pub(crate) fn refuse_symlink(path: &Path) -> anyhow::Result<()> {
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
pub(crate) fn lock_private_dir(dir: &Path) -> anyhow::Result<DataDirLock> {
    refuse_symlink(&dir.join(LOCK_FILE))?;
    DataDirLock::acquire(dir).map_err(|error| match error {
        StoreError::Locked { holder, .. } => anyhow!(
            "another replay (pid {holder}) is running in {}; one replay owns a private dir at a time",
            dir.display()
        ),
        error => anyhow::Error::new(error).context(format!("locking {}", dir.display())),
    })
}

/// `path` resolved against its parent, which must exist: the parent
/// canonical, the file name kept.
fn resolve(path: &Path, what: &str) -> anyhow::Result<PathBuf> {
    refuse_symlink(path)?;
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow!("{what} {} doesn't name a file", path.display()))?;
    let parent = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    let parent = fs::canonicalize(parent)
        .with_context(|| format!("resolving {what}'s directory {}", parent.display()))?;
    Ok(parent.join(file_name))
}

/// Whether `path` is one of replay's own files under `dir`.
fn is_reserved(dir: &Path, path: &Path) -> bool {
    path == dir
        || path == dir.join(LOCK_FILE)
        || path.starts_with(dir.join("store"))
        || path.starts_with(dir.join("bench"))
        || [
            SHADOW_FILE,
            "shadow.db-journal",
            "shadow.db-wal",
            "shadow.db-shm",
        ]
        .iter()
        .any(|name| path == dir.join(name))
}

/// A file derived from real history (TIM-96, decision 8): it must be
/// under the private dir, and not one of replay's own files.
pub(crate) fn inside_private(dir: &Path, path: &Path, what: &str) -> anyhow::Result<PathBuf> {
    let path = resolve(path, what)?;
    if !path.starts_with(dir) {
        bail!(
            "{what} {} is outside the private directory {}; everything derived from real history stays inside it",
            path.display(),
            dir.display()
        );
    }
    if is_reserved(dir, &path) {
        bail!("{what} {} is a reserved replay destination", path.display());
    }
    Ok(path)
}

/// Where the report goes, checked before the run (TIM-96, decision 8):
/// `--report`, or `<replay dir>/reports/<name>.json`. A scripted report
/// may go outside the private dir, since it derives from a checked-in
/// fixture, but never inside a git working tree, through a symlink, or
/// over replay's private directory, lock, shadow table or store.
fn report_path(given: Option<&Path>, dir: &Path, name: &str) -> anyhow::Result<PathBuf> {
    let path = match given {
        Some(path) => path.to_owned(),
        None => default_report(dir, name)?,
    };
    let path = resolve(&path, "--report")?;
    refuse_git_tree(path.parent().unwrap_or(&path))?;
    if is_reserved(dir, &path) {
        bail!(
            "--report {} is a reserved replay destination",
            path.display()
        );
    }
    Ok(path)
}

/// `<replay dir>/reports/<name>.json`, its directory created.
pub(crate) fn default_report(dir: &Path, name: &str) -> anyhow::Result<PathBuf> {
    let reports = dir.join("reports");
    refuse_symlink(&reports)?;
    fs::create_dir_all(&reports).with_context(|| format!("creating {}", reports.display()))?;
    Ok(reports.join(format!("{name}.json")))
}

/// Where the aggregate export goes: anywhere but through a symlink or over
/// replay's own files, since it's the one thing that may leave the private
/// dir.
pub(crate) fn aggregate_path(given: &Path, dir: &Path) -> anyhow::Result<PathBuf> {
    let path = resolve(given, "--aggregate")?;
    if is_reserved(dir, &path) {
        bail!(
            "--aggregate {} is a reserved replay destination",
            path.display()
        );
    }
    Ok(path)
}

/// Writes through a fresh file in the same directory and renames it into
/// place, so a symlink planted at the destination since the check is
/// replaced rather than followed.
pub(crate) fn write_file(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write as _;

    let file_name = path.file_name().unwrap_or_default().to_string_lossy();
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let result = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .and_then(|mut file| file.write_all(bytes).and_then(|()| file.sync_all()))
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
pub(crate) fn open_store(dir: &Path, clock: Arc<dyn Clock>) -> anyhow::Result<Store> {
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

/// Code defaults, the fake floors (on the fakes), `--config`, a scenario's
/// `[tuning]`, then `--overrides` (TIM-98 amendment).
pub(crate) fn layered_tuning(
    args: &ReplayArgs,
    scenario_tuning: Option<&toml::Table>,
    scenario_name: &str,
    fake: bool,
) -> anyhow::Result<Tuning> {
    let mut layers: Vec<(String, String)> = Vec::new();
    if fake {
        layers.push(("the fake floors".into(), fake_floors()));
    }
    if let Some(path) = &args.config {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading --config {}", path.display()))?;
        layers.push((path.display().to_string(), text));
    }
    if let Some(table) = scenario_tuning {
        layers.push((
            format!("the scenario's [tuning] ({scenario_name})"),
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

/// Whether `ASPHODEL_MODELS=fake` asks for the deterministic fakes, which
/// replay honours as `serve` does, for tests only.
pub(crate) fn fake_models_requested() -> anyhow::Result<bool> {
    match std::env::var_os(crate::serve::MODELS_ENV) {
        None => Ok(false),
        Some(value) if value == "fake" => Ok(true),
        Some(value) => bail!(
            "{} is {:?}; replay takes only `fake`, for tests. Unset it to run the ONNX models",
            crate::serve::MODELS_ENV,
            value
        ),
    }
}
