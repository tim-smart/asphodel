//! A durable attempt budget for a real-history replay's live calls:
//! `asphodel replay-budget init|show` and `asphodel replay --attempt-budget
//! <ledger> --budget-run <id>` (`docs/replay.md`, "Attempt budgets").
//!
//! The ledger is one JSON file with every stage's and run's cap and count.
//! `init` creates it and its lock file `<ledger>.lock` from a frozen
//! allocations file, and is the only way a ledger comes into being.
//!
//! Every post to the model a budgeted run makes is admitted first, retries,
//! priming workers and the ChatGPT client's post after a 401 included,
//! since admission sits beneath the retry layer: in [`Budgeted`], or in
//! the ChatGPT client itself, which can post twice in one call. Admission takes an exclusive lock on `<ledger>.lock`, reads the
//! ledger and checks it, takes one slot from the run and its stage, writes
//! the whole ledger to `<ledger>.tmp`, syncs it, renames it over the ledger
//! and syncs the directory. Only then does the call go out. A crash before
//! the rename leaves the last ledger standing with nothing sent; after it,
//! the slot stays taken whether or not the call went out. Nothing is ever
//! refunded, and a call the cassette answers never gets this far.
//!
//! Anything that makes the ledger untrustworthy stops the run's calls for
//! good: a missing ledger or lock file, one that doesn't parse, counts
//! that don't add up, a run bound to another binary, configuration,
//! corpus or cassette, or a failed lock, write or sync. So does running
//! out. The ledger names the cassette a run is bound to, so it stays in the
//! private dir; `show` prints ids and numbers only.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context as _, anyhow, bail};
use asphodel_core::models::{Admission, LlmClient, LlmError, LlmRequest, LlmResponse};
use asphodel_core::{Clock as _, SystemClock};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::cli::{ReplayBudgetCommand, ReplayBudgetInitArgs, ReplayBudgetShowArgs};

/// The ledger's format.
const FORMAT: u32 = 1;

/// The backend code a refused attempt fails with. No retry policy knows it,
/// so it's never retried.
const REFUSED_CODE: &str = "attempt_budget";

pub fn run(command: ReplayBudgetCommand) -> anyhow::Result<()> {
    let result = match command {
        ReplayBudgetCommand::Init(args) => init(&args),
        ReplayBudgetCommand::Show(args) => show(&args),
    };
    match result {
        Ok(json) => {
            println!("{}", serde_json::to_string_pretty(&json)?);
            Ok(())
        }
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::exit(2)
        }
    }
}

/// The allocations file: stages, and runs that each draw on one stage.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Allocations {
    #[serde(default)]
    stage: Vec<StageAllocation>,
    #[serde(default)]
    run: Vec<RunAllocation>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StageAllocation {
    id: String,
    cap: u64,
}

/// A run with no cap of its own draws on its stage's alone.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RunAllocation {
    id: String,
    stage: String,
    cap: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Ledger {
    format: u32,
    ledger: Uuid,
    allocation_sha256: String,
    /// One more on every commit.
    seq: u64,
    stages: BTreeMap<String, Stage>,
    runs: BTreeMap<String, Run>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stage {
    cap: u64,
    consumed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Run {
    stage: String,
    cap: Option<u64>,
    consumed: u64,
    by_template: BTreeMap<String, u64>,
    binding: Option<Binding>,
}

/// What a run is bound to by its first invocation. Every later one must
/// match it exactly, so a new invocation is never a new budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub executable_sha256: String,
    pub config_sha256: Option<String>,
    pub overrides_sha256: Option<String>,
    pub corpus_hash: String,
    /// The cassette's resolved path: the reason the ledger stays private.
    pub cassette: String,
    pub mode: String,
    pub refresh: String,
    pub prime_concurrency: Option<usize>,
    pub until: Option<String>,
}

impl Ledger {
    /// The checks every load makes. Any failure means the counts can't be
    /// trusted.
    fn check(&self) -> anyhow::Result<()> {
        if self.format != FORMAT {
            bail!("the ledger's format is {}, not {FORMAT}", self.format);
        }
        let mut drawn: BTreeMap<&str, u64> = BTreeMap::new();
        let mut capped: BTreeMap<&str, u64> = BTreeMap::new();
        for (id, run) in &self.runs {
            if !self.stages.contains_key(&run.stage) {
                bail!(
                    "run {id:?} draws on stage {:?}, which isn't in the ledger",
                    run.stage
                );
            }
            let by_template = run
                .by_template
                .values()
                .try_fold(0u64, |sum, count| sum.checked_add(*count));
            if by_template != Some(run.consumed) {
                bail!(
                    "run {id:?}'s counts by template don't add up to its {}",
                    run.consumed
                );
            }
            if let Some(cap) = run.cap {
                if run.consumed > cap {
                    bail!("run {id:?} has spent {} of a cap of {cap}", run.consumed);
                }
                *capped.entry(&run.stage).or_default() += cap;
            }
            *drawn.entry(&run.stage).or_default() += run.consumed;
        }
        for (id, stage) in &self.stages {
            if drawn.get(id.as_str()).copied().unwrap_or(0) != stage.consumed {
                bail!("stage {id:?}'s count doesn't match its runs'");
            }
            if stage.consumed > stage.cap {
                bail!(
                    "stage {id:?} has spent {} of a cap of {}",
                    stage.consumed,
                    stage.cap
                );
            }
            if capped.get(id.as_str()).copied().unwrap_or(0) > stage.cap {
                bail!("stage {id:?}'s runs are allocated more than its cap");
            }
        }
        Ok(())
    }
}

/// The paths beside a ledger.
fn beside(ledger: &Path, suffix: &str) -> PathBuf {
    let mut name = ledger.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn directory(ledger: &Path) -> &Path {
    match ledger.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// `replay-budget init`: the ledger and its lock file, both created
/// exclusively, from the allocations file.
fn init(args: &ReplayBudgetInitArgs) -> anyhow::Result<Value> {
    let bytes = fs::read(&args.allocations)
        .with_context(|| format!("reading {}", args.allocations.display()))?;
    let text = std::str::from_utf8(&bytes).context("the allocations file isn't UTF-8")?;
    let allocations: Allocations = toml::from_str(text).context("parsing the allocations file")?;

    let mut stages = BTreeMap::new();
    for stage in allocations.stage {
        let fresh = Stage {
            cap: stage.cap,
            consumed: 0,
        };
        if stages.insert(stage.id.clone(), fresh).is_some() {
            bail!("stage {:?} is allocated twice", stage.id);
        }
    }
    let mut runs = BTreeMap::new();
    for run in allocations.run {
        let Some(stage) = stages.get(&run.stage) else {
            bail!(
                "run {:?} draws on stage {:?}, which isn't allocated",
                run.id,
                run.stage
            );
        };
        if run.cap.is_some_and(|cap| cap > stage.cap) {
            bail!("run {:?}'s cap is more than its stage's", run.id);
        }
        let fresh = Run {
            stage: run.stage,
            cap: run.cap,
            consumed: 0,
            by_template: BTreeMap::new(),
            binding: None,
        };
        if runs.insert(run.id.clone(), fresh).is_some() {
            bail!("run {:?} is allocated twice", run.id);
        }
    }
    let allocation_sha256 = sha(&bytes);
    let now = SystemClock.now();
    let ledger = Ledger {
        format: FORMAT,
        ledger: Uuid::new_v5(
            &Uuid::NAMESPACE_OID,
            format!("{allocation_sha256} {} {now}", args.ledger.display()).as_bytes(),
        ),
        allocation_sha256,
        seq: 0,
        stages,
        runs,
    };
    ledger.check()?;

    let path = &args.ledger;
    let lock = beside(path, ".lock");
    if fs::symlink_metadata(path).is_ok() {
        bail!(
            "{} already exists; a ledger is never replaced",
            path.display()
        );
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock)
        .with_context(|| format!("creating the lock file {}", lock.display()))?
        .sync_all()
        .context("syncing the lock file")?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating {}", path.display()))?;
    file.write_all(&serde_json::to_vec_pretty(&ledger)?)
        .and_then(|()| file.sync_all())
        .with_context(|| format!("writing {}", path.display()))?;
    sync_directory(path)?;
    Ok(shown(&ledger))
}

/// `replay-budget show`: the ledger's ids and numbers, read under its lock
/// and checked as admission checks it.
fn show(args: &ReplayBudgetShowArgs) -> anyhow::Result<Value> {
    let _lock = lock(&args.ledger)?;
    let ledger = load(&args.ledger)?;
    Ok(shown(&ledger))
}

/// What `show` prints: no binding, since it names the cassette.
fn shown(ledger: &Ledger) -> Value {
    let stages: BTreeMap<&str, Value> = ledger
        .stages
        .iter()
        .map(|(id, stage)| {
            (
                id.as_str(),
                json!({"cap": stage.cap, "consumed": stage.consumed}),
            )
        })
        .collect();
    let runs: BTreeMap<&str, Value> = ledger
        .runs
        .iter()
        .map(|(id, run)| {
            (
                id.as_str(),
                json!({
                    "stage": run.stage,
                    "cap": run.cap,
                    "consumed": run.consumed,
                    "by_template": run.by_template,
                    "bound": run.binding.is_some(),
                }),
            )
        })
        .collect();
    json!({
        "ledger": ledger.ledger,
        "allocation_sha256": ledger.allocation_sha256,
        "seq": ledger.seq,
        "stages": stages,
        "runs": runs,
    })
}

/// The exclusive lock on `<ledger>.lock`, held until the file drops. The
/// lock file must exist: only `init` creates it. The ledger itself is never
/// locked, since every commit replaces it.
fn lock(ledger: &Path) -> anyhow::Result<File> {
    let path = beside(ledger, ".lock");
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .with_context(|| format!("opening the ledger's lock file {}", path.display()))?;
    file.lock()
        .with_context(|| format!("locking {}", path.display()))?;
    Ok(file)
}

fn load(path: &Path) -> anyhow::Result<Ledger> {
    let bytes = fs::read(path).with_context(|| format!("reading the ledger {}", path.display()))?;
    let ledger: Ledger = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing the ledger {}", path.display()))?;
    ledger.check()?;
    Ok(ledger)
}

/// Replaces the ledger with `ledger`: written whole to `<ledger>.tmp`,
/// synced, renamed over it, and the directory synced. The caller holds the
/// lock.
fn commit(path: &Path, ledger: &Ledger) -> anyhow::Result<()> {
    let tmp = beside(path, ".tmp");
    let mut file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    file.write_all(&serde_json::to_vec_pretty(ledger)?)
        .and_then(|()| file.sync_all())
        .with_context(|| format!("writing {}", tmp.display()))?;
    drop(file);
    fs::rename(&tmp, path)
        .with_context(|| format!("replacing {} with {}", path.display(), tmp.display()))?;
    sync_directory(path)
}

fn sync_directory(path: &Path) -> anyhow::Result<()> {
    let dir = directory(path);
    File::open(dir)
        .and_then(|dir| dir.sync_all())
        .with_context(|| format!("syncing the directory {}", dir.display()))
}

/// A run's budget: admits each attempt against the ledger, or refuses.
pub struct Budget {
    ledger: PathBuf,
    run: String,
    binding: Binding,
    /// Serializes this process's admissions; the file lock serializes them
    /// with every other process.
    gate: Mutex<()>,
    /// Why the run's calls stopped, once they have. Set once, never cleared.
    stopped: OnceLock<String>,
}

impl Budget {
    /// Opens `run`'s budget in `ledger`, binding the run on its first
    /// invocation and refusing one bound to anything else. Every check
    /// admission makes is made here first, so a ledger that can't be
    /// trusted stops the run before it starts.
    pub fn open(ledger: &Path, run: &str, binding: Binding) -> anyhow::Result<Arc<Self>> {
        let budget = Self {
            ledger: ledger.to_owned(),
            run: run.to_owned(),
            binding,
            gate: Mutex::new(()),
            stopped: OnceLock::new(),
        };
        budget.transact(|_, _| Ok(()))?;
        Ok(Arc::new(budget))
    }

    /// Why the run's calls stopped, if they have.
    pub fn stopped(&self) -> Option<&str> {
        self.stopped.get().map(String::as_str)
    }

    /// Takes one slot for an attempt at `template`, committed before this
    /// returns. Once anything has gone wrong, refuses every attempt.
    fn take_slot(&self, template: &str) -> Result<(), String> {
        if let Some(reason) = self.stopped() {
            return Err(reason.to_owned());
        }
        self.transact(|run, stage| {
            if let Some(cap) = run.cap
                && run.consumed >= cap
            {
                bail!("run {:?} has spent its budget of {cap} attempts", self.run);
            }
            if stage.consumed >= stage.cap {
                bail!(
                    "stage {:?} has spent its budget of {} attempts",
                    run.stage,
                    stage.cap
                );
            }
            run.consumed += 1;
            stage.consumed += 1;
            *run.by_template.entry(template.to_owned()).or_default() += 1;
            Ok(())
        })
        .map_err(|_| self.stopped().unwrap_or_default().to_owned())
    }

    /// Under the lock: loads and checks the ledger, binds the run or checks
    /// its binding, applies `change` to the run and its stage, and commits
    /// whenever anything changed. Any failure stops the run's calls.
    fn transact(
        &self,
        change: impl FnOnce(&mut Run, &mut Stage) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let _gate = self
            .gate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let result = (|| {
            let _lock = lock(&self.ledger)?;
            let mut ledger = load(&self.ledger)?;
            let before = serde_json::to_vec(&ledger)?;
            let run = ledger
                .runs
                .get_mut(&self.run)
                .ok_or_else(|| anyhow!("run {:?} isn't allocated in the ledger", self.run))?;
            match &run.binding {
                None => run.binding = Some(self.binding.clone()),
                Some(bound) if *bound == self.binding => {}
                Some(bound) => bail!(
                    "run {:?} is bound to a different {}",
                    self.run,
                    mismatch(bound, &self.binding)
                ),
            }
            let stage = ledger
                .stages
                .get_mut(&run.stage)
                .ok_or_else(|| anyhow!("run {:?}'s stage isn't in the ledger", self.run))?;
            change(run, stage)?;
            if serde_json::to_vec(&ledger)? != before {
                ledger.seq += 1;
                ledger.check()?;
                commit(&self.ledger, &ledger)?;
            }
            Ok(())
        })();
        if let Err(error) = &result {
            let reason = format!("the attempt budget stopped the run's calls: {error:#}");
            let _ = self.stopped.set(reason);
        }
        result
    }
}

/// What differs between two bindings, by field name only.
fn mismatch(bound: &Binding, now: &Binding) -> String {
    let fields = [
        (
            "executable",
            bound.executable_sha256 != now.executable_sha256,
        ),
        ("--config", bound.config_sha256 != now.config_sha256),
        (
            "--overrides",
            bound.overrides_sha256 != now.overrides_sha256,
        ),
        ("corpus", bound.corpus_hash != now.corpus_hash),
        ("cassette", bound.cassette != now.cassette),
        ("mode", bound.mode != now.mode),
        ("--refresh", bound.refresh != now.refresh),
        (
            "--prime-concurrency",
            bound.prime_concurrency != now.prime_concurrency,
        ),
        ("--until", bound.until != now.until),
    ];
    let differing: BTreeSet<&str> = fields
        .into_iter()
        .filter(|(_, differs)| *differs)
        .map(|(name, _)| name)
        .collect();
    differing.into_iter().collect::<Vec<_>>().join(", ")
}

/// The SHA-256 of a file the run reads, if it names one.
pub fn file_sha(path: Option<&Path>) -> anyhow::Result<Option<String>> {
    path.map(|path| {
        fs::read(path)
            .map(|bytes| sha(&bytes))
            .with_context(|| format!("hashing {}", path.display()))
    })
    .transpose()
}

/// The SHA-256 of the running executable.
pub fn executable_sha() -> anyhow::Result<String> {
    let path = std::env::current_exe().context("finding the running executable")?;
    Ok(sha(
        &fs::read(&path).with_context(|| format!("hashing {}", path.display()))?
    ))
}

impl Admission for Budget {
    /// A refused attempt fails with a code no retry policy knows, so it's
    /// never retried; the run reports the budget's own reason.
    fn admit(&self, request: &LlmRequest) -> Result<(), LlmError> {
        self.take_slot(&request.template.name)
            .map_err(|_| LlmError::Backend {
                code: REFUSED_CODE.into(),
            })
    }
}

/// A backend client that sends one request per call, admitting it against
/// a [`Budget`] first. It wraps the raw client, beneath the retry layer, so
/// a retry is an attempt like any other. [`CodexResponses`], which can post
/// twice in a call, is given the budget as its admission instead.
///
/// [`CodexResponses`]: asphodel_core::models::CodexResponses
pub struct Budgeted {
    inner: Arc<dyn LlmClient>,
    budget: Arc<Budget>,
}

impl Budgeted {
    pub fn new(inner: Arc<dyn LlmClient>, budget: Arc<Budget>) -> Self {
        Self { inner, budget }
    }

    fn admit(&self, request: &LlmRequest) -> Result<(), LlmError> {
        self.budget.admit(request)
    }
}

impl LlmClient for Budgeted {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn reasoning_effort(&self) -> Option<&str> {
        self.inner.reasoning_effort()
    }

    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        self.admit(request)?;
        self.inner.complete(request)
    }

    fn complete_identified(
        &self,
        request: &LlmRequest,
        identities: &[(String, Uuid)],
    ) -> Result<LlmResponse, LlmError> {
        self.admit(request)?;
        self.inner.complete_identified(request, identities)
    }

    fn skips_write(&self, request: &LlmRequest, identities: &[(String, Uuid)]) -> bool {
        self.inner.skips_write(request, identities)
    }
}
