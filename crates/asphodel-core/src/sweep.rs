//! The nightly sweep, and acknowledging a purge pause ("Deletion policy",
//! TIM-97, decisions 1, 3, 4 and 6, as amended by TIM-99; ADR 0008; ADR
//! 0009; ADR 0010, "Sweeps, pauses and failures"; "Erase path, forget,
//! purge and the nightly sweep", TIM-112).
//!
//! **When.** Each bank's sweep runs at `mental_models.sweep_time` bank-local
//! (04:00), on the service's clock, once a day. The daemon runs it before
//! that night's mental model refreshes, so a model that cited a purged
//! memory refreshes once. The last sweep lives in memory: after a restart
//! the next is the first sweep time after the daemon started.
//!
//! **Purge.** A chain whose head is below τ − δ goes through the erase
//! path, one transaction per chain, with strength and the guards read on
//! the head ([`purge_eligible`]). A chain waiting on a forget's erase is
//! left to it. δ null never purges.
//!
//! **The source sweep** runs straight after, past
//! `purge.source_horizon_days` by `ingested_at`. A failed chunk loses its
//! text. A source no memory rests on loses its text and its chunks', unless
//! a chunk is still queued, and its stored in-context set goes. A recall
//! row loses its query and results. Every key stays as the tombstone, so
//! ingest stays idempotent. One `sweep_runs` row per bank records the
//! counts, the fingerprint and δ.
//!
//! **The pause.** While the stored deletion fingerprint differs from the
//! daemon's, purge and the source sweep don't run at all, as if δ were
//! null and the horizon infinite. `purge plan` shows what changed and what
//! the sweep would delete now; `purge ack --hash` must quote the running
//! daemon's hash, records it as a daemon-wide `purge_acked` edit and makes
//! it the stored fingerprint, so purging resumes at the next sweep and
//! after any restart. Forget never pauses.

use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;

use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::config::{DeletionInputs, Fingerprint, PurgePause, Tuning};
use crate::erase::{Aftermath, EraseReason, bank_links, erase_chain};
use crate::mental_models::schedule::next_local;
use crate::store::strength::{StrengthLoader, window};
use crate::store::{META_DELETION_FINGERPRINT, META_DELETION_INPUTS, Store, StoreError, micros};
use crate::strength::{PurgeRule, chain, chain_head, purge_eligible};

/// The edit kind an acknowledgement writes, with no bank.
pub const EDIT_PURGE_ACKED: &str = "purge_acked";

/// One bank's sweep, as its `sweep_runs` row records it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SweepRun {
    pub bank: String,
    pub fingerprint: Fingerprint,
    /// δ it ran under; `None` never purges.
    pub delta: Option<f64>,
    pub purged_memories: usize,
    pub swept_sources: usize,
    pub swept_chunks: usize,
    pub swept_failed_chunks: usize,
    pub swept_recalls: usize,
}

/// What [`crate::Service::run_sweeps`] did, and when the next is due.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Sweeps {
    pub ran: Vec<SweepRun>,
    pub next_due: Option<Timestamp>,
}

/// `purge plan`: what changed and what the sweep would delete now.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PurgePlan {
    pub pause: PurgePause,
    /// The fingerprint this daemon computed: the one `purge ack` takes.
    pub current: Fingerprint,
    /// The fingerprinted values that differ from the stored ones, by tuning
    /// key, with `constants` for the strength constants fixed in code, or
    /// `unknown` for a store that kept only the hash.
    pub changed: Vec<String>,
    /// Memories, counting every version of each chain.
    pub memories: usize,
    pub sources: usize,
    pub failed_chunks: usize,
    pub recalls: usize,
}

/// What `purge ack` takes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PurgeAck {
    /// The running daemon's deletion fingerprint, as `purge plan` and
    /// `GET /v1/config` show it.
    pub hash: String,
}

#[derive(Debug, thiserror::Error)]
pub enum PurgeError {
    /// The hash isn't the one the running daemon computed.
    #[error("that isn't this daemon's deletion fingerprint; `purge plan` shows it")]
    HashMismatch,

    #[error(transparent)]
    Store(#[from] StoreError),
}

impl From<rusqlite::Error> for PurgeError {
    fn from(error: rusqlite::Error) -> Self {
        PurgeError::Store(StoreError::Sqlite(error))
    }
}

/// Each bank's last sweep, in memory.
#[derive(Debug)]
pub(crate) struct SweepSchedule {
    started: Timestamp,
    last: Mutex<HashMap<i64, Timestamp>>,
}

impl SweepSchedule {
    pub(crate) fn new(started: Timestamp) -> Self {
        Self {
            started,
            last: Mutex::default(),
        }
    }

    fn next(&self, bank_id: i64, tz: &TimeZone, tuning: &Tuning) -> Timestamp {
        let last = self
            .last
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&bank_id)
            .copied()
            .unwrap_or(self.started);
        next_local(last, tz, tuning.mental_models.sweep_time)
    }

    fn swept(&self, bank_id: i64, now: Timestamp) {
        self.last
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(bank_id, now);
    }
}

/// What one bank's sweep counted, before it's written.
#[derive(Debug, Default)]
struct Counts {
    purged_memories: usize,
    swept_sources: usize,
    swept_chunks: usize,
    swept_failed_chunks: usize,
    swept_recalls: usize,
}

/// Runs each bank's sweep that's due at `now`. `banks` is every bank's
/// rowid, name and timezone. Returns what ran and when the next is due.
///
/// What each committed purge leaves the service to do is pushed to
/// `settled` as it commits, so a failure later in the sweep can't lose it.
/// A bank counts as swept only once its run row is written: a sweep that
/// fails stays due, and the daemon's timers retry it.
pub(crate) fn run(
    store: &Store,
    tuning: &Tuning,
    schedule: &SweepSchedule,
    pause: &PurgePause,
    banks: &[(i64, String, TimeZone)],
    settled: &mut Vec<(i64, Aftermath)>,
) -> Result<Sweeps, StoreError> {
    let now = store.now();
    let fingerprint = tuning.deletion_fingerprint();
    let mut ran = Vec::new();
    for (bank_id, bank, tz) in banks {
        if schedule.next(*bank_id, tz, tuning) > now {
            continue;
        }
        if *pause != PurgePause::Running {
            schedule.swept(*bank_id, now);
            tracing::warn!(bank = %bank, "the sweep is paused until the deletion fingerprint is acknowledged");
            continue;
        }
        let mut counts = Counts::default();
        for head in candidates(store, tuning, *bank_id, now)? {
            if let Some((purged, aftermath)) =
                purge_chain(store, tuning, pause, *bank_id, head, now)?
            {
                counts.purged_memories += purged.len();
                settled.push((*bank_id, aftermath));
            }
        }
        sweep_sources(store, tuning, *bank_id, now, &mut counts)?;
        {
            let conn = store.connection();
            conn.execute(
                "INSERT INTO sweep_runs (bank_id, started_at, completed_at, fingerprint, delta,
                                         purged_memories, swept_sources, swept_chunks,
                                         swept_failed_chunks, swept_recalls)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                rusqlite::params![
                    bank_id,
                    micros(now),
                    micros(store.now()),
                    fingerprint.as_str(),
                    tuning.purge.delta,
                    counts.purged_memories as i64,
                    counts.swept_sources as i64,
                    counts.swept_chunks as i64,
                    counts.swept_failed_chunks as i64,
                    counts.swept_recalls as i64,
                ],
            )?;
        }
        schedule.swept(*bank_id, now);
        tracing::info!(
            bank = %bank,
            purged = counts.purged_memories,
            sources = counts.swept_sources,
            chunks = counts.swept_chunks,
            failed_chunks = counts.swept_failed_chunks,
            recalls = counts.swept_recalls,
            "swept"
        );
        ran.push(SweepRun {
            bank: bank.clone(),
            fingerprint: fingerprint.clone(),
            delta: tuning.purge.delta,
            purged_memories: counts.purged_memories,
            swept_sources: counts.swept_sources,
            swept_chunks: counts.swept_chunks,
            swept_failed_chunks: counts.swept_failed_chunks,
            swept_recalls: counts.swept_recalls,
        });
    }
    let next_due = banks
        .iter()
        .map(|(bank_id, _, tz)| schedule.next(*bank_id, tz, tuning))
        .min();
    Ok(Sweeps { ran, next_due })
}

/// Purge's first phase: the heads of the bank's chains that are eligible
/// at `now`, with any chain a forget hid left to its erase. Nothing is
/// deleted; [`purge_chain`] decides again before it deletes.
pub(crate) fn candidates(
    store: &Store,
    tuning: &Tuning,
    bank_id: i64,
    now: Timestamp,
) -> Result<Vec<i64>, StoreError> {
    let rule = PurgeRule::from_tuning(tuning);
    if rule.delta.is_none() {
        return Ok(Vec::new());
    }
    let conn = store.connection();
    let loader = StrengthLoader::new(&conn, bank_id, tuning.clock.quiet_rate, now)?;
    let heads: Vec<i64> = {
        let mut statement = conn.prepare(
            "SELECT id FROM memories
             WHERE bank_id = ?1 AND superseded_by IS NULL AND hidden_at IS NULL
             ORDER BY id",
        )?;
        statement
            .query_map([bank_id], |row| row.get(0))?
            .collect::<Result<_, _>>()?
    };
    let mut eligible = Vec::new();
    for head in heads {
        if is_eligible(&conn, &loader, &rule, head, now)? {
            eligible.push(head);
        }
    }
    Ok(eligible)
}

fn is_eligible(
    conn: &Connection,
    loader: &StrengthLoader,
    rule: &PurgeRule,
    head: i64,
    now: Timestamp,
) -> Result<bool, rusqlite::Error> {
    let Some((window, tz)) = window(conn, head)? else {
        return Ok(false);
    };
    let strength = loader.strength(conn, head)?.value;
    Ok(purge_eligible(rule, strength, &window, &tz, now))
}

/// Purge's second phase, in one transaction: re-reads the chain `head` is
/// in now, whether any of it is hidden, its head's strength and both
/// guards, and purges the chain only if it's still eligible (the TIM-112
/// review). Between the phases a keep, a mention or a new successor can
/// make it ineligible, and a forget can hide it, which leaves it to the
/// forget's erase, the one that redacts. `None` when nothing was purged.
pub(crate) fn purge_chain(
    store: &Store,
    tuning: &Tuning,
    pause: &PurgePause,
    bank_id: i64,
    head: i64,
    now: Timestamp,
) -> Result<Option<(BTreeSet<uuid::Uuid>, Aftermath)>, StoreError> {
    let rule = PurgeRule::from_tuning(tuning);
    if *pause != PurgePause::Running || rule.delta.is_none() {
        return Ok(None);
    }
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let exists = tx
        .query_row(
            "SELECT 1 FROM memories WHERE id = ?1 AND bank_id = ?2",
            (head, bank_id),
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    if !exists {
        return Ok(None);
    }
    let links = bank_links(&tx, bank_id)?;
    let head = chain_head(&links, head);
    let members = chain(&links, head);
    for member in &members {
        let hidden = tx
            .query_row(
                "SELECT 1 FROM memories WHERE id = ?1 AND hidden_at IS NOT NULL",
                [member],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if hidden {
            return Ok(None);
        }
    }
    let loader = StrengthLoader::new(&tx, bank_id, tuning.clock.quiet_rate, now)?;
    if !is_eligible(&tx, &loader, &rule, head, now)? {
        return Ok(None);
    }
    let (purged, aftermath) = erase_chain(&tx, store, bank_id, &members, EraseReason::Purge)?;
    tx.commit()?;
    Ok(Some((purged, aftermath)))
}

/// The bank's chains purge would take at `now`, each as its members, read
/// without deleting anything: what `purge plan` counts.
fn purgeable(
    store: &Store,
    tuning: &Tuning,
    bank_id: i64,
    now: Timestamp,
) -> Result<Vec<BTreeSet<i64>>, StoreError> {
    let heads = candidates(store, tuning, bank_id, now)?;
    let conn = store.connection();
    let links = bank_links(&conn, bank_id)?;
    Ok(heads.into_iter().map(|head| chain(&links, head)).collect())
}

/// The source, failed-chunk and recall-log sweep past the horizon.
fn sweep_sources(
    store: &Store,
    tuning: &Tuning,
    bank_id: i64,
    now: Timestamp,
    counts: &mut Counts,
) -> Result<(), StoreError> {
    let horizon = horizon(tuning, now);
    let now = micros(now);
    let mut conn = store.connection();
    let tx = conn.transaction()?;

    counts.swept_failed_chunks = tx.execute(
        "UPDATE chunks SET text = NULL, call1_output = NULL, tombstoned_at = ?3
         WHERE bank_id = ?1 AND failed_at IS NOT NULL AND tombstoned_at IS NULL
           AND source_id IN (SELECT id FROM sources WHERE bank_id = ?1 AND ingested_at < ?2)",
        (bank_id, horizon, now),
    )?;

    let sources: Vec<i64> = sweepable_sources(&tx, bank_id, horizon, &BTreeSet::new())?;
    for source in &sources {
        counts.swept_chunks += tx.execute(
            "UPDATE chunks SET text = NULL, call1_output = NULL, tombstoned_at = ?2
             WHERE source_id = ?1 AND tombstoned_at IS NULL",
            (source, now),
        )?;
        tx.execute(
            "UPDATE sources SET text = NULL, reply = NULL, tombstoned_at = ?2,
                                tombstone_reason = 'swept'
             WHERE id = ?1",
            (source, now),
        )?;
        tx.execute("DELETE FROM turn_in_context WHERE source_id = ?1", [source])?;
        tx.execute("DELETE FROM turn_entries WHERE source_id = ?1", [source])?;
    }
    counts.swept_sources = sources.len();

    tx.execute(
        "DELETE FROM recall_results WHERE recall_id IN
           (SELECT id FROM recalls WHERE bank_id = ?1 AND at < ?2 AND swept_at IS NULL)",
        (bank_id, horizon),
    )?;
    counts.swept_recalls = tx.execute(
        "UPDATE recalls SET query = NULL, swept_at = ?3
         WHERE bank_id = ?1 AND at < ?2 AND swept_at IS NULL",
        (bank_id, horizon, now),
    )?;
    tx.commit()?;
    Ok(())
}

fn horizon(tuning: &Tuning, now: Timestamp) -> i64 {
    let days = SignedDuration::from_hours(24 * i64::from(tuning.purge.source_horizon_days));
    micros(now.checked_sub(days).unwrap_or(Timestamp::MIN))
}

/// Sources past the horizon that still have text, that no memory rests on
/// except those in `purged`, and that have no chunk waiting on the queue.
/// The sweep passes nothing for `purged`, since it has purged by then; the
/// plan passes what it projects the purge will take.
fn sweepable_sources(
    conn: &Connection,
    bank_id: i64,
    horizon: i64,
    purged: &BTreeSet<i64>,
) -> Result<Vec<i64>, rusqlite::Error> {
    let candidates: Vec<i64> = {
        let mut statement = conn.prepare_cached(
            "SELECT s.id FROM sources s
             WHERE s.bank_id = ?1 AND s.ingested_at < ?2 AND s.tombstoned_at IS NULL
               AND NOT EXISTS (SELECT 1 FROM extraction_queue q JOIN chunks c ON c.id = q.chunk_id
                               WHERE c.source_id = s.id)
             ORDER BY s.id",
        )?;
        statement
            .query_map((bank_id, horizon), |row| row.get(0))?
            .collect::<Result<_, _>>()?
    };
    let mut resting = conn.prepare_cached(
        "SELECT m.id FROM memories m JOIN chunks c ON c.id = m.chunk_id WHERE c.source_id = ?1",
    )?;
    let mut sources = Vec::new();
    for source in candidates {
        let mut free = true;
        for memory in resting.query_map([source], |row| row.get::<_, i64>(0))? {
            if !purged.contains(&memory?) {
                free = false;
                break;
            }
        }
        if free {
            sources.push(source);
        }
    }
    Ok(sources)
}

/// What the sweep would delete now, without deleting anything.
pub(crate) fn plan(
    store: &Store,
    tuning: &Tuning,
    pause: &PurgePause,
    banks: &[i64],
) -> Result<PurgePlan, StoreError> {
    let now = store.now();
    let current = tuning.deletion_fingerprint();
    let inputs = DeletionInputs::new(tuning);
    let changed = match store.stored_deletion_inputs()? {
        Some(stored) => inputs.changed_from(&stored),
        None => match pause {
            PurgePause::Paused { .. } => vec!["unknown".to_string()],
            _ => Vec::new(),
        },
    };
    let mut plan = PurgePlan {
        pause: pause.clone(),
        current,
        changed,
        memories: 0,
        sources: 0,
        failed_chunks: 0,
        recalls: 0,
    };
    let horizon = horizon(tuning, now);
    for &bank_id in banks {
        let purged: BTreeSet<i64> = purgeable(store, tuning, bank_id, now)?
            .into_iter()
            .flatten()
            .collect();
        plan.memories += purged.len();
        let conn = store.connection();
        plan.sources += sweepable_sources(&conn, bank_id, horizon, &purged)?.len();
        plan.failed_chunks += conn.query_row(
            "SELECT COUNT(*) FROM chunks
             WHERE bank_id = ?1 AND failed_at IS NOT NULL AND tombstoned_at IS NULL
               AND source_id IN (SELECT id FROM sources WHERE bank_id = ?1 AND ingested_at < ?2)",
            (bank_id, horizon),
            |row| row.get::<_, i64>(0),
        )? as usize;
        plan.recalls += conn.query_row(
            "SELECT COUNT(*) FROM recalls WHERE bank_id = ?1 AND at < ?2 AND swept_at IS NULL",
            (bank_id, horizon),
            |row| row.get::<_, i64>(0),
        )? as usize;
    }
    Ok(plan)
}

/// `purge ack --hash`: `hash` must be this daemon's fingerprint. Stores it
/// and its inputs as the store's, and logs the ack with the fingerprint it
/// replaced.
pub(crate) fn ack(store: &Store, tuning: &Tuning, hash: &str) -> Result<(), PurgeError> {
    let current = tuning.deletion_fingerprint();
    if hash.trim() != current.as_str() {
        return Err(PurgeError::HashMismatch);
    }
    let now = micros(store.now());
    let mut conn = store.connection();
    let tx = conn.transaction()?;
    let previous: Option<String> = tx
        .query_row(
            "SELECT value FROM store_meta WHERE key = ?1",
            [META_DELETION_FINGERPRINT],
            |row| row.get(0),
        )
        .optional()?;
    for (key, value) in [
        (META_DELETION_FINGERPRINT, current.as_str().to_string()),
        (
            META_DELETION_INPUTS,
            serde_json::to_string(&DeletionInputs::new(tuning)).expect("the inputs serialise"),
        ),
    ] {
        tx.execute(
            "INSERT INTO store_meta (key, value, updated_at) VALUES (?1, ?2, ?3)
             ON CONFLICT (key) DO UPDATE SET value = excluded.value,
                                             updated_at = excluded.updated_at",
            (key, value, now),
        )?;
    }
    tx.execute(
        "INSERT INTO edits (uuid, bank_id, kind, details, at) VALUES (?1, NULL, ?2, ?3, ?4)",
        (
            store.new_id().to_string(),
            EDIT_PURGE_ACKED,
            serde_json::json!({ "hash": current.as_str(), "previous": previous }).to_string(),
            now,
        ),
    )?;
    tx.commit()?;
    tracing::info!(hash = %current, "the deletion fingerprint was acknowledged; purge resumes at the next sweep");
    Ok(())
}
