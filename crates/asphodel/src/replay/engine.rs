//! The discrete-event simulation (TIM-96, decision 3; `docs/replay.md`,
//! "The simulation").
//!
//! One queue holds prefetches, syncs, extraction completions, the
//! housekeeping timer (sweeps and refreshes) and probes, ordered by
//! simulated time and then by kind: timers, completions, prefetches, syncs
//! and documents, probes, and within a kind by the order they were
//! scheduled. The clock is set to each event's time before it runs, so
//! every store write carries the simulated time.
//!
//! Extraction is queued at a source's sync. Each bank has one simulated
//! worker. Whenever it's free, at a sync or at its previous completion, it
//! claims the head of the production queue (turns before documents, then
//! observed time) and commits that chunk a latency later. At that moment
//! it also commits the same source's next chunks, for as long as each is
//! the queue's head. Once another source is at the head, for example a
//! turn synced in the meantime, the worker claims that instead. The rest
//! of the document then waits behind it and is charged another latency
//! when its turn comes. A probe or prefetch before a completion sees the
//! store without those memories. Accesses are stamped with the source's
//! ingest time, as in production. The run ends at the latest of the last
//! event, `--until` and the last completion.
//!
//! The LLM is answered one of two ways. For a scenario, from its claims:
//! call 1's reply is built from them, and call 2's from their outcomes
//! against the neighbours the real reconciliation found, so nothing in
//! reconciliation is replay-only. Where the scenario says something
//! production wouldn't do (an outcome against a memory call 2 isn't shown,
//! a `used` memory that isn't in context, a label on a claim its outcomes
//! absorb), the run stops with a scenario error rather than guessing. For
//! real history, through the cassette [`Recorder`] (TIM-96, decision 4),
//! with `fast` mode's call 1 composed from the recording and the rest
//! answered by request.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use asphodel_core::extraction::{Call1Input, DropReason, Extracted, Prepared};
use asphodel_core::ingest::{Document, Outcome as IngestOutcome, Turn, TurnAuthor};
use asphodel_core::inspect::InspectError;
use asphodel_core::models::{FakeLlm, LlmClient, LlmError, LlmRequest, LlmResponse};
use asphodel_core::retrieval::{Band, PrefetchRequest, RecallRequest, band, estimate_tokens};
use asphodel_core::service::Claimed;
use asphodel_core::store::ids::derived;
use asphodel_core::strength::Phase;
use asphodel_core::{Clock, Service, SimulatedClock, Tuning};
use jiff::civil::Date;
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use regex::Regex;
use serde_json::{Value, json};
use uuid::Uuid;

use super::cassette::{Chained, ChunkContext, ChunkKey, Recorder};
use super::labelling::{Collector, Material};
use super::report::{
    Call2Rate, DayCount, InjectedTokens, Lag, LlmCounts, MemoryOutcome, Percentiles, ProbeResult,
    SessionTokens, WeekBands, WeekCount,
};
use super::scenario::{Author, Check, Claim, PROBE_SESSION_PREFIX};
use super::shadow::{Created, ShadowRow};
use super::timeline::{Matching, SessionClass, Timeline};
use crate::cli::ReplayMode;

/// The scripted LLM's model name.
const MODEL: &str = "scripted";

/// Why a run stopped short of a report.
#[derive(Debug)]
pub enum Failure {
    /// The scenario asked for something production wouldn't do.
    Scenario(String),
    /// The service or the store failed.
    Internal(anyhow::Error),
}

impl<E: std::error::Error + Send + Sync + 'static> From<E> for Failure {
    fn from(error: E) -> Self {
        Failure::Internal(anyhow::Error::new(error))
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failure::Scenario(message) => write!(f, "scenario error: {message}"),
            Failure::Internal(error) => write!(f, "{error:#}"),
        }
    }
}

fn internal(message: impl Into<String>) -> Failure {
    Failure::Internal(anyhow::anyhow!(message.into()))
}

/// Who answers the LLM calls.
#[derive(Clone, Copy)]
pub enum Llm<'a> {
    /// A scenario's claims.
    Scripted,
    /// The cassette, in the mode it was opened in.
    Recorded(&'a Recorder, ReplayMode),
}

/// What the engine needs besides the timeline.
pub struct Settings {
    pub bank: String,
    pub timezone: TimeZone,
    /// The simulated extraction latency, when it isn't taken from the
    /// cassette.
    pub latency: SignedDuration,
    /// Take each chunk's latency from how long the calls that answered it
    /// took (TIM-96, decision 3): recorded for a record, measured for a
    /// live call. Otherwise `latency`.
    pub latency_from_cassette: bool,
    pub until: Option<Timestamp>,
    /// Collect the labelling material (TIM-96, decision 6).
    pub labelling: bool,
}

/// What a run produced for the report.
pub struct Outcome {
    pub probes: Vec<ProbeResult>,
    pub purges_per_day: Vec<DayCount>,
    pub fade_outs_per_week: Vec<WeekCount>,
    pub bands_per_week: Vec<WeekBands>,
    pub extraction_lag: Lag,
    pub refresh_calls_per_day: Vec<DayCount>,
    pub injected_tokens: InjectedTokens,
    pub profile_tokens: Percentiles,
    pub call2_rate: Call2Rate,
    pub agenda_lines_per_day: Vec<DayCount>,
    pub significance_histogram: BTreeMap<String, u64>,
    pub kind_histogram: BTreeMap<String, u64>,
    pub memories: Vec<MemoryOutcome>,
    pub llm: LlmCounts,
    pub created: Vec<Created>,
    pub shadow: Vec<ShadowRow>,
    /// The labelling material, when the settings asked for it.
    pub material: Option<Material>,
}

/// A turn as the engine plays it.
struct PlannedTurn {
    at: Timestamp,
    reply_at: Timestamp,
    session: String,
    user: String,
    assistant: String,
    previous_query: Option<String>,
    author: Option<Author>,
    platform: Option<String>,
    class: SessionClass,
    prefetch_only: bool,
    claims: Vec<Claim>,
    used: Vec<String>,
    recall_id: Option<String>,
}

/// A synced source whose chunks aren't all extracted yet.
struct Pending {
    synced_at: Timestamp,
    /// Its chunks still to extract.
    chunks: usize,
    /// Its claims not yet matched to an extracted chunk.
    claims: Vec<Claim>,
    used: Vec<String>,
    /// How errors name the event.
    name: String,
}

/// A chunk the worker has claimed and prepared: its LLM calls ran at the
/// claim, and it commits at its completion.
struct Ready {
    prepared: Prepared,
    source: Uuid,
    position: u32,
    /// Its scripted claims: those whose quote is in it.
    mine: Vec<Claim>,
    /// How long it takes, in simulated time, from the claim to the commit.
    latency: SignedDuration,
}

enum EventKind {
    Timer,
    Completion(Box<Ready>),
    Prefetch(usize),
    Sync(usize),
    Document(usize),
    Clear(usize),
    Probe(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    at: Timestamp,
    rank: u8,
    seq: u64,
}

/// A daily look at every live memory's band, taken after each sweep.
struct Snapshot {
    day: Date,
    strong: u64,
    fading: u64,
    faded: u64,
    fade_outs: u64,
}

/// Answers every refresh with no edits, counting the calls.
pub struct NoEdits {
    calls: Mutex<Vec<Timestamp>>,
    clock: Arc<SimulatedClock>,
}

impl LlmClient for NoEdits {
    fn model(&self) -> &str {
        MODEL
    }

    fn complete(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        self.calls
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(self.clock.now());
        Ok(LlmResponse {
            json: json!({ "operations": [] }),
            usage: None,
            latency: Duration::ZERO,
        })
    }
}

/// Tokens injected by a session's prefetches.
#[derive(Default)]
struct SessionCount {
    prefetches: u64,
    tokens: u64,
    synced: bool,
}

pub struct Engine<'a> {
    service: &'a Service,
    clock: Arc<SimulatedClock>,
    timeline: &'a Timeline,
    settings: Settings,
    tuning: &'a Tuning,
    llm: Llm<'a>,
    turns: Vec<PlannedTurn>,
    queue: BinaryHeap<Reverse<Key>>,
    events: BTreeMap<u64, EventKind>,
    seq: u64,
    timers: BTreeSet<Timestamp>,
    /// Timers due after the end as it stood, scheduled if a completion
    /// moves the end past them.
    deferred: BTreeSet<Timestamp>,
    end: Timestamp,
    /// Claim labels to the memories they created.
    labels: BTreeMap<String, Uuid>,
    /// Each probe's memory regex, in real history.
    regexes: Vec<Option<Regex>>,
    created: Vec<Created>,
    shadow: Vec<ShadowRow>,
    /// Synced sources by id, until their last chunk is extracted.
    pending: BTreeMap<Uuid, Pending>,
    /// Whether the worker holds a lease, with its completion on the queue.
    working: bool,
    refresh_llm: NoEdits,
    probes: Vec<ProbeResult>,
    purges: BTreeMap<String, u64>,
    lags_ms: Vec<u64>,
    llm_calls: u64,
    refresh_calls: BTreeMap<String, u64>,
    snapshots: Vec<Snapshot>,
    previous_strengths: BTreeMap<Uuid, f64>,
    sessions: BTreeMap<String, SessionCount>,
    turn_tokens: Vec<u64>,
    cron_prefetches: u64,
    cron_tokens: u64,
    profile_tokens: Vec<u64>,
    chunks: u64,
    call2_chunks: u64,
    agenda_lines: BTreeMap<String, u64>,
    significance_histogram: BTreeMap<String, u64>,
    kind_histogram: BTreeMap<String, u64>,
    /// The labelling material as it's collected.
    labelling: Option<Collector>,
}

impl<'a> Engine<'a> {
    pub fn new(
        service: &'a Service,
        clock: Arc<SimulatedClock>,
        timeline: &'a Timeline,
        tuning: &'a Tuning,
        settings: Settings,
        llm: Llm<'a>,
    ) -> Result<Self, Failure> {
        let start = timeline.earliest().ok_or_else(|| {
            Failure::Scenario("a scenario needs at least one event or probe".into())
        })?;
        let end = timeline
            .latest()
            .unwrap_or(start)
            .max(settings.until.unwrap_or(start));
        let regexes = match timeline.matching {
            Matching::Label => vec![None; timeline.probes.len()],
            Matching::Sentence => timeline
                .probes
                .iter()
                .map(|probe| Regex::new(probe.check.memory()).map(Some))
                .collect::<Result<_, _>>()
                .map_err(|_| Failure::Scenario("a probe's memory regex doesn't parse".into()))?,
        };
        let labelling = settings.labelling.then(Collector::default);
        let mut engine = Self {
            service,
            clock: Arc::clone(&clock),
            timeline,
            settings,
            tuning,
            llm,
            turns: Vec::new(),
            queue: BinaryHeap::new(),
            events: BTreeMap::new(),
            seq: 0,
            timers: BTreeSet::new(),
            deferred: BTreeSet::new(),
            end,
            labels: BTreeMap::new(),
            regexes,
            created: Vec::new(),
            shadow: Vec::new(),
            pending: BTreeMap::new(),
            working: false,
            refresh_llm: NoEdits {
                calls: Mutex::new(Vec::new()),
                clock,
            },
            probes: Vec::new(),
            purges: BTreeMap::new(),
            lags_ms: Vec::new(),
            llm_calls: 0,
            refresh_calls: BTreeMap::new(),
            snapshots: Vec::new(),
            previous_strengths: BTreeMap::new(),
            sessions: BTreeMap::new(),
            turn_tokens: Vec::new(),
            cron_prefetches: 0,
            cron_tokens: 0,
            profile_tokens: Vec::new(),
            chunks: 0,
            call2_chunks: 0,
            agenda_lines: BTreeMap::new(),
            significance_histogram: BTreeMap::new(),
            kind_histogram: BTreeMap::new(),
            labelling,
        };
        engine.plan();
        Ok(engine)
    }

    /// Lays every timeline event on the queue.
    fn plan(&mut self) {
        for turn in &self.timeline.turns {
            self.turns.push(PlannedTurn {
                at: turn.at,
                reply_at: turn.reply_at,
                session: turn.session.clone(),
                user: turn.user.clone(),
                assistant: turn.assistant.clone(),
                previous_query: turn.previous_query.clone(),
                author: turn.author.clone(),
                platform: turn.platform.clone(),
                class: turn.class,
                prefetch_only: turn.prefetch_only,
                claims: turn.claims.clone(),
                used: turn.used.clone(),
                recall_id: None,
            });
        }
        // Turns in time order, so a prefetch's seq follows the clock.
        self.turns.sort_by_key(|turn| turn.at);
        for index in 0..self.turns.len() {
            let (at, reply_at, prefetch_only) = (
                self.turns[index].at,
                self.turns[index].reply_at,
                self.turns[index].prefetch_only,
            );
            self.push(at, 2, EventKind::Prefetch(index));
            if !prefetch_only {
                self.push(reply_at.max(at), 3, EventKind::Sync(index));
            }
        }
        for (index, document) in self.timeline.documents.iter().enumerate() {
            self.push(document.at, 3, EventKind::Document(index));
        }
        for (index, clear) in self.timeline.clears.iter().enumerate() {
            self.push(clear.at, 3, EventKind::Clear(index));
        }
        for (index, probe) in self.timeline.probes.iter().enumerate() {
            self.push(probe.at, 4, EventKind::Probe(index));
        }
    }

    fn push(&mut self, at: Timestamp, rank: u8, kind: EventKind) {
        self.seq += 1;
        self.queue.push(Reverse(Key {
            at,
            rank,
            seq: self.seq,
        }));
        self.events.insert(self.seq, kind);
    }

    /// Schedules the housekeeping timer at `at`, once per instant, within
    /// the run. One past the end waits in case a completion extends it.
    fn schedule_timer(&mut self, at: Option<Timestamp>) {
        let Some(at) = at else {
            return;
        };
        if at < self.clock.now() || self.timers.contains(&at) {
            return;
        }
        if at > self.end {
            self.deferred.insert(at);
            return;
        }
        self.timers.insert(at);
        self.push(at, 0, EventKind::Timer);
    }

    /// Moves the end to `at` if it's later, with the timers due by then.
    fn extend_end(&mut self, at: Timestamp) {
        if at <= self.end {
            return;
        }
        self.end = at;
        let due: Vec<Timestamp> = self.deferred.range(..=at).copied().collect();
        for at in due {
            self.deferred.remove(&at);
            self.schedule_timer(Some(at));
        }
    }

    /// Runs the queue to the end.
    pub fn run(mut self) -> Result<Outcome, Failure> {
        self.schedule_timer(Some(self.clock.now()));
        while let Some(Reverse(key)) = self.queue.pop() {
            if key.at > self.end {
                break;
            }
            let kind = self
                .events
                .remove(&key.seq)
                .ok_or_else(|| internal("an event vanished from the queue"))?;
            self.clock.set(key.at);
            match kind {
                EventKind::Timer => self.timer()?,
                EventKind::Completion(ready) => self.complete(*ready)?,
                EventKind::Prefetch(index) => self.prefetch(index)?,
                EventKind::Sync(index) => self.sync(index)?,
                EventKind::Document(index) => self.document(index)?,
                EventKind::Clear(index) => self.clear(index)?,
                EventKind::Probe(index) => self.probe(index)?,
            }
        }
        self.clock.set(self.end);
        self.finish()
    }

    fn prefetch(&mut self, index: usize) -> Result<(), Failure> {
        let turn = &self.turns[index];
        let scored = self.service.scored_prefetch(
            &self.settings.bank,
            &PrefetchRequest {
                session_id: turn.session.clone(),
                query: turn.user.clone(),
                previous_query: turn.previous_query.clone(),
                block_id: None,
            },
        )?;
        // The material samples synced turns: what a turn is shown, which
        // cron prefetches and probes aren't.
        if let Some(collector) = &mut self.labelling
            && turn.class == SessionClass::Primary
            && !turn.prefetch_only
        {
            collector.turn(turn.at, &turn.session, &scored.query, &scored.candidates);
        }
        let prefetch = scored.prefetch;
        let tokens = estimate_tokens(&prefetch.text) as u64;
        match turn.class {
            SessionClass::Cron => {
                self.cron_prefetches += 1;
                self.cron_tokens += tokens;
            }
            SessionClass::Primary => {
                let session = self.sessions.entry(turn.session.clone()).or_default();
                session.prefetches += 1;
                session.tokens += tokens;
                if !turn.prefetch_only {
                    session.synced = true;
                    self.turn_tokens.push(tokens);
                }
            }
        }
        self.turns[index].recall_id = Some(prefetch.recall_id.to_string());
        Ok(())
    }

    fn sync(&mut self, index: usize) -> Result<(), Failure> {
        let turn = &self.turns[index];
        let ingested = self.service.ingest_turn(
            &self.settings.bank,
            &Turn {
                session_id: turn.session.clone(),
                message_at: turn.at,
                timezone: None,
                user_text: turn.user.clone(),
                assistant_text: turn.assistant.clone(),
                author: turn.author.as_ref().map(|author| TurnAuthor {
                    id: author.id.clone(),
                    name: author.name.clone(),
                    is_bot: false,
                }),
                platform: turn.platform.clone(),
                recall_id: turn.recall_id.clone(),
                forget_requested: false,
            },
        )?;
        if ingested.outcome == IngestOutcome::Duplicate || ingested.chunks_queued == 0 {
            return Ok(());
        }
        let pending = Pending {
            synced_at: self.clock.now(),
            chunks: ingested.chunks_queued,
            claims: turn.claims.clone(),
            used: turn.used.clone(),
            name: format!("the turn at {}", turn.at),
        };
        self.pending.insert(ingested.source, pending);
        self.start_worker()
    }

    fn document(&mut self, index: usize) -> Result<(), Failure> {
        let document = &self.timeline.documents[index];
        let ingested = self.service.ingest_document(
            &self.settings.bank,
            &Document {
                document_id: document.id.clone(),
                text: document.text.clone(),
                reference_date: document.reference_date,
                reference_date_exact: true,
                timezone: document.timezone.clone(),
            },
        )?;
        if ingested.outcome == IngestOutcome::Duplicate || ingested.chunks_queued == 0 {
            return Ok(());
        }
        let pending = Pending {
            synced_at: self.clock.now(),
            chunks: ingested.chunks_queued,
            claims: document.claims.clone(),
            used: Vec::new(),
            name: format!("document {}", document.id),
        };
        self.pending.insert(ingested.source, pending);
        self.start_worker()
    }

    fn clear(&mut self, index: usize) -> Result<(), Failure> {
        let clear = &self.timeline.clears[index];
        self.service
            .clear_session(&self.settings.bank, &clear.session)?;
        Ok(())
    }

    /// When the bank's one worker is free, it claims the head of the
    /// production queue, runs its LLM calls at once, and commits it a
    /// latency later.
    fn start_worker(&mut self) -> Result<(), Failure> {
        if self.working {
            return Ok(());
        }
        match self.service.next_extraction(&self.settings.bank)? {
            Some(claimed) => {
                let ready = self.prepare_chunk(claimed)?;
                self.schedule_completion(ready)
            }
            None => Ok(()),
        }
    }

    /// Holds the worker on a prepared chunk until its latency from now.
    fn schedule_completion(&mut self, ready: Ready) -> Result<(), Failure> {
        let at = self
            .clock
            .now()
            .checked_add(ready.latency)
            .unwrap_or(Timestamp::MAX);
        self.working = true;
        self.extend_end(at);
        self.push(at, 1, EventKind::Completion(Box::new(ready)));
        Ok(())
    }

    /// Commits the worker's chunk, and with it the rest of its source's
    /// chunks while they are the queue's head: the latency is per source,
    /// so those are prepared and committed here. Then the worker takes
    /// the next head, if any.
    fn complete(&mut self, ready: Ready) -> Result<(), Failure> {
        let source = ready.source;
        self.commit_chunk(ready)?;
        let next = loop {
            match self.service.next_extraction(&self.settings.bank)? {
                Some(claimed) if claimed.lease.source == source => {
                    let ready = self.prepare_chunk(claimed)?;
                    self.commit_chunk(ready)?;
                }
                other => break other,
            }
        };
        // Notable writes start the refresh debounce from the completion.
        let refreshes = self.run_refreshes()?;
        self.schedule_timer(refreshes);
        self.working = false;
        match next {
            Some(claimed) => {
                let ready = self.prepare_chunk(claimed)?;
                self.schedule_completion(ready)
            }
            None => Ok(()),
        }
    }

    /// Runs the refreshes due with the run's refresh client and counts
    /// their calls by bank-local day. Returns when the next is due.
    fn run_refreshes(&mut self) -> Result<Option<Timestamp>, Failure> {
        let (next_due, calls) = match self.llm {
            Llm::Scripted => {
                let refreshes = self.service.run_refreshes(&self.refresh_llm)?;
                let calls: Vec<Timestamp> = std::mem::take(
                    &mut *self
                        .refresh_llm
                        .calls
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                );
                self.llm_calls += calls.len() as u64;
                (refreshes.next_due, calls)
            }
            Llm::Recorded(recorder, _) => {
                let refreshes = self.service.run_refreshes(recorder)?;
                (refreshes.next_due, recorder.take_refresh_times())
            }
        };
        for at in calls {
            let day = at
                .to_zoned(self.settings.timezone.clone())
                .date()
                .to_string();
            *self.refresh_calls.entry(day).or_default() += 1;
        }
        Ok(next_due)
    }

    /// Prepares one claimed chunk: its call 1 and call 2 run now, with the
    /// scripted replies or through the cassette, and the latency it
    /// completes after is the settings' constant, or, when it's taken from
    /// the recording, how long the calls that answered took (TIM-96,
    /// decision 3).
    fn prepare_chunk(&mut self, claimed: Claimed) -> Result<Ready, Failure> {
        let Claimed {
            lease,
            in_context,
            entries,
        } = claimed;
        let source = lease.source;
        let position = lease.position;
        let input = self.service.call1_input(&lease, &in_context)?;
        let (mine, used, name) = {
            let pending = self.pending.get_mut(&source).ok_or_else(|| {
                internal(format!(
                    "the worker claimed source {source}, which the timeline didn't sync"
                ))
            })?;
            // The chunk's scripted claims: those whose quote is in it, in
            // order.
            let mut mine: Vec<Claim> = Vec::new();
            pending.claims.retain(|claim| {
                if input.text.contains(&claim.quote) {
                    mine.push(claim.clone());
                    false
                } else {
                    true
                }
            });
            (mine, pending.used.clone(), pending.name.clone())
        };
        self.chunks += 1;

        let (prepared, latency) = match self.llm {
            Llm::Scripted => {
                let mut used_handles = Vec::new();
                for label in &used {
                    let id = self.labels.get(label).copied();
                    let handle = id.and_then(|id| {
                        input
                            .in_context
                            .iter()
                            .find(|memory| memory.memory == id)
                            .map(|memory| memory.handle.clone())
                    });
                    match handle {
                        Some(handle) => used_handles.push(handle),
                        None => {
                            return Err(Failure::Scenario(format!(
                                "{name} uses {label:?}, which isn't in the session's in-context set"
                            )));
                        }
                    }
                }
                let reply1 = call1_reply(&mine, &used_handles);
                let call2 = self.service.call2_input(&lease, &reply1, &in_context)?;
                if call2.is_some() {
                    self.call2_chunks += 1;
                }
                let reply2 = self.call2_reply(&mine, call2.as_ref(), &name)?;
                let llm = FakeLlm::scripted(MODEL, vec![reply1, reply2]);
                let prepared =
                    self.service
                        .prepare_extraction(lease, &llm, &in_context, &entries)?;
                self.llm_calls += llm.requests().len() as u64;
                (prepared, self.settings.latency)
            }
            Llm::Recorded(recorder, mode) => {
                let (prepared, served) =
                    self.prepare_recorded(recorder, mode, lease, &in_context, &entries, &input)?;
                let latency = if self.settings.latency_from_cassette {
                    SignedDuration::try_from(served).unwrap_or(SignedDuration::MAX)
                } else {
                    self.settings.latency
                };
                (prepared, latency)
            }
        };
        if self.labelling.is_some() {
            let lists = self.service.call2_lists(&prepared)?;
            let now = self.clock.now();
            if let Some(collector) = &mut self.labelling {
                collector.call2(now, lists);
            }
        }
        Ok(Ready {
            prepared,
            source,
            position,
            mine,
            latency,
        })
    }

    /// Commits a prepared chunk at its completion.
    fn commit_chunk(&mut self, ready: Ready) -> Result<(), Failure> {
        let Ready {
            prepared,
            source,
            position,
            mine,
            ..
        } = ready;
        let mut pending = self
            .pending
            .remove(&source)
            .ok_or_else(|| internal(format!("no pending extraction for source {source}")))?;
        let extracted = self.service.commit_extraction(prepared)?;

        let now = self.clock.now();
        self.note_created(&extracted, source, position, &mine, &pending.name)?;
        let lag = now.duration_since(pending.synced_at);
        self.lags_ms
            .push(u64::try_from(lag.as_millis()).unwrap_or(0));
        pending.chunks -= 1;
        if pending.chunks > 0 {
            self.pending.insert(source, pending);
        } else if !pending.claims.is_empty() {
            return Err(Failure::Scenario(format!(
                "{} has {} claim(s) whose quote is in none of its chunks",
                pending.name,
                pending.claims.len()
            )));
        }
        Ok(())
    }

    /// Prepares a chunk through the cassette (TIM-96, decision 4), with
    /// how long the calls that answered took. In `fast` mode call 1 is
    /// composed from the recording when the chunk has one.
    fn prepare_recorded(
        &mut self,
        recorder: &Recorder,
        mode: ReplayMode,
        lease: asphodel_core::queue::Lease,
        in_context: &[Uuid],
        entries: &[asphodel_core::system_prompt::BlockEntry],
        input: &Call1Input,
    ) -> Result<(Prepared, Duration), Failure> {
        let context = ChunkContext::new(
            ChunkKey {
                source: lease.source,
                position: lease.position,
            },
            input,
        );
        let judged = context.in_context.len() as u64;
        recorder.enter(context.clone());
        let composed = match mode {
            ReplayMode::Fast => recorder.compose_call1(&context),
            _ => Ok(None),
        };
        let call2_before = recorder.with_counts(|counts| counts.call2);
        let result = match composed {
            Ok(Some(reply1)) => {
                let chained = Chained::new(reply1, recorder);
                self.service
                    .prepare_extraction(lease, &chained, in_context, entries)
            }
            Ok(None) => {
                recorder.with_counts(|counts| match mode {
                    ReplayMode::Replay => counts.verdicts.recorded += judged,
                    ReplayMode::Live | ReplayMode::Fast => counts.verdicts.live += judged,
                });
                self.service
                    .prepare_extraction(lease, recorder, in_context, entries)
            }
            Err(error) => Err(asphodel_core::extraction::ExtractError::Held { error }),
        };
        let served = recorder.served_latency();
        recorder.leave();
        if recorder.with_counts(|counts| counts.call2) > call2_before {
            self.call2_chunks += 1;
        }
        match result {
            Ok(prepared) => Ok((prepared, served)),
            Err(error) => Err(match recorder.first_miss() {
                Some(miss) => Failure::Internal(anyhow::anyhow!(
                    "{miss}; a replay fails on a miss, and fast needs an LLM for one"
                )),
                None => error.into(),
            }),
        }
    }

    /// Records what the extraction created: the shadow metric's rows, the
    /// histograms, and the scripted labels, which it checks.
    fn note_created(
        &mut self,
        extracted: &Extracted,
        source: Uuid,
        position: u32,
        mine: &[Claim],
        event: &str,
    ) -> Result<(), Failure> {
        let now = self.clock.now();
        match self.llm {
            Llm::Scripted => {
                for (ordinal, claim) in mine.iter().enumerate() {
                    let id = derived(source, &format!("{position}:{ordinal}"));
                    if extracted.memories.contains(&id) {
                        let embedding = self.embed(&claim.content)?;
                        self.created.push(Created {
                            memory: id,
                            content: claim.content.clone(),
                            embedding,
                            created_at: now,
                        });
                        if let Some(label) = &claim.label {
                            self.labels.insert(label.clone(), id);
                        }
                    } else if let Some(label) = &claim.label {
                        let reason = match extracted
                            .dropped
                            .iter()
                            .find(|dropped| dropped.claim == ordinal)
                        {
                            Some(dropped) => format!(
                                "call 1's checks dropped it: {}",
                                drop_reason(dropped.reason)
                            ),
                            None => "it is absorbed by its outcomes".into(),
                        };
                        return Err(Failure::Scenario(format!(
                            "the claim {label:?} in {event} created no memory ({reason}), so its label names nothing"
                        )));
                    }
                }
            }
            Llm::Recorded(..) => {
                for id in &extracted.memories {
                    let Some(view) = self.view(Some(*id))? else {
                        continue;
                    };
                    let embedding = self.embed(&view.sentence)?;
                    self.created.push(Created {
                        memory: *id,
                        content: view.sentence,
                        embedding,
                        created_at: now,
                    });
                }
            }
        }
        for id in &extracted.memories {
            if let Some(view) = self.view(Some(*id))? {
                *self.kind_histogram.entry(view.kind).or_default() += 1;
                *self
                    .significance_histogram
                    .entry(view.significance.extracted)
                    .or_default() += 1;
            }
        }
        Ok(())
    }

    /// Call 2's reply from the claims' outcomes, against the neighbours
    /// reconciliation found. An outcome whose target isn't among them is a
    /// scenario error.
    fn call2_reply(
        &self,
        mine: &[Claim],
        call2: Option<&asphodel_core::extraction::Call2Input>,
        event: &str,
    ) -> Result<Value, Failure> {
        let mut claims = Vec::new();
        for (ordinal, claim) in mine.iter().enumerate() {
            if claim.reconcile.is_empty() {
                continue;
            }
            let name = claim.name(ordinal);
            let Some(input) = call2 else {
                return Err(Failure::Scenario(format!(
                    "{name} in {event} has outcomes, but call 2 doesn't run for its chunk: nothing stored is near any of its claims"
                )));
            };
            let Some(reconcile) = input.claims.iter().find(|c| c.claim == ordinal) else {
                return Err(Failure::Scenario(format!(
                    "{name} in {event} has outcomes, but call 1's checks dropped it"
                )));
            };
            let handle_of: BTreeMap<Uuid, &str> = input
                .neighbours
                .iter()
                .map(|neighbour| (neighbour.memory, neighbour.handle.as_str()))
                .collect();
            let mut labels = Vec::new();
            for outcome in &claim.reconcile {
                let handle = self
                    .labels
                    .get(&outcome.memory)
                    .and_then(|id| handle_of.get(id))
                    .filter(|handle| reconcile.neighbours.iter().any(|n| n == *handle));
                let Some(handle) = handle else {
                    return Err(Failure::Scenario(format!(
                        "{name} in {event}: its outcome names {:?}, which isn't among the neighbours call 2 is shown for it ({})",
                        outcome.memory,
                        if reconcile.neighbours.is_empty() {
                            "none".to_string()
                        } else {
                            reconcile.neighbours.join(", ")
                        }
                    )));
                };
                labels.push(json!({ "neighbour": handle, "label": outcome.outcome.as_str() }));
            }
            claims.push(json!({ "claim": reconcile.handle, "labels": labels }));
        }
        Ok(json!({ "claims": claims }))
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, Failure> {
        let models = self
            .service
            .models()
            .ok_or_else(|| internal("the service has no models"))?;
        let mut vectors = models.embedder.embed(&[text])?;
        Ok(vectors.pop().unwrap_or_default())
    }

    /// The sweep, the shadow table, the daily band snapshot, the daily
    /// agenda and profile samples, and the refreshes due.
    fn timer(&mut self) -> Result<(), Failure> {
        let bank = self.settings.bank.clone();
        // What purge would take now, with its content, read before the
        // sweep deletes it.
        let mut chains: Vec<(Uuid, Vec<(Uuid, String)>)> = Vec::new();
        for (candidate_bank, head) in self.service.purge_candidates()? {
            if candidate_bank != bank {
                continue;
            }
            let view = self.service.show_memory(&bank, &head.to_string())?;
            let mut members = Vec::new();
            for member in &view.chain.members {
                let member_view = self.service.show_memory(&bank, &member.id.to_string())?;
                members.push((member.id, member_view.sentence));
            }
            chains.push((head, members));
        }

        let sweeps = self.service.run_sweeps()?;
        if !sweeps.ran.is_empty() {
            let day = self.local_day();
            let purged: u64 = sweeps
                .ran
                .iter()
                .filter(|run| run.bank == bank)
                .map(|run| run.purged_memories as u64)
                .sum();
            if purged > 0 {
                *self.purges.entry(day.clone()).or_default() += purged;
            }
            let now = self.clock.now();
            for (head, members) in chains {
                let gone = matches!(
                    self.service.show_memory(&bank, &head.to_string()),
                    Err(InspectError::UnknownMemory)
                );
                if !gone {
                    continue;
                }
                for (memory, content) in members {
                    let embedding = self.embed(&content)?;
                    self.shadow.push(ShadowRow {
                        memory,
                        content,
                        embedding,
                        purged_at: now,
                    });
                }
            }
            self.snapshot_bands()?;
            let agenda = self.service.agenda(&bank)?.listed().len() as u64;
            self.agenda_lines.insert(day, agenda);
            self.sample_profile_tokens()?;
        }

        let refreshes = self.run_refreshes()?;
        self.schedule_timer(sweeps.next_due);
        self.schedule_timer(refreshes);
        Ok(())
    }

    /// The tokens the bank's mental model entries hold now.
    fn sample_profile_tokens(&mut self) -> Result<(), Failure> {
        let bank = &self.settings.bank;
        let mut tokens = 0u64;
        for model in self.service.list_models(bank)? {
            let view = self.service.show_model(bank, &model.name, None)?;
            tokens += view
                .entry_views
                .iter()
                .filter(|entry| entry.renders)
                .map(|entry| estimate_tokens(&entry.text) as u64)
                .sum::<u64>();
        }
        self.profile_tokens.push(tokens);
        Ok(())
    }

    fn snapshot_bands(&mut self) -> Result<(), Failure> {
        let strengths: BTreeMap<Uuid, f64> = self
            .service
            .strengths(&self.settings.bank)?
            .into_iter()
            .collect();
        let cutoff = self.tuning.recall.strong_cutoff;
        let (mut strong, mut fading, mut faded) = (0, 0, 0);
        let mut fade_outs = 0;
        for (id, value) in &strengths {
            match band(*value, cutoff) {
                Band::Strong => strong += 1,
                Band::Fading => fading += 1,
                Band::Faded => faded += 1,
            }
            if let Some(previous) = self.previous_strengths.get(id)
                && band(*previous, cutoff) != Band::Faded
                && band(*value, cutoff) == Band::Faded
            {
                fade_outs += 1;
            }
        }
        self.snapshots.push(Snapshot {
            day: self
                .clock
                .now()
                .to_zoned(self.settings.timezone.clone())
                .date(),
            strong,
            fading,
            faded,
            fade_outs,
        });
        self.previous_strengths = strengths;
        Ok(())
    }

    fn local_day(&self) -> String {
        self.clock
            .now()
            .to_zoned(self.settings.timezone.clone())
            .date()
            .to_string()
    }

    /// The memory a probe names: by label in a scenario, or the earliest
    /// created memory whose sentence its regex matches in real history.
    fn probe_memory(&self, index: usize) -> Option<Uuid> {
        let memory = self.timeline.probes[index].check.memory();
        match &self.regexes[index] {
            None => self.labels.get(memory).copied(),
            Some(regex) => self
                .created
                .iter()
                .filter(|created| regex.is_match(&created.content))
                .min_by_key(|created| (created.created_at, created.memory))
                .map(|created| created.memory),
        }
    }

    fn probe(&mut self, index: usize) -> Result<(), Failure> {
        let probe = &self.timeline.probes[index];
        let id = probe.id(index);
        let bank = &self.settings.bank;
        let memory = self.probe_memory(index);
        let (passed, observed) = match &probe.check {
            Check::Band { band: expected, .. } => match self.view(memory)? {
                Some(view) => {
                    let got = band(view.strength.value, self.tuning.recall.strong_cutoff);
                    (
                        got == *expected,
                        json!({ "id": view.id, "band": got, "strength": view.strength.value }),
                    )
                }
                None => (false, json!({ "present": false })),
            },
            Check::FadedAt { between, .. } => match memory {
                Some(id) => match self.service.faded_at(bank, id) {
                    Ok(faded_at) => (
                        faded_at.is_some_and(|at| at >= between[0] && at <= between[1]),
                        json!({ "id": id, "faded_at": faded_at }),
                    ),
                    Err(InspectError::UnknownMemory) => (false, json!({ "present": false })),
                    Err(error) => return Err(error.into()),
                },
                None => (false, json!({ "present": false })),
            },
            Check::Exists {
                memory_kind,
                ended,
                retracted,
                head,
                phase,
                ..
            } => match self.view(memory)? {
                Some(view) => {
                    let is_ended = matches!(
                        view.phase,
                        Some(Phase::RecentlyPast) | Some(Phase::LongPast)
                    );
                    let is_retracted = view.retracted_at.is_some();
                    let is_head = view.chain.head == view.id;
                    let kind_matches = memory_kind.is_none_or(|kind| {
                        serde_json::to_value(kind)
                            .ok()
                            .and_then(|value| value.as_str().map(str::to_owned))
                            .as_deref()
                            == Some(view.kind.as_str())
                    });
                    let passed = kind_matches
                        && ended.is_none_or(|expected| expected == is_ended)
                        && retracted.is_none_or(|expected| expected == is_retracted)
                        && head.is_none_or(|expected| expected == is_head)
                        && phase.is_none_or(|expected| view.phase == Some(expected));
                    (
                        passed,
                        json!({
                            "id": view.id,
                            "present": true,
                            "kind": view.kind,
                            "ended": is_ended,
                            "retracted": is_retracted,
                            "head": is_head,
                            "phase": view.phase,
                        }),
                    )
                }
                None => (false, json!({ "present": false })),
            },
            Check::Absent { .. } => match self.view(memory)? {
                Some(view) => (false, json!({ "id": view.id, "present": true })),
                None => (true, json!({ "id": memory, "present": false })),
            },
            Check::AgendaHas { .. } | Check::AgendaLacks { .. } => {
                let listed = self.service.agenda(bank)?.listed();
                let has = memory.is_some_and(|id| listed.contains(&id));
                let wants = matches!(probe.check, Check::AgendaHas { .. });
                (has == wants, json!({ "id": memory, "listed": listed }))
            }
            Check::RecallFinds { query, .. } | Check::RecallLacks { query, .. } => {
                let recall = self.service.recall(
                    bank,
                    &RecallRequest {
                        query: query.clone(),
                        ..RecallRequest::default()
                    },
                )?;
                let results: Vec<Uuid> = recall.results.iter().map(|r| r.id).collect();
                let has = memory.is_some_and(|id| results.contains(&id));
                let wants = matches!(probe.check, Check::RecallFinds { .. });
                (has == wants, json!({ "id": memory, "results": results }))
            }
            Check::Injects { query, .. } | Check::NotInjects { query, .. } => {
                let prefetch = self.service.prefetch(
                    bank,
                    &PrefetchRequest {
                        session_id: format!("{PROBE_SESSION_PREFIX}{id}"),
                        query: query.clone(),
                        previous_query: None,
                        block_id: None,
                    },
                )?;
                let has = memory.is_some_and(|id| prefetch.injected.contains(&id));
                let wants = matches!(probe.check, Check::Injects { .. });
                (
                    has == wants,
                    json!({ "id": memory, "injected": prefetch.injected }),
                )
            }
            Check::ProfileHas { model, .. } | Check::ProfileLacks { model, .. } => {
                let view = self.service.show_model(bank, model, None)?;
                let cited: Vec<Uuid> = view
                    .entry_views
                    .iter()
                    .flat_map(|entry| entry.cites.iter().map(|cite| cite.id))
                    .collect();
                let has = memory.is_some_and(|id| cited.contains(&id));
                let wants = matches!(probe.check, Check::ProfileHas { .. });
                (has == wants, json!({ "id": memory, "cited": cited }))
            }
        };
        self.probes.push(ProbeResult {
            id,
            at: probe.at,
            kind: probe.check.kind(),
            passed,
            observed,
        });
        Ok(())
    }

    /// The memory's view, or `None` when it isn't in the store.
    fn view(
        &self,
        memory: Option<Uuid>,
    ) -> Result<Option<asphodel_core::inspect::MemoryView>, Failure> {
        let Some(id) = memory else {
            return Ok(None);
        };
        match self
            .service
            .show_memory(&self.settings.bank, &id.to_string())
        {
            Ok(view) => Ok(Some(view)),
            Err(InspectError::UnknownMemory) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    fn finish(self) -> Result<Outcome, Failure> {
        let mut fade_outs: BTreeMap<String, u64> = BTreeMap::new();
        let mut bands: BTreeMap<String, WeekBands> = BTreeMap::new();
        for snapshot in &self.snapshots {
            let iso = snapshot.day.iso_week_date();
            let week = format!("{}-W{:02}", iso.year(), iso.week());
            *fade_outs.entry(week.clone()).or_default() += snapshot.fade_outs;
            bands.insert(
                week.clone(),
                WeekBands {
                    week,
                    strong: snapshot.strong,
                    fading: snapshot.fading,
                    faded: snapshot.faded,
                },
            );
        }
        let mut lags = self.lags_ms.clone();
        lags.sort_unstable();
        let percentile = |p: f64| -> u64 { super::report::percentile(&lags, p) };

        let purged_at: BTreeMap<Uuid, Timestamp> = self
            .shadow
            .iter()
            .map(|row| (row.memory, row.purged_at))
            .collect();
        let mut memories = Vec::new();
        for created in &self.created {
            let faded_at = match self.service.faded_at(&self.settings.bank, created.memory) {
                Ok(faded_at) => faded_at,
                Err(InspectError::UnknownMemory) => None,
                Err(error) => return Err(error.into()),
            };
            memories.push(MemoryOutcome {
                id: created.memory,
                created_at: created.created_at,
                faded_at,
                purged_at: purged_at.get(&created.memory).copied(),
            });
        }

        let mut turn_tokens = self.turn_tokens;
        let mut profile_tokens = self.profile_tokens;
        let llm = match self.llm {
            Llm::Scripted => LlmCounts {
                scripted: self.llm_calls,
                ..LlmCounts::default()
            },
            Llm::Recorded(recorder, _) => recorder.llm_counts(),
        };
        Ok(Outcome {
            probes: self.probes,
            purges_per_day: self
                .purges
                .into_iter()
                .map(|(day, count)| DayCount {
                    day,
                    count,
                    purged: Some(count),
                })
                .collect(),
            fade_outs_per_week: fade_outs
                .into_iter()
                .map(|(week, count)| WeekCount { week, count })
                .collect(),
            bands_per_week: bands.into_values().collect(),
            extraction_lag: Lag {
                samples: lags.len() as u64,
                p50_ms: percentile(0.5),
                p95_ms: percentile(0.95),
            },
            refresh_calls_per_day: self
                .refresh_calls
                .into_iter()
                .map(|(day, count)| DayCount {
                    day,
                    count,
                    purged: None,
                })
                .collect(),
            injected_tokens: InjectedTokens {
                sessions: self
                    .sessions
                    .into_iter()
                    .filter(|(_, count)| count.synced)
                    .map(|(session, count)| SessionTokens {
                        session,
                        prefetches: count.prefetches,
                        tokens: count.tokens,
                    })
                    .collect(),
                per_turn: Percentiles::of(&mut turn_tokens),
                cron: super::report::CronTokens {
                    prefetches: self.cron_prefetches,
                    tokens: self.cron_tokens,
                },
            },
            profile_tokens: Percentiles::of(&mut profile_tokens),
            call2_rate: Call2Rate {
                chunks: self.chunks,
                call2: self.call2_chunks,
                rate: if self.chunks == 0 {
                    0.0
                } else {
                    self.call2_chunks as f64 / self.chunks as f64
                },
            },
            agenda_lines_per_day: self
                .agenda_lines
                .into_iter()
                .map(|(day, count)| DayCount {
                    day,
                    count,
                    purged: None,
                })
                .collect(),
            significance_histogram: self.significance_histogram,
            kind_histogram: self.kind_histogram,
            memories,
            llm,
            created: self.created,
            shadow: self.shadow,
            material: self.labelling.map(Collector::finish),
        })
    }
}

/// Call 1's reply from the chunk's claims.
fn call1_reply(claims: &[Claim], used: &[String]) -> Value {
    let when = |when: &Option<super::scenario::When>| match when {
        Some(when) => json!({ "at": when.at, "precision": when.precision.as_str() }),
        None => Value::Null,
    };
    let claims: Vec<Value> = claims
        .iter()
        .map(|claim| {
            json!({
                "content": claim.content,
                "kind": claim.kind,
                "quote": claim.quote,
                "significance": claim.significance,
                "remember_this": claim.remember_this,
                "changes_something": claim.changes_something,
                "valid_from": when(&claim.valid_from),
                "valid_until": when(&claim.valid_until),
                "window_confidence": if claim.low_confidence { "low" } else { "high" },
                "until_event": claim.until_event,
                "due_at": when(&claim.due_at),
                "volatility": claim.volatility,
                "recurrence_text": claim.recurrence_text,
                "recurrence_rrule": claim.recurrence_rrule,
                "recurrence_start": when(&claim.recurrence_start),
                "entities": [],
            })
        })
        .collect();
    json!({ "claims": claims, "used_injected_ids": used })
}

fn drop_reason(reason: DropReason) -> String {
    format!("{reason:?}")
}
