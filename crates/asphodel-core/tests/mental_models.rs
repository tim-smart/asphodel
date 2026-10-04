//! Agenda, mental models and the system prompt block, including refresh
//! scheduling and memory precedence. A mental model is a cache over
//! memories: reading or injecting it never counts as an access.
//!
//! These tests drive the service on a `SimulatedClock` and refresh models
//! with `FakeLlm`. Triggers go through the write paths that exist:
//! extraction (call 1, and call 2 for retractions, endings and
//! refinements), `keep`, and the owner's model edits. Memories a test only
//! needs present are inserted directly, as an earlier extraction would have
//! left them.
//!
//! The API under test is `asphodel_core::mental_models`,
//! `asphodel_core::agenda`, `asphodel_core::system_prompt` and the
//! `Service` methods over them. The tests check inclusive agenda
//! bounds, disabled models outside the budget, and code dropping entries
//! whose memories left a refresh's input.
//!
//! Every service here runs on a `SimulatedClock` stopped at 20:00 on
//! Thursday 1 October 2026 in Auckland (UTC+13) unless a test advances it.
//! The next 04:00 there is 15:00 UTC the same day, and the next local
//! midnight is 11:00 UTC.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use asphodel_core::ingest::Turn;
use asphodel_core::models::{
    Embedder, FakeEmbedder, FakeLlm, FakeReranker, LlmClient, LlmError, LlmGate, LlmRequest,
    LlmResponse, ModelError as EmbedError, Models, Reranker, Template,
};
use asphodel_core::retrieval::{PrefetchRequest, estimate_tokens};
use asphodel_core::store::bank::{BankIdentity, PROFILE_NAME, PROFILE_QUESTION};
use asphodel_core::store::{OpenOptions, Store, VectorIndex, micros};
use asphodel_core::strength::Kind;
use asphodel_core::{Service, SimulatedClock, Tuning};
use jiff::civil::DateTime;
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use rusqlite::types::FromSql;
use serde_json::{Value, json};
use uuid::Uuid;

use asphodel_core::agenda::Agenda;
use asphodel_core::mental_models::{
    Applied, FailureKind, Model, ModelEdit, ModelError, ModelSpec, Outcome, REFRESH_TEMPLATE,
    RefreshInput, Refreshes, RejectReason,
};
use asphodel_core::system_prompt::Block;

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";
const BANK: &str = "main";
const MODEL: &str = "fake-llm";

/// The next 04:00 in Auckland after [`START`], when the daily sweep runs.
const SWEEP: &str = "2026-10-01T15:00:00Z";

/// When fixture memories were said, unless a test says otherwise.
const EARLIER: &str = "2026-09-01T00:00:00Z";

/// Long enough ago that a trivial memory said then is below τ.
const LONG_AGO: &str = "2021-01-01T00:00:00Z";

const BERLIN: &str = "Tim lives in Berlin.";
const MOVED: &str = "Tim moved out of Berlin and now lives in Lisbon.";
const MAYA: &str = "Tim's daughter is called Maya.";
const MIA: &str = "Tim's daughter is called Mia.";
const JAPAN: &str = "Tim is going to Japan in 2027.";
const TOKYO: &str = "Tim is going to Tokyo, Japan in April 2027.";
const CAT: &str = "Tim's cat is called Miso.";
const TEA: &str = "Tim likes green tea.";

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
}

/// A local date-time in `TZ` as the instant stored for it.
fn local(datetime: &str) -> Timestamp {
    datetime
        .parse::<DateTime>()
        .unwrap()
        .to_zoned(TimeZone::get(TZ).unwrap())
        .unwrap()
        .timestamp()
}

fn minutes(n: i64) -> SignedDuration {
    SignedDuration::from_mins(n)
}

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-models-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn next_uuid() -> Uuid {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    Uuid::from_u128((0xf4_u128 << 120) | u128::from(NEXT.fetch_add(1, Ordering::Relaxed)))
}

/// A memory to insert. `Default` is a notable fact said at [`EARLIER`],
/// with high window confidence and its `created` access then.
#[derive(Clone)]
struct Memory {
    content: &'static str,
    kind: &'static str,
    significance: &'static str,
    observed_at: Timestamp,
    valid_from: Option<(Timestamp, &'static str)>,
    valid_until: Option<(Timestamp, &'static str)>,
    due_at: Option<(Timestamp, &'static str)>,
    volatility: Option<&'static str>,
    recurrence_text: Option<&'static str>,
    recurrence_rrule: Option<&'static str>,
    recurrence_start: Option<(Timestamp, &'static str)>,
    retracted: bool,
    hidden: bool,
}

impl Default for Memory {
    fn default() -> Self {
        Self {
            content: "",
            kind: "fact",
            significance: "notable",
            observed_at: at(EARLIER),
            valid_from: None,
            valid_until: None,
            due_at: None,
            volatility: None,
            recurrence_text: None,
            recurrence_rrule: None,
            recurrence_start: None,
            retracted: false,
            hidden: false,
        }
    }
}

fn fact(content: &'static str) -> Memory {
    Memory {
        content,
        ..Memory::default()
    }
}

/// A sentence made `'static`, for fixtures built in a loop.
fn sentence(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

/// A trivial memory said long enough ago to be below τ.
fn faded(memory: Memory) -> Memory {
    Memory {
        significance: "trivial",
        observed_at: at(LONG_AGO),
        ..memory
    }
}

/// An event starting at `when` (local), to the day.
fn event(content: &'static str, when: &str) -> Memory {
    Memory {
        content,
        kind: "event",
        valid_from: Some((local(when), "day")),
        ..Memory::default()
    }
}

/// An open task due at `when` (local), to the day.
fn task_due(content: &'static str, when: &str) -> Memory {
    Memory {
        content,
        kind: "task",
        due_at: Some((local(when), "day")),
        ..Memory::default()
    }
}

/// An open task with no due date.
fn task(content: &'static str) -> Memory {
    Memory {
        content,
        kind: "task",
        ..Memory::default()
    }
}

/// A recurring memory with `rrule`, first occurring at `start` (local).
fn recurring(content: &'static str, rrule: Option<&'static str>, start: &str) -> Memory {
    Memory {
        content,
        kind: "recurring",
        recurrence_text: Some(content),
        recurrence_rrule: rrule,
        recurrence_start: rrule.map(|_| (local(start), "day")),
        ..Memory::default()
    }
}

struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    tuning: Tuning,
    chunk: i64,
    _dir: TestDir,
}

impl Harness {
    fn new() -> Self {
        Self::with_tuning("")
    }

    /// `extra` is more tuning TOML, appended to the floors the fakes need.
    fn with_tuning(extra: &str) -> Self {
        Self::build(1.0, extra, Models::fake())
    }

    /// [`Harness::with_tuning`] with a relevance scale of `scale` for the
    /// fake reranker.
    fn with_scale(scale: f64, extra: &str) -> Self {
        Self::build(scale, extra, Models::fake())
    }

    /// Default tuning on `models`.
    fn with_models(models: Models) -> Self {
        Self::build(1.0, "", models)
    }

    fn build(scale: f64, extra: &str, models: Models) -> Self {
        let tuning = Tuning::from_toml(&format!(
            "[injection.reranker_floors]\n\"{}\" = 1.0\n\
             [ranking.relevance_scales]\n\"{0}\" = {scale:?}\n\
             [reconcile.embedding_floors]\n\"{}\" = 0.5\n{extra}",
            FakeReranker::MODEL_ID,
            FakeEmbedder::MODEL_ID,
        ))
        .unwrap();
        let dir = TestDir::new();
        let clock = Arc::new(SimulatedClock::new(at(START)));
        let store = Store::open(&dir.0, OpenOptions::default(), clock.clone()).unwrap();
        let service = Service::with_models(clock.clone(), store, tuning.clone(), models).unwrap();
        service
            .ensure_bank_with_models(
                BANK,
                &BankIdentity {
                    owner_name: Some("Tim".into()),
                    assistant_name: Some("Hermes".into()),
                    timezone: Some(TZ.into()),
                    ..BankIdentity::default()
                },
            )
            .unwrap();
        // The chunk fixture memories rest on: a turn said at EARLIER, taken
        // off the queue as if extracted.
        service
            .ingest_turn(BANK, &turn("fixtures", at(EARLIER), "Fixtures."))
            .unwrap();
        let mut harness = Self {
            service,
            clock,
            tuning,
            chunk: 0,
            _dir: dir,
        };
        harness.chunk = harness.one("SELECT id FROM chunks", []);
        harness.execute("DELETE FROM extraction_queue", []);
        harness
    }

    /// The daemon restarting: the service and its in-memory state (sessions,
    /// the block cache, pending debounces) go; the store stays.
    fn restart(self) -> Self {
        let Self {
            service,
            clock,
            tuning,
            chunk,
            _dir,
        } = self;
        drop(service);
        let store = Store::open(&_dir.0, OpenOptions::default(), clock.clone()).unwrap();
        let service =
            Service::with_models(clock.clone(), store, tuning.clone(), Models::fake()).unwrap();
        Self {
            service,
            clock,
            tuning,
            chunk,
            _dir,
        }
    }

    fn now(&self) -> Timestamp {
        self.service.now()
    }

    fn advance(&self, by: SignedDuration) {
        self.clock.advance(by);
    }

    fn set(&self, to: Timestamp) {
        self.clock.set(to);
    }

    fn one<T: FromSql, P: rusqlite::Params>(&self, sql: &str, params: P) -> T {
        self.service
            .store()
            .unwrap()
            .connection()
            .query_row(sql, params, |row| row.get(0))
            .unwrap()
    }

    fn execute<P: rusqlite::Params>(&self, sql: &str, params: P) -> usize {
        self.service
            .store()
            .unwrap()
            .connection()
            .execute(sql, params)
            .unwrap()
    }

    fn bank_id(&self) -> i64 {
        self.one("SELECT id FROM banks WHERE name = ?1", [BANK])
    }

    fn rowid(&self, memory: Uuid) -> i64 {
        self.one(
            "SELECT id FROM memories WHERE uuid = ?1",
            [memory.to_string()],
        )
    }

    fn insert(&self, memory: Memory) -> Uuid {
        let uuid = next_uuid();
        let bank_id = self.bank_id();
        let now = micros(self.now());
        let stamp = |t: Option<(Timestamp, &'static str)>| {
            (
                t.map(|(at, _)| micros(at)),
                t.map(|(_, precision)| precision),
            )
        };
        let (valid_from, valid_from_precision) = stamp(memory.valid_from);
        let (valid_until, valid_until_precision) = stamp(memory.valid_until);
        let (due_at, due_at_precision) = stamp(memory.due_at);
        let (recurrence_start, recurrence_start_precision) = stamp(memory.recurrence_start);
        let store = self.service.store().unwrap();
        let conn = store.connection();
        conn.execute(
            "INSERT INTO memories (uuid, bank_id, content, kind, significance, chunk_id,
                                   source_start, source_end, observed_at, valid_from,
                                   valid_from_precision, valid_until, valid_until_precision,
                                   window_confidence, due_at, due_at_precision, volatility,
                                   recurrence_text, recurrence_rrule, recurrence_start,
                                   recurrence_start_precision, invalidated_at, hidden_at,
                                   created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, 9, ?7, ?8, ?9, ?10, ?11, 'high', ?12, ?13, ?14,
                     ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?21)",
            rusqlite::params![
                uuid.to_string(),
                bank_id,
                memory.content,
                memory.kind,
                memory.significance,
                self.chunk,
                micros(memory.observed_at),
                valid_from,
                valid_from_precision,
                valid_until,
                valid_until_precision,
                due_at,
                due_at_precision,
                memory.volatility,
                memory.recurrence_text,
                memory.recurrence_rrule,
                recurrence_start,
                recurrence_start_precision,
                memory.retracted.then_some(now),
                memory.hidden.then_some(now),
                now,
            ],
        )
        .unwrap();
        let id = conn.last_insert_rowid();
        let vector = FakeEmbedder.embed(&[memory.content]).unwrap().remove(0);
        store.vectors().upsert(&conn, bank_id, id, &vector).unwrap();
        conn.execute(
            "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
             VALUES (?1, ?2, 'created', ?3, 0)",
            (bank_id, id, micros(memory.observed_at)),
        )
        .unwrap();
        uuid
    }

    /// `memory` ended by `by` at `until`, as reconciliation leaves it.
    fn mark_ended(&self, memory: Uuid, by: Uuid, until: Timestamp) {
        self.execute(
            "UPDATE memories SET valid_until = ?2, valid_until_precision = 'day', ended_by = ?3
             WHERE id = ?1",
            (self.rowid(memory), micros(until), self.rowid(by)),
        );
    }

    /// The owner says `quote` a minute ago in session `chat`, and the turn is
    /// extracted with call 1 finding `claim`. Call 2, if the claim lands
    /// near something stored, labels nothing. Returns the new memory.
    fn says(&self, claim: Value) -> Uuid {
        self.ingest_said(&claim);
        let llm = FakeLlm::scripted(
            MODEL,
            vec![
                json!({"claims": [claim], "used_injected_ids": []}),
                json!({"claims": []}),
            ],
        );
        let extracted = self.service.extract_next(BANK, &llm).unwrap();
        extracted.expect("the turn was queued").memories[0]
    }

    /// As [`Harness::says`], with call 2 labelling the claim `label` on
    /// `neighbour`: `retracts`, `ends` or `refines`.
    fn says_changing(&self, claim: Value, neighbour: Uuid, label: &str) -> Uuid {
        self.ingest_said(&claim);
        let call1 = json!({"claims": [claim], "used_injected_ids": []});
        let (claim_handle, neighbour_handle) = {
            let lease = self.service.claim_chunk(BANK).unwrap().expect("queued");
            let input = self
                .service
                .call2_input(&lease, &call1, &[])
                .unwrap()
                .expect("call 2 runs");
            let neighbour = input
                .neighbours
                .iter()
                .find(|n| n.memory == neighbour)
                .expect("the memory is a neighbour")
                .handle
                .clone();
            (input.claims[0].handle.clone(), neighbour)
        };
        let llm = FakeLlm::scripted(
            MODEL,
            vec![
                call1,
                json!({"claims": [{
                    "claim": claim_handle,
                    "labels": [{"neighbour": neighbour_handle, "label": label}],
                }]}),
            ],
        );
        let extracted = self.service.extract_next(BANK, &llm).unwrap();
        extracted.expect("the turn was queued").memories[0]
    }

    fn ingest_said(&self, claim: &Value) {
        let quote = claim["quote"].as_str().unwrap();
        self.service
            .ingest_turn(BANK, &turn("chat", self.now() - minutes(1), quote))
            .unwrap();
    }

    fn keep(&self, memory: Uuid) {
        self.service.keep(BANK, &[memory.to_string()]).unwrap();
    }

    fn profile(&self) -> Model {
        self.model(PROFILE_NAME)
    }

    fn model(&self, name: &str) -> Model {
        self.service
            .list_models(BANK)
            .unwrap()
            .into_iter()
            .find(|model| model.name == name)
            .unwrap_or_else(|| panic!("no model {name}"))
    }

    fn input(&self, name: &str) -> RefreshInput {
        self.service.refresh_input(BANK, name).unwrap()
    }

    /// Forces a refresh of `name` whose reply adds one entry per
    /// `(text, cites)`, and returns what it applied.
    fn refresh_adding(&self, name: &str, entries: &[(&str, &[Uuid])]) -> Applied {
        let input = self.input(name);
        let operations = entries
            .iter()
            .map(|(text, cites)| add(text, &handles(&input, cites)))
            .collect();
        let llm = FakeLlm::scripted(MODEL, vec![reply(operations)]);
        match self.service.refresh_model(BANK, name, &llm, true).unwrap() {
            Outcome::Applied(applied) => applied,
            other => panic!("the refresh didn't apply: {other:?}"),
        }
    }

    /// Runs every refresh due now with `llm`.
    fn tick(&self, llm: &FakeLlm) -> Refreshes {
        self.service.run_refreshes(llm).unwrap()
    }

    fn block(&self, session: Option<&str>) -> Block {
        self.service.system_prompt(BANK, session).unwrap()
    }

    fn agenda(&self) -> Agenda {
        self.service.agenda(BANK).unwrap()
    }

    fn in_context(&self, session: &str) -> Vec<Uuid> {
        self.service.in_context(BANK, session).unwrap()
    }

    fn accesses(&self) -> i64 {
        self.one("SELECT COUNT(*) FROM accesses", [])
    }

    fn used(&self, memory: Uuid) -> i64 {
        self.one(
            "SELECT COUNT(*) FROM accesses WHERE memory_id = ?1 AND kind = 'used'",
            [self.rowid(memory)],
        )
    }
}

/// The owner's turn on the CLI: no author.
fn turn(session: &str, message_at: Timestamp, user: &str) -> Turn {
    Turn {
        session_id: session.into(),
        message_at,
        timezone: Some(TZ.into()),
        user_text: user.into(),
        assistant_text: "Noted.".into(),
        author: None,
        platform: Some("cli".into()),
        recall_id: None,
        forget_requested: false,
    }
}

/// Call 1's claim: the sentence is also the quote, as the owner said it.
fn claim(content: &str, kind: &str, significance: &str) -> Value {
    json!({
        "content": content,
        "kind": kind,
        "quote": content,
        "significance": significance,
        "remember_this": false,
        "changes_something": false,
        "valid_from": null,
        "valid_until": null,
        "window_confidence": "high",
        "until_event": null,
        "due_at": null,
        "volatility": null,
        "recurrence_text": null,
        "recurrence_rrule": null,
        "recurrence_start": null,
        "entities": [],
    })
}

fn notable(content: &str) -> Value {
    claim(content, "fact", "notable")
}

trait With {
    fn with(self, key: &str, value: Value) -> Value;
}

impl With for Value {
    fn with(mut self, key: &str, value: Value) -> Value {
        self[key] = value;
        self
    }
}

// The refresh reply.

fn reply(operations: Vec<Value>) -> Value {
    json!({"operations": operations})
}

fn add(text: &str, cites: &[String]) -> Value {
    json!({"op": "add", "text": text, "cites": cites})
}

fn edit(entry: &str, text: &str, cites: &[String]) -> Value {
    json!({"op": "edit", "entry": entry, "text": text, "cites": cites})
}

fn remove(entry: &str) -> Value {
    json!({"op": "remove", "entry": entry})
}

/// An LLM that answers every refresh with no operations.
fn quiet_llm(calls: usize) -> FakeLlm {
    FakeLlm::scripted(MODEL, vec![reply(vec![]); calls])
}

fn handle(input: &RefreshInput, memory: Uuid) -> String {
    input
        .memories
        .iter()
        .find(|m| m.memory == memory)
        .unwrap_or_else(|| panic!("{memory} is in the refresh input"))
        .handle
        .clone()
}

fn handles(input: &RefreshInput, memories: &[Uuid]) -> Vec<String> {
    memories.iter().map(|m| handle(input, *m)).collect()
}

fn entry_handle(input: &RefreshInput, entry: Uuid) -> String {
    input
        .entries
        .iter()
        .find(|e| e.entry == entry)
        .unwrap_or_else(|| panic!("{entry} is in the refresh input"))
        .handle
        .clone()
}

fn inputs(input: &RefreshInput) -> BTreeSet<Uuid> {
    input.memories.iter().map(|m| m.memory).collect()
}

fn texts(model: &Model) -> Vec<&str> {
    model.entries.iter().map(|e| e.text.as_str()).collect()
}

fn refresh_calls(llm: &FakeLlm) -> usize {
    llm.requests()
        .iter()
        .filter(|r| r.template.name == REFRESH_TEMPLATE)
        .count()
}

fn plans_model() -> ModelSpec {
    ModelSpec {
        name: "Plans".into(),
        question: "Where is Tim going and when?".into(),
        kinds: vec![Kind::Event],
        entity: None,
        min_volatility: None,
        max_tokens: 100,
        enabled: true,
    }
}

// Refresh triggers and scheduling

#[test]
fn a_refresh_scores_relevance_as_the_logit_divided_by_the_relevance_scale() {
    // The refresh's selection scores like prefetch and recall: at scale 1.0
    // relevance is the raw logit, and any other scale divides it.
    let score = |scale: f64| -> f64 {
        let h = Harness::with_scale(scale, "");
        let tea = h.insert(fact(TEA));
        h.refresh_adding(PROFILE_NAME, &[("Tim likes green tea.", &[tea])]);
        h.one::<Option<f64>, _>(
            "SELECT r.score FROM recall_results r
             JOIN recalls c ON c.id = r.recall_id JOIN memories m ON m.id = r.memory_id
             WHERE c.kind = 'refresh' AND m.uuid = ?1",
            [tea.to_string()],
        )
        .unwrap()
    };
    let logit = f64::from(FakeReranker.rerank(PROFILE_QUESTION, &[TEA]).unwrap()[0]);
    let (raw, scaled) = (score(1.0), score(4.0));
    let tuning = Tuning::default();
    let strength = asphodel_core::strength::strength(
        asphodel_core::constants::Significance::Notable.value(),
        &[asphodel_core::strength::Access {
            kind: asphodel_core::strength::AccessKind::Created,
            at: at(EARLIER),
        }],
        None,
        &asphodel_core::strength::BankTime::new(&[at(EARLIER)], tuning.clock.quiet_rate),
        at(START),
    )
    .value;
    // A fact without volatility or a window has zero confidence/phase terms.
    assert!((raw - (logit + 0.5 * strength)).abs() < 1e-9);
    let expected = logit - logit / 4.0;
    assert!(
        (raw - scaled - expected).abs() < 1e-9,
        "{raw} - {scaled} isn't {expected}"
    );
}

#[test]
fn a_notable_memory_triggers_a_refresh_five_minutes_later() {
    let h = Harness::new();
    let said = h.now();
    h.says(notable(TEA));

    h.advance(minutes(4) + SignedDuration::from_secs(59));
    let llm = quiet_llm(1);
    let early = h.tick(&llm);
    assert!(early.ran.is_empty());
    assert_eq!(refresh_calls(&llm), 0);
    assert_eq!(early.next_due, Some(said + minutes(5)));

    h.set(said + minutes(5));
    let ran = h.tick(&llm);
    assert_eq!(ran.ran.len(), 1);
    assert_eq!(ran.ran[0].model, PROFILE_NAME);
    assert_eq!(refresh_calls(&llm), 1);
    assert_eq!(h.profile().last_refreshed_at, Some(said + minutes(5)));
}

#[test]
fn each_trigger_pushes_the_refresh_back_but_never_past_thirty_minutes() {
    // A long conversation that keeps adding notable facts never goes quiet,
    // so the debounce is capped 30 minutes after the first trigger.
    let h = Harness::new();
    let first = h.now();
    let llm = quiet_llm(1);
    for (n, content) in [
        "Tim plays the cello.",
        "Tim keeps bees.",
        "Tim grows chillies.",
        "Tim restores old radios.",
        "Tim speaks Portuguese.",
        "Tim runs on Sundays.",
        "Tim volunteers at the library.",
        "Tim collects maps.",
    ]
    .into_iter()
    .enumerate()
    {
        h.set(first + minutes(4 * n as i64));
        h.says(notable(content));
        h.advance(minutes(1));
        assert!(h.tick(&llm).ran.is_empty(), "refreshed after trigger {n}");
    }
    // The last trigger was at +28 minutes, so the debounce alone would wait
    // until +33.
    h.set(first + minutes(30) - SignedDuration::from_secs(1));
    assert!(h.tick(&llm).ran.is_empty());
    h.set(first + minutes(30));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    assert_eq!(refresh_calls(&llm), 1);
}

#[test]
fn a_memory_below_the_trigger_level_waits_for_the_sweep() {
    let h = Harness::new();
    h.says(claim(TEA, "fact", "minor"));
    h.advance(minutes(5));
    let llm = quiet_llm(1);
    let refreshes = h.tick(&llm);
    assert!(refreshes.ran.is_empty());
    assert_eq!(refreshes.next_due, Some(at(SWEEP)));

    // The sweep lets minor additions in.
    h.set(at(SWEEP));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    assert_eq!(refresh_calls(&llm), 1);
    assert!(llm.requests()[0].user.contains(TEA));
}

#[test]
fn a_memory_the_models_filters_leave_out_triggers_nothing() {
    // The profile takes facts and states of volatility weeks or slower.
    let h = Harness::new();
    h.says(claim("Tim went to the cinema.", "event", "major"));
    h.says(claim("Tim is tired.", "state", "major").with("volatility", json!("days")));
    h.advance(minutes(5));
    assert!(h.tick(&quiet_llm(1)).ran.is_empty());

    h.says(
        claim("Tim is training for a marathon.", "state", "notable")
            .with("volatility", json!("months")),
    );
    h.advance(minutes(5));
    assert_eq!(h.tick(&quiet_llm(1)).ran.len(), 1);
}

#[test]
fn an_owner_edit_triggers_a_refresh_that_isnt_skipped() {
    // The question is in the fingerprint, so the same memories still get an
    // LLM call.
    let h = Harness::new();
    h.insert(fact(TEA));
    h.refresh_adding(PROFILE_NAME, &[]);
    let before = h.input(PROFILE_NAME).fingerprint;

    h.service
        .edit_model(
            BANK,
            PROFILE_NAME,
            &ModelEdit {
                question: Some("What does Tim like to drink?".into()),
                ..ModelEdit::default()
            },
        )
        .unwrap();
    let asked = h.input(PROFILE_NAME).fingerprint;
    assert_ne!(asked, before);
    // So is the size.
    h.service
        .edit_model(
            BANK,
            PROFILE_NAME,
            &ModelEdit {
                max_tokens: Some(400),
                ..ModelEdit::default()
            },
        )
        .unwrap();
    assert_ne!(h.input(PROFILE_NAME).fingerprint, asked);

    h.advance(minutes(30));
    let llm = quiet_llm(1);
    let ran = h.tick(&llm);
    assert_eq!(ran.ran.len(), 1);
    assert!(matches!(ran.ran[0].outcome, Outcome::Applied(_)));
    assert!(
        llm.requests()[0]
            .user
            .contains("What does Tim like to drink?")
    );
}

#[test]
fn a_model_is_refreshed_at_most_every_thirty_minutes() {
    let h = Harness::new();
    h.says(notable(TEA));
    h.advance(minutes(5));
    let llm = quiet_llm(2);
    assert_eq!(h.tick(&llm).ran.len(), 1);
    let refreshed = h.now();

    h.advance(minutes(1));
    h.says(notable("Tim keeps bees."));
    h.set(refreshed + minutes(6));
    let waiting = h.tick(&llm);
    assert!(
        waiting.ran.is_empty(),
        "the debounce alone would run it now"
    );
    assert_eq!(waiting.next_due, Some(refreshed + minutes(30)));

    h.set(refreshed + minutes(30));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    assert_eq!(refresh_calls(&llm), 2);
}

#[test]
fn a_failed_refresh_is_retried_after_thirty_minutes_not_at_the_next_trigger() {
    let h = Harness::new();
    h.says(notable(TEA));
    h.advance(minutes(5));
    let failed_at = h.now();
    let down = FakeLlm::failing(MODEL, || LlmError::Transport {
        reason: "connection refused".into(),
    });
    let ran = h.service.run_refreshes(&down).unwrap();
    assert_eq!(ran.ran[0].outcome, Outcome::Failed(FailureKind::Llm));
    let profile = h.profile();
    assert_eq!(profile.last_error, Some(FailureKind::Llm));
    assert_eq!(profile.last_error_at, Some(failed_at));
    assert_eq!(profile.last_refreshed_at, None);
    // `status` counts it and says it needs attention.
    let status = h.service.status().unwrap();
    assert_eq!(status.banks[BANK].failed_refreshes, 1);
    assert_eq!(status.attention.len(), 1, "{:?}", status.attention);
    assert!(
        status.attention[0].contains("refresh"),
        "{:?}",
        status.attention
    );

    h.advance(minutes(1));
    h.says(notable("Tim keeps bees."));
    h.set(failed_at + minutes(6));
    let llm = quiet_llm(1);
    let waiting = h.tick(&llm);
    assert!(waiting.ran.is_empty());
    assert_eq!(waiting.next_due, Some(failed_at + minutes(30)));

    h.set(failed_at + minutes(30));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    let profile = h.profile();
    assert_eq!(profile.last_error, None);
    assert_eq!(profile.last_refreshed_at, Some(failed_at + minutes(30)));
    let status = h.service.status().unwrap();
    assert_eq!(status.banks[BANK].failed_refreshes, 0);
    assert_eq!(status.attention, Vec::<String>::new());
}

/// The daemon's gate holds every call once any call hits a limit,
/// extraction's included. A refresh it holds never reached the LLM, so it
/// isn't a failure: nothing needs attention, and it runs once the hold
/// lifts rather than thirty minutes after.
fn a_refresh_held_by_the_gate_waits_for_the_hold_not_thirty_minutes(
    limit: Value,
    lifts: Timestamp,
) {
    let h = Harness::new();
    h.says(notable(TEA));
    h.advance(minutes(5));
    let inner = Arc::new(
        FakeLlm::from_script(MODEL, &json!([limit, {"reply": reply(vec![])}]).to_string()).unwrap(),
    );
    let gate = LlmGate::new(inner.clone(), 1, h.clock.clone());
    let extraction = LlmRequest {
        template: Template {
            name: "extract_claims".into(),
            version: 2,
            guidance: None,
        },
        system: String::new(),
        user: String::new(),
        schema_name: "claims".into(),
        schema: json!({}),
        max_tokens: None,
    };
    gate.complete(&extraction)
        .expect_err("the extraction call hits the limit");

    let held = h.service.run_refreshes(&gate).unwrap();
    assert_eq!(
        inner.requests().len(),
        1,
        "the refresh never reached the LLM"
    );
    assert!(
        held.ran
            .iter()
            .all(|run| !matches!(run.outcome, Outcome::Failed(_))),
        "{:?}",
        held.ran
    );
    assert_eq!(h.profile().last_error, None);
    let status = h.service.status().unwrap();
    assert_eq!(status.banks[BANK].failed_refreshes, 0);
    assert_eq!(status.attention, Vec::<String>::new());
    assert_eq!(held.next_due, Some(lifts), "due again when the hold lifts");

    h.set(lifts);
    let ran = h.service.run_refreshes(&gate).unwrap();
    assert_eq!(ran.ran.len(), 1);
    assert_eq!(refresh_calls(&inner), 1);
    assert_eq!(h.profile().last_refreshed_at, Some(lifts));
}

#[test]
fn a_refresh_held_by_a_usage_limit_on_extraction_waits_for_the_reset() {
    let resets_at = at(START) + minutes(7);
    a_refresh_held_by_the_gate_waits_for_the_hold_not_thirty_minutes(
        json!({"fail": "usage_limited", "resets_at": resets_at.to_string()}),
        resets_at,
    );
}

#[test]
fn a_refresh_held_by_a_rate_limit_on_extraction_waits_for_retry_after() {
    // The limit is hit five minutes after the start, after the fact said.
    a_refresh_held_by_the_gate_waits_for_the_hold_not_thirty_minutes(
        json!({"fail": "status", "status": 429, "retry_after_secs": 120}),
        at(START) + minutes(5) + minutes(2),
    );
}

#[test]
fn the_daily_sweep_runs_at_four_bank_local() {
    let h = Harness::new();
    h.insert(fact(TEA));
    assert_eq!(h.tick(&quiet_llm(1)).next_due, Some(at(SWEEP)));

    h.set(at(SWEEP) - SignedDuration::from_secs(1));
    let llm = quiet_llm(2);
    assert!(h.tick(&llm).ran.is_empty());
    h.set(at(SWEEP));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    assert_eq!(refresh_calls(&llm), 1);

    // The next one is 04:00 the next local day.
    assert_eq!(
        h.tick(&llm).next_due,
        Some(at(SWEEP) + SignedDuration::from_hours(24))
    );
}

#[test]
fn a_disabled_model_is_never_refreshed() {
    let h = Harness::new();
    h.service
        .edit_model(
            BANK,
            PROFILE_NAME,
            &ModelEdit {
                enabled: Some(false),
                ..ModelEdit::default()
            },
        )
        .unwrap();
    h.says(notable(TEA));
    h.advance(minutes(5));
    let llm = quiet_llm(1);
    assert!(h.tick(&llm).ran.is_empty());
    h.set(at(SWEEP));
    assert!(h.tick(&llm).ran.is_empty());
    assert_eq!(refresh_calls(&llm), 0);
}

#[test]
fn a_refresh_never_runs_inside_a_block_fetch() {
    // The plugin fetches the block with a 2 s budget.
    let h = Harness::new();
    h.says(notable(TEA));
    h.advance(minutes(10));
    h.block(Some("s1"));
    assert_eq!(h.profile().last_refreshed_at, None);
}

// The refresh input and the fingerprint

#[test]
fn a_refresh_selects_current_memories_above_tau_that_pass_the_filters() {
    let h = Harness::new();
    let tea = h.insert(fact(TEA));
    let marathon = h.insert(Memory {
        kind: "state",
        volatility: Some("months"),
        ..fact("Tim is training for a marathon.")
    });
    let sleepy = h.insert(Memory {
        kind: "state",
        volatility: Some("days"),
        ..fact("Tim is sleepy.")
    });
    let no_volatility = h.insert(Memory {
        kind: "state",
        ..fact("Tim is between jobs.")
    });
    let cinema = h.insert(event("Tim is going to the cinema.", "2026-10-03T00:00"));
    let gone = h.insert(faded(fact("Tim once tried surfing in Raglan.")));
    let wrong = h.insert(Memory {
        retracted: true,
        ..fact("Tim's sister is called Ana.")
    });
    let hidden = h.insert(Memory {
        hidden: true,
        ..fact("Tim's bike is a Brompton.")
    });
    let berlin = h.insert(fact(BERLIN));
    let lisbon = h.insert(fact("Tim lives in Lisbon."));
    h.mark_ended(berlin, lisbon, local("2026-09-12T00:00"));

    let selected = inputs(&h.input(PROFILE_NAME));
    assert!(selected.contains(&tea));
    assert!(selected.contains(&marathon));
    assert!(selected.contains(&no_volatility), "null volatility passes");
    assert!(selected.contains(&lisbon));
    for (memory, why) in [
        (sleepy, "faster than weeks"),
        (cinema, "an event"),
        (gone, "below τ"),
        (wrong, "retracted"),
        (hidden, "forgotten"),
        (berlin, "ended"),
    ] {
        assert!(
            !selected.contains(&memory),
            "selected a memory that's {why}"
        );
    }
}

#[test]
fn the_profile_admits_recurring_memories_with_periods_longer_than_a_week() {
    let h = Harness::new();
    let expected: BTreeSet<_> = [
        (
            "Alex and Jo's wedding anniversary is on 12 June each year.",
            "FREQ=YEARLY;BYMONTH=6;BYMONTHDAY=12",
        ),
        ("Alex attends a monthly book club.", "FREQ=MONTHLY"),
        ("Alex visits Jo every other week.", "FREQ=WEEKLY;INTERVAL=2"),
        (
            "Alex waters the cactus every eight days.",
            "FREQ=DAILY;INTERVAL=8",
        ),
    ]
    .into_iter()
    .map(|(content, rule)| h.insert(recurring(content, Some(rule), "2026-06-12T00:00")))
    .collect();

    assert_eq!(inputs(&h.input(PROFILE_NAME)), expected);
}

// Synthetic dates, unrelated to any private anniversary. Extraction runs on
// 1 October in Auckland, so 5 October is inside the default agenda horizon.
const ANNIVERSARY_RULE: &str = "FREQ=YEARLY;BYMONTH=10;BYMONTHDAY=5";

fn extract_startless_anniversary(h: &Harness) -> Uuid {
    h.says(
        claim(
            "Alex and Jo celebrate their anniversary on 5 October.",
            "recurring",
            "critical",
        )
        .with("recurrence_text", json!("every 5 October"))
        .with("recurrence_rrule", json!(ANNIVERSARY_RULE)),
    )
}

#[test]
fn a_startless_anniversary_survives_extraction() {
    let h = Harness::new();
    let memory = extract_startless_anniversary(&h);
    let view = h.service.show_memory(BANK, &memory.to_string()).unwrap();
    assert_eq!(view.kind, "recurring");
    assert_eq!(view.window.recurrence.as_deref(), Some("every 5 October"));
    assert_eq!(
        view.window.recurrence_rrule.as_deref(),
        Some(ANNIVERSARY_RULE)
    );
}

#[test]
fn a_startless_anniversary_reaches_the_seeded_profile_input() {
    let h = Harness::new();
    let memory = extract_startless_anniversary(&h);
    assert_eq!(inputs(&h.input(PROFILE_NAME)), BTreeSet::from([memory]));
}

#[test]
fn a_startless_anniversary_is_a_dated_agenda_occasion_not_a_routine() {
    let h = Harness::new();
    let memory = extract_startless_anniversary(&h);
    assert_eq!(h.agenda().dated, vec![memory]);
    assert!(h.agenda().routines.is_empty());
    // Deriving a start must not turn a yearly occasion into a daily one.
    h.set(local("2026-10-06T00:00"));
    assert!(h.agenda().dated.is_empty());
    h.set(local("2027-10-01T00:00"));
    assert_eq!(h.agenda().dated, vec![memory]);
}

#[test]
fn a_startless_anniversary_learned_at_noon_starts_today_and_is_on_todays_agenda() {
    let h = Harness::new();
    h.set(local("2026-10-05T12:00"));
    let memory = extract_startless_anniversary(&h);
    // The public memory view does not expose DTSTART. Read just the stored
    // stamp here; extraction and the dated agenda still use Service APIs.
    let start: Option<i64> = h.one(
        "SELECT recurrence_start FROM memories WHERE uuid = ?1",
        [memory.to_string()],
    );
    let precision: Option<String> = h.one(
        "SELECT recurrence_start_precision FROM memories WHERE uuid = ?1",
        [memory.to_string()],
    );
    let agenda = h.agenda();
    // 5 October midnight in Auckland is 4 October 11:00 UTC. A day-only
    // occasion remains relevant at noon even though midnight has passed.
    assert_eq!(
        (start, precision.as_deref(), agenda.dated, agenda.routines),
        (
            Some(micros(at("2026-10-04T11:00:00Z"))),
            Some("day"),
            vec![memory],
            vec![],
        ),
    );
}

#[test]
fn startless_rules_with_unknown_dates_or_interval_phase_keep_only_their_text() {
    for rule in [
        "FREQ=WEEKLY;INTERVAL=2",
        // BY parts do not resolve an interval phase, even with a full date.
        "FREQ=YEARLY;INTERVAL=2;BYMONTH=10;BYMONTHDAY=5",
        "FREQ=DAILY;INTERVAL=2;BYHOUR=9",
        // These BY parts still inherit a day or weekday from DTSTART.
        "FREQ=YEARLY;BYMONTH=10",
        "FREQ=MONTHLY;BYMONTH=10",
        "FREQ=WEEKLY;BYHOUR=9",
    ] {
        let h = Harness::new();
        let memory = h.says(
            claim(
                "Alex has a recurring occasion with no stated first date.",
                "recurring",
                "critical",
            )
            .with("recurrence_text", json!("a recurring occasion"))
            .with("recurrence_rrule", json!(rule)),
        );
        let view = h.service.show_memory(BANK, &memory.to_string()).unwrap();
        assert_eq!(
            view.window.recurrence.as_deref(),
            Some("a recurring occasion")
        );
        assert_eq!(view.window.recurrence_rrule, None, "ambiguous rule: {rule}");
        assert!(inputs(&h.input(PROFILE_NAME)).is_empty(), "{rule}");
        assert!(h.agenda().dated.is_empty(), "{rule}");
        assert_eq!(h.agenda().routines, vec![memory], "text survives: {rule}");
    }
}

#[test]
fn an_explicit_recurrence_start_keeps_its_stated_interval_phase() {
    let h = Harness::new();
    // The explicit start fixes even years. Its two-year phase must survive.
    let rule = "FREQ=YEARLY;INTERVAL=2;BYMONTH=10;BYMONTHDAY=5";
    let memory = h.says(
        claim(
            "Alex and Jo celebrate this occasion every other 5 October, starting in 2024.",
            "recurring",
            "critical",
        )
        .with(
            "recurrence_text",
            json!("every other 5 October, starting in 2024"),
        )
        .with("recurrence_rrule", json!(rule))
        .with(
            "recurrence_start",
            json!({"at": "2024-10-05", "precision": "day"}),
        ),
    );
    let view = h.service.show_memory(BANK, &memory.to_string()).unwrap();
    assert_eq!(view.window.recurrence_rrule.as_deref(), Some(rule));
    assert_eq!(inputs(&h.input(PROFILE_NAME)), BTreeSet::from([memory]));
    assert_eq!(h.agenda().dated, vec![memory]);
    assert!(h.agenda().routines.is_empty());
    h.set(local("2027-10-01T00:00"));
    assert!(h.agenda().dated.is_empty());
    h.set(local("2028-10-01T00:00"));
    assert_eq!(h.agenda().dated, vec![memory]);
    h.set(local("2028-10-06T00:00"));
    assert!(h.agenda().dated.is_empty());
}

#[test]
fn the_profile_excludes_weekly_or_more_frequent_and_unclassified_routines() {
    let h = Harness::new();
    let fact = h.insert(fact("Alex likes green tea."));
    for (content, rule) in [
        (
            "Alex goes to the gym every Tuesday.",
            Some("FREQ=WEEKLY;BYDAY=TU"),
        ),
        (
            "Alex waters the fern every seven days.",
            Some("FREQ=DAILY;INTERVAL=7"),
        ),
        ("Alex walks the dog daily.", Some("FREQ=DAILY")),
        ("Alex checks the clock hourly.", Some("FREQ=HOURLY")),
        ("Alex calls Jo most weekends.", None),
        (
            "Alex has a recurring reminder with an unreadable frequency.",
            Some("FREQ=UNKNOWN"),
        ),
    ] {
        h.insert(recurring(content, rule, "2026-06-12T00:00"));
    }

    assert_eq!(inputs(&h.input(PROFILE_NAME)), BTreeSet::from([fact]));
}

fn custom_profile(kinds: Vec<Kind>) -> Harness {
    let h = Harness::new();
    h.service
        .edit_model(
            BANK,
            PROFILE_NAME,
            &ModelEdit {
                question: Some("What are Alex's recurring activities?".into()),
                kinds: Some(kinds),
                ..ModelEdit::default()
            },
        )
        .unwrap();
    h
}

fn custom_profile_selects_weekly_memories(kinds: Vec<Kind>, includes_facts: bool) {
    let h = custom_profile(kinds);
    let weekly = h.insert(recurring(
        "Alex goes to the gym every Tuesday.",
        Some("FREQ=WEEKLY;BYDAY=TU"),
        "2026-06-12T00:00",
    ));
    let tea = h.insert(fact("Alex likes green tea."));
    let expected = if includes_facts {
        BTreeSet::from([weekly, tea])
    } else {
        BTreeSet::from([weekly])
    };
    assert_eq!(inputs(&h.input(PROFILE_NAME)), expected);
}

#[test]
fn a_custom_recurring_profile_selects_weekly_memories() {
    custom_profile_selects_weekly_memories(vec![Kind::Recurring], false);
}

#[test]
fn a_custom_all_kinds_profile_selects_weekly_memories() {
    custom_profile_selects_weekly_memories(vec![], true);
}

fn custom_profile_refreshes_after_a_weekly_memory_is_extracted(kinds: Vec<Kind>) {
    let h = custom_profile(kinds);
    // Complete the owner-edit refresh before testing a memory-write trigger.
    h.advance(minutes(30));
    h.tick(&quiet_llm(1));
    // Leave the minimum refresh interval behind as well.
    h.advance(minutes(30));
    let llm = quiet_llm(1);
    assert!(h.tick(&llm).ran.is_empty());
    assert_eq!(refresh_calls(&llm), 0);

    let content = "Alex goes to the gym every Tuesday.";
    h.says(
        claim(content, "recurring", "notable")
            .with("recurrence_text", json!("every Tuesday"))
            .with("recurrence_rrule", json!("FREQ=WEEKLY;BYDAY=TU")),
    );
    h.advance(minutes(5) - SignedDuration::from_secs(1));
    assert!(h.tick(&llm).ran.is_empty());
    h.advance(SignedDuration::from_secs(1));
    let refreshed = h.tick(&llm);
    assert_eq!(
        refreshed.ran.len(),
        1,
        "the weekly memory triggers a refresh"
    );
    assert_eq!(refresh_calls(&llm), 1);
    assert!(llm.requests()[0].user.contains(content));
}

#[test]
fn a_custom_recurring_profile_refreshes_after_a_weekly_memory_is_extracted() {
    custom_profile_refreshes_after_a_weekly_memory_is_extracted(vec![Kind::Recurring]);
}

#[test]
fn a_custom_all_kinds_profile_refreshes_after_a_weekly_memory_is_extracted() {
    custom_profile_refreshes_after_a_weekly_memory_is_extracted(vec![]);
}

#[test]
fn a_cited_memory_stays_in_the_input_past_the_top_sixty() {
    // Keeping cited memories stops one that slips from 60th to 61st from
    // being removed and added back on alternate refreshes.
    let h = Harness::new();
    // No word in common with the question, like the facts below, so it
    // ranks below every one of them on strength alone.
    let weak = h.insert(fact("Tim's cat Miso sleeps."));
    h.refresh_adding(PROFILE_NAME, &[("Tim has a cat called Miso.", &[weak])]);
    for n in 1..=65 {
        h.insert(Memory {
            significance: "critical",
            ..fact(sentence(format!("Fact {n}.")))
        });
    }
    let input = h.input(PROFILE_NAME);
    assert_eq!(input.memories.len(), 61, "the top 60 and the cited one");
    assert!(inputs(&input).contains(&weak));
}

#[test]
fn the_relevance_scale_leaves_the_strength_term_alone_in_a_refresh() {
    // With room for one memory: a weak one sharing five words with the
    // profile question, against a strong one sharing none. At scale 1.0 the
    // shared words outweigh w_s_inject·strength; at 100.0 strength decides,
    // unless it were scaled too.
    let selected = |scale: f64| {
        let h = Harness::with_scale(
            scale,
            "[mental_models]\ninput_budget = 1\ninput_budget_with_cited = 1\n",
        );
        let weak = h.insert(Memory {
            significance: "trivial",
            ..fact("The user likes work and home life.")
        });
        let strong = h.insert(Memory {
            significance: "critical",
            observed_at: h.service.now(),
            ..fact("Tim cooks dinner.")
        });
        (inputs(&h.input(PROFILE_NAME)), weak, strong)
    };
    let (input, weak, _) = selected(1.0);
    assert_eq!(input, BTreeSet::from([weak]));
    let (input, _, strong) = selected(100.0);
    assert_eq!(input, BTreeSet::from([strong]));
}

#[test]
fn an_unchanged_fingerprint_skips_the_llm_and_force_doesnt() {
    let h = Harness::new();
    let tea = h.insert(fact(TEA));
    h.refresh_adding(PROFILE_NAME, &[("Tim likes green tea.", &[tea])]);

    let llm = quiet_llm(1);
    let skipped = h
        .service
        .refresh_model(BANK, PROFILE_NAME, &llm, false)
        .unwrap();
    assert_eq!(skipped, Outcome::Unchanged);
    assert_eq!(refresh_calls(&llm), 0);

    let forced = h
        .service
        .refresh_model(BANK, PROFILE_NAME, &llm, true)
        .unwrap();
    assert!(matches!(forced, Outcome::Applied(_)));
    assert_eq!(refresh_calls(&llm), 1);
}

#[test]
fn the_refresh_prompt_states_the_language_rule() {
    // The system prompt of a forced refresh of the profile on `h`.
    let system = |h: &Harness| {
        h.insert(fact(TEA));
        let llm = quiet_llm(1);
        h.service
            .refresh_model(BANK, PROFILE_NAME, &llm, true)
            .unwrap();
        llm.requests()[0].system.clone()
    };

    // Unset, the entries match the memories' language.
    let inferred = system(&Harness::new());
    assert!(
        inferred.contains("Write the entries in the language of the memories they cite."),
        "{inferred}"
    );

    // Set, every entry is in that language.
    let forced = system(&Harness::with_tuning("[llm]\nlanguage = \"English\"\n"));
    assert!(
        forced.contains(
            "Write every entry in English, translating if the memories are in another language."
        ),
        "{forced}"
    );
    assert!(
        !forced.contains("in the language of the memories"),
        "{forced}"
    );
}

#[test]
fn the_fingerprint_follows_the_selection_only() {
    let h = Harness::new();
    h.insert(fact(TEA));
    let before = h.input(PROFILE_NAME).fingerprint;
    h.insert(event("Tim is going to the cinema.", "2026-10-03T00:00"));
    h.insert(faded(fact("Tim once tried surfing in Raglan.")));
    assert_eq!(
        h.input(PROFILE_NAME).fingerprint,
        before,
        "a memory outside the filters changed it"
    );
    h.insert(fact(CAT));
    assert_ne!(h.input(PROFILE_NAME).fingerprint, before);
}

#[test]
fn a_faded_memory_leaves_the_model_at_the_next_sweep() {
    // When a cited memory fades below τ it leaves the input set,
    // and so leaves the model. A model can't keep a memory alive by itself.
    // Here the owner keeps a long-faded memory, the model cites it, and the
    // owner takes the keep back.
    let h = Harness::new();
    let cat = h.insert(fact(CAT));
    let tea = h.insert(faded(fact(TEA)));
    h.keep(tea);
    let applied = h.refresh_adding(
        PROFILE_NAME,
        &[
            ("Tim has a cat called Miso.", &[cat]),
            ("Tim likes green tea.", &[tea]),
        ],
    );
    let tea_entry = applied.added[1];
    h.service.unkeep(BANK, &[tea.to_string()]).unwrap();

    let input = h.input(PROFILE_NAME);
    assert!(
        !inputs(&input).contains(&tea),
        "the faded memory is still selected"
    );
    assert!(inputs(&input).contains(&cat));

    // Even a reply that leaves the entry alone loses it.
    h.set(at(SWEEP));
    let llm = quiet_llm(1);
    let ran = h.tick(&llm);
    assert_eq!(ran.ran.len(), 1);
    let Outcome::Applied(applied) = &ran.ran[0].outcome else {
        panic!("the fingerprint changed, so the LLM is called: {ran:?}");
    };
    assert_eq!(applied.dropped, vec![tea_entry]);
    assert_eq!(texts(&h.profile()), ["Tim has a cat called Miso."]);
}

#[test]
fn a_refresh_logs_its_retrieval_and_writes_no_access() {
    // reading or refreshing a model never counts as an
    // access. The schema's recall log has a `refresh` kind for its
    // retrieval: one row per refresh, with no session.
    let h = Harness::new();
    let tea = h.insert(fact(TEA));
    let accesses = h.accesses();
    h.refresh_adding(PROFILE_NAME, &[("Tim likes green tea.", &[tea])]);
    assert_eq!(h.accesses(), accesses);
    let rows: i64 = h.one("SELECT COUNT(*) FROM recalls WHERE kind = 'refresh'", []);
    assert_eq!(rows, 1);
    let session: Option<String> =
        h.one("SELECT session_id FROM recalls WHERE kind = 'refresh'", []);
    assert_eq!(session, None);
}

#[test]
fn entries_are_never_embedded_extracted_from_or_ingested() {
    // no feedback loops.
    let h = Harness::new();
    let tea = h.insert(fact(TEA));
    let counts = |h: &Harness| -> (i64, i64, i64, i64) {
        (
            h.one("SELECT COUNT(*) FROM memories", []),
            h.one("SELECT COUNT(*) FROM memory_vectors", []),
            h.one("SELECT COUNT(*) FROM sources", []),
            h.one("SELECT COUNT(*) FROM extraction_queue", []),
        )
    };
    let before = counts(&h);
    h.refresh_adding(PROFILE_NAME, &[("Tim likes green tea.", &[tea])]);
    h.block(Some("s1"));
    assert_eq!(counts(&h), before);
}

// Entries and edits

#[test]
fn untouched_entries_are_copied_byte_for_byte_and_edits_keep_their_id() {
    let h = Harness::new();
    let tea = h.insert(fact(TEA));
    let cat = h.insert(fact(CAT));
    let bees = h.insert(fact("Tim keeps bees."));
    let applied = h.refresh_adding(
        PROFILE_NAME,
        &[
            ("Tim likes green tea.", &[tea]),
            ("Tim has a cat  called Miso.\u{00a0}", &[cat]),
            ("Tim keeps bees.", &[bees]),
        ],
    );
    let [tea_entry, cat_entry, bees_entry] = applied.added[..] else {
        panic!("three adds");
    };
    let before = h.profile();

    let input = h.input(PROFILE_NAME);
    let llm = FakeLlm::scripted(
        MODEL,
        vec![reply(vec![
            edit(
                &entry_handle(&input, tea_entry),
                "Tim drinks green tea every morning.",
                &handles(&input, &[tea]),
            ),
            remove(&entry_handle(&input, bees_entry)),
        ])],
    );
    let Outcome::Applied(applied) = h
        .service
        .refresh_model(BANK, PROFILE_NAME, &llm, true)
        .unwrap()
    else {
        panic!("applied");
    };
    assert_eq!(applied.edited, vec![tea_entry]);
    assert_eq!(applied.removed, vec![bees_entry]);

    let after = h.profile();
    assert_eq!(after.entries.len(), 2);
    assert_eq!(after.entries[0].id, tea_entry);
    assert_eq!(after.entries[0].text, "Tim drinks green tea every morning.");
    let untouched = after.entries.iter().find(|e| e.id == cat_entry).unwrap();
    assert_eq!(untouched, &before.entries[1], "the untouched entry changed");
}

#[test]
fn invalid_operations_are_rejected_and_the_rest_apply() {
    // Code refuses an entry whose citations aren't in the refresh's input,
    // an entry that cites nothing, and an edit or remove of an entry that
    // doesn't exist.
    let h = Harness::new();
    let tea = h.insert(fact(TEA));
    let gone = h.insert(faded(fact("Tim once tried surfing in Raglan.")));
    let applied = h.refresh_adding(PROFILE_NAME, &[("Tim likes green tea.", &[tea])]);
    let input = h.input(PROFILE_NAME);
    assert!(!inputs(&input).contains(&gone));
    let llm = FakeLlm::scripted(
        MODEL,
        vec![reply(vec![
            add("Tim surfs.", &["m99".to_string()]),
            add("Tim surfs in Raglan.", &[gone.to_string()]),
            add(
                "Tim likes tea and surfing.",
                &[handle(&input, tea), "m99".into()],
            ),
            edit(
                &entry_handle(&input, applied.added[0]),
                "Tim likes tea.",
                &[],
            ),
            add("Tim is lovely.", &[]),
            edit("e7", "Tim likes tea.", &handles(&input, &[tea])),
            remove("e8"),
            add("Tim drinks green tea.", &handles(&input, &[tea])),
        ])],
    );
    let Outcome::Applied(applied) = h
        .service
        .refresh_model(BANK, PROFILE_NAME, &llm, true)
        .unwrap()
    else {
        panic!("applied");
    };
    let rejected: Vec<(usize, RejectReason)> = applied
        .rejected
        .iter()
        .map(|r| (r.index, r.reason))
        .collect();
    assert_eq!(
        rejected,
        [
            (0, RejectReason::CitesOutsideInput),
            (1, RejectReason::CitesOutsideInput),
            (2, RejectReason::CitesOutsideInput),
            (3, RejectReason::NoCitations),
            (4, RejectReason::NoCitations),
            (5, RejectReason::UnknownEntry),
            (6, RejectReason::UnknownEntry),
        ]
    );
    assert_eq!(
        texts(&h.profile()),
        ["Tim likes green tea.", "Tim drinks green tea."]
    );
}

#[test]
fn a_malformed_reply_leaves_the_entries_untouched_and_records_an_error() {
    let h = Harness::new();
    let tea = h.insert(fact(TEA));
    h.refresh_adding(PROFILE_NAME, &[("Tim likes green tea.", &[tea])]);
    let before = h.profile();
    h.insert(fact(CAT));

    h.advance(minutes(1));
    for nonsense in [
        json!({"edits": []}),
        json!({"operations": [{"op": "rewrite", "text": "Everything."}]}),
        json!({"operations": "add everything"}),
    ] {
        let llm = FakeLlm::scripted(MODEL, vec![nonsense.clone()]);
        let outcome = h
            .service
            .refresh_model(BANK, PROFILE_NAME, &llm, false)
            .unwrap();
        assert_eq!(
            outcome,
            Outcome::Failed(FailureKind::Malformed),
            "{nonsense}"
        );
        let after = h.profile();
        assert_eq!(after.entries, before.entries);
        assert_eq!(after.last_refreshed_at, before.last_refreshed_at);
        assert_eq!(after.last_error, Some(FailureKind::Malformed));
        assert_eq!(after.last_error_at, Some(h.now()));
    }
    // A failed refresh doesn't record the new fingerprint, so the next one
    // isn't skipped.
    let llm = quiet_llm(1);
    assert!(matches!(
        h.service
            .refresh_model(BANK, PROFILE_NAME, &llm, false)
            .unwrap(),
        Outcome::Applied(_)
    ));
}

// The budget

#[test]
fn creating_a_model_past_the_budget_is_refused() {
    // The profile takes 500 of the 800.
    let h = Harness::new();
    let refused = h.service.create_model(
        BANK,
        &ModelSpec {
            max_tokens: 301,
            ..plans_model()
        },
    );
    assert!(matches!(
        refused,
        Err(ModelError::OverBudget {
            requested: 801,
            budget: 800
        })
    ));
    h.service
        .create_model(
            BANK,
            &ModelSpec {
                max_tokens: 300,
                ..plans_model()
            },
        )
        .unwrap();
    assert_eq!(h.service.list_models(BANK).unwrap().len(), 2);
}

#[test]
fn resizing_or_enabling_past_the_budget_is_refused() {
    let h = Harness::new();
    h.service
        .create_model(
            BANK,
            &ModelSpec {
                max_tokens: 300,
                enabled: false,
                ..plans_model()
            },
        )
        .unwrap();
    h.service
        .edit_model(
            BANK,
            PROFILE_NAME,
            &ModelEdit {
                max_tokens: Some(800),
                ..ModelEdit::default()
            },
        )
        .unwrap();
    // A disabled model isn't rendered, so it doesn't count...
    let enabling = h.service.edit_model(
        BANK,
        "Plans",
        &ModelEdit {
            enabled: Some(true),
            ..ModelEdit::default()
        },
    );
    // ...until it's enabled.
    assert!(matches!(enabling, Err(ModelError::OverBudget { .. })));
    let resizing = h.service.edit_model(
        BANK,
        PROFILE_NAME,
        &ModelEdit {
            max_tokens: Some(801),
            ..ModelEdit::default()
        },
    );
    assert!(matches!(resizing, Err(ModelError::OverBudget { .. })));
    assert_eq!(h.profile().max_tokens, 800);
}

#[test]
fn entries_past_max_tokens_are_trimmed_lowest_ranked_first() {
    // Code trims the lowest-ranked entries, ranked by the best score among
    // each entry's cited memories. The three memories here share the same
    // words with the question, so strength decides.
    let h = Harness::new();
    h.service
        .create_model(
            BANK,
            &ModelSpec {
                name: "Mornings".into(),
                question: "What does Tim drink in the morning?".into(),
                kinds: vec![Kind::Fact],
                entity: None,
                min_volatility: None,
                max_tokens: 30,
                enabled: true,
            },
        )
        .unwrap();
    let tea = h.insert(Memory {
        significance: "critical",
        ..fact("Tim drinks tea in the morning.")
    });
    let coffee = h.insert(Memory {
        significance: "minor",
        ..fact("Tim drinks coffee in the morning.")
    });
    let juice = h.insert(Memory {
        significance: "major",
        ..fact("Tim drinks juice in the morning.")
    });
    // 60 characters each: 15 tokens, so two fit in 30.
    let applied = h.refresh_adding(
        "Mornings",
        &[
            (
                "Tim starts every single day with a large pot of green tea..",
                &[tea],
            ),
            (
                "Tim sometimes has a strong black coffee in the morning too.",
                &[coffee],
            ),
            (
                "Tim drinks a glass of fresh orange juice with his breakfast.",
                &[juice],
            ),
        ],
    );
    assert_eq!(applied.trimmed, vec![applied.added[1]]);
    let model = h.model("Mornings");
    assert_eq!(model.entries.len(), 2);
    let tokens: usize = model.entries.iter().map(|e| estimate_tokens(&e.text)).sum();
    assert!(tokens <= 30, "{tokens} tokens");
}

// Memories win

#[test]
fn an_entry_citing_a_retracted_memory_is_dropped_from_the_block() {
    let h = Harness::new();
    let maya = h.insert(fact(MAYA));
    let cat = h.insert(fact(CAT));
    let applied = h.refresh_adding(
        PROFILE_NAME,
        &[
            ("Tim's daughter is Maya.", &[maya]),
            ("Tim has a cat called Miso.", &[cat]),
        ],
    );
    assert!(h.block(None).text.contains("Tim's daughter is Maya."));

    h.advance(minutes(30));
    let corrected = h.now();
    h.says_changing(
        notable(MIA).with("changes_something", json!(true)),
        maya,
        "retracts",
    );
    let block = h.block(None);
    assert!(!block.text.contains("Tim's daughter is Maya."));
    assert!(block.text.contains("Tim has a cat called Miso."));
    assert!(!block.cited.contains(&maya));
    // Dropped at render time, and the model is refreshed: the retraction of
    // a cited memory is a triggering write.
    assert_eq!(h.profile().entries[0].id, applied.added[0]);
    assert_eq!(h.tick(&quiet_llm(1)).next_due, Some(corrected + minutes(5)));
}

#[test]
fn an_entry_is_dropped_when_any_one_of_its_memories_ends() {
    // It isn't enough for one citation to survive: the wording rests on all
    // of them.
    let h = Harness::new();
    let berlin = h.insert(fact(BERLIN));
    let cat = h.insert(fact(CAT));
    h.refresh_adding(
        PROFILE_NAME,
        &[
            ("Tim lives in Berlin with his cat Miso.", &[berlin, cat]),
            ("Tim has a cat called Miso.", &[cat]),
        ],
    );
    h.says_changing(
        claim(MOVED, "event", "notable")
            .with("changes_something", json!(true))
            .with(
                "valid_from",
                json!({"at": "2026-09-12", "precision": "day"}),
            ),
        berlin,
        "ends",
    );
    let block = h.block(None);
    assert!(
        !block
            .text
            .contains("Tim lives in Berlin with his cat Miso.")
    );
    assert!(block.text.contains("Tim has a cat called Miso."));
    assert_eq!(block.cited, vec![cat]);
}

#[test]
fn a_refinement_moves_the_citation_to_the_head_of_the_chain() {
    let h = Harness::new();
    h.service.create_model(BANK, &plans_model()).unwrap();
    let japan = h.insert(Memory {
        kind: "event",
        valid_from: Some((local("2027-01-01T00:00"), "year")),
        ..fact(JAPAN)
    });
    let applied = h.refresh_adding("Plans", &[("Tim is going to Japan next year.", &[japan])]);
    let before = h.input("Plans").fingerprint;

    h.advance(minutes(30));
    let refined = h.now();
    let tokyo = h.says_changing(
        claim(TOKYO, "event", "notable")
            .with("valid_from", json!({"at": "2027-04", "precision": "month"})),
        japan,
        "refines",
    );
    let plans = h.model("Plans");
    assert_eq!(plans.entries[0].id, applied.added[0]);
    assert_eq!(plans.entries[0].cites, vec![tokyo]);
    // The entry still renders, and the next refresh rewords it.
    assert!(
        h.block(None)
            .text
            .contains("Tim is going to Japan next year.")
    );
    assert_ne!(h.input("Plans").fingerprint, before);
    assert_eq!(h.tick(&quiet_llm(1)).next_due, Some(refined + minutes(5)));
}

#[test]
fn an_entry_citing_a_low_confidence_state_shows_its_age() {
    // As in injection, show the age: "observed 30 days ago, Tue 1 Sep".
    let h = Harness::new();
    let job = h.insert(Memory {
        kind: "state",
        volatility: Some("weeks"),
        ..fact("Tim is job hunting.")
    });
    h.refresh_adding(PROFILE_NAME, &[("Tim is looking for a new job.", &[job])]);
    let text = h.block(None).text;
    let line = text
        .lines()
        .find(|line| line.contains("Tim is looking for a new job."))
        .expect("the entry is rendered");
    assert!(line.contains("observed"), "{line}");
    assert!(line.contains("Tue 1 Sep"), "{line}");
}

// The block

#[test]
fn the_block_opens_with_memory_guidance_before_the_agenda_and_models() {
    let h = Harness::new();
    let tea = h.insert(fact(TEA));
    let dentist = h.insert(event(
        "Tim's dentist appointment is on 5 October.",
        "2026-10-05T00:00",
    ));
    h.refresh_adding(PROFILE_NAME, &[("Tim likes green tea.", &[tea])]);
    h.service
        .create_model(
            BANK,
            &ModelSpec {
                enabled: false,
                ..plans_model()
            },
        )
        .unwrap();

    let block = h.block(None);
    assert_memory_guidance(&block);
    assert_eq!(block.built_at, h.now());
    assert_eq!(block.agenda, vec![dentist]);
    assert_eq!(block.cited, vec![tea]);
    for needle in [
        "Tim's dentist appointment is on 5 October.",
        PROFILE_NAME,
        "Tim likes green tea.",
        "memory_recall",
        "upcoming",
        "Thu 1 Oct",
    ] {
        assert!(block.text.contains(needle), "the block lacks {needle:?}");
    }
    assert!(
        !block.text.contains("Plans"),
        "a disabled model was rendered"
    );
}

/// Check the usage contract without requiring the entire draft verbatim.
fn assert_memory_guidance(block: &Block) {
    assert!(
        block.text.starts_with("## Long-term memory (Asphodel)\n"),
        "the block must open with its memory heading:\n{}",
        block.text
    );
    let guidance = block.text.split("Agenda for ").next().unwrap();
    for needle in [
        "saved automatically",
        "never need to save",
        "<memory-context>",
        "Thu 1 Oct 20:00",
        "disagree",
        "Before saying",
        "don't know",
        "don't remember",
        "memory_recall",
        "upcoming",
        "session_search",
        "exact wording",
    ] {
        assert!(guidance.contains(needle), "the guidance lacks {needle:?}");
    }
    assert!(!block.text.contains("memories win:"));
}

#[test]
fn an_empty_bank_still_opens_with_memory_guidance() {
    let h = Harness::new();
    let block = h.block(None);
    assert!(block.agenda.is_empty());
    assert!(block.cited.is_empty());
    assert_memory_guidance(&block);
}

#[test]
fn budget_folding_keeps_memory_guidance_and_counts_it_in_the_budget() {
    let h = Harness::new();
    let tasks: Vec<Uuid> = (0..5)
        .map(|n| {
            h.insert(task(sentence(format!(
                "Tim needs to complete job {n}. {}",
                "There are many details to handle before this job is complete. ".repeat(20)
            ))))
        })
        .collect();
    let block = h.block(None);
    assert!(block.agenda.len() < tasks.len(), "the agenda did not fold");
    assert!(estimate_tokens(&block.text) <= h.tuning.mental_models.budget as usize);
    assert_memory_guidance(&block);
}

#[test]
fn budgets_below_the_memory_guidance_minimum_are_rejected() {
    // The fixture guidance alone needs 117 estimated tokens. Neither the
    // reported 100-token budget nor the token immediately below it can fit.
    let mut accepted = Vec::new();
    for budget in [100, 116] {
        let result = Tuning::from_toml(&format!(
            "[mental_models]\nbudget = {budget}\nprofile_max_tokens = {budget}\n"
        ));
        match result {
            Ok(_) => accepted.push(budget),
            Err(error) => assert!(error.to_string().contains("mental_models.budget")),
        }
    }
    assert!(
        accepted.is_empty(),
        "accepted budgets below the guidance minimum: {accepted:?}"
    );
}

#[test]
fn accepted_boundary_budgets_fit_memory_guidance_and_the_dated_fold_summary() {
    // The safe minimum is 134 tokens on 64-bit targets (131 on 32-bit):
    // guidance with a two-digit day, an agenda heading, and a fold summary
    // with the largest representable count, including their separators.
    let minimum = if usize::BITS == 64 { 134 } else { 131 };
    let below = minimum - 1;
    let error = Tuning::from_toml(&format!(
        "[mental_models]\nbudget = {below}\nprofile_max_tokens = {below}\n"
    ))
    .expect_err("the token immediately below the safe minimum must be rejected");
    assert!(error.to_string().contains("mental_models.budget"));

    for budget in [minimum, minimum + 1] {
        let extra = format!("[mental_models]\nbudget = {budget}\nprofile_max_tokens = {budget}\n");
        Tuning::from_toml(&extra).expect("the safe minimum and higher budgets must be accepted");
        let h = Harness::with_tuning(&extra);
        let empty = h.block(None);
        assert_memory_guidance(&empty);
        assert!(estimate_tokens(&empty.text) <= budget);

        let h = Harness::with_tuning(&extra);
        h.insert(event(
            sentence(format!(
                "Tim has a planning appointment. {}",
                "There are many details to discuss. ".repeat(30)
            )),
            "2026-10-01T00:00",
        ));
        let folded = h.block(None);
        assert!(folded.agenda.is_empty(), "the dated item did not fold");
        assert_memory_guidance(&folded);
        assert!(folded.text.contains("- and 1 more dated item"));
        let tokens = estimate_tokens(&folded.text);
        assert!(
            tokens <= budget,
            "folded block uses {tokens} tokens with budget {budget}"
        );
    }
}

#[test]
fn building_the_block_or_the_agenda_never_writes_an_access() {
    let h = Harness::new();
    let tea = h.insert(fact(TEA));
    h.insert(event(
        "Tim's dentist appointment is on 5 October.",
        "2026-10-05T00:00",
    ));
    h.refresh_adding(PROFILE_NAME, &[("Tim likes green tea.", &[tea])]);
    let accesses = h.accesses();
    h.block(Some("s1"));
    h.block(Some("s2"));
    h.agenda();
    assert_eq!(h.accesses(), accesses, "being injected never counts");
}

#[test]
fn the_block_is_cached_until_its_content_changes() {
    let h = Harness::new();
    let tea = h.insert(fact(TEA));
    let first = h.block(Some("s1"));
    h.advance(minutes(10));
    let again = h.block(Some("s2"));
    assert_eq!(again.id, first.id);
    assert_eq!(again.built_at, first.built_at);

    // A memory that isn't on the agenda and doesn't change a model leaves
    // the block alone.
    h.says(claim("Tim likes walking.", "fact", "minor"));
    assert_eq!(h.block(None).id, first.id);

    // A completed refresh clears it.
    h.refresh_adding(PROFILE_NAME, &[("Tim likes green tea.", &[tea])]);
    let refreshed = h.block(None);
    assert_ne!(refreshed.id, first.id);
    assert_eq!(refreshed.built_at, h.now());
    assert!(refreshed.text.contains("Tim likes green tea."));
}

#[test]
fn a_new_agenda_memory_clears_the_block() {
    let h = Harness::new();
    let first = h.block(None);
    h.says(
        claim("Tim has a haircut on 3 October 2026.", "event", "minor").with(
            "valid_from",
            json!({"at": "2026-10-03", "precision": "day"}),
        ),
    );
    let block = h.block(None);
    assert_ne!(block.id, first.id);
    assert!(block.text.contains("Tim has a haircut on 3 October 2026."));
}

#[test]
fn the_clock_alone_changes_the_block_only_at_local_midnight() {
    let h = Harness::new();
    h.insert(event(
        "Tim's dentist appointment is on 9 October.",
        "2026-10-09T00:00",
    ));
    let first = h.block(None);
    assert!(first.agenda.is_empty(), "9 October is eight days out");

    // 23:59 local is still Thursday.
    h.set(local("2026-10-01T23:59"));
    assert_eq!(h.block(None).id, first.id);

    h.set(local("2026-10-02T00:00"));
    let friday = h.block(None);
    assert_ne!(friday.id, first.id);
    assert_eq!(friday.agenda.len(), 1);
}

// In context

#[test]
fn a_sessions_block_puts_its_agenda_and_cited_memories_in_context() {
    let h = Harness::new();
    let cat = h.insert(fact(CAT));
    let dentist = h.insert(event(
        "Tim's dentist appointment is on 5 October.",
        "2026-10-05T00:00",
    ));
    h.refresh_adding(PROFILE_NAME, &[("Tim has a cat called Miso.", &[cat])]);
    h.block(Some("s1"));

    let in_context: BTreeSet<Uuid> = h.in_context("s1").into_iter().collect();
    assert_eq!(in_context, BTreeSet::from([cat, dentist]));
    assert!(h.in_context("s2").is_empty());

    // Relevance injection skips them.
    let prefetch = h
        .service
        .prefetch(
            BANK,
            &PrefetchRequest {
                session_id: "s1".into(),
                query: "is the cat called Miso".into(),
                previous_query: None,
                previous_reply: None,
                block_id: None,
            },
        )
        .unwrap();
    assert!(!prefetch.injected.contains(&cat));
}

#[test]
fn clearing_a_session_drops_its_mapping_and_the_next_fetch_writes_one() {
    let h = Harness::new();
    let cat = h.insert(fact(CAT));
    h.refresh_adding(PROFILE_NAME, &[("Tim has a cat called Miso.", &[cat])]);
    h.block(Some("s1"));
    h.service.clear_session(BANK, "s1").unwrap();
    assert!(h.in_context("s1").is_empty());
    let mappings: i64 = h.one("SELECT COUNT(*) FROM session_blocks", []);
    assert_eq!(mappings, 0);

    h.block(Some("s1"));
    assert_eq!(h.in_context("s1"), vec![cat]);
}

#[test]
fn a_mapping_expires_after_thirty_days_without_a_turn() {
    let h = Harness::new();
    let cat = h.insert(fact(CAT));
    h.refresh_adding(PROFILE_NAME, &[("Tim has a cat called Miso.", &[cat])]);
    h.block(Some("s1"));

    // Past the in-memory idle timeout, the mapping still holds...
    h.advance(SignedDuration::from_hours(29 * 24));
    let h = h.restart();
    assert_eq!(h.in_context("s1"), vec![cat]);

    // ...until sessions.mapping_expiry_days without a turn.
    h.advance(SignedDuration::from_hours(2 * 24));
    let h = h.restart();
    assert!(h.in_context("s1").is_empty());
}

// The agenda

#[test]
fn dated_lines_hold_events_and_tasks_within_seven_local_days() {
    let h = Harness::new();
    let today = h.insert(event("Tim has pottery tonight.", "2026-10-01T00:00"));
    let thursday = h.insert(event(
        "Tim flies to Sydney on 8 October.",
        "2026-10-08T00:00",
    ));
    let friday = h.insert(event("Tim has a concert on 9 October.", "2026-10-09T00:00"));
    let due = h.insert(task_due(
        "Tim needs to pay the rates by 6 October.",
        "2026-10-06T00:00",
    ));
    let later = h.insert(task_due(
        "Tim needs to renew his passport by 20 October.",
        "2026-10-20T00:00",
    ));
    let past = h.insert(event(
        "Tim went to the beach on 28 September.",
        "2026-09-28T00:00",
    ));

    let agenda = h.agenda();
    assert_eq!(agenda.dated, vec![today, due, thursday]);
    for memory in [friday, later, past] {
        assert!(!agenda.dated.contains(&memory));
    }
}

#[test]
fn overdue_tasks_are_listed_for_overdue_days() {
    let h = Harness::with_tuning("[agenda]\noverdue_days = 10\n");
    let recent = h.insert(task_due("Tim needs to call the bank.", "2026-09-21T00:00"));
    let old = h.insert(task_due("Tim needs to fix the gate.", "2026-09-20T00:00"));
    let done = h.insert(task_due("Tim needs to book the vet.", "2026-09-25T00:00"));
    let booked = h.insert(Memory {
        kind: "event",
        ..fact("Tim booked the vet.")
    });
    h.mark_ended(done, booked, local("2026-09-26T00:00"));

    let agenda = h.agenda();
    assert_eq!(agenda.dated, vec![recent]);
    assert!(!agenda.dated.contains(&old));
    assert!(!agenda.undated_tasks.contains(&done));
}

#[test]
fn undated_tasks_leave_after_the_last_mention_cap_inclusive() {
    let h = Harness::with_tuning("[agenda]\nundated_days = 10\n");
    let boundary = h.insert(Memory {
        significance: "major",
        observed_at: local("2026-09-21T12:00"),
        ..task("Tim needs to sort the tools.")
    });
    let expired = h.insert(Memory {
        significance: "major",
        observed_at: local("2026-09-20T12:00"),
        ..task("Tim needs to paint the shed.")
    });
    assert_eq!(h.agenda().undated_tasks, vec![boundary]);
    assert!(!h.agenda().undated_tasks.contains(&expired));
    h.set(local("2026-10-02T00:00"));
    assert!(h.agenda().undated_tasks.is_empty());
}

#[test]
fn undated_tasks_renew_on_mentions_but_not_on_use() {
    let h = Harness::with_tuning("[agenda]\nundated_days = 10\n");
    let mentioned = h.insert(Memory {
        significance: "major",
        observed_at: local("2026-07-01T12:00"),
        ..task("Tim needs to repair the chair.")
    });
    let used = h.insert(Memory {
        significance: "major",
        observed_at: local("2026-07-01T12:00"),
        ..task("Tim needs to tidy the attic.")
    });
    for (memory, kind) in [(mentioned, "mentioned_again"), (used, "used")] {
        h.execute(
            "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
             VALUES (?1, ?2, ?3, ?4, 1)",
            (
                h.bank_id(),
                h.rowid(memory),
                kind,
                micros(local("2026-09-25T12:00")),
            ),
        );
    }
    assert_eq!(h.agenda().undated_tasks, vec![mentioned]);
    h.set(local("2026-10-05T23:59"));
    assert_eq!(h.agenda().undated_tasks, vec![mentioned]);
    h.set(local("2026-10-06T00:00"));
    assert!(h.agenda().undated_tasks.is_empty());
}

#[test]
fn undated_task_mentions_and_confirmations_are_inherited_by_the_head() {
    // Exercise both access kinds on a head and through two superseded ancestors.
    for kind in ["mentioned_again", "confirmed"] {
        for inherited in [false, true] {
            let h = Harness::new();
            let head = h.insert(Memory {
                significance: "major",
                observed_at: local("2026-07-01T12:00"),
                ..task("Tim needs to mend the garden fence.")
            });
            let mut accessed = head;
            if inherited {
                for content in [
                    "Tim needs to fix the fence.",
                    "Tim needs to fix the boundary fence.",
                ] {
                    let predecessor = h.insert(Memory {
                        significance: "major",
                        observed_at: local("2026-07-01T12:00"),
                        ..task(content)
                    });
                    h.execute(
                        "UPDATE memories SET superseded_by = ?2 WHERE id = ?1",
                        (h.rowid(predecessor), h.rowid(accessed)),
                    );
                    accessed = predecessor;
                }
            }
            h.execute(
                "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
                 VALUES (?1, ?2, ?3, ?4, 1)",
                (
                    h.bank_id(),
                    h.rowid(accessed),
                    kind,
                    micros(local("2026-09-25T12:00")),
                ),
            );
            assert_eq!(
                h.agenda().undated_tasks,
                vec![head],
                "{kind}, inherited={inherited}"
            );
            h.set(local("2026-10-25T23:59"));
            assert_eq!(
                h.agenda().undated_tasks,
                vec![head],
                "{kind}, inherited={inherited}"
            );
            h.set(local("2026-10-26T00:00"));
            assert!(
                h.agenda().undated_tasks.is_empty(),
                "{kind}, inherited={inherited}"
            );
        }
    }
}

#[test]
fn undated_task_cap_uses_bank_local_dates_not_utc_or_elapsed_hours() {
    let h = Harness::new();
    // 1 September UTC is 2 September in Auckland. At the boundary below,
    // more than 30 * 24 hours have elapsed, but it is still local day 30.
    let task = h.insert(Memory {
        significance: "major",
        observed_at: at("2026-09-01T23:30:00Z"),
        ..task("Tim needs to catalogue the spare parts.")
    });
    h.set(at("2026-10-02T10:59:00Z")); // 2 October, 23:59 NZDT
    assert_eq!(h.agenda().undated_tasks, vec![task]);
    h.set(at("2026-10-02T11:00:00Z")); // 3 October, 00:00 NZDT; UTC date unchanged
    assert!(h.agenda().undated_tasks.is_empty());
}

#[test]
fn undated_task_cap_does_not_shorten_overdue_obligations() {
    let h = Harness::with_tuning("[agenda]\nundated_days = 10\n");
    let obligation = h.insert(Memory {
        observed_at: local("2026-07-01T12:00"),
        ..task_due("Tim needs to renew his library card.", "2026-09-05T00:00")
    });
    assert_eq!(h.agenda().dated, vec![obligation]);
    assert!(h.agenda().undated_tasks.is_empty());
    h.set(local("2026-10-05T23:59"));
    assert_eq!(h.agenda().dated, vec![obligation]);
    h.set(local("2026-10-06T00:00"));
    assert!(h.agenda().dated.is_empty());
}

#[test]
fn dated_items_are_listed_even_when_faded() {
    // A minor appointment mentioned three months ahead mustn't fade out on
    // the day it matters.
    let h = Harness::new();
    let dentist = h.insert(faded(event(
        "Tim's dentist appointment is on 2 October.",
        "2026-10-02T00:00",
    )));
    assert_eq!(h.agenda().dated, vec![dentist]);
}

#[test]
fn retracted_and_forgotten_memories_are_never_on_the_agenda() {
    let h = Harness::new();
    h.insert(Memory {
        retracted: true,
        ..event(
            "Tim's dentist appointment is on 2 October.",
            "2026-10-02T00:00",
        )
    });
    h.insert(Memory {
        hidden: true,
        ..task("Tim needs to call Ana.")
    });
    h.insert(Memory {
        hidden: true,
        ..recurring(
            "Tim swims on Tuesdays.",
            Some("FREQ=WEEKLY;BYDAY=TU"),
            "2026-01-06T00:00",
        )
    });
    let agenda = h.agenda();
    assert!(agenda.dated.is_empty());
    assert!(agenda.routines.is_empty());
    assert!(agenda.undated_tasks.is_empty());
}

#[test]
fn over_the_cap_faded_lines_fold_first_then_the_least_significant() {
    let h = Harness::with_tuning("[agenda]\ndated_lines = 3\n");
    let a = h.insert(event("Tim has event A.", "2026-10-02T00:00"));
    let faded_b = h.insert(faded(event("Tim has event B.", "2026-10-03T00:00")));
    let c = h.insert(Memory {
        significance: "minor",
        ..event("Tim has event C.", "2026-10-04T00:00")
    });
    let d = h.insert(Memory {
        significance: "major",
        ..event("Tim has event D.", "2026-10-05T00:00")
    });
    let e = h.insert(event("Tim has event E.", "2026-10-06T00:00"));

    let agenda = h.agenda();
    assert_eq!(agenda.dated, vec![a, d, e], "in date order");
    assert_eq!(agenda.folded, 2);
    assert!(!agenda.dated.contains(&faded_b));
    assert!(!agenda.dated.contains(&c));
}

#[test]
fn routines_are_gated_on_tau_ranked_by_strength_and_capped() {
    let h = Harness::new();
    let weekly = |content, significance| Memory {
        significance,
        ..recurring(content, Some("FREQ=WEEKLY;BYDAY=TU"), "2026-01-06T00:00")
    };
    let swim = h.insert(weekly("Tim swims on Tuesdays.", "critical"));
    let yoga = h.insert(weekly("Tim does yoga on Tuesdays.", "major"));
    let call = h.insert(Memory {
        significance: "notable",
        ..recurring("Tim calls his mum most weekends.", None, "")
    });
    let daily = h.insert(Memory {
        significance: "minor",
        ..recurring(
            "Tim walks the dog every day.",
            Some("FREQ=DAILY"),
            "2026-01-01T00:00",
        )
    });
    let bins = h.insert(weekly("Tim puts the bins out on Tuesdays.", "trivial"));
    let chess = h.insert(faded(recurring("Tim plays chess on Fridays.", None, "")));

    let agenda = h.agenda();
    assert_eq!(agenda.routines, vec![swim, yoga, call, daily]);
    assert!(!agenda.routines.contains(&bins), "past the cap of 4");
    assert!(!agenda.routines.contains(&chess), "below τ");
    assert!(agenda.dated.is_empty());
}

#[test]
fn a_long_period_routine_joins_the_dated_lines_when_it_next_occurs_within_a_week() {
    let h = Harness::new();
    let soon = h.insert(recurring(
        "Tim pays the rent on the 5th of each month.",
        Some("FREQ=MONTHLY;BYMONTHDAY=5"),
        "2026-01-05T00:00",
    ));
    let later = h.insert(recurring(
        "Tim's book club meets on the 20th of each month.",
        Some("FREQ=MONTHLY;BYMONTHDAY=20"),
        "2026-01-20T00:00",
    ));
    let agenda = h.agenda();
    assert_eq!(agenda.dated, vec![soon]);
    assert!(agenda.routines.is_empty(), "a month is longer than a week");
    assert!(!agenda.dated.contains(&later));
}

#[test]
fn an_undated_task_can_fade_out_before_the_last_mention_cap() {
    let h = Harness::with_tuning("[clock]\nquiet_rate = 1.0\n");
    let task = h.insert(Memory {
        significance: "trivial",
        observed_at: at(START),
        ..task("Tim wants to try a new tea.")
    });
    assert_eq!(h.agenda().undated_tasks, vec![task]);

    // Twenty bank days put this single trivial mention below τ, while
    // it is still inside the default 30-day last-mention window. There
    // are no other tasks to exclude it through ranking or the list cap.
    h.advance(SignedDuration::from_hours(20 * 24));
    assert!(h.agenda().undated_tasks.is_empty());
}

#[test]
fn undated_open_tasks_are_gated_on_tau_and_capped() {
    let h = Harness::with_tuning("[agenda]\nundated_tasks = 2\n");
    let passport = h.insert(Memory {
        significance: "major",
        ..task("Tim needs to renew his passport.")
    });
    let gutters = h.insert(task("Tim needs to clean the gutters."));
    let shelf = h.insert(Memory {
        significance: "minor",
        ..task("Tim needs to put up a shelf.")
    });
    let rust = h.insert(faded(task("Tim wants to learn Rust someday.")));
    let tax = h.insert(task("Tim needs to file the tax return."));
    let filed = h.insert(Memory {
        kind: "event",
        ..fact("Tim filed the tax return.")
    });
    h.mark_ended(tax, filed, local("2026-09-30T00:00"));

    let agenda = h.agenda();
    assert_eq!(agenda.undated_tasks, vec![passport, gutters]);
    for memory in [shelf, rust, tax] {
        assert!(!agenda.undated_tasks.contains(&memory));
    }
}

// Refresh retry regressions

/// An embedder that answers like `FakeEmbedder` until a test makes it
/// fail, counting every call.
#[derive(Default)]
struct FlakyEmbedder {
    failing: AtomicBool,
    calls: AtomicUsize,
}

impl Embedder for FlakyEmbedder {
    fn model_id(&self) -> &str {
        FakeEmbedder::MODEL_ID
    }

    fn dimensions(&self) -> usize {
        FakeEmbedder.dimensions()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.failing.load(Ordering::SeqCst) {
            return Err(EmbedError::Inference {
                model: FakeEmbedder::MODEL_ID.into(),
                reason: "the test made it fail".into(),
            });
        }
        FakeEmbedder.embed(texts)
    }
}

#[test]
fn a_refresh_whose_retrieval_fails_waits_thirty_minutes_like_any_failure() {
    // A failed refresh waits 30 minutes before retrying. Logging a
    // retrieval failure without settling the request would leave it due
    // on every timer pass, a minute apart in `serve`.
    let embedder = Arc::new(FlakyEmbedder::default());
    let h = Harness::with_models(Models {
        embedder: embedder.clone(),
        reranker: Arc::new(FakeReranker),
    });
    h.says(notable(TEA));
    h.advance(minutes(5));
    let failed_at = h.now();
    embedder.failing.store(true, Ordering::SeqCst);
    let llm = quiet_llm(1);
    let ran = h.tick(&llm);
    assert_eq!(refresh_calls(&llm), 0);
    let profile = h.profile();
    assert!(
        profile.last_error.is_some(),
        "the failed retrieval wasn't recorded"
    );
    assert_eq!(profile.last_error_at, Some(failed_at));
    assert_eq!(profile.last_refreshed_at, None);
    assert_eq!(ran.next_due, Some(failed_at + minutes(30)));

    // The embedder recovers, but nothing is tried before the interval.
    embedder.failing.store(false, Ordering::SeqCst);
    let calls = embedder.calls.load(Ordering::SeqCst);
    h.advance(minutes(1));
    assert!(h.tick(&llm).ran.is_empty());
    assert_eq!(
        embedder.calls.load(Ordering::SeqCst),
        calls,
        "retried early"
    );

    h.set(failed_at + minutes(30));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    assert_eq!(refresh_calls(&llm), 1);
    let profile = h.profile();
    assert_eq!(profile.last_error, None);
    assert_eq!(profile.last_refreshed_at, Some(failed_at + minutes(30)));
}

// Call 1 sees the entries a session's block holds.
// "Extraction receives each entry's text with its memory ids, so a reply
// that relies on an entry counts as `used` on every memory the entry
// cites". Entries get their own handles, `n1`, `n2`,..., listed with the
// handles of the in-context memories they cite, and a `used_injected_ids`
// naming an entry credits each memory it cites once. They're snapshotted
// with the turn when it's ingested, as its in-context set is.

/// Session `s1` holds a block whose profile entry cites `cat` and `tea`,
/// and the owner's turn in it is queued.
fn a_turn_after_the_block() -> (Harness, Uuid, Uuid) {
    let h = Harness::new();
    let cat = h.insert(fact(CAT));
    let tea = h.insert(fact(TEA));
    h.refresh_adding(
        PROFILE_NAME,
        &[(
            "Tim drinks green tea with his cat Miso nearby.",
            &[cat, tea],
        )],
    );
    h.block(Some("s1"));
    h.service
        .ingest_turn(BANK, &turn("s1", h.now() - minutes(1), "Tea time?"))
        .unwrap();
    (h, cat, tea)
}

#[test]
fn a_reply_relying_on_an_entry_is_credited_on_every_memory_it_cites() {
    let (h, cat, tea) = a_turn_after_the_block();
    let llm = FakeLlm::scripted(
        MODEL,
        vec![json!({"claims": [], "used_injected_ids": ["n1"]})],
    );
    let extracted = h.service.extract_next(BANK, &llm).unwrap().unwrap();
    let used: BTreeSet<Uuid> = extracted.used.iter().copied().collect();
    assert_eq!(used, BTreeSet::from([cat, tea]));
    assert_eq!(h.used(cat), 1);
    assert_eq!(h.used(tea), 1);
}

#[test]
fn a_turns_entries_are_the_ones_its_session_held_when_it_was_synced() {
    // The worker reaches the turn later. A refresh rewording the entry in
    // between changes neither what call 1 is shown nor what's credited, and
    // the snapshot goes once the turn is extracted.
    let (h, cat, tea) = a_turn_after_the_block();
    let snapshots = |h: &Harness| -> i64 { h.one("SELECT COUNT(*) FROM turn_entries", []) };
    assert_eq!(snapshots(&h), 1);

    let input = h.input(PROFILE_NAME);
    let llm = FakeLlm::scripted(
        MODEL,
        vec![reply(vec![edit(
            &input.entries[0].handle,
            "Tim only drinks tea when Miso the cat is asleep.",
            &handles(&input, &[cat, tea]),
        )])],
    );
    h.service
        .refresh_model(BANK, PROFILE_NAME, &llm, true)
        .unwrap();

    let llm = FakeLlm::scripted(
        MODEL,
        vec![json!({"claims": [], "used_injected_ids": ["n1"]})],
    );
    let extracted = h.service.extract_next(BANK, &llm).unwrap().unwrap();
    let user = &llm.requests()[0].user;
    assert!(
        user.contains("Tim drinks green tea with his cat Miso nearby."),
        "{user}"
    );
    assert!(
        !user.contains("asleep"),
        "call 1 saw the reworded entry:\n{user}"
    );
    let used: BTreeSet<Uuid> = extracted.used.iter().copied().collect();
    assert_eq!(used, BTreeSet::from([cat, tea]));
    assert_eq!(snapshots(&h), 0, "the snapshot outlived the extraction");
}

// The block-id fallback: "if Hermes gives no session
// id at `system_prompt_block()` time, the plugin sends the block id with
// its first `prefetch`, and the daemon persists the mapping then."
// `PrefetchRequest::block_id` names the block. The daemon keeps what each
// built block lists and cites by block id, since the cache may have rebuilt
// by then; a session that already has a mapping keeps it, and an unknown
// id, or another bank's, maps nothing.

fn prefetch_holding(h: &Harness, session: &str, query: &str, block: Option<Uuid>) -> Vec<Uuid> {
    h.service
        .prefetch(
            BANK,
            &PrefetchRequest {
                session_id: session.into(),
                query: query.into(),
                previous_query: None,
                previous_reply: None,
                block_id: block,
            },
        )
        .unwrap()
        .injected
}

fn mapped_block(h: &Harness, session: &str) -> Option<String> {
    let bank_id = h.bank_id();
    h.service
        .store()
        .unwrap()
        .connection()
        .query_row(
            "SELECT block_id FROM session_blocks WHERE bank_id = ?1 AND session_id = ?2",
            (bank_id, session),
            |row| row.get(0),
        )
        .ok()
}

#[test]
fn a_prefetch_carrying_the_block_id_maps_a_session_that_fetched_without_one() {
    let h = Harness::new();
    let cat = h.insert(fact(CAT));
    h.refresh_adding(PROFILE_NAME, &[("Tim has a cat called Miso.", &[cat])]);
    let block = h.block(None);
    assert!(h.in_context("s1").is_empty());

    let injected = prefetch_holding(&h, "s1", "is the cat called Miso", Some(block.id));
    assert!(!injected.contains(&cat), "a cited memory was injected");
    assert_eq!(h.in_context("s1"), vec![cat]);
    assert_eq!(mapped_block(&h, "s1"), Some(block.id.to_string()));

    let h = h.restart();
    assert_eq!(h.in_context("s1"), vec![cat]);
}

#[test]
fn the_block_id_finds_a_block_the_cache_has_since_replaced() {
    // The plugin holds the id of the block Hermes froze, which the cache
    // may have rebuilt since.
    let h = Harness::new();
    let cat = h.insert(fact(CAT));
    let tea = h.insert(fact(TEA));
    h.refresh_adding(PROFILE_NAME, &[("Tim has a cat called Miso.", &[cat])]);
    let held = h.block(None);
    h.refresh_adding(PROFILE_NAME, &[("Tim likes green tea.", &[tea])]);
    let newer = h.block(None);
    assert_ne!(newer.id, held.id);
    assert!(newer.cited.contains(&tea));

    prefetch_holding(&h, "s1", "hello there", Some(held.id));
    assert_eq!(h.in_context("s1"), vec![cat]);
    assert_eq!(mapped_block(&h, "s1"), Some(held.id.to_string()));
}

#[test]
fn a_block_id_never_replaces_a_mapping_and_an_unknown_one_maps_nothing() {
    let h = Harness::new();
    let cat = h.insert(fact(CAT));
    let tea = h.insert(fact(TEA));
    h.refresh_adding(PROFILE_NAME, &[("Tim has a cat called Miso.", &[cat])]);
    let held = h.block(Some("s1"));
    h.refresh_adding(PROFILE_NAME, &[("Tim likes green tea.", &[tea])]);
    let newer = h.block(None);

    prefetch_holding(&h, "s1", "hello there", Some(newer.id));
    assert_eq!(mapped_block(&h, "s1"), Some(held.id.to_string()));
    assert_eq!(h.in_context("s1"), vec![cat]);

    prefetch_holding(&h, "s2", "hello there", Some(Uuid::nil()));
    assert_eq!(mapped_block(&h, "s2"), None);
    assert!(h.in_context("s2").is_empty());
}

#[test]
fn a_block_id_from_another_bank_maps_nothing() {
    // Nothing refers across banks (CONTEXT.md, "Bank").
    let h = Harness::new();
    h.service
        .ensure_bank_with_models("other", &BankIdentity::default())
        .unwrap();
    let cat = h.insert(fact(CAT));
    h.refresh_adding(PROFILE_NAME, &[("Tim has a cat called Miso.", &[cat])]);
    h.block(None);
    let theirs = h.service.system_prompt("other", None).unwrap();

    prefetch_holding(&h, "s1", "hello there", Some(theirs.id));
    assert_eq!(mapped_block(&h, "s1"), None);
    assert!(h.in_context("s1").is_empty());
}

// Refresh and budget regressions

/// A refresh LLM that, while its first call is in flight, has the owner
/// keep `keep`: a triggering write landing in the middle of a refresh.
struct KeepsDuringCall<'a> {
    service: &'a Service,
    keep: Uuid,
    calls: AtomicUsize,
}

impl LlmClient for KeepsDuringCall<'_> {
    fn model(&self) -> &str {
        MODEL
    }

    fn complete(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            self.service.keep(BANK, &[self.keep.to_string()]).unwrap();
        }
        Ok(LlmResponse {
            json: reply(vec![]),
            usage: None,
            latency: Duration::ZERO,
        })
    }
}

#[test]
fn a_trigger_during_a_refresh_is_refreshed_after_the_interval() {
    // The refresh selected its inputs before the keep, so
    // the kept memory was never shown to the LLM. The request the keep made
    // has to survive the refresh's completion and run once the minimum
    // interval has passed, not wait for the next write or the 04:00 sweep.
    let h = Harness::new();
    let surfing = h.insert(faded(fact("Tim once tried surfing in Raglan.")));
    h.says(notable(TEA));
    h.advance(minutes(5));
    let refreshed = h.now();
    let llm = KeepsDuringCall {
        service: &h.service,
        keep: surfing,
        calls: AtomicUsize::new(0),
    };
    let ran = h.service.run_refreshes(&llm).unwrap();
    assert_eq!(ran.ran.len(), 1);
    assert_eq!(llm.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        ran.next_due,
        Some(refreshed + minutes(30)),
        "the trigger during the refresh was dropped"
    );

    h.set(refreshed + minutes(30));
    let quiet = quiet_llm(1);
    assert_eq!(h.tick(&quiet).ran.len(), 1);
    assert!(quiet.requests()[0].user.contains("surfing in Raglan"));
}

#[test]
fn a_stated_end_holds_through_its_unit_in_the_block() {
    // A stored time is the start of its unit, so an end
    // of 1 October holds through 1 October, and an end in October through
    // October (`Window::closes_at`). A future ending, already recorded with
    // `ended_by`, hasn't ended anything yet.
    let h = Harness::new();
    let lisbon = h.insert(Memory {
        kind: "state",
        volatility: Some("months"),
        valid_until: Some((local("2026-10-01T00:00"), "day")),
        ..fact("Tim is in Lisbon until 1 October 2026.")
    });
    let course = h.insert(Memory {
        kind: "state",
        volatility: Some("months"),
        valid_until: Some((local("2026-10-01T00:00"), "month")),
        ..fact("Tim is doing a pottery course until October 2026.")
    });
    let acme = h.insert(fact("Tim works at Acme."));
    let leaving = h.insert(event(
        "Tim leaves Acme on 30 November 2026.",
        "2026-11-30T00:00",
    ));
    h.mark_ended(acme, leaving, local("2026-11-30T00:00"));
    h.refresh_adding(
        PROFILE_NAME,
        &[
            ("Tim is in Lisbon for now.", &[lisbon]),
            ("Tim is doing a pottery course.", &[course]),
            ("Tim works at Acme.", &[acme]),
        ],
    );

    let block = h.block(None);
    for entry in [
        "Tim is in Lisbon for now.",
        "Tim is doing a pottery course.",
        "Tim works at Acme.",
    ] {
        assert!(
            block.text.contains(entry),
            "{entry:?} wasn't rendered:\n{}",
            block.text
        );
    }

    // The day ends at local midnight; the month and November go on.
    h.set(local("2026-10-02T00:00"));
    let block = h.block(None);
    assert!(!block.text.contains("Tim is in Lisbon for now."));
    assert!(block.text.contains("Tim is doing a pottery course."));
    assert!(block.text.contains("Tim works at Acme."));
    h.set(local("2026-11-01T00:00"));
    assert!(
        !h.block(None)
            .text
            .contains("Tim is doing a pottery course.")
    );
}

#[test]
fn a_stated_end_holds_through_its_unit_on_the_agenda() {
    let h = Harness::new();
    let gutters = h.insert(Memory {
        valid_until: Some((local("2026-10-01T00:00"), "day")),
        ..task("Tim needs to clean the gutters today.")
    });
    let swim = h.insert(Memory {
        valid_until: Some((local("2026-10-01T00:00"), "month")),
        ..recurring(
            "Tim swims on Tuesdays until October.",
            Some("FREQ=WEEKLY;BYDAY=TU"),
            "2026-01-06T00:00",
        )
    });
    let yoga = h.insert(recurring(
        "Tim does yoga on Thursdays.",
        Some("FREQ=WEEKLY;BYDAY=TH"),
        "2026-01-01T00:00",
    ));
    let stopping = h.insert(event(
        "Tim stops yoga on 1 December 2026.",
        "2026-12-01T00:00",
    ));
    h.mark_ended(yoga, stopping, local("2026-12-01T00:00"));

    let agenda = h.agenda();
    assert_eq!(agenda.undated_tasks, vec![gutters]);
    let routines: BTreeSet<Uuid> = agenda.routines.iter().copied().collect();
    assert_eq!(routines, BTreeSet::from([swim, yoga]));

    h.set(local("2026-10-02T00:00"));
    let agenda = h.agenda();
    assert!(agenda.undated_tasks.is_empty());
    assert!(agenda.routines.contains(&swim));
}

#[test]
fn the_whole_block_stays_within_the_budget_and_records_only_what_it_renders() {
    // Every model "shares about 800 tokens of
    // `system_prompt_block()` with the agenda". The agenda keeps its own
    // caps and is laid out first, so today's appointment can't be pushed
    // out by a model; the models get what's left. What the block lists or
    // cites, and so puts in a session's context, is only what it rendered.
    let h = Harness::new();
    let today = h.insert(event(
        "Tim has a dentist appointment this evening at the clinic on Queen Street.",
        "2026-10-01T00:00",
    ));
    for n in 0..14 {
        let day = 2 + n % 7;
        h.insert(event(
            sentence(format!(
                "Tim has appointment number {n} with the planning committee about the new library."
            )),
            &format!("2026-10-{day:02}T00:00"),
        ));
    }
    for n in 0..4 {
        h.insert(recurring(
            sentence(format!(
                "Tim goes to evening class number {n} at the community hall weekly."
            )),
            Some("FREQ=WEEKLY;BYDAY=WE"),
            "2026-01-07T00:00",
        ));
    }
    for n in 0..5 {
        h.insert(task(sentence(format!(
            "Tim needs to sort out household job number {n} before the weekend."
        ))));
    }
    let facts: Vec<Uuid> = (0..25)
        .map(|n| {
            h.insert(fact(sentence(format!(
                "Tim's profile fact number {n} is here."
            ))))
        })
        .collect();
    let entries: Vec<(String, Uuid)> = facts
        .iter()
        .enumerate()
        .map(|(n, fact)| {
            (
                format!(
                    "Tim is described here by profile entry number {n:02}, as fact {n:02} says."
                ),
                *fact,
            )
        })
        .collect();
    let adds: Vec<(&str, Vec<Uuid>)> = entries
        .iter()
        .map(|(text, fact)| (text.as_str(), vec![*fact]))
        .collect();
    let adds: Vec<(&str, &[Uuid])> = adds.iter().map(|(t, c)| (*t, c.as_slice())).collect();
    h.refresh_adding(PROFILE_NAME, &adds);

    let block = h.block(Some("s1"));
    let budget = h.tuning.mental_models.budget as usize;
    let tokens = estimate_tokens(&block.text);
    assert!(
        tokens <= budget,
        "the block is {tokens} tokens:\n{}",
        block.text
    );
    assert!(
        block.agenda.contains(&today),
        "today's appointment was dropped"
    );
    assert!(block.text.contains("dentist appointment this evening"));
    assert!(
        block.text.contains("memory_recall"),
        "the pointer line was dropped"
    );

    // Only what's rendered is recorded, and so put in context.
    for (text, fact) in &entries {
        assert_eq!(
            block.cited.contains(fact),
            block.text.contains(text.as_str()),
            "{text:?}"
        );
    }
    let in_context: BTreeSet<Uuid> = h.in_context("s1").into_iter().collect();
    let shown: BTreeSet<Uuid> = block.agenda.iter().chain(&block.cited).copied().collect();
    assert_eq!(in_context, shown);
}
