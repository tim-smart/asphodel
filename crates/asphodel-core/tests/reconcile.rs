//! Reconciliation (call 2) contracts cover neighbour search, labels,
//! supersession and accesses. A repeat becomes an access on the memory that
//! already exists, never a second copy of it.
//!
//! These are golden tests against `FakeLlm`: each scripts call 1's reply and
//! then call 2's, and checks what's committed through `show_memory`, or
//! checks the input call 2 is given. Call 2's handles (`c1`, `n1`, …) are
//! read from the input first, the way call 1's tests read entity and memory
//! handles. Fixture memories are extracted the same way, from turns said
//! before the turn under test.
//!
//! Every service here runs on a `SimulatedClock` stopped at one instant
//! unless a test advances it.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use asphodel_core::Service;
use asphodel_core::clock::{Clock, SimulatedClock};
use asphodel_core::config::Tuning;
use asphodel_core::entities::LinkRequest;
use asphodel_core::extraction::{
    Call2Input, Committed, EDIT_END_CLEARED, EDIT_END_REPOINTED, EDIT_ENDED, EDIT_KEPT,
    EDIT_REFINED, EDIT_RETRACTED, EDIT_SIGNIFICANCE_RAISED, ExtractError, Extracted, Prepared,
    call2_request,
};
use asphodel_core::ingest::{Document, Ingested, Turn};
use asphodel_core::inspect::{AccessEntry, BankOverview, ChunkState, ChunkView, MemoryView};
use asphodel_core::models::{
    Embedder, FakeEmbedder, FakeLlm, FakeReranker, LlmClient, LlmError, LlmRequest, LlmResponse,
    Models,
};
use asphodel_core::queue::{Failure, Lease};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{OpenOptions, Store, VectorIndex};
use asphodel_core::strength::{TimePrecision, WorldTime};
use jiff::civil::{Date, DateTime, date};
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use rusqlite::types::FromSql;
use serde_json::{Value, json};
use uuid::Uuid;

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";
const MODEL: &str = "fake-llm";

/// A turn's message time: 19:30 on Thursday 1 October 2026 in Auckland,
/// which is on daylight time (UTC+13).
const T1: &str = "2026-10-01T06:30:00Z";
/// The next day, at the same time.
const T2: &str = "2026-10-02T06:30:00Z";

/// When fixture memories were said, unless a test moves them: well before
/// any turn or document a test ingests.
const EARLIER: &str = "2026-09-01T00:00:00Z";
/// A day later, for a fixture that has to be the newer of two.
const NEXT_DAY: &str = "2026-09-02T00:00:00Z";

/// The fake embedder's floor in [`tuning`].
const FLOOR: f64 = 0.5;

const TEA: &str = "Tim likes green tea.";
const TEA_AGAIN: &str = "Tim really likes green tea.";
const TEA_A_LOT: &str = "Tim likes green tea a lot.";
const ACME: &str = "Tim works at Acme.";
const ACME_STILL: &str = "Tim still works at Acme.";
const ACME_LEFT: &str = "Tim no longer works at Acme, having left in August 2026.";
const BERLIN: &str = "Tim lives in Berlin.";
const LISBON: &str = "Tim lives in Lisbon.";
const MOVED: &str = "Tim moved out of Berlin and now lives in Lisbon.";
const DENTIST_8: &str = "Tim's dentist appointment is on 8 October 2026.";
const DENTIST_9: &str = "Tim's dentist appointment is on 9 October 2026.";
const JAPAN: &str = "Tim is going to Japan in 2027.";
const TOKYO: &str = "Tim is going to Tokyo, Japan in April 2027.";
const COFFEE: &str = "Tim drinks coffee.";
const NO_COFFEE: &str = "Tim no longer drinks coffee.";
const TAX_TASK: &str = "Tim needs to file the tax return.";
const TAX_FILED: &str = "Tim filed the tax return.";
const TAX_FILED_LATER: &str = "Tim filed the tax return on 2 October 2026.";
const TAX_FILED_ONLINE: &str = "Tim filed the tax return online.";
const TAX_NOT_FILED: &str = "Tim has not filed the tax return.";
const TAX_WORRY: &str = "Tim is worried about the tax return.";
const NEVER_ACME: &str = "Tim has never worked at Acme.";
const NO_DENTIST: &str = "Tim has no dentist appointment.";
const MAYA: &str = "Tim's daughter is called Maya.";
const BIKE: &str = "Tim's bike is a Brompton.";
const SURFING: &str = "Tim once tried surfing in Raglan.";
const CAT: &str = "Tim's cat is called Miso.";
const WEATHER: &str = "The weather in Wellington was sunny.";
const ANA: &str = "Tim's sister Ana lives in Porto.";
const ANA_DOG: &str = "Ana adopted a greyhound.";
const PASSPORT: &str = "Tim needs to renew his passport.";
const BOOKING: &str = "Tim has a hotel booking.";
const FLOWERS: &str = "Tim sent flowers to Sam.";
const FLOWERS_WIFE: &str = "Tim sent flowers to Sam, his wife.";

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

/// The start of a local day, at day precision.
fn day(date: &str) -> Option<WorldTime> {
    let at = local(date);
    let precision = TimePrecision::Day;
    Some(WorldTime { at, precision })
}

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-reconcile-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn data(&self) -> PathBuf {
        self.0.join("data")
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A floor for each fake model, the embedder's at `floor`, and bank time
/// at full speed with or without turns.
fn tuning(floor: f64) -> Tuning {
    Tuning::from_toml(&format!(
        "[clock]\nquiet_rate = 1.0\n\
         [injection.reranker_floors]\n\"{}\" = 0.0\n\
         [ranking.relevance_scales]\n\"{0}\" = 1.0\n\
         [reconcile.embedding_floors]\n\"{}\" = {floor:?}\n",
        FakeReranker::MODEL_ID,
        FakeEmbedder::MODEL_ID,
    ))
    .unwrap()
}

/// The owner is Tim and the assistant is Hermes.
fn identity() -> BankIdentity {
    BankIdentity {
        owner_name: Some("Tim".into()),
        owner_platform_ids: vec!["discord:1234".into()],
        assistant_name: Some("Hermes".into()),
        timezone: Some(TZ.into()),
    }
}

/// An LLM that answers its first call and then fails every call with
/// `error`.
struct ThenFails {
    first: Mutex<Option<Value>>,
    error: fn() -> LlmError,
}

impl LlmClient for ThenFails {
    fn model(&self) -> &str {
        MODEL
    }

    fn complete(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        let json = self.first.lock().unwrap().take().ok_or_else(self.error)?;
        let latency = Duration::ZERO;
        Ok(LlmResponse {
            json,
            usage: None,
            latency,
        })
    }
}

/// A service on the fake models with two banks, `main` and `other`. Field
/// order matters: the service drops before the directory it lives in.
struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    _dir: TestDir,
}

impl Harness {
    fn new() -> Self {
        Self::open(tuning(FLOOR))
    }

    /// `main` hands out up to `n` chunks at once.
    fn with_concurrency(n: u32) -> Self {
        let mut tuning = tuning(FLOOR);
        tuning.llm.concurrency = n;
        Self::open(tuning)
    }

    fn open(tuning: Tuning) -> Self {
        let dir = TestDir::new();
        let clock = Arc::new(SimulatedClock::new(at(START)));
        let store = Store::open(&dir.data(), OpenOptions::default(), clock.clone()).unwrap();
        let service = Service::with_models(clock.clone(), store, tuning, Models::fake()).unwrap();
        for bank in ["main", "other"] {
            service.ensure_bank_with_models(bank, &identity()).unwrap();
        }
        Self {
            service,
            clock,
            _dir: dir,
        }
    }

    /// Drops the service, releasing the data-dir lock, and opens a new one on
    /// the same data dir and clock: a daemon restart, which runs any pending
    /// migration.
    fn restart(self) -> Self {
        let Harness {
            service,
            clock,
            _dir: dir,
        } = self;
        drop(service);
        let store = Store::open(&dir.data(), OpenOptions::default(), clock.clone()).unwrap();
        let service =
            Service::with_models(clock.clone(), store, tuning(FLOOR), Models::fake()).unwrap();
        Self {
            service,
            clock,
            _dir: dir,
        }
    }

    fn now(&self) -> Timestamp {
        self.clock.now()
    }

    fn advance(&self, hours: i64) {
        self.clock.advance(SignedDuration::from_hours(hours));
    }

    /// Raw SQL, for states the API can't reach and data the API can't show.
    fn one<T: FromSql, P: rusqlite::Params>(&self, sql: &str, params: P) -> T {
        self.service
            .store()
            .unwrap()
            .connection()
            .query_row(sql, params, |row| row.get(0))
            .unwrap()
    }

    fn sql(&self, sql: &str) {
        let store = self.service.store().unwrap();
        store.connection().execute_batch(sql).unwrap();
    }

    fn show(&self, memory: Uuid) -> MemoryView {
        self.service
            .show_memory("main", &memory.to_string())
            .unwrap()
    }

    /// A memory's own accesses, without those it inherits.
    fn accesses(&self, memory: Uuid) -> Vec<AccessEntry> {
        let accesses = self.show(memory).accesses.into_iter();
        accesses.filter(|a| a.inherited_from.is_none()).collect()
    }

    fn access_kinds(&self, memory: Uuid) -> Vec<String> {
        let accesses = self.accesses(memory).into_iter();
        accesses.map(|access| access.kind).collect()
    }

    /// The kinds of the edits logged on a memory, oldest first.
    fn edits(&self, memory: Uuid) -> Vec<String> {
        let edits = self.show(memory).edits.into_iter();
        edits.map(|edit| edit.kind).collect()
    }

    fn change(&self, memory: Uuid) -> Change {
        let view = self.show(memory);
        let member = view.chain.members.iter().find(|m| m.id == memory);
        Change {
            valid_until: view.window.valid_until,
            window_confidence: view.window.window_confidence,
            retracted_at: view.retracted_at,
            superseded_by: member.and_then(|member| member.superseded_by),
            ended_by: view.chain.ended_by,
        }
    }

    /// The significance extraction gave, and the owner's setting.
    fn significance(&self, memory: Uuid) -> (String, Option<String>) {
        let significance = self.show(memory).significance;
        (significance.extracted, significance.owner)
    }

    fn bank(&self) -> BankOverview {
        let banks = self.service.banks().unwrap();
        banks.into_iter().find(|bank| bank.name == "main").unwrap()
    }

    fn memories(&self) -> usize {
        let counts = self.bank().memories;
        counts.live + counts.superseded + counts.ended + counts.retracted + counts.forgetting
    }

    /// The one chunk of a turn in `main`.
    fn chunk(&self, source: Uuid) -> ChunkView {
        let detail = self.service.show_source("main", &source.to_string());
        detail.unwrap().chunks.remove(0)
    }

    fn lease(&self) -> Lease {
        let lease = self.service.claim_chunk("main").unwrap();
        lease.expect("a chunk is queued")
    }

    fn ingest(&self, turn: &Turn) -> Uuid {
        self.service.ingest_turn("main", turn).unwrap().source
    }

    /// The owner says `user` in session `s1` at [`T1`] and the assistant
    /// answers "Noted.". Returns the source.
    fn says(&self, user: &str) -> Uuid {
        self.says_at(T1, user)
    }

    fn says_at(&self, message_at: &str, user: &str) -> Uuid {
        self.ingest(&turn("s1", message_at, user, "Noted."))
    }

    fn doc(&self, id: &str, text: &str, reference_date: Date) -> Ingested {
        let document = Document {
            document_id: id.into(),
            text: text.into(),
            reference_date,
            reference_date_exact: true,
            timezone: Some(TZ.into()),
        };
        self.service.ingest_document("main", &document).unwrap()
    }

    /// A memory in `bank` extracted from a turn saying `claim`'s quote at
    /// `message_at`. When a neighbour runs call 2, call 2 labels `labels`
    /// on their neighbours, and the claim is new.
    fn seed_in(&self, bank: &str, message_at: &str, claim: Value, labels: &[(Uuid, &str)]) -> Uuid {
        let quote = claim["quote"].as_str().unwrap().to_string();
        let turn = turn("fixtures", message_at, &quote, "Noted.");
        self.service.ingest_turn(bank, &turn).unwrap();
        let call1 = reply(vec![claim]);
        let lease = self.service.claim_chunk(bank).unwrap().unwrap();
        let mut replies = vec![call1.clone()];
        match self.service.call2_input(&lease, &call1, &[]).unwrap() {
            Some(input) => {
                let labels: Vec<(String, &str)> = labels
                    .iter()
                    .map(|(memory, label)| (neighbour_handle(&input, *memory), *label))
                    .collect();
                let claim = &input.claims[0].handle;
                replies.push(call2_reply(vec![labelled(claim, &labels)]));
            }
            None => assert!(labels.is_empty(), "call 2 runs to label {quote:?}"),
        }
        let llm = FakeLlm::scripted(MODEL, replies);
        let extracted = self.service.extract_chunk(lease, &llm, &[]).unwrap();
        extracted.memories[0]
    }

    fn seed(&self, message_at: &str, claim: Value, labels: &[(Uuid, &str)]) -> Uuid {
        self.seed_in("main", message_at, claim, labels)
    }

    /// A memory in `main`, said at [`EARLIER`].
    fn fixture(&self, claim: Value) -> Uuid {
        self.seed(EARLIER, claim, &[])
    }

    /// A minor fact in `main`, said at [`EARLIER`].
    fn fact(&self, content: &str) -> Uuid {
        self.fixture(said(content, "fact"))
    }

    /// The event that Tim left Acme in August, said at `message_at` and
    /// ending `acme`.
    fn left_acme(&self, message_at: &str, acme: Uuid) -> Uuid {
        let left = said(ACME_LEFT, "event").at("valid_from", "2026-08", "month");
        self.seed(message_at, left, &[(acme, "ends")])
    }

    /// `count` versions of one fact in `main`, each refined into the next.
    /// The first is extracted; the rest are copies of it, as that many
    /// refinements would leave them, inserted directly since the API can't
    /// make thousands in reasonable time. Returns the head.
    fn insert_chain(&self, content: &str, count: usize) -> Uuid {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let first = self.fact(content);
        let vector = FakeEmbedder.embed(&[content]).unwrap().remove(0);
        let store = self.service.store().unwrap();
        let mut conn = store.connection();
        let tx = conn.transaction().unwrap();
        let (bank_id, mut previous): (i64, i64) = tx
            .query_row(
                "SELECT bank_id, id FROM memories WHERE uuid = ?1",
                [first.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        let first_id = previous;
        let mut head = first;
        for _ in 1..count {
            let uuid = Uuid::from_u128(
                (0xf2_u128 << 120) | u128::from(NEXT.fetch_add(1, Ordering::Relaxed)),
            );
            tx.execute(
                "INSERT INTO memories (uuid, bank_id, content, kind, significance, chunk_id,
                                       source_start, source_end, observed_at, window_confidence,
                                       created_at, updated_at)
                 SELECT ?1, bank_id, content, kind, significance, chunk_id, source_start,
                        source_end, observed_at, window_confidence, created_at, updated_at
                 FROM memories WHERE id = ?2",
                (uuid.to_string(), first_id),
            )
            .unwrap();
            let id = tx.last_insert_rowid();
            store.vectors().upsert(&tx, bank_id, id, &vector).unwrap();
            tx.execute(
                "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
                 SELECT bank_id, ?2, kind, at, turn FROM accesses WHERE memory_id = ?1",
                (first_id, id),
            )
            .unwrap();
            tx.execute(
                "UPDATE memories SET superseded_by = ?2 WHERE id = ?1",
                (previous, id),
            )
            .unwrap();
            previous = id;
            head = uuid;
        }
        tx.commit().unwrap();
        head
    }
}

/// What ending, retracting and refining write on the memory they change.
#[derive(Debug, Clone, PartialEq)]
struct Change {
    valid_until: Option<WorldTime>,
    window_confidence: String,
    retracted_at: Option<Timestamp>,
    superseded_by: Option<Uuid>,
    ended_by: Option<Uuid>,
}

/// The owner's turn on the CLI: no author.
fn turn(session: &str, message_at: &str, user: &str, assistant: &str) -> Turn {
    Turn {
        session_id: session.into(),
        message_at: at(message_at),
        timezone: Some(TZ.into()),
        user_text: user.into(),
        assistant_text: assistant.into(),
        author: None,
        platform: Some("cli".into()),
        recall_id: None,
        forget_requested: false,
    }
}

/// Call 2's input for the head of `main`'s queue when call 1 replies
/// `call1`, or `None` when call 2 won't run. The lease is released when it
/// drops.
fn try_call2(
    h: &Harness,
    call1: &Value,
    in_context: &[Uuid],
) -> Result<Option<Call2Input>, ExtractError> {
    h.service.call2_input(&h.lease(), call1, in_context)
}

fn call2(h: &Harness, call1: &Value) -> Option<Call2Input> {
    try_call2(h, call1, &[]).unwrap()
}

/// Extracts the head of `main`'s queue with call 1 answering only `call1`,
/// and checks call 2 didn't run.
fn extract_alone(h: &Harness, call1: Value) -> Extracted {
    let llm = FakeLlm::scripted(MODEL, vec![call1]);
    let extracted = h.service.extract_chunk(h.lease(), &llm, &[]).unwrap();
    assert_eq!(llm.requests().len(), 1, "call 1 only");
    extracted
}

/// Extracts the head of `main`'s queue with call 1 answering `call1` and
/// call 2 answering `call2`.
fn reconcile(h: &Harness, call1: Value, call2: Value, in_context: &[Uuid]) -> Extracted {
    let llm = FakeLlm::scripted(MODEL, vec![call1, call2]);
    let extracted = h
        .service
        .extract_chunk(h.lease(), &llm, in_context)
        .unwrap();
    assert_eq!(llm.requests().len(), 2, "call 1 and call 2");
    extracted
}

/// Extracts the head of `main`'s queue where call 1 finds the claims in
/// `call1` and call 2 labels claim `k` `labels[k].1` on the neighbour
/// `labels[k].0`.
fn label_each_with(
    h: &Harness,
    call1: Value,
    labels: &[(Uuid, &str)],
    in_context: &[Uuid],
) -> Extracted {
    let input = try_call2(h, &call1, in_context).unwrap();
    let call2 = each(&input.expect("call 2 runs"), labels);
    reconcile(h, call1, call2, in_context)
}

fn label_each(h: &Harness, call1: Value, labels: &[(Uuid, &str)]) -> Extracted {
    label_each_with(h, call1, labels, &[])
}

/// [`label_each`] for one claim.
fn one_label(h: &Harness, call1: Value, neighbour: Uuid, label: &str) -> Extracted {
    label_each(h, call1, &[(neighbour, label)])
}

fn neighbour_handle(input: &Call2Input, memory: Uuid) -> String {
    let neighbour = input.neighbours.iter().find(|n| n.memory == memory);
    neighbour.expect("a neighbour").handle.clone()
}

/// The neighbours' memories.
fn shown(input: &Call2Input) -> BTreeSet<Uuid> {
    input.neighbours.iter().map(|n| n.memory).collect()
}

/// The memories shown for the claim at `index` in call 1's reply.
fn shown_for(input: &Call2Input, index: usize) -> BTreeSet<Uuid> {
    let claim = input.claims.iter().find(|claim| claim.claim == index);
    let claim = claim.unwrap_or_else(|| panic!("claim {index} reaches call 2"));
    let memory = |handle: &String| {
        let neighbour = input.neighbours.iter().find(|n| &n.handle == handle);
        neighbour
            .expect("a claim's neighbour is in the input")
            .memory
    };
    claim.neighbours.iter().map(memory).collect()
}

// Call 1's reply, as `FakeLlm` scripts it.

/// A claim with nothing but its sentence, kind and quote: minor, high window
/// confidence, no times and no entities.
fn claim(content: &str, kind: &str, quote: &str) -> Value {
    json!({
        "content": content, "kind": kind, "quote": quote, "significance": "minor",
        "remember_this": false, "changes_something": false, "window_confidence": "high",
        "valid_from": null, "valid_until": null, "until_event": null, "due_at": null,
        "volatility": null, "recurrence_text": null, "recurrence_rrule": null,
        "recurrence_start": null, "entities": [],
    })
}

/// A claim quoting all of its sentence, as a fixture's turn says it.
fn said(content: &str, kind: &str) -> Value {
    claim(content, kind, content)
}

trait With: Sized {
    fn with(self, key: &str, value: Value) -> Value;

    /// Sets `key` to a time as call 1 gives it.
    fn at(self, key: &str, at: &str, precision: &str) -> Value {
        self.with(key, json!({"at": at, "precision": precision}))
    }

    fn significance(self, level: &str) -> Value {
        self.with("significance", json!(level))
    }

    fn remember_this(self) -> Value {
        self.with("remember_this", json!(true))
    }
}

impl With for Value {
    fn with(mut self, key: &str, value: Value) -> Value {
        self[key] = value;
        self
    }
}

fn changes(claim: Value) -> Value {
    claim.with("changes_something", json!(true))
}

/// The claim with every time cleared.
fn undated(claim: Value) -> Value {
    claim
        .with("valid_from", Value::Null)
        .with("valid_until", Value::Null)
        .with("due_at", Value::Null)
}

fn reply(claims: Vec<Value>) -> Value {
    json!({"claims": claims, "used_injected_ids": []})
}

// Call 2's reply.

fn labelled(claim: &str, labels: &[(String, &str)]) -> Value {
    let labels: Vec<Value> = labels
        .iter()
        .map(|(neighbour, label)| json!({"neighbour": neighbour, "label": label}))
        .collect();
    json!({"claim": claim, "labels": labels})
}

fn call2_reply(claims: Vec<Value>) -> Value {
    json!({"claims": claims})
}

/// Call 2's reply labelling claim `k` `labels[k].1` on the neighbour
/// `labels[k].0`.
fn each(input: &Call2Input, labels: &[(Uuid, &str)]) -> Value {
    let claims = labels.iter().enumerate().map(|(k, (neighbour, label))| {
        let labels = [(neighbour_handle(input, *neighbour), *label)];
        labelled(&input.claims[k].handle, &labels)
    });
    call2_reply(claims.collect())
}

// When call 2 runs.

#[test]
fn call_2_runs_only_for_a_neighbour_in_the_bank_above_the_models_floor() {
    // The same paraphrase runs call 2 under the fake floor and doesn't under
    // a stricter one, so the floor read is the configured one.
    let again = || reply(vec![claim(TEA_AGAIN, "fact", "I really like green tea")]);
    for (floor, runs) in [(FLOOR, true), (0.99, false)] {
        let h = Harness::open(tuning(floor));
        h.fact(TEA);
        h.says("I really like green tea.");
        assert_eq!(call2(&h, &again()).is_some(), runs, "floor {floor}");
    }

    // A unit that touches nothing known in its bank costs one call: the cat
    // is below the floor, and another bank's memories are never searched.
    let h = Harness::new();
    let tea = h.fact(TEA);
    h.fact(CAT);
    h.seed_in("other", EARLIER, said(WEATHER, "event"), &[]);
    h.says(WEATHER);
    let weather = reply(vec![said(WEATHER, "event")]);
    assert_eq!(call2(&h, &weather), None);
    let extracted = extract_alone(&h, weather);
    assert_eq!(h.show(extracted.memories[0]).sentence, WEATHER);

    // A neighbour above the floor runs call 2.
    let before = (h.change(tea), h.accesses(tea));
    h.says_at("2026-10-01T06:40:00Z", "I really like green tea.");
    let input = call2(&h, &again()).expect("call 2 runs");
    assert_eq!(input.claims.len(), 1);
    assert_eq!(input.claims[0].claim, 0);
    assert_eq!(input.claims[0].content, TEA_AGAIN);
    assert_eq!(input.claims[0].observed_at, at("2026-10-01T06:40:00Z"));
    assert!(!input.claims[0].flagged);
    assert!(shown_for(&input, 0).contains(&tea));
    let neighbour = input.neighbours.iter().find(|n| n.memory == tea).unwrap();
    assert_eq!(neighbour.content, TEA);
    assert_eq!(neighbour.observed_at, at(EARLIER));
    assert!(!neighbour.ended);

    // A claim with no labels is new.
    let extracted = reconcile(&h, again(), call2_reply(vec![]), &[]);
    assert_eq!(extracted.memories.len(), 1);
    assert_eq!(h.show(extracted.memories[0]).sentence, TEA_AGAIN);
    assert_eq!((h.change(tea), h.accesses(tea)), before);
}

#[test]
fn a_flagged_claim_sees_open_tasks_and_current_states_of_its_entities() {
    let h = Harness::new();
    let user = h.service.show_entity("main", "user").unwrap().id;
    let passport = h.fixture(said(PASSPORT, "task"));
    let job_hunting = h.fixture(said("Tim is job hunting.", "state"));
    for memory in [passport, job_hunting] {
        let request = LinkRequest {
            memory: memory.to_string(),
            entity: "user".into(),
        };
        h.service.link_entity("main", &request).unwrap();
    }
    h.says("Travel documents are sorted.");
    let user_handle = {
        let input = h.service.call1_input(&h.lease(), &[]).unwrap();
        let candidate = input.candidates.iter().find(|c| c.entity == user);
        candidate.unwrap().handle.clone()
    };
    let me =
        json!({"entity": user_handle, "new_name": null, "new_kind": null, "surface_form": "I"});
    let travel = said("Travel documents are sorted.", "event").with("entities", json!([me]));

    // Nothing clears the floor, so an unflagged claim costs one call.
    assert_eq!(call2(&h, &reply(vec![travel.clone()])), None);

    // A claim that changes something gets the wider set: the open tasks and
    // current states linked to its entities.
    let input = call2(&h, &reply(vec![changes(travel.clone())])).expect("call 2 runs");
    assert!(input.claims[0].flagged);
    let found = shown_for(&input, 0);
    assert!(found.contains(&passport), "{found:?}");
    assert!(found.contains(&job_hunting), "{found:?}");

    // So does remember-this.
    let input = call2(&h, &reply(vec![travel.remember_this()])).expect("call 2 runs");
    assert!(input.claims[0].flagged);
    assert!(shown_for(&input, 0).contains(&passport));
}

#[test]
fn bm25_hits_fill_out_the_candidates_only_once_call_2_runs() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    let ana = h.fact(ANA);
    h.says("Ana adopted a greyhound. I like green tea.");
    let dog = claim(ANA_DOG, "event", "Ana adopted a greyhound");

    // A BM25 hit alone doesn't run call 2: only the vector floor decides.
    assert_eq!(call2(&h, &reply(vec![dog.clone()])), None);

    // Once another claim clears the floor, it fills out the candidates.
    let call1 = reply(vec![dog, claim(TEA, "fact", "I like green tea")]);
    let input = call2(&h, &call1).expect("call 2 runs");
    assert!(shown_for(&input, 0).contains(&ana));
    // BM25 matches any of the claim's words, so "Tim" can bring Ana's memory
    // in too; tea only has to be among them.
    assert!(shown_for(&input, 1).contains(&tea));
}

#[test]
fn faded_and_ended_memories_are_neighbours() {
    let h = Harness::new();
    // Trivial and untouched until it fades, but reconcile still matches
    // against it, so a re-mention strengthens it.
    let surfing = h.fixture(said(SURFING, "event").significance("trivial"));
    let acme = h.fact(ACME);
    h.left_acme(NEXT_DAY, acme);
    let fades = h.show(surfing).projection.fade.expect("it fades");
    h.clock
        .advance(fades.earliest_at.duration_since(h.now()) + SignedDuration::from_hours(24));
    assert!(!h.show(surfing).strength.recallable, "precondition: faded");

    h.says("I once tried surfing in Raglan. I work at Acme.");
    let call1 = reply(vec![
        claim(SURFING, "event", "I once tried surfing in Raglan"),
        claim(ACME, "fact", "I work at Acme"),
    ]);
    let input = call2(&h, &call1).expect("call 2 runs");
    assert!(shown_for(&input, 0).contains(&surfing));
    assert!(shown_for(&input, 1).contains(&acme));
    let ended = |memory: Uuid| {
        let neighbour = input.neighbours.iter().find(|n| n.memory == memory);
        neighbour.unwrap().ended
    };
    assert!(ended(acme));
    assert!(!ended(surfing));
}

#[test]
fn a_retracted_memory_isnt_a_neighbour_but_its_chain_head_is() {
    let h = Harness::new();
    let dentist_8 = h.fixture(said(DENTIST_8, "event"));
    let dentist_9 = h.seed(
        NEXT_DAY,
        said(DENTIST_9, "event"),
        &[(dentist_8, "retracts")],
    );
    // A retraction with no successor leaves no head to show.
    let maya = h.fact(MAYA);
    h.service.retract("main", &maya.to_string()).unwrap();

    h.says("My dentist appointment is on 8 October. My daughter is called Maya.");
    let call1 = reply(vec![
        claim(DENTIST_8, "event", "My dentist appointment is on 8 October"),
        claim(MAYA, "fact", "My daughter is called Maya"),
    ]);
    let input = call2(&h, &call1);
    let found = input.as_ref().map(shown).unwrap_or_default();
    assert!(!found.contains(&dentist_8), "retracted: {found:?}");
    assert!(!found.contains(&maya), "retracted: {found:?}");
    let input = input.expect("the dentist hit shows its head, so call 2 runs");
    assert!(shown_for(&input, 0).contains(&dentist_9));
}

/// sqlite-vec 0.1.9 refuses a KNN query for more than this many neighbours
/// (`SQLITE_VEC_VEC0_K_MAX`).
const KNN_K_MAX: usize = 4096;

#[test]
fn a_chain_past_the_knn_limit_neither_aborts_extraction_nor_crowds_out_a_neighbour() {
    // More versions of one fact than sqlite-vec returns from one KNN query
    // all sit nearer the claim than another matching memory. They collapse
    // to one head, so the search keeps asking for more: it has to stay within
    // the limit, or the chunk fails, and go past it, or the other memory is
    // left out and a repeat of it becomes a duplicate.
    let h = Harness::new();
    let head = h.insert_chain(TEA, KNN_K_MAX + 4);
    let lot = h.fact(TEA_A_LOT);
    let source = h.says("I like green tea.");

    let call1 = reply(vec![claim(TEA, "fact", "I like green tea")]);
    let input = call2(&h, &call1).expect("call 2 runs");
    assert_eq!(shown_for(&input, 0), BTreeSet::from([head, lot]));
    let extracted = one_label(&h, call1, head, "mentioned_again");
    assert!(extracted.memories.is_empty());
    assert_eq!(h.chunk(source).mentions, vec![head]);
}

// Labels from a newer claim.

#[test]
fn a_repeat_is_one_access_on_the_memory_and_outranks_used() {
    // The reply relied on the memory and the user restated it in the same
    // turn: one access, the heavier mention, at the source's ingested_at and
    // the bank's turn, like every access extraction writes.
    let h = Harness::new();
    let tea = h.fact(TEA);
    let before = (h.change(tea), h.accesses(tea));
    let source = h.says("I like green tea.");
    let used = {
        let input = h.service.call1_input(&h.lease(), &[tea]).unwrap();
        input.in_context[0].handle.clone()
    };
    let call1 = json!({
        "claims": [claim(TEA, "fact", "I like green tea")],
        "used_injected_ids": [used],
    });
    let extracted = label_each_with(&h, call1, &[(tea, "mentioned_again")], &[tea]);

    assert!(extracted.memories.is_empty());
    assert_eq!(extracted.used, vec![tea]);
    assert_eq!(h.memories(), 1);
    let mut accesses = before.1;
    accesses.push(AccessEntry {
        kind: "mentioned_again".into(),
        at: h.now(),
        turn: h.bank().turns,
        source: Some(source),
        inherited_from: None,
    });
    assert_eq!(h.accesses(tea), accesses);
    assert_eq!(h.change(tea), before.0);
    assert_eq!(h.chunk(source).state, ChunkState::Extracted);
}

#[test]
fn a_mention_that_matters_no_more_than_the_memory_is_absorbed_and_keeps_only_on_the_owners_word() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    let acme = h.fixture(said(ACME, "fact").significance("major"));
    let bike = h.fact(BIKE);
    h.says("I like green tea. I work at Acme. Remember this: my bike is a Brompton.");
    let again = |memory: Uuid| (memory, "mentioned_again");
    let extracted = label_each(
        &h,
        reply(vec![
            claim(TEA, "fact", "I like green tea"),
            claim(ACME, "fact", "I work at Acme").significance("trivial"),
            claim(BIKE, "fact", "my bike is a Brompton").remember_this(),
        ]),
        &[again(tea), again(acme), again(bike)],
    );

    // An equal or weaker mention is an access on the memory, and leaves its
    // significance as it was.
    assert!(extracted.memories.is_empty());
    assert_eq!(h.memories(), 3);
    assert_eq!(h.significance(tea), ("minor".into(), None));
    assert!(h.edits(tea).is_empty());
    assert_eq!(h.accesses(tea).len(), 2);
    assert_eq!(h.significance(acme), ("major".into(), None));
    assert!(h.edits(acme).is_empty());
    // Remember-this on a mention keeps the neighbour.
    assert_eq!(h.significance(bike), ("minor".into(), Some("kept".into())));
    assert_eq!(h.edits(bike), [EDIT_KEPT]);

    // Remember-this in a document keeps nothing; the mention still counts.
    h.doc(
        "notes",
        "Remember this: I like green tea.",
        date(2026, 9, 25),
    );
    let doc_tea = claim(TEA, "fact", "I like green tea").remember_this();
    one_label(&h, reply(vec![doc_tea]), tea, "mentioned_again");
    assert_eq!(h.significance(tea), ("minor".into(), None));
    assert!(h.edits(tea).is_empty());
    assert_eq!(h.accesses(tea).len(), 3);
}

#[test]
fn a_repeat_that_matters_more_than_the_memory_becomes_its_head() {
    // Call 2 can label a claim that says more than a memory, such as a
    // relationship the memory only names a person in, a repeat. A newer
    // claim well above the memory's significance isn't absorbed: it becomes
    // the chain head with its own sentence and significance, the memory is
    // superseded but not wrong, and the head inherits its accesses, as on
    // refines.
    for label in ["mentioned_again", "confirmed"] {
        let h = Harness::new();
        let old = h.fixture(said(FLOWERS, "event").significance("trivial"));
        let before = h.accesses(old);
        h.says("I sent flowers to Sam, my wife.");
        let wife = claim(FLOWERS_WIFE, "fact", "I sent flowers to Sam, my wife");
        let wife = reply(vec![wife.significance("major")]);
        let extracted = one_label(&h, wife, old, label);

        assert_eq!(extracted.memories.len(), 1, "{label}: a new memory");
        let new = extracted.memories[0];
        let view = h.show(new);
        assert_eq!(view.sentence, FLOWERS_WIFE, "{label}");
        assert_eq!(view.chain.head, new, "{label}");
        assert_eq!(h.significance(new), ("major".into(), None), "{label}");
        let change = h.change(old);
        assert_eq!(change.superseded_by, Some(new), "{label}");
        assert_eq!(change.retracted_at, None, "{label}");
        assert_eq!(h.edits(old), [EDIT_REFINED], "{label}");
        assert_eq!(h.significance(old), ("trivial".into(), None), "{label}");
        assert_eq!(h.accesses(old), before, "{label}");
        let inherited: Vec<AccessEntry> = view
            .accesses
            .into_iter()
            .filter(|access| access.inherited_from == Some(old))
            .map(|access| AccessEntry {
                inherited_from: None,
                ..access
            })
            .collect();
        assert_eq!(inherited, before, "{label}");
        assert_eq!(h.access_kinds(new), ["created"], "{label}");
    }
}

#[test]
fn a_repeat_is_weighed_against_the_owners_setting_and_the_promotion_gap() {
    // The neighbour counts at the higher of extraction's level and the
    // owner's, and a kept one is never outweighed, so a repeat never undoes
    // the owner's word. Short of the gap, or from an older claim, a mention
    // raises a neighbour the owner hasn't set instead.
    struct Case {
        extracted: &'static str,
        owner: Option<&'static str>,
        claim: &'static str,
        gap: u8,
        older: bool,
        /// The neighbour's significance after an absorbing mention, or
        /// `None` when the claim becomes the head.
        absorbed: Option<&'static str>,
    }
    let case = |extracted, owner, claim, gap, older, absorbed| Case {
        extracted,
        owner,
        claim,
        gap,
        older,
        absorbed,
    };
    for c in [
        case(
            "trivial",
            Some("kept"),
            "critical",
            1,
            false,
            Some("trivial"),
        ),
        case("minor", Some("trivial"), "minor", 1, false, Some("minor")),
        case("minor", Some("trivial"), "major", 1, false, None),
        case("trivial", Some("major"), "major", 1, false, Some("trivial")),
        case("trivial", Some("major"), "critical", 1, false, None),
        case("minor", None, "notable", 2, false, Some("notable")),
        case("minor", None, "major", 2, false, None),
        case("minor", None, "major", 1, true, Some("major")),
    ] {
        let name = format!(
            "{} set {:?}, claim {}, gap {}, older {}",
            c.extracted, c.owner, c.claim, c.gap, c.older
        );
        let mut tuning = tuning(FLOOR);
        tuning.reconcile.promotion_gap = c.gap;
        let h = Harness::open(tuning);
        let old = h.fixture(said(FLOWERS, "event").significance(c.extracted));
        if let Some(level) = c.owner {
            h.service
                .set_significance("main", &old.to_string(), Some(level))
                .unwrap();
        }
        let before = (h.edits(old), h.accesses(old).len());
        let quote = "I sent flowers to Sam, my wife";
        if c.older {
            h.doc("old-notes", &format!("{quote}."), date(2026, 8, 1));
        } else {
            h.says(&format!("{quote}."));
        }
        let wife = claim(FLOWERS_WIFE, "fact", quote).significance(c.claim);
        let extracted = one_label(&h, reply(vec![wife]), old, "mentioned_again");

        let owner = c.owner.map(String::from);
        match c.absorbed {
            Some(level) => {
                assert!(extracted.memories.is_empty(), "{name}: absorbed");
                assert_eq!(h.show(old).chain.head, old, "{name}");
                assert_eq!(h.significance(old), (level.into(), owner), "{name}");
                let mut edits = before.0;
                if level != c.extracted {
                    edits.push(EDIT_SIGNIFICANCE_RAISED.into());
                }
                assert_eq!(h.edits(old), edits, "{name}");
                assert_eq!(h.accesses(old).len(), before.1 + 1, "{name}");
            }
            None => {
                assert_eq!(extracted.memories.len(), 1, "{name}: a new head");
                let new = extracted.memories[0];
                assert_eq!(h.show(old).chain.head, new, "{name}");
                assert_eq!(h.significance(new), (c.claim.into(), None), "{name}");
                assert_eq!(h.significance(old), (c.extracted.into(), owner), "{name}");
            }
        }
    }
}

/// The owner says `first` on 1 October, extracted alone, and `again` the
/// next day, which call 2 labels `label` on the first's memory. Returns the
/// first's memory as it stood before `again`.
fn restate(first: Value, again: Value, label: &str) -> (Harness, MemoryView, Extracted) {
    let h = Harness::new();
    h.says(first["quote"].as_str().unwrap());
    let old = extract_alone(&h, reply(vec![first])).memories[0];
    let before = h.show(old);
    h.advance(24);
    h.says_at(
        T2,
        &format!("As I said, {}.", again["quote"].as_str().unwrap()),
    );
    let extracted = one_label(&h, reply(vec![again]), old, label);
    (h, before, extracted)
}

fn passport_task() -> Value {
    claim(PASSPORT, "task", "renew my passport")
}

/// A booking for 11 to 15 January 2027.
fn booking() -> Value {
    claim(BOOKING, "event", "hotel booking")
        .at("valid_from", "2027-01-11", "day")
        .at("valid_until", "2027-01-15", "day")
}

#[test]
fn repeat_labels_preserve_added_or_changed_windows_as_a_dated_head() {
    for label in ["mentioned_again", "confirmed"] {
        for field in ["due_at", "valid_from", "valid_until"] {
            for initial_date in [None, Some("2026-10-08T08:00")] {
                let mut first = passport_task();
                if let Some(date) = initial_date {
                    first = first.at(field, date, "minute");
                }
                let again = passport_task().at(field, "2026-10-09T08:00", "minute");
                let (h, before, extracted) = restate(first, again, label);
                let old = h.show(before.id);
                let head = h.show(old.chain.head);
                let window = serde_json::to_value(&head.window).unwrap();
                assert_eq!(
                    window[field]["at"],
                    json!(local("2026-10-09T08:00").to_string()),
                    "{label}, {field}, initial date {initial_date:?}"
                );
                assert_eq!(window[field]["precision"], "minute");
                assert_eq!(extracted.memories, vec![head.id]);
                assert_ne!(head.id, old.id);
                assert!(old.retracted_at.is_none(), "adding detail is a refinement");
            }
        }
    }
}

#[test]
fn an_undated_repeat_is_absorbed_into_the_dated_memory() {
    // A restatement without the dates doesn't replace the dated memory: the
    // second turn's observation-day default isn't a newly stated date.
    let task = passport_task().at("due_at", "2026-10-09T08:00", "minute");
    for (first, label) in [
        (task, "mentioned_again"),
        (booking(), "mentioned_again"),
        (booking(), "confirmed"),
    ] {
        let kind = first["kind"].clone();
        let (h, before, extracted) = restate(first.clone(), undated(first), label);
        let view = h.show(before.id);
        assert_eq!(view.chain.head, before.id, "{kind} {label}");
        assert!(extracted.memories.is_empty(), "{kind} {label}");
        assert_eq!(view.window, before.window, "{kind} {label}");
    }
}

#[test]
fn an_undated_retraction_inherits_the_window_of_its_own_kind_only() {
    // Carry-over, not relabelling as mentioned_again: a genuine correction
    // must still retract its predecessor, and missing times aren't evidence
    // the stated dates were cancelled. Only another task can replace an open
    // task, so a deadline never crosses kinds, nor does a state's window.
    let task = passport_task()
        .at("valid_from", "2026-10-11", "day")
        .at("valid_until", "2026-10-15", "day")
        .at("due_at", "2026-10-15T08:00", "minute");
    let state = claim(PASSPORT, "state", "renew my passport")
        .at("valid_from", "2026-10-11", "day")
        .at("valid_until", "2026-10-15", "day");
    for (first, kind, inherits) in [
        (task, "task", true),
        (booking(), "event", true),
        (state.clone(), "event", false),
        (state, "fact", false),
    ] {
        let again = undated(first.clone()).with("kind", json!(kind));
        let (h, before, extracted) = restate(first, again, "retracts");
        let old = h.show(before.id);
        let head = h.show(old.chain.head);
        assert_eq!(extracted.memories, vec![head.id], "{kind}");
        assert!(old.retracted_at.is_some(), "{kind}");
        assert_eq!(head.kind, kind);
        if inherits {
            assert_eq!(head.window, old.window, "{kind} keeps dates and precision");
        } else {
            // It keeps its own start, the observation day for an event,
            // rather than the old state's.
            assert!(head.window.valid_until.is_none(), "{kind}");
            let own_start = (kind == "event").then(|| local("2026-10-02T00:00"));
            assert_eq!(head.window.valid_from.map(|stamp| stamp.at), own_start);
        }
    }
}

#[test]
fn an_explicit_low_confidence_date_is_kept_whatever_the_label() {
    // The observation day with the fallback's precision and confidence: only
    // whether the start was supplied tells it apart from the default.
    for label in ["mentioned_again", "confirmed", "retracts"] {
        let again = claim(BOOKING, "event", "hotel booking")
            .at("valid_from", "2026-10-02", "day")
            .with("window_confidence", json!("low"));
        let (h, before, extracted) = restate(booking(), again, label);
        let old = h.show(before.id);
        let head = h.show(old.chain.head);
        assert_eq!(extracted.memories, vec![head.id], "{label}");
        assert_ne!(head.id, old.id, "{label} must retain an explicit date");
        assert_eq!(old.retracted_at.is_some(), label == "retracts");
        assert_eq!(head.window.valid_from, day("2026-10-02"));
        assert!(head.window.valid_until.is_none(), "not the old end");
        assert_eq!(head.window.window_confidence, "low");
    }
}

#[test]
fn a_related_claim_of_another_kind_never_replaces_an_open_task() {
    // A status event and a general preference on the same topic say
    // nothing about whether the renewal is done or was wrong. Whatever call
    // 2 labels them, the task stays the head of its chain and on the agenda.
    // Only `ends` or a corrected task replaces it. A repeat that matters more
    // than the task, which would otherwise refine it, leaves it alone too.
    let asked = "I asked the post office how to renew my passport";
    let online = "I prefer to renew my passport online";
    for (label, significance) in [
        ("retracts", "minor"),
        ("refines", "minor"),
        ("mentioned_again", "critical"),
    ] {
        let h = Harness::new();
        h.says("I need to renew my passport by 9 October.");
        let task = claim(PASSPORT, "task", "I need to renew my passport").at(
            "due_at",
            "2026-10-09T08:00",
            "minute",
        );
        let task = extract_alone(&h, reply(vec![task])).memories[0];
        h.advance(24);
        h.says_at(T2, &format!("{asked}. {online}."));
        let extracted = label_each(
            &h,
            reply(vec![
                claim("Tim asked the post office about renewing.", "event", asked)
                    .significance(significance),
                claim("Tim prefers to renew his passport online.", "fact", online)
                    .significance(significance),
            ]),
            &[(task, label), (task, label)],
        );

        assert_eq!(extracted.memories.len(), 2, "{label}: both claims are new");
        let view = h.show(task);
        assert!(view.retracted_at.is_none(), "{label}");
        assert_eq!(view.chain.head, task, "{label}");
        assert_eq!(view.chain.ended_by, None, "{label}");
        assert!(
            h.service.agenda("main").unwrap().listed().contains(&task),
            "{label}: the renewal is still outstanding"
        );
    }
}

#[test]
fn a_document_mention_is_an_access_unless_an_earlier_version_said_it() {
    // A document independently restating something is mentioned again
    // (CONTEXT.md). Documents share the turn number of the turn before them,
    // so two documents with no turn between them carry the same number, and
    // the second's access must still land rather than collide with the
    // first's created access.
    let h = Harness::new();
    h.doc("notes", "My bike is a Brompton.", date(2026, 9, 20));
    let bike = claim(BIKE, "fact", "My bike is a Brompton");
    let bike = extract_alone(&h, reply(vec![bike])).memories[0];
    h.advance(1);
    let diary = h.doc("diary", "Rode my bike, a Brompton.", date(2026, 9, 25));
    let mention = reply(vec![claim(BIKE, "fact", "my bike, a Brompton")]);
    let extracted = one_label(&h, mention, bike, "mentioned_again");
    assert!(extracted.memories.is_empty());
    let accesses = h.accesses(bike);
    assert_eq!(accesses.len(), 2, "{accesses:?}");
    assert_eq!(accesses[1].kind, "mentioned_again");
    assert_eq!(accesses[1].source, Some(diary.source));

    // A later version of the same document doesn't reinforce itself.
    h.advance(1);
    let text = "My bike is a Brompton. I ride it to work.";
    let edited = h.doc("notes", text, date(2026, 9, 27));
    assert_eq!(edited.chunks_queued, 1);
    let restated = reply(vec![claim(BIKE, "fact", "My bike is a Brompton")]);
    let extracted = one_label(&h, restated, bike, "mentioned_again");
    assert!(extracted.memories.is_empty());
    assert_eq!(h.accesses(bike), accesses);
}

#[test]
fn a_newer_retraction_or_refinement_supersedes_and_only_a_retraction_invalidates() {
    let moved = "moved my appointment to Friday 9 October";
    let reschedule = (
        said(DENTIST_8, "event").at("valid_from", "2026-10-08", "day"),
        "The dentist moved my appointment to Friday 9 October.",
        changes(claim(DENTIST_9, "event", moved).at("valid_from", "2026-10-09", "day")),
        "retracts",
    );
    let refinement = (
        said(JAPAN, "event")
            .at("valid_from", "2027", "year")
            .significance("notable"),
        "Remember this: it's Tokyo, in April 2027.",
        claim(TOKYO, "event", "it's Tokyo, in April 2027")
            .at("valid_from", "2027-04", "month")
            .significance("notable")
            .remember_this(),
        "refines",
    );
    for (old_claim, says, new_claim, label) in [reschedule, refinement] {
        let h = Harness::new();
        let old = h.fixture(old_claim);
        let before = (h.change(old), h.show(old).window, h.accesses(old));
        h.says(says);
        let new = one_label(&h, reply(vec![new_claim.clone()]), old, label).memories[0];

        assert_eq!(h.show(new).sentence, new_claim["content"].as_str().unwrap());
        // Both set superseded_by and leave the old window alone. Only a
        // retraction sets retracted_at, to the retracting claim's
        // observed_at: the refined memory wasn't wrong.
        let retracted_at = (label == "retracts").then(|| at(T1));
        let superseded_by = Some(new);
        let change = Change {
            retracted_at,
            superseded_by,
            ..before.0
        };
        assert_eq!(h.change(old), change, "{label}");
        assert_eq!(h.show(old).window, before.1, "{label}");
        let edit = match label {
            "retracts" => EDIT_RETRACTED,
            _ => EDIT_REFINED,
        };
        assert_eq!(h.edits(old), [edit]);
        // The new memory has its own created access; the old one gets none.
        assert_eq!(h.accesses(old), before.2, "{label}");
        assert_eq!(h.accesses(new).len(), 1, "{label}");
        let change = h.change(new);
        assert_eq!((change.superseded_by, change.ended_by), (None, None));
        // On refines, remember-this goes on the new memory.
        if label == "refines" {
            assert_eq!(h.significance(new), ("notable".into(), Some("kept".into())));
            assert_eq!(h.significance(old), ("notable".into(), None));
        }
    }
}

#[test]
fn an_ending_closes_the_neighbour_where_the_ender_starts_or_the_day_it_was_said() {
    // valid_until is the ending memory's start, with its precision. With no
    // start, it's the day the ender was said with low confidence, at day
    // precision like an event with no stated time. Ended isn't retracted or
    // superseded: the neighbour stays in history and inherits nothing. A tie
    // on observed_at goes to the later ingest, so the claim, ingested an hour
    // after its neighbour, is the newer and ends it.
    let moved = claim(
        MOVED,
        "event",
        "I moved out of Berlin and now live in Lisbon",
    );
    let no_coffee = claim(NO_COFFEE, "fact", "I don't drink coffee any more");
    for (neighbour, neighbour_at, ender, until, confidence) in [
        (
            BERLIN,
            EARLIER,
            moved.clone().at("valid_from", "2026-09-12", "day"),
            day("2026-09-12"),
            "high",
        ),
        (COFFEE, EARLIER, no_coffee, day("2026-10-01"), "low"),
        (BERLIN, T1, moved, day("2026-10-01"), "low"),
    ] {
        let h = Harness::new();
        let old = h.seed(neighbour_at, said(neighbour, "fact"), &[]);
        let before = (h.change(old), h.accesses(old));
        h.advance(1);
        h.says(ender["quote"].as_str().unwrap());
        let new = one_label(&h, reply(vec![changes(ender)]), old, "ends").memories[0];

        let change = Change {
            valid_until: until,
            window_confidence: confidence.into(),
            ended_by: Some(new),
            ..before.0
        };
        assert_eq!(h.change(old), change, "{neighbour} at {neighbour_at}");
        assert_eq!(h.edits(old), [EDIT_ENDED]);
        assert_eq!(h.accesses(old), before.1);
        let new = h.change(new);
        assert_eq!((new.ended_by, new.superseded_by), (None, None));
    }
}

#[test]
fn labels_code_rejects_leave_the_claim_new() {
    // Code rejects any label on a neighbour that's already ended, `denies`
    // on one retracted earlier in the same unit, and a label on a neighbour
    // it wasn't shown. A claim left with no labels is new.
    let h = Harness::new();
    let acme = h.fact(ACME);
    h.left_acme(NEXT_DAY, acme);
    let dentist = h.fixture(said(DENTIST_8, "event"));
    let tea = h.fact(TEA);
    let untouched = |h: &Harness| [acme, tea].map(|m| (h.change(m), h.accesses(m), h.edits(m)));
    let before = untouched(&h);

    h.says(
        "I work at Acme. I still work at Acme. I've never worked at Acme. \
         The dentist moved my appointment to 9 October. \
         Actually I have no dentist appointment. I like green tea.",
    );
    let moved = "The dentist moved my appointment to 9 October";
    let call1 = reply(vec![
        claim(ACME, "fact", "I work at Acme"),
        changes(claim(ACME_STILL, "fact", "I still work at Acme")),
        changes(claim(NEVER_ACME, "fact", "I've never worked at Acme")),
        changes(claim(DENTIST_9, "event", moved)),
        changes(claim(NO_DENTIST, "fact", "I have no dentist appointment")),
        claim(TEA, "fact", "I like green tea"),
    ]);
    let input = call2(&h, &call1).expect("call 2 runs");
    let n_acme = neighbour_handle(&input, acme);
    let n_dentist = neighbour_handle(&input, dentist);
    let labels = [
        (n_acme.clone(), "mentioned_again"),
        (n_acme.clone(), "ends"),
        (n_acme, "denies"),
        (n_dentist.clone(), "retracts"),
        (n_dentist, "denies"),
        ("n99".to_string(), "mentioned_again"),
    ];
    let call2 = labels
        .into_iter()
        .enumerate()
        .map(|(k, label)| labelled(&input.claims[k].handle, &[label]))
        .collect();
    let extracted = reconcile(&h, call1, call2_reply(call2), &[]);

    // Every claim is a new memory, and only the reschedule changed anything.
    assert_eq!(extracted.memories.len(), 6);
    assert_eq!(untouched(&h), before);
    assert_eq!(h.change(dentist).superseded_by, Some(extracted.memories[3]));
    assert_eq!(h.edits(dentist), [EDIT_RETRACTED]);
}

// Direction: an older claim.

#[test]
fn an_older_claim_labelled_ends_is_created_already_ended() {
    // Code, not the LLM, decides which is newer, so an old document can't
    // overrule a newer memory. It's created ended where the neighbour
    // starts or, when the neighbour has no start, on the day it was said
    // with low confidence.
    for (lisbon, said_at, until, confidence) in [
        (
            said(LISBON, "fact").at("valid_from", "2026-09-12", "day"),
            "2026-09-20T00:00:00Z".to_string(),
            day("2026-09-12"),
            "high",
        ),
        (
            said(LISBON, "fact"),
            local("2026-09-20T09:00").to_string(),
            day("2026-09-20"),
            "low",
        ),
    ] {
        let h = Harness::new();
        let lisbon = h.seed(&said_at, lisbon, &[]);
        let before = h.change(lisbon);
        h.doc("old-notes", "I live in Berlin.", date(2025, 3, 1));
        let berlin = reply(vec![claim(BERLIN, "fact", "I live in Berlin")]);
        let berlin = one_label(&h, berlin, lisbon, "ends").memories[0];
        let change = Change {
            valid_until: until,
            window_confidence: confidence.into(),
            retracted_at: None,
            superseded_by: None,
            ended_by: Some(lisbon),
        };
        assert_eq!(h.change(berlin), change, "said at {said_at}");
        assert_eq!(h.change(lisbon), before);
        assert!(h.edits(lisbon).is_empty());
    }
}

#[test]
fn an_older_claim_corrects_nothing_but_its_mention_still_counts() {
    // An old note retracting, refining or denying a newer memory creates
    // nothing and changes nothing, not even the task the denied filing
    // ended. Only its mention lands, even on an ended neighbour.
    const LATER: &str = "2026-09-20T00:00:00Z";
    const LATEST: &str = "2026-09-25T00:00:00Z";
    let h = Harness::new();
    let dentist = h.seed(LATER, said(DENTIST_9, "event"), &[]);
    let tokyo = h.seed(LATER, said(TOKYO, "event").significance("notable"), &[]);
    let task = h.seed(LATER, said(TAX_TASK, "task"), &[]);
    let filed = said(TAX_FILED, "event").at("valid_from", "2026-09-25", "day");
    let filed = h.seed(LATEST, filed, &[(task, "ends")]);
    let acme = h.seed(LATER, said(ACME, "fact"), &[]);
    h.left_acme(LATEST, acme);
    let state = |h: &Harness, m: Uuid| (h.change(m), h.accesses(m), h.edits(m));
    let corrected = [dentist, tokyo, task, filed];
    let before = corrected.map(|m| state(&h, m));
    let acme_before = state(&h, acme);

    let doc = h.doc(
        "old-notes",
        "Dentist on 8 October. Going to Japan in 2027. \
         I haven't filed the tax return. I work at Acme.",
        date(2026, 9, 10),
    );
    let extracted = label_each(
        &h,
        reply(vec![
            claim(DENTIST_8, "event", "Dentist on 8 October"),
            claim(JAPAN, "event", "Going to Japan in 2027"),
            claim(TAX_NOT_FILED, "fact", "I haven't filed the tax return"),
            claim(ACME, "fact", "I work at Acme"),
        ]),
        &[
            (dentist, "retracts"),
            (tokyo, "refines"),
            (filed, "denies"),
            (acme, "mentioned_again"),
        ],
    );

    assert!(extracted.memories.is_empty());
    assert_eq!(corrected.map(|m| state(&h, m)), before);
    let (change, mut accesses, edits) = state(&h, acme);
    assert_eq!((&change, &edits), (&acme_before.0, &acme_before.2));
    let mention = accesses.pop().unwrap();
    assert_eq!(accesses, acme_before.1);
    assert_eq!(mention.kind, "mentioned_again");
    assert_eq!(mention.source, Some(doc.source));
}

// Reopening.

/// A task and a state that `Tim filed the tax return.` (1 October) ended,
/// and a turn on 2 October saying `user` queued after them. Returns the
/// task, the state and the ending event.
fn filed_and_ended(h: &Harness, user: &str) -> (Uuid, Uuid, Uuid) {
    let task = h.fixture(said(TAX_TASK, "task").significance("notable"));
    let worry = h.fixture(said(TAX_WORRY, "state"));
    let filed = said(TAX_FILED, "event").at("valid_from", "2026-10-01", "day");
    let filed = h.seed(T1, filed, &[(task, "ends"), (worry, "ends")]);
    h.advance(24);
    h.says_at(T2, user);
    (task, worry, filed)
}

#[test]
fn a_successor_of_an_ender_repoints_the_end_and_a_denial_clears_it() {
    // A refinement still completed the task: what it ended is repointed and
    // keeps its end.
    let h = Harness::new();
    let (task, worry, filed) = filed_and_ended(&h, "I filed the tax return online.");
    let before = [task, worry].map(|m| h.change(m));
    let online = claim(TAX_FILED_ONLINE, "event", "I filed the tax return online").at(
        "valid_from",
        "2026-10-01",
        "day",
    );
    let online = one_label(&h, reply(vec![online]), filed, "refines").memories[0];
    assert_eq!(h.change(filed).superseded_by, Some(online));
    for (ended, before) in [task, worry].into_iter().zip(before) {
        let ended_by = Some(online);
        assert_eq!(h.change(ended), Change { ended_by, ..before });
        assert_eq!(h.edits(ended), [EDIT_ENDED, EDIT_END_REPOINTED]);
    }

    // So does a correction filed as another kind: it supersedes the memory
    // that ended the task, so it's a successor. A task correction closes at
    // the correction's observation time, not the successor's start.
    let h = Harness::new();
    let user = "Correction: I filed the tax return on 2 October, not the 1st.";
    let (task, _, filed) = filed_and_ended(&h, user);
    let before = h.change(task);
    let later = claim(
        TAX_FILED_LATER,
        "fact",
        "I filed the tax return on 2 October",
    )
    .at("valid_from", "2026-10-02", "day");
    let later = one_label(&h, reply(vec![changes(later)]), filed, "retracts").memories[0];
    assert_eq!(h.show(later).kind, "fact");
    assert_eq!(h.change(filed).superseded_by, Some(later));
    let closed = Change {
        valid_until: Some(WorldTime {
            at: at(T2),
            precision: TimePrecision::Minute,
        }),
        ended_by: Some(later),
        ..before
    };
    assert_eq!(h.change(task), closed);
    assert_eq!(h.edits(task), [EDIT_ENDED, EDIT_END_REPOINTED]);

    // "I haven't filed it after all" says the filing never happened. It's
    // retracted like any other retraction, and everything it ended is open
    // again.
    let h = Harness::new();
    let (task, worry, filed) = filed_and_ended(&h, "Actually I haven't filed the tax return yet.");
    let filed_before = h.change(filed);
    let denial = changes(claim(
        TAX_NOT_FILED,
        "fact",
        "I haven't filed the tax return yet",
    ));
    let denial = one_label(&h, reply(vec![denial]), filed, "denies").memories[0];
    let retracted = Change {
        retracted_at: Some(at(T2)),
        superseded_by: Some(denial),
        ..filed_before
    };
    assert_eq!(h.change(filed), retracted);
    assert_eq!(h.edits(filed).last().unwrap(), EDIT_RETRACTED);
    for reopened in [task, worry] {
        let change = h.change(reopened);
        assert_eq!((change.valid_until, change.ended_by), (None, None));
        assert_eq!(h.edits(reopened), [EDIT_ENDED, EDIT_END_CLEARED]);
    }
}

// The version 4 migration.

/// Puts the store back to schema version 3, as the version 3 binary would
/// have left it: `accesses` with version 1's `UNIQUE (memory_id, turn)` and
/// every row and id as they are, and one `migrations` row 0 to 3. Reopening
/// migrates it to version 4.
fn downgrade_accesses_to_v3(h: &Harness) {
    h.sql(
        "CREATE TABLE accesses_v3 (
           id        INTEGER PRIMARY KEY AUTOINCREMENT,
           bank_id   INTEGER NOT NULL REFERENCES banks(id),
           memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
           kind      TEXT NOT NULL
                       CHECK (kind IN ('created', 'used', 'mentioned_again', 'confirmed')),
           at        INTEGER NOT NULL,
           turn      INTEGER NOT NULL,
           source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
           UNIQUE (memory_id, turn)
         );
         INSERT INTO accesses_v3 SELECT id, bank_id, memory_id, kind, at, turn, source_id
           FROM accesses;
         DROP TABLE accesses;
         ALTER TABLE accesses_v3 RENAME TO accesses;
         CREATE INDEX accesses_memory_at ON accesses(memory_id, at);
         DELETE FROM migrations;
         INSERT INTO migrations (from_version, to_version, binary_version, started_at,
                                 completed_at)
           VALUES (0, 3, 'v3', 0, 0);
         PRAGMA user_version = 3;",
    );
}

fn insert_raw_access(
    h: &Harness,
    memory: Uuid,
    kind: &str,
    turn: i64,
    source: Option<i64>,
) -> rusqlite::Result<usize> {
    let at = h.now().as_microsecond();
    h.service.store().unwrap().connection().execute(
        "INSERT INTO accesses (bank_id, memory_id, kind, at, turn, source_id)
         SELECT bank_id, id, ?2, ?3, ?4, ?5 FROM memories WHERE uuid = ?1",
        (memory.to_string(), kind, at, turn, source),
    )
}

#[test]
fn a_populated_version_3_store_keeps_its_accesses_through_the_migration() {
    // A version 3 store holding a turn's mention, a document's mention and a
    // used access whose source is gone, each in a turn of its own as version
    // 3's key required.
    let h = Harness::new();
    let tea = h.fact(TEA);
    let acme = h.fact(ACME);
    let turn_source = h.says("I like green tea.");
    let doc_source = h.doc("notes", "I work at Acme.", date(2026, 9, 25)).source;
    let source_id =
        |uuid: Uuid| -> i64 { h.one("SELECT id FROM sources WHERE uuid = ?1", [uuid.to_string()]) };
    let (turn_id, doc_id) = (source_id(turn_source), source_id(doc_source));
    insert_raw_access(&h, tea, "mentioned_again", 12, Some(turn_id)).unwrap();
    insert_raw_access(&h, acme, "mentioned_again", 11, Some(doc_id)).unwrap();
    insert_raw_access(&h, tea, "used", 13, None).unwrap();
    insert_raw_access(&h, acme, "confirmed", 14, Some(turn_id)).unwrap();
    let before = (h.accesses(tea), h.accesses(acme));
    assert_eq!(
        (before.0.len(), before.1.len()),
        (3, 3),
        "created plus two each"
    );
    downgrade_accesses_to_v3(&h);

    let h = h.restart();
    let applied = h.service.store().unwrap().applied().map(|a| (a.from, a.to));
    let migrated = (3, asphodel_core::store::SCHEMA_VERSION);
    assert_eq!(applied, Some(migrated), "version 3 is migrated");
    assert_eq!((h.accesses(tea), h.accesses(acme)), before);
}

// Ordering and failures.

#[test]
fn call_2_and_the_commit_run_in_observed_at_order() {
    // The newer turn arrives first, but the bank's one worker reconciles the
    // older one first, so the newer one is reconciled against it.
    let h = Harness::new();
    h.ingest(&turn(
        "s2",
        "2026-10-01T06:40:00Z",
        "I live in Berlin.",
        "Ok.",
    ));
    let older = h.says("I live in Berlin.");
    assert_eq!(h.lease().source, older);
    let berlin = || reply(vec![claim(BERLIN, "fact", "I live in Berlin")]);
    let berlin_memory = extract_alone(&h, berlin()).memories[0];
    let input = call2(&h, &berlin()).expect("call 2 runs");
    assert_eq!(shown(&input), BTreeSet::from([berlin_memory]));
}

// Chunks in flight together (`[llm] concurrency`). Reconciliation is checked
// at commit against what the search saw, so two chunks stating the same fact
// make one memory.

/// Claims the head of `main`'s queue and prepares it with the LLM
/// answering `replies` in turn: call 1's, then call 2's if it runs.
fn prepared(h: &Harness, replies: Vec<Value>) -> Prepared {
    let claimed = h.service.next_extraction("main").unwrap().unwrap();
    let llm = FakeLlm::scripted(MODEL, replies);
    h.service
        .prepare_extraction(claimed.lease, &llm, &claimed.in_context)
        .unwrap()
}

fn committed(h: &Harness, prepared: Prepared) -> Extracted {
    match h.service.try_commit_extraction(prepared).unwrap() {
        Committed::Extracted(extracted) => extracted,
        Committed::Stale(stale) => panic!("{stale:?} went stale"),
    }
}

fn stale(h: &Harness, prepared: Prepared) -> Prepared {
    match h.service.try_commit_extraction(prepared).unwrap() {
        Committed::Stale(stale) => *stale,
        Committed::Extracted(extracted) => panic!("committed without a redo: {extracted:?}"),
    }
}

/// Reconciles a stale chunk again with call 2 answering `reply`, and
/// checks call 1 didn't run again.
fn redone(h: &Harness, stale: Prepared, reply: Value) -> Prepared {
    let llm = FakeLlm::scripted(MODEL, vec![reply]);
    let redone = h.service.redo_extraction(stale, &llm).unwrap();
    assert_eq!(llm.requests().len(), 1, "only call 2 runs again");
    redone
}

#[test]
fn a_memory_committed_since_the_search_at_the_floor_reconciles_the_chunk_again() {
    // Two chunks out at once state one fact, and neither saw the other's
    // memory when it searched. The second's commit finds it at the floor,
    // so the chunk reconciles again from the same call 1 and is a mention,
    // not a copy.
    let h = Harness::with_concurrency(2);
    h.says("I like green tea.");
    let second_source = h.says_at("2026-10-01T06:40:00Z", "I like green tea, still.");
    let call1 = reply(vec![claim(TEA, "fact", "I like green tea")]);
    let first = prepared(&h, vec![call1.clone()]);
    let second = prepared(&h, vec![call1]);
    assert!(second.call2_input().is_none(), "nothing near at its search");

    let tea = committed(&h, first).memories[0];
    let second = stale(&h, second);
    let input = h.service.redo_input(&second).unwrap();
    let input = input.expect("call 2 runs now");
    assert_eq!(shown(&input), BTreeSet::from([tea]));
    let second = redone(&h, second, each(&input, &[(tea, "mentioned_again")]));
    let extracted = committed(&h, second);

    assert!(extracted.memories.is_empty());
    assert_eq!(h.memories(), 1);
    assert_eq!(h.access_kinds(tea), ["created", "mentioned_again"]);
    let redo = h.chunk(second_source).error_count;
    assert_eq!(redo, 0, "a redo isn't a failed attempt");
}

#[test]
fn an_edit_since_the_search_on_a_shown_neighbour_reconciles_the_chunk_again() {
    // The first chunk's mention keeps the neighbour, an edit and no new
    // memory, so only the edit can send the second back.
    let h = Harness::with_concurrency(2);
    let tea = h.fact(TEA);
    h.says("Remember this: I like green tea.");
    h.says_at("2026-10-01T06:40:00Z", "I like green tea.");
    let mention = |input: &Call2Input| each(input, &[(tea, "mentioned_again")]);
    let kept = claim(TEA, "fact", "I like green tea").remember_this();
    let kept = reply(vec![kept]);
    let first_call2 = mention(&call2(&h, &kept).expect("call 2 runs"));
    let first = prepared(&h, vec![kept, first_call2]);
    // The second chunk is shown the same neighbour, under the same handles.
    let second_call2 = mention(first.call2_input().unwrap());
    let minor = reply(vec![claim(TEA, "fact", "I like green tea")]);
    let second = prepared(&h, vec![minor, second_call2.clone()]);
    assert_eq!(shown(second.call2_input().unwrap()), BTreeSet::from([tea]));

    committed(&h, first);
    assert_eq!(h.edits(tea), [EDIT_KEPT]);
    assert_eq!(h.memories(), 1, "the first chunk made no memory");
    let second = stale(&h, second);
    let second = redone(&h, second, second_call2);
    committed(&h, second);

    assert_eq!(h.memories(), 1);
    let kinds = ["created", "mentioned_again", "mentioned_again"];
    assert_eq!(h.access_kinds(tea), kinds);
}

#[test]
fn a_flagged_claim_reconciles_again_after_any_new_memory_and_an_unflagged_one_only_at_the_floor() {
    // A memory below the floor for a claim leaves an unflagged claim alone,
    // but a flagged claim pulls in its entities' tasks and states whatever
    // their similarity, so any new memory sends it back.
    let similarity = |a: &str, b: &str| -> f64 {
        let vectors = FakeEmbedder.embed(&[a, b]).unwrap();
        let pairs = vectors[0].iter().zip(&vectors[1]);
        pairs.map(|(x, y)| f64::from(*x) * f64::from(*y)).sum()
    };
    for (claim, memory) in [(WEATHER, CAT), (TAX_FILED, CAT), (TAX_FILED, WEATHER)] {
        let near = similarity(claim, memory);
        assert!(near < FLOOR, "{claim} is near {memory}");
    }
    let h = Harness::with_concurrency(3);
    h.says(CAT);
    h.says_at("2026-10-01T06:40:00Z", WEATHER);
    h.says_at("2026-10-01T06:50:00Z", TAX_FILED);
    let cat = prepared(&h, vec![reply(vec![said(CAT, "fact")])]);
    let weather = prepared(&h, vec![reply(vec![said(WEATHER, "event")])]);
    let filed = prepared(&h, vec![reply(vec![changes(said(TAX_FILED, "event"))])]);

    committed(&h, cat);
    committed(&h, weather);
    let filed = stale(&h, filed);
    let filed = match h.service.redo_input(&filed).unwrap() {
        Some(_) => redone(&h, filed, call2_reply(vec![])),
        None => {
            let llm = FakeLlm::scripted(MODEL, vec![]);
            h.service.redo_extraction(filed, &llm).unwrap()
        }
    };
    assert_eq!(committed(&h, filed).memories.len(), 1);
    assert_eq!(h.memories(), 3);
}

#[test]
fn chunks_commit_in_the_order_they_were_claimed() {
    // A later chunk's commit waits for every earlier one to commit, or to
    // be dropped without committing.
    let h = Harness::with_concurrency(3);
    h.says(WEATHER);
    let dropped = h.says_at("2026-10-01T06:40:00Z", BIKE);
    h.says_at("2026-10-01T06:50:00Z", CAT);
    let weather = prepared(&h, vec![reply(vec![said(WEATHER, "event")])]);
    let bike = prepared(&h, vec![reply(vec![])]);
    assert_eq!(bike.lease().source, dropped);
    let cat = prepared(&h, vec![reply(vec![said(CAT, "fact")])]);

    std::thread::scope(|scope| {
        let last = scope.spawn(|| committed(&h, cat));
        std::thread::sleep(Duration::from_millis(200));
        assert!(!last.is_finished(), "two chunks claimed before it are out");

        committed(&h, weather);
        std::thread::sleep(Duration::from_millis(200));
        assert!(!last.is_finished(), "one claimed before it is still out");

        drop(bike);
        let cat = last.join().unwrap().memories[0];
        assert_eq!(h.show(cat).sentence, CAT);
    });
    let again = h.lease();
    assert_eq!(again.source, dropped, "the dropped chunk is queued again");
    assert_eq!(again.error_count, 0);
}

/// Breaks BM25 for the whole store, so every neighbour search fails. Only a
/// test does this; in production it stands for any fault the search hits.
fn break_search(h: &Harness) {
    h.sql("DROP TABLE memories_fts");
}

#[test]
fn a_search_that_keeps_failing_is_counted_until_the_chunk_fails_and_writes_nothing() {
    // Each failed search is counted like any failed attempt, and a fault
    // that recurs fails the chunk rather than holding the bank's queue.
    let h = Harness::new();
    let tea = h.fact(TEA);
    let before = h.accesses(tea);
    let stuck = h.says("I like green tea.");
    let next = h.says_at("2026-10-01T06:40:00Z", "I like coffee.");
    break_search(&h);

    let mut attempts = 0;
    loop {
        let lease = h.lease();
        assert_eq!(lease.source, stuck, "attempt {attempts} retries in place");
        let call1 = reply(vec![claim(TEA, "fact", "I like green tea")]);
        let llm = FakeLlm::scripted(MODEL, vec![call1]);
        let error = h.service.extract_chunk(lease, &llm, &[]).unwrap_err();
        assert!(matches!(error, ExtractError::Search { .. }), "{error:?}");
        // The search runs between the calls, so call 2 never did.
        assert_eq!(llm.requests().len(), 1);
        attempts += 1;
        match error.failure() {
            Some(Failure::Retry { error_count }) => {
                assert_eq!(error_count, attempts);
                let chunk = h.chunk(stuck);
                assert_eq!(chunk.state, ChunkState::Queued);
                assert_eq!(chunk.error_kind.as_deref(), Some("search"));
            }
            Some(Failure::Failed) => break,
            None => panic!("a failed search is counted"),
        }
        assert!(attempts < 100, "the chunk never fails");
    }

    // No memory and no access written.
    assert_eq!(h.memories(), 1);
    assert_eq!(h.accesses(tea), before);
    let failed = h.service.failed_chunks("main").unwrap();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].source, stuck);
    assert_eq!(failed[0].error_kind, "search");
    assert_eq!(failed[0].error_count, attempts);
    assert_eq!(h.service.queue_depth("main").unwrap(), 1);
    assert_eq!(h.lease().source, next);
}

#[test]
fn call2_input_counts_nothing_when_it_fails() {
    // `call2_input` only previews call 2's input, so neither a reply that
    // doesn't fit call 1's schema nor a failed search counts against the
    // chunk or moves the queue.
    let h = Harness::new();
    h.fact(TEA);
    let source = h.says("I like green tea.");
    let unchanged = |h: &Harness| {
        let chunk = h.chunk(source);
        let attempts = (chunk.state, chunk.error_count, chunk.error_kind);
        assert_eq!(attempts, (ChunkState::Queued, 0, None));
        assert_eq!(h.service.queue_depth("main").unwrap(), 1);
    };

    let not_a_list = json!({"claims": "not a list"});
    let rejected = try_call2(&h, &not_a_list, &[]).unwrap_err();
    let is_rejected = matches!(rejected, ExtractError::Rejected { .. });
    assert!(is_rejected, "{rejected:?}");
    assert_eq!(rejected.failure(), None);
    unchanged(&h);

    break_search(&h);
    let call1 = reply(vec![claim(TEA, "fact", "I like green tea")]);
    let failed = try_call2(&h, &call1, &[]).unwrap_err();
    assert!(matches!(failed, ExtractError::Store(_)), "{failed:?}");
    assert_eq!(failed.failure(), None);
    unchanged(&h);
}

#[test]
fn a_failed_call_2_writes_nothing_and_its_retry_is_call_2_alone() {
    // Call 1's reply is saved before call 2 runs, so whatever stops call 2
    // (no reply, a reply that doesn't fit its schema, or an LLM that can't be
    // used at all) the retry resumes from it. Only the last isn't the chunk's
    // fault: nothing is counted and it stays at the head of the queue.
    let call1 = || reply(vec![claim(TEA, "fact", "I like green tea")]);
    for case in ["no reply", "invalid label", "login required"] {
        let h = Harness::new();
        let tea = h.fact(TEA);
        let before = h.accesses(tea);
        let source = h.says("I like green tea.");
        let input = call2(&h, &call1()).expect("call 2 runs");
        let mention = |label: &str| each(&input, &[(tea, label)]);
        let llm: Box<dyn LlmClient> = match case {
            "no reply" => Box::new(FakeLlm::scripted(MODEL, vec![call1()])),
            "invalid label" => {
                let replies = vec![call1(), mention("duplicates")];
                Box::new(FakeLlm::scripted(MODEL, replies))
            }
            _ => Box::new(ThenFails {
                first: Mutex::new(Some(call1())),
                error: || LlmError::LoginRequired,
            }),
        };
        let error = h
            .service
            .extract_chunk(h.lease(), llm.as_ref(), &[])
            .unwrap_err();
        let counted = case != "login required";
        let failure = counted.then_some(Failure::Retry { error_count: 1 });
        assert_eq!(error.failure(), failure, "{case}");
        assert_eq!(h.chunk(source).error_count, u32::from(counted), "{case}");
        assert_eq!(h.memories(), 1, "{case}");
        assert_eq!(h.accesses(tea), before, "{case}");
        assert!(h.edits(tea).is_empty(), "{case}");

        let retry = FakeLlm::scripted(MODEL, vec![mention("mentioned_again")]);
        let extracted = h.service.extract_chunk(h.lease(), &retry, &[]).unwrap();
        assert_eq!(retry.requests(), vec![call2_request(&input)], "{case}");
        assert!(extracted.memories.is_empty());
        assert_eq!(h.accesses(tea).len(), before.len() + 1, "{case}");
        // The saved reply goes when the chunk commits, so the claim text
        // can't outlive a purge or forget in the chunk row.
        let saved: Option<String> = h.one(
            "SELECT call1_output FROM chunks WHERE uuid = ?1",
            [extracted.chunk.to_string()],
        );
        assert_eq!(saved, None, "{case}");
    }
}
