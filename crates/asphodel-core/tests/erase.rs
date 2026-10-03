//! Forget, purge and the nightly sweep follow their deletion contracts.
//!
//! The API under test is `asphodel_core::erase`, `asphodel_core::sweep`
//! and the `Service` methods over them: `forget`, `erase_next`,
//! `run_sweeps`, `purge_plan`, `purge_ack` and the purge pause.
//!
//! Every service here runs on a `SimulatedClock` stopped at 20:00 on
//! Thursday 1 October 2026 in Auckland (UTC+13) unless a test moves it. The
//! next 04:00 there, when the nightly sweep runs, is 15:00 UTC the same day.
//! The tuning sets `clock.quiet_rate = 1.0`, so bank time is world time and
//! a trivial memory said once in 2021 is well past the nine months of bank
//! time after which a trivial memory mentioned once is purged.
//! Memories a test only needs present are inserted directly, as an earlier
//! extraction would have left them; the ones whose passages matter are
//! extracted from real turns with `FakeLlm`.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asphodel_core::config::PurgePause;
use asphodel_core::constants::CHUNK_RETRY_CAP;
use asphodel_core::ingest::{Document, Outcome, Turn};
use asphodel_core::mental_models::{
    Model, Outcome as RefreshOutcome, REFRESH_TEMPLATE, RefreshInput,
};
use asphodel_core::models::{Embedder, FakeEmbedder, FakeLlm, FakeReranker, Models};
use asphodel_core::retrieval::{Recall, RecallRequest};
use asphodel_core::store::bank::{BankIdentity, PROFILE_NAME};
use asphodel_core::store::{OpenOptions, Store, VectorIndex, micros, timestamp};
use asphodel_core::{Service, SimulatedClock, Tuning};
use jiff::civil::DateTime;
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use rusqlite::OptionalExtension;
use rusqlite::types::FromSql;
use serde_json::{Value, json};
use uuid::Uuid;

use asphodel_core::erase::{EraseReason, Forgotten};
use asphodel_core::operations::AuditList;
use asphodel_core::sweep::{PurgeError, Sweeps};

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";
const BANK: &str = "main";
const MODEL: &str = "fake-llm";

/// The next 04:00 in Auckland after [`START`].
const SWEEP: &str = "2026-10-01T15:00:00Z";

/// The first 04:00 in Auckland more than 90 days after [`START`]: 04:00 on
/// Thursday 31 December 2026.
const PAST_HORIZON: &str = "2026-12-30T15:00:00Z";

/// When fixture memories were said, unless a test says otherwise.
const EARLIER: &str = "2026-09-01T00:00:00Z";

/// Long enough ago that a trivial memory said once then is purged.
const LONG_AGO: &str = "2021-01-01T00:00:00Z";

/// The fixture turn every inserted memory rests on, and its passage.
const FIXTURE: &str = "Fixtures.";

const BERLIN: &str = "Tim lives in Berlin.";
const MOVED_OUT: &str = "Tim moved out of Berlin.";
const MOVED: &str = "Tim moved out of Berlin and now lives in Lisbon.";
const MAYA: &str = "Tim's daughter is called Maya.";
const MIA: &str = "Tim's daughter is called Mia.";
const TEA: &str = "Tim likes green tea.";
const CONCERT: &str = "Tim is going to a concert on 12 December 2026.";
const DENTIST_8: &str = "Tim's dentist appointment is on 8 March 2021.";
const DENTIST_9: &str = "Tim's dentist appointment is on 9 December 2026.";
const PASSPORT: &str = "Tim needs to renew his passport.";
const TAX: &str = "Tim needs to file the tax return.";
const BIKE: &str = "Tim needs to fix the bike.";

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

fn days(n: i64) -> SignedDuration {
    SignedDuration::from_hours(24 * n)
}

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-erase-{}-{}",
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
    Uuid::from_u128((0xe5_u128 << 120) | u128::from(NEXT.fetch_add(1, Ordering::Relaxed)))
}

/// A memory to insert. `Default` is a notable fact said at [`EARLIER`] with
/// its `created` access then, resting on the fixture passage.
#[derive(Clone)]
struct Memory {
    content: &'static str,
    kind: &'static str,
    significance: &'static str,
    observed_at: Timestamp,
    valid_from: Option<Timestamp>,
    due_at: Option<Timestamp>,
}

impl Default for Memory {
    fn default() -> Self {
        Self {
            content: "",
            kind: "fact",
            significance: "notable",
            observed_at: at(EARLIER),
            valid_from: None,
            due_at: None,
        }
    }
}

fn fact(content: &'static str) -> Memory {
    Memory {
        content,
        ..Memory::default()
    }
}

/// A point event on the local day `when`.
fn event(content: &'static str, when: &str) -> Memory {
    Memory {
        content,
        kind: "event",
        valid_from: Some(local(when)),
        ..Memory::default()
    }
}

/// An open task due on the local day `when`, or undated.
fn task(content: &'static str, due: Option<&str>) -> Memory {
    Memory {
        content,
        kind: "task",
        due_at: due.map(local),
        ..Memory::default()
    }
}

/// A trivial memory said once, long enough ago to be purged.
fn faded(memory: Memory) -> Memory {
    Memory {
        significance: "trivial",
        observed_at: at(LONG_AGO),
        ..memory
    }
}

struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    tuning: Tuning,
    /// The fixture turn and its chunk.
    source: Uuid,
    chunk: i64,
    _dir: TestDir,
}

impl Harness {
    fn new() -> Self {
        let dir = TestDir::new();
        let clock = Arc::new(SimulatedClock::new(at(START)));
        let mut harness = Self::open(dir, clock, "");
        harness
            .service
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
        harness.source = harness
            .service
            .ingest_turn(BANK, &turn("fixtures", at(EARLIER), FIXTURE))
            .unwrap()
            .source;
        harness.chunk = harness.one("SELECT id FROM chunks", []);
        harness.extracted_with_nothing(harness.source);
        harness
    }

    /// Opens the store in `dir` with the tuning the fakes need, `extra`
    /// TOML on top, and the purge state the store gives it, as `serve` does.
    fn open(dir: TestDir, clock: Arc<SimulatedClock>, extra: &str) -> Self {
        let tuning = Tuning::from_toml(&format!(
            "[clock]\nquiet_rate = 1.0\n\
             [injection.reranker_floors]\n\"{}\" = 1.0\n\
             [reconcile.embedding_floors]\n\"{}\" = 0.5\n{extra}",
            FakeReranker::MODEL_ID,
            FakeEmbedder::MODEL_ID,
        ))
        .unwrap();
        let store = Store::open(&dir.0, OpenOptions::default(), clock.clone()).unwrap();
        let pause = store
            .check_fingerprint(&tuning.deletion_fingerprint())
            .unwrap();
        let service =
            Service::with_models(clock.clone(), store, tuning.clone(), Models::fake()).unwrap();
        let service = service.with_purge_pause(pause);
        Self {
            service,
            clock,
            tuning,
            source: Uuid::nil(),
            chunk: 0,
            _dir: dir,
        }
    }

    /// The daemon restarting with `extra` tuning: the store stays.
    fn restart_with(self, extra: &str) -> Self {
        let Self {
            service,
            clock,
            source,
            chunk,
            _dir,
            ..
        } = self;
        drop(service);
        Self {
            source,
            chunk,
            ..Self::open(_dir, clock, extra)
        }
    }

    fn now(&self) -> Timestamp {
        self.service.now()
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

    fn optional<T: FromSql, P: rusqlite::Params>(&self, sql: &str, params: P) -> Option<T> {
        self.service
            .store()
            .unwrap()
            .connection()
            .query_row(sql, params, |row| row.get(0))
            .optional()
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
        let store = self.service.store().unwrap();
        let conn = store.connection();
        conn.execute(
            "INSERT INTO memories (uuid, bank_id, content, kind, significance, chunk_id,
                                   source_start, source_end, observed_at, valid_from,
                                   valid_from_precision, window_confidence, due_at,
                                   due_at_precision, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, ?7, ?8, ?9, ?10, 'high', ?11, ?12, ?13, ?13)",
            rusqlite::params![
                uuid.to_string(),
                bank_id,
                memory.content,
                memory.kind,
                memory.significance,
                self.chunk,
                FIXTURE.len() as i64,
                micros(memory.observed_at),
                memory.valid_from.map(micros),
                memory.valid_from.map(|_| "day"),
                memory.due_at.map(micros),
                memory.due_at.map(|_| "day"),
                micros(self.now()),
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

    /// `old` replaced by `new`, retracted or refined, as reconciliation
    /// leaves it.
    fn supersede(&self, old: Uuid, new: Uuid, retracted: bool) {
        self.execute(
            "UPDATE memories SET superseded_by = ?2, invalidated_at = ?3 WHERE id = ?1",
            (
                self.rowid(old),
                self.rowid(new),
                retracted.then_some(micros(self.now())),
            ),
        );
    }

    /// `memory` ended by `by` today, as reconciliation leaves it.
    fn end(&self, memory: Uuid, by: Uuid) {
        self.execute(
            "UPDATE memories SET valid_until = ?2, valid_until_precision = 'day', ended_by = ?3
             WHERE id = ?1",
            (self.rowid(memory), micros(self.now()), self.rowid(by)),
        );
    }

    fn ended_by(&self, memory: Uuid) -> Option<Uuid> {
        self.optional::<String, _>(
            "SELECT e.uuid FROM memories m JOIN memories e ON e.id = m.ended_by
             WHERE m.uuid = ?1",
            [memory.to_string()],
        )
        .map(|uuid| uuid.parse().unwrap())
    }

    fn valid_until(&self, memory: Uuid) -> Option<i64> {
        self.one(
            "SELECT valid_until FROM memories WHERE uuid = ?1",
            [memory.to_string()],
        )
    }

    fn superseded_by(&self, memory: Uuid) -> Option<Uuid> {
        self.optional::<String, _>(
            "SELECT s.uuid FROM memories m JOIN memories s ON s.id = m.superseded_by
             WHERE m.uuid = ?1",
            [memory.to_string()],
        )
        .map(|uuid| uuid.parse().unwrap())
    }

    /// How many of `memories` still have a row.
    fn rows(&self, memories: &[Uuid]) -> usize {
        memories
            .iter()
            .filter(|memory| {
                self.optional::<i64, _>(
                    "SELECT id FROM memories WHERE uuid = ?1",
                    [memory.to_string()],
                )
                .is_some()
            })
            .count()
    }

    /// The owner says `text` in `session` a minute ago. Returns the source.
    fn ingest(&self, session: &str, text: &str) -> Uuid {
        let ingested = self
            .service
            .ingest_turn(BANK, &turn(session, self.now() - minutes(1), text))
            .unwrap();
        assert_eq!(ingested.outcome, Outcome::Stored);
        ingested.source
    }

    /// `source`'s chunk taken off the queue as extracted with no claims, so
    /// no memory rests on it.
    fn extracted_with_nothing(&self, source: Uuid) {
        let chunk = self.chunk_of(source);
        self.execute("DELETE FROM extraction_queue WHERE chunk_id = ?1", [chunk]);
        self.execute(
            "UPDATE chunks SET extracted_at = ?2 WHERE id = ?1",
            (chunk, micros(self.now())),
        );
    }

    fn chunk_of(&self, source: Uuid) -> i64 {
        self.one(
            "SELECT c.id FROM chunks c JOIN sources s ON s.id = c.source_id WHERE s.uuid = ?1",
            [source.to_string()],
        )
    }

    /// The owner says the claim's quote in session `chat`, and the turn is
    /// extracted with call 1 finding `claim` and call 2, if it runs,
    /// labelling nothing. Returns the memory and its source.
    fn says(&self, claim: Value) -> (Uuid, Uuid) {
        let source = self.ingest("chat", claim["quote"].as_str().unwrap());
        let llm = FakeLlm::scripted(
            MODEL,
            vec![
                json!({"claims": [claim], "used_injected_ids": []}),
                json!({"claims": []}),
            ],
        );
        let extracted = self.service.extract_next(BANK, &llm).unwrap();
        (extracted.expect("the turn was queued").memories[0], source)
    }

    /// As [`Harness::says`], with call 2 labelling the claim `label` on
    /// `neighbour`.
    fn says_changing(&self, claim: Value, neighbour: Uuid, label: &str) -> (Uuid, Uuid) {
        let source = self.ingest("chat", claim["quote"].as_str().unwrap());
        (self.extract_labelled(claim, neighbour, label), source)
    }

    /// Extracts the head of the queue with call 1 finding `claim` and call
    /// 2 labelling it `label` on `neighbour`. Returns the new memory.
    fn extract_labelled(&self, claim: Value, neighbour: Uuid, label: &str) -> Uuid {
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

    fn message_at(&self, source: Uuid) -> Timestamp {
        timestamp(self.one(
            "SELECT message_at FROM sources WHERE uuid = ?1",
            [source.to_string()],
        ))
    }

    fn source_text(&self, source: Uuid) -> String {
        self.one::<Option<String>, _>(
            "SELECT text FROM sources WHERE uuid = ?1",
            [source.to_string()],
        )
        .unwrap_or_default()
    }

    fn chunk_text(&self, source: Uuid) -> String {
        self.one::<Option<String>, _>(
            "SELECT text FROM chunks WHERE id = ?1",
            [self.chunk_of(source)],
        )
        .unwrap_or_default()
    }

    /// The stored in-context set of a queued turn.
    fn turn_in_context(&self, source: Uuid) -> Option<Vec<Uuid>> {
        self.optional::<String, _>(
            "SELECT t.memories FROM turn_in_context t JOIN sources s ON s.id = t.source_id
             WHERE s.uuid = ?1",
            [source.to_string()],
        )
        .map(|json| serde_json::from_str(&json).unwrap())
    }

    fn recall(&self, request: RecallRequest) -> Recall {
        self.service.recall(BANK, &request).unwrap()
    }

    fn in_context(&self, session: &str) -> Vec<Uuid> {
        self.service.in_context(BANK, session).unwrap()
    }

    fn profile(&self) -> Model {
        self.service
            .list_models(BANK)
            .unwrap()
            .into_iter()
            .find(|model| model.name == PROFILE_NAME)
            .unwrap()
    }

    /// Forces a refresh of the profile whose reply adds one entry citing
    /// `cites`.
    fn profile_entry_citing(&self, text: &str, cites: &[Uuid]) {
        let input: RefreshInput = self.service.refresh_input(BANK, PROFILE_NAME).unwrap();
        let handles: Vec<String> = cites.iter().map(|m| handle(&input, *m)).collect();
        let llm = FakeLlm::scripted(
            MODEL,
            vec![json!({"operations": [{"op": "add", "text": text, "cites": handles}]})],
        );
        let outcome = self
            .service
            .refresh_model(BANK, PROFILE_NAME, &llm, true)
            .unwrap();
        assert!(matches!(outcome, RefreshOutcome::Applied(_)), "{outcome:?}");
        assert_eq!(self.profile().entries.len(), 1);
    }

    /// An entry in the profile citing `memory`, as an earlier refresh left
    /// it while the memory was still strong.
    fn cite_in_profile(&self, text: &str, memory: Uuid) {
        let now = micros(self.now());
        self.execute(
            "INSERT INTO mental_model_entries (uuid, model_id, position, text, created_at,
                                              updated_at)
             SELECT ?1, id, 0, ?2, ?3, ?3 FROM mental_models WHERE bank_id = ?4 AND name = ?5",
            (
                next_uuid().to_string(),
                text,
                now,
                self.bank_id(),
                PROFILE_NAME,
            ),
        );
        self.execute(
            "INSERT INTO mental_model_citations (entry_id, memory_id)
             SELECT max(id), ?1 FROM mental_model_entries",
            [self.rowid(memory)],
        );
        assert_eq!(self.profile().entries.len(), 1);
    }

    /// An entity of the bank, by rowid.
    fn entity(&self, name: &str, merged_into: Option<i64>) -> i64 {
        let now = micros(self.now());
        self.execute(
            "INSERT INTO entities (uuid, bank_id, name, kind, merged_into, created_at, updated_at)
             VALUES (?1, ?2, ?3, 'place', ?4, ?5, ?5)",
            (
                next_uuid().to_string(),
                self.bank_id(),
                name,
                merged_into,
                now,
            ),
        );
        let id: i64 = self.one("SELECT max(id) FROM entities", []);
        self.execute(
            "INSERT INTO entity_aliases (bank_id, entity_id, alias, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            (self.bank_id(), id, name, now),
        );
        id
    }

    fn user_entity(&self) -> i64 {
        self.one(
            "SELECT id FROM entities WHERE bank_id = ?1 AND seeded = 'user'",
            [self.bank_id()],
        )
    }

    fn link(&self, memory: Uuid, entity: i64) {
        self.execute(
            "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?1, ?2)",
            (self.rowid(memory), entity),
        );
    }

    fn entity_exists(&self, entity: i64) -> bool {
        self.optional::<i64, _>("SELECT id FROM entities WHERE id = ?1", [entity])
            .is_some()
    }

    fn aliases_of(&self, entity: i64) -> i64 {
        self.one(
            "SELECT COUNT(*) FROM entity_aliases WHERE entity_id = ?1",
            [entity],
        )
    }

    /// A recall row at `when` with `memory` among its results, as
    /// retrieval writes it.
    fn recall_row(&self, when: Timestamp, memory: Uuid) -> Uuid {
        let uuid = next_uuid();
        self.execute(
            "INSERT INTO recalls (uuid, bank_id, kind, session_id, turn, query, latency_ms, at)
             VALUES (?1, ?2, 'prefetch', 'chat', 0, 'what tea does Tim like?', 12, ?3)",
            (uuid.to_string(), self.bank_id(), micros(when)),
        );
        self.execute(
            "INSERT INTO recall_results (recall_id, memory_id, rank, injected)
             SELECT id, ?2, 0, 1 FROM recalls WHERE uuid = ?1",
            (uuid.to_string(), self.rowid(memory)),
        );
        uuid
    }

    /// The recall row with key `recall`, if it's still there: its query,
    /// when it was swept, and how many results it still names.
    fn recall_tombstone(&self, recall: Uuid) -> Option<(Option<String>, Option<i64>, i64)> {
        self.service
            .store()
            .unwrap()
            .connection()
            .query_row(
                "SELECT r.query, r.swept_at,
                        (SELECT COUNT(*) FROM recall_results WHERE recall_id = r.id)
                 FROM recalls r WHERE r.uuid = ?1",
                [recall.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()
            .unwrap()
    }

    /// Runs the sweeps due now, then any erase they queued.
    fn sweep(&self) -> Sweeps {
        let sweeps = self.service.run_sweeps().unwrap();
        while self.service.erase_next(BANK).unwrap().is_some() {}
        sweeps
    }

    fn forget(&self, memories: &[Uuid]) -> Forgotten {
        let ids: Vec<String> = memories.iter().map(Uuid::to_string).collect();
        self.service.forget(BANK, &ids).unwrap()
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
fn notable(content: &str) -> Value {
    json!({
        "content": content,
        "kind": "fact",
        "quote": content,
        "significance": "notable",
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

/// A claim flagged as changing something, so its neighbours aren't held to
/// the floor.
fn changes(mut claim: Value) -> Value {
    claim["changes_something"] = json!(true);
    claim
}

/// `claim` quoting `quote` rather than its sentence.
fn quoting(mut claim: Value, quote: &str) -> Value {
    claim["quote"] = json!(quote);
    claim
}

fn query(text: &str) -> RecallRequest {
    RecallRequest {
        query: text.into(),
        ..RecallRequest::default()
    }
}

fn ids(recall: &Recall) -> Vec<Uuid> {
    recall.results.iter().map(|r| r.id).collect()
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

// Forget

#[test]
fn forget_hides_at_once_and_erases_behind_a_queued_chunk() {
    let h = Harness::new();
    let (maya, said) = h.says(notable(MAYA));
    h.profile_entry_citing("Tim has a daughter called Maya.", &[maya]);
    let recalled = h.recall(RecallRequest {
        session_id: Some("chat".into()),
        ..query(MAYA)
    });
    assert!(ids(&recalled).contains(&maya));
    // The correction is queued with Maya in context.
    let correction = h.ingest("chat", "My daughter is called Mia, not Maya.");
    assert!(h.turn_in_context(correction).unwrap().contains(&maya));

    let forgotten = h
        .service
        .forget(BANK, &[maya.to_string(), "not-a-memory".into()])
        .unwrap();
    assert_eq!(forgotten.forgotten, vec![maya]);
    assert_eq!(forgotten.unknown, vec!["not-a-memory".to_string()]);

    // Everything that can be undone happens at once.
    assert!(!ids(&h.recall(query(MAYA))).contains(&maya));
    assert!(h.profile().entries.is_empty());
    assert!(!h.in_context("chat").contains(&maya));
    assert!(
        !h.turn_in_context(correction)
            .unwrap_or_default()
            .contains(&maya)
    );
    assert_eq!(
        h.optional::<i64, _>(
            "SELECT id FROM recalls WHERE uuid = ?1",
            [recalled.recall_id.to_string()]
        ),
        None,
        "the recall row naming Maya is deleted"
    );

    // The rows wait behind the chunk queued before the forget.
    assert_eq!(h.rows(&[maya]), 1);
    assert_eq!(h.service.erase_next(BANK).unwrap(), None);

    // That chunk reconciles against the hidden memory, so the correction
    // joins its chain, hidden from the moment it's committed.
    let mia = h.extract_labelled(
        quoting(changes(notable(MIA)), "My daughter is called Mia, not Maya"),
        maya,
        "retracts",
    );
    assert_eq!(h.superseded_by(maya), Some(mia));
    assert!(!ids(&h.recall(query(MIA))).contains(&mia));

    let erased = h
        .service
        .erase_next(BANK)
        .unwrap()
        .expect("the erase is at the head of the queue");
    assert_eq!(erased.reason, EraseReason::Forget);
    assert_eq!(erased.memories, BTreeSet::from([maya, mia]));
    assert_eq!(h.rows(&[maya, mia]), 0);

    // Both passages are redacted, and the content-hash tombstone keeps the
    // correction from being ingested again.
    for (source, word) in [(said, "Maya"), (correction, "Mia")] {
        assert!(!h.source_text(source).contains(word), "{source}");
        assert!(!h.chunk_text(source).contains(word), "{source}");
    }
    let again = h
        .service
        .ingest_turn(
            BANK,
            &turn(
                "chat",
                h.message_at(correction),
                "My daughter is called Mia, not Maya.",
            ),
        )
        .unwrap();
    assert_eq!(again.outcome, Outcome::Duplicate);
    assert_eq!(h.service.queue_depth(BANK).unwrap(), 0);
}

#[test]
fn a_chunk_queued_after_a_forget_isnt_claimed_until_the_erase_has_run() {
    // The erase waits behind the chunks queued before the forget. With a
    // pool of leases, a chunk queued after it could be out at the same
    // time and reconcile against the hidden memory; committed after the
    // erase, its labels would be dropped as vanished and the forgotten
    // content created again. So the erase is a barrier to claims.
    let h = Harness::new().restart_with("[llm]\nconcurrency = 2\n");
    let maya = h.insert(fact(MAYA));
    let before = h.ingest("chat", "My daughter Maya likes tea.");
    h.forget(&[maya]);
    h.set(h.now() + minutes(1));
    let after = h.ingest("chat", "My daughter is called Maya.");

    let first = h
        .service
        .claim_chunk(BANK)
        .unwrap()
        .expect("the chunk queued before the forget");
    assert_eq!(first.source, before);
    assert!(
        h.service.claim_chunk(BANK).unwrap().is_none(),
        "the chunk queued after the forget waits for the erase"
    );
    h.service.complete_chunk(first).unwrap();
    assert!(
        h.service.claim_chunk(BANK).unwrap().is_none(),
        "a due erase still holds it back until it has run"
    );

    h.service
        .erase_next(BANK)
        .unwrap()
        .expect("the erase is at the head of the queue");
    let next = h
        .service
        .claim_chunk(BANK)
        .unwrap()
        .expect("the erase has run");
    assert_eq!(next.source, after);
}

#[test]
fn at_concurrency_one_a_turn_after_a_forget_waits_for_the_erase_behind_an_earlier_document() {
    // Turns go ahead of documents, so the turn queued after the forget
    // sorts first. It still waits: it would reconcile against the hidden
    // memory and could commit after the erase.
    let h = Harness::new();
    let maya = h.insert(fact(MAYA));
    let notes = h
        .service
        .ingest_document(
            BANK,
            &Document {
                document_id: "notes.md".into(),
                text: "# Notes\n\nMaya likes tea.\n".into(),
                reference_date: h.now().to_zoned(TimeZone::UTC).date(),
                reference_date_exact: true,
                timezone: Some(TZ.into()),
            },
        )
        .unwrap()
        .source;
    h.forget(&[maya]);
    let after = h.ingest("chat", "My daughter is called Maya.");

    let first = h
        .service
        .claim_chunk(BANK)
        .unwrap()
        .expect("the document's chunk");
    assert_eq!(first.source, notes);
    h.service.complete_chunk(first).unwrap();
    assert!(h.service.claim_chunk(BANK).unwrap().is_none());
    h.service
        .erase_next(BANK)
        .unwrap()
        .expect("the erase is at the head of the queue");
    assert_eq!(h.service.claim_chunk(BANK).unwrap().unwrap().source, after);
}

#[test]
fn forget_erases_the_whole_chain_and_clears_what_points_into_it() {
    // forget takes every memory along
    // `superseded_by`, whichever version it names. `ended_by` isn't a chain
    // link: Berlin stays, with its end and without the pointer. Orphan
    // entities go, except seeded ones and merge tombstones.
    let h = Harness::new();
    let berlin = h.insert(fact(BERLIN));
    let (moved_out, first) = h.says_changing(changes(notable(MOVED_OUT)), berlin, "ends");
    assert_eq!(h.ended_by(berlin), Some(moved_out));
    let until = h.valid_until(berlin);
    assert!(until.is_some());
    let (moved, second) = h.says_changing(changes(notable(MOVED)), moved_out, "refines");
    assert_eq!(h.superseded_by(moved_out), Some(moved));

    let berlin_place = h.entity("Berlin", None);
    let lisbon_place = h.entity("Lisbon", None);
    let merged = h.entity("Lisboa", Some(berlin_place));
    let user = h.user_entity();
    for entity in [berlin_place, lisbon_place, merged, user] {
        h.link(moved, entity);
    }
    h.link(berlin, berlin_place);

    let forgotten = h.forget(&[moved_out]);
    assert_eq!(
        forgotten.forgotten.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([moved_out, moved])
    );
    assert!(forgotten.unknown.is_empty());
    let erased = h
        .service
        .erase_next(BANK)
        .unwrap()
        .expect("nothing was queued before the forget");
    assert_eq!(erased.memories, BTreeSet::from([moved_out, moved]));

    assert_eq!(h.rows(&[moved_out, moved]), 0);
    assert_eq!(h.rows(&[berlin]), 1);
    assert_eq!(h.ended_by(berlin), None);
    assert_eq!(h.valid_until(berlin), until);

    assert!(!h.entity_exists(lisbon_place));
    assert_eq!(h.aliases_of(lisbon_place), 0);
    for kept in [berlin_place, merged, user] {
        assert!(h.entity_exists(kept), "entity {kept}");
    }

    for source in [first, second] {
        assert!(!h.source_text(source).contains("Berlin"), "{source}");
    }
    let audit: Vec<String> = {
        let store = h.service.store().unwrap();
        let conn = store.connection();
        let mut statement = conn
            .prepare("SELECT details FROM edits WHERE kind = 'forgotten'")
            .unwrap();
        statement
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    assert_eq!(audit.len(), 1, "one audit row for the erase");
    assert!(!audit[0].contains("Berlin") && !audit[0].contains("Lisbon"));
}

// Purge in the nightly sweep

#[test]
fn the_sweep_purges_a_faded_chain_at_four_bank_local_without_redacting() {
    let h = Harness::new();
    let maya = h.insert(faded(fact(MAYA)));
    let mia = h.insert(faded(fact(MIA)));
    h.supersede(maya, mia, true);
    let berlin = h.insert(fact(BERLIN));
    h.end(berlin, mia);
    let tea = h.insert(fact(TEA));

    h.set(at(SWEEP) - minutes(1));
    let early = h.sweep();
    assert!(early.ran.is_empty());
    assert_eq!(early.next_due, Some(at(SWEEP)));
    assert_eq!(h.rows(&[maya, mia]), 2);

    h.set(at(SWEEP));
    let swept = h.sweep();
    assert_eq!(swept.ran.len(), 1);
    let run = &swept.ran[0];
    assert_eq!(run.bank, BANK);
    assert_eq!(run.purged_memories, 2);
    assert_eq!(run.fingerprint, h.tuning.deletion_fingerprint());
    assert_eq!(run.delta, Some(1.0));
    assert_eq!(
        h.one::<i64, _>("SELECT COUNT(*) FROM sweep_runs", []),
        1,
        "one run row per bank"
    );
    assert_eq!(h.rows(&[maya, mia]), 0);
    assert_eq!(h.rows(&[berlin, tea]), 2);
    assert_eq!(h.ended_by(berlin), None);

    // One `purged` row: ids, chunks and spans, never content.
    let details: String = h.one("SELECT details FROM edits WHERE kind = 'purged'", []);
    assert!(!details.contains("Maya") && !details.contains("Mia"));
    let details: Value = serde_json::from_str(&details).unwrap();
    let chunk: String = h.one("SELECT uuid FROM chunks WHERE id = ?1", [h.chunk]);
    let mut purged = BTreeSet::new();
    for memory in details["memories"].as_array().unwrap() {
        purged.insert(memory["memory"].as_str().unwrap().parse::<Uuid>().unwrap());
        assert_eq!(memory["chunk"], json!(chunk));
        assert_eq!(memory["start"], json!(0));
        assert_eq!(memory["end"], json!(FIXTURE.len()));
    }
    assert_eq!(purged, BTreeSet::from([maya, mia]));

    // Purge never redacts: the source still holds the passage.
    assert_eq!(h.source_text(h.source), FIXTURE);

    // The audit lists and status read it back: ids and counts, never
    // content, and only the bank's own.
    h.service
        .ensure_bank_with_models(
            "other",
            &BankIdentity {
                timezone: Some(TZ.into()),
                ..BankIdentity::default()
            },
        )
        .unwrap();
    let list = |bank: &str, list: AuditList, limit: Option<usize>| {
        serde_json::to_value(h.service.audit(bank, list, limit).unwrap()).unwrap()
    };
    let purges = list(BANK, AuditList::Purges, None);
    let rows = purges["purges"].as_array().unwrap();
    assert_eq!(rows.len(), 1, "{purges}");
    let listed: BTreeSet<Uuid> = rows[0]["memories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_str().unwrap().parse().unwrap())
        .collect();
    assert_eq!(listed, BTreeSet::from([maya, mia]));
    assert!(!purges.to_string().contains("Maya") && !purges.to_string().contains("Mia"));
    assert_eq!(list("other", AuditList::Purges, None)["purges"], json!([]));
    let last = h.service.status().unwrap().last_sweep.unwrap();
    assert_eq!((last.bank.as_str(), last.purged_memories), (BANK, 2));

    // A second night: newest first, and a limit keeps the newest.
    h.set(at(SWEEP) + days(1));
    h.sweep();
    let sweeps = list(BANK, AuditList::Sweeps, None);
    let purged: Vec<&Value> = sweeps["sweeps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|run| &run["purged_memories"])
        .collect();
    assert_eq!(purged, [&json!(0), &json!(2)], "{sweeps}");
    let newest = list(BANK, AuditList::Sweeps, Some(1));
    assert_eq!(newest["sweeps"].as_array().unwrap().len(), 1);
    assert_eq!(newest["sweeps"][0]["purged_memories"], 0);
    let other = list("other", AuditList::Sweeps, None);
    assert!(
        other["sweeps"]
            .as_array()
            .unwrap()
            .iter()
            .all(|run| run["bank"] == "other"),
        "{other}"
    );
}

#[test]
fn a_model_citing_a_purged_memory_refreshes_once_that_night() {
    // the sweep purges before the night's refresh, so a
    // model whose entry cited a purged memory refreshes once, without it.
    let h = Harness::new();
    h.insert(fact(TEA));
    let maya = h.insert(faded(fact(MAYA)));
    h.cite_in_profile("Tim has a daughter called Maya.", maya);

    h.set(at(SWEEP));
    h.sweep();
    assert_eq!(h.rows(&[maya]), 0);
    assert!(h.profile().entries.is_empty());
    let llm = FakeLlm::scripted(MODEL, vec![json!({"operations": []}); 3]);
    h.service.run_refreshes(&llm).unwrap();
    for later in [
        minutes(31),
        SignedDuration::from_hours(12),
        days(1) - minutes(1),
    ] {
        h.set(at(SWEEP) + later);
        h.sweep();
        h.service.run_refreshes(&llm).unwrap();
    }
    let refreshes: Vec<_> = llm
        .requests()
        .into_iter()
        .filter(|request| request.template.name == REFRESH_TEMPLATE)
        .collect();
    assert_eq!(refreshes.len(), 1, "one refresh between two sweeps");
    assert!(refreshes[0].user.contains(TEA));
    assert!(!refreshes[0].user.contains("Maya"));
}

#[test]
fn a_date_still_ahead_on_the_head_holds_a_faded_chain_back() {
    // On the head, a retracted predecessor's old
    // slot keeps nothing alive.
    let h = Harness::new();
    let concert = h.insert(faded(event(CONCERT, "2026-12-12T00:00")));
    let old_slot = h.insert(faded(event(DENTIST_9, "2026-12-09T00:00")));
    let new_slot = h.insert(faded(event(DENTIST_8, "2021-03-08T00:00")));
    h.supersede(old_slot, new_slot, true);

    h.set(at(SWEEP));
    h.sweep();
    assert_eq!(h.rows(&[concert]), 1);
    assert_eq!(h.rows(&[old_slot, new_slot]), 0);
}

#[test]
fn a_task_is_held_until_thirty_days_past_its_due_date() {
    // The agenda's 30-day overdue window is the guard. Undated tasks
    // have none.
    let h = Harness::new();
    let passport = h.insert(faded(task(PASSPORT, Some("2026-09-20T00:00"))));
    let tax = h.insert(faded(task(TAX, Some("2026-08-15T00:00"))));
    let bike = h.insert(faded(task(BIKE, None)));

    h.set(at(SWEEP));
    h.sweep();
    assert_eq!(h.rows(&[passport]), 1);
    assert_eq!(h.rows(&[tax, bike]), 0);

    // Held through the 30th day after the due day.
    h.set(local("2026-10-20T04:00"));
    h.sweep();
    assert_eq!(h.rows(&[passport]), 1);
    h.set(local("2026-10-21T04:00"));
    h.sweep();
    assert_eq!(h.rows(&[passport]), 0);
}

// The source, failed-chunk and recall-log sweep, 90 days after ingest

#[test]
fn the_sweep_deletes_text_past_the_horizon_and_keeps_the_keys() {
    let h = Harness::new();
    // The fixture source has a memory resting on it.
    let tea = h.insert(fact(TEA));
    let idle = h.ingest("idle", "Good morning.");
    h.extracted_with_nothing(idle);
    let failed = h.ingest("failed", "Good afternoon.");
    let failed_chunk: String = h.one(
        "SELECT uuid FROM chunks WHERE id = ?1",
        [h.chunk_of(failed)],
    );
    h.execute(
        "DELETE FROM extraction_queue WHERE chunk_id = ?1",
        [h.chunk_of(failed)],
    );
    h.execute(
        "UPDATE chunks SET failed_at = ?2, error_count = ?3, last_error_kind = 'http'
         WHERE id = ?1",
        (h.chunk_of(failed), micros(h.now()), CHUNK_RETRY_CAP),
    );
    h.execute(
        "INSERT INTO turn_in_context (source_id, memories)
         SELECT id, ?2 FROM sources WHERE uuid = ?1",
        (failed.to_string(), json!([tea]).to_string()),
    );
    let pending = h.ingest("pending", "Good evening.");
    let old_recall = h.recall_row(h.now(), tea);

    h.set(at(START) + days(2));
    let young = h.ingest("young", "Good night.");
    h.extracted_with_nothing(young);
    let young_recall = h.recall_row(h.now(), tea);

    h.set(at(PAST_HORIZON));
    let swept = h.sweep();
    let run = &swept.ran[0];
    assert_eq!(run.purged_memories, 0);
    assert_eq!(run.swept_sources, 2, "the idle turn and the failed one");
    assert_eq!(run.swept_failed_chunks, 1);
    assert_eq!(run.swept_recalls, 1);

    for gone in [idle, failed] {
        assert_eq!(h.source_text(gone), "", "{gone}");
        assert_eq!(h.chunk_text(gone), "", "{gone}");
        assert_eq!(
            h.one::<Option<String>, _>(
                "SELECT tombstone_reason FROM sources WHERE uuid = ?1",
                [gone.to_string()]
            )
            .as_deref(),
            Some("swept")
        );
    }
    assert_eq!(h.turn_in_context(failed), None);
    let retried = h
        .service
        .retry_chunks(BANK, Some(&[failed_chunk.parse().unwrap()]))
        .unwrap();
    assert!(retried.retried.is_empty());

    // The key stays, so ingest stays idempotent.
    let again = h
        .service
        .ingest_turn(BANK, &turn("idle", h.message_at(idle), "Good morning."))
        .unwrap();
    assert_eq!(again.outcome, Outcome::Duplicate);

    // Kept: what a memory rests on, what's younger than 90 days by
    // `ingested_at`, and what's still waiting to be extracted.
    assert_eq!(h.source_text(h.source), FIXTURE);
    assert_eq!(h.source_text(young), "Good night.");
    assert_eq!(h.source_text(pending), "Good evening.");
    assert_eq!(h.service.queue_depth(BANK).unwrap(), 1);

    // A recall row past the horizon keeps its key as the tombstone: the row
    // and its id stay, marked swept, and the query and results go.
    assert_eq!(
        h.recall_tombstone(old_recall),
        Some((None, Some(micros(at(PAST_HORIZON))), 0))
    );
    assert_eq!(
        h.recall_tombstone(young_recall),
        Some((Some("what tea does Tim like?".to_string()), None, 1))
    );
}

// The deletion fingerprint, the plan and the ack

#[test]
fn a_changed_fingerprint_pauses_purge_until_the_running_hash_is_acked() {
    let h = Harness::new();
    assert_eq!(h.service.purge_pause(), PurgePause::Running);
    let stored = h.tuning.deletion_fingerprint();
    let maya = h.insert(faded(fact(MAYA)));
    let tea = h.insert(fact(TEA));

    let h = h.restart_with("[purge]\ndelta = 0.5\n");
    let current = h.tuning.deletion_fingerprint();
    assert_ne!(current, stored);
    assert_eq!(
        h.service.purge_pause(),
        PurgePause::Paused {
            stored: stored.clone()
        }
    );

    // The sweep waits...
    h.set(at(SWEEP));
    h.sweep();
    assert_eq!(h.rows(&[maya]), 1);

    // ...but forget never does.
    h.forget(&[tea]);
    assert!(h.service.erase_next(BANK).unwrap().is_some());
    assert_eq!(h.rows(&[tea]), 0);

    // The plan shows what changed and what would go, and deletes nothing.
    let plan = h.service.purge_plan().unwrap();
    assert_eq!(plan.current, current);
    assert_eq!(plan.changed, vec!["purge.delta".to_string()]);
    assert_eq!(plan.memories, 1);
    assert_eq!(h.rows(&[maya]), 1);

    // Only the hash the running daemon computed is accepted.
    for wrong in [stored.as_str(), "not-a-hash"] {
        assert!(matches!(
            h.service.purge_ack(wrong),
            Err(PurgeError::HashMismatch)
        ));
    }
    assert!(matches!(h.service.purge_pause(), PurgePause::Paused { .. }));
    h.service.purge_ack(current.as_str()).unwrap();
    assert_eq!(h.service.purge_pause(), PurgePause::Running);
    let acked: String = h.one(
        "SELECT details FROM edits WHERE kind = 'purge_acked' AND bank_id IS NULL",
        [],
    );
    assert!(acked.contains(current.as_str()));

    // Purging resumes at the next sweep.
    h.set(at(SWEEP) + days(1));
    h.sweep();
    assert_eq!(h.rows(&[maya]), 0);

    // The ack survives a restart.
    let h = h.restart_with("[purge]\ndelta = 0.5\n");
    assert_eq!(h.service.purge_pause(), PurgePause::Running);
}

#[test]
fn a_queued_mention_is_redacted_by_its_span() {
    let h = Harness::new();
    let (maya, _) = h.says(notable(MAYA));
    let mention = h.ingest("chat", "As I said, my daughter is called Maya. Anyway.");
    h.forget(&[maya]);
    let call1 = quoting(notable(MAYA), "my daughter is called Maya");
    h.extract_labelled_none(call1, maya);
    assert!(h.service.erase_next(BANK).unwrap().is_some());
    let text = h.source_text(mention);
    assert!(text.starts_with("As I said, ") && text.ends_with(". Anyway."));
    assert!(!text.contains("Maya"));
    assert_eq!(h.one::<i64, _>("SELECT COUNT(*) FROM accesses", []), 0);
}

impl Harness {
    fn extract_labelled_none(&self, claim: Value, neighbour: Uuid) {
        let call1 = json!({"claims": [claim], "used_injected_ids": []});
        let (claim_handle, neighbour_handle) = {
            let lease = self.service.claim_chunk(BANK).unwrap().expect("queued");
            let input = self
                .service
                .call2_input(&lease, &call1, &[])
                .unwrap()
                .expect("call 2");
            let n = input
                .neighbours
                .iter()
                .find(|n| n.memory == neighbour)
                .unwrap()
                .handle
                .clone();
            (input.claims[0].handle.clone(), n)
        };
        let llm = FakeLlm::scripted(
            MODEL,
            vec![
                call1,
                json!({"claims": [{"claim": claim_handle, "labels": [{"neighbour": neighbour_handle, "label": "mentioned_again"}]}]}),
            ],
        );
        assert!(
            self.service
                .extract_next(BANK, &llm)
                .unwrap()
                .unwrap()
                .memories
                .is_empty()
        );
    }
}

#[test]
fn an_entity_a_model_filter_names_survives_the_erase() {
    let h = Harness::new();
    let tea = h.insert(fact(TEA));
    let filtered = h.entity("Teahouse", None);
    let plain = h.entity("Kettle", None);
    h.link(tea, filtered);
    h.link(tea, plain);
    h.execute("UPDATE mental_models SET filter_entity_id = ?1", [filtered]);
    h.forget(&[tea]);
    h.service.erase_next(BANK).unwrap().unwrap();
    assert!(h.entity_exists(filtered));
    assert!(!h.entity_exists(plain));
}

// Migration regressions

/// Puts the store back to schema version 7, as that binary left it: a
/// mention's span lives in `accesses.spans`, and nothing a later version
/// adds for spans exists. Reopening migrates it forward.
fn downgrade_spans_to_v7(h: &Harness) {
    h.service
        .store()
        .unwrap()
        .connection()
        .execute_batch(
            "DROP TABLE IF EXISTS mention_passages;
             DELETE FROM migrations WHERE to_version > 7;
             PRAGMA user_version = 7;",
        )
        .unwrap();
}

#[test]
fn a_mention_span_stored_by_version_7_still_redacts_after_an_upgrade() {
    // Mention spans move out of `accesses` in a new
    // migration. A store that recorded spans under version 7 has to keep
    // redacting them once it's upgraded.
    let h = Harness::new();
    let (maya, _) = h.says(notable(MAYA));
    let mention = h.ingest("chat", "As I said, my daughter is called Maya. Anyway.");
    h.extract_labelled_none(quoting(notable(MAYA), "my daughter is called Maya"), maya);
    // Version 7 wrote the span on the access: characters 11 to 37 of the
    // turn's chunk.
    h.execute(
        "UPDATE accesses SET spans = ?2 WHERE memory_id = ?1 AND kind = 'mentioned_again'",
        (
            h.rowid(maya),
            json!([[h.chunk_of(mention), 11, 37]]).to_string(),
        ),
    );
    downgrade_spans_to_v7(&h);
    let h = h.restart_with("");

    h.forget(&[maya]);
    assert!(h.service.erase_next(BANK).unwrap().is_some());
    assert_eq!(
        h.source_text(mention),
        format!("As I said, {}. Anyway.", "\u{2588}".repeat(26))
    );
}

/// Regression contracts for transactional purge, erase scheduling, forget
/// request links and legacy mention spans. These exercise `purge_candidates`,
/// `purge_chain`, `run_erases` and `forget_request`.
mod regressions {
    use super::*;

    use std::time::Duration;

    use asphodel_core::erase::ForgetRequest;
    use asphodel_core::extraction::Extracted;
    use asphodel_core::ingest::{Document, Ingested};
    use asphodel_core::models::{LlmClient, LlmError, LlmRequest, LlmResponse};
    use asphodel_core::queue::ChunkError;
    use jiff::civil::date;

    const GARDEN: &str = "Tim's garden needs water.";

    impl Harness {
        fn doc(&self, id: &str, text: &str) -> Ingested {
            self.service
                .ingest_document(
                    BANK,
                    &Document {
                        document_id: id.into(),
                        text: text.into(),
                        reference_date: date(2026, 9, 30),
                        reference_date_exact: true,
                        timezone: Some(TZ.into()),
                    },
                )
                .unwrap()
        }

        /// Extracts the head of the queue with call 1 finding `claims`. Call
        /// 2 labels `(claim index, neighbour, label)`, or nothing.
        fn extract_with(&self, claims: Vec<Value>, labels: &[(usize, Uuid, &str)]) -> Extracted {
            let call1 = json!({"claims": claims, "used_injected_ids": []});
            let call2 = if labels.is_empty() {
                json!({"claims": []})
            } else {
                let lease = self.service.claim_chunk(BANK).unwrap().expect("queued");
                let input = self
                    .service
                    .call2_input(&lease, &call1, &[])
                    .unwrap()
                    .expect("call 2 runs");
                let labelled: Vec<Value> = labels
                    .iter()
                    .map(|&(index, neighbour, label)| {
                        let claim = input
                            .claims
                            .iter()
                            .find(|claim| claim.claim == index)
                            .unwrap_or_else(|| panic!("claim {index} reaches call 2"));
                        let neighbour = input
                            .neighbours
                            .iter()
                            .find(|n| n.memory == neighbour)
                            .expect("the memory is a neighbour");
                        json!({
                            "claim": claim.handle,
                            "labels": [{"neighbour": neighbour.handle, "label": label}],
                        })
                    })
                    .collect();
                json!({ "claims": labelled })
            };
            let llm = FakeLlm::scripted(MODEL, vec![call1, call2]);
            self.service
                .extract_next(BANK, &llm)
                .unwrap()
                .expect("a chunk was queued")
        }

        /// Every mention span gone, as on a store from before version 7.
        fn make_legacy(&self) {
            self.execute("UPDATE accesses SET spans = NULL", []);
            let moved: Option<String> = self.optional(
                "SELECT name FROM sqlite_master WHERE name = 'mention_passages'",
                [],
            );
            if moved.is_some() {
                self.execute("DELETE FROM mention_passages", []);
            }
        }

        fn reply_text(&self, source: Uuid) -> String {
            self.one::<Option<String>, _>(
                "SELECT reply FROM sources WHERE uuid = ?1",
                [source.to_string()],
            )
            .unwrap_or_default()
        }

        /// The only `sweep_runs` row: its purged count and when it started.
        fn sweep_run(&self) -> (i64, i64) {
            self.service
                .store()
                .unwrap()
                .connection()
                .query_row(
                    "SELECT purged_memories, started_at FROM sweep_runs",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap()
        }

        /// The latest `forgotten` edit's details.
        fn forgotten_details(&self) -> Value {
            let details: String = self.one(
                "SELECT details FROM edits WHERE kind = 'forgotten' ORDER BY id DESC LIMIT 1",
                [],
            );
            serde_json::from_str(&details).unwrap()
        }

        fn forget_in(&self, session: Option<&str>, memories: &[Uuid]) {
            self.service
                .forget_request(
                    BANK,
                    &ForgetRequest {
                        ids: memories.iter().map(Uuid::to_string).collect(),
                        session_id: session.map(str::to_string),
                    },
                )
                .unwrap();
        }

        /// The turn that called `memory_forget`, a minute ago.
        fn forget_turn(&self, session: &str, text: &str) -> Ingested {
            let mut request = turn(session, self.now() - minutes(1), text);
            request.forget_requested = true;
            self.service.ingest_turn(BANK, &request).unwrap()
        }

        /// Every `forget` audit row's details, oldest first.
        fn forget_rows(&self) -> Vec<Value> {
            let store = self.service.store().unwrap();
            let conn = store.connection();
            let mut statement = conn
                .prepare("SELECT details FROM edits WHERE kind = 'forget' ORDER BY id")
                .unwrap();
            statement
                .query_map([], |row| row.get::<_, String>(0))
                .unwrap()
                .map(|details| serde_json::from_str(&details.unwrap()).unwrap())
                .collect()
        }
    }

    // purge decides inside each chain's transaction.

    #[test]
    fn purge_rechecks_each_chain_inside_its_own_transaction() {
        let h = Harness::new();
        let kept = h.insert(faded(fact(TEA)));
        let refined = h.insert(faded(fact(BERLIN)));
        let forgotten = h.insert(faded(fact(MAYA)));
        h.set(at(SWEEP));
        let candidates = h.service.purge_candidates().unwrap();
        let heads: BTreeSet<Uuid> = candidates.iter().map(|(_, head)| *head).collect();
        assert_eq!(heads, BTreeSet::from([kept, refined, forgotten]));

        // Between choosing and deleting: a keep, a refinement whose new head
        // is strong, and a forget.
        h.service.keep(BANK, &[kept.to_string()]).unwrap();
        let moved = h.insert(fact(MOVED));
        h.supersede(refined, moved, false);
        h.forget(&[forgotten]);

        for (bank, head) in &candidates {
            assert_eq!(h.service.purge_chain(bank, *head).unwrap(), None, "{head}");
        }
        assert_eq!(h.rows(&[kept, refined, moved, forgotten]), 4);
        assert_eq!(
            h.one::<i64, _>("SELECT COUNT(*) FROM edits WHERE kind = 'purged'", []),
            0
        );

        // The forget's own erase still finds its chain and records it.
        let erased = h.service.erase_next(BANK).unwrap().expect("the erase");
        assert_eq!(erased.memories, BTreeSet::from([forgotten]));
        assert_eq!(h.forgotten_details()["memories"], json!([forgotten]));
    }

    // a same-document mention keeps its passage for the erase.

    #[test]
    fn a_reworded_section_of_the_same_document_queued_before_a_forget_is_redacted() {
        // Not crediting a later version of the same document is right;
        // dropping where it restated the memory isn't. Rewording tests
        // passage preservation separately from exact-text matching.
        let h = Harness::new();
        h.doc("family.md", "# Family\n\nMy daughter is called Maya.\n");
        let maya = h
            .extract_with(
                vec![quoting(notable(MAYA), "My daughter is called Maya")],
                &[],
            )
            .memories[0];
        let v2 = h.doc("family.md", "# Family\n\nMaya, my daughter, is six now.\n");
        h.forget(&[maya]);
        let extracted = h.extract_with(
            vec![quoting(notable(MAYA), "Maya, my daughter")],
            &[(0, maya, "mentioned_again")],
        );
        assert!(extracted.memories.is_empty(), "the repeat is no new memory");

        assert!(h.service.erase_next(BANK).unwrap().is_some());
        let (text, chunk) = (h.source_text(v2.source), h.chunk_text(v2.source));
        assert!(!text.contains("Maya") && !chunk.contains("Maya"), "{text}");
        assert!(text.contains("is six now."), "{text}");
    }

    // a refresh in flight can't bring back a forgotten memory.

    /// A refresh LLM that forgets `memory` while its call is in flight, and
    /// with `erase` runs the erase too, then answers `reply`.
    struct ForgetsDuringCall<'a> {
        service: &'a Service,
        memory: Uuid,
        erase: bool,
        reply: Value,
    }

    impl LlmClient for ForgetsDuringCall<'_> {
        fn model(&self) -> &str {
            MODEL
        }

        fn complete(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmError> {
            self.service
                .forget(BANK, &[self.memory.to_string()])
                .unwrap();
            if self.erase {
                assert!(self.service.erase_next(BANK).unwrap().is_some());
            }
            Ok(LlmResponse {
                json: self.reply.clone(),
                usage: None,
                latency: Duration::ZERO,
            })
        }
    }

    fn refresh_while_forgetting(erase: bool) {
        let h = Harness::new();
        let tea = h.insert(fact(TEA));
        let maya = h.insert(fact(MAYA));
        let input = h.service.refresh_input(BANK, PROFILE_NAME).unwrap();
        let llm = ForgetsDuringCall {
            service: &h.service,
            memory: maya,
            erase,
            reply: json!({"operations": [
                {"op": "add", "text": "Tim has a daughter called Maya.",
                 "cites": [handle(&input, maya)]},
                {"op": "add", "text": "Tim likes green tea.", "cites": [handle(&input, tea)]},
            ]}),
        };
        let outcome = h
            .service
            .refresh_model(BANK, PROFILE_NAME, &llm, true)
            .unwrap();
        assert!(matches!(outcome, RefreshOutcome::Applied(_)), "{outcome:?}");
        let texts: Vec<String> = h.profile().entries.into_iter().map(|e| e.text).collect();
        assert_eq!(texts, vec!["Tim likes green tea.".to_string()]);
    }

    #[test]
    fn a_refresh_in_flight_drops_an_entry_citing_a_memory_hidden_meanwhile() {
        refresh_while_forgetting(false);
    }

    #[test]
    fn a_refresh_in_flight_drops_an_entry_citing_a_memory_erased_meanwhile() {
        refresh_while_forgetting(true);
    }

    // a failed sweep settles what it committed and stays due.

    #[test]
    fn a_sweep_that_fails_after_a_purge_settles_it_and_runs_again() {
        let h = Harness::new();
        h.insert(fact(TEA));
        let maya = h.insert(faded(fact(MAYA)));
        h.cite_in_profile("Tim has a daughter called Maya.", maya);
        h.set(at(SWEEP));
        let before = h.service.system_prompt(BANK, None).unwrap();
        // The store's own connection: a temporary trigger fails the run row,
        // after the purge has committed.
        h.execute(
            "CREATE TEMP TRIGGER injected_failure BEFORE INSERT ON sweep_runs
             BEGIN SELECT RAISE(ABORT, 'injected'); END",
            [],
        );
        assert!(h.service.run_sweeps().is_err());
        assert_eq!(h.rows(&[maya]), 0, "the chain committed");
        assert!(
            h.one::<Option<i64>, _>(
                "SELECT refresh_requested_at FROM mental_models WHERE name = ?1",
                [PROFILE_NAME],
            )
            .is_some(),
            "the model whose entry went is requested"
        );
        assert_ne!(
            h.service.system_prompt(BANK, None).unwrap().id,
            before.id,
            "the block was cleared"
        );

        h.execute("DROP TRIGGER temp.injected_failure", []);
        let again = h.service.run_sweeps().unwrap();
        assert_eq!(again.ran.len(), 1, "the failed night is still due");
        assert_eq!(h.one::<i64, _>("SELECT COUNT(*) FROM sweep_runs", []), 1);
        // The run row counts what the failed
        // attempt deleted, and says when that attempt started.
        assert_eq!(again.ran[0].purged_memories, 1);
        assert_eq!(h.sweep_run(), (1, micros(at(SWEEP))));
    }

    #[test]
    fn a_sweep_that_fails_is_counted_when_it_resumes_after_a_restart() {
        let h = Harness::new();
        h.insert(fact(TEA));
        let maya = h.insert(faded(fact(MAYA)));
        h.set(at(SWEEP));
        h.execute(
            "CREATE TEMP TRIGGER injected_failure BEFORE INSERT ON sweep_runs
             BEGIN SELECT RAISE(ABORT, 'injected'); END",
            [],
        );
        assert!(h.service.run_sweeps().is_err());
        assert_eq!(h.rows(&[maya]), 0, "the chain committed");

        // The temporary trigger goes with the old connection. The restarted
        // daemon's first sweep is the next 04:00.
        let h = h.restart_with("");
        h.set(at(SWEEP) + days(1));
        let resumed = h.service.run_sweeps().unwrap();
        assert_eq!(resumed.ran.len(), 1);
        assert_eq!(resumed.ran[0].purged_memories, 1);
        assert_eq!(h.one::<i64, _>("SELECT COUNT(*) FROM sweep_runs", []), 1);
        assert_eq!(h.sweep_run(), (1, micros(at(SWEEP))));
    }

    // ready erases run without a worker. The daemon half is
    // `a_ready_erase_runs_after_a_restart_without_an_llm` in serve_http.

    #[test]
    fn ready_erases_run_without_a_worker() {
        let h = Harness::new();
        let (maya, said) = h.says(notable(MAYA));
        h.ingest("chat", "Something else entirely.");
        h.forget(&[maya]);
        assert!(
            h.service.run_erases().unwrap().is_empty(),
            "the erase waits behind the queued chunk"
        );
        // That chunk fails for good, with no LLM to extract it.
        for _ in 0..CHUNK_RETRY_CAP {
            let lease = h.service.claim_chunk(BANK).unwrap().expect("queued");
            h.service
                .fail_chunk(
                    lease,
                    ChunkError {
                        kind: "transport",
                        status: None,
                    },
                )
                .unwrap();
        }
        assert_eq!(h.service.run_erases().unwrap().len(), 1);
        assert_eq!(h.rows(&[maya]), 0);
        assert!(!h.source_text(said).contains("Maya"));
    }

    // the plan counts what the sweep frees by purging.

    #[test]
    fn the_plan_counts_a_source_the_sweep_frees_by_purging() {
        let h = Harness::new();
        // The fixture source's only memory.
        h.insert(faded(fact(MAYA)));
        h.set(at(PAST_HORIZON));
        let plan = h.service.purge_plan().unwrap();
        let swept = h.sweep();
        let run = &swept.ran[0];
        assert_eq!((run.purged_memories, run.swept_sources), (1, 1));
        assert_eq!(
            (plan.memories, plan.sources),
            (run.purged_memories, run.swept_sources)
        );
    }

    // every version of a document loses a forgotten passage.

    #[test]
    fn every_version_of_a_document_loses_a_forgotten_passage() {
        let h = Harness::new();
        let family = "# Family\n\nMy daughter is called Maya.\n";
        let v1 = h.doc("family.md", family).source;
        let maya = h
            .extract_with(
                vec![quoting(notable(MAYA), "My daughter is called Maya")],
                &[],
            )
            .memories[0];
        // v2 adds a section; the Family section is skipped as already seen.
        let garden = format!("{family}\n# Garden\n\nThe roses are out.\n");
        let v2 = h.doc("family.md", &garden);
        assert_eq!(v2.chunks_skipped, 1);
        h.extract_with(vec![], &[]);

        h.forget(&[maya]);
        assert!(h.service.erase_next(BANK).unwrap().is_some());
        assert!(
            !h.source_text(v1).contains("Maya"),
            "the memory's own version"
        );
        let text = h.source_text(v2.source);
        assert!(
            !text.contains("Maya"),
            "the version that skipped it: {text}"
        );
        assert!(text.contains("The roses are out."), "{text}");

        // A version sent after the forget doesn't store the passage again.
        let v3 = h.doc(
            "family.md",
            &format!("{garden}\n# Kitchen\n\nThe kettle is new.\n"),
        );
        assert_eq!(v3.chunks_skipped, 2);
        let text = h.source_text(v3.source);
        assert!(!text.contains("Maya"), "{text}");
        assert!(text.contains("The kettle is new."), "{text}");
    }

    // a forget is linked to the turn that asked for it.
    // Each test runs at one instant on the stopped clock, so only the order
    // of writes can tell the turns apart.

    #[test]
    fn every_forget_in_the_turn_is_linked_and_other_sessions_link_none() {
        let h = Harness::new();
        let maya = h.insert(fact(MAYA));
        let tea = h.insert(fact(TEA));
        h.forget_in(Some("s"), &[maya]);
        h.forget_in(Some("s"), &[tea]);

        h.forget_turn("other", "Forget something else.");
        assert!(h.forget_rows().iter().all(|row| row["request"].is_null()));

        let request = h.forget_turn("s", "Forget those two.");
        for row in h.forget_rows() {
            assert_eq!(row["request"], json!(request.source));
        }
    }

    #[test]
    fn a_forget_whose_request_turn_never_arrived_stays_unlinked() {
        // F1's request turn was lost; an ordinary turn followed, then F2 and
        // its own request turn, all at the same instant.
        let h = Harness::new();
        let maya = h.insert(fact(MAYA));
        let tea = h.insert(fact(TEA));
        h.forget_in(Some("s"), &[maya]);
        h.ingest("s", "What's for dinner?");
        h.forget_in(Some("s"), &[tea]);
        let request = h.forget_turn("s", "Forget the tea.");

        let rows = h.forget_rows();
        assert_eq!(rows[0]["memories"], json!([maya]));
        assert_eq!(rows[0]["request"], Value::Null);
        assert_eq!(rows[1]["request"], json!(request.source));
    }

    #[test]
    fn a_forget_without_a_session_or_after_a_resent_turn_links_nothing() {
        let h = Harness::new();
        let maya = h.insert(fact(MAYA));
        let tea = h.insert(fact(TEA));
        // The CLI sends no session.
        h.forget_in(None, &[maya]);
        let first = h.forget_turn("s", "Forget that.");
        assert_eq!(first.outcome, Outcome::Tombstone);
        assert_eq!(h.forget_rows()[0]["request"], Value::Null);

        // The plugin's spool sends the same request turn again after a later
        // forget: a duplicate links nothing.
        h.forget_in(Some("s"), &[tea]);
        let again = h.forget_turn("s", "Forget that.");
        assert_eq!(again.outcome, Outcome::Duplicate);
        assert_eq!(h.forget_rows()[1]["request"], Value::Null);
    }

    // mentions stored before version 7 have no span. The erase masks
    // more rather than less, keeps what surviving memories rest on, and
    // counts the fallbacks in its audit row.

    #[test]
    fn a_legacy_mention_in_a_turn_masks_the_turn_but_what_survives() {
        let h = Harness::new();
        let (maya, _) = h.says(notable(MAYA));
        let mixed = h.ingest("chat", "My daughter is called Maya. I like green tea.");
        let tea = h
            .extract_with(
                vec![
                    quoting(notable(MAYA), "My daughter is called Maya"),
                    quoting(notable(TEA), "I like green tea"),
                ],
                &[(0, maya, "mentioned_again")],
            )
            .memories[0];
        h.make_legacy();

        h.forget(&[maya]);
        assert!(h.service.erase_next(BANK).unwrap().is_some());
        let text = h.source_text(mixed);
        assert!(
            !text.contains("Maya") && !text.contains("daughter"),
            "{text}"
        );
        assert!(text.contains("I like green tea"), "{text}");
        assert!(!h.reply_text(mixed).contains("Noted"));
        assert_eq!(h.rows(&[tea]), 1);
        let audit = h.forgotten_details();
        assert_eq!(audit["legacy_mentions"], 1);
        assert_eq!(audit["whole_source"], 0);
    }

    #[test]
    fn a_legacy_mention_in_a_document_masks_only_an_exact_passage_when_there_is_one() {
        let h = Harness::new();
        // The forgotten memory's own passage is its sentence.
        let (maya, _) = h.says(notable(MAYA));
        let notes = h
            .doc(
                "notes.md",
                &format!("# Notes\n\n{MAYA} The garden needs water.\n"),
            )
            .source;
        h.extract_with(
            vec![quoting(notable(MAYA), MAYA)],
            &[(0, maya, "mentioned_again")],
        );
        h.make_legacy();

        h.forget(&[maya]);
        assert!(h.service.erase_next(BANK).unwrap().is_some());
        let text = h.source_text(notes);
        assert!(!text.contains("Maya"), "{text}");
        assert!(text.contains("# Notes") && text.contains("The garden needs water."));
        let audit = h.forgotten_details();
        assert_eq!(audit["legacy_mentions"], 1);
        assert_eq!(audit["whole_source"], 0);
    }

    #[test]
    fn a_legacy_mention_in_a_document_with_no_exact_passage_masks_all_but_what_survives() {
        let h = Harness::new();
        let (maya, _) = h.says(notable(MAYA));
        let notes = h
            .doc(
                "notes.md",
                "# Notes\n\nMaya is my daughter. The garden needs water.\n",
            )
            .source;
        let garden = h
            .extract_with(
                vec![
                    quoting(notable(MAYA), "Maya is my daughter"),
                    quoting(notable(GARDEN), "The garden needs water"),
                ],
                &[(0, maya, "mentioned_again")],
            )
            .memories[0];
        h.make_legacy();

        h.forget(&[maya]);
        assert!(h.service.erase_next(BANK).unwrap().is_some());
        let text = h.source_text(notes);
        assert!(!text.contains("Maya") && !text.contains("Notes"), "{text}");
        assert!(text.contains("The garden needs water"), "{text}");
        assert_eq!(h.rows(&[garden]), 1);
        let audit = h.forgotten_details();
        assert_eq!(audit["legacy_mentions"], 1);
        assert_eq!(audit["whole_source"], 1);
    }

    // Overlapping passages across
    // versions. Masking one passage in a version mustn't stop a longer one
    // that contains it being found there, in one erase or in a later one.

    const DIAGNOSIS: &str = "Tim has a diagnosis.";
    const HIV: &str = "Tim has a diagnosis. It is HIV.";
    const HEALTH_V2: &str = "# Health\n\nTim has a diagnosis. It is HIV.\n";

    /// v1 says DIAGNOSIS and is extracted; v2 says HIV and is queued.
    /// Returns the DIAGNOSIS memory and v2's source.
    fn health(h: &Harness) -> (Uuid, Uuid) {
        h.doc("health.md", "# Health\n\nTim has a diagnosis.\n");
        let diagnosis = h
            .extract_with(vec![quoting(notable(DIAGNOSIS), DIAGNOSIS)], &[])
            .memories[0];
        (diagnosis, h.doc("health.md", HEALTH_V2).source)
    }

    /// v3 shares v2's Health section, so it has no chunk of its own for it,
    /// and adds a Garden section with nothing in it.
    fn garden_v3(h: &Harness) -> Uuid {
        let v3 = h.doc(
            "health.md",
            &format!("{HEALTH_V2}\n# Garden\n\nThe roses are out.\n"),
        );
        assert_eq!(v3.chunks_skipped, 1);
        h.extract_with(vec![], &[]);
        v3.source
    }

    fn assert_masked(h: &Harness, v2: Uuid, v3: Uuid) {
        for (version, source) in [("v2", v2), ("v3", v3)] {
            let text = h.source_text(source);
            assert!(
                !text.contains("HIV") && !text.contains("diagnosis"),
                "{version}: {text}"
            );
        }
        assert!(h.source_text(v3).contains("The roses are out."));
    }

    #[test]
    fn overlapping_passages_in_one_chain_leave_no_version_unmasked() {
        let h = Harness::new();
        let (diagnosis, v2) = health(&h);
        let hiv = h
            .extract_with(
                vec![quoting(notable(HIV), HIV)],
                &[(0, diagnosis, "refines")],
            )
            .memories[0];
        assert_eq!(h.superseded_by(diagnosis), Some(hiv));
        let v3 = garden_v3(&h);

        h.forget(&[diagnosis]);
        assert!(h.service.erase_next(BANK).unwrap().is_some());
        assert_masked(&h, v2, v3);
    }

    #[test]
    fn a_second_forget_masks_a_passage_an_earlier_forget_partly_masked() {
        let h = Harness::new();
        let (diagnosis, v2) = health(&h);
        let hiv = h
            .extract_with(vec![quoting(notable(HIV), HIV)], &[])
            .memories[0];
        assert_eq!(h.superseded_by(diagnosis), None, "two separate memories");
        let v3 = garden_v3(&h);

        h.forget(&[diagnosis]);
        assert!(h.service.erase_next(BANK).unwrap().is_some());
        h.forget(&[hiv]);
        assert!(h.service.erase_next(BANK).unwrap().is_some());
        assert_masked(&h, v2, v3);
    }
}
