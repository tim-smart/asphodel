//! The erase path, forget, purge and the nightly sweep, checked against
//! "Erase path, forget, purge and the nightly sweep" (TIM-112) and the
//! decisions it rests on: "Deletion policy: when faded memories are purged"
//! (TIM-97, decisions 2 to 6, as amended by TIM-99 for failed chunks),
//! "What is a memory record?" (TIM-90, as amended by TIM-97 for chain-wide
//! forget), "Configuration surface" (TIM-98, the deletion fingerprint),
//! "Operations" (TIM-99, decisions 3, 7 and 8), and ADRs 0002, 0008, 0009
//! and 0010. Where a ticket comment and an ADR disagree, the ADR wins;
//! where two comments disagree, the later amendment wins.
//!
//! The API under test is `asphodel_core::erase`, `asphodel_core::sweep`
//! and the `Service` methods over them: `forget`, `erase_next`,
//! `run_sweeps`, `purge_plan`, `purge_ack` and the purge pause.
//!
//! Every service here runs on a `SimulatedClock` stopped at 20:00 on
//! Thursday 1 October 2026 in Auckland (UTC+13) unless a test moves it. The
//! next 04:00 there, when the nightly sweep runs, is 15:00 UTC the same day.
//! The tuning sets `clock.quiet_rate = 1.0`, so bank time is world time and
//! a trivial memory said once in 2021 is well past ADR 0008's nine months.
//! Memories a test only needs present are inserted directly, as an earlier
//! extraction would have left them; the ones whose passages matter are
//! extracted from real turns with `FakeLlm`.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asphodel_core::config::PurgePause;
use asphodel_core::constants::CHUNK_RETRY_CAP;
use asphodel_core::ingest::{Outcome, Turn};
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

// Forget (ADR 0010, "Forgetting"; TIM-99, decision 3)

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
fn forget_erases_the_whole_chain_and_clears_what_points_into_it() {
    // TIM-97 decisions 2 and 5: forget takes every memory along
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

// Purge in the nightly sweep (ADR 0008; TIM-97, decisions 1 to 6)

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
}

#[test]
fn a_model_citing_a_purged_memory_refreshes_once_that_night() {
    // TIM-97 decision 6: the sweep purges before the night's refresh, so a
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
    // TIM-97 decision 3, read on the head: a retracted predecessor's old
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
    // ADR 0008: the agenda's overdue window is the guard. Undated tasks
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

// The source, failed-chunk and recall-log sweep (ADR 0008; TIM-97 decision
// 4, as amended by TIM-99)

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

// The deletion fingerprint, the plan and the ack (ADR 0009; ADR 0010,
// "Sweeps, pauses and failures"; TIM-99, decision 8)

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
