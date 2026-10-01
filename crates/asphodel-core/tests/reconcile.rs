//! Reconciliation (call 2), checked against "Reconciliation: neighbour
//! search, labels and supersession" (TIM-108) and the decisions it rests on:
//! "Extraction: significance, validity windows and supersession" (TIM-92,
//! round 1, the resolution and the TIM-97 amendment), "Strength model:
//! decay, reinforcement and significance" (TIM-91, decisions 2 and 3),
//! "Mental models" (TIM-95, decision 6), "What is a memory record?" (TIM-90,
//! ended, retracted and refined, and the access log), and ADRs 0001, 0003
//! and 0005.
//!
//! These are golden tests against `FakeLlm`: each scripts call 1's reply and
//! then call 2's, and checks what's committed, or checks the input and
//! request call 2 is given. Call 2's handles (`c1`, `n1`, …) are read from
//! the input first, the way call 1's tests read entity and memory handles.
//!
//! The API under test is call 2's items in `asphodel_core::extraction` and
//! the `Service` methods over it, `call2_input` and `extract_chunk`. Three
//! tests check what the rest rely on: that the schema holds what
//! reconciliation writes, that the floor is keyed by the exact embedding
//! model, and that the fixtures below sit on the side of the fake embedder's
//! floor each test needs.
//!
//! Every service here runs on a `SimulatedClock` stopped at one instant
//! unless a test advances it, so a stored time that equals that instant can
//! only have come from the Clock.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use asphodel_core::Service;
use asphodel_core::clock::{Clock, SimulatedClock};
use asphodel_core::config::Tuning;
use asphodel_core::constants::{SIGNIFICANCE_KEPT, Significance};
use asphodel_core::ingest::{Document, Ingested, Turn};
use asphodel_core::models::{
    Embedder, FakeEmbedder, FakeLlm, FakeReranker, LlmClient, LlmError, LlmRequest, LlmResponse,
    Models,
};
use asphodel_core::queue::{Failure, Lease};
use asphodel_core::store::{OpenOptions, Store, VectorIndex, micros, timestamp};
use asphodel_core::strength::{Access, AccessKind, BankTime, Link, inherits_from, strength};
use jiff::civil::{Date, DateTime, date};
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use rusqlite::OptionalExtension;
use rusqlite::types::FromSql;
use serde_json::{Value, json};
use uuid::Uuid;

use asphodel_core::extraction::{
    CALL2_TEMPLATE, CALL2_VERSION, Call2Input, EDIT_END_REPOINTED, EDIT_ENDED, EDIT_KEPT,
    EDIT_REFINED, EDIT_RETRACTED, EDIT_SIGNIFICANCE_RAISED, Label, NEIGHBOUR_CAP,
    NEIGHBOURS_PER_CLAIM, call2_request,
};

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";
const MODEL: &str = "fake-llm";

/// A turn's message time: 19:30 on Thursday 1 October 2026 in Auckland,
/// which is on daylight time (UTC+13).
const T1: &str = "2026-10-01T06:30:00Z";

/// When fixture memories were said, unless a test moves them: well before
/// any turn or document a test ingests.
const EARLIER: &str = "2026-09-01T00:00:00Z";

/// The fake embedder's floor in [`tuning_for_fakes`].
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
const MAYA: &str = "Tim's daughter is called Maya.";
const MIA: &str = "Tim's daughter is called Mia.";
const BIKE: &str = "Tim's bike is a Brompton.";
const SURFING: &str = "Tim once tried surfing in Raglan.";
const CAT: &str = "Tim's cat is called Miso.";
const WEATHER: &str = "The weather in Wellington was sunny.";
const ANA: &str = "Tim's sister Ana lives in Porto.";
const ANA_DOG: &str = "Ana adopted a greyhound.";
const PASSPORT: &str = "Tim needs to renew his passport.";
const JOB_HUNTING: &str = "Tim is job hunting.";
const TRAVEL: &str = "Travel documents are sorted.";

/// Pairs (claim, stored memory) that must clear [`FLOOR`] with the fake
/// embedder, so call 2 runs on them.
const ABOVE_FLOOR: [(&str, &str); 15] = [
    (TEA, TEA),
    (TEA, TEA_A_LOT),
    (TEA_AGAIN, TEA),
    (ACME_STILL, ACME),
    (ACME, ACME),
    (ACME_LEFT, ACME),
    (MOVED, BERLIN),
    (BERLIN, LISBON),
    (DENTIST_9, DENTIST_8),
    (DENTIST_8, DENTIST_9),
    (TOKYO, JAPAN),
    (NO_COFFEE, COFFEE),
    (TAX_FILED, TAX_TASK),
    (TAX_FILED_LATER, TAX_FILED),
    (MIA, MAYA),
];

/// Pairs that must stay below [`FLOOR`]: a unit with only these doesn't run
/// call 2 unless a claim is flagged.
const BELOW_FLOOR: [(&str, &str); 4] = [
    (WEATHER, CAT),
    (ANA_DOG, ANA),
    (TRAVEL, PASSPORT),
    (TRAVEL, JOB_HUNTING),
];

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

/// Cosine similarity under the fake embedder, which returns unit vectors.
fn similarity(a: &str, b: &str) -> f64 {
    let vectors = FakeEmbedder.embed(&[a, b]).unwrap();
    vectors[0]
        .iter()
        .zip(&vectors[1])
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum()
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

/// A floor for each fake model, the embedder's at `floor`, so a service
/// opens on the fakes.
fn tuning_with_floor(floor: f64) -> Tuning {
    Tuning::from_toml(&format!(
        "[injection.reranker_floors]\n\"{}\" = 0.0\n\
         [reconcile.embedding_floors]\n\"{}\" = {floor:?}\n",
        FakeReranker::MODEL_ID,
        FakeEmbedder::MODEL_ID,
    ))
    .unwrap()
}

fn tuning_for_fakes() -> Tuning {
    tuning_with_floor(FLOOR)
}

/// The owner is Tim and the assistant is Hermes.
fn identity() -> asphodel_core::store::bank::BankIdentity {
    asphodel_core::store::bank::BankIdentity {
        owner_name: Some("Tim".into()),
        owner_platform_ids: vec!["discord:1234".into()],
        assistant_name: Some("Hermes".into()),
        timezone: Some(TZ.into()),
    }
}

/// A public id for a fixture row.
fn next_uuid() -> Uuid {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    Uuid::from_u128((0xf2_u128 << 120) | u128::from(NEXT.fetch_add(1, Ordering::Relaxed)))
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
        match self.first.lock().unwrap().take() {
            Some(json) => Ok(LlmResponse {
                json,
                usage: None,
                latency: Duration::ZERO,
            }),
            None => Err((self.error)()),
        }
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
        Self::with_floor(FLOOR)
    }

    fn with_floor(floor: f64) -> Self {
        let dir = TestDir::new();
        let clock = Arc::new(SimulatedClock::new(at(START)));
        let store = Store::open(&dir.data(), OpenOptions::default(), clock.clone()).unwrap();
        let service = Service::with_models(
            clock.clone(),
            store,
            tuning_with_floor(floor),
            Models::fake(),
        )
        .unwrap();
        service
            .ensure_bank_with_models("main", &identity())
            .unwrap();
        service
            .ensure_bank_with_models("other", &identity())
            .unwrap();
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
            Service::with_models(clock.clone(), store, tuning_for_fakes(), Models::fake()).unwrap();
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

    fn all<T: FromSql, P: rusqlite::Params>(&self, sql: &str, params: P) -> Vec<T> {
        let store = self.service.store().unwrap();
        let conn = store.connection();
        let mut statement = conn.prepare(sql).unwrap();
        statement
            .query_map(params, |row| row.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
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

    fn bank_id(&self, bank: &str) -> i64 {
        self.one("SELECT id FROM banks WHERE name = ?1", [bank])
    }

    /// The bank's turn counter.
    fn turns(&self, bank: &str) -> i64 {
        self.one("SELECT turns FROM banks WHERE name = ?1", [bank])
    }

    fn seeded(&self, bank: &str, which: &str) -> Uuid {
        let uuid: String = self.one(
            "SELECT e.uuid FROM entities e JOIN banks b ON b.id = e.bank_id
             WHERE b.name = ?1 AND e.seeded = ?2",
            [bank, which],
        );
        uuid.parse().unwrap()
    }

    fn rowid(&self, memory: Uuid) -> i64 {
        self.one(
            "SELECT id FROM memories WHERE uuid = ?1",
            [memory.to_string()],
        )
    }

    /// The chunk fixture memories rest on: a turn in session `fixtures`,
    /// ingested once per bank and taken off the queue as if extracted. It
    /// counts as the bank's first turn.
    fn fixture_chunk(&self, bank: &str) -> i64 {
        let find = || {
            self.optional::<i64, _>(
                "SELECT c.id FROM chunks c JOIN sources s ON s.id = c.source_id
                 JOIN banks b ON b.id = s.bank_id
                 WHERE b.name = ?1 AND s.session_id = 'fixtures'",
                [bank],
            )
        };
        if let Some(chunk) = find() {
            return chunk;
        }
        self.service
            .ingest_turn(bank, &turn("fixtures", EARLIER, "Fixtures.", "Noted."))
            .unwrap();
        let chunk = find().unwrap();
        self.execute("DELETE FROM extraction_queue WHERE chunk_id = ?1", [chunk]);
        self.execute(
            "UPDATE chunks SET extracted_at = ?2 WHERE id = ?1",
            (chunk, micros(self.now())),
        );
        chunk
    }

    /// A memory in `bank` observed at [`EARLIER`] with one `created` access
    /// at that time and turn 0, inserted directly, as an earlier extraction
    /// would have left it.
    fn insert_memory(&self, bank: &str, content: &str, kind: &str, significance: &str) -> Uuid {
        let chunk = self.fixture_chunk(bank);
        let uuid = next_uuid();
        let earlier = micros(at(EARLIER));
        let now = micros(self.now());
        self.execute(
            "INSERT INTO memories (uuid, bank_id, content, kind, significance, chunk_id,
                                   source_start, source_end, observed_at, window_confidence,
                                   created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, 9, ?7, 'high', ?8, ?8)",
            (
                uuid.to_string(),
                self.bank_id(bank),
                content,
                kind,
                significance,
                chunk,
                earlier,
                now,
            ),
        );
        // Indexed with the fake embedder, as extraction would have.
        let vector = FakeEmbedder.embed(&[content]).unwrap().remove(0);
        let (bank_id, memory_id) = (self.bank_id(bank), self.rowid(uuid));
        let store = self.service.store().unwrap();
        store
            .vectors()
            .upsert(&store.connection(), bank_id, memory_id, &vector)
            .unwrap();
        self.execute(
            "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
             SELECT bank_id, id, 'created', ?2, 0 FROM memories WHERE uuid = ?1",
            (uuid.to_string(), earlier),
        );
        uuid
    }

    /// `count` versions of one fact in `main`, each refined into the next,
    /// inserted in one transaction as [`Harness::insert_memory`] would insert
    /// each. Returns the head.
    fn insert_chain(&self, content: &str, count: usize) -> Uuid {
        let chunk = self.fixture_chunk("main");
        let bank_id = self.bank_id("main");
        let vector = FakeEmbedder.embed(&[content]).unwrap().remove(0);
        let earlier = micros(at(EARLIER));
        let now = micros(self.now());
        let store = self.service.store().unwrap();
        let mut conn = store.connection();
        let tx = conn.transaction().unwrap();
        let mut previous: Option<i64> = None;
        let mut head = None;
        for _ in 0..count {
            let uuid = next_uuid();
            tx.execute(
                "INSERT INTO memories (uuid, bank_id, content, kind, significance, chunk_id,
                                       source_start, source_end, observed_at, window_confidence,
                                       created_at, updated_at)
                 VALUES (?1, ?2, ?3, 'fact', 'minor', ?4, 0, 9, ?5, 'high', ?6, ?6)",
                (uuid.to_string(), bank_id, content, chunk, earlier, now),
            )
            .unwrap();
            let id = tx.last_insert_rowid();
            store.vectors().upsert(&tx, bank_id, id, &vector).unwrap();
            tx.execute(
                "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
                 VALUES (?1, ?2, 'created', ?3, 0)",
                (bank_id, id, earlier),
            )
            .unwrap();
            if let Some(previous) = previous {
                tx.execute(
                    "UPDATE memories SET superseded_by = ?2 WHERE id = ?1",
                    (previous, id),
                )
                .unwrap();
            }
            previous = Some(id);
            head = Some(uuid);
        }
        tx.commit().unwrap();
        head.expect("a chain has a version")
    }

    fn fact(&self, content: &str) -> Uuid {
        self.insert_memory("main", content, "fact", "minor")
    }

    /// An access on `memory`, inserted directly.
    fn insert_access(&self, memory: Uuid, kind: &str, turn: i64, at: Timestamp) {
        self.execute(
            "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
             SELECT bank_id, id, ?2, ?3, ?4 FROM memories WHERE uuid = ?1",
            (memory.to_string(), kind, micros(at), turn),
        );
    }

    fn set_observed_at(&self, memory: Uuid, observed_at: Timestamp) {
        self.execute(
            "UPDATE memories SET observed_at = ?2 WHERE uuid = ?1",
            (memory.to_string(), micros(observed_at)),
        );
    }

    fn set_valid_from(&self, memory: Uuid, valid_from: Timestamp, precision: &str) {
        self.execute(
            "UPDATE memories SET valid_from = ?2, valid_from_precision = ?3 WHERE uuid = ?1",
            (memory.to_string(), micros(valid_from), precision),
        );
    }

    fn set_owner_significance(&self, memory: Uuid, level: &str) {
        self.execute(
            "UPDATE memories SET owner_significance = ?2 WHERE uuid = ?1",
            (memory.to_string(), level),
        );
    }

    /// Ends `memory` by `by`, as an earlier reconciliation would have.
    fn mark_ended(&self, memory: Uuid, by: Uuid, until: Timestamp, precision: &str) {
        self.execute(
            "UPDATE memories SET valid_until = ?3, valid_until_precision = ?4,
                    ended_by = (SELECT id FROM memories WHERE uuid = ?2)
             WHERE uuid = ?1",
            (memory.to_string(), by.to_string(), micros(until), precision),
        );
    }

    /// Retracts `memory` in favour of `by`, or with no successor.
    fn mark_retracted(&self, memory: Uuid, by: Option<Uuid>) {
        self.execute(
            "UPDATE memories SET invalidated_at = ?3,
                    superseded_by = (SELECT id FROM memories WHERE uuid = ?2)
             WHERE uuid = ?1",
            (
                memory.to_string(),
                by.map(|by| by.to_string()),
                micros(self.now()),
            ),
        );
    }

    /// Refines `memory` into `by`.
    fn mark_refined(&self, memory: Uuid, by: Uuid) {
        self.execute(
            "UPDATE memories SET superseded_by = (SELECT id FROM memories WHERE uuid = ?2)
             WHERE uuid = ?1",
            (memory.to_string(), by.to_string()),
        );
    }

    fn link(&self, memory: Uuid, entity: Uuid) {
        self.execute(
            "INSERT INTO memory_entities (memory_id, entity_id)
             SELECT m.id, e.id FROM memories m, entities e WHERE m.uuid = ?1 AND e.uuid = ?2",
            (memory.to_string(), entity.to_string()),
        );
    }

    /// A memory's accesses, oldest row first.
    fn accesses(&self, memory: Uuid) -> Vec<AccessRow> {
        let store = self.service.store().unwrap();
        let conn = store.connection();
        let mut statement = conn
            .prepare(
                "SELECT a.kind, a.at, a.turn, s.uuid FROM accesses a
                 JOIN memories m ON m.id = a.memory_id LEFT JOIN sources s ON s.id = a.source_id
                 WHERE m.uuid = ?1 ORDER BY a.id",
            )
            .unwrap();
        statement
            .query_map([memory.to_string()], |row| {
                Ok(AccessRow {
                    kind: row.get(0)?,
                    at: timestamp(row.get(1)?),
                    turn: row.get(2)?,
                    source: row
                        .get::<_, Option<String>>(3)?
                        .map(|uuid| uuid.parse().unwrap()),
                })
            })
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    /// The access every fixture memory starts with.
    fn fixture_access() -> AccessRow {
        AccessRow {
            kind: "created".into(),
            at: at(EARLIER),
            turn: 0,
            source: None,
        }
    }

    /// The edit rows of `kind` on `memory`.
    fn edits_on(&self, memory: Uuid, kind: &str) -> i64 {
        self.one(
            "SELECT COUNT(*) FROM edits e JOIN memories m ON m.id = e.memory_id
             WHERE m.uuid = ?1 AND e.kind = ?2",
            (memory.to_string(), kind),
        )
    }

    /// Every edit row on `memory`.
    fn all_edits_on(&self, memory: Uuid) -> i64 {
        self.one(
            "SELECT COUNT(*) FROM edits e JOIN memories m ON m.id = e.memory_id
             WHERE m.uuid = ?1",
            [memory.to_string()],
        )
    }

    fn memories_in(&self, bank: &str) -> i64 {
        self.one(
            "SELECT COUNT(*) FROM memories m JOIN banks b ON b.id = m.bank_id WHERE b.name = ?1",
            [bank],
        )
    }

    fn content(&self, memory: Uuid) -> String {
        self.one(
            "SELECT content FROM memories WHERE uuid = ?1",
            [memory.to_string()],
        )
    }

    fn kind(&self, memory: Uuid) -> String {
        self.one(
            "SELECT kind FROM memories WHERE uuid = ?1",
            [memory.to_string()],
        )
    }

    /// The significance extraction gave, and the owner's setting.
    fn significance(&self, memory: Uuid) -> (String, Option<String>) {
        self.service
            .store()
            .unwrap()
            .connection()
            .query_row(
                "SELECT significance, owner_significance FROM memories WHERE uuid = ?1",
                [memory.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap()
    }

    fn valid_from(&self, memory: Uuid) -> Option<(Timestamp, String)> {
        self.service
            .store()
            .unwrap()
            .connection()
            .query_row(
                "SELECT valid_from, valid_from_precision FROM memories WHERE uuid = ?1",
                [memory.to_string()],
                |row| {
                    let at: Option<i64> = row.get(0)?;
                    let precision: Option<String> = row.get(1)?;
                    Ok(at.map(|at| (timestamp(at), precision.unwrap())))
                },
            )
            .unwrap()
    }

    /// The fields ending, retracting and refining write.
    fn change(&self, memory: Uuid) -> Change {
        self.service
            .store()
            .unwrap()
            .connection()
            .query_row(
                "SELECT m.valid_until, m.valid_until_precision, m.window_confidence,
                        m.invalidated_at, s.uuid, e.uuid
                 FROM memories m
                 LEFT JOIN memories s ON s.id = m.superseded_by
                 LEFT JOIN memories e ON e.id = m.ended_by
                 WHERE m.uuid = ?1",
                [memory.to_string()],
                |row| {
                    let until: Option<i64> = row.get(0)?;
                    let precision: Option<String> = row.get(1)?;
                    let uuid = |text: Option<String>| text.map(|text| text.parse().unwrap());
                    Ok(Change {
                        valid_until: until.map(|until| (timestamp(until), precision.unwrap())),
                        window_confidence: row.get(2)?,
                        invalidated_at: row.get::<_, Option<i64>>(3)?.map(timestamp),
                        superseded_by: uuid(row.get(4)?),
                        ended_by: uuid(row.get(5)?),
                    })
                },
            )
            .unwrap()
    }

    fn chunk_column<T: FromSql>(&self, chunk: Uuid, column: &str) -> T {
        self.one(
            &format!("SELECT {column} FROM chunks WHERE uuid = ?1"),
            [chunk.to_string()],
        )
    }

    /// `memory`'s strength now, from the public pieces the way the store
    /// loads it: its significance, its own accesses and those it inherits
    /// along `superseded_by`, on the bank's clock. For memories with no
    /// window, so there's no close to restart recent use.
    fn strength_of(&self, memory: Uuid) -> f64 {
        let bank_id: i64 = self.one(
            "SELECT bank_id FROM memories WHERE uuid = ?1",
            [memory.to_string()],
        );
        let turns: Vec<Timestamp> = self
            .all::<i64, _>(
                "SELECT message_at FROM sources WHERE bank_id = ?1 AND kind = 'turn'",
                [bank_id],
            )
            .into_iter()
            .map(timestamp)
            .collect();
        let links: Vec<Link> = {
            let store = self.service.store().unwrap();
            let conn = store.connection();
            let mut statement = conn
                .prepare(
                    "SELECT id, superseded_by, ended_by FROM memories
                     WHERE bank_id = ?1 AND superseded_by IS NOT NULL",
                )
                .unwrap();
            statement
                .query_map([bank_id], |row| {
                    Ok(Link {
                        id: row.get(0)?,
                        superseded_by: row.get(1)?,
                        ended_by: row.get(2)?,
                    })
                })
                .unwrap()
                .collect::<Result<_, _>>()
                .unwrap()
        };
        let mut accesses = Vec::new();
        for id in inherits_from(&links, self.rowid(memory)) {
            let uuid: String = self.one("SELECT uuid FROM memories WHERE id = ?1", [id]);
            for row in self.accesses(uuid.parse().unwrap()) {
                let kind = match row.kind.as_str() {
                    "created" => AccessKind::Created,
                    "used" => AccessKind::Used,
                    "mentioned_again" => AccessKind::MentionedAgain,
                    _ => AccessKind::Confirmed,
                };
                accesses.push(Access { kind, at: row.at });
            }
        }
        let (level, owner) = self.significance(memory);
        let significance = match owner.as_deref().unwrap_or(&level) {
            "kept" => SIGNIFICANCE_KEPT,
            level => serde_json::from_value::<Significance>(json!(level))
                .unwrap()
                .value(),
        };
        let bank_time = BankTime::new(&turns, self.service.tuning().clock.quiet_rate);
        strength(significance, &accesses, None, &bank_time, self.now()).value
    }
}

#[derive(Debug, Clone, PartialEq)]
struct AccessRow {
    kind: String,
    at: Timestamp,
    turn: i64,
    source: Option<Uuid>,
}

/// What ending, retracting and refining write on the memory they change.
#[derive(Debug, Clone, PartialEq)]
struct Change {
    valid_until: Option<(Timestamp, String)>,
    window_confidence: String,
    invalidated_at: Option<Timestamp>,
    superseded_by: Option<Uuid>,
    ended_by: Option<Uuid>,
}

impl Change {
    /// A fixture as inserted: open, current and high confidence.
    fn untouched() -> Self {
        Self {
            valid_until: None,
            window_confidence: "high".into(),
            invalidated_at: None,
            superseded_by: None,
            ended_by: None,
        }
    }
}

fn timed(at: Timestamp, precision: &str) -> Option<(Timestamp, String)> {
    Some((at, precision.into()))
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

fn document(id: &str, text: &str, reference_date: Date) -> Document {
    Document {
        document_id: id.into(),
        text: text.into(),
        reference_date,
        reference_date_exact: true,
        timezone: Some(TZ.into()),
    }
}

/// The owner says `user` in session `s1` at [`T1`] and the assistant answers
/// "Noted.". Returns the source.
fn owner_says(h: &Harness, user: &str) -> Uuid {
    h.service
        .ingest_turn("main", &turn("s1", T1, user, "Noted."))
        .unwrap()
        .source
}

fn ingest_doc(h: &Harness, document: &Document) -> Ingested {
    h.service.ingest_document("main", document).unwrap()
}

fn lease(h: &Harness, bank: &str) -> Lease {
    h.service
        .claim_chunk(bank)
        .unwrap()
        .expect("a chunk is queued")
}

/// Call 2's input for the head of `main`'s queue when call 1 replies
/// `call1`, or `None` when call 2 won't run. The lease is released when it
/// drops.
fn call2(h: &Harness, call1: &Value) -> Option<Call2Input> {
    call2_with(h, call1, &[])
}

fn call2_with(h: &Harness, call1: &Value, in_context: &[Uuid]) -> Option<Call2Input> {
    let lease = lease(h, "main");
    h.service.call2_input(&lease, call1, in_context).unwrap()
}

/// Extracts the head of `main`'s queue with call 1 answering only `call1`,
/// and checks call 2 didn't run.
fn extract_alone(h: &Harness, call1: Value) -> asphodel_core::extraction::Extracted {
    let llm = FakeLlm::scripted(MODEL, vec![call1]);
    let extracted = h
        .service
        .extract_chunk(lease(h, "main"), &llm, &[])
        .unwrap();
    assert_eq!(llm.requests().len(), 1, "call 1 only");
    extracted
}

/// Extracts the head of `main`'s queue with call 1 answering `call1` and
/// call 2 answering `call2`.
fn reconcile(h: &Harness, call1: Value, call2: Value) -> asphodel_core::extraction::Extracted {
    reconcile_with(h, call1, call2, &[])
}

fn reconcile_with(
    h: &Harness,
    call1: Value,
    call2: Value,
    in_context: &[Uuid],
) -> asphodel_core::extraction::Extracted {
    let llm = FakeLlm::scripted(MODEL, vec![call1, call2]);
    let extracted = h
        .service
        .extract_chunk(lease(h, "main"), &llm, in_context)
        .unwrap();
    let requests = llm.requests();
    assert_eq!(requests.len(), 2, "call 1 and call 2");
    assert_eq!(requests[1].template.name, CALL2_TEMPLATE);
    extracted
}

/// Extracts the head of `main`'s queue where call 1 finds the one claim in
/// `call1` and call 2 labels it `label` on `neighbour`.
fn one_label(
    h: &Harness,
    call1: Value,
    neighbour: Uuid,
    label: &str,
) -> asphodel_core::extraction::Extracted {
    let input = call2(h, &call1).expect("call 2 runs");
    let claim = input.claims[0].handle.clone();
    reconcile(
        h,
        call1,
        call2_reply(vec![labelled(
            &claim,
            &[(neighbour_handle(&input, neighbour), label)],
        )]),
    )
}

fn neighbour_handle(input: &Call2Input, memory: Uuid) -> String {
    input
        .neighbours
        .iter()
        .find(|neighbour| neighbour.memory == memory)
        .unwrap_or_else(|| panic!("{memory} is a neighbour"))
        .handle
        .clone()
}

/// The neighbours' memories.
fn shown(input: &Call2Input) -> BTreeSet<Uuid> {
    input
        .neighbours
        .iter()
        .map(|neighbour| neighbour.memory)
        .collect()
}

/// The memories shown for the claim at `index` in call 1's reply.
fn shown_for(input: &Call2Input, index: usize) -> BTreeSet<Uuid> {
    let claim = input
        .claims
        .iter()
        .find(|claim| claim.claim == index)
        .unwrap_or_else(|| panic!("claim {index} reaches call 2"));
    claim
        .neighbours
        .iter()
        .map(|handle| {
            input
                .neighbours
                .iter()
                .find(|neighbour| &neighbour.handle == handle)
                .expect("a claim's neighbour is in the input")
                .memory
        })
        .collect()
}

// Call 1's reply, as `FakeLlm` scripts it.

/// A claim with nothing but its sentence, kind and quote: minor, high window
/// confidence, no times and no entities.
fn claim(content: &str, kind: &str, quote: &str) -> Value {
    json!({
        "content": content,
        "kind": kind,
        "quote": quote,
        "significance": "minor",
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

trait With {
    fn with(self, key: &str, value: Value) -> Value;
}

impl With for Value {
    fn with(mut self, key: &str, value: Value) -> Value {
        self[key] = value;
        self
    }
}

fn time(at: &str, precision: &str) -> Value {
    json!({"at": at, "precision": precision})
}

fn changes(claim: Value) -> Value {
    claim.with("changes_something", json!(true))
}

fn reply(claims: Vec<Value>) -> Value {
    json!({"claims": claims, "used_injected_ids": []})
}

// Call 2's reply.

fn labelled(claim: &str, labels: &[(String, &str)]) -> Value {
    json!({
        "claim": claim,
        "labels": labels
            .iter()
            .map(|(neighbour, label)| json!({"neighbour": neighbour, "label": label}))
            .collect::<Vec<_>>(),
    })
}

fn call2_reply(claims: Vec<Value>) -> Value {
    json!({"claims": claims})
}

// What runs now: the schema, the floors and the fixtures.

#[test]
fn the_schema_holds_what_reconciliation_writes() {
    let h = Harness::new();
    for (table, column) in [
        ("memories", "valid_until"),
        ("memories", "valid_until_precision"),
        ("memories", "window_confidence"),
        ("memories", "invalidated_at"),
        ("memories", "superseded_by"),
        ("memories", "ended_by"),
        ("memories", "owner_significance"),
        ("edits", "memory_id"),
        ("chunks", "call1_output"),
        ("mental_model_citations", "memory_id"),
    ] {
        let found: i64 = h.one(
            &format!("SELECT COUNT(*) FROM pragma_table_info('{table}') WHERE name = ?1"),
            [column],
        );
        assert_eq!(found, 1, "{table}.{column}");
    }

    // Both kinds of access reconciliation writes fit the log.
    let memory = h.fact(TEA);
    h.insert_access(memory, "mentioned_again", 1, h.now());
    h.insert_access(memory, "confirmed", 2, h.now());
    assert_eq!(h.accesses(memory).len(), 3);
}

#[test]
fn the_floor_is_keyed_by_the_exact_embedding_model() {
    // TIM-92: the floor is stored per embedding model, and ADR 0009 refuses
    // to run a model without one.
    let tuning = tuning_for_fakes();
    assert_eq!(
        tuning
            .reconcile
            .embedding_floors
            .get(FakeEmbedder::MODEL_ID),
        Some(&FLOOR)
    );
    assert!(
        tuning
            .check_floors(FakeEmbedder::MODEL_ID, FakeReranker::MODEL_ID)
            .is_ok()
    );
    assert!(
        tuning
            .check_floors("BAAI/bge-small-en-v1.5", FakeReranker::MODEL_ID)
            .is_err()
    );
}

#[test]
fn the_fixtures_sit_on_the_side_of_the_floor_each_test_needs() {
    for (claim, memory) in ABOVE_FLOOR {
        let similarity = similarity(claim, memory);
        assert!(
            similarity >= FLOOR,
            "{claim:?} against {memory:?} is {similarity}, below the floor"
        );
    }
    for (claim, memory) in BELOW_FLOOR {
        let similarity = similarity(claim, memory);
        assert!(
            similarity < FLOOR,
            "{claim:?} against {memory:?} is {similarity}, at or above the floor"
        );
    }
    // The floor test needs a pair between the fake floor and a strict one.
    let paraphrase = similarity(TEA_AGAIN, TEA);
    assert!((FLOOR..0.99).contains(&paraphrase), "{paraphrase}");
}

// When call 2 runs.

#[test]
fn a_unit_that_touches_nothing_known_costs_one_call() {
    let h = Harness::new();
    h.fact(CAT);
    owner_says(&h, "The weather in Wellington was sunny.");
    let call1 = reply(vec![claim(
        WEATHER,
        "event",
        "The weather in Wellington was sunny",
    )]);

    assert_eq!(call2(&h, &call1), None);
    let extracted = extract_alone(&h, call1);
    assert_eq!(extracted.memories.len(), 1);
    assert_eq!(h.content(extracted.memories[0]), WEATHER);
}

#[test]
fn a_neighbour_above_the_floor_runs_call_2() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    owner_says(&h, "I really like green tea.");
    let call1 = reply(vec![claim(TEA_AGAIN, "fact", "I really like green tea")]);

    let input = call2(&h, &call1).expect("call 2 runs");
    assert_eq!(input.claims.len(), 1);
    assert_eq!(input.claims[0].handle, "c1");
    assert_eq!(input.claims[0].claim, 0);
    assert_eq!(input.claims[0].content, TEA_AGAIN);
    assert_eq!(input.claims[0].observed_at, at(T1));
    assert!(!input.claims[0].flagged);
    assert_eq!(shown_for(&input, 0), BTreeSet::from([tea]));
    let neighbour = &input.neighbours[0];
    assert_eq!(neighbour.content, TEA);
    assert_eq!(neighbour.observed_at, at(EARLIER));
    assert!(!neighbour.ended);

    // A claim with no labels is new (TIM-92).
    let extracted = reconcile(&h, call1, call2_reply(vec![]));
    assert_eq!(extracted.memories.len(), 1);
    assert_eq!(h.content(extracted.memories[0]), TEA_AGAIN);
    assert_eq!(h.accesses(tea), vec![Harness::fixture_access()]);
}

#[test]
fn the_floor_is_the_one_for_the_embedding_model() {
    // The same paraphrase runs call 2 under the fake floor and doesn't under
    // a stricter one, so the floor read is the configured one.
    for (floor, runs) in [(FLOOR, true), (0.99, false)] {
        let h = Harness::with_floor(floor);
        h.fact(TEA);
        owner_says(&h, "I really like green tea.");
        let call1 = reply(vec![claim(TEA_AGAIN, "fact", "I really like green tea")]);
        assert_eq!(call2(&h, &call1).is_some(), runs, "floor {floor}");
    }
}

#[test]
fn a_flagged_claim_sees_open_tasks_and_current_states_of_its_entities() {
    let h = Harness::new();
    let user = h.seeded("main", "user");
    let passport = h.insert_memory("main", PASSPORT, "task", "minor");
    let job_hunting = h.insert_memory("main", JOB_HUNTING, "state", "minor");
    h.link(passport, user);
    h.link(job_hunting, user);
    owner_says(&h, "Travel documents are sorted.");
    let user_handle = {
        let lease = lease(&h, "main");
        let input = h.service.call1_input(&lease, &[]).unwrap();
        input
            .candidates
            .iter()
            .find(|candidate| candidate.entity == user)
            .unwrap()
            .handle
            .clone()
    };
    let travel = claim(TRAVEL, "event", "Travel documents are sorted").with(
        "entities",
        json!([{"entity": user_handle, "new_name": null, "new_kind": null, "surface_form": "I"}]),
    );

    // Nothing clears the floor, so an unflagged claim costs one call.
    assert_eq!(call2(&h, &reply(vec![travel.clone()])), None);

    // A claim that changes something gets the wider set: the open tasks and
    // current states linked to its entities (TIM-92).
    let input = call2(&h, &reply(vec![changes(travel.clone())])).expect("call 2 runs");
    assert!(input.claims[0].flagged);
    let found = shown_for(&input, 0);
    assert!(found.contains(&passport), "{found:?}");
    assert!(found.contains(&job_hunting), "{found:?}");

    // So does remember-this.
    let input =
        call2(&h, &reply(vec![travel.with("remember_this", json!(true))])).expect("call 2 runs");
    assert!(input.claims[0].flagged);
    assert!(shown_for(&input, 0).contains(&passport));
}

#[test]
fn bm25_hits_fill_out_the_candidates_only_once_call_2_runs() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    let ana = h.fact(ANA);
    owner_says(&h, "Ana adopted a greyhound. I like green tea.");
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
fn a_neighbour_hit_by_several_claims_appears_once() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    owner_says(&h, "I like green tea. I really like green tea.");
    let call1 = reply(vec![
        claim(TEA, "fact", "I like green tea"),
        claim(TEA_AGAIN, "fact", "I really like green tea"),
    ]);
    let input = call2(&h, &call1).expect("call 2 runs");
    assert_eq!(
        input
            .neighbours
            .iter()
            .filter(|neighbour| neighbour.memory == tea)
            .count(),
        1
    );
    assert!(shown_for(&input, 0).contains(&tea));
    assert!(shown_for(&input, 1).contains(&tea));
}

#[test]
fn neighbours_are_capped_per_claim_and_per_unit() {
    // Nine topics with six close memories each: 45 candidates once each
    // claim keeps its top five, so the unit cap bites.
    let h = Harness::new();
    let birds = [
        "kestrel", "heron", "kea", "weka", "kiwi", "ruru", "kaka", "tui", "titi",
    ];
    for bird in birds {
        for count in 1..=6 {
            h.fact(&format!(
                "Tim counted {count} {bird} nests at {bird} point."
            ));
        }
    }
    let sentences: Vec<String> = birds
        .iter()
        .map(|bird| format!("Tim counted {bird} nests at {bird} point."))
        .collect();
    owner_says(&h, &sentences.join(" "));
    let call1 = reply(
        sentences
            .iter()
            .map(|sentence| claim(sentence, "event", sentence.trim_end_matches('.')))
            .collect(),
    );

    let input = call2(&h, &call1).expect("call 2 runs");
    assert_eq!(input.claims.len(), birds.len());
    for claim in &input.claims {
        assert!(claim.neighbours.len() <= NEIGHBOURS_PER_CLAIM, "{claim:?}");
    }
    assert_eq!(input.neighbours.len(), NEIGHBOUR_CAP);
    assert_eq!(shown(&input).len(), NEIGHBOUR_CAP, "each neighbour once");
}

#[test]
fn faded_and_ended_memories_are_neighbours() {
    let h = Harness::new();
    // Trivial and untouched for two years: long faded out, but reconcile
    // still matches against it (ADR 0005, ADR 0008).
    let surfing = h.insert_memory("main", SURFING, "event", "trivial");
    h.execute(
        "UPDATE accesses SET at = ?2 WHERE memory_id = (SELECT id FROM memories WHERE uuid = ?1)",
        (surfing.to_string(), micros(at("2024-09-01T00:00:00Z"))),
    );
    h.set_observed_at(surfing, at("2024-09-01T00:00:00Z"));
    assert!(h.strength_of(surfing) < asphodel_core::constants::TAU);
    let acme = h.fact(ACME);
    let left = h.insert_memory("main", ACME_LEFT, "event", "minor");
    h.mark_ended(acme, left, local("2026-08-01T00:00"), "month");

    owner_says(&h, "I once tried surfing in Raglan. I work at Acme.");
    let call1 = reply(vec![
        claim(SURFING, "event", "I once tried surfing in Raglan"),
        claim(ACME, "fact", "I work at Acme"),
    ]);
    let input = call2(&h, &call1).expect("call 2 runs");
    assert!(shown_for(&input, 0).contains(&surfing));
    assert!(shown_for(&input, 1).contains(&acme));
    let ended = |memory: Uuid| {
        input
            .neighbours
            .iter()
            .find(|neighbour| neighbour.memory == memory)
            .unwrap()
            .ended
    };
    assert!(ended(acme));
    assert!(!ended(surfing));
}

#[test]
fn a_retracted_memory_isnt_a_neighbour_but_its_chain_head_is() {
    let h = Harness::new();
    let dentist_8 = h.insert_memory("main", DENTIST_8, "event", "minor");
    let dentist_9 = h.insert_memory("main", DENTIST_9, "event", "minor");
    h.mark_retracted(dentist_8, Some(dentist_9));
    // A retraction whose successor is gone leaves no head to show.
    let maya = h.fact(MAYA);
    h.mark_retracted(maya, None);

    owner_says(
        &h,
        "My dentist appointment is on 8 October. My daughter is called Maya.",
    );
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

#[test]
fn a_hit_on_a_refined_memory_shows_its_chain_head() {
    let h = Harness::new();
    let japan = h.insert_memory("main", JAPAN, "event", "notable");
    let tokyo = h.insert_memory("main", TOKYO, "event", "notable");
    h.mark_refined(japan, tokyo);
    owner_says(&h, "I'm going to Japan in 2027.");
    let call1 = reply(vec![claim(JAPAN, "event", "I'm going to Japan in 2027")]);
    let input = call2(&h, &call1).expect("call 2 runs");
    let found = shown(&input);
    assert!(found.contains(&tokyo), "{found:?}");
    assert!(!found.contains(&japan), "{found:?}");
}

/// sqlite-vec 0.1.9 refuses a KNN query for more than this many neighbours
/// (`SQLITE_VEC_VEC0_K_MAX`).
const KNN_K_MAX: usize = 4096;

#[test]
#[ignore = "pending fix (TIM-108 re-review): vector search asks sqlite-vec for more than 4,096 neighbours"]
fn a_chain_past_the_knn_limit_doesnt_abort_extraction() {
    // 2,600 versions of one fact all clear the floor and collapse to one
    // head, so the search keeps asking for more: 2,560 hits still leave it
    // short, and the next request would be 5,120, past what sqlite-vec
    // allows. Reconciliation has to stay within the limit and still finish,
    // or the chunk fails instead of becoming an access.
    let h = Harness::new();
    let versions = 2_600;
    assert!(versions > 2_560 && 2 * 2_560 > KNN_K_MAX);
    let head = h.insert_chain(TEA, versions);
    let source = owner_says(&h, "I like green tea.");

    let call1 = reply(vec![claim(TEA, "fact", "I like green tea")]);
    let input = call2(&h, &call1).expect("call 2 runs");
    assert_eq!(shown_for(&input, 0), BTreeSet::from([head]));
    let extracted = one_label(&h, call1, head, "mentioned_again");
    assert!(extracted.memories.is_empty());
    let last = h.accesses(head).pop().unwrap();
    assert_eq!(last.kind, "mentioned_again");
    assert_eq!(last.source, Some(source));
}

#[test]
#[ignore = "pending fix (TIM-108 re-review): vector search can't see past sqlite-vec's 4,096-neighbour limit"]
fn a_memory_past_the_knn_limit_is_still_a_neighbour() {
    // More versions of one fact than sqlite-vec returns from one KNN query
    // all sit nearer the claim than another matching memory. Stopping at the
    // limit would leave that memory out of call 2 and bring back the
    // crowding the 22-version regression covers, just later (ADR 0005).
    let h = Harness::new();
    let head = h.insert_chain(TEA, KNN_K_MAX + 4);
    let lot = h.fact(TEA_A_LOT);
    owner_says(&h, "I like green tea.");
    let input =
        call2(&h, &reply(vec![claim(TEA, "fact", "I like green tea")])).expect("call 2 runs");
    assert_eq!(shown_for(&input, 0), BTreeSet::from([head, lot]));
}

#[test]
fn a_long_chain_doesnt_crowd_out_another_neighbour() {
    // The top five are distinct shown memories, not raw hits: 22 versions of
    // one refined memory collapse to its head, and the next memory that
    // matches still reaches call 2, or a repeat of it would become a
    // duplicate rather than an access (ADR 0005).
    let h = Harness::new();
    let versions: Vec<Uuid> = (0..22).map(|_| h.fact(TEA)).collect();
    for pair in versions.windows(2) {
        h.mark_refined(pair[0], pair[1]);
    }
    let head = versions[versions.len() - 1];
    let lot = h.fact(TEA_A_LOT);
    owner_says(&h, "I like green tea.");
    let input =
        call2(&h, &reply(vec![claim(TEA, "fact", "I like green tea")])).expect("call 2 runs");
    assert_eq!(shown_for(&input, 0), BTreeSet::from([head, lot]));
}

#[test]
fn another_banks_memories_are_never_neighbours() {
    let h = Harness::new();
    h.insert_memory("other", TEA, "fact", "minor");
    owner_says(&h, "I like green tea.");
    let call1 = reply(vec![claim(TEA, "fact", "I like green tea")]);
    assert_eq!(call2(&h, &call1), None);
}

#[test]
fn the_request_is_the_input_and_the_call_2_schema() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    owner_says(&h, "I really like green tea.");
    let call1 = reply(vec![claim(TEA_AGAIN, "fact", "I really like green tea")]);
    let input = call2(&h, &call1).expect("call 2 runs");

    let llm = FakeLlm::scripted(MODEL, vec![call1, call2_reply(vec![])]);
    h.service
        .extract_chunk(lease(&h, "main"), &llm, &[])
        .unwrap();
    let requests = llm.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1], call2_request(&input));

    let request = &requests[1];
    assert_eq!(request.template.name, CALL2_TEMPLATE);
    assert_eq!(request.template.version, CALL2_VERSION);
    assert!(request.user.contains(TEA_AGAIN));
    assert!(request.user.contains(TEA));
    assert!(request.user.contains(&neighbour_handle(&input, tea)));
    // Code decides direction from observed_at, so the prompt never asks the
    // LLM which of the two is newer (ADR 0005).
    let labels: BTreeSet<String> = request.schema["properties"]["claims"]["items"]["properties"]
        ["labels"]["items"]["properties"]["label"]["enum"]
        .as_array()
        .expect("the label is an enum")
        .iter()
        .map(|label| label.as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        labels,
        Label::ALL
            .iter()
            .map(|label| label.as_str().to_owned())
            .collect()
    );
}

// Labels from a newer claim.

#[test]
fn mentioned_again_writes_an_access_and_no_memory() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    let source = owner_says(&h, "I like green tea.");
    let extracted = one_label(
        &h,
        reply(vec![claim(TEA, "fact", "I like green tea")]),
        tea,
        "mentioned_again",
    );

    assert!(extracted.memories.is_empty());
    assert_eq!(h.memories_in("main"), 1);
    // At the source's ingested_at and turn, like every access extraction
    // writes (TIM-92).
    assert_eq!(
        h.accesses(tea),
        vec![
            Harness::fixture_access(),
            AccessRow {
                kind: "mentioned_again".into(),
                at: h.now(),
                turn: h.turns("main"),
                source: Some(source),
            },
        ]
    );
    assert_eq!(h.change(tea), Change::untouched());
    let extracted_at: Option<i64> = h.chunk_column(extracted.chunk, "extracted_at");
    assert!(extracted_at.is_some());
}

#[test]
fn confirmed_writes_a_confirmed_access() {
    let h = Harness::new();
    let acme = h.fact(ACME);
    owner_says(&h, "Yes, I still work at Acme.");
    let extracted = one_label(
        &h,
        reply(vec![claim(ACME_STILL, "fact", "I still work at Acme")]),
        acme,
        "confirmed",
    );
    assert!(extracted.memories.is_empty());
    let kinds: Vec<String> = h.accesses(acme).into_iter().map(|row| row.kind).collect();
    assert_eq!(kinds, ["created", "confirmed"]);
}

#[test]
fn two_labels_on_one_neighbour_keep_the_strongest_access() {
    // At most one access per memory per turn, keeping the strongest kind
    // (TIM-90): confirmed weighs 2, mentioned again 1.5.
    let h = Harness::new();
    let tea = h.fact(TEA);
    owner_says(&h, "I like green tea. Yes, I really like green tea.");
    let call1 = reply(vec![
        claim(TEA, "fact", "I like green tea"),
        claim(TEA_AGAIN, "fact", "I really like green tea"),
    ]);
    let input = call2(&h, &call1).expect("call 2 runs");
    let n = neighbour_handle(&input, tea);
    let extracted = reconcile(
        &h,
        call1,
        call2_reply(vec![
            labelled(&input.claims[0].handle, &[(n.clone(), "mentioned_again")]),
            labelled(&input.claims[1].handle, &[(n, "confirmed")]),
        ]),
    );
    assert!(extracted.memories.is_empty());
    let kinds: Vec<String> = h.accesses(tea).into_iter().map(|row| row.kind).collect();
    assert_eq!(kinds, ["created", "confirmed"]);
}

#[test]
fn mentioned_again_outranks_used_in_the_same_turn() {
    // The reply relied on the memory and the user restated it in the same
    // turn: one access, and it's the heavier mention, whichever is written
    // first.
    let h = Harness::new();
    let tea = h.fact(TEA);
    owner_says(&h, "I like green tea.");
    let used = {
        let lease = lease(&h, "main");
        let input = h.service.call1_input(&lease, &[tea]).unwrap();
        input.in_context[0].handle.clone()
    };
    let call1 = json!({
        "claims": [claim(TEA, "fact", "I like green tea")],
        "used_injected_ids": [used],
    });
    let input = call2_with(&h, &call1, &[tea]).expect("call 2 runs");
    let extracted = reconcile_with(
        &h,
        call1,
        call2_reply(vec![labelled(
            &input.claims[0].handle,
            &[(neighbour_handle(&input, tea), "mentioned_again")],
        )]),
        &[tea],
    );
    assert_eq!(extracted.used, vec![tea]);
    let kinds: Vec<String> = h.accesses(tea).into_iter().map(|row| row.kind).collect();
    assert_eq!(kinds, ["created", "mentioned_again"]);
}

#[test]
fn a_documents_mention_of_another_documents_memory_writes_an_access() {
    // A document independently restating something is mentioned again
    // (CONTEXT.md). Documents share the turn number of the turn before them,
    // so two documents with no turn between them carry the same number, and
    // the second's access must still land rather than collide with the
    // first's created access.
    let h = Harness::new();
    ingest_doc(
        &h,
        &document("notes", "My bike is a Brompton.", date(2026, 9, 20)),
    );
    let bike = extract_alone(
        &h,
        reply(vec![claim(BIKE, "fact", "My bike is a Brompton")]),
    )
    .memories[0];
    h.advance(1);
    let diary = ingest_doc(
        &h,
        &document("diary", "Rode my bike, a Brompton.", date(2026, 9, 25)),
    );
    let extracted = one_label(
        &h,
        reply(vec![claim(BIKE, "fact", "my bike, a Brompton")]),
        bike,
        "mentioned_again",
    );
    assert!(extracted.memories.is_empty());
    let accesses = h.accesses(bike);
    assert_eq!(accesses.len(), 2, "{accesses:?}");
    assert_eq!(accesses[0].kind, "created");
    assert_eq!(accesses[1].kind, "mentioned_again");
    assert_eq!(accesses[1].source, Some(diary.source));
}

#[test]
fn mentioned_again_raises_significance_to_the_larger() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    let acme = h.insert_memory("main", ACME, "fact", "major");
    owner_says(&h, "I like green tea. I work at Acme.");
    let call1 = reply(vec![
        claim(TEA, "fact", "I like green tea").with("significance", json!("notable")),
        claim(ACME, "fact", "I work at Acme").with("significance", json!("trivial")),
    ]);
    let input = call2(&h, &call1).expect("call 2 runs");
    reconcile(
        &h,
        call1,
        call2_reply(vec![
            labelled(
                &input.claims[0].handle,
                &[(neighbour_handle(&input, tea), "mentioned_again")],
            ),
            labelled(
                &input.claims[1].handle,
                &[(neighbour_handle(&input, acme), "mentioned_again")],
            ),
        ]),
    );

    // TIM-92: the larger of the existing and the new score, logged.
    assert_eq!(h.significance(tea), ("notable".into(), None));
    assert_eq!(h.edits_on(tea, EDIT_SIGNIFICANCE_RAISED), 1);
    // A weaker mention never lowers it, and there's nothing to log.
    assert_eq!(h.significance(acme), ("major".into(), None));
    assert_eq!(h.all_edits_on(acme), 0);
}

#[test]
fn mentioned_again_never_changes_a_significance_the_owner_set() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    h.set_owner_significance(tea, "trivial");
    owner_says(&h, "I like green tea.");
    one_label(
        &h,
        reply(vec![
            claim(TEA, "fact", "I like green tea").with("significance", json!("critical")),
        ]),
        tea,
        "mentioned_again",
    );
    assert_eq!(
        h.significance(tea),
        ("minor".into(), Some("trivial".into()))
    );
    assert_eq!(h.edits_on(tea, EDIT_SIGNIFICANCE_RAISED), 0);
    // The access still counts.
    assert_eq!(h.accesses(tea).len(), 2);
}

#[test]
fn remember_this_on_a_mention_keeps_the_neighbour() {
    // TIM-92: remember-this goes on the neighbour when the label is
    // mentioned again or confirmed, and only from the owner's own message.
    let h = Harness::new();
    let tea = h.fact(TEA);
    owner_says(&h, "Remember this: I like green tea.");
    let extracted = one_label(
        &h,
        reply(vec![
            claim(TEA, "fact", "I like green tea").with("remember_this", json!(true)),
        ]),
        tea,
        "mentioned_again",
    );
    assert!(extracted.memories.is_empty());
    assert_eq!(h.significance(tea), ("minor".into(), Some("kept".into())));
    assert_eq!(h.edits_on(tea, EDIT_KEPT), 1);
}

#[test]
fn remember_this_in_a_document_keeps_nothing() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    ingest_doc(
        &h,
        &document(
            "notes",
            "Remember this: I like green tea.",
            date(2026, 9, 25),
        ),
    );
    one_label(
        &h,
        reply(vec![
            claim(TEA, "fact", "I like green tea").with("remember_this", json!(true)),
        ]),
        tea,
        "mentioned_again",
    );
    assert_eq!(h.significance(tea), ("minor".into(), None));
    assert_eq!(h.edits_on(tea, EDIT_KEPT), 0);
    assert_eq!(h.accesses(tea).len(), 2);
}

#[test]
fn a_reschedule_retracts_the_old_appointment() {
    let h = Harness::new();
    let old = h.insert_memory("main", DENTIST_8, "event", "minor");
    h.set_valid_from(old, local("2026-10-08T00:00"), "day");
    owner_says(&h, "The dentist moved my appointment to Friday 9 October.");
    let extracted = one_label(
        &h,
        reply(vec![changes(
            claim(
                DENTIST_9,
                "event",
                "moved my appointment to Friday 9 October",
            )
            .with("valid_from", time("2026-10-09", "day")),
        )]),
        old,
        "retracts",
    );

    let new = extracted.memories[0];
    assert_eq!(h.content(new), DENTIST_9);
    assert_eq!(h.valid_from(new), timed(local("2026-10-09T00:00"), "day"));
    // TIM-90: a reschedule is a retraction, which sets invalidated_at (to
    // the retracting claim's observed_at, TIM-92) and superseded_by, and
    // leaves the old window alone.
    assert_eq!(
        h.change(old),
        Change {
            invalidated_at: Some(at(T1)),
            superseded_by: Some(new),
            ..Change::untouched()
        }
    );
    assert_eq!(h.valid_from(old), timed(local("2026-10-08T00:00"), "day"));
    assert_eq!(h.edits_on(old, EDIT_RETRACTED), 1);
    assert_eq!(h.change(new), Change::untouched());
    // The new appointment has its own created access; the old one gets none.
    assert_eq!(h.accesses(new).len(), 1);
    assert_eq!(h.accesses(old), vec![Harness::fixture_access()]);
}

#[test]
fn a_refinement_supersedes_without_retracting() {
    let h = Harness::new();
    let japan = h.insert_memory("main", JAPAN, "event", "notable");
    h.set_valid_from(japan, local("2027-01-01T00:00"), "year");
    owner_says(&h, "Remember this: it's Tokyo, in April 2027.");
    let extracted = one_label(
        &h,
        reply(vec![
            claim(TOKYO, "event", "it's Tokyo, in April 2027")
                .with("valid_from", time("2027-04", "month"))
                .with("significance", json!("notable"))
                .with("remember_this", json!(true)),
        ]),
        japan,
        "refines",
    );

    let tokyo = extracted.memories[0];
    // TIM-90: refined sets superseded_by only. The old one wasn't wrong.
    assert_eq!(
        h.change(japan),
        Change {
            superseded_by: Some(tokyo),
            ..Change::untouched()
        }
    );
    assert_eq!(h.edits_on(japan, EDIT_REFINED), 1);
    // TIM-92: on refines, remember-this goes on the new memory.
    assert_eq!(
        h.significance(tokyo),
        ("notable".into(), Some("kept".into()))
    );
    assert_eq!(h.significance(japan), ("notable".into(), None));
}

#[test]
fn an_ending_sets_valid_until_and_ended_by() {
    let h = Harness::new();
    let berlin = h.fact(BERLIN);
    owner_says(
        &h,
        "I moved out of Berlin on 12 September and now live in Lisbon.",
    );
    let extracted = one_label(
        &h,
        reply(vec![changes(
            claim(
                MOVED,
                "event",
                "I moved out of Berlin on 12 September and now live in Lisbon",
            )
            .with("valid_from", time("2026-09-12", "day")),
        )]),
        berlin,
        "ends",
    );

    let moved = extracted.memories[0];
    // TIM-92: valid_until is the ending memory's valid_from, with its
    // precision. Ended isn't retracted or superseded: Berlin stays in
    // history and inherits nothing.
    assert_eq!(
        h.change(berlin),
        Change {
            valid_until: timed(local("2026-09-12T00:00"), "day"),
            ended_by: Some(moved),
            ..Change::untouched()
        }
    );
    assert_eq!(h.edits_on(berlin, EDIT_ENDED), 1);
    assert_eq!(h.change(moved), Change::untouched());
    assert_eq!(h.accesses(berlin), vec![Harness::fixture_access()]);
}

#[test]
fn a_late_reported_ending_takes_the_ending_memorys_start() {
    // Said on 1 October, about August: the window closes in August, and
    // strength's restart is at the later of that and when the end became
    // known, which the store reads from ended_by (TIM-91 decision 2).
    let h = Harness::new();
    let acme = h.fact(ACME);
    owner_says(&h, "I left Acme back in August.");
    let extracted = one_label(
        &h,
        reply(vec![changes(
            claim(ACME_LEFT, "event", "I left Acme back in August")
                .with("valid_from", time("2026-08", "month")),
        )]),
        acme,
        "ends",
    );

    let left = extracted.memories[0];
    assert_eq!(
        h.change(acme),
        Change {
            valid_until: timed(local("2026-08-01T00:00"), "month"),
            ended_by: Some(left),
            ..Change::untouched()
        }
    );
    let known_at: i64 = h.one(
        "SELECT e.observed_at FROM memories m JOIN memories e ON e.id = m.ended_by
         WHERE m.uuid = ?1",
        [acme.to_string()],
    );
    assert_eq!(timestamp(known_at), at(T1));
    assert_eq!(h.edits_on(acme, EDIT_ENDED), 1);
}

#[test]
fn an_ending_with_no_start_ends_on_the_day_it_was_said_with_low_confidence() {
    // TIM-92: with no start on the ending memory, valid_until is its
    // observed_at with low window confidence. It's at day precision, like an
    // event with no stated time, so the instant is the start of
    // that day in the source's timezone.
    let h = Harness::new();
    let coffee = h.fact(COFFEE);
    owner_says(&h, "I don't drink coffee any more.");
    let extracted = one_label(
        &h,
        reply(vec![changes(claim(
            NO_COFFEE,
            "fact",
            "I don't drink coffee any more",
        ))]),
        coffee,
        "ends",
    );
    let stopped = extracted.memories[0];
    assert_eq!(h.valid_from(stopped), None, "a fact gets no stated start");
    assert_eq!(
        h.change(coffee),
        Change {
            valid_until: timed(local("2026-10-01T00:00"), "day"),
            window_confidence: "low".into(),
            ended_by: Some(stopped),
            ..Change::untouched()
        }
    );
}

#[test]
fn completing_a_task_creates_an_event_that_ends_it() {
    let h = Harness::new();
    let task = h.insert_memory("main", TAX_TASK, "task", "notable");
    owner_says(&h, "I filed the tax return.");
    let extracted = one_label(
        &h,
        reply(vec![changes(claim(
            TAX_FILED,
            "event",
            "I filed the tax return",
        ))]),
        task,
        "ends",
    );

    // TIM-92: a completion is an event, never a retraction. With no stated
    // time it starts on the day it was said, and the task ends there.
    let filed = extracted.memories[0];
    assert_eq!(h.kind(filed), "event");
    assert_eq!(h.valid_from(filed), timed(local("2026-10-01T00:00"), "day"));
    let change = h.change(task);
    assert_eq!(change.valid_until, timed(local("2026-10-01T00:00"), "day"));
    assert_eq!(change.ended_by, Some(filed));
    assert_eq!(change.invalidated_at, None);
    assert_eq!(change.superseded_by, None);
    assert_eq!(h.edits_on(task, EDIT_ENDED), 1);
}

#[test]
fn maya_corrected_to_mia_keeps_the_strength_maya_had() {
    // TIM-91 decision 3: a successor inherits every access along
    // superseded_by, so correcting the name keeps the corrected one as
    // strong as the wrong one was.
    let h = Harness::new();
    let maya = h.insert_memory("main", MAYA, "fact", "critical");
    h.insert_access(maya, "mentioned_again", 1, at("2026-09-10T00:00:00Z"));
    h.insert_access(maya, "confirmed", 2, at("2026-09-20T00:00:00Z"));
    let before = h.strength_of(maya);

    owner_says(&h, "Sorry, my daughter is called Mia, not Maya.");
    let extracted = one_label(
        &h,
        reply(vec![changes(
            claim(MIA, "fact", "my daughter is called Mia, not Maya")
                .with("significance", json!("critical")),
        )]),
        maya,
        "retracts",
    );

    let mia = extracted.memories[0];
    assert_eq!(h.content(mia), MIA);
    assert_eq!(
        h.change(maya),
        Change {
            invalidated_at: Some(at(T1)),
            superseded_by: Some(mia),
            ..Change::untouched()
        }
    );
    assert_eq!(h.edits_on(maya, EDIT_RETRACTED), 1);
    assert!(
        h.strength_of(mia) >= before,
        "Mia {} against Maya's {before}",
        h.strength_of(mia)
    );

    // Maya is hidden from now on: a later mention of the old name finds Mia.
    h.service
        .ingest_turn(
            "main",
            &turn(
                "s2",
                "2026-10-02T06:30:00Z",
                "My daughter is called Maya.",
                "Mia?",
            ),
        )
        .unwrap();
    let input = call2(
        &h,
        &reply(vec![claim(MAYA, "fact", "My daughter is called Maya")]),
    )
    .expect("call 2 runs");
    let found = shown(&input);
    assert!(found.contains(&mia), "{found:?}");
    assert!(!found.contains(&maya), "{found:?}");
}

#[test]
fn labels_on_an_ended_neighbour_are_rejected() {
    // TIM-92: code rejects any label on a neighbour that's already ended. A
    // claim left with no labels is new.
    let h = Harness::new();
    let acme = h.fact(ACME);
    let left = h.insert_memory("main", ACME_LEFT, "event", "minor");
    h.mark_ended(acme, left, local("2026-08-01T00:00"), "month");
    let ended = h.change(acme);

    owner_says(&h, "I work at Acme. I still work at Acme.");
    let call1 = reply(vec![
        claim(ACME, "fact", "I work at Acme"),
        changes(claim(ACME_STILL, "fact", "I still work at Acme")),
    ]);
    let input = call2(&h, &call1).expect("call 2 runs");
    let n = neighbour_handle(&input, acme);
    let extracted = reconcile(
        &h,
        call1,
        call2_reply(vec![
            labelled(&input.claims[0].handle, &[(n.clone(), "mentioned_again")]),
            labelled(&input.claims[1].handle, &[(n, "ends")]),
        ]),
    );

    assert_eq!(extracted.memories.len(), 2);
    assert_eq!(h.change(acme), ended);
    assert_eq!(h.accesses(acme), vec![Harness::fixture_access()]);
    assert_eq!(h.all_edits_on(acme), 0);
}

#[test]
fn a_label_on_an_unknown_neighbour_is_ignored() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    owner_says(&h, "I like green tea.");
    let call1 = reply(vec![claim(TEA, "fact", "I like green tea")]);
    let input = call2(&h, &call1).expect("call 2 runs");
    let extracted = reconcile(
        &h,
        call1,
        call2_reply(vec![labelled(
            &input.claims[0].handle,
            &[("n99".into(), "mentioned_again")],
        )]),
    );
    assert_eq!(extracted.memories.len(), 1);
    assert_eq!(h.accesses(tea), vec![Harness::fixture_access()]);
}

// Direction: an older claim.

#[test]
fn an_older_claim_arriving_after_a_newer_one_is_created_already_ended() {
    // ADR 0005: code, not the LLM, decides which is newer, so an old
    // document can't overrule a newer memory. An older claim labelled ends
    // is created ended by the neighbour (TIM-92).
    let h = Harness::new();
    let lisbon = h.fact(LISBON);
    h.set_observed_at(lisbon, at("2026-09-20T00:00:00Z"));
    h.set_valid_from(lisbon, local("2026-09-12T00:00"), "day");
    ingest_doc(
        &h,
        &document("old-notes", "I live in Berlin.", date(2025, 3, 1)),
    );
    let extracted = one_label(
        &h,
        reply(vec![claim(BERLIN, "fact", "I live in Berlin")]),
        lisbon,
        "ends",
    );

    let berlin = extracted.memories[0];
    assert_eq!(h.content(berlin), BERLIN);
    assert_eq!(
        h.change(berlin),
        Change {
            valid_until: timed(local("2026-09-12T00:00"), "day"),
            ended_by: Some(lisbon),
            ..Change::untouched()
        }
    );
    assert_eq!(h.change(lisbon), Change::untouched());
    assert_eq!(h.all_edits_on(lisbon), 0);
}

#[test]
fn an_older_claim_ending_a_neighbour_with_no_start_ends_at_its_observed_at() {
    // The neighbour has no start, so the older claim ends where the
    // neighbour was said, with low confidence (TIM-92, "this applies in both
    // directions"). It's at day precision, as for a newer claim.
    let h = Harness::new();
    let lisbon = h.fact(LISBON);
    h.set_observed_at(lisbon, local("2026-09-20T09:00"));
    ingest_doc(
        &h,
        &document("old-notes", "I live in Berlin.", date(2025, 3, 1)),
    );
    let berlin = one_label(
        &h,
        reply(vec![claim(BERLIN, "fact", "I live in Berlin")]),
        lisbon,
        "ends",
    )
    .memories[0];
    assert_eq!(
        h.change(berlin),
        Change {
            valid_until: timed(local("2026-09-20T00:00"), "day"),
            window_confidence: "low".into(),
            ended_by: Some(lisbon),
            ..Change::untouched()
        }
    );
}

#[test]
fn an_older_retraction_or_refinement_creates_nothing() {
    let h = Harness::new();
    let dentist = h.insert_memory("main", DENTIST_9, "event", "minor");
    let tokyo = h.insert_memory("main", TOKYO, "event", "notable");
    h.set_observed_at(dentist, at("2026-09-25T00:00:00Z"));
    h.set_observed_at(tokyo, at("2026-09-25T00:00:00Z"));
    ingest_doc(
        &h,
        &document(
            "old-notes",
            "Dentist on 8 October. Going to Japan in 2027.",
            date(2026, 9, 15),
        ),
    );
    let call1 = reply(vec![
        claim(DENTIST_8, "event", "Dentist on 8 October"),
        claim(JAPAN, "event", "Going to Japan in 2027"),
    ]);
    let input = call2(&h, &call1).expect("call 2 runs");
    let extracted = reconcile(
        &h,
        call1,
        call2_reply(vec![
            labelled(
                &input.claims[0].handle,
                &[(neighbour_handle(&input, dentist), "retracts")],
            ),
            labelled(
                &input.claims[1].handle,
                &[(neighbour_handle(&input, tokyo), "refines")],
            ),
        ]),
    );

    assert!(extracted.memories.is_empty());
    assert_eq!(h.memories_in("main"), 2);
    for memory in [dentist, tokyo] {
        assert_eq!(h.change(memory), Change::untouched());
        assert_eq!(h.accesses(memory), vec![Harness::fixture_access()]);
        assert_eq!(h.all_edits_on(memory), 0);
    }
}

#[test]
fn an_older_mention_writes_its_access_even_on_an_ended_neighbour() {
    let h = Harness::new();
    let acme = h.fact(ACME);
    let left = h.insert_memory("main", ACME_LEFT, "event", "minor");
    h.mark_ended(acme, left, local("2026-08-01T00:00"), "month");
    let ended = h.change(acme);
    let doc = ingest_doc(&h, &document("cv", "I work at Acme.", date(2025, 6, 1)));
    let extracted = one_label(
        &h,
        reply(vec![claim(ACME, "fact", "I work at Acme")]),
        acme,
        "mentioned_again",
    );

    assert!(extracted.memories.is_empty());
    assert_eq!(h.change(acme), ended);
    let accesses = h.accesses(acme);
    assert_eq!(accesses.len(), 2);
    assert_eq!(accesses[1].kind, "mentioned_again");
    assert_eq!(accesses[1].source, Some(doc.source));
}

#[test]
fn a_tie_on_observed_at_goes_to_the_later_ingest() {
    // TIM-92: ties on observed_at are broken by the later ingested_at, then
    // rowid. The fixture's source was ingested an hour before the turn, so
    // the claim is the newer and ends the neighbour.
    let h = Harness::new();
    let berlin = h.fact(BERLIN);
    h.set_observed_at(berlin, at(T1));
    h.advance(1);
    owner_says(&h, "I moved out of Berlin and now live in Lisbon.");
    let moved = one_label(
        &h,
        reply(vec![changes(claim(
            MOVED,
            "event",
            "I moved out of Berlin and now live in Lisbon",
        ))]),
        berlin,
        "ends",
    )
    .memories[0];
    assert_eq!(h.change(berlin).ended_by, Some(moved));
    // TIM-92: an event with no stated time starts on the day it was said,
    // with low window confidence. Nothing ends or supersedes it.
    assert_eq!(
        h.change(moved),
        Change {
            window_confidence: "low".into(),
            ..Change::untouched()
        }
    );
}

// Edited documents.

#[test]
fn a_later_version_of_a_document_doesnt_reinforce_itself() {
    // TIM-92: when the neighbour comes from an earlier version of the same
    // document id, mentioned again and confirmed write no access.
    let h = Harness::new();
    ingest_doc(
        &h,
        &document("notes", "My bike is a Brompton.", date(2026, 9, 20)),
    );
    let bike = extract_alone(
        &h,
        reply(vec![claim(BIKE, "fact", "My bike is a Brompton")]),
    )
    .memories[0];

    h.advance(1);
    let edited = ingest_doc(
        &h,
        &document(
            "notes",
            "My bike is a Brompton. I ride it to work.",
            date(2026, 9, 27),
        ),
    );
    assert_eq!(edited.chunks_queued, 1);
    let extracted = one_label(
        &h,
        reply(vec![claim(BIKE, "fact", "My bike is a Brompton")]),
        bike,
        "mentioned_again",
    );
    assert!(extracted.memories.is_empty());
    assert_eq!(h.accesses(bike).len(), 1, "only its created access");
}

// Reopening.

#[test]
fn retracting_the_memory_that_ended_another_repoints_its_end() {
    // TIM-92, "Reopening": when the memory that ended another is retracted
    // with a successor, ended_by is repointed to the successor and
    // valid_until is taken from the successor's window.
    let h = Harness::new();
    let task = h.insert_memory("main", TAX_TASK, "task", "notable");
    let filed = h.insert_memory("main", TAX_FILED, "event", "minor");
    h.set_valid_from(filed, local("2026-10-01T00:00"), "day");
    h.mark_ended(task, filed, local("2026-10-01T00:00"), "day");
    h.advance(24);
    h.service
        .ingest_turn(
            "main",
            &turn(
                "s1",
                "2026-10-02T06:30:00Z",
                "Correction: I filed the tax return on 2 October, not the 1st.",
                "Noted.",
            ),
        )
        .unwrap();
    let extracted = one_label(
        &h,
        reply(vec![changes(
            claim(
                TAX_FILED_LATER,
                "event",
                "I filed the tax return on 2 October",
            )
            .with("valid_from", time("2026-10-02", "day")),
        )]),
        filed,
        "retracts",
    );

    let later = extracted.memories[0];
    assert_eq!(h.change(filed).superseded_by, Some(later));
    assert_eq!(
        h.change(task),
        Change {
            valid_until: timed(local("2026-10-02T00:00"), "day"),
            ended_by: Some(later),
            ..Change::untouched()
        }
    );
    assert_eq!(h.edits_on(task, EDIT_END_REPOINTED), 1);
}

#[test]
fn a_correction_of_another_kind_still_repoints_the_end() {
    // TIM-92, "Reopening": with a successor, ended_by is repointed. The
    // correction below is filed as a fact rather than an event, but it still
    // supersedes the memory that ended the task, so it is a successor and
    // the task stays ended.
    let h = Harness::new();
    let task = h.insert_memory("main", TAX_TASK, "task", "notable");
    let filed = h.insert_memory("main", TAX_FILED, "event", "minor");
    h.set_valid_from(filed, local("2026-10-01T00:00"), "day");
    h.mark_ended(task, filed, local("2026-10-01T00:00"), "day");
    h.advance(24);
    h.service
        .ingest_turn(
            "main",
            &turn(
                "s1",
                "2026-10-02T06:30:00Z",
                "Correction: I filed the tax return on 2 October, not the 1st.",
                "Noted.",
            ),
        )
        .unwrap();
    let later = one_label(
        &h,
        reply(vec![changes(
            claim(
                TAX_FILED_LATER,
                "fact",
                "I filed the tax return on 2 October",
            )
            .with("valid_from", time("2026-10-02", "day")),
        )]),
        filed,
        "retracts",
    )
    .memories[0];

    assert_eq!(h.kind(later), "fact");
    assert_eq!(h.change(filed).superseded_by, Some(later));
    assert_eq!(
        h.change(task),
        Change {
            valid_until: timed(local("2026-10-02T00:00"), "day"),
            ended_by: Some(later),
            ..Change::untouched()
        }
    );
    assert_eq!(h.edits_on(task, EDIT_END_REPOINTED), 1);
    assert_eq!(h.all_edits_on(task), 1);
}

// Mental models.

#[test]
fn a_refinement_moves_mental_model_citations_to_the_head() {
    // TIM-95 decision 6: reconcile moves a citation of a refined memory to
    // the head of its chain, where its accesses are inherited.
    let h = Harness::new();
    let japan = h.insert_memory("main", JAPAN, "event", "notable");
    let now = micros(h.now());
    h.execute(
        "INSERT INTO mental_models (uuid, bank_id, name, question, max_tokens, created_at,
                                    updated_at)
         VALUES (?1, ?2, 'Travel', 'Where is the user going?', 200, ?3, ?3)",
        (next_uuid().to_string(), h.bank_id("main"), now),
    );
    h.execute(
        "INSERT INTO mental_model_entries (uuid, model_id, position, text, created_at, updated_at)
         SELECT ?1, id, 0, 'Tim is going to Japan in 2027.', ?2, ?2
         FROM mental_models WHERE name = 'Travel'",
        (next_uuid().to_string(), now),
    );
    h.execute(
        "INSERT INTO mental_model_citations (entry_id, memory_id)
         SELECT e.id, m.id FROM mental_model_entries e, memories m WHERE m.uuid = ?1",
        [japan.to_string()],
    );

    owner_says(&h, "It's Tokyo, in April 2027.");
    let tokyo = one_label(
        &h,
        reply(vec![
            claim(TOKYO, "event", "It's Tokyo, in April 2027")
                .with("valid_from", time("2027-04", "month")),
        ]),
        japan,
        "refines",
    )
    .memories[0];

    let cited: Vec<String> = h.all(
        "SELECT m.uuid FROM mental_model_citations c JOIN memories m ON m.id = c.memory_id",
        [],
    );
    assert_eq!(cited, vec![tokyo.to_string()]);
}

// The version 4 migration.

/// One access row, every column.
type AccessColumns = (i64, i64, i64, String, i64, i64, Option<i64>);

fn access_rows(h: &Harness) -> Vec<AccessColumns> {
    let store = h.service.store().unwrap();
    let conn = store.connection();
    let mut statement = conn
        .prepare(
            "SELECT id, bank_id, memory_id, kind, at, turn, source_id FROM accesses ORDER BY id",
        )
        .unwrap();
    statement
        .query_map([], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
            ))
        })
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

/// Puts the store back to schema version 3, as the version 3 binary would
/// have left it: `accesses` with version 1's `UNIQUE (memory_id, turn)` and
/// every row and id as they are, and one `migrations` row 0 to 3. Reopening
/// migrates it to version 4.
fn downgrade_accesses_to_v3(h: &Harness) {
    h.service
        .store()
        .unwrap()
        .connection()
        .execute_batch(
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
        )
        .unwrap();
}

/// A version 3 store holding a turn's mention, a document's mention and a
/// used access whose source is gone, each in a turn of its own as version
/// 3's key required. Returns the harness still at version 3, the two
/// memories, and the rowids of the turn and the document.
fn populated_v3() -> (Harness, Uuid, Uuid, i64, i64) {
    let h = Harness::new();
    let tea = h.fact(TEA);
    let acme = h.fact(ACME);
    let turn_source = owner_says(&h, "I like green tea.");
    let doc_source =
        ingest_doc(&h, &document("notes", "I work at Acme.", date(2026, 9, 25))).source;
    let source_id =
        |uuid: Uuid| -> i64 { h.one("SELECT id FROM sources WHERE uuid = ?1", [uuid.to_string()]) };
    let (turn_id, doc_id) = (source_id(turn_source), source_id(doc_source));
    insert_raw_access(&h, tea, "mentioned_again", 2, Some(turn_id)).unwrap();
    insert_raw_access(&h, acme, "mentioned_again", 1, Some(doc_id)).unwrap();
    insert_raw_access(&h, tea, "used", 3, None).unwrap();
    insert_raw_access(&h, acme, "confirmed", 4, Some(turn_id)).unwrap();
    downgrade_accesses_to_v3(&h);
    (h, tea, acme, turn_id, doc_id)
}

fn insert_raw_access(
    h: &Harness,
    memory: Uuid,
    kind: &str,
    turn: i64,
    source: Option<i64>,
) -> rusqlite::Result<usize> {
    h.service.store().unwrap().connection().execute(
        "INSERT INTO accesses (bank_id, memory_id, kind, at, turn, source_id)
         SELECT bank_id, id, ?2, ?3, ?4, ?5 FROM memories WHERE uuid = ?1",
        (memory.to_string(), kind, micros(h.now()), turn, source),
    )
}

#[test]
fn a_populated_version_3_store_keeps_its_accesses_through_the_migration() {
    let (h, tea, _acme, turn_id, doc_id) = populated_v3();
    let before = access_rows(&h);
    assert_eq!(before.len(), 6, "two created accesses and four more");

    let h = h.restart();
    {
        let store = h.service.store().unwrap();
        let applied = store.applied().expect("version 3 is migrated");
        assert_eq!((applied.from, applied.to), (3, 4));
    }
    // Every row and id as it was.
    assert_eq!(access_rows(&h), before);
    let problems: Vec<String> = h.all("PRAGMA foreign_key_check", []);
    assert!(problems.is_empty(), "{problems:?}");
    let integrity: String = h.one("PRAGMA integrity_check", []);
    assert_eq!(integrity, "ok");

    // A turn keeps one access per memory per turn, with or without a source.
    assert!(insert_raw_access(&h, tea, "confirmed", 2, Some(turn_id)).is_err());
    assert!(insert_raw_access(&h, tea, "confirmed", 3, None).is_err());
    // A document's access in a turn some other source already used lands,
    // and a second one from the same document doesn't.
    insert_raw_access(&h, tea, "mentioned_again", 2, Some(doc_id)).unwrap();
    assert!(insert_raw_access(&h, tea, "confirmed", 2, Some(doc_id)).is_err());
}

#[test]
fn the_version_4_migration_never_reuses_an_access_id() {
    // Rowids are AUTOINCREMENT so they are never reused (the schema's
    // header). The newest access in the version 3 store is deleted, so only
    // AUTOINCREMENT remembers its id; the rebuilt table must not hand it out
    // again.
    let (h, tea, _acme, _turn_id, _doc_id) = populated_v3();
    insert_raw_access(&h, tea, "confirmed", 9, None).unwrap();
    let highest: i64 = h.one("SELECT MAX(id) FROM accesses", []);
    h.execute("DELETE FROM accesses WHERE id = ?1", [highest]);

    let h = h.restart();
    insert_raw_access(&h, tea, "confirmed", 10, None).unwrap();
    let next: i64 = h.one("SELECT MAX(id) FROM accesses", []);
    assert!(next > highest, "access id {next} reuses {highest} or below");
}

// Ordering and failures.

#[test]
fn call_2_and_the_commit_run_in_observed_at_order() {
    // The newer turn arrives first, but the bank's one worker reconciles the
    // older one first, so the newer one is reconciled against it.
    let h = Harness::new();
    h.service
        .ingest_turn(
            "main",
            &turn("s2", "2026-10-01T06:40:00Z", "I live in Berlin.", "Noted."),
        )
        .unwrap();
    let older = owner_says(&h, "I live in Berlin.");

    {
        let lease = lease(&h, "main");
        assert_eq!(lease.source, older);
    }
    let berlin =
        extract_alone(&h, reply(vec![claim(BERLIN, "fact", "I live in Berlin")])).memories[0];

    let input =
        call2(&h, &reply(vec![claim(BERLIN, "fact", "I live in Berlin")])).expect("call 2 runs");
    assert_eq!(shown(&input), BTreeSet::from([berlin]));
}

#[test]
fn a_failed_call_2_counts_and_keeps_call_1s_reply() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    owner_says(&h, "I like green tea.");
    let call1 = reply(vec![claim(TEA, "fact", "I like green tea")]);
    // Call 1 answers; call 2 gets nothing back.
    let llm = FakeLlm::scripted(MODEL, vec![call1]);
    let error = h
        .service
        .extract_chunk(lease(&h, "main"), &llm, &[])
        .unwrap_err();
    assert_eq!(llm.requests().len(), 2);
    assert_eq!(error.failure(), Some(Failure::Retry { error_count: 1 }));

    // Nothing committed but the count, and call 1's reply is saved so the
    // retry resumes from it (TIM-92).
    let chunk = {
        let lease = lease(&h, "main");
        lease.chunk
    };
    assert_eq!(h.memories_in("main"), 1);
    assert_eq!(h.accesses(tea), vec![Harness::fixture_access()]);
    let extracted_at: Option<i64> = h.chunk_column(chunk, "extracted_at");
    assert_eq!(extracted_at, None);
    let saved: Option<String> = h.chunk_column(chunk, "call1_output");
    assert!(saved.is_some());
}

#[test]
fn a_retry_resumes_from_call_1s_saved_reply_and_drops_it_on_commit() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    owner_says(&h, "I like green tea.");
    let call1 = reply(vec![claim(TEA, "fact", "I like green tea")]);
    let input = call2(&h, &call1).expect("call 2 runs");
    let first = FakeLlm::scripted(MODEL, vec![call1]);
    h.service
        .extract_chunk(lease(&h, "main"), &first, &[])
        .unwrap_err();

    // The retry is call 2 alone.
    let retry = FakeLlm::scripted(
        MODEL,
        vec![call2_reply(vec![labelled(
            &input.claims[0].handle,
            &[(neighbour_handle(&input, tea), "mentioned_again")],
        )])],
    );
    let extracted = h
        .service
        .extract_chunk(lease(&h, "main"), &retry, &[])
        .unwrap();
    let requests = retry.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0], call2_request(&input));
    assert!(extracted.memories.is_empty());
    assert_eq!(h.accesses(tea).len(), 2);

    // TIM-97: the saved reply goes when the chunk commits, so the claim text
    // can't outlive a purge or forget in the chunk row.
    let saved: Option<String> = h.chunk_column(extracted.chunk, "call1_output");
    assert_eq!(saved, None);
}

#[test]
fn an_invalid_call_2_reply_writes_nothing() {
    let h = Harness::new();
    let tea = h.fact(TEA);
    owner_says(&h, "I like green tea.");
    let call1 = reply(vec![claim(TEA, "fact", "I like green tea")]);
    let input = call2(&h, &call1).expect("call 2 runs");
    let llm = FakeLlm::scripted(
        MODEL,
        vec![
            call1,
            call2_reply(vec![labelled(
                &input.claims[0].handle,
                &[(neighbour_handle(&input, tea), "duplicates")],
            )]),
        ],
    );
    let error = h
        .service
        .extract_chunk(lease(&h, "main"), &llm, &[])
        .unwrap_err();
    assert_eq!(error.failure(), Some(Failure::Retry { error_count: 1 }));
    assert_eq!(h.memories_in("main"), 1);
    assert_eq!(h.accesses(tea), vec![Harness::fixture_access()]);
    assert_eq!(h.all_edits_on(tea), 0);
}

#[test]
fn an_llm_that_cant_be_used_at_call_2_holds_the_queue() {
    let h = Harness::new();
    h.fact(TEA);
    owner_says(&h, "I like green tea.");
    let llm = ThenFails {
        first: Mutex::new(Some(reply(vec![claim(TEA, "fact", "I like green tea")]))),
        error: || LlmError::LoginRequired,
    };
    let error = h
        .service
        .extract_chunk(lease(&h, "main"), &llm, &[])
        .unwrap_err();
    // Not the chunk's fault: nothing counted, and it stays at the head.
    assert_eq!(error.failure(), None);
    let lease = lease(&h, "main");
    assert_eq!(lease.error_count, 0);
    assert_eq!(h.memories_in("main"), 1);
}
