//! Extraction call 1, checked against "Extraction call 1: claims,
//! significance, windows, entities and used verdicts" (TIM-107) and the
//! decisions it rests on: "What is a memory record?" (TIM-90, the memory,
//! time, entities and accesses), "Strength model: decay, reinforcement and
//! significance" (TIM-91, decisions 6 and 7), "Extraction: significance,
//! validity windows and supersession" (TIM-92, round 1 and the resolution,
//! as amended by TIM-97), "API surface and Hermes transport" (TIM-94,
//! decision 1), and ADRs 0001, 0002, 0005, 0008 and 0010.
//!
//! These are golden tests against `FakeLlm`: each scripts call 1's reply and
//! checks what's committed, or checks the input and request call 1 is given.
//! Reconciliation (call 2, TIM-108) doesn't exist at this stage, so every
//! claim that survives the checks in code becomes a new memory.
//!
//! The API under test is `asphodel_core::extraction` and the `Service`
//! methods over it, `call1_input` and `extract_chunk`. Three tests check the
//! schema and constants call 1 relies on: that the schema TIM-103 landed
//! holds what call 1 commits, that it allows one access per memory per turn,
//! and that no significance level extraction can give is permanent from
//! creation.
//!
//! Every service here runs on a `SimulatedClock` stopped at one instant
//! unless a test advances it, so a stored time that equals that instant can
//! only have come from the Clock.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use asphodel_core::Service;
use asphodel_core::clock::{Clock, SimulatedClock};
use asphodel_core::config::Tuning;
use asphodel_core::constants::{CHUNK_RETRY_CAP, SIGNIFICANCE_KEPT, Significance};
use asphodel_core::ingest::{Document, Ingested, Turn, TurnAuthor};
use asphodel_core::models::{
    Embedder, FakeEmbedder, FakeLlm, FakeReranker, LlmClient, LlmError, LlmRequest, LlmResponse,
    ModelError, Models, Template,
};
use asphodel_core::queue::{Failure, Lease, SourceKind};
use asphodel_core::store::{OpenOptions, Store, micros, timestamp};
use jiff::civil::{Date, DateTime, date};
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp, ToSpan};
use rusqlite::OptionalExtension;
use rusqlite::types::FromSql;
use serde_json::{Value, json};
use uuid::Uuid;

use asphodel_core::extraction::{
    CALENDAR_DAYS, CALL1_TEMPLATE, CALL1_VERSION, CANDIDATE_MEMORIES, CONTEXT_CHARS, CONTEXT_TURNS,
    Call1Input, DropReason, Dropped, ENTITY_CANDIDATE_CAP, EntityKind, ExtractError, Extracted,
    PREVIOUS_CHUNK_CHARS, call1_request,
};

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";
const MODEL: &str = "fake-llm";

/// A turn's message time: 19:30 on Thursday 1 October 2026 in Auckland,
/// which is on daylight time (UTC+13).
const T1: &str = "2026-10-01T06:30:00Z";

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
}

/// A local date-time in `TZ` as the instant stored for it.
fn local(datetime: &str) -> Timestamp {
    local_in(datetime, TZ)
}

fn local_in(datetime: &str, zone: &str) -> Timestamp {
    datetime
        .parse::<DateTime>()
        .unwrap()
        .to_zoned(TimeZone::get(zone).unwrap())
        .unwrap()
        .timestamp()
}

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-extraction-{}-{}",
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

/// A floor for each fake model, so a service opens on the fakes.
fn tuning_for_fakes() -> Tuning {
    Tuning::from_toml(&format!(
        "[injection.reranker_floors]\n\"{}\" = 0.0\n\
         [reconcile.embedding_floors]\n\"{}\" = 0.5\n",
        FakeReranker::MODEL_ID,
        FakeEmbedder::MODEL_ID,
    ))
    .unwrap()
}

/// The owner is Tim, on Discord as `discord:1234`, and the assistant is
/// Hermes.
fn identity() -> asphodel_core::store::bank::BankIdentity {
    asphodel_core::store::bank::BankIdentity {
        owner_name: Some("Tim".into()),
        owner_platform_ids: vec!["discord:1234".into()],
        assistant_name: Some("Hermes".into()),
        timezone: Some(TZ.into()),
    }
}

/// An embedder that always fails, under the fake's id so the floors hold.
struct FailingEmbedder;

impl Embedder for FailingEmbedder {
    fn model_id(&self) -> &str {
        FakeEmbedder::MODEL_ID
    }

    fn dimensions(&self) -> usize {
        FakeEmbedder.dimensions()
    }

    fn embed(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        Err(ModelError::Inference {
            model: FakeEmbedder::MODEL_ID.into(),
            reason: "scripted failure".into(),
        })
    }
}

/// A public id for a fixture row.
fn next_uuid() -> Uuid {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    Uuid::from_u128((0xf1_u128 << 120) | u128::from(NEXT.fetch_add(1, Ordering::Relaxed)))
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
        Self::with_models(Models::fake())
    }

    fn with_models(models: Models) -> Self {
        let dir = TestDir::new();
        let clock = Arc::new(SimulatedClock::new(at(START)));
        let store = Store::open(&dir.data(), OpenOptions::default(), clock.clone()).unwrap();
        let service =
            Service::with_models(clock.clone(), store, tuning_for_fakes(), models).unwrap();
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

    /// A service built without models, as `Service::open` gives.
    fn without_models() -> Self {
        let dir = TestDir::new();
        let clock = Arc::new(SimulatedClock::new(at(START)));
        let store = Store::open(&dir.data(), OpenOptions::default(), clock.clone()).unwrap();
        let service = Service::open(clock.clone(), store, tuning_for_fakes());
        let ids = Models::fake().ids();
        service.ensure_bank("main", &identity(), &ids).unwrap();
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

    fn count(&self, sql: &str) -> i64 {
        self.one(sql, [])
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

    /// Every entity in `bank` called `name`, oldest first.
    fn entities_named(&self, bank: &str, name: &str) -> Vec<Uuid> {
        self.all::<String, _>(
            "SELECT e.uuid FROM entities e JOIN banks b ON b.id = e.bank_id
             WHERE b.name = ?1 AND e.name = ?2 ORDER BY e.id",
            [bank, name],
        )
        .iter()
        .map(|uuid| uuid.parse().unwrap())
        .collect()
    }

    fn entity_name(&self, entity: Uuid) -> String {
        self.one(
            "SELECT name FROM entities WHERE uuid = ?1",
            [entity.to_string()],
        )
    }

    fn entity_kind(&self, entity: Uuid) -> String {
        self.one(
            "SELECT kind FROM entities WHERE uuid = ?1",
            [entity.to_string()],
        )
    }

    /// An entity's aliases, sorted.
    fn aliases(&self, entity: Uuid) -> Vec<String> {
        self.all(
            "SELECT a.alias FROM entity_aliases a JOIN entities e ON e.id = a.entity_id
             WHERE e.uuid = ?1 ORDER BY a.alias",
            [entity.to_string()],
        )
    }

    /// A memory's entity links: (entity, surface form).
    fn links(&self, memory: Uuid) -> BTreeSet<(Uuid, Option<String>)> {
        let store = self.service.store().unwrap();
        let conn = store.connection();
        let mut statement = conn
            .prepare(
                "SELECT e.uuid, me.surface_form FROM memory_entities me
                 JOIN memories m ON m.id = me.memory_id JOIN entities e ON e.id = me.entity_id
                 WHERE m.uuid = ?1",
            )
            .unwrap();
        statement
            .query_map([memory.to_string()], |row| {
                Ok((row.get::<_, String>(0)?.parse().unwrap(), row.get(1)?))
            })
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
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

    fn edits(&self, bank: &str, kind: &str) -> i64 {
        self.one(
            "SELECT COUNT(*) FROM edits e JOIN banks b ON b.id = e.bank_id
             WHERE b.name = ?1 AND e.kind = ?2",
            [bank, kind],
        )
    }

    fn memories_in(&self, bank: &str) -> i64 {
        self.one(
            "SELECT COUNT(*) FROM memories m JOIN banks b ON b.id = m.bank_id WHERE b.name = ?1",
            [bank],
        )
    }

    fn row(&self, memory: Uuid) -> Row {
        let timed = |at: Option<i64>, precision: Option<String>| {
            at.map(|at| (timestamp(at), precision.expect("a time has a precision")))
        };
        self.service
            .store()
            .unwrap()
            .connection()
            .query_row(
                "SELECT content, kind, significance, owner_significance, observed_at,
                        valid_from, valid_from_precision, valid_until, valid_until_precision,
                        until_event, window_confidence, due_at, due_at_precision, volatility,
                        recurrence_text, recurrence_rrule, recurrence_start,
                        recurrence_start_precision, source_start, source_end
                 FROM memories WHERE uuid = ?1",
                [memory.to_string()],
                |row| {
                    Ok(Row {
                        content: row.get(0)?,
                        kind: row.get(1)?,
                        significance: row.get(2)?,
                        owner_significance: row.get(3)?,
                        observed_at: timestamp(row.get(4)?),
                        valid_from: timed(row.get(5)?, row.get(6)?),
                        valid_until: timed(row.get(7)?, row.get(8)?),
                        until_event: row.get(9)?,
                        window_confidence: row.get(10)?,
                        due_at: timed(row.get(11)?, row.get(12)?),
                        volatility: row.get(13)?,
                        recurrence_text: row.get(14)?,
                        recurrence_rrule: row.get(15)?,
                        recurrence_start: timed(row.get(16)?, row.get(17)?),
                        source_start: row.get(18)?,
                        source_end: row.get(19)?,
                    })
                },
            )
            .unwrap()
    }

    /// The chunk at `position` of `source`.
    fn chunk_of(&self, source: Uuid, position: i64) -> Uuid {
        let uuid: String = self.one(
            "SELECT c.uuid FROM chunks c JOIN sources s ON s.id = c.source_id
             WHERE s.uuid = ?1 AND c.position = ?2",
            (source.to_string(), position),
        );
        uuid.parse().unwrap()
    }

    fn chunk_column<T: FromSql>(&self, chunk: Uuid, column: &str) -> T {
        self.one(
            &format!("SELECT {column} FROM chunks WHERE uuid = ?1"),
            [chunk.to_string()],
        )
    }

    /// Takes every other chunk of the chunk's bank off the queue, so it's
    /// the one claimed next.
    fn focus(&self, chunk: Uuid) {
        self.execute(
            "DELETE FROM extraction_queue
             WHERE bank_id = (SELECT bank_id FROM chunks WHERE uuid = ?1)
               AND chunk_id != (SELECT id FROM chunks WHERE uuid = ?1)",
            [chunk.to_string()],
        );
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
        ingest_in(
            self,
            bank,
            &turn("fixtures", "2026-09-01T00:00:00Z", "Fixtures.", "Noted."),
        );
        let chunk = find().unwrap();
        self.execute("DELETE FROM extraction_queue WHERE chunk_id = ?1", [chunk]);
        self.execute(
            "UPDATE chunks SET extracted_at = ?2 WHERE id = ?1",
            (chunk, micros(self.now())),
        );
        chunk
    }

    /// A fact in `bank` with one `created` access at turn 0, inserted
    /// directly, as an earlier extraction would have left it.
    fn insert_memory(&self, bank: &str, content: &str, significance: &str) -> Uuid {
        self.insert_memory_of_kind(bank, content, "fact", significance)
    }

    /// [`Harness::insert_memory`] of any kind. A kind never changes once
    /// stored, so it's chosen here.
    fn insert_memory_of_kind(
        &self,
        bank: &str,
        content: &str,
        kind: &str,
        significance: &str,
    ) -> Uuid {
        let chunk = self.fixture_chunk(bank);
        let uuid = next_uuid();
        let now = micros(self.now());
        self.execute(
            "INSERT INTO memories (uuid, bank_id, content, kind, significance, chunk_id,
                                   source_start, source_end, observed_at, window_confidence,
                                   created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, 9, ?7, 'high', ?7, ?7)",
            (
                uuid.to_string(),
                self.bank_id(bank),
                content,
                kind,
                significance,
                chunk,
                now,
            ),
        );
        self.execute(
            "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
             SELECT bank_id, id, 'created', ?2, 0 FROM memories WHERE uuid = ?1",
            (uuid.to_string(), now),
        );
        uuid
    }

    /// An access on `memory` at `turn`, inserted directly.
    fn insert_access(&self, memory: Uuid, kind: &str, turn: i64) {
        self.execute(
            "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
             SELECT bank_id, id, ?2, ?3, ?4 FROM memories WHERE uuid = ?1",
            (memory.to_string(), kind, micros(self.now()), turn),
        );
    }

    /// An access on `memory` at `turn` and world time `at`, inserted
    /// directly.
    fn insert_access_at(&self, memory: Uuid, kind: &str, turn: i64, at: Timestamp) {
        self.execute(
            "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
             SELECT bank_id, id, ?2, ?3, ?4 FROM memories WHERE uuid = ?1",
            (memory.to_string(), kind, micros(at), turn),
        );
    }

    /// An entity in `bank` with `aliases`, inserted directly.
    fn insert_entity(&self, bank: &str, name: &str, kind: &str, aliases: &[&str]) -> Uuid {
        let uuid = next_uuid();
        let bank_id = self.bank_id(bank);
        let now = micros(self.now());
        self.execute(
            "INSERT INTO entities (uuid, bank_id, name, kind, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
            (uuid.to_string(), bank_id, name, kind, now),
        );
        for alias in aliases {
            self.execute(
                "INSERT INTO entity_aliases (bank_id, entity_id, alias, created_at)
                 SELECT bank_id, id, ?2, ?3 FROM entities WHERE uuid = ?1",
                (uuid.to_string(), *alias, now),
            );
        }
        uuid
    }

    fn link(&self, memory: Uuid, entity: Uuid) {
        self.execute(
            "INSERT INTO memory_entities (memory_id, entity_id)
             SELECT m.id, e.id FROM memories m, entities e WHERE m.uuid = ?1 AND e.uuid = ?2",
            (memory.to_string(), entity.to_string()),
        );
    }

    fn merge(&self, from: Uuid, into: Uuid) {
        self.execute(
            "UPDATE entities SET merged_into = (SELECT id FROM entities WHERE uuid = ?2)
             WHERE uuid = ?1",
            (from.to_string(), into.to_string()),
        );
    }

    /// Hides a memory as forget does before its erase runs (ADR 0010).
    fn hide(&self, memory: Uuid) {
        self.execute(
            "UPDATE memories SET hidden_at = ?2 WHERE uuid = ?1",
            (memory.to_string(), micros(self.now())),
        );
    }
}

#[derive(Debug, Clone, PartialEq)]
struct AccessRow {
    kind: String,
    at: Timestamp,
    turn: i64,
    source: Option<Uuid>,
}

/// A memory row as call 1 commits it.
#[derive(Debug, Clone, PartialEq)]
struct Row {
    content: String,
    kind: String,
    significance: String,
    owner_significance: Option<String>,
    observed_at: Timestamp,
    valid_from: Option<(Timestamp, String)>,
    valid_until: Option<(Timestamp, String)>,
    until_event: Option<String>,
    window_confidence: String,
    due_at: Option<(Timestamp, String)>,
    volatility: Option<String>,
    recurrence_text: Option<String>,
    recurrence_rrule: Option<String>,
    recurrence_start: Option<(Timestamp, String)>,
    source_start: i64,
    source_end: i64,
}

impl Row {
    /// A claim with nothing but its sentence and kind, as [`claim`] scripts
    /// it: minor, high window confidence, no times. `span` is where its quote
    /// is in the chunk.
    fn new(content: &str, kind: &str, observed_at: Timestamp, span: (i64, i64)) -> Self {
        Self {
            content: content.into(),
            kind: kind.into(),
            significance: "minor".into(),
            owner_significance: None,
            observed_at,
            valid_from: None,
            valid_until: None,
            until_event: None,
            window_confidence: "high".into(),
            due_at: None,
            volatility: None,
            recurrence_text: None,
            recurrence_rrule: None,
            recurrence_start: None,
            source_start: span.0,
            source_end: span.1,
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

/// A turn on Discord from Sam, who isn't the owner.
fn sams_turn(message_at: &str, user: &str, assistant: &str) -> Turn {
    Turn {
        author: Some(TurnAuthor {
            id: "5678".into(),
            name: Some("Sam".into()),
            is_bot: false,
        }),
        platform: Some("discord".into()),
        ..turn("thread-1", message_at, user, assistant)
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

/// A turn's chunk text: the message, the separator and the reply.
fn turn_text(user: &str, reply: &str) -> String {
    format!("{user}{}{reply}", asphodel_core::ingest::TURN_SEPARATOR)
}

/// Where `quote` first appears in `text`, in characters.
fn span(text: &str, quote: &str) -> (i64, i64) {
    let byte = text.find(quote).expect("the quote is in the text");
    let start = text[..byte].chars().count();
    (start as i64, (start + quote.chars().count()) as i64)
}

/// Characters `start..end` of `text`.
fn chars(text: &str, start: usize, end: usize) -> String {
    text.chars().skip(start).take(end - start).collect()
}

fn ingest(h: &Harness, turn: &Turn) -> Ingested {
    ingest_in(h, "main", turn)
}

fn ingest_in(h: &Harness, bank: &str, turn: &Turn) -> Ingested {
    h.service.ingest_turn(bank, turn).unwrap()
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

/// Call 1's input for the head of the bank's queue. The lease is released
/// when it drops.
fn input(h: &Harness, bank: &str, in_context: &[Uuid]) -> Call1Input {
    let lease = lease(h, bank);
    h.service.call1_input(&lease, in_context).unwrap()
}

/// Extracts the head of the bank's queue with `llm`.
fn run(
    h: &Harness,
    bank: &str,
    llm: &FakeLlm,
    in_context: &[Uuid],
) -> Result<Extracted, ExtractError> {
    h.service.extract_chunk(lease(h, bank), llm, in_context)
}

/// Extracts the head of `main`'s queue with call 1 answering `reply`.
fn extract(h: &Harness, reply: Value) -> Extracted {
    extract_with(h, reply, &[])
}

fn extract_with(h: &Harness, reply: Value, in_context: &[Uuid]) -> Extracted {
    run(
        h,
        "main",
        &FakeLlm::scripted(MODEL, vec![reply]),
        in_context,
    )
    .unwrap()
}

/// Ingests the owner's turn in session `s1` at [`T1`], extracts it with
/// `claims` and returns the new memories in claim order.
fn golden(h: &Harness, user: &str, reply_text: &str, claims: Vec<Value>) -> Vec<Uuid> {
    ingest(h, &turn("s1", T1, user, reply_text));
    extract(h, reply(claims, &[])).memories
}

fn handle(input: &Call1Input, entity: Uuid) -> String {
    input
        .candidates
        .iter()
        .find(|candidate| candidate.entity == entity)
        .unwrap_or_else(|| panic!("{entity} is a candidate"))
        .handle
        .clone()
}

fn memory_handle(input: &Call1Input, memory: Uuid) -> String {
    input
        .in_context
        .iter()
        .find(|in_context| in_context.memory == memory)
        .unwrap_or_else(|| panic!("{memory} is in context"))
        .handle
        .clone()
}

/// The candidates' entities, without `user` and `assistant`.
fn found(h: &Harness, input: &Call1Input) -> BTreeSet<Uuid> {
    let always = [h.seeded("main", "user"), h.seeded("main", "assistant")];
    input
        .candidates
        .iter()
        .map(|candidate| candidate.entity)
        .filter(|entity| !always.contains(entity))
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

fn link(handle: &str, surface_form: &str) -> Value {
    json!({"entity": handle, "new_name": null, "new_kind": null, "surface_form": surface_form})
}

fn new_entity(name: &str, kind: &str, surface_form: &str) -> Value {
    json!({"entity": null, "new_name": name, "new_kind": kind, "surface_form": surface_form})
}

fn reply(claims: Vec<Value>, used: &[&str]) -> Value {
    json!({"claims": claims, "used_injected_ids": used})
}

/// Extracts the head of `main`'s queue with call 1 answering `reply` and, if
/// the claims land near something stored, call 2 labelling nothing, so each
/// claim stays new. "Tim said One." and "Tim said Two." are close enough under
/// the fake embedder for reconciliation (TIM-108) to compare them.
fn extract_unlabelled(h: &Harness, reply: Value) -> Extracted {
    run(
        h,
        "main",
        &FakeLlm::scripted(MODEL, vec![reply, json!({"claims": []})]),
        &[],
    )
    .unwrap()
}

// Schema helpers.

fn keys(value: &Value) -> BTreeSet<String> {
    value
        .as_object()
        .expect("an object")
        .keys()
        .cloned()
        .collect()
}

fn strings(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .expect("an array")
        .iter()
        .filter_map(|value| value.as_str().map(str::to_owned))
        .collect()
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|item| item.to_string()).collect()
}

/// The object schema in `schema`, looking through a nullable `anyOf`.
fn object(schema: &Value) -> &Value {
    if schema.get("properties").is_some() {
        return schema;
    }
    schema["anyOf"]
        .as_array()
        .and_then(|options| {
            options
                .iter()
                .find(|option| option.get("properties").is_some())
        })
        .expect("an object schema")
}

/// The non-null values of `schema`'s enum, looking through an `anyOf`.
fn enum_values(schema: &Value) -> BTreeSet<String> {
    if let Some(values) = schema.get("enum") {
        return strings(values);
    }
    schema["anyOf"]
        .as_array()
        .and_then(|options| options.iter().find_map(|option| option.get("enum")))
        .map(strings)
        .expect("an enum")
}

const CLAIM_FIELDS: [&str; 16] = [
    "content",
    "kind",
    "quote",
    "significance",
    "remember_this",
    "changes_something",
    "valid_from",
    "valid_until",
    "window_confidence",
    "until_event",
    "due_at",
    "volatility",
    "recurrence_text",
    "recurrence_rrule",
    "recurrence_start",
    "entities",
];

const LEVELS: [&str; 5] = ["trivial", "minor", "notable", "major", "critical"];
const KINDS: [&str; 5] = ["fact", "event", "state", "task", "recurring"];

// The schema call 1 commits into, which runs now.

#[test]
fn the_schema_holds_what_call_1_commits() {
    let h = Harness::new();
    let source = ingest(&h, &turn("s1", T1, "Hello.", "Hi.")).source;
    let chunk_id: i64 = h.one(
        "SELECT c.id FROM chunks c JOIN sources s ON s.id = c.source_id WHERE s.uuid = ?1",
        [source.to_string()],
    );
    let bank_id = h.bank_id("main");
    let now = micros(h.now());
    let insert = |significance: &str, owner: Option<&str>| {
        h.service.store().unwrap().connection().execute(
            "INSERT INTO memories (uuid, bank_id, content, kind, significance, owner_significance,
                                   chunk_id, source_start, source_end, observed_at,
                                   window_confidence, created_at, updated_at)
             VALUES (?1, ?2, 'Tim said hello.', 'fact', ?3, ?4, ?5, 0, 6, ?6, 'high', ?6, ?6)",
            rusqlite::params![
                next_uuid().to_string(),
                bank_id,
                significance,
                owner,
                chunk_id,
                now
            ],
        )
    };

    // Every level call 1 can give fits, with or without the owner keeping it.
    for level in LEVELS {
        insert(level, None).unwrap();
    }
    insert("notable", Some("kept")).unwrap();
    // A level above critical has nowhere to go: only the owner keeps.
    assert!(insert("kept", None).is_err());

    // The columns call 1 writes or reads.
    for (table, column) in [
        ("chunks", "call1_output"),
        ("chunks", "extracted_at"),
        ("memory_entities", "surface_form"),
        ("accesses", "turn"),
        ("accesses", "source_id"),
        ("memories", "hidden_at"),
        ("sources", "ingested_at"),
        ("sources", "reference_date_exact"),
    ] {
        let found: i64 = h.one(
            &format!("SELECT COUNT(*) FROM pragma_table_info('{table}') WHERE name = ?1"),
            [column],
        );
        assert_eq!(found, 1, "{table}.{column}");
    }

    // A kind's fields only on that kind.
    let misplaced = h.service.store().unwrap().connection().execute(
        "INSERT INTO memories (uuid, bank_id, content, kind, significance, chunk_id, source_start,
                               source_end, observed_at, window_confidence, volatility,
                               created_at, updated_at)
         VALUES (?1, ?2, 'Tim is tired.', 'fact', 'minor', ?3, 0, 6, ?4, 'high', 'days', ?4, ?4)",
        (next_uuid().to_string(), bank_id, chunk_id, now),
    );
    assert!(misplaced.is_err(), "volatility belongs to states only");
}

#[test]
fn the_schema_allows_one_access_per_memory_per_turn() {
    let h = Harness::new();
    let memory = h.insert_memory("main", "Tim likes tea.", "minor");
    h.insert_access(memory, "used", 7);
    let again = h.service.store().unwrap().connection().execute(
        "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
         SELECT bank_id, id, 'confirmed', ?2, 7 FROM memories WHERE uuid = ?1",
        (memory.to_string(), micros(h.now())),
    );
    assert!(again.is_err(), "a second access in the same turn");
    h.insert_access(memory, "used", 8);
    assert_eq!(h.accesses(memory).len(), 3);
}

#[test]
fn no_level_extraction_can_give_is_permanent_from_creation() {
    // TIM-91: at 0.925 or above a memory is permanent from creation, so that
    // range is the owner's, and extraction may not score above 0.9.
    let values: Vec<f64> = Significance::ALL
        .iter()
        .map(|level| level.value())
        .collect();
    assert_eq!(values, vec![0.1, 0.3, 0.5, 0.7, 0.9]);
    assert!(values.iter().all(|value| *value < 0.925));
    assert_eq!(SIGNIFICANCE_KEPT, 1.0);
}

// Context assembly.

#[test]
fn a_turns_input_is_its_text_speaker_and_reference_date() {
    let h = Harness::new();
    let ingested = ingest(&h, &turn("s1", T1, "Dentist tomorrow at 3pm.", "Noted."));
    let input = input(&h, "main", &[]);

    assert_eq!(input.chunk, h.chunk_of(ingested.source, 0));
    assert_eq!(input.source_kind, SourceKind::Turn);
    assert_eq!(input.text, turn_text("Dentist tomorrow at 3pm.", "Noted."));
    assert_eq!(input.observed_at, at(T1));
    assert_eq!(input.timezone, TZ);
    assert_eq!(input.reference_date, Some(date(2026, 10, 1)));
    assert!(input.context.is_empty());
    assert!(input.in_context.is_empty());

    let speaker = input.speaker.as_ref().expect("a turn has a speaker");
    assert_eq!(speaker.entity, h.seeded("main", "user"));
    assert_eq!(speaker.name, "Tim");
    assert!(speaker.owner);
    assert_eq!(speaker.handle, handle(&input, speaker.entity));
}

#[test]
fn the_calendar_is_three_weeks_either_side_of_the_local_date() {
    let h = Harness::new();
    // 01:00 on Friday 2 October in Auckland, still the 1st in UTC.
    ingest(&h, &turn("s1", "2026-10-01T12:00:00Z", "Hi.", "Hello."));
    let input = input(&h, "main", &[]);

    let reference = date(2026, 10, 2);
    assert_eq!(input.reference_date, Some(reference));
    let expected: Vec<Date> = (-CALENDAR_DAYS..=CALENDAR_DAYS)
        .map(|offset| reference.checked_add(offset.days()).unwrap())
        .collect();
    assert_eq!(input.calendar, expected);
    assert_eq!(input.calendar.first(), Some(&date(2026, 9, 11)));
    assert_eq!(input.calendar.last(), Some(&date(2026, 10, 23)));
}

#[test]
fn up_to_three_earlier_turns_of_the_session_are_context() {
    let h = Harness::new();
    let earlier: Vec<(String, String)> = (1..=5)
        .map(|k| (format!("Message {k}."), format!("Reply {k}.")))
        .collect();
    for (k, (user, answer)) in earlier.iter().enumerate() {
        let message_at = format!("2026-10-01T06:0{k}:00Z");
        ingest(&h, &turn("s1", &message_at, user, answer));
    }
    // Neither another session, another bank, a forget request nor a later
    // turn is context.
    ingest(
        &h,
        &turn("s2", "2026-10-01T06:10:00Z", "Other session.", "Ok."),
    );
    ingest_in(
        &h,
        "other",
        &turn("s1", "2026-10-01T06:11:00Z", "Other bank.", "Ok."),
    );
    ingest(
        &h,
        &Turn {
            forget_requested: true,
            ..turn("s1", "2026-10-01T06:12:00Z", "Forget my address.", "Done.")
        },
    );
    let current = ingest(&h, &turn("s1", T1, "Now.", "Yes."));
    ingest(&h, &turn("s1", "2026-10-01T06:40:00Z", "Later.", "Ok."));
    h.focus(h.chunk_of(current.source, 0));

    let input = input(&h, "main", &[]);
    assert_eq!(CONTEXT_TURNS, 3);
    let expected: Vec<String> = earlier[2..]
        .iter()
        .map(|(user, answer)| turn_text(user, answer))
        .collect();
    assert_eq!(input.context, expected);
    assert_eq!(input.text, turn_text("Now.", "Yes."));
}

#[test]
fn context_is_clipped_oldest_first() {
    let h = Harness::new();
    // Three earlier turns of 2,500 characters each: 7,500 in all.
    let passages: Vec<String> = (1..=3)
        .map(|k| {
            let user = format!("START{k}{}", "x".repeat(2_500 - 6 - 2 - 2));
            turn_text(&user, "ok")
        })
        .collect();
    for (k, passage) in passages.iter().enumerate() {
        let user = passage.strip_suffix("\n\nok").unwrap();
        let message_at = format!("2026-10-01T06:0{k}:00Z");
        ingest(&h, &turn("s1", &message_at, user, "ok"));
    }
    assert!(passages.iter().all(|p| p.chars().count() == 2_500));
    let current = ingest(&h, &turn("s1", T1, "Now.", "Yes."));
    h.focus(h.chunk_of(current.source, 0));

    let input = input(&h, "main", &[]);
    let total: usize = input.context.iter().map(|c| c.chars().count()).sum();
    assert_eq!(total, CONTEXT_CHARS);
    // The oldest loses its first 1,500 characters; the newer two are whole.
    assert_eq!(
        input.context,
        vec![
            chars(&passages[0], 1_500, 2_500),
            passages[1].clone(),
            passages[2].clone(),
        ]
    );
}

#[test]
fn a_document_chunk_gets_the_text_before_it_as_context() {
    let h = Harness::new();
    let trip = format!("TRIPSTART {}", "We fly out early. ".repeat(40));
    let text = format!("# Trip\n\n{trip}\n\n# Packing\n\nBring the blue tent.\n");
    let ingested = ingest_doc(&h, &document("notes", &text, date(2026, 9, 28)));
    let memory = h.insert_memory("main", "Tim likes tea.", "minor");

    // The first chunk has nothing before it, no speaker and no in-context
    // memories, whatever the caller passes.
    let first = input(&h, "main", &[memory]);
    assert_eq!(first.source_kind, SourceKind::Document);
    assert_eq!(first.reference_date, Some(date(2026, 9, 28)));
    assert!(first.context.is_empty());
    assert!(first.speaker.is_none());
    assert!(first.in_context.is_empty());

    let second = h.chunk_of(ingested.source, 1);
    h.focus(second);
    let start: i64 = h.chunk_column(second, "start_offset");
    let start = usize::try_from(start).unwrap();
    assert!(start > PREVIOUS_CHUNK_CHARS);
    let input = input(&h, "main", &[]);
    assert_eq!(input.chunk, second);
    assert_eq!(
        input.context,
        vec![chars(&text, start - PREVIOUS_CHUNK_CHARS, start)]
    );
    assert!(!input.context[0].contains("TRIPSTART"));
}

#[test]
fn an_inexact_reference_date_gets_no_calendar() {
    let h = Harness::new();
    ingest_doc(
        &h,
        &Document {
            reference_date_exact: false,
            ..document("diary", "Met Ana yesterday.", date(2026, 9, 28))
        },
    );
    let input = input(&h, "main", &[]);
    assert_eq!(input.reference_date, None);
    assert!(input.calendar.is_empty());
}

#[test]
fn user_and_assistant_are_always_candidates() {
    let h = Harness::new();
    ingest(&h, &turn("s1", T1, "Hello there.", "Hi."));
    let input = input(&h, "main", &[]);

    let candidate = |entity: Uuid| {
        input
            .candidates
            .iter()
            .find(|c| c.entity == entity)
            .unwrap_or_else(|| panic!("{entity} is a candidate"))
    };
    let user = candidate(h.seeded("main", "user"));
    assert_eq!(user.name, "Tim");
    assert_eq!(user.kind, EntityKind::Person);
    assert!(user.aliases.iter().any(|alias| alias == "Tim"));
    let assistant = candidate(h.seeded("main", "assistant"));
    assert_eq!(assistant.name, "Hermes");
    assert_eq!(assistant.kind, EntityKind::Thing);
    assert!(assistant.aliases.iter().any(|alias| alias == "Hermes"));

    // Handles are distinct.
    let handles: BTreeSet<&str> = input.candidates.iter().map(|c| c.handle.as_str()).collect();
    assert_eq!(handles.len(), input.candidates.len());
}

#[test]
fn another_speaker_is_the_speaker_and_a_candidate() {
    let h = Harness::new();
    let ingested = ingest(&h, &sams_turn(T1, "I'm moving to Lisbon.", "Exciting!"));
    let sam = ingested.speaker.as_ref().unwrap().entity;
    let input = input(&h, "main", &[]);

    let speaker = input.speaker.as_ref().unwrap();
    assert_eq!(speaker.entity, sam);
    assert_eq!(speaker.name, "Sam");
    assert!(!speaker.owner);
    // "Sam" isn't in the text, but the speaker is always a candidate.
    assert_eq!(speaker.handle, handle(&input, sam));
    handle(&input, h.seeded("main", "user"));
    handle(&input, h.seeded("main", "assistant"));
}

#[test]
fn entities_named_in_the_chunk_or_its_context_are_candidates() {
    let h = Harness::new();
    let ana = h.insert_entity("main", "Ana", "person", &["Ana"]);
    let bob = h.insert_entity("main", "Bob", "person", &["Bob"]);
    let sam = h.insert_entity("main", "Sam", "person", &["Sam"]);
    let sammy = h.insert_entity("main", "Sammy", "person", &["Sammy"]);
    h.merge(sammy, sam);
    h.insert_entity("main", "Carol", "person", &["Carol"]);
    h.insert_entity("other", "Ana", "person", &["Ana"]);

    ingest(
        &h,
        &turn(
            "s1",
            "2026-10-01T06:00:00Z",
            "Bob called.",
            "What did he say?",
        ),
    );
    let current = ingest(
        &h,
        &turn("s1", T1, "Ana and Sammy are coming over.", "Lovely."),
    );
    h.focus(h.chunk_of(current.source, 0));
    let input = input(&h, "main", &[]);

    // Ana from this bank, Bob from the context, and Sammy as Sam, once.
    assert_eq!(found(&h, &input), BTreeSet::from([ana, bob, sam]));
    assert_eq!(
        input.candidates.iter().filter(|c| c.entity == sam).count(),
        1
    );
    let ana = input.candidates.iter().find(|c| c.entity == ana).unwrap();
    assert_eq!(ana.name, "Ana");
    assert_eq!(ana.kind, EntityKind::Person);
    assert_eq!(ana.aliases, vec!["Ana".to_string()]);
}

#[test]
fn a_candidate_shows_its_three_strongest_memories() {
    let h = Harness::new();
    let ana = h.insert_entity("main", "Ana", "person", &["Ana"]);
    for (content, level) in [
        ("Ana likes jazz.", "trivial"),
        ("Ana lives in Wellington.", "minor"),
        ("Ana is Tim's sister.", "critical"),
        ("Ana is a nurse.", "major"),
    ] {
        let memory = h.insert_memory("main", content, level);
        h.link(memory, ana);
    }
    let hidden = h.insert_memory("main", "Ana's address is 4 Elm St.", "critical");
    h.link(hidden, ana);
    h.hide(hidden);

    ingest(&h, &turn("s1", T1, "Ana called.", "How is she?"));
    let input = input(&h, "main", &[]);
    let candidate = input.candidates.iter().find(|c| c.entity == ana).unwrap();
    assert_eq!(CANDIDATE_MEMORIES, 3);
    assert_eq!(
        candidate.memories,
        vec![
            "Ana is Tim's sister.".to_string(),
            "Ana is a nurse.".to_string(),
            "Ana lives in Wellington.".to_string(),
        ]
    );
}

/// 32 entities named in one message, `Name00` to `Name31`, where `NameK` is
/// linked to K memories. Returns them in order, and the message.
fn crowded(h: &Harness) -> (Vec<Uuid>, String) {
    let memories: Vec<Uuid> = (0..31)
        .map(|k| h.insert_memory("main", &format!("Memory {k}."), "trivial"))
        .collect();
    let names: Vec<String> = (0..32).map(|k| format!("Name{k:02}")).collect();
    let entities: Vec<Uuid> = names
        .iter()
        .map(|name| h.insert_entity("main", name, "thing", &[name.as_str()]))
        .collect();
    for (k, entity) in entities.iter().enumerate() {
        for memory in &memories[..k] {
            h.link(*memory, *entity);
        }
    }
    (entities, names.join(" "))
}

#[test]
fn candidates_are_capped_by_how_many_memories_link_to_them() {
    let h = Harness::new();
    let (entities, message) = crowded(&h);
    ingest(&h, &turn("s1", T1, &message, "Quite a list."));
    let input = input(&h, "main", &[]);

    assert_eq!(ENTITY_CANDIDATE_CAP, 30);
    // The two with the fewest links are left out; user and assistant are on
    // top of the cap.
    let expected: BTreeSet<Uuid> = entities[2..].iter().copied().collect();
    assert_eq!(found(&h, &input), expected);
    assert_eq!(input.candidates.len(), ENTITY_CANDIDATE_CAP + 2);
}

#[test]
fn only_the_banks_visible_in_context_memories_are_given() {
    let h = Harness::new();
    let tea = h.insert_memory("main", "Tim likes tea.", "minor");
    let hidden = h.insert_memory("main", "Tim's address is 4 Elm St.", "major");
    h.hide(hidden);
    let elsewhere = h.insert_memory("other", "Tim likes coffee.", "minor");
    let unknown = next_uuid();
    ingest(&h, &turn("s1", T1, "Tea?", "You like tea, so yes."));

    let input = input(&h, "main", &[tea, hidden, elsewhere, unknown]);
    assert_eq!(input.in_context.len(), 1);
    assert_eq!(input.in_context[0].memory, tea);
    assert_eq!(input.in_context[0].content, "Tim likes tea.");
}

#[test]
fn the_request_is_the_input_and_the_call_1_schema() {
    let h = Harness::new();
    let ana = h.insert_entity("main", "Ana", "person", &["Ana"]);
    let tea = h.insert_memory("main", "Tim likes tea.", "minor");
    ingest(
        &h,
        &turn("s1", "2026-10-01T06:00:00Z", "Morning.", "Morning, Tim."),
    );
    let current = ingest(
        &h,
        &turn("s1", T1, "Ana wants tea.", "I'll put the kettle on."),
    );
    h.focus(h.chunk_of(current.source, 0));

    let input = input(&h, "main", &[tea]);
    let llm = FakeLlm::scripted(MODEL, vec![reply(vec![], &[])]);
    run(&h, "main", &llm, &[tea]).unwrap();
    let requests = llm.requests();
    assert_eq!(requests, vec![call1_request(&input)]);
    let request = &requests[0];

    assert_eq!(
        request.template,
        Template {
            name: CALL1_TEMPLATE.into(),
            version: CALL1_VERSION,
        }
    );
    // The user prompt carries the input.
    assert!(request.user.contains(&input.text));
    for context in &input.context {
        assert!(request.user.contains(context.as_str()));
    }
    for candidate in &input.candidates {
        assert!(request.user.contains(&candidate.handle));
        assert!(request.user.contains(&candidate.name));
    }
    assert!(request.user.contains(&handle(&input, ana)));
    assert!(request.user.contains(&memory_handle(&input, tea)));
    assert!(request.user.contains("Tim likes tea."));
    for day in &input.calendar {
        let line = day.strftime("%Y-%m-%d %A").to_string();
        assert!(request.user.contains(&line), "{line}");
    }
    // The system prompt holds the rules, not the chunk.
    assert!(!request.system.contains("Ana wants tea."));
    for word in LEVELS.iter().chain(KINDS.iter()) {
        assert!(request.system.contains(word), "{word}");
    }

    // The reply schema.
    let schema = &request.schema;
    assert_eq!(
        keys(&schema["properties"]),
        set(&["claims", "used_injected_ids"])
    );
    let item = object(&schema["properties"]["claims"]["items"]);
    assert_eq!(keys(&item["properties"]), set(&CLAIM_FIELDS));
    assert_eq!(strings(&item["required"]), set(&CLAIM_FIELDS));
    assert_eq!(
        enum_values(&item["properties"]["significance"]),
        set(&LEVELS)
    );
    assert_eq!(enum_values(&item["properties"]["kind"]), set(&KINDS));
    assert_eq!(
        enum_values(&item["properties"]["window_confidence"]),
        set(&["high", "low"])
    );
    let link = object(&item["properties"]["entities"]["items"]);
    assert_eq!(
        keys(&link["properties"]),
        set(&["entity", "new_name", "new_kind", "surface_form"])
    );
    for field in ["valid_from", "valid_until", "due_at", "recurrence_start"] {
        let time = object(&item["properties"][field]);
        assert_eq!(
            keys(&time["properties"]),
            set(&["at", "precision"]),
            "{field}"
        );
        assert_eq!(
            enum_values(&time["properties"]["precision"]),
            set(&["year", "month", "day", "hour", "minute"]),
            "{field}"
        );
    }
}

#[test]
fn the_system_prompt_is_the_same_for_every_chunk() {
    let h = Harness::new();
    ingest(&h, &turn("s1", T1, "Hello.", "Hi."));
    ingest(
        &h,
        &sams_turn("2026-10-01T06:40:00Z", "Hey all.", "Hi Sam."),
    );
    let llm = FakeLlm::scripted(MODEL, vec![reply(vec![], &[]), reply(vec![], &[])]);
    run(&h, "main", &llm, &[]).unwrap();
    run(&h, "main", &llm, &[]).unwrap();
    let requests = llm.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].system, requests[1].system);
    assert_ne!(requests[0].user, requests[1].user);
}

// Kinds.

#[test]
fn a_fact_keeps_a_stated_start_but_never_an_end() {
    let h = Harness::new();
    let user = "I started at Acme in March 2024.";
    let quote = "I started at Acme in March 2024";
    let content = "Tim has worked at Acme since March 2024.";
    let memories = golden(
        &h,
        user,
        "Nice.",
        vec![
            claim(content, "fact", quote)
                .with("significance", json!("notable"))
                .with("valid_from", time("2024-03", "month"))
                .with("valid_until", time("2027", "year")),
        ],
    );
    assert_eq!(
        h.row(memories[0]),
        Row {
            significance: "notable".into(),
            valid_from: timed(local("2024-03-01T00:00"), "month"),
            ..Row::new(
                content,
                "fact",
                at(T1),
                span(&turn_text(user, "Nice."), quote)
            )
        }
    );
}

#[test]
fn an_event_keeps_its_window_and_precisions() {
    let h = Harness::new();
    let user = "I'm on holiday from Monday until 12 October. Dentist tomorrow at 3pm.";
    let text = turn_text(user, "Enjoy.");
    let holiday = "Tim is on holiday from 5 October 2026 until 12 October 2026.";
    let dentist = "Tim has a dentist appointment on 2 October 2026 at 3pm.";
    let memories = golden(
        &h,
        user,
        "Enjoy.",
        vec![
            claim(holiday, "event", "on holiday from Monday until 12 October")
                .with("valid_from", time("2026-10-05", "day"))
                .with("valid_until", time("2026-10-12", "day")),
            claim(dentist, "event", "Dentist tomorrow at 3pm")
                .with("valid_from", time("2026-10-02T15:00", "hour")),
        ],
    );
    assert_eq!(
        h.row(memories[0]),
        Row {
            valid_from: timed(local("2026-10-05T00:00"), "day"),
            valid_until: timed(local("2026-10-12T00:00"), "day"),
            ..Row::new(
                holiday,
                "event",
                at(T1),
                span(&text, "on holiday from Monday until 12 October")
            )
        }
    );
    assert_eq!(
        h.row(memories[1]),
        Row {
            valid_from: timed(at("2026-10-02T02:00:00Z"), "hour"),
            ..Row::new(
                dentist,
                "event",
                at(T1),
                span(&text, "Dentist tomorrow at 3pm")
            )
        }
    );
}

#[test]
fn an_event_with_no_stated_time_starts_on_the_day_it_was_said() {
    let h = Harness::new();
    let user = "I finally filed the tax return.";
    let quote = "I finally filed the tax return";
    let memories = golden(
        &h,
        user,
        "Well done.",
        vec![claim("Tim filed the tax return.", "event", quote)],
    );
    // TIM-92: valid_from is observed_at at day precision, with low confidence.
    assert_eq!(
        h.row(memories[0]),
        Row {
            valid_from: timed(local("2026-10-01T00:00"), "day"),
            window_confidence: "low".into(),
            ..Row::new(
                "Tim filed the tax return.",
                "event",
                at(T1),
                span(&turn_text(user, "Well done."), quote)
            )
        }
    );
}

#[test]
fn a_state_keeps_its_volatility_and_until_event() {
    let h = Harness::new();
    let user = "I'm chasing a flaky build until the release ships. Feeling tired today.";
    let text = turn_text(user, "Good luck.");
    let build = "Tim is chasing a flaky build.";
    let tired = "Tim is feeling tired.";
    let memories = golden(
        &h,
        user,
        "Good luck.",
        vec![
            claim(
                build,
                "state",
                "chasing a flaky build until the release ships",
            )
            .with("volatility", json!("days"))
            .with("until_event", json!("the release ships")),
            claim(tired, "state", "Feeling tired today"),
        ],
    );
    assert_eq!(
        h.row(memories[0]),
        Row {
            volatility: Some("days".into()),
            until_event: Some("the release ships".into()),
            ..Row::new(
                build,
                "state",
                at(T1),
                span(&text, "chasing a flaky build until the release ships")
            )
        }
    );
    // Null when unsure, never guessed.
    assert_eq!(
        h.row(memories[1]),
        Row::new(tired, "state", at(T1), span(&text, "Feeling tired today"))
    );
}

#[test]
fn a_task_has_a_due_date_and_no_end() {
    let h = Harness::new();
    let user = "Remind me to renew my passport by 20 October.";
    let quote = "renew my passport by 20 October";
    let content = "Tim needs to renew his passport by 20 October 2026.";
    let memories = golden(
        &h,
        user,
        "Will do.",
        vec![
            claim(content, "task", quote)
                .with("due_at", time("2026-10-20", "day"))
                .with("valid_until", time("2026-10-20", "day")),
        ],
    );
    // valid_until is set only when an event ends the task.
    assert_eq!(
        h.row(memories[0]),
        Row {
            due_at: timed(local("2026-10-20T00:00"), "day"),
            ..Row::new(
                content,
                "task",
                at(T1),
                span(&turn_text(user, "Will do."), quote)
            )
        }
    );
}

#[test]
fn a_recurring_memory_keeps_a_rule_that_parses_and_recurs() {
    let h = Harness::new();
    let user = "I play football every Tuesday at 6pm.";
    let quote = "I play football every Tuesday at 6pm";
    let content = "Tim plays football every Tuesday at 6pm.";
    let memories = golden(
        &h,
        user,
        "Fun.",
        vec![
            claim(content, "recurring", quote)
                .with("recurrence_text", json!("every Tuesday at 6pm"))
                .with("recurrence_rrule", json!("FREQ=WEEKLY;BYDAY=TU"))
                .with("recurrence_start", time("2026-10-06T18:00", "hour")),
        ],
    );
    assert_eq!(
        h.row(memories[0]),
        Row {
            recurrence_text: Some("every Tuesday at 6pm".into()),
            recurrence_rrule: Some("FREQ=WEEKLY;BYDAY=TU".into()),
            recurrence_start: timed(local("2026-10-06T18:00"), "hour"),
            ..Row::new(
                content,
                "recurring",
                at(T1),
                span(&turn_text(user, "Fun."), quote)
            )
        }
    );
}

#[test]
fn a_recurring_memory_without_a_usable_rule_keeps_only_its_text() {
    let h = Harness::new();
    let user = "Bins go out every week. I swim on Fridays. We had a reunion every year until 2019.";
    let text = turn_text(user, "Ok.");
    let bins = "The bins go out every week.";
    let swim = "Tim swims every Friday.";
    let reunion = "Tim's family held a reunion every year until 2019.";
    let memories = golden(
        &h,
        user,
        "Ok.",
        vec![
            // Not a rule the rrule crate parses.
            claim(bins, "recurring", "Bins go out every week")
                .with("recurrence_text", json!("every week"))
                .with("recurrence_rrule", json!("FREQ=FORTNIGHTLY"))
                .with("recurrence_start", time("2026-10-05", "day")),
            // A rule with no first occurrence.
            claim(swim, "recurring", "I swim on Fridays")
                .with("recurrence_text", json!("on Fridays"))
                .with("recurrence_rrule", json!("FREQ=WEEKLY;BYDAY=FR")),
            // A rule with no occurrence in the year after the reference date.
            claim(
                reunion,
                "recurring",
                "We had a reunion every year until 2019",
            )
            .with("recurrence_text", json!("every year until 2019"))
            .with(
                "recurrence_rrule",
                json!("FREQ=YEARLY;UNTIL=20190601T000000Z"),
            )
            .with("recurrence_start", time("2015-06-01", "day")),
        ],
    );
    for (memory, content, quote, recurrence_text) in [
        (memories[0], bins, "Bins go out every week", "every week"),
        (memories[1], swim, "I swim on Fridays", "on Fridays"),
        (
            memories[2],
            reunion,
            "We had a reunion every year until 2019",
            "every year until 2019",
        ),
    ] {
        assert_eq!(
            h.row(memory),
            Row {
                recurrence_text: Some(recurrence_text.into()),
                ..Row::new(content, "recurring", at(T1), span(&text, quote))
            }
        );
    }
}

#[test]
fn fields_that_belong_to_another_kind_are_dropped() {
    let h = Harness::new();
    let user =
        "I'm learning Rust. The launch is on 9 October. I'm in Wellington. I need to call Mum.";
    let text = turn_text(user, "Ok.");
    let memories = golden(
        &h,
        user,
        "Ok.",
        vec![
            claim("Tim is learning Rust.", "fact", "I'm learning Rust")
                .with("volatility", json!("months")),
            claim(
                "The launch is on 9 October 2026.",
                "event",
                "The launch is on 9 October",
            )
            .with("valid_from", time("2026-10-09", "day"))
            .with("due_at", time("2026-10-09", "day")),
            claim("Tim is in Wellington.", "state", "I'm in Wellington")
                .with("volatility", json!("hours"))
                .with("recurrence_text", json!("daily"))
                .with("recurrence_rrule", json!("FREQ=DAILY")),
            claim("Tim needs to call Mum.", "task", "I need to call Mum")
                .with("volatility", json!("days")),
        ],
    );
    assert_eq!(
        h.row(memories[0]),
        Row::new(
            "Tim is learning Rust.",
            "fact",
            at(T1),
            span(&text, "I'm learning Rust")
        )
    );
    assert_eq!(
        h.row(memories[1]),
        Row {
            valid_from: timed(local("2026-10-09T00:00"), "day"),
            ..Row::new(
                "The launch is on 9 October 2026.",
                "event",
                at(T1),
                span(&text, "The launch is on 9 October")
            )
        }
    );
    assert_eq!(
        h.row(memories[2]),
        Row {
            volatility: Some("hours".into()),
            ..Row::new(
                "Tim is in Wellington.",
                "state",
                at(T1),
                span(&text, "I'm in Wellington")
            )
        }
    );
    assert_eq!(
        h.row(memories[3]),
        Row::new(
            "Tim needs to call Mum.",
            "task",
            at(T1),
            span(&text, "I need to call Mum")
        )
    );
}

// Relative dates. Call 1 resolves them against the calendar; code turns
// what it gives into instants in the source's timezone and checks weekdays.

#[test]
fn a_time_is_the_start_of_its_unit_in_the_sources_timezone() {
    let h = Harness::new();
    let event = |content: &str, at: &str, precision: &str| {
        claim(content, "event", "Plans").with("valid_from", time(at, precision))
    };
    let memories = golden(
        &h,
        "Plans: Japan next year, a wedding next month, a party on Saturday at 3:45pm.",
        "Busy!",
        vec![
            event("Tim is going to Japan in 2027.", "2027", "year"),
            event("Tim has a wedding in November 2026.", "2026-11-17", "month"),
            event("Tim has a wedding in November 2026.", "2026-11", "month"),
            event(
                "Tim has a party on 3 October 2026.",
                "2026-10-03T15:45",
                "day",
            ),
            event("Tim's party starts at 3pm.", "2026-10-03T15:45", "hour"),
            event(
                "Tim's party starts at 3:45pm.",
                "2026-10-03T15:45",
                "minute",
            ),
        ],
    );
    let starts: Vec<Option<(Timestamp, String)>> =
        memories.iter().map(|m| h.row(*m).valid_from).collect();
    assert_eq!(
        starts,
        vec![
            timed(local("2027-01-01T00:00"), "year"),
            timed(local("2026-11-01T00:00"), "month"),
            timed(local("2026-11-01T00:00"), "month"),
            timed(local("2026-10-03T00:00"), "day"),
            timed(local("2026-10-03T15:00"), "hour"),
            timed(local("2026-10-03T15:45"), "minute"),
        ]
    );
    // Auckland is on daylight time, UTC+13, through all of these.
    assert_eq!(local("2027-01-01T00:00"), at("2026-12-31T11:00:00Z"));
    assert_eq!(local("2026-10-03T15:45"), at("2026-10-03T02:45:00Z"));
}

#[test]
fn a_source_in_another_timezone_resolves_in_its_own() {
    let h = Harness::new();
    let london = "Europe/London";
    // 13:00 on 1 October in London, already the 2nd in the bank's Auckland.
    let user = "Party on Saturday. I mowed the lawn.";
    ingest(
        &h,
        &Turn {
            timezone: Some(london.into()),
            ..turn("s1", "2026-10-01T12:00:00Z", user, "Nice.")
        },
    );
    let input = input(&h, "main", &[]);
    assert_eq!(input.timezone, london);
    assert_eq!(input.reference_date, Some(date(2026, 10, 1)));

    let memories = extract(
        &h,
        reply(
            vec![
                claim(
                    "Tim has a party on 3 October 2026.",
                    "event",
                    "Party on Saturday",
                )
                .with("valid_from", time("2026-10-03", "day")),
                claim("Tim mowed the lawn.", "event", "I mowed the lawn"),
            ],
            &[],
        ),
    )
    .memories;
    assert_eq!(
        h.row(memories[0]).valid_from,
        timed(at("2026-10-02T23:00:00Z"), "day")
    );
    let mowed = h.row(memories[1]);
    assert_eq!(
        mowed.valid_from,
        timed(local_in("2026-10-01T00:00", london), "day")
    );
    assert_eq!(mowed.window_confidence, "low");
}

#[test]
fn a_weekday_in_the_quote_that_doesnt_match_the_date_lowers_confidence() {
    let h = Harness::new();
    // 2 October 2026 is a Friday.
    let memories = golden(
        &h,
        "Lunch with Ana on Friday. Drinks on Saturday. Report due Friday.",
        "Busy week.",
        vec![
            claim(
                "Tim has lunch with Ana on 2 October 2026.",
                "event",
                "Lunch with Ana on Friday",
            )
            .with("valid_from", time("2026-10-02", "day")),
            claim(
                "Tim has drinks on 2 October 2026.",
                "event",
                "Drinks on Saturday",
            )
            .with("valid_from", time("2026-10-02", "day")),
            claim(
                "Tim's report is due on 3 October 2026.",
                "task",
                "Report due Friday",
            )
            .with("due_at", time("2026-10-03", "day")),
            // Call 1's own low confidence stands.
            claim(
                "Tim might have lunch with Ana on 2 October 2026.",
                "event",
                "Lunch with Ana on Friday",
            )
            .with("valid_from", time("2026-10-02", "day"))
            .with("window_confidence", json!("low")),
        ],
    );
    let confidence: Vec<String> = memories
        .iter()
        .map(|m| h.row(*m).window_confidence)
        .collect();
    assert_eq!(confidence, vec!["high", "low", "low", "low"]);
    // The window itself is kept.
    assert_eq!(
        h.row(memories[1]).valid_from,
        timed(local("2026-10-02T00:00"), "day")
    );
}

#[test]
fn a_time_that_doesnt_parse_is_dropped_with_low_confidence() {
    let h = Harness::new();
    let memories = golden(
        &h,
        "The conference runs from Monday until whenever.",
        "Sounds open-ended.",
        vec![
            claim(
                "Tim's conference starts on 5 October 2026.",
                "event",
                "from Monday",
            )
            .with("valid_from", time("2026-10-05", "day"))
            .with("valid_until", time("whenever", "day")),
            claim("Tim is at a conference.", "event", "The conference runs")
                .with("valid_from", time("soonish", "day")),
        ],
    );
    let first = h.row(memories[0]);
    assert_eq!(first.valid_from, timed(local("2026-10-05T00:00"), "day"));
    assert_eq!(first.valid_until, None);
    assert_eq!(first.window_confidence, "low");
    // An event left with no start falls back to the day it was said.
    let second = h.row(memories[1]);
    assert_eq!(second.valid_from, timed(local("2026-10-01T00:00"), "day"));
    assert_eq!(second.window_confidence, "low");
}

#[test]
fn a_documents_memory_is_observed_at_its_reference_date_and_created_at_ingest() {
    let h = Harness::new();
    let text = "Yesterday I met Ana at the market.";
    let ingested = ingest_doc(&h, &document("diary", text, date(2026, 9, 28)));
    let ingested_at = h.now();
    h.advance(2);

    let input = input(&h, "main", &[]);
    assert_eq!(input.reference_date, Some(date(2026, 9, 28)));
    assert_eq!(input.calendar[CALENDAR_DAYS as usize], date(2026, 9, 28));
    let quote = "I met Ana at the market";
    let content = "Tim met Ana at the market on 27 September 2026.";
    let memories = extract(
        &h,
        reply(
            vec![claim(content, "event", quote).with("valid_from", time("2026-09-27", "day"))],
            &[],
        ),
    )
    .memories;

    let observed_at = local("2026-09-28T00:00");
    assert_eq!(
        h.row(memories[0]),
        Row {
            valid_from: timed(local("2026-09-27T00:00"), "day"),
            ..Row::new(content, "event", observed_at, span(text, quote))
        }
    );
    // TIM-92: the created access is the source's ingested_at, never the time
    // extraction ran, so a backdated document doesn't arrive faded.
    assert_eq!(
        h.accesses(memories[0]),
        vec![AccessRow {
            kind: "created".into(),
            at: ingested_at,
            turn: 0,
            source: Some(ingested.source),
        }]
    );
    let created_at: i64 = h.one(
        "SELECT created_at FROM memories WHERE uuid = ?1",
        [memories[0].to_string()],
    );
    assert_eq!(timestamp(created_at), h.now());
}

// Speakers.

#[test]
fn the_owners_claims_link_to_user() {
    let h = Harness::new();
    ingest(
        &h,
        &turn("s1", T1, "I adopted a cat called Miso.", "Lovely!"),
    );
    let input = input(&h, "main", &[]);
    let me = input.speaker.as_ref().unwrap().handle.clone();
    let user = h.seeded("main", "user");
    let aliases_before = h.aliases(user);

    let extracted = extract(
        &h,
        reply(
            vec![
                claim(
                    "Tim adopted a cat called Miso.",
                    "event",
                    "I adopted a cat called Miso",
                )
                .with(
                    "entities",
                    json!([link(&me, "I"), new_entity("Miso", "thing", "Miso")]),
                ),
            ],
            &[],
        ),
    );
    let miso = h.entities_named("main", "Miso");
    assert_eq!(miso.len(), 1);
    assert_eq!(extracted.entities_created, miso);
    assert_eq!(
        h.links(extracted.memories[0]),
        BTreeSet::from([(user, Some("I".into())), (miso[0], Some("Miso".into()))])
    );
    // A pronoun is kept on the link but never becomes an alias.
    assert_eq!(h.aliases(user), aliases_before);
    assert_eq!(h.aliases(miso[0]), vec!["Miso".to_string()]);
    assert_eq!(h.entity_kind(miso[0]), "thing");
}

#[test]
fn another_speakers_claims_link_to_their_own_entity() {
    let h = Harness::new();
    let user_text = "I'm moving to Lisbon in November.";
    let ingested = ingest(&h, &sams_turn(T1, user_text, "Exciting!"));
    let sam = ingested.speaker.as_ref().unwrap().entity;
    let input = input(&h, "main", &[]);
    let me = input.speaker.as_ref().unwrap().handle.clone();
    assert_eq!(me, handle(&input, sam));

    let quote = "I'm moving to Lisbon in November";
    let content = "Sam is moving to Lisbon in November 2026.";
    let memories = extract(
        &h,
        reply(
            vec![
                claim(content, "event", quote)
                    .with("valid_from", time("2026-11", "month"))
                    .with(
                        "entities",
                        json!([link(&me, "I"), new_entity("Lisbon", "place", "Lisbon")]),
                    ),
            ],
            &[],
        ),
    )
    .memories;
    let lisbon = h.entities_named("main", "Lisbon")[0];
    assert_eq!(
        h.links(memories[0]),
        BTreeSet::from([(sam, Some("I".into())), (lisbon, Some("Lisbon".into()))])
    );
    assert_eq!(h.entity_kind(lisbon), "place");
    assert_eq!(
        h.row(memories[0]),
        Row {
            valid_from: timed(local("2026-11-01T00:00"), "month"),
            ..Row::new(
                content,
                "event",
                at(T1),
                span(&turn_text(user_text, "Exciting!"), quote)
            )
        }
    );
}

#[test]
fn an_answer_to_the_assistants_question_quotes_the_answer() {
    let h = Harness::new();
    ingest(
        &h,
        &turn(
            "s1",
            "2026-10-01T06:00:00Z",
            "Morning.",
            "Are you still at Acme?",
        ),
    );
    let current = ingest(&h, &turn("s1", T1, "Yes, still there.", "Great."));
    h.focus(h.chunk_of(current.source, 0));
    let input = input(&h, "main", &[]);
    assert_eq!(
        input.context,
        vec![turn_text("Morning.", "Are you still at Acme?")]
    );

    // TIM-92: "Yes" after "Are you still at Acme?" is the user's claim,
    // written out in full from the context but quoted from the answer.
    let extracted = extract(
        &h,
        reply(
            vec![
                claim("Tim still works at Acme.", "fact", "Yes, still there"),
                claim("Tim works at Acme.", "fact", "Are you still at Acme?"),
            ],
            &[],
        ),
    );
    assert_eq!(extracted.memories.len(), 1);
    assert_eq!(
        h.row(extracted.memories[0]),
        Row::new("Tim still works at Acme.", "fact", at(T1), (0, 16))
    );
    assert_eq!(
        extracted.dropped,
        vec![Dropped {
            claim: 1,
            reason: DropReason::QuoteNotFound,
        }]
    );
}

// Significance.

#[test]
fn each_level_is_stored_as_given() {
    let h = Harness::new();
    let claims = LEVELS
        .iter()
        .map(|level| {
            claim(&format!("Tim said a {level} thing."), "fact", "Five things")
                .with("significance", json!(level))
        })
        .collect();
    let memories = golden(&h, "Five things.", "Ok.", claims);
    let stored: Vec<(String, Option<String>)> = memories
        .iter()
        .map(|m| {
            let row = h.row(*m);
            (row.significance, row.owner_significance)
        })
        .collect();
    let expected: Vec<(String, Option<String>)> = LEVELS
        .iter()
        .map(|level| (level.to_string(), None))
        .collect();
    assert_eq!(stored, expected);
}

#[test]
fn a_significance_above_critical_is_an_invalid_reply() {
    let h = Harness::new();
    ingest(&h, &turn("s1", T1, "This matters.", "Noted."));
    for (attempt, significance) in [json!("kept"), json!(0.95), json!(1.0)]
        .into_iter()
        .enumerate()
    {
        let llm = FakeLlm::scripted(
            MODEL,
            vec![reply(
                vec![
                    claim("Tim said this matters.", "fact", "This matters")
                        .with("significance", significance),
                ],
                &[],
            )],
        );
        let error = run(&h, "main", &llm, &[]).unwrap_err();
        let error_count = u32::try_from(attempt).unwrap() + 1;
        assert!(
            matches!(
                error,
                ExtractError::InvalidReply { failure: Failure::Retry { error_count: n }, .. }
                    if n == error_count
            ),
            "{error:?}"
        );
        assert_eq!(h.memories_in("main"), 0);
    }
}

#[test]
fn remember_this_from_the_owner_keeps_the_memory() {
    let h = Harness::new();
    let memories = golden(
        &h,
        "Remember this: my passport number ends in 42.",
        "Got it.",
        vec![
            claim(
                "Tim's passport number ends in 42.",
                "fact",
                "my passport number ends in 42",
            )
            .with("significance", json!("notable"))
            .with("remember_this", json!(true)),
        ],
    );
    let row = h.row(memories[0]);
    // The owner's keep sits beside the level extraction gave, so unkeep can
    // hand it back (TIM-94, decision 9).
    assert_eq!(row.owner_significance, Some("kept".into()));
    assert_eq!(row.significance, "notable");
}

#[test]
fn remember_this_from_another_speaker_is_capped_at_critical() {
    let h = Harness::new();
    ingest(
        &h,
        &sams_turn(T1, "Remember this: I'm allergic to peanuts.", "Noted, Sam."),
    );
    let memories = extract(
        &h,
        reply(
            vec![
                claim(
                    "Sam is allergic to peanuts.",
                    "fact",
                    "I'm allergic to peanuts",
                )
                .with("significance", json!("critical"))
                .with("remember_this", json!(true)),
            ],
            &[],
        ),
    )
    .memories;
    let row = h.row(memories[0]);
    assert_eq!(row.owner_significance, None);
    assert_eq!(row.significance, "critical");
}

#[test]
fn remember_this_in_a_document_is_ignored() {
    let h = Harness::new();
    ingest_doc(
        &h,
        &document(
            "fridge",
            "Remember this: the wifi password is on the fridge.",
            date(2026, 9, 30),
        ),
    );
    let memories = extract(
        &h,
        reply(
            vec![
                claim(
                    "The wifi password is on the fridge.",
                    "fact",
                    "the wifi password is on the fridge",
                )
                .with("significance", json!("major"))
                .with("remember_this", json!(true)),
            ],
            &[],
        ),
    )
    .memories;
    let row = h.row(memories[0]);
    assert_eq!(row.owner_significance, None);
    assert_eq!(row.significance, "major");
}

#[test]
fn remember_this_quoted_from_the_reply_is_ignored() {
    let h = Harness::new();
    // Only the owner's own message can keep a memory (TIM-92, other
    // decision 3).
    let memories = golden(
        &h,
        "When is Ana's birthday again?",
        "Remember this: Ana's birthday is 4 May.",
        vec![
            claim(
                "Ana's birthday is 4 May.",
                "fact",
                "Ana's birthday is 4 May",
            )
            .with("remember_this", json!(true)),
        ],
    );
    assert_eq!(h.row(memories[0]).owner_significance, None);
}

// Quotes.

#[test]
fn offsets_locate_the_quote_in_the_chunk_in_characters() {
    let h = Harness::new();
    let user = "Café ☕ with Ana on Friday.";
    let assistant = "Lunch with Ana, then lunch with Ana again.";
    let text = turn_text(user, assistant);
    let memories = golden(
        &h,
        user,
        assistant,
        vec![
            claim("Tim is having coffee with Ana.", "fact", "with Ana"),
            claim("Tim is having two lunches.", "fact", "then lunch"),
        ],
    );
    // The first occurrence, counted in characters, not bytes.
    let first = h.row(memories[0]);
    assert_eq!((first.source_start, first.source_end), (7, 15));
    // The reply starts after the message and the separator.
    let second = h.row(memories[1]);
    assert_eq!((second.source_start, second.source_end), (44, 54));
    assert_eq!(chars(&text, 7, 15), "with Ana");
    assert_eq!(chars(&text, 44, 54), "then lunch");
}

#[test]
fn a_claim_without_a_quote_from_the_chunk_is_dropped() {
    let h = Harness::new();
    ingest(
        &h,
        &turn(
            "s1",
            "2026-10-01T06:00:00Z",
            "Morning.",
            "Are you still at Acme?",
        ),
    );
    let current = ingest(&h, &turn("s1", T1, "Yes, still there.", "Great."));
    h.focus(h.chunk_of(current.source, 0));

    let extracted = extract(
        &h,
        reply(
            vec![
                claim("Tim still works at Acme.", "fact", "Yes, still there"),
                claim("Tim works at Acme.", "fact", "Are you still at Acme?"),
                claim("Tim likes Acme.", "fact", "I love working at Acme"),
                claim("Tim is at Acme.", "fact", ""),
                claim("  ", "fact", "still there"),
            ],
            &[],
        ),
    );
    assert_eq!(extracted.memories.len(), 1);
    assert_eq!(
        extracted.dropped,
        vec![
            Dropped {
                claim: 1,
                reason: DropReason::QuoteNotFound,
            },
            Dropped {
                claim: 2,
                reason: DropReason::QuoteNotFound,
            },
            Dropped {
                claim: 3,
                reason: DropReason::QuoteNotFound,
            },
            Dropped {
                claim: 4,
                reason: DropReason::EmptyContent,
            },
        ]
    );
    assert_eq!(h.memories_in("main"), 1);
}

#[test]
fn an_assistant_task_needs_a_due_date_or_an_until_event() {
    let h = Harness::new();
    ingest(
        &h,
        &turn(
            "s1",
            T1,
            "Can you send me the report by Friday? And look into the backup sometime.",
            "I'll send you the report by Friday. I'll look into the backup.",
        ),
    );
    let extracted = extract(
        &h,
        reply(
            vec![
                claim(
                    "Hermes will send Tim the report by 2 October 2026.",
                    "task",
                    "I'll send you the report by Friday",
                )
                .with("due_at", time("2026-10-02", "day")),
                claim(
                    "Hermes will look into the backup.",
                    "task",
                    "I'll look into the backup",
                ),
                claim(
                    "Hermes will look into the backup until it's restored.",
                    "task",
                    "I'll look into the backup",
                )
                .with("until_event", json!("the backup is restored")),
                // The user's own undated task stays.
                claim(
                    "Tim wants the backup looked into.",
                    "task",
                    "look into the backup sometime",
                ),
            ],
            &[],
        ),
    );
    assert_eq!(extracted.memories.len(), 3);
    assert_eq!(
        extracted.dropped,
        vec![Dropped {
            claim: 1,
            reason: DropReason::AssistantTaskUndated,
        }]
    );
}

// Entities.

#[test]
fn a_link_to_a_candidate_records_its_surface_form() {
    let h = Harness::new();
    let ana = h.insert_entity("main", "Ana", "person", &["Ana"]);
    ingest(&h, &turn("s1", T1, "Lunch with Ana.", "Enjoy."));
    let input = input(&h, "main", &[]);
    let edits_before = h.edits("main", "alias_added");

    let memories = extract(
        &h,
        reply(
            vec![
                claim("Tim is having lunch with Ana.", "event", "Lunch with Ana")
                    .with("entities", json!([link(&handle(&input, ana), "Ana")])),
            ],
            &[],
        ),
    )
    .memories;
    assert_eq!(
        h.links(memories[0]),
        BTreeSet::from([(ana, Some("Ana".into()))])
    );
    assert_eq!(h.aliases(ana), vec!["Ana".to_string()]);
    assert_eq!(h.edits("main", "alias_added"), edits_before);
}

#[test]
fn a_new_surface_form_becomes_a_logged_alias() {
    let h = Harness::new();
    let ana = h.insert_entity("main", "Ana", "person", &["Ana"]);
    ingest(
        &h,
        &turn("s1", T1, "Annie (that's Ana) is visiting me.", "Lovely."),
    );
    let input = input(&h, "main", &[]);
    let ana_handle = handle(&input, ana);
    let me = input.speaker.as_ref().unwrap().handle.clone();
    let user = h.seeded("main", "user");
    let user_aliases = h.aliases(user);
    let edits_before = h.edits("main", "alias_added");

    extract(
        &h,
        reply(
            vec![
                claim(
                    "Ana is visiting Tim.",
                    "event",
                    "Annie (that's Ana) is visiting me",
                )
                .with(
                    "entities",
                    json!([link(&ana_handle, "Annie"), link(&me, "me")]),
                ),
                claim("Ana is also called Annie.", "fact", "Annie (that's Ana)")
                    .with("entities", json!([link(&ana_handle, "Ana")])),
            ],
            &[],
        ),
    );
    // TIM-92: a new surface form is added as an alias in a logged edit, so a
    // mislink can be undone. A known one and a pronoun aren't.
    assert_eq!(h.aliases(ana), vec!["Ana".to_string(), "Annie".to_string()]);
    assert_eq!(h.aliases(user), user_aliases);
    assert_eq!(h.edits("main", "alias_added"), edits_before + 1);
    let logged_for: String = h.one(
        "SELECT e.uuid FROM edits d JOIN entities e ON e.id = d.entity_id
         WHERE d.kind = 'alias_added' ORDER BY d.id DESC LIMIT 1",
        [],
    );
    assert_eq!(logged_for, ana.to_string());
}

#[test]
fn a_proposed_entity_is_created_once_per_reply() {
    let h = Harness::new();
    let created_before = h.edits("main", "entity_created");
    let extracted = golden(
        &h,
        "Lisbon was hot. We loved Lisbon.",
        "Sounds great.",
        vec![
            claim("Lisbon was hot.", "event", "Lisbon was hot")
                .with("entities", json!([new_entity("Lisbon", "place", "Lisbon")])),
            claim("Tim loved Lisbon.", "fact", "We loved Lisbon")
                .with("entities", json!([new_entity("Lisbon", "place", "Lisbon")])),
        ],
    );
    let lisbon = h.entities_named("main", "Lisbon");
    assert_eq!(lisbon.len(), 1);
    for memory in &extracted {
        assert_eq!(
            h.links(*memory),
            BTreeSet::from([(lisbon[0], Some("Lisbon".into()))])
        );
    }
    assert_eq!(h.aliases(lisbon[0]), vec!["Lisbon".to_string()]);
    assert_eq!(h.edits("main", "entity_created"), created_before + 1);
    assert!(h.entities_named("other", "Lisbon").is_empty());
}

#[test]
fn a_new_entity_beside_one_call_1_saw_is_a_second_entity() {
    let h = Harness::new();
    let sam = h.insert_entity("main", "Sam", "person", &["Sam"]);
    ingest(&h, &turn("s1", T1, "Sam from work called.", "Which Sam?"));
    let input = input(&h, "main", &[]);
    handle(&input, sam);

    // Call 1 saw Sam and chose a new person: the two-Sams judgement stands
    // (TIM-92).
    let extracted = extract(
        &h,
        reply(
            vec![
                claim(
                    "Sam from Tim's work called Tim.",
                    "event",
                    "Sam from work called",
                )
                .with("entities", json!([new_entity("Sam", "person", "Sam")])),
            ],
            &[],
        ),
    );
    let sams = h.entities_named("main", "Sam");
    assert_eq!(sams.len(), 2);
    assert_eq!(extracted.entities_created, vec![sams[1]]);
    assert_eq!(
        h.links(extracted.memories[0]),
        BTreeSet::from([(sams[1], Some("Sam".into()))])
    );
}

#[test]
fn a_proposed_entity_never_reuses_one_that_existed_before_call_1() {
    let h = Harness::new();
    let (entities, message) = crowded(&h);
    ingest(&h, &turn("s1", T1, &message, "Quite a list."));
    let input = input(&h, "main", &[]);
    assert!(!found(&h, &input).contains(&entities[0]));

    // Name00 missed the cap, so call 1 never compared it and proposed a new
    // entity. TIM-92 reuses only an entity created after call 1 ran, so this
    // is a second Name00, not the one call 1 never saw.
    let extracted = extract(
        &h,
        reply(
            vec![
                claim("Name00 is on the list.", "fact", "Name00")
                    .with("entities", json!([new_entity("Name00", "thing", "Name00")])),
            ],
            &[],
        ),
    );
    let named = h.entities_named("main", "Name00");
    assert_eq!(named.len(), 2);
    assert_eq!(named[0], entities[0]);
    assert_eq!(extracted.entities_created, vec![named[1]]);
    assert_eq!(
        h.links(extracted.memories[0]),
        BTreeSet::from([(named[1], Some("Name00".into()))])
    );
}

/// Call 1 answering `reply`, running `during` while the call is in flight,
/// as another writer to the bank would.
struct Meanwhile<'a> {
    h: &'a Harness,
    during: fn(&Harness),
    reply: Value,
}

impl LlmClient for Meanwhile<'_> {
    fn model(&self) -> &str {
        MODEL
    }

    fn complete(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        (self.during)(self.h);
        Ok(LlmResponse {
            json: self.reply.clone(),
            usage: None,
            latency: Duration::ZERO,
        })
    }
}

#[test]
fn a_proposed_entity_reuses_one_created_while_call_1_ran() {
    let h = Harness::new();
    ingest(&h, &turn("s1", T1, "Lisbon was hot.", "Sounds warm."));
    let llm = Meanwhile {
        h: &h,
        during: |h| {
            h.insert_entity("main", "Lisbon", "place", &["Lisbon"]);
        },
        reply: reply(
            vec![
                claim("Lisbon was hot.", "event", "Lisbon was hot")
                    .with("entities", json!([new_entity("Lisbon", "place", "Lisbon")])),
            ],
            &[],
        ),
    };
    let extracted = h
        .service
        .extract_chunk(lease(&h, "main"), &llm, &[])
        .unwrap();

    // TIM-92: code repeats the exact alias lookup at commit and links the
    // entity created after call 1 ran, rather than a duplicate.
    let lisbon = h.entities_named("main", "Lisbon");
    assert_eq!(lisbon.len(), 1);
    assert!(extracted.entities_created.is_empty());
    assert_eq!(
        h.links(extracted.memories[0]),
        BTreeSet::from([(lisbon[0], Some("Lisbon".into()))])
    );
}

#[test]
fn a_link_that_names_no_entity_is_dropped() {
    let h = Harness::new();
    let entities_before = h.count("SELECT COUNT(*) FROM entities");
    let aliases_before = h.count("SELECT COUNT(*) FROM entity_aliases");
    let memories = golden(
        &h,
        "Zed came by.",
        "Who's Zed?",
        vec![claim("Zed visited Tim.", "event", "Zed came by").with(
            "entities",
            json!([
                link("e999", "Zed"),
                {"entity": null, "new_name": null, "new_kind": null, "surface_form": "Zed"},
            ]),
        )],
    );
    assert!(h.links(memories[0]).is_empty());
    assert_eq!(h.count("SELECT COUNT(*) FROM entities"), entities_before);
    assert_eq!(
        h.count("SELECT COUNT(*) FROM entity_aliases"),
        aliases_before
    );
}

// Accesses.

#[test]
fn each_new_memory_gets_a_created_access_at_ingest_time() {
    let h = Harness::new();
    let ingested = ingest(&h, &turn("s1", T1, "I like tea.", "Noted."));
    let ingested_at = h.now();
    h.advance(1);
    let memories = extract(
        &h,
        reply(vec![claim("Tim likes tea.", "fact", "I like tea")], &[]),
    )
    .memories;
    assert_eq!(
        h.accesses(memories[0]),
        vec![AccessRow {
            kind: "created".into(),
            at: ingested_at,
            turn: 1,
            source: Some(ingested.source),
        }]
    );
}

#[test]
fn used_verdicts_write_one_used_access_each() {
    let h = Harness::new();
    let tea = h.insert_memory("main", "Tim likes tea.", "minor");
    let coffee = h.insert_memory("main", "Tim hates coffee.", "minor");
    let ingested = ingest(
        &h,
        &turn("s1", T1, "What should I drink?", "Tea, as you like it."),
    );
    let turn_number = h.turns("main");
    let ingested_at = h.now();
    h.advance(1);
    let input = input(&h, "main", &[tea, coffee]);
    let tea_handle = memory_handle(&input, tea);

    let extracted = extract_with(
        &h,
        reply(vec![], &[&tea_handle, &tea_handle, "m999"]),
        &[tea, coffee],
    );
    assert_eq!(extracted.used, vec![tea]);
    assert!(extracted.memories.is_empty());
    let accesses = h.accesses(tea);
    assert_eq!(accesses.len(), 2);
    assert_eq!(
        accesses[1],
        AccessRow {
            kind: "used".into(),
            at: ingested_at,
            turn: turn_number,
            source: Some(ingested.source),
        }
    );
    assert_eq!(h.accesses(coffee).len(), 1);
}

#[test]
fn a_used_verdict_keeps_a_stronger_access_in_the_same_turn() {
    let h = Harness::new();
    let tea = h.insert_memory("main", "Tim likes tea.", "minor");
    let cake = h.insert_memory("main", "Tim likes cake.", "minor");
    ingest(&h, &turn("s1", T1, "Tea and cake?", "Tea and cake it is."));
    let turn_number = h.turns("main");
    h.insert_access(tea, "confirmed", turn_number);
    h.insert_access(cake, "used", turn_number);
    let input = input(&h, "main", &[tea, cake]);
    let handles = [memory_handle(&input, tea), memory_handle(&input, cake)];

    extract_with(&h, reply(vec![], &[&handles[0], &handles[1]]), &[tea, cake]);
    // TIM-90: at most one access per memory per turn, keeping the strongest.
    let in_turn = |memory: Uuid| -> Vec<String> {
        h.accesses(memory)
            .into_iter()
            .filter(|access| access.turn == turn_number)
            .map(|access| access.kind)
            .collect()
    };
    assert_eq!(in_turn(tea), vec!["confirmed".to_string()]);
    assert_eq!(in_turn(cake), vec!["used".to_string()]);
}

#[test]
fn accesses_carry_the_turn_number_of_their_source() {
    let h = Harness::new();
    // Everything is ingested before anything is extracted, so the bank's
    // counter has moved on by the time each chunk is.
    ingest(&h, &turn("s1", "2026-10-01T06:00:00Z", "One.", "Ok."));
    ingest(&h, &turn("s1", "2026-10-01T06:10:00Z", "Two.", "Ok."));
    ingest_doc(&h, &document("notes", "Doc.", date(2026, 9, 30)));
    ingest(&h, &turn("s1", "2026-10-01T06:20:00Z", "Three.", "Ok."));
    assert_eq!(h.turns("main"), 3);

    // Turns first in observed_at order, then the document (TIM-92).
    let mut turns = Vec::new();
    for quote in ["One", "Two", "Three", "Doc"] {
        let memory = extract_unlabelled(
            &h,
            reply(
                vec![claim(&format!("Tim said {quote}."), "fact", quote)],
                &[],
            ),
        )
        .memories[0];
        turns.push(h.accesses(memory)[0].turn);
    }
    // A document takes the counter as it stood when it was ingested.
    assert_eq!(turns, vec![1, 2, 3, 2]);
}

// Commit.

#[test]
fn a_commit_writes_vectors_and_marks_the_chunk_extracted() {
    let h = Harness::new();
    let ingested = ingest(&h, &turn("s1", T1, "I like tea.", "Noted."));
    let chunk = h.chunk_of(ingested.source, 0);
    h.advance(1);
    let extracted = extract(
        &h,
        reply(vec![claim("Tim likes tea.", "fact", "I like tea")], &[]),
    );
    assert_eq!(extracted.chunk, chunk);
    let memory = extracted.memories[0];

    let bytes: Vec<u8> = h.one(
        "SELECT v.embedding FROM memory_vectors v JOIN memories m ON m.id = v.memory_id
         WHERE m.uuid = ?1",
        [memory.to_string()],
    );
    let vector: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert_eq!(vector, FakeEmbedder.embed(&["Tim likes tea."]).unwrap()[0]);

    let (memory_chunk, created_at, updated_at): (String, i64, i64) = h
        .service
        .store()
        .unwrap()
        .connection()
        .query_row(
            "SELECT c.uuid, m.created_at, m.updated_at FROM memories m
             JOIN chunks c ON c.id = m.chunk_id WHERE m.uuid = ?1",
            [memory.to_string()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(memory_chunk, chunk.to_string());
    assert_eq!(timestamp(created_at), h.now());
    assert_eq!(timestamp(updated_at), h.now());

    assert_eq!(
        h.chunk_column::<Option<i64>>(chunk, "extracted_at"),
        Some(micros(h.now()))
    );
    assert_eq!(
        h.chunk_column::<Option<String>>(chunk, "call1_output"),
        None
    );
    assert_eq!(h.chunk_column::<i64>(chunk, "error_count"), 0);
    assert_eq!(h.service.queue_depth("main").unwrap(), 0);
    // The lease was released: the bank's next chunk can be claimed.
    ingest(&h, &turn("s1", "2026-10-01T06:40:00Z", "More.", "Ok."));
    assert!(h.service.claim_chunk("main").unwrap().is_some());
}

#[test]
fn a_reply_with_no_claims_completes_the_chunk() {
    let h = Harness::new();
    let ingested = ingest(&h, &turn("s1", T1, "Thanks!", "You're welcome."));
    let extracted = extract(&h, reply(vec![], &[]));
    assert!(extracted.memories.is_empty());
    assert!(extracted.dropped.is_empty());
    assert!(
        h.chunk_column::<Option<i64>>(h.chunk_of(ingested.source, 0), "extracted_at")
            .is_some()
    );
    assert_eq!(h.service.queue_depth("main").unwrap(), 0);
}

// Failure. A failure in call 1 leaves the chunk retryable.

/// What `FakeLlm::failing` fails with.
type MakeError = fn() -> LlmError;

/// Everything a failed extraction must not have touched.
fn untouched(h: &Harness) -> [i64; 6] {
    [
        h.count("SELECT COUNT(*) FROM memories"),
        h.count("SELECT COUNT(*) FROM memory_entities"),
        h.count("SELECT COUNT(*) FROM entities"),
        h.count("SELECT COUNT(*) FROM entity_aliases"),
        h.count("SELECT COUNT(*) FROM edits"),
        h.count("SELECT COUNT(*) FROM accesses"),
    ]
}

/// A reply with a claim that would create an entity and a used verdict, so
/// a partial commit would show.
fn busy_reply(input: &Call1Input, used: Uuid) -> Value {
    reply(
        vec![
            claim("Tim is moving to Lisbon.", "event", "I'm moving to Lisbon")
                .with("entities", json!([new_entity("Lisbon", "place", "Lisbon")])),
        ],
        &[&memory_handle(input, used)],
    )
}

#[test]
fn a_failed_call_1_leaves_the_chunk_retryable() {
    let h = Harness::new();
    let tea = h.insert_memory("main", "Tim likes tea.", "minor");
    let ingested = ingest(&h, &turn("s1", T1, "I'm moving to Lisbon.", "Exciting!"));
    let chunk = h.chunk_of(ingested.source, 0);
    let before = untouched(&h);

    let llm = FakeLlm::failing(MODEL, || LlmError::Status { status: 502 });
    let error = run(&h, "main", &llm, &[tea]).unwrap_err();
    assert!(
        matches!(
            error,
            ExtractError::Call1 {
                error: LlmError::Status { status: 502 },
                failure: Failure::Retry { error_count: 1 },
            }
        ),
        "{error:?}"
    );
    assert_eq!(error.failure(), Some(Failure::Retry { error_count: 1 }));
    assert_eq!(untouched(&h), before);
    assert_eq!(h.chunk_column::<Option<i64>>(chunk, "extracted_at"), None);
    assert_eq!(
        h.chunk_column::<Option<String>>(chunk, "call1_output"),
        None
    );
    assert_eq!(h.chunk_column::<i64>(chunk, "error_count"), 1);
    assert_eq!(
        h.chunk_column::<Option<String>>(chunk, "last_error_kind"),
        Some("llm_status".into())
    );
    assert_eq!(
        h.chunk_column::<Option<i64>>(chunk, "last_error_status"),
        Some(502)
    );

    // Still at the head of the queue, and a good reply then commits once.
    let lease = lease(&h, "main");
    assert_eq!(lease.chunk, chunk);
    assert_eq!(lease.error_count, 1);
    let input = h.service.call1_input(&lease, &[tea]).unwrap();
    drop(lease);
    let extracted = extract_with(&h, busy_reply(&input, tea), &[tea]);
    assert_eq!(extracted.memories.len(), 1);
    assert_eq!(h.entities_named("main", "Lisbon").len(), 1);
    assert_eq!(h.accesses(tea).len(), 2);
}

#[test]
fn each_llm_failure_is_recorded_by_kind() {
    let h = Harness::new();
    let ingested = ingest(&h, &turn("s1", T1, "Hello.", "Hi."));
    let chunk = h.chunk_of(ingested.source, 0);
    let failing: [(MakeError, &str); 3] = [
        (|| LlmError::Timeout, "llm_timeout"),
        (|| LlmError::NotJson { bytes: 12 }, "llm_not_json"),
        (
            || LlmError::Transport {
                reason: "connection reset".into(),
            },
            "llm_transport",
        ),
    ];
    for (make, kind) in failing {
        let error = run(&h, "main", &FakeLlm::failing(MODEL, make), &[]).unwrap_err();
        assert!(matches!(error, ExtractError::Call1 { .. }), "{error:?}");
        assert_eq!(
            h.chunk_column::<Option<String>>(chunk, "last_error_kind"),
            Some(kind.into())
        );
        assert_eq!(
            h.chunk_column::<Option<i64>>(chunk, "last_error_status"),
            None
        );
    }
    assert_eq!(h.chunk_column::<i64>(chunk, "error_count"), 3);
}

#[test]
fn an_invalid_reply_leaves_the_chunk_retryable() {
    let h = Harness::new();
    let ingested = ingest(&h, &turn("s1", T1, "I like tea.", "Noted."));
    let chunk = h.chunk_of(ingested.source, 0);
    let before = untouched(&h);
    let invalid = [
        json!({"facts": []}),
        reply(vec![claim("Tim likes tea.", "opinion", "I like tea")], &[]),
        reply(
            vec![
                claim("Tim likes tea.", "fact", "I like tea")
                    .with("window_confidence", json!("medium")),
            ],
            &[],
        ),
        reply(
            vec![
                claim("Tim likes tea.", "fact", "I like tea")
                    .with("entities", json!([new_entity("Tea", "drink", "tea")])),
            ],
            &[],
        ),
    ];
    for (attempt, reply) in invalid.into_iter().enumerate() {
        let error = run(&h, "main", &FakeLlm::scripted(MODEL, vec![reply]), &[]).unwrap_err();
        let error_count = u32::try_from(attempt).unwrap() + 1;
        assert_eq!(
            error.failure(),
            Some(Failure::Retry { error_count }),
            "{error:?}"
        );
        assert!(
            matches!(error, ExtractError::InvalidReply { .. }),
            "{error:?}"
        );
        assert_eq!(untouched(&h), before);
    }
    assert_eq!(
        h.chunk_column::<Option<String>>(chunk, "last_error_kind"),
        Some("invalid_reply".into())
    );
    assert_eq!(h.chunk_column::<Option<i64>>(chunk, "extracted_at"), None);
}

#[test]
fn the_retry_cap_marks_a_chunk_failed() {
    let h = Harness::new();
    let ingested = ingest(&h, &turn("s1", T1, "Hello.", "Hi."));
    let chunk = h.chunk_of(ingested.source, 0);
    let llm = FakeLlm::failing(MODEL, || LlmError::Status { status: 502 });
    for attempt in 1..CHUNK_RETRY_CAP {
        let error = run(&h, "main", &llm, &[]).unwrap_err();
        assert_eq!(
            error.failure(),
            Some(Failure::Retry {
                error_count: attempt
            })
        );
    }
    let error = run(&h, "main", &llm, &[]).unwrap_err();
    assert_eq!(error.failure(), Some(Failure::Failed));

    assert_eq!(h.service.queue_depth("main").unwrap(), 0);
    let failed = h.service.failed_chunks("main").unwrap();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].chunk, chunk);
    assert_eq!(failed[0].error_kind, "llm_status");
    assert_eq!(failed[0].status, Some(502));
    assert_eq!(h.memories_in("main"), 0);
}

#[test]
fn an_unusable_llm_holds_the_queue_without_counting() {
    let h = Harness::new();
    let ingested = ingest(&h, &turn("s1", T1, "Hello.", "Hi."));
    let chunk = h.chunk_of(ingested.source, 0);
    let held: [MakeError; 3] = [
        || LlmError::UsageLimited {
            resets_at: "2026-10-01T12:00:00Z".parse().unwrap(),
        },
        || LlmError::LoginRequired,
        || LlmError::NotConfigured {
            missing: "llm.model",
        },
    ];
    for make in held {
        let error = run(&h, "main", &FakeLlm::failing(MODEL, make), &[]).unwrap_err();
        assert!(matches!(error, ExtractError::Held { .. }), "{error:?}");
        assert_eq!(error.failure(), None);
    }
    assert_eq!(h.chunk_column::<i64>(chunk, "error_count"), 0);
    assert_eq!(h.chunk_column::<Option<i64>>(chunk, "failed_at"), None);
    let lease = lease(&h, "main");
    assert_eq!(lease.chunk, chunk);
    assert_eq!(lease.error_count, 0);
}

#[test]
fn an_embedding_failure_writes_nothing() {
    let h = Harness::with_models(Models {
        embedder: Arc::new(FailingEmbedder),
        reranker: Arc::new(FakeReranker),
    });
    let tea = h.insert_memory("main", "Tim likes tea.", "minor");
    let ingested = ingest(&h, &turn("s1", T1, "I'm moving to Lisbon.", "Exciting!"));
    let chunk = h.chunk_of(ingested.source, 0);
    let input = input(&h, "main", &[tea]);
    let before = untouched(&h);

    let llm = FakeLlm::scripted(MODEL, vec![busy_reply(&input, tea)]);
    let error = run(&h, "main", &llm, &[tea]).unwrap_err();
    assert!(
        matches!(
            error,
            ExtractError::Embedding {
                failure: Failure::Retry { error_count: 1 },
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(untouched(&h), before);
    assert!(h.entities_named("main", "Lisbon").is_empty());
    assert_eq!(
        h.chunk_column::<Option<String>>(chunk, "last_error_kind"),
        Some("embedding".into())
    );
    assert_eq!(h.chunk_column::<Option<i64>>(chunk, "extracted_at"), None);
    assert_eq!(h.service.queue_depth("main").unwrap(), 1);
}

// The TIM-107 review: regressions for its findings, and the guarantees it
// found untested.

/// Ana with three fresh memories linked, at major, notable and minor.
fn ana_with_memories(h: &Harness) -> Uuid {
    let ana = h.insert_entity("main", "Ana", "person", &["Ana"]);
    for (content, level) in [
        ("Ana is a nurse.", "major"),
        ("Ana lives in Wellington.", "notable"),
        ("Ana likes jazz.", "minor"),
    ] {
        let memory = h.insert_memory("main", content, level);
        h.link(memory, ana);
    }
    ana
}

fn candidate_memories(h: &Harness, entity: Uuid) -> Vec<String> {
    let input = input(h, "main", &[]);
    input
        .candidates
        .into_iter()
        .find(|candidate| candidate.entity == entity)
        .expect("a candidate")
        .memories
}

#[test]
fn inherited_accesses_rank_a_candidates_memories() {
    let h = Harness::new();
    let ana = ana_with_memories(&h);
    // A trivial correction that inherits a well-used predecessor's accesses
    // along superseded_by (TIM-91, decision 3). On its own accesses it would
    // rank last.
    let surname = h.insert_memory("main", "Ana's surname is Ngata.", "trivial");
    h.link(surname, ana);
    let predecessor = h.insert_memory("main", "Ana's surname is Ngati.", "minor");
    for (turn, days) in [(1, 30), (2, 20), (3, 10), (4, 5)] {
        let at = h
            .now()
            .checked_sub(SignedDuration::from_hours(24 * days))
            .unwrap();
        h.insert_access_at(predecessor, "confirmed", turn, at);
    }
    h.execute(
        "UPDATE memories SET superseded_by = (SELECT id FROM memories WHERE uuid = ?2)
         WHERE uuid = ?1",
        (predecessor.to_string(), surname.to_string()),
    );
    ingest(&h, &turn("s1", T1, "Ana called.", "How is she?"));

    let memories = candidate_memories(&h, ana);
    assert_eq!(memories.len(), CANDIDATE_MEMORIES);
    assert_eq!(memories[0], "Ana's surname is Ngata.");
    assert!(!memories.contains(&"Ana likes jazz.".to_string()));
}

#[test]
fn a_closed_window_ranks_a_candidates_memories() {
    let h = Harness::new();
    let ana = h.insert_entity("main", "Ana", "person", &["Ana"]);
    for (content, level) in [
        ("Ana likes jazz.", "minor"),
        ("Ana drinks tea.", "trivial"),
        ("Ana has a cat.", "trivial"),
    ] {
        let memory = h.insert_memory("main", content, level);
        h.link(memory, ana);
    }
    // An event said 400 days ago that ended three days ago. Its window's
    // close restarts recent use (ADR 0003), which lifts it above the fresh
    // minor and trivial memories; on its old created access alone it would
    // rank last.
    let exhibition = h.insert_memory_of_kind(
        "main",
        "Ana's exhibition ran until 28 September 2026.",
        "event",
        "critical",
    );
    h.link(exhibition, ana);
    let said = h
        .now()
        .checked_sub(SignedDuration::from_hours(24 * 400))
        .unwrap();
    h.execute(
        "UPDATE memories SET observed_at = ?2, valid_until = ?3, valid_until_precision = 'day'
         WHERE uuid = ?1",
        (
            exhibition.to_string(),
            micros(said),
            micros(local("2026-09-28T00:00")),
        ),
    );
    h.execute(
        "UPDATE accesses SET at = ?2
         WHERE memory_id = (SELECT id FROM memories WHERE uuid = ?1)",
        (exhibition.to_string(), micros(said)),
    );
    ingest(&h, &turn("s1", T1, "Ana called.", "How is she?"));

    let memories = candidate_memories(&h, ana);
    assert_eq!(memories.len(), CANDIDATE_MEMORIES);
    assert_eq!(memories[0], "Ana's exhibition ran until 28 September 2026.");
}

#[test]
fn retracted_memories_are_not_candidate_examples() {
    let h = Harness::new();
    let ana = h.insert_entity("main", "Ana", "person", &["Ana"]);
    let nurse = h.insert_memory("main", "Ana is a nurse.", "major");
    let doctor = h.insert_memory("main", "Ana is a doctor.", "minor");
    h.link(nurse, ana);
    h.link(doctor, ana);
    h.execute(
        "UPDATE memories SET invalidated_at = ?2,
                             superseded_by = (SELECT id FROM memories WHERE uuid = ?3)
         WHERE uuid = ?1",
        (nurse.to_string(), micros(h.now()), doctor.to_string()),
    );
    ingest(&h, &turn("s1", T1, "Ana called.", "How is she?"));
    assert_eq!(
        candidate_memories(&h, ana),
        vec!["Ana is a doctor.".to_string()]
    );
}

#[test]
fn remember_this_on_a_quote_the_reply_repeats_is_not_kept() {
    let h = Harness::new();
    // The owner asks about the sentence and the reply repeats it with
    // "remember this". The quote's first occurrence is in the owner's
    // message, but the claim can't be shown to come from it, so it isn't
    // kept (TIM-92 other decision 3).
    let memories = golden(
        &h,
        "Is \"Ana's birthday is 4 May\" correct?",
        "Remember this: Ana's birthday is 4 May.",
        vec![
            claim(
                "Ana's birthday is 4 May.",
                "fact",
                "Ana's birthday is 4 May",
            )
            .with("significance", json!("notable"))
            .with("remember_this", json!(true)),
        ],
    );
    let row = h.row(memories[0]);
    assert_eq!(row.owner_significance, None);
    assert_eq!(row.significance, "notable");
}

#[test]
fn every_named_weekday_must_fall_on_one_of_the_dates() {
    let h = Harness::new();
    // 5 October 2026 is a Monday and the 8th a Thursday. The Monday start
    // matches, but nothing falls on the Friday the quote names.
    let memories = golden(
        &h,
        "I'm away Monday through Friday.",
        "Enjoy.",
        vec![
            claim(
                "Tim is away from 5 to 8 October 2026.",
                "event",
                "away Monday through Friday",
            )
            .with("valid_from", time("2026-10-05", "day"))
            .with("valid_until", time("2026-10-08", "day")),
        ],
    );
    let row = h.row(memories[0]);
    assert_eq!(row.window_confidence, "low");
    assert_eq!(row.valid_from, timed(local("2026-10-05T00:00"), "day"));
    assert_eq!(row.valid_until, timed(local("2026-10-08T00:00"), "day"));
}

#[test]
fn weekdays_that_each_match_a_date_keep_high_confidence() {
    let h = Harness::new();
    let memories = golden(
        &h,
        "I'm away Monday through Friday.",
        "Enjoy.",
        vec![
            claim(
                "Tim is away from 5 to 9 October 2026.",
                "event",
                "away Monday through Friday",
            )
            .with("valid_from", time("2026-10-05", "day"))
            .with("valid_until", time("2026-10-09", "day")),
        ],
    );
    assert_eq!(h.row(memories[0]).window_confidence, "high");
}

#[test]
fn a_forget_request_and_a_duplicate_keep_turn_numbers_in_step() {
    let h = Harness::new();
    let first = turn("s1", "2026-10-01T06:00:00Z", "One.", "Ok.");
    ingest(&h, &first);
    ingest(
        &h,
        &Turn {
            forget_requested: true,
            ..turn("s1", "2026-10-01T06:05:00Z", "Forget my address.", "Done.")
        },
    );
    // A duplicate stores nothing and doesn't count.
    ingest(&h, &first);
    ingest(&h, &turn("s1", "2026-10-01T06:10:00Z", "Two.", "Ok."));
    assert_eq!(h.turns("main"), 3);

    let mut turns = Vec::new();
    for quote in ["One", "Two"] {
        let memory = extract_unlabelled(
            &h,
            reply(
                vec![claim(&format!("Tim said {quote}."), "fact", quote)],
                &[],
            ),
        )
        .memories[0];
        turns.push(h.accesses(memory)[0].turn);
    }
    // The forget request is turn 2, so the next turn is 3.
    assert_eq!(turns, vec![1, 3]);
}

#[test]
fn aliases_match_whole_words_in_order() {
    let h = Harness::new();
    let acme = h.insert_entity("main", "Acme Corp", "organisation", &["Acme Corp"]);
    h.insert_entity("main", "Ana", "person", &["Ana"]);
    h.insert_entity("main", "Bob Smith", "person", &["Bob Smith"]);
    h.insert_entity(
        "main",
        "Acme Corporation",
        "organisation",
        &["Acme Corporation Ltd"],
    );
    ingest(
        &h,
        &turn(
            "s1",
            T1,
            "Acme Corp. called about a banana for Smith Bob, and Acme Corporation too.",
            "Busy.",
        ),
    );
    // "Ana" isn't a word here, the words of "Bob Smith" are out of order,
    // and "Acme Corporation Ltd" isn't all there.
    assert_eq!(found(&h, &input(&h, "main", &[])), BTreeSet::from([acme]));
}

#[test]
fn aliases_match_with_or_without_diacritics() {
    let h = Harness::new();
    // The alias FTS removes diacritics (`remove_diacritics 2`), so matching
    // agrees with it both ways round.
    let lucia = h.insert_entity("main", "Lucía", "person", &["Lucía"]);
    let zoe = h.insert_entity("main", "Zoe", "person", &["Zoe"]);
    ingest(&h, &turn("s1", T1, "Lucia and Zoë came over.", "Lovely."));
    assert_eq!(
        found(&h, &input(&h, "main", &[])),
        BTreeSet::from([lucia, zoe])
    );
}

/// The entities found as candidates when `main` has an entity per alias and
/// the owner says `message`. Each alias is its entity's name and only alias.
fn candidates_for(aliases: &[&str], message: &str) -> (Vec<Uuid>, BTreeSet<Uuid>) {
    let h = Harness::new();
    let entities: Vec<Uuid> = aliases
        .iter()
        .map(|alias| h.insert_entity("main", alias, "person", &[alias]))
        .collect();
    ingest(&h, &turn("s1", T1, message, "Lovely."));
    let found = found(&h, &input(&h, "main", &[]));
    (entities, found)
}

#[test]
fn a_decomposed_name_matches_a_precomposed_alias() {
    // "Luci\u{301}a" is "Lucía" with a combining acute: the same name, which
    // the alias FTS indexes as "lucia" either way.
    let (entities, found) = candidates_for(&["Lucía"], "Luci\u{301}a called.");
    assert_eq!(found, BTreeSet::from([entities[0]]));
}

#[test]
fn a_precomposed_name_matches_a_decomposed_alias() {
    let (entities, found) = candidates_for(&["Luci\u{301}a"], "Lucía called.");
    assert_eq!(found, BTreeSet::from([entities[0]]));
}

#[test]
fn an_accented_greek_name_matches_itself() {
    // The FTS keeps Greek accents, so an identical name must still match.
    let (entities, found) = candidates_for(&["Νίκος"], "Ο Νίκος ήρθε.");
    assert_eq!(found, BTreeSet::from([entities[0]]));
}

#[test]
fn a_devanagari_name_matches_itself() {
    // A vowel sign is a combining mark, and part of the name.
    let (entities, found) = candidates_for(&["किरण"], "किरण आया।");
    assert_eq!(found, BTreeSet::from([entities[0]]));
}

// Composed aliases (the TIM-107 re-review of `d4825ab`). Passages are
// searched in NFC, so every alias has to be stored in NFC too: on every
// write, and for the aliases a store already holds.

/// Greek "Νίκος" decomposed: iota, then a combining acute.
const NIKOS_DECOMPOSED: &str = "Νι\u{301}κος";
const NIKOS: &str = "Νίκος";

/// A Discord speaker called `name` says hello, which creates their entity
/// and aliases through ingest. The chunk is taken off the queue.
fn speaker_named(h: &Harness, id: &str, name: &str) -> Uuid {
    let ingested = ingest(
        h,
        &Turn {
            author: Some(TurnAuthor {
                id: id.into(),
                name: Some(name.into()),
                is_bot: false,
            }),
            platform: Some("discord".into()),
            ..turn("thread-9", "2026-10-01T05:00:00Z", "Hello.", "Hi.")
        },
    );
    h.execute(
        "DELETE FROM extraction_queue
         WHERE chunk_id = (SELECT c.id FROM chunks c JOIN sources s ON s.id = c.source_id
                           WHERE s.uuid = ?1)",
        [ingested.source.to_string()],
    );
    ingested.speaker.unwrap().entity
}

/// The candidates found when the owner says `message`.
fn found_in(h: &Harness, message: &str) -> BTreeSet<Uuid> {
    ingest(h, &turn("s1", T1, message, "Lovely."));
    found(h, &input(h, "main", &[]))
}

#[test]
fn a_decomposed_speaker_name_is_found_by_its_composed_spelling() {
    let h = Harness::new();
    let nikos = speaker_named(&h, "7777", NIKOS_DECOMPOSED);
    assert_eq!(found_in(&h, "Ο Νίκος ήρθε."), BTreeSet::from([nikos]));
}

#[test]
fn a_decomposed_speaker_name_is_found_by_the_same_spelling() {
    let h = Harness::new();
    let nikos = speaker_named(&h, "7777", NIKOS_DECOMPOSED);
    assert_eq!(
        found_in(&h, &format!("Ο {NIKOS_DECOMPOSED} ήρθε.")),
        BTreeSet::from([nikos])
    );
}

#[test]
fn a_composed_speaker_name_is_found_by_its_decomposed_spelling() {
    let h = Harness::new();
    let nikos = speaker_named(&h, "7777", NIKOS);
    assert_eq!(
        found_in(&h, &format!("Ο {NIKOS_DECOMPOSED} ήρθε.")),
        BTreeSet::from([nikos])
    );
}

#[test]
fn speaker_names_and_aliases_are_stored_composed() {
    let h = Harness::new();
    let nikos = speaker_named(&h, "7777", NIKOS_DECOMPOSED);
    assert_eq!(h.entity_name(nikos), NIKOS);
    let aliases = h.aliases(nikos);
    assert!(aliases.contains(&NIKOS.to_string()), "{aliases:?}");
    assert!(
        !aliases.contains(&NIKOS_DECOMPOSED.to_string()),
        "{aliases:?}"
    );
}

#[test]
fn bank_config_stores_names_and_aliases_composed() {
    let h = Harness::new();
    let zoe_decomposed = "Ζωη\u{301}";
    h.service
        .ensure_bank_with_models(
            "greek",
            &asphodel_core::store::bank::BankIdentity {
                owner_name: Some(NIKOS_DECOMPOSED.into()),
                owner_platform_ids: Vec::new(),
                assistant_name: Some(zoe_decomposed.into()),
                timezone: Some(TZ.into()),
            },
        )
        .unwrap();
    for (which, composed, decomposed) in [
        ("user", NIKOS, NIKOS_DECOMPOSED),
        ("assistant", "Ζωή", zoe_decomposed),
    ] {
        let entity = h.seeded("greek", which);
        assert_eq!(h.entity_name(entity), composed, "{which}");
        let aliases = h.aliases(entity);
        assert!(
            aliases.contains(&composed.to_string()),
            "{which}: {aliases:?}"
        );
        assert!(
            !aliases.contains(&decomposed.to_string()),
            "{which}: {aliases:?}"
        );
    }
}

#[test]
fn a_proposed_entity_is_stored_composed_and_found_again() {
    let h = Harness::new();
    let created = golden(
        &h,
        &format!("{NIKOS_DECOMPOSED} called."),
        "Who's that?",
        vec![claim("Nikos called Tim.", "event", NIKOS_DECOMPOSED).with(
            "entities",
            json!([new_entity(NIKOS_DECOMPOSED, "person", NIKOS_DECOMPOSED)]),
        )],
    );
    let nikos = h.entities_named("main", NIKOS);
    assert_eq!(nikos.len(), 1);
    assert_eq!(h.aliases(nikos[0]), vec![NIKOS.to_string()]);
    assert_eq!(
        h.links(created[0]),
        BTreeSet::from([(nikos[0], Some(NIKOS.into()))])
    );
    assert_eq!(found_in(&h, "Ο Νίκος ήρθε."), BTreeSet::from([nikos[0]]));
}

/// Puts the store back to schema version 2, the last that stored aliases as
/// written, and leaves one `migrations` row 0 to 2. Everything written
/// stays, so reopening runs only the migrations after version 2.
fn downgrade_to_v2_and_reopen(h: Harness) -> Harness {
    h.service
        .store()
        .unwrap()
        .connection()
        .execute_batch(
            "DELETE FROM migrations;
             INSERT INTO migrations (from_version, to_version, binary_version, started_at,
                                     completed_at)
               VALUES (0, 2, 'v2', 0, 0);
             PRAGMA user_version = 2;",
        )
        .unwrap();
    h.restart()
}

/// Logs an `alias_added` edit for the alias row `alias` of `entity`, as
/// `add_alias` does.
fn log_alias_added(h: &Harness, entity: Uuid, alias: &str) -> String {
    let edit = next_uuid().to_string();
    h.execute(
        "INSERT INTO edits (uuid, bank_id, kind, entity_id, details, at)
         SELECT ?1, e.bank_id, 'alias_added', e.id, json_object('alias_id', a.id), ?4
         FROM entities e JOIN entity_aliases a ON a.entity_id = e.id
         WHERE e.uuid = ?2 AND a.alias = ?3",
        (edit.clone(), entity.to_string(), alias, micros(h.now())),
    );
    edit
}

#[test]
fn an_upgrade_composes_stored_aliases_and_merges_equivalent_ones() {
    let h = Harness::new();
    // As a version 2 store could hold them: one entity with both spellings
    // of its name as aliases, each with its `alias_added` edit, and one
    // known only by a decomposed alias.
    let nikos = h.insert_entity("main", NIKOS, "person", &[NIKOS, NIKOS_DECOMPOSED]);
    let edits = [
        log_alias_added(&h, nikos, NIKOS),
        log_alias_added(&h, nikos, NIKOS_DECOMPOSED),
    ];
    let zoe = h.insert_entity("main", "Ζωη\u{301}", "person", &["Ζωη\u{301}"]);
    let h = downgrade_to_v2_and_reopen(h);

    // One composed alias each; the canonical duplicate is merged away.
    assert_eq!(h.aliases(nikos), vec![NIKOS.to_string()]);
    assert_eq!(h.aliases(zoe), vec!["Ζωή".to_string()]);
    assert_eq!(h.entity_name(zoe), "Ζωή");
    // Both edits still name an alias of the entity: the one that survived.
    let survivor: i64 = h.one(
        "SELECT a.id FROM entity_aliases a JOIN entities e ON e.id = a.entity_id
         WHERE e.uuid = ?1",
        [nikos.to_string()],
    );
    for edit in &edits {
        let alias_id: i64 = h.one(
            "SELECT json_extract(details, '$.alias_id') FROM edits WHERE uuid = ?1",
            [edit],
        );
        assert_eq!(alias_id, survivor, "{edit}");
    }
    // Both are found again, in either spelling.
    assert_eq!(
        found_in(&h, "Ο Νίκος και η Ζωή ήρθαν."),
        BTreeSet::from([nikos, zoe])
    );
}

#[test]
fn an_upgraded_decomposed_alias_is_found_by_the_same_spelling() {
    let h = Harness::new();
    let zoe = h.insert_entity("main", "Ζωη\u{301}", "person", &["Ζωη\u{301}"]);
    let h = downgrade_to_v2_and_reopen(h);
    assert_eq!(found_in(&h, "Η Ζωη\u{301} ήρθε."), BTreeSet::from([zoe]));
}

#[test]
fn accents_outside_latin_and_letters_like_o_slash_stay_distinct() {
    // The FTS folds Latin diacritics only. A Greek accent and "ø" are part of
    // the letter, so these are different names, as the FTS has them.
    let (_, found) = candidates_for(&["Νίκος", "Søren"], "Νικος and Soren came.");
    assert!(found.is_empty(), "{found:?}");
}

#[test]
fn times_in_a_daylight_saving_gap_or_fold_resolve_compatibly() {
    let h = Harness::new();
    // Auckland skips 02:00 to 03:00 on 27 September 2026 and repeats 02:00
    // to 03:00 on 5 April 2026. A time in the gap moves forward by the gap,
    // a time in the fold takes the earlier offset, and neither lowers window
    // confidence.
    let event = |content: &str, at: &str, precision: &str| {
        claim(content, "event", "Plans").with("valid_from", time(at, precision))
    };
    let memories = golden(
        &h,
        "Plans.",
        "Ok.",
        vec![
            event("Tim had an early start.", "2026-09-27T02:30", "minute"),
            event("Tim had an early hour.", "2026-09-27T02:00", "hour"),
            event("Tim had a late night.", "2026-04-05T02:30", "minute"),
        ],
    );
    let rows: Vec<Row> = memories.iter().map(|m| h.row(*m)).collect();
    assert_eq!(
        rows.iter()
            .map(|r| r.valid_from.clone())
            .collect::<Vec<_>>(),
        vec![
            timed(at("2026-09-26T14:30:00Z"), "minute"),
            timed(at("2026-09-26T14:00:00Z"), "hour"),
            timed(at("2026-04-04T13:30:00Z"), "minute"),
        ]
    );
    assert!(rows.iter().all(|r| r.window_confidence == "high"));
}

#[test]
fn a_swept_earlier_turn_is_not_context() {
    let h = Harness::new();
    let swept = ingest(&h, &turn("s1", "2026-10-01T06:00:00Z", "Old news.", "Ok."));
    ingest(&h, &turn("s1", "2026-10-01T06:10:00Z", "Kept news.", "Ok."));
    h.execute(
        "UPDATE sources SET text = NULL, reply = NULL, tombstoned_at = ?2,
                            tombstone_reason = 'swept'
         WHERE uuid = ?1",
        (swept.source.to_string(), micros(h.now())),
    );
    let current = ingest(&h, &turn("s1", T1, "Now.", "Yes."));
    h.focus(h.chunk_of(current.source, 0));
    assert_eq!(
        input(&h, "main", &[]).context,
        vec![turn_text("Kept news.", "Ok.")]
    );
}

#[test]
fn without_models_a_chunk_stays_queued_and_uncounted() {
    let h = Harness::without_models();
    let ingested = ingest(&h, &turn("s1", T1, "I like tea.", "Noted."));
    let chunk = h.chunk_of(ingested.source, 0);
    let llm = FakeLlm::scripted(
        MODEL,
        vec![reply(
            vec![claim("Tim likes tea.", "fact", "I like tea")],
            &[],
        )],
    );
    let error = run(&h, "main", &llm, &[]).unwrap_err();
    assert!(matches!(error, ExtractError::NoModels), "{error:?}");
    assert_eq!(error.failure(), None);
    assert!(llm.requests().is_empty());
    assert_eq!(h.chunk_column::<i64>(chunk, "error_count"), 0);
    assert_eq!(h.service.queue_depth("main").unwrap(), 1);
    assert_eq!(lease(&h, "main").chunk, chunk);
    assert_eq!(h.memories_in("main"), 0);
}

/// An embedder that claims bge-small's width but returns short vectors.
struct ShortEmbedder;

impl Embedder for ShortEmbedder {
    fn model_id(&self) -> &str {
        FakeEmbedder::MODEL_ID
    }

    fn dimensions(&self) -> usize {
        FakeEmbedder.dimensions()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        Ok(texts.iter().map(|_| vec![1.0; 8]).collect())
    }
}

fn vectors(h: &Harness) -> i64 {
    h.count("SELECT COUNT(*) FROM memory_vectors")
}

#[test]
fn a_vector_of_the_wrong_width_is_an_embedding_failure() {
    let h = Harness::with_models(Models {
        embedder: Arc::new(ShortEmbedder),
        reranker: Arc::new(FakeReranker),
    });
    let tea = h.insert_memory("main", "Tim likes tea.", "minor");
    let ingested = ingest(&h, &turn("s1", T1, "I'm moving to Lisbon.", "Exciting!"));
    let chunk = h.chunk_of(ingested.source, 0);
    let input = input(&h, "main", &[tea]);
    let before = (untouched(&h), vectors(&h));

    let llm = FakeLlm::scripted(MODEL, vec![busy_reply(&input, tea)]);
    let error = run(&h, "main", &llm, &[tea]).unwrap_err();
    assert!(
        matches!(
            error,
            ExtractError::Embedding {
                failure: Failure::Retry { error_count: 1 },
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!((untouched(&h), vectors(&h)), before);
    assert_eq!(
        h.chunk_column::<Option<String>>(chunk, "last_error_kind"),
        Some("embedding".into())
    );
    assert_eq!(h.service.queue_depth("main").unwrap(), 1);
}

/// Makes every access insert fail, so a commit fails after its memories,
/// vectors, entities and links are written.
fn break_accesses(h: &Harness) {
    h.execute(
        "CREATE TRIGGER break_accesses BEFORE INSERT ON accesses
         BEGIN SELECT RAISE(ABORT, 'scripted commit failure'); END",
        [],
    );
}

#[test]
fn a_failed_commit_rolls_back_everything_and_is_counted() {
    let h = Harness::new();
    let tea = h.insert_memory("main", "Tim likes tea.", "minor");
    let ingested = ingest(&h, &turn("s1", T1, "I'm moving to Lisbon.", "Exciting!"));
    let chunk = h.chunk_of(ingested.source, 0);
    let input = input(&h, "main", &[tea]);
    break_accesses(&h);
    let before = (untouched(&h), vectors(&h));

    let llm = FakeLlm::scripted(MODEL, vec![busy_reply(&input, tea)]);
    run(&h, "main", &llm, &[tea]).unwrap_err();
    assert_eq!((untouched(&h), vectors(&h)), before);
    assert!(h.entities_named("main", "Lisbon").is_empty());
    assert_eq!(h.chunk_column::<Option<i64>>(chunk, "extracted_at"), None);
    assert_eq!(h.chunk_column::<i64>(chunk, "error_count"), 1);
    assert_eq!(
        h.chunk_column::<Option<String>>(chunk, "last_error_kind"),
        Some("commit".into())
    );
    assert_eq!(h.service.queue_depth("main").unwrap(), 1);

    // Once the fault is gone, the retry commits everything once.
    h.execute("DROP TRIGGER break_accesses", []);
    let extracted = extract_with(&h, busy_reply(&input, tea), &[tea]);
    assert_eq!(extracted.memories.len(), 1);
    assert_eq!(vectors(&h), before.1 + 1);
    assert_eq!(h.entities_named("main", "Lisbon").len(), 1);
    assert_eq!(h.accesses(tea).len(), 2);
}

#[test]
fn a_failed_commit_reports_the_queues_count() {
    let h = Harness::new();
    ingest(&h, &turn("s1", T1, "I like tea.", "Noted."));
    break_accesses(&h);
    let llm = FakeLlm::scripted(
        MODEL,
        vec![reply(
            vec![claim("Tim likes tea.", "fact", "I like tea")],
            &[],
        )],
    );
    let error = run(&h, "main", &llm, &[]).unwrap_err();
    // The queue counted the attempt, so the error says so, as every other
    // counted failure does.
    assert_eq!(error.failure(), Some(Failure::Retry { error_count: 1 }));
}

#[test]
fn the_prompts_state_the_extraction_rules() {
    let h = Harness::new();
    ingest(&h, &turn("s1", T1, "Hello.", "Hi."));
    let system = call1_request(&input(&h, "main", &[])).system;
    // Scripted replies can't show these instructions were given, so check
    // the prompt carries them (TIM-92, "Inputs", "Significance", "Time" and
    // "Language").
    for rule in [
        // Only what the assistant did, and its tasks only when asked and dated.
        "From the assistant's reply, only what the assistant says it has done or will do.",
        "An assistant task is extracted only when the speaker asked for it and it has a due date or an until-event beyond this turn.",
        "Never extract the assistant's suggestions, general knowledge, or findings from tools.",
        // Answers resolved from the context, quoted from the text.
        "A short answer to a question the assistant asked in the context is the speaker's claim, written out in full.",
        "Quote only from the text, never from the context",
        // No claim from a request to forget.
        "Nothing from a request to forget something, and no task to forget it.",
        // The expected distribution.
        "Most claims are trivial or minor. Major is rare and critical is a few per hundred claims.",
        // Language.
        "Write the claim in the language of the passage it quotes and never translate.",
        // The hard rule on ends, and an unknown reference date.
        "Never set `valid_until` unless the text states an end.",
        "If the reference date is unknown, don't resolve relative times",
    ] {
        assert!(system.contains(rule), "{rule}");
    }

    ingest_doc(
        &h,
        &Document {
            reference_date_exact: false,
            ..document("diary", "Met Ana yesterday.", date(2026, 9, 28))
        },
    );
    extract(&h, reply(vec![], &[]));
    let user = call1_request(&input(&h, "main", &[])).user;
    assert!(user.contains("Reference date: unknown"), "{user}");
    assert!(user.contains("Don't resolve relative times."), "{user}");
}
