//! What the dashboard needs from the service: browsing that writes nothing,
//! fade dates on every listed memory, the owner's retraction, and removing a
//! document with every version of it.
//!
//! The API under test is the `Service` methods `list_memories`,
//! `list_sources`, `show_source`, `retract` and `remove_document`, next to
//! the existing `show_memory`. The routes over them are checked in
//! `crates/asphodel/tests/serve_http.rs`.
//!
//! - **Browsing reads only.** Listing and showing never write a row: no
//!   access, so nothing is strengthened by being looked at, and no recall
//!   row. The dashboard never browses through recall.
//! - **Fade on a list** is the same projection `memory show` gives, and
//!   sorting by it puts the soonest first and the memories that never fade
//!   last.
//! - **Retract is a denial by the owner.** The memory is invalidated with
//!   no successor, so everything that reads live memories drops it, and
//!   whatever it ended is open again. It stays visible as retracted until
//!   the sweep purges it.
//! - **Removing a document** takes every version of its id: the memories
//!   resting on its chunks are forgotten, its waiting chunks dequeued, and
//!   its sources tombstoned with their keys kept. A chunk already in flight
//!   leaves nothing behind once the erase has run.
//!
//! Every service here runs on a `SimulatedClock` stopped at 20:00 on
//! Thursday 1 October 2026 in Auckland (UTC+13) unless a test moves it, with
//! `clock.quiet_rate = 1.0` so bank time is world time.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asphodel_core::erase::{DocumentRemoved, RemoveDocumentError};
use asphodel_core::extraction::Committed;
use asphodel_core::ingest::{Document, Ingested, Outcome, Turn};
use asphodel_core::inspect::{
    Gone, InspectError, MemoryQuery, MemorySort, MemoryStatus, MemorySummary, SourceQuery,
};
use asphodel_core::models::{Embedder, FakeEmbedder, FakeLlm, FakeReranker, Models};
use asphodel_core::retract::{RetractError, Retracted};
use asphodel_core::retrieval::{Recall, RecallRequest};
use asphodel_core::store::bank::{BankIdentity, PROFILE_NAME};
use asphodel_core::store::{OpenOptions, Store, VectorIndex, micros};
use asphodel_core::{Service, SimulatedClock, Tuning};
use jiff::civil::date;
use jiff::{SignedDuration, Timestamp};
use rusqlite::OptionalExtension;
use rusqlite::types::FromSql;
use serde_json::{Value, json};
use uuid::Uuid;

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";
const BANK: &str = "main";
const MODEL: &str = "fake-llm";

/// The next 04:00 in Auckland after [`START`], when the nightly sweep runs.
const SWEEP: &str = "2026-10-01T15:00:00Z";

/// When fixture memories were said, unless a test says otherwise.
const EARLIER: &str = "2026-09-01T00:00:00Z";

/// The fixture turn every inserted memory rests on.
const FIXTURE: &str = "Fixtures.";

const BERLIN: &str = "Tim lives in Berlin.";
const MOVED_OUT: &str = "Tim moved out of Berlin.";
const MAYA: &str = "Tim's daughter is called Maya.";
const MIA: &str = "Tim's daughter is called Mia.";
const TEA: &str = "Tim likes green tea.";
const PASSPORT: &str = "Tim needs to renew his passport.";
const BIKE: &str = "Tim needs to fix the bike.";

/// Three versions of one document, each a single chunk of its own.
const NOTES: &str = "notes.md";
const NOTES_V1: &str = "# Notes\n\nTim likes green tea.\n";
const NOTES_V2: &str = "# Notes\n\nTim lives in Berlin.\n";
const NOTES_V3: &str = "# Notes\n\nTim needs to fix the bike.\n";

/// A document that stays.
const FAMILY: &str = "family.md";
const FAMILY_V1: &str = "# Family\n\nTim's daughter is called Maya.\n";

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
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
            "asphodel-dashboard-{}-{}",
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
    Uuid::from_u128((0xdb_u128 << 120) | u128::from(NEXT.fetch_add(1, Ordering::Relaxed)))
}

/// A memory to insert. `Default` is a notable fact said at [`EARLIER`] with
/// its `created` access then, resting on the fixture passage.
#[derive(Clone)]
struct Memory {
    content: &'static str,
    kind: &'static str,
    significance: &'static str,
    observed_at: Timestamp,
}

impl Default for Memory {
    fn default() -> Self {
        Self {
            content: "",
            kind: "fact",
            significance: "notable",
            observed_at: at(EARLIER),
        }
    }
}

fn fact(content: &'static str) -> Memory {
    Memory {
        content,
        ..Memory::default()
    }
}

/// An undated open task.
fn task(content: &'static str) -> Memory {
    Memory {
        content,
        kind: "task",
        ..Memory::default()
    }
}

struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    /// The fixture turn's chunk.
    chunk: i64,
    dir: TestDir,
}

impl Harness {
    fn new() -> Self {
        let dir = TestDir::new();
        let clock = Arc::new(SimulatedClock::new(at(START)));
        let mut harness = Self::open(dir, clock);
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
        let source = harness
            .service
            .ingest_turn(BANK, &turn("fixtures", at(EARLIER), FIXTURE))
            .unwrap()
            .source;
        harness.chunk = harness.one("SELECT id FROM chunks", []);
        harness.extracted_with_nothing(source);
        harness
    }

    /// Opens the store in `dir` with the tuning the fakes need and the
    /// purge state the store gives it, as `serve` does.
    fn open(dir: TestDir, clock: Arc<SimulatedClock>) -> Self {
        let tuning = Tuning::from_toml(&format!(
            "[clock]\nquiet_rate = 1.0\n\
             [injection.reranker_floors]\n\"{}\" = 1.0\n\
             [ranking.relevance_scales]\n\"{0}\" = 1.0\n\
             [reconcile.embedding_floors]\n\"{}\" = 0.5\n",
            FakeReranker::MODEL_ID,
            FakeEmbedder::MODEL_ID,
        ))
        .unwrap();
        let store = Store::open(&dir.0, OpenOptions::default(), clock.clone()).unwrap();
        let pause = store
            .check_fingerprint(&tuning.deletion_fingerprint())
            .unwrap();
        let service = Service::with_models(clock.clone(), store, tuning, Models::fake()).unwrap();
        Self {
            service: service.with_purge_pause(pause),
            clock,
            chunk: 0,
            dir,
        }
    }

    /// The daemon restarting: the store stays.
    fn restart(self) -> Self {
        let Self {
            service,
            clock,
            chunk,
            dir,
        } = self;
        drop(service);
        Self {
            chunk,
            ..Self::open(dir, clock)
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

    fn execute<P: rusqlite::Params>(&self, sql: &str, params: P) -> usize {
        self.service
            .store()
            .unwrap()
            .connection()
            .execute(sql, params)
            .unwrap()
    }

    /// Every row the store's connection has inserted, updated or deleted
    /// since it opened. The service has the one connection, so a call
    /// that leaves this alone wrote nothing.
    fn writes(&self) -> i64 {
        self.one("SELECT total_changes()", [])
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
                                   source_start, source_end, observed_at,
                                   window_confidence, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, ?7, ?8, 'high', ?9, ?9)",
            rusqlite::params![
                uuid.to_string(),
                bank_id,
                memory.content,
                memory.kind,
                memory.significance,
                self.chunk,
                FIXTURE.len() as i64,
                micros(memory.observed_at),
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

    /// How many of `memories` still have a row.
    fn rows(&self, memories: &[Uuid]) -> usize {
        memories
            .iter()
            .filter(|memory| {
                self.service
                    .store()
                    .unwrap()
                    .connection()
                    .query_row(
                        "SELECT id FROM memories WHERE uuid = ?1",
                        [memory.to_string()],
                        |row| row.get::<_, i64>(0),
                    )
                    .optional()
                    .unwrap()
                    .is_some()
            })
            .count()
    }

    /// `source`'s chunk taken off the queue as extracted with no claims.
    fn extracted_with_nothing(&self, source: Uuid) {
        let chunk: i64 = self.one(
            "SELECT c.id FROM chunks c JOIN sources s ON s.id = c.source_id WHERE s.uuid = ?1",
            [source.to_string()],
        );
        self.execute("DELETE FROM extraction_queue WHERE chunk_id = ?1", [chunk]);
        self.execute(
            "UPDATE chunks SET extracted_at = ?2 WHERE id = ?1",
            (chunk, micros(self.now())),
        );
    }

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

    /// Ingests a version of a document and extracts its one chunk with call
    /// 1 finding `sentence`, quoted as written. Returns the source and the
    /// memory.
    fn doc_stating(&self, id: &str, text: &str, sentence: &str) -> (Uuid, Uuid) {
        let source = self.doc(id, text);
        assert_eq!(source.outcome, Outcome::Stored);
        let extracted = self
            .service
            .extract_next(BANK, &states(sentence))
            .unwrap()
            .expect("the document's chunk was queued");
        (source.source, extracted.memories[0])
    }

    fn recall(&self, request: RecallRequest) -> Recall {
        self.service.recall(BANK, &request).unwrap()
    }

    fn recalled(&self, text: &str) -> Vec<Uuid> {
        ids(&self.recall(query(text)))
    }

    fn list(&self, query: MemoryQuery) -> Vec<MemorySummary> {
        self.service.list_memories(BANK, &query).unwrap().memories
    }

    fn listed(&self, status: MemoryStatus) -> BTreeSet<Uuid> {
        self.list(MemoryQuery {
            status: Some(status),
            ..MemoryQuery::default()
        })
        .into_iter()
        .map(|memory| memory.id)
        .collect()
    }

    fn retract(&self, memory: Uuid) -> Result<Retracted, RetractError> {
        self.service.retract(BANK, &memory.to_string())
    }

    fn remove(&self, document: &str) -> DocumentRemoved {
        self.service.remove_document(BANK, document).unwrap()
    }

    /// Runs every erase waiting in the queue.
    fn erase(&self) {
        while self.service.erase_next(BANK).unwrap().is_some() {}
    }

    /// An entry in the profile citing `memory`, as an earlier refresh left
    /// it.
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
    }
}

/// The owner's turn on the CLI.
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

/// A fake LLM whose call 1 finds one notable fact, `sentence`, quoted as
/// written, and whose call 2, if it runs, labels nothing.
fn states(sentence: &str) -> FakeLlm {
    FakeLlm::scripted(
        MODEL,
        vec![
            json!({"claims": [notable(sentence)], "used_injected_ids": []}),
            json!({"claims": []}),
        ],
    )
}

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

fn query(text: &str) -> RecallRequest {
    RecallRequest {
        query: text.into(),
        ..RecallRequest::default()
    }
}

fn ids(recall: &Recall) -> Vec<Uuid> {
    recall.results.iter().map(|r| r.id).collect()
}

// Browsing

#[test]
fn browsing_documents_and_memories_writes_nothing() {
    // Looking at a memory must not strengthen it, and the dashboard's reads
    // must not show up as recalls.
    let h = Harness::new();
    let (source, tea) = h.doc_stating(NOTES, NOTES_V1, TEA);
    let before = h.writes();

    let memories = h.list(MemoryQuery::default());
    let listed = memories
        .iter()
        .find(|memory| memory.id == tea)
        .expect("the extracted memory is listed");
    assert_eq!(listed.sentence, TEA);
    assert_eq!(listed.status, MemoryStatus::Live);

    let page = h
        .service
        .list_sources(
            BANK,
            &SourceQuery {
                document_id: Some(NOTES.into()),
                ..SourceQuery::default()
            },
        )
        .unwrap();
    let sources: Vec<Uuid> = page.sources.iter().map(|source| source.id).collect();
    assert_eq!(sources, [source]);

    let shown = h.service.show_source(BANK, &source.to_string()).unwrap();
    assert_eq!(shown.text.as_deref(), Some(NOTES_V1));
    assert_eq!(shown.chunks.len(), 1);
    assert_eq!(shown.chunks[0].memories, [tea]);

    let memory = h.service.show_memory(BANK, &tea.to_string()).unwrap();
    assert_eq!(h.writes(), before, "browsing wrote to the store");
    let kinds: Vec<&str> = memory.accesses.iter().map(|a| a.kind.as_str()).collect();
    assert_eq!(kinds, ["created"]);
}

#[test]
fn the_memory_list_filters_by_status() {
    let h = Harness::new();
    let berlin = h.insert(fact(BERLIN));
    let moved_out = h.insert(fact(MOVED_OUT));
    h.supersede(berlin, moved_out, true);
    let passport = h.insert(task(PASSPORT));
    h.end(passport, moved_out);
    let maya = h.insert(fact(MAYA));
    h.service.forget(BANK, &[maya.to_string()]).unwrap();

    assert_eq!(h.listed(MemoryStatus::Live), BTreeSet::from([moved_out]));
    assert_eq!(h.listed(MemoryStatus::Retracted), BTreeSet::from([berlin]));
    assert_eq!(h.listed(MemoryStatus::Ended), BTreeSet::from([passport]));
    let mut all: Vec<(Uuid, MemoryStatus)> = h
        .list(MemoryQuery::default())
        .into_iter()
        .map(|memory| (memory.id, memory.status))
        .collect();
    all.sort_by_key(|(id, _)| *id);
    assert_eq!(
        all,
        [
            (berlin, MemoryStatus::Retracted),
            (moved_out, MemoryStatus::Live),
            (passport, MemoryStatus::Ended),
            (maya, MemoryStatus::Forgetting),
        ]
    );
}

#[test]
fn each_listed_memory_fades_when_memory_show_says_and_the_soonest_sorts_first() {
    let h = Harness::new();
    let notable = h.insert(fact(BERLIN));
    let trivial = h.insert(Memory {
        significance: "trivial",
        observed_at: at(START) - days(1),
        ..fact(TEA)
    });
    let kept = h.insert(fact(MAYA));
    h.service.keep(BANK, &[kept.to_string()]).unwrap();

    let fade = |memory: Uuid| {
        h.service
            .show_memory(BANK, &memory.to_string())
            .unwrap()
            .projection
            .fade
    };
    let (soon, later) = (fade(trivial).unwrap(), fade(notable).unwrap());
    assert!(soon.bank_days < later.bank_days, "{soon:?} {later:?}");
    assert_eq!(fade(kept), None, "a kept memory never fades");

    let listed = h.list(MemoryQuery {
        sort: MemorySort::Fade,
        ..MemoryQuery::default()
    });
    let order: Vec<Uuid> = listed.iter().map(|memory| memory.id).collect();
    assert_eq!(order, [trivial, notable, kept]);
    for memory in &listed {
        assert_eq!(memory.fade, fade(memory.id), "{}", memory.id);
    }
}

// Retract

#[test]
fn retracting_takes_a_memory_out_of_everything_live_and_reopens_what_it_ended() {
    let h = Harness::new();
    let berlin = h.insert(fact(BERLIN));
    let moved_out = h.insert(fact(MOVED_OUT));
    h.end(berlin, moved_out);
    let passport = h.insert(task(PASSPORT));
    h.cite_in_profile("Tim no longer lives in Berlin.", moved_out);
    let recalled = h.recall(RecallRequest {
        session_id: Some("chat".into()),
        ..query(MOVED_OUT)
    });
    assert!(ids(&recalled).contains(&moved_out));
    assert!(
        h.service
            .in_context(BANK, "chat")
            .unwrap()
            .contains(&moved_out)
    );
    let block = h.service.system_prompt(BANK, None).unwrap();
    assert!(block.cited.contains(&moved_out));
    let input = h.service.refresh_input(BANK, PROFILE_NAME).unwrap();
    assert!(input.memories.iter().any(|m| m.memory == moved_out));
    assert!(h.service.agenda(BANK).unwrap().listed().contains(&passport));

    let retracted = h.retract(moved_out).unwrap();
    assert_eq!(retracted.memory, moved_out);
    assert_eq!(retracted.retracted_at, h.now());
    assert_eq!(
        retracted.reopened,
        [berlin],
        "a denial: the ending never held"
    );

    assert!(!h.recalled(MOVED_OUT).contains(&moved_out));
    assert!(
        !h.service
            .in_context(BANK, "chat")
            .unwrap()
            .contains(&moved_out)
    );
    let block = h.service.system_prompt(BANK, None).unwrap();
    assert!(!block.cited.contains(&moved_out));
    assert!(!block.text.contains("Tim no longer lives in Berlin."));
    let input = h.service.refresh_input(BANK, PROFILE_NAME).unwrap();
    assert!(input.memories.iter().all(|m| m.memory != moved_out));

    let reopened = h.service.show_memory(BANK, &berlin.to_string()).unwrap();
    assert_eq!(reopened.chain.ended_by, None);
    assert_eq!(reopened.window.valid_until, None);

    let shown = h.service.show_memory(BANK, &moved_out.to_string()).unwrap();
    assert_eq!(shown.retracted_at, Some(h.now()));
    let edit = shown
        .edits
        .iter()
        .find(|edit| edit.kind == "memory_retracted")
        .expect("the retraction is logged");
    assert_eq!(edit.details["by"], "owner");
    assert!(!edit.details.to_string().contains("Berlin"));

    h.retract(passport).unwrap();
    assert!(!h.service.agenda(BANK).unwrap().listed().contains(&passport));
}

#[test]
fn retract_refuses_a_superseded_memory_naming_its_head_and_a_repeat() {
    let h = Harness::new();
    let maya = h.insert(fact(MAYA));
    let mia = h.insert(fact(MIA));
    h.supersede(maya, mia, false);
    let forgotten = h.insert(fact(TEA));
    h.service.forget(BANK, &[forgotten.to_string()]).unwrap();
    let before = h.writes();

    assert!(matches!(
        h.retract(maya),
        Err(RetractError::Superseded { head }) if head == mia
    ));
    assert!(matches!(
        h.retract(forgotten),
        Err(RetractError::UnknownMemory)
    ));
    assert!(matches!(
        h.service.retract(BANK, "not-a-memory"),
        Err(RetractError::UnknownMemory)
    ));
    assert!(matches!(
        h.service.retract("nobody", &mia.to_string()),
        Err(RetractError::UnknownBank)
    ));
    assert_eq!(h.writes(), before, "a refused retraction writes nothing");

    h.retract(mia).unwrap();
    assert!(matches!(
        h.retract(mia),
        Err(RetractError::AlreadyRetracted)
    ));
}

#[test]
fn a_retracted_memory_still_shows_until_the_sweep_purges_it() {
    let h = Harness::new();
    let tea = h.insert(Memory {
        significance: "trivial",
        ..fact(TEA)
    });
    h.retract(tea).unwrap();
    assert_eq!(h.listed(MemoryStatus::Retracted), BTreeSet::from([tea]));
    let purge = h
        .service
        .show_memory(BANK, &tea.to_string())
        .unwrap()
        .projection
        .purge
        .expect("a trivial memory said once is purged once it fades");

    h.set(at(SWEEP));
    h.service.run_sweeps().unwrap();
    h.erase();
    assert_eq!(h.rows(&[tea]), 1, "retracting doesn't purge it early");

    h.set(purge.earliest_at + days(2));
    h.service.run_sweeps().unwrap();
    h.erase();
    assert_eq!(h.rows(&[tea]), 0);
    assert!(matches!(
        h.service.show_memory(BANK, &tea.to_string()),
        Err(InspectError::UnknownMemory)
    ));
}

// Removing a document

#[test]
fn removing_a_document_forgets_what_rests_on_every_version_and_keeps_the_keys() {
    let h = Harness::new();
    let (v1, tea) = h.doc_stating(NOTES, NOTES_V1, TEA);
    let (v2, berlin) = h.doc_stating(NOTES, NOTES_V2, BERLIN);
    let (_, maya) = h.doc_stating(FAMILY, FAMILY_V1, MAYA);
    let v3 = h.doc(NOTES, NOTES_V3);
    assert_eq!(v3.chunks_queued, 1);

    let removed = h.remove(NOTES);
    assert_eq!(
        removed.sources.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([v1, v2, v3.source])
    );
    assert_eq!(
        removed.forgotten.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([tea, berlin])
    );
    assert_eq!(removed.dequeued, 1, "the waiting version is dequeued");

    // Hidden at once, the text gone at once.
    assert!(!h.recalled(TEA).contains(&tea));
    assert!(!h.recalled(BERLIN).contains(&berlin));
    for source in [v1, v2, v3.source] {
        let shown = h.service.show_source(BANK, &source.to_string()).unwrap();
        assert_eq!(shown.text, None, "{source}");
        assert_eq!(shown.gone, Some(Gone::Removed), "{source}");
    }
    assert!(
        h.service
            .extract_next(BANK, &states(BIKE))
            .unwrap()
            .is_none(),
        "nothing of the document is left to extract"
    );

    h.erase();
    assert_eq!(h.rows(&[tea, berlin]), 0);
    assert!(h.recalled(MAYA).contains(&maya), "other documents stay");

    // The keys stay, so sending any version again changes nothing.
    for text in [NOTES_V1, NOTES_V2, NOTES_V3] {
        let again = h.doc(NOTES, text);
        assert_eq!(again.outcome, Outcome::Duplicate);
        assert_eq!(again.chunks_queued, 0);
    }
    assert!(matches!(
        h.service.remove_document(BANK, "never-ingested.md"),
        Err(RemoveDocumentError::UnknownDocument)
    ));
}

#[test]
fn a_chunk_in_flight_when_its_document_is_removed_leaves_no_memory_behind() {
    // Call 1 and call 2 have run; the commit lands after the removal. The
    // memory it writes rests on a removed document, so it must not survive
    // the erase, and the text it quoted must not come back.
    let h = Harness::new();
    let v1 = h.doc(NOTES, NOTES_V1);
    let lease = h.service.claim_chunk(BANK).unwrap().expect("queued");
    assert_eq!(lease.source, v1.source);
    let prepared = h
        .service
        .prepare_extraction(lease, &states(TEA), &[], &[])
        .unwrap();

    let removed = h.remove(NOTES);
    assert_eq!(removed.sources, [v1.source]);
    assert_eq!(removed.dequeued, 0, "a chunk in flight isn't dequeued");

    let committed = match h.service.try_commit_extraction(prepared) {
        Ok(Committed::Extracted(extracted)) => extracted.memories,
        Ok(Committed::Stale(_)) | Err(_) => Vec::new(),
    };
    for memory in &committed {
        assert!(!h.recalled(TEA).contains(memory), "hidden from the commit");
    }
    h.erase();

    assert_eq!(h.rows(&committed), 0);
    assert!(h.list(MemoryQuery::default()).is_empty());
    let shown = h.service.show_source(BANK, &v1.source.to_string()).unwrap();
    assert_eq!(shown.text, None);
    assert!(shown.chunks.iter().all(|chunk| chunk.memories.is_empty()));
    assert_eq!(h.service.queue_depth(BANK).unwrap(), 0);
}

#[test]
fn sources_tombstoned_before_the_upgrade_keep_their_keys() {
    // The new tombstone reason needs `sources` rebuilt. Rows, keys and the
    // older tombstones must come through it.
    let h = Harness::new();
    let (_, tea) = h.doc_stating(NOTES, NOTES_V1, TEA);
    let asked = h
        .service
        .ingest_turn(
            BANK,
            &Turn {
                forget_requested: true,
                ..turn(
                    "chat",
                    h.now() - minutes(1),
                    "Forget what I said about tea.",
                )
            },
        )
        .unwrap();
    assert_eq!(asked.outcome, Outcome::Tombstone);
    h.service
        .store()
        .unwrap()
        .connection()
        .execute_batch("DELETE FROM migrations WHERE to_version > 11; PRAGMA user_version = 11;")
        .unwrap();
    let h = h.restart();

    let again = h
        .service
        .ingest_turn(
            BANK,
            &Turn {
                forget_requested: true,
                ..turn(
                    "chat",
                    h.now() - minutes(1),
                    "Forget what I said about tea.",
                )
            },
        )
        .unwrap();
    assert_eq!(again.outcome, Outcome::Duplicate);
    assert_eq!(again.source, asked.source);
    assert_eq!(h.doc(NOTES, NOTES_V1).outcome, Outcome::Duplicate);
    assert!(h.recalled(TEA).contains(&tea));

    let removed = h.remove(NOTES);
    assert_eq!(removed.forgotten, [tea]);
}
