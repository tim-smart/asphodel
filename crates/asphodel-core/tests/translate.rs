//! Translating a memory into `[llm] language`, one memory at a time.
//!
//! The API under test is `Service::translate_memory`. A translation is a new
//! memory that supersedes the one named, the way a refinement does, so the
//! chain carries strength, accesses and provenance over. Every memory here
//! is synthetic, inserted directly.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use asphodel_core::Service;
use asphodel_core::clock::SimulatedClock;
use asphodel_core::config::Tuning;
use asphodel_core::extraction::EDIT_REFINED;
use asphodel_core::ingest::Turn;
use asphodel_core::inspect::MemoryView;
use asphodel_core::models::{
    Embedder, FakeEmbedder, FakeEmbedderV2, FakeLlm, FakeReranker, LlmClient, LlmError, LlmRequest,
    LlmResponse, ModelError, Models,
};
use asphodel_core::retrieval::RecallRequest;
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{DB_FILE, OpenOptions, Store, VectorIndex, micros};
use asphodel_core::translate::{TranslateError, Translation};
use jiff::Timestamp;
use rusqlite::types::FromSql;
use serde_json::{Value, json};
use uuid::Uuid;

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";

/// What the fixture turn says. The LLM is never shown it.
const PASSAGE: &str = "Passage text the translation must never see.";

const RUSSIAN: &str = "Сэм пьёт чай каждое утро.";
const ENGLISH: &str = "Sam drinks tea every morning.";

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
}

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-translate-{}-{}",
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

/// Floors for the fakes, and `[llm] language` when given.
fn tuning(language: Option<&str>) -> Tuning {
    let mut toml = format!(
        "[injection.reranker_floors]\n\"{}\" = 0.0\n\
         [reconcile.embedding_floors]\n\"{}\" = 0.5\n\"{}\" = 0.5\n",
        FakeReranker::MODEL_ID,
        FakeEmbedder::MODEL_ID,
        FakeEmbedderV2::MODEL_ID,
    );
    if let Some(language) = language {
        toml.push_str(&format!("[llm]\nlanguage = \"{language}\"\n"));
    }
    Tuning::from_toml(&toml).unwrap()
}

fn identity() -> BankIdentity {
    BankIdentity {
        owner_name: Some("Tim".into()),
        owner_platform_ids: vec![],
        assistant_name: Some("Hermes".into()),
        timezone: Some(TZ.into()),
    }
}

fn next_uuid() -> Uuid {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    Uuid::from_u128((0x7a_u128 << 120) | u128::from(NEXT.fetch_add(1, Ordering::Relaxed)))
}

/// The LLM's reply for a translation.
fn reply(sentence: &str) -> Value {
    json!({ "sentence": sentence })
}

/// A service on the fakes with one bank, `main`. Field order matters: the
/// service drops before the directory it lives in.
struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    dir: TestDir,
}

impl Harness {
    fn new(language: Option<&str>) -> Self {
        let dir = TestDir::new();
        let clock = Arc::new(SimulatedClock::new(at(START)));
        let store = Store::open(&dir.data(), OpenOptions::default(), clock.clone()).unwrap();
        let service =
            Service::with_models(clock.clone(), store, tuning(language), Models::fake()).unwrap();
        service
            .ensure_bank_with_models("main", &identity())
            .unwrap();
        Self {
            service,
            clock,
            dir,
        }
    }

    /// A daemon restart on `current`, carrying `previous` for the banks
    /// recorded under it.
    fn restart_with(
        self,
        language: Option<&str>,
        current: Arc<dyn Embedder>,
        previous: Arc<dyn Embedder>,
    ) -> Self {
        let Harness {
            service,
            clock,
            dir,
        } = self;
        drop(service);
        let store = Store::open(&dir.data(), OpenOptions::default(), clock.clone()).unwrap();
        let models = Models {
            embedder: current,
            reranker: Arc::new(FakeReranker),
        };
        let service = Service::with_models(clock.clone(), store, tuning(language), models)
            .unwrap()
            .with_previous_embedder(previous)
            .unwrap();
        Self {
            service,
            clock,
            dir,
        }
    }

    fn now(&self) -> Timestamp {
        use asphodel_core::Clock;
        self.clock.now()
    }

    fn one<T: FromSql, P: rusqlite::Params>(&self, sql: &str, params: P) -> T {
        self.service
            .store()
            .unwrap()
            .connection()
            .query_row(sql, params, |row| row.get(0))
            .unwrap()
    }

    fn execute<P: rusqlite::Params>(&self, sql: &str, params: P) {
        self.service
            .store()
            .unwrap()
            .connection()
            .execute(sql, params)
            .unwrap();
    }

    fn bank_id(&self) -> i64 {
        self.one("SELECT id FROM banks WHERE name = 'main'", [])
    }

    fn memories(&self) -> i64 {
        self.one("SELECT COUNT(*) FROM memories", [])
    }

    fn edits(&self, kind: &str) -> i64 {
        self.one("SELECT COUNT(*) FROM edits WHERE kind = ?1", [kind])
    }

    /// The chunk fixture memories rest on: a turn ingested and marked
    /// extracted, so nothing waits on the queue.
    fn fixture_chunk(&self) -> i64 {
        self.service
            .ingest_turn(
                "main",
                &Turn {
                    session_id: "fixtures".into(),
                    message_at: at("2026-09-01T00:00:00Z"),
                    timezone: Some(TZ.into()),
                    user_text: PASSAGE.into(),
                    assistant_text: "Noted.".into(),
                    author: None,
                    platform: Some("cli".into()),
                    recall_id: None,
                    forget_requested: false,
                },
            )
            .unwrap();
        let chunk: i64 = self.one(
            "SELECT c.id FROM chunks c JOIN sources s ON s.id = c.source_id
             WHERE s.bank_id = ?1 AND s.session_id = 'fixtures'",
            [self.bank_id()],
        );
        self.execute("DELETE FROM extraction_queue WHERE chunk_id = ?1", [chunk]);
        self.execute(
            "UPDATE chunks SET extracted_at = ?2 WHERE id = ?1",
            (chunk, micros(self.now())),
        );
        chunk
    }

    /// A state in `main`, inserted directly with no vector: a window, a
    /// volatility, a span into its chunk, an entity link with a surface
    /// form, and accesses on three separate occasions.
    fn memory(&self, chunk: i64, content: &str) -> Uuid {
        let uuid = next_uuid();
        let bank = self.bank_id();
        let created = micros(at("2026-09-01T00:00:00Z"));
        self.execute(
            "INSERT INTO memories (uuid, bank_id, content, kind, significance, chunk_id,
                                   source_start, source_end, observed_at,
                                   valid_from, valid_from_precision, window_confidence,
                                   volatility, created_at, updated_at)
             VALUES (?1, ?2, ?3, 'state', 'minor', ?4, 3, 27, ?5, ?5, 'day', 'low',
                     'months', ?5, ?5)",
            (uuid.to_string(), bank, content, chunk, created),
        );
        for (kind, when, turn) in [
            ("created", "2026-09-01T00:00:00Z", 1),
            ("used", "2026-09-10T00:00:00Z", 2),
            ("confirmed", "2026-09-20T00:00:00Z", 3),
        ] {
            self.execute(
                "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
                 SELECT bank_id, id, ?2, ?3, ?4 FROM memories WHERE uuid = ?1",
                (uuid.to_string(), kind, micros(at(when)), turn),
            );
        }
        let entity = next_uuid();
        self.execute(
            "INSERT INTO entities (uuid, bank_id, name, kind, created_at, updated_at)
             VALUES (?1, ?2, 'Sam', 'person', ?3, ?3)",
            (entity.to_string(), bank, created),
        );
        self.execute(
            "INSERT INTO memory_entities (memory_id, entity_id, surface_form)
             SELECT m.id, e.id, 'Сэм' FROM memories m, entities e
             WHERE m.uuid = ?1 AND e.uuid = ?2",
            (uuid.to_string(), entity.to_string()),
        );
        uuid
    }

    fn show(&self, memory: Uuid) -> MemoryView {
        self.service
            .show_memory("main", &memory.to_string())
            .unwrap()
    }

    fn translate(&self, memory: Uuid, llm: &dyn LlmClient) -> Result<Translation, TranslateError> {
        self.service
            .translate_memory("main", &memory.to_string(), llm)
    }

    fn recall(&self, query: &str) -> Vec<Uuid> {
        self.service
            .recall(
                "main",
                &RecallRequest {
                    query: query.into(),
                    ..RecallRequest::default()
                },
            )
            .unwrap()
            .results
            .iter()
            .map(|result| result.id)
            .collect()
    }
}

/// The new head of a translation.
fn translated(outcome: Translation) -> Uuid {
    match outcome {
        Translation::Translated { to, .. } => to,
        other => panic!("expected a translation, got {other:?}"),
    }
}

// Supersession

#[test]
fn a_translation_supersedes_the_memory_and_keeps_its_strength_accesses_and_provenance() {
    let h = Harness::new(Some("English"));
    let chunk = h.fixture_chunk();
    let original = h.memory(chunk, RUSSIAN);
    h.service
        .set_significance("main", &original.to_string(), Some("major"))
        .unwrap();
    let before = h.show(original);
    let memories = h.memories();
    let llm = FakeLlm::scripted("fake-llm", vec![reply(ENGLISH)]);

    let outcome = h.translate(original, &llm).unwrap();
    let head = match &outcome {
        Translation::Translated { from, to, language } => {
            assert_eq!(*from, original);
            assert_eq!(language, "English");
            *to
        }
        other => panic!("expected a translation, got {other:?}"),
    };
    assert_ne!(head, original);
    assert_eq!(h.memories(), memories + 1);

    // The LLM is asked for the target language and shown the sentence, never
    // the passage it came from.
    let requests = llm.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].system.contains("English"));
    assert!(requests[0].user.contains(RUSSIAN));
    assert!(!requests[0].system.contains(PASSAGE) && !requests[0].user.contains(PASSAGE));

    let after = h.show(head);
    assert_eq!(after.sentence, ENGLISH);

    // A refinement: one chain, the original superseded but not retracted.
    assert_eq!(after.chain.head, head);
    let old = after
        .chain
        .members
        .iter()
        .find(|member| member.id == original)
        .expect("the original is in the new head's chain");
    assert_eq!(old.superseded_by, Some(head));
    assert!(!old.retracted && !old.hidden);
    assert_eq!(h.show(original).retracted_at, None);
    assert_eq!(h.edits(EDIT_REFINED), 1);

    // Strength is unchanged at the same instant: the same significance, the
    // same accesses, and no new access for the translation itself.
    assert_eq!(after.strength, before.strength);
    assert_eq!(after.accesses.len(), before.accesses.len());
    for (inherited, own) in after.accesses.iter().zip(&before.accesses) {
        assert_eq!(
            (&inherited.kind, inherited.at, inherited.turn),
            (&own.kind, own.at, own.turn)
        );
        assert_eq!(inherited.inherited_from, Some(original));
    }

    // Provenance and everything but the sentence carry over.
    assert_eq!(after.source.chunk, before.source.chunk);
    assert_eq!(
        (after.source.start, after.source.end),
        (before.source.start, before.source.end)
    );
    assert_eq!(after.observed_at, before.observed_at);
    assert_eq!(after.kind, before.kind);
    assert_eq!(after.window, before.window);
    assert_eq!(after.significance, before.significance);
    assert_eq!(after.entities, before.entities);

    // Re-embedded and indexed under the new sentence.
    let vectors: i64 = h.one(
        "SELECT COUNT(*) FROM memory_vectors
         WHERE memory_id = (SELECT id FROM memories WHERE uuid = ?1)",
        [head.to_string()],
    );
    assert_eq!(vectors, 1);
    assert!(h.recall("Sam tea morning").contains(&head));
}

// Refusals

#[test]
fn translation_is_refused_without_a_configured_language() {
    let h = Harness::new(None);
    let chunk = h.fixture_chunk();
    let original = h.memory(chunk, RUSSIAN);
    let memories = h.memories();
    let llm = FakeLlm::scripted("fake-llm", vec![reply(ENGLISH)]);

    let refused = h.translate(original, &llm);

    assert!(
        matches!(refused, Err(TranslateError::LanguageUnset)),
        "{refused:?}"
    );
    assert!(llm.requests().is_empty());
    assert_eq!(h.memories(), memories);
    assert_eq!(h.edits(EDIT_REFINED), 0);
    assert_eq!(h.show(original).chain.head, original);
}

// Repeats and stale heads

#[test]
fn naming_the_translated_memory_again_is_refused_with_its_head() {
    let h = Harness::new(Some("English"));
    let chunk = h.fixture_chunk();
    let original = h.memory(chunk, RUSSIAN);
    let head = translated(
        h.translate(
            original,
            &FakeLlm::scripted("fake-llm", vec![reply(ENGLISH)]),
        )
        .unwrap(),
    );
    let memories = h.memories();

    // A retry of the same request, say after a timeout that hid a commit.
    let llm = FakeLlm::scripted("fake-llm", vec![reply(ENGLISH)]);
    let repeated = h.translate(original, &llm);

    match repeated {
        Err(TranslateError::Superseded { head: named }) => assert_eq!(named, head),
        other => panic!("expected a refusal naming the head, got {other:?}"),
    }
    assert!(llm.requests().is_empty());
    assert_eq!(h.memories(), memories);
    assert_eq!(h.edits(EDIT_REFINED), 1);
    assert_eq!(h.show(head).chain.head, head);
}

#[test]
fn naming_a_translation_again_changes_nothing() {
    let h = Harness::new(Some("English"));
    let chunk = h.fixture_chunk();
    let original = h.memory(chunk, RUSSIAN);
    let head = translated(
        h.translate(
            original,
            &FakeLlm::scripted("fake-llm", vec![reply(ENGLISH)]),
        )
        .unwrap(),
    );
    let memories = h.memories();
    let llm = FakeLlm::scripted("fake-llm", vec![reply(ENGLISH)]);

    let repeated = h.translate(head, &llm).unwrap();

    assert_eq!(
        repeated,
        Translation::AlreadyInLanguage {
            memory: head,
            language: "English".into(),
        }
    );
    assert!(llm.requests().is_empty());
    assert_eq!(h.memories(), memories);
    assert_eq!(h.edits(EDIT_REFINED), 1);
}

#[test]
fn a_sentence_returned_unchanged_writes_nothing() {
    let h = Harness::new(Some("English"));
    let chunk = h.fixture_chunk();
    let english = h.memory(chunk, ENGLISH);
    let memories = h.memories();
    let llm = FakeLlm::scripted("fake-llm", vec![reply(ENGLISH)]);

    let outcome = h.translate(english, &llm).unwrap();

    assert_eq!(
        outcome,
        Translation::AlreadyInLanguage {
            memory: english,
            language: "English".into(),
        }
    );
    assert_eq!(h.memories(), memories);
    assert_eq!(h.edits(EDIT_REFINED), 0);
    assert_eq!(h.show(english).chain.head, english);
}

/// An LLM client that, while it answers, has another writer supersede the
/// memory being translated: what extraction refining it in the meantime
/// would do. It writes through its own connection, so it fails rather than
/// waits if the translation holds a write transaction across the call.
struct SupersededMeanwhile {
    inner: FakeLlm,
    db: PathBuf,
    memory: Uuid,
    by: Uuid,
}

impl LlmClient for SupersededMeanwhile {
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        let conn = rusqlite::Connection::open(&self.db).unwrap();
        conn.busy_timeout(std::time::Duration::ZERO).unwrap();
        let changed = conn
            .execute(
                "UPDATE memories SET superseded_by = (SELECT id FROM memories WHERE uuid = ?2)
                 WHERE uuid = ?1",
                (self.memory.to_string(), self.by.to_string()),
            )
            .expect("the store isn't locked while the LLM answers");
        assert_eq!(changed, 1);
        self.inner.complete(request)
    }
}

#[test]
fn a_memory_superseded_while_the_llm_answers_is_left_to_its_new_head() {
    let h = Harness::new(Some("English"));
    let chunk = h.fixture_chunk();
    let original = h.memory(chunk, RUSSIAN);
    let refinement = h.memory(chunk, "Сэм пьёт зелёный чай каждое утро.");
    let memories = h.memories();
    let llm = SupersededMeanwhile {
        inner: FakeLlm::scripted("fake-llm", vec![reply(ENGLISH)]),
        db: h.dir.data().join(DB_FILE),
        memory: original,
        by: refinement,
    };

    let stale = h.translate(original, &llm);

    match stale {
        Err(TranslateError::Superseded { head }) => assert_eq!(head, refinement),
        other => panic!("expected a refusal naming the new head, got {other:?}"),
    }
    assert_eq!(llm.inner.requests().len(), 1);
    assert_eq!(h.memories(), memories);
    assert_eq!(h.edits(EDIT_REFINED), 0);
    assert_eq!(h.show(refinement).chain.head, refinement);
}

// Concurrency with the bank's other writers

/// Whether a translation has committed: its `memory_refined` edit says so.
fn translation_committed(h: &Harness) -> bool {
    h.one::<bool, _>(
        "SELECT EXISTS (SELECT 1 FROM edits
                        WHERE kind = ?1 AND json_extract(details, '$.translated_to') IS NOT NULL)",
        [EDIT_REFINED],
    )
}

/// Waits up to two seconds for a translation running on another thread to
/// commit. A translation that has to wait for the bank never does, so the
/// caller carries on either way.
fn give_translation_a_chance(h: &Harness) {
    let deadline = Instant::now() + Duration::from_secs(2);
    while !translation_committed(h) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn rowid(h: &Harness, memory: Uuid) -> i64 {
    h.one(
        "SELECT id FROM memories WHERE uuid = ?1",
        [memory.to_string()],
    )
}

/// The stored vector of `memory`, as the index holds it.
fn stored_vector(h: &Harness, memory: Uuid) -> Vec<u8> {
    h.one(
        "SELECT embedding FROM memory_vectors WHERE memory_id = ?1",
        [rowid(h, memory)],
    )
}

/// Which fake model made the stored vector of `memory`, embedding `text`.
fn vector_model(h: &Harness, memory: Uuid, text: &str) -> &'static str {
    let stored = stored_vector(h, memory);
    if stored == vector_bytes(&FakeEmbedderV2, text) {
        FakeEmbedderV2::MODEL_ID
    } else if stored == vector_bytes(&FakeEmbedder, text) {
        FakeEmbedder::MODEL_ID
    } else {
        "neither"
    }
}

fn vector_bytes(embedder: &dyn Embedder, text: &str) -> Vec<u8> {
    embedder.embed(&[text]).unwrap()[0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// [`FakeEmbedderV2`] behind a gate, for a re-embed. Its first call is the
/// job's step: while it runs, a memory is inserted below the job's cursor,
/// so the step never reaches it and the swap has a tail to embed. The
/// second call is the swap's tail, embedded after the swap has read what's
/// unstaged and before its transaction: it says so and waits to be let go.
struct Gate {
    db: PathBuf,
    calls: AtomicUsize,
    paused: Mutex<Option<Sender<()>>>,
    open: Mutex<bool>,
    opened: Condvar,
}

/// The rowid the tail memory takes, below the job's cursor.
const TAIL_ROWID: i64 = 50;

/// A rowid above every other memory, where the job's cursor ends up.
const HIGH_ROWID: i64 = 100;

impl Gate {
    fn new(db: PathBuf) -> (Arc<Self>, Receiver<()>) {
        let (paused, receiver) = mpsc::channel();
        let gate = Arc::new(Self {
            db,
            calls: AtomicUsize::new(0),
            paused: Mutex::new(Some(paused)),
            open: Mutex::new(false),
            opened: Condvar::new(),
        });
        (gate, receiver)
    }

    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.opened.notify_all();
    }
}

impl Embedder for Gate {
    fn model_id(&self) -> &str {
        FakeEmbedderV2.model_id()
    }

    fn dimensions(&self) -> usize {
        FakeEmbedderV2.dimensions()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        match self.calls.fetch_add(1, Ordering::SeqCst) {
            0 => {
                let conn = rusqlite::Connection::open(&self.db).unwrap();
                conn.execute(
                    "INSERT INTO memories (id, uuid, bank_id, content, kind, significance,
                                           chunk_id, source_start, source_end, observed_at,
                                           window_confidence, created_at, updated_at)
                     SELECT ?1, ?2, bank_id, 'Sam walks to work.', kind, significance,
                            chunk_id, source_start, source_end, observed_at,
                            window_confidence, created_at, updated_at
                     FROM memories WHERE id = ?3",
                    (TAIL_ROWID, next_uuid().to_string(), HIGH_ROWID),
                )
                .unwrap();
            }
            1 => {
                if let Some(paused) = self.paused.lock().unwrap().take() {
                    let _ = paused.send(());
                }
                let mut open = self.open.lock().unwrap();
                while !*open {
                    open = self.opened.wait(open).unwrap();
                }
            }
            _ => {}
        }
        FakeEmbedderV2.embed(texts)
    }
}

#[test]
fn a_translation_during_a_reembed_swap_is_embedded_with_the_new_model() {
    // The swap embeds its tail outside the store lock, between reading what's
    // unstaged and its transaction. A translation committed in that gap is in
    // neither, so it has to wait for the swap and be embedded with the model
    // the bank has after it, not keep the old model's vector.
    let h = Harness::new(Some("English"));
    let chunk = h.fixture_chunk();
    let original = h.memory(chunk, RUSSIAN);
    h.execute(
        "INSERT INTO memories (id, uuid, bank_id, content, kind, significance, chunk_id,
                               source_start, source_end, observed_at, window_confidence,
                               created_at, updated_at)
         SELECT ?2, ?3, bank_id, 'Sam reads before bed.', kind, significance, chunk_id,
                source_start, source_end, observed_at, window_confidence,
                created_at, updated_at
         FROM memories WHERE uuid = ?1",
        (original.to_string(), HIGH_ROWID, next_uuid().to_string()),
    );
    let (gate, paused) = Gate::new(h.dir.data().join(DB_FILE));
    let h = h.restart_with(Some("English"), gate.clone(), Arc::new(FakeEmbedder));
    h.service.start_reembed("main").unwrap();

    let (outcome, reembedded) = std::thread::scope(|scope| {
        let reembed = scope.spawn(|| h.service.run_reembed("main"));
        paused
            .recv_timeout(Duration::from_secs(30))
            .expect("the swap reaches its tail");
        let translate = scope.spawn(|| {
            h.translate(
                original,
                &FakeLlm::scripted("fake-llm", vec![reply(ENGLISH)]),
            )
        });
        give_translation_a_chance(&h);
        gate.open();
        let reembedded = reembed.join().unwrap();
        (translate.join().unwrap(), reembedded)
    });

    reembedded.unwrap();
    let head = translated(outcome.unwrap());
    assert_eq!(
        h.one::<String, _>("SELECT embedding_model FROM banks WHERE name = 'main'", []),
        FakeEmbedderV2::MODEL_ID
    );
    assert_eq!(
        vector_model(&h, head, ENGLISH),
        FakeEmbedderV2::MODEL_ID,
        "the translation's vector is the bank's model's"
    );
}

#[test]
fn an_extraction_prepared_before_a_translation_cant_split_the_chain() {
    // Extraction plans a refinement of the memory, then a translation of the
    // same memory arrives before the plan commits. Whichever writes second
    // has to see the first: the memory ends up with one successor and its
    // chain with one head.
    let h = Harness::new(Some("English"));
    let chunk = h.fixture_chunk();
    let before_work = "Sam drinks tea every morning before work.";
    let original = h.memory(chunk, before_work);
    let vector = FakeEmbedder.embed(&[before_work]).unwrap().remove(0);
    let (bank_id, memory_id) = (h.bank_id(), rowid(&h, original));
    let store = h.service.store().unwrap();
    store
        .vectors()
        .upsert(&store.connection(), bank_id, memory_id, &vector)
        .unwrap();

    let green = "Sam drinks green tea every morning before work.";
    h.service
        .ingest_turn(
            "main",
            &Turn {
                session_id: "s1".into(),
                message_at: at("2026-10-01T06:30:00Z"),
                timezone: Some(TZ.into()),
                user_text: green.into(),
                assistant_text: "Noted.".into(),
                author: None,
                platform: Some("cli".into()),
                recall_id: None,
                forget_requested: false,
            },
        )
        .unwrap();
    let call1 = json!({
        "claims": [{
            "content": green,
            "kind": "fact",
            "quote": "Sam drinks green tea every morning before work",
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
        }],
        "used_injected_ids": [],
    });
    let claimed = h
        .service
        .next_extraction("main")
        .unwrap()
        .expect("the turn is queued");
    let input = h
        .service
        .call2_input(&claimed.lease, &call1, &claimed.in_context)
        .unwrap()
        .expect("the memory is a neighbour, so call 2 runs");
    let neighbour = input
        .neighbours
        .iter()
        .find(|neighbour| neighbour.memory == original)
        .expect("the memory is a neighbour")
        .handle
        .clone();
    let call2 = json!({"claims": [{
        "claim": input.claims[0].handle,
        "labels": [{"neighbour": neighbour, "label": "refines"}],
    }]});
    let prepared = h
        .service
        .prepare_extraction(
            claimed.lease,
            &FakeLlm::scripted("fake-llm", vec![call1, call2]),
            &claimed.in_context,
            &claimed.entries,
        )
        .unwrap();

    let (outcome, extracted) = std::thread::scope(|scope| {
        let translate = scope.spawn(|| {
            h.translate(
                original,
                &FakeLlm::scripted("fake-llm", vec![reply(ENGLISH)]),
            )
        });
        give_translation_a_chance(&h);
        let extracted = h.service.commit_extraction(prepared);
        (translate.join().unwrap(), extracted)
    });

    let refinement = extracted.unwrap().memories[0];
    let heads: Vec<Uuid> = [
        Some(refinement),
        match &outcome {
            Ok(Translation::Translated { to, .. }) => Some(*to),
            _ => None,
        },
    ]
    .into_iter()
    .flatten()
    .filter(|memory| h.show(*memory).chain.head == *memory)
    .collect();
    assert_eq!(
        heads.len(),
        1,
        "one head, not {heads:?}; translation {outcome:?}"
    );
    let head = heads[0];
    assert_eq!(h.show(original).chain.head, head);
    assert!(
        h.show(head)
            .chain
            .members
            .iter()
            .any(|member| member.id == original),
        "the head inherits the memory's accesses"
    );
}
