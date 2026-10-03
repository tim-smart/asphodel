//! Translating a memory into `[llm] language`, one memory at a time.
//!
//! The API under test is `Service::translate_memory`. A translation is a new
//! memory that supersedes the one named, the way a refinement does, so the
//! chain carries strength, accesses and provenance over. Every memory here
//! is synthetic, inserted directly.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asphodel_core::Service;
use asphodel_core::clock::SimulatedClock;
use asphodel_core::config::Tuning;
use asphodel_core::extraction::EDIT_REFINED;
use asphodel_core::ingest::Turn;
use asphodel_core::inspect::MemoryView;
use asphodel_core::models::{
    FakeEmbedder, FakeLlm, FakeReranker, LlmClient, LlmError, LlmRequest, LlmResponse, Models,
};
use asphodel_core::retrieval::RecallRequest;
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{DB_FILE, OpenOptions, Store, micros};
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
         [reconcile.embedding_floors]\n\"{}\" = 0.5\n",
        FakeReranker::MODEL_ID,
        FakeEmbedder::MODEL_ID,
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
