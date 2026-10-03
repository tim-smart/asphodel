//! Translating a memory into `[llm] language` (`Service::translate_memory`).
//!
//! A translation is a new memory that supersedes the one named, the way a
//! refinement does, so the chain carries strength, accesses and provenance
//! over. The other two tests race a translation against the bank's other
//! writers. Every memory here is synthetic, inserted directly.

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
    Embedder, FakeEmbedder, FakeEmbedderV2, FakeLlm, FakeReranker, LlmClient, ModelError, Models,
};
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

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn next_uuid() -> Uuid {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    Uuid::from_u128((0x7a_u128 << 120) | u128::from(NEXT.fetch_add(1, Ordering::Relaxed)))
}

fn turn(session: &str, user: &str) -> Turn {
    Turn {
        session_id: session.into(),
        message_at: at("2026-10-01T06:30:00Z"),
        timezone: Some(TZ.into()),
        user_text: user.into(),
        assistant_text: "Noted.".into(),
        author: None,
        platform: Some("cli".into()),
        recall_id: None,
        forget_requested: false,
    }
}

/// The LLM's reply for a translation.
fn reply(sentence: &str) -> Value {
    json!({ "sentence": sentence })
}

fn llm(sentence: &str) -> FakeLlm {
    FakeLlm::scripted("fake-llm", vec![reply(sentence)])
}

/// A service on the fakes with one bank, `main`, and an extracted turn
/// saying [`PASSAGE`] for memories to rest on. Field order matters: the
/// service drops before the directory it lives in.
struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    dir: TestDir,
    chunk: i64,
}

impl Harness {
    fn new(language: Option<&str>) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = TestDir(std::env::temp_dir().join(format!(
            "asphodel-translate-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        let clock = Arc::new(SimulatedClock::new(at(START)));
        let mut h = Self::open(dir, clock, 0, language, Models::fake(), None);
        h.service
            .ensure_bank_with_models(
                "main",
                &BankIdentity {
                    owner_name: Some("Tim".into()),
                    owner_platform_ids: vec![],
                    assistant_name: Some("Hermes".into()),
                    timezone: Some(TZ.into()),
                },
            )
            .unwrap();
        h.service
            .ingest_turn("main", &turn("fixtures", PASSAGE))
            .unwrap();
        h.chunk = h.one("SELECT id FROM chunks", []);
        h.execute("DELETE FROM extraction_queue", []);
        h.execute("UPDATE chunks SET extracted_at = ?1", [micros(at(START))]);
        h
    }

    /// The same store under a restarted daemon: `[llm] language` set to
    /// `language`, serving with `models` and carrying `previous` for a bank
    /// recorded under it.
    fn restart(
        self,
        language: Option<&str>,
        models: Models,
        previous: Option<Arc<dyn Embedder>>,
    ) -> Self {
        let Harness {
            service,
            clock,
            dir,
            chunk,
        } = self;
        drop(service);
        Self::open(dir, clock, chunk, language, models, previous)
    }

    fn open(
        dir: TestDir,
        clock: Arc<SimulatedClock>,
        chunk: i64,
        language: Option<&str>,
        models: Models,
        previous: Option<Arc<dyn Embedder>>,
    ) -> Self {
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
        let store =
            Store::open(&dir.0.join("data"), OpenOptions::default(), clock.clone()).unwrap();
        let mut service = Service::with_models(
            clock.clone(),
            store,
            Tuning::from_toml(&toml).unwrap(),
            models,
        )
        .unwrap();
        if let Some(previous) = previous {
            service = service.with_previous_embedder(previous).unwrap();
        }
        Self {
            service,
            clock,
            dir,
            chunk,
        }
    }

    fn db(&self) -> PathBuf {
        self.dir.0.join("data").join(DB_FILE)
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

    /// A state in `main`, inserted directly with no vector: a window, a
    /// volatility, a span into the fixture chunk, an entity link with a
    /// surface form, and accesses on three separate occasions.
    fn memory(&self, content: &str) -> Uuid {
        let uuid = next_uuid();
        let created = micros(at("2026-09-01T00:00:00Z"));
        self.execute(
            "INSERT INTO memories (uuid, bank_id, content, kind, significance, chunk_id,
                                   source_start, source_end, observed_at,
                                   valid_from, valid_from_precision, window_confidence,
                                   volatility, created_at, updated_at)
             SELECT ?1, id, ?2, 'state', 'minor', ?3, 3, 27, ?4, ?4, 'day', 'low',
                    'months', ?4, ?4
             FROM banks WHERE name = 'main'",
            (uuid.to_string(), content, self.chunk, created),
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
             SELECT ?1, id, 'Sam', 'person', ?2, ?2 FROM banks WHERE name = 'main'",
            (entity.to_string(), created),
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

    fn refined_edits(&self) -> i64 {
        self.one("SELECT COUNT(*) FROM edits WHERE kind = ?1", [EDIT_REFINED])
    }

    /// The stored vector of `memory`, as the index holds it.
    fn vector(&self, memory: Uuid) -> Vec<u8> {
        self.one(
            "SELECT embedding FROM memory_vectors
             WHERE memory_id = (SELECT id FROM memories WHERE uuid = ?1)",
            [memory.to_string()],
        )
    }
}

fn vector_bytes(embedder: &dyn Embedder, text: &str) -> Vec<u8> {
    embedder.embed(&[text]).unwrap()[0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The new head of a translation.
fn translated(outcome: Result<Translation, TranslateError>) -> Uuid {
    match outcome {
        Ok(Translation::Translated { to, .. }) => to,
        other => panic!("expected a translation, got {other:?}"),
    }
}

// Supersession

#[test]
fn a_translation_supersedes_the_memory_and_keeps_its_strength_accesses_and_provenance() {
    let h = Harness::new(None);
    let original = h.memory(RUSSIAN);
    h.service
        .set_significance("main", &original.to_string(), Some("major"))
        .unwrap();
    let asked = llm(ENGLISH);

    // Without a target language there's nothing to translate into.
    let refused = h.translate(original, &asked);
    assert!(
        matches!(refused, Err(TranslateError::LanguageUnset)),
        "{refused:?}"
    );
    assert!(asked.requests().is_empty());

    let h = h.restart(Some("English"), Models::fake(), None);
    let before = h.show(original);
    let head = translated(h.translate(original, &asked));

    // The LLM is asked for the target language and shown the sentence, never
    // the passage it came from.
    let requests = asked.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].system.contains("English"));
    assert!(requests[0].user.contains(RUSSIAN));
    assert!(!requests[0].system.contains(PASSAGE) && !requests[0].user.contains(PASSAGE));

    // A refinement: one chain, the original superseded but not retracted.
    let after = h.show(head);
    assert_eq!(after.sentence, ENGLISH);
    assert_eq!(after.chain.head, head);
    let old = after
        .chain
        .members
        .iter()
        .find(|member| member.id == original)
        .expect("the original is in the new head's chain");
    assert_eq!(old.superseded_by, Some(head));
    assert!(!old.retracted && !old.hidden);
    assert_eq!(h.refined_edits(), 1);

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

    // Provenance and everything but the sentence carry over, and the new
    // sentence is what's embedded.
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
    assert_eq!(h.vector(head), vector_bytes(&FakeEmbedder, ENGLISH));

    // Naming an id again writes nothing and asks the LLM nothing: the old
    // id is refused with its head, and the head is already in the language.
    let again = llm(ENGLISH);
    let repeated = h.translate(original, &again);
    assert!(
        matches!(repeated, Err(TranslateError::Superseded { head: named }) if named == head),
        "{repeated:?}"
    );
    assert_eq!(
        h.translate(head, &again).unwrap(),
        Translation::AlreadyInLanguage {
            memory: head,
            language: "English".into(),
        }
    );
    assert!(again.requests().is_empty());
    assert_eq!(h.refined_edits(), 1);
}

// Concurrency with the bank's other writers

/// Waits briefly for a translation running on another thread to commit. A
/// translation that has to wait for the bank never does, so the caller
/// carries on either way.
fn give_translation_a_chance(h: &Harness) {
    let deadline = Instant::now() + Duration::from_millis(500);
    while h.refined_edits() == 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
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
                rusqlite::Connection::open(&self.db)
                    .unwrap()
                    .execute(
                        "INSERT INTO memories (id, uuid, bank_id, content, kind, significance,
                                               chunk_id, source_start, source_end, observed_at,
                                               window_confidence, created_at, updated_at)
                         SELECT -1, ?1, bank_id, 'Sam walks to work.', kind, significance,
                                chunk_id, source_start, source_end, observed_at,
                                window_confidence, created_at, updated_at
                         FROM memories LIMIT 1",
                        [next_uuid().to_string()],
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
    // the bank has after it.
    let h = Harness::new(Some("English"));
    let original = h.memory(RUSSIAN);
    let (gate, paused) = Gate::new(h.db());
    let models = Models {
        embedder: gate.clone(),
        reranker: Arc::new(FakeReranker),
    };
    let h = h.restart(Some("English"), models, Some(Arc::new(FakeEmbedder)));
    h.service.start_reembed("main").unwrap();

    let (outcome, reembedded) = std::thread::scope(|scope| {
        let reembed = scope.spawn(|| h.service.run_reembed("main"));
        paused
            .recv_timeout(Duration::from_secs(30))
            .expect("the swap reaches its tail");
        let translate = scope.spawn(|| h.translate(original, &llm(ENGLISH)));
        give_translation_a_chance(&h);
        gate.open();
        let reembedded = reembed.join().unwrap();
        (translate.join().unwrap(), reembedded)
    });

    reembedded.unwrap();
    let head = translated(outcome);
    assert!(
        h.vector(head) == vector_bytes(&FakeEmbedderV2, ENGLISH),
        "the translation keeps a vector from a model the bank no longer uses"
    );
}

#[test]
fn an_extraction_prepared_before_a_translation_cant_split_the_chain() {
    // Extraction plans a refinement of the memory, then a translation of the
    // same memory arrives before the plan commits. Whichever writes second
    // has to see the first: the memory ends up with one successor and its
    // chain with one head.
    let h = Harness::new(Some("English"));
    let before_work = "Sam drinks tea every morning before work.";
    let original = h.memory(before_work);
    let (bank_id, memory_id): (i64, i64) = (
        h.one("SELECT id FROM banks WHERE name = 'main'", []),
        h.one(
            "SELECT id FROM memories WHERE uuid = ?1",
            [original.to_string()],
        ),
    );
    let store = h.service.store().unwrap();
    store
        .vectors()
        .upsert(
            &store.connection(),
            bank_id,
            memory_id,
            &FakeEmbedder.embed(&[before_work]).unwrap()[0],
        )
        .unwrap();

    let green = "Sam drinks green tea every morning before work.";
    h.service.ingest_turn("main", &turn("s1", green)).unwrap();
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
        .expect("the memory is a neighbour");
    let call2 = json!({"claims": [{
        "claim": input.claims[0].handle,
        "labels": [{"neighbour": neighbour.handle, "label": "refines"}],
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
        let translate = scope.spawn(|| h.translate(original, &llm(ENGLISH)));
        give_translation_a_chance(&h);
        let extracted = h.service.commit_extraction(prepared);
        (translate.join().unwrap(), extracted)
    });

    let refinement = extracted.unwrap().memories[0];
    let translation = match &outcome {
        Ok(Translation::Translated { to, .. }) => Some(*to),
        _ => None,
    };
    let heads: Vec<Uuid> = [Some(refinement), translation]
        .into_iter()
        .flatten()
        .filter(|memory| h.show(*memory).chain.head == *memory)
        .collect();
    assert_eq!(
        heads,
        [h.show(original).chain.head],
        "translation {outcome:?}"
    );
}
