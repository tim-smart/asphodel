//! Entity correction, re-embedding and bank deletion, checked against
//! "Operations: inspection, entity correction, re-embedding and bank
//! deletion" (TIM-115), TIM-99 decisions 5 and 10, and ADR 0010.
//!
//! The API under test is the `Service` methods: `merge_entities`,
//! `unmerge_entity`, `start_reembed`, `run_reembed`, `reembed_status` and
//! `delete_bank`. A merge made while a chunk's call 1 is in flight is
//! checked in `extraction.rs`.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use asphodel_core::Service;
use asphodel_core::clock::SimulatedClock;
use asphodel_core::config::Tuning;
use asphodel_core::entities::{EntityError, MergeRequest};
use asphodel_core::erase::BankDeleteError;
use asphodel_core::extraction::ExtractError;
use asphodel_core::ingest::Turn;
use asphodel_core::mental_models::{FailureKind, ModelSpec, Outcome as RefreshOutcome};
use asphodel_core::models::{
    Embedder, FakeEmbedder, FakeEmbedderV2, FakeLlm, FakeReranker, ModelError, Models,
};
use asphodel_core::reembed::{REEMBED_BATCH, ReembedError, ReembedState};
use asphodel_core::retrieval::{PrefetchRequest, RecallError, RecallRequest};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{OpenOptions, Store, micros};
use jiff::Timestamp;
use rusqlite::types::FromSql;
use serde_json::{Value, json};
use uuid::Uuid;

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
}

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-correction-{}-{}",
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

/// Floors for both fake embedders and the fake reranker.
fn tuning() -> Tuning {
    Tuning::from_toml(&format!(
        "[injection.reranker_floors]\n\"{}\" = 0.0\n\
         [reconcile.embedding_floors]\n\"{}\" = 0.5\n\"{}\" = 0.5\n",
        FakeReranker::MODEL_ID,
        FakeEmbedder::MODEL_ID,
        FakeEmbedderV2::MODEL_ID,
    ))
    .unwrap()
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
    Uuid::from_u128((0xc0_u128 << 120) | u128::from(NEXT.fetch_add(1, Ordering::Relaxed)))
}

/// An embedder that counts its calls and the texts it embedded, and fails
/// on one call when told to.
struct Counting {
    inner: Arc<dyn Embedder>,
    calls: AtomicUsize,
    texts: AtomicUsize,
    fail_on_call: Option<usize>,
}

impl Counting {
    fn new(inner: Arc<dyn Embedder>, fail_on_call: Option<usize>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            calls: AtomicUsize::new(0),
            texts: AtomicUsize::new(0),
            fail_on_call,
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn texts(&self) -> usize {
        self.texts.load(Ordering::SeqCst)
    }
}

impl Embedder for Counting {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_on_call == Some(call) {
            return Err(ModelError::Inference {
                model: self.model_id().into(),
                reason: "scripted failure".into(),
            });
        }
        let vectors = self.inner.embed(texts)?;
        self.texts.fetch_add(texts.len(), Ordering::SeqCst);
        Ok(vectors)
    }
}

/// A service on the fakes with two banks, `main` and `other`, both
/// recorded under `fake-embedder:v1`. Field order matters: the service
/// drops before the directory it lives in.
struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    dir: TestDir,
}

impl Harness {
    fn new() -> Self {
        let dir = TestDir::new();
        let clock = Arc::new(SimulatedClock::new(at(START)));
        let store = Store::open(&dir.data(), OpenOptions::default(), clock.clone()).unwrap();
        let service = Service::with_models(clock.clone(), store, tuning(), Models::fake()).unwrap();
        for bank in ["main", "other"] {
            service.ensure_bank_with_models(bank, &identity()).unwrap();
        }
        Self {
            service,
            clock,
            dir,
        }
    }

    /// A daemon restart on a new embedding model, `current`, carrying
    /// `previous` for the banks recorded under it.
    fn restart_with(self, current: Arc<dyn Embedder>, previous: Arc<dyn Embedder>) -> Self {
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
        let service = Service::with_models(clock.clone(), store, tuning(), models)
            .unwrap()
            .with_previous_embedder(previous)
            .unwrap();
        Self {
            service,
            clock,
            dir,
        }
    }

    /// A daemon restart on `current` alone: banks recorded under any other
    /// model have no embedder to be served with.
    fn restart_on(self, current: Arc<dyn Embedder>) -> Self {
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
        let service = Service::with_models(clock.clone(), store, tuning(), models).unwrap();
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

    fn execute<P: rusqlite::Params>(&self, sql: &str, params: P) {
        self.service
            .store()
            .unwrap()
            .connection()
            .execute(sql, params)
            .unwrap();
    }

    fn bank_id(&self, bank: &str) -> i64 {
        self.one("SELECT id FROM banks WHERE name = ?1", [bank])
    }

    fn recorded_model(&self, bank: &str) -> String {
        self.one("SELECT embedding_model FROM banks WHERE name = ?1", [bank])
    }

    /// An entity in `main` with `aliases`, inserted directly.
    fn entity(&self, name: &str, aliases: &[&str]) -> Uuid {
        let uuid = next_uuid();
        let now = micros(self.now());
        self.execute(
            "INSERT INTO entities (uuid, bank_id, name, kind, created_at, updated_at)
             VALUES (?1, ?2, ?3, 'person', ?4, ?4)",
            (uuid.to_string(), self.bank_id("main"), name, now),
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

    fn seeded(&self, which: &str) -> Uuid {
        let uuid: String = self.one(
            "SELECT uuid FROM entities WHERE bank_id = ?1 AND seeded = ?2",
            (self.bank_id("main"), which),
        );
        uuid.parse().unwrap()
    }

    /// The chunk fixture memories rest on: a turn ingested and marked
    /// extracted, so nothing waits on the queue.
    fn fixture_chunk(&self, bank: &str) -> i64 {
        let found: Vec<i64> = self.all(
            "SELECT c.id FROM chunks c JOIN sources s ON s.id = c.source_id
             WHERE s.bank_id = ?1 AND s.session_id = 'fixtures'",
            [self.bank_id(bank)],
        );
        if let Some(chunk) = found.first() {
            return *chunk;
        }
        self.service
            .ingest_turn(
                bank,
                &turn("fixtures", "2026-09-01T00:00:00Z", "Fixtures.", "Noted."),
            )
            .unwrap();
        let chunk: i64 = self.one(
            "SELECT c.id FROM chunks c JOIN sources s ON s.id = c.source_id
             WHERE s.bank_id = ?1 AND s.session_id = 'fixtures'",
            [self.bank_id(bank)],
        );
        self.execute("DELETE FROM extraction_queue WHERE chunk_id = ?1", [chunk]);
        self.execute(
            "UPDATE chunks SET extracted_at = ?2 WHERE id = ?1",
            (chunk, micros(self.now())),
        );
        chunk
    }

    /// A fact in `bank`, inserted directly, with no vector.
    fn memory(&self, bank: &str, content: &str) -> Uuid {
        let chunk = self.fixture_chunk(bank);
        let uuid = next_uuid();
        let now = micros(self.now());
        self.execute(
            "INSERT INTO memories (uuid, bank_id, content, kind, significance, chunk_id,
                                   source_start, source_end, observed_at, window_confidence,
                                   created_at, updated_at)
             VALUES (?1, ?2, ?3, 'fact', 'notable', ?4, 0, 9, ?5, 'high', ?5, ?5)",
            (uuid.to_string(), self.bank_id(bank), content, chunk, now),
        );
        self.execute(
            "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
             SELECT bank_id, id, 'created', ?2, 0 FROM memories WHERE uuid = ?1",
            (uuid.to_string(), now),
        );
        uuid
    }

    fn link(&self, memory: Uuid, entity: Uuid) {
        self.execute(
            "INSERT INTO memory_entities (memory_id, entity_id)
             SELECT m.id, e.id FROM memories m, entities e WHERE m.uuid = ?1 AND e.uuid = ?2",
            (memory.to_string(), entity.to_string()),
        );
    }

    /// The entities `memory` links to.
    fn links(&self, memory: Uuid) -> Vec<Uuid> {
        self.all::<String, _>(
            "SELECT e.uuid FROM memory_entities me
             JOIN memories m ON m.id = me.memory_id JOIN entities e ON e.id = me.entity_id
             WHERE m.uuid = ?1 ORDER BY e.id",
            [memory.to_string()],
        )
        .iter()
        .map(|uuid| uuid.parse().unwrap())
        .collect()
    }

    /// An entity's aliases, sorted.
    fn aliases(&self, entity: Uuid) -> Vec<String> {
        self.all(
            "SELECT a.alias FROM entity_aliases a JOIN entities e ON e.id = a.entity_id
             WHERE e.uuid = ?1 ORDER BY a.alias",
            [entity.to_string()],
        )
    }

    fn merged_into(&self, entity: Uuid) -> Option<Uuid> {
        self.one::<Option<String>, _>(
            "SELECT m.uuid FROM entities e LEFT JOIN entities m ON m.id = e.merged_into
             WHERE e.uuid = ?1",
            [entity.to_string()],
        )
        .map(|uuid| uuid.parse().unwrap())
    }

    /// The entity a model's filter is on.
    fn filter(&self, model: &str) -> Option<Uuid> {
        self.one::<Option<String>, _>(
            "SELECT e.uuid FROM mental_models m LEFT JOIN entities e ON e.id = m.filter_entity_id
             WHERE m.name = ?1",
            [model],
        )
        .map(|uuid| uuid.parse().unwrap())
    }

    fn edits(&self, kind: &str) -> i64 {
        self.one("SELECT COUNT(*) FROM edits WHERE kind = ?1", [kind])
    }

    fn model(&self, name: &str, entity: &str) {
        self.service
            .create_model(
                "main",
                &ModelSpec {
                    name: name.into(),
                    question: "Who is this?".into(),
                    kinds: vec![],
                    entity: Some(entity.into()),
                    min_volatility: None,
                    max_tokens: 50,
                    enabled: true,
                },
            )
            .unwrap();
    }

    fn recall(&self, bank: &str, query: &str) -> Vec<Uuid> {
        self.service
            .recall(
                bank,
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

fn merge(
    h: &Harness,
    from: &str,
    into: &str,
) -> Result<asphodel_core::entities::Merged, EntityError> {
    h.service.merge_entities(
        "main",
        &MergeRequest {
            from: from.into(),
            into: into.into(),
        },
    )
}

// Merges

#[test]
fn a_merge_moves_aliases_links_and_filters_and_an_unmerge_moves_them_back() {
    let h = Harness::new();
    let sam = h.entity("Sam", &["Sam", "Sammy"]);
    let samuel = h.entity("Samuel", &["Samuel"]);
    let moving = h.memory("main", "Sam is moving to Wellington.");
    let tea = h.memory("main", "Samuel likes tea.");
    let both = h.memory("main", "Sam and Samuel are one person.");
    h.link(moving, sam);
    h.link(tea, samuel);
    h.link(both, sam);
    h.link(both, samuel);
    h.model("sam", "Sammy");
    assert_eq!(h.filter("sam"), Some(sam));

    let merged = merge(&h, &sam.to_string(), "Samuel").unwrap();

    // ADR 0010: the row is kept with `merged_into` set, and the aliases and
    // links `into` lacked move to it in one logged edit. A link `into`
    // already had stays on `from`, where it resolves through the merge.
    assert_eq!((merged.from, merged.into), (sam, samuel));
    assert_eq!(
        (
            merged.aliases_moved,
            merged.links_moved,
            merged.models_repointed
        ),
        (2, 1, 1)
    );
    assert_eq!(h.merged_into(sam), Some(samuel));
    assert_eq!(h.aliases(samuel), vec!["Sam", "Sammy", "Samuel"]);
    assert!(h.aliases(sam).is_empty());
    assert_eq!(h.links(moving), vec![samuel]);
    assert_eq!(h.links(tea), vec![samuel]);
    assert_eq!(h.links(both), vec![sam, samuel]);
    assert_eq!(h.filter("sam"), Some(samuel));
    assert_eq!(h.edits("entity_merged"), 1);

    let unmerged = h
        .service
        .unmerge_entity("main", &merged.edit.to_string())
        .unwrap();
    assert_eq!(unmerged.merge, merged.edit);
    assert_eq!(
        (
            unmerged.aliases_moved,
            unmerged.links_moved,
            unmerged.models_repointed
        ),
        (2, 1, 1)
    );
    assert_eq!(h.merged_into(sam), None);
    assert_eq!(h.aliases(sam), vec!["Sam", "Sammy"]);
    assert_eq!(h.aliases(samuel), vec!["Samuel"]);
    assert_eq!(h.links(moving), vec![sam]);
    assert_eq!(h.links(tea), vec![samuel]);
    assert_eq!(h.links(both), vec![sam, samuel]);
    assert_eq!(h.filter("sam"), Some(sam));
    assert_eq!(h.edits("entity_unmerged"), 1);

    // An undone merge can't be undone again.
    assert!(matches!(
        h.service.unmerge_entity("main", &merged.edit.to_string()),
        Err(EntityError::AlreadyUnmerged)
    ));
}

#[test]
fn user_and_assistant_can_only_be_merge_targets() {
    let h = Harness::new();
    let sam = h.entity("Sam", &["Sam"]);
    let user = h.seeded("user");
    let assistant = h.seeded("assistant");

    for from in ["user", "assistant"] {
        assert!(matches!(
            merge(&h, from, "Sam"),
            Err(EntityError::SeededFrom)
        ));
    }
    assert!(matches!(
        merge(&h, &user.to_string(), &sam.to_string()),
        Err(EntityError::SeededFrom)
    ));
    assert_eq!(h.merged_into(user), None);
    assert_eq!(h.merged_into(assistant), None);
    assert_eq!(h.edits("entity_merged"), 0);

    // Into `user` is how the owner's second account is joined up.
    let merged = merge(&h, "Sam", "user").unwrap();
    assert_eq!(merged.into, user);
    assert_eq!(h.merged_into(sam), Some(user));
}

#[test]
fn an_unmerge_is_refused_once_the_target_has_been_merged_again() {
    let h = Harness::new();
    let ana = h.entity("Ana", &["Ana"]);
    let anna = h.entity("Anna", &["Anna"]);
    let annie = h.entity("Annie", &["Annie"]);
    let first = merge(&h, "Ana", "Anna").unwrap();
    merge(&h, "Anna", "Annie").unwrap();

    match h.service.unmerge_entity("main", &first.edit.to_string()) {
        Err(EntityError::MergedSince { into }) => assert_eq!(into, annie),
        other => panic!("expected MergedSince, got {other:?}"),
    }
    assert_eq!(h.merged_into(ana), Some(anna));
    assert_eq!(h.merged_into(anna), Some(annie));
    assert_eq!(h.edits("entity_unmerged"), 0);
}

// Re-embedding

#[test]
fn a_reembed_resumes_from_its_cursor_and_swaps_in_the_new_model() {
    let h = Harness::new();
    let total = REEMBED_BATCH + 8;
    let memories: Vec<Uuid> = (0..total)
        .map(|k| h.memory("main", &format!("Memory number {k} about tea.")))
        .collect();
    let other = h.memory("other", "Other bank tea.");

    // The daemon moves to v2, carrying v1. The first batch embeds, the
    // second fails: a run a crash or a model error stopped part way.
    let old = Counting::new(Arc::new(FakeEmbedder), None);
    let new = Counting::new(Arc::new(FakeEmbedderV2), Some(2));
    let h = h.restart_with(new.clone(), old.clone());

    // TIM-99, as amended: until the swap the bank is served with the
    // model it recorded.
    h.recall("main", "tea");
    assert!(old.calls() > 0);
    assert_eq!(new.calls(), 0);

    let started = h.service.start_reembed("main").unwrap();
    assert_eq!(started.state, ReembedState::Pending);
    assert_eq!(started.recorded_model, FakeEmbedder::MODEL_ID);
    assert_eq!(started.model, FakeEmbedderV2::MODEL_ID);
    assert!(matches!(
        h.service.run_reembed("main"),
        Err(ReembedError::Model { .. })
    ));
    let stopped = h.service.reembed_status("main").unwrap();
    assert_eq!(stopped.state, ReembedState::Failed);
    assert_eq!(stopped.embedded, REEMBED_BATCH as u64);
    assert_eq!(h.recorded_model("main"), FakeEmbedder::MODEL_ID);
    let old_calls = old.calls();
    h.recall("main", "tea");
    assert!(old.calls() > old_calls, "still served with v1 after a stop");

    // The second run starts after the cursor: nothing the first embedded
    // is embedded again.
    let done = h.service.run_reembed("main").unwrap();
    assert_eq!(done.state, ReembedState::Current);
    assert_eq!(done.recorded_model, FakeEmbedderV2::MODEL_ID);
    assert_eq!(new.texts(), total);
    assert_eq!(h.recorded_model("main"), FakeEmbedderV2::MODEL_ID);
    assert_eq!(h.recorded_model("other"), FakeEmbedder::MODEL_ID);
    assert_eq!(h.one::<i64, _>("SELECT COUNT(*) FROM reembeds", []), 0);
    assert_eq!(
        h.one::<i64, _>("SELECT COUNT(*) FROM reembed_vectors", []),
        0
    );
    assert_eq!(h.edits("reembedded"), 1);
    for memory in &memories {
        assert_eq!(
            h.one::<i64, _>(
                "SELECT COUNT(*) FROM memory_vectors
                 WHERE memory_id = (SELECT id FROM memories WHERE uuid = ?1)",
                [memory.to_string()],
            ),
            1,
            "every memory has a vector under the new model"
        );
    }

    // After the swap the bank is served with v2, and the other bank still
    // with v1.
    let (old_calls, new_calls) = (old.calls(), new.calls());
    h.recall("main", "tea");
    assert_eq!(old.calls(), old_calls);
    assert!(new.calls() > new_calls);
    assert!(h.recall("other", "tea").contains(&other));
    assert!(old.calls() > old_calls);

    // Nothing left to do: a second request finds the bank current.
    assert_eq!(
        h.service.start_reembed("main").unwrap().state,
        ReembedState::Current
    );
}

// Bank deletion

/// One extraction in `bank`: a turn naming Sam, extracted into one memory
/// linked to a new entity, with a vector, its edits and a recall row.
fn populate(h: &Harness, bank: &str, sentence: &str) -> Uuid {
    h.service
        .ingest_turn(
            bank,
            &turn("s1", "2026-10-01T06:30:00Z", "Sam drinks tea.", "Noted."),
        )
        .unwrap();
    let reply = json!({
        "claims": [{
            "content": sentence,
            "kind": "fact",
            "quote": "Sam drinks tea",
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
            "entities": [
                {"entity": null, "new_name": "Sam", "new_kind": "person", "surface_form": "Sam"}
            ],
        }],
        "used_injected_ids": [],
    });
    let lease = h.service.claim_chunk(bank).unwrap().unwrap();
    let extracted = h
        .service
        .extract_chunk(lease, &FakeLlm::scripted("fake-llm", vec![reply]), &[])
        .unwrap();
    let memory = extracted.memories[0];
    assert!(h.recall(bank, "Sam tea").contains(&memory));
    memory
}

/// Rows `bank_id` holds in each table that keys rows by bank.
fn bank_rows(h: &Harness, bank_id: i64) -> Vec<(&'static str, i64)> {
    [
        "entities",
        "entity_aliases",
        "sources",
        "chunks",
        "extraction_queue",
        "memories",
        "accesses",
        "recalls",
        "edits",
        "mental_models",
        "session_blocks",
        "prompt_blocks",
        "sweep_runs",
        "speaker_ids",
        "reembeds",
        "reembed_vectors",
    ]
    .into_iter()
    .map(|table| {
        (
            table,
            h.one::<i64, _>(
                &format!("SELECT COUNT(*) FROM {table} WHERE bank_id = ?1"),
                [bank_id],
            ),
        )
    })
    .collect()
}

#[test]
fn deleting_a_bank_erases_it_and_leaves_the_other_alone() {
    let h = Harness::new();
    let sentence = "Sam drinks tea every morning.";
    let gone = populate(&h, "main", sentence);
    let kept = populate(&h, "other", "Sam drinks tea in the other bank.");
    h.model("people", "Sam");
    let main_id = h.bank_id("main");
    let other_id = h.bank_id("other");
    let other_before = bank_rows(&h, other_id);

    // The confirmation has to repeat the name; a wrong one changes nothing.
    let unconfirmed = bank_rows(&h, main_id);
    assert!(matches!(
        h.service.delete_bank("main", "other"),
        Err(BankDeleteError::NotConfirmed)
    ));
    assert_eq!(bank_rows(&h, main_id), unconfirmed);
    assert!(h.recall("main", "Sam tea").contains(&gone));
    let main_before: std::collections::BTreeMap<_, _> =
        bank_rows(&h, main_id).into_iter().collect();

    let gone_id: i64 = h.one(
        "SELECT id FROM memories WHERE uuid = ?1",
        [gone.to_string()],
    );
    assert_eq!(
        h.one::<i64, _>(
            "SELECT COUNT(*) FROM memory_vectors WHERE memory_id = ?1",
            [gone_id]
        ),
        1
    );

    let deleted = h.service.delete_bank("main", "main").unwrap();
    assert_eq!(deleted.name, "main");
    assert_eq!(deleted.memories, 1);
    assert_eq!(deleted.entities as i64, main_before["entities"]);
    assert_eq!(deleted.sources as i64, main_before["sources"]);
    assert_eq!(deleted.chunks as i64, main_before["chunks"]);
    assert_eq!(deleted.models as i64, main_before["mental_models"]);
    assert_eq!(deleted.recalls as i64, main_before["recalls"]);

    // Nothing of the bank is left: its rows, its vectors, its tombstones,
    // edit rows and session mappings.
    for (table, rows) in bank_rows(&h, main_id) {
        assert_eq!(rows, 0, "{table} still holds rows of the deleted bank");
    }
    assert_eq!(
        h.one::<i64, _>("SELECT COUNT(*) FROM banks WHERE id = ?1", [main_id]),
        0
    );
    assert_eq!(
        h.one::<i64, _>(
            "SELECT COUNT(*) FROM memory_vectors WHERE memory_id = ?1",
            [gone_id]
        ),
        0
    );
    assert_eq!(
        h.one::<i64, _>(
            "SELECT COUNT(*) FROM memories WHERE uuid = ?1",
            [gone.to_string()]
        ),
        0
    );
    assert!(matches!(
        h.service.recall(
            "main",
            &RecallRequest {
                query: "Sam tea".into(),
                ..RecallRequest::default()
            }
        ),
        Err(asphodel_core::retrieval::RecallError::UnknownBank)
    ));

    // The other bank is as it was, and still recalls its memory.
    assert_eq!(bank_rows(&h, other_id), other_before);
    assert!(h.recall("other", "Sam tea").contains(&kept));

    // One daemon-wide row with counts, never content.
    let rows: Vec<String> = h.all(
        "SELECT details FROM edits WHERE kind = 'bank_deleted' AND bank_id IS NULL",
        [],
    );
    assert_eq!(rows.len(), 1);
    let details: Value = serde_json::from_str(&rows[0]).unwrap();
    assert_eq!(details["memories"], 1);
    assert_eq!(details["bank"], deleted.bank.to_string());
    assert!(!rows[0].contains("tea"), "{}", rows[0]);
}

// A bank whose recorded model isn't loaded

/// The turn `populate` extracts, queued in `main` and not yet extracted.
fn queue_tea(h: &Harness) {
    h.service
        .ingest_turn(
            "main",
            &turn("s1", "2026-10-01T06:30:00Z", "Sam drinks tea.", "Noted."),
        )
        .unwrap();
}

/// Call 1's reply to [`queue_tea`]'s turn: one fact, no entities.
fn tea_reply() -> Value {
    json!({
        "claims": [{
            "content": "Sam drinks tea.",
            "kind": "fact",
            "quote": "Sam drinks tea",
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
        }],
        "used_injected_ids": [],
    })
}

/// The `main` chunk's attempt count and failure time.
fn chunk_errors(h: &Harness) -> (i64, Option<i64>) {
    let bank_id = h.bank_id("main");
    let store = h.service.store().unwrap();
    let conn = store.connection();
    conn.query_row(
        "SELECT c.error_count, c.failed_at FROM chunks c JOIN sources s ON s.id = c.source_id
         WHERE s.bank_id = ?1 AND s.session_id = 's1'",
        [bank_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
    .unwrap()
}

/// The `attention` lines naming a bank whose recorded model isn't loaded.
fn model_attention(h: &Harness) -> Vec<String> {
    h.service
        .status()
        .unwrap()
        .attention
        .into_iter()
        .filter(|line| line.contains("isn't loaded"))
        .collect()
}

#[test]
fn a_bank_whose_recorded_model_isnt_loaded_is_refused_without_embedding() {
    let h = Harness::new();
    h.memory("main", "Tim likes tea.");
    h.service
        .create_model(
            "main",
            &ModelSpec {
                name: "tea".into(),
                question: "What does Tim drink?".into(),
                kinds: vec![],
                entity: None,
                min_volatility: None,
                max_tokens: 50,
                enabled: true,
            },
        )
        .unwrap();
    queue_tea(&h);

    // The daemon moves to v2 without carrying v1, which both banks
    // recorded. A bank created now records v2.
    let new = Counting::new(Arc::new(FakeEmbedderV2), None);
    let h = h.restart_on(new.clone());
    h.service
        .ensure_bank_with_models("fresh", &identity())
        .unwrap();
    let unavailable = |model: &str| model == FakeEmbedder::MODEL_ID;

    // ADR 0010, as decided on TIM-115: recall and prefetch are refused
    // rather than searched with another model's vectors.
    match h.service.recall(
        "main",
        &RecallRequest {
            query: "tea".into(),
            ..RecallRequest::default()
        },
    ) {
        Err(RecallError::ModelUnavailable { model }) => assert!(unavailable(&model)),
        other => panic!("expected ModelUnavailable, got {other:?}"),
    }
    match h.service.prefetch(
        "main",
        &PrefetchRequest {
            session_id: "s1".into(),
            query: "What does Tim drink?".into(),
            previous_query: None,
            block_id: None,
        },
    ) {
        Err(RecallError::ModelUnavailable { model }) => assert!(unavailable(&model)),
        other => panic!("expected ModelUnavailable, got {other:?}"),
    }

    // Extraction is refused before the LLM or the embedder runs, and the
    // attempt isn't counted: the chunk waits at the head of the queue.
    let llm = FakeLlm::scripted("fake-llm", vec![tea_reply()]);
    let lease = h.service.claim_chunk("main").unwrap().unwrap();
    let refused = h.service.extract_chunk(lease, &llm, &[]);
    match &refused {
        Err(error @ ExtractError::ModelUnavailable { model }) => {
            assert!(unavailable(model));
            assert_eq!(error.failure(), None);
        }
        other => panic!("expected ModelUnavailable, got {other:?}"),
    }
    assert!(llm.requests().is_empty());
    assert_eq!(chunk_errors(&h), (0, None));
    assert_eq!(h.service.queue_depth("main").unwrap(), 1);
    assert!(h.service.failed_chunks("main").unwrap().is_empty());
    assert!(h.service.claim_chunk("main").unwrap().is_some());

    // A refresh records a retrieval failure without calling the LLM.
    let llm = FakeLlm::scripted("fake-llm", vec![]);
    let outcome = h.service.refresh_model("main", "tea", &llm, true).unwrap();
    assert!(
        matches!(outcome, RefreshOutcome::Failed(FailureKind::Retrieval)),
        "{outcome:?}"
    );
    assert!(llm.requests().is_empty());

    // Nothing was embedded with the daemon's model for either bank.
    assert_eq!(new.calls(), 0);

    // Status names each refused bank and its model, and not the bank on
    // the daemon's model, which is served.
    let lines = model_attention(&h);
    assert_eq!(lines.len(), 2, "{lines:?}");
    for bank in ["main", "other"] {
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with(&format!("{bank}:"))
                    && line.contains(FakeEmbedder::MODEL_ID)
                    && line.contains(&format!("asphodel reembed --bank {bank}"))),
            "{lines:?}"
        );
    }
    assert!(
        !lines.iter().any(|line| line.contains("fresh")),
        "{lines:?}"
    );
    h.recall("fresh", "tea");
    assert!(new.calls() > 0);
}

#[test]
fn a_reembed_recovers_a_bank_whose_recorded_model_isnt_loaded() {
    let h = Harness::new();
    let tea = h.memory("main", "Tim likes tea.");
    queue_tea(&h);
    let new = Counting::new(Arc::new(FakeEmbedderV2), None);
    let h = h.restart_on(new.clone());
    assert!(matches!(
        h.service.recall(
            "main",
            &RecallRequest {
                query: "tea".into(),
                ..RecallRequest::default()
            }
        ),
        Err(RecallError::ModelUnavailable { .. })
    ));

    // The re-embed needs only the daemon's model.
    assert_eq!(
        h.service.start_reembed("main").unwrap().state,
        ReembedState::Pending
    );
    let done = h.service.run_reembed("main").unwrap();
    assert_eq!(done.state, ReembedState::Current);
    assert_eq!(h.recorded_model("main"), FakeEmbedderV2::MODEL_ID);

    // The bank is served again, on v2, and leaves status; the other bank,
    // still on v1, doesn't.
    assert!(h.recall("main", "tea").contains(&tea));
    let lines = model_attention(&h);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(lines[0].starts_with("other:"), "{lines:?}");

    // The chunk that waited is extracted, its first attempt, and its
    // memory gets a v2 vector.
    let embedded = new.texts();
    let lease = h.service.claim_chunk("main").unwrap().unwrap();
    let extracted = h
        .service
        .extract_chunk(
            lease,
            &FakeLlm::scripted("fake-llm", vec![tea_reply()]),
            &[],
        )
        .unwrap();
    assert_eq!(extracted.memories.len(), 1);
    assert!(new.texts() > embedded);
    assert_eq!(h.service.queue_depth("main").unwrap(), 0);
    let memory_id: i64 = h.one(
        "SELECT id FROM memories WHERE uuid = ?1",
        [extracted.memories[0].to_string()],
    );
    assert_eq!(
        h.one::<i64, _>(
            "SELECT COUNT(*) FROM memory_vectors WHERE memory_id = ?1",
            [memory_id]
        ),
        1
    );
    assert_eq!(
        h.one::<i64, _>(
            "SELECT error_count FROM chunks WHERE id = (SELECT chunk_id FROM memories WHERE id = ?1)",
            [memory_id]
        ),
        0
    );
}
