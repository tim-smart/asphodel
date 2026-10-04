//! Entity correction, re-embedding and bank deletion, each run as a daemon job.
//!
//! The API under test is the `Service` methods: `merge_entities`,
//! `unmerge_entity`, `unlink_entity`, `start_reembed`, `run_reembed`,
//! `reembed_status`, `delete_bank`, and `show_memory`'s purge projection. A
//! merge made while a chunk's call 1 is in flight is checked in
//! `extraction.rs`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

use asphodel_core::Service;
use asphodel_core::clock::{Clock, SimulatedClock};
use asphodel_core::config::Tuning;
use asphodel_core::entities::{EntityError, LinkRequest, MergeRequest, Merged};
use asphodel_core::erase::BankDeleteError;
use asphodel_core::extraction::{ExtractError, Extracted};
use asphodel_core::ingest::Turn;
use asphodel_core::inspect::EntityView;
use asphodel_core::mental_models::{FailureKind, ModelSpec, Outcome as RefreshOutcome};
use asphodel_core::models::{
    Embedder, FakeEmbedder, FakeEmbedderV2, FakeLlm, FakeReranker, ModelError, Models,
};
use asphodel_core::reembed::{REEMBED_BATCH, ReembedError, ReembedState};
use asphodel_core::retrieval::{PrefetchRequest, RecallError, RecallRequest};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{OpenOptions, Store};
use jiff::{SignedDuration, Timestamp};
use rusqlite::types::FromSql;
use serde_json::{Value, json};
use uuid::Uuid;

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";

/// When fixture memories were said: before any turn a test queues.
const EARLIER: &str = "2026-09-01T00:00:00Z";

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

/// Floors for both fake embedders and the fake reranker. Bank time runs at
/// full speed with or without turns, and purge is on.
fn tuning() -> Tuning {
    Tuning::from_toml(&format!(
        "[clock]\nquiet_rate = 1.0\n[purge]\ndelta = 1.0\n\
         [injection.reranker_floors]\n\"{}\" = 0.0\n\
         [ranking.relevance_scales]\n\"{0}\" = 1.0\n\
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

/// An embedder that counts its calls and the texts it embedded, fails on
/// one call when told to and, once armed, stops inside its next call until
/// the test releases it, so something else can happen while the model runs.
struct Probe {
    inner: Arc<dyn Embedder>,
    calls: AtomicUsize,
    texts: AtomicUsize,
    fail_on_call: Option<usize>,
    armed: AtomicBool,
    entered: Barrier,
    release: Barrier,
}

impl Probe {
    fn new(inner: impl Embedder + 'static, fail_on_call: Option<usize>) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(inner),
            calls: AtomicUsize::new(0),
            texts: AtomicUsize::new(0),
            fail_on_call,
            armed: AtomicBool::new(false),
            entered: Barrier::new(2),
            release: Barrier::new(2),
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn texts(&self) -> usize {
        self.texts.load(Ordering::SeqCst)
    }
}

impl Embedder for Probe {
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        if self.armed.swap(false, Ordering::SeqCst) {
            self.entered.wait();
            self.release.wait();
        }
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_on_call == Some(call) {
            let model = self.model_id().into();
            let reason = "scripted failure".into();
            return Err(ModelError::Inference { model, reason });
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

    /// A daemon restart on the embedding model `current`, carrying
    /// `previous` for the banks recorded under it when there is one.
    fn restart(self, current: Arc<dyn Embedder>, previous: Option<Arc<dyn Embedder>>) -> Self {
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
        let mut service = Service::with_models(clock.clone(), store, tuning(), models).unwrap();
        if let Some(previous) = previous {
            service = service.with_previous_embedder(previous).unwrap();
        }
        Self {
            service,
            clock,
            dir,
        }
    }

    /// Raw SQL, only to prove data is gone.
    fn one<T: FromSql, P: rusqlite::Params>(&self, sql: &str, params: P) -> T {
        self.service
            .store()
            .unwrap()
            .connection()
            .query_row(sql, params, |row| row.get(0))
            .unwrap()
    }

    fn bank_id(&self, bank: &str) -> i64 {
        self.one("SELECT id FROM banks WHERE name = ?1", [bank])
    }

    /// Rows `memory` has in `table`, which keys rows by memory id.
    fn rows_of(&self, memory: Uuid, table: &str) -> i64 {
        let column = if table == "memories" {
            "id"
        } else {
            "memory_id"
        };
        self.one(
            &format!(
                "SELECT COUNT(*) FROM {table}
                 WHERE {column} = (SELECT id FROM memories WHERE uuid = ?1)"
            ),
            [memory.to_string()],
        )
    }

    /// The owner says each claim's quote in one turn in `bank` at
    /// `message_at`, and it's extracted with call 1 answering `claims`. Call
    /// 2, if a neighbour brings it in, labels nothing.
    fn says(&self, bank: &str, message_at: &str, claims: Vec<Value>) -> Extracted {
        let quotes: Vec<&str> = claims
            .iter()
            .map(|c| c["quote"].as_str().unwrap())
            .collect();
        let turn = turn(message_at, &quotes.join(" "));
        self.service.ingest_turn(bank, &turn).unwrap();
        let call1 = json!({"claims": claims, "used_injected_ids": []});
        let lease = self.service.claim_chunk(bank).unwrap().unwrap();
        let mut replies = vec![call1.clone()];
        if self
            .service
            .call2_input(&lease, &call1, &[])
            .unwrap()
            .is_some()
        {
            replies.push(json!({"claims": []}));
        }
        let llm = FakeLlm::scripted("fake-llm", replies);
        self.service.extract_chunk(lease, &llm, &[]).unwrap()
    }

    /// A fact in `bank`, said at [`EARLIER`].
    fn memory(&self, bank: &str, content: &str) -> Uuid {
        self.says(bank, EARLIER, vec![fact(content)]).memories[0]
    }

    /// `count` facts about tea in `main`, said at [`EARLIER`].
    fn teas(&self, count: usize) -> Vec<Uuid> {
        let claims = (0..count)
            .map(|k| fact(&format!("Memory number {k} about tea.")))
            .collect();
        self.says("main", EARLIER, claims).memories
    }

    /// A fact in `main` naming new entities, each `(name, surface form)`.
    /// Returns the memory and the entities, in order.
    fn naming(&self, content: &str, names: &[(&str, &str)]) -> (Uuid, Vec<Uuid>) {
        let entities: Vec<Value> = names
            .iter()
            .map(|(name, surface)| {
                json!({"entity": null, "new_name": name, "new_kind": "person", "surface_form": surface})
            })
            .collect();
        let claim = fact(content).with("entities", json!(entities));
        let extracted = self.says("main", EARLIER, vec![claim]);
        (extracted.memories[0], extracted.entities_created)
    }

    fn link(&self, memory: Uuid, entity: Uuid) {
        let request = LinkRequest {
            memory: memory.to_string(),
            entity: entity.to_string(),
        };
        assert!(self.service.link_entity("main", &request).unwrap().changed);
    }

    fn entity(&self, entity: Uuid) -> EntityView {
        let entity = entity.to_string();
        self.service.show_entity("main", &entity).unwrap()
    }

    /// The entities `memory` links to.
    fn links(&self, memory: Uuid) -> Vec<Uuid> {
        let view = self.service.show_memory("main", &memory.to_string());
        view.unwrap().entities.iter().map(|e| e.id).collect()
    }

    /// An entity's aliases, sorted.
    fn aliases(&self, entity: Uuid) -> Vec<String> {
        let mut aliases = self.entity(entity).aliases;
        aliases.sort();
        aliases
    }

    fn merged_into(&self, entity: Uuid) -> Option<Uuid> {
        self.entity(entity).merged_into
    }

    fn recorded_model(&self, bank: &str) -> String {
        self.service.reembed_status(bank).unwrap().recorded_model
    }

    fn model(&self, name: &str, entity: Option<&str>) {
        let spec = ModelSpec {
            name: name.into(),
            question: "Who is this?".into(),
            kinds: vec![],
            entity: entity.map(Into::into),
            min_volatility: None,
            max_tokens: 50,
            enabled: true,
        };
        self.service.create_model("main", &spec).unwrap();
    }

    fn try_recall(&self, bank: &str, query: &str) -> Result<Vec<Uuid>, RecallError> {
        let request = RecallRequest {
            query: query.into(),
            ..RecallRequest::default()
        };
        let recall = self.service.recall(bank, &request)?;
        Ok(recall.results.iter().map(|result| result.id).collect())
    }

    fn recall(&self, bank: &str, query: &str) -> Vec<Uuid> {
        self.try_recall(bank, query).unwrap()
    }
}

/// The owner's turn on the CLI in session `s1`, answered "Noted.".
fn turn(message_at: &str, user: &str) -> Turn {
    Turn {
        session_id: "s1".into(),
        message_at: at(message_at),
        timezone: Some(TZ.into()),
        user_text: user.into(),
        assistant_text: "Noted.".into(),
        author: None,
        platform: Some("cli".into()),
        recall_id: None,
        forget_requested: false,
    }
}

/// Call 1's claim of `content`, quoting all of it: a notable fact with no
/// times and no entities.
fn fact(content: &str) -> Value {
    json!({
        "content": content, "kind": "fact", "quote": content, "significance": "notable",
        "remember_this": false, "changes_something": false, "window_confidence": "high",
        "valid_from": null, "valid_until": null, "until_event": null, "due_at": null,
        "volatility": null, "recurrence_text": null, "recurrence_rrule": null,
        "recurrence_start": null, "entities": [],
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

fn merge(h: &Harness, from: &str, into: &str) -> Result<Merged, EntityError> {
    let request = MergeRequest {
        from: from.into(),
        into: into.into(),
    };
    h.service.merge_entities("main", &request)
}

// Merges

#[test]
fn a_merge_moves_aliases_links_and_filters_and_an_unmerge_moves_them_back() {
    let h = Harness::new();
    let (moving, sam) = h.naming("Sammy is moving to Wellington.", &[("Sam", "Sammy")]);
    let sam = sam[0];
    let (tea, samuel) = h.naming("Samuel likes tea.", &[("Samuel", "Samuel")]);
    let samuel = samuel[0];
    let both = h.memory("main", "Sam and Samuel are one person.");
    h.link(both, sam);
    h.link(both, samuel);
    h.model("sam", Some("Sammy"));
    assert_eq!(h.entity(sam).models, vec!["sam"]);

    let merged = merge(&h, &sam.to_string(), "Samuel").unwrap();

    // The entity is kept, merged into `into`, and the aliases and links
    // `into` lacked move to it. A link `into` already had stays on `from`,
    // where it resolves through the merge.
    assert_eq!((merged.from, merged.into), (sam, samuel));
    let moved = (merged.aliases_moved, merged.links_moved);
    assert_eq!((moved, merged.models_repointed), ((2, 1), 1));
    assert_eq!(h.merged_into(sam), Some(samuel));
    assert_eq!(h.aliases(samuel), vec!["Sam", "Sammy", "Samuel"]);
    assert!(h.aliases(sam).is_empty());
    assert_eq!(h.links(moving), vec![samuel]);
    assert_eq!(h.links(tea), vec![samuel]);
    assert_eq!(h.links(both), vec![sam, samuel]);
    assert_eq!(h.entity(samuel).models, vec!["sam"]);

    let edit = merged.edit.to_string();
    let unmerged = h.service.unmerge_entity("main", &edit).unwrap();
    assert_eq!(unmerged.merge, merged.edit);
    let moved = (unmerged.aliases_moved, unmerged.links_moved);
    assert_eq!((moved, unmerged.models_repointed), ((2, 1), 1));
    assert_eq!(h.merged_into(sam), None);
    assert_eq!(h.aliases(sam), vec!["Sam", "Sammy"]);
    assert_eq!(h.aliases(samuel), vec!["Samuel"]);
    assert_eq!(h.links(moving), vec![sam]);
    assert_eq!(h.links(tea), vec![samuel]);
    assert_eq!(h.links(both), vec![sam, samuel]);
    assert_eq!(h.entity(sam).models, vec!["sam"]);
    assert!(h.entity(samuel).models.is_empty());

    // An undone merge can't be undone again.
    assert!(matches!(
        h.service.unmerge_entity("main", &edit),
        Err(EntityError::AlreadyUnmerged)
    ));
}

#[test]
fn user_and_assistant_can_only_be_merge_targets() {
    let h = Harness::new();
    let sam = h.naming("Sam drinks tea.", &[("Sam", "Sam")]).1[0];
    let user = h.service.show_entity("main", "user").unwrap().id;
    let assistant = h.service.show_entity("main", "assistant").unwrap().id;

    let (user_id, sam_id) = (user.to_string(), sam.to_string());
    let pairs = [("user", "Sam"), ("assistant", "Sam"), (&*user_id, &*sam_id)];
    for (from, into) in pairs {
        let refused = merge(&h, from, into);
        assert!(matches!(refused, Err(EntityError::SeededFrom)), "{from}");
    }
    assert_eq!(h.merged_into(user), None);
    assert_eq!(h.merged_into(assistant), None);

    // Into `user` is how the owner's second account is joined up.
    let merged = merge(&h, "Sam", "user").unwrap();
    assert_eq!(merged.into, user);
    assert_eq!(h.merged_into(sam), Some(user));
}

#[test]
fn after_two_merges_an_unmerge_is_refused_and_an_unlink_reaches_every_hop() {
    let h = Harness::new();
    let two = [("Ana", "Ana"), ("Anna", "Anna")];
    let (memory, named) = h.naming("Ana and Anna are one person.", &two);
    let (ana, anna) = (named[0], named[1]);
    let annie = h.naming("Annie lives in Porto.", &[("Annie", "Annie")]).1[0];
    // The first merge keeps the Ana link Anna already had; the second moves
    // Anna's to Annie, leaving the memory linked to Ana and Annie.
    let first = merge(&h, "Ana", "Anna").unwrap();
    merge(&h, "Anna", "Annie").unwrap();
    assert_eq!(h.links(memory), vec![ana, annie], "precondition");

    // The first merge can't be undone once its target was merged again.
    match h.service.unmerge_entity("main", &first.edit.to_string()) {
        Err(EntityError::MergedSince { into }) => assert_eq!(into, annie),
        other => panic!("expected MergedSince, got {other:?}"),
    }
    assert_eq!(h.merged_into(ana), Some(anna));
    assert_eq!(h.merged_into(anna), Some(annie));

    // Unlinking Annie leaves no link that resolves to her, however many
    // hops it's merged through.
    let request = LinkRequest {
        memory: memory.to_string(),
        entity: "Annie".into(),
    };
    assert!(h.service.unlink_entity("main", &request).unwrap().changed);
    assert_eq!(h.links(memory), Vec::<Uuid>::new());
}

// Re-embedding

#[test]
fn a_reembed_resumes_from_its_cursor_and_swaps_in_the_new_model() {
    let h = Harness::new();
    let total = REEMBED_BATCH + 8;
    let memories = h.teas(total);
    assert_eq!(memories.len(), total);
    let other = h.memory("other", "Other bank tea.");

    // The daemon moves to v2, carrying v1. The first batch embeds, the
    // second fails: a run a crash or a model error stopped part way.
    let old = Probe::new(FakeEmbedder, None);
    let new = Probe::new(FakeEmbedderV2, Some(2));
    let h = h.restart(new.clone(), Some(old.clone()));

    // Until the swap the bank is served with the model it recorded.
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
    assert_eq!(stopped.recorded_model, FakeEmbedder::MODEL_ID);
    let old_calls = old.calls();
    h.recall("main", "tea");
    assert!(old.calls() > old_calls, "still served with v1 after a stop");

    // The second run starts after the cursor: every memory is embedded
    // once across both runs.
    let done = h.service.run_reembed("main").unwrap();
    assert_eq!(done.state, ReembedState::Current);
    assert_eq!(done.recorded_model, FakeEmbedderV2::MODEL_ID);
    assert_eq!(new.texts(), total);
    assert_eq!(h.recorded_model("other"), FakeEmbedder::MODEL_ID);

    // After the swap the bank is served with v2, and the other bank still
    // with v1.
    let (old_calls, new_calls) = (old.calls(), new.calls());
    let third = h.recall("main", "Memory number 3 about tea");
    assert!(third.contains(&memories[3]));
    assert_eq!(old.calls(), old_calls);
    assert!(new.calls() > new_calls);
    assert!(h.recall("other", "tea").contains(&other));
    assert!(old.calls() > old_calls);

    // Nothing left to do: a second request finds the bank current.
    let again = h.service.start_reembed("main").unwrap();
    assert_eq!(again.state, ReembedState::Current);
}

// Bank deletion

/// One extraction in `bank`: a turn naming Sam, extracted into one memory
/// linked to a new entity, with a vector, its edits and a recall row.
fn populate(h: &Harness, bank: &str, sentence: &str) -> Uuid {
    h.service
        .ingest_turn(bank, &turn("2026-10-01T06:30:00Z", "Sam drinks tea."))
        .unwrap();
    let sam = json!([{
        "entity": null, "new_name": "Sam", "new_kind": "person", "surface_form": "Sam",
    }]);
    let claim = fact(sentence)
        .with("quote", json!("Sam drinks tea"))
        .with("entities", sam);
    let reply = json!({"claims": [claim], "used_injected_ids": []});
    let lease = h.service.claim_chunk(bank).unwrap().unwrap();
    let llm = FakeLlm::scripted("fake-llm", vec![reply]);
    let memory = h.service.extract_chunk(lease, &llm, &[]).unwrap().memories[0];
    assert!(h.recall(bank, "Sam tea").contains(&memory));
    memory
}

/// Rows `bank_id` holds in each table that keys rows by bank.
fn bank_rows(h: &Harness, bank_id: i64) -> Vec<(&'static str, i64)> {
    let tables = "entities entity_aliases sources chunks extraction_queue memories accesses \
                  recalls edits mental_models session_blocks prompt_blocks sweep_runs \
                  speaker_ids reembeds reembed_vectors";
    let rows = |table| {
        let sql = format!("SELECT COUNT(*) FROM {table} WHERE bank_id = ?1");
        (table, h.one::<i64, _>(&sql, [bank_id]))
    };
    tables.split_whitespace().map(rows).collect()
}

#[test]
fn deleting_a_bank_erases_it_and_leaves_the_other_alone() {
    let h = Harness::new();
    let gone = populate(&h, "main", "Sam drinks tea every morning.");
    let kept = populate(&h, "other", "Sam drinks tea in the other bank.");
    h.model("people", Some("Sam"));
    let main_id = h.bank_id("main");
    let other_id = h.bank_id("other");
    let other_before = bank_rows(&h, other_id);
    let main_before = bank_rows(&h, main_id);

    // The confirmation has to repeat the name; a wrong one changes nothing.
    assert!(matches!(
        h.service.delete_bank("main", "other"),
        Err(BankDeleteError::NotConfirmed)
    ));
    assert_eq!(bank_rows(&h, main_id), main_before);
    assert!(h.recall("main", "Sam tea").contains(&gone));
    assert_eq!(h.rows_of(gone, "memory_vectors"), 1);

    // The recall above added a row.
    let main_before: std::collections::BTreeMap<_, _> =
        bank_rows(&h, main_id).into_iter().collect();
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
    let banks: i64 = h.one("SELECT COUNT(*) FROM banks WHERE id = ?1", [main_id]);
    assert_eq!(banks, 0);
    assert_eq!(h.rows_of(gone, "memory_vectors"), 0);
    assert_eq!(h.rows_of(gone, "memories"), 0);
    assert!(matches!(
        h.try_recall("main", "Sam tea"),
        Err(RecallError::UnknownBank)
    ));

    // The other bank is as it was, and still recalls its memory.
    assert_eq!(bank_rows(&h, other_id), other_before);
    assert!(h.recall("other", "Sam tea").contains(&kept));

    // No edit, the deletion's own audit row included, keeps its content.
    let sql = "SELECT COUNT(*) FROM edits WHERE instr(details, 'every morning') > 0";
    assert_eq!(h.one::<i64, _>(sql, []), 0);
}

// A bank whose recorded model isn't loaded

#[test]
fn a_bank_whose_recorded_model_isnt_loaded_is_refused_until_a_reembed_moves_it() {
    let h = Harness::new();
    let tea = h.memory("main", "Tim likes tea.");
    h.model("tea", None);
    let ingested = h
        .service
        .ingest_turn("main", &turn("2026-10-01T06:30:00Z", "Sam drinks tea."));
    let queued = ingested.unwrap().source;
    let call1 = json!({"claims": [fact("Sam drinks tea.")], "used_injected_ids": []});

    // The daemon moves to v2 without carrying v1, which both banks
    // recorded. A bank created now records v2.
    let new = Probe::new(FakeEmbedderV2, None);
    let h = h.restart(new.clone(), None);
    h.service
        .ensure_bank_with_models("fresh", &identity())
        .unwrap();
    let v1 = FakeEmbedder::MODEL_ID.to_string();

    // Recall and prefetch are refused rather than searched with another
    // model's vectors.
    match h.try_recall("main", "tea") {
        Err(RecallError::ModelUnavailable { model }) => assert_eq!(model, v1),
        other => panic!("expected ModelUnavailable, got {other:?}"),
    }
    let prefetch = PrefetchRequest {
        session_id: "s1".into(),
        query: "What does Tim drink?".into(),
        previous_query: None,
        previous_reply: None,
        block_id: None,
    };
    match h.service.prefetch("main", &prefetch) {
        Err(RecallError::ModelUnavailable { model }) => assert_eq!(model, v1),
        other => panic!("expected ModelUnavailable, got {other:?}"),
    }

    // Extraction is refused before the LLM or the embedder runs, and the
    // attempt isn't counted: the chunk waits at the head of the queue.
    let chunk = |h: &Harness| {
        let source = h.service.show_source("main", &queued.to_string()).unwrap();
        (source.chunks[0].error_count, source.chunks[0].failed_at)
    };
    let llm = FakeLlm::scripted("fake-llm", vec![call1.clone()]);
    let lease = h.service.claim_chunk("main").unwrap().unwrap();
    match &h.service.extract_chunk(lease, &llm, &[]) {
        Err(error @ ExtractError::ModelUnavailable { model }) => {
            assert_eq!(*model, v1);
            assert_eq!(error.failure(), None);
        }
        other => panic!("expected ModelUnavailable, got {other:?}"),
    }
    assert!(llm.requests().is_empty());
    assert_eq!(chunk(&h), (0, None));
    assert_eq!(h.service.queue_depth("main").unwrap(), 1);
    assert!(h.service.failed_chunks("main").unwrap().is_empty());

    // A refresh records a retrieval failure without calling the LLM.
    let llm = FakeLlm::scripted("fake-llm", vec![]);
    let outcome = h.service.refresh_model("main", "tea", &llm, true).unwrap();
    assert!(
        matches!(outcome, RefreshOutcome::Failed(FailureKind::Retrieval)),
        "{outcome:?}"
    );
    assert!(llm.requests().is_empty());

    // Nothing was embedded with the daemon's model for either bank, and
    // both are named as missing their model; the bank on the daemon's
    // model is served.
    assert_eq!(new.calls(), 0);
    assert_eq!(
        h.service.banks_without_their_model().unwrap(),
        vec![("main".into(), v1.clone()), ("other".into(), v1.clone())]
    );
    h.recall("fresh", "tea");
    assert!(new.calls() > 0);

    // The re-embed needs only the daemon's model.
    let started = h.service.start_reembed("main").unwrap();
    assert_eq!(started.state, ReembedState::Pending);
    let done = h.service.run_reembed("main").unwrap();
    assert_eq!(done.state, ReembedState::Current);
    assert_eq!(h.recorded_model("main"), FakeEmbedderV2::MODEL_ID);

    // The bank is served again, on v2, and is no longer named; the other
    // bank, still on v1, is.
    assert!(h.recall("main", "tea").contains(&tea));
    let missing = h.service.banks_without_their_model().unwrap();
    assert_eq!(missing, vec![("other".into(), v1)]);

    // The chunk that waited is extracted on its first attempt, and its
    // memory is found with v2.
    let lease = h.service.claim_chunk("main").unwrap().unwrap();
    let llm = FakeLlm::scripted("fake-llm", vec![call1]);
    let extracted = h.service.extract_chunk(lease, &llm, &[]).unwrap();
    assert_eq!(h.service.queue_depth("main").unwrap(), 0);
    assert_eq!(chunk(&h), (0, None));
    let found = h.recall("main", "Sam drinks tea");
    assert!(found.contains(&extracted.memories[0]));
}

// Regressions: a re-embed's swap racing recall, staged vectors and
// erasure, and the purge projection across a window close.

#[test]
fn a_query_embedded_before_a_swap_is_embedded_again_with_the_new_model() {
    let h = Harness::new();
    h.teas(5);
    let old = Probe::new(FakeEmbedder, None);
    let new = Probe::new(FakeEmbedderV2, None);
    let h = h.restart(new.clone(), Some(old.clone()));
    old.armed.store(true, Ordering::SeqCst);
    // The recall's v1 embedding stops; the bank is re-embedded and swapped to
    // v2 meanwhile; then the recall carries on.
    let (calls, ok) = std::thread::scope(|scope| {
        let recall = scope.spawn(|| h.try_recall("main", "tea"));
        old.entered.wait();
        h.service.start_reembed("main").unwrap();
        h.service.run_reembed("main").unwrap();
        assert_eq!(h.recorded_model("main"), FakeEmbedderV2::MODEL_ID);
        let calls = new.calls();
        old.release.wait();
        (calls, recall.join().unwrap().is_ok())
    });
    // The query is embedded again with v2 rather than searched against
    // vectors of another model.
    assert!(ok);
    assert!(
        new.calls() > calls,
        "the query was embedded with v1 and searched against the swapped-in v2 vectors"
    );
}

#[test]
fn forgetting_a_memory_during_an_interrupted_reembed_erases_its_staged_vector() {
    let h = Harness::new();
    let memories = h.teas(REEMBED_BATCH + 8);
    let new = Probe::new(FakeEmbedderV2, Some(2));
    let h = h.restart(new, Some(Arc::new(FakeEmbedder)));
    h.service.start_reembed("main").unwrap();
    assert!(h.service.run_reembed("main").is_err());
    // The first batch is staged; find a memory in it.
    let forgotten = *memories
        .iter()
        .find(|memory| h.rows_of(**memory, "reembed_vectors") == 1)
        .expect("precondition: a staged vector");
    h.service.forget("main", &[forgotten.to_string()]).unwrap();
    h.service
        .erase_next("main")
        .unwrap()
        .expect("the erase ran");
    assert_eq!(h.rows_of(forgotten, "memories"), 0);
    let staged: i64 = h.one("SELECT COUNT(*) FROM reembed_vectors", []);
    let survives = "the forgotten memory's staged vector survives its erase";
    assert_eq!(staged, (REEMBED_BATCH - 1) as i64, "{survives}");
}

#[test]
fn deleting_a_bank_while_a_reembed_batch_is_in_flight_leaves_nothing_staged() {
    let h = Harness::new();
    h.teas(5);
    let new = Probe::new(FakeEmbedderV2, None);
    let h = h.restart(new.clone(), Some(Arc::new(FakeEmbedder)));
    h.service.start_reembed("main").unwrap();
    new.armed.store(true, Ordering::SeqCst);
    std::thread::scope(|scope| {
        let run = scope.spawn(|| h.service.run_reembed("main"));
        new.entered.wait();
        h.service.delete_bank("main", "main").unwrap();
        new.release.wait();
        let _ = run.join().unwrap();
    });
    assert_eq!(h.one::<i64, _>("SELECT COUNT(*) FROM reembeds", []), 0);
    let staged: i64 = h.one("SELECT COUNT(*) FROM reembed_vectors", []);
    assert_eq!(
        staged, 0,
        "the in-flight batch staged vectors for a deleted bank"
    );
}

#[test]
fn the_projected_purge_date_waits_for_a_window_close_to_fade() {
    let h = Harness::new();
    // A trivial event said now that holds until 1 January 2036.
    let away = fact("Tim is away until 2036.")
        .with("kind", json!("event"))
        .with("significance", json!("trivial"))
        .with(
            "valid_from",
            json!({"at": "2026-10-01", "precision": "day"}),
        )
        .with(
            "valid_until",
            json!({"at": "2036-01-01", "precision": "day"}),
        );
    let event = h.says("main", START, vec![away]).memories[0];
    let advance_to = |to: Timestamp| h.clock.advance(to.duration_since(h.clock.now()));
    advance_to(at("2035-12-29T00:00:00Z"));

    // Below the purge line now, held by its date until the window closes,
    // which restarts recent use and lifts it above the line again.
    let show = || h.service.show_memory("main", &event.to_string()).unwrap();
    let view = show();
    let below = view.strength.value < view.purge.line.unwrap();
    assert!(below, "precondition: below the purge line now");
    let projected = view.projection.purge.clone().expect("a purge date");
    let purgeable = || {
        let candidates = h.service.purge_candidates().unwrap();
        candidates.iter().any(|(_, head)| *head == event)
    };
    advance_to(projected.earliest_at - SignedDuration::from_secs(120));
    assert!(!purgeable(), "purgeable before the projected date");
    advance_to(projected.earliest_at + SignedDuration::from_secs(60));
    assert!(
        purgeable(),
        "projected purgeable at {} ({} bank days), but at that time the sweep doesn't purge it: {:?}",
        projected.earliest_at,
        projected.bank_days,
        show().purge
    );
}
