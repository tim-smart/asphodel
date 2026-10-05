//! What the dashboard needs from the service: fade dates on every listed
//! memory, the owner's retraction, and removing a document with every
//! version of it. The routes over these `Service` methods, and that browsing
//! them writes nothing, are checked in `crates/asphodel/tests/serve_http.rs`.
//!
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

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asphodel_core::config::Layer;
use asphodel_core::erase::{DocumentRemoved, RemoveDocumentError};
use asphodel_core::extraction::Committed;
use asphodel_core::ingest::{Document, Ingested, Outcome, Turn};
use asphodel_core::inspect::{
    Fading, Gone, InspectError, MemoryPage, MemoryQuery, MemorySort, MemoryStatus, MemoryView,
    Projected, StatusCounts,
};
use asphodel_core::mental_models::Outcome as RefreshOutcome;
use asphodel_core::models::{FakeEmbedder, FakeLlm, FakeReranker, Models};
use asphodel_core::retract::{RetractError, Retracted};
use asphodel_core::retrieval::{Recall, RecallRequest};
use asphodel_core::store::bank::{BankIdentity, PROFILE_NAME};
use asphodel_core::store::{OpenOptions, Store};
use asphodel_core::{Service, SimulatedClock, Tuning};
use jiff::civil::date;
use jiff::{SignedDuration, Timestamp};
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

fn days(n: i64) -> SignedDuration {
    SignedDuration::from_hours(24 * n)
}

struct TestDir(PathBuf);

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    /// How many fixture turns have been said, so each gets its own minute.
    said: Cell<i64>,
    dir: TestDir,
}

impl Harness {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = TestDir(std::env::temp_dir().join(format!(
            "asphodel-dashboard-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )));
        let h = Self::open(dir, Arc::new(SimulatedClock::new(at(START))), "");
        let identity = BankIdentity {
            owner_name: Some("Tim".into()),
            assistant_name: Some("Hermes".into()),
            timezone: Some(TZ.into()),
            ..BankIdentity::default()
        };
        h.service.ensure_bank_with_models(BANK, &identity).unwrap();
        h
    }

    /// Opens the store in `dir` with the tuning the fakes need, `extra` over
    /// it, and the purge state the store gives it, as `serve` does.
    fn open(dir: TestDir, clock: Arc<SimulatedClock>, extra: &str) -> Self {
        let base = format!(
            "[clock]\nquiet_rate = 1.0\n[strength.significance]\ntrivial = 0.0\n\
             [injection.reranker_floors]\n\"{}\" = 1.0\n\
             [ranking.relevance_scales]\n\"{0}\" = 1.0\n\
             [reconcile.embedding_floors]\n\"{}\" = 0.5\n",
            FakeReranker::MODEL_ID,
            FakeEmbedder::MODEL_ID,
        );
        let tuning = Tuning::from_layers(&[
            Layer {
                origin: "base",
                text: &base,
            },
            Layer {
                origin: "test",
                text: extra,
            },
        ])
        .unwrap();
        let store = Store::open(&dir.0, OpenOptions::default(), clock.clone()).unwrap();
        let pause = store
            .check_fingerprint(&tuning.deletion_fingerprint())
            .unwrap();
        let service = Service::with_models(clock.clone(), store, tuning, Models::fake()).unwrap();
        Self {
            service: service.with_purge_pause(pause),
            clock,
            said: Cell::new(0),
            dir,
        }
    }

    /// The daemon restarting with `extra` tuning: the store stays.
    fn restart_with(self, extra: &str) -> Self {
        drop(self.service);
        Self::open(self.dir, self.clock, extra)
    }

    fn now(&self) -> Timestamp {
        self.service.now()
    }

    /// How many of `memories` still have a row.
    fn rows(&self, memories: &[Uuid]) -> i64 {
        let store = self.service.store().unwrap();
        let sql = "SELECT COUNT(*) FROM memories WHERE uuid = ?1";
        let count = |memory: &Uuid| -> i64 {
            let params = [memory.to_string()];
            let count = store.connection().query_row(sql, params, |row| row.get(0));
            count.unwrap()
        };
        memories.iter().map(count).sum()
    }

    /// The owner says the claim's quote at `message_at`, and the turn is
    /// extracted with call 1 finding `claim` and call 2 giving it `labels`
    /// against their memories. Returns the new memory.
    fn says_at(&self, message_at: Timestamp, claim: Value, labels: &[(Uuid, &str)]) -> Uuid {
        let service = &self.service;
        let quote = claim["quote"].as_str().unwrap();
        service.ingest_turn(BANK, &turn(message_at, quote)).unwrap();
        let lease = service.claim_chunk(BANK).unwrap().expect("queued");
        let call1 = json!({"claims": [claim], "used_injected_ids": []});
        let mut replies = vec![call1.clone()];
        if let Some(input) = service.call2_input(&lease, &call1, &[]).unwrap() {
            let labels: Vec<Value> = labels
                .iter()
                .map(|(memory, label)| {
                    let neighbour = input.neighbours.iter().find(|n| n.memory == *memory);
                    let handle = &neighbour.expect("a neighbour").handle;
                    json!({"neighbour": handle, "label": label})
                })
                .collect();
            replies.push(json!({"claims": [{"claim": input.claims[0].handle, "labels": labels}]}));
        }
        let llm = FakeLlm::scripted(MODEL, replies);
        service.extract_chunk(lease, &llm, &[]).unwrap().memories[0]
    }

    /// As [`Harness::says_at`], a minute after the last fixture said from
    /// [`EARLIER`].
    fn says(&self, claim: Value, labels: &[(Uuid, &str)]) -> Uuid {
        self.said.set(self.said.get() + 1);
        let minutes = SignedDuration::from_mins(self.said.get());
        self.says_at(at(EARLIER) + minutes, claim, labels)
    }

    fn fact(&self, content: &str) -> Uuid {
        self.says(notable(content), &[])
    }

    fn doc(&self, id: &str, text: &str) -> Ingested {
        let document = Document {
            document_id: id.into(),
            text: text.into(),
            reference_date: date(2026, 9, 30),
            reference_date_exact: true,
            timezone: Some(TZ.into()),
        };
        self.service.ingest_document(BANK, &document).unwrap()
    }

    /// Ingests a version of a document and extracts its one chunk with call
    /// 1 finding `sentence`, quoted as written. Returns the source and the
    /// memory.
    fn doc_stating(&self, id: &str, text: &str, sentence: &str) -> (Uuid, Uuid) {
        let source = self.doc(id, text);
        assert_eq!(source.outcome, Outcome::Stored);
        let extracted = self.service.extract_next(BANK, &states(sentence)).unwrap();
        let extracted = extracted.expect("the document's chunk was queued");
        (source.source, extracted.memories[0])
    }

    fn recalled(&self, text: &str) -> Vec<Uuid> {
        ids(&self.service.recall(BANK, &query(text)).unwrap())
    }

    fn show(&self, memory: Uuid) -> MemoryView {
        self.service.show_memory(BANK, &memory.to_string()).unwrap()
    }

    fn fade(&self, memory: Uuid) -> Option<Projected> {
        self.show(memory).projection.fade
    }

    fn page(&self, query: MemoryQuery) -> MemoryPage {
        self.service.list_memories(BANK, &query).unwrap()
    }

    fn listed(&self, status: MemoryStatus) -> BTreeSet<Uuid> {
        let query = MemoryQuery {
            status: Some(status),
            ..MemoryQuery::default()
        };
        self.page(query).memories.iter().map(|m| m.id).collect()
    }

    fn retract(&self, memory: Uuid) -> Result<Retracted, RetractError> {
        self.service.retract(BANK, &memory.to_string())
    }

    fn remove(&self, document: &str) -> DocumentRemoved {
        self.service.remove_document(BANK, document).unwrap()
    }

    /// Runs the sweeps at `now`, then every erase waiting in the queue.
    fn sweep_at(&self, now: Timestamp) {
        self.clock.set(now);
        self.service.run_sweeps().unwrap();
        self.erase();
    }

    fn erase(&self) {
        while self.service.erase_next(BANK).unwrap().is_some() {}
    }

    /// Forces a refresh of the profile whose reply is `text`, citing
    /// `memory`.
    fn cite_in_profile(&self, text: &str, memory: Uuid) {
        let input = self.service.refresh_input(BANK, PROFILE_NAME).unwrap();
        let cited = input.memories.iter().find(|m| m.memory == memory);
        let handle = &cited.expect("the profile's selection has it").handle;
        let llm = FakeLlm::scripted(
            MODEL,
            vec![json!({
                "sections": [{"heading": "About Tim", "text": text}],
                "cites": [handle],
            })],
        );
        let outcome = self.service.refresh_model(BANK, PROFILE_NAME, &llm, true);
        let applied = matches!(outcome, Ok(RefreshOutcome::Applied(_)));
        assert!(applied, "{outcome:?}");
    }
}

/// The owner's turn on the CLI.
fn turn(message_at: Timestamp, user: &str) -> Turn {
    Turn {
        session_id: "fixtures".into(),
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

/// A fact with no times and no entities; the nullable fields call 1 may
/// leave out are left out.
fn notable(content: &str) -> Value {
    json!({
        "content": content,
        "kind": "fact",
        "quote": content,
        "significance": "notable",
        "remember_this": false,
        "changes_something": false,
        "window_confidence": "high",
        "entities": [],
    })
}

fn trivial(content: &str) -> Value {
    with(notable(content), "significance", json!("trivial"))
}

/// An undated open task.
fn task(content: &str) -> Value {
    with(notable(content), "kind", json!("task"))
}

/// A claim flagged as changing something, so its neighbours aren't held to
/// the floor.
fn changing(claim: Value) -> Value {
    with(claim, "changes_something", json!(true))
}

fn with(mut claim: Value, key: &str, value: Value) -> Value {
    claim[key] = value;
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

fn set(ids: &[Uuid]) -> BTreeSet<Uuid> {
    ids.iter().copied().collect()
}

// Browsing

#[test]
fn a_significance_override_changes_the_strength_shown_for_an_existing_memory() {
    let h = Harness::new();
    let memory = h.says(trivial(TEA), &[]);
    let before = h.show(memory);
    let h = h.restart_with("[strength.significance]\ntrivial = 0.1\n");
    let after = h.show(memory);

    assert_eq!(before.significance.value, 0.0);
    assert_eq!(after.significance.value, 0.1);
    assert!(after.strength.value > before.strength.value);
}

#[test]
fn the_memory_list_filters_by_status() {
    let h = Harness::new();
    let berlin = h.fact(BERLIN);
    let passport = h.says(task(PASSPORT), &[]);
    let maya = h.fact(MAYA);
    let moved_out = h.says(
        changing(notable(MOVED_OUT)),
        &[(berlin, "retracts"), (passport, "ends")],
    );
    h.service.forget(BANK, &[maya.to_string()]).unwrap();

    assert_eq!(h.listed(MemoryStatus::Live), set(&[moved_out]));
    assert_eq!(h.listed(MemoryStatus::Retracted), set(&[berlin]));
    assert_eq!(h.listed(MemoryStatus::Ended), set(&[passport]));
    let all: BTreeMap<Uuid, MemoryStatus> = h
        .page(MemoryQuery::default())
        .memories
        .into_iter()
        .map(|memory| (memory.id, memory.status))
        .collect();
    let expected = BTreeMap::from([
        (berlin, MemoryStatus::Retracted),
        (moved_out, MemoryStatus::Live),
        (passport, MemoryStatus::Ended),
        (maya, MemoryStatus::Forgetting),
    ]);
    assert_eq!(all, expected);
}

#[test]
fn each_listed_memory_fades_when_memory_show_says_and_the_soonest_sorts_first() {
    let h = Harness::new();
    let notable = h.fact(BERLIN);
    let trivial = h.says_at(at(START) - days(1), trivial(TEA), &[]);
    let kept = h.fact(MAYA);
    h.service.keep(BANK, &[kept.to_string()]).unwrap();

    let (soon, later) = (h.fade(trivial).unwrap(), h.fade(notable).unwrap());
    assert!(soon.bank_days < later.bank_days, "{soon:?} {later:?}");
    assert_eq!(h.fade(kept), None, "a kept memory never fades");

    let by_fade = MemoryQuery {
        sort: MemorySort::Fade,
        ..MemoryQuery::default()
    };
    let listed = h.page(by_fade).memories;
    let order: Vec<Uuid> = listed.iter().map(|memory| memory.id).collect();
    assert_eq!(order, [trivial, notable, kept]);
    for memory in &listed {
        assert_eq!(memory.fade, h.fade(memory.id), "{}", memory.id);
    }
}

#[test]
fn the_status_counts_follow_every_filter_but_status_the_fade_filter_too() {
    // The counts are the status filter's choices: each says how many the
    // list would hold with that status and every other filter as it is.
    let h = Harness::new();
    let kept = h.fact(MAYA);
    let fading = h.fact(BERLIN);
    let retracted = h.fact(TEA);
    h.service
        .keep(BANK, &[kept.to_string(), retracted.to_string()])
        .unwrap();
    h.retract(retracted).unwrap();
    assert!(h.fade(fading).is_some());

    let page = h.page(MemoryQuery {
        fading: Some(Fading::Never),
        status: Some(MemoryStatus::Live),
        ..MemoryQuery::default()
    });
    let listed: Vec<Uuid> = page.memories.iter().map(|memory| memory.id).collect();
    assert_eq!(listed, [kept]);
    assert_eq!(page.total, 1);
    let statuses = StatusCounts {
        live: 1,
        retracted: 1,
        ..StatusCounts::default()
    };
    assert_eq!(page.statuses, statuses);
}

// Retract

#[test]
fn retracting_takes_a_memory_out_of_everything_live_and_reopens_what_it_ended() {
    let h = Harness::new();
    let berlin = h.fact(BERLIN);
    let moved_out = h.says(changing(notable(MOVED_OUT)), &[(berlin, "ends")]);
    let passport = h.says_at(at(START) - days(1), task(PASSPORT), &[]);
    h.cite_in_profile("Tim no longer lives in Berlin.", moved_out);
    let in_chat = || h.service.in_context(BANK, "chat").unwrap();
    let on_agenda = || h.service.agenda(BANK).unwrap().listed();
    let request = RecallRequest {
        session_id: Some("chat".into()),
        ..query(MOVED_OUT)
    };
    assert!(ids(&h.service.recall(BANK, &request).unwrap()).contains(&moved_out));
    assert!(in_chat().contains(&moved_out));
    let block = h.service.system_prompt(BANK, None).unwrap();
    assert!(block.cited.contains(&moved_out));
    assert!(on_agenda().contains(&passport));

    let retracted = h.retract(moved_out).unwrap();
    assert_eq!(retracted.memory, moved_out);
    assert_eq!(retracted.retracted_at, h.now());
    let reopened = retracted.reopened;
    assert_eq!(reopened, [berlin], "a denial: the ending never held");

    assert!(!h.recalled(MOVED_OUT).contains(&moved_out));
    assert!(!in_chat().contains(&moved_out));
    let block = h.service.system_prompt(BANK, None).unwrap();
    assert!(!block.cited.contains(&moved_out));
    assert!(!block.text.contains("Tim no longer lives in Berlin."));
    let input = h.service.refresh_input(BANK, PROFILE_NAME).unwrap();
    assert!(input.memories.iter().all(|m| m.memory != moved_out));

    let reopened = h.show(berlin);
    assert_eq!(reopened.chain.ended_by, None);
    assert_eq!(reopened.window.valid_until, None);

    let shown = h.show(moved_out);
    assert_eq!(shown.retracted_at, Some(h.now()));
    let edit = shown
        .edits
        .iter()
        .find(|edit| edit.kind == "memory_retracted")
        .expect("the retraction is logged");
    assert_eq!(edit.details["by"], "owner");
    assert!(!edit.details.to_string().contains("Berlin"));

    h.retract(passport).unwrap();
    assert!(!on_agenda().contains(&passport));
}

#[test]
fn retract_refuses_a_superseded_memory_naming_its_head_and_a_repeat() {
    use RetractError::{AlreadyRetracted, Superseded, UnknownBank, UnknownMemory};
    let h = Harness::new();
    let maya = h.fact(MAYA);
    let mia = h.says(changing(notable(MIA)), &[(maya, "refines")]);
    let forgotten = h.fact(TEA);
    h.service.forget(BANK, &[forgotten.to_string()]).unwrap();

    assert!(matches!(
        h.retract(maya),
        Err(Superseded { head }) if head == mia
    ));
    assert!(matches!(h.retract(forgotten), Err(UnknownMemory)));
    assert!(matches!(
        h.service.retract(BANK, "not-a-memory"),
        Err(UnknownMemory)
    ));
    let mia_id = mia.to_string();
    assert!(matches!(
        h.service.retract("nobody", &mia_id),
        Err(UnknownBank)
    ));
    assert!(
        h.listed(MemoryStatus::Retracted).is_empty(),
        "a refused retraction retracts nothing"
    );

    h.retract(mia).unwrap();
    assert!(matches!(h.retract(mia), Err(AlreadyRetracted)));
}

#[test]
fn a_retracted_memory_still_shows_until_the_sweep_purges_it() {
    let h = Harness::new();
    let tea = h.says(trivial(TEA), &[]);
    h.retract(tea).unwrap();
    assert_eq!(h.listed(MemoryStatus::Retracted), set(&[tea]));
    let purge = h.show(tea).projection.purge;
    let purge = purge.expect("a trivial memory said once is purged once it fades");

    h.sweep_at(at(SWEEP));
    assert_eq!(h.rows(&[tea]), 1, "retracting doesn't purge it early");

    h.sweep_at(purge.earliest_at + days(2));
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
    assert_eq!(set(&removed.sources), set(&[v1, v2, v3.source]));
    assert_eq!(set(&removed.forgotten), set(&[tea, berlin]));
    assert_eq!(removed.dequeued, 1, "the waiting version is dequeued");

    // Hidden at once, the text gone at once.
    assert!(!h.recalled(TEA).contains(&tea));
    assert!(!h.recalled(BERLIN).contains(&berlin));
    for source in [v1, v2, v3.source] {
        let shown = h.service.show_source(BANK, &source.to_string()).unwrap();
        assert_eq!(shown.text, None, "{source}");
        assert_eq!(shown.gone, Some(Gone::Removed), "{source}");
    }
    let left = h.service.extract_next(BANK, &states(BIKE)).unwrap();
    assert!(left.is_none(), "nothing of the document is left to extract");

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
    let prepared = h.service.prepare_extraction(lease, &states(TEA), &[]);

    let removed = h.remove(NOTES);
    assert_eq!(removed.sources, [v1.source]);
    assert_eq!(removed.dequeued, 0, "a chunk in flight isn't dequeued");

    let committed = match h.service.try_commit_extraction(prepared.unwrap()) {
        Ok(Committed::Extracted(extracted)) => extracted.memories,
        Ok(Committed::Stale(_)) | Err(_) => Vec::new(),
    };
    for memory in &committed {
        assert!(!h.recalled(TEA).contains(memory), "hidden from the commit");
    }
    h.erase();

    assert_eq!(h.rows(&committed), 0);
    assert!(h.page(MemoryQuery::default()).memories.is_empty());
    let shown = h.service.show_source(BANK, &v1.source.to_string()).unwrap();
    assert_eq!(shown.text, None);
    assert!(shown.chunks.iter().all(|chunk| chunk.memories.is_empty()));
    assert_eq!(h.service.queue_depth(BANK).unwrap(), 0);
}

#[test]
fn sources_tombstoned_before_the_upgrade_keep_their_keys() {
    // The new tombstone reason needs `sources` rebuilt. Rows, keys and the
    // older tombstones must come through it.
    let forget = |h: &Harness| {
        let at = h.now() - SignedDuration::from_mins(1);
        let asked = Turn {
            session_id: "chat".into(),
            forget_requested: true,
            ..turn(at, "Forget what I said about tea.")
        };
        h.service.ingest_turn(BANK, &asked).unwrap()
    };
    let h = Harness::new();
    let (_, tea) = h.doc_stating(NOTES, NOTES_V1, TEA);
    let asked = forget(&h);
    assert_eq!(asked.outcome, Outcome::Tombstone);
    h.service
        .store()
        .unwrap()
        .connection()
        .execute_batch("DELETE FROM migrations WHERE to_version > 11; PRAGMA user_version = 11;")
        .unwrap();
    let h = h.restart_with("");

    let again = forget(&h);
    assert_eq!(again.outcome, Outcome::Duplicate);
    assert_eq!(again.source, asked.source);
    assert_eq!(h.doc(NOTES, NOTES_V1).outcome, Outcome::Duplicate);
    assert!(h.recalled(TEA).contains(&tea));

    let removed = h.remove(NOTES);
    assert_eq!(removed.forgotten, [tea]);
}
