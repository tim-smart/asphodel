//! Forget, purge and the nightly sweep follow their deletion contracts.
//!
//! The API under test is the `Service` methods over them: `forget`,
//! `erase_next`, `run_sweeps`, `purge_plan`, `purge_ack` and the purge
//! pause. Memories are extracted from real turns with `FakeLlm`, at the
//! time they were said, and read back through `show_memory`, `show_source`
//! and the audit lists. Raw SQL only proves text is gone, or sets up what
//! this binary's API can't reach: an older schema, or an injected failure.
//!
//! Every service here runs on a `SimulatedClock` stopped at 20:00 on
//! Thursday 1 October 2026 in Auckland (UTC+13) unless a test moves it. The
//! next 04:00 there, when the nightly sweep runs, is 15:00 UTC the same day.
//! The tuning sets `clock.quiet_rate = 1.0`, so bank time is world time,
//! and `purge.delta = 1.0`: a trivial memory said once is purged about 129
//! days later, a notable one after more than twelve years.

use std::cell::Cell;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use asphodel_core::config::PurgePause;
use asphodel_core::entities::{LinkRequest, MergeRequest};
use asphodel_core::erase::{EraseReason, Erased, ForgetRequest, Forgotten};
use asphodel_core::extraction::Extracted;
use asphodel_core::ingest::{Document, Ingested, Outcome, Turn};
use asphodel_core::inspect::{Gone, MemoryView};
use asphodel_core::mental_models::{
    Model, ModelSpec, Outcome as RefreshOutcome, RefreshInput, WRITE_TEMPLATE,
};
use asphodel_core::models::{
    FakeEmbedder, FakeLlm, FakeReranker, LlmClient, LlmError, LlmRequest, LlmResponse, Models,
};
use asphodel_core::operations::{Audit, AuditList, ForgetRow, PurgeRow, RecallRow, SweepRun};
use asphodel_core::queue::{ChunkError, Failure, Lease};
use asphodel_core::retrieval::{PrefetchRequest, Recall, RecallRequest};
use asphodel_core::store::bank::{BankIdentity, PROFILE_NAME};
use asphodel_core::store::{OpenOptions, Store};
use asphodel_core::sweep::{PurgeError, Sweeps};
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

/// The next 04:00 in Auckland after [`START`].
const SWEEP: &str = "2026-10-01T15:00:00Z";

/// The first 04:00 in Auckland more than 90 days after [`START`]: 04:00 on
/// Thursday 31 December 2026.
const PAST_HORIZON: &str = "2026-12-30T15:00:00Z";

/// 04:00 on 15 February 2027 in Auckland, by when a trivial memory said at
/// [`START`] has been purgeable for a week.
const LATER_SWEEP: &str = "2027-02-14T15:00:00Z";

/// When notable fixture memories were said, unless a test says otherwise.
const EARLIER: &str = "2026-09-01T00:00:00Z";

/// Long enough ago that a trivial memory said once then is purged.
const LONG_AGO: &str = "2021-01-01T00:00:00Z";

const BERLIN: &str = "Tim lives in Berlin.";
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
const GARDEN: &str = "Tim's garden needs water.";
const LISBON_WITH_MAYA: &str = "Tim moved to Lisbon with his daughter Maya.";

/// An extraction failure no retry fixes.
const TRANSPORT: ChunkError = ChunkError {
    kind: "transport",
    status: None,
};

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
}

fn minutes(n: i64) -> SignedDuration {
    SignedDuration::from_mins(n)
}

fn days(n: i64) -> SignedDuration {
    SignedDuration::from_hours(24 * n)
}

/// A data dir, which the store creates, removed even when an assertion
/// unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("asphodel-erase-{}-{n}", std::process::id());
        Self(std::env::temp_dir().join(name))
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    tuning: Tuning,
    /// How many turns [`Harness::said`] has sent, so no two share a time.
    seeded: Cell<i64>,
    _dir: TestDir,
}

impl Harness {
    fn new() -> Self {
        Self::starting(START)
    }

    /// A harness on a new data dir with its bank created at `start`.
    fn starting(start: &str) -> Self {
        Self::starting_with(start, "")
    }

    /// As [`Harness::starting`], with `extra` tuning.
    fn starting_with(start: &str, extra: &str) -> Self {
        let clock = Arc::new(SimulatedClock::new(at(start)));
        let harness = Self::open(TestDir::new(), clock, extra);
        harness.create_bank(BANK);
        harness
    }

    /// A harness whose bank was created in 2020, with `past` run before the
    /// clock reaches [`START`] and the daemon starts there. Memories `past`
    /// says with [`Harness::said`] were said, and used, at their time; it
    /// says them in time order.
    fn with_past<T>(past: impl FnOnce(&Harness) -> T) -> (Self, T) {
        let harness = Self::starting("2020-01-01T00:00:00Z");
        let seeded = past(&harness);
        harness.set(at(START));
        (harness.restart_with(""), seeded)
    }

    fn create_bank(&self, name: &str) {
        let identity = BankIdentity {
            owner_name: Some("Tim".into()),
            assistant_name: Some("Hermes".into()),
            timezone: Some(TZ.into()),
            ..BankIdentity::default()
        };
        self.service
            .ensure_bank_with_models(name, &identity)
            .unwrap();
    }

    /// Opens the store in `dir` with the tuning the fakes need, `extra`
    /// TOML on top, and the purge state the store gives it, as `serve` does.
    /// `extra` replaces the `[purge]` table when it has one.
    fn open(dir: TestDir, clock: Arc<SimulatedClock>, extra: &str) -> Self {
        let purge = if extra.contains("[purge]") {
            ""
        } else {
            "[purge]\ndelta = 1.0\n"
        };
        let tuning = Tuning::from_toml(&format!(
            "[clock]\nquiet_rate = 1.0\n{purge}\
             [injection.reranker_floors]\n\"{}\" = 1.0\n\
             [ranking.relevance_scales]\n\"{0}\" = 1.0\n\
             [reconcile.embedding_floors]\n\"{}\" = 0.5\n{extra}",
            FakeReranker::MODEL_ID,
            FakeEmbedder::MODEL_ID,
        ))
        .unwrap();
        let store = Store::open(&dir.0, OpenOptions::default(), clock.clone()).unwrap();
        let fingerprint = tuning.deletion_fingerprint();
        let pause = store.check_fingerprint(&fingerprint).unwrap();
        let service =
            Service::with_models(clock.clone(), store, tuning.clone(), Models::fake()).unwrap();
        Self {
            service: service.with_purge_pause(pause),
            clock,
            tuning,
            seeded: Cell::new(0),
            _dir: dir,
        }
    }

    /// The daemon restarting with `extra` tuning: the store stays.
    fn restart_with(self, extra: &str) -> Self {
        let Self {
            service,
            clock,
            seeded,
            _dir,
            ..
        } = self;
        drop(service);
        Self {
            seeded,
            ..Self::open(_dir, clock, extra)
        }
    }

    fn now(&self) -> Timestamp {
        self.service.now()
    }

    fn set(&self, to: Timestamp) {
        self.clock.set(to);
    }

    /// Runs `sql` on the store's own connection.
    fn sql(&self, sql: &str) {
        let store = self.service.store().unwrap();
        store.connection().execute_batch(sql).unwrap();
    }

    fn count(&self, sql: &str) -> i64 {
        let store = self.service.store().unwrap();
        let conn = store.connection();
        conn.query_row(sql, [], |row| row.get(0)).unwrap()
    }

    fn claim(&self) -> Option<Lease> {
        self.service.claim_chunk(BANK).unwrap()
    }

    fn erase_next(&self) -> Option<Erased> {
        self.service.erase_next(BANK).unwrap()
    }

    fn memory(&self, memory: Uuid) -> MemoryView {
        self.service.show_memory(BANK, &memory.to_string()).unwrap()
    }

    /// Whether each of `memories` still has a row, hidden or not.
    fn exist(&self, memories: &[Uuid]) -> Vec<bool> {
        memories
            .iter()
            .map(|memory| self.service.show_memory(BANK, &memory.to_string()).is_ok())
            .collect()
    }

    fn superseded_by(&self, memory: Uuid) -> Option<Uuid> {
        let view = self.memory(memory);
        let member = view.chain.members.iter().find(|m| m.id == memory);
        member.unwrap().superseded_by
    }

    /// The owner says `text` in `session` a minute ago. Returns the source.
    fn ingest(&self, session: &str, text: &str) -> Uuid {
        let sent = turn(session, self.now() - minutes(1), text);
        let ingested = self.service.ingest_turn(BANK, &sent).unwrap();
        assert_eq!(ingested.outcome, Outcome::Stored);
        ingested.source
    }

    /// A turn in `chat` whose reply call 1 judges used `memory`, which was
    /// in context.
    fn uses(&self, memory: Uuid) {
        self.ingest("chat", "And?");
        let lease = self.claim().expect("queued");
        let input = self.service.call1_input(&lease, &[memory]).unwrap();
        let shown = input.in_context.iter().find(|m| m.memory == memory);
        let call1 = json!({"claims": [], "used_injected_ids": [shown.expect("in context").handle]});
        let llm = FakeLlm::scripted(MODEL, vec![call1]);
        self.service.extract_chunk(lease, &llm, &[memory]).unwrap();
    }

    /// The head of the queue extracted with no claims.
    fn extract_nothing(&self) {
        self.extract_with(vec![], &[]);
    }

    /// The head of the queue extracted with call 1 finding `claim` and call
    /// 2 labelling it `(neighbour, label)` per label. Returns the memory.
    fn extract_one(&self, claim: Value, labels: &[(Uuid, &str)]) -> Uuid {
        let labels: Vec<_> = labels.iter().map(|&(m, label)| (0, m, label)).collect();
        self.extract_with(vec![claim], &labels).memories[0]
    }

    /// The owner said `claim`'s quote at `when`, and the turn was extracted
    /// with call 1 finding `claim` and call 2 labelling nothing. Returns the
    /// memory. The clock moves to `when` if that's ahead.
    fn said(&self, when: &str, claim: Value) -> Uuid {
        self.said_changing(when, claim, &[])
    }

    /// As [`Harness::said`], with call 2 labelling the claim `(neighbour,
    /// label)` for each of `labels`.
    fn said_changing(&self, when: &str, claim: Value, labels: &[(Uuid, &str)]) -> Uuid {
        self.said_turn(when, claim["quote"].as_str().unwrap());
        self.extract_one(claim, labels)
    }

    /// The owner said `text` at `when`, and the turn is queued.
    fn said_turn(&self, when: &str, text: &str) {
        let n = self.seeded.get();
        self.seeded.set(n + 1);
        let message_at = at(when) + SignedDuration::from_secs(n);
        if message_at > self.now() {
            self.set(message_at);
        }
        let sent = turn("earlier", message_at, text);
        self.service.ingest_turn(BANK, &sent).unwrap();
    }

    /// The owner says the claim's quote in session `chat` a minute ago, and
    /// the turn is extracted with call 1 finding `claim` and call 2 labelling
    /// nothing. Returns the memory and its source.
    fn says(&self, claim: Value) -> (Uuid, Uuid) {
        let source = self.ingest("chat", claim["quote"].as_str().unwrap());
        (self.extract_one(claim, &[]), source)
    }

    /// As [`Harness::says`], with call 2 labelling the claim `label` on
    /// `neighbour`.
    fn says_changing(&self, claim: Value, neighbour: Uuid, label: &str) -> (Uuid, Uuid) {
        let source = self.ingest("chat", claim["quote"].as_str().unwrap());
        (self.extract_one(claim, &[(neighbour, label)]), source)
    }

    /// Extracts the head of the queue with call 1 finding `claims`. Call 2
    /// labels `(claim index, neighbour, label)`, or nothing.
    fn extract_with(&self, claims: Vec<Value>, labels: &[(usize, Uuid, &str)]) -> Extracted {
        let call1 = json!({"claims": claims, "used_injected_ids": []});
        let call2 = if labels.is_empty() {
            json!({"claims": []})
        } else {
            let lease = self.claim().expect("queued");
            let input = self.service.call2_input(&lease, &call1, &[]).unwrap();
            let input = input.expect("call 2 runs");
            let labelled: Vec<Value> = labels
                .iter()
                .map(|&(index, neighbour, label)| {
                    let claim = input.claims.iter().find(|c| c.claim == index);
                    let neighbour = input.neighbours.iter().find(|n| n.memory == neighbour);
                    json!({
                        "claim": claim.expect("the claim reaches call 2").handle,
                        "labels": [{
                            "neighbour": neighbour.expect("the memory is a neighbour").handle,
                            "label": label,
                        }],
                    })
                })
                .collect();
            json!({ "claims": labelled })
        };
        let llm = FakeLlm::scripted(MODEL, vec![call1, call2]);
        let extracted = self.service.extract_next(BANK, &llm).unwrap();
        extracted.expect("a chunk was queued")
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

    /// The source's stored text, empty once it's gone.
    fn text(&self, source: Uuid) -> String {
        let detail = self.service.show_source(BANK, &source.to_string());
        detail.unwrap().text.unwrap_or_default()
    }

    /// The source's chunk text, read past the API to prove a passage is gone
    /// from the chunks too.
    fn chunk_text(&self, source: Uuid) -> String {
        let store = self.service.store().unwrap();
        let conn = store.connection();
        conn.query_row(
            "SELECT coalesce(group_concat(c.text, ''), '') FROM chunks c
             JOIN sources s ON s.id = c.source_id WHERE s.uuid = ?1",
            [source.to_string()],
            |row| row.get(0),
        )
        .unwrap()
    }

    fn message_at(&self, source: Uuid) -> Timestamp {
        let detail = self.service.show_source(BANK, &source.to_string());
        detail.unwrap().message_at.unwrap()
    }

    fn recall(&self, request: RecallRequest) -> Recall {
        self.service.recall(BANK, &request).unwrap()
    }

    fn profile(&self) -> Model {
        let models = self.service.list_models(BANK).unwrap();
        let profile = models.into_iter().find(|model| model.name == PROFILE_NAME);
        profile.unwrap()
    }

    /// Forces a refresh of the profile whose reply is `text`, citing
    /// `cites`.
    fn profile_citing(&self, text: &str, cites: &[Uuid]) {
        let input: RefreshInput = self.service.refresh_input(BANK, PROFILE_NAME).unwrap();
        let handles: Vec<String> = cites.iter().map(|m| handle(&input, *m)).collect();
        let reply = written(text, &handles);
        let llm = FakeLlm::scripted(MODEL, vec![reply]);
        let outcome = self.service.refresh_model(BANK, PROFILE_NAME, &llm, true);
        assert!(
            matches!(outcome, Ok(RefreshOutcome::Applied(_))),
            "{outcome:?}"
        );
        assert!(self.profile().answer.is_some());
    }

    /// The names of the bank's entities.
    fn entity_names(&self) -> BTreeSet<String> {
        let entities = self.service.entities(BANK).unwrap();
        entities.into_iter().map(|entity| entity.name).collect()
    }

    /// Runs the sweeps due now, then any erase they queued.
    fn sweep(&self) -> Sweeps {
        let sweeps = self.service.run_sweeps().unwrap();
        while self.erase_next().is_some() {}
        sweeps
    }

    fn forget(&self, memories: &[Uuid]) -> Forgotten {
        let ids: Vec<String> = memories.iter().map(Uuid::to_string).collect();
        self.service.forget(BANK, &ids).unwrap()
    }

    fn audit(&self, bank: &str, list: AuditList, limit: Option<usize>) -> Audit {
        self.service.audit(bank, list, limit).unwrap()
    }

    fn purge_rows(&self, bank: &str) -> Vec<PurgeRow> {
        let Audit::Purges { purges } = self.audit(bank, AuditList::Purges, None) else {
            unreachable!()
        };
        purges
    }

    /// The sweep runs of `bank`, newest first.
    fn sweep_rows(&self, bank: &str, limit: Option<usize>) -> Vec<SweepRun> {
        let Audit::Sweeps { sweeps } = self.audit(bank, AuditList::Sweeps, limit) else {
            unreachable!()
        };
        sweeps
    }

    /// Every forget's audit row, oldest first.
    fn forget_rows(&self) -> Vec<ForgetRow> {
        let Audit::Forgets { mut forgets } = self.audit(BANK, AuditList::Forgets, None) else {
            unreachable!()
        };
        forgets.reverse();
        forgets
    }

    fn recall_rows(&self) -> Vec<RecallRow> {
        let Audit::Recalls { recalls } = self.audit(BANK, AuditList::Recalls, None) else {
            unreachable!()
        };
        recalls
    }

    fn forget_in(&self, session: Option<&str>, memories: &[Uuid]) {
        let request = ForgetRequest {
            ids: memories.iter().map(Uuid::to_string).collect(),
            session_id: session.map(str::to_string),
        };
        self.service.forget_request(BANK, &request).unwrap();
    }

    /// The turn that called `memory_forget`, a minute ago.
    fn forget_turn(&self, session: &str, text: &str) -> Ingested {
        let mut request = turn(session, self.now() - minutes(1), text);
        request.forget_requested = true;
        self.service.ingest_turn(BANK, &request).unwrap()
    }

    /// The next extraction finds `claim` quoting a passage that mentions
    /// `neighbour` again, so no memory is created.
    fn extract_mention(&self, claim: Value, neighbour: Uuid) {
        let extracted = self.extract_with(vec![claim], &[(0, neighbour, "mentioned_again")]);
        assert!(extracted.memories.is_empty(), "the repeat is no new memory");
    }

    /// How many restatements the store holds, read past the API to prove
    /// they're gone.
    fn restatement_rows(&self) -> i64 {
        self.count("SELECT count(*) FROM restatements")
    }

    /// Every mention span gone, as on a store from before version 7.
    fn make_legacy(&self) {
        self.sql("UPDATE accesses SET spans = NULL; DELETE FROM mention_passages;");
    }

    /// Fails the head of the queue for good. Returns the chunk.
    fn fail_head(&self) -> Uuid {
        for _ in 0..100 {
            let lease = self.claim().expect("queued");
            let chunk = lease.chunk;
            if self.service.fail_chunk(lease, TRANSPORT).unwrap() == Failure::Failed {
                return chunk;
            }
        }
        panic!("the chunk never failed");
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
        "content": content, "kind": "fact", "quote": content, "significance": "notable",
        "remember_this": false, "changes_something": false, "valid_from": null,
        "valid_until": null, "window_confidence": "high", "until_event": null, "due_at": null,
        "volatility": null, "recurrence_text": null, "recurrence_rrule": null,
        "recurrence_start": null, "entities": [],
    })
}

fn with(mut claim: Value, key: &str, value: Value) -> Value {
    claim[key] = value;
    claim
}

fn trivial(content: &str) -> Value {
    with(notable(content), "significance", json!("trivial"))
}

/// A trivial point event on the local day `day`.
fn event(content: &str, day: &str) -> Value {
    let claim = with(trivial(content), "kind", json!("event"));
    with(claim, "valid_from", json!({"at": day, "precision": "day"}))
}

/// A trivial open task due on the local day `due`, or undated.
fn task(content: &str, due: Option<&str>) -> Value {
    let claim = with(trivial(content), "kind", json!("task"));
    let due = due.map(|day| json!({"at": day, "precision": "day"}));
    with(claim, "due_at", json!(due))
}

/// A claim flagged as changing something, so its neighbours aren't held to
/// the floor.
fn changes(claim: Value) -> Value {
    with(claim, "changes_something", json!(true))
}

/// `claim` quoting `quote` rather than its sentence.
fn quoting(claim: Value, quote: &str) -> Value {
    with(claim, "quote", json!(quote))
}

/// `claim` proposing a new entity per name.
fn naming(claim: Value, entities: &[(&str, &str)]) -> Value {
    let entities: Vec<Value> = entities
        .iter()
        .map(|(name, kind)| {
            json!({"entity": null, "new_name": name, "new_kind": kind, "surface_form": name})
        })
        .collect();
    with(claim, "entities", json!(entities))
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

/// A refresh write reply: `text` under one heading, citing `cites`.
fn written(text: &str, cites: &[String]) -> Value {
    json!({"sections": [{"heading": "About Tim", "text": text}], "cites": cites})
}

fn handle(input: &RefreshInput, memory: Uuid) -> String {
    let found = input.memories.iter().find(|m| m.memory == memory);
    found
        .expect("the memory is in the refresh input")
        .handle
        .clone()
}

/// A harness where Maya and Tea were said earlier.
fn maya_and_tea() -> (Harness, Uuid, Uuid) {
    let h = Harness::new();
    let maya = h.said(EARLIER, notable(MAYA));
    let tea = h.said(EARLIER, notable(TEA));
    (h, maya, tea)
}

// Forget

#[test]
fn forget_hides_at_once_and_erases_behind_a_queued_chunk() {
    let h = Harness::new();
    let (maya, said) = h.says(notable(MAYA));
    h.profile_citing("Tim has a daughter called Maya.", &[maya]);
    let recalled = h.recall(RecallRequest {
        session_id: Some("chat".into()),
        ..query(MAYA)
    });
    assert!(ids(&recalled).contains(&maya));
    // The correction is queued with Maya in context.
    let correction_text = "My daughter is called Mia, not Maya.";
    let correction = h.ingest("chat", correction_text);

    let asked = [maya.to_string(), "not-a-memory".into()];
    let forgotten = h.service.forget(BANK, &asked).unwrap();
    assert_eq!(forgotten.forgotten, vec![maya]);
    assert_eq!(forgotten.unknown, vec!["not-a-memory".to_string()]);

    // Everything that can be undone happens at once, and the recall row
    // naming Maya is deleted.
    assert!(!ids(&h.recall(query(MAYA))).contains(&maya));
    let profile = h.profile();
    assert_eq!(profile.answer, None);
    assert!(profile.cites.is_empty());
    let block = h.service.system_prompt(BANK, None).unwrap();
    assert!(!block.text.contains("Maya"), "{}", block.text);
    assert!(!h.service.in_context(BANK, "chat").unwrap().contains(&maya));
    let sql = format!(
        "SELECT count(*) FROM recalls WHERE uuid = '{}'",
        recalled.recall_id
    );
    assert_eq!(h.count(&sql), 0);

    // The rows wait behind the chunk queued before the forget.
    assert!(h.memory(maya).hidden_at.is_some());
    assert_eq!(h.erase_next(), None);

    // That chunk reconciles against the hidden memory, so the correction
    // joins its chain, hidden from the moment it's committed.
    let claim = quoting(changes(notable(MIA)), "My daughter is called Mia, not Maya");
    let mia = h.extract_one(claim, &[(maya, "retracts")]);
    assert_eq!(h.superseded_by(maya), Some(mia));
    assert!(!ids(&h.recall(query(MIA))).contains(&mia));

    let erased = h.erase_next().expect("the erase is at the queue's head");
    assert_eq!(erased.reason, EraseReason::Forget);
    assert_eq!(erased.memories, BTreeSet::from([maya, mia]));
    assert_eq!(h.exist(&[maya, mia]), [false, false]);

    // Both passages are redacted, and the content-hash tombstone keeps the
    // correction from being ingested again.
    for (source, word) in [(said, "Maya"), (correction, "Mia")] {
        assert!(!h.text(source).contains(word), "{source}");
        assert!(!h.chunk_text(source).contains(word), "{source}");
    }
    let resent = turn("chat", h.message_at(correction), correction_text);
    let again = h.service.ingest_turn(BANK, &resent).unwrap();
    assert_eq!(again.outcome, Outcome::Duplicate);
    assert_eq!(h.service.queue_depth(BANK).unwrap(), 0);
}

#[test]
fn a_chunk_queued_after_a_forget_isnt_claimed_until_the_erase_has_run() {
    // The erase waits behind the chunks queued before the forget. With a
    // pool of leases, a chunk queued after it could be out at the same
    // time and reconcile against the hidden memory; committed after the
    // erase, its labels would be dropped as vanished and the forgotten
    // content created again. So the erase is a barrier to claims.
    let h = Harness::new().restart_with("[llm]\nconcurrency = 2\n");
    let maya = h.said(EARLIER, notable(MAYA));
    let before = h.ingest("chat", "My daughter Maya likes tea.");
    h.forget(&[maya]);
    h.set(h.now() + minutes(1));
    let after = h.ingest("chat", "My daughter is called Maya.");

    let first = h.claim().expect("the chunk queued before the forget");
    assert_eq!(first.source, before);
    // The chunk queued after the forget waits for the erase, and a due
    // erase still holds it back until it has run.
    assert!(h.claim().is_none());
    h.service.complete_chunk(first).unwrap();
    assert!(h.claim().is_none());

    h.erase_next().expect("the erase is at the queue's head");
    assert_eq!(h.claim().expect("the erase has run").source, after);
}

#[test]
fn at_concurrency_one_a_turn_after_a_forget_waits_for_the_erase_behind_an_earlier_document() {
    // Turns go ahead of documents, so the turn queued after the forget
    // sorts first. It still waits: it would reconcile against the hidden
    // memory and could commit after the erase.
    let h = Harness::new();
    let maya = h.said(EARLIER, notable(MAYA));
    let notes = h.doc("notes.md", "# Notes\n\nMaya likes tea.\n").source;
    h.forget(&[maya]);
    let after = h.ingest("chat", "My daughter is called Maya.");

    let first = h.claim().expect("the document's chunk");
    assert_eq!(first.source, notes);
    h.service.complete_chunk(first).unwrap();
    assert!(h.claim().is_none());
    h.erase_next().expect("the erase is at the queue's head");
    assert_eq!(h.claim().unwrap().source, after);
}

#[test]
fn forget_erases_the_whole_chain_and_clears_what_points_into_it() {
    // forget takes every memory along `superseded_by`, whichever version
    // it names. `ended_by` isn't a chain link: Berlin stays, with its end
    // and without the pointer. Orphan entities go, except seeded ones and
    // merge tombstones, and so does a `used` credit held for the chain.
    let h = Harness::starting_with(START, "[strength]\ncorroborate_used = true\n");
    let berlin = h.said(EARLIER, naming(notable(BERLIN), &[("Berlin", "place")]));
    let moved_out_claim = changes(notable("Tim moved out of Berlin."));
    let (moved_out, first) = h.says_changing(moved_out_claim, berlin, "ends");
    let ended = h.memory(berlin);
    assert_eq!(ended.chain.ended_by, Some(moved_out));
    let until = ended.window.valid_until;
    assert!(until.is_some());
    let moved_claim = naming(
        changes(notable(MOVED)),
        &[("Lisbon", "place"), ("Lisboa", "place")],
    );
    let (moved, second) = h.says_changing(moved_claim, moved_out, "refines");
    assert_eq!(h.superseded_by(moved_out), Some(moved));
    h.uses(moved);
    assert_eq!(h.count("SELECT count(*) FROM pending_credits"), 1);
    for entity in ["Berlin", "user"] {
        let memory = moved.to_string();
        let link = LinkRequest {
            memory,
            entity: entity.into(),
        };
        h.service.link_entity(BANK, &link).unwrap();
    }
    let merge = MergeRequest {
        from: "Lisboa".into(),
        into: "Berlin".into(),
    };
    h.service.merge_entities(BANK, &merge).unwrap();

    let forgotten = h.forget(&[moved_out]);
    let chain = BTreeSet::from([moved_out, moved]);
    let hidden: BTreeSet<Uuid> = forgotten.forgotten.iter().copied().collect();
    assert_eq!(hidden, chain);
    assert!(forgotten.unknown.is_empty());
    let erased = h.erase_next().expect("nothing was queued first");
    assert_eq!(erased.memories, chain);

    assert_eq!(h.exist(&[moved_out, moved, berlin]), [false, false, true]);
    let kept = h.memory(berlin);
    assert_eq!(kept.chain.ended_by, None);
    assert_eq!(kept.window.valid_until, until);

    let names = h.entity_names();
    for kept in ["Berlin", "Lisboa", "Tim", "Hermes"] {
        assert!(names.contains(kept), "entity {kept}: {names:?}");
    }
    assert!(!names.contains("Lisbon"));
    let aliases = h.count("SELECT count(*) FROM entity_aliases WHERE alias='Lisbon'");
    assert_eq!(aliases, 0);
    assert_eq!(h.count("SELECT count(*) FROM pending_credits"), 0);

    for source in [first, second] {
        assert!(!h.text(source).contains("Berlin"), "{source}");
    }
    assert_eq!(h.forget_rows().len(), 1);
    // The audit rows hold no content.
    let content = h.count(
        "SELECT count(*) FROM edits WHERE kind IN ('forget', 'forgotten')
         AND (details LIKE '%Berlin%' OR details LIKE '%Lisbon%')",
    );
    assert_eq!(content, 0);
}

#[test]
fn forget_takes_one_of_two_overlapping_heads_and_leaves_the_other() {
    // A weightier fact call 2 labelled a repeat of a trivial event isn't
    // the event's next version, so the two stand as separate chains that
    // share words. Forgetting the relationship takes its own chain and
    // passage and leaves the event, which never said it. Forgetting the
    // event leaves the relationship head, whose sentence still says the
    // flowers were sent: forget follows chains, not overlapping content.
    const FLOWERS: &str = "Tim sent flowers to Sam.";
    const WIFE: &str = "Tim sent flowers to Sam, his wife.";
    let wife_claim = || with(notable(WIFE), "significance", json!("major"));
    for forget_wife in [true, false] {
        let h = Harness::new();
        let flowers = h.said(EARLIER, with(trivial(FLOWERS), "kind", json!("event")));
        let (wife, said) = h.says_changing(wife_claim(), flowers, "mentioned_again");
        assert_eq!(h.superseded_by(flowers), None, "two heads");

        let (gone, kept) = if forget_wife {
            (wife, flowers)
        } else {
            (flowers, wife)
        };
        h.forget(&[gone]);
        let erased = h.erase_next().expect("nothing was queued first");
        assert_eq!(erased.memories, BTreeSet::from([gone]), "{forget_wife}");
        assert_eq!(h.exist(&[gone, kept]), [false, true], "{forget_wife}");
        assert!(h.memory(kept).hidden_at.is_none(), "{forget_wife}");
        if forget_wife {
            assert!(
                !h.text(said).contains("wife"),
                "the relationship is redacted"
            );
        } else {
            assert_eq!(h.memory(wife).sentence, WIFE, "the overlap survives");
            assert!(h.text(said).contains("flowers"));
        }
    }
}

// Restatements: the sentence of a claim absorbed into a memory as a
// repeat. Its text is the owner's, so it goes wherever the memory or the
// passage it was taken from goes.

#[test]
fn a_restatement_goes_with_its_memory_by_forget_or_bank_deletion() {
    for delete_bank in [false, true] {
        let h = Harness::new();
        let maya = h.said(EARLIER, notable(MAYA));
        h.ingest("chat", "As I said, my daughter is called Maya.");
        h.extract_mention(quoting(notable(MAYA), "my daughter is called Maya"), maya);
        assert_eq!(h.memory(maya).restatements.len(), 1, "{delete_bank}");

        if delete_bank {
            h.service.delete_bank(BANK, BANK).unwrap();
        } else {
            h.forget(&[maya]);
            assert!(h.erase_next().is_some());
        }
        assert_eq!(h.restatement_rows(), 0, "delete_bank: {delete_bank}");
    }
}

#[test]
fn forgetting_a_memory_takes_restatements_on_others_that_overlap_its_passage() {
    // One turn restates Maya and Tea and says something new whose passage
    // overlaps Maya's restatement. Forgetting the new memory masks its
    // passage, so the restatement taken from that text goes, even though
    // it's on a memory that stays. Tea's, elsewhere in the turn, stays.
    let (h, maya, tea) = maya_and_tea();
    h.ingest(
        "chat",
        "My daughter Maya and I moved to Lisbon. I still like green tea.",
    );
    let extracted = h.extract_with(
        vec![
            quoting(notable(MAYA), "My daughter Maya"),
            quoting(notable(LISBON_WITH_MAYA), "Maya and I moved to Lisbon"),
            quoting(notable(TEA), "I still like green tea"),
        ],
        &[(0, maya, "mentioned_again"), (2, tea, "mentioned_again")],
    );
    let [lisbon] = extracted.memories[..] else {
        panic!("one new memory: {:?}", extracted.memories);
    };
    assert_eq!(h.memory(maya).restatements.len(), 1);
    assert_eq!(h.memory(tea).restatements.len(), 1);

    h.forget(&[lisbon]);
    assert!(h.erase_next().is_some());
    assert!(h.memory(maya).restatements.is_empty());
    assert_eq!(h.memory(tea).restatements.len(), 1);
    assert_eq!(h.restatement_rows(), 1, "only Tea's is left");
}

#[test]
fn removing_a_document_takes_its_restatements_but_not_the_memory() {
    // The memory came from a turn and the document only said it again, so
    // removing the document leaves the memory but not the sentence taken
    // from the document.
    let h = Harness::new();
    let tea = h.said(EARLIER, notable(TEA));
    h.doc("notes.md", "# Drinks\n\nI like green tea.\n");
    h.extract_mention(quoting(notable(TEA), "I like green tea"), tea);
    assert_eq!(h.memory(tea).restatements.len(), 1);

    h.service.remove_document(BANK, "notes.md").unwrap();
    assert!(h.memory(tea).restatements.is_empty());
    assert_eq!(h.restatement_rows(), 0);
}

// Purge in the nightly sweep

#[test]
fn the_sweep_purges_a_faded_chain_at_four_bank_local_without_redacting() {
    let (h, [berlin, maya, mia, tea]) = Harness::with_past(|h| {
        // Berlin was said before the chain that ends it, and is notable.
        let berlin = h.said("2020-06-01T00:00:00Z", notable(BERLIN));
        // Maya rests on a turn a lasting memory rests on too, so the
        // purge doesn't free it.
        let both = "My daughter is called Maya. I grew up in Dunedin.";
        h.said_turn(LONG_AGO, both);
        let claims = vec![
            quoting(trivial(MAYA), "My daughter is called Maya"),
            quoting(notable("Tim grew up in Dunedin."), "I grew up in Dunedin"),
        ];
        let maya = h.extract_with(claims, &[]).memories[0];
        let mia = h.said_changing(
            LONG_AGO,
            changes(trivial(MIA)),
            &[(maya, "retracts"), (berlin, "ends")],
        );
        [berlin, maya, mia, h.said(EARLIER, notable(TEA))]
    });
    assert_eq!(h.superseded_by(maya), Some(mia));
    assert_eq!(h.memory(berlin).chain.ended_by, Some(mia));
    let said = h.memory(maya).source.source;

    h.set(at(SWEEP) - minutes(1));
    let early = h.sweep();
    assert!(early.ran.is_empty());
    assert_eq!(early.next_due, Some(at(SWEEP)));
    assert_eq!(h.exist(&[maya, mia]), [true, true]);

    h.set(at(SWEEP));
    let swept = h.sweep();
    assert_eq!(swept.ran.len(), 1);
    let run = &swept.ran[0];
    assert_eq!(run.bank, BANK);
    assert_eq!(run.purged_memories, 2);
    assert_eq!(run.fingerprint, h.tuning.deletion_fingerprint());
    assert_eq!(run.delta, Some(1.0));
    let exist = h.exist(&[maya, mia, berlin, tea]);
    assert_eq!(exist, [false, false, true, true]);
    assert_eq!(h.memory(berlin).chain.ended_by, None);

    // Purge never redacts: the source still holds the passage.
    assert!(h.text(said).contains("My daughter is called Maya"));

    // The audit lists and status read it back: ids and counts, never
    // content, and only the bank's own.
    h.create_bank("other");
    let purges = h.purge_rows(BANK);
    assert_eq!(purges.len(), 1, "{purges:?}");
    let listed: BTreeSet<Uuid> = purges[0].memories.iter().copied().collect();
    assert_eq!(listed, BTreeSet::from([maya, mia]));
    let shown = serde_json::to_string(&purges).unwrap();
    assert!(!shown.contains("Maya") && !shown.contains("Mia"));
    assert!(h.purge_rows("other").is_empty());
    let last = h.service.status().unwrap().last_sweep.unwrap();
    assert_eq!((last.bank.as_str(), last.purged_memories), (BANK, 2));

    // A second night: one run row per bank per night, newest first, and a
    // limit keeps the newest.
    h.set(at(SWEEP) + days(1));
    h.sweep();
    let sweeps = h.sweep_rows(BANK, None);
    let purged: Vec<u64> = sweeps.iter().map(|run| run.purged_memories).collect();
    assert_eq!(purged, [0, 2], "{sweeps:?}");
    let newest = h.sweep_rows(BANK, Some(1));
    assert_eq!(newest.len(), 1);
    assert_eq!(newest[0].purged_memories, 0);
    let other = h.sweep_rows("other", None);
    assert!(other.iter().all(|run| run.bank == "other"), "{other:?}");
}

#[test]
fn a_model_citing_a_purged_memory_refreshes_once_that_night() {
    // The sweep purges before the night's refresh, so a model whose answer
    // cited a purged memory is blanked and refreshes once, without it.
    let h = Harness::new();
    h.said(EARLIER, notable(TEA));
    let (maya, _) = h.says(trivial(MAYA));
    h.profile_citing("Tim has a daughter called Maya.", &[maya]);

    h.set(at(LATER_SWEEP));
    h.sweep();
    assert_eq!(h.exist(&[maya]), [false]);
    let profile = h.profile();
    assert_eq!(profile.answer, None);
    assert!(profile.cites.is_empty());
    // Tea is the one memory left to list, as `m1`.
    let llm = FakeLlm::scripted(MODEL, vec![written(TEA, &["m1".into()]); 3]);
    h.service.run_refreshes(&llm).unwrap();
    for later in [
        minutes(31),
        SignedDuration::from_hours(12),
        days(1) - minutes(1),
    ] {
        h.set(at(LATER_SWEEP) + later);
        h.sweep();
        h.service.run_refreshes(&llm).unwrap();
    }
    let requests = llm.requests().into_iter();
    let refreshes: Vec<_> = requests
        .filter(|request| request.template.name == WRITE_TEMPLATE)
        .collect();
    assert_eq!(refreshes.len(), 1, "one refresh between two sweeps");
    assert!(refreshes[0].user.contains(TEA));
    assert!(!refreshes[0].user.contains("Maya"));
}

#[test]
fn a_date_still_ahead_or_a_task_overdue_under_thirty_days_holds_a_faded_chain_back() {
    // On the head, a retracted predecessor's old slot keeps nothing alive.
    // The agenda's 30-day overdue window is the task guard; undated tasks
    // have none.
    let (h, [concert, old_slot, new_slot, passport, tax, bike]) = Harness::with_past(|h| {
        let concert = h.said(LONG_AGO, event(CONCERT, "2026-12-12"));
        let old_slot = h.said(LONG_AGO, event(DENTIST_9, "2026-12-09"));
        let new_slot = h.said_changing(
            LONG_AGO,
            changes(event(DENTIST_8, "2021-03-08")),
            &[(old_slot, "retracts")],
        );
        let passport = h.said(LONG_AGO, task(PASSPORT, Some("2026-09-20")));
        let tax = h.said(LONG_AGO, task(TAX, Some("2026-08-15")));
        let bike = h.said(LONG_AGO, task(BIKE, None));
        [concert, old_slot, new_slot, passport, tax, bike]
    });
    assert_eq!(h.superseded_by(old_slot), Some(new_slot));

    h.set(at(SWEEP));
    h.sweep();
    assert_eq!(
        h.exist(&[concert, passport, old_slot, new_slot, tax, bike]),
        [true, true, false, false, false, false]
    );

    // Held through the 30th day after the due day: 04:00 on 20 October in
    // Auckland, then on the 21st.
    h.set(at("2026-10-19T15:00:00Z"));
    h.sweep();
    assert_eq!(h.exist(&[passport]), [true]);
    h.set(at("2026-10-20T15:00:00Z"));
    h.sweep();
    assert_eq!(h.exist(&[passport, concert]), [false, true]);
}

// The source, failed-chunk and recall-log sweep, 90 days after ingest

#[test]
fn the_sweep_deletes_text_past_the_horizon_and_keeps_the_keys() {
    let h = Harness::new();
    let prefetch = |session: &str| {
        let request = PrefetchRequest {
            session_id: session.into(),
            query: "[Sam] what tea does Tim like?".into(),
            ..PrefetchRequest::default()
        };
        h.service.prefetch(BANK, &request).unwrap().recall_id
    };
    let (tea, kept) = h.says(notable(TEA));
    let idle = h.ingest("idle", "Good morning.");
    h.extract_nothing();
    // The failed turn is queued with tea in context, from a recall in its
    // session.
    let in_failed = RecallRequest {
        session_id: Some("failed".into()),
        ..query("what tea does Tim like?")
    };
    let old_recall = h.recall(in_failed).recall_id;
    let failed = h.ingest("failed", "Good afternoon.");
    let in_context = format!(
        "SELECT count(*) FROM turn_in_context t JOIN sources s ON s.id = t.source_id
         WHERE s.uuid = '{failed}'"
    );
    assert_eq!(h.count(&in_context), 1);
    let failed_chunk = h.fail_head();
    // Documents queue behind turns, so this one is still waiting.
    let pending = h.doc("evening.md", "# Evening\n\nGood evening.\n").source;
    let old_prefetch = prefetch("chat");

    h.set(at(START) + days(2));
    let young = h.ingest("young", "Good night.");
    h.extract_nothing();
    let young_prefetch = prefetch("chat");
    let before: Vec<RecallRow> = h.recall_rows();

    h.set(at(PAST_HORIZON));
    let swept = h.sweep();
    let run = &swept.ran[0];
    assert_eq!(run.purged_memories, 0);
    assert_eq!(run.swept_sources, 2, "the idle turn and the failed one");
    assert_eq!(run.swept_failed_chunks, 1);
    assert_eq!(run.swept_recalls, 2);

    for gone in [idle, failed] {
        let detail = h.service.show_source(BANK, &gone.to_string()).unwrap();
        assert_eq!((detail.text, detail.reply), (None, None), "{gone}");
        assert!(matches!(detail.gone, Some(Gone::Swept { .. })), "{gone}");
        assert_eq!(h.chunk_text(gone), "", "{gone}");
    }
    assert_eq!(h.count(&in_context), 0);
    let retried = h.service.retry_chunks(BANK, Some(&[failed_chunk])).unwrap();
    assert!(retried.retried.is_empty());

    // The key stays, so ingest stays idempotent.
    let resent = turn("idle", h.message_at(idle), "Good morning.");
    let again = h.service.ingest_turn(BANK, &resent).unwrap();
    assert_eq!(again.outcome, Outcome::Duplicate);

    // Kept: what a memory rests on, what's younger than 90 days by
    // `ingested_at`, and what's still waiting to be extracted.
    assert!(h.memory(tea).source.passage.is_some());
    assert_eq!(h.text(kept), TEA);
    assert_eq!(h.text(young), "Good night.");
    assert!(h.text(pending).contains("Good evening."));
    assert_eq!(h.service.queue_depth(BANK).unwrap(), 1);

    // A recall row past the horizon keeps its key as the tombstone: the row
    // and its id stay, marked swept, and the queries and results go.
    let rows = h.recall_rows();
    let row = |id: Uuid| rows.iter().find(|row| row.id == id).unwrap();
    for old in [old_recall, old_prefetch] {
        let row = row(old);
        assert_eq!((&row.query, &row.raw_query), (&None, &None));
        assert_eq!(row.swept_at, Some(at(PAST_HORIZON)));
        assert!(row.results.is_empty());
    }
    let young_row = row(young_prefetch);
    assert_eq!(young_row.query.as_deref(), Some("what tea does Tim like?"));
    let raw = young_row.raw_query.as_deref();
    assert_eq!(raw, Some("[Sam] what tea does Tim like?"));
    assert_eq!(young_row.swept_at, None);
    let young_before = before.iter().find(|r| r.id == young_prefetch);
    assert_eq!(Some(young_row), young_before);
}

// The deletion fingerprint, the plan and the ack

#[test]
fn a_changed_fingerprint_pauses_purge_until_the_running_hash_is_acked() {
    let (h, [maya, tea]) = Harness::with_past(|h| {
        [
            h.said(LONG_AGO, trivial(MAYA)),
            h.said(EARLIER, notable(TEA)),
        ]
    });
    assert_eq!(h.service.purge_pause(), PurgePause::Running);
    let stored = h.tuning.deletion_fingerprint();

    // A change to tuning purge doesn't read leaves it running.
    let h = h.restart_with("[injection]\ncap = 3\n[recall]\nstrong_cutoff = 0.5\n");
    assert_eq!(h.service.purge_pause(), PurgePause::Running);

    let h = h.restart_with("[purge]\ndelta = 0.5\n");
    let current = h.tuning.deletion_fingerprint();
    assert_ne!(current, stored);
    let paused = PurgePause::Paused {
        stored: stored.clone(),
    };
    assert_eq!(h.service.purge_pause(), paused);

    // The sweep waits...
    h.set(at(SWEEP));
    h.sweep();
    assert_eq!(h.exist(&[maya]), [true]);

    // ...but forget never does.
    h.forget(&[tea]);
    assert!(h.erase_next().is_some());
    assert_eq!(h.exist(&[tea]), [false]);

    // The plan shows what changed and what would go, and deletes nothing.
    let plan = h.service.purge_plan().unwrap();
    assert_eq!(plan.current, current);
    assert_eq!(plan.changed, vec!["purge.delta".to_string()]);
    assert_eq!(plan.memories, 1);
    assert_eq!(h.exist(&[maya]), [true]);

    // Only the hash the running daemon computed is accepted.
    for wrong in [stored.as_str(), "not-a-hash"] {
        let refused = h.service.purge_ack(wrong);
        assert!(matches!(refused, Err(PurgeError::HashMismatch)));
    }
    assert_eq!(h.service.purge_pause(), paused);
    h.service.purge_ack(current.as_str()).unwrap();
    assert_eq!(h.service.purge_pause(), PurgePause::Running);

    // Purging resumes at the next sweep.
    h.set(at(SWEEP) + days(1));
    h.sweep();
    assert_eq!(h.exist(&[maya]), [false]);

    // The ack survives a restart.
    let h = h.restart_with("[purge]\ndelta = 0.5\n");
    assert_eq!(h.service.purge_pause(), PurgePause::Running);
}

#[test]
fn an_entity_a_model_filter_names_survives_the_erase() {
    let h = Harness::new();
    let claim = naming(notable(TEA), &[("Teahouse", "place"), ("Kettle", "thing")]);
    let tea = h.said(EARLIER, claim);
    let spec = ModelSpec {
        name: "Tea".into(),
        question: "Where does Tim drink tea?".into(),
        kinds: vec![],
        entity: Some("Teahouse".into()),
        min_volatility: None,
        max_tokens: 200,
        enabled: true,
    };
    h.service.create_model(BANK, &spec).unwrap();
    h.forget(&[tea]);
    h.erase_next().unwrap();
    let names = h.entity_names();
    assert!(names.contains("Teahouse"));
    assert!(!names.contains("Kettle"));
}

// Migration regressions

#[test]
fn a_mention_is_redacted_by_its_span_when_queued_or_stored_by_version_7() {
    // A mention queued before the forget is redacted by its span. Mention
    // spans move out of `accesses` in a new migration, so a store that
    // recorded spans under version 7 has to keep redacting them once it's
    // upgraded.
    for upgraded in [false, true] {
        let h = Harness::new();
        let (maya, _) = h.says(notable(MAYA));
        let mention = h.ingest("chat", "As I said, my daughter is called Maya. Anyway.");
        if !upgraded {
            h.forget(&[maya]);
        }
        h.extract_mention(quoting(notable(MAYA), "my daughter is called Maya"), maya);
        let h = if upgraded {
            // Version 7 wrote the span on the access, characters 11 to 37 of
            // the turn's chunk, and had nothing a later version adds for
            // spans.
            h.sql(&format!(
                "UPDATE accesses SET spans = json_array(json_array(
                   (SELECT c.id FROM chunks c JOIN sources s ON s.id = c.source_id
                    WHERE s.uuid = '{mention}'), 11, 37))
                 WHERE kind = 'mentioned_again'
                   AND memory_id = (SELECT id FROM memories WHERE uuid = '{maya}');
                 DROP TABLE IF EXISTS mention_passages;
                 DELETE FROM migrations WHERE to_version > 7;
                 PRAGMA user_version = 7;"
            ));
            let h = h.restart_with("");
            h.forget(&[maya]);
            h
        } else {
            h
        };
        assert!(h.erase_next().is_some());
        let masked = format!("As I said, {}. Anyway.", "\u{2588}".repeat(26));
        assert_eq!(h.text(mention), masked, "upgraded: {upgraded}");
        assert_eq!(h.count("SELECT count(*) FROM accesses"), 0);
    }
}

#[test]
fn purge_rechecks_each_chain_inside_its_own_transaction() {
    let (h, [kept, refined, forgotten]) = Harness::with_past(|h| {
        [TEA, BERLIN, MAYA].map(|content| h.said(LONG_AGO, trivial(content)))
    });
    h.set(at(SWEEP));
    let candidates = h.service.purge_candidates().unwrap();
    let heads: BTreeSet<Uuid> = candidates.iter().map(|(_, head)| *head).collect();
    assert_eq!(heads, BTreeSet::from([kept, refined, forgotten]));

    // Between choosing and deleting: a keep, a refinement whose new head
    // is strong, and a forget.
    h.service.keep(BANK, &[kept.to_string()]).unwrap();
    let (moved, _) = h.says_changing(changes(notable(MOVED)), refined, "refines");
    h.forget(&[forgotten]);

    for (bank, head) in &candidates {
        assert_eq!(h.service.purge_chain(bank, *head).unwrap(), None, "{head}");
    }
    let exist = h.exist(&[kept, refined, moved, forgotten]);
    assert_eq!(exist, [true, true, true, true]);
    assert!(h.purge_rows(BANK).is_empty());

    // The forget's own erase still finds its chain.
    let erased = h.erase_next().expect("the erase");
    assert_eq!(erased.memories, BTreeSet::from([forgotten]));
}

#[test]
fn a_reworded_section_of_the_same_document_queued_before_a_forget_is_redacted() {
    // Not crediting a later version of the same document is right;
    // dropping where it restated the memory isn't. Rewording tests
    // passage preservation separately from exact-text matching.
    let h = Harness::new();
    h.doc("family.md", "# Family\n\nMy daughter is called Maya.\n");
    let maya = h.extract_one(quoting(notable(MAYA), "My daughter is called Maya"), &[]);
    let v2 = h.doc("family.md", "# Family\n\nMaya, my daughter, is six now.\n");
    h.forget(&[maya]);
    h.extract_mention(quoting(notable(MAYA), "Maya, my daughter"), maya);

    assert!(h.erase_next().is_some());
    let (text, chunk) = (h.text(v2.source), h.chunk_text(v2.source));
    assert!(!text.contains("Maya") && !chunk.contains("Maya"), "{text}");
    assert!(text.contains("is six now."), "{text}");
}

/// A refresh LLM that forgets `memory` while its call is in flight, and
/// with `erase` runs the erase too, then answers `reply`.
struct ForgetsDuringCall<'a> {
    service: &'a Service,
    memory: Uuid,
    erase: bool,
    reply: Value,
}

impl LlmClient for ForgetsDuringCall<'_> {
    fn model(&self) -> &str {
        MODEL
    }

    fn complete(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        let service = self.service;
        service.forget(BANK, &[self.memory.to_string()]).unwrap();
        if self.erase {
            assert!(service.erase_next(BANK).unwrap().is_some());
        }
        Ok(LlmResponse {
            json: self.reply.clone(),
            usage: None,
            latency: Duration::ZERO,
        })
    }
}

#[test]
fn a_refresh_in_flight_stores_no_answer_citing_a_memory_hidden_or_erased_meanwhile() {
    // The answer rests on every memory it cites, so it isn't stored when
    // one of them was forgotten while the write was being made.
    for erase in [false, true] {
        let (h, maya, tea) = maya_and_tea();
        let input = h.service.refresh_input(BANK, PROFILE_NAME).unwrap();
        let llm = ForgetsDuringCall {
            service: &h.service,
            memory: maya,
            erase,
            reply: written(
                "Tim has a daughter called Maya. Tim likes green tea.",
                &[handle(&input, maya), handle(&input, tea)],
            ),
        };
        let outcome = h.service.refresh_model(BANK, PROFILE_NAME, &llm, true);
        assert!(outcome.is_ok(), "{outcome:?}");
        let profile = h.profile();
        assert_eq!(profile.answer, None, "erase: {erase}");
        assert!(profile.cites.is_empty(), "erase: {erase}");
    }
}

#[test]
fn a_sweep_that_fails_after_a_purge_settles_it_and_runs_again() {
    // Run again at once on the same connection, or the next night after a
    // restart: the run row counts what the failed attempt deleted, and
    // says when that attempt started.
    for restart in [false, true] {
        let h = Harness::new();
        h.said(EARLIER, notable(TEA));
        let (maya, _) = h.says(trivial(MAYA));
        h.profile_citing("Tim has a daughter called Maya.", &[maya]);
        h.set(at(LATER_SWEEP));
        let before = h.service.system_prompt(BANK, None).unwrap();
        // The store's own connection: a temporary trigger fails the run row,
        // after the purge has committed.
        h.sql(
            "CREATE TEMP TRIGGER injected_failure BEFORE INSERT ON sweep_runs
             BEGIN SELECT RAISE(ABORT, 'injected'); END",
        );
        assert!(h.service.run_sweeps().is_err());
        assert_eq!(h.exist(&[maya]), [false], "the chain committed");
        // The block of the model whose answer went was cleared.
        let after = h.service.system_prompt(BANK, None).unwrap();
        assert_ne!(after.id, before.id);

        let h = if restart {
            // The temporary trigger goes with the old connection. The
            // restarted daemon's first sweep is the next 04:00.
            let h = h.restart_with("");
            h.set(at(LATER_SWEEP) + days(1));
            h
        } else {
            h.sql("DROP TRIGGER temp.injected_failure");
            h
        };
        let again = h.service.run_sweeps().unwrap();
        assert_eq!(again.ran.len(), 1, "the failed night is still due");
        assert_eq!(again.ran[0].purged_memories, 1);
        let rows = h.sweep_rows(BANK, None);
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].purged_memories, 1);
        assert_eq!(rows[0].started_at, at(LATER_SWEEP));
    }
}

// Ready erases run without a worker. The daemon half is
// `a_ready_erase_runs_after_a_restart_without_an_llm` in serve_http.

#[test]
fn ready_erases_run_without_a_worker() {
    let h = Harness::new();
    let (maya, said) = h.says(notable(MAYA));
    h.ingest("chat", "Something else entirely.");
    h.forget(&[maya]);
    // The erase waits behind the queued chunk, until it fails for good with
    // no LLM to extract it.
    assert!(h.service.run_erases().unwrap().is_empty());
    h.fail_head();
    assert_eq!(h.service.run_erases().unwrap().len(), 1);
    assert_eq!(h.exist(&[maya]), [false]);
    assert!(!h.text(said).contains("Maya"));
}

#[test]
fn the_plan_counts_a_source_the_sweep_frees_by_purging() {
    // The only memory on its source.
    let (h, _) = Harness::with_past(|h| h.said(LONG_AGO, trivial(MAYA)));
    h.set(at(PAST_HORIZON));
    let plan = h.service.purge_plan().unwrap();
    let swept = h.sweep();
    let run = &swept.ran[0];
    assert_eq!((run.purged_memories, run.swept_sources), (1, 1));
    let planned = (plan.memories, plan.sources);
    assert_eq!(planned, (run.purged_memories, run.swept_sources));
}

#[test]
fn every_version_of_a_document_loses_a_forgotten_passage() {
    let h = Harness::new();
    let family = "# Family\n\nMy daughter is called Maya.\n";
    let v1 = h.doc("family.md", family).source;
    let maya = h.extract_one(quoting(notable(MAYA), "My daughter is called Maya"), &[]);
    // v2 adds a section; the Family section is skipped as already seen.
    let garden = format!("{family}\n# Garden\n\nThe roses are out.\n");
    let v2 = h.doc("family.md", &garden);
    assert_eq!(v2.chunks_skipped, 1);
    h.extract_nothing();

    h.forget(&[maya]);
    assert!(h.erase_next().is_some());
    assert!(!h.text(v1).contains("Maya"), "the memory's own version");
    let text = h.text(v2.source);
    assert!(
        !text.contains("Maya"),
        "the version that skipped it: {text}"
    );
    assert!(text.contains("The roses are out."), "{text}");

    // A version sent after the forget doesn't store the passage again, or
    // queue it.
    let kitchen = format!("{garden}\n# Kitchen\n\nThe kettle is new.\n");
    let v3 = h.doc("family.md", &kitchen);
    assert_eq!(v3.chunks_skipped, 2);
    let text = h.text(v3.source);
    assert!(!text.contains("Maya"), "{text}");
    assert!(text.contains("The kettle is new."), "{text}");
}

// A forget is linked to the turn that asked for it. Each test runs at one
// instant on the stopped clock, so only the order of writes can tell the
// turns apart.

#[test]
fn every_forget_in_the_turn_is_linked_and_other_sessions_link_none() {
    let (h, maya, tea) = maya_and_tea();
    h.forget_in(Some("s"), &[maya]);
    h.forget_in(Some("s"), &[tea]);

    h.forget_turn("other", "Forget something else.");
    assert!(h.forget_rows().iter().all(|row| row.request.is_null()));

    let request = h.forget_turn("s", "Forget those two.");
    for row in h.forget_rows() {
        assert_eq!(row.request, json!(request.source));
    }
}

#[test]
fn a_forget_whose_request_turn_never_arrived_stays_unlinked() {
    // F1's request turn was lost; an ordinary turn followed, then F2 and
    // its own request turn, all at the same instant.
    let (h, maya, tea) = maya_and_tea();
    h.forget_in(Some("s"), &[maya]);
    h.ingest("s", "What's for dinner?");
    h.forget_in(Some("s"), &[tea]);
    let request = h.forget_turn("s", "Forget the tea.");

    let rows = h.forget_rows();
    assert_eq!(rows[0].memories, [maya]);
    assert_eq!(rows[0].request, Value::Null);
    assert_eq!(rows[1].request, json!(request.source));
}

#[test]
fn a_forget_without_a_session_or_after_a_resent_turn_links_nothing() {
    let (h, maya, tea) = maya_and_tea();
    // The CLI sends no session.
    h.forget_in(None, &[maya]);
    let first = h.forget_turn("s", "Forget that.");
    assert_eq!(first.outcome, Outcome::Tombstone);
    assert_eq!(h.forget_rows()[0].request, Value::Null);

    // The plugin's spool sends the same request turn again after a later
    // forget: a duplicate links nothing.
    h.forget_in(Some("s"), &[tea]);
    let again = h.forget_turn("s", "Forget that.");
    assert_eq!(again.outcome, Outcome::Duplicate);
    assert_eq!(h.forget_rows()[1].request, Value::Null);
}

// Mentions stored before version 7 have no span. The erase masks more
// rather than less, and keeps what surviving memories rest on.

#[test]
fn a_legacy_mention_in_a_turn_masks_the_turn_but_what_survives() {
    let h = Harness::new();
    let (maya, _) = h.says(notable(MAYA));
    let mixed = h.ingest("chat", "My daughter is called Maya. I like green tea.");
    let claims = vec![
        quoting(notable(MAYA), "My daughter is called Maya"),
        quoting(notable(TEA), "I like green tea"),
    ];
    let tea = h
        .extract_with(claims, &[(0, maya, "mentioned_again")])
        .memories[0];
    h.make_legacy();

    h.forget(&[maya]);
    assert!(h.erase_next().is_some());
    let detail = h.service.show_source(BANK, &mixed.to_string()).unwrap();
    let text = detail.text.unwrap();
    assert!(!text.contains("Maya"), "{text}");
    assert!(!text.contains("daughter"), "{text}");
    assert!(text.contains("I like green tea"), "{text}");
    assert!(!detail.reply.unwrap_or_default().contains("Noted"));
    assert_eq!(h.exist(&[tea]), [true]);
}

#[test]
fn a_legacy_mention_in_a_document_masks_an_exact_passage_or_all_but_what_survives() {
    let h = Harness::new();
    // The forgotten memory's own passage is its sentence.
    let (maya, _) = h.says(notable(MAYA));
    let exact_text = format!("# Notes\n\n{MAYA} The garden needs water.\n");
    let exact = h.doc("exact.md", &exact_text).source;
    h.extract_mention(quoting(notable(MAYA), MAYA), maya);
    let loose_text = "# Plans\n\nMaya is my daughter. The garden needs water.\n";
    let loose = h.doc("loose.md", loose_text).source;
    let claims = vec![
        quoting(notable(MAYA), "Maya is my daughter"),
        quoting(notable(GARDEN), "The garden needs water"),
    ];
    let garden = h
        .extract_with(claims, &[(0, maya, "mentioned_again")])
        .memories[0];
    h.make_legacy();

    h.forget(&[maya]);
    assert!(h.erase_next().is_some());
    // With the exact passage there, only it goes.
    let text = h.text(exact);
    assert!(!text.contains("Maya"), "{text}");
    assert!(text.contains("# Notes") && text.contains("The garden needs water."));
    // Without it, everything but what a surviving memory rests on goes.
    let text = h.text(loose);
    assert!(!text.contains("Maya") && !text.contains("Plans"), "{text}");
    assert!(text.contains("The garden needs water"), "{text}");
    assert_eq!(h.exist(&[garden]), [true]);
}

// Overlapping passages across versions. Masking one passage in a version
// mustn't stop a longer one that contains it being found there, in one
// erase or in a later one.

const DIAGNOSIS: &str = "Tim has a diagnosis.";
const HIV: &str = "Tim has a diagnosis. It is HIV.";
const HEALTH_V2: &str = "# Health\n\nTim has a diagnosis. It is HIV.\n";

#[test]
fn overlapping_passages_leave_no_version_unmasked_in_one_erase_or_two() {
    // v1 says DIAGNOSIS and is extracted; v2 says HIV, refining it into one
    // chain or as a separate memory forgotten later. v3 shares v2's Health
    // section, so it has no chunk of its own for it, and adds a Garden
    // section with nothing in it.
    for one_chain in [true, false] {
        let h = Harness::new();
        h.doc("health.md", "# Health\n\nTim has a diagnosis.\n");
        let diagnosis = h.extract_one(quoting(notable(DIAGNOSIS), DIAGNOSIS), &[]);
        let v2 = h.doc("health.md", HEALTH_V2).source;
        let hiv_claim = quoting(notable(HIV), HIV);
        let labels = if one_chain {
            &[(diagnosis, "refines")][..]
        } else {
            &[]
        };
        let hiv = h.extract_one(hiv_claim, labels);
        assert_eq!(h.superseded_by(diagnosis), one_chain.then_some(hiv));
        let garden = format!("{HEALTH_V2}\n# Garden\n\nThe roses are out.\n");
        let v3 = h.doc("health.md", &garden);
        assert_eq!(v3.chunks_skipped, 1);
        h.extract_nothing();

        h.forget(&[diagnosis]);
        assert!(h.erase_next().is_some());
        if !one_chain {
            h.forget(&[hiv]);
            assert!(h.erase_next().is_some());
        }
        for source in [v2, v3.source] {
            let text = h.text(source);
            assert!(!text.contains("HIV"), "{one_chain}: {text}");
            assert!(!text.contains("diagnosis"), "{one_chain}: {text}");
        }
        assert!(h.text(v3.source).contains("The roses are out."));
    }
}
