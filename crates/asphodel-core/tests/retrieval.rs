//! Hybrid recall, reranking, injection gates and recall logs. Being
//! recalled or injected is logged but never strengthens a memory.
//!
//! The API under test is `asphodel_core::retrieval` and the `Service`
//! methods over it: `prefetch`, `recall`, `in_context`, `clear_session` and
//! `ingest_turn`'s settling of the pending injection. Every memory comes
//! from extraction: the owner says it in a turn and call 1 (a scripted
//! `FakeLlm`) finds the claim, indexed with the fake embedder.
//!
//! `FakeReranker`'s logit is the number of distinct query words a memory
//! shares, minus one half, so a gate floor of 1.0 lets through a memory
//! that shares two words and stops one that shares one.
//!
//! Every service here runs on a `SimulatedClock` stopped at 20:00 on
//! Thursday 1 October 2026 in Auckland unless a test advances it.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant};

use asphodel_core::config::RerankQuery;
use asphodel_core::constants::{
    CANDIDATES_PER_ARM, RECALL_LIMIT_DEFAULT, RECALL_LIMIT_MAX, RERANKED,
};
use asphodel_core::entities::LinkRequest;
use asphodel_core::ingest::{Outcome, Turn};
use asphodel_core::models::{
    Embedder, FakeEmbedder, FakeLlm, FakeReranker, ModelError, Models, Reranker,
};
use asphodel_core::operations::{Audit, AuditList, RecallRow};
use asphodel_core::retrieval::{
    Arm, Band, Cut, Explain, ExplainInjection, ExplainMode, ExplainRecall, ExplainRequest,
    Explained, On, PhaseFilter, Prefetch, PrefetchRequest, Recall, RecallRequest, estimate_tokens,
};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{OpenOptions, Store};
use asphodel_core::strength::Kind;
use asphodel_core::{Service, SimulatedClock, Tuning};
use jiff::civil::DateTime;
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use serde_json::{Value, json};
use uuid::Uuid;

// Fixtures

/// 20:00 on Thursday 1 October 2026 in Auckland, on daylight time (UTC+13).
const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";
const BANK: &str = "main";

/// When seeded memories were said, unless a test says otherwise.
const EARLIER: &str = "2026-09-01T00:00:00Z";

/// Long enough ago that a trivial memory said then is below τ.
const LONG_AGO: &str = "2021-01-01T00:00:00Z";

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
}

/// A local date-time in `TZ` as the instant stored for it.
fn local(datetime: &str) -> Timestamp {
    let (datetime, tz): (DateTime, _) = (datetime.parse().unwrap(), TimeZone::get(TZ).unwrap());
    datetime.to_zoned(tz).unwrap().timestamp()
}

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("asphodel-retrieval-{}-{n}", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A reranker that answers like `FakeReranker`, only late.
struct SlowReranker(Duration);

impl Reranker for SlowReranker {
    fn model_id(&self) -> &str {
        FakeReranker::MODEL_ID
    }

    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, ModelError> {
        std::thread::sleep(self.0);
        FakeReranker.rerank(query, documents)
    }
}

/// A reranker that answers like `FakeReranker` once a test opens its gate,
/// signalling each call that reaches it and each answer, so a test can tell
/// what ran without timing it.
struct GatedReranker {
    open: Mutex<bool>,
    opened: Condvar,
    arrivals: mpsc::Sender<()>,
    answers: mpsc::Sender<()>,
}

impl GatedReranker {
    fn new() -> (Arc<Self>, mpsc::Receiver<()>, mpsc::Receiver<()>) {
        let (arrivals, arrived) = mpsc::channel();
        let (answers, answered) = mpsc::channel();
        let reranker = Arc::new(Self {
            open: Mutex::new(false),
            opened: Condvar::new(),
            arrivals,
            answers,
        });
        (reranker, arrived, answered)
    }

    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.opened.notify_all();
    }
}

impl Reranker for GatedReranker {
    fn model_id(&self) -> &str {
        FakeReranker::MODEL_ID
    }

    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, ModelError> {
        let _ = self.arrivals.send(());
        let open = self.open.lock().unwrap();
        drop(self.opened.wait_while(open, |open| !*open));
        let answer = FakeReranker.rerank(query, documents);
        let _ = self.answers.send(());
        answer
    }
}

/// A reranker that scores 5 for a document containing its keyword and −1
/// for any other, so a candidate holding it comes first if it's a candidate
/// at all.
struct KeywordReranker(&'static str);

impl Reranker for KeywordReranker {
    fn model_id(&self) -> &str {
        FakeReranker::MODEL_ID
    }

    fn rerank(&self, _query: &str, documents: &[&str]) -> Result<Vec<f32>, ModelError> {
        Ok(documents
            .iter()
            .map(|d| if d.contains(self.0) { 5.0 } else { -1.0 })
            .collect())
    }
}

/// An embedder that answers like `FakeEmbedder`, but puts any text
/// mentioning "Filler" where it puts its question: memories a real model
/// would find close to the question though they share none of its words.
struct CrowdingEmbedder(&'static str);

impl Embedder for CrowdingEmbedder {
    fn model_id(&self) -> &str {
        FakeEmbedder::MODEL_ID
    }

    fn dimensions(&self) -> usize {
        FakeEmbedder.dimensions()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        let mut placed = texts.to_vec();
        for text in placed.iter_mut().filter(|t| t.contains("Filler")) {
            *text = self.0;
        }
        FakeEmbedder.embed(&placed)
    }
}

/// A reranker that scores like `FakeReranker`, but only on what reaches
/// the model: the query and the document as one pair of at most 512 tokens,
/// three of them special, the longer side truncated first and from its end,
/// as fastembed sets up the real one. Each CJK character is a token, as the
/// real tokenizer splits them; otherwise each run of letters and digits and
/// each other character is one. `real_reranker_truncates_a_long_query_from_its_end`
/// in `tests/models.rs` checks the real one behaves this way.
struct TruncatingReranker;

impl TruncatingReranker {
    const MAX_TOKENS: usize = 512 - 3;

    fn tokens(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut word = String::new();
        for c in text.chars() {
            let cjk = matches!(c, '\u{3040}'..='\u{30ff}' | '\u{3400}'..='\u{9fff}');
            if c.is_alphanumeric() && !cjk {
                word.extend(c.to_lowercase());
                continue;
            }
            if !word.is_empty() {
                tokens.push(std::mem::take(&mut word));
            }
            if !c.is_whitespace() {
                tokens.push(c.to_string());
            }
        }
        if !word.is_empty() {
            tokens.push(word);
        }
        tokens
    }

    /// How many of each side's tokens survive: the longer side is cut to
    /// fit, or both to half when even the shorter is more than half.
    fn kept(query: usize, document: usize) -> (usize, usize) {
        let max = Self::MAX_TOKENS;
        if query + document <= max {
            return (query, document);
        }
        let short = query.min(document);
        let (short, long) = if short > max / 2 {
            (max / 2, max / 2 + max % 2)
        } else {
            (short, max - short)
        };
        if query <= document {
            (short, long)
        } else {
            (long, short)
        }
    }
}

impl Reranker for TruncatingReranker {
    fn model_id(&self) -> &str {
        FakeReranker::MODEL_ID
    }

    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, ModelError> {
        let query = Self::tokens(query);
        Ok(documents
            .iter()
            .map(|document| {
                let document = Self::tokens(document);
                let (q, d) = Self::kept(query.len(), document.len());
                let seen: BTreeSet<&String> = query[..q].iter().collect();
                let shared: BTreeSet<&String> = document[..d]
                    .iter()
                    .filter(|token| token.chars().any(char::is_alphanumeric))
                    .filter(|token| seen.contains(token))
                    .collect();
                shared.len() as f32 - 0.5
            })
            .collect())
    }
}

struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    _dir: TestDir,
}

fn open(dir: &TestDir, clock: &Arc<SimulatedClock>, tuning: Tuning, models: Models) -> Service {
    let store = Store::open(&dir.0, OpenOptions::default(), clock.clone()).unwrap();
    Service::with_models(clock.clone(), store, tuning, models).unwrap()
}

impl Harness {
    /// Fake models and a gate floor of 1.0.
    fn new() -> Self {
        Self::with(|_| {})
    }

    fn with_floor(floor: f64) -> Self {
        Self::with(|t| set_floor(t, floor))
    }

    /// `tune` adjusts the tuning after the harness has set its own.
    fn with(tune: impl FnOnce(&mut Tuning)) -> Self {
        Self::build(Models::fake(), tune, Vec::new()).0
    }

    fn with_reranker(reranker: Arc<dyn Reranker>, tune: impl FnOnce(&mut Tuning)) -> Self {
        let embedder = Arc::new(FakeEmbedder);
        Self::build(Models { embedder, reranker }, tune, Vec::new()).0
    }

    /// A harness whose bank already holds `faded`, made trivial and said
    /// at [`LONG_AGO`], with the clock there: below τ by [`START`].
    fn with_faded(tune: impl FnOnce(&mut Tuning), faded: Vec<Value>) -> (Self, Vec<Uuid>) {
        Self::build(Models::fake(), tune, faded)
    }

    fn build(
        models: Models,
        tune: impl FnOnce(&mut Tuning),
        faded: Vec<Value>,
    ) -> (Self, Vec<Uuid>) {
        let mut tuning = Tuning::default();
        set_floor(&mut tuning, 1.0);
        set_scale(&mut tuning, 1.0);
        let floors = &mut tuning.reconcile.embedding_floors;
        floors.insert(FakeEmbedder::MODEL_ID.into(), 0.5);
        tuning.recall.strong_cutoff = 0.3;
        tune(&mut tuning);
        let dir = TestDir::new();
        let seeding = if faded.is_empty() { START } else { LONG_AGO };
        let clock = Arc::new(SimulatedClock::new(at(seeding)));
        let service = open(&dir, &clock, tuning, models);
        let identity = BankIdentity {
            owner_name: Some("Tim".into()),
            timezone: Some(TZ.into()),
            ..BankIdentity::default()
        };
        service.ensure_bank_with_models(BANK, &identity).unwrap();
        let h = Self {
            service,
            clock,
            _dir: dir,
        };
        let faded = if faded.is_empty() {
            Vec::new()
        } else {
            let trivial = faded.into_iter().map(|c| c.level("trivial")).collect();
            h.seed_all(at(LONG_AGO), trivial)
        };
        h.clock.set(at(START));
        (h, faded)
    }

    fn with_deadline(self, deadline: Duration) -> Self {
        let service = self.service.with_reranker_deadline(deadline);
        Self { service, ..self }
    }

    /// The daemon restarting: the service and its in-memory sessions go,
    /// the store and its extraction queue stay.
    fn restart(self) -> Self {
        let tuning = self.service.tuning().clone();
        let models = self.service.models().unwrap().clone();
        drop(self.service);
        let service = open(&self._dir, &self.clock, tuning, models);
        Self { service, ..self }
    }

    /// The owner says `claims` in one turn at `said`, with the clock where
    /// it is, and the turn is extracted: call 1 finds the claims, and call
    /// 2, if it runs, labels nothing. Returns the new memories in claim
    /// order.
    fn seed_all(&self, said: Timestamp, claims: Vec<Value>) -> Vec<Uuid> {
        self.extract(said, TZ, claims, None)
    }

    /// One memory said at [`EARLIER`] in a turn from timezone `tz`.
    fn seed_in(&self, tz: &str, claim: Value) -> Uuid {
        self.extract(at(EARLIER), tz, vec![claim], None)[0]
    }

    /// One memory said at [`EARLIER`].
    fn seed(&self, claim: Value) -> Uuid {
        self.seed_at(at(EARLIER), claim)
    }

    /// `claims` said in one turn at [`EARLIER`].
    fn seed_many<const N: usize>(&self, claims: [Value; N]) -> [Uuid; N] {
        let memories = self.seed_all(at(EARLIER), claims.into());
        memories.try_into().unwrap()
    }

    fn seed_at(&self, said: Timestamp, claim: Value) -> Uuid {
        self.seed_all(said, vec![claim])[0]
    }

    /// One memory said a minute ago.
    fn says(&self, claim: Value) -> Uuid {
        self.seed_at(self.service.now() - SignedDuration::from_mins(1), claim)
    }

    /// As [`Harness::seed`], with call 2 labelling the claim `label` on
    /// `neighbour`: `refines` or `ends`.
    fn seed_changing(&self, claim: Value, neighbour: Uuid, label: &str) -> Uuid {
        self.extract(at(EARLIER), TZ, vec![claim], Some((neighbour, label)))[0]
    }

    fn extract(
        &self,
        said: Timestamp,
        tz: &str,
        claims: Vec<Value>,
        label: Option<(Uuid, &str)>,
    ) -> Vec<Uuid> {
        static SESSIONS: AtomicU64 = AtomicU64::new(0);
        let session = format!("seed-{}", SESSIONS.fetch_add(1, Ordering::Relaxed));
        let quotes: Vec<&str> = claims
            .iter()
            .map(|c| c["quote"].as_str().unwrap())
            .collect();
        let turn = Turn {
            timezone: Some(tz.into()),
            ..turn(&session, said, &quotes.join(" "), None)
        };
        self.service.ingest_turn(BANK, &turn).unwrap();
        let count = claims.len();
        let call1 = json!({"claims": claims, "used_injected_ids": []});
        let call2 = match label {
            None => json!({"claims": []}),
            Some((neighbour, label)) => {
                let lease = self.service.claim_chunk(BANK).unwrap().expect("queued");
                let input = self.service.call2_input(&lease, &call1, &[]).unwrap();
                let input = input.expect("call 2 runs");
                let neighbour = input.neighbours.iter().find(|n| n.memory == neighbour);
                let neighbour = neighbour.expect("the memory is a neighbour");
                json!({"claims": [{
                    "claim": input.claims[0].handle,
                    "labels": [{"neighbour": neighbour.handle, "label": label}],
                }]})
            }
        };
        let llm = FakeLlm::scripted("fake-llm", vec![call1, call2]);
        let extracted = self.service.extract_next(BANK, &llm).unwrap();
        let extracted = extracted.expect("the turn was queued");
        assert_eq!(extracted.memories.len(), count, "{:?}", extracted.dropped);
        extracted.memories
    }

    /// `count` memories that [`CrowdingEmbedder`] puts where it puts its
    /// question, so they fill the vector arm.
    fn crowd(&self, count: usize) {
        let filler = (0..count).map(|n| fact(&format!("Filler entry {n}.")));
        self.seed_all(at(EARLIER), filler.collect());
    }

    fn prefetch(&self, session: &str, query: &str) -> Prefetch {
        self.converse(session, query, None, None)
    }

    /// A prefetch sent with the previous message and the start of the
    /// assistant's reply to it.
    fn converse(
        &self,
        session: &str,
        query: &str,
        previous: Option<&str>,
        reply: Option<&str>,
    ) -> Prefetch {
        let request = request(session, query, previous, reply);
        self.service.prefetch(BANK, &request).unwrap()
    }

    fn recall(&self, request: RecallRequest) -> Recall {
        self.service.recall(BANK, &request).unwrap()
    }

    /// The turn that follows a prefetch, echoing `recall_id`.
    fn sync_turn(&self, session: &str, recall_id: Option<String>) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let message_at = self.service.now() - SignedDuration::from_secs(60);
        let turn = turn(session, message_at, &format!("Message {n}."), recall_id);
        self.service.ingest_turn(BANK, &turn).unwrap();
    }

    /// The turn that follows `prefetch`, echoing its recall id.
    fn commit(&self, session: &str, prefetch: &Prefetch) {
        self.sync_turn(session, Some(prefetch.recall_id.to_string()));
    }

    fn in_context(&self, session: &str) -> Vec<Uuid> {
        self.service.in_context(BANK, session).unwrap()
    }

    /// The recall log, newest first.
    fn recalls(&self) -> Vec<RecallRow> {
        match self.service.audit(BANK, AuditList::Recalls, None).unwrap() {
            Audit::Recalls { recalls } => recalls,
            other => panic!("{other:?}"),
        }
    }

    fn recall_row(&self, recall_id: Uuid) -> RecallRow {
        let row = self.recalls().into_iter().find(|r| r.id == recall_id);
        row.expect("logged")
    }

    /// `memory`'s accesses of `kind`, or of every kind when `None`.
    fn accesses(&self, memory: Uuid, kind: Option<&str>) -> usize {
        let view = self.service.show_memory(BANK, &memory.to_string()).unwrap();
        let accesses = view.accesses.iter();
        accesses
            .filter(|a| kind.is_none_or(|k| a.kind == k))
            .count()
    }

    /// Extracts the head of the queue, as the bank's worker does, with a
    /// call 1 that makes no claims and judges `used` (in-context handles)
    /// used. The LLM is returned so a test can read what call 1 was shown.
    fn extract_next(&self, used: &[&str]) -> FakeLlm {
        let call1 = json!({"claims": [], "used_injected_ids": used});
        let llm = FakeLlm::scripted("fake-llm", vec![call1]);
        let extracted = self.service.extract_next(BANK, &llm).unwrap();
        assert!(extracted.is_some(), "nothing was queued");
        llm
    }
}

fn set_floor(tuning: &mut Tuning, floor: f64) {
    let floors = &mut tuning.injection.reranker_floors;
    floors.insert(FakeReranker::MODEL_ID.into(), floor);
}

fn set_scale(tuning: &mut Tuning, scale: f64) {
    let scales = &mut tuning.ranking.relevance_scales;
    scales.insert(FakeReranker::MODEL_ID.into(), scale);
}

fn turn(session: &str, message_at: Timestamp, user: &str, recall_id: Option<String>) -> Turn {
    Turn {
        session_id: session.into(),
        message_at,
        timezone: Some(TZ.into()),
        user_text: user.into(),
        assistant_text: "Noted.".into(),
        author: None,
        platform: Some("cli".into()),
        recall_id,
        forget_requested: false,
    }
}

fn request(
    session: &str,
    query: &str,
    previous: Option<&str>,
    reply: Option<&str>,
) -> PrefetchRequest {
    PrefetchRequest {
        session_id: session.into(),
        query: query.into(),
        previous_query: previous.map(Into::into),
        previous_reply: reply.map(Into::into),
        block_id: None,
    }
}

/// Call 1's claim: the sentence is also the quote, as the owner said it.
fn claim(content: &str, kind: &str) -> Value {
    json!({
        "content": content,
        "kind": kind,
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

/// A notable fact.
fn fact(content: &str) -> Value {
    claim(content, "fact")
}

/// An event on the local date `on`, to the day.
fn event(content: &str, on: &str) -> Value {
    claim(content, "event").with("valid_from", day(on))
}

fn day(date: &str) -> Value {
    json!({"at": date, "precision": "day"})
}

/// A link to a new person entity named `name`, mentioned as `surface`.
fn person(name: &str, surface: &str) -> Value {
    json!([{"entity": null, "new_name": name, "new_kind": "person", "surface_form": surface}])
}

trait With {
    fn with(self, key: &str, value: Value) -> Value;

    fn level(self, significance: &str) -> Value
    where
        Self: Sized,
    {
        self.with("significance", json!(significance))
    }
}

impl With for Value {
    fn with(mut self, key: &str, value: Value) -> Value {
        self[key] = value;
        self
    }
}

fn query(text: &str) -> RecallRequest {
    RecallRequest {
        query: text.into(),
        ..RecallRequest::default()
    }
}

/// A recall in session `s`.
fn in_session(text: &str) -> RecallRequest {
    RecallRequest {
        session_id: Some("s".into()),
        ..query(text)
    }
}

fn ids(recall: &Recall) -> Vec<Uuid> {
    recall.results.iter().map(|r| r.id).collect()
}

// Short follow-ups

#[test]
fn a_short_follow_up_finds_what_the_previous_query_asked_about() {
    let h = Harness::with(|t| t.injection.rerank_query = RerankQuery::Conversation);
    let dentist = h.seed(fact("Tim's dentist appointment is on Friday."));
    assert!(h.prefetch("s", "yes, book it").injected.is_empty());
    let previous = "dentist appointment Friday";
    let followed = h.converse("s", "yes, book it", Some(previous), None);
    assert_eq!(followed.injected, vec![dentist]);

    // The previous message also reaches the reranker in conversation, once.
    let reply = Some("I can book it for Friday.");
    let request = request("t", "yes, book it", Some(previous), reply);
    let scored = h.service.scored_prefetch(BANK, &request).unwrap();
    assert_eq!(scored.prefetch.injected, vec![dentist]);
    let rerank_query = &scored.rerank_query;
    assert_eq!(rerank_query.matches(previous).count(), 1, "{rerank_query}");
}

// Reranking against the conversation

/// A message that doesn't name its subject finds the memory the previous
/// message, or else the assistant's reply, makes relevant. In message mode
/// the reranker sees only a message of eight words or more.
#[test]
fn the_reranker_sees_the_conversation_only_in_conversation_mode() {
    for (mode, sees) in [
        (RerankQuery::Conversation, true),
        (RerankQuery::Message, false),
    ] {
        let h = Harness::with(|t| t.injection.rerank_query = mode);
        let passkey = h.seed(fact("Tim signs in to Fastmail with a passkey."));
        let flight = h.seed(fact("Tim's flight to Wellington departs from gate four."));
        let message = "go ahead and do that for me right now please";
        assert!(h.prefetch("alone", message).injected.is_empty(), "{mode:?}");

        let previous = Some("Can you sign in to Fastmail for me?");
        let named = h.converse("a", message, previous, None);
        let replied = h.converse(
            "b",
            "remind me two hours before that leaves so I can pack",
            Some("Anything on this week?"),
            Some("Your flight to Wellington departs Friday at nine."),
        );
        let seen = |memory| if sees { vec![memory] } else { Vec::new() };
        assert_eq!(named.injected, seen(passkey), "{mode:?}");
        assert_eq!(replied.injected, seen(flight), "{mode:?}");
    }
}

/// However long the previous message and the reply, and whatever their
/// script, the message still reaches the reranker: the conversation's
/// context is cut to a start of whole characters and gives way to the
/// message within the reranker's 512 tokens. 300 characters of CJK are 300
/// tokens, so two such parts alone are more than the pair holds.
#[test]
fn the_message_reaches_the_reranker_past_a_long_conversation() {
    let h = Harness::with_reranker(Arc::new(TruncatingReranker), |t| {
        t.injection.rerank_query = RerankQuery::Conversation;
    });
    let pottery = h.seed(fact("Tim takes a pottery class."));
    let message = "pottery class schedule please";
    assert_eq!(h.prefetch("alone", message).injected, vec![pottery]);

    let contexts = [
        "東京".repeat(150),
        "a".repeat(2000),
        "東京".repeat(1000),
        "🙂".repeat(1000),
        format!("x{}", "é".repeat(1000)),
        "東京 ".repeat(1000),
    ];
    for (n, context) in contexts.iter().enumerate() {
        let session = format!("s{n}");
        let prefetch = h.converse(&session, message, Some(context), Some(context));
        assert_eq!(prefetch.injected, vec![pottery], "context {n}");
    }
}

// Cleaning the query

/// The note Hermes' Discord gateway puts in front of a turn's message, with
/// a synthetic message id.
const DISCORD_NOTE: &str = "[Triggering message id: `100000000000000001` \u{2014} use as \
                            `message_id` for reply/react/pin via the discord tools.]";

/// Prefetch recalls for the cleaned message, so the note's words don't make
/// a short follow-up long, and the log keeps the raw query beside it.
#[test]
fn a_prefetch_recalls_for_the_cleaned_message_and_logs_the_raw_one() {
    let h = Harness::new();
    let message = "what time is the ferry on Saturday?";
    for (raw, cleaned) in [
        (format!("[Sam] {message}"), message),
        (format!("{DISCORD_NOTE}\n\n{message}"), message),
        (format!("{DISCORD_NOTE}\n\n[Sam] {message}"), message),
        // Only a leading prefix is a speaker's.
        (
            "remind me to pack [the blue bag] tomorrow".to_owned(),
            "remind me to pack [the blue bag] tomorrow",
        ),
        // Only the note and the prefix: nothing is left.
        ("[Sam] ".to_owned(), ""),
        (DISCORD_NOTE.to_owned(), ""),
        (format!("{DISCORD_NOTE}\n\n[Sam] "), ""),
    ] {
        let row = h.recall_row(h.prefetch("s", &raw).recall_id);
        assert_eq!(row.query.as_deref(), Some(cleaned), "{raw:?}");
        assert_eq!(row.raw_query, Some(raw));
    }

    let dentist = h.seed(fact("Tim's dentist appointment is on Friday."));
    let raw = format!("{DISCORD_NOTE}\n\n[Sam] yes, book it");
    let prefetch = h.converse("s", &raw, Some("[Sam] dentist appointment Friday"), None);
    assert_eq!(prefetch.injected, vec![dentist]);
    let query = h.recall_row(prefetch.recall_id).query.unwrap();
    assert!(query.contains("dentist appointment Friday"), "{query}");
    assert!(!query.contains("Sam"), "{query}");
    assert!(!query.contains("Triggering"), "{query}");
}

// Retrievers and clean-up

#[test]
fn the_entity_arm_finds_memories_linked_to_a_named_entity_but_not_the_seeded_user() {
    // Memories close to the question fill the vector arm, and the target
    // shares no word with the question, so only the entity arm can bring it
    // in. The reranker puts the target first if it's a candidate at all.
    for (question, keyword, found) in [
        ("How is Ana doing?", "greyhound", true),
        ("How is Tim doing?", "sourdough", false),
    ] {
        let models = Models {
            embedder: Arc::new(CrowdingEmbedder(question)),
            reranker: Arc::new(KeywordReranker(keyword)),
        };
        let h = Harness::build(models, |_| {}, Vec::new()).0;
        h.crowd(CANDIDATES_PER_ARM);
        let target = if found {
            let ana = person("Ana Silva", "Ana");
            h.seed(fact("Someone adopted a greyhound.").with("entities", ana))
        } else {
            let baked = h.seed(fact("Someone baked sourdough."));
            let link = LinkRequest {
                memory: baked.to_string(),
                entity: "user".into(),
            };
            h.service.link_entity(BANK, &link).unwrap();
            baked
        };

        let recall = h.recall(query(question));
        let first = recall.results.first().map(|r| r.id);
        assert_eq!(first == Some(target), found, "{question}");
        assert_eq!(ids(&recall).contains(&target), found, "{question}");
    }
}

#[test]
fn a_refined_hit_shows_the_head_of_its_chain() {
    let h = Harness::with_floor(0.0);
    let old = h.seed(fact("Tim's bike is a Brompton."));
    let blue = fact("Tim's bike is a blue folding Brompton.");
    let new = h.seed_changing(blue, old, "refines");

    let recall = h.recall(query("bike Brompton"));
    assert!(ids(&recall).contains(&new));
    assert!(!ids(&recall).contains(&old));
    let prefetch = h.prefetch("s", "bike Brompton");
    assert!(prefetch.injected.contains(&new));
    assert!(!prefetch.injected.contains(&old));
}

#[test]
fn retracted_and_hidden_memories_never_come_back() {
    let h = Harness::with_floor(0.0);
    let retracted = h.seed(fact("Tim's dentist appointment is on 8 October 2026."));
    let hidden = h.seed(fact("Tim's dentist appointment is in Wellington."));
    let kept = h.seed(fact("Tim's dentist is called Dr Ngata."));
    h.service.retract(BANK, &retracted.to_string()).unwrap();
    h.service.forget(BANK, &[hidden.to_string()]).unwrap();

    let recall = ids(&h.recall(query("dentist appointment")));
    assert!(recall.contains(&kept));
    let injected = h.prefetch("s", "dentist appointment").injected;
    for gone in [retracted, hidden] {
        assert!(!recall.contains(&gone) && !injected.contains(&gone));
    }
    for row in h.recalls() {
        assert!(!row.results.contains(&retracted) && !row.results.contains(&hidden));
    }
}

// The injection gate

#[test]
fn injection_gates_on_the_reranker_floor() {
    let h = Harness::new(); // floor 1.0: two shared words
    let pottery = h.seed(fact("Tim takes a pottery class."));
    h.seed(fact("The class was cancelled."));
    h.seed(fact("Tim owns a canoe."));

    let prefetch = h.prefetch("s", "pottery class schedule");
    assert_eq!(prefetch.injected, vec![pottery]);
    assert!(prefetch.text.contains("Tim takes a pottery class."));
    assert!(!prefetch.text.contains("cancelled"));
    // The ids stay out of the text.
    assert!(!prefetch.text.contains(&pottery.to_string()));
    assert!(!prefetch.text.contains(&prefetch.recall_id.to_string()));
}

#[test]
fn injection_takes_the_best_memories_up_to_the_cap_in_score_order() {
    let h = Harness::with(|t| {
        set_floor(t, 0.0);
        t.injection.cap = 2;
    });
    // All three pass the floor.
    let [_, three, two] = h.seed_many([
        fact("A pottery note."),
        fact("A pottery class schedule note."),
        fact("A pottery class note."),
    ]);
    let prefetch = h.prefetch("s", "pottery class schedule");
    assert_eq!(prefetch.injected, vec![three, two]);
}

#[test]
fn injection_stays_within_the_token_budget() {
    const LONG: [&str; 4] = [
        "Tim's pottery class meets in the old church hall on Ponsonby Road, and the teacher asks everyone to bring an apron.",
        "Tim's pottery class covers wheel throwing for the first six weeks and glazing for the last two, with a kiln day at the end.",
        "Tim's pottery class costs two hundred dollars a term, which includes clay, glazes and three firings in the shared kiln.",
        "Tim's pottery class has eight students this term, two of whom have been going for years and help the beginners.",
    ];
    let h = Harness::with(|t| t.injection.token_budget = 70);
    h.seed_many(LONG.map(fact));
    let prefetch = h.prefetch("s", "pottery class");
    assert!(!prefetch.injected.is_empty());
    assert!(prefetch.injected.len() < LONG.len());
    let tokens = estimate_tokens(&prefetch.text);
    assert!(tokens <= 70, "{tokens} tokens");
}

// The relevance scale

#[test]
fn the_floor_gates_on_the_raw_logit_whatever_the_relevance_scale() {
    // At 4.0, a gate on relevance would stop the two-word memory
    // (1.5 / 4 < 1.0), and at 0.25 it would pass the one-word one
    // (0.5 / 0.25 >= 1.0).
    for scale in [0.25, 4.0] {
        let h = Harness::with(|t| set_scale(t, scale));
        let pottery = h.seed(fact("Tim takes a pottery class."));
        let one_word = h.seed(fact("The class was cancelled."));
        let request = request("s", "pottery class schedule", None, None);
        let scored = h.service.scored_prefetch(BANK, &request).unwrap();
        assert_eq!(scored.prefetch.injected, vec![pottery], "scale {scale}");
        // The labelling material shows the raw logit too.
        let logit = |memory: Uuid| {
            let candidate = scored.candidates.iter().find(|c| c.memory == memory);
            candidate.unwrap().logit
        };
        assert_eq!(logit(pottery), Some(1.5), "scale {scale}");
        assert_eq!(logit(one_word), Some(0.5), "scale {scale}");
    }
}

#[test]
fn phase_never_gates_injection_and_the_relevance_scale_leaves_its_term_alone() {
    // A long-past memory sharing four query words, against a current one
    // sharing two, both over the floor. At scale 1.0 the extra words
    // outweigh the full phase penalty (3.5 - 1.0 > 1.5). At 10.0 they don't
    // (0.35 - 1.0 < 0.15), unless the penalty were scaled too; the past
    // memory is injected either way.
    let order = |scale: f64| {
        let h = Harness::with(|t| set_scale(t, scale));
        let text = "Tim's pottery class schedule changed at the studio.";
        let past = event(text, "2026-07-20").with("valid_until", day("2026-08-01"));
        let past = h.seed(past);
        let current = h.seed(fact("Tim's pottery class meets weekly."));
        let injected = h.prefetch("s", "pottery class schedule studio").injected;
        (injected, past, current)
    };
    let (injected, past, current) = order(1.0);
    assert_eq!(injected, vec![past, current]);
    let (injected, past, current) = order(10.0);
    assert_eq!(injected, vec![current, past]);
}

#[test]
fn the_relevance_scale_leaves_the_strength_term_alone_in_recall() {
    // A faded memory sharing three query words, against a fresh one sharing
    // two. At scale 1.0 the extra word outweighs w_s_recall·strength; at
    // 100.0 strength decides, unless it were scaled too.
    let order = |scale: f64| {
        let (h, faded) = Harness::with_faded(
            |t| set_scale(t, scale),
            vec![fact("Tim's pottery class schedule note.")],
        );
        let fresh = h.says(fact("Tim's pottery class note."));
        let ranked = ids(&h.recall(query("pottery class schedule")));
        (ranked, faded[0], fresh)
    };
    let (ranked, faded, fresh) = order(1.0);
    assert_eq!(ranked[..2], [faded, fresh]);
    let (ranked, faded, fresh) = order(100.0);
    assert_eq!(ranked[..2], [fresh, faded]);
}

// The reranker deadline

#[test]
fn a_late_reranker_injects_nothing_and_still_logs_the_prefetch_or_explains_the_miss() {
    let deadline = Duration::from_millis(100);
    let slow = SlowReranker(Duration::from_secs(2));
    let h = Harness::with_reranker(Arc::new(slow), |_| {}).with_deadline(deadline);
    let pottery = h.seed(fact("Tim takes a pottery class."));

    // Explain shows the miss: nothing reranked, so nothing passes.
    let explain = h.explain(explain_injection("pottery class schedule", None, None));
    assert!(!explain.reranked && explain.injection.as_ref().unwrap().text.is_empty());
    assert!(explain.latency.total_ms >= deadline.as_millis() as u64);
    let shown = explained(&explain, pottery);
    assert_eq!((shown.reason, shown.logit), (Some(Cut::NotReranked), None));

    let started = Instant::now();
    let prefetch = h.prefetch("s", "pottery class schedule");
    let waited = started.elapsed();
    assert!(waited < Duration::from_millis(1500), "{waited:?}");
    assert!(!prefetch.reranked);
    assert!(prefetch.injected.is_empty() && prefetch.text.is_empty());

    let row = h.recall_row(prefetch.recall_id);
    assert_eq!(row.kind, "prefetch");
    assert_eq!(row.session_id.as_deref(), Some("s"));
    let latency = row.latency_ms;
    assert!(latency >= deadline.as_millis() as i64, "{latency} ms");

    // Nothing is pending, so the turn commits nothing.
    h.commit("s", &prefetch);
    assert!(h.in_context("s").is_empty());
}

#[test]
fn a_timed_out_reranker_call_leaves_no_queued_inference() {
    // The production reranker runs one inference at a time behind a mutex.
    // A caller that can't start before its deadline falls back without
    // queueing work behind the one running, and once that finishes the
    // reranker serves again. The gate makes this independent of timing.
    let (gated, arrived, answered) = GatedReranker::new();
    let h = Harness::with_reranker(gated.clone(), |_| {});
    let h = h.with_deadline(Duration::from_millis(50));
    let pottery = h.seed(fact("Tim takes a pottery class."));

    let wait = Duration::from_secs(5);
    assert!(!h.prefetch("s", "pottery class schedule").reranked);
    arrived.recv_timeout(wait).expect("the first call arrived");

    // While it's stuck: more prefetches, at once and one after another, and
    // an explicit recall.
    let h = &h;
    std::thread::scope(|scope| {
        let query = "pottery class schedule";
        let calls = ["a", "b", "c"].map(|s| scope.spawn(move || h.prefetch(s, query)));
        for call in calls {
            let prefetch = call.join().unwrap();
            assert!(!prefetch.reranked && prefetch.injected.is_empty());
        }
    });
    assert!(!h.prefetch("d", "pottery class schedule").reranked);
    let recall = h.recall(query("pottery class"));
    assert!(!recall.reranked);
    assert_eq!(ids(&recall), vec![pottery]);
    assert!(arrived.try_recv().is_err(), "a timed-out call queued");

    // Once the stuck call answers, the reranker is used again.
    gated.open();
    answered.recv_timeout(wait).expect("answered");
    let recovered = (0..100)
        .map(|_| h.prefetch("e", "pottery class schedule"))
        .find(|prefetch| prefetch.reranked)
        .expect("the reranker serves again after the stuck call");
    assert_eq!(recovered.injected, vec![pottery]);
}

// The injection format

#[test]
fn annotations_give_absolute_dates() {
    let h = Harness::with_floor(0.0);
    let dentist = "Tim has a dentist appointment on 3 October 2026 at 15:00.";
    let passport = "Tim needs to renew his passport.";
    let berlin = "Tim was on a trip to Berlin.";
    let yoga = "Tim goes to yoga every Tuesday.";
    let lisbon = "Tim is staying in Lisbon.";
    let tax = "Tim is working on the tax return.";
    let sister = "Tim's sister visits in November 2026.";
    let state = |content, volatility| claim(content, "state").with("volatility", json!(volatility));
    let minute = json!({"at": "2026-10-03T15:00", "precision": "minute"});
    h.seed(claim(dentist, "event").with("valid_from", minute));
    h.seed(claim(passport, "task").with("due_at", day("2026-09-30")));
    let trip = event(berlin, "2026-09-10").with("valid_until", day("2026-09-12"));
    h.seed_at(at("2026-09-05T00:00:00Z"), trip);
    h.seed(claim(yoga, "recurring").with("recurrence_text", json!("every Tuesday")));
    // Said exactly four days before now, with a three-day volatility, so
    // its confidence is well below 0.9.
    h.seed_at(at("2026-09-27T07:00:00Z"), state(lisbon, "days"));
    // Said today with a years-long volatility: confident, so no age.
    h.seed_at(at("2026-10-01T06:00:00Z"), state(tax, "years"));
    let month = json!({"at": "2026-11", "precision": "month"});
    let visit = claim(sister, "event").with("valid_from", month);
    h.seed(visit.with("window_confidence", json!("low")));

    let prefetch = h.prefetch("s", "dentist passport Berlin yoga Lisbon tax sister");
    let text = &prefetch.text;
    // Every annotation date carries its weekday: 3 October 2026 is a
    // Saturday, 30 September a Wednesday, 12 September a Saturday and 27
    // September a Sunday. The confident state shows no age.
    for (content, annotation, shown) in [
        (dentist, "Sat 3 Oct 15:00", true),
        (passport, "Wed 30 Sep", true),
        (berlin, "Sat 12 Sep", true),
        (yoga, "every Tuesday", true),
        (lisbon, "Sun 27 Sep", true),
        (tax, "1 Oct", false),
        (sister, "Nov", true),
    ] {
        let line = text.lines().find(|l| l.contains(content));
        let line = line.unwrap_or_else(|| panic!("no line for {content:?} in\n{text}"));
        assert_eq!(line.contains(annotation), shown, "{line}");
    }
}

// The in-context skip and per-session state

#[test]
fn a_committed_injection_isnt_injected_again_in_that_session_until_its_cleared() {
    let h = Harness::new();
    let pottery = h.seed(fact("Tim takes a pottery class."));
    let first = h.prefetch("s", "pottery class schedule");
    assert_eq!(first.injected, vec![pottery]);
    h.commit("s", &first);
    assert_eq!(h.in_context("s"), vec![pottery]);

    let again = h.prefetch("s", "pottery class schedule");
    assert!(again.injected.is_empty() && again.text.is_empty());
    // Another session can't see it, so it's injected there.
    let other = h.prefetch("t", "pottery class schedule");
    assert_eq!(other.injected, vec![pottery]);

    h.service.clear_session(BANK, "s").unwrap();
    assert!(h.in_context("s").is_empty());
    let cleared = h.prefetch("s", "pottery class schedule");
    assert_eq!(cleared.injected, vec![pottery]);
}

#[test]
fn only_an_echo_of_an_injections_own_recall_id_commits_it() {
    // Syncs can arrive after the next prefetch, so a turn with no id or an
    // unknown one leaves every pending injection pending. Ingest is
    // idempotent: a turn resent from the spool or retried is a duplicate
    // and settles nothing.
    let h = Harness::new();
    let pottery = h.seed(fact("Tim takes a pottery class."));
    let canoe = h.seed(fact("Tim paddles his canoe on Sundays."));
    let message_at = h.service.now() - SignedDuration::from_mins(10);
    let earlier = turn("s", message_at, "An earlier message.", None);
    h.service.ingest_turn(BANK, &earlier).unwrap();
    let a = h.prefetch("s", "pottery class schedule");
    let b = h.prefetch("s", "canoe paddles weekend");
    assert_eq!(a.injected, vec![pottery]);
    assert_eq!(b.injected, vec![canoe]);

    h.sync_turn("s", None);
    h.sync_turn("s", Some(Uuid::from_u128(7).to_string()));
    let resent = h.service.ingest_turn(BANK, &earlier).unwrap();
    assert_eq!(resent.outcome, Outcome::Duplicate);
    assert!(h.in_context("s").is_empty());

    h.commit("s", &a);
    assert_eq!(h.in_context("s"), vec![pottery]);
    h.commit("s", &b);
    let in_context = h.in_context("s");
    assert!(in_context.contains(&pottery) && in_context.contains(&canoe));
}

#[test]
fn recall_tool_results_join_the_in_context_set() {
    let h = Harness::new();
    let pottery = h.seed(fact("Tim takes a pottery class."));
    assert_eq!(ids(&h.recall(in_session("pottery class"))), vec![pottery]);
    assert_eq!(h.in_context("s"), vec![pottery]);
    let prefetch = h.prefetch("s", "pottery class schedule");
    assert!(prefetch.injected.is_empty());
    // Explicit recall doesn't skip what's in context.
    assert_eq!(ids(&h.recall(in_session("pottery class"))), vec![pottery]);
}

#[test]
fn the_idle_timeout_is_tunable() {
    let h = Harness::with(|t| t.sessions.in_context_idle_days = 1);
    let pottery = h.seed(fact("Tim takes a pottery class."));
    for session in ["kept", "expired"] {
        h.commit(session, &h.prefetch(session, "pottery class schedule"));
    }
    h.clock.advance(SignedDuration::from_hours(23));
    assert_eq!(h.in_context("kept"), vec![pottery]);
    h.clock.advance(SignedDuration::from_hours(2));
    assert!(h.in_context("expired").is_empty());
    let expired = h.prefetch("expired", "pottery class schedule");
    assert_eq!(expired.injected, vec![pottery]);
}

// The recall log and accesses: a recall is logged, never counted as an access

#[test]
fn a_prefetch_logs_one_row_with_its_query_and_every_candidate() {
    let h = Harness::new();
    let pottery = h.seed(fact("Tim takes a pottery class."));
    let canoe = h.seed(fact("Tim owns a canoe."));
    let before = h.recalls().len();

    let prefetch = h.prefetch("s", "pottery class schedule");
    assert_eq!(prefetch.injected, vec![pottery]);
    let rows = h.recalls();
    assert_eq!(rows.len(), before + 1);
    let row = &rows[0];
    assert_eq!(row.id, prefetch.recall_id);
    assert_eq!(row.kind, "prefetch");
    assert_eq!(row.session_id.as_deref(), Some("s"));
    assert_eq!(row.query.as_deref(), Some("pottery class schedule"));
    assert_eq!(row.at, at(START));
    // The candidate that failed the gate is logged too.
    assert!(row.results.contains(&pottery) && row.results.contains(&canoe));
}

#[test]
fn recall_finds_a_faded_memory_that_injection_drops_and_neither_writes_an_access() {
    let (h, faded) = Harness::with_faded(|_| {}, vec![fact("Tim once tried a pottery wheel.")]);
    let faded = faded[0];
    let pottery = h.seed(fact("Tim takes a pottery class."));
    let fresh = h.says(fact("Tim bought a pottery apron.").level("trivial"));
    let band = |text: &str, memory: Uuid| {
        let recall = h.recall(query(text));
        let result = recall.results.iter().find(|r| r.id == memory);
        result.unwrap().strength
    };
    let accesses = |h: &Harness| [pottery, faded].map(|m| h.accesses(m, None));
    let before = accesses(&h);

    assert!(!h.prefetch("s", "pottery wheel").injected.contains(&faded));
    h.commit("s", &h.prefetch("s", "pottery class schedule"));
    assert!(ids(&h.recall(in_session("pottery wheel"))).contains(&faded));
    assert_eq!(band("pottery apron", fresh), Band::Strong);
    assert_eq!(accesses(&h), before);
    // A faded memory stays faded however often it's recalled.
    assert_eq!(band("pottery wheel", faded), Band::Faded);
}

// Explicit recall

#[test]
fn recall_filters_by_phase() {
    let h = Harness::with_floor(0.0);
    let upcoming = h.seed(event("Tim's garden tour is on 10 October.", "2026-10-10"));
    let past = h.seed(event("Tim's garden party was in September.", "2026-09-10"));
    let current = h.seed(fact("Tim's garden has a lemon tree."));
    // An overdue task is neither upcoming nor past; the `current` filter is
    // the only one that reaches it short of `any`.
    let weed = claim("Tim needs to weed the garden.", "task");
    let overdue = h.seed(weed.with("due_at", day("2026-09-30")));

    let only = |phase| {
        let request = RecallRequest {
            phase,
            ..query("garden")
        };
        ids(&h.recall(request))
    };
    assert_eq!(only(PhaseFilter::Upcoming), vec![upcoming]);
    assert_eq!(only(PhaseFilter::Past), vec![past]);
    let now = only(PhaseFilter::Current);
    assert!(now.contains(&current) && now.contains(&overdue));
    assert!(!now.contains(&upcoming) && !now.contains(&past));
    assert_eq!(only(PhaseFilter::Any).len(), 4);
}

#[test]
fn recall_filters_by_entity_taking_the_union_of_its_matches() {
    let h = Harness::with_floor(0.0);
    let [first, second, neither] = h.seed_many([
        fact("Sam Lee sent the garden photos.").with("entities", person("Sam Lee", "Sam")),
        fact("Sam Ortiz lent Tim a garden fork.").with("entities", person("Sam Ortiz", "Sam")),
        fact("Tim's garden has a lemon tree."),
    ]);
    let about = |entity: &str| {
        let request = RecallRequest {
            entity: Some(entity.into()),
            ..query("garden")
        };
        ids(&h.recall(request))
    };
    let found = about("Sam");
    assert!(found.contains(&first) && found.contains(&second));
    assert!(!found.contains(&neither));
    assert!(about("Nobody").is_empty());
}

#[test]
fn a_recall_range_filters_on_when_it_was_said_or_when_it_happened() {
    let h = Harness::with_floor(0.0);
    // From the start of local day `from` to 23:00 on local day `to`.
    let ranged = |on, text: &str, from: &str, to: &str| {
        let request = RecallRequest {
            from: Some(local(&format!("{from}T00:00"))),
            to: Some(local(&format!("{to}T23:00"))),
            on,
            ..query(text)
        };
        ids(&h.recall(request))
    };

    // `said` matches when it was said.
    let lemon = h.seed_at(
        at("2026-09-10T00:00:00Z"),
        fact("Tim's orchard has a lemon tree."),
    );
    let fig = h.seed_at(
        at("2026-09-20T00:00:00Z"),
        fact("Tim's orchard has a fig tree."),
    );
    let said = ranged(On::Said, "orchard tree", "2026-09-15", "2026-09-25");
    assert!(said.contains(&fig) && !said.contains(&lemon), "{said:?}");

    // `happened` matches the window, widening a low-confidence one by its
    // unit.
    let club = h.seed(event("Tim's garden club meets on 6 October.", "2026-10-06"));
    let fair = h.seed(event("Tim's garden fair is on 5 October.", "2026-10-05"));
    let visit = event("Tim's garden visit is around 5 October.", "2026-10-05");
    let visit = h.seed(visit.with("window_confidence", json!("low")));
    let on_the_6th = ranged(On::Happened, "garden", "2026-10-06", "2026-10-06");
    assert!(on_the_6th.contains(&club));
    assert!(!on_the_6th.contains(&fair), "a confident window outside");
    assert!(on_the_6th.contains(&visit), "a low-confidence window");

    // A fact with no window matches only through a stated start inside the
    // range...
    let joined = fact("Tim joined the garden society.");
    let joined = h.seed(joined.with("valid_from", day("2026-10-06")));
    let shed = h.seed(fact("Tim's garden got a new shed.").with("valid_from", day("2026-01-01")));
    let tree = h.seed(fact("Tim's garden has a pear tree."));
    let early_october = ranged(On::Happened, "garden", "2026-10-01", "2026-10-09");
    assert!(early_october.contains(&joined));
    assert!(!early_october.contains(&shed) && !early_october.contains(&tree));

    // ...but once ended, its window is bounded and history can find it.
    let acme = h.seed(fact("Tim works at Acme.").with("valid_from", day("2026-03-01")));
    let leaving = event("Tim works at Acme until 1 June 2026.", "2026-06-01");
    h.seed_changing(leaving, acme, "ends");
    let may = ranged(
        On::Happened,
        "Tim works at Acme",
        "2026-05-10",
        "2026-05-20",
    );
    assert!(may.contains(&acme), "{may:?}");

    // Either end of a window can be open (CONTEXT.md). Said on 15 September
    // with an end of 12 September and no stated start, this state held on
    // 11 and 12 September.
    let berlin = claim("Tim was staying in Berlin.", "state");
    let berlin = h.seed_at(
        at("2026-09-15T00:00:00Z"),
        berlin.with("valid_until", day("2026-09-12")),
    );
    let mid_september = ranged(On::Happened, "Berlin", "2026-09-11", "2026-09-12");
    assert!(mid_september.contains(&berlin), "{mid_september:?}");
}

#[test]
fn recall_returns_the_default_number_of_results_unless_asked_for_up_to_the_max() {
    let h = Harness::with_floor(0.0);
    let notes = (0..RECALL_LIMIT_MAX + 5).map(|n| fact(&format!("Garden note {n}.")));
    h.seed_all(at(EARLIER), notes.collect());
    let asked = |limit: Option<usize>| {
        let request = RecallRequest {
            limit,
            ..query("garden note")
        };
        h.recall(request).results.len()
    };
    assert_eq!(asked(None), RECALL_LIMIT_DEFAULT);
    assert_eq!(asked(Some(3)), 3);
    assert_eq!(asked(Some(RECALL_LIMIT_MAX)), RECALL_LIMIT_MAX);
}

#[test]
fn recall_text_gives_each_result_its_source_local_day_beside_the_structured_results() {
    let h = Harness::new();
    let empty = h.recall(query("Maya concert"));
    assert!(empty.results.is_empty() && !empty.text.trim().is_empty());

    // Said at UTC+14: its day starts at 10:00 UTC, 23:00 the day before in
    // the bank's Auckland.
    let concert = event("Maya has a concert.", "2026-10-04");
    let concert = h.seed_in("Pacific/Kiritimati", concert);
    let recall = h.recall(query("Maya concert"));
    let result = &recall.results[0];
    let starts = result.window.valid_from.map(|from| from.at);
    assert_eq!(
        (result.id, starts),
        (concert, Some(at("2026-10-03T10:00:00Z")))
    );
    let text = &recall.text;
    assert_eq!(text.lines().count(), 1, "{text}");
    for shown in [&concert.to_string(), "Maya has a concert.", "Sun 4 Oct"] {
        assert!(text.contains(shown), "{text}");
    }
    assert!(!text.contains("Sat 3 Oct"), "{text}");
}

// A queued turn's in-context set
//
// Extraction checks a turn for use against the session's in-context set as
// the turn's sync left it. The worker reaches the turn later, so what
// happens to the session in between (a clear, a later recall, a restart)
// must not change which memories the turn is credited with using.

/// A harness with the pottery memory injected in session `s` and the turn
/// that echoes the injection queued for extraction. The canoe memory is
/// stored but not in context.
fn queued_turn_with_pottery_in_context() -> (Harness, Uuid, Uuid) {
    let h = Harness::new();
    let pottery = h.seed(fact("Tim takes a pottery class."));
    let canoe = h.seed(fact("Tim paddles his canoe on Sundays."));
    let prefetch = h.prefetch("s", "pottery class schedule");
    assert_eq!(prefetch.injected, vec![pottery]);
    h.commit("s", &prefetch);
    assert_eq!(h.in_context("s"), vec![pottery]);
    (h, pottery, canoe)
}

#[test]
fn a_turn_keeps_its_in_context_set_across_a_clear_or_a_restart_before_extraction() {
    // Hermes compacts the session after the turn and before the worker
    // reaches it; or SIGTERM finishes only the chunk in flight, and the next
    // daemon, whose sessions start empty, extracts the turn behind it. The
    // reply was still written with the memory in view.
    for restart in [false, true] {
        let (h, pottery, _) = queued_turn_with_pottery_in_context();
        let h = if restart {
            h.restart()
        } else {
            h.service.clear_session(BANK, "s").unwrap();
            h
        };
        assert!(h.in_context("s").is_empty());

        let llm = h.extract_next(&["m1"]);
        let shown = &llm.requests()[0].user;
        assert!(shown.contains("pottery class"), "restart={restart}");
        assert_eq!(h.accesses(pottery, Some("used")), 1, "restart={restart}");
    }
}

#[test]
fn a_recall_after_the_turn_isnt_in_the_turns_in_context_set() {
    // A later turn's recall adds to the session; the queued turn's reply
    // was written before it and can't have used what it returned.
    let (h, pottery, canoe) = queued_turn_with_pottery_in_context();
    let later = h.recall(in_session("canoe paddles weekend"));
    assert_eq!(later.results[0].id, canoe);
    assert_eq!(h.in_context("s"), vec![pottery, canoe]);

    // m2 is no handle of the turn's own set, so it can't credit anything.
    let llm = h.extract_next(&["m2"]);
    assert!(!llm.requests()[0].user.contains("canoe"));
    assert_eq!(h.accesses(canoe, Some("used")), 0);
    assert_eq!(h.accesses(pottery, Some("used")), 0);
}

// Explain: the pipeline's working for one query, with no side effects

impl Harness {
    fn explain(&self, request: ExplainRequest) -> Explain {
        self.service.explain(BANK, &request).unwrap()
    }
}

/// The explain request for `request`'s query and filters.
fn explain_recall(request: &RecallRequest) -> ExplainRequest {
    ExplainRequest::Recall(ExplainRecall {
        query: request.query.clone(),
        from: request.from,
        to: request.to,
        on: request.on,
        phase: request.phase,
        kinds: request.kinds.clone(),
        entity: request.entity.clone(),
        limit: request.limit,
    })
}

fn explain_injection(query: &str, previous: Option<&str>, reply: Option<&str>) -> ExplainRequest {
    ExplainRequest::Injection(ExplainInjection {
        query: query.into(),
        previous_query: previous.map(Into::into),
        previous_reply: reply.map(Into::into),
    })
}

fn included(explain: &Explain) -> Vec<Uuid> {
    let included = explain.candidates.iter().filter(|c| c.included);
    included.map(|c| c.id).collect()
}

fn explained(explain: &Explain, memory: Uuid) -> &Explained {
    let candidate = explain.candidates.iter().find(|c| c.id == memory);
    candidate.unwrap_or_else(|| panic!("{memory} isn't a candidate: {explain:#?}"))
}

#[test]
fn an_explained_recall_includes_what_recall_returns_in_its_order() {
    let (h, _) = Harness::with_faded(|_| {}, vec![fact("Tim once grew tomatoes in the garden.")]);
    h.seed(event(
        "Tim's garden tour is on 10 October 2026.",
        "2026-10-10",
    ));
    let party = event("Tim's garden party was on 10 September 2026.", "2026-09-10");
    h.seed_at(at("2026-09-12T00:00:00Z"), party);
    h.seed_many([
        fact("Tim shares the garden with Sam.").with("entities", person("Sam Lee", "Sam")),
        claim("Tim needs to weed the garden.", "task").with("due_at", day("2026-09-30")),
        fact("Tim's garden has a lemon tree and a garden shed."),
    ]);

    let limited = RecallRequest {
        limit: Some(2),
        ..query("garden")
    };
    for request in [
        query("garden"),
        limited.clone(),
        RecallRequest {
            phase: PhaseFilter::Current,
            ..query("garden shed")
        },
        RecallRequest {
            kinds: vec![Kind::Event],
            ..query("garden")
        },
        RecallRequest {
            entity: Some("Sam".into()),
            ..query("garden")
        },
        RecallRequest {
            on: On::Said,
            from: Some(at("2026-09-11T00:00:00Z")),
            to: Some(at("2026-09-13T00:00:00Z")),
            ..query("garden")
        },
    ] {
        let returned = ids(&h.recall(request.clone()));
        let explain = h.explain(explain_recall(&request));
        assert!(!returned.is_empty(), "{request:?}");
        assert_eq!(included(&explain), returned, "{request:?}");
        assert_eq!(explain.mode, ExplainMode::Recall);
        assert!(explain.injection.is_none());
    }

    // What the limit cut is listed after the results, saying so.
    let explain = h.explain(explain_recall(&limited));
    let (kept, cut) = explain.candidates.split_at(2);
    assert!(kept.iter().all(|c| c.included));
    assert!(!cut.is_empty() && cut.iter().all(|c| c.reason == Some(Cut::OverLimit)));
}

#[test]
fn an_explained_injection_is_a_fresh_prefetch_with_its_working_and_no_side_effects() {
    let (h, faded) = Harness::with_faded(
        |t| t.injection.cap = 1,
        vec![fact("Tim once tried a pottery class schedule.")],
    );
    let faded = faded[0];
    // Logits 2.5, 1.5 and 0.5 against the floor of 1.0.
    let [best, capped, weak] = h.seed_many([
        fact("A pottery class schedule note."),
        fact("A pottery class note."),
        fact("The class was cancelled."),
    ]);
    let message = "pottery class schedule";
    // Session s has the best in context; t holds it pending.
    h.commit("s", &h.prefetch("s", message));
    let pending = h.prefetch("t", message);
    assert_eq!(pending.injected, vec![best]);
    let traces = |h: &Harness| {
        let accesses = [best, capped, weak, faded].map(|m| h.accesses(m, None));
        (h.recalls().len(), accesses)
    };
    let before = traces(&h);

    h.explain(explain_recall(&query(message)));
    let explain = h.explain(explain_injection(message, None, None));
    assert_eq!(traces(&h), before);
    assert_eq!(h.in_context("s"), vec![best]);
    assert!(h.in_context("t").is_empty());
    h.commit("t", &pending);
    assert_eq!(h.in_context("t"), vec![best]);

    // There's no session, so what s has in context is injected anyway.
    assert_eq!(included(&explain), vec![best]);
    assert!(explain.reranked);
    assert_eq!(explain.query, message);
    let shown = explained(&explain, best);
    assert_eq!((shown.reason, shown.logit), (None, Some(2.5)));
    assert_eq!(shown.score.unwrap().relevance, 2.5);
    assert!(shown.rrf_rank.is_some());
    let bm25 = shown.arms.iter().find(|arm| arm.arm == Arm::Bm25);
    assert!(bm25.is_some_and(|arm| arm.rank.is_some()), "{shown:#?}");
    assert_eq!(explained(&explain, capped).reason, Some(Cut::OverCap));
    assert_eq!(explained(&explain, weak).reason, Some(Cut::UnderFloor));
    let below = explained(&explain, faded);
    assert_eq!(
        (below.reason, below.strength),
        (Some(Cut::BelowTau), Band::Faded)
    );
    assert_eq!((below.rrf_rank, below.logit), (None, None));
    assert_eq!(explain.candidates.last().unwrap().id, faded);

    for (session, message, previous, reply) in [
        ("fresh-1", message, None, None),
        (
            "fresh-2",
            "and the notes?",
            Some("pottery class"),
            Some("The pottery class meets on Tuesdays."),
        ),
    ] {
        let explain = h.explain(explain_injection(message, previous, reply));
        let prefetch = h.converse(session, message, previous, reply);
        assert!(!prefetch.injected.is_empty(), "{message}");
        assert_eq!(explain.mode, ExplainMode::Injection);
        assert_eq!(included(&explain), prefetch.injected, "{message}");
        let injection = explain.injection.unwrap();
        assert_eq!(injection.injected, prefetch.injected, "{message}");
        assert_eq!(injection.text, prefetch.text, "{message}");
    }
}

/// A reranker that answers like `FakeReranker` and records how many
/// documents each call was sent.
#[derive(Default)]
struct CountingReranker(Mutex<Vec<usize>>);

impl Reranker for CountingReranker {
    fn model_id(&self) -> &str {
        FakeReranker::MODEL_ID
    }

    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>, ModelError> {
        self.0.lock().unwrap().push(documents.len());
        FakeReranker.rerank(query, documents)
    }
}

#[test]
fn fused_candidates_past_the_rerank_pool_are_listed_and_never_reranked() {
    let reranker = Arc::new(CountingReranker::default());
    let models = Models {
        embedder: Arc::new(FakeEmbedder),
        reranker: reranker.clone(),
    };
    let faded = vec![fact("Tim once tried a pottery class.")];
    let (h, faded) = Harness::build(models, |_| {}, faded);
    let faded = faded[0];
    // More matches than the reranker takes.
    let notes = (0..RERANKED + 5).map(|n| fact(&format!("Pottery class note {n}.")));
    let notes = h.seed_all(at(EARLIER), notes.collect());

    let recall = h.explain(explain_recall(&query("pottery class")));
    let injection = h.explain(explain_injection("pottery class", None, None));
    let sent = reranker.0.lock().unwrap().clone();
    let pool = sent[0];
    assert!(pool < notes.len(), "{sent:?}");
    assert_eq!(sent, vec![pool; 2]);

    // Recall has no τ gate, so the faded memory is fused there.
    for (explain, mut eligible) in [
        (&recall, [notes.clone(), vec![faded]].concat()),
        (&injection, notes.clone()),
    ] {
        let reranked = &explain.candidates[..pool];
        assert!(reranked.iter().all(|c| c.logit.is_some()));
        let overflow: Vec<&Explained> = explain
            .candidates
            .iter()
            .filter(|c| c.reason == Some(Cut::OutsideRerankPool))
            .collect();
        let listed = reranked.iter().chain(overflow.iter().copied());
        let mut listed: Vec<Uuid> = listed.map(|c| c.id).collect();
        listed.sort();
        eligible.sort();
        assert_eq!(listed, eligible, "{:?}", explain.mode);

        // The overflow follows the reranked candidates in RRF order, with
        // the ranks fusion gave it and nothing from the reranker.
        let ranks: Vec<usize> = overflow.iter().map(|c| c.rrf_rank.unwrap()).collect();
        assert_eq!(ranks, (pool + 1..=eligible.len()).collect::<Vec<_>>());
        assert_eq!(explain.candidates[pool].id, overflow[0].id);
        for c in &overflow {
            assert!(!c.included);
            assert_eq!((c.logit, c.score), (None, None));
            assert!(!c.arms.is_empty());
            assert!(c.arms.iter().all(|arm| arm.rank.is_some()), "{c:#?}");
        }
    }
    // Below τ still comes last, after the overflow.
    assert_eq!(injection.candidates.last().unwrap().id, faded);
    assert_eq!(injection.candidates.len(), notes.len() + 1);

    // Production selects the same, from as many reranked documents.
    assert_eq!(included(&recall), ids(&h.recall(query("pottery class"))));
    let prefetch = h.prefetch("fresh", "pottery class");
    assert_eq!(injection.injection.unwrap().injected, prefetch.injected);
    assert_eq!(*reranker.0.lock().unwrap(), vec![pool; 4]);
}

// Restatements in recall
//
// A restatement is what a repeat said when call 2 absorbed it into a
// memory. Recall searches each restatement's sentence in the vector and
// BM25 arms, keyed to its memory, and reranks a memory on the best of its
// head and its newest five restatements. A returned memory reads as its
// head and then those restatements, oldest first, whichever sentence
// matched; explain says which one did. Reconciliation never sees them.
//
// Each restatement here shares at least half its words with its memory, so
// the fake embedder puts the memory above the reconcile floor and call 2
// runs to absorb it.

const CAT: &str = "Tim's cat is called Miso.";
const TABBY: &str = "Tim's cat is called Miso and she is a ginger tabby.";
const PORIRUA: &str = "Tim's cat is called Miso and she came from Porirua.";
const BIKE: &str = "Tim's bike is a Brompton.";

impl Harness {
    /// The owner says `sentence` at `said` and call 2 labels it
    /// `mentioned_again` on `memory`, so the memory keeps it as a
    /// restatement and no memory is made of it.
    fn restate(&self, memory: Uuid, said: Timestamp, sentence: &str) {
        static SESSIONS: AtomicU64 = AtomicU64::new(0);
        let session = format!("restate-{}", SESSIONS.fetch_add(1, Ordering::Relaxed));
        let restated = |h: &Harness| {
            let view = h.service.show_memory(BANK, &memory.to_string()).unwrap();
            view.restatements.len()
        };
        let before = restated(self);
        let turn = turn(&session, said, sentence, None);
        self.service.ingest_turn(BANK, &turn).unwrap();
        let call1 = json!({"claims": [fact(sentence)], "used_injected_ids": []});
        let call2 = {
            let lease = self.service.claim_chunk(BANK).unwrap().expect("queued");
            let input = self.service.call2_input(&lease, &call1, &[]).unwrap();
            let input = input.expect("call 2 runs");
            let neighbour = input.neighbours.iter().find(|n| n.memory == memory);
            let neighbour = neighbour.expect("the memory is a neighbour");
            json!({"claims": [{
                "claim": input.claims[0].handle,
                "labels": [{"neighbour": neighbour.handle, "label": "mentioned_again"}],
            }]})
        };
        let llm = FakeLlm::scripted("fake-llm", vec![call1, call2]);
        let extracted = self.service.extract_next(BANK, &llm).unwrap();
        let extracted = extracted.expect("the turn was queued");
        assert!(extracted.memories.is_empty(), "{sentence:?} was absorbed");
        assert_eq!(restated(self), before + 1, "{sentence:?} is restated");
    }

    /// Call 2's input for a turn saying `claims` now, or `None` when call 2
    /// wouldn't run. The turn stays queued.
    fn call2_for(&self, claims: Vec<Value>) -> Option<asphodel_core::extraction::Call2Input> {
        let quotes: Vec<&str> = claims
            .iter()
            .map(|c| c["quote"].as_str().unwrap())
            .collect();
        let said = self.service.now() - SignedDuration::from_mins(1);
        let turn = turn("call-2", said, &quotes.join(" "), None);
        self.service.ingest_turn(BANK, &turn).unwrap();
        let call1 = json!({"claims": claims, "used_injected_ids": []});
        let lease = self.service.claim_chunk(BANK).unwrap().expect("queued");
        self.service.call2_input(&lease, &call1, &[]).unwrap()
    }
}

/// Restating a memory gives it accesses, which make it stronger than the
/// memories it's ranked against. Relevance at four times the logit lets the
/// words a question shares decide the order instead.
fn by_relevance() -> Harness {
    Harness::with(|t| set_scale(t, 0.25))
}

/// `days` days after [`EARLIER`], when restatements are said in order.
fn days_later(days: i64) -> Timestamp {
    at(EARLIER) + SignedDuration::from_hours(24 * days)
}

/// `memory`'s result in `recall`.
fn result(recall: &Recall, memory: Uuid) -> &asphodel_core::retrieval::Recalled {
    let found = recall.results.iter().find(|r| r.id == memory);
    found.unwrap_or_else(|| panic!("{memory} wasn't recalled: {recall:#?}"))
}

/// `memory`'s candidate in `explain` as the explain endpoint serves it.
fn explained_json(explain: &Explain, memory: Uuid) -> Value {
    let candidate = explained(explain, memory);
    serde_json::to_value(candidate).unwrap()
}

/// The 1-based place `arm` gave `candidate`, or `None` when it didn't find
/// it.
fn arm_rank(candidate: &Explained, arm: Arm) -> Option<usize> {
    let found = candidate.arms.iter().find(|a| a.arm == arm);
    found.and_then(|a| a.rank)
}

#[test]
fn a_query_matching_only_a_restatement_recalls_its_memory() {
    // The question shares no word with the cat's own sentence: only what
    // was said again about her names a ginger tabby from Porirua. Sam's cat
    // shares one word with it and would otherwise come first.
    let h = by_relevance();
    let cat = h.seed(fact(CAT));
    h.restate(cat, days_later(1), TABBY);
    h.restate(cat, days_later(2), PORIRUA);
    let sam = h.seed(fact("Sam's ginger cat sleeps all day on the sofa."));
    let question = "ginger tabby Porirua";

    let first = h.recall(RecallRequest {
        limit: Some(1),
        ..query(question)
    });
    assert_eq!(ids(&first), vec![cat], "{first:#?}");
    let recall = h.recall(query(question));
    assert_eq!(ids(&recall), vec![cat, sam], "{recall:#?}");

    let explain = h.explain(explain_recall(&query(question)));
    let shown = explained(&explain, cat);
    assert_eq!(arm_rank(shown, Arm::Vector), Some(1), "{shown:#?}");
    assert_eq!(arm_rank(shown, Arm::Bm25), Some(1), "{shown:#?}");
    // The best single sentence's logit, the ginger tabby's two words, not
    // the three every sentence of the memory holds between them.
    assert_eq!(shown.logit, Some(1.5), "{shown:#?}");
}

#[test]
fn a_recalled_memory_reads_as_its_head_then_its_newest_five_restatements_oldest_first() {
    let h = Harness::new();
    let cat = h.seed(fact(CAT));
    let likes = [
        "sardines", "boxes", "sunbeams", "string", "naps", "cushions", "rain",
    ];
    let restated: Vec<String> = likes
        .iter()
        .map(|thing| format!("Tim's cat is called Miso and she likes {thing}."))
        .collect();
    for (day, sentence) in (1..).zip(&restated) {
        h.restate(cat, days_later(day), sentence);
    }
    let (left_out, shown) = restated.split_at(2);

    // The same text whichever sentence the question matches: the head's,
    // the newest restatement's, or the oldest's, which is searched but not
    // shown.
    let mut texts = BTreeSet::new();
    for question in ["Miso", "rain", "sardines"] {
        let recall = h.recall(query(question));
        let text = result(&recall, cat).sentence.clone();
        assert!(text.starts_with(CAT), "{question}: {text}");
        assert_eq!(text.matches(CAT).count(), 1, "{question}: {text}");
        let places: Vec<usize> = shown
            .iter()
            .map(|sentence| {
                let place = text.find(sentence.as_str());
                place.unwrap_or_else(|| panic!("{question}: {sentence:?} isn't in {text:?}"))
            })
            .collect();
        assert!(places.is_sorted(), "{question}: {text}");
        for sentence in left_out {
            assert!(!text.contains(sentence.as_str()), "{question}: {text}");
        }
        assert!(recall.text.contains(&text), "{question}: {}", recall.text);
        texts.insert(text);
    }
    assert_eq!(texts.len(), 1, "{texts:#?}");

    // The reranker sees the head and the newest five: the newest
    // restatement's word scores, and the oldest's finds the memory through
    // BM25 but scores as if nothing matched.
    for (question, logit) in [("rain", 0.5), ("sardines", -0.5)] {
        let explain = h.explain(explain_recall(&query(question)));
        let shown = explained(&explain, cat);
        assert!(
            arm_rank(shown, Arm::Bm25).is_some(),
            "{question}: {shown:#?}"
        );
        assert_eq!(shown.logit, Some(logit), "{question}: {shown:#?}");
    }
}

#[test]
fn a_sentence_said_again_is_shown_once() {
    // Rua's sentence is restated word for word, and so is one restatement.
    let dog = "Tim's dog is called Rua.";
    let park = "Tim's dog is called Rua and loves the park.";
    let h = Harness::new();
    let rua = h.seed(fact(dog));
    h.restate(rua, days_later(1), dog);
    h.restate(rua, days_later(2), park);
    h.restate(rua, days_later(3), park);

    let recall = h.recall(query("Rua park"));
    let text = &result(&recall, rua).sentence;
    assert!(text.starts_with(dog), "{text}");
    assert_eq!(text.matches(dog).count(), 1, "{text}");
    assert_eq!(text.matches(park).count(), 1, "{text}");
}

#[test]
fn a_memory_restated_many_times_takes_one_candidate_slot() {
    // Every restatement of the cat matches the question better than Sam's
    // cat does, and the cat still holds one place in each arm and one in
    // the results.
    let h = by_relevance();
    let cat = h.seed(fact(CAT));
    let likes = [
        "sardines", "boxes", "sunbeams", "string", "naps", "cushions",
    ];
    for (day, thing) in (1..).zip(likes) {
        let sentence = format!("Tim's cat is called Miso, a ginger tabby who likes {thing}.");
        h.restate(cat, days_later(day), &sentence);
    }
    let [sam, ..] = h.seed_many([
        fact("Sam has a ginger cat too, and it is called Rangi and it sleeps a lot."),
        fact(BIKE),
        fact("Tim lives in Wellington."),
    ]);
    let request = query("ginger tabby");

    let recall = h.recall(request.clone());
    let returned = ids(&recall);
    assert_eq!(returned.iter().filter(|id| **id == cat).count(), 1);
    assert_eq!(returned[..2], [cat, sam], "{recall:#?}");

    let explain = h.explain(explain_recall(&request));
    let listed = explain.candidates.iter().filter(|c| c.id == cat).count();
    assert_eq!(listed, 1, "{explain:#?}");
    let second = explained(&explain, sam);
    assert_eq!(arm_rank(second, Arm::Vector), Some(2), "{second:#?}");
    assert_eq!(arm_rank(second, Arm::Bm25), Some(2), "{second:#?}");
    assert_eq!(second.rrf_rank, Some(2), "{second:#?}");
}

#[test]
fn a_restatement_never_makes_call_2_run_or_joins_its_neighbours() {
    // The claim is close to what was said again about the cat and to
    // nothing the cat's own sentence says.
    let rescued = "Tim's cat is called Miso, a ginger tabby rescued from Porirua.";
    let claim = || vec![fact("A ginger tabby rescued from Porirua.")];

    // Alone, it runs no call 2.
    let h = Harness::new();
    let cat = h.seed(fact(CAT));
    h.restate(cat, days_later(1), rescued);
    assert!(h.call2_for(claim()).is_none());

    // With a memory near it, call 2 runs, and the cat isn't shown.
    let h = Harness::new();
    let cat = h.seed(fact(CAT));
    h.restate(cat, days_later(1), rescued);
    let winter = h.seed(fact("A ginger tabby was rescued in Porirua last winter."));
    let input = h.call2_for(claim()).expect("call 2 runs");
    let shown: Vec<Uuid> = input.neighbours.iter().map(|n| n.memory).collect();
    assert_eq!(shown, vec![winter], "{input:#?}");
}

#[test]
fn a_memory_without_restatements_scores_and_reads_as_before() {
    // Restating the cat with Tim's bike in it leaves the bike's own score
    // and text alone.
    let h = Harness::new();
    let cat = h.seed(fact(CAT));
    let bike = h.seed(fact(BIKE));
    let question = "Tim bike Brompton";
    let scored = |h: &Harness| {
        let explain = h.explain(explain_recall(&query(question)));
        let shown = explained(&explain, bike);
        (shown.logit, shown.score.map(|parts| parts.relevance))
    };
    let before = scored(&h);
    assert_eq!(before, (Some(2.5), Some(2.5)));

    let sleeps = "Tim's cat is called Miso and she sleeps by the bike.";
    h.restate(cat, days_later(1), sleeps);
    assert_eq!(scored(&h), before);
    let recall = h.recall(query(question));
    assert_eq!(result(&recall, bike).sentence, BIKE);
    assert!(result(&recall, cat).sentence.contains(sleeps));
}

#[test]
fn explain_names_the_sentence_that_matched() {
    // Each arm, and the reranker, says which of a memory's sentences it
    // matched: a restatement, or the head of a memory with none.
    let h = Harness::new();
    let cat = h.seed(fact(CAT));
    h.restate(cat, days_later(1), TABBY);
    h.restate(cat, days_later(2), PORIRUA);
    let bike = h.seed(fact(BIKE));

    for (question, memory, sentence) in [
        ("ginger tabby", cat, TABBY),
        ("Porirua", cat, PORIRUA),
        ("bike Brompton", bike, BIKE),
    ] {
        let explain = h.explain(explain_recall(&query(question)));
        let shown = explained_json(&explain, memory);
        assert_eq!(shown["matched"], sentence, "{question}: {shown:#}");
        for arm in ["vector", "bm25"] {
            let arms = shown["arms"].as_array().unwrap();
            let found = arms.iter().find(|a| a["arm"] == arm);
            let found = found.unwrap_or_else(|| panic!("{question}: no {arm} arm in {shown:#}"));
            assert_eq!(found["sentence"], sentence, "{question}, {arm}: {shown:#}");
        }
    }
}
