//! Hybrid recall, reranking, injection gates and recall logs. Being
//! recalled or injected is logged but never strengthens a memory.
//!
//! The API under test is `asphodel_core::retrieval` and the `Service`
//! methods over it: `prefetch`, `recall`, `in_context`, `clear_session` and
//! `ingest_turn`'s settling of the pending injection. Memories are inserted
//! directly, as an earlier extraction would have left them, and indexed with
//! the fake embedder.
//!
//! `FakeReranker`'s logit is the number of distinct query words a memory
//! shares, minus one half, so a gate floor of 1.0 lets through a memory
//! that shares two words and stops one that shares one.
//!
//! Every service here runs on a `SimulatedClock` stopped at 20:00 on
//! Thursday 1 October 2026 in Auckland unless a test advances it.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant};

use asphodel_core::config::RankingTuning;
use asphodel_core::constants::CANDIDATES_PER_ARM;
use asphodel_core::ingest::{Outcome, Turn};
use asphodel_core::models::{
    Embedder, FakeEmbedder, FakeLlm, FakeReranker, ModelError, Models, Reranker,
};
use asphodel_core::retrieval::{
    Arm, Band, Cut, Explain, ExplainInjection, ExplainMode, ExplainRecall, ExplainRequest,
    Explained, On, PhaseFilter, Prefetch, PrefetchRequest, Recall, RecallRequest, clean_query,
    conversation_query, effective_query, estimate_tokens, fuse, phase_term,
};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{OpenOptions, Store, VectorIndex, micros};
use asphodel_core::strength::{Kind, TimePrecision, Window, WorldTime};
use asphodel_core::{Service, SimulatedClock, Tuning};
use jiff::civil::DateTime;
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use rusqlite::types::FromSql;
use uuid::Uuid;

// Fixtures

/// 20:00 on Thursday 1 October 2026 in Auckland, on daylight time (UTC+13).
const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";
const BANK: &str = "main";

/// When fixture memories were said, unless a test says otherwise.
const EARLIER: &str = "2026-09-01T00:00:00Z";

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
}

/// A local date-time in `TZ` as the instant stored for it.
fn local(datetime: &str) -> Timestamp {
    datetime
        .parse::<DateTime>()
        .unwrap()
        .to_zoned(TimeZone::get(TZ).unwrap())
        .unwrap()
        .timestamp()
}

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-retrieval-{}-{}",
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
    Uuid::from_u128((0xf3_u128 << 120) | u128::from(NEXT.fetch_add(1, Ordering::Relaxed)))
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
/// counting the calls that reached it and signalling each arrival and each
/// answer, so a test can tell what ran without timing it.
struct GatedReranker {
    entered: AtomicUsize,
    open: Mutex<bool>,
    opened: Condvar,
    arrivals: Mutex<mpsc::Sender<()>>,
    answers: Mutex<mpsc::Sender<()>>,
}

impl GatedReranker {
    fn new() -> (Arc<Self>, mpsc::Receiver<()>, mpsc::Receiver<()>) {
        let (arrivals, arrived) = mpsc::channel();
        let (answers, answered) = mpsc::channel();
        let reranker = Arc::new(Self {
            entered: AtomicUsize::new(0),
            open: Mutex::new(false),
            opened: Condvar::new(),
            arrivals: Mutex::new(arrivals),
            answers: Mutex::new(answers),
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
        self.entered.fetch_add(1, Ordering::SeqCst);
        let _ = self.arrivals.lock().unwrap().send(());
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.opened.wait(open).unwrap();
        }
        drop(open);
        let answer = FakeReranker.rerank(query, documents);
        let _ = self.answers.lock().unwrap().send(());
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
                let seen: std::collections::BTreeSet<&String> = query[..q].iter().collect();
                let shared: std::collections::BTreeSet<&String> = document[..d]
                    .iter()
                    .filter(|token| token.chars().any(char::is_alphanumeric))
                    .filter(|token| seen.contains(token))
                    .collect();
                shared.len() as f32 - 0.5
            })
            .collect())
    }
}

/// A memory to insert. `Default` is a notable fact said at [`EARLIER`],
/// with high window confidence and its `created` access then.
#[derive(Clone)]
struct Memory {
    content: &'static str,
    kind: &'static str,
    significance: &'static str,
    owner_significance: Option<&'static str>,
    observed_at: Timestamp,
    /// When its `created` access is; `observed_at` when `None`.
    created_at: Option<Timestamp>,
    valid_from: Option<(Timestamp, &'static str)>,
    valid_until: Option<(Timestamp, &'static str)>,
    due_at: Option<(Timestamp, &'static str)>,
    low_confidence: bool,
    volatility: Option<&'static str>,
    recurrence_text: Option<&'static str>,
    retracted: bool,
    hidden: bool,
    /// Indexed under this vector instead of its content's.
    vector: Option<Vec<f32>>,
}

impl Default for Memory {
    fn default() -> Self {
        Self {
            content: "",
            kind: "fact",
            significance: "notable",
            owner_significance: None,
            observed_at: at(EARLIER),
            created_at: None,
            valid_from: None,
            valid_until: None,
            due_at: None,
            low_confidence: false,
            volatility: None,
            recurrence_text: None,
            retracted: false,
            hidden: false,
            vector: None,
        }
    }
}

fn fact(content: &'static str) -> Memory {
    Memory {
        content,
        ..Memory::default()
    }
}

struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    chunk: i64,
    _dir: TestDir,
}

impl Harness {
    /// Fake models, a gate floor of 1.0 and default tuning otherwise.
    fn new() -> Self {
        Self::with(1.0, "", Arc::new(FakeReranker))
    }

    fn with_floor(floor: f64) -> Self {
        Self::with(floor, "", Arc::new(FakeReranker))
    }

    /// `injection` is extra keys for `[injection]`, and `rest` extra
    /// sections. The relevance scale is 1.0.
    fn with(floor: f64, extra: &str, reranker: Arc<dyn Reranker>) -> Self {
        Self::with_scaled(floor, 1.0, extra, reranker)
    }

    /// A gate floor of `floor` and a relevance scale of `scale` for the
    /// fake reranker.
    fn with_scale(floor: f64, scale: f64) -> Self {
        Self::with_scaled(floor, scale, "", Arc::new(FakeReranker))
    }

    fn with_scaled(floor: f64, scale: f64, extra: &str, reranker: Arc<dyn Reranker>) -> Self {
        let (injection, rest) = extra.split_once("\n---\n").unwrap_or((extra, ""));
        let tuning = Tuning::from_toml(&format!(
            "[injection]\n{injection}\n\
             [injection.reranker_floors]\n\"{}\" = {floor:?}\n\
             [ranking.relevance_scales]\n\"{0}\" = {scale:?}\n\
             [reconcile.embedding_floors]\n\"{}\" = 0.5\n{rest}",
            FakeReranker::MODEL_ID,
            FakeEmbedder::MODEL_ID,
        ))
        .unwrap();
        let dir = TestDir::new();
        let clock = Arc::new(SimulatedClock::new(at(START)));
        let store = Store::open(&dir.0, OpenOptions::default(), clock.clone()).unwrap();
        let models = Models {
            embedder: Arc::new(FakeEmbedder),
            reranker,
        };
        let service = Service::with_models(clock.clone(), store, tuning, models).unwrap();
        service
            .ensure_bank_with_models(
                BANK,
                &BankIdentity {
                    owner_name: Some("Tim".into()),
                    timezone: Some(TZ.into()),
                    ..BankIdentity::default()
                },
            )
            .unwrap();
        // The chunk fixture memories rest on: a turn said at EARLIER, taken
        // off the queue as if extracted.
        service
            .ingest_turn(BANK, &turn("fixtures", EARLIER, "Fixtures.", None))
            .unwrap();
        let mut harness = Self {
            service,
            clock,
            chunk: 0,
            _dir: dir,
        };
        harness.chunk = harness.one("SELECT id FROM chunks", []);
        harness.execute("DELETE FROM extraction_queue", []);
        harness
    }

    fn with_deadline(self, deadline: Duration) -> Self {
        let Self {
            service,
            clock,
            chunk,
            _dir,
        } = self;
        Self {
            service: service.with_reranker_deadline(deadline),
            clock,
            chunk,
            _dir,
        }
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
        let now = micros(self.service.now());
        let stamp = |t: Option<(Timestamp, &'static str)>| {
            (
                t.map(|(at, _)| micros(at)),
                t.map(|(_, precision)| precision),
            )
        };
        let (valid_from, valid_from_precision) = stamp(memory.valid_from);
        let (valid_until, valid_until_precision) = stamp(memory.valid_until);
        let (due_at, due_at_precision) = stamp(memory.due_at);
        {
            let store = self.service.store().unwrap();
            let conn = store.connection();
            conn.execute(
                "INSERT INTO memories (uuid, bank_id, content, kind, significance,
                                       owner_significance, chunk_id, source_start, source_end,
                                       observed_at, valid_from, valid_from_precision,
                                       valid_until, valid_until_precision, window_confidence,
                                       due_at, due_at_precision, volatility, recurrence_text,
                                       invalidated_at, hidden_at, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, 9, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                         ?16, ?17, ?18, ?19, ?20, ?20)",
                rusqlite::params![
                    uuid.to_string(),
                    bank_id,
                    memory.content,
                    memory.kind,
                    memory.significance,
                    memory.owner_significance,
                    self.chunk,
                    micros(memory.observed_at),
                    valid_from,
                    valid_from_precision,
                    valid_until,
                    valid_until_precision,
                    if memory.low_confidence { "low" } else { "high" },
                    due_at,
                    due_at_precision,
                    memory.volatility,
                    memory.recurrence_text,
                    memory.retracted.then_some(now),
                    memory.hidden.then_some(now),
                    now,
                ],
            )
            .unwrap();
            let id = conn.last_insert_rowid();
            let vector = memory
                .vector
                .clone()
                .unwrap_or_else(|| FakeEmbedder.embed(&[memory.content]).unwrap().remove(0));
            store.vectors().upsert(&conn, bank_id, id, &vector).unwrap();
            conn.execute(
                "INSERT INTO accesses (bank_id, memory_id, kind, at, turn)
                 VALUES (?1, ?2, 'created', ?3, 0)",
                (
                    bank_id,
                    id,
                    micros(memory.created_at.unwrap_or(memory.observed_at)),
                ),
            )
            .unwrap();
        }
        uuid
    }

    /// `old` refined into `new`: `old` points at `new` and isn't retracted.
    fn refine(&self, old: Uuid, new: Uuid) {
        self.execute(
            "UPDATE memories SET superseded_by = ?2 WHERE id = ?1",
            (self.rowid(old), self.rowid(new)),
        );
    }

    /// A person entity with `aliases`, returning its rowid.
    fn entity(&self, name: &str, aliases: &[&str]) -> i64 {
        let bank_id = self.bank_id();
        self.execute(
            "INSERT INTO entities (uuid, bank_id, name, kind, created_at, updated_at)
             VALUES (?1, ?2, ?3, 'person', 0, 0)",
            (next_uuid().to_string(), bank_id, name),
        );
        let entity: i64 = self.one("SELECT MAX(id) FROM entities", []);
        for alias in aliases {
            self.execute(
                "INSERT INTO entity_aliases (bank_id, entity_id, alias, created_at)
                 VALUES (?1, ?2, ?3, 0)",
                (bank_id, entity, alias),
            );
        }
        entity
    }

    fn seeded(&self, which: &str) -> i64 {
        self.one(
            "SELECT id FROM entities WHERE bank_id = ?1 AND seeded = ?2",
            (self.bank_id(), which),
        )
    }

    fn link(&self, memory: Uuid, entity: i64) {
        self.execute(
            "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?1, ?2)",
            (self.rowid(memory), entity),
        );
    }

    /// `count` memories indexed under `query`'s own vector, so they fill
    /// the vector arm, with content that shares no word with it.
    fn crowd(&self, query: &str, count: usize) {
        const FILLER: [&str; 4] = [
            "Filler entry alpha.",
            "Filler entry bravo.",
            "Filler entry charlie.",
            "Filler entry delta.",
        ];
        let vector = FakeEmbedder.embed(&[query]).unwrap().remove(0);
        for n in 0..count {
            self.insert(Memory {
                content: FILLER[n % FILLER.len()],
                vector: Some(vector.clone()),
                ..Memory::default()
            });
        }
    }

    fn prefetch(&self, session: &str, query: &str) -> Prefetch {
        self.prefetch_after(session, query, None)
    }

    fn prefetch_after(&self, session: &str, query: &str, previous: Option<&str>) -> Prefetch {
        self.service
            .prefetch(
                BANK,
                &PrefetchRequest {
                    session_id: session.into(),
                    query: query.into(),
                    previous_query: previous.map(Into::into),
                    previous_reply: None,
                    block_id: None,
                },
            )
            .unwrap()
    }

    /// A prefetch sent with the previous message and the start of the
    /// assistant's reply to it.
    fn prefetch_in_conversation(
        &self,
        session: &str,
        query: &str,
        previous: Option<&str>,
        reply: Option<&str>,
    ) -> Prefetch {
        self.service
            .prefetch(
                BANK,
                &PrefetchRequest {
                    session_id: session.into(),
                    query: query.into(),
                    previous_query: previous.map(Into::into),
                    previous_reply: reply.map(Into::into),
                    block_id: None,
                },
            )
            .unwrap()
    }

    fn recall(&self, request: RecallRequest) -> Recall {
        self.service.recall(BANK, &request).unwrap()
    }

    /// The turn that follows a prefetch, echoing `recall_id`.
    fn sync_turn(&self, session: &str, recall_id: Option<String>) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let message_at = self.service.now() - SignedDuration::from_secs(60);
        let mut turn = turn(session, EARLIER, &format!("Message {n}."), recall_id);
        turn.message_at = message_at;
        self.service.ingest_turn(BANK, &turn).unwrap();
    }

    fn in_context(&self, session: &str) -> Vec<Uuid> {
        self.service.in_context(BANK, session).unwrap()
    }

    fn accesses(&self) -> i64 {
        self.one("SELECT COUNT(*) FROM accesses", [])
    }

    /// A recall row: kind, session, turn, query, latency and time.
    fn recall_row(&self, recall_id: Uuid) -> (String, Option<String>, i64, String, i64, i64) {
        self.service
            .store()
            .unwrap()
            .connection()
            .query_row(
                "SELECT kind, session_id, turn, query, latency_ms, at FROM recalls WHERE uuid = ?1",
                [recall_id.to_string()],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .unwrap()
    }

    /// A recall row's cleaned query and the raw query it was cleaned from.
    fn recall_queries(&self, recall_id: Uuid) -> (String, Option<String>) {
        self.service
            .store()
            .unwrap()
            .connection()
            .query_row(
                "SELECT query, raw_query FROM recalls WHERE uuid = ?1",
                [recall_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap()
    }

    /// A recall's results, best rank first: memory and whether injected.
    fn results(&self, recall_id: Uuid) -> Vec<(Uuid, bool)> {
        let store = self.service.store().unwrap();
        let conn = store.connection();
        let mut statement = conn
            .prepare(
                "SELECT m.uuid, r.injected FROM recall_results r
                 JOIN recalls c ON c.id = r.recall_id JOIN memories m ON m.id = r.memory_id
                 WHERE c.uuid = ?1 ORDER BY r.rank",
            )
            .unwrap();
        statement
            .query_map([recall_id.to_string()], |row| {
                Ok((row.get::<_, String>(0)?.parse().unwrap(), row.get(1)?))
            })
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    /// The score a recall logged for `memory`.
    fn score(&self, recall_id: Uuid, memory: Uuid) -> f64 {
        self.one::<Option<f64>, _>(
            "SELECT r.score FROM recall_results r
             JOIN recalls c ON c.id = r.recall_id JOIN memories m ON m.id = r.memory_id
             WHERE c.uuid = ?1 AND m.uuid = ?2",
            [recall_id.to_string(), memory.to_string()],
        )
        .unwrap()
    }
}

fn turn(session: &str, message_at: &str, user: &str, recall_id: Option<String>) -> Turn {
    Turn {
        session_id: session.into(),
        message_at: at(message_at),
        timezone: Some(TZ.into()),
        user_text: user.into(),
        assistant_text: "Noted.".into(),
        author: None,
        platform: Some("cli".into()),
        recall_id,
        forget_requested: false,
    }
}

/// Fixture content built at run time, for `Memory`'s `&'static str`.
fn leak(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
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

// Fusion

#[test]
fn relevance_is_the_logit_divided_by_the_relevance_scale() {
    // At scale 1.0 relevance is the raw logit, so scores are what they
    // were before the scale; any other scale divides it, in prefetch and
    // in recall alike. Nothing else in the score depends on the scale.
    let scores = |scale: f64| {
        let h = Harness::with_scale(0.0, scale);
        let pottery = h.insert(fact("Tim takes a pottery class."));
        let prefetch = h.prefetch("s", "pottery class schedule");
        let recall = h.recall(query("pottery class schedule"));
        (
            h.score(prefetch.recall_id, pottery),
            h.score(recall.recall_id, pottery),
        )
    };
    let (prefetch_raw, recall_raw) = scores(1.0);
    let (prefetch_scaled, recall_scaled) = scores(4.0);
    let tuning = Tuning::default();
    let strength = asphodel_core::strength::strength(
        asphodel_core::constants::Significance::Notable.value(),
        &[asphodel_core::strength::Access {
            kind: asphodel_core::strength::AccessKind::Created,
            at: at(EARLIER),
        }],
        None,
        &asphodel_core::strength::BankTime::new(&[at(EARLIER)], tuning.clock.quiet_rate),
        at(START),
    )
    .value;
    // A fact without volatility or a window has zero confidence/phase terms.
    // Pin the original formula as well as the scale-dependent difference.
    assert!((prefetch_raw - (1.5 + 0.5 * strength)).abs() < 1e-9);
    assert!((recall_raw - (1.5 + 0.2 * strength)).abs() < 1e-9);
    // Two shared words: a logit of 1.5, so relevance 1.5 and then 0.375.
    let expected = 1.5 - 1.5 / 4.0;
    for (raw, scaled) in [(prefetch_raw, prefetch_scaled), (recall_raw, recall_scaled)] {
        assert!(
            (raw - scaled - expected).abs() < 1e-9,
            "{raw} - {scaled} isn't {expected}"
        );
    }
}

#[test]
fn fusion_sums_reciprocal_ranks_with_k_60() {
    // x is first in one list; y is at rank r in two others. y wins iff
    // 2 / (k + r) > 1 / (k + 1), that is r < k + 2: at k = 60, rank 61
    // beats x and rank 63 loses to it.
    let padded = |rank: usize| -> Vec<i64> {
        let mut list: Vec<i64> = (1000..1000 + rank as i64 - 1).collect();
        list.push(2);
        list
    };
    let x_first = [1_i64];
    let y_at_61 = padded(61);
    let fused = fuse(&[&x_first, &y_at_61, &y_at_61.clone()]);
    let rank = |id: i64, fused: &[i64]| fused.iter().position(|f| *f == id).unwrap();
    assert!(rank(2, &fused) < rank(1, &fused));

    let y_at_63 = padded(63);
    let fused = fuse(&[&x_first, &y_at_63, &y_at_63.clone()]);
    assert!(rank(1, &fused) < rank(2, &fused));
}

#[test]
fn a_repeat_within_one_list_counts_once() {
    // If the repeat of 1 counted, 1 would score 1/61 + 1/62 and 2 would be
    // pushed to rank 3; counted once, 2 is second in the first list and
    // first in the other, so it wins.
    assert_eq!(fuse(&[&[1, 1, 2], &[2]]), vec![2, 1]);
}

// Score and phase

fn utc(text: &str) -> WorldTime {
    WorldTime {
        at: at(text),
        precision: TimePrecision::Minute,
    }
}

fn window(kind: Kind) -> Window {
    Window {
        kind,
        valid_from: None,
        valid_until: None,
        due_at: None,
    }
}

const NOW: &str = "2026-10-01T12:00:00Z";

fn phase(window: &Window, low_confidence: bool) -> f64 {
    let ranking = RankingTuning {
        phase_bonus: 1.0,
        phase_penalty: 1.0,
        ..RankingTuning::default()
    };
    phase_term(window, low_confidence, &TimeZone::UTC, at(NOW), &ranking)
}

#[test]
fn an_upcoming_bonus_starts_seven_days_out_and_grows_as_it_approaches() {
    let starting = |at: &str| Window {
        valid_from: Some(utc(at)),
        ..window(Kind::Event)
    };
    assert_eq!(phase(&starting("2026-10-10T12:00:00Z"), false), 0.0);
    let five = phase(&starting("2026-10-06T12:00:00Z"), false);
    let two = phase(&starting("2026-10-03T12:00:00Z"), false);
    let hour = phase(&starting("2026-10-01T13:00:00Z"), false);
    assert!(five > 0.0, "{five}");
    assert!(two > five, "{two} vs {five}");
    assert!(hour > two && hour <= 1.0, "{hour}");
}

#[test]
fn an_overdue_task_keeps_the_full_bonus_for_14_days_then_falls_to_zero_at_60() {
    let due = |at: &str| Window {
        due_at: Some(utc(at)),
        ..window(Kind::Task)
    };
    let close = |a: f64, b: f64| (a - b).abs() < 0.01;
    assert!(close(phase(&due("2026-09-21T12:00:00Z"), false), 1.0)); // 10 days
    // Halfway from 14 to 60 days is 37.
    assert!(close(phase(&due("2026-08-25T12:00:00Z"), false), 0.5));
    assert_eq!(phase(&due("2026-07-01T12:00:00Z"), false), 0.0); // 92 days
}

#[test]
fn an_ended_memory_has_no_term_for_7_days_then_a_penalty_rising_to_full_at_30() {
    let ended = |until: &str| Window {
        valid_from: Some(utc("2026-06-01T00:00:00Z")),
        valid_until: Some(utc(until)),
        ..window(Kind::State)
    };
    let close = |a: f64, b: f64| (a - b).abs() < 0.01;
    assert_eq!(phase(&ended("2026-09-28T12:00:00Z"), false), 0.0); // 3 days
    // Halfway from 7 to 30 days is 18.5.
    assert!(close(phase(&ended("2026-09-13T00:00:00Z"), false), -0.5));
    assert!(close(phase(&ended("2026-08-01T12:00:00Z"), false), -1.0)); // 61 days
}

#[test]
fn low_window_confidence_halves_the_phase_term_in_both_directions() {
    let overdue = Window {
        due_at: Some(utc("2026-09-21T12:00:00Z")),
        ..window(Kind::Task)
    };
    let ended = Window {
        valid_from: Some(utc("2026-06-01T00:00:00Z")),
        valid_until: Some(utc("2026-08-01T12:00:00Z")),
        ..window(Kind::State)
    };
    assert!((phase(&overdue, true) - phase(&overdue, false) / 2.0).abs() < 1e-9);
    assert!((phase(&ended, true) - phase(&ended, false) / 2.0).abs() < 1e-9);
    assert!(phase(&ended, true) < 0.0);
}

// Short follow-ups

#[test]
fn a_message_under_eight_words_borrows_the_previous_prefetch_query() {
    let previous = "When is the dentist appointment";
    let seven = "yes please book it for me now";
    assert_eq!(seven.split_whitespace().count(), 7);
    let borrowed = effective_query(seven, Some(previous));
    assert!(
        borrowed.contains(previous) && borrowed.contains(seven),
        "{borrowed}"
    );

    let eight = "yes please book it for me right now";
    assert_eq!(eight.split_whitespace().count(), 8);
    assert_eq!(effective_query(eight, Some(previous)), eight);
    assert_eq!(effective_query("yes", None), "yes");
}

#[test]
fn a_short_follow_up_finds_what_the_previous_query_asked_about() {
    let h = Harness::new();
    let dentist = h.insert(fact("Tim's dentist appointment is on Friday."));
    let alone = h.prefetch("s", "yes, book it");
    assert!(alone.injected.is_empty(), "{alone:?}");

    let followed = h.prefetch_after("s", "yes, book it", Some("dentist appointment Friday"));
    assert_eq!(followed.injected, vec![dentist]);
}

// Reranking against the conversation

/// Fake models and a gate floor of 1.0, with the reranker scoring against
/// the conversation.
fn in_conversation() -> Harness {
    Harness::with(
        1.0,
        "rerank_query = \"conversation\"",
        Arc::new(FakeReranker),
    )
}

#[test]
fn the_conversation_query_is_the_message_the_previous_message_and_the_reply() {
    let message = "go ahead and order one with that account please";
    assert_eq!(
        conversation_query(
            message,
            Some("Can you add batteries to the shopping doc?"),
            Some("Added them. Which account should I order with?"),
        ),
        "go ahead and order one with that account please\n\
         Can you add batteries to the shopping doc?\n\
         Added them. Which account should I order with?"
    );
    assert_eq!(conversation_query(message, None, None), message);
    assert_eq!(conversation_query(message, Some("  "), Some("")), message);
}

#[test]
fn the_conversation_query_takes_only_the_start_of_a_long_reply() {
    let reply = format!(
        "Your flight departs at nine. {}Tailword.",
        "There is more detail after this. ".repeat(400)
    );
    let query = conversation_query("remind me", Some("What's on?"), Some(&reply));
    assert!(
        query.starts_with("remind me\nWhat's on?\nYour flight departs at nine."),
        "{query}"
    );
    assert!(!query.contains("Tailword"), "{query}");
}

#[test]
fn the_conversation_query_takes_only_the_start_of_a_long_previous_message() {
    let previous = format!(
        "Can you check the Fastmail account? {}Tailword.",
        "Here is some more context. ".repeat(400)
    );
    let query = conversation_query("go ahead", Some(&previous), Some("Sure."));
    assert!(
        query.starts_with("go ahead\nCan you check the Fastmail account?"),
        "{query}"
    );
    assert!(query.ends_with("\nSure."), "{query}");
    assert!(!query.contains("Tailword"), "{query}");
}

/// Text with no space to cut back to, or in a script of multi-byte
/// characters, is still cut to a start made of whole characters.
#[test]
fn a_long_context_without_spaces_or_in_another_script_is_cut_to_whole_characters() {
    for text in [
        "a".repeat(2000),
        "東京".repeat(1000),
        "🙂".repeat(1000),
        format!("x{}", "é".repeat(1000)),
        "東京 ".repeat(1000),
    ] {
        let query = conversation_query("message", Some(&text), None);
        let (message, start) = query
            .split_once('\n')
            .expect("the message, then the context");
        assert_eq!(message, "message");
        assert!(
            !start.is_empty() && start.len() < text.len(),
            "{} of {} bytes",
            start.len(),
            text.len()
        );
        assert!(text.starts_with(start), "{start:?}");
    }
}

/// However long the previous message and the reply, the message still
/// reaches the reranker: the conversation's context gives way to it within
/// the reranker's 512 tokens. 300 characters of CJK are 300 tokens, so two
/// such parts alone are more than the pair holds.
#[test]
fn the_message_reaches_the_reranker_past_a_long_conversation() {
    let h = Harness::with(
        1.0,
        "rerank_query = \"conversation\"",
        Arc::new(TruncatingReranker),
    );
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let message = "pottery class schedule please";
    assert_eq!(h.prefetch("alone", message).injected, vec![pottery]);

    let context = "東京".repeat(150);
    let prefetch = h.prefetch_in_conversation("s", message, Some(&context), Some(&context));
    assert_eq!(prefetch.injected, vec![pottery], "{prefetch:?}");
}

/// A short follow-up still borrows the previous message for the retrievers,
/// and the conversation query holds that message once.
#[test]
fn a_short_follow_up_in_conversation_holds_the_previous_message_once() {
    let h = in_conversation();
    let dentist = h.insert(fact("Tim's dentist appointment is on Friday."));
    let scored = h
        .service
        .scored_prefetch(
            BANK,
            &PrefetchRequest {
                session_id: "s".into(),
                query: "yes, book it".into(),
                previous_query: Some("dentist appointment Friday".into()),
                previous_reply: Some("I can book it for Friday.".into()),
                block_id: None,
            },
        )
        .unwrap();
    assert_eq!(scored.query, "dentist appointment Friday\nyes, book it");
    assert_eq!(
        scored.rerank_query,
        "yes, book it\ndentist appointment Friday\nI can book it for Friday."
    );
    assert_eq!(scored.prefetch.injected, vec![dentist]);
}

/// By default, context can make a memory relevant even when a message
/// of eight words or more doesn't name its subject.
#[test]
fn by_default_the_reranker_sees_the_conversation() {
    let h = Harness::new();
    let passkey = h.insert(fact("Tim signs in to Fastmail with a passkey."));
    let prefetch = h.prefetch_in_conversation(
        "s",
        "go ahead and do that for me right now please",
        Some("Can you sign in to Fastmail for me?"),
        Some("Sure, signing in to Fastmail now."),
    );
    assert_eq!(prefetch.injected, vec![passkey], "{prefetch:?}");
}

/// Explicit message mode ignores context for a message of eight words
/// or more, preserving the message-only comparison.
#[test]
fn in_message_mode_the_reranker_sees_only_the_message() {
    let h = Harness::with(1.0, "rerank_query = \"message\"", Arc::new(FakeReranker));
    let _passkey = h.insert(fact("Tim signs in to Fastmail with a passkey."));
    let prefetch = h.prefetch_in_conversation(
        "s",
        "go ahead and do that for me right now please",
        Some("Can you sign in to Fastmail for me?"),
        Some("Sure, signing in to Fastmail now."),
    );
    assert!(prefetch.injected.is_empty(), "{prefetch:?}");
}

/// A message that doesn't name its subject finds the memory the previous
/// message makes relevant.
#[test]
fn reranking_against_the_conversation_finds_what_the_previous_message_named() {
    let h = in_conversation();
    let passkey = h.insert(fact("Tim signs in to Fastmail with a passkey."));
    let message = "go ahead and do that for me right now please";
    assert!(h.prefetch("alone", message).injected.is_empty());

    let prefetch = h.prefetch_in_conversation(
        "s",
        message,
        Some("Can you sign in to Fastmail for me?"),
        None,
    );
    assert_eq!(prefetch.injected, vec![passkey]);
}

/// The assistant's reply carries what the message leans on when the
/// previous message didn't name it either.
#[test]
fn reranking_against_the_conversation_finds_what_the_reply_named() {
    let h = in_conversation();
    let flight = h.insert(fact("Tim's flight to Wellington departs from gate four."));
    let message = "remind me two hours before that leaves so I can pack";
    let previous = Some("Anything on this week?");

    let without = h.prefetch_in_conversation("a", message, previous, None);
    assert!(without.injected.is_empty(), "{without:?}");

    let with = h.prefetch_in_conversation(
        "b",
        message,
        previous,
        Some("Your flight to Wellington departs Friday at nine."),
    );
    assert_eq!(with.injected, vec![flight]);
}

// Cleaning the query

/// The note Hermes' Discord gateway puts in front of a turn's message, with
/// a synthetic message id.
const DISCORD_NOTE: &str = "[Triggering message id: `100000000000000001` \u{2014} use as \
                            `message_id` for reply/react/pin via the discord tools.]";

#[test]
fn each_format_cleans_to_the_message() {
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
    ] {
        assert_eq!(clean_query(&raw), cleaned, "{raw:?}");
    }
}

#[test]
fn a_message_of_only_the_note_and_prefix_cleans_to_nothing() {
    for raw in [
        "[Sam] ".to_owned(),
        DISCORD_NOTE.to_owned(),
        format!("{DISCORD_NOTE}\n\n[Sam] "),
    ] {
        assert_eq!(clean_query(&raw), "", "{raw:?}");
    }
}

/// Prefetch recalls for the cleaned message, so the note's words don't make
/// a short follow-up long, and the log keeps the raw query beside it.
#[test]
fn a_prefetch_recalls_for_the_cleaned_query_and_logs_both() {
    let h = Harness::new();
    let dentist = h.insert(fact("Tim's dentist appointment is on Friday."));
    let raw = format!("{DISCORD_NOTE}\n\n[Sam] yes, book it");
    let prefetch = h.prefetch_after("s", &raw, Some("[Sam] dentist appointment Friday"));
    assert_eq!(prefetch.injected, vec![dentist]);
    assert_eq!(
        h.recall_queries(prefetch.recall_id),
        (
            "dentist appointment Friday\nyes, book it".to_owned(),
            Some(raw)
        )
    );
}

// Retrievers and clean-up

#[test]
fn the_entity_arm_finds_memories_linked_to_an_entity_the_query_names() {
    // A hundred memories under the query's own vector fill the vector arm,
    // and the target shares no word with the query, so only the entity arm
    // can bring it in.
    let h = Harness::with(1.0, "", Arc::new(KeywordReranker("greyhound")));
    let question = "How is Ana doing?";
    h.crowd(question, CANDIDATES_PER_ARM);
    let target = h.insert(fact("Someone adopted a greyhound."));
    let ana = h.entity("Ana Silva", &["Ana"]);
    h.link(target, ana);

    let recall = h.recall(query(question));
    assert_eq!(recall.results.first().map(|r| r.id), Some(target));
}

#[test]
fn the_entity_arm_skips_the_seeded_user() {
    let h = Harness::with(1.0, "", Arc::new(KeywordReranker("sourdough")));
    let question = "How is Tim doing?";
    h.crowd(question, CANDIDATES_PER_ARM);
    let baked = h.insert(fact("Someone baked sourdough."));
    h.link(baked, h.seeded("user"));

    let recall = h.recall(query(question));
    assert!(!ids(&recall).contains(&baked));
}

#[test]
fn a_refined_hit_shows_the_head_of_its_chain() {
    let h = Harness::with_floor(0.0);
    let old = h.insert(fact("Tim's bike is a Brompton."));
    let new = h.insert(fact("Tim's bike is a blue folding Brompton."));
    h.refine(old, new);

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
    let retracted = h.insert(Memory {
        retracted: true,
        ..fact("Tim's dentist appointment is on 8 October 2026.")
    });
    let hidden = h.insert(Memory {
        hidden: true,
        ..fact("Tim's dentist appointment is in Wellington.")
    });
    let kept = h.insert(fact("Tim's dentist is called Dr Ngata."));

    let recall = h.recall(query("dentist appointment"));
    assert!(ids(&recall).contains(&kept));
    for gone in [retracted, hidden] {
        assert!(!ids(&recall).contains(&gone));
        assert!(
            !h.prefetch("s", "dentist appointment")
                .injected
                .contains(&gone)
        );
    }
    let logged: i64 = h.one(
        "SELECT COUNT(*) FROM recall_results WHERE memory_id IN (?1, ?2)",
        (h.rowid(retracted), h.rowid(hidden)),
    );
    assert_eq!(logged, 0);
}

// The injection gate

#[test]
fn injection_gates_on_the_reranker_floor() {
    let h = Harness::new(); // floor 1.0: two shared words
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let one_word = h.insert(fact("The class was cancelled."));
    let _none = h.insert(fact("Tim owns a canoe."));

    let prefetch = h.prefetch("s", "pottery class schedule");
    assert_eq!(prefetch.injected, vec![pottery]);
    assert!(!prefetch.text.contains("cancelled"));
    assert!(!prefetch.injected.contains(&one_word));
}

#[test]
fn injection_takes_at_most_the_cap() {
    let h = Harness::with(1.0, "cap = 3", Arc::new(FakeReranker));
    for note in ["one", "two", "three", "four", "five"] {
        h.insert(fact(leak(format!("Pottery class note {note}."))));
    }
    let prefetch = h.prefetch("s", "pottery class");
    assert_eq!(prefetch.injected.len(), 3);
    assert_eq!(
        prefetch
            .text
            .lines()
            .filter(|l| l.starts_with("- "))
            .count(),
        3
    );
}

#[test]
fn injection_stays_within_the_token_budget() {
    const LONG: [&str; 4] = [
        "Tim's pottery class meets in the old church hall on Ponsonby Road, and the teacher asks everyone to bring an apron.",
        "Tim's pottery class covers wheel throwing for the first six weeks and glazing for the last two, with a kiln day at the end.",
        "Tim's pottery class costs two hundred dollars a term, which includes clay, glazes and three firings in the shared kiln.",
        "Tim's pottery class has eight students this term, two of whom have been going for years and help the beginners.",
    ];
    let h = Harness::with(1.0, "token_budget = 70", Arc::new(FakeReranker));
    for text in LONG {
        h.insert(fact(text));
    }
    let prefetch = h.prefetch("s", "pottery class");
    assert!(!prefetch.injected.is_empty());
    assert!(prefetch.injected.len() < LONG.len());
    assert!(
        estimate_tokens(&prefetch.text) <= 70,
        "{} tokens",
        estimate_tokens(&prefetch.text)
    );
}

#[test]
fn injection_drops_memories_below_tau_that_recall_still_finds() {
    let h = Harness::new();
    let faded = h.insert(Memory {
        significance: "trivial",
        observed_at: at("2021-01-01T00:00:00Z"),
        ..fact("Tim once tried a pottery class.")
    });
    let recall = h.recall(query("pottery class"));
    let found = recall.results.iter().find(|r| r.id == faded).unwrap();
    assert_eq!(found.strength, Band::Faded);
    assert!(!h.prefetch("s", "pottery class").injected.contains(&faded));
}

#[test]
fn phase_never_keeps_a_relevant_memory_out_of_injection() {
    let h = Harness::new();
    let trip = h.insert(Memory {
        kind: "event",
        observed_at: at("2026-07-01T00:00:00Z"),
        valid_from: Some((local("2026-07-20T00:00"), "day")),
        valid_until: Some((local("2026-08-01T00:00"), "day")),
        ..fact("Tim went on a pottery retreat in Nelson.")
    });
    assert!(h.prefetch("s", "pottery retreat").injected.contains(&trip));
}

#[test]
fn injected_memories_are_in_score_order() {
    let h = Harness::with_floor(0.0);
    let one = h.insert(fact("A pottery note."));
    let three = h.insert(fact("A pottery class schedule note."));
    let two = h.insert(fact("A pottery class note."));
    let prefetch = h.prefetch("s", "pottery class schedule");
    assert_eq!(prefetch.injected, vec![three, two, one]);
    let lines: Vec<&str> = prefetch.text.lines().skip(1).collect();
    assert_eq!(
        lines,
        vec![
            "- A pottery class schedule note.",
            "- A pottery class note.",
            "- A pottery note.",
        ]
    );
}

// The relevance scale

#[test]
fn the_floor_gates_on_the_raw_logit_whatever_the_relevance_scale() {
    // At 4.0, a gate on relevance would stop the two-word memory
    // (1.5 / 4 < 1.0), and at 0.25 it would pass the one-word one
    // (0.5 / 0.25 >= 1.0).
    for scale in [0.25, 4.0] {
        let h = Harness::with_scale(1.0, scale);
        let pottery = h.insert(fact("Tim takes a pottery class."));
        let one_word = h.insert(fact("The class was cancelled."));
        let scored = h
            .service
            .scored_prefetch(
                BANK,
                &PrefetchRequest {
                    session_id: "s".into(),
                    query: "pottery class schedule".into(),
                    previous_query: None,
                    previous_reply: None,
                    block_id: None,
                },
            )
            .unwrap();
        assert_eq!(scored.prefetch.injected, vec![pottery], "scale {scale}");
        // The labelling material shows the raw logit too.
        let logit = |memory: Uuid| {
            scored
                .candidates
                .iter()
                .find(|c| c.memory == memory)
                .unwrap()
                .logit
        };
        assert_eq!(logit(pottery), Some(1.5), "scale {scale}");
        assert_eq!(logit(one_word), Some(0.5), "scale {scale}");
    }
}

#[test]
fn the_relevance_scale_leaves_the_phase_term_alone_in_injection() {
    // A long-past memory sharing four query words, against a current one
    // sharing two. At scale 1.0 the extra words outweigh the full phase
    // penalty (3.5 - 1.0 > 1.5). At 10.0 they don't (0.35 - 1.0 < 0.15),
    // unless the penalty were scaled too.
    let order = |scale: f64| {
        let h = Harness::with_scale(0.0, scale);
        let past = h.insert(Memory {
            kind: "event",
            valid_from: Some((local("2026-07-20T00:00"), "day")),
            valid_until: Some((local("2026-08-01T00:00"), "day")),
            ..fact("Tim's pottery class schedule changed at the studio.")
        });
        let current = h.insert(fact("Tim's pottery class meets weekly."));
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
        let h = Harness::with_scale(1.0, scale);
        let faded = h.insert(Memory {
            significance: "trivial",
            observed_at: at("2021-01-01T00:00:00Z"),
            ..fact("Tim's pottery class schedule note.")
        });
        let fresh = h.insert(Memory {
            observed_at: h.service.now(),
            ..fact("Tim's pottery class note.")
        });
        (
            ids(&h.recall(query("pottery class schedule"))),
            faded,
            fresh,
        )
    };
    let (ranked, faded, fresh) = order(1.0);
    assert_eq!(ranked[..2], [faded, fresh]);
    let (ranked, faded, fresh) = order(100.0);
    assert_eq!(ranked[..2], [fresh, faded]);
}

// The reranker deadline

#[test]
fn a_late_reranker_injects_nothing_and_still_logs_the_prefetch() {
    let deadline = Duration::from_millis(100);
    let h = Harness::with(1.0, "", Arc::new(SlowReranker(Duration::from_secs(2))))
        .with_deadline(deadline);
    h.insert(fact("Tim takes a pottery class."));

    let started = Instant::now();
    let prefetch = h.prefetch("s", "pottery class schedule");
    assert!(
        started.elapsed() < Duration::from_millis(1500),
        "waited for the reranker"
    );
    assert!(!prefetch.reranked);
    assert!(prefetch.injected.is_empty());
    assert!(prefetch.text.is_empty());

    let (kind, session, _, _, latency_ms, _) = h.recall_row(prefetch.recall_id);
    assert_eq!(kind, "prefetch");
    assert_eq!(session.as_deref(), Some("s"));
    assert!(latency_ms >= deadline.as_millis() as i64, "{latency_ms} ms");
    assert!(
        h.results(prefetch.recall_id)
            .iter()
            .all(|(_, injected)| !injected)
    );

    // Nothing is pending, so the turn commits nothing.
    h.sync_turn("s", Some(prefetch.recall_id.to_string()));
    assert!(h.in_context("s").is_empty());
}

// The injection format

#[test]
fn the_injection_has_a_recall_time_header_and_one_line_per_memory() {
    let h = Harness::new();
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let prefetch = h.prefetch("s", "pottery class schedule");
    let mut lines = prefetch.text.lines();
    // 07:00 UTC is 20:00 on Thursday 1 October in the bank's timezone.
    assert_eq!(lines.next(), Some("Recalled Thu 1 Oct 20:00"));
    assert_eq!(lines.next(), Some("- Tim takes a pottery class."));
    assert_eq!(lines.next(), None);
    assert!(!prefetch.text.contains(&pottery.to_string()));
    assert!(!prefetch.text.contains(&prefetch.recall_id.to_string()));
}

#[test]
fn annotations_give_absolute_dates() {
    let h = Harness::with_floor(0.0);
    let line = |memory: Memory| {
        let content = memory.content;
        h.insert(memory);
        content
    };
    let dentist = line(Memory {
        kind: "event",
        valid_from: Some((local("2026-10-03T15:00"), "minute")),
        ..fact("Tim has a dentist appointment on 3 October 2026 at 15:00.")
    });
    let passport = line(Memory {
        kind: "task",
        due_at: Some((local("2026-09-30T00:00"), "day")),
        ..fact("Tim needs to renew his passport.")
    });
    let berlin = line(Memory {
        kind: "event",
        observed_at: at("2026-09-05T00:00:00Z"),
        valid_from: Some((local("2026-09-10T00:00"), "day")),
        valid_until: Some((local("2026-09-12T00:00"), "day")),
        ..fact("Tim was on a trip to Berlin.")
    });
    let yoga = line(Memory {
        kind: "recurring",
        recurrence_text: Some("every Tuesday"),
        ..fact("Tim goes to yoga every Tuesday.")
    });
    // Said exactly four days before now, with a three-day volatility, so
    // its confidence is well below 0.9.
    let lisbon = line(Memory {
        kind: "state",
        volatility: Some("days"),
        observed_at: at("2026-09-27T07:00:00Z"),
        ..fact("Tim is staying in Lisbon.")
    });
    // Said today with a years-long volatility: confident, so no age.
    let tax = line(Memory {
        kind: "state",
        volatility: Some("years"),
        observed_at: at("2026-10-01T06:00:00Z"),
        ..fact("Tim is working on the tax return.")
    });
    let sister = line(Memory {
        kind: "event",
        valid_from: Some((local("2026-11-01T00:00"), "month")),
        low_confidence: true,
        ..fact("Tim's sister visits in November 2026.")
    });

    let prefetch = h.prefetch("s", "dentist passport Berlin yoga Lisbon tax sister");
    let lines: Vec<&str> = prefetch.text.lines().collect();
    assert_eq!(lines.len(), 8, "{}", prefetch.text);
    let has = |expected: &str| {
        assert!(
            lines.contains(&expected),
            "missing {expected:?} in\n{}",
            prefetch.text
        )
    };
    // 3 October 2026 is a Saturday, and 27 September a Sunday.
    has(&format!("- {dentist} [upcoming Sat 3 Oct 15:00]"));
    // Every annotation date carries its weekday: 30 September
    // 2026 is a Wednesday and 12 September a Saturday.
    has(&format!("- {passport} [overdue since Wed 30 Sep]"));
    has(&format!("- {berlin} [ended Sat 12 Sep]"));
    has(&format!("- {yoga} [recurring: every Tuesday]"));
    has(&format!("- {lisbon} [observed 4 days ago, Sun 27 Sep]"));
    has(&format!("- {tax}"));
    let sister_line = lines.iter().find(|l| l.contains(sister)).unwrap();
    assert!(sister_line.contains("date uncertain"), "{sister_line}");
}

// The in-context skip and per-session state

#[test]
fn a_committed_injection_isnt_injected_again_in_that_session() {
    let h = Harness::new();
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let first = h.prefetch("s", "pottery class schedule");
    assert_eq!(first.injected, vec![pottery]);
    h.sync_turn("s", Some(first.recall_id.to_string()));
    assert_eq!(h.in_context("s"), vec![pottery]);

    let again = h.prefetch("s", "pottery class schedule");
    assert!(again.injected.is_empty());
    assert!(again.text.is_empty());
    // Another session can't see it, so it's injected there.
    assert_eq!(
        h.prefetch("t", "pottery class schedule").injected,
        vec![pottery]
    );
}

#[test]
fn a_turn_without_a_recall_id_commits_nothing() {
    // A turn with no id can't be matched to any pending injection: syncs
    // can arrive after the next prefetch, so it leaves them all
    // pending, and only an echo of an injection's own id commits it.
    let h = Harness::new();
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let first = h.prefetch("s", "pottery class schedule");
    h.sync_turn("s", None);
    assert!(h.in_context("s").is_empty());
    h.sync_turn("s", Some(first.recall_id.to_string()));
    assert_eq!(h.in_context("s"), vec![pottery]);
}

#[test]
fn a_turn_with_an_unknown_recall_id_commits_nothing() {
    let h = Harness::new();
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let first = h.prefetch("s", "pottery class schedule");
    h.sync_turn("s", Some(Uuid::from_u128(7).to_string()));
    assert!(h.in_context("s").is_empty());
    h.sync_turn("s", Some(first.recall_id.to_string()));
    assert_eq!(h.in_context("s"), vec![pottery]);
}

#[test]
fn interleaved_syncs_commit_each_acknowledged_injection() {
    // a turn's sync can arrive after the next prefetch.
    let h = Harness::new();
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let canoe = h.insert(fact("Tim paddles his canoe on Sundays."));
    let a = h.prefetch("s", "pottery class schedule");
    let b = h.prefetch("s", "canoe paddles weekend");
    assert_eq!(a.injected, vec![pottery]);
    assert_eq!(b.injected, vec![canoe]);

    h.sync_turn("s", Some(a.recall_id.to_string()));
    assert_eq!(h.in_context("s"), vec![pottery]);
    h.sync_turn("s", Some(b.recall_id.to_string()));
    let in_context = h.in_context("s");
    assert!(
        in_context.contains(&pottery) && in_context.contains(&canoe),
        "{in_context:?}"
    );
}

#[test]
fn a_resent_turn_leaves_the_pending_injection_alone() {
    // Ingest is idempotent: a turn resent from the spool
    // or retried is a duplicate and settles nothing.
    let h = Harness::new();
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let mut earlier = turn("s", EARLIER, "An earlier message.", None);
    earlier.message_at = h.service.now() - SignedDuration::from_mins(10);
    h.service.ingest_turn(BANK, &earlier).unwrap();

    let pending = h.prefetch("s", "pottery class schedule");
    let resent = h.service.ingest_turn(BANK, &earlier).unwrap();
    assert_eq!(resent.outcome, Outcome::Duplicate);
    h.sync_turn("s", Some(pending.recall_id.to_string()));
    assert_eq!(h.in_context("s"), vec![pottery]);
}

#[test]
fn clearing_a_session_lets_its_memories_be_injected_again() {
    let h = Harness::new();
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let first = h.prefetch("s", "pottery class schedule");
    h.sync_turn("s", Some(first.recall_id.to_string()));
    h.service.clear_session(BANK, "s").unwrap();
    assert!(h.in_context("s").is_empty());
    assert_eq!(
        h.prefetch("s", "pottery class schedule").injected,
        vec![pottery]
    );
}

#[test]
fn recall_tool_results_join_the_in_context_set() {
    let h = Harness::new();
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let recall = h.recall(RecallRequest {
        session_id: Some("s".into()),
        ..query("pottery class")
    });
    assert_eq!(ids(&recall), vec![pottery]);
    assert_eq!(h.in_context("s"), vec![pottery]);
    assert!(
        h.prefetch("s", "pottery class schedule")
            .injected
            .is_empty()
    );
    // Explicit recall doesn't skip what's in context.
    assert_eq!(
        ids(&h.recall(RecallRequest {
            session_id: Some("s".into()),
            ..query("pottery class")
        })),
        vec![pottery]
    );
}

#[test]
fn the_idle_timeout_is_tunable() {
    let h = Harness::with(
        1.0,
        "\n---\n[sessions]\nin_context_idle_days = 1\n",
        Arc::new(FakeReranker),
    );
    let pottery = h.insert(fact("Tim takes a pottery class."));
    for session in ["kept", "expired"] {
        let prefetch = h.prefetch(session, "pottery class schedule");
        h.sync_turn(session, Some(prefetch.recall_id.to_string()));
    }
    h.clock.advance(SignedDuration::from_hours(23));
    assert_eq!(h.in_context("kept"), vec![pottery]);
    h.clock.advance(SignedDuration::from_hours(2));
    assert!(h.in_context("expired").is_empty());
    assert_eq!(
        h.prefetch("expired", "pottery class schedule").injected,
        vec![pottery]
    );
}

// The recall log and accesses: a recall is logged, never counted as an access

#[test]
fn a_prefetch_logs_one_row_with_what_was_injected() {
    let h = Harness::new();
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let canoe = h.insert(fact("Tim owns a canoe."));
    let before: i64 = h.one("SELECT COUNT(*) FROM recalls", []);

    let prefetch = h.prefetch("s", "pottery class schedule");
    let after: i64 = h.one("SELECT COUNT(*) FROM recalls", []);
    assert_eq!(after, before + 1);
    let (kind, session, turn, query, latency_ms, recalled_at) = h.recall_row(prefetch.recall_id);
    assert_eq!(kind, "prefetch");
    assert_eq!(session.as_deref(), Some("s"));
    assert_eq!(turn, h.one::<i64, _>("SELECT turns FROM banks", []));
    assert_eq!(query, "pottery class schedule");
    assert!(latency_ms >= 0);
    assert_eq!(recalled_at, micros(at(START)));

    let results = h.results(prefetch.recall_id);
    let injected: Vec<Uuid> = results
        .iter()
        .filter(|(_, injected)| *injected)
        .map(|(memory, _)| *memory)
        .collect();
    assert_eq!(injected, prefetch.injected);
    assert_eq!(injected, vec![pottery]);
    // The candidate that failed the gate came back without being injected.
    assert!(results.contains(&(canoe, false)));
}

#[test]
fn recalling_never_writes_an_access() {
    let h = Harness::new();
    h.insert(fact("Tim takes a pottery class."));
    let faded = h.insert(Memory {
        significance: "trivial",
        observed_at: at("2021-01-01T00:00:00Z"),
        ..fact("Tim once tried a pottery wheel.")
    });
    let before = h.accesses();

    let prefetch = h.prefetch("s", "pottery class schedule");
    h.sync_turn("s", Some(prefetch.recall_id.to_string()));
    let recall = h.recall(RecallRequest {
        session_id: Some("s".into()),
        ..query("pottery wheel")
    });
    assert!(ids(&recall).contains(&faded));
    h.recall(query("pottery class"));

    assert_eq!(h.accesses(), before);
    // A faded memory stays faded however often it's recalled.
    let again = h.recall(query("pottery wheel"));
    let found = again.results.iter().find(|r| r.id == faded).unwrap();
    assert_eq!(found.strength, Band::Faded);
}

// Explicit recall

#[test]
fn a_fresh_memory_is_strong() {
    let h = Harness::new();
    let fresh = h.insert(Memory {
        significance: "trivial",
        observed_at: h.service.now(),
        ..fact("Tim bought a pottery apron.")
    });
    let recall = h.recall(query("pottery apron"));
    let found = recall.results.iter().find(|r| r.id == fresh).unwrap();
    assert_eq!(found.strength, Band::Strong);
}

#[test]
fn recall_filters_by_phase() {
    let h = Harness::with_floor(0.0);
    let upcoming = h.insert(Memory {
        kind: "event",
        valid_from: Some((local("2026-10-10T00:00"), "day")),
        ..fact("Tim's garden tour is on 10 October 2026.")
    });
    let past = h.insert(Memory {
        kind: "event",
        valid_from: Some((local("2026-09-10T00:00"), "day")),
        ..fact("Tim's garden party was on 10 September 2026.")
    });
    let current = h.insert(fact("Tim's garden has a lemon tree."));
    // An overdue task is neither upcoming nor past; the `current` filter is
    // the only one that reaches it short of `any`.
    let overdue = h.insert(Memory {
        kind: "task",
        due_at: Some((local("2026-09-30T00:00"), "day")),
        ..fact("Tim needs to weed the garden.")
    });

    let only = |filter: PhaseFilter| {
        ids(&h.recall(RecallRequest {
            phase: filter,
            ..query("garden")
        }))
    };
    assert_eq!(only(PhaseFilter::Upcoming), vec![upcoming]);
    assert_eq!(only(PhaseFilter::Past), vec![past]);
    let now = only(PhaseFilter::Current);
    assert!(now.contains(&current));
    assert!(now.contains(&overdue));
    assert!(!now.contains(&upcoming) && !now.contains(&past));
    assert_eq!(only(PhaseFilter::Any).len(), 4);
}

#[test]
fn recall_filters_by_entity_taking_the_union_of_its_matches() {
    let h = Harness::with_floor(0.0);
    let lee = h.entity("Sam Lee", &["Sam"]);
    let ortiz = h.entity("Sam Ortiz", &["Sam"]);
    let first = h.insert(fact("Sam Lee sent the garden photos."));
    let second = h.insert(fact("Sam Ortiz lent Tim a garden fork."));
    let neither = h.insert(fact("Tim's garden has a lemon tree."));
    h.link(first, lee);
    h.link(second, ortiz);

    let found = ids(&h.recall(RecallRequest {
        entity: Some("Sam".into()),
        ..query("garden")
    }));
    assert!(
        found.contains(&first) && found.contains(&second),
        "{found:?}"
    );
    assert!(!found.contains(&neither));
    let nobody = h.recall(RecallRequest {
        entity: Some("Nobody".into()),
        ..query("garden")
    });
    assert!(nobody.results.is_empty());
}

#[test]
fn a_said_range_matches_observed_at() {
    let h = Harness::with_floor(0.0);
    h.insert(Memory {
        observed_at: at("2026-09-10T00:00:00Z"),
        ..fact("Tim's garden has a lemon tree.")
    });
    let later = h.insert(Memory {
        observed_at: at("2026-09-20T00:00:00Z"),
        ..fact("Tim's garden has a fig tree.")
    });
    let recall = h.recall(RecallRequest {
        from: Some(at("2026-09-15T00:00:00Z")),
        to: Some(at("2026-09-25T00:00:00Z")),
        on: On::Said,
        ..query("garden tree")
    });
    assert_eq!(ids(&recall), vec![later]);
}

#[test]
fn a_happened_range_matches_the_window_widening_low_confidence_by_a_unit() {
    let h = Harness::with_floor(0.0);
    let event = |day: &str, low_confidence: bool, content: &'static str| {
        h.insert(Memory {
            kind: "event",
            valid_from: Some((local(&format!("{day}T00:00")), "day")),
            low_confidence,
            ..fact(content)
        })
    };
    let inside = event("2026-10-06", false, "Tim's garden club meets on 6 October.");
    let day_before = event("2026-10-05", false, "Tim's garden fair is on 5 October.");
    let unsure = event(
        "2026-10-05",
        true,
        "Tim's garden visit is around 5 October.",
    );

    let recall = h.recall(RecallRequest {
        from: Some(local("2026-10-06T00:00")),
        to: Some(local("2026-10-06T23:00")),
        ..query("garden")
    });
    let found = ids(&recall);
    assert!(found.contains(&inside));
    assert!(
        !found.contains(&day_before),
        "a confident window outside the range is dropped"
    );
    assert!(
        found.contains(&unsure),
        "a low-confidence window is widened by one day"
    );
}

#[test]
fn a_fact_matches_a_happened_range_only_through_a_stated_start_inside_it() {
    let h = Harness::with_floor(0.0);
    let started = h.insert(Memory {
        valid_from: Some((local("2026-10-06T00:00"), "day")),
        ..fact("Tim joined the garden club.")
    });
    let earlier = h.insert(Memory {
        valid_from: Some((local("2026-01-01T00:00"), "day")),
        ..fact("Tim's garden got a new shed.")
    });
    let undated = h.insert(fact("Tim's garden has a lemon tree."));

    let found = ids(&h.recall(RecallRequest {
        from: Some(local("2026-10-01T00:00")),
        to: Some(local("2026-10-10T00:00")),
        ..query("garden")
    }));
    assert!(found.contains(&started));
    assert!(!found.contains(&earlier));
    assert!(!found.contains(&undated));
}

#[test]
fn an_open_start_reaches_back_to_any_range_before_the_end() {
    // Either end of a window can be open (CONTEXT.md), and `happened`
    // filters the window, not when it was said. Said on 15
    // September with an end of 12 September and no stated start, this state
    // held on 11 and 12 September.
    let h = Harness::with_floor(0.0);
    let berlin = h.insert(Memory {
        kind: "state",
        observed_at: at("2026-09-15T00:00:00Z"),
        valid_until: Some((local("2026-09-12T00:00"), "day")),
        ..fact("Tim was staying in Berlin.")
    });
    let found = ids(&h.recall(RecallRequest {
        from: Some(local("2026-09-11T00:00")),
        to: Some(local("2026-09-12T23:00")),
        ..query("Berlin")
    }));
    assert_eq!(found, vec![berlin]);
}

#[test]
fn an_ended_fact_matches_a_happened_range_inside_its_window() {
    // The start-only rule is for a fact with no window; once a
    // fact is ended, its window is bounded and history can find it.
    let h = Harness::with_floor(0.0);
    let acme = h.insert(Memory {
        valid_from: Some((local("2026-03-01T00:00"), "day")),
        valid_until: Some((local("2026-06-01T00:00"), "day")),
        ..fact("Tim works at Acme.")
    });
    let found = ids(&h.recall(RecallRequest {
        from: Some(local("2026-05-10T00:00")),
        to: Some(local("2026-05-20T00:00")),
        ..query("Tim works at Acme")
    }));
    assert_eq!(found, vec![acme]);
}

#[test]
fn a_timed_out_reranker_call_leaves_no_queued_inference() {
    // The production reranker runs one inference at a time behind a mutex.
    // A caller that can't start before its deadline falls back without
    // queueing work behind the one running, and once that finishes the
    // reranker serves again. The gate makes this independent of timing.
    let (gated, arrived, answered) = GatedReranker::new();
    let h = Harness::with(1.0, "", gated.clone()).with_deadline(Duration::from_millis(50));
    let pottery = h.insert(fact("Tim takes a pottery class."));

    let first = h.prefetch("s", "pottery class schedule");
    assert!(!first.reranked);
    arrived
        .recv_timeout(Duration::from_secs(5))
        .expect("the first call reached the reranker");

    // While it's stuck: more prefetches, at once and one after another, and
    // an explicit recall.
    std::thread::scope(|scope| {
        let calls: Vec<_> = ["a", "b", "c"]
            .into_iter()
            .map(|session| {
                let h = &h;
                scope.spawn(move || h.prefetch(session, "pottery class schedule"))
            })
            .collect();
        for call in calls {
            let prefetch = call.join().unwrap();
            assert!(!prefetch.reranked && prefetch.injected.is_empty());
        }
    });
    assert!(!h.prefetch("d", "pottery class schedule").reranked);
    let recall = h.recall(query("pottery class"));
    assert!(!recall.reranked);
    assert_eq!(ids(&recall), vec![pottery]);
    assert_eq!(
        gated.entered.load(Ordering::SeqCst),
        1,
        "timed-out calls queued inference behind the running one"
    );

    // Once the stuck call answers, the reranker is used again.
    gated.open();
    answered
        .recv_timeout(Duration::from_secs(5))
        .expect("the first call answered");
    let recovered = (0..100)
        .map(|_| h.prefetch("e", "pottery class schedule"))
        .find(|prefetch| prefetch.reranked)
        .expect("the reranker serves again after the stuck call");
    assert_eq!(recovered.injected, vec![pottery]);
}

#[test]
fn recall_returns_ten_results_unless_asked_for_up_to_thirty() {
    let h = Harness::with_floor(0.0);
    for n in 1..=35 {
        h.insert(fact(leak(format!("Garden note {n}."))));
    }
    assert_eq!(h.recall(query("garden note")).results.len(), 10);
    let asked = |limit: usize| {
        h.recall(RecallRequest {
            limit: Some(limit),
            ..query("garden note")
        })
        .results
        .len()
    };
    assert_eq!(asked(3), 3);
    assert_eq!(asked(30), 30);
}

// A queued turn's in-context set
//
// Extraction checks a turn for use against the memories the agent could see
// when it wrote the reply: the session's in-context set as the turn's sync
// left it. The bank's worker reaches the turn later, after any chunks ahead
// of it, so what happens to the session in between (a clear on compaction,
// a later recall, a restart) must not change which memories the turn is
// credited with using. Each test queues the turn and extracts it only after
// the session has moved on.

impl Harness {
    /// The daemon restarting: the service and its in-memory sessions go,
    /// the store and its extraction queue stay.
    fn restart(self) -> Self {
        let Self {
            service,
            clock,
            chunk,
            _dir,
        } = self;
        let tuning = service.tuning().clone();
        let models = service.models().unwrap().clone();
        drop(service);
        let store = Store::open(&_dir.0, OpenOptions::default(), clock.clone()).unwrap();
        let service = Service::with_models(clock.clone(), store, tuning, models).unwrap();
        Self {
            service,
            clock,
            chunk,
            _dir,
        }
    }

    /// Extracts the head of the queue, as the bank's worker does, with a
    /// call 1 that makes no claims and judges `used` (in-context handles)
    /// used. The LLM is returned so a test can read what call 1 was shown.
    fn extract_next(&self, used: &[&str]) -> FakeLlm {
        let llm = FakeLlm::scripted(
            "fake-llm",
            vec![serde_json::json!({"claims": [], "used_injected_ids": used})],
        );
        let extracted = self.service.extract_next(BANK, &llm).unwrap();
        assert!(extracted.is_some(), "nothing was queued");
        llm
    }

    /// `used` accesses on `memory`.
    fn used(&self, memory: Uuid) -> i64 {
        self.one(
            "SELECT COUNT(*) FROM accesses WHERE memory_id = ?1 AND kind = 'used'",
            [self.rowid(memory)],
        )
    }
}

/// A harness with the pottery memory injected in session `s` and the turn
/// that echoes the injection queued for extraction.
fn queued_turn_with_pottery_in_context() -> (Harness, Uuid) {
    let h = Harness::new();
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let prefetch = h.prefetch("s", "pottery class schedule");
    assert_eq!(prefetch.injected, vec![pottery]);
    h.sync_turn("s", Some(prefetch.recall_id.to_string()));
    assert_eq!(h.in_context("s"), vec![pottery]);
    (h, pottery)
}

#[test]
fn a_session_cleared_before_extraction_keeps_the_turns_in_context_set() {
    // Hermes compacts the session after the turn and before the worker
    // reaches it. The reply was still written with the memory in view.
    let (h, pottery) = queued_turn_with_pottery_in_context();
    h.service.clear_session(BANK, "s").unwrap();

    let llm = h.extract_next(&["m1"]);
    assert!(
        llm.requests()[0].user.contains("pottery class"),
        "call 1 wasn't shown the turn's in-context memory"
    );
    assert_eq!(
        h.used(pottery),
        1,
        "the turn's use of its injection was lost"
    );
}

#[test]
fn a_recall_after_the_turn_isnt_in_the_turns_in_context_set() {
    // A later turn's recall adds to the session; the queued turn's reply
    // was written before it and can't have used what it returned.
    let (h, pottery) = queued_turn_with_pottery_in_context();
    let canoe = h.insert(fact("Tim paddles his canoe on Sundays."));
    let later = h.recall(RecallRequest {
        session_id: Some("s".into()),
        query: "canoe paddles weekend".into(),
        ..RecallRequest::default()
    });
    assert_eq!(later.results[0].id, canoe);
    assert_eq!(h.in_context("s"), vec![pottery, canoe]);

    // m2 is no handle of the turn's own set, so it can't credit anything.
    let llm = h.extract_next(&["m2"]);
    assert!(
        !llm.requests()[0].user.contains("canoe"),
        "call 1 was shown a memory recalled after the turn"
    );
    assert_eq!(h.used(canoe), 0, "a later recall was credited to the turn");
    assert_eq!(h.used(pottery), 0);
}

#[test]
fn a_turn_queued_across_a_restart_keeps_its_in_context_set() {
    // SIGTERM finishes only the chunk in flight; a turn behind it is
    // extracted by the next daemon, whose sessions start empty.
    let (h, pottery) = queued_turn_with_pottery_in_context();
    let h = h.restart();
    assert!(h.in_context("s").is_empty());

    let llm = h.extract_next(&["m1"]);
    assert!(
        llm.requests()[0].user.contains("pottery class"),
        "call 1 wasn't shown the turn's in-context memory after the restart"
    );
    assert_eq!(
        h.used(pottery),
        1,
        "the turn's use was lost across the restart"
    );
}

// Explain: the pipeline's working for one query, with no side effects

impl Harness {
    fn explain(&self, request: ExplainRequest) -> Explain {
        self.service.explain(BANK, &request).unwrap()
    }

    /// The rows a recall or a session's state would leave: recalls, their
    /// results, accesses and session blocks.
    fn recall_traces(&self) -> [i64; 4] {
        ["recalls", "recall_results", "accesses", "session_blocks"]
            .map(|table| self.one(&format!("SELECT COUNT(*) FROM {table}"), []))
    }
}

/// The explain request for the same query and filters as `request`.
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

fn explain_injection(message: &str) -> ExplainRequest {
    ExplainRequest::Injection(ExplainInjection {
        query: message.into(),
        ..ExplainInjection::default()
    })
}

fn included(explain: &Explain) -> Vec<Uuid> {
    explain
        .candidates
        .iter()
        .filter(|c| c.included)
        .map(|c| c.id)
        .collect()
}

fn explained(explain: &Explain, memory: Uuid) -> &Explained {
    explain
        .candidates
        .iter()
        .find(|c| c.id == memory)
        .unwrap_or_else(|| panic!("{memory} isn't a candidate: {explain:#?}"))
}

#[test]
fn explaining_writes_no_recall_or_access_and_leaves_sessions_alone() {
    let h = Harness::new();
    let pottery = h.insert(fact("Tim takes a pottery class."));
    h.insert(fact("Tim owns a canoe."));
    // Session s has pottery in context; session t holds it pending.
    let committed = h.prefetch("s", "pottery class schedule");
    h.sync_turn("s", Some(committed.recall_id.to_string()));
    let pending = h.prefetch("t", "pottery class schedule");
    assert_eq!(pending.injected, vec![pottery]);
    let request = query("pottery class");
    // Recall can include lower-scoring hits too; take its baseline before
    // the side-effect snapshot, since a real recall writes a log row.
    let returned = ids(&h.recall(request.clone()));
    assert_eq!(returned[0], pottery);
    let before = h.recall_traces();

    let recall = h.explain(explain_recall(&request));
    let injection = h.explain(explain_injection("pottery class schedule"));

    assert_eq!(included(&recall), returned);
    // There's no session, so what s has in context is injected anyway.
    assert_eq!(
        injection.injection.as_ref().unwrap().injected,
        vec![pottery]
    );
    assert_eq!(h.recall_traces(), before);
    assert_eq!(h.in_context("s"), vec![pottery]);
    assert!(h.in_context("t").is_empty());
    // t's pending injection is still the one its turn commits.
    h.sync_turn("t", Some(pending.recall_id.to_string()));
    assert_eq!(h.in_context("t"), vec![pottery]);
}

#[test]
fn explaining_a_recall_includes_what_recall_returns_in_its_order() {
    let h = Harness::with_floor(0.0);
    let lee = h.entity("Sam Lee", &["Sam"]);
    h.insert(Memory {
        kind: "event",
        valid_from: Some((local("2026-10-10T00:00"), "day")),
        ..fact("Tim's garden tour is on 10 October 2026.")
    });
    h.insert(Memory {
        kind: "event",
        observed_at: at("2026-09-12T00:00:00Z"),
        valid_from: Some((local("2026-09-10T00:00"), "day")),
        ..fact("Tim's garden party was on 10 September 2026.")
    });
    let shared = h.insert(fact("Tim shares the garden with Sam."));
    h.link(shared, lee);
    h.insert(Memory {
        significance: "trivial",
        observed_at: at("2021-01-01T00:00:00Z"),
        ..fact("Tim once grew tomatoes in the garden.")
    });
    h.insert(Memory {
        kind: "task",
        due_at: Some((local("2026-09-30T00:00"), "day")),
        ..fact("Tim needs to weed the garden.")
    });
    h.insert(fact("Tim's garden has a lemon tree and a garden shed."));

    for request in [
        query("garden"),
        RecallRequest {
            limit: Some(2),
            ..query("garden")
        },
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

    // What the limit cut is listed under the results, saying so.
    let explain = h.explain(explain_recall(&RecallRequest {
        limit: Some(2),
        ..query("garden")
    }));
    let cut: Vec<&Explained> = explain.candidates.iter().filter(|c| !c.included).collect();
    assert!(!cut.is_empty(), "{explain:#?}");
    assert!(cut.iter().all(|c| c.reason == Some(Cut::OverLimit)));
    assert!(explain.candidates[..2].iter().all(|c| c.included));
}

#[test]
fn explaining_an_injection_injects_what_a_fresh_session_prefetch_would() {
    let h = Harness::with(1.0, "cap = 2", Arc::new(FakeReranker));
    for note in ["one", "two", "three"] {
        h.insert(fact(leak(format!("Pottery class note {note}."))));
    }
    h.insert(fact("A pottery class schedule note."));
    h.insert(fact("The class was cancelled."));
    h.insert(Memory {
        significance: "trivial",
        observed_at: at("2021-01-01T00:00:00Z"),
        ..fact("Tim once tried a pottery class.")
    });

    for (session, message, previous, reply) in [
        ("fresh-1", "pottery class schedule", None, None),
        (
            "fresh-2",
            "and the notes?",
            Some("pottery class"),
            Some("The pottery class meets on Tuesdays."),
        ),
    ] {
        let explain = h.explain(ExplainRequest::Injection(ExplainInjection {
            query: message.into(),
            previous_query: previous.map(Into::into),
            previous_reply: reply.map(Into::into),
        }));
        let prefetch = h.prefetch_in_conversation(session, message, previous, reply);
        assert!(!prefetch.injected.is_empty(), "{message}");

        let injection = explain.injection.clone().unwrap();
        assert_eq!(injection.injected, prefetch.injected, "{message}");
        assert_eq!(injection.text, prefetch.text, "{message}");
        assert_eq!(injection.tokens, estimate_tokens(&prefetch.text));
        assert_eq!(included(&explain), prefetch.injected, "{message}");
        assert_eq!(explain.mode, ExplainMode::Injection);
    }
}

#[test]
fn an_explained_injection_says_why_each_candidate_was_left_out() {
    let h = Harness::with(1.0, "cap = 1", Arc::new(FakeReranker));
    // FakeReranker: shared query words minus a half; the floor is 1.0.
    let best = h.insert(fact("A pottery class schedule note.")); // 2.5
    let capped = h.insert(fact("A pottery class note.")); // 1.5
    let weak = h.insert(fact("The class was cancelled.")); // 0.5
    let faded = h.insert(Memory {
        significance: "trivial",
        observed_at: at("2021-01-01T00:00:00Z"),
        ..fact("Tim once tried a pottery class schedule.")
    });

    let explain = h.explain(explain_injection("pottery class schedule"));
    assert!(explain.reranked);
    assert_eq!(explain.query, "pottery class schedule");

    let shown = explained(&explain, best);
    assert!(shown.included);
    assert_eq!(shown.reason, None);
    assert_eq!(shown.logit, Some(2.5));
    assert_eq!(shown.score.unwrap().relevance, 2.5);
    assert!(shown.rrf_rank.is_some());
    assert!(
        shown
            .arms
            .iter()
            .any(|arm| arm.arm == Arm::Bm25 && arm.rank.is_some()),
        "{shown:#?}"
    );

    assert_eq!(explained(&explain, capped).reason, Some(Cut::OverCap));
    assert_eq!(explained(&explain, weak).reason, Some(Cut::UnderFloor));
    let below = explained(&explain, faded);
    assert_eq!(below.reason, Some(Cut::BelowTau));
    assert_eq!(below.strength, Band::Faded);
    assert_eq!((below.rrf_rank, below.logit), (None, None));
    assert_eq!(explain.candidates.last().unwrap().id, faded);
}

#[test]
fn an_explained_injection_shows_a_reranker_that_missed_its_deadline() {
    let deadline = Duration::from_millis(100);
    let h = Harness::with(1.0, "", Arc::new(SlowReranker(Duration::from_secs(2))))
        .with_deadline(deadline);
    let pottery = h.insert(fact("Tim takes a pottery class."));

    let explain = h.explain(explain_injection("pottery class schedule"));
    assert!(!explain.reranked);
    // Embedding and retrieval consume part of the request-wide deadline.
    assert!(explain.latency.total_ms >= deadline.as_millis() as u64);
    assert!(explain.latency.total_ms >= explain.latency.rerank_ms);
    assert_eq!(explain.injection.as_ref().unwrap().text, "");
    let shown = explained(&explain, pottery);
    assert_eq!(shown.reason, Some(Cut::NotReranked));
    assert_eq!(shown.logit, None);
}
