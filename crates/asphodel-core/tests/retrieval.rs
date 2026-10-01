//! Retrieval, checked against "Retrieval: hybrid recall, reranking, the
//! injection gate and the recall log" (TIM-109) and the decisions it rests
//! on: "Retrieval and ranking" (TIM-93, its resolution and every amendment,
//! including TIM-109's on the reranker deadline and the idle timeout), "API
//! surface and Hermes transport" (TIM-94, decisions 6 and 9), the TIM-99
//! amendment on the previous message, "What is a memory record?" (TIM-90,
//! the recall log), and ADRs 0001 and 0010.
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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use asphodel_core::config::RankingTuning;
use asphodel_core::constants::{CANDIDATES_PER_ARM, RERANKED, RRF_K, SHORT_FOLLOW_UP_WORDS, TAU};
use asphodel_core::ingest::Turn;
use asphodel_core::models::{Embedder, FakeEmbedder, FakeReranker, ModelError, Models, Reranker};
use asphodel_core::retrieval::{
    Band, On, PhaseFilter, Prefetch, PrefetchRequest, Recall, RecallRequest, band, effective_query,
    estimate_tokens, fuse, phase_term, score,
};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{OpenOptions, Store, VectorIndex, micros};
use asphodel_core::strength::{Kind, Phase, TimePrecision, Window, WorldTime};
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
    /// sections.
    fn with(floor: f64, extra: &str, reranker: Arc<dyn Reranker>) -> Self {
        let (injection, rest) = extra.split_once("\n---\n").unwrap_or((extra, ""));
        let tuning = Tuning::from_toml(&format!(
            "[injection]\n{injection}\n\
             [injection.reranker_floors]\n\"{}\" = {floor:?}\n\
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

fn query(text: &str) -> RecallRequest {
    RecallRequest {
        query: text.into(),
        ..RecallRequest::default()
    }
}

fn ids(recall: &Recall) -> Vec<Uuid> {
    recall.results.iter().map(|r| r.id).collect()
}

// The fixed retrieval constants (TIM-93, placed in code by TIM-98)

#[test]
fn retrieval_constants_match_the_decisions() {
    assert_eq!(RRF_K, 60.0); // decision 2
    assert_eq!(CANDIDATES_PER_ARM, 100); // decision 1
    assert_eq!(RERANKED, 40); // decision 3
    assert_eq!(SHORT_FOLLOW_UP_WORDS, 8); // decision 8
}

// Fusion (decision 2)

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
fn fusion_lists_each_id_once_and_ignores_an_empty_arm() {
    let vector = [3_i64, 1, 2];
    let bm25 = [2_i64, 3];
    let with_empty = fuse(&[&vector, &bm25, &[]]);
    assert_eq!(with_empty, fuse(&[&vector, &bm25]));
    let distinct: std::collections::BTreeSet<i64> = with_empty.iter().copied().collect();
    assert_eq!(distinct.len(), with_empty.len());
    assert_eq!(distinct, [1, 2, 3].into());
    // 3 is first and second; 2 is third and first; 1 is in one list only.
    assert_eq!(with_empty, vec![3, 2, 1]);
    assert!(fuse(&[&[], &[]]).is_empty());
}

#[test]
fn a_repeat_within_one_list_counts_once() {
    // If the repeat of 1 counted, 1 would score 1/61 + 1/62 and 2 would be
    // pushed to rank 3; counted once, 2 is second in the first list and
    // first in the other, so it wins.
    assert_eq!(fuse(&[&[1, 1, 2], &[2]]), vec![2, 1]);
}

// Score (decision 5) and phase (decision 6)

#[test]
fn the_score_adds_relevance_weighted_strength_clamped_confidence_and_phase() {
    let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
    assert!(close(score(2.0, 0.5, 1.2, 1.0, 0.0), 2.6));
    assert!(close(
        score(2.0, 0.5, 1.2, 0.5, 0.25),
        2.6 + 0.5_f64.ln() + 0.25
    ));
    // ln(c) is clamped at −3, so a stale state is demoted, never gated.
    assert!(close(score(0.0, 0.0, 0.0, 1e-9, 0.0), -3.0));
    assert!(close(score(0.0, 0.0, 0.0, 0.0, 0.0), -3.0));
    assert!(close(score(0.0, 0.0, 0.0, 0.06, 0.0), 0.06_f64.ln()));
}

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
fn the_phase_term_is_zero_for_current_memories() {
    assert_eq!(phase(&window(Kind::Fact), false), 0.0);
    let open_task = window(Kind::Task);
    assert_eq!(phase(&open_task, false), 0.0);
    let state = Window {
        valid_from: Some(utc("2026-09-01T00:00:00Z")),
        ..window(Kind::State)
    };
    assert_eq!(phase(&state, false), 0.0);
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

#[test]
fn injection_weighs_strength_more_than_explicit_recall_by_default() {
    let ranking = Tuning::default().ranking;
    assert!(ranking.w_s_inject > ranking.w_s_recall); // decision 5
}

// The strength band (decision 13)

#[test]
fn bands_split_at_tau_and_the_strong_cutoff() {
    let cutoff = Tuning::default().recall.strong_cutoff;
    assert_eq!(band(TAU - 0.5, cutoff), Band::Faded);
    assert_eq!(band((TAU + cutoff) / 2.0, cutoff), Band::Fading);
    assert_eq!(band(cutoff + 0.5, cutoff), Band::Strong);
    assert_eq!(band(f64::NEG_INFINITY, cutoff), Band::Faded);
}

// Short follow-ups (decision 8, as amended by TIM-99)

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

#[test]
fn the_recall_log_stores_the_query_that_ran() {
    let h = Harness::new();
    let previous = "dentist appointment Friday";

    let short = h.prefetch_after("s", "yes, book it", Some(previous));
    let (_, _, _, logged, _, _) = h.recall_row(short.recall_id);
    assert_eq!(logged, effective_query("yes, book it", Some(previous)));
    assert!(logged.contains(previous) && logged.contains("yes, book it"));

    let long = "could you please move it to the following Monday instead";
    let full = h.prefetch_after("s", long, Some(previous));
    let (_, _, _, logged, _, _) = h.recall_row(full.recall_id);
    assert_eq!(logged, long);
}

// Retrievers and clean-up (decision 1)

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

// The injection gate (decisions 8 and 9)

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
fn injecting_nothing_is_a_valid_result() {
    let h = Harness::new();
    h.insert(fact("Tim owns a canoe."));
    let prefetch = h.prefetch("s", "pottery class schedule");
    assert!(prefetch.injected.is_empty());
    assert!(prefetch.text.is_empty());
    assert!(prefetch.reranked);
    // The recall is still logged.
    let (kind, ..) = h.recall_row(prefetch.recall_id);
    assert_eq!(kind, "prefetch");
}

#[test]
fn injection_takes_at_most_the_cap() {
    const NOTES: [&str; 10] = [
        "Pottery class note one.",
        "Pottery class note two.",
        "Pottery class note three.",
        "Pottery class note four.",
        "Pottery class note five.",
        "Pottery class note six.",
        "Pottery class note seven.",
        "Pottery class note eight.",
        "Pottery class note nine.",
        "Pottery class note ten.",
    ];
    let h = Harness::new();
    for note in NOTES {
        h.insert(fact(note));
    }
    assert_eq!(Tuning::default().injection.cap, 8);
    assert_eq!(h.prefetch("s", "pottery class").injected.len(), 8);

    let h = Harness::with(1.0, "cap = 3", Arc::new(FakeReranker));
    for note in NOTES {
        h.insert(fact(note));
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

// The reranker deadline (decision 8, as amended by TIM-109)

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

#[test]
fn a_late_reranker_leaves_explicit_recall_in_rrf_order() {
    let h = Harness::with(1.0, "", Arc::new(SlowReranker(Duration::from_secs(2))))
        .with_deadline(Duration::from_millis(100));
    let pottery = h.insert(fact("Tim takes a pottery class."));

    let started = Instant::now();
    let recall = h.recall(query("pottery class"));
    assert!(
        started.elapsed() < Duration::from_millis(1500),
        "waited for the reranker"
    );
    assert!(!recall.reranked);
    assert_eq!(ids(&recall), vec![pottery]);
}

#[test]
fn a_reranker_inside_the_deadline_is_used() {
    let h = Harness::with(1.0, "", Arc::new(SlowReranker(Duration::from_millis(20))))
        .with_deadline(Duration::from_secs(5));
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let prefetch = h.prefetch("s", "pottery class schedule");
    assert!(prefetch.reranked);
    assert_eq!(prefetch.injected, vec![pottery]);
    assert!(h.recall(query("pottery class")).reranked);
}

// The injection format (decision 10)

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
    has(&format!("- {passport} [overdue since 30 Sep]"));
    has(&format!("- {berlin} [ended 12 Sep]"));
    has(&format!("- {yoga} [recurring: every Tuesday]"));
    has(&format!("- {lisbon} [observed 4 days ago, Sun 27 Sep]"));
    has(&format!("- {tax}"));
    let sister_line = lines.iter().find(|l| l.contains(sister)).unwrap();
    assert!(sister_line.contains("date uncertain"), "{sister_line}");
}

// The in-context skip and per-session state (TIM-94, decision 6)

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
fn a_turn_without_the_recall_id_discards_the_pending_injection() {
    let h = Harness::new();
    let pottery = h.insert(fact("Tim takes a pottery class."));
    let first = h.prefetch("s", "pottery class schedule");
    h.sync_turn("s", None);
    assert!(h.in_context("s").is_empty());
    // It was discarded, not held: echoing it later commits nothing.
    h.sync_turn("s", Some(first.recall_id.to_string()));
    assert!(h.in_context("s").is_empty());
    assert_eq!(
        h.prefetch("s", "pottery class schedule").injected,
        vec![pottery]
    );
}

#[test]
fn a_turn_with_another_recall_id_discards_the_pending_injection() {
    let h = Harness::new();
    h.insert(fact("Tim takes a pottery class."));
    let first = h.prefetch("s", "pottery class schedule");
    h.sync_turn("s", Some(Uuid::from_u128(7).to_string()));
    assert!(h.in_context("s").is_empty());
    h.sync_turn("s", Some(first.recall_id.to_string()));
    assert!(h.in_context("s").is_empty());
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
fn an_idle_session_expires_after_in_context_idle_days() {
    assert_eq!(Tuning::default().sessions.in_context_idle_days, 7);
    let h = Harness::new();
    let pottery = h.insert(fact("Tim takes a pottery class."));
    for session in ["kept", "expired"] {
        let prefetch = h.prefetch(session, "pottery class schedule");
        h.sync_turn(session, Some(prefetch.recall_id.to_string()));
    }

    h.clock.advance(SignedDuration::from_hours(6 * 24));
    assert_eq!(h.in_context("kept"), vec![pottery]);
    h.clock
        .advance(SignedDuration::from_hours(24) + SignedDuration::from_mins(1));
    assert!(h.in_context("expired").is_empty());
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

// The recall log (TIM-90) and accesses (ADR 0001)

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
fn a_recall_logs_one_row_with_its_results_in_rank_order() {
    let h = Harness::with_floor(0.0);
    h.insert(fact("A pottery note."));
    h.insert(fact("A pottery class note."));
    let recall = h.recall(RecallRequest {
        session_id: Some("s".into()),
        ..query("pottery class")
    });
    let (kind, session, _, logged, _, _) = h.recall_row(recall.recall_id);
    assert_eq!(kind, "tool");
    assert_eq!(session.as_deref(), Some("s"));
    assert_eq!(logged, "pottery class");
    let results = h.results(recall.recall_id);
    assert_eq!(
        results.iter().map(|(m, _)| *m).collect::<Vec<_>>(),
        ids(&recall)
    );
    assert!(results.iter().all(|(_, injected)| !injected));
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

// Explicit recall (decision 13, TIM-94 decision 9)

#[test]
fn recall_returns_each_result_with_its_fields() {
    let h = Harness::new();
    let dentist = h.insert(Memory {
        kind: "event",
        owner_significance: Some("kept"),
        valid_from: Some((local("2026-10-03T15:00"), "minute")),
        ..fact("Tim has a dentist appointment on 3 October 2026 at 15:00.")
    });
    let recall = h.recall(query("dentist appointment"));
    let result = recall.results.iter().find(|r| r.id == dentist).unwrap();
    assert_eq!(
        result.sentence,
        "Tim has a dentist appointment on 3 October 2026 at 15:00."
    );
    assert_eq!(result.kind, Kind::Event);
    assert_eq!(result.phase, Phase::Upcoming);
    assert_eq!(result.observed_at, at(EARLIER));
    assert_eq!(
        result.window.valid_from.map(|t| t.at),
        Some(local("2026-10-03T15:00"))
    );
    assert!(result.kept);
}

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
fn recall_includes_ended_memories() {
    let h = Harness::new();
    let berlin = h.insert(Memory {
        kind: "state",
        valid_from: Some((local("2024-01-01T00:00"), "month")),
        valid_until: Some((local("2026-06-01T00:00"), "month")),
        ..fact("Tim lives in Berlin.")
    });
    let recall = h.recall(query("Tim lives in Berlin"));
    let found = recall.results.iter().find(|r| r.id == berlin).unwrap();
    assert!(matches!(found.phase, Phase::RecentlyPast | Phase::LongPast));
}

#[test]
fn recall_filters_by_kind() {
    let h = Harness::with_floor(0.0);
    let task = h.insert(Memory {
        kind: "task",
        ..fact("Tim needs to weed the garden.")
    });
    h.insert(fact("Tim's garden has a lemon tree."));
    h.insert(Memory {
        kind: "event",
        valid_from: Some((local("2026-09-20T00:00"), "day")),
        ..fact("Tim planted garlic in the garden.")
    });
    let recall = h.recall(RecallRequest {
        kinds: vec![Kind::Task],
        ..query("garden")
    });
    assert_eq!(ids(&recall), vec![task]);
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
    assert!(!now.contains(&upcoming) && !now.contains(&past));
    assert_eq!(only(PhaseFilter::Any).len(), 3);
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
fn recall_returns_ten_results_unless_asked_for_up_to_thirty() {
    const NOTES: [&str; 35] = [
        "Garden note 1.",
        "Garden note 2.",
        "Garden note 3.",
        "Garden note 4.",
        "Garden note 5.",
        "Garden note 6.",
        "Garden note 7.",
        "Garden note 8.",
        "Garden note 9.",
        "Garden note 10.",
        "Garden note 11.",
        "Garden note 12.",
        "Garden note 13.",
        "Garden note 14.",
        "Garden note 15.",
        "Garden note 16.",
        "Garden note 17.",
        "Garden note 18.",
        "Garden note 19.",
        "Garden note 20.",
        "Garden note 21.",
        "Garden note 22.",
        "Garden note 23.",
        "Garden note 24.",
        "Garden note 25.",
        "Garden note 26.",
        "Garden note 27.",
        "Garden note 28.",
        "Garden note 29.",
        "Garden note 30.",
        "Garden note 31.",
        "Garden note 32.",
        "Garden note 33.",
        "Garden note 34.",
        "Garden note 35.",
    ];
    let h = Harness::with_floor(0.0);
    for note in NOTES {
        h.insert(fact(note));
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
