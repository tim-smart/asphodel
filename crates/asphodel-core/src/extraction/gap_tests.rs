//! Lock-gap regressions. A prepared chunk is
//! committed while the sweep purges one of its neighbours at the last
//! moment before the commit takes the store for its writes, through the
//! [`super::gap`] hook. The commit must plan against the store it writes
//! to: a claim whose neighbour is gone is new, as if it had never been
//! stored, and nothing fails.
//!
//! In production the housekeeping task runs the sweep on another thread
//! against the same store, so the purge can land there; the hook makes the
//! interleaving deterministic.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use jiff::Timestamp;
use serde_json::{Value, json};
use uuid::Uuid;

use super::gap;
use crate::clock::SimulatedClock;
use crate::config::Tuning;
use crate::erase::{EraseReason, erase_chain};
use crate::ingest::Turn;
use crate::models::{FakeEmbedder, FakeLlm, FakeReranker, Models};
use crate::queue::Lease;
use crate::store::bank::BankIdentity;
use crate::store::{OpenOptions, Store};
use crate::{Clock, Service};

const MODEL: &str = "fake-llm";
const LISBON: &str = "Tim lives in Lisbon.";
const BERLIN: &str = "Tim lives in Berlin.";

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-gap-{}-{}",
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

/// A service on the fake models with one bank, `main`. Field order
/// matters: the service drops before the directory it lives in.
struct Harness {
    service: Service,
    _dir: TestDir,
}

impl Harness {
    fn new() -> Self {
        let dir = TestDir::new();
        let clock = Arc::new(SimulatedClock::new(at("2026-10-01T07:00:00Z")));
        let store =
            Store::open(&dir.0.join("data"), OpenOptions::default(), clock.clone()).unwrap();
        let tuning = Tuning::from_toml(&format!(
            "[injection.reranker_floors]\n\"{}\" = 0.0\n\
             [reconcile.embedding_floors]\n\"{}\" = 0.5\n",
            FakeReranker::MODEL_ID,
            FakeEmbedder::MODEL_ID,
        ))
        .unwrap();
        let service =
            Service::with_models(clock as Arc<dyn Clock>, store, tuning, Models::fake()).unwrap();
        service
            .ensure_bank_with_models(
                "main",
                &BankIdentity {
                    owner_name: Some("Tim".into()),
                    owner_platform_ids: vec!["discord:1234".into()],
                    assistant_name: Some("Hermes".into()),
                    timezone: Some("Pacific/Auckland".into()),
                },
            )
            .unwrap();
        Self { service, _dir: dir }
    }

    fn ingest(&self, session: &str, message_at: &str, user: &str) {
        self.service
            .ingest_turn(
                "main",
                &Turn {
                    session_id: session.into(),
                    message_at: at(message_at),
                    timezone: Some("Pacific/Auckland".into()),
                    user_text: user.into(),
                    assistant_text: "Noted.".into(),
                    author: None,
                    platform: Some("cli".into()),
                    recall_id: None,
                    forget_requested: false,
                },
            )
            .unwrap();
    }

    fn lease(&self) -> Lease {
        self.service
            .claim_chunk("main")
            .unwrap()
            .expect("a chunk is queued")
    }

    /// Stores `content` from a turn of its own at `message_at`.
    fn remember(&self, message_at: &str, user: &str, content: &str, quote: &str) -> Uuid {
        self.ingest("s1", message_at, user);
        let llm = FakeLlm::scripted(MODEL, vec![reply(content, quote)]);
        let extracted = self.service.extract_chunk(self.lease(), &llm, &[]).unwrap();
        assert_eq!(extracted.memories.len(), 1, "the fixture memory is stored");
        extracted.memories[0]
    }

    /// Prepares the head of the queue where call 1 finds `content` and
    /// call 2 labels it `label` against `neighbour`.
    fn prepare(&self, content: &str, quote: &str, neighbour: Uuid, label: &str) -> super::Prepared {
        let lease = self.lease();
        let call1 = reply(content, quote);
        let input = self
            .service
            .call2_input(&lease, &call1, &[])
            .unwrap()
            .expect("call 2 runs: the claim is near the neighbour");
        let handle = input
            .neighbours
            .iter()
            .find(|shown| shown.memory == neighbour)
            .expect("the neighbour is shown to call 2")
            .handle
            .clone();
        let call2 = json!({"claims": [{
            "claim": input.claims[0].handle,
            "labels": [{"neighbour": handle, "label": label}],
        }]});
        let llm = FakeLlm::scripted(MODEL, vec![call1, call2]);
        self.service
            .prepare_extraction(lease, &llm, &[], &[])
            .unwrap()
    }

    fn row(&self, memory: Uuid) -> Option<i64> {
        use rusqlite::OptionalExtension as _;
        self.conn()
            .query_row(
                "SELECT id FROM memories WHERE uuid = ?1",
                [memory.to_string()],
                |row| row.get(0),
            )
            .optional()
            .unwrap()
    }

    fn content(&self, memory: Uuid) -> String {
        self.conn()
            .query_row(
                "SELECT content FROM memories WHERE uuid = ?1",
                [memory.to_string()],
                |row| row.get(0),
            )
            .unwrap()
    }

    fn ended_by(&self, memory: Uuid) -> Option<i64> {
        self.conn()
            .query_row(
                "SELECT ended_by FROM memories WHERE uuid = ?1",
                [memory.to_string()],
                |row| row.get(0),
            )
            .unwrap()
    }

    fn conn(&self) -> std::sync::MutexGuard<'_, rusqlite::Connection> {
        self.service
            .store()
            .expect("the service has a store")
            .connection()
    }

    /// Chunks neither extracted nor failed.
    fn queued(&self) -> i64 {
        self.conn()
            .query_row(
                "SELECT COUNT(*) FROM chunks WHERE extracted_at IS NULL AND failed_at IS NULL",
                [],
                |row| row.get(0),
            )
            .unwrap()
    }

    /// Failed attempts counted across every chunk.
    fn errors(&self) -> i64 {
        self.conn()
            .query_row(
                "SELECT COALESCE(SUM(error_count), 0) FROM chunks",
                [],
                |row| row.get(0),
            )
            .unwrap()
    }

    /// At the next commit's gap, purges `memory`'s chain as the sweep
    /// does, in a transaction of its own.
    fn purge_in_the_gap(&self, memory: Uuid) {
        let id = self.row(memory).expect("the memory is stored");
        gap::set(move |store: &Store| {
            let mut conn = store.connection();
            let tx = conn.transaction().unwrap();
            let bank_id: i64 = tx
                .query_row("SELECT bank_id FROM memories WHERE id = ?1", [id], |row| {
                    row.get(0)
                })
                .unwrap();
            erase_chain(
                &tx,
                store,
                bank_id,
                &BTreeSet::from([id]),
                EraseReason::Purge,
            )
            .unwrap();
            tx.commit().unwrap();
        });
    }
}

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
}

fn reply(content: &str, quote: &str) -> Value {
    json!({
        "claims": [{
            "content": content,
            "kind": "fact",
            "quote": quote,
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
    })
}

/// A restatement that call 2 absorbed into a neighbour the sweep then
/// purges in the gap is kept as a new memory, not lost with nowhere to
/// go: the access and passage it would have left on the neighbour can't
/// be written.
#[test]
fn a_neighbour_purged_in_the_commit_gap_leaves_the_claim_new() {
    let h = Harness::new();
    let lisbon = h.remember(
        "2026-10-01T06:30:00Z",
        "I live in Lisbon.",
        LISBON,
        "I live in Lisbon",
    );
    h.ingest("s1", "2026-10-01T06:40:00Z", "I live in Lisbon, still.");
    let prepared = h.prepare(LISBON, "I live in Lisbon", lisbon, "mentioned_again");

    h.purge_in_the_gap(lisbon);
    let extracted = h
        .service
        .commit_extraction(prepared)
        .expect("the commit succeeds");

    assert_eq!(h.row(lisbon), None, "the neighbour was purged");
    assert_eq!(
        extracted.memories.len(),
        1,
        "the restatement is kept as a new memory"
    );
    assert_eq!(h.content(extracted.memories[0]), LISBON);
    assert_eq!(h.queued(), 0, "the chunk left the queue");
}

/// An older claim that a newer neighbour would have ended is created as
/// it is when the sweep purges that neighbour in the gap: new and not
/// ended, rather than a commit that fails on the vanished ender.
#[test]
fn an_ender_purged_in_the_commit_gap_doesnt_fail_the_commit() {
    let h = Harness::new();
    let lisbon = h.remember(
        "2026-10-01T06:30:00Z",
        "I live in Lisbon.",
        LISBON,
        "I live in Lisbon",
    );
    // Said a month before Lisbon, so Lisbon is the newer and would end it.
    h.ingest("s2", "2026-09-01T06:30:00Z", "I live in Berlin.");
    let prepared = h.prepare(BERLIN, "I live in Berlin", lisbon, "ends");

    h.purge_in_the_gap(lisbon);
    let extracted = h
        .service
        .commit_extraction(prepared)
        .expect("the commit succeeds");

    assert_eq!(h.row(lisbon), None, "the neighbour was purged");
    assert_eq!(extracted.memories.len(), 1);
    let berlin = extracted.memories[0];
    assert_eq!(h.content(berlin), BERLIN);
    assert_eq!(h.ended_by(berlin), None, "nothing ends it");
    assert_eq!(h.queued(), 0, "the chunk left the queue");
    assert_eq!(h.errors(), 0, "no failed attempt was counted");
}
