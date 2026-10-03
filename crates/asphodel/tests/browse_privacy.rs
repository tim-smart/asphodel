//! Regression checks for the configuration actually used in evaluation step 5.
//! No real history, models, credentials or network are needed.

mod support;

use std::sync::Arc;

use asphodel_core::config::PurgePause;
use asphodel_core::constants::CHUNK_RETRY_CAP;
use asphodel_core::ingest::Turn;
use asphodel_core::models::Models;
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{OpenOptions, Store};
use asphodel_core::{Service, SimulatedClock, SystemClock, Tuning};

const GUIDE: &str = include_str!("../../../docs/hermes-data-evaluation.md");

fn section(number: u8) -> &'static str {
    let heading = format!("## {number}. ");
    GUIDE
        .split_once(&heading)
        .expect("evaluation step exists")
        .1
        .split("\n## ")
        .next()
        .unwrap()
}

fn toml(step: &str) -> Tuning {
    let text = step
        .split_once("```toml\n")
        .expect("document the config in a TOML block")
        .1
        .split_once("```")
        .unwrap()
        .0;
    Tuning::from_toml(text).expect("documented tuning is valid")
}

fn browse_tuning() -> Tuning {
    let command = section(5)
        .split_once("asphodel serve ")
        .expect("step 5 starts the browse daemon")
        .1
        .split_once("--config ")
        .expect("browse command supplies its config")
        .1;
    let config = command.split_whitespace().next().unwrap();
    match config {
        "\"$ASPHODEL_REPLAY_DIR/replay.toml\"" => toml(section(3)),
        "\"$ASPHODEL_REPLAY_DIR/browse.toml\"" => toml(section(5)),
        other => panic!("add the documented config fixture for {other}"),
    }
}

#[test]
fn documented_browse_config_has_no_llm_backend() {
    let tuning = browse_tuning();
    assert!(
        tuning.llm.model.is_none(),
        "step 5 must not configure an LLM: background refresh can send memory content"
    );
}

#[test]
fn documented_browse_config_pauses_purge_and_source_sweep() {
    let dir = support::TestDir::new();
    let replay = toml(section(3));
    let browse = browse_tuning();
    let data = dir.path("browse-store");
    {
        let store = Store::open(&data, OpenOptions::default(), Arc::new(SystemClock)).unwrap();
        // A copied replay store keeps this fingerprint. Reopening exercises
        // the same comparison as daemon startup, not an assumed mismatch.
        assert_eq!(
            store
                .check_fingerprint(&replay.deletion_fingerprint())
                .unwrap(),
            PurgePause::Running
        );
    }
    let store = Store::open(&data, OpenOptions::default(), Arc::new(SystemClock)).unwrap();
    let pause = store
        .check_fingerprint(&browse.deletion_fingerprint())
        .unwrap();
    assert!(
        matches!(pause, PurgePause::Paused { .. }),
        "browse must pause both purge and the source/recall sweep; got {pause:?}"
    );
}

#[test]
fn scheduled_browse_sweep_preserves_expired_source_chunk_and_recall_contents() {
    let dir = support::TestDir::new();
    let data = dir.path("browse-store");
    let start = "2026-10-01T07:00:00Z".parse().unwrap();
    let clock = Arc::new(SimulatedClock::new(start));
    {
        let replay = toml(section(3));
        let store = Store::open(&data, OpenOptions::default(), clock.clone()).unwrap();
        store
            .check_fingerprint(&replay.deletion_fingerprint())
            .unwrap();
        let service = Service::open(clock.clone(), store, replay);
        service
            .ensure_bank(
                "main",
                &BankIdentity {
                    timezone: Some("Pacific/Auckland".into()),
                    ..BankIdentity::default()
                },
                &Models::fake().ids(),
            )
            .unwrap();
        for (session, text) in [
            ("idle", "Good morning."),
            ("failed", "Good afternoon."),
            ("memory", "The fixture is blue."),
        ] {
            service
                .ingest_turn(
                    "main",
                    &Turn {
                        session_id: session.into(),
                        message_at: start,
                        timezone: None,
                        user_text: text.into(),
                        assistant_text: String::new(),
                        author: None,
                        platform: None,
                        recall_id: None,
                        forget_requested: false,
                    },
                )
                .unwrap();
        }
        let store = service.store().unwrap();
        let conn = store.connection();
        // Match completed extraction and a terminal failure, not pending work
        // (which the sweep would preserve even without the browse pause).
        conn.execute_batch(
            "
            DELETE FROM extraction_queue;
            UPDATE chunks SET extracted_at = 1;
            INSERT INTO memories (uuid, bank_id, content, kind, significance, chunk_id,
                source_start, source_end, observed_at, window_confidence, created_at, updated_at)
            SELECT '00000000-0000-0000-0000-000000000001', c.bank_id, 'The fixture is blue.',
                'fact', 'notable', c.id, 0, 20, s.observed_at, 'high', s.ingested_at, s.ingested_at
            FROM chunks c JOIN sources s ON s.id = c.source_id WHERE s.session_id = 'memory';
            INSERT INTO recalls (uuid, bank_id, kind, turn, query, latency_ms, at)
            SELECT '00000000-0000-0000-0000-000000000002', bank_id, 'tool', 0,
                'What colour is the fixture?', 12, created_at FROM memories;
            INSERT INTO recall_results (recall_id, memory_id, rank, score, injected)
            SELECT r.id, m.id, 0, 0.9, 1 FROM recalls r CROSS JOIN memories m;
        ",
        )
        .unwrap();
        conn.execute(
            "UPDATE chunks SET extracted_at = NULL, failed_at = ?1,
            error_count = ?2, last_error_kind = 'http'
            WHERE source_id IN (SELECT id FROM sources WHERE session_id = 'failed')",
            rusqlite::params![asphodel_core::store::micros(start), CHUNK_RETRY_CAP],
        )
        .unwrap();
    }

    // A fresh browse copy retains replay's fingerprint; startup checks it
    // and passes the resulting pause to the service, as serve does.
    let browse = browse_tuning();
    let store = Store::open(&data, OpenOptions::default(), clock.clone()).unwrap();
    let pause = store
        .check_fingerprint(&browse.deletion_fingerprint())
        .unwrap();
    let service = Service::open(clock.clone(), store, browse).with_purge_pause(pause);
    let due = service.run_sweeps().unwrap().next_due.unwrap();
    // 04:00 in Auckland, beyond the documented 90-day source horizon.
    clock.set("2026-12-30T15:00:00Z".parse().unwrap());
    assert!(due < service.now());
    let plan = service.purge_plan().unwrap();
    assert_eq!(
        (plan.sources, plan.failed_chunks, plan.recalls),
        (2, 1, 1),
        "fixtures must be eligible for deletion if the pause is lost"
    );
    let swept = service.run_sweeps().unwrap();
    assert!(swept.ran.is_empty());
    assert!(
        swept.next_due.unwrap() > service.now(),
        "the due sweep was processed"
    );

    let store = service.store().unwrap();
    let conn = store.connection();
    for (session, text) in [("idle", "Good morning."), ("failed", "Good afternoon.")] {
        let contents: (Option<String>, Option<String>, Option<i64>, Option<i64>) = conn
            .query_row(
                "SELECT s.text, c.text, s.tombstoned_at, c.tombstoned_at
             FROM sources s JOIN chunks c ON c.source_id = s.id WHERE s.session_id = ?1",
                [session],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        // Turn chunks include the separator before the empty assistant reply.
        assert_eq!(
            contents,
            (Some(text.into()), Some(format!("{text}\n\n")), None, None)
        );
    }
    let recall: (Option<String>, Option<i64>, String, i64, f64, i64) = conn
        .query_row(
            "SELECT r.query, r.swept_at, m.content, rr.rank, rr.score, rr.injected
         FROM recalls r JOIN recall_results rr ON rr.recall_id = r.id
         JOIN memories m ON m.id = rr.memory_id",
            [],
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
        .unwrap();
    assert_eq!(
        recall,
        (
            Some("What colour is the fixture?".into()),
            None,
            "The fixture is blue.".into(),
            0,
            0.9,
            1
        )
    );
}
