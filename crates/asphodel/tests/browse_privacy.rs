//! Regression check for the configuration actually used in evaluation step
//! 5: browsing a copy of the replayed store must delete nothing in it. No
//! real history, models, credentials or network are needed.

mod support;

use std::sync::Arc;

use asphodel_core::config::PurgePause;
use asphodel_core::extraction::ExtractError;
use asphodel_core::ingest::Turn;
use asphodel_core::models::{FakeEmbedder, FakeLlm, FakeReranker, LlmError, Models};
use asphodel_core::operations::{Audit, AuditList};
use asphodel_core::queue::Failure;
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{OpenOptions, Store};
use asphodel_core::{Service, SimulatedClock, Tuning};
use serde_json::json;

const GUIDE: &str = include_str!("../../../docs/hermes-data-evaluation.md");

/// What follows the first `marker` in `text`.
fn after<'a>(text: &'a str, marker: &str) -> &'a str {
    let (_, rest) = text.split_once(marker).expect("the guide has the marker");
    rest
}

fn section(number: u8) -> &'static str {
    let heading = format!("## {number}. ");
    after(GUIDE, &heading).split("\n## ").next().unwrap()
}

fn toml(step: &str) -> Tuning {
    let (text, _) = after(step, "```toml\n").split_once("```").unwrap();
    Tuning::from_toml(text).expect("documented tuning is valid")
}

/// The tuning step 5's `asphodel serve --config` reads.
fn browse_tuning() -> Tuning {
    let command = after(after(section(5), "asphodel serve "), "--config ");
    let config = command.split_whitespace().next().unwrap();
    match config {
        "\"$ASPHODEL_REPLAY_DIR/replay.toml\"" => toml(section(3)),
        "\"$ASPHODEL_REPLAY_DIR/browse.toml\"" => toml(section(5)),
        other => panic!("add the documented config fixture for {other}"),
    }
}

/// `tuning` with the floors the fake models need to serve. Floors aren't
/// in the deletion fingerprint.
fn with_fake_models(mut tuning: Tuning) -> Tuning {
    let reranker = FakeReranker::MODEL_ID.to_string();
    let floors = &mut tuning.injection.reranker_floors;
    floors.insert(reranker.clone(), 0.0);
    tuning.ranking.relevance_scales.insert(reranker, 1.0);
    let floors = &mut tuning.reconcile.embedding_floors;
    floors.insert(FakeEmbedder::MODEL_ID.into(), 0.5);
    tuning
}

/// The documented browse config has no LLM, since a background refresh
/// could send memory content to one. Its deletion fingerprint differs from
/// replay's, so a copied replay store starts with purge paused, and paused,
/// the nightly sweep keeps the expired sources, failed chunk and recall it
/// would otherwise delete, with their text.
#[test]
fn the_documented_browse_config_deletes_nothing_in_a_copied_replay_store() {
    let dir = support::TestDir::new();
    let data = dir.path("browse-store");
    let start = "2026-10-01T07:00:00Z".parse().unwrap();
    let clock = Arc::new(SimulatedClock::new(start));
    let (idle, failed) = {
        let replay = with_fake_models(toml(section(3)));
        let store = Store::open(&data, OpenOptions::default(), clock.clone()).unwrap();
        store
            .check_fingerprint(&replay.deletion_fingerprint())
            .unwrap();
        let service = Service::with_models(clock.clone(), store, replay, Models::fake()).unwrap();
        let identity = BankIdentity {
            timezone: Some("Pacific/Auckland".into()),
            ..BankIdentity::default()
        };
        let models = Models::fake().ids();
        service.ensure_bank("main", &identity, &models).unwrap();
        let ingest = |session: &str, text: &str| {
            let turn = json!({
                "session_id": session, "message_at": start, "user_text": text, "assistant_text": ""
            });
            let turn: Turn = serde_json::from_value(turn).unwrap();
            service.ingest_turn("main", &turn).unwrap().source
        };
        let idle = ingest("idle", "Good morning.");
        ingest("memory", "The fixture is blue.");
        let failed = ingest("failed", "Good afternoon.");
        let blue = support::claim("The fixture is blue.", "The fixture is blue", "fact");
        let replies = FakeLlm::scripted(
            "fake",
            vec![
                json!({ "claims": [], "used_injected_ids": [] }),
                json!({ "claims": [blue], "used_injected_ids": [] }),
            ],
        );
        for _ in 0..2 {
            service.extract_next("main", &replies).unwrap().unwrap();
        }
        let down = FakeLlm::failing("fake", || LlmError::Timeout);
        loop {
            match service.extract_next("main", &down) {
                Err(ExtractError::Call1 {
                    failure: Failure::Failed,
                    ..
                }) => break,
                Err(ExtractError::Call1 { .. }) => {}
                other => panic!("the chunk fails until it's failed: {other:?}"),
            }
        }
        let request = serde_json::from_value(json!({ "query": "What colour is the fixture?" }));
        let recall = service.recall("main", &request.unwrap()).unwrap();
        assert!(!recall.results.is_empty(), "the recall finds the memory");
        (idle, failed)
    };

    // A fresh browse copy retains replay's fingerprint; startup checks it
    // and passes the resulting pause to the service, as serve does.
    let browse = browse_tuning();
    assert!(
        browse.llm.model.is_none(),
        "step 5 must not configure an LLM"
    );
    let store = Store::open(&data, OpenOptions::default(), clock.clone()).unwrap();
    let pause = store
        .check_fingerprint(&browse.deletion_fingerprint())
        .unwrap();
    assert!(matches!(pause, PurgePause::Paused { .. }), "{pause:?}");
    let service = Service::open(clock.clone(), store, browse).with_purge_pause(pause);
    let due = service.run_sweeps().unwrap().next_due.unwrap();
    // 04:00 in Auckland, beyond the documented 90-day source horizon.
    clock.set("2026-12-30T15:00:00Z".parse().unwrap());
    assert!(due < service.now());
    let eligible = |service: &Service| {
        let plan = service.purge_plan().unwrap();
        (plan.sources, plan.failed_chunks, plan.recalls)
    };
    assert_eq!(
        eligible(&service),
        (2, 1, 1),
        "fixtures must be eligible for deletion if the pause is lost"
    );
    let swept = service.run_sweeps().unwrap();
    assert!(swept.ran.is_empty());
    assert!(
        swept.next_due.unwrap() > service.now(),
        "the due sweep was processed"
    );
    assert_eq!(eligible(&service), (2, 1, 1), "the sweep deleted nothing");
    for (source, text) in [(idle, "Good morning."), (failed, "Good afternoon.")] {
        let source = service.show_source("main", &source.to_string()).unwrap();
        assert_eq!(source.text.as_deref(), Some(text));
        assert!(source.gone.is_none(), "{:?}", source.gone);
    }
    let Audit::Recalls { recalls } = service.audit("main", AuditList::Recalls, None).unwrap()
    else {
        panic!("the recalls list");
    };
    assert_eq!(recalls.len(), 1, "{recalls:?}");
    assert_eq!(
        recalls[0].query.as_deref(),
        Some("What colour is the fixture?")
    );
    assert_eq!(recalls[0].swept_at, None);
    assert!(!recalls[0].results.is_empty());
}
