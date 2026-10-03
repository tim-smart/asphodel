//! Regression checks for the configuration actually used in evaluation step 5.
//! No real history, models, credentials or network are needed.

mod support;

use std::sync::Arc;

use asphodel_core::config::PurgePause;
use asphodel_core::store::{OpenOptions, Store};
use asphodel_core::{SystemClock, Tuning};

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
