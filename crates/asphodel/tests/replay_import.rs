//! `asphodel import`: a copy of Hermes' `state.db` and the private manifest
//! into a replay corpus, checked against "Replay harness: simulated-clock
//! replay of recorded sessions" (TIM-96, decisions 1 and 8) and TIM-117's
//! scope. Every `state.db` here is synthetic, built by `support::hermes`
//! with the Hermes DDL.
//!
//! What the tests read of the corpus is deliberately small: it's JSON
//! lines, and each event line has `event` (`prefetch`, `sync` or `clear`),
//! `at` and `session`; a prefetch has its `query`, and a sync has `user`,
//! `assistant` and, for a non-owner speaker, `author` with the speaker's
//! `name`, the shape a scenario's turn already has. Lines without `event`
//! (a header, say) are the importer's business.

mod support;

use std::fs;
use std::path::Path;

use jiff::Timestamp;
use serde_json::Value;
use support::hermes::{
    self, API_CONTENT_SENTINEL, Message, PROMPT_TABLE_SENTINEL, SYSTEM_PROMPT_SENTINEL, StateDb,
    epoch,
};
use support::{TestDir, assert_ok, assert_refused, import, stderr, stdout};

/// The corpus's event lines, in file order.
fn events(corpus: &Path) -> Vec<Value> {
    fs::read_to_string(corpus)
        .unwrap_or_else(|_| panic!("no corpus at {}", corpus.display()))
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).expect("every corpus line is JSON"))
        .filter(|line| line.get("event").is_some())
        .collect()
}

/// One session's events, ordered by time and then file order.
fn session_events(corpus: &Path, session: &str) -> Vec<Value> {
    let mut events: Vec<Value> = events(corpus)
        .into_iter()
        .filter(|event| event["session"] == session)
        .collect();
    events.sort_by_key(at);
    events
}

fn at(event: &Value) -> Timestamp {
    event["at"]
        .as_str()
        .unwrap_or_else(|| panic!("an event's `at` is an RFC 3339 string: {event}"))
        .parse()
        .unwrap()
}

fn kinds(events: &[Value]) -> Vec<&str> {
    events
        .iter()
        .map(|event| event["event"].as_str().unwrap())
        .collect()
}

fn ts(at: &str) -> Timestamp {
    at.parse().unwrap()
}

/// Imports `db` (already written at `state_db`) and returns the corpus.
fn imported(dir: &TestDir, state_db: &Path) -> std::path::PathBuf {
    let corpus = dir.private_path("corpus/main.jsonl");
    assert_ok(&import(dir, state_db, &corpus));
    corpus
}

// Turns.

/// TIM-96, decision 1: each turn is a prefetch at the user message's time
/// and a `sync_turn` at the final assistant reply's time. Tool rows, and
/// the assistant rows that only call tools, are skipped.
#[test]
fn each_turn_becomes_a_prefetch_and_a_sync_and_tool_rows_are_skipped() {
    let dir = TestDir::new();
    let state_db = dir.private_path("state.db");
    let db = StateDb::create(&state_db);
    let t = epoch("2026-01-05T09:00:00Z");
    db.session("s1", "discord", Some("discord:1"), None, t);
    db.message(Message {
        session: "s1",
        role: "user",
        content: "What's on my calendar tomorrow?",
        at: t,
        ..Message::default()
    });
    db.message(Message {
        session: "s1",
        role: "assistant",
        content: "INTERMEDIATE-ASSISTANT-TEXT",
        at: t + 5.0,
        tool_calls: Some(
            r#"[{"id":"c1","type":"function","function":{"name":"calendar","arguments":"{}"}}]"#,
        ),
        ..Message::default()
    });
    db.message(Message {
        session: "s1",
        role: "tool",
        content: "TOOL-RESULT-TEXT",
        at: t + 6.0,
        ..Message::default()
    });
    db.message(Message {
        session: "s1",
        role: "assistant",
        content: "A dentist appointment at ten.",
        at: t + 20.0,
        ..Message::default()
    });
    drop(db);

    let corpus = imported(&dir, &state_db);
    let events = session_events(&corpus, "s1");
    assert_eq!(kinds(&events), ["prefetch", "sync"], "{events:#?}");
    assert_eq!(at(&events[0]), ts("2026-01-05T09:00:00Z"));
    assert_eq!(events[0]["query"], "What's on my calendar tomorrow?");
    assert_eq!(at(&events[1]), ts("2026-01-05T09:00:20Z"));
    assert_eq!(events[1]["user"], "What's on my calendar tomorrow?");
    assert_eq!(events[1]["assistant"], "A dentist appointment at ten.");

    let text = fs::read_to_string(&corpus).unwrap();
    assert!(!text.contains("INTERMEDIATE-ASSISTANT-TEXT"), "{text}");
    assert!(!text.contains("TOOL-RESULT-TEXT"), "{text}");
}

/// TIM-96, decision 1: the compacted turns (`active=0, compacted=1`) are
/// replayed, a clear is emitted at the boundary between them and the
/// summary row, and the summary row itself is skipped. The summary row
/// carries Hermes' own `_compressed_summary` flag and sits where TIM-96
/// says it does, so either way of finding it passes.
#[test]
fn compaction_replays_the_compacted_turns_and_clears_at_the_boundary() {
    let dir = TestDir::new();
    let state_db = dir.private_path("state.db");
    let db = StateDb::create(&state_db);
    let t = epoch("2026-01-05T09:00:00Z");
    db.session("s1", "discord", Some("discord:1"), None, t);
    for (offset, user, assistant) in [
        (0.0, "Compacted question one.", "Compacted answer one."),
        (600.0, "Compacted question two.", "Compacted answer two."),
    ] {
        for (role, content, delta) in [("user", user, 0.0), ("assistant", assistant, 30.0)] {
            db.message(Message {
                session: "s1",
                role,
                content,
                at: t + offset + delta,
                active: false,
                compacted: true,
                ..Message::default()
            });
        }
    }
    db.message(Message {
        session: "s1",
        role: "user",
        content: "HERMES-COMPACTION-SUMMARY",
        at: t + 1200.0,
        summary: true,
        ..Message::default()
    });
    db.turn(
        "s1",
        t + 1800.0,
        "A question after compaction.",
        "An answer.",
    );
    drop(db);

    let corpus = imported(&dir, &state_db);
    let events = session_events(&corpus, "s1");
    assert_eq!(
        kinds(&events),
        [
            "prefetch", "sync", "prefetch", "sync", "clear", "prefetch", "sync"
        ],
        "{events:#?}"
    );
    assert_eq!(events[0]["query"], "Compacted question one.");
    assert_eq!(events[3]["assistant"], "Compacted answer two.");
    let clear = at(&events[4]);
    assert!(
        at(&events[3]) <= clear && clear <= at(&events[5]),
        "the clear sits between the compacted turns and the next one: {events:#?}"
    );
    assert_eq!(events[5]["query"], "A question after compaction.");
    let text = fs::read_to_string(&corpus).unwrap();
    assert!(!text.contains("HERMES-COMPACTION-SUMMARY"), "{text}");
}

/// TIM-96, decision 1: the `[Name] ` prefix stays in the text and only picks
/// the speaker, which is the owner unless the name is a non-owner speaker
/// in the manifest. Backfilled history before `[New message]` is stripped
/// the way `plugin/turns.py` strips it: up to the last marker and the
/// whitespace after it.
#[test]
fn speakers_come_from_the_prefix_and_backfill_is_stripped() {
    let dir = TestDir::new();
    let state_db = dir.private_path("state.db");
    let db = StateDb::create(&state_db);
    let t = epoch("2026-01-05T09:00:00Z");
    db.session("s1", "discord", Some("discord:1"), None, t);
    db.turn(
        "s1",
        t,
        "[Sam] I'm Tim's friend from Wellington.",
        "Hi Sam.",
    );
    db.turn("s1", t + 600.0, "[Bob] Is it raining?", "Not yet.");
    db.turn(
        "s1",
        t + 1200.0,
        "[Alice] BACKFILLED-CHANNEL-HISTORY\n[New message] [Sam] See you at noon.",
        "See you then.",
    );
    db.turn("s1", t + 1800.0, "Plain owner text.", "Okay.");
    drop(db);

    let corpus = imported(&dir, &state_db);
    let syncs: Vec<Value> = session_events(&corpus, "s1")
        .into_iter()
        .filter(|event| event["event"] == "sync")
        .collect();
    assert_eq!(syncs.len(), 4, "{syncs:#?}");

    assert_eq!(syncs[0]["user"], "[Sam] I'm Tim's friend from Wellington.");
    assert_eq!(syncs[0]["author"]["name"], "Sam");

    // Bob isn't in the manifest, so the turn is the owner's.
    assert_eq!(syncs[1]["user"], "[Bob] Is it raining?");
    assert!(syncs[1]["author"].is_null(), "{}", syncs[1]);

    assert_eq!(syncs[2]["user"], "[Sam] See you at noon.");
    assert_eq!(syncs[2]["author"]["name"], "Sam");

    assert!(syncs[3]["author"].is_null(), "{}", syncs[3]);

    let text = fs::read_to_string(&corpus).unwrap();
    assert!(!text.contains("BACKFILLED-CHANNEL-HISTORY"), "{text}");
}

/// TIM-96, decision 1: cron sessions get prefetch only, and subagent
/// sessions (those with a `parent_session_id`) are skipped.
#[test]
fn cron_sessions_get_prefetch_only_and_subagent_sessions_nothing() {
    let dir = TestDir::new();
    let state_db = dir.private_path("state.db");
    hermes::small_history(&state_db);
    let corpus = imported(&dir, &state_db);

    let cron = session_events(&corpus, "s-cron");
    assert!(!cron.is_empty(), "the cron session is replayed");
    assert!(
        cron.iter().all(|event| event["event"] == "prefetch"),
        "{cron:#?}"
    );
    assert!(session_events(&corpus, "s-sub").is_empty());
    let text = fs::read_to_string(&corpus).unwrap();
    assert!(!text.contains("Subagent task"), "{text}");

    // The primary sessions are there, a prefetch and a sync per turn.
    assert_eq!(
        kinds(&session_events(&corpus, "s-main")),
        ["prefetch", "sync", "prefetch", "sync", "prefetch", "sync"]
    );
}

// The schema check.

/// TIM-117: the importer checks the schema it was written against and
/// fails loudly on a mismatch, naming what's wrong. It writes no corpus.
#[test]
fn a_missing_column_fails_the_import_and_names_the_column() {
    let dir = TestDir::new();
    let state_db = dir.private_path("state.db");
    let db = hermes::small_history(&state_db);
    db.conn()
        .execute_batch("ALTER TABLE messages DROP COLUMN compacted")
        .unwrap();
    drop(db);
    let corpus = dir.private_path("corpus/main.jsonl");
    let output = import(&dir, &state_db, &corpus);
    assert_refused(&output, "compacted");
    assert!(!corpus.exists(), "a refused import writes no corpus");
}

#[test]
fn a_schema_version_the_importer_wasnt_written_against_fails_the_import() {
    let dir = TestDir::new();
    let state_db = dir.private_path("state.db");
    let db = hermes::small_history(&state_db);
    db.conn()
        .execute_batch("UPDATE schema_version SET version = 9999")
        .unwrap();
    drop(db);
    let corpus = dir.private_path("corpus/main.jsonl");
    let output = import(&dir, &state_db, &corpus);
    assert_refused(&output, "9999");
    assert!(!corpus.exists(), "a refused import writes no corpus");
}

// Privacy (TIM-96, decision 8).

/// The importer never copies the system prompt or `api_content`: not into
/// the corpus, not into its output, not into any file under the private
/// dir other than the `state.db` copy itself.
#[test]
fn the_importer_never_copies_the_system_prompt_or_api_content() {
    let dir = TestDir::new();
    let state_db = dir.private_path("state.db");
    hermes::small_history(&state_db);
    let corpus = dir.private_path("corpus/main.jsonl");
    let output = import(&dir, &state_db, &corpus);
    assert_ok(&output);

    let sentinels = [
        SYSTEM_PROMPT_SENTINEL,
        PROMPT_TABLE_SENTINEL,
        API_CONTENT_SENTINEL,
    ];
    for sentinel in sentinels {
        assert!(!stdout(&output).contains(sentinel), "stdout has {sentinel}");
        assert!(!stderr(&output).contains(sentinel), "stderr has {sentinel}");
    }
    let mut pending = vec![dir.private()];
    while let Some(path) = pending.pop() {
        if path.is_dir() {
            pending.extend(
                fs::read_dir(&path)
                    .unwrap()
                    .map(|entry| entry.unwrap().path()),
            );
            continue;
        }
        if path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("state.db"))
        {
            continue;
        }
        let bytes = fs::read(&path).unwrap();
        let text = String::from_utf8_lossy(&bytes);
        for sentinel in sentinels {
            assert!(
                !text.contains(sentinel),
                "{} holds {sentinel}",
                path.display()
            );
        }
    }
}

/// Everything derived from real history stays in `ASPHODEL_REPLAY_DIR`.
#[test]
fn the_corpus_is_refused_outside_the_private_dir() {
    let dir = TestDir::new();
    let state_db = dir.private_path("state.db");
    hermes::small_history(&state_db);
    let outside = dir.path("outside.jsonl");
    let output = import(&dir, &state_db, &outside);
    assert_refused(&output, "private");
    assert!(!outside.exists());
}

/// The corpus hash in every report means something only if the same input
/// always imports to the same bytes.
#[test]
fn importing_the_same_history_twice_writes_the_same_corpus() {
    let dir = TestDir::new();
    let state_db = dir.private_path("state.db");
    hermes::small_history(&state_db);
    let first = dir.private_path("corpus/first.jsonl");
    let second = dir.private_path("corpus/second.jsonl");
    assert_ok(&import(&dir, &state_db, &first));
    assert_ok(&import(&dir, &state_db, &second));
    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
}
