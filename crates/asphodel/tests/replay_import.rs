//! `asphodel import`: a copy of Hermes' `state.db` and the private manifest
//! into a replay corpus. Every `state.db` here is synthetic, built by
//! `support::hermes`
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
use std::path::{Path, PathBuf};

use jiff::Timestamp;
use serde_json::Value;
use support::hermes::{
    self, API_CONTENT_SENTINEL, PROMPT_TABLE_SENTINEL, SYSTEM_PROMPT_SENTINEL, StateDb, start,
};
use support::{
    TestDir, assert_ok, assert_refused, assert_refused_without, import, import_history,
    import_paths, import_with, stderr, stdout,
};

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

/// The history of one owner session, `s1`, from [`start`] with what
/// `build` adds, imported; returns the corpus.
fn imported(dir: &TestDir, build: impl FnOnce(&StateDb)) -> PathBuf {
    import_history(dir, |path| {
        let db = hermes::one_session(path);
        build(&db);
        db
    })
}

/// The corpus's text.
fn text(corpus: &Path) -> String {
    fs::read_to_string(corpus).unwrap()
}

/// [`hermes::small_history`] in a private `state.db`; returns its path.
fn small_history(dir: &TestDir) -> PathBuf {
    let state_db = dir.private_path("state.db");
    hermes::small_history(&state_db);
    state_db
}

// Turns.

/// Each turn is a prefetch at the user message's time
/// and a `sync_turn` at the final assistant reply's time. Tool rows, and
/// the assistant rows that only call tools, are skipped.
#[test]
fn each_turn_becomes_a_prefetch_and_a_sync_and_tool_rows_are_skipped() {
    let dir = TestDir::new();
    let corpus = imported(&dir, |db| db.tool_call_turn("s1", start()));
    let events = session_events(&corpus, "s1");
    assert_eq!(kinds(&events), ["prefetch", "sync"], "{events:#?}");
    assert_eq!(at(&events[0]), ts("2026-01-05T09:00:00Z"));
    assert_eq!(events[0]["query"], "What's on my calendar tomorrow?");
    assert_eq!(at(&events[1]), ts("2026-01-05T09:00:20Z"));
    assert_eq!(events[1]["user"], "What's on my calendar tomorrow?");
    assert_eq!(events[1]["assistant"], "A dentist appointment at ten.");

    let text = text(&corpus);
    assert!(!text.contains("INTERMEDIATE-ASSISTANT-TEXT"), "{text}");
    assert!(!text.contains("TOOL-RESULT-TEXT"), "{text}");
}

/// Import only rows with
/// `active = 1 OR compacted = 1`, key the clear on the
/// `_compressed_summary` row, and order by timestamp, not row id.
///
/// The compaction carries a verbatim tail, whose clones have later ids
/// but their original timestamps. The carried turn replays once, when it
/// was said, before the clear; the clear sits at the summary row's time.
#[test]
fn a_carried_tail_replays_once_before_the_clear() {
    let dir = TestDir::new();
    let corpus = imported(&dir, |db| {
        db.carried_tail_compaction("s1", start());
        db.turn(
            "s1",
            start() + 1800.0,
            "After the compaction.",
            "Still here.",
        );
    });
    let events = session_events(&corpus, "s1");
    assert_eq!(
        kinds(&events),
        [
            "prefetch", "sync", "prefetch", "sync", "clear", "prefetch", "sync"
        ],
        "{events:#?}"
    );
    assert_eq!(events[0]["query"], "Archived question.");
    assert_eq!(events[2]["query"], "Carried question.");
    assert_eq!(at(&events[2]), ts("2026-01-05T09:10:00Z"));
    assert_eq!(events[3]["assistant"], "Carried answer.");
    assert_eq!(
        at(&events[4]),
        ts("2026-01-05T09:20:00Z"),
        "the clear is at the summary row"
    );
    assert_eq!(events[5]["query"], "After the compaction.");
    let text = text(&corpus);
    assert!(!text.contains("HERMES-COMPACTION-SUMMARY"), "{text}");
}

// Multimodal content: Hermes stores it as `\0json:`
// and a parts list, with the injected memory block inside a text part.

/// Text parts are kept, image parts dropped, and the memory block cut out,
/// so injected memory never becomes source text. The block is in its
/// default `<memory-context>` fence unless the manifest's `[memory_block]`
/// overrides it.
#[test]
fn multimodal_content_keeps_its_text_and_loses_images_and_the_memory_block() {
    let fenced = "\n[memory_block]\nstart = \"<<MEM>>\"\nend = \"<</MEM>>\"\n";
    for (start, end, fence) in [
        ("<memory-context>", "</memory-context>", ""),
        ("<<MEM>>", "<</MEM>>", fenced),
    ] {
        let dir = TestDir::new();
        let state_db = dir.private_path("state.db");
        let db = hermes::one_session(&state_db);
        let parts = serde_json::json!([
            {"type": "text", "text": format!("{start}INJECTED-MEMORY-TEXT{end}\nLook at my garden.")},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE-BYTES-SENTINEL"}}
        ]);
        db.turn(
            "s1",
            hermes::start(),
            &format!("\u{0}json:{parts}"),
            "Lovely roses.",
        );
        drop(db);

        let corpus = dir.private_path("corpus/main.jsonl");
        let manifest = format!("{}{fence}", hermes::MANIFEST);
        assert_ok(&import_with(&dir, &state_db, &corpus, &manifest, &[]));
        let events = session_events(&corpus, "s1");
        assert_eq!(kinds(&events), ["prefetch", "sync"], "{events:#?}");
        assert_eq!(events[0]["query"], "Look at my garden.");
        assert_eq!(events[1]["user"], "Look at my garden.");
        let text = text(&corpus);
        for leaked in [
            start,
            end,
            "INJECTED-MEMORY-TEXT",
            "IMAGE-BYTES-SENTINEL",
            "json:",
        ] {
            assert!(
                !text.contains(leaked),
                "the corpus holds {leaked:?}: {text}"
            );
        }
    }
}

/// The `[Name] ` prefix stays in the text and only picks
/// the speaker, which is the owner unless the name is a non-owner speaker
/// in the manifest. Backfilled history before `[New message]` is stripped
/// the way `plugin/turns.py` strips it: up to the last marker and the
/// whitespace after it.
#[test]
fn speakers_come_from_the_prefix_and_backfill_is_stripped() {
    let dir = TestDir::new();
    let corpus = imported(&dir, |db| {
        let t = start();
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
    });
    let syncs: Vec<Value> = session_events(&corpus, "s1")
        .into_iter()
        .filter(|event| event["event"] == "sync")
        .collect();
    let speakers: Vec<(&str, Option<&str>)> = syncs
        .iter()
        .map(|sync| {
            let author = sync["author"]
                .as_object()
                .map(|author| author["name"].as_str().unwrap());
            (sync["user"].as_str().unwrap(), author)
        })
        .collect();
    // Bob isn't in the manifest, so his turn is the owner's.
    assert_eq!(
        speakers,
        [
            ("[Sam] I'm Tim's friend from Wellington.", Some("Sam")),
            ("[Bob] Is it raining?", None),
            ("[Sam] See you at noon.", Some("Sam")),
            ("Plain owner text.", None),
        ],
        "{syncs:#?}"
    );
    assert!(!text(&corpus).contains("BACKFILLED-CHANNEL-HISTORY"));
}

/// Cron sessions get prefetch only, and subagent
/// sessions (those with a `parent_session_id`) are skipped.
#[test]
fn cron_sessions_get_prefetch_only_and_subagent_sessions_nothing() {
    let dir = TestDir::new();
    let corpus = import_history(&dir, hermes::small_history);

    let cron = session_events(&corpus, "s-cron");
    assert!(!cron.is_empty(), "the cron session is replayed");
    assert!(
        cron.iter().all(|event| event["event"] == "prefetch"),
        "{cron:#?}"
    );
    assert!(session_events(&corpus, "s-sub").is_empty());
    assert!(!text(&corpus).contains("Subagent task"));

    // The primary sessions are there, a prefetch and a sync per turn.
    assert_eq!(
        kinds(&session_events(&corpus, "s-main")),
        ["prefetch", "sync", "prefetch", "sync", "prefetch", "sync"]
    );
}

// The schema check.

/// The importer checks the schema it was written against and fails loudly
/// on a mismatch, naming what's wrong, and writes no corpus: a schema
/// version it wasn't written against, or a column with the right name and
/// the wrong declared type, named with the type expected.
#[test]
fn a_schema_the_importer_wasnt_written_against_fails_the_import() {
    for (change, words) in [
        ("UPDATE schema_version SET version = 9999", &["9999"][..]),
        (
            "ALTER TABLE messages RENAME COLUMN timestamp TO old_ts;
             ALTER TABLE messages ADD COLUMN timestamp TEXT;
             UPDATE messages SET timestamp = CAST(old_ts AS TEXT);",
            &["messages.timestamp", "REAL"][..],
        ),
    ] {
        let dir = TestDir::new();
        let state_db = dir.private_path("state.db");
        hermes::small_history(&state_db)
            .conn()
            .execute_batch(change)
            .unwrap();
        let corpus = dir.private_path("corpus/main.jsonl");
        let output = import(&dir, &state_db, &corpus);
        for word in words {
            assert_refused(&output, word);
        }
        assert!(!corpus.exists(), "a refused import writes no corpus");
    }
}

/// Schema 30 is 31 without seven columns the importer doesn't read, so
/// the same history imports to the same events at either version.
#[test]
fn a_schema_30_state_db_imports_like_31() {
    let dir = TestDir::new();
    let v31 = dir.private_path("state-31.db");
    hermes::small_history(&v31);
    let v30 = dir.private_path("state-30.db");
    hermes::small_history(&v30)
        .conn()
        .execute_batch(
            "ALTER TABLE sessions DROP COLUMN created_source;
             ALTER TABLE sessions DROP COLUMN compression_overload_streak;
             ALTER TABLE sessions DROP COLUMN auto_archived;
             ALTER TABLE messages DROP COLUMN message_uid;
             ALTER TABLE messages DROP COLUMN absorbed_message_uids;
             ALTER TABLE messages DROP COLUMN tool_call_uids;
             ALTER TABLE messages DROP COLUMN tool_call_uid;
             UPDATE schema_version SET version = 30;",
        )
        .unwrap();
    let [corpus_31, corpus_30] = [(&v31, "31"), (&v30, "30")].map(|(state_db, version)| {
        let corpus = dir.private_path(&format!("corpus/state-{version}.jsonl"));
        assert_ok(&import(&dir, state_db, &corpus));
        corpus
    });
    assert_eq!(events(&corpus_30), events(&corpus_31));
}

// Privacy.

/// The importer never copies the system prompt or `api_content`: not into
/// the corpus, not into its output, not into any file under the private
/// dir other than the `state.db` copy itself.
#[test]
fn the_importer_never_copies_the_system_prompt_or_api_content() {
    let dir = TestDir::new();
    let state_db = small_history(&dir);
    let output = import(&dir, &state_db, &dir.private_path("corpus/main.jsonl"));
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
        } else if !path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("state.db")
        {
            let text = String::from_utf8_lossy(&fs::read(&path).unwrap()).into_owned();
            for sentinel in sentinels {
                assert!(
                    !text.contains(sentinel),
                    "{} holds {sentinel}",
                    path.display()
                );
            }
        }
    }
}

/// The corpus hash in every report means something only if the same input
/// always imports to the same bytes.
#[test]
fn importing_the_same_history_twice_writes_the_same_corpus() {
    let dir = TestDir::new();
    let state_db = small_history(&dir);
    let [first, second] = ["first", "second"].map(|name| {
        let corpus = dir.private_path(&format!("corpus/{name}.jsonl"));
        assert_ok(&import(&dir, &state_db, &corpus));
        fs::read(corpus).unwrap()
    });
    assert_eq!(first, second);
}

/// `--dry-run` prints the import's counts and writes nothing. The counts
/// hold no text, so they can be shared to check the importer's
/// assumptions against a real `state.db` (docs/replay.md).
#[test]
fn a_dry_run_prints_counts_without_text_and_writes_nothing() {
    let dir = TestDir::new();
    let state_db = small_history(&dir);
    let corpus = dir.private_path("corpus/main.jsonl");
    let output = import_with(&dir, &state_db, &corpus, hermes::MANIFEST, &["--dry-run"]);
    assert_ok(&output);
    assert!(!corpus.exists(), "a dry run writes no corpus");
    let written: Vec<String> = fs::read_dir(dir.private())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name != "state.db" && name != "manifest.toml" && name != "corpus")
        .collect();
    assert!(written.is_empty(), "a dry run wrote {written:?}");
    let mut corpus_dir = fs::read_dir(dir.private().join("corpus")).unwrap();
    assert!(corpus_dir.next().is_none(), "a dry run wrote into corpus/");

    for text in [
        hermes::HOME_QUOTE,
        "harbour",
        "curry",
        "winter",
        "weekend",
        "Daily digest",
        "Subagent",
        "Noted.",
        SYSTEM_PROMPT_SENTINEL,
        PROMPT_TABLE_SENTINEL,
        API_CONTENT_SENTINEL,
    ] {
        assert!(
            !stdout(&output).contains(text),
            "the dry run printed {text:?}"
        );
        assert!(
            !stderr(&output).contains(text),
            "the dry run logged {text:?}"
        );
    }
}

/// The manifest's `[[model]]` tables reach the corpus, so the corpus hash
/// covers them: a different question is a different corpus.
#[test]
fn manifest_models_change_the_corpus() {
    let dir = TestDir::new();
    let state_db = small_history(&dir);
    let corpus = |name: &str, manifest: &str| {
        let corpus = dir.private_path(&format!("corpus/{name}.jsonl"));
        assert_ok(&import_with(&dir, &state_db, &corpus, manifest, &[]));
        fs::read(corpus).unwrap()
    };
    let plain = corpus("plain", hermes::MANIFEST);
    let home = corpus(
        "home",
        &support::model_manifest("Where does the user live?"),
    );
    let work = corpus(
        "work",
        &support::model_manifest("Where does the user work?"),
    );
    assert_ne!(plain, home, "the models change the corpus");
    assert_ne!(home, work, "a model's question changes the corpus");
}

// Import validation and private-input handling.

/// Content is logged only at `trace`: a manifest that doesn't parse is
/// refused naming the file and the line, a multimodal row whose parts
/// aren't JSON, a manifest that repeats a model's name, and a timezone that
/// isn't one are each refused, never quoting the input. A model name is the
/// manifest's, and nothing in the manifest leaves except into the corpus
/// header. No corpus is written.
#[test]
fn bad_import_input_is_refused_without_quoting_it() {
    let dir = TestDir::new();
    let history = small_history(&dir);
    let multimodal_row = dir.private_path("multimodal.db");
    hermes::one_session(&multimodal_row).turn(
        "s1",
        start(),
        "\u{0}json:[{\"type\": \"text\", \"text\": \"SENTINEL-MULTIMODAL-TEXT-91c",
        "Okay.",
    );

    let unparsed = format!(
        "{}\nnickname = SENTINEL-PRIVATE-NAME-4e1\n",
        hermes::MANIFEST
    );
    let line = format!("line {}", unparsed.lines().count());
    let model = "\n[[model]]\nname = \"SENTINEL-MODEL-NAME-6b3\"\nquestion = \"Where does the user live?\"\nmax_tokens = 100\n";
    let zone = hermes::MANIFEST.replace("\"Pacific/Auckland\"", "\"SENTINEL/Zone-0d8\"");
    let cases = [
        (
            &history,
            unparsed.clone(),
            "SENTINEL-PRIVATE-NAME",
            vec!["manifest.toml", line.as_str()],
        ),
        (
            &multimodal_row,
            hermes::MANIFEST.to_string(),
            "SENTINEL-MULTIMODAL",
            vec![],
        ),
        (
            &history,
            format!("{}{model}{model}", hermes::MANIFEST),
            "SENTINEL-MODEL",
            vec!["model"],
        ),
        (&history, zone, "SENTINEL/Zone", vec!["timezone"]),
    ];
    let corpus = dir.private_path("corpus/main.jsonl");
    for (state_db, manifest, sentinel, words) in cases {
        let output = import_with(&dir, state_db, &corpus, &manifest, &[]);
        assert_refused_without(&output, sentinel, &words);
        assert!(
            !corpus.exists(),
            "{sentinel}: a refused import writes no corpus"
        );
    }
}

/// The manifest and the `state.db` copy
/// are private, so each is refused outside the private dir, and through a
/// symlink inside it that points outside. Nothing is read: the manifest's
/// sentinel never appears, and no corpus is written.
#[test]
fn import_inputs_outside_the_private_dir_are_refused() {
    let dir = TestDir::new();
    let sentinel = "SENTINEL-OUTSIDE-MANIFEST-3d7";
    let private_db = small_history(&dir);
    let private_manifest = dir.private_file("manifest.toml", hermes::MANIFEST);
    let outside_db = dir.path("outside-state.db");
    hermes::small_history(&outside_db);
    let outside_manifest = dir.file(
        "outside-manifest.toml",
        &format!("{}\nnickname = {sentinel}\n", hermes::MANIFEST),
    );
    let link = |target: &Path, name: &str| {
        let link = dir.private().join(name);
        std::os::unix::fs::symlink(target, &link).unwrap();
        link
    };
    let linked_db = link(&outside_db, "linked-state.db");
    let linked_manifest = link(&outside_manifest, "linked-manifest.toml");

    let corpus = dir.private_path("corpus/main.jsonl");
    for (state_db, manifest) in [
        (&private_db, &outside_manifest),
        (&private_db, &linked_manifest),
        (&outside_db, &private_manifest),
        (&linked_db, &private_manifest),
    ] {
        let output = import_paths(&dir, state_db, manifest, &corpus, &[]);
        assert_refused_without(&output, sentinel, &["private"]);
        assert!(
            !corpus.exists(),
            "{} with {} wrote a corpus",
            state_db.display(),
            manifest.display()
        );
    }
}
