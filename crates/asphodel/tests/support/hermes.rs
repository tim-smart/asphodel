//! A synthetic Hermes `state.db`, built from scratch for the importer and
//! real-history replay tests. Nothing here is adapted from a real session.
//!
//! The DDL is the `sessions`, `messages`, `system_prompts` and
//! `schema_version` tables of hermes-agent's `hermes_state_common.py`
//! (schema version 31), every column included, so the importer
//! is checked against the shape it will meet rather than the columns it
//! reads.

use std::path::Path;

use rusqlite::{Connection, params};

/// The Hermes schema version represented by the DDL below.
pub const SCHEMA_VERSION: i64 = 31;

const DDL: &str = "
CREATE TABLE schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE system_prompts (
    hash TEXT PRIMARY KEY,
    prompt TEXT NOT NULL
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    created_source TEXT,
    user_id TEXT,
    session_key TEXT,
    chat_id TEXT,
    chat_type TEXT,
    thread_id TEXT,
    display_name TEXT,
    origin_json TEXT,
    expiry_finalized INTEGER DEFAULT 0,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    system_prompt_hash TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    cwd TEXT,
    git_branch TEXT,
    git_repo_root TEXT,
    git_metadata_generation INTEGER NOT NULL DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    title_source TEXT,
    last_activity_at REAL,
    last_activity_description TEXT,
    last_activity_provenance TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    compression_failure_cooldown_until REAL,
    compression_failure_error TEXT,
    compression_fallback_streak INTEGER NOT NULL DEFAULT 0,
    compression_ineffective_count INTEGER NOT NULL DEFAULT 0,
    compression_recovery_deadline REAL,
    compression_overload_streak INTEGER NOT NULL DEFAULT 0,
    profile_name TEXT,
    transport_profile TEXT,
    rewind_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    auto_archived INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    hidden INTEGER NOT NULL DEFAULT 0,
    last_read_at REAL,
    tool_names TEXT,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id),
    FOREIGN KEY (system_prompt_hash) REFERENCES system_prompts(hash)
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    effect_disposition TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    platform_message_id TEXT,
    observed INTEGER DEFAULT 0,
    _compressed_summary INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0,
    api_content TEXT,
    display_kind TEXT,
    display_metadata TEXT,
    display_identity BLOB,
    display_order INTEGER,
    message_uid TEXT,
    absorbed_message_uids TEXT,
    tool_call_uids TEXT,
    tool_call_uid TEXT
);
";

/// Text that must never leave `state.db`: the system prompt (in both places
/// Hermes keeps it) and `api_content`, which carries injected memory.
pub const SYSTEM_PROMPT_SENTINEL: &str = "SENTINEL-SESSION-SYSTEM-PROMPT-7f3a";
pub const PROMPT_TABLE_SENTINEL: &str = "SENTINEL-PROMPT-TABLE-1c9e";
pub const API_CONTENT_SENTINEL: &str = "SENTINEL-API-CONTENT-5b20";

/// One `messages` row. `Default` is an active, uncompacted row.
#[derive(Debug, Clone)]
pub struct Message<'a> {
    pub session: &'a str,
    pub role: &'a str,
    pub content: &'a str,
    /// Epoch seconds.
    pub at: f64,
    pub tool_calls: Option<&'a str>,
    pub active: bool,
    pub compacted: bool,
    pub summary: bool,
    pub api_content: Option<&'a str>,
}

impl Default for Message<'_> {
    fn default() -> Self {
        Self {
            session: "",
            role: "user",
            content: "",
            at: 0.0,
            tool_calls: None,
            active: true,
            compacted: false,
            summary: false,
            api_content: None,
        }
    }
}

pub struct StateDb {
    conn: Connection,
}

impl StateDb {
    /// A fresh `state.db` at `path` at [`SCHEMA_VERSION`], with the
    /// sentinel system prompt in `system_prompts`.
    pub fn create(path: &Path) -> Self {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(DDL).unwrap();
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?1)",
            [SCHEMA_VERSION],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO system_prompts (hash, prompt) VALUES ('h1', ?1)",
            [PROMPT_TABLE_SENTINEL],
        )
        .unwrap();
        Self { conn }
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// A session row with the sentinel system prompt.
    pub fn session(
        &self,
        id: &str,
        source: &str,
        user_id: Option<&str>,
        parent: Option<&str>,
        started_at: f64,
    ) {
        self.conn
            .execute(
                "INSERT INTO sessions (id, source, user_id, parent_session_id, started_at,
                     system_prompt, system_prompt_hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'h1')",
                params![
                    id,
                    source,
                    user_id,
                    parent,
                    started_at,
                    SYSTEM_PROMPT_SENTINEL
                ],
            )
            .unwrap();
    }

    pub fn message(&self, message: Message<'_>) {
        self.conn
            .execute(
                "INSERT INTO messages (session_id, role, content, timestamp, tool_calls,
                     active, compacted, _compressed_summary, api_content)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    message.session,
                    message.role,
                    message.content,
                    message.at,
                    message.tool_calls,
                    message.active,
                    message.compacted,
                    message.summary,
                    message.api_content,
                ],
            )
            .unwrap();
    }

    /// One plain turn: a user row and the assistant's reply 30 seconds
    /// later, with the sentinel in the user row's `api_content`.
    pub fn turn(&self, session: &str, at: f64, user: &str, assistant: &str) {
        self.message(Message {
            session,
            role: "user",
            content: user,
            at,
            api_content: Some(API_CONTENT_SENTINEL),
            ..Message::default()
        });
        self.message(Message {
            session,
            role: "assistant",
            content: assistant,
            at: at + 30.0,
            ..Message::default()
        });
    }

    /// A session of the owner's on Discord.
    pub fn owner_session(&self, id: &str, at: f64) {
        self.session(id, "discord", Some("discord:1"), None, at);
    }

    /// The turn that says [`HOME_QUOTE`].
    pub fn home_turn(&self, session: &str, at: f64) {
        self.turn(
            session,
            at,
            &format!("{HOME_QUOTE}, near the harbour."),
            "Noted.",
        );
    }

    /// A turn whose assistant calls a tool before it answers: the
    /// intermediate text and the tool's result are rows of their own.
    pub fn tool_call_turn(&self, session: &str, t: f64) {
        let calls =
            r#"[{"id":"c1","type":"function","function":{"name":"calendar","arguments":"{}"}}]"#;
        let rows = [
            ("user", "What's on my calendar tomorrow?", 0.0, None),
            ("assistant", "INTERMEDIATE-ASSISTANT-TEXT", 5.0, Some(calls)),
            ("tool", "TOOL-RESULT-TEXT", 6.0, None),
            ("assistant", "A dentist appointment at ten.", 20.0, None),
        ];
        for (role, content, offset, tool_calls) in rows {
            let at = t + offset;
            self.message(Message {
                session,
                role,
                content,
                at,
                tool_calls,
                ..Message::default()
            });
        }
    }

    /// The shape a compaction that carries a verbatim tail leaves, in row
    /// id order: the turn before the tail archived (`active=0,
    /// compacted=1`), the tail's originals rewound (`active=0,
    /// compacted=0`), then the summary and the tail's clones as fresh
    /// active rows at the tail's times.
    pub fn carried_tail_compaction(&self, session: &str, t: f64) {
        let rows = [
            ("user", "Archived question.", 0.0, false, true, false),
            ("assistant", "Archived answer.", 30.0, false, true, false),
            ("user", "Carried question.", 600.0, false, false, false),
            ("assistant", "Carried answer.", 630.0, false, false, false),
            (
                "user",
                "HERMES-COMPACTION-SUMMARY",
                1200.0,
                true,
                false,
                true,
            ),
            ("user", "Carried question.", 600.0, true, false, false),
            ("assistant", "Carried answer.", 630.0, true, false, false),
        ];
        for (role, content, offset, active, compacted, summary) in rows {
            self.message(Message {
                session,
                role,
                content,
                at: t + offset,
                active,
                compacted,
                summary,
                ..Message::default()
            });
        }
    }
}

/// Epoch seconds of an RFC 3339 instant.
pub fn epoch(at: &str) -> f64 {
    at.parse::<jiff::Timestamp>().unwrap().as_second() as f64
}

/// When the synthetic histories start.
pub fn start() -> f64 {
    epoch("2026-01-05T09:00:00Z")
}

pub const DAY: f64 = 24.0 * 60.0 * 60.0;

/// A fresh `state.db` with one owner session, `s1`, from [`start`].
pub fn one_session(path: &Path) -> StateDb {
    let db = StateDb::create(path);
    db.owner_session("s1", start());
    db
}

/// The manifest every test imports with: Tim owns the
/// bank on `discord:1`, and Sam is a known other speaker. Bob, who appears
/// in a prefix, isn't listed.
pub const MANIFEST: &str = r#"
timezone = "Pacific/Auckland"
bank = "main"
assistant = "Hermes"

[owner]
name = "Tim"
platform_ids = ["discord:1"]

[[speaker]]
name = "Sam"
id = "discord:2"
"#;

/// The one sentence the scripted LLM extracts, from the one turn that
/// quotes it.
pub const HOME_QUOTE: &str = "I live in Auckland";
pub const HOME_SENTENCE: &str = "Tim lives in Auckland.";

/// A small history over a week: a primary session with the home turn and a
/// few more, a cron session and a subagent session.
pub fn small_history(path: &Path) -> StateDb {
    let db = StateDb::create(path);
    let start = start();
    db.owner_session("s-main", start);
    db.home_turn("s-main", start);
    db.turn(
        "s-main",
        start + DAY,
        "What should I cook tonight?",
        "Try a curry.",
    );
    db.turn(
        "s-main",
        start + 2.0 * DAY,
        "Remind me what the weather does in winter.",
        "It gets wet and windy.",
    );

    db.session("s-cron", "cron", None, None, start + 3.0 * DAY);
    db.turn(
        "s-cron",
        start + 3.0 * DAY,
        "Daily digest of upcoming things.",
        "Nothing new today.",
    );

    let sub = start + 4.0 * DAY;
    db.session("s-sub", "discord", Some("discord:1"), Some("s-main"), sub);
    db.turn(
        "s-sub",
        sub,
        "Subagent task: summarise the cooking notes.",
        "Done.",
    );

    db.owner_session("s-later", start + 5.0 * DAY);
    db.turn(
        "s-later",
        start + 5.0 * DAY,
        "Any plans for the weekend?",
        "A walk, maybe.",
    );
    db
}

/// [`small_history`] and two more sessions, the shared fixture of the
/// plugin's backfill tests: one compacted in place with a carried tail, and
/// one with tool rows, an injected memory block, speaker prefixes, a
/// backfilled channel and a multimodal message. `plugin_fixture.rs` checks
/// it into `plugin/tests/fixtures/`.
pub fn backfill_history(path: &Path) -> StateDb {
    let db = small_history(path);
    let t = start() + 6.0 * DAY;
    db.owner_session("s-compacted", t);
    db.carried_tail_compaction("s-compacted", t);
    db.turn(
        "s-compacted",
        t + 1800.0,
        "After the compaction.",
        "Still here.",
    );

    let t = start() + 7.0 * DAY;
    db.owner_session("s-mixed", t);
    db.tool_call_turn("s-mixed", t);
    db.turn(
        "s-mixed",
        t + 600.0,
        "<memory-context>HINDSIGHT-INJECTED-MEMORY</memory-context>\nI started learning the cello.",
        "Good luck.",
    );
    db.turn(
        "s-mixed",
        t + 1200.0,
        "[Sam] I'm Tim's friend from Wellington.",
        "Hi Sam.",
    );
    db.turn("s-mixed", t + 1800.0, "[Bob] Is it raining?", "Not yet.");
    db.turn(
        "s-mixed",
        t + 2400.0,
        "[Alice] BACKFILLED-CHANNEL-HISTORY\n[New message] [Sam] See you at noon.",
        "See you then.",
    );
    let garden = serde_json::json!([
        {"type": "text", "text": "Look at my garden."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE-BYTES-SENTINEL"}}
    ]);
    db.turn(
        "s-mixed",
        t + 3000.0,
        &format!("\u{0}json:{garden}"),
        "Lovely roses.",
    );
    db
}
