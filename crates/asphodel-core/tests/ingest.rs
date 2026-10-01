//! Ingest, checked against "Ingest: sources, chunking, secret scanning and
//! the extraction queue" (TIM-106) and the decisions it rests on: "What is a
//! memory record?" (TIM-90, sources), "Extraction: significance, validity
//! windows and supersession" (TIM-92, inputs, documents and chunks, edited
//! documents, other decision 1), "API surface and Hermes transport" (TIM-94,
//! decisions 1 and 5), "Replay harness" (TIM-96, the `[New message]`
//! amendment), "Operations" (TIM-99, decision 2), "Configuration surface"
//! (TIM-98, the extraction constants), and ADRs 0002 and 0010.
//!
//! The code under test doesn't exist yet. [`contract`] below holds the
//! proposed API with `todo!()` bodies, so this file compiles and every test
//! that needs it is ignored. To activate: move the contract into the crate
//! (`asphodel_core::{ingest, chunking, secrets, queue}`, the two constants
//! into `asphodel_core::constants`, and the free functions onto `Service`),
//! delete the module, import from the crate instead, and drop the `ignore`
//! attributes.
//!
//! Two tests run now. They check that the schema TIM-103 landed has what
//! ingest needs, so the implementation shouldn't need a migration.
//!
//! The credentials in the secret-scanning tests are built at runtime from
//! pieces, so no literal in this file looks like a real credential to a
//! scanner, GitHub push protection included.
//!
//! Every service here runs on a `SimulatedClock` stopped at one instant, so
//! a stored time that equals that instant can only have come from the Clock.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asphodel_core::Service;
use asphodel_core::clock::{Clock, SimulatedClock};
use asphodel_core::config::Tuning;
use asphodel_core::store::bank::{BankIdentity, ModelIds};
use asphodel_core::store::{DB_FILE, OpenOptions, Store, micros};
use jiff::civil::{Date, date};
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use rusqlite::types::FromSql;
use uuid::Uuid;

use contract::chunking::{DocumentChunk, chunk_hash, split_document};
use contract::ingest::{Document, IngestError, Ingested, Outcome, Turn, TurnAuthor};
use contract::queue::{ChunkError, Failure, Lease, SourceKind};
use contract::secrets::{SecretKind, scan};
use contract::*;

/// The proposed ingest API.
///
/// Ingest receives a turn or a document, scans it for secrets, stores it as a
/// source, splits it into chunks and queues the chunks for extraction. It
/// never calls a model: extraction (TIM-107, TIM-108) takes chunks off the
/// queue. Everything is synchronous and runs on the service's clock, so
/// `serve` and replay share it (TIM-96, decision 3).
#[allow(dead_code, unused_variables)]
mod contract {
    use asphodel_core::Service;

    use self::ingest::{Document, IngestError, Ingested, Turn};
    use self::queue::{ChunkError, FailedChunk, Failure, Lease, QueueError};

    /// The proposed `asphodel_core::constants::CHUNK_CHARS`: the size a
    /// document section is split down to when it's too long, in characters
    /// (TIM-92: "at about 3,000 characters"). Fixed in code (TIM-98).
    pub const CHUNK_CHARS: usize = 3_000;

    /// The proposed `asphodel_core::constants::CHUNK_RETRY_CAP`: how many
    /// failed attempts mark a chunk `failed` (TIM-92). Fixed in code
    /// (TIM-98). No ticket gives the number; 5 is a proposal.
    pub const CHUNK_RETRY_CAP: u32 = 5;

    /// The proposed `asphodel_core::secrets`: the regex scan that runs
    /// before anything is stored (ADR 0002, TIM-92 other decision 1).
    pub mod secrets {
        use std::collections::BTreeSet;

        /// The kinds of secret the scan recognises. The name of each kind is
        /// what's recorded on the source and shown by `memory show` (ADR
        /// 0010), and what its marker names.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum SecretKind {
            /// A PEM or OpenSSH private key block, BEGIN to END.
            PrivateKey,
            /// `AKIA` or `ASIA` and 16 upper-case alphanumerics.
            AwsAccessKey,
            /// `ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_` or `github_pat_` tokens.
            GithubToken,
            /// `sk-` and `sk-proj-` keys, but not `sk-ant-`.
            OpenAiKey,
            /// `sk-ant-` keys.
            AnthropicKey,
            /// `xoxb-`, `xoxp-`, `xoxa-`, `xoxr-` and `xoxs-` tokens.
            SlackToken,
            /// `sk_live_`, `sk_test_`, `rk_live_` and `rk_test_` keys.
            StripeKey,
            /// `AIza` and 35 URL-safe characters.
            GoogleApiKey,
            /// Three dot-separated base64url segments, the first two `eyJ`.
            Jwt,
            /// The password in a URL's userinfo (`scheme://user:password@host`).
            /// Only the password is redacted; the user and host stay.
            UrlPassword,
        }

        impl SecretKind {
            pub const ALL: [SecretKind; 10] = [
                SecretKind::PrivateKey,
                SecretKind::AwsAccessKey,
                SecretKind::GithubToken,
                SecretKind::OpenAiKey,
                SecretKind::AnthropicKey,
                SecretKind::SlackToken,
                SecretKind::StripeKey,
                SecretKind::GoogleApiKey,
                SecretKind::Jwt,
                SecretKind::UrlPassword,
            ];

            /// The snake_case name: `private_key`, `aws_access_key`,
            /// `github_token`, `openai_key`, `anthropic_key`, `slack_token`,
            /// `stripe_key`, `google_api_key`, `jwt`, `url_password`.
            pub fn as_str(self) -> &'static str {
                todo!()
            }

            /// The text a match is replaced with. It names the kind, and the
            /// scan never flags a marker itself.
            pub fn marker(self) -> String {
                todo!()
            }
        }

        /// What one scan found.
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct Scan {
            /// The input with every match replaced in place by its kind's
            /// marker. Equal to the input when nothing matched.
            pub text: String,
            /// The kinds that fired, each once however often it matched.
            pub kinds: BTreeSet<SecretKind>,
        }

        /// Scans `text` and redacts every match. Pure: no clock, no store.
        pub fn scan(text: &str) -> Scan {
            todo!()
        }
    }

    /// The proposed `asphodel_core::chunking` (TIM-92, documents and chunks).
    pub mod chunking {
        /// One chunk of a document, before it's stored.
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct DocumentChunk {
            /// The headings above this chunk, outermost first, without their
            /// `#` marks. Empty for text before the first heading and for a
            /// document with no headings.
            pub heading_path: Vec<String>,
            /// The chunk's range in the document, in characters (TIM-90:
            /// "character offsets"), end exclusive.
            pub start: usize,
            pub end: usize,
            /// The document's characters `start..end`.
            pub text: String,
        }

        /// Splits a plain-text or markdown document: one chunk per section
        /// (a section runs from an ATX heading to the next heading of any
        /// level), split further at paragraph breaks only when a section is
        /// longer than [`CHUNK_CHARS`](super::CHUNK_CHARS). A paragraph
        /// longer than that on its own is cut at line or sentence breaks,
        /// or anywhere as a last resort. Small sections are never merged.
        /// Lines inside fenced code blocks are never headings.
        pub fn split_document(text: &str) -> Vec<DocumentChunk> {
            todo!()
        }

        /// A chunk's identity: a hash of its text and its heading path, fixed
        /// at ingest and never recomputed. It's the forget tombstone, not an
        /// integrity check (TIM-92). Different paths never collide however
        /// their headings are spelled.
        pub fn chunk_hash(heading_path: &[String], text: &str) -> String {
            todo!()
        }
    }

    /// The proposed `asphodel_core::ingest`.
    pub mod ingest {
        use std::collections::BTreeSet;

        use asphodel_core::store::StoreError;
        use jiff::Timestamp;
        use jiff::civil::Date;
        use uuid::Uuid;

        use super::secrets::SecretKind;

        /// Hermes' `turn_author` (TIM-94, decision 1).
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct TurnAuthor {
            /// The platform's id for the speaker, such as a Discord user id.
            /// Ingest resolves the speaker by the alias `<platform>:<id>`,
            /// the form the owner's platform ids take in bank config.
            pub id: String,
            pub name: Option<String>,
            pub is_bot: bool,
        }

        /// What `sync_turn` sends (TIM-94, decision 5 as amended by TIM-96
        /// and TIM-99). The plugin has already stripped any backfilled
        /// channel history before `[New message]`; the `[Name] ` prefix in
        /// shared threads stays and is stored as sent.
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct Turn {
            pub session_id: String,
            /// The user message's epoch timestamp. It's the turn's
            /// `observed_at` and part of its key.
            pub message_at: Timestamp,
            /// `hermes_time.get_timezone_name()`. The bank's default when
            /// absent.
            pub timezone: Option<String>,
            /// The clean user message, never `api_content`.
            pub user_text: String,
            pub assistant_text: String,
            /// `None` on the CLI, TUI and Hermes UI, where the turn is the
            /// owner's.
            pub author: Option<TurnAuthor>,
            pub platform: Option<String>,
            /// The `recall_id` the plugin echoes from `prefetch`.
            pub recall_id: Option<String>,
            /// The turn called `memory_forget` (ADR 0010). It's stored only
            /// as a tombstone, never queued, and its recall row is deleted.
            pub forget_requested: bool,
        }

        /// A document sent through the API or `asphodel ingest`.
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct Document {
            /// `--id`; re-ingesting the same id with new text is an edit.
            pub document_id: String,
            /// Plain text or markdown.
            pub text: String,
            /// `--date`. The document's `observed_at` is the start of this
            /// day in its timezone.
            pub reference_date: Date,
            /// `false` with `--inexact` (TIM-92).
            pub reference_date_exact: bool,
            /// The bank's default when absent.
            pub timezone: Option<String>,
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Outcome {
            /// A new source, stored and its chunks queued.
            Stored,
            /// The key was already there: nothing changed (TIM-90).
            Duplicate,
            /// A `forget_requested` turn, stored as a tombstone only.
            Tombstone,
        }

        /// Whose words a turn holds (TIM-94, decision 1).
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct Speaker {
            /// The speaker's entity: the seeded `user` for the owner,
            /// otherwise an entity of their own.
            pub entity: Uuid,
            pub owner: bool,
        }

        /// What one ingest did.
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct Ingested {
            /// The source's public id; the existing one on a duplicate.
            pub source: Uuid,
            pub outcome: Outcome,
            /// Chunks queued for extraction by this call.
            pub chunks_queued: usize,
            /// Chunks of an edited document already seen in an earlier
            /// version of it, and so not queued (TIM-92).
            pub chunks_skipped: usize,
            /// The secret kinds that fired on what this call stored.
            pub secret_kinds: BTreeSet<SecretKind>,
            /// The resolved speaker of a stored turn; `None` for documents,
            /// duplicates and tombstones.
            pub speaker: Option<Speaker>,
        }

        /// Why an ingest was refused. Nothing is stored when it is, and no
        /// variant carries content (ADR 0010).
        #[derive(Debug, thiserror::Error)]
        pub enum IngestError {
            #[error("unknown bank")]
            UnknownBank,
            #[error("unknown timezone")]
            InvalidTimezone,
            #[error(transparent)]
            Store(#[from] StoreError),
        }
    }

    /// The proposed `asphodel_core::queue`: the durable extraction queue in
    /// the `extraction_queue` table (TIM-92, TIM-94 decision 3).
    pub mod queue {
        use asphodel_core::store::StoreError;
        use jiff::Timestamp;
        use uuid::Uuid;

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum SourceKind {
            Turn,
            Document,
        }

        /// A chunk handed to a bank's one worker. While a bank has a lease
        /// out, it hands out no other: one worker per bank. Leases live in
        /// the service, not the store, so a restart releases them and the
        /// chunk is claimed again without counting an error.
        #[derive(Debug, PartialEq, Eq)]
        pub struct Lease {
            pub chunk: Uuid,
            pub source: Uuid,
            pub source_kind: SourceKind,
            pub position: u32,
            pub observed_at: Timestamp,
            /// Failed attempts so far, across restarts.
            pub error_count: u32,
        }

        /// Why an attempt failed: the error kind and HTTP status, never the
        /// response (ADR 0010). `kind` is static so it can't carry content.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct ChunkError {
            pub kind: &'static str,
            pub status: Option<u16>,
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Failure {
            /// The chunk stays at its place in the queue and is retried.
            Retry { error_count: u32 },
            /// The cap was reached: the chunk is marked failed, leaves the
            /// queue and is surfaced instead of retried.
            Failed,
        }

        /// A chunk that reached the retry cap, as `chunks --failed` lists it.
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct FailedChunk {
            pub chunk: Uuid,
            pub source: Uuid,
            pub error_count: u32,
            pub error_kind: String,
            pub status: Option<u16>,
            pub failed_at: Timestamp,
        }

        #[derive(Debug, thiserror::Error)]
        pub enum QueueError {
            #[error("unknown bank")]
            UnknownBank,
            #[error(transparent)]
            Store(#[from] StoreError),
        }
    }

    /// The proposed `Service::ingest_turn(bank, turn)`.
    pub fn ingest_turn(
        service: &Service,
        bank: &str,
        turn: &Turn,
    ) -> Result<Ingested, IngestError> {
        todo!()
    }

    /// The proposed `Service::ingest_document(bank, document)`.
    pub fn ingest_document(
        service: &Service,
        bank: &str,
        document: &Document,
    ) -> Result<Ingested, IngestError> {
        todo!()
    }

    /// The proposed `Service::claim_chunk(bank)`: the head of the bank's
    /// queue, or `None` when the queue is empty or the bank's worker already
    /// holds a lease. Order: turns ahead of document chunks, then
    /// `observed_at`, then `ingested_at`, then rowid (TIM-92).
    pub fn claim_chunk(service: &Service, bank: &str) -> Result<Option<Lease>, QueueError> {
        todo!()
    }

    /// The proposed `Service::complete_chunk(lease)`: sets `extracted_at`,
    /// drops call 1's saved output (ADR 0008) and takes the chunk off the
    /// queue. Extraction will want this inside its own commit transaction
    /// (TIM-92); that variant belongs to TIM-107.
    pub fn complete_chunk(service: &Service, lease: Lease) -> Result<(), QueueError> {
        todo!()
    }

    /// The proposed `Service::fail_chunk(lease, error)`: counts the error
    /// on the chunk and either leaves it at the head for a retry or, at
    /// [`CHUNK_RETRY_CAP`], marks it failed.
    pub fn fail_chunk(
        service: &Service,
        lease: Lease,
        error: ChunkError,
    ) -> Result<Failure, QueueError> {
        todo!()
    }

    /// The proposed `Service::queue_depth(bank)`: chunks waiting or in
    /// flight, not counting failed ones. `asphodel status` shows it.
    pub fn queue_depth(service: &Service, bank: &str) -> Result<usize, QueueError> {
        todo!()
    }

    /// The proposed `Service::failed_chunks(bank)`, oldest failure first.
    pub fn failed_chunks(service: &Service, bank: &str) -> Result<Vec<FailedChunk>, QueueError> {
        todo!()
    }
}

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";

fn start() -> Timestamp {
    START.parse().unwrap()
}

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
}

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-ingest-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn data(&self) -> PathBuf {
        self.0.join("data")
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn models() -> ModelIds {
    ModelIds {
        embedding: "bge-small-en-v1.5:int8".into(),
        reranker: "jina-reranker-v1-turbo-en:int8".into(),
    }
}

/// The owner is Tim, on Discord as `discord:1234`.
fn identity() -> BankIdentity {
    BankIdentity {
        owner_name: Some("Tim".into()),
        owner_platform_ids: vec!["discord:1234".into()],
        assistant_name: Some("Hermes".into()),
        timezone: Some(TZ.into()),
    }
}

/// A service on a data dir with two banks, `main` and `other`. Field order
/// matters: the service drops before the directory it lives in.
struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    dir: TestDir,
}

impl Harness {
    fn new() -> Self {
        let dir = TestDir::new();
        let clock = Arc::new(SimulatedClock::new(start()));
        let service = open_service(&dir, clock.clone());
        service.ensure_bank("main", &identity(), &models()).unwrap();
        service
            .ensure_bank("other", &identity(), &models())
            .unwrap();
        Self {
            service,
            clock,
            dir,
        }
    }

    /// Drops the service, releasing the data-dir lock and every lease, and
    /// opens a new one on the same data dir and clock: a daemon restart.
    fn restart(self) -> Self {
        let Harness {
            service,
            clock,
            dir,
        } = self;
        drop(service);
        let service = open_service(&dir, clock.clone());
        Self {
            service,
            clock,
            dir,
        }
    }

    fn now(&self) -> i64 {
        micros(self.clock.now())
    }

    fn one<T: FromSql, P: rusqlite::Params>(&self, sql: &str, params: P) -> T {
        self.service
            .store()
            .unwrap()
            .connection()
            .query_row(sql, params, |row| row.get(0))
            .unwrap()
    }

    fn count(&self, sql: &str) -> i64 {
        self.one(sql, [])
    }

    fn execute<P: rusqlite::Params>(&self, sql: &str, params: P) -> usize {
        self.service
            .store()
            .unwrap()
            .connection()
            .execute(sql, params)
            .unwrap()
    }

    fn source_column<T: FromSql>(&self, source: Uuid, column: &str) -> T {
        self.one(
            &format!("SELECT {column} FROM sources WHERE uuid = ?1"),
            [source.to_string()],
        )
    }

    fn chunk_column<T: FromSql>(&self, chunk: Uuid, column: &str) -> T {
        self.one(
            &format!("SELECT {column} FROM chunks WHERE uuid = ?1"),
            [chunk.to_string()],
        )
    }

    /// A source's chunks in position order: (uuid, position, heading_path,
    /// start, end, text, content_hash).
    #[allow(clippy::type_complexity)]
    fn chunks_of(
        &self,
        source: Uuid,
    ) -> Vec<(Uuid, i64, Option<String>, i64, i64, Option<String>, String)> {
        let store = self.service.store().unwrap();
        let conn = store.connection();
        let mut statement = conn
            .prepare(
                "SELECT c.uuid, c.position, c.heading_path, c.start_offset, c.end_offset, c.text,
                        c.content_hash
                 FROM chunks c JOIN sources s ON s.id = c.source_id
                 WHERE s.uuid = ?1 ORDER BY c.position",
            )
            .unwrap();
        statement
            .query_map([source.to_string()], |row| {
                Ok((
                    row.get::<_, String>(0)?.parse().unwrap(),
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                ))
            })
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }

    fn seeded(&self, bank: &str, seeded: &str) -> Uuid {
        self.one::<String, _>(
            "SELECT e.uuid FROM entities e JOIN banks b ON b.id = e.bank_id
             WHERE b.name = ?1 AND e.seeded = ?2",
            [bank, seeded],
        )
        .parse()
        .unwrap()
    }

    fn entities_in(&self, bank: &str) -> i64 {
        self.one(
            "SELECT COUNT(*) FROM entities e JOIN banks b ON b.id = e.bank_id WHERE b.name = ?1",
            [bank],
        )
    }

    /// Whether `needle` appears anywhere in the database or its WAL, after
    /// a checkpoint. Nothing stored may hold a secret or a forgotten turn.
    fn store_holds(&self, needle: &str) -> bool {
        let store = self.service.store().unwrap();
        store.checkpoint().unwrap();
        let dir = store.dir().to_owned();
        [DB_FILE.to_string(), format!("{DB_FILE}-wal")]
            .iter()
            .any(|name| match std::fs::read(dir.join(name)) {
                Ok(bytes) => bytes
                    .windows(needle.len())
                    .any(|window| window == needle.as_bytes()),
                Err(_) => false,
            })
    }
}

fn open_service(dir: &TestDir, clock: Arc<SimulatedClock>) -> Service {
    let store = Store::open(&dir.data(), OpenOptions::default(), clock.clone()).unwrap();
    Service::open(clock, store, Tuning::default())
}

/// The owner's turn on the CLI: no author.
fn turn(session: &str, message_at: &str, user: &str, assistant: &str) -> Turn {
    Turn {
        session_id: session.into(),
        message_at: at(message_at),
        timezone: Some(TZ.into()),
        user_text: user.into(),
        assistant_text: assistant.into(),
        author: None,
        platform: Some("cli".into()),
        recall_id: None,
        forget_requested: false,
    }
}

/// A turn on Discord from `author`.
fn discord_turn(message_at: &str, user: &str, author: TurnAuthor) -> Turn {
    Turn {
        author: Some(author),
        platform: Some("discord".into()),
        ..turn("thread-1", message_at, user, "Noted.")
    }
}

fn author(id: &str, name: &str) -> TurnAuthor {
    TurnAuthor {
        id: id.into(),
        name: Some(name.into()),
        is_bot: false,
    }
}

fn document(id: &str, text: &str, reference_date: Date) -> Document {
    Document {
        document_id: id.into(),
        text: text.into(),
        reference_date,
        reference_date_exact: true,
        timezone: Some(TZ.into()),
    }
}

fn ingest(h: &Harness, bank: &str, turn: &Turn) -> Ingested {
    ingest_turn(&h.service, bank, turn).unwrap()
}

fn ingest_doc(h: &Harness, bank: &str, document: &Document) -> Ingested {
    ingest_document(&h.service, bank, document).unwrap()
}

fn claim(h: &Harness, bank: &str) -> Option<Lease> {
    claim_chunk(&h.service, bank).unwrap()
}

fn depth(h: &Harness, bank: &str) -> usize {
    queue_depth(&h.service, bank).unwrap()
}

const LLM_502: ChunkError = ChunkError {
    kind: "llm_status",
    status: Some(502),
};

/// Characters `start..end` of `text`.
fn chars(text: &str, start: usize, end: usize) -> String {
    text.chars().skip(start).take(end - start).collect()
}

fn path(headings: &[&str]) -> Vec<String> {
    headings.iter().map(|heading| heading.to_string()).collect()
}

// Fake credentials, built from pieces at runtime.

const ALNUM: &[u8] = b"A7kQ2mZ9xR4tW1pL8nV3bY6cH5dJ0fGsE";
const UPPER: &[u8] = b"Q7KZ2MX9R4TW1PL8NV3BY6CH5DJ0FGSE";

fn body(alphabet: &[u8], len: usize, seed: usize) -> String {
    (0..len)
        .map(|i| alphabet[(i * 7 + seed * 13 + 3) % alphabet.len()] as char)
        .collect()
}

fn fake(kind: SecretKind, seed: usize) -> String {
    match kind {
        SecretKind::PrivateKey => [
            "-----BEGIN ",
            "OPENSSH PRIVATE",
            " KEY-----\n",
            &body(ALNUM, 64, seed),
            "\n",
            &body(ALNUM, 64, seed + 1),
            "\n-----END ",
            "OPENSSH PRIVATE",
            " KEY-----",
        ]
        .concat(),
        SecretKind::AwsAccessKey => ["AK", "IA", &body(UPPER, 16, seed)].concat(),
        SecretKind::GithubToken => ["gh", "p_", &body(ALNUM, 36, seed)].concat(),
        SecretKind::OpenAiKey => ["sk-", "proj-", &body(ALNUM, 48, seed)].concat(),
        SecretKind::AnthropicKey => ["sk-", "ant-", "api03-", &body(ALNUM, 80, seed)].concat(),
        SecretKind::SlackToken => [
            "xo",
            "xb-",
            "184736291045",
            "-",
            "5729104836271",
            "-",
            &body(ALNUM, 24, seed),
        ]
        .concat(),
        SecretKind::StripeKey => ["sk", "_live_", &body(ALNUM, 24, seed)].concat(),
        SecretKind::GoogleApiKey => ["AI", "za", &body(ALNUM, 35, seed)].concat(),
        SecretKind::Jwt => [
            "ey",
            "JhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            ".",
            "ey",
            "JzdWIiOiIxMjM0NTY3ODkwIn0",
            ".",
            &body(ALNUM, 43, seed),
        ]
        .concat(),
        // Only the password; see `url_with_password`.
        SecretKind::UrlPassword => body(ALNUM, 20, seed),
    }
}

fn url_with_password(password: &str) -> String {
    format!("postgres://hermes:{password}@db.internal:5432/app")
}

/// `text` containing the fake secret of `kind`, as it would be written.
fn sentence_with(kind: SecretKind, seed: usize) -> (String, String) {
    let secret = fake(kind, seed);
    let text = match kind {
        SecretKind::UrlPassword => format!("the database is {} thanks", url_with_password(&secret)),
        _ => format!("my key is {secret} thanks"),
    };
    (text, secret)
}

// What runs now: the schema TIM-103 landed has what ingest needs.

#[test]
fn the_schema_has_the_columns_ingest_needs() {
    let h = Harness::new();
    let columns = |table: &str| -> BTreeSet<String> {
        let store = h.service.store().unwrap();
        let conn = store.connection();
        let mut statement = conn
            .prepare(&format!("SELECT name FROM pragma_table_info('{table}')"))
            .unwrap();
        statement
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };
    let expect = |table: &str, wanted: &[&str]| {
        let have = columns(table);
        for column in wanted {
            assert!(have.contains(*column), "{table}.{column} is missing");
        }
    };
    expect(
        "sources",
        &[
            "kind",
            "session_id",
            "message_at",
            "document_id",
            "content_hash",
            "platform",
            "author_id",
            "author_name",
            "author_is_bot",
            "observed_at",
            "reference_date",
            "reference_date_exact",
            "timezone",
            "text",
            "reply",
            "secret_kinds",
            "recall_id",
            "ingested_at",
            "tombstoned_at",
            "tombstone_reason",
        ],
    );
    expect(
        "chunks",
        &[
            "source_id",
            "position",
            "heading_path",
            "content_hash",
            "start_offset",
            "end_offset",
            "text",
            "call1_output",
            "extracted_at",
            "error_count",
            "last_error_kind",
            "last_error_status",
            "failed_at",
            "tombstoned_at",
        ],
    );
    expect(
        "extraction_queue",
        &[
            "bank_id",
            "kind",
            "chunk_id",
            "priority",
            "observed_at",
            "enqueued_at",
            "attempts",
        ],
    );
}

#[test]
fn a_forget_request_tombstone_fits_the_schema_and_keeps_its_key() {
    // ADR 0010: the turn that asked to forget is a tombstone from the start,
    // with no text, and its key still stops it being ingested again.
    let h = Harness::new();
    let bank: i64 = h.one("SELECT id FROM banks WHERE name = 'main'", []);
    let now = h.now();
    let insert = |uuid: &str| {
        h.service.store().unwrap().connection().execute(
            "INSERT INTO sources (uuid, bank_id, kind, session_id, message_at, content_hash,
                                  observed_at, timezone, text, reply, ingested_at,
                                  tombstoned_at, tombstone_reason)
             VALUES (?1, ?2, 'turn', 'session', ?3, 'hash', ?3, 'UTC', NULL, NULL, ?3, ?3,
                     'forget_requested')",
            (uuid, bank, now),
        )
    };
    insert("tombstone").unwrap();
    assert!(insert("again").is_err(), "the tombstone's key holds");
}

// Constants (TIM-92, TIM-98)

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
#[allow(clippy::assertions_on_constants)]
fn the_chunk_size_and_retry_cap_are_fixed_in_code() {
    assert_eq!(CHUNK_CHARS, 3_000, "TIM-92: about 3,000 characters");
    assert!(
        CHUNK_RETRY_CAP >= 2,
        "a capped retry needs at least one retry before the cap"
    );
}

// Secret scanning (ADR 0002; TIM-92, other decision 1)

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn secret_kinds_have_stable_names() {
    // These names are recorded on sources and shown by `memory show`, so
    // renaming one is a migration.
    let names: Vec<&str> = SecretKind::ALL.iter().map(|kind| kind.as_str()).collect();
    assert_eq!(
        names,
        [
            "private_key",
            "aws_access_key",
            "github_token",
            "openai_key",
            "anthropic_key",
            "slack_token",
            "stripe_key",
            "google_api_key",
            "jwt",
            "url_password",
        ]
    );
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn each_marker_names_its_kind() {
    // TIM-92: "redacted in place with a marker that names the pattern kind".
    let markers: BTreeSet<String> = SecretKind::ALL.iter().map(|kind| kind.marker()).collect();
    assert_eq!(markers.len(), SecretKind::ALL.len(), "markers are distinct");
    for kind in SecretKind::ALL {
        assert!(
            kind.marker().contains(kind.as_str()),
            "{kind:?}'s marker {:?} names it",
            kind.marker()
        );
    }
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn every_kind_is_found_and_redacted_in_place() {
    for kind in SecretKind::ALL {
        let (text, secret) = sentence_with(kind, 1);
        let found = scan(&text);
        assert_eq!(found.kinds, BTreeSet::from([kind]), "{kind:?}");
        assert!(!found.text.contains(&secret), "{kind:?} is gone");
        assert!(found.text.contains(&kind.marker()), "{kind:?}'s marker");
        assert!(found.text.ends_with(" thanks"), "{kind:?}: the rest stays");
        match kind {
            SecretKind::UrlPassword => {
                // Only the password goes; the URL stays useful.
                assert!(found.text.starts_with("the database is postgres://hermes:"));
                assert!(found.text.contains("@db.internal:5432/app"));
            }
            _ => assert!(found.text.starts_with("my key is "), "{kind:?}"),
        }
    }
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_private_key_block_goes_whole() {
    let key = fake(SecretKind::PrivateKey, 2);
    let text = format!("Here it is:\n{key}\nKeep it safe.");
    let found = scan(&text);
    assert_eq!(found.kinds, BTreeSet::from([SecretKind::PrivateKey]));
    assert_eq!(
        found.text,
        format!(
            "Here it is:\n{}\nKeep it safe.",
            SecretKind::PrivateKey.marker()
        )
    );
    for line in key.lines() {
        assert!(!found.text.contains(line), "no line of the block survives");
    }
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn an_anthropic_key_is_not_also_an_openai_key() {
    // Both start `sk-`; the more specific kind wins and fires alone.
    let (text, _) = sentence_with(SecretKind::AnthropicKey, 3);
    assert_eq!(
        scan(&text).kinds,
        BTreeSet::from([SecretKind::AnthropicKey])
    );
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn every_match_is_redacted_and_each_kind_is_recorded_once() {
    let first = fake(SecretKind::GithubToken, 4);
    let second = fake(SecretKind::GithubToken, 5);
    let aws = fake(SecretKind::AwsAccessKey, 6);
    let text = format!("old {first}, new {second}, and {aws}");
    let found = scan(&text);
    assert_eq!(
        found.kinds,
        BTreeSet::from([SecretKind::GithubToken, SecretKind::AwsAccessKey])
    );
    for secret in [&first, &second, &aws] {
        assert!(!found.text.contains(secret.as_str()));
    }
    let github = SecretKind::GithubToken.marker();
    assert_eq!(found.text.matches(&github).count(), 2);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn ordinary_text_is_left_alone() {
    // Over-redaction loses memories, so the patterns are specific.
    let text = "Commit 9f1c2d3e4b5a69788796a5b4c3d2e1f0a9b8c7d6 fixed task-list sorting.\n\
                Ticket 01a0f5ff-97b0-7689-bda0-3390158c3c2d is done.\n\
                See https://example.com/docs?page=2 and ssh://git@github.com/tim/asphodel.\n\
                AKIA is a prefix; sk- is a prefix; eyJ is a prefix.\n\
                Meet at 10:30 on 2026-10-03, password reset by Friday.";
    let found = scan(text);
    assert!(found.kinds.is_empty(), "flagged {:?}", found.kinds);
    assert_eq!(found.text, text);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_scan_of_redacted_text_finds_nothing_more() {
    // A marker is never itself a match, so re-scanning stored text (as a
    // re-extraction might) changes nothing.
    let text = SecretKind::ALL
        .iter()
        .enumerate()
        .map(|(seed, kind)| sentence_with(*kind, seed).0)
        .collect::<Vec<_>>()
        .join("\n");
    let once = scan(&text);
    assert_eq!(once.kinds.len(), SecretKind::ALL.len());
    let twice = scan(&once.text);
    assert!(twice.kinds.is_empty(), "flagged {:?}", twice.kinds);
    assert_eq!(twice.text, once.text);
}

// Chunking (TIM-92, documents and chunks)

/// Every chunk's text is its range of the document, the ranges are in order
/// and don't overlap, and every non-whitespace character is in one.
fn assert_tiles(document: &str, chunks: &[DocumentChunk]) {
    let length = document.chars().count();
    let mut covered = vec![false; length];
    let mut previous_end = 0;
    for chunk in chunks {
        assert!(chunk.start < chunk.end, "{chunk:?} is empty");
        assert!(chunk.start >= previous_end, "{chunk:?} overlaps");
        assert!(chunk.end <= length, "{chunk:?} runs past the end");
        assert_eq!(chunk.text, chars(document, chunk.start, chunk.end));
        for slot in &mut covered[chunk.start..chunk.end] {
            *slot = true;
        }
        previous_end = chunk.end;
    }
    for (index, character) in document.chars().enumerate() {
        assert!(
            covered[index] || character.is_whitespace(),
            "character {index} ({character:?}) is in no chunk"
        );
    }
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_short_plain_document_is_one_chunk() {
    let text = "Buy milk.\n\nCall the dentist about the 3 October appointment.\n";
    let chunks = split_document(text);
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].heading_path.is_empty());
    assert_tiles(text, &chunks);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_markdown_document_is_one_chunk_per_section_with_its_heading_path() {
    let text = "Notes from the week.\n\n\
                # Work\n\nShipping Asphodel v1.\n\n\
                ## Asphodel\n\nIngest is next.\n\n\
                ## Hermes\n\nThe plugin waits.\n\n\
                # Home ##\n\nThe fence needs fixing.\n\n\
                ### Garden\n\nPlant the tomatoes.\n";
    let chunks = split_document(text);
    let paths: Vec<Vec<String>> = chunks.iter().map(|c| c.heading_path.clone()).collect();
    assert_eq!(
        paths,
        [
            path(&[]),
            path(&["Work"]),
            path(&["Work", "Asphodel"]),
            path(&["Work", "Hermes"]),
            path(&["Home"]),
            path(&["Home", "Garden"]),
        ],
        "small sections are never merged, and closing #s aren't part of a heading"
    );
    assert_tiles(text, &chunks);
    assert!(
        chunks[1].text.starts_with("# Work"),
        "a section starts at its heading"
    );
    assert!(chunks[2].text.contains("Ingest is next."));
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn hashes_in_code_blocks_and_hashtags_are_not_headings() {
    let text = "# Setup\n\nRun this:\n\n```sh\n# install the daemon\nnix profile install\n```\n\n\
                #asphodel is the tag we use.\n";
    let chunks = split_document(text);
    assert_eq!(chunks.len(), 1, "{chunks:#?}");
    assert_eq!(chunks[0].heading_path, path(&["Setup"]));
    assert_tiles(text, &chunks);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_section_under_the_budget_is_not_split() {
    let paragraph = "word ".repeat(140); // 700 characters
    let section = format!("# Long\n\n{}", [paragraph.as_str(); 4].join("\n\n"));
    assert!(section.chars().count() < CHUNK_CHARS);
    let chunks = split_document(&section);
    assert_eq!(chunks.len(), 1);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_long_section_is_split_at_paragraphs_under_the_budget() {
    let paragraphs: Vec<String> = (0..12)
        .map(|n| format!("Paragraph {n}. {}", "lorem ipsum ".repeat(75)))
        .collect();
    let text = format!(
        "# Journal\n\n{}\n\n# After\n\nShort.\n",
        paragraphs.join("\n\n")
    );
    let chunks = split_document(&text);
    assert_tiles(&text, &chunks);

    let journal: Vec<&DocumentChunk> = chunks
        .iter()
        .filter(|c| c.heading_path == path(&["Journal"]))
        .collect();
    assert!(journal.len() > 1, "the section was split");
    for chunk in &journal {
        assert!(
            chunk.text.chars().count() <= CHUNK_CHARS,
            "a chunk of {} characters",
            chunk.text.chars().count()
        );
    }
    for paragraph in &paragraphs {
        assert_eq!(
            journal
                .iter()
                .filter(|c| c.text.contains(paragraph.as_str()))
                .count(),
            1,
            "a paragraph is never split across chunks"
        );
    }
    assert_eq!(chunks.last().unwrap().heading_path, path(&["After"]));
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn one_paragraph_over_the_budget_is_still_split() {
    let text = "This sentence repeats. ".repeat(400); // 9,200 characters
    let chunks = split_document(&text);
    assert!(chunks.len() > 1);
    for chunk in &chunks {
        assert!(chunk.text.chars().count() <= CHUNK_CHARS);
    }
    assert_tiles(&text, &chunks);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn offsets_count_characters_not_bytes() {
    // TIM-90: memories point into their source by character offsets.
    let text = "# Café ☕\n\nNaïve résumé.\n\n# Über\n\nGrüße aus Köln. 🌧️\n";
    let chunks = split_document(text);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].heading_path, path(&["Café ☕"]));
    assert_eq!(chunks[1].heading_path, path(&["Über"]));
    assert_eq!(
        chunks[1].start,
        text.find("# Über")
            .map(|b| text[..b].chars().count())
            .unwrap()
    );
    assert_tiles(text, &chunks);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_chunk_hash_covers_its_text_and_heading_path() {
    let text = "Plant the tomatoes.\n";
    let garden = chunk_hash(&path(&["Home", "Garden"]), text);
    assert_eq!(
        garden,
        chunk_hash(&path(&["Home", "Garden"]), text),
        "stable"
    );
    assert_ne!(
        garden,
        chunk_hash(&path(&["Home", "Garden"]), "Plant the beans.\n")
    );
    assert_ne!(
        garden,
        chunk_hash(&path(&["Work", "Garden"]), text),
        "moved"
    );
    assert_ne!(garden, chunk_hash(&path(&[]), text));
    // However the headings are spelled, two paths never collide.
    assert_ne!(
        chunk_hash(&path(&["Home", "Garden"]), text),
        chunk_hash(&path(&["Home > Garden"]), text)
    );
    assert_ne!(
        chunk_hash(&path(&["Home", "Garden"]), text),
        chunk_hash(&path(&["HomeGarden"]), text)
    );
    assert_ne!(
        chunk_hash(&path(&["Home"]), "Garden\nPlant"),
        chunk_hash(&path(&["Home", "Garden"]), "Plant")
    );
}

// Turns (TIM-90 sources; TIM-94 decision 5)

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_turn_is_stored_verbatim_with_its_provenance() {
    let h = Harness::new();
    let mut sent = discord_turn(
        "2026-10-01T06:59:30Z",
        "[Sam] I'm flying to Berlin on 3 October.",
        author("5678", "Sam"),
    );
    sent.assistant_text = "Safe travels, Sam!".into();
    sent.recall_id = Some("recall-1".into());
    let got = ingest(&h, "main", &sent);

    assert_eq!(got.outcome, Outcome::Stored);
    assert_eq!(got.chunks_queued, 1);
    assert_eq!(got.chunks_skipped, 0);
    assert!(got.secret_kinds.is_empty());
    let source = got.source;
    assert_eq!(h.source_column::<String>(source, "kind"), "turn");
    assert_eq!(h.source_column::<String>(source, "session_id"), "thread-1");
    let message_at = micros(at("2026-10-01T06:59:30Z"));
    assert_eq!(h.source_column::<i64>(source, "message_at"), message_at);
    assert_eq!(
        h.source_column::<i64>(source, "observed_at"),
        message_at,
        "observed_at is the message time, not when it arrived"
    );
    assert_eq!(h.source_column::<i64>(source, "ingested_at"), h.now());
    assert_eq!(h.source_column::<String>(source, "timezone"), TZ);
    assert_eq!(
        h.source_column::<String>(source, "text"),
        "[Sam] I'm flying to Berlin on 3 October.",
        "the speaker prefix stays (TIM-96)"
    );
    assert_eq!(
        h.source_column::<String>(source, "reply"),
        "Safe travels, Sam!"
    );
    assert_eq!(h.source_column::<String>(source, "platform"), "discord");
    assert_eq!(h.source_column::<String>(source, "author_name"), "Sam");
    assert_eq!(h.source_column::<i64>(source, "author_is_bot"), 0);
    assert_eq!(h.source_column::<String>(source, "recall_id"), "recall-1");
    assert_eq!(
        h.source_column::<Option<i64>>(source, "tombstoned_at"),
        None
    );
    assert_eq!(
        h.source_column::<Option<String>>(source, "document_id"),
        None
    );
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_turn_without_a_timezone_takes_the_banks() {
    let h = Harness::new();
    h.service
        .ensure_bank(
            "berlin",
            &BankIdentity {
                timezone: Some("Europe/Berlin".into()),
                ..identity()
            },
            &models(),
        )
        .unwrap();
    let got = ingest(
        &h,
        "berlin",
        &Turn {
            timezone: None,
            ..turn("s", "2026-10-01T06:00:00Z", "Hello.", "Hi.")
        },
    );
    assert_eq!(
        h.source_column::<String>(got.source, "timezone"),
        "Europe/Berlin"
    );
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn an_unknown_timezone_or_bank_is_refused_and_nothing_is_stored() {
    let h = Harness::new();
    let bad = Turn {
        timezone: Some("Mars/Olympus_Mons".into()),
        ..turn("s", "2026-10-01T06:00:00Z", "Hello.", "Hi.")
    };
    assert!(matches!(
        ingest_turn(&h.service, "main", &bad),
        Err(IngestError::InvalidTimezone)
    ));
    let bad_doc = Document {
        timezone: Some("Mars/Olympus_Mons".into()),
        ..document("notes.md", "Hello.", date(2026, 9, 15))
    };
    assert!(matches!(
        ingest_document(&h.service, "main", &bad_doc),
        Err(IngestError::InvalidTimezone)
    ));
    assert!(matches!(
        ingest_turn(
            &h.service,
            "nowhere",
            &turn("s", "2026-10-01T06:00:00Z", "Hello.", "Hi.")
        ),
        Err(IngestError::UnknownBank)
    ));
    assert_eq!(h.count("SELECT COUNT(*) FROM sources"), 0);
    assert_eq!(h.count("SELECT COUNT(*) FROM chunks"), 0);
    assert_eq!(h.count("SELECT COUNT(*) FROM extraction_queue"), 0);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_turn_is_one_queued_chunk() {
    let h = Harness::new();
    let got = ingest(
        &h,
        "main",
        &turn(
            "s",
            "2026-10-01T06:00:00Z",
            "My sister Mia arrives on Friday.",
            "I'll remember that Mia arrives on Friday.",
        ),
    );
    let chunks = h.chunks_of(got.source);
    assert_eq!(chunks.len(), 1, "a turn is one chunk");
    let (chunk, position, heading_path, _, _, text, _) = &chunks[0];
    assert_eq!(*position, 0);
    assert_eq!(*heading_path, None);
    let text = text.as_deref().expect("the chunk holds its text");
    assert!(text.contains("My sister Mia arrives on Friday."));
    assert!(text.contains("I'll remember that Mia arrives on Friday."));
    assert_eq!(h.chunk_column::<Option<i64>>(*chunk, "extracted_at"), None);
    assert_eq!(h.chunk_column::<i64>(*chunk, "error_count"), 0);
    assert_eq!(h.chunk_column::<Option<i64>>(*chunk, "failed_at"), None);
    assert_eq!(depth(&h, "main"), 1);
    assert_eq!(depth(&h, "other"), 0);
}

// Idempotency (TIM-90, ADR 0002)

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn the_same_turn_twice_does_nothing_the_second_time() {
    // Hermes retries, and the plugin's spool replays (TIM-94, decision 6).
    let h = Harness::new();
    let sent = turn(
        "s",
        "2026-10-01T06:00:00Z",
        "I moved to Wellington.",
        "Noted.",
    );
    let first = ingest(&h, "main", &sent);
    h.clock.advance(SignedDuration::from_mins(5));
    let second = ingest(&h, "main", &sent);

    assert_eq!(second.outcome, Outcome::Duplicate);
    assert_eq!(second.source, first.source, "the existing source");
    assert_eq!(second.chunks_queued, 0);
    assert_eq!(second.speaker, None);
    assert_eq!(h.count("SELECT COUNT(*) FROM sources"), 1);
    assert_eq!(h.count("SELECT COUNT(*) FROM chunks"), 1);
    assert_eq!(depth(&h, "main"), 1);
    assert_eq!(
        h.source_column::<i64>(first.source, "ingested_at"),
        micros(start()),
        "a duplicate doesn't touch the stored source"
    );
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_duplicate_is_caught_after_a_restart() {
    let h = Harness::new();
    let sent = turn(
        "s",
        "2026-10-01T06:00:00Z",
        "I moved to Wellington.",
        "Noted.",
    );
    let first = ingest(&h, "main", &sent);
    let h = h.restart();
    let second = ingest(&h, "main", &sent);
    assert_eq!(second.outcome, Outcome::Duplicate);
    assert_eq!(second.source, first.source);
    assert_eq!(depth(&h, "main"), 1);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_duplicate_does_not_requeue_an_extracted_chunk() {
    let h = Harness::new();
    let sent = turn(
        "s",
        "2026-10-01T06:00:00Z",
        "I moved to Wellington.",
        "Noted.",
    );
    ingest(&h, "main", &sent);
    let lease = claim(&h, "main").unwrap();
    complete_chunk(&h.service, lease).unwrap();
    let again = ingest(&h, "main", &sent);
    assert_eq!(again.outcome, Outcome::Duplicate);
    assert_eq!(depth(&h, "main"), 0);
    assert!(claim(&h, "main").is_none());
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_swept_source_stays_a_tombstone_when_it_is_sent_again() {
    // ADR 0008: the sweep deletes the text and keeps the key, so re-ingest
    // can't bring the passage back.
    let h = Harness::new();
    let sent = turn(
        "s",
        "2026-10-01T06:00:00Z",
        "I moved to Wellington.",
        "Noted.",
    );
    let first = ingest(&h, "main", &sent);
    h.execute("DELETE FROM extraction_queue", []);
    h.execute(
        "UPDATE sources SET text = NULL, reply = NULL, tombstoned_at = ?1,
                            tombstone_reason = 'swept' WHERE uuid = ?2",
        (h.now(), first.source.to_string()),
    );
    h.execute(
        "UPDATE chunks SET text = NULL, tombstoned_at = ?1",
        [h.now()],
    );
    let again = ingest(&h, "main", &sent);
    assert_eq!(again.outcome, Outcome::Duplicate);
    assert_eq!(
        h.source_column::<Option<String>>(first.source, "text"),
        None
    );
    assert_eq!(depth(&h, "main"), 0);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn the_same_words_in_another_session_at_another_time_or_in_another_bank_are_new() {
    let h = Harness::new();
    let words = ("Good morning.", "Morning!");
    let a = ingest(
        &h,
        "main",
        &turn("s1", "2026-10-01T06:00:00Z", words.0, words.1),
    );
    let b = ingest(
        &h,
        "main",
        &turn("s2", "2026-10-01T06:00:00Z", words.0, words.1),
    );
    let c = ingest(
        &h,
        "main",
        &turn("s1", "2026-10-02T06:00:00Z", words.0, words.1),
    );
    let d = ingest(
        &h,
        "other",
        &turn("s1", "2026-10-01T06:00:00Z", words.0, words.1),
    );
    for got in [&a, &b, &c, &d] {
        assert_eq!(got.outcome, Outcome::Stored);
    }
    let sources: BTreeSet<Uuid> = [a.source, b.source, c.source, d.source].into();
    assert_eq!(sources.len(), 4);
    assert_eq!(depth(&h, "main"), 3);
    assert_eq!(depth(&h, "other"), 1);
}

// Documents (TIM-92; TIM-94 decision 10)

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_document_is_stored_with_its_reference_date() {
    let h = Harness::new();
    let text = "# Trip\n\nFlying to Berlin tomorrow.\n";
    let got = ingest_doc(&h, "main", &document("trip.md", text, date(2026, 9, 15)));

    assert_eq!(got.outcome, Outcome::Stored);
    assert_eq!(got.speaker, None, "documents have no speaker");
    let source = got.source;
    assert_eq!(h.source_column::<String>(source, "kind"), "document");
    assert_eq!(h.source_column::<String>(source, "document_id"), "trip.md");
    assert_eq!(h.source_column::<String>(source, "text"), text);
    assert_eq!(h.source_column::<Option<String>>(source, "reply"), None);
    assert_eq!(
        h.source_column::<Option<String>>(source, "session_id"),
        None
    );
    assert_eq!(
        h.source_column::<String>(source, "reference_date"),
        "2026-09-15"
    );
    assert_eq!(h.source_column::<i64>(source, "reference_date_exact"), 1);
    assert_eq!(h.source_column::<String>(source, "timezone"), TZ);
    assert_eq!(h.source_column::<i64>(source, "ingested_at"), h.now());
    // TIM-90: a time is stored as the start of its unit in the source's
    // timezone, so the reference date is midnight in Auckland.
    let midnight = date(2026, 9, 15)
        .to_zoned(TimeZone::get(TZ).unwrap())
        .unwrap()
        .timestamp();
    assert_eq!(
        h.source_column::<i64>(source, "observed_at"),
        micros(midnight)
    );
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn an_inexact_reference_date_is_recorded() {
    let h = Harness::new();
    let got = ingest_doc(
        &h,
        "main",
        &Document {
            reference_date_exact: false,
            ..document("old.md", "Sometime last year we moved.", date(2025, 6, 1))
        },
    );
    assert_eq!(
        h.source_column::<i64>(got.source, "reference_date_exact"),
        0
    );
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_document_is_stored_as_its_chunks_and_every_chunk_is_queued() {
    let h = Harness::new();
    let text = "Preamble.\n\n# Work\n\nShip v1.\n\n## Ingest\n\nChunk documents.\n\n# Home\n\nFix the fence.\n";
    let got = ingest_doc(&h, "main", &document("notes.md", text, date(2026, 9, 15)));
    let expected = split_document(text);
    let stored = h.chunks_of(got.source);

    assert_eq!(got.chunks_queued, expected.len());
    assert_eq!(stored.len(), expected.len());
    assert_eq!(depth(&h, "main"), expected.len());
    for (index, (want, row)) in expected.iter().zip(&stored).enumerate() {
        let (_, position, _, start, end, chunk_text, hash) = row;
        assert_eq!(*position, index as i64);
        assert_eq!(*start, want.start as i64, "chunk {index} records its range");
        assert_eq!(*end, want.end as i64);
        assert_eq!(chunk_text.as_deref(), Some(want.text.as_str()));
        assert_eq!(*hash, chunk_hash(&want.heading_path, &want.text));
    }
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn the_same_document_twice_does_nothing_the_second_time() {
    let h = Harness::new();
    let text = "# Plans\n\nFly to Berlin on 3 October.\n";
    let first = ingest_doc(&h, "main", &document("plans.md", text, date(2026, 9, 15)));
    // The key is the document id and content hash, so a new reference date
    // alone doesn't make a new version.
    let second = ingest_doc(&h, "main", &document("plans.md", text, date(2026, 9, 20)));
    assert_eq!(second.outcome, Outcome::Duplicate);
    assert_eq!(second.source, first.source);
    assert_eq!(second.chunks_queued, 0);
    assert_eq!(h.count("SELECT COUNT(*) FROM sources"), 1);
    assert_eq!(depth(&h, "main"), 1);
}

const PLANS: &str = "# Plans\n\nFly to Berlin on 3 October.\n\n";
const WORK_V1: &str = "# Work\n\nShip Asphodel v1.\n\n";
const WORK_V2: &str = "# Work\n\nShip Asphodel v1 by Friday.\n\n";
const HOME: &str = "# Home\n\nFix the fence.\n";

/// The hashes of the chunks queued for `source`, in queue order.
fn queued_hashes(h: &Harness, source: Uuid) -> Vec<String> {
    let store = h.service.store().unwrap();
    let conn = store.connection();
    let mut statement = conn
        .prepare(
            "SELECT c.content_hash FROM extraction_queue q
             JOIN chunks c ON c.id = q.chunk_id JOIN sources s ON s.id = c.source_id
             WHERE s.uuid = ?1 ORDER BY c.position",
        )
        .unwrap();
    statement
        .query_map([source.to_string()], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

fn hashes_of(text: &str, sections: &[&str]) -> Vec<String> {
    split_document(text)
        .iter()
        .filter(|chunk| {
            sections
                .iter()
                .any(|s| chunk.text.starts_with(s.trim_end()))
        })
        .map(|chunk| chunk_hash(&chunk.heading_path, &chunk.text))
        .collect()
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn an_edited_document_queues_only_chunks_no_earlier_version_had() {
    // TIM-92, edited documents: chunks are matched by hash, and a chunk seen
    // in any earlier version is skipped.
    let h = Harness::new();
    let v1 = [PLANS, WORK_V1].concat();
    let v2 = [PLANS, WORK_V2, HOME].concat();
    let v3 = [PLANS, WORK_V1, HOME].concat();

    let first = ingest_doc(&h, "main", &document("life.md", &v1, date(2026, 9, 1)));
    assert_eq!(first.chunks_queued, 2);

    let second = ingest_doc(&h, "main", &document("life.md", &v2, date(2026, 9, 8)));
    assert_eq!(
        second.outcome,
        Outcome::Stored,
        "a new version is a new source"
    );
    assert_ne!(second.source, first.source);
    assert_eq!(second.chunks_queued, 2, "the edited Work and the new Home");
    assert_eq!(second.chunks_skipped, 1, "Plans is unchanged");
    assert_eq!(
        queued_hashes(&h, second.source),
        hashes_of(&v2, &[WORK_V2, HOME])
    );
    // New chunks are extracted at the new version's reference date.
    let midnight = date(2026, 9, 8)
        .to_zoned(TimeZone::get(TZ).unwrap())
        .unwrap()
        .timestamp();
    assert_eq!(
        h.source_column::<i64>(second.source, "observed_at"),
        micros(midnight)
    );

    // Every section of v3 was in v1 or v2, if not in both.
    let third = ingest_doc(&h, "main", &document("life.md", &v3, date(2026, 9, 15)));
    assert_eq!(third.outcome, Outcome::Stored);
    assert_eq!(third.chunks_queued, 0);
    assert_eq!(third.chunks_skipped, 3);
    assert_eq!(depth(&h, "main"), 4);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_section_moved_under_another_heading_is_new() {
    let h = Harness::new();
    let v1 = "# Work\n\n## Errands\n\nFix the fence.\n";
    let v2 = "# Home\n\n## Errands\n\nFix the fence.\n";
    ingest_doc(&h, "main", &document("todo.md", v1, date(2026, 9, 1)));
    let second = ingest_doc(&h, "main", &document("todo.md", v2, date(2026, 9, 2)));
    let errands = split_document(v2)
        .into_iter()
        .find(|chunk| chunk.heading_path == path(&["Home", "Errands"]))
        .unwrap();
    assert!(
        queued_hashes(&h, second.source)
            .contains(&chunk_hash(&errands.heading_path, &errands.text)),
        "the heading path is part of a chunk's identity"
    );
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn an_earlier_reference_date_is_accepted() {
    // TIM-92: reconciliation's direction rule makes older claims harmless,
    // and a rejection would be a failure nobody sees.
    let h = Harness::new();
    ingest_doc(&h, "main", &document("life.md", PLANS, date(2026, 9, 15)));
    let older = ingest_doc(
        &h,
        "main",
        &document("life.md", &[PLANS, HOME].concat(), date(2026, 8, 1)),
    );
    assert_eq!(older.outcome, Outcome::Stored);
    assert_eq!(older.chunks_queued, 1);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn memories_from_a_removed_section_are_left_alone() {
    // TIM-92: deleting a line from a note doesn't make it false.
    let h = Harness::new();
    let v1 = [PLANS, WORK_V1].concat();
    let first = ingest_doc(&h, "main", &document("life.md", &v1, date(2026, 9, 1)));
    let (work, _, _, _, _, work_text, _) = h.chunks_of(first.source)[1].clone();
    let (bank, chunk): (i64, i64) = {
        let store = h.service.store().unwrap();
        store
            .connection()
            .query_row(
                "SELECT bank_id, id FROM chunks WHERE uuid = ?1",
                [work.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap()
    };
    h.execute(
        "INSERT INTO memories (uuid, bank_id, content, kind, significance, chunk_id, source_start,
                               source_end, observed_at, window_confidence, created_at, updated_at)
         VALUES ('m', ?1, 'Tim is shipping Asphodel v1.', 'state', 'minor', ?2, 0, 10, ?3, 'high',
                 ?3, ?3)",
        (bank, chunk, h.now()),
    );

    ingest_doc(&h, "main", &document("life.md", PLANS, date(2026, 9, 8)));
    assert_eq!(h.count("SELECT COUNT(*) FROM memories"), 1);
    assert_eq!(
        h.chunk_column::<Option<String>>(work, "text"),
        work_text,
        "the passage the memory rests on stays"
    );
    assert!(work_text.is_some());
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_tombstoned_chunk_is_not_extracted_again_from_a_new_version() {
    // ADR 0002 and TIM-92: the chunk hash is the forget tombstone. Once the
    // erase or the sweep has removed a chunk's text, a new version that
    // still holds the passage must not bring it back.
    let h = Harness::new();
    let v1 = [PLANS, WORK_V1].concat();
    let first = ingest_doc(&h, "main", &document("life.md", &v1, date(2026, 9, 1)));
    h.execute("DELETE FROM extraction_queue", []);
    h.execute(
        "UPDATE chunks SET text = NULL, tombstoned_at = ?1, extracted_at = ?1",
        [h.now()],
    );
    h.execute(
        "UPDATE sources SET text = NULL, tombstoned_at = ?1, tombstone_reason = 'swept'
         WHERE uuid = ?2",
        (h.now(), first.source.to_string()),
    );

    let v2 = [PLANS, WORK_V1, HOME].concat();
    let second = ingest_doc(&h, "main", &document("life.md", &v2, date(2026, 9, 8)));
    assert_eq!(second.chunks_queued, 1, "only Home is new");
    assert_eq!(queued_hashes(&h, second.source), hashes_of(&v2, &[HOME]));
}

// Secrets at ingest (ADR 0002; TIM-92, other decision 1)

/// The kinds recorded on a source, from its `secret_kinds` column.
fn recorded_kinds(h: &Harness, source: Uuid) -> BTreeSet<String> {
    match h.source_column::<Option<String>>(source, "secret_kinds") {
        None => BTreeSet::new(),
        Some(json) => serde_json::from_str::<Vec<String>>(&json)
            .expect("secret_kinds is a JSON array of kind names")
            .into_iter()
            .collect(),
    }
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn secrets_in_a_turn_are_redacted_before_anything_is_stored() {
    let h = Harness::new();
    let (user, github) = sentence_with(SecretKind::GithubToken, 7);
    let (reply, aws) = sentence_with(SecretKind::AwsAccessKey, 8);
    let got = ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:00:00Z", &user, &reply),
    );

    assert_eq!(
        got.secret_kinds,
        BTreeSet::from([SecretKind::GithubToken, SecretKind::AwsAccessKey])
    );
    assert_eq!(
        recorded_kinds(&h, got.source),
        BTreeSet::from(["github_token".to_string(), "aws_access_key".to_string()])
    );
    let text: String = h.source_column(got.source, "text");
    let stored_reply: String = h.source_column(got.source, "reply");
    assert_eq!(text, scan(&user).text);
    assert_eq!(stored_reply, scan(&reply).text);
    let chunk_text = h.chunks_of(got.source)[0].5.clone().unwrap();
    assert!(chunk_text.contains(&SecretKind::GithubToken.marker()));
    for secret in [&github, &aws] {
        assert!(!h.store_holds(secret), "nothing stored holds the secret");
    }
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn secrets_in_a_document_are_redacted_and_the_chunks_hash_the_redacted_text() {
    let h = Harness::new();
    let (line, password) = sentence_with(SecretKind::UrlPassword, 9);
    let text = format!("# Infra\n\n{line}\n\n# Other\n\nNothing secret.\n");
    let got = ingest_doc(&h, "main", &document("infra.md", &text, date(2026, 9, 15)));

    assert_eq!(got.secret_kinds, BTreeSet::from([SecretKind::UrlPassword]));
    assert_eq!(
        recorded_kinds(&h, got.source),
        BTreeSet::from(["url_password".to_string()])
    );
    let redacted = scan(&text).text;
    assert_eq!(h.source_column::<String>(got.source, "text"), redacted);
    let expected = split_document(&redacted);
    let stored = h.chunks_of(got.source);
    assert_eq!(stored.len(), expected.len());
    for (want, row) in expected.iter().zip(&stored) {
        assert_eq!(row.5.as_deref(), Some(want.text.as_str()));
        assert_eq!(row.6, chunk_hash(&want.heading_path, &want.text));
    }
    assert!(!h.store_holds(&password));
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_clean_turn_records_no_secret_kinds() {
    let h = Harness::new();
    let got = ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:00:00Z", "Hello.", "Hi."),
    );
    assert!(got.secret_kinds.is_empty());
    assert!(recorded_kinds(&h, got.source).is_empty());
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn the_key_is_computed_from_the_redacted_text() {
    // Nothing stored is derived from a secret, the content hash included,
    // so a resend whose only difference is the secret itself is the same
    // turn.
    let h = Harness::new();
    let (first_text, _) = sentence_with(SecretKind::OpenAiKey, 10);
    let (second_text, _) = sentence_with(SecretKind::OpenAiKey, 11);
    assert_ne!(first_text, second_text);
    let first = ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:00:00Z", &first_text, "Ok."),
    );
    let second = ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:00:00Z", &second_text, "Ok."),
    );
    assert_eq!(second.outcome, Outcome::Duplicate);
    assert_eq!(second.source, first.source);
}

// The turn that asks to forget (TIM-99 decision 2, ADR 0010)

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_forget_request_is_stored_only_as_a_tombstone_and_never_queued() {
    let h = Harness::new();
    let sent = Turn {
        forget_requested: true,
        recall_id: Some("recall-forget".into()),
        ..turn(
            "s",
            "2026-10-01T06:00:00Z",
            "Forget that my passport number is LZ8841207.",
            "Done, I've forgotten it.",
        )
    };
    let got = ingest(&h, "main", &sent);

    assert_eq!(got.outcome, Outcome::Tombstone);
    assert_eq!(got.chunks_queued, 0);
    assert_eq!(got.speaker, None);
    assert_eq!(h.source_column::<Option<String>>(got.source, "text"), None);
    assert_eq!(h.source_column::<Option<String>>(got.source, "reply"), None);
    assert_eq!(
        h.source_column::<Option<i64>>(got.source, "tombstoned_at"),
        Some(h.now())
    );
    assert_eq!(
        h.source_column::<Option<String>>(got.source, "tombstone_reason")
            .as_deref(),
        Some("forget_requested")
    );
    assert!(h.chunks_of(got.source).is_empty(), "nothing to extract");
    assert_eq!(depth(&h, "main"), 0);
    assert!(claim(&h, "main").is_none());
    assert!(!h.store_holds("LZ8841207"));
    assert!(!h.store_holds("Done, I've forgotten it."));
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_forget_request_deletes_its_recall_row() {
    // ADR 0010: the recall row for the turn's `recall_id` holds the
    // request as its query, so it goes too. Other recalls stay.
    let h = Harness::new();
    let bank: i64 = h.one("SELECT id FROM banks WHERE name = 'main'", []);
    for (uuid, query) in [
        ("recall-forget", "Forget my passport number"),
        ("recall-other", "Berlin"),
    ] {
        h.execute(
            "INSERT INTO recalls (uuid, bank_id, kind, session_id, turn, query, latency_ms, at)
             VALUES (?1, ?2, 'prefetch', 's', 0, ?3, 12, ?4)",
            (uuid, bank, query, h.now()),
        );
    }
    let sent = Turn {
        forget_requested: true,
        recall_id: Some("recall-forget".into()),
        ..turn(
            "s",
            "2026-10-01T06:00:00Z",
            "Forget my passport number.",
            "Done.",
        )
    };
    ingest(&h, "main", &sent);
    assert_eq!(
        h.count("SELECT COUNT(*) FROM recalls WHERE uuid = 'recall-forget'"),
        0
    );
    assert_eq!(
        h.count("SELECT COUNT(*) FROM recalls WHERE uuid = 'recall-other'"),
        1
    );
    assert!(!h.store_holds("Forget my passport number"));
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_forget_request_sent_again_stays_a_tombstone() {
    // The spool may replay the turn, and a resend could lose the flag.
    let h = Harness::new();
    let plain = turn("s", "2026-10-01T06:00:00Z", "Forget where I live.", "Done.");
    let flagged = Turn {
        forget_requested: true,
        ..plain.clone()
    };
    let first = ingest(&h, "main", &flagged);
    for resend in [&flagged, &plain] {
        let again = ingest(&h, "main", resend);
        assert_eq!(again.outcome, Outcome::Duplicate);
        assert_eq!(again.source, first.source);
    }
    assert_eq!(
        h.source_column::<Option<String>>(first.source, "text"),
        None
    );
    assert_eq!(depth(&h, "main"), 0);
    assert!(!h.store_holds("Forget where I live."));
}

// Speakers (TIM-94, decision 1)

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn the_owners_platform_id_resolves_to_user() {
    let h = Harness::new();
    let got = ingest(
        &h,
        "main",
        &discord_turn(
            "2026-10-01T06:00:00Z",
            "I'm off to Berlin.",
            author("1234", "tim"),
        ),
    );
    let speaker = got.speaker.unwrap();
    assert!(speaker.owner);
    assert_eq!(speaker.entity, h.seeded("main", "user"));
    assert_eq!(h.entities_in("main"), 2, "no new entity for the owner");
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_turn_with_no_author_is_the_owners() {
    // The CLI, TUI and Hermes UI send no author.
    let h = Harness::new();
    let got = ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:00:00Z", "Hello.", "Hi."),
    );
    let speaker = got.speaker.unwrap();
    assert!(speaker.owner);
    assert_eq!(speaker.entity, h.seeded("main", "user"));
    assert_eq!(h.entities_in("main"), 2);
}

/// The aliases of the entity `entity`.
fn aliases(h: &Harness, entity: Uuid) -> BTreeSet<String> {
    let store = h.service.store().unwrap();
    let conn = store.connection();
    let mut statement = conn
        .prepare(
            "SELECT a.alias FROM entity_aliases a JOIN entities e ON e.id = a.entity_id
             WHERE e.uuid = ?1",
        )
        .unwrap();
    statement
        .query_map([entity.to_string()], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn anyone_else_becomes_a_person_of_their_own() {
    let h = Harness::new();
    let got = ingest(
        &h,
        "main",
        &discord_turn(
            "2026-10-01T06:00:00Z",
            "[Sam] I'm vegetarian.",
            author("5678", "Sam"),
        ),
    );
    let speaker = got.speaker.unwrap();
    assert!(!speaker.owner);
    assert_ne!(speaker.entity, h.seeded("main", "user"));
    assert_ne!(speaker.entity, h.seeded("main", "assistant"));
    assert_eq!(h.entities_in("main"), 3);
    let entity = speaker.entity.to_string();
    assert_eq!(
        h.one::<String, _>("SELECT name FROM entities WHERE uuid = ?1", [&entity]),
        "Sam"
    );
    assert_eq!(
        h.one::<String, _>("SELECT kind FROM entities WHERE uuid = ?1", [&entity]),
        "person"
    );
    assert!(aliases(&h, speaker.entity).contains("discord:5678"));
    // TIM-92: a new alias is a logged edit, so a mislink can be undone.
    assert!(
        h.one::<i64, _>(
            "SELECT COUNT(*) FROM edits e JOIN entities n ON n.id = e.entity_id
             WHERE n.uuid = ?1 AND e.kind = 'alias_added'",
            [&entity],
        ) >= 1
    );
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_speaker_is_found_again_by_platform_id_even_after_a_rename() {
    let h = Harness::new();
    let first = ingest(
        &h,
        "main",
        &discord_turn("2026-10-01T06:00:00Z", "Hi.", author("5678", "Sam")),
    );
    let second = ingest(
        &h,
        "main",
        &discord_turn(
            "2026-10-01T06:01:00Z",
            "Hi again.",
            author("5678", "Samantha"),
        ),
    );
    assert_eq!(
        second.speaker.unwrap().entity,
        first.speaker.unwrap().entity
    );
    assert_eq!(h.entities_in("main"), 3);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn sharing_the_owners_name_does_not_make_someone_the_owner() {
    // Only the owner can forget, keep or unkeep, so ownership comes from
    // platform ids alone, never from a display name.
    let h = Harness::new();
    let got = ingest(
        &h,
        "main",
        &discord_turn(
            "2026-10-01T06:00:00Z",
            "I'm Tim too.",
            author("9999", "Tim"),
        ),
    );
    let speaker = got.speaker.unwrap();
    assert!(!speaker.owner);
    assert_ne!(speaker.entity, h.seeded("main", "user"));
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn the_owners_id_on_another_platform_is_someone_else() {
    // The owner's platform id is `discord:1234`; `1234` on Telegram is a
    // different account.
    let h = Harness::new();
    let got = ingest(
        &h,
        "main",
        &Turn {
            platform: Some("telegram".into()),
            ..discord_turn("2026-10-01T06:00:00Z", "Hello.", author("1234", "Tim"))
        },
    );
    let speaker = got.speaker.unwrap();
    assert!(!speaker.owner);
    assert!(aliases(&h, speaker.entity).contains("telegram:1234"));
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn two_people_with_one_name_are_two_entities() {
    // TIM-92: no string-similarity matching and no automatic merges.
    let h = Harness::new();
    let a = ingest(
        &h,
        "main",
        &discord_turn("2026-10-01T06:00:00Z", "Hi.", author("1111", "Sam")),
    );
    let b = ingest(
        &h,
        "main",
        &discord_turn("2026-10-01T06:01:00Z", "Hi.", author("2222", "Sam")),
    );
    assert_ne!(a.speaker.unwrap().entity, b.speaker.unwrap().entity);
    assert_eq!(h.entities_in("main"), 4);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_bot_is_a_speaker_of_its_own() {
    let h = Harness::new();
    let got = ingest(
        &h,
        "main",
        &discord_turn(
            "2026-10-01T06:00:00Z",
            "Reminder: standup at 10.",
            TurnAuthor {
                id: "4242".into(),
                name: Some("Dyno".into()),
                is_bot: true,
            },
        ),
    );
    let speaker = got.speaker.unwrap();
    assert!(!speaker.owner);
    assert_ne!(speaker.entity, h.seeded("main", "user"));
    assert_eq!(h.source_column::<i64>(got.source, "author_is_bot"), 1);
    assert_eq!(got.chunks_queued, 1, "bots are ingested like anyone else");
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn speakers_never_cross_banks() {
    let h = Harness::new();
    let a = ingest(
        &h,
        "main",
        &discord_turn("2026-10-01T06:00:00Z", "Hi.", author("5678", "Sam")),
    );
    let b = ingest(
        &h,
        "other",
        &discord_turn("2026-10-01T06:00:00Z", "Hi.", author("5678", "Sam")),
    );
    assert_ne!(a.speaker.unwrap().entity, b.speaker.unwrap().entity);
    assert_eq!(h.entities_in("main"), 3);
    assert_eq!(h.entities_in("other"), 3);
}

// The extraction queue (TIM-92; TIM-94 decision 3)

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn an_empty_queue_hands_out_nothing() {
    let h = Harness::new();
    assert!(claim(&h, "main").is_none());
    assert_eq!(depth(&h, "main"), 0);
    assert!(failed_chunks(&h.service, "main").unwrap().is_empty());
    assert!(claim_chunk(&h.service, "nowhere").is_err());
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn each_bank_has_one_worker() {
    let h = Harness::new();
    ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:00:00Z", "One.", "Ok."),
    );
    ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:01:00Z", "Two.", "Ok."),
    );
    ingest(
        &h,
        "other",
        &turn("s", "2026-10-01T06:00:00Z", "Three.", "Ok."),
    );

    let main = claim(&h, "main").expect("main's head");
    assert!(
        claim(&h, "main").is_none(),
        "main's worker holds a lease, so nothing else in main is handed out"
    );
    let other = claim(&h, "other").expect("another bank isn't held up");
    assert_eq!(depth(&h, "main"), 2, "a leased chunk is still in the queue");

    complete_chunk(&h.service, main).unwrap();
    let next = claim(&h, "main").expect("the lease is released on completion");
    assert!(
        h.chunk_column::<String>(next.chunk, "text")
            .contains("Two.")
    );
    complete_chunk(&h.service, other).unwrap();
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn turns_go_ahead_of_documents_then_observed_at_order() {
    // TIM-92: serialised in observed_at order, with turns ahead of document
    // chunks.
    let h = Harness::new();
    let doc = ingest_doc(
        &h,
        "main",
        &document(
            "notes.md",
            "# A\n\nFirst.\n\n# B\n\nSecond.\n",
            date(2026, 9, 1),
        ),
    );
    let late = ingest(
        &h,
        "main",
        &turn("s", "2026-09-30T06:00:00Z", "Late.", "Ok."),
    );
    // Arrives last, from the spool, but was said first.
    let early = ingest(
        &h,
        "main",
        &turn("s", "2026-09-20T06:00:00Z", "Early.", "Ok."),
    );

    let mut order = Vec::new();
    while let Some(lease) = claim(&h, "main") {
        order.push((lease.source, lease.source_kind, lease.position));
        complete_chunk(&h.service, lease).unwrap();
    }
    assert_eq!(
        order,
        [
            (early.source, SourceKind::Turn, 0),
            (late.source, SourceKind::Turn, 0),
            (doc.source, SourceKind::Document, 0),
            (doc.source, SourceKind::Document, 1),
        ]
    );
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_lease_describes_its_chunk() {
    let h = Harness::new();
    let got = ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:00:00Z", "Hello.", "Hi."),
    );
    let lease = claim(&h, "main").unwrap();
    assert_eq!(lease.source, got.source);
    assert_eq!(lease.chunk, h.chunks_of(got.source)[0].0);
    assert_eq!(lease.source_kind, SourceKind::Turn);
    assert_eq!(lease.position, 0);
    assert_eq!(lease.observed_at, at("2026-10-01T06:00:00Z"));
    assert_eq!(lease.error_count, 0);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn completing_a_chunk_marks_it_extracted_and_takes_it_off_the_queue() {
    let h = Harness::new();
    ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:00:00Z", "Hello.", "Hi."),
    );
    let lease = claim(&h, "main").unwrap();
    let chunk = lease.chunk;
    // Call 1's output is saved on the chunk so a failed call 2 can resume
    // (TIM-92), and dropped when the chunk commits (ADR 0008).
    h.execute(
        "UPDATE chunks SET call1_output = '{\"claims\":[]}' WHERE uuid = ?1",
        [chunk.to_string()],
    );
    h.clock.advance(SignedDuration::from_secs(3));
    complete_chunk(&h.service, lease).unwrap();

    assert_eq!(
        h.chunk_column::<Option<i64>>(chunk, "extracted_at"),
        Some(h.now())
    );
    assert_eq!(
        h.chunk_column::<Option<String>>(chunk, "call1_output"),
        None
    );
    assert_eq!(depth(&h, "main"), 0);
    assert!(claim(&h, "main").is_none());
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_failed_attempt_is_counted_and_retried_in_place() {
    let h = Harness::new();
    ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:00:00Z", "First.", "Ok."),
    );
    ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:01:00Z", "Second.", "Ok."),
    );
    let lease = claim(&h, "main").unwrap();
    let head = lease.chunk;

    assert_eq!(
        fail_chunk(&h.service, lease, LLM_502).unwrap(),
        Failure::Retry { error_count: 1 }
    );
    assert_eq!(h.chunk_column::<i64>(head, "error_count"), 1);
    assert_eq!(
        h.chunk_column::<String>(head, "last_error_kind"),
        "llm_status"
    );
    assert_eq!(
        h.chunk_column::<Option<i64>>(head, "last_error_status"),
        Some(502)
    );
    assert_eq!(h.chunk_column::<Option<i64>>(head, "failed_at"), None);
    assert_eq!(depth(&h, "main"), 2);

    // Order holds: the second chunk never overtakes a head that's being
    // retried. The implementation may back off, so look a day later too.
    if let Some(early) = claim(&h, "main") {
        assert_eq!(early.chunk, head, "the head is retried first");
        assert_eq!(early.error_count, 1);
        return;
    }
    h.clock.advance(SignedDuration::from_hours(24));
    let retried = claim(&h, "main").expect("retried within a day");
    assert_eq!(retried.chunk, head);
    assert_eq!(retried.error_count, 1);
}

/// Fails the head of `bank` until it's marked failed, advancing the clock a
/// day between attempts in case the implementation backs off. Returns the
/// chunk.
fn fail_until_failed(h: &Harness, bank: &str) -> Uuid {
    let first = claim(h, bank).expect("a head to fail");
    let chunk = first.chunk;
    let mut lease = first;
    for attempt in 1..=CHUNK_RETRY_CAP {
        assert_eq!(lease.chunk, chunk);
        let result = fail_chunk(&h.service, lease, LLM_502).unwrap();
        if attempt < CHUNK_RETRY_CAP {
            assert_eq!(
                result,
                Failure::Retry {
                    error_count: attempt
                }
            );
            h.clock.advance(SignedDuration::from_hours(24));
            lease = claim(h, bank).expect("retried");
        } else {
            assert_eq!(result, Failure::Failed);
            return chunk;
        }
    }
    unreachable!("the cap is at least 1")
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn reaching_the_retry_cap_marks_the_chunk_failed_and_the_queue_moves_on() {
    let h = Harness::new();
    let first = ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:00:00Z", "First.", "Ok."),
    );
    let second = ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:01:00Z", "Second.", "Ok."),
    );
    let failed = fail_until_failed(&h, "main");

    assert_eq!(failed, h.chunks_of(first.source)[0].0);
    assert_eq!(
        h.chunk_column::<Option<i64>>(failed, "failed_at"),
        Some(h.now())
    );
    assert_eq!(
        h.chunk_column::<i64>(failed, "error_count"),
        i64::from(CHUNK_RETRY_CAP)
    );
    assert_eq!(h.chunk_column::<Option<i64>>(failed, "extracted_at"), None);
    assert_eq!(depth(&h, "main"), 1, "a failed chunk isn't waiting");

    let listed = failed_chunks(&h.service, "main").unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].chunk, failed);
    assert_eq!(listed[0].source, first.source);
    assert_eq!(listed[0].error_count, CHUNK_RETRY_CAP);
    assert_eq!(listed[0].error_kind, "llm_status");
    assert_eq!(listed[0].status, Some(502));
    assert_eq!(listed[0].failed_at, h.clock.now());

    let next = claim(&h, "main").expect("the queue moves on");
    assert_eq!(next.source, second.source);
    assert!(
        h.chunk_column::<Option<String>>(failed, "text").is_some(),
        "a failed chunk keeps its text so it can be retried or discarded"
    );
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_failure_in_one_bank_does_not_hold_up_another() {
    let h = Harness::new();
    ingest(
        &h,
        "main",
        &turn("s", "2026-10-01T06:00:00Z", "Main.", "Ok."),
    );
    ingest(
        &h,
        "other",
        &turn("s", "2026-10-01T06:00:00Z", "Other.", "Ok."),
    );
    let main = claim(&h, "main").unwrap();
    fail_chunk(&h.service, main, LLM_502).unwrap();
    let other = claim(&h, "other").expect("other's queue is its own");
    assert_eq!(other.error_count, 0);
}

// Recovery after a restart (TIM-94 decision 3: nothing queued is lost)

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn the_queue_survives_a_restart_with_the_chunk_in_flight_first() {
    let h = Harness::new();
    let a = ingest(&h, "main", &turn("s", "2026-10-01T06:00:00Z", "A.", "Ok."));
    let b = ingest(&h, "main", &turn("s", "2026-10-01T06:01:00Z", "B.", "Ok."));
    let c = ingest_doc(
        &h,
        "main",
        &document("c.md", "# C\n\nC.\n", date(2026, 9, 1)),
    );
    let in_flight = claim(&h, "main").unwrap();
    assert_eq!(in_flight.source, a.source);
    let in_flight_chunk = in_flight.chunk;

    let h = h.restart();
    assert_eq!(depth(&h, "main"), 3, "nothing queued is lost");
    let again = claim(&h, "main").expect("the restart released the lease");
    assert_eq!(
        again.chunk, in_flight_chunk,
        "the interrupted chunk goes first"
    );
    assert_eq!(again.error_count, 0, "a restart isn't a failure");
    complete_chunk(&h.service, again).unwrap();

    let mut rest = Vec::new();
    while let Some(lease) = claim(&h, "main") {
        rest.push(lease.source);
        complete_chunk(&h.service, lease).unwrap();
    }
    assert_eq!(rest, [b.source, c.source]);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn the_error_count_survives_a_restart() {
    // The cap counts attempts, not attempts since the last restart, or a
    // crash-looping daemon would retry a poisoned chunk for ever.
    let h = Harness::new();
    ingest(&h, "main", &turn("s", "2026-10-01T06:00:00Z", "A.", "Ok."));
    let lease = claim(&h, "main").unwrap();
    fail_chunk(&h.service, lease, LLM_502).unwrap();

    let h = h.restart();
    h.clock.advance(SignedDuration::from_hours(24));
    let lease = claim(&h, "main").unwrap();
    assert_eq!(lease.error_count, 1);
    let mut lease = lease;
    for attempt in 2..=CHUNK_RETRY_CAP {
        let result = fail_chunk(&h.service, lease, LLM_502).unwrap();
        if attempt == CHUNK_RETRY_CAP {
            assert_eq!(result, Failure::Failed);
            break;
        }
        h.clock.advance(SignedDuration::from_hours(24));
        lease = claim(&h, "main").unwrap();
    }
    assert_eq!(failed_chunks(&h.service, "main").unwrap().len(), 1);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn a_failed_chunk_stays_failed_after_a_restart() {
    let h = Harness::new();
    ingest(&h, "main", &turn("s", "2026-10-01T06:00:00Z", "A.", "Ok."));
    let failed = fail_until_failed(&h, "main");

    let h = h.restart();
    assert_eq!(depth(&h, "main"), 0);
    assert!(claim(&h, "main").is_none(), "surfaced, not retried");
    let listed = failed_chunks(&h.service, "main").unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].chunk, failed);
}

#[test]
#[ignore = "ingest not built yet; activate with the TIM-106 implementation"]
fn the_queue_order_survives_a_restart() {
    let h = Harness::new();
    let doc = ingest_doc(
        &h,
        "main",
        &document(
            "notes.md",
            "# A\n\nFirst.\n\n# B\n\nSecond.\n",
            date(2026, 9, 1),
        ),
    );
    let late = ingest(
        &h,
        "main",
        &turn("s", "2026-09-30T06:00:00Z", "Late.", "Ok."),
    );
    let early = ingest(
        &h,
        "main",
        &turn("s", "2026-09-20T06:00:00Z", "Early.", "Ok."),
    );

    let h = h.restart();
    let mut order = Vec::new();
    while let Some(lease) = claim(&h, "main") {
        order.push((lease.source, lease.position));
        complete_chunk(&h.service, lease).unwrap();
    }
    assert_eq!(
        order,
        [
            (early.source, 0),
            (late.source, 0),
            (doc.source, 0),
            (doc.source, 1),
        ]
    );
}
