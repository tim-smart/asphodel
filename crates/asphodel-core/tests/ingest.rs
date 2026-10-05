//! Source ingest, chunking, secret scanning and the extraction queue. Sources
//! are kept verbatim, so ingest is idempotent and secrets are removed before
//! the text is stored.
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
use std::time::Duration;

use asphodel_core::Service;
use asphodel_core::clock::{Clock, SimulatedClock};
use asphodel_core::config::Tuning;
use asphodel_core::entities::MergeRequest;
use asphodel_core::ingest::{Document, IngestError, Ingested, Outcome, Turn, TurnAuthor};
use asphodel_core::inspect::{ChunkState, ChunkView, Gone, SourceDetail, SourceQuery};
use asphodel_core::models::{FakeEmbedder, FakeLlm, FakeReranker, Models};
use asphodel_core::operations::{Audit, AuditList};
use asphodel_core::queue::{ChunkError, Failure, Lease, SourceKind};
use asphodel_core::retrieval::PrefetchRequest;
use asphodel_core::secrets::SecretKind;
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{DB_FILE, OpenOptions, Store};
use jiff::civil::{Date, date};
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use serde_json::json;
use uuid::Uuid;

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";

/// Message times before [`START`], a minute apart.
const T0: &str = "2026-10-01T06:00:00Z";
const T1: &str = "2026-10-01T06:01:00Z";
const T2: &str = "2026-10-01T06:02:00Z";

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
}

/// A data dir, which the store creates, removed even when an assertion
/// unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("asphodel-ingest-{}-{n}", std::process::id());
        Self(std::env::temp_dir().join(name))
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
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

/// A service on the fake models, on a data dir with two banks, `main` and
/// `other`. Field order matters: the service drops before the directory it
/// lives in.
struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    tuning: Tuning,
    dir: TestDir,
}

impl Harness {
    fn new() -> Self {
        Self::with_tuning("")
    }

    /// With `[llm] concurrency = n`, so each bank hands out up to `n`
    /// leases at once.
    fn with_concurrency(n: u32) -> Self {
        Self::with_tuning(&format!("[llm]\nconcurrency = {n}\n"))
    }

    fn with_tuning(toml: &str) -> Self {
        let tuning = Tuning::from_toml(&format!(
            "[injection.reranker_floors]\n\"{}\" = 0.0\n\
             [ranking.relevance_scales]\n\"{0}\" = 1.0\n\
             [reconcile.embedding_floors]\n\"{}\" = 0.5\n{toml}",
            FakeReranker::MODEL_ID,
            FakeEmbedder::MODEL_ID,
        ))
        .unwrap();
        let clock = Arc::new(SimulatedClock::new(at(START)));
        let h = Self::open(clock, tuning, TestDir::new());
        h.bank("main", &identity());
        h.bank("other", &identity());
        h
    }

    fn open(clock: Arc<SimulatedClock>, tuning: Tuning, dir: TestDir) -> Self {
        let store = Store::open(&dir.0, OpenOptions::default(), clock.clone()).unwrap();
        let service =
            Service::with_models(clock.clone(), store, tuning.clone(), Models::fake()).unwrap();
        Self {
            service,
            clock,
            tuning,
            dir,
        }
    }

    /// Drops the service, releasing the data-dir lock and every lease, and
    /// opens a new one on the same data dir and clock: a daemon restart.
    fn restart(self) -> Self {
        let Harness {
            service,
            clock,
            tuning,
            dir,
        } = self;
        drop(service);
        Self::open(clock, tuning, dir)
    }

    /// Creates `name` or merges `identity` into it, as bank config does.
    fn bank(&self, name: &str, identity: &BankIdentity) {
        self.service
            .ensure_bank_with_models(name, identity)
            .unwrap();
    }

    fn ingest(&self, bank: &str, turn: &Turn) -> Ingested {
        self.service.ingest_turn(bank, turn).unwrap()
    }

    /// The owner says `text` in session `s` at `message_at`.
    fn say(&self, bank: &str, message_at: &str, text: &str) -> Ingested {
        self.ingest(bank, &turn("s", message_at, text, "Ok."))
    }

    /// Ingests a document into `main`.
    fn doc(&self, id: &str, text: &str, reference_date: Date) -> Ingested {
        let document = document(id, text, reference_date);
        self.service.ingest_document("main", &document).unwrap()
    }

    fn claim(&self, bank: &str) -> Option<Lease> {
        self.service.claim_chunk(bank).unwrap()
    }

    fn depth(&self, bank: &str) -> usize {
        self.service.queue_depth(bank).unwrap()
    }

    fn source(&self, bank: &str, source: Uuid) -> SourceDetail {
        self.service.show_source(bank, &source.to_string()).unwrap()
    }

    /// How many sources `bank` holds.
    fn sources(&self, bank: &str) -> usize {
        let query = SourceQuery::default();
        self.service.list_sources(bank, &query).unwrap().total
    }

    /// The only chunk of a turn.
    fn chunk(&self, bank: &str, source: Uuid) -> ChunkView {
        let mut chunks = self.source(bank, source).chunks;
        assert_eq!(chunks.len(), 1);
        chunks.remove(0)
    }

    fn user(&self, bank: &str) -> Uuid {
        self.service.show_entity(bank, "user").unwrap().id
    }

    fn entities_in(&self, bank: &str) -> usize {
        self.service.entities(bank).unwrap().len()
    }

    /// Whether `needle` appears anywhere in the database or its WAL, after
    /// a checkpoint. Nothing stored may hold a secret or a forgotten turn.
    fn store_holds(&self, needle: &str) -> bool {
        let store = self.service.store().unwrap();
        store.checkpoint().unwrap();
        [DB_FILE.to_string(), format!("{DB_FILE}-wal")]
            .iter()
            .filter_map(|name| std::fs::read(store.dir().join(name)).ok())
            .any(|bytes| bytes.windows(needle.len()).any(|w| w == needle.as_bytes()))
    }
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

const LLM_502: ChunkError = ChunkError {
    kind: "llm_status",
    status: Some(502),
};

/// Characters `start..end` of `text`.
fn chars(text: &str, start: i64, end: i64) -> String {
    text.chars()
        .skip(start as usize)
        .take((end - start) as usize)
        .collect()
}

fn headings(chunk: &ChunkView) -> Vec<String> {
    chunk
        .heading_path
        .as_deref()
        .map(|path| serde_json::from_str(path).unwrap())
        .unwrap_or_default()
}

fn path(headings: &[&str]) -> Vec<String> {
    headings.iter().map(|heading| heading.to_string()).collect()
}

/// The stored document and its chunks' texts, after checking that the
/// chunks tile it: each range is in order and doesn't overlap, and every
/// non-whitespace character is in one.
fn tiles(detail: &SourceDetail) -> Vec<String> {
    let text = detail.text.as_deref().unwrap();
    let length = text.chars().count() as i64;
    let mut covered = vec![false; length as usize];
    let mut previous_end = 0;
    for chunk in &detail.chunks {
        assert!(chunk.start < chunk.end, "{chunk:?} is empty");
        assert!(chunk.start >= previous_end, "{chunk:?} overlaps");
        assert!(chunk.end <= length, "{chunk:?} runs past the end");
        covered[chunk.start as usize..chunk.end as usize].fill(true);
        previous_end = chunk.end;
    }
    for (index, character) in text.chars().enumerate() {
        assert!(
            covered[index] || character.is_whitespace(),
            "character {index} ({character:?}) is in no chunk"
        );
    }
    detail
        .chunks
        .iter()
        .map(|chunk| chars(text, chunk.start, chunk.end))
        .collect()
}

/// Three turns in `main`, a minute apart. Returns their sources.
fn three_turns(h: &Harness) -> Vec<Uuid> {
    [(T0, "One."), (T1, "Two."), (T2, "Three.")]
        .iter()
        .map(|(message_at, text)| h.say("main", message_at, text).source)
        .collect()
}

/// Fails the head of `bank` until it's marked failed, advancing the clock a
/// day between attempts in case the implementation backs off. Each attempt
/// counts one more error than the lease had. Returns the chunk and its
/// error count when it failed.
fn fail_until_failed(h: &Harness, bank: &str) -> (Uuid, u32) {
    let chunk = h.claim(bank).expect("a head to fail").chunk;
    for _ in 0..100 {
        let lease = h.claim(bank).expect("retried");
        assert_eq!(lease.chunk, chunk, "the head is retried in place");
        let errors = lease.error_count + 1;
        match h.service.fail_chunk(lease, LLM_502).unwrap() {
            Failure::Retry { error_count } => assert_eq!(error_count, errors),
            Failure::Failed => return (chunk, errors),
        }
        h.clock.advance(SignedDuration::from_hours(24));
    }
    panic!("the chunk never failed");
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
    let alnum = |len| body(ALNUM, len, seed);
    match kind {
        SecretKind::PrivateKey => {
            let label = ["OPENSSH PRIVATE", " KEY"].concat();
            let lines = [alnum(64), body(ALNUM, 64, seed + 1)].join("\n");
            format!("-----BEGIN {label}-----\n{lines}\n-----END {label}-----")
        }
        SecretKind::AwsAccessKey => ["AK", "IA", &body(UPPER, 16, seed)].concat(),
        SecretKind::GithubToken => ["gh", "p_", &alnum(36)].concat(),
        SecretKind::OpenAiKey => ["sk-", "proj-", &alnum(48)].concat(),
        SecretKind::AnthropicKey => ["sk-", "ant-", "api03-", &alnum(80)].concat(),
        SecretKind::SlackToken => ["xo", "xb-184736291045-5729104836271-", &alnum(24)].concat(),
        SecretKind::StripeKey => ["sk", "_live_", &alnum(24)].concat(),
        SecretKind::GoogleApiKey => ["AI", "za", &alnum(35)].concat(),
        SecretKind::Jwt => {
            let header = ["ey", "JhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"].concat();
            let claims = ["ey", "JzdWIiOiIxMjM0NTY3ODkwIn0"].concat();
            format!("{header}.{claims}.{}", alnum(43))
        }
        // Only the password; see `sentence_with`.
        SecretKind::UrlPassword => alnum(20),
    }
}

/// `text` containing the fake secret of `kind`, as it would be written.
fn sentence_with(kind: SecretKind, seed: usize) -> (String, String) {
    let secret = fake(kind, seed);
    let text = match kind {
        SecretKind::UrlPassword => {
            format!("the database is postgres://hermes:{secret}@db.internal:5432/app thanks")
        }
        _ => format!("my key is {secret} thanks"),
    };
    (text, secret)
}

/// A message time no other call has used, so no turn is a duplicate.
fn next_message_at() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("2026-10-01T05:{:02}:{:02}Z", (n / 60) % 60, n % 60)
}

// Secrets

#[test]
fn secrets_are_redacted_in_place_before_anything_is_stored() {
    let h = Harness::new();
    for kind in SecretKind::ALL {
        let (text, secret) = sentence_with(kind, 1);
        let got = h.say("main", &next_message_at(), &text);
        assert_eq!(got.secret_kinds, BTreeSet::from([kind]), "{kind:?}");
        let stored = h.source("main", got.source);
        assert_eq!(stored.secret_kinds, [kind.as_str()]);
        let stored = stored.text.unwrap();
        assert!(!stored.contains(&secret), "{kind:?} is gone");
        // The rest stays: for a URL only the password goes, so the URL
        // stays useful.
        let (before, after) = text.split_once(&secret).unwrap();
        assert!(stored.starts_with(before), "{stored}");
        assert!(stored.ends_with(after), "{stored}");
        assert!(!h.store_holds(&secret), "nothing stored holds {kind:?}");
    }

    // Every match goes, in the reply too, and each kind is recorded once. A
    // private key block goes whole.
    let first = fake(SecretKind::GithubToken, 4);
    let second = fake(SecretKind::GithubToken, 5);
    let key = fake(SecretKind::PrivateKey, 2);
    let (reply, aws) = sentence_with(SecretKind::AwsAccessKey, 8);
    let user = format!("old {first}, new {second}. Here it is:\n{key}\nKeep it safe.");
    let got = h.ingest("main", &turn("s", &next_message_at(), &user, &reply));
    assert_eq!(
        got.secret_kinds,
        BTreeSet::from([
            SecretKind::GithubToken,
            SecretKind::PrivateKey,
            SecretKind::AwsAccessKey
        ])
    );
    let stored = h.source("main", got.source);
    let text = stored.text.unwrap();
    assert!(text.starts_with("old ") && text.ends_with("\nKeep it safe."));
    assert!(stored.reply.unwrap().starts_with("my key is "));
    for secret in [&first, &second, &aws] {
        assert!(!h.store_holds(secret), "nothing stored holds the secret");
    }
    for line in key.lines() {
        assert!(!h.store_holds(line), "no line of the block survives");
    }
}

#[test]
fn ordinary_and_already_redacted_text_is_left_alone() {
    // Over-redaction loses memories, so the patterns are specific. A marker
    // is never itself a match, so storing redacted text again changes
    // nothing.
    let h = Harness::new();
    let ordinary = "Commit 9f1c2d3e4b5a69788796a5b4c3d2e1f0a9b8c7d6 fixed task-list sorting.\n\
                    Record 12345678-1234-7234-8234-123456789abc is complete.\n\
                    See https://example.com/docs?page=2 and ssh://git@github.com/tim/asphodel.\n\
                    AKIA is a prefix; sk- is a prefix; eyJ is a prefix.\n\
                    Meet at 10:30 on 2026-10-03, password reset by Friday.";
    let secrets = SecretKind::ALL
        .iter()
        .enumerate()
        .map(|(seed, kind)| sentence_with(*kind, seed).0)
        .collect::<Vec<_>>()
        .join("\n");
    let once = h.say("main", &next_message_at(), &secrets);
    assert_eq!(once.secret_kinds.len(), SecretKind::ALL.len());
    let redacted = h.source("main", once.source).text.unwrap();

    for text in [ordinary, redacted.as_str()] {
        let got = h.say("main", &next_message_at(), text);
        assert!(got.secret_kinds.is_empty(), "{:?}", got.secret_kinds);
        let stored = h.source("main", got.source);
        assert!(stored.secret_kinds.is_empty());
        assert_eq!(stored.text.as_deref(), Some(text));
    }
}

#[test]
fn the_keys_come_from_the_redacted_text() {
    // Nothing stored is derived from a secret, the content hash and the
    // chunk hashes included, so a resend whose only difference is the
    // secret itself is the same turn, and a document section that differs
    // only in its secret is one an earlier version had.
    let h = Harness::new();
    let (first_text, _) = sentence_with(SecretKind::OpenAiKey, 10);
    let (second_text, _) = sentence_with(SecretKind::OpenAiKey, 11);
    assert_ne!(first_text, second_text);
    let first = h.say("main", T0, &first_text);
    let second = h.say("main", T0, &second_text);
    assert_eq!(second.outcome, Outcome::Duplicate);
    assert_eq!(second.source, first.source);

    let infra = |seed| {
        let (line, password) = sentence_with(SecretKind::UrlPassword, seed);
        let text = format!("# Infra\n\n{line}\n\n# Other\n\nNothing secret.\n");
        (text, password)
    };
    let (v1, password) = infra(9);
    let got = h.doc("infra.md", &v1, date(2026, 9, 15));
    assert_eq!(got.secret_kinds, BTreeSet::from([SecretKind::UrlPassword]));
    assert_eq!(h.source("main", got.source).secret_kinds, ["url_password"]);
    assert!(!h.store_holds(&password));
    let v2 = format!("{}\n# New\n\nAdded.\n", infra(12).0);
    let again = h.doc("infra.md", &v2, date(2026, 9, 16));
    assert_eq!(again.outcome, Outcome::Stored);
    assert_eq!((again.chunks_queued, again.chunks_skipped), (1, 2));
}

// Chunking

#[test]
fn a_markdown_document_is_one_chunk_per_section_with_its_heading_path() {
    // Offsets count characters, not bytes: memories point into their
    // source by them.
    let text = "Notes from the week ☕.\n\n\
                # Work\n\nShipping Asphodel v1.\n\n\
                ## Asphodel\n\nIngest is next.\n\n\
                ## Hermes\n\nThe plugin waits. Grüße aus Köln. 🌧️\n\n\
                # Home ##\n\nThe fence needs fixing. Run this:\n\n\
                ```sh\n# install the daemon\nnix profile install\n```\n\n\
                #asphodel is the tag we use.\n\n\
                ### Garden\n\nPlant the tomatoes.\n";
    let h = Harness::new();
    let got = h.doc("week.md", text, date(2026, 9, 15));
    let stored = h.source("main", got.source);
    let paths: Vec<Vec<String>> = stored.chunks.iter().map(headings).collect();
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
        "small sections are never merged, closing #s aren't part of a heading, \
         and hashes in code blocks and hashtags aren't headings"
    );
    let texts = tiles(&stored);
    // A section starts at its heading.
    assert!(texts[1].starts_with("# Work"));
    assert!(texts[2].contains("Ingest is next."));
    assert!(texts[4].contains("#asphodel is the tag"));
    assert!(texts[5].starts_with("### Garden"));
    assert_eq!(got.chunks_queued, 6);
}

#[test]
fn a_long_section_is_split_at_paragraphs_and_a_long_paragraph_anyway() {
    let paragraphs: Vec<String> = (0..12)
        .map(|n| format!("Paragraph {n}. {}", "lorem ipsum ".repeat(75)))
        .collect();
    let long = "This sentence repeats. ".repeat(400);
    let text = format!(
        "# Journal\n\n{}\n\n# After\n\n{long}\n",
        paragraphs.join("\n\n")
    );
    let h = Harness::new();
    let stored = h.source("main", h.doc("journal.md", &text, date(2026, 9, 15)).source);
    let texts = tiles(&stored);
    let in_section = |name: &str| -> Vec<&String> {
        let chunks = stored.chunks.iter().zip(&texts);
        chunks
            .filter(|(chunk, _)| headings(chunk) == path(&[name]))
            .map(|(_, text)| text)
            .collect()
    };
    let journal = in_section("Journal");
    assert!(journal.len() > 1, "the section was split");
    // A paragraph is never split across chunks, but one long paragraph is.
    for p in &paragraphs {
        let holding = journal.iter().filter(|text| text.contains(p.as_str()));
        assert_eq!(holding.count(), 1);
    }
    assert!(in_section("After").len() > 1);
}

// Turns

#[test]
fn a_turn_is_stored_verbatim_with_its_provenance() {
    let h = Harness::new();
    let said = "[Sam] I'm flying to Berlin on 3 October.";
    let mut sent = discord_turn("2026-10-01T06:59:30Z", said, author("5678", "Sam"));
    sent.assistant_text = "Safe travels, Sam!".into();
    sent.recall_id = Some("recall-1".into());
    let got = h.ingest("main", &sent);

    assert_eq!(got.outcome, Outcome::Stored);
    assert_eq!((got.chunks_queued, got.chunks_skipped), (1, 0));
    assert!(got.secret_kinds.is_empty());
    let stored = h.source("main", got.source);
    assert_eq!(stored.kind, SourceKind::Turn);
    assert_eq!(stored.session_id.as_deref(), Some("thread-1"));
    assert_eq!(stored.message_at, Some(at("2026-10-01T06:59:30Z")));
    // observed_at is the message time, not when it arrived.
    assert_eq!(stored.observed_at, at("2026-10-01T06:59:30Z"));
    assert_eq!(stored.ingested_at, at(START));
    assert_eq!(stored.timezone, TZ);
    // The speaker prefix stays.
    assert_eq!(stored.text.as_deref(), Some(said));
    assert_eq!(stored.reply.as_deref(), Some("Safe travels, Sam!"));
    assert_eq!(stored.platform.as_deref(), Some("discord"));
    assert_eq!(stored.author_name.as_deref(), Some("Sam"));
    assert_eq!((stored.gone, stored.document_id), (None, None));
    assert!(stored.secret_kinds.is_empty());

    // A turn without a timezone takes the bank's.
    let mut berlin = identity();
    berlin.timezone = Some("Europe/Berlin".into());
    h.bank("berlin", &berlin);
    let mut sent = turn("s", T0, "Hello.", "Hi.");
    sent.timezone = None;
    let got = h.ingest("berlin", &sent);
    assert_eq!(h.source("berlin", got.source).timezone, "Europe/Berlin");
}

#[test]
fn an_unknown_timezone_or_bank_is_refused_and_nothing_is_stored() {
    let h = Harness::new();
    let mars = || Some("Mars/Olympus_Mons".into());
    let hello = turn("s", T0, "Hello.", "Hi.");
    let bad = Turn {
        timezone: mars(),
        ..hello.clone()
    };
    assert!(matches!(
        h.service.ingest_turn("main", &bad),
        Err(IngestError::InvalidTimezone)
    ));
    let bad_doc = Document {
        timezone: mars(),
        ..document("notes.md", "Hello.", date(2026, 9, 15))
    };
    assert!(matches!(
        h.service.ingest_document("main", &bad_doc),
        Err(IngestError::InvalidTimezone)
    ));
    assert!(matches!(
        h.service.ingest_turn("nowhere", &hello),
        Err(IngestError::UnknownBank)
    ));
    assert_eq!((h.sources("main"), h.depth("main")), (0, 0));
}

// Idempotency

#[test]
fn the_same_turn_twice_does_nothing_the_second_time() {
    // Hermes retries, and the plugin's spool replays.
    let h = Harness::new();
    let sent = turn("s", T0, "I moved to Wellington.", "Noted.");
    let first = h.ingest("main", &sent);
    h.clock.advance(SignedDuration::from_mins(5));
    let second = h.ingest("main", &sent);

    assert_eq!(second.outcome, Outcome::Duplicate);
    assert_eq!(second.source, first.source, "the existing source");
    assert_eq!((second.chunks_queued, second.speaker), (0, None));
    assert_eq!((h.sources("main"), h.depth("main")), (1, 1));
    // A duplicate doesn't touch the stored source.
    assert_eq!(h.source("main", first.source).ingested_at, at(START));

    // Completing the chunk marks it extracted and takes it off the queue,
    // and a resend doesn't put it back.
    h.service.complete_chunk(h.claim("main").unwrap()).unwrap();
    assert_eq!(h.chunk("main", first.source).state, ChunkState::Extracted);
    assert_eq!(h.depth("main"), 0);
    assert_eq!(h.ingest("main", &sent).outcome, Outcome::Duplicate);
    assert_eq!(h.depth("main"), 0);
    assert!(h.claim("main").is_none());
}

#[test]
fn the_same_words_in_another_session_at_another_time_or_in_another_bank_are_new() {
    let h = Harness::new();
    let sources: BTreeSet<Uuid> = [
        ("main", "s1", T0),
        ("main", "s2", T0),
        ("main", "s1", "2026-10-02T06:00:00Z"),
        ("other", "s1", T0),
    ]
    .iter()
    .map(|(bank, session, message_at)| {
        let sent = turn(session, message_at, "Good morning.", "Morning!");
        let got = h.ingest(bank, &sent);
        assert_eq!(got.outcome, Outcome::Stored);
        got.source
    })
    .collect();
    assert_eq!(sources.len(), 4);
    assert_eq!((h.depth("main"), h.depth("other")), (3, 1));
}

// Documents

#[test]
fn a_document_is_stored_with_its_reference_date() {
    let h = Harness::new();
    let text = "# Trip\n\nFlying to Berlin tomorrow.\n";
    let got = h.doc("trip.md", text, date(2026, 9, 15));

    assert_eq!(got.outcome, Outcome::Stored);
    assert_eq!(got.speaker, None, "documents have no speaker");
    let stored = h.source("main", got.source);
    assert_eq!(stored.kind, SourceKind::Document);
    assert_eq!(stored.document_id.as_deref(), Some("trip.md"));
    assert_eq!(stored.text.as_deref(), Some(text));
    assert_eq!((stored.reply, stored.session_id), (None, None));
    assert_eq!(stored.reference_date.as_deref(), Some("2026-09-15"));
    assert_eq!(stored.timezone, TZ);
    assert_eq!(stored.ingested_at, at(START));
    // A time is stored as the start of its unit in the source's timezone,
    // so the reference date is midnight in Auckland.
    assert_eq!(stored.observed_at, midnight(date(2026, 9, 15)));

    // The key is the document id and content hash, so a new reference date
    // alone doesn't make a new version.
    let second = h.doc("trip.md", text, date(2026, 9, 20));
    assert_eq!(second.outcome, Outcome::Duplicate);
    assert_eq!((second.source, second.chunks_queued), (got.source, 0));
    assert_eq!((h.sources("main"), h.depth("main")), (1, 1));
}

fn midnight(day: Date) -> Timestamp {
    day.to_zoned(TimeZone::get(TZ).unwrap())
        .unwrap()
        .timestamp()
}

const PLANS: &str = "# Plans\n\nFly to Berlin on 3 October.\n\n";
const WORK_V1: &str = "# Work\n\nShip Asphodel v1.\n\n";
const WORK_V2: &str = "# Work\n\nShip Asphodel v1 by Friday.\n\n";
const HOME: &str = "# Home\n\nFix the fence.\n";

#[test]
fn an_edited_document_queues_only_chunks_no_earlier_version_had() {
    // Chunks are matched by hash, and a chunk seen in any earlier version
    // is skipped. A section repeated within the version being ingested
    // isn't skipped.
    let h = Harness::new();
    let v1 = [PLANS, WORK_V1].concat();
    let v2 = [PLANS, WORK_V2, HOME, "\n\n", HOME].concat();
    let v3 = [PLANS, WORK_V1, HOME].concat();

    let first = h.doc("life.md", &v1, date(2026, 9, 1));
    assert_eq!(first.chunks_queued, 2);

    let second = h.doc("life.md", &v2, date(2026, 9, 8));
    // A new version is a new source.
    assert_eq!(second.outcome, Outcome::Stored);
    assert_ne!(second.source, first.source);
    // The edited Work and both Home sections; Plans is unchanged.
    assert_eq!((second.chunks_queued, second.chunks_skipped), (3, 1));
    let stored = h.source("main", second.source);
    let queued: Vec<(Vec<String>, String)> = stored
        .chunks
        .iter()
        .filter(|chunk| chunk.state == ChunkState::Queued)
        .map(|chunk| {
            let text = chars(stored.text.as_deref().unwrap(), chunk.start, chunk.end);
            (headings(chunk), text)
        })
        .collect();
    assert_eq!(queued.len(), 3);
    assert_eq!(queued[0].0, path(&["Work"]));
    assert!(queued[0].1.contains("by Friday"));
    assert_eq!(queued[1].0, path(&["Home"]));
    assert_eq!(queued[2].0, path(&["Home"]));
    // New chunks are extracted at the new version's reference date.
    assert_eq!(stored.observed_at, midnight(date(2026, 9, 8)));

    // Every section of v3 was in v1 or v2, if not in both.
    let third = h.doc("life.md", &v3, date(2026, 9, 15));
    assert_eq!(third.outcome, Outcome::Stored);
    assert_eq!((third.chunks_queued, third.chunks_skipped), (0, 3));
    assert_eq!(h.depth("main"), 5);

    // An earlier reference date is accepted: reconciliation's direction
    // rule makes older claims harmless, and a rejection would be a failure
    // nobody sees.
    let garden = [PLANS, "# Garden\n\nWater the roses.\n"].concat();
    let older = h.doc("life.md", &garden, date(2026, 8, 1));
    assert_eq!((older.outcome, older.chunks_queued), (Outcome::Stored, 1));
}

#[test]
fn memories_from_a_removed_section_are_left_alone() {
    // Deleting a line from a note doesn't make it false.
    let h = Harness::new();
    h.doc("life.md", &[PLANS, WORK_V1].concat(), date(2026, 9, 1));
    let nothing = json!({"claims": [], "used_injected_ids": []});
    let work = json!({"claims": [{
        "content": "Tim is shipping Asphodel v1.", "kind": "state", "quote": "Ship Asphodel v1.",
        "significance": "minor", "remember_this": false, "changes_something": false,
        "valid_from": null, "valid_until": null, "window_confidence": "high",
        "until_event": null, "due_at": null, "volatility": null, "recurrence_text": null,
        "recurrence_rrule": null, "recurrence_start": null, "entities": [],
    }], "used_injected_ids": []});
    let llm = FakeLlm::scripted("fake-llm", vec![nothing, work]);
    h.service.extract_next("main", &llm).unwrap();
    let extracted = h.service.extract_next("main", &llm).unwrap().unwrap();

    let v2 = h.doc("life.md", PLANS, date(2026, 9, 8));
    assert_eq!(v2.outcome, Outcome::Stored);
    let memory = extracted.memories[0].to_string();
    let view = h.service.show_memory("main", &memory).unwrap();
    // The passage the memory rests on stays.
    assert_eq!(view.source.passage.as_deref(), Some("Ship Asphodel v1."));
}

// The turn that asks to forget is never stored

#[test]
fn a_forget_request_is_stored_only_as_a_tombstone_even_when_sent_again() {
    let h = Harness::new();
    // The recall row for the turn's `recall_id` holds the request as its
    // query, so it goes too. Other recalls stay.
    let prefetch = |query: &str| {
        let request = PrefetchRequest {
            session_id: "s".into(),
            query: query.into(),
            ..PrefetchRequest::default()
        };
        h.service.prefetch("main", &request).unwrap().recall_id
    };
    let request = "Forget that my passport number is LZ8841207.";
    let asked = prefetch(request);
    let other = prefetch("Berlin");
    let flagged = Turn {
        forget_requested: true,
        recall_id: Some(asked.to_string()),
        ..turn("s", T0, request, "Done, I've forgotten it.")
    };
    let got = h.ingest("main", &flagged);

    assert_eq!(got.outcome, Outcome::Tombstone);
    assert_eq!((got.chunks_queued, got.speaker), (0, None));
    let stored = h.source("main", got.source);
    assert_eq!((stored.text, stored.reply), (None, None));
    assert_eq!(stored.gone, Some(Gone::ForgetRequested));
    assert!(stored.chunks.is_empty(), "nothing to extract");
    assert_eq!(h.depth("main"), 0);
    assert!(h.claim("main").is_none());
    let Audit::Recalls { recalls } = h.service.audit("main", AuditList::Recalls, None).unwrap()
    else {
        unreachable!()
    };
    let recalls: Vec<Uuid> = recalls.into_iter().map(|recall| recall.id).collect();
    assert_eq!(recalls, [other]);

    // The spool may replay the turn, and a resend could lose the flag.
    let plain = Turn {
        forget_requested: false,
        recall_id: None,
        ..flagged.clone()
    };
    for resend in [&flagged, &plain] {
        let again = h.ingest("main", resend);
        assert_eq!(again.outcome, Outcome::Duplicate);
        assert_eq!(again.source, got.source);
    }
    assert_eq!(h.source("main", got.source).text, None);
    assert_eq!(h.depth("main"), 0);
    assert!(!h.store_holds("LZ8841207"));
    assert!(!h.store_holds("Done, I've forgotten it."));
}

// Speakers

#[test]
fn the_owners_platform_id_or_no_author_is_the_owner() {
    // The CLI, TUI and Hermes UI send no author.
    let h = Harness::new();
    for sent in [
        discord_turn(T0, "I'm off to Berlin.", author("1234", "tim")),
        turn("s", T0, "Hello.", "Hi."),
    ] {
        let speaker = h.ingest("main", &sent).speaker.unwrap();
        assert!(speaker.owner);
        assert_eq!(speaker.entity, h.user("main"));
    }
    assert_eq!(h.entities_in("main"), 2, "no new entity for the owner");
}

#[test]
fn anyone_else_becomes_a_person_of_their_own_in_one_bank() {
    let h = Harness::new();
    let (sam, owner) = speaker(&h, "main", "discord", "5678", "Sam");
    assert!(!owner);
    assert_ne!(sam, h.user("main"));
    assert_eq!(h.entities_in("main"), 3);
    let view = h.service.show_entity("main", &sam.to_string()).unwrap();
    assert_eq!((view.name.as_str(), view.kind.as_str()), ("Sam", "person"));
    assert!(view.aliases.contains(&"discord:5678".to_string()));
    // A new alias is a logged edit, so a mislink can be undone.
    assert!(view.edits.iter().any(|edit| edit.kind == "alias_added"));

    // Found again by platform id, even after a rename, and never from
    // another bank.
    let renamed = speaker(&h, "main", "discord", "5678", "Samantha");
    assert_eq!(renamed, (sam, false));
    assert_eq!(h.entities_in("main"), 3);
    let (elsewhere, _) = speaker(&h, "other", "discord", "5678", "Sam");
    assert_ne!(elsewhere, sam);
    assert_eq!(h.entities_in("other"), 3);
}

#[test]
fn only_the_platform_id_is_an_identity() {
    // Only the owner can forget, keep or unkeep, so ownership comes from
    // platform ids alone. Sharing the owner's name doesn't make someone the
    // owner, the owner's id on another platform is a different account, and
    // a display name that looks like a platform id is free text.
    let h = Harness::new();
    let (namesake, owner) = speaker(&h, "main", "discord", "9999", "Tim");
    assert!(!owner);
    assert_ne!(namesake, h.user("main"));

    let (telegram, owner) = speaker(&h, "main", "telegram", "1234", "Tim");
    assert!(!owner);
    let view = h.service.show_entity("main", &telegram.to_string());
    assert!(view.unwrap().aliases.contains(&"telegram:1234".to_string()));

    let (impostor, _) = speaker(&h, "main", "discord", "2000", "discord:3000");
    let (real, _) = speaker(&h, "main", "discord", "3000", "Sam");
    assert_ne!(real, impostor, "discord:3000 is its own speaker");
    assert_eq!(h.entities_in("main"), 6);
}

// The extraction queue

#[test]
fn a_bank_hands_out_up_to_its_concurrency_in_queue_order() {
    // Each bank has a pool of leases, `[llm] concurrency` of them: the heads
    // of its queue in order, as many as the pool holds. Another bank isn't
    // held up.
    for n in [1, 2] {
        let h = Harness::with_concurrency(n);
        let sources = three_turns(&h);
        h.say("other", T0, "Elsewhere.");

        let mut out: Vec<Lease> = (0..n).map(|_| h.claim("main").unwrap()).collect();
        let leased: Vec<Uuid> = out.iter().map(|lease| lease.source).collect();
        assert_eq!(leased, sources[..n as usize], "concurrency {n}");
        assert!(h.claim("main").is_none(), "the pool is full");
        let other = h.claim("other").expect("another bank isn't held up");
        let queued = h.service.chunks("main", false).unwrap().queued;
        let in_flight = queued.iter().filter(|chunk| chunk.in_flight).count();
        assert_eq!(queued.len(), 3);
        // A leased chunk is still in the queue.
        assert_eq!(in_flight, n as usize);

        h.service.complete_chunk(out.remove(0)).unwrap();
        let next = h.claim("main").expect("a freed lease takes the next head");
        assert_eq!(next.source, sources[n as usize]);
        h.service.complete_chunk(other).unwrap();
    }
}

#[test]
fn a_hold_waits_for_every_lease_in_the_pool_and_hands_out_none_meanwhile() {
    // Deleting a bank holds it. With a pool the hold must wait for every
    // lease that's out and stop new ones, or a busy bank would never come
    // free.
    let h = Harness::with_concurrency(2);
    three_turns(&h);
    let one = h.claim("main").expect("main's head");
    let two = h
        .claim("main")
        .expect("the next head, while the first is out");

    std::thread::scope(|scope| {
        let deleting = scope.spawn(|| h.service.delete_bank("main", "main"));
        std::thread::sleep(Duration::from_millis(300));
        assert!(!deleting.is_finished(), "the delete waits for both leases");

        drop(one);
        std::thread::sleep(Duration::from_millis(300));
        assert!(!deleting.is_finished(), "the second lease is still out");
        // The freed lease isn't handed out while the hold waits.
        assert!(h.claim("main").is_none());

        drop(two);
        deleting.join().unwrap().unwrap();
    });
    assert!(h.service.claim_chunk("main").is_err(), "the bank is gone");
}

#[test]
fn turns_go_ahead_of_documents_then_observed_at_order_across_a_restart() {
    // Serialised in observed_at order, with turns ahead of document chunks.
    // The order is the stored queue's, so it holds after a restart.
    let h = Harness::new();
    let notes = "# A\n\nFirst.\n\n# B\n\nSecond.\n";
    let doc = h.doc("notes.md", notes, date(2026, 9, 1));
    let late = h.say("main", "2026-09-30T06:00:00Z", "Late.");
    // Arrives last, from the spool, but was said first.
    let early = h.say("main", "2026-09-20T06:00:00Z", "Early.");

    let mut order = Vec::new();
    let mut take = |h: &Harness, lease: Lease| {
        order.push((lease.source, lease.source_kind, lease.position));
        h.service.complete_chunk(lease).unwrap();
    };
    take(&h, h.claim("main").unwrap());
    let h = h.restart();
    while let Some(lease) = h.claim("main") {
        take(&h, lease);
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
fn a_failed_attempt_is_counted_and_retried_in_place() {
    let h = Harness::new();
    let first = h.say("main", T0, "First.");
    h.say("main", T1, "Second.");
    let lease = h.claim("main").unwrap();
    let head = lease.chunk;

    let failure = h.service.fail_chunk(lease, LLM_502).unwrap();
    assert_eq!(failure, Failure::Retry { error_count: 1 });
    let chunk = h.chunk("main", first.source);
    assert_eq!(chunk.state, ChunkState::Queued);
    assert_eq!(chunk.error_count, 1);
    assert_eq!(chunk.error_kind.as_deref(), Some("llm_status"));
    assert_eq!(chunk.failed_at, None);
    assert_eq!(h.depth("main"), 2);

    // Order holds: the second chunk never overtakes a head that's being
    // retried. The implementation may back off, so look a day later too.
    let retried = h.claim("main").or_else(|| {
        h.clock.advance(SignedDuration::from_hours(24));
        h.claim("main")
    });
    let retried = retried.expect("retried within a day");
    assert_eq!((retried.chunk, retried.error_count), (head, 1));
}

#[test]
fn reaching_the_retry_cap_marks_the_chunk_failed_and_the_queue_moves_on() {
    let h = Harness::new();
    let first = h.say("main", T0, "First.");
    let second = h.say("main", T1, "Second.");
    let (failed, attempts) = fail_until_failed(&h, "main");

    let chunk = h.chunk("main", first.source);
    assert_eq!((chunk.id, chunk.state), (failed, ChunkState::Failed));
    assert_eq!(chunk.failed_at, Some(h.clock.now()));
    assert_eq!(chunk.error_count, attempts);
    assert_eq!(h.depth("main"), 1, "a failed chunk isn't waiting");

    let listed = h.service.failed_chunks("main").unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!((listed[0].chunk, listed[0].source), (failed, first.source));
    assert_eq!(listed[0].error_count, attempts);
    assert_eq!(listed[0].error_kind, "llm_status");
    assert_eq!(listed[0].failed_at, h.clock.now());

    let next = h.claim("main").expect("the queue moves on");
    assert_eq!(next.source, second.source);
    drop(next);
    // It keeps its text, so it can be retried.
    let retried = h.service.retry_chunks("main", Some(&[failed])).unwrap();
    assert_eq!(retried.retried, [failed]);
    assert_eq!(h.depth("main"), 2);
}

// Recovery after a restart

#[test]
fn the_queue_survives_a_restart_with_the_chunk_in_flight_first() {
    let h = Harness::new();
    let a = h.say("main", T0, "A.");
    let b = h.say("main", T1, "B.");
    let c = h.doc("c.md", "# C\n\nC.\n", date(2026, 9, 1));
    let in_flight = h.claim("main").unwrap();
    assert_eq!(in_flight.source, a.source);
    let in_flight_chunk = in_flight.chunk;

    let h = h.restart();
    assert_eq!(h.depth("main"), 3, "nothing queued is lost");
    let again = h.claim("main").expect("the restart released the lease");
    // The interrupted chunk goes first.
    assert_eq!(again.chunk, in_flight_chunk);
    assert_eq!(again.error_count, 0, "a restart isn't a failure");
    h.service.complete_chunk(again).unwrap();

    let mut rest = Vec::new();
    while let Some(lease) = h.claim("main") {
        rest.push(lease.source);
        h.service.complete_chunk(lease).unwrap();
    }
    assert_eq!(rest, [b.source, c.source]);
}

#[test]
fn the_error_count_and_a_failure_survive_a_restart() {
    // The cap counts attempts, not attempts since the last restart, or a
    // crash-looping daemon would retry a poisoned chunk for ever.
    let h = Harness::new();
    h.say("main", T0, "A.");
    h.service
        .fail_chunk(h.claim("main").unwrap(), LLM_502)
        .unwrap();

    let h = h.restart();
    h.clock.advance(SignedDuration::from_hours(24));
    assert_eq!(h.claim("main").unwrap().error_count, 1);
    let (failed, _) = fail_until_failed(&h, "main");

    let h = h.restart();
    assert_eq!(h.depth("main"), 0);
    assert!(h.claim("main").is_none(), "surfaced, not retried");
    let listed = h.service.failed_chunks("main").unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].chunk, failed);
}

#[test]
fn a_lease_from_before_a_restart_cannot_complete_or_fail_the_new_workers_chunk() {
    // A lease belongs to the service that issued it. After a restart the
    // chunk is claimed again; the old lease must not change it while the
    // new worker still holds it.
    let h = Harness::new();
    let a = h.say("main", T0, "A.");
    let stale = h.claim("main").unwrap();
    let h = h.restart();
    let current = h.claim("main").expect("the restart released the lease");
    assert_eq!(current.chunk, stale.chunk);
    // A stale lease is refused.
    assert!(h.service.complete_chunk(stale).is_err());
    assert_eq!(h.chunk("main", a.source).state, ChunkState::InFlight);
    assert_eq!(h.depth("main"), 1, "the chunk is still queued");

    // Another restart makes the current lease stale in turn.
    let h = h.restart();
    let newest = h.claim("main").unwrap();
    assert!(h.service.fail_chunk(current, LLM_502).is_err());
    let chunk = h.chunk("main", a.source);
    assert_eq!((chunk.error_count, chunk.error_kind), (0, None));
    let failure = h.service.fail_chunk(newest, LLM_502).unwrap();
    assert_eq!(failure, Failure::Retry { error_count: 1 });
}

#[test]
fn a_lease_from_another_store_is_refused() {
    // Two stores whose bank and queue rowids coincide: one store's lease
    // must not complete or fail the other's chunk.
    let a = Harness::new();
    let b = Harness::new();
    a.say("main", T0, "A.");
    let in_b = b.say("main", T0, "A.").source;
    let from_a = a.claim("main").unwrap();
    let held_by_b = b.claim("main").unwrap();

    assert!(b.service.complete_chunk(from_a).is_err());
    assert_eq!(b.depth("main"), 1);

    let from_a = a
        .claim("main")
        .expect("dropping the lease released a's bank");
    assert!(b.service.fail_chunk(from_a, LLM_502).is_err());
    assert_eq!(b.chunk("main", in_b).error_count, 0);

    b.service.complete_chunk(held_by_b).unwrap();
    assert_eq!(b.depth("main"), 0);
    assert_eq!(a.depth("main"), 1, "a's queue is untouched");
}

// Speaker ids and the version 2 migration

/// Puts the harness's store back to schema version 1, as the version 1
/// binary would have left it after the same calls: drops `speaker_ids` and
/// the edits only version 2 writes, and leaves one `migrations` row 0 to 1.
/// Every alias, entity and `entity_created` edit stays as written, because
/// version 1 bank config and ingest wrote them the same way. Reopening
/// migrates it to version 2.
fn downgrade_to_v1_and_reopen(h: Harness) -> Harness {
    h.service
        .store()
        .unwrap()
        .connection()
        .execute_batch(
            "DROP TABLE speaker_ids;
             DELETE FROM edits WHERE kind = 'speaker_id_set';
             DELETE FROM migrations;
             INSERT INTO migrations (from_version, to_version, binary_version, started_at,
                                     completed_at)
               VALUES (0, 1, 'v1', 0, 0);
             PRAGMA user_version = 1;",
        )
        .unwrap();
    h.restart()
}

/// The speaker ids of `bank`: platform id to entity, sorted.
fn speaker_ids(h: &Harness, bank: &str) -> Vec<(String, Uuid)> {
    let mut ids: Vec<(String, Uuid)> = h
        .service
        .entities(bank)
        .unwrap()
        .into_iter()
        .flat_map(|entity| {
            let view = h.service.show_entity(bank, &entity.id.to_string());
            let ids = view.unwrap().speaker_ids.into_iter();
            ids.map(move |id| (id, entity.id))
        })
        .collect();
    ids.sort();
    ids
}

fn speaker(h: &Harness, bank: &str, platform: &str, id: &str, name: &str) -> (Uuid, bool) {
    let sent = Turn {
        platform: Some(platform.into()),
        ..discord_turn(&next_message_at(), "Hello.", author(id, name))
    };
    let speaker = h
        .ingest(bank, &sent)
        .speaker
        .expect("a stored turn has a speaker");
    (speaker.entity, speaker.owner)
}

/// The speaker ids of `bank` that map to its `user`.
fn owner_ids(h: &Harness, bank: &str) -> Vec<String> {
    let mut ids = h.service.show_entity(bank, "user").unwrap().speaker_ids;
    ids.sort();
    ids
}

/// How many speaker id takeovers `bank`'s `user` has logged.
fn takeovers(h: &Harness, bank: &str) -> usize {
    let user = h.service.show_entity(bank, "user").unwrap();
    let edits = user.edits.iter();
    edits.filter(|edit| edit.kind == "speaker_id_set").count()
}

/// Sends `bank`'s owner platform ids again, as every Hermes plugin
/// instance does from `initialize` (docs/upgrading.md).
fn refresh_owner_ids(h: &Harness, bank: &str, ids: &[&str]) {
    let owner_platform_ids = ids.iter().map(|id| id.to_string()).collect();
    let identity = BankIdentity {
        owner_platform_ids,
        ..BankIdentity::default()
    };
    h.bank(bank, &identity);
}

/// `bank` created with no owner platform id.
fn bank_without_owner_ids(h: &Harness, bank: &str) {
    let mut identity = identity();
    identity.owner_platform_ids.clear();
    h.bank(bank, &identity);
}

#[test]
fn an_upgrade_maps_no_owner_ids_until_bank_config_is_sent_again() {
    // Version 1 kept the owner's platform ids only as aliases of `user`,
    // next to the owner's names, renames included, with nothing recording
    // which was which. The upgrade fails closed and maps none of them;
    // bank config sent again maps the ids, and never the names.
    let h = Harness::new();
    let mut renamed = identity();
    renamed.owner_name = Some("Timothy".into());
    renamed.owner_platform_ids = vec!["telegram:42".into()];
    h.bank("main", &renamed);
    let h = downgrade_to_v1_and_reopen(h);
    let user = h.user("main");
    assert_eq!(speaker_ids(&h, "main"), [], "no owner id is inferred");
    let (entity, owner) = speaker(&h, "main", "discord", "1234", "Tim");
    assert!(!owner, "until the config is sent again");
    assert_ne!(entity, user);
    assert!(!speaker(&h, "main", "telegram", "42", "Tim").1);

    refresh_owner_ids(&h, "main", &["discord:1234", "telegram:42"]);
    assert_eq!(owner_ids(&h, "main"), ["discord:1234", "telegram:42"]);
    assert_eq!(speaker(&h, "main", "discord", "1234", "Tim"), (user, true));
    assert_eq!(speaker(&h, "main", "telegram", "42", "Tim"), (user, true));
    for (id, _) in speaker_ids(&h, "main") {
        assert!(id != "Tim" && id != "Timothy", "a name was mapped: {id}");
    }
}

#[test]
fn an_upgrade_does_not_map_an_owner_name_shaped_like_a_platform_id() {
    // The trust boundary of the backfill. An owner whose configured name
    // looks like a platform id still has a name, not an id: a stranger
    // whose platform id happens to match it must not become the owner.
    // The same goes for the assistant's name.
    let h = Harness::new();
    let mut odd = BankIdentity {
        owner_name: Some("discord:9999".into()),
        assistant_name: Some("discord:8888".into()),
        ..identity()
    };
    h.bank("odd", &odd);
    let h = downgrade_to_v1_and_reopen(h);
    let user = h.user("odd");
    let assistant = h.service.show_entity("odd", "assistant").unwrap().id;
    assert_eq!(speaker_ids(&h, "odd"), [], "no owner id is inferred");
    let check_names = |h: &Harness| {
        let (entity, owner) = speaker(h, "odd", "discord", "9999", "Stranger");
        // A stranger with the owner's name as their id isn't the owner.
        assert!(!owner);
        assert_ne!(entity, user);
        let (entity, owner) = speaker(h, "odd", "discord", "8888", "Other");
        assert!(!owner);
        assert_ne!(entity, assistant);
    };
    check_names(&h);

    odd.assistant_name = None;
    h.bank("odd", &odd);
    // Only the configured platform id is an identity.
    assert_eq!(owner_ids(&h, "odd"), ["discord:1234"]);
    check_names(&h);
    assert_eq!(speaker(&h, "odd", "discord", "1234", "Tim"), (user, true));
}

#[test]
fn an_upgrade_does_not_map_a_former_owner_name_shaped_like_a_platform_id() {
    // A rename keeps the old name as an alias of `user`, and version 1
    // recorded nothing that tells it from a platform id. Neither the
    // upgrade nor the config sent again after it makes it an identity.
    let h = Harness::new();
    let named = |name: &str| BankIdentity {
        owner_name: Some(name.into()),
        ..identity()
    };
    h.bank("renamed", &named("discord:9999"));
    h.bank("renamed", &named("Tim"));
    let aliases = h.service.show_entity("renamed", "user").unwrap().aliases;
    // The former name stays an alias of user.
    assert!(aliases.contains(&"discord:9999".to_string()));
    let h = downgrade_to_v1_and_reopen(h);
    let user = h.user("renamed");

    assert_eq!(owner_ids(&h, "renamed"), Vec::<String>::new());
    let (stranger, owner) = speaker(&h, "renamed", "discord", "9999", "Stranger");
    assert!(!owner, "before the config is sent again");
    assert_ne!(stranger, user);

    refresh_owner_ids(&h, "renamed", &["discord:1234"]);
    assert_eq!(owner_ids(&h, "renamed"), ["discord:1234"]);
    let again = speaker(&h, "renamed", "discord", "9999", "Stranger");
    assert_eq!(again, (stranger, false), "after it too");
    let tim = speaker(&h, "renamed", "discord", "1234", "Tim");
    assert_eq!(tim, (user, true));
}

#[test]
fn an_upgrade_keeps_the_speakers_version_1_ingest_created() {
    let h = Harness::new();
    let (sam, _) = speaker(&h, "main", "discord", "5678", "Sam");
    let (impostor, _) = speaker(&h, "main", "discord", "2000", "discord:3000");
    let h = downgrade_to_v1_and_reopen(h);

    assert_eq!(
        speaker_ids(&h, "main"),
        [
            ("discord:2000".to_string(), impostor),
            ("discord:5678".to_string(), sam),
        ],
        "each speaker's first alias, no display name, and no owner id"
    );
    assert_eq!(speaker(&h, "main", "discord", "5678", "Sam"), (sam, false));
    let again = speaker(&h, "main", "discord", "2000", "discord:3000");
    assert_eq!(again, (impostor, false));
    let (real, owner) = speaker(&h, "main", "discord", "3000", "Sam");
    assert!(!owner);
    assert_ne!(real, impostor, "a display name didn't become an identity");
}

#[test]
fn an_upgrade_leaves_an_id_a_stranger_held_first_with_them_until_bank_config_takes_it() {
    // Version 1: the owner spoke from discord:1234 before it was configured,
    // so ingest made them a stranger; then bank config added the id to
    // `user`. The upgrade keeps the stranger's id and infers nothing for
    // the owner; the config sent again takes the id over.
    let h = Harness::new();
    bank_without_owner_ids(&h, "late");
    let (stranger, owner) = speaker(&h, "late", "discord", "1234", "Tim");
    assert!(!owner);
    refresh_owner_ids(&h, "late", &["discord:1234"]);
    let h = downgrade_to_v1_and_reopen(h);
    let user = h.user("late");
    assert_ne!(stranger, user);
    let ids = speaker_ids(&h, "late");
    assert_eq!(ids, [("discord:1234".to_string(), stranger)]);
    let again = speaker(&h, "late", "discord", "1234", "Tim");
    assert_eq!(again, (stranger, false));

    let before = takeovers(&h, "late");
    refresh_owner_ids(&h, "late", &["discord:1234"]);
    assert_eq!(speaker(&h, "late", "discord", "1234", "Tim"), (user, true));
    assert_eq!(takeovers(&h, "late"), before + 1, "the takeover is logged");
}

#[test]
fn an_upgrade_does_not_trust_an_alias_of_unknown_origin() {
    // Shape alone proves nothing: an alias on an entity ingest didn't
    // create, or a later alias of one it did, isn't a platform id. Version
    // 1 could hold both; this binary's API can't make them.
    let h = Harness::new();
    let (sam, _) = speaker(&h, "main", "discord", "5678", "Sam");
    let maya = "01a0f000-0000-7000-8000-0000000000aa";
    h.service
        .store()
        .unwrap()
        .connection()
        .execute_batch(&format!(
            "INSERT INTO entities (uuid, bank_id, name, kind, created_at, updated_at)
               SELECT '{maya}', id, 'Maya', 'person', 0, 0 FROM banks WHERE name = 'main';
             INSERT INTO entity_aliases (bank_id, entity_id, alias, created_at)
               SELECT bank_id, id, 'slack:77', 0 FROM entities WHERE uuid = '{maya}';
             INSERT INTO entity_aliases (bank_id, entity_id, alias, created_at)
               SELECT bank_id, id, 'slack:88', 0 FROM entities WHERE uuid = '{sam}';"
        ))
        .unwrap();
    let h = downgrade_to_v1_and_reopen(h);
    let ids: Vec<String> = speaker_ids(&h, "main")
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(ids, ["discord:5678"]);
    let (entity, _) = speaker(&h, "main", "slack", "77", "Maya");
    assert_ne!(entity, maya.parse::<Uuid>().unwrap());
    let (entity, _) = speaker(&h, "main", "slack", "88", "Sam");
    assert_ne!(entity, sam);
}

#[test]
fn bank_config_takes_a_platform_id_over_from_a_stranger_and_logs_it() {
    // The owner's platform id can be added after strangers have spoken
    // (`PUT /v1/banks/{bank}` merges). The stranger who spoke from it is
    // taken over; one who only used it as a display name stays themselves.
    let h = Harness::new();
    bank_without_owner_ids(&h, "late");
    let (named, owner) = speaker(&h, "late", "discord", "2000", "discord:1234");
    assert!(!owner);
    let (stranger, owner) = speaker(&h, "late", "discord", "1234", "Tim");
    assert!(!owner);
    let before = takeovers(&h, "late");

    refresh_owner_ids(&h, "late", &["discord:1234"]);
    let user = h.user("late");
    assert_eq!(speaker(&h, "late", "discord", "1234", "Tim"), (user, true));
    let again = speaker(&h, "late", "discord", "2000", "discord:1234");
    assert_eq!(again, (named, false));
    assert_eq!(takeovers(&h, "late"), before + 1, "the takeover is logged");
    let edits = h.service.show_entity("late", "user").unwrap().edits;
    let takeover = edits.iter().rfind(|edit| edit.kind == "speaker_id_set");
    let details = takeover.unwrap().details.to_string();
    // The edit never names the id, and the stranger's entity stays.
    assert!(!details.contains("1234"), "{details}");
    let entities = h.service.entities("late").unwrap();
    assert!(entities.iter().any(|entity| entity.id == stranger));

    // Every Hermes instance sends the same config on start; repeating it
    // changes nothing and logs nothing.
    refresh_owner_ids(&h, "late", &["discord:1234"]);
    assert_eq!(takeovers(&h, "late"), before + 1);
}

#[test]
fn a_merged_speaker_resolves_to_the_surviving_entity() {
    // A merge keeps the merged entity. A speaker id pointing at it resolves
    // through it.
    let h = Harness::new();
    let (a, _) = speaker(&h, "main", "discord", "5678", "Sam");
    let (b, _) = speaker(&h, "main", "slack", "U77", "Samuel");
    let merge = |from: Uuid, into: Uuid| {
        let (from, into) = (from.to_string(), into.to_string());
        let request = MergeRequest { from, into };
        h.service.merge_entities("main", &request).unwrap();
    };
    merge(a, b);
    assert_eq!(speaker(&h, "main", "discord", "5678", "Sam"), (b, false));

    // Merged into `user` (the owner's second account), the speaker is the
    // owner.
    let user = h.user("main");
    merge(b, user);
    assert_eq!(speaker(&h, "main", "discord", "5678", "Sam"), (user, true));
    assert_eq!(speaker(&h, "main", "slack", "U77", "Samuel"), (user, true));
}
