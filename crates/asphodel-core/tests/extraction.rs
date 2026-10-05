//! Extraction call 1 contracts cover claims, significance, validity windows,
//! entities and used verdicts.
//!
//! These are golden tests against `FakeLlm`: each scripts call 1's reply and
//! checks what's committed through `show_memory`, `show_entity` and
//! `show_source`, or checks the input call 1 is given. Where a claim lands
//! near something stored, call 2 is scripted to label nothing, so every claim
//! that survives the checks in code becomes a new memory. Reconciliation
//! itself is tested in `reconcile.rs`.
//!
//! The `SimulatedClock` stands still unless a test advances it, so a stored
//! time that equals its instant can only have come from the Clock.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

use asphodel_core::Service;
use asphodel_core::clock::{Clock, SimulatedClock};
use asphodel_core::config::Tuning;
use asphodel_core::entities::MergeRequest;
use asphodel_core::extraction::{
    CANDIDATE_MEMORIES, CONTEXT_CHARS, CONTEXT_TURNS, Call1Input, DropReason, Dropped,
    ENTITY_CANDIDATE_CAP, EntityKind, ExtractError, Extracted,
};
use asphodel_core::ingest::{Document, Ingested, TURN_SEPARATOR, Turn, TurnAuthor};
use asphodel_core::inspect::{
    BankOverview, ChunkState, ChunkView, EntitySummary, EntityView, InspectError, MemoryView,
    WindowView,
};
use asphodel_core::models::{
    Embedder, FakeEmbedder, FakeLlm, FakeReranker, LlmClient, LlmError, LlmRequest, LlmResponse,
    ModelError, Models,
};
use asphodel_core::queue::{Failure, Lease, SourceKind};
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{OpenOptions, Store};
use asphodel_core::strength::TimePrecision::{self, Day, Hour, Minute, Month, Year};
use asphodel_core::strength::WorldTime;
use jiff::civil::{Date, DateTime, date};
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use serde_json::{Value, json};
use uuid::Uuid;

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";
const MODEL: &str = "fake-llm";

/// A turn's message time: 19:30 on Thursday 1 October 2026 in Auckland,
/// which is on daylight time (UTC+13).
const T1: &str = "2026-10-01T06:30:00Z";

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
}

/// A local date-time in `zone` as the instant stored for it.
fn local_in(datetime: &str, zone: &str) -> Timestamp {
    datetime
        .parse::<DateTime>()
        .unwrap()
        .to_zoned(TimeZone::get(zone).unwrap())
        .unwrap()
        .timestamp()
}

/// A stored time of `precision`: an instant when `datetime` ends in `Z`,
/// otherwise a local date-time in `TZ`.
fn time(datetime: &str, precision: TimePrecision) -> Option<WorldTime> {
    let at = match datetime.ends_with('Z') {
        true => at(datetime),
        false => local_in(datetime, TZ),
    };
    Some(WorldTime { at, precision })
}

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-extraction-{}-{}",
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

/// A floor for each fake model, so a service opens on the fakes, with
/// `extra` tuning TOML appended.
fn tuning(extra: &str) -> Tuning {
    Tuning::from_toml(&format!(
        "[injection.reranker_floors]\n\"{}\" = 0.0\n\
         [ranking.relevance_scales]\n\"{0}\" = 1.0\n\
         [reconcile.embedding_floors]\n\"{}\" = 0.5\n{extra}",
        FakeReranker::MODEL_ID,
        FakeEmbedder::MODEL_ID,
    ))
    .unwrap()
}

/// The owner is Tim, on Discord as `discord:1234`, and the assistant is
/// Hermes.
fn identity() -> BankIdentity {
    BankIdentity {
        owner_name: Some("Tim".into()),
        owner_platform_ids: vec!["discord:1234".into()],
        assistant_name: Some("Hermes".into()),
        timezone: Some(TZ.into()),
    }
}

/// A service on the fake models with two banks, `main` and `other`. Field
/// order matters: the service drops before the directory it lives in.
struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    /// Turns [`told`] has ingested, which sets their message times.
    seeds: AtomicI64,
    _dir: TestDir,
}

impl Harness {
    fn new() -> Self {
        Self::build(Some(Models::fake()), false, "")
    }

    /// `extra` is more tuning TOML, appended to the floors the fakes need.
    fn with_tuning(extra: &str) -> Self {
        Self::build(Some(Models::fake()), false, extra)
    }

    /// `None` is a service built without models, as `Service::open` gives.
    /// `deterministic_ids` is replay's store option.
    fn build(models: Option<Models>, deterministic_ids: bool, extra: &str) -> Self {
        let dir = TestDir::new();
        let clock = Arc::new(SimulatedClock::new(at(START)));
        let options = OpenOptions {
            deterministic_ids,
            ..OpenOptions::default()
        };
        let store = Store::open(&dir.data(), options, clock.clone()).unwrap();
        let service = match models {
            Some(models) => Service::with_models(clock.clone(), store, tuning(extra), models),
            None => Ok(Service::open(clock.clone(), store, tuning(extra))),
        }
        .unwrap();
        for bank in ["main", "other"] {
            let ids = Models::fake().ids();
            service.ensure_bank(bank, &identity(), &ids).unwrap();
        }
        let seeds = AtomicI64::new(0);
        Self {
            service,
            clock,
            seeds,
            _dir: dir,
        }
    }

    /// Drops the service, releasing the data-dir lock, and opens a new one on
    /// the same data dir and clock: a daemon restart, which runs any pending
    /// migration.
    fn restart(self) -> Self {
        let Harness {
            service,
            clock,
            seeds,
            _dir: dir,
        } = self;
        drop(service);
        let store = Store::open(&dir.data(), OpenOptions::default(), clock.clone()).unwrap();
        let service =
            Service::with_models(clock.clone(), store, tuning(""), Models::fake()).unwrap();
        Self {
            service,
            clock,
            seeds,
            _dir: dir,
        }
    }

    fn advance(&self, hours: i64) {
        self.clock.advance(SignedDuration::from_hours(hours));
    }

    /// Raw SQL, for states the API can't reach.
    fn sql(&self, sql: &str) {
        let store = self.service.store().unwrap();
        store.connection().execute_batch(sql).unwrap();
    }

    fn bank(&self) -> BankOverview {
        let banks = self.service.banks().unwrap();
        banks.into_iter().find(|bank| bank.name == "main").unwrap()
    }

    fn memory(&self, memory: Uuid) -> MemoryView {
        self.service
            .show_memory("main", &memory.to_string())
            .unwrap()
    }

    fn entity(&self, bank: &str, entity: Uuid) -> EntityView {
        self.service.show_entity(bank, &entity.to_string()).unwrap()
    }

    /// Every entity in `bank` called `name`.
    fn named(&self, bank: &str, name: &str) -> Vec<Uuid> {
        let entities = self.service.entities(bank).unwrap();
        entities
            .into_iter()
            .filter(|entity| entity.name == name)
            .map(|entity| entity.id)
            .collect()
    }

    /// The bank's seeded `user` or `assistant`.
    fn seeded(&self, bank: &str, which: &str) -> Uuid {
        let entities = self.service.entities(bank).unwrap();
        entities
            .into_iter()
            .map(|entity| self.entity(bank, entity.id))
            .find(|entity| entity.seeded.as_deref() == Some(which))
            .unwrap()
            .id
    }

    /// A memory's entity links: (entity, surface form).
    fn links(&self, memory: Uuid) -> BTreeSet<(Uuid, Option<String>)> {
        let entities = self.memory(memory).entities;
        entities
            .into_iter()
            .map(|link| (link.id, link.surface_form))
            .collect()
    }

    /// The chunk at `position` of `source`.
    fn chunk(&self, source: Uuid, position: i64) -> ChunkView {
        let detail = self.service.show_source("main", &source.to_string());
        let chunks = detail.unwrap().chunks;
        chunks.into_iter().find(|c| c.position == position).unwrap()
    }

    fn ingest(&self, turn: &Turn) -> Ingested {
        self.service.ingest_turn("main", turn).unwrap()
    }

    /// The owner says `user` in session `s1`, answered `reply`.
    fn say(&self, message_at: &str, user: &str, reply: &str) -> Ingested {
        self.ingest(&turn("s1", message_at, user, reply))
    }

    fn doc(&self, id: &str, text: &str, reference_date: Date) -> Ingested {
        let document = Document {
            document_id: id.into(),
            text: text.into(),
            reference_date,
            reference_date_exact: true,
            timezone: Some(TZ.into()),
        };
        self.service.ingest_document("main", &document).unwrap()
    }
}

/// A memory's accesses, oldest first: (kind, at, turn, source).
fn accesses(memory: &MemoryView) -> Vec<(&str, Timestamp, i64, Option<Uuid>)> {
    memory
        .accesses
        .iter()
        .map(|access| (access.kind.as_str(), access.at, access.turn, access.source))
        .collect()
}

/// A window with no kind-specific field set.
fn window() -> WindowView {
    WindowView {
        valid_from: None,
        valid_until: None,
        until_event: None,
        window_confidence: "high".into(),
        due_at: None,
        volatility: None,
        recurrence: None,
        recurrence_rrule: None,
        recurrence_start: None,
        timezone: TZ.into(),
    }
}

/// [`window`] with `set` applied.
fn kept(set: impl FnOnce(&mut WindowView)) -> WindowView {
    let mut window = window();
    set(&mut window);
    window
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

/// A turn on Discord from `name`, who isn't the owner.
fn discord_turn(name: &str, message_at: &str, user: &str, assistant: &str) -> Turn {
    let author = TurnAuthor {
        id: format!("discord-{name}"),
        name: Some(name.into()),
        is_bot: false,
    };
    Turn {
        author: Some(author),
        platform: Some("discord".into()),
        ..turn(&format!("thread-{name}"), message_at, user, assistant)
    }
}

/// A turn's chunk text: the message, the separator and the reply.
fn turn_text(user: &str, reply: &str) -> String {
    format!("{user}{TURN_SEPARATOR}{reply}")
}

fn lease(h: &Harness, bank: &str) -> Lease {
    h.service
        .claim_chunk(bank)
        .unwrap()
        .expect("a chunk is queued")
}

/// Call 1's input for the head of `main`'s queue. The lease is released
/// when it drops.
fn input(h: &Harness, in_context: &[Uuid]) -> Call1Input {
    h.service
        .call1_input(&lease(h, "main"), in_context)
        .unwrap()
}

/// The lease on chunk `position` of `source`, extracting every chunk of
/// `main` queued before it with a reply that claims nothing.
fn head(h: &Harness, source: Uuid, position: u32) -> Lease {
    loop {
        let lease = lease(h, "main");
        if (lease.source, lease.position) == (source, position) {
            return lease;
        }
        let llm = FakeLlm::scripted(MODEL, vec![reply(vec![], &[])]);
        h.service.extract_chunk(lease, &llm, &[]).unwrap();
    }
}

/// Call 1's input for the first chunk of `source`, as [`head`] reaches it.
fn input_for(h: &Harness, source: Uuid, in_context: &[Uuid]) -> Call1Input {
    let lease = head(h, source, 0);
    h.service.call1_input(&lease, in_context).unwrap()
}

/// Extracts the head of `main`'s queue.
fn run(h: &Harness, llm: &dyn LlmClient, in_context: &[Uuid]) -> Result<Extracted, ExtractError> {
    h.service.extract_chunk(lease(h, "main"), llm, in_context)
}

/// Call 1 answering `reply` and, if the claims land near something stored,
/// call 2 labelling nothing, so each claim stays new.
fn unlabelled(reply: Value) -> FakeLlm {
    FakeLlm::scripted(MODEL, vec![reply, json!({"claims": []})])
}

/// Extracts the head of `main`'s queue with call 1 claiming `claims`.
fn extract(h: &Harness, claims: Vec<Value>) -> Extracted {
    run(h, &unlabelled(reply(claims, &[])), &[]).unwrap()
}

/// Ingests the owner's turn in session `s1` at [`T1`], extracts it with
/// `claims` and returns the new memories in claim order.
fn golden(h: &Harness, user: &str, reply_text: &str, claims: Vec<Value>) -> Vec<Uuid> {
    h.say(T1, user, reply_text);
    extract(h, claims).memories
}

/// What the owner told `bank` before the test: `user` in session `seed`,
/// extracted with the claims `claims` makes from call 1's input.
fn told(
    h: &Harness,
    bank: &str,
    user: &str,
    claims: impl FnOnce(&Call1Input) -> Vec<Value>,
) -> Extracted {
    let seed = h.seeds.fetch_add(1, Ordering::Relaxed);
    let message_at = format!("2026-09-01T{:02}:{:02}:00Z", seed / 60, seed % 60);
    let ingested = h
        .service
        .ingest_turn(bank, &turn("seed", &message_at, user, "Noted."))
        .unwrap();
    let lease = lease(h, bank);
    assert_eq!(lease.source, ingested.source, "the seed is the head");
    let input = h.service.call1_input(&lease, &[]).unwrap();
    let llm = unlabelled(reply(claims(&input), &[]));
    h.service.extract_chunk(lease, &llm, &[]).unwrap()
}

/// A minor fact `content` stored in `bank`.
fn remember(h: &Harness, bank: &str, content: &str) -> Uuid {
    told(h, bank, content, |_| vec![said("fact", content)]).memories[0]
}

/// Forgets `memory` and runs its erase.
fn forget(h: &Harness, memory: Uuid) {
    h.service.forget("main", &[memory.to_string()]).unwrap();
    while h.service.erase_next("main").unwrap().is_some() {}
}

/// The entities a seed proposes, one claim linking each, in order.
fn entities(h: &Harness, bank: &str, names: &[&str]) -> Vec<Uuid> {
    let user = names.join(", ");
    let links: Vec<Value> = names
        .iter()
        .map(|name| new_entity(name, "person", name))
        .collect();
    let content = format!("Tim knows {user}.");
    told(h, bank, &user, |_| {
        vec![claim(&content, "fact", &user).with("entities", json!(links))]
    })
    .entities_created
}

fn handle(input: &Call1Input, entity: Uuid) -> String {
    let candidate = input.candidates.iter().find(|c| c.entity == entity);
    candidate.expect("a candidate").handle.clone()
}

fn memory_handle(input: &Call1Input, memory: Uuid) -> String {
    let shown = input.in_context.iter().find(|m| m.memory == memory);
    shown.expect("in context").handle.clone()
}

/// The candidates' entities, without `user` and `assistant`.
fn found(h: &Harness, input: &Call1Input) -> BTreeSet<Uuid> {
    let always = [h.seeded("main", "user"), h.seeded("main", "assistant")];
    input
        .candidates
        .iter()
        .map(|candidate| candidate.entity)
        .filter(|entity| !always.contains(entity))
        .collect()
}

/// The candidates found when the owner says `message`.
fn found_in(h: &Harness, message: &str) -> BTreeSet<Uuid> {
    let ingested = h.say("2026-10-01T06:45:00Z", message, "Lovely.");
    found(h, &input_for(h, ingested.source, &[]))
}

// Call 1's reply, as `FakeLlm` scripts it.

/// A claim with nothing but its sentence, kind and quote: minor, high window
/// confidence, no times and no entities.
fn claim(content: &str, kind: &str, quote: &str) -> Value {
    json!({
        "content": content, "kind": kind, "quote": quote, "significance": "minor",
        "remember_this": false, "changes_something": false, "window_confidence": "high",
        "valid_from": null, "valid_until": null, "until_event": null, "due_at": null,
        "volatility": null, "recurrence_text": null, "recurrence_rrule": null,
        "recurrence_start": null, "entities": [],
    })
}

/// A claim whose sentence is its quote.
fn said(kind: &str, quote: &str) -> Value {
    claim(quote, kind, quote)
}

/// The owner's message saying each claim's quote in turn.
fn saying(claims: &[Value]) -> String {
    let sentences: Vec<String> = claims
        .iter()
        .map(|claim| format!("{}.", claim["quote"].as_str().unwrap()))
        .collect();
    sentences.join(" ")
}

trait With: Sized {
    fn with(self, key: &str, value: Value) -> Value;

    /// Sets `key` to a time as call 1 gives it.
    fn at(self, key: &str, at: &str, precision: &str) -> Value {
        self.with(key, json!({"at": at, "precision": precision}))
    }

    fn significance(self, level: &str) -> Value {
        self.with("significance", json!(level))
    }
}

impl With for Value {
    fn with(mut self, key: &str, value: Value) -> Value {
        self[key] = value;
        self
    }
}

fn link(handle: &str, surface_form: &str) -> Value {
    json!({"entity": handle, "new_name": null, "new_kind": null, "surface_form": surface_form})
}

fn new_entity(name: &str, kind: &str, surface_form: &str) -> Value {
    json!({"entity": null, "new_name": name, "new_kind": kind, "surface_form": surface_form})
}

fn reply(claims: Vec<Value>, used: &[&str]) -> Value {
    json!({"claims": claims, "used_injected_ids": used})
}

/// Call 1 answering `call1`, then call 2 labelling its first claim `label`
/// against `neighbour`.
fn labelled(
    h: &Harness,
    lease: &Lease,
    call1: Value,
    neighbour: Uuid,
    label: &str,
    in_context: &[Uuid],
) -> FakeLlm {
    let input = h
        .service
        .call2_input(lease, &call1, in_context)
        .unwrap()
        .expect("call 2 runs: the claim is near the neighbour");
    let shown = input.neighbours.iter().find(|n| n.memory == neighbour);
    let call2 = json!({"claims": [{
        "claim": input.claims[0].handle,
        "labels": [{"neighbour": shown.expect("shown to call 2").handle, "label": label}],
    }]});
    FakeLlm::scripted(MODEL, vec![call1, call2])
}

// Context assembly.

#[test]
fn the_speaker_is_given_and_always_a_candidate() {
    let h = Harness::new();
    let owners = h.say(T1, "Dentist tomorrow at 3pm.", "Noted.");
    let sams = h.ingest(&discord_turn(
        "Sam",
        "2026-10-01T06:40:00Z",
        "I'm moving.",
        "Oh!",
    ));
    let tim = h.seeded("main", "user");
    let sam = sams.speaker.unwrap().entity;

    let input = input_for(&h, owners.source, &[]);
    assert_eq!(input.chunk, h.chunk(owners.source, 0).id);
    assert_eq!(input.source_kind, SourceKind::Turn);
    assert_eq!(input.text, turn_text("Dentist tomorrow at 3pm.", "Noted."));
    assert_eq!(input.observed_at, at(T1));
    assert_eq!(input.timezone, TZ);
    assert_eq!(input.reference_date, Some(date(2026, 10, 1)));

    // "Sam" isn't in Sam's text, but the speaker is always a candidate, as
    // are user and assistant.
    let sams_input = input_for(&h, sams.source, &[]);
    for (input, entity, name, owner) in [(input, tim, "Tim", true), (sams_input, sam, "Sam", false)]
    {
        let speaker = input.speaker.as_ref().expect("a turn has a speaker");
        assert_eq!(
            (speaker.entity, speaker.name.as_str(), speaker.owner),
            (entity, name, owner)
        );
        assert_eq!(speaker.handle, handle(&input, entity));
        handle(&input, tim);
        handle(&input, h.seeded("main", "assistant"));
    }
}

#[test]
fn the_latest_earlier_turns_of_the_session_are_context() {
    let h = Harness::with_tuning("[purge]\nsource_horizon_days = 1\n");
    // An earlier turn whose text the sweep has since removed isn't context.
    let swept = h.ingest(&turn("old", "2026-10-01T05:00:00Z", "Swept news.", "Ok."));
    extract(&h, vec![]);
    h.advance(56);
    h.service.run_sweeps().unwrap();
    let detail = h.service.show_source("main", &swept.source.to_string());
    assert!(detail.unwrap().gone.is_some(), "the sweep removed the text");
    h.ingest(&turn("old", "2026-10-03T16:00:00Z", "Kept news.", "Ok."));
    let after_sweep = h.ingest(&turn("old", "2026-10-03T16:05:00Z", "So?", "Yes."));
    assert_eq!(
        input_for(&h, after_sweep.source, &[]).context,
        vec![turn_text("Kept news.", "Ok.")]
    );

    let earlier: Vec<String> = (0..CONTEXT_TURNS + 2)
        .map(|k| {
            let user = format!("Message {k}.");
            h.say(&format!("2026-10-03T17:0{k}:00Z"), &user, "Ok.");
            turn_text(&user, "Ok.")
        })
        .collect();
    // Neither another session, another bank, a forget request nor a later
    // turn is context.
    h.ingest(&turn("s2", "2026-10-03T17:10:00Z", "Other session.", "Ok."));
    let other_bank = turn("s1", "2026-10-03T17:11:00Z", "Other bank.", "Ok.");
    h.service.ingest_turn("other", &other_bank).unwrap();
    h.ingest(&Turn {
        forget_requested: true,
        ..turn("s1", "2026-10-03T17:12:00Z", "Forget my address.", "Done.")
    });
    let current = h.say("2026-10-03T17:30:00Z", "Now.", "Yes.");
    h.say("2026-10-03T17:40:00Z", "Later.", "Ok.");
    assert_eq!(
        input_for(&h, current.source, &[]).context,
        earlier[earlier.len() - CONTEXT_TURNS..]
    );
}

#[test]
fn context_is_clipped_oldest_first_and_keeps_each_turns_own_time() {
    // Each earlier turn is in its own timezone, so none inherits Auckland's
    // date from the current turn: London is on 1 October while the later Los
    // Angeles turn is still on 30 September.
    let earlier = [
        ("2026-09-30T23:00:00Z", "Pacific/Auckland"),
        ("2026-10-01T00:00:00Z", "Europe/London"),
        ("2026-10-01T01:00:00Z", "America/Los_Angeles"),
    ];
    let half = CONTEXT_CHARS / 2;
    // When the newer two leave room the oldest loses its start; when they
    // fill the budget it's dropped, with its time.
    for lengths in [[half, half, half / 2], [half, half, half]] {
        let h = Harness::new();
        let mut passages = Vec::new();
        for (k, ((message_at, zone), length)) in earlier.iter().zip(lengths).enumerate() {
            let padding = length - format!("TURN{k}").len() - TURN_SEPARATOR.len() - "ok".len();
            let user = format!("TURN{k}{}", "x".repeat(padding));
            h.ingest(&Turn {
                timezone: Some((*zone).into()),
                ..turn("s1", message_at, &user, "ok")
            });
            passages.push(turn_text(&user, "ok"));
        }
        let current = h.say(T1, "Remember those occasions.", "Noted.");
        let input = input_for(&h, current.source, &[]);

        let room = CONTEXT_CHARS - lengths[1] - lengths[2];
        let mut expected = passages[1..].to_vec();
        let mut times: Vec<(Timestamp, &str)> = earlier[1..]
            .iter()
            .map(|(message_at, zone)| (at(message_at), *zone))
            .collect();
        if room > 0 {
            let oldest: String = passages[0].chars().skip(lengths[0] - room).collect();
            expected.insert(0, oldest);
            times.insert(0, (at(earlier[0].0), earlier[0].1));
        }
        assert_eq!(input.context, expected, "{lengths:?}");
        let anchors: Vec<(Timestamp, &str)> = input
            .context_times
            .iter()
            .map(|time| (time.observed_at, time.timezone.as_str()))
            .collect();
        assert_eq!(anchors, times, "{lengths:?}");
    }
}

#[test]
fn a_document_chunk_gets_the_text_just_before_it_as_context() {
    let h = Harness::new();
    let tea = remember(&h, "main", "Tim likes tea.");
    let trip = format!("TRIPSTART {}", "We fly out early. ".repeat(40));
    let text = format!("# Trip\n\n{trip}\n\n# Packing\n\nBring the blue tent.\n");
    let ingested = h.doc("notes", &text, date(2026, 9, 28));

    // The first chunk has nothing before it, no speaker and no in-context
    // memories, whatever the caller passes.
    let first = input(&h, &[tea]);
    assert_eq!(first.source_kind, SourceKind::Document);
    assert_eq!(first.reference_date, Some(date(2026, 9, 28)));
    assert!(first.context.is_empty());
    assert!(first.speaker.is_none());
    assert!(first.in_context.is_empty());

    let second = h.chunk(ingested.source, 1);
    let lease = head(&h, ingested.source, 1);
    let input = h.service.call1_input(&lease, &[]).unwrap();
    assert_eq!(input.chunk, second.id);
    let start = usize::try_from(second.start).unwrap();
    let before: String = text.chars().take(start).collect();
    assert_eq!(input.context.len(), 1);
    assert!(!input.context[0].is_empty());
    assert!(before.ends_with(&input.context[0]));
    assert!(
        !input.context[0].contains("TRIPSTART"),
        "only the text just before the chunk"
    );
}

#[test]
fn entities_named_in_the_chunk_or_its_context_are_candidates() {
    let h = Harness::new();
    let made = entities(&h, "main", &["Ana", "Bob", "Sam", "Sammy", "Carol"]);
    let [ana, bob, sam, sammy, _carol] = made[..] else {
        unreachable!()
    };
    let request = MergeRequest {
        from: sammy.to_string(),
        into: sam.to_string(),
    };
    h.service.merge_entities("main", &request).unwrap();
    entities(&h, "other", &["Ana"]);

    h.say("2026-10-01T06:00:00Z", "Bob called.", "What did he say?");
    let current = h.say(T1, "Ana and Sammy are coming over.", "Lovely.");
    let input = input_for(&h, current.source, &[]);

    // Ana from this bank, Bob from the context, and Sammy as Sam, once.
    assert_eq!(found(&h, &input), BTreeSet::from([ana, bob, sam]));
    let sams = input.candidates.iter().filter(|c| c.entity == sam);
    assert_eq!(sams.count(), 1);
    let ana = input.candidates.iter().find(|c| c.entity == ana).unwrap();
    assert_eq!(ana.name, "Ana");
    assert_eq!(ana.kind, EntityKind::Person);
    assert_eq!(ana.aliases, ["Ana"]);
}

#[test]
fn aliases_match_whole_words_in_order_folding_only_latin_diacritics() {
    for (aliases, message, matched) in [
        // "Ana" isn't a word here, the words of "Bob Smith" are out of order,
        // and "Acme Corporation Ltd" isn't all there.
        (
            &["Acme Corp", "Ana", "Bob Smith", "Acme Corporation Ltd"][..],
            "Acme Corp. called about a banana for Smith Bob, and Acme Corporation too.",
            &["Acme Corp"][..],
        ),
        // Latin diacritics fold both ways round, in either normalization form.
        (
            &["Lucía", "Zoe"],
            "Lucia and Zoë came over.",
            &["Lucía", "Zoe"],
        ),
        (&["Lucía"], "Luci\u{301}a called.", &["Lucía"]),
        // A Greek accent and a vowel sign are part of the name, which still
        // matches itself.
        (&["Νίκος"], "Ο Νίκος ήρθε.", &["Νίκος"]),
        (&["किरण"], "किरण आया।", &["किरण"]),
        // A Greek accent and "ø" are part of the letter, so these are
        // different names.
        (&["Νίκος", "Søren"], "Νικος and Soren came.", &[]),
    ] {
        let h = Harness::new();
        let made = entities(&h, "main", aliases);
        let expected: BTreeSet<Uuid> = matched
            .iter()
            .map(|alias| made[aliases.iter().position(|a| a == alias).unwrap()])
            .collect();
        assert_eq!(found_in(&h, message), expected, "{message}");
    }
}

/// Greek "Νίκος" decomposed: iota, then a combining acute.
const NIKOS_DECOMPOSED: &str = "Νι\u{301}κος";
const NIKOS: &str = "Νίκος";
const ZOE_DECOMPOSED: &str = "Ζωη\u{301}";
const ZOE: &str = "Ζωή";

#[test]
fn names_are_stored_composed_and_found_in_either_form() {
    // From the bank's config.
    let h = Harness::new();
    let greek = BankIdentity {
        owner_name: Some(NIKOS_DECOMPOSED.into()),
        assistant_name: Some(ZOE_DECOMPOSED.into()),
        ..identity()
    };
    h.service.ensure_bank_with_models("greek", &greek).unwrap();
    for (which, composed, decomposed) in [
        ("user", NIKOS, NIKOS_DECOMPOSED),
        ("assistant", ZOE, ZOE_DECOMPOSED),
    ] {
        let entity = h.entity("greek", h.seeded("greek", which));
        assert_eq!(entity.name, composed, "{which}");
        assert!(entity.aliases.contains(&composed.to_string()), "{which}");
        assert!(!entity.aliases.contains(&decomposed.to_string()), "{which}");
    }

    // From a speaker's name.
    for (name, said) in [
        (NIKOS_DECOMPOSED, NIKOS),
        (NIKOS_DECOMPOSED, NIKOS_DECOMPOSED),
        (NIKOS, NIKOS_DECOMPOSED),
    ] {
        let h = Harness::new();
        let hello = discord_turn(name, "2026-10-01T05:00:00Z", "Hello.", "Hi.");
        let nikos = h.ingest(&hello).speaker.unwrap().entity;
        assert_eq!(
            found_in(&h, &format!("Ο {said} ήρθε.")),
            BTreeSet::from([nikos]),
            "{name:?} said as {said:?}"
        );
    }

    // From call 1's proposal.
    let h = Harness::new();
    let proposal = json!([new_entity(NIKOS_DECOMPOSED, "person", NIKOS_DECOMPOSED)]);
    let claim = claim("Nikos called Tim.", "event", NIKOS_DECOMPOSED).with("entities", proposal);
    let created = golden(
        &h,
        &format!("{NIKOS_DECOMPOSED} called."),
        "Who?",
        vec![claim],
    );
    let nikos = h.named("main", NIKOS);
    assert_eq!(nikos.len(), 1);
    assert_eq!(h.entity("main", nikos[0]).aliases, [NIKOS]);
    assert_eq!(
        h.links(created[0]),
        BTreeSet::from([(nikos[0], Some(NIKOS.into()))])
    );
    assert_eq!(found_in(&h, "Ο Νίκος ήρθε."), BTreeSet::from([nikos[0]]));
}

#[test]
fn an_upgrade_composes_stored_aliases_and_merges_equivalent_ones() {
    let h = Harness::new();
    let made = entities(&h, "main", &[NIKOS, ZOE]);
    let [nikos, zoe] = made[..] else {
        unreachable!()
    };
    // As a version 2 store could hold them: Nikos with both spellings of its
    // name as aliases, and Zoe known only by a decomposed name, then back to
    // version 2 with one `migrations` row 0 to 2, so reopening runs only the
    // migrations after it.
    h.sql(&format!(
        "INSERT INTO entity_aliases (bank_id, entity_id, alias, created_at)
           SELECT bank_id, id, '{NIKOS_DECOMPOSED}', created_at
           FROM entities WHERE uuid = '{nikos}';
         UPDATE entities SET name = '{ZOE_DECOMPOSED}' WHERE uuid = '{zoe}';
         UPDATE entity_aliases SET alias = '{ZOE_DECOMPOSED}'
           WHERE entity_id = (SELECT id FROM entities WHERE uuid = '{zoe}');
         DELETE FROM migrations;
         INSERT INTO migrations (from_version, to_version, binary_version, started_at,
                                 completed_at)
           VALUES (0, 2, 'v2', 0, 0);
         PRAGMA user_version = 2;"
    ));
    let h = h.restart();

    // One composed alias each; the canonical duplicate is merged away.
    assert_eq!(h.entity("main", nikos).aliases, [NIKOS]);
    let zoe_view = h.entity("main", zoe);
    assert_eq!(zoe_view.name, ZOE);
    assert_eq!(zoe_view.aliases, [ZOE]);
    assert_eq!(
        found_in(&h, "Ο Νίκος και η Ζωή ήρθαν."),
        BTreeSet::from([nikos, zoe])
    );
}

#[test]
fn the_most_linked_entities_are_candidates_and_one_left_out_is_never_reused() {
    let h = Harness::new();
    let names: Vec<String> = (0..ENTITY_CANDIDATE_CAP + 2)
        .map(|k| format!("Name{k:02}"))
        .collect();
    let message = names.join(" ");
    // Claim m links Name(m) onwards, so NameK has K + 1 memories.
    let made = told(&h, "main", &message, |_| {
        (0..names.len())
            .map(|m| {
                let links: Vec<Value> = names[m..]
                    .iter()
                    .map(|name| new_entity(name, "thing", name))
                    .collect();
                claim(&format!("Memory {m}."), "fact", &names[m]).with("entities", json!(links))
            })
            .collect()
    })
    .entities_created;
    let current = h.say(T1, &message, "Quite a list.");

    // The two with the fewest links are left out; user and assistant are on
    // top of the cap.
    let input = input_for(&h, current.source, &[]);
    let most_linked: BTreeSet<Uuid> = made[2..].iter().copied().collect();
    assert_eq!(found(&h, &input), most_linked);

    // Name00 missed the cap, so call 1 never compared it and proposed a new
    // entity. Commit reuses only an entity created after call 1 ran, so this
    // is a second Name00, not the one call 1 never saw.
    let proposal = json!([new_entity("Name00", "thing", "Name00")]);
    let claim = claim("Name00 is on the list.", "fact", "Name00").with("entities", proposal);
    let extracted = extract(&h, vec![claim]);
    let second = extracted.entities_created[0];
    assert_ne!(second, made[0]);
    let name00: BTreeSet<Uuid> = h.named("main", "Name00").into_iter().collect();
    assert_eq!(name00, BTreeSet::from([made[0], second]));
    assert_eq!(
        h.links(extracted.memories[0]),
        BTreeSet::from([(second, Some("Name00".into()))])
    );
}

#[test]
fn a_candidates_examples_are_its_strongest_visible_memories() {
    // Each case lifts one memory of Ana's above her critical and major ones:
    // the trivial jazz used in a reply, a trivial correction inheriting its
    // predecessor's use through `refines`, or an event said long ago whose
    // window just closed. Unlifted, the trivial jazz never makes the cut.
    for case in ["used", "inherited", "closed"] {
        let h = Harness::new();
        let seeded = told(&h, "main", "About Ana.", |_| {
            [
                ("Ana likes jazz.", "trivial"),
                ("Ana lives in Wellington.", "minor"),
                ("Ana is Tim's sister.", "critical"),
                ("Ana is a nurse.", "major"),
                ("Ana's address is 4 Elm St.", "critical"),
                ("Ana is a doctor.", "critical"),
            ]
            .map(|(content, level)| {
                claim(content, "fact", "About Ana")
                    .significance(level)
                    .with("entities", json!([new_entity("Ana", "person", "Ana")]))
            })
            .to_vec()
        });
        let [jazz, _, _, _, address, doctor] = seeded.memories[..] else {
            unreachable!()
        };
        let ana = seeded.entities_created[0];
        // Neither a forgotten nor a retracted memory is an example.
        forget(&h, address);
        h.service.retract("main", &doctor.to_string()).unwrap();

        let use_in_reply = |memory: Uuid| {
            let used = h.ingest(&turn("chat", "2026-10-01T05:00:00Z", "And?", "Yes!"));
            let lease = head(&h, used.source, 0);
            let shown = memory_handle(&h.service.call1_input(&lease, &[memory]).unwrap(), memory);
            let llm = FakeLlm::scripted(MODEL, vec![reply(vec![], &[&shown])]);
            h.service.extract_chunk(lease, &llm, &[memory]).unwrap();
        };
        // The lease on the owner saying `user`, and call 1 claiming `claim`
        // about Ana.
        let about_ana = |message_at: &str, user: &str, claim: Value| {
            let said = h.ingest(&turn("s2", message_at, user, "Noted."));
            let lease = head(&h, said.source, 0);
            let candidate = handle(&h.service.call1_input(&lease, &[]).unwrap(), ana);
            let claim = claim.with("entities", json!([link(&candidate, "Ana")]));
            (lease, reply(vec![claim], &[]))
        };
        let lifted = match case {
            "used" => {
                use_in_reply(jazz);
                "Ana likes jazz."
            }
            "inherited" => {
                let old = remember(&h, "main", "Ana's surname is Ngati.");
                use_in_reply(old);
                let user = "Ana's surname is Ngata, not Ngati.";
                let fix = claim("Ana's surname is Ngata.", "fact", "Ana's surname is Ngata");
                let fix = fix.significance("trivial");
                let (lease, call1) = about_ana("2026-10-01T06:00:00Z", user, fix);
                let llm = labelled(&h, &lease, call1, old, "refines", &[]);
                h.service.extract_chunk(lease, &llm, &[]).unwrap();
                "Ana's surname is Ngata."
            }
            _ => {
                let user = "Ana's exhibition runs until 28 September 2026";
                let event = said("event", user).significance("critical");
                let event = event.at("valid_until", "2026-09-28", "day");
                let (lease, call1) = about_ana("2025-08-27T06:30:00Z", user, event);
                h.service
                    .extract_chunk(lease, &unlabelled(call1), &[])
                    .unwrap();
                user
            }
        };

        let current = h.say(T1, "Ana called.", "How is she?");
        let input = input_for(&h, current.source, &[]);
        let candidate = input.candidates.iter().find(|c| c.entity == ana).unwrap();
        let strongest = [lifted, "Ana is Tim's sister.", "Ana is a nurse."];
        assert_eq!(
            candidate.memories,
            strongest[..CANDIDATE_MEMORIES],
            "{case}"
        );
    }
}

#[test]
fn only_the_banks_visible_in_context_memories_are_given() {
    let h = Harness::new();
    let tea = remember(&h, "main", "Tim likes tea.");
    let address = remember(&h, "main", "Tim's address is 4 Elm St.");
    forget(&h, address);
    let elsewhere = remember(&h, "other", "Tim likes coffee.");
    let unknown = Uuid::from_u128(1);
    let current = h.say(T1, "Tea?", "You like tea, so yes.");

    let input = input_for(&h, current.source, &[tea, address, elsewhere, unknown]);
    assert_eq!(input.in_context.len(), 1);
    assert_eq!(input.in_context[0].memory, tea);
    assert_eq!(input.in_context[0].content, "Tim likes tea.");
}

#[test]
fn the_request_carries_the_input() {
    let h = Harness::new();
    let ana = entities(&h, "main", &["Ana"])[0];
    let tea = remember(&h, "main", "Tim likes tea.");
    h.say("2026-10-01T06:00:00Z", "Morning.", "Morning, Tim.");
    let current = h.say(T1, "Ana wants tea.", "I'll put the kettle on.");

    let lease = head(&h, current.source, 0);
    let input = h.service.call1_input(&lease, &[tea]).unwrap();
    let llm = FakeLlm::scripted(MODEL, vec![reply(vec![], &[])]);
    h.service.extract_chunk(lease, &llm, &[tea]).unwrap();
    let request = &llm.requests()[0];
    let prompt = format!("{}\n{}", request.system, request.user);

    assert!(prompt.contains(&input.text));
    assert!(!input.context.is_empty());
    for context in &input.context {
        assert!(prompt.contains(context.as_str()));
    }
    handle(&input, ana);
    for candidate in &input.candidates {
        assert!(prompt.contains(&candidate.handle));
        assert!(prompt.contains(&candidate.name));
    }
    assert!(prompt.contains(&memory_handle(&input, tea)));
    assert!(prompt.contains("Tim likes tea."));
}

/// `[extraction] guidance` as it might be written, padded, and the text
/// call 1's prompt gets.
const GUIDANCE: &str =
    "[extraction]\nguidance = \"\"\"\n  Skip build logs.\nKeep release dates.  \n\"\"\"\n";
const GUIDANCE_TEXT: &str = "Skip build logs.\nKeep release dates.";

#[test]
fn language_and_guidance_reach_the_request() {
    // Call 1's prompt for one chunk under `tuning`.
    let prompt = |tuning: &str| {
        let h = Harness::with_tuning(tuning);
        h.say(T1, "Me mudo a Lisboa.", "¡Qué bien!");
        let llm = FakeLlm::scripted(MODEL, vec![reply(vec![], &[])]);
        run(&h, &llm, &[]).unwrap();
        let request = &llm.requests()[0];
        format!("{}\n{}", request.system, request.user)
    };
    let plain = prompt("");
    assert!(prompt(GUIDANCE).contains(GUIDANCE_TEXT));
    assert!(!plain.contains(GUIDANCE_TEXT));
    assert!(prompt("[llm]\nlanguage = \"English\"\n").contains("English"));
    assert!(!plain.contains("English"));
}

// Claims.

#[test]
fn each_kind_keeps_only_the_fields_it_can_use() {
    let h = Harness::new();
    let rows = [
        // A fact keeps a stated start but never an end.
        (
            said("fact", "I started at Acme in March 2024")
                .at("valid_from", "2024-03", "month")
                .at("valid_until", "2027", "year"),
            kept(|w| w.valid_from = time("2024-03-01", Month)),
        ),
        // An event keeps its window and precisions.
        (
            said("event", "I'm on holiday from Monday until 12 October")
                .at("valid_from", "2026-10-05", "day")
                .at("valid_until", "2026-10-12", "day"),
            kept(|w| {
                w.valid_from = time("2026-10-05", Day);
                w.valid_until = time("2026-10-12", Day);
            }),
        ),
        (
            said("event", "Dentist tomorrow at 3pm").at("valid_from", "2026-10-02T15:00", "hour"),
            kept(|w| w.valid_from = time("2026-10-02T15:00", Hour)),
        ),
        // A state keeps its volatility and until-event, null when unsure.
        (
            said("state", "I'm chasing a flaky build until the release ships")
                .with("volatility", json!("days"))
                .with("until_event", json!("the release ships")),
            kept(|w| {
                w.volatility = Some("days".into());
                w.until_event = Some("the release ships".into());
            }),
        ),
        (said("state", "Feeling tired today"), window()),
        // A task has a due date, and an end only when an event ends it.
        (
            said("task", "Remind me to renew my passport by 20 October").at(
                "due_at",
                "2026-10-20",
                "day",
            ),
            kept(|w| w.due_at = time("2026-10-20", Day)),
        ),
        (
            said(
                "task",
                "Remind me to take my card to the dentist on 20 October at 2pm",
            )
            .at("due_at", "2026-10-20T14:00", "minute")
            .at("valid_until", "2026-10-20T14:00", "minute"),
            kept(|w| {
                w.due_at = time("2026-10-20T14:00", Minute);
                w.valid_until = time("2026-10-20T14:00", Minute);
            }),
        ),
        // A recurring memory keeps a rule that parses and recurs.
        (
            said("recurring", "I play football every Tuesday at 6pm")
                .with("recurrence_text", json!("every Tuesday at 6pm"))
                .with("recurrence_rrule", json!("FREQ=WEEKLY;BYDAY=TU"))
                .at("recurrence_start", "2026-10-06T18:00", "hour"),
            kept(|w| {
                w.recurrence = Some("every Tuesday at 6pm".into());
                w.recurrence_rrule = Some("FREQ=WEEKLY;BYDAY=TU".into());
                w.recurrence_start = time("2026-10-06T18:00", Hour);
            }),
        ),
        // Otherwise only its text: a rule that doesn't parse, one with no
        // first occurrence, and one with none in the year after the
        // reference date.
        (
            said("recurring", "Bins go out every week")
                .with("recurrence_text", json!("every week"))
                .with("recurrence_rrule", json!("FREQ=FORTNIGHTLY"))
                .at("recurrence_start", "2026-10-05", "day"),
            kept(|w| w.recurrence = Some("every week".into())),
        ),
        (
            said("recurring", "I swim on Fridays")
                .with("recurrence_text", json!("on Fridays"))
                .with("recurrence_rrule", json!("FREQ=WEEKLY;BYDAY=FR")),
            kept(|w| w.recurrence = Some("on Fridays".into())),
        ),
        (
            said("recurring", "We had a reunion every year until 2019")
                .with("recurrence_text", json!("every year until 2019"))
                .with(
                    "recurrence_rrule",
                    json!("FREQ=YEARLY;UNTIL=20190601T000000Z"),
                )
                .at("recurrence_start", "2015-06-01", "day"),
            kept(|w| w.recurrence = Some("every year until 2019".into())),
        ),
        // Fields that belong to another kind are dropped.
        (
            said("fact", "I'm learning Rust").with("volatility", json!("months")),
            window(),
        ),
        (
            said("event", "The launch is on 9 October")
                .at("valid_from", "2026-10-09", "day")
                .at("due_at", "2026-10-09", "day"),
            kept(|w| w.valid_from = time("2026-10-09", Day)),
        ),
        (
            said("state", "I'm in Wellington")
                .with("volatility", json!("hours"))
                .with("recurrence_text", json!("daily"))
                .with("recurrence_rrule", json!("FREQ=DAILY")),
            kept(|w| w.volatility = Some("hours".into())),
        ),
        (
            said("task", "I need to call Mum").with("volatility", json!("days")),
            window(),
        ),
    ];
    let (claims, expected): (Vec<Value>, Vec<WindowView>) = rows.into_iter().unzip();
    let memories = golden(&h, &saying(&claims), "Noted.", claims);
    let windows: Vec<WindowView> = memories.iter().map(|m| h.memory(*m).window).collect();
    assert_eq!(windows, expected);
}

// Relative dates. Call 1 resolves them against the calendar; code turns
// what it gives into instants in the source's timezone and checks weekdays.

#[test]
fn a_time_is_the_start_of_its_unit_in_the_sources_timezone() {
    let plan = |at: &str, precision: &str| {
        claim("Plans.", "event", "Plans").at("valid_from", at, precision)
    };
    // Call 1's start `at` of `precision`, stored at `start` with high or low
    // window confidence.
    let row = |at: &str, precision: TimePrecision, start: &str, confident: bool| {
        let given = plan(at, json!(precision).as_str().unwrap());
        (given, time(start, precision), confident)
    };
    let auckland = vec![
        row("2027", Year, "2027-01-01", true),
        row("2026-11-17", Month, "2026-11-01", true),
        row("2026-11", Month, "2026-11-01", true),
        row("2026-10-03T15:45", Day, "2026-10-03", true),
        row("2026-10-03T15:45", Hour, "2026-10-03T15:00", true),
        row("2026-10-03T15:45", Minute, "2026-10-03T15:45", true),
        // Auckland skips 02:00 to 03:00 on 27 September 2026 and repeats
        // 02:00 to 03:00 on 5 April 2026. A time in the gap moves forward by
        // the gap, a time in the fold takes the earlier offset, and neither
        // lowers window confidence.
        row("2026-09-27T02:30", Minute, "2026-09-26T14:30:00Z", true),
        row("2026-09-27T02:00", Hour, "2026-09-26T14:00:00Z", true),
        row("2026-04-05T02:30", Minute, "2026-04-04T13:30:00Z", true),
        // A time that doesn't parse is dropped with low confidence, and an
        // event left with no start falls back to the day it was said.
        (
            plan("2026-10-05", "day").at("valid_until", "whenever", "day"),
            time("2026-10-05", Day),
            false,
        ),
        row("soonish", Day, "2026-10-01", false),
    ];
    // 13:00 on 1 October in London, already the 2nd in the bank's Auckland.
    // With no start, the event falls back to midnight on the 1st in London.
    let in_london = vec![
        row("2026-10-03", Day, "2026-10-02T23:00:00Z", true),
        (
            claim("Plans.", "event", "Plans"),
            time("2026-09-30T23:00:00Z", Day),
            false,
        ),
    ];

    let h = Harness::new();
    for (zone, message_at, rows) in [
        (TZ, T1, auckland),
        ("Europe/London", "2026-10-01T12:00:00Z", in_london),
    ] {
        h.ingest(&Turn {
            timezone: Some(zone.into()),
            ..turn(zone, message_at, "Plans.", "Busy!")
        });
        let (claims, expected): (Vec<Value>, Vec<(Option<WorldTime>, bool)>) = rows
            .into_iter()
            .map(|(claim, start, confident)| (claim, (start, confident)))
            .unzip();
        let memories = extract(&h, claims).memories;
        let resolved: Vec<(Option<WorldTime>, bool)> = memories
            .iter()
            .map(|memory| {
                let window = h.memory(*memory).window;
                assert_eq!(window.valid_until, None);
                (window.valid_from, window.window_confidence == "high")
            })
            .collect();
        assert_eq!(resolved, expected, "{zone}");
    }
}

#[test]
fn every_weekday_in_the_quote_must_fall_on_one_of_its_dates() {
    let h = Harness::new();
    let on = |quote: &str, day: &str| said("event", quote).at("valid_from", day, "day");
    // 2 October 2026 is a Friday, 5 October a Monday and the 8th a Thursday.
    let rows = [
        (on("Lunch with Ana on Friday", "2026-10-02"), true),
        (on("Drinks on Saturday", "2026-10-02"), false),
        (
            said("task", "Report due Friday").at("due_at", "2026-10-03", "day"),
            false,
        ),
        // The Monday start matches, but nothing falls on the Friday.
        (
            on("away Monday through Friday", "2026-10-05").at("valid_until", "2026-10-08", "day"),
            false,
        ),
        (
            on("off Monday through Friday", "2026-10-05").at("valid_until", "2026-10-09", "day"),
            true,
        ),
        // Call 1's own low confidence stands.
        (
            on("Coffee with Ana on Friday", "2026-10-02").with("window_confidence", json!("low")),
            false,
        ),
    ];
    let (claims, expected): (Vec<Value>, Vec<bool>) = rows.into_iter().unzip();
    let memories = golden(&h, &saying(&claims), "Busy week.", claims);
    let windows: Vec<WindowView> = memories.iter().map(|m| h.memory(*m).window).collect();
    let confident: Vec<bool> = windows
        .iter()
        .map(|window| window.window_confidence == "high")
        .collect();
    assert_eq!(confident, expected);
    // The window itself is kept.
    assert_eq!(windows[1].valid_from, time("2026-10-02", Day));
}

#[test]
fn a_commit_records_where_and_when_each_memory_came_from() {
    let h = Harness::new();
    let text = "Yesterday I met Ana at the market.";
    let ingested = h.doc("diary", text, date(2026, 9, 28));
    let ingested_at = h.clock.now();
    h.advance(2);

    let content = "Tim met Ana at the market on 27 September 2026.";
    let claim = claim(content, "event", "I met Ana at the market");
    let extracted = extract(&h, vec![claim.at("valid_from", "2026-09-27", "day")]);
    let memory = h.memory(extracted.memories[0]);
    let chunk = h.chunk(ingested.source, 0);
    assert_eq!(extracted.chunk, chunk.id);
    assert_eq!(memory.sentence, content);
    assert_eq!(
        (memory.source.source, memory.source.chunk),
        (ingested.source, chunk.id)
    );
    assert_eq!((memory.source.start, memory.source.end), (10, 33));
    // A document's memory is observed at its reference date, and its
    // created access is the ingest, never the extraction, so a backdated
    // document doesn't arrive faded.
    assert_eq!(memory.observed_at, local_in("2026-09-28T00:00", TZ));
    assert_eq!(memory.window.valid_from, time("2026-09-27", Day));
    assert_eq!(
        accesses(&memory),
        vec![("created", ingested_at, 0, Some(ingested.source))]
    );
    assert_eq!(memory.created_at, h.clock.now());

    // The chunk left the queue and its lease was released: the bank's next
    // chunk can be claimed.
    assert_eq!((chunk.state, chunk.error_count), (ChunkState::Extracted, 0));
    assert_eq!(h.service.queue_depth("main").unwrap(), 0);
    h.say(T1, "More.", "Ok.");
    assert!(h.service.claim_chunk("main").unwrap().is_some());
}

#[test]
fn remember_this_keeps_only_what_the_owner_said() {
    // What's ingested, the quote, the level call 1 gives and the owner's.
    type Row = (
        fn(&Harness) -> Ingested,
        &'static str,
        &'static str,
        Option<&'static str>,
    );
    let rows: [Row; 4] = [
        // The owner's keep sits beside the level extraction gave, so unkeep
        // can hand it back.
        (
            |h| h.say(T1, "Remember this: passport ends in 42.", "Got it."),
            "passport ends in 42",
            "notable",
            Some("kept"),
        ),
        // Another speaker's is capped at critical.
        (
            |h| {
                h.ingest(&discord_turn(
                    "Sam",
                    T1,
                    "Remember this: I'm allergic to nuts.",
                    "Ok.",
                ))
            },
            "I'm allergic to nuts",
            "critical",
            None,
        ),
        // A document's is ignored.
        (
            |h| h.doc("notes", "Remember this: gate code 4321.", date(2026, 9, 30)),
            "gate code 4321",
            "major",
            None,
        ),
        // The quote's first occurrence is the owner's question, but the
        // "remember this" is the reply's, so it can't be shown to be the
        // owner's.
        (
            |h| {
                let reply = "Remember this: Ana's birthday is 4 May.";
                h.say(T1, "Is \"Ana's birthday is 4 May\" correct?", reply)
            },
            "Ana's birthday is 4 May",
            "notable",
            None,
        ),
    ];
    for (source, quote, level, owner) in rows {
        let h = Harness::new();
        source(&h);
        let claim = said("fact", quote)
            .significance(level)
            .with("remember_this", json!(true));
        let memory = extract(&h, vec![claim]).memories[0];
        let significance = h.memory(memory).significance;
        let got = (
            significance.extracted.as_str(),
            significance.owner.as_deref(),
        );
        assert_eq!(got, (level, owner), "{quote}");
    }
}

// Quotes.

#[test]
fn a_claim_is_kept_only_with_a_quote_from_the_chunk_located_in_characters() {
    let h = Harness::new();
    h.say("2026-10-01T06:00:00Z", "Morning.", "Are you still at Acme?");
    let user = "Café ☕ with Ana. Yes, still there.";
    let current = h.say(T1, user, "Lunch with Ana, then lunch.");
    head(&h, current.source, 0);

    // "Yes" after "Are you still at Acme?" is the user's claim, written out
    // in full from the context but quoted from the answer. A quote from the
    // context, or from nowhere, drops the claim.
    let extracted = extract(
        &h,
        vec![
            claim("Tim is having coffee with Ana.", "fact", "with Ana"),
            claim("Tim is having lunch.", "fact", "then lunch"),
            claim("Tim still works at Acme.", "fact", "Yes, still there"),
            claim("Tim works at Acme.", "fact", "Are you still at Acme?"),
            claim("Tim likes Acme.", "fact", "I love working at Acme"),
            claim("Tim is at Acme.", "fact", ""),
            claim("  ", "fact", "still there"),
        ],
    );
    // The first occurrence, counted in characters, not bytes; the reply
    // starts after the message and the separator.
    let spans: Vec<(i64, i64)> = extracted
        .memories
        .iter()
        .map(|m| {
            let source = h.memory(*m).source;
            (source.start, source.end)
        })
        .collect();
    assert_eq!(spans, vec![(7, 15), (52, 62), (17, 33)]);
    let dropped = |claim, reason| Dropped { claim, reason };
    assert_eq!(
        extracted.dropped,
        vec![
            dropped(3, DropReason::QuoteNotFound),
            dropped(4, DropReason::QuoteNotFound),
            dropped(5, DropReason::QuoteNotFound),
            dropped(6, DropReason::EmptyContent),
        ]
    );
    assert_eq!(h.bank().memories.live, 3);
}

#[test]
fn an_assistant_task_needs_a_due_date_or_an_until_event() {
    let h = Harness::new();
    h.say(
        T1,
        "Can you send me the report by Friday? And look into the backup sometime.",
        "I'll send you the report by Friday. I'll look into the backup.",
    );
    let task = |content: &str, quote: &str| claim(content, "task", quote);
    let extracted = extract(
        &h,
        vec![
            task(
                "Hermes will send the report.",
                "I'll send you the report by Friday",
            )
            .at("due_at", "2026-10-02", "day"),
            task(
                "Hermes will look into the backup.",
                "I'll look into the backup",
            ),
            task(
                "Hermes will restore the backup.",
                "I'll look into the backup",
            )
            .with("until_event", json!("the backup is restored")),
            // The user's own undated task stays.
            task(
                "Tim wants the backup looked into.",
                "look into the backup sometime",
            ),
        ],
    );
    assert_eq!(extracted.memories.len(), 3);
    let undated = Dropped {
        claim: 1,
        reason: DropReason::AssistantTaskUndated,
    };
    assert_eq!(extracted.dropped, vec![undated]);
}

// Entities.

#[test]
fn links_keep_their_surface_form_and_a_new_one_becomes_a_logged_alias() {
    let h = Harness::new();
    let ana = entities(&h, "main", &["Ana"])[0];
    let tim = h.seeded("main", "user");
    let tim_aliases = h.entity("main", tim).aliases;
    let alias_edits = |h: &Harness| {
        let edits = h.entity("main", ana).edits;
        edits.iter().filter(|e| e.kind == "alias_added").count()
    };
    let edits_before = alias_edits(&h);
    let user = "Annie (that's Ana) and I loved Lisbon. Lisbon was hot. Zed came by.";
    h.say(T1, user, "Lovely.");
    let input = input(&h, &[]);
    let ana_handle = handle(&input, ana);
    let me = input.speaker.as_ref().unwrap().handle.clone();
    let lisbon = new_entity("Lisbon", "place", "Lisbon");
    let loved = json!([link(&ana_handle, "Annie"), link(&me, "I"), lisbon]);
    let hot = json!([lisbon, link(&ana_handle, "Ana")]);
    // A link that names no entity is dropped.
    let nobody = json!({"entity": null, "new_name": null, "new_kind": null, "surface_form": "Zed"});

    let extracted = extract(
        &h,
        vec![
            claim(
                "Ana and Tim loved Lisbon.",
                "event",
                "Annie (that's Ana) and I loved Lisbon",
            )
            .with("entities", loved),
            claim("Lisbon was hot.", "event", "Lisbon was hot").with("entities", hot),
            claim("Zed visited Tim.", "event", "Zed came by")
                .with("entities", json!([link("e999", "Zed"), nobody])),
        ],
    );
    // A proposed entity is created once per reply, in its bank only.
    let lisbon = h.named("main", "Lisbon");
    assert_eq!(lisbon.len(), 1);
    assert_eq!(extracted.entities_created, lisbon);
    assert!(h.named("other", "Lisbon").is_empty());
    let lisbon = lisbon[0];
    let [loved, hot, zed] = extracted.memories[..] else {
        unreachable!()
    };
    assert_eq!(
        h.links(loved),
        BTreeSet::from([
            (ana, Some("Annie".into())),
            (tim, Some("I".into())),
            (lisbon, Some("Lisbon".into())),
        ])
    );
    assert_eq!(
        h.links(hot),
        BTreeSet::from([(lisbon, Some("Lisbon".into())), (ana, Some("Ana".into()))])
    );
    assert!(h.links(zed).is_empty());

    // A new surface form becomes an alias in a logged edit, so a mislink can
    // be undone. A known one and a pronoun don't.
    assert_eq!(h.entity("main", ana).aliases, ["Ana", "Annie"]);
    assert_eq!(alias_edits(&h), edits_before + 1);
    assert_eq!(h.entity("main", tim).aliases, tim_aliases);
    let lisbon = h.entity("main", lisbon);
    assert_eq!(lisbon.aliases, ["Lisbon"]);
    assert_eq!(lisbon.kind, "place");
}

/// Call 1 through `llm`, running `during` before the first call, as another
/// writer to the bank would while call 1 is in flight.
struct Meanwhile<'a> {
    h: &'a Harness,
    during: fn(&Harness),
    llm: FakeLlm,
}

impl LlmClient for Meanwhile<'_> {
    fn model(&self) -> &str {
        MODEL
    }

    fn complete(&self, request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        if self.llm.requests().is_empty() {
            (self.during)(self.h);
        }
        self.llm.complete(request)
    }
}

#[test]
fn entities_that_change_while_call_1_runs_are_resolved_at_commit() {
    let h = Harness::new();
    let made = entities(&h, "main", &["Sam", "Samuel"]);
    let [sam, samuel] = made[..] else {
        unreachable!()
    };
    let current = h.say(T1, "Sam is moving. Kiri was there.", "Where to?");
    let lease = head(&h, current.source, 0);
    let sam_handle = handle(&h.service.call1_input(&lease, &[]).unwrap(), sam);
    let llm = Meanwhile {
        h: &h,
        // Kiri speaks for the first time, and Sam is merged into Samuel.
        during: |h| {
            h.ingest(&discord_turn(
                "Kiri",
                "2026-10-01T06:35:00Z",
                "Kia ora.",
                "Hi.",
            ));
            let request = MergeRequest {
                from: h.named("main", "Sam")[0].to_string(),
                into: h.named("main", "Samuel")[0].to_string(),
            };
            h.service.merge_entities("main", &request).unwrap();
        },
        llm: unlabelled(reply(
            vec![
                claim("Sam is moving.", "event", "Sam is moving")
                    .with("entities", json!([link(&sam_handle, "Sam")])),
                claim("Kiri was there.", "event", "Kiri was there")
                    .with("entities", json!([new_entity("Kiri", "person", "Kiri")])),
            ],
            &[],
        )),
    };
    let extracted = h.service.extract_chunk(lease, &llm, &[]).unwrap();

    // Commit links a known entity merged since as the survivor, and repeats
    // the exact alias lookup to link an entity created since rather than a
    // duplicate.
    let kiri = h.named("main", "Kiri");
    assert_eq!(kiri.len(), 1);
    assert!(extracted.entities_created.is_empty());
    assert_eq!(
        h.links(extracted.memories[0]),
        BTreeSet::from([(samuel, Some("Sam".into()))])
    );
    assert_eq!(
        h.links(extracted.memories[1]),
        BTreeSet::from([(kiri[0], Some("Kiri".into()))])
    );
}

// Accesses.

#[test]
fn used_verdicts_write_one_access_per_memory_per_turn() {
    let h = Harness::new();
    let tea = remember(&h, "main", "Tim likes tea.");
    let coffee = remember(&h, "main", "Tim hates coffee.");
    let cake = remember(&h, "main", "Tim likes cake.");
    let reply_text = "Tea and cake, as you like them.";
    let current = h.say(T1, "Tea and cake? I like cake.", reply_text);
    let ingested_at = h.clock.now();
    let turn_number = h.bank().turns;
    h.advance(1);

    let in_context = [tea, coffee, cake];
    let lease = head(&h, current.source, 0);
    let input = h.service.call1_input(&lease, &in_context).unwrap();
    let [tea_handle, cake_handle] = [tea, cake].map(|memory| memory_handle(&input, memory));
    // Call 2 confirms cake, a stronger access in the turn than its used.
    let call1 = reply(
        vec![claim("Tim likes cake.", "fact", "I like cake")],
        &[&tea_handle, &tea_handle, "m999", &cake_handle],
    );
    let llm = labelled(&h, &lease, call1, cake, "confirmed", &in_context);
    let extracted = h.service.extract_chunk(lease, &llm, &in_context).unwrap();

    assert_eq!(extracted.used, vec![tea, cake]);
    assert!(extracted.memories.is_empty());
    let tea = h.memory(tea);
    assert_eq!(
        accesses(&tea)[1..],
        [("used", ingested_at, turn_number, Some(current.source))]
    );
    let cake = h.memory(cake);
    let in_turn: Vec<&str> = accesses(&cake)
        .into_iter()
        .filter(|access| access.2 == turn_number)
        .map(|access| access.0)
        .collect();
    assert_eq!(in_turn, vec!["confirmed"]);
    assert_eq!(h.memory(coffee).accesses.len(), 1);
}

#[test]
fn accesses_carry_the_turn_number_their_source_was_ingested_at() {
    let h = Harness::new();
    // Everything is ingested before anything is extracted, so the bank's
    // counter has moved on by the time each chunk is.
    let one = turn("s1", "2026-10-01T06:00:00Z", "One.", "Ok.");
    let mut quotes = BTreeMap::new();
    quotes.insert(h.ingest(&one).source, "One");
    h.ingest(&Turn {
        forget_requested: true,
        ..turn("s1", "2026-10-01T06:05:00Z", "Forget my address.", "Done.")
    });
    // A duplicate stores nothing and doesn't count.
    h.ingest(&one);
    quotes.insert(h.say("2026-10-01T06:10:00Z", "Two.", "Ok.").source, "Two");
    quotes.insert(h.doc("notes", "Doc.", date(2026, 9, 30)).source, "Doc");
    quotes.insert(
        h.say("2026-10-01T06:20:00Z", "Three.", "Ok.").source,
        "Three",
    );
    assert_eq!(h.bank().turns, 4);

    let mut turns = BTreeMap::new();
    while let Some(lease) = h.service.claim_chunk("main").unwrap() {
        let quote = quotes.get(&lease.source).copied();
        let claims = quote
            .map(|quote| vec![claim(&format!("Tim said {quote}."), "fact", quote)])
            .unwrap_or_default();
        let llm = unlabelled(reply(claims, &[]));
        let extracted = h.service.extract_chunk(lease, &llm, &[]).unwrap();
        if let Some(quote) = quote {
            let memory = h.memory(extracted.memories[0]);
            turns.insert(quote, memory.accesses[0].turn);
        }
    }
    // The forget request is turn 2. A document takes the counter as it
    // stood when it was ingested.
    assert_eq!(
        turns,
        BTreeMap::from([("One", 1), ("Two", 3), ("Doc", 3), ("Three", 4)])
    );
}

// Failure. A failed step leaves the chunk retryable.

/// An embedder that returns short vectors while `short` is set.
#[derive(Default)]
struct FlakyEmbedder {
    short: AtomicBool,
}

impl Embedder for FlakyEmbedder {
    fn model_id(&self) -> &str {
        FakeEmbedder::MODEL_ID
    }

    fn dimensions(&self) -> usize {
        FakeEmbedder.dimensions()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, ModelError> {
        if self.short.load(Ordering::Relaxed) {
            return Ok(texts.iter().map(|_| vec![1.0; 8]).collect());
        }
        FakeEmbedder.embed(texts)
    }
}

/// What a reader can see change: the bank's counts and its entities.
fn visible(h: &Harness) -> (BankOverview, Vec<EntitySummary>) {
    (h.bank(), h.service.entities("main").unwrap())
}

/// The state and error count of the first chunk of `source`.
fn attempts(h: &Harness, source: Uuid) -> (ChunkState, u32) {
    let chunk = h.chunk(source, 0);
    (chunk.state, chunk.error_count)
}

#[test]
fn a_failed_step_writes_nothing_and_counts_the_attempt() {
    for step in ["call 1", "embedding", "commit"] {
        let embedder = Arc::new(FlakyEmbedder::default());
        let models = Models {
            embedder: embedder.clone(),
            reranker: Arc::new(FakeReranker),
        };
        let h = Harness::build(Some(models), false, "");
        let tea = remember(&h, "main", "Tim likes tea.");
        let ingested = h.say(T1, "I'm moving to Lisbon.", "Exciting!");
        let input = input(&h, &[tea]);
        // A claim that would create an entity and a used verdict, so a
        // partial commit would show.
        let busy = || {
            let lisbon = json!([new_entity("Lisbon", "place", "Lisbon")]);
            let claim = claim("Tim is moving to Lisbon.", "event", "I'm moving to Lisbon");
            let used = memory_handle(&input, tea);
            unlabelled(reply(vec![claim.with("entities", lisbon)], &[&used]))
        };
        let before = (visible(&h), h.memory(tea).accesses);

        let llm = match step {
            "call 1" => FakeLlm::failing(MODEL, || LlmError::Status { status: 502 }),
            "embedding" => {
                embedder.short.store(true, Ordering::Relaxed);
                busy()
            }
            // Every access insert fails, so the commit fails after its
            // memories, vectors, entities and links are written.
            _ => {
                h.sql(
                    "CREATE TRIGGER break_accesses BEFORE INSERT ON accesses
                     BEGIN SELECT RAISE(ABORT, 'scripted commit failure'); END",
                );
                busy()
            }
        };
        let error = run(&h, &llm, &[tea]).unwrap_err();
        assert!(
            matches!(
                (step, &error),
                ("call 1", ExtractError::Call1 { .. })
                    | ("embedding", ExtractError::Embedding { .. })
                    | ("commit", _)
            ),
            "{step}: {error:?}"
        );
        let retry = Some(Failure::Retry { error_count: 1 });
        assert_eq!(error.failure(), retry, "{step}");
        assert_eq!((visible(&h), h.memory(tea).accesses), before, "{step}");
        let queued = (ChunkState::Queued, 1);
        assert_eq!(attempts(&h, ingested.source), queued, "{step}");

        // Once the fault is gone, the retry commits everything once.
        embedder.short.store(false, Ordering::Relaxed);
        h.sql("DROP TRIGGER IF EXISTS break_accesses");
        let extracted = run(&h, &busy(), &[tea]).unwrap();
        assert_eq!(extracted.memories.len(), 1, "{step}");
        assert_eq!(h.named("main", "Lisbon").len(), 1, "{step}");
        assert_eq!(h.memory(tea).accesses.len(), 2, "{step}");
    }
}

#[test]
fn an_invalid_reply_writes_nothing_and_counts_the_attempt() {
    let tea = |key: &str, value: Value| {
        let claim = claim("Tim likes tea.", "fact", "I like tea").with(key, value);
        reply(vec![claim], &[])
    };
    let invalid = [
        json!({"facts": []}),
        tea("kind", json!("opinion")),
        tea("window_confidence", json!("medium")),
        tea("entities", json!([new_entity("Tea", "drink", "tea")])),
        // Nothing above critical: keeping is the owner's call.
        tea("significance", json!("kept")),
        tea("significance", json!(0.95)),
        tea("significance", json!(1.0)),
    ];
    for reply in invalid {
        let h = Harness::new();
        let ingested = h.say(T1, "I like tea.", "Noted.");
        let before = visible(&h);
        let llm = FakeLlm::scripted(MODEL, vec![reply.clone()]);
        let error = run(&h, &llm, &[]).unwrap_err();
        let retry = Failure::Retry { error_count: 1 };
        assert!(
            matches!(&error, ExtractError::InvalidReply { failure, .. } if *failure == retry),
            "{reply}: {error:?}"
        );
        assert_eq!(visible(&h), before, "{reply}");
        let queued = (ChunkState::Queued, 1);
        assert_eq!(attempts(&h, ingested.source), queued, "{reply}");
    }
}

#[test]
fn without_a_usable_llm_or_models_the_chunk_waits_uncounted() {
    type MakeError = fn() -> LlmError;
    let held: [MakeError; 3] = [
        || LlmError::UsageLimited {
            resets_at: "2026-10-01T12:00:00Z".parse().unwrap(),
        },
        || LlmError::LoginRequired,
        || LlmError::NotConfigured {
            missing: "llm.model",
        },
    ];
    let tea = reply(vec![claim("Tim likes tea.", "fact", "I like tea")], &[]);
    let without_models = (
        Harness::build(None, false, ""),
        FakeLlm::scripted(MODEL, vec![tea]),
    );
    let cases = held
        .map(|make| (Harness::new(), FakeLlm::failing(MODEL, make)))
        .into_iter()
        .chain([without_models]);
    for (h, llm) in cases {
        let ingested = h.say(T1, "I like tea.", "Noted.");
        let error = run(&h, &llm, &[]).unwrap_err();
        match error {
            ExtractError::Held { .. } => {}
            ExtractError::NoModels => assert!(llm.requests().is_empty()),
            _ => panic!("{error:?}"),
        }
        assert_eq!(error.failure(), None);
        assert_eq!(attempts(&h, ingested.source), (ChunkState::Queued, 0));
        assert_eq!(h.bank().memories.live, 0);
        assert_eq!(lease(&h, "main").chunk, h.chunk(ingested.source, 0).id);
    }
}

// Deterministic entity ids. In replay an entity's id derives from its
// creating chunk and the key commit dedups proposals on: the proposed name
// composed to NFC, trimmed and lowercased.

/// The entities each chunk of a two-chunk document creates in a fresh
/// deterministic store, when chunk 0 proposes `first` and chunk 1 `second`,
/// every proposal under the surface form "Anna".
fn deterministic_ids(first: &[&str], second: &[&str]) -> [Vec<Uuid>; 2] {
    let h = Harness::build(Some(Models::fake()), true, "");
    let text = "# Work\nAnna runs the team.\n\n# School\nAnna teaches maths.";
    let ingested = h.doc("notes", text, date(2026, 9, 28));
    assert_eq!(ingested.chunks_queued, 2);
    [first, second].map(|names| {
        let proposals: Vec<Value> = names
            .iter()
            .map(|name| new_entity(name, "person", "Anna"))
            .collect();
        let claim = claim("Anna is mentioned.", "fact", "Anna").with("entities", json!(proposals));
        extract(&h, vec![claim]).entities_created
    })
}

#[test]
fn deterministic_entity_ids_are_reproducible_and_keyed_on_the_proposal() {
    let ids = deterministic_ids(&["Anna"], &["Anna"]);
    assert_eq!(deterministic_ids(&["Anna"], &["Anna"]), ids);
    assert_eq!(ids[0].len(), 1);
    assert_eq!(
        ids[1].len(),
        1,
        "commit never reuses the first chunk's Anna"
    );
    assert_ne!(ids[0], ids[1]);
    // The second chunk's id is the same whether or not the first's exists.
    assert_eq!(deterministic_ids(&[], &["Anna"])[1], ids[1]);
    // Two people proposed under one surface form are two ids.
    let [two, _] = deterministic_ids(&["Anna Smith", "Anna Jones"], &[]);
    assert_eq!(two.len(), 2);
    assert_ne!(two[0], two[1]);

    // Composed, decomposed, padded and in capitals: one key, so one id, and
    // within one reply one entity.
    let variants = ["Zélie", "Ze\u{301}lie", "  Zélie\t", " ZE\u{301}LIE "];
    let [zelie, _] = deterministic_ids(&variants[..1], &[]);
    for variant in variants {
        assert_eq!(deterministic_ids(&[variant], &[])[0], zelie, "{variant:?}");
    }
    assert_eq!(deterministic_ids(&variants, &[])[0], zelie);
}

// The gap between preparing and committing a chunk.

const LISBON: &str = "Tim lives in Lisbon.";

/// The commit plans against the store it writes to: a claim whose neighbour
/// the sweep purged after the chunk was prepared is new, as if the
/// neighbour had never been stored, and nothing fails. In production the
/// housekeeping task runs the sweep on another thread while the extraction
/// worker holds a prepared chunk.
#[test]
fn a_neighbour_purged_between_prepare_and_commit_leaves_the_claim_new() {
    for (label, user, content, message_at) in [
        // A restatement that call 2 absorbed into Lisbon can't leave its
        // access and passage there.
        (
            "mentioned_again",
            "I live in Lisbon, still.",
            LISBON,
            "2026-10-01T06:40:00Z",
        ),
        // An older claim Lisbon would have ended is created as it is.
        (
            "ends",
            "I lived in Berlin.",
            "Tim lives in Berlin.",
            "2026-08-01T06:30:00Z",
        ),
    ] {
        let h = Harness::with_tuning("[purge]\ndelta = 0.0\n");
        let lisbon = remember(&h, "main", LISBON);
        let current = h.say(message_at, user, "Noted.");
        let lease = head(&h, current.source, 0);
        let call1 = reply(vec![claim(content, "fact", user)], &[]);
        let llm = labelled(&h, &lease, call1, lisbon, label, &[]);
        let prepared = h.service.prepare_extraction(lease, &llm, &[]).unwrap();

        // Years later, Lisbon has faded past the purge line.
        h.advance(24 * 365 * 3);
        h.service.run_sweeps().unwrap();
        assert!(
            matches!(
                h.service.show_memory("main", &lisbon.to_string()),
                Err(InspectError::UnknownMemory)
            ),
            "{label}: the sweep purged Lisbon"
        );

        let extracted = h.service.commit_extraction(prepared).expect("it commits");
        assert_eq!(extracted.memories.len(), 1, "{label}");
        let memory = h.memory(extracted.memories[0]);
        assert_eq!(memory.sentence, content, "{label}");
        assert_eq!(memory.chain.ended_by, None, "{label}");
        let committed = (ChunkState::Extracted, 0);
        assert_eq!(attempts(&h, current.source), committed, "{label}");
    }
}
