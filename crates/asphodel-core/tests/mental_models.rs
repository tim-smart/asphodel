//! Agenda, mental models and the system prompt block, including refresh
//! scheduling and memory precedence. A mental model is a cache over
//! memories: reading or injecting it never counts as an access.
//!
//! These tests drive the service on a `SimulatedClock` and refresh models
//! with `FakeLlm`, scripted as the calls a refresh makes: a plan, only for
//! a model whose question has no built-in or stored plan, then a write of
//! the whole answer as sections of prose and the memories it cites. Every
//! memory comes from extraction: the owner says it in a turn, call 1 (a
//! scripted `FakeLlm`) finds the claim, and call 2's labels make
//! retractions, endings and refinements. `keep`, `retract`, `forget` and
//! the owner's model edits do the rest.
//!
//! The API under test is `asphodel_core::mental_models`,
//! `asphodel_core::agenda`, `asphodel_core::system_prompt` and the
//! `Service` methods over them.
//!
//! Every service here runs on a `SimulatedClock` stopped at 20:00 on
//! Thursday 1 October 2026 in Auckland (UTC+13) unless a test advances it.
//! The next 04:00 there is 15:00 UTC the same day, and the next local
//! midnight is 11:00 UTC. The tuning the tests rely on is set in
//! [`Harness::build`], not taken from the defaults.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use asphodel_core::agenda::Agenda;
use asphodel_core::clock::Sleeper;
use asphodel_core::config::{AgendaTuning, ConfigError, MentalModelsTuning};
use asphodel_core::constants::Significance;
use asphodel_core::extraction::{Call1Input, Extracted};
use asphodel_core::ingest::Turn;
use asphodel_core::inspect::{BankOverview, MemoryView};
use asphodel_core::mental_models::{
    Applied, FailureKind, Model, ModelEdit, ModelError, ModelSpec, Outcome, PLAN_TEMPLATE,
    RefreshInput, Refreshes, WRITE_TEMPLATE,
};
use asphodel_core::models::{
    Embedder, FakeEmbedder, FakeLlm, FakeReranker, LlmClient, LlmError, LlmGate, LlmRequest,
    LlmResponse, LlmRetry, ModelError as EmbedError, Models, Reranker, RetryPolicy,
};
use asphodel_core::operations::{Audit, AuditList, RecallRow};
use asphodel_core::retrieval::{Prefetch, PrefetchRequest, Taken, estimate_tokens};
use asphodel_core::store::bank::{BankIdentity, PROFILE_NAME};
use asphodel_core::store::{OpenOptions, Store};
use asphodel_core::strength::{Kind, TimePrecision, WorldTime};
use asphodel_core::system_prompt::{Block, Preview};
use asphodel_core::{Service, SimulatedClock, Tuning};
use jiff::civil::{DateTime, Time};
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};
use serde_json::{Value, json};
use uuid::Uuid;

// Fixtures

const START: &str = "2026-10-01T07:00:00Z";
const TZ: &str = "Pacific/Auckland";
const BANK: &str = "main";
const MODEL: &str = "fake-llm";

/// The next 04:00 in Auckland after [`START`], when the daily sweep runs.
const SWEEP: &str = "2026-10-01T15:00:00Z";

/// When seeded memories were said, unless a test says otherwise.
const EARLIER: &str = "2026-09-01T00:00:00Z";

/// Long enough ago that a trivial memory said then is below τ.
const LONG_AGO: &str = "2021-01-01T00:00:00Z";

const BERLIN: &str = "Tim lives in Berlin.";
const MOVED: &str = "Tim moved out of Berlin and now lives in Lisbon.";
const MAYA: &str = "Tim's daughter is called Maya.";
const MIA: &str = "Tim's daughter is called Mia.";
const JAPAN: &str = "Tim is going to Japan in 2027.";
const TOKYO: &str = "Tim is going to Tokyo, Japan in April 2027.";
const CAT: &str = "Tim's cat is called Miso.";
const TEA: &str = "Tim likes green tea.";
const CAT_ENTRY: &str = "Tim has a cat called Miso.";
const DENTIST: &str = "Tim's dentist appointment is on 5 October.";
const CONCERT: &str = "Tim has a concert on 9 October.";
const HAIRCUT: &str = "Tim has a haircut on 3 October 2026.";
const RATES: &str = "Tim must pay the rates by 4 October 2026.";

fn at(text: &str) -> Timestamp {
    text.parse().unwrap()
}

/// A local date-time in `TZ` as the instant stored for it.
fn local(datetime: &str) -> Timestamp {
    let (datetime, tz): (DateTime, _) = (datetime.parse().unwrap(), TimeZone::get(TZ).unwrap());
    datetime.to_zoned(tz).unwrap().timestamp()
}

fn minutes(n: i64) -> SignedDuration {
    SignedDuration::from_mins(n)
}

fn secs(n: i64) -> SignedDuration {
    SignedDuration::from_secs(n)
}

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("asphodel-models-{}-{n}", std::process::id());
        let path = std::env::temp_dir().join(name);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    tuning: Tuning,
    _dir: TestDir,
}

fn open(dir: &TestDir, clock: &Arc<SimulatedClock>, tuning: Tuning, models: Models) -> Service {
    let store = Store::open(&dir.0, OpenOptions::default(), clock.clone()).unwrap();
    Service::with_models(clock.clone(), store, tuning, models).unwrap()
}

impl Harness {
    fn new() -> Self {
        Self::with(|_| {})
    }

    /// `tune` adjusts the tuning after the harness has set its own.
    fn with(tune: impl FnOnce(&mut Tuning)) -> Self {
        Self::build(tune, Models::fake(), Vec::new()).0
    }

    /// A harness whose bank already holds `faded`, made trivial and said
    /// at [`LONG_AGO`], with the clock there: below τ by [`START`].
    fn with_faded(tune: impl FnOnce(&mut Tuning), faded: Vec<Value>) -> (Self, Vec<Uuid>) {
        Self::build(tune, Models::fake(), faded)
    }

    fn build(
        tune: impl FnOnce(&mut Tuning),
        models: Models,
        faded: Vec<Value>,
    ) -> (Self, Vec<Uuid>) {
        let mut tuning = Tuning::default();
        let floors = &mut tuning.injection.reranker_floors;
        floors.insert(FakeReranker::MODEL_ID.into(), 1.0);
        set_scale(&mut tuning, 1.0);
        let floors = &mut tuning.reconcile.embedding_floors;
        floors.insert(FakeEmbedder::MODEL_ID.into(), 0.5);
        tuning.mental_models = MentalModelsTuning {
            budget: 800,
            profile_max_tokens: 500,
            trigger_level: Significance::Notable,
            refresh_debounce_minutes: 5,
            refresh_max_delay_minutes: 30,
            sweep_time: Time::constant(4, 0, 0, 0),
            max_facets: 6,
            facet_budget: 20,
            input_budget: 60,
            input_budget_with_cited: 70,
        };
        tuning.agenda = AgendaTuning {
            horizon_days: 7,
            overdue_days: 30,
            undated_days: 30,
            dated_lines: 15,
            routines: 4,
            undated_tasks: 5,
            update_budget: 200,
        };
        tuning.sessions.mapping_expiry_days = 30;
        tune(&mut tuning);
        let dir = TestDir::new();
        let seeding = if faded.is_empty() { START } else { LONG_AGO };
        let clock = Arc::new(SimulatedClock::new(at(seeding)));
        let service = open(&dir, &clock, tuning.clone(), models);
        let identity = BankIdentity {
            owner_name: Some("Tim".into()),
            assistant_name: Some("Hermes".into()),
            timezone: Some(TZ.into()),
            ..BankIdentity::default()
        };
        service.ensure_bank_with_models(BANK, &identity).unwrap();
        let h = Self {
            service,
            clock,
            tuning,
            _dir: dir,
        };
        if faded.is_empty() {
            return (h, Vec::new());
        }
        let trivial = faded.into_iter().map(|c| c.level("trivial")).collect();
        let faded = h.seed_all(BANK, at(LONG_AGO), trivial);
        // The daemon starts at START, so its first daily sweep is the next
        // one after it.
        h.set(at(START));
        (h.restart(), faded)
    }

    /// The daemon restarting: the service and its in-memory state (sessions,
    /// the block cache, pending debounces) go; the store stays.
    fn restart(self) -> Self {
        let models = self.service.models().unwrap().clone();
        drop(self.service);
        let service = open(&self._dir, &self.clock, self.tuning.clone(), models);
        Self { service, ..self }
    }

    fn now(&self) -> Timestamp {
        self.service.now()
    }

    fn advance(&self, by: SignedDuration) {
        self.clock.advance(by);
    }

    fn set(&self, to: Timestamp) {
        self.clock.set(to);
    }

    /// The owner says `claims` in one turn of `bank` at `said`, with the
    /// clock where it is, and the turn is extracted: call 1 finds the claims,
    /// and call 2, if it runs, labels nothing. Returns the new memories in
    /// claim order.
    fn seed_all(&self, bank: &str, said: Timestamp, claims: Vec<Value>) -> Vec<Uuid> {
        self.ingest_claims(bank, said, &claims);
        let count = claims.len();
        let call1 = json!({"claims": claims, "used_injected_ids": []});
        let llm = FakeLlm::scripted(MODEL, vec![call1, json!({"claims": []})]);
        let extracted = self.service.extract_next(bank, &llm).unwrap();
        let extracted = extracted.expect("the turn was queued");
        assert_eq!(extracted.memories.len(), count, "{:?}", extracted.dropped);
        extracted.memories
    }

    /// One memory said at [`EARLIER`].
    fn seed(&self, claim: Value) -> Uuid {
        self.seed_at(at(EARLIER), claim)
    }

    fn seed_at(&self, said: Timestamp, claim: Value) -> Uuid {
        self.seed_all(BANK, said, vec![claim])[0]
    }

    /// One memory said a minute ago.
    fn says(&self, claim: Value) -> Uuid {
        self.seed_at(self.now() - minutes(1), claim)
    }

    /// As [`Harness::seed_at`], with call 2 labelling the claim `label` on
    /// `neighbour`. `None` when the label absorbed it into the neighbour.
    fn seed_changing(
        &self,
        said: Timestamp,
        claim: Value,
        neighbour: Uuid,
        label: &str,
    ) -> Option<Uuid> {
        self.ingest_claims(BANK, said, std::slice::from_ref(&claim));
        let call1 = json!({"claims": [claim], "used_injected_ids": []});
        let call2 = {
            let lease = self.service.claim_chunk(BANK).unwrap().expect("queued");
            let input = self.service.call2_input(&lease, &call1, &[]).unwrap();
            let input = input.expect("call 2 runs");
            let neighbour = input.neighbours.iter().find(|n| n.memory == neighbour);
            let neighbour = neighbour.expect("the memory is a neighbour");
            json!({"claims": [{
                "claim": input.claims[0].handle,
                "labels": [{"neighbour": neighbour.handle, "label": label}],
            }]})
        };
        let llm = FakeLlm::scripted(MODEL, vec![call1, call2]);
        let extracted = self.service.extract_next(BANK, &llm).unwrap();
        let memories = extracted.expect("the turn was queued").memories;
        memories.first().copied()
    }

    /// As [`Harness::says`], with call 2 labelling the claim `label` on
    /// `neighbour`: `retracts`, `ends` or `refines`.
    fn says_changing(&self, claim: Value, neighbour: Uuid, label: &str) -> Uuid {
        self.seed_changing(self.now() - minutes(1), claim, neighbour, label)
            .expect("a new memory")
    }

    /// A turn of its own session, so no two collide.
    fn ingest_claims(&self, bank: &str, said: Timestamp, claims: &[Value]) {
        static SESSIONS: AtomicU64 = AtomicU64::new(0);
        let session = format!("seed-{}", SESSIONS.fetch_add(1, Ordering::Relaxed));
        let quotes: Vec<&str> = claims
            .iter()
            .map(|c| c["quote"].as_str().unwrap())
            .collect();
        let turn = turn(&session, said, &quotes.join(" "));
        self.service.ingest_turn(bank, &turn).unwrap();
    }

    fn keep(&self, memory: Uuid) {
        self.service.keep(BANK, &[memory.to_string()]).unwrap();
    }

    fn profile(&self) -> Model {
        self.model(PROFILE_NAME)
    }

    fn model(&self, name: &str) -> Model {
        let models = self.service.list_models(BANK).unwrap();
        let model = models.into_iter().find(|model| model.name == name);
        model.unwrap_or_else(|| panic!("no model {name}"))
    }

    /// The owner's edit of `name`, as `model edit` takes it.
    fn edit(&self, name: &str, edit: Value) -> Result<Model, ModelError> {
        let edit: ModelEdit = serde_json::from_value(edit).unwrap();
        self.service.edit_model(BANK, name, &edit)
    }

    fn input(&self, name: &str) -> RefreshInput {
        self.service.refresh_input(BANK, name).unwrap()
    }

    fn refresh(&self, name: &str, llm: &dyn LlmClient, force: bool) -> Outcome {
        self.service.refresh_model(BANK, name, llm, force).unwrap()
    }

    /// Forces a refresh of `name` whose write is `reply` made from its
    /// input, and returns what it applied. A model that plans must have its
    /// plan already ([`Harness::plan`]).
    fn refresh_with(&self, name: &str, reply: impl FnOnce(&RefreshInput) -> Value) -> Applied {
        let llm = FakeLlm::scripted(MODEL, vec![reply(&self.input(name))]);
        applied(self.refresh(name, &llm, true))
    }

    /// As [`Harness::refresh_with`], writing `sections`, each a heading and
    /// its sentences as `(text, cites)`: a section's sentences join into
    /// its paragraph, and the write cites every memory any sentence does.
    fn rewrite(&self, name: &str, sections: &[Section<'_>]) -> Applied {
        self.refresh_with(name, |input| {
            let mut cited = Vec::new();
            let mut paragraphs = Vec::new();
            for (heading, sentences) in sections {
                let texts: Vec<&str> = sentences.iter().map(|(text, _)| *text).collect();
                paragraphs.push((*heading, texts.join(" ")));
                for memory in sentences.iter().flat_map(|(_, cites)| *cites) {
                    let handle = handle(input, *memory);
                    if !cited.contains(&handle) {
                        cited.push(handle);
                    }
                }
            }
            let paragraphs: Vec<(&str, &str)> =
                paragraphs.iter().map(|(h, t)| (*h, t.as_str())).collect();
            written(&paragraphs, &cited)
        })
    }

    /// As [`Harness::rewrite`], with one section, [`SECTION`].
    fn refresh_adding(&self, name: &str, entries: &[(&str, &[Uuid])]) -> Applied {
        self.rewrite(name, &[(SECTION, entries)])
    }

    /// Plans `name` with one facet per `(heading, query)`: a forced refresh
    /// whose plan call returns them and whose write, if there's anything to
    /// write from, is [`quiet`]. Returns the plan request.
    fn plan(&self, name: &str, facets: &[(&str, &str)]) -> LlmRequest {
        let llm = FakeLlm::scripted(MODEL, vec![planned(facets), quiet()]);
        applied(self.refresh(name, &llm, true));
        calls(&llm, PLAN_TEMPLATE)
            .pop()
            .expect("the refresh planned")
    }

    /// Gives `name` `question`, planned as `facets`.
    fn ask(&self, name: &str, question: &str, facets: &[(&str, &str)]) {
        self.edit(name, json!({"question": question})).unwrap();
        self.plan(name, facets);
    }

    /// The recall log's `refresh` rows, newest first.
    fn refresh_recalls(&self) -> Vec<RecallRow> {
        let audit = self.service.audit(BANK, AuditList::Recalls, None).unwrap();
        let Audit::Recalls { recalls } = audit else {
            panic!("recalls");
        };
        recalls
            .into_iter()
            .filter(|r| r.kind == "refresh")
            .collect()
    }

    fn profile_adding(&self, entries: &[(&str, &[Uuid])]) -> Applied {
        self.refresh_adding(PROFILE_NAME, entries)
    }

    /// Runs every refresh due now with `llm`.
    fn tick(&self, llm: &dyn LlmClient) -> Refreshes {
        self.service.run_refreshes(llm).unwrap()
    }

    /// How many refreshes `status` counts as failed, and what it says
    /// needs attention.
    fn failures(&self) -> (usize, Vec<String>) {
        let status = self.service.status().unwrap();
        (status.banks[BANK].failed_refreshes, status.attention)
    }

    fn block(&self, session: Option<&str>) -> Block {
        self.service.system_prompt(BANK, session).unwrap()
    }

    fn agenda(&self) -> Agenda {
        self.service.agenda(BANK).unwrap()
    }

    fn in_context(&self, session: &str) -> Vec<Uuid> {
        self.service.in_context(BANK, session).unwrap()
    }

    fn prefetch(&self, session: &str, query: &str, block: Option<Uuid>) -> Prefetch {
        let request = PrefetchRequest {
            session_id: session.into(),
            query: query.into(),
            previous_query: None,
            previous_reply: None,
            block_id: block,
        };
        self.service.prefetch(BANK, &request).unwrap()
    }

    /// The owner's next turn in `session`, committing what the prefetch
    /// `recall_id` names held, when given. Each says something new, so
    /// none is taken for a repeat.
    fn sync(&self, session: &str, recall_id: Option<Uuid>) {
        static TURNS: AtomicU64 = AtomicU64::new(0);
        let user = format!("Turn {}.", TURNS.fetch_add(1, Ordering::Relaxed));
        let turn = Turn {
            recall_id: recall_id.map(|id| id.to_string()),
            ..turn(session, self.now() - minutes(1), &user)
        };
        self.service.ingest_turn(BANK, &turn).unwrap();
    }

    /// A prefetch that injects nothing, so its text is the session's agenda
    /// update alone: empty when there's none, and within its budget.
    fn agenda_update(&self, session: &str) -> Prefetch {
        let prefetch = self.prefetch(session, "hello there", None);
        assert!(prefetch.injected.is_empty(), "{}", prefetch.text);
        let budget = self.tuning.agenda.update_budget as usize;
        let tokens = estimate_tokens(&prefetch.text);
        assert!(tokens <= budget, "{tokens} > {budget}:\n{}", prefetch.text);
        prefetch
    }

    /// Prefetches and commits `session`'s agenda updates until there's
    /// none, returning what each listed. Each lists something new and
    /// nothing the session already holds.
    fn drain_updates(&self, session: &str) -> Vec<Vec<Uuid>> {
        let mut updates = Vec::new();
        loop {
            let held = self.in_context(session);
            let update = self.agenda_update(session);
            if update.text.is_empty() {
                return updates;
            }
            for memory in &held {
                let sentence = self.show(*memory).sentence;
                assert!(!update.text.contains(&sentence), "{}", update.text);
            }
            self.sync(session, Some(update.recall_id));
            let mut listed = self.in_context(session);
            listed.retain(|memory| !held.contains(memory));
            assert!(!listed.is_empty(), "nothing new:\n{}", update.text);
            updates.push(listed);
        }
    }

    /// `memory`'s accesses of `kind`, its own and inherited.
    fn accesses(&self, memory: Uuid, kind: &str) -> usize {
        let accesses = self.show(memory).accesses;
        accesses.iter().filter(|a| a.kind == kind).count()
    }

    fn show(&self, memory: Uuid) -> MemoryView {
        self.service.show_memory(BANK, &memory.to_string()).unwrap()
    }

    fn overview(&self) -> BankOverview {
        let banks = self.service.banks().unwrap();
        banks.into_iter().find(|b| b.name == BANK).unwrap()
    }
}

fn set_scale(tuning: &mut Tuning, scale: f64) {
    let scales = &mut tuning.ranking.relevance_scales;
    scales.insert(FakeReranker::MODEL_ID.into(), scale);
}

/// The owner's turn on the CLI: no author.
fn turn(session: &str, message_at: Timestamp, user: &str) -> Turn {
    Turn {
        session_id: session.into(),
        message_at,
        timezone: Some(TZ.into()),
        user_text: user.into(),
        assistant_text: "Noted.".into(),
        author: None,
        platform: Some("cli".into()),
        recall_id: None,
        forget_requested: false,
    }
}

// Call 1's claims.

/// The sentence is also the quote, as the owner said it.
fn claim(content: &str, kind: &str, significance: &str) -> Value {
    json!({
        "content": content,
        "kind": kind,
        "quote": content,
        "significance": significance,
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
    })
}

/// A notable fact.
fn fact(content: &str) -> Value {
    claim(content, "fact", "notable")
}

fn state(content: &str, volatility: &str) -> Value {
    claim(content, "state", "notable").with("volatility", json!(volatility))
}

/// A local date as call 1 gives it, to the day.
fn day(date: &str) -> Value {
    json!({"at": date, "precision": "day"})
}

fn month(date: &str) -> Value {
    json!({"at": date, "precision": "month"})
}

/// An event on the local date `on`.
fn event(content: &str, on: &str) -> Value {
    claim(content, "event", "notable").with("valid_from", day(on))
}

/// An open task with no due date.
fn task(content: &str) -> Value {
    claim(content, "task", "notable")
}

/// An open task due on the local date `due`.
fn task_due(content: &str, due: &str) -> Value {
    task(content).with("due_at", day(due))
}

/// A weekly memory on Tuesdays, first occurring on 6 January 2026.
fn weekly(content: &str) -> Value {
    recurring(content, Some("FREQ=WEEKLY;BYDAY=TU"), "2026-01-06")
}

/// A recurring memory with `rrule`, first occurring on the local date
/// `start`.
fn recurring(content: &str, rrule: Option<&str>, start: &str) -> Value {
    let memory = claim(content, "recurring", "notable")
        .with("recurrence_text", json!(content))
        .with("recurrence_rrule", json!(rrule));
    match rrule {
        Some(_) => memory.with("recurrence_start", day(start)),
        None => memory,
    }
}

trait With {
    fn with(self, key: &str, value: Value) -> Value;

    fn level(self, significance: &str) -> Value
    where
        Self: Sized,
    {
        self.with("significance", json!(significance))
    }
}

impl With for Value {
    fn with(mut self, key: &str, value: Value) -> Value {
        self[key] = value;
        self
    }
}

// The refresh replies.

/// The section [`Harness::refresh_adding`] writes its sentences under.
const SECTION: &str = "About Tim";

/// A section of a scripted write: its heading, and each sentence's text
/// with the memories it cites.
type Section<'a> = (&'a str, &'a [(&'a str, &'a [Uuid])]);

/// A plan: one facet per `(heading, query)`.
fn planned(facets: &[(&str, &str)]) -> Value {
    let facets = facets
        .iter()
        .map(|(h, q)| json!({"heading": h, "query": q}));
    json!({"facets": facets.collect::<Vec<_>>()})
}

/// A write: the whole answer, one section per `(heading, text)`, citing
/// `cites`.
fn written(sections: &[(&str, &str)], cites: &[String]) -> Value {
    let sections = sections
        .iter()
        .map(|(heading, text)| json!({"heading": heading, "text": text}));
    json!({"sections": sections.collect::<Vec<_>>(), "cites": cites})
}

/// What [`quiet`] writes.
const QUIET: &str = "Tim is known here.";

/// A write with one sentence, citing the first memory listed. Any refresh
/// that calls the LLM lists at least one.
fn quiet() -> Value {
    written(&[(SECTION, QUIET)], &["m1".into()])
}

fn applied(outcome: Outcome) -> Applied {
    match outcome {
        Outcome::Applied(applied) => applied,
        other => panic!("the refresh didn't apply: {other:?}"),
    }
}

/// An LLM that answers every write with [`quiet`].
fn quiet_llm(calls: usize) -> FakeLlm {
    FakeLlm::scripted(MODEL, vec![quiet(); calls])
}

/// The requests `llm` was sent with `template`, in order.
fn calls(llm: &FakeLlm, template: &str) -> Vec<LlmRequest> {
    let requests = llm.requests().into_iter();
    requests.filter(|r| r.template.name == template).collect()
}

fn writes(llm: &FakeLlm) -> usize {
    calls(llm, WRITE_TEMPLATE).len()
}

/// The line after `heading`'s own line in a rendered block: the section's
/// paragraph. A heading may be marked up, as `### People`.
fn paragraph<'a>(text: &'a str, heading: &str) -> Option<&'a str> {
    let mut lines = text.lines();
    let heading_line = |line: &&str| line.trim_start_matches('#').trim() == heading;
    lines.by_ref().find(heading_line)?;
    lines.next()
}

fn handle(input: &RefreshInput, memory: Uuid) -> String {
    input
        .memories
        .iter()
        .find(|m| m.memory == memory)
        .unwrap_or_else(|| panic!("{memory} is in the refresh input"))
        .handle
        .clone()
}

fn handles(input: &RefreshInput, memories: &[Uuid]) -> Vec<String> {
    memories.iter().map(|m| handle(input, *m)).collect()
}

fn inputs(input: &RefreshInput) -> BTreeSet<Uuid> {
    input.memories.iter().map(|m| m.memory).collect()
}

fn answer(model: &Model) -> &str {
    model.answer.as_deref().unwrap_or_default()
}

fn cites(model: &Model) -> BTreeSet<Uuid> {
    model.cites.iter().copied().collect()
}

fn plans_model(max_tokens: u32) -> ModelSpec {
    ModelSpec {
        name: "Plans".into(),
        question: "Where is Tim going and when?".into(),
        kinds: vec![Kind::Event],
        entity: None,
        min_volatility: None,
        max_tokens,
        enabled: true,
    }
}

// Refresh triggers and scheduling

#[test]
fn a_notable_memory_triggers_a_refresh_five_minutes_later_and_at_most_every_thirty() {
    let h = Harness::new();
    let said = h.now();
    h.says(fact(TEA));

    h.advance(minutes(4) + secs(59));
    let llm = quiet_llm(2);
    let early = h.tick(&llm);
    assert!(early.ran.is_empty());
    assert_eq!(writes(&llm), 0);
    assert_eq!(early.next_due, Some(said + minutes(5)));

    let refreshed = said + minutes(5);
    h.set(refreshed);
    let ran = h.tick(&llm);
    assert_eq!(ran.ran.len(), 1);
    assert_eq!(ran.ran[0].model, PROFILE_NAME);
    assert_eq!(writes(&llm), 1);
    assert_eq!(h.profile().last_refreshed_at, Some(refreshed));

    // The debounce alone would run the next one at +6 minutes.
    h.advance(minutes(1));
    h.says(fact("Tim keeps bees."));
    h.set(refreshed + minutes(6));
    let waiting = h.tick(&llm);
    assert!(waiting.ran.is_empty());
    assert_eq!(waiting.next_due, Some(refreshed + minutes(30)));
    h.set(refreshed + minutes(30));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    assert_eq!(writes(&llm), 2);
}

#[test]
fn each_trigger_pushes_the_refresh_back_but_never_past_thirty_minutes() {
    // A long conversation that keeps adding notable facts never goes quiet,
    // so the debounce is capped 30 minutes after the first trigger.
    let h = Harness::new();
    let first = h.now();
    let llm = quiet_llm(1);
    let facts = [
        "Tim plays the cello.",
        "Tim keeps bees.",
        "Tim grows chillies.",
        "Tim restores old radios.",
        "Tim speaks Portuguese.",
        "Tim runs on Sundays.",
        "Tim volunteers at the library.",
        "Tim collects maps.",
    ];
    for (n, content) in facts.into_iter().enumerate() {
        h.set(first + minutes(4 * n as i64));
        h.says(fact(content));
        h.advance(minutes(1));
        assert!(h.tick(&llm).ran.is_empty(), "refreshed after trigger {n}");
    }
    // The last trigger was at +28 minutes, so the debounce alone would wait
    // until +33.
    h.set(first + minutes(30) - secs(1));
    assert!(h.tick(&llm).ran.is_empty());
    h.set(first + minutes(30));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    assert_eq!(writes(&llm), 1);
}

#[test]
fn a_memory_below_the_trigger_level_waits_for_the_daily_sweep() {
    // The sweep runs at 04:00 bank-local.
    let h = Harness::new();
    h.says(claim(TEA, "fact", "minor"));
    h.advance(minutes(5));
    let llm = quiet_llm(2);
    let refreshes = h.tick(&llm);
    assert!(refreshes.ran.is_empty());
    assert_eq!(refreshes.next_due, Some(at(SWEEP)));
    h.set(at(SWEEP) - secs(1));
    assert!(h.tick(&llm).ran.is_empty());

    // The sweep lets minor additions in.
    h.set(at(SWEEP));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    assert_eq!(writes(&llm), 1);
    assert!(llm.requests()[0].user.contains(TEA));
    // The next one is 04:00 the next local day.
    let next = h.tick(&llm).next_due;
    assert_eq!(next, Some(at(SWEEP) + SignedDuration::from_hours(24)));
}

#[test]
fn a_memory_the_models_filters_leave_out_triggers_nothing() {
    // The profile takes facts and states of volatility weeks or slower.
    let h = Harness::new();
    h.says(claim("Tim went to the cinema.", "event", "major"));
    h.says(state("Tim is tired.", "days").level("major"));
    h.advance(minutes(5));
    assert!(h.tick(&quiet_llm(1)).ran.is_empty());

    h.says(state("Tim is training for a marathon.", "months"));
    h.advance(minutes(5));
    assert_eq!(h.tick(&quiet_llm(1)).ran.len(), 1);
}

#[test]
fn an_owner_edit_triggers_a_refresh_that_isnt_skipped() {
    // The question and the size are part of what a refresh compares, so the
    // same memories still get an LLM call. A new question is planned first.
    let h = Harness::new();
    h.seed(fact(TEA));
    applied(h.refresh(PROFILE_NAME, &quiet_llm(1), true));
    let question = "What does Tim like to drink?";
    h.edit(PROFILE_NAME, json!({"question": question})).unwrap();

    h.advance(minutes(30));
    let llm = FakeLlm::scripted(MODEL, vec![planned(&[("Drinks", question)]), quiet()]);
    let ran = h.tick(&llm);
    assert_eq!(ran.ran.len(), 1);
    assert!(matches!(ran.ran[0].outcome, Outcome::Applied(_)));
    assert!(calls(&llm, PLAN_TEMPLATE)[0].user.contains(question));
    assert_eq!(writes(&llm), 1);

    h.edit(PROFILE_NAME, json!({"max_tokens": 400})).unwrap();
    applied(h.refresh(PROFILE_NAME, &quiet_llm(1), false));
}

#[test]
fn a_failed_refresh_is_retried_after_thirty_minutes_not_at_the_next_trigger() {
    let h = Harness::new();
    h.says(fact(TEA));
    h.advance(minutes(5));
    let failed_at = h.now();
    let down = FakeLlm::failing(MODEL, || LlmError::Transport {
        reason: "connection refused".into(),
    });
    let ran = h.tick(&down);
    assert_eq!(ran.ran[0].outcome, Outcome::Failed(FailureKind::Llm));
    let profile = h.profile();
    assert_eq!(profile.last_error, Some(FailureKind::Llm));
    assert_eq!(profile.last_error_at, Some(failed_at));
    assert_eq!(profile.last_refreshed_at, None);
    // `status` counts it and says it needs attention.
    let (failed, attention) = h.failures();
    assert_eq!((failed, attention.len()), (1, 1), "{attention:?}");

    h.advance(minutes(1));
    h.says(fact("Tim keeps bees."));
    h.set(failed_at + minutes(6));
    let llm = quiet_llm(1);
    let waiting = h.tick(&llm);
    assert!(waiting.ran.is_empty());
    assert_eq!(waiting.next_due, Some(failed_at + minutes(30)));

    h.set(failed_at + minutes(30));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    let profile = h.profile();
    assert_eq!(profile.last_error, None);
    assert_eq!(profile.last_refreshed_at, Some(failed_at + minutes(30)));
    assert_eq!(h.failures(), (0, vec![]));
}

/// A [`Sleeper`] that returns at once.
struct NoSleep;

impl Sleeper for NoSleep {
    fn sleep(&self, _: Duration) {}
}

#[test]
fn a_transient_error_retried_within_the_call_keeps_the_plan_and_counts_nothing() {
    // The daemon retries a blip inside the call, so a write that fails
    // twice and then answers doesn't throw the plan away or start the
    // thirty-minute wait.
    let h = Harness::new();
    h.seed(fact(TEA));
    applied(h.refresh(PROFILE_NAME, &quiet_llm(1), true));
    let question = "What does Tim like to drink?";
    h.edit(PROFILE_NAME, json!({"question": question})).unwrap();
    h.advance(minutes(30));

    let script = json!([
        {"reply": planned(&[("Drinks", question)])},
        {"fail": "status", "status": 503},
        {"fail": "transport"},
        {"reply": quiet()},
    ]);
    let inner = Arc::new(FakeLlm::from_script(MODEL, &script.to_string()).unwrap());
    let policy = RetryPolicy::daemon();
    let llm = LlmRetry::new(inner.clone(), policy, h.clock.clone(), Arc::new(NoSleep));
    let ran = h.tick(&llm);
    assert_eq!(ran.ran.len(), 1);
    assert!(matches!(ran.ran[0].outcome, Outcome::Applied(_)));
    assert_eq!(calls(&inner, PLAN_TEMPLATE).len(), 1);
    assert_eq!(writes(&inner), 3);
    assert_eq!(h.profile().last_error, None);
    assert_eq!(h.profile().last_refreshed_at, Some(h.now()));
    assert_eq!(h.failures(), (0, vec![]));
}

#[test]
fn a_write_the_request_may_have_failed_is_a_failed_refresh_at_once() {
    // A timeout or a 500 may be the refresh's own doing, so it isn't
    // retried within the call: the refresh fails as it always has and
    // waits thirty minutes.
    for fault in [
        json!({"fail": "timeout"}),
        json!({"fail": "status", "status": 500}),
    ] {
        let h = Harness::new();
        h.says(fact(TEA));
        h.advance(minutes(5));
        let script = json!([fault.clone(), {"reply": quiet()}]);
        let inner = Arc::new(FakeLlm::from_script(MODEL, &script.to_string()).unwrap());
        let llm = LlmRetry::new(
            inner.clone(),
            RetryPolicy::daemon(),
            h.clock.clone(),
            Arc::new(NoSleep),
        );
        let ran = h.tick(&llm);
        assert_eq!(
            ran.ran[0].outcome,
            Outcome::Failed(FailureKind::Llm),
            "{fault}"
        );
        assert_eq!(writes(&inner), 1, "{fault}");
        assert_eq!(h.profile().last_error, Some(FailureKind::Llm), "{fault}");
    }
}

/// A [`Sleeper`] that reports itself stopped once asked to wait, as the
/// daemon's does when shutdown wakes it.
#[derive(Default)]
struct StoppedWhileWaiting(AtomicBool);

impl Sleeper for StoppedWhileWaiting {
    fn sleep(&self, _: Duration) {
        self.0.store(true, Ordering::SeqCst);
    }

    fn stopped(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[test]
fn a_refresh_stopped_by_shutdown_mid_outage_is_not_a_failure() {
    // The daemon stops while the write waits out an outage. Nothing is
    // recorded against the model, and no thirty-minute wait starts.
    let h = Harness::new();
    h.says(fact(TEA));
    h.advance(minutes(5));
    let script = json!([{"fail": "status", "status": 503}, {"reply": quiet()}]);
    let inner = Arc::new(FakeLlm::from_script(MODEL, &script.to_string()).unwrap());
    let policy = RetryPolicy::daemon();
    let sleeper = Arc::new(StoppedWhileWaiting::default());
    let llm = LlmRetry::new(inner.clone(), policy, h.clock.clone(), sleeper);
    let ran = h.tick(&llm);
    assert_eq!(writes(&inner), 1);
    let outcomes: Vec<_> = ran.ran.iter().map(|run| &run.outcome).collect();
    let failed = outcomes.iter().any(|o| matches!(o, Outcome::Failed(_)));
    assert!(!failed, "{outcomes:?}");
    assert_eq!(h.profile().last_error, None);
    assert_eq!(h.failures(), (0, vec![]));
}

#[test]
fn a_refresh_held_by_the_gate_waits_for_the_hold_not_thirty_minutes() {
    // The daemon's gate holds every call once any call hits a limit,
    // extraction's included. A refresh it holds never reached the LLM, so
    // it isn't a failure: nothing needs attention, and it runs once the hold
    // lifts rather than thirty minutes after. Both limits are hit five
    // minutes in and lift two minutes later.
    let lifts = at(START) + minutes(7);
    for limit in [
        json!({"fail": "usage_limited", "resets_at": lifts.to_string()}),
        json!({"fail": "status", "status": 429, "retry_after_secs": 120}),
    ] {
        let h = Harness::new();
        h.says(fact(TEA));
        h.advance(minutes(5));
        let script = json!([limit, {"reply": quiet()}]).to_string();
        let inner = Arc::new(FakeLlm::from_script(MODEL, &script).unwrap());
        let gate = LlmGate::new(inner.clone(), 1, h.clock.clone());
        let hello = turn("gated", h.now(), "Hello.");
        h.service.ingest_turn(BANK, &hello).unwrap();
        let limited = h.service.extract_next(BANK, &gate);
        limited.expect_err("extraction's call 1 hits the limit");

        let held = h.tick(&gate);
        assert_eq!(inner.requests().len(), 1, "{limit}");
        let outcomes: Vec<_> = held.ran.iter().map(|run| &run.outcome).collect();
        let failed = outcomes.iter().any(|o| matches!(o, Outcome::Failed(_)));
        assert!(!failed, "{limit}: {outcomes:?}");
        assert_eq!(h.profile().last_error, None, "{limit}");
        assert_eq!(h.failures(), (0, vec![]), "{limit}");
        assert_eq!(held.next_due, Some(lifts), "{limit}");

        h.set(lifts);
        let ran = h.tick(&gate);
        assert_eq!(ran.ran.len(), 1, "{limit}");
        assert_eq!(writes(&inner), 1, "{limit}");
        assert_eq!(h.profile().last_refreshed_at, Some(lifts), "{limit}");
    }
}

#[test]
fn an_urgent_correction_does_not_bypass_the_wait_after_a_failed_refresh() {
    let h = Harness::new();
    let tea = h.seed(fact(TEA));
    let cat = h.seed(fact(CAT));
    h.profile_adding(&[(TEA, &[tea]), (CAT_ENTRY, &[cat])]);
    h.advance(minutes(1));
    h.service.retract(BANK, &tea.to_string()).unwrap();
    let down = FakeLlm::failing(MODEL, || LlmError::Transport {
        reason: "connection refused".into(),
    });
    // A manual refresh can attempt the repair immediately, but its failure
    // must hold even a subsequent urgent correction for thirty minutes.
    let failed_at = h.now();
    assert_eq!(
        h.refresh(PROFILE_NAME, &down, true),
        Outcome::Failed(FailureKind::Llm)
    );
    h.advance(minutes(1));
    h.service.retract(BANK, &cat.to_string()).unwrap();
    let h = h.restart();
    let llm = quiet_llm(1);
    h.set(failed_at + minutes(30) - secs(1));
    let waiting = h.tick(&llm);
    assert!(waiting.ran.is_empty());
    assert_eq!(writes(&llm), 0);
    assert_eq!(waiting.next_due, Some(failed_at + minutes(30)));
    h.advance(secs(1));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    assert_eq!(h.profile().last_error, None);
    assert_eq!(h.profile().last_refreshed_at, Some(h.now()));
}

#[test]
fn forgetting_a_citation_repairs_the_model_after_the_debounce() {
    let h = Harness::new();
    let tea = h.seed(fact(TEA));
    let cat = h.seed(fact(CAT));
    h.profile_adding(&[(TEA, &[tea]), (CAT_ENTRY, &[cat])]);
    h.advance(minutes(1));
    let removed_at = h.now();
    h.service.forget(BANK, &[tea.to_string()]).unwrap();
    assert!(answer(&h.profile()).is_empty());
    assert!(!h.block(None).text.contains(CAT_ENTRY));
    let h = h.restart();
    let llm = FakeLlm::scripted(
        MODEL,
        vec![written(
            &[(SECTION, CAT_ENTRY)],
            &handles(&h.input(PROFILE_NAME), &[cat]),
        )],
    );
    h.set(removed_at + minutes(5) - secs(1));
    let waiting = h.tick(&llm);
    assert!(waiting.ran.is_empty());
    assert_eq!(waiting.next_due, Some(removed_at + minutes(5)));
    h.advance(secs(1));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    assert_eq!(writes(&llm), 1);
    assert!(h.block(None).text.contains(CAT_ENTRY));
    assert_eq!(cites(&h.profile()), BTreeSet::from([cat]));
}

#[test]
fn a_disabled_model_is_never_refreshed() {
    let h = Harness::new();
    h.edit(PROFILE_NAME, json!({"enabled": false})).unwrap();
    h.says(fact(TEA));
    h.advance(minutes(5));
    let llm = quiet_llm(1);
    assert!(h.tick(&llm).ran.is_empty());
    h.set(at(SWEEP));
    assert!(h.tick(&llm).ran.is_empty());
    assert_eq!(writes(&llm), 0);
}

#[test]
fn a_refresh_never_runs_inside_a_block_fetch() {
    // The plugin fetches the block with a 2 s budget.
    let h = Harness::new();
    h.says(fact(TEA));
    h.advance(minutes(10));
    h.block(Some("s1"));
    assert_eq!(h.profile().last_refreshed_at, None);
}

// The refresh input

#[test]
fn a_refresh_selects_current_memories_above_tau_that_pass_the_filters() {
    let (h, faded) = Harness::with_faded(|_| {}, vec![fact("Tim once tried surfing in Raglan.")]);
    let tea = h.seed(fact(TEA));
    let marathon = h.seed(state("Tim is training for a marathon.", "months"));
    let sleepy = h.seed(state("Tim is sleepy.", "days"));
    let no_volatility = h.seed(claim("Tim is between jobs.", "state", "notable"));
    let cinema = h.seed(event("Tim is going to the cinema.", "2026-10-03"));
    let wrong = h.seed(fact("Tim's sister is called Ana."));
    h.service.retract(BANK, &wrong.to_string()).unwrap();
    let berlin = h.seed(fact(BERLIN));
    let lisbon = h.seed_changing(at(EARLIER), fact("Tim lives in Lisbon."), berlin, "ends");
    let hidden = h.seed(fact("Tim's bike is a Brompton."));
    h.service.forget(BANK, &[hidden.to_string()]).unwrap();

    let selected = inputs(&h.input(PROFILE_NAME));
    for memory in [tea, marathon, no_volatility, lisbon.unwrap()] {
        assert!(selected.contains(&memory), "{memory}");
    }
    for (memory, why) in [
        (sleepy, "faster than weeks"),
        (cinema, "an event"),
        (faded[0], "below τ"),
        (wrong, "retracted"),
        (hidden, "forgotten"),
        (berlin, "ended"),
    ] {
        assert!(!selected.contains(&memory), "{why}");
    }
}

#[test]
fn each_input_memory_carries_its_memorys_significance() {
    // The write weighs memories by significance, so each one comes with
    // the level strength uses: the owner's setting over the extracted one.
    let h = Harness::new();
    let tea = h.seed(fact(TEA));
    let maya = h.seed(fact(MAYA).level("critical"));
    let berlin = h.seed(fact(BERLIN).level("major"));
    let cat = h.says(fact(CAT).level("minor"));
    h.keep(tea);

    let input = serde_json::to_value(h.input(PROFILE_NAME)).unwrap();
    let listed = input["memories"].as_array().unwrap();
    let selected: BTreeSet<Uuid> = listed
        .iter()
        .map(|m| m["memory"].as_str().unwrap().parse().unwrap())
        .collect();
    assert_eq!(selected, BTreeSet::from([tea, maya, berlin, cat]));
    for memory in listed {
        let id: Uuid = memory["memory"].as_str().unwrap().parse().unwrap();
        let effective = h.show(id).significance.effective;
        assert_eq!(memory["significance"], json!(effective), "{memory}");
    }
}

#[test]
fn the_profile_admits_recurring_memories_only_with_periods_longer_than_a_week() {
    let h = Harness::new();
    let tea = h.seed(fact("Alex likes green tea."));
    // Each rule and whether it's admitted.
    let rules = [
        (Some("FREQ=YEARLY;BYMONTH=6;BYMONTHDAY=12"), true),
        (Some("FREQ=MONTHLY"), true),
        (Some("FREQ=WEEKLY;INTERVAL=2"), true),
        (Some("FREQ=DAILY;INTERVAL=8"), true),
        (Some("FREQ=WEEKLY;BYDAY=TU"), false),
        (Some("FREQ=DAILY;INTERVAL=7"), false),
        (Some("FREQ=DAILY"), false),
        (Some("FREQ=HOURLY"), false),
        (None, false),
        (Some("FREQ=UNKNOWN"), false),
    ];
    let claim = |n, rule| recurring(&format!("Alex's routine number {n}."), rule, "2026-06-12");
    let claims = rules
        .iter()
        .enumerate()
        .map(|(n, (rule, _))| claim(n, *rule));
    let memories = h.seed_all(BANK, at(EARLIER), claims.collect());
    let admitted = rules.iter().zip(memories).filter(|(rule, _)| rule.1);
    let admitted: BTreeSet<Uuid> = admitted.map(|(_, memory)| memory).chain([tea]).collect();
    assert_eq!(inputs(&h.input(PROFILE_NAME)), admitted);
}

// Synthetic dates, unrelated to any private anniversary. Extraction runs on
// 1 October in Auckland, so 5 October is inside the agenda horizon.
const ANNIVERSARY_RULE: &str = "FREQ=YEARLY;BYMONTH=10;BYMONTHDAY=5";

/// A critical recurring occasion with `rule` and no stated start.
fn occasion(content: &str, text: &str, rule: &str) -> Value {
    claim(content, "recurring", "critical")
        .with("recurrence_text", json!(text))
        .with("recurrence_rrule", json!(rule))
}

fn startless_anniversary() -> Value {
    let content = "Alex and Jo celebrate their anniversary on 5 October.";
    occasion(content, "every 5 October", ANNIVERSARY_RULE)
}

#[test]
fn a_startless_anniversary_is_a_profiled_dated_occasion_not_a_routine() {
    let h = Harness::new();
    let memory = h.says(startless_anniversary());
    let view = h.show(memory);
    assert_eq!(view.kind, "recurring");
    assert_eq!(view.window.recurrence.as_deref(), Some("every 5 October"));
    let rule = view.window.recurrence_rrule.as_deref();
    assert_eq!(rule, Some(ANNIVERSARY_RULE));
    assert_eq!(inputs(&h.input(PROFILE_NAME)), BTreeSet::from([memory]));
    let agenda = h.agenda();
    assert_eq!((agenda.dated, agenda.routines), (vec![memory], vec![]));
    // Deriving a start must not turn a yearly occasion into a daily one.
    h.set(local("2026-10-06T00:00"));
    assert!(h.agenda().dated.is_empty());
    h.set(local("2027-10-01T00:00"));
    assert_eq!(h.agenda().dated, vec![memory]);

    // Learned at noon on the day, it starts that day: a day-only occasion
    // remains relevant at noon even though midnight has passed.
    let h = Harness::new();
    h.set(local("2026-10-05T12:00"));
    let memory = h.says(startless_anniversary());
    let start = WorldTime {
        at: local("2026-10-05T00:00"),
        precision: TimePrecision::Day,
    };
    assert_eq!(h.show(memory).window.recurrence_start, Some(start));
    let agenda = h.agenda();
    assert_eq!((agenda.dated, agenda.routines), (vec![memory], vec![]));
}

#[test]
fn startless_rules_with_unknown_dates_or_interval_phase_keep_only_their_text() {
    for rule in [
        "FREQ=WEEKLY;INTERVAL=2",
        // BY parts do not resolve an interval phase, even with a full date.
        "FREQ=YEARLY;INTERVAL=2;BYMONTH=10;BYMONTHDAY=5",
        "FREQ=DAILY;INTERVAL=2;BYHOUR=9",
        // These BY parts still inherit a day or weekday from DTSTART.
        "FREQ=YEARLY;BYMONTH=10",
        "FREQ=MONTHLY;BYMONTH=10",
        "FREQ=WEEKLY;BYHOUR=9",
    ] {
        let h = Harness::new();
        let content = "Alex has a recurring occasion with no stated first date.";
        let memory = h.says(occasion(content, "a recurring occasion", rule));
        let window = h.show(memory).window;
        assert_eq!(window.recurrence.as_deref(), Some("a recurring occasion"));
        assert_eq!(window.recurrence_rrule, None, "ambiguous rule: {rule}");
        assert!(inputs(&h.input(PROFILE_NAME)).is_empty(), "{rule}");
        assert!(h.agenda().dated.is_empty(), "{rule}");
        assert_eq!(h.agenda().routines, vec![memory], "text survives: {rule}");
    }
}

#[test]
fn an_explicit_recurrence_start_keeps_its_stated_interval_phase() {
    let h = Harness::new();
    // The explicit start fixes even years. Its two-year phase must survive.
    let rule = "FREQ=YEARLY;INTERVAL=2;BYMONTH=10;BYMONTHDAY=5";
    let text = "every other 5 October, starting in 2024";
    let content = format!("Alex and Jo celebrate this occasion {text}.");
    let claim = occasion(&content, text, rule).with("recurrence_start", day("2024-10-05"));
    let memory = h.says(claim);
    let window = h.show(memory).window;
    assert_eq!(window.recurrence_rrule.as_deref(), Some(rule));
    assert_eq!(inputs(&h.input(PROFILE_NAME)), BTreeSet::from([memory]));
    assert_eq!(h.agenda().dated, vec![memory]);
    assert!(h.agenda().routines.is_empty());
    h.set(local("2027-10-01T00:00"));
    assert!(h.agenda().dated.is_empty());
    h.set(local("2028-10-01T00:00"));
    assert_eq!(h.agenda().dated, vec![memory]);
    h.set(local("2028-10-06T00:00"));
    assert!(h.agenda().dated.is_empty());
}

#[test]
fn a_custom_profile_selects_and_refreshes_on_weekly_memories() {
    // The profile asking about recurring activities, over recurring
    // memories alone and over every kind.
    for (kinds, takes_facts) in [(vec![Kind::Recurring], false), (vec![], true)] {
        let h = Harness::new();
        let tea = h.seed(fact("Alex likes green tea."));
        let question = "What are Alex's recurring activities?";
        h.edit(PROFILE_NAME, json!({"kinds": kinds})).unwrap();
        h.ask(PROFILE_NAME, question, &[("Activities", question)]);
        // Leave the minimum refresh interval behind.
        h.advance(minutes(30));
        let llm = quiet_llm(1);
        assert!(h.tick(&llm).ran.is_empty());

        let gym = "Alex goes to the gym every Tuesday.";
        let weekly = h.says(recurring(gym, Some("FREQ=WEEKLY;BYDAY=TU"), "2026-09-29"));
        h.advance(minutes(5) - secs(1));
        assert!(h.tick(&llm).ran.is_empty());
        h.advance(secs(1));
        assert_eq!(h.tick(&llm).ran.len(), 1, "the weekly memory triggers");
        assert!(llm.requests()[0].user.contains(gym));
        let mut expected = BTreeSet::from([weekly]);
        expected.extend(takes_facts.then_some(tea));
        assert_eq!(inputs(&h.input(PROFILE_NAME)), expected);
    }
}

#[test]
fn a_cited_memory_stays_in_the_input_past_the_input_budget() {
    // Keeping cited memories stops one that slips from 3rd to 4th from
    // being removed and added back on alternate refreshes.
    let h = Harness::with(|t| {
        t.mental_models.facet_budget = 3;
        t.mental_models.input_budget = 3;
        t.mental_models.input_budget_with_cited = 4;
    });
    // No word in common with any facet, like the facts below, so it ranks
    // below every one of them on strength alone.
    let weak = h.seed(fact("Miso sleeps."));
    h.profile_adding(&[(CAT_ENTRY, &[weak])]);
    let strong = (1..=5).map(|n| fact(&format!("Fact {n}.")).level("critical"));
    h.seed_all(BANK, h.now(), strong.collect());
    let input = h.input(PROFILE_NAME);
    assert_eq!(input.memories.len(), 4, "the top 3 and the cited one");
    assert!(inputs(&input).contains(&weak));
}

#[test]
fn the_relevance_scale_leaves_the_strength_term_alone_in_a_refresh() {
    // With room for one memory: a weak one sharing five words with the
    // question, against a strong one sharing none. At scale 1.0 the shared
    // words outweigh w_s_inject·strength; at 100.0 strength decides, unless
    // it were scaled too.
    let selected = |scale: f64| {
        let h = Harness::with(|t| {
            set_scale(t, scale);
            t.mental_models.facet_budget = 1;
            t.mental_models.input_budget = 1;
            t.mental_models.input_budget_with_cited = 1;
        });
        let question = "Who is the user: their work and home?";
        h.ask(PROFILE_NAME, question, &[("Answer", question)]);
        let weak = h.seed(fact("The user likes work and home life.").level("trivial"));
        let strong = h.says(fact("Tim cooks dinner.").level("critical"));
        (inputs(&h.input(PROFILE_NAME)), weak, strong)
    };
    let (input, weak, _) = selected(1.0);
    assert_eq!(input, BTreeSet::from([weak]));
    let (input, _, strong) = selected(100.0);
    assert_eq!(input, BTreeSet::from([strong]));
}

#[test]
fn an_unchanged_selection_skips_the_llm_and_force_doesnt() {
    let h = Harness::new();
    let tea = h.seed(fact(TEA));
    h.profile_adding(&[(TEA, &[tea])]);
    let refresh = |force| {
        let llm = quiet_llm(1);
        (h.refresh(PROFILE_NAME, &llm, force), writes(&llm))
    };
    assert_eq!(refresh(false), (Outcome::Unchanged, 0));
    // A memory outside the filters leaves the selection as it was.
    h.says(event("Tim is going to the cinema.", "2026-10-03"));
    assert_eq!(refresh(false), (Outcome::Unchanged, 0));
    assert!(matches!(refresh(true), (Outcome::Applied(_), 1)));

    // The same memories at a new significance are a new input, once.
    h.service
        .set_significance(BANK, &tea.to_string(), Some("major"))
        .unwrap();
    assert!(matches!(refresh(false), (Outcome::Applied(_), 1)));
    assert_eq!(refresh(false), (Outcome::Unchanged, 0));

    h.says(fact(CAT));
    assert!(matches!(refresh(false), (Outcome::Applied(_), 1)));
}

#[test]
fn a_stale_state_write_includes_its_observed_date_but_a_fresh_state_doesnt() {
    let h = Harness::new();
    let stale_text = "Tim is training for a marathon.";
    let fresh_text = "Tim is learning to swim.";
    let stale = h.seed(state(stale_text, "weeks"));
    let fresh = h.says(state(fresh_text, "weeks"));
    let input = h.input(PROFILE_NAME);
    assert_eq!(inputs(&input), BTreeSet::from([stale, fresh]));
    let llm = quiet_llm(1);
    applied(h.refresh(PROFILE_NAME, &llm, true));
    let requests = calls(&llm, WRITE_TEMPLATE);
    let request = &requests[0].user;
    let stale_line = request
        .lines()
        .find(|line| line.contains(stale_text))
        .unwrap();
    let fresh_line = request
        .lines()
        .find(|line| line.contains(fresh_text))
        .unwrap();
    // Pin the absolute date supplied to the writer, not the annotation wording.
    assert!(stale_line.contains("1 Sep"), "{stale_line}");
    assert!(!fresh_line.contains("1 Oct"), "{fresh_line}");
}

#[test]
fn a_state_crossing_the_stale_threshold_refreshes_an_unchanged_selection() {
    let h = Harness::new();
    let memory = h.seed_at(at(START), state("Tim is training for a marathon.", "weeks"));
    // A weekly state is still fresh at six days and stale at eight.
    h.advance(SignedDuration::from_hours(6 * 24));
    let fresh = h.input(PROFILE_NAME);
    assert_eq!(inputs(&fresh), BTreeSet::from([memory]));
    applied(h.refresh(PROFILE_NAME, &quiet_llm(1), false));
    assert_eq!(
        h.refresh(PROFILE_NAME, &quiet_llm(0), false),
        Outcome::Unchanged
    );

    h.advance(SignedDuration::from_hours(2 * 24));
    let stale = h.input(PROFILE_NAME);
    assert_eq!(inputs(&stale), inputs(&fresh));
    let llm = quiet_llm(1);
    applied(h.refresh(PROFILE_NAME, &llm, false));
    assert_eq!(writes(&llm), 1);
    assert_ne!(stale.fingerprint, fresh.fingerprint);

    // Once stale, another day alone must not cause another write.
    h.advance(SignedDuration::from_hours(24));
    let llm = quiet_llm(0);
    assert_eq!(h.refresh(PROFILE_NAME, &llm, false), Outcome::Unchanged);
    assert_eq!(writes(&llm), 0);
}

#[test]
fn rendering_a_stale_state_keeps_the_written_absolute_date_without_adding_age() {
    let h = Harness::new();
    let memory = h.seed(state("Tim is training for a marathon.", "weeks"));
    let text = "As of 1 Sep, Tim is training for a marathon.";
    h.profile_adding(&[(text, &[memory])]);
    let first = h.block(None);
    assert!(
        first.text.ends_with(&format!("Output:\n{text}")),
        "{}",
        first.text
    );
    assert_eq!(first.cited, vec![memory]);

    // A new local day rebuilds the block, but does not date the answer again.
    h.advance(SignedDuration::from_hours(24));
    let next = h.block(None);
    assert_ne!(first.id, next.id);
    assert!(
        next.text.ends_with(&format!("Output:\n{text}")),
        "{}",
        next.text
    );
    assert_eq!(next.cited, vec![memory]);
    assert_eq!(paragraph(answer(&h.profile()), SECTION), Some(text));
}

#[test]
fn a_faded_memory_leaves_the_model_at_the_next_sweep() {
    // When a cited memory fades below τ it leaves the input set,
    // and so leaves the model. A model can't keep a memory alive by itself.
    // Here the owner keeps a long-faded memory, the model cites it, and the
    // owner takes the keep back.
    let (h, faded) = Harness::with_faded(|_| {}, vec![fact(TEA)]);
    let tea = faded[0];
    let cat = h.seed(fact(CAT));
    h.keep(tea);
    h.profile_adding(&[(CAT_ENTRY, &[cat]), (TEA, &[tea])]);
    h.service.unkeep(BANK, &[tea.to_string()]).unwrap();

    let input = inputs(&h.input(PROFILE_NAME));
    assert!(!input.contains(&tea) && input.contains(&cat), "{input:?}");

    // The write is never shown the faded memory, so it can't cite it.
    h.set(at(SWEEP));
    let input = h.input(PROFILE_NAME);
    let reply = written(&[(SECTION, CAT_ENTRY)], &handles(&input, &[cat]));
    let ran = h.tick(&FakeLlm::scripted(MODEL, vec![reply]));
    assert_eq!(ran.ran.len(), 1);
    assert!(
        matches!(ran.ran[0].outcome, Outcome::Applied(_)),
        "the selection changed, so the LLM is called: {ran:?}"
    );
    let profile = h.profile();
    assert_eq!(paragraph(answer(&profile), SECTION), Some(CAT_ENTRY));
    assert_eq!(cites(&profile), BTreeSet::from([cat]));
}

#[test]
fn refreshing_and_building_the_block_write_nothing_but_the_refreshs_recall_log() {
    // Reading, refreshing or injecting a model never counts as an access,
    // and entries are never embedded, extracted from or ingested: no
    // feedback loops. A refresh's retrieval is logged, with no session.
    let h = Harness::new();
    let tea = h.seed(fact(TEA));
    let dentist = h.seed(event(DENTIST, "2026-10-05"));
    let state = |h: &Harness| {
        let accesses = [tea, dentist].map(|m| h.show(m).accesses);
        (accesses, h.overview(), h.service.queue_depth(BANK).unwrap())
    };
    let before = state(&h);
    h.profile_adding(&[(TEA, &[tea])]);
    h.block(Some("s1"));
    h.block(Some("s2"));
    h.agenda();
    assert_eq!(state(&h), before);

    // One retrieval per facet, each logged with its query and no session.
    let facets = h.input(PROFILE_NAME).facets;
    let recalls = h.refresh_recalls();
    assert_eq!(recalls.len(), facets.len(), "{recalls:?}");
    assert!(
        recalls.iter().all(|r| r.session_id.is_none()),
        "{recalls:?}"
    );
    let queries: BTreeSet<_> = recalls.into_iter().filter_map(|r| r.query).collect();
    assert_eq!(queries, facets.into_iter().map(|f| f.query).collect());
}

// Planning and per-facet recall. A refresh asks one narrow question per
// facet instead of one compound question: a plan names the facets, each is
// one retrieval, and one write turns what they found into the summary.

#[test]
fn the_facet_limit_bounds_built_in_and_stored_plans_without_planning_again() {
    // The seeded profile ships its plan, so it costs no plan call and
    // replays exactly. Any plan keeps its first `max_facets` facets, and a
    // lowered limit bounds a plan stored before it.
    let mut h = Harness::with(|t| t.mental_models.max_facets = 3);
    h.seed(fact(TEA));
    h.seed(event(JAPAN, "2027-01-01"));
    let llm = quiet_llm(1);
    applied(h.refresh(PROFILE_NAME, &llm, true));
    assert!(calls(&llm, PLAN_TEMPLATE).is_empty());
    let profile = h.input(PROFILE_NAME).facets;
    assert!(profile.len() <= 3, "{profile:?}");
    let write = &calls(&llm, WRITE_TEMPLATE)[0].user;
    for facet in &profile {
        assert!(write.contains(&facet.heading), "{write}");
    }
    h.service.create_model(BANK, &plans_model(100)).unwrap();
    let parts: Vec<_> = (1..=5)
        .map(|n| (format!("Part {n}"), format!("Trip part {n}?")))
        .collect();
    let parts: Vec<_> = parts
        .iter()
        .map(|(h, q)| (h.as_str(), q.as_str()))
        .collect();
    h.plan("Plans", &parts);
    let headings = |h: &Harness, name| h.input(name).facets.into_iter().map(|f| f.heading);
    assert_eq!(
        headings(&h, "Plans").collect::<Vec<_>>(),
        ["Part 1", "Part 2", "Part 3"]
    );

    h.tuning.mental_models.max_facets = 2;
    let h = h.restart();
    assert_eq!(h.input(PROFILE_NAME).facets, profile[..2]);
    let recalled = h.refresh_recalls().len();
    let llm = quiet_llm(1);
    applied(h.refresh("Plans", &llm, false)); // fewer facets, a new input
    assert!(calls(&llm, PLAN_TEMPLATE).is_empty(), "planned again");
    assert_eq!(h.refresh_recalls().len(), recalled + 2);
    let write = &calls(&llm, WRITE_TEMPLATE)[0].user;
    assert!(
        write.contains("Part 2") && !write.contains("Part 3"),
        "{write}"
    );
}

/// The profile question banks were seeded with before it stopped excluding
/// personal dates.
const EARLIER_PROFILE_QUESTION: &str = "Who is the user: their preferences, important people, \
     work and home, the platforms they use, and how they like to be helped. Not upcoming \
     events, tasks or routines.";

#[test]
fn a_profile_seeded_with_the_earlier_question_still_takes_the_built_in_plan() {
    // A bank created before the question changed still asks the earlier
    // text. Put it back with the store at the current schema version, so
    // either a migration on open or the plan lookup has to carry it over.
    let h = Harness::new();
    h.seed(fact(TEA));
    {
        let store = h.service.store().unwrap();
        let conn = store.connection();
        conn.execute(
            "UPDATE mental_models SET question = ?1 WHERE name = ?2",
            [EARLIER_PROFILE_QUESTION, PROFILE_NAME],
        )
        .unwrap();
        conn.execute_batch(
            "DELETE FROM migrations WHERE to_version > 17; PRAGMA user_version = 17;",
        )
        .unwrap();
    }
    let h = h.restart();
    let llm = quiet_llm(1);
    applied(h.refresh(PROFILE_NAME, &llm, true));
    assert!(calls(&llm, PLAN_TEMPLATE).is_empty(), "planned the profile");
    let facets = h.input(PROFILE_NAME).facets;
    assert!(facets.len() > 1, "{facets:?}");
}

#[test]
fn a_question_is_planned_once_from_itself_and_the_language_alone() {
    // The plan is stored before the facets are recalled, so a failed write
    // doesn't pay for it again, and it's kept across a restart until the
    // question changes. Its request holds nothing that changes between
    // runs or stores, such as the memories or the entries, so the cassette
    // answers it by key.
    let h = Harness::new();
    h.service.create_model(BANK, &plans_model(100)).unwrap();
    let japan = h.seed(event(JAPAN, "2027-01-01"));
    let trips = [
        ("Trips", "Where is Tim going?"),
        ("Dates", "When is Tim going away?"),
    ];
    let llm = FakeLlm::scripted(MODEL, vec![planned(&trips)]);
    let failed = Outcome::Failed(FailureKind::Llm);
    assert_eq!(h.refresh("Plans", &llm, true), failed, "the write fails");
    let first = calls(&llm, PLAN_TEMPLATE).pop().unwrap();
    let question = plans_model(100).question;
    assert!(first.user.contains(&question), "{}", first.user);
    // Each facet's query is a retrieval of its own.
    let queries: BTreeSet<_> = h.refresh_recalls().into_iter().map(|r| r.query).collect();
    assert_eq!(queries, trips.map(|(_, q)| Some(q.to_owned())).into());

    let h = h.restart();
    let llm = quiet_llm(1);
    applied(h.refresh("Plans", &llm, true));
    assert_eq!((calls(&llm, PLAN_TEMPLATE).len(), writes(&llm)), (0, 1));

    // Another question is planned, then the first again once the store
    // holds an entry.
    let countries = "Which countries is Tim visiting?";
    h.ask("Plans", countries, &[("Countries", countries)]);
    let entry = ("Tim is going to Japan next year.", &[japan][..]);
    h.rewrite("Plans", &[("Countries", &[entry])]);
    h.edit("Plans", json!({"question": question})).unwrap();
    let again = h.plan("Plans", &trips);
    let request = |r: &LlmRequest| (r.system.clone(), r.user.clone(), r.schema.clone());
    assert_eq!(request(&again), request(&first));

    // The language is the other thing it holds.
    let english = Harness::with(|t| t.llm.language = Some("English".into()));
    english
        .service
        .create_model(BANK, &plans_model(100))
        .unwrap();
    let request = english.plan("Plans", &trips);
    assert!((request.system + &request.user).contains("English"));
}

#[test]
fn a_date_lost_under_a_compound_question_is_found_through_its_facet() {
    // Recalled as one question, food and drink memories sharing five words
    // with it outrank an anniversary sharing two. Asked on its own, the
    // dates facet finds the anniversary. It still has to pass the profile's
    // filters: a dinner, an event, is left out however well it matches.
    let h = Harness::with(|t| {
        t.mental_models.facet_budget = 5;
        t.mental_models.input_budget = 20;
        t.mental_models.input_budget_with_cited = 20;
    });
    let question = "What does Tim like to drink and eat, and which dates matter to him?";
    h.edit(PROFILE_NAME, json!({"question": question})).unwrap();
    let rule = Some("FREQ=YEARLY;BYMONTH=6;BYMONTHDAY=12");
    let anniversary = "Tim and Sam's wedding anniversary is on 12 June.";
    let dinner = "Tim and Sam have a wedding anniversary dinner booked on 12 June 2027.";
    let mut claims = vec![
        recurring(anniversary, rule, "2026-06-12"),
        event(dinner, "2027-06-12"),
    ];
    let dish = |n| fact(&format!("Tim likes to drink and eat dish {n}.")).level("critical");
    claims.extend((1..=8).map(dish));
    let memories = h.seed_all(BANK, at(EARLIER), claims);
    let (anniversary, dinner) = (memories[0], memories[1]);
    // Until it's planned, the input shows the question recalled whole.
    assert!(!inputs(&h.input(PROFILE_NAME)).contains(&anniversary));

    let dates = "Which wedding anniversary and birthday dates matter to Tim?";
    let food = "What does Tim like to drink and eat?";
    h.plan(PROFILE_NAME, &[("Food and drink", food), ("Dates", dates)]);
    let input = h.input(PROFILE_NAME);
    let selected = inputs(&input);
    assert!(selected.contains(&anniversary) && !selected.contains(&dinner));
    // Each facet takes its best five, and a memory both find is listed once.
    assert!(input.memories.len() <= 10, "{}", input.memories.len());
    assert_eq!(selected.len(), input.memories.len());
}

#[test]
fn the_total_goes_round_the_facets_instead_of_taking_the_best_scores() {
    // Reranker logits for different queries aren't comparable. Every match
    // for the first facet here outscores every match for the second, and
    // the second still gets its share: the total takes each facet's next
    // best in turn.
    let h = Harness::with(|t| {
        t.mental_models.facet_budget = 3;
        t.mental_models.input_budget = 4;
        t.mental_models.input_budget_with_cited = 4;
    });
    let garden = ModelSpec {
        name: "Garden".into(),
        question: "What does Tim grow and keep?".into(),
        kinds: vec![Kind::Fact],
        ..plans_model(100)
    };
    h.service.create_model(BANK, &garden).unwrap();
    let beds = "Which tomatoes, basil, chillies and garlic does Tim grow?";
    h.plan("Garden", &[("Beds", beds), ("Hives", "bees")]);
    let levels = [("one", "critical"), ("two", "major"), ("three", "notable")];
    let claims = levels.into_iter().flat_map(|(bed, level)| {
        let grows = format!("Tim grows tomatoes, basil, chillies and garlic in bed {bed}.");
        [fact(&grows), fact(&format!("Hive {bed} has bees."))].map(|c| c.level(level))
    });
    let m = h.seed_all(BANK, at(EARLIER), claims.collect());
    let expected = BTreeSet::from([m[0], m[1], m[2], m[3]]);
    assert_eq!(inputs(&h.input("Garden")), expected);
}

#[test]
fn a_changed_question_rewrites_the_answer_even_with_the_same_facets() {
    // The question is compared with the plan. Here the new question plans
    // the same facet and selects the same memories, and the refresh still
    // isn't skipped: the write is shown the previous answer, and what it
    // leaves out goes.
    let h = Harness::new();
    let question = mornings(&h, 100);
    let drinks = ["tea", "coffee", "juice"];
    let drinks = drinks.map(|d| fact(&format!("Tim drinks {d} in the morning.")));
    let m = h.seed_all(BANK, at(EARLIER), drinks.into());
    let wording = [
        "Tim has tea first thing.",
        "Tim has coffee too.",
        "Tim has juice.",
    ];
    let sentences = [0, 1, 2].map(|n| (wording[n], std::slice::from_ref(&m[n])));
    h.rewrite("Mornings", &[("Drinks", &sentences)]);
    let input = h.input("Mornings");

    let breakfast = "What does Tim have with breakfast?";
    h.edit("Mornings", json!({"question": breakfast})).unwrap();
    let kept = wording[..2].join(" ");
    let write = written(&[("Drinks", &kept)], &handles(&input, &m[..2]));
    let llm = FakeLlm::scripted(MODEL, vec![planned(&[("Drinks", question)]), write]);
    let rewritten = applied(h.refresh("Mornings", &llm, false));
    assert!(rewritten.written);
    let mornings = h.model("Mornings");
    assert_eq!(
        paragraph(answer(&mornings), "Drinks"),
        Some(wording[..2].join(" ").as_str())
    );
    assert_eq!(cites(&mornings), BTreeSet::from([m[0], m[1]]));
    let write = &calls(&llm, WRITE_TEMPLATE)[0].user;
    assert!(write.contains(wording[2]), "{write}");
}

// The written answer

#[test]
fn a_write_stores_its_sections_as_one_text_citing_what_the_whole_reply_cites() {
    // Each section is its heading's line and then its paragraph or list,
    // with a blank line between sections. A list keeps its lines, but not
    // blank ones or a heading's marks. A section with no text is left out,
    // heading and all. The next write replaces the text and the citations
    // whole.
    let h = Harness::new();
    let claims = [TEA, "Tim walks to work.", CAT, MAYA];
    let m = h.seed_all(BANK, at(EARLIER), claims.map(fact).into());
    let [tea, walk, cat, maya] = m[..] else {
        unreachable!()
    };
    let first = h.refresh_with(PROFILE_NAME, |input| {
        let sections = [
            ("Preferences", "Tim likes green tea. Tim walks to work."),
            ("Hobbies", "  "),
            ("People", MAYA),
            (
                "Pets",
                "- Miso is a  cat\n\n  - Miso is grey\n-\n##\n### Miso sleeps a lot\n#cats",
            ),
        ];
        written(&sections, &handles(input, &[tea, walk, maya, cat]))
    });
    assert!(first.written);
    assert_eq!(first.trimmed, 0);
    let profile = h.profile();
    assert_eq!(
        answer(&profile),
        "### Preferences\nTim likes green tea. Tim walks to work.\n\n### People\nTim's daughter is called Maya.\n\n### Pets\n- Miso is a cat\n- Miso is grey\nMiso sleeps a lot\n#cats"
    );
    assert_eq!(cites(&profile), BTreeSet::from([tea, walk, maya, cat]));

    h.rewrite(PROFILE_NAME, &[("Pets", &[(CAT_ENTRY, &[cat])])]);
    let profile = h.profile();
    assert_eq!(answer(&profile), "### Pets\nTim has a cat called Miso.");
    assert_eq!(cites(&profile), BTreeSet::from([cat]));
}

#[test]
fn a_reply_without_text_or_citations_or_citing_outside_the_input_changes_nothing() {
    // Code can check the citations, not the prose, so one bad citation
    // refuses the whole reply. It's recorded as a malformed reply.
    let h = Harness::new();
    let tea = h.seed(fact(TEA));
    let outside = h.seed(event("Tim went surfing in Raglan.", "2026-09-01"));
    h.profile_adding(&[(TEA, &[tea])]);
    let before = h.profile();
    let input = h.input(PROFILE_NAME);
    assert!(!inputs(&input).contains(&outside));
    let tea = handle(&input, tea);
    for reply in [
        written(&[("Hobbies", "Tim surfs.")], &["m99".into()]),
        written(
            &[("Hobbies", "Tim surfs in Raglan.")],
            &[outside.to_string()],
        ),
        written(
            &[("Hobbies", "Tim likes tea and surfing.")],
            &[tea.clone(), "m99".into()],
        ),
        written(&[("Preferences", "Tim is lovely.")], &[]),
        written(&[("Preferences", "  ")], std::slice::from_ref(&tea)),
        written(&[], std::slice::from_ref(&tea)),
    ] {
        h.advance(minutes(1));
        let llm = FakeLlm::scripted(MODEL, vec![reply.clone()]);
        let outcome = h.refresh(PROFILE_NAME, &llm, true);
        assert_eq!(outcome, Outcome::Failed(FailureKind::Malformed), "{reply}");
        let after = h.profile();
        assert_eq!(answer(&after), answer(&before), "{reply}");
        assert_eq!(cites(&after), cites(&before), "{reply}");
        assert_eq!(after.last_error, Some(FailureKind::Malformed), "{reply}");
    }
}

#[test]
fn a_malformed_reply_or_plan_writes_nothing_and_records_an_error() {
    let h = Harness::new();
    let tea = h.seed(fact(TEA));
    h.profile_adding(&[(TEA, &[tea])]);
    let before = h.profile();
    h.says(fact(CAT));

    h.advance(minutes(1));
    let malformed = Outcome::Failed(FailureKind::Malformed);
    // The last is a reply as the first version of the write took it,
    // sentence by sentence.
    for nonsense in [
        json!({"operations": []}),
        json!({"sections": [{"heading": "Tea", "text": ["all of it"]}], "cites": ["m1"]}),
        json!({"sections": "everything", "cites": ["m1"]}),
        json!({"sections": [{"heading": "Tea", "sentences": [
            {"text": TEA, "cites": ["m1"]},
        ]}]}),
    ] {
        let llm = FakeLlm::scripted(MODEL, vec![nonsense.clone()]);
        let outcome = h.refresh(PROFILE_NAME, &llm, false);
        assert_eq!(outcome, malformed, "{nonsense}");
        let after = h.profile();
        assert_eq!(after.answer, before.answer);
        assert_eq!(after.cites, before.cites);
        assert_eq!(after.last_refreshed_at, before.last_refreshed_at);
        assert_eq!(after.last_error, Some(FailureKind::Malformed));
        assert_eq!(after.last_error_at, Some(h.now()));
    }
    // A failed refresh doesn't record what it selected, so the next one
    // isn't skipped.
    applied(h.refresh(PROFILE_NAME, &quiet_llm(1), false));

    // A plan that names no facet stops the refresh before the write.
    h.service.create_model(BANK, &plans_model(100)).unwrap();
    for nonsense in [
        json!({"facets": "everything"}),
        json!({"facets": []}),
        json!({"sections": []}),
    ] {
        let llm = FakeLlm::scripted(MODEL, vec![nonsense.clone(), quiet()]);
        assert_eq!(h.refresh("Plans", &llm, true), malformed, "{nonsense}");
        let calls = (calls(&llm, PLAN_TEMPLATE).len(), writes(&llm));
        assert_eq!(calls, (1, 0), "{nonsense}");
        assert_eq!(h.model("Plans").last_error, Some(FailureKind::Malformed));
    }
}

// The budget

#[test]
fn creating_resizing_or_enabling_past_the_budget_is_refused() {
    // The budget is 800 and the profile takes 500.
    let h = Harness::new();
    let refused = h.service.create_model(BANK, &plans_model(301));
    assert!(matches!(
        refused,
        Err(ModelError::OverBudget {
            requested: 801,
            budget: 800
        })
    ));
    h.service.create_model(BANK, &plans_model(300)).unwrap();
    // A disabled model isn't rendered, so it doesn't count...
    h.edit("Plans", json!({"enabled": false})).unwrap();
    h.edit(PROFILE_NAME, json!({"max_tokens": 800})).unwrap();
    // ...until it's enabled.
    let enabling = h.edit("Plans", json!({"enabled": true}));
    assert!(matches!(enabling, Err(ModelError::OverBudget { .. })));
    let resizing = h.edit(PROFILE_NAME, json!({"max_tokens": 801}));
    assert!(matches!(resizing, Err(ModelError::OverBudget { .. })));
    assert_eq!(h.profile().max_tokens, 800);
    assert!(
        !h.model("Plans").enabled,
        "a refused enable changed the model"
    );
}

/// A fact-only model asking what Tim drinks in the morning, in
/// `max_tokens`, planned as one facet that recalls its question. Returns
/// the question.
fn mornings(h: &Harness, max_tokens: u32) -> &'static str {
    let question = "What does Tim drink in the morning?";
    let spec = ModelSpec {
        name: "Mornings".into(),
        question: question.into(),
        kinds: vec![Kind::Fact],
        ..plans_model(max_tokens)
    };
    h.service.create_model(BANK, &spec).unwrap();
    h.plan("Mornings", &[("Drinks", question)]);
    question
}

#[test]
fn an_answer_past_max_tokens_loses_sentences_from_the_end_headings_included() {
    // Measured on the stored text, headings and all. The last sentence or
    // line of the last section goes first, and a section left empty takes
    // its heading with it. A sentence ends at `.`, `!` or `?` before a space
    // or the end, so "2.5" doesn't end one.
    let h = Harness::new();
    mornings(&h, 27);
    let tea = h.seed(fact("Tim drinks tea in the morning."));
    let kept = "Tim starts every day with a pot of tea. Then he has a strong black coffee too!";
    let drinks = format!("{kept} Does he drink 2.5 litres before noon?");
    let applied = h.refresh_with("Mornings", |input| {
        let sections = [
            ("Drinks", drinks.as_str()),
            (
                "Later",
                "- Tim has juice at lunch\n- Tim has water at dinner",
            ),
        ];
        written(&sections, &handles(input, &[tea]))
    });
    assert!(applied.written);
    assert_eq!(applied.trimmed, 3);
    let mornings = h.model("Mornings");
    assert_eq!(answer(&mornings), format!("### Drinks\n{kept}"));
    assert!(estimate_tokens(answer(&mornings)) <= 27);
}

// Memories win

#[test]
fn an_answer_renders_whole_and_a_retracted_citation_takes_the_model_out() {
    // One citation that's no longer current taints the whole text, so the
    // model is left out of the block, question and all, until the refresh
    // the retraction requests rewrites it.
    let h = Harness::new();
    let tea = h.seed(fact(TEA));
    let maya = h.seed(fact(MAYA));
    let cat = h.seed(fact(CAT));
    let daughter = "Tim's daughter is Maya.";
    h.rewrite(
        PROFILE_NAME,
        &[
            ("People", &[(daughter, &[maya])]),
            ("Preferences", &[(TEA, &[tea]), (CAT_ENTRY, &[cat])]),
        ],
    );
    let question = h.profile().question;
    let block = h.block(None);
    assert_eq!(
        paragraph(&block.text, "People"),
        Some(daughter),
        "{}",
        block.text
    );
    let preferences = format!("{TEA} {CAT_ENTRY}");
    assert_eq!(
        paragraph(&block.text, "Preferences"),
        Some(preferences.as_str()),
        "{}",
        block.text
    );
    assert!(block.text.contains(&question), "{}", block.text);
    let cited: BTreeSet<Uuid> = block.cited.iter().copied().collect();
    assert_eq!(cited, BTreeSet::from([tea, maya, cat]));

    h.advance(minutes(1));
    let corrected = h.now();
    let mia = fact(MIA).with("changes_something", json!(true));
    h.says_changing(mia, maya, "retracts");
    let block = h.block(None);
    for text in [daughter, TEA, CAT_ENTRY, question.as_str()] {
        assert!(!block.text.contains(text), "{text:?}:\n{}", block.text);
    }
    for memory in [tea, maya, cat] {
        assert!(!block.cited.contains(&memory), "{:?}", block.cited);
    }
    // The stored answer waits for the refresh, which the retraction of a
    // cited memory triggers.
    assert!(answer(&h.profile()).contains(daughter));
    // Urgency survives a restart, but still waits for the debounce.
    let h = h.restart();
    let llm = quiet_llm(2);
    h.set(corrected + minutes(5) - secs(1));
    let waiting = h.tick(&llm);
    assert!(waiting.ran.is_empty());
    assert_eq!(waiting.next_due, Some(corrected + minutes(5)));
    h.advance(secs(1));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    assert_eq!(writes(&llm), 1);

    // Completing the urgent refresh restores the ordinary write interval.
    let refreshed = h.now();
    h.advance(minutes(1));
    h.says(fact("Tim keeps bees."));
    h.advance(minutes(5));
    let waiting = h.tick(&llm);
    assert!(waiting.ran.is_empty());
    assert_eq!(waiting.next_due, Some(refreshed + minutes(30)));
}

#[test]
fn the_model_is_left_out_when_any_one_of_its_memories_ends() {
    let h = Harness::new();
    let berlin = h.seed(fact(BERLIN));
    let cat = h.seed(fact(CAT));
    h.profile_adding(&[
        ("Tim lives in Berlin with his cat Miso.", &[berlin, cat]),
        (CAT_ENTRY, &[cat]),
    ]);
    assert!(h.block(None).text.contains(CAT_ENTRY));
    let moved = event(MOVED, "2026-09-12").with("changes_something", json!(true));
    h.says_changing(moved, berlin, "ends");
    let block = h.block(None);
    assert!(!block.text.contains("Tim lives in Berlin"));
    assert!(!block.text.contains(CAT_ENTRY));
    assert!(block.cited.is_empty(), "{:?}", block.cited);
}

#[test]
fn a_refinement_moves_the_citation_to_the_head_of_the_chain() {
    let h = Harness::new();
    h.service.create_model(BANK, &plans_model(100)).unwrap();
    h.plan("Plans", &[("Trips", "Where is Tim going and when?")]);
    let year = json!({"at": "2027", "precision": "year"});
    let japan = h.seed(claim(JAPAN, "event", "notable").with("valid_from", year));
    h.refresh_adding("Plans", &[("Tim is going to Japan next year.", &[japan])]);

    h.advance(minutes(30));
    let refined = h.now();
    let tokyo = claim(TOKYO, "event", "notable").with("valid_from", month("2027-04"));
    let tokyo = h.says_changing(tokyo, japan, "refines");
    assert_eq!(h.model("Plans").cites, vec![tokyo]);
    // The answer still renders, and the next refresh, which isn't skipped,
    // rewords it.
    let block = h.block(None);
    assert!(block.text.contains("Tim is going to Japan next year."));
    assert!(block.cited.contains(&tokyo));
    assert_eq!(h.tick(&quiet_llm(1)).next_due, Some(refined + minutes(5)));
    applied(h.refresh("Plans", &quiet_llm(1), false));
}

// The block

#[test]
fn the_block_opens_with_guidance_then_holds_the_agenda_and_each_enabled_model() {
    let h = Harness::new();
    // An empty bank's block is the memory guidance alone, dated.
    let guidance = h.block(None).text;
    assert!(guidance.contains("Thu 1 Oct 20:00"), "{guidance}");
    let tea = h.seed(fact(TEA));
    let dentist = h.seed(event(DENTIST, "2026-10-05"));
    h.profile_adding(&[(TEA, &[tea])]);
    h.service.create_model(BANK, &plans_model(100)).unwrap();
    h.edit("Plans", json!({"enabled": false})).unwrap();

    let block = h.block(None);
    assert_eq!(block.built_at, h.now());
    assert_eq!(block.agenda, vec![dentist]);
    assert_eq!(block.cited, vec![tea]);
    for needle in [DENTIST, PROFILE_NAME, TEA] {
        assert!(block.text.contains(needle), "the block lacks {needle:?}");
    }
    assert!(!block.text.contains("Plans"), "a disabled model");
    let rest = block
        .text
        .strip_prefix(&guidance)
        .expect("the guidance first");
    assert!(rest.find(DENTIST) < rest.find(TEA), "the agenda first");

    // A one-facet answer is a paragraph, without a redundant facet heading.
    let question = h.profile().question;
    let single_facet_block = block.text;

    // Multiple facets sit below the model heading, with the stored answer
    // unchanged, a list keeping its lines. An enabled model without an
    // answer has no prompt section.
    h.edit("Plans", json!({"enabled": true})).unwrap();
    let empty = h.block(None);
    assert!(!empty.text.contains("### Plans"), "{}", empty.text);
    assert!(!empty.text.contains("Where is Tim going and when?"));
    let cat = h.seed(fact(CAT));
    let pets = format!("- {CAT_ENTRY}\n- Miso is grey.");
    h.rewrite(
        PROFILE_NAME,
        &[
            ("Drinks", &[(TEA, &[tea])]),
            ("Pets", &[(pets.as_str(), &[cat])]),
        ],
    );
    let block = h.block(None);
    assert!(
        block.text.ends_with(&format!(
            "### {PROFILE_NAME}\n\nPrompt:\n{question}\n\nOutput:\n#### Drinks\n{TEA}\n\n#### Pets\n{pets}"
        )),
        "{}",
        block.text
    );
    assert_eq!(
        answer(&h.profile()),
        format!("### Drinks\n{TEA}\n\n### Pets\n{pets}")
    );
    assert!(
        single_facet_block.ends_with(&format!(
            "### {PROFILE_NAME}\n\nPrompt:\n{question}\n\nOutput:\n{TEA}"
        )),
        "{single_facet_block}"
    );
}

#[test]
fn model_layout_and_question_count_toward_the_shared_block_budget() {
    let mut h = Harness::new();
    let guidance = h.block(None).text;
    let tea = h.seed(fact(TEA));
    let cat = h.seed(fact(CAT));
    h.rewrite(
        PROFILE_NAME,
        &[
            ("Drinks", &[(TEA, &[tea])]),
            ("Pets", &[(CAT_ENTRY, &[cat])]),
        ],
    );
    let question = h.profile().question;
    let expected = format!(
        "{guidance}\n\n### {PROFILE_NAME}\n\nPrompt:\n{question}\n\nOutput:\n#### Drinks\n{TEA}\n\n#### Pets\n{CAT_ENTRY}"
    );
    let budget = estimate_tokens(&expected);
    h.tuning.mental_models.budget = budget as u32;
    h.tuning.mental_models.profile_max_tokens = 100;
    h = h.restart();
    let block = h.block(Some("exact"));
    assert_eq!(block.text, expected);
    assert_eq!(estimate_tokens(&block.text), budget);
    assert_eq!(
        h.in_context("exact").into_iter().collect::<BTreeSet<_>>(),
        BTreeSet::from([tea, cat])
    );

    // One token less must cut the final sentence, not emit an oversized block.
    h.tuning.mental_models.budget -= 1;
    h = h.restart();
    let block = h.block(None);
    assert!(estimate_tokens(&block.text) < budget, "{}", block.text);
    assert!(block.text.contains(TEA), "{}", block.text);
    assert!(!block.text.contains(CAT_ENTRY), "{}", block.text);

    // A long question leaves no room for even one sentence. The model,
    // including its question and citations, must disappear altogether.
    let long_question = "What does Tim like to drink and which pets does Tim have? ".repeat(100);
    h.ask(
        PROFILE_NAME,
        &long_question,
        &[("Drinks", "What does Tim drink?")],
    );
    h.profile_adding(&[(TEA, &[tea])]);
    let block = h.block(Some("excluded"));
    assert_eq!(block.text, guidance);
    assert!(block.cited.is_empty());
    assert!(h.in_context("excluded").is_empty());
    assert!(!h.service.show_model(BANK, PROFILE_NAME).unwrap().renders);
}

#[test]
fn the_smallest_accepted_budgets_fit_the_guidance_a_fold_and_a_dated_update() {
    // The guidance counts in the block's budget, so a budget too small for
    // it and a fold summary is refused. At the smallest accepted one the
    // agenda folds, and what the block left out reaches the session through
    // agenda updates, the smallest accepted update budget included.
    let smallest = |set: &dyn Fn(&mut Tuning, u32)| {
        let accepts = |budget| {
            let mut tuning = Tuning::default();
            set(&mut tuning, budget);
            tuning.validate()
        };
        let budget = (1..=1000).find(|&b| accepts(b).is_ok()).unwrap();
        (budget, accepts(budget - 1))
    };
    let models = |t: &mut Tuning, budget| {
        t.mental_models.budget = budget;
        t.mental_models.profile_max_tokens = budget;
    };
    let (budget, below) = smallest(&models);
    let Err(ConfigError::Invalid(invalid)) = below else {
        panic!("{below:?}")
    };
    assert!(invalid.iter().any(|v| v.key == "mental_models.budget"));
    let (update_budget, _) = smallest(&|t, budget| t.agenda.update_budget = budget);

    let h = Harness::with(|t| {
        models(t, budget);
        t.agenda.update_budget = update_budget;
    });
    let guidance = h.block(None).text;
    assert!(estimate_tokens(&guidance) <= budget as usize);
    let appointment = "Tim has an appointment with the planning committee about the new library.";
    let claims = vec![
        event(appointment, "2026-10-02"),
        task("Tim needs to book a plumber for the dripping upstairs tap."),
    ];
    let left_out = h.seed_all(BANK, at(EARLIER), claims);
    let block = h.block(Some("s1"));
    assert!(block.text.starts_with(&guidance), "{}", block.text);
    assert!(
        estimate_tokens(&block.text) <= budget as usize,
        "{}",
        block.text
    );
    assert!(block.agenda.is_empty(), "the agenda didn't fold");
    let agenda = h.agenda();
    assert_eq!([agenda.dated, agenda.undated_tasks].concat(), left_out);

    let sent: BTreeSet<Uuid> = h.drain_updates("s1").concat().into_iter().collect();
    assert_eq!(sent, left_out.into_iter().collect());
    h.set(local("2026-10-02T09:00"));
    let dated = h.agenda_update("s1").text;
    assert!(dated.contains("Fri 2 Oct"), "{dated}");
    assert!(!dated.contains(appointment), "{dated}");
}

#[test]
fn the_block_is_cached_until_its_content_changes() {
    let h = Harness::new();
    let tea = h.seed(fact(TEA));
    let first = h.block(Some("s1"));
    h.advance(minutes(10));
    let again = h.block(Some("s2"));
    assert_eq!(again.id, first.id);
    assert_eq!(again.built_at, first.built_at);

    // A memory that isn't on the agenda and doesn't change a model leaves
    // the block alone.
    h.says(claim("Tim likes walking.", "fact", "minor"));
    assert_eq!(h.block(None).id, first.id);

    // A new agenda memory clears it...
    h.says(event(HAIRCUT, "2026-10-03").level("minor"));
    let haircut = h.block(None);
    assert_ne!(haircut.id, first.id);
    assert!(haircut.text.contains("Tim has a haircut"));

    // ...and so does a completed refresh.
    h.profile_adding(&[(TEA, &[tea])]);
    let refreshed = h.block(None);
    assert_ne!(refreshed.id, haircut.id);
    assert_eq!(refreshed.built_at, h.now());
    assert!(refreshed.text.contains(TEA));
}

#[test]
fn taking_a_model_out_of_the_prompt_and_back_shows_in_the_next_block_and_survives_a_restart() {
    let h = Harness::new();
    let tea = h.seed(fact(TEA));
    h.profile_adding(&[("Tim likes green tea.", &[tea])]);
    assert!(h.block(None).text.contains("Tim likes green tea."));

    let toggle = |h: &Harness, enabled: bool| {
        h.service
            .edit_model(
                BANK,
                PROFILE_NAME,
                &ModelEdit {
                    enabled: Some(enabled),
                    ..ModelEdit::default()
                },
            )
            .unwrap();
    };

    // The cached block goes with the edit.
    toggle(&h, false);
    assert!(
        !h.block(None).text.contains("Tim likes green tea."),
        "a model taken out of the prompt was rendered"
    );
    let h = h.restart();
    assert!(!h.profile().enabled);
    assert!(!h.block(None).text.contains("Tim likes green tea."));

    // Its entries were kept, so they're back as soon as it is.
    toggle(&h, true);
    assert!(h.block(None).text.contains("Tim likes green tea."));
    let h = h.restart();
    assert!(h.profile().enabled);
    assert!(h.block(None).text.contains("Tim likes green tea."));
}

#[test]
fn the_preview_is_what_a_new_session_gets_and_keeps_nothing() {
    let h = Harness::new();
    let tea = h.seed(fact(TEA));
    h.profile_adding(&[("Tim likes green tea.", &[tea])]);
    let accesses = h.accesses(tea, "injection");
    let preview = || h.service.preview_system_prompt(BANK).unwrap();

    // Nothing has fetched the block, so the preview lays it out now and
    // keeps it nowhere: a fetch a minute later builds its own.
    let laid_out = preview();
    assert!(laid_out.text.contains("Tim likes green tea."));
    h.advance(minutes(1));
    let served = h.block(None);
    assert_ne!(served.built_at, laid_out.built_at);

    // Once a block is cached, the preview is that block.
    h.advance(minutes(1));
    assert_eq!(
        preview(),
        Preview {
            built_at: served.built_at,
            text: served.text.clone(),
        }
    );

    // A change the cache drops shows at once.
    h.service
        .edit_model(
            BANK,
            PROFILE_NAME,
            &ModelEdit {
                enabled: Some(false),
                ..ModelEdit::default()
            },
        )
        .unwrap();
    assert!(!preview().text.contains("Tim likes green tea."));
    assert_eq!(
        h.accesses(tea, "injection"),
        accesses,
        "previewing never counts"
    );
}

#[test]
fn the_clock_alone_changes_the_block_only_at_local_midnight() {
    let h = Harness::new();
    h.seed(event("Tim sees the dentist on 9 October.", "2026-10-09"));
    let first = h.block(None);
    assert!(first.agenda.is_empty(), "9 October is eight days out");

    // 23:59 local is still Thursday.
    h.set(local("2026-10-01T23:59"));
    assert_eq!(h.block(None).id, first.id);

    h.set(local("2026-10-02T00:00"));
    let friday = h.block(None);
    assert_ne!(friday.id, first.id);
    assert_eq!(friday.agenda.len(), 1);
}

// In context

#[test]
fn a_sessions_block_puts_its_agenda_and_cited_memories_in_context_until_cleared() {
    let h = Harness::new();
    let cat = h.seed(fact(CAT));
    let dentist = h.seed(event(DENTIST, "2026-10-05"));
    h.profile_adding(&[(CAT_ENTRY, &[cat])]);
    h.block(Some("s1"));

    let in_context = |h: &Harness| h.in_context("s1").into_iter().collect::<BTreeSet<_>>();
    assert_eq!(in_context(&h), BTreeSet::from([cat, dentist]));
    assert!(h.in_context("s2").is_empty());

    // Relevance injection skips them.
    let prefetch = h.prefetch("s1", "is the cat called Miso", None);
    assert!(!prefetch.injected.contains(&cat));

    // Clearing the session drops its mapping, and the next fetch writes one.
    h.service.clear_session(BANK, "s1").unwrap();
    assert!(h.in_context("s1").is_empty());
    h.block(Some("s1"));
    assert_eq!(in_context(&h), BTreeSet::from([cat, dentist]));
}

#[test]
fn a_mapping_expires_after_thirty_days_without_a_turn() {
    let h = Harness::new();
    let cat = h.seed(fact(CAT));
    h.profile_adding(&[(CAT_ENTRY, &[cat])]);
    h.block(Some("s1"));

    // Past the in-memory idle timeout, the mapping still holds...
    h.advance(SignedDuration::from_hours(29 * 24));
    let h = h.restart();
    assert_eq!(h.in_context("s1"), vec![cat]);

    // ...until sessions.mapping_expiry_days without a turn.
    h.advance(SignedDuration::from_hours(2 * 24));
    let h = h.restart();
    assert!(h.in_context("s1").is_empty());
}

// Agenda updates. Hermes freezes a session's block, so prefetch puts an
// agenda update ahead of the injection when the bank-local day has moved
// past the session's or the agenda lists items the session lacks. It's held
// under the prefetch's `recall_id` and committed by the turn echoing it.

/// A sentence of about `chars` characters about an appointment on `day`.
fn long_sentence(day: &str, chars: usize) -> String {
    let mut text = format!("Tim has an appointment on {day} October");
    while text.len() < chars {
        text.push_str(" about the garden");
    }
    text.truncate(chars);
    text + "."
}

#[test]
fn prefetch_sends_each_agenda_change_once_and_until_a_turn_acknowledges_it() {
    let h = Harness::new();
    let dentist = h.seed(event(DENTIST, "2026-10-05"));
    // Past Thursday's seven-day horizon, inside Friday's.
    let concert = h.seed(event(CONCERT, "2026-10-09"));
    assert_eq!(h.block(Some("s1")).agenda, vec![dentist]);
    // A session that never fetched a block, or cleared it, gets none.
    h.block(Some("cleared"));
    h.service.clear_session(BANK, "cleared").unwrap();
    let unmapped_get_none = |h: &Harness| {
        for session in ["never", "cleared"] {
            assert_eq!(h.agenda_update(session).text, "", "{session}");
        }
    };
    h.advance(minutes(30));
    assert_eq!(h.agenda_update("s1").text, "", "the block is current");

    // Added mid-session, listed without what the block holds.
    let haircut = h.says(event(HAIRCUT, "2026-10-03"));
    let rates = h.says(task_due(RATES, "2026-10-04"));
    unmapped_get_none(&h);
    let first = h.agenda_update("s1");
    for (sentence, listed) in [(HAIRCUT, true), (RATES, true), (DENTIST, false)] {
        assert_eq!(first.text.contains(sentence), listed, "{}", first.text);
    }
    h.sync("s1", Some(first.recall_id));
    let in_context = BTreeSet::from_iter(h.in_context("s1"));
    assert_eq!(in_context, BTreeSet::from([dentist, haircut, rates]));
    assert_eq!(h.agenda_update("s1").text, "", "sent twice");

    // Each later day is stale, with or without anything new, and the update
    // is sent again until a turn acknowledges it.
    for (day, date, new) in [
        ("2026-10-02T09:00", "Fri 2 Oct", Some(concert)),
        ("2026-10-03T09:00", "Sat 3 Oct", None),
    ] {
        h.set(local(day));
        unmapped_get_none(&h);
        for acknowledged in [false, true] {
            let update = h.agenda_update("s1");
            assert!(update.text.contains(date), "{}", update.text);
            assert_eq!(update.text.contains(CONCERT), new.is_some());
            for held in [DENTIST, HAIRCUT, RATES] {
                assert!(!update.text.contains(held), "{}", update.text);
            }
            h.sync("s1", acknowledged.then_some(update.recall_id));
            let committed = h.in_context("s1").contains(&concert);
            assert_eq!(committed, acknowledged || new.is_none(), "{day}");
        }
        assert_eq!(h.agenda_update("s1").text, "", "{day}: sent twice");
    }

    // Sessions live in daemon memory, so a restart may repeat one update.
    let h = h.restart();
    let again = h.agenda_update("s1");
    h.sync("s1", Some(again.recall_id));
    assert_eq!(h.agenda_update("s1").text, "", "sent twice after a restart");
    // A block fetched on the new day is current again.
    h.set(local("2026-10-04T09:00"));
    h.block(Some("s1"));
    assert_eq!(h.agenda_update("s1").text, "");
}

#[test]
fn an_agenda_update_stays_within_its_budget_and_sends_every_item_once() {
    // The budget caps the whole section. The first missing item always goes
    // in, shortened if it must, and the rest wait for later turns.
    let meetings = [
        (
            "the accountant about the quarterly tax return on 2 October.",
            "2026-10-02",
        ),
        (
            "the builder about the kitchen renovation on 3 October 2026.",
            "2026-10-03",
        ),
        (
            "the school about Maya's reading progress on 4 October 2026.",
            "2026-10-04",
        ),
    ];
    let meetings = meetings.map(|(what, on)| event(&format!("Tim has a meeting with {what}"), on));
    let mut cases = vec![(40, meetings.to_vec())];
    // The first item runs across the point where it fits only without the
    // count of what's left out.
    cases.extend((40..=180).step_by(5).map(|chars| {
        let first = event(&long_sentence("2", chars), "2026-10-02");
        (
            60,
            vec![first, event(&long_sentence("3", 200), "2026-10-03")],
        )
    }));
    // A routine's recurrence runs across the point where it fits beside
    // some of the sentence.
    cases.extend((30..=120).step_by(3).map(|chars| {
        let mut recurrence = String::from("every Monday");
        while recurrence.len() < chars {
            recurrence.push_str(" unless it rains");
        }
        recurrence.truncate(chars);
        let rule = Some("FREQ=WEEKLY;BYDAY=MO");
        let watering = recurring("Tim waters the vegetable garden.", rule, "2026-09-28");
        let watering = watering.with("recurrence_text", json!(recurrence.trim_end()));
        (40, vec![watering, weekly("Tim swims on Tuesdays.")])
    }));
    for (n, (budget, claims)) in cases.into_iter().enumerate() {
        let h = Harness::with(|t| t.agenda.update_budget = budget);
        h.block(Some("s1"));
        let mut memories = h.seed_all(BANK, h.now() - minutes(1), claims);
        let updates = h.drain_updates("s1");
        if n == 0 {
            assert!(
                updates[0].len() < memories.len(),
                "the update wasn't capped"
            );
        }
        let mut sent = updates.concat();
        sent.sort();
        memories.sort();
        assert_eq!(sent, memories, "case {n}");
    }
}

#[test]
fn an_agenda_update_comes_first_and_leaves_injection_its_own_room() {
    let h = Harness::new();
    let cat = h.seed(fact(CAT));
    h.block(Some("s1"));
    let events = (0..15).map(|n| {
        let text = format!(
            "Tim has appointment number {n} on 3 October, which involves a long drive across \
             town, a stop at the hardware store for paint and brushes, and a late lunch."
        );
        event(&text, "2026-10-03")
    });
    let events = h.seed_all(BANK, h.now() - minutes(1), events.collect());

    let prefetch = h.prefetch("s1", "is the cat called Miso", None);
    assert_eq!(prefetch.injected, vec![cat]);
    let (update, injection) = prefetch.text.split_once("\n\n").unwrap();
    let budget = h.tuning.agenda.update_budget as usize;
    assert!(estimate_tokens(update) <= budget, "{update}");
    assert!(
        !update.contains(CAT) && injection.contains(CAT),
        "{}",
        prefetch.text
    );
    h.sync("s1", Some(prefetch.recall_id));
    let in_context = h.in_context("s1");
    assert!(in_context.contains(&cat));
    let sent = events.iter().filter(|e| in_context.contains(e)).count();
    assert!(
        (1..events.len()).contains(&sent),
        "{sent} of the events sent"
    );
}

#[test]
fn a_shortened_item_keeps_its_date_and_the_start_of_its_sentence() {
    // (budget, sentence, its start before any cut, cut at a word boundary).
    // A long run of trailing punctuation trims away before the cut is
    // measured.
    let long = long_sentence("3", 1200);
    let dashes = format!("Tim has an appointment {}", "-".repeat(1200));
    let code = format!("Tim's code is {}.", "X".repeat(200));
    let tokyo = format!("Tim lives in {}.", "東京".repeat(200));
    for (budget, sentence, start, at_word) in [
        (200, &long, "Tim has an appointment", true),
        (200, &dashes, "Tim has an appointment", true),
        (40, &code, "Tim's code is ", false),
        (40, &tokyo, "Tim lives in ", false),
    ] {
        let h = Harness::with(|t| t.agenda.update_budget = budget);
        h.block(Some("s1"));
        h.says(event(sentence, "2026-10-03"));
        let update = h.agenda_update("s1");
        let line = update.text.lines().find(|l| l.contains(start)).unwrap();
        assert!(line.contains("Sat 3 Oct"), "the date was cut: {line}");
        // The longest start of the sentence the line holds.
        let mut boundaries = sentence.char_indices().map(|(i, _)| i);
        let kept = boundaries.rfind(|&i| line.contains(&sentence[..i]));
        let kept = &sentence[..kept.unwrap()];
        assert!(kept.len() >= start.trim_end().len(), "{line}");
        if at_word {
            assert!(
                sentence[kept.len()..].starts_with(' '),
                "a broken word: {line}"
            );
        } else {
            assert!(kept.len() > start.len(), "the whole word went: {line}");
        }
        if sentence == &long {
            let tokens = estimate_tokens(&update.text);
            assert!(tokens + 10 >= budget as usize, "the cut left room: {line}");
        }
        // However it's shown, it's sent once.
        h.sync("s1", Some(update.recall_id));
        assert_eq!(h.agenda_update("s1").text, "");
    }

    // Annotations that would leave too little of the sentence give way,
    // the date last: the routine loses its recurrence, and the dated
    // recurring memory keeps its date but not its recurrence.
    let recurrence = "every Monday morning before work, unless it rained overnight, in which \
                      case on Tuesday, and never during the school holidays or the weeks Tim \
                      is travelling for work";
    let watering = recurring("Tim waters.", Some("FREQ=WEEKLY;BYDAY=MO"), "2026-09-28");
    let lessons = recurring("Tim starts lessons.", Some("FREQ=MONTHLY"), "2026-10-03");
    let lessons = lessons.with("valid_from", day("2026-10-03"));
    for (first, second, date) in [
        (watering, weekly("Tim swims on Tuesdays."), None),
        (lessons, event(CONCERT, "2026-10-04"), Some("Sat 3 Oct")),
    ] {
        let sentence = first["content"].as_str().unwrap().to_owned();
        let first = first.with("recurrence_text", json!(recurrence));
        let h = Harness::with(|t| t.agenda.update_budget = 40);
        h.block(Some("s1"));
        h.seed_all(BANK, h.now() - minutes(1), vec![first, second]);
        let update = h.agenda_update("s1").text;
        let line = update.lines().find(|l| l.contains(&sentence)).unwrap();
        assert!(!line.contains("every Monday"), "{update}");
        if let Some(date) = date {
            assert!(line.contains(date), "{update}");
        }
    }
}

// The agenda

#[test]
fn dated_lines_hold_events_and_tasks_within_seven_local_days_even_when_faded() {
    // A minor appointment mentioned long ago mustn't fade out on the day it
    // matters.
    let dentist = event("Tim's dentist appointment is on 2 October.", "2026-10-02");
    let (h, faded) = Harness::with_faded(|_| {}, vec![dentist]);
    let today = h.seed(event("Tim has pottery tonight.", "2026-10-01"));
    let thursday = h.seed(event("Tim flies to Sydney on 8 October.", "2026-10-08"));
    let friday = h.seed(event(CONCERT, "2026-10-09"));
    let due = h.seed(task_due("Tim must pay the rates.", "2026-10-06"));
    let later = h.seed(task_due("Tim must renew his passport.", "2026-10-20"));
    let past = h.seed(event("Tim went to the beach.", "2026-09-28"));

    let agenda = h.agenda();
    assert_eq!(agenda.dated, vec![today, faded[0], due, thursday]);
    for memory in [friday, later, past] {
        assert!(!agenda.dated.contains(&memory));
    }
}

#[test]
fn overdue_tasks_are_listed_for_overdue_days() {
    let h = Harness::with(|t| t.agenda.overdue_days = 10);
    let recent = h.seed(task_due("Tim needs to call the bank.", "2026-09-21"));
    let old = h.seed(task_due("Tim needs to fix the gate.", "2026-09-20"));
    let done = h.seed(task_due("Tim needs to book the vet.", "2026-09-25"));
    let booked = claim("Tim booked the vet.", "event", "notable");
    h.seed_changing(local("2026-09-26T12:00"), booked, done, "ends");

    let agenda = h.agenda();
    assert_eq!(agenda.dated, vec![recent]);
    assert!(!agenda.dated.contains(&old));
    assert!(!agenda.undated_tasks.contains(&done));

    // A dated task must have been said before its due instant to become
    // an overdue obligation. Equality is excluded, just as for events.
    let h = Harness::new();
    let due = local("2026-09-25T09:00");
    let dated_task = |content| {
        task(content).level("minor").with(
            "due_at",
            json!({"at": "2026-09-25T09:00", "precision": "minute"}),
        )
    };
    let before = h.seed_at(due - secs(1), dated_task("Tim needs to call the plumber."));
    let equal = h.seed_at(due, dated_task("Tim needs to collect the parcel."));
    let after = h.seed_at(
        due + secs(1),
        dated_task("Tim needs to return the library book."),
    );
    let retrospective = h.seed_at(
        at("2026-09-25T13:06:00Z"),
        dated_task("Tim needs to take the bins out."),
    );

    let agenda = h.agenda();
    assert!(
        agenda.dated.contains(&before),
        "a genuine overdue obligation stays listed"
    );
    assert_eq!(
        agenda.listed(),
        vec![before],
        "tasks said at or after their due time are records, not plans: {equal}, {after}, {retrospective}"
    );
}

#[test]
fn undated_tasks_renew_on_mentions_and_confirmations_even_inherited_but_not_on_use() {
    // Each task was said on 1 July, long past the 10-day window, and the
    // access it gets at 20:00 on 1 October decides whether it's listed:
    // through 11 October if it renews the window. An inherited access is
    // one on a task the listed one refines, said later but synced from the
    // spool with the earlier time.
    let said = local("2026-07-01T12:00");
    let content = "Tim needs to mend the garden fence.";
    for (access, inherited, renews) in [
        ("mentioned_again", false, true),
        ("mentioned_again", true, true),
        ("confirmed", false, true),
        ("confirmed", true, true),
        ("used", false, false),
    ] {
        let case = format!("{access}, inherited={inherited}");
        let h = Harness::with(|t| t.agenda.undated_days = 10);
        let fence = h.seed_at(said, task(content).level("major"));
        if access == "used" {
            let prefetch = h.prefetch("s", "mend the garden fence", None);
            assert_eq!(prefetch.injected, vec![fence]);
            let mut used = turn("s", h.now(), "Thanks.");
            used.recall_id = Some(prefetch.recall_id.to_string());
            h.service.ingest_turn(BANK, &used).unwrap();
            let call1 = json!({"claims": [], "used_injected_ids": ["m1"]});
            let llm = FakeLlm::scripted(MODEL, vec![call1]);
            let extracted = h.service.extract_next(BANK, &llm).unwrap().unwrap();
            assert_eq!(extracted.used, vec![fence]);
        } else {
            let again = task(content).level("major");
            h.seed_changing(h.now(), again, fence, access);
        }
        let head = if inherited {
            let refined = task("Tim needs to mend the boundary fence.").level("major");
            let refined = h.seed_changing(said + minutes(1), refined, fence, "refines");
            refined.unwrap()
        } else {
            fence
        };
        let listed = |h: &Harness| h.agenda().undated_tasks == vec![head];
        assert_eq!(listed(&h), renews, "{case}");
        h.set(local("2026-10-11T23:59"));
        assert_eq!(listed(&h), renews, "{case}");
        h.set(local("2026-10-12T00:00"));
        assert!(!listed(&h), "{case}");
    }
}

#[test]
fn undated_task_cap_uses_bank_local_dates_not_utc_or_elapsed_hours() {
    let h = Harness::with(|t| t.agenda.undated_days = 30);
    // 1 September UTC is 2 September in Auckland. At the boundary below,
    // more than 30 * 24 hours have elapsed, but it is still local day 30.
    let parts = task("Tim needs to catalogue the spare parts.").level("major");
    let task = h.seed_at(at("2026-09-01T23:30:00Z"), parts);
    h.set(at("2026-10-02T10:59:00Z")); // 2 October, 23:59 NZDT
    assert_eq!(h.agenda().undated_tasks, vec![task]);
    h.set(at("2026-10-02T11:00:00Z")); // 3 October, 00:00 NZDT; UTC date unchanged
    assert!(h.agenda().undated_tasks.is_empty());
}

#[test]
fn undated_task_cap_does_not_shorten_overdue_obligations() {
    let h = Harness::with(|t| t.agenda.undated_days = 10);
    let card = task_due("Tim needs to renew his library card.", "2026-09-05");
    let obligation = h.seed_at(local("2026-07-01T12:00"), card);
    assert_eq!(h.agenda().dated, vec![obligation]);
    assert!(h.agenda().undated_tasks.is_empty());
    h.set(local("2026-10-05T23:59"));
    assert_eq!(h.agenda().dated, vec![obligation]);
    h.set(local("2026-10-06T00:00"));
    assert!(h.agenda().dated.is_empty());
}

#[test]
fn retracted_and_forgotten_memories_are_never_on_the_agenda() {
    let h = Harness::new();
    let dentist = h.seed(event("Tim sees the dentist on 2 October.", "2026-10-02"));
    h.service.retract(BANK, &dentist.to_string()).unwrap();
    let ana = h.seed(task("Tim needs to call Ana."));
    let swim = h.seed(weekly("Tim swims on Tuesdays."));
    let forgotten = [ana, swim].map(|m| m.to_string());
    h.service.forget(BANK, &forgotten).unwrap();
    assert_eq!(h.agenda(), Agenda::default());
}

#[test]
fn over_the_cap_faded_lines_fold_first_then_the_least_significant() {
    let b = event("Tim has event B.", "2026-10-03");
    let (h, faded) = Harness::with_faded(|t| t.agenda.dated_lines = 3, vec![b]);
    let a = h.seed(event("Tim has event A.", "2026-10-02"));
    let c = h.seed(event("Tim has event C.", "2026-10-04").level("minor"));
    let d = h.seed(event("Tim has event D.", "2026-10-05").level("major"));
    let e = h.seed(event("Tim has event E.", "2026-10-06"));

    let agenda = h.agenda();
    assert_eq!(agenda.dated, vec![a, d, e], "in date order");
    assert_eq!(agenda.folded, 2);
    assert!(!agenda.dated.contains(&faded[0]));
    assert!(!agenda.dated.contains(&c));
}

#[test]
fn routines_are_gated_on_tau_ranked_by_strength_and_capped() {
    let chess = recurring("Tim plays chess on Fridays.", None, "");
    let (h, faded) = Harness::with_faded(|_| {}, vec![chess]);
    let swim = h.seed(weekly("Tim swims on Tuesdays.").level("critical"));
    let yoga = h.seed(weekly("Tim does yoga on Tuesdays.").level("major"));
    let call = h.seed(recurring("Tim calls his mum most weekends.", None, ""));
    let walk = recurring("Tim walks the dog.", Some("FREQ=DAILY"), "2026-01-01");
    let daily = h.seed(walk.level("minor"));
    let bins = h.seed(weekly("Tim puts the bins out on Tuesdays.").level("trivial"));

    let agenda = h.agenda();
    assert_eq!(agenda.routines, vec![swim, yoga, call, daily]);
    assert!(!agenda.routines.contains(&bins), "past the cap of 4");
    assert!(!agenda.routines.contains(&faded[0]), "below τ");
    assert!(agenda.dated.is_empty());
}

#[test]
fn a_long_period_routine_joins_the_dated_lines_when_it_next_occurs_within_a_week() {
    let h = Harness::new();
    let monthly = |content, day: u8| {
        let rule = format!("FREQ=MONTHLY;BYMONTHDAY={day}");
        recurring(content, Some(&rule), &format!("2026-01-{day:02}"))
    };
    let soon = h.seed(monthly("Tim pays the rent on the 5th of each month.", 5));
    let later = h.seed(monthly(
        "Tim's book club meets on the 20th of each month.",
        20,
    ));
    let agenda = h.agenda();
    assert_eq!(agenda.dated, vec![soon]);
    assert!(agenda.routines.is_empty(), "a month is longer than a week");
    assert!(!agenda.dated.contains(&later));
}

#[test]
fn an_undated_task_can_fade_out_before_the_last_mention_cap() {
    let h = Harness::with(|t| {
        t.clock.quiet_rate = 1.0;
        t.agenda.undated_days = 30;
    });
    let task = h.says(task("Tim wants to try a new tea.").level("trivial"));
    assert_eq!(h.agenda().undated_tasks, vec![task]);

    // Twenty bank days put this single trivial mention below τ, while
    // it is still inside the 30-day last-mention window. There are no
    // other tasks to exclude it through ranking or the list cap.
    h.advance(SignedDuration::from_hours(20 * 24));
    assert!(h.agenda().undated_tasks.is_empty());
}

#[test]
fn undated_open_tasks_are_gated_on_tau_and_capped() {
    let rust = task("Tim wants to learn Rust someday.");
    let (h, faded) = Harness::with_faded(|t| t.agenda.undated_tasks = 2, vec![rust]);
    let passport = h.seed(task("Tim needs to renew his passport.").level("major"));
    let gutters = h.seed(task("Tim needs to clean the gutters."));
    let shelf = h.seed(task("Tim needs to put up a shelf.").level("minor"));
    let tax = h.seed(task("Tim needs to file the tax return."));
    let filed = claim("Tim filed the tax return.", "event", "notable");
    h.seed_changing(local("2026-09-30T12:00"), filed, tax, "ends");

    let agenda = h.agenda();
    assert_eq!(agenda.undated_tasks, vec![passport, gutters]);
    for memory in [shelf, faded[0], tax] {
        assert!(!agenda.undated_tasks.contains(&memory));
    }
}

// Refresh retry regressions

/// An embedder that answers like `FakeEmbedder` until a test makes it
/// fail, counting every call.
#[derive(Default)]
struct FlakyEmbedder {
    failing: AtomicBool,
    calls: AtomicUsize,
}

impl Embedder for FlakyEmbedder {
    fn model_id(&self) -> &str {
        FakeEmbedder::MODEL_ID
    }

    fn dimensions(&self) -> usize {
        FakeEmbedder.dimensions()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.failing.load(Ordering::SeqCst) {
            return Err(EmbedError::Inference {
                model: FakeEmbedder::MODEL_ID.into(),
                reason: "the test made it fail".into(),
            });
        }
        FakeEmbedder.embed(texts)
    }
}

#[test]
fn a_refresh_whose_retrieval_fails_waits_thirty_minutes_like_any_failure() {
    // A failed refresh waits 30 minutes before retrying. Logging a
    // retrieval failure without settling the request would leave it due
    // on every timer pass, a minute apart in `serve`.
    let embedder = Arc::new(FlakyEmbedder::default());
    let reranker = Arc::new(FakeReranker);
    let models = Models {
        embedder: embedder.clone(),
        reranker,
    };
    let h = Harness::build(|_| {}, models, Vec::new()).0;
    h.says(fact(TEA));
    h.advance(minutes(5));
    let failed_at = h.now();
    embedder.failing.store(true, Ordering::SeqCst);
    let llm = quiet_llm(1);
    let ran = h.tick(&llm);
    assert_eq!(writes(&llm), 0);
    let profile = h.profile();
    assert!(profile.last_error.is_some());
    assert_eq!(profile.last_error_at, Some(failed_at));
    assert_eq!(profile.last_refreshed_at, None);
    assert_eq!(ran.next_due, Some(failed_at + minutes(30)));

    // The embedder recovers, but nothing is tried before the interval.
    embedder.failing.store(false, Ordering::SeqCst);
    let calls = || embedder.calls.load(Ordering::SeqCst);
    let before = calls();
    h.advance(minutes(1));
    assert!(h.tick(&llm).ran.is_empty());
    assert_eq!(calls(), before, "retried early");

    h.set(failed_at + minutes(30));
    assert_eq!(h.tick(&llm).ran.len(), 1);
    assert_eq!(writes(&llm), 1);
    let profile = h.profile();
    assert_eq!(profile.last_error, None);
    assert_eq!(profile.last_refreshed_at, Some(failed_at + minutes(30)));
}

/// A reranker that never answers.
struct FailingReranker;

impl Reranker for FailingReranker {
    fn model_id(&self) -> &str {
        FakeReranker::MODEL_ID
    }

    fn rerank(&self, _: &str, _: &[&str]) -> Result<Vec<f32>, EmbedError> {
        Err(EmbedError::Inference {
            model: FakeReranker::MODEL_ID.into(),
            reason: "the test made it fail".into(),
        })
    }
}

#[test]
fn a_scored_refresh_the_reranker_fails_has_no_logits_and_ranks_on_strength() {
    // The refresh goes on without the reranker, and what it scored says
    // so: no facet reranked and no candidate with a logit, rather than the
    // 0.0 the ranking stands in with, so no floor can count one as passing.
    let models = Models {
        embedder: Arc::new(FakeEmbedder),
        reranker: Arc::new(FailingReranker),
    };
    let h = Harness::build(|_| {}, models, Vec::new()).0;
    let tea = h.says(fact(TEA));
    let cat = h.says(fact(CAT).level("critical"));
    h.advance(minutes(5));
    let llm = quiet_llm(1);
    let (ran, scored) = h.service.scored_refreshes(&llm).unwrap();
    assert_eq!(ran.ran.len(), 1);
    assert!(matches!(ran.ran[0].outcome, Outcome::Applied(_)), "{ran:?}");
    assert_eq!(writes(&llm), 1, "the refresh still writes");

    let [refresh] = &scored[..] else {
        panic!("one scored refresh: {scored:?}");
    };
    assert_eq!(refresh.model, PROFILE_NAME);
    assert_eq!(refresh.at, h.now());
    assert!(!refresh.facets.is_empty());
    for facet in &refresh.facets {
        assert!(!facet.pool.reranked, "{facet:?}");
        let pool = &facet.pool.candidates;
        let memories: Vec<Uuid> = pool.iter().map(|c| c.memory).collect();
        assert_eq!(memories, vec![cat, tea], "strength decides: {facet:?}");
        for candidate in pool {
            assert_eq!(candidate.logit, None, "{facet:?}");
            assert_eq!(candidate.taken, Taken::Budget, "{facet:?}");
            assert!(candidate.input.is_some(), "{facet:?}");
        }
    }
}

#[test]
fn a_scored_refresh_marks_the_cited_memory_its_fill_took_past_the_budget() {
    // As in a_cited_memory_stays_in_the_input_past_the_input_budget: the
    // cited memory ranks below the top 3 under every facet, so each facet's
    // budget cuts it and the cited fill takes it, while the strong facts
    // past the budget are cut. Every facet shows it under the one handle
    // it reached the write under.
    let h = Harness::with(|t| {
        t.mental_models.facet_budget = 3;
        t.mental_models.input_budget = 3;
        t.mental_models.input_budget_with_cited = 4;
    });
    let weak = h.seed(fact("Miso sleeps."));
    h.profile_adding(&[(CAT_ENTRY, &[weak])]);
    let strong = (1..=5).map(|n| fact(&format!("Fact {n}.")).level("critical"));
    h.seed_all(BANK, h.now(), strong.collect());
    h.advance(minutes(30));
    let llm = quiet_llm(1);
    let (ran, scored) = h.service.scored_refreshes(&llm).unwrap();
    assert_eq!(ran.ran.len(), 1, "{ran:?}");
    let [refresh] = &scored[..] else {
        panic!("one scored refresh: {scored:?}");
    };

    let mut handles = BTreeSet::new();
    for facet in &refresh.facets {
        let pool = &facet.pool.candidates;
        let cited = pool
            .iter()
            .find(|candidate| candidate.memory == weak)
            .unwrap_or_else(|| panic!("the cited memory is scored: {facet:?}"));
        assert!(cited.cited, "{facet:?}");
        assert_eq!(cited.taken, Taken::Cited, "{facet:?}");
        handles.insert(cited.input.clone().expect("the fill took it"));
        let budget = pool.iter().filter(|c| c.taken == Taken::Budget).count();
        assert_eq!(budget, 3, "{facet:?}");
        let cut: Vec<_> = pool.iter().filter(|c| c.taken == Taken::Cut).collect();
        assert_eq!(cut.len(), 2, "{facet:?}");
        assert!(
            cut.iter().all(|c| !c.cited && c.input.is_none()),
            "{facet:?}"
        );
    }
    assert_eq!(handles.len(), 1, "one handle: {handles:?}");
}

// `used` is credited by memory handle alone (docs/mental-model-answer.md,
// section 5): call 1 is given every memory the session's block cites as an
// in-context memory with its own handle, so a reply that relied on a fact
// the agent read in the block is credited to that memory. The block's
// rendered text gets no handles of its own and isn't kept with the turn.

/// The profile sentence session `s1`'s block shows, citing `cat` and `tea`.
const BLOCK_SENTENCE: &str = "Tim drinks green tea with his cat Miso nearby.";

/// Session `s1` holds a block whose profile cites `cat` and `tea` in one
/// sentence, and the owner's turn in it is queued.
fn a_turn_after_the_block() -> (Harness, Uuid, Uuid) {
    let h = Harness::new();
    let cat = h.seed(fact(CAT));
    let tea = h.seed(fact(TEA));
    h.profile_adding(&[(BLOCK_SENTENCE, &[cat, tea])]);
    h.block(Some("s1"));
    let tea_time = turn("s1", h.now() - minutes(1), "Tea time?");
    h.service.ingest_turn(BANK, &tea_time).unwrap();
    (h, cat, tea)
}

/// Extracts the queued turn with a call 1 that says the reply relied on
/// `relied_on`, named by the handle call 1's input gives it, and returns
/// what was extracted, that input, and the request call 1 was sent.
fn extract_relying_on(h: &Harness, relied_on: Uuid) -> (Extracted, Call1Input, LlmRequest) {
    let claimed = h.service.next_extraction(BANK).unwrap().unwrap();
    let input = h
        .service
        .call1_input(&claimed.lease, &claimed.in_context)
        .unwrap();
    let handle = input
        .in_context
        .iter()
        .find(|memory| memory.memory == relied_on)
        .unwrap_or_else(|| panic!("{relied_on} is in call 1's context"))
        .handle
        .clone();
    let call1 = json!({"claims": [], "used_injected_ids": [handle]});
    let llm = FakeLlm::scripted(MODEL, vec![call1]);
    let extracted = h
        .service
        .extract_chunk(claimed.lease, &llm, &claimed.in_context)
        .unwrap();
    let request = llm.requests().remove(0);
    (extracted, input, request)
}

#[test]
fn a_reply_relying_on_a_fact_read_in_the_block_is_credited_to_that_memory_alone() {
    // Both memories are shown, by handle; the block's sentence isn't. A
    // refresh rewording it before the worker gets there changes nothing.
    let (h, cat, tea) = a_turn_after_the_block();
    let asleep = "Tim only drinks tea when Miso the cat is asleep.";
    h.profile_adding(&[(asleep, &[cat, tea])]);

    let (extracted, input, request) = extract_relying_on(&h, cat);
    let shown: BTreeSet<Uuid> = input.in_context.iter().map(|m| m.memory).collect();
    assert_eq!(shown, BTreeSet::from([cat, tea]));
    for sentence in [BLOCK_SENTENCE, asleep] {
        assert!(
            !request.user.contains(sentence),
            "call 1 was shown the block's text:\n{}",
            request.user
        );
    }
    assert_eq!(extracted.used, vec![cat]);
    assert_eq!(h.accesses(cat, "used"), 1);
    assert_eq!(
        h.accesses(tea, "used"),
        0,
        "the memory sharing its sentence was credited"
    );
}

#[test]
fn a_reply_naming_a_handle_call_1_wasnt_given_credits_nothing() {
    // `n1` was the block sentence's handle; nothing has it now.
    let (h, cat, tea) = a_turn_after_the_block();
    let call1 = json!({"claims": [], "used_injected_ids": ["n1"]});
    let llm = FakeLlm::scripted(MODEL, vec![call1]);
    let extracted = h.service.extract_next(BANK, &llm).unwrap().unwrap();
    assert!(extracted.used.is_empty(), "{:?}", extracted.used);
    assert_eq!(h.accesses(cat, "used"), 0);
    assert_eq!(h.accesses(tea, "used"), 0);
}

/// The profile's entries as version 13 kept them: `(section, text, cites)`
/// in position order.
type OldEntry<'a> = (Option<&'a str>, &'a str, &'a [Uuid]);

#[test]
fn an_upgrade_from_version_13_keeps_each_answer_and_extracts_a_queued_turn_without_its_snapshot() {
    // Version 13 kept the block's rendered sentences with each built block
    // and each queued turn, and a model's answer as entries, each citing
    // its own memories. Put the queued turn's snapshot and the profile's
    // entries back as they were left, and the store back at 13.
    let (h, cat, tea) = a_turn_after_the_block();
    let snapshot = json!([{
        "entry": Uuid::from_u128(1),
        "text": BLOCK_SENTENCE,
        "cites": [cat, tea],
    }]);
    {
        let store = h.service.store().unwrap();
        let conn = store.connection();
        let has_entries: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('prompt_blocks') WHERE name = 'entries'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        if has_entries == 0 {
            conn.execute(
                "ALTER TABLE prompt_blocks ADD COLUMN entries TEXT NOT NULL DEFAULT '[]'",
                [],
            )
            .unwrap();
        }
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS turn_entries (
               id        INTEGER PRIMARY KEY AUTOINCREMENT,
               source_id INTEGER NOT NULL UNIQUE REFERENCES sources(id) ON DELETE CASCADE,
               entries   TEXT NOT NULL
             );
             DELETE FROM migrations;
             INSERT INTO migrations (from_version, to_version, binary_version, started_at,
                                     completed_at)
               VALUES (0, 13, 'v13', 0, 0);
             PRAGMA user_version = 13;",
        )
        .unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO turn_entries (source_id, entries)
             SELECT source_id, ?1 FROM turn_in_context",
            [snapshot.to_string()],
        )
        .unwrap();

        conn.execute_batch(
            "DROP TABLE IF EXISTS mental_model_cites;
             ALTER TABLE mental_models DROP COLUMN answer;",
        )
        .unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS mental_model_entries (
               id         INTEGER PRIMARY KEY AUTOINCREMENT,
               uuid       TEXT NOT NULL UNIQUE,
               model_id   INTEGER NOT NULL REFERENCES mental_models(id) ON DELETE CASCADE,
               position   INTEGER NOT NULL,
               text       TEXT NOT NULL,
               created_at INTEGER NOT NULL,
               updated_at INTEGER NOT NULL,
               section    TEXT
             );
             CREATE TABLE IF NOT EXISTS mental_model_citations (
               entry_id  INTEGER NOT NULL REFERENCES mental_model_entries(id) ON DELETE CASCADE,
               memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
               PRIMARY KEY (entry_id, memory_id)
             );
             CREATE INDEX IF NOT EXISTS mental_model_entries_model
               ON mental_model_entries(model_id, position);
             CREATE INDEX IF NOT EXISTS mental_model_citations_memory
               ON mental_model_citations(memory_id);
             DELETE FROM mental_model_entries;",
        )
        .unwrap();
        let model: i64 = conn
            .query_row(
                "SELECT id FROM mental_models WHERE name = ?1",
                [PROFILE_NAME],
                |row| row.get(0),
            )
            .unwrap();
        // An entry from before sections comes first, whatever its position.
        let entries: [OldEntry<'_>; 4] = [
            (Some("Drinks"), TEA, &[tea]),
            (None, CAT_ENTRY, &[cat]),
            (Some("Drinks"), BLOCK_SENTENCE, &[cat, tea]),
            (Some("Pets"), "Miso is Tim's cat.", &[cat]),
        ];
        for (position, (section, text, cites)) in entries.into_iter().enumerate() {
            conn.execute(
                "INSERT INTO mental_model_entries
                   (uuid, model_id, position, section, text, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, 0, 0)",
                rusqlite::params![
                    Uuid::from_u128(position as u128 + 10).to_string(),
                    model,
                    position as i64,
                    section,
                    text
                ],
            )
            .unwrap();
            let entry = conn.last_insert_rowid();
            for memory in cites {
                conn.execute(
                    "INSERT INTO mental_model_citations (entry_id, memory_id)
                     SELECT ?1, id FROM memories WHERE uuid = ?2",
                    rusqlite::params![entry, memory.to_string()],
                )
                .unwrap();
            }
        }
    }
    let h = h.restart();

    // Each section's entries join into its paragraph, in position order.
    let profile = h.profile();
    assert_eq!(
        answer(&profile),
        format!(
            "{CAT_ENTRY}\n\n### Drinks\n{TEA} {BLOCK_SENTENCE}\n\n### Pets\nMiso is Tim's cat."
        )
    );
    assert_eq!(cites(&profile), BTreeSet::from([cat, tea]));

    let (extracted, _, request) = extract_relying_on(&h, cat);
    assert!(
        !request.user.contains(BLOCK_SENTENCE),
        "call 1 was shown the snapshot:\n{}",
        request.user
    );
    assert_eq!(extracted.used, vec![cat]);
    assert_eq!(h.accesses(tea, "used"), 0);
    // The block still puts what it cites in the session's context.
    let in_context: BTreeSet<Uuid> = h.in_context("s1").into_iter().collect();
    assert_eq!(in_context, BTreeSet::from([cat, tea]));
}

// The block-id fallback: a session that fetched its block without an id is
// mapped by the block id its first `prefetch` carries. The daemon keeps
// what each built block lists and cites by id, since the cache may have
// rebuilt by then.

#[test]
fn a_prefetch_carrying_a_block_id_maps_a_session_that_fetched_without_one() {
    // The plugin holds the id of the block Hermes froze, which the cache may
    // have rebuilt since. A session that already has a mapping keeps it, and
    // an unknown id, or another bank's, maps nothing: nothing refers across
    // banks (CONTEXT.md, "Bank").
    let h = Harness::new();
    let other = BankIdentity {
        timezone: Some(TZ.into()),
        ..BankIdentity::default()
    };
    h.service.ensure_bank_with_models("other", &other).unwrap();
    let dinner = event("Tim has dinner with Ana on 2 October.", "2026-10-02");
    let theirs_listed = h.seed_all("other", at(EARLIER), vec![dinner]);
    let theirs = h.service.system_prompt("other", None).unwrap();
    assert_eq!(theirs.agenda, theirs_listed);
    let cat = h.seed(fact(CAT));
    let tea = h.seed(fact(TEA));
    h.profile_adding(&[(CAT_ENTRY, &[cat])]);
    let held = h.block(None);
    h.block(Some("s1"));
    h.profile_adding(&[(TEA, &[tea])]);
    let newer = h.block(None);
    assert_ne!(newer.id, held.id);
    assert!(newer.cited.contains(&tea));
    assert!(h.in_context("s2").is_empty());

    let prefetch = h.prefetch("s2", "is the cat called Miso", Some(held.id));
    assert!(!prefetch.injected.contains(&cat), "a cited memory");
    assert_eq!(h.in_context("s2"), vec![cat]);
    h.prefetch("s1", "hello there", Some(newer.id));
    assert_eq!(h.in_context("s1"), vec![cat]);
    for (session, block) in [("s3", Uuid::nil()), ("s4", theirs.id)] {
        h.prefetch(session, "hello there", Some(block));
        assert!(h.in_context(session).is_empty(), "{session}");
    }

    let h = h.restart();
    assert_eq!(h.in_context("s2"), vec![cat]);
}

// Refresh and budget regressions

/// A refresh LLM that, while its first call is in flight, has the owner
/// keep `keep`: a triggering write landing in the middle of a refresh.
struct KeepsDuringCall<'a> {
    service: &'a Service,
    keep: Uuid,
    calls: AtomicUsize,
}

impl LlmClient for KeepsDuringCall<'_> {
    fn model(&self) -> &str {
        MODEL
    }

    fn complete(&self, _request: &LlmRequest) -> Result<LlmResponse, LlmError> {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            self.service.keep(BANK, &[self.keep.to_string()]).unwrap();
        }
        Ok(LlmResponse {
            json: quiet(),
            usage: None,
            latency: Duration::ZERO,
        })
    }
}

#[test]
fn a_trigger_during_a_refresh_is_refreshed_after_the_interval() {
    // The refresh selected its inputs before the keep, so
    // the kept memory was never shown to the LLM. The request the keep made
    // has to survive the refresh's completion and run once the minimum
    // interval has passed, not wait for the next write or the 04:00 sweep.
    let (h, faded) = Harness::with_faded(|_| {}, vec![fact("Tim once tried surfing in Raglan.")]);
    h.says(fact(TEA));
    h.advance(minutes(5));
    let refreshed = h.now();
    let llm = KeepsDuringCall {
        service: &h.service,
        keep: faded[0],
        calls: AtomicUsize::new(0),
    };
    let ran = h.service.run_refreshes(&llm).unwrap();
    assert_eq!(ran.ran.len(), 1);
    assert_eq!(llm.calls.load(Ordering::SeqCst), 1);
    // The trigger during the refresh wasn't dropped.
    assert_eq!(ran.next_due, Some(refreshed + minutes(30)));

    h.set(refreshed + minutes(30));
    let quiet = quiet_llm(1);
    assert_eq!(h.tick(&quiet).ran.len(), 1);
    assert!(quiet.requests()[0].user.contains("surfing in Raglan"));
}

#[test]
fn a_stated_end_holds_through_its_unit_in_the_block_and_on_the_agenda() {
    // A stored time is the start of its unit, so an end
    // of 1 October holds through 1 October, and an end in October through
    // October (`Window::closes_at`). A future ending, already recorded by
    // the memory that ends it, hasn't ended anything yet.
    let h = Harness::new();
    let lisbon = state("Tim is in Lisbon until 1 October 2026.", "months");
    let lisbon = h.seed(lisbon.with("valid_until", day("2026-10-01")));
    let course = state("Tim is doing a pottery course until October.", "months");
    let course = h.seed(course.with("valid_until", month("2026-10")));
    let acme = h.seed(fact("Tim works at Acme."));
    let leaving = event("Tim works at Acme until 30 November 2026.", "2026-11-30");
    h.seed_changing(at(EARLIER), leaving, acme, "ends");
    let gutters = task("Tim needs to clean the gutters today.");
    let gutters = h.says(gutters.with("valid_until", day("2026-10-01")));
    let swim = weekly("Tim swims on Tuesdays until October.");
    let swim = h.seed(swim.with("valid_until", month("2026-10")));
    let yoga = "Tim does yoga on Thursdays.";
    let yoga = h.seed(recurring(yoga, Some("FREQ=WEEKLY;BYDAY=TH"), "2026-01-01"));
    let stopping = event("Tim does yoga until 1 December 2026.", "2026-12-01");
    h.seed_changing(at(EARLIER), stopping, yoga, "ends");
    let sentences = [
        "Tim is in Lisbon for now.",
        "Tim is doing a pottery course.",
        "Tim works at Acme.",
    ];
    let cites: [&[Uuid]; 3] = [&[lisbon], &[course], &[acme]];
    h.profile_adding(&sentences.into_iter().zip(cites).collect::<Vec<_>>());

    let block = h.block(None);
    for sentence in sentences {
        assert!(
            block.text.contains(sentence),
            "{sentence:?}:\n{}",
            block.text
        );
    }
    let agenda = h.agenda();
    assert_eq!(agenda.undated_tasks, vec![gutters]);
    let routines: BTreeSet<Uuid> = agenda.routines.iter().copied().collect();
    assert_eq!(routines, BTreeSet::from([swim, yoga]));

    // The day ends at local midnight, which takes the model out with
    // Lisbon; the month and November go on, so an answer without Lisbon
    // renders.
    h.set(local("2026-10-02T00:00"));
    assert!(!h.block(None).text.contains(sentences[1]));
    let agenda = h.agenda();
    assert!(agenda.undated_tasks.is_empty());
    assert!(agenda.routines.contains(&swim));
    h.profile_adding(&[(sentences[1], &[course]), (sentences[2], &[acme])]);
    let block = h.block(None);
    assert!(block.text.contains(sentences[1]), "{}", block.text);
    assert!(block.text.contains(sentences[2]), "{}", block.text);
    h.set(local("2026-11-01T00:00"));
    let block = h.block(None);
    assert!(!block.text.contains(sentences[2]), "{}", block.text);
}

#[test]
fn the_whole_block_stays_within_the_budget_and_records_only_what_it_renders() {
    // Every model "shares about 800 tokens of
    // `system_prompt_block()` with the agenda". The agenda keeps its own
    // caps and is laid out first, so today's appointment can't be pushed
    // out by a model; the models get what's left. What the block lists or
    // cites, and so puts in a session's context, is only what it rendered.
    // A model too long for what's left is cut at a sentence or line end, and one
    // with no sentence that fits is left out. `model show` says which.
    let h = Harness::new();
    let today = "Tim has a dentist appointment this evening at the clinic on Queen Street.";
    let mut claims = vec![event(today, "2026-10-01")];
    // Leave room for some, but not all, of the profile after the agenda.
    claims.extend((0..7).map(|n| {
        let text = format!(
            "Tim has appointment number {n} with the planning committee about the new library."
        );
        event(&text, &format!("2026-10-{:02}", 2 + n % 7))
    }));
    claims.extend((0..4).map(|n| {
        let text = format!("Tim goes to evening class number {n} at the community hall weekly.");
        recurring(&text, Some("FREQ=WEEKLY;BYDAY=WE"), "2026-01-07")
    }));
    let job = |n| format!("Tim needs to sort out household job number {n} before the weekend.");
    claims.extend((0..5).map(|n| task(&job(n))));
    claims.extend((0..25).map(|n| fact(&format!("Tim's profile fact number {n} is here."))));
    let memories = h.seed_all(BANK, at(EARLIER), claims);
    let today = memories[0];
    let facts = &memories[memories.len() - 25..];
    let entry =
        |n| format!("Tim is described here by profile entry number {n:02}, as fact {n:02} says.");
    let texts: Vec<String> = (0..25).map(entry).collect();
    let adds = texts.iter().zip(facts);
    let adds: Vec<_> = adds
        .map(|(text, fact)| (text.as_str(), std::slice::from_ref(fact)))
        .collect();
    h.profile_adding(&adds);

    let block = h.block(Some("s1"));
    let budget = h.tuning.mental_models.budget as usize;
    let tokens = estimate_tokens(&block.text);
    assert!(tokens <= budget, "{tokens} tokens:\n{}", block.text);
    assert!(block.agenda.contains(&today), "today's appointment");
    assert!(block.text.contains("dentist appointment this evening"));

    // The answer is cut at a sentence or line end, and the block cites the model's
    // whole citation set, which goes in context, since some of it renders.
    let rendered = texts
        .iter()
        .take_while(|text| block.text.contains(text.as_str()))
        .count();
    assert!(0 < rendered && rendered < texts.len(), "{}", block.text);
    let cut = texts[..rendered].join(" ");
    assert!(
        block.text.contains(&format!("{cut}\n")) || block.text.ends_with(&cut),
        "{}",
        block.text
    );
    for text in &texts[rendered..] {
        assert!(!block.text.contains(text.as_str()), "{text:?}");
    }
    let cited: BTreeSet<Uuid> = block.cited.iter().copied().collect();
    assert_eq!(cited, facts.iter().copied().collect());
    let in_context: BTreeSet<Uuid> = h.in_context("s1").into_iter().collect();
    let shown: BTreeSet<Uuid> = block.agenda.iter().chain(&block.cited).copied().collect();
    assert_eq!(in_context, shown);
    let renders = || h.service.show_model(BANK, PROFILE_NAME).unwrap().renders;
    assert!(renders(), "a cut answer still shows");

    // Seven more appointments leave no room for the model's heading,
    // question and first sentence.
    let more = (7..14).map(|n| {
        let text = format!(
            "Tim has appointment number {n} with the planning committee about the new library."
        );
        event(&text, &format!("2026-10-{:02}", 2 + n % 7))
    });
    h.seed_all(BANK, at(EARLIER), more.collect());
    let block = h.block(Some("s2"));
    assert!(estimate_tokens(&block.text) <= budget, "{}", block.text);
    assert!(!block.text.contains(&texts[0]), "{}", block.text);
    assert!(
        facts.iter().all(|fact| !block.cited.contains(fact)),
        "{:?}",
        block.cited
    );
    assert!(!renders(), "the block left the model out");
}
