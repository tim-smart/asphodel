//! The strength model through `Service`: strength counts use, not
//! retrieval; the lasting floor counts separate occasions in world time;
//! memory runs on bank time and truth on world time; a closed window
//! restarts recent use; a correction inherits what it corrects; and purge
//! waits below its line for the dates still ahead.
//!
//! Memories are made by extracting turns with a scripted LLM and read back
//! with `show_memory` and `faded_at`. The one-mention lifetimes, the purge
//! table, a three-week holiday and the Maya-to-Mia correction run as
//! scenarios under `scenarios/`.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asphodel_core::config::{Layer, Tuning};
use asphodel_core::ingest::Turn;
use asphodel_core::inspect::{Guard, MemoryView, StrengthView};
use asphodel_core::models::{FakeEmbedder, FakeLlm, FakeReranker, Models};
use asphodel_core::retrieval::RecallRequest;
use asphodel_core::store::bank::BankIdentity;
use asphodel_core::store::{OpenOptions, Store};
use asphodel_core::{Service, SimulatedClock};
use jiff::{SignedDuration, Timestamp};
use serde_json::{Value, json};
use uuid::Uuid;

use asphodel_core::strength::*;

const EXACT: f64 = 1e-9;
const DAYS_PER_YEAR: f64 = 365.2425;
const BANK: &str = "main";

fn t0() -> Timestamp {
    "2026-01-05T09:00:00Z".parse().unwrap()
}

/// `days` world days after T0.
fn at(days: f64) -> Timestamp {
    t0().checked_add(SignedDuration::from_secs_f64(days * 86_400.0))
        .unwrap()
}

fn ts(text: &str) -> Timestamp {
    text.parse().unwrap()
}

#[track_caller]
fn assert_near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected} ± {tolerance}, got {actual}"
    );
}

/// A temporary directory removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A service on the fakes with one bank in UTC. Bank time runs at world
/// speed, δ is 1, tasks are held 30 days overdue, and the significance
/// levels are the calibrated ones, unless `extra` says otherwise. Field
/// order matters: the service drops before its directory.
struct Harness {
    service: Service,
    clock: Arc<SimulatedClock>,
    _dir: TestDir,
}

impl Harness {
    fn new(extra: &str) -> Self {
        let base = format!(
            "[clock]\nquiet_rate = 1.0\n[purge]\ndelta = 1.0\n[agenda]\noverdue_days = 30\n\
             [strength.significance]\ntrivial = 0.0\nminor = 0.2\nnotable = 0.5\nmajor = 0.7\n\
             critical = 0.9\n\
             [injection.reranker_floors]\n\"{}\" = 1.0\n\
             [ranking.relevance_scales]\n\"{0}\" = 1.0\n\
             [reconcile.embedding_floors]\n\"{}\" = 0.5\n",
            FakeReranker::MODEL_ID,
            FakeEmbedder::MODEL_ID,
        );
        let layer = |origin, text| Layer { origin, text };
        let tuning = Tuning::from_layers(&[layer("base", &base), layer("test", extra)]).unwrap();
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = TestDir(
            std::env::temp_dir().join(format!("asphodel-strength-{}-{n}", std::process::id())),
        );
        std::fs::create_dir_all(&dir.0).unwrap();
        let clock = Arc::new(SimulatedClock::new(t0()));
        let store = Store::open(&dir.0, OpenOptions::default(), clock.clone()).unwrap();
        let service = Service::with_models(clock.clone(), store, tuning, Models::fake()).unwrap();
        let identity = BankIdentity {
            timezone: Some("UTC".into()),
            ..BankIdentity::default()
        };
        service.ensure_bank_with_models(BANK, &identity).unwrap();
        Self {
            service,
            clock,
            _dir: dir,
        }
    }

    fn set(&self, to: Timestamp) {
        self.clock.set(to);
    }

    /// The owner says each claim's quote at `when` in `tz`, and the turn is
    /// extracted with call 1 finding `claims` and judging `used` used, and
    /// call 2 giving the first claim `labels` against their memories.
    /// Returns the new memories.
    fn turn(
        &self,
        when: Timestamp,
        tz: &str,
        claims: Vec<Value>,
        used: &[Uuid],
        labels: &[(Uuid, &str)],
    ) -> Vec<Uuid> {
        self.set(when);
        let quotes: Vec<&str> = claims
            .iter()
            .map(|c| c["quote"].as_str().unwrap())
            .collect();
        let turn = Turn {
            session_id: "s".into(),
            message_at: when,
            timezone: Some(tz.into()),
            user_text: format!("{} ({when})", quotes.join(" ")),
            assistant_text: "Noted.".into(),
            author: None,
            platform: Some("cli".into()),
            recall_id: None,
            forget_requested: false,
        };
        self.service.ingest_turn(BANK, &turn).unwrap();
        let lease = self.service.claim_chunk(BANK).unwrap().expect("queued");
        let input = self.service.call1_input(&lease, used).unwrap();
        let handles: Vec<&str> = used
            .iter()
            .map(|id| {
                let memory = input.in_context.iter().find(|m| m.memory == *id);
                memory.expect("in context").handle.as_str()
            })
            .collect();
        let call1 = json!({"claims": claims, "used_injected_ids": handles});
        let mut replies = vec![call1.clone()];
        if let Some(input) = self.service.call2_input(&lease, &call1, used).unwrap() {
            let labels: Vec<Value> = labels
                .iter()
                .map(|(memory, label)| {
                    let neighbour = input.neighbours.iter().find(|n| n.memory == *memory);
                    let handle = &neighbour.expect("a neighbour").handle;
                    json!({"neighbour": handle, "label": label})
                })
                .collect();
            replies.push(json!({"claims": [{"claim": input.claims[0].handle, "labels": labels}]}));
        } else {
            assert!(labels.is_empty(), "call 2 didn't run");
        }
        let llm = FakeLlm::scripted("fake-llm", replies);
        let extracted = self.service.extract_chunk(lease, &llm, used).unwrap();
        extracted.memories
    }

    fn says(&self, when: Timestamp, claim: Value) -> Uuid {
        self.turn(when, "UTC", vec![claim], &[], &[])[0]
    }

    /// The owner says `claim`, which call 2 labels against `memory`.
    fn labels(&self, when: Timestamp, claim: Value, memory: Uuid, label: &str) -> Vec<Uuid> {
        self.turn(when, "UTC", vec![claim], &[], &[(memory, label)])
    }

    /// A turn whose reply used `memory`.
    fn uses(&self, when: Timestamp, memory: Uuid) {
        self.turn(when, "UTC", vec![], &[memory], &[]);
    }

    fn show(&self, memory: Uuid) -> MemoryView {
        self.service.show_memory(BANK, &memory.to_string()).unwrap()
    }

    fn strength(&self, memory: Uuid) -> StrengthView {
        self.show(memory).strength
    }

    /// The memories `memory` inherits accesses from.
    fn inherits(&self, memory: Uuid) -> BTreeSet<Uuid> {
        let accesses = self.show(memory).accesses;
        accesses.iter().filter_map(|a| a.inherited_from).collect()
    }
}

/// Call 1's claim of `content`, quoting all of it, with no times.
fn claim(kind: &str, significance: &str, content: &str) -> Value {
    json!({
        "content": content, "quote": content, "kind": kind, "significance": significance,
        "remember_this": false, "changes_something": false, "window_confidence": "high",
        "entities": [],
    })
}

fn trivial(kind: &str, content: &str) -> Value {
    claim(kind, "trivial", content)
}

fn minor(kind: &str, content: &str) -> Value {
    claim(kind, "minor", content)
}

trait With {
    fn with(self, key: &str, value: Value) -> Value;
}

impl With for Value {
    fn with(mut self, key: &str, value: Value) -> Value {
        self[key] = value;
        self
    }
}

/// A claim flagged as changing something, so its neighbours aren't held to
/// the floor.
fn changing(claim: Value) -> Value {
    claim.with("changes_something", json!(true))
}

/// A source-local time at a precision, as call 1 writes it.
fn local(at: &str, precision: &str) -> Value {
    json!({"at": at, "precision": precision})
}

fn day(date: &str) -> Value {
    local(&format!("{date}T00:00"), "day")
}

/// `before` until the minute before `at`, and `after` from `at`.
fn edge(at: &str, before: Phase, after: Phase) -> Vec<(Timestamp, Phase)> {
    let at = ts(at);
    let minute_before = at.checked_sub(SignedDuration::from_mins(1)).unwrap();
    vec![(minute_before, before), (at, after)]
}

#[test]
fn bank_time_runs_at_world_speed_while_the_bank_talks_and_never_faster() {
    // A bank at world speed fades a trivial memory about a week after its
    // one mention. A quiet bank chatting every hour for ten days runs at
    // full speed throughout: its overlapping 24-hour windows merge rather
    // than add up, so the memory fades at the same moment.
    let world = Harness::new("");
    let tea = world.says(at(0.0), trivial("fact", "Tim likes green tea."));
    world.set(at(20.0));

    let busy = Harness::new("[clock]\nquiet_rate = 0.1\n");
    let biscuit = busy.says(at(0.0), trivial("fact", "Tim likes green tea."));
    for hour in 1..=240 {
        busy.turn(at(f64::from(hour) / 24.0), "UTC", vec![], &[], &[]);
    }
    busy.set(at(20.0));

    let fades = |h: &Harness, memory| h.service.faded_at(BANK, memory).unwrap().unwrap();
    let (world, busy) = (fades(&world, tea), fades(&busy, biscuit));
    assert!(world > at(7.0) && world < at(8.0), "{world}");
    assert!(
        busy.duration_since(world).abs() <= SignedDuration::from_mins(1),
        "{busy} vs {world}"
    );
}

#[test]
fn massed_use_spikes_then_falls_below_spaced_use() {
    // Use every three hours for a week, long enough to saturate decay,
    // against four uses a week apart. The same burst broken by 30 cold
    // days recovers when use resumes.
    let h = Harness::new("");
    let massed = h.says(at(0.0), trivial("fact", "Tim's gym locker is number 37."));
    let resumed = h.says(at(0.001), trivial("fact", "Tim's gym code is 4512."));
    let spaced = h.says(
        at(0.002),
        trivial("fact", "Tim's bus to work is number 14."),
    );
    let mut uses: Vec<(f64, Uuid)> = (1..64)
        .flat_map(|i| {
            let day = f64::from(i) / 8.0;
            let gap = if i >= 32 { 30.0 } else { 0.0 };
            [(day, massed), (day + gap + 0.001, resumed)]
        })
        .chain((1..5).map(|week| (7.0 * f64::from(week) + 0.002, spaced)))
        .collect();
    uses.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut check = |until: f64| {
        for &(day, memory) in uses.iter().filter(|(day, _)| *day < until) {
            h.uses(at(day), memory);
        }
        uses.retain(|(day, _)| *day >= until);
        h.set(at(until));
        (h.strength(massed), h.strength(resumed), h.strength(spaced))
    };
    let (m, _, s) = check(7.9);
    assert!(m.value > s.value, "{m:?} {s:?}");
    let (m, r, _) = check(38.0);
    assert!(r.value > m.value && r.recallable, "{m:?} {r:?}");
    let (m, r, s) = check(90.0);
    assert!(m.value < r.value && r.value < s.value, "{m:?} {r:?} {s:?}");
    assert_eq!((m.occasions, r.occasions, s.occasions), (3, 4, 5));
    assert!(!m.recallable && s.recallable, "{m:?} {s:?}");
}

#[test]
fn one_used_credit_strengthens_at_once_and_the_floor_counts_it_like_any_access() {
    // Three memories said together, the one left alone last so it starts
    // freshest. Five world days on, one is used in a reply once and another
    // is mentioned again. The single use lifts recent use straight away,
    // with no second credit needed, and the floor counts it as an occasion
    // whatever its kind.
    let h = Harness::new("");
    let used = h.says(at(0.0), trivial("fact", "Tim's desk is by the window."));
    let mentioned = h.says(at(0.001), trivial("fact", "Tim's car is blue."));
    let cold = h.says(at(0.002), trivial("fact", "Tim's locker is number 12."));
    h.uses(at(5.0), used);
    let again = trivial("fact", "Tim's car is blue.");
    assert!(
        h.labels(at(5.001), again, mentioned, "mentioned_again")
            .is_empty()
    );

    h.set(at(5.5));
    let (cold, used, mentioned) = (h.strength(cold), h.strength(used), h.strength(mentioned));
    assert!(used.recent_use > cold.recent_use, "{used:?} {cold:?}");
    assert_eq!(
        (cold.occasions, used.occasions, mentioned.occasions),
        (1, 2, 2)
    );
    assert_eq!(used.lasting_floor, mentioned.lasting_floor);
    assert!(used.lasting_floor > cold.lasting_floor, "{used:?} {cold:?}");
}

#[test]
fn separate_occasions_in_world_time_make_a_memory_permanent_even_once_it_ends() {
    // A quiet bank, where three world days between turns are only 1.2 bank
    // days. Berlin is mentioned on four occasions exactly three world days
    // apart; Otago four times too, but the last only two days after the
    // third, so it counts three. A notable memory needs four occasions to
    // stay in recall for good, and ending Berlin's window doesn't lower its
    // floor.
    let h = Harness::new("[clock]\nquiet_rate = 0.1\n");
    let berlin_claim = || claim("state", "notable", "Tim lives in Berlin.");
    let otago_claim = || claim("fact", "notable", "Tim studied physics at Otago.");
    let berlin = h.says(at(0.0), berlin_claim());
    let otago = h.says(at(0.001), otago_claim());
    for (day, again, memory) in [
        (3.0, berlin_claim(), berlin),
        (3.001, otago_claim(), otago),
        (6.0, berlin_claim(), berlin),
        (6.001, otago_claim(), otago),
        (8.001, otago_claim(), otago),
        (9.0, berlin_claim(), berlin),
    ] {
        h.labels(at(day), again, memory, "mentioned_again");
    }
    let lisbon = changing(claim("state", "notable", "Tim moved to Lisbon."));
    let lisbon = h.labels(at(12.0), lisbon, berlin, "ends")[0];

    h.set(at(100.0 * DAYS_PER_YEAR));
    let (berlin, otago) = (h.show(berlin), h.show(otago));
    assert_eq!(berlin.chain.ended_by, Some(lisbon));
    assert_eq!(berlin.strength.occasions, 4);
    assert_eq!(otago.strength.occasions, 3);
    assert!(berlin.strength.recallable, "{:?}", berlin.strength);
    assert_eq!(berlin.projection.fade, None);
    assert!(!otago.strength.recallable, "{:?}", otago.strength);
}

// Window close: recent use restarts from one boost at the later of the
// close and when the end became known. The lasting floor stays.

#[test]
fn a_closed_window_restarts_recent_use_once_its_end_is_known() {
    let h = Harness::new("");
    // Staying in Lisbon ends on day 30, but that's only said on day 40.
    let lisbon = h.says(at(0.0), trivial("state", "Tim is staying in Lisbon."));
    // A trivial appointment on day 90, mentioned once.
    let dentist = trivial("event", "Tim has a dentist appointment on 5 April.")
        .with("valid_from", day("2026-04-05"));
    let dentist = h.says(at(0.001), dentist);

    h.set(at(35.0));
    assert!(!h.strength(lisbon).recallable, "closed but not known to be");
    let left = changing(trivial("state", "Tim left Lisbon on 4 February."))
        .with("valid_from", day("2026-02-04"));
    h.labels(at(40.0), left, lisbon, "ends");
    // The restart is when the end was said, not when it was.
    h.set(at(41.0));
    assert!(h.strength(lisbon).recallable);

    // Before the appointment's day is over it has long faded; once it is,
    // it's back in recall, as strong as a fresh mention then.
    h.set(ts("2026-04-05T23:59:00Z"));
    assert!(!h.strength(dentist).recallable);
    let fresh = h.says(
        ts("2026-04-06T00:00:00Z"),
        trivial("fact", "Tim likes tea."),
    );
    h.set(ts("2026-04-06T01:00:00Z"));
    let back = h.show(dentist);
    assert_eq!(back.phase, Some(Phase::RecentlyPast));
    assert!(back.strength.recallable);
    assert_near(back.strength.value, h.strength(fresh).value, EXACT);
}

#[test]
fn a_correction_inherits_along_its_chain_but_nothing_passes_along_an_ending() {
    let h = Harness::new("");
    let name = |n: &str| changing(minor("fact", &format!("Tim's sister is called {n}.")));
    // Maya → Mia → Mya, each correcting the one before.
    let maya = h.says(at(0.0), name("Maya"));
    let mia = h.labels(at(1.0), name("Mia"), maya, "retracts")[0];
    let mya = h.labels(at(2.0), name("Mya"), mia, "retracts")[0];
    assert_eq!(h.show(maya).chain.head, mya);
    assert_eq!(h.inherits(mia), BTreeSet::from([maya]));
    assert_eq!(h.inherits(mya), BTreeSet::from([maya, mia]));

    // Two memories refined into one.
    let acme = h.says(at(3.0), minor("fact", "Tim works at Acme."));
    let wellington = h.says(at(3.001), minor("fact", "Tim works in Wellington."));
    let both = changing(minor("fact", "Tim works at Acme in Wellington."));
    let refines = [(acme, "refines"), (wellington, "refines")];
    let both = h.turn(at(4.0), "UTC", vec![both], &[], &refines)[0];
    assert_eq!(h.show(wellington).chain.head, both);
    assert_eq!(h.inherits(both), BTreeSet::from([acme, wellington]));

    // Moving to Lisbon ends living in Berlin: neither takes the other's
    // accesses.
    let berlin = h.says(at(5.0), minor("state", "Tim lives in Berlin."));
    let lisbon = changing(minor("state", "Tim moved to Lisbon."));
    let lisbon = h.labels(at(6.0), lisbon, berlin, "ends")[0];
    let shown = h.show(berlin);
    assert_eq!(shown.chain.head, berlin);
    assert_eq!(shown.chain.ended_by, Some(lisbon));
    assert!(h.inherits(lisbon).is_empty());
}

#[test]
fn phase_follows_the_window_ending_each_unit_in_the_source_timezone() {
    let h = Harness::new("");
    let event = |content: &str, from: Value| minor("event", content).with("valid_from", from);
    let ends = |at: &str| edge(at, Phase::Upcoming, Phase::RecentlyPast);
    let rows = vec![
        // 3 Oct in Auckland (NZDT, +13) ends at 11:00 UTC.
        (
            "Pacific/Auckland",
            event("Dinner with Sam on 3 October.", day("2026-10-03")),
            ends("2026-10-03T11:00:00Z"),
        ),
        // London's spring-forward day is 23 hours long.
        (
            "Europe/London",
            event("The fair is on 29 March.", day("2026-03-29")),
            ends("2026-03-29T23:00:00Z"),
        ),
        // March in London ends on BST.
        (
            "Europe/London",
            event("The audit is in March.", local("2026-03-01T00:00", "month")),
            ends("2026-03-31T23:00:00Z"),
        ),
        (
            "Pacific/Auckland",
            event("The move is in 2027.", local("2027-01-01T00:00", "year")),
            ends("2027-12-31T11:00:00Z"),
        ),
        (
            "UTC",
            event("The call is at 3pm.", local("2026-10-03T15:00", "hour")),
            ends("2026-10-03T16:00:00Z"),
        ),
        // A span is current between its start and end units.
        (
            "UTC",
            event("Tim is in Rome from 5 to 9 October.", day("2026-10-05"))
                .with("valid_until", day("2026-10-09")),
            [
                edge("2026-10-06T00:00:00Z", Phase::Upcoming, Phase::Current),
                edge("2026-10-10T00:00:00Z", Phase::Current, Phase::RecentlyPast),
            ]
            .concat(),
        ),
        // A task is current until its due day ends, then overdue until it's
        // done; the completion below makes it past.
        (
            "UTC",
            minor("task", "Tim needs to renew his passport by 9 October.")
                .with("due_at", day("2026-10-09")),
            [
                edge("2026-10-10T00:00:00Z", Phase::Current, Phase::Overdue),
                edge("2026-10-12T10:31:00Z", Phase::Overdue, Phase::RecentlyPast),
            ]
            .concat(),
        ),
        (
            "UTC",
            minor("task", "Tim needs to organise the garage."),
            vec![(ts("2030-01-01T00:00:00Z"), Phase::Current)],
        ),
        // Facts, states and routines without an end stay current; a start
        // still ahead is upcoming.
        (
            "UTC",
            minor("fact", "Tim likes green tea."),
            vec![(ts("2026-10-01T12:00:00Z"), Phase::Current)],
        ),
        (
            "UTC",
            minor("state", "Tim has lived in Wellington since March 2020.")
                .with("valid_from", local("2020-03-01T00:00", "month")),
            vec![(ts("2026-10-01T12:00:00Z"), Phase::Current)],
        ),
        (
            "UTC",
            minor("recurring", "Tim does yoga on Tuesdays from 10 October.")
                .with("valid_from", day("2026-10-10"))
                .with("recurrence_text", json!("every Tuesday")),
            vec![(ts("2026-10-01T12:00:00Z"), Phase::Upcoming)],
        ),
        // "Lived in Berlin until 12 Sep": recently past, then long past.
        (
            "UTC",
            minor("state", "Tim lived in Berlin until 12 September.")
                .with("valid_until", day("2026-09-12")),
            vec![
                (ts("2026-10-01T12:00:00Z"), Phase::RecentlyPast),
                (ts("2026-10-13T00:00:00Z"), Phase::LongPast),
            ],
        ),
    ];

    let mut checks = Vec::new();
    let mut passport = None;
    for (n, (tz, claim, expected)) in rows.into_iter().enumerate() {
        let is_task = claim["due_at"].is_object();
        let memory = h.turn(at(n as f64 * 0.001), tz, vec![claim], &[], &[])[0];
        if is_task {
            passport = Some(memory);
        }
        for (when, phase) in expected {
            checks.push((when, memory, phase));
        }
    }
    checks.sort_by_key(|(when, ..)| *when);

    let done_at = ts("2026-10-12T10:30:00Z");
    for (when, memory, phase) in checks {
        if when > done_at
            && let Some(passport) = passport.take()
        {
            let renewed = changing(minor("fact", "Tim renewed his passport."));
            h.labels(done_at, renewed, passport, "ends");
        }
        h.set(when);
        let shown = h.show(memory);
        assert_eq!(shown.phase, Some(phase), "{} at {when}", shown.sentence);
    }
}

// State confidence.

#[test]
fn only_a_mention_or_a_confirmation_renews_a_stale_states_rank() {
    // Strength doesn't rank here, and the states are equally relevant and
    // equally significant, so only state confidence tells them apart: the
    // longer since a state was last said or confirmed, the lower it ranks.
    let h = Harness::new("[ranking]\nw_s_recall = 0.0\n");
    let staying = |city: &str| {
        minor("state", &format!("Tim is staying in {city}.")).with("volatility", json!("days"))
    };
    let lisbon = h.says(at(0.0), staying("Lisbon"));
    let porto = h.says(at(0.001), staying("Porto"));
    let rome = h.says(at(0.002), staying("Rome"));
    let oslo = h.says(at(2.0), staying("Oslo"));
    // Porto is mentioned again on day 9, Rome confirmed on day 10 and Oslo
    // used in a reply then, which doesn't count as hearing it again.
    assert!(
        h.labels(at(9.0), staying("Porto"), porto, "mentioned_again")
            .is_empty()
    );
    assert!(
        h.labels(at(10.0), staying("Rome"), rome, "confirmed")
            .is_empty()
    );
    h.uses(at(10.001), oslo);

    h.set(at(10.5));
    let request = RecallRequest {
        query: "Where is Tim staying?".into(),
        ..RecallRequest::default()
    };
    let recall = h.service.recall(BANK, &request).unwrap();
    let ranked: Vec<Uuid> = recall.results.iter().map(|r| r.id).collect();
    assert_eq!(ranked, [rome, porto, oslo, lisbon]);
}

#[test]
fn purge_waits_below_its_line_for_dates_still_ahead_and_overdue_tasks() {
    // Every memory here is trivial and said once in January, so by July
    // each head is far below τ − δ and only its dates can hold it back.
    let h = Harness::new("");
    let dinner =
        trivial("event", "Dinner with Sam on 3 October.").with("valid_from", day("2026-10-03"));
    let dinner = h.turn(at(0.0), "Pacific/Auckland", vec![dinner], &[], &[])[0];
    let lease = trivial("state", "Tim rents the flat until 1 December at 9:30.")
        .with("valid_until", local("2026-12-01T09:30", "minute"));
    let lease = h.says(at(0.001), lease);
    let passport = trivial("task", "Tim needs to renew his passport by 9 October.")
        .with("due_at", day("2026-10-09"));
    let passport = h.says(at(0.002), passport);
    let yoga = trivial("recurring", "Tim goes to yoga every Tuesday.")
        .with("recurrence_text", json!("every Tuesday"));
    let yoga = h.says(at(0.003), yoga);
    let garage = h.says(at(0.004), trivial("task", "Tim needs to tidy the garage."));
    // A reschedule from 20 October to 1 May.
    let old =
        trivial("event", "The dentist is on 20 October.").with("valid_from", day("2026-10-20"));
    let old = h.says(at(0.005), old);
    let new = changing(trivial("event", "The dentist moved to 1 May."))
        .with("valid_from", day("2026-05-01"));
    let new = h.labels(at(0.006), new, old, "retracts")[0];

    h.set(ts("2026-07-01T00:00:00Z"));
    let guards = |memory| h.show(memory).purge.guards;
    let until = ts("2026-10-03T11:00:00Z");
    assert_eq!(guards(dinner), [Guard::DateAhead { until }]);
    let lease_ends = ts("2026-12-01T09:31:00Z");
    assert_eq!(guards(lease), [Guard::DateAhead { until: lease_ends }]);
    // Due 9 Oct: overdue from 10 Oct, held for 30 days after that.
    let held = ts("2026-11-09T00:00:00Z");
    assert_eq!(guards(passport), [Guard::OverdueTask { until: held }]);
    for memory in [yoga, garage] {
        assert!(h.show(memory).purge.eligible_now, "{memory}");
    }

    // Once the dinner's day is over its date no longer holds it, though
    // the close's fresh boost lifts it above the line again. The dentist's
    // new slot has been past long enough; its old slot is still ahead, but
    // only the head counts.
    h.set(until);
    assert_eq!(guards(dinner), [Guard::Strength]);
    let rescheduled = h.show(old).purge;
    assert_eq!(rescheduled.head, new);
    assert!(rescheduled.eligible_now, "{rescheduled:?}");

    // δ = 0 purges on fading, and a null δ never purges.
    let projection = |extra: &str| {
        let h = Harness::new(extra);
        let tea = h.says(at(0.0), trivial("fact", "Tim likes green tea."));
        h.set(at(1.0));
        h.show(tea)
    };
    let zero = projection("[purge]\ndelta = 0.0\n");
    assert!(zero.projection.purge.is_some());
    assert_eq!(zero.projection.purge, zero.projection.fade);
    let never = projection("[purge]\ndelta = \"never\"\n");
    assert_eq!(never.projection.purge, None);
    assert!(never.purge.guards.contains(&Guard::PurgeDisabled));
}
