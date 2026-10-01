//! The strength model, phase and state confidence, checked against "Strength
//! model, phase and state confidence as pure functions" (TIM-104) and the
//! decisions it rests on: "Strength model: decay, reinforcement and
//! significance" (TIM-91),
//! "Should memories expire, and how?" (TIM-85, as amended), "What is a
//! memory record?" (TIM-90), "Retrieval and ranking" (TIM-93), "Deletion
//! policy" (TIM-97), and ADRs 0001, 0003, 0004 and 0008.
//!
//! The functions under test don't exist yet. [`contract`] below holds their
//! proposed signatures with `todo!()` bodies, so this file compiles and every
//! test that needs them is ignored. To activate: move the contract into
//! `asphodel_core::strength`, delete the module, import from the crate
//! instead, and drop the `ignore` attributes.
//!
//! The calibration tests at the top run now. They check the tables in TIM-91,
//! ADR 0004 and ADR 0008 against the fixed constants in closed form, so the
//! tables, the constants and the tolerances below are known to agree before
//! any code is written.
//!
//! Tolerances:
//!
//! - [`EXACT`] (1e-9, on the log scale strength lives on) for values worked
//!   by hand from the formula. Each test shows the arithmetic.
//! - [`TABLE`] (5% relative) for the rounded figures in the TIM-91 lifetimes
//!   table and the ADR 0008 purge table: "15 days", "2 months", "9 months",
//!   "3 years", "12 years" and "12.5 years" are 15.09 d, 63.0 d, 262.8 d,
//!   1096.6 d, 4576 d and 4576 d, and the worst of them is 4.5% off.
//! - [`ABOUT`] (10% relative) for ADR 0004's "about 10 weeks, 1.7 years and 7
//!   years", which are 9.3 weeks, 1.70 years and 7.17 years.
//! - [`CROSSING`] (1e-6 relative) between a bisected crossing time and its
//!   closed form.
//!
//! Months are 365.2425 / 12 days and years 365.2425 days, as in
//! [`Volatility::rate_days`].

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use asphodel_core::clock::SimulatedClock;
use asphodel_core::config::Tuning;
use asphodel_core::constants::{
    A, C, D_MAX, FLOOR_SPACING_DAYS, G, MIN_ACCESS_AGE_DAYS, N0, S, SIGNIFICANCE_KEPT,
    Significance, TAU, Volatility, WEIGHT_CONFIRMED, WEIGHT_CREATED, WEIGHT_MENTIONED_AGAIN,
    WEIGHT_USED, WEIGHT_WINDOW_CLOSE,
};
use asphodel_core::store::{OpenOptions, Store};
use jiff::tz::TimeZone;
use jiff::{SignedDuration, Timestamp};

use contract::*;

const EXACT: f64 = 1e-9;
const TABLE: f64 = 0.05;
const ABOUT: f64 = 0.10;
const CROSSING: f64 = 1e-6;

const DAYS_PER_MONTH: f64 = 365.2425 / 12.0;
const DAYS_PER_YEAR: f64 = 365.2425;

/// Far enough out that recent use has sunk below every floor in these
/// fixtures, and still inside jiff's range.
const FAR_DAYS: f64 = 1000.0 * DAYS_PER_YEAR;

/// δ as ADR 0008 starts it.
const DELTA: f64 = 1.0;

/// The proposed `asphodel_core::strength` API. Everything here is a pure
/// function of stored facts and `now`; nothing reads a clock or the store.
#[allow(dead_code, unused_variables)]
mod contract {
    use std::collections::BTreeSet;

    use asphodel_core::config::Tuning;
    use asphodel_core::constants::Volatility;
    use jiff::Timestamp;
    use jiff::tz::TimeZone;

    /// The four kinds of access (ADR 0001). Each has its weight in recent use
    /// from `constants`; the floor ignores the kind.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum AccessKind {
        Created,
        Used,
        MentionedAgain,
        Confirmed,
    }

    /// One row of the access log: its kind and when it happened, in world
    /// time.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Access {
        pub kind: AccessKind,
        pub at: Timestamp,
    }

    /// One bank's clock for strength (ADR 0004), built from the times of its
    /// turns. Documents don't start it. Bank time runs at full speed (1.0)
    /// inside the union of `[turn, turn + 24 h)` over every turn, and at
    /// `quiet_rate` everywhere else, so it never runs faster than world time
    /// however many turns there are.
    pub struct BankTime {}

    impl BankTime {
        /// `turns` in any order; duplicates are harmless. `quiet_rate` is
        /// `Tuning::clock.quiet_rate`, in (0, 1].
        pub fn new(turns: &[Timestamp], quiet_rate: f64) -> Self {
            todo!()
        }

        /// Bank days from `from` to `to`, or 0 when `to <= from`. Additive:
        /// `elapsed(a, b) + elapsed(b, c) == elapsed(a, c)`.
        pub fn elapsed_days(&self, from: Timestamp, to: Timestamp) -> f64 {
            todo!()
        }
    }

    /// A validity window's close, for restarting recent use (ADR 0003).
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct WindowClose {
        /// When the window closed: [`Window::closes_at`], so already the end
        /// of `valid_until`'s unit in the source's timezone.
        pub closes_at: Timestamp,
        /// When the end became known: the `observed_at` of the `ended_by`
        /// memory for a late-reported end, or the memory's own `observed_at`
        /// when the end was stated up front.
        pub known_at: Timestamp,
    }

    /// A memory's strength and its parts. Never stored.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Strength {
        /// `S·significance + max(recent_use, lasting_floor)`.
        pub value: f64,
        pub recent_use: f64,
        pub lasting_floor: f64,
        /// n: accesses counted as separate occasions for the floor.
        pub occasions: u32,
    }

    /// Strength at `now` (TIM-91, ADR 0003, ADR 0004).
    ///
    /// - `significance` is the value, not the level: kept is 1.0.
    /// - `accesses` is the memory's own log plus everything it inherits
    ///   along `superseded_by` ([`inherits_from`]), in any order. Accesses
    ///   after `now` are ignored.
    /// - Recent use is `ln Σ w_j · age_j^(−d_j)`, ages in bank days from
    ///   `bank_time`, at least `MIN_ACCESS_AGE_DAYS`. `d_j = min(D_MAX, a +
    ///   c·e^m)`, where m is recent use over the earlier counted accesses at
    ///   the time of access j, and −∞ (so d = a) for the first.
    /// - When `close` is given and `max(closes_at, known_at) <= now`, recent
    ///   use counts only one synthetic access of weight `WEIGHT_WINDOW_CLOSE`
    ///   at `max(closes_at, known_at)` plus the accesses strictly after
    ///   `closes_at`. Otherwise it counts every access.
    /// - The floor counts occasions over every access, closed window or not:
    ///   greedily in time order, an access counts when it's at least
    ///   `FLOOR_SPACING_DAYS` world days after the last one counted. The
    ///   synthetic access is never an occasion. `lasting_floor = τ − g·ln(n0)
    ///   + g·ln(n)`.
    pub fn strength(
        significance: f64,
        accesses: &[Access],
        close: Option<WindowClose>,
        bank_time: &BankTime,
        now: Timestamp,
    ) -> Strength {
        todo!()
    }

    /// How exact a stored time is (TIM-90). A time is stored as the UTC
    /// instant at the start of its unit in the source's timezone.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TimePrecision {
        Year,
        Month,
        Day,
        Hour,
        Minute,
    }

    /// A stored time with its precision.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct WorldTime {
        pub at: Timestamp,
        pub precision: TimePrecision,
    }

    /// The end of the unit `time` starts, one unit later in `tz`: calendar
    /// years, months and days (so a day can be 23 or 25 hours), and absolute
    /// hours and minutes.
    pub fn unit_end(time: WorldTime, tz: &TimeZone) -> Timestamp {
        todo!()
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Kind {
        Fact,
        Event,
        State,
        Task,
        Recurring,
    }

    /// The world-time fields of a memory.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Window {
        pub kind: Kind,
        pub valid_from: Option<WorldTime>,
        pub valid_until: Option<WorldTime>,
        pub due_at: Option<WorldTime>,
    }

    /// Where a memory sits relative to now, on world time in the source's
    /// timezone. Computed, never stored.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Phase {
        /// Before the end of `valid_from`'s unit.
        Upcoming,
        /// An open window, an open task that isn't overdue, or no window.
        Current,
        /// An open task past the end of `due_at`'s unit.
        Overdue,
        /// Less than [`RECENTLY_PAST_DAYS`] world days since the window
        /// closed.
        RecentlyPast,
        /// The window closed at least [`RECENTLY_PAST_DAYS`] world days ago.
        LongPast,
    }

    /// Where recently past ends. Proposed: the 30 days over which TIM-93's
    /// phase penalty ramps to its full value. Belongs in `constants`.
    pub const RECENTLY_PAST_DAYS: f64 = 30.0;

    impl Window {
        /// When the window closes: the end of `valid_until`'s unit, or for an
        /// event with no `valid_until`, the end of `valid_from`'s unit (a
        /// point event). `None` while the window has no end.
        pub fn closes_at(&self, tz: &TimeZone) -> Option<Timestamp> {
            todo!()
        }

        /// Upcoming before the end of `valid_from`'s unit; past from
        /// [`Window::closes_at`]; otherwise overdue for a task past the end
        /// of `due_at`'s unit; otherwise current.
        pub fn phase(&self, tz: &TimeZone, now: Timestamp) -> Phase {
            todo!()
        }
    }

    /// How likely a state still holds: `1 / (1 + (age / T)²)` with T from
    /// [`Volatility::rate_days`], age in world days from the later of
    /// `observed_at` and the last `mentioned_again` or `confirmed` access at
    /// or before `now`, and never negative. 1.0 when `volatility` is `None`.
    pub fn state_confidence(
        volatility: Option<Volatility>,
        observed_at: Timestamp,
        accesses: &[Access],
        now: Timestamp,
    ) -> f64 {
        todo!()
    }

    /// One memory's links, by rowid.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Link {
        pub id: i64,
        pub superseded_by: Option<i64>,
        pub ended_by: Option<i64>,
    }

    /// The head of the supersession chain `id` is in: follow
    /// `superseded_by` until it's `None`. `ended_by` is never followed.
    pub fn chain_head(links: &[Link], id: i64) -> i64 {
        todo!()
    }

    /// Every memory connected to `id` along `superseded_by`, in either
    /// direction. A tree, because two memories can be refined into one.
    pub fn chain(links: &[Link], id: i64) -> BTreeSet<i64> {
        todo!()
    }

    /// `head` and every memory that reaches it along `superseded_by`: the
    /// memories whose accesses `head`'s strength counts (TIM-91, decision
    /// 3).
    pub fn inherits_from(links: &[Link], head: i64) -> BTreeSet<i64> {
        todo!()
    }

    /// The purge settings (ADR 0008).
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct PurgeRule {
        /// δ; `None` never purges.
        pub delta: Option<f64>,
        /// The task guard: `agenda.overdue_days`.
        pub overdue_days: u32,
    }

    impl PurgeRule {
        pub fn from_tuning(tuning: &Tuning) -> Self {
            todo!()
        }
    }

    /// Whether a chain may be purged, read on its head: `head_strength <
    /// τ − δ`, unless δ is `None`, the end of `head.valid_from`'s or
    /// `head.valid_until`'s unit is after `now`, or `head` is a task and
    /// `now` is before the end of `due_at`'s unit plus `overdue_days` × 24 h.
    pub fn purge_eligible(
        rule: &PurgeRule,
        head_strength: f64,
        head: &Window,
        tz: &TimeZone,
        now: Timestamp,
    ) -> bool {
        todo!()
    }
}

// Fixtures.

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

fn tz(name: &str) -> TimeZone {
    TimeZone::get(name).unwrap()
}

fn access(kind: AccessKind, days: f64) -> Access {
    Access { kind, at: at(days) }
}

fn created(days: f64) -> Access {
    access(AccessKind::Created, days)
}

fn used(days: f64) -> Access {
    access(AccessKind::Used, days)
}

fn mentioned(days: f64) -> Access {
    access(AccessKind::MentionedAgain, days)
}

fn confirmed(days: f64) -> Access {
    access(AccessKind::Confirmed, days)
}

/// One access every `FLOOR_SPACING_DAYS`, the first a `created`: `n`
/// separate occasions.
fn occasions(n: u32) -> Vec<Access> {
    (0..n)
        .map(|i| {
            let day = f64::from(i) * FLOOR_SPACING_DAYS;
            if i == 0 { created(day) } else { used(day) }
        })
        .collect()
}

/// Bank time equal to world time, so ages in the tables read as days.
fn always_on() -> BankTime {
    BankTime::new(&[], 1.0)
}

/// The default quiet rate with turns at the given world days.
fn quiet_bank(turn_days: &[f64]) -> BankTime {
    let turns: Vec<_> = turn_days.iter().map(|&d| at(d)).collect();
    BankTime::new(&turns, Tuning::default().clock.quiet_rate)
}

fn strength_at(significance: f64, accesses: &[Access], days: f64) -> Strength {
    strength(significance, accesses, None, &always_on(), at(days))
}

fn wt(text: &str, precision: TimePrecision) -> WorldTime {
    WorldTime {
        at: ts(text),
        precision,
    }
}

fn window(kind: Kind) -> Window {
    Window {
        kind,
        valid_from: None,
        valid_until: None,
        due_at: None,
    }
}

fn rule(delta: Option<f64>) -> PurgeRule {
    PurgeRule {
        delta,
        overdue_days: 30,
    }
}

#[track_caller]
fn assert_near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected} ± {tolerance}, got {actual}"
    );
}

#[track_caller]
fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
    let off = (actual / expected - 1.0).abs();
    assert!(
        off <= tolerance,
        "expected {expected} within {}%, got {actual} ({:.2}% off)",
        tolerance * 100.0,
        off * 100.0
    );
}

/// The first day in `[lo, hi]` at which `value` is below `threshold`, by
/// bisection. `value` must cross once, downwards.
fn crossing(lo: f64, hi: f64, threshold: f64, value: impl Fn(f64) -> f64) -> f64 {
    assert!(value(lo) >= threshold, "already below at {lo}");
    assert!(value(hi) < threshold, "still above at {hi}");
    let (mut lo, mut hi) = (lo, hi);
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if value(mid) < threshold {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

// Closed forms, for the calibration tests only.

/// The floor with n separate occasions.
fn floor_with(n: u32) -> f64 {
    TAU - G * N0.ln() + G * f64::from(n).ln()
}

/// Bank days until a memory mentioned once falls below `threshold`:
/// `S·σ − a·ln t = threshold`, while the floor is lower still.
fn one_mention_days(significance: f64, threshold: f64) -> f64 {
    ((S * significance - threshold) / A).exp()
}

/// The fewest separate occasions that keep `S·σ + floor` at or above
/// `threshold`, if any number up to 1000 does.
fn occasions_to_hold(significance: f64, threshold: f64) -> u32 {
    (1..=1000)
        .find(|&n| S * significance + floor_with(n) >= threshold)
        .unwrap()
}

/// TIM-91's lifetimes table: significance, and how long one mention stays
/// in recall, in bank days.
const LIFETIMES: [(f64, f64); 5] = [
    (0.1, 15.0),
    (0.3, 2.0 * DAYS_PER_MONTH),
    (0.5, 9.0 * DAYS_PER_MONTH),
    (0.7, 3.0 * DAYS_PER_YEAR),
    (0.9, 12.0 * DAYS_PER_YEAR),
];

/// ADR 0008's purge table: significance, how long until one mention is
/// purged in bank days (`None` for never), and the separate occasions that
/// make it unpurgeable.
const PURGES: [(Significance, Option<f64>, u32); 5] = [
    (Significance::Trivial, Some(9.0 * DAYS_PER_MONTH), 4),
    (Significance::Minor, Some(3.0 * DAYS_PER_YEAR), 3),
    (Significance::Notable, Some(12.5 * DAYS_PER_YEAR), 2),
    (Significance::Major, None, 1),
    (Significance::Critical, None, 1),
];

/// TIM-91's permanence figures: significance and the separate occasions
/// that lift the floor to τ.
const PERMANENCE: [(f64, u32); 3] = [(0.0, 18), (0.3, 8), (0.5, 4)];

/// ADR 0004: significance, and how long one mention lasts in an abandoned
/// bank, in world days.
const ABANDONED: [(f64, f64); 3] = [
    (0.0, 10.0 * 7.0),
    (0.3, 1.7 * DAYS_PER_YEAR),
    (0.5, 7.0 * DAYS_PER_YEAR),
];

/// World days until a memory created in a bank's last turn fades out: one
/// bank day at full speed, then the quiet rate.
fn abandoned_days(significance: f64) -> f64 {
    let quiet_rate = Tuning::default().clock.quiet_rate;
    1.0 + (one_mention_days(significance, TAU) - 1.0) / quiet_rate
}

// Calibration: the tables against the constants, in closed form. These run
// now.

#[test]
fn the_lifetimes_table_matches_the_constants() {
    for (significance, days) in LIFETIMES {
        assert_relative(one_mention_days(significance, TAU), days, TABLE);
    }
}

#[test]
fn the_purge_table_matches_the_constants() {
    let purge_at = TAU - DELTA;
    for (level, days, hold) in PURGES {
        let significance = level.value();
        match days {
            Some(days) => {
                assert!(S * significance + floor_with(1) < purge_at, "{level:?}");
                assert_relative(one_mention_days(significance, purge_at), days, TABLE);
            }
            None => assert!(S * significance + floor_with(1) >= purge_at, "{level:?}"),
        }
        assert_eq!(occasions_to_hold(significance, purge_at), hold, "{level:?}");
    }
}

#[test]
fn a_memory_sits_faded_about_17_times_as_long_as_it_was_in_recall() {
    // ADR 0008: e^(δ/a).
    for (significance, _) in LIFETIMES {
        let ratio =
            one_mention_days(significance, TAU - DELTA) / one_mention_days(significance, TAU);
        assert_relative(ratio, 17.0, TABLE);
    }
}

#[test]
fn the_permanence_figures_match_the_constants() {
    for (significance, n) in PERMANENCE {
        assert_eq!(
            occasions_to_hold(significance, TAU),
            n,
            "significance {significance}"
        );
    }
    // Permanent from creation at significance 0.925 or above.
    let permanent_from = (TAU - floor_with(1)) / S;
    assert_near(permanent_from, 0.925, 0.001);
    assert!(SIGNIFICANCE_KEPT >= permanent_from);
    assert!(Significance::Critical.value() < permanent_from);
}

#[test]
fn abandoned_bank_lifetimes_match_adr_0004() {
    for (significance, days) in ABANDONED {
        assert_relative(abandoned_days(significance), days, ABOUT);
    }
}

// Strength is never stored.

/// A temporary data dir removed even when an assertion unwinds.
struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "asphodel-strength-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn no_column_stores_strength_or_its_parts() {
    let dir = TestDir::new();
    let clock = Arc::new(SimulatedClock::new(t0()));
    let store = Store::open(&dir.0.join("data"), OpenOptions::default(), clock).unwrap();
    let conn = store.connection();
    let mut tables = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table'")
        .unwrap();
    let tables: Vec<String> = tables
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert!(tables.iter().any(|t| t == "memories"));

    let mut stored = Vec::new();
    for table in tables {
        let mut columns = conn
            .prepare(&format!("SELECT name FROM pragma_table_info('{table}')"))
            .unwrap();
        for column in columns
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
        {
            let column = column.unwrap();
            let lower = column.to_lowercase();
            if [
                "strength",
                "recent_use",
                "lasting_floor",
                "occasions",
                "activation",
                "faded",
                "phase",
                "confidence",
            ]
            .iter()
            .any(|word| lower.contains(word))
                && column != "window_confidence"
            {
                stored.push(format!("{table}.{column}"));
            }
        }
    }
    assert!(stored.is_empty(), "computed values are stored: {stored:?}");
}

// Bank time (ADR 0004).

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_bank_with_no_turns_runs_at_the_quiet_rate() {
    assert_near(quiet_bank(&[]).elapsed_days(at(0.0), at(10.0)), 1.0, EXACT);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn bank_time_runs_at_full_speed_for_24_hours_after_a_turn() {
    let bank = quiet_bank(&[0.0]);
    assert_near(bank.elapsed_days(at(0.0), at(0.5)), 0.5, EXACT);
    // 1 day at full speed, then 9 at 0.1.
    assert_near(bank.elapsed_days(at(0.0), at(10.0)), 1.9, EXACT);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn the_full_speed_window_ends_exactly_24_hours_after_the_turn() {
    let bank = quiet_bank(&[0.0]);
    assert_near(bank.elapsed_days(at(1.0), at(2.0)), 0.1, EXACT);
    let minute = 1.0 / 1440.0;
    assert_near(bank.elapsed_days(at(1.0 - minute), at(1.0)), minute, EXACT);
    assert_near(
        bank.elapsed_days(at(1.0), at(1.0 + minute)),
        0.1 * minute,
        EXACT,
    );
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn overlapping_windows_merge_and_never_run_faster_than_world_time() {
    // Windows [0, 1) and [0.5, 1.5) cover 1.5 days.
    let bank = quiet_bank(&[0.0, 0.5]);
    assert_near(bank.elapsed_days(at(0.0), at(10.0)), 1.5 + 0.1 * 8.5, EXACT);

    // A turn every hour for ten days: full speed throughout, and no faster.
    let hourly: Vec<f64> = (0..240).map(|h| f64::from(h) / 24.0).collect();
    assert_near(
        quiet_bank(&hourly).elapsed_days(at(0.0), at(10.0)),
        10.0,
        EXACT,
    );

    // How many turns there were doesn't matter, only when.
    assert_near(
        quiet_bank(&[0.0, 0.0, 0.0]).elapsed_days(at(0.0), at(10.0)),
        quiet_bank(&[0.0]).elapsed_days(at(0.0), at(10.0)),
        EXACT,
    );
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_turn_before_the_interval_speeds_up_its_start_and_later_turns_dont_count() {
    assert_near(
        quiet_bank(&[0.0]).elapsed_days(at(0.5), at(2.0)),
        0.5 + 0.1,
        EXACT,
    );
    assert_near(
        quiet_bank(&[5.0]).elapsed_days(at(0.0), at(2.0)),
        0.2,
        EXACT,
    );
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn bank_time_is_additive_and_ignores_turn_order() {
    let bank = quiet_bank(&[0.0, 3.2, 3.7, 9.0]);
    let whole = bank.elapsed_days(at(0.0), at(12.0));
    let parts = bank.elapsed_days(at(0.0), at(5.0)) + bank.elapsed_days(at(5.0), at(12.0));
    assert_near(parts, whole, EXACT);
    assert_near(bank.elapsed_days(at(4.0), at(4.0)), 0.0, EXACT);
    assert_near(bank.elapsed_days(at(5.0), at(4.0)), 0.0, EXACT);

    let shuffled = quiet_bank(&[9.0, 3.7, 0.0, 3.2]);
    assert_near(shuffled.elapsed_days(at(0.0), at(12.0)), whole, EXACT);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_quiet_rate_of_one_is_world_time() {
    assert_near(always_on().elapsed_days(at(0.0), at(123.25)), 123.25, EXACT);
}

// Recent use and the lasting floor (TIM-91).

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn one_access_fades_as_a_power_of_its_age() {
    // d = a for the first access: recent_use = −0.35·ln 10.
    let s = strength_at(0.1, &[created(0.0)], 10.0);
    assert_near(s.recent_use, -A * 10f64.ln(), EXACT);
    assert_near(s.recent_use, -0.805_904_782_547_916, EXACT);
    assert_eq!(s.occasions, 1);
    assert_near(s.lasting_floor, -3.012_297_406_316_931, EXACT);
    assert_near(s.value, 0.25 - 0.805_904_782_547_916, EXACT);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn each_access_kind_counts_with_its_weight() {
    // At an age of 1 day, age^−d = 1 and recent_use = ln w.
    for (kind, weight) in [
        (AccessKind::Created, WEIGHT_CREATED),
        (AccessKind::Used, WEIGHT_USED),
        (AccessKind::MentionedAgain, WEIGHT_MENTIONED_AGAIN),
        (AccessKind::Confirmed, WEIGHT_CONFIRMED),
    ] {
        let s = strength_at(0.0, &[access(kind, 0.0)], 1.0);
        assert_near(s.recent_use, weight.ln(), EXACT);
    }
    assert_eq!(
        [
            WEIGHT_CREATED,
            WEIGHT_USED,
            WEIGHT_MENTIONED_AGAIN,
            WEIGHT_CONFIRMED
        ],
        [1.0, 1.0, 1.5, 2.0]
    );
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_fresh_access_counts_at_the_minimum_age() {
    let expected = -A * MIN_ACCESS_AGE_DAYS.ln();
    assert_near(expected, 1.611_809_565_095_831_7, EXACT);
    assert_near(
        strength_at(0.0, &[created(5.0)], 5.0).recent_use,
        expected,
        EXACT,
    );
    // A minute is younger than the minimum.
    let minute = 1.0 / 1440.0;
    assert_near(
        strength_at(0.0, &[created(5.0)], 5.0 + minute).recent_use,
        expected,
        EXACT,
    );
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn an_access_made_while_fresh_fades_faster() {
    // Pavlik spacing. The second access comes when m = ln(1^−0.35) = 0, so
    // d₂ = 0.35 + 0.2·e⁰ = 0.55. At day 2: ln(1·2^−0.35 + 1.5·1^−0.55).
    let s = strength_at(0.0, &[created(0.0), mentioned(1.0)], 2.0);
    assert_near(s.recent_use, (2f64.powf(-A) + 1.5).ln(), EXACT);
    assert_near(s.recent_use, 0.826_183_993_730_038_7, EXACT);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn the_decay_rate_is_capped() {
    // Two confirmations 0.01 days apart: m = ln(2·0.01^−0.35) ≈ 2.30, so
    // a + c·e^m ≈ 2.35 is capped at D_MAX = 2. At day 1:
    // ln(2·1^−0.35 + 2·0.99^−2).
    assert_eq!(D_MAX, 2.0);
    let m = (2.0 * MIN_ACCESS_AGE_DAYS.powf(-A)).ln();
    assert!(A + C * m.exp() > D_MAX);
    let s = strength_at(0.0, &[confirmed(0.0), confirmed(0.01)], 1.0);
    assert_near(s.recent_use, (2.0 + 2.0 * 0.99f64.powi(-2)).ln(), EXACT);
    assert_near(s.recent_use, 1.396_395_200_748_559_8, EXACT);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn massed_use_spikes_then_falls_below_spaced_use() {
    // TIM-91: 15 uses in one day against 5 uses a week apart.
    let massed: Vec<_> = std::iter::once(created(0.0))
        .chain((1..15).map(|h| used(f64::from(h) / 24.0)))
        .collect();
    let spaced: Vec<_> = std::iter::once(created(0.0))
        .chain((1..5).map(|w| used(7.0 * f64::from(w))))
        .collect();

    let (m, s) = (
        strength_at(0.0, &massed, 1.0),
        strength_at(0.0, &spaced, 1.0),
    );
    assert!(m.recent_use > s.recent_use, "{m:?} {s:?}");
    assert_near(m.recent_use, 3.611_502_811_239_257_8, EXACT);
    assert_near(s.recent_use, 0.0, EXACT);

    let (m, s) = (
        strength_at(0.0, &massed, 90.0),
        strength_at(0.0, &spaced, 90.0),
    );
    assert!(m.recent_use < s.recent_use, "{m:?} {s:?}");
    assert_near(m.recent_use, -1.504_853_433_428_120_5, EXACT);
    assert_near(s.recent_use, -0.449_286_690_871_333_2, EXACT);
    assert_eq!((m.occasions, s.occasions), (1, 5));
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn accesses_after_now_are_ignored() {
    let alone = strength_at(0.3, &[created(0.0)], 10.0);
    let with_later = strength_at(0.3, &[created(0.0), confirmed(20.0)], 10.0);
    assert_eq!(alone, with_later);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn recent_use_ages_on_bank_time() {
    // Created in a turn, then 10 quiet days: 1 + 10·0.1 = 2 bank days.
    let s = strength(0.0, &[created(0.0)], None, &quiet_bank(&[0.0]), at(11.0));
    assert_near(s.recent_use, -A * 2f64.ln(), EXACT);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn the_floor_counts_occasions_at_least_three_world_days_apart() {
    // ADR 0003: daily for a week is about 3 occasions, every ten days for
    // three months is 10.
    let week: Vec<_> = (0..7).map(|d| used(f64::from(d))).collect();
    assert_eq!(strength_at(0.0, &week, 7.0).occasions, 3);
    let months: Vec<_> = (0..10).map(|i| used(10.0 * f64::from(i))).collect();
    assert_eq!(strength_at(0.0, &months, 91.0).occasions, 10);

    // Greedy from the last counted access; exactly 3 days counts.
    let edges = [created(0.0), used(2.99), used(3.0), used(5.99), used(6.0)];
    assert_eq!(strength_at(0.0, &edges, 7.0).occasions, 3);
    assert_eq!(FLOOR_SPACING_DAYS, 3.0);

    // The kind doesn't matter.
    let kinds = [created(0.0), mentioned(3.0), confirmed(6.0), used(9.0)];
    assert_eq!(strength_at(0.0, &kinds, 10.0).occasions, 4);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn floor_spacing_is_world_time_even_in_a_quiet_bank() {
    // Three world days apart is 0.3 bank days in a quiet bank.
    let s = strength(0.0, &occasions(3), None, &quiet_bank(&[]), at(10.0));
    assert_eq!(s.occasions, 3);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn the_floor_follows_the_occasion_count_and_never_goes_down() {
    for n in [1, 2, 3, 4, 8, 18] {
        let s = strength_at(0.0, &occasions(n), FAR_DAYS);
        assert_eq!(s.occasions, n);
        assert_near(
            s.lasting_floor,
            TAU - G * N0.ln() + G * f64::from(n).ln(),
            EXACT,
        );
    }
    assert_near(
        strength_at(0.0, &occasions(3), FAR_DAYS).lasting_floor,
        -2.133_407_575_382_443_5,
        EXACT,
    );
    assert_near(
        strength_at(0.0, &occasions(18), FAR_DAYS).lasting_floor,
        TAU,
        EXACT,
    );

    let accesses = occasions(6);
    let mut previous = f64::NEG_INFINITY;
    for day in [
        0.0, 1.0, 3.0, 4.0, 10.0, 15.0, 16.0, 100.0, 10_000.0, FAR_DAYS,
    ] {
        let floor = strength_at(0.0, &accesses, day).lasting_floor;
        assert!(floor >= previous, "the floor went down at day {day}");
        previous = floor;
    }
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn strength_is_significance_plus_the_larger_part() {
    for significance in [0.0, 0.1, 0.5, SIGNIFICANCE_KEPT] {
        for day in [0.0, 2.0, 30.0, FAR_DAYS] {
            let s = strength_at(significance, &occasions(5), day);
            let expected = S * significance + s.recent_use.max(s.lasting_floor);
            assert_near(s.value, expected, EXACT);
        }
    }
    // Far out, the floor is the larger part.
    let s = strength_at(0.0, &occasions(5), FAR_DAYS);
    assert!(s.lasting_floor > s.recent_use);
    assert_near(s.value, s.lasting_floor, EXACT);
}

// Lifetimes on the strength function itself.

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn one_mention_lifetimes_reproduce_tim_91() {
    for (significance, days) in LIFETIMES {
        let fades = crossing(0.0, 200.0 * DAYS_PER_YEAR, TAU, |day| {
            strength_at(significance, &[created(0.0)], day).value
        });
        assert_relative(fades, one_mention_days(significance, TAU), CROSSING);
        assert_relative(fades, days, TABLE);
    }
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn one_mention_lifetimes_in_an_abandoned_bank_reproduce_adr_0004() {
    let bank = quiet_bank(&[0.0]);
    for (significance, days) in ABANDONED {
        let fades = crossing(0.0, 200.0 * DAYS_PER_YEAR, TAU, |day| {
            strength(significance, &[created(0.0)], None, &bank, at(day)).value
        });
        assert_relative(fades, abandoned_days(significance), CROSSING);
        assert_relative(fades, days, ABOUT);
    }
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn enough_separate_occasions_make_a_memory_permanent() {
    for (significance, n) in PERMANENCE {
        let held = strength_at(significance, &occasions(n), FAR_DAYS);
        assert!(
            held.value >= TAU,
            "{n} occasions at {significance}: {held:?}"
        );
        let fades = strength_at(significance, &occasions(n - 1), FAR_DAYS);
        assert!(
            fades.value < TAU,
            "{} occasions at {significance}: {fades:?}",
            n - 1
        );
    }
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_kept_memory_never_fades_and_a_critical_one_does() {
    let kept = strength_at(SIGNIFICANCE_KEPT, &[created(0.0)], FAR_DAYS);
    assert!(kept.value >= TAU, "{kept:?}");
    let critical = Significance::Critical.value();
    let fades = crossing(0.0, 200.0 * DAYS_PER_YEAR, TAU, |day| {
        strength_at(critical, &[created(0.0)], day).value
    });
    assert_relative(fades, 12.0 * DAYS_PER_YEAR, TABLE);
}

// Window close (ADR 0003).

fn close(closes: f64, known: f64) -> Option<WindowClose> {
    Some(WindowClose {
        closes_at: at(closes),
        known_at: at(known),
    })
}

fn closed_strength(
    significance: f64,
    accesses: &[Access],
    close: Option<WindowClose>,
    day: f64,
) -> Strength {
    strength(significance, accesses, close, &always_on(), at(day))
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn closing_a_window_restarts_recent_use_from_one_synthetic_access() {
    // The end was known from the start, so the synthetic access sits at the
    // close, day 10. At day 12 it's 2 days old and the only one counted.
    let accesses = [created(0.0), mentioned(2.0), mentioned(4.0)];
    let s = closed_strength(0.1, &accesses, close(10.0, 0.0), 12.0);
    assert_eq!(WEIGHT_WINDOW_CLOSE, 1.0);
    assert_near(s.recent_use, -A * 2f64.ln(), EXACT);
    assert_near(s.recent_use, -0.242_601_513_195_980_83, EXACT);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_late_reported_end_places_the_synthetic_access_when_it_became_known() {
    // Valid until day 30, reported on day 40, used on day 35 in between.
    // Counted: used@35 (d = a), then the synthetic access @40, which comes
    // when m = −0.35·ln 5, so d = 0.35 + 0.2·5^−0.35. At day 45:
    // ln(10^−0.35 + 5^−d).
    let accesses = [created(0.0), used(20.0), used(35.0)];
    let s = closed_strength(0.3, &accesses, close(30.0, 40.0), 45.0);
    let d = A + C * 5f64.powf(-A);
    assert_near(s.recent_use, (10f64.powf(-A) + 5f64.powf(-d)).ln(), EXACT);
    assert_near(s.recent_use, -0.082_646_089_892_332_44, EXACT);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn accesses_after_the_close_count_and_one_at_the_close_does_not() {
    // Strictly after: an access at the close itself is inside the window.
    let at_close = closed_strength(0.0, &[created(0.0), used(10.0)], close(10.0, 0.0), 12.0);
    assert_near(at_close.recent_use, -A * 2f64.ln(), EXACT);

    // Synthetic@10 (d = a), then mentioned@11 with m = 0 and d = 0.55. At
    // day 12 that's the two-access fixture again.
    let after = closed_strength(
        0.0,
        &[created(0.0), mentioned(11.0)],
        close(10.0, 0.0),
        12.0,
    );
    assert_near(after.recent_use, 0.826_183_993_730_038_7, EXACT);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_window_that_hasnt_closed_or_isnt_known_to_have_closed_counts_everything() {
    let accesses = [created(0.0), used(4.0), mentioned(8.0)];
    let open = closed_strength(0.3, &accesses, None, 9.0);
    // Closes on day 20.
    assert_eq!(closed_strength(0.3, &accesses, close(20.0, 0.0), 9.0), open);
    // Closed on day 5, but only reported on day 12.
    assert_eq!(closed_strength(0.3, &accesses, close(5.0, 12.0), 9.0), open);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn the_close_never_lowers_the_floor_and_is_not_an_occasion() {
    let accesses: Vec<_> = (0..10).map(|i| used(10.0 * f64::from(i))).collect();
    let open = closed_strength(0.3, &accesses, None, FAR_DAYS);
    let ended = closed_strength(0.3, &accesses, close(95.0, 100.0), FAR_DAYS);
    assert_eq!(ended.occasions, open.occasions);
    assert_near(ended.lasting_floor, open.lasting_floor, EXACT);

    // The synthetic access 5 days after the only access adds no occasion.
    let s = closed_strength(0.1, &[created(0.0)], close(5.0, 0.0), 6.0);
    assert_eq!(s.occasions, 1);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_past_appointment_comes_back_to_recall_when_its_window_closes() {
    // TIM-85: "how did the dentist go?" works the next day. A trivial
    // appointment mentioned once, 90 days ahead.
    let trivial = Significance::Trivial.value();
    let accesses = [created(0.0)];
    let before = closed_strength(trivial, &accesses, close(90.0, 0.0), 89.0);
    assert!(before.value < TAU, "{before:?}");
    assert_near(before.value, 0.25 - A * 89f64.ln(), EXACT);
    let after = closed_strength(trivial, &accesses, close(90.0, 0.0), 91.0);
    assert_near(after.value, 0.25, EXACT);

    // Then it fades like any fresh trivial memory, about 15 days later.
    let fades = crossing(91.0, 200.0, TAU, |day| {
        closed_strength(trivial, &accesses, close(90.0, 0.0), day).value
    });
    assert_relative(fades - 90.0, one_mention_days(trivial, TAU), CROSSING);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn an_appointment_discussed_daily_for_a_week_fades_and_can_be_purged() {
    // ADR 0003: about 3 occasions, so it fades.
    let trivial = Significance::Trivial.value();
    let accesses: Vec<_> = (0..7).map(|d| mentioned(f64::from(d))).collect();
    let s = closed_strength(trivial, &accesses, close(7.0, 0.0), FAR_DAYS);
    assert_eq!(s.occasions, 3);
    assert!(s.value < TAU - DELTA, "{s:?}");
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_state_used_for_three_months_stays_in_history_after_it_ends() {
    // ADR 0003: "User lived in Berlin", minor, used every ten days for three
    // months, ended on day 95 and reported on day 100.
    let minor = Significance::Minor.value();
    let accesses: Vec<_> = std::iter::once(created(0.0))
        .chain((1..10).map(|i| used(10.0 * f64::from(i))))
        .collect();
    let s = closed_strength(minor, &accesses, close(95.0, 100.0), FAR_DAYS);
    assert_eq!(s.occasions, 10);
    assert!(s.value >= TAU, "{s:?}");
    assert_near(s.value, 0.75 + floor_with(10), EXACT);
}

// Inheritance (TIM-91, decision 3).

const MAYA: i64 = 1;
const MIA: i64 = 2;

fn link(id: i64, superseded_by: Option<i64>, ended_by: Option<i64>) -> Link {
    Link {
        id,
        superseded_by,
        ended_by,
    }
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn correcting_maya_to_mia_keeps_the_corrected_name_as_strong() {
    // "Maya" is created, used twice and mentioned again; on day 20 the user
    // corrects it, and "Mia" retracts it.
    let links = [link(MAYA, Some(MIA), None), link(MIA, None, None)];
    assert_eq!(chain_head(&links, MAYA), MIA);
    assert_eq!(inherits_from(&links, MIA), BTreeSet::from([MAYA, MIA]));

    let minor = Significance::Minor.value();
    let maya = [created(0.0), used(3.0), used(7.0), mentioned(10.0)];
    let mia_own = [created(20.0)];
    let mia: Vec<_> = maya.iter().chain(&mia_own).copied().collect();

    let maya_then = strength_at(minor, &maya, 20.0);
    assert_near(maya_then.recent_use, 0.182_036_993_685_896_5, EXACT);
    let mia_next_day = strength_at(minor, &mia, 21.0);
    assert_near(mia_next_day.recent_use, 0.768_503_277_496_431_8, EXACT);
    assert!(mia_next_day.value >= maya_then.value);
    assert_eq!(mia_next_day.occasions, 5);

    for day in [20.0, 20.5, 21.0, 30.0, 100.0, 1000.0, FAR_DAYS] {
        let wrong = strength_at(minor, &maya, day);
        let corrected = strength_at(minor, &mia, day);
        assert!(
            corrected.value >= wrong.value,
            "day {day}: {corrected:?} < {wrong:?}"
        );
        assert!(corrected.value > strength_at(minor, &mia_own, day).value);
    }
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn inheritance_follows_chains_of_several_and_trees() {
    // 1 → 2 → 3.
    let line = [
        link(1, Some(2), None),
        link(2, Some(3), None),
        link(3, None, None),
    ];
    assert_eq!(chain_head(&line, 1), 3);
    assert_eq!(inherits_from(&line, 3), BTreeSet::from([1, 2, 3]));
    assert_eq!(inherits_from(&line, 2), BTreeSet::from([1, 2]));
    assert_eq!(chain(&line, 2), BTreeSet::from([1, 2, 3]));

    // 1 and 2 refined into 3.
    let tree = [
        link(1, Some(3), None),
        link(2, Some(3), None),
        link(3, None, None),
    ];
    assert_eq!(chain_head(&tree, 2), 3);
    assert_eq!(inherits_from(&tree, 3), BTreeSet::from([1, 2, 3]));
    assert_eq!(chain(&tree, 1), BTreeSet::from([1, 2, 3]));
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn nothing_is_inherited_along_ended_by() {
    // "Lives in Berlin" (1) is ended by "moved to Lisbon" (2). An old
    // reschedule (3 retracted by 4) sits beside them.
    let links = [
        link(1, None, Some(2)),
        link(2, None, None),
        link(3, Some(4), None),
        link(4, None, Some(2)),
    ];
    assert_eq!(inherits_from(&links, 2), BTreeSet::from([2]));
    assert_eq!(chain_head(&links, 1), 1);
    assert_eq!(chain(&links, 1), BTreeSet::from([1]));
    assert_eq!(chain(&links, 2), BTreeSet::from([2]));
    assert_eq!(chain(&links, 3), BTreeSet::from([3, 4]));
}

// Time precision and phase (TIM-90, TIM-85), on world time in the source's
// timezone.

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn each_precision_ends_one_unit_later_in_the_source_timezone() {
    use TimePrecision::*;
    let cases = [
        // 3 Oct 2026 in Auckland (NZDT, +13).
        (
            "2026-10-02T11:00:00Z",
            Day,
            "Pacific/Auckland",
            "2026-10-03T11:00:00Z",
        ),
        // London's spring-forward day is 23 hours...
        (
            "2026-03-29T00:00:00Z",
            Day,
            "Europe/London",
            "2026-03-29T23:00:00Z",
        ),
        // ...and its fall-back day 25.
        (
            "2026-10-24T23:00:00Z",
            Day,
            "Europe/London",
            "2026-10-26T00:00:00Z",
        ),
        (
            "2026-03-01T00:00:00Z",
            Month,
            "Europe/London",
            "2026-03-31T23:00:00Z",
        ),
        ("2028-02-01T00:00:00Z", Month, "UTC", "2028-03-01T00:00:00Z"),
        (
            "2026-12-31T11:00:00Z",
            Year,
            "Pacific/Auckland",
            "2027-12-31T11:00:00Z",
        ),
        // 01:00 BST on the fall-back day: an hour is an hour.
        (
            "2026-10-25T00:00:00Z",
            Hour,
            "Europe/London",
            "2026-10-25T01:00:00Z",
        ),
        (
            "2026-10-03T15:00:00Z",
            Minute,
            "UTC",
            "2026-10-03T15:01:00Z",
        ),
    ];
    for (start, precision, zone, end) in cases {
        assert_eq!(
            unit_end(wt(start, precision), &tz(zone)),
            ts(end),
            "{precision:?} from {start} in {zone}"
        );
    }
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_day_precision_event_is_upcoming_until_the_day_ends_for_the_user() {
    let auckland = tz("Pacific/Auckland");
    let mut event = window(Kind::Event);
    event.valid_from = Some(wt("2026-10-02T11:00:00Z", TimePrecision::Day));
    assert_eq!(
        event.phase(&auckland, ts("2026-10-01T00:00:00Z")),
        Phase::Upcoming
    );
    assert_eq!(
        event.phase(&auckland, ts("2026-10-03T10:59:59Z")),
        Phase::Upcoming
    );
    assert_eq!(
        event.phase(&auckland, ts("2026-10-03T11:00:00Z")),
        Phase::RecentlyPast
    );
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_point_event_goes_from_upcoming_to_past_and_a_span_is_current_between() {
    let utc = TimeZone::UTC;
    let mut point = window(Kind::Event);
    point.valid_from = Some(wt("2026-10-03T15:00:00Z", TimePrecision::Hour));
    assert_eq!(point.closes_at(&utc), Some(ts("2026-10-03T16:00:00Z")));
    assert_eq!(
        point.phase(&utc, ts("2026-10-03T15:59:00Z")),
        Phase::Upcoming
    );
    assert_eq!(
        point.phase(&utc, ts("2026-10-03T16:00:00Z")),
        Phase::RecentlyPast
    );

    let mut span = window(Kind::Event);
    span.valid_from = Some(wt("2026-10-05T00:00:00Z", TimePrecision::Day));
    span.valid_until = Some(wt("2026-10-09T00:00:00Z", TimePrecision::Day));
    assert_eq!(span.closes_at(&utc), Some(ts("2026-10-10T00:00:00Z")));
    assert_eq!(
        span.phase(&utc, ts("2026-10-05T12:00:00Z")),
        Phase::Upcoming
    );
    assert_eq!(span.phase(&utc, ts("2026-10-06T00:00:00Z")), Phase::Current);
    assert_eq!(span.phase(&utc, ts("2026-10-09T23:59:00Z")), Phase::Current);
    assert_eq!(
        span.phase(&utc, ts("2026-10-10T00:00:00Z")),
        Phase::RecentlyPast
    );
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn recently_past_turns_long_past_30_days_after_the_close() {
    let utc = TimeZone::UTC;
    let mut event = window(Kind::Event);
    event.valid_from = Some(wt("2026-10-03T15:00:00Z", TimePrecision::Minute));
    assert_eq!(RECENTLY_PAST_DAYS, 30.0);
    assert_eq!(
        event.phase(&utc, ts("2026-11-02T15:00:59Z")),
        Phase::RecentlyPast
    );
    assert_eq!(
        event.phase(&utc, ts("2026-11-02T15:01:00Z")),
        Phase::LongPast
    );
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_task_is_current_until_its_due_day_ends_then_overdue_until_it_ends() {
    let utc = TimeZone::UTC;
    let mut task = window(Kind::Task);
    task.due_at = Some(wt("2026-10-09T00:00:00Z", TimePrecision::Day));
    assert_eq!(task.closes_at(&utc), None);
    assert_eq!(task.phase(&utc, ts("2026-10-09T23:59:00Z")), Phase::Current);
    assert_eq!(task.phase(&utc, ts("2026-10-10T00:00:00Z")), Phase::Overdue);
    assert_eq!(task.phase(&utc, ts("2027-10-10T00:00:00Z")), Phase::Overdue);

    // Completed on 12 Oct: past, not overdue.
    task.valid_until = Some(wt("2026-10-12T10:30:00Z", TimePrecision::Minute));
    assert_eq!(task.phase(&utc, ts("2026-10-12T10:30:59Z")), Phase::Overdue);
    assert_eq!(
        task.phase(&utc, ts("2026-10-12T10:31:00Z")),
        Phase::RecentlyPast
    );

    let undated = window(Kind::Task);
    assert_eq!(
        undated.phase(&utc, ts("2030-01-01T00:00:00Z")),
        Phase::Current
    );
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn facts_states_and_routines_without_an_end_stay_current() {
    let utc = TimeZone::UTC;
    let now = ts("2026-10-01T12:00:00Z");
    for kind in [Kind::Fact, Kind::State, Kind::Recurring] {
        let open = window(kind);
        assert_eq!(open.closes_at(&utc), None);
        assert_eq!(open.phase(&utc, now), Phase::Current, "{kind:?}");

        let mut started = window(kind);
        started.valid_from = Some(wt("2020-03-01T00:00:00Z", TimePrecision::Month));
        assert_eq!(started.closes_at(&utc), None);
        assert_eq!(started.phase(&utc, now), Phase::Current, "{kind:?}");

        let mut starting = window(kind);
        starting.valid_from = Some(wt("2026-10-10T00:00:00Z", TimePrecision::Day));
        assert_eq!(starting.phase(&utc, now), Phase::Upcoming, "{kind:?}");
    }

    // "User lived in Berlin until 12 Sep."
    let mut ended = window(Kind::State);
    ended.valid_until = Some(wt("2026-09-12T00:00:00Z", TimePrecision::Day));
    assert_eq!(ended.closes_at(&utc), Some(ts("2026-09-13T00:00:00Z")));
    assert_eq!(ended.phase(&utc, now), Phase::RecentlyPast);
    assert_eq!(
        ended.phase(&utc, ts("2026-10-13T00:00:00Z")),
        Phase::LongPast
    );
}

// State confidence (TIM-91, decision 8).

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn state_confidence_is_a_coin_flip_at_t_on_the_log_logistic_curve() {
    for volatility in Volatility::ALL {
        let t = volatility.rate_days();
        for (age, expected) in [
            (0.0, 1.0),
            (0.5, 0.8),
            (1.0, 0.5),
            (2.0, 0.2),
            (10.0, 1.0 / 101.0),
        ] {
            let c = state_confidence(Some(volatility), at(0.0), &[], at(age * t));
            assert_near(c, expected, EXACT);
        }
    }
    // T: 3 hours, 3 days, 3 weeks, 3 months, 5 years.
    let hours = state_confidence(Some(Volatility::Hours), at(0.0), &[], at(0.125));
    assert_near(hours, 0.5, EXACT);
    let years = state_confidence(
        Some(Volatility::Years),
        at(0.0),
        &[],
        at(5.0 * DAYS_PER_YEAR),
    );
    assert_near(years, 0.5, EXACT);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn only_a_mention_or_a_confirmation_resets_state_confidence() {
    // Days volatility (T = 3). The anchor is the confirmation on day 20;
    // the use on day 30 doesn't move it, nor does anything after now.
    let accesses = [
        created(0.0),
        mentioned(10.0),
        confirmed(20.0),
        used(30.0),
        confirmed(50.0),
    ];
    let c = state_confidence(Some(Volatility::Days), at(0.0), &accesses, at(23.0));
    assert_near(c, 0.5, EXACT);
    let c = state_confidence(Some(Volatility::Days), at(0.0), &accesses, at(33.0));
    assert_near(c, 1.0 / (1.0 + (13.0f64 / 3.0).powi(2)), EXACT);

    // A later observed_at wins over earlier accesses.
    let c = state_confidence(Some(Volatility::Days), at(25.0), &accesses, at(28.0));
    assert_near(c, 0.5, EXACT);

    // An anchor ahead of now is age 0.
    let c = state_confidence(Some(Volatility::Days), at(25.0), &[], at(24.0));
    assert_near(c, 1.0, EXACT);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn no_volatility_means_no_fading() {
    let c = state_confidence(None, at(0.0), &[created(0.0)], at(FAR_DAYS));
    assert_near(c, 1.0, EXACT);
}

// Purge eligibility (ADR 0008).

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn the_purge_rule_comes_from_tuning() {
    assert_eq!(
        PurgeRule::from_tuning(&Tuning::default()),
        PurgeRule {
            delta: Some(DELTA),
            overdue_days: 30
        }
    );
    let never =
        Tuning::from_toml("[purge]\ndelta = \"never\"\n[agenda]\noverdue_days = 10\n").unwrap();
    assert_eq!(
        PurgeRule::from_tuning(&never),
        PurgeRule {
            delta: None,
            overdue_days: 10
        }
    );
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_memory_is_purged_strictly_below_tau_minus_delta() {
    let fact = window(Kind::Fact);
    let utc = TimeZone::UTC;
    let now = at(0.0);
    let line = TAU - DELTA;
    assert!(!purge_eligible(&rule(Some(DELTA)), line, &fact, &utc, now));
    assert!(purge_eligible(
        &rule(Some(DELTA)),
        line.next_down(),
        &fact,
        &utc,
        now
    ));

    // δ = 0 purges on fading.
    assert!(!purge_eligible(&rule(Some(0.0)), TAU, &fact, &utc, now));
    assert!(purge_eligible(
        &rule(Some(0.0)),
        TAU.next_down(),
        &fact,
        &utc,
        now
    ));
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_null_delta_never_purges() {
    let utc = TimeZone::UTC;
    for kind in [
        Kind::Fact,
        Kind::Event,
        Kind::State,
        Kind::Task,
        Kind::Recurring,
    ] {
        assert!(!purge_eligible(
            &rule(None),
            -1e9,
            &window(kind),
            &utc,
            at(FAR_DAYS)
        ));
    }
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn one_mention_purges_reproduce_adr_0008() {
    let fact = window(Kind::Fact);
    let utc = TimeZone::UTC;
    let purged = |significance: f64, day: f64| {
        let s = strength_at(significance, &[created(0.0)], day);
        purge_eligible(&rule(Some(DELTA)), s.value, &fact, &utc, at(day))
    };
    for (level, days, _) in PURGES {
        let significance = level.value();
        match days {
            Some(days) => {
                let exact = one_mention_days(significance, TAU - DELTA);
                assert_relative(exact, days, TABLE);
                assert!(!purged(significance, exact * 0.99), "{level:?}");
                assert!(purged(significance, exact * 1.01), "{level:?}");
            }
            None => assert!(!purged(significance, FAR_DAYS), "{level:?}"),
        }
    }
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn separate_occasions_make_a_memory_unpurgeable() {
    let fact = window(Kind::Fact);
    let utc = TimeZone::UTC;
    let now = at(FAR_DAYS);
    for (level, _, hold) in PURGES {
        let significance = level.value();
        let held = strength_at(significance, &occasions(hold), FAR_DAYS);
        assert!(
            !purge_eligible(&rule(Some(DELTA)), held.value, &fact, &utc, now),
            "{level:?}"
        );
        if hold > 1 {
            let fewer = strength_at(significance, &occasions(hold - 1), FAR_DAYS);
            assert!(
                purge_eligible(&rule(Some(DELTA)), fewer.value, &fact, &utc, now),
                "{level:?}"
            );
        }
    }
    // Minor with two occasions misses by 0.007.
    let minor = strength_at(Significance::Minor.value(), &occasions(2), FAR_DAYS);
    assert_near(minor.value, -1.707_779_661_868_975_1, EXACT);
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_date_still_ahead_holds_the_head_back() {
    let faded = -5.0;
    let auckland = tz("Pacific/Auckland");

    // A trivial appointment on 3 Oct in Auckland stays until that day ends
    // there.
    let mut event = window(Kind::Event);
    event.valid_from = Some(wt("2026-10-02T11:00:00Z", TimePrecision::Day));
    let r = rule(Some(DELTA));
    assert!(!purge_eligible(
        &r,
        faded,
        &event,
        &auckland,
        ts("2026-06-01T00:00:00Z")
    ));
    assert!(!purge_eligible(
        &r,
        faded,
        &event,
        &auckland,
        ts("2026-10-03T10:59:00Z")
    ));
    assert!(purge_eligible(
        &r,
        faded,
        &event,
        &auckland,
        ts("2026-10-03T11:00:00Z")
    ));

    let mut state = window(Kind::State);
    state.valid_from = Some(wt("2026-01-01T00:00:00Z", TimePrecision::Day));
    state.valid_until = Some(wt("2026-12-01T09:30:00Z", TimePrecision::Minute));
    assert!(!purge_eligible(
        &r,
        faded,
        &state,
        &TimeZone::UTC,
        ts("2026-12-01T09:30:59Z")
    ));
    assert!(purge_eligible(
        &r,
        faded,
        &state,
        &TimeZone::UTC,
        ts("2026-12-01T09:31:00Z")
    ));
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn a_task_is_held_until_its_overdue_window_ends() {
    let utc = TimeZone::UTC;
    let faded = -5.0;
    let mut task = window(Kind::Task);
    // Due 9 Oct: overdue from 10 Oct, held for 30 days after that.
    task.due_at = Some(wt("2026-10-09T00:00:00Z", TimePrecision::Day));
    let r = rule(Some(DELTA));
    assert!(!purge_eligible(
        &r,
        faded,
        &task,
        &utc,
        ts("2026-11-08T23:59:00Z")
    ));
    assert!(purge_eligible(
        &r,
        faded,
        &task,
        &utc,
        ts("2026-11-09T00:00:00Z")
    ));

    let short = PurgeRule {
        delta: Some(DELTA),
        overdue_days: 10,
    };
    assert!(!purge_eligible(
        &short,
        faded,
        &task,
        &utc,
        ts("2026-10-19T23:59:00Z")
    ));
    assert!(purge_eligible(
        &short,
        faded,
        &task,
        &utc,
        ts("2026-10-20T00:00:00Z")
    ));
}

#[test]
#[ignore = "strength model not built yet; activate with the TIM-104 implementation"]
fn routines_undated_tasks_and_retracted_dates_hold_nothing() {
    let utc = TimeZone::UTC;
    let r = rule(Some(DELTA));
    let now = ts("2026-10-01T00:00:00Z");
    assert!(purge_eligible(
        &r,
        -5.0,
        &window(Kind::Recurring),
        &utc,
        now
    ));
    assert!(purge_eligible(&r, -5.0, &window(Kind::Task), &utc, now));

    // A rescheduled appointment: the old slot (1) is still ahead, but it was
    // retracted by the new one (2), which has passed. Only the head counts.
    let links = [link(1, Some(2), None), link(2, None, None)];
    assert_eq!(chain_head(&links, 1), 2);
    let mut head = window(Kind::Event);
    head.valid_from = Some(wt("2026-09-01T00:00:00Z", TimePrecision::Day));
    assert!(purge_eligible(&r, -5.0, &head, &utc, now));
}
