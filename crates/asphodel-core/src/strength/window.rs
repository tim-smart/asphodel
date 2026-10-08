//! Validity windows, time precision and phase, on world time in the source's
//! timezone.

use jiff::tz::TimeZone;
use jiff::{SignedDuration, Span, Timestamp, ToSpan};
use serde::{Deserialize, Serialize};

use super::world_days;
use crate::constants::RECENTLY_PAST_DAYS;

/// How exact a stored time is. A time is stored as the UTC instant at the
/// start of its unit in the source's timezone, and it means the whole unit:
/// a day-precision date lasts until that day ends for the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimePrecision {
    Year,
    Month,
    Day,
    Hour,
    Minute,
}

/// A stored time with its precision.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct WorldTime {
    pub at: Timestamp,
    pub precision: TimePrecision,
}

/// The end of the unit `time` starts, one unit later in `tz`. Years, months
/// and days are calendar units, so a day can be 23 or 25 hours; hours and
/// minutes are absolute. Saturates at the end of jiff's range.
pub fn unit_end(time: WorldTime, tz: &TimeZone) -> Timestamp {
    let calendar = |span: Span| {
        time.at
            .to_zoned(tz.clone())
            .checked_add(span)
            .map(|end| end.timestamp())
    };
    let end = match time.precision {
        TimePrecision::Year => calendar(1.year()),
        TimePrecision::Month => calendar(1.month()),
        TimePrecision::Day => calendar(1.day()),
        TimePrecision::Hour => time.at.checked_add(SignedDuration::from_hours(1)),
        TimePrecision::Minute => time.at.checked_add(SignedDuration::from_mins(1)),
    };
    end.unwrap_or(Timestamp::MAX)
}

/// The kind of a memory, which decides how its window behaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Fact,
    Event,
    State,
    Task,
    Recurring,
}

/// The world-time fields of a memory. For a chain, the head's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    pub kind: Kind,
    pub valid_from: Option<WorldTime>,
    pub valid_until: Option<WorldTime>,
    pub due_at: Option<WorldTime>,
    /// When it was said. Something that had already started then is never
    /// upcoming, however its start was dated.
    pub observed_at: Timestamp,
}

/// Where a memory sits relative to now. Computed, never stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Before the end of `valid_from`'s unit, and said before it began.
    Upcoming,
    /// An open window, an open task that isn't overdue, or no window.
    Current,
    /// An open task past the end of its due date's unit.
    Overdue,
    /// Less than [`RECENTLY_PAST_DAYS`] world days since the window closed.
    RecentlyPast,
    /// At least [`RECENTLY_PAST_DAYS`] world days since the window closed.
    LongPast,
}

/// A dated task said at or after its due time is a record, not an open plan.
/// Eligibility uses the stored instant; precision only controls when an
/// eligible task becomes overdue.
pub(crate) fn task_due_is_eligible(observed_at: Timestamp, due_at: Timestamp) -> bool {
    observed_at < due_at
}

impl Window {
    /// When the window closes: the end of `valid_until`'s unit. An event with
    /// no `valid_until` is a point event, whose window is `valid_from`'s
    /// unit, so it closes at that unit's end. Any other kind with no
    /// `valid_until` has no end yet.
    pub fn closes_at(&self, tz: &TimeZone) -> Option<Timestamp> {
        match (self.valid_until, self.kind) {
            (Some(until), _) => Some(unit_end(until, tz)),
            (None, Kind::Event) => self.valid_from.map(|from| unit_end(from, tz)),
            (None, _) => None,
        }
    }

    /// When an open task becomes overdue: the end of its due date's unit.
    /// `None` unless it is a dated task said before its due time.
    pub fn overdue_from(&self, tz: &TimeZone) -> Option<Timestamp> {
        match (self.kind, self.due_at) {
            (Kind::Task, Some(due)) if task_due_is_eligible(self.observed_at, due.at) => {
                Some(unit_end(due, tz))
            }
            _ => None,
        }
    }

    /// The end of the overdue window: [`Window::overdue_from`] plus
    /// `overdue_days` × 24 h, with `overdue_days` from
    /// `Tuning::agenda.overdue_days`. The agenda lists an overdue task until
    /// then, and purge holds the task back until then, so the two share this
    /// boundary and nothing on the agenda can be purged. A completed task
    /// is held no later than the end of its closing unit.
    pub fn overdue_until(&self, tz: &TimeZone, overdue_days: u32) -> Option<Timestamp> {
        let days = SignedDuration::from_hours(24 * i64::from(overdue_days));
        self.overdue_from(tz).map(|from| {
            let until = from.checked_add(days).unwrap_or(Timestamp::MAX);
            self.closes_at(tz).map_or(until, |close| until.min(close))
        })
    }

    /// Upcoming before the end of `valid_from`'s unit when said before it
    /// began, past from
    /// [`Window::closes_at`], then overdue for a task past
    /// [`Window::overdue_from`], and current otherwise.
    ///
    /// Recently past and long past are labels for rendering and filters.
    /// Ranking works from the days since the close, not from this.
    pub fn phase(&self, tz: &TimeZone, now: Timestamp) -> Phase {
        if self
            .valid_from
            .is_some_and(|from| self.observed_at < from.at && now < unit_end(from, tz))
        {
            return Phase::Upcoming;
        }
        if let Some(closes_at) = self.closes_at(tz)
            && now >= closes_at
        {
            return if world_days(closes_at, now) < RECENTLY_PAST_DAYS {
                Phase::RecentlyPast
            } else {
                Phase::LongPast
            };
        }
        if self.overdue_from(tz).is_some_and(|from| now >= from) {
            return Phase::Overdue;
        }
        Phase::Current
    }
}
