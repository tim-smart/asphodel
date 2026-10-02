//! The injection's text.
//!
//! Hermes replays an injection verbatim on every later turn, so nothing in
//! it is relative to the moment it was made except the header's own time:
//! annotations give absolute dates, and a state's age comes with the date
//! it's counted from. The sentence is the stored content verbatim, so
//! Hermes' identical-bullet dedup works, and there are no memory ids.
//!
//! ```text
//! Recalled Wed 1 Oct 10:42
//! - Tim has a dentist appointment on 3 October 2026 at 15:00. [upcoming Thu 3
//! Oct 15:00]
//! - Tim is in Lisbon. [observed 4 days ago, Sat 27 Sep]
//! ```

use jiff::Timestamp;
use jiff::tz::TimeZone;

use super::candidates::Candidate;
use crate::constants::STATE_AGE_SHOWN_BELOW;
use crate::strength::{Kind, Phase, TimePrecision, WorldTime};

const MICROS_PER_DAY: i64 = 24 * 60 * 60 * 1_000_000;

/// The header: when the recall ran, in the bank's timezone.
pub(crate) fn header(now: Timestamp, tz: &TimeZone) -> String {
    format!(
        "Recalled {}",
        now.to_zoned(tz.clone()).strftime("%a %-d %b %H:%M")
    )
}

/// One memory's line: `- <sentence>`, then its annotations in brackets.
pub(crate) fn line(candidate: &Candidate, now: Timestamp) -> String {
    let annotations = annotations(candidate, now);
    if annotations.is_empty() {
        format!("- {}", candidate.content)
    } else {
        format!("- {} [{}]", candidate.content, annotations.join("; "))
    }
}

/// The block: the header, then one line per memory, in score order.
pub(crate) fn block(header: &str, lines: &[String]) -> String {
    let mut text = header.to_owned();
    for line in lines {
        text.push('\n');
        text.push_str(line);
    }
    text
}

fn annotations(candidate: &Candidate, now: Timestamp) -> Vec<String> {
    let window = &candidate.window;
    let tz = &candidate.tz;
    let mut annotations = Vec::new();
    match candidate.phase {
        Phase::Upcoming => {
            if let Some(from) = window.valid_from {
                annotations.push(format!("upcoming {}", date(from, tz, now)));
            }
        }
        Phase::Overdue => {
            if let Some(due) = window.due_at {
                annotations.push(format!("overdue since {}", date(due, tz, now)));
            }
        }
        // A point event's sentence already says when it happened; only a
        // stated end is news.
        Phase::RecentlyPast | Phase::LongPast => {
            if let Some(until) = window.valid_until {
                annotations.push(format!("ended {}", date(until, tz, now)));
            }
        }
        Phase::Current => {}
    }
    if window.kind == Kind::Recurring
        && let Some(text) = &candidate.recurrence_text
    {
        annotations.push(format!("recurring: {text}"));
    }
    if let Some(age) = state_age(candidate, now) {
        annotations.push(age);
    }
    let dated =
        window.valid_from.is_some() || window.valid_until.is_some() || window.due_at.is_some();
    if candidate.low_confidence && dated {
        annotations.push("date uncertain".to_owned());
    }
    annotations
}

/// `observed 4 days ago, Sat 27 Sep` for a state whose confidence is below
/// [`STATE_AGE_SHOWN_BELOW`], and `None` otherwise. A
/// mental model entry citing such a state shows it too.
pub(crate) fn state_age(candidate: &Candidate, now: Timestamp) -> Option<String> {
    if candidate.window.kind != Kind::State || candidate.state_confidence >= STATE_AGE_SHOWN_BELOW {
        return None;
    }
    let days = (now.as_microsecond() - candidate.last_observed.as_microsecond())
        .max(0)
        .div_euclid(MICROS_PER_DAY);
    let ago = match days {
        0 => "today".to_owned(),
        1 => "1 day ago".to_owned(),
        n => format!("{n} days ago"),
    };
    let on = candidate
        .last_observed
        .to_zoned(candidate.tz.clone())
        .strftime("%a %-d %b");
    Some(format!("observed {ago}, {on}"))
}

/// A stored time as its precision allows: `Thu 3 Oct 15:00`, `Sat 12 Sep`,
/// `Oct 2027`, `2027`. A day always carries its weekday, and
/// its year when it isn't the current one.
fn date(time: WorldTime, tz: &TimeZone, now: Timestamp) -> String {
    let zoned = time.at.to_zoned(tz.clone());
    let this_year = now.to_zoned(tz.clone()).year() == zoned.year();
    let day = if this_year {
        "%a %-d %b"
    } else {
        "%a %-d %b %Y"
    };
    match time.precision {
        TimePrecision::Year => zoned.strftime("%Y").to_string(),
        TimePrecision::Month => zoned.strftime("%b %Y").to_string(),
        TimePrecision::Day => zoned.strftime(day).to_string(),
        TimePrecision::Hour | TimePrecision::Minute => {
            format!("{} {}", zoned.strftime(day), zoned.strftime("%H:%M"))
        }
    }
}
