//! Call 1's reply, parsed and checked in code.
//!
//! The reply must fit the schema: anything else, such as a significance
//! above `critical`, is an invalid reply and the chunk is retried. Within a
//! reply that fits, a claim that fails a check is dropped or trimmed rather
//! than failing the chunk:
//!
//! - a quote that isn't in the chunk drops the claim;
//! - a time that doesn't parse is dropped and lowers window confidence;
//! - a task naming an occasion by handle gets the occasion's start as its
//!   end and due date, each only where the claim gave none;
//! - fields that don't belong to the claim's kind are dropped;
//! - a weekday named in the quote that matches none of the claim's dates
//!   lowers window confidence;
//! - an annual date in plain English gets an RRULE when call 1 gave none;
//! - an RRULE is kept only when it parses and recurs within a year;
//! - remember-this keeps a memory only from the owner's own message.

use std::collections::BTreeSet;

use chrono::TimeZone as _;
use jiff::civil::{Date, DateTime, Weekday};
use jiff::tz::TimeZone;
use jiff::{Timestamp, ToSpan};
use serde::Deserialize;
use serde_json::Value;

use super::input::Unit;
use super::{Call1Input, DropReason, Dropped, EntityKind, IgnoredOccasion, OccasionFilled};
use crate::constants::{Significance, Volatility};
use crate::ingest::TURN_SEPARATOR;
use crate::queue::SourceKind;
use crate::store::nfc;

#[derive(Debug, Deserialize)]
struct Reply {
    claims: Vec<RawClaim>,
    used_injected_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawClaim {
    content: String,
    kind: Kind,
    quote: String,
    significance: Significance,
    remember_this: bool,
    changes_something: bool,
    valid_from: Option<RawTime>,
    valid_until: Option<RawTime>,
    window_confidence: Confidence,
    until_event: Option<String>,
    due_at: Option<RawTime>,
    volatility: Option<Volatility>,
    recurrence_text: Option<String>,
    recurrence_rrule: Option<String>,
    recurrence_start: Option<RawTime>,
    entities: Vec<RawLink>,
    /// The handle of the upcoming occasion a task is for. Missing reads as
    /// null.
    #[serde(default)]
    occasion: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawTime {
    at: String,
    precision: Precision,
}

#[derive(Debug, Deserialize)]
struct RawLink {
    entity: Option<String>,
    new_name: Option<String>,
    new_kind: Option<EntityKind>,
    surface_form: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Kind {
    Fact,
    Event,
    State,
    Task,
    Recurring,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Fact => "fact",
            Kind::Event => "event",
            Kind::State => "state",
            Kind::Task => "task",
            Kind::Recurring => "recurring",
        }
    }

    /// The kind a memory made from the claim is stored as.
    pub fn stored(self) -> crate::strength::Kind {
        match self {
            Kind::Fact => crate::strength::Kind::Fact,
            Kind::Event => crate::strength::Kind::Event,
            Kind::State => crate::strength::Kind::State,
            Kind::Task => crate::strength::Kind::Task,
            Kind::Recurring => crate::strength::Kind::Recurring,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Confidence {
    High,
    Low,
}

/// How exact a time is, coarsest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Precision {
    Year,
    Month,
    Day,
    Hour,
    Minute,
}

impl Precision {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "year" => Some(Precision::Year),
            "month" => Some(Precision::Month),
            "day" => Some(Precision::Day),
            "hour" => Some(Precision::Hour),
            "minute" => Some(Precision::Minute),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Precision::Year => "year",
            Precision::Month => "month",
            Precision::Day => "day",
            Precision::Hour => "hour",
            Precision::Minute => "minute",
        }
    }
}

/// A stored time: the instant at the start of its unit in the source's
/// timezone, and its precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Stamp {
    pub at: Timestamp,
    pub precision: Precision,
}

/// An entity link as code resolved it.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Link {
    /// A candidate call 1 linked by handle.
    Known {
        entity: i64,
        surface_form: Option<String>,
    },
    /// A new entity call 1 proposed.
    Proposed {
        name: String,
        kind: EntityKind,
        surface_form: Option<String>,
    },
}

/// A claim that passed the checks: a new memory to write.
#[derive(Debug, Clone)]
pub(super) struct NewMemory {
    /// The claim's index in call 1's reply.
    pub claim: usize,
    /// `changes_something` or `remember_this`, from anyone: the claim gets
    /// reconciliation's wider candidate set and always runs call 2.
    pub flagged: bool,
    pub content: String,
    pub kind: Kind,
    pub significance: Significance,
    /// The owner asked to remember it in their own message.
    pub kept: bool,
    /// The quote's place in the chunk, in characters.
    pub start: usize,
    pub end: usize,
    pub valid_from: Option<Stamp>,
    /// Only this field was synthesized from the observation day, rather
    /// than supplied by call 1. Confidence alone cannot distinguish them.
    pub valid_from_defaulted: bool,
    pub valid_until: Option<Stamp>,
    pub until_event: Option<String>,
    pub low_confidence: bool,
    pub due_at: Option<Stamp>,
    pub volatility: Option<Volatility>,
    pub recurrence_text: Option<String>,
    pub recurrence_rrule: Option<String>,
    pub recurrence_start: Option<Stamp>,
    pub links: Vec<Link>,
    /// Which of `valid_until` and `due_at` were copied from the occasion
    /// the claim named.
    pub filled: OccasionFilled,
}

impl NewMemory {
    /// The start supplied by call 1, excluding the undated-event fallback.
    pub fn supplied_valid_from(&self) -> Option<Stamp> {
        if self.valid_from_defaulted {
            None
        } else {
            self.valid_from
        }
    }
}

/// The reply after the checks.
#[derive(Debug, Clone)]
pub(super) struct Checked {
    pub memories: Vec<NewMemory>,
    pub dropped: Vec<Dropped>,
    /// In-context memories judged used, by rowid and public id, each once.
    pub used: Vec<(i64, uuid::Uuid)>,
    /// Every claim with a non-null `occasion`, in claim order.
    pub occasions: Vec<CheckedOccasion>,
}

/// A claim's `occasion` reference after the checks. Commit decides what an
/// accepted one came to.
#[derive(Debug, Clone, Copy)]
pub(super) struct CheckedOccasion {
    /// The claim's index in call 1's reply.
    pub claim: usize,
    /// The occasion the handle names, whether or not it was used.
    pub occasion: Option<uuid::Uuid>,
    /// Why it wasn't used. `None` when it was accepted, and
    /// [`NewMemory::filled`] says what it supplied.
    pub ignored: Option<IgnoredOccasion>,
}

/// Parses and checks call 1's reply. `Err` names why the reply doesn't fit
/// the schema, without its content.
pub(super) fn check(
    reply: &Value,
    input: &Call1Input,
    unit: &Unit,
) -> Result<Checked, &'static str> {
    let reply = Reply::deserialize(reply).map_err(|_| "the reply doesn't fit call 1's schema")?;

    let mut memories = Vec::new();
    let mut dropped = Vec::new();
    let mut occasions = Vec::new();
    for (index, claim) in reply.claims.into_iter().enumerate() {
        // The reference is resolved whether or not the claim survives.
        let reference = claim.occasion.as_deref().map(|handle| {
            let occasion = unit.occasions.get(handle.trim());
            (occasion.map(|occasion| occasion.memory), claim.kind)
        });
        let checked = check_claim(index, claim, input, unit);
        if let Some((occasion, kind)) = reference {
            let ignored = if checked.is_err() {
                Some(IgnoredOccasion::Dropped)
            } else if kind != Kind::Task {
                Some(IgnoredOccasion::NotATask)
            } else if occasion.is_none() {
                Some(IgnoredOccasion::UnknownHandle)
            } else {
                None
            };
            occasions.push(CheckedOccasion {
                claim: index,
                occasion,
                ignored,
            });
        }
        match checked {
            Ok(memory) => memories.push(memory),
            Err(reason) => dropped.push(Dropped {
                claim: index,
                reason,
            }),
        }
    }

    let mut used = Vec::new();
    let mut seen = BTreeSet::new();
    for handle in &reply.used_injected_ids {
        // A handle call 1 wasn't given is ignored.
        if let Some(&(memory_id, memory)) = unit.in_context.get(handle.trim())
            && seen.insert(memory_id)
        {
            used.push((memory_id, memory));
        }
    }
    Ok(Checked {
        memories,
        dropped,
        used,
        occasions,
    })
}

fn check_claim(
    index: usize,
    claim: RawClaim,
    input: &Call1Input,
    unit: &Unit,
) -> Result<NewMemory, DropReason> {
    let content = claim.content.trim();
    if content.is_empty() {
        return Err(DropReason::EmptyContent);
    }
    let (start, end) = locate(&input.text, &claim.quote).ok_or(DropReason::QuoteNotFound)?;
    let from_reply = input.reply_start.is_some_and(|reply| start >= reply);
    let only_in_message = input.reply_start.is_some_and(|reply| {
        only_before(
            &input.text,
            &claim.quote,
            reply.saturating_sub(SEPARATOR_CHARS),
        )
    });

    let tz = &unit.tz;
    let mut low = claim.window_confidence == Confidence::Low;
    let mut time = |raw: Option<RawTime>| -> Option<Stamp> {
        let raw = raw?;
        let stamp = resolve(&raw.at, raw.precision, tz).map(|at| Stamp {
            at,
            precision: raw.precision,
        });
        if stamp.is_none() {
            low = true;
        }
        stamp
    };
    // Whether call 1 gave the field at all, before parsing: one it gave
    // but that doesn't parse is still its own, and isn't filled.
    let gave_valid_until = claim.valid_until.is_some();
    let gave_due_at = claim.due_at.is_some();
    let mut valid_from = time(claim.valid_from);
    let mut valid_until = time(claim.valid_until);
    let mut due_at = time(claim.due_at);
    let has_recurrence_start = claim.recurrence_start.is_some();
    let mut recurrence_start = time(claim.recurrence_start);
    let mut until_event = non_empty(claim.until_event);
    let mut volatility = claim.volatility;
    let mut recurrence_text = non_empty(claim.recurrence_text);
    let mut recurrence_rrule = non_empty(claim.recurrence_rrule);

    // Each kind keeps only its own fields.
    if claim.kind != Kind::State {
        volatility = None;
    }
    if claim.kind != Kind::Task {
        due_at = None;
    }
    if claim.kind != Kind::Recurring {
        recurrence_text = None;
        recurrence_rrule = None;
        recurrence_start = None;
    }
    match claim.kind {
        // A fact never gets an end at extraction; a stated start stays.
        Kind::Fact => {
            valid_until = None;
            until_event = None;
        }
        // A task for a dated occasion can end when that occasion passes.
        Kind::Task | Kind::Event | Kind::State | Kind::Recurring => {}
    }

    // Judge assistant tasks on the reply's own dates, before an occasion
    // can supply a deadline.
    if claim.kind == Kind::Task && from_reply && due_at.is_none() && until_event.is_none() {
        return Err(DropReason::AssistantTaskUndated);
    }

    // A task for a listed occasion gets the occasion's start, its instant
    // and precision as stored, for each of its end and due date it left
    // out. The task keeps its own timezone, so a coarse unit can end at a
    // different time than the occasion's does in another zone.
    let mut filled = OccasionFilled::default();
    if claim.kind == Kind::Task
        && let Some(occasion) = claim
            .occasion
            .as_deref()
            .and_then(|handle| unit.occasions.get(handle.trim()))
    {
        if !gave_valid_until {
            valid_until = Some(occasion.at);
            filled.valid_until = true;
        }
        if !gave_due_at {
            due_at = Some(occasion.at);
            filled.due_at = true;
        }
        if (filled.valid_until || filled.due_at) && occasion.low_confidence {
            low = true;
        }
    }

    if claim.kind == Kind::Recurring {
        // Do not repair an explicit but invalid start or guess an interval
        // phase. Only a fully specified annual calendar date can start itself.
        if recurrence_rrule.is_none() && !has_recurrence_start {
            recurrence_rrule = recurrence_text.as_deref().and_then(annual_rule);
        }
        if !has_recurrence_start {
            recurrence_start = recurrence_rrule
                .as_deref()
                .and_then(|rule| infer_annual_start(rule, &input.timezone, tz, input.observed_at));
        }
        // The RRULE is kept only when it parses and recurs in the
        // year after the reference date, from a first occurrence.
        let recurs = match (&recurrence_rrule, recurrence_start) {
            (Some(rule), Some(start)) => {
                recurs(rule, start.at, &input.timezone, tz, input.observed_at)
            }
            _ => false,
        };
        if recurs {
            recurrence_rrule = recurrence_rrule.map(|rule| strip_rrule_prefix(&rule).to_owned());
        } else {
            recurrence_rrule = None;
            recurrence_start = None;
        }
    }

    let dates = [valid_from, valid_until, due_at, recurrence_start];
    if weekday_mismatch(&claim.quote, &dates, tz) {
        low = true;
    }

    let valid_from_defaulted = claim.kind == Kind::Event && valid_from.is_none();
    if valid_from_defaulted {
        // An event with no stated time starts on the day it was said,
        // with low confidence.
        valid_from = start_of_day(input.observed_at, tz).map(|at| Stamp {
            at,
            precision: Precision::Day,
        });
        low = true;
    }

    // Only the owner's own
    // message keeps a memory. Anyone else, a document or the reply keeps the
    // level call 1 gave, which is at most critical. A quote that's also in
    // the reply, or runs into it, can't be shown to be the owner's, so it
    // isn't kept either.
    let kept = claim.remember_this
        && input.source_kind == SourceKind::Turn
        && unit.owner_speaking
        && only_in_message;

    let links = claim
        .entities
        .into_iter()
        .filter_map(|link| resolve_link(link, unit))
        .collect();

    Ok(NewMemory {
        claim: index,
        flagged: claim.remember_this || claim.changes_something,
        content: content.to_owned(),
        kind: claim.kind,
        significance: claim.significance,
        kept,
        start,
        end,
        valid_from,
        valid_from_defaulted,
        valid_until,
        until_event,
        low_confidence: low,
        due_at,
        volatility,
        recurrence_text,
        recurrence_rrule,
        recurrence_start,
        links,
        filled,
    })
}

fn non_empty(text: Option<String>) -> Option<String> {
    text.map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty())
}

/// The first occurrence of `quote` in `text`, in characters. An empty quote
/// is never found.
fn locate(text: &str, quote: &str) -> Option<(usize, usize)> {
    if quote.is_empty() {
        return None;
    }
    let byte = text.find(quote)?;
    let start = text[..byte].chars().count();
    Some((start, start + quote.chars().count()))
}

/// [`TURN_SEPARATOR`] in characters. It's ASCII, so its length in bytes.
const SEPARATOR_CHARS: usize = TURN_SEPARATOR.len();

/// Whether every occurrence of `quote` in `text` ends by character
/// `message_end`: the quote is in the message and nowhere in the reply.
fn only_before(text: &str, quote: &str, message_end: usize) -> bool {
    if quote.is_empty() {
        return false;
    }
    let quote_chars = quote.chars().count();
    // Overlapping occurrences count too, so step one character at a time.
    let mut found = false;
    for (byte, _) in text.char_indices() {
        if text[byte..].starts_with(quote) {
            let start = text[..byte].chars().count();
            if start + quote_chars > message_end {
                return false;
            }
            found = true;
        }
    }
    found
}

/// A time call 1 gave, as the instant at the start of its unit in `tz`.
/// `None` when it doesn't parse, or is coarser than its precision claims.
pub(super) fn resolve(at: &str, precision: Precision, tz: &TimeZone) -> Option<Timestamp> {
    let (datetime, given) = parse_local(at.trim())?;
    if precision > given {
        return None;
    }
    let date = datetime.date();
    let truncated = match precision {
        Precision::Year => DateTime::new(date.year(), 1, 1, 0, 0, 0, 0),
        Precision::Month => DateTime::new(date.year(), date.month(), 1, 0, 0, 0, 0),
        Precision::Day => DateTime::new(date.year(), date.month(), date.day(), 0, 0, 0, 0),
        Precision::Hour => DateTime::new(
            date.year(),
            date.month(),
            date.day(),
            datetime.hour(),
            0,
            0,
            0,
        ),
        Precision::Minute => DateTime::new(
            date.year(),
            date.month(),
            date.day(),
            datetime.hour(),
            datetime.minute(),
            0,
            0,
        ),
    }
    .ok()?;
    Some(truncated.to_zoned(tz.clone()).ok()?.timestamp())
}

/// `YYYY`, `YYYY-MM`, `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM` (a space for the
/// `T`, an hour alone and seconds are accepted too), with the finest
/// precision it gives.
fn parse_local(text: &str) -> Option<(DateTime, Precision)> {
    let digits = |part: &str| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit());
    match text.len() {
        4 if digits(text) => {
            let year: i16 = text.parse().ok()?;
            Some((
                Date::new(year, 1, 1)
                    .ok()?
                    .to_datetime(jiff::civil::Time::midnight()),
                Precision::Year,
            ))
        }
        7 => {
            let date: Date = format!("{text}-01").parse().ok()?;
            Some((
                date.to_datetime(jiff::civil::Time::midnight()),
                Precision::Month,
            ))
        }
        10 => {
            let date: Date = text.parse().ok()?;
            Some((
                date.to_datetime(jiff::civil::Time::midnight()),
                Precision::Day,
            ))
        }
        _ => {
            let text = text.replacen(' ', "T", 1);
            let (_, time) = text.split_once('T')?;
            let text = if digits(time) && time.len() == 2 {
                format!("{text}:00")
            } else {
                text.clone()
            };
            let datetime: DateTime = text.parse().ok()?;
            Some((datetime, Precision::Minute))
        }
    }
}

pub(super) fn start_of_day(at: Timestamp, tz: &TimeZone) -> Option<Timestamp> {
    let date = at.to_zoned(tz.clone()).date();
    Some(date.to_zoned(tz.clone()).ok()?.timestamp())
}

const WEEKDAYS: [(&str, Weekday); 7] = [
    ("monday", Weekday::Monday),
    ("tuesday", Weekday::Tuesday),
    ("wednesday", Weekday::Wednesday),
    ("thursday", Weekday::Thursday),
    ("friday", Weekday::Friday),
    ("saturday", Weekday::Saturday),
    ("sunday", Weekday::Sunday),
];

/// Whether the quote names a weekday that none of the claim's dates at day
/// precision or finer falls on (a mismatch lowers window confidence rather
/// than dropping the window). Every named weekday must match a date,
/// so one that matches can't hide another that doesn't.
fn weekday_mismatch(quote: &str, dates: &[Option<Stamp>], tz: &TimeZone) -> bool {
    let quote = quote.to_lowercase();
    let named: Vec<Weekday> = WEEKDAYS
        .iter()
        .filter(|(name, _)| quote.contains(name))
        .map(|(_, weekday)| *weekday)
        .collect();
    if named.is_empty() {
        return false;
    }
    let checked: Vec<Weekday> = dates
        .iter()
        .flatten()
        .filter(|stamp| stamp.precision >= Precision::Day)
        .map(|stamp| stamp.at.to_zoned(tz.clone()).weekday())
        .collect();
    !checked.is_empty() && named.iter().any(|weekday| !checked.contains(weekday))
}

fn strip_rrule_prefix(rule: &str) -> &str {
    let rule = rule.trim();
    rule.strip_prefix("RRULE:").unwrap_or(rule)
}

const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// The RRULE for an annual date in plain English, such as "Every year on
/// November 12" or "annually on 3 Mar", when call 1 stated the schedule but
/// gave no rule. Anything looser is left without one.
fn annual_rule(text: &str) -> Option<String> {
    let text = text.trim().trim_end_matches('.').to_lowercase();
    let date = [
        "every year on ",
        "each year on ",
        "annually on ",
        "yearly on ",
    ]
    .iter()
    .find_map(|prefix| text.strip_prefix(prefix))?;
    let month = |word: &str| {
        MONTHS
            .iter()
            .position(|name| *name == word || (word.len() == 3 && name.starts_with(word)))
            .map(|index| index + 1)
    };
    let day = |word: &str| {
        ["st", "nd", "rd", "th"]
            .iter()
            .find_map(|suffix| word.strip_suffix(suffix))
            .unwrap_or(word)
            .parse::<u8>()
            .ok()
            .filter(|day| (1..=31).contains(day))
    };
    let (month, day) = match date.split_whitespace().collect::<Vec<_>>().as_slice() {
        [first, second] => match (month(first), month(second)) {
            (Some(month), None) => (month, day(second)?),
            (None, Some(month)) => (month, day(first)?),
            _ => return None,
        },
        _ => return None,
    };
    Some(format!("FREQ=YEARLY;BYMONTH={month};BYMONTHDAY={day}"))
}

/// A deliberately narrow inference: one month and one positive month day,
/// every year. Other selectors, counts, time-of-day fields and interval
/// phases need a stated start. Midnight is a day-precision anchor, not an
/// inferred appointment time. The recurrence library still validates the rule
/// and finds the next real date (rather than normalizing an impossible date).
fn infer_annual_start(rule: &str, tz_name: &str, tz: &TimeZone, from: Timestamp) -> Option<Stamp> {
    let mut seen = BTreeSet::new();
    for part in strip_rrule_prefix(rule).split(';') {
        let (key, value) = part.split_once('=')?;
        if !seen.insert(key) {
            return None;
        }
        match key {
            "FREQ" if value == "YEARLY" => {}
            "INTERVAL" if value == "1" => {}
            "BYMONTH"
                if value
                    .parse::<u8>()
                    .ok()
                    .is_some_and(|n| (1..=12).contains(&n)) => {}
            "BYMONTHDAY"
                if value
                    .parse::<u8>()
                    .ok()
                    .is_some_and(|n| (1..=31).contains(&n)) => {}
            _ => return None,
        }
    }
    if !["FREQ", "BYMONTH", "BYMONTHDAY"]
        .iter()
        .all(|key| seen.contains(key))
    {
        return None;
    }
    let anchor = start_of_day(from, tz)?;
    // Day-precision occasions include today, even when learned after midnight.
    next_occurrence(rule, anchor, tz_name, tz, anchor).map(|at| Stamp {
        at,
        precision: Precision::Day,
    })
}

/// Whether `rule`, starting at `start` in the source's timezone, parses and
/// has an occurrence in the year from `from`.
fn recurs(rule: &str, start: Timestamp, tz_name: &str, tz: &TimeZone, from: Timestamp) -> bool {
    next_occurrence(rule, start, tz_name, tz, from).is_some()
}

fn next_occurrence(
    rule: &str,
    start: Timestamp,
    tz_name: &str,
    tz: &TimeZone,
    from: Timestamp,
) -> Option<Timestamp> {
    let rule = strip_rrule_prefix(rule);
    // One RRULE line and nothing else: no DTSTART, EXRULE or RDATE smuggled
    // in on another line.
    if rule.is_empty() || rule.contains(['\n', '\r', ':']) {
        return None;
    }
    let local = start.to_zoned(tz.clone()).datetime();
    let text = format!(
        "DTSTART;TZID={tz_name}:{}\nRRULE:{rule}",
        local.strftime("%Y%m%dT%H%M%S")
    );
    let set = text.parse::<rrule::RRuleSet>().ok()?;
    let until = from
        .to_zoned(tz.clone())
        .checked_add(1.year())
        .ok()
        .map(|zoned| zoned.timestamp())?;
    let utc = |at: Timestamp| rrule::Tz::UTC.timestamp_opt(at.as_second(), 0).single();
    let next = set
        .after(utc(from)?)
        .before(utc(until)?)
        .all(1)
        .dates
        .into_iter()
        .next()?;
    Timestamp::from_second(next.timestamp()).ok()
}

/// A link as code resolved it. Names and surface forms are composed (NFC)
/// here, so the entity it may create, the exact alias lookup at commit, the
/// aliases it adds and the stored surface form all use the form aliases are
/// stored in (schema version 3).
fn resolve_link(link: RawLink, unit: &Unit) -> Option<Link> {
    let surface_form = non_empty(Some(nfc(&link.surface_form)));
    if let Some(handle) = link.entity {
        let entity = *unit.candidates.get(handle.trim())?;
        return Some(Link::Known {
            entity,
            surface_form,
        });
    }
    let name = non_empty(link.new_name.as_deref().map(nfc))?;
    Some(Link::Proposed {
        name,
        kind: link.new_kind?,
        surface_form,
    })
}

/// English pronouns, which are kept on a link but never become aliases: an
/// alias "I" would match nearly every text.
pub(super) fn is_pronoun(surface_form: &str) -> bool {
    const PRONOUNS: [&str; 31] = [
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "you",
        "your",
        "yours",
        "yourself",
        "we",
        "us",
        "our",
        "ours",
        "ourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "it",
        "its",
        "itself",
        "yourselves",
    ];
    let lower = surface_form.trim().to_lowercase();
    PRONOUNS.contains(&lower.as_str())
}
