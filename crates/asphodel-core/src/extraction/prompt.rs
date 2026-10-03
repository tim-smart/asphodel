//! Call 1's request: the rules, the rendered input and the reply schema.
//!
//! The system prompt is the same for every chunk on a daemon, so a provider
//! can cache it; everything about the chunk goes in the user prompt. Only
//! `[llm] language` changes it.

use std::fmt::Write as _;

use serde_json::{Value, json};

use super::{CALL1_TEMPLATE, CALL1_VERSION, Call1Input, EntityKind};
use crate::models::{LlmRequest, Template};
use crate::queue::SourceKind;

const SCHEMA_NAME: &str = "call1_claims";

const SYSTEM: &str = r#"You extract memories for a personal assistant's long-term memory. You are given a text (a conversation turn or a section of a document) and return the claims in it worth remembering, plus which of the memories already in the assistant's context its reply relied on.

# Claims

Each claim is one sentence that makes sense on its own: use names instead of pronouns, and absolute dates instead of relative ones ("tomorrow" becomes "on 2 October 2026"). Put the why into the sentence; there are no separate fields for who, what or why. {language_rule}

`quote` is the exact passage of the text the claim comes from, copied character for character. Quote only from the text, never from the context: the context is there so you can understand the text, and a claim found only in the context is not extracted. A claim whose quote isn't in the text is thrown away.

What to extract:
- From the speaker's message, what they say about themselves, the people, places and things in their life, their plans, tasks and preferences. "I" and "me" mean the speaker.
- A short answer to a question the assistant asked in the context is the speaker's claim, written out in full. "Yes" after "Are you still at Acme?" becomes "Alex still works at Acme.", quoting "Yes". A "remember that" pointing at something said earlier works the same way.
- From the assistant's reply, only what the assistant says it has done or will do. An assistant task is extracted only when the speaker asked for it and it has a due date or an until-event beyond this turn. Never extract the assistant's suggestions, general knowledge, or findings from tools.
- Nothing from a request to forget something, and no task to forget it.
- Skip greetings, filler and small talk.

# Kinds

Decide `kind` in this order:
- `task`: something to be done. Give `due_at` when the text gives a due date.
- `recurring`: something that repeats on a schedule, when the next occurrence matters. A habit with no schedule ("Alex goes to the gym") is a fact.
- `event`: something that happens at a time or over a span, including anything with an explicit end date ("on holiday until 12 October"). A completed or cancelled task is an event.
- `state`: ongoing and expected to change without anyone announcing it: mood, where someone is, what they're working on, how something is going. A state with an explicit end condition keeps it as `until_event` ("until the release ships").
- `fact`: everything else, including preferences, and things whose change would be announced (a job, a home, a relationship) or that aren't expected to change (a chronic condition).

The rule between fact and state: a fact isn't expected to change, or its change would be announced; a state is expected to change quietly.

`volatility` is for states only: how quickly the state goes stale. `hours` (mood, where someone is today), `days` (an illness, a trip, a bug being chased), `weeks` (a sprint, a visitor), `months` (a project, job hunting), `years` (a degree). Use null when unsure, and null for every other kind.

# Significance

How much the claim matters on its own terms, judged by how long it should be remembered:
- `trivial`: a passing detail, worth about a week. "Alex had pasta for lunch." "Alex is reading the release notes."
- `minor`: useful for about a month. "Alex's sister is visiting next weekend." "Alex needs to reply to the landlord this week."
- `notable`: worth most of a year. "Alex started learning Rust." "Alex's team ships the new API on 15 November 2026."
- `major`: worth years. "Alex moved to Wellington." "Alex is allergic to penicillin."
- `critical`: worth a decade or more, and rare. "Alex's daughter Mia was born on 3 March 2026." "Alex married Jo on 12 June 2027."

Most claims are trivial or minor. Major is rare and critical is a few per hundred claims.

`remember_this` is true only when the text explicitly asks to remember the claim ("remember this", "don't forget that"). `changes_something` is true when the claim ends, corrects, reschedules or completes something that may already be remembered ("I moved out of Berlin", "the dentist is now on Friday", "I filed the tax return").

# Time

Use the calendar to turn relative times into absolute ones. Times are local to the text's timezone. Each time is an object with `at` and `precision`: `at` is `YYYY`, `YYYY-MM`, `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM`, and `precision` is `year`, `month`, `day`, `hour` or `minute`, as exact as the text is. "In March 2024" is `{"at": "2024-03", "precision": "month"}`; "tomorrow at 3pm" is a day and hour.

- `valid_from` is when the claim starts to hold; `valid_until` when it stops. Never set `valid_until` unless the text states an end. A fact never gets `valid_until`, though it keeps a start the text states ("started at Acme in March 2024").
- A task's due date goes in `due_at`, never in `valid_until`.
- `until_event` is an end given as a condition rather than a date.
- `window_confidence` is `low` when you had to guess at the dates, otherwise `high`. A coarse date is a matter of precision, not confidence.
- If the reference date is unknown, don't resolve relative times; use only dates written out in full.

For a recurring claim, always give `recurrence_text`, the schedule in plain words. Add `recurrence_rrule` (an RFC 5545 RRULE such as `FREQ=WEEKLY;BYDAY=TU`) only when the wording maps cleanly, together with `recurrence_start`, the first occurrence. Otherwise leave both null.

# Entities

Link each claim to the entities it's about. Known entities are listed with a handle (`e1`, `e2`, …), their aliases and a few memories about them. Link to a known entity with its handle in `entity`. If the claim is about someone or something not listed, propose it with `new_name` and `new_kind` (`person`, `place`, `organisation`, `project` or `thing`) and leave `entity` null. Two different people can share a name: propose a new entity rather than link to a known one that isn't the same person. `surface_form` is how the text names the entity ("Alex", "my sister", "I"). The user and the assistant are always listed.

# Used

The memories already in the assistant's context are listed with handles (`m1`, `m2`, …), and so are any entries of the assistant's standing notes about the user (`n1`, `n2`, …), each with the memories it rests on. In `used_injected_ids`, give the handles of those the assistant's reply actually relied on: a memory's, or an entry's when the reply relied on the entry. Being shown a memory or an entry isn't using it, and a document has no reply, so for a document this is empty."#;

/// The language rule without `[llm] language`.
const INFERRED_LANGUAGE: &str = "Write the claim in the language of the passage it quotes and never translate. Entity names and dates stay as they appear.";

/// The system prompt, with the language rule for `language`.
fn system(language: Option<&str>) -> String {
    let rule = match language {
        None => INFERRED_LANGUAGE.to_owned(),
        Some(language) => format!(
            "Write every claim in {}, translating if the text is in another language. Entity names keep the form the text uses, and `quote` and `surface_form` stay exactly as the text has them.",
            language.trim()
        ),
    };
    SYSTEM.replace("{language_rule}", &rule)
}

/// Call 1's request for `input`. The reply schema is strict, every property
/// required.
pub fn call1_request(input: &Call1Input) -> LlmRequest {
    LlmRequest {
        template: Template {
            name: CALL1_TEMPLATE.into(),
            version: CALL1_VERSION,
        },
        system: system(input.language.as_deref()),
        user: render(input),
        schema_name: SCHEMA_NAME.into(),
        schema: schema(),
        max_tokens: None,
    }
}

/// The user prompt: the input, section by section.
fn render(input: &Call1Input) -> String {
    let mut out = String::new();
    match input.reference_date {
        Some(reference) => {
            let _ = writeln!(
                out,
                "Reference date: {reference} {}, in {}.",
                reference.strftime("%A"),
                input.timezone
            );
            out.push_str("\nCalendar:\n");
            for day in &input.calendar {
                let _ = writeln!(out, "{day} {}", day.strftime("%A"));
            }
        }
        None => {
            let _ = writeln!(
                out,
                "Reference date: unknown, in {}. Don't resolve relative times.",
                input.timezone
            );
        }
    }

    out.push('\n');
    match (&input.source_kind, &input.speaker) {
        (SourceKind::Turn, Some(speaker)) => {
            let owner = if speaker.owner { ", the user" } else { "" };
            let _ = writeln!(
                out,
                "Speaker: {} ({}{owner}). \"I\" and \"me\" in the message mean them.",
                speaker.name, speaker.handle
            );
        }
        (SourceKind::Turn, None) => out.push_str("Speaker: unknown.\n"),
        (SourceKind::Document, _) => {
            out.push_str(
                "This is a document, not a conversation. remember_this is always false.\n",
            );
        }
    }

    out.push_str("\nEntities:\n");
    for candidate in &input.candidates {
        let _ = writeln!(
            out,
            "- {}: {} ({}). Aliases: {}.",
            candidate.handle,
            candidate.name,
            kind_name(candidate.kind),
            candidate.aliases.join(", ")
        );
        for memory in &candidate.memories {
            let _ = writeln!(out, "  - {memory}");
        }
    }

    out.push_str("\nMemories in the assistant's context:\n");
    if input.in_context.is_empty() {
        out.push_str("(none)\n");
    }
    for memory in &input.in_context {
        let _ = writeln!(out, "- {}: {}", memory.handle, memory.content);
    }
    if !input.entries.is_empty() {
        out.push_str("\nEntries in the assistant's context, with the memories each rests on:\n");
        for entry in &input.entries {
            let _ = writeln!(
                out,
                "- {}: {} [rests on {}]",
                entry.handle,
                entry.text,
                entry.cites.join(", ")
            );
        }
    }

    if !input.context.is_empty() {
        out.push_str("\nContext, for understanding the text only. Never quote from it:\n");
        for passage in &input.context {
            let _ = write!(out, "<context>\n{passage}\n</context>\n");
        }
    }

    out.push('\n');
    match input.reply_start {
        Some(start) => {
            let reply: String = input.text.chars().skip(start).take(60).collect();
            let _ = writeln!(
                out,
                "The text is the speaker's message, a blank line, then the assistant's reply, which begins \"{reply}\"."
            );
        }
        None => out.push_str("The text is a section of the document.\n"),
    }
    let _ = write!(out, "<text>\n{}\n</text>\n", input.text);
    out
}

fn kind_name(kind: EntityKind) -> &'static str {
    kind.as_str()
}

fn nullable_string() -> Value {
    json!({"type": ["string", "null"]})
}

fn nullable_enum(values: &[&str]) -> Value {
    let mut values: Vec<Value> = values.iter().map(|value| json!(value)).collect();
    values.push(Value::Null);
    json!({"type": ["string", "null"], "enum": values})
}

fn time() -> Value {
    json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "at": {"type": "string"},
                    "precision": {"type": "string", "enum": ["year", "month", "day", "hour", "minute"]},
                },
                "required": ["at", "precision"],
                "additionalProperties": false,
            },
            {"type": "null"},
        ]
    })
}

fn schema() -> Value {
    let entity_kinds: Vec<&str> = EntityKind::ALL.iter().map(|kind| kind.as_str()).collect();
    let link = json!({
        "type": "object",
        "properties": {
            "entity": nullable_string(),
            "new_name": nullable_string(),
            "new_kind": nullable_enum(&entity_kinds),
            "surface_form": {"type": "string"},
        },
        "required": ["entity", "new_name", "new_kind", "surface_form"],
        "additionalProperties": false,
    });
    let claim = json!({
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "kind": {"type": "string", "enum": ["fact", "event", "state", "task", "recurring"]},
            "quote": {"type": "string"},
            "significance": {"type": "string", "enum": ["trivial", "minor", "notable", "major", "critical"]},
            "remember_this": {"type": "boolean"},
            "changes_something": {"type": "boolean"},
            "valid_from": time(),
            "valid_until": time(),
            "window_confidence": {"type": "string", "enum": ["high", "low"]},
            "until_event": nullable_string(),
            "due_at": time(),
            "volatility": nullable_enum(&["hours", "days", "weeks", "months", "years"]),
            "recurrence_text": nullable_string(),
            "recurrence_rrule": nullable_string(),
            "recurrence_start": time(),
            "entities": {"type": "array", "items": link},
        },
        "required": [
            "content", "kind", "quote", "significance", "remember_this", "changes_something",
            "valid_from", "valid_until", "window_confidence", "until_event", "due_at",
            "volatility", "recurrence_text", "recurrence_rrule", "recurrence_start", "entities",
        ],
        "additionalProperties": false,
    });
    json!({
        "type": "object",
        "properties": {
            "claims": {"type": "array", "items": claim},
            "used_injected_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["claims", "used_injected_ids"],
        "additionalProperties": false,
    })
}
