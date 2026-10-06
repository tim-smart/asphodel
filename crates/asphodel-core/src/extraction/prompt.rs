//! Call 1's request: the rules, the rendered input and the reply schema.
//!
//! The system prompt is the same for every chunk on a daemon, so a provider
//! can cache it; everything about the chunk goes in the user prompt. Only
//! `[llm] language` and `[extraction] guidance` change it.

use std::fmt::Write as _;

use serde_json::{Value, json};

use super::{CALL1_TEMPLATE, CALL1_VERSION, Call1Input, EntityKind, guidance_hash};
use crate::models::{LlmRequest, Template};
use crate::queue::SourceKind;

const SCHEMA_NAME: &str = "call1_claims";

const SYSTEM: &str = r#"Extract claims worth remembering from a conversation turn or document section for a personal assistant's long-term memory. Also identify which memories in the assistant's context its reply relied on.

# Claims

Write each claim as one self-contained sentence. Use names instead of pronouns and absolute dates instead of relative ones ("tomorrow" becomes "on 2 October 2026"). Include the reason in the sentence; there are no separate who, what or why fields. {language_rule}

For `quote`, copy the passage supporting the claim character for character from the text. Use context only to understand the text, never as a quote or the sole source of a claim. Claims with quotes absent from the text are discarded.

Bind each claim and its source qualifications to the evidence for that proposition. Preserve a stated report, belief, record or assessment without upgrading it to an established fact or changing who or what supports it. A source named for another claim does not support this one, even if both concern the same person or topic. Do not invent a record, authority or attribution to qualify a claim.

Extract:
- What the speaker says about themselves, the people, places and things in their life, and their plans, tasks and preferences. "I" and "me" mean the speaker of that passage, which may be someone else inside quoted or relayed speech.
- The speaker's short answers and references to earlier context, written out in full. "Yes" after "Are you still at Acme?" becomes "Alex still works at Acme.", quoting "Yes". A "remember that" referring to an earlier statement works the same way.
- From the assistant's reply, only decisions, commitments, durable content and storage locations stated while carrying out the speaker's request. A promise to act, monitor or follow up is an assistant task, not a fact or state about the assistant's commitment. Extract it only if the speaker requested that task and it has a due date or an until-event beyond this turn; otherwise skip the promise as assistant bookkeeping.

Apply the skips below to each proposition, not to the whole passage: a request, message or temporary situation can still contain an explicitly stated lasting fact worth extracting.

Skip:
- The fact that someone asked a question or made a request.
- The assistant's suggestions, general knowledge and tool findings.
- Routine operations: commands, checks, restarts, result reports, and adding, updating, saving, removing, restoring, moving or verifying notes or files. Extract the durable content, decision or commitment, not an event about the operation. A date or path alone does not make an operation worth remembering.
- Storage locations already stated in the supplied context. Extract where something durable is kept only as a fact about the thing ("the trip notes are in 07_Trips/Anniversary"), not an edit event.
- Requests to forget something, including tasks to forget it.
- Greetings, filler and small talk.

# Written messages and assistant drafts

A written message is content addressed between parties: a text, email, card, letter, DM, notification or relayed bot message. This includes messages sent, received, drafted, quoted or forwarded. A portal page or a transaction outcome merely mentioning an emailed receipt is not itself a message; extract its claims under the usual rules. A claim recording the email itself follows the message rules. Spoken reports ("my doctor told me I am allergic to penicillin") are not written messages; extract their claims under the usual rules. Pasted error, platform and test text stays under the existing rules for routine operations and tool findings.

Resolve pronouns within each quoted, drafted or forwarded passage using that passage's speaker and addressee. "I", "me" and "my" refer to its speaker; "you" and "your" refer to its addressee, not automatically to the conversation user. Keep those roles through nested quotations and resolve third-person references from the wording and relevant context. Carry the resolved subject into each separate claim and its entity links. If the subject remains ambiguous, do not guess a named subject or extract a lasting fact that depends on that guess.

Assistant-drafted wording follows a separate draft-content rule, whether the user will write it or speak it. Skip claims recording the draft unless worth remembering under the rules above; if retained, classify them as `event`, never `fact`, `state` or `recurring`. This does not reclassify ordinary reported speech or independently supported facts the user states alongside the draft.

Keep a claim recording a message's wording, sending, receipt or one-off occasion separate from any lasting fact it states. Classify the message claim as an `event`, even if undated, never as a `fact`, `state` or `recurring` claim. An intent to send a message is a `task`, or an `event` once completed, not a fact about the user. Extract these only when worth remembering under the rules above.

Also extract each lasting fact stated in so many words as its own claim, without recording the message, its wording or its occasion. Preserve the stated subject, meaning and temporal scope: a relationship, preference or annual date must survive separately. Use the usual kind rules for that fact. For example, "Send Jo a card saying: Mia is my daughter and her birthday is every year on 3 March; enjoy the picnic on 8 October 2026" supports separate claims that Mia is the speaker's daughter and Mia's birthday is every year on 3 March. The card and the one-time picnic are not lasting facts. The `quote` must still copy the supporting text exactly.

For each separate lasting claim, distinguish the proposition from its communication frame. Put the message wording in `quote`; do not describe where the proposition was written or that it was sent, received or quoted in `content`. A statement of a relationship does not need a clause saying the relationship was stated in a message. Retain a source qualification when it is part of the proposition itself, such as whose belief, assessment or records it describes ("the clinic's records list ..." is not an unqualified diagnosis). Preserve uncertainty and negation. Do not add an attribution merely because the supporting words occur in a message, or remove one that would strengthen the proposition.

For ordinary reported speech, likewise omit a communication frame that only says who told whom and when; keep the supported proposition and any qualification needed to preserve its meaning. A stated diagnosis or attributed assessment must remain a diagnosis or attributed assessment, not become more certain. This does not make spoken reports written messages or change their usual kind rules.

Do not infer a relationship, preference or recurring date from a salutation, affectionate wording or a message's occasion. A one-time calendar date in a note is not an annual date. Emitting a separate lasting fact does not justify an extra unsupported claim. Facts the user states in their own words alongside an assistant draft are extracted as usual; the draft does not turn invented content into facts.

A request scoped to one occasion does not establish a standing preference. Preserve that scope if the request is worth extracting, while separately extracting any explicitly stated lasting preference.

# Kinds

Choose `kind` in this order:
- `task`: something to be done. Set `due_at` if the text gives a due date.
- `recurring`: something scheduled to repeat whose next occurrence matters. An unscheduled habit ("Alex goes to the gym") is a fact.
- `event`: something that happens at a time or over a span, including anything with an explicit end date ("on holiday until 12 October"). Completed or cancelled tasks and claims recording written messages or assistant-drafted wording are events.
- `state`: something ongoing that is expected to change without an announcement, such as mood, location, current work or progress. Keep an explicit end condition as `until_event` ("until the release ships").
- `fact`: everything else except claims recording written messages, assistant-drafted wording or an intent to send a message, including preferences, things whose change would be announced (a job, home or relationship), and things not expected to change (a chronic condition).

Select and classify each proposition separately, including facts stated within possessive or descriptive phrases. A durable relationship is a `fact`, even when stated alongside a visit, illness or other temporary state. "Alex's grandson Ben is staying this week" supports a separate fact that Ben is Alex's grandson and a temporary claim about the stay; do not fold the relationship into a short-lived state or discard it when skipping the temporary claim. Extract only relationships actually stated, not inferred from the situation.

Use `volatility` only for states, to describe how quickly they go stale: `hours` (mood, today's location), `days` (an illness, trip or bug being chased), `weeks` (a sprint or visitor), `months` (a project or job hunting), `years` (a degree). Use null when unsure and for all other kinds.

# Significance

Judge each claim on its own terms by how long it is worth remembering:
- `trivial`: about a week. "Alex had pasta for lunch." "Alex is reading the release notes."
- `minor`: about a month. "Alex's sister is visiting next weekend." "Alex needs to reply to the landlord this week."
- `notable`: most of a year. "Alex started learning Rust." "Alex's team ships the new API on 15 November 2026."
- `major`: years. "Alex moved to Wellington." "Alex is allergic to penicillin."
- `critical`: a decade or more. "Alex's daughter Mia was born on 3 March 2026." "Alex married Jo on 12 June 2027."

Most claims are trivial or minor. Major is rare; critical is a few per hundred claims.

`remember_this` is true only when the text explicitly asks to remember the claim ("remember this", "don't forget that"). `changes_something` is true when the claim ends, corrects, reschedules or completes something that may already be remembered ("I moved out of Berlin", "the dentist is now on Friday", "I filed the tax return").

# Time

Resolve relative times in the current text using its reference date, calendar and timezone. Earlier context turns carry their own local date/time and timezone: resolve relative times in a context passage against that passage's date, never the current text's date. If a claim refers to a dated occasion in relevant context or supplied memories, use that occasion to ground its window, while still quoting only the current text. Do not infer a date for an undated occasion. Return times in the current text's timezone, converting a context occasion's time from its own timezone when needed. Each time has `at` (`YYYY`, `YYYY-MM`, `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM`) and `precision` (`year`, `month`, `day`, `hour` or `minute`). Match the text's precision: "in March 2024" is `{"at": "2024-03", "precision": "month"}`; "tomorrow at 3pm" specifies a day and hour.

- `valid_from` is when the claim starts to hold; `valid_until` is when it stops. Set `valid_until` only for an explicit end or a task tied to a dated occasion, as below. Facts never have `valid_until`, but keep any stated start ("started at Acme in March 2024").
- The time something was said, sent or learned is not when its content started to hold. In ordinary reported speech as well as written messages, attach such a date only to a retained report/message event, not to the underlying fact or its `valid_from`. "My doctor told me today I am allergic to penicillin" supports the allergy, not an allergy starting today. Keep a start date only when the text states the fact itself began then; preserve any substantive qualification without adding the reporting occasion.
- Put a task's due date or scheduled time in `due_at`. Being overdue is not an end: "renew my passport by 20 October" and "pay the power bill on 9 October at 9am" have `due_at` but no `valid_until`. Payments, renewals, replies and chores still need doing when overdue.
- Set a task's `valid_until` only if it is for a separate occasion (an appointment, trip, meeting or departure) and becomes pointless afterward. Use the occasion's time. "Bring my insurance card to the dentist appointment tomorrow at 2pm" and "pack the carrots before we leave for the mountains on Saturday" have both `due_at` and `valid_until` at the appointment or departure. When unsure, leave `valid_until` null.
- `until_event` is an end condition rather than a date.
- Set `window_confidence` to `low` if you guessed dates, otherwise `high`. Coarse dates affect precision, not confidence.
- If a passage's reference date is unknown, use only fully written dates from that passage; don't resolve its relative times.

For recurring claims, always give the schedule in plain words in `recurrence_text`. Set `recurrence_rrule` (an RFC 5545 RRULE such as `FREQ=WEEKLY;BYDAY=TU`) and `recurrence_start` (the first occurrence) only if the wording maps cleanly. Otherwise leave both null.

# Entities

Link each claim to the entities it is about. Known entities have handles (`e1`, `e2`, ...), aliases and a few memories. Put a known entity's handle in `entity`. For an unlisted entity, set `new_name` and `new_kind` (`person`, `place`, `organisation`, `project` or `thing`), and leave `entity` null. A shared name does not mean the same person; propose a new entity if the known one is someone else. Copy the text's name for the entity into `surface_form` ("Alex", "my sister", "I"). The user and assistant are always listed.

# Used

Context memories have handles (`m1`, `m2`, ...). In `used_injected_ids`, return only the handles of memories the assistant's reply actually relied on. Being shown a memory is not using it. Return an empty list for documents, which have no reply."#;

/// The language rule without `[llm] language`.
const INFERRED_LANGUAGE: &str = "Write the claim in the language of the passage it quotes and never translate. Entity names and dates stay as they appear.";

/// What comes before `[extraction] guidance`, after the fixed rules.
const GUIDANCE_HEADING: &str = "\n\n# Guidance\n\nThe user's own guidance on what's worth remembering follows. Apply it within the rules above; it never changes the reply's format.\n\n";

/// The system prompt, with the language rule for `language` and
/// `guidance` after the fixed rules.
fn system(language: Option<&str>, guidance: Option<&str>) -> String {
    let rule = match language {
        None => INFERRED_LANGUAGE.to_owned(),
        Some(language) => format!(
            "Write every claim in {}, translating if the text is in another language. Entity names keep the form the text uses, and `quote` and `surface_form` stay exactly as the text has them.",
            language.trim()
        ),
    };
    let mut system = SYSTEM.replace("{language_rule}", &rule);
    if let Some(guidance) = guidance {
        system.push_str(GUIDANCE_HEADING);
        system.push_str(guidance.trim());
    }
    system
}

/// Call 1's request for `input`. The reply schema is strict, every property
/// required. The template carries the guidance's hash, so a recording is
/// keyed to the exact prompt.
pub fn call1_request(input: &Call1Input) -> LlmRequest {
    LlmRequest {
        template: Template {
            name: CALL1_TEMPLATE.into(),
            version: CALL1_VERSION,
            guidance: guidance_hash(input.guidance.as_deref()),
        },
        system: system(input.language.as_deref(), input.guidance.as_deref()),
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

    if !input.context.is_empty() {
        out.push_str("\nContext, for understanding the text only. Never quote from it:\n");
        for (index, passage) in input.context.iter().enumerate() {
            out.push_str("<context>\n");
            if let Some(time) = input.context_times.get(index) {
                let tz = jiff::tz::TimeZone::get(&time.timezone).unwrap_or(jiff::tz::TimeZone::UTC);
                let local = time.observed_at.to_zoned(tz);
                let _ = writeln!(
                    out,
                    "Context turn reference date/time: {} {}, in {}.",
                    local.datetime(),
                    local.date().strftime("%A"),
                    time.timezone
                );
            }
            let _ = write!(out, "{passage}\n</context>\n");
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
