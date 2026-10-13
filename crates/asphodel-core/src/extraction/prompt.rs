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

const SYSTEM: &str = r#"Extract claims worth remembering from a conversation turn or document section for an AI assistant's long-term memory. Also identify which memories in the assistant's context its reply relied on.

# Claims

Write each claim as one self-contained sentence. Use names instead of pronouns and absolute dates instead of relative ones ("tomorrow" becomes "on 2 October 2026"). Name what a task, commitment or payment is about, resolving references from context: "Hermes will start paying the ACC invoice by credit card", not "Hermes will start the credit-card payment". Include the reason in the sentence; there are no separate who, what or why fields. {language_rule}

For `quote`, copy the passage supporting the claim character for character from the text. Use context only to understand the text, never as a quote or the sole source of a claim. Claims with quotes absent from the text are discarded.

Bind each claim and its attribution to the evidence for that proposition. Preserve a stated report, belief, record or assessment without upgrading it to an established fact or changing who or what supports it. A source named for another claim does not support this one, even if both concern the same person or topic. Do not invent a record, authority or attribution to qualify a claim.

Content of a story, script, role-play or other invented work is about that work, not the world: name the work in the sentence and link it as a `project`, not its characters as people. A fact the speaker states about themselves while asking for it is extracted as usual.

Extract:
- What the speaker says about themselves, the people, places and things in their life and work, and their plans, tasks, decisions and preferences, including how they like things done. "I" and "me" mean the speaker of that passage, which may be someone else inside quoted or relayed speech. In a document, the author is the speaker.
- The speaker's short answers and references to earlier context, written out in full. "Yes" after "Are you still at Acme?" becomes "Alex still works at Acme.", quoting "Yes". A "remember that" referring to an earlier statement works the same way.
- From the assistant's reply, decisions the speaker made or approved, commitments, and content the speaker adopted or asked to keep and where it is stored, stated while carrying out the speaker's request. A promise to act, monitor or follow up is an assistant task, not a fact or state about the assistant's commitment. Extract it only if the speaker requested that task and it has a due date or an until-event beyond this turn; otherwise skip the promise as assistant bookkeeping.
- From the assistant's reply, a lasting finding about something of the speaker's, such as a home, vehicle, device, account or piece of work, that the text establishes rather than guesses: a constraint it has or the cause of a recurring problem. Keep its source in the sentence.
- From the assistant's reply, a finding that answers a request about the speaker's life or work and that the text dates or bounds (a local event, a booking, an availability). Keep its source in the sentence ("the festival listing says ...") and quote the reply.
- What the speaker wanted or asked about, when it shows a need, plan or interest of theirs beyond the step at hand: a purchase, a trip, a hire, a topic they are learning, a library they are weighing. Skip requests that are steps in work already under way. Extract an `event` recording the intent, not the asking: "Alex wanted to try the Corner Cafe's online ordering on 7 October 2026, without buying." Do not record whether the assistant acted or the outcome unless the text states it. Usually `trivial`.
- The speaker saying that a plan, task or piece of work in the supplied memories or context is done, dropped or no longer needed, including by asking the assistant to stop, delete or discard the work done for it. Extract an `event` with `changes_something` true that names the work as the memories or context name it and gives the reason if the text does: "Alex no longer needs the Lisbon flight search, having booked the flights." Quote the current text. When the text already states what ended the work, such as booking the flights, extract that one claim with `changes_something` true and no separate one. A request to clear up files or notes whose work continues is a routine operation, not this. Cancelling, deleting or moving a reminder or calendar entry never ends its obligation, as below.

Apply the skips below to each proposition, not to the whole passage: a request, message or temporary situation can still contain an explicitly stated lasting fact worth extracting.

Skip:
- The act of asking a question or making a request. For a question or request, record only what the speaker wanted, as above.
- Requests whose only content is the assistant operating its own tools, notes or system, unless they end remembered work as above.
- The assistant's advice, suggestions, general knowledge, undated findings other than the lasting ones above, and findings that don't answer the speaker's request.
- Routine operations: commands, checks, restarts, result reports, and adding, updating, saving, removing, restoring, moving or verifying notes, files or code. Extract the durable content, decision or commitment, or the end of the work it served as above, not an event about the operation. A date or path alone does not make an operation worth remembering.
- Storage locations already stated in the supplied context. Extract where something durable is kept only as a fact about the thing ("the trip notes are in 07_Trips/Anniversary"), not an edit event.
- What the work itself records and the assistant can read again: document text, code and its structure, commit or ticket history. Keep the speaker's reason, constraint or rule behind it.
- Requests to forget something, meaning to stop remembering it, including tasks to forget it. Dropping a plan in the same words ("forget the flight search, I've booked") ends work as above.
- Greetings, filler and small talk.

# Reminder and calendar requests

Separate the reminder/calendar operation from the underlying obligation or occasion. Skip claims about asking for, setting, changing or confirming a reminder or calendar entry: these are assistant bookkeeping, not open user tasks or events recording the request. An explicitly confirmed operation must not become a task labelled "requested", "wanted" or "unresolved". This skip applies only to the operation, never to the supported obligation or occasion. Completing the operation does not complete the underlying obligation.

Do not infer fulfilment from a request, a promise, a bare acknowledgement such as "Sure!", or an unsuccessful attempt. Uncertainty, failure or refusal does not by itself establish a task for anyone. Do not create an "unresolved" reminder/calendar task merely because completion is unconfirmed. Retain a supported user obligation whether the assistant operation succeeds, fails, is declined or is ambiguous; do not require an explicit failure before retaining it. Name the person who actually owes the underlying action. Extract a future assistant task only under the requested, dated or until-event commitment rule above, not from the request alone. Do not invent a due date or claim that a reminder or calendar entry exists.

Extract the underlying user obligation or planned occasion separately when the text supports it, even when the assistant action is fulfilled or skipped. A calendar request can itself state an occasion: "Add my dentist appointment on 8 October at 2pm to the calendar" supports an event that the speaker has a dentist appointment at that time, not a task to add it or an event about asking. Retain a supported occasion even if the calendar operation fails or receives only an acknowledgement. Use the usual kind rules: an obligation still to be done is a task; a scheduled occasion is an event.

Keep the obligation's own deadline in `due_at` and the occasion's own time in its window. A reminder's notification time is not the obligation's deadline unless the text says they coincide. For example, "Remind me on 5 October to pay the invoice by 8 October" supports a task to pay the invoice by 8 October, not a task due on 5 October. "Remind me on 5 October to pay the invoice" supports an undated invoice-payment task: leave `due_at`, `valid_from` and `valid_until` null, rather than borrowing the notification time. This applies whether the reminder succeeds, fails, is declined or is ambiguous. Only a separately supported obligation date or occasion can populate those fields. Do not fold the reminder operation into the underlying claim or infer that an obligation was completed, cancelled or rescheduled just because its reminder changed.

A reminder-only reschedule does not set `changes_something` on the underlying obligation or change its `due_at` or `valid_until`. When the current text names or refers to the underlying action, including through a named reminder, restate that supported obligation as a task even if it is already in context. Resolve the reference and preserve its own deadline from relevant context, or leave it undated if no obligation deadline is stated. Quote the supporting current text, not context, and omit the reminder operation from the claim. For example, "Move my invoice-payment reminder to 6 October" supports restating the invoice-payment obligation with its existing deadline, not a new deadline of 6 October. "Move my reminder to 6 October; the invoice is still due on 8 October" supports retaining the invoice obligation due on 8 October. This restatement must not end the existing obligation or assert a change to it. If the text refers only to the reminder mechanism ("Move tomorrow's reminder to 4pm"), skip the operation without inventing an obligation claim from context alone. If the text explicitly changes the obligation's own deadline or the occasion's time, extract that actual change under the usual rules.

# Written messages and assistant drafts

A written message is content addressed between parties: a text, email, card, letter, DM, notification or relayed bot message. This includes messages sent, received, drafted, quoted or forwarded. A portal page or a transaction outcome merely mentioning an emailed receipt is not itself a message; extract its claims under the usual rules. A claim recording the email itself follows the message rules. Spoken reports ("my doctor told me I am allergic to penicillin") are not written messages; extract their claims under the usual rules. Pasted error, platform and test text stays under the existing rules for routine operations and tool findings.

Resolve pronouns within each quoted, drafted or forwarded passage using that passage's speaker and addressee. "I", "me" and "my" refer to its speaker; "you" and "your" refer to its addressee, not automatically to the conversation user. Keep those roles through nested quotations and resolve third-person references from the wording and relevant context. Carry the resolved subject into each separate claim and its entity links. If the subject remains ambiguous, do not guess a named subject or extract a lasting fact that depends on that guess.

Assistant-drafted wording follows a separate draft-content rule, whether the user will write it or speak it. Skip claims recording the draft unless worth remembering under the rules above; if retained, classify them as `event`, never `fact`, `state` or `recurring`. This does not reclassify ordinary reported speech or independently supported facts the user states alongside the draft.

Keep a claim recording a message's wording, sending, receipt or one-off occasion separate from any lasting fact it states. Classify the message claim as an `event`, even if undated, never as a `fact`, `state` or `recurring` claim. An intent to send a message is a `task`, or an `event` once completed, not a fact about the user. Extract these only when worth remembering under the rules above.

Also extract each lasting fact stated in so many words as its own claim, without recording the message, its wording or its occasion. Preserve the stated subject, meaning and temporal scope: a relationship, preference or annual date must survive separately. Use the usual kind rules for that fact. For example, "Send Jo a card saying: Mia is my daughter and her birthday is every year on 3 March; enjoy the picnic on 8 October 2026" supports separate claims that Mia is the speaker's daughter and Mia's birthday is every year on 3 March. The card and the one-time picnic are not lasting facts. The `quote` must still copy the supporting text exactly.

For each separate lasting claim, distinguish the proposition from its communication frame. Put the message wording in `quote`; do not describe where the proposition was written or that it was sent, received or quoted in `content`. A statement of a relationship does not need a clause saying the relationship was stated in a message. Retain a source qualification when it is part of the proposition itself, such as whose belief, assessment or records it describes ("the clinic's records list ..." is not an unqualified diagnosis). Preserve uncertainty and negation. Do not add an attribution merely because the supporting words occur in a message, or remove one that would strengthen the proposition.

For ordinary reported speech, likewise omit a communication frame that only says who told whom and when; keep the supported proposition and any qualification needed to preserve its meaning. A stated diagnosis or attributed assessment must remain a diagnosis or attributed assessment, not become more certain. This does not make spoken reports written messages or change their usual kind rules.

Do not infer a relationship, preference or recurring date from a salutation, affectionate wording or a message's occasion. A one-time calendar date in a note is not an annual date. Emitting a separate lasting fact does not justify an extra unsupported claim. Facts the user states in their own words alongside an assistant draft are extracted as usual; the draft does not turn invented content into facts.

# Kinds

Choose `kind` in this order:
- `task`: something to be done. Set `due_at` if the text gives a due date.
- `recurring`: something scheduled to repeat whose next occurrence matters. An unscheduled habit ("Alex goes to the gym") is a fact.
- `event`: something that happens at a time or over a span, including anything with an explicit end date ("on holiday until 12 October"). Completed or cancelled tasks and claims recording written messages or assistant-drafted wording are events.
- `state`: something ongoing that is expected to change without an announcement, such as mood, location, current work or progress. Keep an explicit end condition as `until_event` ("until the release ships").
- `fact`: everything else except claims recording written messages, assistant-drafted wording or an intent to send a message, including preferences, things whose change would be announced (a job, home or relationship), and things not expected to change (a chronic condition).

A request scoped to one occasion does not establish a standing preference. Preserve that scope if the request is worth extracting, while separately extracting any explicitly stated lasting preference.

Permission for the assistant to use an account, card or other resource is not a task. Skip one given for the request at hand ("use my work card for this top-up") as assistant bookkeeping; a standing one ("you can always use my work card for SIM top-ups") is a `fact`. A decision that an approval carries ("yes, book the 200 euro hotel") is extracted as usual.

Select and classify each proposition separately, including facts stated within possessive or descriptive phrases. A durable relationship is a `fact`, even when stated alongside a visit, illness or other temporary state. "Alex's grandson Ben is staying this week" supports a separate fact that Ben is Alex's grandson and a temporary claim about the stay; do not fold the relationship into a short-lived state or discard it when skipping the temporary claim. Extract only relationships actually stated, not inferred from the situation.

Use `volatility` only for states, to describe how quickly they go stale: `hours` (mood, today's location), `days` (an illness, trip or bug being chased), `weeks` (a sprint or visitor), `months` (a project or job hunting), `years` (a degree). Use null when unsure and for all other kinds.

# Significance

Judge each claim on its own terms by how long it stays useful without being mentioned again, not by how dramatic it is:

- `trivial`: about a week. "Alex had pasta for lunch." "Alex is reading the release notes."
- `minor`: about a month. "Alex's sister is visiting next weekend." "Alex needs to reply to the landlord this week."
- `notable`: a few months. "Alex started learning Rust." "Alex's team ships the new API on 15 November 2026."
- `major`: years. "Alex moved to Wellington." "Alex is allergic to penicillin." "Alex owns a Ryobi drill." "Alex's shop website runs on Shopify."
- `critical`: a decade or more. "Alex's daughter Mia was born on 3 March 2026." "Alex married Jo on 12 June 2027."

Rate a claim `major` when it would still shape how the assistant helps a year from now, however mundane: the people and relationships, home, job and health conditions in the speaker's life; what they own, use or have set up, such as a tool, appliance, car, device, account or piece of software; their standing preferences and rules for how their work is done; the reasons behind lasting choices and conclusions they have reached. A choice, constraint or finding tied to one task or occasion is `notable` at most. A recurring claim outlasts its period: a yearly occasion is at least `major`. Passing activity, short plans and progress are trivial or minor.

`remember_this` is true only when the text explicitly asks to remember the claim ("remember this", "don't forget that"). `changes_something` is true when the claim ends, corrects, reschedules or completes something that may already be remembered, or makes a remembered task unnecessary ("I moved out of Berlin", "the dentist is now on Friday", "I filed the tax return", "I booked the flights" after a flight search, "the repo now uses pnpm").

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

    if !input.upcoming.is_empty() {
        out.push_str("\nUpcoming occasions already remembered, for grounding dates only. Never quote from them:\n");
        for event in &input.upcoming {
            let _ = writeln!(out, "- {}", event.content);
        }
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
