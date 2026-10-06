//! Call 2's request and reply: the claims, their neighbours and the labels.
//!
//! The system prompt is the same for every chunk, so a provider can cache
//! it; everything about the chunk goes in the user prompt.
//! The prompt never asks which of a claim and a neighbour is newer: code
//! decides direction from `observed_at`, so an old document can't
//! overrule a newer memory.

use std::fmt::Write as _;

use serde::Deserialize;
use serde_json::{Value, json};

use super::{CALL2_TEMPLATE, CALL2_VERSION, Call2Input, Label};
use crate::models::{LlmRequest, Template};
use crate::strength::Kind;

const SCHEMA_NAME: &str = "call2_labels";

const SYSTEM: &str = r#"You reconcile new claims with a personal assistant's existing memories. You are given claims just extracted from a text, each with a handle (`c1`, `c2`, …), and the stored memories closest to them, each with a handle (`n1`, `n2`, …). For each claim, say what it does to the memories it's about.

# Labels

Give a claim a label on a memory only when the two are about the same thing:
- `mentioned_again`: the claim states what the memory already says, independently. Rewording, or the same fact with less detail, is still mentioned again.
- `confirmed`: the claim says the memory is still right, such as "yes" to a question about it, or "still" true.
- `refines`: the claim is a more precise version of the same statement, and the memory wasn't wrong ("going to Japan in 2027" refined by "going to Tokyo in April 2027"; "needs to pack the carrots" refined by "reminder to pack the carrots at 8am on the 30th"). Adding a date or schedule is a refinement, not a repeat, even when only the structured time fields show it.
- `retracts`: the claim is a corrected version of the memory, which was wrong: a correction, or a rescheduled appointment or plan ("the dentist is on Friday, not Thursday", "her name is Mia, not Maya", "I filed the tax return on the 2nd, not the 1st").
- `denies`: the claim says the memory didn't happen or isn't true at all, with nothing to replace it ("I haven't filed the tax return after all", "the trip to Japan isn't happening"). When unsure between `denies` and `retracts`, use `retracts`.
- `ends`: the memory was true and the claim says it stopped being true: a move, a job left, a habit stopped, or a task completed or cancelled ("moved out of Berlin" ends "lives in Berlin"; "filed the tax return" ends "needs to file the tax return").

A claim can have labels on several memories, and several claims can label the same memory. A claim about something new gets no labels. Don't label a memory just because it's on the same topic: "likes tea" and "likes coffee" are two memories.

Each claim and memory shows its due_at, valid_from and valid_until alongside the sentence; none means no date was supplied. Each memory shows when it was said. Memories marked ended have already stopped being true. Judge only what each claim says about each memory; don't decide which is newer, and label a claim the same whichever was said first.

Return every claim that has at least one label, with its labels. Leave out claims with none."#;

/// Call 2's request for `input`. The reply schema is strict, every property
/// required, and the label is an enum of [`Label::ALL`].
pub fn call2_request(input: &Call2Input) -> LlmRequest {
    LlmRequest {
        template: Template {
            name: CALL2_TEMPLATE.into(),
            version: CALL2_VERSION,
            guidance: None,
        },
        system: SYSTEM.into(),
        user: render(input),
        schema_name: SCHEMA_NAME.into(),
        schema: schema(),
        max_tokens: None,
    }
}

/// The user prompt: the claims, then the memories.
fn render(input: &Call2Input) -> String {
    let mut out = String::from("Claims:\n");
    for claim in &input.claims {
        let _ = write!(
            out,
            "{} (said {}): {}",
            claim.handle,
            claim.observed_at.strftime("%Y-%m-%d"),
            claim.content
        );
        render_window(&mut out, claim.due_at, claim.valid_from, claim.valid_until);
        if !claim.neighbours.is_empty() {
            let _ = write!(out, " [closest: {}]", claim.neighbours.join(", "));
        }
        out.push('\n');
    }
    out.push_str("\nMemories:\n");
    for neighbour in &input.neighbours {
        let _ = write!(
            out,
            "{} ({}, said {}{}): {}",
            neighbour.handle,
            kind_name(neighbour.kind),
            neighbour.observed_at.strftime("%Y-%m-%d"),
            if neighbour.ended { ", ended" } else { "" },
            neighbour.content
        );
        render_window(
            &mut out,
            neighbour.due_at,
            neighbour.valid_from,
            neighbour.valid_until,
        );
        out.push('\n');
    }
    out
}

fn render_window(
    out: &mut String,
    due_at: Option<jiff::Timestamp>,
    valid_from: Option<jiff::Timestamp>,
    valid_until: Option<jiff::Timestamp>,
) {
    out.push_str(" [");
    for (index, (name, at)) in [
        ("due_at", due_at),
        ("valid_from", valid_from),
        ("valid_until", valid_until),
    ]
    .into_iter()
    .enumerate()
    {
        if index > 0 {
            out.push_str(", ");
        }
        let _ = write!(
            out,
            "{name}={}",
            at.map_or_else(|| "none".into(), |at| at.to_string())
        );
    }
    out.push(']');
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Fact => "fact",
        Kind::Event => "event",
        Kind::State => "state",
        Kind::Task => "task",
        Kind::Recurring => "recurring",
    }
}

fn schema() -> Value {
    let labels: Vec<&str> = Label::ALL.iter().map(|label| label.as_str()).collect();
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["claims"],
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["claim", "labels"],
                    "properties": {
                        "claim": {"type": "string"},
                        "labels": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": false,
                                "required": ["neighbour", "label"],
                                "properties": {
                                    "neighbour": {"type": "string"},
                                    "label": {"type": "string", "enum": labels},
                                },
                            },
                        },
                    },
                },
            },
        },
    })
}

#[derive(Debug, Deserialize)]
struct Reply {
    claims: Vec<RawClaim>,
}

#[derive(Debug, Deserialize)]
struct RawClaim {
    claim: String,
    labels: Vec<RawLabel>,
}

#[derive(Debug, Deserialize)]
struct RawLabel {
    neighbour: String,
    label: Label,
}

/// One claim's labels: its handle, and each neighbour handle with its label.
pub(super) type ClaimLabels = (String, Vec<(String, Label)>);

/// Call 2's labels by claim handle, then neighbour handle, in reply order.
/// Handles are trimmed but not checked here: one that isn't in the input
/// is ignored later, as call 1 ignores an unknown entity handle. `Err`
/// names why the reply doesn't fit the schema, without its content.
pub(super) fn parse(reply: &Value) -> Result<Vec<ClaimLabels>, &'static str> {
    let reply = Reply::deserialize(reply).map_err(|_| "the reply doesn't fit call 2's schema")?;
    Ok(reply
        .claims
        .into_iter()
        .map(|claim| {
            (
                claim.claim.trim().to_owned(),
                claim
                    .labels
                    .into_iter()
                    .map(|label| (label.neighbour.trim().to_owned(), label.label))
                    .collect(),
            )
        })
        .collect())
}
