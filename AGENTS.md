# Hermes data evaluations

Follow `docs/hermes-data-evaluation.md` for evaluations against Tim Smart's
private Hermes history. Route future live comparisons to Mac Developer on
Tim's machine.

Tim granted standing authorization on 2026-10-04 for live comparisons,
including live recordings and LLM-backed fast top-ups, using
`https://chatgpt.com/backend-api/codex`, `auth = "chatgpt"`,
`model = "gpt-6-luna"` and `reasoning_effort = "low"`. No per-run approval
is needed within that boundary while the data stays private. Changes to the
endpoint, auth, model or reasoning setting require independent Fable
sub-agent review arranged by Mac Developer before use, under the same
privacy restrictions.

Keep raw history, memory text, queries, cassettes and other private artifacts
inside `ASPHODEL_REPLAY_DIR`, outside git trees and `~/multica_workspaces`.
Share only the typed aggregate export and the feedback template
(IDs and numbers, no private content). This guidance is a rule for agents,
not a technical privacy guarantee.

Standing authorization does not accept evaluation data or result decisions.
Before any evaluation decision, ask Mac Developer to arrange independent
Fable sub-agent review. This includes exact probes and re-anchors, exact
labels, recall and call-2 floor selection, significance tuning, spot-check
verdicts, distribution assessments and backend changes. Mac Developer makes
decisions only after sufficient independent review. No evaluation decision
returns to Tim for approval.

The review must follow the same privacy restrictions above; share only the
permitted aggregate export and feedback template, never private content.
If review is unavailable or the permitted material is insufficient, leave
the decision pending rather than bypassing review or broadening access.
Keep pending probes and labels in private drafts, not the accepted files;
retain current floors and settings until changes pass review. Record the
reviewer, date, exact scope, outcome and pending decisions in the feedback
without private content.
