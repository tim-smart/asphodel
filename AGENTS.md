# Hermes data evaluations

Follow `docs/hermes-data-evaluation.md` for evaluations against Tim Smart's
private Hermes history. Route future live comparisons to Mac Developer on
Tim's machine.

Tim granted standing authorization on 2026-10-04 for live comparisons,
including live recordings and LLM-backed fast top-ups, using
`https://chatgpt.com/backend-api/codex`, `auth = "chatgpt"`,
`model = "gpt-6-luna"` and `reasoning_effort = "low"`. No per-run approval
is needed within that boundary while the data stays private. Ask Tim before
changing the endpoint, auth, model or reasoning setting.

Keep raw history, memory text, queries, cassettes and other private artifacts
inside `ASPHODEL_REPLAY_DIR`, outside git trees and `~/multica_workspaces`.
Share only the typed aggregate export and the feedback template
(IDs and numbers, no private content). This guidance is a rule for agents,
not a technical privacy guarantee.

Standing authorization does not approve labels, probes or probe re-anchors.
Tim must still approve each exact entry as required by the evaluation guide;
floor selection and other review decisions remain his.

Before making decisions based on evaluation results, ask Mac Developer to
obtain an independent review from a Fable sub-agent. The review must follow
the same privacy restrictions above; share only the permitted aggregate
export and feedback template, never private content. If that review is
unavailable, leave the decision pending rather than bypassing it. This
review does not replace Tim's approval of exact labels, probes or probe
re-anchors, or his floor selection and other review decisions.
