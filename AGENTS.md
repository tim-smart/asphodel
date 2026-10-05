# Tests

## Running tests

```
nix develop -c cargo nextest run
```

On linux, prefix the nextest run with `TMPDIR=/dev/shm`.

## Writing tests

Test behavior through public seams only:

- `Service` and the other public core API it hands out
- the HTTP API (`asphodel serve`)
- public clients tested against stub servers for external HTTP contracts
- the CLI
- the plugin's public surface (the Hermes provider and its tools)
- replay scenarios in `scenarios/*.toml`

If a test has to reach past these seams, either the behavior isn't worth
pinning or the seam is missing. Add the seam or drop the test. `pub` on a
module is not a seam by itself.

Don't assert on prompt wording, template version numbers, tuning defaults,
log or status-message text, row ids, or table layout. These change on
purpose, and a test that breaks when they do is noise. Assert the behavior
they produce instead: "a claim made under different rules is not reused",
not "the template version is 7".

A bug fix gets a regression test only when the bug is visible at a public
seam. Extend an existing test or scenario before adding a new `#[test]`.

Raw SQL in tests (`store.connection()` and friends) is allowed only to:

- prove data is gone after erase, forget, delete or other privacy paths
- set up a state the API can't reach, such as a corrupt row or a
  pre-migration schema

Everywhere else, read through the public API. If no public read exists,
adding a small one is fine; call it out in the PR.

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
