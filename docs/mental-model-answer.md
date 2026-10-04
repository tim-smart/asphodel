# Why a mental model is one answer

The decision record for storing a mental model's answer as one prose text
with one set of citations, in place of entries that each cited their own
memories. It says what was decided, what was weighed, and what was given
up. How the shipped code behaves is in `docs/operations.md` ("Mental
models"), `docs/upgrading.md` (schema versions 14 to 16) and the module
docs of `mental_models.rs`, `refresh.rs` and `system_prompt.rs`; this
document doesn't repeat them.

Decided by Tim Smart on 2026-10-04 (TIM-179). Shipped on 2026-10-05 in
TIM-181 to TIM-186.

## The question

Should the stored unit of a mental model stay the sentence, each with its
own citations, or become the whole answer with one citation set?

Entries had been the unit for everything: the refresh edited them, the
block dropped one when a cited memory was retracted, forget deleted one,
the budget trimmed them from the bottom, and call 1 was shown them under
handles so a reply relying on one credited its memories. Schema 13 already
had the refresh write the whole summary in one call and render each
section as a paragraph, but every sentence was still an entry, and the
prompt had to keep each sentence able to stand on its own because code cut
sentences one at a time. The answer read as prose-shaped, not prose.

## What was weighed

**Sentence-level, kept.** No storage change; precise retraction, trimming
and crediting for free. Its ceiling: coherence stops at the sentence
boundary. A sentence that leans on its neighbour ("She also…", a pronoun
whose antecedent is the previous sentence) reads wrong the moment the
neighbour is cut, so the prompt must forbid connective language, and the
answer stays a list in all but punctuation.

**Prose with inline citation handles, stripped at render.** Sentence-level
storage with a worse interface: code would segment prose into sentences
(abbreviations, decimals, CJK punctuation, quoted speech) and guess which
sentence a trailing handle belongs to. The LLM already returned structured
sentences. Rejected.

**A second "polish" LLM pass** rewriting cited sentences into free prose.
One more call per refresh, and it would break the exact-text identity
carry-over that kept entry ids stable between refreshes. Rejected.

**Model-level: one text, one citation set.** The LLM writes connected
prose. The costs, as first assessed:

1. A single retracted, ended or refined citation withholds the whole model
   until a refresh rewrites it, where before one sentence went. Refinement
   matters here: supersession sets `invalidated_at` on the old memory, so
   every "I moved", "new job" correction would blank the profile, and the
   minimum interval between refreshes was 30 minutes.
2. The budget becomes whole-or-nothing per model.
3. `used` crediting goes coarse: one handle for the whole profile means a
   reply that used one fact marks every cited memory used, and profile
   memories never fade. This was first called decisive.
4. Age annotations have no sentence to attach to.
5. Every refresh rewrites every word; wording drifts with nothing to
   anchor it.

## What changed the answer

Cost 3 had a fix that made the rest tractable. Call 1 already received
every in-context memory with its content as its own handle, with no cap,
built from the block's citation list. A reply that relied on a fact read
in the answer can be credited to that memory by content, as a reply that
relied on an injected memory is. The entry handles were a convenience, not
the evidence. With crediting by memory handle, precision matches
injection's whatever the stored unit is, and the entry snapshot kept for
call 1 (`turn_entries`, `prompt_blocks.entries`) could go, which also
removed copies of restated text that forget had to scrub.

Cost 1 was mitigated rather than removed: a correction or forget of a
cited memory requests an urgent refresh that bypasses the minimum interval
after a success, so the gap is the debounce plus one call. Cost 2 was
avoided: with no per-sentence citations to keep consistent, the block can
cut an answer at a sentence end and still cite the whole set. Cost 4 moved
into the write: possibly stale states are supplied with their absolute
observed date and the prose carries it, so nothing is computed at render
and the date doesn't go stale in meaning. Cost 5 was accepted.

## What was given up, knowingly

- A correction leaves the model out of new sessions for about the debounce
  plus one LLM call, where before only one sentence went. Sessions that
  already hold a block keep it either way.
- Code can check an answer's citations, not its prose. A memory that left
  the selection isn't listed, so it can't be cited, and the prompt says
  what the listed memories no longer support must go; that part rests on
  the LLM.
- A cut answer cites its whole set, so a session's in-context set can
  include a memory the agent didn't read. Injection then skips it. Minor.
- Each refresh rewrites the answer, so there is no stable sentence
  identity to diff two refreshes by, and recorded `write_model` replies
  from before the change can't stand in for new ones.
- The "as of" date for a stale state updates at the next refresh, not
  daily. A state crossing the confidence threshold changes the fingerprint
  so it does get one.

## Where it landed

| Decision | Shipped in |
|---|---|
| `used` by memory handle, entry snapshot removed | schema 14, TIM-181 |
| One answer per model, `write_model` v2, whole-or-cut rendering, forget blanks | schema 15, TIM-182 |
| Urgent refresh after a correction or forget | schema 16, TIM-183 |
| Stale-state dates in the prose, in the fingerprint | `write_model` v3, TIM-184 |
| `###` name, `Prompt:`, `Output:`, `####` facet headings | TIM-185 |
| Operator docs, upgrade notes, replay notes | TIM-186 |

Still open after TIM-186: no `write_model` v2 or v3 cassettes were
recorded for the scenarios under `scenarios/`, which run against the
scripted LLM instead. Recording against private history needs its own
authorization under `AGENTS.md`. This is a validation gap, not a design
one.
