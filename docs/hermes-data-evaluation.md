# Evaluating against Hermes data

This page is a handoff for a local agent that evaluates Asphodel against
Tim's real Hermes history, on Tim's machine, with the replay harness
(`docs/replay.md`). Hand the agent everything from "Who does what" down.
It covers the commands, how the agent drafts probes and labels for
independent Fable sub-agent review, what to measure, and a feedback template
that carries numbers and ids but never content.

Every `asphodel` command here exists in `--help` on this branch. The build,
the model fetch, the scripted replay and the report commands were run while
writing it; the real-history commands were checked for argument shape only,
since they need Tim's data.

## Who does what

Route future live comparisons to Mac Developer, running locally on Tim's
machine. You run commands, draft the evaluation data, collect numbers, and
prepare material. Follow these rules:

1. **Fable independently reviews probes and labels.** Ask Mac Developer to
   arrange review of each exact entry, including probe re-anchors and new or
   changed labels. You draft the questions, expected answers and labels,
   grounded in what the replayed store actually holds, and keep every draft
   in `$ASPHODEL_REPLAY_DIR/drafts/`. Nothing reaches `probes.toml` or
   `labels.toml` until that exact entry passes independent Fable review.
   If review is unavailable or the permitted material is insufficient,
   keep the entry pending in drafts. Those two files are the evaluation
   data; the drafts are not. Only runs on the reviewed files go into the
   feedback as accepted evidence; identify draft previews separately.
2. **Live comparisons have standing authorization while the data stays
   private.** Tim granted this on 2026-10-04 for live recordings and
   LLM-backed `fast` top-ups using `https://chatgpt.com/backend-api/codex`,
   `auth = "chatgpt"`, `model = "gpt-6-luna"` and
   `reasoning_effort = "low"`. No new per-run approval is needed within
   this boundary, including login for this backend. Before changing the
   endpoint, auth, model or reasoning setting, ask Mac Developer to arrange
   independent Fable review of the proposed change under rule 4. This does
   not accept labels, probes, probe re-anchors or floor selection. Browsing (step 5),
   `replay` mode and `report` never call a backend.
3. **Nothing from the private directory leaves it except the aggregate
   export and the feedback template below.** No sentence, query, entity
   name, alias, source text or LLM reply goes into a ticket, a PR, a chat
   reply or a file outside `ASPHODEL_REPLAY_DIR`. Memory ids and probe ids
   are fine. The output of `asphodel recall`, `recalls` and `memory show` is
   content: read it locally, never quote it anywhere else.
4. **All evaluation decisions require independent review.** Before making
   them, ask Mac Developer to obtain an independent review from a Fable
   sub-agent. This includes probes, re-anchors, labels, recall and call-2
   floor selection, tuning, spot-check verdicts, distribution assessments
   and backend changes. Mac Developer makes decisions only after sufficient
   independent review; no evaluation decision returns to Tim for approval.
   The review must follow the same privacy restrictions in rule 3: share
   only the permitted aggregate export and feedback template, never private
   content. If review is unavailable or the permitted material is
   insufficient, leave the decision pending rather than bypassing review
   or broadening access. Record the reviewer, date, exact scope, outcome
   and any pending decisions in the feedback without private content.

These rules guide agents; they are not a technical privacy guarantee. Raw
history, memory text, queries and cassettes remain private even when a run
has standing authorization.

## 0. Setup

```sh
git clone https://github.com/tim-smart/asphodel.git && cd asphodel
git checkout main
nix build                        # ./result/bin/asphodel, ORT wired by the wrapper
nix build .#models -o models     # both ONNX models, 72 MB, fixed-output fetch
export ASPHODEL_MODEL_DIR=$PWD/models
export ASPHODEL_REPLAY_DIR=$HOME/asphodel-private   # outside any git tree, outside ~/multica_workspaces
mkdir -p "$ASPHODEL_REPLAY_DIR"
export PATH="$PWD/result/bin:$PATH"   # not an alias: the timing wrapper and scripts need the real executable
asphodel --version
```

Sanity, no data involved:

```sh
nix develop -c cargo test                      # the Rust tests, on fake models, no network
nix develop -c python -m pytest plugin/tests   # the plugin tests
asphodel replay --scenario scenarios/lifetimes.toml   # a scripted run into the private dir, exit 0
```

## 1. A read-only copy of Hermes' `state.db`

No Asphodel command does this. Use SQLite's online backup from inside the
Hermes container; it only reads the live file. The cluster runs a Deployment
in namespace `hermes`, not a StatefulSet. Adjust the Deployment and
container names if needed. Select a running pod once and keep using it for
the backup, copy and cleanup.

```sh
SELECTOR=$(kubectl get deployment/hermes -n hermes -o go-template='{{range $key, $value := .spec.selector.matchLabels}}{{$key}}={{$value}},{{end}}')
SELECTOR=${SELECTOR%,}
test -n "$SELECTOR" || exit 1
HERMES_POD=$(kubectl get pods -n hermes -l "$SELECTOR" --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}')
test -n "$HERMES_POD" || exit 1
kubectl wait -n hermes --for=condition=Ready "pod/$HERMES_POD" --timeout=60s
kubectl exec -n hermes "$HERMES_POD" -c hermes -- python3 -c '
import sqlite3, os
src = sqlite3.connect(os.path.join(os.environ["HERMES_HOME"], "state.db"))
dst = sqlite3.connect("/tmp/state-copy.db")
src.backup(dst); dst.close(); src.close()'
kubectl cp -n hermes -c hermes "$HERMES_POD:/tmp/state-copy.db" "$ASPHODEL_REPLAY_DIR/state.db"
kubectl exec -n hermes "$HERMES_POD" -c hermes -- rm /tmp/state-copy.db
```

## 2. Manifest and import

Use the existing private manifest for the timezone, owner platform ids and
other speakers. If any are missing or ambiguous, leave import pending and
ask Mac Developer to arrange independent Fable review; do not guess or
request an evaluation decision from Tim. Write
`$ASPHODEL_REPLAY_DIR/manifest.toml`:

```toml
timezone = "Pacific/Auckland"
bank = "main"
assistant = "Hermes"

[owner]
name = "Tim"
platform_ids = ["discord:<id>"]

# [[speaker]]
# name = "Sam"
# id = "discord:<id>"
```

```sh
asphodel import --state-db "$ASPHODEL_REPLAY_DIR/state.db" --manifest "$ASPHODEL_REPLAY_DIR/manifest.toml" --dry-run
asphodel import --state-db "$ASPHODEL_REPLAY_DIR/state.db" --manifest "$ASPHODEL_REPLAY_DIR/manifest.toml"
```

The dry run prints counts only; they are safe to report. The corpus is
`$ASPHODEL_REPLAY_DIR/corpus/state.jsonl`. Import accepts Hermes schema
versions 30 (hermes-agent 0.21.5) and 31 (`bfc71526`) and refuses anything else by name:
if it refuses, report the message verbatim and stop.

## 3. Backend configuration

`$ASPHODEL_REPLAY_DIR/replay.toml`. The embedding floor is a placeholder
until step 8. The MiniLM injection floor is Tim's TIM-131 recommendation
from 10 labelled prefetches, pending live-run acceptance. The run refuses
to start without floors for the exact loaded models.

```toml
[llm]
auth = "chatgpt"
model = "gpt-6-luna"
endpoint = "https://chatgpt.com/backend-api/codex"
reasoning_effort = "low"

[reconcile.embedding_floors]
"bge-small-en-v1.5:int8" = 0.8

[injection.reranker_floors]
"ms-marco-MiniLM-L-6-v2:int8" = -8.0

[ranking.relevance_scales]
"ms-marco-MiniLM-L-6-v2:int8" = 3.564211
```

Check that the configuration matches the standing authorization above.
If it differs, stop until Mac Developer obtains sufficient independent
Fable review of the exact proposed backend change, including its privacy
boundary. Keep it pending if review is unavailable or insufficient. For
the authorized ChatGPT backend:

```sh
asphodel llm login --data-dir "$ASPHODEL_REPLAY_DIR"      # chatgpt mode; replay reads tokens from the private dir
```

See `docs/models.md` for backend and login details. If the first call fails,
report the failure without private content. Do not switch to `api_key` or
another endpoint or model before that change passes independent Fable
review arranged by Mac Developer. Review does not relax the privacy rules.

## 4. The recording run

On Linux, including NixOS, resolve GNU time from the nixpkgs input pinned
by this checkout's `flake.lock`. Run this from the repository root, before
measuring, so downloading or building time is not included in the result:

```sh
GNU_TIME="$(nix build --no-link --print-out-paths --inputs-from . nixpkgs#time)/bin/time"
```

The command below uses `"$GNU_TIME" -v`. On macOS, replace that prefix
with `/usr/bin/time -l`; it reports maximum resident set size in bytes
rather than GNU time's KiB. Both send timing output to `live.log`. Use the
same platform-specific prefix for later measurements. To check timing
without private data or live calls, run the scripted scenario first:

```sh
"$GNU_TIME" -v asphodel replay --scenario scenarios/lifetimes.toml
```

GNU time prints `Elapsed (wall clock) time` and `Maximum resident set size
(kbytes)` to stderr. The macOS form prints `real` and `maximum resident
set size`. Only start the live run below with a configuration covered by
the standing authorization or a backend change that has passed independent
Fable review arranged by Mac Developer.

```sh
"$GNU_TIME" -v asphodel replay --corpus "$ASPHODEL_REPLAY_DIR/corpus/state.jsonl" --mode live \
    --config "$ASPHODEL_REPLAY_DIR/replay.toml" \
    --labelling "$ASPHODEL_REPLAY_DIR/labelling.json" \
    --aggregate "$ASPHODEL_REPLAY_DIR/aggregate-live.json" \
    --until 2027-01-01T00:00:00Z 2> "$ASPHODEL_REPLAY_DIR/live.log"
```

One LLM round trip per chunk: hours, or days on a subscription. Outputs:
`reports/state-live.json`, `cassettes/state.jsonl`, the labelling material,
the aggregate. Then, with no LLM:

```sh
asphodel replay --corpus "$ASPHODEL_REPLAY_DIR/corpus/state.jsonl" --mode replay \
    --config "$ASPHODEL_REPLAY_DIR/replay.toml" --self-test
asphodel report html "$ASPHODEL_REPLAY_DIR/reports/state-live.json"
```

## 5. Browse the replayed memories

The report's `memories` list has ids and times but no sentences. To see
what a memory says, serve a copy of the replayed store with no LLM. The
copy keeps the bank's recorded models, so `ASPHODEL_MODEL_DIR` must be set.
Write `$ASPHODEL_REPLAY_DIR/browse.toml` separately from `replay.toml`:

```toml
# No [llm]: neither extraction nor background refresh has a backend.
[purge]
delta = "never"

[reconcile.embedding_floors]
"bge-small-en-v1.5:int8" = 0.8

[injection.reranker_floors]
"jina-reranker-v1-turbo-en:int8" = 0.0
```

Use the same floors as your replay config if you changed them. The replay
config above uses the default numeric purge delta; `"never"` changes its
deletion fingerprint. On the fresh copy this pauses the entire nightly
sweep, including source, failed-chunk and recall-log deletion, not just
memory purge. `delta = "never"` alone does **not** disable the source sweep;
the unacknowledged fingerprint change does. Nothing here calls a backend:
recall runs the local embedder and reranker only, and refresh returns 503.

```sh
test ! -e "$ASPHODEL_REPLAY_DIR/browse" || { echo "Remove the old browse copy first"; exit 1; }
cp -r "$ASPHODEL_REPLAY_DIR/store" "$ASPHODEL_REPLAY_DIR/browse"
asphodel serve --data-dir "$ASPHODEL_REPLAY_DIR/browse" --config "$ASPHODEL_REPLAY_DIR/browse.toml" \
    --listen 127.0.0.1:7741 2> "$ASPHODEL_REPLAY_DIR/browse.log" &
ASPHODEL_BROWSE_PID=$!
export ASPHODEL_URL=http://127.0.0.1:7741
asphodel purge plan --json
asphodel recall --bank main "what is my sister called"
asphodel memory show --bank main <id>
```

Wait for the daemon to be ready, then check that `purge plan --json` shows
`pause.state` as `"paused"` before recalling anything. If it shows `"running"`,
stop and check that this is a fresh copy of the replay store with a numeric
purge delta, not a previously acknowledged browse store. Never run
`purge ack` on the browse daemon, and do not issue `forget` or `erase`: those
explicit deletion commands do not respect the purge pause.

The daemon warns that the deletion fingerprint changed. Both purge and the
source/recall sweep stay paused while you browse. Recall writes the copy's
recall log only. Stop it with `kill "$ASPHODEL_BROWSE_PID"` when done, and
delete `browse/` after
the evaluation: it is a second copy of the history. Never point this at a
`serve` data dir Hermes uses.

`memory show` is how you ground a probe: it gives the sentence, kind,
window, phase, the source passage, the supersession chain, the access log,
and the projected fade date.

## 6. Probes: drafted by you, independently reviewed by Fable

A probe pins a time and an expectation (`docs/replay.md`, "Probes"). The
kinds are `band`, `faded_at`, `exists`, `absent`, `agenda_has`,
`agenda_lacks`, `recall_finds`, `recall_lacks`, `injects`, `not_injects`,
`profile_has`, `profile_lacks`. In real history `memory` is a regex over
sentences. `memory_id` is an optional grounding UUID: when that id is
in the store it takes precedence, otherwise the earliest created memory
matching the regex is probed (including one since purged). A memory that
never resolves fails every kind, including negative probes and `absent`.

**Propose.** Using the browse daemon and `labelling.json`, draft 20 to 40
probes. Cover each of these, so the run says something about every part of
the model:

- *Things Tim told Hermes that it should still know.* A question in Tim's
  words, the memory that answers it (ask the browse daemon the question,
  pick the memory, read it with `memory show`), as `recall_finds` and, for
  the strongest cases, `injects`.
- *Things that changed.* A corrected name, a rescheduled date, a finished
  task: `exists` with `head = true` on the new memory and `retracted =
  true` or `ended = true` on the old one, and `recall_lacks` for the old
  version where the correction should hide it.
- *Things that should have faded.* Trivial one-off mentions: `band` with
  `faded` at a date well after the mention, and `faded_at` with a range
  around the projected fade date `memory show` reports.
- *Things that should not have faded.* Facts mentioned on several occasions:
  `band` with `strong` at the end of the run.
- *Upcoming and overdue.* Appointments and occasion-bound tasks:
  `agenda_has` shortly before the date, `agenda_lacks` well after it.
  Obligations (payments, renewals): `agenda_has` before the date and as
  overdue after it, until done or `agenda.overdue_days` passes.
- *The profile.* `profile_has` with `model = "User profile"` for two or
  three facts about Tim that a profile should carry.

Write them to `$ASPHODEL_REPLAY_DIR/drafts/probes.draft.toml`. Every probe
gets a comment block above it with the question, the expected answer in
plain words, the memory id it was grounded on, and its status. Every
drafted probe also carries that grounding id as `memory_id`, not only
in its comment block. Anchor the regex on distinctive words of the
sentence, not on the whole sentence, and
check with the browse daemon that it matches the memory you mean and no
earlier one.

```toml
# question: what is my sister called
# expected: the corrected name, not the one first given
# grounded on: b4ccd45d-80dd-53dd-9b22-b1c8f9f43bc5
# status: proposed
[[probe]]
id = "p001"
at = "2026-11-01T00:00:00Z"
kind = "recall_finds"
memory_id = "b4ccd45d-80dd-53dd-9b22-b1c8f9f43bc5"
memory = "(?i)sister.*Mia"
query = "what is my sister called"

# question: does the old name stay hidden
# expected: the retracted memory is still in the store but not the head
# grounded on: 41c78166-793c-50c2-bca7-c1d7227c222e
# status: proposed
[[probe]]
id = "p002"
at = "2026-11-01T00:00:00Z"
kind = "exists"
memory_id = "41c78166-793c-50c2-bca7-c1d7227c222e"
memory = "(?i)sister.*Maya"
retracted = true
head = false
```

**Validate.** A probe never changes a run, so the draft can be run as it
is. `replay` mode needs the cassette from step 4 and no backend.

```sh
asphodel replay --corpus "$ASPHODEL_REPLAY_DIR/corpus/state.jsonl" --mode replay \
    --config "$ASPHODEL_REPLAY_DIR/replay.toml" --probes "$ASPHODEL_REPLAY_DIR/drafts/probes.draft.toml" \
    --report "$ASPHODEL_REPLAY_DIR/reports/probes-draft.json"
```

Exit 2 means the file was refused: a duplicate id, a regex that doesn't
parse, a `faded_at` range that ends after its `at`. Fix and re-run. Exit 0
or 1 gives a report whose `probes` list has `observed` for every probe:
the band and strength, the ids recall returned, the agenda's ids, the
fade instant. Read it before the review, so you can compare what the
system actually did with what you expected in the permitted IDs-and-numbers
summary. Resolved probes report
`resolved_by` (`id` or `regex`) and `regex_matches`. Review any
`regex_matches = false`: re-recording may have shifted claim ordinals
so the grounding id now names a different fact. This flag does not fail
the check by itself. `resolved: false` fails the probe and contributes
to exit 1; it is not evidence that a negative expectation was met.

**Review.** Ask Mac Developer to arrange independent Fable sub-agent review
of the draft in batches of about ten, including new probes and re-anchors.
Keep questions, expected-answer text, grounding sentences and full draft
observations in the private directory. Share only the permitted aggregate
export and feedback template with the reviewer, using probe and memory IDs
and numbers, never private content. If review is unavailable or the
permitted material is insufficient to decide, leave the probe pending.

A probe that passes review moves into `$ASPHODEL_REPLAY_DIR/probes.toml`
with its comment block and `# status: reviewed by Fable <date>`. Record the
review outcome in the private drafts. An edited or re-anchored probe needs
independent review again before it moves. A rejected probe stays in the
draft marked `rejected`, so it isn't proposed again. Never write unreviewed
entries into `probes.toml`.

A failing probe is not a reason to change its expectation. If independent
review finds the expectation right and the observation wrong, that is a
finding: retain the probe as written and report its failed id.

**Run the reviewed set.** This is the run the feedback reports.

```sh
asphodel replay --corpus "$ASPHODEL_REPLAY_DIR/corpus/state.jsonl" --mode replay \
    --config "$ASPHODEL_REPLAY_DIR/replay.toml" --probes "$ASPHODEL_REPLAY_DIR/probes.toml" \
    --aggregate "$ASPHODEL_REPLAY_DIR/aggregate-probes.json"
```

Exit 0 all passed, 1 some failed (the report says what each observed), 2
refused.

## 7. Bench

```sh
asphodel bench --config "$ASPHODEL_REPLAY_DIR/replay.toml" --corpus "$ASPHODEL_REPLAY_DIR/corpus/state.jsonl" \
    --concurrency 1 --concurrency 4 --concurrency 16 --requests 64 \
    --report "$ASPHODEL_REPLAY_DIR/reports/bench.json"
```

It copies the replayed store under `bench/` and starts a daemon on loopback
with the production 1.5 s reranker deadline on. It never touches another
store. Stop the browse daemon first if it is on the same port.

## 8. Labels: drafted by you, independently reviewed by Fable

`labelling.json` (step 4) holds 50 sampled prefetches, each with its query
and the reranked candidates before the gate, and every candidate list call
2 was shown, each with its claim, the claim's chunk and ordinal, and its
neighbours. Every candidate has an id (`r1.1`, `c1.1`), the memory's id,
its score and its sentence.

Labels are keyed by what they judge, not by candidate id, so they survive a
re-record. A recall label is the sample's `query` and the candidate's
`memory`. A call-2 label is the sample's `chunk` and `ordinal` and the
candidate's `memory`. Use the ids to track review scope, but never
write them into a label: they number this run's candidates and mean
nothing in the next one.

**The two questions.** For a recall candidate: given this query, would a
good assistant want this memory in front of it for the reply? Related is
not enough; it has to help answer. For a call-2 candidate: is it about the
same thing as the claim, so that the claim restates, confirms, refines,
ends or contradicts it? Same person and topic is not enough; it has to be
the same fact.

**Draft and review.** Go sample by sample locally. Inspect the query (or
claim), each candidate's id, score and sentence, and draft a suggested label
with a one-line reason. Keep this content inside the private directory.
Ten recall samples and ten call-2 samples, about two hundred labels, is
enough to start; more samples sharpen the curve around the floor. Do every
candidate in a sample you start, including low-scored ones.

Write drafts to `$ASPHODEL_REPLAY_DIR/drafts/labels.draft.toml`, with a
comment per sample recording its review status. Ask Mac Developer to
arrange independent Fable review of the exact labels. Share only the
permitted aggregate export and feedback template, using IDs and numbers,
never queries, claims, candidate sentences or private draft files. If that
material cannot support a judgment, or review is unavailable, leave labels
pending in drafts. Do not infer acceptance from silence or from a model's
own drafts. Move only labels that pass independent review into
`$ASPHODEL_REPLAY_DIR/labels.toml`, recording reviewer, date, scope and
outcome locally. Edited labels require review again.

```toml
# sample r1, independently reviewed by Fable <reviewer> <date>
[[recall]]
query = "what time is the dentist"
memory = "b4ccd45d-80dd-53dd-9b22-b1c8f9f43bc5"
relevant = true

[[recall]]
query = "what time is the dentist"
memory = "41c78166-793c-50c2-bca7-c1d7227c222e"
relevant = false

# sample c1, independently reviewed by Fable <reviewer> <date>
[[call2]]
chunk = "0d6f3a52-1c4e-5b7a-9e2f-6a8b3c1d4e5f"
ordinal = 0
memory = "41c78166-793c-50c2-bca7-c1d7227c222e"
relevant = true
```

Copy `query`, `chunk` and `ordinal` from the sample exactly as the
material has them; a label whose key differs by a character matches
nothing.

**The curve.** Run it on the reviewed file. Running it on the draft is fine
for a preview, but say so.

```sh
asphodel report precision --labels "$ASPHODEL_REPLAY_DIR/labels.toml" --material "$ASPHODEL_REPLAY_DIR/labelling.json"
```

It prints, for recall and for call 2, how many candidates were labelled
(`labelled`) and how many weren't (`unlabelled`), how many labels matched a
candidate (`matched`) and how many found nothing (`unmatched`), and one
point per distinct score: at that `floor`, how many labelled candidates
were kept, how many were relevant, and the precision. Numbers only; it may
be reported whole. Propose the two floors with precision targets and sample
coverage: for recall, the lowest qualifying logit; for call 2, the lowest
qualifying cosine. Mac Developer must arrange independent Fable review of
the targets and exact floor choices before adopting them in `replay.toml`
or production `asphodel.toml`. If review is unavailable or insufficient,
leave selection pending and retain the current floors. The recall curve
matches the gate exactly. The call-2 curve covers only candidates the
placeholder reconcile floor let call 2 see, so after reviewed floor changes,
re-run step 4 within the authorized backend boundary and re-label if the
material changed. New or changed labels need independent Fable review too.

**After a re-record.** Pass the accepted labels when you re-run step 4, so
the material samples prefetches whose queries already have accepted labels:

```sh
    --labelling "$ASPHODEL_REPLAY_DIR/labelling.json" \
    --labels "$ASPHODEL_REPLAY_DIR/labels.toml" \
```

Then run the curve on the new material with the same labels. `matched`
says how many carried over and `unlabelled` how many candidates are new.
Expect partial carry-over, not all: a memory's id comes from its source
and its claim's position in call 1's reply, so when a new prompt splits a
turn differently the ids shift and the labels on them stop matching. Label
the unlabelled candidates through the draft and independent-review workflow
above; keep unresolved labels pending.

**Labels in the old form.** A `labels.toml` of candidate ids (`"r1.1" =
true`) is still read against the material it was written for. Convert it
once, against that material, before using it with any other:

```sh
asphodel report precision --labels "$ASPHODEL_REPLAY_DIR/labels-old.toml" \
    --material "$ASPHODEL_REPLAY_DIR/labelling-old.json" \
    --convert "$ASPHODEL_REPLAY_DIR/labels.toml"
```

The output gains `converted`: the `recall` and `call2` labels written,
`dropped_call2`, and `conflicting`. Recall labels always convert. Call-2
labels convert only if that material records each claim's chunk and
ordinal, which material written before this change doesn't, so on old
material every call-2 label is dropped and counted in `dropped_call2`;
those samples need labelling again. `conflicting` counts labels left out
because the same query and memory were labelled both ways in two samples;
ask Mac Developer to arrange independent Fable review of the conflict. Keep
conflicting labels pending until resolved. Report both counts.

## 9. A/B runs

`fast` reuses recorded claims and verdicts and calls the LLM only for pairs
nobody has judged. These top-ups are covered by the standing authorization
only within the same private-data and backend boundary. Overrides take the
shape of `Tuning`, for example `clock.quiet_rate = 0.2` or `injection.cap = 6`.

```sh
asphodel replay --corpus "$ASPHODEL_REPLAY_DIR/corpus/state.jsonl" --mode fast \
    --config "$ASPHODEL_REPLAY_DIR/replay.toml" --overrides "$ASPHODEL_REPLAY_DIR/try-1.toml" \
    --probes "$ASPHODEL_REPLAY_DIR/probes.toml" --report "$ASPHODEL_REPLAY_DIR/reports/try-1.json"
asphodel report diff "$ASPHODEL_REPLAY_DIR/reports/state-live.json" "$ASPHODEL_REPLAY_DIR/reports/try-1.json"
```

The diff refuses a different corpus or cassette without `--force`; it lists
probes that flipped, numbers that moved, and memory ids that faded or purged
in one run only.

## What to measure

Each line says where the number comes from. **Shipped** means the tool
emits it. **Manual** means you run a general tool or read something and
judge it. **Not implemented** means there is no tooling and you should say
so rather than improvise.

### Retrieval quality

- Shipped: probe pass/fail by id (`probes_passed`, `probes_failed` in the
  aggregate).
- Shipped: `injected_tokens` per session and per turn p50/p95, with cron
  apart; `call2_rate`; `llm.used_verdicts` by source (recorded, top-up,
  live, none); `injection_usage` (used, not used, unjudged and the used
  fraction among judged memories); `agenda_lines_per_day`; `bands_per_week`;
  `fade_outs_per_week`; `purged_then_re_mentioned.rate`.
- Shipped, for local inspection: on the browse daemon (step 5), `asphodel
  recalls --bank main --limit 50` after the probe run, then `asphodel memory
  show` on returned ids. Draft relevance verdicts locally; Mac Developer
  arranges independent Fable review using only permitted IDs and numbers.
  Count accepted verdicts and report pending ones if review cannot judge
  relevance without private content.
- Manual: sanity bounds. Per-turn injected p95 should stay well under the
  prompt's budget; a `used` rate near zero across a run means injection
  isn't being relied on; a `call2_rate` near 1.0 means almost every claim is
  flagged.
- Not implemented: reranker versus rank fusion. `Tuning` has no key to turn
  the reranker off, so that comparison cannot be run.

### Calibration

- Shipped: the precision curve for recall and call 2 (`report precision`);
  `significance_histogram` and `kind_histogram` in the report.
- Manual: propose an assessment of the significance and kind histograms.
  Mac Developer arranges independent Fable review before adopting it or
  changing significance values. Report histograms, review outcome and any
  pending assessment; do not share private history.
- Not implemented: any automatic floor selection or optimiser; the harness
  is deliberately A/B only.

### Latency and throughput

- Shipped: bench `p50_ms`, `p95_ms`, `p99_ms`, `max_ms` per concurrency
  level and the fraction the reranker answered within its deadline; replay
  `extraction_lag` p50/p95 in simulated time; `llm.latency_ms` percentiles
  from the live run.
- Manual: wall time of each replay run (step 4's GNU time prefix, "Elapsed",
  or macOS `/usr/bin/time -l`, "real"), and
  cassette size on disk.
- Manual, if a real daemon is reachable: `curl -s -o /dev/null -w
  '%{time_total}\n'` against `/v1/health`, which should be milliseconds. Do
  not send prefetches to the production daemon; prefetch writes the recall
  log.

### Resource use

- Not implemented in Asphodel. Measure manually with step 4's GNU time
  prefix or macOS `/usr/bin/time -l`: "Maximum resident set size" for the
  replay and for `asphodel serve` on the bench copy with both models loaded;
  `du -sh "$ASPHODEL_REPLAY_DIR/store"
  "$ASPHODEL_REPLAY_DIR/cassettes"`; `kubectl top pod -n hermes "$HERMES_POD" --containers`
  if the real sidecar is running. The example deployment requests 512 MiB
  and limits 1 GiB; report whether the daemon's RSS fits with the models
  loaded.

### Backend failures

- Shipped: `asphodel status` (exit non-zero when anything needs attention,
  and `--json` for the fields: queued, failed chunks, failed refreshes,
  purge pause, last sweep, last backup); `asphodel chunks --bank main
  --failed` lists the error kind and HTTP status of each failure, never the
  response; `llm.misses` in a `replay` or `fast` report.
- Shipped: the daemon's error kinds are `transport`, `timeout`, `status`
  (with the HTTP status), `no_content`, `not_json`, `refused`,
  `login_required`, `usage_limited` (with `resets_at`) and `backend` (with a
  code). Count failures by kind from `live.log`, which holds no content.
- Manual, plugin resilience, only on a non-production Hermes: stop the
  daemon, send a few turns, confirm one JSON file per turn under
  `$HERMES_HOME/asphodel/spool/`, confirm the breaker opens after 3
  connection failures and skips the network for 30 s, restart the daemon,
  confirm the spool drains. The spool caps at about 10 MB or 7 days.
- Manual, daemon resilience, no real LLM: `ASPHODEL_LLM_SCRIPT=<file>
  asphodel serve ...` plays scripted replies and failures
  (`docs/models.md`, "The LLM"). Confirm a `usage_limited` step holds the
  queue until `resets_at` and a `status` 500 step counts as a failed chunk
  after the retry cap.
- Not implemented: a metrics endpoint. `status` exiting non-zero is the
  only alert signal.

## Feedback template

Preserve historical provenance for carried-over evidence. Do not relabel
past Tim-approved entries as Fable-reviewed or imply a review occurred
when it did not. Record carried-over counts and their original authority
separately from new Fable-reviewed entries and pending drafts. All new
decisions follow the independent-review policy above.

Copy this, fill it, and attach `aggregate-*.json`, the one artifact the
standing rule lets out of the private dir: its type has no string field
but probe ids and enum names. `bench.json` stays in the private dir. It
holds only hashes and numbers, but it is not the typed export, so copy its
numbers into the template instead of attaching it. Nothing else from the
private dir is attached. Before sending, grep your draft for any sentence
from a report, any query, any name: if one is there, remove it.

```markdown
## Asphodel evaluation feedback

- branch / commit: asphodel-v1 @ <git sha from the report>
- corpus_hash: <first 12 hex> · cassette_hash: <first 12 hex> · hermes schema: <30|31>
- import counts (dry run): <sessions / turns / cron / skipped>
- LLM: endpoint=<endpoint> auth=<api_key|chatgpt> model=<model> reasoning=<effort> · authorization=<standing 2026-10-04 | backend change independently reviewed by Fable: reviewer/date/scope/outcome>
- modes run: live <y/n>, replay --self-test <passed/failed>, fast <n runs>
- overrides tried: <key = value, ...> (or none)
- probes: <n> independently reviewed by Fable (<n> drafted, <n> rejected, <n> pending) · labels: <n> independently reviewed by Fable over <n> samples (<n> pending)
- independent review arranged by Mac Developer: <reviewer/date/exact scope/outcome for probes, labels, floors, tuning, backend changes and other decisions; IDs and numbers only>
- pending decisions: <IDs/counts or none; unavailable or insufficient review>
- carried-over evidence: <probe/label counts and original authority/date; not retroactively Fable-reviewed>

### Retrieval
- probes: <passed>/<total>; failed ids: <p003, p007>
- injected tokens per turn p50/p95: <n>/<n>; cron p95: <n>
- used verdicts: recorded <n>, top-up <n>, live <n>, none <n>
- injection usage: used <n>, not used <n>, unjudged <n>; used fraction <x.xx> (used / judged)
- call2_rate: <x.xx> · agenda lines/day median: <n>
- purged then re-mentioned: <purged>/<re_mentioned> (<rate>)
- independently reviewed spot check of <n> recalls: <relevant>/<n> judged relevant; pending <n>

### Calibration
- recall floor: <reviewed choice or pending/current logit> at precision <x.xx> over <labelled> labels
- call-2 floor: <reviewed choice or pending/current cosine> at precision <x.xx> over <labelled> labels
- significance histogram: trivial <n> minor <n> notable <n> major <n> critical <n>
- kind histogram: fact <n> event <n> state <n> task <n> recurring <n>
- independently reviewed distribution assessment: <outcome or pending; no content>

### Latency and throughput
- bench @1/4/16: p50 <n>/<n>/<n> ms, p95 <n>/<n>/<n> ms, within deadline <x.xx>/<x.xx>/<x.xx>
- extraction lag (simulated) p50/p95: <n>/<n> ms · live LLM latency p50/p95: <n>/<n> ms
- replay wall time: live <h:mm>, replay <m:ss>, fast <m:ss>

### Resources
- replay max RSS: <n> MiB · serve max RSS with models: <n> MiB
- store size: <n> MiB · cassette size: <n> MiB · pod memory (kubectl top): <n> MiB

### Backend failures
- failures by kind during live: transport <n> timeout <n> status-<code> <n> usage_limited <n> other <n>
- failed chunks after retries: <n> · status exit code at end: <0|1>
- plugin spool check: <not run | drained n files after restart> · breaker: <opened after 3 / did not open>

### Problems
- <one line each: what, which number, which command; no content>
```

## Known limits, so you don't chase them

- No `state.db` export command; step 1 is general SQLite tooling.
- The report lists memory ids without sentences. Sentences come from the
  browse daemon (step 5) or the labelling material, both private.
- Floors must be set before the run that calibrates them; one re-run is
  expected.
- Reranker versus rank fusion has no tooling.
- Import is pinned to Hermes schemas 30 and 31.
- ChatGPT subscription mode is unverified against the real backend and
  throttles on usage windows.
- A daemon serving a bank whose embedding model it doesn't carry refuses
  that bank; only the current models are in the image.
