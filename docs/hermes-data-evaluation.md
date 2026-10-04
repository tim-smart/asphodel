# Evaluating against Hermes data

This page is a handoff for a local agent that evaluates Asphodel against
Tim's real Hermes history, on Tim's machine, with the replay harness
(`docs/replay.md`). Hand the agent everything from "Who does what" down.
It covers the commands, how the agent drafts probes and labels for Tim to
approve, what to measure, and a feedback template that carries numbers and
ids but never content.

Every `asphodel` command here exists in `--help` on this branch. The build,
the model fetch, the scripted replay and the report commands were run while
writing it; the real-history commands were checked for argument shape only,
since they need Tim's data.

## Who does what

You are a local agent running on Tim's machine. You run commands, draft
the evaluation data, collect numbers, and prepare material. Three things
are not yours:

1. **Tim approves the probes and the labels.** You draft them: you propose
   the questions, the expected answers and the labels, grounded in what the
   replayed store actually holds, and you keep every draft in
   `$ASPHODEL_REPLAY_DIR/drafts/`. Nothing reaches `probes.toml` or
   `labels.toml` until Tim has approved that exact entry in this session.
   Those two files are the evaluation data; the drafts are not. Only runs on
   the approved files go into the feedback.
2. **Nothing goes to an LLM backend without Tim's explicit approval in this
   session.** Before any `--mode live`, any `--mode fast` with an LLM
   configured, or `asphodel llm login`, stop, state the endpoint and model
   that will see the history, and wait for a yes. Browsing (step 5),
   `replay` mode and `report` never call a backend.
3. **Nothing from the private directory leaves it except the aggregate
   export and the feedback template below.** No sentence, query, entity
   name, alias, source text or LLM reply goes into a ticket, a PR, a chat
   reply or a file outside `ASPHODEL_REPLAY_DIR`. Memory ids and probe ids
   are fine. The output of `asphodel recall`, `recalls` and `memory show` is
   content: read it, show it to Tim, never quote it anywhere else.

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

Ask Tim for the timezone, owner platform ids and other speakers. Write
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
model = "<the exact model Tim will run in production>"
endpoint = "https://api.openai.com/v1"    # or: auth = "chatgpt" and no endpoint

[reconcile.embedding_floors]
"bge-small-en-v1.5:int8" = 0.8

[injection.reranker_floors]
"ms-marco-MiniLM-L-6-v2:int8" = -8.0

[ranking.relevance_scales]
"ms-marco-MiniLM-L-6-v2:int8" = 3.564211
```

**Stop here and get approval.** Then one of:

```sh
export ASPHODEL_LLM_API_KEY=...                           # api_key mode
asphodel llm login --data-dir "$ASPHODEL_REPLAY_DIR"      # chatgpt mode; replay reads tokens from the private dir
```

ChatGPT mode is unverified against the real backend (`docs/models.md`). If
the first call fails there, switch to `api_key` before concluding anything
else.

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
set size`. Only start the live run below after the endpoint/model approval.

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

## 6. Probes: drafted by you, approved by Tim

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
- *Upcoming and overdue.* Appointments and tasks: `agenda_has` shortly
  before the date, `agenda_lacks` well after it.
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
fade instant. Read it before the review, so you can tell Tim what the
system actually did next to what you expected. Resolved probes report
`resolved_by` (`id` or `regex`) and `regex_matches`. Review any
`regex_matches = false`: re-recording may have shifted claim ordinals
so the grounding id now names a different fact. This flag does not fail
the check by itself. `resolved: false` fails the probe and contributes
to exit 1; it is not evidence that a negative expectation was met.

**Review.** Walk Tim through the draft in batches of about ten. For each
probe show the question, your expected answer, the grounding memory's
sentence, and what the draft run observed. Tim approves, edits or rejects.
An approved probe moves into `$ASPHODEL_REPLAY_DIR/probes.toml` with its
comment block and `# status: approved by Tim <date>`. An edited probe is
re-shown before it moves. A rejected probe stays in the draft marked
`rejected`, so it isn't proposed again. Never write anything else into
`probes.toml`.

A failing probe is not a reason to change its expectation. If Tim says the
expectation is right and the observation is wrong, that is a finding:
approve the probe as written and it goes into the feedback as a failed id.

**Run the approved set.** This is the run the feedback reports.

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

## 8. Labels: guided by you, decided by Tim

`labelling.json` (step 4) holds 50 sampled prefetches, each with its query
and the reranked candidates before the gate, and every candidate list call
2 was shown, each with its claim and neighbours. Every candidate has an id
(`r1.1`, `c1.1`), the memory's id, its score and its sentence.

**The two questions.** For a recall candidate: given this query, would a
good assistant want this memory in front of it for the reply? Related is
not enough; it has to help answer. For a call-2 candidate: is it about the
same thing as the claim, so that the claim restates, confirms, refines,
ends or contradicts it? Same person and topic is not enough; it has to be
the same fact.

**Guide.** Go sample by sample. Show Tim the query (or the claim), then
each candidate with its id, score and sentence, your suggested label and a
one-line reason. He confirms or flips each. Suggest, don't decide: the
curve is only as good as his labels. Ten recall samples and ten call-2
samples, about two hundred labels, is enough to start; more samples
sharpen the curve around the floor. Do every candidate in a sample you
start, including the low-scored ones, or the curve's low end is missing.

Write confirmed labels to `$ASPHODEL_REPLAY_DIR/drafts/labels.draft.toml`
as you go, one comment per sample saying which one it was and whether Tim
has finished it. When a sample is finished, move its labels to
`$ASPHODEL_REPLAY_DIR/labels.toml`. The approved file holds only labels Tim
confirmed; a label he hasn't looked at never goes there.

```toml
# sample r1, finished by Tim 2026-10-03
"r1.1" = true
"r1.2" = false
"r1.3" = false
```

**The curve.** Run it on the approved file. Running it on the draft is fine
for a preview, but say so.

```sh
asphodel report precision --labels "$ASPHODEL_REPLAY_DIR/labels.toml" --material "$ASPHODEL_REPLAY_DIR/labelling.json"
```

It prints, for recall and for call 2, how many candidates were labelled and
how many weren't, and one point per distinct score: at that `floor`, how
many labelled candidates were kept, how many were relevant, and the
precision. Numbers only; it may be reported whole. Tim picks the two
floors: for recall, the lowest logit at which precision is still what he
wants; for call 2, the lowest cosine. Put them in `replay.toml` and in the
production `asphodel.toml`. The recall curve matches the gate exactly. The
call-2 curve covers only candidates the placeholder reconcile floor let
call 2 see, so after the floors change, re-run step 4 (approval again) and
re-label if the material changed.

## 9. A/B runs

`fast` reuses recorded claims and verdicts and calls the LLM only for pairs
nobody has judged, so it also needs approval. Overrides take the shape of
`Tuning`, for example `clock.quiet_rate = 0.2` or `injection.cap = 6`.

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
  live, none); `agenda_lines_per_day`; `bands_per_week`;
  `fade_outs_per_week`; `purged_then_re_mentioned.rate`.
- Shipped, for Tim's eyes: on the browse daemon (step 5), `asphodel
  recalls --bank main --limit 50` after the probe run, then `asphodel memory
  show` on returned ids. Tim judges relevance; you count his verdicts.
- Manual: sanity bounds. Per-turn injected p95 should stay well under the
  prompt's budget; a `used` rate near zero across a run means injection
  isn't being relied on; a `call2_rate` near 1.0 means almost every claim is
  flagged.
- Not implemented: reranker versus rank fusion. `Tuning` has no key to turn
  the reranker off, so that comparison cannot be run.

### Calibration

- Shipped: the precision curve for recall and call 2 (`report precision`);
  `significance_histogram` and `kind_histogram` in the report.
- Manual: Tim compares the significance and kind histograms with what he
  expects of his history. Report the histograms and his verdict.
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
- LLM: auth=<api_key|chatgpt> model=<model> · approved by Tim on <date>
- modes run: live <y/n>, replay --self-test <passed/failed>, fast <n runs>
- overrides tried: <key = value, ...> (or none)
- probes: <n> approved by Tim (<n> drafted, <n> rejected) · labels: <n> approved by Tim over <n> samples

### Retrieval
- probes: <passed>/<total>; failed ids: <p003, p007>
- injected tokens per turn p50/p95: <n>/<n>; cron p95: <n>
- used verdicts: recorded <n>, top-up <n>, live <n>, none <n>
- call2_rate: <x.xx> · agenda lines/day median: <n>
- purged then re-mentioned: <purged>/<re_mentioned> (<rate>)
- Tim's spot check of <n> recalls: <relevant>/<n> judged relevant

### Calibration
- recall floor chosen: <logit> at precision <x.xx> over <labelled> labels
- call-2 floor chosen: <cosine> at precision <x.xx> over <labelled> labels
- significance histogram: trivial <n> minor <n> notable <n> major <n> critical <n>
- kind histogram: fact <n> event <n> state <n> task <n> recurring <n>
- Tim's view of the distributions: <one line>

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
