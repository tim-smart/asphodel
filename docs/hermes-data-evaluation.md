# Evaluating against Hermes data

This page is a handoff for a local agent that evaluates Asphodel against
Tim's real Hermes history, on Tim's machine, with the replay harness
(`docs/replay.md`). Hand the agent everything from "Who does what" down.
It covers the commands, what to measure, and a feedback template that
carries numbers and ids but never content.

Every `asphodel` command here exists in `--help` on this branch. The build,
the model fetch, the scripted replay and the report commands were run while
writing it; the real-history commands were checked for argument shape only,
since they need Tim's data.

## Who does what

You are a local agent running on Tim's machine. You run commands, collect
numbers, and prepare material. Three things are not yours:

1. **Tim writes the real-history probes and the labels.** You may draft
   nothing in `probes.toml` or `labels.toml`. You may prepare the files'
   skeletons and tell Tim what the report shows, but every `[[probe]]` and
   every label line is his.
2. **Nothing goes to an LLM backend without Tim's explicit approval in this
   session.** Before any `--mode live`, any `--mode fast` with an LLM
   configured, or `asphodel llm login`, stop, state the endpoint and model
   that will see the history, and wait for a yes.
3. **Nothing from the private directory leaves it except the aggregate
   export and the feedback template below.** No sentence, query, entity
   name, alias, source text or LLM reply goes into a ticket, a PR, a chat
   reply or a file outside `ASPHODEL_REPLAY_DIR`. Memory ids and probe ids
   are fine. The output of `asphodel recalls` and `memory show` is content:
   read it, never quote it.

## 0. Setup

```sh
git clone https://github.com/tim-smart/asphodel.git && cd asphodel
git checkout asphodel-v1
nix build                        # ./result/bin/asphodel, ORT wired by the wrapper
nix build .#models -o models     # both ONNX models, 72 MB, fixed-output fetch
export ASPHODEL_MODEL_DIR=$PWD/models
export ASPHODEL_REPLAY_DIR=$HOME/asphodel-private   # outside any git tree, outside ~/multica_workspaces
mkdir -p "$ASPHODEL_REPLAY_DIR"
export PATH="$PWD/result/bin:$PATH"   # not an alias: `/usr/bin/time -v` and scripts need the real executable
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
Hermes container; it only reads the live file. Adjust pod and container
names.

```sh
kubectl exec hermes-0 -c hermes -- python3 -c '
import sqlite3, os
src = sqlite3.connect(os.path.join(os.environ["HERMES_HOME"], "state.db"))
dst = sqlite3.connect("/tmp/state-copy.db")
src.backup(dst); dst.close(); src.close()'
kubectl cp -c hermes hermes-0:/tmp/state-copy.db "$ASPHODEL_REPLAY_DIR/state.db"
kubectl exec hermes-0 -c hermes -- rm /tmp/state-copy.db
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
version 31 only (hermes-agent `bfc71526`) and refuses anything else by name:
if it refuses, report the message verbatim and stop.

## 3. Backend configuration

`$ASPHODEL_REPLAY_DIR/replay.toml`. Floors are placeholders until step 7;
the run refuses to start without them.

```toml
[llm]
model = "<the exact model Tim will run in production>"
endpoint = "https://api.openai.com/v1"    # or: auth = "chatgpt" and no endpoint

[reconcile.embedding_floors]
"bge-small-en-v1.5:int8" = 0.8

[injection.reranker_floors]
"jina-reranker-v1-turbo-en:int8" = 0.0
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

```sh
/usr/bin/time -v asphodel replay --corpus "$ASPHODEL_REPLAY_DIR/corpus/state.jsonl" --mode live \
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

## 5. Probes (Tim)

Show Tim the HTML page and the `memories` list in the report. He writes
`$ASPHODEL_REPLAY_DIR/probes.toml`, with opaque ids and `memory` a regex
over sentences. The kinds are `band`, `faded_at`, `exists`, `absent`,
`agenda_has`, `agenda_lacks`, `recall_finds`, `recall_lacks`, `injects`,
`not_injects`, `profile_has`, `profile_lacks`; field shapes are in
`docs/replay.md`. Then:

```sh
asphodel replay --corpus "$ASPHODEL_REPLAY_DIR/corpus/state.jsonl" --mode replay \
    --config "$ASPHODEL_REPLAY_DIR/replay.toml" --probes "$ASPHODEL_REPLAY_DIR/probes.toml" \
    --aggregate "$ASPHODEL_REPLAY_DIR/aggregate-probes.json"
```

Exit 0 all passed, 1 some failed (the report says what each observed), 2
refused.

## 6. Bench

```sh
asphodel bench --corpus "$ASPHODEL_REPLAY_DIR/corpus/state.jsonl" \
    --concurrency 1 --concurrency 4 --concurrency 16 --requests 64 \
    --report "$ASPHODEL_REPLAY_DIR/reports/bench.json"
```

It copies the replayed store under `bench/` and starts a daemon on loopback
with the production 1.5 s reranker deadline on. It never touches another
store.

## 7. Labels and calibration (Tim)

Tim labels candidates in `labelling.json` into
`$ASPHODEL_REPLAY_DIR/labels.toml` (`"r1.1" = true`, one line per candidate
id). Then:

```sh
asphodel report precision --labels "$ASPHODEL_REPLAY_DIR/labels.toml" --material "$ASPHODEL_REPLAY_DIR/labelling.json"
```

The curve is numbers only and may be reported whole. Tim picks the two
floors from it; put them in `replay.toml` and in the production
`asphodel.toml`. The recall curve matches the gate exactly. The call-2 curve
covers only candidates the placeholder reconcile floor let call 2 see, so
after the floors change, re-run step 4 (approval again) and re-label if the
material changed.

## 8. A/B runs

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
- Shipped, for Tim's eyes: `asphodel recalls --bank main --limit 50` on the
  bench daemon, then `asphodel memory show` on injected ids. Tim judges
  relevance; you count his verdicts.
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
- Manual: wall time of each replay run (`/usr/bin/time -v`, "Elapsed"), and
  cassette size on disk.
- Manual, if a real daemon is reachable: `curl -s -o /dev/null -w
  '%{time_total}\n'` against `/v1/health`, which should be milliseconds. Do
  not send prefetches to the production daemon; prefetch writes the recall
  log.

### Resource use

- Not implemented in Asphodel. Measure manually: `/usr/bin/time -v`
  "Maximum resident set size" for the replay and for `asphodel serve` on the
  bench copy with both models loaded; `du -sh "$ASPHODEL_REPLAY_DIR/store"
  "$ASPHODEL_REPLAY_DIR/cassettes"`; `kubectl top pod hermes-0 --containers`
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
- corpus_hash: <first 12 hex> · cassette_hash: <first 12 hex> · hermes schema: 31
- import counts (dry run): <sessions / turns / cron / skipped>
- LLM: auth=<api_key|chatgpt> model=<model> · approved by Tim on <date>
- modes run: live <y/n>, replay --self-test <passed/failed>, fast <n runs>
- overrides tried: <key = value, ...> (or none)

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
- Floors must be set before the run that calibrates them; one re-run is
  expected.
- Reranker versus rank fusion has no tooling.
- Import is pinned to Hermes schema 31.
- ChatGPT subscription mode is unverified against the real backend and
  throttles on usage windows.
- A daemon serving a bank whose embedding model it doesn't carry refuses
  that bank; only the current models are in the image.
