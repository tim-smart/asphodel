# Replay

`asphodel replay` runs the service layer in-process as a discrete-event
simulation on a simulated clock, so decay can be watched over years in
seconds. This page is the contract for the scripted side of it: the scenario
file, the command, the report and the probes. It comes from "Replay harness:
simulated-clock replay of recorded sessions" (TIM-96) and its amendments
from TIM-97 and TIM-98, and from ADRs 0004 and 0008. Real-history replay
(the `state.db` importer, cassettes, `fast` mode, `bench` and the A/B diff)
builds on the same engine and is documented when it lands.

## Running a scenario

```
asphodel replay --scenario scenarios/maya-to-mia.toml \
    [--replay-dir DIR] [--report FILE] [--config FILE] [--overrides FILE] \
    [--latency DURATION] [--until TIMESTAMP]
```

- `--replay-dir`, or `ASPHODEL_REPLAY_DIR`, is the one private directory
  everything derived from a run lives in: the replayed store, its lock, the
  shadow table of purged rows and the reports. Replay refuses to run with
  neither set, refuses a directory inside a git working tree (any ancestor
  holding `.git`), and refuses a directory that holds `asphodel.db` at its
  top level, which is a `serve` data dir. Its own store is under
  `<replay dir>/store/`, under the same exclusive lock `serve` takes.
- `--report` is where the JSON report goes. Without it the report is
  written to `<replay dir>/reports/<scenario name>.json`.
- `--config` is the production tuning file and `--overrides` a file in the
  shape of `Tuning`. The layers, lowest first: code defaults, the fake
  floors (group `ci` only, below), `--config`, the scenario's own
  `[tuning]`, then `--overrides`. Each layer must have the shape of
  `Tuning` on its own; an unknown key is refused before anything runs.
- `--model-dir`, or `ASPHODEL_MODEL_DIR`, is where group `models` finds the
  real models.
- `--latency` is the simulated extraction latency, a duration such as
  `10m` or `30s`. It overrides the scenario's `latency`; the default is
  `0s`.
- `--until` advances the clock past the last event and keeps running the
  sweeps and refreshes due until then.

Exit codes: 0 when every probe passed, 1 when a probe failed (the report is
still written, with the failure in it), 2 when the scenario or the
arguments are refused or the run itself failed (no report).

A scenario in group `models` needs the real models in
`ASPHODEL_MODEL_DIR`, and is refused with exit 2 without them. The `cargo
test` runner skips those scenarios when the directory isn't set, so CI
needs no network.

## The scenario file

Scenarios are TOML files checked in under `scenarios/`. They're written
from scratch, never adapted from real sessions, and whoever writes one must
not have seen a private report.

```toml
name = "maya-to-mia"
group = "ci"                    # "ci" runs on the fake models in CI; "models" needs the real ones
description = "Correcting Maya to Mia keeps the corrected name as strong as the wrong one was."
latency = "0s"                  # optional; the default

[bank]                          # optional; every field is
name = "main"
timezone = "Pacific/Auckland"   # the bank's default timezone; "UTC" when absent
owner = "Tim"
assistant = "Hermes"
owner_platform_ids = ["discord:1234"]

[tuning]                        # optional; the shape of Tuning, layered as above
clock.quiet_rate = 1.0

[[turn]]
at = "2026-01-05T09:00:00Z"     # the user message's time: prefetch runs here
reply_at = "2026-01-05T09:00:30Z"  # optional; sync_turn runs here, `at` when absent
session = "s1"
user = "My sister is called Maya."
assistant = "Noted."
used = []                       # labels of memories the reply relied on
# author = { id = "discord:99", name = "Sam" }   # optional; a non-owner speaker
# platform = "discord"

[[turn.claim]]
label = "maya"                  # how probes and later outcomes name the memory
content = "Tim's sister is called Maya."
quote = "My sister is called Maya"
kind = "fact"
significance = "minor"

[[probe]]
id = "mia-is-the-head"
at = "2026-01-17T10:00:00Z"
kind = "exists"
memory = "mia"
head = true
```

Events are `[[turn]]`, `[[chatter]]`, `[[document]]` and `[[clear]]`, each
with an `at`. They may be listed in any order; the engine sorts them. A
scenario with no claims anywhere is allowed and extracts nothing.

### Models

```toml
[[model]]
name = "profile"
question = "Who is the user?"
max_tokens = 500                # optional; the default
kinds = ["fact", "state"]       # optional; every kind when absent
```

Created in the bank before the first event, enabled, with no entity
filter. See "Refreshes" under "The simulation".

### Turns

A turn is one user message and the assistant's reply, as Hermes hands them
over. Prefetch runs at `at` with the user text as its query, and
`sync_turn` at `reply_at`, which defaults to `at`. The turn's speaker is
the owner unless `author` is given.

`used` lists the labels of memories the reply relied on. Each must be in the
session's in-context set at the turn (injected by its prefetch, listed in
the agenda, or returned by an earlier `recall` in the session); a label
that isn't is a scenario error, so a `used` that would silently count for
nothing can't hide in a scenario.

### Chatter

A run of turns with nothing to extract, to keep the bank in conversation so
bank time runs at full speed (ADR 0004):

```toml
[[chatter]]
from = "2026-01-29T09:00:00Z"
every = "1d"
count = 22
session = "chatter"             # optional
```

Each turn's user text is `Just checking in (<n>).` and its reply
`Hello again.`; neither contains a claim. Chatter never overlaps a
scripted turn's words by accident only if the scenario's own sentences
avoid those words.

### Documents

```toml
[[document]]
at = "2026-01-05T09:00:00Z"
id = "notes-2026-01"
text = "..."
reference_date = "2026-01-05"
timezone = "Pacific/Auckland"   # optional
[[document.claim]]
...
```

A document is ingested at `at` with its claims scripted per chunk in
order. Documents never appear in real history; only scenarios ingest them.

### Clear

```toml
[[clear]]
at = "2026-01-06T09:00:00Z"
session = "s1"
```

Clears the session's in-context set and pending injection, as Hermes asks
on compaction, reset or rewind.

### Claims

A claim is what extraction would have found in the chunk. The scenario
skips the LLM by listing them, and the engine answers call 1 and call 2
from the list. Everything after the reply is production code: the checks
on call 1's reply, the neighbour search, reconciliation and the commit.
Reconciliation has no replay-only branch.

| Field | Meaning |
|---|---|
| `label` | Optional. Names the memory the claim creates. Unique across the scenario. |
| `content` | The memory's sentence. |
| `quote` | The words in the chunk it rests on; a quote not in the chunk drops the claim, as in production. |
| `kind` | `fact`, `event`, `state`, `task` or `recurring`. |
| `significance` | `trivial`, `minor`, `notable`, `major` or `critical`. |
| `remember_this` | Keeps the memory. Default false. |
| `changes_something` | Flags the claim for call 2. Default false. |
| `valid_from`, `valid_until`, `due_at`, `recurrence_start` | `{ at = "2026-02-10T14:00", precision = "minute" }`, local to the source's timezone. |
| `low_confidence` | Lowers window confidence. Default false. |
| `until_event`, `volatility`, `recurrence_text`, `recurrence_rrule` | As extraction returns them. |
| `reconcile` | The claim's outcomes against existing memories, below. |

Entity links aren't scripted in the first set; every claim links no
entities. That's open, with the deterministic entity ids of TIM-96.

#### Reconcile outcomes

```toml
reconcile = [{ memory = "maya", outcome = "retracts" }]
```

`outcome` is one of the glossary's reconciliation outcomes:
`mentioned_again`, `confirmed`, `ends`, `retracts`, `denies` or `refines`.
`memory` is the label of a memory an earlier event created.

The target must be among the neighbours call 2 is shown for the claim. If
call 2 doesn't run for the claim, or runs without that neighbour, the run
stops with a scenario error naming the claim's label (or its ordinal) and
the target. Raise the overlap with the target's sentence or flag the claim;
the engine never smuggles a target in, because that would make the
scenario say something production wouldn't do.

A claim whose outcomes absorb it (`mentioned_again` or `confirmed` only)
creates no memory, so a `label` on it is a scenario error.

### Probes

A probe pins a time and an expectation. It never changes the run: no probe
writes an access or joins a session's in-context set. (`injects`,
`recall_finds` and `recall_lacks` write the recall log, which isn't an
access, and never a session's set.)

Every probe has `at`, `kind` and an optional `id`; without one the id is
`p<n>`, counting from 1 in file order. `memory` is a claim label.

| `kind` | Fields | Passes when |
|---|---|---|
| `band` | `memory`, `band` | The memory's band at `at` is `strong`, `fading` or `faded` as given. |
| `faded_at` | `memory`, `between = [from, to]` | The first instant strength fell below τ is within the range, inclusive. It's computed at `at` from the access log and bank time, to the minute, as `memory show`'s projection does. Not yet faded by `at` fails. |
| `exists` | `memory`, and any of `memory_kind`, `ended`, `retracted`, `head`, `phase` | The memory is in the store and every given field matches. `head` is whether it's the head of its supersession chain; `phase` is `upcoming`, `current`, `overdue`, `recently_past` or `long_past`. |
| `absent` | `memory` | The memory isn't in the store: purged or forgotten. |
| `agenda_has`, `agenda_lacks` | `memory` | The bank's agenda at `at` lists, or doesn't list, the memory. |
| `recall_finds`, `recall_lacks` | `memory`, `query` | Explicit recall for `query`, with no session, returns, or doesn't return, the memory. |
| `injects`, `not_injects` | `memory`, `query` | A prefetch for `query` on a session no turn uses injects, or doesn't inject, the memory. Group `models` only. |
| `profile_has`, `profile_lacks` | `model`, `memory` | The mental model has, or lacks, an entry citing the memory. Not in the first set: scripted refresh replies are open. |

`agenda_lacks` and `recall_lacks` aren't in TIM-96's list. They're the
negatives of `agenda_has` and `recall_finds`, the way `not_injects`
negates `injects`, and the rescheduled appointment and Maya to Mia need
them. Exact strength values are unit tests on the pure function, not
probes.

A probe naming a label no claim defines is refused before the run.

## The simulation

- **One event queue** holds prefetches, syncs, extraction completions,
  sweeps, refreshes and probes, ordered by simulated time. At the same
  instant the order is: sweeps and refreshes, then completions, then
  prefetches, then syncs, then probes, and within a kind the order they
  were scheduled in. So a probe at a turn's `at` sees that turn's prefetch
  and, with zero latency, its memories.
- **Extraction** of a turn or document is queued at its sync and committed
  at its completion event, scheduled at the later of the sync time and the
  bank's previous completion, plus the latency. Each bank has one
  extraction worker, so completions never overlap. A probe or prefetch
  between the two sees the store without the turn's memories. Accesses are
  stamped with the source's ingest time, as in production.
- **Sweeps** run at `mental_models.sweep_time` bank-local (04:00) on the
  simulated clock, purge first, then the source and recall-log sweep, and
  then the refreshes due. Replay records the deletion fingerprint on its
  own store and never pauses purge.
- **Refreshes** run on the production schedule: the debounce after a
  notable write starts from the completion event, and the daily sweep
  follows the purge. A scenario's `[[model]]` sections create mental
  models before the first event so there is something to refresh. The
  scripted LLM answers every refresh with no edits, so entries stay empty
  and `refresh_calls_per_day` is what a scenario can watch; scripted
  refresh replies are still open.
- **Bank time** comes from the turns the engine ingests, chatter included.
  Documents don't move it.
- **Wall-clock timeouts.** The reranker deadline is off, so the reranker is
  never skipped. The in-context idle timeout and the mapping expiry run on
  the simulated clock.
- **Ids are deterministic.** A memory id is UUIDv5 of its chunk's id and
  its claim ordinal in call 1's reply, and an entity id of the chunk that
  proposed it and its name (TIM-96 says the source and the surface form;
  the chunk is what commit knows, a turn is one chunk, and the name is what
  one entity is created under when several surface forms propose it). A
  source's id comes from its key and a chunk's from its source and
  position, so the chain is stable from the key down. Ids with no parent
  (recall ids, edits, banks, models) count up under a fixed namespace.
  Production keeps UUIDv7.
- **Determinism.** With the same scenario, layers and flags, a run writes a
  byte-identical report on the same machine and build. The report carries
  no wall-clock time. Every sort that reaches ranking or the report breaks
  ties by id.
- **The fake floors.** Group `ci` runs on the deterministic fake embedder
  and reranker, which need floors (ADR 0009). The engine layers
  `reconcile.embedding_floors."fake-embedder:v1" = 0.5` and
  `injection.reranker_floors."fake-reranker:v1" = 0.0` above the code
  defaults; a scenario's `[tuning]` can change them.
- **The shadow table.** Every purged chain's rows (content and embedding)
  are copied to a table in the replay store, never anywhere else. After
  the run, each memory created after a purge whose nearest shadow row,
  purged before it, is at or above the reconcile floor counts as purged
  then re-mentioned.

## The report

One JSON object per run. The fields the scripted scenarios pin:

```json
{
  "kind": "scripted",
  "scenario": "purge-table",
  "group": "ci",
  "version": "0.1.0",
  "git_sha": "…",
  "tuning": { "clock": { "quiet_rate": 1.0 }, "…": "the resolved Tuning" },
  "flags": { "latency_ms": 0, "until": null, "refresh": "off" },
  "probes": [
    { "id": "trivial-fades", "at": "2026-01-22T09:00:00Z", "kind": "faded_at",
      "passed": true, "observed": { "faded_at": "2026-01-20T11:15:00Z" } }
  ],
  "purges_per_day": [ { "day": "2026-09-26", "purged": 1 } ],
  "purged_then_re_mentioned": { "purged": 3, "re_mentioned": 1, "rate": 0.3333 },
  "fade_outs_per_week": [],
  "bands_per_week": [],
  "extraction_lag": { "p50_ms": 0, "p95_ms": 0 },
  "llm": { "scripted": 12, "cache": 0, "top_up": 0, "live": 0 }
}
```

`kind` is `scripted` for a scenario and `live`, `replay` or `fast` for real
history. `observed` holds what the probe saw, in a shape per probe kind:
`faded_at` gives the instant or null; `band` the band and strength;
`exists` and `absent` the memory's id and the fields `exists` can check;
the agenda probes the ids listed; the recall and inject probes the ids
returned. The other TIM-96 report fields (injected and profile tokens,
refresh calls per day, the call-2 rate, agenda lines per day, the
histograms, probe results) sit beside these and aren't pinned by the
scripted tests. The report is a plain serde value with keys in struct
order, so two runs compare byte for byte.

## The first set

| Scenario | What it checks |
|---|---|
| `lifetimes` | TIM-91's lifetimes table: one mention at each level fades at 15 days, 2 months, 9 months, 3 years and 12 years of bank time, within 5%. |
| `purge-table` | ADR 0008: one trivial, minor or notable mention is purged at 9 months, 3 years and 12.5 years; trivial mentioned on 4 occasions, minor on 3, notable on 2 and anything major never is; a claim stated again after its memory was purged comes back as a new memory and the shadow table counts it. |
| `maya-to-mia` | Correcting Maya to Mia retracts Maya, makes Mia the head, hides Maya from recall, and Mia inherits Maya's accesses: she's still in recall 100 days on, where a fresh minor memory would have faded, and fades at the day the inherited log gives. |
| `rescheduled-appointment` | A reschedule retracts the old slot, the agenda lists the new one and not the old, and the appointment is recently past once it has happened. |
| `three-week-holiday` | ADR 0004: a trivial memory mentioned once is still in recall after a three-week gap, where world time would have faded it, and fades at the bank day the quiet rate gives. |
| `extraction-latency` | Extraction latency is simulated: a turn's memories don't exist until its completion, and a bank's completions queue behind each other. Also the `--until` fixture. |

The tolerance for the lifetimes and purge ranges is 5% of the closed-form
crossing, the same as the unit tests on the pure function, plus one day on
purges for the nightly sweep.
