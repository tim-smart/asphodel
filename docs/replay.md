# Replay

`asphodel replay` runs the service layer in-process as a discrete-event
simulation on a simulated clock, so decay can be watched over years in
seconds. This page is the contract for the scripted side of it: the scenario
file, the command, the report and the probes. See also ADRs 0004 and 0008. Real-history
replay (the `state.db` importer, cassettes, `fast` mode, `bench`, the
A/B diff, the HTML page, and the labelling material and precision curve)
builds on the same engine and is under "Real history" below.

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
  `<replay dir>/store/`, marked as replay's own; a `store` directory
  replay didn't create is refused, never reset. Replay holds a lock on
  the private dir for the whole run, so a second replay on the same dir
  is refused, and the store is reset under that lock.
- `--report` is where the JSON report goes. Without it the report is
  written to `<replay dir>/reports/<scenario name>.json`, so a scenario's
  `name` is one filename component. A report path inside a git working
  tree, or one that is already a symlink, is refused. A scripted
  scenario's report may be written outside the private dir, since it
  derives from a checked-in fixture; a real-history report never leaves
  it.
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
entities. Scripted entity links remain open; entity ids are deterministic.

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
`p<n>`, counting from 1 in file order, and ids are unique. `memory` is a
claim label. The prefetch and recall probes run on sessions named
`probe:<id>`, which no scenario session may use.

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

`agenda_lacks` and `recall_lacks` are the negatives of `agenda_has` and `recall_finds`, the way `not_injects`
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
- **Extraction** is queued at a source's sync. Each bank has one simulated
  worker. Whenever it's free, at a sync or at its previous completion, it
  claims the head of the production queue (turns before documents, then
  observed time), runs its LLM calls and neighbour search at once, as
  production's worker does when it claims, and commits that chunk a
  latency later. A neighbour that a sweep purged in between is planned
  without: call 2's labels on it are dropped, so a claim that would have
  ended, refined or restated it commits as new. The check runs inside the
  commit's transaction, under the same hold on the store as its writes,
  so in `serve`, where the sweep and erases run on another thread, a
  neighbour can't go between the check and the writes either. At the completion the
  worker also prepares and commits the same source's next chunks, for as
  long as each is the queue's head. Once another source is at the head, for example a
  turn synced in the meantime, the worker claims that instead. The rest
  of the document then waits behind it and is charged another latency
  when its turn comes. A probe or prefetch before a completion sees the
  store without those memories. Accesses are stamped with the source's
  ingest time, as in production. The run ends at the latest of the last
  event, `--until` and the last completion.
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
- **Ids are deterministic**. A memory id is UUIDv5
  of its source's id and its claim ordinal, written `<chunk
  position>:<index in call 1's reply>` so a document's chunks can't
  collide. An entity id is UUIDv5 of (creating
  source id, `entity:<chunk position>:<name>`), where the proposed name is
  composed to NFC, trimmed and lowercased: the exact key
  `resolve_proposals` dedups on within a commit. ID generation uses that
  key unchanged, with no second normalization. Neither memory nor entity
  keys depend on other store contents; the `entity:` prefix keeps them
  disjoint under one source. A source's id comes from its key and a chunk's
  from its source and position. Ids with no parent (recall ids, edits, banks, models)
  count up under a fixed namespace. Production keeps UUIDv7.
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
returned. The other report fields (injected and profile tokens,
refresh calls per day, the call-2 rate, agenda lines per day, the
histograms, probe results) sit beside these and aren't pinned by the
scripted tests. The report is a plain serde value with keys in struct
order, so two runs compare byte for byte.

## The first set

| Scenario | What it checks |
|---|---|
| `lifetimes` | The lifetimes table: one mention at each level fades at 15 days, 2 months, 9 months, 3 years and 12 years of bank time, within 5%. |
| `purge-table` | ADR 0008: one trivial, minor or notable mention is purged at 9 months, 3 years and 12.5 years; trivial mentioned on 4 occasions, minor on 3, notable on 2 and anything major never is; a claim stated again after its memory was purged comes back as a new memory and the shadow table counts it. |
| `maya-to-mia` | Correcting Maya to Mia retracts Maya, makes Mia the head, hides Maya from recall, and Mia inherits Maya's accesses: she's still in recall 100 days on, where a fresh minor memory would have faded, and fades at the day the inherited log gives. |
| `rescheduled-appointment` | A reschedule retracts the old slot, the agenda lists the new one and not the old, and the appointment is recently past once it has happened. |
| `three-week-holiday` | ADR 0004: a trivial memory mentioned once is still in recall after a three-week gap, where world time would have faded it, and fades at the bank day the quiet rate gives. |
| `extraction-latency` | Extraction latency is simulated: a turn's memories don't exist until its completion, and a bank's completions queue behind each other. Also the `--until` fixture. |

The tolerance for the lifetimes and purge ranges is 5% of the closed-form
crossing, the same as the unit tests on the pure function, plus one day on
purges for the nightly sweep.

## Real history

Tim's whole Hermes history replays privately. Everything derived from it lives under
`ASPHODEL_REPLAY_DIR` and is refused anywhere else: the `state.db` copy, the
manifest, the corpus, the cassettes, the replayed store with its recall
log, the probes file, the reports and their pages, and the labelling
material and labels. Only the `--aggregate` export may
leave. Agents never read the directory; keep it outside the Multica
workspaces tree.

A `live` run sends the history to the LLM endpoint `--config` names, as
production already does. A hosted endpoint sees it.

`docs/hermes-data-evaluation.md` is the step-by-step handoff for running
this on Tim's history, with what to measure and how to report it.

Errors hold no content (ADR 0010, "Logging"). A manifest, probes file,
corpus or cassette that doesn't parse is named with the line and column,
never quoted; a probe whose regex doesn't compile is named by its id, not
its pattern. The parser's own message, which can quote the input, is
logged at `trace` only.

### Importing

```
asphodel import --state-db <copy of state.db> --manifest <file> \
    [--out <replay dir>/corpus/<name>.jsonl] [--dry-run]
```

Both inputs are real history, so both must be inside the private dir:
the manifest and the `state.db` copy are refused anywhere else, and as a
symlink, before either is read. Copy `state.db` into the private dir
first.

The importer reads only `sessions(id, source, parent_session_id,
started_at)` and `messages(role, content, timestamp, active, compacted,
_compressed_summary, tool_calls)`, and never the system prompt or
`api_content`. It checks `PRAGMA table_info` for each of those columns,
with the type Hermes declares it with, and the `schema_version` table
against the versions it was written against (30, hermes-agent 0.21.5,
and 31, hermes-agent `bfc71526`), and refuses the file naming everything wrong: a missing
column, a column declared with another type, or another version.

It reads only rows with `active = 1 OR compacted = 1`, the predicate
Hermes uses for search, in `timestamp` order, not row id. Then, per primary session in `started_at` order:

- a turn is a `user` row and the final `assistant` row before the next
  `user` row; assistant rows that only call tools, and `tool` rows, are
  skipped. The turn is a `prefetch` event at the user row's time and a
  `sync` event at the reply's time, with the previous user message as the
  prefetch's `previous_query`;
- compacted rows (`active=0, compacted=1`) are replayed. Hermes' summary
  row (`_compressed_summary=1`) is skipped, and a `clear` is emitted at its
  time;
- rows with `active=0, compacted=0` never replay. They are the originals of
  a tail a compaction carried forward, which has a live clone, and turns a
  rewind took back. The clones keep the originals' timestamps under later
  ids, so timestamp order replays a carried turn once, when it was said,
  before the clear;
- the user text is shaped as `sync_turn` shapes it: multimodal content
  (`\0json:` and a parts list) keeps its text parts, the memory block
  (`<memory-context>…</memory-context>`, or the manifest's
  `[memory_block]`) is cut out, backfill before the last `[New message]`
  is stripped, and the `[Name] ` prefix stays and picks the speaker: a
  manifest `[[speaker]]` by name, else the owner;
- cron sessions (`source = "cron"`) get prefetch only, as class `cron`;
  subagent sessions (`parent_session_id` set) produce nothing.

The manifest is TOML: `timezone`, `bank` (default `main`), `assistant`,
`[owner] name, platform_ids`, `[[speaker]] name, id`, an optional
`[memory_block] start, end`, and optional `[[model]]` tables in a
scenario's shape (`name`, `question`, `max_tokens`, `kinds`; see
"Models" above). `state.db` holds no mental models, so these are the ones
a corpus run creates in the bank before the first event, beside the
"User profile" every bank is seeded with (ADR 0007). They share the
bank's mental model token budget with it, as `model create` does, so a
run whose models don't fit is refused.

The corpus is JSON lines: a header (version, bank identity, timezone, the
manifest's models when there are any, the Hermes schema version and the
import's counts) then one event per line in time order. The models are in
the header, so the corpus hash covers them. Its SHA-256 is the `corpus_hash` every report embeds. The same
history always imports to the same bytes. `--dry-run` prints the counts and
writes nothing; the counts hold no text, so they can be shared.

### Running

```
asphodel replay --corpus <file> --mode live|replay|fast \
    [--cassette <file>] [--probes <file>] [--report <file>] [--aggregate <file>] \
    [--labelling <file>] [--no-cache] [--refresh live|recorded|off] [--self-test] \
    [--config FILE] [--overrides FILE] [--latency DURATION] [--until TIMESTAMP] \
    [--onnx-threads N] [--token-dir DIR]
```

- **Modes**. Every LLM call is keyed by SHA-256 of
  the model id, the template name and version, and the whole request.
  `live` answers from the cassette and calls and records on a miss;
  `--no-cache` empties the cassette when the run opens it and records
  afresh, so a re-recording leaves one record per call, and the report's
  cassette hash is of the empty cassette it started from. `replay` answers from
  the cassette and fails on a miss, with exit 2 and no report. `fast`
  reuses call 1's claims by chunk (source id and chunk position) and `used`
  verdicts by (reply hash, sentence hash) pair, judges the pairs nobody has
  judged with one short `judge_used` call, and answers call 2 and refreshes
  by request key, calling the LLM on a miss when one is configured. The
  report counts every miss, so "fast with zero misses" is a number.
- **Refreshes in `fast`**: `--refresh recorded` (the default) substitutes
  the recorded refresh of the same mental model nearest in simulated time,
  among those made with this run's LLM model and template version; `live`
  calls on a miss, and `off` answers with no edits. Refresh handles (`m1`,
  `e1`, …) are positional, so a refresh's record keeps the memory or entry
  each stood for, and a substituted reply is carried over by identity: a
  handle goes to the memory it meant, then to that memory's handle now.
  An operation whose entry or any cited memory isn't in this run's input
  is dropped whole. Citations are checked by identity rather than by name.
  A refresh recorded before identities were
  kept carries nothing over. Triggers are counted
  by code in every mode. The mental models to refresh are the manifest's
  `[[model]]` tables, carried in the corpus header.
- **The LLM** for `live` and `fast` is built as `serve` builds its own:
  `[llm]` in `--config` with `ASPHODEL_LLM_API_KEY`, or the ChatGPT login
  under `--token-dir` (the private dir by default). `replay` refuses one.
- **The cassette** defaults to `<replay dir>/cassettes/<corpus stem>.jsonl`:
  one record per line with the request, the reply, the measured latency,
  the simulated time, and for extraction calls the chunk and the handles
  with a hash of each sentence. The report embeds its SHA-256 as it stood
  when the run started.
- **Latency**. `--latency` sets every chunk's.
  Without it, a chunk's calls run when the worker claims it, and it
  completes after the latency of the responses that answered them: the
  recorded latency of each record served, and the measured round trip of
  each live call, which is what its record says. Only the responses
  actually served count, never older recordings of the chunk or another
  model's, so a `live` run and the `replay` of its cassette simulate the
  same lag. The measured round trips of live calls are reported under
  `llm.latency_ms`.
- **Models.** The real models run with ONNX Runtime's intra-op threads
  pinned to `--onnx-threads` (1 by default). `ASPHODEL_MODELS=fake` runs the
  deterministic fakes, for tests.
- **Probes** are a TOML file of `[[probe]]` tables with the scripted kinds
  and fields, opaque ids and `memory` a regex over sentences: the earliest
  memory whose sentence matches is the one probed. `injects` and
  `not_injects` need the real models.
- **`--self-test`** runs the simulation twice under the one lock and
  requires byte-identical reports. It is refused in `live`, which measures
  latency and records as it goes.
- The report goes to `--report` or `<replay dir>/reports/<stem>-<mode>.json`,
  inside the private dir only.

### The report and the aggregate

A real-history report has `kind` `live`, `replay` or `fast`, `group`
`models` or `fake`, `corpus_hash` and `cassette_hash`, and beside the
scripted fields: `injected_tokens` (per session with a synced turn, the
per-turn p50 and p95, and cron apart), `profile_tokens` (sampled daily),
`call2_rate`, `agenda_lines_per_day`, `significance_histogram`,
`kind_histogram`, `memories` (each created memory with when it faded and
whether it was purged), and `llm` with `cache`, `top_up`, `live`, `misses`,
`used_verdicts` by source and `latency_ms`. A `live` run and the `replay`
of its cassette differ only in `kind`, `flags`, `llm` and `cassette_hash`.

`--aggregate <file>` writes the one thing that may leave the private dir. Its type has no string field but a probe's id: the run's kind
is a set of booleans, days are days since the epoch, weeks are two
integers, and the hashes and the git SHA are byte arrays. It carries the
probe results, the purge, fade and band series, the token, lag and call
counts, and the histograms.

`asphodel report diff A B [--force]` compares two reports: it refuses runs
on a different corpus or cassette unless forced, lists probes whose result
changed, numbers that differ beyond a tolerance of one in a million, and
the memories that faded or were purged in one run and not the other, by
id.

### The HTML page

```
asphodel report html <report> [--out <file>]
```

writes one static page from a JSON report: its
identity (kind, corpus and cassette hashes, git SHA), the probe results,
and every number the report holds, from injected tokens to the histograms
and where the LLM replies came from. Styles and any charts are inline, and
nothing on the page loads from anywhere: no `src`, `href` or `url(` but a
fragment or a `data:` URL, no `@import`, and no script that opens a
connection. The page goes to `--out`, or beside the report with the
extension `html`. Both the report and the page must be inside the private
dir, and neither may be a symlink.

### Labelling and the precision curve

The two calibrated floors, the reranker gate floor and the
reconcile similarity floor, are set from Tim's labels, not by eye. `asphodel replay --corpus ... --labelling <file>`
writes the material to label, inside the private dir. Writing it changes
nothing the run simulates, and the same run writes the same bytes.

The material is one JSON object:

```json
{
  "version": 1,
  "recall": [
    { "sample": "r1", "at": "<prefetch time>", "session": "<session id>",
      "query": "<the query the reranker scored against>",
      "candidates": [
        { "id": "r1.1", "memory": "<uuid>", "score": 1.5, "sentence": "..." } ] } ],
  "call2": [
    { "sample": "c1", "at": "<when the worker claimed the chunk>",
      "claim": "<the claim's sentence>",
      "candidates": [
        { "id": "c1.1", "memory": "<uuid>", "score": 0.93, "sentence": "..." } ] } ]
}
```

- `recall` holds 50 of the run's synced turns, or every one when there
  are fewer. Cron prefetches, probes and refreshes aren't turns and are
  never sampled. The sample is spread evenly over the run in time order
  (turn `i × n / 50` of `n`), so the same run always samples the same
  turns. Each lists the prefetch's reranked candidates in ranked order
  before the gate, including those the gate turned away, scored with the
  reranker logit the gate floor compares. `query` is what the reranker
  scored against, after a short follow-up borrowed the previous message.
- `call2` holds every candidate list call 2 was shown, one per claim:
  the claim and its neighbours, scored with the cosine similarity of the
  claim to each. Only what call 2 was shown is here. Flagged claims bypass
  the vector floor, and BM25 neighbours are not filtered by it, so call 2
  can see neighbours below the reconcile floor. The curve measures
  precision by score threshold over these observed candidates; it does
  not predict what raising or lowering the reconcile floor would retain.
  Unobserved candidates are outside this material.
- A candidate's `id` is unique in the file and is what a label names;
  `memory` is the memory's id in the replayed store.

The labels file is TOML, written by Tim inside the private dir: one key
per candidate id, `true` when the candidate is relevant (for recall, worth
injecting for the query; for call 2, about the same thing as the claim)
and `false` when it isn't. Candidates without a label are left out and
counted.

```toml
"r1.1" = true
"r1.2" = false
"c1.1" = true
```

```
asphodel report precision --labels <file> --material <file>
```

prints the curve as JSON: for `recall` and for `call2`, `labelled`,
`unlabelled` and `curve`, one point per distinct score among the labelled
candidates in ascending order. At each point's `floor`, `kept` counts the
labelled candidates scoring at or above it, `relevant` counts those
labelled `true`, and `precision` is `relevant / kept`. For recall this
matches the gate's logit comparison. For call 2 it is a score-threshold
curve over observed candidates, not a prediction for another reconcile
floor. The curve is numbers only. A label naming no candidate
in the material is refused. Both files must be inside the private dir, and
an error names the file and line, never the text.

### Bench

```
asphodel bench --corpus <file> [--concurrency N]... [--requests N] \
    [--listen 127.0.0.1:0] [--report <file>]
```

`bench` copies the replayed store under `<replay dir>/bench/`, starts the
daemon on the copy on a loopback address with the production reranker
deadline on, and for each concurrency level (1, 4 and 16 by default) runs
`--requests` prefetches (32 by default) over HTTP from that many
connections, with the corpus's prefetch queries in order. It never runs
against any other store: the original is read only, and a private dir
with no replayed store, or a listen address off loopback, is refused. The
report lists per level the p50, p95, p99 and maximum latency in
milliseconds and the fraction the reranker answered within its deadline.
