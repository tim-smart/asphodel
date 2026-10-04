# Operations

How to run Asphodel and look after it. Every operator command is an HTTP
client of the running daemon, and anything that changes the store at scale
runs as a daemon job. There are two exceptions, both offline:
`asphodel restore`, and the copy the daemon takes before a schema
migration.

## The image

`nix build .#image` builds a layered OCI image, `asphodel:<version>`, from
the flake. It holds:

- the `asphodel` binary, wrapped so `ORT_DYLIB_PATH` points at nixpkgs'
  ONNX Runtime;
- the model dir, at `ASPHODEL_MODEL_DIR`, with every file the manifest in
  `crates/asphodel-core/src/models/manifest.rs` lists;
- the timezone database, at `TZDIR`.

There's no shell, no package manager and no `/etc/passwd`. The entrypoint
is `asphodel` and the default command is `serve`, so
`kubectl exec <pod> -c asphodel -- asphodel status` runs any other
subcommand. It runs as uid and gid 65532.

The models come from fixed-output nix fetches at the revisions and
SHA-256s the manifest pins (`nix/models.nix`), so fastembed never downloads
at runtime and the daemon's startup check always passes. The build runs
`asphodel models fetch` against the result as a check: any file the
manifest lists that isn't there would need the network, which the build
sandbox doesn't have, so a manifest change that `nix/models.nix` doesn't
follow fails the build.

`ASPHODEL_DATA_DIR` isn't set and the image has no `/data`. A pod whose
volume didn't mount fails at startup instead of serving an empty store from
the container's own filesystem.

CI builds the image on every push and serves it once, on the real models,
until `/v1/health` answers 200.

`nix build .#asphodel` (the default package) is the wrapped binary on its
own, and `nix build .#models` is the model dir, for a host that runs the
daemon outside a container.

## Deploying next to Hermes

`deploy/kubernetes/hermes.yaml` is the example: Hermes with Asphodel as a
native sidecar (Kubernetes 1.29 or later) in a single-replica StatefulSet.

- **One replica.** The daemon takes an exclusive lock on its data dir, and
  only one process may hold a bank's extraction worker. A StatefulSet never
  runs two pods for one ordinal. A Deployment works too, with
  `replicas: 1` and the `Recreate` strategy; `RollingUpdate` would start
  the new pod while the old one holds the lock.
- **Storage.** The data dir is a `ReadWriteOnce` local or block volume. At
  startup the daemon checks the filesystem and refuses NFS, SMB/CIFS,
  CephFS and FUSE, because SQLite's locking and WAL aren't safe on them.
  `--allow-network-fs` overrides that, at your own risk.
- **Readiness.** `/v1/health` answers 503 while the store migrates and the
  models load, and 200 with the version once the daemon is ready. It needs
  no token. The example gates the sidecar's startup probe on it, so Hermes
  starts once Asphodel can answer.
- **Listening.** The kubelet probes the pod's IP, not its loopback, so the
  example listens on `0.0.0.0:7720`. Off loopback the daemon requires a
  bearer token from `ASPHODEL_TOKEN` and refuses to start without one, and
  once a token is set every client sends it, Hermes included. The `asphodel`
  Service exists only for the backup job; Hermes uses
  `http://127.0.0.1:7720`. A NetworkPolicy can limit port 7720 to the
  backup job's pods, but write it to allow Hermes' own ports too: a policy
  that selects the pod denies every ingress it doesn't allow.
- **Shutdown.** On SIGTERM the daemon stops taking ingest, finishes the
  chunk in flight and checkpoints the WAL. The extraction queue is in
  SQLite, so nothing queued is lost. Native sidecars stop after the main
  containers, so Hermes stops first. The example gives the pod 120 s, which
  covers one LLM call.
- **While the daemon is down** Hermes carries on without memory. The plugin
  spools turns under `$HERMES_HOME/asphodel/spool/` and replays them once
  the daemon answers.

### Secrets and environment

| Variable | Flag | What it is |
|---|---|---|
| `ASPHODEL_TOKEN` | none | The bearer token. Required off loopback. Clients read it too. |
| `ASPHODEL_LLM_API_KEY` | none | The LLM key, for `auth = "api_key"`. |
| `ASPHODEL_LISTEN` | `--listen` | `host:port` or `unix:/path`. Default `127.0.0.1:7720`. |
| `ASPHODEL_DATA_DIR` | `--data-dir` | The store and its lock. Required, with no default. |
| `ASPHODEL_CONFIG` | `--config` | The tuning file. |
| `ASPHODEL_MODEL_DIR` | `--model-dir` | The models. The image sets it. |
| `ASPHODEL_ALLOW_NETWORK_FS` | `--allow-network-fs` | Run on a network filesystem. |
| `ASPHODEL_ONNX_THREADS` | `--onnx-threads` | ONNX Runtime's intra-op threads. |
| `ASPHODEL_URL` | `--url` | Where a client finds the daemon. Default `http://127.0.0.1:7720`. |
| `ASPHODEL_LOG` | none | The log filter (`docs/logging.md`). Default `info`. |

Secrets have no flag, so they never show in a process list.

### The tuning file

The tuning file is TOML, and every key in it is optional except three kinds:

- `[llm] model`, the exact model string. The floors are calibrated against
  one model.
- A floor for each local model the daemon runs:
  `reconcile.embedding_floors."bge-small-en-v1.5:int8"` and
  `injection.reranker_floors."ms-marco-MiniLM-L-6-v2:int8"`. They're set
  in replay from labelled history (`docs/replay.md`, "Labelling and the
  precision curve"), and the daemon refuses to start without them.
- A relevance scale for the reranker:
  `ranking.relevance_scales."ms-marco-MiniLM-L-6-v2:int8" = 3.564211`. The
  score divides the reranker's logit by it, so a reranker with a wider
  logit range doesn't drown out strength, state confidence and phase,
  whose weights were sized for jina-reranker-v1-turbo-en int8. The scale
  is the standard deviation of the new reranker's logits divided by
  jina's, both taken over every candidate in the same pools. 3.564211 was
  measured on the 69-query eval pools; the replay labelling pools give a
  different ratio (2.17), so a re-measure states which pools it used. The
  gate floor still compares the raw logit. The daemon refuses to start
  without one for the loaded reranker.
  Like the floors, the relevance scale is keyed by the exact model string,
  quantisation included, with no fallback.

The deploy and evaluation examples use a raw-logit injection floor of
`-8.0` for `ms-marco-MiniLM-L-6-v2:int8`, as recommended in TIM-131 from
10 labelled prefetches. It keeps over half the relevant memories; higher
floors roughly halve those kept for little precision gain, while the token
budget still caps injection. Live-run injected-token counts and probes
p02/p04 remain to be accepted. The relevance scale stays `3.564211`.
These are configuration values, not built-in defaults: floors and scales
must still be supplied for the exact model id.

`[llm] concurrency` (default 10) is how many LLM calls may be in flight at
once across the daemon, refreshes included, and how many chunks each bank
extracts at once. Chunks still commit in queue order, and a chunk whose
search missed a memory another chunk committed reconciles again, so a
repeat stays one memory. Set it to 1 for serial extraction. Measure changes
in replay before applying them
(`docs/replay.md`, "Concurrency").

`[llm] language` (unset by default) is the language memories and mental
model entries are written in, such as `"English"`. Unset, each is written in
the language of the text it comes from. The local models are English-only,
so set it to `"English"` if the assistant is used in another language
(`docs/models.md`). It applies to new extraction and refreshes only; a
memory already stored in another language is translated with `asphodel
memory translate` (see "Memories" under "Commands"). An empty value stops
the daemon.

`[extraction] guidance` (unset by default) is your own advice to call 1 on
what's worth remembering, added after its fixed rules, such as "Skip build
and deploy logs. Keep release dates." It can't change the rules or the reply
format, and it applies to new extraction only. Try it in replay first: its
hash is part of call 1's template, so a run with it records and reuses
claims apart from runs without it (`docs/replay.md`). An empty value stops
the daemon. Whitespace-only guidance is rejected too. Like the rest of the
tuning file, guidance is daemon-wide and takes effect on restart. It is
trimmed before insertion, and the user prompt, reply schema and fixed rules
are unchanged. Every chunk uses the same system prompt.

Call 1's version 6 prompt leaves out questions, requests and routine
assistant operations, including note and file edits. It extracts durable
content, decisions and commitments rather than the operations that record
them. A date or path alone is not an exception. Where a durable thing is
kept is extracted as a fact about that thing only when the supplied context
does not already state the location. This is a context-local check, not a
guarantee that the location has never been mentioned in the bank.

The earlier version 4 rule followed a ten-day evaluation (436 turns,
451 memories): 129 memories recorded assistant operations or reports and
91 recorded user questions. These crowded recall, while the 49 memories
rated trivial still took 7 to 13 weeks to fade. In the Hermes version 5
evaluation, 115 of 377 memories (31%) began with "Hermes" or "The assistant";
62 of those named a note, path, skill or the vault. Version 6 removes the
broad date/storage exception that admitted routine edits. Its effect on
that share and on fact-targeting probes still needs a fresh recording.

The SHA-256 of the trimmed guidance is recorded alongside the template
version in cassette keys, replay reports and aggregate exports. Without
guidance there is no hash; call 2, refresh and judge keys are unchanged.
Changing guidance does not pause purge because it is not a deletion input,
and stored memories are not re-extracted.

`[strength.significance]` sets the significance value of each level in
`strength = S·significance + max(recent_use, lasting_floor)`, with S fixed
at 2.5:

```toml
[strength.significance]
trivial = 0.0
minor = 0.2
notable = 0.5
major = 0.7
critical = 0.9
```

Those are the defaults. Lowering a level makes the memories at it fade
sooner; raising trivial from 0.0 to 0.1 adds 0.25 to its strength.
Each value is between 0 and 1, and each level must be strictly above the
one below it, so critical can be 1.0. A kept memory's significance is fixed
at 1.0 and isn't a key: it never fades. The values are deletion inputs, so
changing one pauses purge and the sweep until you acknowledge it ("Purge
pauses" below). Measure a change in replay with `--mode fast` first: it
changes what's injected, and so the in-context set call 1 sees
(`docs/replay.md`).

An unknown key or an out-of-range value stops the daemon too. The LLM's two
modes, an API key or a ChatGPT subscription, are in `docs/models.md`. For
the subscription, log in once the pod is up:

```sh
kubectl exec -it hermes-0 -c asphodel -- asphodel llm login --data-dir /data
```

It writes `/data/llm-tokens.json`, which the daemon reads before every call,
so no restart is needed. Without `[llm]` the daemon still serves recall and
injection, and chunks wait on the queue.

Some tuning values decide irreversible deletions. When they change, purge
and the sweep pause until you acknowledge them ("Purge pauses" below).

### Reaching the daemon from a laptop

```sh
kubectl port-forward hermes-0 7720
ASPHODEL_TOKEN=... asphodel status
```

The forward ends on the pod's loopback. A daemon listening only on loopback
with no token configured needs no token; the example's does.

## Setting up Hermes

### Installing the plugin

Copy `plugin/` into `$HERMES_HOME/plugins/asphodel/`. Hermes finds it there
because `__init__.py` names `register_memory_provider`, and it survives
Hermes updates because the plugin declares no Python dependencies. Use the
plugin from the same release as the daemon: the plugin warns at
`initialize` when the daemon's major version differs from the one it was
written for.

### `hermes memory setup`

Pick `asphodel`. Its `post_setup` prompts for each field (an empty answer
keeps the default), writes `$HERMES_HOME/asphodel/config.json`, sets
`memory.provider` to `asphodel`, and sets `memory.memory_enabled` and
`memory.user_profile_enabled` to false, because Hermes' built-in
`MEMORY.md` and `USER.md` are off when Asphodel is in use.

| Field | Default | What it is |
|---|---|---|
| `url` | `http://127.0.0.1:7720` | The daemon. `ASPHODEL_URL` overrides it. |
| `bank` | the profile name | One bank per Hermes profile. |
| `owner_name` | none | The owner's name, an alias of the `user` entity. |
| `owner_platform_ids` | none | The owner's speaker ids, comma-separated, such as `discord:1234`. |
| `assistant_name` | the profile name | An alias of the `assistant` entity. |
| `timezone` | Hermes' timezone | For sources that don't give one. |
| `ingest` | `true` | Off means recall and injection only. |

The token isn't prompted for. Set `ASPHODEL_TOKEN` in `$HERMES_HOME/.env`,
or in the Hermes container's environment as the example does.

**The dashboard skips `post_setup`.** It writes the fields through
`config_schema.py` and `save_config`, but leaves the built-in memory flags
alone. Set `memory.memory_enabled` and `memory.user_profile_enabled` to
false yourself. Until you do, `initialize` warns about it.

On `initialize` the plugin calls `PUT /v1/banks/{bank}`, which creates the
bank or merges the fields in. Fields it sends are set, aliases are added,
and fields it leaves out are untouched, so a running plugin can't undo a
change made with `asphodel bank config`.

Turns are ingested only from primary agents. Cron runs get injection and
the prompt block but are never ingested.

### Backfilling Hermes' history

Hermes' history in `$HERMES_HOME/state.db` can be posted to the daemon, so
Asphodel starts out knowing what Hermes knew. `hermes memory setup` offers
it as its last question. Answer yes, then give a date or `all`. A blank date
skips it. Or run it on its own from the plugin directory:

```
HERMES_HOME=~/.hermes python3 backfill.py --since 2026-01-01
HERMES_HOME=~/.hermes python3 backfill.py --all-history --speaker Sam=discord:1234
```

- **A cutoff is required.** Give either `--since YYYY-MM-DD` (the start of
  that day in the configured timezone, or UTC) or `--all-history`. Each
  turn keeps its original time, so its memories arrive already aged. A
  memory from six months ago that was never reinforced lands near the floor,
  and the first purge will take many of those.
- **`--dry-run`** prints the counts and posts nothing. They're the counts
  `asphodel import --dry-run` prints for the same database, plus, with
  `--since`, `turns_before_since`, the turns the cutoff leaves out.
- **`--speaker NAME=PLATFORM:ID`**, repeatable, names a speaker other than
  the owner: a turn starting `[NAME] ` is theirs. Any other turn is the
  owner's. Setup's prompt passes no speakers.
- **`--bank`** overrides the config's `bank`. With neither set it refuses,
  because outside Hermes there's no profile name to default to.

It reads an online-backup copy of `state.db`, never the live file, using
the rules of `asphodel import`. Only turns Hermes still keeps for search are
read. Tool rows, cron sessions and subagent sessions are skipped. The
injected memory block (Hindsight's `<memory-context>`) is cut from the
user's text, so Hindsight's memories aren't extracted as the user's own
words. A schema version other than 30 or 31 is refused.

Compactions are counted but send nothing. Unlike `asphodel import`, the
backfill never clears a session. A clear changes only live injection state
(the in-context set, pending injections and the session's prompt block),
which historical turns never built up. On a session that's live in Hermes,
a clear would wipe that state, on the first run or on any rerun.

One limit applies to the session Hermes is in while the backfill runs. The
daemon stores each new turn with the session's in-context set at the moment
it arrives, and extraction judges which memories a turn `used` against it.
A backfilled turn from that session's history, before the switch, gets the
live set rather than the empty one it had. Turns from every other session
are unaffected. Running it from setup, before Hermes has used Asphodel,
avoids it.

It stops at the first daemon error and names the session and time it
stopped at. The daemon dedupes turns, so rerunning the same command
resumes, and a finished run rerun reports every turn under `duplicates`.
Posting takes minutes. Extraction is the long part, especially under
subscription windows (`docs/models.md`). The queue is durable, and
`asphodel status` shows how far it has got.

### Plugin prefetch contract

The plugin sends `POST /v1/banks/{bank}/prefetch` with `session_id` and
`query` (the current Hermes message). Optional context fields are:

- `previous_query`, the session's last successful prefetch query.
- `previous_reply`, the first 300 Unicode characters of the assistant reply
  received by `sync_turn` for exactly the selected `previous_query`. It is
  omitted when empty, not yet synced, or ambiguously attributed. The plugin
  keeps only this prefix, even with ingestion disabled or a failed turn delivery.
- `block_id`, a prompt block fetched before the session id was known,
  sent until a prefetch succeeds.

With the default `rerank_query = "conversation"`, the daemon reranks
against the current message plus bounded starts of the previous query
and reply. It trims context and cuts each start to `RERANK_CONTEXT_CHARS`
(300), cutting back to whitespace where possible.

Hermes freezes the prompt block for a session and rebuilds it only on
compaction, so a long-lived chat's agenda goes stale. When the session
holds a block, prefetch's `text` can start with an `Agenda update for
<date>` section ahead of the `Recalled …` injection. It appears when the
bank-local day has moved past the day the block was built for, or when
the agenda lists items the session hasn't seen. It lists only those
items, and counts any that don't fit for a later turn, with the whole
section (header and count included) within `agenda.update_budget` tokens.
The default is 200 and the minimum is 40. The first item is always
listed, so each update makes progress. When that item doesn't fit whole,
its sentence is shortened, at a word boundary where one keeps enough of
it and mid-word otherwise, and ends in "…". Its annotations, such as
`[upcoming Sat 3 Oct]`, stay. If they leave too little of the sentence,
only the date is kept, then nothing. On a later day with nothing new, the
update carries the date alone. Like the injection, it joins the session's in-context set only
when a `sync_turn` echoes the prefetch's `recall_id`, so it's sent once.
A session with no block mapping never gets one.

Query and reply context are isolated by session, including an explicit
`session_id` on a delayed `sync_turn`. An ordinary session switch retains
each session's context for a later return. Compression, reset and rewind
drop the old session's query and reply along with its pending recalls;
a successful `memory_forget` drops the current session's query and reply.
Hermes syncs on a background worker, so a prefetch that arrives before
the preceding turn syncs cannot include that turn's reply yet. It proceeds
without waiting and never substitutes a reply from an older query.

`sync_turn` provides session and user text, but no originating turn ID.
The plugin therefore suppresses reply context for repeated query text within
the same session for the provider's lifetime, including repeats across clears
and failed prefetch attempts. It retains query digests across clears so a
delayed pre-clear sync cannot restore reply context for a repeated query.
Distinct new queries remain eligible. A sync without a matching successful
prefetch contributes no reply context; turn ingestion is unaffected.

## Commands

Every command below except `serve`, `restore`, `models fetch` and `llm
login` talks to the daemon. They take `--url` (or `ASPHODEL_URL`) and read
`ASPHODEL_TOKEN`, and `--json` prints the daemon's reply as it came.

### Banks

- `asphodel bank create <bank> [--owner-name N] [--owner-id P:ID]...
  [--assistant-name N] [--timezone TZ]` creates a bank, or merges the given
  fields into one that exists. The plugin's `initialize` does the same, so
  this is only needed before Hermes first runs or for a bank Hermes doesn't
  use.
- `asphodel bank config <bank> ...` takes the same fields. Fields not given
  are left as they are, and a new name adds an alias without removing the
  old one.
- `asphodel bank list` lists every bank with its memories by status
  (live, superseded, ended, retracted, forgetting), how many are kept, its
  turns and documents, and its queued and failed chunks.
- `asphodel bank delete <bank> --confirm <bank>`: see "Deleting a bank".

### Documents

`asphodel ingest <file> --bank B --date YYYY-MM-DD [--inexact] [--id ID]`
ingests plain text or markdown. `--date` is the reference date relative
times resolve against, and `--inexact` says it's approximate. The id
defaults to the file's name. Ingesting the same id again with new text is
an edit, even with an earlier date. A document never makes a memory kept,
whatever it says.

- `asphodel document list --bank B [--id ID]` lists document versions,
  newest first, each with its chunks, the memories resting on it, and
  anything queued or failed.
- `asphodel document show --bank B <source>` shows one version: its other
  versions, and each chunk with where it is in extraction and the memories
  resting on it or mentioned in it.
- `asphodel document remove --bank B <id>`: see "Removing a document".

### Memories

- `asphodel memory list --bank B [--status S] [--kind K] [--search WORDS]
  [--entity E] [--significance L] [--document ID] [--session S]
  [--fading faded|week|month|never] [--sort created|fade|strength]
  [--limit N] [--cursor C]` lists memories with their status, strength
  and when each fades if it isn't used again. Listing reads only: unlike
  recall, it writes no recall row and never counts as using a memory.
- `asphodel recall --bank B <query> [--from T] [--to T] [--on happened|said]
  [--phase upcoming|past|current|any] [--kind K]... [--entity E]
  [--limit N]` is the same recall the agent's `memory_recall` tool runs.
  It's how you find memory ids.
- `asphodel keep --bank B <id>...` sets the owner's significance to kept,
  so the memory never fades. `asphodel unkeep` hands it back to the level
  extraction gave.
- `asphodel memory significance --bank B <id> <level|clear>` sets the
  owner's significance to `trivial`, `minor`, `notable`, `major`,
  `critical` or `kept`, or clears it. It's the field keep and unkeep write.
- `asphodel memory show --bank B <id>` answers "why do you think X?": the
  sentence, kind, window and phase, both significance fields, the source
  passage or why it's gone, the access and edit logs, the supersession
  chain, the secret-scan kinds, strength in its parts, any guard holding
  back a purge, and projected fade and purge dates. The dates are bank-time
  durations plus the earliest world date at full speed.
- `asphodel memory translate --bank B <id>` translates a memory into
  `[llm] language`. See below.
- `asphodel memory retract --bank B <id>`: see "Retracting a memory".
- `asphodel forget --bank B <id>...`: see "Forgetting".

A memory's sentence, kind and window are never edited from the CLI. Those
change through conversation.

Translation is the one exception, and even it doesn't edit a sentence.
`[llm] language` only shapes new extraction, so memories stored before it
was set stay in the language they were said in, and the English-only models
can't match an English query to them. `memory translate` asks the LLM for
the memory's sentence in `[llm] language` and writes the result as a new
memory that supersedes the old one, the way a refinement does. The new one
keeps everything else: kind, window, both significance fields, the source
passage it points at, and its entity links. It's in the same chain, so it
inherits every access and its strength is unchanged. It's embedded and
indexed under the new sentence, the supersession is logged as
`memory_refined` with `translated_to`, and mental models citing the old
memory move to the new one and refresh. The LLM sees only the sentence,
never the passage.

It's a daemon operation on one memory, so find the ids first, with recall
or a query over a backup copy. It's refused when `[llm] language` is unset,
since there's nothing to translate into, and with 409 when the memory has
been superseded, naming the memory that superseded it. That covers a
repeat: once translated, the old id is superseded by the translation, and
naming the translation answers "already in" the language without asking the
LLM. A memory the LLM hands back unchanged is already in the language, and
nothing is written. The memory is checked again when the translation is
written, so one that extraction refined while the LLM answered is left to
its new version; translate that one if it needs it.

### Entities

An entity is named by its id, `user`, `assistant`, or a name or alias only
one entity in the bank has.

- `asphodel entity show --bank B <entity>`: aliases, merges, linked
  memories and edits.
- `asphodel entity merge --bank B <from> <into>` moves `from`'s aliases and
  links to `into` and keeps `from` as merged. `user` and `assistant` can
  only be `into`. It prints an edit id.
- `asphodel entity unmerge --bank B <edit>` undoes that merge, provided
  `into` hasn't been merged again since.
- `asphodel entity alias rm --bank B <entity> <alias> [--relink-to E]`
  removes a wrong alias. With `--relink-to`, the links that named the
  entity by that alias move to `E`, which gets the alias.
- `asphodel entity link --bank B <memory> <entity>` and `entity unlink`
  edit a memory's entity links.

### Mental models

Only the owner defines mental models, through these commands or the API.

- `asphodel model create --bank B <name> --question Q --max-tokens N
  [--kind K]... [--entity E] [--min-volatility V] [--disabled]`
- `asphodel model list --bank B`
- `asphodel model edit --bank B <name> [--question Q] [--max-tokens N]
  [--kind K... | --all-kinds] [--min-volatility V|none] [--enable |
  --disable]`
- `asphodel model refresh --bank B <name> [--force]` refreshes now. It's
  skipped when the inputs haven't changed, unless `--force`.
- `asphodel model show --bank B <name> [--entry ID]` answers "why does the
  profile say X?": each entry with the memories it cites, and whether the
  prompt block shows it.

### Health and failures

`asphodel status` shows queue depth, failed chunks, failed refreshes, the
purge pause and both fingerprint hashes, the last sweep, the pre-migration
copy, banks recorded under an embedding model the daemon doesn't carry, and
when
`POST /v1/backup` last completed. That last one says nothing about whether
the stream reached its destination. It exits non-zero when anything needs
attention, which is what to alert on; there's no Prometheus endpoint.

```sh
kubectl exec hermes-0 -c asphodel -- asphodel status
```

`asphodel chunks --bank B [--failed [--retry]]` lists a bank's extraction
chunks. `--failed` lists only those whose extraction failed past the retry
cap, with the error kind and HTTP status (never the response), and
`--retry` puts them back on the queue. The nightly sweep deletes failed
chunks past the 90-day horizon.

### Audit lists

`asphodel purges`, `forgets`, `sweeps` and `recalls`, each with `--bank B
[--limit N]`, list newest first, and the daemon caps the limit at 1000.
Purge and forget rows hold ids, times, chunks and spans, never content. A
sweep row holds counts, plus the fingerprint and δ it ran under, and is kept
indefinitely. `recalls` shows each recall's query, which is content, so
treat its output like the store. For a prefetch the query is the cleaned
message, and `raw_query` is the message as Hermes sent it, which the text
output prints on a `raw:` line when cleaning changed it. The sweep clears
both.

### Models

`asphodel models fetch [--model-dir D]` fills the model dir from the
manifest, for hosts outside the image. It skips files already present with
the right checksum and resumes after a failure (`docs/models.md`).

### Replay

`asphodel import`, `replay`, `report` and `bench` belong to the replay
harness, which runs on its own store under `ASPHODEL_REPLAY_DIR` and never
touches a `serve` data dir. `docs/replay.md` covers them, and
`docs/hermes-data-evaluation.md` walks through an evaluation on real
history.

## Backup

`asphodel backup --out <file|->` asks the daemon for `POST /v1/backup`,
which takes an online copy with SQLite's backup API, checks it with
`PRAGMA integrity_check`, and streams it with its SHA-256 and length in
headers. The client checks both. To a file, it also runs its own integrity
check, and the file only appears under its final name once everything
passed. `-` writes to stdout.

Asphodel has no destination, schedule or retention of its own.
`deploy/kubernetes/backup.yaml` is a nightly CronJob. The image has no
shell, so the job takes the copy in two steps: an init container runs
`asphodel backup --out /backup/asphodel.db` against the `asphodel`
Service, and an rclone container uploads the checked file under a
timestamped name. Retention is the bucket's lifecycle rule.

From a host with Bash, the stream can go straight to its destination:

```bash
set -o pipefail
kubectl exec hermes-0 -c asphodel -- asphodel backup --out - \
  | rclone rcat "remote:my-bucket/asphodel/asphodel-$(date -u +%Y%m%dT%H%M%SZ).db"
```

`pipefail` matters. `asphodel backup` checks the length and hash only
once the whole stream has been written, and rclone uploads whatever it
received and exits 0. Without `pipefail` the pipeline's status is rclone's,
so a failed check would still count as a good backup to whatever scheduled
it. With it, the pipeline fails, but the bad object is already in the
bucket and has to be deleted. Writing to a file first, as the CronJob does,
means nothing is uploaded unless the copy passed.

Hermes' own backup doesn't cover Asphodel: the plugin's `backup_paths()` is
empty, because the data lives in the sidecar.

## Restore

Restore is offline. `asphodel restore <file> --data-dir <dir>` takes the
data-dir lock, checks the backup's integrity and that its schema version
isn't newer than the binary, copies the backup in beside the live
database with a `restored` edit row (the backup time, the restore time and
the binary version), and only then installs it: it renames the current
database (and its `-wal` and `-shm`) to `asphodel.db.before-restore-<time>`,
renames the copy to `asphodel.db`, and syncs the directory. Nothing is
deleted.

What a failure leaves behind depends on when it happened:

- **Before installing.** A held lock, a backup that fails its integrity or
  schema check, or an error while staging the copy (including `syncing
  .../asphodel.db.restore.part-...`) leaves the live store untouched.
- **While installing.** An error that names a move (`moving ... to ...`)
  or the sync of the data dir itself (`syncing /data`) can leave it with the
  old store, the restored one, or files part-way between them. A failed
  rename is rolled back on a best-effort basis that ignores its own
  errors, and a failed sync comes after the restored copy is already in
  place. Inspect the data dir before going on.

The lock is taken without waiting: if a daemon still holds it, restore
fails at once rather than queueing behind it. In Kubernetes:

1. Scale the StatefulSet to zero, and wait for the pod to be gone. Scaling
   returns at once, while the old pod can take up to its 120 s grace period
   to drain, and on the same node a `ReadWriteOnce` volume can be mounted
   by both pods.

   ```sh
   kubectl scale statefulset hermes --replicas=0
   kubectl wait --for=delete pod/hermes-0 --timeout=5m
   ```

   That stops Hermes too, so no turns arrive meanwhile.
2. Set the backup's name in `deploy/kubernetes/restore.yaml` and apply it.
   Its init container fetches the backup with the same rclone secret the
   backup job uses, and its main container runs `asphodel restore`.
3. Wait for the pod to succeed. The pod never restarts, so it ends
   `Succeeded` only when both containers exited 0, and `Failed` otherwise.

   ```sh
   kubectl apply -f deploy/kubernetes/restore.yaml
   kubectl wait --for=jsonpath='{.status.phase}'=Succeeded \
     pod/asphodel-restore --timeout=15m
   kubectl logs asphodel-restore -c restore
   ```

   The log names the schema version and where the old database was moved.
   If the wait doesn't succeed, keep the StatefulSet at zero and find out
   why before anything else:

   - A timeout says nothing about the restore. The pod may still be
     running. Check `kubectl get pod asphodel-restore` and wait for it to
     end.
   - If `fetch` failed, `asphodel restore` never ran and the live store is
     untouched. Fix the fetch and run the pod again.
   - If `restore` failed, read `kubectl logs asphodel-restore -c restore`.
     An error from before installing (see above) left the live store
     untouched. An error that names a move or the data dir's sync means
     you need to look at the data dir.

   The image has no shell, so look with a throwaway pod that mounts the
   claim:

   ```sh
   kubectl run asphodel-inspect --rm -it --restart=Never --image=busybox \
     --overrides='{"spec":{"containers":[{"name":"asphodel-inspect",
       "image":"busybox","stdin":true,"tty":true,"command":["sh"],
       "volumeMounts":[{"name":"data","mountPath":"/data"}]}],
       "volumes":[{"name":"data","persistentVolumeClaim":
       {"claimName":"asphodel-data-hermes-0"}}]}}'
   ls -la /data
   ```

   `asphodel.db.before-restore-<time>` (with `-wal` and `-shm` when it
   had them) is the store from before the restore, and
   `asphodel.db.restore.part-*` is a staged copy that was never installed.

   - If `asphodel.db` exists, running the restore again is safe: it moves
     `asphodel.db` and its `-wal` and `-shm` aside under a new name rather
     than overwriting them.
   - If `asphodel.db` is missing but `asphodel.db-wal` or `asphodel.db-shm`
     is still there, they belong to the old store. Rename them beside its
     `before-restore` file (`asphodel.db.before-restore-<time>-wal`, and
     the same for `-shm`) before running the restore again, so SQLite never
     reads them with another database.
   - To go back to the old store instead, rename the `before-restore` files
     to `asphodel.db`, `asphodel.db-wal` and `asphodel.db-shm`.

   Only scale back up once you know which store `asphodel.db` is.
4. Delete the pod, scale back to one replica, wait for it to be ready, and
   run `asphodel status`.

   ```sh
   kubectl delete pod asphodel-restore
   kubectl scale statefulset hermes --replicas=1
   kubectl rollout status statefulset hermes
   kubectl exec hermes-0 -c asphodel -- asphodel status
   ```

Then check what the restore changed:

- Turns between the backup and the restore are lost. Hermes still has them
  in its own transcript, but the plugin doesn't send them again.
- Memories forgotten after the backup was taken are back. Forget them
  again.
- If the restored store's deletion fingerprint differs from the binary's,
  purge and the sweep pause. `asphodel status` shows it, and "Purge pauses"
  below says what to do.
- A backup from an older schema migrates on the first start, as an upgrade
  does. `docs/upgrading.md` says whether that needs anything from you.

## Upgrading

Upgrade by deploying the new image. Before a schema migration the daemon
copies the database into the data dir, keyed by the schema version it came
from. A copy for the same version is never overwritten, so a migration
that crash-loops can't replace the clean copy with a damaged one. The copy
is deleted 7 days after the migration completes, and `asphodel status`
shows it until then. `docs/upgrading.md` lists what each migration needs
from you.

## Purge pauses

The store keeps a fingerprint of every setting that decides an
irreversible deletion: the fixed strength constants, `clock.quiet_rate`,
`strength.significance`, `purge.delta`, `agenda.overdue_days` and
`purge.source_horizon_days`. A significance change alters every memory's
strength at once, so it pauses purge like the others, and nothing purges
under the new values until you acknowledge them.
When the daemon starts with a fingerprint that differs from the stored one,
purge and the sweep of sources and recall rows pause. Forget never pauses.

1. `asphodel purge plan` shows which values changed and how many memories,
   sources and rows the sweep would delete now. It deletes nothing and can
   run at any time.
2. `asphodel purge ack --hash <h>` acknowledges the change. The hash has to
   be the one the running daemon computed, as `purge plan` shows it. The
   ack is stored as an edit row, so it survives restarts and travels with a
   restore. Purging resumes at the next sweep.

Running the plan first is recommended, but nothing enforces it.

## Forgetting

The owner forgets through the agent's `memory_forget` tool, and the
operator with `asphodel forget --bank B <id>...`. Forget is irreversible
and erases a memory, every earlier and later version of it, and the
passages they came from.

The erase happens in two parts. At once, the chain is hidden from recall,
injection, the agenda, mental model refreshes and `used` credit; model
entries citing it are dropped; the prompt block cache is cleared; in-context
sets are scrubbed; and recall rows naming it are deleted. Deleting the
rows, redacting the passages and writing the tombstone wait on the bank's
extraction queue, behind the chunks queued before the forget, so those
chunks reconcile against the hidden memory and their passages are erased
with it. `asphodel forgets` lists what was forgotten, by id.

The turn that asks to forget is never stored. The plugin sees the
`memory_forget` call and sends the turn with `forget_requested`, and the
daemon keeps only its key, as a tombstone.

What forget can't reach:

- **Backups.** A backup taken before the forget still holds the content
  until your retention removes it. Restoring one brings the memory back.
- **The pre-migration copy**, for the 7 days it's kept, if the forget came
  after the migration it precedes.
- **Hermes.** Its own transcript, and the context of the session where the
  forget happened, are outside Asphodel.
- **A queued `ends` claim.** A chunk queued before the forget can end the
  forgotten memory with a new one outside its chain, whose sentence may
  restate the forgotten content. The erase clears its `ended_by` and leaves
  it. Find it with `asphodel recall` and forget it too.
- **New input.** Forget blocks the same input, not the same words. A later
  turn that repeats them, or an edited section of a document that still
  contains them, is new input and can bring the memory back. Take the words
  out of a document before sending its next version.
- **Old mentions.** A memory mentioned before schema version 7 has no
  recorded span for the mention, so forget masks more of that turn or
  document than the mention itself (`docs/upgrading.md`, version 8).

## Retracting a memory

`asphodel memory retract --bank B <id>` is the owner saying a memory never
held: a denial, as when the user says so in conversation, with no new memory
replacing it. The memory leaves recall, the agenda, the prompt block, mental
model refreshes and live sessions at once, and whatever it had ended is open
again, since that ending never happened either. It isn't erased: it stays
listed as retracted, keeps its access log, and the sweep purges it once it
fades, like any other memory. There's no undo.

Only the newest version of a memory can be retracted. An older one is
refused with 409, naming the version that replaced it, and so is a repeat.
Use forget instead when the content itself has to go.

## Removing a document

`asphodel document remove --bank B <id>` removes a document by its id:
every version ever ingested under it. It's irreversible.

- Every memory resting on the document's chunks is forgotten, through the
  same path as `asphodel forget`, so every earlier and later version of it
  goes too, even one a later turn refined.
- Chunks still waiting for extraction leave the queue. A chunk already being
  extracted finishes its LLM calls, and its commit is then refused, so
  nothing it found is written.
- Each version's text is cleared at once. The versions keep their keys, so
  sending any of them again is a duplicate and queues nothing.

Memories the document only mentioned again, which rest on other turns or
documents, stay. Everything listed under "What forget can't reach" applies
here too, and a new version of the document with different text is new
input. Over HTTP it's `POST /v1/banks/{bank}/documents/remove` with
`{"document_id": "<id>"}`, the id exactly as ingested; it's never part of
the path, where a client would turn `folder/../notes` into `notes`. The
dashboard asks for confirmation before it calls the route; the CLI and the
route don't.

## Explaining recall and injection

`POST /v1/banks/{bank}/recall/explain` runs one query through recall or
injection and returns the working: every candidate with what the pipeline
computed for it, and why it was or wasn't returned. It needs the bearer
token like the rest of `/v1`. It runs the same code as `/recall` and
`/prefetch`. For the same bank state and query, recall mode returns
the memories `/recall` would, in the same order, and injection mode injects
what a prefetch for a new session would.

Unlike them, it changes nothing. It writes no recall row and no access, and
it neither reads nor changes any session. In injection mode nothing counts
as already in context, so a memory the agent can already see this session
is still shown as injected. It does take the reranker like any recall, so a
prefetch arriving at the same moment can wait for it, and on a slow
reranker can miss its deadline.

The body is tagged by `mode`:

- `{"mode": "recall", "query": "...", ...}` takes the `/recall` filters:
  `from`, `to`, `on`, `phase`, `kinds`, `entity` and `limit`.
- `{"mode": "injection", "query": "...", ...}` takes the message and,
  optionally, `previous_query` and `previous_reply`, as the plugin sends them
  to `/prefetch`.

Neither takes a session; a `session_id` is ignored. The reply has:

- `mode`, `query` (what the retrievers searched, after cleaning and a short
  follow-up's borrowing), `rerank_query`, and `reranked`, false when the
  reranker missed its deadline.
- `latency`: `embed_ms`, `retrieve_ms`, `rerank_ms` and `total_ms`. The
  reranker's deadline runs from the start of the request, so `rerank_ms` is
  at most what was left of it.
- `candidates`, the reranked candidates in their final order, then the
  candidates fused past the top 40, which the reranker never sees, in fusion
  order. Each has its
  `id`, `sentence`, `kind` and `phase`; `arms`, the retrievers that found it
  (`vector`, `bm25`, `entity`) with its `rank` in each; `rrf_rank`; the raw
  reranker `logit`; `score`, with `relevance`, `w_s`, `strength_term`,
  `confidence_term`, `phase_term` and `total`; the `strength` band; `kept`;
  `included`; and `reason`, null when it was included.
- In injection mode, `candidates` ends with the memories the retrievers
  found below the recall threshold. These never reached fusion, so their
  ranks, logit and score are null.
- `injection`, null in recall mode: the `text` exactly as the agent would get
  it, its `tokens`, the `injected` ids, and the `floor`, `cap` and
  `token_budget` it was gated with.

A candidate left out gives one of these reasons:

| `reason` | Mode | Meaning |
|---|---|---|
| `over_limit` | recall | Ranked past `limit` |
| `outside_rerank_pool` | both | Fused past the top 40, so never reranked; it keeps its arm ranks and `rrf_rank`, with a null logit and score |
| `below_tau` | injection | Strength below the recall threshold |
| `not_reranked` | injection | The reranker missed its deadline, so nothing is injected |
| `under_floor` | injection | Its logit is under the loaded reranker's floor |
| `over_cap` | injection | `injection.cap` memories were already taken |
| `over_budget` | injection | Its line would take the block past `injection.token_budget` |

Recall mode doesn't list memories its filters left out. A memory that isn't
in any retriever's top 100 isn't a candidate at all, so it isn't listed
either.

## The dashboard

The daemon serves a dashboard at `/dashboard` for browsing banks, documents
and memories, and for keeping, retracting and forgetting them and removing
documents. The page itself needs no token, since it holds nothing. It asks
for the token and sends it on every `/v1` call it makes, which need it like
any other client. Browsing never goes through recall, so looking at a
memory doesn't strengthen it or log a recall. The Recall page doesn't
either; see below.

It also shows each bank's counts, the `attention` lines from `status` as a
banner, failed chunks with retry, and the purge pause with its
acknowledgement. Retract, forget, document removal and the purge
acknowledgement each ask first and say what will go.

A daemon on loopback without a token is browsed without asking. Otherwise
the dashboard asks once per browser tab, keeps the token in that tab's
session storage, and asks again if the daemon rejects it. "Sign out" drops
it. The page sends a Content-Security-Policy that allows nothing from
outside the daemon.

The dashboard is plain ES modules and CSS under
`crates/asphodel/assets/dashboard/`, embedded in the binary as they are:
there is no build step and nothing to install. Its serif is Source Serif 4,
Adobe's woff2 release files unmodified, served from the daemon under the SIL
Open Font License (`OFL.txt` beside them, also at `/dashboard/OFL.txt`). Node runs its tests only:
`cd tests/dashboard && nix develop ../.. -c sh -c 'npm ci && npm test'`.

### The Recall page

Each bank has a Recall tab, `#/banks/{bank}/recall`, for testing a query
and answering "why didn't memory X come back?". It runs the query through
recall, as the `memory_recall` tool would, or through injection, as
prefetch would, by calling `POST /v1/banks/{bank}/recall/explain`. Explain
runs the same pipeline code as recall and prefetch, but it writes no recall
row and no access and doesn't read or write session state. A test query
never shows up in the recall log, never strengthens a memory, and never
holds or changes an injection. Since there's no session, nothing counts as
already in context in injection mode. A memory the agent can already see in
a live session is injected here but wouldn't be injected there.

Recall mode takes the recall tool's filters: dates, phase, kinds, entity
and limit. Injection mode takes the user's message and, optionally, the
previous message and the agent's reply, which a short follow-up borrows
from. It applies τ, the reranker's floor, `injection.cap` and
`injection.token_budget`, and shows the injected text exactly as the agent
gets it, with its token count.

For each candidate, the page shows the search arms that found it and its
rank in each, its fused (RRF) rank, the reranker logit, the score and its
parts (relevance, w_s × strength, state confidence, phase), its strength
band and whether it's kept. Each row links to the memory's page. What made
the cut comes first. What didn't is greyed out underneath with the reason:

- both modes: fused too far down for the reranker to see it. Only the top
  of the fused list is reranked; the rest is listed in fusion order with
  its arm and fused ranks, and no logit or score
- recall: ranked past the limit
- injection: strength below τ, a logit under the floor, over the cap, over
  the token budget, or a reranker that missed its deadline, in which case
  nothing passes

Each search arm keeps only its top hits, and recall mode doesn't list what
its filters dropped. So a memory missing from both lists either wasn't in
any arm's top hits or was filtered out. The page also
shows how long embedding, retrieval, reranking and the whole run took, so a
slow reranker is visible.

**Dates.** From and To are calendar days in the bank's timezone. The banks
page and the hint under the fields both show it. From starts at the first
instant of its day: midnight, or the end of a clock change that skips
midnight. To runs to the last instant of its day, so From and To on the
same date cover that one day. Either end can be left open. The page sends
the days as RFC 3339 instants in UTC, the same `from` and `to` that
`/recall` and `asphodel recall` take, and lists the instants it sent above
the results. "Dates match" is `on`. "When it happened" compares the range
with a memory's validity window, widening a low-confidence window by one
unit at each end. A fact with no end matches only if its stated start falls
in the range. "When it was said" compares the range with `observed_at`. The
browser converts days with its own timezone database, so a zone whose rules
changed recently can be off by the change until the browser updates.

## Deleting a bank

`asphodel bank delete <bank> --confirm <bank>` erases everything in the bank
through the same path forget uses, removes its tombstones, edit rows and
session mappings, and writes a daemon-wide `bank_deleted` row with counts.

**Disable the plugin first.** Every plugin `initialize` calls
`PUT /v1/banks/{bank}`, which creates the bank again, empty, and `ingest:
false` doesn't stop that. Set `memory.provider` in the profile's
`config.yaml` to something other than `asphodel`, restart every Hermes
process that uses the profile (the gateway, and any CLI or TUI), and then
delete the bank. Hermes' built-in memory stays off until you turn it back
on.

## Changing the embedding model

A bank records the embedding and reranker models it was created under, and
is served with its recorded embedding model until it's re-embedded. A bank
recorded under a model the daemon doesn't carry is refused: recall and
prefetch answer 503, and its chunks wait on the queue. So the image has to
carry both embedding models for as long as any bank is still on the old
one (`docs/models.md`, "Changing the embedding model").

**Today's daemon can't do this with the real models yet.** Serving two
embedding models works only on the fakes (`ASPHODEL_MODELS=fake-v2`). For
the ONNX models, `Models::load` reads the manifest's first entry as the
embedder and its second as the reranker, and `serve` registers no previous
embedder. Adding a model to the manifest and the image isn't enough on its
own. Before the first release that changes the embedding model, the code
has to:

- tell the manifest's current embedder, previous embedder and reranker
  apart, rather than going by position;
- load the previous embedder from the model dir and register it with
  `Service::with_previous_embedder`, as the `fake-v2` path does;
- keep failing fast when any of the three is missing or corrupt.

With that in place, a change goes like this:

1. **Build an image with both models.** The release adds the new model to
   the manifest and keeps the old one as the previous embedder.
   `nix/models.nix` lists every model the manifest does, and the image
   build fails until it matches.
2. **Calibrate the new floor.** Set the new model's
   `reconcile.embedding_floors` entry from replay before the deploy. The
   daemon won't start without it. Keep the old model's entry as well,
   because banks still on it reconcile against it until they move.
3. **Deploy.** Banks keep serving with their recorded model. New banks get
   the new one.
4. **Re-embed each bank.** `asphodel reembed --bank B` starts a daemon job
   that embeds the bank's memories with the new model into a side table,
   then swaps them in with one transaction and records the new model.
   Extraction carries on with the old model until the swap. The command
   follows the job to the swap; `--no-wait` returns at once, and running it
   again shows where the job stands. A restart resumes the job rather than
   starting over.
5. **Drop the old model** in a later release, once every bank has moved.
   `asphodel bank config <bank>` with no fields changes nothing and prints
   the model a bank is recorded under, and `asphodel reembed --bank B` says
   when a bank is already on the daemon's model.

A bank already refused because the daemon dropped its model recovers the
same way: `asphodel reembed --bank B` needs only the daemon's model.

A reranker-only change needs only its new `injection.reranker_floors` and
`ranking.relevance_scales` entries. Reranker scores aren't stored, so
there's nothing to re-embed.
