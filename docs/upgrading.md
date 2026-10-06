# Upgrading

## Shorter trivial and minor lifetimes

The significance mapping changes from 0.1 to 0.0 for trivial memories and
from 0.3 to 0.2 for minor memories. Notable, major, critical and kept
memories are unchanged. With one mention and the default strength tuning,
a trivial memory now fades after 7.4 bank days and becomes eligible for
purge after 4.2 bank months; a minor memory fades after about 1 bank month
and becomes eligible for purge after about 1.5 bank years. Reinforcement
and purge guards still apply. These are bank-time ages, not wall-clock
deadlines.

Stored memories keep their significance labels and access histories. The
new mapping applies to existing memories as soon as the upgraded daemon
computes their strength, so some trivial and minor memories may immediately
fall below the recall threshold or become eligible for the next purge
sweep. No schema migration or re-extraction is needed.

For a before/after evaluation, use scripted scenarios without LLM calls.
A real-history run in strict `replay` mode can miss the cassette because
strength affects injection and injected memories are part of the recorded
request. `fast` mode reuses recorded claims but may make top-up LLM calls;
get approval before using it on private data.

## A refinement across kinds is rejected

Code now rejects a `refines`, call 2's or a repeat promoted for a date or
for significance, between memories of different kinds, except a task or
recurring claim on a task (`reconcile.kind_guard`, on by default). The claim
becomes a memory of its own and the neighbour is left as it was, with no
access or edit. A weightier repeat of another kind, such as a fact naming a
relationship labelled a repeat of an event, is therefore a second head
beside the event rather than its replacement.

Stored memories and chains are unchanged and nothing is re-extracted. No
schema migration is needed, purge doesn't pause, and call 2's prompt is
unchanged, so recorded replies still answer the same requests. Set
`kind_guard = false` under `[reconcile]` to compare the old behaviour.

## A weightier repeat becomes the head

A newer claim that call 2 labels `mentioned_again` or `confirmed` now
becomes a new chain head refining the memory, rather than an access on it,
when it's at least `reconcile.promotion_gap` significance levels (default
2) above the memory. It keeps the memory's dates when it states none of its
own. Below that gap, a mention still raises the memory's significance.

Stored memories are unchanged and nothing is re-extracted: claims absorbed
before the upgrade stay absorbed. No schema migration is needed and purge
doesn't pause. Call 2's prompt is unchanged (`reconcile_claims` v3), so
recorded call 2 replies still answer the same requests; a store that
diverges from the recording still tops up the calls it no longer matches.

## Schema version 2: send each bank's config again

Schema version 2 adds `speaker_ids`, the only mapping a turn's speaker is
resolved through.
The migration from version 1 fails closed. It carries over the speakers
ingest created, whose platform ids version 1 recorded, but **not the owner's
platform ids**. Version 1 kept those only as aliases of the `user` entity,
next to the owner's current and former names, so a former name that looks
like a platform id can't be told from a real id, and guessing could make a
stranger the owner.

Until a bank's config is sent again, a turn from the owner's platform
account is attributed to a stranger entity, not to `user`. Sending the
config maps the owner's ids to `user`, taking them over from any entity
that held them, and logs a `speaker_id_set` edit.

After upgrading the daemon, send the config of every bank that has owner
platform ids, before Hermes ingests more turns:

- Restart Hermes (or each gateway and CLI process). The plugin's
  `initialize` calls `PUT /v1/banks/{bank}` with the owner's platform ids
  from its config. Plugin instances that were already running don't call it
  again on their own.
- Or call `PUT /v1/banks/{bank}` with `owner_platform_ids` yourself, or
  `asphodel bank config`.

The daemon logs a warning when it migrates a store from version 1, as a
reminder.

## Schema version 3: aliases and entity names in NFC

Extraction searches the alias index with its passages in Unicode NFC, so
aliases have to be stored composed too. Outside Latin script the alias
index doesn't fold a combining accent away, and an alias stored decomposed
(a letter followed by a combining accent) would never be found. From version 3, every alias and
entity name is written in NFC.

The migration composes what earlier versions stored. No action is needed.

- Every alias and entity name is rewritten in NFC. The alias index is
  updated row by row as they change.
- Two aliases of one entity that are canonically the same, such as the
  composed and decomposed spellings of "Νίκος", become one row. The oldest
  stays.
- Each `alias_added` edit that named a removed row is repointed to the row
  that stayed, so every edit still names an alias of its entity. Undoing any
  of them removes that alias, which is the one each of them added.

As with every migration, the daemon copies the database before it runs.

## Schema version 4: a document's access has its own key

Reconciliation writes an access when a document restates something
already remembered. A document carries the
turn number of the turn before it, and version 1 allowed one access per
memory per turn, so a second document ingested with no turn in between
couldn't record its mention of a memory the first one created. From
version 4 the key includes the source. Turns are unaffected: each turn is
one source with its own number, and extraction still keeps one access per
memory per turn, the strongest.

The migration rebuilds the `accesses` table with every row and id as they
were. No action is needed.

## Schema version 5: a queued turn keeps its in-context set

Extraction credits a turn with `used` accesses on the memories the agent
could see when it wrote the reply. Before version 5 the worker read the
session's in-context set when it reached the turn, so a session cleared
on compaction, a later recall, or a restart in between changed the credit. From version 5, ingest stores the set with the turn
in `turn_in_context`, and the row goes once the turn is extracted.

The migration only adds the table. Turns already queued when you upgrade
have no stored set and are extracted as if nothing was in context, which
is what a restart did to them before. No action is needed.

## Schema version 7: where a mention was said

Forget erases a memory's chain behind the chunks queued before it, and
those chunks reconcile against the hidden memory. A claim that
only mentions or confirms the memory leaves an access, and before version
7 an access recorded only its source, so the erase couldn't tell which part
of the turn or document to redact. From version 7, `accesses.spans` holds
each mention's chunk and character span, and forget masks exactly those.

The migration only adds the column. Accesses written before it have no
spans. Version 8 says what forget does about them.

The first start after the upgrade also records the deletion inputs beside
the stored fingerprint, so `asphodel purge plan` can name what changes
later. A store that's paused when it upgrades shows `unknown` until the
pause is acknowledged.

## Schema version 8: mention passages, and what a forget masked

Version 7 kept a mention's span on its access, but a later version of the
same document repeating a memory isn't an access, so its passage
went unrecorded and survived a forget. From version 8,
`mention_passages` records every passage that restated a memory without
becoming one, credited or not, and forget masks them. The migration copies
the spans version 7 stored, and `accesses.spans` is no longer written.

`chunk_redactions` records what forget masked in each chunk. A document's
new version stores its full text, but a section it shares with an earlier
version gets no chunk row, so forget also masks each passage wherever it
appears verbatim in the document's other versions, and a version ingested
after the forget masks the recorded characters before its text is stored.

What forget blocks is the same input, not the same words. A
section a later version leaves unchanged is masked as above. A section the
owner edits is new input: if it still contains the forgotten words, it's
stored and extracted like any other new text, and can bring the memory
back. A later turn that repeats the words is new input too. To keep
something forgotten, take it out of the document before sending the next
version.

**Mentions from before version 7 have no span**, and nothing can recover
one: call 1's reply was dropped when each chunk committed. Forget masks
more rather than less for them:

- in a turn, the whole message and reply, except the passages surviving
  memories rest on;
- in a document, the forgotten memory's own passages wherever they appear
  verbatim, or the whole document, except surviving passages, when they
  appear nowhere.

The `forgotten` edit counts these as `legacy_mentions` and `whole_source`,
so the wider masking can be seen in the edit log. No action is needed, but
forgetting a memory mentioned before version 7 can mask more of an old turn
or document than the mention itself.

## Schema version 9: a sweep's counts survive its failure

The nightly sweep writes one run row per bank, with counts only.
Each purge and the source sweep commit on their own before the row is
written, so before version 9 a sweep that failed after deleting something,
and was run again, wrote a row that left out what the failed attempt
deleted. From version 9, `sweep_progress` holds a bank's counts while its
sweep runs: each deletion adds to it in its own transaction, and the row is
written from it at the end. A sweep resumed after a failure or a restart
keeps counting into the same progress, so its run row counts every attempt
and keeps the first attempt's start time.

The migration only adds the table. No action is needed.

## Schema version 11: a prefetch logs its raw query

Prefetch now recalls for the user's message without the note Hermes'
Discord gateway puts in front of it (`[Triggering message id: …]`) and
without the `[Name] ` speaker prefix of a shared thread. `recalls.query`
holds that cleaned query, and the new `recalls.raw_query` holds the message
as it was sent. The sweep clears both at the 90-day horizon.

The migration only adds the column. Recalls logged before it, and recall
tool and refresh rows, have no raw query. No action is needed, but a
reranker floor calibrated on recalls from before version 11 was calibrated
on uncleaned queries.

## Schema version 12: removed documents

`asphodel document remove` and `POST /v1/banks/{bank}/documents/remove`
clear a document's text and mark each of its versions with the new
`sources.removed_at`. The migration only adds the column. No action is
needed.

## Schema version 13: mental models in sections

A refresh now plans the model's question into facets, recalls each one,
and writes the whole summary as sections of cited sentences
(`docs/operations.md`, "Mental models"). The migration adds
`mental_models.plan` and `mental_model_entries.section`. Existing entries
have no section and render as lines, as before, until their model's next
refresh.

The refresh fingerprint now includes the plan, so every model's next
refresh writes again: one `write_model` call per model, plus one
`plan_model` call for a model whose question isn't the seeded profile's.
Those replace the `refresh_model` v2 call. Each refresh now logs one
`refresh` recall row per facet, five for the seeded profile.

The `[mental_models]` defaults change: `input_budget` 60 to 90 and
`input_budget_with_cited` 70 to 100, with the new `max_facets` (6) and
`facet_budget` (20). A tuning file that sets `input_budget_with_cited`
below 90 without setting `input_budget` no longer validates.

Replay: no recorded `refresh_model` record can stand in for a
`write_model` call, so `--refresh recorded` finds nothing to substitute in
a cassette recorded before this version. The first replay of private
history after upgrading needs live refresh calls, which need their own
authorization.

### Existing banks: opt in to larger summaries

The output defaults also change: `profile_max_tokens` rises from 500 to
2048, and the shared `budget` from 800 to 2560. New banks seed their User
profile with the configured profile cap. Existing model rows keep their
stored `max_tokens`; the schema migration does not resize them. Explicit
tuning overrides still take precedence over the new defaults.

To opt an existing bank into the larger profile, first set the following
in the daemon's tuning file, preserving any other `[mental_models]`
settings, and restart the daemon with that file:

```toml
[mental_models]
budget = 2560
profile_max_tokens = 2048
```

Then update the stored model cap for each bank you choose to migrate:

```sh
asphodel model edit --bank main "User profile" --max-tokens 2048
```

Replace `main` with the bank name. The edit is refused if the enabled
models' caps together exceed the active shared budget, so apply the tuning
first and leave room for any other enabled models. The next
refresh can write to the new cap; editing the cap does not immediately
expand the existing text. Larger summaries can add tokens to every
Hermes turn, and the agenda and other enabled models still share the
2560-token block. Bounded recall and admission filters are unchanged.
These are opt-in operator steps, not automatic production changes or
authorization for private replay or live backend calls.

## Schema version 14: `used` by memory alone

Call 1 used to be shown each sentence of the session's prompt block under
its own handle (`n1`, `n2`, …), and a reply naming one was `used` on every
memory it cited. It now sees only the memories the block cites, each by its
own `m` handle, as before, and credits the one the reply relied on. The
migration drops `turn_entries`, where each queued turn kept the block's
sentences, and `prompt_blocks.entries`. No action is needed.

A turn queued when you upgrade loses its snapshot and is extracted with its
in-context set alone. A chunk whose call 2 failed before the upgrade
resumes from its saved call 1 reply, and an entry handle in that reply
credits nothing.

Replay: `CALL1_VERSION` stays 7. The claim rules haven't changed, so
`fast` keeps reusing recorded claims by chunk, and `used` verdicts are
still keyed by (reply hash, sentence hash). A recorded call 1 reply that
names a handle its record doesn't list, such as an entry's, keeps its
`used` verdicts, and its other pairs are judged again by a top-up. Call 1's
request text has changed, so `replay` mode misses on every call 1 recorded
before this version, as it does after any prompt edit.

## Schema version 15: one answer per mental model

A mental model's answer is now one prose text with one set of citations,
in place of entries that each cited their own memories
(`docs/operations.md`, "Mental models"). The migration adds
`mental_models.answer` and `mental_model_cites`, builds each model's
answer from its entries, grouped by section in position order with
entries from before sections first as a paragraph of their own, takes the
union of their citations, and drops `mental_model_entries` and
`mental_model_citations`. Every model keeps its text and what it cites,
and no refresh is forced; the next one rewrites the answer as connected
prose. No action is needed.

What changes in use:

- A model whose answer cites a memory that's retracted or ended is left
  out of the prompt block whole until its refresh rewrites it, where
  before only that sentence went. Forget blanks the answer of every model
  citing the chain at once, and requests a refresh.
- The block no longer marks a low-confidence state's age after a
  sentence.
- The API's `Model` carries `answer` and `cites` in place of `entries`.
  `GET /v1/banks/{bank}/models/{model}` carries `cited` (each memory with
  its status), `renders` and `shown_answer` (the answer as the block shows
  it now, whole or cut) in place of `entry_views`, and no longer takes
  `?entry=`. `asphodel model show` drops `--entry` and prints the cited
  memories after the answer. A refresh's `applied` detail is `{written,
  rejected, trimmed}`, with `trimmed` a count of sentences.

Replay: `write_model` is now version 2, and its reply is `{"sections":
[{"heading", "text"}], "cites": [...]}`. A `write_model` v1 record,
sentence by sentence, can't stand in for a v2 write, so `--refresh
recorded` finds nothing to substitute in a cassette recorded before this
version, and the first replay of private history after upgrading needs
one re-record of live refresh calls, which needs its own authorization.
`--refresh recorded` now reuses a recorded answer only when every memory
it cites is in the run's input, and otherwise skips the write. `--refresh
off` skips every write rather than answering it with an empty reply.

## Schema version 16: urgent citation repair

The migration adds `mental_models.refresh_urgent`, initially false.
Corrections to cited memories and forget now request urgent repairs that
bypass the 30-minute interval after a successful refresh. Debounce, LLM
holds and retry delays after a failed refresh still apply. The flag is
stored, so an urgent request survives a restart. No action is needed.

## Schema version 17: pending `used` credits

The migration adds an empty `pending_credits` table. Only
`strength.corroborate_used`, off by default, writes to it, so retention and
the deletion fingerprint are unchanged and purge doesn't pause. No action is
needed.

## Write template version 3: dates for possibly stale states

Schema version 15 introduced `write_model` v2 and the one-answer reply
shape. The current template is v3, with the same shape. A state whose
confidence is below 0.9 is now shown to the write with its absolute
observed date in the memory's timezone, and the write is asked to include
that date in its prose. The block no longer adds age annotations itself.
Crossing that threshold changes the refresh fingerprint; another day
below it does not. Existing answers stay as stored until a refresh
rewrites them.

Replay substitutions require the current template version, so neither
v1 nor v2 writes stand in for v3. To carry recorded answers over in
`fast --refresh recorded`, use recordings made with v3 and this run's LLM
model and language, with every cited memory present in the new input.
Re-recording private history still requires authorization under the
evaluation policy.
