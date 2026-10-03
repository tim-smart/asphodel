# Operations are daemon jobs and HTTP clients

Every operator command is an HTTP client of the running daemon, and anything that changes the store at scale runs as a daemon job: backup, re-embedding, bank deletion, merges, and the acknowledgement of a purge pause. There are two exceptions, and both are offline. `asphodel restore` writes the database while holding the data-dir lock, so it can't run alongside the daemon. The daemon itself copies the database before running a schema migration at startup.

## Backup and restore

- **`POST /v1/backup`** runs SQLite's online backup into a temporary file in the data dir, checks the copy with `PRAGMA integrity_check`, and streams it back. The SHA-256 and length go in headers. `asphodel backup --out <file|->` checks the hash, and when the target is a file it runs its own integrity check. Asphodel has no backup destination, schedule or retention of its own. A CronJob or systemd timer pipes the stream wherever it should go.
- **Restore is offline.** `asphodel restore <file> --data-dir <dir>` takes the lock, checks integrity and that the schema version isn't newer than the binary, moves the current database aside and copies the backup in. It writes a `restored` edit row with the backup time, the restore time and the binary version. If the restored store's deletion fingerprint differs from the binary's, purge pauses as ADR 0009 describes.
- **Pre-migration copy.** Before a migration, the daemon takes the same copy into the data dir, keyed by the schema version it came from. A copy for the same from-version is never overwritten, so a migration that crash-loops can't replace the clean copy with a damaged one. The copy is deleted 7 days after the migration that follows it completes.

## Forgetting

- **The erase splits in two.** Everything that can be undone happens as soon as forget is called. The chain is excluded from recall, injection, the agenda, refresh inputs and `used` credit. Mental model entries citing it are dropped, the prompt block cache is cleared, in-context sets are scrubbed, and recall rows naming it are deleted. Row deletion, passage redaction and the content-hash tombstone go on the bank's extraction queue, behind the chunks that were already waiting. Those chunks reconcile against the hidden memory, which is still there, so their passages join its chain and get erased with it. The pending erase is kept in the SQLite queue, so it survives a crash.
- **The turn that asks to forget is never stored.** The plugin scans the turn's `messages` for a `memory_forget` call and sends `forget_requested: true` on `sync_turn`. The daemon keeps only the turn's key as a tombstone, never extracts it, and deletes the recall row for that turn's `recall_id`. The forget audit row holds that key. On `memory_forget` the plugin also drops the session's stored last prefetch query, so the request isn't sent as the next prefetch's previous message. This avoids needing a short-lived embedding tombstone.

## Inspection and correction

- **`memory show`** gives a memory's sentence, kind, window, phase, both significance fields, the source passage or why it's gone, the access log, the edit log, the chain, and the secret-scan kinds that fired on its source. It breaks strength into the significance boost, recent use and lasting floor. It names any guard that's holding back a purge, and gives projected fade and purge dates as bank-time durations plus the earliest world date at full speed, labelled as such. `model show` and `entity show` give the same view for mental model entries and entities. The agent gets no new tool.
- **Merges keep the entity row.** `entity merge <from> <into>` sets `from.merged_into` and moves aliases and links in a logged edit. Commit resolves links through `merged_into` for chunks whose first call ran before the merge. Mental model filters on `from` are repointed, and the model is refreshed. `user` and `assistant` can only ever be `into`. `entity unmerge <edit-id>` reverses the merge, provided `into` hasn't been merged again since. `entity alias rm --relink-to` fixes a wrong alias.
- **Metadata can be edited, content can't.** The CLI changes the owner's significance (`memory significance <id> <level|clear>`, the same field keep and unkeep write) and entity links. It never changes a memory's sentence, kind or window. Those go through conversation.

## Sweeps, pauses and failures

- **The nightly sweep** purges memories and deletes the text of sources, failed chunks and recall rows that are past their 90-day horizon. It writes one run row per bank with counts only, plus the fingerprint hash and δ it ran under.
- **Acknowledging a pause.** `asphodel purge plan` can be run at any time. It shows which fingerprinted values changed and how many memories, sources and rows the sweep would delete now. `asphodel purge ack --hash <h>` has to quote the hash the running daemon computed. The ack is stored as an edit row, so it survives restarts and travels with a restore. Purging resumes at the next sweep. Running the plan first is recommended, but nothing enforces it.
- **`asphodel status`** shows queue depth, failed chunks, failed refreshes, the pause and both hashes, the last sweep, the pre-migration copy, and when `POST /v1/backup` last completed (which says nothing about whether the stream reached its destination). It exits non-zero when anything needs attention.

## Considered Options

- **Server-side backup destinations and retention.** The supervisor already schedules things, and object storage credentials have no business in the daemon.
- **Restore through the API.** Swapping the database under a running daemon is a hot-swap problem for a rare operation. Stopping the pod is cheap.
- **Extracting the forget request turn and then redacting it.** Extraction could re-create the forgotten memory before the redaction knew which span to cut.
- **Erasing immediately on forget.** A chunk still waiting in the queue would bring the content back as a new memory with nothing to reconcile it against.
- **Deleting `from` on merge.** A chunk whose first call ran before the merge would commit links to an entity that no longer exists.
- **Making the plan mandatory before an ack.** The hash is visible in `/v1/config`, so enforcement would be theatre for a single operator.
- **A Prometheus endpoint.** A non-zero exit from `status` is enough to alert on for one user.

## Consequences

- Forget reaches the live store, the pre-migration copy within 7 days, and nothing beyond it. Backups taken before the forget still hold the content until the backup retention removes them, and Hermes' own transcript and the current session's context are outside Asphodel's reach. Turns between a backup and a restore are lost.
- In a queued chunk, a claim labelled `ends` creates a memory outside the chain whose sentence may restate forgotten content. The erase clears `ended_by` and leaves that memory.
- **Logging.** Content is logged only at `trace`. Content means sentences, source text, recall queries, entity names and aliases, and LLM bodies. This applies to the daemon, CLI error paths, HTTP access logs and the plugin, and panics carry ids only. Failed chunk rows store the error kind and HTTP status, never the response.
- **`asphodel reembed --bank`** is a daemon job. It writes vectors to a side table, can resume by rowid, and swaps them in atomically. Until the swap, the bank is served with its recorded model, so the image has to carry both models during a change. The new model's reconcile floor is calibrated in replay before the deploy. A reranker-only change needs only its floor.
- **`bank delete <bank> --confirm <bank>`** erases through the same path, removes the bank's tombstones, edit rows and session mappings, and writes a daemon-wide `bank_deleted` row with counts. A live plugin recreates the bank empty on `initialize`, so disable the plugin first.

## Amendment: the erase is a barrier, and a hold drains the pool (2026-10-03)

`[llm] concurrency` lets a bank have several chunks out at once (ADR 0005's amendment), which changes two things here.

- **The erase is a barrier to claims.** A chunk queued after a forget could otherwise be claimed while the erase waits for the chunks queued before it, reconcile against the hidden memory and commit after the erase. Its labels would be dropped as vanished, and the forgotten content would come back as a new memory. So no chunk queued after a pending erase is handed out until the erase has run. Call 1 alone would be safe past it, since candidates and in-context sets leave hidden memories out, but the barrier sits at the claim to keep the queue simple. At concurrency 1 it only matters when a chunk queued after the forget sorts ahead of one queued before it, such as a turn ahead of a document's chunks, and that chunk now waits for the erase too.
- **A hold drains the pool.** A re-embed's swap and a bank deletion hold the bank. The hold waits for every chunk the bank has out, and nothing more is handed out while it waits, so a busy bank can't starve it.
- **LLM limits are shared.** `[llm] concurrency` also caps the LLM calls in flight across the daemon, refreshes included. A usage limit, or a 429 with a `Retry-After` in seconds or as an HTTP date, on any call holds every caller until it lifts. No chunk counts it as a failure, and a refresh it holds isn't recorded as a failed refresh: it stays requested and is due again when the hold lifts. A 429 without one is still a counted, retryable failure.
