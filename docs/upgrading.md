# Upgrading

## Schema version 2: send each bank's config again

Schema version 2 adds `speaker_ids`, the only mapping a turn's speaker is
resolved through ("API surface and Hermes transport", TIM-94, decision 1).
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
  `asphodel bank config`, once the HTTP API and CLI land (TIM-110).

The daemon logs a warning when it migrates a store from version 1, as a
reminder.

## Schema version 3: aliases and entity names in NFC

Extraction searches the alias index with its passages in Unicode NFC, so
aliases have to be stored composed too. Outside Latin script the alias
index doesn't fold a combining accent away, and an alias stored decomposed
(a letter followed by a combining accent) would never be found
("Extraction call 1", the TIM-107 review). From version 3, every alias and
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
already remembered ("Reconciliation", TIM-108). A document carries the
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
on compaction, a later recall, or a restart in between changed the credit
(the TIM-110 review). From version 5, ingest stores the set with the turn
in `turn_in_context`, and the row goes once the turn is extracted.

The migration only adds the table. Turns already queued when you upgrade
have no stored set and are extracted as if nothing was in context, which
is what a restart did to them before. No action is needed.
