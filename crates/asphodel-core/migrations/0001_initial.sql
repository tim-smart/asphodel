-- Asphodel schema, version 1.
--
-- Every `*_at` column is an INTEGER of microseconds since the Unix epoch, in
-- UTC, written from the service's Clock. No column defaults to SQLite's own
-- clock, so a replay on a simulated clock writes the same rows as production.
-- Timezone-dependent rendering happens in code, from the source's timezone.
--
-- Rowids are AUTOINCREMENT so they are never reused: sqlite-vec keys vectors by
-- rowid, and a reused id would attach an old vector to a new memory. The `uuid`
-- columns are the public UUIDv7 ids the API and plugin see. Nothing refers
-- across banks: every row that belongs to a bank carries its `bank_id`, and
-- joins never leave it.

-- Daemon-wide facts: the stored deletion fingerprint, the time the last
-- backup completed.
CREATE TABLE store_meta (
  key        TEXT PRIMARY KEY,
  value      TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

-- One row per schema migration this store has been through. The
-- pre-migration copy keyed by `from_version` is deleted 7 days after
-- `completed_at`.
CREATE TABLE migrations (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  from_version   INTEGER NOT NULL,
  to_version     INTEGER NOT NULL UNIQUE,
  binary_version TEXT NOT NULL,
  started_at     INTEGER NOT NULL,
  completed_at   INTEGER NOT NULL
);

-- Banks: identity, timezone and the recorded model ids. `turns` is the per-bank
-- monotonic turn counter and `last_turn_at` is where bank time's full-speed
-- window starts: bank time runs at full speed for 24 hours after a turn and at
-- `quiet_rate` otherwise.
CREATE TABLE banks (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid            TEXT NOT NULL UNIQUE,
  name            TEXT NOT NULL UNIQUE,
  owner_name      TEXT,
  assistant_name  TEXT,
  timezone        TEXT NOT NULL,
  embedding_model TEXT NOT NULL,
  reranker_model  TEXT NOT NULL,
  turns           INTEGER NOT NULL DEFAULT 0,
  last_turn_at    INTEGER,
  created_at      INTEGER NOT NULL,
  updated_at      INTEGER NOT NULL
);

-- Entities are first-class but there is no graph. `seeded` marks the `user` and
-- `assistant` every bank starts with; they can only ever be merge targets. A
-- merge keeps the row and sets `merged_into`.
CREATE TABLE entities (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid        TEXT NOT NULL UNIQUE,
  bank_id     INTEGER NOT NULL REFERENCES banks(id),
  name        TEXT NOT NULL,
  kind        TEXT NOT NULL
                CHECK (kind IN ('person', 'place', 'organisation', 'project', 'thing')),
  seeded      TEXT CHECK (seeded IN ('user', 'assistant')),
  merged_into INTEGER REFERENCES entities(id),
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL
);
CREATE UNIQUE INDEX entities_seeded ON entities(bank_id, seeded) WHERE seeded IS NOT NULL;
CREATE INDEX entities_bank ON entities(bank_id);
CREATE INDEX entities_merged_into ON entities(merged_into) WHERE merged_into IS NOT NULL;

-- Aliases get their own table so full-text search can match them. The owner's
-- platform ids (`discord:<id>`) are aliases of `user`.
CREATE TABLE entity_aliases (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_id    INTEGER NOT NULL REFERENCES banks(id),
  entity_id  INTEGER NOT NULL REFERENCES entities(id),
  alias      TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  UNIQUE (entity_id, alias)
);
CREATE INDEX entity_aliases_bank_alias ON entity_aliases(bank_id, alias);

CREATE VIRTUAL TABLE entity_aliases_fts USING fts5(
  alias,
  content = 'entity_aliases',
  content_rowid = 'id',
  tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TRIGGER entity_aliases_fts_insert AFTER INSERT ON entity_aliases BEGIN
  INSERT INTO entity_aliases_fts (rowid, alias) VALUES (new.id, new.alias);
END;
CREATE TRIGGER entity_aliases_fts_delete AFTER DELETE ON entity_aliases BEGIN
  INSERT INTO entity_aliases_fts (entity_aliases_fts, rowid, alias)
    VALUES ('delete', old.id, old.alias);
END;
CREATE TRIGGER entity_aliases_fts_update AFTER UPDATE OF alias ON entity_aliases BEGIN
  INSERT INTO entity_aliases_fts (entity_aliases_fts, rowid, alias)
    VALUES ('delete', old.id, old.alias);
  INSERT INTO entity_aliases_fts (rowid, alias) VALUES (new.id, new.alias);
END;

-- Sources are kept verbatim while a memory rests on them, and otherwise for
-- the 90-day source horizon after ingest. `text` is the user message or the
-- document, `reply` the assistant's reply of a turn; both are the clean turn,
-- never Hermes' `api_content`. The sweep sets them to NULL and
-- `tombstoned_at`, keeping the key so ingest stays idempotent. A turn that
-- asked to forget is a tombstone from the start.
CREATE TABLE sources (
  id                   INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid                 TEXT NOT NULL UNIQUE,
  bank_id              INTEGER NOT NULL REFERENCES banks(id),
  kind                 TEXT NOT NULL CHECK (kind IN ('turn', 'document')),
  session_id           TEXT,
  message_at           INTEGER,
  document_id          TEXT,
  content_hash         TEXT NOT NULL,
  platform             TEXT,
  author_id            TEXT,
  author_name          TEXT,
  author_is_bot        INTEGER NOT NULL DEFAULT 0,
  observed_at          INTEGER NOT NULL,
  reference_date       TEXT,
  reference_date_exact INTEGER NOT NULL DEFAULT 1,
  timezone             TEXT NOT NULL,
  text                 TEXT,
  reply                TEXT,
  secret_kinds         TEXT,
  recall_id            TEXT,
  ingested_at          INTEGER NOT NULL,
  tombstoned_at        INTEGER,
  tombstone_reason     TEXT CHECK (tombstone_reason IN ('forget_requested', 'swept')),
  CHECK (kind != 'turn' OR (session_id IS NOT NULL AND message_at IS NOT NULL)),
  CHECK (kind != 'document' OR document_id IS NOT NULL)
);
-- The idempotency keys: bank, session, message time and content hash for a
-- turn; bank, document id and content hash for a document.
CREATE UNIQUE INDEX sources_turn_key
  ON sources(bank_id, session_id, message_at, content_hash) WHERE kind = 'turn';
CREATE UNIQUE INDEX sources_document_key
  ON sources(bank_id, document_id, content_hash) WHERE kind = 'document';
CREATE INDEX sources_bank_ingested ON sources(bank_id, ingested_at);
CREATE INDEX sources_bank_session ON sources(bank_id, session_id) WHERE kind = 'turn';

-- The chunk is the extraction unit. `content_hash` is fixed at ingest and never
-- recomputed: it is the forget tombstone, not an integrity check.
-- `call1_output` is dropped when the chunk commits, so the claim text doesn't
-- outlive a purge or forget. Offsets are into the source text. A chunk whose
-- extraction is pending or failed is never swept.
CREATE TABLE chunks (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid              TEXT NOT NULL UNIQUE,
  bank_id           INTEGER NOT NULL REFERENCES banks(id),
  source_id         INTEGER NOT NULL REFERENCES sources(id),
  position          INTEGER NOT NULL,
  heading_path      TEXT,
  content_hash      TEXT NOT NULL,
  start_offset      INTEGER NOT NULL,
  end_offset        INTEGER NOT NULL,
  text              TEXT,
  call1_output      TEXT,
  extracted_at      INTEGER,
  error_count       INTEGER NOT NULL DEFAULT 0,
  last_error_kind   TEXT,
  last_error_status INTEGER,
  failed_at         INTEGER,
  tombstoned_at     INTEGER,
  UNIQUE (source_id, position)
);
CREATE INDEX chunks_bank_hash ON chunks(bank_id, content_hash);
CREATE INDEX chunks_failed ON chunks(bank_id, failed_at) WHERE failed_at IS NOT NULL;

-- The extraction queue, in SQLite so nothing queued is lost on SIGTERM. Chunks
-- run in `observed_at` order with turns ahead of document chunks; an erase job
-- waits behind the chunks queued before it.
CREATE TABLE extraction_queue (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_id     INTEGER NOT NULL REFERENCES banks(id),
  kind        TEXT NOT NULL CHECK (kind IN ('chunk', 'erase')),
  chunk_id    INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
  memory_ids  TEXT,
  reason      TEXT CHECK (reason IN ('forget', 'purge')),
  priority    INTEGER NOT NULL,
  observed_at INTEGER NOT NULL,
  enqueued_at INTEGER NOT NULL,
  attempts    INTEGER NOT NULL DEFAULT 0,
  CHECK (kind != 'chunk' OR chunk_id IS NOT NULL),
  CHECK (kind != 'erase' OR (memory_ids IS NOT NULL AND reason IS NOT NULL))
);
CREATE INDEX extraction_queue_order ON extraction_queue(bank_id, priority, observed_at, id);

-- Memories. Content and kind never change: a different claim is a new memory.
-- `significance` is the level extraction gave; `owner_significance` is the
-- owner's setting, which keep, unkeep and `memory significance` write.
-- Strength is never stored. Times in the validity window are UTC instants at
-- the start of their unit, each with its precision. `hidden_at` is set the
-- moment forget is called, before the erase runs behind the queue.
CREATE TABLE memories (
  id                         INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid                       TEXT NOT NULL UNIQUE,
  bank_id                    INTEGER NOT NULL REFERENCES banks(id),
  content                    TEXT NOT NULL,
  kind                       TEXT NOT NULL
                               CHECK (kind IN ('fact', 'event', 'state', 'task', 'recurring')),
  significance               TEXT NOT NULL
                               CHECK (significance IN ('trivial', 'minor', 'notable', 'major', 'critical')),
  owner_significance         TEXT
                               CHECK (owner_significance IN ('trivial', 'minor', 'notable', 'major', 'critical', 'kept')),
  chunk_id                   INTEGER NOT NULL REFERENCES chunks(id),
  source_start               INTEGER NOT NULL,
  source_end                 INTEGER NOT NULL,
  observed_at                INTEGER NOT NULL,
  valid_from                 INTEGER,
  valid_from_precision       TEXT CHECK (valid_from_precision IN ('year', 'month', 'day', 'hour', 'minute')),
  valid_until                INTEGER,
  valid_until_precision      TEXT CHECK (valid_until_precision IN ('year', 'month', 'day', 'hour', 'minute')),
  until_event                TEXT,
  window_confidence          TEXT NOT NULL CHECK (window_confidence IN ('high', 'low')),
  due_at                     INTEGER,
  due_at_precision           TEXT CHECK (due_at_precision IN ('year', 'month', 'day', 'hour', 'minute')),
  volatility                 TEXT CHECK (volatility IN ('hours', 'days', 'weeks', 'months', 'years')),
  recurrence_text            TEXT,
  recurrence_rrule           TEXT,
  recurrence_start           INTEGER,
  recurrence_start_precision TEXT CHECK (recurrence_start_precision IN ('year', 'month', 'day', 'hour', 'minute')),
  invalidated_at             INTEGER,
  superseded_by              INTEGER REFERENCES memories(id) ON DELETE SET NULL,
  ended_by                   INTEGER REFERENCES memories(id) ON DELETE SET NULL,
  hidden_at                  INTEGER,
  created_at                 INTEGER NOT NULL,
  updated_at                 INTEGER NOT NULL,
  CHECK (kind = 'state' OR volatility IS NULL),
  CHECK (kind = 'recurring' OR (recurrence_text IS NULL AND recurrence_rrule IS NULL)),
  CHECK (kind = 'task' OR due_at IS NULL)
);
CREATE INDEX memories_bank ON memories(bank_id);
CREATE INDEX memories_bank_kind ON memories(bank_id, kind);
CREATE INDEX memories_chunk ON memories(chunk_id);
CREATE INDEX memories_superseded_by ON memories(superseded_by) WHERE superseded_by IS NOT NULL;
CREATE INDEX memories_ended_by ON memories(ended_by) WHERE ended_by IS NOT NULL;
CREATE INDEX memories_bank_due ON memories(bank_id, due_at) WHERE due_at IS NOT NULL;
CREATE INDEX memories_bank_valid_from ON memories(bank_id, valid_from) WHERE valid_from IS NOT NULL;

CREATE TRIGGER memories_content_is_fixed BEFORE UPDATE OF content, kind ON memories BEGIN
  SELECT RAISE(ABORT, 'a memory''s content and kind never change; a different claim is a new memory');
END;

-- FTS5 over memory content, as an external-content table so the sentence lives
-- in `memories` only.
CREATE VIRTUAL TABLE memories_fts USING fts5(
  content,
  content = 'memories',
  content_rowid = 'id',
  tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TRIGGER memories_fts_insert AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts (rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER memories_fts_delete AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts (memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

-- Memory vectors: a flat sqlite-vec table, one per memory, partitioned by bank
-- so a KNN never crosses banks. bge-small-en-v1.5 vectors are unit length, so
-- cosine is the metric reconcile's floor is set in.
CREATE VIRTUAL TABLE memory_vectors USING vec0(
  memory_id INTEGER PRIMARY KEY,
  bank_id   INTEGER PARTITION KEY,
  embedding FLOAT[384] distance_metric=cosine
);

-- The memory-entity join. `surface_form` is how the text named the entity, kept
-- so a mislink can be undone.
CREATE TABLE memory_entities (
  memory_id    INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  entity_id    INTEGER NOT NULL REFERENCES entities(id),
  surface_form TEXT,
  PRIMARY KEY (memory_id, entity_id)
);
CREATE INDEX memory_entities_entity ON memory_entities(entity_id);

-- The access log: append-only, and only the events that count towards
-- strength. `at` is world time; `turn` is the bank's turn counter at the
-- time, kept for replay and the recall log. At most one access per memory per
-- turn, keeping the strongest kind.
CREATE TABLE accesses (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_id   INTEGER NOT NULL REFERENCES banks(id),
  memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  kind      TEXT NOT NULL CHECK (kind IN ('created', 'used', 'mentioned_again', 'confirmed')),
  at        INTEGER NOT NULL,
  turn      INTEGER NOT NULL,
  source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL,
  UNIQUE (memory_id, turn)
);
CREATE INDEX accesses_memory_at ON accesses(memory_id, at);

-- The recall log: one row per recall, with what came back. It stores the query,
-- which is the user's message, so rows are swept at the 90-day horizon. Being
-- recalled never counts as an access.
CREATE TABLE recalls (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid       TEXT NOT NULL UNIQUE,
  bank_id    INTEGER NOT NULL REFERENCES banks(id),
  kind       TEXT NOT NULL CHECK (kind IN ('prefetch', 'tool', 'refresh')),
  session_id TEXT,
  turn       INTEGER NOT NULL,
  query      TEXT,
  latency_ms INTEGER NOT NULL,
  at         INTEGER NOT NULL,
  swept_at   INTEGER
);
CREATE INDEX recalls_bank_at ON recalls(bank_id, at);
CREATE INDEX recalls_bank_session ON recalls(bank_id, session_id);

CREATE TABLE recall_results (
  recall_id INTEGER NOT NULL REFERENCES recalls(id) ON DELETE CASCADE,
  memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  rank      INTEGER NOT NULL,
  score     REAL,
  injected  INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (recall_id, memory_id)
);
CREATE INDEX recall_results_memory ON recall_results(memory_id);

-- The edit log: every metadata edit, merge, forget, purge, restore,
-- acknowledgement and bank deletion. `details` is JSON of ids,
-- times, spans and counts, never content. `bank_id` is NULL for daemon-wide
-- rows such as `restored` and `bank_deleted`.
CREATE TABLE edits (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid       TEXT NOT NULL UNIQUE,
  bank_id    INTEGER REFERENCES banks(id),
  kind       TEXT NOT NULL,
  memory_id  INTEGER REFERENCES memories(id) ON DELETE SET NULL,
  entity_id  INTEGER REFERENCES entities(id) ON DELETE SET NULL,
  details    TEXT NOT NULL,
  at         INTEGER NOT NULL
);
CREATE INDEX edits_bank_at ON edits(bank_id, at);
CREATE INDEX edits_bank_kind ON edits(bank_id, kind);
CREATE INDEX edits_memory ON edits(memory_id) WHERE memory_id IS NOT NULL;
CREATE INDEX edits_entity ON edits(entity_id) WHERE entity_id IS NOT NULL;

-- Mental models: a question, filters, a token budget, and entries that each
-- cite the memories they rest on. The `inject` flag is gone: every
-- enabled model is injected.
CREATE TABLE mental_models (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid                  TEXT NOT NULL UNIQUE,
  bank_id               INTEGER NOT NULL REFERENCES banks(id),
  name                  TEXT NOT NULL,
  question              TEXT NOT NULL,
  filter_kinds          TEXT,
  filter_entity_id      INTEGER REFERENCES entities(id),
  filter_min_volatility TEXT CHECK (filter_min_volatility IN ('hours', 'days', 'weeks', 'months', 'years')),
  max_tokens            INTEGER NOT NULL,
  enabled               INTEGER NOT NULL DEFAULT 1,
  last_fingerprint      TEXT,
  last_refreshed_at     INTEGER,
  refresh_requested_at  INTEGER,
  last_error_kind       TEXT,
  last_error_at         INTEGER,
  created_at            INTEGER NOT NULL,
  updated_at            INTEGER NOT NULL,
  UNIQUE (bank_id, name)
);

CREATE TABLE mental_model_entries (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid       TEXT NOT NULL UNIQUE,
  model_id   INTEGER NOT NULL REFERENCES mental_models(id) ON DELETE CASCADE,
  position   INTEGER NOT NULL,
  text       TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX mental_model_entries_model ON mental_model_entries(model_id, position);

-- An entry goes when any memory it cites goes: the cascade here removes the
-- citation, and code drops the entry.
CREATE TABLE mental_model_citations (
  entry_id  INTEGER NOT NULL REFERENCES mental_model_entries(id) ON DELETE CASCADE,
  memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  PRIMARY KEY (entry_id, memory_id)
);
CREATE INDEX mental_model_citations_memory ON mental_model_citations(memory_id);

-- The daemon persists each Hermes session's block id and the memory ids the
-- block's entries cite, so a reply that relies on an entry is credited as
-- `used`. Mappings expire after `sessions.mapping_expiry_days` without a turn.
CREATE TABLE session_blocks (
  bank_id      INTEGER NOT NULL REFERENCES banks(id),
  session_id   TEXT NOT NULL,
  block_id     TEXT NOT NULL,
  cited        TEXT NOT NULL,
  built_at     INTEGER NOT NULL,
  last_turn_at INTEGER NOT NULL,
  PRIMARY KEY (bank_id, session_id)
);

-- One row per bank per nightly sweep, counts only, with the fingerprint and
-- δ it ran under. `delta` NULL means never purge.
CREATE TABLE sweep_runs (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_id             INTEGER NOT NULL REFERENCES banks(id),
  started_at          INTEGER NOT NULL,
  completed_at        INTEGER NOT NULL,
  fingerprint         TEXT NOT NULL,
  delta               REAL,
  purged_memories     INTEGER NOT NULL,
  swept_sources       INTEGER NOT NULL,
  swept_chunks        INTEGER NOT NULL,
  swept_failed_chunks INTEGER NOT NULL,
  swept_recalls       INTEGER NOT NULL
);
CREATE INDEX sweep_runs_bank_at ON sweep_runs(bank_id, completed_at);
