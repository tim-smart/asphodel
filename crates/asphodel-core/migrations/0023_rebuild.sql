-- Asphodel schema, version 23: rebuilding a bank's memories.
--
-- `asphodel rebuild --bank` extracts a bank's memories again from the sources
-- it holds, under the daemon's current templates. It deletes the memories it
-- replaces and puts their chunks back on the bank's extraction queue in one
-- transaction, and `rebuilds` is that transaction's record: one row per bank,
-- its latest rebuild, kept after it's acknowledged.
--
-- `queue_boundary` is the highest queue rowid when the rebuild started, its
-- own rows included: the rebuild is complete once no chunk at or below it is
-- queued. `memory_boundary` is the highest memory rowid when it started: a
-- memory at or below it is one the rebuild kept, and a memory above it is one
-- the rerun made. `chunks`, `replaced` and `kept` count what the start did;
-- `carried` and `dropped` count what the end did with `rebuild_accesses`.
--
-- A row with no `completed_at` turns recall, injection and the agenda off
-- for the bank. A row with no `acked_at` holds the bank's nightly sweep,
-- until `asphodel rebuild ack` sets it. The row stays afterwards: a chunk of
-- a source older than `started_at` that is extracted again keys its memories'
-- ids with `started_at`, so no id of a replaced memory is ever given out
-- again.
--
-- `rebuild_accesses` holds what the rerun can't derive again for a replaced
-- memory: its `used` accesses, its pending credits, and its mentions in
-- chunks that aren't run again, with the passages those were recorded at.
-- Each row names the memory by its own passage, the chunk and the span in
-- characters into its text, and carries the access's kind, time, turn and
-- source, or the mention's chunk and span. No content. When the rebuild
-- completes, each goes to the new memory quoting an overlapping passage of
-- the same chunk, or is dropped, and the rows are deleted.
--
-- IF NOT EXISTS, so running it again over a store that has it is harmless.
CREATE TABLE IF NOT EXISTS rebuilds (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_id         INTEGER NOT NULL UNIQUE REFERENCES banks(id),
  started_at      INTEGER NOT NULL,
  queue_boundary  INTEGER NOT NULL,
  memory_boundary INTEGER NOT NULL,
  chunks          INTEGER NOT NULL,
  replaced        INTEGER NOT NULL,
  kept            INTEGER NOT NULL,
  carried         INTEGER NOT NULL DEFAULT 0,
  dropped         INTEGER NOT NULL DEFAULT 0,
  completed_at    INTEGER,
  acked_at        INTEGER
);

CREATE TABLE IF NOT EXISTS rebuild_accesses (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_id       INTEGER NOT NULL REFERENCES banks(id),
  chunk_id      INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  start_offset  INTEGER NOT NULL,
  end_offset    INTEGER NOT NULL,
  kind          TEXT NOT NULL
                  CHECK (kind IN ('used', 'mentioned_again', 'confirmed', 'pending', 'passage')),
  at            INTEGER,
  turn          INTEGER,
  source_id     INTEGER REFERENCES sources(id) ON DELETE SET NULL,
  mention_chunk INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
  mention_start INTEGER,
  mention_end   INTEGER
);
CREATE INDEX IF NOT EXISTS rebuild_accesses_bank ON rebuild_accesses(bank_id);
