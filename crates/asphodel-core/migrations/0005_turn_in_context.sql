-- Asphodel schema, version 5: a queued turn's in-context set.
--
-- Extraction judges a turn's `used` verdicts against the memories the agent
-- could see when it wrote the reply (ADR 0001; TIM-94, decision 6): the
-- session's in-context set as the turn's sync left it. The bank's worker
-- reaches the turn later, and sessions live in daemon memory, so reading
-- the session then let a clear on compaction, a later recall or a restart
-- change which memories the turn was credited with (TIM-110 review). Ingest
-- now stores the set here, in the turn's own transaction, and extraction
-- reads only this table.
--
-- `memories` is a JSON array of memory public ids, oldest first, never
-- content. A turn with nothing in context has no row, and neither has a
-- document, a tombstone or a turn ingested before version 5; extraction
-- reads a missing row as an empty set. The row is deleted when the turn's
-- chunk is extracted, so it only lives while the turn waits on the queue
-- or sits failed.
--
-- For the erase path and the sweep ("Erase path, forget, purge and the
-- nightly sweep", TIM-112):
-- - forget must remove the forgotten memories' ids from every row here,
--   queued or failed, as part of ADR 0010's scrubbing of in-context sets.
--   Extraction already ignores ids that are hidden or gone, so this is
--   about not keeping the reference.
-- - the sweep must delete a turn's row when it deletes the text of the
--   turn or of its failed chunk. Sources are kept as tombstones, so the
--   cascade below doesn't cover that.
--
-- IF NOT EXISTS, so running it again over a store that has it is harmless.
CREATE TABLE IF NOT EXISTS turn_in_context (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id INTEGER NOT NULL UNIQUE REFERENCES sources(id) ON DELETE CASCADE,
  memories  TEXT NOT NULL
);
