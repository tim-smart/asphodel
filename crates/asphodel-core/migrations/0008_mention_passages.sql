-- Asphodel schema, version 8: where a forgotten memory was said.
--
-- Forgotten text could outlive the erase in two ways.
--
-- Version 7 kept a mention's span on its access. A later version of the same
-- document repeating a memory isn't credited, so it left no access and its
-- passage had nowhere to be recorded. Where a memory was restated is provenance
-- for the erase, not strength, so it moves to a table of its own: one row per
-- claim that mentioned, confirmed, or (as an older claim) retracted or refined
-- the memory without becoming a memory itself. `start_offset` and `end_offset`
-- are characters into the chunk's text, as `memories.source_start` and
-- `source_end` are. The spans version 7 stored are copied across, and
-- `accesses.spans` is no longer written.
--
-- A chunk a document version shares with an earlier one gets no row of its
-- own, but the version's source text holds it again. `chunk_redactions`
-- records what forget masked in a chunk, so a version ingested later masks
-- the same characters in its own text before storing it.
--
-- IF NOT EXISTS and OR IGNORE, so running it again over a store that has it
-- is harmless, as versions 5 and 6 are.
CREATE TABLE IF NOT EXISTS mention_passages (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id    INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  chunk_id     INTEGER NOT NULL REFERENCES chunks(id),
  start_offset INTEGER NOT NULL,
  end_offset   INTEGER NOT NULL,
  UNIQUE (memory_id, chunk_id, start_offset, end_offset)
);
CREATE INDEX IF NOT EXISTS mention_passages_chunk ON mention_passages(chunk_id);

INSERT OR IGNORE INTO mention_passages (memory_id, chunk_id, start_offset, end_offset)
SELECT a.memory_id,
       json_extract(span.value, '$[0]'),
       json_extract(span.value, '$[1]'),
       json_extract(span.value, '$[2]')
FROM accesses a, json_each(a.spans) span
WHERE a.spans IS NOT NULL
  AND EXISTS (SELECT 1 FROM chunks c WHERE c.id = json_extract(span.value, '$[0]'));

CREATE TABLE IF NOT EXISTS chunk_redactions (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  chunk_id INTEGER NOT NULL UNIQUE REFERENCES chunks(id) ON DELETE CASCADE,
  spans    TEXT NOT NULL
);
