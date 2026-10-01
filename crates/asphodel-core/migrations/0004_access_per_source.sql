-- Asphodel schema, version 4: a document's access has its own key.
--
-- Version 1 allowed one access per memory per turn (TIM-90). A document
-- carries the turn number of the turn before it, so two documents ingested
-- with no turn between them shared a number, and the second one's
-- `mentioned_again` on a memory the first created was dropped. A document
-- independently restating a memory is mentioned again (CONTEXT.md), so the
-- key now includes the source. A turn is one source with its own number, so
-- this changes nothing for turns: extraction still keeps one access per
-- memory per turn, the strongest, and counts a document's access apart from
-- the turn whose number it shares. An access with no source keeps the old
-- key.
--
-- SQLite can't drop a table constraint, so the table is rebuilt with every
-- row and id as they were.

CREATE TABLE accesses_v4 (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_id   INTEGER NOT NULL REFERENCES banks(id),
  memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  kind      TEXT NOT NULL CHECK (kind IN ('created', 'used', 'mentioned_again', 'confirmed')),
  at        INTEGER NOT NULL,
  turn      INTEGER NOT NULL,
  source_id INTEGER REFERENCES sources(id) ON DELETE SET NULL
);
INSERT INTO accesses_v4 (id, bank_id, memory_id, kind, at, turn, source_id)
  SELECT id, bank_id, memory_id, kind, at, turn, source_id FROM accesses;
DROP TABLE accesses;
ALTER TABLE accesses_v4 RENAME TO accesses;

CREATE INDEX accesses_memory_at ON accesses(memory_id, at);
CREATE UNIQUE INDEX accesses_one_per_turn_and_source
  ON accesses(memory_id, turn, IFNULL(source_id, 0));
