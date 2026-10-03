-- Asphodel schema, version 12: the owner can remove a document.
--
-- Removing a document id takes every version of it: the memories resting on
-- its chunks are forgotten, its waiting chunks leave the queue, and each
-- version's text is cleared at once. `removed_at` marks those sources. They
-- keep their keys and content hashes, like any tombstone, so sending a
-- version again is a duplicate. A column rather than a new `tombstone_reason`
-- value, since SQLite can't change a CHECK constraint without rebuilding the
-- table.
ALTER TABLE sources ADD COLUMN removed_at INTEGER;
