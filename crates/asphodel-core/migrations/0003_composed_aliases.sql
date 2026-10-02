-- Asphodel schema, version 3: aliases and entity names in NFC.
--
-- Extraction searches its passages in Unicode NFC, so the alias index has to
-- hold aliases in NFC too: a decomposed alias (a letter followed by a combining
-- accent) indexes differently from the composed passage, and outside Latin
-- script the alias FTS doesn't fold the difference away. From this version
-- every write stores aliases and entity names composed; this migration composes
-- what earlier versions stored. `asphodel_nfc` is a deterministic SQL function
-- the store registers on its connection before migrations run.
--
-- Two aliases of one entity that are canonically the same become one row,
-- as `UNIQUE (entity_id, alias)` requires: the oldest stays. Each
-- `alias_added` edit that named a removed row is repointed to the one that
-- stayed, so every edit still names an alias of its entity, and undoing any
-- of them removes the alias each of them added. The edit log holds ids
-- only, never content.

UPDATE edits
SET details = json_set(details, '$.alias_id', (
  SELECT MIN(kept.id)
  FROM entity_aliases removed
  JOIN entity_aliases kept
    ON kept.entity_id = removed.entity_id
   AND asphodel_nfc(kept.alias) = asphodel_nfc(removed.alias)
  WHERE removed.id = json_extract(edits.details, '$.alias_id')
))
WHERE kind = 'alias_added'
  AND json_extract(details, '$.alias_id') IN (
    SELECT later.id
    FROM entity_aliases later
    JOIN entity_aliases earlier
      ON earlier.entity_id = later.entity_id
     AND earlier.id < later.id
     AND asphodel_nfc(earlier.alias) = asphodel_nfc(later.alias)
  );

DELETE FROM entity_aliases
WHERE id IN (
  SELECT later.id
  FROM entity_aliases later
  JOIN entity_aliases earlier
    ON earlier.entity_id = later.entity_id
   AND earlier.id < later.id
   AND asphodel_nfc(earlier.alias) = asphodel_nfc(later.alias)
);

-- The FTS update trigger re-indexes each composed alias from its new value.
UPDATE entity_aliases SET alias = asphodel_nfc(alias) WHERE alias != asphodel_nfc(alias);

UPDATE entities SET name = asphodel_nfc(name) WHERE name != asphodel_nfc(name);
