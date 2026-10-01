-- Asphodel schema, version 2: speaker identities.
--
-- A turn's speaker is attributed by platform id (`discord:<id>`), never by
-- display name ("API surface and Hermes transport", TIM-94, decision 1).
-- Entity aliases are free text: display names, renames, and the surface
-- forms extraction finds. Resolving speakers through them let a display
-- name shaped like a platform id capture that id, the owner's included
-- (TIM-106 review). This table holds the only mappings a speaker is
-- resolved through. Bank config writes the owner's platform ids here, and
-- ingest writes one when it meets a new speaker.
CREATE TABLE speaker_ids (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_id     INTEGER NOT NULL REFERENCES banks(id),
  platform_id TEXT NOT NULL,
  entity_id   INTEGER NOT NULL REFERENCES entities(id),
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL,
  UNIQUE (bank_id, platform_id)
);
CREATE INDEX speaker_ids_entity ON speaker_ids(entity_id);

-- Backfill for a version 1 store. No alias is trusted for its shape alone;
-- each mapping below comes from a row whose origin is known.
--
-- 1. The owner's platform ids. In version 1, every alias of the seeded
--    `user` entity was written by bank config (`PUT /v1/banks/{bank}` and
--    `asphodel bank config`): the owner's name, its renames and the owner's
--    platform ids. Ingest never added an alias to `user`. Of those, the
--    platform ids are the ones of the form `<platform>:<id>` with no
--    whitespace; names are left out. Plugin instances that keep running
--    across a daemon upgrade don't call `initialize` again, so waiting for
--    the next bank config would credit the owner's turns to a stranger
--    until then.
INSERT INTO speaker_ids (bank_id, platform_id, entity_id, created_at, updated_at)
SELECT a.bank_id, a.alias, a.entity_id, a.created_at, a.created_at
FROM entity_aliases a
JOIN entities e ON e.id = a.entity_id
WHERE e.seeded = 'user'
  AND a.alias GLOB '?*:?*'
  AND a.alias NOT GLOB '*[ 	]*'
ORDER BY a.id;

-- 2. Speakers version 1 ingest created. Only ingest wrote an
--    `entity_created` edit, and it always added the speaker's platform id
--    as the entity's first alias, before the display name. Where the owner
--    already holds the id, bank config wins.
INSERT OR IGNORE INTO speaker_ids (bank_id, platform_id, entity_id, created_at, updated_at)
SELECT a.bank_id, a.alias, a.entity_id, a.created_at, a.created_at
FROM entity_aliases a
WHERE a.id IN (
  SELECT min(first.id)
  FROM entity_aliases first
  JOIN edits created ON created.entity_id = first.entity_id AND created.kind = 'entity_created'
  JOIN entities e ON e.id = first.entity_id
  WHERE e.seeded IS NULL
  GROUP BY first.entity_id
)
ORDER BY a.id;
