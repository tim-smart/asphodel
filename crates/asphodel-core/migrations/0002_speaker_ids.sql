-- Asphodel schema, version 2: speaker identities.
--
-- A turn's speaker is attributed by platform id (`discord:<id>`), never by
-- display name. Entity aliases are free text: display names, renames, and the
-- surface forms extraction finds. Resolving speakers through them let a display
-- name shaped like a platform id capture that id, the owner's included. This
-- table holds the only mappings a speaker is resolved through. Bank config
-- writes the owner's platform ids here, and ingest writes one when it meets a
-- new speaker.
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

-- Backfill for a version 1 store. It fails closed: a mapping is written
-- only where version 1 recorded where the alias came from, and no alias is
-- trusted for its shape.
--
-- The owner's platform ids are not carried over. Version 1 kept them only
-- as aliases of the seeded `user`, next to the owner's current and former
-- names, and nothing records which alias was which: a former name shaped
-- like a platform id can't be told from an id. Inferring the owner from
-- them could make a stranger the owner. After the upgrade, each bank's
-- config has to be sent again (`PUT /v1/banks/{bank}`, which every Hermes
-- plugin instance sends from `initialize`) before the owner's turns are
-- attributed to `user`; until then they go to a stranger entity, which the
-- config then takes the id back from. See docs/upgrading.md.
--
-- Speakers version 1 ingest created are carried over. Only ingest wrote an
-- `entity_created` edit, and it always added the speaker's platform id as
-- the entity's first alias, before the display name.
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
