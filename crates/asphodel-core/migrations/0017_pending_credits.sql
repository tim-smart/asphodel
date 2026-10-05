-- Asphodel schema, version 17: `used` credits waiting for a second turn.
--
-- With `strength.corroborate_used` on, the first turn a memory's chain is
-- judged used in writes no access. The credit is kept here instead, so a
-- `used` verdict in a later turn knows it's the second and writes its access.
-- It lives apart from `accesses` so nothing that reads the access log, strength
-- included, ever counts it. With the switch off, which is the default, nothing
-- writes here.
--
-- A row holds ids, a turn number and a time, never content. It goes with its
-- memory: purge, forget and bank deletion delete memories, and the cascade
-- takes the row along.
--
-- IF NOT EXISTS, so running it again over a store that has it is harmless.
CREATE TABLE IF NOT EXISTS pending_credits (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_id   INTEGER NOT NULL REFERENCES banks(id),
  memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  turn      INTEGER NOT NULL,
  at        INTEGER NOT NULL,
  UNIQUE (memory_id, turn)
);
