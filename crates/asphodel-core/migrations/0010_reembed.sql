-- Asphodel schema, version 10: re-embedding a bank.
--
-- `asphodel reembed --bank` is a daemon job. `reembeds` holds a bank's
-- job: the model it embeds with and the last memory rowid it reached, so a
-- job a restart stopped resumes there. `reembed_vectors` holds the new
-- model's vectors until the swap, which replaces the bank's vectors,
-- records the model in `banks.embedding_model` and deletes both rows in one
-- transaction. Until then the bank is served with its recorded model.
--
-- The side table has no foreign key to `memories`: the erase deletes a
-- memory's staged row in its own transaction, staging checks the memory
-- is still there, and the swap takes only rows whose memory is.
--
-- IF NOT EXISTS, so running it again over a store that has it is harmless,
-- including one an earlier build created these tables in at open.
CREATE TABLE IF NOT EXISTS reembeds (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_id    INTEGER NOT NULL UNIQUE REFERENCES banks(id),
  model      TEXT NOT NULL,
  cursor     INTEGER NOT NULL DEFAULT 0,
  embedded   INTEGER NOT NULL DEFAULT 0,
  started_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS reembed_vectors (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_id   INTEGER NOT NULL,
  memory_id INTEGER NOT NULL UNIQUE,
  embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS reembed_vectors_bank ON reembed_vectors(bank_id, memory_id);
