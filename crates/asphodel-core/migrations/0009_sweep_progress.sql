-- Asphodel schema, version 9: a sweep's counts survive its failure.
--
-- A sweep run row holds counts only (ADR 0010). Each purge and the source sweep
-- commit in transactions of their own before the row is written, so a sweep
-- that failed after deleting something, and was run again, wrote a row that
-- left out what the failed attempt deleted. `sweep_progress` holds a bank's
-- counts while its sweep runs. Each deletion adds to it in its own transaction,
-- and the last step writes the run row from it and deletes it, in one
-- transaction. A sweep resumed after a failure or a restart carries on with the
-- same row, so the run row counts every attempt and keeps the first attempt's
-- start. A row left under another fingerprint or delta is written as a run of
-- its own first.
--
-- IF NOT EXISTS, so running it again over a store that has it is harmless.
CREATE TABLE IF NOT EXISTS sweep_progress (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  bank_id             INTEGER NOT NULL UNIQUE REFERENCES banks(id),
  started_at          INTEGER NOT NULL,
  fingerprint         TEXT NOT NULL,
  delta               REAL,
  purged_memories     INTEGER NOT NULL DEFAULT 0,
  swept_sources       INTEGER NOT NULL DEFAULT 0,
  swept_chunks        INTEGER NOT NULL DEFAULT 0,
  swept_failed_chunks INTEGER NOT NULL DEFAULT 0,
  swept_recalls       INTEGER NOT NULL DEFAULT 0
);
