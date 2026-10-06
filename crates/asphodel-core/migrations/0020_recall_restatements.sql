-- Asphodel schema, version 20: restatements searched by recall. A
-- measurement branch only (TIM-206); it never merges.
--
-- Explicit recall searches every restatement's sentence beside its memory's,
-- in the vector arm and the BM25 arm, and finds the memory it was absorbed
-- into. Nothing else reads these tables: reconciliation, injection, mental
-- models and the entity arm keep searching `memories_fts` and
-- `memory_vectors`, which hold memory sentences only.
--
-- `recall_fts` is one BM25 index over both kinds of sentence, so their ranks
-- come from one query and compare. A memory's row has twice the memory's
-- rowid, and a restatement's twice its rowid plus one. Recall joins each row
-- back to `memories` or `restatements`, so a row whose owner is gone matches
-- nothing.
--
-- `restatement_vectors` holds each restatement's sentence embedding, the
-- claim's vector call 1's checking embedded, as little-endian f32s keyed by
-- restatement rowid. It's a plain table searched by an exact scan of the
-- bank's rows with sqlite-vec's cosine distance, the fallback
-- `store::vector` describes: a bank holds far fewer restatements than
-- memories, and the table drops without the vec0 module. A store migrated with
-- restatements already in it gets their BM25 rows here but no vectors: only
-- restatements written from now on have one.
--
-- The backfill and the triggers that keep both tables in step are the
-- migration's conversion in code
-- (`retrieval::restated::index_for_recall`). Rows go when their memory or
-- restatement does: `recall_fts` through those triggers, which also fire for
-- the cascade from a deleted memory, and a vector by cascade.
--
-- IF NOT EXISTS, so running it again over a store that has it is harmless.
CREATE VIRTUAL TABLE IF NOT EXISTS recall_fts USING fts5(
  sentence,
  tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS restatement_vectors (
  restatement_id INTEGER PRIMARY KEY REFERENCES restatements(id) ON DELETE CASCADE,
  bank_id        INTEGER NOT NULL,
  embedding      BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS restatement_vectors_bank ON restatement_vectors(bank_id);
