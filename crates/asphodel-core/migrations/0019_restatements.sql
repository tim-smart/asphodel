-- Asphodel schema, version 19: what a repeat absorbed.
--
-- A newer claim that call 2 labels `mentioned_again` or `confirmed`, that the
-- date and significance guards leave a repeat and that lands as an access, is
-- absorbed: it becomes no memory, and the sentence call 1 wrote went when the
-- chunk's saved reply was cleared at commit. It's kept here instead, one row
-- per absorbed claim and memory it was credited to, so a later pass can look
-- at it again. Nothing reads it for recall, reconciliation or injection, and
-- it has no vector.
--
-- `claim` is the checked claim as call 1 gave it, as JSON: its sentence, kind,
-- resolved window, significance and entity links. `label` is call 2's label
-- and `outcome` the label after the guards. `call2_version` and `model` are
-- the call 2 template and the model that labelled it. The span is the claim's
-- quote, in characters into the chunk's text, as `mention_passages` holds it.
--
-- A row goes with its memory: purge, forget and bank deletion delete
-- memories, and the cascade takes the row along. A forget that masks text in
-- the chunk deletes the rows whose span it overlaps, and removing a document
-- deletes the rows taken from it, whichever memory they're on.
--
-- IF NOT EXISTS, so running it again over a store that has it is harmless.
CREATE TABLE IF NOT EXISTS restatements (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id     INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  chunk_id      INTEGER NOT NULL REFERENCES chunks(id),
  start_offset  INTEGER NOT NULL,
  end_offset    INTEGER NOT NULL,
  claim         TEXT NOT NULL,
  label         TEXT NOT NULL,
  outcome       TEXT NOT NULL,
  observed_at   INTEGER NOT NULL,
  call2_version INTEGER NOT NULL,
  model         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS restatements_memory ON restatements(memory_id, observed_at);
CREATE INDEX IF NOT EXISTS restatements_chunk ON restatements(chunk_id);
