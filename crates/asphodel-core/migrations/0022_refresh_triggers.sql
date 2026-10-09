-- Asphodel schema, version 22: the memories that asked for a refresh.
--
-- A new memory at the trigger level or above, a kept one, or one whose
-- significance went up, asks every model whose filters it passes for a
-- refresh, and is kept here until a completed refresh of that model
-- settles it. The refresh reranks it against each facet's query, and one
-- the reranker finds relevant to a facet takes that facet's turns in the
-- write's input ahead of its own best, so stronger memories can't crowd it
-- out. Rows go with
-- their model or memory. There's no backfill: memories written before the
-- upgrade compete for their place as before.
CREATE TABLE IF NOT EXISTS mental_model_triggers (
  id        INTEGER PRIMARY KEY,
  model_id  INTEGER NOT NULL REFERENCES mental_models(id) ON DELETE CASCADE,
  memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  UNIQUE (model_id, memory_id)
);
CREATE INDEX IF NOT EXISTS mental_model_triggers_memory ON mental_model_triggers(memory_id);
