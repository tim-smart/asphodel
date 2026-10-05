-- Asphodel schema, version 15: a mental model's answer is one text with
-- one set of citations.
--
-- A refresh writes the whole answer as connected prose, a `### heading`
-- line and one paragraph per section, and cites the memories it rests on
-- as a whole. `answer` is NULL until the first refresh, and again once a
-- forget blanks it for the rewrite. `mental_model_cites` holds the
-- citations by model, in place of `mental_model_citations`, which held
-- them by entry.
--
-- The migration runner then builds each model's answer from its entries,
-- grouped by section in position order with entries from before sections
-- first, takes the union of their citations, and drops
-- `mental_model_entries` and `mental_model_citations`. No refresh is
-- forced; the next one rewrites the answer anyway.
ALTER TABLE mental_models ADD COLUMN answer TEXT;
CREATE TABLE IF NOT EXISTS mental_model_cites (
  model_id  INTEGER NOT NULL REFERENCES mental_models(id) ON DELETE CASCADE,
  memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  PRIMARY KEY (model_id, memory_id)
);
CREATE INDEX IF NOT EXISTS mental_model_cites_memory ON mental_model_cites(memory_id);
