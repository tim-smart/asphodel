-- Asphodel schema, version 13: a mental model is planned and written in
-- sections.
--
-- A refresh asks one narrow question per facet of the model's question
-- instead of one compound one. `plan` holds the facets an LLM call made
-- for the question, as JSON with the question it was made for, so a steady
-- model never pays for it again and a changed question is planned anew. The
-- seeded profile's question has a plan built in and stores none.
--
-- A refresh then writes the whole summary as sections of cited sentences.
-- `section` is the heading an entry is written under. Entries written
-- before this version have none, and render as lines, as they did, until
-- their model's next write replaces them.
ALTER TABLE mental_models ADD COLUMN plan TEXT;
ALTER TABLE mental_model_entries ADD COLUMN section TEXT;
