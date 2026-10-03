-- Asphodel schema, version 11: a prefetch logs its raw query.
--
-- Prefetch recalls for the user's message without the note Hermes' Discord
-- gateway puts in front of it and without the `[Name] ` speaker prefix, and
-- `recalls.query` holds that cleaned query, which calibration uses.
-- `raw_query` holds the message as it was sent, so labelling material can
-- show both. It's the user's message too, so the sweep clears it with
-- `query`. Rows logged before this version, and recall tool and refresh
-- rows, have none.
ALTER TABLE recalls ADD COLUMN raw_query TEXT;
