-- Asphodel schema, version 20: searching sources, and search indexes that
-- forget.
--
-- FTS5 over a source's document id, session id, text and reply, as an
-- external-content table so the text lives in `sources` only. Triggers keep
-- it in step with every write, so a sweep, a forget's mask, a document
-- removal or a bank deletion takes the old words out with the text.
-- `secure-delete` removes them from the index itself rather than leaving
-- them in older segments under a delete marker.
--
-- IF NOT EXISTS and a rebuild, so running it again over a store that has it
-- is harmless.
CREATE VIRTUAL TABLE IF NOT EXISTS sources_fts USING fts5(
  document_id,
  session_id,
  text,
  reply,
  content = 'sources',
  content_rowid = 'id',
  tokenize = 'unicode61 remove_diacritics 2'
);
INSERT INTO sources_fts (sources_fts, rank) VALUES ('secure-delete', 1);
CREATE TRIGGER IF NOT EXISTS sources_fts_insert AFTER INSERT ON sources BEGIN
  INSERT INTO sources_fts (rowid, document_id, session_id, text, reply)
  VALUES (new.id, new.document_id, new.session_id, new.text, new.reply);
END;
CREATE TRIGGER IF NOT EXISTS sources_fts_delete AFTER DELETE ON sources BEGIN
  INSERT INTO sources_fts (sources_fts, rowid, document_id, session_id, text, reply)
  VALUES ('delete', old.id, old.document_id, old.session_id, old.text, old.reply);
END;
CREATE TRIGGER IF NOT EXISTS sources_fts_update
AFTER UPDATE OF document_id, session_id, text, reply ON sources BEGIN
  INSERT INTO sources_fts (sources_fts, rowid, document_id, session_id, text, reply)
  VALUES ('delete', old.id, old.document_id, old.session_id, old.text, old.reply);
  INSERT INTO sources_fts (rowid, document_id, session_id, text, reply)
  VALUES (new.id, new.document_id, new.session_id, new.text, new.reply);
END;
INSERT INTO sources_fts (sources_fts) VALUES ('rebuild');

-- `memories_fts` left an erased or purged sentence's words in older segments
-- under a delete marker. `secure-delete` stops that, and the rebuild drops
-- what earlier versions left behind.
INSERT INTO memories_fts (memories_fts, rank) VALUES ('secure-delete', 1);
INSERT INTO memories_fts (memories_fts) VALUES ('rebuild');
