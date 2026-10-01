-- Asphodel schema, version 6: built blocks and a turn's entries.
--
-- "Mental models" (TIM-95, decision 4): a session's block puts the agenda
-- and the memories its entries cite in context, and extraction receives
-- each entry's text with its memory ids, so a reply that relies on an
-- entry is `used` on every memory it cites.
--
-- `prompt_blocks` keeps what each built block listed and cited, and its
-- rendered entries, by the block id the API returned. The block cache is in
-- memory and rebuilds whenever its content changes, so a plugin that got a
-- block without a session id, and sends its id with the first prefetch,
-- may hold one the cache has replaced. `in_context` is a JSON array of
-- memory public ids; `entries` is a JSON array of
-- `{"entry", "text", "cites": [memory public ids]}`. A block no session
-- maps is deleted `sessions.mapping_expiry_days` after it was built.
--
-- `turn_entries.entries` is the same shape: the entries of the block the
-- session held when the turn was synced, stored with its in-context set in
-- the turn's transaction, so a refresh between the sync and the worker
-- can't change what the turn is credited with (TIM-110 review). A turn with
-- none has no row, and the row goes when `turn_in_context`'s does, once the
-- turn's chunk is extracted. It's a table of its own, not a column on
-- `turn_in_context`, so running this again over a store that has it is
-- harmless, as version 5 is.
--
-- For "Erase path, forget, purge and the nightly sweep" (TIM-112): forget
-- must remove a forgotten memory's id from `prompt_blocks.in_context` and
-- drop every entry citing it from `prompt_blocks.entries` and
-- `turn_entries.entries`, since an entry's text restates the memory. The
-- sweep must delete a turn's `turn_entries` row with its `turn_in_context`
-- row.
CREATE TABLE IF NOT EXISTS prompt_blocks (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid       TEXT NOT NULL UNIQUE,
  bank_id    INTEGER NOT NULL REFERENCES banks(id),
  in_context TEXT NOT NULL,
  entries    TEXT NOT NULL,
  built_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS prompt_blocks_bank_built ON prompt_blocks(bank_id, built_at);

CREATE TABLE IF NOT EXISTS turn_entries (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id INTEGER NOT NULL UNIQUE REFERENCES sources(id) ON DELETE CASCADE,
  entries   TEXT NOT NULL
);
