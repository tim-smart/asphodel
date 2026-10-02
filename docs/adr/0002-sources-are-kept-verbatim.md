# Sources are kept verbatim

Every conversation turn and document that Asphodel extracts from is stored verbatim as a source, and each memory points to the exact passage it came from. This makes it possible to answer "why do you think X?" and to re-extract when the prompts improve. It costs little for a single user. The cost is a set of obligations that a store keeping only memories wouldn't have:

- Ingest has to be idempotent, keyed on bank, session, message and content hash, because Hermes retries and resends transcripts.
- The stored text has to be the clean turn, never Hermes' `api_content` with memories injected into it, or re-extraction would turn injected memories back into new claims.
- Secrets have to be scanned for and removed before the text is stored.
- Forgetting a memory has to redact its passage in the source and keep a content-hash tombstone, so that re-ingesting the source doesn't bring the memory back.

## Amendment: what the tombstone blocks (2026-10-02)

Forget's tombstone blocks the same input, not the same words.

- Sending the same turn or the same document again is a duplicate, and brings nothing back. A later version of the document that keeps a section unchanged skips that section by its content hash, and the masks forget recorded for it are applied to that version's text before it's stored.
- Every passage the forgotten chain rested on or was restated in is masked in every version of the document stored at the time of the forget, wherever it appears verbatim.
- An edited section of the same document is new input. If it still contains the forgotten words, it's stored and extracted like any other new text, and can create a memory again. A later turn that repeats the words is new input in the same way, so documents and turns share one boundary.
- Asphodel keeps no digest of forgotten passages, so it can't recognise them in new input. Passage digests were considered and deferred: a hash of short text can be guessed, a keyed hash brings key management, and their scope (document or bank) and the owner's wish to relearn something are open. They need a privacy and scope decision of their own.
