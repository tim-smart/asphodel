# Sources are kept verbatim

Every conversation turn and document that Asphodel extracts from is stored verbatim as a source, and each memory points to the exact passage it came from. This makes it possible to answer "why do you think X?" and to re-extract when the prompts improve. It costs little for a single user. The cost is a set of obligations that a store keeping only memories wouldn't have:

- Ingest has to be idempotent, keyed on bank, session, message and content hash, because Hermes retries and resends transcripts.
- The stored text has to be the clean turn, never Hermes' `api_content` with memories injected into it, or re-extraction would turn injected memories back into new claims.
- Secrets have to be scanned for and removed before the text is stored.
- Forgetting a memory has to redact its passage in the source and keep a content-hash tombstone, so that re-ingesting the source doesn't bring the memory back.

Decided in "What is a memory record?" (TIM-90) on 2026-09-30.
