# Repeats are reconciled at ingest

Extraction makes two LLM calls per chunk. The first extracts claims. The second, reconciliation, compares those claims with the closest existing memories, including faded and ended ones, and decides whether each claim is new, mentioned again, confirmed, or ends, retracts or refines an existing memory. A repeat therefore becomes an access on the memory that already exists, never a second copy of it. Strength depends on this: a memory only gets stronger with use if every repetition is counted against the same memory. Hindsight, which Asphodel borrows from, does it the other way round. Its retain step only appends, and duplicates and contradictions are cleaned up later during consolidation, which Asphodel has ruled out of scope.

## Considered Options

- **Append only, clean up later** (Hindsight). This makes one call per chunk at ingest. But a fact said in ten sessions becomes ten memories that each fade on their own, and a reschedule leaves the old appointment looking current until the cleanup runs.
- **One call, searching the store with the whole chunk.** Whole-chunk search finds poorer neighbours than searching per claim, and the single prompt has to do both jobs.
- **Embedding similarity alone.** This has no way to tell that "moved to Lisbon" ends "lives in Berlin".

## Consequences

- Reconciliation runs only when a claim lands above a similarity floor or signals a change, so a chunk that touches nothing already known costs one call. The floor is tuned for recall on real sessions and stored per embedding model, because a missed neighbour creates a silent duplicate.
- Code, not the LLM, decides from `observed_at` which of two memories is newer, so an old document can't overrule a newer memory.
- Reversing this later is expensive. Once a store has been built this way it holds accesses rather than duplicates, so the history needed to switch to append-and-consolidate no longer exists.
