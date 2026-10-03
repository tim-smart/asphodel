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

## Amendment: reconciliation is checked at commit (2026-10-03)

With `[llm] concurrency` above 1, a bank extracts several chunks at once, and each searches for its neighbours before the others commit. Two chunks that state the same fact would both find nothing and both create a memory. So reconciliation is verified at commit against what the search saw.

- At its search, a chunk notes the bank's newest memory and newest edit. A bank's chunks commit in the order they were claimed, so an older chunk never sees a newer one's memory as a neighbour, as with one worker.
- Inside the commit's transaction, once another chunk of the bank has committed since the search, the chunk searches again if a newer memory is at or above the floor for one of its claims, if a neighbour call 2 was shown has an edit logged since (ended, retracted, refined), or if one of its claims is flagged and any memory is newer. The floor test alone isn't enough for a flagged claim, which pulls in the open tasks and current states of its entities whatever their similarity.
- The redo keeps call 1 and runs only the search and call 2 again, and it isn't counted against the chunk. In the worst case every chunk redoes, which leaves call 1 parallel and call 2 serial. At concurrency 1 nothing else commits in between, so the check never runs.
- Partitioning a bank's chunks by the entities they touch was rejected: `user` and `assistant` are candidates in every turn chunk, so the sets always overlap. A later dedupe pass is the consolidation this ADR rules out.
