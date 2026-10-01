# Purge is a margin below the recall threshold

Asphodel purges a memory once its strength falls a margin δ below the recall threshold, with δ = 1.0 to start. Nothing new is stored for this: like strength, the purge decision is a pure function of the access log and the clock. Strength can't fall below the significance boost plus the lasting floor, so the margin also protects significant and repeatedly used memories without any extra guard. At δ = 1.0, a memory sits faded for about 17 times as long as it was in recall. A trivial memory mentioned once is purged after 9 months of bank time, and a minor one after 3 years. Trivial memories mentioned on 4 separate occasions, minor ones on 3 and notable ones on 2 are never purged, and neither is anything at major or above.

A purge takes the whole supersession chain, and the strength and guards are read on the chain's head. Forget acts on the same chain, through the same erase path. Purge doesn't redact the source passage. Instead, a nightly sweep deletes the text of any source that no memory rests on any more, 90 days after it was ingested, and keeps its key as a tombstone. Recall log rows older than the same horizon go in that sweep.

## Considered Options

- **Never purge.** Fading already stops a memory being brought up unprompted, and storage for one user costs nothing. But it drops "unused memories are eventually forgotten" from the core idea, and verbatim text would be kept forever.
- **A fixed time spent faded out.** This needs a stored or searched "faded since" time, and it gives a trivial memory and a notable one the same grace period.
- **Purge on fading (δ = 0).** A trivial memory would be gone 15 days after its only mention.
- **Redact the passage on purge, as forget does.** Purges are frequent, so this would steadily shred passages that still back live memories. Redaction stays forget's privacy act.

## Consequences

- **Purging works against reinforcement.** Reconciliation (ADR 0005) matches claims against faded memories, so a re-mention strengthens the memory that's already there. Once a memory is purged, the same claim comes back as a new memory starting from zero. This is the main reason δ starts high.
- **Lowering δ later is safe, but raising it can't undo anything.** The next sweep catches up with a lower δ, while a purge can't be reversed. δ is set from the replay harness's purged-then-re-mentioned rate before the first production purge. It's nullable, and null means never purge.
- **Dates still ahead hold a memory back.** A memory whose head has a `valid_from` or `valid_until` whose unit hasn't ended yet isn't purged, so a day-precision appointment is held until that day ends in the source's timezone. Neither is a task until 30 days past the end of its due date's unit (the agenda's overdue window, the same boundary the agenda uses). Otherwise a far-off appointment would be gone before it happens. Events need nothing more, because their window closes at a known time (`valid_until`'s unit end, or `valid_from`'s for a point event) and the recency boost from ADR 0003 gives them a fresh run then.
- **This amends ADR 0002.** Sources are kept verbatim only while a memory rests on them or for 90 days after ingest. Re-extraction, when it's built, reaches only those sources, and it has to skip the recorded spans of purged memories.
- Call 1's saved output is dropped when its chunk commits, so the claim text doesn't survive in the chunk row after a purge or forget.

Decided in "Deletion policy: when faded memories are purged" (TIM-97) on 2026-10-01.
