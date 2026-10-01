# Asphodel

Asphodel is a memory store for the Hermes agent. It works like human memory: it extracts memories from what it's given, and they strengthen with use, fade with disuse, and eventually drop out of recall.

## Memories

**Memory**:
A single statement written as one sentence that makes sense on its own, with names in place of pronouns and dates made absolute.
_Avoid_: Fact (a fact is one kind of memory), memory unit, record, note

**Kind**:
The category of a memory, which decides how its validity behaves: fact, event, state, task or recurring.
_Avoid_: Type, fact type, category

**Fact**:
A memory that isn't expected to change, or whose change would be announced, including preferences, a job or a home. It's never given an end when extracted, though a stated start is kept.
_Avoid_: Preference (as a separate kind), world fact

**Event**:
A memory of something that happens at a particular time or over a particular span, including anything with a stated end date.
_Avoid_: Appointment, occurrence

**State**:
A memory of an ongoing condition that's expected to change without anyone announcing it, like where the user is or what they're working on.
_Avoid_: Status, situation

**Task**:
A memory of something to be done. It stays open until an event recording its completion or cancellation ends it, and it's overdue once its due date passes.
_Avoid_: Todo, reminder

**Recurring**:
A memory of something that repeats on a schedule, held as one memory rather than one per occurrence.
_Avoid_: Rule, routine, repeating event

**Significance**:
How much a memory matters on its own terms, judged at extraction as one of five levels (trivial, minor, notable, major, critical) and adjustable by the user. It's the fixed baseline beneath strength.
_Avoid_: Importance, poignancy, priority

**Volatility**:
How quickly a state is expected to go stale (hours, days, weeks, months or years). It controls how fast confidence in the state fades, and never ends it.
_Avoid_: TTL, lifespan, expected duration

## Time

**Validity window**:
The span of real-world time in which a memory holds true. Either end can be open.
_Avoid_: Expiry, TTL, lifetime

**Until-event**:
A condition, rather than a date, that ends a memory's validity ("until the project ships").
_Avoid_: Expiry condition

**Window confidence**:
How sure extraction was about the validity window it set. A low-confidence window lowers a memory's ranking but never excludes it.
_Avoid_: Date confidence

**Time precision**:
How exact each end of a validity window is: year, month, day, hour or minute.
_Avoid_: Granularity, resolution

**Phase**:
Where a memory sits relative to now (upcoming, recently past or long past), always computed and never stored.
_Avoid_: Status, lifecycle state

**Observed at**:
When the statement behind a memory was made, taken from the source's message time rather than from when it was processed.
_Avoid_: Created at, extracted at

**Due date**:
When a task should be done by. It is separate from the task's validity window.
_Avoid_: Deadline, valid until

**World time**:
Calendar time. Validity windows, phase, due dates and state confidence run on it, because the world doesn't pause when the user stops talking.
_Avoid_: Real time, wall-clock (outside the code)

**Bank time**:
The clock that strength runs on. It runs at full speed while a bank is in conversation and slows down when the bank is quiet.
_Avoid_: Activity time, psychological time, turn time

## Change

**Ended**:
A memory that was true and has stopped being true. It stays in history.
_Avoid_: Expired, closed, completed

**Retracted**:
A memory that turned out to be wrong, such as a correction or a rescheduled appointment. It's hidden from current and history answers, but kept for audit.
_Avoid_: Deleted, invalidated, cancelled

**Refined**:
A memory replaced by a more precise version of the same statement, without having been wrong.
_Avoid_: Updated, merged

**Supersession**:
The link from a memory to the memory that replaced it, whether the old one was retracted or refined.
_Avoid_: Update, overwrite

**Forget**:
A user's explicit request to erase a memory and the passage it came from, as if the store had never been told.
_Avoid_: Delete, purge, expire

**Purge**:
Asphodel removing a memory that has faded long enough to be worthless on its own.
_Avoid_: Forget, garbage collection, expire

## Strength

**Strength**:
How readily a memory comes to mind now, always computed from its accesses, significance and the clock, and never stored.
_Avoid_: Activation (outside the formula), score, weight, heat

**Fading**:
The loss of strength over time without use.
_Avoid_: Decay (outside the formula), forgetting, expiry

**Access**:
An event that counts towards strength: a memory being created, used, mentioned again or confirmed.
_Avoid_: Hit, retrieval, view, read

**Used**:
The kind of access where a reply actually relied on a recalled memory, as opposed to just being shown it.
_Avoid_: Retrieved, returned, injected

**Mentioned again**:
The kind of access where the user or a document independently states something already remembered. A later version of the same document repeating itself doesn't count.
_Avoid_: Duplicate, repeat, re-extraction

**Confirmed**:
The kind of access where the user says a memory is still right.
_Avoid_: Verified, acknowledged

**Recent use**:
The part of strength that comes from recent accesses and fades with disuse. It starts over when a memory's validity window closes.
_Avoid_: Retrieval strength, base level (outside the formula)

**Lasting floor**:
The part of strength built up by accesses on separate occasions. It never goes down, and a memory whose floor is above the recall threshold can't fade out.
_Avoid_: Storage strength (outside the formula), permanence

**Recall threshold**:
The strength below which a memory is no longer injected automatically. Explicit recall can still find it.
_Avoid_: Cutoff, expiry threshold

**Faded out**:
A memory whose strength is below the recall threshold. It still exists and can still be recalled when asked for.
_Avoid_: Forgotten, expired, dormant, deleted

**State confidence**:
How likely a state still holds, which fades with its age at a rate set by its volatility. It only lowers ranking.
_Avoid_: Freshness, staleness, window confidence (that's about the window's dates)

## Recall

**Recall**:
One retrieval of memories for a query, together with a record of what came back. Being recalled never counts as an access by itself.
_Avoid_: Search (for the whole operation), lookup, access

**Injection**:
Recalled memories placed automatically into the agent's prompt before a turn.
_Avoid_: Prefetch (that's the Hermes hook), context stuffing

**Mental model**:
A document that an LLM writes and keeps up to date from memories, to answer a standing question such as "who is the user?".
_Avoid_: Observation, reflection, summary, profile (a profile is one mental model)

## Sources and organisation

**Source**:
The exact input a memory came from, either a conversation turn or a document, kept verbatim.
_Avoid_: Episode, chunk, input, raw message

**Turn**:
A source made up of one user message and the assistant reply to it, as Hermes hands them over.
_Avoid_: Message, exchange

**Document**:
A source of plain text or markdown sent through the API, always with a reference date.
_Avoid_: File, attachment

**Reference date**:
The date that relative times in a source ("tomorrow", "last week") are resolved against.
_Avoid_: Event date, context date

**Extraction**:
Turning a source into memories and access events using the LLM, one chunk at a time.
_Avoid_: Retain, ingestion (ingestion is receiving the source, before extraction)

**Chunk**:
The unit a source is extracted in: a whole turn, or a section of a document, split further only when it's too long.
_Avoid_: Segment, passage, piece

**Claim**:
A statement extraction has found in a chunk but not yet reconciled. It becomes a new memory, an access on an existing memory, or nothing.
_Avoid_: Fact (that's a kind), candidate, extracted fact

**Reconciliation**:
The step of extraction that compares claims with the closest existing memories and decides whether each is new, mentioned again, confirmed, or ends, retracts or refines an existing memory.
_Avoid_: Deduplication, merging, consolidation

**Entity**:
A person, place, organisation or thing that memories are about, recognised under several aliases. Every bank starts with the user and the assistant as entities.
_Avoid_: Node, tag, subject

**Bank**:
An isolated set of memories, entities and sources. Nothing refers across banks.
_Avoid_: Namespace, tenant, collection, profile

**Store**:
One Asphodel installation, which holds all of its banks.
_Avoid_: Database, instance
