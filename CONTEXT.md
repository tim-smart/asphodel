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
How much a memory matters on its own terms, judged at extraction as one of five levels (trivial, minor, notable, major, critical) and adjustable by the owner. It's the fixed baseline beneath strength.
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
Where a memory sits relative to now (upcoming, current, overdue, recently past or long past), always computed and never stored. Recently past lasts 30 days after the validity window closes.
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
A denied memory, one the user says never happened or isn't true, is retracted too. The only difference is what happens to anything it had ended: a correction still ends it, and a denial opens it again.
The owner can also retract a memory directly, from the CLI or the dashboard. That's a denial with nothing replacing it.
_Avoid_: Deleted, invalidated, cancelled

**Refined**:
A memory replaced by a more precise version of the same statement, without having been wrong.
_Avoid_: Updated, merged

**Supersession**:
The link from a memory to the memory that replaced it, whether the old one was retracted or refined.
_Avoid_: Update, overwrite

**Forget**:
The owner's explicit request to erase a memory, every earlier and later version of it, and the passages they came from, as if the store had never been told.
_Avoid_: Delete, purge, expire

**Kept**:
Said of a memory the owner has asked to remember, which sets its significance as high as it goes so that it never fades. The owner can take it back, and the memory returns to the significance extraction gave it.
_Avoid_: Pinned, starred, saved

**Purge**:
Asphodel removing a memory, with every version of it, once its strength has fallen well below the recall threshold. The passage it came from stays until nothing else rests on its source.
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
The kind of access where a reply actually relied on a recalled memory, whether it was injected or read in a mental model, as opposed to just being shown it.
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
The strength below which a memory is no longer injected by relevance, listed among the agenda's routines or undated tasks, or fed to a mental model. Explicit recall can still find it, and the dated agenda can still list it.
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
Recalled memories placed automatically into the agent's prompt before a turn, chosen by relevance to what the user just said.
_Avoid_: Prefetch (that's the Hermes hook), context stuffing

**Agenda**:
The upcoming events, tasks due soon and recently overdue tasks, chosen by world time alone, plus a few routines and undated open tasks chosen by strength, kept in the agent's prompt. It isn't recall, and the recall threshold doesn't apply to its dated items.
_Avoid_: Calendar, schedule, reminders

**Mental model**:
A document that an LLM keeps up to date from memories, to answer a standing question such as "who is the user?". It's one answer, a heading and a paragraph for each part of the question, with one set of citations: the memories it rests on. The memories always win: while a cited memory is retracted or ended the model isn't shown at all, and forgetting one blanks the answer until the next refresh writes it again.
_Avoid_: Observation, reflection, summary, profile (a profile is one mental model)

**Answer**:
A mental model's text: connected prose, a heading and a paragraph per facet, with one set of citations for the whole. Only a refresh writes it, replacing it whole, and forget blanks it whole. A token budget may cut trailing sentences off the end, at the refresh or in the prompt block, and what's left still cites the whole set; no sentence in it is cited or edited on its own.
_Avoid_: Entry (the old unit, one cited sentence), summary, line, bullet, block

**Facet**:
One part of a mental model's question, with a heading and a recall query of its own. The seeded profile's facets are built in; any other question is planned once by an LLM call.
_Avoid_: Section (that's how a facet renders), sub-question, topic

**Refresh**:
Bringing a mental model up to date: one retrieval for each facet of its question, then one LLM call that writes the whole answer again with the previous one in view. It runs shortly after a conversation adds or changes something the model would care about, and once a day besides, and only when the selected memories have changed. A correction or forget of a cited memory asks for one urgently, ahead of the usual minimum interval.
_Avoid_: Reflect, consolidation, rebuild, regenerate

**Explain**:
Running a query through recall or injection to see what each candidate scored and why it was or wasn't returned, without it counting as a recall: nothing is logged, nothing is accessed, and no session is read or changed.
_Avoid_: Test recall, dry run, debug recall

**In context**:
Said of a memory the agent can already see this session: injected earlier, listed in the agenda, returned by the recall tool, or cited by an injected mental model. It isn't injected again, and extraction checks it for use.
_Avoid_: Loaded, seen, in the context window

## Sources and organisation

**Source**:
The exact input a memory came from, either a conversation turn or a document, kept verbatim. Once no memory rests on it and it's 90 days old, its text is swept and only its key is kept, so it can't be ingested again.
_Avoid_: Episode, chunk, input, raw message

**Tombstone**:
What's left of a source, or of a passage in one, once its text is gone: the key and content hash, with no text. It stops the same input being ingested and extracted again. An edited section of a document, or a later turn that repeats the same words, is new input, not the same input. A turn that asked to forget is stored as a tombstone from the start.
_Avoid_: Marker, stub, deleted row

**Sweep**:
The nightly job that purges memories and deletes the text of sources, failed chunks and recall rows past their 90-day horizon. It records only counts, and it pauses when the settings that decide deletion change, until an operator acknowledges them.
_Avoid_: Garbage collection, cleanup, vacuum

**Turn**:
A source made up of one message and the assistant's reply to it, as Hermes hands them over. The message may come from the owner or from anyone else in the conversation.
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
The step of extraction that compares claims with the closest existing memories and decides whether each is new, mentioned again, confirmed, or ends, retracts, denies or refines an existing memory.
_Avoid_: Deduplication, merging, consolidation

**Entity**:
A person, place, organisation or thing that memories are about, recognised under several aliases. Every bank starts with the user and the assistant as entities.
_Avoid_: Node, tag, subject

**Speaker**:
Whoever wrote the message in a turn. A speaker's "I" and "me" resolve to them: the owner's to the user entity, and anyone else's to an entity of their own.
_Avoid_: Author, sender, user (the user is one speaker)

**Owner**:
The one person a bank belongs to, who is the user entity on every platform they speak from. Only the owner can forget, keep or unkeep a memory.
_Avoid_: Account, admin, operator

**Bank**:
An isolated set of memories, entities and sources, one for each Hermes profile. Nothing refers across banks.
_Avoid_: Namespace, tenant, collection, profile

**Store**:
One Asphodel daemon and its data, which holds all of its banks.
_Avoid_: Database, instance
