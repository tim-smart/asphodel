# A mental model is a cache over memories

A mental model is a document an LLM keeps up to date to answer a standing question. The seeded one, "User profile", fills the always-present slot that Hermes' own `USER.md` held before the built-in memory was turned off. A model is made of entries, one sentence each, and every entry cites the memories it rests on. The LLM never rewrites the document. Each refresh sends it the current entries with their ids and the selected memories, and gets back edits: add, edit or remove. Code applies them, copies untouched entries byte for byte, and refuses an entry whose citations aren't in the refresh's input set. The memories always win. When a cited memory is forgotten, code drops the entry at once and rebuilds the prompt block, with no LLM call and no version history to scrub. When any cited memory is retracted or ended, the entry is dropped at render time and the model is refreshed. When a cited memory fades below the recall threshold it leaves the input set, and so leaves the model at the next refresh. A model can't keep a memory alive by itself. Reading or injecting it never counts as an access (ADR 0001), its entries are never embedded, extracted from or reconciled against, and the only path to strength is a reply that relies on an entry, which counts as `used` on the memories it cites.

Hindsight, which Asphodel borrows from, does it the other way round: free-form markdown refreshed by an agentic loop of up to ten steps, fifty versions of history, and a retraction prompt that asks the LLM to strip claims when a cited memory disappears.

## Considered Options

- **A profile built by code from strength-ranked memories, with no LLM.** No paraphrase and nothing to drift. Tim rejected it in "Strength model: decay, reinforcement and significance" (TIM-91), because models should answer any standing question, not only "who is the user?". The agenda does take this route for routines and undated tasks: they're lists, and a query beats a synthesis that's up to a day stale and can misstate a date.
- **Full rewrites on every refresh.** A simpler prompt. Hindsight moved away from it because each rewrite paraphrased the last one and the document drifted from the memories.
- **An agentic reflect loop per refresh**, as Hindsight does. Their evals report 90,000-token prompts that missed deadlines. A refresh here is one retrieval and one synthesis call, once a day, and skipped when the fingerprint of the selected memories hasn't changed.
- **Letting a model keep its cited memories alive.** This is the retrieval loop of ADR 0001 in another form. Instead, cited memories join the session's in-context set, so a reply that uses one is credited and a reply that doesn't isn't.

## Consequences

- Only the owner defines models, through the CLI or API, and there are no hand edits: an owner-written entry would have no citations. Standing instructions belong in Hermes' own prompt.
- Every model is injected and shares about 800 tokens of `system_prompt_block()` with the agenda. A refresh runs as a scheduled daemon job, 04:00 bank-local by default, and never inline in a request, because the plugin fetches the block with a 2 s budget.
- The block is frozen per Hermes session, so a live session keeps an entry that forget has already dropped until the session is compacted or replaced. That's accepted, alongside the forget gap in ADR 0002.
- The daemon persists which block each session holds, so the cited memories stay in that session's in-context set across plugin restarts.
- Refresh prompts and responses are never written to disk, or forget would have something to scrub after all.
- The profile takes facts and states of volatility weeks or slower. Faster states are injection's job: a model is up to a day stale and then frozen for the whole of a Discord thread.

Decided in "Mental models: synthesized documents over memories" (TIM-95) on 2026-10-01.
