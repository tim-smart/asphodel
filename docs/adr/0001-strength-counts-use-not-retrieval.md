# Strength counts use, not retrieval

A memory's strength comes only from accesses of four kinds: `created`, `used`, `mentioned_again` and `confirmed`. Being returned by a search, injected into the prompt, returned by the recall tool, or viewed by an operator never strengthens a memory by itself. Systems that count search hits (MemoryBank, MemoryOS, Mem0's decay) strengthen whatever the embedding happened to match, and then inject those same memories again. That creates a loop in which frequently recalled memories keep getting stronger regardless of whether they matter. Hermes gives no signal that an injected memory was actually used, so "used" is judged by the extraction call on the finished turn, with embedding similarity as the fallback. Every recall is still recorded, but in a separate log that has no effect on strength.

## Considered Options

- **Count every search hit or injection.** This is the simplest option and what most LLM memory systems do. We rejected it because of the loop described above.
- **Count only mentions by the user.** This has no loop at all, but it ignores the assistant using a memory, which is the main way memories get exercised.

## Consequences

- Each extraction call has to receive every memory injected earlier in the session that's still in the context window, because Hermes replays injections on every later turn.
- Only one access per memory per turn is written, so the same use isn't counted twice.
- If the assistant repeats an injected memory, that counts as `used`, never as `mentioned_again`.
