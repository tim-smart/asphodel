# Logging

Asphodel logs through `tracing`. The subscriber is installed once, by
`asphodel_core::logging::init()`, and writes to stderr so a CLI's stdout stays
parseable. `ASPHODEL_LOG` sets the filter, in `tracing_subscriber::EnvFilter`
syntax, and defaults to `info`.

## The rule

**Memory content is logged only at `trace`.** Everything at `debug` and above
carries ids, counts, kinds, durations and status codes, never content.

Content means:

- memory sentences;
- source text, whether a turn or a document;
- recall queries, including the previous user message sent with a prefetch;
- entity names and aliases;
- LLM request and response bodies;
- mental model entries and the system prompt block.

This applies everywhere: the daemon, the CLI's error paths, HTTP access logs,
the replay harness and the Hermes plugin. Failed chunk rows store the error
kind and the HTTP status, never the response.

**Panic messages carry ids only.** A panic that names a memory, entity or
source names it by id. `expect` and `assert!` messages follow the same rule.

Everything derived from Tim's history stays in one private directory, and a
log line that quotes a sentence would leak it.

## Writing a log line

```rust
// Fine at any level: ids and counts.
info!(bank = %bank.id, chunks = queued, "queued chunks for extraction");

// Only at trace, because it carries a sentence.
trace!(memory = %memory.id, sentence = %memory.sentence, "reconciled claim");
```

Prefer structured fields over interpolated messages, and prefer the id of a
thing over the thing itself. When a level above `trace` seems to need the
content to be useful, log the id and read the content with `asphodel memory
show` instead.
