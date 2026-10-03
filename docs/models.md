# Models

Asphodel runs two local models and talks to one LLM. Each sits behind a
trait in `asphodel_core::models`, so tests and the replay harness can stand
in for it.

## The local models

| Role | Model id | Source | Size |
|---|---|---|---|
| Embedding | `bge-small-en-v1.5:int8` | `Xenova/bge-small-en-v1.5` at `ea104da`, `onnx/model_quantized.onnx` | 34 MB |
| Reranker | `jina-reranker-v1-turbo-en:int8` | `jinaai/jina-reranker-v1-turbo-en` at `b8c14f4`, `onnx/model_quantized.onnx` | 38 MB |

The model id is the exact string, quantisation included. It keys the
floors in the tuning file (`reconcile.embedding_floors` and
`injection.reranker_floors`) and is what a bank records when it is created.
A daemon won't start without a floor for each model it runs (ADR 0009).

The manifest in `crates/asphodel-core/src/models/manifest.rs` pins each
model to a Hugging Face revision and lists the SHA-256 of every file. Both
models are English-only.

### The model dir

Models live under one directory, resolved in this order:

1. `--model-dir` or `ASPHODEL_MODEL_DIR`;
2. `$XDG_CACHE_HOME/asphodel/models`;
3. `$HOME/.cache/asphodel/models`.

Each model has its own subdirectory (`bge-small-en-v1.5-int8`,
`jina-reranker-v1-turbo-en-int8`) holding `model.onnx`, `tokenizer.json`,
`config.json`, `special_tokens_map.json` and `tokenizer_config.json`.

`asphodel models fetch` fills the dir from the manifest. It skips files
already present with the right checksum, verifies every download before
writing it, and writes through a temp file and a rename, so a failed run
never leaves a partial file and the next run resumes.

The daemon never downloads. At startup it checks that every file is present
and matches its checksum before ONNX Runtime is touched, and a missing or
corrupt file stops it with a message naming the path. For a deployment, bake
the dir into the image.

### ONNX Runtime

ONNX Runtime is loaded at runtime from `ORT_DYLIB_PATH` (ort's
`load-dynamic`). The nix shell points it at nixpkgs' `onnxruntime`.
`--onnx-threads` / `ASPHODEL_ONNX_THREADS` pins the intra-op thread count,
which replay does for determinism.

Texts are embedded one at a time. The graph is dynamically quantised, so a
text's vector would otherwise depend on what it was batched with, and the
reconcile floor needs the same text to give the same vector.

### The fakes

`FakeEmbedder` and `FakeReranker` are deterministic stand-ins: a hashed bag
of words, and the count of query words found in the document. They have
their own ids (`fake-embedder:v1`, `fake-reranker:v1`), so a floor for the
real models never covers them. The scripted replay scenarios run on them
under `cargo test`.

`ASPHODEL_MODELS=fake` runs `asphodel serve` on them. It is for tests,
environment only, and the resolved config shows `models.fake = true`.

### Changing the embedding model

A bank records the embedding model it was created under and is served with
that model, not the daemon's, until `asphodel reembed --bank <bank>` moves
it (ADR 0010). During a change the image carries both models: the daemon
runs the new one and keeps the old one for banks still recorded under it
(`Service::with_previous_embedder`). The manifest lists one embedding
model today, so a change adds the new model to it and keeps the old one as
the previous embedder until every bank has moved. Only the fakes load a
previous embedder so far. For the ONNX models, `Models::load` takes the
manifest's first two entries as the embedder and reranker, so the first
release that changes the model has to teach the loader and `serve` about
the previous one (`docs/operations.md`, "Changing the embedding model").

A bank recorded under a model the daemon doesn't carry is refused rather
than served with another model, whose vectors aren't comparable with its
own: recall and prefetch answer 503, its chunks wait on the queue without
counting a failure, and its mental models' refreshes fail. The daemon warns
about it at startup and `asphodel status` asks for attention. Re-embedding
needs only the daemon's model, so `asphodel reembed --bank` is how such a
bank recovers.

The re-embed is a daemon job, and its tables are the schema's version 10
migration. It embeds the bank's memories with the new model into a side
table, in rowid order, recording how far it got, so a
restart resumes it rather than starting over. Extraction carries on with
the recorded model meanwhile. The swap waits for the chunk in flight, embeds
whatever arrived since, and in one transaction replaces the bank's vectors,
records the new model and drops the side table's rows. `asphodel reembed`
follows the job to the swap, or returns at once with `--no-wait`; running
it again shows where the job stands.

The new model's reconcile floor (`reconcile.embedding_floors`) is
calibrated in replay before the deploy, and the daemon won't start without
it. A reranker-only change needs only its gate floor.

`ASPHODEL_MODELS=fake-v2` serves with a second fake embedder,
`fake-embedder:v2`, and carries `fake-embedder:v1` for banks recorded under
it, so a re-embed can be driven on the fakes in tests.

## The LLM

One LLM serves extraction, reconciliation and refresh. `llm.model` comes
from the tuning file in both modes and is required: the floors are
calibrated against one model.

Every call asks for structured output and expects JSON back; a reply
fenced in ```` ```json ```` is unwrapped. Errors carry statuses and sizes,
never the prompt or the reply, so logs hold no memory content. `FakeLlm`
hands out scripted replies and records the requests it was given. The
cassette records the logical request (template, prompts, schema), never
the wire body or headers, so a recording made in one mode replays in the
other.

`ASPHODEL_LLM_SCRIPT=<file>` runs the daemon's extraction workers on a
`FakeLlm` that plays the file's steps in order, one per call: a JSON array
of `{"reply": <json>}` or `{"fail": "<kind>"}` steps, each with an optional
`delay_ms`. The kinds are `transport`, `timeout`, `status` (with `status`,
default 500), `no_content`, `not_json`, `refused`, `login_required` and
`usage_limited` (with `resets_at`). Once the script runs out, every call
fails with `no_content`. It is for integration tests, environment only, and
the resolved config shows `fake_llm = true`.

### `auth = "api_key"` (the default)

Any OpenAI-compatible `chat/completions` endpoint. `llm.endpoint` is
required and the key comes from `ASPHODEL_LLM_API_KEY` only. Requests use
`response_format.json_schema` (strict) at temperature 0.

### `auth = "chatgpt"`

A ChatGPT subscription over the Codex backend
(`https://chatgpt.com/backend-api/codex`, the default `llm.endpoint`).
Requests go to `/responses`, streamed, with `text.format` for the schema
and no temperature (the GPT-5 family rejects it). The protocol was read
from `openai/codex` at `6b4daaf`; it is undocumented and OpenAI can change
it, which is why `api_key` stays the default.

Subscription usage windows will throttle the real-history import.
A large backfill could take days. Keep `llm.model` pinned and calibrate
against that exact model; subscription mode does not change ADR 0009's
single-model calibration requirement.

**Logging in.** `asphodel llm login --data-dir <dir>` runs the device-code
flow: it prints a URL and a one-time code, waits for approval, and writes
`<dir>/llm-tokens.json` with mode 0600. The daemon reads that file before
every call, so a login while it runs takes effect without a restart.

Every write of the token file goes through an exclusive `flock` on
`<dir>/llm-tokens.lock`, which holds nothing else. A refresh holds it from
its re-read of the file to its save, so a login, from the CLI or anywhere
else, waits for a refresh in flight and then replaces what it wrote rather
than being overwritten by it. The login takes the lock only for its final
save, never while it waits for approval. If the file was cleared before
a refresh takes the lock, that counts as a logout: the refresh requires
a new login and never recreates the file. Writes go to a new temp file,
created exclusively under a fresh name, and a rename; nothing already in
the data dir is followed or overwritten.

Asphodel never reads `~/.codex/auth.json` and the file must not be copied
from it. Refresh tokens are single-use: if Asphodel and the Codex CLI
shared one token chain, whichever refreshed first would log the other out.

**Refresh.** The daemon refreshes the access token when its expiry is
within five minutes, or after a 401, once. The rotated tokens are written
before the retried request goes out, and concurrent calls share one
refresh, across clients and processes on the same data dir. An issuer 408, 429 or 5xx is retryable and keeps the token file
unchanged. A second 401, or a refresh the issuer rejects, surfaces as "run
`asphodel llm login`"; the old file is kept until the next login replaces
it.

**Usage limits.** A 429 whose body says `usage_limit_reached` carries the
time the window resets. The client returns `UsageLimited { resets_at }`,
which is not a retry: extraction holds the queue until then.

**Secrets.** Tokens never appear in `Debug` output, logs, errors, the
resolved config or cassettes. A failed or incomplete response keeps only a
code from the fixed list codex-rs knows; anything else the backend puts
there becomes `unknown`, since it could echo what it was sent. The resolved config's `llm` section shows
the mode, the token file's path and whether a login is present.

**Unverified against the real backend.** Two details could not be
confirmed from the codex-rs source and wait for the ignored real-backend
test: whether the backend accepts an `originator` other than
`codex_cli_rs` (Asphodel sends `asphodel`), and whether an `OpenAI-Beta`
header is needed (codex-rs sends one only on its websocket transport, so
Asphodel sends none).

**Settings that aren't exposed.** `ASPHODEL_LLM_API_KEY` set together with
`auth = "chatgpt"` stops the daemon. `ASPHODEL_LLM_ISSUER` points the
login at another issuer; it exists for tests and is not in `--help`. `llm.reasoning_effort` (for example `"low"`) is sent as `reasoning.effort`, or `reasoning_effort` in `api_key` mode; unset leaves it to the backend's default. A cassette records the effort with the model, so recordings at another effort are never replayed.
