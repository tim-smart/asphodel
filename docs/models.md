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

## The LLM

One OpenAI-compatible `chat/completions` endpoint serves extraction,
reconciliation and refresh. `llm.endpoint` and `llm.model` come from the
tuning file; the API key comes from `ASPHODEL_LLM_API_KEY` only.

Every call asks for structured output (`response_format.json_schema`,
strict) at temperature 0. A reply fenced in ```` ```json ```` is unwrapped.
Errors carry statuses and sizes, never the prompt or the reply, so logs hold
no memory content. `FakeLlm` hands out scripted replies and records the
requests it was given.
