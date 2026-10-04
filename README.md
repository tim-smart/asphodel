# Asphodel

Brain-like memory for the [Hermes agent](https://github.com/NousResearch/hermes-agent).
Asphodel extracts memories from conversations and documents. They
strengthen with use, the significant ones stick, and the unused ones fade
out of recall. `CONTEXT.md` is the glossary.

It comes in two parts:

- **The daemon**, `asphodel serve`. One process owns the SQLite store, the
  embedding model, the reranker and the extraction worker. Every other
  `asphodel` subcommand is an HTTP client of it.
- **The Hermes plugin**, in `plugin/`. A memory provider with no
  dependencies outside the standard library, which talks to the daemon over
  HTTP. Hermes keeps many provider instances alive for one profile, and they
  all share the one daemon.

Nothing starts the daemon on demand. A supervisor runs it: a Kubernetes
native sidecar next to Hermes, a systemd unit, or anything else that
restarts it.

## Running it

### In Kubernetes

`nix build .#image` builds the OCI image: the binary, ONNX Runtime, both
models and the timezone database, with no shell. Load it with
`docker load < result` and push it to your registry.

`deploy/kubernetes/` has the example deployment: Hermes with Asphodel as a
native sidecar, a nightly backup CronJob and a restore pod.
`docs/operations.md` walks through it.

### On a laptop

```sh
nix build                     # or: cargo build --release, inside `nix develop`
./result/bin/asphodel models fetch
./result/bin/asphodel serve --data-dir ~/.local/share/asphodel --config asphodel.toml
```

`models fetch` downloads the two models into `~/.cache/asphodel/models`
(or `ASPHODEL_MODEL_DIR`), checking each against the SHA-256 in the
manifest. The daemon never downloads anything. The tuning file needs the
LLM and a floor for each model (`docs/operations.md`, "The tuning file").

### The plugin

Copy `plugin/` to `$HERMES_HOME/plugins/asphodel/`, then run
`hermes memory setup` and pick `asphodel`. Setup writes the plugin's config,
makes Asphodel the memory provider and turns Hermes' built-in memory off.
`docs/operations.md` has the details, including the dashboard path that
skips that last step.

## Documentation

- `docs/operations.md`: deploying, setting up a bank, every operator
  command, backup and restore, forgetting, deleting a bank, re-embedding.
- `docs/models.md`: the local models, the model dir and the LLM.
- `docs/upgrading.md`: what each schema migration needs from you.
- `docs/logging.md`: what's logged, and the rule that keeps memory content
  out of logs.
- `docs/replay.md`: the replay harness, and calibrating the floors.
- `docs/mental-model-answer.md`: why a mental model is one prose answer
  with one set of citations, and what that trades away.
- `docs/hermes-data-evaluation.md`: evaluating against real Hermes history
  with a local agent, and the feedback to send back.

## Developing

```sh
nix develop -c cargo fmt --check
nix develop -c cargo clippy --all-targets -- -D warnings
nix develop -c cargo nextest run
nix develop -c cargo test --doc
nix develop -c python -m pytest plugin/tests
```

`cargo test` works too, but it runs the test binaries one at a time and
takes several times as long; nextest runs them side by side. nextest
skips doc tests, hence the second line. The replay tests write a lot of SQLite to
the temp dir, so on a busy disk `TMPDIR=/dev/shm` (Linux) can make the run
several times faster.

The dev shell provides the toolchain, cargo-nextest included, and points
`ORT_DYLIB_PATH` at nixpkgs' ONNX Runtime. Tests run on deterministic fake
models and never touch the network.
