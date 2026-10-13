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

### With Docker

The published image is `ghcr.io/tim-smart/asphodel:latest` (Linux amd64).
It includes both models, so no separate model download is needed. `latest`
tracks `main`; use a Git commit SHA as the tag to pin a build.

Create a persistent data volume and give it to the image's non-root user
(UID/GID 65532):

```sh
docker pull ghcr.io/tim-smart/asphodel:latest
docker volume create asphodel-data
docker run --rm --user 0:0 --entrypoint /bin/sh \
  -v asphodel-data:/data ghcr.io/tim-smart/asphodel:latest \
  -c 'chown 65532:65532 /data'

docker run -d --name asphodel --restart unless-stopped \
  -p 127.0.0.1:7720:7720 \
  -v asphodel-data:/data \
  ghcr.io/tim-smart/asphodel:latest \
  serve --listen 0.0.0.0:7720 --data-dir /data
```

Open `http://127.0.0.1:7720/dashboard` to finish first-run setup. It writes
the configuration and secrets into the data volume, alongside the memory
store. Because the daemon listens off loopback inside the container, setup
creates a bearer token and shows it once. Save it for the plugin's
`ASPHODEL_TOKEN` environment variable below.

Keep the host port bound to `127.0.0.1`: anyone who can reach an
unconfigured daemon can complete setup. Hermes running on the host uses
`http://127.0.0.1:7720`; in another container, use a shared private Docker
network and the daemon's container name (`http://asphodel:7720`) instead.

### In Kubernetes

`nix build .#image` builds the OCI image: the binary, ONNX Runtime, both
models, the timezone database and BusyBox for a shell. Load it with
`docker load < result` and push it to your registry.

`deploy/kubernetes/` has the example deployment: Hermes with Asphodel as a
native sidecar, a nightly backup CronJob and a restore pod. It needs no
config file: on first start the daemon waits for setup from its dashboard,
which writes the configuration into the data volume.
`docs/operations.md` walks through it.

### On a laptop

```sh
nix build                     # or: cargo build --release, inside `nix develop`
./result/bin/asphodel models fetch
./result/bin/asphodel serve --data-dir ~/.local/share/asphodel
```

`models fetch` downloads the two models into `~/.cache/asphodel/models`
(or `ASPHODEL_MODEL_DIR`), checking each against the SHA-256 in the
manifest. The daemon never downloads anything.

The first time, open `http://127.0.0.1:7720/dashboard`. Setup asks only
which LLM to use, and you can skip that for now. It writes `asphodel.toml`
into the data dir, with any secrets in `secrets.toml`, and later starts
read them from there. Setup itself has no protection. Anyone who can reach
an unconfigured daemon can complete it, so keep the daemon private until
you've finished. To keep the tuning file elsewhere, pass `--config`, which
wins over the data dir's (`docs/operations.md`, "First-run setup" and "The
tuning file").

### The plugin

Install the plugin from its GitHub subdirectory using Hermes' shorthand
(`owner/repo/path/to/plugin`):

```sh
hermes plugins install tim-smart/asphodel/plugin --enable
```

The `/plugin` subdirectory matters: the repository root is the daemon, not
the plugin. Hermes installs it as `asphodel`, using `plugin/plugin.yaml`.
Review and accept Hermes' prompts for this custom plugin source.

If the daemon requires a bearer token (including the Docker setup above),
set it in `$HERMES_HOME/.env` (`~/.hermes/.env` by default), or in the
Hermes container's environment:

```dotenv
ASPHODEL_TOKEN=your-dashboard-token
```

Do this before memory setup so its optional history backfill can
authenticate too. Setup does not prompt for the token.

```sh
hermes memory setup
```

Pick `asphodel` in memory setup. Setup writes the plugin's config,
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
