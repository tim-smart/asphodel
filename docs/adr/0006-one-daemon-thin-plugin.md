# One daemon, and a thin Hermes plugin

Asphodel runs as a single long-lived daemon that owns the store, the embedding model, the reranker and the extraction worker. The Hermes plugin is a small Python client with no dependencies, and it talks to the daemon over HTTP on loopback. Hermes keeps many provider instances alive for one profile at once: every CLI process, up to 128 cached gateway agents, and cron jobs. Only a single process can hold the extraction worker for each bank, load the models once, and own the SQLite file.

## Considered Options

- **A PyO3 module loaded inside Hermes.** There's no daemon to run, but every Hermes process loads the models and opens the same database. One extraction worker per bank would need cross-process leader election. We'd also have to build wheels for every platform Hermes ships on (CPython 3.14 on glibc, musl and Android).
- **A sidecar per Hermes process.** It has the same problems as PyO3, plus a process to manage.
- **Spawning the daemon from the plugin on demand**, as OpenViking and cognee do. In a k8s pod the binary isn't in the Hermes container, and on a laptop many instances race for the data-dir lock. A supervisor runs the daemon instead: a k8s sidecar, systemd or nix.

## Consequences

- When the daemon is down, Hermes carries on without memory. Turns are spooled and replayed later, so ingest has to stay idempotent.
- Every operator command is an HTTP client of the daemon. Nothing else opens the database.
- Hermes' own backup doesn't cover Asphodel's data.
