# Configuration is daemon-wide, and deletion inputs are fingerprinted

Every setting sits in one of five places:

- Fixed in code.
- A daemon-wide `Tuning` struct, read from an optional TOML file.
- Deployment flags and environment variables.
- The bank's identity: the owner, the assistant and the timezone.
- Flags on `asphodel replay` and `asphodel bench`.

A bank never overrides a tuning value. With one δ and one quiet-time rate, a lifetime measured in bank time and the purge table mean the same thing in every bank. Bank settings and tuning never share a key, so there's nothing to merge. The replay overrides file has exactly the shape of `Tuning`, and replay layers code defaults, then the production file, then the overrides.

Every constant that the strength model was calibrated with is fixed in code, apart from the quiet-time rate:

- S, τ, a, c, g and n0;
- the floor spacing;
- the access weights;
- the weight of the synthetic access at window close;
- bank time's full-speed window;
- the significance levels;
- the volatility rates.

Strength is never stored, so editing one of these on a live store changes every memory at once. The boundary between faded and fading is τ by definition, and only the strong cut-off is tunable. The reranker deadline is fixed too. It is the first link in a chain of timeouts: 1.5 s in the daemon, 3 s in the plugin and 8 s in Hermes.

Tuning changes take effect on restart. Unknown keys and out-of-range values stop the daemon from starting. A missing floor for a configured model does too. Floors are keyed by the exact model string, including quantisation. `GET /v1/config` returns the resolved config, the fixed constants and the purge-pause state, with secrets redacted.

## The deletion fingerprint

Purge and the source sweep are pure functions of the access log, the clock and a few settings. A change to `quiet_rate` or `delta` recomputes every memory, and the next sweep acts on the new numbers. Fixing S and τ in code doesn't close that hole on its own.

The store keeps a hash of the values that decide an irreversible deletion:

- the fixed strength constants;
- `quiet_rate` and `delta`;
- `agenda.overdue_days`, which is also the guard on purging tasks;
- `purge.source_horizon_days`.

If the hash at startup differs from the stored one, purge and the sweep of sources and recall rows pause, as if δ were null, until an operator acknowledges the change. On first start the hash is just recorded.

## Considered options

- **Per-bank tuning, merged over daemon defaults.** It needs merge rules, it makes strength and purge mean different things in different banks, and it multiplies calibration. There's one user, so nothing needs it.
- **Exposing S, τ and n0 as tuning.** A typo could purge a large part of the store, and a purge can't be undone. Changing them reopens the strength model decision.
- **Hot reload, or `PUT /v1/config`.** Changing δ is irreversible in practice, so it shouldn't be one API call away. Restarting the pod is cheap.
- **A separate LLM model for reconciliation and refresh.** The extraction calibration is against one model. Call 1 is the expensive call and also the one that needs quality. Splitting would triple the calibration and cassette variants to save money on the cheap calls.
- **Falling back when a model has no floor.** A missing gate floor would flood injection, and a missing reconcile floor would skip reconciliation. Both are silent regressions.

## Consequences

- The fingerprint hashes values, not the git SHA, so a deploy that doesn't change them doesn't pause purge.
- Replay never pauses, because the harness is where δ gets changed on purpose.
- Forget never pauses, because it's the owner's explicit act and not a sweep.
- Moving the agenda's overdue window pauses purge until it's acknowledged. That's intended. Reusing the window as the task guard (ADR 0008) keeps one constant, at the cost of this coupling.
- The acknowledgement mechanism, and whether the sweep reports a dry-run count first, are decided with operations.
- The timeouts and the spool and breaker settings in the plugin are fixed, and its config schema stays as ADR 0006 left it.

## Amendment: the LLM's auth mode (2026-10-01)

The LLM can be reached with an API key or with a ChatGPT subscription.
This adds one tuning key and one file; the decision above stands.

- `[llm] auth = "api_key" | "chatgpt"`, default `api_key`. It's tuning,
  not deployment: it changes the wire format, not where the daemon runs.
- The API key stays environment-only. In `chatgpt` mode a set
  `ASPHODEL_LLM_API_KEY` stops the daemon rather than being ignored.
- The subscription's tokens live in one file under the data dir
  (`llm-tokens.json`, mode 0600), written by `asphodel llm login` and
  refreshed by the daemon. They never appear in the tuning file, the
  resolved config, `Debug` output, logs or cassettes. The resolved config
  shows the mode, the file's path and whether a login is present.
- `llm.model` stays required in both modes. The floors are calibrated
  against one model, and the subscription doesn't choose it.
- In `chatgpt` mode `llm.endpoint` defaults to the Codex backend; an
  explicit endpoint still wins.
- A usage limit on the subscription is not a failure. The daemon reports
  when the window resets, and the extraction queue holds until then
  instead of failing chunks or counting retries.
