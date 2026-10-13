# Tests

## Running tests

Use `cargo nextest run` to run all tests, not `cargo test`.

```
TMPDIR=/dev/shm nix develop -c cargo nextest run
```

macOS has no `/dev/shm`. Use a short directory such as `/tmp/ash`: the
daemon tests listen on Unix sockets, whose paths have a length limit, so a
long `TMPDIR` makes them time out.

## Writing tests

Test behavior through public seams only:

- `Service` and the other public core API it hands out
- the HTTP API (`asphodel serve`)
- public clients tested against stub servers for external HTTP contracts
- the CLI
- the plugin's public surface (the Hermes provider and its tools)
- replay scenarios in `scenarios/*.toml`

If a test has to reach past these seams, either the behavior isn't worth
pinning or the seam is missing. Add the seam or drop the test. `pub` on a
module is not a seam by itself.

Don't assert on prompt wording, template version numbers, tuning defaults,
log or status-message text, row ids, or table layout. These change on
purpose, and a test that breaks when they do is noise. Assert the behavior
they produce instead: "a claim made under different rules is not reused",
not "the template version is 7".

A bug fix gets a regression test only when the bug is visible at a public
seam. Extend an existing test or scenario before adding a new `#[test]`.

Raw SQL in tests (`store.connection()` and friends) is allowed only to:

- prove data is gone after erase, forget, delete or other privacy paths
- set up a state the API can't reach, such as a corrupt row or a
  pre-migration schema

Everywhere else, read through the public API. If no public read exists,
adding a small one is fine; call it out in the PR.
