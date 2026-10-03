// The v1 extras: each bank's counts on the banks page, the daemon's
// attention lines as a banner, failed chunks with retry, and the purge pause
// with its acknowledgement.

import { test } from "node:test";
import assert from "node:assert/strict";

import { FakeDaemon, TOKEN, ids } from "./fake-daemon.js";
import {
  click,
  findButton,
  findDialog,
  findRow,
  findText,
  open,
  queryButton,
  queryDialog,
  queryText,
  showsCount,
  stays,
  textOf,
  waitFor,
} from "./support.js";

function paused(daemon) {
  daemon.state.status.purge = { state: "paused", stored: "e0e0e0" };
  daemon.state.status.attention = ["purge is paused until the new deletion settings are acknowledged"];
  daemon.state.plan = {
    pause: { state: "paused", stored: "e0e0e0" },
    current: "f1f1f1",
    changed: ["purge.delta"],
    memories: 23,
    sources: 4,
    failed_chunks: 1,
    recalls: 9,
  };
}

test("the banks page shows each bank's memory, source and chunk counts", async (t) => {
  const page = await open(t, new FakeDaemon(), { token: TOKEN });

  const main = await findRow(page.root, /\bmain\b/);
  for (const [label, count] of [
    ["live", 112],
    ["retracted", 7],
    ["kept", 4],
    ["documents?", 5],
    ["queued", 2],
    ["failed", 3],
  ]) {
    assert.ok(showsCount(main, label, count), `${label} ${count} in: ${textOf(main)}`);
  }
  const work = await findRow(page.root, /\bwork\b/);
  assert.ok(showsCount(work, "live", 31), textOf(work));
});

test("what needs attention shows as a banner over the page", async (t) => {
  const daemon = new FakeDaemon();
  daemon.state.status.attention = [
    "bank main: 3 chunks failed extraction",
    "the last backup is more than 7 days old",
  ];
  const page = await open(t, daemon, { hash: "#/banks/main/memories", token: TOKEN });

  const banner = await waitFor(
    () =>
      [...page.document.querySelectorAll('[role="alert"], [role="status"]')].find((el) =>
        textOf(el).includes("bank main: 3 chunks failed extraction"),
      ),
    "the attention banner",
  );
  assert.ok(textOf(banner).includes("the last backup is more than 7 days old"), textOf(banner));
});

test("failed chunks show their error and link their source, and retry puts one back", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: "#/banks/main/chunks", token: TOKEN });

  const row = await findRow(page.root, "llm_timeout");
  assert.ok(row.querySelector(`a[href*="${ids.documentV2}"]`), row.innerHTML);
  click(queryButton(row, /retry/i));

  const sent = await waitFor(() => daemon.calls("POST", "/v1/banks/main/chunks/retry")[0], "POST retry");
  assert.deepEqual(sent.body.chunks, [ids.chunkWork]);
  await waitFor(() => !queryText(page.root, "llm_timeout"), "the retried chunk to leave the list");
  assert.ok(queryText(page.root, "malformed"), "the other failure stays");
});

test("retry all puts every failed chunk back", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: "#/banks/main/chunks", token: TOKEN });
  await findText(page.root, "malformed");

  click(await findButton(page.root, /retry all/i));

  const sent = await waitFor(() => daemon.calls("POST", "/v1/banks/main/chunks/retry")[0], "POST retry");
  assert.equal(sent.body?.chunks ?? null, null);
  await waitFor(
    () => !queryText(page.root, "llm_timeout") && !queryText(page.root, "malformed"),
    "the list to empty",
  );
});

test("a paused purge shows what changed and what the sweep would delete", async (t) => {
  const daemon = new FakeDaemon();
  paused(daemon);
  const page = await open(t, daemon, { token: TOKEN });

  await findText(page.root, "purge.delta");
  assert.ok(showsCount(page.root, "memor(y|ies)", 23), textOf(page.root));
  assert.ok(showsCount(page.root, "sources?", 4), textOf(page.root));
  assert.ok(showsCount(page.root, "recalls?", 9), textOf(page.root));
});

test("acknowledging the new deletion settings asks first, and cancelling sends nothing", async (t) => {
  const daemon = new FakeDaemon();
  paused(daemon);
  const page = await open(t, daemon, { token: TOKEN });

  click(await findButton(page.root, /acknowledge|resume/i));
  const dialog = await findDialog(page.document);
  assert.ok(showsCount(dialog, "memor(y|ies)", 23), textOf(dialog));
  click(await findButton(dialog, /cancel/i));
  await waitFor(() => !queryDialog(page.document), "the confirmation to close");

  await stays(() => daemon.calls("POST").length === 0, "nothing sent");
});

test("a confirmed acknowledgement quotes this daemon's fingerprint and resumes purge", async (t) => {
  const daemon = new FakeDaemon();
  paused(daemon);
  const page = await open(t, daemon, { token: TOKEN });

  click(await findButton(page.root, /acknowledge|resume/i));
  const dialog = await findDialog(page.document);
  const button = [...dialog.querySelectorAll("button")].find(
    (b) => /acknowledge|resume/i.test(textOf(b)) && !/cancel/i.test(textOf(b)),
  );
  click(button);

  const sent = await waitFor(() => daemon.calls("POST", "/v1/purge/ack")[0], "POST purge ack");
  assert.deepEqual(sent.body, { hash: "f1f1f1" });
  await waitFor(() => !queryText(page.root, "purge.delta"), "the purge card to go");
});

// The daemon's own wording for a bank's failed chunks.
const FAILED_LINE = "main: 3 failed chunks; see `asphodel chunks --bank main --failed`";

function banner(page, line) {
  return waitFor(
    () =>
      [...page.document.querySelectorAll('[role="alert"], [role="status"]')].find((el) =>
        textOf(el).includes(line),
      ),
    `a banner with "${line}"`,
  );
}

test("a bank's failed chunks in the banner link to that bank's ingestion page, from any bank", async (t) => {
  const daemon = new FakeDaemon();
  daemon.state.status.attention = [FAILED_LINE];
  const page = await open(t, daemon, { hash: "#/banks/work/memories", token: TOKEN });

  const shown = await banner(page, "3 failed chunks");
  const link = shown.querySelector('a[href$="#/banks/main/chunks"]');
  assert.ok(link, shown.innerHTML);
  click(link);

  await findRow(page.root, "llm_timeout");
});

test("on a bank's ingestion page the banner doesn't point at the page itself", async (t) => {
  const daemon = new FakeDaemon();
  daemon.state.status.attention = [FAILED_LINE, "the last backup is more than 7 days old"];
  const page = await open(t, daemon, { hash: "#/banks/main/chunks", token: TOKEN });
  await findRow(page.root, "llm_timeout");

  const shown = await banner(page, "the last backup is more than 7 days old");
  assert.ok(!textOf(shown).includes("3 failed chunks"), textOf(shown));
  assert.equal(shown.querySelector('a[href$="#/banks/main/chunks"]'), null);
});
