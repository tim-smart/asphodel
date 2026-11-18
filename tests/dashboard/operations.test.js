// The v1 extras: each bank's counts on the banks page, the daemon's
// attention lines as a banner, failed chunks with retry, the purge pause
// with its acknowledgement, and rebuilding a bank's memories.

import { test } from "node:test";
import assert from "node:assert/strict";

import { FakeDaemon, TOKEN, ids } from "./fake-daemon.js";
import {
  click,
  findButton,
  findDialog,
  findLink,
  findRow,
  findText,
  open,
  queryButton,
  queryDialog,
  queryLink,
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

test("going back from a source to the ingestion page puts its scroll back", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: "#/banks/main/chunks", token: TOKEN });
  const source = await findLink(page.root, /^source /);
  page.window.scrollTo(0, 480);
  click(source);
  await waitFor(() => !page.window.location.hash.includes("chunks") && queryLink(page.root, "← Sources"), "the source's page");
  page.window.scrollTo(0, 0);

  page.window.history.back();

  await findLink(page.root, /^source /);
  await waitFor(() => page.window.scrollY === 480, "the ingestion page's scroll back where it was");
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

const REBUILD = "/v1/banks/main/rebuild";

/// The button in `dialog` that isn't cancel.
function actionOf(dialog) {
  return [...dialog.querySelectorAll("button")].find((b) => !/cancel/i.test(textOf(b)));
}

test("a rebuild's preview counts what it would do and starts nothing", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: "#/banks/main/chunks", token: TOKEN });

  click(await findLink(page.root, /preview a rebuild/i));

  await findButton(page.root, /rebuild memories/i);
  assert.ok(showsCount(page.root, "chunks?", 14), textOf(page.root));
  assert.ok(showsCount(page.root, "sources?", 12), textOf(page.root));
  assert.ok(showsCount(page.root, "memories replaced", 31), textOf(page.root));
  assert.ok(showsCount(page.root, "memories kept", 4), textOf(page.root));
  assert.ok(showsCount(page.root, "chunks a forget masked", 3), textOf(page.root));
  await stays(() => daemon.calls("POST").length === 0, "nothing sent");
});

test("starting a rebuild asks first, naming the downtime and the cost, and cancelling sends nothing", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: "#/banks/main/chunks?rebuild=preview", token: TOKEN });

  click(await findButton(page.root, /rebuild memories/i));
  const dialog = await findDialog(page.document);
  assert.match(textOf(dialog), /recall.*\boff\b/i);
  assert.ok(showsCount(dialog, "LLM calls", 28), textOf(dialog));
  assert.ok(showsCount(dialog, "memories", 31), textOf(dialog));
  assert.match(textOf(dialog), /backup/i);
  click(await findButton(dialog, /cancel/i));
  await waitFor(() => !queryDialog(page.document), "the confirmation to close");

  await stays(() => daemon.calls("POST").length === 0, "nothing sent");
});

test("a confirmed rebuild repeats the bank's name and then shows how far it got", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: "#/banks/main/chunks?rebuild=preview", token: TOKEN });

  click(await findButton(page.root, /rebuild memories/i));
  click(actionOf(await findDialog(page.document)));

  const sent = await waitFor(() => daemon.calls("POST", REBUILD)[0], "POST rebuild");
  assert.equal(sent.query.get("confirm"), "main");
  await findText(page.root, /14 of 14 chunks left/i);
  assert.equal(queryButton(page.root, /rebuild memories/i), null, "no second start while it runs");
});

test("a rebuild that can't start says why and offers no start", async (t) => {
  const daemon = new FakeDaemon();
  daemon.state.rebuild.plan.blockers = ["backup_stale"];
  daemon.state.rebuild.plan.last_backup_at = null;
  const page = await open(t, daemon, { hash: "#/banks/main/chunks?rebuild=preview", token: TOKEN });

  await findText(page.root, /no backup/i);
  assert.equal(queryButton(page.root, /rebuild memories/i), null);
});

function completed(daemon) {
  Object.assign(daemon.state.rebuild.status, {
    state: "complete",
    started_at: "2026-10-02T03:10:00Z",
    completed_at: "2026-10-02T05:00:00Z",
    chunks: 14,
    replaced: 31,
    kept: 4,
    carried: 5,
    dropped: 1,
  });
}

test("a complete rebuild is acknowledged behind a confirmation that counts the next sweep", async (t) => {
  const daemon = new FakeDaemon();
  completed(daemon);
  daemon.state.plan.memories = 23;
  const page = await open(t, daemon, { hash: "#/banks/main/chunks", token: TOKEN });

  click(await findButton(page.root, /acknowledge rebuild/i));
  const dialog = await findDialog(page.document);
  assert.ok(showsCount(dialog, "memor(y|ies)", 23), textOf(dialog));
  click(await findButton(dialog, /cancel/i));
  await waitFor(() => !queryDialog(page.document), "the confirmation to close");
  await stays(() => daemon.calls("POST").length === 0, "nothing sent");

  click(await findButton(page.root, /acknowledge rebuild/i));
  click(actionOf(await findDialog(page.document)));

  await waitFor(() => daemon.calls("POST", `${REBUILD}/ack`)[0], "POST rebuild ack");
  await findLink(page.root, /preview a rebuild/i);
});

test("when the next sweep can't be counted, only the acknowledgement is off: failed chunks still retry", async (t) => {
  const daemon = new FakeDaemon();
  completed(daemon);
  daemon.state.plan = null;
  const page = await open(t, daemon, { hash: "#/banks/main/chunks", token: TOKEN });

  const ack = await findButton(page.root, /acknowledge rebuild/i);
  assert.ok(ack.disabled, "no acknowledgement without the count it confirms against");
  click(queryButton(await findRow(page.root, "llm_timeout"), /retry/i));

  await waitFor(() => daemon.calls("POST", "/v1/banks/main/chunks/retry")[0], "POST retry");
  await waitFor(() => !queryText(page.root, "llm_timeout"), "the retried chunk to leave the list");
  assert.equal(daemon.calls("POST", `${REBUILD}/ack`).length, 0);
});
