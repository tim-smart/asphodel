// Browsing a bank's memories and sources. The list routes do the
// filtering; the dashboard sends what the owner picks and shows what comes
// back. Browsing only ever reads: it never goes through recall, which would
// log the query and could credit use.

import { test } from "node:test";
import assert from "node:assert/strict";

import { DOCUMENT, FakeDaemon, TOKEN, ids, sentences } from "./fake-daemon.js";
import {
  choose,
  click,
  findControl,
  findField,
  findLink,
  findRow,
  findText,
  open,
  queryLink,
  queryText,
  showsCount,
  submit,
  textOf,
  type,
  waitFor,
} from "./support.js";

const MEMORIES = "#/banks/main/memories";
const LIST = "/v1/banks/main/memories";

function queried(daemon, path, key, value) {
  return waitFor(
    () => daemon.calls("GET", path).some((r) => r.query.get(key) === value),
    `GET ${path} with ${key}=${value}`,
  );
}

test("opening a bank from the banks page shows its memories", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { token: TOKEN });

  click(await findLink(page.root, /\bmain\b/));

  await findText(page.root, sentences.auckland);
});

test("the memory list counts each status, and choosing one lists only those", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: MEMORIES, token: TOKEN });
  await findText(page.root, sentences.auckland);

  const retracted = await findControl(page.root, /retracted/i);
  assert.ok(showsCount(retracted, "retracted", 1), textOf(retracted));
  click(retracted);

  await queried(daemon, LIST, "status", "retracted");
  await findText(page.root, sentences.berlin);
  await waitFor(() => !queryText(page.root, sentences.auckland), "the live memories to leave the list");
});

test("searching asks the daemon for memories with those words", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: MEMORIES, token: TOKEN });
  await findText(page.root, sentences.auckland);

  const search = await findField(page.root, /search/i);
  type(search, "Auck");
  submit(search);

  await queried(daemon, LIST, "q", "Auck");
  await waitFor(() => !queryText(page.root, sentences.wellington), "the list to narrow");
  await findText(page.root, sentences.auckland);
});

test("sorting by fade asks the daemon for the soonest to fade first", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: MEMORIES, token: TOKEN });
  await findText(page.root, sentences.auckland);

  choose(await findField(page.root, /sort/i), "fade");

  await queried(daemon, LIST, "sort", "fade");
});

test("more memories load from the next page under the ones shown", async (t) => {
  const daemon = new FakeDaemon();
  daemon.state.pageSize = 4;
  const page = await open(t, daemon, { hash: MEMORIES, token: TOKEN });
  await findText(page.root, sentences.auckland);
  assert.equal(queryText(page.root, sentences.berlin), null);

  click(await findControl(page.root, /more/i));

  await queried(daemon, LIST, "cursor", "4");
  await findText(page.root, sentences.berlin);
  assert.ok(queryText(page.root, sentences.auckland), "the first page stays");
});

test("a memory opens from the list with its passage and a link to its source", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: MEMORIES, token: TOKEN });

  click(await findLink(page.root, sentences.auckland));
  await findText(page.root, "Sam lives in Auckland (passage)");
  const source = await waitFor(
    () => queryLink(page.root, DOCUMENT) ?? queryLink(page.root, /source/i),
    "a link to the memory's source",
  );
  click(source);

  await findText(page.root, "Sam started at Effectful.");
});

test("the sources page lists each document once and each turn, and says which were removed", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: "#/banks/main/sources", token: TOKEN });

  await findLink(page.root, DOCUMENT);
  const links = [...page.root.querySelectorAll("a[href]")].filter((a) => textOf(a).includes(DOCUMENT));
  assert.equal(links.length, 1, "two versions, one entry");
  const recipes = await findRow(page.root, "recipes.md");
  assert.match(textOf(recipes), /removed/i);

  click(await findLink(page.root, /session-1/));
  await findText(page.root, "My daughter Ada starts school next week.");
});

test("the sources page searches and filters through the daemon", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: "#/banks/main/sources", token: TOKEN });
  await findLink(page.root, DOCUMENT);

  const search = await findField(page.root, /search/i);
  type(search, "ada");
  submit(search);
  await queried(daemon, "/v1/banks/main/sources", "q", "ada");
  await findLink(page.root, /session-1/);
  await waitFor(() => !queryLink(page.root, DOCUMENT), "the list to narrow");

  choose(await findField(page.root, /kind/i), "document");
  await queried(daemon, "/v1/banks/main/sources", "kind", "document");
  await findText(page.root, /no sources match/i);

  page.window.location.hash = "#/banks/main/sources";
  await findLink(page.root, DOCUMENT);
  choose(await findField(page.root, /text/i), "true");
  await queried(daemon, "/v1/banks/main/sources", "gone", "true");
  await findRow(page.root, "recipes.md");
  await waitFor(() => !queryLink(page.root, DOCUMENT), "only gone sources");
});

test("a document's page shows its text, its versions and each chunk's state", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: "#/banks/main/sources", token: TOKEN });

  click(await findLink(page.root, DOCUMENT));

  await findText(page.root, "Sam started at Effectful.");
  const failed = await findRow(page.root, "llm_timeout");
  assert.match(textOf(failed), /failed/i);
  await waitFor(() => page.root.querySelector(`a[href*="${ids.auckland}"]`), "a link to the memory its first chunk holds");
  await waitFor(() => page.root.querySelector(`a[href*="${ids.documentV1}"]`), "a link to the older version");
});

test("a removed document's page says it was removed in place of its text", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: `#/banks/main/sources/${ids.recipes}`, token: TOKEN });

  await findText(page.root, /removed/i);
});

test("browsing banks, memories and sources only reads", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { token: TOKEN });

  click(await findLink(page.root, /\bmain\b/));
  click(await findLink(page.root, sentences.auckland));
  await findText(page.root, "Sam lives in Auckland (passage)");
  page.window.location.hash = "#/banks/main/sources";
  click(await findLink(page.root, DOCUMENT));
  await findText(page.root, "Sam started at Effectful.");

  const writes = daemon.requests.filter((r) => r.method !== "GET" || r.path.endsWith("/recall"));
  assert.deepEqual(writes, []);
});
