// Forget, retract and document removal can't be undone, so each asks first
// and says what will go. Nothing is sent until the owner confirms; cancelling
// sends nothing. Afterwards the page says what the daemon did, even when what
// it acted on is already gone.
//
// A document is removed with `POST /v1/banks/{bank}/documents/remove` and
// `{"document_id": ...}`: the id is the exact string ingested, which a URL
// path can't carry (`notes/../victim.md` normalizes to `victim.md`).

import { test } from "node:test";
import assert from "node:assert/strict";

import { DOCUMENT, FakeDaemon, TOKEN, document, ids, sentences } from "./fake-daemon.js";
import {
  click,
  findButton,
  findDialog,
  findText,
  open,
  queryDialog,
  queryText,
  stays,
  textOf,
  waitFor,
} from "./support.js";

const REMOVE = "/v1/banks/main/documents/remove";
const DOTTED_SOURCE = "01a10400-0000-7000-8000-0000000000d1";
const VICTIM_SOURCE = "01a10400-0000-7000-8000-0000000000d2";

function memoryPage(id) {
  return `#/banks/main/memories/${id}`;
}

async function cancel(page) {
  const dialog = await findDialog(page.document);
  click(await findButton(dialog, /cancel/i));
  await waitFor(() => !queryDialog(page.document), "the confirmation to close");
}

async function confirm(page, action) {
  const dialog = await findDialog(page.document);
  const button = [...dialog.querySelectorAll("button")].find(
    (b) => action.test(textOf(b)) && !/cancel/i.test(textOf(b)),
  );
  assert.ok(button, `a ${action} button in the confirmation`);
  click(button);
}

function writes(daemon) {
  return daemon.requests.filter((r) => r.method !== "GET");
}

test("forgetting a memory asks first, and cancelling sends nothing", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: memoryPage(ids.auckland), token: TOKEN });

  click(await findButton(page.root, /forget/i));
  const dialog = await findDialog(page.document);
  assert.match(textOf(dialog), /undo|undone|irreversible|permanent/i);
  assert.match(textOf(dialog), /version/i);
  await cancel(page);

  await stays(() => writes(daemon).length === 0, "nothing sent");
});

test("a confirmed forget sends the memory and says how many memories went with its chain", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: memoryPage(ids.auckland), token: TOKEN });

  click(await findButton(page.root, /forget/i));
  await confirm(page, /forget/i);

  const sent = await waitFor(() => daemon.calls("POST", "/v1/banks/main/forget")[0], "POST forget");
  assert.deepEqual(sent.body.ids, [ids.auckland]);
  await findText(page.root, /\b2 memories\b/i);
});

test("retracting asks first and says it reopens what the memory ended", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: memoryPage(ids.job), token: TOKEN });

  click(await findButton(page.root, /retract/i));
  const dialog = await findDialog(page.document);
  assert.match(textOf(dialog), /reopen/i);
  assert.match(textOf(dialog), /\b1 memory\b/i);
  await cancel(page);

  await stays(() => writes(daemon).length === 0, "nothing sent");
});

test("a confirmed retract marks the memory retracted and links what it reopened", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: memoryPage(ids.job), token: TOKEN });

  click(await findButton(page.root, /retract/i));
  await confirm(page, /retract/i);

  await waitFor(() => daemon.calls("POST", `/v1/banks/main/memories/${ids.job}/retract`)[0], "POST retract");
  await waitFor(() => !queryDialog(page.document), "the confirmation to close");
  await findText(page.root, /\bretracted\b/i);
  await waitFor(() => page.root.querySelector(`a[href*="${ids.oldJob}"]`), "a link to the reopened memory");
});

test("a refused retract shows the daemon's reason", async (t) => {
  const daemon = new FakeDaemon();
  // Reconciliation retracted it after the page loaded.
  const page = await open(t, daemon, { hash: memoryPage(ids.job), token: TOKEN });
  await findText(page.root, sentences.job);
  daemon.state.memories.find((m) => m.id === ids.job).status = "retracted";

  click(await findButton(page.root, /retract/i));
  await confirm(page, /retract/i);

  await findText(page.root, "already retracted");
});

test("removing a document asks first, warning that every version and its memories go", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: `#/banks/main/sources/${ids.documentV2}`, token: TOKEN });

  click(await findButton(page.root, /remove/i));
  const dialog = await findDialog(page.document);
  assert.ok(textOf(dialog).includes(DOCUMENT), textOf(dialog));
  assert.match(textOf(dialog), /version/i);
  assert.match(textOf(dialog), /forg/i);
  assert.match(textOf(dialog), /backup/i);
  await cancel(page);

  await stays(() => writes(daemon).length === 0, "nothing sent");
});

test("a confirmed removal sends the document's id in the body and says what went", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: `#/banks/main/sources/${ids.documentV2}`, token: TOKEN });

  click(await findButton(page.root, /remove/i));
  await confirm(page, /remove/i);

  const sent = await waitFor(() => daemon.calls("POST", REMOVE)[0], "POST documents/remove");
  assert.deepEqual(sent.body, { document_id: DOCUMENT });
  await waitFor(() => !queryDialog(page.document), "the confirmation to close");
  await findText(page.root, /\b6 memories\b/i);
});

test("removing a document whose id has dot segments removes that document and no other", async (t) => {
  const daemon = new FakeDaemon();
  daemon.state.sources.push(
    document(DOTTED_SOURCE, "notes/../victim.md", "Notes that only look like a path."),
    document(VICTIM_SOURCE, "victim.md", "The victim's own text."),
  );
  const page = await open(t, daemon, { hash: `#/banks/main/sources/${DOTTED_SOURCE}`, token: TOKEN });

  click(await findButton(page.root, /remove/i));
  const dialog = await findDialog(page.document);
  assert.ok(textOf(dialog).includes("notes/../victim.md"), textOf(dialog));
  await confirm(page, /remove/i);

  const sent = await waitFor(() => daemon.calls("POST", REMOVE)[0], "POST documents/remove");
  assert.deepEqual(sent.body, { document_id: "notes/../victim.md" });
  assert.deepEqual(
    writes(daemon).filter((r) => r !== sent).map((r) => `${r.method} ${r.path}`),
    [],
    "nothing else is written",
  );
  page.window.location.hash = `#/banks/main/sources/${VICTIM_SOURCE}`;
  await findText(page.root, "The victim's own text.");
});

test("a forget the daemon erases at once still says how many memories went", async (t) => {
  const daemon = new FakeDaemon({ eraseOnForget: true });
  const page = await open(t, daemon, { hash: memoryPage(ids.auckland), token: TOKEN });

  click(await findButton(page.root, /forget/i));
  await confirm(page, /forget/i);

  await waitFor(() => daemon.calls("POST", "/v1/banks/main/forget")[0], "POST forget");
  await waitFor(
    () => /^#\/banks\/main\/memories(\?|$)/.test(page.window.location.hash),
    "the memory list, since the memory is gone",
  );
  await findText(page.root, /\b2 memories\b/i);
  assert.equal(queryText(page.root, /didn't load/i), null);
  assert.equal(page.root.querySelector(`a[href*="${ids.auckland}"]`), null, "the forgotten memory isn't listed");
});
