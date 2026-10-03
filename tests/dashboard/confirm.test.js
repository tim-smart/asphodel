// Forget, retract and document removal can't be undone, so each asks first
// and says what will go. Nothing is sent until the owner confirms; cancelling
// sends nothing. Afterwards the page says what the daemon did.

import { test } from "node:test";
import assert from "node:assert/strict";

import { DOCUMENT, FakeDaemon, TOKEN, ids, sentences } from "./fake-daemon.js";
import {
  click,
  findButton,
  findDialog,
  findText,
  open,
  queryDialog,
  stays,
  textOf,
  waitFor,
} from "./support.js";

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

test("a confirmed removal deletes the document by its id, slashes and all, and says what went", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: `#/banks/main/sources/${ids.documentV2}`, token: TOKEN });

  click(await findButton(page.root, /remove/i));
  await confirm(page, /remove/i);

  const sent = await waitFor(() => daemon.calls("DELETE")[0], "DELETE the document");
  assert.equal(decodeURIComponent(sent.path), `/v1/banks/main/documents/${DOCUMENT}`);
  await waitFor(() => !queryDialog(page.document), "the confirmation to close");
  await findText(page.root, /\b6 memories\b/i);
});
