// The bearer token. On loopback a daemon may have none, and then the
// dashboard never asks. Otherwise it asks once per tab, keeps the token in
// session storage only, and sends it on every `/v1` call. A token the daemon
// rejects is dropped and asked for again.

import { test } from "node:test";
import assert from "node:assert/strict";

import { FakeDaemon, TOKEN } from "./fake-daemon.js";
import { findLink, findText, findTokenInput, open, signIn, waitFor } from "./support.js";

const BANK = /\bmain\b/;

function stored(storage) {
  return Array.from({ length: storage.length }, (_, i) => storage.getItem(storage.key(i)));
}

test("a daemon without a token is browsed without asking for one", async (t) => {
  const daemon = new FakeDaemon({ token: null });
  const page = await open(t, daemon);

  await findLink(page.root, BANK);
  assert.equal(page.root.querySelector('input[type="password"]'), null);
});

test("a daemon that wants a token is asked with it on every call once it is given", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon);

  await findTokenInput(page.root);
  const before = daemon.requests.length;
  await signIn(page, TOKEN);
  await findLink(page.root, BANK);

  const after = daemon.requests.slice(before).filter((r) => r.path !== "/v1/health");
  assert.ok(after.length > 0, "the dashboard called the daemon after sign-in");
  for (const r of after) assert.equal(r.authorization, `Bearer ${TOKEN}`, `${r.method} ${r.path}`);
});

test("the token stays in the tab's session storage, out of the URL and local storage", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { token: TOKEN });
  await findLink(page.root, BANK);

  assert.ok(stored(page.window.sessionStorage).some((value) => value?.includes(TOKEN)));
  assert.equal(page.window.localStorage.length, 0);
  assert.ok(!page.window.location.href.includes(TOKEN));
  assert.ok(!page.document.cookie.includes(TOKEN));
});

test("reloading the tab doesn't ask for the token again", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { token: TOKEN });
  await findLink(page.root, BANK);

  await page.reload();

  await findLink(page.root, BANK);
  assert.equal(page.root.querySelector('input[type="password"]'), null);
});

test("a rejected token is dropped and asked for again", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon);

  await signIn(page, "wrong-token");
  await waitFor(() => daemon.requests.some((r) => r.authorization === "Bearer wrong-token"), "a call with the wrong token");
  await findText(page.root, /rejected|refused|not accepted|wrong|invalid|incorrect/i);
  await findTokenInput(page.root);
  assert.ok(!stored(page.window.sessionStorage).some((value) => value?.includes("wrong-token")));

  await signIn(page, TOKEN);
  await findLink(page.root, BANK);
});
