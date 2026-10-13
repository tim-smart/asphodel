// First-run setup. A daemon with no tuning file shows the wizard in place of
// the dashboard. Setup asks for no code or token: it sends the LLM the owner
// chose, shows the token it made once, and signs the tab in with it. With a
// ChatGPT subscription, the wizard runs the daemon's device-code login: it
// shows the issuer's link and code and follows the login until it lands.

import { test } from "node:test";
import assert from "node:assert/strict";

import { FakeDaemon, SETUP_TOKEN, UNCALIBRATED, VERIFICATION_URL } from "./fake-daemon.js";
import { click, findButton, findField, findLink, findText, open, submit, type, waitFor } from "./support.js";

const BANK = /\bmain\b/;
const LOGIN = "/v1/setup/llm-login";

function chooseChatgpt(page) {
  return waitFor(() => page.root.querySelector('input[type="radio"][value="chatgpt"]'), "the ChatGPT choice").then(click);
}

/// The issuer's link and the code to enter there, both on the page.
async function findLoginCode(root, code) {
  await findText(root, code);
  return waitFor(() => root.querySelector(`a[href="${VERIFICATION_URL}"]`), "a link to the issuer");
}

test("setup asks for no code or token, sends the LLM and signs the tab in with the token it made", async (t) => {
  const daemon = new FakeDaemon({ setup: true });
  const page = await open(t, daemon);

  const model = await findField(page.root, /^model/i);
  // The placeholder floor is named before anything is written.
  await findText(page.root, UNCALIBRATED);
  assert.equal(daemon.calls("POST", "/v1/setup").length, 0);
  assert.equal(page.root.querySelector('input[type="password"]:not(#setup-api-key)'), null);

  type(model, "model-under-test");
  type(await findField(page.root, /^api key$/i), "sk-key");
  submit(model);
  const token = await findField(page.root, /bearer token/i);
  assert.equal(token.value, SETUP_TOKEN);

  const [sent] = daemon.calls("POST", "/v1/setup");
  assert.equal(sent.authorization, null);
  assert.equal(sent.body.llm.auth, "api_key");
  assert.equal(sent.body.llm.model, "model-under-test");
  assert.equal(sent.body.llm.api_key, "sk-key");
  assert.match(sent.body.llm.endpoint, /^https:\/\//);

  click(await findButton(page.root, /open the dashboard/i));
  await findLink(page.root, BANK);
  const after = daemon.requests.filter((r) => r.path.startsWith("/v1/") && !["/v1/health", "/v1/setup"].includes(r.path));
  assert.ok(after.length > 0);
  for (const r of after) assert.equal(r.authorization, `Bearer ${SETUP_TOKEN}`, `${r.method} ${r.path}`);
});

test("setup finishes while the ChatGPT login waits, and the page follows the login until it lands", async (t) => {
  const daemon = new FakeDaemon({ setup: true });
  const page = await open(t, daemon);

  await chooseChatgpt(page);
  click(await findButton(page.root, /^log in/i));
  await findLoginCode(page.root, "ABCD-0001");
  const [started] = daemon.calls("POST", LOGIN);
  assert.equal(started.authorization, null);

  type(await findField(page.root, /^model/i), "model-under-test");
  click(await findButton(page.root, /finish setup/i));
  await findField(page.root, /bearer token/i);
  const [sent] = daemon.calls("POST", "/v1/setup");
  assert.equal(sent.body.llm.auth, "chatgpt");
  assert.equal(sent.body.llm.api_key, undefined);

  // The daemon won't show the code again, so the page keeps it.
  await findLoginCode(page.root, "ABCD-0001");
  daemon.approveLogin();
  await findText(page.root, /logged in/i);
  assert.equal(daemon.calls("POST", LOGIN).length, 1);
});

test("a failed or expired ChatGPT login is shown and can be retried with a new code", async (t) => {
  const daemon = new FakeDaemon({ setup: true });
  const page = await open(t, daemon);
  const retry = /log in|try again|retry/i;

  await chooseChatgpt(page);
  click(await findButton(page.root, /^log in/i));
  await findLoginCode(page.root, "ABCD-0001");
  daemon.failLogin("the issuer refused the device code");
  await findText(page.root, "the issuer refused the device code");

  click(await findButton(page.root, retry));
  await findLoginCode(page.root, "ABCD-0002");
  daemon.expireLogin();
  await findText(page.root, /expired/i);

  click(await findButton(page.root, retry));
  await findLoginCode(page.root, "ABCD-0003");
  daemon.approveLogin();
  await findText(page.root, /logged in/i);
  assert.equal(daemon.calls("POST", LOGIN).length, 3);
});

test("setting the LLM up later sends none", async (t) => {
  const daemon = new FakeDaemon({ setup: true });
  const page = await open(t, daemon);

  click(await waitFor(() => page.root.querySelector('input[type="radio"][value="later"]'), "the Later choice"));
  click(await findButton(page.root, /finish setup/i));
  await findField(page.root, /bearer token/i);

  const [sent] = daemon.calls("POST", "/v1/setup");
  assert.deepEqual(sent.body, {});
});

test("a daemon that is set up goes straight to the dashboard", async (t) => {
  const daemon = new FakeDaemon({ token: null });
  const page = await open(t, daemon);

  await findLink(page.root, BANK);
  assert.equal(page.root.querySelector("#setup-model"), null);
});
