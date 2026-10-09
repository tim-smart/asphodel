// First-run setup. A daemon with no tuning file shows the wizard in place of
// the dashboard. Setup needs the code from the data dir, sends the LLM the
// owner chose, shows the token it made once, and signs the tab in with it.

import { test } from "node:test";
import assert from "node:assert/strict";

import { FakeDaemon, SETUP_CODE, SETUP_TOKEN, UNCALIBRATED } from "./fake-daemon.js";
import { click, findButton, findField, findLink, findText, open, submit, type, waitFor } from "./support.js";

const BANK = /\bmain\b/;

test("setup needs the code, sends the LLM and signs the tab in with the token it made", async (t) => {
  const daemon = new FakeDaemon({ setup: true });
  const page = await open(t, daemon);

  const code = await findField(page.root, /setup code/i);
  // The placeholder floor is named before anything is written.
  await findText(page.root, UNCALIBRATED);
  assert.equal(daemon.calls("POST", "/v1/setup").length, 0);
  type(await findField(page.root, /^model/i), "model-under-test");
  type(await findField(page.root, /^api key$/i), "sk-key");
  type(code, "not-the-code");
  submit(code);
  await findText(page.root, /isn't the setup code/i);

  type(code, SETUP_CODE);
  submit(code);
  const token = await findField(page.root, /bearer token/i);
  assert.equal(token.value, SETUP_TOKEN);

  const [sent] = daemon.calls("POST", "/v1/setup").filter((r) => r.authorization === `Bearer ${SETUP_CODE}`);
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

test("setting the LLM up later sends none", async (t) => {
  const daemon = new FakeDaemon({ setup: true });
  const page = await open(t, daemon);

  click(await waitFor(() => page.root.querySelector('input[type="radio"][value="later"]'), "the Later choice"));
  const code = await findField(page.root, /setup code/i);
  type(code, SETUP_CODE);
  submit(code);
  await findField(page.root, /bearer token/i);

  const [sent] = daemon.calls("POST", "/v1/setup");
  assert.deepEqual(sent.body, {});
});

test("a daemon that is set up goes straight to the dashboard", async (t) => {
  const daemon = new FakeDaemon({ token: null });
  const page = await open(t, daemon);

  await findLink(page.root, BANK);
  assert.equal(page.root.querySelector("#setup-code"), null);
});
