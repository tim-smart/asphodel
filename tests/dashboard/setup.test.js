// First-run setup. A daemon with no tuning file shows the wizard in place of
// the dashboard. Setup asks for no code or token: it sends the LLM the owner
// chose, shows the token it made once, and signs the tab in with it.

import { test } from "node:test";
import assert from "node:assert/strict";

import { FakeDaemon, SETUP_TOKEN, UNCALIBRATED } from "./fake-daemon.js";
import { click, findButton, findField, findLink, findText, open, submit, type, waitFor } from "./support.js";

const BANK = /\bmain\b/;

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
