// Choosing which mental models go into Hermes's system prompt. A model is in
// the prompt while it's enabled, so the toggle edits `enabled` and nothing
// else. The daemon checks the budget: the dashboard shows the enabled
// models' `max_tokens` against the budget the list carries, and when the
// daemon refuses an enable it shows why and leaves the model out.
//
// A model's toggle is the checkbox, `role="switch"` or `aria-pressed` button
// in its row.

import { test } from "node:test";
import assert from "node:assert/strict";

import { FakeDaemon, TOKEN, questions } from "./fake-daemon.js";
import {
  click,
  findRow,
  findText,
  open,
  queryText,
  rowOf,
  showsCount,
  stays,
  textOf,
  waitFor,
} from "./support.js";

const MODELS = "#/banks/main/models";

function toggleIn(row) {
  return row.querySelector('input[type="checkbox"], [role="switch"], button[aria-pressed]');
}

function isOn(toggle) {
  if (toggle.matches('input[type="checkbox"]')) return toggle.checked;
  return (toggle.getAttribute("aria-checked") ?? toggle.getAttribute("aria-pressed")) === "true";
}

/// The toggle in `name`'s row, if the page shows it.
function queryToggle(page, name) {
  const row = rowOf(page.root, name);
  return row && toggleIn(row);
}

function findToggle(page, name) {
  return waitFor(() => queryToggle(page, name), `a prompt toggle for ${name}`);
}

/// The budget use the page shows, as "used ... budget": "700 of 800
/// tokens", "700 / 800".
function findUse(page, used, budget) {
  return waitFor(
    () => queryText(page.root, new RegExp(`(^|\\D)${used}\\D{1,16}${budget}(\\D|$)`)),
    `${used} of ${budget} tokens`,
  );
}

function patches(daemon, name) {
  return daemon.calls("PATCH", `/v1/banks/main/models/${encodeURIComponent(name)}`);
}

test("a bank's model count on the banks page opens its mental models", async (t) => {
  const page = await open(t, new FakeDaemon(), { token: TOKEN });

  const main = await findRow(page.root, /\bmain\b/);
  const link = await waitFor(() => main.querySelector('a[href$="#/banks/main/models"]'), "a link to the models");
  assert.ok(showsCount(link, "(mental )?models?", 3), textOf(link));
  click(link);

  await findText(page.root, questions.plans);
});

test("each model shows its question, token limit, whether it's in the prompt, its last refresh and error", async (t) => {
  const page = await open(t, new FakeDaemon(), { hash: MODELS, token: TOKEN });

  const profile = await findRow(page.root, "User profile");
  assert.ok(textOf(profile).includes(questions.profile), textOf(profile));
  assert.match(textOf(profile), /(^|\D)500(\D|$)/);
  assert.ok(profile.querySelector('time[datetime="2026-10-03T08:30:00Z"]'), profile.innerHTML);
  assert.ok(isOn(toggleIn(profile)), "the profile is in the prompt");

  const plans = await findRow(page.root, "Plans");
  assert.ok(textOf(plans).includes(questions.plans), textOf(plans));
  assert.match(textOf(plans), /(^|\D)200(\D|$)/);
  assert.match(textOf(plans), /malformed/i);
  assert.ok(isOn(toggleIn(plans)), "Plans is in the prompt");

  const travel = await findRow(page.root, "Travel");
  assert.ok(textOf(travel).includes(questions.travel), textOf(travel));
  assert.match(textOf(travel), /(^|\D)150(\D|$)/);
  assert.ok(!isOn(toggleIn(travel)), "Travel is out of the prompt");
});

test("budget use is the enabled models' tokens against the budget the daemon sends", async (t) => {
  const daemon = new FakeDaemon();
  daemon.state.budget = 900;
  const page = await open(t, daemon, { hash: MODELS, token: TOKEN });

  // The profile's 500 and Plans' 200; Travel is out.
  await findUse(page, 700, 900);
});

test("a model taken out of the prompt and put back is edited through enabled alone", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: MODELS, token: TOKEN });
  await findUse(page, 700, 800);

  click(await findToggle(page, "Plans"));
  const out = await waitFor(() => patches(daemon, "Plans")[0], "PATCH Plans");
  assert.deepEqual(out.body, { enabled: false });
  await waitFor(() => queryToggle(page, "Plans") && !isOn(queryToggle(page, "Plans")), "Plans out of the prompt");
  await findUse(page, 500, 800);

  click(await findToggle(page, "Plans"));
  const back = await waitFor(() => patches(daemon, "Plans")[1], "a second PATCH Plans");
  assert.deepEqual(back.body, { enabled: true });
  await findUse(page, 700, 800);
  assert.ok(isOn(await findToggle(page, "Plans")), "Plans back in the prompt");
  assert.equal(daemon.state.models.find((m) => m.name === "Plans").enabled, true);
});

test("enabling past the budget shows the daemon's error and leaves the model out", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: MODELS, token: TOKEN });

  click(await findToggle(page, "Travel"));

  await waitFor(() => patches(daemon, "Travel")[0], "PATCH Travel");
  await findText(page.root, "850 tokens is over the 800-token budget for mental models");
  await stays(() => !isOn(queryToggle(page, "Travel")), "Travel left out");
  assert.ok(queryText(page.root, /(^|\D)700\D{1,16}800(\D|$)/), "the budget use is unchanged");
  assert.equal(daemon.state.models.find((m) => m.name === "Travel").enabled, false);
});

test("the page says changes reach new Hermes sessions only, and that a model out of the prompt isn't refreshed", async (t) => {
  const page = await open(t, new FakeDaemon(), { hash: MODELS, token: TOKEN });
  await findRow(page.root, "Plans");

  const shown = textOf(page.root);
  assert.match(shown, /\bnew\b[^.]*\bsessions?\b/i);
  assert.match(shown, /\bpaus/i);
  assert.match(shown, /\brefresh/i);
});
