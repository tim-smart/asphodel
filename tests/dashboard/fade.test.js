// When memories fade. The daemon gives each a fade as bank days from now and
// the earliest world date, or null when it never fades, and 0 bank days once
// it has. Bank time runs slower while a bank is quiet, so the date is a
// floor: the dashboard says "no sooner than" and marks it up as a <time>.

import { test } from "node:test";
import assert from "node:assert/strict";

import { FakeDaemon, TOKEN, ids, sentences } from "./fake-daemon.js";
import { findRow, findText, open, textOf, waitFor } from "./support.js";

const MEMORIES = "#/banks/main/memories";

test("a memory that will fade shows its bank days and the date it fades no sooner than", async (t) => {
  const page = await open(t, new FakeDaemon(), { hash: MEMORIES, token: TOKEN });

  const row = await findRow(page.root, sentences.auckland);

  assert.match(textOf(row), /\b14 bank days\b/i);
  assert.match(textOf(row), /no sooner than/i);
  assert.ok(row.querySelector('time[datetime="2026-10-17T09:00:00Z"]'), row.innerHTML);
});

test("a kept memory says it never fades", async (t) => {
  const page = await open(t, new FakeDaemon(), { hash: MEMORIES, token: TOKEN });

  const row = await findRow(page.root, sentences.ada);

  assert.match(textOf(row), /never fades/i);
  assert.doesNotMatch(textOf(row), /no sooner than/i);
});

test("a memory already under the line says it has faded", async (t) => {
  const page = await open(t, new FakeDaemon(), { hash: MEMORIES, token: TOKEN });

  const row = await findRow(page.root, sentences.dentist);

  assert.match(textOf(row), /\bfaded\b/i);
  assert.doesNotMatch(textOf(row), /\b0 bank days\b/i);
});

test("a memory's page shows when it fades and when it can be purged", async (t) => {
  const page = await open(t, new FakeDaemon(), { hash: `#/banks/main/memories/${ids.auckland}`, token: TOKEN });
  await findText(page.root, "Sam lives in Auckland (passage)");

  await waitFor(() => page.root.querySelector('time[datetime="2026-10-17T09:00:00Z"]'), "the fade date");
  await waitFor(() => page.root.querySelector('time[datetime="2026-11-12T09:00:00Z"]'), "the purge date");
  assert.match(textOf(page.root), /no sooner than/i);
  assert.match(textOf(page.root), /purge/i);
});
