// The Recall page: a query against a bank, through the recall or the
// injection pipeline, with the working shown. It only ever calls
// `POST /v1/banks/{bank}/recall/explain`, which logs nothing and touches no
// session; `/recall` and `/prefetch` would log the query and could hold an
// injection. The page answers "why didn't memory X come back?", so what was
// left out is listed under the results with the reason.

import { test } from "node:test";
import assert from "node:assert/strict";

import { FakeDaemon, TOKEN, ids, missedReranker, sentences } from "./fake-daemon.js";
import {
  choose,
  click,
  findField,
  findLink,
  findText,
  nameOf,
  open,
  queryButton,
  showsCount,
  stays,
  submit,
  textOf,
  type,
  waitFor,
} from "./support.js";

const RECALL = "#/banks/main/recall";
const EXPLAIN = "/v1/banks/main/recall/explain";
const QUERY = "where does Sam live";

/// The box the query or message goes in, not a previous turn's.
function findQuery(root) {
  return waitFor(
    () =>
      [...root.querySelectorAll("input, textarea")].find(
        (el) => /query|message/i.test(nameOf(el)) && !/previous/i.test(nameOf(el)),
      ),
    "a query box",
  );
}

/// Picks recall or injection, whether the page offers radios, buttons or a
/// select.
async function chooseMode(root, mode) {
  const named = new RegExp(mode, "i");
  const control = await waitFor(
    () =>
      [...root.querySelectorAll('input[type="radio"]')].find((el) => named.test(nameOf(el))) ??
      [...root.querySelectorAll("select")].find((el) => [...el.options].some((o) => o.value === mode)) ??
      queryButton(root, named),
    `a control to choose ${mode}`,
  );
  if (control.tagName === "SELECT") choose(control, mode);
  else click(control);
}

/// Ticks one kind, whether the page offers a checkbox per kind or a
/// multiple select.
async function pickKind(root, kind) {
  const control = await waitFor(
    () =>
      [...root.querySelectorAll('input[type="checkbox"]')].find((el) => new RegExp(`^${kind}s?$`, "i").test(nameOf(el))) ??
      [...root.querySelectorAll("select[multiple]")].find((el) => [...el.options].some((o) => o.value === kind)),
    `a control to pick the ${kind} kind`,
  );
  if (control.tagName === "SELECT") {
    for (const option of control.options) option.selected = option.value === kind;
    choose(control, kind);
  } else click(control);
}

async function explain(daemon, root, query = QUERY) {
  const box = await findQuery(root);
  const before = daemon.calls("POST", EXPLAIN).length;
  type(box, query);
  submit(box);
  await waitFor(() => daemon.calls("POST", EXPLAIN).length > before, `POST ${EXPLAIN}`);
  return daemon.calls("POST", EXPLAIN).at(-1).body;
}

/// The text of `row`'s cell under the column headed `header`, or the whole
/// row's text when it isn't a table row under such a header.
function column(row, header) {
  const table = row.closest("table");
  const heads = table ? [...table.querySelectorAll("thead th")] : [];
  const at = heads.findIndex((th) => header.test(textOf(th)));
  return at >= 0 && row.cells?.[at] ? textOf(row.cells[at]) : textOf(row);
}

/// The result row for `key`'s memory, found by its link to the memory's
/// page: the injected text holds the same sentence.
function findResult(root, key) {
  return waitFor(
    () => root.querySelector(`a[href="#/banks/main/memories/${ids[key]}"]`)?.closest('tr, li, article, [role="row"], [role="listitem"]'),
    `a result row linking to ${key}`,
  );
}

function follows(earlier, later) {
  return Boolean(earlier.compareDocumentPosition(later) & earlier.ownerDocument.defaultView.Node.DOCUMENT_POSITION_FOLLOWING);
}

test("the bank's pages link to its Recall page, which waits for a query", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: "#/banks/main/memories", token: TOKEN });
  await findText(page.root, sentences.auckland);

  click(await findLink(page.root, /^recall$/i));

  await waitFor(() => page.window.location.hash === RECALL, `the hash to become ${RECALL}`);
  await findQuery(page.root);
  await stays(() => daemon.calls("POST", EXPLAIN).length === 0, "no explain before a query is sent");
});

test("a recall explain sends the query and the filters, and never calls recall or prefetch", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: RECALL, token: TOKEN });
  await findQuery(page.root);

  choose(await findField(page.root, /phase/i), "upcoming");
  await pickKind(page.root, "event");
  type(await findField(page.root, /entity/i), "Sam");
  type(await findField(page.root, /limit/i), "2");
  const body = await explain(daemon, page.root);

  assert.equal(body.mode, "recall");
  assert.equal(body.query, QUERY);
  assert.equal(body.phase, "upcoming");
  assert.deepEqual(body.kinds, ["event"]);
  assert.equal(body.entity, "Sam");
  assert.equal(body.limit, 2);
  await findText(page.root, sentences.auckland);
  const others = daemon.requests.filter((r) => r.method !== "GET" && r.path !== EXPLAIN);
  assert.deepEqual(others, []);
  assert.ok(!daemon.requests.some((r) => /\/(recall|prefetch)$/.test(r.path)), "no recall or prefetch");
});

test("each result shows the arms that found it, its fused rank, its logit, its score and its strength", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: RECALL, token: TOKEN });
  await explain(daemon, page.root);

  const row = await findResult(page.root, "auckland");

  const arms = column(row, /arms?|found/i);
  assert.match(arms, /vector\D{0,12}4/i);
  assert.match(arms, /bm25\D{0,12}7/i);
  assert.match(column(row, /rrf|fused/i), /\b3\b/);
  assert.match(column(row, /logit/i), /2\.75/);
  assert.match(column(row, /score/i), /3\.05/);
  assert.match(textOf(row), /0\.40?\b/, "w_s × strength");
  assert.match(textOf(row), /[-−]0\.10?\b/, "the state confidence term");
  assert.match(column(row, /strength|band/i), /strong/i);
  assert.match(textOf(await findResult(page.root, "ada")), /kept/i);
});

test("what recall leaves out comes after what it returns, saying it was past the limit", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: RECALL, token: TOKEN });
  await explain(daemon, page.root);

  const returned = await findResult(page.root, "auckland");
  const job = await findResult(page.root, "job");
  const cut = await findResult(page.root, "ada");

  assert.ok(follows(returned, cut) && follows(job, cut), "left out after the results");
  assert.match(textOf(cut), /limit/i);
  assert.doesNotMatch(textOf(returned), /limit/i);
});

test("an injection explain sends the message and the previous turn, not the recall filters", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: RECALL, token: TOKEN });
  await findQuery(page.root);
  type(await findField(page.root, /limit/i), "2");

  await chooseMode(page.root, "injection");
  type(await findField(page.root, /previous (query|message)/i), "Where's Sam these days?");
  type(await findField(page.root, /previous reply|reply/i), "Sam moved recently.");
  const body = await explain(daemon, page.root, "and before that?");

  assert.equal(body.mode, "injection");
  assert.equal(body.query, "and before that?");
  assert.equal(body.previous_query, "Where's Sam these days?");
  assert.equal(body.previous_reply, "Sam moved recently.");
  assert.equal(body.limit ?? null, null, "no recall limit");
  assert.equal(body.session_id ?? null, null, "no session");
});

test("an injection shows the injected text exactly as the agent gets it, with its token count", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: RECALL, token: TOKEN });
  await chooseMode(page.root, "injection");
  await explain(daemon, page.root);

  const { text, tokens } = daemon.state.explain.injection.injection;
  const block = await waitFor(
    () => [...page.root.querySelectorAll("*")].find((el) => el.textContent === text),
    "the injected text, newlines and all",
  );
  assert.ok(showsCount(block.parentElement, "tokens?", tokens) || showsCount(page.root, "tokens?", tokens), "the token count");
  assert.match(textOf(page.root), /session|in context/i, "says nothing counts as already in context");
});

test("an injection says why each candidate it left out didn't pass", async (t) => {
  const daemon = new FakeDaemon();
  const page = await open(t, daemon, { hash: RECALL, token: TOKEN });
  await chooseMode(page.root, "injection");
  await explain(daemon, page.root);

  const injected = await findResult(page.root, "auckland");
  for (const [key, reason] of [
    ["job", /\bcap\b/i],
    ["oldJob", /budget/i],
    ["ada", /floor/i],
    ["dentist", /τ|\btau\b/i],
  ]) {
    const row = await findResult(page.root, key);
    assert.ok(follows(injected, row), `${key} after what was injected`);
    assert.match(textOf(row), reason, `${key}: ${textOf(row)}`);
  }
  const faded = await findResult(page.root, "dentist");
  assert.doesNotMatch(textOf(faded), /null|undefined|NaN/, "below τ has no rank, logit or score to show");
});

test("each stage's latency shows, and a reranker that missed its deadline says so", async (t) => {
  const daemon = new FakeDaemon();
  daemon.state.explain.injection = missedReranker();
  const page = await open(t, daemon, { hash: RECALL, token: TOKEN });
  await chooseMode(page.root, "injection");
  await explain(daemon, page.root);

  await findText(page.root, /missed|deadline|not reranked|wasn't reranked/i);
  for (const [stage, ms] of [
    ["embed", 12],
    ["retriev", 34],
    ["rerank", 250],
    ["total", 296],
  ]) {
    assert.ok(showsCount(page.root, stage, ms), `${stage} ${ms} ms`);
  }
  const row = await findResult(page.root, "auckland");
  assert.match(textOf(row), /missed|deadline|not reranked|wasn't reranked/i);
});

test("an explain the daemon refuses shows its error", async (t) => {
  const daemon = new FakeDaemon();
  daemon.state.explain = {};
  const page = await open(t, daemon, { hash: RECALL, token: TOKEN });
  await explain(daemon, page.root);

  await findText(page.root, "mode must be recall or injection");
});
