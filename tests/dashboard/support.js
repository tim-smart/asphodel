// The page the dashboard tests drive, and the queries they read it with.
//
// The contract the tests pin on `crates/asphodel/assets/dashboard/app.js`:
//
// - It is a plain ES module the daemon serves as it is: no build step, no
//   imports outside `assets/dashboard/`, nothing from node_modules.
// - It exports `mount(root, { fetch })`, which renders the dashboard into
//   `root` and returns a function that stops it (timers, listeners). The
//   page's own script calls it with the window's `fetch`; the tests pass a
//   fake daemon's. Everything else (`location`, `sessionStorage`, timers,
//   `confirm`-style dialogs) comes from `root.ownerDocument.defaultView`.
// - Routes are hash routes under `/dashboard`, so a deep link works:
//   `#/` the banks, `#/banks/{bank}/memories`, `#/banks/{bank}/memories/{id}`,
//   `#/banks/{bank}/documents`, `#/banks/{bank}/sources/{id}`,
//   `#/banks/{bank}/chunks` and `#/banks/{bank}/models`.
// - A confirmation is an element in the page with `role="dialog"` or
//   `role="alertdialog"`, or an open `<dialog>`, holding a cancel button and a
//   button that carries out the action.

import { JSDOM } from "jsdom";

const APP = new URL("../../crates/asphodel/assets/dashboard/app.js", import.meta.url);

/// How long a query waits for the page to catch up.
const PATIENCE_MS = 1500;

/// Opens the dashboard at `hash` against `daemon`. With `token`, the session
/// is signed in first, the way a person would sign in.
export async function start(daemon, { hash = "#/", token } = {}) {
  const dom = new JSDOM(`<!doctype html><html><body><div id="app"></div></body></html>`, {
    url: `http://127.0.0.1:7720/dashboard${hash}`,
    pretendToBeVisual: true,
  });
  const window = dom.window;
  dialogs(window);
  const { mount } = await import(APP.href);
  let root = window.document.getElementById("app");
  let stop = mount(root, { fetch: daemon.fetch });

  const page = {
    window,
    document: window.document,
    get root() {
      return root;
    },
    /// Reloads the page in the same tab: session storage survives.
    async reload() {
      if (typeof stop === "function") stop();
      root.remove();
      root = window.document.createElement("div");
      root.id = "app";
      window.document.body.append(root);
      stop = mount(root, { fetch: daemon.fetch });
    },
    close() {
      if (typeof stop === "function") stop();
      window.close();
    },
  };
  if (token !== undefined) await signIn(page, token);
  return page;
}

/// Gives jsdom's `<dialog>` the methods browsers have.
function dialogs(window) {
  const proto = window.HTMLDialogElement?.prototype;
  if (!proto || proto.showModal) return;
  proto.show = function () {
    this.setAttribute("open", "");
  };
  proto.showModal = proto.show;
  proto.close = function (value) {
    if (value !== undefined) this.returnValue = value;
    this.removeAttribute("open");
    this.dispatchEvent(new window.Event("close"));
  };
}

/// Opens the dashboard for one test and closes it after.
export async function open(t, daemon, options) {
  const page = await start(daemon, options);
  t.after(() => page.close());
  return page;
}

/// Types `token` into the token prompt and sends it.
export async function signIn(page, token) {
  const input = await findTokenInput(page.root);
  type(input, token);
  submit(input);
}

/// The password field the dashboard asks for the token in.
export function findTokenInput(root) {
  return waitFor(() => root.querySelector('input[type="password"]'), "a token prompt");
}

/// Polls `probe` until it returns something truthy, or fails naming `what`.
export async function waitFor(probe, what = "the page to settle") {
  const until = Date.now() + PATIENCE_MS;
  let last;
  for (;;) {
    try {
      const found = probe();
      if (found) return found;
    } catch (error) {
      last = error;
    }
    if (Date.now() > until) {
      throw new Error(`timed out waiting for ${what}${last ? `: ${last.message}` : ""}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

/// Lets the page run what it queued, then checks `probe` still holds: for
/// asserting that something does not happen.
export async function stays(probe, what) {
  await new Promise((resolve) => setTimeout(resolve, 150));
  if (!probe()) throw new Error(`expected ${what}`);
}

function text(node) {
  return (node.textContent ?? "").replace(/\s+/g, " ").trim();
}

function matches(value, matcher) {
  return typeof matcher === "string" ? value.includes(matcher) : matcher.test(value);
}

/// The innermost visible element whose text matches.
export function queryText(root, matcher) {
  const all = [...root.querySelectorAll("*")].filter(
    (el) => !["SCRIPT", "STYLE", "TEMPLATE"].includes(el.tagName) && matches(text(el), matcher),
  );
  return all.find((el) => ![...el.children].some((child) => matches(text(child), matcher))) ?? null;
}

export function findText(root, matcher) {
  return waitFor(() => queryText(root, matcher), `text ${matcher}`);
}

function accessibleName(el) {
  const label = el.getAttribute("aria-label");
  if (label) return label;
  const labelledBy = el.getAttribute("aria-labelledby");
  if (labelledBy) {
    const named = el.ownerDocument.getElementById(labelledBy);
    if (named) return text(named);
  }
  if (el.labels?.length) return text(el.labels[0]);
  return text(el) || el.getAttribute("title") || el.getAttribute("value") || "";
}

/// A button (or a submit input) whose accessible name matches.
export function queryButton(root, matcher) {
  return (
    [...root.querySelectorAll('button, [role="button"], input[type="submit"]')].find((el) =>
      matches(accessibleName(el), matcher),
    ) ?? null
  );
}

export function findButton(root, matcher) {
  return waitFor(() => queryButton(root, matcher), `a button named ${matcher}`);
}

/// A button or a link whose accessible name matches, for controls that can
/// fairly be either, such as a filter.
export function findControl(root, matcher) {
  return waitFor(() => queryButton(root, matcher) ?? queryLink(root, matcher), `a control named ${matcher}`);
}

/// A link whose accessible name matches.
export function queryLink(root, matcher) {
  return [...root.querySelectorAll("a[href]")].find((el) => matches(accessibleName(el), matcher)) ?? null;
}

export function findLink(root, matcher) {
  return waitFor(() => queryLink(root, matcher), `a link named ${matcher}`);
}

/// A form control whose label matches.
export function findField(root, matcher) {
  return waitFor(
    () =>
      [...root.querySelectorAll("input, select, textarea")].find((el) =>
        matches(accessibleName(el), matcher),
      ),
    `a field labelled ${matcher}`,
  );
}

/// The open confirmation, if there is one.
export function queryDialog(document) {
  return (
    [...document.querySelectorAll('[role="dialog"], [role="alertdialog"], dialog')].find(
      (el) => (el.tagName !== "DIALOG" || el.open) && !el.hidden,
    ) ?? null
  );
}

export function findDialog(document) {
  return waitFor(() => queryDialog(document), "a confirmation");
}

/// The list row or card a piece of text sits in.
export function rowOf(root, matcher) {
  const el = queryText(root, matcher);
  return el?.closest('tr, li, article, [role="row"], [role="listitem"]') ?? null;
}

export function findRow(root, matcher) {
  return waitFor(() => rowOf(root, matcher), `a row with ${matcher}`);
}

/// Whether `content` shows `count` beside `label`, either way round and
/// with a few words between but no other number: "Live 112", "112 live",
/// "Failed chunks: 3".
export function showsCount(content, label, count) {
  const value = text(content);
  return (
    new RegExp(`(^|\\D)${count}\\D{0,24}?${label}`, "i").test(value) ||
    new RegExp(`${label}\\D{0,24}?${count}(\\D|$)`, "i").test(value)
  );
}

export { text as textOf };

export function click(el) {
  el.dispatchEvent(new el.ownerDocument.defaultView.MouseEvent("click", { bubbles: true, cancelable: true }));
}

export function type(input, value) {
  const window = input.ownerDocument.defaultView;
  input.value = value;
  input.dispatchEvent(new window.Event("input", { bubbles: true }));
  input.dispatchEvent(new window.Event("change", { bubbles: true }));
}

export function choose(select, value) {
  const window = select.ownerDocument.defaultView;
  select.value = value;
  select.dispatchEvent(new window.Event("input", { bubbles: true }));
  select.dispatchEvent(new window.Event("change", { bubbles: true }));
}

/// Submits the form `input` is in, as pressing Enter would.
export function submit(input) {
  const form = input.form ?? input.closest("form");
  if (form) form.requestSubmit();
  else {
    const window = input.ownerDocument.defaultView;
    input.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  }
}
