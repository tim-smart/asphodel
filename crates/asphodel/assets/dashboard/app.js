// The dashboard: hash routes over the daemon's read-only list routes, and
// the owner's actions (keep, retract, forget, remove a document, retry
// chunks, resume purge), each destructive one behind a confirmation.
//
// `mount(root, { fetch })` renders into `root` and returns a function that
// stops it. Everything else (location, session storage, timers) comes from
// the root's window. No build step: the daemon serves these files as they
// are.

import { ApiError, Unauthorized, client } from "./api.js";
import { elements } from "./dom.js";
import { pages, route } from "./pages.js";

/// How often the attention banner asks the daemon again.
const STATUS_EVERY_MS = 20_000;

export function mount(root, { fetch }) {
  const document = root.ownerDocument;
  const window = document.defaultView;
  const api = client(fetch, window.sessionStorage);
  const ui = elements(document);
  const { h } = ui;

  let generation = 0;
  let stopped = false;
  let notice = null;
  // Set while an action moves to another page, so its notice survives the
  // navigation.
  let carryNotice = false;
  let banner = null;
  let tabs = null;
  let main = null;
  const dialogs = new Set();

  // The page's chrome: header, banner slot and main, rebuilt per render.
  function shell(at) {
    const bank = at.bank;
    banner = h("div", { class: "banner-slot" });
    main = h("main", { id: "content", class: "content", tabindex: "-1", "aria-busy": "true" });
    tabs = bank
      ? h(
          "nav",
          { class: "tabs", "aria-label": `Bank ${bank}` },
          [
            ["memories", "Memories"],
            ["documents", "Documents"],
            ["chunks", "Ingestion"],
          ].map(([name, label]) =>
            h(
              "a",
              {
                href: `#/banks/${encodeURIComponent(bank)}/${name}`,
                "aria-current": at.section === name ? "page" : null,
                "data-tab": name,
              },
              label,
            ),
          ),
        )
      : null;
    const skip = h(
      "a",
      {
        class: "skip",
        href: "#content",
        onclick: (event) => {
          event.preventDefault();
          main.focus();
        },
      },
      "Skip to content",
    );
    const header = h(
      "header",
      { class: "top" },
      h(
        "div",
        { class: "top-row" },
        h("a", { class: "mark", href: "#/", "aria-label": "Asphodel, all banks" }, flower(), h("span", {}, "Asphodel")),
        bank ? h("span", { class: "crumb" }, h("span", { "aria-hidden": "true" }, "/"), h("span", { class: "bank-name" }, bank)) : null,
        api.token.get()
          ? h(
              "button",
              {
                type: "button",
                class: "quiet sign-out",
                onclick: () => {
                  api.token.clear();
                  render();
                },
              },
              "Sign out",
            )
          : null,
      ),
      tabs,
    );
    root.replaceChildren(h("div", { class: "shell" }, skip, header, banner, main));
  }

  function flower() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "-12 -12 24 24");
    svg.setAttribute("class", "flower");
    svg.setAttribute("aria-hidden", "true");
    for (const angle of [0, 60, 120]) {
      const petal = document.createElementNS("http://www.w3.org/2000/svg", "ellipse");
      petal.setAttribute("rx", "3.2");
      petal.setAttribute("ry", "9");
      petal.setAttribute("cy", "-1");
      petal.setAttribute("transform", `rotate(${angle})`);
      svg.append(petal);
    }
    const heart = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    heart.setAttribute("r", "2.4");
    heart.setAttribute("class", "heart");
    svg.append(heart);
    return svg;
  }

  function showStatus(status, at) {
    if (!banner || !status) return;
    const lines = status.attention ?? [];
    banner.replaceChildren(
      lines.length
        ? h(
            "section",
            { class: "attention", role: "status", "aria-label": "Needs attention" },
            h("p", { class: "attention-title" }, "Needs attention"),
            h("ul", {}, lines.map((line) => h("li", {}, line))),
          )
        : "",
    );
    const failed = at.bank ? status.banks?.[at.bank]?.failed_chunks : 0;
    const tab = tabs?.querySelector('[data-tab="chunks"]');
    if (tab) {
      tab.querySelector(".count")?.remove();
      if (failed) tab.append(h("span", { class: "count", "aria-label": `${failed} failed` }, String(failed)));
    }
  }

  async function loadStatus() {
    try {
      return await api.status();
    } catch (error) {
      if (error instanceof Unauthorized) throw error;
      return null;
    }
  }

  /// Asks before a destructive action. Resolves true only on the action
  /// button; cancel, Escape and closing resolve false.
  function confirm({ title, body, action }) {
    return new Promise((resolve) => {
      const titleId = `confirm-title-${generation}-${dialogs.size}`;
      const bodyId = `${titleId}-body`;
      let done = false;
      const finish = (confirmed) => {
        if (done) return;
        done = true;
        dialog.close();
        dialog.remove();
        dialogs.delete(dialog);
        resolve(confirmed);
      };
      const cancel = h("button", { type: "button", class: "quiet", onclick: () => finish(false) }, "Cancel");
      const dialog = h(
        "dialog",
        { class: "confirm", "aria-labelledby": titleId, "aria-describedby": bodyId },
        h("h2", { id: titleId }, title),
        h("div", { id: bodyId, class: "confirm-body" }, body),
        h(
          "div",
          { class: "confirm-actions" },
          cancel,
          h("button", { type: "button", class: "danger", onclick: () => finish(true) }, action),
        ),
      );
      dialog.addEventListener("cancel", (event) => {
        event.preventDefault();
        finish(false);
      });
      dialogs.add(dialog);
      document.body.append(dialog);
      dialog.showModal();
      cancel.focus();
    });
  }

  /// Runs an action, then shows what it did, or why the daemon refused. With
  /// `then`, a success moves to that page (the action may have taken this
  /// one away); a refusal stays put.
  async function act(work, { then } = {}) {
    try {
      const said = await work();
      notice = said ? { tone: "done", body: said } : null;
      if (then && window.location.hash !== then) {
        carryNotice = true;
        window.location.hash = then;
        return;
      }
    } catch (error) {
      if (error instanceof Unauthorized) return prompt(error.rejected);
      notice = { tone: "error", body: error instanceof ApiError ? error.message : String(error.message ?? error) };
    }
    await render({ focusNotice: true });
  }

  function noticeElement() {
    if (!notice) return null;
    return h(
      "div",
      { class: "notice", "data-tone": notice.tone, role: notice.tone === "error" ? "alert" : "status", tabindex: "-1" },
      notice.body,
    );
  }

  function prompt(rejected) {
    generation += 1;
    const input = h("input", {
      id: "token",
      type: "password",
      name: "token",
      autocomplete: "current-password",
      required: true,
      spellcheck: "false",
      "aria-describedby": rejected ? "token-help token-error" : "token-help",
    });
    const form = h(
      "form",
      {
        class: "token-card",
        onsubmit: (event) => {
          event.preventDefault();
          const value = input.value.trim();
          if (!value) return;
          api.token.set(value);
          render();
        },
      },
      h("a", { class: "mark", href: "#/", tabindex: "-1" }, flower(), h("span", {}, "Asphodel")),
      h("h1", {}, "Sign in to this daemon"),
      rejected ? h("p", { id: "token-error", class: "notice", "data-tone": "error", role: "alert" }, "The daemon rejected that token. Check ASPHODEL_TOKEN and try again.") : null,
      h("label", { for: "token" }, "Token"),
      input,
      h("p", { id: "token-help", class: "help" }, "The bearer token the daemon was started with (ASPHODEL_TOKEN). It stays in this tab and is gone when you close it."),
      h("button", { type: "submit" }, "Sign in"),
    );
    root.replaceChildren(h("main", { class: "token-page" }, form));
    input.focus();
  }

  async function render({ focusHeading = false, focusNotice = false } = {}) {
    const mine = ++generation;
    const at = route(window.location.hash);
    shell(at);
    const statusLoad = loadStatus();
    const page = pages[at.page] ?? pages.missing;
    const ctx = {
      api,
      ui,
      at,
      bank: at.bank,
      params: at.params,
      status: statusLoad,
      navigate: (hash) => {
        window.location.hash = hash;
      },
      confirm,
      act,
    };
    try {
      const [status, view] = await Promise.all([statusLoad, page(ctx)]);
      if (mine !== generation || stopped) return;
      document.title = view.title ? `${view.title} · Asphodel` : "Asphodel";
      main.replaceChildren(...[noticeElement(), view.content].flat().filter(Boolean));
      main.setAttribute("aria-busy", "false");
      showStatus(status, at);
      if (focusNotice) main.querySelector(".notice")?.focus();
      else if (focusHeading) main.querySelector("h1")?.focus();
    } catch (error) {
      if (mine !== generation || stopped) return;
      if (error instanceof Unauthorized) return prompt(error.rejected);
      main.replaceChildren(
        ...[noticeElement()].filter(Boolean),
        h("h1", { tabindex: "-1" }, "This page didn't load"),
        h("p", { class: "notice", "data-tone": "error", role: "alert" }, error instanceof ApiError ? error.message : String(error.message ?? error)),
        h("p", {}, h("a", { href: at.bank ? `#/banks/${encodeURIComponent(at.bank)}/memories` : "#/" }, at.bank ? "Back to the memories" : "Back to the banks")),
      );
      main.setAttribute("aria-busy", "false");
    }
  }

  function onHashChange() {
    const carried = carryNotice;
    carryNotice = false;
    if (!carried) notice = null;
    for (const dialog of dialogs) dialog.remove();
    dialogs.clear();
    render(carried ? { focusNotice: true } : { focusHeading: true });
  }

  async function poll() {
    if (stopped || !banner) return;
    try {
      const status = await api.status();
      if (!stopped) showStatus(status, route(window.location.hash));
    } catch {
      // The next page load says what went wrong.
    }
  }

  window.addEventListener("hashchange", onHashChange);
  const timer = window.setInterval(poll, STATUS_EVERY_MS);
  render();

  return () => {
    stopped = true;
    window.removeEventListener("hashchange", onHashChange);
    window.clearInterval(timer);
    for (const dialog of dialogs) dialog.remove();
    dialogs.clear();
  };
}
