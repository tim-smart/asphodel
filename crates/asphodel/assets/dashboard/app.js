// The dashboard: hash routes over the daemon's read-only list routes, the
// Recall page's explain (which logs nothing), and the owner's actions (keep, retract, forget, remove a document, retry
// chunks, resume purge, a model in or out of the prompt), each destructive one behind a confirmation.
// A daemon with no tuning file yet gets the setup wizard instead.
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

/// How often, and for how long, the page waits for the daemon to start
/// after setup.
const STARTING_EVERY_MS = 500;
const STARTING_FOR_MS = 120_000;

/// Where an OpenAI-compatible API usually is, as the endpoint's placeholder.
const DEFAULT_ENDPOINT = "https://api.openai.com/v1";

export function mount(root, { fetch }) {
  const document = root.ownerDocument;
  const window = document.defaultView;
  const api = client(fetch, window.sessionStorage);
  const ui = elements(document);
  const { h } = ui;

  let generation = 0;
  let stopped = false;
  // Set while the setup wizard is showing: the routes wait until it's done.
  let settingUp = false;
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
            ["sources", "Sources"],
            ["chunks", "Ingestion"],
            ["recall", "Recall"],
            ["models", "Models"],
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

  /// On a narrow screen the tabs scroll sideways: scrolls this page's tab
  /// into view, sideways only.
  function showCurrentTab() {
    const current = tabs?.querySelector('[aria-current="page"]');
    if (!current) return;
    const over = current.getBoundingClientRect().right - tabs.getBoundingClientRect().right;
    if (over > 0) tabs.scrollLeft += over;
  }

  /// The daemon writes commands in its attention lines between backticks;
  /// they're shown as code, without the backticks.
  function codeSpans(line) {
    return line.split("`").map((part, i) => (i % 2 ? h("code", {}, part) : part));
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

  /// One attention line as the banner shows it. A bank's failed-chunks line,
  /// found from the counts in `status.banks`, links to that bank's ingestion
  /// page in place of the CLI hint, and is left out on that page itself.
  function attentionLine(line, status, at) {
    for (const [name, counts] of Object.entries(status.banks ?? {})) {
      if (!counts.failed_chunks || !line.startsWith(`${name}: ${counts.failed_chunks} failed chunk`)) continue;
      if (at.bank === name && at.section === "chunks") return [];
      return [
        [
          `${line.split(";")[0]}. `,
          h("a", { href: `#/banks/${encodeURIComponent(name)}/chunks` }, "Review them on the Ingestion page"),
        ],
      ];
    }
    return [codeSpans(line)];
  }

  function showStatus(status, at) {
    if (!banner || !status) return;
    const lines = (status.attention ?? []).flatMap((line) => attentionLine(line, status, at));
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
      rejected ? h("p", { id: "token-error", class: "notice", "data-tone": "error", role: "alert" }, ["The daemon rejected that token. Check ", h("code", {}, "ASPHODEL_TOKEN"), " and try again."]) : null,
      h("label", { for: "token" }, "Token"),
      input,
      h("p", { id: "token-help", class: "help" }, "The bearer token the daemon was started with, from ", h("code", {}, "ASPHODEL_TOKEN"), ". It's kept in this tab only and cleared when you close it."),
      h("button", { type: "submit" }, "Sign in"),
    );
    root.replaceChildren(h("main", { class: "token-page" }, form));
    input.focus();
  }

  /// A labelled text input in a setup card, with help under it.
  function field(id, label, attributes, help) {
    const input = h("input", { id, name: id, type: "text", autocomplete: "off", spellcheck: "false", ...attributes });
    const helpId = help ? `${id}-help` : null;
    if (helpId) input.setAttribute("aria-describedby", helpId);
    return {
      input,
      nodes: [h("label", { for: id }, label), input, help ? h("p", { id: helpId, class: "help" }, help) : null],
    };
  }

  /// The setup wizard: the code from the data dir, then the LLM. It writes
  /// the tuning file and the secrets into the data dir, and the daemon goes
  /// on starting.
  function setupPage(state) {
    settingUp = true;
    generation += 1;
    const codeFile = state.code_file ?? "the data dir's setup-code file";
    const code = field(
      "setup-code",
      "Setup code",
      { type: "password", required: true, autocomplete: "one-time-code" },
      [
        "It's in ",
        h("code", {}, codeFile),
        ", which only someone with access to the daemon's data dir can read. In Kubernetes: ",
        h("code", {}, `kubectl exec <pod> -c asphodel -- cat ${codeFile}`),
        ". If ",
        h("code", {}, "ASPHODEL_TOKEN"),
        " is set, it works here too.",
      ],
    );
    let mode = "api_key";
    const modes = h(
      "fieldset",
      { class: "segmented setup-modes" },
      h("legend", {}, "The LLM"),
      [
        ["api_key", "API key", "An OpenAI-compatible API"],
        ["chatgpt", "ChatGPT", "A ChatGPT subscription"],
        ["later", "Later", "Ingest waits on the queue"],
      ].map(([value, label, detail]) =>
        h(
          "label",
          { class: "segment" },
          h("input", { type: "radio", name: "setup-llm", value, checked: value === mode, onchange: () => setMode(value) }),
          h("span", {}, h("strong", {}, label), h("span", { class: "segment-detail" }, detail)),
        ),
      ),
    );
    const endpoint = field("setup-endpoint", "Endpoint", { type: "url", placeholder: DEFAULT_ENDPOINT, value: DEFAULT_ENDPOINT });
    const model = field("setup-model", "Model", { required: true }, "The exact model string. The floors are calibrated against one model.");
    const apiKey = state.llm_api_key_from_env
      ? { input: null, nodes: [h("p", { class: "help" }, h("code", {}, "ASPHODEL_LLM_API_KEY"), " is set, and the daemon uses it.")] }
      : field("setup-api-key", "API key", { type: "password", autocomplete: "off" }, "Kept in the data dir's secrets file, readable by the daemon only. Leave it empty for an endpoint that takes none.");
    const effort = field("setup-effort", "Reasoning effort", { placeholder: "low, medium or high" }, "Optional. Left empty, the backend decides.");
    const language = field("setup-language", "Language", { placeholder: "English" }, "Optional. Every memory is written in it, translated if need be.");
    // Each mode's fields sit in a fieldset that's disabled, not just hidden,
    // so a hidden required field doesn't block the form.
    const keyFields = h("fieldset", { class: "setup-fields" }, endpoint.nodes, apiKey.nodes);
    const llmFields = h("fieldset", { class: "setup-fields" }, model.nodes, effort.nodes, language.nodes);
    const loginNote = h(
      "p",
      { class: "help", hidden: true },
      "Once setup is done, log the daemon in with ",
      h("code", {}, "asphodel llm login --data-dir <data dir>"),
      ". Extraction waits until it is.",
    );
    function setMode(value) {
      mode = value;
      keyFields.disabled = keyFields.hidden = mode !== "api_key";
      llmFields.disabled = llmFields.hidden = mode === "later";
      loginNote.hidden = mode !== "chatgpt";
    }
    const error = h("p", { class: "notice", "data-tone": "error", role: "alert", tabindex: "-1", hidden: true });
    const submit = h("button", { type: "submit" }, "Finish setup");
    const form = h(
      "form",
      {
        class: "token-card setup-card",
        onsubmit: async (event) => {
          event.preventDefault();
          const value = (input) => input?.value.trim() || undefined;
          const body =
            mode === "later"
              ? {}
              : {
                  llm: {
                    auth: mode,
                    endpoint: mode === "api_key" ? value(endpoint.input) : undefined,
                    model: value(model.input),
                    api_key: mode === "api_key" ? value(apiKey.input) : undefined,
                    reasoning_effort: value(effort.input),
                    language: value(language.input),
                  },
                };
          submit.disabled = true;
          error.hidden = true;
          try {
            setupDone(await api.setup(code.input.value.trim(), body));
          } catch (failure) {
            submit.disabled = false;
            error.textContent =
              failure instanceof ApiError && failure.status === 401
                ? "That isn't the setup code."
                : String(failure.message ?? failure);
            error.hidden = false;
            error.focus();
          }
        },
      },
      h("a", { class: "mark", href: "#/", tabindex: "-1" }, flower(), h("span", {}, "Asphodel")),
      h("h1", {}, "Set up this daemon"),
      h(
        "p",
        { class: "help" },
        "There's no tuning file yet. This writes one into the data dir, with any secrets beside it, and starts the daemon. Change it later by editing the file and restarting.",
      ),
      error,
      code.nodes,
      modes,
      keyFields,
      llmFields,
      loginNote,
      state.makes_token
        ? h("p", { class: "help" }, "The daemon listens off loopback, so setup makes a bearer token and shows it once.")
        : null,
      state.uncalibrated?.length
        ? h(
            "div",
            { class: "notice setup-uncalibrated", "data-tone": "warning", role: "note" },
            h("p", {}, h("strong", {}, "Uncalibrated floor. "), "Setup writes this placeholder, which nobody has calibrated yet:"),
            h("ul", {}, state.uncalibrated.map((line) => h("li", {}, h("code", {}, line)))),
            h(
              "p",
              {},
              "It decides which new claims are matched against existing memories, so a poor value can duplicate or merge memories. Calibrate it in replay (",
              h("code", {}, "docs/replay.md"),
              ") and edit the tuning file before relying on the store.",
            ),
          )
        : null,
      submit,
    );
    setMode(mode);
    root.replaceChildren(h("main", { class: "token-page" }, form));
    code.input.focus();
  }

  /// What setup did, and the token it made, shown this once.
  function setupDone(done) {
    if (done.token) api.token.set(done.token);
    const open = h("button", { type: "button", onclick: () => openAfterSetup(open) }, "Open the dashboard");
    const card = h(
      "section",
      { class: "token-card setup-card", "aria-labelledby": "setup-done" },
      h("a", { class: "mark", href: "#/", tabindex: "-1" }, flower(), h("span", {}, "Asphodel")),
      h("h1", { id: "setup-done", tabindex: "-1" }, "Asphodel is set up"),
      done.token
        ? [
            h("label", { for: "setup-token" }, "Bearer token"),
            h("input", { id: "setup-token", type: "text", readonly: true, value: done.token, spellcheck: "false" }),
            h(
              "p",
              { class: "help" },
              "Give it to Hermes and every other client as ",
              h("code", {}, "ASPHODEL_TOKEN"),
              ". It's shown here once, and kept in ",
              h("code", {}, done.secrets),
              ".",
            ),
          ]
        : null,
      done.llm_login
        ? h("p", { class: "help" }, "Log the daemon in to ChatGPT with ", h("code", {}, `asphodel llm login --data-dir ${done.data_dir}`), ". Extraction waits until it is.")
        : null,
      h(
        "p",
        { class: "help" },
        "The tuning file is ",
        h("code", {}, done.config),
        ". Its floors are starting values: recalibrate them in replay.",
      ),
      open,
    );
    root.replaceChildren(h("main", { class: "token-page" }, card));
    card.querySelector("h1").focus();
  }

  /// Waits for the daemon to finish starting, then shows the banks.
  async function openAfterSetup(button) {
    button.disabled = true;
    button.textContent = "Starting…";
    const deadline = Date.now() + STARTING_FOR_MS;
    while (!stopped && Date.now() < deadline) {
      try {
        if ((await api.health())?.ready) break;
      } catch {
        // Still starting.
      }
      await new Promise((resolve) => window.setTimeout(resolve, STARTING_EVERY_MS));
    }
    if (stopped) return;
    settingUp = false;
    if (window.location.hash && window.location.hash !== "#/") window.location.hash = "#/";
    else render({ focusHeading: true });
  }

  async function start() {
    const state = await api.setupState();
    if (stopped) return;
    if (state?.needed) setupPage(state);
    else render();
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
      showCurrentTab();
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
    if (settingUp) return;
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
  start();

  return () => {
    stopped = true;
    window.removeEventListener("hashchange", onHashChange);
    window.clearInterval(timer);
    for (const dialog of dialogs) dialog.remove();
    dialogs.clear();
  };
}
