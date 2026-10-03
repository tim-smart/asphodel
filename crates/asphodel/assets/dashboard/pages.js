// The dashboard's pages. Each takes the render context and returns its
// title and content; actions go through `ctx.act`, destructive ones after
// `ctx.confirm`.

import { capitalize, fadeStage, goneReason, plural, shortId } from "./dom.js";

const STATUSES = ["live", "superseded", "ended", "retracted", "forgetting"];
const KINDS = ["fact", "event", "state", "task", "recurring"];
const LEVELS = ["trivial", "minor", "notable", "major", "critical", "kept"];
const SORTS = [
  ["created", "Newest first"],
  ["fade", "Fading soonest"],
  ["strength", "Strongest first"],
];
const FADING = [
  ["faded", "Already faded"],
  ["week", "Fading within a week"],
  ["month", "Fading within a month"],
  ["never", "Never fading"],
];

/// `#/banks/{bank}/{section}/{id}?query` as a route.
export function route(hash) {
  const [path, search = ""] = (hash ?? "").replace(/^#/, "").split("?");
  const parts = path
    .split("/")
    .filter(Boolean)
    .map((part) => {
      try {
        return decodeURIComponent(part);
      } catch {
        return part;
      }
    });
  const params = Object.fromEntries(new URLSearchParams(search));
  if (parts.length === 0) return { page: "banks", params };
  if (parts[0] !== "banks" || !parts[1]) return { page: "missing", params };
  const [, bank, section = "memories", id] = parts;
  const page =
    section === "memories"
      ? id
        ? "memory"
        : "memories"
      : section === "documents" && !id
        ? "documents"
        : section === "sources" && id
          ? "source"
          : section === "chunks" && !id
            ? "chunks"
            : "missing";
  return { page, bank, section, id, params };
}

const bankHash = (bank, ...rest) => `#/banks/${[bank, ...rest].map(encodeURIComponent).join("/")}`;

const memoryHash = (bank, id) => bankHash(bank, "memories", id);
const sourceHash = (bank, id) => bankHash(bank, "sources", id);

function listHash(bank, params) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) if (value) search.set(key, value);
  const text = search.toString();
  return `${bankHash(bank, "memories")}${text ? `?${text}` : ""}`;
}

function memoryCount(count) {
  return plural(count, "memory", "memories");
}

/// What a memory's page knows about where it stands, in the list's terms.
function memoryStatus(view) {
  if (view.hidden_at) return "forgetting";
  if (view.retracted_at) return "retracted";
  if (view.chain.head !== view.id) return "superseded";
  if (view.chain.ended_by) return "ended";
  return "live";
}

/// Children with a space between each, so their text reads apart when it's
/// copied or read aloud.
function spaced(children) {
  return children.filter(Boolean).flatMap((child, i) => (i ? [" ", child] : [child]));
}

function heading(h, title, ...after) {
  return h("div", { class: "page-head" }, h("h1", { tabindex: "-1" }, title), ...after);
}

function counts(h, label, rows) {
  return h(
    "div",
    { class: "count-group" },
    h("h3", {}, label),
    h(
      "dl",
      {},
      rows
        .filter(Boolean)
        .map(([term, value, extra]) => h("div", { class: "count" }, h("dt", {}, term), h("dd", {}, value, extra ? [" ", extra] : null))),
    ),
  );
}

// The banks: each bank's counts, the purge pause and the daemon's state.
async function banks(ctx) {
  const { api, ui } = ctx;
  const { h, time } = ui;
  const [{ banks: list }, status] = await Promise.all([api.banks(), ctx.status]);
  const plan = status?.purge?.state === "paused" ? await api.purgePlan() : null;

  const purge = plan
    ? h(
        "section",
        { class: "card purge-card", "aria-labelledby": "purge-title" },
        h("h2", { id: "purge-title" }, "Purge is paused"),
        h(
          "p",
          {},
          "The deletion settings changed since purge last ran, so the nightly sweep deletes nothing until you acknowledge the new ones.",
        ),
        plan.changed.length
          ? h("p", {}, "Changed: ", plan.changed.map((key, i) => [i ? ", " : "", h("code", {}, key)]))
          : null,
        h("p", {}, "Under the new settings the next sweep deletes:"),
        h(
          "ul",
          { class: "purge-counts" },
          h("li", {}, h("strong", {}, String(plan.memories)), " ", plan.memories === 1 ? "memory" : "memories", " (every version counted)"),
          h("li", {}, h("strong", {}, String(plan.sources)), " ", plan.sources === 1 ? "source" : "sources"),
          h("li", {}, h("strong", {}, String(plan.failed_chunks)), " ", plan.failed_chunks === 1 ? "failed chunk" : "failed chunks"),
          h("li", {}, h("strong", {}, String(plan.recalls)), " ", plan.recalls === 1 ? "recall" : "recalls"),
        ),
        h(
          "button",
          {
            type: "button",
            class: "danger",
            onclick: async () => {
              const confirmed = await ctx.confirm({
                title: "Resume purge?",
                body: [
                  h(
                    "p",
                    {},
                    `The next sweep deletes ${memoryCount(plan.memories)}, ${plural(plan.sources, "source")}, ${plural(plan.failed_chunks, "failed chunk")} and ${plural(plan.recalls, "recall")} under the new settings.`,
                  ),
                  h("p", {}, "Purged memories can't be brought back."),
                ],
                action: "Acknowledge and resume purge",
              });
              if (!confirmed) return;
              ctx.act(async () => {
                await api.purgeAck(plan.current);
                return "Purge resumed. The next sweep runs under the new settings.";
              });
            },
          },
          "Acknowledge and resume purge…",
        ),
      )
    : null;

  const cards = list.map((bank) => {
    const m = bank.memories;
    const failedLink =
      bank.chunks.failed > 0
        ? h("a", { href: bankHash(bank.name, "chunks") }, String(bank.chunks.failed))
        : String(bank.chunks.failed);
    return h(
      "article",
      { class: "card bank" },
      h(
        "header",
        {},
        h("h2", {}, h("a", { href: bankHash(bank.name, "memories") }, bank.name)),
        h(
          "p",
          { class: "quiet-text" },
          [bank.owner_name, bank.assistant_name].filter(Boolean).join(" and ") || "No names yet",
          " · ",
          bank.timezone,
        ),
      ),
      h(
        "div",
        { class: "count-groups" },
        counts(h, "Memories", [
          ["Live", String(m.live)],
          ["Kept", String(bank.kept)],
          ["Superseded", String(m.superseded)],
          ["Ended", String(m.ended)],
          ["Retracted", String(m.retracted)],
          m.forgetting ? ["Forgetting", String(m.forgetting)] : null,
        ]),
        counts(h, "Sources", [
          [
            h("a", { href: bankHash(bank.name, "documents") }, "Documents"),
            String(bank.sources.documents),
            bank.sources.document_versions !== bank.sources.documents
              ? h("span", { class: "quiet-text" }, `(${plural(bank.sources.document_versions, "version")})`)
              : null,
          ],
          ["Turns", String(bank.sources.turns)],
          bank.sources.removed ? ["Removed", String(bank.sources.removed)] : null,
        ]),
        counts(h, "Ingestion", [
          ["Queued", String(bank.chunks.queued)],
          ["Failed", failedLink],
        ]),
      ),
      h(
        "footer",
        { class: "quiet-text" },
        bank.last_turn_at ? ["Last turn ", time(bank.last_turn_at, { withTime: true })] : "No turns yet",
        ` · ${plural(bank.models, "mental model")}`,
      ),
    );
  });

  const daemon = status
    ? h(
        "section",
        { class: "card daemon", "aria-labelledby": "daemon-title" },
        h("h2", { id: "daemon-title" }, "Daemon"),
        h(
          "dl",
          { class: "facts" },
          h("div", {}, h("dt", {}, "Version"), h("dd", {}, status.version)),
          h("div", {}, h("dt", {}, "Purge"), h("dd", {}, status.purge?.state === "paused" ? "Paused" : "Running")),
          h("div", {}, h("dt", {}, "Last sweep"), h("dd", {}, status.last_sweep ? time(status.last_sweep.completed_at, { withTime: true }) : "Not yet")),
          h("div", {}, h("dt", {}, "Last backup"), h("dd", {}, status.last_backup_at ? time(status.last_backup_at, { withTime: true }) : "Never")),
        ),
      )
    : null;

  return {
    title: "Banks",
    content: [
      heading(h, "Banks"),
      purge,
      list.length
        ? h("div", { class: "bank-grid" }, cards)
        : h("p", { class: "empty" }, "No banks yet. A bank appears when the Hermes plugin first connects to this daemon."),
      daemon,
    ],
  };
}

function select(h, { id, label, value, options, onchange }) {
  return h(
    "div",
    { class: "field" },
    h("label", { for: id }, label),
    h(
      "select",
      { id, onchange: (event) => onchange(event.target.value) },
      options.map(([v, text]) => h("option", { value: v, selected: v === value }, text)),
    ),
  );
}

function memoryRow(ctx, memory) {
  const { h, fadeLabel, pill, time } = ctx.ui;
  const kept = memory.significance.effective === "kept";
  return h(
    "li",
    { class: "memory", "data-fade": fadeStage(memory.fade), "data-status": memory.status },
    h(
      "a",
      { class: "sentence", href: memoryHash(ctx.bank, memory.id) },
      kept ? h("span", { class: "seal", title: "Kept" }, "✻ ") : null,
      memory.sentence,
    ),
    " ",
    h(
      "div",
      { class: "meta" },
      spaced([
        memory.status !== "live" ? pill(memory.status) : null,
        h("span", {}, memory.kind),
        h("span", {}, kept ? "kept" : memory.significance.effective),
        h("span", {}, memory.document_id ?? (memory.source_kind === "turn" ? "conversation" : memory.source_kind)),
        h("span", {}, "observed ", time(memory.observed_at)),
      ]),
    ),
    " ",
    fadeLabel(memory.fade),
  );
}

// A bank's memories, filtered and sorted by the daemon.
async function memories(ctx) {
  const { api, ui, bank, params } = ctx;
  const { h } = ui;
  const filters = {
    status: params.status ?? "",
    q: params.q ?? "",
    sort: params.sort ?? "",
    kind: params.kind ?? "",
    fading: params.fading ?? "",
    significance: params.significance ?? "",
  };
  const page = await api.memories(bank, { ...filters, limit: 50 });
  const go = (change) => ctx.navigate(listHash(bank, { ...filters, ...change }));
  const all = STATUSES.reduce((sum, s) => sum + (page.statuses[s] ?? 0), 0);

  const search = h("input", { id: "search", type: "search", name: "q", value: filters.q, autocomplete: "off" });
  const toolbar = h(
    "div",
    { class: "toolbar" },
    h(
      "form",
      {
        role: "search",
        class: "search",
        onsubmit: (event) => {
          event.preventDefault();
          go({ q: search.value.trim() });
        },
      },
      h("label", { for: "search" }, "Search memories"),
      h("div", { class: "search-row" }, search, h("button", { type: "submit" }, "Search")),
    ),
    select(h, {
      id: "sort",
      label: "Sort",
      value: filters.sort || "created",
      options: SORTS,
      onchange: (sort) => go({ sort: sort === "created" ? "" : sort }),
    }),
    select(h, {
      id: "kind",
      label: "Kind",
      value: filters.kind,
      options: [["", "Any kind"], ...KINDS.map((k) => [k, capitalize(k)])],
      onchange: (kind) => go({ kind }),
    }),
    select(h, {
      id: "fading",
      label: "Fading",
      value: filters.fading,
      options: [["", "Any time"], ...FADING],
      onchange: (fading) => go({ fading }),
    }),
    select(h, {
      id: "significance",
      label: "Significance",
      value: filters.significance,
      options: [["", "Any level"], ...LEVELS.map((l) => [l, capitalize(l)])],
      onchange: (significance) => go({ significance }),
    }),
  );

  const chip = (status, label, count) =>
    h(
      "button",
      {
        type: "button",
        class: "chip",
        "data-status": status || "all",
        "aria-pressed": String(filters.status === status),
        onclick: () => go({ status }),
      },
      label,
      " ",
      h("span", { class: "chip-count" }, String(count)),
    );
  const chips = h(
    "div",
    { class: "chips", role: "group", "aria-label": "Status" },
    chip("", "All", all),
    STATUSES.map((s) => chip(s, capitalize(s), page.statuses[s] ?? 0)),
  );

  const rows = h("ol", { class: "memories" }, page.memories.map((m) => memoryRow(ctx, m)));
  let cursor = page.next_cursor;
  const more = h(
    "button",
    {
      type: "button",
      class: "quiet more",
      onclick: async () => {
        more.disabled = true;
        try {
          const next = await api.memories(bank, { ...filters, cursor, limit: 50 });
          rows.append(...next.memories.map((m) => memoryRow(ctx, m)));
          cursor = next.next_cursor;
        } catch (error) {
          more.after(h("p", { class: "notice", "data-tone": "error", role: "alert" }, error.message));
        }
        more.disabled = false;
        if (!cursor) more.remove();
      },
    },
    "Load more memories",
  );
  const filtered = Object.entries(filters).some(([key, value]) => key !== "sort" && value);

  return {
    title: `Memories · ${bank}`,
    content: [
      heading(h, "Memories", h("p", { class: "quiet-text" }, `${memoryCount(page.total)} listed, as of `, ui.time(page.as_of, { withTime: true }))),
      toolbar,
      chips,
      page.memories.length
        ? rows
        : h(
            "p",
            { class: "empty" },
            filtered ? ["No memories match these filters. ", h("a", { href: bankHash(bank, "memories") }, "Clear the filters")] : "This bank has no memories yet.",
          ),
      cursor ? more : null,
      h(
        "p",
        { class: "footnote" },
        "Fade dates assume a memory isn't used again. Bank time only moves while the bank is in use, so each date is the soonest it can come.",
      ),
    ],
  };
}

const GUARDS = {
  purge_disabled: "Purge is turned off (no δ is set).",
  purge_paused: "Purge is paused until the new deletion settings are acknowledged.",
  forgotten: "It's being forgotten; the erase removes it instead.",
  strength: "The chain's head is still above the purge line.",
};

function guardText(ui, guard) {
  if (guard.guard === "date_ahead") return ["Its start or end date hasn't passed yet (", ui.time(guard.until), ")."];
  if (guard.guard === "overdue_task") return ["It's an overdue task, held until ", ui.time(guard.until), "."];
  return GUARDS[guard.guard] ?? guard.guard;
}

// One memory: where it came from, how strong it is, when it fades, and the
// owner's actions on it.
async function memory(ctx) {
  const { api, ui, bank, at } = ctx;
  const { h, time, fadeLabel, purgeLabel, pill } = ui;
  const view = await api.memory(bank, at.id);
  const status = memoryStatus(view);
  const kept = view.significance.owner === "kept";
  const versions = view.chain.members.length;
  const link = (id, text) => h("a", { href: memoryHash(bank, id) }, text ?? `memory ${shortId(id)}`);

  const source = view.source;
  const sourceName = source.document_id ?? (source.session_id ? `Turn in session ${source.session_id}` : "Its source");

  const actions = [];
  if (status !== "forgetting") {
    if (status === "live" || status === "ended") {
      actions.push(
        h(
          "div",
          { class: "action" },
          h(
            "button",
            {
              type: "button",
              onclick: () =>
                ctx.act(async () => {
                  if (kept) {
                    await api.unkeep(bank, [view.id]);
                    return "No longer kept. It fades like any other memory again.";
                  }
                  await api.keep(bank, [view.id]);
                  return "Kept. It won't fade while it's kept.";
                }),
            },
            kept ? "Stop keeping" : "Keep",
          ),
          h("p", {}, kept ? "Let it fade again when it goes unused." : "Hold it at full strength so it never fades."),
        ),
      );
    }
    if (status !== "retracted" && status !== "superseded") {
      actions.push(
        h(
          "div",
          { class: "action" },
          h(
            "button",
            {
              type: "button",
              class: "danger-outline",
              onclick: async () => {
                const ends = view.chain.ends.length;
                const confirmed = await ctx.confirm({
                  title: "Retract this memory?",
                  body: [
                    h("p", { class: "quoted" }, view.sentence),
                    h(
                      "p",
                      {},
                      "Retracting says it never held. Recall, the agenda and the system prompt stop using it at once, and it fades out on its own.",
                    ),
                    ends ? h("p", {}, `It also reopens ${memoryCount(ends)} it ended.`) : null,
                    h("p", {}, "There's no undo."),
                  ],
                  action: "Retract memory",
                });
                if (!confirmed) return;
                ctx.act(async () => {
                  const done = await api.retract(bank, view.id);
                  return done.reopened.length
                    ? ["Retracted. Reopened ", memoryCount(done.reopened.length), ": ", done.reopened.map((id, i) => [i ? ", " : "", link(id)]), "."]
                    : "Retracted. Recall, the agenda and the system prompt no longer use it.";
                });
              },
            },
            "Retract…",
          ),
          h("p", {}, "It's wrong: take it out of use and reopen what it ended."),
        ),
      );
    }
    actions.push(
      h(
        "div",
        { class: "action" },
        h(
          "button",
          {
            type: "button",
            class: "danger-outline",
            onclick: async () => {
              const confirmed = await ctx.confirm({
                title: "Forget this memory?",
                body: [
                  h("p", { class: "quoted" }, view.sentence),
                  h(
                    "p",
                    {},
                    versions > 1
                      ? `Forget erases it and every version of it in its chain (${plural(versions, "version")}), and masks the passages they came from.`
                      : "Forget erases it and every version of it, and masks the passage it came from.",
                  ),
                  h("p", {}, "It can't be undone."),
                ],
                action: "Forget memory",
              });
              if (!confirmed) return;
              // The erase may run before forget answers, so the result is
              // shown on the list rather than on this memory's page.
              ctx.act(
                async () => {
                  const done = await api.forget(bank, [view.id]);
                  return done.forgotten.length
                    ? `Forgot ${memoryCount(done.forgotten.length)}: “${view.sentence}” and every version in its chain.`
                    : "Nothing to forget: it was already gone.";
                },
                { then: bankHash(bank, "memories") },
              );
            },
          },
          "Forget…",
        ),
        h("p", {}, "Erase it and every version, as if it was never said."),
      ),
    );
  }

  const chain =
    versions > 1 || view.chain.ended_by || view.chain.ends.length
      ? h(
          "section",
          { class: "panel", "aria-labelledby": "chain-title" },
          h("h2", { id: "chain-title" }, "Versions and links"),
          versions > 1
            ? h(
                "ol",
                { class: "chain" },
                view.chain.members.map((member) =>
                  h(
                    "li",
                    {},
                    member.id === view.id ? h("span", { "aria-current": "page" }, `This memory (${shortId(member.id)})`) : link(member.id),
                    member.retracted ? " · retracted" : null,
                    member.hidden ? " · being forgotten" : null,
                    member.superseded_by ? " · refined by the next" : " · newest",
                  ),
                ),
              )
            : null,
          view.chain.ended_by ? h("p", {}, "Ended by ", link(view.chain.ended_by), ".") : null,
          view.chain.ends.length ? h("p", {}, "It ended ", view.chain.ends.map((id, i) => [i ? ", " : "", link(id)]), ".") : null,
        )
      : null;

  const strength = view.strength;
  return {
    title: view.sentence,
    content: [
      h("p", { class: "back" }, h("a", { href: bankHash(bank, "memories") }, "← Memories")),
      h(
        "article",
        { class: "specimen", "data-fade": fadeStage(view.projection.fade), "data-status": status },
        h(
          "p",
          { class: "eyebrow" },
          spaced([
            pill(status),
            h("span", {}, view.kind),
            h("span", {}, kept ? "kept" : `${view.significance.effective} significance`),
            view.phase ? h("span", {}, view.phase) : null,
          ]),
        ),
        h("h1", { class: "sentence", tabindex: "-1" }, kept ? h("span", { class: "seal", title: "Kept" }, "✻ ") : null, view.sentence),
        h(
          "div",
          { class: "outlook" },
          h("div", { class: "outlook-fade" }, fadeLabel(view.projection.fade)),
          h("div", { class: "outlook-purge" }, purgeLabel(view.projection.purge)),
          !view.projection.purge && view.purge.guards.length
            ? h("ul", { class: "guards" }, view.purge.guards.map((g) => h("li", {}, guardText(ui, g))))
            : null,
          h("p", { class: "footnote" }, `${capitalize(view.projection.basis)}.`),
        ),
      ),
      h(
        "div",
        { class: "memory-grid" },
        h(
          "div",
          { class: "memory-main" },
          h(
            "section",
            { class: "panel", "aria-labelledby": "source-title" },
            h("h2", { id: "source-title" }, "Where it came from"),
            source.passage
              ? h("blockquote", { class: "passage" }, source.passage)
              : h("p", { class: "quiet-text" }, goneReason(source.gone) ?? "The passage is gone."),
            h(
              "p",
              {},
              h("a", { href: sourceHash(bank, source.source) }, sourceName),
              source.message_at ? [" · ", time(source.message_at, { withTime: true })] : null,
            ),
            source.secret_kinds.length ? h("p", { class: "quiet-text" }, `Secrets were redacted before storage: ${source.secret_kinds.join(", ")}.`) : null,
          ),
          chain,
          view.entities.length
            ? h(
                "section",
                { class: "panel", "aria-labelledby": "entities-title" },
                h("h2", { id: "entities-title" }, "About"),
                h("ul", { class: "inline-list" }, view.entities.map((e) => h("li", {}, e.name))),
              )
            : null,
          h(
            "details",
            { class: "panel" },
            h("summary", {}, `History: ${plural(view.accesses.length, "access", "accesses")}, ${plural(view.edits.length, "edit")}`),
            h(
              "ul",
              { class: "history" },
              view.accesses.map((a) => h("li", {}, h("span", {}, a.kind.replaceAll("_", " ")), " ", time(a.at, { withTime: true }), a.inherited_from ? [" · inherited from ", link(a.inherited_from)] : null)),
              view.edits.map((e) => h("li", {}, h("span", {}, e.kind.replaceAll("_", " ")), " ", time(e.at, { withTime: true }))),
            ),
          ),
        ),
        h(
          "aside",
          { class: "memory-side" },
          h(
            "section",
            { class: "panel", "aria-labelledby": "strength-title" },
            h("h2", { id: "strength-title" }, "Strength"),
            h(
              "dl",
              { class: "facts" },
              h("div", {}, h("dt", {}, "Now"), h("dd", {}, strength.value === null ? "—" : strength.value.toFixed(2))),
              h("div", {}, h("dt", {}, "Recall finds it below"), h("dd", {}, strength.threshold.toFixed(2))),
              h("div", {}, h("dt", {}, "Recallable"), h("dd", {}, strength.recallable ? "Yes" : "No")),
              h("div", {}, h("dt", {}, "Occasions used"), h("dd", {}, String(strength.occasions))),
              h("div", {}, h("dt", {}, "Observed"), h("dd", {}, time(view.observed_at, { withTime: true }))),
              h("div", {}, h("dt", {}, "Remembered"), h("dd", {}, time(view.created_at, { withTime: true }))),
            ),
          ),
          actions.length
            ? h("section", { class: "panel actions", "aria-labelledby": "actions-title" }, h("h2", { id: "actions-title" }, "Actions"), actions)
            : null,
        ),
      ),
    ],
  };
}

/// A chunk's headings, stored as a JSON array, as "Home › Garden".
function headingPath(path) {
  if (!path) return null;
  try {
    const parts = JSON.parse(path);
    return Array.isArray(parts) ? parts.join(" › ") || null : String(parts);
  } catch {
    return path;
  }
}

function groupDocuments(sources) {
  const groups = new Map();
  for (const source of sources) {
    const key = source.document_id ?? source.id;
    const group = groups.get(key) ?? { id: key, newest: source, versions: [] };
    group.versions.push(source);
    groups.set(key, group);
  }
  return [...groups.values()];
}

// The bank's documents, one entry per document id, newest version first.
async function documents(ctx) {
  const { api, ui, bank } = ctx;
  const { h, time, pill } = ui;
  const first = await api.sources(bank, { kind: "document", limit: 200 });
  const loaded = [...first.sources];
  let cursor = first.next_cursor;
  const list = h("ul", { class: "documents" });

  const draw = () =>
    list.replaceChildren(
      ...groupDocuments(loaded).map(({ id, newest, versions }) => {
        const chunks = versions.reduce((sum, v) => sum + v.chunks, 0);
        const failed = versions.reduce((sum, v) => sum + v.failed, 0);
        const remembered = versions.reduce((sum, v) => sum + v.memories, 0);
        return h(
          "li",
          { class: "document", "data-gone": newest.gone?.reason ?? null },
          h("a", { class: "document-name", href: sourceHash(bank, newest.id) }, id),
          " ",
          h(
            "div",
            { class: "meta" },
            spaced([
              newest.gone ? pill(newest.gone.reason.replaceAll("_", " "), "gone") : null,
              h("span", {}, plural(versions.length, "version")),
              h("span", {}, plural(chunks, "chunk")),
              h("span", {}, memoryCount(remembered)),
              failed ? pill(`${failed} failed`, "failed") : null,
              h("span", {}, "ingested ", time(newest.ingested_at)),
            ]),
          ),
        );
      }),
    );
  draw();

  const more = h(
    "button",
    {
      type: "button",
      class: "quiet more",
      onclick: async () => {
        more.disabled = true;
        const next = await api.sources(bank, { kind: "document", limit: 200, cursor });
        loaded.push(...next.sources);
        cursor = next.next_cursor;
        draw();
        more.disabled = false;
        if (!cursor) more.remove();
      },
    },
    "Load more documents",
  );

  return {
    title: `Documents · ${bank}`,
    content: [
      heading(h, "Documents"),
      loaded.length ? list : h("p", { class: "empty" }, "No documents yet. Documents the agent ingests show up here."),
      cursor ? more : null,
    ],
  };
}

// One source, a document version or a turn: its text, versions and chunks.
async function source(ctx) {
  const { api, ui, bank, at } = ctx;
  const { h, time, pill } = ui;
  const s = await api.source(bank, at.id);
  const isDocument = s.kind === "document";
  const removed = s.gone?.reason === "removed";
  const title = isDocument ? s.document_id : `Turn${s.session_id ? ` in session ${s.session_id}` : ""}`;

  const remove =
    isDocument && !removed
      ? h(
          "section",
          { class: "panel danger-zone", "aria-labelledby": "remove-title" },
          h("h2", { id: "remove-title" }, "Remove this document"),
          h("p", {}, "Removes every version and forgets the memories resting on it. Memories that only mention it stay."),
          h(
            "button",
            {
              type: "button",
              class: "danger-outline",
              onclick: async () => {
                const versions = s.versions.filter((v) => v.gone?.reason !== "removed").length || 1;
                const confirmed = await ctx.confirm({
                  title: "Remove this document?",
                  body: [
                    h("p", { class: "quoted" }, s.document_id),
                    h(
                      "p",
                      {},
                      `This removes every version of it (${plural(versions, "version")}) and forgets the memories resting on it, with all their versions. Its text is deleted now, and sending the same text again is ignored as a duplicate.`,
                    ),
                    h("p", {}, "It can't be undone here. A backup, or a later edited copy of the document, can bring the content back."),
                  ],
                  action: "Remove document",
                });
                if (!confirmed) return;
                ctx.act(async () => {
                  const done = await api.removeDocument(bank, s.document_id);
                  const dequeued = done.dequeued ? ` and took ${plural(done.dequeued, "chunk")} off the queue` : "";
                  return `Removed ${done.document_id}: ${plural(done.sources.length, "version")}. Forgot ${memoryCount(done.forgotten.length)}${dequeued}.`;
                });
              },
            },
            "Remove document…",
          ),
        )
      : null;

  const chunks = s.chunks.length
    ? h(
        "section",
        { class: "panel", "aria-labelledby": "chunks-title" },
        h("h2", { id: "chunks-title" }, plural(s.chunks.length, "chunk")),
        h(
          "div",
          { class: "table-wrap" },
          h(
            "table",
            {},
            h("thead", {}, h("tr", {}, ["#", "Section", "State", "Memories"].map((c) => h("th", { scope: "col" }, c)))),
            h(
              "tbody",
              {},
              s.chunks.map((c) =>
                h(
                  "tr",
                  { "data-state": c.state },
                  h("td", { class: "num" }, String(c.position + 1)),
                  h("td", {}, headingPath(c.heading_path) ?? h("span", { class: "quiet-text" }, "—")),
                  h(
                    "td",
                    {},
                    pill(c.state.replaceAll("_", " "), c.state),
                    c.state === "failed" && c.error_kind ? [" ", h("code", {}, c.error_kind), ` after ${plural(c.error_count, "attempt")}`] : null,
                  ),
                  h(
                    "td",
                    {},
                    c.memories.length
                      ? h("ul", { class: "inline-list" }, c.memories.map((id) => h("li", {}, h("a", { href: memoryHash(bank, id) }, `memory ${shortId(id)}`))))
                      : h("span", { class: "quiet-text" }, "none"),
                  ),
                ),
              ),
            ),
          ),
        ),
        s.chunks.some((c) => c.state === "failed")
          ? h("p", {}, h("a", { href: bankHash(bank, "chunks") }, "Retry failed chunks on the ingestion page"))
          : null,
      )
    : null;

  const versions =
    isDocument && s.versions.length
      ? h(
          "section",
          { class: "panel", "aria-labelledby": "versions-title" },
          h("h2", { id: "versions-title" }, plural(s.versions.length, "version")),
          h(
            "ol",
            { class: "versions" },
            s.versions.map((v) =>
              h(
                "li",
                {},
                v.id === s.id
                  ? h("span", { "aria-current": "page" }, "This version, ingested ", time(v.ingested_at, { withTime: true }))
                  : h("a", { href: sourceHash(bank, v.id) }, "Ingested ", time(v.ingested_at, { withTime: true })),
                v.gone ? [" · ", v.gone.reason.replaceAll("_", " ")] : null,
              ),
            ),
          ),
        )
      : null;

  return {
    title,
    content: [
      h("p", { class: "back" }, h("a", { href: isDocument ? bankHash(bank, "documents") : bankHash(bank, "memories") }, isDocument ? "← Documents" : "← Memories")),
      heading(
        h,
        title,
        h(
          "p",
          { class: "meta" },
          spaced([
            pill(isDocument ? "document" : "turn", "kind"),
            h("span", {}, "ingested ", time(s.ingested_at, { withTime: true })),
            s.platform ? h("span", {}, s.platform) : null,
            s.author_name ? h("span", {}, s.author_name) : null,
            s.reference_date ? h("span", {}, `dated ${s.reference_date}`) : null,
          ]),
        ),
      ),
      s.secret_kinds.length ? h("p", { class: "notice", "data-tone": "info" }, `Secrets were redacted before storage: ${s.secret_kinds.join(", ")}.`) : null,
      h(
        "div",
        { class: "source-grid" },
        h(
          "div",
          {},
          s.text !== null
            ? h("section", { class: "panel", "aria-label": "Text" }, h("pre", { class: "source-text" }, s.text), s.reply ? [h("h2", {}, "Reply"), h("pre", { class: "source-text" }, s.reply)] : null)
            : h("p", { class: "notice", "data-tone": "gone" }, goneReason(s.gone) ?? "The text is gone."),
          chunks,
        ),
        h("aside", {}, versions, remove),
      ),
    ],
  };
}

// The bank's ingestion: failed chunks to retry, and the queue.
async function chunks(ctx) {
  const { api, ui, bank } = ctx;
  const { h, time, pill } = ui;
  const list = await api.chunks(bank);
  const retry = (ids) =>
    ctx.act(async () => {
      const done = await api.retryChunks(bank, ids);
      const unknown = done.unknown.length ? ` ${plural(done.unknown.length, "chunk")} had already left the failed list.` : "";
      return `Put ${plural(done.retried.length, "chunk")} back on the queue.${unknown}`;
    });

  const failed = h(
    "section",
    { class: "panel", "aria-labelledby": "failed-title" },
    h(
      "div",
      { class: "section-head" },
      h("h2", { id: "failed-title" }, `Failed chunks (${list.failed.length})`),
      list.failed.length ? h("button", { type: "button", onclick: () => retry() }, "Retry all") : null,
    ),
    list.failed.length
      ? [
          h("p", { class: "quiet-text" }, "These chunks hit the retry cap. Retrying puts them back on the queue in their place."),
          h(
            "div",
            { class: "table-wrap" },
            h(
              "table",
              {},
              h("thead", {}, h("tr", {}, ["Error", "Attempts", "Failed", "Source", ""].map((c) => h("th", { scope: "col" }, c)))),
              h(
                "tbody",
                {},
                list.failed.map((f) =>
                  h(
                    "tr",
                    {},
                    h("td", {}, h("code", {}, f.error_kind), f.status ? h("span", { class: "quiet-text" }, ` HTTP ${f.status}`) : null),
                    h("td", { class: "num" }, String(f.error_count)),
                    h("td", {}, time(f.failed_at, { withTime: true })),
                    h("td", {}, h("a", { href: sourceHash(bank, f.source) }, `source ${shortId(f.source)}`)),
                    h("td", {}, h("button", { type: "button", class: "quiet", "aria-label": `Retry chunk ${shortId(f.chunk)}`, onclick: () => retry([f.chunk]) }, "Retry")),
                  ),
                ),
              ),
            ),
          ),
        ]
      : h("p", { class: "empty" }, "No failed chunks."),
  );

  const queued = h(
    "section",
    { class: "panel", "aria-labelledby": "queue-title" },
    h("h2", { id: "queue-title" }, `Queue (${list.queued.length})`),
    list.queued.length
      ? h(
          "div",
          { class: "table-wrap" },
          h(
            "table",
            {},
            h("thead", {}, h("tr", {}, ["Order", "Source", "Chunk", "Attempts", "State"].map((c) => h("th", { scope: "col" }, c)))),
            h(
              "tbody",
              {},
              list.queued.map((q, i) =>
                h(
                  "tr",
                  {},
                  h("td", { class: "num" }, String(i + 1)),
                  h("td", {}, h("a", { href: sourceHash(bank, q.source) }, `${q.source_kind} ${shortId(q.source)}`)),
                  h("td", { class: "num" }, String(q.position + 1)),
                  h("td", { class: "num" }, String(q.error_count)),
                  h("td", {}, q.in_flight ? pill("in flight", "in_flight") : pill("waiting", "queued")),
                ),
              ),
            ),
          ),
        )
      : h("p", { class: "empty" }, "Nothing waiting. New turns and documents are extracted as they arrive."),
  );

  return { title: `Ingestion · ${bank}`, content: [heading(h, "Ingestion"), failed, queued] };
}

async function missing(ctx) {
  const { h } = ctx.ui;
  return {
    title: "Not found",
    content: [heading(h, "There's no page here"), h("p", {}, h("a", { href: "#/" }, "Go to the banks"))],
  };
}

export const pages = { banks, memories, memory, documents, source, chunks, missing };
