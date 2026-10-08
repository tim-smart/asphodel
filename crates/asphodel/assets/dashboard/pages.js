// The dashboard's pages. Each takes the render context and returns its
// title and content; actions go through `ctx.act`, destructive ones after
// `ctx.confirm`.

import { ApiError } from "./api.js";
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
      : section === "sources"
        ? id
          ? "source"
          : "sources"
        : section === "documents" && !id
          ? "sources"
          : section === "chunks" && !id
            ? "chunks"
            : section === "recall" && !id
              ? "recall"
              : section === "models" && !id
                ? "models"
                : "missing";
  return { page, bank, section, id, params };
}

const bankHash = (bank, ...rest) => `#/banks/${[bank, ...rest].map(encodeURIComponent).join("/")}`;

const memoryHash = (bank, id) => bankHash(bank, "memories", id);
const sourceHash = (bank, id) => bankHash(bank, "sources", id);

function listHash(bank, params, section = "memories") {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) if (value) search.set(key, value);
  const text = search.toString();
  return `${bankHash(bank, section)}${text ? `?${text}` : ""}`;
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
          "The deletion settings changed since purge last ran. Until you acknowledge them, the nightly sweep deletes nothing.",
        ),
        plan.changed.length
          ? h("p", {}, "Changed: ", plan.changed.map((key, i) => [i ? ", " : "", h("code", {}, key)]))
          : null,
        h("p", {}, "Under the new settings, the next sweep deletes:"),
        h(
          "ul",
          { class: "purge-counts" },
          h("li", {}, h("strong", {}, String(plan.memories)), " ", h("span", {}, plan.memories === 1 ? "memory" : "memories", h("span", { class: "quiet-text" }, ", counting every version"))),
          h("li", {}, h("strong", {}, String(plan.sources)), " ", plan.sources === 1 ? "source" : "sources"),
          h("li", {}, h("strong", {}, String(plan.failed_chunks)), " ", plan.failed_chunks === 1 ? "failed chunk" : "failed chunks"),
          h("li", {}, h("strong", {}, String(plan.recalls)), " ", plan.recalls === 1 ? "recall" : "recalls"),
        ),
        h(
          "button",
          {
            type: "button",
            class: "danger-outline",
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
                return "Purge resumed. The next sweep uses the new settings.";
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
            h("a", { href: bankHash(bank.name, "sources") }, "Documents"),
            String(bank.sources.documents),
            bank.sources.document_versions !== bank.sources.documents
              ? h("span", { class: "quiet-text" }, `(${plural(bank.sources.document_versions, "version")})`)
              : null,
          ],
          [h("a", { href: bankHash(bank.name, "sources") }, "Turns"), String(bank.sources.turns)],
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
        " · ",
        h("a", { href: bankHash(bank.name, "models") }, plural(bank.models, "mental model")),
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
      list.length || daemon
        ? h("div", { class: "bank-grid" }, list.length ? cards : h("p", { class: "empty" }, "No banks yet. One appears when the Hermes plugin first connects to this daemon."), daemon)
        : h("p", { class: "empty" }, "No banks yet. One appears when the Hermes plugin first connects to this daemon."),
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

/// Retracted and forgotten memories are out of use: what lies ahead of them
/// is a purge or an erase, not a fade.
const OUT_OF_USE = ["retracted", "forgetting"];

function memoryRow(ctx, memory) {
  const { h, fadeLabel, outOfUseLabel, pill, time } = ctx.ui;
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
    OUT_OF_USE.includes(memory.status) ? outOfUseLabel(memory.status, memory.purge) : fadeLabel(memory.fade),
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
      heading(h, "Memories", h("p", { class: "quiet-text" }, `${capitalize(memoryCount(page.total))} as of `, ui.time(page.as_of, { withTime: true }))),
      toolbar,
      chips,
      page.memories.length
        ? rows
        : h(
            "p",
            { class: "empty" },
            filtered ? ["No memories match these filters. ", h("a", { href: bankHash(bank, "memories") }, "Clear filters")] : "No memories yet. They appear once the first turn or document is extracted.",
          ),
      cursor ? more : null,
      h("p", { class: "footnote" }, FADE_BASIS),
    ],
  };
}

const GUARDS = {
  purge_disabled: "Purge is off because no δ is set.",
  purge_paused: "Purge is paused until someone acknowledges the new deletion settings.",
  forgotten: "It's being forgotten, so the erase removes it instead.",
  strength: "Its strength is above the purge line.",
  lasting: "It has come up often enough, or matters enough, that it's never purged.",
};

/// What the fade and purge dates mean, wherever they're shown.
const FADE_BASIS =
  "Dates assume the memory goes unused from now on. Bank time only moves while the bank is in use, so each date is the earliest it could happen.";

/// A strength figure, with a real minus sign.
function figure(value) {
  return value === null || value === undefined ? "—" : value.toFixed(2).replace("-", "−");
}

/// A date-only value such as "2026-09-30", formatted without shifting it
/// through the reader's time zone.
const dayFormat = new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
function day(value) {
  const date = /^\d{4}-\d{2}-\d{2}$/.test(value) ? new Date(`${value}T00:00:00Z`) : null;
  return date && !Number.isNaN(date.getTime()) ? dayFormat.format(date) : value;
}

function guardText(ui, guard) {
  if (guard.guard === "date_ahead") return ["Its start or end date is still ahead, on ", ui.time(guard.until), "."];
  if (guard.guard === "overdue_task") return ["It's an overdue task, held until ", ui.time(guard.until), "."];
  return GUARDS[guard.guard] ?? guard.guard;
}

// One memory: where it came from, how strong it is, when it fades, and the
// owner's actions on it.
async function memory(ctx) {
  const { api, ui, bank, at } = ctx;
  const { h, time, fadeLabel, outOfUseLabel, purgeLabel, pill, outlook } = ui;
  const view = await api.memory(bank, at.id);
  const status = memoryStatus(view);
  const kept = view.significance.owner === "kept";
  const versions = view.chain.members.length;
  const link = (id, text) => h("a", { href: memoryHash(bank, id) }, text ?? `memory ${shortId(id)}`);

  const source = view.source;
  const sourceName = source.document_id ?? (source.session_id ? `Turn in ${source.session_id}` : "Its source");

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
                    return "No longer kept. It can fade again like any other memory.";
                  }
                  await api.keep(bank, [view.id]);
                  return "Kept. It won't fade until you stop keeping it.";
                }),
            },
            kept ? "Stop keeping" : "Keep",
          ),
          h("p", {}, kept ? "Let it fade again if it goes unused." : "Hold it at full strength so it never fades."),
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
                      "Retracting marks it as never true. Recall, the agenda and the system prompt stop using it at once, and it fades out on its own.",
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
          h("p", {}, "For a memory that's wrong. Takes it out of use and reopens anything it ended."),
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
                      ? `This erases all ${plural(versions, "version")} of it and masks the passages they came from.`
                      : "This erases it and masks the passage it came from.",
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
                    ? [`Forgot ${memoryCount(done.forgotten.length)}, every version of `, h("q", {}, view.sentence), "."]
                    : "Nothing to forget. It was already gone.";
                },
                { then: bankHash(bank, "memories") },
              );
            },
          },
          "Forget…",
        ),
        h("p", {}, "Erases it and every version, as if it was never said."),
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
                    member.superseded_by ? " · replaced by the next" : " · current",
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
          h(
            "div",
            { class: "outlook-fade" },
            status === "forgetting"
              ? outOfUseLabel(status, null)
              : status === "retracted"
                ? outlook("out", "Out of use", "Recall, the agenda and the system prompt skip it")
                : fadeLabel(view.projection.fade),
          ),
          h("div", { class: "outlook-purge" }, purgeLabel(view.projection.purge)),
          !view.projection.purge && view.purge.guards.length
            ? h("ul", { class: "guards" }, view.purge.guards.map((g) => h("li", {}, guardText(ui, g))))
            : null,
          h("p", { class: "footnote" }, FADE_BASIS),
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
            source.secret_kinds.length ? h("p", { class: "quiet-text" }, `Redacted before storage: ${source.secret_kinds.join(", ")}.`) : null,
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
          view.restatements?.length
            ? h(
                "section",
                { class: "panel", "aria-labelledby": "restated-title" },
                h("h2", { id: "restated-title" }, "Said again"),
                h(
                  "ul",
                  { class: "history" },
                  view.restatements.map((r) => h("li", {}, r.sentence, " · ", h("span", {}, r.label.replaceAll("_", " ")), " ", time(r.observed_at, { withTime: true }))),
                ),
              )
            : null,
          h(
            "details",
            { class: "panel" },
            h("summary", {}, "History ", h("span", { class: "quiet-text" }, `${plural(view.accesses.length, "access", "accesses")}, ${plural(view.edits.length, "edit")}`)),
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
              h("div", {}, h("dt", {}, "Now"), h("dd", {}, figure(strength.value))),
              h("div", {}, h("dt", {}, "Recall threshold"), h("dd", {}, figure(strength.threshold))),
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

function groupVersions(sources) {
  const groups = new Map();
  for (const source of sources) {
    const key = source.document_id ?? source.id;
    const group = groups.get(key) ?? { id: key, newest: source, versions: [] };
    group.versions.push(source);
    groups.set(key, group);
  }
  return [...groups.values()];
}

/// Asks, then removes every version of a document and forgets the memories
/// resting on it.
async function removeDocument(ctx, documentId) {
  const { h } = ctx.ui;
  const confirmed = await ctx.confirm({
    title: "Remove this document?",
    body: [
      h("p", { class: "quoted" }, documentId),
      h(
        "p",
        {},
        "This removes every version and forgets the memories drawn from it, with all their versions. The text is deleted now. Sending the same text again is ignored as a duplicate.",
      ),
      h("p", {}, "There's no undo here. A backup or a later edited copy of the document can bring it back."),
    ],
    action: "Remove document",
  });
  if (!confirmed) return;
  ctx.act(async () => {
    const done = await ctx.api.removeDocument(ctx.bank, documentId);
    const dequeued = done.dequeued ? ` and took ${plural(done.dequeued, "chunk")} off the queue` : "";
    return `Removed ${done.document_id}: ${plural(done.sources.length, "version")}. Forgot ${memoryCount(done.forgotten.length)}${dequeued}.`;
  });
}

/// Asks, then removes a turn and forgets the memories resting on it.
async function removeTurn(ctx, turn) {
  const { h } = ctx.ui;
  const confirmed = await ctx.confirm({
    title: "Remove this turn?",
    body: [
      h("p", {}, "This forgets the memories drawn from it, with all their versions. The text is deleted now. Sending the same turn again is ignored as a duplicate."),
      h("p", {}, "There's no undo here. Only a backup can bring it back."),
    ],
    action: "Remove turn",
  });
  if (!confirmed) return;
  ctx.act(async () => {
    const done = await ctx.api.removeTurn(ctx.bank, turn);
    const dequeued = done.dequeued ? ` and took ${plural(done.dequeued, "chunk")} off the queue` : "";
    return `Removed the turn. Forgot ${memoryCount(done.forgotten.length)}${dequeued}.`;
  });
}

// The bank's turns and documents, newest first: one entry per turn, and one
// per document id with its newest version.
async function sources(ctx) {
  const { api, ui, bank, params } = ctx;
  const { h, time, pill } = ui;
  const filters = { q: params.q ?? "", kind: params.kind ?? "", gone: params.gone ?? "" };
  const go = (change) => ctx.navigate(listHash(bank, { ...filters, ...change }, "sources"));
  const first = await api.sources(bank, { ...filters, limit: 200 });
  const loaded = [...first.sources];
  let cursor = first.next_cursor;
  const list = h("ul", { class: "sources" });

  const draw = () =>
    list.replaceChildren(
      ...groupVersions(loaded).map(({ id, newest, versions }) => {
        const isTurn = newest.kind === "turn";
        const chunks = versions.reduce((sum, v) => sum + v.chunks, 0);
        const failed = versions.reduce((sum, v) => sum + v.failed, 0);
        const remembered = versions.reduce((sum, v) => sum + v.memories, 0);
        return h(
          "li",
          { class: "source", "data-gone": newest.gone?.reason ?? null },
          h(
            "a",
            { class: "source-name", href: sourceHash(bank, newest.id) },
            isTurn ? `Turn${newest.session_id ? ` in ${newest.session_id}` : ""}` : id,
          ),
          " ",
          h(
            "div",
            { class: "meta" },
            spaced([
              newest.gone ? pill(newest.gone.reason.replaceAll("_", " "), "gone") : null,
              isTurn ? null : h("span", {}, plural(versions.length, "version")),
              h("span", {}, plural(chunks, "chunk")),
              h("span", {}, memoryCount(remembered)),
              failed ? pill(`${failed} failed`, "failed") : null,
              isTurn
                ? h("span", {}, "said ", time(newest.message_at ?? newest.observed_at, { withTime: true }))
                : h("span", {}, "ingested ", time(newest.ingested_at)),
            ]),
          ),
          newest.gone?.reason === "removed"
            ? null
            : [
                " ",
                h(
                  "button",
                  {
                    type: "button",
                    class: "danger-outline source-remove",
                    "aria-label": `Remove ${isTurn ? `turn ${newest.id}` : id}`,
                    onclick: () => (isTurn ? removeTurn(ctx, newest.id) : removeDocument(ctx, id)),
                  },
                  "Remove…",
                ),
              ],
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
        const next = await api.sources(bank, { ...filters, limit: 200, cursor });
        loaded.push(...next.sources);
        cursor = next.next_cursor;
        draw();
        more.disabled = false;
        if (!cursor) more.remove();
      },
    },
    "Load more sources",
  );

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
      h("label", { for: "search" }, "Search sources"),
      h("div", { class: "search-row" }, search, h("button", { type: "submit" }, "Search")),
    ),
    select(h, {
      id: "kind",
      label: "Kind",
      value: filters.kind,
      options: [
        ["", "Any kind"],
        ["turn", "Turns"],
        ["document", "Documents"],
      ],
      onchange: (kind) => go({ kind }),
    }),
    select(h, {
      id: "gone",
      label: "Text",
      value: filters.gone,
      options: [
        ["", "Stored or gone"],
        ["false", "Still stored"],
        ["true", "Gone"],
      ],
      onchange: (gone) => go({ gone }),
    }),
  );
  const filtered = Object.values(filters).some(Boolean);

  return {
    title: `Sources · ${bank}`,
    content: [
      heading(h, "Sources"),
      toolbar,
      loaded.length
        ? list
        : h(
            "p",
            { class: "empty" },
            filtered
              ? ["No sources match these filters. ", h("a", { href: bankHash(bank, "sources") }, "Clear filters")]
              : "Nothing ingested yet. Turns and documents will appear here as Hermes sends them.",
          ),
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
  const title = isDocument ? s.document_id : `Turn${s.session_id ? ` in ${s.session_id}` : ""}`;

  const remove = removed
    ? null
    : h(
        "section",
        { class: "panel danger-zone", "aria-labelledby": "remove-title" },
        h("h2", { id: "remove-title" }, isDocument ? "Remove this document" : "Remove this turn"),
        h(
          "p",
          {},
          isDocument
            ? "Removes every version and forgets the memories drawn from it. Memories that only mention it are left alone."
            : "Forgets the memories drawn from it. Memories that only mention it are left alone.",
        ),
        h(
          "button",
          { type: "button", class: "danger-outline", onclick: () => (isDocument ? removeDocument(ctx, s.document_id) : removeTurn(ctx, s.id)) },
          isDocument ? "Remove document…" : "Remove turn…",
        ),
      );

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
          ? h("p", {}, h("a", { href: bankHash(bank, "chunks") }, "Retry failed chunks on the Ingestion page"))
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
      h("p", { class: "back" }, h("a", { href: bankHash(bank, "sources") }, "← Sources")),
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
            s.reference_date ? h("span", {}, `dated ${day(s.reference_date)}`) : null,
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
            ? h(
                "section",
                { class: "panel", "aria-label": "Text" },
                s.reply ? h("h2", {}, "Message") : null,
                h("pre", { class: "source-text" }, s.text),
                s.reply ? [h("h2", { class: "reply-title" }, "Reply"), h("pre", { class: "source-text" }, s.reply)] : null,
              )
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
      h("h2", { id: "failed-title" }, "Failed chunks ", h("span", { class: "title-count" }, String(list.failed.length))),
      list.failed.length ? h("button", { type: "button", class: "small", onclick: () => retry() }, "Retry all") : null,
    ),
    list.failed.length
      ? [
          h("p", { class: "quiet-text" }, "Extraction gave up on these after too many attempts. Retrying puts them back in the queue where they were."),
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
      : h("p", { class: "empty" }, "No failed chunks. Everything extracted cleanly."),
  );

  const queued = h(
    "section",
    { class: "panel", "aria-labelledby": "queue-title" },
    h("h2", { id: "queue-title" }, "Queue ", h("span", { class: "title-count" }, String(list.queued.length))),
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

// The Recall page: one query through recall or injection, with the working
// shown. It calls explain, never `/recall` or `/prefetch`, so a test query
// writes no recall row, credits no use and touches no session.

const RECALL_LIMIT_DEFAULT = 10;
const RECALL_LIMIT_MAX = 30;
const PHASES = [
  ["any", "Any phase"],
  ["upcoming", "Upcoming"],
  ["current", "Current"],
  ["past", "Past"],
];
const ON = [
  ["happened", "When it happened"],
  ["said", "When it was said"],
];

/// Why explain left a candidate out, in words.
function cutReason(reason, { injection, limit }) {
  switch (reason) {
    case "below_tau":
      return "Too weak (below τ) to reach fusion.";
    case "not_reranked":
      return "Not reranked. The reranker missed its deadline, so nothing passes the gate.";
    case "under_floor":
      return `Logit below the reranker's floor of ${figure(injection?.floor)}.`;
    case "over_cap":
      return `Over the cap of ${memoryCount(injection?.cap ?? 0)}. Higher scores took every place.`;
    case "over_budget":
      return `Over budget. Its line would take the block past ${injection?.token_budget ?? "the"} tokens.`;
    case "over_limit":
      return `Ranked past the limit of ${limit}.`;
    case "outside_rerank_pool":
      return "Fused too far down for the reranker to see.";
    default:
      return reason;
  }
}

/// How far `timeZone`'s wall clock runs ahead of UTC at the instant `ms`,
/// in milliseconds.
function zoneOffset(ms, timeZone) {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone,
      hourCycle: "h23",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    })
      .formatToParts(new Date(ms))
      .map((part) => [part.type, part.value]),
  );
  const wall = Date.UTC(Number(parts.year), Number(parts.month) - 1, Number(parts.day), Number(parts.hour), Number(parts.minute), Number(parts.second));
  return wall - Math.floor(ms / 1000) * 1000;
}

/// The calendar day the instant `ms` falls on in `timeZone`, as
/// "2026-10-03".
function zoneDate(ms, timeZone) {
  return new Intl.DateTimeFormat("en-CA", { timeZone, year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date(ms));
}

/// The first instant of the calendar day `date` in `timeZone`, as epoch
/// milliseconds: midnight, or the end of a clock change that skips it.
function startOfDay(date, timeZone) {
  const [year, month, dayOfMonth] = date.split("-").map(Number);
  const wall = Date.UTC(year, month - 1, dayOfMonth);
  const hours = 60 * 60 * 1000;
  const candidates = [-36, 0, 36].map((shift) => wall - zoneOffset(wall + shift * hours, timeZone));
  const onTheDay = candidates.filter((ms) => zoneDate(ms, timeZone) === date).sort((a, b) => a - b);
  return onTheDay[0] ?? wall - zoneOffset(wall, timeZone);
}

/// The day after `date`, as a date.
function nextDay(date) {
  const [year, month, dayOfMonth] = date.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, dayOfMonth + 1)).toISOString().slice(0, 10);
}

/// A From day as the instant it starts, in the bank's timezone.
function rangeStart(date, timeZone) {
  return date ? new Date(startOfDay(date, timeZone)).toISOString().replace(".000Z", "Z") : null;
}

/// A To day as the last instant before the next day starts, so the range
/// takes the whole day.
function rangeEnd(date, timeZone) {
  if (!date) return null;
  const next = startOfDay(nextDay(date), timeZone);
  return new Date(next - 1000).toISOString().replace(".000Z", ".999999999Z");
}

function milliseconds(ms) {
  return `${ms} ms`;
}

async function recall(ctx) {
  const { api, ui, bank } = ctx;
  const { h } = ui;
  const { banks: list } = await api.banks();
  const timeZone = list.find((b) => b.name === bank)?.timezone ?? "UTC";

  let mode = "recall";
  let generation = 0;

  const field = (id, label, control, hint) =>
    h("div", { class: "field" }, h("label", { for: id }, label), control, hint ? h("p", { class: "hint" }, hint) : null);

  // The one box both modes share: the recall tool's query, or the user's
  // message injection would answer.
  const queryLabel = h("label", { for: "explain-query" }, "Query");
  const query = h("textarea", {
    id: "explain-query",
    rows: 2,
    required: true,
    autocomplete: "off",
    onkeydown: (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        form.requestSubmit();
      }
    },
  });

  const modes = h(
    "fieldset",
    { class: "segmented" },
    h("legend", {}, "Run as"),
    [
      ["recall", "Recall", "What memory_recall would return"],
      ["injection", "Injection", "What prefetch would put in the prompt"],
    ].map(([value, label, detail]) =>
      h(
        "label",
        { class: "segment" },
        h("input", { type: "radio", name: "explain-mode", value, checked: value === mode, onchange: () => setMode(value) }),
        h("span", {}, h("strong", {}, label), h("span", { class: "segment-detail" }, detail)),
      ),
    ),
  );

  const on = h("select", { id: "explain-on" }, ON.map(([value, text]) => h("option", { value }, text)));
  const from = h("input", { id: "explain-from", type: "date", onchange: () => (to.min = from.value) });
  const to = h("input", { id: "explain-to", type: "date", onchange: () => (from.max = to.value) });
  const phase = h("select", { id: "explain-phase" }, PHASES.map(([value, text]) => h("option", { value }, text)));
  const kinds = KINDS.map((kind) => h("input", { type: "checkbox", value: kind }));
  const entity = h("input", { id: "explain-entity", type: "text", autocomplete: "off", placeholder: "A name or alias" });
  const limit = h("input", {
    id: "explain-limit",
    type: "number",
    min: 1,
    max: RECALL_LIMIT_MAX,
    inputmode: "numeric",
    placeholder: String(RECALL_LIMIT_DEFAULT),
  });

  // Each mode's fields sit in a fieldset that's disabled, not just hidden,
  // while the other mode is on: a disabled control keeps its value but isn't
  // validated, so a recall filter left invalid can't block an injection.
  const recallFilters = h(
    "fieldset",
    { class: "explain-filters", "aria-label": "Recall filters" },
    field("explain-from", "From", from),
    field("explain-to", "To", to),
    field("explain-on", "Dates match", on),
    h("p", { class: "hint range-hint" }, `Whole days in the bank's timezone (${timeZone}). Either end can be left open.`),
    field("explain-phase", "Phase", phase),
    field("explain-entity", "Entity", entity),
    field("explain-limit", "Limit", limit),
    h(
      "fieldset",
      { class: "kinds" },
      h("legend", {}, "Kinds"),
      kinds.map((box) => h("label", { class: "check" }, box, capitalize(box.value))),
      h("p", { class: "hint" }, "Leave all unticked to search every kind."),
    ),
  );

  const previousQuery = h("textarea", { id: "explain-previous-query", rows: 3, autocomplete: "off" });
  const previousReply = h("textarea", { id: "explain-previous-reply", rows: 3, autocomplete: "off" });
  const injectionFields = h(
    "fieldset",
    { class: "explain-filters", "aria-label": "Conversation", hidden: true },
    field("explain-previous-query", "Previous message", previousQuery, "The user's message before this one. Optional."),
    field("explain-previous-reply", "Previous reply", previousReply, "The agent's answer to it. A short follow-up borrows context from both."),
    h(
      "p",
      { class: "notice", "data-tone": "info" },
      "This runs outside any session. Nothing counts as already in context, and nothing is held back for a later turn.",
    ),
  );

  const run = h("button", { type: "submit" }, "Explain");
  const results = h("div", { class: "explain-results", "aria-live": "polite" });

  function setMode(next) {
    mode = next;
    queryLabel.textContent = mode === "recall" ? "Query" : "Message";
    query.placeholder = mode === "recall" ? "What the agent would search for" : "What the user just said";
    recallFilters.hidden = recallFilters.disabled = mode !== "recall";
    injectionFields.hidden = injectionFields.disabled = mode !== "injection";
  }
  setMode(mode);

  function request() {
    const text = query.value.trim();
    if (mode === "injection") {
      return {
        mode,
        query: text,
        previous_query: previousQuery.value.trim() || null,
        previous_reply: previousReply.value.trim() || null,
      };
    }
    return {
      mode,
      query: text,
      on: on.value,
      from: rangeStart(from.value, timeZone),
      to: rangeEnd(to.value, timeZone),
      phase: phase.value,
      kinds: kinds.filter((box) => box.checked).map((box) => box.value),
      entity: entity.value.trim() || null,
      limit: limit.value ? Number(limit.value) : null,
    };
  }

  const form = h(
    "form",
    {
      class: "panel explain-form",
      "aria-label": "Explain a query",
      onsubmit: async (event) => {
        event.preventDefault();
        const sent = request();
        if (!sent.query) return;
        const mine = ++generation;
        run.disabled = true;
        results.setAttribute("aria-busy", "true");
        try {
          const explained = await api.explain(bank, sent);
          if (mine !== generation) return;
          results.replaceChildren(...explainResults(ctx, explained, sent, timeZone));
        } catch (error) {
          if (mine !== generation) return;
          results.replaceChildren(h("p", { class: "notice", "data-tone": "error", role: "alert" }, error.message));
        } finally {
          if (mine === generation) {
            run.disabled = false;
            results.setAttribute("aria-busy", "false");
          }
        }
      },
    },
    modes,
    h("div", { class: "field explain-query" }, queryLabel, query),
    recallFilters,
    injectionFields,
    h("div", { class: "explain-run" }, run, h("p", { class: "hint" }, "A dry run. Nothing is logged, no memory is credited with a use, and no session is touched.")),
  );

  return {
    title: `Recall · ${bank}`,
    content: [
      heading(h, "Recall", h("p", { class: "quiet-text" }, "Try a query and see what comes back, and why the rest didn't.")),
      form,
      results,
    ],
  };
}

/// What explain answered: the summary, the injected text, then what made
/// the cut and, greyed under it, what didn't.
function explainResults(ctx, explained, sent, timeZone) {
  const { h } = ctx.ui;
  const injection = explained.injection;
  const isInjection = explained.mode === "injection";
  const limit = sent.limit ?? RECALL_LIMIT_DEFAULT;
  const kept = explained.candidates.filter((c) => c.included);
  const cut = explained.candidates.filter((c) => !c.included);
  const latency = explained.latency;

  const summary = h(
    "section",
    { class: "panel explain-summary", "aria-label": "How it ran" },
    explained.reranked
      ? null
      : h(
          "p",
          { class: "notice", "data-tone": "error" },
          isInjection
            ? "The reranker missed its deadline. Nothing passes the gate, so nothing would be injected."
            : "The reranker missed its deadline. These are in fusion order, which is how recall returns them.",
        ),
    h(
      "dl",
      { class: "facts" },
      h("div", {}, h("dt", {}, "Searched for"), h("dd", {}, explained.query)),
      explained.rerank_query !== explained.query ? h("div", {}, h("dt", {}, "Reranked against"), h("dd", {}, explained.rerank_query)) : null,
      sent.from || sent.to
        ? h(
            "div",
            {},
            h("dt", {}, sent.on === "said" ? "Said" : "Happened"),
            h(
              "dd",
              {},
              sent.from ? ["from ", h("code", {}, sent.from)] : null,
              sent.from && sent.to ? " " : null,
              sent.to ? ["up to ", h("code", {}, sent.to)] : null,
              h("span", { class: "quiet-text" }, ` (whole days, ${timeZone})`),
            ),
          )
        : null,
    ),
    h(
      "dl",
      { class: "latency", "aria-label": "Latency" },
      [
        ["Embed", latency.embed_ms],
        ["Retrieve", latency.retrieve_ms],
        ["Rerank", latency.rerank_ms],
        ["Total", latency.total_ms],
      ].map(([stage, ms]) => h("div", {}, h("dt", {}, stage), " ", h("dd", { class: "num" }, milliseconds(ms)))),
    ),
  );

  const injected = isInjection
    ? h(
        "section",
        { class: "panel", "aria-labelledby": "injected-title" },
        h("h2", { id: "injected-title" }, "Injected text"),
        injection?.text
          ? h("pre", { class: "injection-text" }, injection.text)
          : h("p", { class: "empty" }, "Nothing passed the gate, so nothing would be injected."),
        injection
          ? h(
              "p",
              { class: "meta" },
              spaced([
                h("span", {}, `${injection.tokens} tokens of ${injection.token_budget}`),
                h("span", {}, `${injection.injected.length} of a cap of ${injection.cap}`),
                h("span", {}, `floor ${figure(injection.floor)}`),
              ]),
            )
          : null,
        h("p", { class: "hint" }, "Shown exactly as the agent receives it."),
      )
    : null;

  const reasons = { injection, limit };
  const table = (rows, { left }) =>
    h(
      "div",
      { class: "table-wrap" },
      h(
        "table",
        { class: left ? "explain-table left-out" : "explain-table" },
        h(
          "thead",
          {},
          h(
            "tr",
            {},
            ["Memory", "Found by", "Fused", "Logit", "Score", "Strength", left ? "Why not" : null]
              .filter(Boolean)
              .map((c) => h("th", { scope: "col" }, c)),
          ),
        ),
        h("tbody", {}, rows.map((c) => candidateRow(ctx, c, left ? cutReason(c.reason, reasons) : null))),
      ),
    );

  // With nothing injected, the injected text already says so.
  const madeIt =
    isInjection && !kept.length
      ? null
      : h(
          "section",
          { class: "panel", "aria-labelledby": "made-title" },
          h("h2", { id: "made-title" }, isInjection ? "Injected " : "Returned ", h("span", { class: "title-count" }, String(kept.length))),
          kept.length ? table(kept, { left: false }) : h("p", { class: "empty" }, "Recall returns nothing for this query."),
        );

  const leftOut = h(
    "section",
    { class: "panel", "aria-labelledby": "left-title" },
    h("h2", { id: "left-title" }, "Left out ", h("span", { class: "title-count" }, String(cut.length))),
    cut.length ? table(cut, { left: true }) : null,
    h(
      "p",
      { class: "hint" },
      isInjection
        ? "A memory missing from both lists wasn't in any search arm's top hits."
        : "A memory missing from both lists wasn't in any search arm's top hits, or a filter dropped it.",
    ),
  );

  return [summary, injected, madeIt, leftOut].filter(Boolean);
}

/// A model's stored answer, each `### heading` and its paragraph or list,
/// or that it has none yet.
function modelAnswer(h, answer) {
  if (!answer) return h("p", { class: "answer quiet-text" }, "No answer yet.");
  const sections = answer.split("\n\n").map((block) => {
    const [first, ...rest] = block.split("\n");
    return first.startsWith("### ")
      ? [h("h3", {}, first.slice(4)), answerLines(h, rest)]
      : answerLines(h, [first, ...rest]);
  });
  return h("div", { class: "answer" }, sections);
}

/// A section's lines: a run of `- ` lines as a list, any other line a
/// paragraph.
function answerLines(h, lines) {
  const blocks = [];
  for (const line of lines) {
    const last = blocks.at(-1);
    if (!line.startsWith("- ")) blocks.push(line);
    else if (Array.isArray(last)) last.push(line.slice(2));
    else blocks.push([line.slice(2)]);
  }
  return blocks.map((block) =>
    Array.isArray(block) ? h("ul", {}, block.map((item) => h("li", {}, item))) : h("p", {}, block),
  );
}

function candidateRow(ctx, c, reason) {
  const { h, pill } = ctx.ui;
  const score = c.score;
  return h(
    "tr",
    { "data-cut": reason ? c.reason : null },
    h(
      "td",
      { class: "explain-memory", "data-label": "Memory" },
      h("a", { class: "sentence", href: memoryHash(ctx.bank, c.id) }, c.kept ? h("span", { class: "seal", title: "Kept" }, "✻ ") : null, c.sentence),
      h("div", { class: "meta" }, spaced([h("span", {}, c.kind), h("span", {}, c.phase.replaceAll("_", " "))])),
    ),
    h(
      "td",
      { "data-label": "Found by" },
      h(
        "ul",
        { class: "inline-list arms" },
        c.arms.map((a) => h("li", {}, a.arm === "bm25" ? "BM25" : a.arm, a.rank === null ? null : [" ", h("span", { class: "num" }, `#${a.rank}`)])),
      ),
    ),
    h("td", { class: "num", "data-label": "Fused" }, c.rrf_rank === null ? "—" : String(c.rrf_rank)),
    h("td", { class: "num", "data-label": "Logit" }, figure(c.logit)),
    h(
      "td",
      { "data-label": "Score" },
      score
        ? [
            h("span", { class: "num score-total" }, figure(score.total)),
            h(
              "span",
              { class: "score-parts", title: "relevance + w_s × strength + state confidence + phase" },
              `${figure(score.relevance)} relevance + ${figure(score.strength_term)} strength (w_s ${figure(score.w_s)}) + ${figure(score.confidence_term)} confidence + ${figure(score.phase_term)} phase`,
            ),
          ]
        : "—",
    ),
    h("td", { "data-label": "Strength" }, pill(c.strength, `band-${c.strength}`), c.kept ? [" ", pill("kept", "kept")] : null),
    reason ? h("td", { class: "why", "data-label": "Why not" }, reason) : null,
  );
}

// The bank's mental models, and which of them go into Hermes's system
// prompt. A model is in the prompt while it's enabled; the daemon refuses an
// enable past the budget, and the page shows its reason. The prompt preview
// is what a new session would get now, and looking writes nothing.

const REFRESH_FAILURES = {
  llm: "The LLM call failed",
  malformed: "The reply was malformed",
  retrieval: "Retrieval failed",
};

async function models(ctx) {
  const { api, ui, bank } = ctx;
  const { h, time, pill } = ui;
  const [{ models: list, budget }, preview] = await Promise.all([
    api.models(bank),
    api.previewSystemPrompt(bank),
  ]);
  const enabled = list.filter((m) => m.enabled);
  const used = enabled.reduce((sum, m) => sum + m.max_tokens, 0);
  const left = budget - used;

  const setEnabled = (model, on) =>
    ctx.act(async () => {
      try {
        await api.editModel(bank, model.name, { enabled: on });
      } catch (error) {
        if (on && error instanceof ApiError && error.status === 422) {
          error.message = `That model stays out of the prompt because ${error.message}. Take another model out first.`;
        }
        throw error;
      }
      return on
        ? "Back in the prompt for new Hermes sessions. A refresh has started."
        : "Out of the prompt for new Hermes sessions. Its refreshes are paused."
    });

  const usage = h(
    "section",
    { class: "panel budget", "aria-labelledby": "budget-title" },
    h("h2", { id: "budget-title" }, "System prompt budget"),
    h(
      "p",
      { class: "budget-figure" },
      h("strong", {}, String(used)),
      ` of ${budget} tokens`,
    ),
    h(
      "div",
      { class: "budget-bar", "aria-hidden": "true" },
      enabled.map((m) =>
        h("span", { title: `${m.name}: ${m.max_tokens} tokens`, style: { "--share": `${Math.min(100, (m.max_tokens / budget) * 100).toFixed(2)}%` } }),
      ),
    ),
    h(
      "p",
      { class: "quiet-text" },
      left >= 0
        ? `${plural(enabled.length, "enabled model")}, ${left} tokens to spare. `
        : `${plural(enabled.length, "enabled model")}, ${-left} tokens over. `,
      "The agenda is paid for first, so a model can get less than its limit.",
    ),
    h(
      "ul",
      { class: "model-notes" },
      h("li", {}, "Changes reach new Hermes sessions only. A running session keeps the system prompt it started with."),
      h("li", {}, "Taking a model out pauses its refreshes. Putting it back starts one."),
    ),
  );

  const prompt = h(
    "section",
    { class: "panel prompt-preview", "aria-labelledby": "preview-title" },
    h(
      "div",
      { class: "section-head" },
      h("h2", { id: "preview-title" }, "System prompt"),
      h("p", { class: "quiet-text" }, "Built ", time(preview.built_at, { withTime: true })),
    ),
    h("p", { class: "quiet-text" }, "What a new Hermes session would get right now, character for character."),
    h("pre", { class: "injection-text", tabindex: "0", "aria-label": "The system prompt" }, preview.text),
  );

  const cards = list.map((m, i) => {
    const nameId = `model-${i}-name`;
    const stateId = `model-${i}-state`;
    const toggle = h("input", {
      type: "checkbox",
      role: "switch",
      class: "switch-input",
      checked: m.enabled,
      "aria-labelledby": `${nameId} ${stateId}`,
      onchange: () => {
        toggle.disabled = true;
        setEnabled(m, toggle.checked);
      },
    });
    const kinds = m.kinds.length ? m.kinds.join(", ") : "every kind";
    return h(
      "article",
      { class: "card model", "data-enabled": m.enabled ? "true" : "false" },
      h(
        "header",
        { class: "model-head" },
        h("h2", { id: nameId }, m.name),
        h(
          "label",
          { class: "switch" },
          toggle,
          h("span", { class: "switch-track", "aria-hidden": "true" }),
          h("span", { id: stateId }, "In the prompt"),
        ),
      ),
      h("p", { class: "question" }, m.question),
      modelAnswer(h, m.answer),
      m.last_error
        ? h(
            "p",
            { class: "model-error" },
            pill("refresh failed", "failed"),
            " ",
            h("span", {}, REFRESH_FAILURES[m.last_error] ?? m.last_error, m.last_error_at ? [", ", time(m.last_error_at, { withTime: true })] : null),
          )
        : null,
      h(
        "ul",
        { class: "meta inline-list" },
        h("li", {}, h("span", { class: "num" }, String(m.max_tokens)), " tokens"),
        h("li", {}, plural(m.cites.length, "memory cited", "memories cited")),
        h("li", {}, kinds),
        h("li", {}, m.last_refreshed_at ? ["Refreshed ", time(m.last_refreshed_at, { withTime: true })] : "Never refreshed"),
        m.enabled ? null : h("li", {}, "Refreshes paused"),
      ),
    );
  });

  return {
    title: `Mental models · ${bank}`,
    content: [
      heading(h, "Mental models", h("p", { class: "quiet-text" }, "Answers to standing questions, written from this bank's memories. Each enabled model goes into Hermes's system prompt after the agenda.")),
      usage,
      list.length
        ? h("div", { class: "model-grid" }, cards)
        : h("p", { class: "empty" }, "No mental models yet. Create one with ", h("code", {}, "asphodel model create"), "."),
      prompt,
    ],
  };
}

async function missing(ctx) {
  const { h } = ctx.ui;
  return {
    title: "Not found",
    content: [heading(h, "Nothing here"), h("p", {}, "That address doesn't match any page. ", h("a", { href: "#/" }, "Go to the banks"))],
  };
}

export const pages = { banks, memories, memory, sources, source, chunks, recall, models, missing };
