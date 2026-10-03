// Building the page with DOM calls, and the words and labels shared across
// pages. Text always goes in as text, never as markup: memories and
// documents are the owner's content.

export function elements(document) {
  const window = document.defaultView;

  /// `h("a", { href, class }, ...children)`. `on*` attributes are listeners;
  /// `null`, `undefined` and `false` attributes and children are skipped.
  function h(tag, attributes = {}, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(attributes ?? {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
      else if (key === "style" && typeof value === "object") {
        for (const [property, v] of Object.entries(value)) el.style.setProperty(property, v);
      } else el.setAttribute(key, value === true ? "" : String(value));
    }
    append(el, children);
    return el;
  }

  function append(el, children) {
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      el.append(child instanceof window.Node ? child : String(child));
    }
    return el;
  }

  /// A date the reader sees, with the exact instant in `datetime`.
  function time(iso, { withTime = false } = {}) {
    if (!iso) return null;
    return h("time", { datetime: iso, title: iso }, withTime ? formatDateTime(iso) : formatDate(iso));
  }

  /// The stem gauge: how much life a memory has left before it fades.
  /// Decorative; the label beside it says the same in words.
  function gauge(stage, fade) {
    const fill =
      stage === "never" ? 1 : stage === "faded" || stage === "out" ? 0 : Math.min(1, Math.log1p(fade.bank_days) / Math.log1p(180));
    return h("span", { class: "gauge", "aria-hidden": "true", style: { "--fill": fill.toFixed(3) } }, h("span"));
  }

  /// When a memory fades if it isn't used again.
  function fadeLabel(fade, { gauged = true } = {}) {
    const stage = fadeStage(fade);
    const words =
      stage === "never"
        ? ["Never fades"]
        : stage === "faded"
          ? ["Faded: recall no longer finds it"]
          : [`Fades in ${bankDays(fade.bank_days)}, no sooner than `, time(fade.earliest_at)];
    return h("span", { class: "fade", "data-fade": stage }, gauged ? gauge(stage, fade) : null, h("span", {}, words));
  }

  /// What a retracted or forgotten memory's row says in place of a fade:
  /// it's out of use already, so only its purge, or its erase, is ahead.
  function outOfUseLabel(status, purge) {
    const words =
      status === "forgetting"
        ? ["Being forgotten: the erase removes it"]
        : !purge
          ? ["Out of use. Never purged while things stay as they are"]
          : purge.bank_days === 0
            ? ["Out of use. Can be purged at the next sweep"]
            : [`Out of use. Purged in ${bankDays(purge.bank_days)}, no sooner than `, time(purge.earliest_at)];
    return h("span", { class: "fade", "data-fade": "out" }, gauge("out"), h("span", {}, words));
  }

  /// When a memory's chain can be purged.
  function purgeLabel(purge) {
    if (!purge) return h("span", { class: "purge" }, "Never purged while things stay as they are");
    if (purge.bank_days === 0) return h("span", { class: "purge" }, "Can be purged at the next sweep");
    return h(
      "span",
      { class: "purge" },
      `Purged in ${bankDays(purge.bank_days)}, no sooner than `,
      time(purge.earliest_at),
    );
  }

  function pill(text, tone) {
    return h("span", { class: "pill", "data-tone": tone ?? text }, text);
  }

  return { h, append, time, gauge, fadeLabel, outOfUseLabel, purgeLabel, pill };
}

/// Where a memory sits on its way to fading: `never`, `far`, `near`,
/// `soon` or `faded`.
export function fadeStage(fade) {
  if (!fade) return "never";
  if (fade.bank_days <= 0) return "faded";
  if (fade.bank_days < 7) return "soon";
  if (fade.bank_days < 30) return "near";
  return "far";
}

export function plural(count, one, many = `${one}s`) {
  return `${count} ${count === 1 ? one : many}`;
}

export function bankDays(days) {
  if (days < 1) return "under a bank day";
  return plural(Math.round(days), "bank day");
}

const dateFormat = new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "numeric" });
const dateTimeFormat = new Intl.DateTimeFormat(undefined, {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

export function formatDate(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : dateFormat.format(date);
}

export function formatDateTime(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : dateTimeFormat.format(date);
}

/// The first and last few characters of a UUID, enough to tell them apart.
export function shortId(id) {
  return id.length > 13 ? `${id.slice(0, 8)}…${id.slice(-4)}` : id;
}

export function capitalize(text) {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}

/// Why a source's text is gone, in words.
export function goneReason(gone) {
  switch (gone?.reason) {
    case "removed":
      return "Removed by the owner, with every version of the document.";
    case "swept":
      return "Swept: the nightly sweep deleted the text at its horizon.";
    case "forget_requested":
      return "Never stored: the turn asked to be forgotten.";
    case "redacted":
      return "Masked by a forget.";
    default:
      return gone ? `Gone (${gone.reason}).` : null;
  }
}
