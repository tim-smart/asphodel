// A stand-in for the daemon's HTTP API, the dashboard's only boundary. It
// answers the routes the dashboard calls with the shapes the daemon sends
// (see `docs/operations.md` and `crates/asphodel/src/serve/api.rs`), keeps
// enough state for an action to show on the next read, and records every
// request.

export const TOKEN = "dashboard-token";

export const NOW = "2026-10-03T09:00:00Z";

export const DOCUMENT = "notes/2026/today.md";

export const ids = {
  auckland: "01a10400-0000-7000-8000-000000000001",
  wellington: "01a10400-0000-7000-8000-000000000002",
  ada: "01a10400-0000-7000-8000-000000000003",
  dentist: "01a10400-0000-7000-8000-000000000004",
  job: "01a10400-0000-7000-8000-000000000005",
  oldJob: "01a10400-0000-7000-8000-000000000006",
  berlin: "01a10400-0000-7000-8000-000000000007",
  documentV1: "01a10400-0000-7000-8000-0000000000a1",
  documentV2: "01a10400-0000-7000-8000-0000000000a2",
  recipes: "01a10400-0000-7000-8000-0000000000a3",
  turn: "01a10400-0000-7000-8000-0000000000a4",
  chunkHome: "01a10400-0000-7000-8000-0000000000c1",
  chunkWork: "01a10400-0000-7000-8000-0000000000c2",
  chunkTurn: "01a10400-0000-7000-8000-0000000000c3",
  profile: "01a10400-0000-7000-8000-0000000000d1",
  plans: "01a10400-0000-7000-8000-0000000000d2",
  travel: "01a10400-0000-7000-8000-0000000000d3",
  profileEntry: "01a10400-0000-7000-8000-0000000000e1",
};

export const sentences = {
  auckland: "Sam lives in Auckland.",
  wellington: "Sam lives in Wellington.",
  ada: "Sam's daughter is called Ada.",
  dentist: "Sam has a dentist appointment on Tuesday.",
  job: "Sam works at Effectful.",
  oldJob: "Sam works at Acme.",
  berlin: "Sam visited Berlin in May.",
};

/// The questions the fixture models answer.
export const questions = {
  profile: "Who is Sam, and what matters to them?",
  plans: "What is Sam planning, and when?",
  travel: "Where has Sam travelled?",
};

function model(key, name, fields) {
  return {
    id: ids[key],
    name,
    question: questions[key],
    kinds: [],
    entity: null,
    min_volatility: null,
    max_tokens: 100,
    enabled: true,
    entries: [],
    last_refreshed_at: null,
    last_error: null,
    last_error_at: null,
    ...fields,
  };
}

function memory(key, fields) {
  return {
    id: ids[key],
    sentence: sentences[key],
    kind: "fact",
    phase: null,
    status: "live",
    significance: { extracted: "notable", owner: null, effective: "notable", value: 0.5 },
    observed_at: "2026-10-01T08:00:00Z",
    created_at: "2026-10-01T08:00:05Z",
    source: ids.documentV2,
    source_kind: "document",
    document_id: DOCUMENT,
    session_id: null,
    superseded_by: null,
    ended_by: null,
    strength: 1.2,
    recallable: true,
    fade: { bank_days: 30, earliest_at: "2026-11-02T09:00:00Z" },
    purge: { bank_days: 60, earliest_at: "2026-12-02T09:00:00Z" },
    ...fields,
  };
}

/// The bank `main` as the tests start from it.
export function fixtures() {
  const memories = [
    memory("auckland", {
      fade: { bank_days: 14, earliest_at: "2026-10-17T09:00:00Z" },
      purge: { bank_days: 40, earliest_at: "2026-11-12T09:00:00Z" },
    }),
    memory("wellington", { status: "superseded", superseded_by: ids.auckland, source: ids.documentV1 }),
    memory("ada", {
      significance: { extracted: "major", owner: "kept", effective: "kept", value: 1 },
      fade: null,
      purge: null,
      source_kind: "turn",
      source: ids.turn,
      document_id: null,
      session_id: "session-1",
    }),
    memory("dentist", {
      kind: "event",
      strength: -0.4,
      recallable: false,
      fade: { bank_days: 0, earliest_at: NOW },
      purge: { bank_days: 3, earliest_at: "2026-10-06T09:00:00Z" },
    }),
    memory("job", {}),
    memory("oldJob", { status: "ended", ended_by: ids.job }),
    memory("berlin", { status: "retracted" }),
  ];

  const views = Object.fromEntries(
    memories.map((row) => [row.id, view(row)]),
  );
  views[ids.auckland].chain.members = [
    { id: ids.wellington, superseded_by: ids.auckland, retracted: false, hidden: false },
    { id: ids.auckland, superseded_by: null, retracted: false, hidden: false },
  ];
  views[ids.wellington].chain = views[ids.auckland].chain;
  views[ids.job].chain.ends = [ids.oldJob];
  views[ids.oldJob].chain.ended_by = ids.job;
  views[ids.berlin].retracted_at = "2026-10-02T10:00:00Z";

  const homeChunk = {
    id: ids.chunkHome,
    position: 0,
    heading_path: "Home",
    start: 0,
    end: 30,
    state: "extracted",
    error_count: 0,
    error_kind: null,
    failed_at: null,
    memories: [ids.auckland],
    mentions: [],
  };
  const workChunk = {
    id: ids.chunkWork,
    position: 1,
    heading_path: "Work",
    start: 31,
    end: 60,
    state: "failed",
    error_count: 5,
    error_kind: "llm_timeout",
    failed_at: "2026-10-02T03:00:00Z",
    memories: [],
    mentions: [],
  };

  const sources = [
    source(ids.documentV2, {
      ingested_at: "2026-10-01T08:00:00Z",
      text: "# Home\nSam lives in Auckland now.\n# Work\nSam started at Effectful.",
      chunks: [homeChunk, workChunk],
    }),
    source(ids.documentV1, {
      ingested_at: "2026-09-01T08:00:00Z",
      text: "# Home\nSam lives in Wellington.",
      chunks: [{ ...homeChunk, id: "01a10400-0000-7000-8000-0000000000c9", memories: [ids.wellington] }],
    }),
    source(ids.recipes, {
      document_id: "recipes.md",
      ingested_at: "2026-08-01T08:00:00Z",
      text: null,
      gone: { reason: "removed" },
      chunks: [],
    }),
    source(ids.turn, {
      kind: "turn",
      document_id: null,
      session_id: "session-1",
      message_at: "2026-10-02T08:00:00Z",
      ingested_at: "2026-10-02T08:00:01Z",
      text: "My daughter Ada starts school next week.",
      reply: "That's exciting!",
      chunks: [],
    }),
  ];
  const versions = (documentId) =>
    sources
      .filter((s) => s.document_id === documentId)
      .map((s) => ({ id: s.id, ingested_at: s.ingested_at, gone: s.gone }));
  for (const s of sources) s.versions = s.document_id ? versions(s.document_id) : [];

  return {
    banks: [
      bank("main", {
        memories: { live: 112, superseded: 9, ended: 6, retracted: 7, forgetting: 0 },
        kept: 4,
        sources: { turns: 80, documents: 5, document_versions: 8, tombstoned: 1, removed: 1 },
        chunks: { queued: 2, failed: 3 },
      }),
      bank("work", {
        memories: { live: 31, superseded: 0, ended: 0, retracted: 0, forgetting: 0 },
        kept: 0,
        sources: { turns: 12, documents: 0, document_versions: 0, tombstoned: 0, removed: 0 },
        chunks: { queued: 0, failed: 0 },
      }),
    ],
    memories,
    views,
    sources,
    // The enabled models take 700 of the 800 tokens.
    models: [
      model("profile", "User profile", {
        max_tokens: 500,
        entries: [{ id: ids.profileEntry, text: sentences.auckland, cites: [ids.auckland] }],
        last_refreshed_at: "2026-10-03T08:30:00Z",
      }),
      model("plans", "Plans", {
        kinds: ["event", "task"],
        max_tokens: 200,
        last_refreshed_at: "2026-10-01T06:00:00Z",
        last_error: "malformed",
        last_error_at: "2026-10-02T06:00:00Z",
      }),
      model("travel", "Travel", { kinds: ["event"], max_tokens: 150, enabled: false }),
    ],
    /// `mental_models.budget`.
    budget: 800,
    queued: [],
    failed: [
      {
        chunk: ids.chunkWork,
        source: ids.documentV2,
        error_count: 5,
        error_kind: "llm_timeout",
        status: 504,
        failed_at: "2026-10-02T03:00:00Z",
      },
      {
        chunk: ids.chunkTurn,
        source: ids.turn,
        error_count: 5,
        error_kind: "malformed",
        status: null,
        failed_at: "2026-10-02T04:00:00Z",
      },
    ],
    status: {
      version: "0.1.0",
      attention: [],
      purge: { state: "running" },
      deletion_fingerprint: "f1f1f1",
      banks: { main: { queued: 2, failed_chunks: 3, failed_refreshes: 0 } },
      last_sweep: null,
      pre_migration_copy: null,
      last_backup_at: null,
    },
    plan: {
      pause: { state: "running" },
      current: "f1f1f1",
      changed: [],
      memories: 0,
      sources: 0,
      failed_chunks: 0,
      recalls: 0,
    },
    /// Rows per memory page.
    pageSize: 50,
  };
}

function bank(name, counts) {
  return {
    name,
    owner_name: "Sam",
    assistant_name: "Hermes",
    timezone: "Pacific/Auckland",
    created_at: "2026-01-01T00:00:00Z",
    turns: counts.sources.turns,
    last_turn_at: "2026-10-02T08:00:00Z",
    kinds: { fact: 100, event: 12 },
    significance: { notable: 80, major: 20 },
    models: 3,
    ...counts,
  };
}

/// One more document, its only version, for a test to add to `state.sources`.
export function document(id, documentId, text) {
  const added = source(id, { document_id: documentId, ingested_at: "2026-10-02T08:00:00Z", text, chunks: [] });
  added.versions = [{ id, ingested_at: added.ingested_at, gone: null }];
  return added;
}

function source(id, fields) {
  return {
    id,
    kind: "document",
    document_id: DOCUMENT,
    session_id: null,
    message_at: null,
    observed_at: fields.ingested_at,
    reference_date: null,
    timezone: "Pacific/Auckland",
    platform: null,
    author_name: null,
    reply: null,
    gone: null,
    secret_kinds: [],
    ...fields,
  };
}

function view(row) {
  return {
    id: row.id,
    sentence: row.sentence,
    kind: row.kind,
    phase: row.phase,
    window: {
      valid_from: null,
      valid_until: null,
      until_event: null,
      window_confidence: "none",
      due_at: null,
      volatility: null,
      recurrence: null,
      recurrence_rrule: null,
      timezone: "Pacific/Auckland",
    },
    observed_at: row.observed_at,
    created_at: row.created_at,
    hidden_at: null,
    retracted_at: null,
    significance: row.significance,
    source: {
      source: row.source,
      kind: row.source_kind,
      session_id: row.session_id,
      document_id: row.document_id,
      message_at: null,
      chunk: ids.chunkHome,
      start: 7,
      end: 30,
      passage: `${row.sentence.replace(/\.$/, "")} (passage)`,
      gone: null,
      secret_kinds: [],
    },
    accesses: [{ kind: "created", at: row.created_at, turn: 1, source: row.source, inherited_from: null }],
    edits: [],
    chain: {
      head: row.superseded_by ?? row.id,
      members: [{ id: row.id, superseded_by: row.superseded_by, retracted: false, hidden: false }],
      ended_by: row.ended_by,
      ends: [],
    },
    entities: [],
    strength: {
      value: row.strength,
      significance_boost: 0.5,
      recent_use: 0.7,
      lasting_floor: 0,
      occasions: 1,
      threshold: 0,
      recallable: row.recallable,
    },
    purge: {
      head: row.superseded_by ?? row.id,
      head_strength: row.strength,
      delta: 1,
      line: -1,
      guards: [{ guard: "strength" }],
      eligible_now: false,
    },
    projection: {
      basis: "if it isn't used again; bank days from now, and the earliest world date at full speed",
      fade: row.fade,
      purge: row.purge,
    },
  };
}

function json(status, body, headers = {}) {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

function error(status, message, headers) {
  return json(status, { error: message }, headers);
}

export class FakeDaemon {
  /// `token: null` is a loopback daemon without one, which checks nothing.
  /// `eraseOnForget` runs the erase before forget answers, as the daemon does
  /// when nothing is queued: the forgotten chain's rows are gone at once, so
  /// reading one of its memories is a 404. Otherwise they stay, hidden, as
  /// they do while the erase waits behind the queue.
  constructor({ token = TOKEN, state = fixtures(), eraseOnForget = false } = {}) {
    this.token = token;
    this.eraseOnForget = eraseOnForget;
    this.state = state;
    this.requests = [];
    this.fetch = this.fetch.bind(this);
  }

  /// The requests the dashboard sent to `/v1` routes, oldest first.
  calls(method, path) {
    return this.requests.filter(
      (r) => (!method || r.method === method) && (!path || (typeof path === "string" ? r.path === path : path.test(r.path))),
    );
  }

  /// The query of the newest request to `path`.
  lastQuery(path) {
    const found = this.calls("GET", path).at(-1);
    if (!found) throw new Error(`no GET ${path}`);
    return found.query;
  }

  async fetch(input, init = {}) {
    const isRequest = typeof input === "object" && input !== null && "url" in input;
    const url = new URL(isRequest ? input.url : String(input), "http://127.0.0.1:7720/dashboard");
    const method = (init.method ?? (isRequest ? input.method : "GET")).toUpperCase();
    const headers = new Headers(init.headers ?? (isRequest ? input.headers : undefined));
    const raw = init.body ?? (isRequest && method !== "GET" ? await input.text() : undefined);
    const body = raw ? JSON.parse(raw) : undefined;
    const request = {
      method,
      path: url.pathname,
      query: url.searchParams,
      authorization: headers.get("authorization"),
      body,
    };
    this.requests.push(request);
    await Promise.resolve();

    if (url.pathname === "/v1/health") {
      return json(200, { version: "0.1.0", ready: true, now: NOW });
    }
    if (this.token !== null && request.authorization !== `Bearer ${this.token}`) {
      return error(401, "a bearer token is required: set ASPHODEL_TOKEN on the client", {
        "www-authenticate": "Bearer",
      });
    }
    return this.route(request);
  }

  route({ method, path, query, body }) {
    const state = this.state;
    const segments = path.split("/").slice(1).map(decodeURIComponent);
    const [v1, top, bank, collection, id, action] = segments;
    if (v1 !== "v1") return error(404, "no such route");

    if (top === "status" && method === "GET") return json(200, state.status);
    if (top === "purge" && bank === "plan" && method === "GET") return json(200, state.plan);
    if (top === "purge" && bank === "ack" && method === "POST") {
      if (body?.hash !== state.status.deletion_fingerprint) {
        return error(409, "that isn't this daemon's deletion fingerprint; `purge plan` shows it");
      }
      state.status.purge = { state: "running" };
      state.status.attention = state.status.attention.filter((line) => !/purge/i.test(line));
      state.plan.pause = { state: "running" };
      return new Response(null, { status: 204 });
    }
    if (top !== "banks") return error(404, "no such route");
    if (bank === undefined && method === "GET") return json(200, { banks: state.banks });
    if (!state.banks.some((b) => b.name === bank)) return error(404, "unknown bank");
    if (bank !== "main") return this.emptyBank(collection, method);

    if (collection === "memories" && id === undefined && method === "GET") return this.memories(query);
    if (collection === "memories" && action === undefined && method === "GET") {
      const found = state.views[id];
      return found ? json(200, found) : error(404, "unknown memory");
    }
    if (collection === "memories" && action === "retract" && method === "POST") return this.retract(id);
    if (collection === "forget" && method === "POST") return this.forget(body.ids);
    if (collection === "sources" && id === undefined && method === "GET") return this.sources(query);
    if (collection === "sources" && method === "GET") {
      const found = state.sources.find((s) => s.id === id);
      return found ? json(200, found) : error(404, "unknown source");
    }
    // The document id travels in the body, never the path: a path can't
    // carry `..` or `.` past URL normalization.
    if (collection === "documents" && id === "remove" && method === "POST") {
      if (typeof body?.document_id !== "string" || body.document_id === "") {
        return error(400, "document_id is required");
      }
      return this.removeDocument(body.document_id);
    }
    if (collection === "chunks" && id === undefined && method === "GET") {
      const failedOnly = query.get("failed") === "true";
      return json(200, { queued: failedOnly ? [] : state.queued, failed: state.failed });
    }
    if (collection === "chunks" && id === "retry" && method === "POST") return this.retry(body?.chunks);
    if (collection === "models" && id === undefined && method === "GET") {
      return json(200, { models: state.models, budget: state.budget });
    }
    if (collection === "models" && action === undefined && method === "PATCH") return this.editModel(id, body);
    return error(404, "no such route");
  }

  emptyBank(collection, method) {
    if (collection === "memories" && method === "GET") {
      return json(200, {
        memories: [],
        total: 0,
        statuses: { live: 0, superseded: 0, ended: 0, retracted: 0, forgetting: 0 },
        next_cursor: null,
        as_of: NOW,
      });
    }
    if (collection === "sources" && method === "GET") return json(200, { sources: [], total: 0, next_cursor: null });
    if (collection === "chunks" && method === "GET") return json(200, { queued: [], failed: [] });
    if (collection === "models" && method === "GET") return json(200, { models: [], budget: 800 });
    return error(404, "no such route");
  }

  memories(query) {
    const words = (query.get("q") ?? "").toLowerCase().split(/\s+/).filter(Boolean);
    const matching = this.state.memories.filter((m) =>
      words.every((word) => m.sentence.toLowerCase().split(/\W+/).some((w) => w.startsWith(word))),
    );
    const statuses = { live: 0, superseded: 0, ended: 0, retracted: 0, forgetting: 0 };
    for (const m of matching) statuses[m.status] += 1;
    const status = query.get("status");
    const listed = status ? matching.filter((m) => m.status === status) : matching;
    const offset = Number(query.get("cursor") ?? 0);
    const size = Math.min(Number(query.get("limit") ?? this.state.pageSize), this.state.pageSize);
    const page = listed.slice(offset, offset + size);
    return json(200, {
      memories: page,
      total: listed.length,
      statuses,
      next_cursor: offset + size < listed.length ? String(offset + size) : null,
      as_of: NOW,
    });
  }

  retract(id) {
    const row = this.state.memories.find((m) => m.id === id);
    const shown = this.state.views[id];
    if (!row || row.status === "forgetting") return error(404, "unknown memory");
    if (row.status === "retracted") return error(409, "already retracted");
    if (row.status === "superseded") return error(409, `superseded: retract its head ${row.superseded_by}`);
    const retractedAt = "2026-10-03T09:05:00Z";
    row.status = "retracted";
    shown.retracted_at = retractedAt;
    const reopened = shown.chain.ends;
    for (const other of reopened) {
      const ended = this.state.memories.find((m) => m.id === other);
      ended.status = "live";
      ended.ended_by = null;
      this.state.views[other].chain.ended_by = null;
    }
    shown.chain.ends = [];
    return json(200, { memory: id, retracted_at: retractedAt, reopened });
  }

  forget(given) {
    const forgotten = [];
    const unknown = [];
    for (const id of given) {
      const shown = this.state.views[id];
      const row = this.state.memories.find((m) => m.id === id);
      if (!shown || row.status === "forgetting") {
        unknown.push(id);
        continue;
      }
      for (const member of shown.chain.members) {
        forgotten.push(member.id);
        this.state.memories.find((m) => m.id === member.id).status = "forgetting";
        this.state.views[member.id].hidden_at = "2026-10-03T09:05:00Z";
      }
    }
    if (this.eraseOnForget) {
      this.state.memories = this.state.memories.filter((m) => !forgotten.includes(m.id));
      for (const id of forgotten) delete this.state.views[id];
    }
    return json(200, { forgotten, unknown });
  }

  sources(query) {
    const kind = query.get("kind");
    const documentId = query.get("document_id");
    const listed = this.state.sources
      .filter((s) => (!kind || s.kind === kind) && (!documentId || s.document_id === documentId))
      .map(({ id, kind, document_id, session_id, message_at, observed_at, ingested_at, gone, secret_kinds, chunks }) => ({
        id,
        kind,
        document_id,
        session_id,
        message_at,
        observed_at,
        ingested_at,
        chunks: chunks.length,
        queued: chunks.filter((c) => c.state === "queued").length,
        failed: chunks.filter((c) => c.state === "failed").length,
        memories: chunks.reduce((sum, c) => sum + c.memories.length, 0),
        gone,
        secret_kinds,
      }));
    return json(200, { sources: listed, total: listed.length, next_cursor: null });
  }

  removeDocument(documentId) {
    const versions = this.state.sources.filter((s) => s.document_id === documentId && s.gone?.reason !== "removed");
    if (versions.length === 0) return error(404, "unknown document");
    const forgotten = this.state.memories.filter((m) => m.document_id === documentId).map((m) => m.id);
    for (const s of versions) {
      s.text = null;
      s.gone = { reason: "removed" };
    }
    for (const id of forgotten) {
      this.state.memories.find((m) => m.id === id).status = "forgetting";
      this.state.views[id].hidden_at = "2026-10-03T09:05:00Z";
    }
    return json(200, { document_id: documentId, sources: versions.map((s) => s.id), forgotten, dequeued: 0 });
  }

  /// Only `enabled` is edited here. Enabling is refused, changing nothing,
  /// when the enabled models' `max_tokens` would sum past the budget.
  editModel(name, edit) {
    const found = this.state.models.find((m) => m.name === name);
    if (!found) return error(404, "no such model");
    if (edit?.enabled === true && !found.enabled) {
      const requested = this.state.models
        .filter((m) => m.enabled || m === found)
        .reduce((sum, m) => sum + m.max_tokens, 0);
      if (requested > this.state.budget) {
        return error(422, `${requested} tokens is over the ${this.state.budget}-token budget for mental models`);
      }
    }
    if (typeof edit?.enabled === "boolean") found.enabled = edit.enabled;
    return json(200, found);
  }

  retry(chunks) {
    const named = chunks ?? this.state.failed.map((f) => f.chunk);
    const retried = named.filter((c) => this.state.failed.some((f) => f.chunk === c));
    const unknown = named.filter((c) => !retried.includes(c));
    this.state.failed = this.state.failed.filter((f) => !retried.includes(f.chunk));
    return json(200, { retried, unknown });
  }
}
