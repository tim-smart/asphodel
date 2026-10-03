// The daemon's HTTP API, one function per route the dashboard calls.
//
// The bearer token lives in the tab's session storage and nowhere else. A
// call goes without one until the daemon asks for it with a 401: a loopback
// daemon with no token configured never does. A 401 on a call that carried
// a token means the token is wrong, so it is dropped.

const TOKEN_KEY = "asphodel.token";

/// The daemon wants a token, or rejected the one sent.
export class Unauthorized extends Error {
  constructor(rejected) {
    super(rejected ? "The daemon rejected that token." : "The daemon needs its token.");
    this.rejected = rejected;
  }
}

/// The daemon refused a request; `message` is its `error`.
export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

const path = (...segments) => segments.map((s) => encodeURIComponent(s)).join("/");

const query = (params) => {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== null && value !== "") search.set(key, value);
  }
  const text = search.toString();
  return text ? `?${text}` : "";
};

export function client(fetch, storage) {
  const token = {
    get: () => storage.getItem(TOKEN_KEY),
    set: (value) => storage.setItem(TOKEN_KEY, value),
    clear: () => storage.removeItem(TOKEN_KEY),
  };

  async function call(method, url, body) {
    const sent = token.get();
    const headers = { accept: "application/json" };
    if (sent) headers.authorization = `Bearer ${sent}`;
    if (body !== undefined) headers["content-type"] = "application/json";
    const response = await fetch(url, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (response.status === 401) {
      if (sent) token.clear();
      throw new Unauthorized(Boolean(sent));
    }
    const text = response.status === 204 ? "" : await response.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = null;
    }
    if (!response.ok) {
      throw new ApiError(response.status, data?.error ?? `The daemon answered ${response.status}.`);
    }
    return data;
  }

  const bank = (name) => `/v1/banks/${path(name)}`;

  return {
    token,
    status: () => call("GET", "/v1/status"),
    banks: () => call("GET", "/v1/banks"),
    purgePlan: () => call("GET", "/v1/purge/plan"),
    purgeAck: (hash) => call("POST", "/v1/purge/ack", { hash }),
    memories: (name, params) => call("GET", `${bank(name)}/memories${query(params)}`),
    memory: (name, id) => call("GET", `${bank(name)}/memories/${path(id)}`),
    retract: (name, id) => call("POST", `${bank(name)}/memories/${path(id)}/retract`),
    forget: (name, ids) => call("POST", `${bank(name)}/forget`, { ids }),
    keep: (name, ids) => call("POST", `${bank(name)}/keep`, { ids }),
    unkeep: (name, ids) => call("POST", `${bank(name)}/unkeep`, { ids }),
    sources: (name, params) => call("GET", `${bank(name)}/sources${query(params)}`),
    source: (name, id) => call("GET", `${bank(name)}/sources/${path(id)}`),
    // The id goes in the body exactly as ingested: a URL path would lose `..`
    // and `.` segments to normalization and could name another document.
    removeDocument: (name, documentId) => call("POST", `${bank(name)}/documents/remove`, { document_id: documentId }),
    chunks: (name) => call("GET", `${bank(name)}/chunks`),
    retryChunks: (name, chunks) => call("POST", `${bank(name)}/chunks/retry`, chunks ? { chunks } : {}),
  };
}
