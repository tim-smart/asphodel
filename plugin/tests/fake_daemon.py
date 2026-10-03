"""A fake Asphodel daemon: ``http.server`` on an ephemeral loopback port,
answering the routes the plugin uses with bodies shaped like the Rust types in
``crates/asphodel/src/serve/api.rs``. It records every request, and a test
can override a route's response, delay it past the plugin's budget, or drop
connections to simulate a daemon that's gone.

Like the real daemon, every route but health answers 503 while ``ready`` is
false. With ``enforce_banks`` on, a bank route answers 404 until a ``PUT``
has created the bank, as the real API does."""

from __future__ import annotations

import json
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlsplit

Response = Tuple[int, Any]

_ROUTES = [
    ("GET", re.compile(r"^/v1/health$"), "health"),
    ("PUT", re.compile(r"^/v1/banks/(?P<bank>[^/]+)$"), "put_bank"),
    ("POST", re.compile(r"^/v1/banks/(?P<bank>[^/]+)/turns$"), "turns"),
    ("POST", re.compile(r"^/v1/banks/(?P<bank>[^/]+)/prefetch$"), "prefetch"),
    ("POST", re.compile(r"^/v1/banks/(?P<bank>[^/]+)/recall$"), "recall"),
    ("POST", re.compile(r"^/v1/banks/(?P<bank>[^/]+)/forget$"), "forget"),
    ("POST", re.compile(r"^/v1/banks/(?P<bank>[^/]+)/keep$"), "keep"),
    ("POST", re.compile(r"^/v1/banks/(?P<bank>[^/]+)/unkeep$"), "unkeep"),
    ("POST", re.compile(r"^/v1/banks/(?P<bank>[^/]+)/sessions/(?P<session>[^/]+)/clear$"), "clear"),
    ("GET", re.compile(r"^/v1/banks/(?P<bank>[^/]+)/system-prompt$"), "system_prompt"),
]

NOW = "2026-10-02T02:19:17.622328Z"
INJECTION = "- The user drinks oolong tea every morning.\n- The user's dentist is Dr Rao."


@dataclass
class Request:
    key: str
    method: str
    path: str
    params: Dict[str, str]
    query: Dict[str, str]
    body: Any
    headers: Dict[str, str]

    @property
    def bank(self) -> Optional[str]:
        return self.params.get("bank")

    @property
    def session(self) -> Optional[str]:
        return self.params.get("session")

    @property
    def authorization(self) -> Optional[str]:
        return self.headers.get("authorization")


@dataclass
class FakeDaemon:
    version: str = "0.1.0"
    ready: bool = True
    enforce_banks: bool = False
    banks: Set[str] = field(default_factory=set)
    requests: List[Request] = field(default_factory=list)
    responses: Dict[str, Any] = field(default_factory=dict)
    delays: Dict[str, float] = field(default_factory=dict)
    drops: Dict[str, int] = field(default_factory=dict)
    _server: Optional[ThreadingHTTPServer] = None
    _thread: Optional[threading.Thread] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> "FakeDaemon":
        daemon = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # silence
                pass

            def _handle(self):
                daemon._handle(self)

            do_GET = do_POST = do_PUT = _handle

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=lambda: self._server.serve_forever(poll_interval=0.02), name="fake-daemon", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def port(self) -> int:
        assert self._server is not None
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def go_down(self) -> str:
        """Stops listening and returns the URL, which now refuses connections."""
        url = self.url
        self.stop()
        return url

    # -- scripting -------------------------------------------------------------

    def set_response(self, key: str, status: int, body: Any = None) -> None:
        self.responses[key] = (status, body)

    def set_handler(self, key: str, handler: Callable[[Request], Response]) -> None:
        self.responses[key] = handler

    def set_delay(self, key: str, seconds: float) -> None:
        self.delays[key] = seconds

    def drop_connections(self, key: str, count: int) -> None:
        """The next ``count`` requests to ``key`` are read and recorded, then
        the connection is closed with no response."""
        self.drops[key] = self.drops.get(key, 0) + count

    def requests_for(self, key: str) -> List[Request]:
        return [r for r in self.requests if r.key == key]

    # -- serving ---------------------------------------------------------------

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        parts = urlsplit(handler.path)
        key, params = "unknown", {}
        for method, pattern, name in _ROUTES:
            match = pattern.match(parts.path)
            if match and handler.command == method:
                key, params = name, match.groupdict()
                break
        length = int(handler.headers.get("Content-Length") or 0)
        raw = handler.rfile.read(length) if length else b""
        body = json.loads(raw) if raw else None
        query = {k: v[-1] for k, v in parse_qs(parts.query).items()}
        request = Request(
            key=key,
            method=handler.command,
            path=parts.path,
            params=params,
            query=query,
            body=body,
            headers={k.lower(): v for k, v in handler.headers.items()},
        )
        with self._lock:
            self.requests.append(request)
            drop = self.drops.get(key, 0)
            if drop:
                self.drops[key] = drop - 1
        if drop:
            try:
                handler.request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            handler.close_connection = True
            return
        delay = self.delays.get(key)
        if delay:
            time.sleep(delay)
        status, payload = self._respond(request)
        data = b"" if payload is None else json.dumps(payload).encode()
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            if data:
                handler.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _respond(self, request: Request) -> Response:
        if not self.ready and request.key != "health":
            return 503, {"error": "the daemon is starting: the store and models aren't ready yet"}
        if self.enforce_banks and request.bank is not None and request.key != "put_bank":
            if request.bank not in self.banks:
                return 404, {"error": f"unknown bank {request.bank}"}
        scripted = self.responses.get(request.key)
        if callable(scripted):
            status, body = scripted(request)
        elif scripted is not None:
            status, body = scripted
        else:
            status, body = self._default(request)
        if request.key == "put_bank" and 200 <= status < 300:
            with self._lock:
                self.banks.add(request.bank)
        return status, body

    def _default(self, request: Request) -> Response:
        key, body = request.key, request.body or {}
        if key == "health":
            if not self.ready:
                return 503, {"version": self.version, "ready": False, "now": NOW}
            return 200, {"version": self.version, "ready": True, "now": NOW}
        if key == "put_bank":
            return 201, {
                "id": str(uuid.uuid4()),
                "name": request.bank,
                "owner_name": body.get("owner_name"),
                "assistant_name": body.get("assistant_name"),
                "timezone": body.get("timezone") or "UTC",
                "embedding_model": "fake-embedder",
                "reranker_model": "fake-reranker",
                "created": True,
            }
        if key == "turns":
            return 200, {
                "source": str(uuid.uuid4()),
                "outcome": "tombstone" if body.get("forget_requested") else "stored",
                "chunks_queued": 1,
                "chunks_skipped": 0,
                "secret_kinds": [],
                "speaker": {"entity": str(uuid.uuid4()), "owner": True},
            }
        if key == "prefetch":
            return 200, {
                "recall_id": str(uuid.uuid4()),
                "text": INJECTION,
                "injected": [str(uuid.uuid4()), str(uuid.uuid4())],
                "reranked": True,
            }
        if key == "recall":
            return 200, {
                "recall_id": str(uuid.uuid4()),
                "results": [recalled("The user drinks oolong tea every morning.")],
                "reranked": True,
            }
        if key == "forget":
            return 200, {"forgotten": body.get("ids", []), "unknown": []}
        if key == "keep":
            return 200, {"kept": body.get("ids", []), "unknown": []}
        if key == "unkeep":
            return 200, {"unkept": body.get("ids", []), "unknown": []}
        if key == "clear":
            return 204, None
        if key == "system_prompt":
            return 200, {
                "id": str(uuid.uuid4()),
                "built_at": NOW,
                "text": "## Agenda (built 2026-10-02)\n- 2026-10-09: dentist, Dr Rao\n\nUse memory_recall for history.",
                "agenda": [str(uuid.uuid4())],
                "cited": [],
            }
        return 404, {"error": "no such route"}


def recalled(sentence: str, **overrides) -> Dict[str, Any]:
    result = {
        "id": str(uuid.uuid4()),
        "sentence": sentence,
        "kind": "recurring",
        "window": {
            "valid_from": None,
            "valid_until": None,
            "until_event": None,
            "due_at": None,
            "recurrence": "FREQ=DAILY",
            "uncertain": False,
        },
        "phase": "current",
        "observed_at": NOW,
        "strength": "strong",
        "kept": False,
    }
    result.update(overrides)
    return result
