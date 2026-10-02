"""The HTTP/JSON client of the daemon (TIM-94, decisions 2 and 10).

Standard library only. ``url`` is ``http://host:port`` or ``unix:/path``,
the two forms the daemon's ``--listen`` takes. A bearer token, when set, goes
in ``Authorization``. Timeouts are per request and are set by the caller:
each hook has its own budget (:class:`asphodel_plugin.provider.Timeouts`).
"""

from __future__ import annotations

import http.client
import json
import socket
from typing import Any, Optional, Tuple
from urllib.parse import quote, urlencode, urlsplit

from .breaker import CircuitBreaker

UNIX_PREFIX = "unix:"


class DaemonUnavailable(Exception):
    """The daemon could not be reached: connection refused or reset, a
    timeout, or the circuit breaker is open. These count as connection
    failures for the breaker."""


class DaemonError(Exception):
    """The daemon answered with a non-2xx status. ``message`` is the daemon's
    ``error`` field, which names kinds and ids only (ADR 0010)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"daemon answered {status}: {message}")
        self.status = status
        self.message = message


class _UnixConnection(http.client.HTTPConnection):
    """``http.client`` over a Unix domain socket, for ``unix:/path``."""

    def __init__(self, path: str, *, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.timeout)
            sock.connect(self._path)
        except BaseException:
            sock.close()
            raise
        self.sock = sock


def _segment(value: str) -> str:
    return quote(value, safe="")


class DaemonClient:
    def __init__(self, url: str, *, token: Optional[str] = None, breaker: Optional[CircuitBreaker] = None) -> None:
        self.url = url
        self.token = token
        self.breaker = breaker or CircuitBreaker()

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        timeout: float,
        query: Optional[dict] = None,
    ) -> Tuple[int, Any]:
        """One request. ``path`` is under ``/v1``. Returns ``(status, json)``,
        with ``None`` for a 204. Raises :class:`DaemonUnavailable` or
        :class:`DaemonError`; never anything else."""
        status, payload = self._exchange(method, path, body, timeout=timeout, query=query)
        if not 200 <= status < 300:
            message = payload.get("error") if isinstance(payload, dict) else None
            raise DaemonError(status, str(message) if message is not None else "")
        return status, payload

    def _exchange(
        self,
        method: str,
        path: str,
        body: Any,
        *,
        timeout: float,
        query: Optional[dict],
    ) -> Tuple[int, Any]:
        """Sends one request and returns whatever status came back. Only a
        failure to get an answer raises, and it counts against the breaker."""
        if not self.breaker.allow():
            raise DaemonUnavailable("circuit breaker open")
        target = "/v1" + path
        if query:
            target += "?" + urlencode(query)
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            connection = self._connection(timeout)
        except ValueError as error:
            self.breaker.record_failure()
            raise DaemonUnavailable(str(error)) from None
        try:
            connection.request(method, target, body=data, headers=headers)
            response = connection.getresponse()
            status = response.status
            raw = response.read()
        except (OSError, http.client.HTTPException) as error:
            self.breaker.record_failure()
            raise DaemonUnavailable(type(error).__name__) from None
        finally:
            connection.close()
        self.breaker.record_success()
        try:
            payload = json.loads(raw) if raw else None
        except ValueError:
            payload = None
        return status, payload

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        if self.url.startswith(UNIX_PREFIX):
            path = self.url[len(UNIX_PREFIX):]
            if not path:
                raise ValueError("unix: URL without a socket path")
            return _UnixConnection(path, timeout=timeout)
        parts = urlsplit(self.url)
        if not parts.hostname:
            raise ValueError("daemon URL has no host")
        if parts.scheme == "http":
            return http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
        if parts.scheme == "https":
            return http.client.HTTPSConnection(parts.hostname, parts.port, timeout=timeout)
        raise ValueError(f"unsupported URL scheme {parts.scheme!r}")

    # Convenience wrappers, one per route the plugin uses.

    def health(self, *, timeout: float) -> Tuple[int, Any]:
        """``GET /v1/health``: 503 while starting or draining, 200 when ready."""
        status, payload = self._exchange("GET", "/health", None, timeout=timeout, query=None)
        if status not in (200, 503):
            message = payload.get("error") if isinstance(payload, dict) else None
            raise DaemonError(status, str(message) if message is not None else "")
        return status, payload

    def put_bank(self, bank: str, identity: dict, *, timeout: float) -> Any:
        return self.request("PUT", f"/banks/{_segment(bank)}", identity, timeout=timeout)[1]

    def ingest_turn(self, bank: str, turn: dict, *, timeout: float) -> Any:
        return self.request("POST", f"/banks/{_segment(bank)}/turns", turn, timeout=timeout)[1]

    def prefetch(self, bank: str, request: dict, *, timeout: float) -> Any:
        return self.request("POST", f"/banks/{_segment(bank)}/prefetch", request, timeout=timeout)[1]

    def system_prompt(self, bank: str, session_id: Optional[str], *, timeout: float) -> Any:
        query = {"session_id": session_id} if session_id else None
        return self.request("GET", f"/banks/{_segment(bank)}/system-prompt", timeout=timeout, query=query)[1]

    def recall(self, bank: str, request: dict, *, timeout: float) -> Any:
        return self.request("POST", f"/banks/{_segment(bank)}/recall", request, timeout=timeout)[1]

    def forget(self, bank: str, ids: list, session_id: Optional[str], *, timeout: float) -> Any:
        body: dict = {"ids": ids}
        if session_id:
            body["session_id"] = session_id
        return self.request("POST", f"/banks/{_segment(bank)}/forget", body, timeout=timeout)[1]

    def keep(self, bank: str, ids: list, *, timeout: float) -> Any:
        return self.request("POST", f"/banks/{_segment(bank)}/keep", {"ids": ids}, timeout=timeout)[1]

    def unkeep(self, bank: str, ids: list, *, timeout: float) -> Any:
        return self.request("POST", f"/banks/{_segment(bank)}/unkeep", {"ids": ids}, timeout=timeout)[1]

    def clear_session(self, bank: str, session_id: str, *, timeout: float) -> None:
        self.request("POST", f"/banks/{_segment(bank)}/sessions/{_segment(session_id)}/clear", timeout=timeout)
