"""The HTTP/JSON client of the daemon (TIM-94, decisions 2 and 10).

Standard library only. ``url`` is ``http://host:port`` or ``unix:/path``,
the two forms the daemon's ``--listen`` takes. A bearer token, when set, goes
in ``Authorization``. Timeouts are per request and are set by the caller:
each hook has its own budget (:class:`asphodel_plugin.provider.Timeouts`).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from .breaker import CircuitBreaker


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
        raise NotImplementedError

    # Convenience wrappers, one per route the plugin uses.

    def health(self, *, timeout: float) -> Tuple[int, Any]:
        """``GET /v1/health``: 503 while starting or draining, 200 when ready."""
        raise NotImplementedError

    def put_bank(self, bank: str, identity: dict, *, timeout: float) -> Any:
        raise NotImplementedError

    def ingest_turn(self, bank: str, turn: dict, *, timeout: float) -> Any:
        raise NotImplementedError

    def prefetch(self, bank: str, request: dict, *, timeout: float) -> Any:
        raise NotImplementedError

    def system_prompt(self, bank: str, session_id: Optional[str], *, timeout: float) -> Any:
        raise NotImplementedError

    def recall(self, bank: str, request: dict, *, timeout: float) -> Any:
        raise NotImplementedError

    def forget(self, bank: str, ids: list, session_id: Optional[str], *, timeout: float) -> Any:
        raise NotImplementedError

    def keep(self, bank: str, ids: list, *, timeout: float) -> Any:
        raise NotImplementedError

    def unkeep(self, bank: str, ids: list, *, timeout: float) -> Any:
        raise NotImplementedError

    def clear_session(self, bank: str, session_id: str, *, timeout: float) -> None:
        raise NotImplementedError
