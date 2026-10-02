"""The ``MemoryProvider`` implementation: TIM-94 decision 5's hook table.

Threading and budgets follow TIM-88: ``prefetch`` is the only hook on the
reply path and Hermes cuts it off at 8 s, so the plugin's own budget is 3 s.
``system_prompt_block`` runs at session start with a 2 s budget and one
retry. ``sync_turn`` already runs on Hermes' single background worker, so it
sends synchronously and spools on failure. Nothing here ever raises into
Hermes: every hook catches everything and logs at ``warning`` or below with
ids, counts and status codes only (ADR 0010). Content is logged only at
:data:`TRACE`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus

from .breaker import CircuitBreaker
from .client import DaemonClient
from .config import PluginConfig
from .spool import Spool

log = logging.getLogger(__name__)

PROVIDER_NAME = "asphodel"
#: Shown on the "🧠 recalled N memories" status line.
PROVIDER_LABEL = "Asphodel"
#: The daemon major version this plugin was written for. ``initialize`` warns
#: when the health response's major differs (TIM-94, decision 2).
DAEMON_MAJOR_VERSION = 0
#: Python's logging has no TRACE; this is the level content may be logged at.
TRACE = 5
#: Hermes' ``agent_context`` value for a user-facing agent. Only these ingest.
PRIMARY_CONTEXT = "primary"
#: Hermes' built-in memory flags that ``post_setup`` turns off and
#: ``initialize`` warns about (TIM-94, round 1 item 5).
BUILTIN_MEMORY_FLAGS = ("memory_enabled", "user_profile_enabled")


@dataclass(frozen=True)
class Timeouts:
    """Client timeouts in seconds, one per hook (TIM-94, decisions 5 and 8).
    Tests shorten them; production uses the defaults."""

    health: float = 2.0
    prefetch: float = 3.0
    system_prompt: float = 2.0
    #: Attempts beyond the first for ``system_prompt_block``.
    system_prompt_retries: int = 1
    ingest: float = 10.0
    tool: float = 10.0
    session_clear: float = 2.0


class AsphodelMemoryProvider(MemoryProvider):
    """One instance per Hermes agent, so effectively one per session. Every
    network call goes through one :class:`DaemonClient` and one
    :class:`CircuitBreaker`."""

    def __init__(
        self,
        *,
        timeouts: Optional[Timeouts] = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.timeouts = timeouts or Timeouts()
        self._clock = clock
        self._wall_clock = wall_clock
        self.config: Optional[PluginConfig] = None
        self.client: Optional[DaemonClient] = None
        self.spool: Optional[Spool] = None
        self.breaker = CircuitBreaker(clock=clock)
        self.bank: Optional[str] = None
        self._hermes_home: Optional[str] = None
        self._session_id: str = ""
        self._platform: Optional[str] = None
        self._agent_context: Optional[str] = None
        self._profile: Optional[str] = None
        # Decision 1: the current turn's author, from on_turn_start.
        self._author_id: Optional[str] = None
        self._author_name: Optional[str] = None
        self._author_is_bot: bool = False
        # Decision 6: the recall_id prefetch stored for the session, which
        # sync_turn echoes once; TIM-99: the last prefetch query, sent as the
        # previous message and dropped on memory_forget.
        self._pending_recall_id: Dict[str, str] = {}
        self._last_query: Dict[str, str] = {}
        self._last_injected: int = 0
        # TIM-95 decision 4: a block fetched before the session id was known.
        self._pending_block_id: Optional[str] = None

    # -- identity and availability -------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        """Config only (TIM-94, decision 3): True when a URL resolves from
        ``ASPHODEL_URL`` or ``config.json`` under the active ``HERMES_HOME``.
        Never touches the network; a dead daemon never drops the provider."""
        raise NotImplementedError

    def unavailable_reason(self) -> str:
        raise NotImplementedError

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        """Loads config, makes one health probe (warning through
        ``warning_callback`` when the daemon isn't ready or its major version
        differs), ``PUT``s the bank with the configured identity, and warns if
        Hermes' built-in memory flags are still on. Never fails."""
        raise NotImplementedError

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Records ``author_id``, ``author_name`` and ``author_is_bot`` for the
        owner check (decision 1)."""
        raise NotImplementedError

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Rebinds to ``new_session_id``. On compression
        (``reason="compression"``), ``reset`` or ``rewound`` it also clears the
        old session's in-context set through
        ``POST /v1/banks/{bank}/sessions/{id}/clear`` and drops the pending
        recall id, last query and block mapping."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Nothing to drain: the plugin owns no threads."""

    def backup_paths(self) -> List[str]:
        """``[]``: the data lives in the daemon's container (decision 5)."""
        return []

    # -- the reply path ------------------------------------------------------

    def system_prompt_block(self) -> str:
        """``GET /v1/banks/{bank}/system-prompt?session_id=...`` with the
        ``system_prompt`` budget and ``system_prompt_retries`` more attempts
        on a connection failure. Returns the block's text, or "" when the
        daemon can't be reached or the breaker is open."""
        raise NotImplementedError

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """``POST /v1/banks/{bank}/prefetch`` with the query, the session's
        last prefetch query as ``previous_query`` and, once, a pending block
        id. Stores the ``recall_id`` for ``sync_turn`` and the injected count
        for ``recall_status``. "" on any failure."""
        raise NotImplementedError

    def recall_status(self) -> Optional[RecallStatus]:
        """The last prefetch's injected count; ``None`` when it injected
        nothing or failed."""
        raise NotImplementedError

    # -- ingest --------------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        turn_author: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ingests the turn when ``agent_context`` is ``primary`` and
        ``ingest`` is on. Builds the ``Turn`` body with :func:`build_turn`,
        ``POST``s it, and spools it on a connection failure or 5xx. After a
        2xx it replays the spool."""
        raise NotImplementedError

    def build_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str,
        messages: Optional[List[Dict[str, Any]]],
        turn_author: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """The ``POST /v1/banks/{bank}/turns`` body: ``session_id``,
        ``message_at`` (RFC 3339, from the user row's epoch), ``timezone``
        (``hermes_time.get_timezone_name()`` or the configured default),
        ``user_text`` with the backfill stripped, ``assistant_text``,
        ``author`` (``{id, name, is_bot}`` or ``None``), ``platform``, the
        echoed ``recall_id`` and ``forget_requested``."""
        raise NotImplementedError

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatches the four tools. Owner-only tools return ``tool_error``
        without a request on a non-owner's turn. ``memory_recall`` returns the
        daemon's ``results`` list as JSON. A daemon error or an unreachable
        daemon returns ``tool_error``; ``memory_forget`` drops the session's
        last prefetch query on success."""
        raise NotImplementedError

    def is_owner_turn(self) -> bool:
        raise NotImplementedError

    # -- setup ---------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        raise NotImplementedError

    def post_setup(self, hermes_home: str, config: Dict[str, Any], *, prompt: Callable[[str], str] = input) -> None:
        """``hermes memory setup``: prompts for each schema field (empty keeps
        the default), writes ``config.json``, sets ``memory.provider`` to
        ``asphodel`` and ``memory.memory_enabled`` and
        ``memory.user_profile_enabled`` to false, and saves Hermes' config
        through ``hermes_cli.config.save_config``."""
        raise NotImplementedError
