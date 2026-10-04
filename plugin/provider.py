"""The ``MemoryProvider`` hook implementation.

``prefetch`` is the only hook on the
reply path and Hermes cuts it off at 8 s, so the plugin's own budget is 3 s.
``system_prompt_block`` runs at session start with a 2 s budget and one
retry. ``sync_turn`` already runs on Hermes' single background worker, so it
sends synchronously and spools on failure. Nothing here ever raises into
Hermes: every hook catches everything and logs at ``warning`` or below with
ids, counts and status codes only. Content is logged only at :data:`TRACE`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider, RecallStatus
from tools.registry import tool_error

from . import config as plugin_config
from . import tools, turns
from .breaker import CircuitBreaker
from .client import DaemonClient, DaemonError, DaemonUnavailable
from .config import PluginConfig
from .spool import Spool

log = logging.getLogger(__name__)

PROVIDER_NAME = "asphodel"
#: Shown on the "🧠 recalled N memories" status line.
PROVIDER_LABEL = "Asphodel"
#: The daemon major version this plugin was written for. ``initialize`` warns
#: when the health response's major differs.
DAEMON_MAJOR_VERSION = 0
#: Python's logging has no TRACE; this is the level content may be logged at.
TRACE = 5
#: Hermes' ``agent_context`` value for a user-facing agent. Only these ingest.
PRIMARY_CONTEXT = "primary"
#: Hermes' built-in memory flags that ``post_setup`` turns off and
#: ``initialize`` warns about.
BUILTIN_MEMORY_FLAGS = ("memory_enabled", "user_profile_enabled")
#: Hermes' ``utils.TRUTHY_STRINGS``: how it reads a string flag.
HERMES_TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})
#: Pending recalls kept per session, the daemon's ``PENDING_PER_SESSION``.
PENDING_RECALLS_PER_SESSION = 4
#: Reply prefix sent for conversation reranking, matching RERANK_CONTEXT_CHARS.
RERANK_CONTEXT_CHARS = 300
#: What the daemon's ``clean_query`` strips from the front of a prefetch
#: query: Hermes' Discord message-id note, then the ``[Name] `` speaker
#: prefix. The daemon does the cleaning; the plugin only uses this to skip a
#: query with nothing left.
_QUERY_NOISE = re.compile(r"^(?:\[Triggering message id: `[^\]\n]*\]\s*)?(?:\[[^\]\n]+\] )?")


@dataclass(frozen=True)
class Timeouts:
    """Client timeouts in seconds, one per hook.
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
        # The current turn's author, from on_turn_start.
        self._author_id: Optional[str] = None
        self._author_name: Optional[str] = None
        self._author_is_bot: bool = False
        # The (query, recall_id) of each prefetch whose turn
        # hasn't synced yet, oldest first. Hermes syncs on a background worker,
        # so the next turn's prefetch can come before this turn's sync
        # and sync_turn matches its own by the user text. The last prefetch
        # query is sent as the previous message and
        # dropped on memory_forget.
        self._pending_recalls: Dict[str, List[Tuple[str, str]]] = {}
        self._last_query: Dict[str, str] = {}
        self._last_reply: Dict[str, str] = {}
        # sync_turn has no turn id. Repeated text can name an older turn,
        # even after a clear, so only a first occurrence is reply-eligible.
        # Keep digests, not message bodies, for the provider lifetime.
        self._reply_queries: Dict[str, set[bytes]] = {}
        self._reply_eligible: Dict[str, bool] = {}
        self._context_versions: Dict[str, int] = {}
        self._context_lock = threading.Lock()
        self._last_injected: int = 0
        # A block fetched before the session id was known.
        self._pending_block_id: Optional[str] = None
        # Whether a PUT has set the bank up. Until one does, every bank
        # operation retries it first.
        self._bank_ready: bool = False

    # -- identity and availability -------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        """Config only: True when a URL resolves from
        ``ASPHODEL_URL`` or ``config.json`` under the active ``HERMES_HOME``.
        Never touches the network; a dead daemon never drops the provider."""
        try:
            return bool(plugin_config.load_config(self._resolve_home()).url)
        except Exception:
            return False

    def unavailable_reason(self) -> str:
        return (
            "No Asphodel daemon URL: set url in $HERMES_HOME/asphodel/config.json "
            f"or {plugin_config.URL_ENV_VAR} in the environment."
        )

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        """Loads config, makes one health probe (warning through
        ``warning_callback`` when the daemon isn't ready or its major version
        differs), ``PUT``s the bank with the configured identity, and warns if
        Hermes' built-in memory flags are still on. Never fails."""
        warn = kwargs.get("warning_callback")
        try:
            self._hermes_home = str(kwargs.get("hermes_home") or self._resolve_home())
            self._session_id = session_id or ""
            self._platform = kwargs.get("platform") or None
            self._agent_context = kwargs.get("agent_context") or PRIMARY_CONTEXT
            self._profile = kwargs.get("agent_identity") or "default"
            self.config = plugin_config.load_config(self._hermes_home)
            self.client = DaemonClient(self.config.url, token=self.config.token, breaker=self.breaker)
            self.spool = Spool(plugin_config.spool_dir(self._hermes_home))
            self.bank = self.config.bank or self._profile
            self._probe_health(warn)
            self._put_bank(warn)
            self._check_builtin_memory(warn)
        except Exception as error:
            log.warning("initialize failed: %s", type(error).__name__)

    def _probe_health(self, warn) -> None:
        url = self.config.url
        try:
            status, health = self.client.health(timeout=self.timeouts.health)
        except DaemonUnavailable:
            self._warn(warn, f"Asphodel: can't reach the daemon at {url}; memory is off until it's back.")
            return
        except DaemonError as error:
            self._warn(warn, f"Asphodel: the daemon's health check answered {error.status}.")
            return
        health = health if isinstance(health, dict) else {}
        if status != 200 or not health.get("ready", False):
            self._warn(warn, f"Asphodel: the daemon at {url} isn't ready yet (status {status}).")
        version = health.get("version")
        if isinstance(version, str):
            major = version.split(".", 1)[0]
            if major != str(DAEMON_MAJOR_VERSION):
                self._warn(
                    warn,
                    f"Asphodel: the daemon's version is {version}, but this plugin was written "
                    f"for {DAEMON_MAJOR_VERSION}.x.",
                )

    def _put_bank(self, warn) -> None:
        try:
            self._ensure_bank(self.timeouts.health)
        except DaemonUnavailable:
            # The health probe has already warned.
            log.debug("put bank: daemon unreachable")
        except DaemonError as error:
            if error.status == 503:
                # Starting or draining: the health probe has already warned,
                # and the next bank operation retries.
                log.debug("put bank: daemon answered 503")
            else:
                self._warn(warn, f"Asphodel: the daemon refused bank {self.bank} ({error.status}).")

    def _ensure_bank(self, timeout: float) -> None:
        """``PUT``s the bank unless one already succeeded. Raises what the
        client raises. Once the bank is set up a later 404 never recreates it:
        ``bank delete`` expects the plugin to be disabled first."""
        if self._bank_ready:
            return
        identity = {
            "owner_name": self.config.owner_name,
            "owner_platform_ids": list(self.config.owner_platform_ids),
            "assistant_name": self.config.assistant_name or self._profile,
            "timezone": self.config.timezone,
        }
        self.client.put_bank(self.bank, identity, timeout=timeout)
        self._bank_ready = True
        log.debug("bank %s set up", self.bank)

    def _check_builtin_memory(self, warn) -> None:
        """Warns for each flag Hermes reads as on. Hermes merges its defaults,
        which turn both on, and reads each with ``is_truthy_value(value,
        default=True)``, so a missing key or section and ``null`` are on."""
        try:
            from hermes_cli.config import load_config as load_hermes_config

            memory = (load_hermes_config() or {}).get("memory")
        except Exception as error:
            log.debug("could not read Hermes' config: %s", type(error).__name__)
            return
        if not isinstance(memory, dict):
            memory = {}
        for flag in BUILTIN_MEMORY_FLAGS:
            if _hermes_truthy(memory.get(flag), default=True):
                self._warn(
                    warn,
                    f"Asphodel: Hermes' built-in memory.{flag} is on, so two memories run at once. "
                    "Run `hermes memory setup` or set it to false.",
                )

    @staticmethod
    def _warn(callback, message: str) -> None:
        log.warning("%s", message)
        if callable(callback):
            try:
                callback(message)
            except Exception:
                pass

    def _resolve_home(self) -> str:
        if self._hermes_home:
            return self._hermes_home
        from hermes_constants import get_hermes_home

        return str(get_hermes_home())

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Records ``author_id``, ``author_name`` and ``author_is_bot`` for the
        owner check."""
        author_id = kwargs.get("author_id")
        self._author_id = str(author_id) if author_id not in (None, "") else None
        author_name = kwargs.get("author_name")
        self._author_name = str(author_name) if author_name else None
        self._author_is_bot = bool(kwargs.get("author_is_bot", False))

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
        recall id, last query, last reply and block mapping."""
        old_session = self._session_id or parent_session_id
        clear = reset or rewound or kwargs.get("reason") == "compression"
        try:
            if clear and old_session:
                self._pending_recalls.pop(old_session, None)
                self._clear_reply_context(old_session)
                self._pending_block_id = None
                if self.client is not None and self.bank:
                    try:
                        self._ensure_bank(self.timeouts.session_clear)
                        self.client.clear_session(self.bank, old_session, timeout=self.timeouts.session_clear)
                    except DaemonUnavailable:
                        log.debug("clear session: daemon unreachable")
                    except DaemonError as error:
                        log.warning("clear session: daemon answered %d", error.status)
        except Exception as error:
            log.warning("on_session_switch failed: %s", type(error).__name__)
        finally:
            self._session_id = new_session_id or self._session_id

    def shutdown(self) -> None:
        """Nothing to drain: the plugin owns no threads."""

    def backup_paths(self) -> List[str]:
        """``[]``: the data lives in the daemon's container."""
        return []

    def _clear_reply_context(self, session: str) -> None:
        with self._context_lock:
            self._last_query.pop(session, None)
            self._last_reply.pop(session, None)
            self._reply_eligible.pop(session, None)
            self._context_versions[session] = self._context_versions.get(session, 0) + 1
            # Do not forget seen queries: an old sync can arrive after a clear.

    # -- the reply path ------------------------------------------------------

    def system_prompt_block(self) -> str:
        """``GET /v1/banks/{bank}/system-prompt?session_id=...`` with the
        ``system_prompt`` budget and ``system_prompt_retries`` more attempts
        on a connection failure. Returns the block's text, or "" when the
        daemon can't be reached or the breaker is open."""
        try:
            if self.client is None or not self.bank:
                return ""
            session = self._session_id or None
            block = None
            for attempt in range(1 + max(0, self.timeouts.system_prompt_retries)):
                try:
                    deadline = time.monotonic() + self.timeouts.system_prompt
                    self._ensure_bank(self.timeouts.system_prompt)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise DaemonUnavailable("system prompt budget exhausted")
                    block = self.client.system_prompt(self.bank, session, timeout=remaining)
                    break
                except DaemonUnavailable as error:
                    log.debug("system prompt attempt %d: daemon unreachable (%s)", attempt + 1, error)
                    if not self.breaker.allow():
                        break
            if not isinstance(block, dict):
                log.warning("system prompt: daemon unreachable")
                return ""
            text = block.get("text") or ""
            block_id = block.get("id")
            self._pending_block_id = block_id if not session and block_id else None
            log.debug("system prompt: block %s", block_id)
            log.log(TRACE, "system prompt block: %s", text)
            return text
        except DaemonError as error:
            log.warning("system prompt: daemon answered %d", error.status)
        except Exception as error:
            log.warning("system_prompt_block failed: %s", type(error).__name__)
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """``POST /v1/banks/{bank}/prefetch`` with the query, the session's
        last prefetch query as ``previous_query``, its bounded assistant reply
        as ``previous_reply`` only when attribution is unambiguous and, once,
        a pending block id. Stores the ``recall_id`` for ``sync_turn`` and the injected count
        for ``recall_status``. The query goes as Hermes gave it, for the daemon
        to clean, but one that cleans to nothing isn't sent. "" then and on
        any failure."""
        self._last_injected = 0
        if not isinstance(query, str) or not _QUERY_NOISE.sub("", query.lstrip(), count=1).strip():
            log.debug("prefetch: skipped, the query cleans to nothing")
            return ""
        try:
            deadline = time.monotonic() + self.timeouts.prefetch
            if self.client is None or not self.bank:
                return ""
            session = session_id or self._session_id
            if not self._session_id:
                self._session_id = session
            request: Dict[str, Any] = {"session_id": session, "query": query}
            with self._context_lock:
                previous = self._last_query.get(session)
                if previous:
                    request["previous_query"] = previous
                    reply = self._last_reply.get(session)
                    if reply:
                        request["previous_reply"] = reply
                version = self._context_versions.get(session, 0)
                seen = self._reply_queries.setdefault(session, set())
                fingerprint = hashlib.sha256(query.encode("utf-8", errors="surrogatepass")).digest()
                eligible = fingerprint not in seen
                # Record attempts too: a turn may sync despite a failed prefetch.
                seen.add(fingerprint)
                if not eligible and query == previous:
                    # Even if this request fails, a later sync with this text
                    # cannot be attributed to the last successful turn.
                    self._reply_eligible[session] = False
            block_id = self._pending_block_id
            if block_id:
                request["block_id"] = block_id
            log.log(TRACE, "prefetch query: %s", query)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DaemonUnavailable("prefetch budget exhausted")
            self._ensure_bank(remaining)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DaemonUnavailable("prefetch budget exhausted")
            result = self.client.prefetch(self.bank, request, timeout=remaining)
        except DaemonUnavailable as error:
            log.warning("prefetch: daemon unreachable (%s)", error)
            return ""
        except DaemonError as error:
            log.warning("prefetch: daemon answered %d", error.status)
            return ""
        except Exception as error:
            log.warning("prefetch failed: %s", type(error).__name__)
            return ""
        if block_id and self._pending_block_id == block_id:
            self._pending_block_id = None
        with self._context_lock:
            # A clear during the HTTP request must not restore local context.
            if self._context_versions.get(session, 0) == version:
                self._last_query[session] = query
                self._last_reply.pop(session, None)
                self._reply_eligible[session] = eligible
        result = result if isinstance(result, dict) else {}
        recall_id = result.get("recall_id")
        if recall_id:
            pending = self._pending_recalls.setdefault(session, [])
            pending.append((query, str(recall_id)))
            del pending[:-PENDING_RECALLS_PER_SESSION]
        injected = result.get("injected")
        self._last_injected = len(injected) if isinstance(injected, list) else 0
        log.debug("prefetch: recall %s injected %d", recall_id, self._last_injected)
        return result.get("text") or ""

    def recall_status(self) -> Optional[RecallStatus]:
        """The last prefetch's injected count; ``None`` when it injected
        nothing or failed."""
        if self._last_injected <= 0:
            return None
        return RecallStatus(provider_label=PROVIDER_LABEL, count=self._last_injected)

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
        """Keeps the bounded reply prefix only for an unambiguous latest query.
        Ingests the turn when ``agent_context`` is ``primary`` and
        ``ingest`` is on. Builds the ``Turn`` body with :func:`build_turn`,
        ``POST``s it, and spools it on a connection failure or 5xx. After a
        2xx it replays the spool."""
        try:
            session = session_id or self._session_id
            # Never attach a late reply to a newer query. Text alone cannot
            # distinguish repeated turns, including ones invalidated by a clear.
            # This local capture is independent of ingestion and takes no I/O lock.
            with self._context_lock:
                if (
                    self._last_query.get(session) == user_content
                    and self._reply_eligible.get(session, False)
                ):
                    self._last_reply[session] = (assistant_content or "")[:RERANK_CONTEXT_CHARS]
            if self.client is None or self.config is None or not self.bank:
                return
            if self._agent_context != PRIMARY_CONTEXT or not self.config.ingest:
                return
            turn = self.build_turn(
                user_content,
                assistant_content,
                session_id=session,
                messages=messages,
                turn_author=turn_author,
            )
            # The recall id is echoed once, whatever happens to the turn.
            self._consume_recall(session, user_content)
            log.log(TRACE, "turn user text: %s", turn["user_text"])
            try:
                self._ensure_bank(self.timeouts.ingest)
            except (DaemonUnavailable, DaemonError) as error:
                # Kept until the bank exists, rather than refused as unknown.
                log.warning("ingest: bank %s isn't set up (%s); spooling the turn", self.bank, _reason(error))
                self._spool_turn(turn)
                return
            try:
                self.client.ingest_turn(self.bank, turn, timeout=self.timeouts.ingest)
            except DaemonUnavailable as error:
                log.warning("ingest: daemon unreachable (%s); spooling the turn", error)
                self._spool_turn(turn)
                return
            except DaemonError as error:
                if error.status >= 500:
                    log.warning("ingest: daemon answered %d; spooling the turn", error.status)
                    self._spool_turn(turn)
                else:
                    log.warning("ingest: daemon answered %d; dropping the turn", error.status)
                return
            log.debug("ingest: turn delivered (forget_requested=%s)", turn["forget_requested"])
            if self.spool is not None:
                self.spool.replay(self._replay_one)
        except Exception as error:
            log.warning("sync_turn failed: %s", type(error).__name__)

    def _recall_index(self, session: str, user_content: str) -> Optional[int]:
        """The session's oldest pending recall whose prefetch query is this
        turn's user text. Hermes prefetches and syncs the same text for a text
        turn; a turn that matches none (no prefetch, or a multimodal turn)
        echoes nothing, which the daemon treats as changing nothing."""
        for index, (query, _) in enumerate(self._pending_recalls.get(session, ())):
            if query == user_content:
                return index
        return None

    def _consume_recall(self, session: str, user_content: str) -> None:
        """Drops the matched recall and every older one: Hermes syncs turns in
        order, so those belong to turns that were interrupted and never sync."""
        index = self._recall_index(session, user_content)
        if index is not None:
            del self._pending_recalls[session][: index + 1]

    def _spool_turn(self, turn: Dict[str, Any]) -> None:
        if self.spool is not None:
            self.spool.write(turn)

    def _replay_one(self, turn: Dict[str, Any]) -> bool:
        """``Spool.replay``'s sender: a 4xx drops the turn, a 5xx stops the
        replay like a connection failure."""
        try:
            self.client.ingest_turn(self.bank, turn, timeout=self.timeouts.ingest)
        except DaemonError as error:
            if error.status >= 500:
                raise DaemonUnavailable(f"daemon answered {error.status}") from None
            log.warning("replay: daemon answered %d; dropping a spooled turn", error.status)
            return False
        return True

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
        session = session_id or self._session_id
        index = self._recall_index(session, user_content)
        return {
            "session_id": session,
            "message_at": turns.message_at(messages, self._wall_clock()),
            "timezone": self._timezone(),
            "user_text": turns.strip_backfill(user_content or ""),
            "assistant_text": assistant_content or "",
            "author": _author(turn_author),
            "platform": self._platform,
            "recall_id": None if index is None else self._pending_recalls[session][index][1],
            "forget_requested": turns.forget_requested(messages),
        }

    def _timezone(self) -> Optional[str]:
        try:
            from hermes_time import get_timezone_name

            name = get_timezone_name()
        except Exception:
            name = None
        if name:
            return name
        return self.config.timezone if self.config is not None else None

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(tools.TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatches the four tools. Owner-only tools return ``tool_error``
        without a request on a non-owner's turn. ``memory_recall`` returns the
        daemon's ``results`` list as JSON. A daemon error or an unreachable
        daemon returns ``tool_error``; ``memory_forget`` drops the session's
        last prefetch query and assistant reply on success."""
        try:
            return self._handle_tool_call(tool_name, args)
        except DaemonUnavailable:
            log.warning("%s: daemon unreachable", tool_name)
            return tool_error("Asphodel's daemon can't be reached right now; try again later.")
        except DaemonError as error:
            log.warning("%s: daemon answered %d", tool_name, error.status)
            return tool_error(f"Asphodel answered {error.status}: {error.message}")
        except Exception as error:
            log.warning("%s failed: %s", tool_name, type(error).__name__)
            return tool_error(f"{tool_name} failed inside the Asphodel plugin.")

    def _handle_tool_call(self, tool_name: str, args: Any) -> str:
        if tool_name not in (tools.RECALL_TOOL, *tools.OWNER_ONLY_TOOLS):
            return tool_error(f"Asphodel has no tool named {tool_name}.")
        if not isinstance(args, dict):
            return tool_error(f"{tool_name} takes an object of arguments.")
        if tool_name in tools.OWNER_ONLY_TOOLS and not self.is_owner_turn():
            return tool_error(f"Only the owner may call {tool_name}.")
        if self.client is None or not self.bank:
            return tool_error("Asphodel isn't initialised.")
        session = self._session_id or None
        timeout = self.timeouts.tool

        if tool_name == tools.RECALL_TOOL:
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return tool_error("memory_recall needs a non-empty query.")
            request = {key: args[key] for key in _RECALL_ARGUMENTS if args.get(key) is not None}
            if session:
                request["session_id"] = session
            log.log(TRACE, "recall query: %s", query)
            self._ensure_bank(timeout)
            result = self.client.recall(self.bank, request, timeout=timeout)
            results = result.get("results", []) if isinstance(result, dict) else []
            log.debug("recall: %d results", len(results))
            return json.dumps(results, ensure_ascii=False)

        ids = args.get("ids")
        if (
            not isinstance(ids, list)
            or not ids
            or len(ids) > tools.MAX_IDS
            or not all(isinstance(item, str) and item for item in ids)
        ):
            return tool_error(f"{tool_name} needs ids: a list of 1 to {tools.MAX_IDS} memory ids.")
        self._ensure_bank(timeout)
        if tool_name == tools.FORGET_TOOL:
            result = self.client.forget(self.bank, ids, session, timeout=timeout)
            # The forget request isn't sent again as the next previous query.
            if session:
                self._clear_reply_context(session)
        elif tool_name == tools.KEEP_TOOL:
            result = self.client.keep(self.bank, ids, timeout=timeout)
        else:
            result = self.client.unkeep(self.bank, ids, timeout=timeout)
        log.debug("%s: %d ids", tool_name, len(ids))
        return json.dumps(result, ensure_ascii=False)

    def is_owner_turn(self) -> bool:
        return tools.is_owner(
            author_id=self._author_id,
            author_is_bot=self._author_is_bot,
            platform=self._platform,
            owner_platform_ids=self.config.owner_platform_ids if self.config is not None else [],
            agent_context=self._agent_context,
        )

    # -- setup ---------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return plugin_config.config_schema()

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Writes the non-secret values to ``config.json``. The token never
        goes there: Hermes writes it to ``.env`` as ``ASPHODEL_TOKEN``."""
        secrets = {field["key"] for field in plugin_config.config_schema() if field.get("secret")}
        cleaned: Dict[str, Any] = {}
        for key, value in values.items():
            if key in secrets or value is None:
                continue
            if key == "owner_platform_ids":
                cleaned[key] = plugin_config.parse_platform_ids(value)
            elif key == "ingest":
                cleaned[key] = plugin_config.parse_bool(value, default=True)
            elif isinstance(value, str):
                cleaned[key] = value.strip() or None
            else:
                cleaned[key] = value
        plugin_config.save_config(hermes_home, cleaned)

    def post_setup(self, hermes_home: str, config: Dict[str, Any], *, prompt: Callable[[str], str] = input) -> None:
        """``hermes memory setup``: prompts for each schema field (empty keeps
        the default), writes ``config.json``, sets ``memory.provider`` to
        ``asphodel`` and ``memory.memory_enabled`` and
        ``memory.user_profile_enabled`` to false, and saves Hermes' config
        through ``hermes_cli.config.save_config``."""
        from hermes_cli.config import save_config as save_hermes_config

        values: Dict[str, Any] = {}
        for field in plugin_config.config_schema():
            if field.get("secret"):
                continue
            default = field.get("default")
            hint = f" [{str(default).lower() if isinstance(default, bool) else default}]" if default is not None else ""
            answer = prompt(f"{field['key']}: {field['description']}{hint} ").strip()
            if answer:
                values[field["key"]] = answer
        self.save_config(values, hermes_home)
        memory = config.get("memory")
        if not isinstance(memory, dict):
            memory = config["memory"] = {}
        memory["provider"] = PROVIDER_NAME
        for flag in BUILTIN_MEMORY_FLAGS:
            memory[flag] = False
        save_hermes_config(config)
        print(f"Set {plugin_config.TOKEN_ENV_VAR} in $HERMES_HOME/.env if the daemon needs a bearer token.")


def _hermes_truthy(value: Any, *, default: bool) -> bool:
    """Hermes' ``utils.is_truthy_value``."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in HERMES_TRUTHY_STRINGS
    return bool(value)


def _reason(error: Exception) -> str:
    """A log-safe reason: the status code, or the kind of connection failure."""
    if isinstance(error, DaemonError):
        return f"daemon answered {error.status}"
    return f"daemon unreachable: {error}"


_RECALL_ARGUMENTS = ("query", "from", "to", "on", "phase", "kinds", "entity", "limit")


def _author(turn_author: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Hermes' ``turn_author`` as the daemon's ``TurnAuthor``."""
    if not isinstance(turn_author, dict) or turn_author.get("id") in (None, ""):
        return None
    name = turn_author.get("name")
    return {
        "id": str(turn_author["id"]),
        "name": str(name) if name else None,
        "is_bot": bool(turn_author.get("is_bot", False)),
    }
