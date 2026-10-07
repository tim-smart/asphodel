"""Shaping a Hermes turn into the daemon's ``Turn`` body: the backfill strip,
the ``memory_forget`` scan and the message time."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

#: The gateway puts a channel's recent history and this marker in front of
#: the user text on backfill. Everything before it is other people's words.
NEW_MESSAGE_MARKER = "[New message]"
FORGET_TOOL = "memory_forget"
#: Hermes session sources that aren't a human conversation: kanban workers,
#: delegated subagents, tool integrations and one-shot runs.
NON_HUMAN_SOURCES = frozenset({"kanban", "subagent", "tool", "oneshot"})
#: The one user-row ``display_kind`` a human wrote: a typed ``/steer``. Every
#: other kind marks a row Hermes made itself, such as an async delegation
#: result or a process completion.
STEER_DISPLAY_KIND = "steer"


def strip_backfill(user_text: str) -> str:
    """Drops everything up to and including the last ``[New message]``
    marker and the whitespace after it. The ``[Name] `` speaker prefix of a
    shared thread stays. Text without the marker is returned unchanged."""
    index = user_text.rfind(NEW_MESSAGE_MARKER)
    if index < 0:
        return user_text
    return user_text[index + len(NEW_MESSAGE_MARKER):].lstrip()


def current_turn_rows(messages: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The rows of the turn just completed: from the last ``user`` row to
    the end of the transcript."""
    if not messages:
        return []
    for index in range(len(messages) - 1, -1, -1):
        row = messages[index]
        if isinstance(row, dict) and row.get("role") == "user":
            return list(messages[index:])
    return []


def human_display_kind(kind: Optional[str]) -> bool:
    return not kind or kind == STEER_DISPLAY_KIND


def human_turn(messages: Optional[List[Dict[str, Any]]]) -> bool:
    """False when the current turn's user row is one Hermes made itself.
    A turn without a transcript is taken as human."""
    rows = current_turn_rows(messages)
    return not rows or human_display_kind(rows[0].get("display_kind"))


def forget_requested(messages: Optional[List[Dict[str, Any]]]) -> bool:
    """True when an assistant row of the current turn carries a
    ``memory_forget`` tool call. Earlier turns don't count, and no other tool
    does."""
    for row in current_turn_rows(messages):
        if row.get("role") != "assistant":
            continue
        for call in row.get("tool_calls") or ():
            if isinstance(call, dict) and _tool_call_name(call) == FORGET_TOOL:
                return True
    return False


def _tool_call_name(call: Dict[str, Any]) -> Optional[str]:
    function = call.get("function")
    if isinstance(function, dict):
        return function.get("name")
    return call.get("name")


def message_at(messages: Optional[List[Dict[str, Any]]], fallback_epoch: float) -> str:
    """The current turn's user message time as an RFC 3339 UTC string, from
    the epoch float Hermes stamps on the row; ``fallback_epoch`` when the
    transcript has none."""
    rows = current_turn_rows(messages)
    stamp = rows[0].get("timestamp") if rows else None
    if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
        try:
            return epoch_to_rfc3339(float(stamp))
        except (OverflowError, OSError, ValueError):
            pass
    return epoch_to_rfc3339(fallback_epoch)


def epoch_to_rfc3339(epoch: float) -> str:
    moment = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")
