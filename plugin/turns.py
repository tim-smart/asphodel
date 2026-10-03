"""Shaping a Hermes turn into the daemon's ``Turn`` body: the backfill strip,
the ``memory_forget`` scan and the message time."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

#: The gateway puts a channel's recent history and this marker in front of
#: the user text on backfill. Everything before it is other people's words.
NEW_MESSAGE_MARKER = "[New message]"
FORGET_TOOL = "memory_forget"


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
