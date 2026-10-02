"""Shaping a Hermes turn into the daemon's ``Turn`` body: the backfill strip
(TIM-96 amendment to TIM-94), the ``memory_forget`` scan (ADR 0010) and the
message time (TIM-88)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: The gateway puts a channel's recent history and this marker in front of
#: the user text on backfill. Everything before it is other people's words.
NEW_MESSAGE_MARKER = "[New message]"
FORGET_TOOL = "memory_forget"


def strip_backfill(user_text: str) -> str:
    """Drops everything up to and including the last ``[New message]``
    marker and the whitespace after it. The ``[Name] `` speaker prefix of a
    shared thread stays. Text without the marker is returned unchanged."""
    raise NotImplementedError


def current_turn_rows(messages: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The rows of the turn just completed: from the last ``user`` row to
    the end of the transcript."""
    raise NotImplementedError


def forget_requested(messages: Optional[List[Dict[str, Any]]]) -> bool:
    """True when an assistant row of the current turn carries a
    ``memory_forget`` tool call. Earlier turns don't count, and no other tool
    does."""
    raise NotImplementedError


def message_at(messages: Optional[List[Dict[str, Any]]], fallback_epoch: float) -> str:
    """The current turn's user message time as an RFC 3339 UTC string, from
    the epoch float Hermes stamps on the row; ``fallback_epoch`` when the
    transcript has none."""
    raise NotImplementedError


def epoch_to_rfc3339(epoch: float) -> str:
    raise NotImplementedError
