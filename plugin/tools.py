"""The four model tools (TIM-94, decision 9) and the owner check (decision 1).

``memory_recall`` is open to every speaker. ``memory_forget``, ``memory_keep``
and ``memory_unkeep`` are owner-only: on a non-owner's turn, a bot's turn or
in any cron run they return ``tool_error`` and make no request.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

RECALL_TOOL = "memory_recall"
FORGET_TOOL = "memory_forget"
KEEP_TOOL = "memory_keep"
UNKEEP_TOOL = "memory_unkeep"
OWNER_ONLY_TOOLS = frozenset({FORGET_TOOL, KEEP_TOOL, UNKEEP_TOOL})
#: ``ids`` arrays are capped by the daemon too (``keep::MAX_IDS``).
MAX_IDS = 50
RECALL_LIMIT_DEFAULT = 10
RECALL_LIMIT_MAX = 30

_IDS_PARAMETER = {
    "type": "array",
    "items": {"type": "string"},
    "maxItems": MAX_IDS,
    "description": "Memory ids, as returned by memory_recall.",
}

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": RECALL_TOOL,
        "description": (
            "Search Asphodel's long-term memory for what the user has said or done. "
            "Use it for anything from an earlier conversation, and with phase "
            "'upcoming' for what is coming up after the agenda in the system prompt was built."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "from": {"type": "string", "format": "date-time"},
                "to": {"type": "string", "format": "date-time"},
                "on": {
                    "type": "string",
                    "enum": ["happened", "said"],
                    "description": "Match the date range against when it happened (default) or when it was said.",
                },
                "phase": {"type": "string", "enum": ["upcoming", "past", "current", "any"]},
                "kinds": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["fact", "event", "state", "task", "recurring"]},
                },
                "entity": {"type": "string", "description": "A person, place or thing, by name or alias."},
                "limit": {"type": "integer", "minimum": 1, "maximum": RECALL_LIMIT_MAX, "default": RECALL_LIMIT_DEFAULT},
            },
            "required": ["query"],
        },
    },
    {
        "name": FORGET_TOOL,
        "description": (
            "Irreversibly erase memories and every version of them. Only when the owner "
            "explicitly asks to forget something; only the owner may call it."
        ),
        "parameters": {"type": "object", "properties": {"ids": _IDS_PARAMETER}, "required": ["ids"]},
    },
    {
        "name": KEEP_TOOL,
        "description": "Mark memories to keep so they never fade. Only the owner may call it.",
        "parameters": {"type": "object", "properties": {"ids": _IDS_PARAMETER}, "required": ["ids"]},
    },
    {
        "name": UNKEEP_TOOL,
        "description": "Undo memory_keep, returning memories to the significance extraction gave them. Only the owner may call it.",
        "parameters": {"type": "object", "properties": {"ids": _IDS_PARAMETER}, "required": ["ids"]},
    },
]


def is_owner(
    *,
    author_id: Optional[str],
    author_is_bot: bool,
    platform: Optional[str],
    owner_platform_ids: List[str],
    agent_context: Optional[str],
) -> bool:
    """A turn with no author (CLI, TUI, Hermes UI) is the owner's. With an
    author, the owner is matched by speaker id ``<platform>:<author_id>``
    against the configured owner ids. Bots and cron runs are never the
    owner."""
    if agent_context == "cron" or author_is_bot:
        return False
    if not author_id:
        return True
    if not platform:
        return False
    return f"{platform}:{author_id}" in owner_platform_ids
