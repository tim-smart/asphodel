"""Plugin configuration (TIM-94, decision 7).

The file is ``$HERMES_HOME/asphodel/config.json``. ``ASPHODEL_URL`` in the
environment overrides ``url`` (decision 2), and the token comes only from
``ASPHODEL_TOKEN``, which Hermes writes to ``.env`` from the ``secret`` field.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Where the daemon listens by default (TIM-94, decision 2).
DEFAULT_URL = "http://127.0.0.1:7720"
#: Environment variables the plugin reads.
URL_ENV_VAR = "ASPHODEL_URL"
TOKEN_ENV_VAR = "ASPHODEL_TOKEN"
#: Relative to ``$HERMES_HOME``.
CONFIG_RELATIVE_PATH = Path("asphodel") / "config.json"
SPOOL_RELATIVE_PATH = Path("asphodel") / "spool"


@dataclass
class PluginConfig:
    """The fields of ``config.json``. ``None`` means "not set, use the
    runtime default": the bank and assistant name default to the Hermes
    profile name, and the timezone to ``hermes_time``."""

    url: str = DEFAULT_URL
    token: Optional[str] = None
    bank: Optional[str] = None
    owner_name: Optional[str] = None
    #: Speaker ids of the owner, ``<platform>:<id>`` (TIM-94, decision 1).
    owner_platform_ids: List[str] = field(default_factory=list)
    assistant_name: Optional[str] = None
    timezone: Optional[str] = None
    ingest: bool = True


def config_path(hermes_home: str | os.PathLike) -> Path:
    return Path(hermes_home) / CONFIG_RELATIVE_PATH


def spool_dir(hermes_home: str | os.PathLike) -> Path:
    return Path(hermes_home) / SPOOL_RELATIVE_PATH


def load_config(hermes_home: str | os.PathLike, environ: Optional[Dict[str, str]] = None) -> PluginConfig:
    """Reads ``config.json`` (absent or unreadable means defaults), then
    applies ``ASPHODEL_URL`` over ``url`` and ``ASPHODEL_TOKEN`` as the token.
    Never raises."""
    env = os.environ if environ is None else environ
    data = _read(config_path(hermes_home))
    url = data.get("url", DEFAULT_URL)
    config = PluginConfig(
        url=url.strip() if isinstance(url, str) else DEFAULT_URL,
        bank=_text(data.get("bank")),
        owner_name=_text(data.get("owner_name")),
        owner_platform_ids=parse_platform_ids(data.get("owner_platform_ids")),
        assistant_name=_text(data.get("assistant_name")),
        timezone=_text(data.get("timezone")),
        ingest=parse_bool(data.get("ingest"), default=True),
    )
    env_url = (env.get(URL_ENV_VAR) or "").strip()
    if env_url:
        config.url = env_url
    config.token = (env.get(TOKEN_ENV_VAR) or "").strip() or None
    return config


def save_config(hermes_home: str | os.PathLike, values: Dict[str, Any]) -> Path:
    """Merges ``values`` into ``config.json`` (creating ``asphodel/``) and
    writes it by rename. Returns the path written."""
    path = config_path(hermes_home)
    data = _read(path)
    data.update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        partial.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()
    return path


def config_schema() -> List[Dict[str, Any]]:
    """``get_config_schema()``: the fields ``hermes memory setup`` prompts for
    and the dashboard renders. The token is ``secret`` with ``env_var``
    ``ASPHODEL_TOKEN``; ``owner_platform_ids`` is entered comma-separated."""
    return [
        {
            "key": "url",
            "description": "Daemon URL: http://host:port or unix:/path. ASPHODEL_URL in the environment overrides it.",
            "default": DEFAULT_URL,
        },
        {
            "key": "token",
            "description": "Bearer token, required only when the daemon listens off loopback.",
            "secret": True,
            "env_var": TOKEN_ENV_VAR,
        },
        {"key": "bank", "description": "Bank. One bank per Hermes profile; defaults to the profile name."},
        {"key": "owner_name", "description": "Owner's name."},
        {
            "key": "owner_platform_ids",
            "description": "Owner's platform ids, comma-separated speaker ids such as discord:1234.",
        },
        {"key": "assistant_name", "description": "Assistant's name. Defaults to the profile name."},
        {"key": "timezone", "description": "Default timezone, an IANA name. Defaults to Hermes' configured timezone."},
        {
            "key": "ingest",
            "description": "Ingest turns. Off means recall and injection only; nothing new is remembered.",
            "type": "boolean",
            "default": True,
        },
    ]


def parse_platform_ids(value: Any) -> List[str]:
    """A list of speaker ids, or the comma-separated form setup collects."""
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def parse_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
    return default


def _text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _read(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
