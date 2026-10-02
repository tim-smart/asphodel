"""Plugin configuration (TIM-94, decision 7).

The file is ``$HERMES_HOME/asphodel/config.json``. ``ASPHODEL_URL`` in the
environment overrides ``url`` (decision 2), and the token comes only from
``ASPHODEL_TOKEN``, which Hermes writes to ``.env`` from the ``secret`` field.
"""

from __future__ import annotations

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
    raise NotImplementedError


def save_config(hermes_home: str | os.PathLike, values: Dict[str, Any]) -> Path:
    """Merges ``values`` into ``config.json`` (creating ``asphodel/``) and
    writes it by rename. Returns the path written."""
    raise NotImplementedError


def config_schema() -> List[Dict[str, Any]]:
    """``get_config_schema()``: the fields ``hermes memory setup`` prompts for
    and the dashboard renders. The token is ``secret`` with ``env_var``
    ``ASPHODEL_TOKEN``; ``owner_platform_ids`` is entered comma-separated."""
    raise NotImplementedError
