"""Stand-ins for the Hermes modules the plugin imports, so the tests run
without a Hermes checkout. The shapes follow hermes-agent at ``be5e9f7``:
``agent/memory_provider.py`` (the base class, abstract members and
``spawn_context_thread``), ``tools/registry.py`` (``tool_error``),
``hermes_time``, ``hermes_constants``, ``hermes_cli.config`` and
``plugins/memory/config_schema.py``."""

from __future__ import annotations

import contextvars
import json
import os
import sys
import threading
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


# -- agent.memory_provider ------------------------------------------------------

def ctx_bound(fn: Callable[..., Any]) -> Callable[..., Any]:
    ctx = contextvars.copy_context()
    return lambda *args, **kwargs: ctx.run(fn, *args, **kwargs)


def spawn_context_thread(target, *, name, daemon=True, args=(), kwargs=None) -> threading.Thread:
    return threading.Thread(target=ctx_bound(target), args=args, kwargs=kwargs, name=name, daemon=daemon)


@dataclass(frozen=True)
class RecallStatus:
    provider_label: str
    count: int
    glyph: str = "🧠"


class MemoryProvider(ABC):
    pre_compress_checkpoint_api_version = 1

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None: ...

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]: ...

    def unavailable_reason(self) -> str:
        return ""

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None: ...

    def recall_status(self) -> Optional[RecallStatus]:
        return None

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None, turn_author=None) -> None: ...

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None: ...

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None: ...

    def identity_signature(self) -> Dict[str, Any]:
        return {}

    def on_session_end(self, messages) -> None: ...

    def on_session_switch(self, new_session_id, *, parent_session_id="", reset=False, rewound=False, **kwargs) -> None: ...

    def on_pre_compress(self, messages) -> str:
        return ""

    def on_delegation(self, task, result, *, child_session_id="", **kwargs) -> None: ...

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None: ...

    def on_memory_write(self, action, target, content, metadata=None) -> None: ...

    def backup_paths(self) -> List[str]:
        return []


# -- tools.registry -------------------------------------------------------------

def tool_error(message, **extra) -> str:
    return json.dumps({"error": str(message)[:2000], **extra}, ensure_ascii=False)


def tool_result(data=None, **kwargs) -> str:
    return json.dumps(data if data is not None else kwargs, ensure_ascii=False)


# -- hermes_time, hermes_constants, hermes_cli.config -------------------------------

class HermesState:
    """What the stubs read and record. Tests reset it through the fixture."""

    timezone_name: str = "Pacific/Auckland"
    config: Dict[str, Any] = {}
    saved: List[Dict[str, Any]] = []

    @classmethod
    def reset(cls) -> None:
        cls.timezone_name = "Pacific/Auckland"
        cls.config = {"memory": {}}
        cls.saved = []


def get_timezone_name() -> str:
    return HermesState.timezone_name


def get_hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def load_config() -> Dict[str, Any]:
    return HermesState.config


def save_config(config: Dict[str, Any], *args, **kwargs) -> None:
    HermesState.saved.append(config)


# -- plugins.memory.config_schema ---------------------------------------------------

KIND_TEXT = "text"
KIND_SELECT = "select"
KIND_SECRET = "secret"
KIND_BOOL = "bool"
KIND_NUMBER = "number"
KIND_JSON = "json"
STORAGE_FLAT_JSON = "flat_json"


@dataclass(frozen=True)
class ProviderFieldOption:
    value: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class ProviderField:
    key: str
    label: str
    kind: str = KIND_TEXT
    default: str = ""
    description: str = ""
    placeholder: str = ""
    options: tuple = ()
    env_key: Optional[str] = None
    aliases: tuple = ()
    env_fallbacks: tuple = ()
    inline: bool = False
    group: str = ""
    info: str = ""
    scope: str = "host"

    @property
    def is_secret(self) -> bool:
        return self.kind == KIND_SECRET


@dataclass(frozen=True)
class ProviderConfigSchema:
    name: str
    label: str
    storage: str = STORAGE_FLAT_JSON
    docs_url: str = ""
    fields: tuple = dataclass_field(default_factory=tuple)


def install() -> None:
    """Puts the stubs in ``sys.modules``. Idempotent."""
    if "agent.memory_provider" in sys.modules and getattr(sys.modules["agent.memory_provider"], "_asphodel_stub", False):
        return
    agent = _module("agent")
    memory_provider = _module("agent.memory_provider")
    memory_provider._asphodel_stub = True
    memory_provider.MemoryProvider = MemoryProvider
    memory_provider.RecallStatus = RecallStatus
    memory_provider.spawn_context_thread = spawn_context_thread
    memory_provider.ctx_bound = ctx_bound
    agent.memory_provider = memory_provider

    tools = _module("tools")
    registry = _module("tools.registry")
    registry.tool_error = tool_error
    registry.tool_result = tool_result
    tools.registry = registry

    hermes_time = _module("hermes_time")
    hermes_time.get_timezone_name = get_timezone_name

    hermes_constants = _module("hermes_constants")
    hermes_constants.get_hermes_home = get_hermes_home

    hermes_cli = _module("hermes_cli")
    config = _module("hermes_cli.config")
    config.load_config = load_config
    config.save_config = save_config
    hermes_cli.config = config

    plugins = _module("plugins")
    memory = _module("plugins.memory")
    schema = _module("plugins.memory.config_schema")
    for name in (
        "KIND_TEXT", "KIND_SELECT", "KIND_SECRET", "KIND_BOOL", "KIND_NUMBER", "KIND_JSON",
        "STORAGE_FLAT_JSON", "ProviderFieldOption", "ProviderField", "ProviderConfigSchema",
    ):
        setattr(schema, name, globals()[name])
    memory.config_schema = schema
    plugins.memory = memory
    HermesState.reset()
