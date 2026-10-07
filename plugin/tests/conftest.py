"""Fixtures: the Hermes stubs, the plugin loaded the way Hermes loads it
(from its directory, under a synthetic package name), a fake daemon, a fake
monotonic clock and a provider factory with short budgets."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
PLUGIN_DIR = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))

import hermes_stubs  # noqa: E402
from fake_daemon import FakeDaemon  # noqa: E402

hermes_stubs.install()

PACKAGE_NAME = "asphodel_plugin"


def load_plugin():
    """Imports ``plugin/`` as ``asphodel_plugin``, as Hermes' loader does
    under its own synthetic name, so nothing in the package may depend on the
    directory's name."""
    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    for name in ("breaker", "client", "config", "provider", "spool", "tools", "turns"):
        importlib.import_module(f"{PACKAGE_NAME}.{name}")
    return module


plugin = load_plugin()

# Budgets short enough that a timeout test takes well under a second.
FAST = plugin.provider.Timeouts(
    health=0.5, prefetch=0.4, system_prompt=0.4, system_prompt_retries=1, ingest=0.5, tool=0.5, session_clear=0.5
)

SESSION = "sess-0001"
PROFILE = "tim"
OWNER_IDS = ["discord:111"]


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("ASPHODEL_URL", raising=False)
    monkeypatch.delenv("ASPHODEL_TOKEN", raising=False)
    for name in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(name, raising=False)
    hermes_stubs.HermesState.reset()
    return home


@pytest.fixture
def hermes():
    """The stubbed Hermes state: timezone, config and what was saved."""
    return hermes_stubs.HermesState


@pytest.fixture
def daemon():
    daemon = FakeDaemon().start()
    yield daemon
    daemon.stop()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def warnings():
    return []


def write_config(hermes_home: Path, **values) -> Path:
    path = hermes_home / "asphodel" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values))
    return path


@pytest.fixture
def make_provider(hermes_home, daemon, clock, warnings):
    """``make_provider(**config)`` writes ``config.json`` pointing at the fake
    daemon, builds a provider with short budgets, and initialises it as a
    primary Discord agent for profile ``tim`` whose owner is ``discord:111``.
    Pass ``initialize=False`` to stop before ``initialize``, or override any
    ``initialize`` kwarg through ``init=dict(...)``."""

    def make(*, initialize=True, init=None, timeouts=FAST, url=None, **config):
        values = {"url": daemon.url if url is None else url, "owner_platform_ids": OWNER_IDS, "owner_name": "Tim"}
        values.update(config)
        write_config(hermes_home, **values)
        provider = plugin.AsphodelMemoryProvider(timeouts=timeouts, clock=clock)
        if initialize:
            kwargs = dict(
                hermes_home=str(hermes_home),
                platform="discord",
                agent_context="primary",
                agent_identity=PROFILE,
                warning_callback=warnings.append,
            )
            kwargs.update(init or {})
            provider.initialize(kwargs.pop("session_id", SESSION), **kwargs)
        return provider

    return make


def transcript(
    user_text: str, assistant_text: str, *, epoch: float = 1759371557.622, tool_calls=None, earlier=(), display_kind=None
):
    """An OpenAI-style transcript as Hermes passes it to ``sync_turn``: any
    ``earlier`` rows, then this turn's user row stamped with ``epoch`` and
    typed with any ``display_kind``, the
    assistant's tool calls (each ``(name, args)`` becomes an assistant row
    with ``tool_calls`` and a tool row) and the final assistant row."""
    rows = list(earlier)
    rows.append({"role": "user", "content": user_text, "timestamp": epoch})
    if display_kind:
        rows[-1]["display_kind"] = display_kind
    for index, (name, args) in enumerate(tool_calls or ()):
        call_id = f"call_{index}"
        rows.append(
            {
                "role": "assistant",
                "content": None,
                "timestamp": epoch + 1 + index,
                "tool_calls": [
                    {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                ],
            }
        )
        rows.append({"role": "tool", "tool_call_id": call_id, "content": "{}", "timestamp": epoch + 1.5 + index})
    rows.append({"role": "assistant", "content": assistant_text, "timestamp": epoch + 5})
    return rows
