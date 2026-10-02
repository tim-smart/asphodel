"""The package is what Hermes' discovery and loader expect (TIM-88,
"Packaging and discovery"), and it depends on nothing outside the standard
library and Hermes itself (ADR 0006)."""

import ast
import importlib.util
import sys
from pathlib import Path

from conftest import PLUGIN_DIR, plugin

HERMES_MODULES = {"agent", "tools", "hermes_time", "hermes_cli", "hermes_constants", "plugins"}


def plugin_sources():
    return [p for p in PLUGIN_DIR.glob("*.py")]


def test_init_names_the_provider_contract_for_discovery():
    text = (PLUGIN_DIR / "__init__.py").read_text()
    assert "MemoryProvider" in text and "register_memory_provider" in text


def test_register_hands_the_manager_one_provider():
    registered = []

    class Ctx:
        def register_memory_provider(self, instance):
            registered.append(instance)

    plugin.register(Ctx())
    assert len(registered) == 1
    assert isinstance(registered[0], plugin.AsphodelMemoryProvider)
    assert registered[0].name == "asphodel"


def test_plugin_yaml_declares_no_dependencies():
    lines = [line.strip() for line in (PLUGIN_DIR / "plugin.yaml").read_text().splitlines()]
    assert "name: asphodel" in lines
    assert "pip_dependencies: []" in lines
    assert "requires_env: []" in lines


def test_pyproject_declares_no_dependencies():
    text = (PLUGIN_DIR / "pyproject.toml").read_text()
    assert "dependencies = []" in text
    assert "package = false" in text


def test_only_stdlib_and_hermes_imports():
    offenders = []
    for path in plugin_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top not in sys.stdlib_module_names and top not in HERMES_MODULES:
                    offenders.append((path.name, name))
    assert offenders == []


def test_config_schema_loads_standalone_by_path():
    """The dashboard execs ``config_schema.py`` by path; it must import only
    ``plugins.memory.config_schema`` and expose ``CONFIG_SCHEMA``."""
    path = PLUGIN_DIR / "config_schema.py"
    spec = importlib.util.spec_from_file_location("_hermes_memory_config_schema.asphodel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schema = module.CONFIG_SCHEMA
    assert schema.name == "asphodel"
    keys = {f.key for f in schema.fields}
    assert keys == {"url", "token", "bank", "owner_name", "owner_platform_ids", "assistant_name", "timezone", "ingest"}
    token = next(f for f in schema.fields if f.key == "token")
    assert token.is_secret and token.env_key == "ASPHODEL_TOKEN"


def test_dashboard_schema_matches_setup_schema():
    dashboard = {f.key for f in _load_dashboard_schema().fields}
    setup = {f["key"] for f in plugin.AsphodelMemoryProvider().get_config_schema()}
    assert dashboard == setup


def _load_dashboard_schema():
    spec = importlib.util.spec_from_file_location("_schema_check", PLUGIN_DIR / "config_schema.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG_SCHEMA
