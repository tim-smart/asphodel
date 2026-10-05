"""``initialize``: one health
probe, ``PUT`` the bank with the configured identity, warnings through
``warning_callback``, and never a failure."""

import re

import pytest

from conftest import OWNER_IDS, PLUGIN_DIR, PROFILE


def test_probes_health_then_puts_the_bank(make_provider, daemon):
    make_provider(assistant_name="Hermes", timezone="Pacific/Auckland")
    keys = [r.key for r in daemon.requests]
    assert keys == ["health", "put_bank"]
    put = daemon.requests_for("put_bank")[0]
    assert put.bank == PROFILE
    assert put.body == {
        "owner_name": "Tim",
        "owner_platform_ids": OWNER_IDS,
        "assistant_name": "Hermes",
        "timezone": "Pacific/Auckland",
    }


def test_the_configured_bank_overrides_the_profile(make_provider, daemon):
    make_provider(bank="shared")
    assert daemon.requests_for("put_bank")[0].bank == "shared"


def workspace_major():
    cargo = (PLUGIN_DIR.parent / "Cargo.toml").read_text()
    return re.search(r'^version = "(\d+)\.', cargo, re.M).group(1)


@pytest.mark.parametrize(
    "daemon_state, warned",
    [
        ("the-workspace-version", 0),
        ("another-major-version", 1),
        ("not-ready", 1),
        ("down", 1),
    ],
)
def test_warns_once_unless_the_daemon_is_ready_at_the_workspace_major(make_provider, daemon, warnings, daemon_state, warned):
    """The plugin is written for the daemon in this repo. A daemon it can't
    use yet is a warning, never a failure."""
    major = workspace_major()
    url = None
    if daemon_state == "the-workspace-version":
        daemon.version = f"{major}.99.0"
    elif daemon_state == "another-major-version":
        daemon.version = f"{int(major) + 1}.2.0"
    elif daemon_state == "not-ready":
        daemon.ready = False
    else:
        url = daemon.go_down()
    make_provider(url=url)
    assert len(warnings) == warned


def test_never_fails_without_a_warning_callback(make_provider, daemon):
    daemon.ready = False
    make_provider(init={"warning_callback": None})


# -- built-in memory flags, coerced the way Hermes does (``is_truthy_value(v,
# default=True)`` in ``tools/memory_tool.py``) -----------------------------------


@pytest.mark.parametrize(
    "config, warned",
    [
        ({"memory": {}}, {"memory_enabled", "user_profile_enabled"}),
        ({"memory": {"memory_enabled": "true", "user_profile_enabled": "yes"}}, {"memory_enabled", "user_profile_enabled"}),
        ({"memory": {"memory_enabled": "false", "user_profile_enabled": "off"}}, set()),
    ],
    ids=["empty-section", "truthy-strings", "false-strings"],
)
def test_warns_for_each_builtin_memory_flag_hermes_reads_as_on(make_provider, hermes, warnings, config, warned):
    hermes.config = config
    make_provider()
    assert {flag for flag in ("memory_enabled", "user_profile_enabled") if any(flag in w for w in warnings)} == warned
    assert len(warnings) == len(warned)
