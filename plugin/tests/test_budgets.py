"""The production budgets and constants as the tickets set them. The other
tests shorten them; this pins the defaults."""

from conftest import plugin


def test_hook_budgets():
    timeouts = plugin.provider.Timeouts()
    assert timeouts.prefetch == 3.0
    assert timeouts.system_prompt == 2.0
    assert timeouts.system_prompt_retries == 1


def test_breaker_constants():
    assert plugin.breaker.BREAKER_FAILURES == 3
    assert plugin.breaker.BREAKER_COOLDOWN_S == 30.0


def test_spool_caps():
    assert plugin.spool.SPOOL_MAX_BYTES == 10 * 1024 * 1024
    assert plugin.spool.SPOOL_MAX_AGE_S == 7 * 24 * 3600


def test_daemon_major_version_matches_the_workspace():
    """The plugin is written for the daemon in this repo."""
    import re
    from conftest import PLUGIN_DIR

    cargo = (PLUGIN_DIR.parent / "Cargo.toml").read_text()
    major = int(re.search(r'^version = "(\d+)\.', cargo, re.M).group(1))
    assert plugin.provider.DAEMON_MAJOR_VERSION == major
