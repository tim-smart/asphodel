"""Plugin config: ``config.json`` under
``$HERMES_HOME/asphodel/``, ``ASPHODEL_URL`` winning over ``url``, the token
from ``ASPHODEL_TOKEN`` only, and ``is_available`` that never touches the
network."""

import pytest

from conftest import SESSION, plugin, write_config


@pytest.mark.parametrize(
    "contents, env_url, available",
    [
        ('{"url": "{daemon}"}', None, True),
        ('{"url": ""}', None, False),
        ('{"url": ""}', "{daemon}", True),
        ("{not json", None, True),
    ],
    ids=["file-url", "blank-url", "environment-url", "unreadable-file-means-the-default"],
)
def test_is_available_reads_config_only(hermes_home, daemon, monkeypatch, contents, env_url, available):
    write_config(hermes_home).write_text(contents.replace("{daemon}", daemon.url))
    if env_url:
        monkeypatch.setenv("ASPHODEL_URL", env_url.replace("{daemon}", daemon.url))
    assert plugin.AsphodelMemoryProvider().is_available() is available
    assert daemon.requests == []


def test_the_environment_url_overrides_the_file(make_provider, daemon, monkeypatch):
    monkeypatch.setenv("ASPHODEL_URL", daemon.url)
    make_provider(url="http://127.0.0.1:1")
    assert [r.key for r in daemon.requests] == ["health", "put_bank"]


def test_the_bearer_token_comes_from_the_environment_only(make_provider, daemon, monkeypatch):
    make_provider(token="in-the-file").prefetch("tea?", session_id=SESSION)
    assert daemon.requests and all(r.authorization is None for r in daemon.requests)
    monkeypatch.setenv("ASPHODEL_TOKEN", "s3cret")
    mark = len(daemon.requests)
    make_provider().prefetch("tea?", session_id=SESSION)
    assert daemon.requests[mark:]
    assert all(r.authorization == "Bearer s3cret" for r in daemon.requests[mark:])
