"""Plugin config (TIM-94, decisions 2, 3 and 7): ``config.json`` under
``$HERMES_HOME/asphodel/``, ``ASPHODEL_URL`` winning over ``url``, the token
from ``ASPHODEL_TOKEN``, and ``is_available`` that never touches the
network."""

import json

from conftest import plugin, write_config

load_config = plugin.config.load_config


def test_environment_url_overrides_the_file(hermes_home, monkeypatch):
    write_config(hermes_home, url="http://10.0.0.5:7720")
    monkeypatch.setenv("ASPHODEL_URL", "unix:/run/asphodel.sock")
    assert load_config(hermes_home).url == "unix:/run/asphodel.sock"


def test_token_comes_from_the_environment_only(hermes_home, monkeypatch):
    write_config(hermes_home, token="in-the-file")
    assert load_config(hermes_home).token is None
    monkeypatch.setenv("ASPHODEL_TOKEN", "s3cret")
    assert load_config(hermes_home).token == "s3cret"


def test_unreadable_file_means_defaults(hermes_home):
    path = write_config(hermes_home)
    path.write_text("{not json")
    assert load_config(hermes_home).url == "http://127.0.0.1:7720"


def test_save_config_merges_and_writes_by_rename(hermes_home):
    write_config(hermes_home, url="http://a", bank="old")
    plugin.config.save_config(hermes_home, {"bank": "new", "ingest": False})
    path = hermes_home / "asphodel" / "config.json"
    assert json.loads(path.read_text()) == {"url": "http://a", "bank": "new", "ingest": False}
    assert [p.name for p in path.parent.iterdir()] == ["config.json"]


def test_is_available_is_config_only(hermes_home, daemon):
    write_config(hermes_home, url=daemon.url)
    provider = plugin.AsphodelMemoryProvider()
    assert provider.is_available() is True
    assert daemon.requests == []


def test_not_available_when_the_url_is_blank(hermes_home):
    write_config(hermes_home, url="")
    provider = plugin.AsphodelMemoryProvider()
    assert provider.is_available() is False
    assert "ASPHODEL_URL" in provider.unavailable_reason()


def test_bearer_token_is_sent_on_every_request(make_provider, daemon, monkeypatch):
    monkeypatch.setenv("ASPHODEL_TOKEN", "s3cret")
    provider = make_provider()
    provider.prefetch("what do I drink in the morning?", session_id="sess-0001")
    assert daemon.requests
    assert all(r.authorization == "Bearer s3cret" for r in daemon.requests)
