"""``initialize`` (TIM-94, decisions 3 and 7, round 1 item 5): one health
probe, ``PUT`` the bank with the configured identity, warnings through
``warning_callback``, and never a failure."""

from conftest import OWNER_IDS, PROFILE, SESSION, plugin


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


def test_bank_name_defaults_to_the_profile_and_config_overrides_it(make_provider, daemon):
    make_provider(bank="shared")
    assert daemon.requests_for("put_bank")[0].bank == "shared"


def test_assistant_name_defaults_to_the_profile(make_provider, daemon):
    make_provider()
    assert daemon.requests_for("put_bank")[0].body["assistant_name"] == PROFILE


def test_absent_identity_fields_are_sent_as_null_not_invented(make_provider, daemon):
    make_provider(owner_name=None, owner_platform_ids=[])
    body = daemon.requests_for("put_bank")[0].body
    assert body["owner_name"] is None
    assert body["owner_platform_ids"] == []
    assert body["timezone"] is None


def test_health_budget_and_no_warning_when_ready(make_provider, warnings):
    make_provider()
    assert warnings == []


def test_warns_when_the_daemon_is_not_ready(make_provider, daemon, warnings):
    daemon.ready = False
    make_provider()
    assert len(warnings) == 1
    assert "ready" in warnings[0].lower()


def test_warns_when_the_daemon_major_version_differs(make_provider, daemon, warnings):
    daemon.version = "1.2.0"
    make_provider()
    assert any("1.2.0" in w and "version" in w.lower() for w in warnings)


def test_same_major_different_minor_is_silent(make_provider, daemon, warnings):
    daemon.version = "0.9.3"
    make_provider()
    assert warnings == []


def test_warns_when_builtin_memory_flags_are_still_on(make_provider, hermes, warnings):
    hermes.config = {"memory": {"memory_enabled": True, "user_profile_enabled": False}}
    make_provider()
    assert any("memory_enabled" in w for w in warnings)
    assert not any("user_profile_enabled" in w for w in warnings)


def test_warns_when_flags_are_absent_because_hermes_defaults_them_on(make_provider, hermes, warnings):
    hermes.config = {"memory": {"provider": "asphodel"}}
    make_provider()
    assert any("memory_enabled" in w for w in warnings)
    assert any("user_profile_enabled" in w for w in warnings)


def test_never_fails_when_the_daemon_is_down(make_provider, daemon, warnings):
    url = daemon.go_down()
    provider = make_provider(url=url)
    assert provider.bank == PROFILE
    assert len(warnings) == 1
    assert "reach" in warnings[0].lower() or "ready" in warnings[0].lower()


def test_never_fails_without_a_warning_callback(make_provider, daemon):
    daemon.ready = False
    make_provider(init={"warning_callback": None})


def test_records_session_platform_and_context(make_provider):
    provider = make_provider(init={"agent_context": "cron", "platform": "telegram"})
    assert provider._session_id == SESSION
    assert provider._platform == "telegram"
    assert provider._agent_context == "cron"


def test_backup_paths_is_empty(make_provider):
    assert make_provider().backup_paths() == []
