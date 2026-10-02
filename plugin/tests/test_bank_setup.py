"""Bank setup that ``initialize`` couldn't finish (TIM-94, decision 3; ADR
0010): until a ``PUT`` succeeds, every bank operation retries it first and
turns are spooled, not dropped. Once the bank exists, a 404 never recreates
it: ``bank delete`` expects the plugin to be disabled first."""

from conftest import SESSION, transcript
from fake_daemon import INJECTION


def spooled(hermes_home):
    return list((hermes_home / "asphodel" / "spool").glob("*.json"))


def test_a_bank_put_while_the_daemon_starts_is_retried_and_the_spool_delivered(make_provider, daemon, hermes_home):
    daemon.enforce_banks = True
    daemon.ready = False
    provider = make_provider()
    provider.sync_turn("while starting", "ok", session_id=SESSION, messages=transcript("while starting", "ok", epoch=1.0))
    assert len(spooled(hermes_home)) == 1

    daemon.ready = True
    mark = len(daemon.requests)
    provider.sync_turn("after ready", "ok", session_id=SESSION, messages=transcript("after ready", "ok", epoch=2.0))
    assert daemon.requests[mark].key == "put_bank"
    delivered = [r.body["user_text"] for r in daemon.requests[mark:] if r.key == "turns"]
    assert delivered == ["after ready", "while starting"]
    assert spooled(hermes_home) == []


def test_a_bank_put_that_could_not_connect_is_retried_by_the_next_hook(make_provider, daemon):
    daemon.enforce_banks = True
    daemon.drop_connections("health", 1)
    daemon.drop_connections("put_bank", 1)
    provider = make_provider()
    assert provider.prefetch("what do I drink in the morning?", session_id=SESSION) == INJECTION
    keys = [r.key for r in daemon.requests]
    assert keys == ["health", "put_bank", "put_bank", "prefetch"]


def test_a_bank_that_was_set_up_is_put_only_once(make_provider, daemon):
    daemon.enforce_banks = True
    provider = make_provider()
    provider.system_prompt_block()
    provider.prefetch("tea?", session_id=SESSION)
    provider.handle_tool_call("memory_recall", {"query": "tea"})
    provider.sync_turn("tea", "ok", session_id=SESSION, messages=transcript("tea", "ok"))
    assert len(daemon.requests_for("put_bank")) == 1
    assert len(daemon.requests_for("turns")) == 1


def test_a_404_after_setup_does_not_recreate_the_bank(make_provider, daemon, hermes_home):
    daemon.enforce_banks = True
    provider = make_provider()
    daemon.banks.clear()  # `asphodel bank delete` while the plugin is live
    provider.sync_turn("after delete", "ok", session_id=SESSION, messages=transcript("after delete", "ok"))
    assert len(daemon.requests_for("put_bank")) == 1
    assert spooled(hermes_home) == []
