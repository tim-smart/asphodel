"""Bank setup that ``initialize`` couldn't finish: until a ``PUT``
succeeds, every bank operation retries it first and
turns are spooled, not dropped. Once the bank exists, a 404 never recreates
it: ``bank delete`` expects the plugin to be disabled first."""

import time

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


# -- a delayed setup shares the hook's budget -------------------------------------


def _recover_slowly(make_provider, daemon, *, put_delay, route, route_delay):
    """A provider whose bank PUT failed while the daemon started, against a
    daemon that is now ready but slow to PUT and to answer ``route``."""
    daemon.enforce_banks = True
    daemon.ready = False
    provider = make_provider()
    daemon.ready = True
    daemon.set_delay("put_bank", put_delay)
    daemon.set_delay(route, route_delay)
    return provider


def test_prefetch_setup_and_request_share_one_budget(make_provider, daemon):
    from conftest import FAST

    provider = _recover_slowly(make_provider, daemon, put_delay=0.35, route="prefetch", route_delay=0.35)
    started = time.monotonic()
    assert provider.prefetch("what do I drink in the morning?", session_id=SESSION) == ""
    assert time.monotonic() - started < FAST.prefetch + 0.15
    assert provider.recall_status() is None
    # The PUT finished inside the budget, so the bank is set up for the next turn.
    daemon.delays.clear()
    assert provider.prefetch("what do I drink in the morning?", session_id=SESSION) == INJECTION
    assert len(daemon.requests_for("put_bank")) == 2


def test_each_system_prompt_attempt_shares_its_budget_with_setup(make_provider, daemon):
    from conftest import FAST

    provider = _recover_slowly(make_provider, daemon, put_delay=0.3, route="system_prompt", route_delay=1.0)
    started = time.monotonic()
    assert provider.system_prompt_block() == ""
    attempts = 1 + FAST.system_prompt_retries
    assert time.monotonic() - started < attempts * FAST.system_prompt + 0.15
    # The retry policy is unchanged: a timed-out GET is retried once, and the
    # bank, set up by the first attempt, is not PUT again.
    assert len(daemon.requests_for("system_prompt")) == attempts
    assert len(daemon.requests_for("put_bank")) == 2
