"""``system_prompt_block`` (TIM-94, decision 8; TIM-95 decision 4): the GET
carries the session id, the budget is 2 s with one retry, and a miss is ""."""

from conftest import SESSION


def test_fetches_the_block_for_the_session(make_provider, daemon):
    provider = make_provider()
    text = provider.system_prompt_block()
    assert text.startswith("## Agenda")
    request = daemon.requests_for("system_prompt")[0]
    assert request.bank == "tim"
    assert request.query["session_id"] == SESSION


def test_retries_once_after_a_dropped_connection(make_provider, daemon):
    provider = make_provider()
    daemon.drop_connections("system_prompt", 1)
    assert provider.system_prompt_block().startswith("## Agenda")
    assert len(daemon.requests_for("system_prompt")) == 2


def test_gives_up_after_the_retry(make_provider, daemon):
    provider = make_provider()
    daemon.drop_connections("system_prompt", 5)
    assert provider.system_prompt_block() == ""
    assert len(daemon.requests_for("system_prompt")) == 2


def test_does_not_retry_a_daemon_error(make_provider, daemon):
    provider = make_provider()
    daemon.set_response("system_prompt", 500, {"error": "store"})
    assert provider.system_prompt_block() == ""
    assert len(daemon.requests_for("system_prompt")) == 1


def test_gives_up_at_the_budget(make_provider, daemon):
    import time

    provider = make_provider()
    daemon.set_delay("system_prompt", 2.0)
    started = time.monotonic()
    assert provider.system_prompt_block() == ""
    # Two attempts at the shortened 0.4 s budget, never the 2 s delay.
    assert time.monotonic() - started < 1.5


def test_empty_when_the_daemon_is_down(make_provider, daemon):
    url = daemon.go_down()
    provider = make_provider(url=url)
    assert provider.system_prompt_block() == ""


def test_block_fetched_with_a_session_leaves_no_pending_block_id(make_provider, daemon):
    provider = make_provider()
    provider.system_prompt_block()
    provider.prefetch("tea?", session_id=SESSION)
    assert daemon.requests_for("prefetch")[0].body.get("block_id") is None
