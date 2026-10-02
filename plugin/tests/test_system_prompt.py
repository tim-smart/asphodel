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


def test_does_not_retry_a_daemon_error(make_provider, daemon):
    provider = make_provider()
    daemon.set_response("system_prompt", 500, {"error": "store"})
    assert provider.system_prompt_block() == ""
    assert len(daemon.requests_for("system_prompt")) == 1
