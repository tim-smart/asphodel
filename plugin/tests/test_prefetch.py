"""``prefetch`` and ``recall_status`` (TIM-94, decisions 5 and 6; TIM-99
amendment): the body, the previous query, the pending ``recall_id``, the 3 s
budget and "" on every failure."""

from conftest import SESSION, plugin
from fake_daemon import INJECTION


def test_sends_query_session_and_returns_the_injection(make_provider, daemon):
    provider = make_provider()
    text = provider.prefetch("what do I drink in the morning?", session_id=SESSION)
    assert text == INJECTION
    request = daemon.requests_for("prefetch")[0]
    assert request.bank == "tim"
    assert request.body["session_id"] == SESSION
    assert request.body["query"] == "what do I drink in the morning?"
    assert request.body.get("previous_query") is None
    assert request.body.get("block_id") is None


def test_previous_query_is_the_last_prefetch_query_for_the_session(make_provider, daemon):
    provider = make_provider()
    provider.prefetch("first", session_id=SESSION)
    provider.prefetch("second", session_id=SESSION)
    provider.prefetch("other session", session_id="sess-0002")
    bodies = [r.body for r in daemon.requests_for("prefetch")]
    assert bodies[1]["previous_query"] == "first"
    assert bodies[2].get("previous_query") is None


def test_a_failed_prefetch_does_not_become_the_previous_query(make_provider, daemon):
    provider = make_provider()
    provider.prefetch("first", session_id=SESSION)
    daemon.set_response("prefetch", 500, {"error": "boom"})
    provider.prefetch("lost", session_id=SESSION)
    daemon.responses.clear()
    provider.prefetch("third", session_id=SESSION)
    assert daemon.requests_for("prefetch")[-1].body["previous_query"] == "first"


def test_recall_status_reports_the_last_injected_count(make_provider, daemon):
    provider = make_provider()
    provider.prefetch("tea?", session_id=SESSION)
    status = provider.recall_status()
    assert status is not None
    assert status.count == 2
    assert status.provider_label == "Asphodel"


def test_recall_status_is_none_when_nothing_was_injected(make_provider, daemon):
    provider = make_provider()
    provider.prefetch("tea?", session_id=SESSION)
    daemon.set_response("prefetch", 200, {"recall_id": "r2", "text": "", "injected": [], "reranked": True})
    assert provider.prefetch("nothing", session_id=SESSION) == ""
    assert provider.recall_status() is None


def test_returns_empty_and_no_status_when_the_daemon_is_down(make_provider, daemon):
    url = daemon.go_down()
    provider = make_provider(url=url)
    assert provider.prefetch("tea?", session_id=SESSION) == ""
    assert provider.recall_status() is None


def test_returns_empty_on_a_daemon_error(make_provider, daemon):
    provider = make_provider()
    daemon.set_response("prefetch", 404, {"error": "unknown bank"})
    assert provider.prefetch("tea?", session_id=SESSION) == ""


def test_gives_up_at_the_budget(make_provider, daemon, clock):
    import time

    provider = make_provider()
    daemon.set_delay("prefetch", 2.0)
    started = time.monotonic()
    assert provider.prefetch("tea?", session_id=SESSION) == ""
    assert time.monotonic() - started < 1.5
    assert provider.recall_status() is None


def test_falls_back_to_the_initialised_session_id(make_provider, daemon):
    provider = make_provider()
    provider.prefetch("tea?")
    assert daemon.requests_for("prefetch")[0].body["session_id"] == SESSION


def test_sends_a_pending_block_id_once_when_the_block_had_no_session(make_provider, daemon):
    """TIM-95 decision 4 fallback: a block fetched before the session id was
    known is mapped through the session's first prefetch."""
    provider = make_provider(init={"session_id": ""})
    provider.system_prompt_block()
    block_id = daemon.requests_for("system_prompt")[0]
    assert "session_id" not in block_id.query
    provider.prefetch("tea?", session_id=SESSION)
    provider.prefetch("more tea?", session_id=SESSION)
    bodies = [r.body for r in daemon.requests_for("prefetch")]
    assert bodies[0]["block_id"]
    assert bodies[1].get("block_id") is None
