"""The circuit breaker: three consecutive connection
failures open it for 30 s; while open no request is made and every hook
answers as if the daemon were down. HTTP errors never trip it."""

from conftest import SESSION, transcript
from fake_daemon import INJECTION


def test_prefetch_skips_the_network_after_three_consecutive_connection_failures(make_provider, daemon, clock):
    provider = make_provider()
    # Two failures and a success: the success resets the count.
    daemon.drop_connections("prefetch", 2)
    for n in range(3):
        provider.prefetch(f"q{n}", session_id=SESSION)
    daemon.drop_connections("prefetch", 100)
    for n in range(5):
        assert provider.prefetch(f"down{n}", session_id=SESSION) == ""
    assert len(daemon.requests_for("prefetch")) == 3 + 3
    # After the cooldown one attempt goes out, and its failure reopens the breaker at once.
    clock.advance(31)
    provider.prefetch("after", session_id=SESSION)
    provider.prefetch("again", session_id=SESSION)
    assert len(daemon.requests_for("prefetch")) == 7
    daemon.drops.clear()
    clock.advance(31)
    assert provider.prefetch("recovered", session_id=SESSION) == INJECTION


def test_all_hooks_share_one_breaker(make_provider, daemon, clock):
    provider = make_provider()
    daemon.drop_connections("prefetch", 100)
    daemon.drop_connections("system_prompt", 100)
    for _ in range(3):
        provider.prefetch("q", session_id=SESSION)
    assert provider.system_prompt_block() == ""
    assert daemon.requests_for("system_prompt") == []
    assert "error" in provider.handle_tool_call("memory_recall", {"query": "tea"})
    assert daemon.requests_for("recall") == []


def test_http_errors_do_not_trip_the_breaker(make_provider, daemon):
    provider = make_provider()
    daemon.set_response("prefetch", 500, {"error": "store"})
    for n in range(5):
        provider.prefetch(f"q{n}", session_id=SESSION)
    assert len(daemon.requests_for("prefetch")) == 5


def test_an_open_breaker_spools_without_a_connection_attempt(make_provider, daemon, clock, hermes_home):
    provider = make_provider()
    daemon.drop_connections("turns", 100)
    for n in range(4):
        provider.sync_turn(f"t{n}", "ok", session_id=SESSION, messages=transcript(f"t{n}", "ok", epoch=float(n)))
    assert len(daemon.requests_for("turns")) == 3
    assert len(list((hermes_home / "asphodel" / "spool").glob("*.json"))) == 4
