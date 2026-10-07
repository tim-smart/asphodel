"""``prefetch`` and ``recall_status``: the body, the previous query, the
pending ``recall_id``, the prefetch budget, "" on every failure and no call for a
query that cleans to nothing."""

import pytest

from conftest import SESSION
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


def test_successful_recall_injects_memories_without_a_status(make_provider, daemon):
    provider = make_provider()
    assert provider.recall_status() is None
    assert provider.prefetch("tea?", session_id=SESSION) == INJECTION
    assert provider.recall_status() is None


def test_returns_empty_and_no_status_when_the_daemon_is_down(make_provider, daemon):
    url = daemon.go_down()
    provider = make_provider(url=url)
    assert provider.prefetch("tea?", session_id=SESSION) == ""
    assert provider.recall_status() is None


def test_gives_up_at_the_budget(make_provider, daemon, clock):
    import time

    provider = make_provider()
    daemon.set_delay("prefetch", 2.0)
    started = time.monotonic()
    assert provider.prefetch("tea?", session_id=SESSION) == ""
    assert time.monotonic() - started < 1.5
    assert provider.recall_status() is None


def test_sends_a_pending_block_id_once_when_the_block_had_no_session(make_provider, daemon):
    """Session fallback: a block fetched before the session id was
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


#: The note Hermes' Discord gateway puts in front of a turn's message, with
#: a synthetic message id.
DISCORD_NOTE = (
    "[Triggering message id: `100000000000000001` \u2014 use as `message_id` "
    "for reply/react/pin via the discord tools.]"
)


def test_sends_the_raw_query_for_the_daemon_to_clean(make_provider, daemon):
    provider = make_provider()
    raw = f"{DISCORD_NOTE}\n\n[Sam] what time is the ferry on Saturday?"
    assert provider.prefetch(raw, session_id=SESSION) == INJECTION
    assert daemon.requests_for("prefetch")[0].body["query"] == raw


@pytest.mark.parametrize(
    "query",
    ["", "   ", "[Sam] ", DISCORD_NOTE, f"{DISCORD_NOTE}\n\n[Sam] "],
    ids=["empty", "blank", "prefix", "note", "note-and-prefix"],
)
def test_a_query_that_cleans_to_nothing_is_not_prefetched(make_provider, daemon, query):
    provider = make_provider()
    assert provider.prefetch(query, session_id=SESSION) == ""
    assert daemon.requests_for("prefetch") == []
    assert provider.recall_status() is None
