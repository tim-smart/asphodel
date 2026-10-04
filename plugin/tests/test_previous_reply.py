"""Conversation context sent over the daemon HTTP boundary."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from conftest import FAST, SESSION


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("Try oolong tea.", "Try oolong tea."),
        ("a" * 300, "a" * 300),
        ("茶" * 300 + "DO NOT SEND THIS TAIL", "茶" * 300),
    ],
    ids=["short", "at-limit", "unicode-over-limit"],
)
def test_next_prefetch_sends_a_bounded_reply_prefix(make_provider, daemon, reply, expected):
    provider = make_provider()
    provider.prefetch("first")
    assert daemon.requests_for("prefetch")[-1].body.get("previous_reply") is None
    provider.sync_turn("first", reply)
    provider.prefetch("second")
    body = daemon.requests_for("prefetch")[-1].body
    assert body["previous_query"] == "first"
    assert body.get("previous_reply") == expected
    # The next completed turn replaces the reply, rather than keeping the first.
    provider.sync_turn("second", "A different answer.")
    provider.prefetch("third")
    body = daemon.requests_for("prefetch")[-1].body
    assert body["previous_query"] == "second"
    assert body.get("previous_reply") == "A different answer."


def test_reply_context_is_isolated_by_session(make_provider, daemon):
    provider = make_provider()
    provider.prefetch("first")
    provider.sync_turn("first", "Answer for the first session.")
    provider.on_session_switch("sess-0002")
    provider.prefetch("other session")
    body = daemon.requests_for("prefetch")[-1].body
    assert body.get("previous_query") is None
    assert body.get("previous_reply") is None
    # A delayed sync names its own session, not the currently bound one.
    provider.sync_turn("first", "Updated first-session answer.", session_id=SESSION)
    provider.sync_turn("other session", "Answer for the second session.", session_id="sess-0002")
    provider.prefetch("more in the second session")
    assert daemon.requests_for("prefetch")[-1].body.get("previous_reply") == "Answer for the second session."
    provider.on_session_switch(SESSION)
    provider.prefetch("back in the first session")
    assert daemon.requests_for("prefetch")[-1].body.get("previous_reply") == "Updated first-session answer."


@pytest.mark.parametrize("clear", ["forget", "compression", "reset", "rewind"])
def test_clearing_discards_reply_context(make_provider, daemon, clear):
    provider = make_provider()
    provider.prefetch("before")
    provider.sync_turn("before", "Old answer.")
    provider.prefetch("check context")
    # Establish that there is context to clear, so absence alone cannot pass.
    assert daemon.requests_for("prefetch")[-1].body.get("previous_reply") == "Old answer."
    if clear == "forget":
        result = json.loads(provider.handle_tool_call("memory_forget", {"ids": ["m1"]}))
        assert result == {"forgotten": ["m1"], "unknown": []}
    else:
        kwargs = {"compression": {"reason": "compression"}, "reset": {"reset": True}, "rewind": {"rewound": True}}
        provider.on_session_switch(SESSION, **kwargs[clear])
        assert daemon.requests_for("clear")[-1].session == SESSION
    provider.prefetch("after clear")
    body = daemon.requests_for("prefetch")[-1].body
    assert body.get("previous_query") is None
    assert body.get("previous_reply") is None
    # Once a query exists again, the forgotten reply must still be absent.
    provider.prefetch("another query before sync")
    body = daemon.requests_for("prefetch")[-1].body
    assert body["previous_query"] == "after clear"
    assert body.get("previous_reply") is None


@pytest.mark.parametrize("delayed_sync", [False, True], ids=["unsynced-next-turn", "late-sync"])
def test_reply_is_not_paired_with_a_different_query(make_provider, daemon, delayed_sync):
    provider = make_provider()
    provider.prefetch("first")
    if not delayed_sync:
        provider.sync_turn("first", "Answer to first.")
    provider.prefetch("second")
    if delayed_sync:
        provider.sync_turn("first", "Answer to first.")
    provider.prefetch("third")
    body = daemon.requests_for("prefetch")[-1].body
    assert body["previous_query"] == "second"
    assert body.get("previous_reply") is None
    # A reply that actually belongs to the current previous query still works.
    provider.sync_turn("third", "Answer to third.")
    provider.prefetch("fourth")
    body = daemon.requests_for("prefetch")[-1].body
    assert body["previous_query"] == "third"
    assert body.get("previous_reply") == "Answer to third."


@pytest.mark.parametrize("clear", ["forget", "compression", "reset", "rewind"])
@pytest.mark.parametrize("repeated_query", [False, True], ids=["distinct-query", "repeated-query"])
def test_late_sync_cannot_restore_cleared_reply_context(make_provider, daemon, clear, repeated_query):
    provider = make_provider()
    provider.prefetch("old query")
    if clear == "forget":
        result = json.loads(provider.handle_tool_call("memory_forget", {"ids": ["m1"]}))
        assert result == {"forgotten": ["m1"], "unknown": []}
    else:
        kwargs = {"compression": {"reason": "compression"}, "reset": {"reset": True}, "rewind": {"rewound": True}}
        provider.on_session_switch(SESSION, **kwargs[clear])
        assert daemon.requests_for("clear")[-1].session == SESSION
    fresh_query = "old query" if repeated_query else "fresh query"
    provider.prefetch(fresh_query)
    body = daemon.requests_for("prefetch")[-1].body
    assert body.get("previous_query") is None
    assert body.get("previous_reply") is None
    # This completion belongs to the turn before the clear, even when its
    # text matches a new turn. Query-text equality alone cannot establish that.
    provider.sync_turn("old query", "Answer from before the clear.")
    provider.prefetch("next query")
    body = daemon.requests_for("prefetch")[-1].body
    assert body["previous_query"] == fresh_query
    assert body.get("previous_reply") is None


def test_clear_during_prefetch_cannot_restore_reply_context(make_provider, daemon):
    provider = make_provider(timeouts=replace(FAST, prefetch=5.0))
    started = Event()
    release = Event()

    def held_prefetch(request):
        started.set()
        release.wait(timeout=10.0)
        return 200, {"text": "Completed held prefetch."}

    daemon.set_handler("prefetch", held_prefetch)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(provider.prefetch, "old query")
        try:
            assert started.wait(timeout=3.0), "prefetch did not reach the daemon"
            provider.on_session_switch(SESSION, reset=True)
            assert daemon.requests_for("clear")[-1].session == SESSION
        finally:
            release.set()
        # Require a successful response: a timeout would not exercise the
        # completed request trying to restore its pre-clear context.
        assert pending.result(timeout=5.0) == "Completed held prefetch."

    daemon.set_response("prefetch", 200, {"text": "Fresh prefetch."})
    provider.sync_turn("old query", "Late answer from before the clear.")
    assert provider.prefetch("fresh query") == "Fresh prefetch."
    body = daemon.requests_for("prefetch")[-1].body
    assert body.get("previous_query") is None
    assert body.get("previous_reply") is None


@pytest.mark.parametrize(
    "first_attempt_fails", [True, False], ids=["failed-then-successful", "successful-then-failed"]
)
def test_failed_repeated_prefetch_makes_reply_ambiguous(make_provider, daemon, first_attempt_fails):
    provider = make_provider()
    for fails in (first_attempt_fails, not first_attempt_fails):
        if fails:
            daemon.set_response("prefetch", 500, {"error": "prefetch failed"})
            assert provider.prefetch("repeated query") == ""
        else:
            daemon.set_response("prefetch", 200, {"text": "Successful prefetch."})
            assert provider.prefetch("repeated query") == "Successful prefetch."
    assert [request.body["query"] for request in daemon.requests_for("prefetch")] == [
        "repeated query", "repeated query"
    ]

    provider.sync_turn("repeated query", "Answer with ambiguous turn attribution.")
    daemon.set_response("prefetch", 200, {"text": "Next prefetch."})
    assert provider.prefetch("next query") == "Next prefetch."
    body = daemon.requests_for("prefetch")[-1].body
    assert body["previous_query"] == "repeated query"
    assert body.get("previous_reply") is None
