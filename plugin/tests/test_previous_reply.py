"""Conversation context sent over the daemon HTTP boundary."""

import json

import pytest

from conftest import SESSION


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
