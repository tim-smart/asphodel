"""The forget request turn (ADR 0010; TIM-99 decision 2): ``sync_turn`` finds
a ``memory_forget`` call in this turn's ``messages`` and sends
``forget_requested: true``; the tool call also drops the session's stored
previous query."""

from conftest import SESSION, plugin, transcript

forget_requested = plugin.turns.forget_requested


def test_true_when_this_turn_called_memory_forget():
    messages = transcript("forget my old address", "Done.", tool_calls=[("memory_forget", {"ids": ["m1"]})])
    assert forget_requested(messages) is True


def test_false_without_tool_calls():
    assert forget_requested(transcript("hello", "hi")) is False
    assert forget_requested(None) is False
    assert forget_requested([]) is False


def test_other_tools_do_not_count():
    messages = transcript(
        "what's my address?", "It's ...", tool_calls=[("memory_recall", {"query": "address"}), ("memory_keep", {"ids": ["m1"]})]
    )
    assert forget_requested(messages) is False


def test_a_forget_in_an_earlier_turn_does_not_mark_this_one():
    earlier = transcript("forget my old address", "Done.", epoch=1000.0, tool_calls=[("memory_forget", {"ids": ["m1"]})])
    messages = transcript("thanks, what's next?", "Nothing.", epoch=2000.0, earlier=earlier)
    assert forget_requested(messages) is False


def test_forget_among_several_calls_in_one_turn_counts():
    messages = transcript(
        "forget that", "Done.", tool_calls=[("memory_recall", {"query": "x"}), ("memory_forget", {"ids": ["m1"]})]
    )
    assert forget_requested(messages) is True


def test_sync_turn_sends_forget_requested(make_provider, daemon):
    provider = make_provider()
    messages = transcript("forget my old address", "Done.", tool_calls=[("memory_forget", {"ids": ["m1"]})])
    provider.sync_turn("forget my old address", "Done.", session_id=SESSION, messages=messages)
    assert daemon.requests_for("turns")[0].body["forget_requested"] is True


def test_sync_turn_defaults_forget_requested_to_false(make_provider, daemon):
    provider = make_provider()
    provider.sync_turn("hello", "hi", session_id=SESSION, messages=transcript("hello", "hi"))
    assert daemon.requests_for("turns")[0].body["forget_requested"] is False


def test_memory_forget_tool_call_drops_the_previous_query(make_provider, daemon):
    provider = make_provider()
    provider.prefetch("forget my old address at 12 Elm St", session_id=SESSION)
    provider.handle_tool_call("memory_forget", {"ids": ["m1"]})
    provider.prefetch("thanks", session_id=SESSION)
    assert daemon.requests_for("prefetch")[-1].body.get("previous_query") is None


def test_a_refused_forget_keeps_the_previous_query(make_provider, daemon):
    provider = make_provider()
    provider.prefetch("first", session_id=SESSION)
    provider.on_turn_start(2, "forget it", author_id="222", author_name="Maya", author_is_bot=False)
    provider.handle_tool_call("memory_forget", {"ids": ["m1"]})
    provider.prefetch("second", session_id=SESSION)
    assert daemon.requests_for("prefetch")[-1].body["previous_query"] == "first"
