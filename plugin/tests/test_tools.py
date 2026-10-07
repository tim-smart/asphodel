"""The four tools and the owner check."""

import json

import pytest

from conftest import SESSION
from fake_daemon import recalled


def error_of(result: str):
    return json.loads(result).get("error")


def test_four_tools_with_bare_function_schemas(make_provider):
    schemas = make_provider().get_tool_schemas()
    assert [s["name"] for s in schemas] == ["memory_recall", "memory_forget", "memory_keep", "memory_unkeep"]
    for schema in schemas:
        assert set(schema) == {"name", "description", "parameters"}
        assert schema["parameters"]["type"] == "object"
    for schema in schemas[1:]:
        assert set(schema["parameters"]["properties"]) == {"ids"}


# -- the owner check -------------------------------------------------------------


@pytest.mark.parametrize(
    "author",
    [{}, dict(author_id="111", author_name="Tim", author_is_bot=False)],
    ids=["no-author", "owner-id"],
)
def test_the_owner_may_forget_keep_and_unkeep(make_provider, daemon, author):
    provider = make_provider()
    provider.on_turn_start(1, "forget it", **author)
    assert json.loads(provider.handle_tool_call("memory_forget", {"ids": ["m1"]})) == {"forgotten": ["m1"], "unknown": []}
    assert json.loads(provider.handle_tool_call("memory_keep", {"ids": ["m2"]})) == {"kept": ["m2"], "unknown": []}
    assert json.loads(provider.handle_tool_call("memory_unkeep", {"ids": ["m2"]})) == {"unkept": ["m2"], "unknown": []}
    assert daemon.requests_for("forget")[0].body == {"ids": ["m1"], "session_id": SESSION}
    assert daemon.requests_for("keep")[0].body == {"ids": ["m2"]}
    assert daemon.requests_for("unkeep")[0].body == {"ids": ["m2"]}


@pytest.mark.parametrize(
    "init, author",
    [
        ({}, dict(author_id="222", author_name="Maya", author_is_bot=False)),
        ({}, dict(author_id="111", author_name="Relay", author_is_bot=True)),
        ({"platform": "telegram"}, dict(author_id="111", author_name="Tim", author_is_bot=False)),
        ({"agent_context": "cron"}, {}),
        ({"agent_context": "cron"}, dict(author_id="111", author_name="Tim", author_is_bot=False)),
        ({"agent_context": "subagent"}, {}),
    ],
    ids=["another-speaker", "bot-with-owner-id", "owner-id-on-another-platform", "cron", "cron-with-owner-id", "subagent"],
)
def test_a_non_owner_is_refused_without_a_request(make_provider, daemon, init, author):
    provider = make_provider(init=init)
    provider.on_turn_start(1, "forget it", **author)
    for tool in ("memory_forget", "memory_keep", "memory_unkeep"):
        assert error_of(provider.handle_tool_call(tool, {"ids": ["m1"]}))
    assert daemon.requests_for("forget") == daemon.requests_for("keep") == daemon.requests_for("unkeep") == []


def test_the_author_is_read_per_turn(make_provider, daemon):
    provider = make_provider()
    provider.on_turn_start(1, "x", author_id="111", author_name="Tim", author_is_bot=False)
    provider.on_turn_start(2, "y", author_id="222", author_name="Maya", author_is_bot=False)
    assert error_of(provider.handle_tool_call("memory_forget", {"ids": ["m1"]}))
    provider.on_turn_start(3, "z", author_id="111", author_name="Tim", author_is_bot=False)
    assert "forgotten" in json.loads(provider.handle_tool_call("memory_forget", {"ids": ["m1"]}))


# -- recall ----------------------------------------------------------------------


def test_anyone_may_recall_and_gets_the_daemons_lines(make_provider, daemon):
    provider = make_provider()
    provider.on_turn_start(1, "x", author_id="222", author_name="Maya", author_is_bot=False)
    text = "m9 Maya lives in Wellington.\nm10 Maya has a concert. [event; upcoming Sun 4 Oct; kept]"
    results = [recalled("Maya lives in Wellington.", id="m9"), recalled("Maya has a concert.", id="m10")]
    daemon.set_response("recall", 200, {"recall_id": "r1", "results": results, "text": text, "reranked": True})
    assert provider.handle_tool_call("memory_recall", {"query": "Maya"}) == text


@pytest.mark.parametrize("reply", [{}, {"text": ""}, {"text": " \n"}], ids=["no-text", "empty", "blank"])
def test_an_empty_recall_says_so_in_words_not_an_error(make_provider, daemon, reply):
    daemon.set_response("recall", 200, dict(reply, recall_id="r1", results=[], reranked=True))
    result = make_provider().handle_tool_call("memory_recall", {"query": "concert"})
    assert result.strip() and not result.lstrip().startswith("{"), result


def test_recall_forwards_the_arguments_and_session(make_provider, daemon):
    provider = make_provider()
    args = {
        "query": "dentist",
        "from": "2026-10-01T00:00:00Z",
        "to": "2026-10-31T00:00:00Z",
        "on": "happened",
        "phase": "upcoming",
        "kinds": ["event"],
        "entity": "Dr Rao",
        "limit": 5,
    }
    provider.handle_tool_call("memory_recall", dict(args))
    body = daemon.requests_for("recall")[0].body
    assert body == dict(args, session_id=SESSION)


# -- failures ---------------------------------------------------------------------


def test_a_daemon_error_or_an_unreachable_daemon_is_a_tool_error(make_provider, daemon):
    provider = make_provider()
    daemon.set_response("recall", 400, {"error": "`from` is after `to`"})
    assert error_of(provider.handle_tool_call("memory_recall", {"query": "tea"}))
    # Results without their rendered lines come from a daemon too old to read.
    daemon.set_response("recall", 200, {"recall_id": "r1", "results": [recalled("Tea.")], "reranked": True})
    assert error_of(provider.handle_tool_call("memory_recall", {"query": "tea"}))
    daemon.go_down()
    assert error_of(provider.handle_tool_call("memory_recall", {"query": "tea"}))
    assert error_of(provider.handle_tool_call("memory_keep", {"ids": ["m1"]}))


def test_bad_arguments_are_a_tool_error_without_a_request(make_provider, daemon):
    provider = make_provider()
    max_ids = provider.get_tool_schemas()[1]["parameters"]["properties"]["ids"]["maxItems"]
    too_many = [f"m{n}" for n in range(max_ids + 1)]
    assert error_of(provider.handle_tool_call("memory_forget", {}))
    assert error_of(provider.handle_tool_call("memory_forget", {"ids": "m1"}))
    assert error_of(provider.handle_tool_call("memory_forget", {"ids": too_many}))
    assert error_of(provider.handle_tool_call("memory_recall", {}))
    assert daemon.requests_for("forget") == daemon.requests_for("recall") == []
    assert "kept" in json.loads(provider.handle_tool_call("memory_keep", {"ids": too_many[:max_ids]}))
