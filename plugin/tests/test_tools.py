"""The four tools and the owner check."""

import json

import pytest

from conftest import SESSION, plugin
from fake_daemon import recalled

TOOLS = plugin.tools


def error_of(result: str):
    return json.loads(result).get("error")


# -- schemas --------------------------------------------------------------------


def test_four_tools_with_bare_function_schemas(make_provider):
    schemas = make_provider().get_tool_schemas()
    assert [s["name"] for s in schemas] == ["memory_recall", "memory_forget", "memory_keep", "memory_unkeep"]
    for schema in schemas:
        assert set(schema) == {"name", "description", "parameters"}
        assert schema["parameters"]["type"] == "object"


def test_forget_takes_ids_only():
    forget = next(s for s in TOOLS.TOOL_SCHEMAS if s["name"] == "memory_forget")
    assert set(forget["parameters"]["properties"]) == {"ids"}
    assert forget["parameters"]["properties"]["ids"]["maxItems"] == 50


# -- the owner check -------------------------------------------------------------


def test_is_owner_rules():
    owner = dict(platform="discord", owner_platform_ids=["discord:111"], agent_context="primary")
    assert TOOLS.is_owner(author_id=None, author_is_bot=False, **owner)
    assert TOOLS.is_owner(author_id="111", author_is_bot=False, **owner)
    assert not TOOLS.is_owner(author_id="222", author_is_bot=False, **owner)
    assert not TOOLS.is_owner(author_id="111", author_is_bot=True, **owner)
    assert not TOOLS.is_owner(author_id="111", author_is_bot=False, **dict(owner, agent_context="cron"))
    assert not TOOLS.is_owner(author_id=None, author_is_bot=False, **dict(owner, agent_context="cron"))
    assert not TOOLS.is_owner(author_id="111", author_is_bot=False, **dict(owner, platform="telegram"))


def test_the_owner_may_forget_keep_and_unkeep(make_provider, daemon):
    provider = make_provider()
    provider.on_turn_start(1, "forget it", author_id="111", author_name="Tim", author_is_bot=False)
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
        ({"agent_context": "cron"}, {}),
    ],
    ids=["another-speaker", "bot-with-owner-id", "cron"],
)
def test_a_non_owner_is_refused_without_a_request(make_provider, daemon, init, author):
    provider = make_provider(init=init)
    provider.on_turn_start(1, "forget it", **author)
    for tool in ("memory_forget", "memory_keep", "memory_unkeep"):
        message = error_of(provider.handle_tool_call(tool, {"ids": ["m1"]}))
        assert message and "owner" in message.lower()
    assert daemon.requests_for("forget") == daemon.requests_for("keep") == daemon.requests_for("unkeep") == []


def test_the_author_is_read_per_turn(make_provider, daemon):
    provider = make_provider()
    provider.on_turn_start(1, "x", author_id="111", author_name="Tim", author_is_bot=False)
    provider.on_turn_start(2, "y", author_id="222", author_name="Maya", author_is_bot=False)
    assert error_of(provider.handle_tool_call("memory_forget", {"ids": ["m1"]}))
    provider.on_turn_start(3, "z", author_id="111", author_name="Tim", author_is_bot=False)
    assert "forgotten" in json.loads(provider.handle_tool_call("memory_forget", {"ids": ["m1"]}))


# -- recall ----------------------------------------------------------------------


def test_anyone_may_recall_and_gets_plain_lines(make_provider, daemon):
    provider = make_provider()
    provider.on_turn_start(1, "x", author_id="222", author_name="Maya", author_is_bot=False)
    text = "m9 Maya lives in Wellington.\nm10 Maya has a concert. [event; upcoming Sun 4 Oct; kept]"
    concert = recalled("Maya has a concert.", id="m10", kind="event", kept=True)
    concert["window"]["valid_from"] = {"at": "2026-10-03T14:00:00Z", "precision": "day"}
    daemon.set_response(
        "recall",
        200,
        {
            "recall_id": "r1",
            "results": [recalled("Maya lives in Wellington.", id="m9"), concert],
            "text": text,
            "reranked": True,
        },
    )
    result = provider.handle_tool_call("memory_recall", {"query": "Maya"})
    assert result == text
    assert "2026-10-03T14:00:00Z" not in result
    assert "null" not in result


def test_empty_recall_says_so_in_words(make_provider, daemon):
    provider = make_provider()
    daemon.set_response("recall", 200, {"recall_id": "r1", "results": [], "text": "No memories recalled.", "reranked": True})
    assert provider.handle_tool_call("memory_recall", {"query": "concert"}) == "No memories recalled."


@pytest.mark.parametrize("tool, route", [("memory_forget", "forget"), ("memory_keep", "keep"), ("memory_unkeep", "unkeep")])
def test_recalled_line_id_can_be_used_unchanged_by_owner_tools(make_provider, daemon, tool, route):
    provider = make_provider()
    memory_id = "7d0a9ac0-2b6d-4e89-93be-3ea8f1ae7b2f"
    daemon.set_response(
        "recall",
        200,
        {
            "recall_id": "r1",
            "results": [recalled("Maya lives in Wellington.", id=memory_id)],
            "text": f"{memory_id} Maya lives in Wellington.",
            "reranked": True,
        },
    )
    result = provider.handle_tool_call("memory_recall", {"query": "Maya"})
    recalled_id = result.splitlines()[0].split()[0]
    provider.on_turn_start(1, "keep it", author_id="111", author_is_bot=False)
    response = provider.handle_tool_call(tool, {"ids": [recalled_id]})
    assert not error_of(response)
    assert daemon.requests_for(route)[0].body["ids"] == [memory_id]


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


def test_daemon_errors_become_tool_errors_without_content(make_provider, daemon):
    provider = make_provider()
    daemon.set_response("recall", 400, {"error": "`from` is after `to`"})
    message = error_of(provider.handle_tool_call("memory_recall", {"query": "tea"}))
    assert message and "400" in message


def test_an_unreachable_daemon_is_a_tool_error(make_provider, daemon):
    url = daemon.go_down()
    provider = make_provider(url=url)
    assert error_of(provider.handle_tool_call("memory_recall", {"query": "tea"}))
    assert error_of(provider.handle_tool_call("memory_keep", {"ids": ["m1"]}))


def test_bad_arguments_are_a_tool_error_not_an_exception(make_provider, daemon):
    provider = make_provider()
    assert error_of(provider.handle_tool_call("memory_forget", {}))
    assert error_of(provider.handle_tool_call("memory_forget", {"ids": "m1"}))
    assert error_of(provider.handle_tool_call("memory_recall", {}))
    assert daemon.requests_for("forget") == daemon.requests_for("recall") == []
