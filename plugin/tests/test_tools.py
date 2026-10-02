"""The four tools (TIM-94, decision 9) and the owner check (decision 1)."""

import json

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


def test_recall_schema_matches_the_daemons_request():
    recall = next(s for s in TOOLS.TOOL_SCHEMAS if s["name"] == "memory_recall")
    props = recall["parameters"]["properties"]
    assert set(props) == {"query", "from", "to", "on", "phase", "kinds", "entity", "limit"}
    assert recall["parameters"]["required"] == ["query"]
    assert props["on"]["enum"] == ["happened", "said"]
    assert props["phase"]["enum"] == ["upcoming", "past", "current", "any"]
    assert props["kinds"]["items"]["enum"] == ["fact", "event", "state", "task", "recurring"]
    assert props["limit"]["maximum"] == 30 and props["limit"]["default"] == 10


def test_forget_takes_ids_only_and_says_it_is_irreversible():
    forget = next(s for s in TOOLS.TOOL_SCHEMAS if s["name"] == "memory_forget")
    assert set(forget["parameters"]["properties"]) == {"ids"}
    assert forget["parameters"]["properties"]["ids"]["maxItems"] == 50
    assert "irreversibl" in forget["description"].lower()
    assert "owner" in forget["description"].lower()


def test_owner_only_tools_say_so():
    for schema in TOOLS.TOOL_SCHEMAS:
        if schema["name"] in TOOLS.OWNER_ONLY_TOOLS:
            assert "owner" in schema["description"].lower()


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


def test_a_turn_with_no_author_is_the_owners(make_provider, daemon):
    provider = make_provider(init={"platform": "cli"})
    provider.on_turn_start(1, "keep that")
    assert "kept" in json.loads(provider.handle_tool_call("memory_keep", {"ids": ["m1"]}))


def test_another_speaker_is_refused_without_a_request(make_provider, daemon):
    provider = make_provider()
    provider.on_turn_start(1, "forget it", author_id="222", author_name="Maya", author_is_bot=False)
    for tool in ("memory_forget", "memory_keep", "memory_unkeep"):
        message = error_of(provider.handle_tool_call(tool, {"ids": ["m1"]}))
        assert message and "owner" in message.lower()
    assert daemon.requests_for("forget") == daemon.requests_for("keep") == daemon.requests_for("unkeep") == []


def test_a_bot_is_refused_even_with_the_owners_id(make_provider, daemon):
    provider = make_provider()
    provider.on_turn_start(1, "forget it", author_id="111", author_name="Relay", author_is_bot=True)
    assert "owner" in error_of(provider.handle_tool_call("memory_forget", {"ids": ["m1"]})).lower()
    assert daemon.requests_for("forget") == []


def test_cron_runs_are_refused(make_provider, daemon):
    provider = make_provider(init={"agent_context": "cron"})
    provider.on_turn_start(1, "daily digest")
    assert "owner" in error_of(provider.handle_tool_call("memory_forget", {"ids": ["m1"]})).lower()
    assert daemon.requests_for("forget") == []


def test_the_author_is_read_per_turn(make_provider, daemon):
    provider = make_provider()
    provider.on_turn_start(1, "x", author_id="111", author_name="Tim", author_is_bot=False)
    provider.on_turn_start(2, "y", author_id="222", author_name="Maya", author_is_bot=False)
    assert error_of(provider.handle_tool_call("memory_forget", {"ids": ["m1"]}))
    provider.on_turn_start(3, "z", author_id="111", author_name="Tim", author_is_bot=False)
    assert "forgotten" in json.loads(provider.handle_tool_call("memory_forget", {"ids": ["m1"]}))


# -- recall ----------------------------------------------------------------------


def test_anyone_may_recall_and_gets_the_results_list(make_provider, daemon):
    provider = make_provider()
    provider.on_turn_start(1, "x", author_id="222", author_name="Maya", author_is_bot=False)
    daemon.set_response(
        "recall", 200, {"recall_id": "r1", "results": [recalled("Maya lives in Wellington.", id="m9")], "reranked": True}
    )
    result = json.loads(provider.handle_tool_call("memory_recall", {"query": "where does Maya live?"}))
    assert isinstance(result, list) and result[0]["id"] == "m9"
    assert set(result[0]) == {"id", "sentence", "kind", "window", "phase", "observed_at", "strength", "kept"}


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


def test_recall_sends_only_what_the_model_gave(make_provider, daemon):
    provider = make_provider()
    provider.handle_tool_call("memory_recall", {"query": "tea"})
    assert daemon.requests_for("recall")[0].body == {"query": "tea", "session_id": SESSION}


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


def test_an_unknown_tool_is_a_tool_error(make_provider):
    assert error_of(make_provider().handle_tool_call("memory_retain", {"text": "x"}))


def test_bad_arguments_are_a_tool_error_not_an_exception(make_provider, daemon):
    provider = make_provider()
    assert error_of(provider.handle_tool_call("memory_forget", {}))
    assert error_of(provider.handle_tool_call("memory_forget", {"ids": "m1"}))
    assert error_of(provider.handle_tool_call("memory_recall", {}))
    assert daemon.requests_for("forget") == daemon.requests_for("recall") == []
