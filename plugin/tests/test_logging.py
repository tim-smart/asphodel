"""ADR 0010's logging rule, as ``docs/logging.md`` states it for the plugin:
content (queries, user and assistant text, injections, sentences) appears
only at ``TRACE``, never at ``DEBUG`` or above."""

import logging

from conftest import SESSION, plugin, transcript

SECRETS = ["pineapple on pizza", "Wellington", "oolong", "Dr Rao", "Maya"]


def test_trace_is_below_debug():
    assert plugin.provider.TRACE < logging.DEBUG


def test_no_content_at_debug_or_above(make_provider, daemon, caplog):
    caplog.set_level(logging.DEBUG)
    provider = make_provider()
    provider.system_prompt_block()
    provider.prefetch("do I like pineapple on pizza?", session_id=SESSION)
    provider.on_turn_start(1, "x", author_id="222", author_name="Maya", author_is_bot=False)
    provider.handle_tool_call("memory_recall", {"query": "pineapple on pizza"})
    provider.handle_tool_call("memory_forget", {"ids": ["m1"]})
    user = "I moved to Wellington and I drink oolong."
    provider.sync_turn(user, "Noted, Dr Rao.", session_id=SESSION, messages=transcript(user, "Noted, Dr Rao."))
    daemon.set_response("turns", 500, {"error": "store"})
    provider.sync_turn(user, "again", session_id=SESSION, messages=transcript(user, "again", epoch=5.0))
    url = daemon.go_down()
    provider.prefetch("pineapple on pizza again", session_id=SESSION)
    provider.sync_turn(user, "down", session_id=SESSION, messages=transcript(user, "down", epoch=6.0))
    text = "\n".join(record.getMessage() for record in caplog.records if record.levelno >= logging.DEBUG)
    for secret in SECRETS:
        assert secret not in text


def test_failures_are_logged_at_warning_with_status_codes_only(make_provider, daemon, caplog):
    caplog.set_level(logging.DEBUG)
    provider = make_provider()
    daemon.set_response("prefetch", 503, {"error": "the daemon is starting"})
    provider.prefetch("pineapple on pizza", session_id=SESSION)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings
    assert any("503" in r.getMessage() for r in warnings)
    assert not any("pineapple" in r.getMessage() for r in caplog.records if r.levelno >= logging.DEBUG)
