"""``sync_turn`` (TIM-94, decisions 1, 5 and 6; TIM-88 timestamps): the Turn
body, the primary gate, the ``ingest`` switch and the echoed ``recall_id``."""

from datetime import datetime, timezone

import pytest

from conftest import SESSION, transcript

EPOCH = 1759371557.622


def sync(provider, user="I drink oolong every morning.", assistant="Noted.", **kwargs):
    messages = kwargs.pop("messages", None) or transcript(user, assistant, epoch=EPOCH)
    provider.sync_turn(user, assistant, session_id=SESSION, messages=messages, **kwargs)


def test_posts_the_turn_body(make_provider, daemon, hermes):
    provider = make_provider()
    sync(provider, turn_author={"id": "111", "name": "Tim", "is_bot": False})
    request = daemon.requests_for("turns")[0]
    assert request.bank == "tim"
    body = request.body
    assert body["session_id"] == SESSION
    assert body["user_text"] == "I drink oolong every morning."
    assert body["assistant_text"] == "Noted."
    assert body["timezone"] == "Pacific/Auckland"
    assert body["platform"] == "discord"
    assert body["author"] == {"id": "111", "name": "Tim", "is_bot": False}
    assert body["recall_id"] is None
    assert body["forget_requested"] is False


def test_message_at_is_the_user_rows_epoch_as_rfc3339_utc(make_provider, daemon):
    provider = make_provider()
    sync(provider)
    message_at = daemon.requests_for("turns")[0].body["message_at"]
    assert message_at.endswith("Z")
    parsed = datetime.fromisoformat(message_at.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.timestamp() == pytest.approx(EPOCH, abs=0.001)


def test_message_at_comes_from_this_turns_user_row_not_an_earlier_one(make_provider, daemon):
    provider = make_provider()
    earlier = transcript("yesterday's message", "ok", epoch=EPOCH - 86400)
    sync(provider, messages=transcript("today", "ok", epoch=EPOCH, earlier=earlier))
    message_at = daemon.requests_for("turns")[0].body["message_at"]
    assert datetime.fromisoformat(message_at.replace("Z", "+00:00")).timestamp() == pytest.approx(EPOCH, abs=0.001)


def test_message_at_falls_back_to_the_wall_clock_without_messages(make_provider, daemon):
    provider = make_provider()
    before = datetime.now(timezone.utc).timestamp()
    provider.sync_turn("hi there friend", "hello", session_id=SESSION, messages=None)
    message_at = daemon.requests_for("turns")[0].body["message_at"]
    assert datetime.fromisoformat(message_at.replace("Z", "+00:00")).timestamp() >= before - 1


def test_no_author_is_sent_as_null(make_provider, daemon):
    provider = make_provider()
    sync(provider, turn_author=None)
    assert daemon.requests_for("turns")[0].body["author"] is None


def test_timezone_falls_back_to_config_then_null(make_provider, daemon, hermes):
    hermes.timezone_name = ""
    provider = make_provider(timezone="Europe/London")
    sync(provider)
    assert daemon.requests_for("turns")[0].body["timezone"] == "Europe/London"
    hermes.timezone_name = ""
    provider = make_provider()
    sync(provider)
    assert daemon.requests_for("turns")[-1].body["timezone"] is None


def test_echoes_the_recall_id_from_the_turns_prefetch_once(make_provider, daemon):
    provider = make_provider()
    daemon.set_response("prefetch", 200, {"recall_id": "r-1", "text": "x", "injected": ["m1"], "reranked": True})
    provider.prefetch("I drink oolong every morning.", session_id=SESSION)
    sync(provider)
    sync(provider)
    bodies = [r.body for r in daemon.requests_for("turns")]
    assert bodies[0]["recall_id"] == "r-1"
    assert bodies[1]["recall_id"] is None


def test_recall_id_is_per_session(make_provider, daemon):
    provider = make_provider()
    daemon.set_response("prefetch", 200, {"recall_id": "r-other", "text": "x", "injected": ["m1"], "reranked": True})
    provider.prefetch("tea?", session_id="sess-0002")
    sync(provider)
    assert daemon.requests_for("turns")[0].body["recall_id"] is None


def test_only_primary_agents_ingest(make_provider, daemon):
    provider = make_provider(init={"agent_context": "cron"})
    sync(provider)
    assert daemon.requests_for("turns") == []
    assert provider.prefetch("tea?", session_id=SESSION) != ""


def test_ingest_false_sends_nothing(make_provider, daemon):
    provider = make_provider(ingest=False)
    sync(provider)
    assert daemon.requests_for("turns") == []


def test_never_raises_on_a_daemon_error(make_provider, daemon):
    provider = make_provider()
    daemon.set_response("turns", 400, {"error": "invalid timezone"})
    sync(provider)


# -- recall ids across overlapping turns (TIM-99; Hermes syncs on a background
# worker, so the next turn's prefetch can run before this turn's sync) ----------


def _recall_id_per_query(daemon):
    daemon.set_handler(
        "prefetch",
        lambda request: (200, {"recall_id": f"r-{request.body['query']}", "text": "x", "injected": ["m1"], "reranked": True}),
    )


def _echoed(daemon):
    return [r.body["recall_id"] for r in daemon.requests_for("turns")]


def test_a_sync_after_the_next_prefetch_echoes_its_own_recall_id(make_provider, daemon):
    provider = make_provider()
    _recall_id_per_query(daemon)
    provider.prefetch("where is the dentist?", session_id=SESSION)
    provider.prefetch("and when is it?", session_id=SESSION)
    sync(provider, user="where is the dentist?", messages=transcript("where is the dentist?", "Noted.", epoch=EPOCH))
    sync(provider, user="and when is it?", messages=transcript("and when is it?", "Noted.", epoch=EPOCH + 60))
    assert _echoed(daemon) == ["r-where is the dentist?", "r-and when is it?"]


def test_an_interrupted_turns_recall_id_is_never_echoed(make_provider, daemon):
    """Hermes skips the sync of an interrupted turn. The next turn echoes its
    own id, and the interrupted turn's id is dropped rather than echoed later."""
    provider = make_provider()
    _recall_id_per_query(daemon)
    provider.prefetch("book the dentist", session_id=SESSION)
    provider.prefetch("book the dentist for Friday", session_id=SESSION)
    sync(provider, user="book the dentist for Friday", messages=transcript("book the dentist for Friday", "Done.", epoch=EPOCH))
    sync(provider, user="thanks a lot then", messages=transcript("thanks a lot then", "Any time.", epoch=EPOCH + 60))
    assert _echoed(daemon) == ["r-book the dentist for Friday", None]


def test_a_turn_without_a_prefetch_echoes_nothing(make_provider, daemon):
    provider = make_provider()
    _recall_id_per_query(daemon)
    provider.prefetch("where is the dentist?", session_id=SESSION)
    sync(provider, user="where is the dentist?", messages=transcript("where is the dentist?", "Noted.", epoch=EPOCH))
    sync(provider, user="/skip", messages=transcript("/skip", "ok", epoch=EPOCH + 60))
    assert _echoed(daemon) == ["r-where is the dentist?", None]


def test_an_unmatched_sync_echoes_nothing_and_leaves_the_pending_id(make_provider, daemon):
    provider = make_provider()
    _recall_id_per_query(daemon)
    provider.prefetch("where is the dentist?", session_id=SESSION)
    sync(provider, user="a photo of the clinic", messages=transcript("a photo of the clinic", "Nice.", epoch=EPOCH))
    sync(provider, user="where is the dentist?", messages=transcript("where is the dentist?", "Noted.", epoch=EPOCH + 60))
    assert _echoed(daemon) == [None, "r-where is the dentist?"]


def test_the_match_is_on_the_text_before_the_backfill_strip(make_provider, daemon):
    """Hermes prefetches with the same unstripped text it later syncs."""
    provider = make_provider()
    _recall_id_per_query(daemon)
    text = "[Maya] I'm moving to Wellington.\n[New message]\n[Tim] Remind me about the dentist."
    provider.prefetch(text, session_id=SESSION)
    sync(provider, user=text, messages=transcript(text, "Will do.", epoch=EPOCH))
    assert _echoed(daemon) == [f"r-{text}"]
