"""``sync_turn``: the Turn
body, the primary gate, the ``ingest`` switch and the echoed ``recall_id``."""

from datetime import datetime

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


def test_message_at_is_this_turns_user_row_epoch_in_utc(make_provider, daemon):
    provider = make_provider()
    earlier = transcript("yesterday's message", "ok", epoch=EPOCH - 86400)
    sync(provider, messages=transcript("today", "ok", epoch=EPOCH, earlier=earlier))
    parsed = datetime.fromisoformat(daemon.requests_for("turns")[0].body["message_at"].replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.timestamp() == pytest.approx(EPOCH, abs=0.001)


BACKFILL = (
    "[Recent channel history]\n"
    "[Maya] I'm moving to Wellington next month.\n"
    "[Jo] Congratulations!\n"
    "\n[New message]\n"
    "[Tim] Remind me to book the dentist."
)


@pytest.mark.parametrize(
    "user, sent",
    [
        (BACKFILL, "[Tim] Remind me to book the dentist."),
        ("[Jo] someone wrote [New message] in chat\n[New message]\nreal message", "real message"),
    ],
    ids=["backfill", "last-marker-wins"],
)
def test_the_backfill_before_the_new_message_marker_is_stripped(make_provider, daemon, user, sent):
    """The gateway puts other people's recent messages and a ``[New
    message]`` marker in front of the user text; the ``[Name] `` prefix of a
    shared thread stays."""
    provider = make_provider()
    sync(provider, user=user, messages=transcript(user, "Booked."))
    body = daemon.requests_for("turns")[0].body
    assert body["user_text"] == sent
    assert "Wellington" not in str(body) and "someone wrote" not in str(body)


@pytest.mark.parametrize(
    "messages, requested",
    [
        (transcript("forget that", "Done.", tool_calls=[("memory_recall", {"query": "x"}), ("memory_forget", {"ids": ["m1"]})]), True),
        (transcript("forget that", "Done.", tool_calls=[("memory_recall", {"query": "x"})]), False),
        (
            transcript(
                "thanks, what's next?",
                "Nothing.",
                epoch=2000.0,
                earlier=transcript("forget that", "Done.", epoch=1000.0, tool_calls=[("memory_forget", {"ids": ["m1"]})]),
            ),
            False,
        ),
    ],
    ids=["forget-among-other-calls", "no-forget", "forget-in-an-earlier-turn"],
)
def test_forget_requested_marks_a_turn_that_called_memory_forget(make_provider, daemon, messages, requested):
    provider = make_provider()
    sync(provider, messages=messages)
    assert daemon.requests_for("turns")[0].body["forget_requested"] is requested


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


@pytest.mark.parametrize(
    "setup",
    [dict(init={"agent_context": "cron"}), dict(ingest=False)],
    ids=["not-a-primary-agent", "ingest-off"],
)
def test_nothing_is_ingested_but_recall_still_works(make_provider, daemon, setup):
    provider = make_provider(**setup)
    sync(provider)
    assert daemon.requests_for("turns") == []
    assert provider.prefetch("tea?", session_id=SESSION) != ""


# -- recall ids across overlapping turns -----------------------------------------
# Hermes syncs on a background worker, so the next turn's prefetch can run
# before this turn's sync.


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
