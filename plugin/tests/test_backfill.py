"""The backfill strip (TIM-96's amendment to TIM-94): on history backfill the
gateway puts other people's recent messages and a ``[New message]`` marker in
front of the user text. Everything before the marker goes; the ``[Name] ``
prefix of a shared thread stays."""

from conftest import SESSION, plugin, transcript

strip_backfill = plugin.turns.strip_backfill

BACKFILL = (
    "[Recent channel history]\n"
    "[Maya] I'm moving to Wellington next month.\n"
    "[Jo] Congratulations!\n"
    "\n[New message]\n"
    "[Tim] Remind me to book the dentist."
)


def test_strips_everything_before_the_marker():
    assert strip_backfill(BACKFILL) == "[Tim] Remind me to book the dentist."


def test_the_last_marker_wins_when_history_itself_contains_one():
    text = "[Jo] someone wrote [New message] in chat\n[New message]\nreal message"
    assert strip_backfill(text) == "real message"


def test_sync_turn_sends_the_stripped_text(make_provider, daemon):
    provider = make_provider()
    provider.sync_turn(BACKFILL, "Booked.", session_id=SESSION, messages=transcript(BACKFILL, "Booked."))
    body = daemon.requests_for("turns")[0].body
    assert body["user_text"] == "[Tim] Remind me to book the dentist."
    assert "Wellington" not in str(body)
