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


def test_text_without_a_marker_is_unchanged():
    assert strip_backfill("[Tim] Remind me to book the dentist.") == "[Tim] Remind me to book the dentist."
    assert strip_backfill("plain text") == "plain text"


def test_the_last_marker_wins_when_history_itself_contains_one():
    text = "[Jo] someone wrote [New message] in chat\n[New message]\nreal message"
    assert strip_backfill(text) == "real message"


def test_marker_with_no_text_after_it_leaves_an_empty_message():
    assert strip_backfill("history\n[New message]\n") == ""


def test_sync_turn_sends_the_stripped_text(make_provider, daemon):
    provider = make_provider()
    provider.sync_turn(BACKFILL, "Booked.", session_id=SESSION, messages=transcript(BACKFILL, "Booked."))
    body = daemon.requests_for("turns")[0].body
    assert body["user_text"] == "[Tim] Remind me to book the dentist."
    assert "Wellington" not in str(body)


def test_spooled_turns_are_stripped_too(make_provider, daemon, hermes_home):
    url = daemon.go_down()
    provider = make_provider(url=url)
    provider.sync_turn(BACKFILL, "Booked.", session_id=SESSION, messages=transcript(BACKFILL, "Booked."))
    files = list((hermes_home / "asphodel" / "spool").glob("*.json"))
    assert len(files) == 1
    assert "Wellington" not in files[0].read_text()
