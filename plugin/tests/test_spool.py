"""The spool: when the daemon is down ``sync_turn`` keeps one file per turn
under ``$HERMES_HOME/asphodel/spool/``, replayed oldest first after the next
2xx, capped at about 10 MB or 7 days."""

import os
import time

import pytest

from conftest import SESSION, transcript


def spool_files(hermes_home):
    return list((hermes_home / "asphodel" / "spool").glob("*.json"))


def sync(provider, text, epoch, session=SESSION):
    provider.sync_turn(text, "ok", session_id=session, messages=transcript(text, "ok", epoch=epoch))


def spool(provider, hermes_home, text, *, epoch, ago, session=SESSION):
    """Syncs a turn the daemon can't take, then dates its spool file ``ago``
    seconds back: files written within one test can share an mtime."""
    before = set(spool_files(hermes_home))
    sync(provider, text, epoch, session)
    [path] = set(spool_files(hermes_home)) - before
    stamp = time.time() - ago
    os.utime(path, (stamp, stamp))


def delivered(daemon, since=0):
    return [r.body["user_text"] for r in daemon.requests_for("turns")[since:]]


def test_turns_spooled_while_the_daemon_is_down_are_delivered_oldest_first(make_provider, daemon, hermes_home, clock):
    provider = make_provider()
    daemon.drop_connections("turns", 100)
    # Written out of order: the order is the files' age, not the write order.
    spool(provider, hermes_home, "third", epoch=3.0, ago=10)
    spool(provider, hermes_home, "first", epoch=1.0, ago=30, session="telegram:-100/abc def")
    spool(provider, hermes_home, "second", epoch=2.0, ago=20)
    # A repeat of a spooled turn is the same file, so it's delivered once.
    sync(provider, "third", 3.0)
    (hermes_home / "asphodel" / "spool" / "junk.json").write_text("{nope")
    assert len(spool_files(hermes_home)) == 4

    daemon.drops.clear()
    clock.advance(31)
    mark = len(daemon.requests_for("turns"))
    sync(provider, "back", 4.0)
    assert delivered(daemon, mark) == ["back", "first", "second", "third"]
    assert daemon.requests_for("turns")[mark + 1].body["session_id"] == "telegram:-100/abc def"
    # Delivered turns are deleted, and so is the unreadable file.
    assert spool_files(hermes_home) == []


def test_a_5xx_spools_the_turn_but_a_4xx_drops_it(make_provider, daemon):
    provider = make_provider()
    daemon.set_response("turns", 503, {"error": "draining"})
    sync(provider, "one", 1.0)
    daemon.set_response("turns", 400, {"error": "invalid timezone"})
    sync(provider, "two", 2.0)
    del daemon.responses["turns"]
    sync(provider, "three", 3.0)
    assert delivered(daemon) == ["one", "two", "three", "one"]


@pytest.mark.parametrize(
    "failure, kept",
    [("drop", ["second", "third"]), (503, ["second", "third"]), (400, [])],
    ids=["connection-failure", "5xx", "4xx"],
)
def test_a_failed_replay_keeps_the_rest_unless_the_daemon_rejects_the_turn(make_provider, daemon, hermes_home, failure, kept):
    provider = make_provider()
    daemon.set_response("turns", 503, {"error": "draining"})
    for text, ago in (("first", 30), ("second", 20), ("third", 10)):
        spool(provider, hermes_home, text, epoch=float(ago), ago=ago)
    pending = [failure]

    def answer(request):
        text = request.body["user_text"]
        if pending and failure == "drop" and text == "first":
            daemon.drop_connections("turns", 1)  # the next request: "second"
            pending.clear()
        elif pending and text == "second":
            pending.clear()
            return failure, {"error": "refused"}
        return 200, {}

    daemon.set_handler("turns", answer)
    mark = len(daemon.requests_for("turns"))
    sync(provider, "back", 4.0)
    assert delivered(daemon, mark)[:3] == ["back", "first", "second"]
    mark = len(daemon.requests_for("turns"))
    sync(provider, "next", 5.0)
    assert delivered(daemon, mark) == ["next", *kept]


def test_turns_spooled_over_a_week_ago_are_dropped_not_delivered(make_provider, daemon, hermes_home):
    provider = make_provider()
    daemon.set_response("turns", 503, {"error": "draining"})
    spool(provider, hermes_home, "stale", epoch=1.0, ago=8 * 24 * 3600)
    del daemon.responses["turns"]
    mark = len(daemon.requests_for("turns"))
    sync(provider, "fresh", 2.0)
    assert delivered(daemon, mark) == ["fresh"]
    assert spool_files(hermes_home) == []


def test_a_spool_over_its_size_cap_drops_the_oldest_turns(make_provider, daemon, hermes_home):
    provider = make_provider()
    daemon.set_response("turns", 503, {"error": "draining"})
    # 1 MB turns, well past the spool's ~10 MB cap.
    sent = [f"{n:02d}" for n in range(25)]
    for n, label in enumerate(sent):
        spool(provider, hermes_home, label + "x" * 1_000_000, epoch=float(n), ago=100 - n)
    del daemon.responses["turns"]
    mark = len(daemon.requests_for("turns"))
    sync(provider, "back", 100.0)
    replayed = [text[:2] for text in delivered(daemon, mark + 1)]
    assert replayed and len(replayed) < len(sent)
    assert replayed == sent[-len(replayed):]
