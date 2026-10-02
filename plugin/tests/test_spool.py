"""The spool (TIM-94, decision 6; ADR 0006): one JSON file per turn under
``$HERMES_HOME/asphodel/spool/`` when the daemon is down, written by rename,
replayed after the next 2xx, capped at about 10 MB or 7 days."""

import json
import os
import time

import pytest

from conftest import SESSION, plugin, transcript

Spool = plugin.spool.Spool


def spool_dir(hermes_home):
    return hermes_home / "asphodel" / "spool"


def turn(n: int, size: int = 0) -> dict:
    return {
        "session_id": SESSION,
        "message_at": f"2026-10-02T02:19:{n:02d}Z",
        "timezone": None,
        "user_text": "x" * size,
        "assistant_text": "",
        "author": None,
        "platform": "cli",
        "recall_id": None,
        "forget_requested": False,
    }


# -- the Spool class ------------------------------------------------------------


def test_write_creates_the_directory_and_one_file_per_turn(tmp_path):
    spool = Spool(tmp_path / "spool")
    spool.write(turn(1))
    spool.write(turn(2))
    files = spool.files()
    assert len(files) == 2
    assert all(f.suffix == ".json" for f in files)
    assert json.loads(files[0].read_text()) == turn(1)


def test_file_name_is_the_source_id_so_a_repeat_overwrites(tmp_path):
    spool = Spool(tmp_path / "spool")
    spool.write(turn(1))
    spool.write(turn(1))
    assert len(spool.files()) == 1
    assert plugin.spool.spool_file_name(SESSION, "2026-10-02T02:19:01Z") == spool.files()[0].name


def test_file_name_is_safe_for_any_session_id(tmp_path):
    name = plugin.spool.spool_file_name("telegram:-100/abc def", "2026-10-02T02:19:01Z")
    assert "/" not in name and " " not in name and ":" not in name


def test_written_by_rename_leaves_no_partial_files(tmp_path):
    spool = Spool(tmp_path / "spool")
    spool.write(turn(1))
    names = os.listdir(tmp_path / "spool")
    assert names == [spool.files()[0].name]


def test_files_are_oldest_first(tmp_path):
    spool = Spool(tmp_path / "spool")
    recent = time.time() - 100
    for n in (3, 1, 2):
        path = spool.write(turn(n))
        os.utime(path, (recent + n, recent + n))
    assert [json.loads(f.read_text())["message_at"] for f in spool.files()] == [
        turn(1)["message_at"],
        turn(2)["message_at"],
        turn(3)["message_at"],
    ]


def test_size_cap_drops_the_oldest(tmp_path):
    spool = Spool(tmp_path / "spool", max_bytes=2500)
    recent = time.time() - 100
    for n in range(1, 6):
        path = spool.write(turn(n, size=800))
        os.utime(path, (recent + n, recent + n))
    kept = [json.loads(f.read_text())["message_at"] for f in spool.files()]
    assert sum(f.stat().st_size for f in spool.files()) <= 2500
    assert kept == [turn(4)["message_at"], turn(5)["message_at"]]


def test_age_cap_drops_files_older_than_seven_days(tmp_path):
    spool = Spool(tmp_path / "spool")
    old = spool.write(turn(1))
    stale = time.time() - 8 * 24 * 3600
    os.utime(old, (stale, stale))
    spool.write(turn(2))
    assert [json.loads(f.read_text())["message_at"] for f in spool.files()] == [turn(2)["message_at"]]


def test_replay_sends_oldest_first_and_deletes_delivered(tmp_path):
    spool = Spool(tmp_path / "spool")
    recent = time.time() - 100
    for n in (1, 2, 3):
        path = spool.write(turn(n))
        os.utime(path, (recent + n, recent + n))
    sent = []
    delivered = spool.replay(lambda body: sent.append(body["message_at"]) or True)
    assert delivered == 3
    assert sent == [turn(n)["message_at"] for n in (1, 2, 3)]
    assert spool.files() == []


def test_replay_stops_at_the_first_connection_failure_and_keeps_the_rest(tmp_path):
    spool = Spool(tmp_path / "spool")
    recent = time.time() - 100
    for n in (1, 2, 3):
        path = spool.write(turn(n))
        os.utime(path, (recent + n, recent + n))
    calls = []

    def send(body):
        calls.append(body)
        if len(calls) == 2:
            raise plugin.client.DaemonUnavailable("gone again")
        return True

    assert spool.replay(send) == 1
    assert len(spool.files()) == 2


def test_replay_drops_a_turn_the_daemon_rejects(tmp_path):
    spool = Spool(tmp_path / "spool")
    spool.write(turn(1))
    assert spool.replay(lambda body: False) == 0
    assert spool.files() == []


def test_replay_of_an_unreadable_file_drops_it(tmp_path):
    spool = Spool(tmp_path / "spool")
    (tmp_path / "spool").mkdir()
    (tmp_path / "spool" / "junk.json").write_text("{nope")
    assert spool.replay(lambda body: True) == 0
    assert spool.files() == []


def test_write_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    Spool(blocker / "spool").write(turn(1))


# -- through the provider --------------------------------------------------------


def test_sync_turn_spools_when_the_daemon_is_down(make_provider, daemon, hermes_home):
    url = daemon.go_down()
    provider = make_provider(url=url)
    messages = transcript("I moved to Wellington.", "Noted.")
    provider.sync_turn("I moved to Wellington.", "Noted.", session_id=SESSION, messages=messages)
    files = list(spool_dir(hermes_home).glob("*.json"))
    assert len(files) == 1
    body = json.loads(files[0].read_text())
    assert body == provider.build_turn(
        "I moved to Wellington.", "Noted.", session_id=SESSION, messages=messages, turn_author=None
    )


def test_sync_turn_spools_on_a_5xx_but_not_a_4xx(make_provider, daemon, hermes_home):
    provider = make_provider()
    daemon.set_response("turns", 503, {"error": "draining"})
    provider.sync_turn("one", "ok", session_id=SESSION, messages=transcript("one", "ok", epoch=1.0))
    daemon.set_response("turns", 400, {"error": "invalid timezone"})
    provider.sync_turn("two", "ok", session_id=SESSION, messages=transcript("two", "ok", epoch=2.0))
    files = list(spool_dir(hermes_home).glob("*.json"))
    assert [json.loads(f.read_text())["user_text"] for f in files] == ["one"]


def test_the_next_2xx_replays_the_spool(hermes_home, clock, warnings):
    from conftest import FAST, write_config
    from fake_daemon import FakeDaemon

    daemon = FakeDaemon().start()
    url = daemon.url
    daemon.stop()
    write_config(hermes_home, url=url, owner_platform_ids=["discord:111"])
    provider = plugin.AsphodelMemoryProvider(timeouts=FAST, clock=clock)
    provider.initialize(SESSION, hermes_home=str(hermes_home), platform="cli", agent_context="primary", agent_identity="tim")
    provider.sync_turn("while down", "ok", session_id=SESSION, messages=transcript("while down", "ok", epoch=1.0))
    assert len(list(spool_dir(hermes_home).glob("*.json"))) == 1

    # Listen again on the same port the provider was configured with.
    port = int(url.rsplit(":", 1)[1])
    restarted = _restart(FakeDaemon(), port)
    try:
        clock.advance(60)
        provider.sync_turn("back up", "ok", session_id=SESSION, messages=transcript("back up", "ok", epoch=2.0))
        texts = [r.body["user_text"] for r in restarted.requests_for("turns")]
        assert texts == ["back up", "while down"]
        assert list(spool_dir(hermes_home).glob("*.json")) == []
    finally:
        restarted.stop()


def _restart(daemon, port):
    """Starts a fake daemon on a specific port (the one the provider was
    configured with)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _handle(self):
            daemon._handle(self)

        do_GET = do_POST = do_PUT = _handle

    ThreadingHTTPServer.allow_reuse_address = True
    deadline = time.monotonic() + 5
    while True:
        try:
            daemon._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
    daemon._server.daemon_threads = True
    daemon._thread = threading.Thread(target=lambda: daemon._server.serve_forever(poll_interval=0.02), daemon=True)
    daemon._thread.start()
    return daemon
