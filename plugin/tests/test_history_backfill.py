"""The history backfill: ``backfill.main(argv)`` reads ``$HERMES_HOME/state.db``
and posts its turns to the daemon with their original times. It runs on the
fixture ``crates/asphodel/tests/plugin_fixture.rs`` checks in, and its counts
match what ``asphodel import --dry-run`` prints for the same database."""

import importlib
import json
import shutil
import sqlite3
import tomllib
import uuid
from datetime import datetime

import pytest
from conftest import PACKAGE_NAME, TESTS_DIR, write_config

FIXTURES = TESTS_DIR / "fixtures"
MANIFEST = tomllib.loads((FIXTURES / "manifest.toml").read_text())
IMPORT_COUNTS = json.loads((FIXTURES / "import-counts.json").read_text())


@pytest.fixture
def history(hermes_home, daemon):
    """``$HERMES_HOME`` holding the fixture ``state.db`` and a config with
    the manifest's owner, pointing at the fake daemon."""
    shutil.copy(FIXTURES / "state.db", hermes_home / "state.db")
    write_config(
        hermes_home,
        url=daemon.url,
        bank=MANIFEST["bank"],
        owner_name=MANIFEST["owner"]["name"],
        owner_platform_ids=MANIFEST["owner"]["platform_ids"],
        assistant_name=MANIFEST["assistant"],
        timezone=MANIFEST["timezone"],
    )
    return hermes_home


def run(*args):
    """``main`` with the manifest's speakers. Imported here, so a missing
    module fails these tests rather than the whole collection."""
    backfill = importlib.import_module(f"{PACKAGE_NAME}.backfill")
    speakers = [f"--speaker={s['name']}={s['id']}" for s in MANIFEST.get("speaker", ())]
    return backfill.main([*args, *speakers])


def at(text):
    return datetime.fromisoformat(text)


def posted(daemon):
    """The turns and clears in the order they were posted."""
    out = []
    for request in daemon.requests:
        if request.key == "turns":
            out.append((request.body["session_id"], at(request.body["message_at"]), request.body["user_text"]))
        elif request.key == "clear":
            out.append((request.session, "clear"))
    return out


def deduping(fail_on=None):
    """A turns handler that answers ``duplicate`` for a turn it has stored,
    as the daemon does, and 500 on the ``fail_on``th call."""
    seen, calls = set(), [0]

    def handle(request):
        calls[0] += 1
        if calls[0] == fail_on:
            return 500, {"error": "the store is busy"}
        body = request.body
        key = (body["session_id"], at(body["message_at"]), body["user_text"])
        outcome = "duplicate" if key in seen else "stored"
        seen.add(key)
        return 200, {
            "source": str(uuid.uuid4()),
            "outcome": outcome,
            "chunks_queued": 1 if outcome == "stored" else 0,
            "chunks_skipped": 0,
            "secret_kinds": [],
            "speaker": None,
        }

    return handle


def test_a_dry_run_prints_the_importers_counts_and_posts_nothing(history, daemon, capsys):
    assert run("--all-history", "--dry-run") == 0
    printed = json.loads(capsys.readouterr().out)
    assert {key: printed.get(key) for key in IMPORT_COUNTS} == IMPORT_COUNTS
    assert daemon.requests == []


def test_posts_every_owner_turn_at_its_original_time_in_order(history, daemon):
    """Only ``active = 1 OR compacted = 1`` rows: the archived turn posts,
    the carried tail once, at its original time. Cron and subagent sessions
    post nothing, and a compaction summary is a clear where it sits."""
    assert run("--all-history") == 0
    assert posted(daemon) == [
        ("s-main", at("2026-01-05T09:00:00Z"), "I live in Auckland, near the harbour."),
        ("s-main", at("2026-01-06T09:00:00Z"), "What should I cook tonight?"),
        ("s-main", at("2026-01-07T09:00:00Z"), "Remind me what the weather does in winter."),
        ("s-later", at("2026-01-10T09:00:00Z"), "Any plans for the weekend?"),
        ("s-compacted", at("2026-01-11T09:00:00Z"), "Archived question."),
        ("s-compacted", at("2026-01-11T09:10:00Z"), "Carried question."),
        ("s-compacted", "clear"),
        ("s-compacted", at("2026-01-11T09:30:00Z"), "After the compaction."),
        ("s-mixed", at("2026-01-12T09:00:00Z"), "What's on my calendar tomorrow?"),
        ("s-mixed", at("2026-01-12T09:10:00Z"), "I started learning the cello."),
        ("s-mixed", at("2026-01-12T09:20:00Z"), "[Sam] I'm Tim's friend from Wellington."),
        ("s-mixed", at("2026-01-12T09:30:00Z"), "[Bob] Is it raining?"),
        ("s-mixed", at("2026-01-12T09:40:00Z"), "[Sam] See you at noon."),
        ("s-mixed", at("2026-01-12T09:50:00Z"), "Look at my garden."),
    ]
    bodies = json.dumps([r.body for r in daemon.requests])
    for leaked in ("HINDSIGHT-INJECTED-MEMORY", "memory-context", "HERMES-COMPACTION-SUMMARY", "SENTINEL-API-CONTENT-5b20"):
        assert leaked not in bodies


def test_a_turn_posts_its_final_reply_not_its_tool_rows(history, daemon):
    assert run("--all-history") == 0
    calendar = next(r.body for r in daemon.requests_for("turns") if r.body["user_text"].startswith("What's on"))
    assert calendar["assistant_text"] == "A dentist appointment at ten."


def test_a_known_speakers_prefix_makes_them_the_author(history, daemon):
    """Sam is in the manifest, so Sam's turns are Sam's, with the platform
    id the daemon prefixes with the session's platform. Bob isn't, so his
    turn is the owner's, as the importer reads it."""
    assert run("--all-history") == 0
    authors = {r.body["user_text"]: (r.body["author"], r.body["platform"]) for r in daemon.requests_for("turns")}
    sam = ({"id": "2", "name": "Sam", "is_bot": False}, "discord")
    assert authors["[Sam] I'm Tim's friend from Wellington."] == sam
    assert authors["[Sam] See you at noon."] == sam
    assert authors["[Bob] Is it raining?"][0] is None
    assert authors["What should I cook tonight?"][0] is None


def test_since_posts_only_turns_from_that_date_on(history, daemon):
    assert run("--since", "2026-01-07") == 0
    times = [r.body["message_at"] for r in daemon.requests_for("turns")]
    assert len(times) == 11
    assert at(times[0]) == at("2026-01-07T09:00:00Z")


def test_refuses_to_run_without_a_cutoff_decision(history, daemon, capsys):
    assert run() == 2
    err = capsys.readouterr().err
    assert "--since" in err and "--all-history" in err
    assert daemon.requests == []


def test_refuses_a_schema_version_it_was_not_written_against(history, daemon, capsys):
    db = sqlite3.connect(history / "state.db")
    with db:
        db.execute("UPDATE schema_version SET version = 9999")
    db.close()
    assert run("--all-history") == 2
    assert "9999" in capsys.readouterr().err
    assert daemon.requests == []


def test_stops_at_the_first_daemon_error_and_a_rerun_resumes(history, daemon, capsys):
    daemon.set_handler("turns", deduping(fail_on=3))
    assert run("--all-history") == 1
    err = capsys.readouterr().err
    assert "s-main" in err and "2026-01-07" in err
    assert len(daemon.requests_for("turns")) == 3

    assert run("--all-history") == 0
    printed = json.loads(capsys.readouterr().out)
    assert (printed["stored"], printed["duplicates"]) == (11, 2)

    assert run("--all-history") == 0
    printed = json.loads(capsys.readouterr().out)
    assert (printed["stored"], printed["duplicates"]) == (0, 13)
