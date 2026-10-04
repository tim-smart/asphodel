"""Backfill: Hermes' history in ``$HERMES_HOME/state.db`` posted to the
daemon, so Asphodel starts out knowing what Hermes said before the switch.

It reads ``state.db`` with the rules of ``asphodel import``
(``crates/asphodel/src/replay/import.rs``), from an online-backup copy so the
live writer is never read mid-transaction:

- a schema version other than 30 or 31 is refused, as is a changed column;
- only rows with ``active = 1 OR compacted = 1`` are read, in timestamp order;
- each user message is paired with the final assistant reply, and tool rows
  and assistant rows that only call tools are skipped;
- a ``_compressed_summary`` row is skipped and becomes a session clear at its
  time;
- cron sessions and subagent sessions (``parent_session_id`` set) post
  nothing;
- the injected memory block is cut from the text, multimodal content is
  reduced to its text, backfilled channel history is stripped, and a
  ``[Name] `` prefix naming a ``--speaker`` makes that speaker the author.

Each turn goes through ``POST /v1/banks/{bank}/turns`` with its original
``message_at``, so its memories arrive with their real age. The daemon
dedupes turns, so a rerun posts nothing twice and resumes after an error.

Run it from the plugin directory::

    HERMES_HOME=~/.hermes python3 backfill.py --since 2026-01-01 [--dry-run]

It prints the counts ``asphodel import --dry-run`` prints for the same
database, with ``--since`` how many turns are older than the cutoff, and,
unless it's a dry run, how many turns were stored and how many the daemon
already had. Exit 0 when done, 1 when the daemon stopped it, 2
when it refused to start. Nothing it prints holds message text.
"""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    # Run as a script: load this directory as a package without running its
    # ``__init__``, which needs Hermes, so the relative imports below work.
    import importlib
    import sys
    import types
    from pathlib import Path

    _package = types.ModuleType("asphodel_backfill")
    _package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[_package.__name__] = _package
    sys.exit(importlib.import_module(f"{_package.__name__}.backfill").main(sys.argv[1:]))

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import config as plugin_config
from . import turns
from .client import DaemonClient, DaemonError, DaemonUnavailable

#: The Hermes schema versions these rules were written against.
HERMES_SCHEMA_VERSIONS = (30, 31)
#: The columns read, per table, with the type Hermes' DDL declares.
REQUIRED_COLUMNS = (
    ("sessions", (("id", "TEXT"), ("source", "TEXT"), ("parent_session_id", "TEXT"), ("started_at", "REAL"))),
    (
        "messages",
        (
            ("id", "INTEGER"),
            ("session_id", "TEXT"),
            ("role", "TEXT"),
            ("content", "TEXT"),
            ("timestamp", "REAL"),
            ("active", "INTEGER"),
            ("compacted", "INTEGER"),
            ("_compressed_summary", "INTEGER"),
            ("tool_calls", "TEXT"),
        ),
    ),
    ("schema_version", (("version", "INTEGER"),)),
)
#: How Hermes stores multimodal content: this prefix and a JSON parts list.
MULTIMODAL_PREFIX = "\x00json:"
#: The session ``source`` Hermes gives cron runs.
CRON_SOURCE = "cron"
#: The fence of the memory block a provider injects into the user message.
#: Hindsight's is the default, and historical user rows carry it.
MEMORY_BLOCK = ("<memory-context>", "</memory-context>")
SPEAKER_PREFIX = re.compile(r"^\[([^\]\n]+)\] ")
#: Per request. Generous: the daemon may be busy extracting.
TURN_TIMEOUT = 30.0
#: A progress line on stderr every this many turns.
PROGRESS_EVERY = 100
COUNT_KEYS = (
    "primary_sessions",
    "cron_sessions",
    "subagent_sessions_skipped",
    "turns",
    "prefetch_only_turns",
    "compactions",
    "summary_rows_skipped",
    "tool_rows_skipped",
    "inactive_rows_skipped",
    "multimodal_rows",
    "memory_blocks_stripped",
    "image_parts_dropped",
    "backfills_stripped",
    "non_owner_turns",
)

EXIT_STOPPED = 1
EXIT_REFUSED = 2


class Refused(Exception):
    """The backfill won't start. The message names what's wrong, never
    message text."""


@dataclass
class Event:
    """A turn to post or a clear, at the user message's time or the summary
    row's."""

    at: float
    seq: int
    session: str
    turn: Optional[Dict[str, Any]] = None


@dataclass
class History:
    events: List[Event] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=lambda: dict.fromkeys(COUNT_KEYS, 0))


def main(argv: Optional[List[str]] = None, *, hermes_home: Optional[str] = None) -> int:
    parser = argparse.ArgumentParser(prog="backfill.py", description="Post Hermes' history in state.db to Asphodel.")
    parser.add_argument("--since", metavar="YYYY-MM-DD", help="post turns from this date on, in the bank's timezone")
    parser.add_argument("--all-history", action="store_true", help="post every turn, however old")
    parser.add_argument("--dry-run", action="store_true", help="print the counts and post nothing")
    parser.add_argument(
        "--speaker",
        action="append",
        default=[],
        metavar="NAME=PLATFORM:ID",
        help="a speaker other than the owner, named by a [Name] prefix; repeatable",
    )
    parser.add_argument("--bank", help="the bank; defaults to the plugin config's")
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:
        return stop.code if isinstance(stop.code, int) else EXIT_REFUSED
    home = Path(hermes_home or os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    try:
        return _run(args, home)
    except Refused as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_REFUSED


def _run(args: argparse.Namespace, home: Path) -> int:
    if args.since and args.all_history:
        raise Refused("give --since or --all-history, not both")
    if not args.since and not args.all_history:
        raise Refused(
            "choose how far back to go: --since YYYY-MM-DD or --all-history. "
            "Old memories arrive already faded, so this is a decision, not a default."
        )
    config = plugin_config.load_config(home)
    since = _since(args.since, config.timezone) if args.since else None
    speakers = _speakers(args.speaker)
    bank = args.bank or config.bank
    if not args.dry_run and not bank:
        raise Refused("no bank: set bank in the plugin config or pass --bank")

    history = _read(home / "state.db", home, speakers=speakers, timezone_name=config.timezone)
    counts = dict(history.counts)
    events = [event for event in history.events if since is None or event.at >= since]
    if since is not None:
        counts["turns_before_since"] = counts["turns"] - sum(1 for event in events if event.turn is not None)
    if args.dry_run:
        print(json.dumps(counts, indent=2))
        return 0

    client = DaemonClient(config.url, token=config.token)
    stored = duplicates = 0
    total = sum(1 for event in events if event.turn is not None)
    try:
        identity = {
            "owner_name": config.owner_name,
            "owner_platform_ids": list(config.owner_platform_ids),
            "assistant_name": config.assistant_name,
            "timezone": config.timezone,
        }
        client.put_bank(bank, {k: v for k, v in identity.items() if v not in (None, [])}, timeout=TURN_TIMEOUT)
    except (DaemonUnavailable, DaemonError) as error:
        print(f"error: setting up bank {bank} failed: {error}", file=sys.stderr)
        return EXIT_STOPPED
    for event in events:
        try:
            if event.turn is None:
                client.clear_session(bank, event.session, timeout=TURN_TIMEOUT)
                continue
            reply = client.ingest_turn(bank, event.turn, timeout=TURN_TIMEOUT)
        except (DaemonUnavailable, DaemonError) as error:
            what = "the clear" if event.turn is None else "the turn"
            print(
                f"error: stopped at {what} in session {event.session} at {turns.epoch_to_rfc3339(event.at)}: "
                f"{error}. {stored} stored and {duplicates} already there before it; "
                "rerun the same command to resume.",
                file=sys.stderr,
            )
            return EXIT_STOPPED
        if isinstance(reply, dict) and reply.get("outcome") == "duplicate":
            duplicates += 1
        else:
            stored += 1
        done = stored + duplicates
        if done % PROGRESS_EVERY == 0:
            print(f"posted {done} of {total} turns", file=sys.stderr)
    counts.update(stored=stored, duplicates=duplicates)
    print(json.dumps(counts, indent=2))
    return 0


def _since(text: str, timezone_name: Optional[str]) -> float:
    """The start of ``text``'s day in the bank's timezone, or UTC without
    one, as epoch seconds."""
    try:
        day = date.fromisoformat(text)
    except ValueError:
        raise Refused(f"--since takes a date as YYYY-MM-DD, not {text!r}") from None
    zone: Any = timezone.utc
    if timezone_name:
        try:
            zone = ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError):
            raise Refused(f"the configured timezone {timezone_name!r} isn't one") from None
    return datetime(day.year, day.month, day.day, tzinfo=zone).timestamp()


def _speakers(values: List[str]) -> Dict[str, str]:
    speakers = {}
    for value in values:
        name, _, speaker_id = value.partition("=")
        if not name.strip() or ":" not in speaker_id:
            raise Refused(f"--speaker takes NAME=PLATFORM:ID, not {value!r}")
        speakers[name.strip()] = speaker_id.strip()
    return speakers


def _read(path: Path, home: Path, *, speakers: Dict[str, str], timezone_name: Optional[str]) -> History:
    """Reads an online-backup copy of ``path``, kept next to the plugin's
    config and removed afterwards."""
    try:
        source = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True)
    except sqlite3.Error:
        raise Refused(f"can't open {path}") from None
    scratch = plugin_config.config_path(home).parent
    scratch.mkdir(parents=True, exist_ok=True)
    handle, copy_path = tempfile.mkstemp(prefix=".backfill-", suffix=".db", dir=scratch)
    os.close(handle)
    try:
        try:
            copy = sqlite3.connect(copy_path)
            try:
                source.backup(copy)
            except sqlite3.Error:
                raise Refused(f"can't read {path}") from None
        finally:
            source.close()
        try:
            _check_schema(copy, path)
            return _walk(copy, speakers=speakers, timezone_name=timezone_name)
        finally:
            copy.close()
    finally:
        os.unlink(copy_path)


def _check_schema(conn: sqlite3.Connection, path: Path) -> None:
    """Every column read is there with its declared type, and the version is
    one these rules know. Everything wrong is listed at once."""
    problems = []
    for table, columns in REQUIRED_COLUMNS:
        present = {row[1]: row[2] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not present:
            problems.append(f"the table {table} is missing")
            continue
        for column, declared in columns:
            found = present.get(column)
            if found is None:
                problems.append(f"the column {table}.{column} is missing")
            elif found.upper() != declared:
                problems.append(f"the column {table}.{column} is declared {found!r}; the backfill was written against {declared}")
    if not any("schema_version" in problem for problem in problems):
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        if version is None:
            problems.append("the schema_version table has no version")
        elif version not in HERMES_SCHEMA_VERSIONS:
            written = ", ".join(str(v) for v in HERMES_SCHEMA_VERSIONS)
            problems.append(f"the schema version is {version}; the backfill was written against {written}")
    if problems:
        raise Refused(f"{path} isn't the Hermes state.db the backfill was written against:\n" + "\n".join(problems))


def _walk(conn: sqlite3.Connection, *, speakers: Dict[str, str], timezone_name: Optional[str]) -> History:
    history = History()
    counts = history.counts
    seq = 0
    sessions = conn.execute("SELECT id, source, parent_session_id FROM sessions ORDER BY started_at, id").fetchall()
    for session, source, parent in sessions:
        if parent is not None:
            counts["subagent_sessions_skipped"] += 1
            continue
        cron = source == CRON_SOURCE
        counts["cron_sessions" if cron else "primary_sessions"] += 1
        counts["inactive_rows_skipped"] += conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ? AND active = 0 AND compacted = 0", (session,)
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT role, content, timestamp, _compressed_summary, tool_calls FROM messages "
            "WHERE session_id = ? AND (active = 1 OR compacted = 1) ORDER BY timestamp, id",
            (session,),
        ).fetchall()

        def emit(open_turn: Optional[Tuple[tuple, Optional[tuple]]]) -> None:
            nonlocal seq
            if open_turn is None:
                return
            user, assistant = open_turn
            user_text, author = _user_text(user[1] or "", speakers, counts)
            if assistant is None or cron:
                counts["prefetch_only_turns"] += 1
                return
            assistant_text = _plain_text(assistant[1] or "", counts)
            if author is not None:
                counts["non_owner_turns"] += 1
            counts["turns"] += 1
            at = _epoch(user[2])
            turn_author, platform = _author(author, source)
            seq += 1
            history.events.append(
                Event(
                    at=at,
                    seq=seq,
                    session=session,
                    turn={
                        "session_id": session,
                        "message_at": turns.epoch_to_rfc3339(at),
                        "timezone": timezone_name,
                        "user_text": user_text,
                        "assistant_text": assistant_text,
                        "author": turn_author,
                        "platform": platform,
                        "recall_id": None,
                        "forget_requested": False,
                    },
                )
            )

        open_turn = None
        for row in rows:
            role, _, stamp, summary, tool_calls = row
            if summary:
                # Hermes replaced the turns before here with this summary, and
                # the session's context started over.
                emit(open_turn)
                open_turn = None
                counts["compactions"] += 1
                counts["summary_rows_skipped"] += 1
                if not cron:
                    seq += 1
                    history.events.append(Event(at=_epoch(stamp), seq=seq, session=session))
                continue
            if role == "user":
                emit(open_turn)
                open_turn = (row, None)
            elif role == "assistant":
                if _has_tool_calls(tool_calls):
                    counts["tool_rows_skipped"] += 1
                elif open_turn is not None:
                    open_turn = (open_turn[0], row)
            else:
                counts["tool_rows_skipped"] += 1
        emit(open_turn)
    history.events.sort(key=lambda event: (event.at, event.seq))
    return history


def _author(speaker: Optional[Tuple[str, str]], platform: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The turn's ``author`` and ``platform``. The daemon looks a speaker up
    as ``<platform>:<author.id>``, so a speaker on the session's platform is
    sent by their platform-local id, and one on another platform by their
    whole speaker id with no platform."""
    if speaker is None:
        return None, platform
    name, local = speaker
    if platform and local.startswith(f"{platform}:"):
        return {"id": local[len(platform) + 1:], "name": name, "is_bot": False}, platform
    return {"id": local, "name": name, "is_bot": False}, None


def _user_text(content: str, speakers: Dict[str, str], counts: Dict[str, int]) -> Tuple[str, Optional[Tuple[str, str]]]:
    """The user message as ``sync_turn`` sends it, and the ``(name, speaker
    id)`` its prefix names, if a known speaker."""
    text = _plain_text(content, counts)
    if turns.NEW_MESSAGE_MARKER in text:
        counts["backfills_stripped"] += 1
        text = turns.strip_backfill(text)
    match = SPEAKER_PREFIX.match(text)
    if match and match.group(1) in speakers:
        return text, (match.group(1), speakers[match.group(1)])
    return text, None


def _plain_text(content: str, counts: Dict[str, int]) -> str:
    """A row's text: multimodal content reduced to its text parts, and in
    either case the memory block cut out."""
    if content.startswith(MULTIMODAL_PREFIX):
        counts["multimodal_rows"] += 1
        try:
            parts = json.loads(content[len(MULTIMODAL_PREFIX):])
        except ValueError:
            raise Refused("a multimodal row's parts aren't JSON") from None
        if not isinstance(parts, list):
            raise Refused("a multimodal row's parts aren't a list")
        texts = []
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                if isinstance(part.get("text"), str):
                    texts.append(part["text"])
            else:
                counts["image_parts_dropped"] += 1
        content = "\n".join(texts)
    return _strip_memory_block(content, counts)


def _strip_memory_block(text: str, counts: Dict[str, int]) -> str:
    """Cuts every memory block out. A start with no end cuts to the end, since
    injected memory must never become the user's words."""
    start, end = MEMORY_BLOCK
    out = []
    rest = text
    while (index := rest.find(start)) >= 0:
        counts["memory_blocks_stripped"] += 1
        out.append(rest[:index])
        after = rest[index + len(start):]
        close = after.find(end)
        if close < 0:
            rest = ""
            break
        rest = after[close + len(end):]
    out.append(rest)
    return "".join(out).strip()


def _has_tool_calls(tool_calls: Optional[str]) -> bool:
    """Whether an assistant row only calls tools: ``tool_calls`` is a
    non-empty JSON array."""
    if tool_calls is None:
        return False
    try:
        calls = json.loads(tool_calls)
    except ValueError:
        return bool(tool_calls.strip())
    if isinstance(calls, list):
        return bool(calls)
    return calls is not None


def _epoch(seconds: Any) -> float:
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        raise Refused("a message timestamp isn't a number")
    return float(seconds)
