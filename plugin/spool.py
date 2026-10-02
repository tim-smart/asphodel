"""The turn spool.

When the daemon is down, ``sync_turn`` writes one JSON file per turn, the
exact ``POST /v1/banks/{bank}/turns`` body, under
``$HERMES_HOME/asphodel/spool/``. Files are named by the turn's source id
(session id and message time) and written by rename, so a reader never sees
a partial file. Any instance that gets a 2xx from the daemon replays the
directory oldest first and deletes what succeeded. The spool is capped at
about 10 MB or 7 days, dropping the oldest. Ingest is idempotent on the
source id, so a turn replayed twice is a duplicate, not a second source.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, List, Tuple

from .client import DaemonUnavailable

log = logging.getLogger(__name__)

SPOOL_MAX_BYTES = 10 * 1024 * 1024
SPOOL_MAX_AGE_S = 7 * 24 * 60 * 60.0


def spool_file_name(session_id: str, message_at: str) -> str:
    """The file name for a turn's source id: a stable digest of the session
    id and the message time, so the name is filesystem-safe whatever the
    session id contains."""
    digest = hashlib.sha256(f"{session_id}\0{message_at}".encode("utf-8")).hexdigest()
    return f"{digest[:32]}.json"


class Spool:
    def __init__(
        self,
        directory: str | os.PathLike,
        *,
        max_bytes: int = SPOOL_MAX_BYTES,
        max_age: float = SPOOL_MAX_AGE_S,
    ) -> None:
        self.directory = Path(directory)
        self.max_bytes = max_bytes
        self.max_age = max_age

    def write(self, turn: dict) -> Path:
        """Writes ``turn`` by rename, then enforces the caps. Never raises."""
        path = self.directory / spool_file_name(str(turn.get("session_id", "")), str(turn.get("message_at", "")))
        partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps(turn), encoding="utf-8")
            os.replace(partial, path)
        except OSError as error:
            log.warning("could not spool a turn: %s", type(error).__name__)
            try:
                partial.unlink()
            except OSError:
                pass
            return path
        dropped = self.enforce_caps()
        log.debug("spooled a turn (%d dropped by the caps)", dropped)
        return path

    def files(self) -> List[Path]:
        """Spooled turns, oldest first."""
        return [path for path, _ in self._stats()]

    def _stats(self) -> List[Tuple[Path, os.stat_result]]:
        try:
            paths = list(self.directory.glob("*.json"))
        except OSError:
            return []
        stats = []
        for path in paths:
            try:
                stats.append((path, path.stat()))
            except OSError:
                continue
        stats.sort(key=lambda item: (item[1].st_mtime, item[0].name))
        return stats

    def enforce_caps(self) -> int:
        """Deletes files older than ``max_age`` and then the oldest until the
        total is under ``max_bytes``. Returns how many were dropped."""
        dropped = 0
        cutoff = time.time() - self.max_age
        kept = []
        for path, stat in self._stats():
            if stat.st_mtime < cutoff:
                dropped += _unlink(path)
            else:
                kept.append((path, stat))
        total = sum(stat.st_size for _, stat in kept)
        for path, stat in kept:
            if total <= self.max_bytes:
                break
            dropped += _unlink(path)
            total -= stat.st_size
        if dropped:
            log.warning("spool caps dropped %d turns", dropped)
        return dropped

    def replay(self, send: Callable[[dict], bool]) -> int:
        """Sends each spooled turn oldest first. ``send`` returns True on a
        2xx (the file is deleted), False when the turn can never succeed
        (a 4xx: the file is dropped), and raises
        :class:`asphodel_plugin.client.DaemonUnavailable` to stop the replay
        and keep the rest. Returns how many were delivered."""
        # An outage's turns past the age cap are dropped, not delivered late.
        self.enforce_caps()
        delivered = 0
        dropped = 0
        for path in self.files():
            try:
                turn = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                # Another instance replayed it first.
                continue
            except (OSError, ValueError):
                dropped += _unlink(path)
                continue
            if not isinstance(turn, dict):
                dropped += _unlink(path)
                continue
            try:
                ok = send(turn)
            except DaemonUnavailable:
                break
            _unlink(path)
            if ok:
                delivered += 1
            else:
                dropped += 1
        if delivered or dropped:
            log.info("replayed the spool: %d delivered, %d dropped", delivered, dropped)
        return delivered


def _unlink(path: Path) -> int:
    try:
        path.unlink()
    except OSError:
        return 0
    return 1
