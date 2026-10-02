"""The turn spool (TIM-94, decision 6).

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

import os
from pathlib import Path
from typing import Callable, List

SPOOL_MAX_BYTES = 10 * 1024 * 1024
SPOOL_MAX_AGE_S = 7 * 24 * 60 * 60.0


def spool_file_name(session_id: str, message_at: str) -> str:
    """The file name for a turn's source id: a stable digest of the session
    id and the message time, so the name is filesystem-safe whatever the
    session id contains."""
    raise NotImplementedError


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
        raise NotImplementedError

    def files(self) -> List[Path]:
        """Spooled turns, oldest first."""
        raise NotImplementedError

    def enforce_caps(self) -> int:
        """Deletes files older than ``max_age`` and then the oldest until the
        total is under ``max_bytes``. Returns how many were dropped."""
        raise NotImplementedError

    def replay(self, send: Callable[[dict], bool]) -> int:
        """Sends each spooled turn oldest first. ``send`` returns True on a
        2xx (the file is deleted), False when the turn can never succeed
        (a 4xx: the file is dropped), and raises
        :class:`asphodel_plugin.client.DaemonUnavailable` to stop the replay
        and keep the rest. Returns how many were delivered."""
        raise NotImplementedError
