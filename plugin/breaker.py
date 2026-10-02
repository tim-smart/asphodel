"""The circuit breaker (TIM-94, decision 6): after three consecutive
connection failures the plugin skips the network for 30 s, so a sidecar
restart doesn't cost every turn a timeout."""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

BREAKER_FAILURES = 3
BREAKER_COOLDOWN_S = 30.0


class CircuitBreaker:
    """Counts connection failures only. An HTTP error status means the daemon
    answered, so it never trips the breaker."""

    def __init__(
        self,
        *,
        failures: int = BREAKER_FAILURES,
        cooldown: float = BREAKER_COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failures = failures
        self.cooldown = cooldown
        self._clock = clock
        self._lock = threading.Lock()
        self._consecutive = 0
        self._opened_at: Optional[float] = None

    def allow(self) -> bool:
        """False while open: no request should be made."""
        with self._lock:
            if self._opened_at is None:
                return True
            return self._clock() - self._opened_at >= self.cooldown

    def record_success(self) -> None:
        with self._lock:
            self._consecutive = 0
            self._opened_at = None

    def record_failure(self) -> None:
        """After the cooldown the count isn't reset, so one more failure
        reopens the breaker at once."""
        with self._lock:
            self._consecutive += 1
            if self._consecutive >= self.failures:
                self._opened_at = self._clock()

    @property
    def is_open(self) -> bool:
        return not self.allow()
