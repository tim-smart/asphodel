"""The circuit breaker (TIM-94, decision 6): after three consecutive
connection failures the plugin skips the network for 30 s, so a sidecar
restart doesn't cost every turn a timeout."""

from __future__ import annotations

import time
from typing import Callable

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

    def allow(self) -> bool:
        """False while open: no request should be made."""
        raise NotImplementedError

    def record_success(self) -> None:
        raise NotImplementedError

    def record_failure(self) -> None:
        raise NotImplementedError

    @property
    def is_open(self) -> bool:
        raise NotImplementedError
