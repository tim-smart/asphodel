"""Asphodel memory provider for the Hermes agent.

A thin, dependency-free Python client of the Asphodel daemon (ADR 0006, "One
daemon, and a thin Hermes plugin"). Hermes discovers this directory because
this file names ``MemoryProvider`` and ``register_memory_provider``, loads it
under a synthetic package name, and calls :func:`register`. Every submodule
is imported relatively for that reason.

The Hermes hook contract is defined in ``agent/memory_provider.py``.
"""

from __future__ import annotations

from .provider import PROVIDER_NAME, AsphodelMemoryProvider

__all__ = ["AsphodelMemoryProvider", "PROVIDER_NAME", "register"]


def register(ctx) -> None:
    """Hermes' plugin entry point: hands one provider instance to the manager."""
    ctx.register_memory_provider(AsphodelMemoryProvider())
