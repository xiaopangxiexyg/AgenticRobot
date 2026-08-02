"""Compatibility proxy for lower-owned RealTeleop status projections."""

from __future__ import annotations

from ..real_teleop import status_payloads as _implementation


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
