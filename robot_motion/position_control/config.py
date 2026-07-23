from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .interpolation import (
    DEFAULT_POSITION_COMMAND_HZ,
    DEFAULT_POSITION_REFERENCE_HZ,
)


DEFAULT_POSITION_SAMPLE_HZ = 15.0
POSITION_PROFILE_NAME = "position"
POSITION_45_150_PROFILE_NAME = "position_45_150hz"
POSITION_15_60_120_PROFILE_NAME = "position_15_60_120hz"


def _finite_positive(value: float, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


@dataclass(frozen=True)
class PositionControlRates:
    """Clock contract shared by simulation, teleoperation, replay, and deploy.

    The clocks are independent.  Consumers must advance them from timestamps;
    they must not assume that ``reference_hz / sample_hz`` or
    ``command_hz / reference_hz`` is an integer.
    """

    sample_hz: float = DEFAULT_POSITION_SAMPLE_HZ
    reference_hz: float = DEFAULT_POSITION_REFERENCE_HZ
    ik_hz: float = DEFAULT_POSITION_REFERENCE_HZ
    command_hz: float = DEFAULT_POSITION_COMMAND_HZ

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_hz",
            _finite_positive(self.sample_hz, name="sample_hz"),
        )
        object.__setattr__(
            self,
            "reference_hz",
            _finite_positive(self.reference_hz, name="reference_hz"),
        )
        object.__setattr__(
            self,
            "ik_hz",
            _finite_positive(self.ik_hz, name="ik_hz"),
        )
        object.__setattr__(
            self,
            "command_hz",
            _finite_positive(self.command_hz, name="command_hz"),
        )

    @property
    def sample_period_sec(self) -> float:
        return 1.0 / self.sample_hz

    @property
    def reference_period_sec(self) -> float:
        return 1.0 / self.reference_hz

    @property
    def ik_period_sec(self) -> float:
        return 1.0 / self.ik_hz

    @property
    def command_period_sec(self) -> float:
        return 1.0 / self.command_hz

    @property
    def ordering_warnings(self) -> tuple[str, ...]:
        warnings: list[str] = []
        if self.reference_hz < self.sample_hz:
            warnings.append("reference_hz_below_sample_hz")
        if self.command_hz < self.reference_hz:
            warnings.append("command_hz_below_reference_hz")
        return tuple(warnings)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "sample_hz": self.sample_hz,
            "reference_hz": self.reference_hz,
            "ik_hz": self.ik_hz,
            "command_hz": self.command_hz,
            "sample_period_sec": self.sample_period_sec,
            "reference_period_sec": self.reference_period_sec,
            "ik_period_sec": self.ik_period_sec,
            "command_period_sec": self.command_period_sec,
            "ordering_warnings": list(self.ordering_warnings),
            "independent_clocks": True,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PositionControlRates":
        return cls(
            sample_hz=float(values.get("sample_hz", DEFAULT_POSITION_SAMPLE_HZ)),
            reference_hz=float(
                values.get("reference_hz", DEFAULT_POSITION_REFERENCE_HZ)
            ),
            ik_hz=float(values.get("ik_hz", DEFAULT_POSITION_REFERENCE_HZ)),
            command_hz=float(values.get("command_hz", DEFAULT_POSITION_COMMAND_HZ)),
        )


@dataclass(frozen=True)
class PositionControlProfile:
    """Reusable position-control profile without hardware-specific transport."""

    name: str
    rates: PositionControlRates
    interpolation: str = "linear"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Position control profile name must be non-empty.")
        interpolation = str(self.interpolation).strip().lower()
        if interpolation != "linear":
            raise ValueError("Only timestamp-driven linear interpolation is supported.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "interpolation", interpolation)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interpolation": self.interpolation,
            "rates": self.rates.to_metadata(),
        }


POSITION_PROFILE = PositionControlProfile(
    name=POSITION_PROFILE_NAME,
    rates=PositionControlRates(),
)

# Explicit name for the canonical real position-control clock contract.  It is
# intentionally rate-identical to POSITION_PROFILE; the longer name remains a
# stable CLI/config spelling for existing deployments.
POSITION_15_60_120_PROFILE = PositionControlProfile(
    name=POSITION_15_60_120_PROFILE_NAME,
    rates=PositionControlRates(
        sample_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
    ),
)

# Explicit compatibility profile for historical commands and datasets.  The
# production default is 15 -> 60 -> 120 Hz; selecting this legacy name must
# continue to mean 15 -> 45 -> 150 Hz rather than silently changing an existing
# replay/deploy command.
POSITION_45_150_PROFILE = PositionControlProfile(
    name=POSITION_45_150_PROFILE_NAME,
    rates=PositionControlRates(
        reference_hz=45.0,
        ik_hz=45.0,
        command_hz=150.0,
    ),
)

DEFAULT_POSITION_PROFILE = POSITION_PROFILE


__all__ = [
    "DEFAULT_POSITION_SAMPLE_HZ",
    "DEFAULT_POSITION_PROFILE",
    "POSITION_45_150_PROFILE",
    "POSITION_45_150_PROFILE_NAME",
    "POSITION_15_60_120_PROFILE",
    "POSITION_15_60_120_PROFILE_NAME",
    "POSITION_PROFILE",
    "POSITION_PROFILE_NAME",
    "PositionControlProfile",
    "PositionControlRates",
]
