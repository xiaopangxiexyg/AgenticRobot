"""Runtime clock and stable fact identity contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from urllib.parse import quote


def _required_identifier(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class ClockDomainDescriptor:
    clock_domain_id: str
    runtime_session_id: str
    boot_id: str
    reset_epoch: int
    kind: str = "local_monotonic"
    resolution_sec: float | None = None
    monotonic: bool = True
    mapped_clock_domain_id: str | None = None
    offset_estimate_sec: float | None = None
    offset_uncertainty_sec: float | None = None
    mapping_sampled_from: float | None = None
    mapping_sampled_until: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "clock_domain_id", _required_identifier(self.clock_domain_id, field_name="clock_domain_id"))
        object.__setattr__(self, "runtime_session_id", _required_identifier(self.runtime_session_id, field_name="runtime_session_id"))
        object.__setattr__(self, "boot_id", _required_identifier(self.boot_id, field_name="boot_id"))
        object.__setattr__(self, "kind", _required_identifier(self.kind, field_name="kind"))
        if int(self.reset_epoch) < 0:
            raise ValueError("reset_epoch must be non-negative.")
        object.__setattr__(self, "reset_epoch", int(self.reset_epoch))
        if self.resolution_sec is not None:
            resolution = float(self.resolution_sec)
            if not math.isfinite(resolution) or resolution <= 0:
                raise ValueError("resolution_sec must be finite and positive.")
            object.__setattr__(self, "resolution_sec", resolution)
        if self.mapped_clock_domain_id is not None:
            object.__setattr__(
                self,
                "mapped_clock_domain_id",
                _required_identifier(
                    self.mapped_clock_domain_id,
                    field_name="mapped_clock_domain_id",
                ),
            )
        for name in (
            "offset_estimate_sec",
            "offset_uncertainty_sec",
            "mapping_sampled_from",
            "mapping_sampled_until",
        ):
            value = getattr(self, name)
            if value is not None:
                normalized = float(value)
                if not math.isfinite(normalized):
                    raise ValueError(f"{name} must be finite.")
                if name == "offset_uncertainty_sec" and normalized < 0:
                    raise ValueError("offset_uncertainty_sec must be non-negative.")
                object.__setattr__(self, name, normalized)
        if (
            self.mapping_sampled_from is not None
            and self.mapping_sampled_until is not None
            and self.mapping_sampled_until < self.mapping_sampled_from
        ):
            raise ValueError("Clock mapping sample interval is reversed.")


@dataclass(frozen=True)
class RuntimeFactIdentity:
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    sequence: int
    namespace: str = "fact"

    def __post_init__(self) -> None:
        for name in ("runtime_session_id", "control_domain_id", "producer_id", "namespace"):
            object.__setattr__(self, name, _required_identifier(getattr(self, name), field_name=name))
        if int(self.producer_epoch) < 0:
            raise ValueError("producer_epoch must be non-negative.")
        if int(self.sequence) < 0:
            raise ValueError("sequence must be non-negative.")
        object.__setattr__(self, "producer_epoch", int(self.producer_epoch))
        object.__setattr__(self, "sequence", int(self.sequence))

    @property
    def fact_id(self) -> str:
        parts = (
            self.namespace,
            self.runtime_session_id,
            self.control_domain_id,
            self.producer_id,
            str(self.producer_epoch),
            str(self.sequence),
        )
        return "dexfact:v1:" + ":".join(quote(part, safe="") for part in parts)


__all__ = ["ClockDomainDescriptor", "RuntimeFactIdentity"]
