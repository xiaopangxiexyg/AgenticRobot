"""Version-neutral read-time mapping from Episode time to sample indices."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math
from typing import Any, Iterable, Protocol, cast


class SampleAxisError(ValueError):
    """Raised when a causal sample axis is ambiguous or malformed."""


class CausalSamplePoint(Protocol):
    sample_index: int
    cut_at: float
    cut_clock_domain_id: str


def _finite(value: object, *, name: str) -> float:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise SampleAxisError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise SampleAxisError(f"{name} must be finite.")
    return result


def _clock(value: object) -> str:
    if not isinstance(value, str):
        raise SampleAxisError("clock_domain_id must be a string.")
    result = value.strip()
    if not result:
        raise SampleAxisError("clock_domain_id must be non-empty.")
    return result


@dataclass(frozen=True)
class SampleIndexInterval:
    """A half-open ``[begin, end)`` interval on one contiguous sample axis."""

    sample_index_begin: int
    sample_index_end_exclusive: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sample_index_begin, int)
            or not isinstance(self.sample_index_end_exclusive, int)
            or isinstance(self.sample_index_begin, bool)
            or isinstance(self.sample_index_end_exclusive, bool)
            or self.sample_index_begin < 0
            or self.sample_index_end_exclusive < self.sample_index_begin
        ):
            raise SampleAxisError("Sample index interval must be ordered and non-negative.")

    @property
    def sample_count(self) -> int:
        return self.sample_index_end_exclusive - self.sample_index_begin

    @property
    def empty(self) -> bool:
        return self.sample_count == 0


@dataclass(frozen=True)
class EpisodeSampleAxis:
    """One monotonic causal-cut axis bound to an Episode clock."""

    clock_domain_id: str
    cut_times: tuple[float, ...]

    def __post_init__(self) -> None:
        clock = _clock(self.clock_domain_id)
        cuts = tuple(
            _finite(value, name=f"cut_times[{index}]")
            for index, value in enumerate(self.cut_times)
        )
        if any(later < earlier for earlier, later in zip(cuts, cuts[1:])):
            raise SampleAxisError("Causal sample cut times must be non-decreasing.")
        object.__setattr__(self, "clock_domain_id", clock)
        object.__setattr__(self, "cut_times", cuts)

    @classmethod
    def from_causal_samples(
        cls,
        samples: Iterable[CausalSamplePoint],
        *,
        clock_domain_id: str,
    ) -> "EpisodeSampleAxis":
        expected_clock = _clock(clock_domain_id)
        ordered = tuple(samples)
        for expected_index, sample in enumerate(ordered):
            if (
                not isinstance(sample.sample_index, int)
                or isinstance(sample.sample_index, bool)
                or sample.sample_index != expected_index
            ):
                raise SampleAxisError(
                    "Causal sample indices must be contiguous from zero."
                )
            if _clock(sample.cut_clock_domain_id) != expected_clock:
                raise SampleAxisError(
                    "Causal sample cut must use the bound Episode clock."
                )
        return cls(
            clock_domain_id=expected_clock,
            cut_times=tuple(float(sample.cut_at) for sample in ordered),
        )

    @property
    def sample_count(self) -> int:
        return len(self.cut_times)

    def first_at_or_after(self, at: float) -> int:
        return bisect_left(self.cut_times, _finite(at, name="at"))

    def first_strictly_after(self, at: float) -> int:
        return bisect_right(self.cut_times, _finite(at, name="at"))

    def half_open_time_interval(
        self,
        *,
        start_at: float,
        end_at: float,
    ) -> SampleIndexInterval:
        """Map ``[start_at, end_at)`` to a half-open sample interval."""

        start = _finite(start_at, name="start_at")
        end = _finite(end_at, name="end_at")
        if end < start:
            raise SampleAxisError("end_at cannot precede start_at.")
        return SampleIndexInterval(
            sample_index_begin=bisect_left(self.cut_times, start),
            sample_index_end_exclusive=bisect_left(self.cut_times, end),
        )

    def closed_time_interval(
        self,
        *,
        start_at: float,
        end_at: float,
    ) -> SampleIndexInterval:
        """Map ``[start_at, end_at]`` to a half-open sample interval."""

        start = _finite(start_at, name="start_at")
        end = _finite(end_at, name="end_at")
        if end < start:
            raise SampleAxisError("end_at cannot precede start_at.")
        return SampleIndexInterval(
            sample_index_begin=bisect_left(self.cut_times, start),
            sample_index_end_exclusive=bisect_right(self.cut_times, end),
        )

    def from_time_to_end(self, *, start_at: float) -> SampleIndexInterval:
        return SampleIndexInterval(
            sample_index_begin=self.first_at_or_after(start_at),
            sample_index_end_exclusive=self.sample_count,
        )


__all__ = [
    "CausalSamplePoint",
    "EpisodeSampleAxis",
    "SampleAxisError",
    "SampleIndexInterval",
]
