"""Passive real-gripper state provider and Gateway executor."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

from .gripper_state_deployment import RealGripperStateRegistration

_ASSUMED_SOURCES = frozenset(
    {
        "cached_assumption",
        "commanded_fallback",
        "configured_disconnected_assumption",
    }
)


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class PublishedGripperState:
    """One immutable state cut published by the lower gripper owner."""

    measurement_id: str
    clock_domain_id: str
    close_fraction_by_component: Mapping[str, float]
    measured_at_by_component: Mapping[str, float]
    state_source_by_component: Mapping[str, str]
    available_at: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "measurement_id",
            _required(self.measurement_id, name="measurement_id"),
        )
        object.__setattr__(
            self,
            "clock_domain_id",
            _required(self.clock_domain_id, name="clock_domain_id"),
        )
        close = dict(self.close_fraction_by_component)
        measured = dict(self.measured_at_by_component)
        sources = dict(self.state_source_by_component)
        if not close or set(close) != set(measured) or set(close) != set(sources):
            raise ValueError(
                "Published gripper mappings must exactly cover one component set."
            )
        normalized_close: dict[str, float] = {}
        normalized_measured: dict[str, float] = {}
        normalized_sources: dict[str, str] = {}
        available_at = _finite(self.available_at, name="available_at")
        for component in sorted(close):
            component_id = _required(component, name="component_id")
            fraction = _finite(close[component], name=f"close_fraction.{component_id}")
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("Gripper close fractions must lie in [0, 1].")
            measured_at = _finite(
                measured[component], name=f"measured_at.{component_id}"
            )
            if measured_at > available_at:
                raise ValueError(
                    "Gripper measurement cannot be available before it was measured."
                )
            source = _required(sources[component], name=f"state_source.{component_id}")
            if source in _ASSUMED_SOURCES:
                raise ValueError(
                    "Published gripper state must be measured, not assumed."
                )
            normalized_close[component_id] = fraction
            normalized_measured[component_id] = measured_at
            normalized_sources[component_id] = source
        refs = tuple(
            sorted(
                {_required(item, name="evidence_ref") for item in self.evidence_refs}
            )
        )
        if not refs:
            raise ValueError("Published gripper state requires lower-owned evidence.")
        object.__setattr__(
            self, "close_fraction_by_component", MappingProxyType(normalized_close)
        )
        object.__setattr__(
            self, "measured_at_by_component", MappingProxyType(normalized_measured)
        )
        object.__setattr__(
            self, "state_source_by_component", MappingProxyType(normalized_sources)
        )
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "evidence_refs", refs)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "measurement_id": self.measurement_id,
                "clock_domain_id": self.clock_domain_id,
                "close_fraction_by_component": dict(self.close_fraction_by_component),
                "measured_at_by_component": dict(self.measured_at_by_component),
                "state_source_by_component": dict(self.state_source_by_component),
                "available_at": self.available_at,
                "evidence_refs": self.evidence_refs,
            },
            domain="dexteleop.agent_skill.published_gripper_state.v1",
        )


class PublishedGripperStateSource(Protocol):
    """No-I/O source owned and updated by a lower runtime."""

    no_io: bool

    def latest_gripper_state(self) -> PublishedGripperState: ...


class PublishedGripperStateBuffer:
    """Thread-safe latest-value buffer; reads never touch gripper hardware."""

    no_io = True

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._latest: PublishedGripperState | None = None

    def publish(self, state: PublishedGripperState) -> None:
        if not isinstance(state, PublishedGripperState):
            raise TypeError("state must be PublishedGripperState.")
        with self._lock:
            if (
                self._latest is not None
                and state.available_at < self._latest.available_at
            ):
                raise ValueError("Published gripper-state availability regressed.")
            self._latest = state

    def latest_gripper_state(self) -> PublishedGripperState:
        with self._lock:
            state = self._latest
        if state is None:
            raise RuntimeError("No lower-owned gripper-state publication is available.")
        return state


def read_published_gripper_state(
    *,
    registration: RealGripperStateRegistration,
    source: PublishedGripperStateSource,
    parameters: Mapping[str, Any],
    station_revision: int,
    now: float,
) -> PassiveReadResult:
    """Project one lower-published cut without performing backend I/O."""

    profile = registration.profile
    route = registration.skill.routes[0]
    registration.payload_validator.validate_route(route)
    registration.payload_validator.validate_parameters(parameters)
    state = source.latest_gripper_state()
    if state.clock_domain_id != profile.clock_domain_id:
        raise RuntimeError("Published gripper-state clock domain mismatch.")
    if set(state.close_fraction_by_component) != set(profile.component_ids):
        raise RuntimeError("Published gripper-state component inventory mismatch.")
    read_at = _finite(now, name="now")
    if read_at < state.available_at:
        raise RuntimeError("Published gripper state is not causally available.")
    requested = tuple(sorted(parameters["component_ids"]))
    maximum_age = parameters["max_measurement_age_ms"] / 1_000.0
    if any(
        read_at - state.measured_at_by_component[item] > maximum_age
        for item in requested
    ):
        raise RuntimeError("Published gripper-state measurement is stale.")
    result = {
        "component_ids": list(requested),
        "station_revision": station_revision,
        "measurement_id": state.measurement_id,
        "measurement_hash": state.semantic_content_hash,
        "measured_at_by_component": {
            item: state.measured_at_by_component[item] for item in requested
        },
        "available_at": state.available_at,
        "clock_domain_id": state.clock_domain_id,
        "close_fraction_by_component": {
            item: state.close_fraction_by_component[item] for item in requested
        },
        "state_source_by_component": {
            item: state.state_source_by_component[item] for item in requested
        },
        "observation_binding_id": profile.observation_binding_id,
        "observation_binding_hash": profile.observation_binding_hash,
    }
    registration.payload_validator.validate_execution_result(result)
    contextual = registration.payload_validator.validate_execution_result_for_parameters
    if contextual is None:
        raise RuntimeError("Gripper-state registration lacks contextual validation.")
    contextual(parameters, result)
    return PassiveReadResult(
        execution_result=result,
        evidence_refs=tuple(
            sorted({*state.evidence_refs, state.semantic_content_hash})
        ),
        reason_code="passive_real_gripper_state_read_completed",
    )


def build_real_gripper_state_executor(
    *,
    registration: RealGripperStateRegistration,
    source: PublishedGripperStateSource,
    current_station_revision: Callable[[], int],
    max_retained_operations: int = 32,
    clock: Callable[[], float] = time.time,
) -> PassiveReadSkillExecutor:
    """Build a passive executor over an already-published measurement source."""

    if not isinstance(registration, RealGripperStateRegistration):
        raise TypeError("registration must be typed.")
    if not callable(getattr(source, "latest_gripper_state", None)) or not bool(
        getattr(source, "no_io", False)
    ):
        raise TypeError("source must be an explicitly no-I/O gripper-state source.")
    if not callable(current_station_revision) or not callable(clock):
        raise TypeError("current_station_revision and clock must be callable.")

    def read(request: Any, operation: Any) -> PassiveReadResult:
        del operation
        return read_published_gripper_state(
            registration=registration,
            source=source,
            parameters=request.parameters,
            station_revision=current_station_revision(),
            now=clock(),
        )

    return PassiveReadSkillExecutor(
        port_id="agent-gateway:real-passive-gripper-state:"
        + registration.binding.runtime_session_id,
        executor_id=registration.binding.executor_id,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )


__all__ = [
    "PublishedGripperState",
    "PublishedGripperStateBuffer",
    "PublishedGripperStateSource",
    "build_real_gripper_state_executor",
    "read_published_gripper_state",
]
