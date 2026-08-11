"""Typed application seam for one Sim-HIL control cycle.

The port selects an application composition profile and returns the lower-owned
execution result.  It does not own source production, authority, canonical
safety, target resolution, reference generation, IK, dispatch, or episode
lifecycle.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from control_composition.activation import CompositionActivationScope
from robot_motion.action import AuthoritativeControlStep
from robot_motion.deploy import PolicyActionEmission

from .validation import _required


SIM_HIL_CONTROL_CYCLE_FIXED_ALPHA_COMPATIBILITY = "fixed_alpha_compatibility"
SIM_HIL_POLICY_SOURCE_COMPOSITION_ALPHA = "composition_alpha"
SIM_HIL_POLICY_SOURCE_ALWAYS = "always"
SIM_HIL_ACTIVATION_PRIMING_APPLICATION_COMPATIBILITY = "application_compatibility"
SIM_HIL_ACTIVATION_PRIMING_CONTROL_CYCLE_PORT = "control_cycle_port"


@dataclass(frozen=True)
class SimHILControlCyclePortSpec:
    """Application-visible behavior that must not leak between profiles."""

    port_id: str
    cycle_kind: str
    policy_source_requirement: str
    application_blend_failure_fallback_enabled: bool
    application_human_timeout_fallback_enabled: bool
    activation_priming_owner: str = SIM_HIL_ACTIVATION_PRIMING_APPLICATION_COMPATIBILITY
    activation_human_emission_required: bool = True
    application_human_blend_control_enabled: bool = True
    application_owns_human_source_close: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "port_id", _required(self.port_id, name="port_id"))
        object.__setattr__(
            self,
            "cycle_kind",
            _required(self.cycle_kind, name="cycle_kind"),
        )
        requirement = _required(
            self.policy_source_requirement,
            name="policy_source_requirement",
        )
        if requirement not in {
            SIM_HIL_POLICY_SOURCE_COMPOSITION_ALPHA,
            SIM_HIL_POLICY_SOURCE_ALWAYS,
        }:
            raise ValueError("Unsupported Sim-HIL policy source requirement.")
        object.__setattr__(self, "policy_source_requirement", requirement)
        priming_owner = _required(
            self.activation_priming_owner,
            name="activation_priming_owner",
        )
        if priming_owner not in {
            SIM_HIL_ACTIVATION_PRIMING_APPLICATION_COMPATIBILITY,
            SIM_HIL_ACTIVATION_PRIMING_CONTROL_CYCLE_PORT,
        }:
            raise ValueError("Unsupported Sim-HIL activation priming owner.")
        object.__setattr__(
            self,
            "activation_priming_owner",
            priming_owner,
        )
        for name in (
            "application_blend_failure_fallback_enabled",
            "application_human_timeout_fallback_enabled",
            "activation_human_emission_required",
            "application_human_blend_control_enabled",
            "application_owns_human_source_close",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")


@runtime_checkable
class SimHILControlCycleResult(Protocol):
    """Minimum result shape needed by the application scheduler."""

    execution: AuthoritativeControlStep


@runtime_checkable
class SimHILControlCyclePort(Protocol):
    """One replaceable cycle path under the application lifecycle owner."""

    @property
    def spec(self) -> SimHILControlCyclePortSpec: ...

    @property
    def active(self) -> bool: ...

    def activate(
        self,
        *,
        activation_scope: CompositionActivationScope | None,
    ) -> None: ...

    def deactivate(self, *, reason: str) -> None: ...

    def run_cycle(
        self,
        *,
        now: float,
        policy_emission: PolicyActionEmission | None,
        poll_human: bool,
        target_effective_at: float,
    ) -> SimHILControlCycleResult: ...

    def close(self) -> None: ...


class FixedAlphaCompatibilityControlCyclePort:
    """Exact adapter over the existing ``MujocoSimHILComposition.step`` path."""

    def __init__(self, composition: Any) -> None:
        self.composition = composition
        self._spec = SimHILControlCyclePortSpec(
            port_id="sim-hil-fixed-alpha-compatibility",
            cycle_kind=SIM_HIL_CONTROL_CYCLE_FIXED_ALPHA_COMPATIBILITY,
            policy_source_requirement=SIM_HIL_POLICY_SOURCE_COMPOSITION_ALPHA,
            application_blend_failure_fallback_enabled=True,
            application_human_timeout_fallback_enabled=True,
        )
        self._lock = threading.RLock()
        # The compatibility adapter owns no worker or publication resource.
        # Keep it immediately usable for legacy application fixtures that set
        # the application started latch directly; formal startup later binds
        # the real composition activation identity.
        self._active = True
        self._closed = False
        self._activation_id: str | None = None
        self._last_deactivation_reason: str | None = None

    @property
    def spec(self) -> SimHILControlCyclePortSpec:
        return self._spec

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active and not self._closed

    def activate(
        self,
        *,
        activation_scope: CompositionActivationScope | None,
    ) -> None:
        activation_id = (
            None if activation_scope is None else activation_scope.activation_id
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("Control-cycle port is closed.")
            if (
                self._active
                and self._activation_id is not None
                and self._activation_id != activation_id
            ):
                raise RuntimeError(
                    "Control-cycle port must deactivate before a new activation."
                )
            self._active = True
            self._activation_id = activation_id
            self._last_deactivation_reason = None

    def deactivate(self, *, reason: str) -> None:
        normalized_reason = _required(reason, name="reason")
        with self._lock:
            if self._closed:
                return
            self._active = False
            self._activation_id = None
            self._last_deactivation_reason = normalized_reason

    def run_cycle(
        self,
        *,
        now: float,
        policy_emission: PolicyActionEmission | None,
        poll_human: bool,
        target_effective_at: float,
    ) -> SimHILControlCycleResult:
        with self._lock:
            if self._closed:
                raise RuntimeError("Control-cycle port is closed.")
            if not self._active:
                raise RuntimeError("Control-cycle port is not active.")
            step = getattr(self.composition, "step", None)
            if not callable(step):
                raise TypeError(
                    "Fixed-alpha compatibility requires composition.step()."
                )
            result = step(
                now=now,
                policy_emission=policy_emission,
                poll_human=bool(poll_human),
                target_effective_at=target_effective_at,
            )
        execution = getattr(result, "execution", None)
        if execution is not None and not isinstance(
            execution,
            AuthoritativeControlStep,
        ):
            raise TypeError(
                "A Sim-HIL control cycle must return AuthoritativeControlStep."
            )
        return result

    def close(self) -> None:
        with self._lock:
            self._active = False
            self._activation_id = None
            self._closed = True


__all__ = [
    "SIM_HIL_ACTIVATION_PRIMING_APPLICATION_COMPATIBILITY",
    "SIM_HIL_ACTIVATION_PRIMING_CONTROL_CYCLE_PORT",
    "SIM_HIL_CONTROL_CYCLE_FIXED_ALPHA_COMPATIBILITY",
    "SIM_HIL_POLICY_SOURCE_ALWAYS",
    "SIM_HIL_POLICY_SOURCE_COMPOSITION_ALPHA",
    "FixedAlphaCompatibilityControlCyclePort",
    "SimHILControlCyclePort",
    "SimHILControlCyclePortSpec",
    "SimHILControlCycleResult",
]
