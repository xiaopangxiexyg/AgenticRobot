"""Lazy application-control-profile selection for MuJoCo Sim-HIL.

A selectable profile is the atomic pair of one application session lifecycle
and one control-cycle port.  Selecting only the cycle port would incorrectly
leave start/HOLD/rearm/close bound to another profile's lifecycle.  Builders
are evaluated lazily and own no authority merely by being registered here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .control_cycle import (
    SIM_HIL_CONTROL_CYCLE_FIXED_ALPHA_COMPATIBILITY,
    FixedAlphaCompatibilityControlCyclePort,
    SimHILControlCyclePort,
    SimHILControlCyclePortSpec,
)


SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY = (
    SIM_HIL_CONTROL_CYCLE_FIXED_ALPHA_COMPATIBILITY
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class SimHILControlProfileAssembly:
    """Existing resources visible to one explicitly selected profile builder."""

    compatibility_session: Any
    runtime: Any
    policy_source: Any
    human_source: Any
    readiness_provider: Any


@dataclass(frozen=True)
class SimHILApplicationControlProfile:
    """Atomic lifecycle/cycle pair consumed by ``MujocoSimHILApplication``."""

    profile_id: str
    session: Any
    control_cycle_port: SimHILControlCyclePort

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_id",
            _required(self.profile_id, name="profile_id"),
        )
        for method_name in (
            "preflight",
            "start",
            "request_hold",
            "rearm",
            "close",
        ):
            if not callable(getattr(self.session, method_name, None)):
                raise TypeError(
                    "Selected profile session is missing required method "
                    f"{method_name}()."
                )
        spec = getattr(self.control_cycle_port, "spec", None)
        if not isinstance(spec, SimHILControlCyclePortSpec):
            raise TypeError("Selected control-cycle port has no typed spec.")
        for method_name in ("activate", "deactivate", "run_cycle", "close"):
            if not callable(getattr(self.control_cycle_port, method_name, None)):
                raise TypeError(
                    "Selected control-cycle port is missing required method "
                    f"{method_name}()."
                )


SimHILControlProfileBuilder = Callable[
    [SimHILControlProfileAssembly],
    SimHILApplicationControlProfile,
]


@dataclass(frozen=True)
class SimHILControlProfileBinding:
    """One declarative profile ID to lazy application-profile builder."""

    profile_id: str
    cycle_kind: str
    builder: SimHILControlProfileBuilder

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_id",
            _required(self.profile_id, name="profile_id"),
        )
        object.__setattr__(
            self,
            "cycle_kind",
            _required(self.cycle_kind, name="cycle_kind"),
        )
        if not callable(self.builder):
            raise TypeError("profile builder must be callable.")


def _fixed_alpha_binding() -> SimHILControlProfileBinding:
    def build(
        assembly: SimHILControlProfileAssembly,
    ) -> SimHILApplicationControlProfile:
        return SimHILApplicationControlProfile(
            profile_id=SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY,
            session=assembly.compatibility_session,
            control_cycle_port=FixedAlphaCompatibilityControlCyclePort(
                assembly.compatibility_session
            ),
        )

    return SimHILControlProfileBinding(
        profile_id=SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY,
        cycle_kind=SIM_HIL_CONTROL_CYCLE_FIXED_ALPHA_COMPATIBILITY,
        builder=build,
    )


def _binding_by_profile_id(
    additional_bindings: Iterable[SimHILControlProfileBinding],
) -> dict[str, SimHILControlProfileBinding]:
    bindings = (_fixed_alpha_binding(), *tuple(additional_bindings))
    if any(
        not isinstance(binding, SimHILControlProfileBinding)
        for binding in bindings
    ):
        raise TypeError("additional_bindings must contain typed profile bindings.")
    identifiers = tuple(binding.profile_id for binding in bindings)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Sim-HIL control profile IDs must be unique.")
    return {binding.profile_id: binding for binding in bindings}


def require_sim_hil_control_profile_available(
    profile_id: str,
    *,
    additional_bindings: Iterable[SimHILControlProfileBinding] = (),
) -> str:
    """Validate selection before concrete application resources are built."""

    selected = _required(profile_id, name="control_profile_id")
    available = _binding_by_profile_id(additional_bindings)
    if selected not in available:
        raise ValueError(
            "No Sim-HIL application control binding is registered for profile "
            f"{selected!r}."
        )
    return selected


def select_sim_hil_application_control_profile(
    profile_id: str,
    *,
    assembly: SimHILControlProfileAssembly,
    additional_bindings: Iterable[SimHILControlProfileBinding] = (),
) -> SimHILApplicationControlProfile:
    """Construct only the selected atomic session/cycle profile."""

    if not isinstance(assembly, SimHILControlProfileAssembly):
        raise TypeError("assembly must be a SimHILControlProfileAssembly.")
    extra = tuple(additional_bindings)
    selected = require_sim_hil_control_profile_available(
        profile_id,
        additional_bindings=extra,
    )
    binding = _binding_by_profile_id(extra)[selected]
    profile = binding.builder(assembly)
    if not isinstance(profile, SimHILApplicationControlProfile):
        raise TypeError(
            "Selected builder must return SimHILApplicationControlProfile."
        )
    if profile.profile_id != binding.profile_id:
        raise ValueError("Selected profile identity differs from its binding.")
    if profile.control_cycle_port.spec.cycle_kind != binding.cycle_kind:
        raise ValueError(
            "Selected control-cycle behavior differs from its profile binding."
        )
    return profile


__all__ = [
    "SIM_HIL_CONTROL_PROFILE_FIXED_ALPHA_COMPATIBILITY",
    "SimHILApplicationControlProfile",
    "SimHILControlProfileAssembly",
    "SimHILControlProfileBinding",
    "SimHILControlProfileBuilder",
    "require_sim_hil_control_profile_available",
    "select_sim_hil_application_control_profile",
]
