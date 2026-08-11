"""Representation-neutral policy source port for the Sim-HIL application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from robot_motion.deploy import (
    PolicyActionEmission,
    PolicyExecutionFeedback,
    PolicyExecutionFeedbackApplication,
)


@dataclass(frozen=True)
class SimHILPolicySourceBuildContext:
    """Lower-owned objects available to an injected policy-source builder."""

    scene: Any
    task_config: Any
    task_config_path: Path
    target_runtime: Any
    task_initialization_info: Mapping[str, object]
    downstream_target_lead_sec: float
    scene_capture: Callable[[Callable[[Any], object]], object]


@runtime_checkable
class SimHILPolicySourcePort(Protocol):
    """A source of canonical absolute emissions, independent of model encoding."""

    @property
    def ready(self) -> bool: ...

    @property
    def last_error(self) -> str | None: ...

    @property
    def maximum_hold_sec(self) -> float: ...

    def start(self) -> None: ...

    def reset_for_activation(self) -> None: ...

    def step(
        self,
        *,
        now: float | None = None,
        step_index: int | None = None,
    ) -> PolicyActionEmission | None: ...

    def status(self, *, now: float | None = None) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


@runtime_checkable
class SimHILPolicyExecutionFeedbackPort(Protocol):
    """Optional passive feedback consumer; policy sources need not implement it."""

    def observe_execution_feedback(
        self,
        feedback: PolicyExecutionFeedback,
    ) -> PolicyExecutionFeedbackApplication: ...


def require_sim_hil_policy_source(value: Any) -> SimHILPolicySourcePort:
    """Reject sources that bypass the common immutable emission boundary."""

    if not isinstance(value, SimHILPolicySourcePort):
        raise TypeError(
            "Sim-HIL policy source must implement the typed PolicyActionEmission port."
        )
    maximum_hold = float(value.maximum_hold_sec)
    if maximum_hold <= 0.0:
        raise ValueError("Sim-HIL policy source maximum_hold_sec must be positive.")
    return value


__all__ = [
    "SimHILPolicySourceBuildContext",
    "SimHILPolicyExecutionFeedbackPort",
    "SimHILPolicySourcePort",
    "require_sim_hil_policy_source",
]
