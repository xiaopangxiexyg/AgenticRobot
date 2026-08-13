"""Sim-HIL build context and compatibility names for the shared source port."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from robot_motion.deploy import (
    PolicyEmissionSourcePort,
    PolicyExecutionFeedbackPort,
    require_policy_emission_source,
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


SimHILPolicySourcePort = PolicyEmissionSourcePort
SimHILPolicyExecutionFeedbackPort = PolicyExecutionFeedbackPort


def require_sim_hil_policy_source(value: Any) -> SimHILPolicySourcePort:
    """Compatibility wrapper around the backend-neutral source validator."""

    return require_policy_emission_source(value)


__all__ = [
    "SimHILPolicySourceBuildContext",
    "SimHILPolicyExecutionFeedbackPort",
    "SimHILPolicySourcePort",
    "require_sim_hil_policy_source",
]
