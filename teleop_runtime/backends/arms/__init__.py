"""Arm backend compatibility exports loaded only on explicit access."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "RM65ArmBackend",
    "RM65ArmBackendConfig",
    "DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN",
    "RM65_BACKGROUND_TARGET_ADMISSION_CHOICES",
    "RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING",
    "RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK",
    "RM65ImpedanceArmBackend",
    "RM65ImpedanceArmBackendConfig",
    "RM65_ARMS_ONLY_DISPATCH_INVENTORY_V1",
    "RM65_ARMS_ONLY_REQUIRED_SEAM_ROLES_V1",
    "assess_rm65_arms_only_dispatch_inventory",
    "build_rm65_arms_only_dispatch_inventory",
    "RM65_JOINT_TARGET_CALL_PATHS_V1",
    "RM65JointTargetDispatchContext",
    "RM65PermittedPreSdkDispatchAdapter",
    "RM65PreSdkDispatchAdapter",
    "RM65ControlDispatchContext",
    "RM65PermittedPreSdkControlAdapter",
    "RM65PreSdkControlAdapter",
    "RM65_ARMS_ONLY_WORKER_DRAIN_PLAN_V1",
    "RM65ArmsOnlyWorkerDrainPlan",
    "RM65ArmsOnlyWorkerDrainResult",
    "RM65ExecutionDrainParticipant",
    "RM65WorkerDrainParticipantBinding",
    "build_rm65_arms_only_worker_drain_plan",
    "drain_rm65_arms_only_workers",
]

_LAZY_EXPORT_GROUPS = (
    (
        "teleop_runtime.backends.arms.rm65",
        (
            "RM65ArmBackend",
            "RM65ArmBackendConfig",
            "DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN",
            "RM65_BACKGROUND_TARGET_ADMISSION_CHOICES",
            "RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING",
            "RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK",
        ),
    ),
    (
        "teleop_runtime.backends.arms.rm65_impedance",
        ("RM65ImpedanceArmBackend", "RM65ImpedanceArmBackendConfig"),
    ),
    (
        "teleop_runtime.backends.arms.rm65_dispatch_inventory",
        (
            "RM65_ARMS_ONLY_DISPATCH_INVENTORY_V1",
            "RM65_ARMS_ONLY_REQUIRED_SEAM_ROLES_V1",
            "assess_rm65_arms_only_dispatch_inventory",
            "build_rm65_arms_only_dispatch_inventory",
        ),
    ),
    (
        "teleop_runtime.backends.arms.rm65_dispatch_call",
        (
            "RM65_JOINT_TARGET_CALL_PATHS_V1",
            "RM65JointTargetDispatchContext",
            "RM65PermittedPreSdkDispatchAdapter",
            "RM65PreSdkDispatchAdapter",
        ),
    ),
    (
        "teleop_runtime.backends.arms.rm65_control_dispatch_call",
        (
            "RM65ControlDispatchContext",
            "RM65PermittedPreSdkControlAdapter",
            "RM65PreSdkControlAdapter",
        ),
    ),
    (
        "teleop_runtime.backends.arms.rm65_worker_drain",
        (
            "RM65_ARMS_ONLY_WORKER_DRAIN_PLAN_V1",
            "RM65ArmsOnlyWorkerDrainPlan",
            "RM65ArmsOnlyWorkerDrainResult",
            "RM65ExecutionDrainParticipant",
            "RM65WorkerDrainParticipantBinding",
            "build_rm65_arms_only_worker_drain_plan",
            "drain_rm65_arms_only_workers",
        ),
    ),
)
_LAZY_EXPORTS = {
    name: module_name
    for module_name, names in _LAZY_EXPORT_GROUPS
    for name in names
}

if set(_LAZY_EXPORTS) != set(__all__):
    raise RuntimeError("arm backend lazy exports do not match __all__.")


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
