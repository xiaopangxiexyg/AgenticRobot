"""Device-start admission for the canonical real-teleop startup lifecycle.

This module owns the one-shot ``backend.start()`` boundary, including strict
versus capability admission and its progress/error receipts.  It does not
construct devices, retry them, or own steady-state health.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


_DEVICE_LABELS = {
    "arm.left.qpos": "左臂状态反馈",
    "arm.right.qpos": "右臂状态反馈",
    "arm.force_guard": "双臂位控受力急停",
    "hand.left": "左手 / 左夹爪",
    "hand.right": "右手 / 右夹爪",
    "rig.session": "遥操核心 Rig",
    "human_shadow": "Human shadow 前端",
    "vive.clock_sync": "VIVE 时钟同步",
    "dataset.writer": "数采 writer 子进程",
    "gripper.input": "Q6 / 夹爪输入",
}


def startup_device_label(key: str) -> str:
    """Return the operator-facing startup label for one resource key."""

    normalized = str(key)
    return _DEVICE_LABELS.get(normalized, normalized)


def startup_device_requirement(key: str) -> str:
    """Classify the startup resource without deciding dataset admission."""

    normalized = str(key)
    if normalized.startswith("arm.") or normalized == "rig.session":
        return "motion_safety"
    if normalized.startswith("camera.") or normalized.startswith("dataset."):
        return "required_for_collection"
    return "feature_dependent"


class StartupDeviceRuntime:
    """Start configured backends and publish one deterministic admission receipt."""

    def __init__(
        self,
        *,
        capability_admission: bool,
        startup_progress: Any | None,
        startup_device_errors: dict[str, str],
        control_ui: Any | None,
        dmtac_gripper_backend_type: type,
        log: Callable[..., None],
    ) -> None:
        self._capability_admission = bool(capability_admission)
        self._startup_progress = startup_progress
        self._startup_device_errors = startup_device_errors
        self._control_ui = control_ui
        self._dmtac_gripper_backend_type = dmtac_gripper_backend_type
        self._log = log

    @staticmethod
    def label(key: str) -> str:
        return startup_device_label(key)

    @staticmethod
    def requirement(key: str) -> str:
        return startup_device_requirement(key)

    def start(self, key: str, backend: Any | None) -> bool:
        """Start one backend or record its capability-mode failure."""

        if backend is None:
            return False
        progress = self._startup_progress
        if progress is not None:
            progress.begin(
                key,
                label=self.label(key),
                message="正在调用设备 backend.start()",
                requirement=self.requirement(key),
            )
        try:
            backend.start()
            if isinstance(backend, self._dmtac_gripper_backend_type) and not backend.is_ready():
                try:
                    state = backend.get_state(include_tactile_arrays=False)
                except TypeError:
                    state = backend.get_state()
                raise RuntimeError(
                    str(state.get("telemetry_error") or "DMTAC backend is not ready")
                )
            if progress is not None:
                progress.succeed(key)
            return True
        except Exception as exc:  # noqa: BLE001 - capability admission records failures.
            detail = f"{type(exc).__name__}: {exc}"
            if progress is not None:
                progress.fail(key, detail)
            if not self._capability_admission:
                if progress is not None:
                    progress.close()
                raise
            self._startup_device_errors[str(key)] = detail
            self._log("Device", f"WARNING: {key} unavailable: {detail}")
            if self._control_ui is not None:
                self._control_ui.append_message(
                    "Device",
                    f"{key} 连接失败：{detail}",
                )
            return False


__all__ = [
    "StartupDeviceRuntime",
    "startup_device_label",
    "startup_device_requirement",
]
