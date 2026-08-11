from __future__ import annotations

from types import SimpleNamespace

import pytest

from teleop_runtime.real_teleop.startup_device_runtime import (
    StartupDeviceRuntime,
    startup_device_label,
    startup_device_requirement,
)


class _Progress:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def begin(self, key: str, **details: object) -> None:
        self.events.append(("begin", key, details))

    def succeed(self, key: str) -> None:
        self.events.append(("succeed", key))

    def fail(self, key: str, detail: str) -> None:
        self.events.append(("fail", key, detail))

    def close(self) -> None:
        self.events.append(("close",))


class _DmtacBackend:
    def __init__(self, *, ready: bool = True, error: str = "") -> None:
        self.ready = ready
        self.error = error
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_ready(self) -> bool:
        return self.ready

    def get_state(self, *, include_tactile_arrays: bool = True) -> dict[str, str]:
        assert include_tactile_arrays is False
        return {"telemetry_error": self.error}


def _runtime(*, capability: bool, progress: _Progress | None = None):
    errors: dict[str, str] = {}
    logs: list[tuple[str, str]] = []
    messages: list[tuple[str, str]] = []
    runtime = StartupDeviceRuntime(
        capability_admission=capability,
        startup_progress=progress,
        startup_device_errors=errors,
        control_ui=SimpleNamespace(
            append_message=lambda tag, message: messages.append((tag, message))
        ),
        dmtac_gripper_backend_type=_DmtacBackend,
        log=lambda tag, message: logs.append((tag, message)),
    )
    return runtime, errors, logs, messages


def test_startup_device_labels_and_requirements_are_stable() -> None:
    assert startup_device_label("arm.left.qpos") == "左臂状态反馈"
    assert startup_device_label("custom") == "custom"
    assert startup_device_requirement("arm.force_guard") == "motion_safety"
    assert startup_device_requirement("rig.session") == "motion_safety"
    assert startup_device_requirement("camera.front") == "required_for_collection"
    assert startup_device_requirement("dataset.writer") == "required_for_collection"
    assert startup_device_requirement("hand.left") == "feature_dependent"


def test_startup_device_success_publishes_one_receipt() -> None:
    progress = _Progress()
    runtime, errors, logs, messages = _runtime(
        capability=False,
        progress=progress,
    )
    backend = _DmtacBackend()

    assert runtime.start("hand.left", backend) is True
    assert backend.started is True
    assert errors == {}
    assert logs == []
    assert messages == []
    assert progress.events == [
        (
            "begin",
            "hand.left",
            {
                "label": "左手 / 左夹爪",
                "message": "正在调用设备 backend.start()",
                "requirement": "feature_dependent",
            },
        ),
        ("succeed", "hand.left"),
    ]


def test_startup_device_strict_failure_closes_progress_and_reraises() -> None:
    progress = _Progress()
    runtime, errors, logs, messages = _runtime(
        capability=False,
        progress=progress,
    )

    class FailingBackend:
        def start(self) -> None:
            raise OSError("offline")

    with pytest.raises(OSError, match="offline"):
        runtime.start("arm.left.qpos", FailingBackend())

    assert errors == {}
    assert logs == []
    assert messages == []
    assert progress.events[-2:] == [
        ("fail", "arm.left.qpos", "OSError: offline"),
        ("close",),
    ]


def test_startup_device_capability_mode_records_dmtac_readiness_failure() -> None:
    progress = _Progress()
    runtime, errors, logs, messages = _runtime(
        capability=True,
        progress=progress,
    )

    assert runtime.start(
        "hand.right",
        _DmtacBackend(ready=False, error="telemetry unavailable"),
    ) is False
    assert errors == {"hand.right": "RuntimeError: telemetry unavailable"}
    assert logs == [
        (
            "Device",
            "WARNING: hand.right unavailable: RuntimeError: telemetry unavailable",
        )
    ]
    assert messages == [
        (
            "Device",
            "hand.right 连接失败：RuntimeError: telemetry unavailable",
        )
    ]
    assert progress.events[-1] == (
        "fail",
        "hand.right",
        "RuntimeError: telemetry unavailable",
    )


def test_startup_device_none_is_not_an_error() -> None:
    runtime, errors, logs, messages = _runtime(capability=True)

    assert runtime.start("hand.left", None) is False
    assert errors == {}
    assert logs == []
    assert messages == []
