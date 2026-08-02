from __future__ import annotations

from types import SimpleNamespace

from teleop_runtime.collection.umi.components import UmiComponents
from teleop_runtime.gui.startup_progress import GuiStartupProgress


class _FakeUi:
    def __init__(self) -> None:
        self.updates: list[dict[str, object]] = []

    def publish(self, **updates: object) -> None:
        self.updates.append(dict(updates))


def test_startup_progress_reports_live_elapsed_and_completion() -> None:
    now = [10.0]
    ui = _FakeUi()
    progress = GuiStartupProgress(
        ui,
        heartbeat_interval_sec=None,
        clock=lambda: now[0],
    )

    progress.start()
    progress.begin(
        "arm.left.qpos",
        label="左臂状态反馈",
        message="正在调用设备 backend.start()",
        requirement="motion_safety",
    )
    now[0] = 12.25
    progress.publish_now()

    waiting = ui.updates[-1]
    waiting_item = waiting["hardware_status"]["arm.left.qpos"]
    assert waiting["stage"] == "设备连接中 · 2.2s"
    assert "左臂状态反馈" in waiting["control_hint"]
    assert waiting_item["status_label"] == "连接中"
    assert waiting_item["elapsed_sec"] == 2.25

    progress.succeed("arm.left.qpos", message="反馈已就绪")
    completed_item = ui.updates[-1]["hardware_status"]["arm.left.qpos"]
    assert completed_item["online"] is True
    assert completed_item["status_label"] == "已连接"
    assert completed_item["message"] == "反馈已就绪"


def test_startup_progress_preserves_failure_and_optional_warning() -> None:
    now = [3.0]
    ui = _FakeUi()
    progress = GuiStartupProgress(
        ui,
        heartbeat_interval_sec=None,
        clock=lambda: now[0],
    )

    progress.begin(
        "camera.front",
        label="相机 front",
        message="正在启动视频流",
        requirement="required_for_collection",
    )
    now[0] = 4.0
    progress.fail("camera.front", "RuntimeError: no frames")
    failed = ui.updates[-1]["hardware_status"]["camera.front"]
    assert failed["status_level"] == "error"
    assert failed["error"] == "RuntimeError: no frames"
    assert failed["elapsed_sec"] == 1.0

    progress.begin(
        "vive.clock_sync",
        label="VIVE 时钟同步",
        message="正在检查",
    )
    progress.warn(
        "vive.clock_sync",
        "clock synchronization is not ready",
        message="当前配置允许继续",
    )
    warned = ui.updates[-1]["hardware_status"]["vive.clock_sync"]
    assert warned["status_level"] == "warn"
    assert warned["status_label"] == "未就绪（可继续）"


def test_umi_components_publish_each_synchronous_start_step() -> None:
    ui = _FakeUi()
    progress = GuiStartupProgress(ui, heartbeat_interval_sec=None)

    class _Startable:
        def start(self) -> None:
            return None

        def close(self) -> None:
            return None

    ready_gripper = SimpleNamespace(is_ready=lambda: True)
    components = UmiComponents(
        vive=_Startable(),
        cameras={"left": _Startable(), "right": _Startable()},
        angle_trigger=_Startable(),
        grippers={"left": ready_gripper, "right": ready_gripper},
        rig=_Startable(),
        retargeters={},
        hand_state_poller=_Startable(),
    )

    components.start(startup_progress=progress)

    hardware_status = ui.updates[-1]["hardware_status"]
    assert set(hardware_status) == {
        "end_effector.rig",
        "end_effector.state_history",
        "pose_source",
        "camera.left",
        "camera.right",
        "command_input",
    }
    assert all(item["online"] for item in hardware_status.values())
