"""Real-device assembly for the arm-free UMI collector."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Protocol

from robot_motion.runtime.lifecycle import run_best_effort_cleanup

from hardware.angle_trigger.angle_trigger import load_configured_reader
from ...backends.grippers.dmtac import DMTacGripperBackend
from ...configs.dmtac_hardware import (
    dmtac_backend_configs_from_assignment,
    load_dmtac_hardware_assignment,
)
from ...retargeters.scalar_to_gripper import (
    ScalarToGripperRetargeter,
    ScalarToGripperRetargeterConfig,
)
from ...rigs.hand_only import HandOnlyRig
from ...sources.camera.realsense_rgb import (
    RealSenseRGBCameraConfig,
    RealSenseRGBSource,
)
from ...sources.hand.angle_trigger import AngleTriggerSource
from ...real_teleop.foot_pad_controls import (
    FootPadSpecialArmInput,
    FootPadUnavailableError,
)
from ...sources.pose.vive import VivePoseSource, ViveTrackerConfig
from ..hand_state_polling import HandStateHistoryPoller
from ..sample_assembler import CameraFrameMatch
from .config import UmiCollectionConfig


class UmiStartupProgress(Protocol):
    def begin(
        self,
        key: str,
        *,
        label: str,
        message: str,
        requirement: str = "feature_dependent",
    ) -> None: ...

    def succeed(self, key: str, *, message: str = "连接完成") -> None: ...

    def fail(self, key: str, error: str, *, message: str = "连接失败") -> None: ...


def _umi_startup_label(name: str) -> str:
    if name == "dmtac.rig":
        return "DMTac 双夹爪"
    if name == "dmtac.state_poller":
        return "DMTac 状态采样"
    if name == "vive":
        return "VIVE tracker"
    if name == "angle_trigger":
        return "AngleTrigger"
    if name.startswith("camera."):
        return f"腕部相机 {name.split('.', 1)[1]}"
    return name


def _umi_startup_requirement(name: str) -> str:
    return (
        "collection_core"
        if name in {"dmtac.rig", "dmtac.state_poller", "vive", "angle_trigger"}
        else "collection"
    )


@dataclass
class UmiComponents:
    vive: VivePoseSource
    cameras: dict[str, RealSenseRGBSource]
    angle_trigger: AngleTriggerSource
    grippers: dict[str, DMTacGripperBackend]
    rig: HandOnlyRig
    retargeters: dict[str, ScalarToGripperRetargeter]
    hand_state_poller: HandStateHistoryPoller
    foot_pad: FootPadSpecialArmInput | None = None
    foot_pad_error: str = ""
    strict_startup: bool = False
    startup_errors: dict[str, str] = field(default_factory=dict)
    _started: bool = False

    def _require_dmtac_ready(self) -> None:
        unavailable = []
        for side, backend in sorted(self.grippers.items()):
            if backend.is_ready():
                continue
            try:
                state = backend.get_state(include_tactile_arrays=False)
            except TypeError:
                state = backend.get_state()
            detail = str(state.get("telemetry_error") or "backend is not ready")
            unavailable.append(f"{side}={detail}")
        if unavailable:
            raise RuntimeError("DMTAC startup incomplete: " + "; ".join(unavailable))

    def start(self, *, startup_progress: UmiStartupProgress | None = None) -> None:
        if self._started:
            return
        if not self.strict_startup:
            self.startup_errors.clear()
            for name, component in (
                ("dmtac.rig", self.rig),
                ("dmtac.state_poller", self.hand_state_poller),
                ("vive", self.vive),
                *((f"camera.{key}", camera) for key, camera in self.cameras.items()),
                ("angle_trigger", self.angle_trigger),
            ):
                if startup_progress is not None:
                    startup_progress.begin(
                        name,
                        label=_umi_startup_label(name),
                        message="正在调用设备 start()",
                        requirement=_umi_startup_requirement(name),
                    )
                try:
                    component.start()
                    if name == "dmtac.rig":
                        self._require_dmtac_ready()
                except Exception as exc:  # noqa: BLE001 - report each independent device.
                    detail = f"{type(exc).__name__}: {exc}"
                    self.startup_errors[str(name)] = detail
                    if startup_progress is not None:
                        startup_progress.fail(name, detail)
                else:
                    if startup_progress is not None:
                        startup_progress.succeed(name)
            self._started = True
            return
        started: list[tuple[str, Any]] = []
        current_name = "dmtac.rig"
        try:
            if startup_progress is not None:
                startup_progress.begin(
                    "dmtac.rig",
                    label=_umi_startup_label("dmtac.rig"),
                    message="正在启动双夹爪 Rig",
                    requirement=_umi_startup_requirement("dmtac.rig"),
                )
            self.rig.start()
            self._require_dmtac_ready()
            started.append(("rig", self.rig))
            if startup_progress is not None:
                startup_progress.succeed("dmtac.rig")
                current_name = "dmtac.state_poller"
                startup_progress.begin(
                    "dmtac.state_poller",
                    label=_umi_startup_label("dmtac.state_poller"),
                    message="正在启动状态采样",
                    requirement=_umi_startup_requirement("dmtac.state_poller"),
                )
            self.hand_state_poller.start()
            started.append(("hand_state_poller", self.hand_state_poller))
            if startup_progress is not None:
                startup_progress.succeed("dmtac.state_poller")
                current_name = "vive"
                startup_progress.begin(
                    "vive",
                    label=_umi_startup_label("vive"),
                    message="正在启动 tracker 接收",
                    requirement=_umi_startup_requirement("vive"),
                )
            self.vive.start()
            started.append(("vive", self.vive))
            if startup_progress is not None:
                startup_progress.succeed("vive")
            for key, camera in self.cameras.items():
                name = f"camera.{key}"
                current_name = name
                if startup_progress is not None:
                    startup_progress.begin(
                        name,
                        label=_umi_startup_label(name),
                        message="正在启动视频流",
                        requirement=_umi_startup_requirement(name),
                    )
                camera.start()
                started.append((f"camera[{key}]", camera))
                if startup_progress is not None:
                    startup_progress.succeed(name, message="视频流已启动")
            current_name = "angle_trigger"
            if startup_progress is not None:
                startup_progress.begin(
                    "angle_trigger",
                    label=_umi_startup_label("angle_trigger"),
                    message="正在启动串口输入",
                    requirement=_umi_startup_requirement("angle_trigger"),
                )
            self.angle_trigger.start()
            started.append(("angle_trigger", self.angle_trigger))
            if startup_progress is not None:
                startup_progress.succeed("angle_trigger")
        except BaseException as exc:
            if startup_progress is not None:
                startup_progress.fail(
                    current_name,
                    f"{type(exc).__name__}: {exc}",
                )
            run_best_effort_cleanup(
                "umi_components.start.rollback",
                ((name, component.close) for name, component in reversed(started)),
            )
            raise
        self._started = True

    def close(self) -> None:
        steps = [("angle_trigger", self.angle_trigger.close)]
        steps.extend(
            (f"camera[{key}]", camera.close)
            for key, camera in reversed(tuple(self.cameras.items()))
        )
        steps.extend(
            [
                ("vive", self.vive.close),
                ("hand_state_poller", self.hand_state_poller.close),
                ("rig", self.rig.close),
            ]
        )
        if self.foot_pad is not None:
            steps.append(("foot_pad", self.foot_pad.close))
        run_best_effort_cleanup("umi_components.close", steps)
        self._started = False

    def camera_selectors(self) -> dict[str, Any]:
        def select(camera: RealSenseRGBSource, timestamp: float) -> CameraFrameMatch | None:
            frame = camera.frame_buffer().get_latest_before_or_at(timestamp)
            if frame is None:
                return None
            return CameraFrameMatch(
                camera_key=camera.camera_key,
                frame=frame,
                selected_timestamp=float(frame.source_timestamp),
                age_sec=float(timestamp - frame.source_timestamp),
            )

        return {
            key: partial(select, camera)
            for key, camera in self.cameras.items()
        }


def _configure_dmtac_backend_configs(
    config: UmiCollectionConfig,
    backend_configs: dict[str, Any],
) -> None:
    for backend_config in backend_configs.values():
        backend_config.send_commands = bool(config.send_gripper_commands)
        backend_config.tactile_read_hz = float(config.dmtac_tactile_read_hz)
        backend_config.command_hz = float(config.dmtac_gripper_command_hz)
        backend_config.telemetry_hz = float(config.dmtac_gripper_telemetry_hz)
        backend_config.gripper_telemetry_command_grace_s = float(
            config.dmtac_gripper_telemetry_command_grace_s
        )


def build_umi_components(
    config: UmiCollectionConfig,
    *,
    dmtac_assignment: dict[str, Any] | None = None,
) -> UmiComponents:
    tracker_configs = [
        ViveTrackerConfig(
            name=config.side(side).tracker_name,
            side=side,
            role=config.side(side).tracker_role,
            channel_name=config.side(side).tracker_channel,
            legacy_index=1 if side == "left" else 2,
            critical=True,
        )
        for side in ("left", "right")
    ]
    vive = VivePoseSource(
        tracker_configs,
        host=config.vive_host,
        port=config.vive_port,
        clock_sync_port=config.vive_clock_sync_port,
        name="vive_umi_pose",
    )
    cameras = {
        f"{side}_wrist": RealSenseRGBSource(
            RealSenseRGBCameraConfig(
                serial=config.side(side).camera_serial,
                width=640,
                height=480,
                fps=15,
                camera_key=f"{side}_wrist",
                name=f"umi_{side}_wrist_camera",
                keep_last=8,
                color_space="rgb",
                enable_depth=False,
            )
        )
        for side in ("left", "right")
    }
    angle_reader = load_configured_reader(config_root=config.angle_trigger_config_root)
    angle_trigger = AngleTriggerSource(
        angle_reader,
        rate_hz=config.angle_trigger_rate_hz,
        config_root=config.angle_trigger_config_root,
    )

    assignment = (
        dmtac_assignment
        if dmtac_assignment is not None
        else load_dmtac_hardware_assignment(
            config.dmtac_assignment,
            validate_current_usb=True,
        )
    )
    backend_configs = dmtac_backend_configs_from_assignment(
        assignment,
        active_sides=("left", "right"),
    )
    missing = sorted({"left", "right"} - set(backend_configs))
    if missing:
        raise RuntimeError(f"DMTac assignment is missing UMI side(s): {missing}")
    _configure_dmtac_backend_configs(config, backend_configs)
    grippers = {
        side: DMTacGripperBackend(backend_configs[side])
        for side in ("left", "right")
    }
    rig = HandOnlyRig(
        {f"{side}_hand": backend for side, backend in grippers.items()},
        name="umi_hand_only_rig",
        parallel_dispatch=True,
        fail_stop_on_dispatch_error=True,
    )
    retargeters = {
        side: ScalarToGripperRetargeter(
            ScalarToGripperRetargeterConfig(
                side=side,
                input_channel_name=f"{side}_angle_trigger",
                effector_name=f"{side}_hand",
                source_dim=None,
            )
        )
        for side in ("left", "right")
    }
    poller = HandStateHistoryPoller(backends_by_side=grippers)
    foot_pad = None
    foot_pad_error = ""
    if config.foot_pad_enabled:
        try:
            foot_pad = FootPadSpecialArmInput.open_online()
        except (FootPadUnavailableError, OSError, RuntimeError, ValueError) as exc:
            # UMI remains usable with the legacy GUI label controls when the
            # optional virtual keyboard is absent.  The adapter will only be
            # created when a valid 2/4-pedal device is actually online.
            foot_pad_error = str(exc)
    return UmiComponents(
        vive=vive,
        cameras=cameras,
        angle_trigger=angle_trigger,
        grippers=grippers,
        rig=rig,
        retargeters=retargeters,
        hand_state_poller=poller,
        foot_pad=foot_pad,
        foot_pad_error=foot_pad_error,
        strict_startup=config.hardware_admission_mode == "strict",
    )


__all__ = ["UmiComponents", "build_umi_components"]
