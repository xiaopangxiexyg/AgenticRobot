"""Real-device assembly for the arm-free UMI collector."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

from robot_motion.runtime.lifecycle import run_best_effort_cleanup

from hardware.angle_trigger.angle_trigger import prepare_reader
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
from ...real_teleop.foot_pad_controls import FootPadSpecialArmInput
from ...sources.pose.vive import VivePoseSource, ViveTrackerConfig
from ..hand_state_polling import HandStateHistoryPoller
from ..sample_assembler import CameraFrameMatch
from .config import UmiCollectionConfig


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
    _started: bool = False

    def start(self) -> None:
        if self._started:
            return
        started: list[tuple[str, Any]] = []
        try:
            self.rig.start()
            started.append(("rig", self.rig))
            self.hand_state_poller.start()
            started.append(("hand_state_poller", self.hand_state_poller))
            self.vive.start()
            started.append(("vive", self.vive))
            for key, camera in self.cameras.items():
                camera.start()
                started.append((f"camera[{key}]", camera))
            self.angle_trigger.start()
            started.append(("angle_trigger", self.angle_trigger))
        except BaseException:
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


def build_umi_components(config: UmiCollectionConfig) -> UmiComponents:
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
    angle_reader = prepare_reader(config_root=config.angle_trigger_config_root)
    angle_trigger = AngleTriggerSource(
        angle_reader,
        rate_hz=config.angle_trigger_rate_hz,
        config_root=config.angle_trigger_config_root,
    )

    assignment = load_dmtac_hardware_assignment(
        config.dmtac_assignment,
        validate_current_usb=True,
    )
    backend_configs = dmtac_backend_configs_from_assignment(
        assignment,
        active_sides=("left", "right"),
    )
    missing = sorted({"left", "right"} - set(backend_configs))
    if missing:
        raise RuntimeError(f"DMTac assignment is missing UMI side(s): {missing}")
    for backend_config in backend_configs.values():
        backend_config.send_commands = bool(config.send_gripper_commands)
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
    foot_pad = FootPadSpecialArmInput.open_online() if config.foot_pad_enabled else None
    return UmiComponents(
        vive=vive,
        cameras=cameras,
        angle_trigger=angle_trigger,
        grippers=grippers,
        rig=rig,
        retargeters=retargeters,
        hand_state_poller=poller,
        foot_pad=foot_pad,
    )


__all__ = ["UmiComponents", "build_umi_components"]
