from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from ...calibration import TeleopCalibration
from .controller import InitArmCommandController


@dataclass(frozen=True)
class InitArmHomeCaptureProgress:
    active: bool
    completed: bool = False
    failed: bool = False
    side: str = ""
    message: str = ""
    sample_count: int = 0
    min_samples: int = 0
    valid: bool = False
    waiting_for_samples: bool = False
    elapsed_sec: float = 0.0
    timeout_sec: float = 0.0


def advance_init_arm_home_capture(
    *,
    controller: InitArmCommandController | None,
    channels: Mapping[str, object],
    calibration: TeleopCalibration | None,
    now: float,
    pose_from_channel: Callable[[object, TeleopCalibration], np.ndarray],
    mean_pose: Callable[[list[np.ndarray]], np.ndarray],
    resolve_robot_home: Callable[[str], tuple[np.ndarray, np.ndarray | None, str]],
    side_retargeter: Callable[[str], object],
    tcp_mode: str | None = None,
    log: Callable[[str, str], None] | None = None,
) -> InitArmHomeCaptureProgress:
    if controller is None or not controller.recapturing:
        return InitArmHomeCaptureProgress(active=False)
    if calibration is None:
        transition = controller.fail_recalibration(
            now=now,
            message="init-arm home recapture failed: calibration is unavailable",
        )
        return InitArmHomeCaptureProgress(
            active=True,
            failed=True,
            side=transition.side,
            message=transition.message,
        )

    side = controller.active_side
    started_at = controller.recalibration_started_at
    timeout_sec = controller.recalibration_timeout_sec
    elapsed_sec = max(0.0, float(now) - started_at)
    if elapsed_sec > timeout_sec:
        sample_count = controller.sample_count
        min_samples = controller.min_samples
        transition = controller.fail_recalibration(
            now=now,
            message=(
                f"init-arm {side} home recapture timed out: "
                f"{sample_count}/{min_samples} samples"
            ),
        )
        return InitArmHomeCaptureProgress(
            active=True,
            failed=True,
            side=side,
            message=transition.message,
            sample_count=sample_count,
            min_samples=min_samples,
            elapsed_sec=elapsed_sec,
            timeout_sec=timeout_sec,
            waiting_for_samples=True,
        )

    channel = channels.get(f"{side}_wrist_pose")
    valid = bool(channel is not None and getattr(channel, "valid", False))
    if valid:
        controller.add_recalibration_sample(pose_from_channel(channel, calibration))

    if controller.sample_count < controller.min_samples:
        return InitArmHomeCaptureProgress(
            active=True,
            side=side,
            sample_count=controller.sample_count,
            min_samples=controller.min_samples,
            valid=valid,
            waiting_for_samples=True,
            elapsed_sec=elapsed_sec,
            timeout_sec=timeout_sec,
        )

    samples = controller.recalibration_samples()
    min_samples = controller.min_samples
    human_home = mean_pose(samples)
    robot_home_desk, qpos, robot_source = resolve_robot_home(side)
    retargeter = side_retargeter(side)
    retargeter.set_home_anchors(
        human_pose_desk=human_home,
        robot_pose_desk=robot_home_desk,
    )
    calibration.capture_home_pose(f"{side}_arm", robot_home_desk, tcp_mode=tcp_mode)
    if qpos is not None:
        calibration.capture_home_qpos(f"{side}_arm", qpos)
    message = (
        f"init-arm {side} home recaptured from {robot_source}; "
        f"samples={len(samples)}"
    )
    if log is not None:
        log("InitArm", message)
    transition = controller.complete_recalibration(now=now, message=message)
    return InitArmHomeCaptureProgress(
        active=True,
        completed=True,
        side=side,
        message=transition.message,
        sample_count=len(samples),
        min_samples=min_samples,
        valid=True,
        waiting_for_samples=False,
        elapsed_sec=elapsed_sec,
        timeout_sec=timeout_sec,
    )
