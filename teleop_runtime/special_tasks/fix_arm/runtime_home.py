from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from ...calibration import TeleopCalibration
from .controller import FixArmCommandController


@dataclass(frozen=True)
class FixArmHomeCaptureProgress:
    active: bool
    completed: bool = False
    failed: bool = False
    side: str = ""
    message: str = ""
    sample_count: int = 0
    min_samples: int = 0
    valid: bool = False
    waiting_for_samples: bool = False
    waiting_for_stability: bool = False
    stability_reason: str = ""
    requires_hold: bool = False
    elapsed_sec: float = 0.0
    timeout_sec: float = 0.0


def advance_fix_arm_home_capture(
    *,
    controller: FixArmCommandController | None,
    channels: Mapping[str, object],
    calibration: TeleopCalibration | None,
    now: float,
    pose_from_channel: Callable[[object, TeleopCalibration], np.ndarray],
    mean_pose: Callable[[list[np.ndarray]], np.ndarray],
    resolve_robot_home: Callable[[str], tuple[np.ndarray, np.ndarray | None, str]],
    side_retargeter: Callable[[str], object],
    tcp_mode: str | None = None,
    log: Callable[[str, str], None] | None = None,
) -> FixArmHomeCaptureProgress:
    if controller is None or not controller.recapturing:
        return FixArmHomeCaptureProgress(active=False)
    if calibration is None:
        transition = controller.fail_recalibration(
            now=now,
            message="fix-arm home recapture failed: calibration is unavailable",
        )
        return FixArmHomeCaptureProgress(
            active=True,
            failed=True,
            side=transition.side,
            message=transition.message,
        )

    side = controller.fixed_side
    started_at = controller.recalibration_started_at
    timeout_sec = controller.recalibration_timeout_sec
    elapsed_sec = max(0.0, float(now) - started_at)
    if elapsed_sec > timeout_sec:
        sample_count = controller.sample_count
        min_samples = controller.min_samples
        transition = controller.fail_recalibration(
            now=now,
            message=(
                f"fix-arm {side} home recapture timed out: "
                f"{sample_count}/{min_samples} samples"
            ),
        )
        return FixArmHomeCaptureProgress(
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
    source_timestamp = float(getattr(channel, "timestamp", float("nan")))
    source_age_sec = float(now) - source_timestamp
    valid = bool(
        channel is not None
        and getattr(channel, "valid", False)
        and np.isfinite(source_timestamp)
        and -0.05 <= source_age_sec <= 0.5
    )
    if valid:
        try:
            pose = pose_from_channel(channel, calibration)
            controller.observe_recalibration_sample(
                pose,
                source_timestamp=source_timestamp,
            )
        except Exception as exc:  # noqa: BLE001 - keep the fixed command active.
            transition = controller.fail_recalibration(
                now=now,
                message=(
                    f"fix-arm {side} Home tracker sample rejected: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            return FixArmHomeCaptureProgress(
                active=True,
                failed=True,
                side=side,
                message=transition.message,
                sample_count=controller.sample_count,
                min_samples=controller.min_samples,
                valid=False,
                elapsed_sec=elapsed_sec,
                timeout_sec=timeout_sec,
            )

    stability_reason = (
        controller.recalibration_stability_reason
        if valid
        else "waiting_for_valid_fresh_tracker_sample"
    )

    if controller.sample_count < controller.min_samples:
        return FixArmHomeCaptureProgress(
            active=True,
            side=side,
            sample_count=controller.sample_count,
            min_samples=controller.min_samples,
            valid=valid,
            waiting_for_samples=True,
            waiting_for_stability=controller.sample_count == 0,
            stability_reason=stability_reason,
            elapsed_sec=elapsed_sec,
            timeout_sec=timeout_sec,
        )

    samples = controller.recalibration_samples()
    min_samples = controller.min_samples
    try:
        human_home = mean_pose(samples)
        robot_home_desk, qpos, robot_source = resolve_robot_home(side)
        retargeter = side_retargeter(side)
    except Exception as exc:  # noqa: BLE001 - no anchor has been modified yet.
        transition = controller.fail_recalibration(
            now=now,
            message=(
                f"fix-arm {side} Home recapture could not prepare a complete snapshot: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
        return FixArmHomeCaptureProgress(
            active=True,
            failed=True,
            side=side,
            message=transition.message,
            sample_count=len(samples),
            min_samples=min_samples,
            valid=True,
            elapsed_sec=elapsed_sec,
            timeout_sec=timeout_sec,
        )
    try:
        retargeter.set_home_anchors(
            human_pose_desk=human_home,
            robot_pose_desk=robot_home_desk,
        )
        calibration.capture_home_pose(f"{side}_arm", robot_home_desk, tcp_mode=tcp_mode)
        if qpos is not None:
            calibration.capture_home_qpos(f"{side}_arm", qpos)
    except Exception as exc:  # noqa: BLE001 - a partial anchor update requires Hold.
        transition = controller.fail_recalibration(
            now=now,
            message=(
                f"fix-arm {side} Home recapture apply failed; runtime Hold is required: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
        return FixArmHomeCaptureProgress(
            active=True,
            failed=True,
            side=side,
            message=transition.message,
            sample_count=len(samples),
            min_samples=min_samples,
            valid=True,
            requires_hold=True,
            elapsed_sec=elapsed_sec,
            timeout_sec=timeout_sec,
        )
    message = (
        f"fix-arm {side} home recaptured from {robot_source}; "
        f"samples={len(samples)}"
    )
    if log is not None:
        log("FixArm", message)
    transition = controller.complete_recalibration(now=now, message=message)
    return FixArmHomeCaptureProgress(
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
