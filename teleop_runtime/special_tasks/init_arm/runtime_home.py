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
    sides: tuple[str, ...] = ()
    qpos_operation_id: str = ""
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
    sides = controller.recalibration_sides or (side,)
    qpos_operation_id = str(controller.qpos_operation_id or "")
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
            sides=tuple(sides),
            qpos_operation_id=qpos_operation_id,
            message=transition.message,
            sample_count=sample_count,
            min_samples=min_samples,
            elapsed_sec=elapsed_sec,
            timeout_sec=timeout_sec,
            waiting_for_samples=True,
        )

    validity_by_side: dict[str, bool] = {}
    stability_reason_by_side: dict[str, str] = {}
    for capture_side in sides:
        channel = channels.get(f"{capture_side}_wrist_pose")
        source_timestamp = float(getattr(channel, "timestamp", float("nan")))
        source_age_sec = float(now) - source_timestamp
        valid = bool(
            channel is not None
            and getattr(channel, "valid", False)
            and np.isfinite(source_timestamp)
            and -0.05 <= source_age_sec <= 0.5
        )
        validity_by_side[capture_side] = valid
        if valid:
            try:
                pose = pose_from_channel(channel, calibration)
                controller.observe_recalibration_sample(
                    pose,
                    side=capture_side,
                    source_timestamp=source_timestamp,
                )
            except Exception as exc:  # noqa: BLE001 - fail closed at the recapture boundary.
                transition = controller.fail_recalibration(
                    now=now,
                    message=(
                        f"init-arm {capture_side} Home tracker sample rejected: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
                return InitArmHomeCaptureProgress(
                    active=True,
                    failed=True,
                    side=side,
                    sides=tuple(sides),
                    qpos_operation_id=qpos_operation_id,
                    message=transition.message,
                    sample_count=controller.sample_count,
                    min_samples=controller.min_samples,
                    valid=False,
                    elapsed_sec=elapsed_sec,
                    timeout_sec=timeout_sec,
                )
            stability_reason_by_side[capture_side] = (
                controller.recalibration_stability_reason(capture_side)
            )
        else:
            stability_reason_by_side[capture_side] = (
                "waiting_for_valid_fresh_tracker_sample"
            )
    valid = all(validity_by_side.values())
    stability_reason = "; ".join(
        f"{capture_side}:{stability_reason_by_side[capture_side]}"
        for capture_side in sides
        if stability_reason_by_side.get(capture_side)
    )

    if controller.sample_count < controller.min_samples:
        return InitArmHomeCaptureProgress(
            active=True,
            side=side,
            sides=tuple(sides),
            qpos_operation_id=qpos_operation_id,
            sample_count=controller.sample_count,
            min_samples=controller.min_samples,
            valid=valid,
            waiting_for_samples=True,
            waiting_for_stability=controller.sample_count == 0,
            stability_reason=stability_reason,
            elapsed_sec=elapsed_sec,
            timeout_sec=timeout_sec,
        )

    min_samples = controller.min_samples
    completed_sample_count = controller.sample_count
    messages: list[str] = []
    sample_counts: list[int] = []
    prepared: list[
        tuple[
            str,
            list[np.ndarray],
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            str,
            object,
        ]
    ] = []
    try:
        for capture_side in sides:
            samples = controller.recalibration_samples(capture_side)
            human_home = mean_pose(samples)
            robot_home_desk, qpos, robot_source = resolve_robot_home(capture_side)
            prepared.append(
                (
                    capture_side,
                    samples,
                    human_home,
                    robot_home_desk,
                    qpos,
                    robot_source,
                    side_retargeter(capture_side),
                )
            )
    except Exception as exc:  # noqa: BLE001 - qpos recapture failure must fail to Hold.
        transition = controller.fail_recalibration(
            now=now,
            message=(
                "init-arm Home recapture could not resolve a complete robot snapshot: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
        return InitArmHomeCaptureProgress(
            active=True,
            failed=True,
            side=side,
            sides=tuple(sides),
            qpos_operation_id=qpos_operation_id,
            message=transition.message,
            sample_count=completed_sample_count,
            min_samples=min_samples,
            valid=True,
            waiting_for_samples=False,
            elapsed_sec=elapsed_sec,
            timeout_sec=timeout_sec,
        )

    try:
        for (
            capture_side,
            samples,
            human_home,
            robot_home_desk,
            qpos,
            robot_source,
            retargeter,
        ) in prepared:
            sample_counts.append(len(samples))
            retargeter.set_home_anchors(
                human_pose_desk=human_home,
                robot_pose_desk=robot_home_desk,
            )
            calibration.capture_home_pose(
                f"{capture_side}_arm",
                robot_home_desk,
                tcp_mode=tcp_mode,
            )
            if qpos is not None:
                calibration.capture_home_qpos(f"{capture_side}_arm", qpos)
            messages.append(
                f"{capture_side} from {robot_source} ({len(samples)} samples)"
            )
    except Exception as exc:  # noqa: BLE001 - any partial anchor update requires safe Hold.
        transition = controller.fail_recalibration(
            now=now,
            message=(
                "init-arm Home recapture apply failed; runtime Hold is required: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
        return InitArmHomeCaptureProgress(
            active=True,
            failed=True,
            side=side,
            sides=tuple(sides),
            qpos_operation_id=qpos_operation_id,
            message=transition.message,
            sample_count=min(sample_counts) if sample_counts else min_samples,
            min_samples=min_samples,
            valid=True,
            waiting_for_samples=False,
            requires_hold=True,
            elapsed_sec=elapsed_sec,
            timeout_sec=timeout_sec,
        )
    message = "init-arm tracker Home recaptured: " + "; ".join(messages)
    if log is not None:
        log("InitArm", message)
    transition = controller.complete_recalibration(now=now, message=message)
    return InitArmHomeCaptureProgress(
        active=True,
        completed=True,
        side=side,
        sides=tuple(sides),
        qpos_operation_id=qpos_operation_id,
        message=transition.message,
        sample_count=min(sample_counts),
        min_samples=min_samples,
        valid=True,
        waiting_for_samples=False,
        elapsed_sec=elapsed_sec,
        timeout_sec=timeout_sec,
    )
