"""Canonical runtime-Home transaction and its operator diagnostics.

This module owns the real-teleop Home capture transaction that was historically
embedded in the retired SAPIEN debug executable. It does not construct or close
devices and it does not own the outer control loop.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import time
import warnings

import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_motion.rm65 import (
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
    DualArmSceneProfile,
)
from sim.rm65.sapien_renderer import (
    DebugBoxOverlay,
    DebugBoxTemplate,
    SapienArmHandRenderer,
)

from ..backends.arms import RM65ArmBackend
from ..backends.hands import InspireTacHandBackend
from ..calibration import TeleopCalibration
from ..cli_value_parsers import _normalize_active_sides
from ..hand_render import inspire_q6_to_render_qpos, open_hand_render_qpos
from ..hand_tcp_alignment import vive_tracker_local_translation_offset_from_args
from ..initial_alignment import TeleopInitialAlignment
from ..log_utils import ChangeGate, log_event as _log
from ..recovery import (
    component_side_retargeter as _side_retargeter,
    component_tcp_pose_desk as _preview_tcp_pose_desk,
)
from ..schemas import TeleopObservation, latest_channel_timestamp, merge_teleop_commands
from ..session import TeleopSession
from ..sources.misc import KeyboardButtonSource
from ..special_tasks.home_capture_stability import HomeCaptureStabilityState
from ..transforms import as_pose_matrix, identity_pose
from .arm_runtime import _sync_dual_arm_retargeter_from_preview
from .arm_state_sync import _qpos_summary, _state_qpos
from .component_builder import _collision_debug_box_templates
from .components import RealTeleopComponents

_TRACKER_BEFORE_CORRECTION_MARKER_SIZE_M = (0.018, 0.018, 0.018)
_TRACKER_AFTER_CORRECTION_MARKER_SIZE_M = (0.024, 0.024, 0.024)
_TRACKER_BEFORE_CORRECTION_COLOR_RGBA = (1.0, 0.55, 0.05, 1.0)
_TRACKER_AFTER_CORRECTION_COLOR_RGBA = (0.05, 0.75, 1.0, 1.0)
DEFAULT_RUNTIME_HOME_START_DELAY_SEC = 2.0

def _active_side_set(components) -> set[str]:
    return set(_normalize_active_sides(getattr(components, "active_sides", ("left", "right"))))

def _side_state_backend(components, side: str) -> RM65ArmBackend | None:
    return components.left_state_backend if side == "left" else components.right_state_backend

def _vive_tracker_local_translation_offset(args: argparse.Namespace) -> np.ndarray | None:
    return vive_tracker_local_translation_offset_from_args(args)

def _has_live_real_state(components: RealTeleopComponents) -> bool:
    return any(_side_state_backend(components, side) is not None for side in components.active_sides)

def _preview_should_follow_live_real_state(
    *,
    arm_mode: str,
    components: RealTeleopComponents,
) -> bool:
    return arm_mode == "real" and _has_live_real_state(components)

def _manual_calibration_confirmation_required(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> bool:
    _ = components
    return not bool(args.auto_calibrate)

def _pose_translation_summary(pose: np.ndarray | None) -> str:
    if pose is None:
        return "None"
    return str(np.round(np.asarray(pose[:3, 3], dtype=float), 4).tolist())

def _pose_rotation_summary(pose: np.ndarray | None) -> str:
    if pose is None:
        return "None"
    pose_matrix = as_pose_matrix(pose)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        euler_deg = R.from_matrix(pose_matrix[:3, :3]).as_euler("xyz", degrees=True)
    return str(np.round(np.asarray(euler_deg, dtype=float), 1).tolist())

def _relative_rotation_summary(reference_pose: np.ndarray | None, pose: np.ndarray | None) -> str:
    if reference_pose is None or pose is None:
        return "None"
    relative_pose = np.linalg.inv(as_pose_matrix(reference_pose)) @ as_pose_matrix(pose)
    return _pose_rotation_summary(relative_pose)

def _dominant_axis_label(axis: np.ndarray) -> str:
    vector = np.asarray(axis, dtype=float).reshape(3)
    dominant_index = int(np.argmax(np.abs(vector)))
    dominant_value = float(vector[dominant_index])
    sign = "+" if dominant_value >= 0.0 else "-"
    return f"{sign}{'xyz'[dominant_index]}({abs(dominant_value):.3f})"

def _tracker_axis_mapping_summary(offset: np.ndarray | None) -> str:
    if offset is None:
        return "None"
    rotation = as_pose_matrix(offset)[:3, :3]
    tracker_axes = np.eye(3, dtype=float)
    return ", ".join(
        f"tracker_{axis_name}->hand_{_dominant_axis_label(rotation.T @ tracker_axes[index])}"
        for index, axis_name in enumerate(("x", "y", "z"))
    )

def _tracker_rpy_mapping_summary(offset: np.ndarray | None) -> str:
    if offset is None:
        return "None"
    rotation = as_pose_matrix(offset)[:3, :3]
    tracker_axes = np.eye(3, dtype=float)
    labels = ("roll(x)", "pitch(y)", "yaw(z)")
    return ", ".join(
        f"{labels[index]}->{_dominant_axis_label(rotation.T @ tracker_axes[index])}"
        for index in range(3)
    )

def _channel_pose_raw_in_desk(channel, calibration: TeleopCalibration) -> np.ndarray:
    pose = as_pose_matrix(channel.value)
    frame_name = channel.frame_name or ""
    if frame_name in {"vive_world", "world"}:
        pose_operator = calibration.world_to_operator(pose, tracker_role=None)
        return calibration.operator_to_desk(pose_operator)
    if frame_name.endswith("_operator") or frame_name == "operator":
        return calibration.operator_to_desk(pose)
    if frame_name in {calibration.desk_frame.name, "desk"}:
        return pose
    raise ValueError(f"Unsupported calibration input frame: {frame_name or '<none>'}")

def _robot_home_mode_label(components: RealTeleopComponents) -> str:
    if components.initial_alignment is not None:
        return "tracker_home + alignment_robot_home"
    if _has_live_real_state(components):
        return "tracker_home + real_robot_home"
    return "tracker_home + preview_robot_home"

def _resolve_robot_home_for_side(
    components: RealTeleopComponents,
    *,
    side: str,
    qpos_snapshot: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    calibration = components.session.calibration
    assert calibration is not None
    preview_backend = (
        components.left_preview_backend
        if side == "left"
        else components.right_preview_backend
    )
    state_backend = (
        components.left_state_backend
        if side == "left"
        else components.right_state_backend
    )

    if components.initial_alignment is not None:
        alignment_state = components.initial_alignment.get_side(side)
        if alignment_state is not None:
            qpos = None
            if alignment_state.arm_qpos is not None:
                qpos = np.asarray(alignment_state.arm_qpos, dtype=float).copy()
                preview_backend.set_local_qpos(qpos)
                _sync_dual_arm_retargeter_from_preview(components)
            if alignment_state.tcp_pose_desk is not None:
                return alignment_state.tcp_pose_desk.copy(), qpos, "initial_alignment_pose_desk"
            if alignment_state.tcp_pose_base is not None:
                return (
                    calibration.get_arm_for_side(side).T_D_B @ alignment_state.tcp_pose_base,
                    qpos,
                    "initial_alignment_pose_base",
                )
            if qpos is not None:
                return _preview_tcp_pose_desk(components, side=side), qpos, "initial_alignment_qpos"

    if qpos_snapshot is not None and side in qpos_snapshot:
        anchor_qpos = np.asarray(qpos_snapshot[side], dtype=float).reshape(6).copy()
        preview_backend.set_local_qpos(anchor_qpos)
        _sync_dual_arm_retargeter_from_preview(components)
        return (
            _preview_tcp_pose_desk(components, side=side),
            anchor_qpos,
            "real_atomic_home_snapshot",
        )

    if state_backend is not None:
        state = state_backend.refresh_state_from_hardware()
        qpos = state.get("current_qpos")
        if qpos is not None:
            qpos_array = np.asarray(qpos, dtype=float).copy()
            preview_backend.set_local_qpos(qpos_array)
            # A preview backend may normalize a tiny SDK feedback quantization
            # overshoot at a hard endpoint. Persist that bounded anchor for
            # this session's home/IK seed rather than the raw observation.
            preview_state = preview_backend.get_state()
            anchor_qpos = preview_state.get("current_qpos")
            if anchor_qpos is None:
                raise RuntimeError(f"{side}_arm preview did not retain a local qpos anchor.")
            anchor_qpos = np.asarray(anchor_qpos, dtype=float).reshape(6).copy()
            normalization = preview_state.get("local_qpos_normalization")
            if isinstance(normalization, dict) and normalization.get("status") == "clipped":
                clipped_joints = normalization.get("clipped_joint_indices", [])
                _log(
                    "Calibration",
                    f"{side}_arm hardware qpos had a tiny endpoint quantization overshoot; "
                    f"preview/home anchor clipped at joint(s) {clipped_joints}. "
                    "The arm remains at the hard limit; move it away before normal tracking.",
                )
            _sync_dual_arm_retargeter_from_preview(components)
            return _preview_tcp_pose_desk(components, side=side), anchor_qpos, "real_current_state"

    qpos = preview_backend.get_state()["current_qpos"]
    return _preview_tcp_pose_desk(components, side=side), qpos, "preview_current_state"

def _tracker_after_height_correction_visual_pose(
    retargeter: PoseToArmRetargeter,
    tracker_pose_desk: np.ndarray,
) -> np.ndarray:
    corrected = retargeter.prepare_input_pose_desk(tracker_pose_desk)
    visual_pose = corrected.copy()
    visual_pose[:3, :3] = tracker_pose_desk[:3, :3]
    return visual_pose

def _debug_marker_pose_at_origin(pose: np.ndarray) -> np.ndarray:
    marker_pose = identity_pose()
    marker_pose[:3, 3] = as_pose_matrix(pose)[:3, 3]
    return marker_pose

def _tracker_correction_visual_poses(
    components: RealTeleopComponents,
    *,
    side: str,
    channels: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not channels:
        return None
    calibration = components.session.calibration
    assert calibration is not None
    channel = channels.get(f"{side}_wrist_pose")
    if channel is None or not channel.valid:
        return None
    before_pose = _channel_pose_in_desk(channel, calibration)
    after_pose = _tracker_after_height_correction_visual_pose(
        _side_retargeter(components, side),
        before_pose,
    )
    return before_pose, after_pose

def _build_renderer_debug_frames(
    components: RealTeleopComponents,
    *,
    channels: dict[str, object] | None,
) -> dict[str, np.ndarray]:
    calibration = components.session.calibration
    assert calibration is not None
    frames: dict[str, np.ndarray] = {"desk": identity_pose()}
    for side in components.active_sides:
        retargeter = _side_retargeter(components, side)
        arm = calibration.get_arm_for_side(side)
        if arm.T_D_B is not None:
            frames[f"{side}_arm_base"] = arm.T_D_B.copy()
        frames[f"{side}_tcp_actual"] = _preview_tcp_pose_desk(components, side=side)
        if retargeter.last_target_pose_desk is not None:
            frames[f"{side}_tcp_target"] = retargeter.last_target_pose_desk
        if retargeter.home_human_pose_desk is not None:
            frames[f"{side}_hand_home"] = retargeter.home_human_pose_desk
        if retargeter.home_robot_pose_desk is not None:
            frames[f"{side}_robot_home"] = retargeter.home_robot_pose_desk
        try:
            tracker_poses = _tracker_correction_visual_poses(
                components,
                side=side,
                channels=channels,
            )
        except Exception:
            continue
        if tracker_poses is None:
            continue
        before_pose, after_pose = tracker_poses
        frames[f"{side}_tracker_before_correction"] = before_pose
        frames[f"{side}_tracker_after_correction"] = after_pose
        frames[f"{side}_tracker"] = after_pose
    return frames

def _build_renderer_debug_boxes(
    components: RealTeleopComponents,
    *,
    channels: dict[str, object] | None = None,
) -> dict[str, DebugBoxOverlay]:
    profile = components.collision_scene_profile
    boxes = {
        spec.name: DebugBoxOverlay(
            pose_matrix=_preview_tcp_pose_desk(components, side=spec.side)
            @ spec.parent_to_box_pose_matrix(),
            size_xyz=spec.box_size_xyz,
            color_rgba=spec.color_rgba,
        )
        for spec in profile.hand_proxy_boxes
    }
    boxes.update(
        {
            spec.name: DebugBoxOverlay(
                pose_matrix=spec.world_to_box_pose_matrix(),
                size_xyz=spec.box_size_xyz,
                color_rgba=spec.color_rgba,
            )
            for spec in profile.fixed_scene_boxes
        }
    )
    for side in components.active_sides:
        try:
            tracker_poses = _tracker_correction_visual_poses(
                components,
                side=side,
                channels=channels,
            )
        except Exception:
            continue
        if tracker_poses is None:
            continue
        before_pose, after_pose = tracker_poses
        boxes[f"{side}_tracker_before_correction_marker"] = DebugBoxOverlay(
            pose_matrix=_debug_marker_pose_at_origin(before_pose),
            size_xyz=_TRACKER_BEFORE_CORRECTION_MARKER_SIZE_M,
            color_rgba=_TRACKER_BEFORE_CORRECTION_COLOR_RGBA,
        )
        boxes[f"{side}_tracker_after_correction_marker"] = DebugBoxOverlay(
            pose_matrix=_debug_marker_pose_at_origin(after_pose),
            size_xyz=_TRACKER_AFTER_CORRECTION_MARKER_SIZE_M,
            color_rgba=_TRACKER_AFTER_CORRECTION_COLOR_RGBA,
        )
    return boxes

def _emit_tracker_axis_mapping_logs(components: RealTeleopComponents) -> None:
    calibration = components.session.calibration
    assert calibration is not None
    for tracker_role in tuple(f"{side}_wrist" for side in components.active_sides):
        offset = calibration.tracker_offsets.get(tracker_role)
        if offset is None:
            continue
        _log(
            "TrackerAxes",
            f"{tracker_role} tracker_mount_compensation_rpy_xyz_deg={_pose_rotation_summary(offset)} "
            f"tracker_axis_to_hand_axis={_tracker_axis_mapping_summary(offset)} "
            f"tracker_rpy_to_hand_rpy={_tracker_rpy_mapping_summary(offset)}",
        )

def _emit_home_alignment_logs(components: RealTeleopComponents) -> None:
    for side in components.active_sides:
        retargeter = _side_retargeter(components, side)
        if retargeter.home_human_pose_desk is None or retargeter.home_robot_pose_desk is None:
            continue
        relative_pose = (
            np.linalg.inv(retargeter.home_human_pose_desk) @ retargeter.home_robot_pose_desk
        )
        _log(
            "HomeAxes",
            f"{side} home_human_rpy_xyz_deg={_pose_rotation_summary(retargeter.home_human_pose_desk)} "
            f"home_robot_rpy_xyz_deg={_pose_rotation_summary(retargeter.home_robot_pose_desk)} "
            f"robot_from_human_rpy_xyz_deg={_pose_rotation_summary(relative_pose)}",
        )

def _emit_collision_proxy_logs(components: RealTeleopComponents) -> None:
    profile = components.collision_scene_profile
    if not profile.hand_proxy_boxes and not profile.fixed_scene_boxes:
        return
    expected_self_pairs = {
        (
            spec.geometry_name,
            "arm_link2_0" if spec.side == "left" else "right_arm_link2_0",
        )
        for spec in profile.hand_proxy_boxes
    }
    self_pairs = tuple(pair for pair in profile.collision_pairs if pair in expected_self_pairs)
    _log(
        "CollisionProxy",
        f"profile={profile.name} urdf={profile.urdf_path} self_pairs={self_pairs}",
    )
    for spec in profile.hand_proxy_boxes:
        _log(
            "CollisionProxy",
            f"{spec.side} geometry={spec.geometry_name} parent={spec.parent_link_name} "
            f"size_m={np.round(np.asarray(spec.box_size_xyz, dtype=float), 4).tolist()} "
            f"joint_xyz_m={np.round(np.asarray(spec.joint_origin_xyz, dtype=float), 4).tolist()} "
            f"joint_rpy_deg={np.round(np.rad2deg(np.asarray(spec.joint_origin_rpy, dtype=float)), 1).tolist()} "
            f"box_xyz_m={np.round(np.asarray(spec.box_origin_xyz, dtype=float), 4).tolist()} "
            f"box_rpy_deg={np.round(np.rad2deg(np.asarray(spec.box_origin_rpy, dtype=float)), 1).tolist()}",
        )
    for spec in profile.fixed_scene_boxes:
        _log(
            "CollisionProxy",
            f"fixed_box geometry={spec.geometry_name} parent={spec.parent_link_name} "
            f"size_m={np.round(np.asarray(spec.box_size_xyz, dtype=float), 4).tolist()} "
            f"joint_xyz_m={np.round(np.asarray(spec.joint_origin_xyz, dtype=float), 4).tolist()} "
            f"joint_rpy_deg={np.round(np.rad2deg(np.asarray(spec.joint_origin_rpy, dtype=float)), 1).tolist()} "
            f"box_xyz_m={np.round(np.asarray(spec.box_origin_xyz, dtype=float), 4).tolist()} "
            f"box_rpy_deg={np.round(np.rad2deg(np.asarray(spec.box_origin_rpy, dtype=float)), 1).tolist()}",
        )

def _emit_runtime_orientation_logs(
    components: RealTeleopComponents,
    *,
    channels: dict[str, object],
    now: float,
    gate: ChangeGate,
) -> None:
    calibration = components.session.calibration
    assert calibration is not None
    for side in components.active_sides:
        retargeter = _side_retargeter(components, side)
        channel = channels.get(f"{side}_wrist_pose")
        if channel is None or not channel.valid:
            continue
        try:
            raw_pose = _channel_pose_raw_in_desk(channel, calibration)
            human_pose = _channel_pose_in_desk(channel, calibration)
        except Exception:
            continue
        target_pose = retargeter.last_target_pose_desk
        actual_pose = _preview_tcp_pose_desk(components, side=side)
        snapshot = (
            tuple(np.round(raw_pose[:3, :3].reshape(-1), 2).tolist()),
            tuple(np.round(human_pose[:3, :3].reshape(-1), 2).tolist()),
            tuple(np.round(actual_pose[:3, :3].reshape(-1), 2).tolist()),
            None
            if target_pose is None
            else tuple(np.round(target_pose[:3, :3].reshape(-1), 2).tolist()),
        )
        if not gate.should_emit(
            f"orientation_debug_{side}",
            snapshot,
            now=now,
            heartbeat_sec=1.5,
        ):
            continue
        _log(
            "Orientation",
            f"{side} raw_rpy_xyz_deg={_pose_rotation_summary(raw_pose)} "
            f"hand_rpy_xyz_deg={_pose_rotation_summary(human_pose)} "
            f"target_rpy_xyz_deg={_pose_rotation_summary(target_pose)} "
            f"actual_rpy_xyz_deg={_pose_rotation_summary(actual_pose)} "
            f"human_rel_rpy_xyz_deg={_relative_rotation_summary(retargeter.home_human_pose_desk, human_pose)} "
            f"target_rel_rpy_xyz_deg={_relative_rotation_summary(retargeter.home_robot_pose_desk, target_pose)} "
            f"actual_rel_rpy_xyz_deg={_relative_rotation_summary(retargeter.home_robot_pose_desk, actual_pose)}",
            now=now,
        )

def _sync_preview_from_real_state(
    components: RealTeleopComponents,
    *,
    now: float | None = None,
    required: bool,
) -> dict[str, bool]:
    synced = {side: False for side in components.active_sides}
    backends = (
        ("left", components.left_state_backend, components.left_preview_backend),
        ("right", components.right_state_backend, components.right_preview_backend),
    )
    active = [
        (side, state_backend, preview_backend)
        for side, state_backend, preview_backend in backends
        if side in _active_side_set(components) and state_backend is not None
    ]
    state_read_executor = getattr(components, "arm_state_read_executor", None)
    parallel_submit_failed = False
    if state_read_executor is not None and len(active) > 1:
        try:
            future_map = {
                state_read_executor.submit(state_backend.refresh_state_from_hardware): (
                    side,
                    preview_backend,
                )
                for side, state_backend, preview_backend in active
            }
        except RuntimeError:
            parallel_submit_failed = True
        else:
            for future, (side, preview_backend) in future_map.items():
                state = future.result()
                qpos = state.get("current_qpos")
                if qpos is None:
                    if required:
                        raise RuntimeError(
                            f"{side}_arm real-state sync failed: no current_qpos available from hardware."
                        )
                    continue
                preview_backend.set_local_qpos(qpos)
                synced[side] = True
    if state_read_executor is None or len(active) <= 1 or parallel_submit_failed:
        for side, state_backend, preview_backend in active:
            state = state_backend.refresh_state_from_hardware()
            qpos = state.get("current_qpos")
            if qpos is None:
                if required:
                    raise RuntimeError(
                        f"{side}_arm real-state sync failed: no current_qpos available from hardware."
                    )
                continue
            preview_backend.set_local_qpos(qpos)
            synced[side] = True
    if components.dual_arm_retargeter is not None:
        left_qpos = components.left_preview_backend.get_state()["current_qpos"]
        right_qpos = components.right_preview_backend.get_state()["current_qpos"]
        if left_qpos is not None and right_qpos is not None:
            components.dual_arm_retargeter.set_current_qpos_pair(
                left_qpos=left_qpos,
                right_qpos=right_qpos,
            )
    return synced

def _mean_pose(
    poses: list[np.ndarray],
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    if not poses:
        raise ValueError("Cannot average an empty pose list.")
    if weights is None:
        normalized_weights = np.full(len(poses), 1.0 / len(poses), dtype=float)
    else:
        weight_values = np.asarray(weights, dtype=float).reshape(-1)
        if weight_values.size != len(poses):
            raise ValueError("Pose-average weights must match the pose count.")
        if not np.all(np.isfinite(weight_values)) or np.any(weight_values <= 0.0):
            raise ValueError("Pose-average weights must be finite and positive.")
        normalized_weights = weight_values / float(np.sum(weight_values))
    mean_pose = np.eye(4, dtype=float)
    mean_pose[:3, 3] = np.sum(
        [weight * pose[:3, 3] for weight, pose in zip(normalized_weights, poses)],
        axis=0,
    )
    rot_sum = np.sum(
        [weight * pose[:3, :3] for weight, pose in zip(normalized_weights, poses)],
        axis=0,
    )
    u, _, vt = np.linalg.svd(rot_sum)
    mean_pose[:3, :3] = u @ vt
    if np.linalg.det(mean_pose[:3, :3]) < 0.0:
        u[:, -1] *= -1.0
        mean_pose[:3, :3] = u @ vt
    return mean_pose

def _capture_atomic_real_qpos_snapshot(
    components: RealTeleopComponents,
    *,
    required: bool,
) -> dict[str, np.ndarray]:
    """Read one dual-arm Home snapshot and normalize it through preview state.

    The returned qpos values are the sole real-arm anchors for this Home
    transaction. Callers must reuse them for robot Home, IK synchronization,
    and command-backend reset instead of reading hardware again.
    """

    active: list[tuple[str, object, object]] = []
    for side, state_backend, preview_backend in (
        ("left", components.left_state_backend, components.left_preview_backend),
        ("right", components.right_state_backend, components.right_preview_backend),
    ):
        if side in _active_side_set(components) and state_backend is not None:
            active.append((side, state_backend, preview_backend))

    if required and len(active) != len(_active_side_set(components)):
        available = {side for side, _, _ in active}
        missing = sorted(_active_side_set(components) - available)
        raise RuntimeError(
            "runtime Home atomic arm snapshot is missing real state backend(s): "
            f"{missing}"
        )

    states: dict[str, Mapping[str, object]] = {}

    def read_one(item: tuple[str, object, object]) -> tuple[str, Mapping[str, object]]:
        side, state_backend, _ = item
        state = state_backend.refresh_state_from_hardware()
        if not isinstance(state, Mapping):
            raise RuntimeError(f"{side}_arm returned a non-mapping hardware state.")
        return side, state

    configured_executor = getattr(components, "arm_state_read_executor", None)
    parallel_reads_enabled = bool(
        configured_executor is not None
        and int(getattr(configured_executor, "_max_workers", 1)) > 1
    )
    if len(active) > 1 and parallel_reads_enabled:
        # Runtime Home is a rare control boundary. Use a fresh executor rather
        # than the startup executor, which is deliberately shut down after the
        # first Home and cannot serve later Hold -> Home transitions. Preserve
        # an explicit --arm-io-threads=1 request by keeping this read sequential.
        with ThreadPoolExecutor(
            max_workers=len(active),
            thread_name_prefix="rm65_home_snapshot",
        ) as executor:
            futures = [executor.submit(read_one, item) for item in active]
            for future in futures:
                side, state = future.result()
                states[side] = state
    else:
        for item in active:
            side, state = read_one(item)
            states[side] = state

    snapshot: dict[str, np.ndarray] = {}
    for side, _, preview_backend in active:
        state = states.get(side)
        qpos = None if state is None else state.get("current_qpos")
        if qpos is None:
            if required:
                raise RuntimeError(
                    f"{side}_arm atomic runtime Home snapshot has no current_qpos."
                )
            continue
        qpos_array = np.asarray(qpos, dtype=float).reshape(6)
        if not np.all(np.isfinite(qpos_array)):
            raise RuntimeError(
                f"{side}_arm atomic runtime Home snapshot contains non-finite qpos."
            )
        preview_backend.set_local_qpos(qpos_array)
        preview_state = preview_backend.get_state()
        anchor_qpos = preview_state.get("current_qpos")
        if anchor_qpos is None:
            raise RuntimeError(
                f"{side}_arm preview did not retain the atomic Home qpos anchor."
            )
        anchor_qpos = np.asarray(anchor_qpos, dtype=float).reshape(6).copy()
        normalization = preview_state.get("local_qpos_normalization")
        if isinstance(normalization, dict) and normalization.get("status") == "clipped":
            _log(
                "Calibration",
                f"{side}_arm atomic Home qpos had a tiny endpoint quantization "
                "overshoot; the shared preview/IK/command anchor was clipped at "
                f"joint(s) {normalization.get('clipped_joint_indices', [])}.",
            )
        snapshot[side] = anchor_qpos

    if components.dual_arm_retargeter is not None:
        left_qpos = components.left_preview_backend.get_state().get("current_qpos")
        right_qpos = components.right_preview_backend.get_state().get("current_qpos")
        if left_qpos is not None and right_qpos is not None:
            components.dual_arm_retargeter.set_current_qpos_pair(
                left_qpos=left_qpos,
                right_qpos=right_qpos,
            )
    return snapshot

def _channel_pose_in_desk(channel, calibration) -> np.ndarray:
    pose = as_pose_matrix(channel.value)
    frame_name = channel.frame_name or ""
    metadata = channel.metadata or {}
    tracker_role = metadata.get("tracker_role")

    if frame_name in {"vive_world", "world"}:
        return calibration.world_to_desk(
            pose,
            tracker_role=str(tracker_role) if tracker_role else None,
        )
    if frame_name.endswith("_operator") or frame_name == "operator":
        return calibration.operator_to_desk(pose)
    if frame_name in {calibration.desk_frame.name, "desk"}:
        return pose
    raise ValueError(f"Unsupported calibration input frame: {frame_name or '<none>'}")

def _runtime_human_home_pose(
    samples: list[np.ndarray],
    *,
    late_weighted: bool,
    fresh_post_snapshot_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Resolve the human Home anchor without mixing different time cuts.

    A post-snapshot pose is already the accepted latest tracker observation for
    the robot Home transaction and therefore must be used exactly. Historical
    samples remain useful for stillness validation, but averaging them into the
    anchor creates a deterministic first-target residual whenever the tracker
    drifted during capture.
    """

    if fresh_post_snapshot_pose is not None:
        pose = as_pose_matrix(fresh_post_snapshot_pose)
        if not np.all(np.isfinite(pose)):
            raise ValueError("Fresh runtime human Home pose must be finite.")
        return pose.copy()
    weights = (
        np.arange(1, len(samples) + 1, dtype=float)
        if late_weighted
        else None
    )
    return _mean_pose(samples, weights=weights)

def _runtime_home_calibration(
    components: RealTeleopComponents,
    *,
    arm_mode: str,
    timeout_sec: float,
    num_samples: int,
    progress_callback=None,
    require_stable: bool = False,
    capture_sides: tuple[str, ...] | None = None,
    stability_profile: str = "strict",
    atomic_robot_home_snapshot: bool = True,
    late_weighted_human_home: bool = True,
    fresh_data_timeout_sec: float | None = None,
) -> dict[str, np.ndarray]:
    calibration = components.session.calibration
    assert calibration is not None
    active_sides = components.active_sides
    if capture_sides is None:
        home_sides = tuple(active_sides)
    else:
        requested_sides = tuple(dict.fromkeys(str(side) for side in capture_sides))
        invalid_sides = tuple(side for side in requested_sides if side not in active_sides)
        if invalid_sides or not requested_sides:
            raise ValueError(
                "runtime Home capture_sides must be a non-empty subset of active sides; "
                f"requested={requested_sides!r} active={active_sides!r}"
            )
        home_sides = requested_sides
    tracker_expectations = ", ".join(
        f"{side}_wrist_pose<-{calibration.metadata.get(f'{side}_tracker_name') or f'{side}_wrist role'}"
        for side in home_sides
    )
    _log(
        "Calibration",
        f"expecting {tracker_expectations}",
    )
    mapping_summary = ", ".join(
        f"{side}_wrist_pose -> {side}_arm" for side in home_sides
    )
    _log(
        "Calibration",
        f"logical arm mapping: {mapping_summary}",
    )
    try:
        left_arm = calibration.get_arm_for_side("left")
        right_arm = calibration.get_arm_for_side("right")
        if left_arm.T_D_B is not None and right_arm.T_D_B is not None:
            _log(
                "Calibration",
                "visual slots: "
                f"left_arm desk_y={left_arm.T_D_B[1, 3]:+.4f}, "
                f"right_arm desk_y={right_arm.T_D_B[1, 3]:+.4f}",
            )
    except Exception:
        pass
    _log(
        "Calibration",
        "keep active wrist tracker(s) steady in the desired home pose "
        f"for about {max(num_samples / 60.0, 0.5):.1f}s. "
        "This captures per-session tracker home on top of the saved tracker/station calibration.",
    )
    _log(
        "Calibration",
        f"home_capture_mode={_robot_home_mode_label(components)}",
    )
    if _preview_should_follow_live_real_state(arm_mode=arm_mode, components=components):
        if atomic_robot_home_snapshot:
            _log(
                "Calibration",
                "start-pose source=real; after the initial stable VIVE window, "
                "one dual-arm snapshot will be reused by robot Home, preview/IK, "
                "and command reset; a subsequent fresh stable VIVE cut will be "
                "the exact human Home anchor",
            )
        else:
            synced = _sync_preview_from_real_state(components, required=False)
            _log(
                "Calibration",
                f"start-pose source=real synced={synced}; pressing 'p' will "
                "capture tracker home and current real-arm home",
            )
    elif _has_live_real_state(components):
        _log(
            "Calibration",
            "start-pose source=virtual preview; pressing 'p' will capture tracker home and read current real-arm home from the provided arm IPs",
        )
    elif components.initial_alignment is not None:
        _log(
            "Calibration",
            "start-pose source=initial_alignment; pressing 'p' will capture tracker home only, while robot home comes from the saved alignment",
        )
    else:
        _log(
            "Calibration",
            "start-pose source=virtual preview defaults/current local state; pressing 'p' will capture tracker home and current preview arm home",
        )
    follows_live_real_state = _preview_should_follow_live_real_state(
        arm_mode=arm_mode,
        components=components,
    )
    # The default real-robot transaction uses the existing late/fresh feature
    # switch as its rollback boundary.  When enabled, historical samples only
    # prove stillness; a fresh tracker cut captured after the atomic arm qpos is
    # the exact human Home anchor.
    fresh_post_snapshot_active = bool(
        late_weighted_human_home
        and atomic_robot_home_snapshot
        and follows_live_real_state
    )
    stability_required = bool(require_stable or fresh_post_snapshot_active)
    calibration_timing_start = time.perf_counter()
    timeout_sec = float(timeout_sec)
    timeout_started_at = time.monotonic()
    timeout_deadline = timeout_started_at + timeout_sec
    freshness_timeout = (
        None
        if fresh_data_timeout_sec is None
        else float(fresh_data_timeout_sec)
    )
    if freshness_timeout is not None and freshness_timeout <= 0.0:
        raise ValueError("fresh_data_timeout_sec must be positive when provided")
    start_time = time.time()
    samples: dict[str, list[np.ndarray]] = {side: [] for side in home_sides}
    last_sample_timestamp: dict[str, float | None] = {
        side: None for side in home_sides
    }
    last_observed_timestamp: dict[str, float | None] = {
        side: None for side in home_sides
    }
    last_fresh_packet_at = {
        side: timeout_started_at for side in home_sides
    }
    fresh_packet_counts = {side: 0 for side in home_sides}
    late_weighted_active = bool(late_weighted_human_home and num_samples > 1)
    sample_target = int(num_samples) + (1 if late_weighted_active else 0)
    profile_name = str(stability_profile or "strict").strip().lower()
    if profile_name not in {"strict", "transition"}:
        raise ValueError(
            "runtime Home stability_profile must be 'strict' or 'transition', "
            f"got {stability_profile!r}"
        )
    if fresh_post_snapshot_active:
        human_home_estimator = "fresh_stable_post_arm_snapshot"
    elif late_weighted_active:
        human_home_estimator = "late_linear"
    else:
        human_home_estimator = "legacy_uniform"
    _log(
        "Calibration",
        f"stability_profile={profile_name} capture_sides={','.join(home_sides)} "
        f"human_home_estimator={human_home_estimator} "
        f"robot_home_snapshot="
        f"{'atomic_shared' if atomic_robot_home_snapshot else 'legacy_reread'}",
    )
    stability_factory = (
        HomeCaptureStabilityState.transition_profile
        if profile_name == "transition"
        else HomeCaptureStabilityState
    )
    stability_by_side = {side: stability_factory() for side in home_sides}
    observed_channels: set[str] = set()
    sample_markers = sorted(
        {1, max(1, num_samples // 2), num_samples, sample_target}
    )
    reported: dict[str, set[int]] = {side: set() for side in home_sides}

    def pending_sample_sides() -> tuple[str, ...]:
        return tuple(
            side for side in home_sides if len(samples[side]) < sample_target
        )

    def fresh_stale_sides(
        sides: tuple[str, ...],
        *,
        checked_at: float,
        last_fresh_at: dict[str, float],
    ) -> tuple[str, ...]:
        if freshness_timeout is None:
            return ()
        return tuple(
            side
            for side in sides
            if checked_at - last_fresh_at[side] >= freshness_timeout
        )

    def observe_fresh_packets(
        channels,
        sides: tuple[str, ...],
        *,
        observed_at: float,
        last_timestamp: dict[str, float | None],
        last_fresh_at: dict[str, float],
        packet_counts: dict[str, int],
    ) -> None:
        for side in sides:
            channel = channels.get(f"{side}_wrist_pose")
            if channel is None or not channel.valid:
                continue
            channel_timestamp = float(channel.timestamp)
            if not np.isfinite(channel_timestamp):
                raise ValueError(
                    f"{side}_wrist_pose timestamp must be finite during runtime Home"
                )
            previous_timestamp = last_timestamp[side]
            if (
                previous_timestamp is None
                or channel_timestamp > previous_timestamp
            ):
                last_timestamp[side] = channel_timestamp
                last_fresh_at[side] = observed_at
                packet_counts[side] += 1

    def raise_freshness_timeout(
        *,
        phase: str,
        stale_sides: tuple[str, ...],
        packet_counts: dict[str, int],
        stability_reasons: dict[str, str],
    ) -> None:
        raise TimeoutError(
            "Timed out because required VIVE tracker timestamp(s) did not "
            f"advance for {freshness_timeout:.3f}s during {phase}. "
            f"stale_sides={list(stale_sides)} "
            f"fresh_packet_counts={dict(packet_counts)} "
            f"accepted_sample_counts="
            f"{{{', '.join(f'{side!r}: {len(samples[side])}' for side in home_sides)}}} "
            f"stability_reasons={stability_reasons}"
        )

    def publish_progress(*, now: float, channels=None, waiting: bool = False) -> None:
        if progress_callback is None:
            return
        valid = {}
        if channels is not None:
            valid = {
                side: bool(
                    channels.get(f"{side}_wrist_pose") is not None
                    and channels[f"{side}_wrist_pose"].valid
                )
                for side in home_sides
            }
        progress_callback(
            sample_counts={side: len(samples[side]) for side in home_sides},
            valid_by_side=valid,
            min_samples=int(sample_target),
            elapsed_sec=max(0.0, time.monotonic() - timeout_started_at),
            timeout_sec=float(timeout_sec),
            waiting_for_samples=bool(waiting),
            now=now,
        )

    publish_progress(now=start_time, waiting=False)

    while any(len(samples[side]) < sample_target for side in home_sides):
        now = time.time()
        monotonic_now = time.monotonic()
        if monotonic_now >= timeout_deadline:
            counts = {side: len(samples[side]) for side in home_sides}
            publish_progress(now=now, waiting=True)
            raise TimeoutError(
                "Runtime Home overall timeout while validating stable VIVE data. "
                f"accepted_sample_counts={counts} "
                f"fresh_packet_counts={fresh_packet_counts} "
                "stability_reasons="
                f"{{{', '.join(f'{side!r}: {stability_by_side[side].reason!r}' for side in home_sides)}}}"
            )
        pending_sides = pending_sample_sides()
        stale_sides = fresh_stale_sides(
            pending_sides,
            checked_at=monotonic_now,
            last_fresh_at=last_fresh_packet_at,
        )
        if stale_sides:
            raise_freshness_timeout(
                phase="initial_stable_window",
                stale_sides=stale_sides,
                packet_counts=fresh_packet_counts,
                stability_reasons={
                    side: stability_by_side[side].reason for side in pending_sides
                },
            )
        channels, _ = components.session.poll_sources(now)
        poll_finished_at = time.monotonic()
        if poll_finished_at >= timeout_deadline:
            counts = {side: len(samples[side]) for side in home_sides}
            publish_progress(now=time.time(), channels=channels, waiting=True)
            raise TimeoutError(
                "Runtime Home overall timeout while polling stable VIVE data. "
                f"accepted_sample_counts={counts} "
                f"fresh_packet_counts={fresh_packet_counts}"
            )
        stale_sides = fresh_stale_sides(
            pending_sides,
            checked_at=poll_finished_at,
            last_fresh_at=last_fresh_packet_at,
        )
        if stale_sides:
            raise_freshness_timeout(
                phase="initial_stable_window",
                stale_sides=stale_sides,
                packet_counts=fresh_packet_counts,
                stability_reasons={
                    side: stability_by_side[side].reason for side in pending_sides
                },
            )
        observe_fresh_packets(
            channels,
            pending_sides,
            observed_at=poll_finished_at,
            last_timestamp=last_observed_timestamp,
            last_fresh_at=last_fresh_packet_at,
            packet_counts=fresh_packet_counts,
        )
        _update_optional_hand_preview_from_channels(
            components,
            channels,
            timestamp=now,
        )
        if components.renderer is not None:
            left_qpos = _state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS)
            right_qpos = _state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS)
            components.renderer.set_debug_frames(
                _build_renderer_debug_frames(
                    components,
                    channels=channels,
                )
            )
            components.renderer.set_debug_boxes(
                _build_renderer_debug_boxes(
                    components,
                    channels=channels,
                )
            )
            components.renderer.render_data(
                left_qpos_arm=left_qpos,
                left_qpos_hand=_hand_render_qpos(components.left_hand_backend, "left"),
                right_qpos_arm=right_qpos,
                right_qpos_hand=_hand_render_qpos(components.right_hand_backend, "right"),
                capture_image=False,
            )
        for side in home_sides:
            channel = channels.get(f"{side}_wrist_pose")
            if channel is None:
                continue
            if channel.channel_name not in observed_channels:
                metadata = channel.metadata or {}
                _log(
                    "Calibration",
                    "observed "
                    f"{channel.channel_name} <- "
                    f"{metadata.get('tracker_name')}:{metadata.get('tracker_role')} "
                    f"frame={channel.frame_name}",
                    now=now,
                )
                observed_channels.add(channel.channel_name)
            if channel.valid and len(samples[side]) < sample_target:
                channel_timestamp = float(channel.timestamp)
                previous_timestamp = last_sample_timestamp[side]
                if (
                    late_weighted_active
                    and previous_timestamp is not None
                    and channel_timestamp <= previous_timestamp
                ):
                    # The legacy path counted repeated polls of one latest-value
                    # packet as independent samples. The new estimator admits
                    # only fresh VIVE packets and reserves the final packet as
                    # the highest-weight Home observation.
                    continue
                pose_desk = _channel_pose_in_desk(channel, calibration)
                if stability_required:
                    decision = stability_by_side[side].observe(
                        pose_desk,
                        source_timestamp=channel_timestamp,
                        first_accepted_pose=(
                            samples[side][0] if samples[side] else None
                        ),
                    )
                    if decision.reset_samples:
                        samples[side].clear()
                        last_sample_timestamp[side] = None
                    if not decision.accepted:
                        continue
                samples[side].append(pose_desk)
                last_sample_timestamp[side] = channel_timestamp
                for marker in sample_markers:
                    if len(samples[side]) >= marker and marker not in reported[side]:
                        reported[side].add(marker)
                        _log(
                            "Calibration",
                            f"{side} progress {len(samples[side])}/{sample_target}",
                            now=now,
                        )
        if any(len(samples[side]) < sample_target for side in home_sides):
            publish_progress(
                now=now,
                channels=channels,
                waiting=True,
            )
            time.sleep(0.01)

    sample_done_perf = time.perf_counter()
    sample_elapsed_ms = (sample_done_perf - calibration_timing_start) * 1000.0
    publish_progress(now=time.time(), waiting=False)

    sync_elapsed_ms = 0.0
    robot_qpos_snapshot: dict[str, np.ndarray] = {}
    if follows_live_real_state and atomic_robot_home_snapshot:
        sync_start = time.perf_counter()
        robot_qpos_snapshot = _capture_atomic_real_qpos_snapshot(
            components,
            required=True,
        )
        sync_elapsed_ms = (time.perf_counter() - sync_start) * 1000.0
        _log(
            "Calibration",
            "locked one atomic real-arm Home snapshot for robot Home, preview, "
            "IK, and command reset "
            f"sides={sorted(robot_qpos_snapshot)} elapsed={sync_elapsed_ms:.1f}ms",
        )
    elif follows_live_real_state:
        sync_start = time.perf_counter()
        synced = _sync_preview_from_real_state(components, required=False)
        sync_elapsed_ms = (time.perf_counter() - sync_start) * 1000.0
        _log(
            "Calibration",
            f"locked preview start pose from real state synced={synced} "
            f"elapsed={sync_elapsed_ms:.1f}ms",
        )
    fresh_human_home_by_side: dict[str, np.ndarray] = {}
    fresh_human_home_elapsed_ms = 0.0
    if fresh_post_snapshot_active:
        # Start a new stillness window after the arm snapshot. This makes the
        # accepted tracker pose causally newer than the qpos anchor and avoids
        # treating an old latest-value packet as a fresh Home observation.
        fresh_start = time.perf_counter()
        fresh_stability_by_side = {
            side: stability_factory() for side in home_sides
        }
        pending_sides = set(home_sides)
        fresh_phase_started_at = time.monotonic()
        post_snapshot_last_timestamp = {
            side: last_observed_timestamp[side] for side in home_sides
        }
        post_snapshot_last_fresh_at = {
            side: fresh_phase_started_at for side in home_sides
        }
        post_snapshot_fresh_packet_counts = {
            side: 0 for side in home_sides
        }
        _log(
            "Calibration",
            "arm Home is locked; validating one fresh stable VIVE cut for the "
            "exact human Home anchor",
        )
        while pending_sides:
            now = time.time()
            monotonic_now = time.monotonic()
            if monotonic_now >= timeout_deadline:
                reasons = {
                    side: fresh_stability_by_side[side].reason
                    for side in sorted(pending_sides)
                }
                raise TimeoutError(
                    "Timed out waiting for a fresh stable post-arm-snapshot VIVE "
                    f"Home cut. pending_sides={sorted(pending_sides)} "
                    f"reasons={reasons}"
                )
            stale_sides = fresh_stale_sides(
                tuple(sorted(pending_sides)),
                checked_at=monotonic_now,
                last_fresh_at=post_snapshot_last_fresh_at,
            )
            if stale_sides:
                raise_freshness_timeout(
                    phase="post_arm_snapshot_exact_home",
                    stale_sides=stale_sides,
                    packet_counts=post_snapshot_fresh_packet_counts,
                    stability_reasons={
                        side: fresh_stability_by_side[side].reason
                        for side in sorted(pending_sides)
                    },
                )
            channels, _ = components.session.poll_sources(now)
            poll_finished_at = time.monotonic()
            if poll_finished_at >= timeout_deadline:
                reasons = {
                    side: fresh_stability_by_side[side].reason
                    for side in sorted(pending_sides)
                }
                raise TimeoutError(
                    "Timed out waiting for a fresh stable post-arm-snapshot VIVE "
                    f"Home cut. pending_sides={sorted(pending_sides)} "
                    f"reasons={reasons}"
                )
            stale_sides = fresh_stale_sides(
                tuple(sorted(pending_sides)),
                checked_at=poll_finished_at,
                last_fresh_at=post_snapshot_last_fresh_at,
            )
            if stale_sides:
                raise_freshness_timeout(
                    phase="post_arm_snapshot_exact_home",
                    stale_sides=stale_sides,
                    packet_counts=post_snapshot_fresh_packet_counts,
                    stability_reasons={
                        side: fresh_stability_by_side[side].reason
                        for side in sorted(pending_sides)
                    },
                )
            observe_fresh_packets(
                channels,
                tuple(sorted(pending_sides)),
                observed_at=poll_finished_at,
                last_timestamp=post_snapshot_last_timestamp,
                last_fresh_at=post_snapshot_last_fresh_at,
                packet_counts=post_snapshot_fresh_packet_counts,
            )
            _update_optional_hand_preview_from_channels(
                components,
                channels,
                timestamp=now,
            )
            for side in tuple(sorted(pending_sides)):
                channel = channels.get(f"{side}_wrist_pose")
                if channel is None or not channel.valid:
                    continue
                channel_timestamp = float(channel.timestamp)
                cutoff_timestamp = last_sample_timestamp[side]
                if (
                    cutoff_timestamp is not None
                    and channel_timestamp <= cutoff_timestamp
                ):
                    continue
                pose_desk = _channel_pose_in_desk(channel, calibration)
                decision = fresh_stability_by_side[side].observe(
                    pose_desk,
                    source_timestamp=channel_timestamp,
                    first_accepted_pose=None,
                )
                if not decision.accepted:
                    continue
                fresh_human_home_by_side[side] = pose_desk.copy()
                last_sample_timestamp[side] = channel_timestamp
                pending_sides.remove(side)
                _log(
                    "Calibration",
                    f"{side} exact human Home locked from fresh post-snapshot "
                    f"tracker timestamp={channel_timestamp:.6f}",
                    now=now,
                )
            if pending_sides:
                publish_progress(now=now, channels=channels, waiting=True)
                time.sleep(0.01)
        fresh_human_home_elapsed_ms = (
            time.perf_counter() - fresh_start
        ) * 1000.0
        publish_progress(now=time.time(), waiting=False)
    home_summaries: list[str] = []
    qpos_summaries: list[str] = []
    side_timing_summaries: list[str] = []
    post_sample_start = time.perf_counter()
    for side in home_sides:
        side_start = time.perf_counter()
        mean_start = time.perf_counter()
        human_home = _runtime_human_home_pose(
            samples[side],
            late_weighted=late_weighted_active,
            fresh_post_snapshot_pose=fresh_human_home_by_side.get(side),
        )
        mean_elapsed_ms = (time.perf_counter() - mean_start) * 1000.0
        robot_home_start = time.perf_counter()
        robot_home_desk, qpos, robot_source = _resolve_robot_home_for_side(
            components,
            side=side,
            qpos_snapshot=(
                robot_qpos_snapshot if atomic_robot_home_snapshot else None
            ),
        )
        robot_home_elapsed_ms = (time.perf_counter() - robot_home_start) * 1000.0
        anchor_start = time.perf_counter()
        _side_retargeter(components, side).set_home_anchors(
            human_pose_desk=human_home,
            robot_pose_desk=robot_home_desk,
        )
        anchor_elapsed_ms = (time.perf_counter() - anchor_start) * 1000.0
        capture_start = time.perf_counter()
        calibration.capture_home_pose(f"{side}_arm", robot_home_desk)
        if qpos is not None:
            calibration.capture_home_qpos(f"{side}_arm", qpos)
        capture_elapsed_ms = (time.perf_counter() - capture_start) * 1000.0
        side_elapsed_ms = (time.perf_counter() - side_start) * 1000.0
        home_summaries.append(
            f"{side}_source={robot_source} {side}_xyz={_pose_translation_summary(robot_home_desk)}"
        )
        qpos_summaries.append(f"{side}_q={_qpos_summary(qpos)}")
        side_timing_summaries.append(
            f"{side}:total={side_elapsed_ms:.1f}ms "
            f"mean={mean_elapsed_ms:.1f}ms "
            f"robot_home={robot_home_elapsed_ms:.1f}ms "
            f"anchors={anchor_elapsed_ms:.1f}ms "
            f"capture={capture_elapsed_ms:.1f}ms "
            f"source={robot_source}"
        )
    post_sample_elapsed_ms = (time.perf_counter() - post_sample_start) * 1000.0
    _log(
        "CalibrationTiming",
        f"runtime_home sampling={sample_elapsed_ms:.1f}ms "
        f"post_sample={post_sample_elapsed_ms:.1f}ms "
        f"real_state_sync={sync_elapsed_ms:.1f}ms "
        f"fresh_human_home={fresh_human_home_elapsed_ms:.1f}ms "
        + " | ".join(side_timing_summaries),
    )
    _log("Calibration", "robot_home " + " ".join(home_summaries))
    sample_counts = {side: len(values) for side, values in samples.items()}
    _log(
        "Calibration",
        f"runtime home captured sample_counts={sample_counts} "
        + " ".join(qpos_summaries),
    )
    return {
        side: np.asarray(qpos, dtype=float).reshape(6).copy()
        for side, qpos in robot_qpos_snapshot.items()
    }

def _hand_observation_from_channels(
    components: RealTeleopComponents,
    channels,
    *,
    timestamp: float | None = None,
) -> TeleopObservation | None:
    if not channels:
        return None
    observation_timestamp = latest_channel_timestamp(channels)
    if observation_timestamp is None:
        observation_timestamp = time.time() if timestamp is None else float(timestamp)
    return TeleopObservation(
        channels={name: channel.copy() for name, channel in channels.items()},
        timestamp=observation_timestamp,
        metadata={"tcp_mode": components.session.config.tcp_mode},
    )

def _build_optional_hand_command_from_observation(
    components: RealTeleopComponents,
    observation: TeleopObservation | None,
):
    if (
        components.left_hand_retargeter is None
        and components.right_hand_retargeter is None
    ):
        return None
    if observation is None:
        return None
    commands = []
    if components.left_hand_retargeter is not None:
        commands.append(
            components.left_hand_retargeter.retarget(
                observation,
                components.session.calibration,
            )
        )
    if components.right_hand_retargeter is not None:
        commands.append(
            components.right_hand_retargeter.retarget(
                observation,
                components.session.calibration,
            )
        )
    if not commands:
        return None
    command = merge_teleop_commands(commands)
    if not command.commands:
        return None
    return command

def _send_optional_hand_command(
    components: RealTeleopComponents,
    hand_command,
) -> None:
    if hand_command is None:
        return
    left_command = hand_command.commands.get("left_hand")
    if left_command is not None and components.left_hand_backend is not None:
        components.left_hand_backend.send_command(left_command)
    right_command = hand_command.commands.get("right_hand")
    if right_command is not None and components.right_hand_backend is not None:
        components.right_hand_backend.send_command(right_command)

def _update_optional_hand_preview_from_channels(
    components: RealTeleopComponents,
    channels,
    *,
    timestamp: float | None = None,
):
    hand_command = _build_optional_hand_command_from_observation(
        components,
        _hand_observation_from_channels(
            components,
            channels,
            timestamp=timestamp,
        ),
    )
    _send_optional_hand_command(components, hand_command)
    return hand_command

def _hand_state_q6(backend: InspireTacHandBackend | None) -> np.ndarray | None:
    if backend is None:
        return None
    joint_pos = backend.get_state().get("joint_pos")
    if joint_pos is None:
        return None
    q6 = np.asarray(joint_pos, dtype=float).reshape(-1)
    if q6.shape != (6,) or not np.all(np.isfinite(q6)):
        return None
    return np.clip(q6, 0.0, 1.0)

def _hand_render_qpos(backend: InspireTacHandBackend | None, side: str) -> np.ndarray:
    q6 = _hand_state_q6(backend)
    if q6 is None:
        return open_hand_render_qpos(side)
    return inspire_q6_to_render_qpos(q6)


def _hand_observation_for_step(
    components: RealTeleopComponents,
    step_result: object,
) -> TeleopObservation | None:
    if step_result.observation is not None:
        return step_result.observation
    return _hand_observation_from_channels(
        components,
        step_result.channels,
        timestamp=step_result.timestamp,
    )


def _build_optional_hand_command(
    components: RealTeleopComponents,
    step_result: object,
):
    return _build_optional_hand_command_from_observation(
        components,
        _hand_observation_for_step(components, step_result),
    )

__all__ = [
    "DEFAULT_RUNTIME_HOME_START_DELAY_SEC",
    "_build_renderer_debug_boxes",
    "_build_renderer_debug_frames",
    "_build_optional_hand_command",
    "_channel_pose_in_desk",
    "_emit_collision_proxy_logs",
    "_emit_home_alignment_logs",
    "_emit_runtime_orientation_logs",
    "_emit_tracker_axis_mapping_logs",
    "_hand_render_qpos",
    "_manual_calibration_confirmation_required",
    "_mean_pose",
    "_resolve_robot_home_for_side",
    "_runtime_home_calibration",
    "_runtime_human_home_pose",
    "_send_optional_hand_command",
    "_update_optional_hand_preview_from_channels",
    "_vive_tracker_local_translation_offset",
]
