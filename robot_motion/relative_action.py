"""Hardware-independent absolute/relative arm action conversion.

The stored dataset action remains an absolute pose.  A relative action is a
chunk-local representation whose anchor is the actual TCP at the beginning of
that chunk.  ``last_action`` has a different rule: when converted for row
``t`` it uses the current row's anchor, never the previous row/chunk anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from .transforms import invert_pose, parse_pose_matrix, validate_pose_matrix


ACTION_POSE_MODE_ABSOLUTE = "absolute"
ACTION_POSE_MODE_RELATIVE = "relative"
ACTION_POSE_MODES = (ACTION_POSE_MODE_ABSOLUTE, ACTION_POSE_MODE_RELATIVE)
DEFAULT_ARM_POSE_SLICES: dict[str, slice] = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}


class RelativeActionError(ValueError):
    """Raised when an action/anchor cannot be converted safely."""


def normalize_action_pose_mode(value: str | None) -> str:
    mode = str(value or ACTION_POSE_MODE_ABSOLUTE).strip().lower()
    aliases = {
        "abs": ACTION_POSE_MODE_ABSOLUTE,
        "absolute_pose": ACTION_POSE_MODE_ABSOLUTE,
        "rel": ACTION_POSE_MODE_RELATIVE,
        "relative_pose": ACTION_POSE_MODE_RELATIVE,
    }
    mode = aliases.get(mode, mode)
    if mode not in ACTION_POSE_MODES:
        raise RelativeActionError(
            f"Action pose mode must be one of {ACTION_POSE_MODES}, got {value!r}."
        )
    return mode


def _strict_matrix(value: Any, *, name: str) -> np.ndarray:
    try:
        matrix = parse_pose_matrix(value)
    except (TypeError, ValueError) as exc:
        raise RelativeActionError(f"{name} is not a supported pose encoding.") from exc
    valid, reason = validate_pose_matrix(matrix)
    if not valid:
        raise RelativeActionError(f"{name} is invalid: {reason}")
    return np.asarray(matrix, dtype=float).copy()


def xyzquat_to_pose_matrix(value: Any, *, name: str = "pose") -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RelativeActionError(f"{name} must be a finite xyzquat vector.") from exc
    if vector.shape != (7,) or not np.all(np.isfinite(vector)):
        raise RelativeActionError(f"{name} must have seven finite xyzquat values.")
    quaternion = vector[3:7]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise RelativeActionError(f"{name} quaternion norm is zero.")
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = vector[:3]
    matrix[:3, :3] = R.from_quat(quaternion / norm).as_matrix()
    return matrix


def _canonical_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12 or not np.all(np.isfinite(quaternion)):
        raise RelativeActionError("Pose quaternion is invalid.")
    quaternion = quaternion / norm
    # q and -q encode the same rotation.  Keep a deterministic representative
    # for stable parquet/model values while all equivalence checks use SE(3).
    if quaternion[3] < 0.0 or (
        abs(quaternion[3]) <= 1e-12
        and next((float(value) for value in quaternion[:3] if abs(value) > 1e-12), 0.0) < 0.0
    ):
        quaternion = -quaternion
    return quaternion


def pose_matrix_to_xyzquat(value: Any, *, name: str = "pose") -> np.ndarray:
    matrix = _strict_matrix(value, name=name)
    quaternion = _canonical_quaternion(R.from_matrix(matrix[:3, :3]).as_quat())
    return np.concatenate((matrix[:3, 3], quaternion)).astype(np.float32, copy=False)


def relative_pose(anchor_pose: Any, absolute_target_pose: Any) -> np.ndarray:
    """Express an absolute target in the anchor TCP frame."""

    anchor = _strict_matrix(anchor_pose, name="anchor_pose")
    target = _strict_matrix(absolute_target_pose, name="absolute_target_pose")
    return invert_pose(anchor) @ target


def absolute_pose(anchor_pose: Any, relative_target_pose: Any) -> np.ndarray:
    """Recover an absolute target from a chunk anchor and relative target."""

    anchor = _strict_matrix(anchor_pose, name="anchor_pose")
    relative = _strict_matrix(relative_target_pose, name="relative_target_pose")
    return anchor @ relative


def _pose_from_vector(value: Any, *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise RelativeActionError(f"{name} must be an xyzquat vector or 4x4 matrix.") from exc
    if vector.shape == (7,):
        return xyzquat_to_pose_matrix(vector, name=name)
    return _strict_matrix(value, name=name)


def _convert_pose_vector(
    value: Any,
    *,
    anchor: Any,
    source_mode: str,
    target_mode: str,
    name: str,
) -> np.ndarray:
    source = normalize_action_pose_mode(source_mode)
    target = normalize_action_pose_mode(target_mode)
    if source == target:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape != (7,) or not np.all(np.isfinite(vector)):
            raise RelativeActionError(f"{name} must be a finite xyzquat vector.")
        # Validate and canonicalize even on a no-op path so bad values cannot
        # cross a contract boundary.
        return pose_matrix_to_xyzquat(xyzquat_to_pose_matrix(vector, name=name), name=name)
    pose = _pose_from_vector(value, name=name)
    anchor_matrix = _strict_matrix(anchor, name="anchor_pose")
    converted = (
        relative_pose(anchor_matrix, pose)
        if source == ACTION_POSE_MODE_ABSOLUTE and target == ACTION_POSE_MODE_RELATIVE
        else absolute_pose(anchor_matrix, pose)
    )
    return pose_matrix_to_xyzquat(converted, name=name)


def _validate_slices(arm_pose_slices: Mapping[str, slice]) -> dict[str, slice]:
    output: dict[str, slice] = {}
    for side, pose_slice in dict(arm_pose_slices).items():
        normalized_side = str(side).strip().lower()
        if normalized_side not in {"left", "right"}:
            raise RelativeActionError(f"Unsupported arm side {side!r}.")
        if not isinstance(pose_slice, slice) or pose_slice.step not in (None, 1):
            raise RelativeActionError(f"Arm pose slice for {side!r} must be a contiguous slice.")
        if pose_slice.start is None or pose_slice.stop is None or pose_slice.stop - pose_slice.start != 7:
            raise RelativeActionError(f"Arm pose slice for {side!r} must have width 7.")
        output[normalized_side] = pose_slice
    return output


def convert_arm_pose_chunk(
    actions: Any,
    *,
    anchors_by_side: Mapping[str, Any],
    source_mode: str,
    target_mode: str,
    arm_pose_slices: Mapping[str, slice] | None = None,
    active_sides: Sequence[str] | None = None,
) -> np.ndarray:
    """Convert arm pose slots in a fixed-anchor action chunk.

    ``anchors_by_side`` contains one anchor per side, shared by every row in
    the chunk.  Non-arm slots are copied byte-for-byte apart from the returned
    independent array.
    """

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2:
        raise RelativeActionError(f"actions must have shape [horizon, dim], got {values.shape}.")
    if values.shape[0] <= 0:
        raise RelativeActionError("actions chunk must contain at least one row.")
    slices = _validate_slices(DEFAULT_ARM_POSE_SLICES if arm_pose_slices is None else arm_pose_slices)
    sides = tuple(slices) if active_sides is None else tuple(str(side).strip().lower() for side in active_sides)
    unknown = sorted(set(sides) - set(slices))
    if unknown:
        raise RelativeActionError(f"Unsupported active arm side(s): {unknown}.")
    output = values.copy()
    for side in sides:
        if side not in anchors_by_side:
            raise RelativeActionError(f"Missing fixed chunk anchor for {side!r}.")
        pose_slice = slices[side]
        if int(pose_slice.stop) > output.shape[1]:
            raise RelativeActionError(f"Action vector is missing the {side!r} arm pose slot.")
        anchor = _pose_from_vector(anchors_by_side[side], name=f"{side}.chunk_anchor")
        for index in range(output.shape[0]):
            output[index, pose_slice] = _convert_pose_vector(
                output[index, pose_slice],
                anchor=anchor,
                source_mode=source_mode,
                target_mode=target_mode,
                name=f"{side}.action[{index}]",
            )
    return output


def _row_anchor(value: Any, *, index: int, horizon: int, side: str) -> Any:
    array = np.asarray(value)
    if array.shape in {(4, 4), (7,)}:
        return value
    if array.ndim == 3 and array.shape == (horizon, 4, 4):
        return array[index]
    if array.ndim == 2 and array.shape == (horizon, 7):
        return array[index]
    raise RelativeActionError(
        f"{side}.current_anchors must be one pose or [{horizon}, pose] sequence; got {array.shape}."
    )


def convert_last_action_chunk(
    last_actions: Any,
    *,
    current_anchors_by_side: Mapping[str, Any],
    source_mode: str,
    target_mode: str,
    arm_pose_slices: Mapping[str, slice] | None = None,
    active_sides: Sequence[str] | None = None,
) -> np.ndarray:
    """Convert ``observation.last_action`` using each row's current anchor."""

    values = np.asarray(last_actions, dtype=np.float32)
    if values.ndim != 2:
        raise RelativeActionError(
            f"last_actions must have shape [horizon, dim], got {values.shape}."
        )
    if values.shape[0] <= 0:
        raise RelativeActionError("last_actions chunk must contain at least one row.")
    slices = _validate_slices(DEFAULT_ARM_POSE_SLICES if arm_pose_slices is None else arm_pose_slices)
    sides = tuple(slices) if active_sides is None else tuple(str(side).strip().lower() for side in active_sides)
    output = values.copy()
    for side in sides:
        if side not in current_anchors_by_side:
            raise RelativeActionError(f"Missing current anchor sequence for {side!r} last_action.")
        pose_slice = slices[side]
        if int(pose_slice.stop) > output.shape[1]:
            raise RelativeActionError(f"last_actions is missing the {side!r} arm pose slot.")
        for index in range(output.shape[0]):
            anchor = _row_anchor(
                current_anchors_by_side[side],
                index=index,
                horizon=values.shape[0],
                side=side,
            )
            output[index, pose_slice] = _convert_pose_vector(
                output[index, pose_slice],
                anchor=anchor,
                source_mode=source_mode,
                target_mode=target_mode,
                name=f"{side}.last_action[{index}]",
            )
    return output


def convert_action_and_last_action_chunk(
    actions: Any,
    last_actions: Any | None,
    *,
    action_anchors_by_side: Mapping[str, Any],
    last_action_current_anchors_by_side: Mapping[str, Any] | None,
    source_mode: str,
    target_mode: str,
    arm_pose_slices: Mapping[str, slice] | None = None,
    active_sides: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert action and last_action with their distinct anchor rules."""

    converted_actions = convert_arm_pose_chunk(
        actions,
        anchors_by_side=action_anchors_by_side,
        source_mode=source_mode,
        target_mode=target_mode,
        arm_pose_slices=arm_pose_slices,
        active_sides=active_sides,
    )
    if last_actions is None:
        return converted_actions, None
    if last_action_current_anchors_by_side is None:
        raise RelativeActionError("last_action current anchors are required when last_actions is provided.")
    converted_last_actions = convert_last_action_chunk(
        last_actions,
        current_anchors_by_side=last_action_current_anchors_by_side,
        source_mode=source_mode,
        target_mode=target_mode,
        arm_pose_slices=arm_pose_slices,
        active_sides=active_sides,
    )
    return converted_actions, converted_last_actions


@dataclass(frozen=True)
class PoseChunkErrorMetrics:
    max_translation_error_m: float
    max_rotation_error_rad: float
    frame_count: int


def compare_pose_chunks(
    expected_absolute_actions: Any,
    reconstructed_absolute_actions: Any,
    *,
    arm_pose_slices: Mapping[str, slice] | None = None,
    active_sides: Sequence[str] | None = None,
) -> dict[str, PoseChunkErrorMetrics]:
    """Compare absolute pose chunks using translation and rotation metrics."""

    expected = np.asarray(expected_absolute_actions, dtype=np.float32)
    reconstructed = np.asarray(reconstructed_absolute_actions, dtype=np.float32)
    if expected.shape != reconstructed.shape or expected.ndim != 2:
        raise RelativeActionError(
            f"Pose chunks must have equal shape [horizon, dim], got {expected.shape} and {reconstructed.shape}."
        )
    slices = _validate_slices(DEFAULT_ARM_POSE_SLICES if arm_pose_slices is None else arm_pose_slices)
    sides = tuple(slices) if active_sides is None else tuple(str(side).strip().lower() for side in active_sides)
    metrics: dict[str, PoseChunkErrorMetrics] = {}
    for side in sides:
        pose_slice = slices[side]
        translation_errors: list[float] = []
        rotation_errors: list[float] = []
        for index in range(expected.shape[0]):
            first = xyzquat_to_pose_matrix(expected[index, pose_slice], name=f"{side}.expected[{index}]")
            second = xyzquat_to_pose_matrix(
                reconstructed[index, pose_slice],
                name=f"{side}.reconstructed[{index}]",
            )
            translation_errors.append(float(np.linalg.norm(first[:3, 3] - second[:3, 3])))
            relative_rotation = first[:3, :3].T @ second[:3, :3]
            rotation_errors.append(float(R.from_matrix(relative_rotation).magnitude()))
        metrics[side] = PoseChunkErrorMetrics(
            max_translation_error_m=max(translation_errors, default=0.0),
            max_rotation_error_rad=max(rotation_errors, default=0.0),
            frame_count=int(expected.shape[0]),
        )
    return metrics


__all__ = [
    "ACTION_POSE_MODES",
    "ACTION_POSE_MODE_ABSOLUTE",
    "ACTION_POSE_MODE_RELATIVE",
    "DEFAULT_ARM_POSE_SLICES",
    "PoseChunkErrorMetrics",
    "RelativeActionError",
    "absolute_pose",
    "compare_pose_chunks",
    "convert_action_and_last_action_chunk",
    "convert_arm_pose_chunk",
    "convert_last_action_chunk",
    "normalize_action_pose_mode",
    "pose_matrix_to_xyzquat",
    "relative_pose",
    "xyzquat_to_pose_matrix",
]
