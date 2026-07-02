from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np

from ..arm_frames import ARM_ENDPOINT_FRAME, ARM_ENDPOINT_SEMANTICS
from .codec import ActionVectorEncoding
from .schema import RobotAction
from .synthetic import SyntheticReplayFrame


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _mapping(value: Any) -> Mapping[str, Any]:
    parsed = _maybe_json(value)
    return parsed if isinstance(parsed, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    parsed = _maybe_json(value)
    if parsed is None:
        return []
    if isinstance(parsed, np.ndarray):
        return parsed.reshape(-1).tolist()
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return []


def _finite_vector(value: Any, *, length: int) -> np.ndarray | None:
    seq = _sequence(value)
    if not seq:
        return None
    try:
        array = np.asarray(seq, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if array.shape != (int(length),):
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array.astype(np.float32, copy=True)


def _vector_with_optional_nan(value: Any, *, length: int) -> np.ndarray | None:
    seq = _sequence(value)
    if not seq:
        return None
    try:
        array = np.asarray(seq, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if array.shape != (int(length),):
        return None
    return array.astype(np.float32, copy=True)


def _canonical_action_vector_from_metadata(row: Mapping[str, Any]) -> ActionVectorEncoding | None:
    metadata = _mapping(row.get("sample_metadata_json"))
    canonical = _mapping(metadata.get("canonical_action_record"))
    action_vector = _mapping(canonical.get("action_vector"))
    if not action_vector:
        return None
    try:
        return ActionVectorEncoding(
            action_space=str(action_vector["action_space"]),
            vector=np.asarray(action_vector["vector"], dtype=np.float32),
            component_mask=np.asarray(action_vector["component_mask"], dtype=bool),
            effector_mask=_mapping(action_vector.get("effector_mask")),
            layout_version=str(action_vector.get("layout_version", "robot_action_vector.v1")),
        )
    except Exception:
        return None


def _active_sides(row: Mapping[str, Any]) -> set[str]:
    raw = _sequence(row.get("active_sides"))
    sides = {str(side) for side in raw if str(side) in {"left", "right"}}
    return sides


def _legacy_arm_endpoint_specs(*, left_arm: Any | None, right_arm: Any | None) -> dict[str, Any]:
    specs: dict[str, Any] = {}
    for side, value in (("left", left_arm), ("right", right_arm)):
        if value is None:
            continue
        specs[side] = {
            "side": side,
            "controlled_frame": ARM_ENDPOINT_FRAME,
            "reference_frame": f"{side}_arm_base",
            "representation": "xyz_quat",
            "dof": 7,
            "semantics": ARM_ENDPOINT_SEMANTICS,
            "metadata": {
                "compatibility_source": "legacy_collection_row_without_action_spec",
                "compatibility_note": (
                    "Rows without canonical action metadata are interpreted as the historical "
                    "arm_endpoint pose contract for replay."
                ),
            },
        }
    return specs


def _fallback_ee_pose_action_from_row(row: Mapping[str, Any]) -> RobotAction | None:
    action = _vector_with_optional_nan(row.get("action"), length=26)
    if action is None:
        left_arm = _finite_vector(row.get("action.left_arm.pose"), length=7)
        right_arm = _finite_vector(row.get("action.right_arm.pose"), length=7)
        left_hand = _finite_vector(row.get("action.left_hand.joint_target"), length=6)
        right_hand = _finite_vector(row.get("action.right_hand.joint_target"), length=6)
    else:
        left_arm = action[0:7] if np.all(np.isfinite(action[0:7])) else None
        right_arm = action[7:14] if np.all(np.isfinite(action[7:14])) else None
        left_hand = action[14:20] if np.all(np.isfinite(action[14:20])) else None
        right_hand = action[20:26] if np.all(np.isfinite(action[20:26])) else None

    sides = _active_sides(row)
    if sides:
        if "left" not in sides:
            left_arm = None
            left_hand = None
        if "right" not in sides:
            right_arm = None
            right_hand = None

    if left_arm is None and right_arm is None and left_hand is None and right_hand is None:
        return None
    timestamp = row.get("timestamp")
    try:
        resolved_timestamp = None if timestamp is None else float(timestamp)
    except (TypeError, ValueError):
        resolved_timestamp = None
    return RobotAction.bimanual_ee_pose(
        left_arm=left_arm,
        right_arm=right_arm,
        left_hand=left_hand,
        right_hand=right_hand,
        timestamp=resolved_timestamp,
        source="collection_row",
        metadata={
            "row_index": row.get("index"),
            "episode_index": row.get("episode_index"),
            "frame_index": row.get("frame_index"),
            "source": "fallback_action_vector",
            "legacy_replay_compatibility": {
                "arm_pose_frame": ARM_ENDPOINT_FRAME,
                "reason": "missing canonical action metadata",
            },
            "arm_action_specs_by_side": _legacy_arm_endpoint_specs(
                left_arm=left_arm,
                right_arm=right_arm,
            ),
        },
    )


def synthetic_frame_from_collection_row(
    row: Mapping[str, Any],
    *,
    prefer_canonical: bool = True,
    include_observation: bool = True,
) -> SyntheticReplayFrame:
    action: RobotAction | ActionVectorEncoding | None = None
    if prefer_canonical:
        action = _canonical_action_vector_from_metadata(row)
    if action is None:
        action = _fallback_ee_pose_action_from_row(row)

    timestamp = row.get("timestamp")
    try:
        resolved_timestamp = None if timestamp is None else float(timestamp)
    except (TypeError, ValueError):
        resolved_timestamp = None
    observation = dict(row) if include_observation else None
    return SyntheticReplayFrame(
        action=action,
        observation=observation,
        timestamp=resolved_timestamp,
        metadata={
            "row_index": row.get("index"),
            "episode_index": row.get("episode_index"),
            "frame_index": row.get("frame_index"),
            "source": "collection_row",
            "prefer_canonical": bool(prefer_canonical),
        },
    )


def synthetic_frames_from_collection_rows(
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    prefer_canonical: bool = True,
    include_observation: bool = True,
) -> tuple[SyntheticReplayFrame, ...]:
    return tuple(
        synthetic_frame_from_collection_row(
            row,
            prefer_canonical=prefer_canonical,
            include_observation=include_observation,
        )
        for row in rows
    )
