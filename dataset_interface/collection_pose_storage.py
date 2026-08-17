"""Detached collection-row pose storage adaptation and validation.

This dataset-interface module operates on a newly built parquet-row mapping.
It deliberately does not accept or mutate ``TeleopCommand``, hardware state,
or a ``CollectorSample`` so dataset conversion cannot affect robot control.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from dataset_contracts.causal_snapshot import DEFAULT_CAUSAL_SNAPSHOT_CONTRACT
from dataset_contracts.pose_frames import (
    POSE_STORAGE_FRAME_DESK,
    PoseFrameContract,
    PoseFrameContractError,
)
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23


_PACKED_ARM_POSE_FIELDS = (
    "action",
    "action.arm.pose",
    "observation.state",
    "observation.arm.pose",
    "observation.last_action",
)

_SIDE_POSE_FIELD_TEMPLATES = (
    "action.{side}_arm.pose",
    "observation.{side}_arm.pose",
    "observation.arm.{side}.current_tcp_pose",
    "observation.arm.{side}.commanded_tcp_pose",
    "observation.arm.{side}.actual_tcp_pose",
)

_RAW_AUDIT_POSE_FIELD_TEMPLATES = (
    "action.reference.{side}_arm.pose_ref",
)


def _finite_scalar(value: Any) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def _is_true(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _is_false(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and not bool(value)


def _same_pose7(left: np.ndarray, right: np.ndarray) -> bool:
    if not np.allclose(left[:3], right[:3], atol=1e-6, rtol=0.0):
        return False
    left_norm = float(np.linalg.norm(left[3:]))
    right_norm = float(np.linalg.norm(right[3:]))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return False
    left_quat = left[3:] / left_norm
    right_quat = right[3:] / right_norm
    return bool(abs(float(np.dot(left_quat, right_quat))) >= 1.0 - 1e-6)


def _packed_pose7_or_none(
    row: Mapping[str, Any],
    *,
    key: str,
    side: str,
) -> np.ndarray | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        packed = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if packed.size < 14:
        return None
    start = 0 if side == "left" else 7
    pose = packed[start : start + 7]
    return pose.copy() if np.all(np.isfinite(pose)) else None


def _require_matching_pose_copy(
    failures: list[str],
    *,
    row: Mapping[str, Any],
    key: str,
    side: str,
    expected: np.ndarray,
    packed: bool,
    failure_name: str,
) -> None:
    try:
        candidate = (
            _packed_pose7_or_none(row, key=key, side=side)
            if packed
            else _pose7_or_none(row.get(key), name=f"row.{key}")
        )
    except PoseFrameContractError:
        candidate = None
    if candidate is None or not _same_pose7(candidate, expected):
        failures.append(f"{side}:{failure_name}")


def validate_v23_collection_row_actual_tcp(
    row: Mapping[str, Any],
    *,
    contract: PoseFrameContract,
) -> None:
    """Fail closed unless every active arm has a fresh causal measured TCP.

    Validation happens on the detached raw writer row, before any raw-to-desk
    conversion.  It therefore cannot affect control, and it ensures a v2.3
    episode never labels commanded/cached poses as measured state anchors.
    """

    sample_timestamp = _finite_scalar(row.get("timestamp"))
    if sample_timestamp is None:
        raise PoseFrameContractError(
            "v2.3 collection row requires a finite timestamp T0."
        )

    failures: list[str] = []
    rm65_source = str(contract.source_id).strip().lower().startswith("rm65_")
    for side in contract.source.active_sides:
        prefix = f"observation.arm.{side}."
        pose_key = f"{prefix}actual_tcp_pose"
        try:
            pose = _pose7_or_none(row.get(pose_key), name=f"row.{pose_key}")
        except PoseFrameContractError as exc:
            failures.append(f"{side}:invalid_actual_tcp_pose:{exc}")
            continue
        quaternion_norm = None if pose is None else float(np.linalg.norm(pose[3:]))
        if (
            pose is None
            or quaternion_norm is None
            or not np.isclose(quaternion_norm, 1.0, atol=1e-5, rtol=0.0)
        ):
            failures.append(f"{side}:missing_or_nonfinite_actual_tcp_pose")
            continue

        _require_matching_pose_copy(
            failures,
            row=row,
            key=f"observation.{side}_arm.pose",
            side=side,
            expected=pose,
            packed=False,
            failure_name="observation_state_pose_is_not_actual_tcp",
        )
        _require_matching_pose_copy(
            failures,
            row=row,
            key="observation.state",
            side=side,
            expected=pose,
            packed=True,
            failure_name="packed_observation_state_pose_is_not_actual_tcp",
        )
        _require_matching_pose_copy(
            failures,
            row=row,
            key="observation.arm.pose",
            side=side,
            expected=pose,
            packed=True,
            failure_name="observation_arm_pose_copy_is_not_actual_tcp",
        )

        action_pose = _packed_pose7_or_none(row, key="action", side=side)
        if action_pose is None:
            failures.append(f"{side}:missing_packed_action_pose")
        else:
            _require_matching_pose_copy(
                failures,
                row=row,
                key="action.arm.pose",
                side=side,
                expected=action_pose,
                packed=True,
                failure_name="action_arm_pose_copy_disagrees",
            )
            _require_matching_pose_copy(
                failures,
                row=row,
                key=f"action.{side}_arm.pose",
                side=side,
                expected=action_pose,
                packed=False,
                failure_name="action_side_pose_copy_disagrees",
            )

        pose_source = str(row.get(f"{prefix}actual_tcp_pose_source") or "").strip()
        if rm65_source:
            hardware_qpos = row.get(f"{prefix}actual_hardware_qpos")
            try:
                qpos = np.asarray(hardware_qpos, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                qpos = np.asarray([], dtype=float)
            if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
                failures.append(f"{side}:missing_or_invalid_actual_hardware_qpos")
            if pose_source != "hardware_qpos_fk":
                failures.append(f"{side}:actual_tcp_pose_not_from_hardware_qpos_fk")
        elif not pose_source or pose_source in {
            "current_tcp_pose",
            "commanded_tcp_pose",
            "last_commanded_action",
        }:
            failures.append(f"{side}:actual_tcp_pose_source_not_measured")
        if not _is_true(row.get(f"{prefix}actual_state_read_success")):
            failures.append(f"{side}:hardware_read_not_successful")
        if not _is_true(row.get(f"{prefix}actual_state_valid")):
            failures.append(f"{side}:actual_state_invalid")
        if not _is_false(row.get(f"{prefix}actual_state_stale")):
            failures.append(f"{side}:actual_state_stale_or_unconfirmed")
        if not _is_true(row.get(f"{prefix}causal_measured_state_eligible")):
            reason = row.get(f"{prefix}causal_measured_state_rejection_reason")
            failures.append(f"{side}:causal_state_ineligible:{reason}")

        sequence_value = row.get(f"{prefix}actual_state_sequence")
        if isinstance(sequence_value, (bool, np.bool_)):
            sequence = None
        else:
            try:
                sequence = int(sequence_value)
            except (TypeError, ValueError):
                sequence = None
        if sequence is None or sequence <= 0 or sequence != sequence_value:
            failures.append(f"{side}:invalid_actual_state_sequence")

        timestamps = {
            "actual_state_timestamp": _finite_scalar(
                row.get(f"{prefix}actual_state_timestamp")
            ),
            "source_timestamp": _finite_scalar(row.get(f"{prefix}source_timestamp")),
            "available_timestamp": _finite_scalar(
                row.get(f"{prefix}available_timestamp")
            ),
        }
        for name, timestamp in timestamps.items():
            if timestamp is None:
                failures.append(f"{side}:missing_{name}")
            elif timestamp > sample_timestamp + 1e-6:
                failures.append(f"{side}:{name}_after_T0")
        actual_timestamp = timestamps["actual_state_timestamp"]
        source_timestamp = timestamps["source_timestamp"]
        if (
            actual_timestamp is not None
            and source_timestamp is not None
            and abs(actual_timestamp - source_timestamp) > 1e-6
        ):
            failures.append(f"{side}:actual_and_source_timestamp_disagree")
        source_clock = row.get(f"{prefix}source_clock")
        allowed_source_clocks = {"local_wall_time"}
        if str(contract.source.source_family) == "vive_umi":
            allowed_source_clocks.update(
                {
                    "vive_sender_monotonic_calibrated_to_local_wall",
                    "vive_receiver_wall_fallback",
                }
            )
        if source_clock not in allowed_source_clocks:
            failures.append(f"{side}:unsupported_source_clock:{source_clock}")
        if row.get(f"{prefix}available_clock") != "local_wall_time":
            failures.append(f"{side}:available_clock_not_local_wall_time")

    if failures:
        raise PoseFrameContractError(
            "v2.3 collection row has no valid causal measured TCP anchor: "
            + "; ".join(failures)
        )


def _pose7_or_none(value: Any, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    try:
        pose = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise PoseFrameContractError(f"{name} must be a numeric pose7 vector.") from exc
    if pose.shape != (7,):
        raise PoseFrameContractError(f"{name} must have shape (7,), got {pose.shape}.")
    if not np.all(np.isfinite(pose)):
        return None
    return pose


def _convert_pose7(
    value: Any,
    *,
    side: str,
    contract: PoseFrameContract,
    name: str,
) -> list[float] | Any:
    pose = _pose7_or_none(value, name=name)
    if pose is None:
        return value
    return contract.source.raw_pose_vector_to_canonical_desk(
        pose, side=side
    ).tolist()


def _convert_packed_arm_prefix(
    row: MutableMapping[str, Any],
    *,
    key: str,
    contract: PoseFrameContract,
) -> None:
    value = row.get(key)
    if value is None:
        return
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PoseFrameContractError(f"row.{key} must be a pose-prefixed sequence.")
    packed = list(value)
    if len(packed) < 14:
        raise PoseFrameContractError(
            f"row.{key} must contain the 14D left/right arm pose prefix."
        )
    for side, start in (("left", 0), ("right", 7)):
        if side not in contract.source.active_sides:
            continue
        packed[start : start + 7] = _convert_pose7(
            packed[start : start + 7],
            side=side,
            contract=contract,
            name=f"row.{key}.{side}",
        )
    row[key] = packed


def bind_collection_row_pose_storage(
    row: Mapping[str, Any],
    *,
    contract: PoseFrameContract,
) -> dict[str, Any]:
    """Return a detached v2.3 row in the contract's declared storage frame.

    Collection rows currently enter this boundary in each source side's raw
    absolute frame.  Raw storage is therefore label-only; canonical desk
    storage transforms every standard arm pose copy.  Explicit base-frame
    audit fields such as ``actual_hardware_pose_6d`` and wrist-camera
    ``pose_base`` intentionally remain raw.
    """

    output = deepcopy(dict(row))
    if contract.pose_storage_frame == POSE_STORAGE_FRAME_DESK:
        for key in _PACKED_ARM_POSE_FIELDS:
            _convert_packed_arm_prefix(output, key=key, contract=contract)
        for side in contract.source.active_sides:
            for template in _SIDE_POSE_FIELD_TEMPLATES:
                key = template.format(side=side)
                if key not in output:
                    continue
                output[key] = _convert_pose7(
                    output[key],
                    side=side,
                    contract=contract,
                    name=f"row.{key}",
                )

    # q_ref FK is a controller/raw-frame audit value, not a model action copy.
    # Preserve its numeric value and make its per-side source frame explicit.
    for side in contract.source.active_sides:
        for template in _RAW_AUDIT_POSE_FIELD_TEMPLATES:
            key = template.format(side=side)
            if key in output and output[key] is not None:
                output[f"{key}_frame"] = contract.source.raw_frame(side)

    output["dexteleop.dataset_schema"] = DEXTELEOP_DATASET_SCHEMA_V23
    output["dexteleop.pose_storage_frame"] = contract.pose_storage_frame
    output["dexteleop.pose_source_id"] = contract.source_id
    output["dexteleop.pose_frame_contract_hash"] = contract.contract_hash
    output["dexteleop.causal_snapshot_contract"] = (
        DEFAULT_CAUSAL_SNAPSHOT_CONTRACT.schema
    )
    output["dexteleop.causal_snapshot_contract_hash"] = (
        DEFAULT_CAUSAL_SNAPSHOT_CONTRACT.contract_hash
    )
    return output


__all__ = [
    "bind_collection_row_pose_storage",
    "validate_v23_collection_row_actual_tcp",
]
