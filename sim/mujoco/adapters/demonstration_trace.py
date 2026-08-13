"""Read-only export of committed MuJoCo ControlTrace into nominal demos.

The exported reference comes from ``safe_action``: after shared canonical
safety, but before pose resolution, IK, backend command generation, and plant
measurement.  Accepted dispatch and a post-dispatch MuJoCo measurement are
required as execution evidence; neither is relabelled as the pose target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from dataset_interface.control_trace import (
    ActualDecisionRow,
    ControlTraceReadError,
    EpisodeControlTrace,
    materialize_actual_decision_view,
)
from robot_motion.action import HandActionSpec
from sim.demonstration_editing import DemonstrationTrajectory


SAFE_ACTION_DEMONSTRATION_STAGE_V1 = (
    "safe_action_post_shared_safety_pre_resolution_v1"
)


class ControlTraceDemonstrationError(ControlTraceReadError):
    """A committed trace cannot support the requested demonstration export."""


def _mujoco_backend_measurement_version() -> str:
    # Importing sim.mujoco.deploy during adapters package initialization would
    # create deploy -> adapters -> deploy.  Measurement validation runs only
    # after the adapters package is fully initialized, so keep this edge lazy.
    from sim.mujoco.deploy.measurement import MUJOCO_BACKEND_MEASUREMENT_VERSION

    return MUJOCO_BACKEND_MEASUREMENT_VERSION


def _required_text(value: object, *, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must be non-empty.")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _sha256(value: object, *, name: str) -> str:
    result = _required_text(value, name=name).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest.")
    return result


def _non_negative_index(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


@dataclass(frozen=True)
class DemonstrationSuccessEvidence:
    """Task-owned success attestation bound to one explicit decision span.

    The adapter verifies the binding, not the task-specific success predicate.
    The evidence producer remains responsible for the meaning of ``kind`` and
    ``semantic_content_hash``.
    """

    evidence_id: str
    kind: str
    semantic_content_hash: str
    episode_uuid: str
    first_decision_id: str
    last_decision_id: str

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "kind",
            "episode_uuid",
            "first_decision_id",
            "last_decision_id",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "semantic_content_hash",
            _sha256(self.semantic_content_hash, name="semantic_content_hash"),
        )


@dataclass(frozen=True)
class CommittedTraceDemonstrationSelection:
    """Explicit, task-neutral selection of one pose-and-hand trace span."""

    trajectory_id: str
    arm_side: str
    hand_side: str
    hand_component_id: str
    hand_payload_field: str
    hand_target_representation: str
    hand_target_names: tuple[str, ...]
    hand_target_units: tuple[str, ...]
    execution_hand_effector_id: str | None = None
    hand_payload_indices: tuple[int, ...] | None = None
    decision_start_index: int = 0
    decision_end_index: int | None = None
    hand_edit_start_index: int = 0
    hand_edit_end_index: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "trajectory_id",
            "arm_side",
            "hand_side",
            "hand_component_id",
            "hand_target_representation",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name=name),
            )
        if self.hand_payload_field not in {
            "hand_qpos_by_side",
            "end_effector_targets_by_side",
        }:
            raise ValueError(
                "hand_payload_field must be hand_qpos_by_side or "
                "end_effector_targets_by_side."
            )
        for name in ("hand_target_names", "hand_target_units"):
            values = tuple(
                _required_text(value, name=f"{name} item")
                for value in getattr(self, name)
            )
            if not values:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, values)
        if len(self.hand_target_names) != len(self.hand_target_units):
            raise ValueError("hand target names and units must have the same length.")
        if len(set(self.hand_target_names)) != len(self.hand_target_names):
            raise ValueError("hand_target_names must be unique.")
        execution_hand_effector_id = (
            self.hand_component_id
            if self.execution_hand_effector_id is None
            else _required_text(
                self.execution_hand_effector_id,
                name="execution_hand_effector_id",
            )
        )
        object.__setattr__(
            self,
            "execution_hand_effector_id",
            execution_hand_effector_id,
        )
        if self.hand_payload_indices is not None:
            indices = tuple(
                _non_negative_index(value, name="hand_payload_indices item")
                for value in self.hand_payload_indices
            )
            if len(indices) != len(self.hand_target_names):
                raise ValueError(
                    "hand_payload_indices must match the hand target dimension."
                )
            if len(indices) != len(set(indices)):
                raise ValueError("hand_payload_indices must be unique.")
            object.__setattr__(self, "hand_payload_indices", indices)
        object.__setattr__(
            self,
            "decision_start_index",
            _non_negative_index(
                self.decision_start_index,
                name="decision_start_index",
            ),
        )
        object.__setattr__(
            self,
            "hand_edit_start_index",
            _non_negative_index(
                self.hand_edit_start_index,
                name="hand_edit_start_index",
            ),
        )
        for name in ("decision_end_index", "hand_edit_end_index"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _non_negative_index(value, name=name),
                )

    @property
    def arm_component_id(self) -> str:
        return f"{self.arm_side}_arm"


def _record_by_fact_id(trace: EpisodeControlTrace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in (*trace.dependency_records, *trace.episode_records):
        fact_id = str(record.fact.fact_id)
        if fact_id in result:
            raise ControlTraceDemonstrationError(
                f"ControlTrace contains duplicate fact ID {fact_id!r}."
            )
        result[fact_id] = record.fact
    return result


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlTraceDemonstrationError(f"{name} must be a mapping.")
    return value


def _finite_vector(value: object, *, name: str, size: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if size is not None and result.shape != (size,):
        raise ControlTraceDemonstrationError(
            f"{name} must have shape ({size},), got {result.shape}."
        )
    if not result.size or not np.all(np.isfinite(result)):
        raise ControlTraceDemonstrationError(f"{name} must contain finite values.")
    return result


def _extract_hand_target(
    action: Mapping[str, Any],
    *,
    selection: CommittedTraceDemonstrationSelection,
    row_label: str,
) -> np.ndarray:
    values_by_side = _mapping(
        action.get(selection.hand_payload_field),
        name=f"{row_label}.{selection.hand_payload_field}",
    )
    if selection.hand_side not in values_by_side:
        raise ControlTraceDemonstrationError(
            f"{row_label} has no {selection.hand_payload_field} value for "
            f"side {selection.hand_side!r}."
        )
    raw_value = values_by_side[selection.hand_side]
    if selection.hand_payload_field == "end_effector_targets_by_side":
        target = _mapping(raw_value, name=f"{row_label}.end_effector_target")
        if target.get("value_type") != "continuous":
            raise ControlTraceDemonstrationError(
                f"{row_label} end-effector target must be continuous."
            )
        payload_names = tuple(str(value) for value in target.get("names", ()))
        raw_value = target.get("values")
    else:
        payload_names = ()
    source = _finite_vector(
        raw_value,
        name=f"{row_label}.hand_target",
    )
    if selection.hand_payload_indices is None:
        if payload_names and payload_names != selection.hand_target_names:
            raise ControlTraceDemonstrationError(
                f"{row_label} end-effector target names do not match selection."
            )
        if source.shape != (len(selection.hand_target_names),):
            raise ControlTraceDemonstrationError(
                f"{row_label}.hand_target must have shape "
                f"({len(selection.hand_target_names)},), got {source.shape}."
            )
        return source

    indices = selection.hand_payload_indices
    assert indices is not None
    if selection.hand_payload_field == "hand_qpos_by_side":
        execution_metadata = _mapping(
            action.get("execution_metadata"),
            name=f"{row_label}.execution_metadata",
        )
        hand_specs = _mapping(
            execution_metadata.get("hand_action_specs_by_side"),
            name=f"{row_label}.hand_action_specs_by_side",
        )
        raw_spec = _mapping(
            hand_specs.get(selection.hand_side),
            name=f"{row_label}.hand_action_spec[{selection.hand_side!r}]",
        )
        try:
            hand_spec = HandActionSpec.from_metadata(raw_spec)
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlTraceDemonstrationError(
                f"{row_label} hand action spec is invalid: {exc}"
            ) from exc
        if hand_spec.side != selection.hand_side:
            raise ControlTraceDemonstrationError(
                f"{row_label} hand action spec side does not match selection."
            )
        for name, index in zip(
            selection.hand_target_names,
            indices,
            strict=True,
        ):
            if hand_spec.slot_map.get(name) != index:
                raise ControlTraceDemonstrationError(
                    f"{row_label} hand payload slot {index} is not bound to "
                    f"target {name!r}."
                )
    if max(indices) >= source.shape[0]:
        raise ControlTraceDemonstrationError(
            f"{row_label}.hand_target does not contain all selected payload indices."
        )
    if payload_names:
        if len(payload_names) != source.shape[0]:
            raise ControlTraceDemonstrationError(
                f"{row_label} end-effector names and values have different lengths."
            )
        selected_names = tuple(payload_names[index] for index in indices)
        if selected_names != selection.hand_target_names:
            raise ControlTraceDemonstrationError(
                f"{row_label} selected end-effector target names do not match selection."
            )
    return source[np.asarray(indices, dtype=np.int64)]


def _validate_execution_evidence(
    row: ActualDecisionRow,
    *,
    facts: Mapping[str, Any],
    selection: CommittedTraceDemonstrationSelection,
) -> None:
    label = f"decision {row.decision_id!r}"
    if row.execution_status != "normal":
        raise ControlTraceDemonstrationError(
            f"{label} execution directive is {row.execution_status!r}, not normal."
        )
    if row.safety_status not in {"safe", "modified"}:
        raise ControlTraceDemonstrationError(
            f"{label} shared safety status {row.safety_status!r} is not executable."
        )
    if row.command_id is None:
        raise ControlTraceDemonstrationError(f"{label} has no command.")
    if len(row.dispatch_receipt_ids) != 1:
        raise ControlTraceDemonstrationError(
            f"{label} must have exactly one dispatch receipt."
        )
    if len(row.measurement_alignment_ids) != 1:
        raise ControlTraceDemonstrationError(
            f"{label} must have exactly one post-dispatch measurement."
        )

    dispatch = facts.get(row.dispatch_receipt_ids[0])
    if dispatch is None or dispatch.event_type != "dispatch_receipt":
        raise ControlTraceDemonstrationError(f"{label} dispatch receipt is missing.")
    dispatch_payload = _mapping(dispatch.payload, name=f"{label}.dispatch")
    if dispatch_payload.get("command_id") != row.command_id:
        raise ControlTraceDemonstrationError(f"{label} dispatch command lineage differs.")
    if dispatch_payload.get("all_accepted") is not True:
        raise ControlTraceDemonstrationError(f"{label} dispatch was not fully accepted.")
    receipts = _mapping(
        dispatch_payload.get("receipts_by_effector"),
        name=f"{label}.receipts_by_effector",
    )
    for component_id in (
        selection.arm_component_id,
        selection.execution_hand_effector_id,
    ):
        receipt = _mapping(
            receipts.get(component_id),
            name=f"{label}.receipt[{component_id!r}]",
        )
        if receipt.get("accepted") is not True:
            raise ControlTraceDemonstrationError(
                f"{label} component {component_id!r} was not accepted."
            )

    measurement = facts.get(row.measurement_alignment_ids[0])
    if measurement is None or measurement.event_type != "measurement_alignment":
        raise ControlTraceDemonstrationError(f"{label} measurement is missing.")
    measurement_payload = _mapping(
        measurement.payload,
        name=f"{label}.measurement_alignment",
    )
    for key, expected in (
        ("decision_id", row.decision_id),
        ("command_id", row.command_id),
        ("dispatch_receipt_id", row.dispatch_receipt_ids[0]),
    ):
        if measurement_payload.get(key) != expected:
            raise ControlTraceDemonstrationError(
                f"{label} measurement {key} lineage differs."
            )
    if measurement_payload.get("measurement_schema_id") != "dexteleop.joint_measurement.v1":
        raise ControlTraceDemonstrationError(
            f"{label} measurement schema is not the MuJoCo joint measurement contract."
        )
    measured = _mapping(
        measurement_payload.get("measurement"),
        name=f"{label}.measurement",
    )
    if measured.get("schema_version") != _mujoco_backend_measurement_version():
        raise ControlTraceDemonstrationError(
            f"{label} measurement is not a MuJoCo backend measurement."
        )
    if float(measurement_payload["measured_at"]) < float(dispatch_payload["dispatched_at"]):
        raise ControlTraceDemonstrationError(
            f"{label} measurement precedes its dispatch receipt."
        )


def demonstration_from_committed_control_trace(
    trace: EpisodeControlTrace,
    *,
    selection: CommittedTraceDemonstrationSelection,
    success_evidence: DemonstrationSuccessEvidence,
) -> DemonstrationTrajectory:
    """Export an explicitly selected, successfully executed nominal demo."""

    if not isinstance(trace, EpisodeControlTrace):
        raise TypeError("trace must be an EpisodeControlTrace.")
    if not isinstance(selection, CommittedTraceDemonstrationSelection):
        raise TypeError("selection must be a CommittedTraceDemonstrationSelection.")
    if not isinstance(success_evidence, DemonstrationSuccessEvidence):
        raise TypeError("success_evidence must be a DemonstrationSuccessEvidence.")
    if not trace.committed or trace.commit_payload is None:
        raise ControlTraceDemonstrationError(
            "Demonstration export requires a committed ControlTrace episode."
        )
    if "control_trace_core" not in trace.capabilities:
        raise ControlTraceDemonstrationError(
            "Committed trace lacks the control_trace_core capability."
        )
    if success_evidence.episode_uuid != trace.episode_uuid:
        raise ControlTraceDemonstrationError(
            "Success evidence is bound to another episode."
        )
    try:
        commit_hash = _sha256(
            trace.commit_payload.get("semantic_content_hash"),
            name="capture commit semantic_content_hash",
        )
    except ValueError as exc:
        raise ControlTraceDemonstrationError(str(exc)) from exc

    rows = materialize_actual_decision_view(trace)
    end = (
        len(rows) - 1
        if selection.decision_end_index is None
        else selection.decision_end_index
    )
    if selection.decision_start_index > end or end >= len(rows):
        raise ControlTraceDemonstrationError(
            "Selected decision span is outside the actual decision view."
        )
    selected = rows[selection.decision_start_index : end + 1]
    if len(selected) < 2:
        raise ControlTraceDemonstrationError(
            "A demonstration requires at least two authoritative decisions."
        )
    if (
        success_evidence.first_decision_id != selected[0].decision_id
        or success_evidence.last_decision_id != selected[-1].decision_id
    ):
        raise ControlTraceDemonstrationError(
            "Success evidence decision span does not match the selection."
        )

    facts = _record_by_fact_id(trace)
    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    hands: list[np.ndarray] = []
    reference_frame: str | None = None
    controlled_frame: str | None = None
    action_space: str | None = None
    first_effective_at = float(selected[0].effective_interval_start)
    for row in selected:
        _validate_execution_evidence(
            row,
            facts=facts,
            selection=selection,
        )
        action = _mapping(row.safe_action, name=f"decision {row.decision_id!r}.safe_action")
        active = tuple(str(value) for value in action.get("active_effectors", ()))
        required = {
            selection.arm_component_id,
            selection.execution_hand_effector_id,
        }
        if not required.issubset(active):
            raise ControlTraceDemonstrationError(
                f"decision {row.decision_id!r} safe action does not activate "
                f"{sorted(required)}."
            )
        arm_by_side = _mapping(
            action.get("arm_pose_by_side"),
            name=f"decision {row.decision_id!r}.arm_pose_by_side",
        )
        pose = _finite_vector(
            arm_by_side.get(selection.arm_side),
            name=f"decision {row.decision_id!r}.arm_pose",
            size=7,
        )
        quaternion_norm = float(np.linalg.norm(pose[3:7]))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ControlTraceDemonstrationError(
                f"decision {row.decision_id!r} arm pose quaternion is not unit length."
            )
        hand = _extract_hand_target(
            action,
            selection=selection,
            row_label=f"decision {row.decision_id!r}",
        )
        frames = _mapping(
            action.get("frame_by_effector"),
            name=f"decision {row.decision_id!r}.frame_by_effector",
        )
        controlled_frames = _mapping(
            action.get("controlled_frame_by_effector"),
            name=f"decision {row.decision_id!r}.controlled_frame_by_effector",
        )
        row_reference = _required_text(
            frames.get(selection.arm_component_id),
            name="safe action reference frame",
        )
        row_controlled = _required_text(
            controlled_frames.get(selection.arm_component_id),
            name="safe action controlled frame",
        )
        row_action_space = _required_text(
            action.get("action_space"),
            name="safe action action_space",
        )
        if reference_frame is None:
            reference_frame = row_reference
            controlled_frame = row_controlled
            action_space = row_action_space
        elif (
            row_reference != reference_frame
            or row_controlled != controlled_frame
            or row_action_space != action_space
        ):
            raise ControlTraceDemonstrationError(
                "Selected safe actions do not share one frame/action contract."
            )
        timestamp = float(row.effective_interval_start) - first_effective_at
        if not math.isfinite(timestamp):
            raise ControlTraceDemonstrationError("Decision timestamp is not finite.")
        timestamps.append(timestamp)
        poses.append(pose)
        hands.append(hand)

    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ControlTraceDemonstrationError(
            "Selected decision effective times must be strictly increasing."
        )
    hand_end = (
        len(selected) - 1
        if selection.hand_edit_end_index is None
        else selection.hand_edit_end_index
    )
    if selection.hand_edit_start_index > hand_end or hand_end >= len(selected):
        raise ControlTraceDemonstrationError(
            "Hand edit interval is outside the selected demonstration span."
        )
    assert reference_frame is not None
    assert controlled_frame is not None
    assert action_space is not None
    return DemonstrationTrajectory(
        trajectory_id=selection.trajectory_id,
        reference_frame=reference_frame,
        controlled_frame=controlled_frame,
        hand_component_id=selection.hand_component_id,
        hand_target_representation=selection.hand_target_representation,
        hand_target_names=selection.hand_target_names,
        hand_target_units=selection.hand_target_units,
        timestamps_sec=np.asarray(timestamps, dtype=np.float64),
        tool_pose_xyzquat_xyzw=np.stack(poses, axis=0),
        hand_targets=np.stack(hands, axis=0),
        hand_edit_start_index=selection.hand_edit_start_index,
        hand_edit_end_index=hand_end,
        source_kind="committed_mujoco_control_trace_safe_action",
        source_id=trace.episode_uuid,
        source_content_hash=commit_hash,
        metadata={
            "trace_session_id": trace.trace_session_id,
            "trace_profile_id": trace.profile_id,
            "nominal_action_stage": SAFE_ACTION_DEMONSTRATION_STAGE_V1,
            "execution_evidence": (
                "accepted_dispatch_with_post_dispatch_mujoco_measurement"
            ),
            "success_evidence_id": success_evidence.evidence_id,
            "success_evidence_kind": success_evidence.kind,
            "success_evidence_hash": success_evidence.semantic_content_hash,
            "first_decision_id": selected[0].decision_id,
            "last_decision_id": selected[-1].decision_id,
            "action_space": action_space,
            "execution_hand_effector_id": selection.execution_hand_effector_id,
            "hand_payload_indices": (
                "all"
                if selection.hand_payload_indices is None
                else ",".join(str(index) for index in selection.hand_payload_indices)
            ),
        },
    )


__all__ = [
    "SAFE_ACTION_DEMONSTRATION_STAGE_V1",
    "CommittedTraceDemonstrationSelection",
    "ControlTraceDemonstrationError",
    "DemonstrationSuccessEvidence",
    "demonstration_from_committed_control_trace",
]
