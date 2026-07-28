"""RobotAction-v1 compatibility execution-target admission contracts.

This module preserves the existing MuJoCo shared-safety/RobotAction-v1 target
path while it is migrated.  New target handlers should use the component-keyed
contracts in :mod:`robot_motion.action.execution_handoff`; those contracts do
not require a universal safe-action stage.

The compatibility contracts here define linearization evidence between the
existing shared canonical safety result and an execution worker.  They
deliberately do not enqueue a position source keyframe, publish a reference,
dispatch a command, or own an anchor registry.  In particular, frontend queue
admission is not execution-target admission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.immutable import deep_freeze

from .proposal import freeze_robot_action, robot_action_semantic_payload
from .safety_anchor import SafetyAnchorCommitEvidence, safety_anchor_action_hash
from .schema import RobotAction, split_effector_name


EXECUTION_TARGET_HASH_DOMAIN_V1 = "dexteleop.resolved_execution_target.v1"
EXECUTION_TARGET_PAYLOAD_HASH_DOMAIN_V1 = (
    "dexteleop.resolved_execution_target_payload.v1"
)
EXECUTION_TARGET_ADMISSION_HASH_DOMAIN_V1 = (
    "dexteleop.execution_target_admission.v1"
)

EXECUTION_TARGET_KINDS_V1 = frozenset(
    {
        "direct_command_intent",
        "pose_reference",
        "qpos_reference",
        "trajectory_segment",
    }
)
EXECUTION_TARGET_ANCHOR_COMMIT_POLICIES_V1 = frozenset(
    {"on_target_admission", "on_backend_acceptance"}
)


def _required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite_time(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


def _target_commit_policy(target_kind: str) -> str:
    if target_kind == "direct_command_intent":
        return "on_backend_acceptance"
    return "on_target_admission"


@dataclass(frozen=True)
class ResolvedExecutionTarget:
    """Legacy RobotAction-v1 resolver output for one selection group.

    ``scheduled_from``/``scheduled_until`` form a half-open interval.  The
    payload is stage-specific and interpreted only under
    ``output_target_schema_hash`` and the resolver/planner contract hashes.
    """

    target_id: str
    control_domain_id: str
    selection_group_id: str
    execution_epoch: int
    group_segment_generation: int
    source_safety_result_id: str
    source_safe_action_hash: str
    group_safe_action: RobotAction
    group_safe_action_hash: str
    target_kind: str
    anchor_commit_policy: str
    component_dependencies: tuple[str, ...]
    resolved_at: float
    scheduled_from: float
    scheduled_until: float
    clock_domain_id: str
    input_action_schema_hash: str
    output_target_schema_hash: str
    resolver_contract_id: str
    resolver_contract_hash: str
    transform_contract_hash: str | None = None
    planner_contract_hash: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    payload_hash: str = ""
    semantic_content_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "target_id",
            "control_domain_id",
            "selection_group_id",
            "source_safety_result_id",
            "source_safe_action_hash",
            "group_safe_action_hash",
            "clock_domain_id",
            "input_action_schema_hash",
            "output_target_schema_hash",
            "resolver_contract_id",
            "resolver_contract_hash",
            "payload_hash",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        if self.target_kind not in EXECUTION_TARGET_KINDS_V1:
            raise ValueError("Unsupported resolved execution-target kind.")
        if self.anchor_commit_policy not in EXECUTION_TARGET_ANCHOR_COMMIT_POLICIES_V1:
            raise ValueError("Unsupported execution-target anchor commit policy.")
        if self.anchor_commit_policy != _target_commit_policy(self.target_kind):
            raise ValueError(
                "Execution-target kind and anchor commit policy disagree."
            )
        for name in ("transform_contract_hash", "planner_contract_hash"):
            value = getattr(self, name)
            if value is not None:
                _required_text(value, name=name)
        _non_negative_int(self.execution_epoch, name="execution_epoch")
        _non_negative_int(
            self.group_segment_generation,
            name="group_segment_generation",
        )
        dependencies = tuple(
            _required_text(value, name="component_dependency")
            for value in self.component_dependencies
        )
        if not dependencies or len(set(dependencies)) != len(dependencies):
            raise ValueError(
                "component_dependencies must be non-empty and unique."
            )
        resolved_at = _finite_time(self.resolved_at, name="resolved_at")
        scheduled_from = _finite_time(self.scheduled_from, name="scheduled_from")
        scheduled_until = _finite_time(self.scheduled_until, name="scheduled_until")
        if scheduled_until <= scheduled_from:
            raise ValueError(
                "Resolved execution-target interval must be [from, until)."
            )
        payload = deep_freeze(dict(self.payload))
        payload_hash = canonical_content_hash(
            payload,
            domain=EXECUTION_TARGET_PAYLOAD_HASH_DOMAIN_V1,
        )
        if payload_hash != self.payload_hash:
            raise ValueError("Resolved execution-target payload hash mismatch.")
        group_action = freeze_robot_action(self.group_safe_action)
        if safety_anchor_action_hash(group_action) != self.group_safe_action_hash:
            raise ValueError("Resolved execution-target group action hash mismatch.")
        if tuple(group_action.active_effectors) != dependencies:
            raise ValueError(
                "Resolved execution-target dependencies must equal the projected "
                "group action effectors."
            )
        object.__setattr__(self, "component_dependencies", dependencies)
        object.__setattr__(self, "resolved_at", resolved_at)
        object.__setattr__(self, "scheduled_from", scheduled_from)
        object.__setattr__(self, "scheduled_until", scheduled_until)
        object.__setattr__(self, "group_safe_action", group_action)
        object.__setattr__(self, "payload", MappingProxyType(dict(payload)))


def project_robot_action_to_components(
    action: RobotAction,
    component_dependencies: tuple[str, ...],
) -> RobotAction:
    """Project one canonical action onto an explicit component subset."""

    frozen = freeze_robot_action(action)
    dependencies = tuple(
        _required_text(value, name="component_dependency")
        for value in component_dependencies
    )
    if not dependencies or len(set(dependencies)) != len(dependencies):
        raise ValueError("component_dependencies must be non-empty and unique.")
    unavailable = tuple(
        value for value in dependencies if value not in frozen.active_effectors
    )
    if unavailable:
        raise ValueError(
            "Execution-target dependencies are absent from the shared-safe action: "
            + ",".join(unavailable)
        )
    arm_pose: dict[str, Any] = {}
    arm_qpos: dict[str, Any] = {}
    hand_qpos: dict[str, Any] = {}
    native_targets: dict[str, Any] = {}
    frames: dict[str, str] = {}
    controlled_frames: dict[str, str] = {}
    for component in dependencies:
        side, kind = split_effector_name(component)
        if kind == "arm":
            if side in frozen.arm_pose_by_side:
                arm_pose[side] = frozen.arm_pose_by_side[side]
            elif side in frozen.arm_qpos_by_side:
                arm_qpos[side] = frozen.arm_qpos_by_side[side]
            else:
                raise ValueError(f"Safe action lacks payload for {component}.")
        else:
            if side in frozen.hand_qpos_by_side:
                hand_qpos[side] = frozen.hand_qpos_by_side[side]
            elif side in frozen.end_effector_targets_by_side:
                native_targets[side] = frozen.end_effector_targets_by_side[side]
            else:
                raise ValueError(f"Safe action lacks payload for {component}.")
        if component in frozen.frame_by_effector:
            frames[component] = frozen.frame_by_effector[component]
        if component in frozen.controlled_frame_by_effector:
            controlled_frames[component] = frozen.controlled_frame_by_effector[component]
    return freeze_robot_action(
        RobotAction(
            action_space=frozen.action_space,
            timestamp=frozen.timestamp,
            arm_pose_by_side=arm_pose,
            arm_qpos_by_side=arm_qpos,
            hand_qpos_by_side=hand_qpos,
            end_effector_targets_by_side=native_targets,
            active_effectors=dependencies,
            frame_by_effector=frames,
            controlled_frame_by_effector=controlled_frames,
            source=frozen.source,
            # Anchor continuity uses values/frame/schema.  Arbitrary action
            # metadata is intentionally not projected across group boundaries.
            metadata={},
        )
    )


@dataclass(frozen=True)
class ExecutionTargetAdmissionReceipt:
    admission_id: str
    target_id: str
    target_hash: str
    control_domain_id: str
    selection_group_id: str
    execution_epoch: int
    group_segment_generation: int
    receiver_id: str
    admitted_at: float
    clock_domain_id: str
    accepted: bool
    rejection_reason: str | None
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "admission_id",
            "target_id",
            "target_hash",
            "control_domain_id",
            "selection_group_id",
            "receiver_id",
            "clock_domain_id",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        _non_negative_int(self.execution_epoch, name="execution_epoch")
        _non_negative_int(
            self.group_segment_generation,
            name="group_segment_generation",
        )
        _finite_time(self.admitted_at, name="admitted_at")
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("Accepted target admission cannot have a rejection reason.")
        if not self.accepted and not str(self.rejection_reason or "").strip():
            raise ValueError("Rejected target admission requires a reason.")


def build_resolved_execution_target(
    *,
    control_domain_id: str,
    selection_group_id: str,
    execution_epoch: int,
    group_segment_generation: int,
    source_safety_result_id: str,
    safe_action: RobotAction,
    target_kind: str,
    component_dependencies: tuple[str, ...],
    resolved_at: float,
    scheduled_from: float,
    scheduled_until: float,
    clock_domain_id: str,
    input_action_schema_hash: str,
    output_target_schema_hash: str,
    resolver_contract_id: str,
    resolver_contract_hash: str,
    payload: Mapping[str, Any],
    transform_contract_hash: str | None = None,
    planner_contract_hash: str | None = None,
) -> ResolvedExecutionTarget:
    kind = _required_text(target_kind, name="target_kind")
    if kind not in EXECUTION_TARGET_KINDS_V1:
        raise ValueError("Unsupported resolved execution-target kind.")
    frozen_payload = deep_freeze(dict(payload))
    payload_hash = canonical_content_hash(
        frozen_payload,
        domain=EXECUTION_TARGET_PAYLOAD_HASH_DOMAIN_V1,
    )
    frozen_safe_action = freeze_robot_action(safe_action)
    safe_hash = safety_anchor_action_hash(frozen_safe_action)
    dependencies = tuple(component_dependencies)
    group_safe_action = project_robot_action_to_components(
        frozen_safe_action,
        dependencies,
    )
    group_safe_action_hash = safety_anchor_action_hash(group_safe_action)
    semantic_payload = {
        "control_domain_id": _required_text(
            control_domain_id, name="control_domain_id"
        ),
        "selection_group_id": _required_text(
            selection_group_id, name="selection_group_id"
        ),
        "execution_epoch": _non_negative_int(
            execution_epoch, name="execution_epoch"
        ),
        "group_segment_generation": _non_negative_int(
            group_segment_generation,
            name="group_segment_generation",
        ),
        "source_safety_result_id": _required_text(
            source_safety_result_id, name="source_safety_result_id"
        ),
        "source_safe_action_hash": safe_hash,
        "group_safe_action": robot_action_semantic_payload(group_safe_action),
        "group_safe_action_hash": group_safe_action_hash,
        "target_kind": kind,
        "anchor_commit_policy": _target_commit_policy(kind),
        "component_dependencies": dependencies,
        "resolved_at": _finite_time(resolved_at, name="resolved_at"),
        "scheduled_from": _finite_time(scheduled_from, name="scheduled_from"),
        "scheduled_until": _finite_time(scheduled_until, name="scheduled_until"),
        "clock_domain_id": _required_text(
            clock_domain_id, name="clock_domain_id"
        ),
        "input_action_schema_hash": _required_text(
            input_action_schema_hash, name="input_action_schema_hash"
        ),
        "output_target_schema_hash": _required_text(
            output_target_schema_hash, name="output_target_schema_hash"
        ),
        "resolver_contract_id": _required_text(
            resolver_contract_id, name="resolver_contract_id"
        ),
        "resolver_contract_hash": _required_text(
            resolver_contract_hash, name="resolver_contract_hash"
        ),
        "transform_contract_hash": (
            None
            if transform_contract_hash is None
            else _required_text(
                transform_contract_hash,
                name="transform_contract_hash",
            )
        ),
        "planner_contract_hash": (
            None
            if planner_contract_hash is None
            else _required_text(planner_contract_hash, name="planner_contract_hash")
        ),
        "payload_hash": payload_hash,
        "payload": frozen_payload,
    }
    if semantic_payload["scheduled_until"] <= semantic_payload["scheduled_from"]:
        raise ValueError("Resolved execution-target interval must be [from, until).")
    semantic_hash = canonical_content_hash(
        semantic_payload,
        domain=EXECUTION_TARGET_HASH_DOMAIN_V1,
    )
    return ResolvedExecutionTarget(
        target_id=f"resolved-execution-target:{semantic_hash}",
        **{
            key: value
            for key, value in semantic_payload.items()
            if key != "group_safe_action"
        },
        group_safe_action=group_safe_action,
        semantic_content_hash=semantic_hash,
    )


def admit_resolved_execution_target(
    target: ResolvedExecutionTarget,
    *,
    receiver_id: str,
    admitted_at: float,
    current_execution_epoch: int,
    expected_group_segment_generation: int,
    receiver_accepts: bool = True,
    rejection_reason: str | None = None,
) -> ExecutionTargetAdmissionReceipt:
    """Create receiver-side admission evidence without mutating a worker.

    Scope, generation, clock, and expiry checks fail closed.  A receiver may
    additionally reject a valid target with an explicit reason.
    """

    if not isinstance(target, ResolvedExecutionTarget):
        raise TypeError("target must be a ResolvedExecutionTarget.")
    timestamp = _finite_time(admitted_at, name="admitted_at")
    current_epoch = _non_negative_int(
        current_execution_epoch, name="current_execution_epoch"
    )
    expected_generation = _non_negative_int(
        expected_group_segment_generation,
        name="expected_group_segment_generation",
    )
    reason: str | None = None
    if current_epoch != target.execution_epoch:
        reason = "execution_epoch_mismatch"
    elif expected_generation != target.group_segment_generation:
        reason = "group_segment_generation_mismatch"
    elif timestamp < target.resolved_at:
        reason = "target_not_yet_resolved"
    elif timestamp >= target.scheduled_until:
        reason = "target_interval_expired"
    elif not receiver_accepts:
        reason = _required_text(rejection_reason, name="rejection_reason")
    elif rejection_reason is not None:
        raise ValueError("Accepted target admission cannot have a rejection reason.")
    accepted = reason is None
    payload = {
        "target_id": target.target_id,
        "target_hash": target.semantic_content_hash,
        "control_domain_id": target.control_domain_id,
        "selection_group_id": target.selection_group_id,
        "execution_epoch": target.execution_epoch,
        "group_segment_generation": target.group_segment_generation,
        "receiver_id": _required_text(receiver_id, name="receiver_id"),
        "admitted_at": timestamp,
        "clock_domain_id": target.clock_domain_id,
        "accepted": accepted,
        "rejection_reason": reason,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=EXECUTION_TARGET_ADMISSION_HASH_DOMAIN_V1,
    )
    return ExecutionTargetAdmissionReceipt(
        admission_id=f"execution-target-admission:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


def execution_target_anchor_commit_evidence(
    target: ResolvedExecutionTarget,
    admission: ExecutionTargetAdmissionReceipt,
    *,
    safety_result_id: str,
) -> SafetyAnchorCommitEvidence:
    """Convert a reference/trajectory admission into anchor commit evidence."""

    if not isinstance(target, ResolvedExecutionTarget):
        raise TypeError("target must be a ResolvedExecutionTarget.")
    if not isinstance(admission, ExecutionTargetAdmissionReceipt):
        raise TypeError("admission must be an ExecutionTargetAdmissionReceipt.")
    if target.anchor_commit_policy != "on_target_admission":
        raise ValueError(
            "Direct command intents require backend-acceptance evidence and cannot "
            "commit on target admission."
        )
    if (
        admission.target_id != target.target_id
        or admission.target_hash != target.semantic_content_hash
        or admission.control_domain_id != target.control_domain_id
        or admission.selection_group_id != target.selection_group_id
        or admission.execution_epoch != target.execution_epoch
        or admission.group_segment_generation != target.group_segment_generation
        or admission.clock_domain_id != target.clock_domain_id
    ):
        raise ValueError("Execution-target admission lineage mismatch.")
    expected_safety_result = _required_text(
        safety_result_id, name="safety_result_id"
    )
    if expected_safety_result != target.source_safety_result_id:
        raise ValueError("Execution target references another safety result.")
    return SafetyAnchorCommitEvidence(
        evidence_id=admission.admission_id,
        evidence_kind="execution_target_admitted",
        target_id=target.target_id,
        safety_result_id=expected_safety_result,
        safe_action_hash=target.group_safe_action_hash,
        control_domain_id=target.control_domain_id,
        selection_group_id=target.selection_group_id,
        execution_epoch=target.execution_epoch,
        group_segment_generation=target.group_segment_generation,
        occurred_at=admission.admitted_at,
        accepted=admission.accepted,
        effective_from=target.scheduled_from,
        rejection_reason=admission.rejection_reason,
    )


__all__ = [
    "EXECUTION_TARGET_ANCHOR_COMMIT_POLICIES_V1",
    "EXECUTION_TARGET_KINDS_V1",
    "ExecutionTargetAdmissionReceipt",
    "ResolvedExecutionTarget",
    "admit_resolved_execution_target",
    "build_resolved_execution_target",
    "execution_target_anchor_commit_evidence",
    "project_robot_action_to_components",
]
