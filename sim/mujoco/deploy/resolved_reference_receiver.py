"""Scheduled qpos receiver for a separately resolved shared-safe pose target.

The canonical safe action remains pose.  Coupled IK produces an immutable
``ResolvedExecutionTarget`` carrying qpos payload; only that payload becomes a
reference segment endpoint.  Target admission, segment admission, and the pose
safety-anchor commit evidence remain distinct facts.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from robot_motion.action import (
    AuthoritativeCommandDispatchContext,
    AuthoritativeTargetDispatchContext,
    ComponentExecutionTarget,
    ComponentExecutionTargetAdmissionReceipt,
    ComponentExecutionTargetActivationReceipt,
    CompoundDispatchManifest,
    ExecutionTargetAdmissionReceipt,
    ReferenceSegment,
    ReferenceSegmentAdmissionReceipt,
    ReferenceSegmentTerminal,
    ResolvedExecutionTarget,
    SCHEDULED_REFERENCE_LATE_ADMISSION_REJECTION_V1,
    SafetyAnchorCommitEvidence,
    TargetExecutionCompletionReceipt,
    TargetExecutionDispatchResult,
    admit_component_execution_target,
    admit_reference_segment,
    admit_resolved_execution_target,
    build_pending_group_generation_reservation,
    build_component_execution_target_activation_receipt,
    build_reference_endpoint_parent,
    build_reference_segment,
    build_reference_segment_terminal,
    execution_target_anchor_commit_evidence,
    hand_action_specs_from_metadata,
    project_robot_action_to_components,
    safety_anchor_action_hash,
    validate_compound_dispatch_manifest,
    validate_authoritative_receiver_context,
    validate_authoritative_target_receiver_context,
    validate_reference_segment_successor,
)
from robot_motion.runtime import CommandDispatchReceipt, TeleopCommand

from .pose_resolution import MujocoPoseToQposResolution
from .reference_receiver import (
    MujocoScheduledQposReferenceReceiver,
    ScheduledReferenceAdmissionResult,
    _serialized_receiver_transaction,
)


@dataclass(frozen=True)
class ResolvedQposReferenceAdmissionResult:
    target: ResolvedExecutionTarget
    target_admission: ExecutionTargetAdmissionReceipt
    segment_result: ScheduledReferenceAdmissionResult
    anchor_commit_evidence: SafetyAnchorCommitEvidence | None
    compound_manifest: CompoundDispatchManifest | None = None


@dataclass(frozen=True)
class ComponentQposReferenceAdmissionResult:
    target: ComponentExecutionTarget
    completion_receipt: TargetExecutionCompletionReceipt
    target_admission: ComponentExecutionTargetAdmissionReceipt
    segment_result: ScheduledReferenceAdmissionResult


class MujocoScheduledResolvedQposReferenceReceiver(
    MujocoScheduledQposReferenceReceiver
):
    """Admit coupled IK output without relabeling it as shared-safe qpos."""

    def __init__(
        self,
        *args,
        pose_action_schema_hash: str,
        compound_activation_callback=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pose_action_schema_hash = str(pose_action_schema_hash).strip()
        if not self.pose_action_schema_hash:
            raise ValueError("pose_action_schema_hash must be non-empty.")
        self._resolved_targets: list[ResolvedExecutionTarget] = []
        self._target_admissions: list[ExecutionTargetAdmissionReceipt] = []
        self._anchor_commit_evidence: list[SafetyAnchorCommitEvidence] = []
        if (
            compound_activation_callback is not None
            and not callable(compound_activation_callback)
        ):
            raise TypeError("compound_activation_callback must be callable.")
        self._compound_activation_callback = compound_activation_callback
        self._compound_manifest_by_segment_id: dict[
            str, CompoundDispatchManifest
        ] = {}
        self._last_resolved_result: ResolvedQposReferenceAdmissionResult | None = None
        self._component_targets: list[ComponentExecutionTarget] = []
        self._component_target_admissions: list[
            ComponentExecutionTargetAdmissionReceipt
        ] = []
        self._component_target_activations: list[
            ComponentExecutionTargetActivationReceipt
        ] = []
        self._component_result_by_segment_id: dict[
            str, ComponentQposReferenceAdmissionResult
        ] = {}
        self._component_target_by_segment_id: dict[
            str, ComponentExecutionTarget
        ] = {}
        self._component_admission_by_segment_id: dict[
            str, ComponentExecutionTargetAdmissionReceipt
        ] = {}
        self._component_group_generations_by_segment_id: dict[
            str, Mapping[str, int]
        ] = {}
        self._last_component_result: (
            ComponentQposReferenceAdmissionResult | None
        ) = None

    @property
    def resolved_targets(self) -> tuple[ResolvedExecutionTarget, ...]:
        with self._lock:
            return tuple(self._resolved_targets)

    @property
    def target_admissions(self) -> tuple[ExecutionTargetAdmissionReceipt, ...]:
        with self._lock:
            return tuple(self._target_admissions)

    @property
    def anchor_commit_evidence(self) -> tuple[SafetyAnchorCommitEvidence, ...]:
        with self._lock:
            return tuple(self._anchor_commit_evidence)

    @property
    def last_resolved_result(self) -> ResolvedQposReferenceAdmissionResult | None:
        with self._lock:
            return self._last_resolved_result

    @property
    def component_targets(self) -> tuple[ComponentExecutionTarget, ...]:
        with self._lock:
            return tuple(self._component_targets)

    @property
    def component_target_admissions(
        self,
    ) -> tuple[ComponentExecutionTargetAdmissionReceipt, ...]:
        with self._lock:
            return tuple(self._component_target_admissions)

    @property
    def component_target_activations(
        self,
    ) -> tuple[ComponentExecutionTargetActivationReceipt, ...]:
        with self._lock:
            return tuple(self._component_target_activations)

    @property
    def last_component_result(
        self,
    ) -> ComponentQposReferenceAdmissionResult | None:
        with self._lock:
            return self._last_component_result

    @_serialized_receiver_transaction
    def dispatch_component_target(
        self,
        command: TeleopCommand,
        context: AuthoritativeTargetDispatchContext,
        *,
        target: ComponentExecutionTarget,
        completion_receipt: TargetExecutionCompletionReceipt,
        resolution: MujocoPoseToQposResolution,
        supersedes_pending_segment_id: str | None = None,
    ) -> TargetExecutionDispatchResult:
        """Admit one component target into the existing scheduled owner."""

        with self._lock:
            self._validate_component_resolution_lineage(
                command,
                context,
                target,
                completion_receipt,
                resolution,
            )
            live = self._live_state_provider()
            validation = validate_authoritative_target_receiver_context(
                context,
                live,
            )
            self._validations.append(validation)
            segment = self._build_component_segment(
                target,
                resolution,
                context,
                admitted_at=live.captured_at,
            )
            self._segment_attempts.append(segment)
            self._component_targets.append(target)

            pending = self._pending_reservation_by_group.get(
                segment.selection_group_id
            )
            active_generation = self._active_generation_by_group.get(
                segment.selection_group_id,
                0,
            )
            relation_kind = "initial"
            predecessor_segment_id: str | None = None
            supersedes_reservation_id: str | None = None
            superseded: tuple[ReferenceSegment, ...] = ()
            relation_rejection: str | None = None
            if pending is not None:
                if supersedes_pending_segment_id is None:
                    relation_rejection = "pending_generation_reservation_exists"
                elif supersedes_pending_segment_id != pending.source_segment_id:
                    relation_rejection = "supersedes_pending_segment_id_mismatch"
                elif live.captured_at + 1e-12 >= pending.effective_from:
                    relation_rejection = "pending_replacement_not_before_start"
                else:
                    relation_kind = "supersedes_pending"
                    supersedes_reservation_id = pending.reservation_id
                    superseded = (
                        self._segment_by_id(pending.source_segment_id),
                    )
            else:
                predecessor = self._active_segment_by_group.get(
                    segment.selection_group_id
                )
                if predecessor is not None:
                    try:
                        validate_reference_segment_successor(
                            predecessor,
                            segment,
                        )
                    except ValueError as exc:
                        relation_rejection = f"invalid_successor:{exc}"
                    else:
                        relation_kind = "successor"
                        predecessor_segment_id = (
                            predecessor.reference_segment_id
                        )

            before_start = live.captured_at + 1e-12 < segment.scheduled_from
            receiver_accepts = bool(
                validation.accepted
                and before_start
                and relation_rejection is None
            )
            rejection_reason = (
                validation.status
                if not validation.accepted
                else relation_rejection
                if relation_rejection is not None
                else SCHEDULED_REFERENCE_LATE_ADMISSION_REJECTION_V1
            )
            target_sequence = self._take_sequence()
            target_admission = admit_component_execution_target(
                target,
                receiver_id=(
                    "mujoco-scheduled-component-qpos-reference-receiver"
                ),
                admission_sequence=target_sequence,
                admitted_at=live.captured_at,
                available_at=live.captured_at,
                accepted=receiver_accepts,
                rejection_code=(
                    None if receiver_accepts else rejection_reason
                ),
                admission_token=(
                    f"mujoco-reference-slot:{target.target_id}"
                    if receiver_accepts
                    else None
                ),
            )
            self._component_target_admissions.append(target_admission)
            sequence = self._take_sequence()
            segment_admission = admit_reference_segment(
                segment,
                receiver_id=(
                    "mujoco-scheduled-component-qpos-reference-receiver"
                ),
                admitted_at=live.captured_at,
                admission_sequence=sequence,
                current_execution_epoch=live.current_execution_epoch,
                expected_group_segment_generation=active_generation + 1,
                achieved_admission_guarantee="endpoint_resolved",
                supersedes_segment_ids=tuple(
                    item.reference_segment_id for item in superseded
                ),
                receiver_accepts=(
                    receiver_accepts and target_admission.accepted
                ),
                rejection_reason=(
                    None
                    if receiver_accepts and target_admission.accepted
                    else target_admission.rejection_code or rejection_reason
                ),
            )
            self._admissions.append(segment_admission)
            self._validation_by_admission_id[
                segment_admission.admission_id
            ] = validation

            if not target_admission.accepted or not segment_admission.accepted:
                status = (
                    target_admission.rejection_code
                    or segment_admission.rejection_reason
                    or "component_target_rejected"
                )
                receipts = self._rejected_receipts(
                    command,
                    status=str(status),
                    timestamp=live.captured_at,
                )
                segment_result = ScheduledReferenceAdmissionResult(
                    context_validation=validation,
                    segment=segment,
                    admission=segment_admission,
                    anchor_candidate=None,
                    terminals=(),
                    dispatch_receipts=receipts,
                )
                result = ComponentQposReferenceAdmissionResult(
                    target=target,
                    completion_receipt=completion_receipt,
                    target_admission=target_admission,
                    segment_result=segment_result,
                )
                self._last_result = segment_result
                self._last_component_result = result
                return TargetExecutionDispatchResult(
                    completion_receipt=completion_receipt,
                    target=target,
                    target_admission=target_admission,
                    dispatch_receipts=receipts,
                )

            terminals = tuple(
                self._terminal_for_supersession(
                    old,
                    replacement=segment,
                    available_at=live.captured_at,
                )
                for old in superseded
            )
            self._segments.append(segment)
            self._terminals.extend(terminals)
            reservation = build_pending_group_generation_reservation(
                segment,
                segment_admission,
                active_generation=active_generation,
                relation_kind=relation_kind,
                predecessor_segment_id=predecessor_segment_id,
                supersedes_reservation_id=supersedes_reservation_id,
            )
            if pending is not None:
                self._pending_reservation_by_group.pop(
                    segment.selection_group_id,
                    None,
                )
                self._component_group_generations_by_segment_id.pop(
                    pending.source_segment_id,
                    None,
                )
                self._component_target_by_segment_id.pop(
                    pending.source_segment_id,
                    None,
                )
                self._component_admission_by_segment_id.pop(
                    pending.source_segment_id,
                    None,
                )
            self._pending_reservation_by_group[
                segment.selection_group_id
            ] = reservation
            self._reservations.append(reservation)
            self._component_group_generations_by_segment_id[
                segment.reference_segment_id
            ] = dict(context.group_segment_generation_by_group)
            self._component_target_by_segment_id[
                segment.reference_segment_id
            ] = target
            self._component_admission_by_segment_id[
                segment.reference_segment_id
            ] = target_admission
            try:
                receipts = self._send_scheduled_commands_outside_state_lock(
                    command,
                    activate_at=segment.scheduled_from,
                    reference_segment_id=segment.reference_segment_id,
                    execution_epoch=segment.execution_epoch,
                    group_segment_generation=(
                        segment.group_segment_generation
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - becomes typed receipts.
                if (
                    self._pending_reservation_by_group.get(
                        segment.selection_group_id
                    )
                    == reservation
                ):
                    self._pending_reservation_by_group.pop(
                        segment.selection_group_id,
                        None,
                    )
                self._component_group_generations_by_segment_id.pop(
                    segment.reference_segment_id,
                    None,
                )
                self._component_target_by_segment_id.pop(
                    segment.reference_segment_id,
                    None,
                )
                self._component_admission_by_segment_id.pop(
                    segment.reference_segment_id,
                    None,
                )
                terminal = build_reference_segment_terminal(
                    segment,
                    status="faulted",
                    available_at=float(self._clock()),
                    terminal_admission_sequence=self._take_sequence(),
                    effective_at=float(self._clock()),
                    evidence_ids=(
                        segment_admission.admission_id,
                        target_admission.receipt_id,
                    ),
                )
                self._terminals.append(terminal)
                rejected_at = float(self._clock())
                rejected_activation = (
                    build_component_execution_target_activation_receipt(
                        target,
                        target_admission,
                        receiver_id=(
                            "mujoco-scheduled-component-qpos-reference-receiver"
                        ),
                        activation_sequence=self._take_sequence(),
                        activation_status="rejected",
                        activation_at=rejected_at,
                        available_at=rejected_at,
                        rejection_code=(
                            "scheduled_backend_exception:"
                            f"{type(exc).__name__}"
                        ),
                    )
                )
                self._component_target_activations.append(
                    rejected_activation
                )
                receipts = self._rejected_receipts(
                    command,
                    status="scheduled_backend_exception",
                    timestamp=rejected_at,
                )
                segment_result = ScheduledReferenceAdmissionResult(
                    context_validation=validation,
                    segment=segment,
                    admission=segment_admission,
                    anchor_candidate=None,
                    terminals=(terminal,),
                    dispatch_receipts=receipts,
                )
                result = ComponentQposReferenceAdmissionResult(
                    target=target,
                    completion_receipt=completion_receipt,
                    target_admission=target_admission,
                    segment_result=segment_result,
                )
                self._last_result = segment_result
                self._last_component_result = result
                return TargetExecutionDispatchResult(
                    completion_receipt=completion_receipt,
                    target=target,
                    target_admission=target_admission,
                    target_activation=rejected_activation,
                    dispatch_receipts=receipts,
                )
            segment_result = ScheduledReferenceAdmissionResult(
                context_validation=validation,
                segment=segment,
                admission=segment_admission,
                anchor_candidate=None,
                terminals=terminals,
                dispatch_receipts=receipts,
            )
            result = ComponentQposReferenceAdmissionResult(
                target=target,
                completion_receipt=completion_receipt,
                target_admission=target_admission,
                segment_result=segment_result,
            )
            self._last_result = segment_result
            self._last_component_result = result
            self._component_result_by_segment_id[
                segment.reference_segment_id
            ] = result
            return TargetExecutionDispatchResult(
                completion_receipt=completion_receipt,
                target=target,
                target_admission=target_admission,
                dispatch_receipts=receipts,
            )

    @_serialized_receiver_transaction
    def dispatch_resolved_target(
        self,
        command: TeleopCommand,
        context: AuthoritativeCommandDispatchContext,
        *,
        target: ResolvedExecutionTarget,
        resolution: MujocoPoseToQposResolution,
        compound_manifest: CompoundDispatchManifest | None = None,
        supersedes_pending_segment_id: str | None = None,
    ) -> Mapping[str, CommandDispatchReceipt]:
        with self._lock:
            self._validate_resolution_lineage(
                command,
                context,
                target,
                resolution,
                compound_manifest=compound_manifest,
            )
            live = self._live_state_provider()
            validation = validate_authoritative_receiver_context(context, live)
            self._validations.append(validation)
            segment = self._build_resolved_segment(
                target,
                resolution,
                context,
                admitted_at=live.captured_at,
            )
            self._segment_attempts.append(segment)
            self._resolved_targets.append(target)

            pending = self._pending_reservation_by_group.get(
                segment.selection_group_id
            )
            active_generation = self._active_generation_by_group.get(
                segment.selection_group_id,
                0,
            )
            relation_kind = "initial"
            predecessor_segment_id: str | None = None
            supersedes_reservation_id: str | None = None
            superseded: tuple[ReferenceSegment, ...] = ()
            relation_rejection: str | None = None
            if pending is not None:
                if supersedes_pending_segment_id is None:
                    relation_rejection = "pending_generation_reservation_exists"
                elif supersedes_pending_segment_id != pending.source_segment_id:
                    relation_rejection = "supersedes_pending_segment_id_mismatch"
                elif live.captured_at + 1e-12 >= pending.effective_from:
                    relation_rejection = "pending_replacement_not_before_start"
                else:
                    relation_kind = "supersedes_pending"
                    supersedes_reservation_id = pending.reservation_id
                    superseded = (self._segment_by_id(pending.source_segment_id),)
            else:
                predecessor = self._active_segment_by_group.get(
                    segment.selection_group_id
                )
                if predecessor is not None:
                    try:
                        validate_reference_segment_successor(predecessor, segment)
                    except ValueError as exc:
                        relation_rejection = f"invalid_successor:{exc}"
                    else:
                        relation_kind = "successor"
                        predecessor_segment_id = predecessor.reference_segment_id

            before_start = live.captured_at + 1e-12 < segment.scheduled_from
            receiver_accepts = bool(
                validation.accepted
                and before_start
                and relation_rejection is None
            )
            rejection_reason = (
                validation.status
                if not validation.accepted
                else relation_rejection
                if relation_rejection is not None
                else SCHEDULED_REFERENCE_LATE_ADMISSION_REJECTION_V1
            )
            target_admission = admit_resolved_execution_target(
                target,
                receiver_id="mujoco-scheduled-resolved-qpos-reference-receiver-v1",
                admitted_at=live.captured_at,
                current_execution_epoch=live.current_execution_epoch,
                expected_group_segment_generation=active_generation + 1,
                receiver_accepts=receiver_accepts,
                rejection_reason=None if receiver_accepts else rejection_reason,
            )
            self._target_admissions.append(target_admission)
            sequence = self._take_sequence()
            segment_admission = admit_reference_segment(
                segment,
                receiver_id="mujoco-scheduled-resolved-qpos-reference-receiver-v1",
                admitted_at=live.captured_at,
                admission_sequence=sequence,
                current_execution_epoch=live.current_execution_epoch,
                expected_group_segment_generation=active_generation + 1,
                achieved_admission_guarantee="endpoint_resolved",
                supersedes_segment_ids=tuple(
                    item.reference_segment_id for item in superseded
                ),
                receiver_accepts=receiver_accepts and target_admission.accepted,
                rejection_reason=(
                    None
                    if receiver_accepts and target_admission.accepted
                    else target_admission.rejection_reason or rejection_reason
                ),
            )
            self._admissions.append(segment_admission)
            self._validation_by_admission_id[
                segment_admission.admission_id
            ] = validation

            if not target_admission.accepted or not segment_admission.accepted:
                status = (
                    target_admission.rejection_reason
                    or segment_admission.rejection_reason
                    or "resolved_target_rejected"
                )
                receipts = self._rejected_receipts(
                    command,
                    status=str(status),
                    timestamp=live.captured_at,
                )
                segment_result = ScheduledReferenceAdmissionResult(
                    context_validation=validation,
                    segment=segment,
                    admission=segment_admission,
                    anchor_candidate=None,
                    terminals=(),
                    dispatch_receipts=receipts,
                )
                self._last_result = segment_result
                self._last_resolved_result = ResolvedQposReferenceAdmissionResult(
                    target=target,
                    target_admission=target_admission,
                    segment_result=segment_result,
                    anchor_commit_evidence=None,
                    compound_manifest=compound_manifest,
                )
                return receipts

            terminals = tuple(
                self._terminal_for_supersession(
                    old,
                    replacement=segment,
                    available_at=live.captured_at,
                )
                for old in superseded
            )
            self._segments.append(segment)
            self._terminals.extend(terminals)
            evidence = execution_target_anchor_commit_evidence(
                target,
                target_admission,
                safety_result_id=context.safety_result_id,
            )
            self._anchor_commit_evidence.append(evidence)
            reservation = build_pending_group_generation_reservation(
                segment,
                segment_admission,
                active_generation=active_generation,
                relation_kind=relation_kind,
                predecessor_segment_id=predecessor_segment_id,
                supersedes_reservation_id=supersedes_reservation_id,
            )
            if pending is not None:
                self._pending_reservation_by_group.pop(
                    segment.selection_group_id,
                    None,
                )
                self._compound_manifest_by_segment_id.pop(
                    pending.source_segment_id,
                    None,
                )
            self._pending_reservation_by_group[
                segment.selection_group_id
            ] = reservation
            self._reservations.append(reservation)
            if compound_manifest is not None:
                self._compound_manifest_by_segment_id[
                    segment.reference_segment_id
                ] = compound_manifest
            try:
                receipts = self._send_scheduled_commands_outside_state_lock(
                    command,
                    activate_at=segment.scheduled_from,
                    reference_segment_id=segment.reference_segment_id,
                    execution_epoch=segment.execution_epoch,
                    group_segment_generation=segment.group_segment_generation,
                )
            except Exception:
                if (
                    self._pending_reservation_by_group.get(
                        segment.selection_group_id
                    )
                    == reservation
                ):
                    self._pending_reservation_by_group.pop(
                        segment.selection_group_id,
                        None,
                    )
                self._compound_manifest_by_segment_id.pop(
                    segment.reference_segment_id,
                    None,
                )
                terminal = build_reference_segment_terminal(
                    segment,
                    status="faulted",
                    available_at=float(self._clock()),
                    terminal_admission_sequence=self._take_sequence(),
                    effective_at=float(self._clock()),
                    evidence_ids=(
                        segment_admission.admission_id,
                        target_admission.admission_id,
                    ),
                )
                self._terminals.append(terminal)
                raise
            segment_result = ScheduledReferenceAdmissionResult(
                context_validation=validation,
                segment=segment,
                admission=segment_admission,
                anchor_candidate=None,
                terminals=terminals,
                dispatch_receipts=receipts,
            )
            self._last_result = segment_result
            self._last_resolved_result = ResolvedQposReferenceAdmissionResult(
                target=target,
                target_admission=target_admission,
                segment_result=segment_result,
                anchor_commit_evidence=evidence,
                compound_manifest=compound_manifest,
            )
            return receipts

    def _build_resolved_segment(
        self,
        target: ResolvedExecutionTarget,
        resolution: MujocoPoseToQposResolution,
        context: AuthoritativeCommandDispatchContext,
        *,
        admitted_at: float,
    ) -> ReferenceSegment:
        scheduled_until = float(target.scheduled_from)
        scheduled_from = scheduled_until - float(
            self.activation_spec.selected_keyframe_period_sec
        )
        start_parent = self._start_parent(
            context,
            scheduled_from=scheduled_from,
            admitted_at=admitted_at,
        )
        end_parent = build_reference_endpoint_parent(
            role="end",
            source_kind="resolved_execution_target",
            source_fact_id=target.target_id,
            projected_group_action_hash=resolution.output_qpos_action_hash,
            action_schema_hash=self.action_schema_hash,
            frame_schema_hash="joint-space-frame-not-applicable-v1",
            target_effective_at=scheduled_until,
        )
        return build_reference_segment(
            control_domain_id=context.control_domain_id,
            selection_group_id="arms",
            execution_epoch=context.execution_epoch,
            group_segment_generation=context.group_segment_generation_by_group["arms"],
            authority_kind="authoritative_shared",
            start_parent=start_parent,
            end_parent=end_parent,
            component_dependencies=("left_arm", "right_arm"),
            created_at=admitted_at,
            clock_domain_id=self.clock_domain_id,
            interpolation_contract_hash="mujoco-resolved-qpos-reference-linear-v1",
            resolver_contract_hash=target.resolver_contract_hash,
            planner_contract_hash=target.planner_contract_hash,
            input_action_schema_hash=self.action_schema_hash,
            output_reference_schema_hash="mujoco-bimanual-qref-v1",
            required_admission_guarantee="endpoint_resolved",
            segment_safety_evidence_level="endpoints_only",
        )

    def _build_component_segment(
        self,
        target: ComponentExecutionTarget,
        resolution: MujocoPoseToQposResolution,
        context: AuthoritativeTargetDispatchContext,
        *,
        admitted_at: float,
    ) -> ReferenceSegment:
        scheduled_until = float(target.scheduled_for)
        scheduled_from = scheduled_until - float(
            self.activation_spec.selected_keyframe_period_sec
        )
        start_parent = self._start_parent(
            context,
            scheduled_from=scheduled_from,
            admitted_at=admitted_at,
        )
        end_parent = build_reference_endpoint_parent(
            role="end",
            source_kind="component_execution_target",
            source_fact_id=target.target_id,
            projected_group_action_hash=(
                resolution.output_qpos_action_hash
            ),
            action_schema_hash=self.action_schema_hash,
            frame_schema_hash="joint-space-frame-not-applicable-v1",
            target_effective_at=scheduled_until,
        )
        return build_reference_segment(
            control_domain_id=context.control_domain_id,
            selection_group_id="arms",
            execution_epoch=context.execution_fence_sequence,
            group_segment_generation=(
                context.group_segment_generation_by_group["arms"]
            ),
            authority_kind="authoritative_shared",
            start_parent=start_parent,
            end_parent=end_parent,
            component_dependencies=("left_arm", "right_arm"),
            created_at=admitted_at,
            clock_domain_id=self.clock_domain_id,
            interpolation_contract_hash=(
                "mujoco-component-qpos-reference-linear-development"
            ),
            resolver_contract_hash=resolution.resolver_contract_hash,
            planner_contract_hash=resolution.resolver_contract_hash,
            input_action_schema_hash=self.action_schema_hash,
            output_reference_schema_hash=self.action_schema_hash,
            required_admission_guarantee="endpoint_resolved",
            segment_safety_evidence_level="endpoints_only",
        )

    def _validate_component_resolution_lineage(
        self,
        command: TeleopCommand,
        context: AuthoritativeTargetDispatchContext,
        target: ComponentExecutionTarget,
        completion_receipt: TargetExecutionCompletionReceipt,
        resolution: MujocoPoseToQposResolution,
    ) -> None:
        if target.target_execution_request_id != (
            context.target_execution_request.request_id
        ) or target.target_execution_request_hash != (
            context.target_execution_request.semantic_content_hash
        ):
            raise ValueError("Component target references another request.")
        if (
            target.submission_receipt_id
            != context.submission_receipt.receipt_id
            or target.submission_receipt_hash
            != context.submission_receipt.semantic_content_hash
        ):
            raise ValueError("Component target references another submission.")
        if (
            completion_receipt.status != "resolved"
            or completion_receipt.target_id != target.target_id
            or completion_receipt.target_hash
            != target.semantic_content_hash
        ):
            raise ValueError(
                "Component target requires its exact resolved completion."
            )
        if target.component_dependencies != context.component_coverage:
            raise ValueError("Component target coverage/context mismatch.")
        if target.output_target_schema_hash != self.action_schema_hash:
            raise ValueError("Component target qpos schema mismatch.")
        if (
            target.execution_fence_sequence
            != context.execution_fence_sequence
            or target.authority_sequence != context.authority_sequence
        ):
            raise ValueError("Component target sequence lineage mismatch.")
        if not math.isclose(
            target.scheduled_for,
            context.target_effective_at,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Component target effective time mismatch.")
        if target.payload.get("command_id") != context.command_id:
            raise ValueError("Component target command lineage mismatch.")
        if target.payload.get("resolution_id") != resolution.resolution_id or (
            target.payload.get("resolution_hash")
            != resolution.semantic_content_hash
        ):
            raise ValueError("Component target references another IK resolution.")
        if target.payload.get("output_qpos_action_hash") != (
            resolution.output_qpos_action_hash
        ):
            raise ValueError("Component target qpos payload hash mismatch.")
        self._require_command_matches_qpos_action(
            command,
            resolution.output_qpos_action,
        )

    def _activate_runtime_groups(
        self,
        *,
        reference_segment_id: str,
        selection_group_id: str,
        group_segment_generation: int,
    ) -> None:
        component_generations = (
            self._component_group_generations_by_segment_id.get(
                str(reference_segment_id)
            )
        )
        if component_generations is not None:
            if selection_group_id != "arms":
                raise RuntimeError(
                    "Component compound activation must be led by arms."
                )
            if component_generations.get("arms") != int(
                group_segment_generation
            ):
                raise RuntimeError(
                    "Component target arms generation mismatch."
                )
            live = self._live_state_provider()
            for group, generation in component_generations.items():
                if (
                    live.expected_next_group_generation_by_group.get(group)
                    != generation
                ):
                    raise RuntimeError(
                        "Component target activation generation is no longer "
                        f"next for {group!r}."
                    )
            if len(component_generations) > 1:
                if self._compound_activation_callback is None:
                    raise RuntimeError(
                        "Multi-group component target requires atomic "
                        "activation callback."
                    )
                self._compound_activation_callback(
                    group_segment_generation_by_group=(
                        component_generations
                    )
                )
            else:
                super()._activate_runtime_groups(
                    reference_segment_id=reference_segment_id,
                    selection_group_id=selection_group_id,
                    group_segment_generation=group_segment_generation,
                )
            target = self._component_target_by_segment_id[
                str(reference_segment_id)
            ]
            admission = self._component_admission_by_segment_id[
                str(reference_segment_id)
            ]
            segment = self._segment_by_id(str(reference_segment_id))
            activated_at = float(self._clock())
            activation = build_component_execution_target_activation_receipt(
                target,
                admission,
                receiver_id=(
                    "mujoco-scheduled-component-qpos-reference-receiver"
                ),
                activation_sequence=self._take_sequence(),
                activation_status="activated",
                activation_at=activated_at,
                available_at=activated_at,
                effective_from=segment.scheduled_from,
                effective_until=segment.scheduled_until,
            )
            self._component_target_activations.append(activation)
            return
        manifest = self._compound_manifest_by_segment_id.get(
            str(reference_segment_id)
        )
        if manifest is None:
            super()._activate_runtime_groups(
                reference_segment_id=reference_segment_id,
                selection_group_id=selection_group_id,
                group_segment_generation=group_segment_generation,
            )
            return
        if selection_group_id != "arms":
            raise RuntimeError("Compound pose activation must be led by arms.")
        generations = dict(manifest.group_segment_generation_by_group)
        if generations.get("arms") != int(group_segment_generation):
            raise RuntimeError("Compound manifest arms generation mismatch.")
        live = self._live_state_provider()
        for group, generation in generations.items():
            if (
                live.expected_next_group_generation_by_group.get(group)
                != generation
            ):
                raise RuntimeError(
                    f"Compound activation generation is no longer next for {group!r}."
                )
        assert self._compound_activation_callback is not None
        self._compound_activation_callback(
            group_segment_generation_by_group=generations
        )

    def _require_successor_predecessor_evidence(
        self,
        predecessor: ReferenceSegment,
        *,
        activated_at: float,
    ) -> None:
        """Component targets use activation receipts, not legacy safety candidates."""

        component_result = self._component_result_by_segment_id.get(
            predecessor.reference_segment_id
        )
        if component_result is None:
            super()._require_successor_predecessor_evidence(
                predecessor,
                activated_at=activated_at,
            )
            return
        predecessor_activation = next(
            (
                item
                for item in reversed(self._component_target_activations)
                if item.target_id == component_result.target.target_id
                and item.activation_status == "activated"
            ),
            None,
        )
        if predecessor_activation is None:
            raise RuntimeError(
                "Component successor activation requires the predecessor "
                "target activation receipt."
            )

    @_serialized_receiver_transaction
    def cancel_all(self, *, reason: str, effective_at: float) -> None:
        with self._lock:
            activated_target_ids = {
                item.target_id
                for item in self._component_target_activations
            }
            for segment_id, target in tuple(
                self._component_target_by_segment_id.items()
            ):
                if target.target_id in activated_target_ids:
                    continue
                admission = self._component_admission_by_segment_id.get(
                    segment_id
                )
                if admission is None or not admission.accepted:
                    continue
                rejected_at = max(
                    float(effective_at),
                    admission.admitted_at,
                )
                self._component_target_activations.append(
                    build_component_execution_target_activation_receipt(
                        target,
                        admission,
                        receiver_id=(
                            "mujoco-scheduled-component-qpos-reference-receiver"
                        ),
                        activation_sequence=self._take_sequence(),
                        activation_status="rejected",
                        activation_at=rejected_at,
                        available_at=rejected_at,
                        rejection_code=(
                            f"cancelled_before_activation:{str(reason)}"
                        ),
                    )
                )
        super().cancel_all(reason=reason, effective_at=effective_at)
        with self._lock:
            self._compound_manifest_by_segment_id.clear()
            self._component_group_generations_by_segment_id.clear()
            self._component_target_by_segment_id.clear()
            self._component_admission_by_segment_id.clear()

    def _validate_resolution_lineage(
        self,
        command: TeleopCommand,
        context: AuthoritativeCommandDispatchContext,
        target: ResolvedExecutionTarget,
        resolution: MujocoPoseToQposResolution,
        *,
        compound_manifest: CompoundDispatchManifest | None,
    ) -> None:
        if target.source_safety_result_id != context.safety_result_id:
            raise ValueError("Resolved target references another safety result.")
        context_anchor_hash = safety_anchor_action_hash(context.safe_action)
        if target.source_safe_action_hash != context_anchor_hash:
            raise ValueError("Resolved target references another shared-safe action.")
        if target.input_action_schema_hash != self.pose_action_schema_hash:
            raise ValueError("Resolved target pose action schema mismatch.")
        if target.output_target_schema_hash != self.action_schema_hash:
            raise ValueError("Resolved target qpos action schema mismatch.")
        if target.execution_epoch != context.execution_epoch or (
            target.group_segment_generation
            != context.group_segment_generation_by_group.get("arms")
        ):
            raise ValueError("Resolved target epoch/generation lineage mismatch.")
        context_groups = set(context.group_segment_generation_by_group)
        if context_groups == {"arms"}:
            if compound_manifest is not None:
                raise ValueError(
                    "Arms-only resolved dispatch cannot carry a compound manifest."
                )
        else:
            if compound_manifest is None:
                raise ValueError(
                    "Multi-group resolved dispatch requires a compound manifest."
                )
            if self._compound_activation_callback is None:
                raise RuntimeError(
                    "Compound dispatch requires an atomic activation callback."
                )
            validate_compound_dispatch_manifest(compound_manifest, context)
            if target.payload.get("compound_dispatch_manifest_id") != (
                compound_manifest.manifest_id
            ) or target.payload.get("compound_dispatch_manifest_hash") != (
                compound_manifest.semantic_content_hash
            ):
                raise ValueError(
                    "Resolved target references another compound manifest."
                )
        if not math.isclose(
            target.scheduled_from,
            context.target_effective_at,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Resolved target effective time mismatch.")
        if target.payload.get("resolution_id") != resolution.resolution_id or (
            target.payload.get("resolution_hash")
            != resolution.semantic_content_hash
        ):
            raise ValueError("Resolved target references another IK resolution.")
        arm_output = project_robot_action_to_components(
            resolution.output_qpos_action,
            ("left_arm", "right_arm"),
        )
        if target.payload.get("output_qpos_action_hash") != (
            safety_anchor_action_hash(arm_output)
        ):
            raise ValueError("Resolved target qpos payload hash mismatch.")
        if resolution.output_qpos_action.hand_qpos_by_side:
            if target.payload.get("compound_output_qpos_action_hash") != (
                resolution.output_qpos_action_hash
            ):
                raise ValueError(
                    "Resolved target compound qpos payload hash mismatch."
                )
        elif "compound_output_qpos_action_hash" in target.payload:
            raise ValueError("Arms-only resolved target has compound qpos payload.")
        if resolution.input_safe_action_hash != context_anchor_hash:
            raise ValueError("IK resolution references another safe action.")
        self._require_command_matches_qpos_action(
            command,
            resolution.output_qpos_action,
        )

    @staticmethod
    def _require_command_matches_qpos_action(
        command: TeleopCommand,
        qpos_action,
    ) -> None:
        if set(qpos_action.arm_qpos_by_side) != {"left", "right"}:
            raise ValueError("Resolved qpos action must contain both arms.")
        for side in ("left", "right"):
            effector = command.commands.get(f"{side}_arm")
            if effector is None or effector.command_type != "joint_vector":
                raise ValueError("Resolved receiver requires coupled joint commands.")
            command_q = np.asarray(effector.value, dtype=float).reshape(-1)
            target_q = np.asarray(qpos_action.arm_qpos_by_side[side], dtype=float)
            if command_q.shape != (6,) or not np.array_equal(command_q, target_q):
                raise ValueError("Command does not match resolved qpos target.")
        expected_hands = set(qpos_action.hand_qpos_by_side)
        present_hands = {
            side
            for side in ("left", "right")
            if command.commands.get(f"{side}_hand") is not None
        }
        if present_hands != expected_hands:
            raise ValueError("Command hand components do not match resolved qpos target.")
        if expected_hands:
            hand_specs = hand_action_specs_from_metadata(qpos_action.metadata)
            if set(hand_specs) != expected_hands:
                raise ValueError("Resolved qpos target lacks exact hand specs.")
            for side in sorted(expected_hands):
                effector = command.commands[f"{side}_hand"]
                if effector.command_type != "joint_vector":
                    raise ValueError("Resolved hand command must be a joint vector.")
                expected = hand_specs[side].active_values(
                    qpos_action.hand_qpos_by_side[side]
                )
                actual = np.asarray(effector.value, dtype=np.float32).reshape(-1)
                if not np.array_equal(actual, expected):
                    raise ValueError(
                        "Command does not match resolved hand qpos target."
                    )


__all__ = [
    "ComponentQposReferenceAdmissionResult",
    "MujocoScheduledResolvedQposReferenceReceiver",
    "ResolvedQposReferenceAdmissionResult",
]
