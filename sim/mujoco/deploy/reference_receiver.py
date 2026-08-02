"""Scoped authoritative qpos segment receiver for deployable MuJoCo.

This is the M5.7e2-c2 vertical slice.  It owns a receiver-side immutable
segment timeline and performs a second live context validation before segment
admission.  It intentionally supports only one coupled bimanual-qpos group.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Callable, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    AuthoritativeCommandDispatchContext,
    AuthoritativeTargetDispatchContext,
    AuthoritativeReceiverStateSnapshot,
    AuthoritativeReceiverValidationReceipt,
    GroupGenerationActivation,
    PendingGroupGenerationReservation,
    ReferenceActivationSpec,
    ReferenceSegment,
    ReferenceSegmentAdmissionReceipt,
    ReferenceSegmentTerminal,
    RobotAction,
    SafetyAnchorCommitCandidate,
    activate_group_generation_reservation,
    admit_reference_segment,
    build_pending_group_generation_reservation,
    build_reference_endpoint_parent,
    build_reference_segment,
    build_reference_segment_terminal,
    build_safety_anchor_commit_candidate,
    safety_anchor_action_hash,
    validate_authoritative_receiver_context,
    validate_reference_segment_successor,
)
from robot_motion.runtime import CommandDispatchReceipt, TeleopCommand

from ..adapters import MujocoPositionExecutionRig


MEASURED_REBASE_HASH_DOMAIN_V1 = "dexteleop.mujoco.measured_rebase.v1"


@dataclass(frozen=True)
class ScheduledReferenceAdmissionResult:
    context_validation: AuthoritativeReceiverValidationReceipt
    segment: ReferenceSegment | None
    admission: ReferenceSegmentAdmissionReceipt | None
    anchor_candidate: SafetyAnchorCommitCandidate | None
    terminals: tuple[ReferenceSegmentTerminal, ...]
    dispatch_receipts: Mapping[str, CommandDispatchReceipt]


class MujocoScheduledQposReferenceReceiver:
    """Receiver timeline plus delayed qref activation for one sim domain."""

    def __init__(
        self,
        rig: MujocoPositionExecutionRig,
        *,
        activation_spec: ReferenceActivationSpec,
        action_schema_hash: str,
        clock_domain_id: str,
        live_state_provider: Callable[[], AuthoritativeReceiverStateSnapshot],
        activation_callback: Callable[[str, int], object],
        clock: Callable[[], float],
    ) -> None:
        if activation_spec.activation_mode != "scheduled_segment":
            raise ValueError("Scheduled receiver requires a scheduled activation spec.")
        if (
            not callable(live_state_provider)
            or not callable(activation_callback)
            or not callable(clock)
        ):
            raise TypeError(
                "live_state_provider, activation_callback, and clock must be callable."
            )
        self.rig = rig
        self.activation_spec = activation_spec
        self.action_schema_hash = str(action_schema_hash).strip()
        self.clock_domain_id = str(clock_domain_id).strip()
        if not self.action_schema_hash or not self.clock_domain_id:
            raise ValueError("Receiver schema and clock-domain IDs must be non-empty.")
        self._live_state_provider = live_state_provider
        self._activation_callback = activation_callback
        self._clock = clock
        self._lock = threading.RLock()
        self._next_timeline_sequence = 1
        self._validations: list[AuthoritativeReceiverValidationReceipt] = []
        self._segment_attempts: list[ReferenceSegment] = []
        self._segments: list[ReferenceSegment] = []
        self._admissions: list[ReferenceSegmentAdmissionReceipt] = []
        self._validation_by_admission_id: dict[
            str, AuthoritativeReceiverValidationReceipt
        ] = {}
        self._anchor_candidates: list[SafetyAnchorCommitCandidate] = []
        self._reservations: list[PendingGroupGenerationReservation] = []
        self._activations: list[GroupGenerationActivation] = []
        self._pending_reservation_by_group: dict[
            str, PendingGroupGenerationReservation
        ] = {}
        self._active_generation_by_group: dict[str, int] = {"arms": 0}
        self._active_segment_by_group: dict[str, ReferenceSegment] = {}
        self._terminals: list[ReferenceSegmentTerminal] = []
        self._last_result: ScheduledReferenceAdmissionResult | None = None
        self.rig.bind_scheduled_generation_guards(
            activation_guard=self._activate_reserved_generation,
            active_work_guard=self._require_active_generation,
        )

    @property
    def validations(self) -> tuple[AuthoritativeReceiverValidationReceipt, ...]:
        with self._lock:
            return tuple(self._validations)

    @property
    def segment_attempts(self) -> tuple[ReferenceSegment, ...]:
        with self._lock:
            return tuple(self._segment_attempts)

    @property
    def segments(self) -> tuple[ReferenceSegment, ...]:
        with self._lock:
            return tuple(self._segments)

    @property
    def admissions(self) -> tuple[ReferenceSegmentAdmissionReceipt, ...]:
        with self._lock:
            return tuple(self._admissions)

    def validation_for_admission(
        self,
        admission_id: str,
    ) -> AuthoritativeReceiverValidationReceipt:
        with self._lock:
            return self._validation_by_admission_id[str(admission_id)]

    @property
    def anchor_candidates(self) -> tuple[SafetyAnchorCommitCandidate, ...]:
        with self._lock:
            return tuple(self._anchor_candidates)

    @property
    def generation_reservations(
        self,
    ) -> tuple[PendingGroupGenerationReservation, ...]:
        with self._lock:
            return tuple(self._reservations)

    @property
    def generation_activations(self) -> tuple[GroupGenerationActivation, ...]:
        with self._lock:
            return tuple(self._activations)

    @property
    def terminals(self) -> tuple[ReferenceSegmentTerminal, ...]:
        with self._lock:
            return tuple(self._terminals)

    @property
    def last_result(self) -> ScheduledReferenceAdmissionResult | None:
        with self._lock:
            return self._last_result

    def dispatch(
        self,
        command: TeleopCommand,
        context: AuthoritativeCommandDispatchContext,
        *,
        supersedes_pending_segment_id: str | None = None,
    ) -> Mapping[str, CommandDispatchReceipt]:
        """Validate, admit, then queue one future qref while holding receiver state."""

        with self._lock:
            live = self._live_state_provider()
            validation = validate_authoritative_receiver_context(context, live)
            self._validations.append(validation)
            if not validation.accepted:
                receipts = self._rejected_receipts(
                    command,
                    status=validation.status,
                    timestamp=live.captured_at,
                )
                self._last_result = ScheduledReferenceAdmissionResult(
                    context_validation=validation,
                    segment=None,
                    admission=None,
                    anchor_candidate=None,
                    terminals=(),
                    dispatch_receipts=receipts,
                )
                return receipts

            segment = self._build_segment(command, context, admitted_at=live.captured_at)
            self._segment_attempts.append(segment)
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
                        validate_reference_segment_successor(predecessor, segment)
                    except ValueError as exc:
                        relation_rejection = f"invalid_successor:{exc}"
                    else:
                        relation_kind = "successor"
                        predecessor_segment_id = predecessor.reference_segment_id
            sequence = self._take_sequence()
            before_start = live.captured_at + 1e-12 < segment.scheduled_from
            admission = admit_reference_segment(
                segment,
                receiver_id="mujoco-scheduled-qpos-reference-receiver-v1",
                admitted_at=live.captured_at,
                admission_sequence=sequence,
                current_execution_epoch=live.current_execution_epoch,
                expected_group_segment_generation=active_generation + 1,
                achieved_admission_guarantee="interpolation_segment_accepted",
                supersedes_segment_ids=tuple(
                    item.reference_segment_id for item in superseded
                ),
                receiver_accepts=before_start and relation_rejection is None,
                rejection_reason=(
                    relation_rejection
                    if relation_rejection is not None
                    else None
                    if before_start
                    else "admission_deadline_missed_before_start"
                ),
            )
            self._admissions.append(admission)
            self._validation_by_admission_id[admission.admission_id] = validation
            if not admission.accepted:
                receipts = self._rejected_receipts(
                    command,
                    status=str(admission.rejection_reason),
                    timestamp=live.captured_at,
                )
                self._last_result = ScheduledReferenceAdmissionResult(
                    context_validation=validation,
                    segment=segment,
                    admission=admission,
                    anchor_candidate=None,
                    terminals=(),
                    dispatch_receipts=receipts,
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
            candidate = build_safety_anchor_commit_candidate(
                segment,
                admission,
                safety_result_id=context.safety_result_id,
                safe_action=context.safe_action,
            )
            self._anchor_candidates.append(candidate)
            reservation = build_pending_group_generation_reservation(
                segment,
                admission,
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
            self._pending_reservation_by_group[
                segment.selection_group_id
            ] = reservation
            self._reservations.append(reservation)
            try:
                receipts = self.rig.send_scheduled_commands(
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
                terminal = build_reference_segment_terminal(
                    segment,
                    status="faulted",
                    available_at=float(self._clock()),
                    terminal_admission_sequence=self._take_sequence(),
                    effective_at=float(self._clock()),
                    evidence_ids=(admission.admission_id,),
                )
                self._terminals.append(terminal)
                raise
            self._last_result = ScheduledReferenceAdmissionResult(
                context_validation=validation,
                segment=segment,
                admission=admission,
                anchor_candidate=candidate,
                terminals=terminals,
                dispatch_receipts=receipts,
            )
            return receipts

    def cancel_all(self, *, reason: str, effective_at: float) -> None:
        """Invalidate every nonterminal segment before the rig enters HOLD."""

        with self._lock:
            terminal_ids = {item.reference_segment_id for item in self._terminals}
            for segment in self._segments:
                if segment.reference_segment_id in terminal_ids:
                    continue
                terminal = build_reference_segment_terminal(
                    segment,
                    status="canceled",
                    available_at=effective_at,
                    terminal_admission_sequence=self._take_sequence(),
                    effective_at=effective_at,
                    last_reference_frame_id=self._last_frame_id(
                        segment.reference_segment_id
                    ),
                    evidence_ids=(str(reason),),
                )
                self._terminals.append(terminal)
            self._pending_reservation_by_group.clear()
            self._active_segment_by_group.clear()
            self._active_generation_by_group = {"arms": 0}

    def _build_segment(
        self,
        command: TeleopCommand,
        context: AuthoritativeCommandDispatchContext,
        *,
        admitted_at: float,
    ) -> ReferenceSegment:
        if set(context.group_segment_generation_by_group) != {"arms"}:
            raise ValueError("Scheduled qpos receiver v1 only supports the arms group.")
        if set(context.safe_action.arm_qpos_by_side) != {"left", "right"}:
            raise ValueError("Scheduled qpos receiver requires bimanual safe qpos.")
        self._require_command_matches_safe_action(command, context.safe_action)
        scheduled_until = float(context.target_effective_at)
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
            source_kind="current_safe_target",
            source_fact_id=context.safety_result_id,
            projected_group_action_hash=safety_anchor_action_hash(
                context.safe_action
            ),
            action_schema_hash=self.action_schema_hash,
            frame_schema_hash="joint-space-frame-not-applicable-v1",
            target_effective_at=scheduled_until,
        )
        return build_reference_segment(
            control_domain_id=context.control_domain_id,
            selection_group_id="arms",
            execution_epoch=context.execution_epoch,
            group_segment_generation=context.group_segment_generation_by_group[
                "arms"
            ],
            authority_kind="authoritative_shared",
            start_parent=start_parent,
            end_parent=end_parent,
            component_dependencies=("left_arm", "right_arm"),
            created_at=admitted_at,
            clock_domain_id=self.clock_domain_id,
            interpolation_contract_hash="mujoco-qpos-reference-linear-v1",
            resolver_contract_hash="robot-action-resolver-bimanual-qpos-v1",
            input_action_schema_hash=self.action_schema_hash,
            output_reference_schema_hash="mujoco-bimanual-qref-v1",
            required_admission_guarantee="interpolation_segment_accepted",
            segment_safety_evidence_level="endpoints_only",
        )

    def _start_parent(
        self,
        context: (
            AuthoritativeCommandDispatchContext
            | AuthoritativeTargetDispatchContext
        ),
        *,
        scheduled_from: float,
        admitted_at: float,
    ):
        execution_sequence = (
            context.execution_epoch
            if isinstance(context, AuthoritativeCommandDispatchContext)
            else context.execution_fence_sequence
        )
        live = [
            item
            for item in self._segments
            if item.execution_epoch == execution_sequence
            and item.reference_segment_id
            not in {terminal.reference_segment_id for terminal in self._terminals}
        ]
        if live:
            previous = max(
                live,
                key=lambda item: (
                    item.scheduled_until,
                    item.group_segment_generation,
                    item.reference_segment_id,
                ),
            )
            if math.isclose(
                previous.scheduled_until,
                scheduled_from,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                return build_reference_endpoint_parent(
                    role="start",
                    source_kind="committed_safe_endpoint",
                    source_fact_id=previous.reference_segment_id,
                    projected_group_action_hash=(
                        previous.end_parent.projected_group_action_hash
                    ),
                    action_schema_hash=self.action_schema_hash,
                    frame_schema_hash="joint-space-frame-not-applicable-v1",
                    target_effective_at=scheduled_from,
                )
            if math.isclose(
                previous.scheduled_from,
                scheduled_from,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                return build_reference_endpoint_parent(
                    role="start",
                    source_kind=previous.start_parent.source_kind,
                    source_fact_id=previous.start_parent.source_fact_id,
                    projected_group_action_hash=(
                        previous.start_parent.projected_group_action_hash
                    ),
                    action_schema_hash=self.action_schema_hash,
                    frame_schema_hash="joint-space-frame-not-applicable-v1",
                    target_effective_at=scheduled_from,
                )
            raise ValueError(
                "Scheduled reference segment is not contiguous or replacing: "
                f"previous=[{previous.scheduled_from:.12f},"
                f"{previous.scheduled_until:.12f}], "
                f"requested_from={scheduled_from:.12f}."
            )

        measured = RobotAction.bimanual_qpos(
            left_arm=self.rig.accepted_qcmd_by_side()["left"],
            right_arm=self.rig.accepted_qcmd_by_side()["right"],
            timestamp=admitted_at,
            source="mujoco-measured-rebase",
        )
        measured_hash = safety_anchor_action_hash(measured)
        evidence_hash = canonical_content_hash(
            {
                "control_domain_id": context.control_domain_id,
                "execution_epoch": execution_sequence,
                "captured_at": admitted_at,
                "measured_action_hash": measured_hash,
            },
            domain=MEASURED_REBASE_HASH_DOMAIN_V1,
        )
        return build_reference_endpoint_parent(
            role="start",
            source_kind="measured_rebase",
            source_fact_id=f"mujoco-measured-rebase:{evidence_hash}",
            projected_group_action_hash=measured_hash,
            action_schema_hash=self.action_schema_hash,
            frame_schema_hash="joint-space-frame-not-applicable-v1",
            target_effective_at=scheduled_from,
        )

    def _activate_reserved_generation(
        self,
        reference_segment_id: str,
        execution_epoch: int,
        group_segment_generation: int,
        activated_at: float,
    ) -> str:
        with self._lock:
            group = "arms"
            reservation = self._pending_reservation_by_group.get(group)
            if reservation is None:
                raise RuntimeError("Scheduled activation has no pending reservation.")
            if reservation.source_segment_id != str(reference_segment_id):
                raise RuntimeError("Scheduled activation references another reservation.")
            activation = activate_group_generation_reservation(
                reservation,
                current_execution_epoch=int(execution_epoch),
                current_active_generation=self._active_generation_by_group[group],
                activated_at=float(activated_at),
                activation_sequence=self._take_sequence(),
            )
            segment = self._segment_by_id(reservation.source_segment_id)
            if reservation.relation_kind == "successor":
                predecessor = self._active_segment_by_group.get(group)
                if predecessor is None:
                    raise RuntimeError("Successor activation has no active predecessor.")
                self._require_successor_predecessor_evidence(
                    predecessor,
                    activated_at=float(activated_at),
                )
                predecessor_terminal = build_reference_segment_terminal(
                    predecessor,
                    status="retired_by_successor_activation",
                    available_at=float(activated_at),
                    terminal_admission_sequence=self._take_sequence(),
                    effective_at=float(activated_at),
                    last_reference_frame_id=self._last_frame_id(
                        predecessor.reference_segment_id
                    ),
                    evidence_ids=(segment.reference_segment_id,),
                )
            else:
                predecessor_terminal = None
            live = self._live_state_provider()
            if live.current_execution_epoch != reservation.execution_epoch:
                raise RuntimeError("Scheduled activation execution epoch is stale.")
            if (
                live.expected_next_group_generation_by_group.get(group)
                != reservation.reserved_generation
            ):
                raise RuntimeError("Scheduled activation generation is no longer next.")
            self._activate_runtime_groups(
                reference_segment_id=str(reference_segment_id),
                selection_group_id=group,
                group_segment_generation=int(group_segment_generation),
            )
            self._active_generation_by_group[group] = activation.active_generation
            self._active_segment_by_group[group] = segment
            self._pending_reservation_by_group.pop(group, None)
            self._activations.append(activation)
            if predecessor_terminal is not None:
                self._terminals.append(predecessor_terminal)
            return activation.activation_id

    def _require_successor_predecessor_evidence(
        self,
        predecessor: ReferenceSegment,
        *,
        activated_at: float,
    ) -> None:
        """Require the legacy safe-endpoint evidence before successor activation."""

        predecessor_candidate = next(
            (
                item
                for item in self._anchor_candidates
                if item.reference_segment_id
                == predecessor.reference_segment_id
            ),
            None,
        )
        if (
            predecessor_candidate is None
            or predecessor_candidate.effective_from > float(activated_at) + 1e-12
        ):
            raise RuntimeError(
                "Successor activation requires an effective "
                "predecessor END candidate: "
                f"candidate_effective_from="
                f"{None if predecessor_candidate is None else predecessor_candidate.effective_from}, "
                f"activated_at={float(activated_at):.12f}."
            )

    def _activate_runtime_groups(
        self,
        *,
        reference_segment_id: str,
        selection_group_id: str,
        group_segment_generation: int,
    ) -> None:
        """Activation hook; compound receivers may promote multiple groups."""

        del reference_segment_id
        self._activation_callback(
            selection_group_id,
            int(group_segment_generation),
        )

    def _require_active_generation(
        self,
        reference_segment_id: str,
        execution_epoch: int,
        group_segment_generation: int,
        _target_time: float,
    ) -> None:
        with self._lock:
            active = self._active_segment_by_group.get("arms")
            if (
                active is None
                or active.reference_segment_id != str(reference_segment_id)
                or active.execution_epoch != int(execution_epoch)
                or self._active_generation_by_group["arms"]
                != int(group_segment_generation)
            ):
                raise RuntimeError("Scheduled work generation is not active.")

    def _segment_by_id(self, reference_segment_id: str) -> ReferenceSegment:
        for segment in reversed(self._segments):
            if segment.reference_segment_id == reference_segment_id:
                return segment
        raise RuntimeError("Reference segment is not admitted in this receiver.")

    def _overlapping_live_segments(
        self,
        segment: ReferenceSegment,
    ) -> tuple[ReferenceSegment, ...]:
        terminal_ids = {item.reference_segment_id for item in self._terminals}
        return tuple(
            item
            for item in self._segments
            if item.reference_segment_id not in terminal_ids
            and item.execution_epoch == segment.execution_epoch
            and item.selection_group_id == segment.selection_group_id
            and item.scheduled_from < segment.scheduled_until
            and segment.scheduled_from < item.scheduled_until
        )

    def _terminal_for_supersession(
        self,
        old: ReferenceSegment,
        *,
        replacement: ReferenceSegment,
        available_at: float,
    ) -> ReferenceSegmentTerminal:
        return build_reference_segment_terminal(
            old,
            status="superseded",
            available_at=available_at,
            terminal_admission_sequence=self._take_sequence(),
            effective_at=available_at,
            last_reference_frame_id=self._last_frame_id(
                old.reference_segment_id
            ),
            replacement_segment_id=replacement.reference_segment_id,
            evidence_ids=(replacement.reference_segment_id,),
        )

    def _last_frame_id(self, reference_segment_id: str) -> str | None:
        for frame in reversed(self.rig.planner_command_frame_observations):
            if frame.reference_segment_id == reference_segment_id:
                return frame.observation_id
        return None

    @staticmethod
    def _require_command_matches_safe_action(
        command: TeleopCommand,
        safe_action: RobotAction,
    ) -> None:
        for side in ("left", "right"):
            effector = command.commands.get(f"{side}_arm")
            if effector is None or effector.command_type != "joint_vector":
                raise ValueError("Scheduled receiver requires coupled arm joint commands.")
            command_q = np.asarray(effector.value, dtype=float).reshape(-1)
            safe_q = np.asarray(safe_action.arm_qpos_by_side[side], dtype=float)
            if command_q.shape != (6,) or not np.array_equal(command_q, safe_q):
                raise ValueError("Resolved qref does not match shared-safe qpos.")

    @staticmethod
    def _rejected_receipts(
        command: TeleopCommand,
        *,
        status: str,
        timestamp: float,
    ) -> Mapping[str, CommandDispatchReceipt]:
        return {
            name: CommandDispatchReceipt(
                effector_name=name,
                accepted=False,
                operation="admit_mujoco_scheduled_reference_segment",
                status=status,
                timestamp=timestamp,
            )
            for name in command.commands
        }

    def _take_sequence(self) -> int:
        result = self._next_timeline_sequence
        self._next_timeline_sequence += 1
        return result


__all__ = [
    "MujocoScheduledQposReferenceReceiver",
    "ScheduledReferenceAdmissionResult",
]
