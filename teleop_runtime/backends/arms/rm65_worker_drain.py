"""Pure worker-drain plan for the first arms-only RM65 execution profile.

This module identifies the concrete asynchronous participants that must prove
an old execution epoch can no longer reach an RM65 SDK call.  It does not
attach callbacks, stop threads, inspect live workers, or manufacture drain
receipts.  Those are M5.8-b integration facts.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.execution_manifest import (
    M6_CANDIDATE_POLICY_ARM_QPOS_V1,
    ControlExecutionManifest,
)
from robot_motion.runtime import (
    DispatchSeamInventoryManifest,
    ExecutionCancellation,
    ExecutionCancellationParticipantSpec,
    WorkerEpochDrainAggregate,
    WorkerEpochDrainReceipt,
    aggregate_worker_epoch_drains,
)


RM65_ARMS_ONLY_WORKER_DRAIN_PLAN_V1 = (
    "m6_candidate_policy_arm_qpos_v1.rm65_worker_drain_plan_v1"
)
RM65_ARMS_ONLY_WORKER_DRAIN_PLAN_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_arms_only_worker_drain_plan.v1"
)


@dataclass(frozen=True)
class RM65WorkerDrainParticipantBinding:
    participant: ExecutionCancellationParticipantSpec
    implementation_path: str
    related_dispatch_seam_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "participant_id": self.participant.participant_id,
            "participant_spec_hash": self.participant.semantic_content_hash,
            "implementation_path": self.implementation_path,
            "related_dispatch_seam_ids": self.related_dispatch_seam_ids,
        }


@dataclass(frozen=True)
class RM65ArmsOnlyWorkerDrainPlan:
    plan_id: str
    execution_manifest_id: str
    execution_manifest_hash: str
    dispatch_inventory_id: str
    dispatch_inventory_hash: str
    participant_bindings: tuple[RM65WorkerDrainParticipantBinding, ...]
    integration_status: str
    semantic_content_hash: str

    @property
    def participants(self) -> tuple[ExecutionCancellationParticipantSpec, ...]:
        return tuple(binding.participant for binding in self.participant_bindings)


class RM65ExecutionDrainParticipant(Protocol):
    def drain_execution_cancellation(
        self,
        cancellation: ExecutionCancellation,
    ) -> WorkerEpochDrainReceipt: ...


@dataclass(frozen=True)
class RM65ArmsOnlyWorkerDrainResult:
    plan_id: str
    cancellation_id: str
    receipts: tuple[WorkerEpochDrainReceipt, ...]
    aggregate: WorkerEpochDrainAggregate


def drain_rm65_arms_only_workers(
    *,
    plan: RM65ArmsOnlyWorkerDrainPlan,
    cancellation: ExecutionCancellation,
    position_reference_worker: RM65ExecutionDrainParticipant,
    left_background_worker: RM65ExecutionDrainParticipant,
    right_background_worker: RM65ExecutionDrainParticipant,
    clock=time.monotonic,
) -> RM65ArmsOnlyWorkerDrainResult:
    """Drain upstream first, then both final background senders.

    This coordinator never calls HOLD or an SDK.  It is intentionally not
    wired into a production frontend yet; callers must already have advanced
    the shared execution epoch and separately coordinate the required stop.
    """

    if not isinstance(plan, RM65ArmsOnlyWorkerDrainPlan):
        raise TypeError("plan must be an RM65ArmsOnlyWorkerDrainPlan")
    if not isinstance(cancellation, ExecutionCancellation):
        raise TypeError("cancellation must be an ExecutionCancellation")
    if not callable(clock):
        raise TypeError("clock must be callable")
    by_id = {
        "position-deploy-reference-worker": position_reference_worker,
        "rm65-left-background-stream-worker": left_background_worker,
        "rm65-right-background-stream-worker": right_background_worker,
    }
    required_ids = tuple(
        binding.participant.participant_id for binding in plan.participant_bindings
    )
    if set(required_ids) != set(by_id):
        raise RuntimeError("worker drain plan does not match the arms-only coordinator")

    # Upstream first is a safety property: once it acknowledges, no old
    # reference callback can enqueue another target behind a drained backend.
    ordered_ids = (
        "position-deploy-reference-worker",
        "rm65-left-background-stream-worker",
        "rm65-right-background-stream-worker",
    )
    receipts: list[WorkerEpochDrainReceipt] = []
    for participant_id in ordered_ids:
        worker = by_id[participant_id]
        drain = getattr(worker, "drain_execution_cancellation", None)
        if not callable(drain):
            raise TypeError(f"{participant_id} does not implement execution drain")
        receipt = drain(cancellation)
        if not isinstance(receipt, WorkerEpochDrainReceipt):
            raise TypeError(f"{participant_id} returned an invalid drain receipt")
        if receipt.participant_id != participant_id:
            raise RuntimeError(
                f"{participant_id} returned receipt for {receipt.participant_id}"
            )
        receipts.append(receipt)
    aggregated_at = max(float(clock()), cancellation.effective_at)
    aggregate = aggregate_worker_epoch_drains(
        cancellation=cancellation,
        participants=plan.participants,
        receipts=tuple(receipts),
        aggregated_at=aggregated_at,
    )
    return RM65ArmsOnlyWorkerDrainResult(
        plan_id=plan.plan_id,
        cancellation_id=cancellation.cancellation_id,
        receipts=tuple(receipts),
        aggregate=aggregate,
    )


def build_rm65_arms_only_worker_drain_plan(
    *,
    manifest: ControlExecutionManifest,
    inventory: DispatchSeamInventoryManifest,
) -> RM65ArmsOnlyWorkerDrainPlan:
    """Bind exact candidate workers without claiming they emit acknowledgements."""

    if not isinstance(manifest, ControlExecutionManifest):
        raise TypeError("manifest must be a ControlExecutionManifest.")
    if not isinstance(inventory, DispatchSeamInventoryManifest):
        raise TypeError("inventory must be a DispatchSeamInventoryManifest.")
    if manifest.manifest_id != M6_CANDIDATE_POLICY_ARM_QPOS_V1:
        raise ValueError("Worker-drain plan requires the first M6 candidate manifest.")
    if (
        inventory.execution_manifest_id != manifest.manifest_id
        or inventory.execution_manifest_hash != manifest.semantic_content_hash
    ):
        raise ValueError("Dispatch inventory is bound to another execution manifest.")

    inventory_seams = {entry.seam_id for entry in inventory.entries}

    def binding(
        *,
        participant_id: str,
        worker_kind: str,
        implementation_path: str,
        related_dispatch_seam_ids: tuple[str, ...],
    ) -> RM65WorkerDrainParticipantBinding:
        missing = set(related_dispatch_seam_ids).difference(inventory_seams)
        if missing:
            raise ValueError(
                "Worker-drain participant references uninventoried seams: "
                f"{sorted(missing)!r}."
            )
        return RM65WorkerDrainParticipantBinding(
            participant=ExecutionCancellationParticipantSpec.create(
                participant_id=participant_id,
                worker_kind=worker_kind,
                affected_selection_group_ids=("arms",),
            ),
            implementation_path=implementation_path,
            related_dispatch_seam_ids=tuple(sorted(related_dispatch_seam_ids)),
        )

    bindings = (
        binding(
            participant_id="position-deploy-reference-worker",
            worker_kind="reference_worker",
            implementation_path=(
                "teleop_runtime.position_control.deploy_executor."
                "PositionDeployExecutor._run"
            ),
            related_dispatch_seam_ids=(
                "rm65.motion.target_admission.send_command",
                "rm65.safety.position_stream_disengage.stop",
            ),
        ),
        binding(
            participant_id="rm65-left-background-stream-worker",
            worker_kind="backend_stream_worker",
            implementation_path=(
                "teleop_runtime.backends.arms.rm65."
                "RM65ArmBackend._background_stream_loop_impl[left]"
            ),
            related_dispatch_seam_ids=(
                "rm65.motion.background.worker_dequeue",
                "rm65.motion.pre_sdk.send_joint_target",
                "rm65.safety.quick_stop.tick.pre_sdk",
            ),
        ),
        binding(
            participant_id="rm65-right-background-stream-worker",
            worker_kind="backend_stream_worker",
            implementation_path=(
                "teleop_runtime.backends.arms.rm65."
                "RM65ArmBackend._background_stream_loop_impl[right]"
            ),
            related_dispatch_seam_ids=(
                "rm65.motion.background.worker_dequeue",
                "rm65.motion.pre_sdk.send_joint_target",
                "rm65.safety.quick_stop.tick.pre_sdk",
            ),
        ),
    )
    bindings = tuple(
        sorted(bindings, key=lambda item: item.participant.participant_id)
    )
    semantic_payload = {
        "plan_id": RM65_ARMS_ONLY_WORKER_DRAIN_PLAN_V1,
        "execution_manifest_id": manifest.manifest_id,
        "execution_manifest_hash": manifest.semantic_content_hash,
        "dispatch_inventory_id": inventory.inventory_id,
        "dispatch_inventory_hash": inventory.semantic_content_hash,
        "participant_bindings": tuple(item.to_payload() for item in bindings),
        "integration_status": "scoped_unbound",
    }
    return RM65ArmsOnlyWorkerDrainPlan(
        plan_id=RM65_ARMS_ONLY_WORKER_DRAIN_PLAN_V1,
        execution_manifest_id=manifest.manifest_id,
        execution_manifest_hash=manifest.semantic_content_hash,
        dispatch_inventory_id=inventory.inventory_id,
        dispatch_inventory_hash=inventory.semantic_content_hash,
        participant_bindings=bindings,
        integration_status="scoped_unbound",
        semantic_content_hash=canonical_content_hash(
            semantic_payload,
            domain=RM65_ARMS_ONLY_WORKER_DRAIN_PLAN_HASH_DOMAIN_V1,
        ),
    )


__all__ = [
    "RM65_ARMS_ONLY_WORKER_DRAIN_PLAN_V1",
    "RM65ArmsOnlyWorkerDrainPlan",
    "RM65ArmsOnlyWorkerDrainResult",
    "RM65ExecutionDrainParticipant",
    "RM65WorkerDrainParticipantBinding",
    "build_rm65_arms_only_worker_drain_plan",
    "drain_rm65_arms_only_workers",
]
