"""Typed, default-off mediation seam for RM65 joint-target SDK calls.

The RM65 backend uses one private transport function for three different
control semantics: ordinary motion, bounded quick-stop ticks, and lifecycle
home/qpos-initialization.  A final dispatch gate cannot infer those semantics
from the SDK method or target qpos alone.

This module defines the immutable context passed to an explicitly injected
pre-SDK adapter.  It does not own a dispatch registry, issue permits, authorize
hardware, or install itself on a backend.  With no adapter configured the
legacy call path is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.dispatch_seam import (
    DispatchPermitInvocation,
    DispatchSeamGate,
    DispatchSeamSpec,
    PermittedDispatchResult,
)
from robot_motion.runtime.dispatch_owner import DispatchOwnerRegistry
from robot_motion.runtime.cancellation import ExecutionWorkItem


RM65_JOINT_TARGET_CALL_HASH_DOMAIN_V1 = "dexteleop.rm65_joint_target_call.v1"
RM65_JOINT_TARGET_CALL_PATHS_V1 = ("direct", "background_worker")


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class RM65JointTargetDispatchContext:
    """Exact semantic context adjacent to one RM65 SDK joint write."""

    side: str
    seam_id: str
    permit_kind: str
    operation: str | None
    call_path: str
    command_source: str
    target_sequence: int | None
    target_qpos_hash: str
    execution_work_item: ExecutionWorkItem | None
    semantic_content_hash: str

    @classmethod
    def create(
        cls,
        *,
        side: str,
        qpos_rad: np.ndarray,
        command_source: str,
        call_path: str,
        quick_stop_active: bool,
        lifecycle_home: bool,
        target_sequence: int | None = None,
        execution_work_item: ExecutionWorkItem | None = None,
    ) -> "RM65JointTargetDispatchContext":
        normalized_path = _identifier(call_path, name="call_path")
        if normalized_path not in RM65_JOINT_TARGET_CALL_PATHS_V1:
            raise ValueError("Unsupported RM65 joint-target call path.")
        if bool(quick_stop_active) and bool(lifecycle_home):
            # A queued home target may be interrupted by quick-stop.  The
            # resulting SDK tick is safety control, not lifecycle motion.
            lifecycle_home = False
        if bool(quick_stop_active):
            seam_id = "rm65.safety.quick_stop.tick.pre_sdk"
            permit_kind = "safety_control"
            operation = "rm65_quick_stop_tick"
        elif bool(lifecycle_home):
            seam_id = "rm65.lifecycle.home.pre_sdk.send_joint_target"
            permit_kind = "lifecycle_control"
            operation = "home"
        else:
            seam_id = "rm65.motion.pre_sdk.send_joint_target"
            permit_kind = "motion"
            operation = None

        normalized_sequence: int | None
        if target_sequence is None:
            normalized_sequence = None
        else:
            if isinstance(target_sequence, bool) or int(target_sequence) != target_sequence:
                raise ValueError("target_sequence must be an integer when provided.")
            if int(target_sequence) < 0:
                raise ValueError("target_sequence must be non-negative.")
            normalized_sequence = int(target_sequence)

        qpos = np.asarray(qpos_rad, dtype=float).reshape(-1)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise ValueError("qpos_rad must contain six finite values.")
        qpos_payload = tuple(float(value) for value in qpos)
        if execution_work_item is not None and not isinstance(
            execution_work_item,
            ExecutionWorkItem,
        ):
            raise TypeError("execution_work_item must be an ExecutionWorkItem or None.")
        target_qpos_hash = canonical_content_hash(
            {"qpos_rad": qpos_payload},
            domain="dexteleop.rm65_joint_target_qpos.v1",
        )
        payload = {
            "side": _identifier(side, name="side"),
            "seam_id": seam_id,
            "permit_kind": permit_kind,
            "operation": operation,
            "call_path": normalized_path,
            "command_source": _identifier(command_source, name="command_source"),
            "target_sequence": normalized_sequence,
            "target_qpos_hash": target_qpos_hash,
            "execution_work_item": (
                None
                if execution_work_item is None
                else {
                    "work_item_id": execution_work_item.work_item_id,
                    "work_kind": execution_work_item.work_kind,
                    "control_domain_id": execution_work_item.control_domain_id,
                    "authority_epoch": execution_work_item.authority_epoch,
                    "execution_epoch": execution_work_item.execution_epoch,
                    "segment_generation_by_group": dict(
                        execution_work_item.segment_generation_by_group
                    ),
                    "source_decision_id": execution_work_item.source_decision_id,
                    "source_reference_segment_id": (
                        execution_work_item.source_reference_segment_id
                    ),
                    "source_safety_operation_id": (
                        execution_work_item.source_safety_operation_id
                    ),
                    "source_safety_tick_evidence_id": (
                        execution_work_item.source_safety_tick_evidence_id
                    ),
                }
            ),
        }
        return cls(
            side=payload["side"],
            seam_id=payload["seam_id"],
            permit_kind=payload["permit_kind"],
            operation=payload["operation"],
            call_path=payload["call_path"],
            command_source=payload["command_source"],
            target_sequence=payload["target_sequence"],
            target_qpos_hash=payload["target_qpos_hash"],
            execution_work_item=execution_work_item,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=RM65_JOINT_TARGET_CALL_HASH_DOMAIN_V1,
            ),
        )


class RM65PreSdkDispatchAdapter(Protocol):
    """Explicit injection point; implementations must call or reject once."""

    def invoke(
        self,
        *,
        context: RM65JointTargetDispatchContext,
        backend_call: Callable[[], object],
    ) -> object: ...


class RM65PermittedPreSdkDispatchAdapter:
    """Route each typed RM65 call through an exact one-shot permit gate.

    The invocation provider is called for every SDK attempt and may not return
    a cached upstream/keyframe permit.  This adapter still does not mint
    permits itself; production wiring must provide the authoritative registry
    and a scope-aware provider.
    """

    def __init__(
        self,
        *,
        registry: DispatchOwnerRegistry,
        invocation_provider: Callable[
            [RM65JointTargetDispatchContext], DispatchPermitInvocation
        ],
        receipt_sink: Callable[
            [RM65JointTargetDispatchContext, PermittedDispatchResult[object]], None
        ]
        | None = None,
    ) -> None:
        if not isinstance(registry, DispatchOwnerRegistry):
            raise TypeError("registry must be a DispatchOwnerRegistry.")
        if not callable(invocation_provider):
            raise TypeError("invocation_provider must be callable.")
        if receipt_sink is not None and not callable(receipt_sink):
            raise TypeError("receipt_sink must be callable when provided.")
        self.registry = registry
        self.invocation_provider = invocation_provider
        self.receipt_sink = receipt_sink

    def invoke(
        self,
        *,
        context: RM65JointTargetDispatchContext,
        backend_call: Callable[[], object],
    ) -> object:
        if not isinstance(context, RM65JointTargetDispatchContext):
            raise TypeError("context must be an RM65JointTargetDispatchContext.")
        if not callable(backend_call):
            raise TypeError("backend_call must be callable.")
        invocation = self.invocation_provider(context)
        if not isinstance(invocation, DispatchPermitInvocation):
            raise TypeError(
                "invocation_provider must return DispatchPermitInvocation."
            )
        gate = DispatchSeamGate(
            registry=self.registry,
            spec=DispatchSeamSpec.create(
                seam_id=context.seam_id,
                permit_kind=context.permit_kind,
                operation=context.operation,
                call_stage="pre_sdk_call",
            ),
        )
        result = gate.invoke(
            invocation.permit,
            current_authority_epoch=invocation.current_authority_epoch,
            current_execution_epoch=invocation.current_execution_epoch,
            backend_call=backend_call,
        )
        if self.receipt_sink is not None:
            self.receipt_sink(context, result)
        return result.result


__all__ = [
    "RM65_JOINT_TARGET_CALL_PATHS_V1",
    "RM65JointTargetDispatchContext",
    "RM65PermittedPreSdkDispatchAdapter",
    "RM65PreSdkDispatchAdapter",
]
