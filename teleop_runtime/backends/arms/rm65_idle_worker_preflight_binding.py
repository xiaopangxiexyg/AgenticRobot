"""Isolated M5.8-b1 real idle-worker drain/stop-only composition.

This module is intentionally *not* a canonical deploy binding.  It enables an
RM65 background worker only while no target can be admitted, binds the three
arms-only drain participants to one execution-epoch registry, and permits at
most one ``Arm_Slow_Stop`` per authorized side.  Motion, lifecycle, home,
rearm, force-position, replay, and owner migration remain unavailable.

The narrow preflight profile is a runtime authorization object.  It references
the candidate execution manifest but never mutates it or any production
backend defaults, so conservative preflight settings cannot leak into later
real operation.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ControlExecutionManifest
from robot_motion.runtime import (
    DispatchPermitInvocation,
    DispatchSeamInventoryManifest,
    ExecutionCancellation,
    ExecutionCancellationParticipantSpec,
    ExecutionEpochGuard,
    PermittedDispatchResult,
)
from robot_motion.runtime.dispatch_owner import (
    DispatchPermit,
    DispatchPermitValidationReceipt,
)
from teleop_runtime.action import LegacyControlDomainExecutionContext

from .rm65_control_dispatch_call import (
    RM65ControlDispatchContext,
    RM65PermittedPreSdkControlAdapter,
)
from .rm65_execution_evidence import RM65ExecutionEvidenceBundle
from .rm65_stop_only_binding import RM65DenyAllMotionDispatchAdapter
from .rm65_worker_drain import RM65ArmsOnlyWorkerDrainResult

RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1 = (
    "m5_8b1_real_a_idle_worker_drain_stop_only_v1"
)
RM65_IDLE_WORKER_AUTHORIZATION_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_idle_worker_preflight_authorization.v1"
)
RM65_IDLE_WORKER_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_idle_worker_preflight_binding.v1"
)
RM65_IDLE_WORKER_BACKEND_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_idle_worker_backend_binding.v1"
)
RM65_IDLE_WORKER_STOP_SEAM_ID = "rm65.safety.hold.direct_stop"
RM65_IDLE_WORKER_STOP_SDK_METHOD = "Arm_Slow_Stop"


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _parse_sdk_result(result: object) -> tuple[int | None, str]:
    """Preserve the RM65 stop result as typed evidence without importing backend internals."""

    if isinstance(result, Integral) and not isinstance(result, bool):
        return int(result), str(int(result))
    text = "" if result is None else str(result)
    prefix = text.split(":", 1)[0].strip()
    try:
        return int(prefix), text
    except (TypeError, ValueError):
        return None, text


@dataclass(frozen=True)
class RM65IdleWorkerDrainHardwareAuthorization:
    authorization_id: str
    operator_authorization_id: str
    profile_id: str
    execution_manifest_id: str
    execution_manifest_hash: str
    dispatch_inventory_id: str
    dispatch_inventory_hash: str
    worker_drain_plan_id: str
    worker_drain_plan_hash: str
    control_domain_id: str
    embodiment_instance_id: str
    allowed_sides: tuple[str, ...]
    required_participant_ids: tuple[str, ...]
    allowed_stop_seam_id: str
    allowed_stop_sdk_method: str
    issued_at: float
    expires_at: float
    motion_authorized: bool
    lifecycle_authorized: bool
    semantic_content_hash: str

    @classmethod
    def create(
        cls,
        *,
        operator_authorization_id: str,
        execution_manifest: ControlExecutionManifest,
        dispatch_inventory: DispatchSeamInventoryManifest,
        worker_drain_plan,
        control_domain_id: str,
        embodiment_instance_id: str,
        allowed_sides: tuple[str, ...],
        issued_at: float,
        expires_at: float,
    ) -> RM65IdleWorkerDrainHardwareAuthorization:
        if not isinstance(execution_manifest, ControlExecutionManifest):
            raise TypeError("execution_manifest must be a ControlExecutionManifest.")
        if not isinstance(dispatch_inventory, DispatchSeamInventoryManifest):
            raise TypeError("dispatch_inventory must be a DispatchSeamInventoryManifest.")
        if (
            dispatch_inventory.execution_manifest_id != execution_manifest.manifest_id
            or dispatch_inventory.execution_manifest_hash
            != execution_manifest.semantic_content_hash
        ):
            raise ValueError("Dispatch inventory is bound to another manifest.")
        if (
            getattr(worker_drain_plan, "execution_manifest_hash", None)
            != execution_manifest.semantic_content_hash
            or getattr(worker_drain_plan, "dispatch_inventory_hash", None)
            != dispatch_inventory.semantic_content_hash
        ):
            raise ValueError("Worker-drain plan is bound to another evidence scope.")
        sides = tuple(_identifier(side, name="allowed_sides") for side in allowed_sides)
        if sides != ("left", "right"):
            raise ValueError(
                "The arms-only idle-worker preflight requires exact left/right scope."
            )
        participant_ids = tuple(
            sorted(item.participant_id for item in worker_drain_plan.participants)
        )
        expected_participants = (
            "position-deploy-reference-worker",
            "rm65-left-background-stream-worker",
            "rm65-right-background-stream-worker",
        )
        if participant_ids != expected_participants:
            raise ValueError("Worker-drain plan does not match the exact arms-only scope.")
        start = _finite(issued_at, name="issued_at")
        end = _finite(expires_at, name="expires_at")
        if end <= start:
            raise ValueError("expires_at must be later than issued_at.")
        payload = {
            "operator_authorization_id": _identifier(
                operator_authorization_id,
                name="operator_authorization_id",
            ),
            "profile_id": RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1,
            "execution_manifest_id": execution_manifest.manifest_id,
            "execution_manifest_hash": execution_manifest.semantic_content_hash,
            "dispatch_inventory_id": dispatch_inventory.inventory_id,
            "dispatch_inventory_hash": dispatch_inventory.semantic_content_hash,
            "worker_drain_plan_id": worker_drain_plan.plan_id,
            "worker_drain_plan_hash": worker_drain_plan.semantic_content_hash,
            "control_domain_id": _identifier(
                control_domain_id,
                name="control_domain_id",
            ),
            "embodiment_instance_id": _identifier(
                embodiment_instance_id,
                name="embodiment_instance_id",
            ),
            "allowed_sides": sides,
            "required_participant_ids": participant_ids,
            "allowed_stop_seam_id": RM65_IDLE_WORKER_STOP_SEAM_ID,
            "allowed_stop_sdk_method": RM65_IDLE_WORKER_STOP_SDK_METHOD,
            "issued_at": start,
            "expires_at": end,
            "motion_authorized": False,
            "lifecycle_authorized": False,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=RM65_IDLE_WORKER_AUTHORIZATION_HASH_DOMAIN_V1,
        )
        return cls(
            authorization_id=f"rm65-idle-worker-authorization:{semantic_hash}",
            **payload,
            semantic_content_hash=semantic_hash,
        )

    def require_active(self, now: float) -> None:
        current = _finite(now, name="now")
        if current < self.issued_at or current > self.expires_at:
            raise RuntimeError("RM65 idle-worker hardware authorization is not active.")


@dataclass(frozen=True)
class RM65IdleWorkerStopCallReceipt:
    side: str
    control_context_hash: str
    evidence_event_id: str
    permit_id: str
    validation_receipt_id: str
    dispatch_accepted: bool
    dispatch_status: str
    dispatch_validation_hash: str
    sdk_result_code: int | None
    sdk_result_text: str
    sdk_succeeded: bool
    sdk_result_repr: str


@dataclass(frozen=True)
class RM65IdleWorkerBackendBinding:
    binding_id: str
    side: str
    authorization: RM65IdleWorkerDrainHardwareAuthorization
    execution_epoch_guard: ExecutionEpochGuard
    worker_drain_participant: ExecutionCancellationParticipantSpec
    pre_sdk_dispatch_adapter: object
    pre_sdk_control_adapter: object
    semantic_content_hash: str

    def validate_config(
        self,
        *,
        side: str,
        connect_hardware: bool,
        background_streaming_enabled: bool,
        force_position_control: bool,
        configure_singularity_thresholds: bool,
        initial_qpos: object,
        now: float,
    ) -> None:
        self.authorization.require_active(now)
        if str(side) != self.side or self.side not in self.authorization.allowed_sides:
            raise ValueError("RM65 idle-worker backend side does not match authorization.")
        if not bool(connect_hardware) or not bool(background_streaming_enabled):
            raise ValueError(
                "RM65 idle-worker preflight requires real hardware and an idle background worker."
            )
        if bool(force_position_control) or bool(configure_singularity_thresholds):
            raise ValueError(
                "RM65 idle-worker preflight forbids force-position and startup SDK configuration."
            )
        if initial_qpos is not None:
            raise ValueError(
                "RM65 idle-worker preflight must establish its anchor from fresh hardware state."
            )


@dataclass(frozen=True)
class RM65IdleWorkerDrainPreflightSnapshot:
    binding_id: str
    runtime_session_id: str
    authorization_id: str
    profile_id: str
    context_mode: str
    production_binding_enabled: bool
    motion_authorized: bool
    lifecycle_authorized: bool
    execution_manifest_hash: str
    dispatch_inventory_hash: str
    worker_drain_plan_hash: str
    execution_epoch: int
    authority_epoch: int
    completed_drain_cancellation_id: str | None
    completed_drain_aggregate_id: str | None
    attempted_stop_sides: tuple[str, ...]
    stop_receipt_count: int
    semantic_content_hash: str


class RM65IdleWorkerDrainPreflightBinding:
    """Own the one registry used by the scoped real drain and stop preflight."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        owner_runtime_id: str,
        evidence_bundle: RM65ExecutionEvidenceBundle,
        authorization: RM65IdleWorkerDrainHardwareAuthorization,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(evidence_bundle, RM65ExecutionEvidenceBundle):
            raise TypeError("evidence_bundle must be RM65ExecutionEvidenceBundle.")
        if not isinstance(authorization, RM65IdleWorkerDrainHardwareAuthorization):
            raise TypeError(
                "authorization must be RM65IdleWorkerDrainHardwareAuthorization."
            )
        if not callable(clock):
            raise TypeError("clock must be callable.")
        authorization.require_active(clock())
        if (
            authorization.execution_manifest_hash
            != evidence_bundle.execution_manifest.semantic_content_hash
            or authorization.dispatch_inventory_hash
            != evidence_bundle.dispatch_inventory.semantic_content_hash
            or authorization.worker_drain_plan_hash
            != evidence_bundle.worker_drain_plan.semantic_content_hash
        ):
            raise ValueError("Hardware authorization belongs to another evidence bundle.")
        self.runtime_session_id = _identifier(
            runtime_session_id,
            name="runtime_session_id",
        )
        self.evidence_bundle = evidence_bundle
        self.authorization = authorization
        self.clock = clock
        binding_identity_payload = {
            "runtime_session_id": self.runtime_session_id,
            "owner_runtime_id": _identifier(
                owner_runtime_id,
                name="owner_runtime_id",
            ),
            "authorization_id": authorization.authorization_id,
            "execution_manifest_hash": authorization.execution_manifest_hash,
            "dispatch_inventory_hash": authorization.dispatch_inventory_hash,
            "worker_drain_plan_hash": authorization.worker_drain_plan_hash,
        }
        self.binding_id = "rm65-idle-worker-preflight:" + canonical_content_hash(
            binding_identity_payload,
            domain=RM65_IDLE_WORKER_BINDING_HASH_DOMAIN_V1,
        )
        self.context = (
            LegacyControlDomainExecutionContext.create_hardware_idle_drain_stop_only_preflight(
                runtime_session_id=self.runtime_session_id,
                control_domain_id=authorization.control_domain_id,
                embodiment_instance_id=authorization.embodiment_instance_id,
                clock_domain_id="control-monotonic",
                owner_runtime_id=owner_runtime_id,
                hardware_authorization_id=authorization.authorization_id,
                execution_manifest=evidence_bundle.execution_manifest,
                dispatch_inventory=evidence_bundle.dispatch_inventory,
                inventory_assessment=evidence_bundle.inventory_assessment,
                clock=clock,
            )
        )
        self.motion_adapter = RM65DenyAllMotionDispatchAdapter()
        self.control_adapter = RM65PermittedPreSdkControlAdapter(
            registry=self.context.dispatch_owners,
            invocation_provider=self._provide_stop_invocation,
            receipt_sink=self._record_stop_receipt,
        )
        self._attempted_stop_sides: set[str] = set()
        self._stop_receipts: list[RM65IdleWorkerStopCallReceipt] = []
        self._completed_worker_drain: RM65ArmsOnlyWorkerDrainResult | None = None
        self._completed_worker_drain_cancellation: ExecutionCancellation | None = None
        self._provided_stop_permit_by_context_hash: dict[str, DispatchPermit] = {}
        self._validated_stop_dispatches: list[
            tuple[DispatchPermit, DispatchPermitValidationReceipt]
        ] = []
        self._stop_admission_sequence = 0
        self._lock = threading.RLock()

    @property
    def stop_receipts(self) -> tuple[RM65IdleWorkerStopCallReceipt, ...]:
        with self._lock:
            return tuple(self._stop_receipts)

    @property
    def worker_drain_plan(self):
        return self.evidence_bundle.worker_drain_plan

    @property
    def production_binding_enabled(self) -> bool:
        return True

    @property
    def confers_motion_authorization(self) -> bool:
        return False

    def position_executor_kwargs(self) -> dict[str, object]:
        return {
            "execution_epoch_guard": self.context.execution_guard,
            "worker_drain_participant": self._participant(
                "position-deploy-reference-worker"
            ),
        }

    def backend_binding(self, side: str) -> RM65IdleWorkerBackendBinding:
        normalized = str(side).strip().lower()
        if normalized not in self.authorization.allowed_sides:
            raise ValueError("RM65 idle-worker side is outside hardware authorization.")
        participant = self._participant(
            f"rm65-{normalized}-background-stream-worker"
        )
        payload = {
            "binding_id": (
                f"rm65-idle-worker-backend:{self.runtime_session_id}:{normalized}"
            ),
            "side": normalized,
            "authorization_id": self.authorization.authorization_id,
            "participant_id": participant.participant_id,
            "participant_hash": participant.semantic_content_hash,
            "execution_manifest_hash": self.authorization.execution_manifest_hash,
            "dispatch_inventory_hash": self.authorization.dispatch_inventory_hash,
            "worker_drain_plan_hash": self.authorization.worker_drain_plan_hash,
            "motion_authorized": False,
            "lifecycle_authorized": False,
        }
        return RM65IdleWorkerBackendBinding(
            binding_id=payload["binding_id"],
            side=normalized,
            authorization=self.authorization,
            execution_epoch_guard=self.context.execution_guard,
            worker_drain_participant=participant,
            pre_sdk_dispatch_adapter=self.motion_adapter,
            pre_sdk_control_adapter=self.control_adapter,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=RM65_IDLE_WORKER_BACKEND_BINDING_HASH_DOMAIN_V1,
            ),
        )

    def snapshot(self) -> RM65IdleWorkerDrainPreflightSnapshot:
        with self._lock:
            context = self.context.snapshot()
            payload = {
                "binding_id": self.binding_id,
                "runtime_session_id": self.runtime_session_id,
                "authorization_id": self.authorization.authorization_id,
                "profile_id": self.authorization.profile_id,
                "context_mode": context.composition_mode,
                "production_binding_enabled": context.production_binding_enabled,
                "motion_authorized": False,
                "lifecycle_authorized": False,
                "execution_manifest_hash": self.authorization.execution_manifest_hash,
                "dispatch_inventory_hash": self.authorization.dispatch_inventory_hash,
                "worker_drain_plan_hash": self.authorization.worker_drain_plan_hash,
                "execution_epoch": context.execution_epoch,
                "authority_epoch": context.authority_epoch,
                "completed_drain_cancellation_id": (
                    None
                    if self._completed_worker_drain is None
                    else self._completed_worker_drain.cancellation_id
                ),
                "completed_drain_aggregate_id": (
                    None
                    if self._completed_worker_drain is None
                    else self._completed_worker_drain.aggregate.aggregate_id
                ),
                "attempted_stop_sides": tuple(sorted(self._attempted_stop_sides)),
                "stop_receipt_count": len(self._stop_receipts),
            }
            return RM65IdleWorkerDrainPreflightSnapshot(
                **payload,
                semantic_content_hash=canonical_content_hash(
                    payload,
                    domain=RM65_IDLE_WORKER_BINDING_HASH_DOMAIN_V1,
                ),
            )

    def accept_completed_worker_drain(
        self,
        result: RM65ArmsOnlyWorkerDrainResult,
        *,
        cancellation: ExecutionCancellation,
    ) -> None:
        """Latch exact completed drain evidence before any stop permit exists."""

        if not isinstance(result, RM65ArmsOnlyWorkerDrainResult):
            raise TypeError("result must be RM65ArmsOnlyWorkerDrainResult.")
        if not isinstance(cancellation, ExecutionCancellation):
            raise TypeError("cancellation must be ExecutionCancellation.")
        with self._lock:
            if self._completed_worker_drain is not None:
                raise RuntimeError("A completed worker drain is already latched.")
            execution = self.context.execution_epochs.snapshot()
            if (
                result.plan_id != self.worker_drain_plan.plan_id
                or result.cancellation_id != cancellation.cancellation_id
                or result.cancellation_id != execution.last_cancellation_id
                or not result.aggregate.complete
            ):
                raise ValueError(
                    "Stop authorization requires the exact complete current worker drain."
                )
            self._completed_worker_drain = result
            self._completed_worker_drain_cancellation = cancellation

    def project_stop_dispatch_facts(self, trace_context) -> tuple[object, ...]:
        """Project completed permit validations without owning trace lifecycle."""

        from robot_motion.action.authoritative_trace import (
            AuthoritativeTraceContext,
            project_dispatch_permit_validation,
        )

        if not isinstance(trace_context, AuthoritativeTraceContext):
            raise TypeError("trace_context must be AuthoritativeTraceContext.")
        with self._lock:
            return tuple(
                project_dispatch_permit_validation(
                    trace_context,
                    permit=permit,
                    validation=validation,
                )
                for permit, validation in self._validated_stop_dispatches
            )

    def project_authorization_fact(self, trace_context):
        """Project the immutable narrow authorization as cancellation source."""

        from robot_motion.action.authoritative_trace import AuthoritativeTraceContext

        if not isinstance(trace_context, AuthoritativeTraceContext):
            raise TypeError("trace_context must be AuthoritativeTraceContext.")
        payload = {
            name: getattr(self.authorization, name)
            for name in self.authorization.__dataclass_fields__
            if name != "semantic_content_hash"
        }
        payload["semantic_content_hash"] = (
            self.authorization.semantic_content_hash
        )
        return trace_context.fact(
            fact_id=self.authorization.authorization_id,
            event_type="hardware_preflight_authorization",
            occurred_at=self.authorization.issued_at,
            payload=payload,
        )

    def _participant(self, participant_id: str) -> ExecutionCancellationParticipantSpec:
        matches = tuple(
            participant
            for participant in self.worker_drain_plan.participants
            if participant.participant_id == participant_id
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Worker-drain plan lacks exact participant {participant_id!r}."
            )
        return matches[0]

    def _provide_stop_invocation(
        self,
        context: RM65ControlDispatchContext,
    ) -> DispatchPermitInvocation:
        if not isinstance(context, RM65ControlDispatchContext):
            raise TypeError("context must be RM65ControlDispatchContext.")
        with self._lock:
            self.authorization.require_active(self.clock())
            if self._completed_worker_drain is None:
                raise RuntimeError(
                    "M5.8-b1 stop is unavailable before the exact worker drain completes."
                )
            cancellation = self._completed_worker_drain_cancellation
            if cancellation is None:
                raise RuntimeError("Completed worker drain lacks cancellation evidence.")
            if (
                context.seam_id != self.authorization.allowed_stop_seam_id
                or context.permit_kind != "safety_control"
                or context.operation != "hold"
                or context.sdk_method
                != self.authorization.allowed_stop_sdk_method
                or not context.source.startswith(
                    "manual_hold:m5_8b1_idle_worker_drain"
                )
            ):
                raise RuntimeError(
                    "M5.8-b1 authorization permits only the exact manual "
                    "Arm_Slow_Stop seam after worker drain."
                )
            if context.side not in self.authorization.allowed_sides:
                raise RuntimeError(f"RM65 side {context.side!r} is not authorized.")
            if context.side in self._attempted_stop_sides:
                raise RuntimeError(
                    f"RM65 side {context.side!r} already consumed its stop attempt."
                )
            owner = self.context.dispatch_owners.snapshot()
            execution = self.context.execution_epochs.snapshot()
            if owner.current_execution_epoch != execution.execution_epoch:
                raise RuntimeError("Owner and execution registries disagree on epoch.")
            self._stop_admission_sequence += 1
            permit = self.context.dispatch_owners.issue_safety_control_permit(
                operation="hold",
                evidence=cancellation,
                safety_source="execution_cancellation_runtime",
            )
            self._attempted_stop_sides.add(context.side)
            self._provided_stop_permit_by_context_hash[
                context.semantic_content_hash
            ] = permit
            return DispatchPermitInvocation(
                permit=permit,
                current_authority_epoch=owner.current_authority_epoch,
                current_execution_epoch=execution.execution_epoch,
            )

    def _record_stop_receipt(
        self,
        context: RM65ControlDispatchContext,
        result: PermittedDispatchResult[object],
    ) -> None:
        with self._lock:
            permit = self._provided_stop_permit_by_context_hash.pop(
                context.semantic_content_hash,
                None,
            )
            if permit is None or permit.permit_id != result.validation.permit_id:
                raise RuntimeError("Stop validation has no matching issued permit.")
            self._validated_stop_dispatches.append(
                (permit, result.owner_validation)
            )
            sdk_result_code, sdk_result_text = _parse_sdk_result(result.result)
            self._stop_receipts.append(
                RM65IdleWorkerStopCallReceipt(
                    side=context.side,
                    control_context_hash=context.semantic_content_hash,
                    evidence_event_id=str(permit.evidence_event_id),
                    permit_id=result.validation.permit_id,
                    validation_receipt_id=(
                        "rm65-idle-worker-stop-validation:"
                        f"{result.validation.semantic_content_hash}"
                    ),
                    dispatch_accepted=result.validation.accepted,
                    dispatch_status=result.validation.status,
                    dispatch_validation_hash=result.validation.semantic_content_hash,
                    sdk_result_code=sdk_result_code,
                    sdk_result_text=sdk_result_text,
                    sdk_succeeded=sdk_result_code == 0,
                    sdk_result_repr=repr(result.result),
                )
            )


__all__ = [
    "RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1",
    "RM65IdleWorkerBackendBinding",
    "RM65IdleWorkerDrainHardwareAuthorization",
    "RM65IdleWorkerDrainPreflightBinding",
    "RM65IdleWorkerDrainPreflightSnapshot",
    "RM65IdleWorkerStopCallReceipt",
]
