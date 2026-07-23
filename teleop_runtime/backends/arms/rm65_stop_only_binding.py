"""Narrow M5.8-b0 binding for an authorized stationary RM65 slow-stop check.

This module intentionally does *not* provide motion or lifecycle permits.  It
admits at most one ``Arm_Slow_Stop`` call per explicitly authorized arm and
routes that call through the real one-shot final SDK seam.  It is not the full
M5.8-b production composition and must not be reused as M6 authority evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Callable

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ControlExecutionManifest
from robot_motion.runtime import (
    AdmittedSafetyControlEvidence,
    DispatchOwnerRegistry,
    DispatchPermitInvocation,
    DispatchSeamInventoryManifest,
    PermittedDispatchResult,
)

from .rm65_control_dispatch_call import (
    RM65ControlDispatchContext,
    RM65PermittedPreSdkControlAdapter,
)
from .rm65_dispatch_call import RM65JointTargetDispatchContext


RM65_STOP_ONLY_AUTHORIZATION_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_stop_only_hardware_authorization.v1"
)
RM65_STOP_ONLY_BINDING_HASH_DOMAIN_V1 = "dexteleop.rm65_stop_only_binding.v1"
RM65_STOP_ONLY_SEAM_ID = "rm65.safety.hold.direct_stop"
RM65_STOP_ONLY_SDK_METHOD = "Arm_Slow_Stop"


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


@dataclass(frozen=True)
class RM65StopOnlyHardwareAuthorization:
    authorization_id: str
    operator_authorization_id: str
    execution_manifest_id: str
    execution_manifest_hash: str
    dispatch_inventory_id: str
    dispatch_inventory_hash: str
    control_domain_id: str
    embodiment_instance_id: str
    allowed_sides: tuple[str, ...]
    allowed_seam_id: str
    allowed_sdk_method: str
    issued_at: float
    expires_at: float
    semantic_content_hash: str

    @classmethod
    def create(
        cls,
        *,
        operator_authorization_id: str,
        execution_manifest: ControlExecutionManifest,
        dispatch_inventory: DispatchSeamInventoryManifest,
        control_domain_id: str,
        embodiment_instance_id: str,
        allowed_sides: tuple[str, ...],
        issued_at: float,
        expires_at: float,
    ) -> "RM65StopOnlyHardwareAuthorization":
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
        sides = tuple(_identifier(side, name="allowed_sides") for side in allowed_sides)
        if not sides or len(set(sides)) != len(sides) or set(sides) - {"left", "right"}:
            raise ValueError("allowed_sides must be a unique subset of left/right.")
        start = _finite(issued_at, name="issued_at")
        end = _finite(expires_at, name="expires_at")
        if end <= start:
            raise ValueError("expires_at must be later than issued_at.")
        payload = {
            "operator_authorization_id": _identifier(
                operator_authorization_id,
                name="operator_authorization_id",
            ),
            "execution_manifest_id": execution_manifest.manifest_id,
            "execution_manifest_hash": execution_manifest.semantic_content_hash,
            "dispatch_inventory_id": dispatch_inventory.inventory_id,
            "dispatch_inventory_hash": dispatch_inventory.semantic_content_hash,
            "control_domain_id": _identifier(
                control_domain_id,
                name="control_domain_id",
            ),
            "embodiment_instance_id": _identifier(
                embodiment_instance_id,
                name="embodiment_instance_id",
            ),
            "allowed_sides": sides,
            "allowed_seam_id": RM65_STOP_ONLY_SEAM_ID,
            "allowed_sdk_method": RM65_STOP_ONLY_SDK_METHOD,
            "issued_at": start,
            "expires_at": end,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=RM65_STOP_ONLY_AUTHORIZATION_HASH_DOMAIN_V1,
        )
        return cls(
            authorization_id=f"rm65-stop-only-authorization:{semantic_hash}",
            **payload,
            semantic_content_hash=semantic_hash,
        )


@dataclass(frozen=True)
class RM65StopOnlyCallReceipt:
    side: str
    seam_id: str
    permit_kind: str
    operation: str
    source: str
    sdk_method: str
    attempt_sequence: int
    control_context_hash: str
    permit_id: str
    validation_receipt_id: str
    owner_validation_receipt_id: str | None
    dispatch_accepted: bool
    dispatch_status: str
    dispatch_validation_hash: str
    sdk_result_repr: str


@dataclass(frozen=True)
class RM65StopOnlyBindingSnapshot:
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    authorization_id: str
    execution_manifest_hash: str
    dispatch_inventory_hash: str
    owner_token_id: str
    authority_epoch: int
    execution_epoch: int
    permitted_sides: tuple[str, ...]
    attempted_sides: tuple[str, ...]
    receipt_count: int
    production_binding_enabled: bool
    motion_authorized: bool
    semantic_content_hash: str


class RM65DenyAllMotionDispatchAdapter:
    """Code-level firewall: no joint-target SDK call is legal in M5.8-b0."""

    def invoke(self, *, context: RM65JointTargetDispatchContext, backend_call):
        del backend_call
        raise RuntimeError(
            "M5.8-b0 stop-only binding rejects every RM65 joint-target SDK call "
            f"(side={context.side}, seam={context.seam_id})."
        )


class RM65StopOnlyLegacyBinding:
    """Own the sole registry and exact stop-only permit provider for M5.8-b0."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        owner_runtime_id: str,
        authorization: RM65StopOnlyHardwareAuthorization,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(authorization, RM65StopOnlyHardwareAuthorization):
            raise TypeError("authorization must be RM65StopOnlyHardwareAuthorization.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.runtime_session_id = _identifier(
            runtime_session_id,
            name="runtime_session_id",
        )
        self.authorization = authorization
        self.clock = clock
        now = _finite(clock(), name="clock")
        if now < authorization.issued_at or now > authorization.expires_at:
            raise RuntimeError("Stop-only hardware authorization is not currently active.")
        self.registry = DispatchOwnerRegistry(
            runtime_session_id=self.runtime_session_id,
            control_domain_id=authorization.control_domain_id,
            initial_authority_epoch=0,
            initial_execution_epoch=0,
            clock=clock,
        )
        self.legacy_owner_token = self.registry.grant_initial_owner(
            owner_runtime_id=_identifier(owner_runtime_id, name="owner_runtime_id"),
            owner_kind="legacy_control_path",
        )
        self.motion_adapter = RM65DenyAllMotionDispatchAdapter()
        self.control_adapter = RM65PermittedPreSdkControlAdapter(
            registry=self.registry,
            invocation_provider=self._provide_invocation,
            receipt_sink=self._record_receipt,
        )
        self._attempted_sides: set[str] = set()
        self._receipts: list[RM65StopOnlyCallReceipt] = []
        self._admission_sequence = 0
        self._lock = threading.RLock()

    @property
    def receipts(self) -> tuple[RM65StopOnlyCallReceipt, ...]:
        with self._lock:
            return tuple(self._receipts)

    def snapshot(self) -> RM65StopOnlyBindingSnapshot:
        with self._lock:
            owner = self.registry.snapshot()
            token = owner.current_token
            if token is None:
                raise RuntimeError("Stop-only binding lost its legacy owner token.")
            payload = {
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.authorization.control_domain_id,
                "embodiment_instance_id": self.authorization.embodiment_instance_id,
                "authorization_id": self.authorization.authorization_id,
                "execution_manifest_hash": self.authorization.execution_manifest_hash,
                "dispatch_inventory_hash": self.authorization.dispatch_inventory_hash,
                "owner_token_id": token.token_id,
                "authority_epoch": owner.current_authority_epoch,
                "execution_epoch": owner.current_execution_epoch,
                "permitted_sides": self.authorization.allowed_sides,
                "attempted_sides": tuple(sorted(self._attempted_sides)),
                "receipt_count": len(self._receipts),
                "production_binding_enabled": True,
                "motion_authorized": False,
            }
            return RM65StopOnlyBindingSnapshot(
                **payload,
                semantic_content_hash=canonical_content_hash(
                    payload,
                    domain=RM65_STOP_ONLY_BINDING_HASH_DOMAIN_V1,
                ),
            )

    def _provide_invocation(
        self,
        context: RM65ControlDispatchContext,
    ) -> DispatchPermitInvocation:
        if not isinstance(context, RM65ControlDispatchContext):
            raise TypeError("context must be RM65ControlDispatchContext.")
        with self._lock:
            now = _finite(self.clock(), name="clock")
            if now > self.authorization.expires_at:
                raise RuntimeError("Stop-only hardware authorization expired.")
            if (
                context.seam_id != self.authorization.allowed_seam_id
                or context.permit_kind != "safety_control"
                or context.operation != "hold"
                or context.sdk_method != self.authorization.allowed_sdk_method
                or not context.source.startswith("manual_hold:m5_8b0_stop_only")
            ):
                raise RuntimeError(
                    "M5.8-b0 authorization permits only the exact manual "
                    "Arm_Slow_Stop hold seam."
                )
            if context.side not in self.authorization.allowed_sides:
                raise RuntimeError(f"RM65 side {context.side!r} is not authorized.")
            if context.side in self._attempted_sides:
                raise RuntimeError(
                    f"RM65 side {context.side!r} already consumed its stop-only attempt."
                )
            owner = self.registry.snapshot()
            if (
                owner.current_token is None
                or owner.current_token.token_id != self.legacy_owner_token.token_id
                or owner.current_token.owner_kind != "legacy_control_path"
            ):
                raise RuntimeError("Legacy owner token is no longer current.")
            self._admission_sequence += 1
            evidence = AdmittedSafetyControlEvidence(
                event_id=(
                    f"{self.authorization.authorization_id}:"
                    f"{context.side}:{self._admission_sequence}"
                ),
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.authorization.control_domain_id,
                execution_epoch=owner.current_execution_epoch,
                evidence_kind="safety_event",
                safety_source="runtime_supervisor",
                control_admission_sequence=self._admission_sequence,
            )
            permit = self.registry.issue_safety_control_permit(
                operation="hold",
                evidence=evidence,
                safety_source="runtime_supervisor",
            )
            # Consume the side budget before the SDK gate.  A failed call must
            # never silently retry under the same operator authorization.
            self._attempted_sides.add(context.side)
            return DispatchPermitInvocation(
                permit=permit,
                current_authority_epoch=owner.current_authority_epoch,
                current_execution_epoch=owner.current_execution_epoch,
            )

    def _record_receipt(
        self,
        context: RM65ControlDispatchContext,
        result: PermittedDispatchResult[object],
    ) -> None:
        with self._lock:
            self._receipts.append(
                RM65StopOnlyCallReceipt(
                    side=context.side,
                    seam_id=context.seam_id,
                    permit_kind=context.permit_kind,
                    operation=context.operation,
                    source=context.source,
                    sdk_method=context.sdk_method,
                    attempt_sequence=context.attempt_sequence,
                    control_context_hash=context.semantic_content_hash,
                    permit_id=result.validation.permit_id,
                    validation_receipt_id=(
                        f"rm65-stop-only-validation:"
                        f"{result.validation.semantic_content_hash}"
                    ),
                    owner_validation_receipt_id=(
                        result.validation.owner_validation_receipt_id
                    ),
                    dispatch_accepted=result.validation.accepted,
                    dispatch_status=result.validation.status,
                    dispatch_validation_hash=result.validation.semantic_content_hash,
                    sdk_result_repr=repr(result.result),
                )
            )


__all__ = [
    "RM65DenyAllMotionDispatchAdapter",
    "RM65StopOnlyBindingSnapshot",
    "RM65StopOnlyCallReceipt",
    "RM65StopOnlyHardwareAuthorization",
    "RM65StopOnlyLegacyBinding",
    "RM65_STOP_ONLY_SDK_METHOD",
    "RM65_STOP_ONLY_SEAM_ID",
]
