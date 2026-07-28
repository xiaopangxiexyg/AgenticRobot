"""Typed representation adaptation after authoritative action selection.

Adaptation converts an immutable authoritative ``ComponentAction`` decision
into a lower representation candidate.  It is not canonical safety, target
resolution/admission, reference activation, or dispatch.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.immutable import deep_freeze

from .component_action import ComponentAction
from .component_action_v1 import materialize_robot_action_v1
from .proposal import freeze_robot_action, robot_action_semantic_payload
from .schema import RobotAction
from .target_capability import (
    ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH,
    ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID,
    ComponentTargetCapabilitySpec,
    TargetCapabilityAssessment,
    assess_component_target_capability,
)


ACTION_ADAPTATION_REQUEST_HASH_DOMAIN_V1 = (
    "dexteleop.action_adaptation_request.v1"
)
ACTION_ADAPTATION_REQUEST_HASH_DOMAIN_V2 = (
    "dexteleop.action_adaptation_request.v2"
)
ADAPTED_ACTION_CANDIDATE_HASH_DOMAIN_V1 = (
    "dexteleop.adapted_action_candidate.v1"
)
ACTION_ADAPTATION_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.action_adaptation_receipt.v1"
)
TARGET_CAPABILITY_ASSESSMENT_INSTANCE_HASH_DOMAIN_V1 = (
    "dexteleop.target_capability_assessment_instance.v1"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


@dataclass(frozen=True)
class ActionAdaptationRequest:
    """Exact authoritative candidate submitted to one representation adapter."""

    request_id: str
    authoritative_decision_id: str
    authoritative_decision_hash: str
    arbitration_result_id: str
    candidate_action: ComponentAction
    candidate_action_hash: str
    action_schema_hash: str
    component_target_capability_id: str
    component_target_capability_hash: str
    component_coverage: tuple[str, ...]
    target_effective_at: float
    supported_until: float
    authority_sequence: int
    execution_gate_sequence: int
    execution_fence_sequence: int
    requested_at: float
    clock_domain_id: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "authoritative_decision_id",
            "authoritative_decision_hash",
            "arbitration_result_id",
            "candidate_action_hash",
            "action_schema_hash",
            "component_target_capability_id",
            "component_target_capability_hash",
            "clock_domain_id",
            "semantic_content_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        if not isinstance(self.candidate_action, ComponentAction):
            raise TypeError("candidate_action must be a ComponentAction.")
        if self.candidate_action.semantic_content_hash != self.candidate_action_hash:
            raise ValueError("Action adaptation request candidate hash mismatch.")
        if (
            self.candidate_action.action_schema.semantic_content_hash
            != self.action_schema_hash
        ):
            raise ValueError("Action adaptation request schema hash mismatch.")
        coverage = tuple(str(value).strip() for value in self.component_coverage)
        if (
            not coverage
            or len(coverage) != len(set(coverage))
            or coverage != self.candidate_action.component_coverage
        ):
            raise ValueError(
                "Action adaptation request coverage must equal the candidate."
            )
        object.__setattr__(self, "component_coverage", coverage)
        for name in (
            "authority_sequence",
            "execution_gate_sequence",
            "execution_fence_sequence",
        ):
            object.__setattr__(
                self,
                name,
                _non_negative_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "target_effective_at",
            _finite(self.target_effective_at, name="target_effective_at"),
        )
        supported_until = _finite(
            self.supported_until,
            name="supported_until",
        )
        if supported_until <= self.target_effective_at:
            raise ValueError(
                "Adaptation support must extend beyond the target."
            )
        object.__setattr__(self, "supported_until", supported_until)
        object.__setattr__(
            self,
            "requested_at",
            _finite(self.requested_at, name="requested_at"),
        )
        if self.requested_at >= supported_until:
            raise ValueError(
                "Adaptation cannot be requested after support expires."
            )


def build_action_adaptation_request(
    *,
    authoritative_decision_id: str,
    authoritative_decision_hash: str,
    arbitration_result_id: str,
    candidate_action: ComponentAction,
    component_target_capability: ComponentTargetCapabilitySpec,
    target_effective_at: float,
    supported_until: float,
    authority_sequence: int,
    execution_gate_sequence: int,
    execution_fence_sequence: int,
    requested_at: float,
    clock_domain_id: str,
) -> ActionAdaptationRequest:
    if not isinstance(candidate_action, ComponentAction):
        raise TypeError("candidate_action must be a ComponentAction.")
    if not isinstance(
        component_target_capability,
        ComponentTargetCapabilitySpec,
    ):
        raise TypeError(
            "component_target_capability must be a "
            "ComponentTargetCapabilitySpec."
        )
    payload = {
        "authoritative_decision_id": _required(
            authoritative_decision_id,
            name="authoritative_decision_id",
        ),
        "authoritative_decision_hash": _required(
            authoritative_decision_hash,
            name="authoritative_decision_hash",
        ),
        "arbitration_result_id": _required(
            arbitration_result_id,
            name="arbitration_result_id",
        ),
        "candidate_action_hash": candidate_action.semantic_content_hash,
        "action_schema_hash": candidate_action.action_schema.semantic_content_hash,
        "component_target_capability_id": (
            component_target_capability.capability_id
        ),
        "component_target_capability_hash": (
            component_target_capability.semantic_content_hash
        ),
        "component_coverage": candidate_action.component_coverage,
        "target_effective_at": _finite(
            target_effective_at,
            name="target_effective_at",
        ),
        "supported_until": _finite(
            supported_until,
            name="supported_until",
        ),
        "authority_sequence": _non_negative_int(
            authority_sequence,
            name="authority_sequence",
        ),
        "execution_gate_sequence": _non_negative_int(
            execution_gate_sequence,
            name="execution_gate_sequence",
        ),
        "execution_fence_sequence": _non_negative_int(
            execution_fence_sequence,
            name="execution_fence_sequence",
        ),
        "requested_at": _finite(requested_at, name="requested_at"),
        "clock_domain_id": _required(
            clock_domain_id,
            name="clock_domain_id",
        ),
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=ACTION_ADAPTATION_REQUEST_HASH_DOMAIN_V2,
    )
    return ActionAdaptationRequest(
        request_id=f"action-adaptation-request:{semantic_hash}",
        candidate_action=candidate_action,
        **payload,
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class AdaptedActionCandidate:
    """One representation-adapted, explicitly non-safe candidate."""

    adapted_candidate_id: str
    request_id: str
    request_hash: str
    source_candidate_action_hash: str
    payload_kind: str
    output_action_contract_hash: str
    adapter_contract_id: str
    adapter_contract_hash: str
    payload_semantic: Mapping[str, Any]
    runtime_payload: object = field(repr=False, compare=False)
    created_at: float = 0.0
    clock_domain_id: str = ""
    semantic_content_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "adapted_candidate_id",
            "request_id",
            "request_hash",
            "source_candidate_action_hash",
            "payload_kind",
            "output_action_contract_hash",
            "adapter_contract_id",
            "adapter_contract_hash",
            "clock_domain_id",
            "semantic_content_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "payload_semantic",
            MappingProxyType(dict(deep_freeze(dict(self.payload_semantic)))),
        )
        object.__setattr__(
            self,
            "created_at",
            _finite(self.created_at, name="created_at"),
        )


@dataclass(frozen=True)
class ActionAdaptationReceipt:
    """Receiver completion evidence; not execution-target admission."""

    receipt_id: str
    request_id: str
    request_hash: str
    capability_assessment_id: str
    capability_assessment_hash: str
    receiver_id: str
    processing_started_at: float
    processing_completed_at: float
    clock_domain_id: str
    accepted: bool
    rejection_code: str | None
    adapted_candidate_id: str | None
    adapted_candidate_hash: str | None
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "receipt_id",
            "request_id",
            "request_hash",
            "capability_assessment_id",
            "capability_assessment_hash",
            "receiver_id",
            "clock_domain_id",
            "semantic_content_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        started = _finite(
            self.processing_started_at,
            name="processing_started_at",
        )
        completed = _finite(
            self.processing_completed_at,
            name="processing_completed_at",
        )
        if completed < started:
            raise ValueError(
                "Action adaptation cannot complete before it starts."
            )
        object.__setattr__(self, "processing_started_at", started)
        object.__setattr__(self, "processing_completed_at", completed)
        if self.accepted:
            if self.rejection_code is not None:
                raise ValueError(
                    "Accepted action adaptation cannot have a rejection code."
                )
            if self.adapted_candidate_id is None or (
                self.adapted_candidate_hash is None
            ):
                raise ValueError(
                    "Accepted action adaptation requires a candidate identity."
                )
        else:
            if not str(self.rejection_code or "").strip():
                raise ValueError(
                    "Rejected action adaptation requires a rejection code."
                )
            if self.adapted_candidate_id is not None or (
                self.adapted_candidate_hash is not None
            ):
                raise ValueError(
                    "Rejected action adaptation cannot reference a candidate."
                )
        for name in ("adapted_candidate_id", "adapted_candidate_hash"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _required(value, name=name))


@dataclass(frozen=True)
class ActionAdaptationResult:
    request: ActionAdaptationRequest
    capability_assessment: TargetCapabilityAssessment
    adapted_candidate: AdaptedActionCandidate | None
    receipt: ActionAdaptationReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.request, ActionAdaptationRequest):
            raise TypeError("request must be an ActionAdaptationRequest.")
        if not isinstance(
            self.capability_assessment,
            TargetCapabilityAssessment,
        ):
            raise TypeError(
                "capability_assessment must be a TargetCapabilityAssessment."
            )
        if not isinstance(self.receipt, ActionAdaptationReceipt):
            raise TypeError("receipt must be an ActionAdaptationReceipt.")
        if (
            self.receipt.request_id != self.request.request_id
            or self.receipt.request_hash != self.request.semantic_content_hash
            or self.receipt.capability_assessment_id
            != target_capability_assessment_instance_id(
                self.request,
                self.capability_assessment,
            )
            or self.receipt.capability_assessment_hash
            != self.capability_assessment.semantic_content_hash
        ):
            raise ValueError("Action adaptation result lineage mismatch.")
        if self.receipt.accepted:
            if not isinstance(self.adapted_candidate, AdaptedActionCandidate):
                raise ValueError(
                    "Accepted adaptation result requires an adapted candidate."
                )
            if (
                self.receipt.adapted_candidate_id
                != self.adapted_candidate.adapted_candidate_id
                or self.receipt.adapted_candidate_hash
                != self.adapted_candidate.semantic_content_hash
            ):
                raise ValueError(
                    "Action adaptation receipt references another candidate."
                )
        elif self.adapted_candidate is not None:
            raise ValueError(
                "Rejected adaptation result cannot carry a candidate."
            )


class ActionRepresentationAdapter(Protocol):
    """Lower-owned representation adapter with one typed completion result."""

    @property
    def component_target_capability(self) -> ComponentTargetCapabilitySpec:
        ...

    def adapt(
        self,
        request: ActionAdaptationRequest,
    ) -> ActionAdaptationResult:
        ...


def target_capability_assessment_instance_id(
    request: ActionAdaptationRequest,
    assessment: TargetCapabilityAssessment,
) -> str:
    """Bind reusable pure assessment content to one submitted request."""

    if not isinstance(request, ActionAdaptationRequest):
        raise TypeError("request must be an ActionAdaptationRequest.")
    if not isinstance(assessment, TargetCapabilityAssessment):
        raise TypeError("assessment must be a TargetCapabilityAssessment.")
    identity_hash = canonical_content_hash(
        {
            "request_id": request.request_id,
            "request_hash": request.semantic_content_hash,
            "assessment_hash": assessment.semantic_content_hash,
        },
        domain=TARGET_CAPABILITY_ASSESSMENT_INSTANCE_HASH_DOMAIN_V1,
    )
    return f"target-capability-assessment:{identity_hash}"


def build_action_adaptation_receipt(
    *,
    request: ActionAdaptationRequest,
    assessment: TargetCapabilityAssessment,
    receiver_id: str,
    processing_started_at: float,
    processing_completed_at: float,
    accepted: bool,
    rejection_code: str | None,
    adapted_candidate: AdaptedActionCandidate | None,
) -> ActionAdaptationReceipt:
    """Build exact receiver completion evidence for one adaptation request."""

    if not isinstance(request, ActionAdaptationRequest):
        raise TypeError("request must be an ActionAdaptationRequest.")
    if not isinstance(assessment, TargetCapabilityAssessment):
        raise TypeError("assessment must be a TargetCapabilityAssessment.")
    if accepted and not assessment.accepted:
        raise ValueError(
            "A structurally rejected candidate cannot be adapter-accepted."
        )
    receipt_payload = {
        "request_id": request.request_id,
        "request_hash": request.semantic_content_hash,
        "capability_assessment_id": (
            target_capability_assessment_instance_id(
                request,
                assessment,
            )
        ),
        "capability_assessment_hash": assessment.semantic_content_hash,
        "receiver_id": _required(receiver_id, name="receiver_id"),
        "processing_started_at": _finite(
            processing_started_at,
            name="processing_started_at",
        ),
        "processing_completed_at": _finite(
            processing_completed_at,
            name="processing_completed_at",
        ),
        "clock_domain_id": request.clock_domain_id,
        "accepted": bool(accepted),
        "rejection_code": rejection_code,
        "adapted_candidate_id": (
            None
            if adapted_candidate is None
            else adapted_candidate.adapted_candidate_id
        ),
        "adapted_candidate_hash": (
            None
            if adapted_candidate is None
            else adapted_candidate.semantic_content_hash
        ),
    }
    receipt_hash = canonical_content_hash(
        receipt_payload,
        domain=ACTION_ADAPTATION_RECEIPT_HASH_DOMAIN_V1,
    )
    return ActionAdaptationReceipt(
        receipt_id=f"action-adaptation-receipt:{receipt_hash}",
        **receipt_payload,
        semantic_content_hash=receipt_hash,
    )


class RobotActionV1RepresentationAdapter:
    """ComponentAction -> lower RobotAction carrier through an explicit adapter.

    The default materializer preserves the historical lossless v1 bridge.
    Target packages may inject another typed materializer and route contract
    without changing arbitration or composition.  The materializer remains
    responsible for preserving units and representation semantics; this seam
    never relabels velocity, wrench, or effort values as pose/qpos.
    """

    def __init__(
        self,
        *,
        receiver_id: str,
        component_target_capability: ComponentTargetCapabilitySpec,
        output_source: str = "action_arbiter",
        adapter_contract_id: str = ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_ID,
        adapter_contract_hash: str = ROBOT_ACTION_V1_TARGET_ADAPTER_CONTRACT_HASH,
        materializer: Callable[..., RobotAction] = materialize_robot_action_v1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(
            component_target_capability,
            ComponentTargetCapabilitySpec,
        ):
            raise TypeError(
                "component_target_capability must be a "
                "ComponentTargetCapabilitySpec."
            )
        if not callable(clock) or not callable(materializer):
            raise TypeError("clock and materializer must be callable.")
        self.receiver_id = _required(receiver_id, name="receiver_id")
        self._component_target_capability = component_target_capability
        self.output_source = _required(output_source, name="output_source")
        self.adapter_contract_id = _required(
            adapter_contract_id,
            name="adapter_contract_id",
        )
        self.adapter_contract_hash = _required(
            adapter_contract_hash,
            name="adapter_contract_hash",
        )
        self.materializer = materializer
        self.clock = clock
        routes = component_target_capability.component_routes
        if any(
            route.handler_contract_id
            != self.adapter_contract_id
            or route.handler_contract_hash
            != self.adapter_contract_hash
            for route in routes
        ):
            raise ValueError(
                "RobotAction adapter requires its exact declared materializer "
                "contract on every route."
            )
        output_hashes = {
            route.output_target_schema_hash for route in routes
        }
        if len(output_hashes) != 1:
            raise ValueError(
                "RobotAction v1 adapter requires one output action contract."
            )
        self.output_action_contract_hash = next(iter(output_hashes))

    @property
    def component_target_capability(self) -> ComponentTargetCapabilitySpec:
        return self._component_target_capability

    def adapt(
        self,
        request: ActionAdaptationRequest,
    ) -> ActionAdaptationResult:
        if not isinstance(request, ActionAdaptationRequest):
            raise TypeError("request must be an ActionAdaptationRequest.")
        if (
            request.component_target_capability_hash
            != self.component_target_capability.semantic_content_hash
            or request.component_target_capability_id
            != self.component_target_capability.capability_id
        ):
            raise ValueError(
                "Action adaptation request targets another capability."
            )
        started_at = float(self.clock())
        assessment = assess_component_target_capability(
            self.component_target_capability,
            request.candidate_action,
        )
        adapted_candidate: AdaptedActionCandidate | None = None
        if assessment.accepted:
            materialized = self.materializer(
                request.candidate_action,
                timestamp=request.target_effective_at,
                source=self.output_source,
            )
            if not isinstance(materialized, RobotAction):
                raise TypeError(
                    "RobotAction materializer must return a RobotAction."
                )
            action = freeze_robot_action(materialized)
            payload_semantic = robot_action_semantic_payload(action)
            completed_at = float(self.clock())
            candidate_payload = {
                "request_id": request.request_id,
                "request_hash": request.semantic_content_hash,
                "source_candidate_action_hash": request.candidate_action_hash,
                "payload_kind": "robot_action_v1",
                "output_action_contract_hash": (
                    self.output_action_contract_hash
                ),
                "adapter_contract_id": self.adapter_contract_id,
                "adapter_contract_hash": self.adapter_contract_hash,
                "payload_semantic": payload_semantic,
                "created_at": completed_at,
                "clock_domain_id": request.clock_domain_id,
            }
            candidate_hash = canonical_content_hash(
                candidate_payload,
                domain=ADAPTED_ACTION_CANDIDATE_HASH_DOMAIN_V1,
            )
            adapted_candidate = AdaptedActionCandidate(
                adapted_candidate_id=(
                    f"adapted-action-candidate:{candidate_hash}"
                ),
                runtime_payload=action,
                **candidate_payload,
                semantic_content_hash=candidate_hash,
            )
        else:
            completed_at = float(self.clock())
        receipt = build_action_adaptation_receipt(
            request=request,
            assessment=assessment,
            receiver_id=self.receiver_id,
            processing_started_at=started_at,
            processing_completed_at=completed_at,
            accepted=assessment.accepted,
            rejection_code=(
                None
                if assessment.accepted
                else "target_capability_rejected"
            ),
            adapted_candidate=adapted_candidate,
        )
        return ActionAdaptationResult(
            request=request,
            capability_assessment=assessment,
            adapted_candidate=adapted_candidate,
            receipt=receipt,
        )


def require_robot_action_v1(
    candidate: AdaptedActionCandidate,
) -> RobotAction:
    """Explicit compatibility edge into the existing shared safety path."""

    if not isinstance(candidate, AdaptedActionCandidate):
        raise TypeError("candidate must be an AdaptedActionCandidate.")
    if candidate.payload_kind != "robot_action_v1" or not isinstance(
        candidate.runtime_payload,
        RobotAction,
    ):
        raise TypeError(
            "The current shared safety path requires RobotAction v1."
        )
    return freeze_robot_action(candidate.runtime_payload)


__all__ = [
    "ACTION_ADAPTATION_RECEIPT_HASH_DOMAIN_V1",
    "ACTION_ADAPTATION_REQUEST_HASH_DOMAIN_V1",
    "ACTION_ADAPTATION_REQUEST_HASH_DOMAIN_V2",
    "ADAPTED_ACTION_CANDIDATE_HASH_DOMAIN_V1",
    "ActionAdaptationReceipt",
    "ActionAdaptationRequest",
    "ActionAdaptationResult",
    "ActionRepresentationAdapter",
    "AdaptedActionCandidate",
    "RobotActionV1RepresentationAdapter",
    "TARGET_CAPABILITY_ASSESSMENT_INSTANCE_HASH_DOMAIN_V1",
    "build_action_adaptation_request",
    "build_action_adaptation_receipt",
    "require_robot_action_v1",
    "target_capability_assessment_instance_id",
]
