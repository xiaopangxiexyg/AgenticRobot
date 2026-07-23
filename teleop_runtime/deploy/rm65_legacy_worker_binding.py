"""Default-off composition of the first RM65 legacy worker execution scope.

This module creates one shared owner/epoch context and exposes constructor
kwargs for the three required asynchronous workers.  It is deliberately
``fake_only_unbound`` and refuses hardware configuration; a later explicitly
authorized production composition must use a different maturity state.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from dataset_contracts.canonical_encoding import canonical_content_hash
from teleop_runtime.action import LegacyControlDomainExecutionContext
from teleop_runtime.backends.arms.rm65_worker_drain import (
    RM65ArmsOnlyWorkerDrainPlan,
)

from .rm65_execution_evidence import RM65ExecutionEvidenceBundle


RM65_LEGACY_WORKER_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_legacy_worker_binding.v1"
)


@dataclass(frozen=True)
class RM65LegacyWorkerExecutionBinding:
    binding_id: str
    evidence_bundle_hash: str
    context: LegacyControlDomainExecutionContext
    worker_drain_plan: RM65ArmsOnlyWorkerDrainPlan
    integration_status: str
    semantic_content_hash: str

    @property
    def production_binding_enabled(self) -> bool:
        return False

    @property
    def confers_execution_authorization(self) -> bool:
        return False

    def position_executor_kwargs(self) -> dict[str, object]:
        return {
            "execution_epoch_guard": self.context.execution_guard,
            "worker_drain_participant": self._participant(
                "position-deploy-reference-worker"
            ),
        }

    def arm_backend_kwargs(
        self,
        side: str,
        *,
        connect_hardware: bool,
    ) -> dict[str, object]:
        normalized = str(side).strip().lower()
        if normalized not in {"left", "right"}:
            raise ValueError("RM65 worker binding side must be left or right.")
        if bool(connect_hardware):
            raise RuntimeError(
                "fake-only RM65 worker binding cannot be attached to hardware"
            )
        return {
            "execution_epoch_guard": self.context.execution_guard,
            "worker_drain_participant": self._participant(
                f"rm65-{normalized}-background-stream-worker"
            ),
        }

    def _participant(self, participant_id: str):
        matches = tuple(
            binding.participant
            for binding in self.worker_drain_plan.participant_bindings
            if binding.participant.participant_id == participant_id
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"worker drain plan lacks exact participant {participant_id!r}"
            )
        return matches[0]


def build_fake_only_rm65_legacy_worker_binding(
    *,
    evidence_bundle: RM65ExecutionEvidenceBundle,
    runtime_session_id: str,
    control_domain_id: str = "rm65-arms",
    embodiment_instance_id: str = "rm65-bimanual-0",
    clock_domain_id: str = "control-monotonic",
    owner_runtime_id: str = "legacy-policy-frontend",
    initial_authority_epoch: int = 0,
    initial_execution_epoch: int = 0,
    clock: Callable[[], float] = time.monotonic,
) -> RM65LegacyWorkerExecutionBinding:
    """Build one fake-only context from the inseparable evidence bundle."""

    if not isinstance(evidence_bundle, RM65ExecutionEvidenceBundle):
        raise TypeError("evidence_bundle must be RM65ExecutionEvidenceBundle.")
    if evidence_bundle.production_binding_enabled:
        raise ValueError("Fake binding requires an unbound evidence bundle.")
    context = LegacyControlDomainExecutionContext.create_fake_only_unbound(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        clock_domain_id=clock_domain_id,
        owner_runtime_id=owner_runtime_id,
        execution_manifest=evidence_bundle.execution_manifest,
        dispatch_inventory=evidence_bundle.dispatch_inventory,
        inventory_assessment=evidence_bundle.inventory_assessment,
        initial_authority_epoch=initial_authority_epoch,
        initial_execution_epoch=initial_execution_epoch,
        clock=clock,
    )
    payload = {
        "binding_id": (
            f"rm65-legacy-worker-binding:{runtime_session_id}:{control_domain_id}"
        ),
        "evidence_bundle_hash": evidence_bundle.semantic_content_hash,
        "runtime_session_id": context.runtime_session_id,
        "control_domain_id": context.control_domain_id,
        "embodiment_instance_id": context.embodiment_instance_id,
        "execution_manifest_hash": context.execution_manifest.semantic_content_hash,
        "dispatch_inventory_hash": context.dispatch_inventory.semantic_content_hash,
        "worker_drain_plan_hash": (
            evidence_bundle.worker_drain_plan.semantic_content_hash
        ),
        "owner_token_id": context.legacy_owner_token.token_id,
        "integration_status": "fake_only_unbound",
        "production_binding_enabled": False,
        "confers_execution_authorization": False,
    }
    return RM65LegacyWorkerExecutionBinding(
        binding_id=payload["binding_id"],
        evidence_bundle_hash=evidence_bundle.semantic_content_hash,
        context=context,
        worker_drain_plan=evidence_bundle.worker_drain_plan,
        integration_status="fake_only_unbound",
        semantic_content_hash=canonical_content_hash(
            payload,
            domain=RM65_LEGACY_WORKER_BINDING_HASH_DOMAIN_V1,
        ),
    )


__all__ = [
    "RM65LegacyWorkerExecutionBinding",
    "build_fake_only_rm65_legacy_worker_binding",
]
