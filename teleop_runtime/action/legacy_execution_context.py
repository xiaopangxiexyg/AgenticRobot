"""Default-off composition prerequisite for legacy-owner dispatch mediation.

The context binds one execution manifest and one seam inventory to exactly one
dispatch-owner registry, legacy token, execution-epoch registry, and epoch
guard.  The general-purpose path remains ``fake_only_unbound``.  A second,
deliberately narrow mode exists only for the M5.8-b1 real idle-worker drain
preflight.  That mode binds real worker epoch guards but confers no motion or
lifecycle authority and must not be reused by canonical deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Callable
import time

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.execution_manifest import ControlExecutionManifest
from robot_motion.runtime.cancellation import (
    ExecutionCancellation,
    ExecutionCancellationRequest,
    ExecutionEpochCheckReceipt,
    ExecutionEpochGuard,
    ExecutionEpochRegistry,
    ExecutionWorkItem,
)
from robot_motion.position_control import ExecutionCapabilityProfile
from robot_motion.runtime.dispatch_inventory import (
    DispatchSeamInventoryAssessment,
    DispatchSeamInventoryEntry,
    DispatchSeamInventoryManifest,
)
from robot_motion.runtime.dispatch_owner import (
    DispatchOwnerRegistry,
    DispatchOwnerToken,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.execution_gate import (
    EXECUTION_GATE_STATUSES,
    ExecutionGateEventDraft,
    ExecutionGateStore,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.safety_operation import (
    RM65_QUICK_STOP_TICK_DISPATCH_OPERATION_V1,
    SafetyControlOperation,
    SafetyControlOperationRegistry,
    SafetyControlOperationTickEvidence,
)
from robot_motion.runtime.dispatch_seam import (
    DispatchPermitInvocation,
    DispatchSeamGate,
    DispatchSeamSpec,
)


LEGACY_EXECUTION_CONTEXT_MODES_V1 = frozenset(
    {
        "fake_only_unbound",
        "hardware_idle_drain_stop_only_preflight",
    }
)
LEGACY_CONTROL_OWNER_STATE_HASH_DOMAIN_V1 = (
    "dexteleop.legacy_control_owner_state.v1"
)
LEGACY_CONTROL_OWNER_TRANSITION_HASH_DOMAIN_V1 = (
    "dexteleop.legacy_control_owner_transition.v1"
)
LEGACY_REFERENCE_WORK_ITEM_HASH_DOMAIN_V1 = (
    "dexteleop.legacy_reference_work_item.v1"
)
LEGACY_CONTROL_LIFECYCLE_STATES_V1 = frozenset(
    {
        "configured",
        "connected",
        "preflighted",
        "shadow_ready",
        "armed",
        "running",
        "transitioning",
        "hold",
        "fault",
        "stopped",
    }
)
LEGACY_CONTROL_LIFECYCLE_GATE_COMPATIBILITY_V1 = {
    "configured": "hold",
    "connected": "hold",
    "preflighted": "hold",
    "shadow_ready": "hold",
    "armed": "hold",
    "running": "ready",
    "transitioning": "transitioning",
    "hold": "hold",
    "fault": "fault",
    "stopped": "hold",
}
_LEGACY_CONTROL_LIFECYCLE_TRANSITIONS_V1 = {
    "configured": frozenset(
        {
            "connected",
            "preflighted",
            "shadow_ready",
            "armed",
            "transitioning",
            "hold",
            "fault",
            "stopped",
        }
    ),
    "connected": frozenset(
        {
            "preflighted",
            "shadow_ready",
            "armed",
            "transitioning",
            "hold",
            "fault",
            "stopped",
        }
    ),
    "preflighted": frozenset(
        {"shadow_ready", "armed", "transitioning", "hold", "fault", "stopped"}
    ),
    "shadow_ready": frozenset(
        {"running", "transitioning", "hold", "fault", "stopped"}
    ),
    "armed": frozenset(
        {"running", "transitioning", "hold", "fault", "stopped"}
    ),
    "running": frozenset({"transitioning", "hold", "fault", "stopped"}),
    "transitioning": frozenset({"hold", "fault", "stopped"}),
    "hold": frozenset(
        {"shadow_ready", "armed", "running", "transitioning", "fault", "stopped"}
    ),
    "fault": frozenset({"hold", "stopped"}),
    "stopped": frozenset(),
}


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class LegacyMotionDispatchPreparation:
    """Fake-only preparation evidence adjacent to a pure final-call gate."""

    seam_id: str
    seam_inventory_hash: str
    work_item_id: str
    epoch_check: ExecutionEpochCheckReceipt
    invocation: DispatchPermitInvocation
    production_binding_enabled: bool = False


@dataclass(frozen=True)
class LegacySafetyDispatchPreparation:
    """Fake-only new-epoch quick-stop tick evidence and invocation."""

    seam_id: str
    seam_inventory_hash: str
    operation: SafetyControlOperation
    tick_evidence: SafetyControlOperationTickEvidence
    work_item: ExecutionWorkItem
    epoch_check: ExecutionEpochCheckReceipt
    invocation: DispatchPermitInvocation
    production_binding_enabled: bool = False


@dataclass(frozen=True)
class LegacyControlOwnerStateTransitionRequest:
    """Typed request submitted to the unique legacy control-state owner."""

    target_lifecycle_state: str
    target_execution_gate: str
    reason: str
    source_event_ids: tuple[str, ...] = ()
    recovery_authorization_id: str | None = None

    def __post_init__(self) -> None:
        lifecycle = str(self.target_lifecycle_state).strip().lower()
        gate = str(self.target_execution_gate).strip().lower()
        reason = str(self.reason).strip()
        if lifecycle not in LEGACY_CONTROL_LIFECYCLE_STATES_V1:
            raise ValueError("Unsupported legacy control lifecycle state.")
        if gate not in EXECUTION_GATE_STATUSES:
            raise ValueError("Unsupported legacy control execution gate.")
        if not reason:
            raise ValueError("Legacy control owner transition reason must be non-empty.")
        source_event_ids = tuple(
            _identifier(value, name="source_event_id")
            for value in self.source_event_ids
        )
        recovery_authorization_id = self.recovery_authorization_id
        if recovery_authorization_id is not None:
            recovery_authorization_id = _identifier(
                recovery_authorization_id,
                name="recovery_authorization_id",
            )
        object.__setattr__(self, "target_lifecycle_state", lifecycle)
        object.__setattr__(self, "target_execution_gate", gate)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "source_event_ids", source_event_ids)
        object.__setattr__(
            self,
            "recovery_authorization_id",
            recovery_authorization_id,
        )


@dataclass(frozen=True)
class LegacyControlOwnerStateTransitionReceipt:
    transition_id: str
    runtime_session_id: str
    control_domain_id: str
    sequence: int
    previous_lifecycle_state: str
    lifecycle_state: str
    previous_execution_gate: str
    execution_gate: str
    reason: str
    source_event_ids: tuple[str, ...]
    recovery_authorization_id: str | None
    transitioned_at: float
    clock_domain_id: str
    execution_gate_event_id: str
    execution_gate_event_hash: str
    semantic_content_hash: str


@dataclass(frozen=True)
class LegacyControlDomainExecutionContextSnapshot:
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    clock_domain_id: str
    execution_manifest_id: str
    execution_manifest_hash: str
    dispatch_inventory_id: str
    dispatch_inventory_hash: str
    composition_mode: str
    current_owner_token_id: str | None
    current_owner_kind: str | None
    current_owner_generation: int | None
    authority_epoch: int
    execution_epoch: int
    segment_generation_by_group: dict[str, int]
    lifecycle_state: str
    execution_gate: str
    execution_gate_epoch: int
    execution_gate_snapshot_hash: str
    owner_state_revision: int
    last_execution_cancellation_id: str | None
    last_execution_cancellation_hash: str | None
    last_owner_transition_id: str | None
    last_owner_transition_hash: str | None
    captured_at: float
    semantic_content_hash: str
    production_binding_enabled: bool
    hardware_authorization_id: str | None


@dataclass(frozen=True)
class LegacyControlOwnerStateTransitionOutcome:
    changed: bool
    receipt: LegacyControlOwnerStateTransitionReceipt | None
    snapshot: LegacyControlDomainExecutionContextSnapshot


class LegacyControlDomainExecutionContext:
    """Own one in-memory legacy execution scope without touching hardware."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        control_domain_id: str,
        embodiment_instance_id: str,
        clock_domain_id: str,
        execution_manifest: ControlExecutionManifest,
        dispatch_inventory: DispatchSeamInventoryManifest,
        inventory_assessment: DispatchSeamInventoryAssessment,
        dispatch_owners: DispatchOwnerRegistry,
        legacy_owner_token: DispatchOwnerToken,
        execution_epochs: ExecutionEpochRegistry,
        composition_mode: str,
        hardware_authorization_id: str | None = None,
        clock: Callable[[], float],
    ) -> None:
        if composition_mode not in LEGACY_EXECUTION_CONTEXT_MODES_V1:
            raise ValueError("Unsupported legacy execution context mode.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        if not isinstance(execution_manifest, ControlExecutionManifest):
            raise TypeError("execution_manifest must be a ControlExecutionManifest.")
        if not isinstance(dispatch_inventory, DispatchSeamInventoryManifest):
            raise TypeError("dispatch_inventory must be a DispatchSeamInventoryManifest.")
        if not isinstance(inventory_assessment, DispatchSeamInventoryAssessment):
            raise TypeError(
                "inventory_assessment must be a DispatchSeamInventoryAssessment."
            )
        if not isinstance(dispatch_owners, DispatchOwnerRegistry):
            raise TypeError("dispatch_owners must be a DispatchOwnerRegistry.")
        if not isinstance(legacy_owner_token, DispatchOwnerToken):
            raise TypeError("legacy_owner_token must be a DispatchOwnerToken.")
        if not isinstance(execution_epochs, ExecutionEpochRegistry):
            raise TypeError("execution_epochs must be an ExecutionEpochRegistry.")

        self.runtime_session_id = _identifier(
            runtime_session_id,
            name="runtime_session_id",
        )
        self.control_domain_id = _identifier(
            control_domain_id,
            name="control_domain_id",
        )
        self.embodiment_instance_id = _identifier(
            embodiment_instance_id,
            name="embodiment_instance_id",
        )
        self.clock_domain_id = _identifier(
            clock_domain_id,
            name="clock_domain_id",
        )
        self.execution_manifest = execution_manifest
        self.dispatch_inventory = dispatch_inventory
        self.inventory_assessment = inventory_assessment
        self.dispatch_owners = dispatch_owners
        self.legacy_owner_token = legacy_owner_token
        self.execution_epochs = execution_epochs
        self.execution_guard = ExecutionEpochGuard(execution_epochs, clock=clock)
        self.safety_operations = SafetyControlOperationRegistry(
            runtime_session_id=self.runtime_session_id,
            control_domain_id=self.control_domain_id,
            clock=clock,
        )
        self.composition_mode = composition_mode
        self.clock = clock
        if composition_mode == "fake_only_unbound":
            if hardware_authorization_id is not None:
                raise ValueError(
                    "fake_only_unbound context cannot carry hardware authorization"
                )
            self.production_binding_enabled = False
            self.hardware_authorization_id = None
        else:
            self.production_binding_enabled = True
            self.hardware_authorization_id = _identifier(
                hardware_authorization_id,
                name="hardware_authorization_id",
            )
        self._entry_by_seam_id = {
            entry.seam_id: entry for entry in dispatch_inventory.entries
        }
        self._owner_state_lock = threading.RLock()
        self._control_admission = ControlAdmissionCoordinator(
            self.control_domain_id,
            clock_domain=ClockDomainDescriptor(
                clock_domain_id=self.clock_domain_id,
                runtime_session_id=self.runtime_session_id,
                boot_id=f"legacy-control-context:{self.runtime_session_id}",
                reset_epoch=0,
            ),
            clock=self.clock,
        )
        self._execution_gate_store = ExecutionGateStore(
            runtime_session_id=self.runtime_session_id,
            coordinator=self._control_admission,
            initial_status="hold",
        )
        initial_gate_capture = self._control_admission.capture_cut(
            projectors={
                "execution_gate": lambda cut: self._execution_gate_store.build_snapshot(
                    cut=cut
                ),
            }
        )
        self._owner_execution_gate_snapshot = initial_gate_capture.projections[
            "execution_gate"
        ]
        self._owner_state_effective_at = initial_gate_capture.cut.cutoff_at
        self._lifecycle_state = "configured"
        self._owner_state_revision = 0
        self._next_owner_transition_sequence = 0
        self._last_execution_cancellation: ExecutionCancellation | None = None
        self._last_owner_transition: LegacyControlOwnerStateTransitionReceipt | None = None
        self._validate_scope()

    @classmethod
    def create_fake_only_unbound(
        cls,
        *,
        runtime_session_id: str,
        control_domain_id: str,
        embodiment_instance_id: str,
        clock_domain_id: str,
        owner_runtime_id: str,
        execution_manifest: ControlExecutionManifest,
        dispatch_inventory: DispatchSeamInventoryManifest,
        inventory_assessment: DispatchSeamInventoryAssessment,
        initial_authority_epoch: int = 0,
        initial_execution_epoch: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> "LegacyControlDomainExecutionContext":
        owners = DispatchOwnerRegistry(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            initial_authority_epoch=initial_authority_epoch,
            initial_execution_epoch=initial_execution_epoch,
            clock=clock,
        )
        legacy_token = owners.grant_initial_owner(
            owner_runtime_id=_identifier(owner_runtime_id, name="owner_runtime_id"),
            owner_kind="legacy_control_path",
        )
        epochs = ExecutionEpochRegistry(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            clock_domain_id=clock_domain_id,
            initial_execution_epoch=initial_execution_epoch,
            initial_segment_generation_by_group={
                group: 0 for group in execution_manifest.selection_groups
            },
            clock=clock,
        )
        return cls(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=clock_domain_id,
            execution_manifest=execution_manifest,
            dispatch_inventory=dispatch_inventory,
            inventory_assessment=inventory_assessment,
            dispatch_owners=owners,
            legacy_owner_token=legacy_token,
            execution_epochs=epochs,
            composition_mode="fake_only_unbound",
            clock=clock,
        )

    @classmethod
    def create_hardware_idle_drain_stop_only_preflight(
        cls,
        *,
        runtime_session_id: str,
        control_domain_id: str,
        embodiment_instance_id: str,
        clock_domain_id: str,
        owner_runtime_id: str,
        hardware_authorization_id: str,
        execution_manifest: ControlExecutionManifest,
        dispatch_inventory: DispatchSeamInventoryManifest,
        inventory_assessment: DispatchSeamInventoryAssessment,
        initial_authority_epoch: int = 0,
        initial_execution_epoch: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> "LegacyControlDomainExecutionContext":
        """Create the isolated M5.8-b1 real idle-worker preflight scope.

        The mode makes one owner/epoch registry available to explicitly
        authorized worker-drain instrumentation.  It does not authorize a
        target SDK call, home, rearm, force-position, or owner transition.
        """

        owners = DispatchOwnerRegistry(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            initial_authority_epoch=initial_authority_epoch,
            initial_execution_epoch=initial_execution_epoch,
            clock=clock,
        )
        legacy_token = owners.grant_initial_owner(
            owner_runtime_id=_identifier(owner_runtime_id, name="owner_runtime_id"),
            owner_kind="legacy_control_path",
        )
        epochs = ExecutionEpochRegistry(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            clock_domain_id=clock_domain_id,
            initial_execution_epoch=initial_execution_epoch,
            initial_segment_generation_by_group={
                group: 0 for group in execution_manifest.selection_groups
            },
            clock=clock,
        )
        return cls(
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
            embodiment_instance_id=embodiment_instance_id,
            clock_domain_id=clock_domain_id,
            execution_manifest=execution_manifest,
            dispatch_inventory=dispatch_inventory,
            inventory_assessment=inventory_assessment,
            dispatch_owners=owners,
            legacy_owner_token=legacy_token,
            execution_epochs=epochs,
            composition_mode="hardware_idle_drain_stop_only_preflight",
            hardware_authorization_id=hardware_authorization_id,
            clock=clock,
        )

    def snapshot(self) -> LegacyControlDomainExecutionContextSnapshot:
        with self._owner_state_lock:
            return self._snapshot_locked()

    def transition_owner_state(
        self,
        request: LegacyControlOwnerStateTransitionRequest,
    ) -> LegacyControlOwnerStateTransitionOutcome:
        """Atomically advance lifecycle and execution gate under one owner.

        The method performs no device, SDK, cancellation, HOLD, or dispatch call.
        Production callers must submit it only after their lower-owned control
        consequence has completed.  In particular, a FAULT-to-HOLD transition
        requires an explicit recovery authorization and never jumps to READY.
        """

        if not isinstance(request, LegacyControlOwnerStateTransitionRequest):
            raise TypeError("request must be a typed owner-state transition.")
        with self._owner_state_lock:
            previous_lifecycle = self._lifecycle_state
            previous_gate = self._current_execution_gate_locked()
            target_lifecycle = request.target_lifecycle_state
            target_gate = request.target_execution_gate
            expected_gate = LEGACY_CONTROL_LIFECYCLE_GATE_COMPATIBILITY_V1[
                target_lifecycle
            ]
            if target_gate != expected_gate:
                raise ValueError(
                    "Legacy lifecycle/execution-gate pair is inconsistent: "
                    f"{target_lifecycle!r} requires {expected_gate!r}."
                )
            if (
                previous_lifecycle == target_lifecycle
                and previous_gate == target_gate
            ):
                return LegacyControlOwnerStateTransitionOutcome(
                    changed=False,
                    receipt=None,
                    snapshot=self._snapshot_locked(),
                )
            if target_lifecycle not in _LEGACY_CONTROL_LIFECYCLE_TRANSITIONS_V1[
                previous_lifecycle
            ]:
                raise ValueError(
                    "Illegal legacy lifecycle transition: "
                    f"{previous_lifecycle!r} -> {target_lifecycle!r}."
                )
            if (
                previous_lifecycle == "fault"
                and target_lifecycle == "hold"
                and request.recovery_authorization_id is None
            ):
                raise ValueError(
                    "FAULT recovery requires explicit recovery authorization."
                )

            transitioned_at = float(self.clock())
            if not math.isfinite(transitioned_at):
                raise ValueError("Legacy control owner clock returned a non-finite value.")
            sequence = self._next_owner_transition_sequence
            gate_event = self._execution_gate_store.admit(
                ExecutionGateEventDraft(
                    runtime_session_id=self.runtime_session_id,
                    control_domain_id=self.control_domain_id,
                    sequence=sequence,
                    status=target_gate,
                    produced_at=transitioned_at,
                    produced_clock_domain_id=self.clock_domain_id,
                    reason=request.reason,
                    source_event_ids=request.source_event_ids,
                    audit_metadata={
                        "previous_lifecycle_state": previous_lifecycle,
                        "lifecycle_state": target_lifecycle,
                        "recovery_authorization_id": (
                            request.recovery_authorization_id
                        ),
                    },
                )
            )
            payload = {
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "sequence": sequence,
                "previous_lifecycle_state": previous_lifecycle,
                "lifecycle_state": target_lifecycle,
                "previous_execution_gate": previous_gate,
                "execution_gate": target_gate,
                "reason": request.reason,
                "source_event_ids": request.source_event_ids,
                "recovery_authorization_id": request.recovery_authorization_id,
                "transitioned_at": transitioned_at,
                "clock_domain_id": self.clock_domain_id,
                "execution_gate_event_id": gate_event.event_id,
                "execution_gate_event_hash": gate_event.semantic_content_hash,
            }
            transition_hash = canonical_content_hash(
                payload,
                domain=LEGACY_CONTROL_OWNER_TRANSITION_HASH_DOMAIN_V1,
            )
            receipt = LegacyControlOwnerStateTransitionReceipt(
                transition_id="legacy-control-owner-transition:" + transition_hash,
                **payload,
                semantic_content_hash=transition_hash,
            )
            self._lifecycle_state = target_lifecycle
            self._owner_state_revision += 1
            self._next_owner_transition_sequence += 1
            self._last_owner_transition = receipt
            gate_capture = self._control_admission.capture_cut(
                cutoff_at=gate_event.available_at,
                projectors={
                    "execution_gate": (
                        lambda cut: self._execution_gate_store.build_snapshot(cut=cut)
                    ),
                },
            )
            self._owner_execution_gate_snapshot = gate_capture.projections[
                "execution_gate"
            ]
            self._owner_state_effective_at = gate_capture.cut.cutoff_at
            return LegacyControlOwnerStateTransitionOutcome(
                changed=True,
                receipt=receipt,
                snapshot=self._snapshot_locked(),
            )

    def _current_execution_gate_locked(self) -> str:
        return self._owner_execution_gate_snapshot.status

    def _snapshot_locked(self) -> LegacyControlDomainExecutionContextSnapshot:
        gate = self._owner_execution_gate_snapshot
        owner = self.dispatch_owners.snapshot()
        execution = self.execution_epochs.snapshot()
        if owner.current_execution_epoch != execution.execution_epoch:
            raise RuntimeError(
                "Legacy owner/execution registries are between cancellation "
                "linearization points; no owner-state snapshot may be published."
            )
        token = owner.current_token
        last_cancellation = self._last_execution_cancellation
        last_transition = self._last_owner_transition
        payload = {
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "embodiment_instance_id": self.embodiment_instance_id,
            "clock_domain_id": self.clock_domain_id,
            "execution_manifest_id": self.execution_manifest.manifest_id,
            "execution_manifest_hash": self.execution_manifest.semantic_content_hash,
            "dispatch_inventory_id": self.dispatch_inventory.inventory_id,
            "dispatch_inventory_hash": self.dispatch_inventory.semantic_content_hash,
            "composition_mode": self.composition_mode,
            "current_owner_token_id": None if token is None else token.token_id,
            "current_owner_kind": None if token is None else token.owner_kind,
            "current_owner_generation": (
                None if token is None else token.owner_generation
            ),
            "authority_epoch": owner.current_authority_epoch,
            "execution_epoch": execution.execution_epoch,
            "segment_generation_by_group": dict(
                execution.segment_generation_by_group
            ),
            "lifecycle_state": self._lifecycle_state,
            "execution_gate": gate.status,
            "execution_gate_epoch": gate.gate_epoch,
            "execution_gate_snapshot_hash": gate.semantic_content_hash,
            "owner_state_revision": self._owner_state_revision,
            "last_execution_cancellation_id": (
                None
                if last_cancellation is None
                else last_cancellation.cancellation_id
            ),
            "last_execution_cancellation_hash": (
                None
                if last_cancellation is None
                else last_cancellation.semantic_content_hash
            ),
            "last_owner_transition_id": (
                None if last_transition is None else last_transition.transition_id
            ),
            "last_owner_transition_hash": (
                None
                if last_transition is None
                else last_transition.semantic_content_hash
            ),
            "captured_at": self._owner_state_effective_at,
            "production_binding_enabled": self.production_binding_enabled,
            "hardware_authorization_id": self.hardware_authorization_id,
        }
        return LegacyControlDomainExecutionContextSnapshot(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=LEGACY_CONTROL_OWNER_STATE_HASH_DOMAIN_V1,
            ),
        )

    def fake_motion_gate(self, seam_id: str) -> DispatchSeamGate:
        """Build a pure gate only for an inventoried final motion SDK seam."""

        entry = self._motion_final_entry(seam_id)
        return DispatchSeamGate(
            registry=self.dispatch_owners,
            spec=DispatchSeamSpec.create(
                seam_id=entry.seam_id,
                permit_kind="motion",
                call_stage="pre_sdk_call",
            ),
        )

    def create_reference_segment_work_item(
        self,
        *,
        source_decision_id: str,
        source_reference_segment_id: str,
    ) -> ExecutionWorkItem:
        """Mint work from the same owner/epoch cut used by every worker guard."""

        decision_id = _identifier(source_decision_id, name="source_decision_id")
        segment_id = _identifier(
            source_reference_segment_id,
            name="source_reference_segment_id",
        )
        with self._owner_state_lock:
            snapshot = self._snapshot_locked()
            if (
                snapshot.lifecycle_state != "running"
                or snapshot.execution_gate != "ready"
            ):
                raise RuntimeError(
                    "Reference work requires the unique owner to be RUNNING/READY."
                )
            if (
                snapshot.current_owner_token_id != self.legacy_owner_token.token_id
                or snapshot.current_owner_kind != "legacy_control_path"
            ):
                raise RuntimeError("Legacy reference work requires the current legacy owner.")
            payload = {
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "owner_state_hash": snapshot.semantic_content_hash,
                "owner_state_revision": snapshot.owner_state_revision,
                "authority_epoch": snapshot.authority_epoch,
                "execution_epoch": snapshot.execution_epoch,
                "segment_generation_by_group": dict(
                    snapshot.segment_generation_by_group
                ),
                "source_decision_id": decision_id,
                "source_reference_segment_id": segment_id,
            }
            work_hash = canonical_content_hash(
                payload,
                domain=LEGACY_REFERENCE_WORK_ITEM_HASH_DOMAIN_V1,
            )
            return ExecutionWorkItem(
                work_item_id="legacy-reference-work:" + work_hash,
                work_kind="reference_segment",
                control_domain_id=self.control_domain_id,
                authority_epoch=snapshot.authority_epoch,
                execution_epoch=snapshot.execution_epoch,
                segment_generation_by_group=dict(
                    snapshot.segment_generation_by_group
                ),
                source_decision_id=decision_id,
                source_reference_segment_id=segment_id,
            )

    def prepare_fake_motion_invocation(
        self,
        *,
        seam_id: str,
        work_item: ExecutionWorkItem,
        command_lineage_id: str,
        directive_lineage_id: str,
    ) -> LegacyMotionDispatchPreparation:
        """Check group/epoch evidence, then mint one fake-only one-shot permit.

        This method is not a production linearization point.  The real worker
        must eventually make its group/epoch check adjacent to the SDK call
        under the execution design accepted for M5.8-b.
        """

        entry = self._motion_final_entry(seam_id)
        if not isinstance(work_item, ExecutionWorkItem):
            raise TypeError("work_item must be an ExecutionWorkItem.")
        owner = self.dispatch_owners.snapshot()
        token = owner.current_token
        if token is None or token.token_id != self.legacy_owner_token.token_id:
            raise RuntimeError("Legacy owner token is no longer current.")
        if token.owner_kind != "legacy_control_path":
            raise RuntimeError("Current owner is not the legacy control path.")
        if work_item.authority_epoch != owner.current_authority_epoch:
            raise RuntimeError("Work authority epoch does not match the owner registry.")

        epoch_check = self.execution_guard.check_before_backend_call(work_item)
        if not epoch_check.accepted:
            raise RuntimeError(
                "Execution work item failed final fake-only epoch check: "
                f"{epoch_check.status}"
            )
        execution = self.execution_epochs.snapshot()
        if execution.execution_epoch != owner.current_execution_epoch:
            raise RuntimeError("Owner and execution registries disagree on execution epoch.")

        permit = self.dispatch_owners.issue_motion_permit(
            token=token,
            authority_epoch=owner.current_authority_epoch,
            execution_epoch=execution.execution_epoch,
            command_lineage_id=_identifier(command_lineage_id, name="command_lineage_id"),
            directive_lineage_id=_identifier(
                directive_lineage_id,
                name="directive_lineage_id",
            ),
        )
        return LegacyMotionDispatchPreparation(
            seam_id=entry.seam_id,
            seam_inventory_hash=entry.semantic_content_hash,
            work_item_id=work_item.work_item_id,
            epoch_check=epoch_check,
            invocation=DispatchPermitInvocation(
                permit=permit,
                current_authority_epoch=owner.current_authority_epoch,
                current_execution_epoch=execution.execution_epoch,
            ),
        )

    def observe_execution_cancellation(
        self,
        cancellation: ExecutionCancellation,
    ) -> LegacyControlDomainExecutionContextSnapshot:
        """Synchronize the owner gate after this context's epoch registry canceled.

        The caller must first create ``cancellation`` through this context's
        ``execution_epochs`` registry.  This method does not call a worker or
        backend and remains fake-only.
        """

        if not isinstance(cancellation, ExecutionCancellation):
            raise TypeError("cancellation must be an ExecutionCancellation.")
        with self._owner_state_lock:
            execution = self.execution_epochs.snapshot()
            owner = self.dispatch_owners.snapshot()
            if (
                cancellation.runtime_session_id != self.runtime_session_id
                or cancellation.control_domain_id != self.control_domain_id
            ):
                raise ValueError("Cancellation belongs to another execution context.")
            if (
                execution.last_cancellation_id != cancellation.cancellation_id
                or execution.execution_epoch != cancellation.execution_epoch
            ):
                raise ValueError(
                    "Cancellation was not linearized by this context's execution registry."
                )
            if owner.current_execution_epoch != cancellation.previous_execution_epoch:
                raise ValueError(
                    "Owner registry is not at the cancellation's previous epoch."
                )
            self.dispatch_owners.observe_execution_cancellation(cancellation)
            self._last_execution_cancellation = cancellation
            self._owner_state_revision += 1
            self._owner_state_effective_at = max(
                cancellation.effective_at,
                float(self.clock()),
            )
            self.transition_owner_state(
                LegacyControlOwnerStateTransitionRequest(
                    target_lifecycle_state="transitioning",
                    target_execution_gate="transitioning",
                    reason="execution_cancellation_worker_drain_pending",
                    source_event_ids=(cancellation.cancellation_id,),
                )
            )
            return self._snapshot_locked()

    def cancel_execution(
        self,
        request: ExecutionCancellationRequest,
        *,
        capability: ExecutionCapabilityProfile,
    ) -> ExecutionCancellation:
        """Atomically advance both epoch registries under the context lock.

        This only invalidates old-epoch work and closes the registry cut. It
        does not drain a worker, call HOLD, or change lifecycle/gate state.
        Those consequences require their own lower-owned evidence.
        """

        if not isinstance(request, ExecutionCancellationRequest):
            raise TypeError("request must be an ExecutionCancellationRequest.")
        if not isinstance(capability, ExecutionCapabilityProfile):
            raise TypeError("capability must be an ExecutionCapabilityProfile.")
        with self._owner_state_lock:
            cancellation = self.execution_epochs.cancel(
                request,
                capability=capability,
            )
            self.dispatch_owners.observe_execution_cancellation(cancellation)
            self._last_execution_cancellation = cancellation
            self._owner_state_revision += 1
            self._owner_state_effective_at = max(
                cancellation.effective_at,
                float(self.clock()),
            )
            self.transition_owner_state(
                LegacyControlOwnerStateTransitionRequest(
                    target_lifecycle_state="transitioning",
                    target_execution_gate="transitioning",
                    reason="execution_cancellation_worker_drain_pending",
                    source_event_ids=(cancellation.cancellation_id,),
                )
            )
            return cancellation

    def observe_fake_cancellation(
        self,
        cancellation: ExecutionCancellation,
    ) -> LegacyControlDomainExecutionContextSnapshot:
        """Compatibility alias for existing no-hardware fixtures."""

        return self.observe_execution_cancellation(cancellation)

    def admit_fake_rm65_quick_stop(
        self,
        *,
        cancellation: ExecutionCancellation,
        operation_id: str,
        frozen_stop_target_hash: str,
        maximum_duration_sec: float,
        maximum_tick_count: int,
    ) -> SafetyControlOperation:
        """Admit one bounded quick-stop operation in the cancellation's new epoch."""

        if not isinstance(cancellation, ExecutionCancellation):
            raise TypeError("cancellation must be an ExecutionCancellation.")
        owner = self.dispatch_owners.snapshot()
        execution = self.execution_epochs.snapshot()
        if (
            cancellation.runtime_session_id != self.runtime_session_id
            or cancellation.control_domain_id != self.control_domain_id
            or cancellation.cancellation_id != execution.last_cancellation_id
            or cancellation.execution_epoch != execution.execution_epoch
            or cancellation.execution_epoch != owner.current_execution_epoch
        ):
            raise ValueError(
                "Quick-stop operation requires one synchronized new execution epoch."
            )
        if not cancellation.requests_backend_hold:
            raise ValueError("RM65 quick-stop requires a control-domain backend hold.")
        operation = SafetyControlOperation.create_rm65_quick_stop(
            operation_id=operation_id,
            runtime_session_id=self.runtime_session_id,
            control_domain_id=self.control_domain_id,
            safety_source="execution_cancellation_runtime",
            source_event_id=cancellation.cancellation_id,
            target_execution_epoch=cancellation.execution_epoch,
            affected_selection_groups=tuple(self.execution_manifest.selection_groups),
            frozen_stop_target_hash=frozen_stop_target_hash,
            maximum_duration_sec=maximum_duration_sec,
            maximum_tick_count=maximum_tick_count,
            authorized_at=self.clock(),
        )
        self.safety_operations.admit(operation)
        return operation

    def fake_safety_gate(self, seam_id: str) -> DispatchSeamGate:
        entry = self._safety_final_entry(seam_id)
        return DispatchSeamGate(
            registry=self.dispatch_owners,
            spec=DispatchSeamSpec.create(
                seam_id=entry.seam_id,
                permit_kind="safety_control",
                operation=RM65_QUICK_STOP_TICK_DISPATCH_OPERATION_V1,
                call_stage="pre_sdk_call",
            ),
        )

    def prepare_fake_rm65_quick_stop_tick(
        self,
        *,
        seam_id: str,
        operation_id: str,
        tick_sequence: int,
        command_lineage_id: str,
    ) -> LegacySafetyDispatchPreparation:
        """Mint new work/evidence/permit for exactly one fake quick-stop tick."""

        entry = self._safety_final_entry(seam_id)
        operation = self.safety_operations.active_operation()
        if operation.operation_id != str(operation_id):
            raise ValueError("Requested quick-stop operation is not active.")
        owner = self.dispatch_owners.snapshot()
        execution = self.execution_epochs.snapshot()
        if owner.current_execution_epoch != execution.execution_epoch:
            raise RuntimeError("Owner and execution registries disagree on execution epoch.")
        tick = self.safety_operations.admit_tick(
            operation_id=operation.operation_id,
            tick_sequence=tick_sequence,
            current_execution_epoch=execution.execution_epoch,
            command_lineage_id=command_lineage_id,
        )
        work = tick.as_execution_work_item(
            operation,
            authority_epoch=owner.current_authority_epoch,
            segment_generation_by_group=dict(
                execution.segment_generation_by_group
            ),
        )
        epoch_check = self.execution_guard.check_before_backend_call(work)
        if not epoch_check.accepted:
            raise RuntimeError(
                "Quick-stop work failed final fake-only epoch check: "
                f"{epoch_check.status}"
            )
        permit = self.dispatch_owners.issue_safety_control_permit(
            operation=operation.allowed_dispatch_operation,
            evidence=tick.as_dispatch_evidence(operation),
            safety_source=operation.safety_source,
        )
        return LegacySafetyDispatchPreparation(
            seam_id=entry.seam_id,
            seam_inventory_hash=entry.semantic_content_hash,
            operation=operation,
            tick_evidence=tick,
            work_item=work,
            epoch_check=epoch_check,
            invocation=DispatchPermitInvocation(
                permit=permit,
                current_authority_epoch=owner.current_authority_epoch,
                current_execution_epoch=execution.execution_epoch,
            ),
        )

    def _validate_scope(self) -> None:
        if (
            self.dispatch_inventory.execution_manifest_id
            != self.execution_manifest.manifest_id
            or self.dispatch_inventory.execution_manifest_hash
            != self.execution_manifest.semantic_content_hash
        ):
            raise ValueError("Dispatch inventory is bound to another execution manifest.")
        if (
            self.inventory_assessment.inventory_id
            != self.dispatch_inventory.inventory_id
            or self.inventory_assessment.inventory_hash
            != self.dispatch_inventory.semantic_content_hash
        ):
            raise ValueError("Inventory assessment does not describe this inventory.")
        if not self.inventory_assessment.inventory_structurally_complete:
            raise ValueError("Legacy execution context requires a structurally complete inventory.")
        if self.inventory_assessment.confers_execution_authorization:
            raise ValueError("Inventory assessment must not confer execution authorization.")

        owner = self.dispatch_owners.snapshot()
        execution = self.execution_epochs.snapshot()
        if (
            owner.runtime_session_id != self.runtime_session_id
            or owner.control_domain_id != self.control_domain_id
            or self.execution_epochs.runtime_session_id != self.runtime_session_id
            or self.execution_epochs.control_domain_id != self.control_domain_id
            or self.execution_epochs.clock_domain_id != self.clock_domain_id
        ):
            raise ValueError("Legacy execution registries belong to another runtime scope.")
        if owner.current_token is None:
            raise ValueError("Legacy execution context requires one current owner token.")
        if (
            owner.current_token.token_id != self.legacy_owner_token.token_id
            or owner.current_token.semantic_content_hash
            != self.legacy_owner_token.semantic_content_hash
            or owner.current_token.owner_kind != "legacy_control_path"
        ):
            raise ValueError("Injected legacy owner token is not current.")
        if owner.current_execution_epoch != execution.execution_epoch:
            raise ValueError("Owner and execution registries disagree on initial epoch.")
        missing_groups = set(self.execution_manifest.selection_groups).difference(
            execution.segment_generation_by_group
        )
        if missing_groups:
            raise ValueError(
                "Execution registry lacks manifest selection groups: "
                f"{sorted(missing_groups)!r}."
            )

    def _motion_final_entry(self, seam_id: str) -> DispatchSeamInventoryEntry:
        normalized = _identifier(seam_id, name="seam_id")
        try:
            entry = self._entry_by_seam_id[normalized]
        except KeyError as exc:
            raise KeyError(f"Dispatch seam {normalized!r} is not inventoried.") from exc
        if (
            entry.permit_kind != "motion"
            or entry.call_stage != "pre_sdk_call"
            or entry.authorization_requirement != "one_shot_dispatch_permit"
        ):
            raise ValueError(
                "Fake motion invocation requires an inventoried one-shot pre-SDK motion seam."
            )
        return entry

    def _safety_final_entry(self, seam_id: str) -> DispatchSeamInventoryEntry:
        normalized = _identifier(seam_id, name="seam_id")
        try:
            entry = self._entry_by_seam_id[normalized]
        except KeyError as exc:
            raise KeyError(f"Dispatch seam {normalized!r} is not inventoried.") from exc
        if (
            entry.permit_kind != "safety_control"
            or entry.call_stage != "pre_sdk_call"
            or entry.typed_operation
            != RM65_QUICK_STOP_TICK_DISPATCH_OPERATION_V1
            or entry.authorization_requirement
            != "bounded_operation_then_one_shot"
        ):
            raise ValueError(
                "Fake quick-stop invocation requires the exact inventoried "
                "bounded one-shot pre-SDK safety seam."
            )
        return entry
