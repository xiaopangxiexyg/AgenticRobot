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
from typing import Callable
import time

from robot_motion.action.execution_manifest import ControlExecutionManifest
from robot_motion.runtime.cancellation import (
    ExecutionCancellation,
    ExecutionEpochCheckReceipt,
    ExecutionEpochGuard,
    ExecutionEpochRegistry,
    ExecutionWorkItem,
)
from robot_motion.runtime.dispatch_inventory import (
    DispatchSeamInventoryAssessment,
    DispatchSeamInventoryEntry,
    DispatchSeamInventoryManifest,
)
from robot_motion.runtime.dispatch_owner import (
    DispatchOwnerRegistry,
    DispatchOwnerToken,
)
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
    production_binding_enabled: bool
    hardware_authorization_id: str | None


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
        owner = self.dispatch_owners.snapshot()
        execution = self.execution_epochs.snapshot()
        token = owner.current_token
        return LegacyControlDomainExecutionContextSnapshot(
            runtime_session_id=self.runtime_session_id,
            control_domain_id=self.control_domain_id,
            embodiment_instance_id=self.embodiment_instance_id,
            clock_domain_id=self.clock_domain_id,
            execution_manifest_id=self.execution_manifest.manifest_id,
            execution_manifest_hash=self.execution_manifest.semantic_content_hash,
            dispatch_inventory_id=self.dispatch_inventory.inventory_id,
            dispatch_inventory_hash=self.dispatch_inventory.semantic_content_hash,
            composition_mode=self.composition_mode,
            current_owner_token_id=None if token is None else token.token_id,
            current_owner_kind=None if token is None else token.owner_kind,
            current_owner_generation=None if token is None else token.owner_generation,
            authority_epoch=owner.current_authority_epoch,
            execution_epoch=execution.execution_epoch,
            segment_generation_by_group=dict(execution.segment_generation_by_group),
            production_binding_enabled=self.production_binding_enabled,
            hardware_authorization_id=self.hardware_authorization_id,
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
            raise ValueError("Owner registry is not at the cancellation's previous epoch.")
        self.dispatch_owners.observe_execution_cancellation(cancellation)
        return self.snapshot()

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
