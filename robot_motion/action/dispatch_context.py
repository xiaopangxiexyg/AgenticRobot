"""Immutable authoritative context passed to an execution receiver.

The context carries control lineage to the final receiver without claiming that
the receiver enforced it.  A reference worker must still compare the supplied
execution epoch and group generations with its live registries before it may
produce a canonical admission receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from .proposal import freeze_robot_action, robot_action_semantic_payload
from .schema import RobotAction
from .execution_handoff import (
    TargetExecutionRequest,
    TargetExecutionSubmissionReceipt,
)


AUTHORITATIVE_DISPATCH_CONTEXT_HASH_DOMAIN_V1 = (
    "dexteleop.authoritative_dispatch_context.v1"
)
AUTHORITATIVE_TARGET_DISPATCH_CONTEXT_HASH_DOMAIN = (
    "dexteleop.authoritative_target_dispatch_context.development"
)
AUTHORITATIVE_RECEIVER_STATE_HASH_DOMAIN_V1 = (
    "dexteleop.authoritative_receiver_state.v1"
)
AUTHORITATIVE_RECEIVER_VALIDATION_HASH_DOMAIN_V1 = (
    "dexteleop.authoritative_receiver_validation.v1"
)
COMPOUND_DISPATCH_MANIFEST_HASH_DOMAIN_V1 = (
    "dexteleop.compound_dispatch_manifest.v1"
)


def _required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


@dataclass(frozen=True)
class AuthoritativeCommandDispatchContext:
    """Lineage available at the final authoritative dispatch call.

    ``*_by_group`` values describe the directive being submitted.  They are
    not proof that a downstream worker checked the values; that proof belongs
    to the receiver's admission receipt.
    """

    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    decision_id: str
    directive_id: str
    safety_result_id: str
    command_id: str
    dispatch_permit_id: str
    owner_token_id: str
    owner_generation: int
    target_effective_at: float
    authority_epoch: int
    execution_epoch: int
    group_segment_generation_by_group: Mapping[str, int]
    selected_action_ref_by_group: Mapping[str, str]
    safe_action: RobotAction
    safe_action_hash: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "decision_id",
            "directive_id",
            "safety_result_id",
            "command_id",
            "dispatch_permit_id",
            "owner_token_id",
            "safe_action_hash",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        target = float(self.target_effective_at)
        if not math.isfinite(target):
            raise ValueError("target_effective_at must be finite.")
        generations = {
            _required_text(group, name="selection_group_id"): _non_negative_int(
                generation,
                name="group_segment_generation",
            )
            for group, generation in self.group_segment_generation_by_group.items()
        }
        selected = {
            _required_text(group, name="selection_group_id"): _required_text(
                proposal_id,
                name="selected_action_ref",
            )
            for group, proposal_id in self.selected_action_ref_by_group.items()
        }
        if not generations or set(generations) != set(selected):
            raise ValueError(
                "Dispatch context generations and selected refs must cover the same groups."
            )
        action = freeze_robot_action(self.safe_action)
        action_hash = canonical_content_hash(
            robot_action_semantic_payload(action),
            domain="dexteleop.robot_action_record.v1",
        )
        if action_hash != self.safe_action_hash:
            raise ValueError("Dispatch context safe-action hash mismatch.")
        object.__setattr__(self, "target_effective_at", target)
        object.__setattr__(
            self,
            "owner_generation",
            _non_negative_int(self.owner_generation, name="owner_generation"),
        )
        object.__setattr__(
            self,
            "authority_epoch",
            _non_negative_int(self.authority_epoch, name="authority_epoch"),
        )
        object.__setattr__(
            self,
            "execution_epoch",
            _non_negative_int(self.execution_epoch, name="execution_epoch"),
        )
        object.__setattr__(
            self,
            "group_segment_generation_by_group",
            MappingProxyType(dict(sorted(generations.items()))),
        )
        object.__setattr__(
            self,
            "selected_action_ref_by_group",
            MappingProxyType(dict(sorted(selected.items()))),
        )
        object.__setattr__(self, "safe_action", action)


@dataclass(frozen=True)
class AuthoritativeTargetDispatchContext:
    """Permit-bound context for the component-keyed target handoff.

    Unlike :class:`AuthoritativeCommandDispatchContext`, this context carries
    no universal ``SafeAction`` or safety-result claim.  It binds the exact
    accepted target-processing submission to the sole motion permit and to the
    group generations that the final receiver must re-check.
    """

    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    decision_id: str
    directive_id: str
    command_id: str
    dispatch_permit_id: str
    owner_token_id: str
    owner_generation: int
    target_effective_at: float
    authority_sequence: int
    execution_fence_sequence: int
    group_segment_generation_by_group: Mapping[str, int]
    selected_action_ref_by_group: Mapping[str, str]
    component_coverage: tuple[str, ...]
    target_execution_request: TargetExecutionRequest = field(
        repr=False,
        compare=False,
    )
    submission_receipt: TargetExecutionSubmissionReceipt = field(
        repr=False,
        compare=False,
    )
    semantic_content_hash: str = ""

    def __post_init__(self) -> None:
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "decision_id",
            "directive_id",
            "command_id",
            "dispatch_permit_id",
            "owner_token_id",
            "semantic_content_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name=name),
            )
        request = self.target_execution_request
        receipt = self.submission_receipt
        if not isinstance(request, TargetExecutionRequest):
            raise TypeError(
                "target_execution_request must be a TargetExecutionRequest."
            )
        if not isinstance(receipt, TargetExecutionSubmissionReceipt):
            raise TypeError(
                "submission_receipt must be a "
                "TargetExecutionSubmissionReceipt."
            )
        if (
            not receipt.accepted_for_processing
            or receipt.target_execution_request_id != request.request_id
            or receipt.target_execution_request_hash
            != request.semantic_content_hash
            or request.control_domain_id != self.control_domain_id
            or request.adaptation_result.request.authoritative_decision_id
            != self.decision_id
        ):
            raise ValueError(
                "Target dispatch context requires the accepted receipt for "
                "this exact authoritative request."
            )
        target = float(self.target_effective_at)
        if not math.isfinite(target):
            raise ValueError("target_effective_at must be finite.")
        if abs(target - request.adaptation_result.request.target_effective_at) > 1e-12:
            raise ValueError("Target dispatch context effective time mismatch.")
        authority_sequence = _non_negative_int(
            self.authority_sequence,
            name="authority_sequence",
        )
        execution_fence_sequence = _non_negative_int(
            self.execution_fence_sequence,
            name="execution_fence_sequence",
        )
        if (
            authority_sequence
            != request.adaptation_result.request.authority_sequence
            or execution_fence_sequence
            != request.adaptation_result.request.execution_fence_sequence
        ):
            raise ValueError("Target dispatch context sequence lineage mismatch.")
        generations = {
            _required_text(group, name="selection_group_id"): _non_negative_int(
                generation,
                name="group_segment_generation",
            )
            for group, generation in self.group_segment_generation_by_group.items()
        }
        selected = {
            _required_text(group, name="selection_group_id"): _required_text(
                selected_ref,
                name="selected_action_ref",
            )
            for group, selected_ref in self.selected_action_ref_by_group.items()
        }
        if not generations or set(generations) != set(selected):
            raise ValueError(
                "Target dispatch generations and selected refs must cover the "
                "same groups."
            )
        coverage = tuple(
            _required_text(value, name="component_id")
            for value in self.component_coverage
        )
        if (
            not coverage
            or len(coverage) != len(set(coverage))
            or coverage != request.adaptation_result.request.component_coverage
        ):
            raise ValueError(
                "Target dispatch component coverage must equal the request."
            )
        object.__setattr__(self, "target_effective_at", target)
        object.__setattr__(
            self,
            "owner_generation",
            _non_negative_int(self.owner_generation, name="owner_generation"),
        )
        object.__setattr__(self, "authority_sequence", authority_sequence)
        object.__setattr__(
            self,
            "execution_fence_sequence",
            execution_fence_sequence,
        )
        object.__setattr__(
            self,
            "group_segment_generation_by_group",
            MappingProxyType(dict(sorted(generations.items()))),
        )
        object.__setattr__(
            self,
            "selected_action_ref_by_group",
            MappingProxyType(dict(sorted(selected.items()))),
        )
        object.__setattr__(self, "component_coverage", coverage)


@dataclass(frozen=True)
class AuthoritativeReceiverStateSnapshot:
    """Live state sampled inside a receiver admission critical section."""

    runtime_session_id: str
    control_domain_id: str
    current_owner_token_id: str | None
    current_owner_generation: int | None
    current_authority_epoch: int
    current_execution_epoch: int
    expected_next_group_generation_by_group: Mapping[str, int]
    captured_at: float
    process_scope: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "process_scope",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        if self.current_owner_token_id is not None:
            _required_text(self.current_owner_token_id, name="current_owner_token_id")
        if self.current_owner_generation is not None:
            object.__setattr__(
                self,
                "current_owner_generation",
                _non_negative_int(
                    self.current_owner_generation,
                    name="current_owner_generation",
                ),
            )
        for name in ("current_authority_epoch", "current_execution_epoch"):
            object.__setattr__(
                self,
                name,
                _non_negative_int(getattr(self, name), name=name),
            )
        generations = {
            _required_text(group, name="selection_group_id"): _non_negative_int(
                generation,
                name="expected_next_group_generation",
            )
            for group, generation in self.expected_next_group_generation_by_group.items()
        }
        object.__setattr__(
            self,
            "expected_next_group_generation_by_group",
            MappingProxyType(dict(sorted(generations.items()))),
        )
        captured = float(self.captured_at)
        if not math.isfinite(captured):
            raise ValueError("captured_at must be finite.")
        object.__setattr__(self, "captured_at", captured)


@dataclass(frozen=True)
class AuthoritativeReceiverValidationReceipt:
    validation_id: str
    dispatch_context_hash: str
    receiver_state_hash: str
    accepted: bool
    status: str
    checked_at: float
    semantic_content_hash: str


@dataclass(frozen=True)
class CompoundDispatchGroupBinding:
    """Exact selected fact and generation for one compound selection group."""

    selection_group_id: str
    selected_action_ref: str
    group_segment_generation: int
    component_dependencies: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selection_group_id",
            _required_text(self.selection_group_id, name="selection_group_id"),
        )
        object.__setattr__(
            self,
            "selected_action_ref",
            _required_text(self.selected_action_ref, name="selected_action_ref"),
        )
        object.__setattr__(
            self,
            "group_segment_generation",
            _non_negative_int(
                self.group_segment_generation,
                name="group_segment_generation",
            ),
        )
        dependencies = tuple(
            _required_text(value, name="component_dependency")
            for value in self.component_dependencies
        )
        if not dependencies or len(set(dependencies)) != len(dependencies):
            raise ValueError(
                "component_dependencies must be non-empty and unique per group."
            )
        object.__setattr__(self, "component_dependencies", dependencies)


@dataclass(frozen=True)
class CompoundDispatchManifest:
    """Immutable exact-group bundle for one authoritative backend submission.

    This fact does not add authority.  It freezes how the already-authorized
    dispatch context's independently selected groups are joined into one
    backend command at one target time.
    """

    manifest_id: str
    dispatch_context_hash: str
    control_domain_id: str
    command_id: str
    safety_result_id: str
    safe_action_hash: str
    target_effective_at: float
    authority_epoch: int
    execution_epoch: int
    group_binding_by_id: Mapping[str, CompoundDispatchGroupBinding]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "manifest_id",
            "dispatch_context_hash",
            "control_domain_id",
            "command_id",
            "safety_result_id",
            "safe_action_hash",
            "semantic_content_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required_text(getattr(self, name), name=name),
            )
        target = float(self.target_effective_at)
        if not math.isfinite(target):
            raise ValueError("target_effective_at must be finite.")
        bindings: dict[str, CompoundDispatchGroupBinding] = {}
        all_components: list[str] = []
        for raw_group, raw_binding in self.group_binding_by_id.items():
            group = _required_text(raw_group, name="selection_group_id")
            if not isinstance(raw_binding, CompoundDispatchGroupBinding):
                raise TypeError(
                    "group_binding_by_id values must be "
                    "CompoundDispatchGroupBinding."
                )
            if raw_binding.selection_group_id != group:
                raise ValueError("Compound group key/binding mismatch.")
            bindings[group] = raw_binding
            all_components.extend(raw_binding.component_dependencies)
        if len(bindings) < 2:
            raise ValueError("A compound dispatch manifest requires multiple groups.")
        if len(all_components) != len(set(all_components)):
            raise ValueError("A component cannot belong to multiple compound groups.")
        object.__setattr__(self, "target_effective_at", target)
        object.__setattr__(
            self,
            "authority_epoch",
            _non_negative_int(self.authority_epoch, name="authority_epoch"),
        )
        object.__setattr__(
            self,
            "execution_epoch",
            _non_negative_int(self.execution_epoch, name="execution_epoch"),
        )
        object.__setattr__(
            self,
            "group_binding_by_id",
            MappingProxyType(dict(sorted(bindings.items()))),
        )
        semantic_payload = {
            "dispatch_context_hash": self.dispatch_context_hash,
            "control_domain_id": self.control_domain_id,
            "command_id": self.command_id,
            "safety_result_id": self.safety_result_id,
            "safe_action_hash": self.safe_action_hash,
            "target_effective_at": target,
            "authority_epoch": self.authority_epoch,
            "execution_epoch": self.execution_epoch,
            "group_binding_by_id": {
                group: {
                    "selection_group_id": binding.selection_group_id,
                    "selected_action_ref": binding.selected_action_ref,
                    "group_segment_generation": (
                        binding.group_segment_generation
                    ),
                    "component_dependencies": binding.component_dependencies,
                }
                for group, binding in sorted(bindings.items())
            },
        }
        expected_hash = canonical_content_hash(
            semantic_payload,
            domain=COMPOUND_DISPATCH_MANIFEST_HASH_DOMAIN_V1,
        )
        if expected_hash != self.semantic_content_hash:
            raise ValueError("Compound dispatch manifest hash mismatch.")
        if self.manifest_id != f"compound-dispatch-manifest:{expected_hash}":
            raise ValueError("Compound dispatch manifest ID/hash mismatch.")

    @property
    def component_dependencies(self) -> tuple[str, ...]:
        return tuple(
            component
            for binding in self.group_binding_by_id.values()
            for component in binding.component_dependencies
        )

    @property
    def group_segment_generation_by_group(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                group: binding.group_segment_generation
                for group, binding in self.group_binding_by_id.items()
            }
        )


def build_compound_dispatch_manifest(
    context: AuthoritativeCommandDispatchContext,
    *,
    component_dependencies_by_group: Mapping[str, tuple[str, ...]],
) -> CompoundDispatchManifest:
    """Freeze an exact multi-group join already present in a dispatch context."""

    if not isinstance(context, AuthoritativeCommandDispatchContext):
        raise TypeError("context must be AuthoritativeCommandDispatchContext.")
    dependencies = {
        _required_text(group, name="selection_group_id"): tuple(values)
        for group, values in component_dependencies_by_group.items()
    }
    context_groups = set(context.group_segment_generation_by_group)
    if set(dependencies) != context_groups:
        raise ValueError(
            "Compound dependencies must exactly cover dispatch context groups."
        )
    bindings = {
        group: CompoundDispatchGroupBinding(
            selection_group_id=group,
            selected_action_ref=context.selected_action_ref_by_group[group],
            group_segment_generation=(
                context.group_segment_generation_by_group[group]
            ),
            component_dependencies=dependencies[group],
        )
        for group in sorted(context_groups)
    }
    covered = {
        component
        for binding in bindings.values()
        for component in binding.component_dependencies
    }
    if covered != set(context.safe_action.active_effectors):
        raise ValueError(
            "Compound group components must exactly cover safe-action effectors."
        )
    payload = {
        "dispatch_context_hash": context.semantic_content_hash,
        "control_domain_id": context.control_domain_id,
        "command_id": context.command_id,
        "safety_result_id": context.safety_result_id,
        "safe_action_hash": context.safe_action_hash,
        "target_effective_at": context.target_effective_at,
        "authority_epoch": context.authority_epoch,
        "execution_epoch": context.execution_epoch,
        "group_binding_by_id": {
            group: {
                "selection_group_id": binding.selection_group_id,
                "selected_action_ref": binding.selected_action_ref,
                "group_segment_generation": binding.group_segment_generation,
                "component_dependencies": binding.component_dependencies,
            }
            for group, binding in bindings.items()
        },
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=COMPOUND_DISPATCH_MANIFEST_HASH_DOMAIN_V1,
    )
    return CompoundDispatchManifest(
        manifest_id=f"compound-dispatch-manifest:{semantic_hash}",
        **{
            key: value
            for key, value in payload.items()
            if key != "group_binding_by_id"
        },
        group_binding_by_id=bindings,
        semantic_content_hash=semantic_hash,
    )


def validate_compound_dispatch_manifest(
    manifest: CompoundDispatchManifest,
    context: AuthoritativeCommandDispatchContext,
) -> None:
    """Fail closed unless a manifest is an exact projection of the context."""

    if not isinstance(manifest, CompoundDispatchManifest):
        raise TypeError("manifest must be a CompoundDispatchManifest.")
    rebuilt = build_compound_dispatch_manifest(
        context,
        component_dependencies_by_group={
            group: binding.component_dependencies
            for group, binding in manifest.group_binding_by_id.items()
        },
    )
    if (
        rebuilt.manifest_id != manifest.manifest_id
        or rebuilt.semantic_content_hash != manifest.semantic_content_hash
    ):
        raise ValueError("Compound dispatch manifest does not match context.")


def build_authoritative_receiver_state_snapshot(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    current_owner_token_id: str | None,
    current_owner_generation: int | None,
    current_authority_epoch: int,
    current_execution_epoch: int,
    expected_next_group_generation_by_group: Mapping[str, int],
    captured_at: float,
    process_scope: str = "in_process_only",
) -> AuthoritativeReceiverStateSnapshot:
    payload = {
        "runtime_session_id": _required_text(
            runtime_session_id, name="runtime_session_id"
        ),
        "control_domain_id": _required_text(
            control_domain_id, name="control_domain_id"
        ),
        "current_owner_token_id": (
            None
            if current_owner_token_id is None
            else _required_text(current_owner_token_id, name="current_owner_token_id")
        ),
        "current_owner_generation": (
            None
            if current_owner_generation is None
            else _non_negative_int(
                current_owner_generation, name="current_owner_generation"
            )
        ),
        "current_authority_epoch": _non_negative_int(
            current_authority_epoch, name="current_authority_epoch"
        ),
        "current_execution_epoch": _non_negative_int(
            current_execution_epoch, name="current_execution_epoch"
        ),
        "expected_next_group_generation_by_group": {
            str(group): int(generation)
            for group, generation in sorted(
                expected_next_group_generation_by_group.items()
            )
        },
        "captured_at": float(captured_at),
        "process_scope": _required_text(process_scope, name="process_scope"),
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=AUTHORITATIVE_RECEIVER_STATE_HASH_DOMAIN_V1,
    )
    return AuthoritativeReceiverStateSnapshot(
        **payload,
        semantic_content_hash=semantic_hash,
    )


def validate_authoritative_receiver_context(
    context: AuthoritativeCommandDispatchContext,
    state: AuthoritativeReceiverStateSnapshot,
) -> AuthoritativeReceiverValidationReceipt:
    """Compare supplied lineage with receiver-observed live state."""

    if not isinstance(context, AuthoritativeCommandDispatchContext):
        raise TypeError("context must be AuthoritativeCommandDispatchContext.")
    if not isinstance(state, AuthoritativeReceiverStateSnapshot):
        raise TypeError("state must be AuthoritativeReceiverStateSnapshot.")
    status = "accepted"
    if context.runtime_session_id != state.runtime_session_id:
        status = "runtime_session_mismatch"
    elif context.control_domain_id != state.control_domain_id:
        status = "control_domain_mismatch"
    elif context.owner_token_id != state.current_owner_token_id:
        status = "owner_token_mismatch"
    elif context.owner_generation != state.current_owner_generation:
        status = "owner_generation_mismatch"
    elif context.authority_epoch != state.current_authority_epoch:
        status = "authority_epoch_mismatch"
    elif context.execution_epoch != state.current_execution_epoch:
        status = "execution_epoch_mismatch"
    else:
        for group, generation in context.group_segment_generation_by_group.items():
            if state.expected_next_group_generation_by_group.get(group) != generation:
                status = f"group_segment_generation_mismatch:{group}"
                break
    payload = {
        "dispatch_context_hash": context.semantic_content_hash,
        "receiver_state_hash": state.semantic_content_hash,
        "accepted": status == "accepted",
        "status": status,
        "checked_at": state.captured_at,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=AUTHORITATIVE_RECEIVER_VALIDATION_HASH_DOMAIN_V1,
    )
    return AuthoritativeReceiverValidationReceipt(
        validation_id=f"authoritative-receiver-validation:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


def validate_authoritative_target_receiver_context(
    context: AuthoritativeTargetDispatchContext,
    state: AuthoritativeReceiverStateSnapshot,
) -> AuthoritativeReceiverValidationReceipt:
    """Re-check target-handoff authority and fence state at final admission."""

    if not isinstance(context, AuthoritativeTargetDispatchContext):
        raise TypeError(
            "context must be an AuthoritativeTargetDispatchContext."
        )
    if not isinstance(state, AuthoritativeReceiverStateSnapshot):
        raise TypeError("state must be an AuthoritativeReceiverStateSnapshot.")
    status = "accepted"
    if context.runtime_session_id != state.runtime_session_id:
        status = "runtime_session_mismatch"
    elif context.control_domain_id != state.control_domain_id:
        status = "control_domain_mismatch"
    elif context.owner_token_id != state.current_owner_token_id:
        status = "owner_token_mismatch"
    elif context.owner_generation != state.current_owner_generation:
        status = "owner_generation_mismatch"
    elif context.authority_sequence != state.current_authority_epoch:
        status = "authority_sequence_mismatch"
    elif context.execution_fence_sequence != state.current_execution_epoch:
        status = "execution_fence_sequence_mismatch"
    else:
        for group, generation in context.group_segment_generation_by_group.items():
            if state.expected_next_group_generation_by_group.get(group) != generation:
                status = f"group_segment_generation_mismatch:{group}"
                break
    payload = {
        "dispatch_context_hash": context.semantic_content_hash,
        "receiver_state_hash": state.semantic_content_hash,
        "accepted": status == "accepted",
        "status": status,
        "checked_at": state.captured_at,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=AUTHORITATIVE_RECEIVER_VALIDATION_HASH_DOMAIN_V1,
    )
    return AuthoritativeReceiverValidationReceipt(
        validation_id=f"authoritative-receiver-validation:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


def build_authoritative_command_dispatch_context(
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    decision_id: str,
    directive_id: str,
    safety_result_id: str,
    command_id: str,
    dispatch_permit_id: str,
    owner_token_id: str,
    owner_generation: int,
    target_effective_at: float,
    authority_epoch: int,
    execution_epoch: int,
    group_segment_generation_by_group: Mapping[str, int],
    selected_action_ref_by_group: Mapping[str, str],
    safe_action: RobotAction,
) -> AuthoritativeCommandDispatchContext:
    action = freeze_robot_action(safe_action)
    safe_action_payload = robot_action_semantic_payload(action)
    safe_action_hash = canonical_content_hash(
        safe_action_payload,
        domain="dexteleop.robot_action_record.v1",
    )
    payload = {
        "runtime_session_id": _required_text(
            runtime_session_id,
            name="runtime_session_id",
        ),
        "control_domain_id": _required_text(
            control_domain_id,
            name="control_domain_id",
        ),
        "embodiment_instance_id": _required_text(
            embodiment_instance_id,
            name="embodiment_instance_id",
        ),
        "decision_id": _required_text(decision_id, name="decision_id"),
        "directive_id": _required_text(directive_id, name="directive_id"),
        "safety_result_id": _required_text(
            safety_result_id,
            name="safety_result_id",
        ),
        "command_id": _required_text(command_id, name="command_id"),
        "dispatch_permit_id": _required_text(
            dispatch_permit_id, name="dispatch_permit_id"
        ),
        "owner_token_id": _required_text(owner_token_id, name="owner_token_id"),
        "owner_generation": _non_negative_int(
            owner_generation, name="owner_generation"
        ),
        "target_effective_at": float(target_effective_at),
        "authority_epoch": _non_negative_int(
            authority_epoch,
            name="authority_epoch",
        ),
        "execution_epoch": _non_negative_int(
            execution_epoch,
            name="execution_epoch",
        ),
        "group_segment_generation_by_group": {
            str(group): int(generation)
            for group, generation in sorted(
                group_segment_generation_by_group.items()
            )
        },
        "selected_action_ref_by_group": {
            str(group): str(proposal_id)
            for group, proposal_id in sorted(selected_action_ref_by_group.items())
        },
        "safe_action": safe_action_payload,
        "safe_action_hash": safe_action_hash,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=AUTHORITATIVE_DISPATCH_CONTEXT_HASH_DOMAIN_V1,
    )
    return AuthoritativeCommandDispatchContext(
        runtime_session_id=payload["runtime_session_id"],
        control_domain_id=payload["control_domain_id"],
        embodiment_instance_id=payload["embodiment_instance_id"],
        decision_id=payload["decision_id"],
        directive_id=payload["directive_id"],
        safety_result_id=payload["safety_result_id"],
        command_id=payload["command_id"],
        dispatch_permit_id=payload["dispatch_permit_id"],
        owner_token_id=payload["owner_token_id"],
        owner_generation=payload["owner_generation"],
        target_effective_at=payload["target_effective_at"],
        authority_epoch=payload["authority_epoch"],
        execution_epoch=payload["execution_epoch"],
        group_segment_generation_by_group=payload[
            "group_segment_generation_by_group"
        ],
        selected_action_ref_by_group=payload["selected_action_ref_by_group"],
        safe_action=action,
        safe_action_hash=safe_action_hash,
        semantic_content_hash=semantic_hash,
    )


def build_authoritative_target_dispatch_context(
    *,
    runtime_session_id: str,
    embodiment_instance_id: str,
    directive_id: str,
    command_id: str,
    dispatch_permit_id: str,
    owner_token_id: str,
    owner_generation: int,
    group_segment_generation_by_group: Mapping[str, int],
    selected_action_ref_by_group: Mapping[str, str],
    target_execution_request: TargetExecutionRequest,
    submission_receipt: TargetExecutionSubmissionReceipt,
) -> AuthoritativeTargetDispatchContext:
    """Bind one accepted target-processing submission to a motion permit."""

    if not isinstance(target_execution_request, TargetExecutionRequest):
        raise TypeError(
            "target_execution_request must be a TargetExecutionRequest."
        )
    if not isinstance(submission_receipt, TargetExecutionSubmissionReceipt):
        raise TypeError(
            "submission_receipt must be a TargetExecutionSubmissionReceipt."
        )
    request = target_execution_request
    adaptation = request.adaptation_result
    payload = {
        "runtime_session_id": _required_text(
            runtime_session_id,
            name="runtime_session_id",
        ),
        "control_domain_id": request.control_domain_id,
        "embodiment_instance_id": _required_text(
            embodiment_instance_id,
            name="embodiment_instance_id",
        ),
        "decision_id": adaptation.request.authoritative_decision_id,
        "directive_id": _required_text(directive_id, name="directive_id"),
        "command_id": _required_text(command_id, name="command_id"),
        "dispatch_permit_id": _required_text(
            dispatch_permit_id,
            name="dispatch_permit_id",
        ),
        "owner_token_id": _required_text(
            owner_token_id,
            name="owner_token_id",
        ),
        "owner_generation": _non_negative_int(
            owner_generation,
            name="owner_generation",
        ),
        "target_effective_at": adaptation.request.target_effective_at,
        "authority_sequence": adaptation.request.authority_sequence,
        "execution_fence_sequence": (
            adaptation.request.execution_fence_sequence
        ),
        "group_segment_generation_by_group": {
            str(group): int(generation)
            for group, generation in sorted(
                group_segment_generation_by_group.items()
            )
        },
        "selected_action_ref_by_group": {
            str(group): str(selected_ref)
            for group, selected_ref in sorted(
                selected_action_ref_by_group.items()
            )
        },
        "component_coverage": adaptation.request.component_coverage,
        "target_execution_request_id": request.request_id,
        "target_execution_request_hash": request.semantic_content_hash,
        "submission_receipt_id": submission_receipt.receipt_id,
        "submission_receipt_hash": submission_receipt.semantic_content_hash,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=AUTHORITATIVE_TARGET_DISPATCH_CONTEXT_HASH_DOMAIN,
    )
    return AuthoritativeTargetDispatchContext(
        runtime_session_id=payload["runtime_session_id"],
        control_domain_id=payload["control_domain_id"],
        embodiment_instance_id=payload["embodiment_instance_id"],
        decision_id=payload["decision_id"],
        directive_id=payload["directive_id"],
        command_id=payload["command_id"],
        dispatch_permit_id=payload["dispatch_permit_id"],
        owner_token_id=payload["owner_token_id"],
        owner_generation=payload["owner_generation"],
        target_effective_at=payload["target_effective_at"],
        authority_sequence=payload["authority_sequence"],
        execution_fence_sequence=payload["execution_fence_sequence"],
        group_segment_generation_by_group=payload[
            "group_segment_generation_by_group"
        ],
        selected_action_ref_by_group=payload["selected_action_ref_by_group"],
        component_coverage=payload["component_coverage"],
        target_execution_request=request,
        submission_receipt=submission_receipt,
        semantic_content_hash=semantic_hash,
    )


__all__ = [
    "AUTHORITATIVE_DISPATCH_CONTEXT_HASH_DOMAIN_V1",
    "AUTHORITATIVE_TARGET_DISPATCH_CONTEXT_HASH_DOMAIN",
    "AUTHORITATIVE_RECEIVER_STATE_HASH_DOMAIN_V1",
    "AUTHORITATIVE_RECEIVER_VALIDATION_HASH_DOMAIN_V1",
    "COMPOUND_DISPATCH_MANIFEST_HASH_DOMAIN_V1",
    "AuthoritativeCommandDispatchContext",
    "AuthoritativeTargetDispatchContext",
    "AuthoritativeReceiverStateSnapshot",
    "AuthoritativeReceiverValidationReceipt",
    "CompoundDispatchGroupBinding",
    "CompoundDispatchManifest",
    "build_authoritative_receiver_state_snapshot",
    "build_authoritative_command_dispatch_context",
    "build_authoritative_target_dispatch_context",
    "build_compound_dispatch_manifest",
    "validate_authoritative_receiver_context",
    "validate_authoritative_target_receiver_context",
    "validate_compound_dispatch_manifest",
]
