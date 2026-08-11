"""Target selection snapshots over non-admitted projected inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import ControlCut
from robot_motion.runtime.producer_epoch import ProducerEpochSnapshot

from control_composition.activation import CompositionActivationScope
from control_composition.eligibility import CompositionSourceEligibilitySnapshot
from control_composition.temporal import (
    DEFAULT_COMPOSITION_TEMPORAL_SELECTION_SPEC,
    CompositionTemporalSelectionSpec,
)

from .core import CompositionSpec
from .input_selection import (
    CompositionInputSelectionCandidate,
    select_composition_input_candidates,
)
from .projected_input import (
    ProjectedCompositionInput,
    _finite,
    _nonnegative_int,
    _required,
    frozen_control_cut_semantic_hash,
)


PROJECTED_COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.projected_composition_input_snapshot.development"
)
PROJECTED_COMPOSITION_EPOCH_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.projected_composition_epoch_snapshot.development"
)


def _producer_epoch_snapshot_hash(snapshot: ProducerEpochSnapshot) -> str:
    return canonical_content_hash(
        {
            "control_domain_id": snapshot.control_domain_id,
            "cutoff_at": snapshot.cutoff_at,
            "cutoff_admission_sequence": snapshot.cutoff_admission_sequence,
            "active_epoch_by_producer": snapshot.active_epoch_by_producer,
            "activation_id_by_producer": snapshot.activation_id_by_producer,
        },
        domain=PROJECTED_COMPOSITION_EPOCH_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT,
    )


@dataclass(frozen=True)
class ProjectedCompositionInputSnapshot:
    """One target selection over projected inputs from one frozen basis."""

    projection_basis_hash: str
    control_cut_hash: str
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    target_effective_at: float
    target_clock_domain_id: str
    composition_spec_hash: str
    action_schema_hash: str
    temporal_selection_spec_hash: str
    producer_epoch_snapshot_hash: str
    selected_input_by_role: Mapping[str, ProjectedCompositionInput]
    selected_input_before_dependency_by_role: Mapping[
        str, ProjectedCompositionInput
    ]
    dependency_ineligibility_reason_by_role: Mapping[str, str]
    missing_roles: tuple[str, ...]
    required_roles: tuple[str, ...]
    input_action_schema_hash_by_role: Mapping[str, str]
    expected_activation_id: str | None = None
    expected_activation_hash: str | None = None
    source_eligibility_snapshot_id: str | None = None
    source_eligibility_snapshot_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        basis_hash = _required(
            self.projection_basis_hash,
            name="projection_basis_hash",
        )
        cut_hash = _required(self.control_cut_hash, name="control_cut_hash")
        domain = _required(self.control_domain_id, name="control_domain_id")
        cutoff = _finite(self.cutoff_at, name="cutoff_at")
        cutoff_sequence = _nonnegative_int(
            self.cutoff_admission_sequence,
            name="cutoff_admission_sequence",
        )
        target = _finite(
            self.target_effective_at,
            name="target_effective_at",
        )
        target_clock = _required(
            self.target_clock_domain_id,
            name="target_clock_domain_id",
        )
        spec_hash = _required(
            self.composition_spec_hash,
            name="composition_spec_hash",
        )
        action_schema_hash = _required(
            self.action_schema_hash,
            name="action_schema_hash",
        )
        temporal_hash = _required(
            self.temporal_selection_spec_hash,
            name="temporal_selection_spec_hash",
        )
        epoch_hash = _required(
            self.producer_epoch_snapshot_hash,
            name="producer_epoch_snapshot_hash",
        )
        selected = dict(self.selected_input_by_role)
        before_dependency = dict(
            self.selected_input_before_dependency_by_role
        )
        if any(
            not isinstance(value, ProjectedCompositionInput)
            for value in (*selected.values(), *before_dependency.values())
        ):
            raise TypeError("Projected snapshots require projected inputs.")
        if any(
            value.projection_basis_hash != basis_hash
            for value in (*selected.values(), *before_dependency.values())
        ):
            raise ValueError("Projected snapshot mixes projection bases.")
        reasons = dict(self.dependency_ineligibility_reason_by_role)
        missing = tuple(
            _required(role, name="missing_role") for role in self.missing_roles
        )
        required_roles = tuple(
            _required(role, name="required_role") for role in self.required_roles
        )
        role_schemas = {
            _required(role, name="input schema role"): _required(
                schema_hash,
                name="input action schema hash",
            )
            for role, schema_hash in dict(
                self.input_action_schema_hash_by_role
            ).items()
        }
        activation_id = self.expected_activation_id
        activation_hash = self.expected_activation_hash
        if (activation_id is None) != (activation_hash is None):
            raise ValueError("Expected activation identity must be complete.")
        if activation_id is not None:
            activation_id = _required(
                activation_id,
                name="expected_activation_id",
            )
            activation_hash = _required(
                activation_hash,
                name="expected_activation_hash",
            )
        eligibility_id = self.source_eligibility_snapshot_id
        eligibility_hash = self.source_eligibility_snapshot_hash
        if (eligibility_id is None) != (eligibility_hash is None):
            raise ValueError("Source eligibility snapshot identity must be complete.")
        if eligibility_id is not None:
            eligibility_id = _required(
                eligibility_id,
                name="source_eligibility_snapshot_id",
            )
            eligibility_hash = _required(
                eligibility_hash,
                name="source_eligibility_snapshot_hash",
            )
        payload = {
            "projection_basis_hash": basis_hash,
            "control_cut_hash": cut_hash,
            "control_domain_id": domain,
            "cutoff_at": cutoff,
            "cutoff_admission_sequence": cutoff_sequence,
            "target_effective_at": target,
            "target_clock_domain_id": target_clock,
            "composition_spec_hash": spec_hash,
            "action_schema_hash": action_schema_hash,
            "temporal_selection_spec_hash": temporal_hash,
            "producer_epoch_snapshot_hash": epoch_hash,
            "selected_projected_input_ids_by_role": {
                role: value.projected_input_id
                for role, value in selected.items()
            },
            "selected_projected_input_hashes_by_role": {
                role: value.semantic_content_hash
                for role, value in selected.items()
            },
            "selected_before_dependency_hashes_by_role": {
                role: value.semantic_content_hash
                for role, value in before_dependency.items()
            },
            "dependency_ineligibility_reason_by_role": reasons,
            "missing_roles": missing,
            "required_roles": required_roles,
            "input_action_schema_hash_by_role": role_schemas,
            "expected_activation_id": activation_id,
            "expected_activation_hash": activation_hash,
            "source_eligibility_snapshot_id": eligibility_id,
            "source_eligibility_snapshot_hash": eligibility_hash,
            "admitted": False,
            "promotable": False,
        }
        object.__setattr__(self, "projection_basis_hash", basis_hash)
        object.__setattr__(self, "control_cut_hash", cut_hash)
        object.__setattr__(self, "control_domain_id", domain)
        object.__setattr__(self, "cutoff_at", cutoff)
        object.__setattr__(self, "cutoff_admission_sequence", cutoff_sequence)
        object.__setattr__(self, "target_effective_at", target)
        object.__setattr__(self, "target_clock_domain_id", target_clock)
        object.__setattr__(self, "composition_spec_hash", spec_hash)
        object.__setattr__(self, "action_schema_hash", action_schema_hash)
        object.__setattr__(self, "temporal_selection_spec_hash", temporal_hash)
        object.__setattr__(self, "producer_epoch_snapshot_hash", epoch_hash)
        object.__setattr__(
            self,
            "selected_input_by_role",
            MappingProxyType(selected),
        )
        object.__setattr__(
            self,
            "selected_input_before_dependency_by_role",
            MappingProxyType(before_dependency),
        )
        object.__setattr__(
            self,
            "dependency_ineligibility_reason_by_role",
            MappingProxyType(reasons),
        )
        object.__setattr__(self, "missing_roles", missing)
        object.__setattr__(self, "required_roles", required_roles)
        object.__setattr__(
            self,
            "input_action_schema_hash_by_role",
            MappingProxyType(role_schemas),
        )
        object.__setattr__(self, "expected_activation_id", activation_id)
        object.__setattr__(self, "expected_activation_hash", activation_hash)
        object.__setattr__(
            self,
            "source_eligibility_snapshot_id",
            eligibility_id,
        )
        object.__setattr__(
            self,
            "source_eligibility_snapshot_hash",
            eligibility_hash,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    PROJECTED_COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def projected_input_snapshot_id(self) -> str:
        return f"projected-composition-input-snapshot:{self.semantic_content_hash}"

    @property
    def admitted(self) -> bool:
        return False

    @property
    def promotable(self) -> bool:
        return False


def build_projected_composition_input_snapshot(
    inputs: tuple[ProjectedCompositionInput, ...],
    *,
    projection_basis_hash: str,
    control_cut: ControlCut,
    producer_epochs: ProducerEpochSnapshot,
    spec: CompositionSpec,
    target_effective_at: float,
    activation_scope: CompositionActivationScope | None = None,
    source_eligibility_snapshot: CompositionSourceEligibilitySnapshot | None = None,
    temporal_selection_spec: CompositionTemporalSelectionSpec = (
        DEFAULT_COMPOSITION_TEMPORAL_SELECTION_SPEC
    ),
) -> ProjectedCompositionInputSnapshot:
    """Select projected inputs without admission, retention, or watermark writes."""

    if not isinstance(control_cut, ControlCut):
        raise TypeError("control_cut must be a ControlCut.")
    if not isinstance(producer_epochs, ProducerEpochSnapshot):
        raise TypeError("producer_epochs must be a ProducerEpochSnapshot.")
    if not isinstance(spec, CompositionSpec):
        raise TypeError("spec must be a CompositionSpec.")
    if not isinstance(temporal_selection_spec, CompositionTemporalSelectionSpec):
        raise TypeError(
            "temporal_selection_spec must be CompositionTemporalSelectionSpec."
        )
    basis_hash = _required(
        projection_basis_hash,
        name="projection_basis_hash",
    )
    target = _finite(
        target_effective_at,
        name="target_effective_at",
    )
    temporal_selection_spec.validate_target(
        cutoff_at=control_cut.cutoff_at,
        target_effective_at=target,
    )
    if (
        producer_epochs.control_domain_id != control_cut.control_domain_id
        or producer_epochs.cutoff_at != control_cut.cutoff_at
        or producer_epochs.cutoff_admission_sequence
        != control_cut.cutoff_admission_sequence
    ):
        raise ValueError("Producer epochs and projected control cut differ.")
    if activation_scope is not None:
        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be CompositionActivationScope.")
        if activation_scope.required_roles != spec.required_roles:
            raise ValueError("Activation required roles mismatch CompositionSpec.")
    if source_eligibility_snapshot is not None:
        if not isinstance(
            source_eligibility_snapshot,
            CompositionSourceEligibilitySnapshot,
        ):
            raise TypeError(
                "source_eligibility_snapshot must be typed or None."
            )
        if (
            source_eligibility_snapshot.control_domain_id
            != control_cut.control_domain_id
            or source_eligibility_snapshot.cutoff_at != control_cut.cutoff_at
            or source_eligibility_snapshot.cutoff_admission_sequence
            != control_cut.cutoff_admission_sequence
            or source_eligibility_snapshot.target_effective_at != target
        ):
            raise ValueError("Projected inputs and eligibility must share one cut.")

    role_schemas = dict(spec.input_action_schema_hash_by_role)
    eligible: list[
        CompositionInputSelectionCandidate[ProjectedCompositionInput]
    ] = []
    for projected in tuple(inputs):
        if not isinstance(projected, ProjectedCompositionInput):
            raise TypeError("inputs must contain ProjectedCompositionInput.")
        if (
            projected.projection_basis_hash != basis_hash
            or projected.runtime_session_id != control_cut.runtime_session_id
            or projected.control_domain_id != control_cut.control_domain_id
            or projected.effective_clock_domain_id != control_cut.clock_domain_id
        ):
            raise ValueError("Projected input belongs to another frozen basis.")
        if projected.input_role not in spec.input_roles:
            continue
        expected_schema = role_schemas.get(
            projected.input_role,
            spec.action_schema_hash,
        )
        if expected_schema is not None and (
            projected.action.action_schema.semantic_content_hash
            != expected_schema
        ):
            raise ValueError("Projected input action schema mismatch.")
        if producer_epochs.active_epoch_by_producer.get(
            projected.producer_id
        ) != projected.producer_epoch:
            continue
        if activation_scope is not None and (
            projected.composition_activation_id
            != activation_scope.activation_id
            or projected.composition_activation_hash
            != activation_scope.semantic_content_hash
        ):
            continue
        if source_eligibility_snapshot is not None:
            source_state = (
                source_eligibility_snapshot.selected_state_by_role.get(
                    projected.input_role
                )
            )
            if (
                not source_eligibility_snapshot.is_role_eligible(
                    projected.input_role
                )
                or source_state is None
                or source_state.producer_id != projected.producer_id
                or source_state.producer_epoch != projected.producer_epoch
            ):
                continue
        eligible.append(
            CompositionInputSelectionCandidate(
                carrier=projected,
                input_role=projected.input_role,
                effective_from=projected.effective_from,
                effective_until=projected.effective_until,
                selection_sequence=projected.projection_sequence,
                candidate_id=projected.projected_input_id,
            )
        )
    selection = select_composition_input_candidates(
        tuple(eligible),
        input_roles=spec.input_roles,
        depends_on_roles_by_role=spec.depends_on_roles_by_role,
        target_effective_at=target,
        overlap_behavior=temporal_selection_spec.overlap_behavior,
    )
    action_schema_hash = spec.action_schema_hash
    if action_schema_hash is None:
        observed = {
            value.action.action_schema.semantic_content_hash
            for value in selection.selected_by_role.values()
        }
        if len(observed) != 1:
            raise ValueError(
                "Projected output action schema must be explicit or unique."
            )
        action_schema_hash = next(iter(observed))
    return ProjectedCompositionInputSnapshot(
        projection_basis_hash=basis_hash,
        control_cut_hash=frozen_control_cut_semantic_hash(control_cut),
        control_domain_id=control_cut.control_domain_id,
        cutoff_at=control_cut.cutoff_at,
        cutoff_admission_sequence=control_cut.cutoff_admission_sequence,
        target_effective_at=target,
        target_clock_domain_id=control_cut.clock_domain_id,
        composition_spec_hash=spec.semantic_content_hash,
        action_schema_hash=action_schema_hash,
        temporal_selection_spec_hash=(
            temporal_selection_spec.semantic_content_hash
        ),
        producer_epoch_snapshot_hash=_producer_epoch_snapshot_hash(
            producer_epochs
        ),
        selected_input_by_role=selection.selected_by_role,
        selected_input_before_dependency_by_role=(
            selection.selected_before_dependency_by_role
        ),
        dependency_ineligibility_reason_by_role=(
            selection.dependency_ineligibility_reason_by_role
        ),
        missing_roles=selection.missing_roles,
        required_roles=spec.required_roles,
        input_action_schema_hash_by_role=role_schemas,
        expected_activation_id=(
            None if activation_scope is None else activation_scope.activation_id
        ),
        expected_activation_hash=(
            None
            if activation_scope is None
            else activation_scope.semantic_content_hash
        ),
        source_eligibility_snapshot_id=(
            None
            if source_eligibility_snapshot is None
            else source_eligibility_snapshot.source_eligibility_snapshot_id
        ),
        source_eligibility_snapshot_hash=(
            None
            if source_eligibility_snapshot is None
            else source_eligibility_snapshot.semantic_content_hash
        ),
    )


__all__ = [
    "PROJECTED_COMPOSITION_EPOCH_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT",
    "PROJECTED_COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT",
    "ProjectedCompositionInputSnapshot",
    "build_projected_composition_input_snapshot",
]
