"""Pure development contracts for nominal-trajectory editing state."""

from __future__ import annotations

from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    ComponentAction,
    ComponentActionTrajectory,
    ComponentActionTrajectoryKeyframe,
)

from ..residual_anchor import ResidualAnchorReceipt

from ._validation import _finite, _nonnegative_int, _required
from .lifecycle import (
    TRAJECTORY_INTERACTION_MODE_PASS_THROUGH,
    TRAJECTORY_INTERACTION_MODE_RESIDUAL_ACTIVE,
    TRAJECTORY_INTERACTION_MODES_DEVELOPMENT,
    NominalBaseBinding,
    ResidualSessionActivationBinding,
    validate_trajectory_interaction_base_state,
)
from .anchor_lifecycle import (
    ActiveResidualAnchorBinding,
    ResidualRebaseRequest,
)


# Compatibility aliases for the R0-R2 development import surface.  Runtime
# state now names this axis ``interaction_mode`` explicitly.
TRAJECTORY_EDITING_MODE_PASS_THROUGH = TRAJECTORY_INTERACTION_MODE_PASS_THROUGH
TRAJECTORY_EDITING_MODE_RESIDUAL_ACTIVE = (
    TRAJECTORY_INTERACTION_MODE_RESIDUAL_ACTIVE
)
TRAJECTORY_EDITING_MODES_DEVELOPMENT = (
    TRAJECTORY_INTERACTION_MODES_DEVELOPMENT
)

TRAJECTORY_TARGET_STATUS_READY = "ready"
TRAJECTORY_TARGET_STATUS_NO_ACTION = "no_action"
TRAJECTORY_TARGET_STATUS_ALREADY_EMITTED = "already_emitted"
TRAJECTORY_TARGET_STATUSES_DEVELOPMENT = frozenset(
    {
        TRAJECTORY_TARGET_STATUS_READY,
        TRAJECTORY_TARGET_STATUS_NO_ACTION,
        TRAJECTORY_TARGET_STATUS_ALREADY_EMITTED,
    }
)

TRAJECTORY_EDITING_STATE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_editing_state.development"
)
TRAJECTORY_PUBLICATION_BASIS_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_publication_basis.development"
)
TRAJECTORY_TARGET_PREPARATION_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_target_preparation.development"
)


@dataclass(frozen=True)
class TrajectoryEditingState:
    """Reducer state with no execution or authority ownership."""

    reducer_id: str
    revision: int
    interaction_mode: str
    base_lifecycle_status: str
    active_trajectory: ComponentActionTrajectory | None = None
    active_offer_sequence: int | None = None
    shadow_trajectory: ComponentActionTrajectory | None = None
    shadow_offer_sequence: int | None = None
    residual_session_activation_binding: (
        ResidualSessionActivationBinding | None
    ) = None
    active_base_binding: NominalBaseBinding | None = None
    active_residual_anchor_binding: ActiveResidualAnchorBinding | None = None
    pending_residual_rebase: ResidualRebaseRequest | None = None
    last_emitted_target_at: float | None = None
    last_emitted_action_hash: str | None = None
    last_emission_receipt_id: str | None = None
    last_emission_receipt_hash: str | None = None
    publication_basis_hash: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        reducer_id = _required(self.reducer_id, name="reducer_id")
        revision = _nonnegative_int(self.revision, name="revision")
        interaction_mode = _required(
            self.interaction_mode,
            name="interaction_mode",
        )
        base_lifecycle_status = _required(
            self.base_lifecycle_status,
            name="base_lifecycle_status",
        )
        validate_trajectory_interaction_base_state(
            interaction_mode,
            base_lifecycle_status,
        )

        active = self.active_trajectory
        if active is not None and not isinstance(
            active,
            ComponentActionTrajectory,
        ):
            raise TypeError(
                "active_trajectory must be ComponentActionTrajectory or None."
            )
        active_sequence = self.active_offer_sequence
        if (active is None) != (active_sequence is None):
            raise ValueError(
                "active_trajectory and active_offer_sequence must coexist."
            )
        if active_sequence is not None:
            active_sequence = _nonnegative_int(
                active_sequence,
                name="active_offer_sequence",
            )

        shadow = self.shadow_trajectory
        if shadow is not None and not isinstance(
            shadow,
            ComponentActionTrajectory,
        ):
            raise TypeError(
                "shadow_trajectory must be ComponentActionTrajectory or None."
            )
        shadow_sequence = self.shadow_offer_sequence
        if (shadow is None) != (shadow_sequence is None):
            raise ValueError(
                "shadow_trajectory and shadow_offer_sequence must coexist."
            )
        if shadow_sequence is not None:
            shadow_sequence = _nonnegative_int(
                shadow_sequence,
                name="shadow_offer_sequence",
            )

        activation_binding = self.residual_session_activation_binding
        base_binding = self.active_base_binding
        anchor_binding = self.active_residual_anchor_binding
        pending_rebase = self.pending_residual_rebase
        if interaction_mode == TRAJECTORY_EDITING_MODE_PASS_THROUGH:
            if any(
                value is not None
                for value in (
                    activation_binding,
                    base_binding,
                    anchor_binding,
                    pending_rebase,
                )
            ):
                raise ValueError(
                    "Pass-through state cannot carry residual/base/rebase bindings."
                )
            if shadow is not None:
                raise ValueError(
                    "Pass-through state cannot retain a shadow trajectory."
                )
        else:
            if active is None:
                raise ValueError(
                    "A session interaction requires an active trajectory."
                )
            if not isinstance(
                activation_binding,
                ResidualSessionActivationBinding,
            ):
                raise TypeError(
                    "A session interaction requires a residual session "
                    "activation binding."
                )
            if not isinstance(base_binding, NominalBaseBinding):
                raise TypeError(
                    "A session interaction requires a nominal base binding."
                )
            if not isinstance(anchor_binding, ActiveResidualAnchorBinding):
                raise TypeError(
                    "A session interaction requires an active residual anchor "
                    "binding."
                )
            if (
                base_binding.residual_session_id
                != activation_binding.residual_session_id
                or anchor_binding.residual_session_id
                != activation_binding.residual_session_id
            ):
                raise ValueError(
                    "Active base or anchor belongs to another residual session."
                )
            if (
                base_binding.trajectory.semantic_content_hash
                != active.semantic_content_hash
                or base_binding.active_offer_sequence != active_sequence
            ):
                raise ValueError(
                    "Active trajectory must equal the active nominal base."
                )
            if base_binding.binding_revision == 0 and (
                base_binding.semantic_content_hash
                != activation_binding.initial_base_binding.semantic_content_hash
            ):
                raise ValueError(
                    "Initial active base must equal the session activation base."
                )
            initial_receipt = activation_binding.residual_anchor_receipt
            active_receipt = anchor_binding.residual_anchor_receipt
            if anchor_binding.binding_revision == 0 and (
                active_receipt.semantic_content_hash
                != initial_receipt.semantic_content_hash
                or anchor_binding.effective_for_target_at
                != activation_binding.first_editable_target_at
            ):
                raise ValueError(
                    "Initial active anchor must equal the session activation anchor."
                )
            for field_name in (
                "source_role",
                "composition_activation_id",
                "composition_activation_hash",
                "source_binding_id",
                "source_binding_hash",
                "absolute_action_schema_hash",
                "relative_action_schema_hash",
                "clock_domain_id",
            ):
                if getattr(active_receipt, field_name) != getattr(
                    initial_receipt,
                    field_name,
                ):
                    raise ValueError(
                        "Active residual anchor changed the session protocol."
                    )
            if set(active_receipt.component_binding_by_id) != set(
                initial_receipt.component_binding_by_id
            ):
                raise ValueError(
                    "Active residual anchor changed component coverage."
                )
            if active.keyframe_at(anchor_binding.effective_for_target_at) is None:
                raise ValueError(
                    "Active residual anchor boundary must be a trajectory keyframe."
                )
            if pending_rebase is not None:
                if not isinstance(pending_rebase, ResidualRebaseRequest):
                    raise TypeError(
                        "pending_residual_rebase must be ResidualRebaseRequest."
                    )
                if (
                    pending_rebase.residual_session_id
                    != activation_binding.residual_session_id
                    or pending_rebase.expected_anchor_binding_id
                    != anchor_binding.anchor_binding_id
                    or pending_rebase.expected_anchor_binding_hash
                    != anchor_binding.semantic_content_hash
                ):
                    raise ValueError(
                        "Pending rebase was built from another active anchor."
                    )
                if active.keyframe_at(
                    pending_rebase.effective_not_before_target_at
                ) is None:
                    raise ValueError(
                        "Pending rebase boundary must be a trajectory keyframe."
                    )

        last_target = self.last_emitted_target_at
        last_action_hash = self.last_emitted_action_hash
        last_receipt_id = self.last_emission_receipt_id
        last_receipt_hash = self.last_emission_receipt_hash
        last_fields = (
            last_target,
            last_action_hash,
            last_receipt_id,
            last_receipt_hash,
        )
        if any(value is not None for value in last_fields) and not all(
            value is not None for value in last_fields
        ):
            raise ValueError("Last-emission lineage must be complete.")
        if last_target is not None:
            last_target = _finite(
                last_target,
                name="last_emitted_target_at",
            )
            last_action_hash = _required(
                last_action_hash,
                name="last_emitted_action_hash",
            )
            last_receipt_id = _required(
                last_receipt_id,
                name="last_emission_receipt_id",
            )
            last_receipt_hash = _required(
                last_receipt_hash,
                name="last_emission_receipt_hash",
            )

        if activation_binding is not None:
            initial_target = (
                activation_binding.initial_publication_finalized_target_at
            )
            if initial_target is not None:
                if last_target is None or last_target < initial_target:
                    raise ValueError(
                        "Runtime publication barrier cannot precede the "
                        "session's initial barrier."
                    )
                if last_target == initial_target and (
                    last_action_hash
                    != activation_binding.initial_published_action_hash
                    or last_receipt_id
                    != activation_binding.initial_emission_receipt_id
                    or last_receipt_hash
                    != activation_binding.initial_emission_receipt_hash
                ):
                    raise ValueError(
                        "Initial publication barrier lineage changed in-place."
                    )
        if (
            pending_rebase is not None
            and last_target is not None
            and last_target >= pending_rebase.effective_not_before_target_at
        ):
            raise ValueError(
                "A target at or after a pending rebase boundary cannot be "
                "committed with the old anchor."
            )

        publication_basis_payload = {
            "reducer_id": reducer_id,
            "interaction_mode": interaction_mode,
            "base_lifecycle_status": base_lifecycle_status,
            "active_trajectory_hash": (
                None if active is None else active.semantic_content_hash
            ),
            "active_offer_sequence": active_sequence,
            "residual_session_activation_binding_hash": (
                None
                if activation_binding is None
                else activation_binding.semantic_content_hash
            ),
            "active_base_binding_hash": (
                None
                if base_binding is None
                else base_binding.semantic_content_hash
            ),
            "active_residual_anchor_binding_hash": (
                None
                if anchor_binding is None
                else anchor_binding.semantic_content_hash
            ),
            "pending_residual_rebase_hash": (
                None
                if pending_rebase is None
                else pending_rebase.semantic_content_hash
            ),
            "last_emitted_target_at": last_target,
            "last_emitted_action_hash": last_action_hash,
            "last_emission_receipt_id": last_receipt_id,
            "last_emission_receipt_hash": last_receipt_hash,
        }
        publication_basis_hash = canonical_content_hash(
            publication_basis_payload,
            domain=TRAJECTORY_PUBLICATION_BASIS_HASH_DOMAIN_DEVELOPMENT,
        )
        payload = {
            "reducer_id": reducer_id,
            "revision": revision,
            "interaction_mode": interaction_mode,
            "base_lifecycle_status": base_lifecycle_status,
            "active_trajectory_hash": (
                None if active is None else active.semantic_content_hash
            ),
            "active_offer_sequence": active_sequence,
            "shadow_trajectory_hash": (
                None if shadow is None else shadow.semantic_content_hash
            ),
            "shadow_offer_sequence": shadow_sequence,
            "residual_session_activation_binding_hash": (
                None
                if activation_binding is None
                else activation_binding.semantic_content_hash
            ),
            "active_base_binding_hash": (
                None
                if base_binding is None
                else base_binding.semantic_content_hash
            ),
            "active_residual_anchor_binding_hash": (
                None
                if anchor_binding is None
                else anchor_binding.semantic_content_hash
            ),
            "pending_residual_rebase_hash": (
                None
                if pending_rebase is None
                else pending_rebase.semantic_content_hash
            ),
            "last_emitted_target_at": last_target,
            "last_emitted_action_hash": last_action_hash,
            "last_emission_receipt_id": last_receipt_id,
            "last_emission_receipt_hash": last_receipt_hash,
            "publication_basis_hash": publication_basis_hash,
        }
        object.__setattr__(self, "reducer_id", reducer_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "interaction_mode", interaction_mode)
        object.__setattr__(
            self,
            "base_lifecycle_status",
            base_lifecycle_status,
        )
        object.__setattr__(self, "active_offer_sequence", active_sequence)
        object.__setattr__(self, "shadow_offer_sequence", shadow_sequence)
        object.__setattr__(self, "last_emitted_target_at", last_target)
        object.__setattr__(self, "last_emitted_action_hash", last_action_hash)
        object.__setattr__(self, "last_emission_receipt_id", last_receipt_id)
        object.__setattr__(self, "last_emission_receipt_hash", last_receipt_hash)
        object.__setattr__(
            self,
            "publication_basis_hash",
            publication_basis_hash,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TRAJECTORY_EDITING_STATE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def mode(self) -> str:
        """R0-R2 compatibility alias for ``interaction_mode``."""

        return self.interaction_mode

    @property
    def residual_session_id(self) -> str | None:
        binding = self.residual_session_activation_binding
        return None if binding is None else binding.residual_session_id

    @property
    def edit_from(self) -> float | None:
        binding = self.residual_session_activation_binding
        return None if binding is None else binding.first_editable_target_at

    @property
    def active_residual_anchor_receipt(self) -> ResidualAnchorReceipt | None:
        binding = self.active_residual_anchor_binding
        return None if binding is None else binding.residual_anchor_receipt

    @property
    def maximum_offer_sequence(self) -> int | None:
        sequences = tuple(
            value
            for value in (
                self.active_offer_sequence,
                self.shadow_offer_sequence,
            )
            if value is not None
        )
        return max(sequences) if sequences else None


@dataclass(frozen=True)
class TrajectoryTargetPreparation:
    """One non-authoritative preparation; committing it is separate."""

    status: str
    target_effective_at: float
    publication_basis_hash: str
    reason: str
    active_trajectory_id: str | None = None
    active_trajectory_hash: str | None = None
    keyframe: ComponentActionTrajectoryKeyframe | None = None
    action: ComponentAction | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        status = _required(self.status, name="status")
        if status not in TRAJECTORY_TARGET_STATUSES_DEVELOPMENT:
            raise ValueError("Unsupported trajectory target status.")
        target = _finite(
            self.target_effective_at,
            name="target_effective_at",
        )
        publication_basis_hash = _required(
            self.publication_basis_hash,
            name="publication_basis_hash",
        )
        reason = _required(self.reason, name="reason")
        identity_fields = (
            self.active_trajectory_id,
            self.active_trajectory_hash,
        )
        if any(value is not None for value in identity_fields) and not all(
            value is not None for value in identity_fields
        ):
            raise ValueError("Active trajectory identity must be complete.")
        trajectory_id = self.active_trajectory_id
        trajectory_hash = self.active_trajectory_hash
        if trajectory_id is not None:
            trajectory_id = _required(
                trajectory_id,
                name="active_trajectory_id",
            )
            trajectory_hash = _required(
                trajectory_hash,
                name="active_trajectory_hash",
            )

        if status == TRAJECTORY_TARGET_STATUS_READY:
            if not isinstance(
                self.keyframe,
                ComponentActionTrajectoryKeyframe,
            ) or not isinstance(self.action, ComponentAction):
                raise ValueError(
                    "Ready preparation requires a keyframe and action."
                )
            if self.action.semantic_content_hash != (
                self.keyframe.action.semantic_content_hash
            ):
                raise ValueError(
                    "Prepared action must equal the selected keyframe action."
                )
            if self.keyframe.target_effective_at != target:
                raise ValueError("Prepared keyframe target mismatch.")
            if trajectory_id is None:
                raise ValueError(
                    "Ready preparation requires active trajectory identity."
                )
        elif self.keyframe is not None or self.action is not None:
            raise ValueError(
                "Non-ready preparation cannot carry a keyframe/action."
            )

        payload = {
            "status": status,
            "target_effective_at": target,
            "publication_basis_hash": publication_basis_hash,
            "reason": reason,
            "active_trajectory_id": trajectory_id,
            "active_trajectory_hash": trajectory_hash,
            "keyframe_hash": (
                None
                if self.keyframe is None
                else self.keyframe.semantic_content_hash
            ),
            "action_hash": (
                None
                if self.action is None
                else self.action.semantic_content_hash
            ),
        }
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "target_effective_at", target)
        object.__setattr__(
            self,
            "publication_basis_hash",
            publication_basis_hash,
        )
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "active_trajectory_id", trajectory_id)
        object.__setattr__(self, "active_trajectory_hash", trajectory_hash)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TRAJECTORY_TARGET_PREPARATION_HASH_DOMAIN_DEVELOPMENT,
            ),
        )


__all__ = [
    "TRAJECTORY_EDITING_MODE_PASS_THROUGH",
    "TRAJECTORY_EDITING_MODE_RESIDUAL_ACTIVE",
    "TRAJECTORY_EDITING_MODES_DEVELOPMENT",
    "TRAJECTORY_EDITING_STATE_HASH_DOMAIN_DEVELOPMENT",
    "TRAJECTORY_PUBLICATION_BASIS_HASH_DOMAIN_DEVELOPMENT",
    "TRAJECTORY_TARGET_PREPARATION_HASH_DOMAIN_DEVELOPMENT",
    "TRAJECTORY_TARGET_STATUS_ALREADY_EMITTED",
    "TRAJECTORY_TARGET_STATUS_NO_ACTION",
    "TRAJECTORY_TARGET_STATUS_READY",
    "TRAJECTORY_TARGET_STATUSES_DEVELOPMENT",
    "TrajectoryEditingState",
    "TrajectoryTargetPreparation",
]
