"""Stateless temporal support for trajectory-relative action inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ComponentAction

from ._validation import _finite, _required
from .contracts import (
    TRAJECTORY_EDITING_MODE_RESIDUAL_ACTIVE,
    TRAJECTORY_TARGET_STATUS_READY,
    TrajectoryEditingState,
    TrajectoryTargetPreparation,
)


RAMP_AND_HOLD_RESIDUAL_WARP_V1 = "ramp_and_hold_residual_v1"
RESIDUAL_TEMPORAL_WARP_SPEC_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.residual_temporal_warp_spec.development"
)
TRAJECTORY_RESIDUAL_TEMPORAL_SAMPLE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_residual_temporal_sample.development"
)


@dataclass(frozen=True)
class ResidualTemporalWarpSpec:
    """Profile-selected temporal support, not a motion safety envelope."""

    spec_id: str
    ramp_duration_sec: float
    algorithm: str = RAMP_AND_HOLD_RESIDUAL_WARP_V1
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        spec_id = _required(self.spec_id, name="spec_id")
        algorithm = _required(self.algorithm, name="algorithm")
        if algorithm != RAMP_AND_HOLD_RESIDUAL_WARP_V1:
            raise ValueError("Unsupported residual temporal warp algorithm.")
        duration = _finite(
            self.ramp_duration_sec,
            name="ramp_duration_sec",
        )
        if duration <= 0.0:
            raise ValueError("ramp_duration_sec must be positive.")
        object.__setattr__(self, "spec_id", spec_id)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "ramp_duration_sec", duration)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "spec_id": spec_id,
                    "algorithm": algorithm,
                    "ramp_duration_sec": duration,
                },
                domain=RESIDUAL_TEMPORAL_WARP_SPEC_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    def weight_at(self, *, edit_from: float, target_effective_at: float) -> float:
        boundary = _finite(edit_from, name="edit_from")
        target = _finite(
            target_effective_at,
            name="target_effective_at",
        )
        if target <= boundary:
            return 0.0
        elapsed = target - boundary
        if elapsed >= self.ramp_duration_sec or math.isclose(
            elapsed,
            self.ramp_duration_sec,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return 1.0
        return float(elapsed / self.ramp_duration_sec)


@dataclass(frozen=True)
class TrajectoryResidualTemporalSample:
    """One weighted relative input bound to a latched trajectory cut."""

    action: ComponentAction
    target_effective_at: float
    edit_from: float
    weight: float
    warp_spec_id: str
    warp_spec_hash: str
    editing_basis_hash: str
    active_trajectory_id: str
    active_trajectory_hash: str
    nominal_keyframe_hash: str
    residual_action_hash: str
    residual_fact_id: str
    residual_fact_hash: str
    active_anchor_binding_id: str
    active_anchor_binding_hash: str
    residual_anchor_receipt_id: str
    residual_anchor_receipt_hash: str
    anchor_request_generation: int
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.action, ComponentAction):
            raise TypeError("action must be a ComponentAction.")
        target = _finite(
            self.target_effective_at,
            name="target_effective_at",
        )
        boundary = _finite(self.edit_from, name="edit_from")
        weight = _finite(self.weight, name="weight")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1].")
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "warp_spec_id",
                "warp_spec_hash",
                "editing_basis_hash",
                "active_trajectory_id",
                "active_trajectory_hash",
                "nominal_keyframe_hash",
                "residual_action_hash",
                "residual_fact_id",
                "residual_fact_hash",
                "active_anchor_binding_id",
                "active_anchor_binding_hash",
                "residual_anchor_receipt_id",
                "residual_anchor_receipt_hash",
            )
        }
        generation = int(self.anchor_request_generation)
        if (
            isinstance(self.anchor_request_generation, bool)
            or generation != self.anchor_request_generation
            or generation < 0
        ):
            raise ValueError(
                "anchor_request_generation must be a non-negative integer."
            )
        payload = {
            "action_hash": self.action.semantic_content_hash,
            "action_schema_hash": self.action.action_schema.semantic_content_hash,
            "component_coverage": self.action.component_coverage,
            "target_effective_at": target,
            "edit_from": boundary,
            "weight": weight,
            "anchor_request_generation": generation,
            **text,
        }
        object.__setattr__(self, "target_effective_at", target)
        object.__setattr__(self, "edit_from", boundary)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "anchor_request_generation", generation)
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    TRAJECTORY_RESIDUAL_TEMPORAL_SAMPLE_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def sample_id(self) -> str:
        return f"trajectory-residual-sample:{self.semantic_content_hash}"


def _scaled_relative_values(
    action: ComponentAction,
    *,
    weight: float,
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for component_id in action.component_coverage:
        component = action.value(component_id)
        schema = component.payload_schema
        if schema.value_semantics != "relative_delta":
            raise ValueError(
                "Trajectory residual input must use relative_delta semantics."
            )
        raw = np.asarray(component.values, dtype=np.float64)
        if schema.representation_family == "cartesian_pose":
            rotation = Rotation.from_quat(raw[3:7])
            if rotation.magnitude() > 1e-10:
                raise ValueError(
                    "Pose rotation temporal warp is not implemented in the "
                    "translation-only gate."
                )
            scaled = np.zeros(7, dtype=np.float64)
            scaled[:3] = weight * raw[:3]
            scaled[6] = 1.0
        elif schema.representation_family == "joint_position":
            scaled = weight * raw
        else:
            raise ValueError(
                "Initial residual temporal warp supports only Cartesian pose "
                "translation and joint position."
            )
        values[component_id] = np.asarray(
            scaled,
            dtype=np.dtype(schema.dtype),
        )
    return values


def sample_residual_for_trajectory_target(
    state: TrajectoryEditingState,
    preparation: TrajectoryTargetPreparation,
    *,
    residual_action: ComponentAction,
    residual_fact_id: str,
    residual_fact_hash: str,
    residual_anchor_receipt_id: str,
    residual_anchor_receipt_hash: str,
    warp_spec: ResidualTemporalWarpSpec,
) -> TrajectoryResidualTemporalSample:
    """Weight one state-valued residual for an exact prepared target.

    The returned action is still a ``relative_delta`` composition input.  It
    is not an absolute corrected candidate and carries no execution claim.
    """

    if not isinstance(state, TrajectoryEditingState):
        raise TypeError("state must be a TrajectoryEditingState.")
    if state.mode != TRAJECTORY_EDITING_MODE_RESIDUAL_ACTIVE:
        raise ValueError("A residual temporal sample requires an active session.")
    if not isinstance(preparation, TrajectoryTargetPreparation):
        raise TypeError(
            "preparation must be a TrajectoryTargetPreparation."
        )
    if preparation.status != TRAJECTORY_TARGET_STATUS_READY:
        raise ValueError("Residual sampling requires a ready target preparation.")
    if preparation.publication_basis_hash != state.publication_basis_hash:
        raise ValueError("Preparation was produced from another editing basis.")
    if not isinstance(residual_action, ComponentAction):
        raise TypeError("residual_action must be a ComponentAction.")
    if not isinstance(warp_spec, ResidualTemporalWarpSpec):
        raise TypeError("warp_spec must be a ResidualTemporalWarpSpec.")
    active = state.active_trajectory
    anchor_binding = state.active_residual_anchor_binding
    if active is None or state.edit_from is None or preparation.keyframe is None:
        raise ValueError("Residual-active editing state is incomplete.")
    if anchor_binding is None:
        raise ValueError("Residual-active state has no active anchor binding.")
    if preparation.active_trajectory_hash != active.semantic_content_hash:
        raise ValueError("Prepared target no longer matches the latched base.")
    if not set(residual_action.component_coverage).issubset(
        set(active.component_coverage)
    ):
        raise ValueError("Residual components must be a subset of the base.")
    receipt = anchor_binding.residual_anchor_receipt
    if (
        _required(
            residual_anchor_receipt_id,
            name="residual_anchor_receipt_id",
        )
        != receipt.receipt_id
        or _required(
            residual_anchor_receipt_hash,
            name="residual_anchor_receipt_hash",
        )
        != receipt.semantic_content_hash
    ):
        raise ValueError("Residual action belongs to another active anchor.")
    if residual_action.action_schema.semantic_content_hash != (
        receipt.relative_action_schema_hash
    ):
        raise ValueError("Residual action schema differs from its active anchor.")
    if preparation.target_effective_at < anchor_binding.effective_for_target_at:
        raise ValueError("Prepared target predates the active residual anchor.")

    weight = warp_spec.weight_at(
        edit_from=state.edit_from,
        target_effective_at=preparation.target_effective_at,
    )
    weighted_action = ComponentAction(
        action_schema=residual_action.action_schema,
        values_by_component=_scaled_relative_values(
            residual_action,
            weight=weight,
        ),
    )
    return TrajectoryResidualTemporalSample(
        action=weighted_action,
        target_effective_at=preparation.target_effective_at,
        edit_from=state.edit_from,
        weight=weight,
        warp_spec_id=warp_spec.spec_id,
        warp_spec_hash=warp_spec.semantic_content_hash,
        editing_basis_hash=state.publication_basis_hash,
        active_trajectory_id=active.trajectory_id,
        active_trajectory_hash=active.semantic_content_hash,
        nominal_keyframe_hash=preparation.keyframe.semantic_content_hash,
        residual_action_hash=residual_action.semantic_content_hash,
        residual_fact_id=_required(residual_fact_id, name="residual_fact_id"),
        residual_fact_hash=_required(
            residual_fact_hash,
            name="residual_fact_hash",
        ),
        active_anchor_binding_id=anchor_binding.anchor_binding_id,
        active_anchor_binding_hash=anchor_binding.semantic_content_hash,
        residual_anchor_receipt_id=receipt.receipt_id,
        residual_anchor_receipt_hash=receipt.semantic_content_hash,
        anchor_request_generation=receipt.request_generation,
    )


__all__ = [
    "RAMP_AND_HOLD_RESIDUAL_WARP_V1",
    "RESIDUAL_TEMPORAL_WARP_SPEC_HASH_DOMAIN_DEVELOPMENT",
    "TRAJECTORY_RESIDUAL_TEMPORAL_SAMPLE_HASH_DOMAIN_DEVELOPMENT",
    "ResidualTemporalWarpSpec",
    "TrajectoryResidualTemporalSample",
    "sample_residual_for_trajectory_target",
]
