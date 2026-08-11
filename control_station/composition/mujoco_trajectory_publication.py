"""Thin MuJoCo publication wiring for one trajectory-composed due target.

The reusable reducer and input preparation remain in ``control_composition``.
This Station adapter only binds one ready evaluation to the existing MuJoCo
authority/proposal chain and feeds its proposal admission receipt back to the
reducer.  It deliberately does not step the lower control loop: target
resolution, reference generation, and dispatch remain lower-owned operations.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Callable

from dataset_contracts.canonical_encoding import canonical_content_hash

from control_composition.composition import CompositionSpec
from control_composition.trajectory import (
    ResidualTemporalWarpSpec,
    TrajectoryCompositionContributorStamp,
    TrajectoryCompositionInputPreparation,
    TrajectoryEditingState,
    commit_trajectory_target_emission,
    prepare_trajectory_composition_inputs,
)
from robot_motion.action import ComponentAction

from ..publication_barrier import StationPublicationBarrier
from .mujoco_sim_hil import MujocoSimHILComposition, MujocoSimHILCompositionStep


MUJOCO_TRAJECTORY_PUBLICATION_PROFILE_OFF = "off"
MUJOCO_TRAJECTORY_PUBLICATION_CYCLE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.control_station.mujoco_trajectory_publication_cycle.development"
)
MUJOCO_TRAJECTORY_PUBLICATION_REQUEST_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.control_station.mujoco_trajectory_publication_request.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _stamp_payload(
    stamp: TrajectoryCompositionContributorStamp,
) -> dict[str, object]:
    return {
        "producer_id": stamp.producer_id,
        "producer_epoch": stamp.producer_epoch,
        "sequence": stamp.sequence,
        "produced_at": stamp.produced_at,
        "produced_clock_domain_id": stamp.produced_clock_domain_id,
        "context_ids": stamp.context_ids,
        "observation_cut_id": stamp.observation_cut_id,
        "policy_output_id": stamp.policy_output_id,
        "audit_metadata": dict(stamp.audit_metadata),
    }


@dataclass(frozen=True)
class MujocoTrajectoryPublicationProfile:
    """Explicit default-off switch for the first Sim-only publication gate."""

    profile_id: str = MUJOCO_TRAJECTORY_PUBLICATION_PROFILE_OFF
    enabled: bool = False
    send_actions: bool = False

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        if not isinstance(self.enabled, bool) or not isinstance(
            self.send_actions,
            bool,
        ):
            raise TypeError("enabled and send_actions must be booleans.")
        if self.send_actions and not self.enabled:
            raise ValueError("send_actions=true requires enabled=true.")
        if self.enabled and not self.send_actions:
            raise ValueError(
                "Use the trajectory shadow profile when send_actions=false."
            )
        if self.enabled and profile_id == MUJOCO_TRAJECTORY_PUBLICATION_PROFILE_OFF:
            raise ValueError(
                "An enabled publication profile requires an explicit id."
            )
        object.__setattr__(self, "profile_id", profile_id)


@dataclass(frozen=True)
class MujocoTrajectoryPublicationRequest:
    """Externally prepared source facts for one exact trajectory lattice time."""

    target_effective_at: float
    nominal_stamp: TrajectoryCompositionContributorStamp
    residual_stamp: TrajectoryCompositionContributorStamp
    residual_action: ComponentAction
    residual_fact_id: str
    residual_fact_hash: str
    residual_anchor_receipt_id: str
    residual_anchor_receipt_hash: str
    residual_effective_until: float
    residual_warp_spec: ResidualTemporalWarpSpec
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = _finite(self.target_effective_at, name="target_effective_at")
        until = _finite(
            self.residual_effective_until,
            name="residual_effective_until",
        )
        if not isinstance(
            self.nominal_stamp,
            TrajectoryCompositionContributorStamp,
        ) or not isinstance(
            self.residual_stamp,
            TrajectoryCompositionContributorStamp,
        ):
            raise TypeError("Request stamps must be contributor stamps.")
        if not isinstance(self.residual_action, ComponentAction):
            raise TypeError("residual_action must be ComponentAction.")
        if not isinstance(self.residual_warp_spec, ResidualTemporalWarpSpec):
            raise TypeError("residual_warp_spec must be ResidualTemporalWarpSpec.")
        fact_id = _required(self.residual_fact_id, name="residual_fact_id")
        fact_hash = _required(self.residual_fact_hash, name="residual_fact_hash")
        anchor_receipt_id = _required(
            self.residual_anchor_receipt_id,
            name="residual_anchor_receipt_id",
        )
        anchor_receipt_hash = _required(
            self.residual_anchor_receipt_hash,
            name="residual_anchor_receipt_hash",
        )
        payload = {
            "target_effective_at": target,
            "nominal_stamp": _stamp_payload(self.nominal_stamp),
            "residual_stamp": _stamp_payload(self.residual_stamp),
            "residual_action_hash": self.residual_action.semantic_content_hash,
            "residual_fact_id": fact_id,
            "residual_fact_hash": fact_hash,
            "residual_anchor_receipt_id": anchor_receipt_id,
            "residual_anchor_receipt_hash": anchor_receipt_hash,
            "residual_effective_until": until,
            "residual_warp_spec_hash": (
                self.residual_warp_spec.semantic_content_hash
            ),
        }
        object.__setattr__(self, "target_effective_at", target)
        object.__setattr__(self, "residual_effective_until", until)
        object.__setattr__(self, "residual_fact_id", fact_id)
        object.__setattr__(self, "residual_fact_hash", fact_hash)
        object.__setattr__(
            self,
            "residual_anchor_receipt_id",
            anchor_receipt_id,
        )
        object.__setattr__(
            self,
            "residual_anchor_receipt_hash",
            anchor_receipt_hash,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    MUJOCO_TRAJECTORY_PUBLICATION_REQUEST_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )


@dataclass(frozen=True)
class MujocoTrajectoryPublicationCycle:
    """One exact Station proposal-publication attempt and its receipt."""

    profile_id: str
    status: str
    state_before_hash: str | None = None
    state_after: TrajectoryEditingState | None = None
    request: MujocoTrajectoryPublicationRequest | None = None
    input_preparation: TrajectoryCompositionInputPreparation | None = None
    composition_step: MujocoSimHILCompositionStep | None = None
    emission_receipt_id: str | None = None
    emission_receipt_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        status = _required(self.status, name="status")
        if status not in {
            "disabled",
            "no_action",
            "invalid",
            "proposal_not_admitted",
            "proposal_admitted",
        }:
            raise ValueError("Unsupported MuJoCo trajectory cycle status.")
        if status == "disabled":
            if any(
                value is not None
                for value in (
                    self.state_before_hash,
                    self.state_after,
                    self.request,
                    self.input_preparation,
                    self.composition_step,
                    self.emission_receipt_id,
                    self.emission_receipt_hash,
                )
            ):
                raise ValueError("A disabled cycle cannot carry runtime facts.")
        else:
            if (
                self.state_before_hash is None
                or not isinstance(self.state_after, TrajectoryEditingState)
                or not isinstance(self.request, MujocoTrajectoryPublicationRequest)
                or not isinstance(
                    self.input_preparation,
                    TrajectoryCompositionInputPreparation,
                )
            ):
                raise ValueError(
                    "An enabled publication cycle requires complete facts."
                )
        receipt_id = (
            None
            if self.emission_receipt_id is None
            else _required(self.emission_receipt_id, name="emission_receipt_id")
        )
        receipt_hash = (
            None
            if self.emission_receipt_hash is None
            else _required(
                self.emission_receipt_hash,
                name="emission_receipt_hash",
            )
        )
        if (receipt_id is None) != (receipt_hash is None):
            raise ValueError("Emission receipt identity must be complete.")
        if status == "proposal_admitted" and receipt_id is None:
            raise ValueError("An admitted cycle requires its proposal receipt.")
        if status != "proposal_admitted" and receipt_id is not None:
            raise ValueError("Only an admitted cycle may carry an emission receipt.")
        state_after_hash = (
            None
            if self.state_after is None
            else self.state_after.semantic_content_hash
        )
        payload = {
            "profile_id": profile_id,
            "status": status,
            "state_before_hash": self.state_before_hash,
            "state_after_hash": state_after_hash,
            "request_hash": (
                None if self.request is None else self.request.semantic_content_hash
            ),
            "input_preparation_hash": (
                None
                if self.input_preparation is None
                else self.input_preparation.semantic_content_hash
            ),
            "composition_step_hash": (
                None
                if self.composition_step is None
                else self.composition_step.semantic_content_hash
            ),
            "emission_receipt_id": receipt_id,
            "emission_receipt_hash": receipt_hash,
        }
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "emission_receipt_id", receipt_id)
        object.__setattr__(self, "emission_receipt_hash", receipt_hash)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    MUJOCO_TRAJECTORY_PUBLICATION_CYCLE_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )


class MujocoTrajectoryPublicationController:
    """Serialize one residual-active trajectory publication stream."""

    def __init__(
        self,
        *,
        profile: MujocoTrajectoryPublicationProfile,
        composition: MujocoSimHILComposition | None = None,
        initial_state: TrajectoryEditingState | None = None,
        composition_spec: CompositionSpec | None = None,
        publication_barrier: StationPublicationBarrier | None = None,
        nominal_role: str = "policy",
        residual_role: str = "human_residual",
    ) -> None:
        if not isinstance(profile, MujocoTrajectoryPublicationProfile):
            raise TypeError("profile must be MujocoTrajectoryPublicationProfile.")
        if profile.enabled:
            if not isinstance(composition, MujocoSimHILComposition):
                raise TypeError(
                    "Enabled publication requires a Sim-HIL composition."
                )
            if not isinstance(initial_state, TrajectoryEditingState):
                raise TypeError("Enabled publication requires trajectory state.")
            if not isinstance(composition_spec, CompositionSpec):
                raise TypeError("Enabled publication requires CompositionSpec.")
            if not isinstance(publication_barrier, StationPublicationBarrier):
                raise TypeError(
                    "Enabled publication requires a StationPublicationBarrier."
                )
            barrier_snapshot = publication_barrier.snapshot()
            if (
                not barrier_snapshot.enabled
                or barrier_snapshot.authority_operation_result_id is None
                or barrier_snapshot.transition_reason
                not in {
                    "typed_composition_session_completion",
                    "typed_composition_session_rearm_completion",
                }
            ):
                raise RuntimeError(
                    "Enabled publication requires an active typed composition "
                    "session barrier."
                )
            if composition_spec.semantic_content_hash != (
                composition.contracts.composition_spec.semantic_content_hash
            ):
                raise ValueError("Controller and Sim-HIL use another CompositionSpec.")
        else:
            barrier_snapshot = None
        self.profile = profile
        self.composition = composition
        self._state = initial_state
        self.composition_spec = composition_spec
        self.publication_barrier = publication_barrier
        self._publication_barrier_generation = (
            None if barrier_snapshot is None else barrier_snapshot.generation
        )
        self._publication_authority_operation_result_id = (
            None
            if barrier_snapshot is None
            else barrier_snapshot.authority_operation_result_id
        )
        self.nominal_role = _required(nominal_role, name="nominal_role")
        self.residual_role = _required(residual_role, name="residual_role")
        if self.nominal_role == self.residual_role:
            raise ValueError("nominal_role and residual_role must differ.")
        self._lock = threading.RLock()

    @property
    def state(self) -> TrajectoryEditingState | None:
        with self._lock:
            return self._state

    def publish_exact_target(
        self,
        request_factory: Callable[[], MujocoTrajectoryPublicationRequest],
    ) -> MujocoTrajectoryPublicationCycle:
        """Publish one exact target without advancing the lower control loop.

        The target may be admitted before its effective time.  The existing
        MuJoCo loop remains the sole owner of authority selection, target
        execution, reference generation, and dispatch at that later time.

        ``request_factory`` is a lazy, bounded read of an already-produced
        application fact.  It must not run model inference, poll a device, or
        advance a source cursor: the Station barrier remains held until the
        resulting proposal admission is complete so release can linearize
        before lower HOLD/cancellation.
        """

        if not callable(request_factory):
            raise TypeError("request_factory must be callable.")
        if not self.profile.enabled:
            return MujocoTrajectoryPublicationCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        assert self.publication_barrier is not None
        return self.publication_barrier.execute_if_enabled(
            lambda: self._publish_enabled(request_factory)
        )

    def _publish_enabled(
        self,
        request_factory: Callable[[], MujocoTrajectoryPublicationRequest],
    ) -> MujocoTrajectoryPublicationCycle:
        """Publish under the externally owned Station session barrier."""

        assert self.publication_barrier is not None
        barrier_snapshot = self.publication_barrier.snapshot()
        if (
            barrier_snapshot.generation
            != self._publication_barrier_generation
            or barrier_snapshot.authority_operation_result_id
            != self._publication_authority_operation_result_id
        ):
            raise RuntimeError(
                "Trajectory publication controller belongs to another Station "
                "composition-session activation."
            )
        with self._lock:
            assert self.composition is not None
            assert self.composition_spec is not None
            state = self._state
            assert state is not None
            request = request_factory()
            if not isinstance(request, MujocoTrajectoryPublicationRequest):
                raise TypeError(
                    "request_factory must return "
                    "MujocoTrajectoryPublicationRequest."
                )
            activation = state.residual_session_activation_binding
            active_scope = self.composition.activation_scope
            if activation is None or active_scope is None:
                raise RuntimeError(
                    "Trajectory publication requires an active residual session."
                )
            if (
                activation.activation_scope.activation_id
                != active_scope.activation_id
                or activation.activation_scope.semantic_content_hash
                != active_scope.semantic_content_hash
            ):
                raise ValueError(
                    "Trajectory session and Sim-HIL composition activation differ."
                )
            inputs = prepare_trajectory_composition_inputs(
                self.composition.runtime.policy_topology,
                state=state,
                target_effective_at=request.target_effective_at,
                composition_spec=self.composition_spec,
                nominal_role=self.nominal_role,
                residual_role=self.residual_role,
                nominal_stamp=request.nominal_stamp,
                residual_stamp=request.residual_stamp,
                residual_action=request.residual_action,
                residual_fact_id=request.residual_fact_id,
                residual_fact_hash=request.residual_fact_hash,
                residual_anchor_receipt_id=(
                    request.residual_anchor_receipt_id
                ),
                residual_anchor_receipt_hash=(
                    request.residual_anchor_receipt_hash
                ),
                residual_effective_until=request.residual_effective_until,
                residual_warp_spec=request.residual_warp_spec,
                admission_mode="publication_candidate",
            )
            composition_step = None
            receipt_id = None
            receipt_hash = None
            next_state = state
            if inputs.output_status == "ready":
                decision_key = (
                    f"{self.profile.profile_id}:"
                    f"{inputs.preparation.semantic_content_hash}"
                )
                composition_step = self.composition.compose_at(
                    decision_key=decision_key,
                    target_effective_at=request.target_effective_at,
                    expected_input_identity_by_role=(
                        inputs.selected_input_identity_by_role
                    ),
                )
                receipt = composition_step.proposal_admission
                if receipt is not None and receipt.accepted:
                    proposal = receipt.proposal
                    if (
                        proposal is None
                        or receipt.proposal_id is None
                        or receipt.semantic_content_hash is None
                    ):
                        raise RuntimeError(
                            "Accepted proposal admission has incomplete identity."
                        )
                    evaluation = composition_step.evaluation
                    if evaluation is None:
                        raise RuntimeError(
                            "Accepted proposal has no composition evaluation."
                        )
                    composed = (
                        evaluation.composition_evaluation.composed_candidate
                    )
                    if (
                        composed is None
                        or proposal.action.semantic_content_hash
                        != composed.semantic_content_hash
                        or not math.isclose(
                            proposal.target_effective_at,
                            request.target_effective_at,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    ):
                        raise RuntimeError(
                            "Proposal receipt does not bind the evaluated target."
                        )
                    try:
                        next_state = commit_trajectory_target_emission(
                            state,
                            inputs.preparation,
                            emission_receipt_id=receipt.proposal_id,
                            emission_receipt_hash=receipt.semantic_content_hash,
                            emitted_action_hash=(
                                proposal.action.semantic_content_hash
                            ),
                        )
                    except Exception:
                        self.composition.request_hold(
                            reason=(
                                "trajectory_publication_receipt_commit_failed"
                            ),
                            source_event_ids=(receipt.proposal_id,),
                            force_cancellation=True,
                        )
                        raise
                    self._state = next_state
                    receipt_id = receipt.proposal_id
                    receipt_hash = receipt.semantic_content_hash
            if inputs.output_status == "no_action":
                status = "no_action"
            elif inputs.output_status == "invalid":
                status = "invalid"
            elif receipt_id is None:
                status = "proposal_not_admitted"
            else:
                status = "proposal_admitted"
            return MujocoTrajectoryPublicationCycle(
                profile_id=self.profile.profile_id,
                status=status,
                state_before_hash=state.semantic_content_hash,
                state_after=next_state,
                request=request,
                input_preparation=inputs,
                composition_step=composition_step,
                emission_receipt_id=receipt_id,
                emission_receipt_hash=receipt_hash,
            )


__all__ = [
    "MUJOCO_TRAJECTORY_PUBLICATION_CYCLE_HASH_DOMAIN_DEVELOPMENT",
    "MUJOCO_TRAJECTORY_PUBLICATION_PROFILE_OFF",
    "MUJOCO_TRAJECTORY_PUBLICATION_REQUEST_HASH_DOMAIN_DEVELOPMENT",
    "MujocoTrajectoryPublicationController",
    "MujocoTrajectoryPublicationCycle",
    "MujocoTrajectoryPublicationProfile",
    "MujocoTrajectoryPublicationRequest",
]
