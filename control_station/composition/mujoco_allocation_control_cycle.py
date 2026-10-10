"""Thin MuJoCo binding for an allocation-aware Sim-HIL control cycle.

The caller supplies source admission and allocation-basis production because
those operations depend on the selected policy, teleoperation device, role
mapping, and activity profile.  This binding only joins those already-typed
operations to the existing allocation context worker, exact proposal gate,
and one lower ``MujocoDeployRuntime.step`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

from control_composition.activation import CompositionActivationScope
from control_composition.allocation import (
    ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_EXISTING,
    ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW,
    ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY,
    COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
    AllocationPublicationBasis,
)
from control_station.sim_hil_profile.allocation_control_cycle import (
    SimHILAllocationActivationPrime,
    SimHILAllocationActivationBinding,
    SimHILAllocationCyclePublication,
)
from robot_motion.action import AuthoritativeControlStep
from robot_motion.deploy import PolicyActionEmission
from sim.mujoco.deploy.runtime import MujocoDeployRuntime

from .allocation_publication import (
    StationAllocationPublicationCycle,
    StationAllocationPublicationPort,
)
from .mujoco_allocation_publication import (
    MujocoAllocationProposalPublicationController,
)


@dataclass(frozen=True)
class MujocoAllocationSourcePublication:
    """Owner-preserving results of one profile-specific source update."""

    observed_at: float
    observed_clock_domain_id: str
    human_step: Any | None = None
    human_submission: Any | None = None
    policy_submission: Any | None = None
    input_publication: Any | None = None

    def __post_init__(self) -> None:
        observed_at = float(self.observed_at)
        if not math.isfinite(observed_at):
            raise ValueError("observed_at must be finite.")
        clock_domain = str(self.observed_clock_domain_id).strip()
        if not clock_domain:
            raise ValueError("observed_clock_domain_id must be non-empty.")
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(
            self,
            "observed_clock_domain_id",
            clock_domain,
        )


MujocoAllocationSourcePublisher = Callable[
    [float, PolicyActionEmission | None, Any, bool, float],
    MujocoAllocationSourcePublication,
]
MujocoAllocationBasisFactory = Callable[
    [float, float],
    AllocationPublicationBasis | None,
]


class MujocoAllocationControlCycleBinding(SimHILAllocationActivationBinding):
    """One activation-local binding with no source or authority inference."""

    def __init__(
        self,
        *,
        activation_scope: CompositionActivationScope,
        runtime: MujocoDeployRuntime,
        allocation_publication_port: StationAllocationPublicationPort,
        proposal_publication_controller: MujocoAllocationProposalPublicationController,
        source_publisher: MujocoAllocationSourcePublisher,
        basis_factory: MujocoAllocationBasisFactory,
        operation_controller: Any | None = None,
        physics_ticks_after_dispatch: int = 0,
        suspend_policy_during_absolute_owner: bool = False,
    ) -> None:
        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be typed.")
        if not isinstance(runtime, MujocoDeployRuntime):
            raise TypeError("runtime must be MujocoDeployRuntime.")
        if not isinstance(
            allocation_publication_port,
            StationAllocationPublicationPort,
        ):
            raise TypeError("allocation_publication_port must be typed.")
        if not isinstance(
            proposal_publication_controller,
            MujocoAllocationProposalPublicationController,
        ):
            raise TypeError("proposal_publication_controller must be typed.")
        if not callable(source_publisher):
            raise TypeError("source_publisher must be callable.")
        if not callable(basis_factory):
            raise TypeError("basis_factory must be callable.")
        if not isinstance(suspend_policy_during_absolute_owner, bool):
            raise TypeError(
                "suspend_policy_during_absolute_owner must be bool."
            )
        physics_ticks = physics_ticks_after_dispatch
        if (
            isinstance(physics_ticks, bool)
            or not isinstance(physics_ticks, int)
            or physics_ticks < 0
        ):
            raise ValueError(
                "physics_ticks_after_dispatch must be a non-negative integer."
            )
        allocation_runtime = allocation_publication_port.allocation_runtime
        if allocation_runtime is None:
            raise ValueError("Allocation publication port is unbound.")
        if (
            allocation_runtime.activation_scope.activation_id
            != activation_scope.activation_id
            or allocation_runtime.activation_scope.semantic_content_hash
            != activation_scope.semantic_content_hash
            or allocation_runtime.runtime_session_id
            != runtime.config.runtime_session_id
            or allocation_runtime.control_domain_id != runtime.config.control_domain_id
        ):
            raise ValueError("Allocation publication and runtime activation differ.")
        if (
            proposal_publication_controller.runtime is not runtime
            or proposal_publication_controller.activation_scope is None
            or proposal_publication_controller.activation_scope.activation_id
            != activation_scope.activation_id
            or proposal_publication_controller.activation_scope.semantic_content_hash
            != activation_scope.semantic_content_hash
            or proposal_publication_controller.allocation_publication_port
            is not allocation_publication_port
        ):
            raise ValueError("Proposal publication belongs to another binding.")
        self._activation_scope = activation_scope
        self.runtime = runtime
        self.allocation_publication_port = allocation_publication_port
        self.proposal_publication_controller = proposal_publication_controller
        self.source_publisher = source_publisher
        self.basis_factory = basis_factory
        self.operation_controller = operation_controller
        self.physics_ticks_after_dispatch = physics_ticks
        self.suspend_policy_during_absolute_owner = (
            suspend_policy_during_absolute_owner
        )
        self._closed = False
        self._close_reason: str | None = None
        # Construction only occurs from the application port's activate hook,
        # after its Station publication barrier is active.
        self.allocation_publication_port.start()

    def submit_operation_trigger(self, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("MuJoCo allocation binding is closed.")
        controller = self.operation_controller
        submit = getattr(controller, "submit_button_event", None)
        if not callable(submit):
            raise RuntimeError("Allocation operation triggers are not configured.")
        return submit(**kwargs)

    def operation_status(self) -> dict[str, object]:
        controller = self.operation_controller
        status = getattr(controller, "status", None)
        if not callable(status):
            return {"supported_signals": []}
        return dict(status())

    def policy_emission_required(self) -> bool:
        """Report whether the selected allocation mode consumes Policy.

        Human absolute ownership is an alternative complete source.  Keeping
        synchronous Policy production in that mode would add model latency to
        every teleoperation cycle even though the result cannot be selected.
        All other modes retain Policy as the baseline contributor.
        """

        if not self.suspend_policy_during_absolute_owner:
            return True
        state = self.allocation_publication_port.allocation_runtime
        if state is None:
            raise RuntimeError("Allocation publication port is unbound.")
        override = state.allocation_override_state
        return not (
            override is not None
            and override.requested_mode
            == COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER
        )

    def synchronize_operation_context(
        self,
        *,
        now: float,
        target_effective_at: float,
        timeout_sec: float,
    ) -> Any:
        """Publish the post-operation allocation before target cycles resume."""

        if self._closed:
            raise RuntimeError("MuJoCo allocation binding is closed.")
        timeout = float(timeout_sec)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_sec must be positive.")
        deadline = time.monotonic() + timeout
        cutoff_at = max(float(now), float(self.runtime.clock()))
        target = max(float(target_effective_at), cutoff_at)
        basis = self.basis_factory(cutoff_at, target)
        # Activity evidence intentionally requires two causal samples.  The
        # activation prime supplies the first; poll the already-attached Human
        # source while execution remains in HOLD until the second sample can
        # close a post-operation allocation cut.
        while basis is None and time.monotonic() < deadline:
            source = self.source_publisher(
                cutoff_at,
                None,
                None,
                True,
                target,
            )
            cutoff_at = max(cutoff_at, source.observed_at, float(self.runtime.clock()))
            target = max(target, cutoff_at)
            basis = self.basis_factory(cutoff_at, target)
            if basis is None:
                time.sleep(0.005)
        if not isinstance(basis, AllocationPublicationBasis):
            raise TimeoutError(
                "Timed out acquiring a post-operation allocation basis."
            )
        cycle = self.allocation_publication_port.offer_basis(lambda: basis)
        outcome = cycle.outcome
        if (
            outcome is None
            or outcome.outcome_kind != ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY
            or outcome.attempt_id is None
        ):
            raise RuntimeError(
                "Allocation operation could not prepare a context publication."
            )
        terminal = self.allocation_publication_port.wait_for_terminal_attempt(
            outcome.attempt_id,
            timeout_sec=max(0.0, deadline - time.monotonic()),
        )
        if terminal is None:
            raise TimeoutError(
                "Timed out publishing the selected allocation context."
            )
        if terminal.outcome_kind not in {
            ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW,
            ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_EXISTING,
        }:
            raise RuntimeError(
                "Selected allocation context was not admitted: "
                f"{terminal.outcome_kind}:{terminal.reason}"
            )
        return terminal

    @property
    def activation_scope(self) -> CompositionActivationScope:
        return self._activation_scope

    def publish_cycle(
        self,
        *,
        now: float,
        policy_emission: PolicyActionEmission | None,
        poll_human: bool,
        target_effective_at: float,
    ) -> SimHILAllocationCyclePublication:
        if self._closed:
            raise RuntimeError("MuJoCo allocation binding is closed.")
        source, allocation_cycle, cutoff_at, target = (
            self._publish_sources_and_offer_basis(
                now=now,
                policy_emission=policy_emission,
                human_emission=None,
                poll_human=poll_human,
                target_effective_at=target_effective_at,
            )
        )
        proposal = self.proposal_publication_controller.publish_at(
            target_effective_at=target,
            cutoff_at=cutoff_at,
        )
        return SimHILAllocationCyclePublication(
            proposal_publication=proposal,
            observed_at=cutoff_at,
            human_step=source.human_step,
            human_submission=source.human_submission,
            policy_submission=source.policy_submission,
            input_publication=source.input_publication,
            allocation_publication=allocation_cycle,
        )

    def prime_sources(
        self,
        *,
        now: float,
        policy_emission: PolicyActionEmission | None,
        human_emission: Any,
        poll_human: bool,
        target_effective_at: float,
    ) -> SimHILAllocationActivationPrime:
        if self._closed:
            raise RuntimeError("MuJoCo allocation binding is closed.")
        source, allocation_cycle, cutoff_at, _target = (
            self._publish_sources_and_offer_basis(
                now=now,
                policy_emission=policy_emission,
                human_emission=human_emission,
                poll_human=poll_human,
                target_effective_at=target_effective_at,
            )
        )
        return SimHILAllocationActivationPrime(
            observed_at=cutoff_at,
            human_step=source.human_step,
            human_submission=source.human_submission,
            policy_submission=source.policy_submission,
            input_publication=source.input_publication,
            allocation_publication=allocation_cycle,
        )

    def _publish_sources_and_offer_basis(
        self,
        *,
        now: float,
        policy_emission: PolicyActionEmission | None,
        human_emission: Any,
        poll_human: bool,
        target_effective_at: float,
    ) -> tuple[
        MujocoAllocationSourcePublication,
        StationAllocationPublicationCycle | None,
        float,
        float,
    ]:
        source = self.source_publisher(
            float(now),
            policy_emission,
            human_emission,
            bool(poll_human),
            float(target_effective_at),
        )
        if not isinstance(source, MujocoAllocationSourcePublication):
            raise TypeError(
                "source_publisher must return MujocoAllocationSourcePublication."
            )
        if source.observed_clock_domain_id != self.runtime.config.clock_domain_id:
            raise ValueError(
                "Source publication and MuJoCo runtime require one mapped clock."
            )
        cutoff_at = max(float(now), source.observed_at)
        target = float(target_effective_at)
        if self.runtime.reference_activation_spec.activation_mode == (
            "immediate_target"
        ):
            target = max(target, cutoff_at)
        basis = self.basis_factory(cutoff_at, target)
        allocation_cycle: StationAllocationPublicationCycle | None = None
        if basis is not None:
            if not isinstance(basis, AllocationPublicationBasis):
                raise TypeError(
                    "basis_factory must return AllocationPublicationBasis or None."
                )
            allocation_cycle = self.allocation_publication_port.offer_basis(
                lambda: basis
            )
        # Basis construction/publication may admit receiver facts after the
        # source cut. Close that boundary before proposal evaluation; an
        # immediate target follows the new cut, while scheduled targets keep
        # their exact deadline and remain subject to lower late admission.
        cutoff_at = max(cutoff_at, float(self.runtime.clock()))
        if self.runtime.reference_activation_spec.activation_mode == (
            "immediate_target"
        ):
            target = max(target, cutoff_at)
        return source, allocation_cycle, cutoff_at, target

    def execute_lower(self, *, now: float) -> AuthoritativeControlStep:
        if self._closed:
            raise RuntimeError("MuJoCo allocation binding is closed.")
        return self.runtime.step(
            now=max(float(now), float(self.runtime.clock())),
            physics_ticks_after_dispatch=self.physics_ticks_after_dispatch,
        )

    def fence(self, *, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_reason = str(reason)
        self.allocation_publication_port.close(
            reason=self._close_reason,
            timeout_sec=0.0,
        )

    def join(self, *, timeout_sec: float) -> bool:
        return self.allocation_publication_port.close(
            reason=(
                self._close_reason or "mujoco_allocation_control_cycle_binding_joined"
            ),
            timeout_sec=float(timeout_sec),
        )


__all__ = [
    "MujocoAllocationBasisFactory",
    "MujocoAllocationControlCycleBinding",
    "MujocoAllocationSourcePublication",
    "MujocoAllocationSourcePublisher",
]
