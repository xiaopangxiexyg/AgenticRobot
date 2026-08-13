"""Passive policy/human composition shadow for sim or real applications.

Producer callbacks only enqueue already-created immutable emissions.  A
Control Station thread later admits their inputs and evaluates one composition
cut.  No proposal publication or execution operation exists in this module.
"""

from __future__ import annotations

import itertools
import threading
from typing import Callable

from control_composition.activation import CompositionActivationScope
from control_composition.composition import CompositionSpec
from control_composition.eligibility import CompositionSourceEligibilitySpec
from control_composition.observability import SourceDifferenceSpec
from control_composition.runtime import ControlCompositionRuntime
from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.deploy import PolicyActionEmission
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanProposalAdapter,
)

from ..sources.policy import PolicyCompositionInputAdapter
from .policy_human_shadow_contracts import (
    POLICY_HUMAN_COMPOSITION_SHADOW_PROFILE_OFF,
    CompositionShadowInputSubmission,
    PassiveCompositionIngressStatus,
    PolicyHumanCompositionShadowCycle,
    PolicyHumanCompositionShadowProfile,
    normalize_finite_float,
    normalize_required_text,
)
from .policy_human_shadow_ingress import (
    LatestPreservingShadowIngress,
    ObservedShadowEmission,
)
from .policy_human_shadow_sources import PolicyHumanShadowInputBinder


class PolicyHumanCompositionShadowSession:
    """Bounded observers plus an application-thread composition evaluator."""

    nonblocking_bounded = True

    def __init__(
        self,
        *,
        profile: PolicyHumanCompositionShadowProfile,
        runtime: ControlCompositionRuntime,
        composition_spec: CompositionSpec,
        policy_input_adapter: PolicyCompositionInputAdapter,
        human_input_adapter: HumanProposalAdapter,
        human_role: str,
        human_effective_horizon_sec: float,
        activation_scope: CompositionActivationScope | None = None,
        required_context_channels: tuple[str, ...] = (),
        source_difference_spec: SourceDifferenceSpec | None = None,
        source_eligibility_spec: CompositionSourceEligibilitySpec | None = None,
        residual_anchor_lifecycle_scope_id: str | None = None,
        source_fact_sink: Callable[[ControlTraceFact], object] | None = None,
        policy_queue_capacity: int = 16,
        human_queue_capacity: int = 16,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._validate_types(
            profile,
            runtime,
            composition_spec,
            policy_input_adapter,
            human_input_adapter,
            activation_scope,
            source_difference_spec,
            source_eligibility_spec,
            source_fact_sink,
        )
        resolved_clock = runtime.clock if clock is None else clock
        if not callable(resolved_clock):
            raise TypeError("clock must be callable.")
        policy_role = policy_input_adapter.binding.role
        resolved_human_role = normalize_required_text(
            human_role,
            name="human_role",
        )
        if policy_role == resolved_human_role:
            raise ValueError("Policy and human roles must be distinct.")
        if not {policy_role, resolved_human_role}.issubset(
            composition_spec.input_roles
        ):
            raise ValueError("Policy/human roles must belong to CompositionSpec.")
        self._validate_runtime_bindings(
            runtime,
            policy_input_adapter,
            human_input_adapter,
        )
        horizon = normalize_finite_float(
            human_effective_horizon_sec,
            name="human_effective_horizon_sec",
        )
        if horizon <= 0.0:
            raise ValueError("human_effective_horizon_sec must be positive.")
        channels = tuple(
            normalize_required_text(item, name="required_context_channel")
            for item in required_context_channels
        )
        if len(channels) != len(set(channels)):
            raise ValueError("required_context_channels must be unique.")

        self.profile = profile
        self.runtime = runtime
        self.composition_spec = composition_spec
        self.policy_role = policy_role
        self.human_role = resolved_human_role
        self.activation_scope = activation_scope
        self.required_context_channels = channels
        self.source_difference_spec = source_difference_spec
        self.source_eligibility_spec = source_eligibility_spec
        self.residual_anchor_lifecycle_scope_id = (
            None
            if residual_anchor_lifecycle_scope_id is None
            else normalize_required_text(
                residual_anchor_lifecycle_scope_id,
                name="residual_anchor_lifecycle_scope_id",
            )
        )
        self.clock = resolved_clock
        clock_domain_id = runtime.coordinator.clock_domain.clock_domain_id
        self._source_binder = PolicyHumanShadowInputBinder(
            runtime=runtime,
            policy_input_adapter=policy_input_adapter,
            human_input_adapter=human_input_adapter,
            human_role=resolved_human_role,
            human_effective_horizon_sec=horizon,
            clock_domain_id=clock_domain_id,
            activation_scope=activation_scope,
            source_fact_sink=source_fact_sink,
        )
        self._policy_ingress = LatestPreservingShadowIngress(
            "policy",
            capacity=policy_queue_capacity,
        )
        self._human_ingress = LatestPreservingShadowIngress(
            "human",
            capacity=human_queue_capacity,
        )
        self._ingress_sequence = itertools.count()
        self._evaluation_lock = threading.RLock()
        self._last_target_effective_at: float | None = None

    @staticmethod
    def _validate_types(
        profile,
        runtime,
        composition_spec,
        policy_input_adapter,
        human_input_adapter,
        activation_scope,
        source_difference_spec,
        source_eligibility_spec,
        source_fact_sink,
    ) -> None:
        expected = (
            (profile, PolicyHumanCompositionShadowProfile, "profile"),
            (runtime, ControlCompositionRuntime, "runtime"),
            (composition_spec, CompositionSpec, "composition_spec"),
            (
                policy_input_adapter,
                PolicyCompositionInputAdapter,
                "policy_input_adapter",
            ),
            (human_input_adapter, HumanProposalAdapter, "human_input_adapter"),
        )
        for value, value_type, name in expected:
            if not isinstance(value, value_type):
                raise TypeError(f"{name} must be {value_type.__name__}.")
        optional = (
            (activation_scope, CompositionActivationScope, "activation_scope"),
            (
                source_difference_spec,
                SourceDifferenceSpec,
                "source_difference_spec",
            ),
            (
                source_eligibility_spec,
                CompositionSourceEligibilitySpec,
                "source_eligibility_spec",
            ),
        )
        for value, value_type, name in optional:
            if value is not None and not isinstance(value, value_type):
                raise TypeError(f"{name} must be {value_type.__name__} or None.")
        if source_fact_sink is not None and not callable(source_fact_sink):
            raise TypeError("source_fact_sink must be callable or None.")

    @staticmethod
    def _validate_runtime_bindings(
        runtime: ControlCompositionRuntime,
        policy_adapter: PolicyCompositionInputAdapter,
        human_adapter: HumanProposalAdapter,
    ) -> None:
        policy = policy_adapter.binding
        human = human_adapter.config
        comparisons = (
            ("policy session", policy.runtime_session_id, runtime.runtime_session_id),
            ("policy domain", policy.control_domain_id, runtime.control_domain_id),
            ("human session", human.runtime_session_id, runtime.runtime_session_id),
            ("human domain", human.control_domain_id, runtime.control_domain_id),
            (
                "human embodiment",
                human.embodiment_instance_id,
                runtime.embodiment_instance_id,
            ),
        )
        for name, actual, expected in comparisons:
            if actual != expected:
                raise ValueError(f"{name} does not match composition runtime.")
        receiver_clock = runtime.coordinator.clock_domain.clock_domain_id
        if human.effective_clock_domain_id != receiver_clock:
            raise ValueError(
                "Human effective clock must be the composition receiver clock."
            )

    def put_nowait(self, emission: PolicyActionEmission) -> None:
        """Passive deploy observer; never waits for Station evaluation."""

        if not isinstance(emission, PolicyActionEmission):
            raise TypeError("Policy shadow ingress accepts PolicyActionEmission only.")
        if not self.profile.enabled:
            self._policy_ingress.ignore_while_disabled()
            return
        self._policy_ingress.offer(
            ObservedShadowEmission(
                next(self._ingress_sequence),
                normalize_finite_float(
                    self.clock(),
                    name="policy observed_at",
                ),
                emission,
            )
        )

    def observe_human_emission(self, emission: HumanCommandEmission) -> bool:
        """Passive callback for ``PassiveHumanCommandObserverFanout``."""

        if not isinstance(emission, HumanCommandEmission):
            raise TypeError("Human shadow ingress accepts HumanCommandEmission only.")
        if not self.profile.enabled:
            self._human_ingress.ignore_while_disabled()
            return True
        return self._human_ingress.offer(
            ObservedShadowEmission(
                next(self._ingress_sequence),
                normalize_finite_float(
                    self.clock(),
                    name="human observed_at",
                ),
                emission,
            )
        )

    def ingress_status(
        self,
    ) -> tuple[PassiveCompositionIngressStatus, PassiveCompositionIngressStatus]:
        return self._policy_ingress.status(), self._human_ingress.status()

    def evaluate_at(
        self,
        *,
        target_effective_at: float,
        cutoff_at: float | None = None,
    ) -> PolicyHumanCompositionShadowCycle:
        """Drain one causal prefix and evaluate without proposal publication."""

        if not self.profile.enabled:
            policy_status, human_status = self.ingress_status()
            return PolicyHumanCompositionShadowCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
                policy_ingress=policy_status,
                human_ingress=human_status,
            )
        target = normalize_finite_float(
            target_effective_at,
            name="target_effective_at",
        )
        cutoff = (
            None
            if cutoff_at is None
            else normalize_finite_float(cutoff_at, name="cutoff_at")
        )
        with self._evaluation_lock:
            if (
                self._last_target_effective_at is not None
                and target <= self._last_target_effective_at
            ):
                raise ValueError(
                    "Composition shadow target times must be strictly increasing; "
                    "this session has no future-target replacement protocol."
                )
            self._last_target_effective_at = target
            pending = sorted(
                (
                    *self._policy_ingress.drain_nowait(),
                    *self._human_ingress.drain_nowait(),
                ),
                key=lambda item: item.ingress_sequence,
            )
            submissions = tuple(
                self._source_binder.submit_policy(
                    item.emission,
                    observed_at=item.observed_at,
                )
                if isinstance(item.emission, PolicyActionEmission)
                else self._source_binder.submit_human(
                    item.emission,
                    observed_at=item.observed_at,
                )
                for item in pending
            )
            try:
                evaluation = self.runtime.evaluate_composition(
                    spec=self.composition_spec,
                    target_effective_at=target,
                    required_context_channels=self.required_context_channels,
                    cutoff_at=cutoff,
                    source_difference_spec=self.source_difference_spec,
                    activation_scope=self.activation_scope,
                    source_eligibility_spec=self.source_eligibility_spec,
                    residual_anchor_lifecycle_scope_id=(
                        self.residual_anchor_lifecycle_scope_id
                    ),
                )
                if evaluation.proposal_admission is not None:
                    raise RuntimeError(
                        "evaluate_composition unexpectedly published a proposal."
                    )
            except Exception as error:
                policy_status, human_status = self.ingress_status()
                return PolicyHumanCompositionShadowCycle(
                    profile_id=self.profile.profile_id,
                    status="evaluation_error",
                    target_effective_at=target,
                    submissions=submissions,
                    policy_ingress=policy_status,
                    human_ingress=human_status,
                    error=f"{type(error).__name__}:{error}",
                )
            policy_status, human_status = self.ingress_status()
            return PolicyHumanCompositionShadowCycle(
                profile_id=self.profile.profile_id,
                status="evaluated",
                target_effective_at=target,
                submissions=submissions,
                evaluation=evaluation,
                policy_ingress=policy_status,
                human_ingress=human_status,
            )


__all__ = [
    "POLICY_HUMAN_COMPOSITION_SHADOW_PROFILE_OFF",
    "CompositionShadowInputSubmission",
    "PassiveCompositionIngressStatus",
    "PolicyHumanCompositionShadowCycle",
    "PolicyHumanCompositionShadowProfile",
    "PolicyHumanCompositionShadowSession",
]
