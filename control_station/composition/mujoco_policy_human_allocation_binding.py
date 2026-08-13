"""Activation-local assembly for the MuJoCo policy/human allocation profile.

The factory joins existing typed ports. It does not construct a model, human
device, MuJoCo target, composition session, authority, safety, or executor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

from control_composition.activation import CompositionActivationScope
from control_composition.allocation import (
    AllocationPublicationRuntime,
    CompositionAllocationOverrideRequest,
    empty_composition_allocation_override_state,
    empty_composition_allocation_state,
)
from control_composition.composition import DerivedCompositionInputRoleBinding
from control_station.publication_barrier import StationPublicationBarrier
from sim.mujoco.deploy.runtime import MujocoDeployRuntime
from teleop_runtime.action.human_proposal import HumanProposalAdapter

from ..sources.policy import PolicyCompositionInputAdapter
from .allocation_publication import StationAllocationPublicationPort
from .derived_input_publication import StationDerivedInputPublicationPort
from .mujoco_allocation_control_cycle import MujocoAllocationControlCycleBinding
from .mujoco_allocation_publication import (
    MujocoAllocationProposalPublicationController,
)
from .mujoco_policy_human_allocation import (
    MujocoPolicyHumanAllocationProfileBundle,
    MujocoPolicyHumanAllocationStationProfiles,
)
from .mujoco_policy_human_allocation_admission import (
    MujocoPolicyHumanAllocationInputAdmission,
)
from .mujoco_policy_human_allocation_input import (
    MujocoPolicyHumanAllocationInputController,
)
from .mujoco_policy_human_allocation_source import (
    MujocoPolicyHumanAllocationSourcePublisher,
)
from .mujoco_policy_human_allocation_operations import (
    MujocoPolicyHumanAllocationOperationController,
)


@dataclass(frozen=True)
class MujocoPolicyHumanAllocationBindingConfig:
    allocation_scope_id: str
    allocation_producer_id: str
    allocation_producer_epoch: int
    anchor_lifecycle_producer_id: str
    anchor_lifecycle_producer_epoch: int
    control_clock_reset_epoch: int = 0
    physics_ticks_after_dispatch: int = 0

    def __post_init__(self) -> None:
        for name in (
            "allocation_scope_id",
            "allocation_producer_id",
            "anchor_lifecycle_producer_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        for name in (
            "allocation_producer_epoch",
            "anchor_lifecycle_producer_epoch",
            "control_clock_reset_epoch",
            "physics_ticks_after_dispatch",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")


class MujocoPolicyHumanAllocationBindingFactory:
    """Create all activation-scoped publishers behind one Station barrier."""

    def __init__(
        self,
        *,
        runtime: MujocoDeployRuntime,
        bundle: MujocoPolicyHumanAllocationProfileBundle,
        station_profiles: MujocoPolicyHumanAllocationStationProfiles,
        publication_barrier: StationPublicationBarrier,
        human_source: Any,
        policy_input_adapter: PolicyCompositionInputAdapter,
        human_adapter: HumanProposalAdapter,
        human_role_bindings: Sequence[DerivedCompositionInputRoleBinding],
        config: MujocoPolicyHumanAllocationBindingConfig,
    ) -> None:
        if not isinstance(runtime, MujocoDeployRuntime):
            raise TypeError("runtime must be MujocoDeployRuntime.")
        if not isinstance(bundle, MujocoPolicyHumanAllocationProfileBundle):
            raise TypeError("bundle must be a typed allocation profile bundle.")
        if not bundle.config.enabled:
            raise ValueError("Allocation binding requires an enabled profile.")
        if not isinstance(
            station_profiles,
            MujocoPolicyHumanAllocationStationProfiles,
        ):
            raise TypeError("station_profiles must be typed.")
        if station_profiles.proposal.promotable_allocation_modes != (
            bundle.config.proposal_promotable_modes
        ):
            raise ValueError("Proposal modes differ from the allocation profile.")
        if not isinstance(publication_barrier, StationPublicationBarrier):
            raise TypeError("publication_barrier must be typed.")
        if not callable(getattr(human_source, "step", None)):
            raise TypeError("human_source must expose step(now=...).")
        if not isinstance(policy_input_adapter, PolicyCompositionInputAdapter):
            raise TypeError("policy_input_adapter must be typed.")
        if not isinstance(human_adapter, HumanProposalAdapter):
            raise TypeError("human_adapter must be typed.")
        if not isinstance(config, MujocoPolicyHumanAllocationBindingConfig):
            raise TypeError("config must be typed.")
        bindings = tuple(human_role_bindings)
        expected_roles = {
            bundle.config.absolute_overlay_role,
            bundle.config.residual_role,
        }
        if {binding.input_role for binding in bindings} != expected_roles:
            raise ValueError("Human role bindings do not match the profile.")
        policy_binding = policy_input_adapter.binding
        if policy_binding.role != bundle.config.baseline_role:
            raise ValueError("Policy input adapter does not bind the baseline role.")
        if policy_binding.action_schema.semantic_content_hash != (
            bundle.composition_contracts.output_action_schema.semantic_content_hash
        ):
            raise ValueError("Policy input schema differs from the profile output.")
        self.runtime = runtime
        self.bundle = bundle
        self.station_profiles = station_profiles
        self.publication_barrier = publication_barrier
        self.human_source = human_source
        self.policy_input_adapter = policy_input_adapter
        self.human_adapter = human_adapter
        self.human_role_bindings = bindings
        self.config = config

    @staticmethod
    def _activation_epoch(base_epoch: int, session_epoch: int) -> int:
        if session_epoch < 1:
            raise ValueError("Allocation activation session_epoch must start at one.")
        return int(base_epoch) + int(session_epoch) - 1

    def required_producer_epochs(
        self,
        activation_scope: CompositionActivationScope,
    ) -> dict[str, int]:
        """Return non-policy receiver producers required by this activation."""

        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be typed.")
        session_epoch = activation_scope.session_epoch
        values = {
            binding.producer_id: self._activation_epoch(
                binding.producer_epoch,
                session_epoch,
            )
            for binding in self.human_role_bindings
        }
        for producer_id, base_epoch in (
            (
                self.config.anchor_lifecycle_producer_id,
                self.config.anchor_lifecycle_producer_epoch,
            ),
            (
                self.config.allocation_producer_id,
                self.config.allocation_producer_epoch,
            ),
        ):
            effective = self._activation_epoch(base_epoch, session_epoch)
            previous = values.get(producer_id)
            if previous is not None and previous != effective:
                raise ValueError("One producer ID has conflicting activation epochs.")
            values[producer_id] = effective
        return values

    def __call__(
        self,
        activation_scope: CompositionActivationScope,
    ) -> MujocoAllocationControlCycleBinding:
        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be typed.")
        runtime = self.runtime
        bundle = self.bundle
        config = self.config
        if (
            activation_scope.clock_domain_id != runtime.config.clock_domain_id
            or activation_scope.required_roles
            != bundle.composition_contracts.composition_spec.required_roles
        ):
            raise ValueError("Activation scope differs from the allocation profile.")
        effective_epochs = self.required_producer_epochs(activation_scope)
        bundle.composition_contracts.strategy.activate_allocation_transition_continuity(
            activation_scope
        )
        human_role_bindings = tuple(
            replace(
                binding,
                producer_epoch=effective_epochs[binding.producer_id],
            )
            for binding in self.human_role_bindings
        )

        allocation_runtime = AllocationPublicationRuntime(
            allocation_scope_id=config.allocation_scope_id,
            allocation_spec=bundle.allocation_spec,
            activation_scope=activation_scope,
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            control_clock_reset_epoch=config.control_clock_reset_epoch,
            context_channel=bundle.config.allocation_context_channel,
            producer_id=config.allocation_producer_id,
            producer_epoch=effective_epochs[config.allocation_producer_id],
            initial_reducer_state=empty_composition_allocation_state(
                config.allocation_producer_id,
                spec=bundle.allocation_spec,
            ),
            initial_override_state=empty_composition_allocation_override_state(
                runtime_session_id=runtime.config.runtime_session_id,
                control_domain_id=runtime.config.control_domain_id,
                composition_activation_id=activation_scope.activation_id,
                composition_activation_hash=(
                    activation_scope.semantic_content_hash
                ),
                allocation_spec_hash=bundle.allocation_spec.semantic_content_hash,
                clock_domain_id=runtime.config.clock_domain_id,
                component_ids=bundle.allocation_spec.component_ids,
                allowed_modes=bundle.config.proposal_promotable_modes,
            ),
            persistent_override_selection_lease_sec=(
                bundle.config.persistent_override_selection_lease_sec
            ),
            clock=runtime.clock,
        )
        initial_mode = bundle.config.initial_override_mode
        if initial_mode is not None:
            selected_at = max(
                float(activation_scope.input_barrier_at),
                float(runtime.clock()),
            )
            allocation_runtime.apply_override(
                CompositionAllocationOverrideRequest(
                    runtime_session_id=runtime.config.runtime_session_id,
                    control_domain_id=runtime.config.control_domain_id,
                    composition_activation_id=activation_scope.activation_id,
                    composition_activation_hash=(
                        activation_scope.semantic_content_hash
                    ),
                    allocation_spec_hash=(
                        bundle.allocation_spec.semantic_content_hash
                    ),
                    component_ids=bundle.allocation_spec.component_ids,
                    requested_mode=initial_mode,
                    requested_at=selected_at,
                    available_at=selected_at,
                    clock_domain_id=runtime.config.clock_domain_id,
                    source_fact_ids=(activation_scope.activation_id,),
                    source_fact_hashes=(
                        activation_scope.semantic_content_hash,
                    ),
                )
            )
        allocation_port = StationAllocationPublicationPort(
            profile=self.station_profiles.allocation_context,
            allocation_runtime=allocation_runtime,
            publication_barrier=self.publication_barrier,
            admit_context=runtime.submit_composition_context,
            resolve_context=runtime.resolve_composition_context,
            worker_name=f"{bundle.config.profile_id}-allocation-context",
        )
        lifecycle_spec = bundle.composition_contracts.residual_anchor_lifecycle_spec
        if lifecycle_spec is None:
            raise RuntimeError("Allocation profile lacks anchor lifecycle semantics.")
        derived_port = StationDerivedInputPublicationPort(
            profile=self.station_profiles.derived_input,
            activation_scope=activation_scope,
            publication_barrier=self.publication_barrier,
            admit_batch=lambda batch: runtime.submit_composition_inputs(
                batch,
                activation_scope=activation_scope,
            ),
            runtime_session_id=runtime.config.runtime_session_id,
            control_domain_id=runtime.config.control_domain_id,
            lifecycle_producer_id=config.anchor_lifecycle_producer_id,
            lifecycle_producer_epoch=effective_epochs[
                config.anchor_lifecycle_producer_id
            ],
            lifecycle_scope_id=config.allocation_scope_id,
            lifecycle_spec=lifecycle_spec,
            admit_lifecycle=lambda draft: runtime.submit_residual_anchor_lifecycle(
                draft,
                spec=lifecycle_spec,
                activation_scope=activation_scope,
            ),
        )
        human_controller = MujocoPolicyHumanAllocationInputController(
            bundle=bundle,
            activation_scope=activation_scope,
            role_bindings=human_role_bindings,
            publication_port=derived_port,
            allocation_publication_port=allocation_port,
            clock_reset_epoch=config.control_clock_reset_epoch,
        )
        # Initial activation always needs an explicit anchor transaction. This
        # is not the later operator-triggered rebase path.
        anchor_at = max(
            float(activation_scope.input_barrier_at),
            float(runtime.clock()),
        )
        human_controller.request_anchor(
            requested_at=anchor_at,
            effective_not_before_target_at=anchor_at,
        )
        input_admission = MujocoPolicyHumanAllocationInputAdmission(
            runtime=runtime,
            activation_scope=activation_scope,
            policy_input_adapter=self.policy_input_adapter,
            human_adapter=self.human_adapter,
            human_input_controller=human_controller,
        )
        source_publisher = MujocoPolicyHumanAllocationSourcePublisher(
            human_source=self.human_source,
            publish_policy_input=input_admission.publish_policy,
            publish_human_input=input_admission.publish_human,
            clock=runtime.clock,
            clock_domain_id=runtime.config.clock_domain_id,
        )
        proposal_controller = MujocoAllocationProposalPublicationController(
            profile=self.station_profiles.proposal,
            runtime=runtime,
            composition_spec=bundle.composition_contracts.composition_spec,
            allocation_spec=bundle.allocation_spec,
            activation_scope=activation_scope,
            allocation_publication_port=allocation_port,
            publication_barrier=self.publication_barrier,
        )
        operation_controller = MujocoPolicyHumanAllocationOperationController(
            runtime=runtime,
            bundle=bundle,
            allocation_publication_port=allocation_port,
            human_input_controller=human_controller,
        )

        def capture_basis(cutoff_at: float, target_at: float):
            return human_controller.capture_basis(
                allocation_scope_id=config.allocation_scope_id,
                control_cut=runtime.capture_control_cut(cutoff_at=cutoff_at),
                requested_effective_from=target_at,
            )

        return MujocoAllocationControlCycleBinding(
            activation_scope=activation_scope,
            runtime=runtime,
            allocation_publication_port=allocation_port,
            proposal_publication_controller=proposal_controller,
            source_publisher=source_publisher,
            basis_factory=capture_basis,
            operation_controller=operation_controller,
            physics_ticks_after_dispatch=config.physics_ticks_after_dispatch,
            suspend_policy_during_absolute_owner=(
                bundle.config.baseline_role
                not in bundle.composition_contracts.composition_spec.required_roles
            ),
        )


__all__ = [
    "MujocoPolicyHumanAllocationBindingConfig",
    "MujocoPolicyHumanAllocationBindingFactory",
]
