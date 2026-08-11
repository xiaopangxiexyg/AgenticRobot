from __future__ import annotations

import pytest

from control_composition import (
    ResidualAnchorLifecycleAdmissionReceipt,
    ResidualAnchorLifecycleStore,
    build_composition_activation_scope,
)
from control_composition.allocation import (
    AllocationPublicationRuntime,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
    ComponentAllocationEvidencePolicy,
    ComponentAllocationPolicy,
    empty_composition_allocation_state,
)
from control_composition.composition import (
    CompositionInputStore,
    DerivedCompositionInputRoleBinding,
)
from control_composition.retention import TimeWindowRetentionSpec
from control_station import (
    StationOperationResultDraft,
    StationPublicationBarrier,
    admit_station_operation_result,
)
from control_station.composition import (
    MujocoPolicyHumanAllocationInputController,
    MujocoPolicyHumanAllocationProfileConfig,
    StationDerivedInputPublicationPort,
    StationDerivedInputPublicationProfile,
    StationAllocationPublicationPort,
    StationAllocationPublicationProfile,
    build_mujoco_policy_human_allocation_profile,
    build_mujoco_policy_human_allocation_station_profiles,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry
from tests.test_control_station_policy_human_residual_profile import (
    _action,
    _schema,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _policies():
    threshold = ComponentAllocationEvidencePolicy(
        relative_enter_level=1.0,
        relative_exit_level=0.5,
        absolute_enter_level=3.0,
        absolute_exit_level=2.0,
    )
    return tuple(
        ComponentAllocationPolicy(
            component_id=component_id,
            motion_source_role="human_activity",
            residual_source_role="human_residual",
            motion_policy=threshold,
            residual_magnitude_policy=threshold,
        )
        for component_id in ("arm", "hand")
    )


def _bundle(*, enabled: bool):
    return build_mujoco_policy_human_allocation_profile(
        absolute_action_schema=_schema(),
        frame_graph=CoordinateFrameGraph(
            graph_id="virtual-policy-human-allocation",
            frames=(
                CoordinateFrameSpec("arm/base"),
                CoordinateFrameSpec("arm/tool"),
            ),
            transforms=(),
        ),
        concrete_reference_frame_by_component={"arm": "arm/base"},
        concrete_controlled_frame_by_component={"arm": "arm/tool"},
        source_binding_id="virtual-human-source:1",
        source_binding_hash="virtual-human-source-hash:1",
        composer_producer_id="virtual-allocation-composer",
        component_policies=_policies(),
        window_duration_sec=0.5,
        evidence_ttl_sec=0.25,
        activity_normalization_scale_by_component={
            "arm": {"translation_displacement_m": 0.02},
            "hand": {"coordinate_linf_displacement_native": 0.1},
        },
        residual_normalization_scale_by_component={
            "arm": {"translation_from_anchor_m": 0.02},
            "hand": {"coordinate_linf_from_anchor_native": 0.1},
        },
        config=MujocoPolicyHumanAllocationProfileConfig(
            profile_id=("virtual-allocation" if enabled else "off"),
            enabled=enabled,
        ),
    )


def test_allocation_profile_defaults_to_scheme_a_promotion_modes() -> None:
    bundle = _bundle(enabled=True)

    assert bundle.config.proposal_promotable_modes == (
        COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
        COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
    )
    assert "absolute_overlay_owner" not in (bundle.config.proposal_promotable_modes)


def test_station_profiles_carry_scheme_a_into_the_only_proposal_gate() -> None:
    bundle = _bundle(enabled=True)

    profiles = build_mujoco_policy_human_allocation_station_profiles(bundle)

    assert profiles.derived_input.enabled
    assert profiles.allocation_context.enabled
    assert profiles.proposal.enabled
    assert profiles.proposal.send_actions
    assert profiles.proposal.allocation_context_channel == (
        bundle.config.allocation_context_channel
    )
    assert profiles.proposal.promotable_allocation_modes == (
        COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
        COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
    )


def test_station_profiles_reject_the_default_off_bundle() -> None:
    with pytest.raises(ValueError, match="enabled allocation profile"):
        build_mujoco_policy_human_allocation_station_profiles(
            _bundle(enabled=False)
        )


def _activation(bundle):
    contracts = bundle.composition_contracts
    return build_composition_activation_scope(
        operation_spec_id=bundle.config.profile_id,
        operation_spec_hash=contracts.composition_spec.semantic_content_hash,
        session_epoch=1,
        activated_at=9.9,
        clock_domain_id="control.monotonic",
        input_barrier_at=9.9,
        required_roles=contracts.composition_spec.required_roles,
    )


def _bindings(bundle):
    schemas = bundle.input_action_schema_by_role
    return (
        DerivedCompositionInputRoleBinding(
            input_role="human_absolute",
            producer_id="human-absolute",
            producer_epoch=1,
            action_schema_hash=(schemas["human_absolute"].semantic_content_hash),
        ),
        DerivedCompositionInputRoleBinding(
            input_role="human_residual",
            producer_id="human-residual",
            producer_epoch=1,
            action_schema_hash=(schemas["human_residual"].semantic_content_hash),
        ),
    )


def _barrier():
    barrier = StationPublicationBarrier(source_slot_id="composer")
    result = admit_station_operation_result(
        StationOperationResultDraft(
            source_intent_id="intent:start-allocation-input",
            operation_kind="start_composition_session",
            subject_id="composer",
            succeeded=True,
            completed_at=9.9,
            completed_clock_domain_id="control.monotonic",
            status="composition_session_ready",
            evidence_refs=("virtual-allocation-input",),
        ),
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        available_at=9.9,
        available_clock_domain_id="control.monotonic",
        admission_sequence=0,
    )
    barrier.enable_from_composition_session_result(result)
    return barrier


def _enabled_fixture():
    bundle = _bundle(enabled=True)
    activation = _activation(bundle)
    clock = _Clock(9.9)
    coordinator = ControlAdmissionCoordinator(
        "robot-0",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="control.monotonic",
            runtime_session_id="session-0",
            boot_id="boot-0",
            reset_epoch=0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="session-0",
        coordinator=coordinator,
    )
    for producer_id in (
        "human-absolute",
        "human-residual",
        "human-anchor-lifecycle",
    ):
        epochs.activate(
            producer_id=producer_id,
            producer_epoch=1,
            produced_at=9.9,
            produced_clock_domain_id="human-device.monotonic",
            reason="test_startup",
        )
    store = CompositionInputStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
        retention_spec=TimeWindowRetentionSpec(
            spec_id="human-allocation-input-window",
            history_window_sec=1.0,
            max_future_lead_sec=1.0,
            max_effective_duration_sec=1.0,
        ),
        action_schema=_schema(),
        input_action_schema_by_role=bundle.input_action_schema_by_role,
    )
    lifecycle_store = ResidualAnchorLifecycleStore(
        runtime_session_id="session-0",
        coordinator=coordinator,
        producer_epochs=epochs,
    )
    lifecycle_spec = bundle.composition_contracts.residual_anchor_lifecycle_spec
    barrier = _barrier()
    port = StationDerivedInputPublicationPort(
        profile=StationDerivedInputPublicationProfile(
            profile_id="virtual-allocation-input-publication",
            enabled=True,
        ),
        activation_scope=activation,
        publication_barrier=barrier,
        admit_batch=lambda batch: store.admit_batch(
            batch,
            activation_scope=activation,
        ),
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        lifecycle_producer_id="human-anchor-lifecycle",
        lifecycle_producer_epoch=1,
        lifecycle_scope_id="virtual-human-allocation",
        lifecycle_spec=lifecycle_spec,
        admit_lifecycle=lambda draft: lifecycle_store.admit(
            draft,
            spec=lifecycle_spec,
            activation_scope=activation,
        ),
    )
    allocation_runtime = AllocationPublicationRuntime(
        allocation_scope_id="virtual-human-allocation",
        allocation_spec=bundle.allocation_spec,
        activation_scope=activation,
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        control_clock_reset_epoch=0,
        context_channel=bundle.config.allocation_context_channel,
        producer_id="allocation-reducer",
        producer_epoch=1,
        initial_reducer_state=empty_composition_allocation_state(
            "allocation-reducer",
            spec=bundle.allocation_spec,
        ),
        clock=clock,
    )
    allocation_port = StationAllocationPublicationPort(
        profile=StationAllocationPublicationProfile(
            profile_id="virtual-allocation-publication",
            enabled=True,
        ),
        allocation_runtime=allocation_runtime,
        publication_barrier=barrier,
        admit_context=lambda _draft: None,
    )
    controller = MujocoPolicyHumanAllocationInputController(
        bundle=bundle,
        activation_scope=activation,
        role_bindings=_bindings(bundle),
        publication_port=port,
        allocation_publication_port=allocation_port,
    )
    return (
        bundle,
        activation,
        clock,
        coordinator,
        store,
        lifecycle_store,
        controller,
    )


def _process(controller, clock, *, index: int, at: float, x: float, hand):
    action = _action(x, hand)
    clock.value = at
    return controller.process_and_publish(
        action,
        source_action_id=f"human:{index}",
        source_action_hash=action.semantic_content_hash,
        source_sequence=index,
        source_available_at=at,
        measured_at=20.0 + index * 0.1,
        measured_clock_domain_id="human.source",
        observed_at=at,
        observed_clock_domain_id="control.monotonic",
        runtime_session_id="session-0",
        control_domain_id="robot-0",
        produced_at=20.0 + index * 0.1,
        produced_clock_domain_id="human.source",
        effective_from=at,
        effective_until=at + 0.2,
        effective_clock_domain_id="control.monotonic",
        observation_cut_id=f"human-observation:{index}",
    )


def test_default_off_controller_does_not_transform_or_publish() -> None:
    bundle = _bundle(enabled=False)
    activation = _activation(bundle)
    controller = MujocoPolicyHumanAllocationInputController(
        bundle=bundle,
        activation_scope=activation,
        role_bindings=_bindings(bundle),
    )

    cycle = _process(
        controller,
        _Clock(10.0),
        index=0,
        at=10.0,
        x=0.2,
        hand=(0.4, 0.6),
    )

    assert cycle.status == "disabled"
    assert bundle.input_contracts.input_adapter.anchor is None
    assert bundle.evidence_window.snapshot().retained_pair_count == 0


def test_anchor_sample_is_first_pair_and_feeds_causal_basis() -> None:
    (
        bundle,
        activation,
        clock,
        coordinator,
        store,
        _lifecycle_store,
        controller,
    ) = _enabled_fixture()
    controller.request_anchor(requested_at=9.9)
    anchor = _process(
        controller,
        clock,
        index=0,
        at=10.0,
        x=0.2,
        hand=(0.4, 0.6),
    )
    assert anchor.status == "batch_admitted"
    assert anchor.derivation.captured_anchor
    assert anchor.anchor_lifecycle_admission.accepted
    assert len(store) == 2
    assert bundle.evidence_window.snapshot().retained_pair_count == 1
    assert anchor.evidence_pair.source_sample_id == "human:0"

    first = _process(
        controller,
        clock,
        index=1,
        at=10.1,
        x=0.22,
        hand=(0.45, 0.55),
    )
    second = _process(
        controller,
        clock,
        index=2,
        at=10.2,
        x=0.24,
        hand=(0.50, 0.50),
    )
    assert first.status == second.status == "batch_admitted"
    assert len(store) == 6
    assert first.publication.receipt.available_at == pytest.approx(10.1)
    assert second.evidence_pair.source_sample_id == "human:2"

    cut = coordinator.capture_cut(cutoff_at=10.2).cut
    basis = controller.capture_basis(
        allocation_scope_id="virtual-human-allocation",
        control_cut=cut,
        requested_effective_from=10.2,
    )
    assert basis is not None
    assert set(basis.activity_by_component) == {"arm", "hand"}
    assert basis.activity_by_component["arm"].activity_level == pytest.approx(2.0)
    assert basis.residual_magnitude_by_component[
        "arm"
    ].magnitude_level == pytest.approx(2.0)
    assert basis.residual_magnitude_by_component["arm"].source_fact_ids == (
        second.evidence_pair.pair_id,
    )
    assert basis.activation_scope.semantic_content_hash == (
        activation.semantic_content_hash
    )


def test_rebase_request_immediately_fences_evidence_window() -> None:
    (
        bundle,
        _activation_scope,
        clock,
        coordinator,
        _store,
        _lifecycle_store,
        controller,
    ) = _enabled_fixture()
    controller.request_anchor(requested_at=9.9)
    _process(
        controller,
        clock,
        index=0,
        at=10.0,
        x=0.2,
        hand=(0.4, 0.6),
    )
    _process(
        controller,
        clock,
        index=1,
        at=10.1,
        x=0.22,
        hand=(0.45, 0.55),
    )
    _process(
        controller,
        clock,
        index=2,
        at=10.2,
        x=0.24,
        hand=(0.50, 0.50),
    )
    assert bundle.evidence_window.snapshot().retained_pair_count == 3

    controller.request_anchor(requested_at=10.25)

    assert bundle.evidence_window.snapshot().retained_pair_count == 0
    assert (
        controller.capture_basis(
            allocation_scope_id="virtual-human-allocation",
            control_cut=coordinator.capture_cut(cutoff_at=10.25).cut,
            requested_effective_from=10.25,
        )
        is None
    )


def test_rejected_ready_retries_exact_capture_before_new_sample() -> None:
    (
        bundle,
        activation,
        clock,
        _coordinator,
        store,
        lifecycle_store,
        controller,
    ) = _enabled_fixture()
    controller.request_anchor(requested_at=9.9)
    publisher = controller.publication_port
    assert publisher is not None
    lifecycle_spec = bundle.composition_contracts.residual_anchor_lifecycle_spec
    calls = 0

    def reject_first_ready(draft):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ResidualAnchorLifecycleAdmissionReceipt(
                accepted=False,
                rejection_code="synthetic_transport_rejection",
                message="retry exact ready",
                lifecycle_record_id=("residual-anchor-lifecycle:synthetic-rejected"),
            )
        return lifecycle_store.admit(
            draft,
            spec=lifecycle_spec,
            activation_scope=activation,
        )

    publisher._admit_lifecycle = reject_first_ready
    rejected = _process(
        controller,
        clock,
        index=0,
        at=10.0,
        x=0.2,
        hand=(0.4, 0.6),
    )
    assert rejected.status == "anchor_lifecycle_rejected"
    assert len(store) == 2
    assert bundle.evidence_window.snapshot().retained_pair_count == 0

    retried = _process(
        controller,
        clock,
        index=1,
        at=10.1,
        x=9.0,
        hand=(0.9, 0.9),
    )
    assert retried.status == "batch_admitted"
    assert retried.batch is rejected.batch
    assert retried.derivation is rejected.derivation
    assert retried.evidence_pair.source_sample_id == "human:0"
    assert len(store) == 2
    assert len(lifecycle_store) == 2
    assert bundle.evidence_window.snapshot().retained_pair_count == 1
