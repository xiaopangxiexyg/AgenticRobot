from __future__ import annotations

import pytest

from control_composition.trajectory import (
    TrajectoryCorrectedPreviewSnapshot,
    TrajectoryPreviewFreshnessEvidence,
    TrajectoryPreviewFreshnessSpec,
    build_trajectory_intent_preview,
)
from control_station.composition.trajectory_corrected_preview import (
    TrajectoryCorrectedPreviewController,
    TrajectoryCorrectedPreviewProfile,
)
from control_station.composition.trajectory_intent_preview import (
    TrajectoryIntentPreviewController,
    TrajectoryIntentPreviewProfile,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ComponentActionTrajectory,
    ComponentActionTrajectoryKeyframe,
    ControlComponentSpecV2,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _trajectory() -> ComponentActionTrajectory:
    payload = ActionPayloadSchema(
        schema_id="station-preview-tool",
        schema_version="development",
        representation_family="cartesian_pose",
        dtype="float64",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="absolute_target",
        reference_frame="world",
        controlled_frame="tool",
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="station-preview",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="tool",
                controlled_entity_id="tool",
                command_channel_id="tool",
                selection_group_id="tool",
                resource_claims=("resource:tool",),
                accepted_payload_schema_hashes=(payload.semantic_content_hash,),
            ),
        ),
    )
    schema = ComponentActionSchema(
        schema_id="station-preview",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component={"tool": payload},
    )
    keyframes = tuple(
        ComponentActionTrajectoryKeyframe(
            target_effective_at=5.0 + 0.1 * index,
            effective_until=5.1 + 0.1 * index,
            action=ComponentAction(
                action_schema=schema,
                values_by_component={
                    "tool": (
                        0.1 * index,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    )
                },
            ),
            source_index=index,
            source_fact_ids=(f"source:{index}",),
            source_fact_hashes=(f"source-hash:{index}",),
        )
        for index in range(2)
    )
    return ComponentActionTrajectory(
        trajectory_id="station-preview",
        revision=1,
        source_kind="generic_source",
        source_instance_id="generic-source:1",
        clock_domain_id="control.monotonic",
        created_at=4.9,
        keyframes=keyframes,
        source_fact_ids=("source-plan:1",),
        source_fact_hashes=("source-plan-hash:1",),
        source_immutable_through=5.0,
    )


def test_default_off_preview_does_not_call_source_supplier() -> None:
    calls = 0

    def source():
        nonlocal calls
        calls += 1
        return _trajectory()

    controller = TrajectoryIntentPreviewController(
        profile=TrajectoryIntentPreviewProfile(),
        clock=lambda: 4.95,
        clock_domain_id="station.monotonic",
    )

    cycle = controller.run_preview_cycle(source)

    assert cycle.status == "disabled"
    assert cycle.preview is None
    assert calls == 0


def test_enabled_preview_projects_generic_carrier_without_execution_claim() -> None:
    controller = TrajectoryIntentPreviewController(
        profile=TrajectoryIntentPreviewProfile(
            profile_id="generic-future-visualization",
            enabled=True,
        ),
        clock=lambda: 4.95,
        clock_domain_id="station.monotonic",
    )

    cycle = controller.run_preview_cycle(_trajectory)

    assert cycle.status == "projected"
    assert cycle.preview is not None
    assert len(cycle.preview.producer_locked_points) == 1
    assert len(cycle.preview.speculative_points) == 1
    assert cycle.preview.authoritative is False


def test_preview_profile_rejects_send_actions() -> None:
    with pytest.raises(ValueError, match="cannot send actions"):
        TrajectoryIntentPreviewProfile(
            profile_id="invalid",
            enabled=True,
            send_actions=True,
        )


def _freshness_spec() -> TrajectoryPreviewFreshnessSpec:
    return TrajectoryPreviewFreshnessSpec(
        spec_id="station-trajectory-display",
        fresh_max_age_sec=0.2,
        aging_max_age_sec=0.5,
        stale_displayable_max_age_sec=1.0,
    )


def _freshness_evidence(
    trajectory: ComponentActionTrajectory,
) -> TrajectoryPreviewFreshnessEvidence:
    return TrajectoryPreviewFreshnessEvidence(
        trajectory_id=trajectory.trajectory_id,
        trajectory_hash=trajectory.semantic_content_hash,
        observation_fact_id="observation:1",
        observation_fact_hash="observation-hash:1",
        observed_at=4.8,
        observed_clock_domain_id="station.monotonic",
        available_at=4.9,
        available_clock_domain_id="station.monotonic",
        valid_until=5.5,
    )


def test_intent_preview_fetches_freshness_only_when_profile_requests_it() -> None:
    trajectory = _trajectory()
    evidence_calls = 0
    clock_calls = 0

    def evidence_factory():
        nonlocal evidence_calls
        evidence_calls += 1
        return _freshness_evidence(trajectory)

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        return 4.95

    controller = TrajectoryIntentPreviewController(
        profile=TrajectoryIntentPreviewProfile(
            profile_id="intent-with-freshness",
            enabled=True,
            freshness_spec=_freshness_spec(),
        ),
        clock=clock,
        clock_domain_id="station.monotonic",
    )

    cycle = controller.run_preview_cycle(
        lambda: trajectory,
        freshness_evidence_factory=evidence_factory,
    )

    assert cycle.freshness is not None
    assert cycle.freshness.freshness_class == "fresh"
    assert evidence_calls == 1
    assert clock_calls == 1


def test_intent_preview_does_not_fetch_unrequested_freshness() -> None:
    evidence_calls = 0

    def evidence_factory():
        nonlocal evidence_calls
        evidence_calls += 1
        return None

    controller = TrajectoryIntentPreviewController(
        profile=TrajectoryIntentPreviewProfile(
            profile_id="intent-without-freshness",
            enabled=True,
        ),
        clock=lambda: 4.95,
        clock_domain_id="station.monotonic",
    )

    cycle = controller.run_preview_cycle(
        _trajectory,
        freshness_evidence_factory=evidence_factory,
    )

    assert cycle.freshness is None
    assert evidence_calls == 0


def _typed_corrected_preview_shell() -> TrajectoryCorrectedPreviewSnapshot:
    """Port fixture; corrected numerical construction is tested in its owner."""

    intent = build_trajectory_intent_preview(
        _trajectory(),
        generated_at=4.95,
        generated_clock_domain_id="station.monotonic",
    )
    preview = object.__new__(TrajectoryCorrectedPreviewSnapshot)
    object.__setattr__(preview, "intent_preview", intent)
    return preview


def test_corrected_preview_port_is_default_off_and_lazy() -> None:
    preview_calls = 0
    evidence_calls = 0

    def preview_factory():
        nonlocal preview_calls
        preview_calls += 1
        return _typed_corrected_preview_shell()

    def evidence_factory():
        nonlocal evidence_calls
        evidence_calls += 1
        return None

    controller = TrajectoryCorrectedPreviewController(
        profile=TrajectoryCorrectedPreviewProfile(
            freshness_spec=_freshness_spec(),
        ),
        clock=lambda: 4.95,
        clock_domain_id="station.monotonic",
    )

    cycle = controller.run_preview_cycle(
        preview_factory,
        freshness_evidence_factory=evidence_factory,
    )

    assert cycle.status == "disabled"
    assert cycle.preview is None
    assert preview_calls == 0
    assert evidence_calls == 0


def test_corrected_preview_port_accepts_backend_neutral_snapshot() -> None:
    preview = _typed_corrected_preview_shell()
    trajectory = preview.intent_preview.trajectory
    controller = TrajectoryCorrectedPreviewController(
        profile=TrajectoryCorrectedPreviewProfile(
            profile_id="conditional-corrected-view",
            enabled=True,
            freshness_spec=_freshness_spec(),
        ),
        clock=lambda: 4.95,
        clock_domain_id="station.monotonic",
    )

    cycle = controller.run_preview_cycle(
        lambda: preview,
        freshness_evidence_factory=lambda: _freshness_evidence(trajectory),
    )

    assert cycle.status == "projected"
    assert cycle.preview is preview
    assert cycle.freshness is not None
    assert cycle.freshness.freshness_class == "fresh"
