from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from control_composition import build_composition_activation_scope
from control_station.composition.execution_measured_pose_baseline import (
    EXECUTION_MEASURED_POSE_BASELINE_SEMANTIC_BINDING_KIND,
    ExecutionMeasuredPoseBaselineController,
    ExecutionMeasuredPoseBaselineProfile,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentActionSchema,
    ControlComponentSpecV2,
    ExecutionMeasurementObservation,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _schema() -> ComponentActionSchema:
    payload = ActionPayloadSchema(
        schema_id="rm65-right-tool-pose",
        schema_version="development",
        representation_family="cartesian_pose",
        dtype="float64",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="absolute_target",
        reference_frame="right_arm_base",
        controlled_frame="tool_tcp",
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="rm65-measured-baseline",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="right_arm",
                controlled_entity_id="rm65-right",
                command_channel_id="rm65-right-arm",
                selection_group_id="rm65-right-arm",
                resource_claims=("resource:rm65-right-arm",),
                accepted_payload_schema_hashes=(payload.semantic_content_hash,),
            ),
        ),
    )
    return ComponentActionSchema(
        schema_id="rm65-measured-baseline",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component={"right_arm": payload},
    )


def _profile(*, enabled: bool = True) -> ExecutionMeasuredPoseBaselineProfile:
    return ExecutionMeasuredPoseBaselineProfile(
        profile_id="rm65-right-measured-baseline-v1",
        input_role="target_baseline",
        runtime_session_id="session-1",
        control_domain_id="rm65-sim",
        producer_id="rm65-measured-baseline",
        producer_epoch=3,
        action_schema=_schema(),
        component_ids=("right_arm",),
        observation_binding_id="rm65-pose-observation-binding",
        observation_plan_hash="rm65-pose-observation-plan-hash",
        maximum_measurement_age_sec=0.1,
        enabled=enabled,
    )


def _activation(*, epoch: int = 1, at: float = 10.0):
    return build_composition_activation_scope(
        operation_spec_id="b601-rm65-motion-only",
        operation_spec_hash="b601-rm65-motion-only-hash",
        session_epoch=epoch,
        activated_at=at,
        input_barrier_at=at,
        clock_domain_id="control.monotonic",
        required_roles=("target_baseline", "human_residual"),
    )


def _observation(
    *,
    suffix: str = "0",
    measured_at: float = 10.01,
    available_at: float = 10.02,
) -> ExecutionMeasurementObservation:
    payload = _schema().payload_schema_by_component["right_arm"]
    return ExecutionMeasurementObservation(
        measurement_id=f"rm65-pose-measurement:{suffix}",
        source_measurement_hash=f"rm65-qpos-measurement-hash:{suffix}",
        control_domain_id="rm65-sim",
        measured_at=measured_at,
        available_at=available_at,
        clock_domain_id="control.monotonic",
        value_by_component={"right_arm": (0.4, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0)},
        representation_by_component={"right_arm": "cartesian_pose"},
        schema_hash_by_component={"right_arm": payload.semantic_content_hash},
        observation_binding_id="rm65-pose-observation-binding",
        observation_plan_hash="rm65-pose-observation-plan-hash",
    )


def test_module_does_not_import_hardware_or_sim_backends() -> None:
    source = (
        Path(__file__).parents[1]
        / "control_station"
        / "composition"
        / "execution_measured_pose_baseline.py"
    ).read_text(encoding="utf-8")

    assert "from hardware" not in source
    assert "import hardware" not in source
    assert "from sim" not in source
    assert "import sim" not in source


def test_profile_is_default_off_for_capture() -> None:
    controller = ExecutionMeasuredPoseBaselineController(_profile(enabled=False))

    with pytest.raises(RuntimeError, match="disabled"):
        controller.activate(_activation())

    assert controller.current is None


def test_fresh_measurement_becomes_non_authoritative_baseline_input() -> None:
    profile = _profile()
    controller = ExecutionMeasuredPoseBaselineController(profile)
    activation = controller.activate(_activation())
    observation = _observation()

    result = controller.capture(
        observation,
        anchor_generation=0,
        captured_at=10.03,
        effective_until=10.5,
        context_ids=("context:motion-only",),
    )

    assert controller.current is result
    assert result.anchor.anchor_generation == 0
    assert result.anchor.source_measurement_hash == observation.semantic_content_hash
    assert result.anchor.composition_activation_hash == (
        activation.semantic_content_hash
    )
    assert result.anchor.action.component_coverage == ("right_arm",)
    assert tuple(result.anchor.action.value("right_arm").values) == (
        0.4,
        -0.1,
        0.3,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    draft = result.input_draft
    assert draft.input_role == "target_baseline"
    assert draft.producer_epoch == 3
    assert draft.sequence == 0
    assert draft.produced_at == observation.available_at
    assert draft.effective_from == 10.03
    assert draft.effective_until == 10.5
    assert draft.effective_clock_domain_id == activation.clock_domain_id
    assert draft.observation_cut_id == observation.measurement_id
    assert draft.source_fact_ids == (observation.measurement_id,)
    assert draft.source_fact_hashes == (observation.semantic_content_hash,)
    assert not draft.authority_candidate
    kind = EXECUTION_MEASURED_POSE_BASELINE_SEMANTIC_BINDING_KIND
    assert draft.semantic_binding_id_by_kind[kind] == result.anchor.anchor_id
    assert draft.semantic_binding_hash_by_kind[kind] == (
        result.anchor.semantic_content_hash
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("control_domain_id", "other-domain", "control domain"),
        ("clock_domain_id", "other-clock", "clock domains"),
        ("observation_binding_id", "other-binding", "observation binding"),
        (
            "observation_plan_hash",
            "other-plan",
            "observation binding",
        ),
    ),
)
def test_wrong_measurement_contract_is_rejected_without_state_change(
    field: str,
    value: str,
    message: str,
) -> None:
    controller = ExecutionMeasuredPoseBaselineController(_profile())
    controller.activate(_activation())
    observation = replace(_observation(), **{field: value})

    with pytest.raises(ValueError, match=message):
        controller.capture(
            observation,
            anchor_generation=0,
            captured_at=10.03,
            effective_until=10.5,
        )

    assert controller.current is None


def test_schema_or_frame_identity_mismatch_is_rejected() -> None:
    controller = ExecutionMeasuredPoseBaselineController(_profile())
    controller.activate(_activation())
    observation = replace(
        _observation(),
        schema_hash_by_component={"right_arm": "another-pose-schema"},
    )

    with pytest.raises(ValueError, match="schema/frame binding"):
        controller.capture(
            observation,
            anchor_generation=0,
            captured_at=10.03,
            effective_until=10.5,
        )

    assert controller.current is None


def test_failed_rebase_keeps_last_complete_baseline() -> None:
    controller = ExecutionMeasuredPoseBaselineController(_profile())
    controller.activate(_activation())
    first = controller.capture(
        _observation(),
        anchor_generation=0,
        captured_at=10.03,
        effective_until=10.5,
    )
    stale = _observation(
        suffix="stale",
        measured_at=10.04,
        available_at=10.05,
    )

    with pytest.raises(ValueError, match="stale"):
        controller.capture(
            stale,
            anchor_generation=1,
            captured_at=10.3,
            effective_until=10.6,
        )

    assert controller.current is first

    second = controller.capture(
        _observation(
            suffix="rebase",
            measured_at=10.19,
            available_at=10.20,
        ),
        anchor_generation=1,
        captured_at=10.21,
        effective_until=10.7,
    )
    assert controller.current is second
    assert second.anchor.anchor_generation == 1
    assert second.input_draft.sequence == 1


def test_generation_and_activation_advance_only_explicitly() -> None:
    controller = ExecutionMeasuredPoseBaselineController(_profile())
    activation = _activation()
    controller.activate(activation)
    first = controller.capture(
        _observation(),
        anchor_generation=0,
        captured_at=10.03,
        effective_until=10.5,
    )

    assert controller.activate(activation) is activation
    assert controller.current is first
    with pytest.raises(ValueError, match="advance exactly once"):
        controller.capture(
            _observation(suffix="duplicate"),
            anchor_generation=0,
            captured_at=10.04,
            effective_until=10.5,
        )

    next_activation = _activation(epoch=2, at=11.0)
    controller.activate(next_activation)
    assert controller.current is None
    restarted = controller.capture(
        _observation(
            suffix="next-activation",
            measured_at=11.01,
            available_at=11.02,
        ),
        anchor_generation=0,
        captured_at=11.03,
        effective_until=11.5,
    )
    assert restarted.anchor.anchor_generation == 0
    assert restarted.input_draft.sequence == 1


def test_pre_activation_or_old_measurement_is_rejected() -> None:
    controller = ExecutionMeasuredPoseBaselineController(_profile())
    controller.activate(_activation())

    with pytest.raises(ValueError, match="activation barrier"):
        controller.capture(
            _observation(measured_at=9.99, available_at=10.01),
            anchor_generation=0,
            captured_at=10.02,
            effective_until=10.5,
        )
    with pytest.raises(ValueError, match="stale"):
        controller.capture(
            _observation(measured_at=10.01, available_at=10.02),
            anchor_generation=0,
            captured_at=10.2,
            effective_until=10.5,
        )

    assert controller.current is None
