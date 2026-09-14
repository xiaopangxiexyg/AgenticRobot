from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control_composition import build_composition_activation_scope
from control_composition.composition import (
    build_relative_baseline_residual_contracts,
)
from control_composition.retention import TimeWindowRetentionSpec
from control_composition.runtime import ControlCompositionRuntime
from control_station.composition.execution_measured_pose_baseline import (
    ExecutionMeasuredPoseBaselineController,
    ExecutionMeasuredPoseBaselineProfile,
)
from control_station.composition.rebot_b601_rm65_motion_input import (
    B601_RM65_RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND,
    B601RM65MotionInputProfile,
    adapt_b601_rm65_residual_to_composition_input,
)
from control_station.composition.rebot_b601_rm65_residual import (
    B601RM65ResidualSource,
)
from control_station.sources.rebot_b601_pose import B601ReadOnlyPoseSource
from hardware.rebot_b601_rs import (
    B601_RS_FEEDBACK_FRAME,
    B601HardwareState,
    B601JointState,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentActionSchema,
    ControlComponentSpecV2,
    ExecutionMeasurementObservation,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry
from teleop_runtime.action.human_residual import HumanResidualPoseAdapter

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _Backend:
    def __init__(self, states: list[B601HardwareState]) -> None:
        self.states = list(states)

    def poll_feedback(
        self,
        *,
        request_feedback: bool,
        require_host_freshness: bool,
    ) -> B601HardwareState:
        assert not request_feedback
        assert require_host_freshness
        return self.states.pop(0)


class _Model:
    arm_joint_names = tuple(f"joint{index}" for index in range(1, 7))
    end_effector_frame = "gripper_end"

    def forward_kinematics(self, qpos: object) -> np.ndarray:
        values = np.asarray(qpos, dtype=np.float64)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Rotation.from_rotvec(
            (0.0, 0.0, float(values[1]))
        ).as_matrix()
        pose[:3, 3] = values[:3]
        return pose


def _schemas(
    *,
    reference_frame: str = "rm65_base",
) -> tuple[ComponentActionSchema, ComponentActionSchema]:
    common = {
        "schema_version": "development",
        "representation_family": "cartesian_pose",
        "dtype": "float64",
        "shape": (7,),
        "coordinate_names": ("x", "y", "z", "qx", "qy", "qz", "qw"),
        "units": ("m", "m", "m", "1", "1", "1", "1"),
        "reference_frame": reference_frame,
        "controlled_frame": "rm65_tool",
    }
    absolute = ActionPayloadSchema(
        schema_id="rm65-pose-absolute",
        value_semantics="absolute_target",
        **common,
    )
    residual = ActionPayloadSchema(
        schema_id="rm65-pose-residual",
        value_semantics="relative_delta",
        semantic_parameters={
            "translation_application": "reference_frame_additive",
            "rotation_application": "baseline_local_right_multiply",
        },
        **common,
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="b601-rm65-motion-shadow",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="right_arm",
                controlled_entity_id="rm65-right",
                command_channel_id="rm65-right-arm",
                selection_group_id="rm65-right-arm",
                resource_claims=("resource:rm65-right-arm",),
                accepted_payload_schema_hashes=(
                    absolute.semantic_content_hash,
                    residual.semantic_content_hash,
                ),
            ),
        ),
    )
    return (
        ComponentActionSchema(
            schema_id="rm65-pose-absolute",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component={"right_arm": absolute},
        ),
        ComponentActionSchema(
            schema_id="rm65-pose-residual",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component={"right_arm": residual},
        ),
    )


def _state(
    sequence: int,
    qpos: tuple[float, ...],
    *,
    valid: bool = True,
) -> B601HardwareState:
    observed_at = 9.0 + sequence * 0.01
    receive_at = observed_at + 0.001
    joints = tuple(
        B601JointState(
            name=f"joint{index + 1}",
            motor_id=index + 1,
            feedback_id=0xFD,
            arbitration_id=index + 1,
            status_code=0,
            position_rad=qpos[index],
            velocity_reported=0.0,
            torque_reported_nm=0.0,
            observed_at_sec=observed_at,
            sample_sequence=sequence,
            valid=valid,
            stale=not valid,
            frame_id=B601_RS_FEEDBACK_FRAME,
            position_source="synthetic_host_fresh_cache",
            status_sequence=100 * sequence + index,
            receive_age_sec=receive_at - observed_at,
            host_receive_freshness_verified=True,
        )
        for index in range(6)
    )
    return B601HardwareState(
        sequence=sequence,
        receive_timestamp_sec=receive_at,
        joints=joints,
        valid=valid,
        stale=not valid,
        freshness_verified=False,
        freshness_semantics="host_receive_sequence_and_age_verified",
        fault_reason=None if valid else "synthetic rejected frame",
        host_receive_freshness_verified=True,
    )


def _residual_source(
    states: list[B601HardwareState],
    residual_schema: ComponentActionSchema,
) -> B601RM65ResidualSource:
    graph = CoordinateFrameGraph(
        graph_id="b601-rm65-motion-shadow",
        frames=tuple(
            CoordinateFrameSpec(frame_id)
            for frame_id in ("base_link", "gripper_end", "rm65_base", "rm65_tool")
        ),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="b601-base-to-rm65-base",
                source_frame_id="base_link",
                target_frame_id="rm65_base",
                matrix_target_source=np.eye(4, dtype=np.float64),
            ),
            RigidFrameTransformSpec(
                transform_id="rm65-tool-to-b601-gripper",
                source_frame_id="rm65_tool",
                target_frame_id="gripper_end",
                matrix_target_source=np.eye(4, dtype=np.float64),
            ),
        ),
    )
    return B601RM65ResidualSource(
        B601ReadOnlyPoseSource(_Backend(states), _Model()),
        HumanResidualPoseAdapter(
            output_action_schema=residual_schema,
            frame_graph=graph,
        ),
    )


def _activation():
    return build_composition_activation_scope(
        operation_spec_id="b601-rm65-motion-shadow",
        operation_spec_hash="b601-rm65-motion-shadow-hash",
        session_epoch=1,
        activated_at=10.0,
        input_barrier_at=10.0,
        clock_domain_id="control.monotonic",
        required_roles=("target_baseline", "human_residual"),
    )


def _residual_step(
    residual_schema: ComponentActionSchema,
    *,
    valid_current: bool = True,
    adapted_clock: str = "control.monotonic",
):
    source = _residual_source(
        [
            _state(1, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            _state(
                2,
                (0.02, 0.10, -0.03, 0.0, 0.0, 0.0),
                valid=valid_current,
            ),
        ],
        residual_schema,
    )
    anchor = source.capture_anchor(
        source.poll_pose(),
        anchor_generation=0,
        captured_at=10.01,
        captured_clock_domain_id="control.monotonic",
    )
    return source.poll_and_adapt(
        anchor=anchor,
        adapted_at=10.03,
        adapted_clock_domain_id=adapted_clock,
    )


def _profile(residual_schema: ComponentActionSchema, *, enabled: bool = True):
    return B601RM65MotionInputProfile(
        profile_id="b601-rm65-motion-shadow",
        input_role="human_residual",
        runtime_session_id="session",
        control_domain_id="rm65-sim",
        producer_id="b601-residual",
        producer_epoch=1,
        action_schema=residual_schema,
        maximum_adaptation_age_sec=0.1,
        enabled=enabled,
    )


def _runtime(
    absolute_schema: ComponentActionSchema,
    residual_schema: ComponentActionSchema,
):
    clock = _Clock(10.04)
    coordinator = ControlAdmissionCoordinator(
        "rm65-sim",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="control.monotonic",
            runtime_session_id="session",
            boot_id="boot",
            reset_epoch=0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="session",
        coordinator=coordinator,
    )
    for producer in ("rm65-measured-baseline", "b601-residual"):
        epochs.activate(
            producer_id=producer,
            producer_epoch=1,
            produced_at=9.9,
            produced_clock_domain_id="control.monotonic",
            reason="shadow_start",
        )
    contracts = build_relative_baseline_residual_contracts(
        contract_id="b601-rm65-motion-shadow",
        output_action_schema=absolute_schema,
        residual_action_schema=residual_schema,
        baseline_role="target_baseline",
        residual_role="human_residual",
    )
    retention = TimeWindowRetentionSpec(
        spec_id="b601-rm65-motion-shadow-window",
        history_window_sec=0.2,
        max_future_lead_sec=1.0,
        max_effective_duration_sec=1.0,
    )
    runtime = ControlCompositionRuntime(
        runtime_session_id="session",
        control_domain_id="rm65-sim",
        embodiment_instance_id="rm65-mujoco-shadow",
        coordinator=coordinator,
        producer_epochs=epochs,
        action_schema=absolute_schema,
        input_action_schema_by_role={
            "target_baseline": absolute_schema,
            "human_residual": residual_schema,
        },
        event_emitter=lambda _events: None,
        proposal_admitter=lambda *_args, **_kwargs: pytest.fail(
            "shadow evaluation must not publish a proposal"
        ),
        context_retention_spec=retention,
        composition_input_retention_spec=retention,
        composition_strategy=contracts.strategy,
        clock=clock,
    )
    return runtime, contracts


def test_module_does_not_import_target_backend_or_own_dispatch() -> None:
    source = (
        Path(__file__).parents[1]
        / "control_station"
        / "composition"
        / "rebot_b601_rm65_motion_input.py"
    ).read_text(encoding="utf-8")

    assert "from hardware" not in source
    assert "import hardware" not in source
    assert "from sim" not in source
    assert "import sim" not in source


def test_profile_is_default_off() -> None:
    _, residual_schema = _schemas()

    with pytest.raises(RuntimeError, match="disabled"):
        adapt_b601_rm65_residual_to_composition_input(
            _residual_step(residual_schema),
            profile=_profile(residual_schema, enabled=False),
            activation_scope=_activation(),
            sequence=0,
            effective_from=10.04,
            effective_until=10.5,
        )


def test_b601_residual_and_measured_rm65_baseline_compose_in_shadow() -> None:
    absolute_schema, residual_schema = _schemas()
    activation = _activation()
    runtime, contracts = _runtime(absolute_schema, residual_schema)
    baseline = ExecutionMeasuredPoseBaselineController(
        ExecutionMeasuredPoseBaselineProfile(
            profile_id="rm65-measured-baseline",
            input_role="target_baseline",
            runtime_session_id="session",
            control_domain_id="rm65-sim",
            producer_id="rm65-measured-baseline",
            producer_epoch=1,
            action_schema=absolute_schema,
            component_ids=("right_arm",),
            observation_binding_id="rm65-pose-observation",
            observation_plan_hash="rm65-pose-plan",
            maximum_measurement_age_sec=0.1,
            enabled=True,
        )
    )
    baseline.activate(activation)
    payload = absolute_schema.payload_schema_by_component["right_arm"]
    baseline_capture = baseline.capture(
        ExecutionMeasurementObservation(
            measurement_id="rm65-measured-pose:1",
            source_measurement_hash="rm65-qpos:1",
            control_domain_id="rm65-sim",
            measured_at=10.01,
            available_at=10.02,
            clock_domain_id="control.monotonic",
            value_by_component={
                "right_arm": (0.4, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0)
            },
            representation_by_component={"right_arm": "cartesian_pose"},
            schema_hash_by_component={
                "right_arm": payload.semantic_content_hash
            },
            observation_binding_id="rm65-pose-observation",
            observation_plan_hash="rm65-pose-plan",
        ),
        anchor_generation=0,
        captured_at=10.03,
        effective_until=10.5,
    )
    residual = adapt_b601_rm65_residual_to_composition_input(
        _residual_step(residual_schema),
        profile=_profile(residual_schema),
        activation_scope=activation,
        sequence=0,
        effective_from=10.04,
        effective_until=10.5,
    )

    assert residual.input_draft is not None
    assert runtime.admit_composition_input(
        baseline_capture.input_draft,
        activation_scope=activation,
    ).accepted
    assert runtime.admit_composition_input(
        residual.input_draft,
        activation_scope=activation,
    ).accepted
    submission = runtime.evaluate_composition(
        spec=contracts.composition_spec,
        target_effective_at=10.05,
        activation_scope=activation,
    )

    assert submission.proposal_admission is None
    assert submission.composition_evaluation.output_status == "ready"
    composed = submission.composition_evaluation.composed_candidate
    assert composed is not None
    value = np.asarray(composed.value("right_arm").values)
    np.testing.assert_allclose(value[:3], (0.42, 0.0, 0.27), atol=1e-9)
    np.testing.assert_allclose(
        Rotation.from_quat(value[3:]).as_rotvec(),
        (0.0, 0.0, 0.10),
        atol=1e-9,
    )
    kind = B601_RM65_RESIDUAL_ANCHOR_SEMANTIC_BINDING_KIND
    assert kind in residual.input_draft.semantic_binding_id_by_kind
    assert residual.input_draft.observation_cut_id == (
        residual.residual_step.projection.source_observation_sample_id
    )
    assert residual.residual_step.projection.pose_sample is not None
    assert residual.input_draft.source_fact_ids[0] == (
        residual.residual_step.projection.pose_sample.sample_id
    )
    assert residual.input_draft.source_fact_hashes[0] == (
        residual.residual_step.projection.pose_sample.semantic_content_hash
    )
    assert not residual.command_dispatched
    assert not residual.authoritative


def test_rejected_current_sample_withdraws_input_without_fallback() -> None:
    _, residual_schema = _schemas()
    previous = adapt_b601_rm65_residual_to_composition_input(
        _residual_step(residual_schema),
        profile=_profile(residual_schema),
        activation_scope=_activation(),
        sequence=0,
        effective_from=10.04,
        effective_until=10.5,
    )
    rejected = adapt_b601_rm65_residual_to_composition_input(
        _residual_step(residual_schema, valid_current=False),
        profile=_profile(residual_schema),
        activation_scope=_activation(),
        sequence=1,
        effective_from=10.04,
        effective_until=10.5,
    )

    assert previous.input_draft is not None
    assert rejected.input_draft is None
    assert "current_residual_unavailable" in rejected.rejection_reason
    assert "synthetic rejected frame" in rejected.rejection_reason


@pytest.mark.parametrize(
    ("adapted_clock", "effective_from", "reason"),
    (
        ("another.clock", 10.04, "adaptation_clock_mismatch"),
        ("control.monotonic", 10.20, "adaptation_is_stale"),
    ),
)
def test_wrong_clock_or_stale_adaptation_is_not_admitted(
    adapted_clock: str,
    effective_from: float,
    reason: str,
) -> None:
    _, residual_schema = _schemas()
    result = adapt_b601_rm65_residual_to_composition_input(
        _residual_step(residual_schema, adapted_clock=adapted_clock),
        profile=_profile(residual_schema),
        activation_scope=_activation(),
        sequence=0,
        effective_from=effective_from,
        effective_until=10.5,
    )

    assert result.input_draft is None
    assert result.rejection_reason == reason


def test_wrong_target_frame_schema_is_not_admitted() -> None:
    _, residual_schema = _schemas()
    _, wrong_frame_schema = _schemas(reference_frame="another_rm65_base")
    result = adapt_b601_rm65_residual_to_composition_input(
        _residual_step(residual_schema),
        profile=_profile(wrong_frame_schema),
        activation_scope=_activation(),
        sequence=0,
        effective_from=10.04,
        effective_until=10.5,
    )

    assert result.input_draft is None
    assert result.rejection_reason == "residual_action_schema_mismatch"
