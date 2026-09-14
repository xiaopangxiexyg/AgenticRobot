from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from control_composition import build_composition_activation_scope
from control_composition.composition import (
    build_relative_baseline_residual_contracts,
)
from control_station.composition.execution_measured_pose_baseline import (
    ExecutionMeasuredPoseBaselineProfile,
)
from control_station.composition.rebot_b601_rm65_motion_input import (
    B601RM65MotionInputProfile,
)
from control_station.composition.rebot_b601_rm65_mujoco_shadow import (
    B601RM65MujocoShadowController,
    B601RM65MujocoShadowProfile,
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
    ActionComponentCatalog,
    ActionComponentCatalogV2,
    ActionComponentSpec,
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
from sim.contracts import TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.deploy import (
    MujocoCompositionRuntimeConfig,
    MujocoDeployConfig,
    MujocoDeployRuntime,
)
from sim.mujoco.deploy.runtime import MujocoComponentExecutionBinding
from teleop_runtime.action.human_residual import HumanResidualPoseAdapter

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _Clock:
    def __init__(self, value: float = 10.04) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _Scene:
    arm_specs: ClassVar = {"left": object(), "right": object()}
    gripper_specs: ClassVar = {}

    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }
        self.physics_commands: list[object] = []

    def get_arm_qpos(self, side: str) -> np.ndarray:
        return self.qpos[side].copy()

    def step_kinematic_direct_qpos_debug(
        self,
        qpos_by_side,
        *,
        gripper_close_by_name,
        zero_controlled_qvel,
    ) -> None:
        del qpos_by_side, gripper_close_by_name, zero_controlled_qvel
        self.physics_commands.append(object())


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
    reference_frame: str = "right_arm_base",
) -> tuple[ComponentActionSchema, ComponentActionSchema]:
    common = {
        "schema_version": "development",
        "representation_family": "cartesian_pose",
        "dtype": "float64",
        "shape": (7,),
        "coordinate_names": ("x", "y", "z", "qx", "qy", "qz", "qw"),
        "units": ("m", "m", "m", "1", "1", "1", "1"),
        "reference_frame": reference_frame,
        "controlled_frame": "tool_tcp",
    }
    absolute = ActionPayloadSchema(
        schema_id="rm65-right-pose-absolute",
        value_semantics="absolute_target",
        **common,
    )
    residual = ActionPayloadSchema(
        schema_id="rm65-right-pose-residual",
        value_semantics="relative_delta",
        semantic_parameters={
            "translation_application": "reference_frame_additive",
            "rotation_application": "baseline_local_right_multiply",
        },
        **common,
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="b601-rm65-mujoco-shadow",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id="right_arm",
                controlled_entity_id="rm65-right",
                command_channel_id="rm65-right-arm",
                selection_group_id="arms",
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
            schema_id="rm65-right-pose-absolute",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component={"right_arm": absolute},
        ),
        ComponentActionSchema(
            schema_id="rm65-right-pose-residual",
            schema_version="development",
            component_catalog=catalog,
            payload_schema_by_component={"right_arm": residual},
        ),
    )


def _state(sequence: int, qpos: tuple[float, ...], *, valid: bool = True):
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


def _residual_step(
    residual_schema: ComponentActionSchema,
    *,
    valid_current: bool = True,
):
    graph = CoordinateFrameGraph(
        graph_id="b601-rm65-mujoco-shadow",
        frames=tuple(
            CoordinateFrameSpec(frame_id)
            for frame_id in (
                "base_link",
                "gripper_end",
                "right_arm_base",
                "tool_tcp",
            )
        ),
        transforms=(
            RigidFrameTransformSpec(
                transform_id="b601-base-to-rm65-base",
                source_frame_id="base_link",
                target_frame_id="right_arm_base",
                matrix_target_source=np.eye(4, dtype=np.float64),
            ),
            RigidFrameTransformSpec(
                transform_id="rm65-tool-to-b601-gripper",
                source_frame_id="tool_tcp",
                target_frame_id="gripper_end",
                matrix_target_source=np.eye(4, dtype=np.float64),
            ),
        ),
    )
    source = B601RM65ResidualSource(
        B601ReadOnlyPoseSource(
            _Backend(
                [
                    _state(1, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                    _state(
                        2,
                        (0.02, 0.10, -0.03, 0.0, 0.0, 0.0),
                        valid=valid_current,
                    ),
                ]
            ),
            _Model(),
        ),
        HumanResidualPoseAdapter(
            output_action_schema=residual_schema,
            frame_graph=graph,
        ),
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
        adapted_clock_domain_id="control.monotonic",
    )


def _activation(*, epoch: int = 1, at: float = 10.0):
    return build_composition_activation_scope(
        operation_spec_id="b601-rm65-mujoco-shadow",
        operation_spec_hash="b601-rm65-mujoco-shadow-operation-hash",
        session_epoch=epoch,
        activated_at=at,
        input_barrier_at=at,
        clock_domain_id="control.monotonic",
        required_roles=("target_baseline", "human_residual"),
    )


def _observation(
    absolute_schema: ComponentActionSchema,
    *,
    suffix: str = "1",
    measured_at: float = 10.01,
    available_at: float = 10.02,
):
    payload = absolute_schema.payload_schema_by_component["right_arm"]
    return ExecutionMeasurementObservation(
        measurement_id=f"rm65-measured-pose:{suffix}",
        source_measurement_hash=f"rm65-measured-qpos:{suffix}",
        control_domain_id="rm65-sim",
        measured_at=measured_at,
        available_at=available_at,
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
    )


def _build_runtime(*, activate_inputs: bool = True):
    absolute, residual = _schemas()
    contracts = build_relative_baseline_residual_contracts(
        contract_id="b601-rm65-mujoco-shadow",
        output_action_schema=absolute,
        residual_action_schema=residual,
        baseline_role="target_baseline",
        residual_role="human_residual",
    )
    scene = _Scene()
    clock = _Clock()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=TimingSpec(
                sample_hz=15.0,
                policy_hz=15.0,
                reference_hz=60.0,
                ik_hz=60.0,
                command_hz=120.0,
                physics_hz=1000.0,
            ),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    component_catalog = ActionComponentCatalog(
        catalog_id="b601-rm65-mujoco-shadow-execution",
        action_space=absolute.schema_id,
        components=(
            ActionComponentSpec(
                component_id="right_arm",
                selection_group_id="arms",
                effector_name="right",
            ),
        ),
    )
    binding = MujocoComponentExecutionBinding(
        binding_id="b601-rm65-mujoco-shadow-binding",
        action_schema=absolute,
        execution_catalog=component_catalog,
        handler_contract_id="shadow-never-materialized",
        handler_contract_hash="shadow-never-materialized-hash",
        materializer=lambda *_args, **_kwargs: pytest.fail(
            "shadow must not materialize an execution action"
        ),
        output_effector_by_component={"right_arm": "right"},
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="session",
            control_domain_id="rm65-sim",
            embodiment_instance_id="rm65-mujoco-shadow",
            clock_domain_id="control.monotonic",
        ),
        composition_runtime_config=MujocoCompositionRuntimeConfig(),
        composition_strategy=contracts.strategy,
        composition_input_action_schema_by_role={
            "target_baseline": absolute,
            "human_residual": residual,
        },
        component_execution_binding=binding,
        clock=clock,
    )
    runtime.start()
    if activate_inputs:
        for producer in ("rm65-measured-baseline", "b601-residual"):
            runtime.activate_policy_node_producer(producer, producer_epoch=1)
    baseline_profile = ExecutionMeasuredPoseBaselineProfile(
        profile_id="rm65-measured-baseline",
        input_role="target_baseline",
        runtime_session_id="session",
        control_domain_id="rm65-sim",
        producer_id="rm65-measured-baseline",
        producer_epoch=1,
        action_schema=absolute,
        component_ids=("right_arm",),
        observation_binding_id="rm65-pose-observation",
        observation_plan_hash="rm65-pose-plan",
        maximum_measurement_age_sec=0.1,
        enabled=True,
    )
    residual_profile = B601RM65MotionInputProfile(
        profile_id="b601-rm65-residual",
        input_role="human_residual",
        runtime_session_id="session",
        control_domain_id="rm65-sim",
        producer_id="b601-residual",
        producer_epoch=1,
        action_schema=residual,
        maximum_adaptation_age_sec=0.1,
        enabled=True,
    )
    return (
        runtime,
        scene,
        contracts,
        baseline_profile,
        residual_profile,
    )


def _controller(runtime, contracts, baseline_profile, residual_profile):
    return B601RM65MujocoShadowController(
        profile=B601RM65MujocoShadowProfile(
            profile_id="b601-rm65-mujoco-shadow",
            enabled=True,
            send_actions=False,
        ),
        runtime=runtime,
        contracts=contracts,
        baseline_profile=baseline_profile,
        residual_profile=residual_profile,
    )


def _activate(controller, absolute_schema):
    return controller.activate(
        _activation(),
        _observation(absolute_schema),
        captured_at=10.03,
        effective_until=10.5,
    )


def test_profile_is_default_off_without_runtime_or_source() -> None:
    controller = B601RM65MujocoShadowController(
        profile=B601RM65MujocoShadowProfile()
    )

    assert controller.activate(
        _activation(),
        _observation(_schemas()[0]),
        captured_at=10.03,
        effective_until=10.5,
    ).status == "disabled"
    assert controller.evaluate(
        None,
        effective_from=10.04,
        effective_until=10.5,
        target_effective_at=10.05,
    ).status == "disabled"
    assert controller.active is None


def test_current_b601_residual_composes_on_actual_mujoco_shadow_owner() -> None:
    runtime, scene, contracts, baseline_profile, residual_profile = _build_runtime()
    controller = _controller(
        runtime,
        contracts,
        baseline_profile,
        residual_profile,
    )

    activation = _activate(controller, contracts.output_action_schema)
    cycle = controller.evaluate(
        _residual_step(contracts.residual_action_schema),
        effective_from=10.04,
        effective_until=10.5,
        target_effective_at=10.05,
    )

    assert activation.status == "activated"
    assert cycle.status == "evaluated"
    assert cycle.submission is not None
    assert cycle.submission.proposal_admission is None
    assert cycle.submission.composition_evaluation.output_status == "ready"
    assert cycle.conditional_action is not None
    value = np.asarray(cycle.conditional_action.value("right_arm").values)
    np.testing.assert_allclose(value[:3], (0.42, 0.0, 0.27), atol=1e-9)
    np.testing.assert_allclose(
        Rotation.from_quat(value[3:]).as_rotvec(),
        (0.0, 0.0, 0.10),
        atol=1e-9,
    )
    assert len(runtime.core.proposals) == 0
    assert scene.physics_commands == []
    assert not cycle.authoritative
    assert not cycle.promotable
    assert not cycle.command_dispatched
    runtime.close()


def test_rejected_current_source_does_not_evaluate_or_reuse_previous(
    monkeypatch,
) -> None:
    runtime, _scene, contracts, baseline_profile, residual_profile = _build_runtime()
    controller = _controller(
        runtime,
        contracts,
        baseline_profile,
        residual_profile,
    )
    _activate(controller, contracts.output_action_schema)
    first = controller.evaluate(
        _residual_step(contracts.residual_action_schema),
        effective_from=10.04,
        effective_until=10.5,
        target_effective_at=10.05,
    )
    assert first.status == "evaluated"

    monkeypatch.setattr(
        type(runtime.composition_port),
        "evaluate_composition_shadow",
        lambda _port, **_kwargs: pytest.fail(
            "a rejected current source must not evaluate retained inputs"
        ),
    )
    rejected = controller.evaluate(
        _residual_step(
            contracts.residual_action_schema,
            valid_current=False,
        ),
        effective_from=10.04,
        effective_until=10.5,
        target_effective_at=10.05,
    )

    assert rejected.status == "source_rejected"
    assert rejected.submission is None
    assert rejected.conditional_action is None
    assert rejected.residual_adaptation is not None
    assert "current_residual_unavailable" in (
        rejected.residual_adaptation.rejection_reason
    )
    runtime.close()


def test_shadow_rejects_a_cut_that_does_not_select_the_current_input(
    monkeypatch,
) -> None:
    runtime, _scene, contracts, baseline_profile, residual_profile = _build_runtime()
    controller = _controller(
        runtime,
        contracts,
        baseline_profile,
        residual_profile,
    )
    _activate(controller, contracts.output_action_schema)
    first = controller.evaluate(
        _residual_step(contracts.residual_action_schema),
        effective_from=10.04,
        effective_until=10.5,
        target_effective_at=10.05,
    )
    assert first.submission is not None
    monkeypatch.setattr(
        type(runtime.composition_port),
        "evaluate_composition_shadow",
        lambda _port, **_kwargs: first.submission,
    )

    with pytest.raises(RuntimeError, match="exact current inputs"):
        controller.evaluate(
            _residual_step(contracts.residual_action_schema),
            effective_from=10.04,
            effective_until=10.5,
            target_effective_at=10.05,
        )
    assert controller.active is None
    runtime.close()


def test_rejected_baseline_admission_does_not_leave_an_active_shadow() -> None:
    runtime, _scene, contracts, baseline_profile, residual_profile = _build_runtime(
        activate_inputs=False
    )
    controller = _controller(
        runtime,
        contracts,
        baseline_profile,
        residual_profile,
    )

    activation = _activate(controller, contracts.output_action_schema)

    assert activation.status == "baseline_rejected"
    assert activation.baseline_admission is not None
    assert not activation.baseline_admission.accepted
    assert controller.active is None
    with pytest.raises(RuntimeError, match=r"activate\(\)"):
        controller.evaluate(
            _residual_step(contracts.residual_action_schema),
            effective_from=10.04,
            effective_until=10.5,
            target_effective_at=10.05,
        )
    runtime.close()


def test_baseline_producer_sequence_advances_across_activations() -> None:
    runtime, _scene, contracts, baseline_profile, residual_profile = _build_runtime()
    controller = _controller(
        runtime,
        contracts,
        baseline_profile,
        residual_profile,
    )
    first = _activate(controller, contracts.output_action_schema)
    assert first.status == "activated"
    assert first.baseline_capture is not None
    assert first.baseline_capture.input_draft.sequence == 0

    runtime.clock.value = 11.04
    second = controller.activate(
        _activation(epoch=2, at=11.0),
        _observation(
            contracts.output_action_schema,
            suffix="2",
            measured_at=11.01,
            available_at=11.02,
        ),
        captured_at=11.03,
        effective_until=11.5,
    )

    assert second.status == "activated"
    assert second.baseline_capture is not None
    assert second.baseline_capture.input_draft.sequence == 1
    assert controller.active is second
    runtime.close()


def test_enabled_binding_requires_exact_runtime_identity() -> None:
    runtime, _scene, contracts, baseline_profile, residual_profile = _build_runtime()

    with pytest.raises(ValueError, match="runtime identity"):
        _controller(
            runtime,
            contracts,
            baseline_profile,
            replace(residual_profile, control_domain_id="another-domain"),
        )
    runtime.close()


def test_module_has_no_promotion_or_dispatch_path() -> None:
    source = (
        Path(__file__).parents[1]
        / "control_station"
        / "composition"
        / "rebot_b601_rm65_mujoco_shadow.py"
    ).read_text(encoding="utf-8")

    assert "from hardware" not in source
    assert "import hardware" not in source
    assert ".compose_and_submit(" not in source
    assert ".submit_evaluated_composition(" not in source
    assert ".step(" not in source
