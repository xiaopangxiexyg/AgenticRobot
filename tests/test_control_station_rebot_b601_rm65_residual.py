from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

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
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
    RigidFrameTransformSpec,
)
from teleop_runtime.action.human_residual import HumanResidualPoseAdapter


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


def test_composition_module_does_not_import_hardware_or_sim_backends() -> None:
    source_path = (
        Path(__file__).parents[1]
        / "control_station"
        / "composition"
        / "rebot_b601_rm65_residual.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "from hardware" not in source
    assert "import hardware" not in source
    assert "from sim" not in source
    assert "import sim" not in source


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


def _state(
    sequence: int,
    *,
    qpos: tuple[float, ...],
    observed_at: float,
    receive_at: float,
    valid: bool = True,
    stale: bool = False,
) -> B601HardwareState:
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
            stale=stale,
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
        stale=stale,
        freshness_verified=False,
        freshness_semantics="host_receive_sequence_and_age_verified",
        fault_reason=None if valid else "synthetic rejected frame",
        host_receive_freshness_verified=True,
    )


def _edge(
    transform_id: str,
    source: str,
    target: str,
    matrix: np.ndarray,
) -> RigidFrameTransformSpec:
    return RigidFrameTransformSpec(
        transform_id=transform_id,
        source_frame_id=source,
        target_frame_id=target,
        matrix_target_source=matrix,
    )


def _schema(component_id: str = "right_arm") -> ComponentActionSchema:
    payload = ActionPayloadSchema(
        schema_id="rm65-pose-residual",
        schema_version="development",
        representation_family="cartesian_pose",
        dtype="float64",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="relative_delta",
        reference_frame="rm65_base",
        controlled_frame="rm65_tool",
        semantic_parameters={
            "translation_application": "reference_frame_additive",
            "rotation_application": "baseline_local_right_multiply",
        },
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="rm65-pose-residual",
        catalog_version="development",
        components=(
            ControlComponentSpecV2(
                component_id=component_id,
                controlled_entity_id="rm65",
                command_channel_id="rm65-arm",
                selection_group_id="rm65-arm",
                resource_claims=("resource:rm65-arm",),
                accepted_payload_schema_hashes=(payload.semantic_content_hash,),
            ),
        ),
    )
    return ComponentActionSchema(
        schema_id="rm65-pose-residual",
        schema_version="development",
        component_catalog=catalog,
        payload_schema_by_component={component_id: payload},
    )


def _frame_graph(*, source_to_target_rotation_deg: float = 0.0) -> CoordinateFrameGraph:
    reference_transform = np.eye(4, dtype=np.float64)
    reference_transform[:3, :3] = Rotation.from_euler(
        "z", source_to_target_rotation_deg, degrees=True
    ).as_matrix()
    return CoordinateFrameGraph(
        graph_id="b601-rm65-test",
        frames=tuple(
            CoordinateFrameSpec(frame_id)
            for frame_id in (
                "base_link",
                "gripper_end",
                "rm65_base",
                "rm65_tool",
            )
        ),
        transforms=(
            _edge(
                "b601-base-to-rm65-base",
                "base_link",
                "rm65_base",
                reference_transform,
            ),
            _edge(
                "rm65-tool-to-b601-gripper",
                "rm65_tool",
                "gripper_end",
                np.eye(4, dtype=np.float64),
            ),
        ),
    )


def _source(
    states: list[B601HardwareState],
    *,
    source_to_target_rotation_deg: float = 0.0,
    translation_gain: object = 1.0,
) -> B601RM65ResidualSource:
    pose_source = B601ReadOnlyPoseSource(_Backend(states), _Model())
    residual_adapter = HumanResidualPoseAdapter(
        output_action_schema=_schema(),
        frame_graph=_frame_graph(
            source_to_target_rotation_deg=source_to_target_rotation_deg
        ),
        translation_gain_by_component={"right_arm": translation_gain},
    )
    return B601RM65ResidualSource(pose_source, residual_adapter)


def test_anchor_relative_translation_rotation_and_lineage() -> None:
    source = _source(
        [
            _state(
                1,
                qpos=(0.10, 0.20, 0.30, 0.0, 0.0, 0.0),
                observed_at=1.0,
                receive_at=1.1,
            ),
            _state(
                2,
                qpos=(0.12, 0.30, 0.27, 0.0, 0.0, 0.0),
                observed_at=1.2,
                receive_at=1.3,
            ),
        ],
        translation_gain=(2.0, 0.5, 1.0),
    )
    initial = source.poll_pose()
    anchor = source.capture_anchor(
        initial,
        anchor_generation=4,
        captured_at=2.0,
        captured_clock_domain_id="station.monotonic",
    )

    current = source.poll_pose()
    step = source.adapt(
        current,
        anchor=anchor,
        adapted_at=2.1,
        adapted_clock_domain_id="station.monotonic",
    )

    assert step.residual_available
    assert step.relative_action is not None
    assert not step.command_dispatched
    assert not step.authoritative
    assert current.source_emission_hash is not None
    assert current.source_observation_sample_id is not None
    assert current.pose_sample is not None
    assert current.source_emission_hash in current.pose_sample.sample_id
    value = np.asarray(step.relative_action.value("right_arm").values)
    np.testing.assert_allclose(value[:3], (0.04, 0.05, -0.03), atol=1e-9)
    np.testing.assert_allclose(
        Rotation.from_quat(value[3:]).as_rotvec(),
        (0.0, 0.0, 0.10),
        atol=1e-9,
    )
    assert step.adaptation is not None
    assert step.adaptation.receipt.anchor_generation == 4
    assert (
        step.adaptation.receipt.source_sample_hash_by_component["right_arm"]
        == current.pose_sample.semantic_content_hash
    )


def test_reference_frame_rotation_maps_b601_x_into_rm65_y() -> None:
    source = _source(
        [
            _state(
                1,
                qpos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                observed_at=1.0,
                receive_at=1.1,
            ),
            _state(
                2,
                qpos=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
                observed_at=1.2,
                receive_at=1.3,
            ),
        ],
        source_to_target_rotation_deg=90.0,
    )
    initial = source.poll_pose()
    anchor = source.capture_anchor(
        initial,
        anchor_generation=0,
        captured_at=2.0,
        captured_clock_domain_id="station.monotonic",
    )

    step = source.poll_and_adapt(
        anchor=anchor,
        adapted_at=2.1,
        adapted_clock_domain_id="station.monotonic",
    )

    assert step.relative_action is not None
    value = np.asarray(step.relative_action.value("right_arm").values)
    np.testing.assert_allclose(value[:3], (0.0, 0.1, 0.0), atol=1e-9)


def test_host_receive_time_orders_samples_when_estimated_time_regresses() -> None:
    """Receive-age jitter must not reject an advancing host-fresh frame."""

    source = _source(
        [
            _state(
                1,
                qpos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                observed_at=10.00,
                receive_at=10.10,
            ),
            _state(
                2,
                qpos=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
                # The reconstructed device-observation estimate regresses,
                # while the verified host receive cut advances.
                observed_at=9.99,
                receive_at=10.20,
            ),
        ]
    )
    initial = source.poll_pose()
    assert initial.pose_sample is not None
    assert initial.pose_sample.measured_at == pytest.approx(10.10)
    anchor = source.capture_anchor(
        initial,
        anchor_generation=1,
        captured_at=11.0,
        captured_clock_domain_id="station.monotonic",
    )

    current = source.poll_pose()
    assert current.pose_sample is not None
    assert current.pose_sample.measured_at == pytest.approx(10.20)
    step = source.adapt(
        current,
        anchor=anchor,
        adapted_at=11.1,
        adapted_clock_domain_id="station.monotonic",
    )

    assert step.residual_available, step.rejection_reason
    assert step.relative_action is not None
    np.testing.assert_allclose(
        np.asarray(step.relative_action.value("right_arm").values)[:3],
        (0.1, 0.0, 0.0),
        atol=1e-9,
    )


def test_rejected_current_sample_withdraws_residual_without_reuse() -> None:
    source = _source(
        [
            _state(
                1,
                qpos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                observed_at=1.0,
                receive_at=1.1,
            ),
            _state(
                2,
                qpos=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
                observed_at=1.2,
                receive_at=1.3,
            ),
            _state(
                3,
                qpos=(0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
                observed_at=1.4,
                receive_at=1.5,
                valid=False,
                stale=True,
            ),
        ]
    )
    initial = source.poll_pose()
    anchor = source.capture_anchor(
        initial,
        anchor_generation=0,
        captured_at=2.0,
        captured_clock_domain_id="station.monotonic",
    )
    valid = source.poll_and_adapt(
        anchor=anchor,
        adapted_at=2.1,
        adapted_clock_domain_id="station.monotonic",
    )

    rejected = source.poll_and_adapt(
        anchor=anchor,
        adapted_at=2.2,
        adapted_clock_domain_id="station.monotonic",
    )

    assert valid.relative_action is not None
    assert rejected.relative_action is None
    assert not rejected.residual_available
    assert "source_rejected" in rejected.rejection_reason
    assert "synthetic rejected frame" in rejected.rejection_reason


def test_recenter_uses_new_explicit_anchor_generation() -> None:
    source = _source(
        [
            _state(
                1,
                qpos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                observed_at=1.0,
                receive_at=1.1,
            ),
            _state(
                2,
                qpos=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
                observed_at=1.2,
                receive_at=1.3,
            ),
        ]
    )
    source.poll_pose()
    recentered = source.poll_pose()
    anchor = source.capture_anchor(
        recentered,
        anchor_generation=7,
        captured_at=2.0,
        captured_clock_domain_id="station.monotonic",
    )

    step = source.adapt(
        recentered,
        anchor=anchor,
        adapted_at=2.1,
        adapted_clock_domain_id="station.monotonic",
    )

    assert step.relative_action is not None
    np.testing.assert_allclose(
        step.relative_action.value("right_arm").values,
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        atol=1e-9,
    )
    assert step.adaptation is not None
    assert step.adaptation.receipt.anchor_generation == 7


def test_constructor_rejects_nonmatching_residual_component() -> None:
    pose_source = B601ReadOnlyPoseSource(
        _Backend(
            [
                _state(
                    1,
                    qpos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    observed_at=1.0,
                    receive_at=1.1,
                )
            ]
        ),
        _Model(),
    )
    adapter = HumanResidualPoseAdapter(
        output_action_schema=_schema("arm"),
        frame_graph=_frame_graph(),
    )

    with pytest.raises(ValueError, match="exact single-component"):
        B601RM65ResidualSource(pose_source, adapter)
