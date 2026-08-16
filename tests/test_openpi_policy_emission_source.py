from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from deployments.openpi.client import policy as openpi_policy_module
from control_station.sim_hil import (
    _config_from_args,
    _run_dry_run,
    build_arg_parser,
)
from dataset_contracts.policy_artifact import (
    PolicyArtifactContract,
    PolicyCameraInputContract,
)
from dataset_contracts.pose_chunk import PoseActionChunkContract
from dataset_contracts.pose_training import PoseTrainingDataContract
from dataset_contracts.v22 import PolicyIOLayout
from deployments.openpi.client import (
    OpenPIEmissionSourceConfig,
    build_openpi_policy_emission_source,
    validate_openpi_state_policy_artifact,
)
from deployments.openpi.client.normalization import (
    make_openpi_quantile_normalization_contract,
)
from deployments.openpi.state_only import (
    OPENPI_STATE_ONLY_CAPABILITY_KEY_V1,
    OpenPIStateOnlyInputBridge,
)
from robot_motion.action import ArmActionSpec, HandActionSpec
from robot_motion.deploy import build_deployment_observation
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_HAND_SLICES
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_runtime_pose_frame_contract,
)
from sim.contracts import TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.config import GripperSpec
from sim.mujoco.deploy import (
    MujocoComponentActionRobotActionV1Materializer,
    MujocoDeployConfig,
    MujocoDeployRuntime,
    MujocoRobotActionV1ComponentIngressAdapter,
    build_mujoco_component_execution_binding,
    build_rm65_right_cube_lift_component_target_profile,
)


def _artifact(*, prepared: bool = True) -> PolicyArtifactContract:
    pose_contract = (
        default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
    )
    metadata = {
        OPENPI_STATE_ONLY_CAPABILITY_KEY_V1: OpenPIStateOnlyInputBridge(
            image_keys=("base_0_rgb",)
        ).contract,
    }
    if prepared:
        metadata["normalization_contract"] = (
            make_openpi_quantile_normalization_contract(
                stats_ref="norm_stats.json",
                stats_hash="sha256:" + "1" * 64,
            )
        )
    return PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id="state-only-fixture",
        checkpoint_step=60_000,
        dataset_schema="dexteleop_dataset_v2.3",
        camera_input=PolicyCameraInputContract(inputs=(), required=False),
        pose_action_chunk=PoseActionChunkContract(
            action_horizon=2,
            action_pose_mode="absolute",
        ),
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=(pose_contract.contract_hash,),
        metadata=metadata,
    )


def _observation(step_index: int):
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = (0.2, 0.0, 0.3)
    return build_deployment_observation(
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        end_effector_sides=("left", "right"),
        step_index=step_index,
        arm_states={
            side: {
                "current_qpos": np.zeros(6),
                "actual_qpos": np.zeros(6),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
            for side in ("left", "right")
        },
        hand_states={
            side: {
                "joint_pos": np.zeros(6),
                "gripper_state_scalar": 0.0,
                "joint_timestamp": time.monotonic(),
            }
            for side in ("left", "right")
        },
        timestamp=time.monotonic(),
        metadata={
            "arm_action_specs_by_side": {
                side: ArmActionSpec.tool_tcp(
                    side,
                    reference_frame=f"{side}_arm_base",
                ).to_metadata()
                for side in ("left", "right")
            },
            "hand_action_specs_by_side": {
                side: HandActionSpec.dmtac_gripper(side).to_metadata()
                for side in ("left", "right")
            },
            "pose_frame_contract": (
                default_rm65_workspace_near_boundary_runtime_pose_frame_contract().to_dict()
            ),
        },
    )


class _FakeStatePolicy:
    name = "fake-state-policy"

    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        return None

    def predict(self, observation):
        assert tuple(observation.active_sides) == ("left", "right")
        assert observation.state.shape == (38,)
        self.calls += 1
        actions = np.full((2, 26), np.nan, dtype=np.float32)
        for index in range(2):
            for side in ("left", "right"):
                actions[index, ACTION_ARM_POSE_SLICES[side]] = (
                    0.1 + index * 0.01,
                    0.0,
                    0.2,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                )
                actions[index, ACTION_HAND_SLICES[side]] = 0.25
        return actions


class _FakeMujocoTargetScene:
    arm_specs = {"left": object(), "right": object()}
    gripper_specs = {
        "right_gripper": GripperSpec(
            name="right_gripper",
            joint_names=(
                "right_gripper_left_finger_joint",
                "right_gripper_right_finger_joint",
            ),
            open_qpos=(0.0, 0.0),
            closed_qpos=(-0.035, -0.035),
            side="right",
        )
    }

    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }

    def get_arm_qpos(self, side: str) -> np.ndarray:
        return self.qpos[side].copy()

    def step_kinematic_direct_qpos_debug(
        self,
        qpos_by_side,
        *,
        gripper_close_by_name,
        zero_controlled_qvel,
    ) -> None:
        del gripper_close_by_name, zero_controlled_qvel
        self.qpos = {
            side: np.asarray(qpos, dtype=float).reshape(6).copy()
            for side, qpos in qpos_by_side.items()
        }


def test_openpi_client_deferred_connect_waits_for_worker_predict(
    monkeypatch,
) -> None:
    events: list[str] = []

    class _DeferredTransport:
        def __init__(self, **_kwargs) -> None:
            events.append("transport_constructed")

        def reset(self) -> None:
            events.append("transport_reset")

        def request_cancel(self) -> bool:
            events.append("transport_cancel")
            return True

        def begin_session(self) -> bool:
            events.append("transport_begin_session")
            return True

        def close(self) -> None:
            events.append("transport_close")

    class _DeferredRemotePolicy:
        def __init__(self, *, transport, observation_adapter, config) -> None:
            events.append("metadata_negotiated")
            self.transport = transport
            self.observation_adapter = observation_adapter
            self.config = config
            self.server_metadata = {"model_id": "deferred-fixture"}
            self.metadata = {"server": dict(self.server_metadata)}

        def predict(self, observation):
            events.append("predict")
            return observation

    monkeypatch.setattr(
        openpi_policy_module,
        "OpenPIWebsocketTransport",
        _DeferredTransport,
    )
    monkeypatch.setattr(
        openpi_policy_module,
        "RemoteModelPolicy",
        _DeferredRemotePolicy,
    )

    policy = openpi_policy_module.OpenPIClientPolicy(
        openpi_policy_module.OpenPIClientPolicyConfig(defer_connect=True)
    )

    assert events == ["transport_constructed"]
    assert policy.metadata["connection"] == "deferred"
    policy.reset()
    assert events == ["transport_constructed", "transport_reset"]

    observation = object()
    assert policy.predict(observation) is observation
    assert events == [
        "transport_constructed",
        "transport_reset",
        "metadata_negotiated",
        "predict",
    ]
    assert policy.name == "openpi:deferred-fixture"
    assert "connection" not in policy.metadata


def test_openpi_source_requires_prepared_state_only_artifact() -> None:
    artifact = _artifact()
    validate_openpi_state_policy_artifact(artifact)
    config = OpenPIEmissionSourceConfig(
        policy_artifact=artifact,
        runtime_pose_frame_contract=(
            default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
        ),
        output_active_sides=("right",),
        model_observation_sides=("left", "right"),
        prediction_request_queue_threshold=1,
    )
    assert config.output_active_sides == ("right",)
    assert config.model_observation_sides == ("left", "right")

    with pytest.raises(ValueError, match="prepared deployment artifact"):
        OpenPIEmissionSourceConfig(
            policy_artifact=_artifact(prepared=False),
            runtime_pose_frame_contract=(
                default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
            ),
            output_active_sides=("right",),
            prediction_request_queue_threshold=1,
        )

    tampered = _artifact().to_dict()
    tampered["metadata"][OPENPI_STATE_ONLY_CAPABILITY_KEY_V1][
        "contract_hash"
    ] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="capability hash or fields"):
        validate_openpi_state_policy_artifact(
            PolicyArtifactContract.from_dict(tampered)
        )


def test_openpi_source_observes_bimanual_state_and_emits_explicit_right_scope() -> None:
    policy = _FakeStatePolicy()
    next_step = 0

    def capture(*, step_index=None):
        nonlocal next_step
        selected = next_step if step_index is None else int(step_index)
        next_step = selected + 1
        return _observation(selected)

    source = build_openpi_policy_emission_source(
        OpenPIEmissionSourceConfig(
            policy_artifact=_artifact(),
            runtime_pose_frame_contract=(
                default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
            ),
            output_active_sides=("right",),
            model_observation_sides=("left", "right"),
            action_hz=50.0,
            policy_hz=50.0,
            prediction_request_queue_threshold=1,
        ),
        observation_capture=capture,
        policy=policy,
    )
    source.start()
    try:
        deadline = time.monotonic() + 1.0
        emission = None
        while emission is None and time.monotonic() < deadline:
            emission = source.step(now=time.monotonic())
            time.sleep(0.002)
        assert emission is not None
        assert emission.pre_safety_robot_action is not None
        assert emission.pre_safety_robot_action.active_effectors == (
            "right_arm",
            "right_hand",
        )
        assert source.status()["model_observation_sides"] == (
            "left",
            "right",
        )
        assert policy.calls == 1
    finally:
        source.close()


def test_openpi_right_scope_enters_native_mujoco_target_without_composition() -> None:
    policy = _FakeStatePolicy()
    next_step = 0

    def capture(*, step_index=None):
        nonlocal next_step
        selected = next_step if step_index is None else int(step_index)
        next_step = selected + 1
        return _observation(selected)

    source = build_openpi_policy_emission_source(
        OpenPIEmissionSourceConfig(
            policy_artifact=_artifact(),
            runtime_pose_frame_contract=(
                default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
            ),
            output_active_sides=("right",),
            model_observation_sides=("left", "right"),
            action_hz=50.0,
            policy_hz=50.0,
            prediction_request_queue_threshold=1,
        ),
        observation_capture=capture,
        policy=policy,
    )
    scene = _FakeMujocoTargetScene()
    profile = build_rm65_right_cube_lift_component_target_profile(
        scene,
        SimpleNamespace(
            controlled_side="right",
            right_gripper_name="right_gripper",
        ),
    )
    materializer = MujocoComponentActionRobotActionV1Materializer(
        scene,
        profile,
        hand_action_spec_by_component={
            "right_gripper": HandActionSpec.dmtac_gripper("right")
        },
        support_arm_pose_provider=lambda: {
            "left": np.asarray(
                [0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
        },
        support_arm_sides=("left",),
        support_pose_contract_id="test.measured-left-support.v1",
    )
    ingress = MujocoRobotActionV1ComponentIngressAdapter(
        scene,
        profile,
        component_id_by_effector={
            "right_arm": "right_arm",
            "right_hand": "right_gripper",
        },
    )
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=TimingSpec(
                sample_hz=50.0,
                policy_hz=50.0,
                reference_hz=100.0,
                ik_hz=100.0,
                command_hz=100.0,
                physics_hz=1000.0,
            ),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="openpi-native-target-test",
            control_domain_id="openpi-native-target-domain",
            embodiment_instance_id="openpi-native-target-instance",
            policy_node_id="openpi-policy",
            decision_hz=50.0,
            clock_domain_id="local_monotonic",
        ),
        component_execution_binding=build_mujoco_component_execution_binding(
            profile,
            materializer,
            policy_action_ingress=ingress,
        ),
        clock=time.monotonic,
    )

    runtime.start()
    source.start()
    try:
        deadline = time.monotonic() + 1.0
        emission = None
        while emission is None and time.monotonic() < deadline:
            emission = source.step(now=time.monotonic())
            time.sleep(0.002)
        assert emission is not None

        submission = runtime.submit_policy_action_emission(emission)

        assert submission.proposal_admission is not None
        assert submission.proposal_admission.accepted is True
        assert runtime.action_schema.component_ids == (
            "right_arm",
            "right_gripper",
        )
        assert runtime.composition_runtime_config is None
        assert runtime.composition_port is None
    finally:
        source.close()
        runtime.close()


@pytest.mark.slow_sim
def test_sim_hil_openpi_dry_run_is_offline_and_reports_explicit_projection(
    tmp_path,
    capsys,
) -> None:
    artifact_path = tmp_path / "prepared.policy_artifact.json"
    artifact_path.write_text(
        json.dumps(_artifact().to_dict()),
        encoding="utf-8",
    )
    args = build_arg_parser().parse_args(
        [
            "--dry-run",
            "--policy-backend",
            "openpi_state",
            "--openpi-policy-artifact",
            str(artifact_path),
            "--openpi-prediction-queue-threshold",
            "1",
            "--vive-coordinate-profile",
            str(
                (
                    Path(__file__).resolve().parents[1]
                    / "teleop_records"
                    / "control_station_vive_coordinate_profiles"
                    / "vive_rm65_virtual_desk_coordinate_v1.json"
                )
            ),
        ]
    )

    assert _run_dry_run(_config_from_args(args)) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["policy_backend"] == "openpi_state"
    assert report["policy_topology"] == {
        "model_observation_sides": ["left", "right"],
        "output_active_sides": ["right"],
    }
    assert report["compatibility"]["server_connection_checked"] is False
    assert report["compatibility"]["full_bimanual_execution"] is False
