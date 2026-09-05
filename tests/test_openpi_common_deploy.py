from __future__ import annotations

import argparse
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from deployments.client import (
    DeployPolicyContract,
    LoadedPolicy,
    RemoteModelPolicy,
    RemotePolicyConfig,
    ServerMetadataError,
    UnifiedActionChunk,
    validate_server_metadata,
)
from deployments.client.artifact import deploy_contract_from_policy_artifact
from deployments.openpi.client import (
    OPENPI_LOCAL_PROTOCOL_VERSION,
    OpenPIClientPolicy,
    OpenPIClientPolicyConfig,
    OpenPIObservationAdapterConfig,
    deployment_observation_to_openpi,
)
from deployments.openpi.client import real as openpi_real_launcher
from dataset_contracts.v22 import DiscreteCategory, DiscreteTokenSpec, EndEffectorSpec, PolicyIOLayout
from dataset_contracts import (
    MODEL_CONTINUATION_FLOW_INPAINT_V1,
    MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
    STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT,
    TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    ModelContinuationDataContract,
    ModelContinuationDataContractV2,
    ModelContinuationRuntimeContractV2,
    PolicyArtifactContract,
    PolicyCameraInput,
    PolicyCameraInputContract,
)
from dataset_contracts.pose_chunk import PoseActionChunkContract
from dataset_contracts.pose_training import PoseTrainingDataContract
from teleop_runtime.deploy.actions import normalize_policy_output
from teleop_runtime.deploy import camera_runtime, cli as deploy_cli
from teleop_runtime.deploy.observation import DeploymentObservation
from teleop_runtime.deploy import run_policy
from teleop_runtime.deploy.schema import ACTION_DIM, OBSERVATION_STATE_DIM
from robot_motion.deploy import (
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
)


def _server_metadata() -> dict:
    return {
        "protocol_version": OPENPI_LOCAL_PROTOCOL_VERSION,
        "deploy_schema_version": "dexteleop_deploy_v1",
        "model_id": "dual_rm65_dmgripper_latest",
        "deployment": {
            "config_name": "dual_rm65_dmgripper",
            "checkpoint_dir": "/tmp/ckpt/100000",
            "checkpoint_step": 100000,
        },
        "model": {"action_horizon": 50},
        "local_robot": {
            "output_action_dim": ACTION_DIM,
            "active_sides": ["left", "right"],
            "hand_layout": "q6_slot0_gripper",
        },
        "dataset_adapter": {
            "deployment_action_dim": ACTION_DIM,
            "camera_slots": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
            "deployment_action_order": [
                "left_arm_pose_7",
                "right_arm_pose_7",
                "left_hand_q6_slot0_gripper",
                "right_hand_q6_slot0_gripper",
            ],
        },
    }


def _observation() -> DeploymentObservation:
    return DeploymentObservation(
        timestamp=1.0,
        step_index=3,
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        images={"front": np.full((8, 10, 3), 127, dtype=np.uint8)},
        state=np.zeros((OBSERVATION_STATE_DIM,), dtype=np.float32),
        metadata={"prompt": "test prompt"},
    )


class _FakeTransport:
    def __init__(self, *, actions: np.ndarray | None = None, metadata: dict | None = None) -> None:
        self._metadata = _server_metadata() if metadata is None else dict(metadata)
        self.actions = np.zeros((2, ACTION_DIM), dtype=np.float32) if actions is None else actions
        self.requests: list[dict] = []
        self.reset_count = 0

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    def infer(self, observation: dict) -> dict:
        self.requests.append(observation)
        return {
            "actions": self.actions,
            "server_timing": {"infer_ms": 1.0},
            "policy_timing": {"infer_ms": 0.5},
        }

    def reset(self) -> None:
        self.reset_count += 1


class _V22FakeTransport(_FakeTransport):
    def __init__(self, layout: PolicyIOLayout) -> None:
        metadata = _server_metadata()
        metadata["local_robot"]["output_action_dim"] = layout.action_dim
        metadata["local_robot"]["output_action_discrete_dim"] = layout.action_discrete_dim
        metadata["dataset_adapter"]["policy_io_layout_hash"] = layout.layout_hash
        metadata["model"]["action_discrete_dim"] = layout.action_discrete_dim
        actions = np.asarray(
            [[-0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        super().__init__(actions=actions, metadata=metadata)
        self.discrete_actions = np.asarray([[1]], dtype=np.int64)

    def infer(self, observation: dict) -> dict:
        response = super().infer(observation)
        response["actions_discrete"] = self.discrete_actions
        return response


class OpenPICommonDeployTest(unittest.TestCase):
    def test_run_policy_reexports_canonical_camera_runtime(self) -> None:
        for name in (
            "DEFAULT_DEPLOYMENT_CAMERA_FRAME_MAX_AGE_SEC",
            "DEFAULT_DEPLOYMENT_CAMERA_KEEP_LAST",
            "REAL_TELEOP_CAMERA_START_STAGGER_SEC",
            "build_camera_sources",
            "resolve_policy_camera_input_keys",
            "run_camera_startup_self_check",
            "select_preview_camera_sources",
            "start_camera_sources",
        ):
            self.assertIs(getattr(run_policy, name), getattr(camera_runtime, name))

    def test_openpi_policy_artifact_selects_protocol_keys_and_dimensions(self) -> None:
        layout = PolicyIOLayout.legacy_v21_q6()
        artifact = PolicyArtifactContract(
            policy_io_layout=layout,
            model_id="artifact-model",
            state_key="robot_state",
            state_discrete_key="robot_state_tokens",
            action_key="robot_actions",
            action_discrete_key="robot_action_tokens",
        )

        config = OpenPIClientPolicyConfig(policy_artifact=artifact)

        self.assertEqual(config.deploy_schema_version, "dexteleop_deploy_v2")
        self.assertIsNone(config.expected_model_id)
        self.assertIsNone(config.expected_checkpoint_step)
        self.assertEqual(config.action_dim, layout.action_dim)
        self.assertEqual(config.action_key, "robot_actions")
        self.assertEqual(config.policy_io_layout_hash, layout.layout_hash)

    def test_artifact_protocol_contract_does_not_bind_model_identity_by_default(self) -> None:
        layout = PolicyIOLayout.legacy_v21_q6()
        artifact = PolicyArtifactContract(
            policy_io_layout=layout,
            model_id="artifact-model-150000",
            checkpoint_step=150000,
        )
        contract = deploy_contract_from_policy_artifact(
            artifact,
            protocol_version=OPENPI_LOCAL_PROTOCOL_VERSION,
        )

        metadata = {
            "protocol_version": OPENPI_LOCAL_PROTOCOL_VERSION,
            "deploy_schema_version": "dexteleop_deploy_v2",
            "model_id": "switched-model-90000",
            "deployment": {"checkpoint_step": 90000},
            "model": {
                "action_horizon": 50,
                "action_discrete_dim": layout.action_discrete_dim,
            },
            "local_robot": {
                "input_state_key": "state",
                "input_state_dim": layout.state_dim,
                "input_state_discrete_key": "state_discrete",
                "input_state_discrete_dim": layout.state_discrete_dim,
                "output_action_dim": layout.action_dim,
                "output_action_discrete_dim": layout.action_discrete_dim,
                "active_sides": ["left", "right"],
            },
            "dataset_adapter": {
                "policy_io_layout_hash": layout.layout_hash,
            },
        }

        result = validate_server_metadata(metadata, contract)

        self.assertTrue(result.ok)
        self.assertIsNone(contract.model_id)
        self.assertIsNone(contract.checkpoint_step)

    def test_openpi_policy_accepts_switched_checkpoint_with_same_protocol(self) -> None:
        layout = PolicyIOLayout.legacy_v21_q6()
        artifact = PolicyArtifactContract(
            policy_io_layout=layout,
            model_id="artifact-model-150000",
            checkpoint_step=150000,
        )
        metadata = _server_metadata()
        metadata.update(
            {
                "deploy_schema_version": "dexteleop_deploy_v2",
                "model_id": "switched-model-90000",
                "deployment": {"checkpoint_step": 90000},
            }
        )
        metadata["model"]["action_discrete_dim"] = layout.action_discrete_dim
        metadata["local_robot"].update(
            {
                "input_state_key": "state",
                "input_state_dim": layout.state_dim,
                "input_state_discrete_key": "state_discrete",
                "input_state_discrete_dim": layout.state_discrete_dim,
                "output_action_dim": layout.action_dim,
                "output_action_discrete_dim": layout.action_discrete_dim,
            }
        )
        metadata["dataset_adapter"]["policy_io_layout_hash"] = layout.layout_hash
        remote = RemoteModelPolicy(
            transport=_FakeTransport(metadata=metadata),
            observation_adapter=lambda observation: {},
            config=RemotePolicyConfig(action_dim=layout.action_dim),
        )

        policy = OpenPIClientPolicy(
            OpenPIClientPolicyConfig(policy_artifact=artifact),
            remote_policy=remote,
        )

        self.assertEqual(policy.name, "openpi:switched-model-90000")

    def test_openpi_policy_artifact_selects_only_declared_wrist_view(self) -> None:
        artifact = PolicyArtifactContract(
            policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
            camera_input=PolicyCameraInputContract(
                inputs=(
                    PolicyCameraInput(
                        source_key="left_wrist",
                        model_input_key="observation.images.base_0_rgb",
                        server_camera_slot="base_0_rgb",
                    ),
                ),
            ),
        )

        config = OpenPIClientPolicyConfig(policy_artifact=artifact)
        adapter = OpenPIObservationAdapterConfig(
            image_key_map=config.image_key_map,
            required_image_keys=config.required_image_keys,
        )
        base = _observation()
        observation = DeploymentObservation(
            timestamp=base.timestamp,
            step_index=base.step_index,
            active_sides=base.active_sides,
            arms_enabled=base.arms_enabled,
            hands_enabled=base.hands_enabled,
            images={
                "front": np.full((8, 10, 3), 1, dtype=np.uint8),
                "left_wrist": np.full((8, 10, 3), 2, dtype=np.uint8),
            },
            state=base.state,
            metadata=base.metadata,
        )

        request = deployment_observation_to_openpi(observation, adapter)

        self.assertEqual(config.required_camera_slots, ("base_0_rgb",))
        self.assertEqual(config.required_image_keys, ("left_wrist",))
        self.assertIn("observation.images.base_0_rgb", request)
        self.assertNotIn("observation.images.front", request)
        np.testing.assert_array_equal(
            request["observation.images.base_0_rgb"], observation.images["left_wrist"]
        )
        with self.assertRaisesRegex(ValueError, "disagrees with policy_artifact"):
            OpenPIClientPolicyConfig(
                policy_artifact=artifact,
                required_image_keys=("front",),
            )

    def test_artifact_camera_contract_rejects_none_mode_when_required(self) -> None:
        artifact = PolicyArtifactContract(
            policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
            camera_input=PolicyCameraInputContract(
                inputs=(
                    PolicyCameraInput(
                        source_key="left_wrist",
                        model_input_key="observation.images.base_0_rgb",
                        server_camera_slot="base_0_rgb",
                    ),
                ),
            ),
        )
        args = argparse.Namespace(camera_mode="none", camera_serial="", camera_key="front")

        with self.assertRaisesRegex(ValueError, "requires camera inputs"):
            run_policy.resolve_policy_camera_input_keys(
                args,
                artifact,
                raw_argv=[],
            )

    def test_artifact_camera_contract_rekeys_serial_camera_for_wrist_only_model(self) -> None:
        artifact = PolicyArtifactContract(
            policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
            camera_input=PolicyCameraInputContract(
                inputs=(
                    PolicyCameraInput(
                        source_key="right_wrist",
                        model_input_key="observation.images.base_0_rgb",
                        server_camera_slot="base_0_rgb",
                    ),
                ),
            ),
        )
        args = argparse.Namespace(camera_mode="single", camera_serial="123", camera_key="front")

        selected = run_policy.resolve_policy_camera_input_keys(args, artifact, raw_argv=[])

        self.assertEqual(selected, ("right_wrist",))
        self.assertEqual(args.camera_key, "right_wrist")

    def test_runtime_opens_only_artifact_selected_camera_configs(self) -> None:
        args = argparse.Namespace(
            camera_mode="multi",
            camera_serial="",
            camera_config="unused.json",
            camera_keep_last=4,
        )
        configs = {
            "front": object(),
            "left_wrist": object(),
            "right_wrist": object(),
        }
        with mock.patch(
            "teleop_runtime.deploy.camera_runtime.load_realsense_camera_config_map",
            return_value=configs,
        ), mock.patch(
            "teleop_runtime.deploy.camera_runtime.RealSenseRGBSource",
            side_effect=lambda config: config,
        ):
            sources = run_policy.build_camera_sources(
                args,
                selected_camera_keys=("left_wrist",),
            )

        self.assertEqual(sources, {"left_wrist": configs["left_wrist"]})

    def test_runtime_applies_deployment_camera_history_minimum_to_config_sources(self) -> None:
        args = argparse.Namespace(
            camera_mode="multi",
            camera_serial="",
            camera_config="unused.json",
            camera_keep_last=4,
        )
        config = camera_runtime.RealSenseRGBCameraConfig(
            serial="123",
            camera_key="left_wrist",
            keep_last=2,
            color_format="yuyv",
            startup_warmup_policy="legacy",
        )
        with mock.patch(
            "teleop_runtime.deploy.camera_runtime.load_realsense_camera_config_map",
            return_value={"left_wrist": config},
        ), mock.patch(
            "teleop_runtime.deploy.camera_runtime.RealSenseRGBSource",
            side_effect=lambda value: value,
        ):
            sources = run_policy.build_camera_sources(
                args,
                selected_camera_keys=("left_wrist",),
            )

        self.assertEqual(sources["left_wrist"].keep_last, 4)
        self.assertEqual(sources["left_wrist"].color_format, "bgr8")
        self.assertEqual(
            sources["left_wrist"].startup_warmup_policy,
            "bounded_first_frame",
        )
        self.assertEqual(
            sources["left_wrist"].metadata["configured_color_format"],
            "yuyv",
        )
        self.assertEqual(
            sources["left_wrist"].metadata[
                "configured_startup_warmup_policy"
            ],
            "legacy",
        )

    def test_runtime_serial_camera_uses_formal_collection_startup_defaults(self) -> None:
        parser = argparse.ArgumentParser()
        run_policy.add_args(parser)
        args = parser.parse_args(
            [
                "--camera-mode",
                "single",
                "--camera-serial",
                "123",
            ]
        )

        with mock.patch(
            "teleop_runtime.deploy.camera_runtime.RealSenseRGBSource",
            side_effect=lambda config: config,
        ):
            sources = run_policy.build_camera_sources(args)

        config = sources["front"]
        self.assertEqual(config.color_format, "bgr8")
        self.assertEqual(config.startup_warmup_policy, "bounded_first_frame")
        self.assertFalse(config.startup_hardware_reset)

    def test_deployment_camera_age_default_has_jitter_margin(self) -> None:
        parser = argparse.ArgumentParser()
        run_policy.add_args(parser)

        args = parser.parse_args([])

        self.assertEqual(
            args.camera_frame_max_age_sec,
            run_policy.DEFAULT_DEPLOYMENT_CAMERA_FRAME_MAX_AGE_SEC,
        )

    def test_validate_server_metadata_accepts_current_dual_rm65_contract(self) -> None:
        result = validate_server_metadata(
            _server_metadata(),
            DeployPolicyContract(
                protocol_version=OPENPI_LOCAL_PROTOCOL_VERSION,
                deploy_schema_version="dexteleop_deploy_v1",
                model_id="dual_rm65_dmgripper_latest",
                checkpoint_step=100000,
                required_camera_slots=("base_0_rgb",),
            ),
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.server_summary["action_dim"], ACTION_DIM)

    def test_validate_server_metadata_rejects_action_dim_mismatch(self) -> None:
        metadata = _server_metadata()
        metadata["local_robot"]["output_action_dim"] = 16

        with self.assertRaises(ServerMetadataError):
            validate_server_metadata(metadata, DeployPolicyContract(action_dim=ACTION_DIM))

    def test_openpi_observation_adapter_builds_request(self) -> None:
        request = deployment_observation_to_openpi(_observation())

        self.assertIn("observation.images.front", request)
        self.assertEqual(request["observation.images.front"].dtype, np.uint8)
        self.assertEqual(request["state"].shape, (OBSERVATION_STATE_DIM,))
        self.assertEqual(request["prompt"], "test prompt")

    def test_openpi_observation_adapter_enforces_artifact_prompt_allowlist(self) -> None:
        allowed = OpenPIObservationAdapterConfig(
            allowed_prompts=("test prompt", "return the brush"),
        )
        request = deployment_observation_to_openpi(_observation(), allowed)
        self.assertEqual(request["prompt"], "test prompt")

        rejected = OpenPIObservationAdapterConfig(
            prompt="unknown stage",
            allowed_prompts=("test prompt", "return the brush"),
        )
        with self.assertRaisesRegex(ValueError, "not allowed"):
            deployment_observation_to_openpi(_observation(), rejected)

    def test_openpi_observation_adapter_requires_front_image(self) -> None:
        observation = _observation()
        observation = DeploymentObservation(
            timestamp=observation.timestamp,
            step_index=observation.step_index,
            active_sides=observation.active_sides,
            arms_enabled=observation.arms_enabled,
            hands_enabled=observation.hands_enabled,
            state=observation.state,
        )

        with self.assertRaises(ValueError):
            deployment_observation_to_openpi(observation, OpenPIObservationAdapterConfig())

    def test_generic_remote_policy_returns_unified_action_chunk(self) -> None:
        transport = _FakeTransport()
        policy = RemoteModelPolicy(
            transport=transport,
            observation_adapter=lambda observation: deployment_observation_to_openpi(observation),
            config=RemotePolicyConfig(
                contract=DeployPolicyContract(
                    protocol_version=OPENPI_LOCAL_PROTOCOL_VERSION,
                    model_id="dual_rm65_dmgripper_latest",
                ),
                source="openpi_remote_policy",
            ),
        )

        chunk = policy.predict(_observation())

        self.assertIsInstance(chunk, UnifiedActionChunk)
        self.assertEqual(chunk.actions.shape, (2, ACTION_DIM))
        self.assertEqual(transport.requests[0]["prompt"], "test prompt")
        self.assertEqual(chunk.metadata["source"], "openpi_remote_policy")

    def test_generic_remote_policy_rejects_bad_action_dim(self) -> None:
        transport = _FakeTransport(actions=np.zeros((2, 16), dtype=np.float32))
        policy = RemoteModelPolicy(
            transport=transport,
            observation_adapter=lambda observation: {},
            config=RemotePolicyConfig(action_dim=ACTION_DIM),
        )

        with self.assertRaises(ValueError):
            policy.predict(_observation())

    def test_openpi_specific_policy_wraps_generic_remote_policy(self) -> None:
        remote = RemoteModelPolicy(
            transport=_FakeTransport(),
            observation_adapter=lambda observation: deployment_observation_to_openpi(observation),
            config=RemotePolicyConfig(source="openpi_remote_policy"),
        )
        policy = OpenPIClientPolicy(remote_policy=remote)

        chunk = policy.predict(_observation())

        self.assertEqual(chunk.actions.shape, (2, ACTION_DIM))
        self.assertEqual(policy.name, "openpi:dual_rm65_dmgripper_latest")

    def test_openpi_v22_policy_sends_int64_state_tokens_and_uses_layout_bound_output(self) -> None:
        token = DiscreteTokenSpec(
            "mode",
            (
                DiscreteCategory(0, "open", backend_value=0.0),
                DiscreteCategory(1, "close", backend_value=0.9),
            ),
            source_metadata_key="mode",
        )
        layout = PolicyIOLayout(
            name="openpi_v22_right_discrete",
            end_effectors={
                "left": EndEffectorSpec.none("left"),
                "right": EndEffectorSpec.discrete(
                    "right",
                    adapter="two_state_gripper_v1",
                    target_tokens=(token,),
                    target_legacy_q6_slot_indices=(0,),
                ),
            },
        )
        transport = _V22FakeTransport(layout)
        remote = RemoteModelPolicy(
            transport=transport,
            observation_adapter=lambda observation: deployment_observation_to_openpi(observation),
            config=RemotePolicyConfig(
                action_dim=layout.action_dim,
                action_discrete_key="actions_discrete",
                action_discrete_dim=layout.action_discrete_dim,
                contract=DeployPolicyContract(
                    action_dim=layout.action_dim,
                    action_discrete_dim=layout.action_discrete_dim,
                    policy_io_layout_hash=layout.layout_hash,
                    hand_layout=None,
                ),
            ),
        )
        policy = OpenPIClientPolicy(
            OpenPIClientPolicyConfig(policy_io_layout=layout),
            remote_policy=remote,
        )
        base = _observation()
        observation = DeploymentObservation(
            timestamp=base.timestamp,
            step_index=base.step_index,
            active_sides=base.active_sides,
            arms_enabled=base.arms_enabled,
            hands_enabled=base.hands_enabled,
            images=base.images,
            state=base.state,
            state_valid=base.state_valid,
            hand_states={"right": {"mode": "open"}},
            metadata=base.metadata,
        )

        actions = policy.predict(observation)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].metadata["end_effector_enabled_sides"], ["right"])
        self.assertEqual(transport.requests[0]["state_discrete"].dtype, np.int64)
        self.assertEqual(transport.requests[0]["state_discrete"].tolist(), [0])

    def test_openpi_masked_state_artifact_requires_mode_and_sends_all_available_mask(self) -> None:
        token = DiscreteTokenSpec(
            "mode",
            (
                DiscreteCategory(0, "open", backend_value=0.0),
                DiscreteCategory(1, "close", backend_value=0.9),
            ),
            source_metadata_key="mode",
        )
        layout = PolicyIOLayout(
            name="openpi_masked_right_discrete",
            end_effectors={
                "left": EndEffectorSpec.none("left"),
                "right": EndEffectorSpec.discrete(
                    "right",
                    adapter="two_state_gripper_v1",
                    target_tokens=(token,),
                    target_legacy_q6_slot_indices=(0,),
                ),
            },
        )
        artifact = PolicyArtifactContract(
            policy_io_layout=layout,
            model_id="masked-state-model",
            state_input_mode=STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT,
        )
        transport = _V22FakeTransport(layout)
        transport._metadata["deploy_schema_version"] = "dexteleop_deploy_v2"
        transport._metadata["model_id"] = "masked-state-model"
        transport._metadata["local_robot"].update(
            {
                "input_state_key": "state",
                "input_state_dim": layout.state_dim * 2,
                "input_state_mode": STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT,
                "input_state_discrete_key": "state_discrete",
                "input_state_discrete_dim": layout.state_discrete_dim,
            }
        )
        remote = RemoteModelPolicy(
            transport=transport,
            observation_adapter=lambda observation: deployment_observation_to_openpi(observation),
            config=RemotePolicyConfig(
                action_dim=layout.action_dim,
                action_discrete_key="actions_discrete",
                action_discrete_dim=layout.action_discrete_dim,
            ),
        )
        policy = OpenPIClientPolicy(
            OpenPIClientPolicyConfig(policy_artifact=artifact),
            remote_policy=remote,
        )
        base = _observation()
        observation = DeploymentObservation(
            timestamp=base.timestamp,
            step_index=base.step_index,
            active_sides=base.active_sides,
            arms_enabled=base.arms_enabled,
            hands_enabled=base.hands_enabled,
            images=base.images,
            state=base.state,
            state_valid=base.state_valid,
            hand_states={"right": {"mode": "open"}},
            metadata=base.metadata,
        )

        policy.predict(observation)

        request = transport.requests[0]
        self.assertEqual(request["state"].shape, (layout.state_dim * 2,))
        np.testing.assert_array_equal(request["state"][layout.state_dim :], np.ones((layout.state_dim,)))

    def test_teleop_normalizer_accepts_unified_action_chunk_metadata(self) -> None:
        chunk = UnifiedActionChunk(
            actions=np.zeros((2, ACTION_DIM), dtype=np.float32),
            metadata={"source": "openpi_remote_policy", "remote_policy": {"action_key": "actions"}},
        )

        actions = normalize_policy_output(chunk, _observation())

        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0].metadata["source"], "openpi_remote_policy")
        self.assertEqual(actions[0].metadata["remote_policy"]["action_key"], "actions")
        self.assertIn("arm_action_specs_by_side", actions[0].metadata)

    def test_openpi_real_launcher_expands_to_generic_policy_spec(self) -> None:
        with mock.patch("teleop_runtime.deploy.run_policy.main", return_value=7) as run_main:
            result = openpi_real_launcher.main(
                [
                    "--host",
                    "127.0.0.2",
                    "--port",
                    "9000",
                    "--active-sides",
                    "both",
                    "--arm-control-mode",
                    "position",
                    "--position-reference-hz",
                    "45",
                    "--arm-control-hz",
                    "150",
                    "--arm-canfd-follow",
                    "--position-collision-check",
                ]
            )

        self.assertEqual(result, 7)
        launched_args = run_main.call_args.args[0]
        self.assertIn("deployments.openpi.client.policy:create_policy", launched_args)
        kwargs_json = launched_args[launched_args.index("--policy-kwargs-json") + 1]
        kwargs = json.loads(kwargs_json)
        self.assertEqual(kwargs["host"], "127.0.0.2")
        self.assertEqual(kwargs["port"], 9000)
        self.assertNotIn("expected_model_id", kwargs)
        self.assertNotIn("expected_checkpoint_step", kwargs)
        self.assertIn("--active-sides", launched_args)
        self.assertEqual(
            launched_args[launched_args.index("--arm-control-mode") + 1],
            "position",
        )
        self.assertEqual(
            launched_args[launched_args.index("--position-reference-hz") + 1],
            "45",
        )
        self.assertEqual(
            launched_args[launched_args.index("--arm-control-hz") + 1],
            "150",
        )
        self.assertIn("--arm-canfd-follow", launched_args)
        self.assertIn("--position-collision-check", launched_args)

    def test_runtime_keyframe_period_changes_only_for_explicit_rtc_profile(self) -> None:
        legacy_period = run_policy._runtime_action_keyframe_period_sec(
            SimpleNamespace(rtc_profile="off")
        )
        raw_rtc_period = run_policy._runtime_action_keyframe_period_sec(
            SimpleNamespace(rtc_profile=RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1)
        )
        model_rtc_period = run_policy._runtime_action_keyframe_period_sec(
            SimpleNamespace(
                rtc_profile=MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1
            )
        )

        self.assertEqual(legacy_period, 1.0 / 15.0)
        self.assertEqual(raw_rtc_period, 1.0 / 30.0)
        self.assertEqual(model_rtc_period, 1.0 / 30.0)

    def test_openpi_real_launcher_passes_explicit_wrist_image_map(self) -> None:
        with mock.patch("teleop_runtime.deploy.run_policy.main", return_value=0) as run_main:
            result = openpi_real_launcher.main(
                [
                    "--image-key-map-json",
                    '{"left_wrist":"observation.images.base_0_rgb"}',
                    "--required-image-key",
                    "left_wrist",
                    "--required-camera-slot",
                    "base_0_rgb",
                ]
            )

        self.assertEqual(result, 0)
        launched_args = run_main.call_args.args[0]
        kwargs_json = launched_args[launched_args.index("--policy-kwargs-json") + 1]
        kwargs = json.loads(kwargs_json)
        self.assertEqual(
            kwargs["image_key_map"],
            {"left_wrist": "observation.images.base_0_rgb"},
        )
        self.assertEqual(kwargs["required_image_keys"], ["left_wrist"])

    def test_openpi_real_launcher_wires_raw_rtc_only_when_explicit(self) -> None:
        artifact = PolicyArtifactContract(
            policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
            model_id="raw-rtc-openpi-unit",
            checkpoint_step=123,
            camera_input=PolicyCameraInputContract(
                inputs=(
                    PolicyCameraInput(
                        source_key="front",
                        model_input_key="observation.images.base_0_rgb",
                        server_camera_slot="base_0_rgb",
                    ),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = f"{tmp_dir}/policy_artifact.json"
            with open(artifact_path, "w", encoding="utf-8") as stream:
                json.dump(artifact.to_dict(), stream)
            with mock.patch("teleop_runtime.deploy.run_policy.main", return_value=0) as run_main:
                result = openpi_real_launcher.main(
                    [
                        "--policy-artifact-json",
                        artifact_path,
                        "--rtc-profile",
                        RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
                        "--rtc-normalization-hash",
                        "normalization:openpi-unit-v1",
                        "--rtc-frame-contract-hash",
                        "frame:openpi-unit-v1",
                    ]
                )

        self.assertEqual(result, 0)
        delegated = run_main.call_args.args[0]
        self.assertIn("--rtc-profile", delegated)
        self.assertIn(RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1, delegated)
        self.assertNotIn("--policy-prediction-mode", delegated)
        factory = run_main.call_args.kwargs["rtc_model_adapter_factory"]
        self.assertTrue(callable(factory))
        creator = mock.Mock(return_value="absolute-adapter")
        result = factory(
            policy=SimpleNamespace(create_raw_rtc_adapter=creator),
            deploy_session_contract=SimpleNamespace(
                runtime_pose_frame_contract=None,
            ),
            policy_artifact=artifact,
            args=SimpleNamespace(policy_action_pose_mode="absolute"),
            active_sides=("left", "right"),
        )
        self.assertEqual(result, "absolute-adapter")
        self.assertEqual(
            creator.call_args.kwargs["result_action_representation"],
            "legacy_normalized_v1",
        )
        self.assertEqual(creator.call_args.kwargs["action_pose_mode"], "absolute")

        loaded_policy = LoadedPolicy(
            policy=SimpleNamespace(create_raw_rtc_adapter=creator),
            name="framework-policy:test",
        )
        creator.reset_mock(return_value=True)
        creator.return_value = "wrapped-adapter"
        wrapped_result = factory(
            policy=loaded_policy,
            deploy_session_contract=SimpleNamespace(
                runtime_pose_frame_contract=None,
            ),
            policy_artifact=artifact,
            args=SimpleNamespace(policy_action_pose_mode="absolute"),
            active_sides=("left", "right"),
        )
        self.assertEqual(wrapped_result, "wrapped-adapter")
        self.assertEqual(creator.call_count, 1)

    def test_openpi_real_launcher_forwards_model_rtc_through_loaded_policy(
        self,
    ) -> None:
        layout = PolicyIOLayout.legacy_v21_q6()
        legacy_continuation = ModelContinuationDataContract(
            continuation_mode=MODEL_CONTINUATION_FLOW_INPAINT_V1,
            continuation_implementation_id="fixture-flow-inpaint-v1",
            continuation_implementation_parameters_hash="sha256:" + "a" * 64,
            prediction_horizon_steps=16,
            minimum_forecast_delay_steps=1,
            maximum_forecast_delay_steps=12,
            minimum_prefix_steps=1,
            maximum_prefix_steps=1,
            model_action_dim=layout.action_dim,
            model_action_discrete_dim=layout.action_discrete_dim,
            component_scope=("left_arm", "right_arm"),
            supported_action_pose_modes=("absolute",),
            sampler_constraints=(MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,),
            training_prefix_source_kinds=(
                TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
            ),
            runtime_prefix_source_roles=(
                RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
            ),
            action_codec_hash="sha256:" + "b" * 64,
            layout_hash=layout.layout_hash,
            frame_contract_hash="frame:model-rtc-unit",
            normalization_hash="normalization:model-rtc-unit-v1",
            training_projection_implementation_id="fixture-teacher-v1",
            training_projection_parameters_hash="sha256:" + "c" * 64,
        )
        artifact = PolicyArtifactContract(
            policy_io_layout=layout,
            model_id="model-rtc-openpi-unit",
            checkpoint_step=123,
            camera_input=PolicyCameraInputContract(
                inputs=(
                    PolicyCameraInput(
                        source_key="front",
                        model_input_key="observation.images.base_0_rgb",
                        server_camera_slot="base_0_rgb",
                    ),
                ),
            ),
            model_continuation=ModelContinuationDataContractV2(
                conditioning_origin=MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
                runtime=ModelContinuationRuntimeContractV2.from_v1(
                    legacy_continuation
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = f"{tmp_dir}/policy_artifact.json"
            with open(artifact_path, "w", encoding="utf-8") as stream:
                json.dump(artifact.to_dict(), stream)
            with mock.patch(
                "teleop_runtime.deploy.run_policy.main",
                return_value=0,
            ) as run_main:
                result = openpi_real_launcher.main(
                    [
                        "--policy-artifact-json",
                        artifact_path,
                        "--rtc-profile",
                        MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
                        "--rtc-real-shadow",
                    ]
                )

        self.assertEqual(result, 0)
        factory = run_main.call_args.kwargs["rtc_model_adapter_factory"]
        creator = mock.Mock(return_value="model-adapter")
        loaded_policy = LoadedPolicy(
            policy=SimpleNamespace(create_model_rtc_adapter=creator),
            name="framework-policy:model-rtc-test",
        )
        adapter = factory(
            policy=loaded_policy,
            deploy_session_contract=SimpleNamespace(
                runtime_pose_frame_contract=SimpleNamespace(
                    contract_hash="frame:runtime-model-rtc-unit",
                ),
            ),
            policy_artifact=artifact,
            args=SimpleNamespace(policy_action_pose_mode="absolute"),
            active_sides=("left", "right"),
        )
        self.assertEqual(adapter, "model-adapter")
        self.assertEqual(creator.call_count, 1)
        self.assertEqual(
            creator.call_args.kwargs["frame_contract_hash"],
            "frame:model-rtc-unit",
        )
        self.assertEqual(
            creator.call_args.kwargs["normalization_hash"],
            "normalization:model-rtc-unit-v1",
        )

    def test_openpi_real_launcher_selects_typed_results_for_relative_artifact(
        self,
    ) -> None:
        frame_hash = "sha256:" + "3" * 64
        artifact = PolicyArtifactContract(
            policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
            model_id="raw-rtc-openpi-relative-unit",
            checkpoint_step=456,
            dataset_schema="dexteleop_dataset_v2.3",
            camera_input=PolicyCameraInputContract(
                inputs=(
                    PolicyCameraInput(
                        source_key="front",
                        model_input_key="observation.images.base_0_rgb",
                        server_camera_slot="base_0_rgb",
                    ),
                ),
            ),
            pose_action_chunk=PoseActionChunkContract(
                action_horizon=16,
                action_pose_mode="relative",
                last_action_pose_mode="relative",
            ),
            pose_training_data=PoseTrainingDataContract.v22_raw_identity(
                active_sides=("left", "right"),
            ),
            pose_frame_contract_hashes=(frame_hash,),
            metadata={
                "pose_representation": {
                    "action": "relative",
                    "last_action": "relative",
                }
            },
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = f"{tmp_dir}/policy_artifact.json"
            with open(artifact_path, "w", encoding="utf-8") as stream:
                json.dump(artifact.to_dict(), stream)
            with mock.patch(
                "teleop_runtime.deploy.run_policy.main",
                return_value=0,
            ) as run_main:
                result = openpi_real_launcher.main(
                    [
                        "--policy-artifact-json",
                        artifact_path,
                        "--rtc-profile",
                        RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
                        "--rtc-normalization-hash",
                        "normalization:openpi-relative-unit-v1",
                        "--rtc-frame-contract-hash",
                        frame_hash,
                    ]
                )

        self.assertEqual(result, 0)
        factory = run_main.call_args.kwargs["rtc_model_adapter_factory"]
        creator = mock.Mock(return_value="relative-adapter")
        adapter = factory(
            policy=SimpleNamespace(create_raw_rtc_adapter=creator),
            deploy_session_contract=SimpleNamespace(
                runtime_pose_frame_contract=None,
            ),
            policy_artifact=artifact,
            args=SimpleNamespace(policy_action_pose_mode="relative"),
            active_sides=("left", "right"),
        )
        self.assertEqual(adapter, "relative-adapter")
        self.assertEqual(
            creator.call_args.kwargs["result_action_representation"],
            "typed_canonical_v1",
        )
        self.assertEqual(creator.call_args.kwargs["action_pose_mode"], "relative")
        self.assertEqual(creator.call_args.kwargs["frame_contract_hash"], frame_hash)

    def test_openpi_real_launcher_forwards_policy_artifact_to_client_and_runtime(self) -> None:
        artifact = PolicyArtifactContract(
            policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
            model_id="0712-wrist-cube",
            checkpoint_step=92599,
            camera_input=PolicyCameraInputContract(
                inputs=(
                    PolicyCameraInput(
                        source_key="left_wrist",
                        model_input_key="observation.images.left_wrist",
                        server_camera_slot="left_wrist_0_rgb",
                    ),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_path = f"{tmp_dir}/policy_artifact.json"
            with open(artifact_path, "w", encoding="utf-8") as stream:
                json.dump(artifact.to_dict(), stream)
            with mock.patch("teleop_runtime.deploy.run_policy.main", return_value=0) as run_main:
                result = openpi_real_launcher.main(
                    ["--policy-artifact-json", artifact_path]
                )

        self.assertEqual(result, 0)
        launched_args = run_main.call_args.args[0]
        kwargs_json = launched_args[launched_args.index("--policy-kwargs-json") + 1]
        kwargs = json.loads(kwargs_json)
        self.assertEqual(kwargs["policy_artifact"], artifact.to_dict())
        self.assertEqual(
            launched_args[launched_args.index("--policy-artifact-json") + 1],
            artifact_path,
        )

    def test_canonical_deploy_cli_delegates_to_supported_run_policy_runtime(self) -> None:
        with mock.patch("teleop_runtime.deploy.run_policy.main", return_value=13) as run_main:
            result = deploy_cli.main(["--policy", "hold", "--max-steps", "1"])

        self.assertEqual(result, 13)
        run_main.assert_called_once_with(["--policy", "hold", "--max-steps", "1"])

    def test_openpi_real_launcher_preserves_tested_impedance_overlay(self) -> None:
        with mock.patch("teleop_runtime.deploy.run_policy.main", return_value=0) as run_main:
            result = openpi_real_launcher.main(
                [
                    "--host",
                    "127.0.0.2",
                    "--active-sides",
                    "both",
                    "--action-hz",
                    "15",
                    "--arm-control-mode",
                    "impedance",
                    "--arm-control-hz",
                    "40",
                    "--no-arm-canfd-follow",
                ]
            )

        self.assertEqual(result, 0)
        launched_args = run_main.call_args.args[0]
        self.assertEqual(
            launched_args[launched_args.index("--arm-control-mode") + 1],
            "impedance",
        )
        self.assertEqual(
            launched_args[launched_args.index("--arm-control-hz") + 1],
            "40",
        )
        self.assertIn("--no-arm-canfd-follow", launched_args)

    def test_openpi_launcher_rejects_retired_prediction_scheduler_args(self) -> None:
        with mock.patch(
            "teleop_runtime.deploy.run_policy.main", return_value=0
        ) as run_main:
            result = openpi_real_launcher.main([])

        self.assertEqual(result, 0)
        launched_args = run_main.call_args.args[0]
        self.assertNotIn("--policy-prediction-mode", launched_args)
        self.assertNotIn(
            "--policy-prediction-prefetch-queue-threshold",
            launched_args,
        )

        def validate_runtime_parser(argv, **_kwargs):
            parser = argparse.ArgumentParser()
            run_policy.add_args(parser)
            parser.parse_args(argv)
            return 0

        with mock.patch(
            "teleop_runtime.deploy.run_policy.main",
            side_effect=validate_runtime_parser,
        ) as run_main:
            with self.assertRaises(SystemExit) as raised:
                openpi_real_launcher.main(
                    [
                        "--policy-prediction-mode",
                        "threaded-prefetch",
                        "--policy-prediction-prefetch-queue-threshold",
                        "4",
                    ]
                )

        self.assertEqual(raised.exception.code, 2)
        run_main.assert_called_once()

    def test_openpi_profile_forwards_no_init_qpos_over_injected_dataset_default(self) -> None:
        resolved: list[tuple[bool, bool, object, object]] = []

        def validate_runtime(argv):
            parser = argparse.ArgumentParser()
            run_policy.add_args(parser)
            args = parser.parse_args(argv)
            run_policy.validate_args(args, raw_argv=argv)
            resolved.append(
                (
                    bool(args.no_init_qpos),
                    bool(args.policy_qpos_init_suppressed),
                    args.policy_qpos_init_dataset,
                    args.policy_qpos_init_parquet,
                )
            )
            return 0

        with mock.patch(
            "teleop_runtime.deploy.run_policy.main",
            side_effect=validate_runtime,
        ):
            result = openpi_real_launcher.main(
                [
                    "--deployment-profile",
                    "rm65_bimanual_dmtac",
                    "--no-init-qpos",
                    "--send-actions",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(resolved, [(True, True, None, None)])

    def test_openpi_launcher_reaches_shared_execution_profile_resolver(self) -> None:
        resolved: list[tuple[str, float, float, bool]] = []

        def validate_runtime(argv):
            parser = argparse.ArgumentParser()
            run_policy.add_args(parser)
            args = parser.parse_args(argv)
            run_policy.validate_args(args, raw_argv=argv)
            resolved.append(
                (
                    args.arm_control_mode,
                    float(args.position_reference_hz),
                    float(args.arm_control_hz),
                    bool(args.arm_canfd_follow),
                )
            )
            return 0

        with mock.patch(
            "teleop_runtime.deploy.run_policy.main",
            side_effect=validate_runtime,
        ):
            openpi_real_launcher.main(
                [
                    "--host",
                    "127.0.0.2",
                    "--active-sides",
                    "both",
                    "--arm-execution-profile",
                    "position",
                ]
            )
            openpi_real_launcher.main(
                [
                    "--host",
                    "127.0.0.2",
                    "--active-sides",
                    "both",
                    "--arm-execution-profile",
                    "impedance_40hz",
                ]
            )

        self.assertEqual(
            resolved,
            [
                ("position", 60.0, 120.0, True),
                ("impedance", 60.0, 40.0, False),
            ],
        )

    def test_openpi_real_launcher_rejects_removed_identity_args(self) -> None:
        with self.assertRaises(SystemExit):
            openpi_real_launcher.main(
                [
                    "--host",
                    "127.0.0.2",
                    "--port",
                    "9000",
                    "--model",
                    "",
                    "--checkpoint-step",
                    "",
                ]
            )

    def test_openpi_real_launcher_does_not_abbreviate_model_as_rtc_profile(self) -> None:
        with self.assertRaises(SystemExit):
            openpi_real_launcher.main(
                [
                    "--model",
                    "old-model",
                    "--checkpoint-step",
                    "150000",
                ]
            )

    def test_openpi_real_launcher_can_show_delegated_runtime_help(self) -> None:
        with mock.patch("teleop_runtime.deploy.run_policy.main", return_value=0) as run_main:
            result = openpi_real_launcher.main(["--runtime-help"])

        self.assertEqual(result, 0)
        run_main.assert_called_once_with(["--help"])

    def test_openpi_real_launcher_passes_debug_to_runtime(self) -> None:
        with mock.patch("teleop_runtime.deploy.run_policy.main", return_value=0) as run_main:
            result = openpi_real_launcher.main(
                [
                    "--host",
                    "127.0.0.2",
                    "--port",
                    "9000",
                    "--api-key",
                    "secret-token",
                    "--debug",
                    "--debug-dir",
                    "logs/rtc-debug-test",
                    "--debug-payload-mode",
                    "full",
                    "--rtc-debug-level",
                    "trace",
                    "--rtc-debug-queue-capacity",
                    "32",
                    "--max-steps",
                    "1",
                ]
            )

        self.assertEqual(result, 0)
        launched_args = run_main.call_args.args[0]
        kwargs_json = launched_args[launched_args.index("--policy-kwargs-json") + 1]
        kwargs = json.loads(kwargs_json)
        self.assertEqual(kwargs["api_key"], "secret-token")
        self.assertIn("--debug", launched_args)
        self.assertEqual(
            launched_args[launched_args.index("--debug-dir") + 1],
            "logs/rtc-debug-test",
        )
        self.assertEqual(
            launched_args[launched_args.index("--debug-payload-mode") + 1],
            "full",
        )
        self.assertEqual(
            launched_args[launched_args.index("--rtc-debug-level") + 1],
            "trace",
        )
        self.assertEqual(
            launched_args[
                launched_args.index("--rtc-debug-queue-capacity") + 1
            ],
            "32",
        )
        self.assertIn("--max-steps", launched_args)

    def test_openpi_real_launcher_defaults_debug_payload_to_compact(self) -> None:
        with mock.patch("teleop_runtime.deploy.run_policy.main", return_value=0) as run_main:
            result = openpi_real_launcher.main(["--debug", "--max-steps", "1"])

        self.assertEqual(result, 0)
        launched_args = run_main.call_args.args[0]
        self.assertEqual(
            launched_args[launched_args.index("--debug-payload-mode") + 1],
            "compact",
        )


if __name__ == "__main__":
    unittest.main()
