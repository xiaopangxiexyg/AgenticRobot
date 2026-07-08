from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np

from deployments.client import (
    DeployPolicyContract,
    RemoteModelPolicy,
    RemotePolicyConfig,
    ServerMetadataError,
    UnifiedActionChunk,
    validate_server_metadata,
)
from deployments.openpi.client import (
    OPENPI_LOCAL_PROTOCOL_VERSION,
    OpenPIClientPolicy,
    OpenPIObservationAdapterConfig,
    deployment_observation_to_openpi,
)
from deployments.openpi.client import real as openpi_real_launcher
from teleop_runtime.deploy.actions import normalize_policy_output
from teleop_runtime.deploy.observation import DeploymentObservation
from teleop_runtime.deploy.schema import ACTION_DIM, OBSERVATION_STATE_DIM


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


class OpenPICommonDeployTest(unittest.TestCase):
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
                    "--model",
                    "dual_rm65_dmgripper_latest",
                    "--active-sides",
                    "both",
                ]
            )

        self.assertEqual(result, 7)
        launched_args = run_main.call_args.args[0]
        self.assertIn("deployments.openpi.client.policy:create_policy", launched_args)
        kwargs_json = launched_args[launched_args.index("--policy-kwargs-json") + 1]
        kwargs = json.loads(kwargs_json)
        self.assertEqual(kwargs["host"], "127.0.0.2")
        self.assertEqual(kwargs["port"], 9000)
        self.assertEqual(kwargs["expected_model_id"], "dual_rm65_dmgripper_latest")
        self.assertIn("--active-sides", launched_args)

    def test_openpi_real_launcher_ignores_empty_optional_contract_args(self) -> None:
        with mock.patch("teleop_runtime.deploy.run_policy.main", return_value=0) as run_main:
            result = openpi_real_launcher.main(
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

        self.assertEqual(result, 0)
        launched_args = run_main.call_args.args[0]
        kwargs_json = launched_args[launched_args.index("--policy-kwargs-json") + 1]
        kwargs = json.loads(kwargs_json)
        self.assertNotIn("expected_model_id", kwargs)
        self.assertNotIn("expected_checkpoint_step", kwargs)

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
        self.assertIn("--max-steps", launched_args)


if __name__ == "__main__":
    unittest.main()
