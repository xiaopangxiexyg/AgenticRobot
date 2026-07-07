from __future__ import annotations

import pytest

pytest.importorskip("openpi05_deployment", reason="legacy openpi05_deployment package was removed")

import unittest
from unittest import mock
import argparse

import numpy as np

from openpi05_deployment.action_adapter import adapt_openpi_result, compact_action_to_full_vector
from openpi05_deployment.contracts import resolve_embodiment_contract
from openpi05_deployment.dataset_source import DatasetCompareMetrics
from openpi05_deployment.errors import DatasetTrajectoryError, ObservationAdapterError, ServerMetadataError
from openpi05_deployment.metadata import parse_server_contract
from openpi05_deployment.observation_adapter import build_openpi_observation
try:
    from openpi05_deployment.policy import OpenPIChunkPolicy
except ModuleNotFoundError as exc:
    if exc.name == "openpi_client":
        raise unittest.SkipTest("openpi_client is not installed") from exc
    raise
from teleop_runtime.deploy.run_policy import add_args, validate_args
from teleop_runtime.deploy.observation import DeploymentObservation
from teleop_runtime.deploy.startup_init import StartupInitializer, resolve_startup_init_action
from teleop_runtime.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
    STATE_HAND_SLICES,
)


def _metadata(embodiment: str, *, tactile: bool) -> dict:
    contract = resolve_embodiment_contract(embodiment)
    tactile_tokens = len(contract.tactile_token_order) if tactile else 0
    tactile_kind = "vector" if tactile else "none"
    return {
        "protocol_version": "openpi.local_policy.websocket.v1",
        "deployment": {
            "config_name": f"{contract.name}_cfg",
            "checkpoint_dir": "/tmp/checkpoint/10000",
            "checkpoint_step": 10000,
            "default_prompt": "paper_fold",
        },
        "model": {
            "action_horizon": 16,
            "tactile_kind": tactile_kind,
            "tactile_tokens": tactile_tokens,
            "tactile_dim": 1062 if tactile else 0,
        },
        "local_robot": {
            "embodiment": contract.name,
            "state_dim": contract.state_dim,
            "output_action_dim": contract.output_action_dim,
            "active_sides": list(contract.active_sides),
            "tactile_kind": tactile_kind,
        },
    }


def _observation(
    *,
    active_sides: tuple[str, ...],
    hands_enabled: bool = True,
    tactile_valid_left: bool = True,
    tactile_valid_right: bool = True,
) -> DeploymentObservation:
    state = np.full((OBSERVATION_STATE_DIM,), np.nan, dtype=np.float32)
    valid = np.zeros((OBSERVATION_STATE_DIM,), dtype=bool)

    left_pose = np.array([0.11, 0.12, 0.13, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    right_pose = np.array([0.21, 0.22, 0.23, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    left_hand = np.array([0.31, 0.32, 0.33, 0.34, 0.35, 0.36], dtype=np.float32)
    right_hand = np.array([0.41, 0.42, 0.43, 0.44, 0.45, 0.46], dtype=np.float32)

    state[STATE_ARM_POSE_SLICES["left"]] = left_pose
    state[STATE_ARM_POSE_SLICES["right"]] = right_pose
    state[STATE_HAND_SLICES["left"]] = left_hand
    state[STATE_HAND_SLICES["right"]] = right_hand
    valid[STATE_ARM_POSE_SLICES["left"]] = True
    valid[STATE_ARM_POSE_SLICES["right"]] = True
    valid[STATE_HAND_SLICES["left"]] = True
    valid[STATE_HAND_SLICES["right"]] = True

    return DeploymentObservation(
        timestamp=1.0,
        step_index=7,
        active_sides=active_sides,
        arms_enabled=True,
        hands_enabled=hands_enabled,
        images={"front": np.zeros((4, 5, 3), dtype=np.uint8)},
        image_timestamps={"front": 1.0},
        state=state,
        state_dict={},
        state_valid=valid,
        tactile=np.zeros((0,), dtype=np.float32),
        tactile_by_side={
            "left": np.ones((1062,), dtype=np.float32),
            "right": np.full((1062,), 2.0, dtype=np.float32),
        },
        tactile_valid_by_side={
            "left": tactile_valid_left,
            "right": tactile_valid_right,
        },
        last_action=np.zeros((ACTION_DIM,), dtype=np.float32),
        arm_states={},
        hand_states={},
        faults=[],
        metadata={"send_actions": False},
    )


class OpenPI05DeploymentTest(unittest.TestCase):
    def test_run_policy_accepts_space_separated_workspace_bounds(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args(
            [
                "--active-sides",
                "left",
                "right",
                "--arm-workspace-min",
                "-0.60",
                "-0.80",
                "0.00",
                "--arm-workspace-max",
                "0.60",
                "0.80",
                "0.90",
            ]
        )

        active_sides = validate_args(args)

        self.assertEqual(active_sides, ("left", "right"))
        self.assertEqual(args.arm_workspace_min, (-0.60, -0.80, 0.00))
        self.assertEqual(args.arm_workspace_max, (0.60, 0.80, 0.90))

    def test_run_policy_accepts_comma_separated_workspace_bounds(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args(
            [
                "--arm-workspace-min=-0.60,-0.80,0.00",
                "--arm-workspace-max=0.60,0.80,0.90",
            ]
        )

        validate_args(args)

        self.assertEqual(args.arm_workspace_min, (-0.60, -0.80, 0.00))
        self.assertEqual(args.arm_workspace_max, (0.60, 0.80, 0.90))

    def test_policy_dataset_initial_state_default_is_true(self) -> None:
        with mock.patch.object(
            OpenPIChunkPolicy,
            "_ensure_client",
            autospec=True,
            return_value=(mock.Mock(), mock.Mock()),
        ):
            policy = OpenPIChunkPolicy(embodiment="realman_right_arm_right_hand")
            disabled = OpenPIChunkPolicy(
                embodiment="realman_right_arm_right_hand",
                use_dataset_initial_state=False,
            )
        self.assertTrue(policy.use_dataset_initial_state)
        self.assertFalse(disabled.use_dataset_initial_state)

    def test_parse_server_contract_rejects_wrong_protocol(self) -> None:
        metadata = _metadata("realman_dual_arm_hands", tactile=False)
        metadata["protocol_version"] = "wrong"
        with self.assertRaises(ServerMetadataError):
            parse_server_contract(metadata, host="127.0.0.1", port=8000)

    def test_build_openpi_observation_dual_tactile(self) -> None:
        contract = parse_server_contract(
            _metadata("realman_dual_arm_hands", tactile=True),
            host="127.0.0.1",
            port=8000,
        )
        adapted = build_openpi_observation(
            _observation(active_sides=("left", "right")),
            contract,
            camera_key="front",
            prompt="paper_fold",
        )
        payload = adapted.payload
        self.assertEqual(payload["observation/state"].shape, (26,))
        self.assertEqual(payload["observation/tactile"].shape, (2, 1062))
        self.assertTrue(np.all(payload["observation/tactile_mask"]))
        self.assertEqual(payload["prompt"], "paper_fold")

    def test_build_openpi_observation_right_non_tactile(self) -> None:
        contract = parse_server_contract(
            _metadata("realman_right_arm_right_hand", tactile=False),
            host="127.0.0.1",
            port=8000,
        )
        adapted = build_openpi_observation(
            _observation(active_sides=("right",)),
            contract,
            camera_key="front",
            prompt=None,
        )
        payload = adapted.payload
        self.assertEqual(payload["observation/state"].shape, (13,))
        self.assertNotIn("observation/tactile", payload)
        np.testing.assert_allclose(payload["observation/state"][:7], [0.21, 0.22, 0.23, 0.0, 0.0, 0.0, 1.0])

    def test_missing_tactile_can_be_relaxed_in_dry_run(self) -> None:
        contract = parse_server_contract(
            _metadata("realman_dual_arm_hands", tactile=True),
            host="127.0.0.1",
            port=8000,
        )
        adapted = build_openpi_observation(
            _observation(active_sides=("left", "right"), tactile_valid_left=False),
            contract,
            allow_missing_tactile=True,
        )
        self.assertEqual(adapted.warnings, ("missing_tactile:left",))
        self.assertFalse(bool(adapted.payload["observation/tactile_mask"][0]))
        self.assertTrue(bool(adapted.payload["observation/tactile_mask"][1]))
        self.assertAlmostEqual(float(np.sum(adapted.payload["observation/tactile"][0])), 0.0)

    def test_missing_tactile_is_fatal_in_strict_mode(self) -> None:
        contract = parse_server_contract(
            _metadata("realman_dual_arm_hands", tactile=True),
            host="127.0.0.1",
            port=8000,
        )
        with self.assertRaises(ObservationAdapterError):
            build_openpi_observation(
                _observation(active_sides=("left", "right"), tactile_valid_right=False),
                contract,
                allow_missing_tactile=False,
            )

    def test_adapt_openpi_result_dual(self) -> None:
        contract = parse_server_contract(
            _metadata("realman_dual_arm_hands", tactile=False),
            host="127.0.0.1",
            port=8000,
        )
        observation = _observation(active_sides=("left", "right"))
        actions = np.tile(np.arange(26, dtype=np.float32), (16, 1))
        chunk = adapt_openpi_result({"actions": actions}, observation, contract)
        self.assertEqual(len(chunk), 16)
        np.testing.assert_allclose(chunk[0].vector[:26], np.arange(26, dtype=np.float32))
        self.assertEqual(chunk[0].metadata["safety_overrides"]["max_arm_xyz_step_m"], 0.0)

    def test_adapt_openpi_result_right_fills_right_slices(self) -> None:
        contract = parse_server_contract(
            _metadata("realman_right_arm_right_hand", tactile=False),
            host="127.0.0.1",
            port=8000,
        )
        observation = _observation(active_sides=("right",))
        compact = np.tile(np.arange(13, dtype=np.float32), (16, 1))
        chunk = adapt_openpi_result({"actions": compact}, observation, contract)
        vector = chunk[0].vector
        np.testing.assert_allclose(vector[ACTION_ARM_POSE_SLICES["right"]], np.arange(7, dtype=np.float32))
        np.testing.assert_allclose(vector[ACTION_HAND_SLICES["right"]], np.arange(7, 13, dtype=np.float32))

    def test_policy_closes_client_on_strict_tactile_failure(self) -> None:
        class _StubClient:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        contract = parse_server_contract(
            _metadata("realman_dual_arm_hands", tactile=True),
            host="127.0.0.1",
            port=8000,
        )
        client = _StubClient()
        policy = OpenPIChunkPolicy.__new__(OpenPIChunkPolicy)
        policy.host = "127.0.0.1"
        policy.port = 8000
        policy.expected_embodiment = contract.embodiment.name
        policy.camera_key = "front"
        policy.prompt = None
        policy.api_key = None
        policy.wait_timeout_sec = 600.0
        policy.metadata_timeout_sec = 3.0
        policy.metadata_poll_interval_sec = 1.0
        policy.allow_missing_tactile_in_dry_run = True
        policy.disable_deploy_rate_limits = True
        policy.close_client_on_fatal_error = True
        policy._client = client
        policy._contract = contract
        policy._status_text = "openpi=init"
        policy._last_warning_text = ""
        policy._ensure_client = lambda: (client, contract)

        observation = _observation(
            active_sides=("left", "right"),
            tactile_valid_left=False,
        )
        observation.metadata["send_actions"] = True

        with self.assertRaises(ObservationAdapterError):
            policy.predict(observation)
        self.assertTrue(client.closed)

    def test_compact_action_to_full_vector_with_nan_seed(self) -> None:
        contract = parse_server_contract(
            _metadata("realman_right_arm_right_hand", tactile=False),
            host="127.0.0.1",
            port=8000,
        )
        full = compact_action_to_full_vector(
            np.arange(13, dtype=np.float32),
            contract,
            seed_vector=np.full((ACTION_DIM,), np.nan, dtype=np.float32),
        )
        np.testing.assert_allclose(full[ACTION_ARM_POSE_SLICES["right"]], np.arange(7, dtype=np.float32))
        np.testing.assert_allclose(full[ACTION_HAND_SLICES["right"]], np.arange(7, 13, dtype=np.float32))
        self.assertTrue(np.all(np.isnan(full[ACTION_ARM_POSE_SLICES["left"]])))

    def test_startup_init_prefers_policy_provided_action(self) -> None:
        class _Policy:
            startup_init_source = "policy:first_frame"

            def get_startup_init_action(self) -> np.ndarray:
                return np.ones((ACTION_DIM,), dtype=np.float32)

        class _Args:
            init_source = "auto"
            init_time_s = 5.0
            init_dataset = None
            init_parquet = None

        action, source = resolve_startup_init_action(_Args(), _Policy())
        self.assertEqual(source, "policy:first_frame")
        np.testing.assert_allclose(action, np.ones((ACTION_DIM,), dtype=np.float32))

    def test_dataset_compare_does_not_provide_startup_init_action(self) -> None:
        policy = OpenPIChunkPolicy.__new__(OpenPIChunkPolicy)
        policy.input_source = "dataset_compare"
        policy.use_dataset_initial_state = True
        policy._ensure_client = mock.Mock()
        self.assertIsNone(policy.get_startup_init_action())
        self.assertEqual(policy.startup_init_source, "disabled")
        self.assertTrue(policy.disable_auto_startup_init)
        policy._ensure_client.assert_not_called()

    def test_auto_startup_init_can_be_disabled_by_policy(self) -> None:
        class _Policy:
            disable_auto_startup_init = True

        class _Args:
            init_source = "auto"
            init_time_s = 5.0
            init_dataset = None
            init_parquet = None

        action, source = resolve_startup_init_action(_Args(), _Policy())
        self.assertIsNone(action)
        self.assertEqual(source, "disabled")

    def test_dataset_execute_can_provide_startup_init_action(self) -> None:
        action = np.ones((ACTION_DIM,), dtype=np.float32)

        class _DatasetSource:
            startup_init_source = "dataset_state:/tmp/fake:episode=0:frame=0"

            def get_startup_init_action(self) -> np.ndarray:
                return action

        policy = OpenPIChunkPolicy.__new__(OpenPIChunkPolicy)
        policy.input_source = "dataset_execute"
        policy.use_dataset_initial_state = True
        policy._dataset_source = _DatasetSource()
        policy._ensure_client = mock.Mock()

        np.testing.assert_allclose(policy.get_startup_init_action(), action)
        self.assertEqual(policy.startup_init_source, _DatasetSource.startup_init_source)
        self.assertEqual(policy._ensure_client.call_count, 2)

    def test_startup_init_reports_unreached_side(self) -> None:
        observation, target = self._startup_init_right_offset_observation()

        initializer = StartupInitializer(
            duration_s=0.01,
            source="dataset_state:/tmp/fake:episode=0:frame=0",
            target_action=target,
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
            arm_position_tolerance_m=0.01,
            arm_rotation_tolerance_deg=5.0,
        )
        initializer.reset()
        action, faults, ready = initializer.next_action(observation, now=1.0)
        self.assertIsNotNone(action)
        self.assertFalse(faults)
        action, faults, ready = initializer.next_action(observation, now=2.0)
        self.assertIsNotNone(action)
        self.assertFalse(faults)
        self.assertFalse(ready)
        self.assertIn("right:state:pos=20.0mm", initializer.status_text)

    def test_startup_init_prefers_state_pose_over_hardware_pose_diagnostic(self) -> None:
        observation, target = self._startup_init_right_offset_observation()
        observation.state[STATE_ARM_POSE_SLICES["right"]] = target[ACTION_ARM_POSE_SLICES["right"]]
        observation.arm_states["right"] = {
            "hardware_pose_6d": np.array([0.226, 0.20, 0.20, 0.0, 0.0, 0.3], dtype=np.float32),
            "tcp_pose_source": "hardware_qpos_fk",
        }

        initializer = StartupInitializer(
            duration_s=0.01,
            source="dataset_state:/tmp/fake:episode=0:frame=0",
            target_action=target,
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        )
        initializer.reset()
        initializer.next_action(observation, now=1.0)
        _, faults, ready = initializer.next_action(observation, now=2.0)

        self.assertFalse(faults)
        self.assertTrue(ready)

    def test_startup_init_default_tolerance_accepts_small_residual(self) -> None:
        observation, target = self._startup_init_right_offset_observation()

        initializer = StartupInitializer(
            duration_s=0.01,
            source="dataset_state:/tmp/fake:episode=0:frame=0",
            target_action=target,
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        )
        initializer.reset()
        initializer.next_action(observation, now=1.0)
        _, faults, ready = initializer.next_action(observation, now=2.0)
        self.assertFalse(faults)
        self.assertTrue(ready)
        self.assertEqual(initializer.wait_reason, "")

    def test_startup_init_default_tolerance_accepts_current_right_residual(self) -> None:
        observation, target = self._startup_init_right_offset_observation()
        observation.state[STATE_ARM_POSE_SLICES["right"]] = np.array(
            [0.2251, 0.20, 0.20, 0.0, 0.0, 0.144, 0.9896],
            dtype=np.float32,
        )
        observation.arm_states["right"] = {"tcp_pose_source": "hardware_qpos_fk"}

        initializer = StartupInitializer(
            duration_s=0.01,
            source="dataset_state:/tmp/fake:episode=0:frame=0",
            target_action=target,
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        )
        initializer.reset()
        initializer.next_action(observation, now=1.0)
        _, faults, ready = initializer.next_action(observation, now=2.0)

        self.assertFalse(faults)
        self.assertTrue(ready)

    def _startup_init_right_offset_observation(self) -> tuple[DeploymentObservation, np.ndarray]:
        target = np.zeros((ACTION_DIM,), dtype=np.float32)
        target[ACTION_ARM_POSE_SLICES["left"]] = np.array(
            [0.10, 0.10, 0.10, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        target[ACTION_ARM_POSE_SLICES["right"]] = np.array(
            [0.20, 0.20, 0.20, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        target[ACTION_HAND_SLICES["left"]] = np.full((6,), 0.5, dtype=np.float32)
        target[ACTION_HAND_SLICES["right"]] = np.full((6,), 0.5, dtype=np.float32)

        observation = _observation(active_sides=("left", "right"))
        observation.state[STATE_ARM_POSE_SLICES["left"]] = target[ACTION_ARM_POSE_SLICES["left"]]
        observation.state[STATE_ARM_POSE_SLICES["right"]] = np.array(
            [0.22, 0.20, 0.20, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        return observation, target

    def test_dataset_compare_mode_forbids_send_actions(self) -> None:
        contract = parse_server_contract(
            _metadata("realman_dual_arm_hands", tactile=False),
            host="127.0.0.1",
            port=8000,
        )

        class _StubClient:
            def infer(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                return {"actions": np.zeros((16, 26), dtype=np.float32)}

        class _StubDatasetSource:
            def next_step(self):
                return mock.Mock(
                    payload={"observation/image": np.zeros((4, 5, 3), dtype=np.uint8), "observation/state": np.zeros((26,), dtype=np.float32), "prompt": "task"},
                    compact_ground_truth_actions=np.zeros((16, 26), dtype=np.float32),
                    frame_index=0,
                    episode_index=12,
                    prompt="task",
                )

        policy = OpenPIChunkPolicy.__new__(OpenPIChunkPolicy)
        policy.host = "127.0.0.1"
        policy.port = 8000
        policy.expected_embodiment = contract.embodiment.name
        policy.camera_key = "front"
        policy.prompt = None
        policy.api_key = None
        policy.wait_timeout_sec = 600.0
        policy.metadata_timeout_sec = 3.0
        policy.metadata_poll_interval_sec = 1.0
        policy.allow_missing_tactile_in_dry_run = True
        policy.disable_deploy_rate_limits = True
        policy.close_client_on_fatal_error = True
        policy.input_source = "dataset_compare"
        policy.dataset_root = "/tmp/fake"
        policy.dataset_episode = 0
        policy.dataset_episode_selection = "specified"
        policy.dataset_start_frame = 0
        policy.dataset_camera = None
        policy.dataset_prompt = None
        policy.dataset_loop = False
        policy.dataset_seed = 0
        policy.dataset_stride = None
        policy.use_dataset_initial_state = True
        policy._client = _StubClient()
        policy._contract = contract
        policy._dataset_source = _StubDatasetSource()
        policy._status_text = "openpi=init"
        policy._last_warning_text = ""
        policy._ensure_client = lambda: (policy._client, contract)

        observation = _observation(active_sides=("left", "right"))
        observation.metadata["send_actions"] = True

        with self.assertRaises(ObservationAdapterError):
            policy.predict(observation)

    def test_dataset_execute_mode_advances_and_reports_metrics(self) -> None:
        contract = parse_server_contract(
            _metadata("realman_right_arm_right_hand", tactile=False),
            host="127.0.0.1",
            port=8000,
        )

        class _StubClient:
            def infer(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
                return {
                    "actions": np.ones((16, 13), dtype=np.float32),
                    "server_timing": {"infer_ms": 12.5},
                }

        class _StubDatasetSource:
            def __init__(self) -> None:
                self.advanced = None

            def next_step(self):
                return mock.Mock(
                    payload={"observation/image": np.zeros((4, 5, 3), dtype=np.uint8), "observation/state": np.zeros((13,), dtype=np.float32), "prompt": "task"},
                    compact_ground_truth_actions=np.zeros((16, 13), dtype=np.float32),
                    frame_index=5,
                    episode_index=3,
                    prompt="task",
                )

            def compare_prediction(self, predicted_actions, ground_truth_actions):
                return DatasetCompareMetrics(
                    mae=1.0,
                    rmse=1.0,
                    max_abs=1.0,
                    first_step_mae=1.0,
                    first_step_max_abs=1.0,
                )

            def advance(self, steps: int) -> None:
                self.advanced = steps

        policy = OpenPIChunkPolicy.__new__(OpenPIChunkPolicy)
        policy.host = "127.0.0.1"
        policy.port = 8000
        policy.expected_embodiment = contract.embodiment.name
        policy.camera_key = "front"
        policy.prompt = None
        policy.api_key = None
        policy.wait_timeout_sec = 600.0
        policy.metadata_timeout_sec = 3.0
        policy.metadata_poll_interval_sec = 1.0
        policy.allow_missing_tactile_in_dry_run = True
        policy.disable_deploy_rate_limits = True
        policy.close_client_on_fatal_error = True
        policy.input_source = "dataset_execute"
        policy.dataset_root = "/tmp/fake"
        policy.dataset_episode = 0
        policy.dataset_episode_selection = "specified"
        policy.dataset_start_frame = 0
        policy.dataset_camera = None
        policy.dataset_prompt = None
        policy.dataset_loop = False
        policy.dataset_seed = 0
        policy.dataset_stride = None
        policy.use_dataset_initial_state = True
        policy._client = _StubClient()
        policy._contract = contract
        policy._dataset_source = _StubDatasetSource()
        policy._status_text = "openpi=init"
        policy._last_warning_text = ""
        policy._ensure_client = lambda: (policy._client, contract)

        observation = _observation(active_sides=("right",))
        chunk = policy.predict(observation)
        self.assertEqual(len(chunk), 16)
        self.assertEqual(policy._dataset_source.advanced, 16)
        self.assertIn("openpi=ds_exec:ep=3,frame=5", policy.status_text)
        self.assertIn("infer_ms=12.5", policy.status_text)


if __name__ == "__main__":
    unittest.main()
