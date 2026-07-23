from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pytest

from deployments.client.policy import LoadedPolicy
from deployments.openpi.client.observation_adapter import (
    deployment_observation_to_openpi,
)
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.observation_view import PolicyObservationView
from robot_motion.deploy.runtime import PolicyDeployRuntime, PolicyDeployRuntimeConfig
from robot_motion.deploy.schema import ACTION_DIM, OBSERVATION_STATE_DIM
from teleop_runtime.deploy import run_policy


def _observation() -> DeploymentObservation:
    state = np.zeros(OBSERVATION_STATE_DIM, dtype=np.float32)
    state[25:32] = np.asarray([0.4, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    return DeploymentObservation(
        timestamp=4.0,
        step_index=3,
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        images={"front": np.zeros((2, 3, 3), dtype=np.uint8)},
        image_timestamps={"front": 3.95},
        image_ages_sec={"front": 0.05},
        state=state,
        state_dict={"right_arm.pose": state[25:32].copy()},
        state_valid=np.ones(OBSERVATION_STATE_DIM, dtype=bool),
        tactile=np.asarray([0.1, 0.2], dtype=np.float32),
        tactile_by_side={"right": np.asarray([0.1, 0.2], dtype=np.float32)},
        tactile_valid_by_side={"right": True},
        last_action=np.zeros(ACTION_DIM, dtype=np.float32),
        arm_states={
            "right": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": np.eye(4, dtype=np.float64),
                "backend_status": "must-not-cross",
            }
        },
        hand_states={
            "right": {
                "joint_pos": np.zeros(6, dtype=np.float32),
                "mode": 1,
                "backend_secret": "must-not-cross",
            }
        },
        faults=["must-not-cross"],
        metadata={
            "prompt": "pick up the block",
            "pose_frame_contract": {"schema": "test"},
            "selected_owner": "human",
            "dispatch_receipt": {"accepted": True},
            "writer_status": "must-not-cross",
        },
    )


class _CaptureObjectPolicy:
    def __init__(self) -> None:
        self.observation = None

    def predict(self, observation):
        self.observation = observation
        return {"action": observation.hold_action()}


def _runtime(*, mode: str = "strict") -> PolicyDeployRuntime:
    return PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
            observation_firewall_mode=mode,
        )
    )


def test_strict_runtime_uses_view_as_unique_source_and_sanitizes_legacy_adapter() -> None:
    capture = _CaptureObjectPolicy()
    policy = LoadedPolicy(policy=capture, name="capture")

    result = _runtime().predict_and_queue(policy, _observation(), now=4.0)

    assert result.accepted
    assert isinstance(result.actor_observation, PolicyObservationView)
    assert result.observation_parity is not None
    assert result.observation_parity.matched
    adapted = capture.observation
    assert isinstance(adapted, DeploymentObservation)
    assert adapted.faults == []
    assert set(adapted.metadata) == {"prompt", "pose_frame_contract"}
    assert "backend_status" not in adapted.arm_states["right"]
    assert "backend_secret" not in adapted.hand_states["right"]
    assert adapted.hand_states["right"]["mode"] == 1
    assert result.queue_len_before == 0
    assert result.queue_len_after == 1


def test_function_policy_receives_only_strict_model_input() -> None:
    captured: dict[str, object] = {}

    def predict(model_input):
        captured.update(model_input)
        return {"action": np.zeros(ACTION_DIM, dtype=np.float32)}

    policy = LoadedPolicy(
        policy=predict,
        name="function-policy",
        call_with_model_input=True,
    )
    result = _runtime().predict_and_queue(policy, _observation(), now=4.0)

    assert result.accepted
    assert "faults" not in captured
    assert "metadata" not in captured
    assert "selected_owner" not in captured
    assert set(captured) == set(result.actor_observation.model_input())


def test_policy_observation_preserves_image_alias_without_duplicate_copy() -> None:
    source_images = {
        "front": np.zeros((4, 5, 3), dtype=np.uint8),
        "left_wrist": np.ones((4, 5, 3), dtype=np.uint8),
        "right_wrist": np.full((4, 5, 3), 2, dtype=np.uint8),
    }
    observation = replace(
        _observation(),
        images=source_images,
        image_timestamps={key: 3.95 for key in source_images},
        image_ages_sec={key: 0.05 for key in source_images},
    )
    actor, parity = _runtime().prepare_actor_observation(observation)

    assert parity.matched
    assert actor.payload["image"] is actor.payload["images"]
    model_input = actor.model_input()
    assert model_input["image"] is model_input["images"]

    source_images["front"].fill(9)
    np.testing.assert_array_equal(
        actor.payload["images"]["front"],
        np.zeros((4, 5, 3), dtype=np.uint8),
    )
    model_input["images"]["front"].fill(7)
    np.testing.assert_array_equal(
        actor.payload["images"]["front"],
        np.zeros((4, 5, 3), dtype=np.uint8),
    )


def test_dual_validate_and_strict_have_identical_queue_consumption() -> None:
    strict = _runtime(mode="strict")
    dual = _runtime(mode="dual_validate")
    strict_policy = LoadedPolicy(policy=_CaptureObjectPolicy(), name="strict")
    dual_policy = LoadedPolicy(policy=_CaptureObjectPolicy(), name="dual")

    strict_result = strict.predict_and_queue(strict_policy, _observation(), now=4.0)
    dual_result = dual.predict_and_queue(dual_policy, _observation(), now=4.0)

    assert strict_result.queue_len_after == dual_result.queue_len_after == 1
    np.testing.assert_array_equal(strict.pop_action().vector, dual.pop_action().vector)
    assert strict.action_queue_length() == dual.action_queue_length() == 0


def test_openpi_debug_metadata_does_not_expose_faults() -> None:
    request = deployment_observation_to_openpi(_observation())

    assert "faults" not in request["dexteleop"]
    assert request["prompt"] == "pick up the block"


def test_authoritative_frontend_rejects_dual_validate_firewall() -> None:
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    args = parser.parse_args(
        [
            "--send-actions",
            "--policy-observation-firewall-mode",
            "dual-validate",
        ]
    )

    with pytest.raises(ValueError, match="requires.*strict"):
        run_policy.validate_args(args)


def test_firewall_manifest_records_compatibility_retirement_gate() -> None:
    manifest = _runtime().observation_firewall_manifest()

    assert manifest["mode"] == "strict"
    assert manifest["legacy_unrestricted_input"] is False
    assert manifest["activation_deadline"] == "before_m6_authoritative_gate"
    assert len(manifest["contract_hash"]) == 64
