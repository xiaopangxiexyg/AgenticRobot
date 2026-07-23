from __future__ import annotations

import numpy as np
import pytest

from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.observation_view import (
    LastActionContract,
    PolicyObservationContract,
    PolicyObservationFieldSpec,
    build_policy_observation_view,
)


def _observation() -> DeploymentObservation:
    return DeploymentObservation(
        timestamp=12.5,
        step_index=7,
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        images={"front": np.arange(12, dtype=np.uint8).reshape(2, 2, 3)},
        image_timestamps={"front": 12.45},
        image_ages_sec={"front": 0.05},
        state=np.arange(50, dtype=np.float32),
        state_valid=np.ones(50, dtype=bool),
        tactile=np.asarray([0.1, 0.2], dtype=np.float32),
        last_action=np.arange(26, dtype=np.float32),
        faults=["control fault must remain private"],
        metadata={
            "selected_owner": "policy",
            "reward": 123.0,
            "dispatch_receipt": {"accepted": True},
            "writer_health": "private",
            "backend_health": "private",
        },
    )


def _contract() -> PolicyObservationContract:
    return PolicyObservationContract(
        contract_id="single-policy-v23-strict-view",
        fields=(
            PolicyObservationFieldSpec("images"),
            PolicyObservationFieldSpec("image_timestamps"),
            PolicyObservationFieldSpec("state", dtype="float32", shape=(50,)),
            PolicyObservationFieldSpec("state_valid", dtype="bool", shape=(50,)),
            PolicyObservationFieldSpec("tactile", dtype="float32", shape=(None,)),
            PolicyObservationFieldSpec("last_action", dtype="float32", shape=(26,)),
            PolicyObservationFieldSpec("timestamp"),
            PolicyObservationFieldSpec("active_sides"),
        ),
        last_action_contract=LastActionContract(
            stage="previous_safe_action",
            producer_scope="policy:main",
            pose_mode="absolute",
            anchor_contract_hash="v23-anchor",
        ),
    )


def test_actor_view_is_allowlisted_and_detached_from_runtime_state() -> None:
    observation = _observation()
    view = build_policy_observation_view(observation, contract=_contract())
    model_input = view.model_input()

    assert set(model_input) == {
        "images",
        "image_timestamps",
        "state",
        "state_valid",
        "tactile",
        "last_action",
        "timestamp",
        "active_sides",
    }
    assert "faults" not in model_input
    assert "metadata" not in model_input
    observation.state[0] = 999.0
    observation.images["front"][0, 0, 0] = 255
    assert view.payload["state"][0] == 0.0
    assert view.payload["images"]["front"][0, 0, 0] == 0
    model_input["state"][1] = 777.0
    assert view.payload["state"][1] == 1.0


def test_control_private_and_unbound_last_action_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="Control-private"):
        PolicyObservationContract(
            contract_id="bad-private",
            fields=(PolicyObservationFieldSpec("faults"),),
        )
    with pytest.raises(ValueError, match="LastActionContract"):
        PolicyObservationContract(
            contract_id="bad-last-action",
            fields=(PolicyObservationFieldSpec("last_action"),),
        )


def test_typed_view_rejects_layout_mismatch() -> None:
    contract = PolicyObservationContract(
        contract_id="bad-shape",
        fields=(PolicyObservationFieldSpec("state", dtype="float32", shape=(49,)),),
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        build_policy_observation_view(_observation(), contract=contract)
