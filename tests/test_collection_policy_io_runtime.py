from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
from types import SimpleNamespace

from teleop_runtime.backends.grippers import DMTacGripperBackend, DMTacGripperBackendConfig
from teleop_runtime.backends.hands import InspireTacHandBackend, InspireTacHandBackendConfig
from teleop_runtime.collection.policy_io_runtime import resolve_collection_policy_io_layout
from teleop_runtime.collection.sample_assembler import CollectorSample
from teleop_runtime.collection.schema_v22 import collector_sample_to_v22_row
from teleop_runtime.retargeters import (
    JointsToDexHandRetargeter,
    JointsToDexHandRetargeterConfig,
    ScalarToGripperRetargeter,
    ScalarToGripperRetargeterConfig,
)
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop


def _pose(offset: float) -> np.ndarray:
    return np.asarray([offset, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _dmtac_pair(side: str) -> tuple[DMTacGripperBackend, ScalarToGripperRetargeter]:
    backend = DMTacGripperBackend(
        DMTacGripperBackendConfig(
            side=side,
            connect_hardware=False,
            connect_tactile=False,
            max_action=0.95,
        )
    )
    retargeter = ScalarToGripperRetargeter(
        ScalarToGripperRetargeterConfig(
            side=side,
            input_channel_name=f"{side}_hand_q6",
            output_max=1.0,
        )
    )
    return backend, retargeter


def _strict_bimanual_dmtac_sample(*, timestamp: float = 10.0) -> CollectorSample:
    action = np.full((26,), np.nan, dtype=np.float32)
    action[:7] = _pose(0.1)
    action[7:14] = _pose(0.2)
    action[14] = 0.25
    action[20] = 0.75
    state = np.full((38,), np.nan, dtype=np.float32)
    state[:6] = np.arange(6, dtype=np.float32)
    state[6:13] = _pose(0.1)
    state[13:19] = np.arange(6, dtype=np.float32) + 10.0
    state[19:25] = np.arange(6, dtype=np.float32) + 20.0
    state[25:32] = _pose(0.2)
    return CollectorSample(
        index=0,
        episode_index=0,
        frame_index=0,
        timestamp=timestamp,
        session_state="RUNNING",
        control_hz=15.0,
        action=action,
        observation_state=state,
        observation_last_action=action.copy(),
        arm_state_source="causal_measured_hardware",
        active_sides=("left", "right"),
        hand_states={
            "left": {
                "gripper_measured_scalar": 0.20,
                "joint_timestamp": timestamp - 0.10,
            },
            "right": {
                "gripper_measured_scalar": 0.80,
                "joint_timestamp": timestamp - 0.10,
            },
        },
    )


def test_auto_resolver_selects_checked_in_bimanual_dmtac_preset() -> None:
    left_backend, left_retargeter = _dmtac_pair("left")
    right_backend, right_retargeter = _dmtac_pair("right")

    resolved = resolve_collection_policy_io_layout(
        "auto",
        active_sides=("left", "right"),
        hand_backends_by_side={"left": left_backend, "right": right_backend},
        hand_retargeters_by_side={"left": left_retargeter, "right": right_retargeter},
    )

    assert resolved.layout is not None
    assert resolved.provenance["mode"] == "auto_preset"
    assert resolved.provenance["preset"] == "dual_arm_bimanual_dmtac_gripper_v22.json"
    assert resolved.layout.action_dim == 16
    assert resolved.layout.state_dim == 16
    assert resolved.layout.end_effector("left").state_source_field == "gripper_measured_scalar"
    assert left_backend.config.max_action == 1.0


def test_auto_resolver_selects_checked_in_asymmetric_inspire_preset() -> None:
    backend = InspireTacHandBackend(
        InspireTacHandBackendConfig(
            side="right",
            hand_ip="0.0.0.0",
            connect_hardware=False,
            enable_telemetry=False,
        )
    )
    retargeter = JointsToDexHandRetargeter(
        JointsToDexHandRetargeterConfig(
            side="right",
            input_channel_name="right_hand_q6",
        )
    )

    resolved = resolve_collection_policy_io_layout(
        "auto",
        active_sides=("left", "right"),
        hand_backends_by_side={"left": None, "right": backend},
        hand_retargeters_by_side={"left": None, "right": retargeter},
    )

    assert resolved.layout is not None
    assert resolved.provenance["mode"] == "auto_preset"
    assert resolved.provenance["preset"] == "dual_arm_left_none_right_inspire_v22.json"


def test_strict_bimanual_dmtac_row_uses_measured_scalar_not_q6_fallback() -> None:
    left_backend, left_retargeter = _dmtac_pair("left")
    right_backend, right_retargeter = _dmtac_pair("right")
    layout = resolve_collection_policy_io_layout(
        "auto",
        active_sides=("left", "right"),
        hand_backends_by_side={"left": left_backend, "right": right_backend},
        hand_retargeters_by_side={"left": left_retargeter, "right": right_retargeter},
    ).layout
    assert layout is not None

    row, _ = collector_sample_to_v22_row(
        _strict_bimanual_dmtac_sample(),
        layout=layout,
        strict=True,
    )

    np.testing.assert_allclose(row["action"][-2:], [0.25, 0.75])
    np.testing.assert_allclose(row["observation.state"][-2:], [0.20, 0.80])


def test_strict_v22_rejects_missing_or_stale_measured_gripper_state() -> None:
    left_backend, left_retargeter = _dmtac_pair("left")
    right_backend, right_retargeter = _dmtac_pair("right")
    layout = resolve_collection_policy_io_layout(
        "auto",
        active_sides=("left", "right"),
        hand_backends_by_side={"left": left_backend, "right": right_backend},
        hand_retargeters_by_side={"left": left_retargeter, "right": right_retargeter},
    ).layout
    assert layout is not None

    missing = _strict_bimanual_dmtac_sample()
    missing.hand_states["right"].pop("gripper_measured_scalar")
    with pytest.raises(ValueError, match="required measured field"):
        collector_sample_to_v22_row(missing, layout=layout, strict=True)

    stale = _strict_bimanual_dmtac_sample()
    stale.hand_states["left"]["joint_timestamp"] = 1.0
    with pytest.raises(ValueError, match="stale"):
        collector_sample_to_v22_row(stale, layout=layout, strict=True)

    commanded_arm = replace(
        _strict_bimanual_dmtac_sample(),
        arm_state_source="last_commanded_action",
    )
    with pytest.raises(ValueError, match="causal measured arm feedback"):
        collector_sample_to_v22_row(commanded_arm, layout=layout, strict=True)


def test_explicit_layout_must_match_runtime_pair() -> None:
    left_backend, left_retargeter = _dmtac_pair("left")
    right_backend, right_retargeter = _dmtac_pair("right")

    with pytest.raises(ValueError, match="does not match"):
        resolve_collection_policy_io_layout(
            "teleop_runtime/configs/policy_io_layouts/dual_arm_left_none_right_inspire_v22.json",
            active_sides=("left", "right"),
            hand_backends_by_side={"left": left_backend, "right": right_backend},
            hand_retargeters_by_side={"left": left_retargeter, "right": right_retargeter},
        )


def test_real_teleop_side_backend_overrides_feed_dmtac_action_specs() -> None:
    args = SimpleNamespace(
        hand_backend="inspire",
        left_hand_backend="none",
        right_hand_backend="dmtac-gripper",
    )

    assert real_teleop._hand_backend_mode_for_side(args, "left") == "none"
    assert real_teleop._hand_backend_mode_for_side(args, "right") == "dmtac-gripper"
    specs = real_teleop._collection_hand_action_specs(args, ("left", "right"))
    assert specs is not None
    assert set(specs) == {"right"}
