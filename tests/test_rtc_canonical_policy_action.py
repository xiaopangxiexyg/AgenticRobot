from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import EndEffectorTarget, RobotAction
from robot_motion.deploy.action_adapter import DeployRobotActionAdapter
from robot_motion.deploy.actions import NormalizedAction
from robot_motion.deploy.canonical_policy_action import (
    CanonicalPolicyAction,
    PolicyActionCanonicalizationReceipt,
    build_canonical_policy_action,
)
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.pose_adapter import (
    convert_normalized_action_chunk_to_absolute,
    convert_normalized_action_chunk_to_relative,
)
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    OBSERVATION_STATE_DIM,
)


COMMON_RECEIPT_FIELDS = {
    "observation_cut_id": "observation:gold:0",
    "action_codec_hash": "codec:pose-v23",
    "frame_contract_hash": "frame:runtime-raw",
    "layout_hash": "layout:bimanual-pose-hand",
    "normalization_hash": "normalization:gold",
    "conversion_contract_hash": "conversion:fixed-anchor-v1",
}


def _pose(x: float, y: float) -> np.ndarray:
    return np.asarray([x, y, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _typed_robot_action() -> RobotAction:
    return RobotAction(
        action_space="bimanual_ee_pose_v1",
        timestamp=1.25,
        arm_pose_by_side={
            "left": _pose(0.4, 0.1),
            "right": _pose(0.6, -0.1),
        },
        end_effector_targets_by_side={
            "left": EndEffectorTarget(
                value_type="continuous",
                values=np.linspace(0.0, 1.0, 6, dtype=np.float32),
                adapter="inspire_hand_q6_v1",
            ),
            "right": EndEffectorTarget(
                value_type="continuous",
                values=np.asarray([0.35], dtype=np.float32),
                adapter="parallel_gripper_position_v1",
                names=("position",),
            ),
        },
        source="captured-sync-policy",
    )


def _build(
    action: RobotAction,
    *,
    source_pose_mode: str,
    source_action_hash: str,
    anchor_identity_hash: str | None = None,
):
    return build_canonical_policy_action(
        action,
        source_pose_mode=source_pose_mode,
        source_action_hash=source_action_hash,
        anchor_identity_hash=anchor_identity_hash,
        **COMMON_RECEIPT_FIELDS,
    )


def test_typed_canonical_policy_action_is_immutable_and_native_end_effector_safe() -> None:
    source = _typed_robot_action()
    canonical = _build(
        source,
        source_pose_mode="absolute",
        source_action_hash="source:absolute:0",
    )

    assert canonical.canonicalization_receipt.anchor_identity_hash is None
    assert canonical.canonicalization_receipt.receipt_id.startswith(
        "policy-action-canonicalization:"
    )
    assert not canonical.action.arm_pose_by_side["left"].flags.writeable
    assert not canonical.action.end_effector_targets_by_side["left"].values.flags.writeable
    assert canonical.semantic_content_hash == replace(canonical).semantic_content_hash

    with pytest.raises(ValueError, match="no lossless legacy 26D view"):
        canonical.to_legacy_normalized_action()


def test_legacy_view_is_derived_from_the_canonical_robot_action() -> None:
    action = RobotAction(
        action_space="bimanual_ee_pose_v1",
        timestamp=2.0,
        arm_pose_by_side={"right": _pose(0.5, -0.2)},
        hand_qpos_by_side={
            "right": np.linspace(0.0, 0.5, 6, dtype=np.float32),
        },
        source="captured-sync-policy",
    )
    canonical = _build(
        action,
        source_pose_mode="absolute",
        source_action_hash="source:absolute:legacy",
    )
    legacy = canonical.to_legacy_normalized_action()

    assert legacy.vector.shape == (ACTION_DIM,)
    np.testing.assert_allclose(
        legacy.vector[ACTION_ARM_POSE_SLICES["right"]],
        action.arm_pose_by_side["right"],
    )
    np.testing.assert_allclose(
        legacy.vector[ACTION_HAND_SLICES["right"]],
        action.hand_qpos_by_side["right"],
    )
    legacy.vector[:] = 0.0
    np.testing.assert_allclose(
        canonical.action.arm_pose_by_side["right"],
        action.arm_pose_by_side["right"],
    )


def test_canonicalization_receipt_rejects_ambiguous_anchor_and_payload_identity() -> None:
    action = _typed_robot_action()
    canonical = _build(
        action,
        source_pose_mode="relative",
        source_action_hash="source:relative:0",
        anchor_identity_hash="anchor:observation:gold:0",
    )
    assert canonical.canonicalization_receipt.source_pose_mode == "relative"

    with pytest.raises(ValueError, match="anchor_identity_hash"):
        _build(
            action,
            source_pose_mode="relative",
            source_action_hash="source:relative:missing-anchor",
        )
    with pytest.raises(ValueError, match="must not carry an anchor"):
        _build(
            action,
            source_pose_mode="absolute",
            source_action_hash="source:absolute:unexpected-anchor",
            anchor_identity_hash="anchor:unexpected",
        )
    with pytest.raises(ValueError, match="does not match"):
        CanonicalPolicyAction(
            action=action,
            canonicalization_receipt=replace(
                canonical.canonicalization_receipt,
                canonical_action_hash="canonical:tampered",
            ),
        )

    qpos_action = RobotAction(
        action_space="bimanual_qpos_v1",
        arm_qpos_by_side={"right": np.zeros((6,), dtype=np.float32)},
        source="direct-qpos-fixture",
    )
    receipt = replace(
        canonical.canonicalization_receipt,
        canonical_action_hash=canonical_content_hash(
            {
                "unused": "the qpos rejection occurs before this identity is accepted",
            },
            domain="test.rtc.canonical.qpos",
        ),
    )
    with pytest.raises(ValueError, match="arm qpos is test/debug only"):
        CanonicalPolicyAction(action=qpos_action, canonicalization_receipt=receipt)


def _gold_observation(*, left_anchor_x: float, right_anchor_x: float) -> DeploymentObservation:
    return DeploymentObservation(
        timestamp=8.0,
        step_index=80,
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        end_effector_sides=("left", "right"),
        state=np.full((OBSERVATION_STATE_DIM,), np.nan, dtype=np.float32),
        state_valid=np.zeros((OBSERVATION_STATE_DIM,), dtype=bool),
        arm_states={
            "left": {"actual_tcp_pose": _pose(left_anchor_x, 0.1)},
            "right": {"actual_tcp_pose": _pose(right_anchor_x, -0.1)},
        },
    )


def _captured_no_rtc_absolute_action() -> NormalizedAction:
    vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    vector[ACTION_ARM_POSE_SLICES["left"]] = _pose(0.45, 0.1)
    vector[ACTION_ARM_POSE_SLICES["right"]] = _pose(0.65, -0.1)
    vector[ACTION_HAND_SLICES["left"]] = np.linspace(
        0.0, 0.5, 6, dtype=np.float32
    )
    vector[ACTION_HAND_SLICES["right"]] = np.linspace(
        0.5, 1.0, 6, dtype=np.float32
    )
    return NormalizedAction(
        vector=vector,
        metadata={
            "source": "captured-sync-policy",
            "end_effector_enabled_sides": ("left", "right"),
        },
    )


def _adapt(action: NormalizedAction, observation: DeploymentObservation) -> RobotAction:
    result = DeployRobotActionAdapter(
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
    ).adapt(action, observation, policy_name="gold-policy")
    assert result.faults == ()
    assert result.robot_action is not None
    return result.robot_action


def test_gold_ladder_absolute_then_relative_raw_rtc_preserves_canonical_output() -> None:
    observation = _gold_observation(left_anchor_x=0.1, right_anchor_x=0.2)
    captured_absolute = _captured_no_rtc_absolute_action()
    source_absolute_hash = canonical_content_hash(
        captured_absolute.vector,
        domain="test.rtc.gold.absolute-source",
    )

    # G0: no-RTC synchronous absolute output is the only execution gold.
    gold_action = _adapt(captured_absolute, observation)
    g0 = _build(
        gold_action,
        source_pose_mode="absolute",
        source_action_hash=source_absolute_hash,
    )

    # G1: absolute raw-RTC replays the same captured final chunk.
    g1 = _build(
        _adapt(captured_absolute, observation),
        source_pose_mode="absolute",
        source_action_hash=source_absolute_hash,
    )
    assert g1.semantic_content_hash == g0.semantic_content_hash

    # G2/G3: only the source representation changes; both use the original cut.
    relative = convert_normalized_action_chunk_to_relative(
        (captured_absolute,),
        observation,
        active_sides=("left", "right"),
    )[0]
    relative_hash = canonical_content_hash(
        relative.vector,
        domain="test.rtc.gold.relative-source",
    )
    decoded = convert_normalized_action_chunk_to_absolute(
        (relative,),
        observation,
        source_mode="relative",
        active_sides=("left", "right"),
    )[0]
    anchor_hash = canonical_content_hash(
        {
            side: observation.arm_states[side]["actual_tcp_pose"]
            for side in ("left", "right")
        },
        domain="test.rtc.gold.anchor",
    )
    g2 = _build(
        _adapt(decoded, observation),
        source_pose_mode="relative",
        source_action_hash=relative_hash,
        anchor_identity_hash=anchor_hash,
    )
    g3 = _build(
        _adapt(decoded, observation),
        source_pose_mode="relative",
        source_action_hash=relative_hash,
        anchor_identity_hash=anchor_hash,
    )

    assert g2.semantic_content_hash == g3.semantic_content_hash
    assert g0.canonical_action_hash == g2.canonical_action_hash
    assert g0.semantic_content_hash != g2.semantic_content_hash
    for side in ("left", "right"):
        np.testing.assert_allclose(
            g0.action.arm_pose_by_side[side],
            g3.action.arm_pose_by_side[side],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            g0.action.hand_qpos_by_side[side],
            g3.action.hand_qpos_by_side[side],
            atol=1e-6,
        )

    # Negative control: decoding with completion-time measured TCP changes the target.
    moved_observation = _gold_observation(left_anchor_x=0.3, right_anchor_x=0.4)
    wrongly_reanchored = convert_normalized_action_chunk_to_absolute(
        (relative,),
        moved_observation,
        source_mode="relative",
        active_sides=("left", "right"),
    )[0]
    wrong_action = _adapt(wrongly_reanchored, moved_observation)
    wrong = _build(
        wrong_action,
        source_pose_mode="relative",
        source_action_hash=relative_hash,
        anchor_identity_hash="anchor:wrong-current",
    )
    assert wrong.canonical_action_hash != g0.canonical_action_hash
