from __future__ import annotations

import numpy as np
import pytest

from robot_motion.action import (
    CanonicalActionSafetyAnchor,
    CanonicalKeyframeSafetyProfile,
    EndEffectorSafetyCapability,
    EndEffectorTarget,
    RobotAction,
)
from robot_motion.deploy import (
    ACTION_ARM_POSE_SLICES,
    DEPLOY_ACTION_DIM,
    POLICY_PLAN_KEYFRAME_MUTABLE,
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
    CanonicalPolicyActionPipeline,
    NormalizedAction,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    TimedPolicyActionKeyframe,
    build_canonical_policy_action,
    build_deployment_observation,
    nan_vector,
    policy_action_emission_payload,
)
from robot_motion.deploy.pose_adapter import (
    convert_normalized_action_chunk_to_relative,
)


class _Clock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return float(next(self._values))


class _Policy:
    name = "m2-test-policy"

    def __init__(self, output: np.ndarray) -> None:
        self.output = np.asarray(output, dtype=np.float32)

    def predict(self, _observation):
        return self.output.copy()


def _observation():
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.10, 0.20, 0.30]
    return build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        step_index=7,
        arm_states={
            "right": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
        },
        hand_states={},
        timestamp=10.0,
    )


def _absolute_action(*, x: float = 0.30) -> np.ndarray:
    action = nan_vector(DEPLOY_ACTION_DIM)
    action[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [x, 0.20, 0.30, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    return action


def _pipeline() -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=False,
                max_arm_xyz_step_m=0.05,
                max_arm_rot_step_deg=20.0,
                max_arm_joint_step_rad=0.2,
            )
        ),
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
    )


def test_pre_safety_adapter_reuses_legacy_mapping_without_applying_safety() -> None:
    observation = _observation()
    pipeline = _pipeline()
    raw = NormalizedAction(
        vector=_absolute_action(),
        metadata={"position_reference_sequence": 9},
    )

    adapted = pipeline.adapt_pre_safety(raw, observation, policy_name="test")
    legacy = pipeline.prepare(raw, observation, policy_name="test")

    assert adapted.source_validation.accepted is True
    assert adapted.representation_receipt.input_stage == "decoded_absolute_pre_safety"
    assert adapted.representation_receipt.output_stage == (
        "pre_safety_canonical_robot_action"
    )
    assert adapted.robot_action is not None
    np.testing.assert_allclose(
        adapted.robot_action.arm_pose_by_side["right"],
        _absolute_action()[ACTION_ARM_POSE_SLICES["right"]],
    )
    assert "position_reference_sequence" not in adapted.robot_action.metadata
    assert adapted.compatibility_robot_action is not None
    assert adapted.compatibility_robot_action.metadata["position_reference_sequence"] == 9

    # The existing owner still applies policy safety exactly once after the tap.
    assert legacy.robot_action is not None
    assert legacy.filtered.action.vector[ACTION_ARM_POSE_SLICES["right"]][0] == pytest.approx(
        0.15
    )
    assert legacy.robot_action.arm_pose_by_side["right"][0] == pytest.approx(0.15)


def test_one_queue_pop_creates_one_immutable_emission_and_peek_is_passive() -> None:
    observation = _observation()
    first = _absolute_action(x=0.12)
    second = _absolute_action(x=0.13)
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
        ),
        monotonic_clock=_Clock(10.001, 10.002, 10.003, 10.004),
    )
    prediction = runtime.predict_and_queue(
        _Policy(np.stack((first, second))),
        observation,
        now=10.0,
        action_keyframe_period_sec=0.025,
    )

    assert prediction.accepted is True
    assert prediction.chunk_lineage is not None
    assert runtime.action_queue_pop_count() == 0

    step0 = runtime.prepare_next_action(
        observation,
        preparer=_pipeline(),
        policy_name="test",
    )
    assert step0.emission_error is None
    assert step0.emission is not None
    assert step0.raw_lookahead_action is not None
    assert runtime.action_queue_pop_count() == 1
    assert step0.emission.queue_pop_count == 1
    assert (
        step0.emission.semantic_content_hash
        == "32008b4cd7a04ca1a90a0c5bd7dac99ad252a07179a0bb48dd53c92745efc499"
    )
    assert step0.emission.chunk_keyframe_index == 0
    assert step0.emission.produced_at == pytest.approx(10.002)
    assert step0.emission.emitted_at == pytest.approx(10.003)
    assert step0.emission.effective_from == pytest.approx(10.002)
    assert step0.emission.effective_until == pytest.approx(10.027)
    np.testing.assert_array_equal(step0.emission.decoded_absolute_action.vector, first)

    # Reading the immutable emission and lookahead never consumes another keyframe.
    _ = step0.emission.semantic_content_hash
    _ = step0.raw_lookahead_action.vector
    assert runtime.action_queue_pop_count() == 1
    with pytest.raises(ValueError):
        step0.emission.decoded_absolute_action.vector[0] = 0.0
    with pytest.raises(TypeError):
        step0.emission.decoded_absolute_action.metadata["new"] = "value"

    step1 = runtime.prepare_next_action(
        observation,
        preparer=_pipeline(),
        policy_name="test",
    )
    assert step1.emission_error is None
    assert step1.emission is not None
    assert step1.raw_lookahead_action is None
    assert runtime.action_queue_pop_count() == 2
    assert step1.emission.chunk_keyframe_index == 1
    assert step1.emission.effective_from == pytest.approx(10.027)
    assert step1.emission.effective_until == pytest.approx(10.052)


def test_relative_chunk_emission_preserves_decode_anchor_provenance() -> None:
    observation = _observation()
    absolute = NormalizedAction(vector=_absolute_action(x=0.18))
    relative = convert_normalized_action_chunk_to_relative(
        [absolute],
        observation,
        active_sides=("right",),
    )[0]
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
            policy_action_pose_mode="relative",
        ),
        monotonic_clock=_Clock(10.001, 10.002, 10.003),
    )

    prediction = runtime.predict_and_queue(
        _Policy(relative.vector),
        observation,
        now=10.0,
    )
    step = runtime.prepare_next_action(
        observation,
        preparer=_pipeline(),
        policy_name="relative-test",
    )

    assert prediction.accepted is True
    assert step.emission_error is None
    assert step.emission is not None
    receipt = step.emission.representation_receipt
    assert receipt is not None
    assert receipt.source_pose_mode == "relative"
    assert receipt.anchor_policy == "current_actual_tcp_fixed_for_chunk"
    assert receipt.anchor_sides == ("right",)
    np.testing.assert_allclose(
        step.emission.decoded_absolute_action.vector[
            ACTION_ARM_POSE_SLICES["right"]
        ],
        absolute.vector[ACTION_ARM_POSE_SLICES["right"]],
        atol=1e-6,
    )


def test_typed_timed_keyframe_emits_before_missing_safety_fails_closed() -> None:
    canonical = build_canonical_policy_action(
        RobotAction(
            action_space="bimanual_ee_pose_v1",
            timestamp=10.0,
            arm_pose_by_side={
                "right": _absolute_action()[ACTION_ARM_POSE_SLICES["right"]],
            },
            end_effector_targets_by_side={
                "right": EndEffectorTarget(
                    value_type="continuous",
                    values=np.asarray([0.4], dtype=np.float32),
                    adapter="parallel_gripper_position_v1",
                    names=("position",),
                    metadata={"units": ("normalized",)},
                )
            },
            source="typed-policy-test",
        ),
        source_pose_mode="absolute",
        observation_cut_id="observation:typed:0",
        source_action_hash="source:typed:0",
        action_codec_hash="codec:typed",
        frame_contract_hash="frame:typed",
        layout_hash="layout:typed",
        normalization_hash="normalization:typed",
        conversion_contract_hash="conversion:typed",
    )
    keyframe = TimedPolicyActionKeyframe(
        target_effective_at=10.0,
        effective_until=10.1,
        effective_clock_domain_id="monotonic:test",
        action_period_sec=0.1,
        decoded_absolute_action=None,
        request_id="request:typed:0",
        inference_id="inference:typed:0",
        policy_output_id="output:typed:0",
        chunk_id=1,
        chunk_keyframe_index=0,
        chunk_size=1,
        observation_cut_id="observation:typed:0",
        producer_epoch=1,
        policy_node_id="policy:typed",
        artifact_hash="artifact:typed",
        produced_at=9.9,
        schedule_spec_id="schedule:typed",
        schedule_spec_hash="schedule-hash:typed",
        admitted_plan_generation=1,
        state=POLICY_PLAN_KEYFRAME_MUTABLE,
        valid_until=10.1,
        canonical_action=canonical,
    )
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
        )
    )

    result = runtime.prepare_timed_action_keyframe(
        keyframe,
        _observation(),
        preparer=_pipeline(),
        policy_name="typed-test",
        plan_keyframe_count_before=1,
        plan_keyframe_count_after=0,
        plan_pop_count=1,
    )

    assert result.emission is not None
    assert result.emission.action_payload_kind == (
        "typed_canonical_policy_action_v1"
    )
    assert result.emission.decoded_absolute_action is None
    assert result.emission.canonical_action is canonical
    assert result.emission.pre_safety_robot_action is canonical.action
    assert result.emission.representation_receipt is None
    assert result.emission.representation_trace_id == (
        canonical.canonicalization_receipt.receipt_id
    )
    trace_payload = policy_action_emission_payload(result.emission)
    assert trace_payload["action_payload_kind"] == (
        "typed_canonical_policy_action_v1"
    )
    assert trace_payload["canonical_policy_action_hash"] == (
        canonical.semantic_content_hash
    )
    assert trace_payload["representation_trace_id"] == (
        canonical.canonicalization_receipt.receipt_id
    )
    assert result.prepared_action is None
    assert isinstance(result.emission_error, RuntimeError)
    assert "formal emission remains no-action" in str(result.emission_error)


def test_typed_timed_keyframe_uses_shared_native_safety_then_resolver() -> None:
    target = EndEffectorTarget(
        value_type="continuous",
        values=np.asarray([0.8], dtype=np.float32),
        adapter="fake_parallel_gripper_v1",
        names=("opening",),
        metadata={"units": ("normalized",)},
    )
    canonical = build_canonical_policy_action(
        RobotAction(
            action_space="bimanual_ee_pose_v1",
            timestamp=10.0,
            arm_pose_by_side={
                "right": _absolute_action()[ACTION_ARM_POSE_SLICES["right"]],
            },
            end_effector_targets_by_side={"right": target},
            source="typed-policy-test",
        ),
        source_pose_mode="absolute",
        observation_cut_id="observation:typed:safe",
        source_action_hash="source:typed:safe",
        action_codec_hash="codec:typed",
        frame_contract_hash="frame:typed",
        layout_hash="layout:typed",
        normalization_hash="normalization:typed",
        conversion_contract_hash="conversion:typed",
    )
    keyframe = TimedPolicyActionKeyframe(
        target_effective_at=10.0,
        effective_until=10.1,
        effective_clock_domain_id="monotonic:test",
        action_period_sec=0.1,
        decoded_absolute_action=None,
        request_id="request:typed:safe",
        inference_id="inference:typed:safe",
        policy_output_id="output:typed:safe",
        chunk_id=2,
        chunk_keyframe_index=0,
        chunk_size=1,
        observation_cut_id="observation:typed:safe",
        producer_epoch=1,
        policy_node_id="policy:typed",
        artifact_hash="artifact:typed",
        produced_at=9.9,
        schedule_spec_id="schedule:typed",
        schedule_spec_hash="schedule-hash:typed",
        admitted_plan_generation=1,
        state=POLICY_PLAN_KEYFRAME_MUTABLE,
        valid_until=10.1,
        canonical_action=canonical,
    )
    anchor_target = EndEffectorTarget(
        value_type="continuous",
        values=np.asarray([0.2], dtype=np.float32),
        adapter=target.adapter,
        names=target.names,
        metadata=target.metadata,
    )
    anchor = CanonicalActionSafetyAnchor(
        anchor_id="previous-safe-typed",
        anchor_kind="previous_safe_committed",
        action_space=canonical.action.action_space,
        arm_pose_by_side=canonical.action.arm_pose_by_side,
        end_effector_targets_by_side={"right": anchor_target},
        frame_by_side={"right": "right_arm_base"},
        observed_at=9.9,
    )
    preparer = CanonicalPolicyActionPipeline(
        safety_profile=CanonicalKeyframeSafetyProfile(
            profile_id="typed-native-fake-v1",
            end_effector_capabilities_by_side={
                "right": EndEffectorSafetyCapability(
                    adapter=target.adapter,
                    value_type="continuous",
                    names=target.names,
                    units=("normalized",),
                    lower=(0.0,),
                    upper=(1.0,),
                    max_step=(0.3,),
                )
            },
        ),
        anchor_provider=lambda _action, _observation, _target_at: anchor,
    )
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=True,
        ),
        monotonic_clock=_Clock(10.001),
    )

    result = runtime.prepare_timed_action_keyframe(
        keyframe,
        _observation(),
        preparer=preparer,
        policy_name="typed-safe-test",
        plan_keyframe_count_before=1,
        plan_keyframe_count_after=0,
        plan_pop_count=1,
    )

    assert result.emission is not None
    assert result.emission_error is None
    prepared = result.prepared_action
    assert prepared is not None
    assert prepared.safety_result.status == "modified"
    assert prepared.faults == ()
    assert prepared.robot_action is not None
    np.testing.assert_allclose(
        prepared.robot_action.end_effector_targets_by_side["right"].values,
        [0.5],
    )
    assert prepared.command is not None
    assert set(prepared.command.commands) == {"right_arm", "right_hand"}


def test_emission_tap_failure_does_not_change_legacy_preparation_or_pop_count() -> None:
    observation = _observation()
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
        ),
        monotonic_clock=_Clock(10.001, 10.002, float("nan")),
    )
    prediction = runtime.predict_and_queue(
        _Policy(_absolute_action(x=0.12)),
        observation,
        now=10.0,
    )

    step = runtime.prepare_next_action(
        observation,
        preparer=_pipeline(),
        policy_name="test",
    )

    assert prediction.accepted is True
    assert step.emission is None
    assert isinstance(step.emission_error, ValueError)
    assert step.prepared_action is not None
    assert step.prepared_action.command is not None
    assert runtime.action_queue_pop_count() == 1
    assert runtime.action_queue_length() == 0
