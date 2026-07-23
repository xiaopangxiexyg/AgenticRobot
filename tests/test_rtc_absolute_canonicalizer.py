from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from deployments.client import (
    FinalChunkModelOutput,
    FinalChunkRtcBackend,
    FinalChunkRtcBinding,
    ModelAgnosticRtcClient,
)
from robot_motion.action import EndEffectorTarget, RobotAction
from robot_motion.deploy.canonical_policy_action import (
    AbsolutePolicyActionChunkCanonicalizer,
    CanonicalPolicyAction,
    RelativePolicyActionChunkCanonicalizer,
    canonical_policy_robot_action_chunk_hash,
)
from robot_motion.deploy.action_representation import (
    ACTION_ANCHOR_POLICY_NONE_V1,
    ACTION_ANCHOR_POLICY_REQUEST_ACTUAL_TCP_FIXED_V1,
    PolicyActionRepresentationContext,
)
from robot_motion.deploy.chunk_plan import (
    empty_policy_action_plan,
    mark_policy_action_plan_in_flight,
    merge_policy_chunk_result,
)
from robot_motion.deploy.inference import (
    ASYNC_ALIGNED_CHUNK_MODE_V1,
    PolicyInferenceRequest,
)
from robot_motion.relative_action import convert_arm_pose_chunk


def _binding(horizon: int) -> FinalChunkRtcBinding:
    return FinalChunkRtcBinding(
        implementation_id="absolute-canonicalizer-unit-v1",
        implementation_parameters_hash="parameters:absolute-canonicalizer-unit-v1",
        provider_id="provider:absolute-canonicalizer-unit-v1",
        artifact_hash="artifact:absolute-canonicalizer-unit-v1",
        model_action_space="dual-arm-pose-left2-right-token-v1",
        model_action_dim=16,
        model_action_discrete_dim=1,
        action_pose_mode="absolute",
        component_scope=(
            "left.arm.pose",
            "right.arm.pose",
            "left.end_effector.target",
            "right.end_effector.target",
        ),
        action_codec_hash="codec:absolute-canonicalizer-unit-v1",
        frame_contract_hash="frame:model-to-canonical-desk-unit-v1",
        layout_hash="layout:absolute-canonicalizer-unit-v1",
        normalization_hash="normalization:absolute-canonicalizer-unit-v1",
        prediction_horizon_steps=horizon,
    )


def _source_chunk(horizon: int) -> tuple[np.ndarray, np.ndarray]:
    actions = np.zeros((horizon, 16), dtype=np.float32)
    actions[:, 6] = 1.0
    actions[:, 13] = 1.0
    for index in range(horizon):
        actions[index, 0] = np.float32(0.1 + 0.01 * index)
        actions[index, 7] = np.float32(-0.1 - 0.01 * index)
        actions[index, 14:16] = np.asarray(
            [0.2 + 0.01 * index, 0.3 + 0.01 * index],
            dtype=np.float32,
        )
    tokens = (np.arange(horizon, dtype=np.int64) % 2).reshape(horizon, 1)
    return actions, tokens


def _input_hash(value: Any) -> str:
    return canonical_content_hash(
        value,
        domain="dexteleop.test.absolute_canonicalizer_input.v1",
    )


def _plan():
    return empty_policy_action_plan(
        plan_id="plan:absolute-canonicalizer-unit",
        session_id="session:absolute-canonicalizer-unit",
        control_domain_id="domain:absolute-canonicalizer-unit",
        producer_id="producer:absolute-canonicalizer-unit",
        producer_epoch=1,
        policy_node_id="node:absolute-canonicalizer-unit",
        clock_domain_id="monotonic:absolute-canonicalizer-unit",
        cursor_at=10.0,
        created_at=10.0,
    )


def _request(
    backend: FinalChunkRtcBackend,
    *,
    valid_until: float = 11.0,
) -> PolicyInferenceRequest:
    spec = backend.request_spec
    plan = _plan()
    return PolicyInferenceRequest(
        request_id="request:absolute-canonicalizer-unit",
        inference_id="inference:absolute-canonicalizer-unit",
        rtc_mode=ASYNC_ALIGNED_CHUNK_MODE_V1,
        session_id=plan.session_id,
        control_domain_id=plan.control_domain_id,
        producer_id=plan.producer_id,
        producer_epoch=plan.producer_epoch,
        policy_node_id=plan.policy_node_id,
        artifact_hash=spec.artifact_hash,
        observation_cut_id="observation:absolute-canonicalizer-unit",
        observation_timestamp=9.9,
        observation_step_index=99,
        triggered_at=10.0,
        admitted_at=10.0,
        first_target_effective_at=10.1,
        clock_domain_id=plan.clock_domain_id,
        action_period_sec=0.1,
        prediction_horizon_steps=spec.prediction_horizon_steps,
        minimum_execution_horizon_steps=spec.minimum_execution_horizon_steps,
        source_plan_id=plan.plan_id,
        source_plan_generation=plan.generation,
        source_plan_hash=plan.semantic_content_hash,
        source_plan_cursor_at=plan.cursor_at,
        producer_plan_committed_through=plan.producer_committed_through,
        forecast_delay_steps=1,
        delay_estimator_spec_hash="delay:absolute-canonicalizer-unit",
        action_codec_hash=spec.action_codec_hash,
        frame_contract_hash=spec.frame_contract_hash,
        layout_hash=spec.layout_hash,
        normalization_hash=spec.normalization_hash,
        valid_until=valid_until,
    )


def _decode_to_canonical_frame(
    actions: np.ndarray,
    discrete: np.ndarray | None,
    request: PolicyInferenceRequest,
) -> tuple[RobotAction, ...]:
    assert discrete is not None
    decoded = []
    for index, row in enumerate(actions):
        left_pose = row[:7].copy()
        right_pose = row[7:14].copy()
        # Synthetic declared-frame conversion: model x -> canonical desk x.
        left_pose[0] += np.float32(0.5)
        right_pose[0] += np.float32(0.5)
        decoded.append(
            RobotAction(
                action_space="dexteleop_policy_io_v2.2",
                timestamp=request.observation_timestamp,
                arm_pose_by_side={"left": left_pose, "right": right_pose},
                end_effector_targets_by_side={
                    "left": EndEffectorTarget(
                        value_type="continuous",
                        values=row[14:16],
                        adapter="two_axis_hand_v1",
                        names=("spread", "close"),
                        metadata={"units": ("rad", "rad")},
                    ),
                    "right": EndEffectorTarget(
                        value_type="discrete",
                        values=discrete[index],
                        adapter="two_state_gripper_v1",
                        names=("mode",),
                        metadata={"units": ("token",)},
                    ),
                },
                frame_by_effector={
                    "left_arm": "canonical_desk",
                    "right_arm": "canonical_desk",
                },
                controlled_frame_by_effector={
                    "left_arm": "tool_tcp",
                    "right_arm": "tool_tcp",
                },
                source="absolute-canonicalizer-unit",
            )
        )
    return tuple(decoded)


def _bundle(horizon: int):
    actions, tokens = _source_chunk(horizon)
    clock_values = iter((10.01, 10.02, 10.03))
    backend = FinalChunkRtcBackend(
        binding=_binding(horizon),
        infer_model=lambda _value: FinalChunkModelOutput(
            actions=actions,
            discrete_actions=tokens,
            source_output_id=f"source-output:absolute:{horizon}",
        ),
        hash_model_input=_input_hash,
        clock=lambda: next(clock_values),
    )
    canonicalizer = AbsolutePolicyActionChunkCanonicalizer(
        decode_robot_actions=_decode_to_canonical_frame,
        conversion_contract_hash="conversion:model-to-canonical-desk-unit-v1",
    )
    client = ModelAgnosticRtcClient(
        backend=backend,
        decoder=canonicalizer,
        request_spec=backend.request_spec,
    )
    request = _request(backend)
    model_input = {"state": np.zeros((4,), dtype=np.float32)}
    bundle = client.infer(
        request,
        model_input=model_input,
        model_input_hash=_input_hash(model_input),
    )
    return backend, request, bundle, actions, tokens


@pytest.mark.parametrize("horizon", (2, 4, 8))
def test_absolute_canonicalizer_binds_source_and_canonical_chunks(
    horizon: int,
) -> None:
    _backend, request, bundle, source, tokens = _bundle(horizon)
    result = bundle.policy_chunk_result

    assert result.decoded_absolute_actions == ()
    assert len(result.canonical_actions) == horizon
    receipts = tuple(
        action.canonicalization_receipt for action in result.canonical_actions
    )
    assert {receipt.source_pose_mode for receipt in receipts} == {"absolute"}
    assert {receipt.observation_cut_id for receipt in receipts} == {
        request.observation_cut_id
    }
    assert len({receipt.source_chunk_hash for receipt in receipts}) == 1
    assert len({receipt.canonical_chunk_hash for receipt in receipts}) == 1
    assert tuple(receipt.source_index for receipt in receipts) == tuple(range(horizon))
    assert receipts[0].canonical_chunk_hash == canonical_policy_robot_action_chunk_hash(
        tuple(action.action for action in result.canonical_actions)
    )

    for index, canonical in enumerate(result.canonical_actions):
        np.testing.assert_allclose(
            canonical.action.arm_pose_by_side["left"][0],
            source[index, 0] + 0.5,
        )
        np.testing.assert_array_equal(
            canonical.action.end_effector_targets_by_side["left"].values,
            source[index, 14:16],
        )
        np.testing.assert_array_equal(
            canonical.action.end_effector_targets_by_side["right"].values,
            tokens[index],
        )
        assert canonical.action.arm_frame("left") == "canonical_desk"
        assert canonical.action.arm_controlled_frame("left") == "tool_tcp"


def test_absolute_canonicalizer_rejects_tampered_canonical_chunk_identity() -> None:
    _backend, _request_value, bundle, _source, _tokens = _bundle(2)
    result = bundle.policy_chunk_result
    tampered = tuple(
        CanonicalPolicyAction(
            action=value.action,
            canonicalization_receipt=replace(
                value.canonicalization_receipt,
                canonical_chunk_hash="canonical-chunk:tampered",
            ),
        )
        for value in result.canonical_actions
    )

    with pytest.raises(ValueError, match="chunk hash does not match"):
        replace(result, canonical_actions=tampered)


def test_late_typed_absolute_result_is_rejected_before_plan_admission() -> None:
    backend, request, bundle, _source, _tokens = _bundle(4)
    assert request.valid_until == 11.0
    in_flight = mark_policy_action_plan_in_flight(_plan(), request)

    outcome = merge_policy_chunk_result(
        in_flight,
        request,
        bundle.policy_chunk_result,
        admitted_at=11.1,
        actual_cursor_at=11.1,
    )

    assert not outcome.receipt.accepted
    assert outcome.receipt.reasons == ("request_deadline_expired",)
    assert outcome.plan.in_flight_request_id is None
    assert outcome.plan.keyframes == ()
    assert backend.request_spec.action_pose_mode == "absolute"


def _representation_context(
    *,
    left_x: float = 0.4,
    right_x: float = 0.6,
) -> PolicyActionRepresentationContext:
    return PolicyActionRepresentationContext(
        source_pose_mode="relative",
        observation_cut_id="observation:absolute-canonicalizer-unit",
        observation_contract_hash="observation-contract:unit-v1",
        observation_timestamp=9.9,
        observation_step_index=99,
        active_sides=("left", "right"),
        anchor_policy=ACTION_ANCHOR_POLICY_REQUEST_ACTUAL_TCP_FIXED_V1,
        arm_reference_frame_by_side={
            "left": "canonical_desk",
            "right": "canonical_desk",
        },
        arm_controlled_frame_by_side={
            "left": "tool_tcp",
            "right": "tool_tcp",
        },
        arm_anchor_pose_by_side={
            "left": np.asarray(
                [left_x, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
            "right": np.asarray(
                [right_x, -0.1, 0.3, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        },
    )


def _decode_relative_absolute_rows(
    actions: np.ndarray,
    discrete: np.ndarray | None,
    request: PolicyInferenceRequest,
) -> tuple[RobotAction, ...]:
    assert discrete is not None
    result = []
    for index, row in enumerate(actions):
        result.append(
            RobotAction(
                action_space="dexteleop_policy_io_v2.2",
                timestamp=request.observation_timestamp,
                arm_pose_by_side={"left": row[:7], "right": row[7:14]},
                end_effector_targets_by_side={
                    "left": EndEffectorTarget(
                        value_type="continuous",
                        values=row[14:16],
                        adapter="two_axis_hand_v1",
                        names=("spread", "close"),
                        metadata={"units": ("rad", "rad")},
                    ),
                    "right": EndEffectorTarget(
                        value_type="discrete",
                        values=discrete[index],
                        adapter="two_state_gripper_v1",
                        names=("mode",),
                        metadata={"units": ("token",)},
                    ),
                },
                frame_by_effector={
                    "left_arm": "canonical_desk",
                    "right_arm": "canonical_desk",
                },
                controlled_frame_by_effector={
                    "left_arm": "tool_tcp",
                    "right_arm": "tool_tcp",
                },
                source="relative-canonicalizer-unit",
            )
        )
    return tuple(result)


def _relative_bundle(
    context: PolicyActionRepresentationContext,
    *,
    source_actions: np.ndarray | None = None,
):
    horizon = 4
    desired_absolute, tokens = _source_chunk(horizon)
    relative = (
        convert_arm_pose_chunk(
            desired_absolute,
            anchors_by_side=context.arm_anchor_pose_by_side,
            source_mode="absolute",
            target_mode="relative",
            active_sides=context.active_sides,
        )
        if source_actions is None
        else np.asarray(source_actions, dtype=np.float32)
    )
    binding = replace(_binding(horizon), action_pose_mode="relative")
    clock_values = iter((10.01, 10.02, 10.03))
    backend = FinalChunkRtcBackend(
        binding=binding,
        infer_model=lambda _value: FinalChunkModelOutput(
            actions=relative,
            discrete_actions=tokens,
        ),
        hash_model_input=_input_hash,
        clock=lambda: next(clock_values),
    )
    canonicalizer = RelativePolicyActionChunkCanonicalizer(
        decode_absolute_robot_actions=_decode_relative_absolute_rows,
        conversion_contract_hash="conversion:fixed-request-anchor-unit-v1",
    )
    client = ModelAgnosticRtcClient(
        backend=backend,
        decoder=canonicalizer,
        request_spec=backend.request_spec,
    )
    request = replace(
        _request(backend),
        action_representation_context=context,
    )
    model_input = {"state": np.zeros((4,), dtype=np.float32)}
    bundle = client.infer(
        request,
        model_input=model_input,
        model_input_hash=_input_hash(model_input),
    )
    return request, bundle, desired_absolute, relative, tokens


def test_relative_context_is_explicit_for_absolute_and_fixed_for_relative() -> None:
    absolute = PolicyActionRepresentationContext(
        source_pose_mode="absolute",
        observation_cut_id="observation:context-unit",
        observation_contract_hash="observation-contract:unit-v1",
        observation_timestamp=9.9,
        observation_step_index=99,
        active_sides=("left", "right"),
        anchor_policy=ACTION_ANCHOR_POLICY_NONE_V1,
        arm_reference_frame_by_side={
            "left": "canonical_desk",
            "right": "canonical_desk",
        },
        arm_controlled_frame_by_side={
            "left": "tool_tcp",
            "right": "tool_tcp",
        },
        arm_anchor_pose_by_side={},
    )
    assert absolute.anchor_identity_hash is None

    with pytest.raises(ValueError, match="fixed request anchor policy"):
        replace(_representation_context(), anchor_policy=ACTION_ANCHOR_POLICY_NONE_V1)
    with pytest.raises(ValueError, match="explicit no-anchor"):
        replace(
            absolute,
            arm_anchor_pose_by_side={
                "left": np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
            },
        )


def test_relative_canonicalizer_uses_request_anchor_and_preserves_end_effectors() -> None:
    context = _representation_context()
    request, bundle, expected, relative, tokens = _relative_bundle(context)
    result = bundle.policy_chunk_result

    assert request.action_representation_context is context
    assert result.decoded_absolute_actions == ()
    assert len(result.canonical_actions) == 4
    for index, canonical in enumerate(result.canonical_actions):
        receipt = canonical.canonicalization_receipt
        assert receipt.source_pose_mode == "relative"
        assert receipt.anchor_identity_hash == context.semantic_content_hash
        np.testing.assert_allclose(
            canonical.action.arm_pose_by_side["left"],
            expected[index, :7],
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            canonical.action.end_effector_targets_by_side["left"].values,
            relative[index, 14:16],
        )
        np.testing.assert_array_equal(
            canonical.action.end_effector_targets_by_side["right"].values,
            tokens[index],
        )


def test_relative_same_source_chunk_cannot_be_reanchored_at_completion() -> None:
    original = _representation_context()
    moved = _representation_context(left_x=0.7, right_x=0.9)
    _request_a, bundle_a, _expected, relative, _tokens = _relative_bundle(original)
    _request_b, bundle_b, _expected_b, _relative_b, _tokens_b = _relative_bundle(
        moved,
        source_actions=relative,
    )

    assert (
        bundle_a.policy_chunk_result.canonical_actions[0].canonical_action_hash
        != bundle_b.policy_chunk_result.canonical_actions[0].canonical_action_hash
    )
    assert original.semantic_content_hash != moved.semantic_content_hash


def test_control_thread_rejects_relative_receipt_for_another_anchor() -> None:
    context = _representation_context()
    request, bundle, _expected, _relative, _tokens = _relative_bundle(context)
    tampered_actions = tuple(
        CanonicalPolicyAction(
            action=canonical.action,
            canonicalization_receipt=replace(
                canonical.canonicalization_receipt,
                anchor_identity_hash="anchor:wrong-observation-cut",
            ),
        )
        for canonical in bundle.policy_chunk_result.canonical_actions
    )
    tampered_result = replace(
        bundle.policy_chunk_result,
        canonical_actions=tampered_actions,
    )
    in_flight = mark_policy_action_plan_in_flight(_plan(), request)

    outcome = merge_policy_chunk_result(
        in_flight,
        request,
        tampered_result,
        admitted_at=10.04,
        actual_cursor_at=10.04,
    )

    assert not outcome.receipt.accepted
    assert outcome.receipt.reasons == (
        "representation_anchor_identity_mismatch",
    )
    assert outcome.plan.keyframes == ()


def test_relative_client_rejects_missing_context_before_model_call() -> None:
    calls: list[object] = []
    binding = replace(_binding(2), action_pose_mode="relative")
    backend = FinalChunkRtcBackend(
        binding=binding,
        infer_model=lambda value: calls.append(value),
        hash_model_input=_input_hash,
    )
    client = ModelAgnosticRtcClient(
        backend=backend,
        decoder=RelativePolicyActionChunkCanonicalizer(
            decode_absolute_robot_actions=_decode_relative_absolute_rows,
            conversion_contract_hash="conversion:fixed-request-anchor-unit-v1",
        ),
        request_spec=backend.request_spec,
    )

    with pytest.raises(
        RuntimeError,
        match="relative_representation_context_missing",
    ):
        client.infer(
            _request(backend),
            model_input={"state": np.zeros((4,), dtype=np.float32)},
            model_input_hash=_input_hash(
                {"state": np.zeros((4,), dtype=np.float32)}
            ),
        )
    assert calls == []


def test_request_rejects_representation_context_from_another_cut() -> None:
    backend = FinalChunkRtcBackend(
        binding=replace(_binding(2), action_pose_mode="relative"),
        infer_model=lambda _value: None,
        hash_model_input=_input_hash,
    )
    with pytest.raises(ValueError, match="observation cut differs"):
        replace(
            _request(backend),
            action_representation_context=replace(
                _representation_context(),
                observation_cut_id="observation:wrong-cut",
            ),
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("observation_timestamp", 9.8, "timestamp differs"),
        ("observation_step_index", 98, "step index differs"),
    ),
)
def test_request_rejects_representation_context_from_another_observation_fact(
    field_name: str,
    value: object,
    message: str,
) -> None:
    backend = FinalChunkRtcBackend(
        binding=replace(_binding(2), action_pose_mode="relative"),
        infer_model=lambda _value: None,
        hash_model_input=_input_hash,
    )
    context = replace(
        _representation_context(),
        **{field_name: value},
    )

    with pytest.raises(ValueError, match=message):
        replace(
            _request(backend),
            action_representation_context=context,
        )
