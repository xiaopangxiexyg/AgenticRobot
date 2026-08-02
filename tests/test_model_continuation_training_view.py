from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dataset_contracts import (
    DEXTELEOP_DATASET_SCHEMA_V23,
    MODEL_CONTINUATION_FLOW_INPAINT_V1,
    MODEL_CONTINUATION_LOSS_ALL_VALID_V1,
    MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
    MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    POSE_CHUNK_TAIL_PAD_LAST,
    RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
    TRAINING_PREFIX_RECORDED_POLICY_OUTPUT_V1,
    TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    ModelContinuationDataContract,
    ModelContinuationDataContractV2,
    ModelContinuationRuntimeContractV2,
    ModelContinuationTrainingContractV2,
    PolicyArtifactContract,
    PoseActionChunkContract,
    PoseFrameContract,
    PoseSourceProfile,
    PoseTrainingDataContract,
)
from dataset_contracts.v22 import EndEffectorSpec, PolicyIOLayout
from dataset_interface import (
    TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1,
    TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1,
    ModelContinuationTrainingSamplingSpec,
    ModelContinuationTrainingWindow,
    ModelContinuationViewError,
    model_continuation_training_sample_from_episode_rows,
    model_continuation_training_sample_from_pose_chunk,
    model_continuation_training_window_from_episode_rows,
    pose_action_chunk_from_episode_rows,
    select_model_continuation_training_delay_prefix,
)


def _pose(x: float) -> list[float]:
    return [x, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]


def _layout() -> PolicyIOLayout:
    return PolicyIOLayout(
        name="model_continuation_training_view_test",
        end_effectors={
            "left": EndEffectorSpec.none("left"),
            "right": EndEffectorSpec.none("right"),
        },
    )


def _frame_contract() -> PoseFrameContract:
    return PoseFrameContract.v23_forced_raw(
        PoseSourceProfile(
            source_id="model_continuation_training_view_fixture_v1",
            source_family="test",
            source_description="fixture",
            topology="dual_arm",
            raw_to_desk_transform_by_side={
                "left": np.eye(4),
                "right": np.eye(4),
            },
        )
    )


def _row(index: int) -> dict[str, object]:
    layout = _layout()
    return {
        "action": _pose(0.2 + index) + _pose(0.4 + index),
        "observation.state": _pose(0.1 + index) + _pose(0.3 + index),
        "observation.last_action": _pose(0.15 + index) + _pose(0.35 + index),
        "observation.arm.left.actual_tcp_pose": _pose(0.01 * index),
        "observation.arm.right.actual_tcp_pose": _pose(0.1 + 0.01 * index),
        "action.discrete": [],
        "observation.state.discrete": [],
        "observation.last_action.discrete": [],
        "policy_io_layout_hash": layout.layout_hash,
    }


def _artifact(
    *,
    minimum_prefix_steps: int = 1,
    supports_unconditioned_bootstrap: bool = False,
    loss_mask_semantics: str | None = None,
    action_pose_mode: str = "absolute",
    training_sources: tuple[str, ...] = (
        TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    ),
    projection_id: str = TEACHER_OVERLAP_PROJECTION_IMPLEMENTATION_ID_V1,
) -> PolicyArtifactContract:
    layout = _layout()
    frame = _frame_contract()
    chunk = PoseActionChunkContract(
        action_horizon=4,
        action_pose_mode=action_pose_mode,
        tail_policy=POSE_CHUNK_TAIL_PAD_LAST,
    )
    continuation_kwargs: dict[str, object] = {}
    if loss_mask_semantics is not None:
        continuation_kwargs["loss_mask_semantics"] = loss_mask_semantics
    continuation = ModelContinuationDataContract(
        continuation_mode=MODEL_CONTINUATION_FLOW_INPAINT_V1,
        continuation_implementation_id="fixture-flow-inpaint-v1",
        continuation_implementation_parameters_hash="sha256:fixture-sampler",
        prediction_horizon_steps=4,
        minimum_forecast_delay_steps=1,
        maximum_forecast_delay_steps=2,
        minimum_prefix_steps=minimum_prefix_steps,
        maximum_prefix_steps=2,
        model_action_dim=layout.action_dim,
        model_action_discrete_dim=0,
        component_scope=("left_arm", "right_arm"),
        supported_action_pose_modes=(action_pose_mode,),
        sampler_constraints=(MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,),
        training_prefix_source_kinds=training_sources,
        runtime_prefix_source_roles=(
            RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
        ),
        action_codec_hash="sha256:fixture-codec",
        layout_hash=layout.layout_hash,
        frame_contract_hash=frame.contract_hash,
        normalization_hash="sha256:fixture-normalization",
        training_projection_implementation_id=projection_id,
        training_projection_parameters_hash=(
            TEACHER_OVERLAP_PROJECTION_PARAMETERS_HASH_V1
        ),
        supports_unconditioned_bootstrap=supports_unconditioned_bootstrap,
        **continuation_kwargs,
    )
    return PolicyArtifactContract(
        policy_io_layout=layout,
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        pose_action_chunk=chunk,
        pose_training_data=PoseTrainingDataContract.native_v23(),
        pose_frame_contract_hashes=(frame.contract_hash,),
        model_continuation=continuation,
    )


def _sample(
    *,
    artifact: PolicyArtifactContract | None = None,
    rows: int = 5,
    start_index: int = 0,
    delay: int = 2,
    prefix_steps: int = 2,
):
    return model_continuation_training_sample_from_episode_rows(
        [_row(index) for index in range(rows)],
        _artifact() if artifact is None else artifact,
        pose_frame_contract=_frame_contract(),
        source_episode_id="episode-7",
        start_index=start_index,
        forecast_delay_steps=delay,
        prefix_steps=prefix_steps,
    )


def _trained_artifact(
    *,
    artifact: PolicyArtifactContract | None = None,
) -> PolicyArtifactContract:
    base = _artifact() if artifact is None else artifact
    legacy_contract = base.model_continuation
    assert isinstance(legacy_contract, ModelContinuationDataContract)
    return replace(
        base,
        model_continuation=ModelContinuationDataContractV2(
            conditioning_origin=MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
            runtime=ModelContinuationRuntimeContractV2.from_v1(
                legacy_contract
            ),
            training=ModelContinuationTrainingContractV2.from_v1(
                legacy_contract
            ),
        ),
    )


def _sampling_spec(
    *,
    pairs: tuple[tuple[int, int], ...] = (
        (1, 1),
        (2, 1),
        (2, 2),
    ),
    salt: str = "trained-model-rtc-test-v1",
) -> ModelContinuationTrainingSamplingSpec:
    return ModelContinuationTrainingSamplingSpec(
        delay_prefix_pairs=pairs,
        selection_salt=salt,
    )


def test_teacher_view_is_deterministic_and_keeps_masks_distinct() -> None:
    first = _sample()
    second = _sample()

    np.testing.assert_array_equal(first.prefix_mask, [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(first.target_is_pad, [False] * 4)
    np.testing.assert_array_equal(first.loss_mask, [False, False, True, True])
    np.testing.assert_array_equal(
        first.conditioning_prefix[:2],
        first.target_action[:2],
    )
    np.testing.assert_array_equal(first.conditioning_prefix[2:], 0.0)
    assert first.sample_hash == second.sample_hash
    assert first.semantic_payload().keys() == second.semantic_payload().keys()

    copied = first.artifact_action_dict()
    assert set(copied) == {
        "continuation.prefix",
        "continuation.prefix_mask",
        "action",
        "action_is_pad",
        "action_loss_mask",
        "forecast_delay_steps",
        "training_prefix_source_kind",
        "model_continuation_data_contract_hash",
        "model_continuation_training_sample_hash",
    }
    copied["action"][0, 0] = -100.0
    assert first.target_action[0, 0] != -100.0


def test_tail_padding_stays_episode_local_and_out_of_prefix_and_loss() -> None:
    sample = _sample(rows=5, start_index=3, delay=1, prefix_steps=1)

    np.testing.assert_array_equal(
        sample.target_is_pad,
        [False, False, True, True],
    )
    np.testing.assert_array_equal(sample.prefix_mask, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(sample.loss_mask, [False, True, False, False])
    np.testing.assert_array_equal(
        sample.target_action[2:],
        np.repeat(sample.target_action[1:2], 2, axis=0),
    )
    assert sample.source_start_index == 3


@pytest.mark.parametrize(
    ("delay", "prefix_steps", "message"),
    (
        (0, 1, "forecast_delay_steps is outside"),
        (3, 2, "forecast_delay_steps is outside"),
        (1, 2, "prefix_steps cannot exceed"),
        (2, 3, "prefix_steps is outside"),
    ),
)
def test_projection_rejects_out_of_contract_delay_or_prefix(
    delay: int,
    prefix_steps: int,
    message: str,
) -> None:
    with pytest.raises(ModelContinuationViewError, match=message):
        _sample(delay=delay, prefix_steps=prefix_steps)


def test_projection_requires_a_real_post_delay_target_at_episode_tail() -> None:
    with pytest.raises(ModelContinuationViewError, match="no real executable"):
        _sample(rows=5, start_index=3, delay=2, prefix_steps=2)


def test_projection_rejects_nonfinite_artifact_actions() -> None:
    artifact = _artifact()
    chunk = pose_action_chunk_from_episode_rows(
        [_row(index) for index in range(4)],
        artifact.policy_io_layout,
        pose_frame_contract=_frame_contract(),
        chunk_contract=artifact.pose_action_chunk,
        start_index=0,
    )
    invalid = chunk.action.copy()
    invalid[1, 2] = np.inf
    invalid_chunk = replace(chunk, action=invalid)

    with pytest.raises(ModelContinuationViewError, match="NaN or infinity"):
        model_continuation_training_sample_from_pose_chunk(
            invalid_chunk,
            artifact,
            source_episode_id="episode-7",
            source_start_index=0,
            forecast_delay_steps=2,
            prefix_steps=2,
        )


def test_projection_fails_closed_for_relative_and_unproven_sources() -> None:
    with pytest.raises(ModelContinuationViewError, match="only absolute"):
        _sample(artifact=_artifact(action_pose_mode="relative"))

    recorded_artifact = _artifact(
        training_sources=(
            TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
            TRAINING_PREFIX_RECORDED_POLICY_OUTPUT_V1,
        )
    )
    with pytest.raises(ModelContinuationViewError, match="cannot derive"):
        model_continuation_training_sample_from_episode_rows(
            [_row(index) for index in range(5)],
            recorded_artifact,
            pose_frame_contract=_frame_contract(),
            source_episode_id="episode-7",
            start_index=0,
            forecast_delay_steps=2,
            prefix_steps=2,
            training_prefix_source_kind=(
                TRAINING_PREFIX_RECORDED_POLICY_OUTPUT_V1
            ),
        )


def test_projection_rejects_inference_only_v2_artifact() -> None:
    artifact = _artifact()
    legacy_contract = artifact.model_continuation
    assert isinstance(legacy_contract, ModelContinuationDataContract)
    inference_only = ModelContinuationDataContractV2(
        conditioning_origin=MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
        runtime=ModelContinuationRuntimeContractV2.from_v1(
            legacy_contract
        ),
    )

    with pytest.raises(
        ModelContinuationViewError,
        match="has no training projection",
    ):
        _sample(
            artifact=replace(
                artifact,
                model_continuation=inference_only,
            )
        )


def test_projection_accepts_trained_prefix_v2_artifact() -> None:
    artifact = _artifact()
    legacy_contract = artifact.model_continuation
    assert isinstance(legacy_contract, ModelContinuationDataContract)
    trained = ModelContinuationDataContractV2(
        conditioning_origin=MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
        runtime=ModelContinuationRuntimeContractV2.from_v1(
            legacy_contract
        ),
        training=ModelContinuationTrainingContractV2.from_v1(
            legacy_contract
        ),
    )
    sample = _sample(
        artifact=replace(
            artifact,
            model_continuation=trained,
        )
    )

    assert sample.continuation_contract_hash == trained.contract_hash
    assert sample.training_prefix_source_kind == (
        TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1
    )


def test_projection_binding_and_sample_masks_are_self_validating() -> None:
    with pytest.raises(ModelContinuationViewError, match="projection binding"):
        _sample(artifact=_artifact(projection_id="other-projection-v1"))

    sample = _sample()
    bad_mask = sample.prefix_mask.copy()
    bad_mask[3] = 1.0
    with pytest.raises(ModelContinuationViewError, match="prefix_mask"):
        replace(sample, prefix_mask=bad_mask)


def test_bootstrap_and_all_valid_loss_are_explicit_contract_choices() -> None:
    bootstrap = _sample(
        artifact=_artifact(
            minimum_prefix_steps=0,
            supports_unconditioned_bootstrap=True,
        ),
        delay=1,
        prefix_steps=0,
    )
    np.testing.assert_array_equal(bootstrap.prefix_mask, [0.0] * 4)
    np.testing.assert_array_equal(bootstrap.loss_mask, [True] * 4)

    all_valid = _sample(
        artifact=_artifact(
            loss_mask_semantics=MODEL_CONTINUATION_LOSS_ALL_VALID_V1,
        )
    )
    np.testing.assert_array_equal(all_valid.loss_mask, [True] * 4)


def test_trained_window_binds_observation_generation_and_executable_indices() -> None:
    rows = [_row(index) for index in range(6)]
    artifact = _trained_artifact()
    spec = _sampling_spec()

    first = model_continuation_training_window_from_episode_rows(
        rows,
        artifact,
        pose_frame_contract=_frame_contract(),
        source_episode_id="episode-trained-1",
        start_index=1,
        sampling_spec=spec,
        draw_index=3,
    )
    second = model_continuation_training_window_from_episode_rows(
        rows,
        artifact,
        pose_frame_contract=_frame_contract(),
        source_episode_id="episode-trained-1",
        start_index=1,
        sampling_spec=spec,
        draw_index=3,
    )

    assert isinstance(first, ModelContinuationTrainingWindow)
    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.sampling_spec_hash == spec.semantic_content_hash
    assert first.sampling_draw_index == 3
    binding = first.continuation.causal_window_binding_payload()
    assert binding["observation_source_index"] == 1
    assert binding["generation_start_source_index"] == 1
    assert binding["first_executable_source_index"] == (
        1 + first.continuation.forecast_delay_steps
    )
    assert binding["valid_action_stop_source_index_exclusive"] == 5

    payload = first.training_dict()
    np.testing.assert_array_equal(
        payload["observation.state"],
        first.pose_chunk.observation_state,
    )
    np.testing.assert_array_equal(
        payload["continuation.prefix"],
        first.continuation.conditioning_prefix,
    )
    assert payload["observation_source_index"] == 1
    assert payload["generation_start_source_index"] == 1
    assert payload["first_executable_source_index"] == (
        1 + first.continuation.forecast_delay_steps
    )
    assert payload["model_continuation_training_window_hash"] == (
        first.semantic_content_hash
    )


def test_sampling_spec_is_order_independent_and_tail_filters_safely() -> None:
    artifact = _trained_artifact()
    ordered = _sampling_spec()
    reordered = _sampling_spec(pairs=((2, 2), (1, 1), (2, 1)))
    assert ordered.semantic_content_hash == reordered.semantic_content_hash
    assert (
        ModelContinuationTrainingSamplingSpec.from_dict(ordered.to_dict())
        == ordered
    )
    assert (
        _sampling_spec(salt="other-trained-model-rtc-test-v1")
        .semantic_content_hash
        != ordered.semantic_content_hash
    )

    selection = select_model_continuation_training_delay_prefix(
        ordered,
        artifact,
        source_episode_id="episode-tail",
        source_start_index=3,
        draw_index=0,
        valid_action_count=2,
    )
    assert selection == (1, 1)

    window = model_continuation_training_window_from_episode_rows(
        [_row(index) for index in range(5)],
        artifact,
        pose_frame_contract=_frame_contract(),
        source_episode_id="episode-tail",
        start_index=3,
        sampling_spec=ordered,
        draw_index=0,
    )
    assert window.continuation.valid_action_count == 2
    assert window.continuation.forecast_delay_steps == 1
    assert window.continuation.prefix_steps == 1


def test_sampling_spec_rejects_ambiguous_or_unknown_configuration() -> None:
    with pytest.raises(ModelContinuationViewError, match="duplicates"):
        _sampling_spec(pairs=((1, 1), (1, 1)))
    with pytest.raises(ModelContinuationViewError, match="0 <= prefix_steps"):
        _sampling_spec(pairs=((1, 2),))
    payload = _sampling_spec().to_dict()
    payload["unknown"] = "value"
    with pytest.raises(ModelContinuationViewError, match="unknown entries"):
        ModelContinuationTrainingSamplingSpec.from_dict(payload)


def test_trained_window_rejects_unbound_profile_and_window_tampering() -> None:
    rows = [_row(index) for index in range(5)]
    spec = _sampling_spec()
    with pytest.raises(
        ModelContinuationViewError,
        match="trained_prefix_conditioning_v1 provenance",
    ):
        model_continuation_training_window_from_episode_rows(
            rows,
            replace(
                _artifact(),
                model_continuation=ModelContinuationDataContractV2(
                    conditioning_origin=(
                        MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1
                    ),
                    runtime=ModelContinuationRuntimeContractV2.from_v1(
                        _artifact().model_continuation
                    ),
                    training=None,
                ),
            ),
            pose_frame_contract=_frame_contract(),
            source_episode_id="episode-unbound",
            start_index=0,
            sampling_spec=spec,
            draw_index=0,
        )

    with pytest.raises(
        ModelContinuationViewError,
        match="outside the artifact-supported range",
    ):
        select_model_continuation_training_delay_prefix(
            _sampling_spec(pairs=((3, 1),)),
            _trained_artifact(),
            source_episode_id="episode-invalid-spec",
            source_start_index=0,
            draw_index=0,
            valid_action_count=4,
        )

    window = model_continuation_training_window_from_episode_rows(
        rows,
        _trained_artifact(),
        pose_frame_contract=_frame_contract(),
        source_episode_id="episode-tamper",
        start_index=0,
        sampling_spec=spec,
        draw_index=0,
    )
    with pytest.raises(
        ModelContinuationViewError,
        match="source_start_index differ",
    ):
        replace(
            window,
            pose_chunk=replace(
                window.pose_chunk,
                metadata={
                    **window.pose_chunk.metadata,
                    "start_index": 1,
                },
            ),
        )
