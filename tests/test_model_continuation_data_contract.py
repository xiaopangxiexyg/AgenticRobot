from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from dataset_contracts import (
    MODEL_CONTINUATION_DATA_HASH_KEY,
    MODEL_CONTINUATION_DATA_METADATA_KEY,
    MODEL_CONTINUATION_FLOW_INPAINT_V1,
    MODEL_CONTINUATION_LOSS_ALL_VALID_V1,
    MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
    MODEL_CONTINUATION_ORIGIN_LEGACY_V1_COMPAT,
    MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    MODEL_CONTINUATION_SAMPLER_WEIGHTED_V1,
    RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
    TRAINING_PREFIX_RECORDED_POLICY_OUTPUT_V1,
    TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    ModelContinuationDataContract,
    ModelContinuationDataContractV2,
    ModelContinuationRuntimeContractV2,
    ModelContinuationTrainingContractV2,
    PolicyArtifactContract,
    PoseActionChunkContract,
    PoseTrainingDataContract,
    model_continuation_contract_from_dict,
    normalize_model_continuation_contract,
)
from dataset_contracts.v22 import (
    DatasetLayoutBindingError,
    PolicyIOLayout,
)
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from deployments.openpi.client.rtc import openpi_rtc_artifact_hash


LEGACY_OPENPI_ARTIFACT_HASH = (
    "8b04f51a97e7eadaba8812da506b69f492492efc1b2640856b98e5b4bb9707de"
)
MODEL_CONTINUATION_V1_CONTRACT_HASH = (
    "sha256:61ab57928e6ff402f330a639884c64d30e111672a50d4125b07c73bb5f1a6ca4"
)


def _layout() -> PolicyIOLayout:
    return PolicyIOLayout.legacy_v21_q6()


def _contract(
    *,
    layout: PolicyIOLayout | None = None,
    prediction_horizon_steps: int = 8,
    model_action_dim: int | None = None,
    model_action_discrete_dim: int = 0,
    minimum_forecast_delay_steps: int = 2,
    maximum_forecast_delay_steps: int = 4,
    minimum_prefix_steps: int = 1,
    maximum_prefix_steps: int = 4,
    supports_unconditioned_bootstrap: bool = False,
) -> ModelContinuationDataContract:
    resolved_layout = _layout() if layout is None else layout
    return ModelContinuationDataContract(
        continuation_mode=MODEL_CONTINUATION_FLOW_INPAINT_V1,
        continuation_implementation_id="fake-flow-inpaint-v1",
        continuation_implementation_parameters_hash="parameters:test-v1",
        prediction_horizon_steps=prediction_horizon_steps,
        minimum_forecast_delay_steps=minimum_forecast_delay_steps,
        maximum_forecast_delay_steps=maximum_forecast_delay_steps,
        minimum_prefix_steps=minimum_prefix_steps,
        maximum_prefix_steps=maximum_prefix_steps,
        model_action_dim=(
            resolved_layout.action_dim
            if model_action_dim is None
            else model_action_dim
        ),
        model_action_discrete_dim=model_action_discrete_dim,
        component_scope=("right_arm", "left_arm"),
        supported_action_pose_modes=("absolute",),
        sampler_constraints=(
            MODEL_CONTINUATION_SAMPLER_WEIGHTED_V1,
            MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
        ),
        training_prefix_source_kinds=(
            TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
        ),
        runtime_prefix_source_roles=(
            RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
        ),
        action_codec_hash="codec:test-v1",
        layout_hash=resolved_layout.layout_hash,
        frame_contract_hash="frame:test-v1",
        normalization_hash="normalization:test-v1",
        training_projection_implementation_id="teacher-overlap-projection-v1",
        training_projection_parameters_hash="projection-parameters:test-v1",
        supports_unconditioned_bootstrap=supports_unconditioned_bootstrap,
    )


def test_contract_round_trip_is_canonical_and_source_roles_stay_distinct() -> None:
    contract = _contract()
    restored = ModelContinuationDataContract.from_dict(contract.to_dict())

    assert restored == contract
    assert restored.component_scope == ("left_arm", "right_arm")
    assert restored.sampler_constraints == (
        MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
        MODEL_CONTINUATION_SAMPLER_WEIGHTED_V1,
    )
    assert restored.training_prefix_source_kinds == (
        TRAINING_PREFIX_TRAJECTORY_OVERLAP_TEACHER_V1,
    )
    assert restored.runtime_prefix_source_roles == (
        RUNTIME_PREFIX_PRODUCER_LOCAL_OVERLAP_V1,
    )
    assert restored.contract_hash.startswith("sha256:")
    assert len(restored.contract_hash) == len("sha256:") + 64


def test_v1_normalizes_as_legacy_v2_profile_without_identity_promotion() -> None:
    contract = _contract()
    normalized = normalize_model_continuation_contract(contract)

    assert contract.contract_hash == MODEL_CONTINUATION_V1_CONTRACT_HASH
    assert normalized.source_schema == contract.schema
    assert normalized.source_contract_hash == contract.contract_hash
    assert (
        normalized.conditioning_origin
        == MODEL_CONTINUATION_ORIGIN_LEGACY_V1_COMPAT
    )
    assert normalized.runtime.layout_hash == contract.layout_hash
    assert normalized.runtime.sampler_constraints == (
        contract.sampler_constraints
    )
    assert normalized.training is not None
    assert normalized.training.prefix_source_kinds == (
        contract.training_prefix_source_kinds
    )
    assert model_continuation_contract_from_dict(contract.to_dict()) == contract


def test_v2_inference_only_round_trip_has_no_training_claim() -> None:
    v1 = _contract()
    contract = ModelContinuationDataContractV2(
        conditioning_origin=MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
        runtime=ModelContinuationRuntimeContractV2.from_v1(v1),
        training=None,
    )
    restored = model_continuation_contract_from_dict(contract.to_dict())
    normalized = normalize_model_continuation_contract(restored)

    assert restored == contract
    assert normalized.source_contract_hash == contract.contract_hash
    assert (
        normalized.conditioning_origin
        == MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1
    )
    assert normalized.training is None
    assert normalized.runtime.layout_hash == v1.layout_hash
    assert contract.contract_hash != v1.contract_hash


def test_v2_trained_profile_requires_exact_training_contract() -> None:
    v1 = _contract()
    runtime = ModelContinuationRuntimeContractV2.from_v1(v1)
    training = ModelContinuationTrainingContractV2.from_v1(v1)
    trained = ModelContinuationDataContractV2(
        conditioning_origin=MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
        runtime=runtime,
        training=training,
    )

    assert (
        ModelContinuationDataContractV2.from_dict(trained.to_dict())
        == trained
    )
    with pytest.raises(
        DatasetLayoutBindingError,
        match="requires a training contract",
    ):
        ModelContinuationDataContractV2(
            conditioning_origin=MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
            runtime=runtime,
            training=None,
        )
    with pytest.raises(
        DatasetLayoutBindingError,
        match="requires training=null",
    ):
        ModelContinuationDataContractV2(
            conditioning_origin=MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
            runtime=runtime,
            training=training,
        )
    with pytest.raises(
        DatasetLayoutBindingError,
        match="Unsupported conditioning_origin",
    ):
        ModelContinuationDataContractV2(
            conditioning_origin=MODEL_CONTINUATION_ORIGIN_LEGACY_V1_COMPAT,
            runtime=runtime,
            training=training,
        )


def test_contract_identity_binds_training_source_and_loss_semantics() -> None:
    teacher = _contract()
    recorded = replace(
        teacher,
        training_prefix_source_kinds=(
            TRAINING_PREFIX_RECORDED_POLICY_OUTPUT_V1,
        ),
    )
    all_valid_loss = replace(
        teacher,
        loss_mask_semantics=MODEL_CONTINUATION_LOSS_ALL_VALID_V1,
    )

    assert teacher.contract_hash != recorded.contract_hash
    assert teacher.contract_hash != all_valid_loss.contract_hash


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {"maximum_prefix_steps": 5},
            "cannot exceed their forecast-delay range",
        ),
        (
            {
                "minimum_forecast_delay_steps": 0,
                "minimum_prefix_steps": 1,
            },
            "cannot exceed their forecast-delay range",
        ),
        (
            {
                "minimum_prefix_steps": 0,
                "supports_unconditioned_bootstrap": False,
            },
            "exactly when unconditioned bootstrap is supported",
        ),
        (
            {
                "minimum_prefix_steps": 1,
                "supports_unconditioned_bootstrap": True,
            },
            "exactly when unconditioned bootstrap is supported",
        ),
        (
            {"model_action_discrete_dim": 1},
            "supports only continuous prefixes",
        ),
        (
            {"runtime_prefix_source_roles": ("trajectory_overlap_teacher_v1",)},
            "unsupported values",
        ),
    ),
)
def test_contract_rejects_ambiguous_or_unsupported_semantics(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(DatasetLayoutBindingError, match=message):
        replace(_contract(), **changes)


def test_contract_parser_rejects_unknown_fields() -> None:
    payload = _contract().to_dict()
    payload["runtime_plan_id"] = "must-not-enter-pure-data-contract"
    with pytest.raises(DatasetLayoutBindingError, match="unknown entries"):
        ModelContinuationDataContract.from_dict(payload)


def test_contract_direct_constructor_rejects_scalar_sequence_fields() -> None:
    with pytest.raises(DatasetLayoutBindingError, match="component_scope"):
        replace(_contract(), component_scope="left_arm")
    with pytest.raises(DatasetLayoutBindingError, match="must be an integer"):
        replace(_contract(), maximum_prefix_steps=4.0)

    payload = _contract().to_dict()
    payload["action_codec_hash"] = None
    with pytest.raises(DatasetLayoutBindingError, match="must be a string"):
        ModelContinuationDataContract.from_dict(payload)


def test_legacy_artifact_payload_and_raw_rtc_hash_are_unchanged() -> None:
    artifact = PolicyArtifactContract(policy_io_layout=_layout())
    payload = artifact.to_dict()

    assert MODEL_CONTINUATION_DATA_METADATA_KEY not in payload
    assert MODEL_CONTINUATION_DATA_HASH_KEY not in payload
    assert PolicyArtifactContract.from_dict(payload).to_dict() == payload
    assert openpi_rtc_artifact_hash(artifact) == LEGACY_OPENPI_ARTIFACT_HASH


def test_artifact_optionally_binds_exact_continuation_contract_and_hash() -> None:
    layout = _layout()
    continuation = _contract(layout=layout)
    artifact = PolicyArtifactContract(
        policy_io_layout=layout,
        model_continuation=continuation,
    )
    payload = artifact.to_dict()
    restored = PolicyArtifactContract.from_dict(payload)

    assert restored.model_continuation == continuation
    assert (
        payload[MODEL_CONTINUATION_DATA_HASH_KEY]
        == continuation.contract_hash
    )
    assert payload[MODEL_CONTINUATION_DATA_METADATA_KEY] == (
        continuation.to_dict()
    )
    assert openpi_rtc_artifact_hash(artifact) != LEGACY_OPENPI_ARTIFACT_HASH

    tampered = deepcopy(payload)
    tampered[MODEL_CONTINUATION_DATA_HASH_KEY] = "sha256:" + "0" * 64
    with pytest.raises(DatasetLayoutBindingError, match="does not match"):
        PolicyArtifactContract.from_dict(tampered)

    missing = deepcopy(payload)
    del missing[MODEL_CONTINUATION_DATA_METADATA_KEY]
    with pytest.raises(
        DatasetLayoutBindingError,
        match="hash without its contract",
    ):
        PolicyArtifactContract.from_dict(missing)


def test_artifact_preserves_v2_source_contract_and_hash() -> None:
    layout = _layout()
    v1 = _contract(layout=layout)
    continuation = ModelContinuationDataContractV2(
        conditioning_origin=MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
        runtime=ModelContinuationRuntimeContractV2.from_v1(v1),
    )
    artifact = PolicyArtifactContract(
        policy_io_layout=layout,
        model_continuation=continuation,
    )
    payload = artifact.to_dict()
    restored = PolicyArtifactContract.from_dict(payload)

    assert restored.model_continuation == continuation
    assert (
        payload[MODEL_CONTINUATION_DATA_HASH_KEY]
        == continuation.contract_hash
    )
    assert (
        payload[MODEL_CONTINUATION_DATA_METADATA_KEY]
        == continuation.to_dict()
    )


def test_artifact_rejects_layout_dimension_and_pose_chunk_mismatches() -> None:
    layout = _layout()
    with pytest.raises(DatasetLayoutBindingError, match="layout hash"):
        PolicyArtifactContract(
            policy_io_layout=layout,
            model_continuation=replace(
                _contract(layout=layout),
                layout_hash="sha256:" + "1" * 64,
            ),
        )

    with pytest.raises(DatasetLayoutBindingError, match="action dimension"):
        PolicyArtifactContract(
            policy_io_layout=layout,
            model_continuation=_contract(
                layout=layout,
                model_action_dim=layout.action_dim + 1,
            ),
        )

    with pytest.raises(DatasetLayoutBindingError, match="prediction horizon"):
        PolicyArtifactContract(
            policy_io_layout=layout,
            dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
            pose_action_chunk=PoseActionChunkContract(
                action_horizon=4,
                action_pose_mode="absolute",
            ),
            pose_training_data=PoseTrainingDataContract.native_v23(),
            pose_frame_contract_hashes=("sha256:" + "2" * 64,),
            model_continuation=_contract(
                layout=layout,
                prediction_horizon_steps=8,
            ),
        )

    frame_hash = "sha256:" + "2" * 64
    with pytest.raises(DatasetLayoutBindingError, match="frame contract hash"):
        PolicyArtifactContract(
            policy_io_layout=layout,
            dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
            pose_action_chunk=PoseActionChunkContract(
                action_horizon=8,
                action_pose_mode="absolute",
            ),
            pose_training_data=PoseTrainingDataContract.native_v23(),
            pose_frame_contract_hashes=(frame_hash,),
            model_continuation=_contract(
                layout=layout,
                prediction_horizon_steps=8,
            ),
        )
