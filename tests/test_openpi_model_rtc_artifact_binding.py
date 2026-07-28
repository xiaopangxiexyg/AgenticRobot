from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1,
    MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
    ModelContinuationDataContractV2,
)
from dataset_contracts.policy_artifact import (
    PolicyArtifactContract,
    PolicyCameraInputContract,
)
from dataset_contracts.v22 import PolicyIOLayout
from deployments.openpi.client import (
    OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1,
    OPENPI_CONDITIONED_RTC_WIRE_V1,
    OPENPI_RTC_RESULT_LEGACY_NORMALIZED_V1,
    OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
    OpenPIFinalChunkRtcConfig,
    OpenPIInferenceOnlyRtcBindingSpec,
    bind_openpi_inference_only_rtc_artifact,
    openpi_conditioned_provider_capability_hash,
)
from deployments.client import load_policy_artifact


_MODEL_RTC_CONFIG_DIR = (
    Path(__file__).resolve().parents[1]
    / "deployments"
    / "openpi"
    / "configs"
    / "model_rtc"
)


def _artifact(
    *,
    model_id: str = "portable-openpi-model",
    checkpoint_step: int = 123,
) -> PolicyArtifactContract:
    return PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id=model_id,
        checkpoint_step=checkpoint_step,
        state_key="state",
        action_key="actions",
        camera_input=PolicyCameraInputContract(
            inputs=(),
            required=False,
        ),
        metadata={"fixture": "model-rtc-artifact-binding"},
    )


def _config(
    artifact: PolicyArtifactContract,
    *,
    result_representation: str = OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
) -> OpenPIFinalChunkRtcConfig:
    return OpenPIFinalChunkRtcConfig(
        artifact=artifact,
        frame_contract_hash="frame:absolute:portable",
        normalization_hash="normalization:portable",
        action_pose_mode="absolute",
        result_action_representation=result_representation,
    )


def _provider(
    artifact: PolicyArtifactContract,
    *,
    horizon: int = 50,
    supports_bootstrap: bool = True,
) -> dict[str, object]:
    provider: dict[str, object] = {
        "provider_capability_schema": (
            OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1
        ),
        "wire_protocol": OPENPI_CONDITIONED_RTC_WIRE_V1,
        "enabled": True,
        "conditioning_transform_owner": "provider_v1",
        "external_action_dim": artifact.policy_io_layout.action_dim,
        "internal_sampler_action_dim": 32,
        "prediction_horizon_steps": horizon,
        "sampler_constraint": MODEL_CONTINUATION_SAMPLER_CLAMPED_V1,
        "supports_unconditioned_bootstrap": supports_bootstrap,
        "continuation_implementation_id": (
            "openpi.pi0.hard-flow-prefix.v1"
        ),
        "continuation_implementation_parameters_hash": (
            "sha256:" + "a" * 64
        ),
    }
    provider["provider_capability_hash"] = (
        openpi_conditioned_provider_capability_hash(provider)
    )
    return provider


def _spec() -> OpenPIInferenceOnlyRtcBindingSpec:
    return OpenPIInferenceOnlyRtcBindingSpec(
        minimum_forecast_delay_steps=1,
        maximum_forecast_delay_steps=16,
        minimum_prefix_steps=0,
        maximum_prefix_steps=16,
    )


@pytest.mark.parametrize(
    ("model_id", "checkpoint_step", "horizon"),
    (
        ("portable-openpi-a", 1, 20),
        ("portable-openpi-b", 92599, 50),
        ("portable-openpi-c", 150000, 64),
    ),
)
def test_inference_only_binding_has_no_model_or_checkpoint_whitelist(
    model_id: str,
    checkpoint_step: int,
    horizon: int,
) -> None:
    artifact = _artifact(
        model_id=model_id,
        checkpoint_step=checkpoint_step,
    )
    maximum = min(16, horizon - 1)
    binding = bind_openpi_inference_only_rtc_artifact(
        base_config=_config(artifact),
        provider_capability=_provider(artifact, horizon=horizon),
        spec=OpenPIInferenceOnlyRtcBindingSpec(
            minimum_forecast_delay_steps=1,
            maximum_forecast_delay_steps=maximum,
            minimum_prefix_steps=0,
            maximum_prefix_steps=maximum,
        ),
    )

    assert artifact.model_continuation is None
    continuation = binding.bound_artifact.model_continuation
    assert isinstance(continuation, ModelContinuationDataContractV2)
    assert (
        continuation.conditioning_origin
        == MODEL_CONTINUATION_ORIGIN_INFERENCE_ONLY_V1
    )
    assert continuation.training is None
    assert continuation.runtime.prediction_horizon_steps == horizon
    assert continuation.runtime.model_action_dim == (
        artifact.policy_io_layout.action_dim
    )
    assert binding.source_artifact_hash != binding.bound_artifact_hash
    assert binding.receipt_dict()["checkpoint_step"] == checkpoint_step


def test_bound_artifact_round_trip_preserves_exact_receipt_identity() -> None:
    artifact = _artifact()
    provider = _provider(artifact)
    binding = bind_openpi_inference_only_rtc_artifact(
        base_config=_config(artifact),
        provider_capability=provider,
        spec=_spec(),
    )

    restored = PolicyArtifactContract.from_dict(
        binding.bound_artifact.to_dict()
    )
    assert restored == binding.bound_artifact
    assert restored.model_continuation is not None
    assert (
        restored.model_continuation.contract_hash
        == binding.continuation_contract_hash
    )
    assert (
        binding.receipt_dict()["provider_capability_hash"]
        == provider["provider_capability_hash"]
    )
    assert binding.receipt_dict()["training"] is None


def test_binding_rejects_capability_or_representation_drift() -> None:
    artifact = _artifact()
    provider = _provider(artifact)

    tampered_hash = dict(provider)
    tampered_hash["provider_capability_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="capability hash mismatch"):
        bind_openpi_inference_only_rtc_artifact(
            base_config=_config(artifact),
            provider_capability=tampered_hash,
            spec=_spec(),
        )

    wrong_dim = dict(provider)
    wrong_dim["external_action_dim"] = (
        artifact.policy_io_layout.action_dim + 1
    )
    wrong_dim["provider_capability_hash"] = (
        openpi_conditioned_provider_capability_hash(wrong_dim)
    )
    with pytest.raises(ValueError, match="dimension mismatches"):
        bind_openpi_inference_only_rtc_artifact(
            base_config=_config(artifact),
            provider_capability=wrong_dim,
            spec=_spec(),
        )

    with pytest.raises(ValueError, match="typed canonical"):
        bind_openpi_inference_only_rtc_artifact(
            base_config=_config(
                artifact,
                result_representation=(
                    OPENPI_RTC_RESULT_LEGACY_NORMALIZED_V1
                ),
            ),
            provider_capability=provider,
            spec=_spec(),
        )


def test_binding_rejects_unattested_bootstrap_and_existing_contract() -> None:
    artifact = _artifact()
    provider = _provider(artifact, supports_bootstrap=False)
    with pytest.raises(ValueError, match="bootstrap unsupported"):
        bind_openpi_inference_only_rtc_artifact(
            base_config=_config(artifact),
            provider_capability=provider,
            spec=_spec(),
        )

    first = bind_openpi_inference_only_rtc_artifact(
        base_config=_config(artifact),
        provider_capability=_provider(artifact),
        spec=_spec(),
    )
    with pytest.raises(ValueError, match="existing artifact"):
        bind_openpi_inference_only_rtc_artifact(
            base_config=_config(first.bound_artifact),
            provider_capability=_provider(first.bound_artifact),
            spec=_spec(),
        )


def test_binding_range_is_validated_by_the_shared_contract() -> None:
    artifact = _artifact()
    with pytest.raises(
        ValueError,
        match="leave a non-empty executable suffix",
    ):
        bind_openpi_inference_only_rtc_artifact(
            base_config=_config(artifact),
            provider_capability=_provider(artifact, horizon=8),
            spec=replace(
                _spec(),
                maximum_forecast_delay_steps=8,
                maximum_prefix_steps=8,
            ),
        )


def test_registered_92599_binding_replays_from_its_persisted_facts() -> None:
    bound_artifact = load_policy_artifact(
        _MODEL_RTC_CONFIG_DIR
        / "0712-wrist-cube-92599.policy_artifact.json"
    )
    with (
        _MODEL_RTC_CONFIG_DIR
        / "0712-wrist-cube-92599.binding_receipt.json"
    ).open("r", encoding="utf-8") as stream:
        receipt = json.load(stream)
    source_artifact = replace(
        bound_artifact,
        model_continuation=None,
    )
    continuation = bound_artifact.model_continuation
    assert isinstance(continuation, ModelContinuationDataContractV2)
    runtime = continuation.runtime
    replayed = bind_openpi_inference_only_rtc_artifact(
        base_config=OpenPIFinalChunkRtcConfig(
            artifact=source_artifact,
            frame_contract_hash=receipt["frame_binding"]["hash"],
            normalization_hash=receipt["normalization_binding"]["hash"],
            action_pose_mode="absolute",
            result_action_representation=(
                receipt["result_action_representation"]
            ),
        ),
        provider_capability=receipt["provider_capability"],
        spec=OpenPIInferenceOnlyRtcBindingSpec(
            minimum_forecast_delay_steps=(
                runtime.minimum_forecast_delay_steps
            ),
            maximum_forecast_delay_steps=(
                runtime.maximum_forecast_delay_steps
            ),
            minimum_prefix_steps=runtime.minimum_prefix_steps,
            maximum_prefix_steps=runtime.maximum_prefix_steps,
        ),
    )

    assert replayed.bound_artifact == bound_artifact
    for key, value in replayed.receipt_dict().items():
        assert receipt[key] == value
