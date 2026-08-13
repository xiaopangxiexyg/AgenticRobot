"""OpenPI-specific deployment client adapters with lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, str] = {
    "OpenPIObservationAdapterConfig": ".observation_adapter",
    "deployment_observation_to_openpi": ".observation_adapter",
    "make_openpi_quantile_normalization_contract": ".normalization",
    "OPENPI_LOCAL_PROTOCOL_VERSION": ".policy",
    "OpenPIClientPolicy": ".policy",
    "OpenPIClientPolicyConfig": ".policy",
    "create_policy": ".policy",
    "OPENPI_SIM_ADAPTER_SCHEMA_V1": ".sim_adapter",
    "OPENPI_SIM_CODEC_HASH_DOMAIN_V1": ".sim_adapter",
    "OpenPIModelPolicy": ".sim_adapter",
    "OpenPISimAdapter": ".sim_adapter",
    "OpenPISimAdapterConfig": ".sim_adapter",
    "OpenPISimPrediction": ".sim_adapter",
    "OpenPIArtifactPreflightReport": ".preflight",
    "preflight_policy_artifact": ".preflight",
    "preflight_policy_artifact_path": ".preflight",
    "OPENPI_POLICY_EMISSION_SOURCE_ID_V1": ".emission_source",
    "OpenPIEmissionSourceConfig": ".emission_source",
    "build_openpi_policy_emission_source": ".emission_source",
    "validate_openpi_state_policy_artifact": ".emission_source",
    "OPENPI_FINAL_CHUNK_MODEL_ACTION_SPACE_V1": ".rtc",
    "OPENPI_FINAL_CHUNK_RTC_ADAPTER_V1": ".rtc",
    "OPENPI_RTC_RESULT_LEGACY_NORMALIZED_V1": ".rtc",
    "OPENPI_RTC_RESULT_TYPED_CANONICAL_V1": ".rtc",
    "OpenPIFinalChunkRtcBackend": ".rtc",
    "OpenPIFinalChunkRtcConfig": ".rtc",
    "OpenPIRtcPreparedInput": ".rtc",
    "OpenPIV22FinalChunkDecoder": ".rtc",
    "OpenPIV22NativeRobotActionDecoder": ".rtc",
    "openpi_arm_pose_slices": ".rtc",
    "openpi_arm_reference_frames": ".rtc",
    "openpi_absolute_conversion_contract_hash": ".rtc",
    "openpi_relative_conversion_contract_hash": ".rtc",
    "openpi_rtc_artifact_hash": "..artifact_identity",
    "openpi_rtc_component_scope": ".rtc",
    "openpi_v22_legacy_codec_hash": ".rtc",
    "openpi_v22_native_codec_hash": ".rtc",
    "openpi_v22_relative_native_codec_hash": ".rtc",
    "OPENPI_CONDITIONED_RTC_METADATA_KEY_V1": ".rtc_conditioned",
    "OPENPI_CONDITIONED_RTC_OPERATION_V1": ".rtc_conditioned",
    "OPENPI_CONDITIONED_PROVIDER_CAPABILITY_SCHEMA_V1": ".rtc_conditioned",
    "OPENPI_CONDITIONED_RTC_RECEIPT_KEY_V1": ".rtc_conditioned",
    "OPENPI_CONDITIONED_RTC_WIRE_V1": ".rtc_conditioned",
    "OpenPIConditionedRtcBackendV3": ".rtc_conditioned",
    "OpenPIConditionedRtcConfig": ".rtc_conditioned",
    "OpenPIV22ConditioningCodec": ".rtc_conditioned",
    "openpi_conditioned_provider_capability_hash": ".rtc_conditioned",
    "OPENPI_INFERENCE_ONLY_RTC_BINDING_HASH_DOMAIN_V1": ".rtc_artifact_binding",
    "OPENPI_INFERENCE_ONLY_RTC_BINDING_SCHEMA_V1": ".rtc_artifact_binding",
    "OpenPIInferenceOnlyRtcArtifactBinding": ".rtc_artifact_binding",
    "OpenPIInferenceOnlyRtcBindingSpec": ".rtc_artifact_binding",
    "bind_openpi_inference_only_rtc_artifact": ".rtc_artifact_binding",
    "OpenPITransportCloseReceipt": ".transport",
    "OpenPITransportLifecycleReceipt": ".transport",
    "OpenPITransportSessionCancelledError": ".transport",
    "OpenPIWebsocketTransport": ".transport",
    "OpenPIWebsocketTransportConfig": ".transport",
    "RtcTransportDeadlineExceeded": ".transport",
    "RtcTransportDeadlineSpec": ".transport",
    "RtcTransportPhaseReceipt": ".transport",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = sorted(_EXPORTS)
