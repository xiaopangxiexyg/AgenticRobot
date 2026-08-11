"""Model-agnostic deployment client primitives with lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, str] = {
    "PolicyActionChunk": ".action",
    "UnifiedActionChunk": ".action",
    "deploy_contract_from_policy_artifact": ".artifact",
    "load_policy_artifact": ".artifact",
    "DEFAULT_ACTION_KEY": ".contracts",
    "DEXTELEOP_DEPLOY_SCHEMA_V2": ".contracts",
    "DEXTELEOP_DEPLOY_SCHEMA_VERSION": ".contracts",
    "DeployPolicyContract": ".contracts",
    "MetadataValidationResult": ".contracts",
    "ServerMetadataError": ".contracts",
    "validate_server_metadata": ".contracts",
    "ObservationAdapter": ".remote_policy",
    "RemoteModelPolicy": ".remote_policy",
    "RemotePolicyConfig": ".remote_policy",
    "RemoteTransport": ".remote_policy",
    "HoldPolicy": ".policy",
    "LoadedPolicy": ".policy",
    "load_policy": ".policy",
    "ActionPreparer": ".runtime",
    "ClientRuntimeStepResult": ".runtime",
    "CommandDispatcher": ".runtime",
    "DeploymentClientRuntime": ".runtime",
    "DeploymentObservationSource": ".runtime",
    "ModelAgnosticRtcClient": ".rtc",
    "RtcClientInferenceBundle": ".rtc",
    "RtcClientProtocolError": ".rtc",
    "RtcModelBackend": ".rtc",
    "RtcModelSpaceDecoder": ".rtc",
    "RtcConditionedModelClientInferenceExecutorV3": ".rtc_executor",
    "RtcModelClientInferenceExecutor": ".rtc_executor",
    "RtcModelInputPreparer": ".rtc_executor",
    "RtcPreparedModelInput": ".rtc_executor",
    "rtc_client_inference_bundle_debug_payload": ".rtc_executor",
    "rtc_conditioned_client_inference_bundle_debug_payload": ".rtc_executor",
    "RtcConditionedClientInferenceBundleV3": ".rtc_conditioned_client",
    "RtcConditionedModelClientV3": ".rtc_conditioned_client",
    "RtcConditioningCodec": ".rtc_conditioning",
    "RtcConditioningCodecIdentity": ".rtc_conditioning",
    "RtcConditioningEncodingReceipt": ".rtc_conditioning",
    "RtcEncodedConditioning": ".rtc_conditioning",
    "encode_external_continuation_artifact_actions": ".rtc_conditioning",
    "encode_rtc_conditioning_bundle": ".rtc_conditioning",
    "rtc_binary_mask_hash": ".rtc_conditioning",
    "rtc_float32_array_hash": ".rtc_conditioning",
    "RTC_CONDITIONING_TRANSFORM_CLIENT_V1": ".rtc_conditioning_protocol",
    "RTC_CONDITIONING_TRANSFORM_PROVIDER_V1": ".rtc_conditioning_protocol",
    "RtcConditionedInferenceEnvelopeV3": ".rtc_conditioning_protocol",
    "RtcConditionedModelBackendV3": ".rtc_conditioning_protocol",
    "RtcConditionedModelResponseV3": ".rtc_conditioning_protocol",
    "RtcConditionedResponseValidationV3": ".rtc_conditioning_protocol",
    "RtcConditioningConsumptionReceiptV3": ".rtc_conditioning_protocol",
    "RtcModelContinuationCapabilityV3": ".rtc_conditioning_protocol",
    "build_rtc_conditioned_inference_envelope_v3": ".rtc_conditioning_protocol",
    "build_rtc_conditioning_consumption_receipt_v3": ".rtc_conditioning_protocol",
    "rtc_model_continuation_capability_from_loaded_codec": ".rtc_conditioning_protocol",
    "validate_rtc_conditioned_response_v3": ".rtc_conditioning_protocol",
    "validate_rtc_model_continuation_capability_v3_binding": ".rtc_conditioning_protocol",
    "FinalChunkModelOutput": ".rtc_final_chunk",
    "FinalChunkRtcBackend": ".rtc_final_chunk",
    "FinalChunkRtcBinding": ".rtc_final_chunk",
    "MappingFinalChunkOutputAdapter": ".rtc_final_chunk_adapter",
    "create_final_chunk_rtc_runtime_adapter": ".rtc_final_chunk_adapter",
    "RtcConditionedModelRuntimeAdapterV3": ".rtc_frontend",
    "RtcModelPolicyIdentity": ".rtc_frontend",
    "RtcModelRuntimeAdapter": ".rtc_frontend",
    "SimpleRtcModelRuntimeAdapter": ".rtc_frontend",
    "build_rtc_conditioned_model_shadow_frontend_v3": ".rtc_frontend",
    "build_rtc_model_producer_runtime": ".rtc_frontend",
    "build_rtc_model_shadow_frontend": ".rtc_frontend",
    "rtc_conditioned_config_from_model_adapter_v3": ".rtc_frontend",
    "rtc_producer_config_from_model_adapter": ".rtc_frontend",
    "rtc_shadow_config_from_model_adapter": ".rtc_frontend",
    "RtcTransportDeadlineExceeded": ".rtc_transport",
    "RtcTransportDeadlineSpec": ".rtc_transport",
    "RtcTransportPhaseReceipt": ".rtc_transport",
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
