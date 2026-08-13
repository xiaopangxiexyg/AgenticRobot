"""Policy deployment compatibility exports, loaded on explicit access."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "ActionChunkBuffer",
    "ActionSafetyFilter",
    "ActionSafetyFilterConfig",
    "CanonicalDeployActionPipeline",
    "CanonicalDeployActionResult",
    "DeploymentAction",
    "DeploymentObservation",
    "FilteredNormalizedAction",
    "HoldPolicy",
    "LoadedPolicy",
    "NormalizedAction",
    "PoseDeploymentError",
    "PolicyActionEmissionFanout",
    "PolicyActionEmissionDeliveryOutcome",
    "PolicyActionEmissionPublicationReceipt",
    "RM65ExecutionEvidenceBundle",
    "RM65LegacyWorkerExecutionBinding",
    "RM65IdleWorkerStopEvidence",
    "RM65IdleWorkerStopEvidenceBinding",
    "RM65LegacyBackendHoldReceipt",
    "RM65LegacyControlStopBinding",
    "RM65LegacyControlStopReceipt",
    "RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V1",
    "RTC_3C_LOCAL_POLICY_PROTOCOL_V1",
    "RTC_DEBUG_LEVEL_OFF",
    "RTC_DEBUG_LEVEL_SUMMARY",
    "RTC_DEBUG_LEVEL_TRACE",
    "RTC_DEBUG_LEVELS",
    "RTC_FRONTEND_STEP_SOURCE_BINDING_HASH_DOMAIN_V1",
    "RTC_POLICY_EMISSION_BINDING_ACK_HASH_DOMAIN_V1",
    "RTC_POLICY_EMISSION_PUBLICATION_KEY_HASH_DOMAIN_V1",
    "RTC_POLICY_EMISSION_PUBLICATION_RECEIPT_HASH_DOMAIN_V1",
    "RTC_POLICY_EMISSION_SOURCE_BINDING_HASH_DOMAIN_V1",
    "RtcDecodedPolicyExecutor",
    "RtcDebugEventObserver",
    "RtcDebugEventObserverStatus",
    "RtcFrontendStepSourceBinding",
    "RtcFrontendLifecycleTransition",
    "RtcFrontendStep",
    "RtcFrontendStepFanout",
    "RtcFrontendStepSubscriberStatus",
    "RtcPolicyShadowConfig",
    "RtcPolicyShadowFrontend",
    "RtcPolicyProducerConfig",
    "RtcPolicyProducerRuntime",
    "RtcProducerTickResult",
    "RtcPolicyEmissionSourceBinding",
    "RtcPolicyEmissionBindingAck",
    "RtcPolicyEmissionPublisher",
    "RtcProducerDebugStepFanout",
    "build_deployment_observation",
    "load_policy",
    "normalize_policy_output",
    "normalize_rtc_debug_level",
    "convert_last_action_to_policy_mode",
    "convert_policy_absolute_vector_to_runtime_raw",
    "convert_normalized_action_chunk_to_absolute",
    "convert_normalized_action_chunk_to_relative",
    "measured_tcp_anchors_by_side",
    "prepare_observation_for_policy_pose_mode",
    "build_rm65_execution_evidence_bundle",
    "build_fake_only_rm65_legacy_worker_binding",
    "compile_rm65_idle_worker_backend_hold_receipt",
    "converge_rm65_legacy_control_stop",
    "build_rtc_frontend_step_source_binding",
    "build_rtc_policy_emission_binding_ack",
    "build_rtc_policy_emission_source_binding",
]


_LAZY_EXPORT_GROUPS = (
    (
        "teleop_runtime.deploy.actions",
        (
            "ActionChunkBuffer",
            "ActionSafetyFilter",
            "ActionSafetyFilterConfig",
            "DeploymentAction",
            "FilteredNormalizedAction",
            "NormalizedAction",
            "normalize_policy_output",
        ),
    ),
    (
        "teleop_runtime.deploy.canonical_action",
        ("CanonicalDeployActionPipeline", "CanonicalDeployActionResult"),
    ),
    (
        "teleop_runtime.deploy.observation",
        ("DeploymentObservation", "build_deployment_observation"),
    ),
    (
        "robot_motion.deploy.pose_adapter",
        (
            "PoseDeploymentError",
            "convert_last_action_to_policy_mode",
            "convert_policy_absolute_vector_to_runtime_raw",
            "convert_normalized_action_chunk_to_absolute",
            "convert_normalized_action_chunk_to_relative",
            "measured_tcp_anchors_by_side",
            "prepare_observation_for_policy_pose_mode",
        ),
    ),
    (
        "teleop_runtime.deploy.policy",
        ("HoldPolicy", "LoadedPolicy", "load_policy"),
    ),
    (
        "teleop_runtime.deploy.rm65_execution_evidence",
        (
            "RM65ExecutionEvidenceBundle",
            "build_rm65_execution_evidence_bundle",
        ),
    ),
    (
        "teleop_runtime.deploy.rm65_legacy_worker_binding",
        (
            "RM65LegacyWorkerExecutionBinding",
            "build_fake_only_rm65_legacy_worker_binding",
        ),
    ),
    (
        "teleop_runtime.deploy.rm65_legacy_hold_convergence",
        (
            "RM65IdleWorkerStopEvidence",
            "RM65IdleWorkerStopEvidenceBinding",
            "RM65LegacyBackendHoldReceipt",
            "RM65LegacyControlStopBinding",
            "RM65LegacyControlStopReceipt",
            "compile_rm65_idle_worker_backend_hold_receipt",
            "converge_rm65_legacy_control_stop",
        ),
    ),
    (
        "teleop_runtime.deploy.rtc_frontend",
        (
            "RTC_3C_FRONTEND_CONFIG_HASH_DOMAIN_V1",
            "RTC_3C_LOCAL_POLICY_PROTOCOL_V1",
            "RtcDecodedPolicyExecutor",
            "RtcFrontendLifecycleTransition",
            "RtcFrontendStep",
            "RtcPolicyProducerConfig",
            "RtcPolicyProducerRuntime",
            "RtcPolicyShadowConfig",
            "RtcPolicyShadowFrontend",
            "RtcProducerTickResult",
        ),
    ),
    (
        "teleop_runtime.deploy.rtc_debug",
        (
            "RTC_DEBUG_LEVEL_OFF",
            "RTC_DEBUG_LEVEL_SUMMARY",
            "RTC_DEBUG_LEVEL_TRACE",
            "RTC_DEBUG_LEVELS",
            "RtcDebugEventObserver",
            "RtcDebugEventObserverStatus",
            "normalize_rtc_debug_level",
        ),
    ),
    (
        "teleop_runtime.deploy.rtc_step_fanout",
        (
            "PolicyActionEmissionDeliveryOutcome",
            "PolicyActionEmissionFanout",
            "PolicyActionEmissionPublicationReceipt",
            "RTC_FRONTEND_STEP_SOURCE_BINDING_HASH_DOMAIN_V1",
            "RTC_POLICY_EMISSION_BINDING_ACK_HASH_DOMAIN_V1",
            "RTC_POLICY_EMISSION_PUBLICATION_KEY_HASH_DOMAIN_V1",
            "RTC_POLICY_EMISSION_PUBLICATION_RECEIPT_HASH_DOMAIN_V1",
            "RTC_POLICY_EMISSION_SOURCE_BINDING_HASH_DOMAIN_V1",
            "RtcFrontendStepSourceBinding",
            "RtcFrontendStepFanout",
            "RtcFrontendStepSubscriberStatus",
            "RtcPolicyEmissionSourceBinding",
            "RtcPolicyEmissionBindingAck",
            "RtcPolicyEmissionPublisher",
            "RtcProducerDebugStepFanout",
            "build_rtc_frontend_step_source_binding",
            "build_rtc_policy_emission_binding_ack",
            "build_rtc_policy_emission_source_binding",
        ),
    ),
)
_LAZY_EXPORTS = {
    name: module_name
    for module_name, names in _LAZY_EXPORT_GROUPS
    for name in names
}

if set(_LAZY_EXPORTS) != set(__all__):
    raise RuntimeError("teleop_runtime.deploy lazy exports do not match __all__.")


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
