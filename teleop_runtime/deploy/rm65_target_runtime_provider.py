"""Compatibility exports for the RM65 target-runtime projection.

The adapter is no longer deployment-owned.  Keep this path as a stateless
shim for existing callers while its canonical implementation lives under
``teleop_runtime.target_runtime.integrations``.
"""

from teleop_runtime.target_runtime.integrations.rm65_policy_arm_qpos_candidate import (
    RM65_POLICY_ARM_QPOS_CANDIDATE_EMBODIMENT_DEFINITION_ID,
    RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_ID,
    RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_IDENTITY,
    RM65_POLICY_ARM_QPOS_CANDIDATE_REQUIREMENT_ID_PREFIX,
    RM65_POLICY_ARM_QPOS_CANDIDATE_SUPPORTED_CLOCK_DOMAIN_IDS,
    RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_HANDLER_ID,
    RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_PROFILE_ID,
    RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_PROFILE_VERSION,
    RM65PolicyArmQposCandidatePreflight,
    RM65PolicyArmQposCandidateTargetConfig,
    RM65PolicyArmQposCandidateTargetProvider,
    build_rm65_policy_arm_qpos_candidate_embodiment_definition_hash,
    build_rm65_policy_arm_qpos_candidate_provider_entry,
    build_rm65_policy_arm_qpos_candidate_target_profile_binding,
    preflight_rm65_policy_arm_qpos_candidate_target,
)

__all__ = [
    "RM65_POLICY_ARM_QPOS_CANDIDATE_EMBODIMENT_DEFINITION_ID",
    "RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_ID",
    "RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_IDENTITY",
    "RM65_POLICY_ARM_QPOS_CANDIDATE_REQUIREMENT_ID_PREFIX",
    "RM65_POLICY_ARM_QPOS_CANDIDATE_SUPPORTED_CLOCK_DOMAIN_IDS",
    "RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_HANDLER_ID",
    "RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_PROFILE_ID",
    "RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_PROFILE_VERSION",
    "RM65PolicyArmQposCandidatePreflight",
    "RM65PolicyArmQposCandidateTargetConfig",
    "RM65PolicyArmQposCandidateTargetProvider",
    "build_rm65_policy_arm_qpos_candidate_embodiment_definition_hash",
    "build_rm65_policy_arm_qpos_candidate_provider_entry",
    "build_rm65_policy_arm_qpos_candidate_target_profile_binding",
    "preflight_rm65_policy_arm_qpos_candidate_target",
]
