"""Compatibility exports for the RM65 target execution candidate.

The canonical target-backend contract lives in
``teleop_runtime.backends.arms.rm65_candidate_manifest``. This deployment path
remains a stateless re-export for existing callers.
"""

from teleop_runtime.backends.arms.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_capability,
    build_rm65_policy_arm_qpos_candidate_catalog,
    build_rm65_policy_arm_qpos_candidate_manifest,
    build_rm65_policy_arm_qpos_candidate_target_capability,
)

__all__ = [
    "build_rm65_policy_arm_qpos_candidate_capability",
    "build_rm65_policy_arm_qpos_candidate_catalog",
    "build_rm65_policy_arm_qpos_candidate_manifest",
    "build_rm65_policy_arm_qpos_candidate_target_capability",
]
