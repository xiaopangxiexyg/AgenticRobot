from __future__ import annotations

from robot_motion.action import M6_CANDIDATE_POLICY_ARM_QPOS_V1
from teleop_runtime.backends.arms.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
    build_rm65_policy_arm_qpos_candidate_target_capability,
)


def test_rm65_candidate_manifest_is_deterministic_and_does_not_authorize_motion() -> None:
    first = build_rm65_policy_arm_qpos_candidate_manifest()
    second = build_rm65_policy_arm_qpos_candidate_manifest()

    assert first.manifest_id == M6_CANDIDATE_POLICY_ARM_QPOS_V1
    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.motion_authority_sources == ("policy",)
    assert first.human_motion_proposal_status == "not_applicable"
    assert first.safety_control_requests == ("hold", "estop", "manual_stop")
    assert first.controlled_components == ("left_arm", "right_arm")


def test_rm65_candidate_exposes_the_exact_preexisting_target_capability() -> None:
    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    capability = build_rm65_policy_arm_qpos_candidate_target_capability()

    assert capability.semantic_content_hash == (
        manifest.component_target_capability_hash
    )
    assert capability.action_schema_hash == manifest.action_schema_hash
    assert capability.action_schema.component_ids == manifest.controlled_components
