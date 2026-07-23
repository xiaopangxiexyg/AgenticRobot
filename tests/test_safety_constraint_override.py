from __future__ import annotations

import pytest

from robot_motion.action import (
    SafetyConstraintOverrideSpec,
    admit_safety_constraint_override,
)


BASELINE_HASH = "baseline-policy-qpos-safety-v1"
BASELINE = {
    "max_arm_xyz_step_m": 0.08,
    "max_arm_rot_step_deg": 20.0,
    "max_arm_joint_step_rad": 0.20,
    "max_hand_step": 0.0,
}


def _spec(requested_bounds):
    return SafetyConstraintOverrideSpec.create(
        spec_id="policy-artifact-static-safety-v1",
        spec_version="v1",
        source_artifact_id="policy-artifact:sha256-test",
        baseline_safety_profile_hash=BASELINE_HASH,
        applicable_action_spaces=("bimanual_qpos_v1",),
        applicable_selection_groups=("arms",),
        allowed_override_fields=("max_arm_joint_step_rad",),
        requested_bounds=requested_bounds,
    )


def _admit(spec, **kwargs):
    return admit_safety_constraint_override(
        spec,
        baseline_safety_profile_hash=kwargs.pop("baseline_hash", BASELINE_HASH),
        baseline_bounds=kwargs.pop("baseline", BASELINE),
        action_space=kwargs.pop("action_space", "bimanual_qpos_v1"),
        selection_groups=kwargs.pop("selection_groups", ("arms",)),
        admitted_at=kwargs.pop("admitted_at", 10.0),
        **kwargs,
    )


def test_static_override_admits_only_tighter_bound() -> None:
    receipt = _admit(_spec({"max_arm_joint_step_rad": 0.1}))

    assert receipt.accepted
    assert receipt.checks[0].status == "tightened"
    assert receipt.effective_bounds["max_arm_joint_step_rad"] == 0.1
    assert receipt.effective_safety_profile_hash is not None


@pytest.mark.parametrize("requested", [0.0, 0.3])
def test_static_override_rejects_disable_or_loosen(requested: float) -> None:
    receipt = _admit(_spec({"max_arm_joint_step_rad": requested}))

    assert not receipt.accepted
    assert receipt.checks[0].status == "not_tightening"
    assert receipt.effective_safety_profile_hash is None


def test_override_rejects_wrong_baseline_or_target_binding() -> None:
    spec = _spec({"max_arm_joint_step_rad": 0.1})

    wrong_baseline = _admit(spec, baseline_hash="another-baseline")
    assert not wrong_baseline.accepted
    assert "baseline_profile_hash_mismatch" in wrong_baseline.rejection_reasons

    wrong_target = _admit(spec, action_space="bimanual_ee_pose_v1")
    assert not wrong_target.accepted
    assert "action_space_not_applicable" in wrong_target.rejection_reasons


def test_spec_rejects_dynamic_scope_and_unknown_fields() -> None:
    with pytest.raises(ValueError, match="static session"):
        SafetyConstraintOverrideSpec.create(
            spec_id="dynamic",
            spec_version="v1",
            source_artifact_id="artifact",
            baseline_safety_profile_hash=BASELINE_HASH,
            applicable_action_spaces=("bimanual_qpos_v1",),
            applicable_selection_groups=("arms",),
            allowed_override_fields=("max_arm_joint_step_rad",),
            requested_bounds={"max_arm_joint_step_rad": 0.1},
            scope="per_action_dynamic",
        )

    with pytest.raises(ValueError, match="Unsupported safety override fields"):
        SafetyConstraintOverrideSpec.create(
            spec_id="unknown",
            spec_version="v1",
            source_artifact_id="artifact",
            baseline_safety_profile_hash=BASELINE_HASH,
            applicable_action_spaces=("bimanual_qpos_v1",),
            applicable_selection_groups=("arms",),
            allowed_override_fields=("disable_collision",),
            requested_bounds={},
        )


def test_disabled_baseline_can_only_remain_disabled() -> None:
    spec = SafetyConstraintOverrideSpec.create(
        spec_id="disabled-hand-v1",
        spec_version="v1",
        source_artifact_id="artifact",
        baseline_safety_profile_hash=BASELINE_HASH,
        applicable_action_spaces=("bimanual_qpos_v1",),
        applicable_selection_groups=("arms",),
        allowed_override_fields=("max_hand_step",),
        requested_bounds={"max_hand_step": 0.1},
    )
    receipt = _admit(spec)

    assert not receipt.accepted
    assert receipt.checks[0].status == "disabled_baseline_cannot_change"
