from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from robot_motion.runtime import TeleopObservation
from teleop_runtime.configs import build_annotated_dual_rm65_calibration
from teleop_runtime.skill_deploy import (
    ExclusiveRM65SkillStartupSafetyBinder,
    ExclusiveRM65SkillTargetSafetyRuntime,
    RealSkillMotionHoldReceipt,
    RealSkillMotionObservationCut,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PROFILE_DIR = REPO_ROOT / "teleop_runtime/configs/skill_deploy"
LEGACY_PROFILE_DIR = REPO_ROOT / "control_station/real_teleop/configs"


class _TargetBuilder:
    def __init__(self) -> None:
        self.pre_ik_target_safety_filter = None
        self.syncs: list[tuple[np.ndarray, np.ndarray]] = []

    def set_current_qpos_pair(
        self,
        *,
        left_qpos,
        right_qpos,
        reset_tracking=True,
    ) -> None:
        assert reset_tracking is True
        self.syncs.append(
            (
                np.asarray(left_qpos, dtype=float).copy(),
                np.asarray(right_qpos, dtype=float).copy(),
            )
        )

    def tool_tcp_world_target_from_solver_world_target(
        self,
        *,
        side,
        solver_world_target,
    ):
        assert side in {"left", "right"}
        return np.asarray(solver_world_target, dtype=float).copy()

    def solver_world_target_from_tool_tcp_world_target(
        self,
        *,
        side,
        tool_tcp_world_target,
    ):
        assert side in {"left", "right"}
        return np.asarray(tool_tcp_world_target, dtype=float).copy()

    def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
        result = {}
        for side, qpos in (("left", left_qpos), ("right", right_qpos)):
            pose = np.eye(4)
            pose[0, 3] = float(np.asarray(qpos, dtype=float)[0])
            result[side] = pose
        return result


class _Ports:
    def __init__(
        self,
        *,
        target_builder,
        cut,
        hold_receipt,
    ) -> None:
        self.target_builder = target_builder
        self.startup_hold_receipt = hold_receipt
        self.cut = cut
        self.observations = 0

    def observe(self):
        self.observations += 1
        return self.cut

    def synchronize_target_builder(self, qpos_by_side):
        self.target_builder.set_current_qpos_pair(
            left_qpos=qpos_by_side["left"],
            right_qpos=qpos_by_side["right"],
            reset_tracking=True,
        )


def _cut() -> RealSkillMotionObservationCut:
    return RealSkillMotionObservationCut(
        observation=TeleopObservation(timestamp=2.0),
        qpos_by_side={
            "left": np.full(6, 0.1),
            "right": np.full(6, 0.2),
        },
        measured_at=1.9,
        available_at=2.0,
        evidence_refs=("rm65-hardware-qpos-cut:test",),
    )


def _runtime(builder: _TargetBuilder):
    calibration = build_annotated_dual_rm65_calibration()
    runtime = ExclusiveRM65SkillTargetSafetyRuntime.from_profile_path(
        target_builder=builder,
        calibration=calibration,
        envelope_profile_path=(
            CANONICAL_PROFILE_DIR
            / "agent_primitive_left_relative_tight_v1.json"
        ),
        execution_epoch=7,
        clock_domain_id="exclusive-real-skill-test-clock",
    )
    return runtime, calibration


def test_startup_binder_requires_hold_then_binds_fresh_active_anchor() -> None:
    builder = _TargetBuilder()
    runtime, calibration = _runtime(builder)
    ports = _Ports(
        target_builder=builder,
        cut=_cut(),
        hold_receipt=RealSkillMotionHoldReceipt(
            succeeded=True,
            evidence_refs=("rm65-skill-hold:test",),
        ),
    )
    binder = ExclusiveRM65SkillStartupSafetyBinder(
        ports=ports,
        target_safety=runtime,
        clock=lambda: 2.1,
    )

    receipt = binder()

    assert receipt is runtime.startup_receipt
    assert receipt.anchor_receipt_hash is not None
    assert receipt.hold_evidence_refs == ("rm65-skill-hold:test",)
    assert receipt.observation_evidence_refs == (
        "rm65-hardware-qpos-cut:test",
    )
    assert runtime.envelope_filter.apply_to_target is True
    assert (
        runtime.envelope_filter.anchor_receipt.semantic_content_hash
        == receipt.anchor_receipt_hash
    )
    np.testing.assert_allclose(
        calibration.get_arm_for_side("left")
        .home_pose_for_tcp_mode("tool_tcp")[:3, 3],
        np.array([0.1, 0.0, 0.0]),
    )
    assert ports.observations == 1
    assert len(builder.syncs) == 1

    with pytest.raises(RuntimeError, match="already bound"):
        binder()
    assert ports.observations == 1
    assert len(builder.syncs) == 1


def test_startup_binder_rejects_unconfirmed_hold_without_observing() -> None:
    builder = _TargetBuilder()
    runtime, _ = _runtime(builder)
    ports = _Ports(
        target_builder=builder,
        cut=_cut(),
        hold_receipt=RealSkillMotionHoldReceipt(
            succeeded=False,
            evidence_refs=(),
        ),
    )
    binder = ExclusiveRM65SkillStartupSafetyBinder(
        ports=ports,
        target_safety=runtime,
        clock=lambda: 2.1,
    )

    with pytest.raises(RuntimeError, match="was not confirmed"):
        binder()

    assert ports.observations == 0
    assert builder.syncs == []
    assert runtime.startup_receipt is None


@pytest.mark.parametrize(
    "name",
    (
        "agent_primitive_left_relative_tight_v1.json",
        "agent_primitive_right_relative_tight_v1.json",
    ),
)
def test_legacy_realteleop_profile_is_a_canonical_compatibility_mirror(
    name: str,
) -> None:
    canonical = json.loads((CANONICAL_PROFILE_DIR / name).read_text())
    legacy = json.loads((LEGACY_PROFILE_DIR / name).read_text())

    assert legacy == canonical


def test_target_safety_import_has_no_realteleop_or_composition_dependency() -> None:
    script = """
import json
import sys
import teleop_runtime.skill_deploy.rm65_target_safety
print(json.dumps(sorted(
    name for name in sys.modules
    if name == 'control_composition'
    or name.startswith('control_composition.')
    or name == 'teleop_runtime.real_teleop'
    or name.startswith('teleop_runtime.real_teleop.')
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
