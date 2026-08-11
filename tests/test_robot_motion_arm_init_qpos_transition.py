from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robot_motion.arm_init import (
    ArmQposTransitionConfig,
    load_qpos_by_side_file,
    plan_slow_qpos_transition,
)
from robot_motion.joint_limits import JointVelocityLimitProfile
from robot_motion.rm65.arm_init.qpos_init import main as rm65_qpos_init_main
from robot_motion.rm65.arm_init.schema import QposTargetSpec, save_qpos_target


def test_generic_slow_qpos_transition_plans_rate_limited_steps() -> None:
    profile = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(10.0, 10.0),
        max_dt_sec=1.0,
    )

    plan = plan_slow_qpos_transition(
        start_qpos_by_side={"arm": np.zeros(2)},
        target_qpos_by_side={"arm": np.deg2rad([25.0, -5.0])},
        config=ArmQposTransitionConfig(dt_sec=0.5, joint_velocity_profile=profile),
    )

    assert plan.completed
    assert len(plan.steps) == 5
    np.testing.assert_allclose(plan.steps[0].qpos_by_side["arm"], np.deg2rad([5.0, -5.0]))
    np.testing.assert_allclose(plan.final_qpos_by_side["arm"], np.deg2rad([25.0, -5.0]))


@pytest.mark.integration
def test_common_qpos_loader_accepts_target_schema(tmp_path: Path) -> None:
    path = tmp_path / "target.json"
    save_qpos_target(QposTargetSpec.from_qpos_by_side({"right": np.ones(6) * 0.1}), path)

    loaded = load_qpos_by_side_file(path)

    np.testing.assert_allclose(loaded["right"], np.ones(6) * 0.1)


@pytest.mark.integration
def test_rm65_qpos_init_cli_writes_slow_plan_from_target_schema(tmp_path: Path) -> None:
    current_path = tmp_path / "current.json"
    target_path = tmp_path / "target.json"
    output_path = tmp_path / "plan.json"
    current_path.write_text(
        json.dumps({"qpos_by_side": {"right": np.zeros(6).tolist()}}) + "\n",
        encoding="utf-8",
    )
    save_qpos_target(
        QposTargetSpec.from_qpos_by_side({"right": np.deg2rad([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])}),
        target_path,
    )

    code = rm65_qpos_init_main(
        [
            "--current-qpos",
            str(current_path),
            "--qpos-target",
            str(target_path),
            "--arm",
            "right",
            "--qpos-dt-sec",
            "0.1",
            "--output",
            str(output_path),
            "--include-steps",
        ]
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert code == 0
    assert payload["data_type"] == "robot_motion_rm65_slow_qpos_init_plan"
    assert payload["summary"]["completed"]
    assert payload["steps"]
