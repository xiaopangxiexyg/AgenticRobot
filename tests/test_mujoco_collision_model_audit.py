from __future__ import annotations

import argparse

import pytest


def test_collision_model_audit_cli_requires_exact_task_profile() -> None:
    from sim.mujoco.diagnostics.collision_model import build_arg_parser

    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--task-config",
            "sim/configs/tasks/rm65_right_cube_pick_goal_delta_xyz_gripper_v2.json",
        ]
    )
    assert isinstance(args, argparse.Namespace)


@pytest.mark.optional
def test_collision_model_audit_reports_compiled_geometry_and_solver_facts() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")
    from sim.mujoco.diagnostics.collision_model import (
        COLLISION_AUDIT_SCHEMA,
        run,
    )

    report = run(
        argparse.Namespace(
            task_config=(
                "sim/configs/tasks/"
                "rm65_right_cube_pick_goal_delta_xyz_gripper_v2.json"
            ),
            seed=3,
            model_cache_dir=None,
        )
    )
    assert report["schema"] == COLLISION_AUDIT_SCHEMA
    model = report["model"]
    assert model["collision_geom_count"] > 0
    assert model["visual_only_geom_count"] > 0
    assert model["forbidden_pair_count"] > 0
    type_counts = report["collision_geom_type_counts"]
    assert sum(sum(group.values()) for group in type_counts.values()) == model[
        "collision_geom_count"
    ]
    assert type_counts["object"]["mjGEOM_BOX"] == 1
    solver = report["solver"]
    assert solver["timestep_sec"] == pytest.approx(0.001)
    assert solver["integrator"] == "mjINT_IMPLICITFAST"
    if solver["multiccd_runtime_flag_available"]:
        assert solver["multiccd_enabled"] is True
    assert report["contact_material_profiles"]["object"]
