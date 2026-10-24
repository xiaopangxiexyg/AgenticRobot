"""Offline sampled geometry evidence, not hardware or dynamic acceptance."""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from control_station.b601_torque_console.automatic_calibration import (
    AutomaticCalibrationConfig,
    load_automatic_profile,
)
from control_station.b601_torque_console.workspace_calibration import (
    WorkspaceCalibrationConfig,
)
from robot_motion.rebot_b601_rs.model import B601_RS_URDF_RELATIVE_PATH
from scripts.probes.b601_workspace_collision_audit import (
    WorkspaceGeometryAudit,
    audit_plan,
    nominal_segments,
)

pytestmark = [pytest.mark.contract, pytest.mark.optional]
ROOT = Path(os.environ.get("B601_REBOT_SOURCE_ROOT",
    str(Path(__file__).resolve().parents[1] / "hardware/reBotArm_control_py")))
ZERO = (0.0,) * 6


@pytest.fixture(scope="module")
def geometry():
    pytest.importorskip("mujoco")
    pytest.importorskip("hppfcl")
    if not (ROOT / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("pinned reBotArm source is absent")
    return WorkspaceGeometryAudit(ROOT)


def test_geometry_reproduces_actual_wide_collision_but_does_not_relabel_zero_clearance(geometry):
    zero = geometry.check(ZERO)
    assert zero["triangle_surface_collision_pairs"] == []
    assert 0 < zero["minimum_triangle_surface_distance_m"] < .002
    assert not zero["convex_check"]["safe"]
    unfolded = geometry.check((0, math.radians(3), math.radians(3), 0, 0, 0))
    assert unfolded["triangle_surface_collision_pairs"] == []
    assert unfolded["minimum_triangle_surface_distance_m"] >= .002
    collision = geometry.check((0, math.radians(115), math.radians(5), 0, 0, 0))
    assert ["base_link_collision_0", "gripper_end_collision_0"] in collision["triangle_surface_collision_pairs"]
    assert geometry.oracle.forbidden_pair_count == len(geometry.oracle.forbidden_geom_pairs())
    assert zero["ground_contacts"] == []


def test_supported_early_return_samples_existing_route_from_each_coverage_leg(geometry):
    # Nominal kinematic samples only: no support stand, cables, tracking error
    # or dynamic stopping certification is inferred from triangle clearance.
    profile = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j2_j3_coverage_support_v1.json"
    values = load_automatic_profile(profile)
    workspace = WorkspaceCalibrationConfig(**values.pop("workspace"))
    values.pop("cycle")
    values.pop("motion_profile")
    config = AutomaticCalibrationConfig(**values)
    from robot_motion.rebot_b601_rs.calibration import CalibrationCondition

    condition = CalibrationCondition("return_geometry", "a"*64, geometry.urdf_sha256,
        1, (1.,)*6, ZERO, ZERO, ZERO, .02, ZERO)
    samples = {}
    # Every nominal positioning/sweep endpoint can precede a soft stop. Return
    # is the same canonical J2-then-J3 route, not simultaneous interpolation.
    for _, _, end in nominal_segments(workspace, config, ZERO, condition):
        q = end
        for task in workspace.tasks(ZERO)[-3:-1]:
            j = int(task["joint"][-1])-1
            count = max(1, math.ceil(abs(task["target"]-q[j])/math.radians(.5)))
            for n in range(count+1):
                point = list(q)
                point[j] += (task["target"]-q[j])*n/count
                key = tuple(round(v, 12) for v in point)
                if key not in samples:
                    samples[key] = geometry.check(point)
            q = tuple(task["target"] if i == j else v for i, v in enumerate(q))
    assert samples and all(not item["triangle_surface_collision_pairs"] and not item["ground_contacts"]
                           for item in samples.values())
    # Keep the known conservative zero-origin clearance rejection visible.
    assert any(not item["convex_check"]["safe"] for item in samples.values())


def test_audit_checks_intermediate_paths_and_preserves_clearance_warnings():
    class ScriptedGeometry:
        urdf_sha256 = "a" * 64
        oracle = SimpleNamespace(policy=SimpleNamespace(minimum_clearance_m=.002))

        def check(self, q):
            # Endpoints are clear; a collision lies strictly inside a move.
            collision = math.radians(1.3) < q[2] < math.radians(1.7)
            return {"qpos_deg": [math.degrees(v) for v in q],
                "minimum_triangle_surface_distance_m": 0 if collision else .003,
                "triangle_surface_collision_pairs": [["a", "b"]] if collision else [],
                "ground_contacts": [], "convex_check": {"safe": not collision}}

    config = AutomaticCalibrationConfig(motion_plan="local_identification")
    result = audit_plan(ScriptedGeometry(), WorkspaceCalibrationConfig(), config, ZERO,
        max_step_deg=.5, profile_sha256="b" * 64)
    assert result["status"] == "COLLISIONS_DETECTED"
    assert result["hardware_authorized"] is False
    assert result["segments"][0]["first_surface_or_ground_collision"] is not None
    assert result["segments"][-1]["task"]["kind"] == "return"
    with pytest.raises(ValueError, match="at most 0.5"):
        audit_plan(ScriptedGeometry(), WorkspaceCalibrationConfig(), config, ZERO,
            max_step_deg=1, profile_sha256="b" * 64)


def test_audit_does_not_turn_no_intersection_into_sufficient_margin():
    class NarrowGeometry:
        urdf_sha256 = "a" * 64
        oracle = SimpleNamespace(policy=SimpleNamespace(minimum_clearance_m=.002))

        def check(self, q):
            return {"minimum_triangle_surface_distance_m": .0008,
                "triangle_surface_collision_pairs": [], "ground_contacts": [],
                "convex_check": {"safe": False}}

    result = audit_plan(NarrowGeometry(), WorkspaceCalibrationConfig(),
        AutomaticCalibrationConfig(motion_plan="local_identification"), ZERO, profile_sha256="b" * 64)
    assert result["status"] == "CLEARANCE_REVIEW_REQUIRED"
    assert result["surface_or_ground_collision_samples"] == 0
    assert result["below_preventive_margin_samples"] > 0
    assert result["hardware_authorized"] is False
