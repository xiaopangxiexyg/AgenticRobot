"""Offline sampled geometry evidence, not hardware or dynamic acceptance."""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from control_station.b601_torque_console.automatic_calibration import (
    AutomaticCalibrationConfig,
)
from control_station.b601_torque_console.workspace_calibration import (
    WorkspaceCalibrationConfig,
)
from robot_motion.rebot_b601_rs.model import B601_RS_URDF_RELATIVE_PATH
from scripts.probes.b601_workspace_collision_audit import (
    WorkspaceGeometryAudit,
    audit_plan,
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
