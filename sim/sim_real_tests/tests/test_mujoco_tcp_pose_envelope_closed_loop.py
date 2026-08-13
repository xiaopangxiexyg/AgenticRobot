from __future__ import annotations

import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("pink")
pytest.importorskip("qpsolvers")
pytest.importorskip("mujoco")

from sim.sim_real_tests.mujoco import (  # noqa: E402
    run_tcp_pose_envelope_active_closed_loop,
)


def test_active_tcp_pose_envelope_in_physical_position_chain() -> None:
    result = run_tcp_pose_envelope_active_closed_loop()

    assert result.passed, result.to_dict()
    assert result.metrics["source_hz"] == pytest.approx(15.0)
    assert result.metrics["reference_hz"] == pytest.approx(60.0)
    assert result.metrics["command_hz"] == pytest.approx(120.0)
    assert result.metrics["projection_count"] == result.metrics["reference_count"]
    assert result.metrics["modified_reference_count"] > 0
    assert result.metrics["translation_modified_reference_count"] > 0
    assert result.metrics["rotation_modified_reference_count"] > 0
    assert result.metrics["inward_release_count"] > 0
    assert result.metrics["max_raw_translation_component_m"] > 0.015
    assert result.metrics["max_raw_rotation_deg"] > 8.0
    assert result.metrics["max_raw_direction_angle_deg"] > 8.0
    assert result.metrics["max_effective_direction_angle_deg"] <= 8.0 + 1e-5
    assert result.metrics["max_actual_direction_angle_deg"] <= 8.0 + 1e-5
    assert result.metrics["max_actual_residual_m"] > 0.0
    assert result.metrics["ik_failure_count"] == 0
    assert result.metrics["nonfinite_qcmd_count"] == 0
    assert result.metrics["right_projection_change"] == pytest.approx(0.0)
    assert result.metrics["swept_tcp_certified"] is False
