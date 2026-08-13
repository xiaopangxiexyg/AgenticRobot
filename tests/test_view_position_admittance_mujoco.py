from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest


mujoco = pytest.importorskip("mujoco")

from robot_motion.position_control.cartesian_admittance import (
    CartesianAdmittanceConfig,
)
from teleop_runtime.debug.replay_position_admittance_mujoco import (
    _PhysicalRollout,
    _admittance_config_for_envelope,
    _force_release_transients,
)
from teleop_runtime.debug.view_position_admittance_mujoco import (
    ForceControlResultVisualizer,
    ForceControlVisualizationConfig,
    _compact_report,
    _frame_status,
    parse_args,
    run,
)


class _FakeViewerHandle:
    def __init__(self, user_scene) -> None:
        self.user_scn = user_scene

    def lock(self):
        return nullcontext()


def _pose(xyz: tuple[float, float, float]) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = xyz
    return pose


def _rollout(
    name: str,
    *,
    selected_qpos: np.ndarray,
    selected_pose: np.ndarray,
    selected_qref: np.ndarray,
    selected_qcmd: np.ndarray,
    force: np.ndarray,
    offset: np.ndarray,
) -> _PhysicalRollout:
    zero_qpos = np.zeros(6, dtype=float)
    identity = np.eye(4, dtype=float)
    return _PhysicalRollout(
        name=name,
        report={},
        sample_times_sec=[12.25],
        measured_qpos={
            "left": [selected_qpos.copy()],
            "right": [zero_qpos.copy()],
        },
        measured_pose={
            "left": [selected_pose.copy()],
            "right": [identity.copy()],
        },
        qref={
            "left": [selected_qref.copy()],
            "right": [zero_qpos.copy()],
        },
        qcmd={
            "left": [selected_qcmd.copy()],
            "right": [zero_qpos.copy()],
        },
        raw_target_pose={
            "left": [identity.copy()],
            "right": [identity.copy()],
        },
        applied_target_pose={
            "left": [identity.copy()],
            "right": [identity.copy()],
        },
        filtered_force_control_n=[force.copy()],
        active_offset_m=[offset.copy()],
    )


def _release_rollout(
    name: str,
    *,
    physical_effect_m: tuple[float, ...],
    filtered_force_n: tuple[float, ...],
) -> _PhysicalRollout:
    count = len(physical_effect_m)
    assert len(filtered_force_n) == count
    times = [0.05 * index for index in range(count)]
    zero_qpos = np.zeros(6, dtype=float)
    identity = np.eye(4, dtype=float)
    return _PhysicalRollout(
        name=name,
        report={},
        sample_times_sec=times,
        measured_qpos={
            side: [zero_qpos.copy() for _ in range(count)]
            for side in ("left", "right")
        },
        measured_pose={
            "left": [_pose((value, 0.0, 0.0)) for value in physical_effect_m],
            "right": [identity.copy() for _ in range(count)],
        },
        qref={
            side: [zero_qpos.copy() for _ in range(count)]
            for side in ("left", "right")
        },
        qcmd={
            side: [zero_qpos.copy() for _ in range(count)]
            for side in ("left", "right")
        },
        raw_target_pose={
            side: [identity.copy() for _ in range(count)]
            for side in ("left", "right")
        },
        applied_target_pose={
            side: [identity.copy() for _ in range(count)]
            for side in ("left", "right")
        },
        filtered_force_control_n=[
            np.array([value, 0.0, 0.0], dtype=float)
            for value in filtered_force_n
        ],
        active_offset_m=[
            np.array([0.003 if index == 0 else 0.002, 0.0, 0.0])
            for index in range(count)
        ],
    )


def test_force_release_transient_gate_accepts_short_submillimeter_settling() -> None:
    baseline = _release_rollout(
        "baseline",
        physical_effect_m=(0.0, 0.0, 0.0, 0.0),
        filtered_force_n=(0.0, 0.0, 0.0, 0.0),
    )
    candidate = _release_rollout(
        "candidate",
        physical_effect_m=(0.0030, 0.0030, 0.0038, 0.0037),
        filtered_force_n=(2.0, 0.0, 0.0, 0.0),
    )

    result = _force_release_transients(
        baseline=baseline,
        candidate=candidate,
        selected_side="left",
        release_force_deadband_n=(0.45, 0.45, 0.45),
    )

    assert result["passed"] is True
    assert result["event_count"] == 1
    assert result["max_outward_catchup_m"] == pytest.approx(0.0008)
    assert result["max_outward_catchup_duration_sec"] == pytest.approx(0.05)


def test_force_release_transient_gate_rejects_delayed_outward_catchup() -> None:
    baseline = _release_rollout(
        "baseline",
        physical_effect_m=(0.0, 0.0, 0.0, 0.0),
        filtered_force_n=(0.0, 0.0, 0.0, 0.0),
    )
    candidate = _release_rollout(
        "candidate",
        physical_effect_m=(0.003, 0.003, 0.0042, 0.0060),
        filtered_force_n=(2.0, 0.0, 0.0, 0.0),
    )

    result = _force_release_transients(
        baseline=baseline,
        candidate=candidate,
        selected_side="left",
        release_force_deadband_n=(0.45, 0.45, 0.45),
    )

    assert result["passed"] is False
    assert result["event_count"] == 1
    assert result["max_outward_catchup_m"] == pytest.approx(0.003)
    assert result["events"][0]["passed"] is False


def test_force_release_transient_gate_ignores_submargin_settle_duration() -> None:
    baseline = _release_rollout(
        "baseline",
        physical_effect_m=(0.0, 0.0, 0.0, 0.0, 0.0),
        filtered_force_n=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    candidate = _release_rollout(
        "candidate",
        physical_effect_m=(0.0030, 0.0030, 0.0031, 0.0032, 0.0033),
        filtered_force_n=(2.0, 0.0, 0.0, 0.0, 0.0),
    )

    result = _force_release_transients(
        baseline=baseline,
        candidate=candidate,
        selected_side="left",
        release_force_deadband_n=(0.45, 0.45, 0.45),
    )

    assert result["passed"] is True
    assert result["max_outward_catchup_m"] == pytest.approx(0.0003)
    assert result["max_outward_catchup_duration_sec"] == pytest.approx(0.15)
    assert result["events"][0]["duration_relevant"] is False


def test_force_result_visualizer_uses_fixed_passive_overlay_slots() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco model='force_result_viz'/>")
    data = mujoco.MjData(model)
    data.time = 4.25
    before_time = float(data.time)
    user_scene = mujoco.MjvScene(model, 16)
    viewer = _FakeViewerHandle(user_scene)
    visualizer = ForceControlResultVisualizer(
        ForceControlVisualizationConfig(
            force_arrow_scale_m_per_n=0.02,
            modifier_arrow_scale=20.0,
            physical_delta_arrow_scale=10.0,
        )
    )
    baseline = _pose((0.10, 0.20, 0.30))
    active = _pose((0.101, 0.198, 0.303))
    raw = _pose((0.20, 0.30, 0.40))
    applied = _pose((0.21, 0.31, 0.41))

    assert visualizer.update(
        viewer,
        baseline_endpoint_pose=baseline,
        active_endpoint_pose=active,
        raw_target_pose=raw,
        applied_target_pose=applied,
        filtered_force_control_n=[1.0, 0.0, 0.0],
        active_offset_m=[0.001, -0.002, 0.0],
        selected_side="left",
    ) is True

    assert user_scene.ngeom == 6
    assert [geom.label for geom in user_scene.geoms[:6]] == [
        "left_baseline_measured_endpoint",
        "left_active_measured_endpoint",
        "left_active_applied_target",
        "left_filtered_force_solver_world",
        "left_modifier_x20",
        "left_physical_delta_x10",
    ]
    np.testing.assert_allclose(user_scene.geoms[0].pos, baseline[:3, 3])
    np.testing.assert_allclose(user_scene.geoms[1].pos, active[:3, 3])
    np.testing.assert_allclose(user_scene.geoms[2].pos, applied[:3, 3])
    assert user_scene.geoms[3].size[2] == pytest.approx(0.02)
    assert user_scene.geoms[4].size[2] == pytest.approx(
        np.linalg.norm([0.02, -0.04, 0.0])
    )
    assert user_scene.geoms[5].size[2] == pytest.approx(
        10.0 * np.linalg.norm(active[:3, 3] - baseline[:3, 3])
    )
    assert data.time == before_time


def test_force_result_visualizer_hides_zero_vectors() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco model='zero_force_viz'/>")
    user_scene = mujoco.MjvScene(model, 8)
    visualizer = ForceControlResultVisualizer()
    identity = np.eye(4, dtype=float)

    visualizer.update(
        _FakeViewerHandle(user_scene),
        baseline_endpoint_pose=identity,
        active_endpoint_pose=identity,
        raw_target_pose=identity,
        applied_target_pose=identity,
        filtered_force_control_n=np.zeros(3),
        active_offset_m=np.zeros(3),
        selected_side="right",
    )

    np.testing.assert_allclose(user_scene.geoms[3].rgba[3], 0.0)
    np.testing.assert_allclose(user_scene.geoms[4].rgba[3], 0.0)
    np.testing.assert_allclose(user_scene.geoms[5].rgba[3], 0.0)


def test_frame_status_separates_modifier_reference_command_and_measured() -> None:
    baseline = _rollout(
        "baseline",
        selected_qpos=np.zeros(6),
        selected_pose=_pose((0.10, 0.20, 0.30)),
        selected_qref=np.zeros(6),
        selected_qcmd=np.zeros(6),
        force=np.zeros(3),
        offset=np.zeros(3),
    )
    active = _rollout(
        "active",
        selected_qpos=np.deg2rad([0.25, 0.0, 0.0, 0.0, 0.0, 0.0]),
        selected_pose=_pose((0.101, 0.198, 0.302)),
        selected_qref=np.deg2rad([0.50, 0.0, 0.0, 0.0, 0.0, 0.0]),
        selected_qcmd=np.deg2rad([0.40, 0.0, 0.0, 0.0, 0.0, 0.0]),
        force=np.array([1.0, 2.0, 2.0]),
        offset=np.array([0.001, 0.0, 0.0]),
    )

    status = _frame_status(
        index=0,
        baseline=baseline,
        active=active,
        selected_side="left",
    )

    assert status["force_norm_n"] == pytest.approx(3.0)
    assert status["active_offset_norm_m"] == pytest.approx(0.001)
    assert status["qref_delta_max_deg"] == pytest.approx(0.50)
    assert status["qcmd_delta_max_deg"] == pytest.approx(0.40)
    assert status["measured_qpos_delta_max_deg"] == pytest.approx(0.25)
    assert status["measured_endpoint_delta_m"] == pytest.approx(0.003)


def test_recorded_shadow_envelope_preserves_saved_normal_limits() -> None:
    recorded = CartesianAdmittanceConfig(
        enabled_axes=(True, True, True),
        response_model="profile_compliance",
        max_offset_m=(0.08, 0.08, 0.08),
        max_velocity_m_s=(0.06, 0.06, 0.06),
        max_return_velocity_m_s=(0.03, 0.03, 0.03),
        max_output_slew_m_s=(0.06, 0.06, 0.06),
        stale_after_sec=0.15,
        max_force_dt_sec=0.15,
    )

    recorded_shadow = _admittance_config_for_envelope(
        recorded,
        admittance_envelope="recorded-shadow",
    )
    cautious = _admittance_config_for_envelope(
        recorded,
        admittance_envelope="cautious-active",
    )
    offline_normal = _admittance_config_for_envelope(
        CartesianAdmittanceConfig(
            enabled_axes=(False, False, False),
            response_model="profile_compliance",
            max_offset_m=(0.002, 0.002, 0.002),
            max_velocity_m_s=(0.005, 0.005, 0.005),
            max_return_velocity_m_s=(0.005, 0.005, 0.005),
            max_output_slew_m_s=(0.005, 0.005, 0.005),
        ),
        admittance_envelope="normal-parity-offline",
    )
    production_normal = _admittance_config_for_envelope(
        recorded,
        admittance_envelope="normal-parity-active",
    )

    assert recorded_shadow is recorded
    assert recorded_shadow.max_offset_m == (0.08, 0.08, 0.08)
    assert recorded_shadow.max_velocity_m_s == (0.06, 0.06, 0.06)
    assert recorded_shadow.max_return_velocity_m_s == (0.03, 0.03, 0.03)
    assert cautious.max_offset_m == (0.002, 0.002, 0.002)
    assert cautious.max_velocity_m_s == (0.005, 0.005, 0.005)
    assert offline_normal.enabled_axes == (True, True, True)
    assert offline_normal.max_offset_m == (0.08, 0.08, 0.08)
    assert offline_normal.max_velocity_m_s == (0.06, 0.06, 0.06)
    assert offline_normal.max_return_velocity_m_s == (0.03, 0.03, 0.03)
    assert offline_normal.max_output_slew_m_s == (0.06, 0.06, 0.06)
    assert production_normal.max_return_velocity_m_s == (0.42, 0.42, 0.42)
    assert production_normal.return_rate_s == pytest.approx(4.0)


def test_prepare_only_runs_evaluation_without_constructing_viewer(
    monkeypatch,
) -> None:
    forwarded: dict[str, object] = {}
    report = {
        "path": "/tmp/force.jsonl",
        "admittance_envelope": "cautious-active",
        "hardware_authorization": False,
        "safety_passed": True,
        "coverage_passed": True,
        "input": {"recorded_duration_sec": 1.0},
        "rollouts": {
            "zoh_active": {
                "completed_reference_ticks": 61,
                "physics_ticks": 1000,
                "active_offset_norm_m": {"max": 0.002},
                "final_active_offset_m": 0.0,
                "active_qref_motion_ticks": 5,
            }
        },
        "comparisons": {
            "zoh": {
                "selected_measured_translation_m": {"max": 0.001},
                "selected_measured_rotation_deg": {"max": 0.05},
                "opposite_measured_qpos_delta_rad": {"max": 0.0},
            }
        },
    }
    def fake_evaluation(*args, **kwargs):
        forwarded.update(kwargs)
        return report, {}

    monkeypatch.setattr(
        "teleop_runtime.debug.view_position_admittance_mujoco."
        "_replay_recorded_wrench_through_mujoco_evaluation",
        fake_evaluation,
    )
    monkeypatch.setattr(
        "teleop_runtime.debug.view_position_admittance_mujoco."
        "_build_visualization_scene",
        lambda path: pytest.fail("prepare-only must not construct a scene"),
    )
    args = parse_args(
        [
            "--log",
            "/tmp/force.jsonl",
            "--prepare-only",
            "--translation-output-follow-tau-sec",
            "0.025",
            "--translation-max-output-acceleration-m-s2",
            "1.2",
        ]
    )

    assert run(args) == report
    assert args.interpolation == "zoh"
    assert args.admittance_envelope == "cautious-active"
    assert Path(args.log) == Path("/tmp/force.jsonl")
    assert forwarded["translation_output_follow_tau_sec"] == pytest.approx(0.025)
    assert forwarded["rotation_output_follow_tau_sec"] is None
    assert forwarded[
        "translation_max_output_acceleration_m_s2"
    ] == pytest.approx(1.2)
    assert forwarded[
        "rotation_max_output_angular_acceleration_rad_s2"
    ] is None


def test_recorded_shadow_viewer_mode_is_explicit() -> None:
    args = parse_args(
        [
            "--log",
            "/tmp/force.jsonl",
            "--admittance-envelope",
            "recorded-shadow",
        ]
    )

    assert args.admittance_envelope == "recorded-shadow"


def test_compact_report_preserves_clock_mismatch_without_masking_it() -> None:
    report = {
        "path": "/tmp/force.jsonl",
        "admittance_envelope": "normal-parity-offline",
        "hardware_authorization": False,
        "safety_passed": False,
        "coverage_passed": False,
        "comparisons": {
            "zoh": {
                "passed": False,
                "safety_passed": False,
                "error": "baseline and Active MuJoCo sample clocks differ",
            }
        },
        "rollouts": {
            "zoh_active": {
                "completed_reference_ticks": 14,
                "physics_ticks": 28,
                "active_offset_norm_m": {"max": 0.001},
                "final_active_offset_m": 0.001,
                "active_rotation_offset_norm_rad": {"max": 0.002},
                "final_active_rotation_offset_rad": 0.002,
                "active_qref_motion_ticks": 10,
            }
        },
    }

    compact = _compact_report(report, mode="zoh")

    assert compact["comparison_error"] == (
        "baseline and Active MuJoCo sample clocks differ"
    )
    assert compact["measured_endpoint_delta_max_m"] is None
    assert compact["measured_endpoint_rotation_max_deg"] is None


def test_offline_normal_parity_viewer_mode_is_explicit() -> None:
    args = parse_args(
        [
            "--log",
            "/tmp/force.jsonl",
            "--admittance-envelope",
            "normal-parity-offline",
            "--rotation-envelope",
            "balanced-5deg",
        ]
    )

    assert args.admittance_envelope == "normal-parity-offline"
    assert args.rotation_envelope == "balanced-5deg"
