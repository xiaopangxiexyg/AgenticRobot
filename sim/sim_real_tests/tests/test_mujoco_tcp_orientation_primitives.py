from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("pinocchio")
pytest.importorskip("pink")
pytest.importorskip("qpsolvers")

from robot_motion.primitives import LEVEL_TCP_ORIENTATION_PRIMITIVES
from sim.mujoco.replay.level_tcp_orientation_visualization import (
    LevelTcpOrientationVisualizer,
)
from sim.sim_real_tests.mujoco import (
    LevelTcpOrientationClosedLoopConfig,
    build_level_tcp_orientation_world_trajectory,
    run_level_tcp_orientation_primitives_active_closed_loop,
)
from sim.sim_real_tests.mujoco.view_tcp_orientation_primitives import (
    build_parser,
)


pytestmark = [pytest.mark.integration, pytest.mark.optional]


class _FakeViewerHandle:
    def __init__(self, user_scene) -> None:
        self.user_scn = user_scene

    def lock(self):
        return nullcontext()


def _anchors() -> dict[str, np.ndarray]:
    left = np.eye(4, dtype=float)
    right = np.eye(4, dtype=float)
    left[:3, 3] = (0.3, 0.2, 0.5)
    right[:3, 3] = (0.3, -0.2, 0.5)
    return {"left": left, "right": right}


def _world_from_desk() -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.diag((-1.0, -1.0, 1.0))
    pose[:3, 3] = (-0.472, 0.0, 0.01)
    return pose


def test_world_trajectory_holds_tcp_position_and_executes_left_then_right() -> None:
    anchors = _anchors()
    config = LevelTcpOrientationClosedLoopConfig(
        samples_per_leg=2,
        endpoint_hold_frames=1,
        anchor_hold_frames=1,
    )
    frames = build_level_tcp_orientation_world_trajectory(
        anchor_pose_world_by_side=anchors,
        world_from_desk=_world_from_desk(),
        config=config,
    )

    assert len(frames) == 1 + 2 * 3 * (2 + 1 + 2 + 1)
    active_sides = tuple(
        dict.fromkeys(
            frame["active_side"] for frame in frames if frame["active_side"] is not None
        )
    )
    assert active_sides == ("left", "right")
    for frame in frames:
        for side in ("left", "right"):
            np.testing.assert_allclose(
                frame["target_pose_world_by_side"][side][:3, 3],
                anchors[side][:3, 3],
                rtol=0.0,
                atol=1e-12,
            )

    desk_from_world = np.linalg.inv(_world_from_desk())
    endpoints = [frame for frame in frames if frame["phase"] == "target_hold"]
    assert len(endpoints) == 6
    for frame in endpoints:
        primitive = next(
            item
            for item in LEVEL_TCP_ORIENTATION_PRIMITIVES
            if item.heading.value == frame["heading"]
        )
        target_desk = (
            desk_from_world @ frame["target_pose_world_by_side"][frame["active_side"]]
        )
        np.testing.assert_allclose(
            target_desk[:3, :3],
            primitive.target_rotation_reference,
            rtol=0.0,
            atol=1e-12,
        )


def test_visualizer_draws_three_headings_for_both_arms() -> None:
    anchors = _anchors()
    actual = {
        side: tuple(anchors[side].copy() for _ in range(3))
        for side in ("left", "right")
    }
    targets = tuple(
        {side: anchors[side].copy() for side in ("left", "right")} for _ in range(3)
    )
    visualizer = LevelTcpOrientationVisualizer(
        anchor_pose_world_by_side=anchors,
        world_from_desk=_world_from_desk(),
        actual_pose_world_by_side=actual,
        target_pose_world_by_frame=targets,
        active_side_by_frame=(None, "left", "right"),
        heading_by_frame=(None, "forward_left_30", "forward_right_30"),
    )
    model = mujoco.MjModel.from_xml_string("<mujoco model='orientation_viz'/>")
    user_scene = mujoco.MjvScene(model, 128)
    viewer = _FakeViewerHandle(user_scene)

    assert visualizer.update(viewer, frame_index=1) is True
    assert user_scene.ngeom == visualizer.geom_count
    labels = {str(user_scene.geoms[index].label) for index in range(user_scene.ngeom)}
    assert "left_orientation_anchor" in labels
    assert "right_orientation_anchor" in labels
    for side in ("left", "right"):
        for primitive in LEVEL_TCP_ORIENTATION_PRIMITIVES:
            assert f"{side}_{primitive.heading.value}_tool_plus_z" in labels
    assert "active_left_forward_left_30_target_tool_plus_z" in labels
    assert "left_actual_tool_plus_z" in labels
    assert "right_actual_tool_plus_z" in labels


@pytest.mark.slow_sim
def test_both_arms_execute_every_orientation_in_physical_mujoco_chain() -> None:
    result = run_level_tcp_orientation_primitives_active_closed_loop()

    assert result.passed, result.to_dict()
    assert result.metrics["reference_hz"] == pytest.approx(60.0)
    assert result.metrics["command_hz"] == pytest.approx(120.0)
    assert result.metrics["physics_hz"] == pytest.approx(1000.0)
    assert result.metrics["arm_primitive_execution_count"] == 6
    assert result.metrics["ik_failure_count"] == 0
    assert result.metrics["nonfinite_qcmd_count"] == 0
    assert result.metrics["max_target_translation_drift_m"] < 1e-12
    assert result.metrics["max_qcmd_step_deg"] <= 0.6075 + 1e-6
    for side in ("left", "right"):
        for primitive in LEVEL_TCP_ORIENTATION_PRIMITIVES:
            prefix = f"{side}_{primitive.heading.value}"
            assert result.metrics[f"{prefix}_final_position_error_m"] < 0.002
            assert result.metrics[f"{prefix}_final_orientation_error_deg"] < 0.25
            assert result.metrics[f"{prefix}_final_level_error_deg"] < 0.15
        assert result.metrics[f"{side}_final_anchor_position_error_m"] < 0.002
        assert result.metrics[f"{side}_final_anchor_orientation_error_deg"] < 0.25


def test_viewer_defaults_to_smooth_complete_repeating_playback() -> None:
    args = build_parser().parse_args([])

    assert args.samples_per_leg == 120
    assert args.endpoint_hold_frames == 30
    assert args.anchor_hold_frames == 30
    assert args.speed == pytest.approx(1.0)
    assert args.repeat == 0
