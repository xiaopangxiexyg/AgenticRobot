from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("pinocchio")
pytest.importorskip("pink")
pytest.importorskip("qpsolvers")

from robot_motion.primitives import (
    CARDINAL_TCP_SHORT_STEP_PRIMITIVES,
)
from sim.mujoco.replay.cardinal_tcp_displacement_visualization import (
    CardinalTcpPrimitiveVisualizer,
)
from sim.sim_real_tests.mujoco import (
    CardinalTcpPrimitiveClosedLoopConfig,
    build_cardinal_tcp_primitive_world_trajectory,
    run_cardinal_tcp_primitives_active_closed_loop,
)
from sim.sim_real_tests.mujoco.view_tcp_displacement_primitives import (
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


def test_world_trajectory_executes_left_then_right_with_exact_3cm_endpoints() -> None:
    config = CardinalTcpPrimitiveClosedLoopConfig(
        samples_per_leg=2,
        endpoint_hold_frames=1,
        anchor_hold_frames=1,
    )
    anchors = _anchors()
    frames = build_cardinal_tcp_primitive_world_trajectory(
        anchor_pose_world_by_side=anchors,
        world_from_desk=_world_from_desk(),
        config=config,
    )

    assert len(frames) == 1 + 2 * 6 * (2 + 1 + 2 + 1)
    active_sides = tuple(
        dict.fromkeys(
            frame["active_side"]
            for frame in frames
            if frame["active_side"] is not None
        )
    )
    assert active_sides == ("left", "right")

    endpoint_frames = [
        frame for frame in frames if frame["phase"] == "target_hold"
    ]
    assert len(endpoint_frames) == 12
    for frame in endpoint_frames:
        side = frame["active_side"]
        direction = frame["direction"]
        primitive = next(
            item
            for item in CARDINAL_TCP_SHORT_STEP_PRIMITIVES
            if item.direction.value == direction
        )
        target = frame["target_pose_world_by_side"][side]
        np.testing.assert_allclose(
            target[:3, 3] - anchors[side][:3, 3],
            _world_from_desk()[:3, :3]
            @ primitive.delta_translation_reference_m,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_array_equal(target[:3, :3], anchors[side][:3, :3])
        inactive = "right" if side == "left" else "left"
        np.testing.assert_array_equal(
            frame["target_pose_world_by_side"][inactive],
            anchors[inactive],
        )


def test_visualizer_draws_both_primitive_stars_and_actual_trails() -> None:
    anchors = _anchors()
    actual = {
        side: tuple(
            np.asarray(anchors[side], dtype=float).copy() for _ in range(3)
        )
        for side in ("left", "right")
    }
    actual["left"][1][:3, 3] += (0.01, 0.0, 0.0)
    actual["left"][2][:3, 3] += (0.02, 0.0, 0.0)
    actual["right"][1][:3, 3] += (0.0, -0.01, 0.0)
    actual["right"][2][:3, 3] += (0.0, -0.02, 0.0)
    target_frames = tuple(
        {side: anchors[side].copy() for side in ("left", "right")}
        for _ in range(3)
    )
    target_frames[1]["left"][:3, 3] += (0.03, 0.0, 0.0)
    target_frames[2]["right"][:3, 3] += (0.0, -0.03, 0.0)
    visualizer = CardinalTcpPrimitiveVisualizer(
        anchor_pose_world_by_side=anchors,
        world_from_desk=_world_from_desk(),
        actual_pose_world_by_side=actual,
        target_pose_world_by_frame=target_frames,
        active_side_by_frame=(None, "left", "right"),
        direction_by_frame=(None, "forward", "right"),
    )
    model = mujoco.MjModel.from_xml_string("<mujoco model='primitive_viz'/>")
    user_scene = mujoco.MjvScene(model, 256)
    viewer = _FakeViewerHandle(user_scene)

    assert visualizer.update(viewer, frame_index=1) is True
    assert user_scene.ngeom == visualizer.geom_count
    labels = {
        str(user_scene.geoms[index].label)
        for index in range(user_scene.ngeom)
    }
    assert "left_tcp_primitive_anchor" in labels
    assert "right_tcp_primitive_anchor" in labels
    for side in ("left", "right"):
        for primitive in CARDINAL_TCP_SHORT_STEP_PRIMITIVES:
            assert f"{side}_{primitive.direction.value}_3cm" in labels
    assert "active_left_forward_target" in labels
    assert "left_actual_tcp" in labels
    assert "right_actual_tcp" in labels


def test_station_desk_axes_reverse_world_xy_but_not_world_z() -> None:
    anchors = _anchors()
    frames = build_cardinal_tcp_primitive_world_trajectory(
        anchor_pose_world_by_side=anchors,
        world_from_desk=_world_from_desk(),
        config=CardinalTcpPrimitiveClosedLoopConfig(
            samples_per_leg=1,
            endpoint_hold_frames=1,
            anchor_hold_frames=1,
        ),
    )
    endpoints = {
        (frame["active_side"], frame["direction"]): np.asarray(
            frame["target_pose_world_by_side"][frame["active_side"]],
            dtype=float,
        )[:3, 3]
        for frame in frames
        if frame["phase"] == "target_hold"
    }
    left_anchor = anchors["left"][:3, 3]
    np.testing.assert_allclose(
        endpoints[("left", "forward")] - left_anchor,
        (-0.03, 0.0, 0.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        endpoints[("left", "backward")] - left_anchor,
        (0.03, 0.0, 0.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        endpoints[("left", "left")] - left_anchor,
        (0.0, -0.03, 0.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        endpoints[("left", "right")] - left_anchor,
        (0.0, 0.03, 0.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        endpoints[("left", "up")] - left_anchor,
        (0.0, 0.0, 0.03),
        atol=1e-12,
    )


@pytest.mark.slow_sim
def test_both_arms_execute_every_primitive_in_physical_mujoco_chain() -> None:
    result = run_cardinal_tcp_primitives_active_closed_loop()

    assert result.passed, result.to_dict()
    assert result.metrics["reference_hz"] == pytest.approx(60.0)
    assert result.metrics["command_hz"] == pytest.approx(120.0)
    assert result.metrics["physics_hz"] == pytest.approx(1000.0)
    assert result.metrics["arm_primitive_execution_count"] == 12
    assert result.metrics["ik_failure_count"] == 0
    assert result.metrics["nonfinite_qcmd_count"] == 0
    assert result.metrics["max_qcmd_step_deg"] <= 0.6075 + 1e-6
    for side in ("left", "right"):
        for primitive in CARDINAL_TCP_SHORT_STEP_PRIMITIVES:
            prefix = f"{side}_{primitive.direction.value}"
            assert result.metrics[
                f"{prefix}_peak_actual_projection_m"
            ] > 0.025
            assert result.metrics[f"{prefix}_final_endpoint_error_m"] < 0.003
        assert result.metrics[f"{side}_final_anchor_error_m"] < 0.003


def test_viewer_defaults_to_complete_slow_repeating_playback() -> None:
    args = build_parser().parse_args([])

    assert args.samples_per_leg == 18
    assert args.endpoint_hold_frames == 8
    assert args.anchor_hold_frames == 8
    assert args.speed == pytest.approx(0.65)
    assert args.repeat == 0
