from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.mujoco.replay.tcp_pose_absolute_workspace_visualization import (  # noqa: E402
    AbsoluteWorkspaceVisualizer,
    build_review_trajectory,
    derive_absolute_workspace,
    load_absolute_workspace_preview_config,
    trajectory_to_world,
)
from sim.sim_real_tests.mujoco.view_tcp_pose_absolute_workspace import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    _generate_closed_loop_playback,
    build_parser,
)
from sim.mujoco.control.desk_tcp_position import DeskTcpPositionControl  # noqa: E402


pytestmark = [pytest.mark.integration, pytest.mark.optional]


class _FakeViewerHandle:
    def __init__(self, user_scene) -> None:
        self.user_scn = user_scene

    def lock(self):
        return nullcontext()


def test_urdf_derived_absolute_workspace_candidates_have_expected_dimensions() -> None:
    config = load_absolute_workspace_preview_config(DEFAULT_CONFIG_PATH)
    loose_workspace = derive_absolute_workspace(config, "absolute_loose")
    standard_workspace = derive_absolute_workspace(config, "absolute_standard")
    tight_workspace = derive_absolute_workspace(config, "absolute_tight")

    assert loose_workspace.table_min_xyz_m == pytest.approx((-0.412, -0.65, -0.05))
    assert loose_workspace.table_max_xyz_m == pytest.approx((0.412, 0.65, 0.0))
    assert loose_workspace.pole_top_z_m == pytest.approx(0.99)
    assert loose_workspace.absolute_min_xyz_m == pytest.approx(
        (-0.3708, -0.585, 0.0)
    )
    assert loose_workspace.absolute_max_xyz_m == pytest.approx(
        (0.3708, 0.585, 0.693)
    )
    assert standard_workspace.absolute_min_xyz_m == pytest.approx(
        (-0.22248, -0.468, 0.02)
    )
    assert standard_workspace.absolute_max_xyz_m == pytest.approx(
        (0.3708, 0.468, 0.5544)
    )
    assert standard_workspace.absolute_max_xyz_m[0] == pytest.approx(
        loose_workspace.absolute_max_xyz_m[0]
    )
    assert standard_workspace.size_xyz_m[:2] == pytest.approx(
        0.8 * loose_workspace.size_xyz_m[:2]
    )
    assert standard_workspace.absolute_min_xyz_m[2] == pytest.approx(0.02)
    assert tight_workspace.absolute_min_xyz_m == pytest.approx(
        (-0.0412, -0.325, 0.08)
    )
    assert tight_workspace.absolute_max_xyz_m == pytest.approx(
        (0.3708, 0.325, 0.396)
    )
    assert tight_workspace.absolute_max_xyz_m[0] == pytest.approx(
        loose_workspace.absolute_max_xyz_m[0]
    )


@pytest.mark.parametrize(
    "profile_name",
    ["absolute_loose", "absolute_standard", "absolute_tight"],
)
def test_review_trajectory_uses_real_projector_and_never_exceeds_bounds(
    profile_name: str,
) -> None:
    config = load_absolute_workspace_preview_config(DEFAULT_CONFIG_PATH)
    workspace = derive_absolute_workspace(config, profile_name)
    raw, effective = build_review_trajectory(workspace, samples_per_leg=8)
    lower = np.asarray(workspace.absolute_min_xyz_m, dtype=float)
    upper = np.asarray(workspace.absolute_max_xyz_m, dtype=float)

    assert any(
        np.any(pose[:3, 3] < lower) or np.any(pose[:3, 3] > upper)
        for pose in raw
    )
    assert all(np.all(pose[:3, 3] >= lower - 1e-12) for pose in effective)
    assert all(np.all(pose[:3, 3] <= upper + 1e-12) for pose in effective)
    assert any(
        not np.array_equal(raw_pose, effective_pose)
        for raw_pose, effective_pose in zip(raw, effective)
    )


def test_absolute_workspace_visualizer_draws_volume_edges_paths_and_desk_axes() -> None:
    config = load_absolute_workspace_preview_config(DEFAULT_CONFIG_PATH)
    workspace = derive_absolute_workspace(config, "absolute_standard")
    raw_desk, effective_desk = build_review_trajectory(
        workspace,
        samples_per_leg=3,
    )
    visualizer = AbsoluteWorkspaceVisualizer(
        workspace=workspace,
        raw_pose_world=trajectory_to_world(workspace, raw_desk),
        effective_pose_world=trajectory_to_world(workspace, effective_desk),
        actual_pose_world=trajectory_to_world(workspace, effective_desk),
    )
    model = mujoco.MjModel.from_xml_string("<mujoco model='absolute_workspace'/>")
    user_scene = mujoco.MjvScene(model, 512)
    viewer = _FakeViewerHandle(user_scene)

    assert visualizer.update(viewer, frame_index=4) is True
    assert user_scene.ngeom == visualizer.geom_count
    labels = {str(user_scene.geoms[index].label) for index in range(user_scene.ngeom)}
    assert "absolute_standard_volume" in labels
    assert "absolute_standard_edge" in labels
    assert "absolute_workspace_raw_trail" in labels
    assert "absolute_workspace_effective_trail" in labels
    assert "absolute_workspace_actual_trail" in labels
    assert "absolute_workspace_actual_tcp" in labels
    assert "absolute_workspace_projection_residual" in labels
    assert {"desk_x_axis", "desk_y_axis", "desk_z_axis"} <= labels
    volume = next(
        user_scene.geoms[index]
        for index in range(user_scene.ngeom)
        if str(user_scene.geoms[index].label) == "absolute_standard_volume"
    )
    assert int(volume.type) == int(mujoco.mjtGeom.mjGEOM_BOX)
    assert 0.0 < float(volume.rgba[3]) < 1.0


def test_viewer_requires_explicit_candidate_selection() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    for profile_name in (
        "absolute_loose",
        "absolute_standard",
        "absolute_tight",
    ):
        assert parser.parse_args(["--profile", profile_name]).profile == profile_name


@pytest.mark.slow_sim
def test_closed_loop_playback_contains_real_arm_motion() -> None:
    config = load_absolute_workspace_preview_config(DEFAULT_CONFIG_PATH)
    workspace = derive_absolute_workspace(config, "absolute_standard")
    control = DeskTcpPositionControl.build()
    try:
        playback = _generate_closed_loop_playback(
            control=control,
            workspace=workspace,
            samples_per_leg=3,
        )
    finally:
        control.close()

    assert playback["max_qpos_motion_deg"] > 0.01
    assert len(playback["actual_qpos"]) == len(playback["actual_world"])
    assert sum(playback["status_counts"].values()) == len(playback["actual_qpos"])
