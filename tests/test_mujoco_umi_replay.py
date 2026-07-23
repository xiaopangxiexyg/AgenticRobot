from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco.viewer")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from robot_motion.pose_frame_profiles import (
    DEFAULT_RM65_DUAL_ARM_COLLECTION_POSE_FRAME_CONTRACT_PATH,
)
from robot_motion.rm65 import (
    DEFAULT_COLLISION_SCENE,
    DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS,
    LEGACY_POSITION_CONTROL_PROFILE_NAME,
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_BIMANUAL_POSITION_V1_QREF_PROFILE,
    build_rm65_dual_arm_scene_profile,
)
from sim.contracts import TimingSpec
from sim.mujoco.control.desk_tcp_position import (
    DeskTcpPositionControlConfig,
    DualArmCollisionAwareRetargeter,
    _replay_ik_tuning,
    position_15_60_120_timing,
)
from sim.mujoco.replay.umi_dataset import load_umi_replay_episode
from sim.mujoco.replay.tcp_target_visualization import (
    TcpTargetVisualizationConfig,
    TcpTargetVisualizer,
)
from sim.mujoco.replay.__main__ import main as package_replay_main
from sim.mujoco.replay.umi_runner import main as canonical_replay_main
from sim.mujoco.replay.umi_runner import parse_args
from sim.mujoco.viewer import passive_viewer_handle


class _FakeViewerHandle:
    def __init__(self, user_scene) -> None:
        self.user_scn = user_scene

    def lock(self):
        return nullcontext()


def test_sim_replay_defaults_match_formal_position_profile() -> None:
    config = DeskTcpPositionControlConfig()

    assert config.solve_min_iterations == 2
    assert config.solve_max_iterations == 4
    assert config.solve_adaptive_iterations_enabled is True
    assert config.solve_iteration_reserve_ms == pytest.approx(5.0)
    assert config.damping == pytest.approx(0.03)
    assert config.orientation_weight == pytest.approx(0.35)
    assert config.position_control_profile_name == RM65_BIMANUAL_POSITION_V1_NAME
    assert config.joint_velocity_limit_profile == RM65_BIMANUAL_POSITION_V1_QREF_PROFILE
    assert config.post_ik_qpos_smoothing_alpha == pytest.approx(1.0)
    assert config.post_ik_qpos_smoothing_adaptive_enabled is False

    weights = _replay_ik_tuning(config.orientation_weight).objective_weights()
    assert weights.single_arm_position_weight == pytest.approx(1.0)
    assert weights.single_arm_orientation_weight == pytest.approx(0.35)
    assert weights.dual_arm_position_cost == pytest.approx(1.0)
    assert weights.dual_arm_orientation_cost == pytest.approx(0.35)


def test_sim_replay_legacy_profile_is_explicit_rollback() -> None:
    args = parse_args(["--position-control-profile", LEGACY_POSITION_CONTROL_PROFILE_NAME])
    assert args.position_control_profile == LEGACY_POSITION_CONTROL_PROFILE_NAME


def test_passive_viewer_is_closed_before_scene_can_be_released(monkeypatch) -> None:
    events: list[str] = []

    class FakeHandle:
        def close(self) -> None:
            events.append("viewer_close")

        def is_running(self) -> bool:
            events.append("viewer_drained")
            return False

    class FakeScene:
        model = object()
        data = object()

    handle = FakeHandle()
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *args, **kwargs: handle)

    with passive_viewer_handle(FakeScene()) as yielded:
        assert yielded is handle
        events.append("replay_done")
    events.append("scene_release")

    assert events == [
        "replay_done",
        "viewer_close",
        "viewer_drained",
        "scene_release",
    ]


def _write_dataset(root: Path, *, action_dim: int = 16) -> None:
    if action_dim not in (16, 26):
        raise ValueError("test fixture only supports canonical 16D/26D actions")
    contract = json.loads(
        DEFAULT_RM65_DUAL_ARM_COLLECTION_POSE_FRAME_CONTRACT_PATH.read_text(
            encoding="utf-8"
        )
    )
    pose_hash = "sha256:test-pose"
    layout_hash = "sha256:test-layout"
    info = {
        "nominal_fps": 15.0,
        "policy_io_layout_hash": layout_hash,
        "pose_frame_contract": contract,
        "pose_frame_contract_hash": pose_hash,
        "arm_pose_action_mode": "absolute",
        "arm_action_specs_by_side": {
            side: {
                "controlled_frame": "tool_tcp",
                "reference_frame": "desk",
                "representation": "xyz_quat",
                "dof": 7,
            }
            for side in ("left", "right")
        },
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    tactile_paths = {}
    for side in ("left", "right"):
        relative = Path(f"tactile/{side}/episode_000000/manifest.json")
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "format": "binary_stream_v1",
                    "streams": {
                        "sensor": {
                            "encoding": "chunked_zstd",
                            "compression": "zstd",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        tactile_paths[side] = str(relative)

    left = [-0.30, 0.20, 0.18, 0.0, 0.0, 0.0, 1.0]
    right = [-0.30, -0.20, 0.18, 0.0, 0.0, 0.0, 1.0]
    rows = []
    for index, timestamp in enumerate((100.0, 100.08)):
        hand_action = (
            [0.1 * index, 0.2 * index]
            if action_dim == 16
            else [
                0.1 * index,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.2 * index,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        rows.append(
            {
                "frame_index": index,
                "timestamp": timestamp,
                "action": left + right + hand_action,
                "policy_io_layout_hash": layout_hash,
                "dexteleop.pose_storage_frame": "desk_absolute",
                "dexteleop.pose_frame_contract_hash": pose_hash,
                "observation.hand.left.tactile_path": tactile_paths["left"],
                "observation.hand.right.tactile_path": tactile_paths["right"],
                "observation.hand.left.tactile_frame_index": index,
                "observation.hand.right.tactile_frame_index": index,
            }
        )
    parquet = root / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), parquet)


def test_position_timing_preserves_scene_clocks() -> None:
    timing = position_15_60_120_timing(
        TimingSpec(physics_hz=1000.0, camera_hz=29.97)
    )
    assert (timing.sample_hz, timing.ik_hz, timing.command_hz) == (15.0, 60.0, 120.0)
    assert timing.physics_hz == 1000.0
    assert timing.camera_hz == 29.97


def test_position_replay_uses_shared_collision_aware_retargeter() -> None:
    assert DualArmCollisionAwareRetargeter.__module__ == (
        "robot_motion.rm65.dual_arm_target_solver"
    )


def test_replay_collision_profile_is_real_teleop_12_pair_profile() -> None:
    profile = build_rm65_dual_arm_scene_profile(DEFAULT_COLLISION_SCENE)

    assert profile.name == "dual_arm_hand_proxy_open"
    assert profile.collision_pairs == DUAL_ARM_HAND_PROXY_CHECKED_COLLISION_PAIRS
    assert len(profile.collision_pairs) == 12


def test_tcp_target_visualization_cli_defaults_on_and_can_be_disabled() -> None:
    assert parse_args([]).tcp_target_visualization is True
    assert (
        parse_args(["--no-tcp-target-visualization"]).tcp_target_visualization
        is False
    )


def test_package_entrypoint_is_canonical_cartesian_replay() -> None:
    assert package_replay_main is canonical_replay_main


def test_tcp_target_visualizer_updates_fixed_passive_overlay_slots() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco model='tcp_target_viz'/>")
    user_scene = mujoco.MjvScene(model, 32)
    viewer = _FakeViewerHandle(user_scene)
    visualizer = TcpTargetVisualizer()
    left = np.eye(4, dtype=float)
    left[:3, 3] = [0.10, 0.20, 0.30]
    right = np.eye(4, dtype=float)
    right[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    right[:3, 3] = [-0.10, -0.20, 0.25]

    assert visualizer.update(viewer, {"left": left, "right": right}) is True
    assert user_scene.ngeom == 8
    assert [geom.label for geom in user_scene.geoms[:8]] == [
        "left_tcp_target",
        "left_tcp_target_x",
        "left_tcp_target_y",
        "left_tcp_target_z",
        "right_tcp_target",
        "right_tcp_target_x",
        "right_tcp_target_y",
        "right_tcp_target_z",
    ]
    np.testing.assert_allclose(user_scene.geoms[0].pos, left[:3, 3])
    np.testing.assert_allclose(user_scene.geoms[4].pos, right[:3, 3])
    np.testing.assert_allclose(
        user_scene.geoms[1].mat.reshape(3, 3)[:, 2], left[:3, 0], atol=1e-7
    )
    np.testing.assert_allclose(
        user_scene.geoms[5].mat.reshape(3, 3)[:, 2], right[:3, 0], atol=1e-7
    )
    assert user_scene.geoms[1].size[2] == pytest.approx(0.08)

    moved_left = left.copy()
    moved_left[:3, 3] += [0.01, 0.02, -0.03]
    assert visualizer.update(
        viewer, {"left": moved_left, "right": right}
    ) is True
    assert user_scene.ngeom == 8
    np.testing.assert_allclose(user_scene.geoms[0].pos, moved_left[:3, 3])


def test_disabled_tcp_target_visualizer_does_not_touch_viewer_scene() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco model='tcp_target_viz_off'/>")
    user_scene = mujoco.MjvScene(model, 8)
    visualizer = TcpTargetVisualizer(
        TcpTargetVisualizationConfig(enabled=False)
    )

    assert visualizer.update(
        _FakeViewerHandle(user_scene), {"left": np.eye(4), "right": np.eye(4)}
    ) is False
    assert user_scene.ngeom == 0


def test_tcp_target_visualizer_cannot_mutate_mujoco_physics_state() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco model="tcp_target_viz_physics_isolation">
          <worldbody>
            <body>
              <joint name="probe" type="hinge"/>
              <geom type="sphere" size="0.01"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qpos[0] = 0.37
    data.qvel[0] = -0.12
    data.time = 4.25
    before = (data.qpos.copy(), data.qvel.copy(), float(data.time))
    user_scene = mujoco.MjvScene(model, 16)

    TcpTargetVisualizer().update(
        _FakeViewerHandle(user_scene),
        {"left": np.eye(4), "right": np.eye(4)},
    )

    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.qvel, before[1])
    assert data.time == before[2]


def test_umi_loader_keeps_desk_pose_and_manifest_storage(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    episode = load_umi_replay_episode(tmp_path, 0)

    assert len(episode.frames) == 2
    assert episode.action_dim == 16
    assert episode.pose_frame_contract.pose_storage_frame == "desk_absolute"
    assert episode.tactile_storage_by_side == {
        "left": "chunked_zstd/zstd",
        "right": "chunked_zstd/zstd",
    }
    np.testing.assert_allclose(
        episode.frames[0].pose_desk_by_side["left"][:3, 3], [-0.30, 0.20, 0.18]
    )
    targets = episode.motion_targets()
    np.testing.assert_allclose(
        [target.timestamp for target in targets], [0.0, 0.08], atol=1e-12
    )
    grippers = episode.interpolated_gripper_targets(
        left_source_index=0, right_source_index=1, alpha=0.5
    )
    assert grippers == {"left": 0.05, "right": 0.1}


def test_canonical_loader_accepts_real_teleop_26d_action_without_qpos(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path, action_dim=26)

    episode = load_umi_replay_episode(tmp_path, 0)

    assert episode.action_dim == 26
    np.testing.assert_allclose(
        episode.frames[1].pose_desk_by_side["left"][:3, 3],
        [-0.30, 0.20, 0.18],
    )
    assert episode.frames[1].gripper_close_by_side == {
        "left": pytest.approx(0.1),
        "right": pytest.approx(0.2),
    }
