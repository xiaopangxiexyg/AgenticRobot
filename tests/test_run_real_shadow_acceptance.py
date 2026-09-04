from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_real_shadow_acceptance as acceptance
from robot_motion.scene_obstacles import ObstacleSceneSnapshot
from teleop_runtime.skill_deploy.rm65_profile import (
    ExclusiveRM65SkillDeploymentProfile,
    resolve_exclusive_rm65_skill_deployment_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _profile():
    return resolve_exclusive_rm65_skill_deployment_profile(
        ExclusiveRM65SkillDeploymentProfile(
            arm_ip_assignment_path=(
                REPO_ROOT / "teleop_runtime/configs/hardware/rm65_arm_ips/latest.json"
            ),
            tool_profile_path_by_side={
                "left": REPO_ROOT
                / "hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json",
                "right": REPO_ROOT
                / "hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json",
            },
            envelope_profile_path=(
                REPO_ROOT
                / "teleop_runtime/configs/skill_deploy/agent_primitive_left_relative_tight_v1.json"
            ),
        )
    )


def test_pose_cli_rejects_missing_orientation() -> None:
    args = acceptance.build_parser().parse_args(
        [
            "--arm-ip-assignment",
            "/tmp/arms.json",
            "--dmtac-assignment",
            "/tmp/dmtac.json",
            "--target-xyz-m",
            "0.1",
            "0.2",
            "0.3",
        ]
    )
    with pytest.raises(ValueError, match="orientation"):
        acceptance._pose_from_cli(args)


def test_scene_cli_builds_sorted_cubes_and_clearance() -> None:
    args = acceptance.build_parser().parse_args(
        [
            "--arm-ip-assignment",
            "/tmp/arms.json",
            "--dmtac-assignment",
            "/tmp/dmtac.json",
            "--target-from-current",
            "--obstacle-clearance-m",
            "0.03",
            "--obstacle-cube-world-m",
            "0.2",
            "0.0",
            "0.4",
            "0.1",
            "--obstacle-cube-world-m",
            "-0.2",
            "0.0",
            "0.4",
            "0.2",
        ]
    )
    scene = acceptance._scene_from_cli(args)
    assert isinstance(scene, ObstacleSceneSnapshot)
    assert [item.obstacle_id for item in scene.obstacles] == ["cube-0", "cube-1"]
    assert scene.minimum_clearance_m == pytest.approx(0.03)
    assert scene.reference_frame_id == acceptance.OBSTACLE_FRAME_ID


def test_static_identity_and_binding_use_explicit_sim_contract() -> None:
    profile = _profile()
    identity = acceptance._build_identity(profile)
    assert identity.reference_frame_id == "desk"
    assert set(identity.tool_profile_hash_by_side) == {"left", "right"}
    binding = acceptance._build_binding(
        profile, SimpleNamespace(scene_hash="scene:test")
    )
    assert binding.scene_revision == "scene:test"
    assert binding.collision_profile_name == profile.collision_scene_profile
    assert binding.gripper_side_by_name


def test_jsonable_preserves_arrays_and_enums() -> None:
    payload = acceptance._jsonable(
        {"pose": np.eye(4), "path": Path("/tmp/result.json")}
    )
    assert payload["pose"][0][0] == 1.0
    assert payload["path"] == "/tmp/result.json"


def test_execute_real_parser_uses_post_simulation_confirmation() -> None:
    args = acceptance.build_parser().parse_args(
        [
            "--arm-ip-assignment",
            "/tmp/arms.json",
            "--dmtac-assignment",
            "/tmp/dmtac.json",
            "--target-offset-xyz-m",
            "0.02",
            "0.00",
            "0.03",
            "--execute-real",
        ]
    )
    assert args.execute_real is True
    assert args.no_motion is False
    assert args.target_offset_xyz_m == [0.02, 0.0, 0.03]

    confirmation = acceptance._PostSimulationConfirmation(
        reference="interactive-post-mujoco:test"
    )
    issue = acceptance._build_real_authority_factory(
        confirmation=confirmation,
        ttl_sec=5.0,
        clock=lambda: 100.0,
    )
    authority, consume = issue(
        SimpleNamespace(semantic_content_hash="trajectory:test"),
        SimpleNamespace(semantic_content_hash="simulation:test"),
    )
    assert authority.trajectory_hash == "trajectory:test"
    assert authority.simulation_receipt_hash == "simulation:test"
    assert authority.operator_confirmation_ref == "interactive-post-mujoco:test"
    assert callable(consume)


def test_authority_factory_rejects_when_terminal_confirmation_was_not_obtained() -> (
    None
):
    issue = acceptance._build_real_authority_factory(
        confirmation=acceptance._PostSimulationConfirmation(),
        clock=lambda: 100.0,
    )
    with pytest.raises(
        acceptance.OperatorConfirmationDeclined,
        match="did not confirm",
    ):
        issue(
            SimpleNamespace(semantic_content_hash="trajectory:test"),
            SimpleNamespace(semantic_content_hash="simulation:test"),
        )


def test_terminal_confirmation_is_interactive_and_bound_to_completed_receipts() -> None:
    confirmation = acceptance._PostSimulationConfirmation()
    accepted = acceptance._confirm_real_replay_after_simulation(
        trajectory=SimpleNamespace(semantic_content_hash="a" * 64),
        simulation=SimpleNamespace(semantic_content_hash="b" * 64),
        viewer_handle=SimpleNamespace(is_running=lambda: True),
        confirmation=confirmation,
        input_fn=lambda _prompt: "EXECUTE",
    )
    assert accepted is True
    assert confirmation.declined is False
    assert confirmation.reference == (
        "interactive-post-mujoco:aaaaaaaaaaaaaaaa:bbbbbbbbbbbbbbbb"
    )


def test_relative_target_preserves_measured_orientation_and_requires_fresh_ik(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sim.sim_real_tests.mujoco import arbitrary_grasp_pose_motion

    measured = np.eye(4)
    measured[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    measured[:3, 3] = (0.20, -0.10, 0.35)

    class FakeControl:
        def __init__(self):
            self.closed = False
            self.scene = SimpleNamespace(
                get_arm_qpos=lambda side: np.full(6, 1.0 if side == "left" else 2.0)
            )
            self.retargeter = SimpleNamespace(
                tool_tcp_world_targets_for_qpos_pair=lambda **_kwargs: {
                    "left": np.eye(4),
                    "right": measured,
                }
            )
            self.world_to_base_by_side = {
                "left": np.eye(4),
                "right": np.eye(4),
            }

        def close(self):
            self.closed = True

    control = FakeControl()
    monkeypatch.setattr(acceptance, "_simulation_config", lambda *_args: object())
    monkeypatch.setattr(
        arbitrary_grasp_pose_motion,
        "build_shadow_initialized_control",
        lambda *_args, **_kwargs: control,
    )
    target = acceptance._target_from_current(
        SimpleNamespace(),
        SimpleNamespace(),
        side="right",
        translation_offset_m=(0.01, 0.02, -0.03),
    )

    np.testing.assert_allclose(
        target.target_pose_base_from_dmtac_tcp[:3, :3], measured[:3, :3]
    )
    np.testing.assert_allclose(
        target.target_pose_base_from_dmtac_tcp[:3, 3],
        measured[:3, 3] + np.array([0.01, 0.02, -0.03]),
    )
    assert target.known_qpos_rad is None
    assert control.closed is True


def test_execute_real_rejects_zero_motion_target_before_hardware_setup() -> None:
    assert (
        acceptance.main(
            [
                "--arm-ip-assignment",
                "/does/not/matter.json",
                "--dmtac-assignment",
                "/does/not/matter.json",
                "--target-from-current",
                "--execute-real",
            ]
        )
        == 2
    )


def test_viewer_executor_animates_exact_sim_then_confirms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sim.mujoco.viewer as mujoco_viewer
    from robot_motion.shadow_execution import ShadowSimulationStatus
    from sim.mujoco.replay import tcp_target_visualization
    from sim.sim_real_tests.mujoco import (
        exact_shadow_trajectory,
        view_arbitrary_grasp_pose_motion,
    )

    class FakeViewer:
        def __init__(self):
            self.cam = SimpleNamespace(
                lookat=np.zeros(3), distance=0.0, azimuth=0.0, elevation=0.0
            )
            self.sync_count = 0

        def is_running(self):
            return True

        def sync(self):
            self.sync_count += 1

    viewer = FakeViewer()

    @contextmanager
    def fake_viewer(_scene, **_kwargs):
        yield viewer

    class FakeVisualizer:
        def __init__(self, *_args, **_kwargs):
            pass

        def update(self, *_args, **_kwargs):
            return True

    class FakeControl:
        def __init__(self):
            self.scene = SimpleNamespace(data=SimpleNamespace(time=0.0))

        def step_physics(self):
            return "tick"

    control = FakeControl()
    original_bound_step = control.step_physics
    simulation = SimpleNamespace(
        status=ShadowSimulationStatus.COMPLETED,
        semantic_content_hash="s" * 64,
    )

    def execute(**kwargs):
        assert kwargs["control"].step_physics() == "tick"
        return simulation

    monkeypatch.setattr(mujoco_viewer, "passive_viewer_handle", fake_viewer)
    monkeypatch.setattr(tcp_target_visualization, "TcpTargetVisualizer", FakeVisualizer)
    monkeypatch.setattr(
        view_arbitrary_grasp_pose_motion,
        "_ObstacleSceneVisualizer",
        FakeVisualizer,
    )
    monkeypatch.setattr(
        exact_shadow_trajectory,
        "execute_registered_shadow_trajectory",
        execute,
    )
    monkeypatch.setattr(
        acceptance,
        "_target_pose_world_by_side",
        lambda *_args: {"left": np.eye(4), "right": np.eye(4)},
    )
    confirmation = acceptance._PostSimulationConfirmation()
    observed = acceptance._execute_registered_shadow_with_viewer(
        control=control,
        trajectory=SimpleNamespace(semantic_content_hash="t" * 64),
        model_binding=object(),
        target=SimpleNamespace(side="left", target_pose_base_from_dmtac_tcp=np.eye(4)),
        obstacle_scene=object(),
        confirmation=confirmation,
        require_confirmation=True,
        playback_hz=60.0,
        terminal_display_sec=0.0,
        camera_distance=1.5,
        camera_azimuth_deg=135.0,
        camera_elevation_deg=-24.0,
        input_fn=lambda _prompt: "EXECUTE",
    )

    assert observed is simulation
    assert viewer.sync_count >= 3
    assert confirmation.reference is not None
    assert control.step_physics == original_bound_step
