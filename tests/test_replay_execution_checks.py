from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from hardware.rm65.calibration_profiles import make_profile
from teleop_runtime.action import HandActionSpec
from teleop_runtime.action.executor import RobotActionResolver
from teleop_runtime.deploy.replay import (
    load_replay_arm_joint_targets_from_parquet,
    load_replay_arm_observation_qpos_from_parquet,
)
from teleop_runtime.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM, ACTION_HAND_SLICES
from teleop_runtime.replay.action_frame import replay_teleop_command_at_index
from teleop_runtime.replay.checks import run_replay_preflight
from teleop_runtime.replay.executor import ReplayExecutionController, ReplayExecutionSettings
from teleop_runtime.replay.hardware import ReplayHardwareConfig, ReplayHardwareRig
from teleop_runtime.replay.session import ReplayGuiSession
from teleop_runtime.replay.trajectory import REPLAY_TRAJECTORY_QPOS_TARGET, ReplayTrajectoryGroup
from teleop_runtime.schemas import EffectorCommand, TeleopCommand


def _write_profile(path: Path, *, side: str, tcp: list[float]) -> Path:
    profile = make_profile(name=f"test_{side}", arm=side, tool_name="test_tool")
    profile["tcp"]["pose_flange_m_rad"] = list(tcp)
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def _group(
    profile_path: Path,
    *,
    qpos_targets: bool = True,
    observation_current_qpos: bool = True,
    observation_actual_qpos: bool = False,
    hand_targets: bool = False,
    hand_padding_nan: bool = False,
    hand_spec: bool = False,
) -> ReplayTrajectoryGroup:
    actions = np.full((2, ACTION_DIM), np.nan, dtype=np.float32)
    actions[:, ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [
            [0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            [0.21, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    if hand_targets:
        padding = np.nan if hand_padding_nan else 0.0
        actions[:, ACTION_HAND_SLICES["right"]] = np.asarray(
            [
                [0.4, padding, padding, padding, padding, padding],
                [0.5, padding, padding, padding, padding, padding],
            ],
            dtype=np.float32,
        )
    episode_metadata = {"right_tool_profile": str(profile_path)}
    dataset_info = {}
    if hand_spec:
        spec = HandActionSpec.dmtac_gripper("right").to_metadata()
        episode_metadata["hand_action_specs_by_side"] = {"right": spec}
        dataset_info["hand_action_specs_by_side"] = {"right": spec}
    return ReplayTrajectoryGroup(
        key="g",
        label="test",
        source_path="episode_000000.parquet",
        source_episode_index=0,
        start_index=0,
        end_index=2,
        frame_count=2,
        action_hz=10.0,
        actions=actions,
        arm_joint_targets={}
        if not qpos_targets
        else {"right": np.zeros((2, 6), dtype=np.float32)},
        arm_observation_current_qpos={}
        if not observation_current_qpos
        else {"right": np.zeros((2, 6), dtype=np.float32)},
        arm_observation_actual_qpos={}
        if not observation_actual_qpos
        else {"right": np.zeros((2, 6), dtype=np.float32)},
        metadata={
            "source_metadata": {
                "episode_metadata": episode_metadata,
                "dataset_info": dataset_info,
            }
        },
    )


class _FakeReplayRig:
    def __init__(self) -> None:
        self.connected = True
        self.close_count = 0
        self.hold_reasons: list[str] = []
        self.restore_count = 0

    def close(self) -> None:
        self.close_count += 1
        self.connected = False

    def hold_motion(self, reason: str = "hardware replay hold") -> None:
        self.hold_reasons.append(reason)

    def restore_replay_speed(self) -> None:
        self.restore_count += 1


class _FailingConnectRig:
    def __init__(self, config: ReplayHardwareConfig) -> None:
        del config
        self.connected = False
        self.close_count = 0

    def connect(self) -> None:
        raise RuntimeError("connect boom")

    def close(self) -> None:
        self.close_count += 1


class _FakeReplayWorkerRig(_FakeReplayRig):
    def __init__(self) -> None:
        super().__init__()
        self.begin_count = 0
        self.sent_joint_targets: list[dict[str, np.ndarray]] = []
        self.sent_action_poses: list[dict[str, np.ndarray]] = []
        self.sent_commands = []
        self.action_resolver = RobotActionResolver()

    def begin_replay_speed(self) -> None:
        self.begin_count += 1

    def send_joint_targets(self, qpos_by_side, *, source: str) -> None:
        self.sent_joint_targets.append(
            {
                str(side): np.asarray(qpos, dtype=float).copy()
                for side, qpos in dict(qpos_by_side).items()
            }
        )

    def send_action_poses(self, action_by_side, *, source: str) -> None:
        self.sent_action_poses.append(
            {
                str(side): np.asarray(pose, dtype=float).copy()
                for side, pose in dict(action_by_side).items()
            }
        )

    def send_commands(self, command, phase_recorder=None) -> None:
        del phase_recorder
        self.sent_commands.append(command)


class _SingularityFaultReplayRig(_FakeReplayWorkerRig):
    def send_joint_targets(self, qpos_by_side, *, source: str) -> None:
        super().send_joint_targets(qpos_by_side, source=source)
        raise RuntimeError(
            "left_arm replay command blocked by singularity gate: "
            "status=hold_entry source=joint_vector target=wrist_singularity current=normal"
        )


class _CaptureBackend:
    def __init__(self) -> None:
        self.commands = []

    def is_ready(self) -> bool:
        return True

    def send_command(self, command) -> None:
        self.commands.append(command.copy())


class _SingularityHoldBackend(_CaptureBackend):
    def get_state(self) -> dict[str, object]:
        return {
            "singularity_metadata": {
                "singularity_gate_status": "hold_entry",
                "source": "joint_vector",
                "target_singularity_label": "wrist_singularity",
                "current_singularity_label": "normal",
            }
        }


@dataclass(frozen=True)
class _FakePlanStep:
    index: int
    qpos_by_side: dict[str, np.ndarray]


class _FakeQposInitPlan:
    completed = True

    def __init__(self, steps: list[_FakePlanStep]) -> None:
        self.steps = tuple(steps)

    def summary(self) -> dict[str, object]:
        return {"completed": True, "step_count": len(self.steps), "metadata": {"planner": "fake"}}


class _FakeQposInitRig(_FakeReplayWorkerRig):
    def __init__(self) -> None:
        super().__init__()
        self.current = {"right": np.zeros(6, dtype=float)}

    def current_qpos_by_side(self) -> dict[str, np.ndarray]:
        return {side: qpos.copy() for side, qpos in self.current.items()}

    def send_joint_targets(self, qpos_by_side, *, source: str) -> None:
        super().send_joint_targets(qpos_by_side, source=source)
        for side, qpos in dict(qpos_by_side).items():
            self.current[str(side)] = np.asarray(qpos, dtype=float).copy()

    def qpos_reached(self, qpos_by_side, *, tolerance: float) -> bool:
        for side, target in dict(qpos_by_side).items():
            current = self.current.get(str(side))
            if current is None:
                return False
            if float(np.max(np.abs(current - np.asarray(target, dtype=float).reshape(6)))) > float(tolerance):
                return False
        return True


class ReplayExecutionChecksTest(unittest.TestCase):
    def test_optional_replay_qpos_loader_skips_null_side_instead_of_raising(self) -> None:
        class _FakeParquet:
            def read_schema(self, path):
                del path
                return SimpleNamespace(names=["action.replay.left_arm_joint_target"])

            def read_table(self, path, columns):
                del path, columns
                return SimpleNamespace(
                    to_pylist=lambda: [
                        {"action.replay.left_arm_joint_target": None},
                    ]
                )

        with mock.patch("teleop_runtime.deploy.replay._require_pyarrow_parquet", return_value=_FakeParquet()):
            targets = load_replay_arm_joint_targets_from_parquet(
                "/tmp/fake.parquet",
                active_sides=("left",),
                required=False,
            )

        self.assertEqual(targets, {})

    def test_required_replay_qpos_loader_rejects_null_target(self) -> None:
        class _FakeParquet:
            def read_schema(self, path):
                del path
                return SimpleNamespace(names=["action.replay.left_arm_joint_target"])

            def read_table(self, path, columns):
                del path, columns
                return SimpleNamespace(
                    to_pylist=lambda: [
                        {"action.replay.left_arm_joint_target": None},
                    ]
                )

        with mock.patch("teleop_runtime.deploy.replay._require_pyarrow_parquet", return_value=_FakeParquet()):
            with self.assertRaisesRegex(ValueError, "cannot use qpos init"):
                load_replay_arm_joint_targets_from_parquet(
                    "/tmp/fake.parquet",
                    active_sides=("left",),
                    required=True,
                )

    def test_observation_qpos_loader_preserves_rows_with_nan_for_invalid_values(self) -> None:
        class _FakeParquet:
            def read_schema(self, path):
                del path
                return SimpleNamespace(names=["observation.arm.left.current_qpos"])

            def read_table(self, path, columns):
                del path, columns
                return SimpleNamespace(
                    to_pylist=lambda: [
                        {"observation.arm.left.current_qpos": [1, 2, 3, 4, 5, 6]},
                        {"observation.arm.left.current_qpos": None},
                    ]
                )

        with mock.patch("teleop_runtime.deploy.replay._require_pyarrow_parquet", return_value=_FakeParquet()):
            targets = load_replay_arm_observation_qpos_from_parquet(
                "/tmp/fake.parquet",
                active_sides=("left",),
                field_name="current_qpos",
            )

        self.assertEqual(set(targets), {"left"})
        self.assertEqual(targets["left"].shape, (2, 6))
        np.testing.assert_allclose(targets["left"][0], np.asarray([1, 2, 3, 4, 5, 6]))
        self.assertTrue(np.all(np.isnan(targets["left"][1])))

    def test_preflight_accepts_matching_tcp_and_qpos(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            report = run_replay_preflight(
                group=_group(profile),
                active_sides=("right",),
                current_qpos_by_side={"right": np.zeros(6)},
                current_tcp_pose_by_side={"right": np.eye(4)},
                current_tcp_flange_by_side={"right": np.asarray([0.01, 0, 0, 0, 0, 0])},
            )
            self.assertTrue(report.ok)

    def test_replay_action_frame_builds_dmtac_end_effector_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile, hand_targets=True, hand_spec=True)

            command = replay_teleop_command_at_index(
                group,
                ("right",),
                index=0,
                include_end_effectors=True,
                source="test_replay",
            )

            self.assertEqual(set(command.commands), {"right_arm", "right_hand"})
            hand_command = command.commands["right_hand"]
            np.testing.assert_allclose(hand_command.value, np.asarray([0.4], dtype=np.float32))
            self.assertEqual(hand_command.metadata["hand_action_spec"]["hand_model"], "dmtac_gripper")
            self.assertEqual(hand_command.metadata["source"], "test_replay")

    def test_replay_action_frame_allows_dmtac_inactive_nan_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile, hand_targets=True, hand_padding_nan=True, hand_spec=True)

            command = replay_teleop_command_at_index(
                group,
                ("right",),
                index=0,
                include_end_effectors=True,
                source="test_replay",
            )

            hand_command = command.commands["right_hand"]
            np.testing.assert_allclose(hand_command.value, np.asarray([0.4], dtype=np.float32))
            self.assertEqual(hand_command.metadata["hand_active_mask"], [True, False, False, False, False, False])

    def test_preflight_blocks_tcp_mismatch_and_missing_qpos_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            mismatch = run_replay_preflight(
                group=_group(profile),
                active_sides=("right",),
                current_qpos_by_side={"right": np.zeros(6)},
                current_tcp_pose_by_side={"right": np.eye(4)},
                current_tcp_flange_by_side={"right": np.zeros(6)},
            )
            self.assertFalse(mismatch.ok)
            failed = {check.name for check in mismatch.checks if not check.ok}
            self.assertIn("right_tcp_numeric_match", failed)

            missing_qpos = run_replay_preflight(
                group=_group(profile, qpos_targets=False, observation_current_qpos=False),
                active_sides=("right",),
                current_qpos_by_side={"right": np.zeros(6)},
                current_tcp_pose_by_side={"right": np.eye(4)},
                current_tcp_flange_by_side={"right": np.asarray([0.01, 0, 0, 0, 0, 0])},
            )
            self.assertFalse(missing_qpos.ok)
            failed = {check.name for check in missing_qpos.checks if not check.ok}
            self.assertIn("right_qpos_init_target_present", failed)

            missing_qpos_trajectory = run_replay_preflight(
                group=_group(profile, qpos_targets=False),
                active_sides=("right",),
                current_qpos_by_side={"right": np.zeros(6)},
                current_tcp_pose_by_side={"right": np.eye(4)},
                current_tcp_flange_by_side={"right": np.asarray([0.01, 0, 0, 0, 0, 0])},
                trajectory_source=REPLAY_TRAJECTORY_QPOS_TARGET,
            )
            self.assertFalse(missing_qpos_trajectory.ok)
            failed = {check.name for check in missing_qpos_trajectory.checks if not check.ok}
            self.assertIn("right_qpos_targets_present", failed)

            missing_end_effector_backend = run_replay_preflight(
                group=_group(profile, hand_targets=True, hand_spec=True),
                active_sides=("right",),
                current_qpos_by_side={"right": np.zeros(6)},
                current_tcp_pose_by_side={"right": np.eye(4)},
                current_tcp_flange_by_side={"right": np.asarray([0.01, 0, 0, 0, 0, 0])},
                end_effectors_enabled=True,
                end_effector_backend_sides=(),
            )
            self.assertFalse(missing_end_effector_backend.ok)
            failed = {check.name for check in missing_end_effector_backend.checks if not check.ok}
            self.assertIn("right_end_effector_backend_present", failed)

    def test_executor_blocks_motion_until_output_is_enabled(self) -> None:
        controller = ReplayExecutionController()
        with self.assertRaisesRegex(RuntimeError, "Arm output is disabled"):
            controller.move_to_start()

    def test_executor_abort_holds_hardware_and_restores_speed(self) -> None:
        controller = ReplayExecutionController()
        rig = _FakeReplayRig()
        controller._rig = rig  # type: ignore[assignment]
        controller._state = "replaying"

        status = controller.abort()

        self.assertEqual(status["state"], "aborted")
        self.assertEqual(rig.hold_reasons, ["hardware replay abort"])
        self.assertEqual(rig.restore_count, 1)

    def test_configure_preserves_connected_rig_when_hardware_config_unchanged(self) -> None:
        config = ReplayHardwareConfig(active_sides=("right",), connect_hardware=False)
        controller = ReplayExecutionController(hardware_config=config)
        rig = _FakeReplayRig()
        controller._rig = rig  # type: ignore[assignment]
        controller._state = "connected"

        status = controller.configure_from_payload(
            {
                "active_sides": "right",
                "connect_hardware": False,
                "send_actions": True,
            }
        )

        self.assertIs(controller._rig, rig)  # type: ignore[attr-defined]
        self.assertEqual(rig.close_count, 0)
        self.assertTrue(status["connected"])
        self.assertEqual(status["state"], "connected")
        self.assertTrue(status["settings"]["send_actions_enabled"])

    def test_configure_payload_sets_qpos_init_speed(self) -> None:
        controller = ReplayExecutionController(
            hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False)
        )

        status = controller.configure_from_payload(
            {
                "active_sides": "right",
                "connect_hardware": False,
                "qpos_init_max_angvel_rad_s": "0.2,0.2,0.2,0.3,0.3,0.3",
            }
        )

        self.assertEqual(
            status["settings"]["qpos_init_max_angvel_rad_s"],
            [0.2, 0.2, 0.2, 0.3, 0.3, 0.3],
        )

    def test_replay_hardware_config_passes_qpos_init_singularity_slowdown_to_arm_backend(self) -> None:
        config = ReplayHardwareConfig.from_payload(
            {
                "active_sides": "right",
                "connect_hardware": False,
                "singularity_qpos_init_slowdown": 0.35,
            }
        )
        self.assertAlmostEqual(config.singularity_qpos_init_slowdown, 0.35)
        self.assertAlmostEqual(config.to_dict()["singularity_qpos_init_slowdown"], 0.35)
        rig = ReplayHardwareRig(config)
        try:
            rig.connect()
            self.assertAlmostEqual(
                rig.arm_backends["right"].config.singularity_qpos_init_slowdown,
                0.35,
            )
        finally:
            rig.close()

        with self.assertRaisesRegex(ValueError, "singularity_qpos_init_slowdown"):
            ReplayHardwareConfig.from_payload(
                {
                    "active_sides": "right",
                    "connect_hardware": False,
                    "singularity_qpos_init_slowdown": 0.0,
                }
            )

    def test_connect_failure_closes_failed_rig_and_clears_reference(self) -> None:
        controller = ReplayExecutionController(
            hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False)
        )
        created: list[_FailingConnectRig] = []

        def factory(config: ReplayHardwareConfig) -> _FailingConnectRig:
            rig = _FailingConnectRig(config)
            created.append(rig)
            return rig

        with mock.patch("teleop_runtime.replay.executor.ReplayHardwareRig", side_effect=factory):
            with self.assertRaisesRegex(RuntimeError, "connect boom"):
                controller.connect()

        status = controller.status()
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].close_count, 1)
        self.assertFalse(status["connected"])
        self.assertEqual(status["state"], "fault")
        self.assertEqual(status["last_message"], "hardware connect failed")
        self.assertIn("connect boom", status["last_error"])
        self.assertIsNone(controller._rig)  # type: ignore[attr-defined]

    def test_session_blocks_dataset_switch_while_hardware_worker_busy(self) -> None:
        session = ReplayGuiSession()
        try:
            session.executor._state = "replaying"
            with self.assertRaisesRegex(RuntimeError, "before loading a dataset"):
                session.load_dataset("/tmp/nonexistent_replay_dataset")
        finally:
            session.close()

    def test_replay_worker_dispatches_action_poses_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile, hand_targets=True, hand_spec=True)
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False),
                settings=ReplayExecutionSettings(send_actions_enabled=True),
            )
            rig = _FakeReplayWorkerRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            self.assertEqual(controller.status()["state"], "finished")
            self.assertEqual(rig.begin_count, 1)
            self.assertEqual(rig.restore_count, 1)
            self.assertEqual(len(rig.sent_joint_targets), 0)
            self.assertEqual(len(rig.sent_action_poses), 0)
            self.assertEqual(len(rig.sent_commands), 2)
            self.assertEqual(set(rig.sent_commands[0].commands), {"right_arm"})
            self.assertEqual(set(rig.sent_commands[1].commands), {"right_arm"})

    def test_replay_worker_dispatches_end_effector_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile, hand_targets=True, hand_spec=True)
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("right",),
                    connect_hardware=False,
                    end_effectors_enabled=True,
                    end_effector_backend_by_side={"right": "dmtac_gripper"},
                ),
                settings=ReplayExecutionSettings(send_actions_enabled=True, end_effectors_enabled=True),
            )
            rig = _FakeReplayWorkerRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            self.assertEqual(controller.status()["state"], "finished")
            self.assertEqual(len(rig.sent_commands), 2)
            self.assertEqual(set(rig.sent_commands[0].commands), {"right_arm", "right_hand"})
            hand_command = rig.sent_commands[0].commands["right_hand"]
            self.assertEqual(hand_command.command_type, "joint_vector")
            np.testing.assert_allclose(hand_command.value, np.asarray([0.4], dtype=np.float32))
            self.assertEqual(hand_command.metadata["hand_action_spec"]["hand_model"], "dmtac_gripper")

    def test_replay_worker_can_dispatch_recorded_qpos_targets_in_qpos_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile)
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    trajectory_source=REPLAY_TRAJECTORY_QPOS_TARGET,
                ),
            )
            rig = _FakeReplayWorkerRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            self.assertEqual(controller.status()["state"], "finished")
            self.assertEqual(rig.begin_count, 1)
            self.assertEqual(rig.restore_count, 1)
            self.assertEqual(len(rig.sent_action_poses), 0)
            self.assertEqual(len(rig.sent_joint_targets), 2)
            np.testing.assert_allclose(rig.sent_joint_targets[0]["right"], np.zeros(6))
            np.testing.assert_allclose(rig.sent_joint_targets[1]["right"], np.zeros(6))

    def test_replay_worker_fails_fast_on_singularity_gate_hold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile)
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    trajectory_source=REPLAY_TRAJECTORY_QPOS_TARGET,
                ),
            )
            rig = _SingularityFaultReplayRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            status = controller.status()
            self.assertEqual(status["state"], "fault")
            self.assertIn("frame_index=0", status["last_error"])
            self.assertIn("trajectory_source=qpos_target", status["last_error"])
            self.assertIn("singularity gate", status["last_error"])
            self.assertEqual(rig.hold_reasons, ["hardware replay fault"])
            self.assertEqual(rig.restore_count, 1)
            self.assertEqual(len(rig.sent_joint_targets), 1)

    def test_hardware_rig_raises_when_backend_singularity_gate_holds(self) -> None:
        rig = ReplayHardwareRig(ReplayHardwareConfig(active_sides=("left",), connect_hardware=False))
        backend = _SingularityHoldBackend()
        rig.effector_backends["left_arm"] = backend  # type: ignore[assignment]
        command = TeleopCommand(metadata={"source": "test"})
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.zeros(6, dtype=float),
                frame_name=None,
                metadata={"source": "test"},
            )
        )

        with self.assertRaisesRegex(RuntimeError, "left_arm replay command blocked by singularity gate"):
            rig.send_commands(command)

        self.assertEqual(len(backend.commands), 1)

    def test_hardware_send_action_poses_uses_robot_action_command_metadata(self) -> None:
        rig = ReplayHardwareRig(ReplayHardwareConfig(active_sides=("right",), connect_hardware=False))
        backend = _CaptureBackend()
        rig.arm_backends["right"] = backend  # type: ignore[assignment]
        rig.effector_backends["right_arm"] = backend  # type: ignore[assignment]

        rig.send_action_poses(
            {"right": np.asarray([0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)},
            source="replay_action_pose",
        )

        self.assertEqual(len(backend.commands), 1)
        command = backend.commands[0]
        self.assertEqual(command.command_type, "cartesian_pose")
        self.assertEqual(command.frame_name, "right_arm_base")
        self.assertEqual(command.metadata["source"], "replay_action_pose")
        self.assertEqual(command.metadata["arm_controlled_frame"], "tool_tcp")
        self.assertEqual(command.metadata["target_pose_controlled_frame"], "tool_tcp")
        self.assertIn("arm_action_spec", command.metadata)
        self.assertIn("source_target_pose_base", command.metadata)
        self.assertIn("target_pose_tcp_base", command.metadata)

    def test_move_to_start_executes_qpos_init_plan_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile)
            target = {"right": np.full(6, 0.2, dtype=float)}
            plan = _FakeQposInitPlan(
                [
                    _FakePlanStep(index=1, qpos_by_side={"right": np.full(6, 0.1, dtype=float)}),
                    _FakePlanStep(index=2, qpos_by_side=target),
                ]
            )
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    qpos_init_dt_sec=0.001,
                    qpos_init_max_angvel_rad_s=(0.25, 0.25, 0.25, 0.4, 0.4, 0.4),
                ),
            )
            rig = _FakeQposInitRig()

            with mock.patch("teleop_runtime.replay.executor.build_rm65_qpos_init_plan", return_value=plan) as build_plan:
                controller._move_to_start_worker(rig, group, target)  # type: ignore[arg-type]

            self.assertEqual(
                build_plan.call_args.kwargs["qpos_init_max_angvel_rad_s"],
                (0.25, 0.25, 0.25, 0.4, 0.4, 0.4),
            )
            status = controller.status()
            self.assertEqual(status["state"], "start_ready")
            self.assertEqual(status["qpos_init"]["step_index"], 2)
            self.assertEqual(status["qpos_init"]["step_count"], 2)
            self.assertEqual(len(rig.sent_joint_targets), 2)
            np.testing.assert_allclose(rig.sent_joint_targets[0]["right"], np.full(6, 0.1))
            np.testing.assert_allclose(rig.sent_joint_targets[1]["right"], np.full(6, 0.2))

    def test_session_hardware_config_supports_both_sides(self) -> None:
        session = ReplayGuiSession(
            hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False)
        )
        try:
            state = session.dispatch(
                {
                    "type": "configure_hardware",
                    "active_sides": "both",
                    "left_arm_ip": "192.168.1.10",
                    "right_arm_ip": "192.168.1.11",
                    "left_tool_profile": "/tmp/left.json",
                    "right_tool_profile": "/tmp/right.json",
                    "connect_hardware": False,
                    "send_actions": False,
                }
            )
            execution = state["execution"]
            self.assertEqual(execution["active_sides"], ["left", "right"])
            self.assertFalse(execution["settings"]["send_actions_enabled"])
            self.assertFalse(execution["hardware_config"]["connect_hardware"])
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
