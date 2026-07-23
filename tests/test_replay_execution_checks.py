from __future__ import annotations

import json
import queue
import tempfile
import threading
import time
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
    load_dataset_replay_policy,
    load_replay_arm_joint_targets_from_parquet,
    load_replay_arm_observation_qpos_from_parquet,
    load_replay_frame_times_from_parquet,
    load_replay_hand_joint_targets_from_parquet,
)
from teleop_runtime.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM, ACTION_HAND_SLICES
from teleop_runtime.backends.arms import RM65ArmBackend, RM65ImpedanceArmBackend
from teleop_runtime.defaults import (
    DEFAULT_RM65_ARM_CONTROL_HZ,
    DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
    DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    DEFAULT_TELEOP_SAMPLE_HZ,
)
from teleop_runtime.replay.action_frame import replay_teleop_command_at_index
from teleop_runtime.replay.checks import (
    ReplayCheckResult,
    ReplayPreflightReport,
    run_replay_preflight,
)
from teleop_runtime.replay.executor import ReplayExecutionController, ReplayExecutionSettings
from teleop_runtime.replay.hardware import ReplayHardwareConfig, ReplayHardwareRig
from teleop_runtime.replay.session import ReplayGuiSession
from teleop_runtime.replay.trajectory import REPLAY_TRAJECTORY_QPOS_TARGET, ReplayTrajectoryGroup
from teleop_runtime.replay.web_gui import (
    _resolve_replay_dmtac_fields,
    build_parser as build_replay_web_gui_parser,
)
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
    frame_times_sec: tuple[float, ...] = (),
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
        frame_times_sec=frame_times_sec,
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
            "frame_timing": {
                "source_timestamp_mode": (
                    "dataset" if frame_times_sec else "nominal_fallback"
                )
            },
            "source_metadata": {
                "episode_metadata": episode_metadata,
                "dataset_info": dataset_info,
            }
        },
    )


def _dual_group(
    left_profile_path: Path,
    right_profile_path: Path,
    *,
    hand_targets: bool = False,
    hand_spec: bool = False,
) -> ReplayTrajectoryGroup:
    group = _group(
        right_profile_path,
        hand_targets=hand_targets,
        hand_spec=hand_spec,
    )
    group.actions[:, ACTION_ARM_POSE_SLICES["left"]] = np.asarray(
        [
            [0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0],
            [0.21, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    if hand_targets:
        group.actions[:, ACTION_HAND_SLICES["left"]] = np.asarray(
            [
                [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.35, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    group.arm_joint_targets["left"] = np.zeros((2, 6), dtype=np.float32)
    group.arm_observation_current_qpos["left"] = np.zeros((2, 6), dtype=np.float32)
    episode_metadata = group.metadata["source_metadata"]["episode_metadata"]
    episode_metadata["left_tool_profile"] = str(left_profile_path)
    if hand_spec:
        left_spec = HandActionSpec.dmtac_gripper("left").to_metadata()
        episode_metadata["hand_action_specs_by_side"]["left"] = left_spec
        group.metadata["source_metadata"]["dataset_info"][
            "hand_action_specs_by_side"
        ]["left"] = left_spec
    return group


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
        self.sent_joint_metadata: list[dict[str, object]] = []
        self.sent_action_poses: list[dict[str, np.ndarray]] = []
        self.sent_action_metadata: list[dict[str, object]] = []
        self.sent_commands = []
        self.command_dt_history: list[float] = []
        self.action_resolver = RobotActionResolver()

    def set_arm_command_dt_sec(self, control_dt_sec: float) -> None:
        self.command_dt_history.append(float(control_dt_sec))

    def begin_replay_speed(self) -> None:
        self.begin_count += 1

    def send_joint_targets(self, qpos_by_side, *, source: str, metadata=None) -> None:
        self.sent_joint_targets.append(
            {
                str(side): np.asarray(qpos, dtype=float).copy()
                for side, qpos in dict(qpos_by_side).items()
            }
        )
        self.sent_joint_metadata.append(dict(metadata or {}))

    def send_action_poses(self, action_by_side, *, source: str, metadata=None) -> None:
        self.sent_action_poses.append(
            {
                str(side): np.asarray(pose, dtype=float).copy()
                for side, pose in dict(action_by_side).items()
            }
        )
        self.sent_action_metadata.append(dict(metadata or {}))

    def send_commands(self, command, phase_recorder=None) -> None:
        del phase_recorder
        self.sent_commands.append(command)


class _DualFakeReplayWorkerRig(_FakeReplayWorkerRig):
    def current_qpos_by_side(self) -> dict[str, np.ndarray]:
        return {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }

    def current_tcp_flange_by_side(self) -> dict[str, np.ndarray]:
        return {
            "left": np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
            "right": np.asarray([0.02, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        }

    def current_tcp_pose_by_side(self) -> dict[str, np.ndarray]:
        return {"left": np.eye(4, dtype=float), "right": np.eye(4, dtype=float)}

    def end_effector_backend_sides(self) -> tuple[str, ...]:
        return ()


class _HealthFaultBeforeHandRig(_DualFakeReplayWorkerRig):
    def __init__(self) -> None:
        super().__init__()
        self.health_stages: list[str] = []

    def assert_position_stream_healthy(self, *, stage: str):
        self.health_stages.append(str(stage))
        if stage == "immediately_before_end_effector_dispatch":
            raise RuntimeError("right 150 Hz position stream fault is latched")
        return {"left": {"healthy": True}, "right": {"healthy": True}}


class _HealthyTransportSnapshotRig(_DualFakeReplayWorkerRig):
    def __init__(self) -> None:
        super().__init__()
        self.health_stages: list[str] = []

    def assert_position_stream_healthy(self, *, stage: str):
        self.health_stages.append(str(stage))
        return {
            "left": {"background_streaming": {"healthy": True, "actual_hz": 150.0}},
            "right": {"background_streaming": {"healthy": True, "actual_hz": 149.5}},
        }


class _FreshReadFailureRig(_DualFakeReplayWorkerRig):
    def fresh_hardware_qpos_by_side(
        self,
        *,
        max_age_sec: float,
        pair_max_skew_sec: float,
    ):
        del max_age_sec, pair_max_skew_sec
        raise RuntimeError("fresh hardware qpos read failed")


class _FakePositionCollisionPlanner:
    collision_pair_count = 12

    def __init__(
        self,
        *,
        hold: bool = False,
        valid: bool = True,
        singularity_slow_recheck_safe: bool = True,
    ) -> None:
        self.hold = bool(hold)
        self.valid = bool(valid)
        self.singularity_slow_recheck_safe = bool(singularity_slow_recheck_safe)
        self.reset_inputs: list[dict[str, np.ndarray]] = []
        self.plan_inputs: list[dict[str, object]] = []
        self.validate_inputs: list[dict[str, object]] = []

    def reset(self, current_qpos_by_side) -> dict[str, object]:
        self.reset_inputs.append(
            {
                str(side): np.asarray(value, dtype=float).copy()
                for side, value in dict(current_qpos_by_side).items()
            }
        )
        return {
            "initialized": True,
            "collision_pair_count": self.collision_pair_count,
            "collision_distance_m": 0.08,
            "temporal_smoothing_applied": False,
        }

    def plan_pose_reference(self, pose_by_side, **kwargs):
        sequence = len(self.plan_inputs)
        poses = {
            str(side): np.asarray(value, dtype=float).copy()
            for side, value in dict(pose_by_side).items()
        }
        self.plan_inputs.append({"poses": poses, "kwargs": dict(kwargs)})
        qpos = {
            "left": np.full(6, 0.001 * sequence, dtype=float),
            "right": np.full(6, -0.001 * sequence, dtype=float),
        }
        failure = None
        if self.hold or not self.valid:
            failure = {"code": "collision_margin_low", "error": "fake hold"}
        return SimpleNamespace(
            sequence=sequence,
            status="hold_last_safe" if self.hold else "tracking",
            valid=self.valid,
            hold=self.hold,
            modification_reasons=("collision_margin_low",) if self.hold else (),
            q_ik_raw_by_side={side: value + 0.01 for side, value in qpos.items()},
            q_ref_by_side=qpos,
            target_endpoint_pose_world_by_side=poses,
            realized_endpoint_pose_world_by_side=poses,
            metadata={
                "planner": "fake_coupled_position_collision",
                "scene_profile": "dual_arm_hand_proxy_open",
                "collision_pair_count": self.collision_pair_count,
                "collision_distance_m": 0.08,
                "collision_min_distance_m": 0.03,
                "collision_distance_epsilon_m": 1e-4,
                "endpoint_collision_checked": True,
                "endpoint_collision_passed": not self.hold and self.valid,
                "segment_collision_checked": False,
                "segment_collision_certified": False,
                "solve_ms": 0.2,
                "frame_total_ms": 0.3,
                "failure": failure,
            },
        )

    def validate_qpos(self, qpos_by_side, *, source: str):
        qpos = {
            str(side): np.asarray(value, dtype=float).copy()
            for side, value in dict(qpos_by_side).items()
        }
        self.validate_inputs.append({"qpos": qpos, "source": str(source)})
        safe = bool(self.singularity_slow_recheck_safe)
        return SimpleNamespace(
            safe=safe,
            reason=None if safe else "fake_singularity_slow_target_collision",
            metadata={
                "scene_profile": "dual_arm_hand_proxy_open",
                "collision_pair_count": self.collision_pair_count,
                "endpoint_collision_checked": True,
                "endpoint_collision_passed": safe,
                "segment_collision_checked": False,
                "segment_collision_certified": False,
            },
        )


class _SingularityFaultReplayRig(_FakeReplayWorkerRig):
    def send_joint_targets(self, qpos_by_side, *, source: str, metadata=None) -> None:
        super().send_joint_targets(qpos_by_side, source=source, metadata=metadata)
        raise RuntimeError(
            "left_arm replay command blocked by singularity gate: "
            "status=hold_entry source=joint_vector target=wrist_singularity current=normal"
        )


class _SlowSingularityReplayRig(_DualFakeReplayWorkerRig):
    def prepare_replay_singularity_targets(self, qpos_by_side, *, source: str):
        targets = {
            str(side): np.asarray(qpos, dtype=float).copy()
            for side, qpos in dict(qpos_by_side).items()
        }
        targets["left"][0] += 0.02
        return targets, {
            "left": {
                "applied": True,
                "source": str(source),
                "max_joint_step_rad": [0.1] * 6,
            },
            "right": {"applied": False, "source": str(source)},
        }


class _BlockingDispatchReplayRig(_FakeReplayWorkerRig):
    def __init__(self) -> None:
        super().__init__()
        self.dispatch_entered = threading.Event()
        self.dispatch_release = threading.Event()

    def send_joint_targets(self, qpos_by_side, *, source: str, metadata=None) -> None:
        self.dispatch_entered.set()
        assert self.dispatch_release.wait(timeout=1.0)
        super().send_joint_targets(qpos_by_side, source=source, metadata=metadata)


class _CaptureBackend:
    def __init__(self) -> None:
        self.commands = []

    def is_ready(self) -> bool:
        return True

    def send_command(self, command) -> None:
        self.commands.append(command.copy())


class _PrecheckedSlowTargetCaptureBackend(_CaptureBackend):
    def send_command(self, command) -> None:
        command.metadata["backend_final_target_qpos"] = np.asarray(
            command.value,
            dtype=float,
        ).reshape(6).copy()
        super().send_command(command)


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
    def test_deploy_dataset_replay_uses_explicit_hand_target_for_invalid_action_slots(self) -> None:
        actions = np.zeros((2, ACTION_DIM), dtype=np.float32)
        actions[1, ACTION_HAND_SLICES["right"]] = np.nan
        explicit_targets = np.asarray(
            [[0.2, 0, 0, 0, 0, 0], [0.3, 0, 0, 0, 0, 0]],
            dtype=np.float32,
        )
        with (
            mock.patch(
                "teleop_runtime.deploy.replay.resolve_replay_parquet_path",
                return_value=(Path("/tmp/fake.parquet"), {"episode_index": 0}),
            ),
            mock.patch(
                "teleop_runtime.deploy.replay.load_replay_actions_from_parquet",
                return_value=actions,
            ),
            mock.patch(
                "teleop_runtime.deploy.replay.load_replay_hand_joint_targets_from_parquet",
                return_value={"right": explicit_targets},
            ),
            mock.patch(
                "teleop_runtime.deploy.replay.infer_replay_action_hz",
                return_value={"hz": 15.0, "source": "test"},
            ),
        ):
            loaded = load_dataset_replay_policy(
                dataset_root=None,
                parquet_path="/tmp/fake.parquet",
                episode_index=0,
                active_sides=("right",),
            )

        np.testing.assert_allclose(
            loaded.policy.actions[0, ACTION_HAND_SLICES["right"]],
            np.zeros(6, dtype=np.float32),
        )
        np.testing.assert_allclose(
            loaded.policy.actions[1, ACTION_HAND_SLICES["right"]],
            explicit_targets[1],
        )
        self.assertEqual(loaded.metadata["explicit_hand_replay_sides"], ["right"])

    def test_replay_frame_time_loader_preserves_valid_jitter_and_marks_fallback(self) -> None:
        class _FakeParquet:
            def __init__(self, timestamps):
                self.timestamps = list(timestamps)

            def read_schema(self, path):
                del path
                return SimpleNamespace(names=["timestamp"])

            def read_table(self, path, columns):
                del path, columns
                return SimpleNamespace(
                    to_pylist=lambda: [
                        {"timestamp": value} for value in self.timestamps
                    ]
                )

        with mock.patch(
            "teleop_runtime.deploy.replay._require_pyarrow_parquet",
            return_value=_FakeParquet((100.0, 100.071, 100.139)),
        ):
            times, metadata = load_replay_frame_times_from_parquet(
                "/tmp/fake.parquet",
                fallback_hz=15.0,
            )
        np.testing.assert_allclose(times, (0.0, 0.071, 0.139))
        self.assertEqual(metadata["source_timestamp_mode"], "dataset")

        with mock.patch(
            "teleop_runtime.deploy.replay._require_pyarrow_parquet",
            return_value=_FakeParquet((100.0, 100.0, 100.2)),
        ):
            times, metadata = load_replay_frame_times_from_parquet(
                "/tmp/fake.parquet",
                fallback_hz=10.0,
            )
        np.testing.assert_allclose(times, (0.0, 0.1, 0.2))
        self.assertEqual(metadata["source_timestamp_mode"], "nominal_fallback")

        class _MissingTimestampParquet:
            def read_schema(self, path):
                del path
                return SimpleNamespace(names=[])

            def read_metadata(self, path):
                del path
                return SimpleNamespace(num_rows=2)

        with mock.patch(
            "teleop_runtime.deploy.replay._require_pyarrow_parquet",
            return_value=_MissingTimestampParquet(),
        ):
            times, metadata = load_replay_frame_times_from_parquet(
                "/tmp/fake.parquet",
                fallback_hz=20.0,
            )
        np.testing.assert_allclose(times, (0.0, 0.05))
        self.assertEqual(
            metadata["source_timestamp_fallback_reason"],
            "timestamp_column_missing",
        )

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

    def test_explicit_hand_replay_target_loader_requires_complete_finite_side(self) -> None:
        class _FakeParquet:
            def read_schema(self, path):
                del path
                return SimpleNamespace(
                    names=["action.replay.right_hand_joint_target"]
                )

            def read_table(self, path, columns):
                del path, columns
                return SimpleNamespace(
                    to_pylist=lambda: [
                        {
                            "action.replay.right_hand_joint_target": [
                                0.2,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                            ]
                        },
                        {
                            "action.replay.right_hand_joint_target": [
                                0.3,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                            ]
                        },
                    ]
                )

        with mock.patch(
            "teleop_runtime.deploy.replay._require_pyarrow_parquet",
            return_value=_FakeParquet(),
        ):
            targets = load_replay_hand_joint_targets_from_parquet(
                "/tmp/fake.parquet",
                active_sides=("right",),
            )

        np.testing.assert_allclose(
            targets["right"],
            np.asarray(
                [[0.2, 0, 0, 0, 0, 0], [0.3, 0, 0, 0, 0, 0]],
                dtype=np.float32,
            ),
        )

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
            group = _group(
                profile,
                hand_targets=True,
                hand_spec=True,
            )

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

    def test_replay_action_frame_falls_back_to_explicit_hand_replay_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(
                Path(tmpdir) / "right.json",
                side="right",
                tcp=[0.01, 0, 0, 0, 0, 0],
            )
            group = _group(profile, hand_targets=True, hand_spec=True)
            group.actions[:, ACTION_HAND_SLICES["right"]] = np.nan
            group.hand_joint_targets["right"] = np.asarray(
                [[0.25, 0, 0, 0, 0, 0], [0.35, 0, 0, 0, 0, 0]],
                dtype=np.float32,
            )

            command = replay_teleop_command_at_index(
                group,
                ("right",),
                index=0,
                include_end_effectors=True,
                source="test_explicit_hand_replay_target",
            )

            np.testing.assert_allclose(
                command.commands["right_hand"].value,
                np.asarray([0.25], dtype=np.float32),
            )
            report = run_replay_preflight(
                group=group,
                active_sides=("right",),
                current_qpos_by_side={"right": np.zeros(6)},
                current_tcp_pose_by_side={"right": np.eye(4)},
                current_tcp_flange_by_side={
                    "right": np.asarray([0.01, 0, 0, 0, 0, 0])
                },
                end_effectors_enabled=True,
                end_effector_backend_sides=("right",),
            )
            self.assertTrue(report.ok)

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

    def test_controller_preflight_blocks_when_collision_planner_cannot_initialize(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_profile = _write_profile(
                Path(tmpdir) / "left.json", side="left", tcp=[0, 0, 0, 0, 0, 0]
            )
            right_profile = _write_profile(
                Path(tmpdir) / "right.json", side="right", tcp=[0, 0, 0, 0, 0, 0]
            )
            group = _dual_group(left_profile, right_profile)

            def failing_factory(**kwargs):
                del kwargs
                raise RuntimeError("pink unavailable")

            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("left", "right"),
                    connect_hardware=False,
                ),
                settings=ReplayExecutionSettings(send_actions_enabled=True),
                position_collision_planner_factory=failing_factory,
            )
            controller._rig = _DualFakeReplayWorkerRig()  # type: ignore[assignment]
            base_report = ReplayPreflightReport(
                checks=(ReplayCheckResult(name="base", ok=True),)
            )

            with (
                mock.patch(
                    "teleop_runtime.replay.executor.build_episode_replay_group",
                    return_value=group,
                ),
                mock.patch(
                    "teleop_runtime.replay.executor.run_replay_preflight",
                    return_value=base_report,
                ),
            ):
                status = controller.preflight(object())  # type: ignore[arg-type]

            self.assertEqual(status["state"], "preflight_failed")
            check = status["preflight"]["checks"][-1]
            self.assertEqual(check["name"], "position_collision_planner_ready")
            self.assertFalse(check["ok"])
            self.assertIn("pink unavailable", check["message"])

    def test_executor_blocks_motion_until_output_is_enabled(self) -> None:
        controller = ReplayExecutionController()
        with self.assertRaisesRegex(RuntimeError, "Arm output is disabled"):
            controller.move_to_start()

    def test_executor_blocks_move_to_start_before_required_collision_contract(self) -> None:
        single_arm = ReplayExecutionController(
            hardware_config=ReplayHardwareConfig(
                active_sides=("right",),
                connect_hardware=False,
            ),
            settings=ReplayExecutionSettings(send_actions_enabled=True),
        )
        with self.assertRaisesRegex(RuntimeError, "requires active_sides=both"):
            single_arm.move_to_start()

        qpos_mode = ReplayExecutionController(
            hardware_config=ReplayHardwareConfig(
                active_sides=("left", "right"),
                connect_hardware=False,
            ),
            settings=ReplayExecutionSettings(
                send_actions_enabled=True,
                trajectory_source=REPLAY_TRAJECTORY_QPOS_TARGET,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "accepts target poses only"):
            qpos_mode.move_to_start()

        disabled_real_output = ReplayExecutionController(
            hardware_config=ReplayHardwareConfig(
                active_sides=("left", "right"),
                connect_hardware=False,
            ),
            settings=ReplayExecutionSettings(
                send_actions_enabled=True,
                position_collision_required=False,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "non-motion dry-run only"):
            disabled_real_output.move_to_start()

    def test_executor_abort_holds_hardware_and_restores_speed(self) -> None:
        controller = ReplayExecutionController()
        rig = _FakeReplayRig()
        controller._rig = rig  # type: ignore[assignment]
        controller._state = "replaying"

        status = controller.abort()

        self.assertEqual(status["state"], "aborted")
        self.assertEqual(rig.hold_reasons, ["hardware replay abort"])
        self.assertEqual(rig.restore_count, 1)

    def test_abort_dispatch_barrier_prevents_command_after_hold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile)
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    trajectory_source=REPLAY_TRAJECTORY_QPOS_TARGET,
                    position_collision_required=False,
                ),
            )
            rig = _BlockingDispatchReplayRig()
            controller._rig = rig  # type: ignore[assignment]
            controller._state = "replaying"
            controller._stop_event.clear()
            worker = threading.Thread(target=controller._replay_worker, args=(rig, group))  # type: ignore[arg-type]
            controller._worker = worker
            worker.start()
            self.assertTrue(rig.dispatch_entered.wait(timeout=1.0))

            abort_thread = threading.Thread(target=lambda: controller.abort(wait=True))
            abort_thread.start()
            time.sleep(0.02)
            self.assertTrue(abort_thread.is_alive())
            self.assertEqual(rig.hold_reasons, [])
            rig.dispatch_release.set()
            abort_thread.join(timeout=1.0)
            worker.join(timeout=1.0)

            self.assertFalse(abort_thread.is_alive())
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(rig.sent_joint_targets), 1)
            self.assertEqual(rig.hold_reasons, ["hardware replay abort"])
            self.assertEqual(controller.status()["state"], "aborted")

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

    def test_position_collision_settings_payload_and_cli_are_explicit(self) -> None:
        settings = ReplayExecutionSettings.from_payload(
            {
                "position_collision_required": False,
                "position_collision_scene_profile": "test_scene",
                "position_collision_min_distance_m": 0.04,
                "position_collision_distance_epsilon_m": 0.002,
                "position_collision_max_joint_velocity_rad_s": "0.3",
                "position_collision_max_joint_step_rad": "0.01,0.02,0.03,0.04,0.05,0.06",
                "position_collision_debug": True,
            },
            ReplayExecutionSettings(),
        )

        payload = settings.to_dict()
        self.assertEqual(
            payload["position_control_profile_name"],
            "rm65_bimanual_position_v1",
        )
        self.assertFalse(payload["position_collision_required"])
        self.assertEqual(payload["position_collision_scene_profile"], "test_scene")
        self.assertEqual(
            payload["position_collision_max_joint_velocity_rad_s"],
            [0.3] * 6,
        )
        self.assertEqual(
            payload["position_collision_max_joint_step_rad"],
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        )
        self.assertTrue(payload["position_collision_debug"])

        args = build_replay_web_gui_parser().parse_args(
            [
                "--no-position-collision-required",
                "--position-collision-scene-profile",
                "test_scene",
                "--position-collision-debug",
            ]
        )
        self.assertFalse(args.position_collision_required)
        self.assertEqual(args.position_collision_scene_profile, "test_scene")
        self.assertTrue(args.position_collision_debug)
        self.assertEqual(args.position_control_profile, "rm65_bimanual_position_v1")

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

    def test_replay_hardware_config_position_defaults_to_150hz_follow(self) -> None:
        config = ReplayHardwareConfig(active_sides=("right",), connect_hardware=False)

        self.assertEqual(config.arm_control_mode, "position")
        self.assertEqual(config.arm_control_hz, DEFAULT_RM65_POSITION_ARM_CONTROL_HZ)
        self.assertTrue(config.arm_canfd_follow)

        rig = ReplayHardwareRig(config)
        try:
            rig.connect()
            backend = rig.arm_backends["right"]
            self.assertIsInstance(backend, RM65ArmBackend)
            self.assertEqual(backend.config.control_dt, 1.0 / DEFAULT_TELEOP_POSITION_REFERENCE_HZ)
            self.assertEqual(backend.config.background_streaming_hz, DEFAULT_RM65_POSITION_ARM_CONTROL_HZ)
            self.assertTrue(backend.config.canfd_follow)
            rig.set_arm_command_dt_sec(0.1)
            self.assertEqual(backend.config.control_dt, 0.1)
            self.assertEqual(backend.get_state()["background_streaming"]["command_control_dt_sec"], 0.1)
        finally:
            rig.close()

    def test_replay_hardware_config_impedance_defaults_to_40hz_no_follow(self) -> None:
        config = ReplayHardwareConfig.from_payload(
            {
                "active_sides": "right",
                "connect_hardware": False,
                "arm_control_mode": "impedance",
            }
        )

        self.assertEqual(config.arm_control_mode, "impedance")
        self.assertEqual(config.arm_control_hz, DEFAULT_RM65_ARM_CONTROL_HZ)
        self.assertFalse(config.arm_canfd_follow)

        rig = ReplayHardwareRig(config)
        try:
            rig.connect()
            backend = rig.arm_backends["right"]
            self.assertIsInstance(backend, RM65ImpedanceArmBackend)
            self.assertEqual(backend.config.control_dt, 1.0 / DEFAULT_RM65_ARM_CONTROL_HZ)
            self.assertEqual(backend.config.impedance_overrides["canfd_follow"], False)
            overrides = backend.config.impedance_overrides
            self.assertTrue(overrides["joint_rate_limit_use_nominal_dt"])
            self.assertTrue(overrides["admittance_use_nominal_dt"])
            self.assertEqual(overrides["joint_operating_margin_deg"], 3.0)
            self.assertTrue(overrides["ik_hard_deadline_enabled"])
            self.assertEqual(overrides["ik_resolver_local_step_max_position_worsen_m"], 0.003)
            self.assertEqual(overrides["ik_resolver_local_step_max_orientation_worsen_deg"], 0.5)
            self.assertFalse(overrides["ik_resolver_local_step_dynamic_worsen_enabled"])
            self.assertEqual(overrides["direct_joint_hint_max_linear_offset_m"], 0.003)
            self.assertAlmostEqual(overrides["direct_joint_hint_max_angular_offset_rad"], np.deg2rad(0.5))
            self.assertTrue(backend.get_state()["impedance_enabled"])
            rig.set_arm_command_dt_sec(1.0 / DEFAULT_TELEOP_SAMPLE_HZ)
            self.assertEqual(backend.config.control_dt, 1.0 / DEFAULT_RM65_ARM_CONTROL_HZ)
            self.assertEqual(backend.config.action_hz, DEFAULT_TELEOP_SAMPLE_HZ)
        finally:
            rig.close()

    def test_replay_hardware_config_parses_string_booleans(self) -> None:
        config = ReplayHardwareConfig.from_payload(
            {
                "active_sides": "right",
                "connectHardware": "false",
                "backgroundStreamingEnabled": "false",
                "endEffectorsEnabled": "true",
                "strictToolProfile": "false",
                "armControlMode": "position",
                "armCanfdFollow": "false",
            }
        )

        self.assertFalse(config.connect_hardware)
        self.assertFalse(config.background_streaming_enabled)
        self.assertTrue(config.end_effectors_enabled)
        self.assertFalse(config.strict_tool_profile)
        self.assertFalse(config.arm_canfd_follow)

    def test_replay_web_gui_parser_defers_hz_and_follow_to_mode_defaults(self) -> None:
        parser = build_replay_web_gui_parser()
        position_args = parser.parse_args([])
        self.assertEqual(position_args.active_sides, "both")
        np.testing.assert_allclose(
            [
                float(value)
                for value in (
                    position_args.position_collision_max_joint_velocity_rad_s.split(",")
                )
            ],
            np.deg2rad([170.0, 170.0, 190.0, 210.0, 240.0, 240.0]),
        )
        position_config = ReplayHardwareConfig(
            active_sides=("right",),
            arm_control_mode=position_args.arm_control_mode,
            arm_control_hz=position_args.arm_control_hz,
            arm_canfd_follow=position_args.arm_canfd_follow,
            connect_hardware=False,
        )

        self.assertEqual(position_config.arm_control_hz, DEFAULT_RM65_POSITION_ARM_CONTROL_HZ)
        self.assertTrue(position_config.arm_canfd_follow)

        impedance_args = parser.parse_args(["--arm-control-mode", "impedance"])
        impedance_config = ReplayHardwareConfig(
            active_sides=("right",),
            arm_control_mode=impedance_args.arm_control_mode,
            arm_control_hz=impedance_args.arm_control_hz,
            arm_canfd_follow=impedance_args.arm_canfd_follow,
            connect_hardware=False,
        )

        self.assertEqual(impedance_config.arm_control_hz, DEFAULT_RM65_ARM_CONTROL_HZ)
        self.assertFalse(impedance_config.arm_canfd_follow)

    def test_replay_hardware_config_defaults_to_collision_safe_dual_arm_topology(self) -> None:
        self.assertEqual(ReplayHardwareConfig(connect_hardware=False).active_sides, ("left", "right"))
        self.assertEqual(
            ReplayHardwareConfig.from_payload({"connect_hardware": False}).active_sides,
            ("left", "right"),
        )

    def test_replay_web_gui_resolves_shared_dmtac_assignment_and_manual_override(self) -> None:
        parser = build_replay_web_gui_parser()
        args = parser.parse_args(
            [
                "--active-sides",
                "both",
                "--end-effectors-enabled",
                "--hand-backend",
                "dmtac-gripper",
                "--dmtac-config",
                "fixture-assignment",
                "--right-gripper-port",
                "/dev/manual-right",
                "--right-gripper-id",
                "9",
                "--right-gripper-baudrate",
                "19200",
                "--right-gripper-timeout",
                "0.2",
            ]
        )
        assignment_configs = {
            "left": SimpleNamespace(
                connect_hardware=True,
                gripper_port="/dev/shared-left",
                gripper_id=3,
                gripper_baudrate=57600,
                gripper_timeout=0.04,
            ),
            "right": SimpleNamespace(
                connect_hardware=True,
                gripper_port="/dev/shared-right",
                gripper_id=4,
                gripper_baudrate=38400,
                gripper_timeout=0.05,
            ),
        }
        with (
            mock.patch(
                "teleop_runtime.replay.web_gui.load_dmtac_hardware_assignment",
                return_value={"schema": "fixture"},
            ) as load_assignment,
            mock.patch(
                "teleop_runtime.replay.web_gui.dmtac_backend_configs_from_assignment",
                return_value=assignment_configs,
            ),
        ):
            fields = _resolve_replay_dmtac_fields(
                args,
                active_sides=("left", "right"),
                end_effector_backend_by_side={
                    "left": "dmtac_gripper",
                    "right": "dmtac_gripper",
                },
            )

        load_assignment.assert_called_once_with(
            "fixture-assignment",
            validate_current_usb=True,
            allow_missing=False,
            assume_yes_for_extra=True,
        )
        self.assertEqual(fields["port"]["left"], "/dev/shared-left")
        self.assertEqual(fields["id"]["left"], 3)
        self.assertEqual(fields["baudrate"]["left"], 57600)
        self.assertEqual(fields["timeout"]["left"], 0.04)
        self.assertEqual(fields["port"]["right"], "/dev/manual-right")
        self.assertEqual(fields["id"]["right"], 9)
        self.assertEqual(fields["baudrate"]["right"], 19200)
        self.assertEqual(fields["timeout"]["right"], 0.2)

    def test_replay_hardware_config_preserves_dmtac_transport_settings(self) -> None:
        config = ReplayHardwareConfig.from_payload(
            {
                "active_sides": "left",
                "end_effectors_enabled": True,
                "end_effector_backend_by_side": {"left": "dmtac_gripper"},
                "gripper_port_by_side": {"left": "/dev/shared-left"},
                "gripper_id_by_side": {"left": 7},
                "gripper_baudrate_by_side": {"left": 57600},
                "gripper_timeout_by_side": {"left": 0.04},
                "connect_hardware": False,
            }
        )
        self.assertEqual(config.to_dict()["gripper_baudrate_by_side"]["left"], 57600)
        self.assertEqual(config.to_dict()["gripper_timeout_by_side"]["left"], 0.04)

        rig = ReplayHardwareRig(config)
        backend = rig._build_end_effector_backend("left")
        self.assertEqual(backend.config.gripper_port, "/dev/shared-left")
        self.assertEqual(backend.config.gripper_id, 7)
        self.assertEqual(backend.config.gripper_baudrate, 57600)
        self.assertEqual(backend.config.gripper_timeout, 0.04)

    def test_replay_configure_payload_mode_switch_recomputes_mode_defaults(self) -> None:
        controller = ReplayExecutionController(
            hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False)
        )

        status = controller.configure_from_payload(
            {
                "active_sides": "right",
                "connect_hardware": False,
                "arm_control_mode": "impedance",
            }
        )

        self.assertEqual(status["hardware_config"]["arm_control_mode"], "impedance")
        self.assertEqual(status["hardware_config"]["arm_control_hz"], DEFAULT_RM65_ARM_CONTROL_HZ)
        self.assertFalse(status["hardware_config"]["arm_canfd_follow"])

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

    def test_collision_required_position_replay_plans_reference_pose_and_sends_only_qpos(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_profile = _write_profile(
                Path(tmpdir) / "left.json",
                side="left",
                tcp=[0.01, 0, 0, 0, 0, 0],
            )
            right_profile = _write_profile(
                Path(tmpdir) / "right.json",
                side="right",
                tcp=[0.02, 0, 0, 0, 0, 0],
            )
            group = _dual_group(left_profile, right_profile)
            planner = _FakePositionCollisionPlanner()
            factory_kwargs: dict[str, object] = {}

            def planner_factory(**kwargs):
                factory_kwargs.update(kwargs)
                return planner

            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("left", "right"),
                    connect_hardware=False,
                ),
                settings=ReplayExecutionSettings(send_actions_enabled=True),
                position_collision_planner_factory=planner_factory,
            )
            rig = _DualFakeReplayWorkerRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            self.assertEqual(controller.status()["state"], "finished")
            self.assertEqual(len(planner.reset_inputs), 1)
            self.assertEqual(set(planner.reset_inputs[0]), {"left", "right"})
            self.assertGreater(len(planner.plan_inputs), 1)
            self.assertEqual(len(rig.sent_joint_targets), len(planner.plan_inputs))
            self.assertEqual(len(rig.sent_action_poses), 0)
            self.assertEqual(len(rig.sent_commands), 0)
            np.testing.assert_allclose(
                factory_kwargs["endpoint_to_tool_by_side"]["left"][:3, 3],
                [0.01, 0.0, 0.0],
            )
            planner_config = factory_kwargs["config"]
            self.assertEqual(planner_config.solve_min_iterations, 2)
            self.assertEqual(planner_config.solve_max_iterations, 4)
            self.assertTrue(planner_config.solve_adaptive_iterations_enabled)
            self.assertAlmostEqual(planner_config.solve_iteration_reserve_ms, 5.0)
            self.assertAlmostEqual(planner_config.damping, 0.03)
            objective = planner_config.ik_tuning.objective_weights()
            self.assertAlmostEqual(objective.dual_arm_position_cost, 1.0)
            self.assertAlmostEqual(objective.dual_arm_orientation_cost, 0.35)
            np.testing.assert_allclose(
                planner_config.max_joint_velocity_rad_s,
                np.deg2rad([170.0, 170.0, 190.0, 210.0, 240.0, 240.0]),
            )
            np.testing.assert_allclose(
                rig.sent_joint_targets[-1]["right"],
                np.full(6, -0.001 * (len(planner.plan_inputs) - 1)),
            )
            metadata = rig.sent_joint_metadata[1]
            self.assertTrue(metadata["position_collision_required"])
            self.assertTrue(metadata["position_endpoint_collision_checked"])
            self.assertTrue(metadata["position_endpoint_collision_passed"])
            self.assertFalse(metadata["position_segment_collision_checked"])
            self.assertFalse(metadata["position_segment_collision_certified"])
            self.assertFalse(metadata["temporal_smoothing_applied"])
            self.assertNotIn("debug", metadata["position_collision"])

    def test_collision_required_replay_rechecks_singularity_slow_target_before_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_profile = _write_profile(
                Path(tmpdir) / "left.json", side="left", tcp=[0, 0, 0, 0, 0, 0]
            )
            right_profile = _write_profile(
                Path(tmpdir) / "right.json", side="right", tcp=[0, 0, 0, 0, 0, 0]
            )
            planner = _FakePositionCollisionPlanner()
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("left", "right"),
                    connect_hardware=False,
                ),
                settings=ReplayExecutionSettings(send_actions_enabled=True),
                position_collision_planner_factory=lambda **_kwargs: planner,
            )
            rig = _SlowSingularityReplayRig()

            controller._replay_worker(rig, _dual_group(left_profile, right_profile))  # type: ignore[arg-type]

            self.assertEqual(controller.status()["state"], "finished")
            self.assertGreater(len(planner.validate_inputs), 0)
            self.assertEqual(len(planner.validate_inputs), len(rig.sent_joint_targets))
            self.assertTrue(
                all(
                    item["source"].startswith("replay_collision_qpos_reference_")
                    and item["source"].endswith("hz_singularity_slow_recheck")
                    for item in planner.validate_inputs
                )
            )
            np.testing.assert_allclose(planner.validate_inputs[0]["qpos"]["left"], np.full(6, 0.0) + [0.02, 0, 0, 0, 0, 0])
            np.testing.assert_allclose(rig.sent_joint_targets[0]["left"], [0.02, 0, 0, 0, 0, 0])
            metadata = rig.sent_joint_metadata[0]
            self.assertTrue(metadata["position_singularity_replay_slowdown"]["collision_rechecked"])
            self.assertEqual(metadata["position_singularity_replay_slowdown"]["sides"], ["left"])
            self.assertTrue(
                metadata["position_collision"]["singularity_slow_target_recheck"]
                ["endpoint_collision_passed"]
            )

    def test_collision_required_replay_faults_before_dispatch_when_slow_target_recheck_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_profile = _write_profile(
                Path(tmpdir) / "left.json", side="left", tcp=[0, 0, 0, 0, 0, 0]
            )
            right_profile = _write_profile(
                Path(tmpdir) / "right.json", side="right", tcp=[0, 0, 0, 0, 0, 0]
            )
            planner = _FakePositionCollisionPlanner(singularity_slow_recheck_safe=False)
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("left", "right"),
                    connect_hardware=False,
                ),
                settings=ReplayExecutionSettings(send_actions_enabled=True),
                position_collision_planner_factory=lambda **_kwargs: planner,
            )
            rig = _SlowSingularityReplayRig()

            controller._replay_worker(rig, _dual_group(left_profile, right_profile))  # type: ignore[arg-type]

            status = controller.status()
            self.assertEqual(status["state"], "fault")
            self.assertIn("singularity slow target failed", status["last_error"])
            self.assertEqual(len(planner.validate_inputs), 1)
            self.assertEqual(rig.sent_joint_targets, [])
            self.assertEqual(rig.hold_reasons, ["hardware replay fault"])

    def test_collision_debug_adds_raw_arrays_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_profile = _write_profile(
                Path(tmpdir) / "left.json", side="left", tcp=[0, 0, 0, 0, 0, 0]
            )
            right_profile = _write_profile(
                Path(tmpdir) / "right.json", side="right", tcp=[0, 0, 0, 0, 0, 0]
            )
            planner = _FakePositionCollisionPlanner()
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("left", "right"),
                    connect_hardware=False,
                    end_effectors_enabled=True,
                ),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    end_effectors_enabled=True,
                    position_collision_debug=True,
                ),
                position_collision_planner_factory=lambda **kwargs: planner,
            )
            rig = _HealthyTransportSnapshotRig()

            controller._replay_worker(  # type: ignore[arg-type]
                rig,
                _dual_group(
                    left_profile,
                    right_profile,
                    hand_targets=True,
                    hand_spec=True,
                ),
            )

            debug = rig.sent_joint_metadata[0]["position_collision"]["debug"]
            self.assertEqual(set(debug["q_ik_raw_by_side"]), {"left", "right"})
            self.assertEqual(set(debug["q_ref_by_side"]), {"left", "right"})
            self.assertIn("planner_metadata", debug)
            diagnostics = controller.status()["position_collision_diagnostics"]
            self.assertEqual(diagnostics["plan_count"], len(planner.plan_inputs))
            self.assertEqual(diagnostics["event_count"], len(planner.plan_inputs))
            self.assertTrue(diagnostics["endpoint_collision_certified"])
            recorded = controller.position_collision_debug_snapshot(limit=2)
            self.assertEqual(len(recorded["events"]), 2)
            self.assertIn(
                "debug",
                recorded["events"][-1]["collision"],
            )
            snapshots = recorded["events"][-1]["transport_health_snapshots"]
            self.assertEqual(
                set(snapshots),
                {
                    "before_arm_reference",
                    "after_arm_reference_before_hand",
                    "immediately_before_end_effector_dispatch",
                },
            )
            for stage in snapshots:
                self.assertEqual(snapshots[stage]["status"], "checked")
                self.assertTrue(snapshots[stage]["observations"][-1]["ok"])
            self.assertAlmostEqual(
                snapshots["before_arm_reference"]["observations"][-1][
                    "health_by_side"
                ]["left"]["background_streaming"]["actual_hz"],
                150.0,
            )
            self.assertIn(
                "transport_health_snapshots",
                diagnostics["last"],
            )

    def test_collision_debug_marks_hand_health_not_applicable_without_hand_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_profile = _write_profile(
                Path(tmpdir) / "left.json", side="left", tcp=[0, 0, 0, 0, 0, 0]
            )
            right_profile = _write_profile(
                Path(tmpdir) / "right.json", side="right", tcp=[0, 0, 0, 0, 0, 0]
            )
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("left", "right"),
                    connect_hardware=False,
                ),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    position_collision_debug=True,
                ),
                position_collision_planner_factory=(
                    lambda **_kwargs: _FakePositionCollisionPlanner()
                ),
            )
            rig = _HealthyTransportSnapshotRig()

            controller._replay_worker(  # type: ignore[arg-type]
                rig,
                _dual_group(left_profile, right_profile),
            )

            event = controller.position_collision_debug_snapshot(limit=1)["events"][-1]
            slots = event["transport_health_snapshots"]
            self.assertEqual(slots["before_arm_reference"]["status"], "checked")
            self.assertEqual(
                slots["after_arm_reference_before_hand"]["status"],
                "checked",
            )
            self.assertEqual(
                slots["immediately_before_end_effector_dispatch"]["status"],
                "not_applicable",
            )
            self.assertEqual(
                slots["immediately_before_end_effector_dispatch"]["observations"],
                [],
            )

    def test_collision_hold_fails_closed_before_same_tick_hand_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_profile = _write_profile(
                Path(tmpdir) / "left.json", side="left", tcp=[0, 0, 0, 0, 0, 0]
            )
            right_profile = _write_profile(
                Path(tmpdir) / "right.json", side="right", tcp=[0, 0, 0, 0, 0, 0]
            )
            planner = _FakePositionCollisionPlanner(hold=True)
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("left", "right"),
                    connect_hardware=False,
                    end_effectors_enabled=True,
                ),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    end_effectors_enabled=True,
                    position_collision_debug=True,
                ),
                position_collision_planner_factory=lambda **kwargs: planner,
            )
            rig = _DualFakeReplayWorkerRig()

            controller._replay_worker(  # type: ignore[arg-type]
                rig,
                _dual_group(
                    left_profile,
                    right_profile,
                    hand_targets=True,
                    hand_spec=True,
                ),
            )

            status = controller.status()
            self.assertEqual(status["state"], "fault")
            self.assertIn("failed closed before arm/hand dispatch", status["last_error"])
            self.assertEqual(len(rig.sent_joint_targets), 0)
            self.assertEqual(len(rig.sent_action_poses), 0)
            self.assertEqual(len(rig.sent_commands), 0)
            self.assertEqual(rig.hold_reasons, ["hardware replay fault"])
            self.assertFalse(
                status["position_collision_diagnostics"][
                    "endpoint_collision_certified"
                ]
            )

    def test_latched_arm_stream_fault_blocks_same_tick_hand_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_profile = _write_profile(
                Path(tmpdir) / "left.json", side="left", tcp=[0, 0, 0, 0, 0, 0]
            )
            right_profile = _write_profile(
                Path(tmpdir) / "right.json", side="right", tcp=[0, 0, 0, 0, 0, 0]
            )
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("left", "right"),
                    connect_hardware=False,
                    end_effectors_enabled=True,
                ),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    end_effectors_enabled=True,
                    position_collision_debug=True,
                ),
                position_collision_planner_factory=(
                    lambda **_kwargs: _FakePositionCollisionPlanner()
                ),
            )
            rig = _HealthFaultBeforeHandRig()

            controller._replay_worker(  # type: ignore[arg-type]
                rig,
                _dual_group(
                    left_profile,
                    right_profile,
                    hand_targets=True,
                    hand_spec=True,
                ),
            )

            status = controller.status()
            self.assertEqual(status["state"], "fault")
            self.assertIn("150 Hz position stream fault", status["last_error"])
            self.assertEqual(len(rig.sent_joint_targets), 1)
            self.assertEqual(len(rig.sent_commands), 0)
            self.assertIn(
                "immediately_before_end_effector_dispatch",
                rig.health_stages,
            )
            recorded = controller.position_collision_debug_snapshot(limit=1)
            failed = recorded["events"][-1]["transport_health_snapshots"][
                "immediately_before_end_effector_dispatch"
            ]
            self.assertEqual(failed["status"], "failed")
            self.assertFalse(failed["observations"][-1]["ok"])
            self.assertIn("150 Hz position stream fault", failed["observations"][-1]["error"])

    def test_real_collision_reset_refuses_failed_fresh_hardware_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_profile = _write_profile(
                Path(tmpdir) / "left.json", side="left", tcp=[0, 0, 0, 0, 0, 0]
            )
            right_profile = _write_profile(
                Path(tmpdir) / "right.json", side="right", tcp=[0, 0, 0, 0, 0, 0]
            )
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("left", "right"),
                    connect_hardware=True,
                ),
                settings=ReplayExecutionSettings(send_actions_enabled=True),
                position_collision_planner_factory=(
                    lambda **_kwargs: _FakePositionCollisionPlanner()
                ),
            )
            rig = _FreshReadFailureRig()

            controller._replay_worker(  # type: ignore[arg-type]
                rig,
                _dual_group(left_profile, right_profile),
            )

            status = controller.status()
            self.assertEqual(status["state"], "fault")
            self.assertIn("fresh hardware qpos read failed", status["last_error"])
            self.assertEqual(len(rig.sent_joint_targets), 0)
            self.assertEqual(rig.begin_count, 0)

    def test_fresh_hardware_qpos_api_rejects_failed_or_stale_cache(self) -> None:
        class _StateBackend:
            def __init__(self, state):
                self.state = state

            def refresh_state_from_hardware(self):
                return dict(self.state)

        now = time.monotonic()
        cached = np.zeros(6, dtype=float)
        failed_state = {
            "hardware_read_success": False,
            "hardware_qpos": cached,
            "measured_timestamp_monotonic": now,
            "hardware_result_text": "refresh_failed",
        }
        rig = object.__new__(ReplayHardwareRig)
        rig.config = SimpleNamespace(active_sides=("left", "right"))
        rig.arm_backends = {
            "left": _StateBackend(failed_state),
            "right": _StateBackend(failed_state),
        }
        with self.assertRaisesRegex(RuntimeError, "fresh hardware qpos read failed"):
            rig.fresh_hardware_qpos_by_side(max_age_sec=0.25)

        stale_state = {
            "hardware_read_success": True,
            "hardware_measurement_valid": True,
            "hardware_ready": True,
            "hardware_qpos": cached,
            "measured_qpos": cached,
            "measured_timestamp_monotonic": now - 2.0,
            "hardware_read_timestamp_monotonic": now - 2.0,
            "hardware_read_sequence": 1,
            "measured_sequence": 1,
            "hardware_arm_err": 0,
            "hardware_sys_err": 0,
        }
        rig.arm_backends = {
            "left": _StateBackend(stale_state),
            "right": _StateBackend(stale_state),
        }
        with self.assertRaisesRegex(RuntimeError, "invalid/stale"):
            rig.fresh_hardware_qpos_by_side(max_age_sec=0.25)

    def test_replay_rig_rejects_backend_rewrite_of_coupled_qpos(self) -> None:
        class _RewriteBackend:
            def send_command(self, command):
                command.metadata["backend_final_target_qpos"] = np.zeros(6)
                return SimpleNamespace(accepted=True, status="queued")

            def get_state(self):
                return {
                    "singularity_metadata": {
                        "singularity_gate_status": "pass"
                    }
                }

        rig = object.__new__(ReplayHardwareRig)
        rig.effector_backends = {"left_arm": _RewriteBackend()}
        command = TeleopCommand()
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.full(6, 0.1),
                metadata={"position_collision_prelimited_qpos": True},
            )
        )
        with self.assertRaisesRegex(RuntimeError, "rewrote a coupled collision qpos"):
            rig.send_commands(command)

    def test_collision_required_rejects_qpos_target_legacy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_profile = _write_profile(
                Path(tmpdir) / "left.json", side="left", tcp=[0, 0, 0, 0, 0, 0]
            )
            right_profile = _write_profile(
                Path(tmpdir) / "right.json", side="right", tcp=[0, 0, 0, 0, 0, 0]
            )
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("left", "right"),
                    connect_hardware=False,
                ),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    trajectory_source=REPLAY_TRAJECTORY_QPOS_TARGET,
                ),
            )
            rig = _DualFakeReplayWorkerRig()

            controller._replay_worker(  # type: ignore[arg-type]
                rig,
                _dual_group(left_profile, right_profile),
            )

            status = controller.status()
            self.assertEqual(status["state"], "fault")
            self.assertIn("accepts target poses only", status["last_error"])
            self.assertIn("position_collision_required=false", status["last_error"])
            self.assertEqual(len(rig.sent_joint_targets), 0)
            self.assertEqual(rig.begin_count, 0)

    def test_collision_required_rejects_single_active_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(
                Path(tmpdir) / "right.json", side="right", tcp=[0, 0, 0, 0, 0, 0]
            )
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("right",),
                    connect_hardware=False,
                ),
                settings=ReplayExecutionSettings(send_actions_enabled=True),
            )
            rig = _DualFakeReplayWorkerRig()

            controller._replay_worker(rig, _group(profile))  # type: ignore[arg-type]

            status = controller.status()
            self.assertEqual(status["state"], "fault")
            self.assertIn("requires active_sides=both", status["last_error"])
            self.assertEqual(len(rig.sent_joint_targets), 0)

    def test_replay_worker_dispatches_action_poses_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile, hand_targets=True, hand_spec=True)
            observer_queue: queue.Queue = queue.Queue(maxsize=4)
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    position_collision_required=False,
                ),
                action_observer_queue=observer_queue,
            )
            rig = _FakeReplayWorkerRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            self.assertEqual(controller.status()["state"], "finished")
            self.assertEqual(
                rig.command_dt_history,
                [1.0 / DEFAULT_TELEOP_POSITION_REFERENCE_HZ],
            )
            self.assertEqual(rig.begin_count, 1)
            self.assertEqual(rig.restore_count, 1)
            self.assertEqual(len(rig.sent_joint_targets), 0)
            self.assertGreater(len(rig.sent_action_poses), 1)
            self.assertEqual(len(rig.sent_commands), 0)
            np.testing.assert_allclose(
                rig.sent_action_poses[0]["right"],
                np.asarray([0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]),
            )
            np.testing.assert_allclose(
                rig.sent_action_poses[-1]["right"],
                np.asarray([0.21, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]),
            )
            self.assertEqual(
                rig.sent_action_metadata[1]["position_pipeline"],
                (
                    f"safe_10hz_to_reference_"
                    f"{DEFAULT_TELEOP_POSITION_REFERENCE_HZ:g}hz_to_command_"
                    f"{DEFAULT_RM65_POSITION_ARM_CONTROL_HZ:g}hz"
                ),
            )
            self.assertEqual(
                rig.sent_action_metadata[1]["position_default_profile_contract"],
                (
                    f"safe_{DEFAULT_TELEOP_SAMPLE_HZ:g}hz_to_reference_"
                    f"{DEFAULT_TELEOP_POSITION_REFERENCE_HZ:g}hz_to_command_"
                    f"{DEFAULT_RM65_POSITION_ARM_CONTROL_HZ:g}hz"
                ),
            )
            self.assertFalse(rig.sent_action_metadata[1]["temporal_smoothing_applied"])
            observed = [observer_queue.get_nowait(), observer_queue.get_nowait()]
            self.assertEqual([item.local_frame_index for item in observed], [0, 1])
            self.assertEqual([item.source_action_index for item in observed], [0, 1])
            self.assertTrue(
                observed[0].target_effective_at < observed[1].target_effective_at
            )
            self.assertEqual(controller.status()["action_observer"]["offer_count"], 2)
            self.assertEqual(controller.status()["action_observer"]["drop_count"], 0)

    def test_replay_worker_preserves_impedance_action_rate_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(
                profile,
                hand_targets=True,
                hand_spec=True,
                frame_times_sec=(100.0, 100.073),
            )
            observer_queue: queue.Queue = queue.Queue(maxsize=4)
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("right",),
                    arm_control_mode="impedance",
                    connect_hardware=False,
                ),
                settings=ReplayExecutionSettings(send_actions_enabled=True),
                action_observer_queue=observer_queue,
            )
            rig = _FakeReplayWorkerRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            self.assertEqual(controller.status()["state"], "finished")
            self.assertEqual(rig.command_dt_history, [0.1])
            self.assertEqual(len(rig.sent_action_poses), 0)
            self.assertEqual(len(rig.sent_commands), 2)
            self.assertEqual(set(rig.sent_commands[0].commands), {"right_arm"})
            observed = [observer_queue.get_nowait(), observer_queue.get_nowait()]
            self.assertEqual([item.local_frame_index for item in observed], [0, 1])
            self.assertAlmostEqual(
                observed[1].source_relative_time - observed[0].source_relative_time,
                0.073,
            )
            self.assertAlmostEqual(
                observed[1].target_effective_at - observed[0].target_effective_at,
                0.1,
            )
            self.assertAlmostEqual(
                observed[0].effective_until - observed[0].effective_from,
                0.1,
            )
            self.assertEqual(controller.status()["action_observer"]["offer_count"], 2)

    def test_replay_action_observer_failure_never_changes_legacy_dispatch(self) -> None:
        class FailingObserverQueue(queue.Queue):
            def __init__(self) -> None:
                super().__init__(maxsize=4)

            def put_nowait(self, _item: object) -> None:
                raise RuntimeError("observer unavailable")

        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(
                Path(tmpdir) / "right.json",
                side="right",
                tcp=[0.01, 0, 0, 0, 0, 0],
            )
            group = _group(profile)
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("right",),
                    arm_control_mode="impedance",
                    connect_hardware=False,
                ),
                settings=ReplayExecutionSettings(send_actions_enabled=True),
                action_observer_queue=FailingObserverQueue(),
            )
            rig = _FakeReplayWorkerRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            status = controller.status()
            self.assertEqual(status["state"], "finished")
            self.assertEqual(len(rig.sent_commands), 2)
            self.assertEqual(status["action_observer"]["offer_count"], 0)
            self.assertEqual(status["action_observer"]["error_count"], 2)
            self.assertIn("observer unavailable", status["action_observer"]["last_error"])

    def test_position_replay_limits_keyframe_before_reference_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile)
            group.actions[1, ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
                [0.50, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    position_max_xyz_step_m=0.10,
                    position_collision_required=False,
                ),
            )
            rig = _FakeReplayWorkerRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            self.assertEqual(controller.status()["state"], "finished")
            np.testing.assert_allclose(rig.sent_action_poses[-1]["right"][:3], [0.30, 0.0, 0.30])
            safety = rig.sent_action_metadata[-1]["position_keyframe_safety"]
            self.assertIn("translation_step_limit", safety["reasons_by_side"]["right"])
            self.assertFalse(rig.sent_action_metadata[-1]["temporal_smoothing_applied"])

    def test_position_replay_preserves_source_timestamp_jitter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(
                Path(tmpdir) / "right.json",
                side="right",
                tcp=[0.01, 0, 0, 0, 0, 0],
            )
            group = _group(profile, frame_times_sec=(100.0, 100.073))
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("right",),
                    connect_hardware=False,
                ),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    position_collision_required=False,
                ),
            )
            rig = _FakeReplayWorkerRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            self.assertGreater(len(rig.sent_action_poses), 1)
            metadata = rig.sent_action_metadata[-1]
            self.assertEqual(metadata["position_source_timestamp_mode"], "dataset")
            self.assertAlmostEqual(metadata["position_source_frame_time_sec"], 0.073)

    def test_position_replay_metadata_uses_custom_effective_rates_without_follow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile)
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(
                    active_sides=("right",),
                    arm_control_hz=60.0,
                    position_reference_hz=20.0,
                    arm_canfd_follow=False,
                    connect_hardware=False,
                ),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    position_collision_required=False,
                ),
            )
            rig = _FakeReplayWorkerRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            self.assertEqual(controller.status()["state"], "finished")
            self.assertEqual(len(rig.sent_action_poses), 3)
            metadata = rig.sent_action_metadata[1]
            self.assertEqual(
                metadata["position_pipeline"],
                "safe_10hz_to_reference_20hz_to_command_20hz",
            )
            self.assertEqual(metadata["position_command_hz"], 20.0)
            self.assertFalse(metadata["position_background_streaming_active"])

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
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    end_effectors_enabled=True,
                    position_collision_required=False,
                ),
            )
            rig = _FakeReplayWorkerRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            self.assertEqual(controller.status()["state"], "finished")
            self.assertEqual(
                rig.command_dt_history,
                [1.0 / DEFAULT_TELEOP_POSITION_REFERENCE_HZ],
            )
            self.assertEqual(len(rig.sent_commands), 2)
            self.assertGreater(len(rig.sent_action_poses), 1)
            self.assertEqual(set(rig.sent_commands[0].commands), {"right_hand"})
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
                    position_collision_required=False,
                ),
            )
            rig = _FakeReplayWorkerRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            self.assertEqual(controller.status()["state"], "finished")
            self.assertEqual(
                rig.command_dt_history,
                [1.0 / DEFAULT_TELEOP_POSITION_REFERENCE_HZ],
            )
            self.assertEqual(rig.begin_count, 1)
            self.assertEqual(rig.restore_count, 1)
            self.assertEqual(len(rig.sent_action_poses), 0)
            self.assertGreater(len(rig.sent_joint_targets), 1)
            np.testing.assert_allclose(rig.sent_joint_targets[0]["right"], np.zeros(6))
            np.testing.assert_allclose(rig.sent_joint_targets[-1]["right"], np.zeros(6))
            self.assertFalse(rig.sent_joint_metadata[1]["temporal_smoothing_applied"])

    def test_replay_worker_fails_fast_on_singularity_gate_hold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile)
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    trajectory_source=REPLAY_TRAJECTORY_QPOS_TARGET,
                    position_collision_required=False,
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

    def test_hardware_rig_marks_only_collision_rechecked_replay_slow_target(self) -> None:
        rig = ReplayHardwareRig(
            ReplayHardwareConfig(active_sides=("left",), connect_hardware=False)
        )
        backend = _PrecheckedSlowTargetCaptureBackend()
        rig.arm_backends["left"] = backend  # type: ignore[assignment]
        rig.effector_backends["left_arm"] = backend  # type: ignore[assignment]
        qpos = np.full(6, 0.02, dtype=float)
        metadata = {
            "position_endpoint_collision_passed": True,
            "position_singularity_replay_slowdown": {"collision_rechecked": True},
            "singularity_replay_preparation_by_side": {
                "left": {
                    "applied": True,
                    "max_joint_step_rad": [0.1] * 6,
                }
            },
        }

        rig.send_joint_targets(
            {"left": qpos},
            source=(
                "replay_collision_qpos_reference_"
                f"{DEFAULT_TELEOP_POSITION_REFERENCE_HZ:g}hz"
            ),
            metadata=metadata,
        )

        self.assertEqual(len(backend.commands), 1)
        command = backend.commands[0]
        self.assertTrue(command.metadata["singularity_replay_slowdown_preapplied"])
        self.assertTrue(
            command.metadata["singularity_replay_slowdown_collision_rechecked"]
        )
        self.assertEqual(command.metadata["max_joint_step_rad"], [0.1] * 6)

        with self.assertRaisesRegex(RuntimeError, "missing the required coupled collision recheck"):
            rig.send_joint_targets(
                {"left": qpos},
                source=(
                    "replay_collision_qpos_reference_"
                    f"{DEFAULT_TELEOP_POSITION_REFERENCE_HZ:g}hz"
                ),
                metadata={
                    "singularity_replay_preparation_by_side": metadata[
                        "singularity_replay_preparation_by_side"
                    ]
                },
            )

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
            self.assertEqual(rig.command_dt_history, [0.001])
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
