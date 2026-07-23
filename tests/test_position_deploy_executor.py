from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

from robot_motion.deploy.actions import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    FilteredNormalizedAction,
    NormalizedAction,
)
from robot_motion.deploy.canonical_action import CanonicalDeployActionPipeline
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
)
from robot_motion.runtime import CommandDispatchReceipt, EffectorCommand, TeleopCommand
from teleop_runtime.backends.hands.inspire_tac import (
    InspireTacHandBackend,
    InspireTacHandBackendConfig,
)
from teleop_runtime.deploy.run_policy import (
    DeploymentDebugLogger,
    _PreparedActionExecution,
    _dispatch_position_discrete_command_checked,
)
from teleop_runtime.deploy.dispatch import (
    dispatch_teleop_command,
    require_arm_backends_ready_for_position_dispatch,
)
from teleop_runtime.position_control.deploy_executor import (
    PositionDeployExecutor,
    PositionDeployExecutorConfig,
    PositionDeployExecutorFault,
    PositionDeployQueueFullError,
)
from teleop_runtime.position_control.reference_observation import (
    LegacyPositionReferenceObservationBatch,
    LegacyPositionReferenceObserverRuntime,
    LegacyPositionReferenceResolutionFailureObservation,
    LegacyPositionReferenceResolutionObservation,
    LegacyPositionReferenceSegmentTerminalObservation,
)
from teleop_runtime.position_control.rm65_collision import (
    endpoint_to_tool_transforms_from_started_backends,
)


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    assert predicate(), "timed out waiting for background position executor"


def _pose(x: float, y: float = 0.0, z: float = 0.3) -> np.ndarray:
    return np.asarray([x, y, z, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _observation(
    *,
    sides: tuple[str, ...] = ("left",),
    left_pose: np.ndarray | None = None,
    right_pose: np.ndarray | None = None,
    left_qpos: np.ndarray | None = None,
    right_qpos: np.ndarray | None = None,
    arm_state_overrides: dict[str, dict[str, object]] | None = None,
) -> DeploymentObservation:
    now_monotonic = time.monotonic()
    state = np.full((OBSERVATION_STATE_DIM,), np.nan, dtype=np.float32)
    valid = np.zeros((OBSERVATION_STATE_DIM,), dtype=bool)
    values = {
        "left": (
            _pose(0.1) if left_pose is None else left_pose,
            np.zeros(6, dtype=np.float32) if left_qpos is None else left_qpos,
        ),
        "right": (
            _pose(0.1, -0.2) if right_pose is None else right_pose,
            np.zeros(6, dtype=np.float32) if right_qpos is None else right_qpos,
        ),
    }
    for side in sides:
        pose, qpos = values[side]
        state[STATE_ARM_POSE_SLICES[side]] = pose
        state[STATE_ARM_QPOS_SLICES[side]] = qpos
        valid[STATE_ARM_POSE_SLICES[side]] = True
        valid[STATE_ARM_QPOS_SLICES[side]] = True
    arm_states: dict[str, dict[str, object]] = {}
    for side in sides:
        qpos = np.asarray(values[side][1], dtype=float).copy()
        arm_states[side] = {
            "hardware_qpos": qpos.copy(),
            "measured_qpos": qpos.copy(),
            "hardware_read_success": True,
            "hardware_read_timestamp_monotonic": now_monotonic,
            "measured_timestamp_monotonic": now_monotonic,
            "hardware_read_sequence": 1,
            "measured_sequence": 1,
            "hardware_measurement_valid": True,
            "hardware_ready": True,
            "hardware_result_code": 0,
            "emergency_stopped": False,
        }
        arm_states[side].update((arm_state_overrides or {}).get(side, {}))
    return DeploymentObservation(
        timestamp=time.time(),
        step_index=0,
        active_sides=sides,
        arms_enabled=True,
        hands_enabled=False,
        state=state,
        state_valid=valid,
        arm_states=arm_states,
    )


def _pipeline(sides: tuple[str, ...]) -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=sides,
                arms_enabled=True,
                hands_enabled=False,
                max_arm_xyz_step_m=0.0,
                max_arm_rot_step_deg=0.0,
                max_arm_joint_step_rad=0.0,
            )
        ),
        active_sides=sides,
        arms_enabled=True,
        hands_enabled=False,
    )


def _filtered_pose(side: str, target_pose: np.ndarray) -> FilteredNormalizedAction:
    vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    vector[ACTION_ARM_POSE_SLICES[side]] = target_pose
    return FilteredNormalizedAction(
        action=NormalizedAction(vector=vector, metadata={"source": "unit_safe_15hz"}),
        metadata={"safety_pass_count": 1},
    )


def _hand_command(side: str, value: float) -> TeleopCommand:
    command = TeleopCommand(timestamp=123.0, metadata={"source": "safe_15hz_action"})
    command.add_command(
        EffectorCommand(
            effector_name=f"{side}_hand",
            side=side,
            command_type="joint_vector",
            value=np.full(6, value, dtype=np.float32),
        )
    )
    return command


def _accepted_collision_arm_dispatch(
    commands: list[TeleopCommand],
    command: TeleopCommand,
) -> dict[str, object]:
    commands.append(command)
    return {
        "accepted": True,
        "effectors": {
            name: {
                "accepted": True,
                "backend_ready": True,
                "command_type": "joint_vector",
                "requested_value": np.asarray(item.value, dtype=float).copy(),
                "final_value": np.asarray(item.value, dtype=float).copy(),
                "rewritten": False,
                "singularity_gate_status": "pass",
                "state_error": None,
                "background_streaming": {},
            }
            for name, item in command.commands.items()
        },
    }


class _FakePositionCollisionPlanner:
    def __init__(self, *, hold: bool = False, endpoint_checked: bool = True) -> None:
        self.hold = bool(hold)
        self.endpoint_checked = bool(endpoint_checked)
        self.reset_calls: list[dict[str, np.ndarray]] = []
        self.plan_calls: list[dict[str, np.ndarray]] = []
        self.sequence = 0

    def reset(self, qpos_by_side):
        copied = {
            str(side): np.asarray(value, dtype=float).copy()
            for side, value in dict(qpos_by_side).items()
        }
        self.reset_calls.append(copied)
        return {
            "initialized": True,
            "collision_distance_m": 0.08,
            "temporal_smoothing_applied": False,
        }

    def plan_pose_reference(self, pose_by_side, **_kwargs):
        poses = {
            str(side): np.asarray(value, dtype=float).copy()
            for side, value in dict(pose_by_side).items()
        }
        self.plan_calls.append(poses)
        sequence = self.sequence
        self.sequence += 1
        qpos = {
            side: np.full(6, float(pose[0, 3]), dtype=float)
            for side, pose in poses.items()
        }
        reasons = ("collision_margin_low", "hold_last_safe") if self.hold else ()
        return SimpleNamespace(
            sequence=sequence,
            status="hold_last_safe" if self.hold else "tracking",
            valid=True,
            hold=self.hold,
            modification_reasons=reasons,
            q_ik_raw_by_side=qpos,
            q_ref_by_side=qpos,
            target_endpoint_pose_world_by_side=poses,
            realized_endpoint_pose_world_by_side=poses,
            metadata={
                "planner": "fake_coupled_collision",
                "scene_profile": "fake_scene",
                "collision_pair_count": 3,
                "collision_distance_m": 0.08,
                "endpoint_collision_checked": self.endpoint_checked,
                "endpoint_collision_passed": not self.hold,
                "segment_collision_certified": False,
                "solve_ms": 0.2,
                "frame_total_ms": 0.3,
                "solve_time_budget_exceeded": False,
            },
        )


def _filtered_bimanual_pose(left_x: float, right_x: float) -> FilteredNormalizedAction:
    vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    vector[ACTION_ARM_POSE_SLICES["left"]] = _pose(left_x, 0.2)
    vector[ACTION_ARM_POSE_SLICES["right"]] = _pose(right_x, -0.2)
    return FilteredNormalizedAction(
        action=NormalizedAction(vector=vector, metadata={"source": "unit_safe_15hz"}),
        metadata={"safety_pass_count": 1},
    )


def test_collision_adapter_reads_started_backend_active_tcp_state() -> None:
    class _Backend:
        def __init__(self, pose6):
            self.pose6 = pose6

        def get_state(self):
            return {"tool_tcp_pose_flange_m_rad": self.pose6}

    transforms = endpoint_to_tool_transforms_from_started_backends(
        {
            "left": _Backend([0.01, 0.0, 0.12, 0.0, 0.0, 0.0]),
            "right": _Backend([0.02, 0.0, 0.13, 0.0, 0.0, 0.0]),
        }
    )
    np.testing.assert_allclose(transforms["left"][:3, 3], [0.01, 0.0, 0.12])
    np.testing.assert_allclose(transforms["right"][:3, 3], [0.02, 0.0, 0.13])
    with pytest.raises(RuntimeError, match="passive arm"):
        endpoint_to_tool_transforms_from_started_backends(
            {"right": _Backend(np.zeros(6))}
        )


def test_pose_segment_emits_reference_points_without_rerunning_safety() -> None:
    pipeline = _pipeline(("left",))

    def fail_if_safety_repeats(*_args, **_kwargs):
        raise AssertionError("15 Hz keyframe safety must not run at reference rate")

    pipeline.safety_filter.filter_normalized = fail_if_safety_repeats  # type: ignore[method-assign]
    commands = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=pipeline,
        dispatch_callback=lambda command: commands.append(command),
        wall_clock=lambda: 1_700_000_000.0,
    )
    executor.start()
    try:
        source_index = executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit_policy",
        )
        assert source_index > 0
        expected_points = int(round(executor.config.reference_hz / executor.config.sample_hz))
        _wait_until(lambda: len(commands) >= expected_points)

        first_segment = commands[:expected_points]
        x_values = [
            float(command.commands["left_arm"].metadata["target_xyzquat"][0])
            for command in first_segment
        ]
        alphas = [
            float(command.metadata["position_interpolation_alpha"])
            for command in first_segment
        ]
        assert x_values == sorted(x_values)
        assert x_values[0] >= 0.1 - 1e-5
        assert x_values[-1] < 0.16
        assert alphas == sorted(alphas)
        assert alphas[-1] > 0.5

        metadata = first_segment[1].metadata
        assert first_segment[1].timestamp == 1_700_000_000.0
        assert metadata["position_reference_emitted_timestamp"] == 1_700_000_000.0
        assert metadata["position_reference_emitted_monotonic_time"] < 1_700_000_000.0
        assert metadata["position_execution_contract"] == (
            f"safe_{executor.config.sample_hz:g}"
            f"->causal_{executor.config.reference_hz:g}"
            f"->{executor.config.command_hz:g}"
        )
        assert metadata["position_safety_reapplied_at_reference"] is False
        assert metadata["no_temporal_smoothing"] is True
        assert metadata["temporal_operation"] == "timestamp_interpolation_only"
        assert metadata["position_parent_source_indices"] == [source_index - 1, source_index]
        assert metadata["position_causal_output_lag_sec"] == pytest.approx(1.0 / 15.0)
        assert executor.drain_debug_events() == []
        assert executor.snapshot()["last_reference"]["position_arm_commands"] == {}
    finally:
        executor.close()


def test_position_reference_observer_is_passive_and_uses_actual_parent_pair() -> None:
    commands: list[TeleopCommand] = []
    observed = []
    observer = LegacyPositionReferenceObserverRuntime(
        worker_instance_id="position-worker:test",
        executor_instance_id="position-executor:test",
        sink=observed.append,
        queue_capacity=64,
    )
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=commands.append,
        reference_observer=observer,
    )
    executor.start()
    try:
        source_index = executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit_policy",
        )
        _wait_until(lambda: len(commands) >= 4)
        observer.flush()

        pose_batches = [
            batch
            for batch in observed
            if isinstance(batch, LegacyPositionReferenceObservationBatch)
            and batch.frame_observation.modality == "pose"
        ]
        pose_resolutions = [
            item
            for item in observed
            if isinstance(item, LegacyPositionReferenceResolutionObservation)
            and item.modality == "pose"
        ]
        segments = [
            batch.segment_observation
            for batch in pose_batches
            if batch.segment_observation is not None
        ]
        assert len(segments) == 1
        segment = segments[0]
        assert segment is not None
        assert tuple(
            item.source_index for item in segment.endpoint_observations
        ) == (source_index - 1, source_index)
        assert tuple(
            item.source_stage for item in segment.endpoint_observations
        ) == ("current_observation_anchor", "safe_15hz_keyframe")
        assert all(
            not batch.frame_observation.anchor_commit_eligible
            for batch in pose_batches
        )
        assert pose_resolutions
        assert all(
            item.execution_context_enforcement
            == "executor_local_generation_only"
            for item in pose_resolutions
        )
        assert all(not item.backend_accepted for item in pose_resolutions)
        assert {
            item.source_frame_observation_id for item in pose_resolutions
        }.issubset(
            {
                batch.frame_observation.frame_observation_id
                for batch in pose_batches
            }
        )
        assert executor.snapshot()["reference_observer_error_count"] == 0

        expected_points = int(
            round(executor.config.reference_hz / executor.config.sample_hz)
        )
        x_values = [
            float(command.commands["left_arm"].metadata["target_xyzquat"][0])
            for command in commands[:expected_points]
        ]
        assert x_values == sorted(x_values)
        assert commands[1].metadata["position_parent_source_indices"] == [
            source_index - 1,
            source_index,
        ]
        executor.clear()
        observer.flush()
        terminals = [
            item
            for item in observed
            if isinstance(
                item,
                LegacyPositionReferenceSegmentTerminalObservation,
            )
        ]
        assert terminals
        assert all(item.terminal_status == "canceled" for item in terminals)
        assert all(not item.backend_stop_acknowledged for item in terminals)
    finally:
        executor.close()
        observer.close()


def test_position_reference_observer_failure_does_not_fault_or_block_dispatch() -> None:
    class _FailingObserver:
        @staticmethod
        def submit_nowait(_derivation) -> bool:
            raise RuntimeError("observer-only failure")

    commands: list[TeleopCommand] = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=commands.append,
        reference_observer=_FailingObserver(),
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit_policy",
        )
        _wait_until(lambda: len(commands) >= 2)
        snapshot = executor.snapshot()
        assert not snapshot["fault_latched"]
        assert snapshot["dispatch_count"] >= 2
        assert snapshot["reference_observer_error_count"] >= 2
        assert "observer-only failure" in snapshot["last_reference_observer_error"]
    finally:
        executor.close()


def test_position_resolution_observer_failure_is_isolated_from_dispatch() -> None:
    class _ResolutionFailingObserver:
        @staticmethod
        def submit_nowait(_derivation) -> bool:
            return True

        @staticmethod
        def submit_resolution_nowait(_derivation) -> bool:
            raise RuntimeError("resolution-observer-only failure")

    commands: list[TeleopCommand] = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=commands.append,
        reference_observer=_ResolutionFailingObserver(),
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit_policy",
        )
        _wait_until(lambda: len(commands) >= 2)
        snapshot = executor.snapshot()
        assert not snapshot["fault_latched"]
        assert snapshot["dispatch_count"] >= 2
        assert snapshot["reference_observer_error_count"] >= 2
        assert "resolution-observer-only failure" in (
            snapshot["last_reference_observer_error"]
        )
    finally:
        executor.close()


def test_same_chunk_lookahead_interpolates_without_one_action_period_delay() -> None:
    commands: list[TeleopCommand] = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
            stale_after_sec=0.5,
            chunk_lookahead_enabled=True,
        ),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=commands.append,
    )
    current = _filtered_pose("left", _pose(0.16))
    lookahead = _filtered_pose("left", _pose(0.20))
    executor.start()
    try:
        current_index = executor.submit(
            current,
            _observation(),
            "chunk_policy",
            lookahead_filtered=lookahead,
            lookahead_observation=_observation(),
        )
        _wait_until(lambda: len(commands) >= 3)

        x_values = [
            float(command.commands["left_arm"].metadata["target_xyzquat"][0])
            for command in commands[:3]
        ]
        assert x_values[0] == pytest.approx(0.16, abs=2e-3)
        assert x_values == sorted(x_values)
        assert x_values[-1] > x_values[0]
        metadata = commands[1].metadata
        assert metadata["position_causal_delay_periods"] == 0
        assert metadata["position_chunk_lookahead_enabled"] is True
        assert metadata["position_causal_output_lag_sec"] == pytest.approx(0.0)
        assert metadata["position_safety_applied_at"] == "15hz_keyframe_only"
        assert metadata["position_default_profile_contract"] == (
            "safe_15->causal_60->120"
        )
        assert metadata["position_configured_profile_contract"] == (
            "safe_15->causal_60->120"
        )
        assert metadata["position_parent_source_indices"] == [
            current_index,
            current_index + 1,
        ]
        pose_speed = metadata["pose_action_interval_velocity"]
        assert pose_speed["selection"] in {
            "current_to_next",
            "bracketing_current_to_next",
        }
        assert pose_speed["start_source_index"] == current_index
        assert pose_speed["end_source_index"] == current_index + 1
        assert pose_speed["source_rate_hz"] == pytest.approx(15.0)
        assert pose_speed["by_side"]["left"][
            "linear_speed_p90_m_s"
        ] == pytest.approx(0.6, rel=1e-5)
        assert metadata["position_pose_speed_source"] == (
            "adjacent_pose_action_interval"
        )

        # When the staged successor becomes current it reuses the existing
        # endpoint instead of adding a duplicate timestamp.  With no further
        # same-chunk successor, the endpoint is held at the chunk tail.
        staged_index = executor.submit(
            lookahead,
            _observation(),
            "chunk_policy",
        )
        assert staged_index == current_index + 1
        _wait_until(
            lambda: any(
                abs(
                    float(command.commands["left_arm"].metadata["target_xyzquat"][0])
                    - 0.20
                )
                < 2e-3
                for command in commands
            )
        )
        assert executor.snapshot()["causal_delay_periods"] == 0
    finally:
        executor.close()


def test_chunk_lookahead_discrete_command_is_dispatched_once_per_source() -> None:
    hand_commands: list[TeleopCommand] = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
            stale_after_sec=0.5,
            chunk_lookahead_enabled=True,
        ),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
        discrete_dispatch_callback=hand_commands.append,
    )
    current = _filtered_pose("left", _pose(0.16))
    lookahead = _filtered_pose("left", _pose(0.20))
    executor.start()
    try:
        current_index = executor.submit(
            current,
            _observation(),
            "chunk_policy",
            discrete_command=_hand_command("left", 0.2),
            lookahead_filtered=lookahead,
            lookahead_observation=_observation(),
            lookahead_discrete_command=_hand_command("left", 0.4),
        )
        staged_index = executor.submit(
            lookahead,
            _observation(),
            "chunk_policy",
            # The runtime supplies the same canonical hand action again when
            # the staged endpoint becomes current.
            discrete_command=_hand_command("left", 0.4),
        )
        assert staged_index == current_index + 1
        _wait_until(lambda: len(hand_commands) >= 2)
        time.sleep(0.04)

        assert len(hand_commands) == 2
        assert [
            command.metadata["position_discrete_source_index"]
            for command in hand_commands
        ] == [current_index, staged_index]
        assert [
            float(command.commands["left_hand"].value[0])
            for command in hand_commands
        ] == pytest.approx([0.2, 0.4])
        assert all(
            command.metadata["position_arm_hand_phase"]
            == "same_logical_reference_tick_with_chunk_lookahead"
            for command in hand_commands
        )
        assert executor.snapshot()["discrete_submitted_count"] == 2
    finally:
        executor.close()


def test_cross_chunk_first_action_uses_one_period_boundary_transition() -> None:
    commands: list[TeleopCommand] = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
            stale_after_sec=0.5,
            chunk_lookahead_enabled=True,
        ),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=commands.append,
    )
    current = _filtered_pose("left", _pose(0.16))
    tail = _filtered_pose("left", _pose(0.20))
    next_chunk_first = _filtered_pose("left", _pose(0.30))
    executor.start()
    try:
        executor.submit(
            current,
            _observation(),
            "chunk_policy",
            lookahead_filtered=tail,
            lookahead_observation=_observation(),
        )
        executor.submit(tail, _observation(), "chunk_policy")
        _wait_until(
            lambda: any(
                abs(
                    float(command.commands["left_arm"].metadata["target_xyzquat"][0])
                    - 0.20
                )
                < 2e-3
                for command in commands
            )
        )

        before_new_chunk = len(commands)
        executor.submit(next_chunk_first, _observation(), "chunk_policy")
        _wait_until(lambda: len(commands) >= before_new_chunk + 3)
        boundary_values = [
            float(command.commands["left_arm"].metadata["target_xyzquat"][0])
            for command in commands[before_new_chunk : before_new_chunk + 3]
        ]
        assert boundary_values == sorted(boundary_values)
        assert boundary_values[0] == pytest.approx(0.20, abs=2e-3)
        assert boundary_values[-1] < 0.30 - 1e-3
        _wait_until(
            lambda: any(
                abs(
                    float(command.commands["left_arm"].metadata["target_xyzquat"][0])
                    - 0.30
                )
                < 2e-3
                for command in commands
            )
        )
    finally:
        executor.close()


def test_mixed_pose_and_qpos_sides_share_one_causal_reference_clock() -> None:
    pipeline = _pipeline(("left", "right"))
    commands = []
    vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    vector[ACTION_ARM_POSE_SLICES["left"]] = _pose(0.16)
    filtered = FilteredNormalizedAction(
        action=NormalizedAction(
            vector=vector,
            arm_joint_targets={"right": np.full(6, 0.3, dtype=np.float32)},
            metadata={"source": "unit_mixed_safe_15hz"},
        ),
        metadata={"safety_pass_count": 1},
    )
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=pipeline,
        dispatch_callback=lambda command: commands.append(command),
    )
    executor.start()
    try:
        executor.submit(
            filtered,
            _observation(sides=("left", "right")),
            "mixed_policy",
        )
        _wait_until(lambda: len(commands) >= 2)
        command = commands[1]
        assert set(command.commands) == {"left_arm", "right_arm"}
        assert command.commands["left_arm"].command_type == "cartesian_pose"
        assert command.commands["right_arm"].command_type == "joint_vector"
        right_qpos = np.asarray(command.commands["right_arm"].value, dtype=float)
        assert np.all(right_qpos > 0.0)
        assert np.all(right_qpos < 0.3)
        assert command.metadata["position_pose_interpolation"].endswith("slerp")
        assert command.metadata["position_qpos_interpolation"] == "linear"
        assert set(command.metadata["position_interpolation_alpha_by_modality"]) == {
            "pose",
            "qpos",
        }
    finally:
        executor.close()


def test_collision_required_pose_reference_becomes_dual_joint_command() -> None:
    planner = _FakePositionCollisionPlanner()
    commands: list[TeleopCommand] = []
    observed = []
    observer = LegacyPositionReferenceObserverRuntime(
        worker_instance_id="position-worker:collision-test",
        executor_instance_id="position-executor:collision-test",
        sink=observed.append,
        queue_capacity=64,
    )
    pipeline = _pipeline(("left", "right"))

    def fail_if_safety_repeats(*_args, **_kwargs):
        raise AssertionError("15 Hz keyframe safety must not run again at reference rate")

    pipeline.safety_filter.filter_normalized = fail_if_safety_repeats  # type: ignore[method-assign]
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            stale_after_sec=0.5,
            collision_required=True,
            debug_events_enabled=True,
        ),
        canonical_pipeline=pipeline,
        dispatch_callback=lambda command: _accepted_collision_arm_dispatch(
            commands,
            command,
        ),
        collision_planner_factory=lambda: planner,
        reference_observer=observer,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_bimanual_pose(0.16, 0.18),
            _observation(sides=("left", "right")),
            "collision_policy",
        )
        _wait_until(lambda: len(commands) >= 2)
        observer.flush()

        assert len(planner.reset_calls) == 1
        assert set(planner.reset_calls[0]) == {"left", "right"}
        assert len(planner.plan_calls) >= 2
        command = commands[1]
        assert set(command.commands) == {"left_arm", "right_arm"}
        assert all(
            item.command_type == "joint_vector"
            for item in command.commands.values()
        )
        assert all(
            item.metadata["action_pose_source"] == "limited_solver_output_qpos"
            and item.metadata["position_collision_prelimited_qpos"] is True
            for item in command.commands.values()
        )
        assert command.metadata["position_endpoint_collision_checked"] is True
        assert command.metadata["position_endpoint_collision_passed"] is True
        assert command.metadata["position_segment_collision_certified"] is False
        assert command.metadata["no_temporal_smoothing"] is True
        assert "position_collision_debug" in command.metadata
        snapshot = executor.snapshot()
        assert snapshot["collision_reset_count"] == 1
        assert snapshot["collision_tracking_count"] >= 2
        assert snapshot["endpoint_collision_certified"] is True
        assert snapshot["last_collision_reset"]["authoritative"] is True
        assert snapshot["last_reference"]["position_arm_backend_dispatch_verified"] is True
        collision_resolutions = [
            item
            for item in observed
            if isinstance(item, LegacyPositionReferenceResolutionObservation)
        ]
        assert collision_resolutions
        assert all(
            item.planner_evidence_status
            == "legacy_collision_planner_accepted"
            for item in collision_resolutions
        )
        assert all(item.planner_input_hash for item in collision_resolutions)
        assert all(item.planner_output_hash for item in collision_resolutions)
        assert all(
            item.measured_state_evidence_status
            == "legacy_planner_state_not_typed"
            for item in collision_resolutions
        )

        executor.clear()
        executor.submit(
            _filtered_bimanual_pose(0.17, 0.19),
            _observation(sides=("left", "right")),
            "collision_policy_after_clear",
        )
        _wait_until(lambda: len(planner.reset_calls) == 2)
        assert executor.snapshot()["collision_reset_count"] == 2
    finally:
        executor.close()
        observer.close()


def test_collision_hold_fault_latches_before_arm_or_hand_dispatch() -> None:
    planner = _FakePositionCollisionPlanner(hold=True)
    arm_commands: list[TeleopCommand] = []
    hand_commands: list[TeleopCommand] = []
    hold_called = threading.Event()
    observed = []
    observer = LegacyPositionReferenceObserverRuntime(
        worker_instance_id="position-worker:fault-test",
        executor_instance_id="position-executor:fault-test",
        sink=observed.append,
        queue_capacity=64,
    )
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            stale_after_sec=0.5,
            collision_required=True,
            debug_events_enabled=True,
        ),
        canonical_pipeline=_pipeline(("left", "right")),
        dispatch_callback=arm_commands.append,
        discrete_dispatch_callback=hand_commands.append,
        hold_callback=hold_called.set,
        collision_planner_factory=lambda: planner,
        reference_observer=observer,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_bimanual_pose(0.16, 0.18),
            _observation(sides=("left", "right")),
            "collision_policy",
            discrete_command=_hand_command("left", 0.4),
        )
        _wait_until(lambda: executor.snapshot()["fault_latched"])
        observer.flush()
        assert hold_called.wait(timeout=0.5)
        assert arm_commands == []
        terminals = [
            item
            for item in observed
            if isinstance(
                item,
                LegacyPositionReferenceSegmentTerminalObservation,
            )
        ]
        assert terminals
        assert all(item.terminal_status == "faulted" for item in terminals)
        failures = [
            item
            for item in observed
            if isinstance(
                item,
                LegacyPositionReferenceResolutionFailureObservation,
            )
        ]
        assert failures
        assert all(item.failure_stage == "planner" for item in failures)
        assert all(item.non_authoritative for item in failures)
        assert all(not item.anchor_commit_eligible for item in failures)
        assert all(item.admission_receipt_id is None for item in failures)
        assert hand_commands == []
        snapshot = executor.snapshot()
        assert snapshot["collision_hold_count"] == 1
        assert snapshot["last_collision"]["position_collision_hold"] is True
        assert snapshot["last_collision"]["position_endpoint_collision_passed"] is False
    finally:
        executor.close()
        observer.close()


def test_collision_required_faults_if_endpoint_check_was_not_executed() -> None:
    planner = _FakePositionCollisionPlanner(endpoint_checked=False)
    arm_commands: list[TeleopCommand] = []
    hand_commands: list[TeleopCommand] = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            stale_after_sec=0.5,
            collision_required=True,
        ),
        canonical_pipeline=_pipeline(("left", "right")),
        dispatch_callback=arm_commands.append,
        discrete_dispatch_callback=hand_commands.append,
        collision_planner_factory=lambda: planner,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_bimanual_pose(0.16, 0.18),
            _observation(sides=("left", "right")),
            "unchecked_collision_policy",
            discrete_command=_hand_command("left", 0.4),
        )
        _wait_until(lambda: executor.snapshot()["fault_latched"])
        assert arm_commands == []
        assert hand_commands == []
        snapshot = executor.snapshot()
        assert snapshot["last_collision"]["position_endpoint_collision_checked"] is False
        assert "endpoint_checked=False" in snapshot["fault"]
    finally:
        executor.close()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hardware_read_success": False}, "latest hardware read was not successful"),
        (
            {
                "hardware_read_timestamp_monotonic": 1.0,
                "measured_timestamp_monotonic": 1.0,
            },
            "authoritative qpos is stale",
        ),
        ({"state_error": "refresh_failed:unit"}, "state_error"),
    ],
)
def test_real_collision_reset_rejects_cached_or_stale_measured_qpos(
    overrides: dict[str, object],
    message: str,
) -> None:
    planner = _FakePositionCollisionPlanner()
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            collision_required=True,
            allow_non_authoritative_collision_reset_in_dry_run=True,
        ),
        canonical_pipeline=_pipeline(("left", "right")),
        dispatch_callback=lambda command: _accepted_collision_arm_dispatch([], command),
        collision_planner_factory=lambda: planner,
    )
    try:
        observation = _observation(
            sides=("left", "right"),
            arm_state_overrides={"right": overrides},
        )
        with pytest.raises(ValueError, match=message):
            executor.submit(
                _filtered_bimanual_pose(0.16, 0.18),
                observation,
                "strict_collision_reset",
            )
        assert planner.reset_calls == []
    finally:
        executor.close()


def test_collision_reset_dry_run_fallback_is_explicit_and_observable() -> None:
    planner = _FakePositionCollisionPlanner()
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            collision_required=True,
            stale_after_sec=0.5,
            allow_non_authoritative_collision_reset_in_dry_run=True,
        ),
        canonical_pipeline=_pipeline(("left", "right")),
        collision_planner_factory=lambda: planner,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_bimanual_pose(0.16, 0.18),
            _observation(
                sides=("left", "right"),
                arm_state_overrides={
                    "left": {"hardware_read_success": False},
                    "right": {"hardware_read_success": False},
                },
            ),
            "dry_run_collision_reset",
        )
        _wait_until(lambda: executor.snapshot()["collision_reset_count"] == 1)
        reset = executor.snapshot()["last_collision_reset"]
        assert reset["authoritative"] is False
        assert reset["source"] == "dry_run_observation_state_vector"
        assert reset["real_dispatch_fallback_allowed"] is False
    finally:
        executor.close()


def test_authoritative_collision_reset_is_rechecked_when_worker_applies_it() -> None:
    planner = _FakePositionCollisionPlanner()
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            collision_required=True,
            collision_reset_measurement_max_age_sec=0.25,
        ),
        canonical_pipeline=_pipeline(("left", "right")),
        dispatch_callback=lambda command: _accepted_collision_arm_dispatch([], command),
        collision_planner_factory=lambda: planner,
        clock=lambda: 10.30,
    )
    try:
        observation = _observation(
            sides=("left", "right"),
            arm_state_overrides={
                "left": {
                    "hardware_read_timestamp_monotonic": 10.0,
                    "measured_timestamp_monotonic": 10.0,
                },
                "right": {
                    "hardware_read_timestamp_monotonic": 10.0,
                    "measured_timestamp_monotonic": 10.0,
                },
            },
        )
        executor.submit(
            _filtered_bimanual_pose(0.16, 0.18),
            observation,
            "apply_freshness",
            timestamp_monotonic=10.0,
        )
        with pytest.raises(RuntimeError, match="became stale before planner reset"):
            executor._drain_submissions()
        assert planner.reset_calls == []
    finally:
        executor.close()


def test_collision_dispatch_rewrite_faults_before_discrete_hand_queue() -> None:
    planner = _FakePositionCollisionPlanner()
    hand_commands: list[TeleopCommand] = []
    hold_called = threading.Event()

    def rewritten_dispatch(command: TeleopCommand) -> dict[str, object]:
        receipts: dict[str, dict[str, object]] = {}
        for name, item in command.commands.items():
            requested = np.asarray(item.value, dtype=float).copy()
            final = requested.copy()
            singularity_status = "pass"
            if name == "left_arm":
                final[0] -= 0.02
                singularity_status = "hold_entry"
            receipts[name] = {
                "accepted": True,
                "backend_ready": True,
                "command_type": "joint_vector",
                "requested_value": requested,
                "final_value": final,
                "rewritten": name == "left_arm",
                "singularity_gate_status": singularity_status,
                "state_error": None,
                "background_streaming": {},
            }
        return {"accepted": True, "effectors": receipts}

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            collision_required=True,
            stale_after_sec=0.5,
        ),
        canonical_pipeline=_pipeline(("left", "right")),
        dispatch_callback=rewritten_dispatch,
        discrete_dispatch_callback=hand_commands.append,
        hold_callback=hold_called.set,
        collision_planner_factory=lambda: planner,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_bimanual_pose(0.16, 0.18),
            _observation(sides=("left", "right")),
            "collision_rewrite",
            discrete_command=_hand_command("left", 0.4),
        )
        _wait_until(lambda: executor.snapshot()["fault_latched"])
        assert hold_called.wait(timeout=0.5)
        assert hand_commands == []
        snapshot = executor.snapshot()
        assert snapshot["arm_dispatch_failed_count"] == 1
        assert "was not preserved" in snapshot["fault"]
    finally:
        executor.close()


def test_collision_required_rejects_single_arm_and_direct_qpos_streams() -> None:
    with pytest.raises(ValueError, match=r"left\+right"):
        PositionDeployExecutor(
            config=PositionDeployExecutorConfig(collision_required=True),
            canonical_pipeline=_pipeline(("left",)),
            collision_planner_factory=_FakePositionCollisionPlanner,
        )

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(collision_required=True),
        canonical_pipeline=_pipeline(("left", "right")),
        collision_planner_factory=_FakePositionCollisionPlanner,
    )
    vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    filtered = FilteredNormalizedAction(
        action=NormalizedAction(
            vector=vector,
            arm_joint_targets={
                "left": np.zeros(6, dtype=np.float32),
                "right": np.zeros(6, dtype=np.float32),
            },
        )
    )
    try:
        with pytest.raises(ValueError, match="target-pose"):
            executor.submit(
                filtered,
                _observation(sides=("left", "right")),
                "qpos_policy",
            )
    finally:
        executor.close()


def test_clear_pauses_output_empties_stream_and_calls_hold() -> None:
    commands = []
    holds = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda command: commands.append(command),
        hold_callback=lambda: holds.append(time.monotonic()),
    )
    executor.start()
    try:
        executor.submit(_filtered_pose("left", _pose(0.16)), _observation(), "unit")
        _wait_until(lambda: len(commands) >= 2)
        executor.clear()
        count_after_clear = len(commands)
        time.sleep(0.07)
        assert len(commands) == count_after_clear
        assert len(holds) == 1
        snapshot = executor.snapshot()
        assert snapshot["active"] is False
        assert snapshot["queue_depth"] == 0

        executor.submit(_filtered_pose("left", _pose(0.13)), _observation(), "unit")
        _wait_until(lambda: len(commands) > count_after_clear)
    finally:
        executor.close()


def test_dispatch_fault_is_latched_and_requests_hold() -> None:
    hold_called = threading.Event()

    def broken_dispatch(_command) -> None:
        raise RuntimeError("backend rejected reference")

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=broken_dispatch,
        hold_callback=hold_called.set,
    )
    executor.start()
    try:
        executor.submit(_filtered_pose("left", _pose(0.16)), _observation(), "unit")
        _wait_until(lambda: executor.snapshot()["fault_latched"])
        assert hold_called.wait(timeout=0.5)
        with pytest.raises(PositionDeployExecutorFault, match="backend rejected reference"):
            executor.raise_if_fault()
        with pytest.raises(PositionDeployExecutorFault):
            executor.submit(_filtered_pose("left", _pose(0.17)), _observation(), "unit")
        assert executor.snapshot()["fault_stage"] == "reference_worker"
    finally:
        executor.close()


def test_15hz_safety_fault_latches_clears_and_holds_before_any_reference_dispatch() -> None:
    holds: list[str] = []
    dispatched: list[TeleopCommand] = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda command: dispatched.append(command),
        hold_callback=lambda: holds.append("hold"),
    )
    try:
        with pytest.raises(PositionDeployExecutorFault, match="safety faults"):
            executor.submit(
                FilteredNormalizedAction(
                    action=NormalizedAction(vector=np.full(ACTION_DIM, np.nan, dtype=np.float32)),
                    faults=["left_arm_qpos_constraint_infeasible"],
                ),
                _observation(),
                "unit",
            )
        snapshot = executor.snapshot()
        assert snapshot["fault_latched"] is True
        assert snapshot["fault_stage"] == "15hz_keyframe_safety"
        assert holds == ["hold"]
        assert dispatched == []
        assert snapshot["queue_depth"] == 0
    finally:
        executor.close()


def test_manual_safety_rearm_requires_fresh_measured_feedback() -> None:
    holds: list[str] = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            stale_after_sec=0.5,
            safety_rearm_measurement_max_age_sec=0.1,
        ),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
        hold_callback=lambda: holds.append("hold"),
    )
    try:
        executor.latch_safety_fault(["left_arm_pose_constraint_infeasible"])
        stale = _observation(
            arm_state_overrides={
                "left": {"hardware_measurement_timestamp": time.time() - 1.0}
            }
        )
        with pytest.raises(PositionDeployExecutorFault, match="measurement age"):
            executor.manual_rearm(stale)
        assert executor.snapshot()["fault_latched"] is True

        fresh = _observation(
            arm_state_overrides={
                "left": {"hardware_measurement_timestamp": time.time()}
            }
        )
        result = executor.manual_rearm(fresh)
        assert result["rearmed"] is True
        assert executor.snapshot()["fault_latched"] is False
        assert holds == ["hold"]
    finally:
        executor.close()


def test_bounded_fifo_rejects_overflow_and_missing_dispatch_is_dry_run() -> None:
    pipeline = _pipeline(("left",))
    queued = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5, queue_size=1),
        canonical_pipeline=pipeline,
    )
    now = time.monotonic()
    queued.submit(
        _filtered_pose("left", _pose(0.16)),
        _observation(),
        "unit",
        timestamp_monotonic=now,
    )
    with pytest.raises(PositionDeployQueueFullError):
        queued.submit(
            _filtered_pose("left", _pose(0.17)),
            _observation(),
            "unit",
            timestamp_monotonic=now + 1.0 / 15.0,
        )
    assert queued.snapshot()["queue_full_count"] == 1
    queued.close()

    dry_run = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=pipeline,
        dispatch_callback=None,
    )
    dry_run.start()
    try:
        dry_run.submit(_filtered_pose("left", _pose(0.16)), _observation(), "unit")
        _wait_until(lambda: dry_run.snapshot()["dry_run_count"] >= 1)
        snapshot = dry_run.snapshot()
        assert snapshot["dry_run"] is True
        assert snapshot["dispatch_count"] == 0
    finally:
        dry_run.close()


def test_discrete_hand_keyframe_shares_delayed_reference_tick_without_interpolation() -> None:
    events: list[tuple[str, TeleopCommand]] = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda command: events.append(("arm", command)),
        discrete_dispatch_callback=lambda command: events.append(("hand", command)),
        wall_clock=lambda: 1_700_000_001.0,
    )
    executor.start()
    try:
        source_index = executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit",
            discrete_command=_hand_command("left", 0.4),
        )
        _wait_until(lambda: executor.snapshot()["discrete_dispatch_count"] >= 1)

        hand_positions = [index for index, item in enumerate(events) if item[0] == "hand"]
        assert len(hand_positions) == 1
        hand_position = hand_positions[0]
        assert hand_position > 0
        assert events[hand_position - 1][0] == "arm"
        hand = events[hand_position][1]
        assert hand.timestamp == 1_700_000_001.0
        assert hand.metadata["position_discrete_source_index"] == source_index
        assert hand.metadata["position_discrete_interpolation"] == "none"
        assert hand.metadata["position_discrete_catch_up"] is False
        assert hand.metadata["position_arm_hand_phase"].startswith(
            "same_logical_reference_tick"
        )
        assert hand.metadata["position_discrete_logical_causal_lag_sec"] == pytest.approx(
            1.0 / 15.0,
            abs=0.03,
        )
        assert hand.metadata["position_discrete_actual_causal_lag_sec"] >= (
            hand.metadata["position_discrete_logical_causal_lag_sec"]
        )
        time.sleep(0.04)
        assert sum(kind == "hand" for kind, _ in events) == 1
        snapshot = executor.snapshot()
        assert snapshot["discrete_submitted_count"] == 1
        assert snapshot["discrete_reference_count"] == 1
        assert snapshot["discrete_dispatch_count"] == 1
        assert snapshot["discrete_skipped_count"] == 0
    finally:
        executor.close()


def test_clear_drops_queued_discrete_hand_before_worker_start() -> None:
    hands: list[TeleopCommand] = []
    holds: list[bool] = []
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
        discrete_dispatch_callback=hands.append,
        hold_callback=lambda: holds.append(True),
    )
    executor.submit(
        _filtered_pose("left", _pose(0.16)),
        _observation(),
        "unit",
        discrete_command=_hand_command("left", 0.4),
    )
    executor.clear()
    executor.start()
    try:
        time.sleep(0.1)
        assert hands == []
        assert holds == [True]
        assert executor.snapshot()["queue_depth"] == 0
    finally:
        executor.close()


def test_discrete_hand_dispatch_fault_latches_and_holds_arm_stream() -> None:
    hold_called = threading.Event()

    def broken_hand_dispatch(_command: TeleopCommand) -> None:
        raise RuntimeError("hand backend rejected delayed keyframe")

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
        discrete_dispatch_callback=broken_hand_dispatch,
        hold_callback=hold_called.set,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit",
            discrete_command=_hand_command("left", 0.4),
        )
        _wait_until(lambda: executor.snapshot()["fault_latched"])
        assert hold_called.wait(timeout=0.5)
        with pytest.raises(PositionDeployExecutorFault, match="hand backend rejected"):
            executor.raise_if_fault()
    finally:
        executor.close()


def test_false_discrete_result_is_a_real_fault_not_a_success_counter() -> None:
    hold_called = threading.Event()
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
        discrete_dispatch_callback=lambda _command: False,
        hold_callback=hold_called.set,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit",
            discrete_command=_hand_command("left", 0.4),
        )
        _wait_until(lambda: executor.snapshot()["fault_latched"])
        assert hold_called.wait(timeout=0.5)
        snapshot = executor.snapshot()
        assert snapshot["discrete_dispatch_attempt_count"] == 1
        assert snapshot["discrete_dispatch_count"] == 0
        assert snapshot["discrete_dispatch_failed_count"] == 1
        assert snapshot["last_discrete"]["position_discrete_dispatch_status"] == "failed"
    finally:
        executor.close()


def test_nonfatal_backend_rate_skip_is_observable_without_faulting_arm() -> None:
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
        discrete_dispatch_callback=lambda _command: {
            "accepted": False,
            "skipped": True,
            "status": "rate_limited",
        },
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit",
            discrete_command=_hand_command("left", 0.4),
        )
        _wait_until(lambda: executor.snapshot()["discrete_backend_skipped_count"] == 1)
        snapshot = executor.snapshot()
        assert snapshot["fault_latched"] is False
        assert snapshot["discrete_dispatch_attempt_count"] == 1
        assert snapshot["discrete_dispatch_count"] == 0
        assert snapshot["last_discrete"]["position_discrete_dispatch_status"] == (
            "backend_skipped"
        )
    finally:
        executor.close()


def test_blocking_hand_io_does_not_stall_arm_clock_and_clear_orders_hold_after_it() -> None:
    hand_started = threading.Event()
    release_hand = threading.Event()
    clear_done = threading.Event()
    arm_commands: list[float] = []
    order: list[str] = []

    def arm_dispatch(_command: TeleopCommand) -> None:
        arm_commands.append(time.monotonic())

    def hand_dispatch(_command: TeleopCommand) -> None:
        hand_started.set()
        assert release_hand.wait(timeout=0.5)
        order.append("hand_done")

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=arm_dispatch,
        discrete_dispatch_callback=hand_dispatch,
        hold_callback=lambda: order.append("hold"),
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit",
            discrete_command=_hand_command("left", 0.4),
        )
        assert hand_started.wait(timeout=0.5)
        arm_count_when_hand_started = len(arm_commands)
        time.sleep(0.055)
        assert len(arm_commands) >= arm_count_when_hand_started + 2

        clear_thread = threading.Thread(
            target=lambda: (executor.clear(), clear_done.set()),
            daemon=True,
        )
        clear_thread.start()
        time.sleep(0.02)
        assert not clear_done.is_set()
        release_hand.set()
        assert clear_done.wait(timeout=0.5)
        clear_thread.join(timeout=0.5)
        assert order == ["hand_done", "hold"]
        arm_count_after_hold = len(arm_commands)
        time.sleep(0.05)
        assert len(arm_commands) == arm_count_after_hold
    finally:
        release_hand.set()
        executor.close()


def test_late_discrete_keyframes_coalesce_to_latest_without_callback_burst() -> None:
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
    )
    now = time.monotonic()
    source_indices = []
    for offset, value in enumerate((0.2, 0.3, 0.4)):
        source_indices.append(
            executor.submit(
                _filtered_pose("left", _pose(0.12 + 0.01 * offset)),
                _observation(),
                "unit",
                timestamp_monotonic=now + offset / 15.0,
                discrete_command=_hand_command("left", value),
            )
        )
    executor._drain_submissions()
    with executor._lock:
        source_index, _, command, skipped = executor._take_due_discrete_command_locked(
            now + 2.0 / 15.0
        )
    assert source_index == source_indices[-1]
    assert command is not None
    np.testing.assert_allclose(command.commands["left_hand"].value, np.full(6, 0.4))
    assert skipped == 2
    assert executor.snapshot()["discrete_skipped_count"] == 2
    executor.close()


def test_mixed_position_dry_run_records_arm_and_hand_without_dispatch() -> None:
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            stale_after_sec=0.5,
            debug_events_enabled=True,
        ),
        canonical_pipeline=_pipeline(("left",)),
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit",
            discrete_command=_hand_command("left", 0.4),
        )
        _wait_until(lambda: executor.snapshot()["discrete_dry_run_count"] == 1)
        snapshot = executor.snapshot()
        assert snapshot["dry_run"] is True
        assert snapshot["dry_run_count"] >= 1
        assert snapshot["discrete_dispatch_attempt_count"] == 0
        assert snapshot["last_discrete"]["position_discrete_dispatch_status"] == "dry_run"
        events = executor.drain_debug_events()
        assert any(event["event"] == "keyframe_submitted" for event in events)
        assert any(
            event["event"] == "discrete_dispatch"
            and event["position_discrete_dispatch_status"] == "dry_run"
            for event in events
        )
    finally:
        executor.close()


def test_checked_real_hand_dispatch_converts_inspire_swallowed_io_error_to_fault() -> None:
    class BrokenHand:
        def set_joint_pos(self, _value) -> None:
            raise OSError("unit transport failure")

        def close(self) -> None:
            return None

    backend = InspireTacHandBackend(
        InspireTacHandBackendConfig(
            hand_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            reconnect_on_error=False,
        )
    )
    backend.hand = BrokenHand()
    command = _hand_command("left", 0.4)

    with pytest.raises(RuntimeError, match="discrete dispatch rejected"):
        _dispatch_position_discrete_command_checked(
            command,
            hand_backends={"left": backend},
        )

    receipt = command.metadata["position_discrete_backend_receipts"]["left_hand"]
    assert receipt["accepted"] is False
    assert receipt["fatal"] is True
    assert receipt["status"] == "rejected"


def test_structured_dispatch_receipt_exposes_backend_joint_rewrite() -> None:
    class RewritingArmBackend:
        def is_ready(self) -> bool:
            return True

        def get_state(self) -> dict[str, object]:
            return {
                "hardware_ready": True,
                "singularity_metadata": {"singularity_gate_status": "hold_entry"},
                "background_streaming": {"enabled": False},
            }

        def send_command(self, command: EffectorCommand) -> CommandDispatchReceipt:
            command.value = np.asarray(command.value, dtype=float) - 0.01
            return CommandDispatchReceipt(
                effector_name="left_arm",
                accepted=True,
                operation="unit_queue",
                status="queued",
            )

    command = TeleopCommand()
    command.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.full(6, 0.2, dtype=float),
        )
    )
    result = dispatch_teleop_command(
        command,
        arm_backends={"left": RewritingArmBackend()},
        hand_backends={},
        collect_backend_evidence=True,
    )
    receipt = result["effectors"]["left_arm"]
    assert result["accepted"] is True
    assert receipt["rewritten"] is True
    assert receipt["singularity_gate_status"] == "hold_entry"
    np.testing.assert_allclose(receipt["requested_value"], np.full(6, 0.2))
    np.testing.assert_allclose(receipt["final_value"], np.full(6, 0.19))


def test_structured_dispatch_preserves_partial_dual_arm_receipts_on_exception() -> None:
    class UnitArmBackend:
        def __init__(self, side: str, *, reject: bool) -> None:
            self.side = side
            self.reject = reject

        def is_ready(self) -> bool:
            return True

        def get_state(self) -> dict[str, object]:
            return {
                "hardware_ready": True,
                "background_streaming": {"enabled": False},
            }

        def send_command(self, _command: EffectorCommand) -> CommandDispatchReceipt:
            if self.reject:
                raise RuntimeError(f"{self.side} unit reject")
            return CommandDispatchReceipt(
                effector_name=f"{self.side}_arm",
                accepted=True,
                operation="unit_queue",
                status="queued",
            )

    command = TeleopCommand()
    for side in ("left", "right"):
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=np.full(6, 0.2, dtype=float),
            )
        )
    with pytest.raises(RuntimeError, match="right unit reject"):
        dispatch_teleop_command(
            command,
            arm_backends={
                "left": UnitArmBackend("left", reject=False),
                "right": UnitArmBackend("right", reject=True),
            },
            hand_backends={},
            collect_backend_evidence=True,
        )
    receipts = command.metadata["deploy_backend_dispatch_receipts"]
    assert receipts["left_arm"]["accepted"] is True
    assert receipts["right_arm"]["accepted"] is False
    assert receipts["right_arm"]["status"] == "exception"


def test_pre_hand_arm_health_gate_rejects_background_stream_fault() -> None:
    class FaultedArmBackend:
        def is_ready(self) -> bool:
            return True

        def get_state(self) -> dict[str, object]:
            return {
                "hardware_ready": True,
                "background_streaming": {
                    "enabled": True,
                    "healthy": False,
                    "thread_alive": False,
                    "target_active": False,
                    "target_stale": False,
                    "fault": "unit sdk failure",
                    "shutdown_requested": False,
                    "shutdown_complete": False,
                },
            }

    with pytest.raises(RuntimeError, match="background_fault"):
        require_arm_backends_ready_for_position_dispatch(
            {"left": FaultedArmBackend(), "right": FaultedArmBackend()},
            require_background_target=True,
            stage="pre_discrete_hand_dispatch",
        )
    with pytest.raises(RuntimeError, match="right=backend_missing"):
        require_arm_backends_ready_for_position_dispatch(
            {"left": FaultedArmBackend()},
            require_background_target=True,
            stage="pre_discrete_hand_dispatch",
            required_sides=("left", "right"),
        )


def test_close_rejects_concurrent_submit_and_waits_for_inflight_hand_before_hold() -> None:
    hand_started = threading.Event()
    release_hand = threading.Event()
    close_done = threading.Event()
    order: list[str] = []

    def hand_dispatch(_command: TeleopCommand) -> None:
        hand_started.set()
        assert release_hand.wait(timeout=0.5)
        order.append("hand_done")

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
        discrete_dispatch_callback=hand_dispatch,
        hold_callback=lambda: order.append("hold"),
    )
    executor.start()
    executor.submit(
        _filtered_pose("left", _pose(0.16)),
        _observation(),
        "unit",
        discrete_command=_hand_command("left", 0.4),
    )
    assert hand_started.wait(timeout=0.5)

    close_thread = threading.Thread(
        target=lambda: (executor.close(), close_done.set()),
        daemon=True,
    )
    close_thread.start()
    _wait_until(lambda: executor.snapshot()["closing"])
    with pytest.raises(RuntimeError, match="closed"):
        executor.submit(
            _filtered_pose("left", _pose(0.17)),
            _observation(),
            "unit",
            discrete_command=_hand_command("left", 0.5),
        )
    assert not close_done.is_set()
    release_hand.set()
    assert close_done.wait(timeout=0.5)
    close_thread.join(timeout=0.5)
    assert order == ["hand_done", "hold"]
    snapshot = executor.snapshot()
    assert snapshot["closed"] is True
    assert snapshot["reference_worker_alive"] is False
    assert snapshot["discrete_worker_alive"] is False


def test_deploy_debug_marks_position_submission_as_async_queue_not_dispatch(tmp_path) -> None:
    logger = DeploymentDebugLogger(
        enabled=True,
        root_dir=tmp_path,
        active_sides=("left",),
    )
    raw = _filtered_pose("left", _pose(0.16)).action
    logger.log_action_step(
        timestamp=123.0,
        step_index=7,
        policy_name="unit",
        raw_action=raw,
        canonical_action=SimpleNamespace(
            action=raw,
            command=_hand_command("left", 0.4),
            metadata={},
        ),
        pre_arm_states={},
        post_arm_states=None,
        arm_backends={},
        send_actions=True,
        queue_len_before_pop=1,
        queue_len_after_pop=0,
        faults=[],
        dispatch_error=None,
        execution=_PreparedActionExecution(
            accepted=True,
            mode="position_async_queue",
            dispatch_attempted=False,
            dispatch_completed=False,
            asynchronous=True,
            position_source_index=9,
        ),
    )
    path = logger.path
    logger.close()
    assert path is not None
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["dispatch_mode"] == "position_async_queue"
    assert record["dispatch_attempted"] is False
    assert record["dispatch_completed"] is False
    assert record["dispatch_asynchronous"] is True
    assert record["position_source_index"] == 9


def test_blocked_hand_worker_mailbox_keeps_only_latest_pending_keyframe() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    callback_values: list[float] = []
    callback_started_at: list[float] = []

    def hand_dispatch(command: TeleopCommand) -> None:
        value = float(np.asarray(command.commands["left_hand"].value)[0])
        callback_values.append(value)
        callback_started_at.append(time.monotonic())
        if len(callback_values) == 1:
            first_started.set()
            assert release_first.wait(timeout=0.8)

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.6),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
        discrete_dispatch_callback=hand_dispatch,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.13)),
            _observation(),
            "unit",
            discrete_command=_hand_command("left", 0.1),
        )
        assert first_started.wait(timeout=0.5)
        for index, value in enumerate((0.2, 0.3, 0.4), start=1):
            time.sleep(0.07)
            executor.submit(
                _filtered_pose("left", _pose(0.13 + 0.01 * index)),
                _observation(),
                "unit",
                discrete_command=_hand_command("left", value),
            )
        _wait_until(
            lambda: executor.snapshot()["discrete_worker_coalesced_count"] >= 2,
            timeout=0.6,
        )
        release_first.set()
        _wait_until(lambda: executor.snapshot()["discrete_dispatch_count"] == 2)
        time.sleep(0.05)
        assert callback_values == pytest.approx([0.1, 0.4])
        assert len(callback_started_at) == 2
        assert callback_started_at[1] - callback_started_at[0] > 0.20
        snapshot = executor.snapshot()
        assert snapshot["discrete_worker_coalesced_count"] == 2
        assert snapshot["discrete_skipped_count"] >= 2
        assert snapshot["last_discrete"]["position_discrete_coalesced"] is True
        assert snapshot["last_discrete"]["position_discrete_skipped_before_this"] >= 2
    finally:
        release_first.set()
        executor.close()


def test_hand_fault_pending_gate_prevents_new_arm_ticks_before_hold_latch() -> None:
    failure_known = threading.Event()
    tail_arm_dispatches: list[float] = []
    hold_called = threading.Event()

    def arm_dispatch(_command: TeleopCommand) -> None:
        if failure_known.is_set():
            tail_arm_dispatches.append(time.monotonic())

    def reject_hand(_command: TeleopCommand) -> bool:
        failure_known.set()
        return False

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=arm_dispatch,
        discrete_dispatch_callback=reject_hand,
        hold_callback=hold_called.set,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit",
            discrete_command=_hand_command("left", 0.4),
        )
        _wait_until(lambda: executor.snapshot()["fault_latched"])
        assert hold_called.wait(timeout=0.5)
        time.sleep(0.05)
        assert tail_arm_dispatches == []
        snapshot = executor.snapshot()
        assert snapshot["fault_pending"] is False
        assert snapshot["fault_stage"] == "discrete_dispatch_worker"
    finally:
        executor.close()


def test_fault_hold_failure_is_recorded_and_retried_during_close() -> None:
    hold_attempts: list[int] = []

    def flaky_hold() -> None:
        hold_attempts.append(len(hold_attempts) + 1)
        if len(hold_attempts) == 1:
            raise RuntimeError("first hold attempt failed")

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
        discrete_dispatch_callback=lambda _command: False,
        hold_callback=flaky_hold,
    )
    executor.start()
    executor.submit(
        _filtered_pose("left", _pose(0.16)),
        _observation(),
        "unit",
        discrete_command=_hand_command("left", 0.4),
    )
    _wait_until(lambda: executor.snapshot()["fault_latched"])
    _wait_until(lambda: executor.snapshot()["hold_failure_count"] == 1)
    before_close = executor.snapshot()
    assert before_close["hold_retry_required"] is True
    assert "first hold attempt failed" in before_close["last_hold_error"]

    executor.close()

    after_close = executor.snapshot()
    assert hold_attempts == [1, 2]
    assert after_close["hold_retry_required"] is False
    assert after_close["hold_count"] == 1


def test_discrete_callback_reentrant_clear_faults_instead_of_deadlocking() -> None:
    holder: dict[str, PositionDeployExecutor] = {}

    def reentrant_clear(_command: TeleopCommand) -> None:
        holder["executor"].clear()

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
        discrete_dispatch_callback=reentrant_clear,
        hold_callback=lambda: None,
    )
    holder["executor"] = executor
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit",
            discrete_command=_hand_command("left", 0.4),
        )
        _wait_until(lambda: executor.snapshot()["fault_latched"])
        with pytest.raises(PositionDeployExecutorFault, match="cannot be called"):
            executor.raise_if_fault()
    finally:
        executor.close()


def test_start_close_race_never_joins_an_unstarted_worker() -> None:
    for _ in range(20):
        executor = PositionDeployExecutor(
            config=PositionDeployExecutorConfig(stale_after_sec=0.5),
            canonical_pipeline=_pipeline(("left",)),
        )
        start_errors: list[BaseException] = []
        close_errors: list[BaseException] = []

        def start_executor() -> None:
            try:
                executor.start()
            except BaseException as exc:  # close is allowed to win the race.
                start_errors.append(exc)

        def close_executor() -> None:
            try:
                executor.close()
            except BaseException as exc:
                close_errors.append(exc)

        start_thread = threading.Thread(target=start_executor)
        close_thread = threading.Thread(target=close_executor)
        start_thread.start()
        close_thread.start()
        start_thread.join(timeout=1.0)
        close_thread.join(timeout=1.0)
        assert not start_thread.is_alive()
        assert not close_thread.is_alive()
        assert close_errors == []
        assert all(isinstance(exc, RuntimeError) for exc in start_errors)
        snapshot = executor.snapshot()
        assert snapshot["closed"] is True
        assert snapshot["reference_worker_alive"] is False


def test_hold_callback_reentrant_clear_is_rejected_without_deadlock() -> None:
    holder: dict[str, PositionDeployExecutor] = {}
    hold_attempts = 0

    def reentrant_hold() -> None:
        nonlocal hold_attempts
        hold_attempts += 1
        if hold_attempts == 1:
            holder["executor"].clear()

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
        hold_callback=reentrant_hold,
    )
    holder["executor"] = executor
    executor.submit(_filtered_pose("left", _pose(0.16)), _observation(), "unit")
    with pytest.raises(RuntimeError, match="worker/hold callback"):
        executor.clear()
    assert executor.snapshot()["hold_retry_required"] is True
    executor.close()
    assert hold_attempts == 2


def test_inflight_arm_accept_is_audited_when_hand_fault_closes_gate() -> None:
    hand_callback_entered = threading.Event()
    allow_hand_failure = threading.Event()
    overlapping_arm_entered = threading.Event()
    release_overlapping_arm = threading.Event()
    hold_called = threading.Event()

    def arm_dispatch(_command: TeleopCommand) -> None:
        if hand_callback_entered.is_set():
            overlapping_arm_entered.set()
            assert release_overlapping_arm.wait(timeout=0.8)

    def reject_hand_after_arm_enters(_command: TeleopCommand) -> bool:
        hand_callback_entered.set()
        assert allow_hand_failure.wait(timeout=0.8)
        return False

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.8),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=arm_dispatch,
        discrete_dispatch_callback=reject_hand_after_arm_enters,
        hold_callback=hold_called.set,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit",
            discrete_command=_hand_command("left", 0.4),
        )
        assert hand_callback_entered.wait(timeout=0.5)
        assert overlapping_arm_entered.wait(timeout=0.5)

        allow_hand_failure.set()
        _wait_until(lambda: executor.snapshot()["fault_pending"], timeout=0.5)
        release_overlapping_arm.set()
        _wait_until(lambda: executor.snapshot()["fault_latched"], timeout=0.5)
        assert hold_called.wait(timeout=0.5)

        snapshot = executor.snapshot()
        assert snapshot["fault_stage"] == "discrete_dispatch_worker"
        assert snapshot["dispatch_count"] >= 2
        assert snapshot["last_reference"]["position_arm_dispatch_status"] == (
            "accepted_before_async_abort"
        )
        assert snapshot["last_reference"]["position_arm_dispatch_partial"] is True
        assert snapshot["last_reference"]["position_arm_dispatch_abort_reason"] == (
            "fault_pending"
        )
    finally:
        allow_hand_failure.set()
        release_overlapping_arm.set()
        executor.close()


def test_discrete_worker_outer_exception_is_fault_latched_as_internal() -> None:
    hold_called = threading.Event()

    def discrete_thread_failing_clock() -> float:
        if threading.current_thread().name == "position-deploy-discrete-dispatch":
            raise RuntimeError("unit discrete worker clock failure")
        return time.monotonic()

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
        discrete_dispatch_callback=lambda _command: True,
        hold_callback=hold_called.set,
        clock=discrete_thread_failing_clock,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "unit",
            discrete_command=_hand_command("left", 0.4),
        )
        _wait_until(lambda: executor.snapshot()["fault_latched"], timeout=0.6)
        assert hold_called.wait(timeout=0.5)

        snapshot = executor.snapshot()
        assert snapshot["fault_stage"] == "discrete_dispatch_worker_internal"
        assert snapshot["fault_type"] == "RuntimeError"
        assert snapshot["fault"] == "unit discrete worker clock failure"
        assert snapshot["fault_metadata"]["worker"] == "discrete"
        assert snapshot["fault_metadata"]["internal_exception"] is True
        assert snapshot["discrete_dispatch_attempt_count"] == 1
        assert snapshot["discrete_dispatch_count"] == 0
        assert snapshot["discrete_dispatch_failed_count"] == 0
        _wait_until(lambda: not executor.snapshot()["discrete_worker_alive"])
        with pytest.raises(
            PositionDeployExecutorFault,
            match="discrete_dispatch_worker_internal",
        ):
            executor.raise_if_fault()
    finally:
        executor.close()


def test_failed_close_is_rethrown_to_concurrent_and_later_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hold_started = threading.Event()
    release_hold = threading.Event()
    waiter_entered_close_wait = threading.Event()
    errors: dict[str, BaseException] = {}

    def failing_hold() -> None:
        hold_started.set()
        assert release_hold.wait(timeout=0.5)
        raise RuntimeError("unit close hold failure")

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            stale_after_sec=0.5,
            shutdown_timeout_sec=0.5,
        ),
        canonical_pipeline=_pipeline(("left",)),
        hold_callback=failing_hold,
    )
    executor.submit(_filtered_pose("left", _pose(0.16)), _observation(), "unit")
    original_close_wait = executor._closed_event.wait

    def observed_close_wait(timeout: float | None = None) -> bool:
        waiter_entered_close_wait.set()
        return original_close_wait(timeout=timeout)

    monkeypatch.setattr(executor._closed_event, "wait", observed_close_wait)

    def close_and_capture(label: str) -> None:
        try:
            executor.close()
        except BaseException as exc:
            errors[label] = exc

    primary = threading.Thread(target=close_and_capture, args=("primary",))
    primary.start()
    assert hold_started.wait(timeout=0.5)

    waiter = threading.Thread(target=close_and_capture, args=("waiter",))
    waiter.start()
    assert waiter_entered_close_wait.wait(timeout=0.5)
    release_hold.set()
    primary.join(timeout=0.5)
    waiter.join(timeout=0.5)

    assert not primary.is_alive()
    assert not waiter.is_alive()
    assert set(errors) == {"primary", "waiter"}
    close_error = errors["primary"]
    assert errors["waiter"] is close_error
    assert isinstance(close_error, RuntimeError)
    assert "unit close hold failure" in str(close_error)
    assert isinstance(close_error.__cause__, RuntimeError)

    with pytest.raises(RuntimeError) as later:
        executor.close()
    assert later.value is close_error
