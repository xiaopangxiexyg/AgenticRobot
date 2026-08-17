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
from robot_motion.position_control import (
    POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED,
    POSITION_PLAN_GAP_NONE,
    POSITION_PLAN_RELATION_REPLACEMENT,
    POSITION_PLAN_RELATION_STREAM_START,
    POSITION_PLAN_SPLICE_HOLD_RESTART,
    PositionStreamStartAcquisitionConfig,
    PositionSourcePlanCut,
    resolve_position_plan_splice,
)
from robot_motion.runtime import CommandDispatchReceipt, EffectorCommand, TeleopCommand
from teleop_runtime.backends.hands.inspire_tac import (
    InspireTacHandBackend,
    InspireTacHandBackendConfig,
)
from teleop_runtime.deploy.run_policy import (
    DEPLOY_DEBUG_PAYLOAD_FULL,
    DeploymentDebugLogger,
    _PreparedActionExecution,
    _debug_arm_state_summary,
    _defer_position_action_after_synchronous_prediction,
    _dispatch_position_discrete_command_checked,
    _hold_active_position_stream_for_synchronous_chunk_gap,
    _prepare_position_plan_cut_for_action_step,
    _position_source_tail_pending_before_synchronous_chunk_gap,
)
from teleop_runtime.deploy.position_plan_cut import (
    DeploymentPositionPlanCutDetector,
)
from teleop_runtime.deploy.position_stream_handoff import (
    defer_position_action_after_synchronous_prediction,
    hold_active_position_stream_for_synchronous_chunk_gap,
    position_source_tail_pending_before_synchronous_chunk_gap,
)
from teleop_runtime.deploy.dispatch import (
    POSITION_REFERENCE_DISPATCH_AUDIT_FULL,
    POSITION_REFERENCE_DISPATCH_AUDIT_POST,
    dispatch_position_reference_command,
    dispatch_teleop_command,
    require_arm_backends_ready_for_position_dispatch,
)
from teleop_runtime.position_control.deploy_executor import (
    PositionDeployExecutor,
    PositionDeployExecutorConfig,
    PositionDeployExecutorFault,
    PositionDeployQueueFullError,
)
from teleop_runtime.position_control import deploy_executor as deploy_executor_module
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


def test_run_policy_preserves_position_stream_handoff_exports() -> None:
    assert (
        _defer_position_action_after_synchronous_prediction
        is defer_position_action_after_synchronous_prediction
    )
    assert (
        _hold_active_position_stream_for_synchronous_chunk_gap
        is hold_active_position_stream_for_synchronous_chunk_gap
    )
    assert (
        _position_source_tail_pending_before_synchronous_chunk_gap
        is position_source_tail_pending_before_synchronous_chunk_gap
    )


def test_synchronous_position_prediction_requires_next_observation_cut() -> None:
    assert _defer_position_action_after_synchronous_prediction(
        prediction_accepted=True,
        position_executor_enabled=True,
    )
    assert not _defer_position_action_after_synchronous_prediction(
        prediction_accepted=False,
        position_executor_enabled=True,
    )
    assert not _defer_position_action_after_synchronous_prediction(
        prediction_accepted=True,
        position_executor_enabled=False,
    )


def test_synchronous_position_chunk_gap_holds_only_an_active_stream() -> None:
    class FakePositionExecutor:
        def __init__(self, *, active: bool, tail_drained: bool = True) -> None:
            self.active = bool(active)
            self.tail_drained = bool(tail_drained)
            self.clear_count = 0

        def snapshot(self) -> dict[str, object]:
            return {"active": self.active}

        def source_tail_drain_status(self) -> dict[str, object]:
            return {
                "required": not self.tail_drained,
                "drained": self.tail_drained,
            }

        def clear(self) -> None:
            self.clear_count += 1
            self.active = False

    active = FakePositionExecutor(active=True)
    assert _hold_active_position_stream_for_synchronous_chunk_gap(active) is True
    assert active.clear_count == 1
    assert active.active is False

    idle = FakePositionExecutor(active=False)
    assert _hold_active_position_stream_for_synchronous_chunk_gap(idle) is False
    assert idle.clear_count == 0
    assert _hold_active_position_stream_for_synchronous_chunk_gap(None) is False

    draining = FakePositionExecutor(active=True, tail_drained=False)
    assert _position_source_tail_pending_before_synchronous_chunk_gap(draining)
    assert _hold_active_position_stream_for_synchronous_chunk_gap(draining) is False
    assert draining.clear_count == 0


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


def _filtered_qpos(side: str, target_qpos: np.ndarray) -> FilteredNormalizedAction:
    vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    return FilteredNormalizedAction(
        action=NormalizedAction(
            vector=vector,
            arm_joint_targets={
                side: np.asarray(target_qpos, dtype=np.float32).reshape(6)
            },
            metadata={"source": "unit_safe_15hz"},
        ),
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
        self.shutdown_calls: list[float] = []
        self.prepare_staged_source_horizon_calls = 0
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

    def prepare_staged_source_horizon(self):
        self.prepare_staged_source_horizon_calls += 1
        return {
            "prepared": True,
            "reason": "fake_initial_source_horizon_prepared",
            "prepare_ms": 0.0,
        }

    def shutdown_background_workers(self, *, timeout_sec: float) -> None:
        self.shutdown_calls.append(float(timeout_sec))


def _filtered_bimanual_pose(left_x: float, right_x: float) -> FilteredNormalizedAction:
    vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    vector[ACTION_ARM_POSE_SLICES["left"]] = _pose(left_x, 0.2)
    vector[ACTION_ARM_POSE_SLICES["right"]] = _pose(right_x, -0.2)
    return FilteredNormalizedAction(
        action=NormalizedAction(vector=vector, metadata={"source": "unit_safe_15hz"}),
        metadata={"safety_pass_count": 1},
    )


def _hold_restart_splice_decision():
    return resolve_position_plan_splice(
        PositionSourcePlanCut(
            cut_id="unit-hold-restart",
            current_source_plan_id="chunk-1",
            current_source_step=0,
            previous_source_plan_id="chunk-0",
            previous_source_step=15,
            relation=POSITION_PLAN_RELATION_REPLACEMENT,
            execution_gap=POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED,
            detected_at=10.0,
        )
    )


def _stream_start_splice_decision():
    return resolve_position_plan_splice(
        PositionSourcePlanCut(
            cut_id="unit-stream-start",
            current_source_plan_id="chunk-0",
            current_source_step=0,
            previous_source_plan_id=None,
            previous_source_step=None,
            relation=POSITION_PLAN_RELATION_STREAM_START,
            execution_gap=POSITION_PLAN_GAP_NONE,
            detected_at=10.0,
        )
    )


def _live_replacement_splice_decision():
    return resolve_position_plan_splice(
        PositionSourcePlanCut(
            cut_id="unit-live-replacement",
            current_source_plan_id="rtc-1",
            current_source_step=0,
            previous_source_plan_id="chunk-0",
            previous_source_step=0,
            relation=POSITION_PLAN_RELATION_REPLACEMENT,
            execution_gap=POSITION_PLAN_GAP_NONE,
            detected_at=10.0 + 1.0 / 15.0,
        )
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


def test_reference_worker_affinity_receipt_is_published(monkeypatch) -> None:
    monkeypatch.setattr(
        deploy_executor_module,
        "bind_current_thread_to_cpu",
        lambda requested_cpu, *, role: {
            "requested": True,
            "applied": True,
            "requested_cpu": requested_cpu,
            "effective_cpu_ids": [requested_cpu],
            "role": role,
        },
    )
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(reference_worker_cpu=6),
        canonical_pipeline=_pipeline(("left",)),
    )
    executor.start()
    try:
        _wait_until(
            lambda: executor.snapshot()["reference_worker_affinity_receipt"]
            is not None
        )
        snapshot = executor.snapshot()
        assert snapshot["reference_worker_cpu"] == 6
        assert snapshot["reference_worker_affinity_receipt"][
            "effective_cpu_ids"
        ] == [6]
    finally:
        executor.close()


def test_reference_worker_affinity_failure_stops_startup(monkeypatch) -> None:
    def fail_affinity(_requested_cpu, *, role):
        raise OSError(f"cannot pin {role}")

    monkeypatch.setattr(
        deploy_executor_module,
        "bind_current_thread_to_cpu",
        fail_affinity,
    )
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(reference_worker_cpu=6),
        canonical_pipeline=_pipeline(("left",)),
    )
    try:
        with pytest.raises(RuntimeError, match="CPU-affinity setup failed"):
            executor.start()
        _wait_until(lambda: executor.snapshot()["fault_latched"] is True)
        assert "cannot pin" in executor.snapshot()[
            "reference_worker_affinity_error"
        ]
    finally:
        executor.close()


def test_cyclic_gc_guard_is_refcounted_and_restores_on_last_close(
    monkeypatch,
) -> None:
    gc_state = {"enabled": True}
    calls: list[object] = []

    monkeypatch.setattr(deploy_executor_module, "_CYCLIC_GC_GUARD_OWNER_COUNT", 0)
    monkeypatch.setattr(
        deploy_executor_module,
        "_CYCLIC_GC_GUARD_RESTORE_ENABLED",
        False,
    )
    monkeypatch.setattr(
        deploy_executor_module.gc,
        "isenabled",
        lambda: bool(gc_state["enabled"]),
    )
    monkeypatch.setattr(deploy_executor_module.gc, "get_count", lambda: (7, 2, 1))

    def collect(generation: int) -> int:
        calls.append(("collect", generation))
        return 19

    def disable() -> None:
        calls.append("disable")
        gc_state["enabled"] = False

    def enable() -> None:
        calls.append("enable")
        gc_state["enabled"] = True

    monkeypatch.setattr(deploy_executor_module.gc, "collect", collect)
    monkeypatch.setattr(deploy_executor_module.gc, "disable", disable)
    monkeypatch.setattr(deploy_executor_module.gc, "enable", enable)

    first = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(cyclic_gc_guard_enabled=True),
        canonical_pipeline=_pipeline(("left",)),
    )
    second = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(cyclic_gc_guard_enabled=True),
        canonical_pipeline=_pipeline(("left",)),
    )
    try:
        assert calls == []
        first.submit(
            _filtered_pose("left", _pose(0.11)),
            _observation(),
            "unit",
        )
        first_activation = first.snapshot()["cyclic_gc_guard_activation"]
        assert calls == [("collect", 2), "disable"]
        assert first_activation["owner_count_before"] == 0
        assert first_activation["owner_count_after"] == 1
        assert first_activation["collection_count_before"] == [7, 2, 1]
        assert first_activation["collected_objects"] == 19
        assert first.snapshot()["cyclic_gc_guard_acquired"] is True

        second.submit(
            _filtered_pose("left", _pose(0.12)),
            _observation(),
            "unit",
        )
        second_activation = second.snapshot()["cyclic_gc_guard_activation"]
        assert calls == [("collect", 2), "disable"]
        assert second_activation["owner_count_before"] == 1
        assert second_activation["owner_count_after"] == 2

        first.close()
        assert calls == [("collect", 2), "disable"]
        assert first.snapshot()["cyclic_gc_guard_release"]["owner_count_after"] == 1
        assert gc_state["enabled"] is False

        second.close()
        assert calls == [("collect", 2), "disable", "enable"]
        assert second.snapshot()["cyclic_gc_guard_release"]["owner_count_after"] == 0
        assert gc_state["enabled"] is True
    finally:
        if not first.snapshot()["closed"]:
            first.close()
        if not second.snapshot()["closed"]:
            second.close()


def test_cyclic_gc_guard_preserves_an_externally_disabled_collector(
    monkeypatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(deploy_executor_module, "_CYCLIC_GC_GUARD_OWNER_COUNT", 0)
    monkeypatch.setattr(
        deploy_executor_module,
        "_CYCLIC_GC_GUARD_RESTORE_ENABLED",
        False,
    )
    monkeypatch.setattr(deploy_executor_module.gc, "isenabled", lambda: False)
    monkeypatch.setattr(deploy_executor_module.gc, "get_count", lambda: (0, 0, 0))
    monkeypatch.setattr(
        deploy_executor_module.gc,
        "collect",
        lambda generation: calls.append(("collect", generation)),
    )
    monkeypatch.setattr(
        deploy_executor_module.gc,
        "disable",
        lambda: calls.append("disable"),
    )
    monkeypatch.setattr(
        deploy_executor_module.gc,
        "enable",
        lambda: calls.append("enable"),
    )

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(cyclic_gc_guard_enabled=True),
        canonical_pipeline=_pipeline(("left",)),
    )
    executor.submit(
        _filtered_pose("left", _pose(0.11)),
        _observation(),
        "unit",
    )
    activation = executor.snapshot()["cyclic_gc_guard_activation"]
    assert activation["gc_enabled_before"] is False
    assert activation["collected_objects"] is None
    executor.close()
    assert calls == []


def test_cyclic_gc_guard_is_not_restored_after_failed_close(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        deploy_executor_module,
        "_acquire_cyclic_gc_guard",
        lambda: {
            "owner_count_before": 0,
            "owner_count_after": 1,
            "gc_enabled_before": True,
            "gc_enabled_after": False,
        },
    )
    monkeypatch.setattr(
        deploy_executor_module,
        "_release_cyclic_gc_guard",
        lambda: calls.append("release"),
    )
    monkeypatch.setattr(deploy_executor_module.gc, "isenabled", lambda: False)

    def fail_hold() -> None:
        raise RuntimeError("hold failed")

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            cyclic_gc_guard_enabled=True,
            debug_events_enabled=True,
        ),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
        hold_callback=fail_hold,
    )
    executor.submit(
        _filtered_pose("left", _pose(0.11)),
        _observation(),
        "unit",
    )

    with pytest.raises(RuntimeError, match="hold failed"):
        executor.close()

    snapshot = executor.snapshot()
    assert calls == []
    assert snapshot["closed"] is True
    assert snapshot["cyclic_gc_guard_acquired"] is True
    assert snapshot["cyclic_gc_guard_release"] is None
    assert snapshot["cyclic_gc_guard_release_deferred"] == {
        "reason": "executor_close_failed",
        "error_count": 1,
        "gc_enabled": False,
    }
    events = executor.drain_debug_events()
    assert any(
        event["event"] == "cyclic_gc_guard_release_deferred"
        for event in events
    )


def test_third_lookahead_requires_second_lookahead() -> None:
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            stale_after_sec=0.5,
            chunk_lookahead_enabled=True,
        ),
        canonical_pipeline=_pipeline(("left",)),
    )
    try:
        with pytest.raises(
            ValueError,
            match="third_lookahead_filtered requires second_lookahead_filtered",
        ):
            executor.submit(
                _filtered_pose("left", _pose(0.1)),
                _observation(),
                "unit",
                lookahead_filtered=_filtered_pose("left", _pose(0.11)),
                third_lookahead_filtered=_filtered_pose("left", _pose(0.13)),
            )
    finally:
        executor.close()


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


def test_pose_target_projector_runs_once_before_deploy_reference_generation() -> None:
    calls: list[dict[str, object]] = []

    def projector(
        targets,
        *,
        observation,
        execution_work_item,
        target_role,
    ):
        calls.append(
            {
                "targets": {
                    side: np.asarray(pose, dtype=float).copy()
                    for side, pose in targets.items()
                },
                "observation": observation,
                "execution_work_item": execution_work_item,
                "target_role": target_role,
            }
        )
        projected = {
            side: np.asarray(pose, dtype=float).copy()
            for side, pose in targets.items()
        }
        projected["left"][0, 3] = 0.12
        return projected

    commands: list[TeleopCommand] = []
    observation = _observation()
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda command: commands.append(command),
        pose_target_projector=projector,
        wall_clock=lambda: 1_700_000_000.0,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.20)),
            observation,
            "unit_policy",
        )
        expected_points = int(
            round(executor.config.reference_hz / executor.config.sample_hz)
        )
        _wait_until(lambda: len(commands) >= expected_points)

        assert len(calls) == 1
        assert calls[0]["observation"] is observation
        assert calls[0]["execution_work_item"] is None
        assert calls[0]["target_role"] == "current"
        emitted_x = [
            float(command.commands["left_arm"].metadata["target_xyzquat"][0])
            for command in commands[:expected_points]
        ]
        assert emitted_x[-1] > 0.11
        assert max(emitted_x) <= 0.12 + 1e-7
    finally:
        executor.close()


def test_qpos_target_does_not_enter_pose_target_projector() -> None:
    calls = 0

    def projector(targets, **_kwargs):
        nonlocal calls
        calls += 1
        return targets

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        pose_target_projector=projector,
    )
    try:
        executor.submit(
            _filtered_qpos("left", np.zeros(6, dtype=float)),
            _observation(),
            "unit_policy",
        )
        assert calls == 0
    finally:
        executor.close()


def test_pose_target_projector_feedback_is_passively_exposed_in_snapshot() -> None:
    class _Projector:
        def __call__(self, targets, **_kwargs):
            return targets

        def feedback_snapshot(self):
            return {
                "schema_version": "unit.tcp_pose_envelope_feedback.v1",
                "state": "limited",
                "profile_id": "unit-envelope",
                "modified_sides": ["left"],
                "receipt_ref": "unit-receipt",
            }

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        pose_target_projector=_Projector(),
    )
    try:
        feedback = executor.snapshot()["pose_target_projector_feedback"]
        assert feedback == {
            "schema_version": "unit.tcp_pose_envelope_feedback.v1",
            "state": "limited",
            "profile_id": "unit-envelope",
            "modified_sides": ["left"],
            "receipt_ref": "unit-receipt",
        }
        assert executor.snapshot()["fault_latched"] is False
    finally:
        executor.close()


def test_active_pose_target_projector_failure_latches_hold_before_ik() -> None:
    hold_calls: list[str] = []

    def reject_projector(_targets, **_kwargs):
        raise RuntimeError("unit binding drift")

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        pose_target_projector=reject_projector,
        dispatch_callback=lambda _command: None,
        hold_callback=lambda: hold_calls.append("hold"),
    )
    try:
        with pytest.raises(
            PositionDeployExecutorFault,
            match="pre-IK tool_tcp pose envelope rejected current",
        ):
            executor.submit(
                _filtered_pose("left", _pose(0.16)),
                _observation(),
                "unit_policy",
            )
        snapshot = executor.snapshot()
        assert snapshot["fault_latched"] is True
        assert snapshot["fault_stage"] == "target_pose_envelope"
        assert snapshot["fault_metadata"]["manual_rearm_allowed"] is False
        assert snapshot["fault_metadata"]["manual_rearm_required"] is False
        assert snapshot["fault_metadata"]["restart_and_rehome_required"] is True
        assert hold_calls == ["hold"]
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


def test_plan_splice_clock_rebase_requires_typed_governor() -> None:
    with pytest.raises(
        ValueError,
        match="plan_splice_clock_rebase_enabled requires",
    ):
        PositionDeployExecutorConfig(plan_splice_clock_rebase_enabled=True)


def test_stream_start_acquisition_is_single_shot_and_rtc_replacement_safe() -> None:
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
            stale_after_sec=1.0,
            plan_splice_governor_enabled=True,
            stream_start_acquisition=PositionStreamStartAcquisitionConfig(
                enabled=True,
                max_translation_step_m=0.005,
                max_rotation_step_rad=np.deg2rad(0.5),
                max_source_intervals=16,
            ),
        ),
        canonical_pipeline=_pipeline(("left",)),
    )
    try:
        first_index = executor.submit(
            _filtered_pose("left", _pose(0.143)),
            _observation(left_pose=_pose(0.1)),
            "rtc_or_sync_policy",
            timestamp_monotonic=10.0,
            source_plan_cut=_stream_start_splice_decision().cut,
        )
        assert first_index == 9
        executor._drain_submissions()

        first_sample = executor._pose_resampler.sample_at(10.0)
        np.testing.assert_allclose(
            first_sample.pose_by_side["left"][:3, 3],
            [0.1, 0.0, 0.3],
            atol=1e-7,
        )
        before_replacement = executor.snapshot()
        assert before_replacement["stream_start_acquisition_count"] == 1
        assert before_replacement["stream_source_timestamp_offset_sec"] == (
            pytest.approx(9.0 / 15.0)
        )

        replacement_index = executor.submit(
            _filtered_pose("left", _pose(0.145)),
            _observation(left_pose=_pose(0.1)),
            "rtc_or_sync_policy",
            timestamp_monotonic=10.0 + 1.0 / 15.0,
            source_plan_cut=_live_replacement_splice_decision().cut,
        )
        assert replacement_index == 10
        executor._drain_submissions()

        after_replacement = executor.snapshot()
        assert after_replacement["stream_start_acquisition_count"] == 1
        assert after_replacement["stream_source_timestamp_offset_sec"] == (
            pytest.approx(9.0 / 15.0)
        )
        target_sample = executor._pose_resampler.sample_at(
            10.0 + 9.0 / 15.0
        )
        np.testing.assert_allclose(
            target_sample.pose_by_side["left"][:3, 3],
            [0.143, 0.0, 0.3],
            atol=1e-6,
        )
    finally:
        executor.close()


def test_synchronous_first_chunk_cut_reaches_stream_start_acquisition() -> None:
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
            stale_after_sec=1.0,
            plan_splice_governor_enabled=True,
            stream_start_acquisition=PositionStreamStartAcquisitionConfig(
                enabled=True,
                max_translation_step_m=0.005,
                max_rotation_step_rad=np.deg2rad(0.5),
                max_source_intervals=16,
            ),
        ),
        canonical_pipeline=_pipeline(("left",)),
    )
    detector = DeploymentPositionPlanCutDetector()
    raw_action = NormalizedAction(
        vector=np.full(ACTION_DIM, np.nan, dtype=np.float32),
        metadata={
            "deploy_action_chunk_id": 0,
            "deploy_action_chunk_index": 0,
            "deploy_action_chunk_size": 16,
        },
    )
    proposal = _prepare_position_plan_cut_for_action_step(
        detector,
        SimpleNamespace(raw_action=raw_action, emission=None),
        rtc_enabled=False,
        policy_node_id="deploy-policy",
        # qpos-init/startup may clear the idle executor before policy starts.
        hold_generation=1,
        detected_at=10.0,
    )
    try:
        source_index = executor.submit(
            _filtered_pose("left", _pose(0.143)),
            _observation(left_pose=_pose(0.1)),
            "sync_policy",
            timestamp_monotonic=10.0,
            source_plan_cut=proposal.cut,
        )
        detector.commit(proposal)
        executor._drain_submissions()

        assert source_index == 9
        snapshot = executor.snapshot()
        assert snapshot["stream_start_acquisition_count"] == 1
        assert snapshot["last_stream_start_acquisition"]["active"] is True
    finally:
        executor.close()


def test_synchronous_next_chunk_after_hold_restarts_stream_start_acquisition() -> None:
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
            stale_after_sec=1.0,
            plan_splice_governor_enabled=True,
            stream_start_acquisition=PositionStreamStartAcquisitionConfig(
                enabled=True,
                max_translation_step_m=0.005,
                max_rotation_step_rad=np.deg2rad(0.5),
                max_source_intervals=16,
            ),
        ),
        canonical_pipeline=_pipeline(("left",)),
    )
    detector = DeploymentPositionPlanCutDetector()

    def synchronous_action(chunk_id: int) -> NormalizedAction:
        return NormalizedAction(
            vector=np.full(ACTION_DIM, np.nan, dtype=np.float32),
            metadata={
                "deploy_action_chunk_id": chunk_id,
                "deploy_action_chunk_index": 0,
                "deploy_action_chunk_size": 16,
            },
        )

    try:
        first = detector.prepare_synchronous_action(
            synchronous_action(0),
            policy_node_id="deploy-policy",
            detected_at=10.0,
        )
        executor.submit(
            _filtered_pose("left", _pose(0.143)),
            _observation(left_pose=_pose(0.1)),
            "sync-policy",
            timestamp_monotonic=10.0,
            source_plan_cut=first.cut,
        )
        detector.commit(first)
        executor._drain_submissions()
        assert executor.snapshot()["stream_start_acquisition_count"] == 1

        executor.clear()
        second = _prepare_position_plan_cut_for_action_step(
            detector,
            SimpleNamespace(
                raw_action=synchronous_action(1),
                emission=None,
            ),
            rtc_enabled=False,
            policy_node_id="deploy-policy",
            hold_generation=int(executor.snapshot()["clear_count"]),
            detected_at=11.0,
        )
        assert resolve_position_plan_splice(second.cut).mode == (
            POSITION_PLAN_SPLICE_HOLD_RESTART
        )
        executor.submit(
            _filtered_pose("left", _pose(0.19)),
            _observation(left_pose=_pose(0.143)),
            "sync-policy",
            timestamp_monotonic=11.0,
            source_plan_cut=second.cut,
        )
        detector.commit(second)
        executor._drain_submissions()

        snapshot = executor.snapshot()
        assert snapshot["stream_start_acquisition_count"] == 2
        assert snapshot["last_stream_start_acquisition"]["active"] is True
    finally:
        executor.close()


def test_default_on_stream_start_acquisition_preserves_pure_qpos_streams() -> None:
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
            stale_after_sec=1.0,
            plan_splice_governor_enabled=True,
            stream_start_acquisition=PositionStreamStartAcquisitionConfig(
                enabled=True,
            ),
        ),
        canonical_pipeline=_pipeline(("left",)),
    )
    try:
        source_index = executor.submit(
            _filtered_qpos("left", np.full(6, 0.1)),
            _observation(),
            "qpos_policy",
            timestamp_monotonic=10.0,
            source_plan_cut=_stream_start_splice_decision().cut,
        )
        executor._drain_submissions()

        assert source_index == 1
        assert executor.snapshot()["stream_start_acquisition_count"] == 0
    finally:
        executor.close()


def test_active_stream_start_acquisition_rebases_after_initial_planning() -> None:
    observation = _observation(
        sides=("left", "right"),
        left_pose=_pose(0.1, 0.2),
        right_pose=_pose(0.1, -0.2),
    )
    started_at = max(
        float(observation.arm_states[side]["measured_timestamp_monotonic"])
        for side in ("left", "right")
    )
    now = {"value": started_at}

    class AdvancingPlanner(_FakePositionCollisionPlanner):
        def prepare_staged_source_horizon(self):
            receipt = super().prepare_staged_source_horizon()
            now["value"] += 0.050
            return {**receipt, "prepare_ms": 50.0}

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
            stale_after_sec=1.0,
            collision_required=True,
            allow_non_authoritative_collision_reset_in_dry_run=True,
            plan_splice_governor_enabled=True,
            plan_splice_clock_rebase_enabled=True,
            stream_start_acquisition=PositionStreamStartAcquisitionConfig(
                enabled=True,
                max_translation_step_m=0.005,
                max_source_intervals=16,
            ),
        ),
        canonical_pipeline=_pipeline(("left", "right")),
        collision_planner_factory=AdvancingPlanner,
        clock=lambda: now["value"],
        wall_clock=lambda: now["value"],
    )
    try:
        executor.submit(
            _filtered_bimanual_pose(0.143, 0.143),
            observation,
            "stream_start",
            timestamp_monotonic=started_at,
            source_plan_cut=_stream_start_splice_decision().cut,
        )
        executor._drain_submissions()

        snapshot = executor.snapshot()
        assert snapshot["stream_start_acquisition_count"] == 1
        assert snapshot["plan_splice_clock_rebase_count"] == 1
        assert snapshot["plan_splice_reference_query_offset_sec"] == (
            pytest.approx(0.050)
        )
        assert snapshot["plan_splice_tail_drain_required"] is True
    finally:
        executor.close()


def test_hold_restart_rebases_after_planning_and_reports_tail_drain() -> None:
    now = {"value": 10.0}

    class AdvancingPlanner(_FakePositionCollisionPlanner):
        def prepare_staged_source_horizon(self):
            receipt = super().prepare_staged_source_horizon()
            now["value"] += 0.050
            return {**receipt, "prepare_ms": 50.0}

    planner = AdvancingPlanner()
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
            stale_after_sec=0.5,
            collision_required=True,
            allow_non_authoritative_collision_reset_in_dry_run=True,
            chunk_lookahead_enabled=True,
            plan_splice_governor_enabled=True,
            plan_splice_clock_rebase_enabled=True,
        ),
        canonical_pipeline=_pipeline(("left", "right")),
        collision_planner_factory=lambda: planner,
        clock=lambda: now["value"],
        wall_clock=lambda: now["value"],
    )
    try:
        executor.submit(
            _filtered_bimanual_pose(0.16, 0.18),
            _observation(sides=("left", "right")),
            "hold_restart",
            timestamp_monotonic=10.0,
            discrete_command=_hand_command("left", 0.2),
            lookahead_filtered=_filtered_bimanual_pose(0.20, 0.22),
            lookahead_observation=_observation(sides=("left", "right")),
            lookahead_discrete_command=_hand_command("left", 0.4),
            source_plan_cut=_hold_restart_splice_decision().cut,
        )
        executor._drain_submissions()

        prepared = executor.snapshot()
        assert prepared["plan_splice_reference_query_offset_sec"] == pytest.approx(
            0.050
        )
        assert prepared["plan_splice_clock_rebase_count"] == 1

        executor._emit_reference(scheduled_timestamp=now["value"])
        first = executor.snapshot()
        first_metadata = first["last_reference"]
        assert first_metadata["position_reference_query_monotonic_time"] == pytest.approx(
            10.0
        )
        assert first_metadata["position_interpolation_alpha"] == pytest.approx(0.0)
        assert first["plan_splice_tail_drain"]["drained"] is False
        assert first["plan_splice_tail_drain"][
            "terminal_discrete_drained"
        ] is False
        assert _position_source_tail_pending_before_synchronous_chunk_gap(executor)
        assert _hold_active_position_stream_for_synchronous_chunk_gap(executor) is False

        now["value"] += 1.0 / 15.0
        executor._emit_reference(scheduled_timestamp=now["value"])
        drained = executor.source_tail_drain_status()
        assert drained["required"] is True
        assert drained["drained"] is True
        assert drained["reason"] == "terminal_source_emitted"
        assert drained["terminal_discrete_drained"] is True

        executor.clear()
        reset = executor.snapshot()
        assert reset["plan_splice_reference_query_offset_sec"] == 0.0
        assert reset["plan_splice_tail_drain"]["required"] is False
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
    assert planner.shutdown_calls == [pytest.approx(2.0)]
    assert planner.prepare_staged_source_horizon_calls == 2


def test_collision_dispatch_allows_transient_low_send_rate_receipt() -> None:
    planner = _FakePositionCollisionPlanner()
    commands: list[TeleopCommand] = []

    def timing_degraded_dispatch(command: TeleopCommand) -> dict[str, object]:
        commands.append(command)
        background = {
            "enabled": True,
            "healthy": False,
            "health_reasons": ["low_send_rate"],
            "thread_alive": True,
            "target_active": True,
            "target_stale": False,
            "fault": None,
        }
        return {
            "accepted": True,
            "position_dispatch_timing": {
                "total_ms": 7.5,
                "backend_dispatch_ms": 6.25,
            },
            "effectors": {
                name: {
                    "accepted": True,
                    "command_type": "joint_vector",
                    "requested_value": np.asarray(item.value, dtype=float).copy(),
                    "final_value": np.asarray(item.value, dtype=float).copy(),
                    "rewritten": False,
                    "singularity_gate_status": "pass",
                    "dispatch_timing": {"send_command_ms": 2.5},
                    "receipt_metadata": {
                        "target_admission_timing": {
                            "mode": "latest_nonblocking",
                            "total_ms": 0.4,
                            "send_lock_wait_ms": 0.0,
                        }
                    },
                }
                for name, item in command.commands.items()
            },
            "arm_backend_health": {
                "ready": True,
                "sides": {
                    side: {
                        "ready": True,
                        "reasons": [],
                        "backend_ready": True,
                        "state_error": None,
                        "background_streaming": dict(background),
                        "warnings": ["low_send_rate"],
                    }
                    for side in ("left", "right")
                },
            },
        }

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            stale_after_sec=0.5,
            collision_required=True,
        ),
        canonical_pipeline=_pipeline(("left", "right")),
        dispatch_callback=timing_degraded_dispatch,
        collision_planner_factory=lambda: planner,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_bimanual_pose(0.16, 0.18),
            _observation(sides=("left", "right")),
            "collision_policy_timing_degraded",
        )
        _wait_until(lambda: executor.snapshot()["reference_count"] >= 1)
        snapshot = executor.snapshot()
        assert snapshot["fault_latched"] is False
        assert commands
        assert snapshot["last_reference"][
            "position_arm_backend_dispatch_verified"
        ] is True
        assert snapshot["last_reference"][
            "position_arm_dispatch_callback_ms"
        ] >= 0.0
        assert snapshot["last_reference"][
            "position_arm_backend_dispatch_timing"
        ]["backend_dispatch_ms"] == pytest.approx(6.25)
        left_summary = snapshot["last_reference"][
            "position_arm_backend_dispatch_by_effector"
        ]["left_arm"]
        assert left_summary["dispatch_timing"]["send_command_ms"] == pytest.approx(2.5)
        assert left_summary["target_admission_timing"]["mode"] == "latest_nonblocking"
    finally:
        executor.close()


def test_collision_dispatch_still_rejects_stale_background_target_receipt() -> None:
    planner = _FakePositionCollisionPlanner()

    def stale_target_dispatch(command: TeleopCommand) -> dict[str, object]:
        background = {
            "enabled": True,
            "healthy": False,
            "health_reasons": ["target_stale", "low_send_rate"],
            "thread_alive": True,
            "target_active": True,
            "target_stale": True,
            "fault": None,
        }
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
                    "background_streaming": dict(background),
                }
                for name, item in command.commands.items()
            },
        }

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            stale_after_sec=0.5,
            collision_required=True,
        ),
        canonical_pipeline=_pipeline(("left", "right")),
        dispatch_callback=stale_target_dispatch,
        collision_planner_factory=lambda: planner,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_bimanual_pose(0.16, 0.18),
            _observation(sides=("left", "right")),
            "collision_policy_stale_target",
        )
        _wait_until(lambda: executor.snapshot()["fault_latched"] is True)
        with pytest.raises(
            PositionDeployExecutorFault,
            match="became unhealthy",
        ):
            executor.raise_if_fault()
    finally:
        executor.close()


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


def test_collision_reset_preflight_waits_without_mutating_planner_or_stream() -> None:
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
        stale = _observation(
            sides=("left", "right"),
            arm_state_overrides={
                side: {
                    "hardware_read_timestamp_monotonic": 10.0,
                    "measured_timestamp_monotonic": 10.0,
                }
                for side in ("left", "right")
            },
        )
        waiting = executor.collision_reset_preflight(stale)
        assert waiting["required"] is True
        assert waiting["ready"] is False
        assert "authoritative qpos is stale" in str(waiting["reason"])
        assert planner.reset_calls == []
        assert executor.snapshot()["submitted_count"] == 0

        fresh = _observation(
            sides=("left", "right"),
            arm_state_overrides={
                side: {
                    "hardware_read_timestamp_monotonic": 10.30,
                    "measured_timestamp_monotonic": 10.30,
                }
                for side in ("left", "right")
            },
        )
        ready = executor.collision_reset_preflight(fresh)
        assert ready["required"] is True
        assert ready["ready"] is True
        assert ready["reason"] == "fresh_collision_reset_available"
        assert planner.reset_calls == []
        assert executor.snapshot()["submitted_count"] == 0
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


def test_reference_deadline_burst_faults_instead_of_catching_up() -> None:
    hold_called = threading.Event()

    def overrun_dispatch(_command: TeleopCommand) -> None:
        time.sleep(0.060)

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
            stale_after_sec=0.5,
            max_reference_tick_skip_burst=1,
            debug_events_enabled=True,
        ),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=overrun_dispatch,
        hold_callback=hold_called.set,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "deadline-overrun",
        )
        _wait_until(lambda: executor.snapshot()["fault_latched"])
        assert hold_called.wait(timeout=0.5)
        snapshot = executor.snapshot()
        assert snapshot["fault_stage"] == "reference_worker"
        assert snapshot["fault_type"] == "TimeoutError"
        assert snapshot["missed_reference_ticks"] > 1
        assert snapshot["reference_timing_observation_count"] >= 1
        assert snapshot["reference_tick_skip_event_count"] >= 1
        assert snapshot["last_reference_timing"]["skipped_ticks"] > 1
        assert snapshot["max_reference_emit_elapsed_ms"] >= 50.0
        assert snapshot["max_reference_emit_cpu_elapsed_ms"] >= 0.0
        assert snapshot["max_reference_non_cpu_wall_ms"] >= 0.0
        assert snapshot["max_reference_tick_skip_burst"] == 1
        timing_events = [
            event
            for event in executor.drain_debug_events()
            if event["event"] == "reference_timing_overrun"
        ]
        assert timing_events
        assert timing_events[-1]["skipped_ticks"] > 1
        assert timing_events[-1]["classification"] in {
            "thread_cpu_overrun",
            "non_cpu_wall_stall",
            "start_late",
            "combined_minor_overrun",
        }
        with pytest.raises(
            PositionDeployExecutorFault,
            match="deadline burst exceeded",
        ) as exc_info:
            executor.raise_if_fault()
        message = str(exc_info.value)
        assert "emit_cpu_elapsed_ms=" in message
        assert "planner_frame_total_ms=" in message
        assert "planner_precompute_submit_ms=" in message
    finally:
        executor.close()


def test_allowed_reference_tick_skip_is_observable_before_fault_threshold() -> None:
    def single_tick_overrun(_command: TeleopCommand) -> None:
        time.sleep(0.025)

    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
            stale_after_sec=0.5,
            max_reference_tick_skip_burst=100,
            debug_events_enabled=True,
        ),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=single_tick_overrun,
    )
    executor.start()
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "allowed-deadline-overrun",
        )
        _wait_until(lambda: executor.snapshot()["missed_reference_ticks"] >= 1)
        snapshot = executor.snapshot()
        assert snapshot["fault_latched"] is False
        assert snapshot["reference_tick_skip_event_count"] >= 1
        assert snapshot["last_reference_timing"]["skipped_ticks"] >= 1
        timing_events = [
            event
            for event in executor.drain_debug_events()
            if event["event"] == "reference_timing_overrun"
        ]
        assert timing_events
        assert timing_events[-1]["emit_elapsed_ms"] >= 20.0
    finally:
        executor.close()


def test_reference_submission_drain_can_be_bounded_to_preserve_emit_deadline() -> None:
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(stale_after_sec=0.5),
        canonical_pipeline=_pipeline(("left",)),
        dispatch_callback=lambda _command: None,
    )
    try:
        executor.submit(
            _filtered_pose("left", _pose(0.16)),
            _observation(),
            "first",
        )
        executor.submit(
            _filtered_pose("left", _pose(0.17)),
            _observation(),
            "second",
        )

        assert executor.snapshot()["queue_depth"] == 2
        assert executor._drain_submissions(max_items=1) == 1
        snapshot = executor.snapshot()
        assert snapshot["queue_depth"] == 1
        assert snapshot["active"] is True

        assert executor._drain_submissions(max_items=1) == 1
        assert executor.snapshot()["queue_depth"] == 0
        assert executor._drain_submissions(max_items=1) == 0
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


@pytest.mark.parametrize(
    ("audit_mode", "expected_state_reads"),
    [
        (POSITION_REFERENCE_DISPATCH_AUDIT_POST, 1),
        (POSITION_REFERENCE_DISPATCH_AUDIT_FULL, 3),
    ],
)
def test_position_reference_dispatch_audit_mode_bounds_synchronous_state_reads(
    audit_mode: str,
    expected_state_reads: int,
) -> None:
    class CountingArmBackend:
        def __init__(self, side: str) -> None:
            self.side = side
            self.ready_reads = 0
            self.state_reads = 0

        def is_ready(self) -> bool:
            self.ready_reads += 1
            return True

        def get_state(self) -> dict[str, object]:
            self.state_reads += 1
            return {
                "hardware_ready": True,
                "emergency_stopped": False,
                "send_result_code": 0,
                "background_streaming": {
                    "enabled": True,
                    "healthy": True,
                    "health_reasons": [],
                    "thread_alive": True,
                    "target_active": True,
                    "target_stale": False,
                    "fault": None,
                },
            }

        def send_command(self, command: EffectorCommand) -> CommandDispatchReceipt:
            command.metadata["singularity_metadata"] = {
                "singularity_gate_status": "pass"
            }
            return CommandDispatchReceipt(
                effector_name=f"{self.side}_arm",
                accepted=True,
                operation="queue_background_joint_target",
                status="queued",
            )

    backends = {side: CountingArmBackend(side) for side in ("left", "right")}
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

    result = dispatch_position_reference_command(
        command,
        arm_backends=backends,
        required_sides=("left", "right"),
        audit_mode=audit_mode,
    )

    assert result["accepted"] is True
    assert result["position_dispatch_audit_mode"] == audit_mode
    assert result["arm_backend_health"]["ready"] is True
    for backend in backends.values():
        assert backend.ready_reads == expected_state_reads
        assert backend.state_reads == expected_state_reads


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
    assert result["dispatch_timing"]["total_ms"] >= 0.0
    assert receipt["dispatch_timing"]["send_command_ms"] >= 0.0
    assert receipt["dispatch_timing"]["evidence_get_state_ms"] >= 0.0
    assert receipt["dispatch_timing"]["evidence_is_ready_ms"] >= 0.0
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


def test_position_dispatch_allows_transient_low_send_rate_warning() -> None:
    class TimingDegradedArmBackend:
        def is_ready(self) -> bool:
            return True

        def get_state(self) -> dict[str, object]:
            return {
                "hardware_ready": True,
                "background_streaming": {
                    "enabled": True,
                    "healthy": False,
                    "health_reasons": ["low_send_rate"],
                    "thread_alive": True,
                    "target_active": True,
                    "target_stale": False,
                    "fault": None,
                    "shutdown_requested": False,
                    "shutdown_complete": False,
                },
            }

    health = require_arm_backends_ready_for_position_dispatch(
        {
            "left": TimingDegradedArmBackend(),
            "right": TimingDegradedArmBackend(),
        },
        require_background_target=True,
        stage="pre_arm_reference_queue",
    )

    assert health["ready"] is True
    assert health["sides"]["left"]["reasons"] == []
    assert health["sides"]["left"]["warnings"] == ["low_send_rate"]


def test_position_dispatch_rejects_unknown_background_unhealthy_state() -> None:
    class UnknownUnhealthyArmBackend:
        def is_ready(self) -> bool:
            return True

        def get_state(self) -> dict[str, object]:
            return {
                "hardware_ready": True,
                "background_streaming": {
                    "enabled": True,
                    "healthy": False,
                    "health_reasons": [],
                    "thread_alive": True,
                    "target_active": True,
                    "target_stale": False,
                    "fault": None,
                    "shutdown_requested": False,
                    "shutdown_complete": False,
                },
            }

    with pytest.raises(RuntimeError, match="background_unhealthy_unknown"):
        require_arm_backends_ready_for_position_dispatch(
            {"left": UnknownUnhealthyArmBackend()},
            require_background_target=True,
            stage="pre_arm_reference_queue",
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
        phase_timing={
            "before_primary_prepare_ms": 1.25,
            "position_submit_ms": 2.5,
            "loop_before_debug_ms": 4.0,
        },
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
    assert record["phase_timing"] == {
        "before_primary_prepare_ms": 1.25,
        "position_submit_ms": 2.5,
        "loop_before_debug_ms": 4.0,
    }


def test_deploy_debug_serialization_error_does_not_disable_later_events(
    tmp_path,
) -> None:
    logger = DeploymentDebugLogger(
        enabled=True,
        root_dir=tmp_path,
        active_sides=("left",),
    )
    logger.log_event("bad_payload", {"unsupported": object()})
    assert logger.enabled is True
    logger.log_event("good_payload", {"value": 7})
    path = logger.path
    logger.close()

    assert path is not None
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["event"] == "debug_serialization_error"
    assert records[0]["source_event"] == "bad_payload"
    assert records[1]["event"] == "good_payload"
    assert records[1]["value"] == 7


def test_deploy_debug_compacts_repeated_policy_metadata_by_default(tmp_path) -> None:
    large_policy_artifact = {
        "checkpoint": "stage1/150000",
        "manifest": "x" * 100_000,
    }
    action = NormalizedAction(
        vector=np.zeros(ACTION_DIM, dtype=np.float32),
        metadata={
            "source": "unit",
            "deploy_action_chunk_id": "chunk-1",
            "remote_policy": {
                "server": {
                    "model_id": "writing-stage1",
                    "policy_artifact": large_policy_artifact,
                },
                "server_summary": {
                    "model_id": "writing-stage1",
                    "checkpoint_step": "150000",
                },
                "policy_timing": {"infer_ms": 12.5},
            },
        },
    )
    logger = DeploymentDebugLogger(
        enabled=True,
        root_dir=tmp_path / "compact",
        active_sides=("left",),
    )
    logger.log_policy_chunk(
        timestamp=1.0,
        step_index=2,
        policy_name="unit",
        policy_output=object(),
        actions=[action],
        accepted=True,
        discarded_by_ik=False,
        queue_policy="replace",
        queue_len_before=0,
        queue_len_after=1,
        faults=[],
    )
    path = logger.path
    logger.close()

    assert path is not None
    encoded = path.read_text(encoding="utf-8").strip()
    record = json.loads(encoded)
    remote_policy = record["actions"][0]["metadata"]["remote_policy"]
    assert record["debug_payload_mode"] == "compact"
    assert remote_policy["server_summary"]["checkpoint_step"] == "150000"
    assert remote_policy["server_identity"]["model_id"] == "writing-stage1"
    assert remote_policy["_debug_omitted_server_detail"] is True
    assert "policy_artifact" not in remote_policy
    assert len(encoded) < 20_000


def test_deploy_debug_full_payload_mode_is_explicit_rollback(tmp_path) -> None:
    action = NormalizedAction(
        vector=np.zeros(ACTION_DIM, dtype=np.float32),
        metadata={
            "source": "unit",
            "remote_policy": {
                "server": {
                    "model_id": "writing-stage1",
                    "policy_artifact": {"manifest": "x" * 20_000},
                }
            },
        },
    )
    logger = DeploymentDebugLogger(
        enabled=True,
        root_dir=tmp_path / "full",
        active_sides=("left",),
        payload_mode=DEPLOY_DEBUG_PAYLOAD_FULL,
    )
    logger.log_policy_chunk(
        timestamp=1.0,
        step_index=2,
        policy_name="unit",
        policy_output=object(),
        actions=[action],
        accepted=True,
        discarded_by_ik=False,
        queue_policy="replace",
        queue_len_before=0,
        queue_len_after=1,
        faults=[],
    )
    path = logger.path
    logger.close()

    assert path is not None
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["debug_payload_mode"] == "full"
    assert (
        record["actions"][0]["metadata"]["remote_policy"]["server"]
        ["policy_artifact"]["manifest"]
        == "x" * 20_000
    )


def test_compact_arm_state_keeps_timing_but_drops_duplicate_command_detail() -> None:
    state = {
        "current_qpos": np.zeros(6, dtype=np.float32),
        "hardware_read_phase_timing": {
            "total_wall_ms": 15.0,
            "total_thread_cpu_ms": 2.0,
            "sdk_call_wall_ms": 8.0,
        },
        "last_command": {
            "command_type": "joint_vector",
            "value": np.ones(6, dtype=np.float32),
            "execution_work_item": {"large": "x" * 20_000},
            "metadata": {
                "position_reference_sequence": 17,
                "position_collision_status": "safe",
                "position_collision_frame_total_ms": 2.5,
                "position_collision_debug": {"large": "x" * 20_000},
            },
        },
    }

    compact_state = _debug_arm_state_summary({"left": state})["left"]
    full_state = _debug_arm_state_summary(
        {"left": state},
        full_payload=True,
    )["left"]
    compact = compact_state["last_command"]
    full = full_state["last_command"]

    assert compact_state["hardware_read_phase_timing"] == state[
        "hardware_read_phase_timing"
    ]
    assert full_state["hardware_read_phase_timing"] == state[
        "hardware_read_phase_timing"
    ]
    assert compact["metadata"]["position_reference_sequence"] == 17
    assert compact["metadata"]["position_collision_status"] == "safe"
    assert compact["metadata"]["position_collision_frame_total_ms"] == 2.5
    assert "position_collision_debug" not in compact["metadata"]
    assert "execution_work_item" not in compact
    assert full["metadata"]["position_collision_debug"]["large"] == "x" * 20_000
    assert full["execution_work_item"]["large"] == "x" * 20_000


def test_compact_position_executor_event_keeps_dispatch_evidence_bounded(
    tmp_path,
) -> None:
    logger = DeploymentDebugLogger(
        enabled=True,
        root_dir=tmp_path,
        active_sides=("left",),
    )
    logger.log_event(
        "position_executor_arm_reference_dispatch",
        {
            "position_reference_sequence": 21,
            "position_collision_status": "safe",
            "position_collision_frame_total_ms": 3.0,
            "position_collision_debug": {
                "q_ik_raw_by_side": {"left": [0.0] * 6},
                "q_ref_by_side": {"left": [0.1] * 6},
                "planner_metadata": {"large": "x" * 100_000},
            },
            "position_arm_backend_dispatch_by_effector": {
                "left_arm": {
                    "accepted": True,
                    "status": "queued",
                    "dispatch_timing": {"total_ms": 0.4},
                    "background_streaming": {
                        "healthy": True,
                        "target_sequence": 42,
                        "last_call_ms": 0.2,
                        "large": "x" * 100_000,
                    },
                }
            },
            "position_arm_backend_group_health": {
                "ready": True,
                "sides": {
                    "left": {
                        "ready": True,
                        "timing": {"total_ms": 0.1},
                        "background_streaming": {"large": "x" * 100_000},
                    }
                },
            },
        },
    )
    path = logger.path
    logger.close()

    assert path is not None
    encoded = path.read_text(encoding="utf-8").strip()
    record = json.loads(encoded)
    assert record["position_reference_sequence"] == 21
    assert record["position_collision_status"] == "safe"
    assert record["position_collision_debug"]["q_ref_by_side"]["left"] == [0.1] * 6
    dispatch = record["position_arm_backend_dispatch_by_effector"]["left_arm"]
    assert dispatch["dispatch_timing"]["total_ms"] == 0.4
    assert dispatch["background_streaming"]["target_sequence"] == 42
    assert record["position_arm_backend_group_health"]["sides"]["left"][
        "timing"
    ]["total_ms"] == 0.1
    assert len(encoded) < 10_000


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
