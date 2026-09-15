from __future__ import annotations

from collections import deque
from contextlib import ExitStack, contextmanager
import threading
import time

import numpy as np
import pytest

from robot_motion.position_control import (
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
)
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.deploy import run_policy
from teleop_runtime.schemas import EffectorCommand
from teleop_runtime.position_control.deploy_executor import PositionDeployExecutor, PositionDeployExecutorConfig
from tests.test_position_deploy_executor import _filtered_qpos, _observation, _pipeline


class _DelayedArm:
    """Fake SDK boundary with delayed feedback; no socket or device is opened."""

    def __init__(self) -> None:
        self.sent: list[np.ndarray] = []
        self.feedback = deque([np.zeros(6) for _ in range(6)], maxlen=6)
        self.stop_count = 0
        self.lock = threading.Lock()

    def Movej_CANFD(self, joint_deg, **kwargs):
        del kwargs
        qpos = np.deg2rad(np.asarray(joint_deg, dtype=float))
        with self.lock:
            self.sent.append(qpos.copy())
            self.feedback.append(qpos.copy())
        return 0

    def Get_Current_Arm_State(self):
        with self.lock:
            return 0, np.rad2deg(self.feedback[0]).tolist(), [0.0] * 6, 0, 0

    def Arm_Slow_Stop(self):
        self.stop_count += 1
        return 0

    def Arm_Stop(self):
        self.stop_count += 1
        return 0

    def RM_API_UnInit(self):
        pass

    def Arm_Socket_Close(self):
        pass


@contextmanager
def _position_backend(side="left", *, planner=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED):
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side=side,
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            control_dt=1.0 / 60.0,
            background_streaming_enabled=True,
            background_streaming_hz=120.0,
            position_command_planner=planner,
            position_velocity_ramp_sec=0.10,
            singularity_protection_enabled=False,
        )
    )
    arm = _DelayedArm()
    backend.arm = arm
    # Exercise hardware-mode admission using only the injected SDK.  Never
    # call start(): that is the device-opening boundary, not part of this test.
    backend.config.connect_hardware = True
    backend.refresh_state_from_hardware(include_derived_pose=False)
    backend._stream_thread = threading.Thread(target=backend._background_stream_loop, daemon=True)
    backend._stream_thread.start()
    try:
        yield backend, arm
    finally:
        backend.close()
        assert backend._stream_thread is None or not backend._stream_thread.is_alive()


def _command(side, target):
    return EffectorCommand(
        effector_name=f"{side}_arm",
        side=side,
        command_type="joint_vector",
        value=np.full(6, target),
        metadata={
            "action_pose_source": "limited_solver_output_qpos",
            "collision_avoidance_status": "tracking",
            "ik_jump_gate_status": "pass",
            "execution_step_metadata": {"enabled": True, "step_limited": False},
            "ik_solve_pose_error": {"translation_m": 0.0, "rotation_deg": 0.0},
        },
    )


def _state(backend):
    return backend.refresh_state_from_hardware(
        include_derived_pose=False, include_timing_percentiles=False
    )["background_streaming"]


def _wait(backend, predicate, *, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _state(backend)
        assert state["fault"] is None
        if predicate(state):
            return state
        time.sleep(0.005)
    pytest.fail("fake-SDK position worker did not reach the expected state")


class _Executor:
    def __init__(self):
        self.clear_count = 0

    def clear(self):
        self.clear_count += 1

    def snapshot(self):
        return {"active": True}


def test_active_rtc_bimanual_position_hold_decelerates_keeps_sequence_and_recovers():
    with ExitStack() as stack:
        pairs = {side: stack.enter_context(_position_backend(side)) for side in ("left", "right")}
        backends = {side: pair[0] for side, pair in pairs.items()}
        executor = PositionDeployExecutor(
            config=PositionDeployExecutorConfig(),
            canonical_pipeline=_pipeline(("left", "right")),
            hold_callback=lambda: run_policy.hold_deploy_arm_backends(backends, reason="position_clear"),
        )
        stack.callback(executor.close)
        obsolete = _filtered_qpos("left", np.full(6, 0.30))
        obsolete.action.arm_joint_targets["right"] = np.full(6, 0.30)
        executor.submit(obsolete, _observation(sides=("left", "right")), "rtc_old_plan")
        for side, backend in backends.items():
            backend.send_command(_command(side, 0.30))
            _wait(backend, lambda state: state["accepted_send_count"] >= 5)
        snapshots = {side: _state(backend) for side, backend in backends.items()}
        for state in snapshots.values():
            assert np.max(np.abs(state["trajectory_planner"]["qd_command_rad_s"])) > 0.0

        receipts = run_policy._hold_active_rtc_gap(
            reason="rtc_plan_underrun", send_actions=True,
            arm_backends=backends, position_deploy_executor=executor,
        )
        assert executor.snapshot()["clear_count"] == 1
        assert executor.snapshot()["queue_depth"] == 0
        for side, receipt in receipts.items():
            assert receipt["mode"] == "position_planner_quick_stop"
            assert receipt["active"] is True
            # Delayed measurements must not replace committed qcmd.
            assert not np.allclose(receipt["measured_qpos"], receipt["command_anchor_qpos"])

        for _ in range(30):
            for backend in backends.values():
                _state(backend)
            repeated = run_policy._hold_active_rtc_gap(
                reason="rtc_producer_not_eligible", send_actions=True,
                arm_backends=backends, position_deploy_executor=executor,
            )
            for side in backends:
                assert repeated[side]["status"] == "already_active"
                assert repeated[side]["target_sequence"] == receipts[side]["target_sequence"]
                dynamics = _state(backends[side])["trajectory_planner"]
                assert np.all(np.abs(dynamics["qd_command_rad_s"]) <= np.asarray(dynamics["max_velocity_rad_s"]) + 1e-8)
                assert np.all(np.abs(dynamics["qdd_command_rad_s2"]) <= np.asarray(dynamics["max_acceleration_rad_s2"]) + 1e-8)
            assert executor.snapshot()["clear_count"] == 1
            time.sleep(0.005)

        for side, (backend, arm) in pairs.items():
            held = _wait(backend, lambda state: np.max(np.abs(state["trajectory_planner"]["qd_command_rad_s"])) < 1e-8)
            assert held["continuous_hold"] is True
            assert arm.stop_count == 0
            assert len(arm.sent) > 10
            steps = np.abs(np.diff(np.asarray(arm.sent), axis=0))
            assert np.all(steps <= backend._base_max_joint_step + 1e-8)
            # The obsolete 0.30-rad target must not run to completion in Hold.
            assert np.max(np.abs(arm.sent[-1])) < 0.20
            np.testing.assert_allclose(arm.sent[-1], arm.sent[-5], atol=1e-7)
            sequence = held["target_sequence"]
            with pytest.raises(ValueError):
                backend.send_command(_command(side, float("nan")))
            assert _state(backend)["quick_stop_active"] is True
            assert _state(backend)["target_sequence"] == sequence
            backend.send_command(_command(side, 0.08))
            resumed = _wait(backend, lambda state: not state["quick_stop_active"])
            assert resumed["target_sequence"] > sequence
            _wait(backend, lambda state: state["accepted_send_count"] > held["accepted_send_count"] + 3)
            assert arm.stop_count == 0


@pytest.mark.parametrize("fault", ("stale", "future", "nan", "read_failed", "device_fault", "latched", "stream_fault", "worker_stopped"))
def test_position_inference_hold_rejects_invalid_feedback_or_worker(fault):
    with _position_backend() as (backend, arm):
        with backend._state_lock:
            if fault == "stale":
                backend._measured_monotonic -= 1.0
            elif fault == "future":
                backend._measured_monotonic += 1.0
            elif fault == "nan":
                backend._measured_qpos[0] = float("nan")
            elif fault == "read_failed":
                backend._last_hardware_read_success = False
            elif fault == "device_fault":
                backend._last_hardware_arm_err = 1
            elif fault == "latched":
                backend._emergency_stopped = True
            elif fault == "stream_fault":
                backend._stream_fault = "fake fault"
            else:
                backend._stream_stop_event.set()
        with pytest.raises(RuntimeError, match="inference hold"):
            backend.request_policy_inference_soft_hold("rtc_active_latency_warmup")
        assert arm.sent == []


def test_position_rtc_partial_failure_requests_stronger_hold_on_both_sides(monkeypatch):
    with ExitStack() as stack:
        left, _ = stack.enter_context(_position_backend("left"))
        right, _ = stack.enter_context(_position_backend("right"))
        right._last_hardware_read_success = False
        held = []
        for side, backend in (("left", left), ("right", right)):
            original = backend.hold_motion
            def record(reason, *, side=side, original=original):
                held.append(side)
                original(reason)
            monkeypatch.setattr(backend, "hold_motion", record)
        with pytest.raises(RuntimeError, match="fresh, finite"):
            run_policy._hold_active_rtc_gap(
                reason="rtc_active_latency_warmup", send_actions=True,
                arm_backends={"left": left, "right": right},
                position_deploy_executor=_Executor(),
            )
        assert held == ["left", "right"]
        assert left._stream_quick_stop_active and right._stream_quick_stop_active


@pytest.mark.parametrize("planner", (POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED, POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED))
def test_position_inference_hold_supports_existing_dynamics_limited_planners(planner):
    with _position_backend(planner=planner) as (backend, _):
        receipt = backend.request_policy_inference_soft_hold("rtc_active_latency_warmup")
        assert receipt["active"] is True
        assert _state(backend)["quick_stop_active"] is True


def test_position_inference_hold_dry_run_does_not_create_worker_or_send():
    backend = RM65ArmBackend(RM65ArmBackendConfig(arm_ip="0.0.0.0", side="left", connect_hardware=False))
    assert backend.request_policy_inference_soft_hold()["status"] == "skipped"
    assert backend.arm is None and backend._stream_thread is None
    backend.close()
