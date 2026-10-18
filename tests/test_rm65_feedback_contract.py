from __future__ import annotations

import threading
from unittest import mock

import numpy as np
import pytest

from robot_motion.action import (
    ActionExecutor,
    ActionExecutorConfig,
    RigCommandDispatcher,
    RobotAction,
)
from robot_motion.deploy.schema import STATE_ARM_QPOS_SLICES
from robot_motion.runtime import (
    BackendDispatchError,
    BimanualRig,
    EffectorCommand,
    RigDispatchError,
    TeleopCommand,
)
from robot_motion.safety import MotionSafetyConfig
from teleop_runtime.backends.arms import rm65 as rm65_module
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.collection.arm_state_polling import (
    ArmHardwareStatePoller,
    ArmHardwareStatePollerConfig,
)
from teleop_runtime.collection.build_collect_components import CollectorComponents
from teleop_runtime.collection.sample_assembler import (
    CollectorSampleBuilder,
    CollectorSampleBuilderConfig,
)
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.session import (
    SessionStepResult,
    TeleopSession,
    TeleopSessionConfig,
)


def _snapshot_backend() -> RM65ArmBackend:
    backend = RM65ArmBackend(RM65ArmBackendConfig(
        arm_ip="0.0.0.0", side="left", connect_hardware=False,
        initial_qpos=(0.0,) * 6, singularity_protection_enabled=False,
    ))
    with backend._state_lock:
        backend._record_command_attempt_locked(
            np.zeros(6), timestamp=101.0, monotonic_time=11.0,
        )
        backend._record_command_accepted_locked(
            np.zeros(6), timestamp=101.0, monotonic_time=11.0,
        )
    return backend


def test_get_state_does_not_mix_accepted_qpos_with_a_newer_receipt(monkeypatch) -> None:
    backend = _snapshot_backend()
    with backend._state_lock:
        backend._last_command = np.full(7, 0.2)
        backend._last_command_timestamp = 101.0
        backend._last_send_result_code = 0
        backend._last_send_result_text = "old-result"
        backend._last_effective_control_dt = 0.02
        backend._last_effective_step_scale = 0.8
    original = backend._active_tcp_pose_flange_m_rad

    def publish_after_capture():
        with backend._state_lock:
            backend._record_command_attempt_locked(
                np.full(6, 0.01), timestamp=102.0, monotonic_time=12.0,
            )
            backend._record_command_accepted_locked(
                np.full(6, 0.01), timestamp=102.0, monotonic_time=12.0,
            )
            backend._last_command = np.full(7, 0.3)
            backend._last_command_timestamp = 102.0
            backend._last_send_result_code = 7
            backend._last_send_result_text = "new-result"
            backend._last_effective_control_dt = 0.04
            backend._last_effective_step_scale = 0.4
        return original()

    monkeypatch.setattr(backend, "_active_tcp_pose_flange_m_rad", publish_after_capture)
    snapshot = backend.get_state(include_derived_pose=False, include_timing_percentiles=False)
    for family in ("attempted", "accepted"):
        np.testing.assert_array_equal(snapshot[f"last_{family}_qpos"], np.zeros(6))
        assert snapshot[f"last_{family}_timestamp"] == 101.0
        assert snapshot[f"last_{family}_timestamp_monotonic"] == 11.0
        assert snapshot[f"last_{family}_sequence"] == 1
    np.testing.assert_array_equal(snapshot["last_command"], np.full(7, 0.2))
    assert snapshot["last_command_timestamp"] == 101.0
    assert snapshot["send_result_code"] == 0
    assert snapshot["send_result_text"] == "old-result"
    assert snapshot["effective_control_dt_sec"] == 0.02
    assert snapshot["effective_step_scale"] == 0.8
    np.testing.assert_array_equal(backend._last_accepted_qpos, np.full(6, 0.01))
    assert backend._last_accepted_sequence == 2


def test_get_state_payload_copy_does_not_hold_the_command_publication_lock() -> None:
    backend = _snapshot_backend()
    copy_started, published = threading.Event(), threading.Event()
    errors = []

    class CopyBarrier(np.ndarray):
        def copy(self, *args, **kwargs):
            copy_started.set()
            assert published.wait(1.0), "state payload copy held the command publication lock"
            return np.asarray(self).copy(*args, **kwargs)

    backend._current_qpos = np.zeros(6).view(CopyBarrier)

    def publish():
        if not copy_started.wait(2.0):
            errors.append("state copy was not reached")
            return
        with backend._state_lock:
            backend._record_command_accepted_locked(
                np.full(6, 0.01), timestamp=102.0, monotonic_time=12.0,
            )
        published.set()

    worker = threading.Thread(target=publish)
    worker.start()
    try:
        snapshot = backend.get_state(include_derived_pose=False, include_timing_percentiles=False)
    finally:
        worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert not errors
    np.testing.assert_array_equal(snapshot["current_qpos"], np.zeros(6))
    np.testing.assert_array_equal(snapshot["last_accepted_qpos"], np.zeros(6))
    assert snapshot["last_accepted_sequence"] == 1
    assert backend._last_accepted_sequence == 2


@pytest.mark.parametrize("attribute,key,dimension", [
    ("_current_qpos", "current_qpos", 6),
    ("_last_attempted_qpos", "last_attempted_qpos", 6),
    ("_last_accepted_qpos", "last_accepted_qpos", 6),
    ("_measured_qpos", "measured_qpos", 6),
    ("_last_hardware_qpos", "hardware_qpos", 6),
    ("_last_target_qpos", "last_target_qpos", 6),
    ("_last_command", "last_command", 7),
    ("_last_hardware_pose_6d", "hardware_pose_6d", 6),
])
def test_get_state_array_payload_stays_detached(attribute, key, dimension) -> None:
    backend = _snapshot_backend()
    original = np.arange(dimension, dtype=float) / 100
    with backend._state_lock:
        setattr(backend, attribute, original.copy())
    snapshot = backend.get_state(include_derived_pose=False, include_timing_percentiles=False)
    result = snapshot[key]
    np.testing.assert_array_equal(result, original)
    assert not np.shares_memory(result, getattr(backend, attribute))
    result[:] = 7.0
    np.testing.assert_array_equal(getattr(backend, attribute), original)


class _FakeRM65Arm:
    def __init__(self, state_results: list[tuple] | None = None, send_result: object = 0):
        self.state_results = list(state_results or [])
        self.send_result = send_result
        self.uninit_calls = 0
        self.socket_close_calls = 0
        self.sent_joint_deg: list[np.ndarray] = []

    def Get_Current_Arm_State(self):
        if self.state_results:
            if len(self.state_results) == 1:
                return self.state_results[0]
            return self.state_results.pop(0)
        return 7, None, None, 0, 0

    def Movej_CANFD(self, joint_deg, **_kwargs):
        self.sent_joint_deg.append(np.asarray(joint_deg, dtype=float).copy())
        return self.send_result

    def RM_API_UnInit(self):
        self.uninit_calls += 1

    def Arm_Socket_Close(self):
        self.socket_close_calls += 1


def _hardware_backend(
    fake_arm: _FakeRM65Arm,
    *,
    initial_qpos: tuple[float, ...] | None = None,
    verified_qpos: tuple[float, ...] | None = None,
) -> RM65ArmBackend:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.1",
            side="left",
            connect_hardware=True,
            initial_qpos=initial_qpos,
            verified_hardware_initial_qpos=verified_qpos,
            hardware_read_retries=0,
            singularity_protection_enabled=False,
        )
    )
    backend._auto_load_active_tool_profile = lambda: None
    backend._configure_singularity_protection = lambda: None
    return backend


def _start_with_fake_sdk(backend: RM65ArmBackend, fake_arm: _FakeRM65Arm) -> None:
    with mock.patch.object(
        rm65_module,
        "_load_rm65_sdk",
        return_value=(lambda _model, _ip: fake_arm, object()),
    ):
        backend.start()


def test_rm65_start_failure_does_not_invent_zero_qpos_or_report_ready() -> None:
    fake_arm = _FakeRM65Arm([(7, None, None, 0, 0)])
    backend = _hardware_backend(fake_arm, initial_qpos=(0.0,) * 6)

    with pytest.raises(RuntimeError, match="trusted 6D qpos"):
        _start_with_fake_sdk(backend, fake_arm)

    state = backend.get_state()
    assert backend.arm is None
    assert backend.is_ready() is False
    assert state["current_qpos"] is None
    assert state["hardware_ready"] is False
    assert state["hardware_ready_source"] == "startup_failed"
    assert fake_arm.uninit_calls == 1
    assert fake_arm.socket_close_calls == 1


def test_rm65_start_allows_only_explicit_verified_initial_qpos_fallback() -> None:
    verified = (0.1, -0.1, 0.2, -0.2, 0.1, -0.1)
    fake_arm = _FakeRM65Arm([(7, None, None, 0, 0)])
    backend = _hardware_backend(fake_arm, verified_qpos=verified)

    _start_with_fake_sdk(backend, fake_arm)

    state = backend.get_state()
    assert backend.is_ready() is True
    np.testing.assert_allclose(state["current_qpos"], verified)
    assert state["hardware_read_success"] is False
    assert state["hardware_ready_source"] == "verified_hardware_initial_qpos"
    backend.close()


def test_rm65_local_observation_clips_tiny_endpoint_quantization_without_relaxing_transport() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.1",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            singularity_protection_enabled=False,
        )
    )
    observed = np.zeros(6, dtype=float)
    observed[2] = 2.355008

    backend.set_local_qpos(observed)

    state = backend.get_state()
    assert state["current_qpos"][2] == pytest.approx(2.355)
    normalization = state["local_qpos_normalization"]
    assert normalization["status"] == "clipped"
    assert normalization["clipped_joint_indices"] == [3]
    assert normalization["observed_qpos_rad"][2] == pytest.approx(observed[2])

    too_far = observed.copy()
    too_far[2] = 2.355 + np.deg2rad(0.02)
    with pytest.raises(ValueError, match="local-observation tolerance"):
        backend.set_local_qpos(too_far)

    # The final transport gate intentionally remains exact: this observation
    # tolerance must never turn an out-of-range qpos into a sendable command.
    with pytest.raises(ValueError, match="absolute limits"):
        backend._validate_final_joint_target(observed, label="transport target")


def test_rm65_transport_absorbs_endpoint_roundoff_into_an_inward_margin() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="192.0.2.1",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            singularity_protection_enabled=False,
        )
    )
    lower, upper = backend._joint_position_limits_rad()
    tolerance = rm65_module._RM65_ABSOLUTE_LIMIT_TOLERANCE_RAD
    margin = rm65_module._RM65_ABSOLUTE_LIMIT_DISPATCH_MARGIN_RAD
    assert tolerance == pytest.approx(2.0e-8)
    assert margin == pytest.approx(1.0e-8)

    upper_roundoff = np.zeros(6, dtype=float)
    upper_roundoff[5] = upper[5] + 0.5 * tolerance
    normalized_upper = backend._validate_final_joint_target(
        upper_roundoff,
        label="transport target",
    )
    assert normalized_upper[5] == pytest.approx(upper[5] - margin, abs=1e-15)
    assert normalized_upper[5] < upper[5]

    lower_roundoff = np.zeros(6, dtype=float)
    lower_roundoff[5] = lower[5] - 0.5 * tolerance
    normalized_lower = backend._validate_final_joint_target(
        lower_roundoff,
        label="transport target",
    )
    assert normalized_lower[5] == pytest.approx(lower[5] + margin, abs=1e-15)
    assert normalized_lower[5] > lower[5]

    real_violation = upper_roundoff.copy()
    real_violation[5] = upper[5] + 2.0 * tolerance
    with pytest.raises(ValueError, match="absolute limits"):
        backend._validate_final_joint_target(
            real_violation,
            label="transport target",
        )


class _LifecycleBackend:
    def __init__(self, *, fail_start: bool = False):
        self.fail_start = fail_start
        self.start_calls = 0
        self.close_calls = 0

    def start(self):
        self.start_calls += 1
        if self.fail_start:
            raise RuntimeError("start failed")

    def close(self):
        self.close_calls += 1


def test_rig_start_rolls_back_all_attempted_backends_after_partial_failure() -> None:
    left = _LifecycleBackend()
    right = _LifecycleBackend(fail_start=True)
    rig = BimanualRig({"left_arm": left, "right_arm": right})

    with pytest.raises(RuntimeError, match="start failed"):
        rig.start()

    assert left.start_calls == 1
    assert right.start_calls == 1
    assert left.close_calls == 1
    assert right.close_calls == 1


def _direct_backend(send_result: object) -> RM65ArmBackend:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=(0.0,) * 6,
            singularity_protection_enabled=False,
        )
    )
    backend.arm = _FakeRM65Arm(send_result=send_result)
    return backend


def _joint_command(value: float = 0.1) -> EffectorCommand:
    return EffectorCommand(
        effector_name="left_arm",
        side="left",
        command_type="joint_vector",
        value=np.full(6, value, dtype=float),
    )


def test_rm65_nonzero_direct_send_raises_typed_error_and_is_not_accepted() -> None:
    backend = _direct_backend(7)

    with pytest.raises(BackendDispatchError) as caught:
        backend.send_command(_joint_command())

    assert caught.value.receipt.result_code == 7
    assert caught.value.receipt.accepted is False
    state = backend.get_state()
    assert state["last_attempted_sequence"] == 1
    assert state["last_accepted_sequence"] == 0
    assert state["last_target_qpos"] is None
    assert state["last_command"] is None
    assert state["send_result_code"] == 7


def test_rm65_direct_send_error_propagates_through_rig_and_action_executor() -> None:
    backend = _direct_backend(7)
    rig = BimanualRig({"left_arm": backend})
    command = TeleopCommand(commands={"left_arm": _joint_command()})

    with pytest.raises(BackendDispatchError):
        rig.send_commands(command)

    executor = ActionExecutor(
        dispatcher=RigCommandDispatcher(BimanualRig({"left_arm": _direct_backend(7)})),
        config=ActionExecutorConfig(raise_on_error=False),
    )
    result = executor.execute(
        RobotAction.bimanual_qpos(
            left_arm=np.full(6, 0.1),
            active_effectors=("left_arm",),
        )
    )

    assert result.sent is False
    assert result.faults and result.faults[0].startswith("BackendDispatchError:")


def test_rm65_direct_send_error_remains_typed_with_group_fail_stop() -> None:
    rig = BimanualRig(
        {"left_arm": _direct_backend(7)},
        fail_stop_on_dispatch_error=True,
    )

    with pytest.raises(RigDispatchError) as caught:
        rig.send_commands(TeleopCommand(commands={"left_arm": _joint_command()}))

    backend_error = caught.value.failures["left_arm"]
    assert isinstance(backend_error, BackendDispatchError)
    assert backend_error.receipt.result_code == 7


class _StaticRetargeter:
    name = "static_test_retargeter"

    def retarget(self, _observation, _calibration=None):
        return TeleopCommand(commands={"left_arm": _joint_command()})


def test_rm65_direct_send_error_propagates_through_session_before_dispatched_state() -> None:
    backend = _direct_backend(7)
    session = TeleopSession(
        TeleopSessionConfig(record_enabled=False),
        retargeters=[_StaticRetargeter()],
        rig=BimanualRig({"left_arm": backend}),
        safety_config=MotionSafetyConfig(minimum_control_hz=0.0),
    )
    session._started = True
    session.state_machine.transition(TeleopRunState.ARMED, "unit test")

    with pytest.raises(BackendDispatchError):
        session.run_step(now=10.0)

    assert session.state_machine.state == TeleopRunState.ARMED
    assert backend.get_state()["last_accepted_sequence"] == 0


def test_failed_feedback_refresh_keeps_last_success_timestamp_and_sequence() -> None:
    fake_arm = _FakeRM65Arm(
        [
            (0, [10.0] * 6, [0.0] * 6, 0, 0),
            (7, None, None, 0, 0),
        ]
    )
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=(0.0,) * 6,
            singularity_protection_enabled=False,
        )
    )
    backend.arm = fake_arm

    with mock.patch.object(rm65_module.time, "time", side_effect=[100.0, 200.0]), mock.patch.object(
        rm65_module.time,
        "monotonic",
        side_effect=[10.0, 10.05, 10.1, 20.0, 20.05, 20.1],
    ):
        first = backend.refresh_state_from_hardware(hardware_read_retries=0)
        failed = backend.refresh_state_from_hardware(hardware_read_retries=0)

    assert first["hardware_refresh_time"] == 100.0
    assert first["measured_sequence"] == 1
    assert failed["hardware_refresh_time"] == 100.0
    assert failed["measured_timestamp"] == 100.0
    assert failed["measured_sequence"] == 1
    assert failed["hardware_read_time"] == 200.0
    assert failed["hardware_read_sequence"] == 2
    assert failed["hardware_read_success"] is False
    phase_timing = failed["hardware_read_phase_timing"]
    assert set(phase_timing) == {
        "total_wall_ms",
        "total_thread_cpu_ms",
        "hardware_read_lock_wait_wall_ms",
        "hardware_read_lock_wait_thread_cpu_ms",
        "sdk_call_wall_ms",
        "sdk_call_thread_cpu_ms",
        "parse_validation_wall_ms",
        "parse_validation_thread_cpu_ms",
        "retry_sleep_wall_ms",
        "attempt_count",
    }
    assert phase_timing["attempt_count"] == 1.0
    assert all(value >= 0.0 for value in phase_timing.values())
    assert phase_timing["total_wall_ms"] >= phase_timing["sdk_call_wall_ms"]
    np.testing.assert_allclose(failed["hardware_qpos"], np.deg2rad([10.0] * 6))


class _FailedRefreshBackend:
    def refresh_state_from_hardware(self):
        return {
            "hardware_qpos": np.full(6, 0.4),
            "current_qpos": np.full(6, 0.8),
            "hardware_measurement_timestamp": 99.0,
            "hardware_measurement_sequence": 4,
            "hardware_measurement_valid": True,
            "hardware_read_success": False,
            "hardware_result_code": 7,
            "hardware_result_text": "7",
        }

    def compute_tcp_pose(self, *, qpos=None):
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = np.asarray(qpos, dtype=float)[:3]
        return pose


class _DivergentCommandAndMeasuredBackend:
    def __init__(self) -> None:
        self.hardware_qpos = np.asarray([0.2, 0.3, 0.4, 0.0, 0.0, 0.0])
        self.command_pose = np.eye(4, dtype=float)
        self.command_pose[:3, 3] = [9.0, 8.0, 7.0]

    def refresh_state_from_hardware(self):
        return {
            "hardware_qpos": self.hardware_qpos.copy(),
            "current_tcp_pose": self.command_pose.copy(),
            "hardware_measurement_timestamp": 99.0,
            "hardware_measurement_sequence": 4,
            "hardware_measurement_valid": True,
            "hardware_read_success": True,
            "hardware_result_code": 0,
            "hardware_result_text": "ok",
        }

    def compute_tcp_pose(self, *, qpos=None):
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = np.asarray(qpos, dtype=float)[:3]
        return pose


def test_poller_actual_tcp_pose_uses_hardware_qpos_fk_not_command_pose() -> None:
    backend = _DivergentCommandAndMeasuredBackend()
    poller = ArmHardwareStatePoller(
        backends_by_side={"left": backend},
        config=ArmHardwareStatePollerConfig(stale_after_sec=0.2),
    )

    poller._poll_once("left", backend)
    snapshot = poller.latest_at_or_before(100.0)["left"]
    expected = backend.compute_tcp_pose(qpos=backend.hardware_qpos)

    np.testing.assert_allclose(snapshot["actual_tcp_pose"], expected)
    np.testing.assert_allclose(snapshot["wrist_camera_pose_base"], expected)
    assert not np.allclose(snapshot["actual_tcp_pose"], backend.command_pose)
    assert snapshot["actual_tcp_pose_source"] == "hardware_qpos_fk"


def test_poller_does_not_retimestamp_cached_qpos_after_failed_read() -> None:
    backend = _FailedRefreshBackend()
    poller = ArmHardwareStatePoller(
        backends_by_side={"left": backend},
        config=ArmHardwareStatePollerConfig(stale_after_sec=0.2),
    )

    with mock.patch(
        "teleop_runtime.collection.arm_state_polling.time.time",
        side_effect=[100.4, 100.5],
    ):
        poller._poll_once("left", backend)
    snapshot = poller.latest_at_or_before(101.0)["left"]

    assert snapshot["actual_state_event_timestamp"] == 100.5
    assert snapshot["actual_state_timestamp"] is None
    assert snapshot["actual_hardware_refresh_time"] == 99.0
    assert snapshot["actual_state_read_success"] is False
    assert snapshot["actual_state_valid"] is False
    assert snapshot["actual_state_stale"] is True
    assert snapshot["actual_qpos"] is None
    assert snapshot["actual_tcp_pose"] is None


class _CommandedArmBackend:
    def get_state(self):
        return {
            "current_qpos": np.zeros(6),
            "current_tcp_pose": np.eye(4),
        }

    def compute_tcp_pose(self, *, qpos=None):
        pose = np.eye(4)
        if qpos is not None:
            pose[:3, 3] = np.asarray(qpos, dtype=float)[:3]
        return pose


def test_assembler_rejects_failed_cached_feedback_even_when_qpos_is_present() -> None:
    builder = CollectorSampleBuilder(
        CollectorSampleBuilderConfig(
            active_sides=("left",),
            prefer_causal_measured_arm_state=True,
        ),
        arm_state_metadata_provider=lambda _components, sample_timestamp: {
            "left": {
                "actual_qpos": np.full(6, 0.7),
                "actual_hardware_qpos": np.full(6, 0.7),
                "actual_hardware_pose_6d": np.full(6, 0.2),
                "actual_tcp_pose": np.eye(4),
                "actual_state_timestamp": sample_timestamp - 0.01,
                "actual_state_read_success": False,
                "actual_state_valid": False,
                "actual_state_stale": False,
                "actual_state_sequence": 8,
            }
        },
    )

    sample = builder(
        SessionStepResult(
            timestamp=10.0,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
        ),
        CollectorComponents(
            session=None,
            backends={"left_arm": _CommandedArmBackend()},
        ),
    )

    assert sample is not None
    np.testing.assert_allclose(
        sample.observation_state[STATE_ARM_QPOS_SLICES["left"]],
        np.zeros(6),
    )
    state = sample.arm_states["left"]
    assert state["runtime_arm_state_source"] == "last_commanded_action"
    assert state["causal_measured_state_eligible"] is False
    assert state["causal_measured_state_rejection_reason"] == "hardware_read_failed"
    assert state["actual_qpos"] is None
    assert state["actual_tcp_pose"] is None
    assert state["actual_hardware_qpos"] is None
    assert state["actual_hardware_pose_6d"] is None


def test_assembler_accepts_successful_valid_fresh_sequenced_feedback() -> None:
    measured = np.full(6, 0.25)
    builder = CollectorSampleBuilder(
        CollectorSampleBuilderConfig(
            active_sides=("left",),
            prefer_causal_measured_arm_state=True,
        ),
        arm_state_metadata_provider=lambda _components, sample_timestamp: {
            "left": {
                "actual_qpos": measured.copy(),
                "actual_tcp_pose": np.eye(4),
                "actual_state_timestamp": sample_timestamp - 0.01,
                "actual_state_read_success": True,
                "actual_state_valid": True,
                "actual_state_stale": False,
                "actual_state_sequence": 9,
            }
        },
    )

    sample = builder(
        SessionStepResult(
            timestamp=10.0,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
        ),
        CollectorComponents(
            session=None,
            backends={"left_arm": _CommandedArmBackend()},
        ),
    )

    assert sample is not None
    np.testing.assert_allclose(
        sample.observation_state[STATE_ARM_QPOS_SLICES["left"]],
        measured,
    )
    state = sample.arm_states["left"]
    assert state["runtime_arm_state_source"] == "causal_measured_hardware"
    assert state["causal_measured_state_eligible"] is True
    assert state["causal_measured_state_rejection_reason"] == "accepted"


@pytest.mark.parametrize("publisher,attribute,key", [
    ("_record_command_attempt_locked", "_last_attempted_qpos", "last_attempted_qpos"),
    ("_record_command_accepted_locked", "_last_accepted_qpos", "last_accepted_qpos"),
    ("_record_measured_qpos_locked", "_measured_qpos", "measured_qpos"),
])
def test_get_state_publisher_owns_input_and_preserves_replaced_binding(
    publisher, attribute, key,
) -> None:
    backend = _snapshot_backend()
    expected = np.arange(6, dtype=float) / 1000
    caller_value = expected.copy()
    with backend._state_lock:
        getattr(backend, publisher)(caller_value, timestamp=201.0, monotonic_time=21.0)
        old_binding = getattr(backend, attribute)
    assert not np.shares_memory(old_binding, caller_value)
    caller_value[:] = 7.0
    snapshot = backend.get_state(include_derived_pose=False, include_timing_percentiles=False)
    np.testing.assert_array_equal(snapshot[key], expected)
    with backend._state_lock:
        getattr(backend, publisher)(np.full(6, 0.02), timestamp=202.0, monotonic_time=22.0)
    np.testing.assert_array_equal(old_binding, expected)
    np.testing.assert_array_equal(snapshot[key], expected)
    assert getattr(backend, attribute) is not old_binding


def test_get_state_real_command_value_stays_detached_after_publication_and_replacement() -> None:
    backend = _direct_backend(0)
    command = _joint_command(0.01)
    backend.send_command(command)
    snapshot = backend.get_state(include_derived_pose=False, include_timing_percentiles=False)
    stored = backend._last_command
    result = snapshot["last_command"]
    assert isinstance(stored, EffectorCommand)
    assert isinstance(result, EffectorCommand)
    assert result is not stored
    assert not np.shares_memory(stored.value, command.value)
    assert not np.shares_memory(result.value, stored.value)
    command.value[:] = 7.0
    np.testing.assert_array_equal(stored.value, np.full(6, 0.01))
    backend.send_command(_joint_command(0.02))
    np.testing.assert_array_equal(result.value, np.full(6, 0.01))
    result.value[:] = 9.0
    np.testing.assert_array_equal(backend._last_command.value, np.full(6, 0.02))


@pytest.mark.parametrize("include_background_streaming", [False, True])
@pytest.mark.parametrize("include_timing_percentiles", [False, True])
def test_get_state_preserves_optional_background_snapshot_work(
    monkeypatch, include_background_streaming, include_timing_percentiles,
) -> None:
    backend = _snapshot_backend()
    calls = []

    def background():
        calls.append("background")
        return {"snapshot": "marker"}

    def timings():
        calls.append("timings")
        return {"samples": 3}

    monkeypatch.setattr(backend, "_background_stream_snapshot_locked", background)
    monkeypatch.setattr(backend, "_background_stream_timing_percentiles_snapshot", timings)
    snapshot = backend.get_state(
        include_derived_pose=False,
        include_background_streaming=include_background_streaming,
        include_timing_percentiles=include_timing_percentiles,
    )
    expected = {}
    expected_calls = []
    if include_background_streaming:
        expected["snapshot"] = "marker"
        expected_calls.append("background")
        if include_timing_percentiles:
            expected["timing_percentiles"] = {"samples": 3}
            expected_calls.append("timings")
    assert snapshot["background_streaming"] == expected
    assert calls == expected_calls


def test_get_state_keeps_legacy_sdk_pose_copy_inside_publication_lock() -> None:
    backend = _snapshot_backend()
    acquired_by_other_thread = []
    threads = []

    class LockProbe(np.ndarray):
        def copy(self, *args, **kwargs):
            def try_lock():
                acquired = backend._state_lock.acquire(blocking=False)
                acquired_by_other_thread.append(acquired)
                if acquired:
                    backend._state_lock.release()

            worker = threading.Thread(target=try_lock)
            threads.append(worker)
            worker.start()
            worker.join(timeout=2.0)
            assert not worker.is_alive()
            return np.asarray(self).copy(*args, **kwargs)

    with backend._state_lock:
        backend._last_hardware_pose_6d = np.arange(6, dtype=float).view(LockProbe)
    try:
        snapshot = backend.get_state(include_derived_pose=False, include_timing_percentiles=False)
    finally:
        for worker in threads:
            worker.join(timeout=2.0)
    assert acquired_by_other_thread == [False]
    np.testing.assert_array_equal(snapshot["hardware_pose_6d"], np.arange(6, dtype=float))
