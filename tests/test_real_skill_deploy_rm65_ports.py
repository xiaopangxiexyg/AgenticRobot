from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from agent_gateway.profiles.real.exclusive_factory import (
    assemble_exclusive_rm65_agent_primitive_deployment,
)
from robot_motion.runtime import (
    CommandDispatchReceipt,
    EffectorCommand,
    TeleopCommand,
    TeleopObservation,
)
from teleop_runtime.interfaces import EffectorBackend
from teleop_runtime.skill_deploy import (
    REAL_AGENT_MOTION_COMMAND_METADATA_KEY,
    ExclusiveRM65SkillDeployPorts,
    ExclusiveRM65SkillPortProfile,
    build_exclusive_rm65_skill_position_force_guard,
)
from teleop_runtime.skill_deploy.rm65_ports import (
    _next_periodic_measurement_deadline,
)


class _Backend(EffectorBackend):
    def __init__(
        self,
        side: str,
        *,
        measured_at: float,
        accepted: bool = True,
        close_failures: int = 0,
        hold_failures: int = 0,
        events: list[str] | None = None,
    ) -> None:
        self.side = side
        self.measured_at = measured_at
        self.accepted = bool(accepted)
        self.target_active = False
        self.sent: list[EffectorCommand] = []
        self.holds: list[str] = []
        self.starts = 0
        self.closes = 0
        self.close_failures = int(close_failures)
        self.hold_failures = int(hold_failures)
        self.emergency_stops = 0
        self.events = events if events is not None else []

    @property
    def name(self) -> str:
        return f"{self.side}_arm"

    def is_ready(self) -> bool:
        return True

    def start(self) -> None:
        self.starts += 1
        self.events.append(f"backend_start:{self.side}")

    def close(self) -> None:
        self.closes += 1
        self.events.append(f"backend_close:{self.side}")
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError(f"{self.side} close failed")

    def refresh_state_from_hardware(self):
        return self.get_state()

    def get_state(self):
        return {
            "hardware_ready": True,
            "hardware_qpos": np.full(6, 0.1 if self.side == "left" else 0.2),
            "hardware_measurement_timestamp": self.measured_at,
            "hardware_measurement_sequence": 4 if self.side == "left" else 5,
            "hardware_measurement_valid": True,
            "hardware_read_success": True,
            "hardware_read_phase_timing": {
                "total_wall_ms": 8.0,
                "total_thread_cpu_ms": 0.5,
                "hardware_read_lock_wait_wall_ms": 0.1,
                "sdk_call_wall_ms": 7.0,
                "sdk_call_thread_cpu_ms": 0.3,
                "parse_validation_wall_ms": 0.2,
            },
            "emergency_stopped": False,
            "send_result_code": 0,
            "background_streaming": {
                "enabled": True,
                "healthy": True,
                "health_reasons": [],
                "thread_alive": True,
                "target_active": self.target_active,
                "target_stale": False,
                "fault": None,
            },
        }

    def compute_tcp_pose(self, *, qpos=None):
        pose = np.eye(4)
        pose[0, 3] = float(np.asarray(qpos, dtype=float).reshape(6)[0])
        return pose

    def send_command(self, command: EffectorCommand):
        self.sent.append(command.copy())
        self.target_active = self.accepted
        return CommandDispatchReceipt(
            effector_name=self.name,
            accepted=self.accepted,
            operation="queue_background_joint_target",
            status="queued" if self.accepted else "rejected",
            result_code=0 if self.accepted else 2,
        )

    def hold_motion(self, reason: str = "manual hold") -> None:
        self.holds.append(reason)
        self.events.append(f"backend_hold:{self.side}")
        self.target_active = False
        if self.hold_failures:
            self.hold_failures -= 1
            raise RuntimeError(f"{self.side} hold failed")

    def emergency_stop(self) -> None:
        self.emergency_stops += 1
        self.target_active = False

    def read_force_data(self):
        return {
            "sources": {
                "raw": np.zeros(6),
                "zero": np.zeros(6),
                "work": np.zeros(6),
                "tool": np.zeros(6),
            },
            "monotonic_time_sec": time.monotonic(),
            "wall_time_sec": time.time(),
        }


class _TargetBuilder:
    def __init__(self) -> None:
        self.syncs: list[tuple[np.ndarray, np.ndarray, bool]] = []

    def set_current_qpos_pair(
        self,
        *,
        left_qpos,
        right_qpos,
        reset_tracking=True,
    ) -> None:
        self.syncs.append(
            (
                np.asarray(left_qpos, dtype=float).copy(),
                np.asarray(right_qpos, dtype=float).copy(),
                bool(reset_tracking),
            )
        )


class _ForceGuard:
    def __init__(
        self,
        *,
        ready_on_start: bool = True,
        close_failures: int = 0,
        events: list[str] | None = None,
    ) -> None:
        self.motion_ready = False
        self.fault = None
        self.starts = 0
        self.closes = 0
        self.close_failures = int(close_failures)
        self.ready_on_start = bool(ready_on_start)
        self.events = events if events is not None else []

    def start(self, *, arm_stale_watchdog=True):
        self.starts += 1
        self.events.append("force_guard_start")
        self.motion_ready = bool(arm_stale_watchdog and self.ready_on_start)
        return {"motion_ready": self.motion_ready}

    def close(self) -> None:
        self.closes += 1
        self.events.append("force_guard_close")
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("force guard close failed")
        self.motion_ready = False


def _ports(
    *,
    accepted: bool = True,
    age_sec: float = 0.01,
    force_guard_ready: bool = True,
    force_guard_close_failures: int = 0,
    backend_close_failures: dict[str, int] | None = None,
    backend_hold_failures: dict[str, int] | None = None,
    background_measurement_sampling: bool = False,
    state_refresh_hz: float = 30.0,
    max_measurement_age_sec: float = 0.2,
):
    now = time.time()
    events: list[str] = []
    backends = {
        side: _Backend(
            side,
            measured_at=now - age_sec,
            accepted=accepted,
            close_failures=(backend_close_failures or {}).get(side, 0),
            hold_failures=(backend_hold_failures or {}).get(side, 0),
            events=events,
        )
        for side in ("left", "right")
    }
    target_builder = _TargetBuilder()
    force_guard = _ForceGuard(
        ready_on_start=force_guard_ready,
        close_failures=force_guard_close_failures,
        events=events,
    )
    ports = ExclusiveRM65SkillDeployPorts(
        arm_backends=backends,
        target_builder=target_builder,
        position_force_guard=force_guard,
        profile=ExclusiveRM65SkillPortProfile(
            state_refresh_hz=state_refresh_hz,
            max_measurement_age_sec=max_measurement_age_sec,
            background_measurement_sampling=background_measurement_sampling,
        ),
        clock=lambda: now,
    )
    return ports, backends, target_builder


def _command() -> TeleopCommand:
    command = TeleopCommand(timestamp=time.time())
    command.metadata[REAL_AGENT_MOTION_COMMAND_METADATA_KEY] = {
        "operation_id": "operation:test",
    }
    for side in ("left", "right"):
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=np.full(6, 0.2),
            )
        )
    return command


def test_measurement_sampler_deadline_preserves_phase_and_skips_missed_ticks() -> None:
    assert _next_periodic_measurement_deadline(
        previous_deadline=1.0,
        completed_at=1.04,
        period_sec=0.1,
    ) == pytest.approx(1.1)
    assert _next_periodic_measurement_deadline(
        previous_deadline=1.0,
        completed_at=1.25,
        period_sec=0.1,
    ) == pytest.approx(1.3)


def test_background_sampler_owns_one_persistent_dual_refresh_executor() -> None:
    ports, _, _ = _ports(background_measurement_sampling=True)
    executor = ports._measurement_refresh_executor

    assert executor is not None
    assert ports.arm_state_reader.concurrent_refresh_executor is executor

    ports.start()
    ports.close()
    ports.close()

    assert ports._measurement_refresh_executor_shutdown is True
    assert executor._shutdown is True


def test_rm65_ports_read_only_authoritative_dual_hardware_qpos() -> None:
    ports, _, target_builder = _ports()
    assert ports.arm_state_reader.compute_tcp_pose is False
    ports.start()

    cut = ports.observe()
    ports.synchronize_target_builder(cut.qpos_by_side)
    ports.synchronize_target_builder(cut.qpos_by_side)

    np.testing.assert_allclose(cut.qpos_by_side["left"], np.full(6, 0.1))
    np.testing.assert_allclose(cut.qpos_by_side["right"], np.full(6, 0.2))
    assert cut.evidence_refs[0].startswith("rm65-hardware-qpos-cut:")
    assert cut.observation.metadata["source"] == "exclusive_rm65_skill_deploy"
    assert cut.observation.metadata["hardware_read_phase_timing_by_side"][
        "left"
    ]["sdk_call_wall_ms"] == 7.0
    assert [item[2] for item in target_builder.syncs] == [True, False]


def test_rm65_ports_reject_stale_hardware_cut() -> None:
    ports, _, _ = _ports(age_sec=1.0)
    ports.start()

    with pytest.raises(RuntimeError, match="coherence_diagnostic") as captured:
        ports.observe()

    assert '"concurrently_refreshed_sides":["left","right"]' in str(captured.value)


def test_rm65_ports_request_one_concurrent_dual_arm_fresh_cut() -> None:
    ports, _, _ = _ports()
    ports.start()
    ports.clock = lambda: 100.21

    def state(side: str, measured_at: float) -> dict[str, object]:
        return {
            "hardware_qpos": np.full(6, 0.1 if side == "left" else 0.2),
            "hardware_measurement_timestamp": measured_at,
            "hardware_measurement_sequence": 4 if side == "left" else 5,
            "hardware_measurement_valid": True,
            "hardware_read_success": True,
        }

    class _CoherenceReader:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def snapshot(
            self,
            _now: float,
            *,
            cut_after_refresh: bool,
            concurrent_refresh_sides: tuple[str, ...] = (),
        ) -> dict[str, dict[str, object]]:
            assert cut_after_refresh is True
            self.calls.append(concurrent_refresh_sides)
            assert concurrent_refresh_sides == ("left", "right")
            return {
                "left": state("left", 100.205),
                "right": state("right", 100.20),
            }

    reader = _CoherenceReader()
    ports.arm_state_reader = reader

    cut = ports.observe()

    assert reader.calls == [("left", "right")]
    assert cut.observation.metadata["concurrently_refreshed_sides"] == (
        "left",
        "right",
    )
    assert cut.observation.metadata["hardware_measurement_span_sec"] == pytest.approx(
        0.005
    )
    assert cut.measured_at == pytest.approx(100.20)


def test_rm65_ports_avoid_serial_read_induced_staleness() -> None:
    ports, backends, _ = _ports()
    ports.start()
    ports.clock = time.time
    ports.arm_state_reader.clock = time.time

    for backend in backends.values():
        def delayed_refresh(*, selected_backend=backend):
            time.sleep(0.21)
            selected_backend.measured_at = time.time()
            return selected_backend.get_state()

        backend.refresh_state_from_hardware = delayed_refresh

    cut = ports.observe()

    assert cut.available_at - cut.measured_at < 0.05
    assert cut.observation.metadata["hardware_measurement_span_sec"] < 0.05


def test_rm65_ports_qpos_sampler_excludes_slow_backend_fk() -> None:
    ports, backends, _ = _ports(max_measurement_age_sec=0.2)
    ports.start()
    ports.clock = time.time
    ports.arm_state_reader.clock = time.time
    derived_state_calls = {"left": 0, "right": 0}

    for side, backend in backends.items():
        original_get_state = backend.get_state

        def get_state(
            *,
            include_derived_pose=True,
            include_timing_percentiles=True,
            include_pose_source_attestation=None,
            selected_side=side,
            selected_get_state=original_get_state,
        ):
            if include_derived_pose:
                derived_state_calls[selected_side] += 1
                time.sleep(0.12)
            return selected_get_state()

        def refresh_state_from_hardware(
            *,
            include_derived_pose=True,
            include_timing_percentiles=True,
            include_pose_source_attestation=None,
            selected_backend=backend,
            selected_get_state=get_state,
        ):
            selected_backend.measured_at = time.time()
            return selected_get_state(
                include_derived_pose=include_derived_pose,
                include_timing_percentiles=include_timing_percentiles,
                include_pose_source_attestation=include_pose_source_attestation,
            )

        backend.get_state = get_state
        backend.refresh_state_from_hardware = refresh_state_from_hardware

    cut = ports.observe()

    assert cut.available_at - cut.measured_at < 0.05
    assert derived_state_calls == {"left": 0, "right": 0}


def test_background_sampler_keeps_control_observe_nonblocking_during_read() -> None:
    ports, _, _ = _ports(
        background_measurement_sampling=True,
        state_refresh_hz=100.0,
    )
    ports.clock = time.time
    ports.arm_state_reader.clock = time.time
    ports.start()
    entered = threading.Event()
    release = threading.Event()

    class _BlockingReader:
        def snapshot(
            self,
            _now: float,
            *,
            cut_after_refresh: bool,
            concurrent_refresh_sides: tuple[str, ...] = (),
        ):
            assert cut_after_refresh is True
            assert concurrent_refresh_sides == ("left", "right")
            entered.set()
            assert release.wait(timeout=1.0)
            measured_at = time.time()
            return {
                side: {
                    "hardware_qpos": np.full(
                        6, 0.1 if side == "left" else 0.2
                    ),
                    "hardware_measurement_timestamp": measured_at,
                    "hardware_measurement_sequence": (
                        14 if side == "left" else 15
                    ),
                    "hardware_measurement_valid": True,
                    "hardware_read_success": True,
                }
                for side in ("left", "right")
            }

    ports.arm_state_reader = _BlockingReader()
    assert entered.wait(timeout=1.0)

    started = time.monotonic()
    cut = ports.observe()
    elapsed = time.monotonic() - started
    release.set()
    deadline = time.monotonic() + 1.0
    while (
        ports.measurement_sampler_snapshot()["sample_count"] < 2
        and time.monotonic() < deadline
    ):
        time.sleep(0.005)
    ports.close()

    assert elapsed < 0.03
    assert dict(cut.source_sequence_by_side) == {"left": 4, "right": 5}


def test_background_sampler_latches_real_measurement_stall() -> None:
    ports, _, _ = _ports(
        background_measurement_sampling=True,
        state_refresh_hz=100.0,
        max_measurement_age_sec=0.05,
    )
    ports.clock = time.time
    ports.arm_state_reader.clock = time.time
    ports.start()
    deadline = time.monotonic() + 1.0
    snapshot = ports.measurement_sampler_snapshot()
    while snapshot["fault"] is None and time.monotonic() < deadline:
        time.sleep(0.005)
        snapshot = ports.measurement_sampler_snapshot()

    with pytest.raises(RuntimeError, match="measurement sampler failed"):
        ports.observe()
    ports.close()

    assert snapshot["fault"] is not None
    assert "stale" in str(snapshot["fault"])


def test_rm65_ports_use_checked_dispatch_and_lower_hold() -> None:
    ports, backends, _ = _ports()
    ports.start()
    command = _command()

    assert ports.admit_command(
        command,
        TeleopObservation(timestamp=time.time()),
        time.time(),
    )
    dispatch = ports.dispatch(command)
    hold = ports.hold("skill complete")

    assert dispatch.succeeded is True
    assert dispatch.evidence_refs[0].startswith("rm65-skill-dispatch:")
    assert all(len(backend.sent) == 1 for backend in backends.values())
    assert hold.succeeded is True
    assert hold.evidence_refs[0].startswith("rm65-skill-hold:")
    assert all(
        backend.holds
        == ["exclusive real Skill startup fence", "skill complete"]
        for backend in backends.values()
    )
    assert all(backend.target_active is False for backend in backends.values())


def test_rm65_ports_own_start_and_hold_before_close() -> None:
    ports, backends, _ = _ports()

    ports.start()
    ports.start()
    ports.close()

    assert ports.resources_started is False
    assert ports.resources_closed is True
    assert ports.startup_hold_receipt is not None
    assert ports.startup_hold_receipt.succeeded is True
    assert all(backend.starts == 1 for backend in backends.values())
    assert ports.position_force_guard.starts == 1
    assert ports.position_force_guard.closes == 1
    assert all(
        backend.holds
        == [
            "exclusive real Skill startup fence",
            "exclusive real Skill resource shutdown",
        ]
        for backend in backends.values()
    )
    assert all(backend.closes == 1 for backend in backends.values())
    events = ports.position_force_guard.events
    hold_indices = [
        index
        for index, event in enumerate(events)
        if event in {"backend_hold:left", "backend_hold:right"}
    ]
    assert max(hold_indices) < events.index("force_guard_close")
    assert events.index("force_guard_close") < min(
        events.index("backend_close:left"),
        events.index("backend_close:right"),
    )


def test_rm65_ports_retry_only_unfinished_backend_close_stage() -> None:
    ports, backends, _ = _ports(backend_close_failures={"right": 1})
    ports.start()

    with pytest.raises(
        RuntimeError,
        match="right: RuntimeError: right close failed",
    ):
        ports.close()

    assert ports.resources_started is False
    assert ports.resources_closed is False
    assert ports.position_force_guard.closes == 1
    assert backends["right"].closes == 1
    assert backends["left"].closes == 1
    assert all(len(backend.holds) == 2 for backend in backends.values())

    ports.close()
    ports.close()

    assert ports.resources_closed is True
    assert ports.position_force_guard.closes == 1
    assert backends["right"].closes == 2
    assert backends["left"].closes == 1
    assert all(len(backend.holds) == 2 for backend in backends.values())


def test_rm65_ports_retry_only_unfinished_force_guard_close_stage() -> None:
    ports, backends, _ = _ports(force_guard_close_failures=1)
    ports.start()

    with pytest.raises(RuntimeError, match="force_guard: RuntimeError"):
        ports.close()

    assert ports.resources_started is False
    assert ports.resources_closed is False
    assert ports.position_force_guard.closes == 1
    assert all(backend.closes == 1 for backend in backends.values())

    ports.close()

    assert ports.resources_closed is True
    assert ports.position_force_guard.closes == 2
    assert all(backend.closes == 1 for backend in backends.values())


def test_rm65_ports_do_not_turn_unconfirmed_shutdown_hold_into_success() -> None:
    ports, backends, _ = _ports()
    ports.start()
    backends["left"].hold_failures = 1

    with pytest.raises(RuntimeError, match="lower HOLD was not confirmed"):
        ports.close()

    assert ports.resources_started is False
    assert ports.resources_closed is True
    assert all(backend.closes == 1 for backend in backends.values())

    with pytest.raises(RuntimeError, match="lower HOLD was not confirmed"):
        ports.close()

    assert all(backend.closes == 1 for backend in backends.values())


def test_rm65_ports_start_failure_is_closed_and_not_retryable() -> None:
    ports, backends, _ = _ports(
        force_guard_ready=False,
        background_measurement_sampling=True,
    )

    with pytest.raises(RuntimeError, match="force guard is not motion-ready"):
        ports.start()

    assert ports.resources_started is False
    assert ports.position_force_guard.closes == 1
    assert ports._measurement_refresh_executor_shutdown is True
    assert all(backend.holds for backend in backends.values())
    assert all(backend.closes == 1 for backend in backends.values())
    with pytest.raises(RuntimeError, match="cannot restart"):
        ports.start()


def test_rm65_ports_start_rollback_keeps_failed_cleanup_retryable() -> None:
    ports, backends, _ = _ports(
        force_guard_ready=False,
        force_guard_close_failures=1,
    )

    with pytest.raises(RuntimeError, match="startup rollback incomplete"):
        ports.start()

    assert ports.resources_started is False
    assert ports.resources_closed is False
    assert ports.position_force_guard.closes == 1
    assert all(backend.closes == 1 for backend in backends.values())
    with pytest.raises(RuntimeError, match="cannot restart"):
        ports.start()

    ports.close()

    assert ports.resources_closed is True
    assert ports.position_force_guard.closes == 2
    assert all(backend.closes == 1 for backend in backends.values())


def test_rm65_ports_convert_backend_rejection_to_failed_receipt() -> None:
    ports, _, _ = _ports(accepted=False)
    ports.start()

    receipt = ports.dispatch(_command())

    assert receipt.succeeded is False
    assert receipt.evidence_refs[0].startswith("rm65-skill-dispatch:")


def test_rm65_ports_reject_non_coupled_or_unbound_command() -> None:
    ports, _, _ = _ports()
    ports.start()
    command = _command()
    command.commands.pop("left_arm")

    assert not ports.admit_command(
        command,
        TeleopObservation(timestamp=time.time()),
        time.time(),
    )


def test_rm65_ports_fail_closed_when_force_guard_is_not_motion_ready() -> None:
    ports, _, _ = _ports()
    ports.start()
    ports.position_force_guard.motion_ready = False

    with pytest.raises(RuntimeError, match="force guard is not motion-ready"):
        ports.admit_command(
            _command(),
            TeleopObservation(timestamp=time.time()),
            time.time(),
        )

    receipt = ports.dispatch(_command())
    assert receipt.succeeded is False


def test_force_guard_factory_uses_rm65_ports_without_realteleop() -> None:
    _, backends, _ = _ports()

    guard = build_exclusive_rm65_skill_position_force_guard(
        arm_backends=backends,
    )

    assert guard.motion_ready is False
    assert set(guard.snapshot()["last_reading_by_side"]) == set()


def test_rm65_skill_ports_import_does_not_load_realteleop_or_composition() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import json
import sys
import teleop_runtime.skill_deploy.rm65_ports
print(json.dumps(sorted(
    name for name in sys.modules
    if name == 'control_composition'
    or name.startswith('control_composition.')
    or name == 'teleop_runtime.real_teleop'
    or name.startswith('teleop_runtime.real_teleop.')
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_rm65_factory_binds_every_effect_to_the_same_port_owner() -> None:
    ports, _, target_builder = _ports()
    sentinel = object()

    with patch(
        "agent_gateway.profiles.real.exclusive_factory."
        "assemble_exclusive_real_agent_primitive_deployment",
        return_value=sentinel,
    ) as assemble:
        result = assemble_exclusive_rm65_agent_primitive_deployment(
            target_builder=target_builder,
            rm65_ports=ports,
            port_config=object(),
            observation_attachment=object(),
            frame_graph=object(),
            primitive_reference_frame_id="station",
            safety_envelope_refs=(object(),),
            allowed_client_ids=("rpent:test",),
        )

    assert result is sentinel
    kwargs = assemble.call_args.kwargs
    for field_name in (
        "observation_provider",
        "synchronize_target_builder",
        "command_admission",
        "command_dispatch",
        "hold_controller",
    ):
        assert getattr(kwargs[field_name], "__self__", None) is ports
