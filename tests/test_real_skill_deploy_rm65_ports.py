from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from robot_motion.runtime import (
    CommandDispatchReceipt,
    EffectorCommand,
    TeleopCommand,
    TeleopObservation,
)
from agent_gateway.profiles.real.exclusive_factory import (
    assemble_exclusive_rm65_agent_primitive_deployment,
)
from teleop_runtime.interfaces import EffectorBackend
from teleop_runtime.skill_deploy import (
    REAL_AGENT_MOTION_COMMAND_METADATA_KEY,
    ExclusiveRM65SkillDeployPorts,
    ExclusiveRM65SkillPortProfile,
    build_exclusive_rm65_skill_position_force_guard,
)


class _Backend(EffectorBackend):
    def __init__(
        self,
        side: str,
        *,
        measured_at: float,
        accepted: bool = True,
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
        events: list[str] | None = None,
    ) -> None:
        self.motion_ready = False
        self.fault = None
        self.starts = 0
        self.closes = 0
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
        self.motion_ready = False


def _ports(
    *,
    accepted: bool = True,
    age_sec: float = 0.01,
    force_guard_ready: bool = True,
):
    now = time.time()
    events: list[str] = []
    backends = {
        side: _Backend(
            side,
            measured_at=now - age_sec,
            accepted=accepted,
            events=events,
        )
        for side in ("left", "right")
    }
    target_builder = _TargetBuilder()
    force_guard = _ForceGuard(
        ready_on_start=force_guard_ready,
        events=events,
    )
    ports = ExclusiveRM65SkillDeployPorts(
        arm_backends=backends,
        target_builder=target_builder,
        position_force_guard=force_guard,
        profile=ExclusiveRM65SkillPortProfile(
            state_refresh_hz=30.0,
            max_measurement_age_sec=0.2,
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


def test_rm65_ports_read_only_authoritative_dual_hardware_qpos() -> None:
    ports, _, target_builder = _ports()
    ports.start()

    cut = ports.observe()
    ports.synchronize_target_builder(cut.qpos_by_side)
    ports.synchronize_target_builder(cut.qpos_by_side)

    np.testing.assert_allclose(cut.qpos_by_side["left"], np.full(6, 0.1))
    np.testing.assert_allclose(cut.qpos_by_side["right"], np.full(6, 0.2))
    assert cut.evidence_refs[0].startswith("rm65-hardware-qpos-cut:")
    assert cut.observation.metadata["source"] == "exclusive_rm65_skill_deploy"
    assert [item[2] for item in target_builder.syncs] == [True, False]


def test_rm65_ports_reject_stale_hardware_cut() -> None:
    ports, _, _ = _ports(age_sec=1.0)
    ports.start()

    with pytest.raises(RuntimeError, match="hardware qpos cut is stale"):
        ports.observe()


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
    ports.close()

    assert ports.resources_started is False
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


def test_rm65_ports_start_failure_is_closed_and_not_retryable() -> None:
    ports, backends, _ = _ports(force_guard_ready=False)

    with pytest.raises(RuntimeError, match="force guard is not motion-ready"):
        ports.start()

    assert ports.resources_started is False
    assert ports.position_force_guard.closes == 1
    assert all(backend.holds for backend in backends.values())
    assert all(backend.closes == 1 for backend in backends.values())
    with pytest.raises(RuntimeError, match="cannot restart"):
        ports.start()


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
