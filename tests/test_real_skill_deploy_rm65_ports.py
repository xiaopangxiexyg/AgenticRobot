from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

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
)


class _Backend(EffectorBackend):
    def __init__(
        self,
        side: str,
        *,
        measured_at: float,
        accepted: bool = True,
    ) -> None:
        self.side = side
        self.measured_at = measured_at
        self.accepted = bool(accepted)
        self.target_active = False
        self.sent: list[EffectorCommand] = []
        self.holds: list[str] = []

    @property
    def name(self) -> str:
        return f"{self.side}_arm"

    def is_ready(self) -> bool:
        return True

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
        self.target_active = False


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


def _ports(*, accepted: bool = True, age_sec: float = 0.01):
    # ArmStateReader publishes availability when get_state returns. Keep the
    # requested causal cut just after that bounded in-process publication.
    now = time.time() + 0.05
    backends = {
        side: _Backend(
            side,
            measured_at=now - age_sec,
            accepted=accepted,
        )
        for side in ("left", "right")
    }
    target_builder = _TargetBuilder()
    ports = ExclusiveRM65SkillDeployPorts(
        arm_backends=backends,
        target_builder=target_builder,
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

    with pytest.raises(RuntimeError, match="hardware qpos cut is stale"):
        ports.observe()


def test_rm65_ports_use_checked_dispatch_and_lower_hold() -> None:
    ports, backends, _ = _ports()
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
    assert all(backend.holds == ["skill complete"] for backend in backends.values())
    assert all(backend.target_active is False for backend in backends.values())


def test_rm65_ports_convert_backend_rejection_to_failed_receipt() -> None:
    ports, _, _ = _ports(accepted=False)

    receipt = ports.dispatch(_command())

    assert receipt.succeeded is False
    assert receipt.evidence_refs[0].startswith("rm65-skill-dispatch:")


def test_rm65_ports_reject_non_coupled_or_unbound_command() -> None:
    ports, _, _ = _ports()
    command = _command()
    command.commands.pop("left_arm")

    assert not ports.admit_command(
        command,
        TeleopObservation(timestamp=time.time()),
        time.time(),
    )


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
