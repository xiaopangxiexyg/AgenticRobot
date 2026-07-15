from __future__ import annotations

import inspect

import numpy as np

from robot_motion.diagnostics import (
    FaultEvent as RobotMotionFaultEvent,
    FaultSeverity as RobotMotionFaultSeverity,
    RateTracker as RobotMotionRateTracker,
    TeleopRunState as RobotMotionTeleopRunState,
    TeleopStateMachine as RobotMotionTeleopStateMachine,
)
from robot_motion.runtime import (
    BimanualRig as RobotMotionBimanualRig,
    EffectorBackend as RobotMotionEffectorBackend,
    EffectorCommand as RobotMotionEffectorCommand,
    ObservationChannel as RobotMotionObservationChannel,
    TeleopCommand as RobotMotionTeleopCommand,
    merge_teleop_commands as robot_motion_merge_teleop_commands,
)
from robot_motion.safety import (
    MotionSafetyConfig as RobotMotionSafetyConfig,
    SafetyMonitor as RobotMotionSafetyMonitor,
    SafetyViolation as RobotMotionSafetyViolation,
    TeleopSafetyConfig as RobotMotionTeleopSafetyConfig,
)
from sim.mujoco import backend as mujoco_backend
from teleop_runtime.diagnostics import (
    FaultEvent as LegacyFaultEvent,
    FaultSeverity as LegacyFaultSeverity,
    RateTracker as LegacyRateTracker,
    TeleopRunState as LegacyTeleopRunState,
    TeleopStateMachine as LegacyTeleopStateMachine,
)
from teleop_runtime.interfaces import EffectorBackend as LegacyEffectorBackend
from teleop_runtime.rigs.bimanual import BimanualRig as LegacyBimanualRig
from teleop_runtime.safety import (
    SafetyMonitor as LegacySafetyMonitor,
    SafetyViolation as LegacySafetyViolation,
    TeleopSafetyConfig as LegacyTeleopSafetyConfig,
)
from teleop_runtime.schemas import (
    EffectorCommand as LegacyEffectorCommand,
    ObservationChannel as LegacyObservationChannel,
    TeleopCommand as LegacyTeleopCommand,
    merge_teleop_commands as legacy_merge_teleop_commands,
)


def test_legacy_runtime_contracts_reexport_robot_motion_objects() -> None:
    assert LegacyEffectorCommand is RobotMotionEffectorCommand
    assert LegacyObservationChannel is RobotMotionObservationChannel
    assert LegacyTeleopCommand is RobotMotionTeleopCommand
    assert legacy_merge_teleop_commands is robot_motion_merge_teleop_commands
    assert LegacyEffectorBackend is RobotMotionEffectorBackend
    assert LegacyBimanualRig is RobotMotionBimanualRig
    assert LegacyTeleopSafetyConfig is RobotMotionTeleopSafetyConfig
    assert LegacyTeleopSafetyConfig is RobotMotionSafetyConfig
    assert LegacySafetyMonitor is RobotMotionSafetyMonitor
    assert LegacySafetyViolation is RobotMotionSafetyViolation
    assert LegacyFaultEvent is RobotMotionFaultEvent
    assert LegacyFaultSeverity is RobotMotionFaultSeverity
    assert LegacyRateTracker is RobotMotionRateTracker
    assert LegacyTeleopRunState is RobotMotionTeleopRunState
    assert LegacyTeleopStateMachine is RobotMotionTeleopStateMachine


def test_runtime_contract_copy_and_merge_match_legacy_behavior() -> None:
    command = LegacyTeleopCommand(timestamp=1.0, metadata={"source": "legacy_path"})
    value = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    command.add_command(
        LegacyEffectorCommand(
            effector_name="right_hand",
            side="right",
            command_type="joint_vector",
            value=value,
            metadata={"nested": {"value": value}},
        )
    )

    merged = robot_motion_merge_teleop_commands([command])

    assert isinstance(merged, RobotMotionTeleopCommand)
    assert merged.timestamp == 1.0
    assert merged.metadata == {"source": "legacy_path"}
    copied = merged.commands["right_hand"]
    assert isinstance(copied, RobotMotionEffectorCommand)
    assert copied.value is not value
    assert copied.metadata is not command.commands["right_hand"].metadata
    assert copied.metadata["nested"]["value"] is value
    np.testing.assert_array_equal(copied.value, value)


def test_mujoco_backend_uses_robot_motion_runtime_contracts() -> None:
    assert mujoco_backend.EffectorBackend is RobotMotionEffectorBackend
    assert mujoco_backend.EffectorCommand is RobotMotionEffectorCommand
    assert "teleop_runtime" not in inspect.getsource(mujoco_backend)
