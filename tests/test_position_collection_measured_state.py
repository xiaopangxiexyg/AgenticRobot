from __future__ import annotations

import numpy as np

from robot_motion.deploy.schema import STATE_ARM_POSE_SLICES, STATE_ARM_QPOS_SLICES
from teleop_runtime.collection.build_collect_components import CollectorComponents
from teleop_runtime.collection.sample_assembler import (
    CollectorSampleBuilder,
    CollectorSampleBuilderConfig,
)
from teleop_runtime.collection.schema_v21 import collector_sample_to_row
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.schemas import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.session import SessionStepResult


class _ArmBackend:
    def __init__(self) -> None:
        self.commanded_qpos = np.zeros(6, dtype=float)
        self.commanded_pose = np.eye(4, dtype=float)

    def get_state(self):
        return {
            "current_qpos": self.commanded_qpos.copy(),
            "current_tcp_pose": self.commanded_pose.copy(),
            "last_attempted_qpos": np.full(6, 0.2),
            "last_attempted_sequence": 12,
            "last_accepted_qpos": np.full(6, 0.1),
            "last_accepted_sequence": 11,
            "measured_qpos": np.full(6, 0.05),
            "measured_sequence": 10,
            "background_streaming": {
                "enabled": True,
                "hz": 150.0,
                "target_sequence": 7,
                "target_stale": False,
                "accepted_send_count": 30,
                "failed_send_count": 0,
                "healthy": True,
                "health_reasons": [],
            },
        }

    def compute_tcp_pose(self, *, qpos=None):
        if qpos is None:
            return self.commanded_pose.copy()
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = np.asarray(qpos, dtype=float)[:3]
        return pose


def test_position_collection_prefers_causal_measured_state_and_keeps_commanded_sidecar() -> None:
    measured_qpos = np.arange(6, dtype=float) * 0.1
    measured_pose = np.eye(4, dtype=float)
    measured_pose[:3, 3] = measured_qpos[:3]
    builder = CollectorSampleBuilder(
        CollectorSampleBuilderConfig(
            active_sides=("left",),
            prefer_causal_measured_arm_state=True,
        ),
        arm_state_metadata_provider=lambda _components, sample_timestamp: {
            "left": {
                "actual_qpos": measured_qpos.copy(),
                "actual_tcp_pose": measured_pose.copy(),
                "actual_state_timestamp": float(sample_timestamp) - 0.01,
                "actual_state_stale": False,
            }
        },
    )
    sample = builder(
        SessionStepResult(
            timestamp=10.0,
            state=TeleopRunState.RUNNING,
            control_hz=45.0,
        ),
        CollectorComponents(
            session=None,
            backends={"left_arm": _ArmBackend()},
        ),
    )

    assert sample is not None
    np.testing.assert_allclose(
        sample.observation_state[STATE_ARM_QPOS_SLICES["left"]],
        measured_qpos,
    )
    np.testing.assert_allclose(
        sample.observation_state[STATE_ARM_POSE_SLICES["left"]][:3],
        measured_pose[:3, 3],
    )
    np.testing.assert_allclose(sample.arm_states["left"]["commanded_qpos"], np.zeros(6))
    np.testing.assert_allclose(sample.arm_states["left"]["current_qpos"], measured_qpos)
    assert sample.arm_state_source == "causal_measured_hardware"
    row, _ = collector_sample_to_row(sample)
    np.testing.assert_allclose(row["observation.arm.left.current_qpos"], measured_qpos)
    np.testing.assert_allclose(row["observation.arm.left.commanded_qpos"], np.zeros(6))
    assert row["observation.arm.left.state_source"] == "causal_measured_hardware"
    assert row["position.transport.left.target_sequence"] == 7
    assert row["position.transport.left.accepted_send_count"] == 30
    assert row["position.transport.left.healthy"] is True


def test_legacy_collection_keeps_commanded_state_when_measured_preference_is_off() -> None:
    builder = CollectorSampleBuilder(
        CollectorSampleBuilderConfig(
            active_sides=("left",),
            prefer_causal_measured_arm_state=False,
        ),
        arm_state_metadata_provider=lambda _components, _sample_timestamp: {
            "left": {"actual_qpos": np.ones(6, dtype=float)}
        },
    )
    sample = builder(
        SessionStepResult(
            timestamp=10.0,
            state=TeleopRunState.RUNNING,
            control_hz=15.0,
        ),
        CollectorComponents(
            session=None,
            backends={"left_arm": _ArmBackend()},
        ),
    )

    assert sample is not None
    np.testing.assert_allclose(
        sample.observation_state[STATE_ARM_QPOS_SLICES["left"]],
        np.zeros(6),
    )
    assert "commanded_qpos" not in sample.arm_states["left"]
    assert sample.arm_state_source == "last_commanded_action"


def test_init_phase_is_collector_audit_not_a_second_model_label() -> None:
    target = np.eye(4, dtype=float)
    measured_qpos = np.zeros(6, dtype=float)

    def command_at(timestamp: float) -> TeleopCommand:
        command = TeleopCommand(timestamp=timestamp)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="cartesian_pose",
                value=target.copy(),
                frame_name="left_arm_base",
                metadata={
                    "init_arm_active": True,
                    "init_arm_side": "left",
                    "init_arm_mode": "moving",
                    "init_arm_target_pose_base": target.copy(),
                },
            )
        )
        return command

    builder = CollectorSampleBuilder(
        CollectorSampleBuilderConfig(
            active_sides=("left",),
            prefer_causal_measured_arm_state=True,
        ),
        arm_state_metadata_provider=lambda _components, sample_timestamp: {
            "left": {
                "actual_qpos": measured_qpos.copy(),
                "actual_tcp_pose": target.copy(),
                "actual_state_timestamp": float(sample_timestamp),
                "actual_state_stale": False,
            }
        },
    )
    components = CollectorComponents(session=None, backends={"left_arm": _ArmBackend()})

    sample = None
    for timestamp in np.arange(0.0, 3.3, 0.1):
        sample = builder(
            SessionStepResult(
                timestamp=float(timestamp),
                state=TeleopRunState.RUNNING,
                control_hz=15.0,
                observation=TeleopObservation(timestamp=float(timestamp)),
                command=command_at(float(timestamp)),
            ),
            components,
        )

    assert sample is not None
    side_metadata = sample.metadata["arm_mode"]["sides"]["left"]
    assert side_metadata["mode"] == "init_pose"
    assert side_metadata["audit"]["init_phase"] == "init_hold"
    row, _ = collector_sample_to_row(sample)
    assert row["teleop.arm_mode.left.label"] == "init_pose"
    assert row["teleop.arm_mode.left.init_phase"] == "init_hold"
    assert "phase_target" not in row
