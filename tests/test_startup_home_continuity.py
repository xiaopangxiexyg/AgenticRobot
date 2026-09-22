from __future__ import annotations

import numpy as np

from robot_motion.runtime import EffectorCommand, TeleopCommand
from teleop_runtime.real_teleop.startup_home_continuity import (
    StartupHomeContinuityConfig,
    StartupHomeContinuityGate,
)


def _fk(qpos: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[0, 3] = float(qpos[0])
    return pose


def _command(left: np.ndarray, right: np.ndarray) -> TeleopCommand:
    command = TeleopCommand(timestamp=1.0, metadata={"source": "test"})
    for side, qpos in (("left", left), ("right", right)):
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=np.asarray(qpos, dtype=float).copy(),
                frame_name="arm_base",
                metadata={"candidate_metadata_preserved": True},
            )
        )
    return command


def _gate(*, required_frames: int = 3) -> StartupHomeContinuityGate:
    return StartupHomeContinuityGate(
        config=StartupHomeContinuityConfig(
            active_sides=("left", "right"),
            max_joint_delta_deg=0.75,
            max_tcp_translation_m=0.003,
            max_tcp_rotation_deg=2.0,
            required_consecutive_frames=required_frames,
        ),
        fk_by_side={"left": _fk, "right": _fk},
    )


def test_outside_home_qref_is_withheld_without_modifying_candidate() -> None:
    gate = _gate()
    home = {"left": np.zeros(6), "right": np.zeros(6)}
    gate.arm_from_home(home)
    candidate = _command(
        np.asarray([0.02, 0, 0, 0, 0, 0]),
        np.zeros(6),
    )
    before = candidate.commands["left_arm"].value.copy()

    assert not gate.admit(candidate, None, 1.0)
    np.testing.assert_array_equal(candidate.commands["left_arm"].value, before)
    assert candidate.commands["left_arm"].metadata == {
        "candidate_metadata_preserved": True
    }
    receipt = candidate.metadata["startup_home_continuity"]
    assert receipt["status"] == "outside_home_envelope"
    assert receipt["dispatch_allowed"] is False
    assert candidate.metadata["dataset_recording_eligible"] is False


def test_home_qpos_snapshot_is_defensive_and_matches_gate_anchor() -> None:
    gate = _gate()
    home = {"left": np.zeros(6), "right": np.full(6, 0.1)}

    assert gate.home_qpos_snapshot() is None
    gate.arm_from_home(home)

    snapshot = gate.home_qpos_snapshot()
    assert snapshot is not None
    np.testing.assert_array_equal(snapshot["left"], home["left"])
    np.testing.assert_array_equal(snapshot["right"], home["right"])

    snapshot["right"][0] = 99.0
    np.testing.assert_array_equal(
        gate.home_qpos_snapshot()["right"],
        home["right"],
    )


def test_inside_home_qref_releases_only_after_consecutive_dwell() -> None:
    gate = _gate(required_frames=3)
    gate.arm_from_home({"left": np.zeros(6), "right": np.zeros(6)})
    near = np.asarray([0.002, 0, 0, 0, 0, 0])

    first = _command(near, near)
    second = _command(near, near)
    third = _command(near, near)
    assert not gate.admit(first, None, 1.0)
    assert not gate.admit(second, None, 1.1)
    assert gate.admit(third, None, 1.2)
    assert third.metadata["startup_home_continuity"]["status"] == "released"

    # The gate is intentionally one-shot for the epoch. Normal teleop motion
    # is governed by the existing IK, collision, endpoint, and planner limits.
    far_after_release = _command(np.ones(6), np.ones(6))
    assert gate.admit(far_after_release, None, 1.3)
    assert "startup_home_continuity" not in far_after_release.metadata


def test_outlier_resets_dwell_and_new_home_rearms_released_gate() -> None:
    gate = _gate(required_frames=2)
    zero = np.zeros(6)
    near = np.asarray([0.002, 0, 0, 0, 0, 0])
    far = np.asarray([0.02, 0, 0, 0, 0, 0])
    gate.arm_from_home({"left": zero, "right": zero})

    assert not gate.admit(_command(near, near), None, 1.0)
    assert not gate.admit(_command(far, near), None, 1.1)
    assert not gate.admit(_command(near, near), None, 1.2)
    assert gate.admit(_command(near, near), None, 1.3)

    gate.arm_from_home({"left": zero, "right": zero})
    assert not gate.admit(_command(near, near), None, 2.0)
    assert gate.snapshot()["epoch"] == 2


def test_missing_or_nonfinite_final_arm_qref_fails_closed() -> None:
    gate = _gate(required_frames=1)
    gate.arm_from_home({"left": np.zeros(6), "right": np.zeros(6)})
    missing = TeleopCommand(timestamp=1.0)
    assert not gate.admit(missing, None, 1.0)
    assert gate.snapshot()["status"] == "missing_arm_qref"

    invalid = _command(np.zeros(6), np.zeros(6))
    invalid.commands["left_arm"].value[0] = np.nan
    assert not gate.admit(invalid, None, 1.1)
    assert gate.snapshot()["status"] == "invalid_arm_qref"
