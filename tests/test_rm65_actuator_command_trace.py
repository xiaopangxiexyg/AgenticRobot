from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from robot_motion.position_control import PositionCommandAuditEvent
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.real_teleop.actuator_command_trace import (
    ACTUATOR_COMMAND_TRACE_DATA_NAME,
    ACTUATOR_COMMAND_TRACE_MANIFEST_NAME,
    ActuatorCommandTraceConfig,
    ActuatorCommandTraceRecorder,
)
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop
from teleop_runtime.schemas import EffectorCommand


class _FakeArm:
    def __init__(self) -> None:
        self.sent_joint_deg: list[np.ndarray] = []

    def Movej_CANFD(self, joint_deg, follow=False, trajectory_mode=0, radio=0):
        del follow, trajectory_mode, radio
        self.sent_joint_deg.append(np.asarray(joint_deg, dtype=float).copy())
        return 0


def _record_one(recorder: ActuatorCommandTraceRecorder, *, tick: int = 7) -> bool:
    return recorder.record_command(
        "left",
        tick,
        3,
        1,
        4,
        10.0,
        10.001,
        5,
        1_700_000_000.0,
        10.002,
        10.003,
        np.linspace(0.0, 0.5, 6),
        0,
        True,
        5,
    )


def test_preallocated_trace_writes_no_command_files_until_close(tmp_path: Path) -> None:
    output_dir = tmp_path / "trace"
    recorder = ActuatorCommandTraceRecorder(
        ActuatorCommandTraceConfig(
            output_dir=output_dir,
            active_sides=("left",),
            command_hz=120.0,
            max_duration_sec=1.0,
            metadata={"profile": "rm65_bimanual_position_v1"},
        )
    )

    assert _record_one(recorder)
    assert list(output_dir.iterdir()) == []

    recorder.close()

    with np.load(output_dir / ACTUATOR_COMMAND_TRACE_DATA_NAME, allow_pickle=False) as data:
        rows = data["left"]
        assert rows.shape == (1,)
        assert rows["tick_sequence"].tolist() == [7]
        assert rows["sdk_result_code"].tolist() == [0]
        assert rows["sdk_accepted"].tolist() == [1]
        np.testing.assert_allclose(rows["q_cmd_rad"][0], np.linspace(0.0, 0.5, 6))
        assert rows["sdk_attempt_monotonic_ns"].tolist() == [10_002_000_000]

    manifest = json.loads(
        (output_dir / ACTUATOR_COMMAND_TRACE_MANIFEST_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["result"]["closed"] is True
    assert manifest["result"]["rows_by_side"] == {"left": 1}
    assert manifest["runtime_metadata"]["profile"] == "rm65_bimanual_position_v1"
    assert "no queue/json/fk/disk_write" in manifest["hot_path_policy"]


def test_preallocated_trace_persists_sparse_singularity_rows_on_close(
    tmp_path: Path,
) -> None:
    recorder = ActuatorCommandTraceRecorder(
        ActuatorCommandTraceConfig(
            output_dir=tmp_path / "singularity",
            active_sides=("left",),
            command_hz=120.0,
            reference_hz=60.0,
            max_duration_sec=1.0,
        )
    )
    metadata = {
        "singularity_gate_status": "allow_position_elbow_singular",
        "target_singularity_code": -2,
        "target_singularity_label": "elbow_singularity",
        "current_singularity_code": 0,
        "current_singularity_label": "normal",
        "position_elbow_singularity_fallback_requested": True,
        "position_elbow_singularity_fallback_allowed": True,
        "position_singularity_fallback": {"fully_non_singular_count": 0},
        "position_candidate_certificate_validation": {
            "status": "matched",
            "valid": True,
            "sequence": 8,
            "actual_qref_fingerprint": "a" * 32,
            "certificate": {
                "final_qref_position_singularity": {
                    "sides": {
                        "left": {
                            "is_elbow_singular": True,
                            "is_wrist_singular": False,
                            "is_shoulder_singular": False,
                        }
                    }
                }
            },
        },
    }
    assert recorder.record_singularity_transition(
        "left",
        12,
        10.0,
        metadata,
        np.full(6, 0.1),
        np.zeros(6),
    )
    assert list(recorder.output_dir.iterdir()) == []
    recorder.close()

    with np.load(
        recorder.output_dir / ACTUATOR_COMMAND_TRACE_DATA_NAME,
        allow_pickle=False,
    ) as data:
        rows = data["left_singularity"]
        assert rows.shape == (1,)
        assert rows["target_sequence"].tolist() == [12]
        assert rows["fallback_requested_mask"].tolist() == [2]
        assert rows["fallback_allowed_mask"].tolist() == [2]
        assert rows["selected_singular_mask"].tolist() == [2]
        assert rows["candidate_certificate_valid"].tolist() == [1]
        assert rows["gate_status"][0].decode() == "allow_position_elbow_singular"


def test_preallocated_trace_persists_fixed_field_ik_shadow_on_close(
    tmp_path: Path,
) -> None:
    recorder = ActuatorCommandTraceRecorder(
        ActuatorCommandTraceConfig(
            output_dir=tmp_path / "ik_shadow",
            active_sides=("left", "right"),
            command_hz=120.0,
            reference_hz=60.0,
            max_duration_sec=1.0,
        )
    )
    pre_q = np.linspace(0.0, 1.1, 12)
    raw_q = pre_q + 0.01
    canonical_q = pre_q + 0.005
    admitted_q = pre_q + 0.002
    left_target = np.eye(4)
    right_target = np.eye(4)
    right_target[1, 3] = 0.2

    assert recorder.record_ik_shadow(
        12.5,
        pre_q,
        raw_q,
        canonical_q,
        admitted_q,
        left_target,
        right_target,
        np.arange(6, dtype=float),
        np.arange(6, dtype=float) + 1.0,
        0.0,
        0.5,
        "tracking",
    )
    assert list(recorder.output_dir.iterdir()) == []
    recorder.close()

    with np.load(
        recorder.output_dir / ACTUATOR_COMMAND_TRACE_DATA_NAME,
        allow_pickle=False,
    ) as data:
        rows = data["ik_shadow"]
        assert rows.shape == (1,)
        np.testing.assert_allclose(rows["pre_q_rad"][0], pre_q)
        np.testing.assert_allclose(rows["raw_solver_q_rad"][0], raw_q)
        np.testing.assert_allclose(rows["canonical_solver_q_rad"][0], canonical_q)
        np.testing.assert_allclose(rows["admitted_q_rad"][0], admitted_q)
        np.testing.assert_allclose(
            rows["target_world_pose"][0, 1],
            right_target,
        )
        np.testing.assert_allclose(
            rows["wrist_null_scale"][0],
            [0.0, 0.5],
        )
        assert rows["status"][0].decode() == "tracking"


def test_backend_singularity_trace_is_sparse_for_normal_and_dense_for_abnormal() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
        )
    )
    observed: list[tuple[object, ...]] = []
    backend.set_singularity_transition_trace_sink(
        lambda *sample: observed.append(sample)
    )
    normal = {
        "singularity_metadata": {
            "singularity_gate_status": "pass",
            "target_singularity_code": 0,
            "current_singularity_code": 0,
        }
    }
    abnormal = {
        "singularity_metadata": {
            "singularity_gate_status": "hold_entry",
            "target_singularity_code": -2,
            "current_singularity_code": 0,
        }
    }
    for sequence in (1, 2):
        backend._emit_singularity_transition_trace(
            target_sequence=sequence,
            monotonic_time=float(sequence),
            qref=np.zeros(6),
            reference_qpos=np.zeros(6),
            command_metadata=normal,
        )
    for sequence in (3, 4):
        backend._emit_singularity_transition_trace(
            target_sequence=sequence,
            monotonic_time=float(sequence),
            qref=np.zeros(6),
            reference_qpos=np.zeros(6),
            command_metadata=abnormal,
        )

    assert [int(sample[1]) for sample in observed] == [1, 3, 4]


def test_preallocated_trace_drops_after_capacity_without_growing(tmp_path: Path) -> None:
    recorder = ActuatorCommandTraceRecorder(
        ActuatorCommandTraceConfig(
            output_dir=tmp_path / "bounded",
            active_sides=("left",),
            command_hz=0.1,
            max_duration_sec=0.1,
        )
    )
    assert recorder.config.capacity_per_side == 1

    assert _record_one(recorder, tick=1)
    assert not _record_one(recorder, tick=2)
    assert recorder.snapshot()["drop_count_by_side"] == {"left": 1}
    recorder.close()


def test_rm65_trace_hook_observes_post_sdk_command_and_is_failure_isolated() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            singularity_protection_enabled=False,
            background_streaming_enabled=True,
            background_streaming_hz=120.0,
            background_streaming_target_max_age_sec=1.0,
        )
    )
    arm = _FakeArm()
    backend.arm = arm
    observed: list[tuple[object, ...]] = []

    def trace_sink(*sample: object) -> None:
        observed.append(sample)
        raise RuntimeError("synthetic trace failure")

    def audit_sink(event: object) -> None:
        if isinstance(event, PositionCommandAuditEvent):
            backend._stream_stop_event.set()

    backend.set_actuator_command_trace_sink(trace_sink)
    backend.set_position_audit_sink(audit_sink)
    backend.send_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.full(6, 0.1),
        )
    )

    backend._stream_stop_event.clear()
    backend._background_stream_loop_impl()

    assert len(arm.sent_joint_deg) == 1
    assert len(observed) == 1
    assert observed[0][0] == "left"
    assert observed[0][12] == 0
    assert observed[0][13] is True
    state = backend.get_state()["background_streaming"]
    assert state["actuator_command_trace_error_count"] == 1
    assert "synthetic trace failure" in state["last_actuator_command_trace_error"]
    assert state["fault"] is None


def test_rm65_command_worker_skips_audit_fk_when_no_audit_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            singularity_protection_enabled=False,
            background_streaming_enabled=True,
            background_streaming_hz=120.0,
            background_streaming_target_max_age_sec=1.0,
        )
    )
    arm = _FakeArm()
    backend.arm = arm

    def stop_after_first_command(*_sample: object) -> None:
        backend._stream_stop_event.set()

    def unexpected_audit_fk(_q_cmd: np.ndarray) -> dict[str, object]:
        pytest.fail("accepted-qcmd audit FK ran without a position-audit sink")

    backend.set_actuator_command_trace_sink(stop_after_first_command)
    monkeypatch.setattr(
        backend,
        "_accepted_qcmd_fk_audit_metadata",
        unexpected_audit_fk,
    )
    backend.send_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.full(6, 0.1),
        )
    )

    backend._stream_stop_event.clear()
    backend._background_stream_loop_impl()

    assert len(arm.sent_joint_deg) == 1
    assert backend.get_state()["background_streaming"]["audit_sink_attached"] is False


def test_actuator_trace_cli_resolves_formal_position_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "teleop",
            "--arm-control-mode",
            "position",
            "--position-control-profile",
            "rm65_bimanual_position_v1",
            "--control-debug",
            "off",
            "--actuator-trace",
            "--no-device-preflight",
            "--no-camera",
        ],
    )

    args = real_teleop.parse_args()

    assert args.arm_control_mode == "position"
    assert args.position_control_profile_resolved == "rm65_bimanual_position_v1"
    assert args.control_debug_config.level == "off"
    assert args.actuator_trace is True
    assert args.actuator_singularity_trace is True
    assert args.position_ik_candidate_certification_reserve_ms == pytest.approx(3.5)
    assert args.position_singularity_candidate_certificate is True
    assert args.no_collection is False
    assert args.arm_execution_config.position_background_streaming_hz == 120.0
    metadata = real_teleop._debug_timing_metadata(args)
    assert metadata["actuator_trace"] is True
    assert metadata["actuator_singularity_trace"] is True
    assert "no queue/json/fk/disk_write" in metadata["actuator_trace_hot_path_policy"]
