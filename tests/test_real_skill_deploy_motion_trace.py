from __future__ import annotations

import json
import stat
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.runtime import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.skill_deploy import (
    REAL_AGENT_MOTION_COMMAND_METADATA_KEY,
    RealSkillControlTraceSample,
    RealSkillMotionDispatchReceipt,
    RealSkillMotionObservationCut,
)
from teleop_runtime.skill_deploy.motion_trace import (
    SkillMotionTraceConfig,
    SkillMotionTraceRecorder,
    inspect_trace,
)


def _command(*, index: int) -> TeleopCommand:
    target = np.eye(4)
    target[2, 3] = 0.20
    command = TeleopCommand(timestamp=10.0 + index * 0.01)
    command.metadata.update(
        {
            REAL_AGENT_MOTION_COMMAND_METADATA_KEY: {
                "operation_id": "operation:long-move",
                "request_id": "request:long-move",
                "side": "left",
                "speed_scale": 0.2,
                "target_tool_tcp_pose_world": target,
            },
            "collision_avoidance_status": "tracking",
            "collision_distance_m": 0.08,
            "ik_branch_candidate": "original",
            "ik_solve_iterations": 2,
            "ik_frame_total_ms": 9.0 + index,
            "ik_time_budget_ms": 16.667,
            "ik_time_budget_exceeded": index == 2,
            "ik_debug_profile": {
                "solve_calls": index + 1,
                "solve_total_ms": 4.0 + index,
                "solve_qp_ms": 2.0 + index,
                "solve_integrate_ms": 0.4 + index,
                "solve_set_configuration_ms": 0.2 + index,
                "min_distance_calls": index + 2,
                "min_distance_total_ms": 3.0 + index,
            },
            "ik_jump_gate_status": "pass",
        }
    )
    for side_index, side in enumerate(("left", "right")):
        pose = np.eye(4)
        pose[2, 3] = 0.10 + 0.01 * index
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=np.full(6, 0.01 * index + side_index * 0.1),
                metadata={
                    "q_ik_raw": np.full(6, 0.012 * index + side_index * 0.1),
                    "target_pose_tcp_base": pose,
                    "joint_limit_margin": {"minimum_margin_deg": 20.0},
                },
            )
        )
    return command


def _sample(*, index: int) -> RealSkillControlTraceSample:
    measured = {
        "left": np.full(6, 0.008 * index),
        "right": np.full(6, 0.1),
    }
    cut = RealSkillMotionObservationCut(
        observation=TeleopObservation(
            timestamp=10.0 + index * 0.01,
            metadata={
                "hardware_measurement_span_sec": 0.002,
                "hardware_read_phase_timing_by_side": {
                    side: {
                        "total_wall_ms": 8.0 + index + side_index,
                        "total_thread_cpu_ms": 0.5,
                        "hardware_read_lock_wait_wall_ms": 0.1,
                        "hardware_read_lock_wait_thread_cpu_ms": 0.01,
                        "sdk_call_wall_ms": 7.0 + index + side_index,
                        "sdk_call_thread_cpu_ms": 0.3,
                        "parse_validation_wall_ms": 0.2,
                        "parse_validation_thread_cpu_ms": 0.1,
                        "retry_sleep_wall_ms": 0.0,
                        "attempt_count": 1.0,
                    }
                    for side_index, side in enumerate(("left", "right"))
                },
            },
        ),
        qpos_by_side=measured,
        measured_at=10.0 + index * 0.01,
        available_at=10.001 + index * 0.01,
        evidence_refs=(f"hardware:{index}",),
        source_sequence_by_side={"left": index + 1, "right": index + 1},
    )
    measured_pose = np.eye(4)
    measured_pose[2, 3] = 0.10 + 0.02 * index
    started = 100.0 + index * 0.02
    receipt = RealSkillMotionDispatchReceipt(
        succeeded=True,
        evidence_refs=(f"dispatch:{index}",),
        diagnostics={
            "backend_streaming_by_side": {
                side: {
                    "target_sequence": index + 4,
                    "actual_hz": 118.0 - index,
                    "last_send_dt_sec": 0.0085,
                    "last_call_ms": 0.4,
                    "last_deadline_lag_ms": 0.2,
                    "deadline_miss_count": index,
                    "skipped_tick_count": index,
                    "low_rate_count": index,
                }
                for side in ("left", "right")
            }
        },
    )
    return RealSkillControlTraceSample(
        cycle_sequence=index + 1,
        cycle_started_monotonic=started,
        control_time=10.002 + index * 0.01,
        observation_completed_monotonic=started + 0.001,
        synchronization_completed_monotonic=started + 0.002,
        owner_requests_completed_monotonic=started + 0.003,
        ik_completed_monotonic=started + 0.012,
        admission_completed_monotonic=started + 0.013,
        dispatch_ready_at=10.014 + index * 0.01,
        dispatch_completed_monotonic=started + 0.015,
        cycle_finished_monotonic=started + 0.016,
        cut=cut,
        measured_pose_cut=SimpleNamespace(
            revision=index,
            pose_world_by_component={
                "left_arm": measured_pose,
                "right_arm": np.eye(4),
            },
        ),
        command=_command(index=index),
        dispatch_receipt=receipt,
        port_snapshot=SimpleNamespace(phase=SimpleNamespace(value="active")),
        command_present=True,
        admitted=True,
        dispatch_attempted=True,
        dispatch_succeeded=True,
        outcome="dispatch_succeeded",
    )


def test_skill_motion_trace_records_aligned_reference_ik_and_actuator_rows(
    tmp_path,
) -> None:
    output = tmp_path / "trace"
    recorder = SkillMotionTraceRecorder(
        SkillMotionTraceConfig(
            output_dir=output,
            reference_hz=60.0,
            command_hz=120.0,
            measurement_hz=30.0,
            max_duration_sec=1.0,
            metadata={"test": "long-distance"},
        )
    )
    for index in range(3):
        assert recorder.record_control_cycle(_sample(index=index))
        assert recorder.record_ik_shadow(
            100.01 + index * 0.02,
            np.full(12, index * 0.01),
            np.full(12, index * 0.011),
            np.full(12, index * 0.0105),
            np.full(12, index * 0.01),
            np.eye(4),
            np.eye(4),
            np.arange(6, dtype=float),
            np.arange(6, dtype=float),
            1.0,
            1.0,
            "tracking",
        )
        for side in ("left", "right"):
            assert recorder.record_actuator_command(
                side,
                index + 1,
                index + 4,
                1,
                index + 1,
                100.0 + index * 0.01,
                100.0002 + index * 0.01,
                index + 1,
                10.0 + index * 0.01,
                100.0003 + index * 0.01,
                100.0007 + index * 0.01,
                np.full(6, index * 0.005),
                0,
                True,
                index + 1,
            )

    result = recorder.close()

    assert result["closed"] is True
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    with np.load(output / "motion_trace.npz") as arrays:
        assert arrays["reference"].shape == (3,)
        assert arrays["ik_shadow"].shape == (3,)
        assert arrays["left_actuator"].shape == (3,)
        assert arrays["right_actuator"].shape == (3,)
        assert arrays["reference"][2]["ik_time_budget_exceeded"] == 1
        assert arrays["reference"][2]["ik_profile_solve_calls"] == 3
        assert arrays["reference"][2]["ik_profile_solve_qp_ms"] == pytest.approx(
            4.0
        )
        assert arrays["reference"][2][
            "ik_profile_min_distance_total_ms"
        ] == pytest.approx(5.0)
        assert arrays["reference"][1]["measurement_span_ms"] == pytest.approx(2.0)
        np.testing.assert_allclose(
            arrays["reference"][1]["hardware_read_sdk_call_wall_ms"],
            [8.0, 9.0],
        )
        np.testing.assert_allclose(
            arrays["reference"][1]["target_tool_tcp_pose_world"][:3, 3],
            [0.0, 0.0, 0.2],
        )
    inspected = inspect_trace(output)
    summary = inspected["summary"]
    assert summary["operation_ids"] == ["operation:long-move"]
    assert summary["dispatch_succeeded_count"] == 3
    assert summary["ik_time_budget_exceeded_count"] == 1
    assert summary["ik_branch_switch_count"] == 0
    assert summary["ik_profile"]["profiled_reference_count"] == 3
    assert summary["ik_profile"]["solve_qp_ms"]["max"] == pytest.approx(4.0)
    assert summary["ik_profile"]["min_distance_total_ms"][
        "max"
    ] == pytest.approx(5.0)
    assert summary["actuator_timing"]["left"]["accepted_count"] == 3
    assert summary["actuator_timing"]["left"]["post_stall_burst_count"] == 0
    assert summary["actuator_timing"]["left"]["moving_send_under_4ms_count"] == 0
    assert summary["measurement_timing"]["target_hz"] == 30.0
    assert summary["measurement_timing"]["unique_measurement_count"] == 3
    assert summary["measurement_timing"]["achieved_hz"] == pytest.approx(100.0)
    assert summary["measurement_timing"]["stall_count"] == 0
    assert summary["measurement_timing"]["hardware_read_timing_by_side"][
        "left"
    ]["sdk_call_wall_ms"]["max"] == pytest.approx(9.0)
    assert np.isclose(
        summary["target_vs_measured_tcp"]["final_translation_error_m"],
        0.06,
    )
    assert json.loads((output / "manifest.json").read_text())["result"][
        "closed"
    ] is True
    assert json.loads((output / "manifest.json").read_text())["schema"].endswith(
        ".v2"
    )


def test_skill_motion_trace_overflow_drops_diagnostics_without_growing(
    tmp_path,
) -> None:
    recorder = SkillMotionTraceRecorder(
        SkillMotionTraceConfig(
            output_dir=tmp_path / "bounded",
            reference_hz=1.0,
            command_hz=1.0,
            max_duration_sec=0.01,
        )
    )
    capacity = recorder.config.reference_capacity
    for index in range(capacity):
        assert recorder.record_control_cycle(_sample(index=index))
    assert recorder.record_control_cycle(_sample(index=capacity)) is False
    assert recorder.snapshot()["drop_count"]["reference"] == 1
    recorder.close()


def test_motion_trace_summary_ignores_missing_or_non_monotonic_timestamps(
    tmp_path,
) -> None:
    recorder = SkillMotionTraceRecorder(
        SkillMotionTraceConfig(
            output_dir=tmp_path / "invalid-time",
            reference_hz=1.0,
            command_hz=2.0,
            max_duration_sec=1.0,
        )
    )
    q = np.zeros(6)
    assert recorder.record_actuator_command(
        "left", 1, 1, 1, 1, 2.0, 2.1, 1, None, None, None, q, None, None, None
    )
    assert recorder.record_actuator_command(
        "left", 2, 1, 1, 1, 1.0, 1.1, 2, 10.0, 1.2, 1.3, q, 0, True, 1
    )
    assert recorder.record_actuator_command(
        "left", 3, 1, 1, 1, 1.2, 1.3, 3, 10.2, 1.4, 1.5, q, 0, True, 2
    )

    recorder.close()

    timing = inspect_trace(tmp_path / "invalid-time")["summary"][
        "actuator_timing"
    ]["left"]
    assert timing["sdk_call_ms"]["max"] == pytest.approx(100.0)
    assert timing["wakeup_lag_ms"]["max"] == pytest.approx(100.0)
    assert timing["send_gap_ms"]["max"] == pytest.approx(200.0)

    dynamics = SkillMotionTraceRecorder._dynamics(
        np.asarray([[0.0], [100.0], [1.0], [2.0]]),
        np.asarray([1_000_000_000, 1_000_000_000, 900_000_000, 2_000_000_000]),
    )
    assert dynamics["max_joint_step_deg"] == pytest.approx(np.degrees(2.0))


def test_motion_trace_summary_counts_post_stall_moving_burst(tmp_path) -> None:
    recorder = SkillMotionTraceRecorder(
        SkillMotionTraceConfig(
            output_dir=tmp_path / "post-stall-burst",
            reference_hz=60.0,
            command_hz=120.0,
            max_duration_sec=1.0,
        )
    )
    for sequence, result_time in enumerate((1.0, 1.020, 1.021, 1.030), start=1):
        assert recorder.record_actuator_command(
            "left",
            sequence,
            1,
            1,
            sequence,
            result_time - 0.002,
            result_time - 0.0015,
            sequence,
            10.0 + result_time,
            result_time - 0.001,
            result_time,
            np.full(6, sequence * 0.001),
            0,
            True,
            sequence,
        )
    recorder.close()

    timing = inspect_trace(tmp_path / "post-stall-burst")["summary"][
        "actuator_timing"
    ]["left"]
    assert timing["post_stall_minimum_gap_ms"] == pytest.approx(1.0)
    assert timing["post_stall_burst_count"] == 1
    assert timing["moving_send_under_4ms_count"] == 1
