from __future__ import annotations

import numpy as np

from hardware.quest_dev.features.visualization.realman_feedback import (
    RealManArmSpec,
    RealManFeedbackReader,
    _build_backend,
    parse_realman_arm_specs,
)
from hardware.quest_dev.tools.read_realman_feedback import build_arg_parser


def _spec() -> RealManArmSpec:
    return RealManArmSpec(
        robot_id="rm65_left",
        side="left",
        arm_ip="192.0.2.10",
        arm_urdf_path="/tmp/rm65-arm.urdf",
    )


def test_read_probe_requires_explicit_realman_endpoint_and_urdf() -> None:
    args = build_arg_parser().parse_args(
        [
            "--side",
            "right",
            "--arm-ip",
            "192.0.2.20",
            "--arm-urdf",
            "/tmp/rm65-arm.urdf",
        ]
    )

    assert args.side == "right"
    assert args.arm_ip == "192.0.2.20"
    assert args.arm_urdf.name == "rm65-arm.urdf"
    assert args.samples == 3


def test_parse_realman_config_is_direct_and_has_no_cache_transport() -> None:
    specs = parse_realman_arm_specs(
        {
            "arms": {
                "rm65_left": {
                    "enabled": True,
                    "side": "left",
                    "arm_ip": "192.0.2.10",
                    "arm_urdf_path": "/tmp/rm65-arm.urdf",
                },
                "rm65_right": {
                    "enabled": False,
                    "side": "right",
                    "arm_ip": "192.0.2.11",
                },
            }
        }
    )

    assert specs == (_spec(),)


def test_backend_factory_enforces_observation_only_rm65_profile() -> None:
    backend = _build_backend(_spec())

    assert backend.config.connect_hardware is True
    assert backend.config.observation_only is True
    assert backend.config.background_streaming_enabled is False
    assert backend.config.force_position_control is False
    assert backend.config.configure_singularity_thresholds is False
    assert backend.config.canfd_follow is False


def test_state_record_is_fresh_but_not_quest_aligned_without_reviewed_binding() -> None:
    reader = RealManFeedbackReader(
        (_spec(),),
        max_age_s=0.25,
        monotonic_ns_clock=lambda: 1_000_000_050,
    )
    state = {
        "hardware_read_success": True,
        "hardware_result_text": "0",
        "hardware_read_timestamp_monotonic": 1.0,
        "hardware_measurement_sequence": 3,
        "current_tcp_pose": np.eye(4),
    }

    record = reader._state_record(
        _spec(),
        state,
        now_monotonic_ns=1_000_000_050,
    )

    assert record["source"] == "realman_rm65_direct"
    assert record["fresh"] is True
    assert record["control_ready"] is False
    assert record["alignment_ready"] is False
    assert "matrix_robot_from_tcp" in record
    assert "matrix_quest_from_tcp" not in record


def test_failed_realman_read_never_reuses_old_pose() -> None:
    reader = RealManFeedbackReader(
        (_spec(),),
        max_age_s=0.25,
        monotonic_ns_clock=lambda: 1_000_000_050,
    )

    record = reader._state_record(
        _spec(),
        {
            "hardware_read_success": False,
            "hardware_result_text": "timeout",
            "current_tcp_pose": np.eye(4),
        },
        now_monotonic_ns=1_000_000_050,
    )

    assert record["fresh"] is False
    assert record["error"] == "hardware_read_failed:timeout"
    assert "matrix_robot_from_tcp" not in record


def test_poll_calls_the_direct_backend_and_publishes_rm65_identity() -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.refresh_count = 0
            self.closed = False

        def start(self) -> None:
            return None

        def refresh_state_from_hardware(self, **_kwargs):
            self.refresh_count += 1
            return {
                "hardware_read_success": True,
                "hardware_result_text": "0",
                "hardware_read_timestamp_monotonic": 1.0,
                "hardware_measurement_sequence": self.refresh_count,
                "current_tcp_pose": np.eye(4),
            }

        def close(self) -> None:
            self.closed = True

    fake = FakeBackend()
    reader = RealManFeedbackReader(
        (_spec(),),
        max_age_s=0.25,
        monotonic_ns_clock=lambda: 1_000_000_050,
        backend_factory=lambda _spec: fake,
    )
    reader.start()
    message = reader.poll()
    reader.stop()

    assert fake.refresh_count == 1
    assert fake.closed is True
    item = message["robots"]["rm65_left"]
    assert item["source"] == "realman_rm65_direct"
    assert item["fresh"] is True
    assert item["control_ready"] is False
