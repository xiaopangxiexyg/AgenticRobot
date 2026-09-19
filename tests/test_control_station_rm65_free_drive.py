from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from control_station.sources.rm65_free_drive import (
    RM65FreeDriveRealtimePushSource,
    RM65FreeDriveSource,
    RM65FreeDriveSourceSpec,
)


def _state(
    *,
    sequence: int = 1,
    observation_only: bool = True,
    measurement_valid: bool = True,
    read_success: bool = True,
    measurement_timestamp: float = 10.0,
    receive_timestamp: float = 10.0,
    receive_monotonic: float = 20.0,
    pose: object | None = None,
) -> dict[str, object]:
    return {
        "observation_only": observation_only,
        "arm_state_source": "hardware_observation_only",
        "hardware_measurement_valid": measurement_valid,
        "hardware_read_success": read_success,
        "hardware_measurement_sequence": sequence,
        "hardware_measurement_timestamp": measurement_timestamp,
        "hardware_read_time": receive_timestamp,
        "hardware_read_timestamp_monotonic": receive_monotonic,
        "hardware_measurement_clock_domain_id": "local_wall_time",
        "hardware_qpos": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        "current_tcp_pose": np.eye(4) if pose is None else pose,
        "background_streaming": {"enabled": False},
    }


class _Backend:
    def __init__(self, states: list[dict[str, object]], *, observation_only: bool = True):
        self.states = list(states)
        self.config = SimpleNamespace(observation_only=observation_only)
        self.calls: list[dict[str, object]] = []

    def refresh_state_from_hardware(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return self.states.pop(0)


class _PushBackend(_Backend):
    def __init__(self) -> None:
        super().__init__([_state()])
        self.push_callback = None

    def compute_tcp_pose(self, *, qpos):
        assert len(qpos) == 6
        return np.eye(4)

    def start_realtime_push_observation(self, callback, **kwargs):
        self.push_callback = callback
        callback(
            {
                "joint_deg": (0.0, 5.72957795, 11.4591559, 17.18873385, 22.9183118, 28.64788976),
                "err_code": 0,
                "arm_error": 0,
                "system_error": 0,
            }
        )
        return {"cycle": 5, "enable": True, "port": 8080}


def _source(
    backend: _Backend,
    *,
    wall: float = 10.1,
    monotonic: float = 20.1,
) -> RM65FreeDriveSource:
    return RM65FreeDriveSource(
        backend,
        spec=RM65FreeDriveSourceSpec(
            maximum_measurement_age_sec=0.25,
            maximum_receive_age_sec=0.25,
        ),
        wall_clock=lambda: wall,
        monotonic_clock=lambda: monotonic,
    )


def test_source_refreshes_once_and_emits_observation_only_sample() -> None:
    backend = _Backend([_state()])
    step = _source(backend).step()

    assert step.admitted
    assert step.sample is not None
    assert step.sample.sequence == 1
    assert step.sample.source_reference_frame_id == "right_arm_base"
    assert step.sample.source_controlled_frame_id == "tool_tcp"
    assert backend.calls == [
        {
            "include_derived_pose": True,
            "include_timing_percentiles": False,
            "include_pose_source_attestation": False,
        }
    ]


def test_source_rejects_non_observation_backend_without_refresh() -> None:
    backend = _Backend([_state()], observation_only=False)
    step = _source(backend).step()

    assert not step.admitted
    assert step.rejection_reason == "backend_not_observation_only"
    assert backend.calls == []


def test_source_rejects_invalid_state_and_does_not_reuse_previous_sample() -> None:
    backend = _Backend([_state(), _state(sequence=1, measurement_valid=False)])
    source = _source(backend)

    first = source.step()
    second = source.step()

    assert first.admitted
    assert not second.admitted
    assert "hardware measurement is not valid" in (second.rejection_reason or "")
    assert source.last_measurement_sequence == 1


def test_source_rejects_stale_receive_cut() -> None:
    backend = _Backend([_state(receive_monotonic=19.0)])
    step = _source(backend).step()

    assert not step.admitted
    assert "hardware read is stale" in (step.rejection_reason or "")


def test_source_rejects_non_rigid_pose_without_fallback() -> None:
    bad_pose = np.eye(4)
    bad_pose[0, 0] = 2.0
    backend = _Backend([_state(pose=bad_pose)])
    step = _source(backend).step()

    assert not step.admitted
    assert "current_tcp_pose is invalid" in (step.rejection_reason or "")


def test_realtime_push_source_admits_complete_callback_without_sdk_poll() -> None:
    backend = _PushBackend()
    source = RM65FreeDriveRealtimePushSource(
        backend,
        spec=RM65FreeDriveSourceSpec(
            maximum_measurement_age_sec=0.25,
            maximum_receive_age_sec=0.25,
        ),
        wall_clock=lambda: 10.1,
        monotonic_clock=lambda: 20.1,
    )
    source.start()
    step = source.step()

    assert step.admitted
    assert step.sample is not None
    assert step.sample.sequence == 1
    assert np.allclose(step.sample.qpos_rad, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5))
    assert backend.calls == []
    source.close()


def test_realtime_push_source_rejects_duplicate_latest_packet() -> None:
    backend = _PushBackend()
    source = RM65FreeDriveRealtimePushSource(
        backend,
        wall_clock=lambda: 10.1,
        monotonic_clock=lambda: 20.1,
    )
    source.start()
    assert source.step().admitted
    duplicate = source.step()
    assert not duplicate.admitted
    assert "measurement_not_advancing" in (duplicate.rejection_reason or "")


def test_realtime_push_source_fails_closed_on_vendor_error() -> None:
    backend = _PushBackend()
    source = RM65FreeDriveRealtimePushSource(
        backend,
        wall_clock=lambda: 10.1,
        monotonic_clock=lambda: 20.1,
    )
    source.start()
    assert backend.push_callback is not None
    backend.push_callback(
        {
            "joint_deg": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "err_code": 9,
            "arm_error": 0,
            "system_error": 0,
        }
    )
    step = source.step()
    assert not step.admitted
    assert "vendor error" in (step.rejection_reason or "")
