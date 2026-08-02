from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from control_station.composition import MujocoSimHILReadinessEvidence
from control_station.sim_hil import (
    DEFAULT_VIVE_COORDINATE_PROFILE,
    MujocoSimHILApplication,
    MujocoSimHILApplicationConfig,
    SIM_HIL_STATUS_SCHEMA_V1,
    _acceptance_profile_config,
    _advance_control_deadline,
    _bind_vive_coordinate_profile,
    _build_sim_hil_vive_setup_controller,
    _config_from_args,
    _vive_receiver_status,
    build_arg_parser,
)
from control_station.sim_hil_gui import SimHILStatusServer
from control_station.vive_calibration import (
    ViveCoordinateCalibrationController,
)
from control_station.vive_calibration_profiles import (
    new_coordinate_calibration,
    save_coordinate_profile,
)
from robot_motion.action import (
    TargetExecutionStageStatus,
    TargetExecutionStatusSnapshot,
)
from teleop_runtime.calibration import ArmCalibration, TeleopCalibration
from teleop_runtime.sources.pose.vive import VivePoseSource, ViveTrackerConfig


def _single_vive_sender_packet() -> bytes:
    return json.dumps(
        {
            "protocol": "teleop_runtime.vive/v1",
            "seq": 17,
            "sender_timestamp": 1000.0,
            "trackers": [
                {
                    "tracker_name": "physical-single",
                    "tracker_role": "left_wrist",
                    "valid": True,
                    "pose_matrix_4x4": np.eye(4).tolist(),
                    "error": None,
                },
                {
                    "tracker_name": "synthetic-right",
                    "tracker_role": "right_wrist",
                    "valid": False,
                    "pose_matrix_4x4": np.eye(4).tolist(),
                    "error": "single_tracker_mode:not_connected",
                },
            ],
        }
    ).encode("utf-8")


def test_sim_hil_vive_status_shows_both_sender_roles_before_mapping() -> None:
    source = VivePoseSource(
        tracker_configs=[
            ViveTrackerConfig(
                name="physical-single",
                side="right",
                role="left_wrist",
                channel_name="right_wrist_pose",
            )
        ],
        host="127.0.0.1",
        port=9999,
        clock_sync_port=0,
    )
    source._running = True
    source._process_received_batch(
        [(_single_vive_sender_packet(), 100.0, 20.0, ("192.0.2.7", 4000))]
    )

    status = _vive_receiver_status(
        SimpleNamespace(sources=[source]),
        wall_now=100.1,
    )

    assert status["status"] == "receiving"
    assert status["packet_count"] == 1
    assert status["packet_age_wall_sec"] == pytest.approx(0.1)
    assert status["packet_age_monotonic_sec"] is not None
    assert status["channel_update_counts"] == {"right_wrist_pose": 1}
    assert status["trackers"]["left"]["status_level"] == "ok"
    assert status["trackers"]["left"]["selected_for_output_channels"] == [
        "right_wrist_pose"
    ]
    assert status["trackers"]["right"]["status_level"] == "invalid"
    assert (
        status["trackers"]["right"]["sender_error"]
        == "single_tracker_mode:not_connected"
    )

    source.swap_left_right_source_roles()
    swapped = _vive_receiver_status(
        SimpleNamespace(sources=[source]),
        wall_now=100.2,
    )
    assert swapped["trackers"]["left"]["selected_for_output_channels"] == []
    assert swapped["trackers"]["right"]["selected_for_output_channels"] == [
        "right_wrist_pose"
    ]


def test_sim_hil_rate_config_keeps_compose_rate_explicit_and_legacy_alias() -> None:
    default_config = MujocoSimHILApplicationConfig()
    assert default_config.position_control_profile_name == "position_15_60_120hz"
    assert default_config.compose_target_hz == 30.0
    assert default_config.control_rate_hz is None

    explicit = MujocoSimHILApplicationConfig(compose_target_hz=45.0)
    assert explicit.compose_target_hz == 45.0
    assert explicit.control_rate_hz is None

    # Station configuration is no longer the owner of the historical 60 Hz
    # ceiling.  The active deploy/authority capability may still reject a
    # rate it cannot execute; that gate is checked when the target is built.
    future_rate = MujocoSimHILApplicationConfig(compose_target_hz=120.0)
    assert future_rate.compose_target_hz == 120.0

    legacy = MujocoSimHILApplicationConfig(control_rate_hz=45.0)
    assert legacy.compose_target_hz == 45.0
    assert legacy.control_rate_hz == 45.0

    with pytest.raises(ValueError, match="compatibility alias"):
        MujocoSimHILApplicationConfig(
            compose_target_hz=45.0,
            control_rate_hz=60.0,
        )


def test_sim_hil_rate_cli_exposes_profile_and_compose_cadence() -> None:
    default_config = _config_from_args(build_arg_parser().parse_args([]))
    assert default_config.compose_target_hz == 30.0

    args = build_arg_parser().parse_args(
        [
            "--position-control-profile",
            "position",
            "--compose-hz",
            "45",
        ]
    )

    assert args.position_control_profile == "position"
    assert args.compose_hz == 45.0
    assert args.rate is None

    legacy_args = build_arg_parser().parse_args(["--rate", "45"])
    legacy_config = _config_from_args(legacy_args)
    assert legacy_config.compose_target_hz == 45.0
    assert legacy_config.control_rate_hz == 45.0


def test_policy_only_gui_setup_survives_vive_receiver_start_failure(
    monkeypatch,
) -> None:
    def fail_start(self, *, start_source=True):
        del self, start_source
        raise RuntimeError("synthetic VIVE bind failure")

    monkeypatch.setattr(
        ViveCoordinateCalibrationController,
        "start",
        fail_start,
    )

    setup = _build_sim_hil_vive_setup_controller(
        MujocoSimHILApplicationConfig(),
        source_role="left_wrist",
    )

    assert setup.snapshot()["source_started"] is False
    setup.close()


def test_sim_hil_control_gui_cli_is_explicit_and_has_a_capture_root() -> None:
    args = build_arg_parser().parse_args(
        [
            "--enable",
            "--control-gui",
            "--acceptance-root",
            "/tmp/sim-hil-acceptance",
            "--acceptance-device-timeout",
            "4.5",
        ]
    )

    assert args.enable is True
    assert args.acceptance_gui is True
    assert args.open_browser is True
    assert str(args.acceptance_root) == "/tmp/sim-hil-acceptance"
    assert args.acceptance_device_timeout == 4.5
    assert _config_from_args(args).mujoco_viewer_enabled is True
    assert (
        _config_from_args(args).vive_coordinate_profile_path
        == DEFAULT_VIVE_COORDINATE_PROFILE
        == Path("latest")
    )

    calibration_args = build_arg_parser().parse_args(
        [
            "--enable",
            "--control-gui",
            "--vive-calibration-gui",
            "--vive-calibration-ui-port",
            "8899",
            "--no-vive-calibration-open-browser",
        ]
    )
    assert calibration_args.vive_calibration_gui is True
    assert calibration_args.vive_calibration_ui_port == 8899
    assert calibration_args.vive_calibration_open_browser is False

    headless_args = build_arg_parser().parse_args(
        ["--enable", "--control-gui", "--no-mujoco-viewer"]
    )
    assert _config_from_args(headless_args).mujoco_viewer_enabled is False

    ordinary_args = build_arg_parser().parse_args(["--enable"])
    assert _config_from_args(ordinary_args).mujoco_viewer_enabled is False
    compatibility_args = build_arg_parser().parse_args(
        ["--enable", "--acceptance-gui"]
    )
    assert compatibility_args.acceptance_gui is True
    assert build_arg_parser().parse_args(["--no-open-browser"]).open_browser is False


def test_control_profile_base_alpha_is_only_a_factory_fallback() -> None:
    ordinary = MujocoSimHILApplicationConfig(
        enabled=True,
        initial_human_alpha=0.65,
        control_rate_hz=45.0,
        mujoco_viewer_enabled=True,
    )

    acceptance = _acceptance_profile_config(ordinary)

    assert acceptance.initial_human_alpha == 1.0
    assert acceptance.control_rate_hz == ordinary.control_rate_hz
    assert acceptance.mujoco_viewer_enabled is True
    assert ordinary.initial_human_alpha == 0.65


def test_failed_application_startup_preserves_primary_error_and_requires_rebuild() -> None:
    calls: list[str] = []

    class Viewer:
        @staticmethod
        def start():
            calls.append("viewer.start")

        @staticmethod
        def sync():
            calls.append("viewer.sync")

        @staticmethod
        def status():
            return {
                "enabled": True,
                "state": "running",
                "running": True,
                "sync_count": 0,
                "last_error": None,
            }

        @staticmethod
        def close():
            calls.append("viewer.close")

    class Policy:
        last_emission = None

        @staticmethod
        def start():
            calls.append("policy.start")

        @staticmethod
        def step(*, now):
            calls.append("policy.step")
            return SimpleNamespace()

        @staticmethod
        def close():
            calls.append("policy.close")

    class Human:
        started = False
        last_emission = None

        @staticmethod
        def start():
            Human.started = True
            calls.append("human.start")

        @staticmethod
        def set_calibration_ready():
            calls.append("human.calibration")

        @staticmethod
        def step(*, now):
            calls.append("human.step")
            return SimpleNamespace(emission=SimpleNamespace())

        @staticmethod
        def close():
            Human.started = False
            calls.append("human.close")

    class Composition:
        started = False

        @staticmethod
        def readiness_failures(_evidence, *, target_effective_at):
            return ()

        @staticmethod
        def start():
            raise RuntimeError("primary composition startup failure")

    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(enabled=True),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=Composition(),
        policy_source=Policy(),
        human_source=Human(),
        readiness_provider=SimpleNamespace(
            snapshot=lambda *, now: SimpleNamespace()
        ),
        viewer=Viewer(),
        clock=lambda: 10.0,
        sleep=lambda _duration: None,
    )

    with pytest.raises(RuntimeError, match="primary composition startup failure"):
        app.start()

    assert app.restart_required is True
    assert app._startup_failure == (
        "RuntimeError:primary composition startup failure"
    )
    assert calls == [
        "policy.start",
        "human.start",
        "human.calibration",
        "policy.step",
        "human.step",
        "viewer.start",
        "policy.step",
        "human.step",
        "human.close",
        "policy.close",
        "viewer.close",
    ]
    with pytest.raises(RuntimeError, match="must be rebuilt.*First startup failure"):
        app.start()
    assert calls.count("human.start") == 1


def test_viewer_starts_before_realtime_composition_worker() -> None:
    calls: list[str] = []
    report = SimpleNamespace(passed=True)
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(enabled=True),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(clock=lambda: 10.0),
        composition=SimpleNamespace(
            start=lambda: (
                calls.append("composition.start"),
                report,
            )[1],
        ),
        policy_source=SimpleNamespace(),
        human_source=SimpleNamespace(),
        readiness_provider=SimpleNamespace(),
        viewer=SimpleNamespace(
            start=lambda: calls.append("viewer.start"),
        ),
        clock=lambda: 10.0,
    )
    app._warmup_sources = lambda **_kwargs: calls.append("source.warmup")
    app._wait_for_source_readiness = lambda **_kwargs: calls.append(
        "source.refresh"
    )
    app._prime_activation_inputs = lambda: calls.append("activation.prime")

    assert app.start() is report

    assert calls == [
        "source.warmup",
        "viewer.start",
        "source.refresh",
        "composition.start",
        "activation.prime",
    ]
    assert app.started is True


def test_viewer_latency_reacquires_source_before_composition_preflight() -> None:
    now = [10.0]
    report = SimpleNamespace(passed=True)

    class Source:
        def __init__(self) -> None:
            self.last_emitted_at: float | None = None
            self.step_count = 0

        def start(self) -> None:
            return None

        def set_calibration_ready(self) -> None:
            return None

        def step(self, *, now: float) -> SimpleNamespace:
            self.last_emitted_at = now
            self.step_count += 1
            return SimpleNamespace(emission=SimpleNamespace())

    human = Source()

    class Composition:
        human_alpha_by_component = {"right_arm": 1.0}

        @staticmethod
        def readiness_failures(_evidence, *, target_effective_at: float):
            assert human.last_emitted_at is not None
            if target_effective_at - human.last_emitted_at > 0.2:
                return ("human_device_ready",)
            return ()

        @staticmethod
        def start():
            assert human.last_emitted_at == 10.4
            return report

    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(
            enabled=True,
            warmup_timeout_sec=0.5,
        ),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(clock=lambda: now[0]),
        composition=Composition(),
        policy_source=Source(),
        human_source=human,
        readiness_provider=SimpleNamespace(
            snapshot=lambda *, now: SimpleNamespace(
                detail_by_check={"human_device_ready": "fresh"}
            )
        ),
        viewer=SimpleNamespace(
            start=lambda: now.__setitem__(0, 10.4),
        ),
        clock=lambda: now[0],
        sleep=lambda _duration: None,
    )
    app._prime_activation_inputs = lambda: None

    assert app.start() is report
    assert human.step_count == 2


def test_source_warmup_observes_after_emission_step() -> None:
    """A synchronous source emission is checked at the post-step cut."""

    class Clock:
        def __init__(self) -> None:
            self.values = iter((10.0, 10.0, 10.005))

        def __call__(self) -> float:
            return next(self.values)

    class Source:
        def start(self) -> None:
            return None

        def set_calibration_ready(self) -> None:
            return None

        def step(self, *, now: float) -> None:
            return None

    class Readiness:
        def __init__(self) -> None:
            self.observed_at: list[float] = []

        def snapshot(self, *, now: float):
            self.observed_at.append(now)
            return SimpleNamespace(detail_by_check={"human_device_ready": "fresh"})

    class Composition:
        @staticmethod
        def readiness_failures(_evidence, *, target_effective_at: float):
            return () if target_effective_at >= 10.005 else ("human_device_ready",)

    clock = Clock()
    readiness = Readiness()
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(
            enabled=True,
            warmup_timeout_sec=0.1,
        ),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=Composition(),
        policy_source=Source(),
        human_source=Source(),
        readiness_provider=readiness,
        clock=clock,
        sleep=lambda _duration: None,
    )

    app._warmup_sources()

    assert readiness.observed_at == [10.005]


def test_human_only_warmup_does_not_run_unused_policy_inference() -> None:
    class Source:
        def __init__(self, *, fail_on_step: bool = False) -> None:
            self.fail_on_step = fail_on_step

        def start(self) -> None:
            return None

        def set_calibration_ready(self) -> None:
            return None

        def step(self, *, now: float) -> None:
            if self.fail_on_step:
                raise AssertionError("unused policy inference was executed")
            return None

    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(
            enabled=True,
            warmup_timeout_sec=0.1,
        ),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=SimpleNamespace(
            human_alpha_by_component={"right_arm": 1.0},
            readiness_failures=lambda _evidence, **_kwargs: (),
        ),
        policy_source=Source(fail_on_step=True),
        human_source=Source(),
        readiness_provider=SimpleNamespace(
            snapshot=lambda **_kwargs: SimpleNamespace(
                detail_by_check={"human_device_ready": "fresh"}
            )
        ),
        clock=lambda: 10.0,
        sleep=lambda _duration: None,
    )

    app._warmup_sources()


def test_partial_composition_startup_establishes_hold_before_cleanup() -> None:
    calls: list[str] = []

    class Policy:
        last_emission = None

        @staticmethod
        def start():
            calls.append("policy.start")

        @staticmethod
        def step(*, now):
            calls.append("policy.step")
            return SimpleNamespace()

        @staticmethod
        def reset_for_activation():
            raise RuntimeError("activation prime failed")

        @staticmethod
        def close():
            calls.append("policy.close")

    class Human:
        @staticmethod
        def start():
            calls.append("human.start")

        @staticmethod
        def set_calibration_ready():
            calls.append("human.calibration")

        @staticmethod
        def step(*, now):
            calls.append("human.step")
            return SimpleNamespace(emission=SimpleNamespace())

    class Composition:
        started = False
        human_alpha_by_component = {"right_arm": 0.0}

        @classmethod
        def readiness_failures(cls, _evidence, *, target_effective_at):
            return ()

        @classmethod
        def start(cls):
            calls.append("composition.start")
            cls.started = True
            return SimpleNamespace(passed=True)

        @classmethod
        def request_hold(cls, *, reason):
            calls.append(f"hold:{reason}")
            return SimpleNamespace()

        @classmethod
        def close(cls):
            calls.append("composition.close")
            cls.started = False

    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(enabled=True),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=Composition(),
        policy_source=Policy(),
        human_source=Human(),
        readiness_provider=SimpleNamespace(
            snapshot=lambda *, now: SimpleNamespace()
        ),
        clock=lambda: 10.0,
        sleep=lambda _duration: None,
    )

    with pytest.raises(RuntimeError, match="activation prime failed"):
        app.start()

    assert calls == [
        "policy.start",
        "human.start",
        "human.calibration",
        "policy.step",
        "human.step",
        "composition.start",
        "hold:sim_hil_application:shutdown",
        "composition.close",
        "policy.close",
    ]
    assert app.restart_required is True
    assert app.closed is True


def test_control_deadline_skips_overrun_without_burst_catchup() -> None:
    deadline, skipped = _advance_control_deadline(
        previous_deadline=10.0,
        completed_at=10.105,
        period=0.02,
    )
    assert skipped == 5
    assert deadline == pytest.approx(10.12)
    assert deadline > 10.105

    on_time, skipped = _advance_control_deadline(
        previous_deadline=10.0,
        completed_at=10.005,
        period=0.02,
    )
    assert skipped == 0
    assert on_time == pytest.approx(10.02)


def test_application_close_establishes_hold_before_lower_cleanup() -> None:
    order: list[str] = []

    class Composition:
        started = True
        activation_scope = None
        contracts = SimpleNamespace(
            composition_spec=SimpleNamespace(
                spec_id="composition",
                semantic_content_hash="composition-hash",
                required_roles=("human", "policy"),
            ),
            strategy_spec=SimpleNamespace(
                strategy_id="strategy",
                semantic_content_hash="strategy-hash",
            ),
            authority_profile=SimpleNamespace(
                profile_id="authority",
                semantic_content_hash="authority-hash",
            ),
        )

        @staticmethod
        def request_hold(*, reason):
            order.append(f"hold:{reason}")
            return SimpleNamespace()

        @staticmethod
        def close():
            order.append("composition.close")

    policy = SimpleNamespace(
        close=lambda: order.append("policy.close"),
        last_emission=None,
        status=lambda *, now: {},
    )
    human = SimpleNamespace(
        close=lambda: order.append("human.close"),
        last_emission=None,
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=Composition(),
        policy_source=policy,
        human_source=human,
        readiness_provider=SimpleNamespace(),
        viewer=SimpleNamespace(
            close=lambda: order.append("viewer.close"),
        ),
        clock=lambda: 10.0,
    )
    app._started = True

    app.close()

    assert order == [
        "hold:sim_hil_application:shutdown",
        "composition.close",
        "policy.close",
        "viewer.close",
    ]
    assert app.started is False


def test_runtime_home_recapture_is_passive_and_keeps_viewer_open() -> None:
    captured_anchors: list[tuple[np.ndarray, np.ndarray]] = []

    class Retargeter:
        config = SimpleNamespace(
            side="right",
            input_channel_name="right_wrist_pose",
        )

        @staticmethod
        def prepare_input_pose_desk(pose):
            raise AssertionError(
                "Runtime Home must pass raw desk-frame samples to set_home_anchors."
            )

        @staticmethod
        def set_home_anchors(*, human_pose_desk, robot_pose_desk):
            captured_anchors.append(
                (
                    np.asarray(human_pose_desk, dtype=float),
                    np.asarray(robot_pose_desk, dtype=float),
                )
            )

    recalibrated: list[str] = []
    calibration = SimpleNamespace(desk_frame=SimpleNamespace(name="desk"))
    session = SimpleNamespace(
        calibration=calibration,
        retargeters=(Retargeter(),),
        mark_home_recalibrated=lambda reason: recalibrated.append(reason),
    )
    source_steps: list[float] = []

    def source_step(*, now):
        source_steps.append(float(now))
        channel = SimpleNamespace(
            value=np.eye(4),
            frame_name="desk",
            metadata={},
            valid=True,
        )
        return SimpleNamespace(
            session_step=SimpleNamespace(
                channels={"right_wrist_pose": channel}
            ),
            command_dispatched=False,
            authoritative=False,
        )

    viewer_closes: list[str] = []
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(
            config=SimpleNamespace(clock_domain_id="test-monotonic")
        ),
        composition=SimpleNamespace(started=False),
        policy_source=SimpleNamespace(close=lambda: None),
        human_source=SimpleNamespace(
            session=session,
            step=source_step,
        ),
        readiness_provider=SimpleNamespace(),
        runtime_home_pose_desk_by_side_capture=lambda: {
            "right": np.eye(4)
        },
        viewer=SimpleNamespace(
            close=lambda: viewer_closes.append("closed")
        ),
        clock=lambda: 10.0,
        sleep=lambda _duration: None,
    )
    app._sources_started = True
    app._started = True
    app._held = True

    receipt = app.capture_runtime_home(samples_required=3)

    assert receipt["sample_count_by_side"] == {"right": 3}
    assert receipt["command_dispatched"] is False
    assert receipt["authoritative"] is False
    assert len(source_steps) == 3
    assert len(captured_anchors) == 1
    assert recalibrated == ["sim_hil_runtime_home_captured"]
    assert viewer_closes == []
    assert app.started is True
    assert app._held is True


def test_held_passive_poll_does_not_step_policy_or_composition() -> None:
    calls: list[str] = []
    passive_step = SimpleNamespace(
        command_dispatched=False,
        authoritative=False,
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=SimpleNamespace(
            started=True,
            step=lambda **_kwargs: calls.append("composition.step"),
        ),
        policy_source=SimpleNamespace(
            step=lambda **_kwargs: calls.append("policy.step"),
        ),
        human_source=SimpleNamespace(
            step=lambda **_kwargs: (
                calls.append("human.step") or passive_step
            )
        ),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )
    app._sources_started = True
    app._started = True
    app._held = True

    assert app.poll_passive_human_input(now=10.0) is passive_step
    assert calls == ["human.step"]
    assert app._last_human_input_check_step is passive_step


def test_application_latches_when_lower_gate_holds_or_hold_transition_fails() -> None:
    runtime = SimpleNamespace(
        clock=lambda: 10.0,
        config=SimpleNamespace(decision_hz=60.0),
        reference_activation_spec=SimpleNamespace(
            target_lead_time_sec=1.0 / 30.0
        ),
        capture_execution_gate_snapshot=lambda *, now: SimpleNamespace(
            status="hold"
        ),
    )
    composition = SimpleNamespace(
        step=lambda **_kwargs: SimpleNamespace(),
        request_hold=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("lower hold failed")
        ),
    )
    policy = SimpleNamespace(step=lambda *, now: None)
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=runtime,
        composition=composition,
        policy_source=policy,
        human_source=SimpleNamespace(),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )
    app._started = True

    app.step()
    assert app._held is True
    assert app._last_error == "ExecutionGateNotReady:hold:unspecified"
    assert app._first_control_fault == app._last_error
    with pytest.raises(RuntimeError, match="explicit rearm"):
        app.step()

    app._held = False
    with pytest.raises(RuntimeError, match="lower hold failed"):
        app.hold(reason="fault injection")
    assert app._held is True
    assert app._last_error == "RuntimeError:lower hold failed"


def test_application_turns_native_viewer_close_into_hold_and_exit_request() -> None:
    holds: list[str] = []
    runtime = SimpleNamespace(
        clock=lambda: 10.0,
        config=SimpleNamespace(decision_hz=30.0),
        reference_activation_spec=SimpleNamespace(
            activation_mode="immediate_target",
            target_lead_time_sec=0.0,
        ),
        capture_execution_gate_snapshot=lambda *, now: SimpleNamespace(
            status="ready"
        ),
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=runtime,
        composition=SimpleNamespace(
            human_alpha_by_component={"right_arm": 1.0},
            step=lambda **_kwargs: SimpleNamespace(),
            request_hold=lambda *, reason: holds.append(reason),
        ),
        policy_source=SimpleNamespace(),
        human_source=SimpleNamespace(),
        readiness_provider=SimpleNamespace(),
        viewer=SimpleNamespace(
            sync=lambda: False,
            status=lambda: {"state": "closed_by_user"},
        ),
        clock=lambda: 10.0,
    )
    app._started = True

    app.step(now=10.0)

    assert app.viewer_exit_requested is True
    assert app._held is True
    assert holds == [
        "sim_hil_application:mujoco_viewer_closed_by_user"
    ]


def test_application_uses_exact_target_grid_and_skips_missed_periods() -> None:
    targets: list[float] = []
    holds: list[str] = []

    def compose_step(**kwargs):
        targets.append(kwargs["target_effective_at"])
        return SimpleNamespace()

    runtime = SimpleNamespace(
        clock=lambda: 10.0,
        config=SimpleNamespace(decision_hz=60.0),
        reference_activation_spec=SimpleNamespace(
            target_lead_time_sec=1.0 / 30.0
        ),
        capture_execution_gate_snapshot=lambda *, now: SimpleNamespace(
            status="ready"
        ),
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=runtime,
        composition=SimpleNamespace(
            step=compose_step,
            request_hold=lambda *, reason: holds.append(reason),
        ),
        policy_source=SimpleNamespace(step=lambda *, now: None),
        human_source=SimpleNamespace(),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )
    app._started = True
    app._next_decision_at = 10.0

    app.step(now=10.0)
    app.step(now=10.017)

    assert targets == pytest.approx(
        [
            10.0 + 1.0 / 30.0,
            10.0 + 1.0 / 60.0 + 1.0 / 30.0,
        ]
    )
    app.step(now=10.1)
    assert holds == []
    assert app._held is False
    assert app._overrun_dropped_cycles >= 1
    assert app._last_schedule_overrun is not None
    assert (
        app._last_schedule_overrun["resolution"]
        == "skip_to_next_compose_grid"
    )
    assert targets[-1] > 10.1


def test_application_realigns_after_slow_policy_before_target_publication() -> None:
    runtime_now = [10.0]
    targets: list[float] = []

    def slow_policy_step(*, now: float):
        assert now == 10.0
        runtime_now[0] = 10.055
        return None

    runtime = SimpleNamespace(
        clock=lambda: runtime_now[0],
        config=SimpleNamespace(decision_hz=60.0),
        reference_activation_spec=SimpleNamespace(
            target_lead_time_sec=1.0 / 30.0
        ),
        capture_execution_gate_snapshot=lambda *, now: SimpleNamespace(
            status="ready"
        ),
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=runtime,
        composition=SimpleNamespace(
            step=lambda **kwargs: (
                targets.append(kwargs["target_effective_at"])
                or SimpleNamespace()
            ),
        ),
        policy_source=SimpleNamespace(step=slow_policy_step),
        human_source=SimpleNamespace(),
        readiness_provider=SimpleNamespace(),
        clock=lambda: runtime_now[0],
    )
    app._started = True
    app._next_decision_at = 10.0

    app.step(now=10.0)

    assert targets[0] > runtime_now[0]
    assert app._last_schedule_overrun is not None
    assert app._last_schedule_overrun["phase"] == "post_policy_source"
    assert app._held is False


def test_episode_end_is_terminal_and_requires_target_rebuild() -> None:
    reasons: list[str] = []
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=SimpleNamespace(
            request_hold=lambda *, reason: reasons.append(reason)
        ),
        policy_source=SimpleNamespace(),
        human_source=SimpleNamespace(),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )
    app._started = True

    app.end_episode(reason="task_success")

    assert reasons == [
        "sim_hil_application:episode_end:task_success"
    ]
    assert app._held is True
    assert app._episode_ended is True
    with pytest.raises(RuntimeError, match="target rebuild"):
        app.rearm()


def test_application_rearm_uses_bounded_freshness_loop_before_preflight(
    monkeypatch,
) -> None:
    calls: list[str] = []
    runtime = SimpleNamespace(clock=lambda: 10.0)
    composition = SimpleNamespace(
        human_alpha_by_component={"right_arm": 1.0},
        request_hold=lambda *, reason: calls.append(
            f"composition.hold:{reason}"
        ),
        rearm=lambda *, now: (
            calls.append(f"composition.rearm:{now}") or SimpleNamespace(passed=True)
        ),
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=runtime,
        composition=composition,
        policy_source=SimpleNamespace(),
        human_source=SimpleNamespace(),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )
    app._started = True
    app._held = True
    app._episode_ended = False
    cancel = lambda: False

    monkeypatch.setattr(
        app,
        "_wait_for_source_readiness",
        lambda *, cancel_requested, phase: calls.append(
            f"freshness:{phase}:{cancel_requested is cancel}"
        ),
    )
    monkeypatch.setattr(
        app,
        "_prime_activation_inputs",
        lambda: calls.append("prime"),
    )

    report = app.rearm(cancel_requested=cancel)

    assert report.passed
    assert calls == [
        "composition.hold:sim_hil_application:rearm_boundary",
        "freshness:rearm_source_refresh:True",
        "composition.rearm:10.0",
        "prime",
    ]
    assert app._held is False


def test_application_rearm_waits_until_human_emission_is_actually_fresh() -> None:
    now = [10.0]

    class HumanSource:
        def __init__(self) -> None:
            self.step_count = 0
            self.last_emitted_at = 8.0

        def step(self, *, now: float):
            self.step_count += 1
            if self.step_count == 3:
                self.last_emitted_at = now
            return SimpleNamespace(emission=None)

    human = HumanSource()

    class Composition:
        human_alpha_by_component = {"right_arm": 1.0}

        @staticmethod
        def readiness_failures(_evidence, *, target_effective_at: float):
            if target_effective_at - human.last_emitted_at > 0.2:
                return ("human_device_ready",)
            return ()

        @staticmethod
        def request_hold(*, reason: str):
            assert reason == "sim_hil_application:rearm_boundary"

        @staticmethod
        def rearm(*, now: float):
            return SimpleNamespace(passed=True)

    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(
            enabled=True,
            warmup_timeout_sec=0.5,
        ),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(clock=lambda: now[0]),
        composition=Composition(),
        policy_source=SimpleNamespace(),
        human_source=human,
        readiness_provider=SimpleNamespace(
            snapshot=lambda *, now: SimpleNamespace(
                detail_by_check={"human_device_ready": "fresh"}
            )
        ),
        clock=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )
    app._started = True
    app._held = True
    app._prime_activation_inputs = lambda: None

    report = app.rearm()

    assert report.passed
    assert human.step_count == 3
    assert app._held is False


def test_application_rearm_repairs_ready_gate_at_a_later_causal_cut(
    monkeypatch,
) -> None:
    now = [10.0]
    gate = ["ready"]
    hold_at = [0.0]
    observed: list[tuple[str, float | str]] = []

    def clock() -> float:
        now[0] += 0.001
        return now[0]

    class Composition:
        human_alpha_by_component = {"right_arm": 1.0}

        @staticmethod
        def request_hold(*, reason: str):
            observed.append(("hold", reason))
            hold_at[0] = clock()
            gate[0] = "hold"

        @staticmethod
        def rearm(*, now: float):
            observed.append(("rearm", now))
            assert gate[0] == "hold"
            return SimpleNamespace(passed=True)

    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(clock=clock),
        composition=Composition(),
        policy_source=SimpleNamespace(),
        human_source=SimpleNamespace(),
        readiness_provider=SimpleNamespace(),
        clock=clock,
    )
    app._started = True
    app._held = True
    monkeypatch.setattr(
        app,
        "_wait_for_source_readiness",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(app, "_prime_activation_inputs", lambda: None)

    report = app.rearm()

    assert report.passed
    assert observed[0] == (
        "hold",
        "sim_hil_application:rearm_boundary",
    )
    assert observed[1][0] == "rearm"
    assert float(observed[1][1]) >= hold_at[0]
    assert app._held is False


def test_application_rearm_freshness_wait_is_cancellable() -> None:
    human_step_count = [0]
    hold_reasons: list[str] = []
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(enabled=True),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(clock=lambda: 10.0),
        composition=SimpleNamespace(
            human_alpha_by_component={"right_arm": 1.0},
            request_hold=lambda *, reason: hold_reasons.append(reason),
        ),
        policy_source=SimpleNamespace(),
        human_source=SimpleNamespace(
            step=lambda *, now: human_step_count.__setitem__(
                0,
                human_step_count[0] + 1,
            )
        ),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )
    app._started = True
    app._held = True

    with pytest.raises(InterruptedError, match="rearm_source_refresh"):
        app.rearm(cancel_requested=lambda: True)

    assert human_step_count == [0]
    assert hold_reasons == ["sim_hil_application:rearm_boundary"]
    assert app._held is True


def _request(
    address: tuple[str, int],
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(address[0], address[1], timeout=2.0)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_sim_hil_status_server_is_loopback_only() -> None:
    with pytest.raises(ValueError, match="loopback"):
        SimHILStatusServer(lambda: {}, host="0.0.0.0")

def test_sim_hil_status_server_serves_structured_read_only_surface() -> None:
    value = {
        "schema": "dexteleop.control_station.sim_hil_status.v1",
        "started": False,
        "readiness": {"ready": False},
    }
    server = SimHILStatusServer(lambda: value, port=0)
    address = server.start()
    try:
        status, headers, payload = _request(address, "GET", "/api/status")
        assert status == 200
        assert headers["Content-Type"].startswith("application/json")
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert json.loads(payload) == value

        status, headers, payload = _request(address, "GET", "/")
        assert status == 200
        assert headers["X-Frame-Options"] == "DENY"
        assert b"Readiness gate" in payload
        assert b"no raw action values" in payload

        status, headers, payload = _request(address, "POST", "/api/status")
        assert status == 405
        assert headers["Allow"] == "GET"
        assert payload == b"read-only status server\n"
    finally:
        server.close()


def test_sim_hil_status_server_can_open_browser_without_blocking_start(
    monkeypatch,
) -> None:
    opened: list[str] = []
    opened_event = threading.Event()

    def open_browser(url: str, *, new: int, autoraise: bool) -> bool:
        assert new == 1
        assert autoraise is True
        opened.append(url)
        opened_event.set()
        return True

    monkeypatch.setattr(
        "control_station.sim_hil_gui.webbrowser.open",
        open_browser,
    )
    server = SimHILStatusServer(lambda: {}, port=0, open_browser=True)
    address = server.start()
    try:
        assert opened_event.wait(1.0)
        assert opened == [f"http://{address[0]}:{address[1]}/"]
    finally:
        server.close()


def test_sim_hil_status_server_reports_provider_or_json_contract_errors() -> None:
    server = SimHILStatusServer(lambda: {"invalid": object()}, port=0)
    address = server.start()
    try:
        status, _headers, payload = _request(address, "GET", "/api/status")
        assert status == 500
        result = json.loads(payload)
        assert result["schema"] == "dexteleop.control_station.sim_hil_status.v1"
        assert "status_error" in result
    finally:
        server.close()


def test_sim_hil_status_server_exposes_low_cost_live_vive_endpoint() -> None:
    live_calls: list[int] = []

    def live_status() -> dict[str, object]:
        live_calls.append(1)
        return {
            "schema": "dexteleop.control_station.sim_hil_vive_status.v1",
            "vive_receiver": {"packet_count": len(live_calls)},
        }

    server = SimHILStatusServer(
        lambda: {"full": True},
        live_status_provider=live_status,
        port=0,
    )
    address = server.start()
    try:
        status, _headers, body = _request(address, "GET", "/api/vive-status")
        assert status == 200
        assert json.loads(body)["vive_receiver"] == {"packet_count": 1}
        assert live_calls == [1]
    finally:
        server.close()


def test_sim_hil_acceptance_endpoint_is_tokened_and_only_submits_intents() -> None:
    submitted = []
    token = "acceptance-control-token-" + "x" * 32

    def submitter(payload):
        submitted.append(payload)
        return {
            "accepted": True,
            "intent_id": "intent-1",
            "revision": 1,
        }

    server = SimHILStatusServer(
        lambda: {"acceptance": {"revision": 0}},
        port=0,
        intent_submitter=submitter,
        control_token=token,
    )
    address = server.start()
    payload = json.dumps(
        {
            "client_id": "browser",
            "client_sequence": 1,
            "expected_revision": 0,
            "intent_kind": "hold",
        }
    ).encode("utf-8")
    try:
        status, _headers, body = _request(address, "GET", "/")
        assert status == 200
        assert token.encode("utf-8") in body
        assert "Sim-HIL 正式控制".encode("utf-8") in body
        assert "交换左右 VIVE 来源".encode("utf-8") in body
        assert "操作区 · Sim-HIL 正式控制".encode("utf-8") in body
        assert "确认当前阶段通过".encode("utf-8") not in body
        assert b"set_control_mode" in body
        assert b"set_blend_failure_fallback" in body
        assert b'id="blend-failure-fallback"' in body
        assert "应用回退源".encode("utf-8") in body
        assert "Sim-HIL 控制与 VIVE 标定".encode("utf-8") in body
        assert b'id="begin-vive-calibration"' in body
        assert b'id="capture-vive-calibration"' in body
        assert b'id="cancel-vive-calibration"' in body
        assert b'id="cancel-operation"' in body
        assert b'id="rebuild-runtime"' in body
        assert b'id="capture-runtime-home"' in body
        assert "采集当前 Runtime Home".encode("utf-8") in body
        assert "结束当前仿真并返回准备页".encode("utf-8") in body
        assert "操控使能就绪".encode("utf-8") in body
        assert "当前是只读状态页".encode("utf-8") in body
        assert b"policyRuntimeLaunchReady" in body
        assert b"humanRuntimeLaunchReady" in body
        assert b'id="vive-left-primary"' in body
        assert b'id="vive-right-primary"' in body
        assert b'id="vive-left-title"' in body
        assert b'id="vive-right-title"' in body
        assert b'id="vive-source-mapping"' in body
        assert b"source_role_by_output_channel" in body
        assert b'id="capture-card"' in body
        assert b"const inactive = selectedChannels.length === 0" in body
        assert "inactive ? '未选用'".encode("utf-8") in body
        assert b"RESTART REQUIRED" in body
        assert b"/api/vive-status" in body
        assert b"setInterval(refreshVive, 250)" in body
        assert b"intentKind === 'swap_vive_sources'" in body
        assert "输出 ${selected}".encode("utf-8") in body

        status, _headers, body = _request(
            address,
            "POST",
            "/api/acceptance-intents",
            body=payload,
            headers={
                "Content-Type": "application/json",
                "X-Sim-HIL-Control-Token": "wrong",
            },
        )
        assert status == 403
        assert submitted == []

        status, _headers, body = _request(
            address,
            "POST",
            "/api/acceptance-intents",
            body=payload,
            headers={
                "Content-Type": "application/json",
                "X-Sim-HIL-Control-Token": token,
            },
        )
        assert status == 202
        assert json.loads(body)["accepted"] is True
        assert submitted == [json.loads(payload)]
    finally:
        server.close()


def test_sim_hil_application_status_is_versioned_and_json_strict() -> None:
    readiness = MujocoSimHILReadinessEvidence(
        policy_source_ready=True,
        human_device_ready=False,
        calibration_ready=True,
        home_ready=True,
        rebase_ready=False,
        deadman_ready=False,
        policy_chunk_ttl_ready=True,
        measured_state_ready=True,
        detail_by_check={
            "policy_source_ready": "ready",
            "human_device_ready": "no_emission",
            "calibration_ready": "ready",
            "home_ready": "ready",
            "rebase_ready": "rebase_required",
            "deadman_ready": "teleop_state:INIT",
            "policy_chunk_ttl_ready": "ready",
            "measured_state_ready": "finite",
        },
    )
    runtime_config = SimpleNamespace(
        runtime_session_id="session",
        control_domain_id="domain",
        clock_domain_id="clock",
        enable_composer_authority=True,
        enable_primary_policy_authority=False,
        enable_replay_authority=False,
    )
    contracts = SimpleNamespace(
        composition_spec=SimpleNamespace(
            spec_id="composition",
            semantic_content_hash="composition-hash",
            required_roles=("human", "policy"),
        ),
        strategy_spec=SimpleNamespace(
            strategy_id="strategy",
            semantic_content_hash="strategy-hash",
        ),
        authority_profile=SimpleNamespace(
            profile_id="authority",
            semantic_content_hash="authority-hash",
        ),
    )
    policy_source = SimpleNamespace(
        last_emission=None,
        status=lambda *, now: {"source_id": "ppo", "started": False},
    )
    human_source = SimpleNamespace(
        session_state="DEBUG_ONLY",
        started=False,
        last_emission=None,
        binding_receipt=SimpleNamespace(
            source_contract_hash="human-source-hash",
            observer_binding_id="observer",
            semantic_content_hash="binding-hash",
            rig_absent=True,
            observer_exclusive=True,
            authoritative=False,
        ),
        session=SimpleNamespace(
            home_recalibration_required=False,
            motion_hold_mode="",
            motion_hold_reason="",
            tracker_loss_duration_sec=0.0,
            tracker_loss_channels=(),
        ),
    )
    runtime = SimpleNamespace(
        config=runtime_config,
        action_schema=SimpleNamespace(semantic_content_hash="action-schema-hash"),
    )
    composition = SimpleNamespace(
        contracts=contracts,
        started=False,
        activation_scope=None,
        required_readiness_checks=lambda *, target_effective_at: (
            "policy_source_ready",
            "calibration_ready",
            "home_ready",
            "policy_chunk_ttl_ready",
            "measured_state_ready",
        ),
        readiness_failures=lambda _readiness, *, target_effective_at: (),
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=runtime,
        composition=composition,
        policy_source=policy_source,
        human_source=human_source,
        readiness_provider=SimpleNamespace(snapshot=lambda *, now: readiness),
        gripper_input_selection=SimpleNamespace(
            requested_source="auto",
            resolved_source="angle_trigger",
            q6_port=None,
            fallback_reason="no_reachable_q6",
        ),
        clock=lambda: 10.0,
    )

    status = app.status()
    json.dumps(status, allow_nan=False)
    live_status = app.vive_status()
    json.dumps(live_status, allow_nan=False)
    assert live_status["schema"] == "dexteleop.control_station.sim_hil_vive_status.v1"
    assert live_status["vive_receiver"]["status"] == "unavailable"
    assert status["schema"] == SIM_HIL_STATUS_SCHEMA_V1
    assert status["readiness"]["ready"] is True
    assert status["readiness"]["failed_checks"] == []
    assert status["readiness"]["all_evidence_ready"] is False
    assert status["readiness"]["all_evidence_failed_checks"] == [
        "human_device_ready",
        "rebase_ready",
        "deadman_ready",
    ]
    assert "human_device_ready" not in status["readiness"]["required_checks"]
    assert status["contracts"]["composition"]["spec_hash"] == "composition-hash"
    assert status["human"]["binding"]["authoritative"] is False
    assert status["execution_gate"]["status"] == "not_started"
    device = status["source_configuration"]["input_device_bindings"][0]
    assert device == {
        "component_id": "right_gripper",
        "requested_device_kind": "auto",
        "device_kind": "angle_trigger",
        "endpoint": None,
        "fallback_reason": "no_reachable_q6",
    }

    snapshot = TargetExecutionStatusSnapshot(
        control_domain_id="domain",
        target_handler_id="target-handler",
        clock_domain_id="clock",
        activation_mode="required",
        captured_at=10.0,
        request=TargetExecutionStageStatus(
            stage="request",
            status="submitted",
            record_id="request-1",
            record_hash="request-hash-1",
            available_at=9.90,
        ),
        submission=TargetExecutionStageStatus(
            stage="submission",
            status="accepted_for_processing",
            record_id="submission-1",
            record_hash="submission-hash-1",
            available_at=9.91,
        ),
        completion=TargetExecutionStageStatus(
            stage="completion",
            status="resolved",
            record_id="completion-1",
            record_hash="completion-hash-1",
            available_at=9.92,
        ),
        admission=TargetExecutionStageStatus(
            stage="admission",
            status="accepted",
            record_id="admission-1",
            record_hash="admission-hash-1",
            available_at=9.93,
        ),
        activation=TargetExecutionStageStatus(
            stage="activation",
            status="pending",
        ),
        target_id="target-1",
        target_hash="target-hash-1",
    )
    runtime.capture_execution_gate_snapshot = lambda *, now: SimpleNamespace(
        status="active",
        reason=None,
        gate_epoch=1,
        source_event_ids=(),
        semantic_content_hash="gate-hash",
    )
    runtime.capture_target_execution_status_snapshot = (
        lambda *, now: snapshot
    )
    app._started = True

    running_status = app.status()
    target_status = running_status["target_execution"]
    assert target_status["request_id"] == "request-1"
    assert target_status["stages"]["submission"]["status"] == (
        "accepted_for_processing"
    )
    assert target_status["stages"]["admission"]["record_id"] == "admission-1"
    assert target_status["stages"]["activation"]["status"] == "pending"
    json.dumps(running_status, allow_nan=False)


def test_station_coordinate_binding_does_not_mutate_home_profile(tmp_path) -> None:
    coordinate = new_coordinate_calibration(active_sides=("left", "right"))
    coordinate.set_world_to_desk(np.asarray(np.eye(4), dtype=float))
    coordinate.set_tracker_offset("left_wrist", np.asarray(np.eye(4), dtype=float))
    right_offset = np.asarray(np.eye(4), dtype=float)
    right_offset[0, 3] = 0.12
    coordinate.set_tracker_offset("right_wrist", right_offset)
    saved_path = save_coordinate_profile(
        coordinate,
        root=tmp_path,
        profile_name="bimanual",
    )
    home = TeleopCalibration(
        arms={
            side: ArmCalibration(name=f"{side}_arm", side=side, T_D_B=np.eye(4))
            for side in ("left", "right")
        },
        metadata={"home_ready": True},
    )
    with patch(
        "control_station.vive_calibration_profiles.control_station_vive_coordinate_profiles_dir",
        return_value=tmp_path,
    ):
        merged, binding = _bind_vive_coordinate_profile(
            home,
            coordinate_profile_path=saved_path,
        )

    assert home.world_to_desk_pose() is None
    assert home.tracker_offsets == {}
    np.testing.assert_allclose(merged.world_to_desk_pose(), np.eye(4))
    np.testing.assert_allclose(merged.tracker_offsets["right_wrist"], right_offset)
    assert binding["mode"] == (
        "independent_vive_coordinate_plus_stored_human_home_v1"
    )
    assert binding["vive_coordinate_profile_path"] == str(saved_path)


def test_station_coordinate_binding_accepts_profile_selected_single_side(
    tmp_path,
) -> None:
    coordinate = new_coordinate_calibration(active_sides=("right",))
    coordinate.set_world_to_desk(np.eye(4))
    coordinate.set_tracker_offset("right_wrist", np.eye(4))
    saved_path = save_coordinate_profile(
        coordinate,
        root=tmp_path,
        profile_name="right-only",
    )
    home = TeleopCalibration(
        arms={
            "right_arm": ArmCalibration(
                name="right_arm",
                side="right",
                T_D_B=np.eye(4),
            )
        },
        metadata={"home_ready": True},
    )

    with patch(
        "control_station.vive_calibration_profiles.control_station_vive_coordinate_profiles_dir",
        return_value=tmp_path,
    ):
        merged, binding = _bind_vive_coordinate_profile(
            home,
            coordinate_profile_path=saved_path,
            required_sides=("right",),
        )

    assert tuple(binding["active_sides"]) == ("right",)
    assert set(merged.tracker_offsets) == {"right_wrist"}
    with patch(
        "control_station.vive_calibration_profiles.control_station_vive_coordinate_profiles_dir",
        return_value=tmp_path,
    ):
        with pytest.raises(ValueError, match="requires VIVE coordinate sides"):
            _bind_vive_coordinate_profile(
                home,
                coordinate_profile_path=saved_path,
            )


def test_station_coordinate_binding_rejects_home_from_another_profile(
    tmp_path,
) -> None:
    coordinate = new_coordinate_calibration(active_sides=("left", "right"))
    coordinate.set_world_to_desk(np.eye(4))
    coordinate.set_tracker_offset("left_wrist", np.eye(4))
    coordinate.set_tracker_offset("right_wrist", np.eye(4))
    coordinate.metadata["source_profile_sha256"] = "different-home-profile"
    saved_path = save_coordinate_profile(
        coordinate,
        root=tmp_path,
        profile_name="mismatch",
    )
    home_path = tmp_path / "home.json"
    TeleopCalibration(metadata={"home_ready": True}).save_json(home_path)

    with patch(
        "control_station.vive_calibration_profiles.control_station_vive_coordinate_profiles_dir",
        return_value=tmp_path,
    ):
        with pytest.raises(ValueError, match="another VIVE coordinate profile"):
            _bind_vive_coordinate_profile(
                TeleopCalibration.load_json(home_path),
                coordinate_profile_path=saved_path,
                home_profile_path=home_path,
            )
