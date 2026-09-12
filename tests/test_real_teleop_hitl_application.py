from __future__ import annotations

import queue
import time
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest

from control_station.real_teleop import hitl_application
from control_station.real_teleop.hitl_application import (
    ExclusiveHITLRealTeleopExtension,
)
from control_station.real_teleop_application import (
    build_real_teleop_progress_application,
)
from dataset_contracts.hitl import EXCLUSIVE_HITL_POLICY_MODES
from robot_motion.action import BIMANUAL_QPOS_V1, RobotAction
from robot_motion.deploy import NormalizedAction, PolicyActionEmission
from robot_motion.deploy.runtime import SourceValidationResult
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.real_teleop import hitl_policy_runtime
from teleop_runtime.real_teleop.cli import build_real_teleop_argument_parser
from teleop_runtime.real_teleop.hitl_cli import add_exclusive_hitl_arguments
from teleop_runtime.real_teleop.hitl_terminal import HITLTerminalClutchInput
from teleop_runtime.schemas import EffectorCommand, TeleopCommand, TeleopObservation


class _PolicySource:
    maximum_hold_sec = 0.2

    def __init__(self) -> None:
        self.emissions: list[PolicyActionEmission] = []
        self.started = False
        self.closed = False
        self.reset_count = 0
        self._last_error = None

    @property
    def ready(self) -> bool:
        return self.started and not self.closed

    @property
    def last_error(self):
        return self._last_error

    def start(self) -> None:
        self.started = True

    def reset_for_activation(self) -> None:
        self.reset_count += 1

    def step(self, *, now=None, step_index=None):
        del now, step_index
        return None if not self.emissions else self.emissions.pop(0)

    def status(self, *, now=None):
        del now
        return {"ready": self.ready}

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self) -> None:
        self.command_override_provider = None
        self.human_command_selection_provider = None
        self.state_machine = SimpleNamespace(state=TeleopRunState.ARMED)
        self.calibration = SimpleNamespace(to_dict=lambda: {"home": "fixture"})
        self.hold_reasons: list[str] = []
        self.arm_count = 0

    def hold_hardware(self, reason: str, **kwargs) -> bool:
        del kwargs
        self.hold_reasons.append(reason)
        self.state_machine.state = TeleopRunState.DEBUG_ONLY
        return True

    def arm_hardware(self) -> None:
        self.arm_count += 1
        self.state_machine.state = TeleopRunState.ARMED


class _Clutch:
    def __init__(self) -> None:
        self.requests: list[str] = []
        self.hold_reasons: list[str] = []
        self.home_requests = 0

    def consume_hitl_source_requests(self):
        requests = tuple(self.requests)
        self.requests.clear()
        return requests

    def request_hold(self, *, reason: str) -> None:
        self.hold_reasons.append(reason)

    def request_runtime_home_for_hitl(self) -> bool:
        self.home_requests += 1
        return True


def _components(session: _Session):
    return SimpleNamespace(
        session=session,
        active_sides=("right",),
        keyboard_source=object(),
    )


def _args(**overrides):
    values = {
        "active_sides": ("right",),
        "hitl_policy_control": True,
        "hitl_policy_mode": "normal",
        "hitl_initial_source": "policy",
        "hitl_takeover_key": "u",
        "hitl_policy_resume_key": "i",
        "hitl_policy_artifact_json": "",
        "_hitl_policy_emission_source": _PolicySource(),
    }
    values.update(overrides)
    return Namespace(**values)


def _human_command() -> TeleopCommand:
    command = TeleopCommand(timestamp=100.0, metadata={"source": "human"})
    command.add_command(
        EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="joint_vector",
            value=np.zeros(6, dtype=np.float32),
        )
    )
    return command


def _emission(index: int, *, emitted_at: float) -> PolicyActionEmission:
    action = RobotAction(
        action_space=BIMANUAL_QPOS_V1,
        timestamp=emitted_at,
        arm_qpos_by_side={
            "right": np.full(6, float(index), dtype=np.float32),
        },
        active_effectors=("right_arm",),
        source="hitl-application-fixture",
    )
    return PolicyActionEmission(
        emission_id=f"emission:{index}",
        admission_mode="on_emission_pop_v1",
        policy_node_id="policy-fixture",
        inference_id=f"inference:{index}",
        policy_output_id=f"output:{index}",
        chunk_id=index,
        chunk_keyframe_index=0,
        chunk_size=1,
        decoded_absolute_action=NormalizedAction(
            vector=np.asarray([float(index)], dtype=np.float32)
        ),
        pre_safety_robot_action=action,
        source_validation=SourceValidationResult(accepted=True),
        representation_receipt=None,
        pre_safety_faults=(),
        produced_at=emitted_at - 0.01,
        produced_clock_domain_id="policy.monotonic",
        emitted_at=emitted_at,
        target_effective_at=emitted_at + 0.01,
        effective_from=emitted_at,
        effective_until=emitted_at + 0.2,
        effective_clock_domain_id="policy.monotonic",
        schedule_spec_id="fixture-schedule",
        schedule_spec_hash="fixture-schedule-hash",
        queue_len_before=1,
        queue_len_after=0,
        queue_pop_count=index,
        semantic_content_hash=f"emission-hash:{index}",
    )


def _bind_and_start(extension, components, clutch) -> None:
    extension.attach_components(components)
    extension.bind_runtime(
        components=components,
        clutch_input=clutch,
        camera_sources={},
        dataset_recorder=None,
        dataset_recorder_provider=lambda: None,
        episode_recorder=SimpleNamespace(is_recording=False),
        control_ui=None,
        recording_active=lambda: False,
        pause_recording=lambda now, *, reason: True,
    )
    extension.start(components=components)


def _configured_extension(args):
    build_real_teleop_progress_application(args)
    return args._real_teleop_runtime_extension


def test_application_leaf_installs_one_extension_and_additive_terminal() -> None:
    args = _args()
    application = build_real_teleop_progress_application(args)

    assert application is not None
    assert isinstance(
        args._real_teleop_runtime_extension,
        ExclusiveHITLRealTeleopExtension,
    )
    terminal = args._real_teleop_terminal_factory(object())
    assert isinstance(terminal, HITLTerminalClutchInput)
    assert terminal.takeover_key == "u"
    assert terminal.policy_resume_key == "i"
    dataset_contract = args._exclusive_hitl_dataset_contract
    assert dataset_contract.policy_source_kind == "injected_policy_emission"
    assert dataset_contract.policy_identity_status == "unbound_injected_source"
    assert dataset_contract.active_sides == ("right",)


def test_application_leaf_rejects_competing_runtime_extension() -> None:
    args = _args(_real_teleop_runtime_extension=object())
    with pytest.raises(RuntimeError, match="single RealTeleop application-extension"):
        build_real_teleop_progress_application(args)


def test_normal_application_preflight_fails_before_runtime_without_artifact() -> None:
    args = _args(
        _hitl_policy_emission_source=None,
        hitl_policy_artifact_json="",
    )
    with pytest.raises(ValueError, match="hitl-policy-artifact-json"):
        build_real_teleop_progress_application(args)


def test_hitl_cli_is_default_off_and_parses_additive_profile_options() -> None:
    parser = build_real_teleop_argument_parser()
    add_exclusive_hitl_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.hitl_policy_control is False
    assert defaults.hitl_policy_mode == "normal"
    assert defaults.hitl_takeover_key == "u"
    assert defaults.hitl_policy_resume_key == "i"
    assert defaults.hitl_policy_prefetch_threshold == 4

    selected = parser.parse_args(
        [
            "--hitl-policy-control",
            "--hitl-policy-mode",
            "rtc",
            "--hitl-initial-source",
            "human",
            "--hitl-takeover-key",
            "h",
            "--hitl-policy-resume-key",
            "j",
        ]
    )
    assert selected.hitl_policy_control is True
    assert selected.hitl_policy_mode == "rtc"
    assert selected.hitl_initial_source == "human"
    assert selected.hitl_takeover_key == "h"
    assert selected.hitl_policy_resume_key == "j"


def test_policy_human_policy_round_trip_waits_for_home_reset_and_dispatch() -> None:
    args = _args()
    source = args._hitl_policy_emission_source
    extension = _configured_extension(args)
    session = _Session()
    components = _components(session)
    clutch = _Clutch()
    _bind_and_start(extension, components, clutch)

    clutch.requests.append("human")
    extension.after_control_input_poll(now=100.0)
    assert extension.state.phase == "hold_requested"
    assert clutch.hold_reasons == ["hitl_policy_to_human"]

    session.state_machine.state = TeleopRunState.DEBUG_ONLY
    extension.after_control_input_poll(now=100.1)
    assert extension.state.phase == "waiting_human_home"
    assert clutch.home_requests == 1

    session.state_machine.state = TeleopRunState.ARMED
    extension.after_control_input_poll(now=100.2)
    assert extension.state.phase == "waiting_target_dispatch"
    human = _human_command()
    assert extension.selector(TeleopObservation(timestamp=100.2), human, 100.2) is None
    extension.after_session_step(
        step_result=SimpleNamespace(command=human, command_dispatched=True),
        now=100.2,
    )
    assert extension.state.phase == "idle"
    assert extension.state.active_source == "human"

    source.emissions.append(_emission(2, emitted_at=time.monotonic() + 1.0))
    clutch.requests.append("policy")
    extension.after_control_input_poll(now=101.0)
    assert extension.state.phase == "waiting_policy_emission"
    assert source.reset_count == 1
    assert session.hold_reasons

    waiting = extension.selector(
        TeleopObservation(timestamp=101.0),
        _human_command(),
        101.0,
    )
    assert waiting is not None and waiting.command is None
    extension.after_control_input_poll(now=101.1)
    assert extension.state.phase == "waiting_target_dispatch"
    assert session.arm_count == 1

    selected = extension.selector(
        TeleopObservation(timestamp=101.2),
        _human_command(),
        101.2,
    )
    assert selected is not None and selected.command is not None
    assert selected.command.metadata["hitl_exclusive_source"] == "policy"
    extension.after_session_step(
        step_result=SimpleNamespace(
            command=selected.command,
            command_dispatched=True,
        ),
        now=101.2,
    )
    assert extension.state.phase == "idle"
    assert extension.state.active_source == "policy"
    extension.close_before_session(components=components, terminal_error=None)


def test_rtc_committed_emission_intervention_fails_held_without_motion() -> None:
    inbox: queue.Queue[PolicyActionEmission] = queue.Queue(maxsize=2)
    args = _args(
        hitl_policy_mode="rtc",
        hitl_initial_source="human",
        _hitl_policy_emission_source=None,
        _hitl_rtc_committed_emissions=inbox,
    )
    extension = _configured_extension(args)
    assert args._exclusive_hitl_dataset_contract.policy_mode == "rtc"
    assert args._exclusive_hitl_dataset_contract.rtc_no_action is True
    assert (
        args._exclusive_hitl_dataset_contract.policy_identity_status
        == "rtc_no_action"
    )
    session = _Session()
    components = _components(session)
    clutch = _Clutch()
    _bind_and_start(extension, components, clutch)

    clutch.requests.append("policy")
    extension.after_control_input_poll(now=200.0)
    assert extension.state.phase == "waiting_policy_emission"
    inbox.put_nowait(_emission(3, emitted_at=time.monotonic() + 1.0))
    waiting = extension.selector(
        TeleopObservation(timestamp=200.0),
        _human_command(),
        200.0,
    )
    assert waiting is not None and waiting.command is None

    extension.after_control_input_poll(now=200.1)
    assert extension.state.phase == "failed"
    assert "no-action" in str(extension.state.error)
    assert session.state_machine.state == TeleopRunState.DEBUG_ONLY
    assert session.arm_count == 0


def test_hitl_policy_mode_constants_share_the_dataset_contract_source() -> None:
    assert hitl_policy_runtime.HITL_POLICY_MODES == EXCLUSIVE_HITL_POLICY_MODES
    assert hitl_application.HITL_POLICY_MODES == EXCLUSIVE_HITL_POLICY_MODES
    assert (
        hitl_policy_runtime.HITL_POLICY_MODE_NORMAL
        == hitl_application.HITL_POLICY_MODE_NORMAL
        == "normal"
    )
    assert (
        hitl_policy_runtime.HITL_POLICY_MODE_RTC
        == hitl_application.HITL_POLICY_MODE_RTC
        == "rtc"
    )


def test_hitl_cli_policy_mode_choices_follow_the_dataset_contract() -> None:
    parser = build_real_teleop_argument_parser()
    add_exclusive_hitl_arguments(parser)
    action = next(
        action for action in parser._actions if action.dest == "hitl_policy_mode"
    )
    assert tuple(action.choices) == EXCLUSIVE_HITL_POLICY_MODES
    assert action.default == EXCLUSIVE_HITL_POLICY_MODES[0]
