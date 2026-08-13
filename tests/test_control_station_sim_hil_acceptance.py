from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from control_station.sim_hil import (
    MujocoSimHILApplication,
    MujocoSimHILApplicationConfig,
)
from control_station.sim_hil_acceptance import (
    SIM_HIL_ACCEPTANCE_STAGES_V1,
    STATION_CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1,
    STATION_CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1,
    SimHILAcceptanceController,
    SimHILAcceptanceIntent,
    SimHILAcceptanceTraceCapture,
    StationControlTraceEpisodeSessionCapture,
    StationControlTraceFinalizationReceipt,
    sim_hil_acceptance_intent_from_json,
)
from control_station.hitl import (
    EXCLUSIVE_HITL_HUMAN_TO_POLICY,
    EXCLUSIVE_HITL_POLICY_TO_HUMAN,
    ExclusiveHITLSwitchEvidence,
)
from control_station.control_trace import project_exclusive_hitl_switch
from dataset_contracts.control_trace import ControlTraceFact
from teleop_runtime.control_trace import (
    CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1,
    CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1,
    ControlTraceEpisodeCapturePort,
    ControlTraceEpisodeSessionCapture,
    ControlTraceSessionFinalizationReceipt,
)
from teleop_runtime.sources.pose.vive import (
    VivePoseSource,
    ViveTrackerConfig,
)


def test_acceptance_capture_compatibility_names_resolve_to_lower_owner(
    tmp_path,
) -> None:
    assert SimHILAcceptanceTraceCapture is StationControlTraceEpisodeSessionCapture
    assert issubclass(
        StationControlTraceEpisodeSessionCapture,
        ControlTraceEpisodeSessionCapture,
    )
    assert StationControlTraceFinalizationReceipt is (
        ControlTraceSessionFinalizationReceipt
    )
    assert STATION_CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1 == (
        CONTROL_TRACE_TERMINAL_RECEIPT_HASH_DOMAIN_V1
    )
    assert STATION_CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1 == (
        CONTROL_TRACE_FINALIZATION_HASH_DOMAIN_V1
    )
    capture = StationControlTraceEpisodeSessionCapture(tmp_path)
    try:
        assert isinstance(capture, ControlTraceEpisodeCapturePort)
        assert capture.runtime_session_id == "control-station-sim-hil-ppo-session"
        assert capture.control_domain_id == "mujoco-sim-hil"
    finally:
        capture.close()

    generic = ControlTraceEpisodeSessionCapture(
        tmp_path,
        runtime_session_id="generic-runtime",
        control_domain_id="generic-control-domain",
        trace_requirement_profile_id=(
            "deployable_sim_immediate_component_policy_human_composer_v1"
        ),
        clock_domain_id="generic.monotonic",
        capture_purpose="generic_capture_test",
    )
    try:
        assert isinstance(generic, ControlTraceEpisodeCapturePort)
        assert generic.runtime_session_id == "generic-runtime"
        assert generic.control_domain_id == "generic-control-domain"
        assert generic.clock_domain_id == "generic.monotonic"
    finally:
        generic.close()


class _FakeApplication:
    def __init__(self) -> None:
        self.started = False
        self.initial_alpha = 0.0
        self._held = False
        self._episode_ended = False
        self._viewer_preview_started = False
        self.alpha_updates: list[float] = []
        self.calls: list[str] = []
        self._clock = 10.0
        self.cycle_index = -1
        self.vive_swaps = 0
        self.runtime_home_captures = 0
        self.blend_failure_fallback = "hold"
        self.blend_failure_grace_sec = 0.0
        self.allocation_operations: list[str] = []

    def clock(self) -> float:
        self._clock += 0.01
        return self._clock

    def run_human_input_acceptance_check(self, *, timeout_sec: float):
        self.calls.append(f"device:{timeout_sec}")
        return {
            "status": "ready",
            "command_dispatched": False,
            "authoritative": False,
        }

    def swap_vive_left_right_sources(self):
        self.vive_swaps += 1
        self.calls.append("swap_vive_sources")
        return {
            "status": "swapped",
            "swap_count": self.vive_swaps,
        }

    def start(self):
        self.calls.append("start")
        self.started = True
        self._viewer_preview_started = False
        return SimpleNamespace(passed=True)

    @property
    def viewer_preview_started(self) -> bool:
        return bool(self._viewer_preview_started and not self.started)

    def start_viewer_preview(self):
        self.calls.append("start_viewer_preview")
        self._viewer_preview_started = True
        return {
            "status": "viewer_preview",
            "command_dispatched": False,
            "authoritative": False,
        }

    def capture_runtime_home(self):
        self.runtime_home_captures += 1
        self.calls.append("capture_runtime_home")
        return {
            "capture_id": f"runtime-home:{self.runtime_home_captures}",
            "command_dispatched": False,
            "authoritative": False,
        }

    def set_human_blend_alpha(self, alpha: float):
        self.calls.append(f"alpha:{alpha}")
        self.alpha_updates.append(float(alpha))
        record = SimpleNamespace(
            parameter_record_id=f"alpha-record:{alpha}:{len(self.alpha_updates)}",
            semantic_content_hash=f"alpha-record-hash:{alpha}:{len(self.alpha_updates)}",
        )
        return SimpleNamespace(
            accepted=True,
            parameter_record=record,
            # Retain the compatibility identity used by older lightweight tests.
            parameter_snapshot_id=f"alpha:{alpha}",
        )

    def set_blend_failure_fallback(
        self,
        source: str,
        *,
        grace_sec: float | None = None,
    ):
        self.blend_failure_fallback = str(source)
        if grace_sec is not None:
            self.blend_failure_grace_sec = float(grace_sec)
        self.calls.append(f"blend_fallback:{source}")
        return {
            "status": "selected",
            "source": str(source),
            "grace_sec": self.blend_failure_grace_sec,
            "latched_source": None,
        }

    def submit_allocation_button_operation(
        self,
        *,
        signal_name: str,
        event_id: str,
        source_fact_id: str,
        source_fact_hash: str,
        occurred_at: float,
    ):
        assert event_id == source_fact_id
        assert source_fact_hash
        assert occurred_at > 0.0
        self.allocation_operations.append(signal_name)
        if signal_name == "residual_rebase_and_clear":
            return SimpleNamespace(
                receipt_id=f"operation:{signal_name}",
                semantic_content_hash=f"operation-hash:{signal_name}",
                override_receipt=None,
                anchor_request=SimpleNamespace(request_id="anchor:1"),
            )
        return SimpleNamespace(
            receipt_id=f"operation:{signal_name}",
            semantic_content_hash=f"operation-hash:{signal_name}",
            override_receipt=SimpleNamespace(
                receipt_id="override:1",
                semantic_content_hash=f"override-hash:{signal_name}",
            ),
            anchor_request=None,
        )

    def hold(self, *, reason: str):
        self.calls.append(f"hold:{reason}")
        self._held = True

    def rearm(self, *, cancel_requested=None):
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("cancelled")
        self.calls.append("rearm")
        self._held = False
        return SimpleNamespace(passed=True)

    def end_episode(self, *, reason: str):
        self.calls.append(f"end:{reason}")
        self._held = True
        self._episode_ended = True

    def run_acceptance_cycle(self) -> None:
        assert self.started and not self._held
        self.cycle_index += 1

    def status(self):
        alpha = (
            self.alpha_updates[-1]
            if self.alpha_updates
            else self.initial_alpha
        )
        mode = (
            "policy_only"
            if alpha == 0.0
            else "human_only"
            if alpha == 1.0
            else "proportional_blend"
        )
        return {
            "started": self.started,
            "held": self._held,
            "episode_ended": self._episode_ended,
            "visualization": {
                "enabled": True,
                "running": self.started or self._viewer_preview_started,
            },
            "execution_gate": {
                "status": "hold" if self._held else "ready",
            },
            "composition": {
                "mix": {
                    "mode": mode,
                    "human_alpha_by_component": {
                        "right_arm": alpha,
                        "right_gripper": alpha,
                    },
                },
                "last_cycle": (
                    None
                    if self.cycle_index < 0
                    else {
                        "cycle_index": self.cycle_index,
                        "execution": {"status": "dispatched"},
                        "composition_step": {
                            "immediate_difference": {
                                "observation_id": "difference",
                            }
                        },
                    }
                ),
            },
            "execution_difference": {
                "enabled": True,
                "latest": {"status": "available"},
            },
        }


class _FakeTraceCapture:
    def __init__(self) -> None:
        self.runtime_session_id = "control-station-sim-hil-ppo-session"
        self.control_domain_id = "mujoco-sim-hil"
        self.clock_domain_id = "mujoco.sim_hil.monotonic"
        self.prepared = False
        self.finalized = False
        self.aborted = False
        self.closed = False
        self.rotation_count = 0
        self.facts: list[ControlTraceFact] = []

    def admit_trace(self, fact: ControlTraceFact) -> None:
        assert isinstance(fact, ControlTraceFact)
        self.facts.append(fact)

    def prepare(self, *, now: float):
        assert now > 0.0
        self.prepared = True
        return self.status()

    def finalize(self, *, now: float):
        assert self.prepared
        assert now > 0.0
        self.prepared = False
        self.finalized = True
        return self.status()

    def abort(self, *, now: float):
        assert now > 0.0
        self.prepared = False
        self.aborted = True

    def rotate_runtime_session(self, *, now: float):
        assert now > 0.0
        assert not self.prepared
        self.rotation_count += 1
        return self.status()

    def status(self):
        return {
            "enabled": True,
            "status": (
                "committed"
                if self.finalized
                else "recording"
                if self.prepared
                else "ready"
            ),
            "trace_session_id": "trace",
            "session_root": "/tmp/trace",
            "episode_uuid": "episode" if self.prepared else None,
            "commit_status": "committed" if self.finalized else None,
            "message": "",
        }

    def close(self, *, now: float):
        assert now > 0.0
        self.closed = True


class _ExclusiveFakeApplication(_FakeApplication):
    """Virtual policy/Human source switch with lower evidence identities."""

    human_input_attached = True

    def __init__(self) -> None:
        super().__init__()
        self.runtime = SimpleNamespace(
            config=SimpleNamespace(
                clock_domain_id="mujoco.sim_hil.monotonic",
            )
        )
        self.last_hold_outcome = None
        self.last_activation_prime_evidence = None
        self.activation_sequence = 0

    def hold(self, *, reason: str):
        super().hold(reason=reason)
        self.last_hold_outcome = SimpleNamespace(
            lower_event_ids=(
                f"lower-hold:{len([call for call in self.calls if call.startswith('hold:')])}",
            )
        )

    def capture_runtime_home(self):
        receipt = super().capture_runtime_home()
        return {
            **receipt,
            "captured_at": self.clock(),
            "clock_domain_id": "mujoco.sim_hil.monotonic",
            "sample_count_by_side": {"right": 5},
            "human_home_pose_desk_by_side": {"right": [1.0] * 16},
            "robot_home_pose_desk_by_side": {"right": [2.0] * 16},
        }

    def rearm_with_allocation_operation(
        self,
        *,
        signal_name,
        event_id,
        source_fact_id,
        source_fact_hash,
        occurred_at,
        human_emission_required,
        cancel_requested=None,
    ):
        report = super().rearm(cancel_requested=cancel_requested)
        receipt = self.submit_allocation_button_operation(
            signal_name=signal_name,
            event_id=event_id,
            source_fact_id=source_fact_id,
            source_fact_hash=source_fact_hash,
            occurred_at=occurred_at,
        )
        self.activation_sequence += 1
        target_source = (
            "human"
            if signal_name == "allocation_absolute_overlay_owner"
            else "policy"
        )
        assert human_emission_required is (target_source == "human")
        activation_id = f"activation:{self.activation_sequence}"

        def target_emission_identity(source: str) -> tuple[str, str]:
            assert source == target_source
            return (
                f"{source}-emission:{self.activation_sequence}",
                f"{source}-emission-hash:{self.activation_sequence}",
            )

        self.last_activation_prime_evidence = SimpleNamespace(
            activation_id=activation_id,
            activation_hash=f"{activation_id}:hash",
            policy_reset_performed=True,
            target_emission_identity=target_emission_identity,
        )
        return report, receipt

    def establish_activation_execution(self, *, target_source: str):
        self.calls.append(f"establish:{target_source}")
        sequence = self.activation_sequence
        return SimpleNamespace(
            target_id=f"target:{sequence}",
            target_hash=f"target-hash:{sequence}",
            target_admission_receipt_id=f"target-admission:{sequence}",
            target_completion_receipt_id=f"target-completion:{sequence}",
            command_id=f"command:{sequence}",
            dispatch_receipt_id=f"dispatch:{sequence}",
        )


class _FakeViveSetup:
    def __init__(
        self,
        *,
        input_ready: bool,
        source_role: str = "left_wrist",
        current_profile_path=None,
    ) -> None:
        self.input_ready = bool(input_ready)
        self.source_role = str(source_role)
        self.current_profile_path = current_profile_path
        self.workflow_active = False
        self.stage = "idle"
        self.phase = "idle"
        self.last_saved_profile_path = None
        self.closed = False
        self.events: list[tuple[str, dict[str, object]]] = []

    def snapshot(self):
        return {
            "source_started": not self.closed,
            "workflow_active": self.workflow_active,
            "stage": self.stage,
            "stage_label": self.stage,
            "phase": self.phase,
            "message": "ready" if self.input_ready else "waiting for VIVE",
            "stage_guide": {
                "title": "guide",
                "direction": "back to front",
                "steps": ["move once"],
            },
            "input_preflight": {
                "ready": self.input_ready,
                "message": (
                    "VIVE input ready"
                    if self.input_ready
                    else "VIVE input missing"
                ),
            },
            "source_role_by_output_channel": {
                "right_wrist_pose": self.source_role,
            },
            "profiles": [],
            "live_channels": {},
        }

    def dispatch(self, event: str, payload=None) -> None:
        values = dict(payload or {})
        self.events.append((event, values))
        if event == "begin_calibration":
            self.workflow_active = True
            self.stage = "virtual_desk_x"
            self.phase = "ready"
        elif event == "cancel_calibration":
            self.workflow_active = False
            self.stage = "idle"
            self.phase = "idle"
        elif event == "swap_vive_sources":
            self.source_role = (
                "right_wrist"
                if self.source_role == "left_wrist"
                else "left_wrist"
            )

    def close(self) -> None:
        self.closed = True


class _LazyFakeApplication(_FakeApplication):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def run_human_input_acceptance_check(
        self,
        *,
        timeout_sec: float,
        cancel_requested,
    ):
        assert cancel_requested() is False
        return super().run_human_input_acceptance_check(
            timeout_sec=timeout_sec
        )

    def start(self, *, cancel_requested):
        assert cancel_requested() is False
        return super().start()

    def close(self) -> None:
        self.closed = True


def _intent(
    controller: SimHILAcceptanceController,
    *,
    sequence: int,
    kind: str,
    stage_id: str | None = None,
    control_mode: str | None = None,
    allocation_mode: str | None = None,
    control_profile_id: str | None = None,
    sim_collision_stop_enabled: bool | None = None,
    alpha: float | None = None,
    fallback_source: str | None = None,
    blend_failure_grace_sec: float | None = None,
) -> SimHILAcceptanceIntent:
    return SimHILAcceptanceIntent(
        client_id="test-client",
        client_sequence=sequence,
        expected_revision=controller.status()["revision"],
        intent_kind=kind,
        stage_id=stage_id,
        control_mode=control_mode,
        allocation_mode=allocation_mode,
        control_profile_id=control_profile_id,
        sim_collision_stop_enabled=sim_collision_stop_enabled,
        human_alpha=alpha,
        fallback_source=fallback_source,
        blend_failure_grace_sec=blend_failure_grace_sec,
        reason="test",
    )


def _submit_and_process(
    controller: SimHILAcceptanceController,
    intent: SimHILAcceptanceIntent,
) -> dict[str, object]:
    receipt = controller.submit(intent)
    assert receipt.accepted, receipt
    result = controller.process_next()
    assert result is not None
    return result


def test_acceptance_intent_json_is_exact_and_blend_is_bounded() -> None:
    value = sim_hil_acceptance_intent_from_json(
        {
            "client_id": "browser",
            "client_sequence": 1,
            "expected_revision": 0,
            "intent_kind": "start_stage",
            "stage_id": "proportional_blend",
            "human_alpha": 0.4,
        }
    )
    assert value.human_alpha == 0.4
    assert value.intent_id.startswith("sim-hil-acceptance-intent:")

    fallback = sim_hil_acceptance_intent_from_json(
        {
            "client_id": "browser",
            "client_sequence": 2,
            "expected_revision": 0,
            "intent_kind": "set_blend_failure_fallback",
            "fallback_source": "human",
            "blend_failure_grace_sec": 0.25,
        }
    )
    assert fallback.fallback_source == "human"
    assert fallback.blend_failure_grace_sec == pytest.approx(0.25)

    rebuild = sim_hil_acceptance_intent_from_json(
        {
            "client_id": "browser",
            "client_sequence": 3,
            "expected_revision": 0,
            "intent_kind": "rebuild_runtime",
        }
    )
    assert rebuild.intent_kind == "rebuild_runtime"
    runtime_home = sim_hil_acceptance_intent_from_json(
        {
            "client_id": "browser",
            "client_sequence": 4,
            "expected_revision": 0,
            "intent_kind": "capture_runtime_home",
        }
    )
    assert runtime_home.intent_kind == "capture_runtime_home"

    formal_mode = sim_hil_acceptance_intent_from_json(
        {
            "client_id": "browser",
            "client_sequence": 5,
            "expected_revision": 0,
            "intent_kind": "set_control_mode",
            "control_mode": "proportional_blend",
            "human_alpha": 0.35,
        }
    )
    assert formal_mode.control_mode == "proportional_blend"
    assert formal_mode.human_alpha == 0.35

    allocation_mode = sim_hil_acceptance_intent_from_json(
        {
            "client_id": "browser",
            "client_sequence": 6,
            "expected_revision": 0,
            "intent_kind": "set_allocation_mode",
            "allocation_mode": "relative_overlay",
        }
    )
    assert allocation_mode.allocation_mode == "relative_overlay"

    absolute_allocation_mode = sim_hil_acceptance_intent_from_json(
        {
            "client_id": "browser",
            "client_sequence": 7,
            "expected_revision": 0,
            "intent_kind": "set_allocation_mode",
            "allocation_mode": "absolute_overlay_owner",
        }
    )
    assert absolute_allocation_mode.allocation_mode == "absolute_overlay_owner"

    allocation_start = sim_hil_acceptance_intent_from_json(
        {
            "client_id": "browser",
            "client_sequence": 7,
            "expected_revision": 0,
            "intent_kind": "start_allocation_runtime",
        }
    )
    assert allocation_start.control_mode is None
    assert allocation_start.human_alpha is None

    rebase = sim_hil_acceptance_intent_from_json(
        {
            "client_id": "browser",
            "client_sequence": 8,
            "expected_revision": 0,
            "intent_kind": "request_residual_rebase_and_clear",
        }
    )
    assert rebase.allocation_mode is None

    control_profile = sim_hil_acceptance_intent_from_json(
        {
            "client_id": "browser",
            "client_sequence": 9,
            "expected_revision": 0,
            "intent_kind": "select_control_profile",
            "control_profile_id": "policy_human_allocation_scheme_b",
        }
    )
    assert control_profile.control_profile_id == (
        "policy_human_allocation_scheme_b"
    )

    with pytest.raises(ValueError, match="requires control_profile_id"):
        sim_hil_acceptance_intent_from_json(
            {
                "client_id": "browser",
                "client_sequence": 10,
                "expected_revision": 0,
                "intent_kind": "select_control_profile",
            }
        )

    with pytest.raises(ValueError, match="requires hold, human, or policy"):
        sim_hil_acceptance_intent_from_json(
            {
                "client_id": "browser",
                "client_sequence": 9,
                "expected_revision": 0,
                "intent_kind": "set_blend_failure_fallback",
                "fallback_source": "automatic",
            }
        )


def test_gui_control_profile_selection_is_preconstruction_and_atomic(
    tmp_path,
) -> None:
    coordinate_profile = tmp_path / "coordinate.json"
    coordinate_profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(
        input_ready=True,
        current_profile_path=coordinate_profile,
    )
    application = _LazyFakeApplication()
    factory_calls: list[tuple[float, str, bool]] = []

    def application_factory(
        _path,
        _role,
        alpha,
        control_profile_id,
        sim_collision_stop_enabled,
    ):
        factory_calls.append(
            (
                float(alpha),
                str(control_profile_id),
                bool(sim_collision_stop_enabled),
            )
        )
        return application

    controller = SimHILAcceptanceController(
        None,
        application_factory=application_factory,
        application_factory_accepts_control_profile=True,
        application_factory_accepts_sim_collision_stop=True,
        vive_setup_controller=setup,
        vive_setup_factory=lambda role: _FakeViveSetup(
            input_ready=True,
            source_role=role,
            current_profile_path=coordinate_profile,
        ),
        coordinate_profile_path=coordinate_profile,
        selected_control_profile_id="fixed_alpha_compatibility",
        available_control_profile_ids=(
            "fixed_alpha_compatibility",
            "exclusive_truncated_hitl",
            "policy_human_allocation_scheme_a",
            "policy_human_allocation_scheme_b",
        ),
    )

    selected = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="select_control_profile",
            control_profile_id="policy_human_allocation_scheme_b",
        ),
    )
    assert selected["status"] == "control_profile_selected"
    status = controller.status()
    assert status["selected_control_profile_id"] == (
        "policy_human_allocation_scheme_b"
    )
    assert status["allocation_runtime_enabled"] is True
    assert status["exclusive_hitl_enabled"] is False
    assert status["control_profile_selection_locked"] is False

    collision_selected = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="select_sim_collision_stop",
            sim_collision_stop_enabled=False,
        ),
    )
    assert collision_selected["status"] == "sim_collision_stop_selected"
    assert controller.status()[
        "selected_sim_collision_stop_enabled"
    ] is False

    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=3,
            kind="start_allocation_runtime",
        ),
    )
    assert started["status"] == "running"
    assert factory_calls == [
        (0.0, "policy_human_allocation_scheme_b", False)
    ]
    assert controller.status()["control_profile_selection_locked"] is True

    rejected = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=4,
            kind="select_control_profile",
            control_profile_id="exclusive_truncated_hitl",
        ),
    )
    assert rejected["status"] == "failed"
    assert "fixed after MuJoCo construction" in rejected["error"]
    assert controller.status()["selected_control_profile_id"] == (
        "policy_human_allocation_scheme_b"
    )

    collision_rejected = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=5,
            kind="select_sim_collision_stop",
            sim_collision_stop_enabled=True,
        ),
    )
    assert collision_rejected["status"] == "failed"
    assert "fixed after MuJoCo construction" in collision_rejected["error"]

    rebuilt = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=6,
            kind="rebuild_runtime",
        ),
    )
    assert rebuilt["status"] == "runtime_rebuilt"
    assert controller.status()["control_profile_selection_locked"] is False
    switched = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=7,
            kind="select_control_profile",
            control_profile_id="exclusive_truncated_hitl",
        ),
    )
    assert switched["status"] == "control_profile_selected"
    assert controller.status()["exclusive_hitl_enabled"] is True


def test_collision_stop_intent_requires_an_explicit_boolean() -> None:
    intent = sim_hil_acceptance_intent_from_json(
        {
            "client_id": "browser",
            "client_sequence": 1,
            "expected_revision": 0,
            "intent_kind": "select_sim_collision_stop",
            "sim_collision_stop_enabled": False,
        }
    )
    assert intent.sim_collision_stop_enabled is False

    with pytest.raises(ValueError, match="requires a boolean"):
        sim_hil_acceptance_intent_from_json(
            {
                "client_id": "browser",
                "client_sequence": 2,
                "expected_revision": 0,
                "intent_kind": "select_sim_collision_stop",
                "sim_collision_stop_enabled": "false",
            }
        )


def test_allocation_buttons_are_executed_through_the_acceptance_owner() -> None:
    application = _FakeApplication()
    application.started = True
    controller = SimHILAcceptanceController(application)

    selected = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="set_allocation_mode",
            allocation_mode="base_only",
        ),
    )
    assert selected["status"] == "allocation_mode_selected"
    assert selected["allocation_mode"] == "base_only"
    assert selected["allocation_override_receipt_id"] == "override:1"

    absolute = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="set_allocation_mode",
            allocation_mode="absolute_overlay_owner",
        ),
    )
    assert absolute["status"] == "allocation_mode_selected"
    assert absolute["allocation_mode"] == "absolute_overlay_owner"

    rebased = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=3,
            kind="request_residual_rebase_and_clear",
        ),
    )
    assert rebased["status"] == "residual_rebase_requested"
    assert rebased["anchor_request_id"] == "anchor:1"
    assert application.allocation_operations == [
        "allocation_base_only",
        "allocation_absolute_overlay_owner",
        "residual_rebase_and_clear",
    ]

    with pytest.raises(ValueError, match="Unknown acceptance intent"):
        sim_hil_acceptance_intent_from_json(
            {
                "client_id": "browser",
                "client_sequence": 1,
                "expected_revision": 0,
                "intent_kind": "hold",
                "raw_action": [1, 2, 3],
            }
        )


def test_allocation_runtime_start_has_no_fixed_alpha_mode_semantics() -> None:
    application = _FakeApplication()
    controller = SimHILAcceptanceController(
        application,
        allocation_runtime_enabled=True,
    )

    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_allocation_runtime",
        ),
    )

    assert started["status"] == "running"
    assert started["control_profile_start"] == "allocation"
    assert started["human_input_required"] is False
    assert started["human_input_deferred"] is True
    assert started["initial_allocation_mode"] == "automatic"
    assert "control_mode" not in started
    assert "human_alpha" not in started
    assert application.alpha_updates == []
    assert application.calls == ["start"]
    assert controller.status()["active_control_mode"] is None
    assert controller.status()["available_control_modes"] == []


def test_allocation_profile_rejects_fixed_alpha_intent_before_application() -> None:
    application = _FakeApplication()
    application.started = True
    controller = SimHILAcceptanceController(
        application,
        allocation_runtime_enabled=True,
    )

    failed = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="set_control_mode",
            control_mode="human_teleop",
        ),
    )

    assert failed["status"] == "failed"
    assert "use set_allocation_mode" in failed["error"]
    assert application.alpha_updates == []


def test_allocation_runtime_start_is_rejected_without_profile_capability() -> None:
    controller = SimHILAcceptanceController(_FakeApplication())

    failed = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_allocation_runtime",
        ),
    )

    assert failed["status"] == "failed"
    assert "requires an allocation-aware application profile" in failed["error"]


def test_formal_runtime_modes_are_freely_switchable_and_not_gated() -> None:
    application = _FakeApplication()
    capture = _FakeTraceCapture()
    controller = SimHILAcceptanceController(
        application,
        trace_capture=capture,
    )

    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="ppo_only",
        ),
    )
    assert started["status"] == "running"
    assert application.alpha_updates == [0.0]
    assert capture.prepared
    assert controller.status()["formal_runtime_started_at"] is not None
    assert controller.status()["formal_runtime_finalized_at"] is None

    blended = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="set_control_mode",
            control_mode="proportional_blend",
            alpha=0.35,
        ),
    )
    assert blended["status"] == "mode_selected"
    assert blended["human_alpha"] == 0.35

    updated_blend = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=3,
            kind="set_control_mode",
            control_mode="proportional_blend",
            alpha=0.9,
        ),
    )
    assert updated_blend["status"] == "mode_selected"
    assert updated_blend["human_alpha"] == 0.9
    assert application.alpha_updates[-2:] == [0.35, 0.9]

    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=4,
            kind="set_control_mode",
            control_mode="human_teleop",
        ),
    )
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=5,
            kind="set_control_mode",
            control_mode="ppo_only",
        ),
    )
    status = controller.status()
    assert status["active_control_mode"] == "ppo_only"
    assert status["active_human_alpha"] == 0.0
    assert status["control_mode_change_count"] == 5
    assert status["completed_stage_ids"] == []

    finalized = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=6,
            kind="finalize_capture",
        ),
    )
    assert finalized["status"] == "complete"
    assert capture.finalized
    assert controller.status()["formal_runtime_finalized_at"] is not None


def test_human_demonstration_profile_rejects_policy_and_commits_human_only() -> None:
    application = _FakeApplication()
    capture = _FakeTraceCapture()
    controller = SimHILAcceptanceController(
        application,
        trace_capture=capture,
        capture_profile="human_demonstration",
    )

    assert controller.status()["capture_profile"] == "human_demonstration"
    assert controller.status()["available_control_modes"] == ["human_teleop"]

    policy = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="ppo_only",
        ),
    )
    assert policy["status"] == "failed"
    assert "permits only human_teleop" in policy["error"]

    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="prepare_sim_preview",
        ),
    )
    assert started["status"] == "viewer_preview"
    assert capture.prepared is False

    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=3,
            kind="start_runtime",
            control_mode="human_teleop",
        ),
    )
    assert started["status"] == "running"
    assert application.calls[:4] == [
        "start_viewer_preview",
        f"device:{controller.device_check_timeout_sec}",
        "capture_runtime_home",
        "start",
    ]
    assert capture.prepared

    fallback = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=4,
            kind="set_human_timeout_fallback",
            fallback_source="policy",
        ),
    )
    assert fallback["status"] == "failed"
    assert "only HOLD fallback" in fallback["error"]

    finalized = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=5,
            kind="finalize_capture",
        ),
    )
    assert finalized["status"] == "complete"
    assert capture.finalized


def test_human_demo_preview_defers_devices_control_and_trace_until_start(
    tmp_path,
) -> None:
    profile = tmp_path / "coordinate.json"
    profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(input_ready=False)

    class DeferredDemoApplication(_LazyFakeApplication):
        def __init__(self) -> None:
            super().__init__()
            self.human_input_attached = False

        def prepare_human_attachment(
            self,
            *,
            coordinate_profile_path,
            source_role,
        ):
            assert self.viewer_preview_started is True
            assert coordinate_profile_path == profile
            assert source_role == "left_wrist"
            self.calls.append("prepare_human")
            return SimpleNamespace(
                source=SimpleNamespace(close=lambda: None),
            )

        def activate_human_attachment(self, _attachment):
            assert self.viewer_preview_started is True
            assert self.started is False
            self.human_input_attached = True
            self.calls.append("attach_human")
            return {
                "status": "attached",
                "command_dispatched": False,
                "authoritative": False,
            }

    app = DeferredDemoApplication()
    capture = _FakeTraceCapture()
    factory_alphas: list[float] = []

    def application_factory(_path, _role, initial_alpha):
        app.initial_alpha = float(initial_alpha)
        factory_alphas.append(float(initial_alpha))
        return app

    controller = SimHILAcceptanceController(
        None,
        trace_capture=capture,
        application_factory=application_factory,
        vive_setup_controller=setup,
        vive_setup_factory=lambda role: _FakeViveSetup(
            input_ready=True,
            source_role=role,
        ),
        coordinate_profile_path=profile,
        capture_profile="human_demonstration",
    )

    preview = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="prepare_sim_preview",
        ),
    )

    assert preview["status"] == "viewer_preview"
    assert preview["trace_recording"] is False
    assert factory_alphas == [1.0]
    assert app.calls == ["start_viewer_preview"]
    assert app.started is False
    assert app.human_input_attached is False
    assert setup.closed is False
    assert capture.prepared is False
    assert controller.status()["sim_preview_ready"] is True
    assert controller.status()["setup"]["phase"] == "sim_preview_ready"

    setup.input_ready = True
    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="start_runtime",
            control_mode="human_teleop",
        ),
    )

    assert started["status"] == "running"
    assert started["mode_activation"] == "initial_profile"
    assert app.calls == [
        "start_viewer_preview",
        "prepare_human",
        "attach_human",
        f"device:{controller.device_check_timeout_sec}",
        "capture_runtime_home",
        "start",
    ]
    assert app.alpha_updates == []
    assert app.human_input_attached is True
    assert setup.closed is True
    assert capture.prepared is True
    assert controller.status()["sim_preview_ready"] is False


def test_human_demo_rejects_recording_start_before_preview() -> None:
    application = _FakeApplication()
    controller = SimHILAcceptanceController(
        application,
        trace_capture=_FakeTraceCapture(),
        capture_profile="human_demonstration",
    )

    result = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="human_teleop",
        ),
    )

    assert result["status"] == "failed"
    assert "open the passive Sim preview" in result["error"]
    assert application.started is False


def test_human_demo_device_preflight_failure_keeps_camera_preview_for_retry() -> None:
    class RetryableApplication(_FakeApplication):
        def __init__(self) -> None:
            super().__init__()
            self.input_checks = 0

        def run_human_input_acceptance_check(self, *, timeout_sec: float):
            self.input_checks += 1
            self.calls.append(f"device:{timeout_sec}")
            if self.input_checks == 1:
                raise TimeoutError("angle trigger is not ready")
            return {
                "status": "ready",
                "command_dispatched": False,
                "authoritative": False,
            }

    application = RetryableApplication()
    capture = _FakeTraceCapture()
    controller = SimHILAcceptanceController(
        application,
        trace_capture=capture,
        capture_profile="human_demonstration",
    )
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="prepare_sim_preview",
        ),
    )

    failed = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="start_runtime",
            control_mode="human_teleop",
        ),
    )

    assert failed["status"] == "failed"
    assert application.viewer_preview_started is True
    assert application.started is False
    assert capture.prepared is False
    assert capture.aborted is False
    assert controller.status()["sim_preview_ready"] is True

    retried = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=3,
            kind="start_runtime",
            control_mode="human_teleop",
        ),
    )

    assert retried["status"] == "running"
    assert application.started is True
    assert capture.prepared is True


def test_exclusive_hitl_switches_policy_human_policy_with_one_trace() -> None:
    application = _ExclusiveFakeApplication()
    capture = _FakeTraceCapture()
    controller = SimHILAcceptanceController(
        application,
        trace_capture=capture,
        allocation_runtime_enabled=True,
        exclusive_hitl_enabled=True,
    )

    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="ppo_only",
        ),
    )
    assert started["status"] == "running"

    takeover = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="request_human_takeover",
        ),
    )
    assert takeover["status"] == "switch_ready"
    assert takeover["control_mode"] == "human_teleop"
    assert takeover["human_alpha"] == 1.0
    human_switch = takeover["exclusive_hitl_switch"]
    assert human_switch["direction"] == EXCLUSIVE_HITL_POLICY_TO_HUMAN
    assert human_switch["target_source"] == "human"
    assert human_switch["home_capture_id"] == "runtime-home:1"
    assert human_switch["home_capture"]["sample_count_by_side"] == {"right": 5}
    assert human_switch["source_selection_record_id"].startswith(
        "override:1"
    )
    assert human_switch["activation_id"] == "activation:1"
    assert human_switch["target_emission_id"] == "human-emission:1"
    assert human_switch["source_reset_performed"] is False

    calls_before_duplicate = list(application.calls)
    duplicate = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=3,
            kind="request_human_takeover",
        ),
    )
    assert duplicate["status"] == "switch_ready"
    assert duplicate["exclusive_hitl_switch"]["status"] == "already_active"
    assert duplicate["exclusive_hitl_switch"]["previous_source"] == "human"
    assert duplicate["exclusive_hitl_switch"]["target_source"] == "human"
    assert application.calls == calls_before_duplicate

    resume = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=4,
            kind="request_policy_resume",
        ),
    )
    assert resume["status"] == "switch_ready"
    assert resume["control_mode"] == "ppo_only"
    assert resume["human_alpha"] == 0.0
    policy_switch = resume["exclusive_hitl_switch"]
    assert policy_switch["direction"] == EXCLUSIVE_HITL_HUMAN_TO_POLICY
    assert policy_switch["home_capture"] is None
    assert policy_switch["activation_id"] == "activation:2"
    assert policy_switch["target_emission_id"] == "policy-emission:2"
    assert policy_switch["source_reset_performed"] is True

    assert application.calls == [
        "start",
        "hold:human_mode_preflight",
        f"device:{controller.device_check_timeout_sec}",
        "capture_runtime_home",
        "rearm",
        "establish:human",
        "hold:exclusive_hitl_policy_resume",
        "rearm",
        "establish:policy",
    ]
    assert application.allocation_operations == [
        "allocation_absolute_overlay_owner",
        "allocation_base_only",
    ]
    assert [fact.event_type for fact in capture.facts] == [
        "exclusive_hitl_switch",
        "exclusive_hitl_switch",
        "exclusive_hitl_switch",
    ]
    assert tuple(capture.facts[0].payload["home_capture"][
        "human_home_pose_desk_by_side"
    ]["right"]) == (1.0,) * 16
    status = controller.status()
    assert status["active_control_mode"] == "ppo_only"
    assert status["last_exclusive_hitl_switch"]["target_source"] == "policy"


def test_exclusive_hitl_cannot_switch_source_from_collision_hold() -> None:
    application = _ExclusiveFakeApplication()
    controller = SimHILAcceptanceController(
        application,
        allocation_runtime_enabled=True,
        exclusive_hitl_enabled=True,
    )
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="ppo_only",
        ),
    )
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="request_human_takeover",
        ),
    )
    application.hold(reason="collision_margin_low")

    rejected = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=3,
            kind="request_policy_resume",
        ),
    )

    assert rejected["status"] == "failed"
    assert "requires a running episode" in rejected["error"]
    assert application.status()["held"] is True
    assert "exclusive_hitl_policy_recovery_from_hold" not in application.calls


def test_exclusive_hitl_trace_failure_is_observable_but_not_control() -> None:
    class FailingTraceCapture(_FakeTraceCapture):
        def admit_trace(self, fact: ControlTraceFact) -> None:
            raise RuntimeError(f"trace-rejected:{fact.fact_id}")

    application = _ExclusiveFakeApplication()
    controller = SimHILAcceptanceController(
        application,
        trace_capture=FailingTraceCapture(),
        allocation_runtime_enabled=True,
        exclusive_hitl_enabled=True,
    )
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="ppo_only",
        ),
    )

    takeover = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="request_human_takeover",
        ),
    )

    assert takeover["status"] == "switch_ready"
    assert application.status()["held"] is False
    status = controller.status()
    assert status["active_control_mode"] == "human_teleop"
    assert status["last_exclusive_hitl_trace_error"].startswith(
        "RuntimeError:trace-rejected:"
    )


def test_exclusive_hitl_accepts_prestart_human_selection_and_rejects_blend() -> None:
    application = _ExclusiveFakeApplication()
    controller = SimHILAcceptanceController(
        application,
        allocation_runtime_enabled=True,
        exclusive_hitl_enabled=True,
    )

    human_start = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="human_teleop",
        ),
    )
    assert human_start["status"] == "running"
    assert human_start["control_mode"] == "human_teleop"
    assert human_start["initial_control_mode_requested"] == "human_teleop"
    assert human_start["exclusive_hitl_switch"]["target_source"] == "human"
    assert application.calls == [
        "start",
        "hold:human_mode_preflight",
        f"device:{controller.device_check_timeout_sec}",
        "capture_runtime_home",
        "rearm",
        "establish:human",
    ]
    direct_mode = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="set_control_mode",
            control_mode="human_teleop",
        ),
    )
    assert direct_mode["status"] == "failed"
    assert "explicit takeover/resume triggers" in direct_mode["error"]


def test_exclusive_hitl_missing_fresh_emission_stays_held_and_is_recorded() -> None:
    class MissingPrimeApplication(_ExclusiveFakeApplication):
        def rearm_with_allocation_operation(self, **kwargs):
            result = super().rearm_with_allocation_operation(**kwargs)
            self.last_activation_prime_evidence = None
            return result

    application = MissingPrimeApplication()
    capture = _FakeTraceCapture()
    controller = SimHILAcceptanceController(
        application,
        trace_capture=capture,
        allocation_runtime_enabled=True,
        exclusive_hitl_enabled=True,
    )
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="ppo_only",
        ),
    )

    failed = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="request_human_takeover",
        ),
    )

    assert failed["status"] == "failed"
    assert "no activation-prime evidence" in failed["error"]
    assert application.status()["held"] is True
    assert application.calls[-1] == "hold:exclusive_hitl_switch_failed"
    assert capture.facts[-1].payload["status"] == "failed"
    assert capture.facts[-1].payload["target_source"] == "human"


def test_exclusive_hitl_does_not_publish_source_mode_before_target_admission() -> None:
    class MissingTargetApplication(_ExclusiveFakeApplication):
        def establish_activation_execution(self, *, target_source: str):
            self.calls.append(f"establish:{target_source}")
            raise RuntimeError("target admission unavailable")

    application = MissingTargetApplication()
    capture = _FakeTraceCapture()
    controller = SimHILAcceptanceController(
        application,
        trace_capture=capture,
        allocation_runtime_enabled=True,
        exclusive_hitl_enabled=True,
    )
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="ppo_only",
        ),
    )

    failed = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="request_human_takeover",
        ),
    )

    assert failed["status"] == "failed"
    assert "target admission unavailable" in failed["error"]
    assert application.status()["held"] is True
    assert controller.status()["active_control_mode"] == "ppo_only"
    assert capture.facts[-1].payload["status"] == "failed"


def test_exclusive_hitl_trace_fact_is_accepted_by_formal_capture(tmp_path) -> None:
    capture = SimHILAcceptanceTraceCapture(
        tmp_path,
        trace_session_id="exclusive-hitl-trace-test",
        max_spool_bytes=1 << 20,
    )
    home = {
        "capture_id": "runtime-home:trace",
        "captured_at": 10.1,
        "clock_domain_id": capture.clock_domain_id,
        "human_home_pose_desk_by_side": {"right": [1.0] * 16},
        "robot_home_pose_desk_by_side": {"right": [2.0] * 16},
        "command_dispatched": False,
        "authoritative": False,
    }
    try:
        capture.prepare(now=10.0)
        evidence = ExclusiveHITLSwitchEvidence(
            switch_id="exclusive-hitl-switch:trace",
            trigger_intent_id="intent:trace",
            trigger_intent_hash="intent-hash:trace",
            direction=EXCLUSIVE_HITL_POLICY_TO_HUMAN,
            previous_source="policy",
            target_source="human",
            requested_at=10.0,
            completed_at=10.2,
            clock_domain_id=capture.clock_domain_id,
            succeeded=True,
            status="ready",
            hold_evidence_refs=("lower-hold:trace",),
            home_capture_id="runtime-home:trace",
            home_capture_hash="runtime-home-hash:trace",
            home_capture=home,
            source_selection_record_id="alpha-record:trace",
            source_selection_record_hash="alpha-record-hash:trace",
            activation_id="activation:trace",
            activation_hash="activation-hash:trace",
            target_emission_id="human-emission:trace",
            target_emission_hash="human-emission-hash:trace",
            target_id="target:trace",
            target_hash="target-hash:trace",
            target_admission_receipt_id="target-admission:trace",
            target_completion_receipt_id="target-completion:trace",
            command_id="command:trace",
            dispatch_receipt_id="dispatch:trace",
        )
        fact = project_exclusive_hitl_switch(
            evidence,
            runtime_session_id=capture.runtime_session_id,
            control_domain_id=capture.control_domain_id,
        )

        capture.admit_trace(fact)

        records = [
            record
            for record in capture.trace_runtime.records()
            if record.fact.event_type == "exclusive_hitl_switch"
        ]
        assert len(records) == 1
        assert dict(records[0].fact.payload["home_capture"])["capture_id"] == (
            home["capture_id"]
        )
        assert records[0].fact.payload["source_selection_record_id"] == (
            "alpha-record:trace"
        )
    finally:
        capture.abort(now=10.3)
        capture.close(now=10.4)


def test_blend_failure_fallback_is_a_typed_station_intent() -> None:
    application = _FakeApplication()
    application.started = True
    controller = SimHILAcceptanceController(application)

    result = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="set_blend_failure_fallback",
            fallback_source="policy",
            blend_failure_grace_sec=0.3,
        ),
    )

    assert result["status"] == "selected"
    assert result["source"] == "policy"
    assert result["grace_sec"] == pytest.approx(0.3)
    assert application.blend_failure_fallback == "policy"
    assert application.blend_failure_grace_sec == pytest.approx(0.3)
    assert application.calls == ["blend_fallback:policy"]
    assert controller.status()["available_blend_failure_fallbacks"] == [
        "hold",
        "human",
        "policy",
    ]


def test_blend_failure_grace_is_nonnegative_and_intent_scoped() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        SimHILAcceptanceIntent(
            client_id="test",
            client_sequence=1,
            expected_revision=0,
            intent_kind="set_blend_failure_fallback",
            fallback_source="human",
            blend_failure_grace_sec=-0.01,
        )
    with pytest.raises(ValueError, match="only valid"):
        SimHILAcceptanceIntent(
            client_id="test",
            client_sequence=2,
            expected_revision=0,
            intent_kind="hold",
            blend_failure_grace_sec=0.2,
        )


def test_human_timeout_fallback_is_a_separate_typed_intent() -> None:
    intent = SimHILAcceptanceIntent(
        client_id="test",
        client_sequence=1,
        expected_revision=0,
        intent_kind="set_human_timeout_fallback",
        fallback_source="policy",
    )
    assert intent.fallback_source == "policy"
    with pytest.raises(ValueError, match="requires hold or policy"):
        SimHILAcceptanceIntent(
            client_id="test",
            client_sequence=2,
            expected_revision=0,
            intent_kind="set_human_timeout_fallback",
            fallback_source="human",
        )


def test_policy_human_policy_human_reactivates_sources_and_rebinds_home() -> None:
    class HomeTrackingApplication(_FakeApplication):
        def __init__(self) -> None:
            super().__init__()
            self.runtime_home = None

        def capture_runtime_home(self):
            receipt = super().capture_runtime_home()
            self.runtime_home = dict(receipt)
            return receipt

        def status(self):
            value = super().status()
            value["runtime_home"] = self.runtime_home
            return value

    application = HomeTrackingApplication()
    controller = SimHILAcceptanceController(application)
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="ppo_only",
        ),
    )
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="set_control_mode",
            control_mode="human_teleop",
        ),
    )
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=3,
            kind="set_control_mode",
            control_mode="ppo_only",
        ),
    )
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=4,
            kind="set_control_mode",
            control_mode="human_teleop",
        ),
    )

    assert application.runtime_home_captures == 2
    assert application.calls.count("hold:human_mode_preflight") == 2
    assert application.calls.count("hold:policy_mode_preflight") == 1
    assert application.calls.count("rearm") == 3
    assert controller.status()["active_control_mode"] == "human_teleop"


def test_lazy_acceptance_keeps_mujoco_unconstructed_until_vive_setup_ready(
    tmp_path,
) -> None:
    profile = tmp_path / "coordinate.json"
    profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(input_ready=False)
    factory_calls: list[tuple[object, object]] = []
    controller = SimHILAcceptanceController(
        None,
        application_factory=lambda path, role, _alpha: factory_calls.append(
            (path, role)
        ),
        vive_setup_controller=setup,
        vive_setup_factory=lambda _role: _FakeViveSetup(input_ready=False),
        coordinate_profile_path=profile,
    )

    status = controller.status()
    assert status["setup"]["runtime_launch_ready"] is False
    assert status["setup"]["input_ready"] is False
    assert controller.current_application is None

    failed = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_stage",
            stage_id="human_teleop",
        ),
    )

    assert failed["status"] == "failed"
    assert "blocked until live VIVE input" in failed["error"]
    assert factory_calls == []
    assert setup.closed is False
    assert controller.current_application is None


def test_lazy_acceptance_freezes_loaded_latest_profile_for_runtime(
    tmp_path,
) -> None:
    resolved_profile = tmp_path / "resolved-latest.json"
    resolved_profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(
        input_ready=True,
        current_profile_path=resolved_profile,
    )
    app = _LazyFakeApplication()
    factory_paths = []

    def factory(path, _role, _alpha):
        factory_paths.append(path)
        return app

    controller = SimHILAcceptanceController(
        None,
        application_factory=factory,
        vive_setup_controller=setup,
        vive_setup_factory=lambda role: _FakeViveSetup(
            input_ready=True,
            source_role=role,
            current_profile_path=resolved_profile,
        ),
        coordinate_profile_path="latest",
    )

    status = controller.status()
    assert status["setup"]["selected_profile_path"] == str(resolved_profile)
    assert status["setup"]["profile_ready"] is True

    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="human_teleop",
        ),
    )

    assert started["status"] == "running"
    assert factory_paths == [resolved_profile]


def test_lazy_acceptance_hands_off_one_vive_owner_before_runtime_build(
    tmp_path,
) -> None:
    profile = tmp_path / "coordinate.json"
    profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(
        input_ready=True,
        source_role="right_wrist",
    )
    app = _LazyFakeApplication()
    factory_calls: list[tuple[object, object]] = []

    def factory(path, role, _alpha):
        assert setup.closed is False
        factory_calls.append((path, role))
        return app

    controller = SimHILAcceptanceController(
        None,
        application_factory=factory,
        vive_setup_controller=setup,
        vive_setup_factory=lambda role: _FakeViveSetup(
            input_ready=True,
            source_role=role,
        ),
        coordinate_profile_path=profile,
    )

    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_stage",
            stage_id="human_teleop",
        ),
    )

    assert started["status"] == "awaiting_confirmation"
    assert factory_calls == [(profile, "right_wrist")]
    assert setup.closed is True
    assert controller.current_application is app
    assert app.started is True
    assert app.runtime_home_captures == 1
    assert controller.status()["setup"]["phase"] == "runtime_constructed"


@pytest.mark.parametrize(
    ("control_mode", "requested_alpha", "expected_alpha"),
    [
        ("human_teleop", None, 1.0),
        ("ppo_only", None, 0.0),
        ("proportional_blend", 0.6, 0.6),
    ],
)
def test_lazy_formal_control_can_start_directly_in_any_mode(
    tmp_path,
    control_mode,
    requested_alpha,
    expected_alpha,
) -> None:
    profile = tmp_path / "coordinate.json"
    profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(input_ready=True)
    app = _LazyFakeApplication()
    factory_alphas: list[float] = []

    def application_factory(_path, _role, initial_alpha):
        app.initial_alpha = float(initial_alpha)
        factory_alphas.append(float(initial_alpha))
        return app

    controller = SimHILAcceptanceController(
        None,
        trace_capture=_FakeTraceCapture(),
        application_factory=application_factory,
        vive_setup_controller=setup,
        vive_setup_factory=lambda role: _FakeViveSetup(
            input_ready=True,
            source_role=role,
        ),
        coordinate_profile_path=profile,
    )

    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode=control_mode,
            alpha=requested_alpha,
        ),
    )

    assert started["status"] == "running"
    assert started["control_mode"] == control_mode
    assert factory_alphas == [expected_alpha]
    assert app.alpha_updates == []
    expected_mix_mode = {
        "human_teleop": "human_only",
        "ppo_only": "policy_only",
        "proportional_blend": "proportional_blend",
    }[control_mode]
    assert app.status()["composition"]["mix"]["mode"] == expected_mix_mode
    assert setup.closed is (control_mode != "ppo_only")
    if control_mode == "ppo_only":
        assert (
            controller.status()["setup"]["phase"]
            == "policy_runtime_human_setup_available"
        )


def test_lazy_allocation_start_defers_human_like_policy_only_mode(
    tmp_path,
) -> None:
    profile = tmp_path / "coordinate.json"
    profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(input_ready=True)
    app = _LazyFakeApplication()
    factory_alphas: list[float] = []

    def application_factory(_path, _role, initial_alpha):
        factory_alphas.append(float(initial_alpha))
        return app

    controller = SimHILAcceptanceController(
        None,
        trace_capture=_FakeTraceCapture(),
        application_factory=application_factory,
        vive_setup_controller=setup,
        vive_setup_factory=lambda role: _FakeViveSetup(
            input_ready=True,
            source_role=role,
        ),
        coordinate_profile_path=profile,
        allocation_runtime_enabled=True,
    )

    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_allocation_runtime",
        ),
    )

    assert started["status"] == "running"
    assert started["control_profile_start"] == "allocation"
    assert started["human_input_required"] is False
    assert started["human_input_deferred"] is True
    assert factory_alphas == [0.0]
    assert app.alpha_updates == []
    assert setup.closed is False
    assert controller.status()["setup"]["phase"] == (
        "policy_runtime_human_setup_available"
    )
    assert controller.status()["formal_runtime_started"] is True


def test_deferred_allocation_attaches_human_at_relative_mode_boundary(
    tmp_path,
) -> None:
    profile = tmp_path / "coordinate.json"
    profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(input_ready=True)

    class DeferredAllocationApplication(_LazyFakeApplication):
        def __init__(self) -> None:
            super().__init__()
            self.human_input_attached = False

        def prepare_human_attachment(
            self,
            *,
            coordinate_profile_path,
            source_role,
        ):
            assert coordinate_profile_path == profile
            assert source_role == "left_wrist"
            return SimpleNamespace(
                source=SimpleNamespace(close=lambda: None),
            )

        def activate_human_attachment(self, _attachment):
            assert self._held is True
            self.human_input_attached = True
            self.calls.append("attach_human")
            return {
                "status": "attached",
                "command_dispatched": False,
                "authoritative": False,
            }

    app = DeferredAllocationApplication()
    controller = SimHILAcceptanceController(
        None,
        trace_capture=_FakeTraceCapture(),
        application_factory=lambda _path, _role, _alpha: app,
        vive_setup_controller=setup,
        vive_setup_factory=lambda role: _FakeViveSetup(
            input_ready=True,
            source_role=role,
        ),
        coordinate_profile_path=profile,
        allocation_runtime_enabled=True,
    )
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_allocation_runtime",
        ),
    )

    switched = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="set_allocation_mode",
            allocation_mode="relative_overlay",
        ),
    )

    assert switched["status"] == "allocation_mode_selected"
    assert switched["human_preparation"]["attachment"]["status"] == (
        "attached"
    )
    assert app.human_input_attached is True
    assert app.allocation_operations == ["allocation_relative_overlay"]
    assert app.calls == [
        "start",
        "hold:human_input_attachment",
        "attach_human",
        f"device:{controller.device_check_timeout_sec}",
        "capture_runtime_home",
        "rearm",
    ]
    assert setup.closed is True
    assert controller.status()["setup"]["phase"] == "runtime_constructed"


def test_policy_only_launch_does_not_require_vive_or_coordinate_profile(
    tmp_path,
) -> None:
    missing_profile = tmp_path / "not-created-coordinate.json"
    setup = _FakeViveSetup(input_ready=False)
    app = _LazyFakeApplication()
    factory_calls: list[tuple[object, str, float]] = []

    def application_factory(path, role, initial_alpha):
        factory_calls.append((path, role, float(initial_alpha)))
        app.initial_alpha = float(initial_alpha)
        return app

    controller = SimHILAcceptanceController(
        None,
        trace_capture=_FakeTraceCapture(),
        application_factory=application_factory,
        vive_setup_controller=setup,
        vive_setup_factory=lambda role: _FakeViveSetup(
            input_ready=False,
            source_role=role,
        ),
        coordinate_profile_path=missing_profile,
    )

    before = controller.status()["setup"]
    assert before["policy_runtime_launch_ready"] is True
    assert before["human_runtime_launch_ready"] is False

    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="ppo_only",
        ),
    )

    assert started["status"] == "running"
    assert factory_calls == [
        (missing_profile, "left_wrist", 0.0),
    ]
    assert app.calls == ["start"]
    assert setup.closed is False
    assert controller.status()["setup"]["phase"] == (
        "policy_runtime_human_setup_available"
    )


def test_policy_runtime_attaches_human_input_only_at_held_mode_boundary(
    tmp_path,
) -> None:
    profile = tmp_path / "coordinate.json"
    profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(input_ready=True)

    class DeferredHumanApplication(_LazyFakeApplication):
        def __init__(self) -> None:
            super().__init__()
            self.human_input_attached = False
            self.prepared_roles: list[str] = []

        def prepare_human_attachment(
            self,
            *,
            coordinate_profile_path,
            source_role,
        ):
            assert coordinate_profile_path == profile
            self.prepared_roles.append(source_role)
            return SimpleNamespace(
                source=SimpleNamespace(close=lambda: None),
            )

        def activate_human_attachment(self, _attachment):
            assert self._held is True
            self.human_input_attached = True
            self.calls.append("attach_human")
            return {
                "status": "attached",
                "command_dispatched": False,
                "authoritative": False,
            }

    app = DeferredHumanApplication()
    controller = SimHILAcceptanceController(
        None,
        trace_capture=_FakeTraceCapture(),
        application_factory=lambda _path, _role, alpha: (
            setattr(app, "initial_alpha", float(alpha)) or app
        ),
        vive_setup_controller=setup,
        vive_setup_factory=lambda role: _FakeViveSetup(
            input_ready=True,
            source_role=role,
        ),
        coordinate_profile_path=profile,
    )
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="ppo_only",
        ),
    )

    switched = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="set_control_mode",
            control_mode="human_teleop",
        ),
    )

    assert switched["status"] == "mode_selected"
    assert app.prepared_roles == ["left_wrist"]
    assert app.human_input_attached is True
    assert setup.closed is True
    assert app.calls == [
        "start",
        "hold:human_input_attachment",
        "attach_human",
        f"device:{controller.device_check_timeout_sec}",
        "capture_runtime_home",
        "alpha:1.0",
        "rearm",
    ]
    assert app.status()["composition"]["mix"]["mode"] == "human_only"
    assert controller.status()["setup"]["phase"] == "runtime_constructed"


def test_human_mode_reasserts_authoritative_hold_at_activation_boundary() -> None:
    class GateDriftApplication(_FakeApplication):
        human_input_attached = True

        def __init__(self) -> None:
            super().__init__()
            self.gate_status = "ready"

        def status(self):
            value = super().status()
            value["execution_gate"]["status"] = self.gate_status
            return value

        def hold(self, *, reason: str):
            super().hold(reason=reason)
            self.gate_status = "hold"

        def set_human_blend_alpha(self, alpha: float):
            receipt = super().set_human_blend_alpha(alpha)
            if float(alpha) > 0.0:
                # Model a lower cut becoming READY between the initial
                # attachment HOLD and final mode activation.
                self.gate_status = "ready"
            return receipt

        def rearm(self, *, cancel_requested=None):
            # The real application reconciles its lower gate inside rearm.
            self.hold(reason="sim_hil_application:rearm_boundary")
            report = super().rearm(
                cancel_requested=cancel_requested
            )
            self.gate_status = "ready"
            return report

    app = GateDriftApplication()
    controller = SimHILAcceptanceController(app)
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="ppo_only",
        ),
    )

    switched = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="set_control_mode",
            control_mode="human_teleop",
        ),
    )

    assert switched["status"] == "mode_selected"
    assert app.gate_status == "ready"
    assert app.calls[-3:] == [
        "alpha:1.0",
        "hold:sim_hil_application:rearm_boundary",
        "rearm",
    ]


def test_unready_human_switch_leaves_running_policy_mode_unchanged(
    tmp_path,
) -> None:
    profile = tmp_path / "coordinate.json"
    profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(input_ready=False)
    app = _LazyFakeApplication()
    app.human_input_attached = False
    controller = SimHILAcceptanceController(
        None,
        trace_capture=_FakeTraceCapture(),
        application_factory=lambda _path, _role, alpha: (
            setattr(app, "initial_alpha", float(alpha)) or app
        ),
        vive_setup_controller=setup,
        vive_setup_factory=lambda role: _FakeViveSetup(
            input_ready=False,
            source_role=role,
        ),
        coordinate_profile_path=profile,
    )
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_runtime",
            control_mode="ppo_only",
        ),
    )

    failed = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="set_control_mode",
            control_mode="human_teleop",
        ),
    )

    assert failed["status"] == "failed"
    assert "requires live human input" in failed["error"]
    assert app.status()["composition"]["mix"]["mode"] == "policy_only"
    assert app.calls == ["start"]
    assert app._held is False
    assert setup.closed is False


def test_lazy_runtime_construction_failure_keeps_setup_receiver_alive(
    tmp_path,
) -> None:
    profile = tmp_path / "coordinate.json"
    profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(input_ready=True)
    setup_replacements: list[object] = []

    def fail_factory(_path, _role, _alpha):
        assert setup.closed is False
        raise RuntimeError("synthetic construction failure")

    controller = SimHILAcceptanceController(
        None,
        application_factory=fail_factory,
        vive_setup_controller=setup,
        vive_setup_factory=lambda _role: setup_replacements.append(object()),
        coordinate_profile_path=profile,
    )

    result = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_stage",
            stage_id="human_teleop",
        ),
    )

    assert result["status"] == "failed"
    assert "synthetic construction failure" in result["error"]
    assert setup.closed is False
    assert setup_replacements == []
    assert controller.current_application is None
    assert controller.status()["setup"]["phase"] == "vive_setup"


def test_failed_lazy_start_restores_setup_and_retries_with_fresh_application(
    tmp_path,
) -> None:
    profile = tmp_path / "coordinate.json"
    profile.write_text("{}", encoding="utf-8")
    initial_setup = _FakeViveSetup(input_ready=True)

    class FailedApplication(_LazyFakeApplication):
        restart_required = False

        def start(self, *, cancel_requested):
            assert cancel_requested() is False
            self.restart_required = True
            raise RuntimeError("synthetic viewer startup failure")

    failed_app = FailedApplication()
    retry_app = _LazyFakeApplication()
    applications = iter((failed_app, retry_app))
    setup_replacements: list[_FakeViveSetup] = []

    def setup_factory(role: str) -> _FakeViveSetup:
        setup = _FakeViveSetup(
            input_ready=True,
            source_role=role,
        )
        setup_replacements.append(setup)
        return setup

    controller = SimHILAcceptanceController(
        None,
        application_factory=lambda _path, _role, _alpha: next(applications),
        vive_setup_controller=initial_setup,
        vive_setup_factory=setup_factory,
        coordinate_profile_path=profile,
    )

    first = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_stage",
            stage_id="human_teleop",
        ),
    )

    assert first["status"] == "failed"
    assert "synthetic viewer startup failure" in first["error"]
    assert failed_app.closed is True
    assert controller.current_application is None
    assert controller.status()["setup"]["phase"] == "vive_setup"
    assert len(setup_replacements) == 1

    second = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="start_stage",
            stage_id="human_teleop",
        ),
    )

    assert second["status"] == "awaiting_confirmation"
    assert retry_app.started is True
    assert controller.current_application is retry_app


def test_held_lazy_runtime_can_be_rebuilt_and_rebind_home_before_stage_commit(
    tmp_path,
) -> None:
    profile = tmp_path / "coordinate.json"
    profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(input_ready=True)
    app = _LazyFakeApplication()
    capture = _FakeTraceCapture()
    replacements: list[_FakeViveSetup] = []

    def setup_factory(role: str):
        replacement = _FakeViveSetup(
            input_ready=True,
            source_role=role,
        )
        replacements.append(replacement)
        return replacement

    controller = SimHILAcceptanceController(
        None,
        trace_capture=capture,
        application_factory=lambda _path, _role, _alpha: app,
        vive_setup_controller=setup,
        vive_setup_factory=setup_factory,
        coordinate_profile_path=profile,
    )
    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_stage",
            stage_id="human_teleop",
        ),
    )
    assert started["status"] == "awaiting_confirmation"
    assert capture.prepared is True

    held = _submit_and_process(
        controller,
        _intent(controller, sequence=2, kind="hold"),
    )
    assert held["status"] == "held"
    rebuilt = _submit_and_process(
        controller,
        _intent(controller, sequence=3, kind="rebuild_runtime"),
    )

    assert rebuilt["status"] == "runtime_rebuilt"
    assert app.closed is True
    assert capture.aborted is True
    assert capture.rotation_count == 1
    assert controller.current_application is None
    assert controller.status()["setup"]["phase"] == "vive_setup"
    assert controller.status()["stage_status_by_id"]["human_teleop"] == "available"
    assert len(replacements) == 1


def test_stop_runtime_cancels_active_operation_and_stays_station_queued() -> None:
    controller = SimHILAcceptanceController(_FakeApplication())
    with controller._lock:
        controller._active_intent_id = "active-rearm"
    intent = _intent(controller, sequence=1, kind="stop_runtime")

    receipt = controller.submit(intent)

    assert receipt.accepted
    assert controller._cancel_requested.is_set()
    assert tuple(controller._queue) == (intent,)
    assert controller.shutdown_requested is False

    # The active operation owns the Station thread until it observes the
    # cancellation.  Once it unwinds, the already-queued stop remains the next
    # typed lifecycle mutation and is executed by that same thread.
    with controller._lock:
        controller._active_intent_id = None
    result = controller.process_next()
    assert result is not None
    assert result["status"] == "shutdown_requested"
    assert controller.shutdown_requested is True


def test_cancel_and_stop_do_not_require_a_fresh_status_revision() -> None:
    cancel_controller = SimHILAcceptanceController(_FakeApplication())
    with cancel_controller._lock:
        cancel_controller._active_intent_id = "active-start"
        cancel_controller._revision = 7
    stale_cancel = SimHILAcceptanceIntent(
        client_id="browser",
        client_sequence=1,
        expected_revision=1,
        intent_kind="cancel_operation",
        reason="test",
    )
    assert cancel_controller.submit(stale_cancel).accepted is True
    assert cancel_controller._cancel_requested.is_set()

    stop_controller = SimHILAcceptanceController(_FakeApplication())
    with stop_controller._lock:
        stop_controller._active_intent_id = "active-home"
        stop_controller._revision = 9
    stale_stop = SimHILAcceptanceIntent(
        client_id="browser",
        client_sequence=1,
        expected_revision=0,
        intent_kind="stop_runtime",
        reason="test",
    )
    assert stop_controller.submit(stale_stop).accepted is True
    assert tuple(stop_controller._queue) == (stale_stop,)
    assert stop_controller._cancel_requested.is_set()


def test_stop_runtime_interrupts_active_rearm_then_exits_on_station_thread() -> None:
    entered_rearm = threading.Event()

    class BlockingApplication(_FakeApplication):
        def rearm(self, *, cancel_requested=None):
            entered_rearm.set()
            while cancel_requested is None or not cancel_requested():
                time.sleep(0.001)
            raise InterruptedError("cancelled during rearm")

    controller = SimHILAcceptanceController(BlockingApplication())
    rearm = _intent(controller, sequence=1, kind="rearm")
    assert controller.submit(rearm).accepted
    result: list[dict[str, object] | None] = []
    station_thread = threading.Thread(
        target=lambda: result.append(controller.process_next()),
        daemon=True,
    )
    station_thread.start()
    assert entered_rearm.wait(timeout=1.0)

    stop = _intent(controller, sequence=2, kind="stop_runtime")
    assert controller.submit(stop).accepted
    station_thread.join(timeout=1.0)

    assert not station_thread.is_alive()
    assert result and result[0]["status"] == "failed"
    assert "InterruptedError" in str(result[0]["error"])
    stopped = controller.process_next()
    assert stopped is not None
    assert stopped["status"] == "shutdown_requested"
    assert controller.shutdown_requested is True


def test_formal_trace_writer_fault_latches_hold_and_blocks_rearm() -> None:
    class FaultedCapture(_FakeTraceCapture):
        def status(self):
            return {
                **super().status(),
                "status": "faulted",
                "message": "ControlTrace writer fault: synthetic_failure",
                "writer": {
                    "thread_alive": False,
                    "fatal_error_type": None,
                    "last_failure_code": "synthetic_failure",
                },
            }

    app = _FakeApplication()
    app.started = True
    controller = SimHILAcceptanceController(
        app,
        trace_capture=FaultedCapture(),
    )

    failure = controller.enforce_capture_health()
    revision = controller.status()["revision"]
    assert failure == "ControlTrace writer fault: synthetic_failure"
    assert app._held is True
    assert app.calls == ["hold:formal_control_trace_capture_fault"]
    assert controller.enforce_capture_health() == failure
    assert controller.status()["revision"] == revision

    result = _submit_and_process(
        controller,
        _intent(controller, sequence=1, kind="rearm"),
    )
    assert result["status"] == "failed"
    assert "synthetic_failure" in result["error"]
    assert app._held is True


def test_held_runtime_home_capture_does_not_rebuild_application() -> None:
    app = _FakeApplication()
    controller = SimHILAcceptanceController(app)
    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_stage",
            stage_id="human_teleop",
        ),
    )
    assert started["status"] == "awaiting_confirmation"
    _submit_and_process(
        controller,
        _intent(controller, sequence=2, kind="hold"),
    )

    captured = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=3,
            kind="capture_runtime_home",
        ),
    )

    assert captured["status"] == "runtime_home_captured"
    assert app.runtime_home_captures == 2
    assert app.started is True
    assert app._held is True
    assert controller.current_application is app


def test_integrated_vive_calibration_intents_use_setup_controller_only(
    tmp_path,
) -> None:
    profile = tmp_path / "coordinate.json"
    profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(input_ready=True)
    controller = SimHILAcceptanceController(
        None,
        application_factory=lambda _path, _role, _alpha: _LazyFakeApplication(),
        vive_setup_controller=setup,
        vive_setup_factory=lambda role: _FakeViveSetup(
            input_ready=True,
            source_role=role,
        ),
        coordinate_profile_path=profile,
    )

    result = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="begin_vive_calibration",
        ),
    )

    assert result["status"] == "accepted"
    assert setup.events == [("begin_calibration", {})]
    assert setup.workflow_active is True
    assert setup.closed is False
    assert controller.current_application is None
    assert controller.status()["setup"]["runtime_launch_ready"] is False


def test_active_lazy_start_can_be_cancelled_and_restores_vive_setup(
    tmp_path,
) -> None:
    profile = tmp_path / "coordinate.json"
    profile.write_text("{}", encoding="utf-8")
    setup = _FakeViveSetup(input_ready=True)
    entered_wait = threading.Event()

    class BlockingApplication(_LazyFakeApplication):
        def run_human_input_acceptance_check(
            self,
            *,
            timeout_sec: float,
            cancel_requested,
        ):
            entered_wait.set()
            deadline = time.monotonic() + timeout_sec
            while not cancel_requested():
                if time.monotonic() >= deadline:
                    raise TimeoutError("test wait timed out")
                time.sleep(0.001)
            raise InterruptedError("cancelled by test")

    app = BlockingApplication()
    restored: list[_FakeViveSetup] = []

    def setup_factory(role: str):
        replacement = _FakeViveSetup(
            input_ready=True,
            source_role=role,
        )
        restored.append(replacement)
        return replacement

    controller = SimHILAcceptanceController(
        None,
        application_factory=lambda _path, _role, _alpha: app,
        vive_setup_controller=setup,
        vive_setup_factory=setup_factory,
        coordinate_profile_path=profile,
        device_check_timeout_sec=2.0,
    )
    start_intent = _intent(
        controller,
        sequence=1,
        kind="start_stage",
        stage_id="human_teleop",
    )
    assert controller.submit(start_intent).accepted is True
    results: list[dict[str, object] | None] = []
    worker = threading.Thread(
        target=lambda: results.append(controller.process_next())
    )
    worker.start()
    assert entered_wait.wait(timeout=1.0)
    assert controller.status()["operation_cancel_available"] is True

    cancel = _intent(
        controller,
        sequence=2,
        kind="cancel_operation",
    )
    receipt = controller.submit(cancel)
    assert receipt.accepted is True
    worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert results[0]["status"] == "failed"
    assert "cancelled by test" in results[0]["error"]
    assert app.closed is True
    assert len(restored) == 1
    assert controller.current_application is None
    assert controller.status()["setup"]["phase"] == "vive_setup"
    assert (
        controller.status()["stage_status_by_id"]["human_teleop"]
        == "faulted"
    )
    with pytest.raises(ValueError, match="strictly between"):
        SimHILAcceptanceIntent(
            client_id="browser",
            client_sequence=1,
            expected_revision=0,
            intent_kind="start_stage",
            stage_id="proportional_blend",
            human_alpha=1.0,
        )


def test_acceptance_controller_runs_stages_in_order_and_commits_capture() -> None:
    app = _FakeApplication()
    capture = _FakeTraceCapture()
    controller = SimHILAcceptanceController(
        app,
        trace_capture=capture,
        device_check_timeout_sec=3.0,
    )
    sequence = 0
    for stage_id in SIM_HIL_ACCEPTANCE_STAGES_V1:
        sequence += 1
        result = _submit_and_process(
            controller,
            _intent(
                controller,
                sequence=sequence,
                kind="start_stage",
                stage_id=stage_id,
                alpha=0.35
                if stage_id == "proportional_blend"
                else None,
            ),
        )
        assert result["status"] == "awaiting_confirmation"
        app.run_acceptance_cycle()
        sequence += 1
        confirmed = _submit_and_process(
            controller,
            _intent(
                controller,
                sequence=sequence,
                kind="confirm_stage",
                stage_id=stage_id,
            ),
        )
        assert confirmed["status"] == "passed"

    assert app.alpha_updates == [1.0, 0.0, 0.35]
    assert capture.prepared is True
    sequence += 1
    finalized = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=sequence,
            kind="finalize_capture",
        ),
    )
    assert finalized["status"] == "complete"
    assert capture.finalized is True
    assert app._episode_ended is True
    assert controller.status()["capture"]["commit_status"] == "committed"


def test_device_stage_start_failure_is_visible_and_retryable() -> None:
    class RetryApplication(_FakeApplication):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def run_human_input_acceptance_check(self, *, timeout_sec: float):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError(
                    "right angle-trigger configured reader is unavailable"
                )
            return super().run_human_input_acceptance_check(
                timeout_sec=timeout_sec
            )

    app = RetryApplication()
    controller = SimHILAcceptanceController(app)
    first = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_stage",
            stage_id="human_teleop",
        ),
    )
    assert first["status"] == "failed"
    assert "configured reader is unavailable" in first["error"]
    assert (
        controller.status()["stage_status_by_id"]["human_teleop"]
        == "faulted"
    )

    second = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="start_stage",
            stage_id="human_teleop",
        ),
    )
    assert second["status"] == "awaiting_confirmation"
    assert (
        controller.status()["stage_status_by_id"]["human_teleop"]
        == "awaiting_confirmation"
    )


def test_full_start_failure_is_terminal_instead_of_retrying_closed_sources() -> None:
    class TerminalStartApplication(_FakeApplication):
        restart_required = False

        def start(self):
            self.calls.append("start")
            self.restart_required = True
            raise RuntimeError("primary full-start failure")

    app = TerminalStartApplication()
    controller = SimHILAcceptanceController(app)
    failed = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_stage",
            stage_id="human_teleop",
        ),
    )

    assert failed["status"] == "failed"
    assert "primary full-start failure" in failed["error"]
    assert (
        controller.status()["stage_status_by_id"]["human_teleop"]
        == "restart_required"
    )


def test_human_teleop_confirmation_rejects_a_wrong_mix_mode() -> None:
    class BlendedApplication(_FakeApplication):
        def status(self):
            value = super().status()
            value["composition"]["mix"] = {
                "mode": "proportional_blend",
                "human_alpha_by_component": {
                    "right_arm": 0.5,
                    "right_gripper": 0.5,
                },
            }
            return value

    app = BlendedApplication()
    controller = SimHILAcceptanceController(app)
    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_stage",
            stage_id="human_teleop",
        ),
    )
    assert started["status"] == "awaiting_confirmation"
    app.run_acceptance_cycle()
    result = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=2,
            kind="confirm_stage",
            stage_id="human_teleop",
        ),
    )

    assert result["status"] == "failed"
    assert "composition_mix_mode_mismatch" in result["error"]


def test_vive_source_swap_is_typed_and_ignores_device_readiness() -> None:
    app = _FakeApplication()
    controller = SimHILAcceptanceController(app)
    controller._stage_status_by_id["human_teleop"] = "blocked"
    swapped = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="swap_vive_sources",
        ),
    )
    assert swapped["status"] == "swapped"
    assert swapped["receipt"]["swap_count"] == 1
    assert app.vive_swaps == 1

    sequence = 1
    for kind in ("start_stage", "confirm_stage"):
        sequence += 1
        result = _submit_and_process(
            controller,
            _intent(
                controller,
                sequence=sequence,
                kind=kind,
                stage_id="human_teleop",
            ),
        )
        if kind == "start_stage":
            app.run_acceptance_cycle()
        assert result["status"] in {
            "awaiting_confirmation",
            "passed",
        }

    rejected = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=sequence + 1,
            kind="swap_vive_sources",
        ),
    )
    assert rejected["status"] == "failed"
    assert "only before human input is attached" in rejected["error"]
    assert app.vive_swaps == 1


def test_acceptance_controller_rejects_out_of_order_and_stale_intents() -> None:
    controller = SimHILAcceptanceController(
        _FakeApplication(),
        trace_capture=_FakeTraceCapture(),
    )
    out_of_order = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_stage",
            stage_id="ppo_only",
        ),
    )
    assert out_of_order["status"] == "failed"
    assert "Next acceptance stage is human_teleop" in out_of_order["error"]

    stale = SimHILAcceptanceIntent(
        client_id="test-client",
        client_sequence=2,
        expected_revision=0,
        intent_kind="hold",
    )
    receipt = controller.submit(stale)
    assert receipt.accepted is False
    assert receipt.rejection_code == "stale_revision"


def test_acceptance_confirmation_requires_a_post_stage_dispatched_cycle() -> None:
    app = _FakeApplication()
    controller = SimHILAcceptanceController(
        app,
        trace_capture=_FakeTraceCapture(),
    )
    sequence = 0
    sequence += 1
    started = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=sequence,
            kind="start_stage",
            stage_id="human_teleop",
        ),
    )
    assert started["status"] == "awaiting_confirmation"
    app.run_acceptance_cycle()
    sequence += 1
    confirmed_human = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=sequence,
            kind="confirm_stage",
            stage_id="human_teleop",
        ),
    )
    assert confirmed_human["status"] == "passed"

    sequence += 1
    _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=sequence,
            kind="start_stage",
            stage_id="ppo_only",
        ),
    )
    sequence += 1
    premature = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=sequence,
            kind="confirm_stage",
            stage_id="ppo_only",
        ),
    )
    assert premature["status"] == "failed"
    assert "no_post_stage_control_cycle" in premature["error"]
    assert (
        controller.status()["stage_status_by_id"]["ppo_only"]
        == "awaiting_confirmation"
    )

    app.run_acceptance_cycle()
    sequence += 1
    confirmed = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=sequence,
            kind="confirm_stage",
            stage_id="ppo_only",
        ),
    )
    assert confirmed["status"] == "passed"


def test_passive_device_acceptance_never_dispatches_or_claims_authority() -> None:
    emission = SimpleNamespace(
        emission_id="human-emission",
        semantic_content_hash="human-hash",
    )
    step = SimpleNamespace(
        emission=emission,
        emission_valid=True,
        command_dispatched=False,
        authoritative=False,
        rejection_reason=None,
        session_step=SimpleNamespace(
            channels={
                "right_wrist_pose": SimpleNamespace(),
                "right_angle_trigger": SimpleNamespace(),
            }
        ),
    )

    class HumanSource:
        binding_receipt = SimpleNamespace(rig_absent=True)

        def __init__(self) -> None:
            self.started = False
            self.calibration_ready = False

        def start(self) -> None:
            self.started = True

        def set_calibration_ready(self) -> None:
            self.calibration_ready = True

        def step(self, *, now: float):
            assert self.started and self.calibration_ready and now == 10.0
            return step

    human = HumanSource()
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=SimpleNamespace(started=False),
        policy_source=SimpleNamespace(),
        human_source=human,
        readiness_provider=SimpleNamespace(),
        gripper_input_selection=SimpleNamespace(
            resolved_source="angle_trigger",
            q6_port=None,
            fallback_reason="no_reachable_q6",
        ),
        clock=lambda: 10.0,
        sleep=lambda _duration: None,
    )

    report = app.run_human_input_acceptance_check(timeout_sec=1.0)

    assert report["status"] == "ready"
    assert report["command_dispatched"] is False
    assert report["authoritative"] is False
    assert report["channel_names"] == [
        "right_angle_trigger",
        "right_wrist_pose",
    ]
    assert app.started is False


def test_passive_device_calibration_failure_remains_owned_for_cleanup() -> None:
    calls: list[str] = []

    class HumanSource:
        def start(self) -> None:
            calls.append("human.start")

        def set_calibration_ready(self) -> None:
            calls.append("human.calibration")
            raise RuntimeError("calibration transition failed")

        def close(self) -> None:
            calls.append("human.close")

    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=SimpleNamespace(started=False),
        policy_source=SimpleNamespace(
            close=lambda: calls.append("policy.close")
        ),
        human_source=HumanSource(),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
        sleep=lambda _duration: None,
    )

    with pytest.raises(RuntimeError, match="calibration transition failed"):
        app.run_human_input_acceptance_check(timeout_sec=1.0)
    assert app._sources_started is True

    app.close()

    assert calls == [
        "human.start",
        "human.calibration",
        "human.close",
        "policy.close",
    ]
    assert app.closed is True
    with pytest.raises(RuntimeError, match="instance is closed"):
        app.run_human_input_acceptance_check(timeout_sec=1.0)


def test_passive_device_acceptance_does_not_block_status_lock_while_waiting() -> None:
    entered_step = threading.Event()
    release_step = threading.Event()
    result: list[dict[str, object]] = []

    class HumanSource:
        binding_receipt = SimpleNamespace(rig_absent=True)

        def start(self) -> None:
            return None

        def set_calibration_ready(self) -> None:
            return None

        def step(self, *, now: float):
            entered_step.set()
            assert release_step.wait(timeout=1.0)
            return SimpleNamespace(
                emission=SimpleNamespace(
                    emission_id="human-emission",
                    semantic_content_hash="human-hash",
                ),
                emission_valid=True,
                command_dispatched=False,
                authoritative=False,
                rejection_reason=None,
                session_step=SimpleNamespace(channels={}),
            )

    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=SimpleNamespace(started=False),
        policy_source=SimpleNamespace(),
        human_source=HumanSource(),
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
        sleep=lambda _duration: None,
    )
    worker = threading.Thread(
        target=lambda: result.append(
            app.run_human_input_acceptance_check(timeout_sec=1.0)
        )
    )
    worker.start()
    assert entered_step.wait(timeout=1.0)

    assert app._lock.acquire(timeout=0.2)
    app._lock.release()
    release_step.set()
    worker.join(timeout=1.0)

    assert worker.is_alive() is False
    assert result[0]["status"] == "ready"
    assert app._human_input_check_active is False


def test_passive_device_timeout_reports_missing_and_invalid_channels() -> None:
    session_step = SimpleNamespace(
        channels={
            "right_wrist_pose": SimpleNamespace(valid=False),
            "right_angle_trigger": SimpleNamespace(valid=True),
        },
        faults=[SimpleNamespace(code="invalid_channel")],
    )

    class HumanSource:
        def __init__(self) -> None:
            self.session = SimpleNamespace(
                sources=[
                    SimpleNamespace(
                        expected_channel_names=lambda: (
                            "right_wrist_pose",
                            "right_angle_trigger",
                        )
                    )
                ]
            )

        def start(self) -> None:
            return None

        def set_calibration_ready(self) -> None:
            return None

        def step(self, *, now: float):
            return SimpleNamespace(
                emission=None,
                emission_valid=False,
                command_dispatched=False,
                authoritative=False,
                rejection_reason=None,
                session_step=session_step,
            )

    now = 10.0

    def clock() -> float:
        nonlocal now
        now += 0.02
        return now

    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=SimpleNamespace(),
        policy_source=SimpleNamespace(),
        human_source=HumanSource(),
        readiness_provider=SimpleNamespace(),
        clock=clock,
        sleep=lambda _duration: None,
    )

    with pytest.raises(TimeoutError) as captured:
        app.run_human_input_acceptance_check(timeout_sec=0.01)

    message = str(captured.value)
    assert "rejection=none" in message
    assert "invalid_channels=['right_wrist_pose']" in message
    assert "available_channels=['right_angle_trigger', 'right_wrist_pose']" in message
    assert "fault_codes=['invalid_channel']" in message


def test_application_vive_swap_is_mapping_only_and_prestart() -> None:
    vive = VivePoseSource(
        [
            ViveTrackerConfig(
                name="tracker_1",
                side="right",
                role="left_wrist",
                channel_name="right_wrist_pose",
                legacy_index=0,
            )
        ],
        host="127.0.0.1",
        port=0,
        clock_sync_port=0,
    )
    human = SimpleNamespace(session=SimpleNamespace(sources=[vive]))
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=SimpleNamespace(),
        policy_source=SimpleNamespace(),
        human_source=human,
        readiness_provider=SimpleNamespace(),
        clock=lambda: 10.0,
    )

    receipt = app.swap_vive_left_right_sources()
    assert receipt["swap_count"] == 1
    assert vive.source_role_by_output_channel == {
        "right_wrist_pose": "right_wrist"
    }
    assert app._vive_source_swap_count == 1

    app._started = True
    with pytest.raises(RuntimeError, match="after Sim-HIL composition"):
        app.swap_vive_left_right_sources()


def test_acceptance_trace_capture_prepares_and_aborts_without_false_commit(
    tmp_path,
) -> None:
    capture = SimHILAcceptanceTraceCapture(
        tmp_path,
        trace_session_id="acceptance-abort-test",
        max_spool_bytes=1 << 20,
    )
    try:
        status = capture.prepare(now=10.0)
        assert status["status"] == "recording"
        assert status["commit_status"] is None
        assert status["episode_uuid"]

        capture.abort(now=11.0)
        status = capture.status()
        assert status["status"] == "ready"
        assert status["commit_status"] is None
    finally:
        capture.close(now=12.0)


def test_acceptance_trace_drains_durably_before_episode_teardown(
    tmp_path,
) -> None:
    capture = SimHILAcceptanceTraceCapture(
        tmp_path,
        trace_session_id="acceptance-background-drain-test",
        max_spool_bytes=1 << 20,
    )
    fact = ControlTraceFact.create(
        fact_id="background-drain:1",
        event_type="source_validation",
        runtime_session_id="control-station-sim-hil-ppo-session",
        control_domain_id="mujoco-sim-hil",
        embodiment_instance_id="rm65-pair-0",
        occurred_at=10.0,
        clock_domain_id="mujoco.sim_hil.monotonic",
        payload={
            "validation_id": "background-drain-validation",
            "source_event_id": "background-drain-source",
            "accepted": True,
            "reasons": (),
        },
    )
    try:
        capture.admit_trace(fact)
        deadline = time.monotonic() + 2.0
        while (
            capture.trace_runtime.snapshot().sink_durable_high_watermark < 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)

        assert (
            capture.trace_runtime.snapshot().sink_durable_high_watermark
            == 1
        )
        writer = capture.status()["writer"]
        assert writer["thread_alive"] is True
        assert writer["fatal_error_type"] is None
        assert writer["last_failure_code"] is None
    finally:
        capture.close(now=12.0)


def test_acceptance_trace_grows_without_the_old_fixed_wal_cutoff(
    tmp_path,
) -> None:
    capture = SimHILAcceptanceTraceCapture(
        tmp_path,
        trace_session_id="acceptance-growing-wal-test",
        max_spool_bytes=4096,
        runtime_capacity=128,
    )
    try:
        for sequence in range(1, 33):
            capture.admit_trace(
                ControlTraceFact.create(
                    fact_id=f"growing-validation:{sequence}",
                    event_type="source_validation",
                    runtime_session_id=(
                        "control-station-sim-hil-ppo-session"
                    ),
                    control_domain_id="mujoco-sim-hil",
                    embodiment_instance_id="rm65-pair-0",
                    occurred_at=10.0 + sequence * 0.001,
                    clock_domain_id="mujoco.sim_hil.monotonic",
                    payload={
                        "validation_id": f"validation:{sequence}",
                        "source_event_id": f"source:{sequence}",
                        "accepted": True,
                        "reasons": (),
                    },
                )
            )
        deadline = time.monotonic() + 2.0
        while (
            capture.trace_runtime.snapshot().sink_durable_high_watermark
            < 32
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)

        status = capture.status()
        assert status["status"] == "ready"
        assert status["writer"]["last_failure_code"] is None
        assert status["writer"]["allocated_bytes"] > 4096
        assert status["writer"]["write_offset_bytes"] > 4096
    finally:
        capture.close(now=12.0)


def test_acceptance_trace_admission_failure_is_a_latched_capture_fault(
    tmp_path,
) -> None:
    capture = SimHILAcceptanceTraceCapture(
        tmp_path,
        trace_session_id="acceptance-admission-fault-test",
        max_spool_bytes=4096,
        runtime_capacity=1,
    )

    def fact(sequence: int) -> ControlTraceFact:
        return ControlTraceFact.create(
            fact_id=f"bounded-validation:{sequence}",
            event_type="source_validation",
            runtime_session_id="control-station-sim-hil-ppo-session",
            control_domain_id="mujoco-sim-hil",
            embodiment_instance_id="rm65-pair-0",
            occurred_at=10.0 + sequence * 0.001,
            clock_domain_id="mujoco.sim_hil.monotonic",
            payload={
                "validation_id": f"validation:{sequence}",
                "source_event_id": f"source:{sequence}",
                "accepted": True,
                "reasons": (),
            },
        )

    try:
        capture.admit_trace(fact(1))
        with pytest.raises(RuntimeError, match="journal_capacity_exhausted"):
            capture.admit_trace(fact(2))
        status = capture.status()
        assert status["status"] == "faulted"
        assert "journal_capacity_exhausted" in status["message"]
    finally:
        capture.close(now=12.0)


def test_acceptance_trace_rotation_allows_rebuilt_runtime_fact_namespace(
    tmp_path,
) -> None:
    capture = SimHILAcceptanceTraceCapture(
        tmp_path,
        trace_session_id="acceptance-runtime-rotation-test",
        max_spool_bytes=1 << 20,
    )
    fact = ControlTraceFact.create(
        fact_id="runtime-bootstrap:stable-id",
        event_type="source_validation",
        runtime_session_id="control-station-sim-hil-ppo-session",
        control_domain_id="mujoco-sim-hil",
        embodiment_instance_id="rm65-pair-0",
        occurred_at=10.0,
        clock_domain_id="mujoco.sim_hil.monotonic",
        payload={
            "validation_id": "runtime-bootstrap-validation",
            "source_event_id": "runtime-bootstrap-source",
            "accepted": True,
            "reasons": (),
        },
    )
    try:
        original_session_id = capture.status()["trace_session_id"]
        capture.admit_trace(fact)
        with pytest.raises(RuntimeError, match="duplicate_fact_id"):
            capture.admit_trace(fact)

        rotated = capture.rotate_runtime_session(now=11.0)
        assert rotated["trace_session_id"] != original_session_id
        capture.admit_trace(fact)
    finally:
        capture.close(now=12.0)


def test_acceptance_trace_prepare_rolls_back_a_bind_failure(tmp_path) -> None:
    capture = SimHILAcceptanceTraceCapture(
        tmp_path,
        trace_session_id="acceptance-bind-rollback-test",
        max_spool_bytes=1 << 20,
    )
    original_bind = capture.recorder.bind_prepared_episode

    def fail_bind(*_args, **_kwargs):
        raise RuntimeError("synthetic bind failure")

    capture.recorder.bind_prepared_episode = fail_bind
    try:
        with pytest.raises(RuntimeError, match="synthetic bind failure"):
            capture.prepare(now=10.0)
        assert capture.status()["status"] == "ready"
        assert capture.recorder.prepared_episode is None
        assert capture.recorder.lifecycle.active_episode_uuid is None

        capture.recorder.bind_prepared_episode = original_bind
        assert capture.prepare(now=11.0)["status"] == "recording"
    finally:
        capture.close(now=12.0)


def test_start_failure_does_not_prepare_acceptance_episode() -> None:
    class FailingStartApplication(_FakeApplication):
        def start(self, *, cancel_requested=None):
            self.calls.append("start")
            raise RuntimeError("synthetic native-start boundary failure")

    app = FailingStartApplication()
    capture = _FakeTraceCapture()
    controller = SimHILAcceptanceController(app, trace_capture=capture)
    result = _submit_and_process(
        controller,
        _intent(
            controller,
            sequence=1,
            kind="start_stage",
            stage_id="human_teleop",
        ),
    )
    assert result["status"] == "failed"
    assert "synthetic native-start boundary failure" in result["error"]
    assert capture.prepared is False
