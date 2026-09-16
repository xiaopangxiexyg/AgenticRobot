from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import control_station.sim_hil as sim_hil_module
from control_composition.composition import hand_action_spec_semantic_hash
from control_composition import GuaranteedNextComponentAction
from control_composition.activation import build_composition_activation_scope
from control_composition.retention import TimeWindowRetentionSpec
from dataset_contracts.completeness import (
    TRACE_REQUIREMENT_PROFILES_V1,
    evaluate_control_trace_lineage,
)
from dataset_interface.control_trace import load_episode_control_trace
from control_station.sim_hil import (
    DEFAULT_CALIBRATION_PROFILE,
    DEFAULT_VIVE_COORDINATE_PROFILE,
    _bind_sim_target_session_anchors,
    _bind_vive_coordinate_profile,
    _file_sha256,
    MujocoSimHILApplication,
    MujocoSimHILApplicationConfig,
)
from control_station.sim_hil_acceptance import (
    SimHILAcceptanceTraceCapture,
)
from control_station.sim_hil_profile.human import _SimHILHumanAttachment
from control_station.composition import (
    MUJOCO_SIM_HIL_HUMAN_ROLE_V1,
    MUJOCO_SIM_HIL_POLICY_ROLE_V1,
    MujocoSimHILComposition,
    MujocoSimHILCompositionStep,
    MujocoSimHILCycleResult,
    MujocoSimHILPreflightReport,
    MujocoSimHILProfileConfig,
    MujocoSimHILReadinessEvidence,
    build_component_keyed_mujoco_sim_hil_contracts,
    build_default_mujoco_sim_hil_contracts,
    build_mujoco_sim_hil_composition_runtime_config,
    build_mujoco_sim_hil_strategy_control_factory,
    build_scoped_mujoco_sim_hil_composition,
)
from control_station.composition.mujoco_policy_human_residual import (
    MujocoPolicyHumanResidualProfileConfig,
    build_mujoco_policy_human_residual_profile,
)
from control_station.sources import (
    PolicyCompositionInputAdapter,
    PolicyCompositionInputBinding,
)
from control_station.sources.teleop_pose import (
    PassiveTeleopComponentBindingReceipt,
    PassiveTeleopComponentStep,
    PassiveTeleopComponentSourceSpec,
    PassiveTeleopComponentSpec,
)
from robot_motion.action import (
    ActionResolutionFailure,
    AuthoritativeControlStep,
    ArmActionSpec,
    BIMANUAL_EE_POSE_V1,
    ExecutionMeasurementObservation,
    ExecutionObservationPlan,
    ExecutionObservationWatermark,
    HandActionSpec,
    RobotAction,
    SourceValidationResult,
    project_robot_action_v1_to_component_action,
)
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
)
from robot_motion.runtime import EffectorCommand, TeleopCommand
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry
from robot_motion.deploy import NormalizedAction, PolicyActionEmission
from robot_motion.diagnostics import TeleopRunState
from sim.mujoco.deploy import (
    MujocoSimHILTargetConfig,
    build_mujoco_pose_action_schema,
    build_mujoco_sim_hil_target,
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
    prepare_mujoco_sim_hil_target,
)
from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
from sim.mujoco_temp_pipeline.checkpoint import (
    check_task_config_compatibility,
    load_checkpoint_payload,
)
from teleop_runtime.calibration import TeleopCalibration
from teleop_runtime.action.human_proposal import HumanCommandEmission
from teleop_runtime.calibration import ArmCalibration
from teleop_runtime.configs import (
    PassiveSingleViveQ6SessionConfig,
    build_passive_single_vive_q6_session,
)
from teleop_runtime.schemas import ObservationChannel, TeleopObservation


def _require_mujoco_runtime() -> None:
    pytest.importorskip("mujoco")


def _require_optional_default_checkpoint_fixture(
    config: object | None = None,
) -> None:
    selected = MujocoSimHILApplicationConfig() if config is None else config
    try:
        payload = load_checkpoint_payload(
            selected.checkpoint_path,
            map_location="cpu",
        )
        check_task_config_compatibility(payload, selected.task_config_path)
        CubeLiftTaskConfig.load(selected.task_config_path).base_contract()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        pytest.skip(
            "optional Sim-HIL checkpoint fixture unavailable: "
            f"{type(exc).__name__}: {exc}"
        )


def test_optional_default_checkpoint_fixture_skips_when_incomplete(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        checkpoint_path=tmp_path / "missing-checkpoint.pt",
        task_config_path=tmp_path / "missing-task.json",
    )

    with pytest.raises(pytest.skip.Exception, match="fixture unavailable"):
        _require_optional_default_checkpoint_fixture(config)


def test_preflight_uses_readiness_required_by_active_mix() -> None:
    """Human-only startup must not require an unused policy emission."""

    readiness = MujocoSimHILReadinessEvidence(
        calibration_ready=True,
        home_ready=True,
        rebase_ready=True,
        human_device_ready=True,
        policy_source_ready=False,
        policy_chunk_ttl_ready=False,
        deadman_ready=True,
        measured_state_ready=True,
    )
    report = MujocoSimHILPreflightReport(
        readiness=readiness,
        checks={
            "profile_enabled": True,
            "required_readiness": True,
        },
    )

    assert report.passed is True
    assert report.reasons == ()


@pytest.mark.parametrize(
    ("fallback_source", "expected_alpha"),
    (("human", 1.0), ("policy", 0.0)),
)
def test_blend_resolution_fallback_latches_through_formal_alpha_control(
    fallback_source: str,
    expected_alpha: float,
) -> None:
    class Composition:
        def __init__(self) -> None:
            self.human_alpha_by_component = {
                "right_arm": 0.5,
                "right_gripper": 0.5,
            }
            self.calls: list[tuple[float, float]] = []

        def set_human_blend_alpha(self, alpha, *, effective_from):
            self.calls.append((float(alpha), float(effective_from)))
            self.human_alpha_by_component = {
                component_id: float(alpha)
                for component_id in self.human_alpha_by_component
            }
            return SimpleNamespace(
                accepted=True,
                rejection_code=None,
                message="",
                parameter_snapshot_id=f"alpha:{alpha}",
            )

    composition = Composition()
    app = object.__new__(MujocoSimHILApplication)
    app.composition = composition
    app.runtime = SimpleNamespace(clock=lambda: 10.1)
    app._blend_failure_fallback = fallback_source
    app._blend_failure_fallback_latched_source = None
    app._last_blend_failure_fallback = None
    cycle = SimpleNamespace(
        cycle_index=7,
        execution=AuthoritativeControlStep(
            status="compound_resolution_failure_hold",
            resolution_failure=ActionResolutionFailure(
                reason_code="collision_margin_low",
                resolver_status="hold_last_safe",
                message="blend endpoint is infeasible",
                diagnostics={"best_candidate_distance_m": 0.029},
            ),
        ),
    )

    app._apply_blend_failure_fallback(cycle, observed_at=10.0)
    app._apply_blend_failure_fallback(cycle, observed_at=10.2)

    assert composition.calls == [(expected_alpha, 10.1)]
    assert app._blend_failure_fallback_latched_source == fallback_source
    assert app._last_blend_failure_fallback["status"] == "latched"
    assert app._last_blend_failure_fallback["failure"]["reason_code"] == (
        "collision_margin_low"
    )


def test_sim_collision_rejection_is_promoted_to_terminal_episode() -> None:
    failure = ActionResolutionFailure(
        reason_code="collision_margin_low",
        resolver_status="hold_last_safe",
        message="candidate is inside the configured collision margin",
    )
    cycle = SimpleNamespace(
        execution=AuthoritativeControlStep(
            status="compound_resolution_failure_hold",
            resolution_failure=failure,
        )
    )
    app = object.__new__(MujocoSimHILApplication)
    app._last_error = None
    app._first_control_fault = None
    terminal_calls: list[dict[str, object]] = []
    app.end_episode = lambda **kwargs: terminal_calls.append(dict(kwargs))

    selected = app._collision_stop_failure(cycle)
    assert selected is failure
    app._terminate_episode_for_collision_stop(selected)

    assert terminal_calls == [
        {
            "reason": "sim_collision_stop:collision_margin_low",
            "force_cancellation": True,
        }
    ]
    assert app._last_error == "SimCollisionStop:collision_margin_low"
    assert app._first_control_fault == app._last_error

    non_collision = SimpleNamespace(
        execution=AuthoritativeControlStep(
            status="compound_resolution_failure_hold",
            resolution_failure=ActionResolutionFailure(
                reason_code="target_unresolvable",
                resolver_status="hold_last_safe",
                message="IK did not converge",
            ),
        )
    )
    assert app._collision_stop_failure(non_collision) is None


def test_actual_mujoco_forbidden_contact_is_promoted_separately() -> None:
    app = object.__new__(MujocoSimHILApplication)
    app._last_error = None
    app._first_control_fault = None
    terminal_calls: list[dict[str, object]] = []
    app.end_episode = lambda **kwargs: terminal_calls.append(dict(kwargs))
    recorded_facts = []
    def record_guard(observation):
        from sim.mujoco.deploy.contact_trace import (
            project_mujoco_forbidden_contact_guard_result,
        )

        recorded_facts.append(
            project_mujoco_forbidden_contact_guard_result(
                observation,
                runtime_session_id="runtime-session",
                control_domain_id="sim-domain",
                embodiment_instance_id="sim-embodiment",
                clock_domain_id="sim-clock",
            )
        )

    app.runtime = SimpleNamespace(
        record_active_contact_guard_result=record_guard,
    )
    observation = {
        "available": True,
        "safe": False,
        "observed_at": 12.0,
        "physics_tick": 123,
        "hard_penetration_limit_m": 0.001,
        "geom1": "right_gripper_collision",
        "geom2": "table_collision",
        "penetration_depth_m": 0.0015,
    }

    app._terminate_episode_for_active_forbidden_contact(observation)

    assert terminal_calls == [
        {
            "reason": "sim_collision_stop:active_forbidden_contact",
            "force_cancellation": True,
        }
    ]
    assert app._last_error == (
        "SimCollisionStop:active_forbidden_contact:"
        "right_gripper_collision ↔ table_collision"
    )
    assert len(recorded_facts) == 1
    assert recorded_facts[0].event_type == "backend_guard_result"
    assert recorded_facts[0].payload["reason_code"] == (
        "active_forbidden_contact_hard_penetration"
    )


def test_blend_resolution_fallback_hold_does_not_publish_source_switch() -> None:
    composition = SimpleNamespace(
        human_alpha_by_component={"right_arm": 0.5},
        set_human_blend_alpha=lambda *_args, **_kwargs: pytest.fail(
            "hold fallback must not publish an alpha update"
        ),
    )
    app = object.__new__(MujocoSimHILApplication)
    app.composition = composition
    app.runtime = SimpleNamespace(clock=lambda: 10.0)
    app._blend_failure_fallback = "hold"
    app._blend_failure_fallback_latched_source = None
    app._last_blend_failure_fallback = None
    cycle = SimpleNamespace(
        cycle_index=2,
        execution=AuthoritativeControlStep(
            status="compound_resolution_failure_hold",
            resolution_failure=ActionResolutionFailure(
                reason_code="collision_margin_low",
                resolver_status="hold_last_safe",
                message="blend endpoint is infeasible",
            ),
        ),
    )

    app._apply_blend_failure_fallback(cycle, observed_at=10.0)

    assert app._blend_failure_fallback_latched_source is None
    assert app._last_blend_failure_fallback["status"] == "hold_selected"


def test_blend_resolution_fallback_waits_for_continuous_failure_window() -> None:
    class Composition:
        human_alpha_by_component = {"right_arm": 0.5}

        def __init__(self) -> None:
            self.calls: list[tuple[float, float]] = []

        def set_human_blend_alpha(self, alpha, *, effective_from):
            self.calls.append((float(alpha), float(effective_from)))
            return SimpleNamespace(
                accepted=True,
                rejection_code=None,
                message="",
                parameter_snapshot_id="alpha:0",
            )

    app = object.__new__(MujocoSimHILApplication)
    app.composition = Composition()
    app.runtime = SimpleNamespace(clock=lambda: 10.5)
    app._blend_failure_fallback = "policy"
    app._blend_failure_grace_sec = 0.2
    app._blend_failure_fallback_latched_source = None
    app._last_blend_failure_fallback = None
    app._blend_failure_window_started_at = None
    app._blend_failure_consecutive_count = 0
    failure_cycle = SimpleNamespace(
        cycle_index=3,
        execution=AuthoritativeControlStep(
            status="compound_resolution_failure_hold",
            resolution_failure=ActionResolutionFailure(
                reason_code="target_unresolvable",
                resolver_status="hold",
                message="blend endpoint is infeasible",
            ),
        ),
    )

    app._apply_blend_failure_fallback(failure_cycle, observed_at=10.0)
    assert app.composition.calls == []
    assert app._last_blend_failure_fallback["status"] == "grace_active"
    assert app._last_blend_failure_fallback["consecutive_failure_count"] == 1

    # A scheduling-only not_due cycle is not evidence that resolution recovered.
    app._apply_blend_failure_fallback(
        SimpleNamespace(
            cycle_index=4,
            execution=AuthoritativeControlStep(status="not_due"),
        ),
        observed_at=10.1,
    )
    app._apply_blend_failure_fallback(failure_cycle, observed_at=10.19)
    assert app.composition.calls == []

    # Any actual non-failure outcome closes the continuous failure window.
    app._apply_blend_failure_fallback(
        SimpleNamespace(
            cycle_index=5,
            execution=AuthoritativeControlStep(status="dispatched"),
        ),
        observed_at=10.2,
    )
    assert app._blend_failure_window_started_at is None

    app._apply_blend_failure_fallback(failure_cycle, observed_at=10.21)
    app._apply_blend_failure_fallback(failure_cycle, observed_at=10.42)
    assert app.composition.calls == [(0.0, 10.5)]
    assert app._blend_failure_fallback_latched_source == "policy"
    assert app._last_blend_failure_fallback["status"] == "latched"
    assert app._last_blend_failure_fallback["failure_elapsed_sec"] == pytest.approx(
        0.21
    )


def test_human_timeout_policy_fallback_switches_without_reusing_human_pose() -> None:
    class Composition:
        human_alpha_by_component = {"right_arm": 1.0}

        def __init__(self) -> None:
            self.calls = []

        def set_human_blend_alpha(self, alpha, *, effective_from):
            self.calls.append((float(alpha), float(effective_from)))
            self.human_alpha_by_component = {"right_arm": float(alpha)}
            return SimpleNamespace(
                accepted=True,
                rejection_code=None,
                message="",
                parameter_snapshot_id="timeout-fallback-receipt",
            )

    app = object.__new__(MujocoSimHILApplication)
    app.composition = Composition()
    app.readiness_provider = SimpleNamespace(
        snapshot=lambda *, now: SimpleNamespace(
            human_device_ready=False,
            detail_by_check={"human_device_ready": "emission_stale:5.1s>5.0s"},
        )
    )
    app.policy_source = SimpleNamespace(
        step=lambda *, now: SimpleNamespace(emission_id="fresh-policy")
    )
    app._human_timeout_fallback = "policy"
    app._human_timeout_fallback_latched = False
    app._last_human_timeout_fallback = None

    emission = app._apply_human_timeout_fallback(
        observed_at=12.0,
        policy_emission=None,
    )

    assert emission.emission_id == "fresh-policy"
    assert app.composition.calls == [(0.0, 12.0)]
    assert app._human_timeout_fallback_latched is True
    assert app._last_human_timeout_fallback["status"] == "policy_latched"


def test_sim_hil_human_timeout_defaults_to_five_seconds_and_hold() -> None:
    config = MujocoSimHILApplicationConfig()
    assert config.human_emission_timeout_sec == pytest.approx(5.0)
    assert config.human_timeout_fallback == "hold"
    assert config.blend_failure_grace_sec == pytest.approx(0.0)
    assert config.runtime_session_id == (
        "control-station-sim-hil-ppo-session"
    )
    assert config.control_domain_id == "mujoco-sim-hil"


def test_sim_hil_blend_failure_grace_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        MujocoSimHILApplicationConfig(blend_failure_grace_sec=-0.01)


def test_sim_hil_application_config_accepts_explicit_operation_session() -> None:
    config = MujocoSimHILApplicationConfig(
        runtime_session_id="agent-skill:cube-lift:operation-7",
        control_domain_id="agent-skill:mujoco-sim-hil",
    )
    assert config.runtime_session_id == "agent-skill:cube-lift:operation-7"
    assert config.control_domain_id == "agent-skill:mujoco-sim-hil"
    with pytest.raises(ValueError, match="runtime_session_id"):
        MujocoSimHILApplicationConfig(runtime_session_id=" ")


def test_application_viewer_preview_starts_only_passive_visualization() -> None:
    class Viewer:
        def __init__(self) -> None:
            self.start_count = 0
            self.sync_count = 0

        def start(self):
            self.start_count += 1
            return {"state": "running", "running": True}

        def sync(self):
            self.sync_count += 1
            return True

        def status(self):
            return {"state": "running", "running": True}

    viewer = Viewer()
    app = object.__new__(MujocoSimHILApplication)
    app._lock = threading.RLock()
    app._started = False
    app._closed = False
    app._restart_required = False
    app._viewer_preview_started = False
    app._viewer_exit_requested = False
    app.viewer = viewer

    receipt = app.start_viewer_preview()

    assert receipt["status"] == "viewer_preview"
    assert receipt["sources_started"] is False
    assert receipt["composition_started"] is False
    assert receipt["command_dispatched"] is False
    assert app.viewer_preview_started is True
    assert app.sync_viewer_preview() is True
    assert viewer.start_count == 1
    assert viewer.sync_count == 1


def test_human_attachment_is_allowed_during_nonexecuting_viewer_preview() -> None:
    class HumanSource:
        def __init__(self) -> None:
            self.attached = False

        def attach(self, _source):
            self.attached = True
            return {"status": "attached"}

    app = object.__new__(MujocoSimHILApplication)
    app._lock = threading.RLock()
    app._started = False
    app._held = False
    app._viewer_preview_started = True
    app._closed = False
    app.human_source = HumanSource()
    app.readiness_provider = SimpleNamespace(
        calibration_ready=False,
        home_ready=False,
    )
    app.gripper_input_selection = None
    app.calibration_binding = {}
    app._runtime_home_pose_desk_by_side_capture = None
    attachment = _SimHILHumanAttachment(
        source=object(),
        gripper_input_selection=SimpleNamespace(resolved_source="angle_trigger"),
        calibration_binding={"profile": "test"},
        runtime_home_pose_desk_by_side_capture=lambda: {},
    )

    receipt = app.activate_human_attachment(attachment)

    assert receipt["status"] == "attached"
    assert app.human_source.attached is True
    assert app.readiness_provider.calibration_ready is True
    assert app.readiness_provider.home_ready is False


def _hand_specs() -> dict[str, HandActionSpec]:
    return {side: HandActionSpec.dmtac_gripper(side) for side in ("left", "right")}


def test_sim_hil_target_requires_application_composition_runtime_config() -> None:
    hands = _hand_specs()
    contracts = build_default_mujoco_sim_hil_contracts(
        hands,
        composer_producer_id="fixture-composer",
        proposal_ttl_sec=0.1,
    )

    with pytest.raises(
        ValueError,
        match="application-supplied composition_runtime_config",
    ):
        build_mujoco_sim_hil_target(
            config=MujocoSimHILTargetConfig(
                task_config_path=Path("not-read-before-composition-binding.json"),
                realtime_worker=False,
            ),
            contracts=contracts,
            hand_specs_by_side=hands,
        )


def test_dynamic_sim_hil_target_requires_application_strategy_control_factory() -> None:
    hands = _hand_specs()
    schema = build_mujoco_pose_action_schema(hands)
    contracts = build_component_keyed_mujoco_sim_hil_contracts(
        schema,
        composer_producer_id="fixture-composer",
        proposal_ttl_sec=0.1,
    )

    with pytest.raises(
        ValueError,
        match="application-supplied composition_strategy_control_factory",
    ):
        build_mujoco_sim_hil_target(
            config=MujocoSimHILTargetConfig(
                task_config_path=Path("not-read-before-composition-binding.json"),
                realtime_worker=False,
            ),
            contracts=contracts,
            composition_runtime_config=(
                build_mujoco_sim_hil_composition_runtime_config(contracts)
            ),
            hand_specs_by_side=hands,
        )


def test_application_strategy_control_factory_preserves_contract_identity() -> None:
    hands = _hand_specs()
    schema = build_mujoco_pose_action_schema(hands)
    contracts = build_component_keyed_mujoco_sim_hil_contracts(
        schema,
        composer_producer_id="fixture-composer",
        proposal_ttl_sec=0.1,
    )
    runtime_config = build_mujoco_sim_hil_composition_runtime_config(contracts)
    assert runtime_config.enable_source_eligibility is True
    assert runtime_config.information_plan is None
    factory = build_mujoco_sim_hil_strategy_control_factory(contracts)
    assert factory is not None
    coordinator = ControlAdmissionCoordinator(
        "fixture-control-domain",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="fixture-clock",
            runtime_session_id="fixture-session",
            boot_id="fixture-boot",
            reset_epoch=0,
        ),
        clock=lambda: 10.0,
    )
    producer_epochs = ProducerEpochRegistry(
        runtime_session_id="fixture-session",
        coordinator=coordinator,
    )

    control = factory(coordinator, producer_epochs, 10.0)
    revision = control.latest_revision

    assert revision.runtime_session_id == "fixture-session"
    assert revision.control_domain_id == "fixture-control-domain"
    assert revision.available_clock_domain_id == "fixture-clock"
    assert revision.composition_spec is contracts.composition_spec
    assert revision.strategy_spec is contracts.strategy_spec
    assert revision.component_rule_set is contracts.component_rule_set
    assert revision.dynamic_parameter_contract is (
        contracts.dynamic_parameter_contract
    )
    assert dict(revision.initial_alpha_by_component) == dict(
        contracts.initial_alpha_by_component
    )
    assert control.retention_spec == TimeWindowRetentionSpec(
        spec_id="mujoco-sim-hil-strategy-control-window-v1",
        history_window_sec=2.0,
        max_future_lead_sec=1.0,
        max_effective_duration_sec=1.0,
        max_late_arrival_sec=0.0,
    )


def test_sim_hil_readiness_accepts_canonical_debug_only_enum() -> None:
    human_source = SimpleNamespace(
        session=SimpleNamespace(
            state_machine=SimpleNamespace(state=TeleopRunState.DEBUG_ONLY),
            home_recalibration_required=False,
            motion_hold_mode="",
        ),
        last_emission=SimpleNamespace(emitted_at=10.0),
    )
    policy_source = SimpleNamespace(
        ready=True,
        last_error=None,
        last_emission=SimpleNamespace(
            emitted_at=10.0,
            effective_until=11.0,
        ),
    )
    scene = SimpleNamespace(state_is_finite=lambda: True)
    captures = []

    def capture_scene(capture):
        captures.append("captured")
        return capture(scene)

    provider = sim_hil_module._SimHILReadinessProvider(
        policy_source=policy_source,
        human_source=human_source,
        scene=scene,
        calibration_ready=True,
        home_ready=True,
        max_human_age_sec=1.0,
        max_policy_age_sec=1.0,
        clock=lambda: 10.0,
        scene_capture=capture_scene,
    )

    readiness = provider.snapshot(now=10.0)

    assert readiness.ready is True
    assert readiness.deadman_ready is True
    assert readiness.detail_by_check["deadman_ready"] == "passive_debug_only"
    assert captures == ["captured"]


def test_sim_hil_readiness_treats_transient_tracker_grace_as_non_motion_hold() -> None:
    human_source = SimpleNamespace(
        session=SimpleNamespace(
            state_machine=SimpleNamespace(state=TeleopRunState.DEBUG_ONLY),
            home_recalibration_required=False,
            motion_hold_mode="grace",
            motion_hold_reason="tracker target unavailable",
        ),
        last_emission=SimpleNamespace(emitted_at=10.0),
    )
    provider = sim_hil_module._SimHILReadinessProvider(
        policy_source=SimpleNamespace(
            ready=True,
            last_error=None,
            last_emission=SimpleNamespace(
                emitted_at=10.0,
                effective_until=11.0,
            ),
        ),
        human_source=human_source,
        scene=SimpleNamespace(state_is_finite=lambda: True),
        calibration_ready=True,
        home_ready=True,
        max_human_age_sec=5.0,
        max_policy_age_sec=1.0,
        clock=lambda: 10.0,
    )

    readiness = provider.snapshot(now=10.0)

    assert readiness.deadman_ready is True
    assert readiness.detail_by_check["deadman_ready"] == ("passive_debug_only_grace")


def test_single_vive_q6_builder_is_passive_and_topology_selected() -> None:
    identity = np.eye(4)
    calibration = TeleopCalibration(
        arms={
            "right_arm": ArmCalibration(
                name="right_arm",
                side="right",
                T_D_B=identity,
            )
        },
        metadata={
            "home_ready": True,
            "home_capture": {
                "right_human_home_virtual": identity.tolist(),
                "right_robot_home_desk": identity.tolist(),
            },
        },
    )

    session = build_passive_single_vive_q6_session(
        calibration=calibration,
        source_config=PassiveSingleViveQ6SessionConfig(
            side="right",
            q6_port="/dev/null",
        ),
    )

    assert session.rig is None
    assert session.config.record_enabled is False
    assert session.config.allow_hardware_after_debug_pass is False
    assert session.config.tracker_channel_names == ("right_wrist_pose",)
    assert [source.name for source in session.sources] == [
        "vive_pose",
        "hand_q6_right",
    ]
    tracker = session.sources[0].tracker_configs["tracker_1"]
    assert (
        tracker.side,
        tracker.role,
        tracker.channel_name,
        tracker.legacy_index,
    ) == ("right", "left_wrist", "right_wrist_pose", 0)
    assert [retargeter.config.effector_name for retargeter in session.retargeters] == [
        "right_arm",
        "right_hand",
    ]
    assert session.retargeters[0].config.tracker_role == "right_wrist"
    q6_command = session.retargeters[1].retarget(
        TeleopObservation(
            channels={
                "right_hand_q6": ObservationChannel(
                    channel_name="right_hand_q6",
                    capability="joint_position",
                    side="right",
                    value=np.asarray(
                        [0.0, 0.0, 0.6, 0.0, 0.0, 0.0],
                        dtype=np.float32,
                    ),
                    frame_name=None,
                    timestamp=1.0,
                    source_name="virtual_q6",
                )
            },
            timestamp=1.0,
        ),
        session.calibration,
    )
    np.testing.assert_allclose(
        q6_command.commands["right_hand"].value,
        np.asarray([0.625, 0, 0, 0, 0, 0], dtype=np.float32),
    )


def test_single_vive_gripper_source_contract_uses_target_component_ids() -> None:
    spec = sim_hil_module._build_single_vive_gripper_component_source_spec(
        arm_reference_frame="right_arm_base",
        target_action_spec_hash_by_component={
            "right_arm": "arm-target-hash",
            "right_gripper": "gripper-target-hash",
        },
    )

    assert tuple(
        (item.component_id, item.effector_name) for item in spec.components
    ) == (
        ("right_arm", "right_arm"),
        ("right_gripper", "right_hand"),
    )


@pytest.mark.optional
@pytest.mark.slow_sim
def test_real_checkpoint_virtual_human_runs_hold_rearm_and_differences(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Exercise the concrete application without opening VIVE/Q6 devices."""

    _require_mujoco_runtime()
    _require_optional_default_checkpoint_fixture()

    class DebugOnlyState:
        value = "DEBUG_ONLY"

    class VirtualHumanSource:
        def __init__(
            self,
            spec: PassiveTeleopComponentSourceSpec,
            calibration: TeleopCalibration,
        ) -> None:
            self.spec = spec
            self.started = False
            self.last_emission = None
            self.sequence = 0
            self.binding_receipt = PassiveTeleopComponentBindingReceipt(
                source_contract_hash=spec.semantic_content_hash,
                observer_binding_id=spec.observer_binding_id,
                session_mode="debug",
                rig_absent=True,
                observer_exclusive=True,
                dispatch_forbidden_states=("armed", "running"),
                authoritative=False,
                semantic_content_hash="virtual-right-vive-q6-binding-v1",
            )
            self.session = SimpleNamespace(
                state_machine=SimpleNamespace(state=DebugOnlyState()),
                home_recalibration_required=False,
                motion_hold_mode="",
                motion_hold_reason="",
                tracker_loss_duration_sec=0.0,
                tracker_loss_channels=(),
            )
            arm = calibration.get_arm_for_side("right")
            target_home_desk = np.asarray(
                calibration.metadata["home_capture"]["right_robot_home_desk"],
                dtype=np.float64,
            )
            self.pose = np.linalg.inv(arm.T_D_B) @ target_home_desk
            self.gripper_close = 0.0

        @property
        def session_state(self):
            return self.session.state_machine.state

        def start(self):
            self.started = True
            return self.binding_receipt

        @staticmethod
        def set_calibration_ready() -> None:
            return None

        def step(self, *, now: float):
            command = TeleopCommand(timestamp=now)
            command.add_command(
                EffectorCommand(
                    effector_name="right_arm",
                    side="right",
                    command_type="cartesian_pose",
                    value=self.pose,
                    frame_name="right_arm_base",
                    metadata={
                        "arm_action_spec": ArmActionSpec.tool_tcp(
                            "right",
                            reference_frame="right_arm_base",
                        ).to_metadata()
                    },
                )
            )
            hand_spec = HandActionSpec.dmtac_gripper("right")
            command.add_command(
                EffectorCommand(
                    effector_name="right_hand",
                    side="right",
                    command_type="joint_vector",
                    value=np.asarray(
                        [self.gripper_close, 0, 0, 0, 0, 0],
                        dtype=np.float32,
                    ),
                    metadata={
                        "hand_action_spec": hand_spec.to_metadata(),
                    },
                )
            )
            emission = HumanCommandEmission.capture(
                emission_sequence=self.sequence,
                emitted_at=now,
                contributions=(("virtual-vive-q6", command),),
                merged_command=command,
            )
            self.sequence += 1
            self.last_emission = emission
            return PassiveTeleopComponentStep(
                session_step=SimpleNamespace(
                    command_dispatched=False,
                    channels={},
                ),
                emission=emission,
                emission_valid=True,
                rejection_reason=None,
                command_dispatched=False,
            )

        def close(self) -> None:
            self.started = False

    virtual_sources = []

    def virtual_builder(
        *,
        calibration,
        source_config,
        target_action_spec_hash_by_component,
        target_units_convention_id,
        source_spec,
    ):
        del (
            source_config,
            target_action_spec_hash_by_component,
            target_units_convention_id,
        )
        source = VirtualHumanSource(source_spec, calibration)
        virtual_sources.append(source)
        return source, source_spec

    monkeypatch.setattr(
        sim_hil_module,
        "_build_passive_single_vive_gripper_component_source",
        virtual_builder,
    )
    profile = TRACE_REQUIREMENT_PROFILES_V1[
        "deployable_sim_immediate_component_policy_human_composer_v1"
    ]
    trace_session_id = "sim-hil-three-stage-trace"
    capture = SimHILAcceptanceTraceCapture(
        tmp_path,
        trace_session_id=trace_session_id,
        max_spool_bytes=64 << 20,
    )
    trace_facts = []

    def admit_trace(fact) -> None:
        capture.admit_trace(fact)
        trace_facts.append(fact)

    app = sim_hil_module._build_application(
        MujocoSimHILApplicationConfig(
            enabled=True,
            gripper_input_source="angle_trigger",
            device="cpu",
            realtime_worker=False,
            warmup_timeout_sec=5.0,
            human_emission_timeout_sec=30.0,
            initial_human_alpha=0.0,
            position_control_profile_name="position_15_60_120hz",
            # This deterministic fixture manually advances only a minimal
            # non-realtime reference interval; keep its historical cadence.
            # The interactive 30 Hz default is covered separately.
            compose_target_hz=15.0,
        ),
        trace_sink=admit_trace,
    )
    episode_uuid = None
    finalized = None
    try:
        assert set(app.composition.required_readiness_checks()) == {
            "policy_source_ready",
            "policy_chunk_ttl_ready",
            "measured_state_ready",
        }
        report = app.start()
        assert report.passed
        # Match the GUI acceptance boundary: native runtime/viewer startup
        # succeeds before a retryable episode membership is prepared.
        capture_status = capture.prepare(now=app.clock())
        episode_uuid = capture_status["episode_uuid"]
        assert capture_status["dependency_count"] > 0
        dispatched_cycle = app.step()
        assert dispatched_cycle.execution.status == "dispatched"

        def advance_reference_worker() -> None:
            ticks = (
                int(
                    math.ceil(
                        app.runtime.reference_activation_spec.target_lead_time_sec
                        * app.task_config.physics_hz
                    )
                )
                + 2
            )
            for _ in range(ticks):
                app.runtime.rig.step_physics()

        evaluation = dispatched_cycle.composition_step.evaluation
        assert evaluation is not None
        policy_input = evaluation.input_snapshot.selected_input_by_role["policy"]
        policy_only = evaluation.composition_evaluation.composed_candidate
        assert policy_only is not None
        for component_id, value in policy_only.values_by_component.items():
            np.testing.assert_allclose(
                value.values,
                policy_input.action.values_by_component[component_id].values,
                rtol=0.0,
                atol=1e-10,
            )
        assert app.status()["composition"]["mix"]["mode"] == "policy_only"

        human_receipt = app.set_human_blend_alpha(1.0)
        assert human_receipt.accepted
        assert set(app.composition.required_readiness_checks()) == {
            "human_device_ready",
            "calibration_ready",
            "home_ready",
            "rebase_ready",
            "deadman_ready",
            "measured_state_ready",
        }
        qpos_before_human_motion = app.scene.get_arm_qpos("right").copy()
        gripper_before_human_motion = app.scene.get_gripper_close(
            app.task_config.right_gripper_name
        )
        assert len(virtual_sources) == 1
        virtual_sources[0].pose[0, 3] += 0.005
        virtual_sources[0].gripper_close = 0.8
        # This test runs without the realtime physics worker, so explicitly
        # advance the lower reference clock through the prior activation.
        advance_reference_worker()
        app.sleep(max(0.0, float(app._next_decision_at) - app.clock()))
        human_cycle = app.step()
        assert human_cycle.execution.status == "dispatched"
        assert not np.allclose(
            app.scene.get_arm_qpos("right"),
            qpos_before_human_motion,
            rtol=0.0,
            atol=1e-10,
        )
        selected_human = human_cycle.composition_step.evaluation.input_snapshot.selected_input_by_role[
            "human"
        ]
        np.testing.assert_allclose(
            selected_human.action.values_by_component["right_gripper"].values,
            np.asarray([-0.028, -0.028], dtype=np.float32),
            rtol=0.0,
            atol=1e-7,
        )
        gripper_status = app.runtime.rig.snapshot()["gripper_execution"]
        assert gripper_status is not None
        assert gripper_status["accepted_close_by_gripper"][
            app.task_config.right_gripper_name
        ] == pytest.approx(0.8)
        for _ in range(int(app.task_config.physics_hz)):
            app.runtime.rig.step_physics()
        assert (
            app.scene.get_gripper_close(app.task_config.right_gripper_name)
            > gripper_before_human_motion
        )
        human_evaluation = human_cycle.composition_step.evaluation
        assert human_evaluation is not None
        assert dict(
            [
                fact
                for fact in trace_facts
                if fact.event_type == "composition_dynamic_parameter_snapshot"
            ][-1].payload["alpha_by_component"]
        ) == {
            "right_arm": 1.0,
            "right_gripper": 1.0,
        }
        human_input = human_evaluation.input_snapshot.selected_input_by_role["human"]
        human_only = human_evaluation.composition_evaluation.composed_candidate
        assert human_only is not None
        for component_id, value in human_only.values_by_component.items():
            np.testing.assert_allclose(
                value.values,
                human_input.action.values_by_component[component_id].values,
                rtol=0.0,
                atol=1e-10,
            )
        assert app.status()["composition"]["mix"]["mode"] == "human_only"

        blend_receipt = app.set_human_blend_alpha(0.25)
        assert blend_receipt.accepted
        assert set(app.composition.required_readiness_checks()) == {
            "policy_source_ready",
            "human_device_ready",
            "calibration_ready",
            "home_ready",
            "rebase_ready",
            "deadman_ready",
            "policy_chunk_ttl_ready",
            "measured_state_ready",
        }
        advance_reference_worker()
        app.sleep(max(0.0, float(app._next_decision_at) - app.clock()))
        blended_cycle = app.step()
        assert blended_cycle.execution.status == "dispatched"
        blended_evaluation = blended_cycle.composition_step.evaluation
        assert blended_evaluation is not None
        selected = blended_evaluation.input_snapshot.selected_input_by_role
        assert set(selected) == {"human", "policy"}
        blended = blended_evaluation.composition_evaluation.composed_candidate
        assert blended is not None
        np.testing.assert_allclose(
            blended.values_by_component["right_gripper"].values,
            0.75 * selected["policy"].action.values_by_component["right_gripper"].values
            + 0.25
            * selected["human"].action.values_by_component["right_gripper"].values,
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            blended.values_by_component["right_arm"].values[:3],
            0.75 * selected["policy"].action.values_by_component["right_arm"].values[:3]
            + 0.25
            * selected["human"].action.values_by_component["right_arm"].values[:3],
            rtol=0.0,
            atol=1e-7,
        )
        assert app.status()["composition"]["mix"] == {
            "mode": "proportional_blend",
            "human_alpha_by_component": {
                "right_arm": 0.25,
                "right_gripper": 0.25,
            },
        }
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            app.set_human_blend_alpha(-0.01)
        with pytest.raises(ValueError, match="exactly cover"):
            app.set_human_blend_alpha({"right_arm": 0.5})
        future_receipt = app.set_human_blend_alpha(
            0.75,
            effective_from=app.clock() + 0.5,
        )
        assert future_receipt.accepted
        assert app.status()["composition"]["mix"] == {
            "mode": "proportional_blend",
            "human_alpha_by_component": {
                "right_arm": 0.25,
                "right_gripper": 0.25,
            },
        }
        difference = evaluation.source_difference_observation
        assert difference is not None
        assert len(difference.pairwise_differences) == 1
        assert len(difference.output_residuals) == 2
        execution_difference = app.status()["execution_difference"]
        assert execution_difference["enabled"] is True
        assert execution_difference["measurement_count"] >= 1
        assert execution_difference["latest"] is not None
        assert execution_difference["latest"]["status"] == "available"
        assert execution_difference["latest"]["reason"] == (
            "downstream_actual_reference_at_measurement"
        )
        assert (
            execution_difference["latest"]["component_difference"]["right_arm"][
                "metric"
            ]
            == "cartesian_pose_residual"
        )

        app.hold(reason="test_pause")
        assert app.status()["execution_gate"]["status"] == "hold"
        with pytest.raises(RuntimeError, match="explicit rearm"):
            app.step()
        assert app.rearm().passed
        resumed = app.step()
        assert resumed.composition_step.blocked_reason is None
        app.end_episode(reason="three_stage_acceptance_complete")
        finalized = capture.finalize(now=app.clock())
        assert finalized["commit_status"] == "committed"
    finally:
        app.close()
        capture.close(now=app.clock())
    observed = {item.event_type for item in trace_facts}
    assert (
        not {
            "replay_source_frame_emission",
            "replay_source_adapter_receipt",
            "station_run_manifest",
            "station_intent_transition",
            "station_operation_transition",
            "station_publication_barrier_transition",
            "composition_session_spec",
            "composition_session_transition",
        }
        & observed
    )
    assert {
        "policy_action_emission",
        "human_command_emission",
        "source_adapter_receipt",
        "source_validation",
        "composition_activation_scope",
        "composition_activation_binding",
        "composition_contributor_input",
        "composition_input_admission",
        "composition_evaluation",
        "composition_authority_disposition",
        "source_difference_observation",
        "composition_dynamic_parameter",
        "composition_dynamic_parameter_snapshot",
        "composition_strategy_control_binding",
    } <= observed
    assert profile.required_event_types <= observed
    lineage_issues = evaluate_control_trace_lineage(
        tuple(trace_facts),
        profile=profile,
    )
    lineage_issue_codes = {issue.code for issue in lineage_issues}
    assert (
        not {
            "policy_component_emission_missing_source_validation",
            "policy_component_emission_missing_input",
            "policy_component_input_missing_admission",
            "human_component_emission_missing_source_receipt",
            "human_component_emission_missing_source_validation",
            "human_component_emission_input_count_mismatch",
            "human_component_input_missing_admission",
        }
        & lineage_issue_codes
    )
    assert lineage_issue_codes == set(), tuple(
        (
            issue.code,
            (issue.root_fact_id, *issue.related_fact_ids),
            tuple(
                (
                    fact.event_type,
                    fact.fact_id,
                    fact.payload.get("accepted_for_processing"),
                    fact.payload.get("completion_status"),
                    fact.payload.get("rejection_code"),
                    fact.payload.get("target_execution_request_id"),
                    fact.payload.get("target_id"),
                )
                for fact in trace_facts
                if fact.fact_id in (issue.root_fact_id, *issue.related_fact_ids)
            ),
        )
        for issue in lineage_issues
    )
    assert finalized is not None
    assert episode_uuid is not None
    captured = load_episode_control_trace(tmp_path, episode_uuid)
    assert captured.committed
    captured_event_types = {
        record.fact.event_type
        for record in (
            *captured.dependency_records,
            *captured.episode_records,
        )
    }
    dependency_event_types = {
        record.fact.event_type for record in captured.dependency_records
    }
    assert {
        "producer_epoch_activation",
        "conservative_compound_authority_profile",
    } <= dependency_event_types
    assert (
        profile.required_event_types - {"runtime_trace_terminal"}
    ) <= captured_event_types
    assert {
        "episode_membership_transition",
        "composition_dynamic_parameter",
        "composition_dynamic_parameter_snapshot",
        "source_difference_observation",
        "measurement_alignment",
        "execution_cancellation",
    } <= captured_event_types
    assert (
        sum(
            record.fact.event_type == "episode_membership_transition"
            for record in captured.episode_records
        )
        == 2
    )


@pytest.mark.optional
def test_policy_only_application_defers_human_device_and_profile_binding(
    monkeypatch,
) -> None:
    _require_mujoco_runtime()
    _require_optional_default_checkpoint_fixture()

    def reject_human_device_resolution(**_kwargs):
        raise AssertionError("policy-only construction touched a human device")

    monkeypatch.setattr(
        "teleop_runtime.configs.resolve_single_gripper_input",
        reject_human_device_resolution,
    )
    app = sim_hil_module._build_application(
        MujocoSimHILApplicationConfig(
            enabled=True,
            device="cpu",
            realtime_worker=False,
            initial_human_alpha=0.0,
            defer_human_input=True,
            vive_coordinate_profile_path=None,
            position_control_profile_name="position_15_60_120hz",
            # This fixture expects one new policy target per compose cycle.
            # The interactive default intentionally composes faster than the
            # 15 Hz policy source and may keep the existing execution.
            compose_target_hz=15.0,
        )
    )
    try:
        assert app.human_input_attached is False
        assert app.calibration_binding["mode"] == "human_input_deferred"
        assert set(app.composition.required_readiness_checks()) == {
            "policy_source_ready",
            "policy_chunk_ttl_ready",
            "measured_state_ready",
        }
        report = app.start()
        assert report.passed
        assert app.status()["source_configuration"]["human_input_attached"] is False
        assert app.runtime.reference_receiver is None
        assert app.runtime.reference_activation_spec.activation_mode == (
            "immediate_target"
        )
        assert app.runtime.target_execution_port is not None
        assert app.runtime.target_execution_port.activation_mode == "omitted"
        assert app.policy_source.maximum_hold_sec == pytest.approx(0.25)
        assert app.runtime.config.proposal_horizon_sec == pytest.approx(0.25)
        assert (
            app.composition.contracts.authority_gate_spec.proposal_ttl_sec
            == pytest.approx(0.25)
        )
        assert (
            app.composition.contracts.authority_gate_spec.missing_behavior_by_role
            == {
                "human": "no_action",
                "policy": "no_action",
            }
        )
        assert (
            app.composition.contracts.authority_gate_spec.all_inputs_missing_behavior
            == "no_action"
        )
        assert (
            app.runtime.composition_runtime_config.enable_source_eligibility
            is True
        )
        assert (
            app.runtime.composition_runtime_config.information_plan
            is getattr(app.composition.contracts, "information_plan", None)
        )
        initial_qpos = app.scene.get_arm_qpos("right").copy()
        cycles = []
        dispatched_cycles = []
        for _ in range(40):
            due = app._next_decision_at
            if due is not None:
                time.sleep(max(0.0, due - app.clock()))
            cycle = app.step()
            cycles.append(cycle)
            assert cycle.execution.status in {
                "dispatched",
                "not_due",
                "execution_kept_existing",
            }
            assert app._held is False
            if cycle.execution.status == "dispatched":
                dispatched_cycles.append(cycle)
            if len(dispatched_cycles) == 20:
                break

        assert len(dispatched_cycles) == 20
        assert all(
            cycle.execution.target_dispatch_result is not None
            and cycle.execution.target_dispatch_result.target_admission.accepted
            and cycle.execution.target_dispatch_result.target_activation is None
            for cycle in dispatched_cycles
        )
        target_support_windows = [
            (
                cycle.execution.target_dispatch_result.target.supported_until
                - cycle.execution.target_dispatch_result.target.target_effective_at
            )
            for cycle in dispatched_cycles
        ]
        # Synchronous CPU inference legitimately consumes part of the
        # source-owned support window before target publication.  The
        # remaining interval must stay positive and bounded by the declared
        # 250 ms horizon; it must not be fabricated from the compose period.
        assert min(target_support_windows) > 0.0
        assert max(target_support_windows) <= 0.25 + 1e-6
        assert all(
            right > left
            for left, right in zip(
                (
                    cycle.composition_step.target_effective_at
                    for cycle in dispatched_cycles
                ),
                (
                    cycle.composition_step.target_effective_at
                    for cycle in dispatched_cycles[1:]
                ),
            )
        )
        assert np.linalg.norm(app.scene.get_arm_qpos("right") - initial_qpos) > 1e-5

        original_resolve = app.runtime.pose_resolver.resolution_for_command

        def resolve_with_cpu_tail(command):
            time.sleep(0.08)
            return original_resolve(command)

        monkeypatch.setattr(
            app.runtime.pose_resolver,
            "resolution_for_command",
            resolve_with_cpu_tail,
        )
        slow_cycle = None
        for _ in range(5):
            due = app._next_decision_at
            if due is not None:
                time.sleep(max(0.0, due - app.clock()))
            candidate = app.step()
            assert candidate.execution.status in {
                "dispatched",
                "not_due",
                "execution_kept_existing",
            }
            assert app._held is False
            if candidate.execution.status == "dispatched":
                slow_cycle = candidate
                break
        assert slow_cycle is not None
        slow_admission = slow_cycle.execution.target_dispatch_result.target_admission
        assert (
            slow_admission.admitted_at
            - slow_cycle.execution.target_dispatch_result.target.target_effective_at
        ) >= 0.075

        event_types = {fact.event_type for fact in app.runtime.trace_facts}
        assert {
            "component_execution_target",
            "component_execution_target_admission",
            "target_execution_completion",
        } <= event_types
        assert len(app.runtime._immediate_target_by_reference_sequence) == 1
        assert app.status()["first_control_fault"] is None
    finally:
        app.close()


@pytest.mark.optional
def test_selected_v5_target_reuses_task_owned_scripted_pregrasp_reset() -> None:
    _require_mujoco_runtime()

    prepared = prepare_mujoco_sim_hil_target(
        MujocoSimHILTargetConfig(
            task_config_path=Path(
                "sim/configs/tasks/"
                "rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose.json"
            ),
            realtime_worker=False,
            exposed_pose_sides=("right",),
        )
    )

    initialization = dict(prepared.task_initialization_info)
    assert math.isfinite(float(initialization["resting_center_z_m"]))
    assert {
        key: value
        for key, value in initialization.items()
        if key != "resting_center_z_m"
    } == {
        "owner": "sim.mujoco.tasks.cube_lift.CubeLiftKernel.reset",
        "seed": 0,
        "scripted_pregrasp_reset": True,
        "scripted_contact_reset": False,
    }
    assert not np.allclose(
        prepared.scene.get_arm_qpos("right"),
        prepared.task_config.right_home_qpos,
    )


def _action(*, arm_x: float = 0.0) -> RobotAction:
    return RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            side: np.asarray(
                [arm_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
            for side in ("left", "right")
        },
        hand_qpos_by_side={
            side: np.zeros(6, dtype=np.float32) for side in ("left", "right")
        },
        active_effectors=(
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ),
        frame_by_effector={
            "left_arm": "left_arm_base",
            "right_arm": "right_arm_base",
        },
        controlled_frame_by_effector={
            "left_arm": "tool_tcp",
            "right_arm": "tool_tcp",
        },
        source="sim-hil-policy-fixture",
        metadata={
            "hand_action_specs_by_side": {
                side: spec.to_metadata() for side, spec in _hand_specs().items()
            }
        },
    )


def _emission() -> PolicyActionEmission:
    return PolicyActionEmission(
        emission_id="policy-action-emission:fixture:0",
        admission_mode="on_emission_pop_v1",
        policy_node_id="sim-policy",
        inference_id="inference:0",
        policy_output_id="policy-output:0",
        chunk_id=0,
        chunk_keyframe_index=0,
        chunk_size=1,
        decoded_absolute_action=NormalizedAction(vector=np.zeros(1, dtype=np.float32)),
        pre_safety_robot_action=_action(),
        source_validation=SourceValidationResult(accepted=True),
        representation_receipt=None,
        pre_safety_faults=(),
        produced_at=10.0,
        produced_clock_domain_id="sim-clock",
        emitted_at=10.0,
        target_effective_at=10.0,
        effective_from=10.0,
        effective_until=10.1,
        effective_clock_domain_id="sim-clock",
        schedule_spec_id="schedule:fixture",
        schedule_spec_hash="schedule-hash",
        queue_len_before=1,
        queue_len_after=0,
        queue_pop_count=1,
        semantic_content_hash="policy-emission-hash",
    )


def _human_source_spec(runtime) -> PassiveTeleopComponentSourceSpec:
    hashes = dict(runtime.target_runtime_action_contract.action_spec_hash_by_component)
    return PassiveTeleopComponentSourceSpec(
        components=(
            *(
                PassiveTeleopComponentSpec.arm_pose(
                    f"{side}_arm",
                    f"{side}_arm",
                    action_spec_hash=hashes[f"{side}_arm"],
                    arm_action_spec=ArmActionSpec.tool_tcp(
                        side,
                        reference_frame=f"{side}_arm_base",
                    ),
                )
                for side in ("left", "right")
            ),
            *(
                PassiveTeleopComponentSpec.hand_qpos(
                    f"{side}_hand",
                    f"{side}_hand",
                    action_spec_hash=hashes[f"{side}_hand"],
                    hand_action_spec=_hand_specs()[side],
                )
                for side in ("left", "right")
            ),
        ),
        units_convention_id=(
            runtime.target_runtime_action_contract.units_convention_id
        ),
    )


def _human_emission(
    *,
    sequence: int,
    emitted_at: float,
    arm_x: float,
    hand_qpos: float,
) -> HumanCommandEmission:
    command = TeleopCommand(timestamp=emitted_at)
    for side in ("left", "right"):
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = arm_x
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="cartesian_pose",
                value=pose,
                frame_name=f"{side}_arm_base",
                metadata={
                    "arm_action_spec": ArmActionSpec.tool_tcp(
                        side,
                        reference_frame=f"{side}_arm_base",
                    ).to_metadata(),
                },
            )
        )
        values = np.full(6, hand_qpos, dtype=np.float32)
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_hand",
                side=side,
                command_type="joint_vector",
                value=values,
                metadata={
                    "hand_action_spec": _hand_specs()[side].to_metadata(),
                },
            )
        )
    return HumanCommandEmission.capture(
        emission_sequence=sequence,
        emitted_at=emitted_at,
        contributions=(("virtual-vive-q6", command),),
        merged_command=command,
    )


def test_sim_hil_profile_is_default_off() -> None:
    assert MujocoSimHILProfileConfig().enabled is False


def test_sim_hil_cycle_hash_binds_only_target_facts_carried_at_return() -> None:
    composition_step = MujocoSimHILCompositionStep(
        decision_key="cycle-hash-fixture",
        target_effective_at=10.1,
        control_cut=None,
        readiness=None,
        evaluation=None,
        disposition=None,
        proposal_submission=None,
        hold_transition=None,
    )
    dispatch = SimpleNamespace(
        target=SimpleNamespace(target_id="target-1"),
        target_admission=SimpleNamespace(receipt_id="admission-1"),
        target_activation=None,
    )
    execution = AuthoritativeControlStep(
        status="dispatched",
        command_id="command-1",
        dispatch_receipt_id="dispatch-1",
        target_execution_request=SimpleNamespace(request_id="request-1"),
        target_submission_receipt=SimpleNamespace(receipt_id="submission-1"),
        target_completion_receipt=SimpleNamespace(receipt_id="completion-1"),
        target_dispatch_result=dispatch,
    )
    pending = MujocoSimHILCycleResult(
        cycle_index=0,
        human_step=None,
        human_submission=None,
        policy_submission=None,
        composition_step=composition_step,
        execution=execution,
    )
    activated = replace(
        pending,
        execution=replace(
            execution,
            target_dispatch_result=SimpleNamespace(
                target=dispatch.target,
                target_admission=dispatch.target_admission,
                target_activation=SimpleNamespace(activation_id="activation-1"),
            ),
        ),
    )
    another_request = replace(
        pending,
        execution=replace(
            execution,
            target_execution_request=SimpleNamespace(request_id="request-2"),
        ),
    )

    assert pending.semantic_content_hash != activated.semantic_content_hash
    assert pending.semantic_content_hash != another_request.semantic_content_hash


def test_default_vive_coordinate_profile_is_exact_and_right_compatible() -> None:
    assert DEFAULT_VIVE_COORDINATE_PROFILE.is_absolute()
    assert DEFAULT_VIVE_COORDINATE_PROFILE.name == (
        "vive_rm65_virtual_desk_coordinate_v1.json"
    )
    source = TeleopCalibration.load_json(DEFAULT_CALIBRATION_PROFILE)

    merged, binding = _bind_vive_coordinate_profile(
        source,
        coordinate_profile_path=DEFAULT_VIVE_COORDINATE_PROFILE,
        home_profile_path=DEFAULT_CALIBRATION_PROFILE,
        required_sides=("right",),
    )

    assert Path(binding["vive_coordinate_profile_path"]) == (
        DEFAULT_VIVE_COORDINATE_PROFILE.resolve()
    )
    assert "right" in tuple(binding["active_sides"])
    assert "right_wrist" in merged.tracker_offsets


@pytest.mark.optional
def test_sim_hil_default_target_rates_and_measured_reset_session_home() -> None:
    _require_mujoco_runtime()

    hands = _hand_specs()
    schema = build_mujoco_pose_action_schema(hands)
    composer_id = "control-station-sim-hil-composer"
    contracts = build_default_mujoco_sim_hil_contracts(
        hands,
        composer_producer_id=composer_id,
        proposal_ttl_sec=0.1,
        action_schema_hash=schema.semantic_content_hash,
    )
    target = build_mujoco_sim_hil_target(
        config=MujocoSimHILTargetConfig(
            task_config_path=Path(
                "sim/configs/tasks/"
                "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
            ),
            realtime_worker=False,
            composer_producer_id=composer_id,
        ),
        contracts=contracts,
        composition_runtime_config=(
            build_mujoco_sim_hil_composition_runtime_config(contracts)
        ),
        composition_strategy_control_factory=(
            build_mujoco_sim_hil_strategy_control_factory(contracts)
        ),
        hand_specs_by_side=hands,
    )
    try:
        timing = target.runtime.rig.config.timing
        assert target.runtime.config.decision_hz == 15.0
        assert timing.policy_hz == 15.0
        assert timing.sample_hz == 15.0
        assert timing.reference_hz == 60.0
        assert timing.ik_hz == 60.0
        assert timing.command_hz == 120.0
        planner_config = target.runtime.pose_resolver.ik.planner.config
        assert planner_config.reference_hz == target.runtime.config.decision_hz
        assert planner_config.solve_time_budget_ms == pytest.approx(
            1000.0 / timing.ik_hz
        )
        assert planner_config.collision_min_distance_m == pytest.approx(
            target.scene.scene_spec.collision.safety_minimum_clearance_m
        )
        assert planner_config.collision_min_distance_m == pytest.approx(0.002)
        assert target.runtime.config.enable_component_execution_handoff is True
        target_port = target.runtime.target_execution_port
        assert target_port is not None
        assert target_port.activation_mode == "omitted"
        assert (
            target_port.component_target_capability.semantic_content_hash
            == target.runtime.component_target_capability.semantic_content_hash
        )
        assert all(
            route.output_target_schema_hash
            == target.runtime.target_runtime_action_contract.semantic_content_hash
            for route in target.runtime.component_target_capability.component_routes
        )
        # Both checks observe the same unstarted, physically reset target.
        # Do not repeat its scripted reset just to read the session anchors.
        source = TeleopCalibration.load_json(DEFAULT_CALIBRATION_PROFILE)
        old_robot_home = {
            side: np.asarray(
                source.metadata["home_capture"][f"{side}_robot_home_desk"],
                dtype=np.float64,
            )
            for side in ("left", "right")
        }
        merged, binding = _bind_vive_coordinate_profile(
            source,
            coordinate_profile_path=DEFAULT_VIVE_COORDINATE_PROFILE,
            home_profile_path=DEFAULT_CALIBRATION_PROFILE,
            required_sides=("right",),
        )
        binding["home_profile_path"] = str(DEFAULT_CALIBRATION_PROFILE)
        binding["home_profile_sha256"] = _file_sha256(DEFAULT_CALIBRATION_PROFILE)
        merged, binding = _bind_sim_target_session_anchors(
            merged,
            target=target,
            calibration_binding=binding,
            captured_at=10.0,
        )
        measured = capture_mujoco_sim_hil_tool_tcp_pose_by_side(
            target.scene,
            target.task_config,
        )
        receipt = binding["session_anchor_receipt"]

        assert binding["mode"] == "independent_vive_plus_scene_target_session_anchor_v1"
        assert receipt["source_anchor_origin"] == (
            "stored_human_home_from_selected_profile"
        )
        assert receipt["target_anchor_origin"] == (
            "measured_mujoco_reset_state_tool_tcp"
        )
        assert receipt["target_frame_profile_hash"] == (
            target.target_frame_profile.semantic_content_hash
        )
        assert receipt["tool_frame_profile_hash"] == (
            target.tool_frame_profile.semantic_content_hash
        )
        for side in ("left", "right"):
            component_id = f"{side}_arm"
            reference_frame = f"{side}_arm_base"
            canonical_from_base = target.target_frame_profile.frame_graph.resolve(
                reference_frame,
                target.target_frame_profile.canonical_reference_frame_id,
            ).matrix_target_source
            np.testing.assert_allclose(
                source.get_arm_for_side(side).T_D_B,
                canonical_from_base,
                rtol=0.0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                merged.get_arm_for_side(side).T_D_B,
                canonical_from_base,
                rtol=0.0,
                atol=1e-12,
            )
            robot_home = np.asarray(
                merged.metadata["home_capture"][f"{side}_robot_home_desk"],
                dtype=np.float64,
            )
            np.testing.assert_allclose(
                merged.desk_to_base(robot_home, side),
                measured[side],
                rtol=0.0,
                atol=1e-8,
            )
            assert not np.allclose(
                old_robot_home[side],
                robot_home,
                rtol=0.0,
                atol=1e-3,
            )
            anchor = next(
                item
                for item in receipt["anchors"]
                if item["component_id"] == component_id
            )
            np.testing.assert_allclose(
                anchor["target_home_pose_reference"],
                measured[side],
                rtol=0.0,
                atol=1e-8,
            )
    finally:
        target.runtime.close()


@pytest.mark.optional
def test_sim_hil_target_accepts_a_profile_selected_compose_rate() -> None:
    _require_mujoco_runtime()

    hands = _hand_specs()
    schema = build_mujoco_pose_action_schema(hands)
    composer_id = "control-station-sim-hil-composer-custom-rate"
    contracts = build_default_mujoco_sim_hil_contracts(
        hands,
        composer_producer_id=composer_id,
        proposal_ttl_sec=0.1,
        action_schema_hash=schema.semantic_content_hash,
    )
    target = build_mujoco_sim_hil_target(
        config=MujocoSimHILTargetConfig(
            task_config_path=Path(
                "sim/configs/tasks/"
                "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
            ),
            realtime_worker=False,
            composer_producer_id=composer_id,
            # The historical position profile keeps the source-compatible
            # 15 Hz minimum while allowing a 45 Hz composed target.  The
            # downstream 60/60/120 clocks remain unchanged.
            position_control_profile_name="position",
            compose_target_hz=45.0,
        ),
        contracts=contracts,
        composition_runtime_config=(
            build_mujoco_sim_hil_composition_runtime_config(contracts)
        ),
        composition_strategy_control_factory=(
            build_mujoco_sim_hil_strategy_control_factory(contracts)
        ),
        hand_specs_by_side=hands,
    )
    try:
        timing = target.runtime.rig.config.timing
        assert target.runtime.config.decision_hz == 45.0
        assert timing.policy_hz == 15.0
        assert timing.sample_hz == 45.0
        assert timing.reference_hz == 60.0
        assert timing.ik_hz == 60.0
        assert timing.command_hz == 120.0
        assert target.runtime.pose_resolver.ik.planner.config.reference_hz == 45.0
    finally:
        target.runtime.close()


@pytest.mark.optional
def test_sim_hil_residual_schema_wiring_and_relative_human_submission() -> None:
    _require_mujoco_runtime()

    class PassiveSource:
        def __init__(self, spec: PassiveTeleopComponentSourceSpec) -> None:
            self.spec = spec
            self.started = False
            self.binding_receipt = PassiveTeleopComponentBindingReceipt(
                source_contract_hash=spec.semantic_content_hash,
                observer_binding_id=spec.observer_binding_id,
                session_mode="debug",
                rig_absent=True,
                observer_exclusive=True,
                dispatch_forbidden_states=("armed", "running"),
                authoritative=False,
                semantic_content_hash="virtual-passive-source-binding",
            )

        def start(self) -> None:
            self.started = True

        def step(self, *, now: float):
            raise AssertionError("This test submits immutable emissions directly.")

        def close(self) -> None:
            self.started = False

    hands = _hand_specs()
    schema = build_mujoco_pose_action_schema(hands)
    profile = build_mujoco_policy_human_residual_profile(
        absolute_action_schema=schema,
        frame_graph=CoordinateFrameGraph(
            graph_id="mujoco-profile-transform-fixture",
            frames=tuple(
                CoordinateFrameSpec(frame_id)
                for frame_id in (
                    "left_arm_base",
                    "left_arm/tool_tcp",
                    "right_arm_base",
                    "right_arm/tool_tcp",
                )
            ),
            transforms=(),
        ),
        concrete_reference_frame_by_component={
            f"{side}_arm": f"{side}_arm_base" for side in ("left", "right")
        },
        concrete_controlled_frame_by_component={
            f"{side}_arm": f"{side}_arm/tool_tcp" for side in ("left", "right")
        },
        source_binding_id="virtual-human-source",
        source_binding_hash="virtual-human-source-hash",
        composer_producer_id="control-station-sim-hil-composer",
        proposal_ttl_sec=0.1,
        config=MujocoPolicyHumanResidualProfileConfig(enabled=True),
    )
    target = build_mujoco_sim_hil_target(
        config=MujocoSimHILTargetConfig(
            task_config_path=Path(
                "sim/configs/tasks/"
                "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
            ),
            realtime_worker=False,
        ),
        contracts=profile.sim_hil_contracts,
        composition_runtime_config=(
            build_mujoco_sim_hil_composition_runtime_config(
                profile.sim_hil_contracts
            )
        ),
        composition_strategy_control_factory=(
            build_mujoco_sim_hil_strategy_control_factory(
                profile.sim_hil_contracts
            )
        ),
        hand_specs_by_side=hands,
        composition_input_action_schema_by_role=(profile.input_action_schema_by_role),
    )
    source_spec = _human_source_spec(target.runtime)
    source = PassiveSource(source_spec)
    composition = build_scoped_mujoco_sim_hil_composition(
        runtime=target.runtime,
        human_source=source,
        human_source_spec=source_spec,
        human_input_transform=profile.input_adapter,
        readiness_provider=lambda *, now: (MujocoSimHILReadinessEvidence.all_ready()),
        contracts=profile.sim_hil_contracts,
        config=MujocoSimHILProfileConfig(
            enabled=True,
            profile_id=profile.config.profile_id,
            human_role=profile.config.residual_role,
            policy_role=profile.config.baseline_role,
        ),
    )
    try:
        # Verify installation before exercising the same target's input path.
        installed = {
            role: role_schema.semantic_content_hash
            for role, role_schema in (
                target.runtime.policy_topology.input_action_schema_by_role.items()
            )
        }
        assert installed == dict(
            profile.sim_hil_contracts.composition_spec.input_action_schema_hash_by_role
        )
        assert installed["policy"] == schema.semantic_content_hash
        assert installed["human_residual"] != schema.semantic_content_hash
        composition.start()
        first_at = float(target.runtime.clock()) + 0.001
        first = _human_emission(
            sequence=0,
            emitted_at=first_at,
            arm_x=0.2,
            hand_qpos=0.4,
        )
        first_submission = composition.submit_human_emission(
            first,
            observed_at=first_at,
        )
        assert first_submission is not None and first_submission.accepted
        assert (
            composition.submit_human_emission(
                first,
                observed_at=first_at,
            )
            is first_submission
        )
        assert len(target.runtime.policy_topology.composition_inputs) == 1

        second_at = first_at + 0.01
        second = _human_emission(
            sequence=1,
            emitted_at=second_at,
            arm_x=0.3,
            hand_qpos=0.1,
        )
        submission = composition.submit_human_emission(
            second,
            observed_at=second_at,
        )
        assert submission is not None and submission.accepted
        assert len(target.runtime.policy_topology.composition_inputs) == 2
        cut = target.runtime.capture_control_cut(
            cutoff_at=max(
                second_at,
                float(submission.input_receipt.available_at),
            )
        )
        records = (
            target.runtime.policy_topology.composition_inputs.records_at_or_before(cut)
        )
        assert len(records) == 2
        anchor_record, record = records
        for component_id in anchor_record.action.action_schema.component_ids:
            anchor_value = anchor_record.action.value(component_id)
            expected_identity = np.zeros_like(anchor_value.values)
            if anchor_value.payload_schema.representation_family == "cartesian_pose":
                expected_identity[-1] = 1.0
            np.testing.assert_allclose(
                anchor_value.values,
                expected_identity,
                atol=1e-12,
            )
        assert record.input_role == "human_residual"
        assert (
            record.action.action_schema.semantic_content_hash
            == profile.relative_contracts.residual_action_schema.semantic_content_hash
        )
        for side in ("left", "right"):
            np.testing.assert_allclose(
                record.action.value(f"{side}_arm").values[:3],
                (0.1, 0.0, 0.0),
                atol=1e-6,
            )
            np.testing.assert_allclose(
                record.action.value(f"{side}_hand").values,
                np.full_like(
                    record.action.value(f"{side}_hand").values,
                    -0.3,
                ),
                atol=1e-6,
            )
    finally:
        composition.close()


@pytest.mark.optional
def test_sim_hil_target_binds_non_off_execution_observation_plan() -> None:
    _require_mujoco_runtime()

    class ObservationSink:
        def __init__(self) -> None:
            self.values: list[object] = []

        def publish_execution_observation(self, observation: object) -> bool:
            self.values.append(observation)
            return True

    hands = _hand_specs()
    schema = build_mujoco_pose_action_schema(hands)
    composer_id = "control-station-sim-hil-observation-binding"
    contracts = build_default_mujoco_sim_hil_contracts(
        hands,
        composer_producer_id=composer_id,
        proposal_ttl_sec=0.1,
        action_schema_hash=schema.semantic_content_hash,
    )
    plan = ExecutionObservationPlan(
        plan_id="sim-hil-observation:required",
        reference_mode="full_frames",
        measurement_mode="full_frames",
        component_ids=("right_arm",),
    )
    sink = ObservationSink()
    trace_facts = []
    target = build_mujoco_sim_hil_target(
        config=MujocoSimHILTargetConfig(
            task_config_path=Path(
                "sim/configs/tasks/"
                "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
            ),
            realtime_worker=False,
            composer_producer_id=composer_id,
            execution_observation_plan=plan,
        ),
        contracts=contracts,
        composition_runtime_config=(
            build_mujoco_sim_hil_composition_runtime_config(contracts)
        ),
        composition_strategy_control_factory=(
            build_mujoco_sim_hil_strategy_control_factory(contracts)
        ),
        hand_specs_by_side=hands,
        execution_observation_sink=sink,
        trace_sink=trace_facts.append,
    )
    try:
        binding = target.runtime.execution_observation_binding
        capability = target.runtime.execution_observation_capability
        assert binding is not None
        assert capability is not None
        assert binding.plan_hash == plan.semantic_content_hash
        assert binding.capability_hash == capability.semantic_content_hash
        assert binding.component_ids == ("right_arm",)
        assert sink.values == []
        target.runtime.start()
        target.runtime.step()
        assert any(
            isinstance(value, ExecutionMeasurementObservation) for value in sink.values
        )
        assert {
            value.stream
            for value in sink.values
            if isinstance(value, ExecutionObservationWatermark)
        } == {"reference", "measurement"}
        assert trace_facts
    finally:
        target.runtime.close()


def test_sim_hil_default_off_rejects_before_starting_any_lower_runtime() -> None:
    class UnstartedRuntime:
        def __init__(self):
            self.start_count = 0

        def start(self):
            self.start_count += 1

    runtime = UnstartedRuntime()
    composition = object.__new__(MujocoSimHILComposition)
    composition.runtime = runtime
    composition.config = MujocoSimHILProfileConfig()
    composition._lock = threading.RLock()
    composition._started = False

    with pytest.raises(RuntimeError, match="default-off"):
        composition.start()
    assert runtime.start_count == 0


def test_sim_hil_lower_hold_failure_is_not_silently_dropped() -> None:
    class FailingRuntime:
        @staticmethod
        def transition_execution_gate(*_args, **_kwargs):
            raise RuntimeError("lower hold failed")

    composition = object.__new__(MujocoSimHILComposition)
    composition.runtime = FailingRuntime()

    with pytest.raises(RuntimeError, match="lower hold failed"):
        composition._hold(reason="test_lower_failure")


def _readiness_gate_test_composition(
    *,
    readiness,
    human_alpha: float = 1.0,
) -> tuple[object, object]:
    class Runtime:
        def __init__(self) -> None:
            self.transitions: list[tuple[str, dict[str, object]]] = []

        @staticmethod
        def clock() -> float:
            return 10.0

        @staticmethod
        def capture_execution_gate_snapshot(*, now: float):
            assert now == 10.0
            return SimpleNamespace(status="ready")

        def transition_execution_gate(self, status, **kwargs):
            self.transitions.append((status, kwargs))
            return SimpleNamespace(
                event=SimpleNamespace(event_id=f"gate:{status}"),
                snapshot=SimpleNamespace(status=status),
            )

    runtime = Runtime()
    composition = object.__new__(MujocoSimHILComposition)
    composition.runtime = runtime
    composition.config = MujocoSimHILProfileConfig(enabled=True)
    composition._lock = threading.RLock()
    composition._started = True
    composition._step_by_decision_key = {}
    composition._readiness_provider = lambda *, now: readiness
    composition._human_alpha_by_component = {"right_arm": human_alpha}
    composition._blend_alpha_schedule = []
    return composition, runtime


@pytest.mark.parametrize(
    ("failed_field", "human_alpha"),
    (
        ("human_device_ready", 1.0),
        ("policy_source_ready", 0.0),
        ("policy_chunk_ttl_ready", 0.0),
    ),
)
def test_sim_hil_temporary_source_gap_is_not_an_execution_gate_failure(
    failed_field: str,
    human_alpha: float,
) -> None:
    readiness = replace(
        MujocoSimHILReadinessEvidence.all_ready(),
        **{failed_field: False},
        detail_by_check={failed_field: "temporarily_unavailable"},
    )
    composition, runtime = _readiness_gate_test_composition(
        readiness=readiness,
        human_alpha=human_alpha,
    )

    failures = composition._runtime_gate_readiness_failures(
        readiness,
        target_effective_at=10.0,
    )

    assert failures == ()
    assert runtime.transitions == []


def test_sim_hil_nonrecoverable_readiness_failure_still_latches_hold() -> None:
    readiness = replace(
        MujocoSimHILReadinessEvidence.all_ready(),
        measured_state_ready=False,
        detail_by_check={"measured_state_ready": "mujoco_state_nonfinite"},
    )
    composition, runtime = _readiness_gate_test_composition(readiness=readiness)

    result = composition.compose_at(
        decision_key="invalid-measured-state",
        target_effective_at=10.0,
    )

    assert result.blocked_reason == "manual_recovery_required"
    assert result.hold_transition is not None
    assert runtime.transitions == [
        (
            "hold",
            {
                "reason": "sim_hil:readiness_not_ready:measured_state_ready",
                "source_event_ids": (),
                "force_cancellation": True,
            },
        )
    ]


def test_sim_hil_existing_lower_hold_does_not_repeat_cancellation() -> None:
    class HeldRuntime:
        def __init__(self):
            self.transition_count = 0

        @staticmethod
        def clock():
            return 10.0

        @staticmethod
        def capture_execution_gate_snapshot(*, now):
            assert now == 10.0
            return SimpleNamespace(status="hold")

        def transition_execution_gate(self, *_args, **_kwargs):
            self.transition_count += 1
            raise AssertionError("existing HOLD must not be relinearized")

    runtime = HeldRuntime()
    composition = object.__new__(MujocoSimHILComposition)
    composition.runtime = runtime
    composition.config = MujocoSimHILProfileConfig(enabled=True)
    composition._lock = threading.RLock()
    composition._started = True
    composition._step_by_decision_key = {}
    composition._readiness_provider = lambda *, now: (
        MujocoSimHILReadinessEvidence.all_ready()
    )

    result = composition.compose_at(
        decision_key="already-held",
        target_effective_at=10.0,
    )

    assert result.blocked_reason == "execution_gate_not_ready:hold"
    assert result.hold_transition is None
    assert runtime.transition_count == 0


def test_sim_hil_public_hold_is_idempotent_and_preserves_fault() -> None:
    class Runtime:
        def __init__(self):
            self.status = "ready"
            self.transitions: list[tuple[str, dict[str, object]]] = []

        @staticmethod
        def clock():
            return 10.0

        def capture_execution_gate_snapshot(self, *, now):
            assert now == 10.0
            return SimpleNamespace(status=self.status)

        def transition_execution_gate(self, status, **kwargs):
            self.transitions.append((status, kwargs))
            self.status = status
            return SimpleNamespace(snapshot=SimpleNamespace(status=status))

    runtime = Runtime()
    composition = object.__new__(MujocoSimHILComposition)
    composition.runtime = runtime
    composition._lock = threading.RLock()
    composition._started = True

    first = composition.request_hold(reason="pause")
    second = composition.request_hold(reason="duplicate")
    runtime.status = "fault"
    fault = composition.request_hold(reason="must_not_downgrade")

    assert first is not None
    assert second is None
    assert fault is None
    assert [item[0] for item in runtime.transitions] == ["hold"]


def test_default_sim_hil_contract_is_all_required_and_compound() -> None:
    contracts = build_default_mujoco_sim_hil_contracts(
        _hand_specs(),
        composer_producer_id="sim-hil-composer",
        proposal_ttl_sec=0.1,
    )

    assert contracts.composition_spec.input_roles == (
        MUJOCO_SIM_HIL_HUMAN_ROLE_V1,
        MUJOCO_SIM_HIL_POLICY_ROLE_V1,
    )
    assert (
        contracts.composition_spec.required_roles
        == contracts.composition_spec.input_roles
    )
    assert contracts.authority_profile.component_coverage == (
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    )
    assert contracts.authority_profile.selection_groups == (
        "arms",
        "left_hand",
        "right_hand",
    )
    assert contracts.authority_gate_spec.missing_behavior_by_role == {
        "human": "hold",
        "policy": "hold",
    }


def test_sim_hil_contract_can_bind_the_exact_mujoco_component_schema() -> None:
    schema = build_mujoco_pose_action_schema(_hand_specs())
    contracts = build_default_mujoco_sim_hil_contracts(
        _hand_specs(),
        composer_producer_id="sim-hil-composer",
        proposal_ttl_sec=0.1,
        policy_active_effectors=("right_arm", "right_hand"),
        action_schema_hash=schema.semantic_content_hash,
    )

    assert contracts.composition_spec.action_schema_hash == (
        schema.semantic_content_hash
    )


def test_policy_emission_becomes_non_authoritative_input_with_activation_lineage() -> (
    None
):
    binding = PolicyCompositionInputBinding(
        source_slot_id="sim-policy",
        role=MUJOCO_SIM_HIL_POLICY_ROLE_V1,
        runtime_session_id="sim-hil-session",
        control_domain_id="sim-hil-domain",
        producer_id="sim-policy",
        producer_epoch=1,
        policy_node_id="sim-policy",
        policy_clock_domain_id="sim-clock",
        component_catalog_hash="catalog-hash",
        action_space=BIMANUAL_EE_POSE_V1,
        required_active_effectors=(
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ),
        action_schema=project_robot_action_v1_to_component_action(
            _action()
        ).action_schema,
    )
    activation = build_composition_activation_scope(
        operation_spec_id="sim-hil",
        operation_spec_hash="sim-hil-contract",
        session_epoch=1,
        activated_at=10.0,
        clock_domain_id="sim-clock",
        input_barrier_at=10.0,
        required_roles=("human", "policy"),
    )
    adapter = PolicyCompositionInputAdapter(binding)
    draft = adapter.adapt(_emission(), activation_scope=activation)

    assert draft.input_role == "policy"
    assert draft.authority_candidate is False
    assert draft.composition_activation_id == activation.activation_id
    assert draft.composition_activation_hash == activation.semantic_content_hash
    assert draft.audit_metadata["chunk_id"] == 0
    assert draft.audit_metadata["inference_id"] == "inference:0"

    with pytest.raises(ValueError, match="another composition activation"):
        adapter.adapt(
            _emission(),
            activation_scope=build_composition_activation_scope(
                operation_spec_id="sim-hil",
                operation_spec_hash="sim-hil-contract",
                session_epoch=2,
                activated_at=11.0,
                clock_domain_id="sim-clock",
                input_barrier_at=11.0,
                required_roles=("human", "policy"),
            ),
        )


def test_policy_adapter_temporal_port_requires_explicit_guaranteed_next() -> None:
    action_schema = project_robot_action_v1_to_component_action(_action()).action_schema
    binding = PolicyCompositionInputBinding(
        source_slot_id="sim-policy",
        role="policy",
        runtime_session_id="sim-hil-session",
        control_domain_id="sim-hil-domain",
        producer_id="sim-policy",
        producer_epoch=1,
        policy_node_id="sim-policy",
        policy_clock_domain_id="sim-clock",
        component_catalog_hash="catalog-hash",
        action_space=BIMANUAL_EE_POSE_V1,
        required_active_effectors=(
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ),
        action_schema=action_schema,
    )
    adapter = PolicyCompositionInputAdapter(binding)
    emission = _emission()

    fallback = adapter.adapt(emission)
    np.testing.assert_allclose(
        fallback.action.value("left_arm").values[0],
        0.0,
    )

    guaranteed = GuaranteedNextComponentAction(
        guarantee_id="sync-chunk:0:next:guaranteed",
        source_fact_id="sync-chunk:0:step:1",
        source_fact_hash="sync-chunk-step-1-hash",
        action=project_robot_action_v1_to_component_action(
            _action(arm_x=0.2),
            action_schema=action_schema,
        ),
        target_effective_at=10.1,
        clock_domain_id="sim-clock",
    )
    interpolated = adapter.adapt_interpolated_at(
        emission,
        target_effective_at=10.05,
        effective_until=10.06,
        sampled_at=10.049,
        guaranteed_next_step=guaranteed,
    )
    assert interpolated.sample.derivation_kind == ("interpolated_guaranteed_next")
    np.testing.assert_allclose(
        interpolated.draft.action.value("left_arm").values[0],
        0.1,
    )
    assert interpolated.draft.source_fact_ids == (
        interpolated.sample.sample_id,
        emission.emission_id,
        guaranteed.source_fact_id,
    )
    with pytest.raises(TypeError, match="GuaranteedNextComponentAction"):
        adapter.adapt_interpolated_at(
            emission,
            target_effective_at=10.06,
            effective_until=10.07,
            sampled_at=10.059,
            guaranteed_next_step=None,  # type: ignore[arg-type]
        )


def test_policy_binding_rejects_partial_component_action() -> None:
    binding = PolicyCompositionInputBinding(
        source_slot_id="sim-policy",
        role="policy",
        runtime_session_id="sim-hil-session",
        control_domain_id="sim-hil-domain",
        producer_id="sim-policy",
        producer_epoch=1,
        policy_node_id="sim-policy",
        policy_clock_domain_id="sim-clock",
        component_catalog_hash="catalog-hash",
        action_space=BIMANUAL_EE_POSE_V1,
        required_active_effectors=("left_arm", "right_arm", "left_hand", "right_hand"),
        action_schema=project_robot_action_v1_to_component_action(
            _action()
        ).action_schema,
    )
    partial = _emission()
    partial_action = RobotAction(
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side={
            side: np.asarray(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
            for side in ("left", "right")
        },
        active_effectors=("left_arm", "right_arm"),
    )
    object.__setattr__(partial, "pre_safety_robot_action", partial_action)
    with pytest.raises(ValueError, match="exactly cover"):
        binding.validate_emission(partial)


def test_fixed_alpha_composition_close_preserves_pending_runtime_for_retry() -> None:
    class FailOnceClose:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("synthetic runtime close failure")

    class RecordingClose:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    runtime = FailOnceClose()
    human_source = RecordingClose()
    composition = object.__new__(MujocoSimHILComposition)
    composition.runtime = runtime
    composition.human_source = human_source
    composition.contracts = SimpleNamespace(
        initial_alpha_by_component={"right_arm": 0.5}
    )
    composition._lock = threading.RLock()
    composition._started = True
    composition._runtime_close_pending = True
    composition._human_source_close_pending = True
    composition._activation_scope = object()
    composition._submission_by_key = {("human", "right_arm"): object()}
    composition._human_anchor_only_source_ids = {"human-anchor"}
    composition._step_by_decision_key = {"decision": object()}
    composition._cycle_index = 4
    composition._next_blend_sequence = 3
    composition._human_alpha_by_component = {"right_arm": 0.75}
    composition._blend_alpha_schedule = [(10.0, 1, {"right_arm": 0.75})]
    activation = composition._activation_scope

    with pytest.raises(RuntimeError, match="synthetic runtime close failure"):
        composition.close()

    assert composition.started is True
    assert composition.activation_scope is activation
    assert composition._submission_by_key

    composition.close()

    assert runtime.close_calls == 2
    assert human_source.close_calls == 1
    assert composition.started is False
    assert composition.activation_scope is None


def test_application_close_waits_for_runtime_before_sources_and_viewer() -> None:
    events: list[str] = []

    class Composition:
        started = True

        def close(self) -> None:
            events.append("composition.close")
            if events.count("composition.close") == 1:
                raise RuntimeError("synthetic composition close failure")
            self.started = False
            # Fixed-alpha composition owns the source close on success.
            human_source.close()

    class Source:
        def __init__(self, name: str) -> None:
            self.name = name
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            events.append(f"{self.name}.close")

    class Viewer(Source):
        pass

    composition = Composition()
    policy_source = Source("policy")
    human_source = Source("human")
    viewer = Viewer("viewer")
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=composition,
        policy_source=policy_source,
        human_source=human_source,
        readiness_provider=SimpleNamespace(),
        viewer=viewer,
    )
    app._started = True
    app._sources_started = True
    app.hold = lambda *, reason: events.append(f"hold:{reason}")

    with pytest.raises(RuntimeError, match="synthetic composition close failure"):
        app.close()

    assert app._closed is False
    assert policy_source.close_calls == 0
    assert human_source.close_calls == 0
    assert viewer.close_calls == 0

    app.close()

    assert app._closed is True
    assert policy_source.close_calls == 1
    assert human_source.close_calls == 1
    assert viewer.close_calls == 1
    assert events.index("composition.close") < events.index("viewer.close")


def test_preview_only_close_does_not_enter_unstarted_composition() -> None:
    close_events: list[str] = []
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=SimpleNamespace(),
        composition=SimpleNamespace(
            started=False,
            close=lambda: pytest.fail(
                "preview-only close must not enter composition lifecycle"
            ),
        ),
        policy_source=SimpleNamespace(
            close=lambda: close_events.append("policy.close")
        ),
        human_source=SimpleNamespace(
            close=lambda: close_events.append("human.close")
        ),
        readiness_provider=SimpleNamespace(),
        viewer=SimpleNamespace(
            close=lambda: close_events.append("viewer.close")
        ),
    )
    app._viewer_preview_started = True

    app.close()

    assert app._closed is True
    assert close_events == ["policy.close", "viewer.close"]
