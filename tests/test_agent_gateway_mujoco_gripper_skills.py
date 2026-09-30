from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from agent_gateway.profiles.mujoco.gripper_skill import (
    MujocoGripperActionSkillLifecycle,
)
from agent_gateway.profiles.mujoco.motion_application import (
    MujocoAgentSkillMotionApplication,
)
from agent_gateway.profiles.mujoco.motion_factory import (
    build_mujoco_agent_tcp_pose_gateway_session,
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
)
from agent_gateway.protocol.contracts import AgentSkillRequest
from sim.mujoco.actuation.gripper_profile import cad_bimanual_dmtac_gripper_profile
from sim.mujoco.config import GripperSpec
from sim.mujoco.deploy import MujocoSimHILTargetConfig
from sim.mujoco.deploy.component_action_profile import (
    MujocoComponentActionRobotActionV1Materializer,
    build_mujoco_pose_qpos_component_target_profile,
)


TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_reach_fixed_pose_xyz_proprio_v1.json"
)


@pytest.fixture
def master_joint_application(monkeypatch):
    profile = cad_bimanual_dmtac_gripper_profile()
    name = profile.gripper_for_side("right")
    spec = GripperSpec(
        name=name, joint_names=profile.joint_names_by_gripper[name],
        open_qpos=profile.open_qpos_by_gripper[name],
        closed_qpos=profile.closed_qpos_by_gripper[name], side="right",
    )
    # Measured V4 residuals from the previously rejected return movement.
    measured = np.array([
        -5.053170752944425e-05, -3.963131803175202e-06,
        6.590426346519962e-05, -4.054239070683252e-06,
        -4.5451964979292825e-05, 4.1206105379387736e-05,
    ])
    scene = SimpleNamespace(
        arm_specs={"right": object()}, gripper_specs={name: spec},
        state_is_finite=lambda: bool(np.isfinite(measured).all()),
        get_gripper_qpos=lambda _name: measured.copy(),
        get_gripper_close=lambda _name: spec.close_from_qpos(measured),
    )
    target_profile = build_mujoco_pose_qpos_component_target_profile(
        scene, profile_id="test.master-joint-support.v1",
        arm_side_by_component={"right_arm": "right"},
        gripper_name_by_component={"right_gripper": name},
        arm_reference_frame_by_component={"right_arm": "right_arm_base"},
        arm_controlled_frame_by_component={"right_arm": "tool_tcp"},
        selection_group_by_component={"right_arm": "arms", "right_gripper": "gripper"},
    )
    application = object.__new__(MujocoAgentSkillMotionApplication)
    application._lock = threading.RLock()
    application._closed = False
    application.task_config = SimpleNamespace(right_gripper_name=name)
    capture = Mock(side_effect=lambda callback: callback(scene))
    application.runtime = SimpleNamespace(
        action_schema=target_profile.action_schema, clock=lambda: 10.0,
        rig=SimpleNamespace(capture_scene=capture, config=SimpleNamespace(
            gripper_actuation_profile=profile,
        )),
    )
    monkeypatch.setattr(
        "agent_gateway.profiles.mujoco.motion_application."
        "capture_mujoco_sim_hil_tool_tcp_pose_by_side",
        lambda *_args: {"right": np.eye(4)},
    )
    return application, scene, measured, capture


@pytest.mark.parametrize("master_index", (0, 2))
def test_arm_only_support_uses_configured_master_without_changing_measurements(
    master_joint_application, master_index,
):
    application, scene, measured, capture = master_joint_application
    name, spec = next(iter(scene.gripper_specs.items()))
    config = application.runtime.rig.config
    config.gripper_actuation_profile = replace(
        config.gripper_actuation_profile,
        master_joint_by_gripper={name: spec.joint_names[master_index]},
    )
    action = application.complete_component_action({})
    assert capture.call_count == 1  # Arm and gripper use the same locked cut.
    target = action.value("right_gripper").values
    assert target == pytest.approx([measured[master_index]] * 6)
    convert = MujocoComponentActionRobotActionV1Materializer._physical_qpos_to_close
    assert convert(target, open_qpos=spec.open_qpos, closed_qpos=spec.closed_qpos) == (
        pytest.approx(measured[master_index] / 0.864)
    )
    readback = application.capture_gripper_state_by_component()["right_gripper"]
    assert readback["qpos"] == pytest.approx(measured)
    assert readback["close_fraction"] == pytest.approx(spec.close_from_qpos(measured))
    # Explicit independent joint targets are not repaired or silently accepted.
    explicit = application.complete_component_action({"right_gripper": measured})
    assert explicit.value("right_gripper").values == pytest.approx(measured)
    with pytest.raises(ValueError, match="scalar-close compatibility manifold"):
        convert(explicit.value("right_gripper").values,
                open_qpos=spec.open_qpos, closed_qpos=spec.closed_qpos)


@pytest.mark.parametrize("close", (0.25, 1.1))
def test_master_support_respects_joint_geometry_without_clamping(
    master_joint_application, close,
):
    application, scene, measured, _capture = master_joint_application
    name, spec = next(iter(scene.gripper_specs.items()))
    opened = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    closed = np.array([0.7, 0.8, -0.3, 1.0, 1.1, 1.2])
    scene.gripper_specs[name] = replace(
        spec, open_qpos=tuple(opened), closed_qpos=tuple(closed),
    )
    measured[2] = opened[2] + close * (closed[2] - opened[2])
    action = application.complete_component_action({})
    assert action.value("right_gripper").values == pytest.approx(
        opened + close * (closed - opened)
    )


@pytest.mark.parametrize("invalid", ("nonfinite", "fixed_master"))
def test_arm_only_support_rejects_invalid_measurement_or_master(
    master_joint_application, invalid,
):
    application, scene, measured, _capture = master_joint_application
    name, spec = next(iter(scene.gripper_specs.items()))
    if invalid == "nonfinite":
        measured[0] = np.nan  # A follower remains subject to finite-state checks.
        error, message = RuntimeError, "non-finite"
    else:
        closed = list(spec.closed_qpos)
        closed[2] = spec.open_qpos[2]
        scene.gripper_specs[name] = replace(spec, closed_qpos=tuple(closed))
        error, message = ValueError, "master joint has no closing range"
    with pytest.raises(error, match=message):
        application.complete_component_action({})


def _request(
    *,
    session,
    adapter,
    skill_id: str,
    parameters: dict[str, object],
    sequence: int,
) -> AgentSkillRequest:
    observation = adapter.get_observation_snapshot()
    entry = session.contract_bundle.route_manifest.route(skill_id, "1")
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id="test:mujoco-gripper",
        request_id=f"request:mujoco-gripper:{sequence}",
        idempotency_key=f"test:mujoco-gripper:{sequence}",
        client_sequence=sequence,
        runtime_session_id=session.contract_bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=(
            session.contract_bundle.route_manifest.semantic_content_hash
        ),
        expected_station_revision=int(observation["station_revision"]),
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=(
            entry.deployment_binding.deployment_binding_hash
        ),
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters=parameters,
        observation_snapshot_id=observation["snapshot_id"],
        requested_mode="sim_active",
        requested_timeout_ms=min(8_000, route.max_operation_duration_ms),
    )


def _submit_and_wait(
    *,
    session,
    adapter,
    skill_id: str,
    parameters: dict[str, object],
    sequence: int,
):
    request = _request(
        session=session,
        adapter=adapter,
        skill_id=skill_id,
        parameters=parameters,
        sequence=sequence,
    )
    admission = adapter.submit_skill_request(request.to_wire())
    assert admission["accepted"]
    assert session.wait(timeout_sec=15.0)
    operation = session.service.operation_store.get(
        admission["operation"]["operation_id"]
    )
    assert operation is not None
    return operation


@pytest.mark.integration
@pytest.mark.parametrize("inject_not_due", (False, True))
def test_gripper_close_state_open_round_trip_on_real_mujoco(
    monkeypatch, inject_not_due: bool
) -> None:
    trace_facts = []
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=TASK_CONFIG,
            realtime_worker=False,
            runtime_session_id="test:mujoco-gripper-round-trip",
            control_domain_id="test:mujoco-gripper-round-trip",
            execution_observation_plan=(
                mujoco_agent_tcp_pose_execution_observation_plan_v1()
            ),
            exposed_pose_sides=("right",),
            allow_scripted_task_reset=False,
        ),
        allowed_client_ids=("test:mujoco-gripper",),
        trace_sink=trace_facts.append,
    )
    try:
        session.application.start_observation_ready()
        adapter = session.wire_adapter(
            authenticated_client_id="test:mujoco-gripper"
        )
        original_submit = session.application.submit_and_step

        def submit_with_pending_ticks(**kwargs):
            cycle = int(kwargs["source_event_id"].rsplit(":", 1)[1])
            if inject_not_due and cycle in (0, 2):
                return SimpleNamespace(
                    proposal_admission=SimpleNamespace(accepted=True),
                    authoritative_step=SimpleNamespace(status="not_due"),
                )
            return original_submit(**kwargs)

        monkeypatch.setattr(
            session.application, "submit_and_step", submit_with_pending_ticks
        )
        initial_arm = session.application.capture_tcp_pose_by_component()[
            "right_arm"
        ]

        close_request = _request(
            session=session,
            adapter=adapter,
            skill_id="motion.close_gripper",
            parameters={"component_id": "right_gripper"},
            sequence=0,
        )
        close_admission = adapter.submit_skill_request(close_request.to_wire())
        assert close_admission["accepted"]
        assert session.wait(timeout_sec=15.0)
        close_operation = session.service.operation_store.get(
            close_admission["operation"]["operation_id"]
        )
        assert close_operation is not None
        assert close_operation.status == "completed", (
            close_operation.reason_code,
            dict(close_operation.progress_summary),
        )
        assert close_operation.task_outcome == "succeeded"
        assert close_operation.execution_result is not None
        command_receipt_ref = close_operation.execution_result[
            "command_receipt_ref"
        ]
        dispatch_fact_ids = {
            fact.fact_id
            for fact in trace_facts
            if fact.event_type == "dispatch_receipt"
        }
        assert command_receipt_ref in dispatch_fact_ids
        assert command_receipt_ref in close_operation.execution_receipt_refs
        assert len(close_operation.evaluator_receipt_refs) == 1
        assert close_operation.evaluator_receipt_refs[0].startswith(
            "agent-skill-evaluation:"
        )
        assert close_operation.aggregate_evaluation_receipt_ref is not None
        assert close_operation.aggregate_evaluation_receipt_ref.startswith(
            "agent-skill-aggregate-evaluation:"
        )
        assert close_operation.task_evaluation is not None
        assert close_operation.task_evaluation[
            "aggregate_evaluation_receipt_ref"
        ] == close_operation.aggregate_evaluation_receipt_ref
        closed_fraction = float(
            close_operation.execution_result["final_close_fraction"]
        )
        assert closed_fraction > 0.5, [
            (fact.event_type, dict(fact.payload))
            for fact in trace_facts
            if fact.event_type
            in {
                "component_execution_target",
                "dispatch_receipt",
                "measurement_alignment",
                "shared_safety_result",
            }
        ]

        state_request = _request(
            session=session,
            adapter=adapter,
            skill_id="state.read_gripper_state",
            parameters={
                "component_ids": ["right_gripper"],
                "max_measurement_age_ms": 500,
            },
            sequence=1,
        )
        state_admission = adapter.submit_skill_request(state_request.to_wire())
        assert state_admission["accepted"]
        assert session.gripper_state_executor.wait_all(timeout_sec=5.0)
        state_operation = session.service.operation_store.get(
            state_admission["operation"]["operation_id"]
        )
        assert state_operation is not None
        assert state_operation.status == "completed"
        assert state_operation.execution_result is not None
        assert float(
            state_operation.execution_result[
                "close_fraction_by_component"
            ]["right_gripper"]
        ) == pytest.approx(closed_fraction, abs=0.02)

        open_request = _request(
            session=session,
            adapter=adapter,
            skill_id="motion.open_gripper",
            parameters={"component_id": "right_gripper"},
            sequence=2,
        )
        open_admission = adapter.submit_skill_request(open_request.to_wire())
        assert open_admission["accepted"]
        assert session.wait(timeout_sec=15.0)
        open_operation = session.service.operation_store.get(
            open_admission["operation"]["operation_id"]
        )
        assert open_operation is not None
        assert open_operation.status == "completed", (
            open_operation.reason_code,
            dict(open_operation.progress_summary),
        )
        assert open_operation.task_outcome == "succeeded"
        assert open_operation.execution_result is not None
        assert float(open_operation.execution_result["final_close_fraction"]) <= (
            session.contract_bundle.gripper_state.profile.open_endpoint_tolerance
        )
        final_arm = session.application.capture_tcp_pose_by_component()["right_arm"]
        assert final_arm == pytest.approx(initial_arm, abs=2e-3)
    finally:
        session.close()


@pytest.mark.parametrize(
    ("stop", "status", "reason", "attempts"),
    (
        ("cancel", "not_due", "gripper_action_cancelled", 1),
        ("deadline", "not_due", "gripper_action_deadline_exceeded", 1),
        ("budget", "not_due", "gripper_action_control_cycles_exhausted", 3),
        ("fault", "no_action", "gripper_action_authoritative_step_not_dispatched", 1),
    ),
)
def test_gripper_pending_ticks_remain_bounded_and_never_prove_completion(
    stop: str, status: str, reason: str, attempts: int
) -> None:
    # Isolate the run loop from activation/physics; the tests above exercise
    # the actual constructor, dispatch receipts, measurements and lower HOLD.
    lifecycle = object.__new__(MujocoGripperActionSkillLifecycle)
    lifecycle.request = SimpleNamespace(
        skill_id="motion.close_gripper", semantic_content_hash="request:pending"
    )
    lifecycle.operation = SimpleNamespace(
        operation_id="operation:pending",
        deadline_at=0.01 if stop == "deadline" else 1.0,
    )
    lifecycle.registration = SimpleNamespace(
        profile=SimpleNamespace(
            max_control_cycles=3,
            physics_ticks_per_control_cycle=6,
            open_endpoint_tolerance=0.05,
        )
    )
    step = SimpleNamespace(
        proposal_admission=SimpleNamespace(accepted=True),
        authoritative_step=SimpleNamespace(status=status),
    )
    lifecycle.application = SimpleNamespace(
        runtime=SimpleNamespace(config=SimpleNamespace(decision_hz=20.0)),
        submit_and_step=Mock(return_value=step),
    )
    lifecycle._initial = {"close_fraction": 0.0}
    lifecycle._action = object()
    lifecycle._execution_refs = {"activation:pending"}
    lifecycle._first_step_refs = lifecycle._latest_step_refs = ()
    lifecycle._capture = Mock(side_effect=AssertionError("No dispatch to measure"))
    lifecycle.request_hold = Mock(
        return_value=SimpleNamespace(lower_hold_confirmed=True)
    )
    now = 0.0
    waits = []

    def sleep(seconds):
        nonlocal now
        waits.append(seconds)
        now += seconds

    lifecycle.clock = lambda: now
    lifecycle.sleep = sleep
    result = lifecycle.run(
        request=lifecycle.request,
        operation=lifecycle.operation,
        cancel_requested=lambda: stop == "cancel" and bool(waits),
    )
    assert not result.succeeded
    assert result.reason_code == reason
    assert result.terminal_hold_confirmed
    assert result.execution_result is None
    assert result.evaluation is None
    assert lifecycle.application.submit_and_step.call_count == attempts
    lifecycle._capture.assert_not_called()
    lifecycle.request_hold.assert_called_once_with(
        operation=lifecycle.operation, reason=reason
    )
    expected_waits = [0.05] * attempts if status == "not_due" else []
    assert waits == ([0.01] if stop == "deadline" else expected_waits)


@pytest.mark.integration
def test_gripper_action_fails_closed_on_invalid_lower_evidence(
    monkeypatch,
) -> None:
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=TASK_CONFIG,
            realtime_worker=False,
            runtime_session_id="test:mujoco-gripper-fail-closed",
            control_domain_id="test:mujoco-gripper-fail-closed",
            execution_observation_plan=(
                mujoco_agent_tcp_pose_execution_observation_plan_v1()
            ),
            exposed_pose_sides=("right",),
            allow_scripted_task_reset=False,
        ),
        allowed_client_ids=("test:mujoco-gripper",),
    )
    try:
        session.application.start_observation_ready()
        adapter = session.wire_adapter(
            authenticated_client_id="test:mujoco-gripper"
        )
        original_submit = session.application.submit_and_step

        cases = (
            (
                SimpleNamespace(
                    proposal_admission=SimpleNamespace(
                        accepted=False,
                        rejection_code="test_rejected",
                        proposal_id="proposal:rejected",
                        semantic_content_hash="proposal-hash:rejected",
                    ),
                    authoritative_step=None,
                ),
                "gripper_action_proposal_rejected:test_rejected",
            ),
            (
                SimpleNamespace(
                    proposal_admission=SimpleNamespace(
                        accepted=True,
                        rejection_code=None,
                        proposal_id="proposal:no-authoritative",
                        semantic_content_hash="proposal-hash:no-authoritative",
                    ),
                    authoritative_step=None,
                ),
                "gripper_action_authoritative_step_missing",
            ),
            (
                SimpleNamespace(
                    proposal_admission=SimpleNamespace(
                        accepted=True,
                        rejection_code=None,
                        proposal_id="proposal:no-receipt",
                        semantic_content_hash="proposal-hash:no-receipt",
                    ),
                    authoritative_step=SimpleNamespace(
                        status="dispatched",
                        trace_facts=(),
                        target_execution_request=None,
                    ),
                ),
                "gripper_action_dispatch_receipt_missing",
            ),
        )
        for sequence, (step, expected_reason) in enumerate(cases):
            monkeypatch.setattr(
                session.application,
                "submit_and_step",
                lambda **_kwargs: step,
            )
            operation = _submit_and_wait(
                session=session,
                adapter=adapter,
                skill_id="motion.close_gripper",
                parameters={"component_id": "right_gripper"},
                sequence=sequence,
            )
            assert operation.status == "failed"
            assert operation.reason_code == expected_reason
            assert operation.execution_receipt_refs
            assert operation.task_outcome == "not_assessed"

        close_samples = iter((0.0, 0.5, 0.7, 0.7, 0.7, 0.7, 0.85))

        def capture_gripper_state():
            close = next(close_samples)
            return {
                "right_gripper": {
                    "qpos": (close,),
                    "close_fraction": close,
                    "measured_at": session.application.runtime.clock(),
                    "state_source": "mujoco_measured_joint_state",
                }
            }

        dispatch_index = 0

        def dispatched_step(**_kwargs):
            nonlocal dispatch_index
            dispatch_index += 1
            return SimpleNamespace(
                proposal_admission=SimpleNamespace(
                    accepted=True,
                    rejection_code=None,
                    proposal_id=f"proposal:drift:{dispatch_index}",
                    semantic_content_hash=f"proposal-hash:drift:{dispatch_index}",
                ),
                authoritative_step=SimpleNamespace(
                    status="dispatched",
                    trace_facts=(
                        SimpleNamespace(
                            event_type="dispatch_receipt",
                            fact_id=f"dispatch-receipt:drift:{dispatch_index}",
                        ),
                    ),
                    target_execution_request=SimpleNamespace(
                        request_id=f"target-request:drift:{dispatch_index}"
                    ),
                ),
            )

        monkeypatch.setattr(
            session.application,
            "capture_gripper_state_by_component",
            capture_gripper_state,
        )
        monkeypatch.setattr(
            session.application,
            "submit_and_step",
            dispatched_step,
        )
        drift_operation = _submit_and_wait(
            session=session,
            adapter=adapter,
            skill_id="motion.close_gripper",
            parameters={"component_id": "right_gripper"},
            sequence=len(cases),
        )
        assert drift_operation.status == "failed"
        assert drift_operation.reason_code == "gripper_close_post_hold_not_settled"
        assert drift_operation.execution_receipt_refs
        assert drift_operation.task_outcome == "not_assessed"

        wrong_direction_reads = 0

        def capture_wrong_direction_gripper_state():
            nonlocal wrong_direction_reads
            close = 0.8 if wrong_direction_reads == 0 else 0.4
            wrong_direction_reads += 1
            return {
                "right_gripper": {
                    "qpos": (close,),
                    "close_fraction": close,
                    "measured_at": session.application.runtime.clock(),
                    "state_source": "mujoco_measured_joint_state",
                }
            }

        monkeypatch.setattr(
            session.application,
            "capture_gripper_state_by_component",
            capture_wrong_direction_gripper_state,
        )
        wrong_direction_operation = _submit_and_wait(
            session=session,
            adapter=adapter,
            skill_id="motion.close_gripper",
            parameters={"component_id": "right_gripper"},
            sequence=len(cases) + 1,
        )
        assert wrong_direction_operation.status == "failed"
        assert (
            wrong_direction_operation.reason_code
            == "gripper_action_control_cycles_exhausted"
        )
        assert wrong_direction_operation.task_outcome == "not_assessed"

        reversed_after_effect_reads = 0

        def capture_reversed_after_effect_gripper_state():
            nonlocal reversed_after_effect_reads
            if reversed_after_effect_reads == 0:
                close = 0.1
            elif reversed_after_effect_reads == 1:
                close = 0.6
            else:
                close = 0.1
            reversed_after_effect_reads += 1
            return {
                "right_gripper": {
                    "qpos": (close,),
                    "close_fraction": close,
                    "measured_at": session.application.runtime.clock(),
                    "state_source": "mujoco_measured_joint_state",
                }
            }

        monkeypatch.setattr(
            session.application,
            "capture_gripper_state_by_component",
            capture_reversed_after_effect_gripper_state,
        )
        reversed_operation = _submit_and_wait(
            session=session,
            adapter=adapter,
            skill_id="motion.close_gripper",
            parameters={"component_id": "right_gripper"},
            sequence=len(cases) + 2,
        )
        assert reversed_operation.status == "failed"
        assert (
            reversed_operation.reason_code
            == "gripper_action_control_cycles_exhausted"
        )
        assert reversed_operation.task_outcome == "not_assessed"

        monkeypatch.setattr(
            session.application,
            "submit_and_step",
            original_submit,
        )
    finally:
        session.close()
