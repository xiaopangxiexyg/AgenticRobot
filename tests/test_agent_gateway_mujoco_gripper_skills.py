from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.mujoco.motion_factory import (
    build_mujoco_agent_tcp_pose_gateway_session,
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
)
from agent_gateway.protocol.contracts import AgentSkillRequest
from sim.mujoco.deploy import MujocoSimHILTargetConfig


TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_reach_fixed_pose_xyz_proprio_v1.json"
)


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
def test_gripper_close_state_open_round_trip_on_real_mujoco() -> None:
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
