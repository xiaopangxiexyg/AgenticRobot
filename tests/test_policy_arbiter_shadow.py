from __future__ import annotations

import argparse
import inspect
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.action import (
    ActionComponentCatalog,
    ActionComponentSpec,
    ActionArbiter,
    ActionPersistencePolicy,
    ActiveExecutionSnapshot,
    SharedCanonicalSafetyResult,
    build_robot_action_v1_runtime_schema,
    evaluate_shared_execution_shadow,
)
from robot_motion.deploy import (
    ACTION_ARM_POSE_SLICES,
    DEPLOY_ACTION_DIM,
    DEPLOY_ACTION_VECTOR_ACTION_SPACE,
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    build_deployment_observation,
    nan_vector,
)
from teleop_runtime.deploy.arbiter_shadow import (
    PassivePolicyArbiterShadow,
    PolicyArbiterShadowConfig,
)
from teleop_runtime.deploy import run_policy


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class Policy:
    name = "shadow-policy"

    def __init__(self, action: np.ndarray) -> None:
        self.action = np.asarray(action, dtype=np.float32)

    def predict(self, _observation):
        return self.action.copy()


def observation():
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    return build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        step_index=0,
        arm_states={
            "right": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
        },
        hand_states={},
        timestamp=100.0,
    )


def action(x: float = 0.12) -> np.ndarray:
    value = nan_vector(DEPLOY_ACTION_DIM)
    value[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [x, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    return value


def pipeline() -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=False,
                max_arm_xyz_step_m=0.08,
            )
        ),
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
    )


def runtime(clock: Clock) -> PolicyDeployRuntime:
    return PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
        ),
        monotonic_clock=clock,
    )


def _shadow_contract():
    catalog = ActionComponentCatalog(
        catalog_id="policy-shadow-right-arm-v1",
        action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
        components=(
            ActionComponentSpec(
                component_id="right_arm",
                selection_group_id="right_arm",
                effector_name="right_arm",
            ),
        ),
    )
    return catalog, build_robot_action_v1_runtime_schema(
        component_catalog=catalog,
    )


def shadow(clock: Clock, *, capacity: int = 32) -> PassivePolicyArbiterShadow:
    catalog, action_schema = _shadow_contract()
    return PassivePolicyArbiterShadow(
        PolicyArbiterShadowConfig(
            runtime_session_id="shadow-session",
            control_domain_id="robot-0",
            embodiment_instance_id="rm65-pair-0",
            producer_id="policy-main",
            active_effectors=("right_arm",),
            action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
            decision_hz=15.0,
            component_catalog=catalog,
            action_schema=action_schema,
            proposal_capacity=capacity,
        ),
        clock=clock,
    )


def emit(
    deploy_runtime: PolicyDeployRuntime,
    clock: Clock,
    *,
    x: float = 0.12,
):
    obs = observation()
    prediction = deploy_runtime.predict_and_queue(
        Policy(action(x)),
        obs,
        now=clock.value,
        action_keyframe_period_sec=1.0 / 15.0,
    )
    assert prediction.accepted
    step = deploy_runtime.prepare_next_action(
        obs,
        preparer=pipeline(),
        policy_name="shadow-policy",
    )
    assert step.emission_error is None
    assert step.emission is not None
    return obs, step


def test_shadow_consumes_same_emission_without_second_queue_pop() -> None:
    clock = Clock()
    deploy_runtime = runtime(clock)
    coordinator = shadow(clock)
    _, action_step = emit(deploy_runtime, clock)
    pop_count = deploy_runtime.action_queue_pop_count()

    result = coordinator.evaluate_emission(action_step.emission)

    assert result.status == "evaluated"
    assert result.admission_receipt is not None
    assert result.admission_receipt.accepted
    assert result.evaluation is not None
    assert result.evaluation.non_authoritative is True
    assert result.evaluation.mode == "shadow"
    assert result.evaluation.output_status == "selected"
    assert result.evaluation.would_select_action is not None
    assert not hasattr(result.evaluation, "selected_action")
    assert deploy_runtime.action_queue_pop_count() == pop_count == 1
    assert deploy_runtime.action_queue_length() == 0

    legacy = coordinator.observe_legacy_execution(
        emission=action_step.emission,
        shadow_step=result,
        canonical_action=action_step.prepared_action,
        execution=SimpleNamespace(
            accepted=True,
            mode="dry_run",
            dispatch_attempted=False,
            dispatch_completed=False,
        ),
        dispatch_error=None,
        observed_at=clock.value,
    )
    assert legacy.owner_kind == "legacy_control_path"
    assert legacy.component_parity is not None
    assert legacy.component_parity.matched
    assert legacy.dispatch_attempted is False


def test_same_frozen_input_snapshot_replays_to_same_computation() -> None:
    clock = Clock()
    deploy_runtime = runtime(clock)
    coordinator = shadow(clock)
    _, action_step = emit(deploy_runtime, clock)
    step = coordinator.evaluate_emission(action_step.emission)
    assert step.arbitration_input_snapshot is not None
    assert step.computation is not None

    replayed = ActionArbiter().arbitrate(step.arbitration_input_snapshot)

    assert replayed.semantic_content_hash == step.computation.semantic_content_hash
    assert replayed.selected_action_hash == step.computation.selected_action_hash


def test_shared_execution_shadow_stops_at_immutable_command_intent() -> None:
    clock = Clock()
    deploy_runtime = runtime(clock)
    coordinator = shadow(clock)
    _, action_step = emit(deploy_runtime, clock)
    step = coordinator.evaluate_emission(action_step.emission)
    assert step.computation is not None

    evaluated = evaluate_shared_execution_shadow(
        step.computation,
        active_execution=ActiveExecutionSnapshot(
            control_domain_id="robot-0",
            execution_epoch=0,
            active_segment_by_group={},
        ),
        persistence_policy=ActionPersistencePolicy(
            policy_id="policy-shadow-segment-v1",
            selected_segment_max_lifetime_sec=0.1,
        ),
        shared_safety=lambda selected, result_id: SharedCanonicalSafetyResult(
            input_action_id=result_id,
            safe_action=selected,
            status="safe",
        ),
        now=clock.value,
        supports_local_group_stop=False,
        source_arbitration_evaluation_id=step.evaluation.evaluation_id,
    )

    assert evaluated.non_authoritative
    assert evaluated.source_arbitration_evaluation_id == step.evaluation.evaluation_id
    assert evaluated.output_status == "command_intent_ready"
    assert {
        item.directive
        for item in evaluated.proposed_directive.directive_by_selection_group.values()
    } == {"publish_selected"}
    assert evaluated.command_intent is not None
    legacy_command = action_step.prepared_action.command
    assert legacy_command is not None
    np.testing.assert_allclose(
        evaluated.command_intent["commands"]["right_arm"]["value"],
        legacy_command.commands["right_arm"].value,
    )
    with pytest.raises(ValueError):
        evaluated.command_intent["commands"]["right_arm"]["value"][0, 3] = 999.0
    assert not hasattr(evaluated, "dispatch")
    assert not hasattr(evaluated, "dispatch_owner_token")


def test_shared_execution_shadow_hold_never_calls_safety_or_backend() -> None:
    clock = Clock()
    deploy_runtime = runtime(clock)
    coordinator = shadow(clock)
    _, first_action = emit(deploy_runtime, clock)
    first = coordinator.evaluate_emission(first_action.emission)
    assert first.computation is not None

    clock.value += 0.1
    coordinator.update_execution_gate("hold", reason="test hold")
    _, second_action = emit(deploy_runtime, clock, x=0.13)
    second = coordinator.evaluate_emission(second_action.emission)
    assert second.computation is not None

    def unexpected_safety(_selected, _result_id):
        raise AssertionError("hold evaluation must not call shared safety")

    evaluated = evaluate_shared_execution_shadow(
        second.computation,
        active_execution=ActiveExecutionSnapshot(
            control_domain_id="robot-0",
            execution_epoch=0,
            active_segment_by_group={},
        ),
        persistence_policy=ActionPersistencePolicy(
            policy_id="policy-shadow-segment-v1",
            selected_segment_max_lifetime_sec=0.1,
        ),
        shared_safety=unexpected_safety,
        now=clock.value,
        supports_local_group_stop=False,
    )

    assert evaluated.output_status == "proposed_hold"
    assert evaluated.safety_result is None
    assert evaluated.command_intent is None
    assert evaluated.proposed_directive.requires_backend_hold


def test_late_gate_change_only_affects_next_snapshot() -> None:
    clock = Clock()
    deploy_runtime = runtime(clock)
    coordinator = shadow(clock)
    _, first_action = emit(deploy_runtime, clock)
    first = coordinator.evaluate_emission(first_action.emission)
    assert first.arbitration_input_snapshot is not None
    assert first.computation is not None
    first_hash = first.computation.semantic_content_hash

    clock.value += 0.1
    coordinator.update_execution_gate("hold", reason="test hold")
    replayed = ActionArbiter().arbitrate(first.arbitration_input_snapshot)
    assert replayed.semantic_content_hash == first_hash

    _, second_action = emit(deploy_runtime, clock, x=0.13)
    second = coordinator.evaluate_emission(second_action.emission)
    assert second.evaluation is not None
    assert second.evaluation.output_status == "hold"
    assert second.evaluation.would_select_action is None


def test_expired_proposals_are_pruned_without_reusing_sequence_identity() -> None:
    clock = Clock()
    deploy_runtime = runtime(clock)
    coordinator = shadow(clock, capacity=2)

    proposal_ids = []
    for index in range(4):
        clock.value = 10.0 + index * 0.1
        _, action_step = emit(deploy_runtime, clock, x=0.12 + index * 0.01)
        step = coordinator.evaluate_emission(action_step.emission)
        assert step.admission_receipt is not None
        assert step.admission_receipt.accepted
        proposal_ids.append(step.admission_receipt.proposal_id)
        assert len(coordinator.proposal_buffer) <= 1

    assert len(set(proposal_ids)) == 4


def test_invalid_emission_is_status_not_arbitration_decision() -> None:
    clock = Clock()
    deploy_runtime = runtime(clock)
    coordinator = shadow(clock)
    _, action_step = emit(deploy_runtime, clock)
    emission = replace(action_step.emission, pre_safety_robot_action=None)

    step = coordinator.evaluate_emission(emission)

    assert step.status == "emission_rejected"
    assert step.status_event is not None
    assert step.status_event.non_authoritative
    assert step.evaluation is None
    assert step.computation is None


def test_shadow_coordinator_has_no_queue_dispatch_or_owner_capability() -> None:
    source = inspect.getsource(PassivePolicyArbiterShadow)

    assert "pop_action(" not in source
    assert "prepare_next_action(" not in source
    assert "predict_and_queue(" not in source
    assert "DispatchOwner" not in source
    assert "dispatch_teleop_command" not in source
    assert not hasattr(PassivePolicyArbiterShadow, "publish")
    assert not hasattr(PassivePolicyArbiterShadow, "dispatch")


def test_shadow_rate_above_60_is_rejected() -> None:
    catalog, action_schema = _shadow_contract()
    with pytest.raises(ValueError, match="60"):
        PolicyArbiterShadowConfig(
            runtime_session_id="s",
            control_domain_id="d",
            embodiment_instance_id="e",
            producer_id="p",
            active_effectors=("right_arm",),
            action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
            decision_hz=60.1,
            component_catalog=catalog,
            action_schema=action_schema,
        )


def test_deploy_shadow_mode_defaults_off_and_rejects_real_send() -> None:
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    defaults = parser.parse_args([])
    assert defaults.action_arbiter_mode == "off"

    args = parser.parse_args(["--action-arbiter-mode", "shadow", "--send-actions"])
    with pytest.raises(ValueError, match="dry-run/replay only"):
        run_policy.validate_args(
            args,
            raw_argv=("--action-arbiter-mode", "shadow", "--send-actions"),
        )


def test_frontend_passes_monotonic_control_time_and_same_emission_to_shadow() -> None:
    source = inspect.getsource(run_policy.main)

    assert "control_loop_started_at = time.monotonic()" in source
    assert "now=control_loop_started_at" in source
    assert "policy_arbiter_shadow.evaluate_emission(" in source
    assert "action_step.emission" in source
    assert "policy_arbiter_shadow.prepare_next_action" not in source
    assert "policy_arbiter_shadow.pop_action" not in source
    assert "evaluate_policy_legacy_aligned_safety(" in source
    assert "emission=action_step.emission" in source
    assert "legacy_result=canonical_action" in source
    assert '"policy_safety_shadow_error"' in source
