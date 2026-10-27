"""Real no-model target wiring for lower checkpoint components, not full recovery."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_motion.action import BIMANUAL_EE_POSE_V1, ComponentAction, RobotAction
from robot_motion.action.admission_checkpoint import (
    capture_source_admission,
    restore_source_admission,
)
from robot_motion.action.authoritative_checkpoint import capture_authoritative_control
from robot_motion.action.checkpoint_codec import (
    decode_control_checkpoint,
    encode_control_checkpoint,
)
from robot_motion.action.decision_checkpoint import (
    capture_decision_owners,
    restore_decision_owners,
)
from robot_motion.deploy.schema import pose_matrix_to_xyzquat
from robot_motion.runtime.execution_checkpoint import (
    capture_execution_authority,
    restore_execution_authority,
)
from sim.mujoco.adapters.checkpoint import (
    ManualExecutionClock,
    capture_execution_rig,
    restore_execution_rig,
)
from sim.mujoco.deploy import agent_skill_target
from sim.mujoco.deploy.component_action_profile import (
    MujocoComponentActionRobotActionV1Materializer,
    build_mujoco_component_execution_binding,
    build_mujoco_pose_component_target_profile,
)
from sim.mujoco.deploy.target_config import MujocoSimHILTargetConfig
from sim.mujoco.deploy.target_measurement import (
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
)
from sim.mujoco.deploy.target_preparation import prepare_mujoco_sim_hil_target


def _configuration():
    return MujocoSimHILTargetConfig(
        task_config_path=Path(
            "sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"
        ),
        realtime_worker=False,
        allow_scripted_task_reset=False,
        exposed_pose_sides=("right",),
    )


@pytest.mark.unit
@pytest.mark.parametrize("clock", [None, 5])
def test_invalid_clock_is_rejected_before_preparing_any_resource(monkeypatch, clock):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Invalid clock cannot acquire native resources.")

    monkeypatch.setattr(agent_skill_target, "prepare_mujoco_sim_hil_target", forbidden)
    with pytest.raises(TypeError, match="clock must be callable"):
        agent_skill_target.build_mujoco_agent_skill_motion_target(
            config=_configuration(), component_execution_binding=None, clock=clock
        )


def _native_target(
    *, observation_plan=None, observation_sink=None, envelope_factory=None,
    prune_expired_proposals=False,
):
    """Build the original fixture once per test, without starting its owners."""
    config = replace(_configuration(), execution_observation_plan=observation_plan)
    prepared = prepare_mujoco_sim_hil_target(config)
    scene = prepared.scene
    profile = build_mujoco_pose_component_target_profile(
        scene,
        profile_id="test.checkpoint.single-arm-pose.v1",
        arm_side_by_component={"right_arm": "right"},
        arm_reference_frame_by_component={"right_arm": "right_arm_base"},
        arm_controlled_frame_by_component={"right_arm": "tool_tcp"},
        selection_group_by_component={"right_arm": "arms"},
    )
    materializer = MujocoComponentActionRobotActionV1Materializer(
        scene,
        profile,
        hand_action_spec_by_component={},
        support_arm_sides=("left",),
        support_pose_contract_id="test.checkpoint.measured-left-support.v1",
        support_arm_pose_provider=lambda: {
            "left": capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                scene, prepared.task_config
            )["left"]
        },
    )
    clock = ManualExecutionClock(10.0)
    target = agent_skill_target.build_mujoco_agent_skill_motion_target(
        config=config,
        prepared_target=prepared,
        component_execution_binding=build_mujoco_component_execution_binding(
            profile, materializer
        ),
        clock=clock,
        execution_observation_sink=observation_sink,
        prune_expired_proposals=prune_expired_proposals,
        tcp_pose_envelope_runtime=(
            None if envelope_factory is None else envelope_factory(prepared, config)
        ),
    )
    return config, prepared, profile, clock, target


@pytest.mark.integration
def test_explicit_pruning_target_keeps_manual_checkpoint_rejection():
    from sim.mujoco.deploy.runtime_checkpoint import capture_runtime

    _config, _prepared, _profile, _clock, target = _native_target(
        prune_expired_proposals=True)
    runtime = target.runtime
    try:
        runtime.start()
        assert runtime.config.prune_expired_proposals is True
        with pytest.raises(ValueError, match="direct manual immediate profile"):
            capture_runtime(runtime)
    finally:
        runtime.close()


@pytest.mark.integration
@pytest.mark.slow_sim
def test_native_no_model_target_components_share_an_exact_clock_and_history(
    monkeypatch,
):
    """A caller-fenced cut; restored components are never published as a target."""
    config, prepared, profile, clock, target = _native_target()
    scene = prepared.scene
    runtime, rig = target.runtime, target.runtime.rig
    core = runtime.core
    restored_rig = None
    try:
        assert runtime.config.prune_expired_proposals is False
        runtime.start()
        assert runtime.clock is rig._clock is core.clock is clock
        assert core.coordinator._clock is core.scheduler._clock is clock
        for _ in range(17):
            rig.step_physics()
        clock.advance_to(10.1)
        runtime.step()
        assert core.scheduler._tick_index == 1
        assert core.producer_epochs.current_epoch(config.producer_id) == 1
        assert core.producer_epochs.current_epoch(config.agent_skill_producer_id) == 1

        poses = capture_mujoco_sim_hil_tool_tcp_pose_by_side(
            scene, prepared.task_config
        )
        action = ComponentAction(
            action_schema=profile.action_schema,
            values_by_component={"right_arm": pose_matrix_to_xyzquat(poses["right"])},
        )
        receipt = runtime.submit_component_action(
            action, source_event_id="test:pending-component"
        )
        assert receipt.accepted

        # Observe the real safety anchor without replacing its computation.
        anchor_times = []
        capture_anchor = agent_skill_target.capture_mujoco_sim_hil_safety_anchor

        def observe_anchor(*args, **kwargs):
            anchor_times.append(kwargs["now"])
            return capture_anchor(*args, **kwargs)

        monkeypatch.setattr(
            agent_skill_target, "capture_mujoco_sim_hil_safety_anchor", observe_anchor
        )
        core.shared_safety(
            RobotAction(
                action_space=BIMANUAL_EE_POSE_V1,
                arm_pose_by_side={
                    side: pose_matrix_to_xyzquat(pose) for side, pose in poses.items()
                },
                source="test:checkpoint-anchor",
            ),
            "test:checkpoint-anchor",
        )
        assert anchor_times == [clock()]

        # No worker/admission thread is running. This test owns the whole cut.
        source = capture_source_admission(
            core.coordinator, core.producer_epochs, core.proposals
        )
        execution = capture_execution_authority(
            core.dispatch_owners,
            core.execution_epochs,
            core.cancellation_acknowledgements,
        )
        decision = capture_decision_owners(core.scheduler, core.arbitration_runtime)
        budget = 128 * 1024 * 1024
        physics = capture_execution_rig(rig, max_native_bytes=budget)
        assert source.captured_at == execution.captured_at == decision.captured_at
        assert decision.captured_at == physics.clock_value
        restored_rig = restore_execution_rig(
            physics,
            config=rig.config,
            load_result=scene.load_result,
            expected_model_sha256=physics.native.model_sha256,
            max_native_bytes=budget,
        )
        branch_clock = restored_rig._clock
        branch_source = restore_source_admission(
            source,
            control_domain_id=core.config.control_domain_id,
            clock_domain=core.config.clock_domain,
            proposal_capacity=core.config.proposal_capacity,
            clock=branch_clock,
        )
        branch_execution = restore_execution_authority(
            execution,
            runtime_session_id=core.config.runtime_session_id,
            control_domain_id=core.config.control_domain_id,
            clock_domain_id=core.config.clock_domain.clock_domain_id,
            owner_runtime_id=core.config.owner_runtime_id,
            owner_kind="shared_arbitration_runtime",
            clock=branch_clock,
        )
        branch_decision = restore_decision_owners(
            decision, spec=core.config.decision_schedule, clock=branch_clock
        )
        assert branch_clock is not clock
        assert branch_source.coordinator._clock is branch_clock
        assert branch_execution.dispatch_owners._clock is branch_clock
        assert branch_decision.scheduler._clock is branch_clock
        restored_source = capture_source_admission(
            branch_source.coordinator,
            branch_source.producer_epochs,
            branch_source.proposals,
        )
        assert replace(restored_source, proposal_records=()) == replace(
            source, proposal_records=()
        )
        assert tuple(
            r.semantic_content_hash for r in restored_source.proposal_records
        ) == tuple(r.semantic_content_hash for r in source.proposal_records)
        assert (
            capture_execution_authority(
                branch_execution.dispatch_owners,
                branch_execution.execution_epochs,
                branch_execution.cancellation_acknowledgements,
            )
            == execution
        )
        assert (
            capture_decision_owners(
                branch_decision.scheduler, branch_decision.publication
            )
            == decision
        )
        assert branch_source.proposals.get(receipt.proposal_id) is not None
        for name in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
            np.testing.assert_array_equal(
                getattr(restored_rig.scene.data, name), getattr(scene.data, name)
            )
        # Exercise the complete core cut with a real nonzero Cartesian command
        # through the existing target, including IK, dispatch and measurement.
        clock.advance_to(10.2)
        moved = pose_matrix_to_xyzquat(
            capture_mujoco_sim_hil_tool_tcp_pose_by_side(scene, prepared.task_config)[
                "right"
            ]
        )
        moved[0] += 0.005
        assert runtime.submit_component_action(
            ComponentAction(
                action_schema=profile.action_schema,
                values_by_component={"right_arm": moved},
            ),
            source_event_id="test:moved-target",
        ).accepted
        executed = runtime.step(physics_ticks_after_dispatch=30)
        assert executed.status == "dispatched"
        clock.advance_to(10.3)
        full_core = capture_authoritative_control(core)
        assert full_core.state["_active_segment_by_group"]
        encoded = encode_control_checkpoint(full_core)
        decoded = decode_control_checkpoint(encoded, max_bytes=16 * 1024 * 1024)
        assert encode_control_checkpoint(decoded) == encoded
        assert decoded.state["_previous_decision_id"] == executed.decision.decision_id
        # These components are not published as a restored target. Its IK,
        # observation and lifecycle closure remain a separate integration.
    finally:
        if restored_rig is not None:
            restored_rig.close()
        runtime.close()
