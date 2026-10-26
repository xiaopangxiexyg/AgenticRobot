"""Independent native target continuation through the original factory."""

import threading
from dataclasses import replace

import numpy as np
import pytest

from robot_motion.action import ComponentAction
from robot_motion.deploy.schema import pose_matrix_to_xyzquat
from robot_motion.rm65.retargeter_checkpoint import capture_retargeter
from robot_motion.target_safety import TCPPoseEnvelopeSafetyRuntime
from robot_motion.target_safety.checkpoint import capture_safety_envelope
from sim.mujoco.adapters.checkpoint import capture_execution_rig
from sim.mujoco.deploy.component_action_profile import (
    MujocoComponentActionRobotActionV1Materializer,
    build_mujoco_component_execution_binding,
)
from sim.mujoco.deploy.pose_resolution import MujocoCoupledPoseToQposResolver
from sim.mujoco.deploy.runtime_checkpoint import capture_runtime
from sim.mujoco.deploy.target_checkpoint import (
    capture_motion_target,
    restore_motion_target,
)
from sim.mujoco.deploy.target_measurement import (
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
)
from sim.mujoco.ik import RM65RetargeterIK
from tests.test_mujoco_runtime_checkpoint import target_source  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.slow_sim]
BUDGET = 128 * 1024 * 1024


@pytest.fixture(params=(False, True))
def envelope_source(request):
    from agent_gateway.profiles.mujoco.motion_factory import (
        _build_standard_tcp_pose_envelope_runtime,
        mujoco_agent_tcp_pose_execution_observation_plan_v1,
    )
    from sim.mujoco.deploy.execution_difference import MujocoExecutionDifferenceObserver
    from tests.test_mujoco_agent_control_checkpoint import _native_target

    observation = request.param
    values = _native_target(
        observation_plan=(
            mujoco_agent_tcp_pose_execution_observation_plan_v1(motion_side="right")
            if observation
            else None
        ),
        observation_sink=(
            MujocoExecutionDifferenceObserver(
                selection_group_id="arms", component_ids=("right_arm",)
            )
            if observation
            else None
        ),
        envelope_factory=lambda prepared, config: (
            _build_standard_tcp_pose_envelope_runtime(
                prepared=prepared, target_config=config
            )
        ),
    )
    _config, prepared, _profile, clock, target = values
    runtime = target.runtime
    runtime.start()
    runtime.transition_execution_gate("hold", reason="envelope-checkpoint")
    envelope = target.tcp_pose_envelope_runtime
    poses = capture_mujoco_sim_hil_tool_tcp_pose_by_side(
        target.scene, prepared.task_config
    )
    envelope.commit_reference(
        reference_id="original-qpos-initialization",
        tcp_pose_by_side={"right": poses["right"]},
        captured_at_by_side={"right": clock()},
        source_event_ids=("completed-original-initialization",),
        lower_state="hold",
    )
    clock.advance_to(clock() + 0.2)
    try:
        yield values
    finally:
        runtime.close()


def test_inactive_envelope_restores_historical_reference_with_enabled_observation(
    envelope_source, monkeypatch
):
    _config, prepared, profile, clock, target = envelope_source
    envelope = target.tcp_pose_envelope_runtime
    saved = capture_motion_target(target, prepared=prepared, max_native_bytes=BUDGET)

    def no_lifecycle(*args, **kwargs):
        raise AssertionError(
            "Restoring a reference must not manufacture initialization."
        )

    with monkeypatch.context() as patch:
        for name in (
            "commit_reference",
            "invalidate_reference",
            "activate",
            "deactivate_after_hold",
        ):
            patch.setattr(TCPPoseEnvelopeSafetyRuntime, name, no_lifecycle)
        branch = _restore(saved, prepared, target, profile)
    try:
        fresh = branch.tcp_pose_envelope_runtime
        assert fresh is not envelope
        assert branch.runtime.pose_resolver.pose_target_safety_filter.runtime is fresh
        assert capture_safety_envelope(fresh, captured_at=clock()).content_hash == (
            saved.safety_envelope.content_hash
        )
        assert fresh._reference.captured_at_by_side["right"] < clock()
        assert (
            capture_runtime(branch.runtime).content_hash == saved.runtime.content_hash
        )
        entry = fresh.catalog_entries[0]
        raw = {"right": fresh._reference.tcp_pose_by_side["right"]}
        assert fresh.preview_for_envelope(
            raw, deployment_envelope_hash=entry.semantic_content_hash
        ).semantic_content_hash == (
            envelope.preview_for_envelope(
                raw, deployment_envelope_hash=entry.semantic_content_hash
            ).semantic_content_hash
        )
        assert fresh._active is None
        assert (
            capture_motion_target(
                target, prepared=prepared, max_native_bytes=BUDGET
            ).safety_envelope.content_hash
            == saved.safety_envelope.content_hash
        )
        branch.runtime.pose_resolver.pose_target_safety_filter.runtime = envelope
        with pytest.raises(ValueError, match="envelope"):
            capture_motion_target(
                branch,
                prepared=replace(prepared, scene=branch.scene),
                max_native_bytes=BUDGET,
            )
    finally:
        branch.runtime.close()


def test_active_envelope_is_not_a_gateway_operation_restore(envelope_source):
    from robot_motion.target_safety import TCPPoseEnvelopeSelection

    _config, prepared, _profile, _clock, target = envelope_source
    envelope = target.tcp_pose_envelope_runtime
    entry = envelope.catalog_entries[0]
    envelope.activate(
        selection=TCPPoseEnvelopeSelection(
            safety_profile_id="caller-profile",
            safety_profile_hash="caller-profile-hash",
            deployment_envelope_id=entry.envelope_id,
            deployment_envelope_hash=entry.semantic_content_hash,
            constraint_set_hash=entry.constraint_set_hash,
        ),
        operation_id="not-yet-restorable-application-operation",
        route_id=entry.allowed_route_ids[0],
        execution_epoch=7,
        source_event_ids=("application-ready",),
        lower_state="hold",
    )
    before = envelope.snapshot()
    with pytest.raises(ValueError, match="active envelope"):
        capture_motion_target(target, prepared=prepared, max_native_bytes=BUDGET)
    assert envelope.snapshot() == before


@pytest.mark.parametrize("damage", ("digest", "active"))
def test_bad_envelope_restore_is_rejected_before_native_allocation(
    envelope_source, monkeypatch, damage
):
    from robot_motion.target_safety import TCPPoseEnvelopeSelection
    from sim.mujoco.deploy import target_checkpoint

    _config, prepared, profile, clock, target = envelope_source
    saved = capture_motion_target(target, prepared=prepared, max_native_bytes=BUDGET)
    envelope = target.tcp_pose_envelope_runtime
    if damage == "digest":
        envelope_checkpoint = replace(saved.safety_envelope, content_hash="corrupt")
    else:
        entry = envelope.catalog_entries[0]
        envelope.activate(
            selection=TCPPoseEnvelopeSelection(
                safety_profile_id="caller-profile",
                safety_profile_hash="caller-profile-hash",
                deployment_envelope_id=entry.envelope_id,
                deployment_envelope_hash=entry.semantic_content_hash,
                constraint_set_hash=entry.constraint_set_hash,
            ),
            operation_id="unrestored-application-operation",
            route_id=entry.allowed_route_ids[0],
            execution_epoch=7,
            source_event_ids=("application-ready",),
            lower_state="hold",
        )
        envelope_checkpoint = capture_safety_envelope(envelope, captured_at=clock())

    def no_native_load(*args, **kwargs):
        raise AssertionError("Bad envelope cannot acquire a native branch.")

    before = target.scene.data.__getstate__()
    monkeypatch.setattr(target_checkpoint, "restore_execution_rig", no_native_load)
    with pytest.raises(ValueError):
        _restore(
            replace(saved, safety_envelope=envelope_checkpoint),
            prepared,
            target,
            profile,
        )
    assert target.scene.data.__getstate__() == before


def _binding(prepared, profile):
    materializer = MujocoComponentActionRobotActionV1Materializer(
        prepared.scene,
        profile,
        hand_action_spec_by_component={},
        support_arm_sides=("left",),
        support_pose_contract_id="test.checkpoint.measured-left-support.v1",
        support_arm_pose_provider=lambda: {
            "left": capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                prepared.scene, prepared.task_config
            )["left"]
        },
    )
    return build_mujoco_component_execution_binding(profile, materializer)


def _submit(target, profile, event, ticks=0):
    value = pose_matrix_to_xyzquat(
        capture_mujoco_sim_hil_tool_tcp_pose_by_side(target.scene, target.task_config)[
            "right"
        ]
    )
    value[0] += 0.005
    admission = target.runtime.submit_component_action(
        ComponentAction(
            action_schema=profile.action_schema,
            values_by_component={"right_arm": value},
        ),
        source_event_id=event,
    )
    assert admission.accepted
    return target.runtime.step(physics_ticks_after_dispatch=ticks)


def _restore(checkpoint, prepared, target, profile, **kwargs):
    return restore_motion_target(
        checkpoint,
        prepared=prepared,
        rig_config=target.runtime.rig.config,
        expected_model_sha256=checkpoint.rig.native.model_sha256,
        binding_factory=lambda fresh: _binding(fresh, profile),
        max_native_bytes=BUDGET,
        **kwargs,
    )


@pytest.mark.parametrize("ticks", (0, 30))
def test_fresh_target_retains_pending_and_active_then_continues(
    source, monkeypatch, ticks
):
    _config, prepared, profile, clock, target = source
    runtime = target.runtime
    runtime.start()
    runtime.capture_policy_observation()
    assert _submit(target, profile, "first", ticks).status == "dispatched"
    saved = capture_motion_target(target, prepared=prepared, max_native_bytes=BUDGET)

    def no_reset(*_args, **_kwargs):
        raise AssertionError("Restoration must not reset the IK adapter.")

    monkeypatch.setattr(RM65RetargeterIK, "reset_from_scene", no_reset)
    branch = _restore(saved, prepared, target, profile)
    try:
        assert branch.scene is not target.scene
        assert branch.runtime.clock is branch.runtime.rig._clock
        assert branch.runtime.core.clock is branch.runtime.clock
        assert (
            capture_runtime(branch.runtime).content_hash == saved.runtime.content_hash
        )
        assert (
            capture_retargeter(
                branch.runtime.pose_resolver.ik.retargeter, max_native_bytes=BUDGET
            ).content_hash
            == saved.retargeter.content_hash
        )
        native = capture_execution_rig(
            branch.runtime.rig, max_native_bytes=BUDGET
        ).native
        # Official binary restoration clears compile-only signatures; this
        # is the already-tested native contract, not a runtime state exemption.
        assert native.source_compilation_signatures == (0, 0)
        assert (
            replace(
                native,
                source_compilation_signatures=saved.rig.native.source_compilation_signatures,
            )
            == saved.rig.native
        )
        before = len(branch.runtime.core.trace_facts)
        branch.runtime._sync_reference_trace()
        assert len(branch.runtime.core.trace_facts) == before
        assert (
            branch.runtime.observation_adapter
            is branch.runtime._observation_adapter_by_node[
                runtime.config.policy_node_id
            ]
        )
        assert branch.runtime.observation_adapter._sequence == 1
        for owner in (runtime, branch.runtime):
            for _ in range(10):
                owner.rig.step_physics()
        np.testing.assert_array_equal(target.scene.data.qpos, branch.scene.data.qpos)
        np.testing.assert_array_equal(target.scene.data.qvel, branch.scene.data.qvel)
        # The ordinary wall-clock adaptive solver remains enabled. New solver
        # telemetry is not asserted bit-identical across sequential branches.
        source_native = target.scene.data.__getstate__()
        source_ik_hash = capture_retargeter(
            runtime.pose_resolver.ik.retargeter, max_native_bytes=BUDGET
        ).content_hash
        branch.runtime.clock.advance_to(clock() + 0.1)
        assert _submit(branch, profile, "second", 10).status == "dispatched"
        assert branch.runtime._next_direct_component_proposal_sequence == 2
        assert runtime._next_direct_component_proposal_sequence == 1
        assert capture_runtime(runtime).content_hash == saved.runtime.content_hash
        assert target.scene.data.__getstate__() == source_native
        assert (
            capture_retargeter(
                runtime.pose_resolver.ik.retargeter, max_native_bytes=BUDGET
            ).content_hash
            == source_ik_hash
        )
    finally:
        branch.runtime.close()


def test_bad_binding_does_not_mutate_or_close_source(source):
    _config, prepared, _profile, _clock, target = source
    target.runtime.start()
    saved = capture_motion_target(target, prepared=prepared, max_native_bytes=BUDGET)
    with pytest.raises(ValueError, match="binding"):
        restore_motion_target(
            saved,
            prepared=prepared,
            rig_config=target.runtime.rig.config,
            expected_model_sha256=saved.rig.native.model_sha256,
            binding_factory=lambda _fresh: target.runtime.component_execution_binding,
            max_native_bytes=BUDGET,
        )
    assert capture_runtime(target.runtime).content_hash == saved.runtime.content_hash
    assert target.runtime.rig._started


def test_unknown_resolver_owner_and_changed_diagnostics_reject(source):
    _config, prepared, profile, _clock, target = source
    target.runtime.start()
    saved = capture_motion_target(target, prepared=prepared, max_native_bytes=BUDGET)
    with pytest.raises(ValueError, match="diagnostics"):
        _restore(
            replace(saved, resolver_diagnostics={"changed": True}),
            prepared,
            target,
            profile,
        )
    target.runtime.pose_resolver.foreign_owner = object()
    with pytest.raises(ValueError, match="resolver"):
        capture_motion_target(target, prepared=prepared, max_native_bytes=BUDGET)


@pytest.mark.parametrize("ending", ("pending", "expired", "hold"))
def test_last_request_active_reference_and_pending_reference_are_distinct(
    source, monkeypatch, ending
):
    _config, prepared, profile, clock, target = source
    runtime = target.runtime
    runtime.start()
    assert _submit(target, profile, "active-a", 30).status == "dispatched"
    first = runtime._latest_target_execution_step
    old_reference = runtime.rig._last_reference
    clock.advance_to(10.1)
    if ending == "hold":
        runtime.transition_execution_gate("hold", reason="checkpoint-test")
        assert runtime.rig._last_reference is None
        assert runtime._immediate_target_by_reference_sequence
    elif ending == "expired":
        original = MujocoCoupledPoseToQposResolver.resolve

        def delayed(owner, *args, **kwargs):
            command = original(owner, *args, **kwargs)
            clock.advance_to(clock() + 0.3)
            return command

        with monkeypatch.context() as patch:
            patch.setattr(MujocoCoupledPoseToQposResolver, "resolve", delayed)
            result = _submit(target, profile, "rejected-b")
        assert result.status == "target_lifecycle_rejected_keep_existing"
        assert runtime.rig._last_reference is old_reference
        assert (
            runtime._last_dispatch_context.target_execution_request.request_id
            != first.target_execution_request.request_id
        )
    else:
        assert _submit(target, profile, "pending-b").status == "dispatched"
        assert runtime.rig._last_reference is old_reference
        assert runtime.rig._pending_reference.sequence != old_reference.sequence
    saved = capture_motion_target(target, prepared=prepared, max_native_bytes=BUDGET)
    branch = _restore(saved, prepared, target, profile)
    try:
        assert (
            capture_runtime(branch.runtime).content_hash == saved.runtime.content_hash
        )
        assert branch.runtime.capture_target_execution_status_snapshot(
            now=clock()
        ) == runtime.capture_target_execution_status_snapshot(now=clock())
        for owner in (runtime, branch.runtime):
            owner.rig.step_physics()
        np.testing.assert_array_equal(target.scene.data.qpos, branch.scene.data.qpos)
        if ending == "expired":
            # A genuine old request cannot replace the latest rejected one.
            runtime._latest_target_execution_step = first
            with pytest.raises(ValueError, match="last request"):
                capture_runtime(runtime)
    finally:
        branch.runtime.close()


def test_backend_exception_cut_is_explicitly_unsupported(source, monkeypatch):
    _config, prepared, profile, _clock, target = source
    target.runtime.start()

    def failed(_command):
        raise RuntimeError("unacknowledged backend")

    with monkeypatch.context() as patch:
        patch.setattr(target.runtime.core, "dispatcher", failed)
        result = _submit(target, profile, "failed-dispatch")
    assert result.target_completion_receipt.outcome_code == "target_dispatch_exception"
    with pytest.raises(ValueError, match="unacknowledged"):
        capture_motion_target(target, prepared=prepared, max_native_bytes=BUDGET)


@pytest.mark.parametrize(
    "field", ("target_frame_profile", "tool_frame_profile", "task_initialization_info")
)
def test_changed_public_metadata_rejects_before_building_a_branch(source, field):
    _config, prepared, profile, _clock, target = source
    target.runtime.start()
    saved = capture_motion_target(target, prepared=prepared, max_native_bytes=BUDGET)
    value = (
        {"changed": True}
        if field == "task_initialization_info"
        else replace(getattr(prepared, field), profile_id="another-profile")
    )
    with pytest.raises(ValueError, match="profile metadata"):
        _restore(saved, replace(prepared, **{field: value}), target, profile)
    assert capture_runtime(target.runtime).content_hash == saved.runtime.content_hash


@pytest.mark.parametrize(
    "field",
    (
        "resolver",
        "dispatcher",
        "contextual_dispatcher",
        "backend_hold",
        "target_execution_port",
        "action_representation_adapter",
        "shared_safety",
    ),
)
def test_changed_core_binding_cannot_silently_revert_to_factory(
    source, monkeypatch, field
):
    _config, prepared, _profile, _clock, target = source
    target.runtime.start()
    with monkeypatch.context() as patch:
        patch.setattr(target.runtime.core, field, object())
        with pytest.raises(ValueError, match="factory|callback"):
            capture_motion_target(target, prepared=prepared, max_native_bytes=BUDGET)


def test_close_requested_during_capture_cannot_publish_a_target_cut(
    source, monkeypatch
):
    from sim.mujoco.deploy import target_checkpoint

    _config, prepared, _profile, _clock, target = source
    target.runtime.start()
    original = target_checkpoint.capture_retargeter
    closer = threading.Thread(target=target.runtime.close, daemon=True)

    def closing(*args, **kwargs):
        state = original(*args, **kwargs)
        closer.start()
        assert target.runtime.rig._stop_event.wait(2)
        return state

    with monkeypatch.context() as patch:
        patch.setattr(target_checkpoint, "capture_retargeter", closing)
        with pytest.raises(ValueError, match="boundary"):
            capture_motion_target(target, prepared=prepared, max_native_bytes=BUDGET)
    closer.join(3)
    assert not closer.is_alive()
    assert not target.runtime._started
