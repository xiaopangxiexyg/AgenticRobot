"""Retain enabled observations without emitting old frames or changing physics."""

from dataclasses import replace

import pytest

from robot_motion.action import (
    ExecutionMeasurementObservation,
    ExecutionObservationPlan,
    ExecutionObservationWatermark,
    ExecutionReferenceFrameObservation,
)
from sim.mujoco.deploy.execution_difference import MujocoExecutionDifferenceObserver
from sim.mujoco.deploy.observation_checkpoint import (
    _copy,
    _same,
    _validate_runtime_observations,
    capture_observer,
    capture_runtime_observations,
    restore_observer,
    restore_runtime_observations,
)
from sim.mujoco.deploy.runtime_checkpoint import capture_runtime
from sim.mujoco.deploy.target_checkpoint import capture_motion_target
from tests.test_mujoco_agent_control_checkpoint import _native_target
from tests.test_mujoco_target_checkpoint import BUDGET, _restore, _submit

pytestmark = [pytest.mark.integration, pytest.mark.slow_sim]


@pytest.fixture
def source():
    plan = ExecutionObservationPlan(
        plan_id="test.native-observation.v1",
        reference_mode="full_frames",
        measurement_mode="full_frames",
        component_ids=("right_arm",),
        required_representation_by_component={"right_arm": "cartesian_pose"},
    )
    sink = MujocoExecutionDifferenceObserver(
        selection_group_id="arms", component_ids=("right_arm",)
    )
    config, prepared, profile, clock, target = _native_target(
        observation_plan=plan, observation_sink=sink
    )
    target.runtime.start()
    try:
        yield config, prepared, profile, clock, target
    finally:
        target.runtime.close()


@pytest.mark.parametrize(
    "ending",
    ("pending", "active", "hold", "watermark_rejected", "measurement_rejected"),
)
def test_enabled_cartesian_cut_restores_without_publish_and_continues(
    source, monkeypatch, ending
):
    _config, prepared, profile, clock, target = source
    runtime = target.runtime
    assert (
        _submit(
            target, profile, "observed-first", 0 if ending == "pending" else 30
        ).status
        == "dispatched"
    )
    clock.advance_to(10.01)
    if ending == "hold":
        runtime.transition_execution_gate("hold", reason="observed-checkpoint")
        runtime.step()
    if ending in ("watermark_rejected", "measurement_rejected"):
        publish = MujocoExecutionDifferenceObserver.publish_execution_observation

        def reject_watermark(owner, observation):
            if isinstance(
                observation,
                ExecutionObservationWatermark
                if ending == "watermark_rejected"
                else ExecutionMeasurementObservation,
            ):
                return False
            return publish(owner, observation)

        with monkeypatch.context() as patch:
            patch.setattr(
                MujocoExecutionDifferenceObserver,
                "publish_execution_observation",
                reject_watermark,
            )
            runtime.step()
        clock.advance_to(10.02)
    old_binding = runtime._execution_observation_binding
    assert old_binding.bound_at < clock()
    native_before = target.scene.data.__getstate__()

    def no_publish(*_args, **_kwargs):
        raise AssertionError("Capture/restore must not publish observations")

    with monkeypatch.context() as patch:
        patch.setattr(
            MujocoExecutionDifferenceObserver,
            "publish_execution_observation",
            no_publish,
        )
        saved = capture_motion_target(
            target, prepared=prepared, max_native_bytes=BUDGET
        )
        branch = _restore(saved, prepared, target, profile)
    if ending == "measurement_rejected":
        assert len(saved.runtime.execution_observation.observer.pending) == 1
    try:
        restored = branch.runtime
        assert _same(old_binding, restored._execution_observation_binding)
        assert capture_runtime(restored).content_hash == saved.runtime.content_hash
        assert (
            restored._execution_observation_sink
            is not runtime._execution_observation_sink
        )
        assert (
            restored._execution_observation_sink._lock
            is not runtime._execution_observation_sink._lock
        )
        assert (
            restored._execution_pose_observation_projector._scratch
            is not runtime._execution_pose_observation_projector._scratch
        )
        assert (
            restored._execution_pose_observation_projector._model is branch.scene.model
        )
        assert target.scene.data.__getstate__() == native_before
        # Before the next decision deadline, the ordinary step still publishes
        # a coherent observation cut without running a fresh adaptive IK solve.
        runtime.step()
        restored.step()
        assert _same(
            capture_runtime_observations(runtime),
            capture_runtime_observations(restored),
        )
        state = capture_runtime_observations(restored)
        if ending == "watermark_rejected":
            assert (
                state.fields["_execution_observation_watermark_by_stream"][
                    "measurement"
                ].status
                == "violated"
            )
        assert restored._execution_observation_sink.snapshot().measurement_count >= 2
        assert target.scene.data.__getstate__() == native_before
    finally:
        branch.runtime.close()


@pytest.mark.parametrize(
    "field", ("sink", "projector", "plan", "live_scratch", "fk_binding")
)
def test_enabled_observation_owner_cannot_be_silently_replaced(
    source, monkeypatch, field
):
    _config, prepared, _profile, _clock, target = source
    runtime = target.runtime
    with monkeypatch.context() as patch:
        if field == "sink":
            patch.setattr(runtime, "_execution_observation_sink", object())
        elif field == "projector":
            patch.setattr(
                runtime._execution_pose_observation_projector, "_model", object()
            )
        elif field == "live_scratch":
            patch.setattr(
                runtime._execution_pose_observation_projector,
                "_scratch",
                target.scene.data,
            )
        elif field == "fk_binding":
            projector = runtime._execution_pose_observation_projector
            patch.setattr(
                projector,
                "_bindings",
                {
                    key: replace(value, pose_schema_hash="another")
                    for key, value in projector._bindings.items()
                },
            )
        else:
            patch.setattr(
                runtime,
                "config",
                replace(runtime.config, execution_observation_plan=None),
            )
        with pytest.raises(ValueError, match="observation|observer|projector"):
            capture_motion_target(target, prepared=prepared, max_native_bytes=BUDGET)


def test_ordered_pending_cache_eviction_and_following_pair_are_preserved(
    source, monkeypatch
):
    _config, _prepared, profile, clock, target = source
    frames = []
    publish = MujocoExecutionDifferenceObserver.publish_execution_observation

    def record(owner, observation):
        frames.append(observation)
        return publish(owner, observation)

    with monkeypatch.context() as patch:
        patch.setattr(
            MujocoExecutionDifferenceObserver, "publish_execution_observation", record
        )
        assert _submit(target, profile, "frame-fixture", 30).status == "dispatched"
    reference = next(
        frame
        for frame in frames
        if isinstance(frame, ExecutionReferenceFrameObservation)
    )
    measurement = next(
        frame for frame in frames if isinstance(frame, ExecutionMeasurementObservation)
    )
    # These are typed observer inputs derived from the actual producer's frame.
    # The standalone passive owner must preserve its ordered cache semantics.
    old = MujocoExecutionDifferenceObserver(
        selection_group_id="arms",
        component_ids=("right_arm",),
        max_pending_references=2,
    )
    second = replace(reference, scheduled_for=10.01, available_at=10.01)
    third = replace(reference, scheduled_for=10.02, available_at=10.02)
    for frame in (reference, second, reference, third):
        assert old.publish_execution_observation(frame)
    saved = capture_observer(old)
    assert saved.reference_count == 4
    assert tuple(frame.scheduled_for for frame in saved.pending) == (10.0, 10.02)
    runtime = target.runtime
    with pytest.raises(ValueError, match="ordered identity"):
        restore_observer(
            replace(saved, pending=(saved.pending[0], saved.pending[0])),
            binding=runtime._execution_observation_binding,
            capability=runtime._execution_observation_capability,
            now=10.02,
            control_domain_id=runtime.config.control_domain_id,
        )
    branch = restore_observer(
        saved,
        binding=runtime._execution_observation_binding,
        capability=runtime._execution_observation_capability,
        now=10.02,
        control_domain_id=runtime.config.control_domain_id,
    )
    assert branch._reference_by_key is not old._reference_by_key
    assert branch._lock is not old._lock
    for item in (
        measurement,
        replace(measurement, measured_at=10.01, available_at=10.01),
    ):
        assert old.publish_execution_observation(item)
        assert branch.publish_execution_observation(item)
        assert _same(capture_observer(old), capture_observer(branch))
    assert branch.snapshot().unmatched_measurement_count == 1
    assert clock() == 10.0


@pytest.mark.parametrize(
    "bad",
    (
        "binding",
        "schema",
        "projector",
        "future",
        "counter",
        "missing_watermark",
        "capacity",
        "duplicate_watermark",
        "sequence_behind",
        "missing_last_time",
    ),
)
def test_observer_contract_rejects_invalid_saved_facts_without_repair(source, bad):
    _config, _prepared, profile, clock, target = source
    assert _submit(target, profile, "valid-frame", 30).status == "dispatched"
    runtime = target.runtime
    original = capture_runtime_observations(runtime)
    state = _copy(original)
    if bad == "binding":
        state = replace(state, binding=replace(state.binding, bound_at=clock() + 1))
    elif bad == "schema":
        state = replace(
            state,
            capability=replace(
                state.capability, schema_hash_by_component={"right_arm": "another"}
            ),
        )
    elif bad == "projector":
        state = replace(state, projector_hash="wrong-projector")
    elif bad == "future":
        state.fields["_execution_observation_last_published_at"]["reference"] = (
            clock() + 1
        )
    elif bad == "counter":
        state.fields["_execution_observation_latest_sequence"]["reference"] = True
    elif bad == "missing_watermark":
        state.fields["_execution_observation_watermark_by_stream"] = {}
    elif bad == "capacity":
        state = replace(state, observer=replace(state.observer, capacity=1))
    elif bad == "sequence_behind":
        state.fields["_execution_observation_latest_sequence"]["reference"] = 0
        wm = state.fields["_execution_observation_watermark_by_stream"]
        wm["reference"] = replace(wm["reference"], latest_sequence=0)
    elif bad == "missing_last_time":
        state.fields["_execution_observation_last_published_at"]["reference"] = None
    else:
        state = replace(
            state,
            observer=replace(state.observer, watermarks=state.observer.watermarks * 2),
        )
    with pytest.raises(ValueError):
        _validate_runtime_observations(runtime, state)
    assert _same(original, capture_runtime_observations(runtime))
    with pytest.raises(ValueError, match="unpublished"):
        restore_runtime_observations(runtime, original)
