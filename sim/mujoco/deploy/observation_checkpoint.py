"""Exact passive observer state for an unpublished manual target branch.

The enclosing runtime checkpoint hashes this typed bundle; its store owns provenance.
Capture requires the same application fence as the whole runtime step, including
direct sink publication. Neither capture nor restore publishes or samples a cut.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

from control_composition.observability import (
    DifferenceReceiptReference,
    MeasurementReferenceAssociation,
)
from robot_motion.action.checkpoint_codec import (
    decode_checkpoint_value,
    encode_checkpoint_value,
)
from robot_motion.action.execution_observation import (
    ExecutionFactReference,
    ExecutionMeasurementObservation,
    ExecutionObservationBinding,
    ExecutionObservationCapabilitySpec,
    ExecutionObservationPlan,
    ExecutionObservationWatermark,
    ExecutionReferenceFrameObservation,
    bind_execution_observation_plan,
    validate_execution_observation_binding,
)

from .execution_difference import MujocoExecutionDifferenceObserver
from .pose_kinematics import (
    MujocoPoseKinematicsBinding,
    MujocoPoseObservationProjector,
    pose_projector_contract_hash,
)

_STREAMS = {"reference", "measurement"}
_COUNTERS = (
    "_execution_observation_emitted_count",
    "_execution_observation_dropped_count",
    "_execution_observation_latest_sequence",
)
RUNTIME_OBSERVATION_FIELDS = {
    "_execution_observation_last_published_at",
    *_COUNTERS,
    "_execution_observation_watermark_by_stream",
}
_SINK_FIELDS = {
    "selection_group_id",
    "component_ids",
    "max_pending_references",
    "_lock",
    "_reference_by_key",
    "_latest_association",
    "_watermark_by_stream",
    "_reference_count",
    "_measurement_count",
    "_unmatched_measurement_count",
}
_PROJECTOR_FIELDS = {
    "_mujoco",
    "_model",
    "_nq",
    "_bindings",
    "_scratch",
    "_lock",
    "model_contract_id",
    "model_contract_hash",
    "semantic_content_hash",
    "contract_id",
}


@dataclass(frozen=True)
class ExecutionObserverCheckpoint:
    selection_group_id: str
    component_ids: tuple[str, ...]
    capacity: int
    pending: tuple[ExecutionReferenceFrameObservation, ...]
    latest: MeasurementReferenceAssociation | None
    watermarks: tuple[ExecutionObservationWatermark, ...]
    reference_count: int
    measurement_count: int
    unmatched_count: int


@dataclass(frozen=True)
class RuntimeObservationCheckpoint:
    binding: ExecutionObservationBinding
    capability: ExecutionObservationCapabilitySpec
    projector_hash: str
    side_to_component: dict
    fields: dict
    observer: ExecutionObserverCheckpoint


OBSERVATION_RECORDS = (
    ExecutionFactReference,
    ExecutionMeasurementObservation,
    ExecutionObservationBinding,
    ExecutionObservationCapabilitySpec,
    ExecutionObservationPlan,
    ExecutionObservationWatermark,
    ExecutionReferenceFrameObservation,
    DifferenceReceiptReference,
    MeasurementReferenceAssociation,
    ExecutionObserverCheckpoint,
    RuntimeObservationCheckpoint,
)


def _copy(value):
    return decode_checkpoint_value(
        encode_checkpoint_value(value, extra_record_types=OBSERVATION_RECORDS),
        extra_record_types=OBSERVATION_RECORDS,
    )


def _same(left, right):
    return encode_checkpoint_value(
        left, extra_record_types=OBSERVATION_RECORDS
    ) == encode_checkpoint_value(right, extra_record_types=OBSERVATION_RECORDS)


def _counter(value):
    if type(value) is not int or value < 0:
        raise ValueError("Invalid observation checkpoint counter.")
    return value


def _time(value, now):
    if type(value) not in (float, int) or not math.isfinite(value) or value > now:
        raise ValueError("Observation checkpoint time is invalid or in the future.")


def validate_plan(plan):
    if plan is not None and (
        type(plan) is not ExecutionObservationPlan
        or plan.reference_mode != "full_frames"
        or plan.measurement_mode != "full_frames"
        or len(plan.component_ids) != 1
        or dict(plan.required_representation_by_component)
        != {plan.component_ids[0]: "cartesian_pose"}
    ):
        raise ValueError(
            "Observation checkpoint requires the original Cartesian full_frames plan."
        )


def qualify_observation_graph(runtime):
    plan = runtime.config.execution_observation_plan
    validate_plan(plan)
    if plan is None:
        if (
            any(
                getattr(runtime, field) is not None
                for field in (
                    "_execution_observation_binding",
                    "_execution_observation_capability",
                    "_execution_observation_sink",
                    "_execution_pose_observation_projector",
                )
            )
            or runtime._execution_observation_representation != "off"
            or runtime._execution_observation_last_published_at
            != dict.fromkeys(_STREAMS)
            or any(
                getattr(runtime, field) != dict.fromkeys(_STREAMS, 0)
                for field in _COUNTERS
            )
            or runtime._execution_observation_watermark_by_stream
        ):
            raise ValueError("Off observation profile contains live observer state.")
        return
    sink = runtime._execution_observation_sink
    projector = runtime._execution_pose_observation_projector
    if (
        type(sink) is not MujocoExecutionDifferenceObserver
        or set(vars(sink)) != _SINK_FIELDS
        or sink.component_ids != plan.component_ids
        or sink.selection_group_id not in runtime.catalog.selection_group_ids
        or type(projector) is not MujocoPoseObservationProjector
        or set(vars(projector)) != _PROJECTOR_FIELDS
        or projector._model is not runtime.rig.scene.model
        or projector._mujoco is not runtime.rig.scene.mujoco
        or type(projector._scratch) is not runtime.rig.scene.mujoco.MjData
        or projector._scratch is runtime.rig.scene.data
        or projector._scratch.model is not projector._model
        or projector._nq != runtime.rig.scene.model.nq
        or projector.component_ids != plan.component_ids
        or runtime._execution_observation_representation != "cartesian_pose"
    ):
        raise ValueError("Unsupported execution observer/projector owner or binding.")
    if (
        any(
            type(value) is not MujocoPoseKinematicsBinding or key != value.component_id
            for key, value in projector._bindings.items()
        )
        or projector.semantic_content_hash
        != pose_projector_contract_hash(
            model_contract_id=projector.model_contract_id,
            model_contract_hash=projector.model_contract_hash,
            bindings=projector._bindings.values(),
        )
        or projector.contract_id
        != f"mujoco-pose-observation-projector:{projector.semantic_content_hash}"
    ):
        raise ValueError("FK projector identity differs from its actual bindings.")
    expected = bind_execution_observation_plan(
        plan,
        runtime._execution_observation_capability,
        bound_at=runtime._execution_observation_binding.bound_at,
    )
    if not _same(expected, runtime._execution_observation_binding):
        raise ValueError(
            "Execution observation binding differs from its original plan."
        )
    _time(expected.bound_at, runtime.clock())


def capture_observer(observer):
    """Detach the standard sink's complete ordered cache under its own lock."""
    if (
        type(observer) is not MujocoExecutionDifferenceObserver
        or set(vars(observer)) != _SINK_FIELDS
    ):
        raise ValueError("Unknown execution observer owner.")
    with observer._lock:
        if type(observer._reference_by_key) is not OrderedDict or any(
            key != observer._key(value)
            for key, value in observer._reference_by_key.items()
        ):
            raise ValueError(
                "Execution observer pending keys differ from their frames."
            )
        state = ExecutionObserverCheckpoint(
            observer.selection_group_id,
            observer.component_ids,
            observer.max_pending_references,
            tuple(observer._reference_by_key.values()),
            observer._latest_association,
            tuple(observer._watermark_by_stream.values()),
            observer._reference_count,
            observer._measurement_count,
            observer._unmatched_measurement_count,
        )
        if any(
            key != value.stream for key, value in observer._watermark_by_stream.items()
        ):
            raise ValueError("Execution observer watermark keys differ.")
        return _copy(state)


def new_observer(state):
    if (
        type(state) is not ExecutionObserverCheckpoint
        or _counter(state.capacity) < 1
        or type(state.selection_group_id) is not str
        or not state.selection_group_id.strip()
        or type(state.component_ids) is not tuple
        or not state.component_ids
        or any(
            type(value) is not str or not value.strip() for value in state.component_ids
        )
    ):
        raise ValueError("Invalid execution observer capacity or record.")
    return MujocoExecutionDifferenceObserver(
        selection_group_id=state.selection_group_id,
        component_ids=state.component_ids,
        max_pending_references=state.capacity,
    )


def _watermarks(values, binding, now):
    if type(values) is not dict or not set(values).issubset(_STREAMS):
        raise ValueError("Invalid observation watermark streams.")
    for stream, value in values.items():
        if (
            type(value) is not ExecutionObservationWatermark
            or stream != value.stream
            or value.observation_binding_id != binding.binding_id
            or value.observation_plan_hash != binding.plan_hash
            or value.clock_domain_id != binding.clock_domain_id
        ):
            raise ValueError("Observation watermark identity differs.")
        _time(value.processed_through, now)
        _time(value.available_at, now)


def _validate_observer(state, *, binding, capability, now, control_domain_id):
    new_observer(state)  # Reuse the original constructor's configuration checks.
    for value in (
        state.reference_count,
        state.measurement_count,
        state.unmatched_count,
    ):
        _counter(value)
    if (
        state.unmatched_count > state.measurement_count
        or type(state.pending) is not tuple
        or len(state.pending) > state.capacity
        or len(state.pending) > state.reference_count
        or (state.latest is None) != (state.measurement_count == 0)
    ):
        raise ValueError("Invalid execution observer retained counts.")
    keys = set()
    for frame in state.pending:
        if type(frame) is not ExecutionReferenceFrameObservation:
            raise ValueError("Unknown pending observation frame.")
        validate_execution_observation_binding(frame, binding, capability)
        key = MujocoExecutionDifferenceObserver._key(frame)
        if (
            key in keys
            or frame.selection_group_id != state.selection_group_id
            or frame.control_domain_id != control_domain_id
        ):
            raise ValueError("Pending observation scope or ordered identity differs.")
        keys.add(key)
        _time(frame.scheduled_for, now)
        _time(frame.available_at, now)
    if state.latest is not None:
        latest = state.latest
        if (
            type(latest) is not MeasurementReferenceAssociation
            or latest.selection_group_id != state.selection_group_id
            or latest.control_domain_id != control_domain_id
            or latest.clock_domain_id != binding.clock_domain_id
            or not set(latest.component_difference).issubset(state.component_ids)
        ):
            raise ValueError("Latest association belongs to another observer.")
        _time(latest.measured_at, now)
    if type(state.watermarks) is not tuple or any(
        type(value) is not ExecutionObservationWatermark for value in state.watermarks
    ):
        raise ValueError("Invalid observer watermark records.")
    if len({v.stream for v in state.watermarks}) != len(state.watermarks):
        raise ValueError("Duplicate observer watermark streams.")
    _watermarks({v.stream: v for v in state.watermarks}, binding, now)


def _validate_runtime_observations(runtime, state):
    if (
        type(state) is not RuntimeObservationCheckpoint
        or type(state.binding) is not ExecutionObservationBinding
        or type(state.capability) is not ExecutionObservationCapabilitySpec
    ):
        raise ValueError("Missing execution observation checkpoint.")
    now = runtime.clock()
    expected = bind_execution_observation_plan(
        runtime.config.execution_observation_plan,
        runtime._execution_observation_capability,
        bound_at=state.binding.bound_at,
    )
    if (
        not _same(expected, state.binding)
        or not _same(state.capability, runtime._execution_observation_capability)
        or state.projector_hash
        != runtime._execution_pose_observation_projector.semantic_content_hash
        or state.side_to_component
        != runtime._execution_observation_side_to_component_id
        or type(state.fields) is not dict
        or set(state.fields) != RUNTIME_OBSERVATION_FIELDS
    ):
        raise ValueError("Saved observation capability, projector or plan differs.")
    _time(state.binding.bound_at, now)
    for field in (*_COUNTERS, "_execution_observation_last_published_at"):
        values = state.fields[field]
        if type(values) is not dict or set(values) != _STREAMS:
            raise ValueError("Invalid observation per-stream fields.")
        for value in values.values():
            if field in _COUNTERS:
                _counter(value)
            elif value is not None:
                _time(value, now)
    watermarks = state.fields["_execution_observation_watermark_by_stream"]
    _watermarks(watermarks, state.binding, now)
    for stream in _STREAMS:
        emitted = state.fields["_execution_observation_emitted_count"][stream]
        latest = state.fields["_execution_observation_latest_sequence"][stream]
        published_at = state.fields["_execution_observation_last_published_at"][stream]
        if emitted > latest or (emitted > 0 and published_at is None):
            raise ValueError("Impossible observation publication progress.")
    for stream in _STREAMS - set(watermarks):
        if (
            any(state.fields[name][stream] for name in _COUNTERS)
            or state.fields["_execution_observation_last_published_at"][stream]
            is not None
        ):
            raise ValueError("Observation progress is missing its runtime watermark.")
    for stream, value in watermarks.items():
        for field, attribute in zip(
            _COUNTERS, ("emitted_count", "dropped_count", "latest_sequence")
        ):
            if state.fields[field][stream] != getattr(value, attribute):
                raise ValueError(
                    "Runtime counters differ from their retained watermark."
                )
    sink = runtime._execution_observation_sink
    saved = state.observer
    if (sink.selection_group_id, sink.component_ids, sink.max_pending_references) != (
        saved.selection_group_id,
        saved.component_ids,
        saved.capacity,
    ):
        raise ValueError("Execution observer configuration differs.")
    _validate_observer(
        saved,
        binding=state.binding,
        capability=state.capability,
        now=now,
        control_domain_id=runtime.config.control_domain_id,
    )
    if (
        saved.reference_count
        != state.fields["_execution_observation_emitted_count"]["reference"]
        or saved.measurement_count
        != state.fields["_execution_observation_emitted_count"]["measurement"]
    ):
        raise ValueError("Runtime publication counts differ from the owned observer.")


def capture_runtime_observations(runtime):
    qualify_observation_graph(runtime)
    if runtime.config.execution_observation_plan is None:
        return None
    state = _copy(
        RuntimeObservationCheckpoint(
            runtime._execution_observation_binding,
            runtime._execution_observation_capability,
            runtime._execution_pose_observation_projector.semantic_content_hash,
            runtime._execution_observation_side_to_component_id,
            {name: getattr(runtime, name) for name in RUNTIME_OBSERVATION_FIELDS},
            capture_observer(runtime._execution_observation_sink),
        )
    )
    _validate_runtime_observations(runtime, state)
    return state


def restore_runtime_observations(runtime, state):
    """Install exact retained facts into a fresh sink; never call publish."""
    if runtime._started or runtime._lifecycle_entered or runtime._lifecycle_terminal:
        raise ValueError("Observation restore requires an unpublished runtime.")
    qualify_observation_graph(runtime)
    if runtime.config.execution_observation_plan is None:
        if state is not None:
            raise ValueError("Enabled observations cannot restore into an off profile.")
        return
    state = _copy(state)
    _validate_runtime_observations(runtime, state)
    sink = runtime._execution_observation_sink
    _install_observer(sink, state.observer)
    runtime._execution_observation_binding = state.binding
    for name, value in state.fields.items():
        setattr(runtime, name, value)


def restore_observer(state, *, binding, capability, now, control_domain_id):
    """Restore one passive owner without replaying its old input stream."""
    state = _copy(state)
    _validate_observer(
        state,
        binding=binding,
        capability=capability,
        now=now,
        control_domain_id=control_domain_id,
    )
    sink = new_observer(state)
    _install_observer(sink, state)
    return sink


def _install_observer(sink, saved):
    with sink._lock:
        if (
            sink._reference_by_key
            or sink._latest_association is not None
            or sink._watermark_by_stream
            or sink._reference_count
            or sink._measurement_count
            or sink._unmatched_measurement_count
        ):
            raise ValueError(
                "Observation restoration requires an unpublished fresh sink."
            )
        sink._reference_by_key = OrderedDict(
            (sink._key(value), value) for value in saved.pending
        )
        sink._latest_association = saved.latest
        sink._watermark_by_stream = {value.stream: value for value in saved.watermarks}
        sink._reference_count = saved.reference_count
        sink._measurement_count = saved.measurement_count
        sink._unmatched_measurement_count = saved.unmatched_count
