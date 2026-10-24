"""Checkpoint the existing manual, actuated execution rig at a completed tick.

This is a trusted in-memory lower component, not a Gateway/file importer.
The application must fence external admission and authenticate its source and
profile. No worker, observer, scheduled receiver or custom guard is restored.
There is no new control loop and no start/reset/replay during restoration.
"""

from __future__ import annotations

import copy
import json
import math
import threading
from dataclasses import asdict, dataclass, fields

import numpy as np

from robot_motion.position_control.trajectory import (
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
)
from robot_motion.position_control.trajectory_checkpoint import (
    capture_acceleration_planner,
    restore_acceleration_planner,
)
from sim.mujoco.actuation.checkpoint import (
    ActuationOwnerState,
    capture_arm_owner,
    capture_gripper_owner,
    restore_arm_owner,
    restore_gripper_owner,
)
from sim.mujoco.actuation.executor import COLLISION_PROJECTION_MODE_EXECUTOR
from sim.mujoco.native_checkpoint import NativeMujocoState, capture_native_state
from sim.mujoco.scene import MujocoScene, SceneLoadResult

from .robot_motion import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
    _QueuedReference,
)

_RIG_LAYOUT = {
    "scene",
    "config",
    "_clock",
    "_sleep",
    "_lock",
    "_stop_event",
    "_worker",
    "_started",
    "_clock_origin",
    "_physics_step_observer",
    "_initial_physics_time_sec",
    "_initial_command_sequence",
    "_initial_actuation_state",
    "_initial_gripper_state",
    "_initial_qcmd_by_side",
    "_initial_gripper_close_by_name",
    "_next_reference_sequence",
    "_next_command_sequence",
    "_pending_reference",
    "_scheduled_activation_guard",
    "_scheduled_active_work_guard",
    "_hold_reason",
    "_last_reference",
    "_last_worker_error",
    "_worker_physics_ticks",
    "_worker_deadline_misses",
    "_pending_proposals",
    "_planners",
    "_execution",
    "_direct",
    "_gripper_execution",
    "_accepted_qcmd_by_side",
    "_reference_activation_observations",
    "_planner_command_frame_observations",
}
_STATE_FIELDS = {
    "_clock_origin",
    "_next_reference_sequence",
    "_next_command_sequence",
    "_pending_reference",
    "_last_reference",
    "_hold_reason",
    "_worker_physics_ticks",
    "_worker_deadline_misses",
    "_accepted_qcmd_by_side",
}


class ManualExecutionClock:
    """Explicit simulation caller clock; physics is advanced only by the rig.

    Keep this clock in the same logical domain after recovery. In particular,
    never replace its saved value with the receiving process's wall clock.
    """

    def __init__(self, value: float = 0.0):
        self._lock = threading.RLock()
        self._value = _finite(value, "manual clock")
        if self._value < 0:
            raise ValueError("Manual clock must be non-negative.")

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance_to(self, value: float) -> None:
        value = _finite(value, "manual clock")
        with self._lock:
            if value < self._value:
                raise ValueError("Manual clock cannot move backwards.")
            self._value = value


@dataclass(frozen=True)
class ExecutionRigCheckpoint:
    """Detached native, planner and transport cut of one running manual rig.

    The enclosing store owns code/profile provenance and atomic persistence.
    These fields are not a portable or untrusted-input serialization format.
    """

    configuration: str
    scene_metadata: str
    clock_value: float
    native: NativeMujocoState
    arm: ActuationOwnerState
    gripper: ActuationOwnerState
    planners: dict
    fields: dict


def _finite(value, name) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Invalid {name}.")
    return float(value)


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _configuration(config: MujocoPositionExecutionConfig) -> str:
    if (
        type(config) is not MujocoPositionExecutionConfig
        or config.realtime_worker is not False
        or config.actuation_profile is None
        or config.gripper_actuation_profile is None
        or config.active_sides != ("left", "right")
        or config.collision_projection_mode != COLLISION_PROJECTION_MODE_EXECUTOR
        or config.position_command_planner
        != POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
    ):
        raise ValueError("Checkpoint requires manual actuated acceleration execution.")
    values = {field.name: getattr(config, field.name) for field in fields(config)}
    values["timing"] = config.timing.to_dict()
    for name in ("actuation_profile", "gripper_actuation_profile"):
        values[name] = values[name].to_metadata()
    return _json(values)


def _scene_metadata(metadata: SceneLoadResult) -> str:
    if type(metadata) is not SceneLoadResult:
        raise ValueError("Checkpoint requires the standard scene metadata.")
    # File paths are locators, not native identity. Preserve all runtime scene
    # semantics, including primitive v2, without requiring the old source tree.
    return _json(
        {
            "arms": {side: asdict(spec) for side, spec in metadata.arms.items()},
            "grippers": {
                name: asdict(spec) for name, spec in metadata.grippers.items()
            },
            "cameras": [asdict(camera) for camera in metadata.cameras],
            "scene_spec": None
            if metadata.scene_spec is None
            else metadata.scene_spec.to_dict(),
            "conversion": {
                name: getattr(metadata.conversion, name)
                for name in ("joint_names", "body_names", "scene_profile_name")
            },
        }
    )


def _metadata(value):
    """Copy admitted data, retaining tuple/array semantics; reject live bindings."""
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _metadata(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_metadata(item) for item in value)
    if (
        type(value) is np.ndarray
        and value.dtype.kind in "bifu"
        and np.isfinite(value).all()
    ):
        return value.copy()
    raise ValueError(
        "Checkpoint metadata contains an unsupported live or non-finite value."
    )


def _vectors(values, sides) -> dict[str, np.ndarray]:
    if type(values) is not dict or set(values) != set(sides):
        raise ValueError("Incomplete rig arm vectors.")
    result = {}
    for side, vector in values.items():
        vector = np.array(vector, dtype=float, copy=True)
        if vector.shape != (6,) or not np.isfinite(vector).all():
            raise ValueError("Invalid rig arm vector.")
        result[side] = vector
    return result


def _reference(reference, sides):
    if reference is None:
        return None
    if (
        type(reference) is not _QueuedReference
        or type(reference.sequence) is not int
        or reference.sequence < 1
        or reference.reference_segment_id is not None
        or reference.execution_epoch is not None
        or reference.group_segment_generation is not None
        or reference.gripper_close_by_name
        or type(reference.metadata_by_side) is not dict
        or set(reference.metadata_by_side) != set(sides)
        or any(type(value) is not dict for value in reference.metadata_by_side.values())
    ):
        raise ValueError(
            "Checkpoint cannot restore scheduled or invalid rig references."
        )
    published = _finite(reference.published_monotonic_time, "reference publication")
    activation = _finite(reference.activate_at, "reference activation")
    if published != activation:
        raise ValueError(
            "Manual immediate reference publication and activation differ."
        )
    return _QueuedReference(
        sequence=reference.sequence,
        published_monotonic_time=published,
        activate_at=activation,
        reference_segment_id=None,
        execution_epoch=None,
        group_segment_generation=None,
        qref_by_side=_vectors(reference.qref_by_side, sides),
        metadata_by_side={
            side: _metadata(reference.metadata_by_side[side]) for side in sides
        },
        gripper_close_by_name={},
    )


def _state_fields(values, sides) -> dict:
    if type(values) is not dict or set(values) != _STATE_FIELDS:
        raise ValueError("Incomplete rig checkpoint fields.")
    result = dict(values)
    result["_clock_origin"] = _finite(values["_clock_origin"], "rig clock origin")
    for name in (
        "_next_reference_sequence",
        "_next_command_sequence",
        "_worker_physics_ticks",
        "_worker_deadline_misses",
    ):
        value = values[name]
        if type(value) is not int or value < (1 if name.startswith("_next_") else 0):
            raise ValueError(f"Invalid rig counter {name}.")
    if result["_worker_deadline_misses"] != 0:
        raise ValueError("Manual rig cannot have worker deadline misses.")
    if values["_hold_reason"] is not None:
        raise ValueError(
            "Checkpoint cannot capture an incomplete rig HOLD transaction."
        )
    result["_accepted_qcmd_by_side"] = _vectors(values["_accepted_qcmd_by_side"], sides)
    for name in ("_pending_reference", "_last_reference"):
        reference = result[name] = _reference(values[name], sides)
        if (
            reference is not None
            and reference.sequence >= values["_next_reference_sequence"]
        ):
            raise ValueError("Reference exceeds the rig sequence high-water mark.")
    pending, active = result["_pending_reference"], result["_last_reference"]
    if pending is not None and (
        pending.sequence != result["_next_reference_sequence"] - 1
        or (active is not None and pending.sequence <= active.sequence)
    ):
        raise ValueError("Invalid rig mailbox sequence order.")
    return result


def _qualify(rig: MujocoPositionExecutionRig) -> None:
    if type(rig) is not MujocoPositionExecutionRig or set(vars(rig)) != _RIG_LAYOUT:
        raise ValueError("Checkpoint requires the exact standard rig layout.")
    _configuration(rig.config)
    if (
        type(rig.scene) is not MujocoScene
        or type(rig._clock) is not ManualExecutionClock
        or set(vars(rig._clock)) != {"_lock", "_value"}
        or not rig._started
        or rig._stop_event.is_set()
        or rig._last_worker_error is not None
        or rig._worker is not None
        or rig._physics_step_observer is not None
        or rig._scheduled_activation_guard is not None
        or rig._scheduled_active_work_guard is not None
        or rig._pending_proposals
        or rig._direct is not None
        or rig._gripper_execution is None
        or rig._reference_activation_observations
        or rig._planner_command_frame_observations
        or rig._initial_physics_time_sec != 0.0
        or rig._initial_command_sequence != 0
        or any(
            getattr(rig, name) is not None
            for name in (
                "_initial_actuation_state",
                "_initial_gripper_state",
                "_initial_qcmd_by_side",
                "_initial_gripper_close_by_name",
            )
        )
        or set(rig._planners) != set(rig.config.active_sides)
        or rig._execution.scene is not rig.scene
        or rig._gripper_execution.scene is not rig.scene
        or float(rig.scene.model.opt.timestep) != rig.config.timing.physics_timestep_sec
    ):
        raise ValueError(
            "Checkpoint requires a running, callback-free completed manual rig."
        )


def _coherent(rig, fields, arm):
    if fields["_next_command_sequence"] != arm.fields["_held_command"].sequence + 1:
        raise ValueError("Rig and held command sequence differ.")
    if arm.fields["_held_command"].sequence != arm.core["accepted_command_sequence"]:
        raise ValueError("Held command and arm core sequence differ.")
    if fields["_worker_physics_ticks"] > arm.core["physics_tick"]:
        raise ValueError("Rig tick count exceeds its execution timeline.")
    for name in ("_pending_reference", "_last_reference"):
        reference = fields[name]
        if reference is not None and reference.published_monotonic_time > rig._clock():
            raise ValueError("Reference publication exceeds the caller clock.")
    for side in rig.config.active_sides:
        planner = rig._planners[side]
        if planner._clock is not rig._clock:
            raise ValueError("Planner and rig logical clock bindings differ.")
        if planner._target_sequence is None:
            raise ValueError("A started rig requires both planners to be primed.")
        if not (
            np.array_equal(fields["_accepted_qcmd_by_side"][side], planner._q)
            and np.array_equal(planner._q, arm.core["accepted_qcmd_by_side"][side])
        ):
            raise ValueError("Rig, planner and actuator accepted command differ.")
        if (
            planner._target_sequence is not None
            and planner._target_sequence >= fields["_next_reference_sequence"]
        ):
            raise ValueError("Planner target exceeds rig reference high-water mark.")
        active = fields["_last_reference"]
        if active is not None and planner._target_sequence != active.sequence:
            raise ValueError("Active reference and planner target identity differ.")
        if (
            planner._last_commit_monotonic
            > fields["_clock_origin"] + rig._execution.physics_time_sec + 1e-12
        ):
            raise ValueError("Planner command time exceeds the completed physics cut.")


def capture_execution_rig(
    rig: MujocoPositionExecutionRig, *, max_native_bytes: int
) -> ExecutionRigCheckpoint:
    """Capture one completed cut; caller first drains external admissions.

    Rejecting callbacks is essential: the rig uses an RLock and an observer
    can run after mj_step but before planner commit, even on a non-command tick.
    A local lock alone is not an application/transport admission fence.
    """
    if type(rig) is not MujocoPositionExecutionRig:
        raise ValueError("Checkpoint requires the standard rig owner.")
    with rig._lock:
        _qualify(rig)
        with rig._clock._lock:
            fields = _state_fields(
                {name: getattr(rig, name) for name in _STATE_FIELDS},
                rig.config.active_sides,
            )
            arm = capture_arm_owner(rig._execution)
            gripper = capture_gripper_owner(rig._gripper_execution)
            planners = {
                side: capture_acceleration_planner(planner)
                for side, planner in rig._planners.items()
            }
            _coherent(rig, fields, arm)
            result = ExecutionRigCheckpoint(
                configuration=_configuration(rig.config),
                scene_metadata=_scene_metadata(rig.scene.load_result),
                clock_value=rig._clock(),
                native=capture_native_state(
                    rig.scene.data, max_native_bytes=max_native_bytes
                ),
                arm=arm,
                gripper=gripper,
                planners=planners,
                fields=fields,
            )
            # close() sets the event before taking this lock. Do not publish a
            # usable cut after observing a concurrent lifecycle termination.
            _qualify(rig)
            return result


def restore_execution_rig(
    state: ExecutionRigCheckpoint,
    *,
    config: MujocoPositionExecutionConfig,
    load_result: SceneLoadResult,
    expected_model_sha256: str,
    max_native_bytes: int,
) -> MujocoPositionExecutionRig:
    """Build an unpublished independent rig, preserving its logical timeline.

    Config/metadata and expected native model identity come from the trusted
    destination profile. The caller's store must also bind implementation,
    owner and parent provenance; this component does not waive those checks.
    """
    if (
        type(state) is not ExecutionRigCheckpoint
        or state.configuration != _configuration(config)
        or state.scene_metadata != _scene_metadata(load_result)
        or type(state.native) is not NativeMujocoState
        or type(expected_model_sha256) is not str
        or len(expected_model_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_model_sha256)
        or state.native.model_sha256 != expected_model_sha256
        or type(state.planners) is not dict
        or set(state.planners) != set(config.active_sides)
    ):
        raise ValueError(
            "Rig checkpoint configuration, metadata or native identity differs."
        )
    clock = ManualExecutionClock(state.clock_value)
    values = _state_fields(state.fields, config.active_sides)
    scene = MujocoScene.from_native_checkpoint(
        state.native, load_result=load_result, max_native_bytes=max_native_bytes
    )
    candidate = MujocoPositionExecutionRig(scene, config, monotonic_clock=clock)
    candidate._execution = restore_arm_owner(
        state.arm,
        scene=scene,
        timing=config.timing,
        profile=config.actuation_profile,
        active_sides=config.active_sides,
        collision_projection_mode=config.collision_projection_mode,
        allow_gripper_self_contact=config.allow_gripper_self_contact,
    )
    candidate._gripper_execution = restore_gripper_owner(
        state.gripper, scene=scene, profile=config.gripper_actuation_profile
    )
    for side in config.active_sides:
        prototype = candidate._new_planner()
        candidate._planners[side] = restore_acceleration_planner(
            copy.deepcopy(state.planners[side]),
            limits=prototype.limits,
            monotonic_clock=clock,
            velocity_ramp_sec=config.velocity_ramp_sec,
            target_velocity_filter_tau_sec=config.target_velocity_filter_tau_sec,
            reference_period_sec=1.0 / config.timing.reference_hz,
        )
    for name, value in values.items():
        setattr(candidate, name, value)
    _coherent(candidate, values, state.arm)
    candidate._started = True
    _qualify(candidate)
    return candidate
