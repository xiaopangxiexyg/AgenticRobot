"""Typed state of the existing actuated arm and gripper owners.

These are private in-memory components, not a file importer or Gateway API.
The enclosing rig supplies exclusion, trusted provenance and native physics.
Restoration binds fresh owners without reset/forward/step or writes to ctrl.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, replace

import numpy as np

from sim.rm65.actuation import DmtacPositionActuator, Rm65PositionActuator

from ..collision_safety import CollisionCheckResult, CollisionProjectionResult
from ..control import DirectQposCommand
from .executor import MujocoActuatedPositionExecutor
from .gripper_executor import MujocoDmtacGripperExecutor

_ARM_FIELDS = {
    "_held_command",
    "_last_safety_result",
    "_last_active_collision_check",
    "_last_admitted_target_check",
    "_last_safety_status",
    "_hard_contact_latched",
    "_last_requested_qpos_by_side",
    "_final_command_fault_latched",
    "_last_final_command_projection_alpha",
}
_ARM_BINDINGS = {
    "scene",
    "timing",
    "profile",
    "collision_projection_mode",
    "allow_gripper_self_contact",
    "active_sides",
    "_actuator_ids",
    "_actuator",
    "_final_command_guard",
}
_GRIPPER_FIELDS = {
    "_accepted_close_by_gripper",
    "_servo_goal_close_by_gripper",
    "_actuator_target_close_by_gripper",
    "_actuator_target_rate_by_gripper",
    "_last_status_by_gripper",
    "_latest_command_by_gripper",
    "_last_physics_time_sec",
    "_reset",
}
_GRIPPER_BINDINGS = {
    "scene",
    "profile",
    "_actuator_ids",
    "_gripper_by_side",
    "_position_actuator",
}


@dataclass(frozen=True)
class ActuationOwnerState:
    """Detached owner component; enclosing storage defines its persistence."""

    owner: str
    configuration: str
    native_time_sec: float
    core: dict
    fields: dict


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _native_clock(
    owner, completed_time: float, *, saved_time: float | None = None
) -> float:
    now = float(owner.scene.data.time)
    dt = float(owner.scene.model.opt.timestep)
    if (
        not all(math.isfinite(value) for value in (now, dt, completed_time))
        or dt <= 0
        or min(now, completed_time) < 0
    ):
        raise ValueError("Invalid native clock state.")
    if saved_time is not None and (type(saved_time) is not float or now != saved_time):
        raise ValueError("Owner component and native clock evidence differ.")
    # MuJoCo repeatedly adds dt; the arm core computes tick*dt. Account for
    # accumulated rounding, rather than rejecting legitimate long timelines
    # with one fixed absolute epsilon. The bound must still identify a tick.
    ticks = math.ceil(completed_time / dt) + 1
    error_bound = (ticks + 4) * math.ulp(max(now, completed_time, dt))
    if error_bound >= dt / 2 or abs(now - completed_time) > error_bound:
        raise ValueError("Owner and native clock are not at the same completed tick.")
    return now


def _arm_configuration(owner) -> str:
    return _json(
        {
            "profile": owner.profile.to_metadata(),
            "timing": owner.timing.to_dict(),
            "active_sides": list(owner.active_sides),
            "collision_projection_mode": owner.collision_projection_mode,
            "allow_gripper_self_contact": owner.allow_gripper_self_contact,
        }
    )


def _arm_layout(owner) -> None:
    if (
        type(owner) is not MujocoActuatedPositionExecutor
        or type(owner._actuator) is not Rm65PositionActuator
    ):
        raise ValueError("Checkpoint requires the standard actuated arm owner.")
    if set(vars(owner)) - {"_additional_contact_oracle"} != _ARM_FIELDS | _ARM_BINDINGS:
        raise ValueError("Arm owner implementation layout differs.")
    if (
        owner._final_command_guard is not None
        or getattr(owner, "_additional_contact_oracle", None) is not None
    ):
        raise ValueError("Checkpoint cannot restore additional arm guard bindings.")
    if owner._hard_contact_latched or owner._final_command_fault_latched:
        raise ValueError("Checkpoint does not admit a faulted arm owner.")


def _check(value):
    if value is None:
        return None
    if type(value) is not CollisionCheckResult or type(value.safe) is not bool:
        raise ValueError("Invalid collision check state.")
    if math.isnan(value.minimum_distance_m):
        raise ValueError("Invalid collision distance state.")
    # All members are scalar/immutable. Keep +inf as the actual no-obstacle
    # distance rather than using the lossy diagnostic to_dict projection.
    return replace(value)


def _projection(value):
    if value is None:
        return None
    if type(value) is not CollisionProjectionResult:
        raise ValueError("Invalid collision projection state.")
    arrays = [
        np.array(getattr(value, name), dtype=float, copy=True)
        for name in ("requested_qpos", "accepted_qpos")
    ]
    if (
        arrays[0].ndim != 1
        or arrays[0].shape != arrays[1].shape
        or any(not np.isfinite(a).all() for a in arrays)
    ):
        raise ValueError("Invalid collision projection vectors.")
    return replace(
        value,
        requested_qpos=arrays[0],
        accepted_qpos=arrays[1],
        check=_check(value.check),
        blocking_check=_check(value.blocking_check),
    )


def _arm_fields(fields, sides) -> dict:
    if type(fields) is not dict or set(fields) != _ARM_FIELDS:
        raise ValueError("Incomplete arm wrapper state.")
    result = dict(fields)
    command = fields["_held_command"]
    if type(command) is not DirectQposCommand or set(command.qpos_by_side) != set(
        sides
    ):
        raise ValueError("Invalid held arm command.")
    result["_held_command"] = DirectQposCommand(
        sequence=command.sequence,
        qpos_by_side=command.qpos_by_side,
        gripper_close_by_name=command.gripper_close_by_name,
        source=command.source,
    )
    result["_last_safety_result"] = _projection(fields["_last_safety_result"])
    for name in ("_last_active_collision_check", "_last_admitted_target_check"):
        result[name] = _check(fields[name])
    requested = fields["_last_requested_qpos_by_side"]
    if type(requested) is not dict or set(requested) != set(sides):
        raise ValueError("Incomplete last requested arm command.")
    result["_last_requested_qpos_by_side"] = {}
    for side, value in requested.items():
        vector = np.array(value, dtype=float, copy=True)
        if vector.shape != (6,) or not np.isfinite(vector).all():
            raise ValueError("Invalid requested arm vector.")
        result["_last_requested_qpos_by_side"][side] = vector
    for name in ("_hard_contact_latched", "_final_command_fault_latched"):
        if type(fields[name]) is not bool or fields[name]:
            raise ValueError("Checkpoint does not admit a faulted arm owner.")
    alpha = fields["_last_final_command_projection_alpha"]
    if type(alpha) is not float or not 0 <= alpha <= 1:
        raise ValueError("Invalid final-command projection state.")
    if type(fields["_last_safety_status"]) is not str:
        raise ValueError("Invalid arm safety status.")
    return result


def _arm_ctrl(owner) -> None:
    for side, target in owner._actuator.actuator_target_by_side.items():
        if not np.array_equal(owner.scene.data.ctrl[owner._actuator_ids[side]], target):
            raise ValueError("Arm state and native ctrl differ.")


def capture_arm_owner(owner: MujocoActuatedPositionExecutor) -> ActuationOwnerState:
    """Capture under the parent rig lock, outside any physics/command step."""
    _arm_layout(owner)
    fields = _arm_fields(
        {name: getattr(owner, name) for name in _ARM_FIELDS}, owner.active_sides
    )
    core = owner._actuator.snapshot()
    _arm_ctrl(owner)
    now = _native_clock(owner, core["physics_time_sec"])
    return ActuationOwnerState("arm", _arm_configuration(owner), now, core, fields)


def restore_arm_owner(
    state: ActuationOwnerState,
    *,
    scene,
    timing,
    profile,
    active_sides,
    collision_projection_mode,
    allow_gripper_self_contact,
):
    """Return a fresh owner; the existing core parses its own complete state."""
    if (
        type(state) is not ActuationOwnerState
        or state.owner != "arm"
        or type(state.native_time_sec) is not float
    ):
        raise ValueError("Expected an arm owner component.")
    candidate = MujocoActuatedPositionExecutor(
        scene,
        timing,
        profile,
        active_sides=active_sides,
        collision_projection_mode=collision_projection_mode,
        allow_gripper_self_contact=allow_gripper_self_contact,
    )
    if state.configuration != _arm_configuration(candidate):
        raise ValueError("Arm owner configuration differs.")
    fields = _arm_fields(state.fields, candidate.active_sides)
    core = json.loads(_json(state.core))
    command = fields["_held_command"]
    candidate._actuator.reset(
        command_sequence=command.sequence,
        qpos_by_side=command.qpos_by_side,
        initial_physics_time_sec=core["physics_time_sec"],
        initial_state=core,
    )
    if _json(candidate._actuator.snapshot()) != _json(core):
        raise ValueError("Arm core did not restore the exact component.")
    for name, value in fields.items():
        setattr(candidate, name, value)
    _arm_layout(candidate)
    _arm_ctrl(candidate)
    _native_clock(candidate, core["physics_time_sec"], saved_time=state.native_time_sec)
    return candidate


def _gripper_layout(owner) -> None:
    if (
        type(owner) is not MujocoDmtacGripperExecutor
        or type(owner._position_actuator) is not DmtacPositionActuator
    ):
        raise ValueError("Checkpoint requires the standard gripper owner.")
    if set(vars(owner)) != _GRIPPER_FIELDS | _GRIPPER_BINDINGS:
        raise ValueError("Gripper owner implementation layout differs.")
    if not owner._reset:
        raise ValueError("Checkpoint requires an initialized gripper owner.")


def _gripper_fields(fields, names) -> dict:
    if (
        type(fields) is not dict
        or set(fields) != _GRIPPER_FIELDS
        or fields["_reset"] is not True
    ):
        raise ValueError("Incomplete gripper wrapper state.")
    names = set(names)
    result = copy.deepcopy(fields)
    for name in _GRIPPER_FIELDS - {"_reset", "_last_physics_time_sec"}:
        if type(result[name]) is not dict or set(result[name]) != names:
            raise ValueError(f"Incomplete gripper mapping {name}.")
    for name in (
        "_accepted_close_by_gripper",
        "_servo_goal_close_by_gripper",
        "_actuator_target_close_by_gripper",
    ):
        if any(
            type(value) is not float or not 0 <= value <= 1
            for value in result[name].values()
        ):
            raise ValueError("Invalid gripper target cache.")
    if any(
        type(value) is not float or not math.isfinite(value)
        for value in result["_actuator_target_rate_by_gripper"].values()
    ):
        raise ValueError("Invalid gripper rate cache.")
    for item in result["_latest_command_by_gripper"].values():
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("Invalid gripper queued intent shape.")
        sequence, target = item
        if (
            type(sequence) is not int
            or sequence < 0
            or (
                target is not None
                and (type(target) is not float or not 0 <= target <= 1)
            )
        ):
            raise ValueError("Invalid gripper queued intent.")
        if (sequence == 0) != (target is None):
            raise ValueError("Invalid initial gripper intent identity.")
    if any(
        type(value) is not str for value in result["_last_status_by_gripper"].values()
    ):
        raise ValueError("Invalid gripper status cache.")
    now = result["_last_physics_time_sec"]
    if type(now) is not float or not math.isfinite(now) or now < 0:
        raise ValueError("Invalid gripper physics time.")
    return result


def _gripper_ctrl(owner) -> None:
    for name, ids in owner._actuator_ids.items():
        target = owner.profile.qpos_from_close(
            name, owner._actuator_target_close_by_gripper[name]
        )
        if not np.array_equal(owner.scene.data.ctrl[ids], target):
            raise ValueError("Gripper state and native ctrl differ.")
    core = owner._position_actuator.snapshot()
    caches = {
        "_accepted_close_by_gripper": "accepted_target_by_side",
        "_servo_goal_close_by_gripper": "servo_goal_by_side",
        "_actuator_target_close_by_gripper": "actuator_position_by_side",
        "_actuator_target_rate_by_gripper": "actuator_rate_by_side",
    }
    for name, key in caches.items():
        if getattr(owner, name) != owner._map_by_gripper(core[key]):
            raise ValueError("Gripper wrapper cache and core state differ.")
    for name, (sequence, _target) in owner._latest_command_by_gripper.items():
        if sequence > core["sequence"]:
            raise ValueError(
                "Gripper intent sequence exceeds the core high-water mark."
            )
    for side, pending in core["pending"].items():
        name = owner._gripper_by_side[side]
        # Existing reset(initial_state=...) restores core work but starts a
        # new wrapper receipt history at (0, None). Preserve that explicit
        # unknown-local-history state instead of inventing a prior receipt.
        if owner._latest_command_by_gripper[name] == (0, None):
            continue
        if owner._latest_command_by_gripper[name] != (
            pending["sequence"],
            pending["target"],
        ):
            raise ValueError(
                "Gripper pending core intent differs from its wrapper receipt."
            )


def capture_gripper_owner(owner: MujocoDmtacGripperExecutor) -> ActuationOwnerState:
    """Preserve queued intent separately from accepted/servo targets."""
    _gripper_layout(owner)
    fields = _gripper_fields(
        {name: getattr(owner, name) for name in _GRIPPER_FIELDS},
        owner.profile.joint_names_by_gripper,
    )
    _gripper_ctrl(owner)
    now = _native_clock(owner, fields["_last_physics_time_sec"])
    return ActuationOwnerState(
        "gripper",
        _json(owner.profile.to_metadata()),
        now,
        owner._position_actuator.snapshot(),
        fields,
    )


def restore_gripper_owner(state: ActuationOwnerState, *, scene, profile):
    """Bind and restore the existing core without wrapper reset or ctrl writes."""
    if (
        type(state) is not ActuationOwnerState
        or state.owner != "gripper"
        or type(state.native_time_sec) is not float
    ):
        raise ValueError("Expected a gripper owner component.")
    candidate = MujocoDmtacGripperExecutor(scene, profile)
    if state.configuration != _json(profile.to_metadata()):
        raise ValueError("Gripper owner configuration differs.")
    fields = _gripper_fields(state.fields, profile.joint_names_by_gripper)
    core = json.loads(_json(state.core))
    initial = {
        profile.side_by_gripper[name]: value
        for name, value in fields["_actuator_target_close_by_gripper"].items()
    }
    candidate._position_actuator.reset(
        initial,
        initial_physics_time_sec=fields["_last_physics_time_sec"],
        initial_state=core,
    )
    if _json(candidate._position_actuator.snapshot()) != _json(core):
        raise ValueError("Gripper core did not restore the exact component.")
    for name, value in fields.items():
        setattr(candidate, name, value)
    _gripper_layout(candidate)
    _gripper_ctrl(candidate)
    _native_clock(
        candidate, fields["_last_physics_time_sec"], saved_time=state.native_time_sec
    )
    return candidate
