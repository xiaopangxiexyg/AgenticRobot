"""Exact acceleration-planner state at a completed command transaction.

This codec does not reset, advance, dispatch, or acquire robot authority. The
existing planner remains the state owner; restoration creates a fresh planner
with caller-supplied limits and the same logical clock coordinates.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import numpy as np

from .trajectory import (
    DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    AccelerationLimitedJointTrajectoryPlanner,
    JointTrajectoryLimits,
)

_SCHEMA = "acceleration_limited_trajectory_checkpoint.v1"
_VECTORS = (
    "_q",
    "_qd",
    "_qdd",
    "_target_q",
    "_previous_target_q",
    "_braking_guard_q",
    "_braking_guard_direction",
    "_target_qd_raw",
    "_target_qd_unfiltered",
    "_target_qd_reference",
    "_target_qd_desired",
    "_target_qd_desired_raw",
    "_target_tracking_error",
    "_target_braking_offset",
    "_target_reference_lead_reserve",
    "_last_safe_target_q",
)
_FLOATS = (
    "_last_commit_monotonic",
    "_target_reference_period_sec",
    "_target_velocity_filter_alpha",
    "_target_velocity_filter_dt_sec",
    "_target_published_monotonic",
    "_last_safe_target_published_monotonic",
    "_last_observed_reference_dt_sec",
    "_speed_scale",
)
_COUNTERS = (
    "_target_sequence",
    "_anchor_resync_count",
    "_generation",
    "_next_token",
    "_last_proposed_generation",
)
_BOOLS = ("_target_velocity_filter_applied", "_quick_stop")
_TEXT = ("_mode", "_reason")
_SCALARS = _FLOATS + _COUNTERS + _BOOLS + _TEXT
_CONFIG_FIELDS = {
    "limits",
    "velocity_ramp_sec",
    "target_velocity_filter_tau_sec",
    "reference_period_sec",
    "_clock",
    "_lock",
    "_pending",
}
_OPTIONAL = "_last_anchor_resynced"  # Created lazily by accept_target, not reset.


def _configuration(planner) -> dict:
    limits = planner.limits
    return {
        "max_velocity_rad_s": limits.max_velocity_rad_s.tolist(),
        "max_acceleration_rad_s2": limits.max_acceleration_rad_s2.tolist(),
        "max_command_step_rad": (
            None
            if limits.max_command_step_rad is None
            else limits.max_command_step_rad.tolist()
        ),
        "velocity_ramp_sec": planner.velocity_ramp_sec,
        "target_velocity_filter_tau_sec": planner.target_velocity_filter_tau_sec,
        "reference_period_sec": planner.reference_period_sec,
    }


def _layout(planner) -> None:
    if type(planner) is not AccelerationLimitedJointTrajectoryPlanner:
        raise ValueError("Checkpoint requires the exact acceleration planner owner.")
    expected = set(_VECTORS + _SCALARS) | _CONFIG_FIELDS
    if set(vars(planner)) - {_OPTIONAL} != expected:
        raise ValueError("Acceleration planner implementation layout differs.")
    if planner._pending is not None:
        raise ValueError("Checkpoint requires a completed command transaction.")


def capture_acceleration_planner(
    planner: AccelerationLimitedJointTrajectoryPlanner,
) -> dict:
    """Detach a JSON-safe snapshot under the planner's own lock.

    Any parent execution owner must also exclude new target admission across
    capture of its other components. This local lock is not a whole-rig fence.
    """
    if type(planner) is not AccelerationLimitedJointTrajectoryPlanner:
        raise ValueError("Checkpoint requires the exact acceleration planner owner.")
    with planner._lock:
        _layout(planner)
        planner._require_initialized()
        state = {
            "state_schema": _SCHEMA,
            "configuration": _configuration(planner),
            "vectors": {
                name: None
                if getattr(planner, name) is None
                else getattr(planner, name).tolist()
                for name in _VECTORS
            },
            "scalars": {name: getattr(planner, name) for name in _SCALARS},
            "optional": (
                {_OPTIONAL: getattr(planner, _OPTIONAL)}
                if hasattr(planner, _OPTIONAL)
                else {}
            ),
        }
        _decode(state, planner)
        return state


def _decode(state, planner) -> dict:
    if (
        type(state) is not dict
        or set(state)
        != {"state_schema", "configuration", "vectors", "scalars", "optional"}
        or state["state_schema"] != _SCHEMA
        or type(state["configuration"]) is not dict
        or type(state["vectors"]) is not dict
        or set(state["vectors"]) != set(_VECTORS)
        or type(state["scalars"]) is not dict
        or set(state["scalars"]) != set(_SCALARS)
        or type(state["optional"]) is not dict
        or not set(state["optional"]).issubset({_OPTIONAL})
    ):
        raise ValueError("Invalid acceleration planner checkpoint fields.")
    try:
        actual = json.dumps(state["configuration"], sort_keys=True, allow_nan=False)
        expected = json.dumps(_configuration(planner), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid planner configuration identity.") from error
    if actual != expected:
        raise ValueError(
            "Checkpoint does not match the caller's planner limits/configuration."
        )
    values = {}
    for name, value in state["vectors"].items():
        if value is None:
            values[name] = None
            continue
        if (
            type(value) is not list
            or len(value) != planner.dimension
            or any(
                type(item) not in (int, float) or not math.isfinite(item)
                for item in value
            )
        ):
            raise ValueError(f"Invalid planner vector {name}.")
        if name == "_braking_guard_direction":
            if any(type(item) is not int or item not in (-1, 0, 1) for item in value):
                raise ValueError("Invalid braking guard directions.")
            values[name] = np.array(value, dtype=np.int8)
        else:
            values[name] = np.array(value, dtype=float)
    scalars = state["scalars"]
    nullable_times = {
        "_target_velocity_filter_dt_sec",
        "_target_published_monotonic",
        "_last_safe_target_published_monotonic",
        "_last_observed_reference_dt_sec",
    }
    for name in _FLOATS:
        value = scalars[name]
        if value is None and name in nullable_times:
            continue
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(f"Invalid planner time/filter scalar {name}.")
    if (
        scalars["_target_reference_period_sec"] < 0
        or not 0 <= scalars["_speed_scale"] <= 1
        or not 0 <= scalars["_target_velocity_filter_alpha"] <= 1
        or any(
            scalars[name] is not None and scalars[name] < 0
            for name in (
                "_target_velocity_filter_dt_sec",
                "_last_observed_reference_dt_sec",
            )
        )
    ):
        raise ValueError("Invalid planner filter/period range.")
    for name in _COUNTERS:
        value = scalars[name]
        nullable = name in {"_target_sequence", "_last_proposed_generation"}
        if not (value is None and nullable) and (type(value) is not int or value < 0):
            raise ValueError(f"Invalid planner counter {name}.")
    for name in _BOOLS:
        if type(scalars[name]) is not bool:
            raise ValueError(f"Invalid planner boolean {name}.")
    for name in _TEXT:
        if type(scalars[name]) is not str or not scalars[name]:
            raise ValueError(f"Invalid planner text {name}.")
    if any(type(value) is not bool for value in state["optional"].values()):
        raise ValueError("Invalid optional planner state.")
    if any(
        values[name] is None
        for name in (
            "_q",
            "_qd",
            "_qdd",
            "_braking_guard_q",
            "_braking_guard_direction",
        )
    ):
        raise ValueError("Checkpoint has no initialized command state.")
    if scalars["_last_commit_monotonic"] is None:
        raise ValueError("Checkpoint has no command time.")
    if not (
        (values["_target_q"] is None)
        == (scalars["_target_sequence"] is None)
        == (scalars["_target_published_monotonic"] is None)
    ):
        raise ValueError("Checkpoint target identity is incomplete.")
    if (values["_last_safe_target_q"] is None) != (
        scalars["_last_safe_target_published_monotonic"] is None
    ):
        raise ValueError("Checkpoint safe-target history is incomplete.")
    if values["_target_q"] is not None and any(
        values[name] is None for name in _VECTORS if name != "_last_safe_target_q"
    ):
        raise ValueError("Checkpoint target/filter state is incomplete.")
    values.update(scalars)
    values.update(state["optional"])
    return values


def restore_acceleration_planner(
    state: dict,
    *,
    limits: JointTrajectoryLimits,
    monotonic_clock: Callable[[], float],
    velocity_ramp_sec: float = DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    target_velocity_filter_tau_sec: float = 0.0,
    reference_period_sec: float | None = None,
) -> AccelerationLimitedJointTrajectoryPlanner:
    """Build an independent owner with exact times; never rebase or replay targets.

    Limits/configuration come from the destination profile, not the saved
    document. The caller supplies a clock in the checkpoint's logical domain.
    """
    candidate = AccelerationLimitedJointTrajectoryPlanner(
        limits,
        velocity_ramp_sec=velocity_ramp_sec,
        target_velocity_filter_tau_sec=target_velocity_filter_tau_sec,
        reference_period_sec=reference_period_sec,
        monotonic_clock=monotonic_clock,
    )
    values = _decode(state, candidate)
    for name, value in values.items():
        setattr(candidate, name, value)
    _layout(candidate)
    return candidate
