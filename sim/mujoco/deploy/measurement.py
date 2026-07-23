"""Sim-only measurement projection for formal control traces."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash


MUJOCO_BACKEND_MEASUREMENT_VERSION = "dexteleop.sim.mujoco.measurement.v1"
MUJOCO_QPOS_MEASUREMENT_HASH_DOMAIN_V1 = (
    "dexteleop.sim.mujoco.qpos_measurement.v1"
)


@dataclass(frozen=True)
class MujocoQposMeasurement:
    """Backend-owned raw arm-qpos sample without policy/source identity."""

    measurement_id: str
    qpos_by_side: Mapping[str, tuple[float, ...]]
    measured_at: float
    clock_domain_id: str
    physics_tick: int | None
    semantic_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "qpos_by_side",
            MappingProxyType(
                {
                    str(side): tuple(float(value) for value in qpos)
                    for side, qpos in self.qpos_by_side.items()
                }
            ),
        )


def _qpos_by_side(values: object, *, field_name: str) -> dict[str, list[float]]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{field_name} must contain at least one arm side.")
    result: dict[str, list[float]] = {}
    for raw_side, raw_qpos in values.items():
        side = str(raw_side).strip()
        if not side or side in result:
            raise ValueError(f"{field_name} contains an invalid side {side!r}.")
        qpos = np.asarray(raw_qpos, dtype=float).reshape(-1)
        if qpos.size != 6 or not np.all(np.isfinite(qpos)):
            raise ValueError(
                f"{field_name}[{side!r}] must contain six finite qpos values."
            )
        result[side] = [float(value) for value in qpos]
    return result


def _optional_qpos_by_side(
    values: object,
    *,
    field_name: str,
) -> dict[str, list[float]] | None:
    if values is None:
        return None
    return _qpos_by_side(values, field_name=field_name)


def _finite_gripper_mapping(values: object, *, field_name: str) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    result: dict[str, float] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name).strip()
        value = float(raw_value)
        if (
            not name
            or name in result
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"{field_name} contains an invalid value for {name!r}.")
        result[name] = value
    return result


def _measured_qpos_from_snapshot(
    rig: Any,
    snapshot: Mapping[str, object],
    *,
    sides: tuple[str, ...],
) -> tuple[dict[str, list[float]], int | None]:
    execution = snapshot.get("execution")
    if not isinstance(execution, Mapping):
        execution = {}
    measured_raw = execution.get("measured_qpos_by_side")
    if measured_raw is None:
        get_arm_qpos = getattr(getattr(rig, "scene", None), "get_arm_qpos", None)
        if not callable(get_arm_qpos):
            raise TypeError(
                "Mujoco rig must expose measured qpos in its execution snapshot "
                "or scene.get_arm_qpos()."
            )
        measured_raw = {side: get_arm_qpos(side) for side in sides}
    measured = _qpos_by_side(
        measured_raw,
        field_name="measured_qpos_by_side",
    )
    if set(measured) != set(sides):
        raise ValueError("Measured qpos must cover exactly the requested arm sides.")
    physics_tick_raw = execution.get("physics_tick", snapshot.get("physics_ticks"))
    physics_tick = None if physics_tick_raw is None else int(physics_tick_raw)
    if physics_tick is not None and physics_tick < 0:
        raise ValueError("physics_tick must be non-negative when present.")
    return measured, physics_tick


def capture_mujoco_qpos_measurement(
    rig: Any,
    *,
    sides: tuple[str, ...],
    clock_domain_id: str,
    measured_at: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> MujocoQposMeasurement:
    """Capture one raw MuJoCo qpos sample for any application consumer.

    This seam deliberately carries no policy node, replay cursor, human
    emission, or Station identity.  Those identities belong to the caller.
    """

    normalized_sides = tuple(str(side).strip() for side in sides)
    if (
        not normalized_sides
        or any(not side for side in normalized_sides)
        or len(set(normalized_sides)) != len(normalized_sides)
    ):
        raise ValueError("sides must contain unique non-empty arm names.")
    clock_domain = str(clock_domain_id).strip()
    if not clock_domain:
        raise ValueError("clock_domain_id must be non-empty.")
    timestamp = float(clock() if measured_at is None else measured_at)
    if not math.isfinite(timestamp):
        raise ValueError("measured_at must be finite.")
    snapshot = rig.snapshot()
    if not isinstance(snapshot, Mapping):
        raise TypeError("Mujoco rig snapshot must be a mapping.")
    measured, physics_tick = _measured_qpos_from_snapshot(
        rig,
        snapshot,
        sides=normalized_sides,
    )
    payload = {
        "qpos_by_side": measured,
        "measured_at": timestamp,
        "clock_domain_id": clock_domain,
        "physics_tick": physics_tick,
    }
    content_hash = canonical_content_hash(
        payload,
        domain=MUJOCO_QPOS_MEASUREMENT_HASH_DOMAIN_V1,
    )
    return MujocoQposMeasurement(
        measurement_id=f"mujoco-qpos-measurement:{content_hash}",
        qpos_by_side={side: tuple(values) for side, values in measured.items()},
        measured_at=timestamp,
        clock_domain_id=clock_domain,
        physics_tick=physics_tick,
        semantic_content_hash=content_hash,
    )


def build_mujoco_backend_measurement(
    rig: Any,
    *,
    physics_ticks_after_dispatch: int,
) -> dict[str, object]:
    """Capture accepted, actuator-target, and measured state without conflation.

    The rig snapshot is the primary source because its actuated execution
    section is captured under the rig lock. Direct-qpos debug mode has no
    actuator target, so measured qpos is sampled from the scene and remains
    explicitly labelled as prescribed debug state.
    """

    ticks = int(physics_ticks_after_dispatch)
    if (
        isinstance(physics_ticks_after_dispatch, bool)
        or ticks != physics_ticks_after_dispatch
        or ticks < 0
    ):
        raise ValueError("physics_ticks_after_dispatch must be a non-negative integer.")
    snapshot = rig.snapshot()
    if not isinstance(snapshot, Mapping):
        raise TypeError("Mujoco rig snapshot must be a mapping.")

    accepted = _qpos_by_side(
        snapshot.get("accepted_qcmd_by_side"),
        field_name="accepted_qcmd_by_side",
    )
    measured, physics_tick = _measured_qpos_from_snapshot(
        rig,
        snapshot,
        sides=tuple(accepted),
    )
    if set(measured) != set(accepted):
        raise ValueError("Measured and accepted qpos must cover the same arm sides.")

    execution = snapshot.get("execution")
    if not isinstance(execution, Mapping):
        execution = {}
    actuator_target = _optional_qpos_by_side(
        execution.get("actuator_target_by_side"),
        field_name="actuator_target_qpos_by_side",
    )
    if actuator_target is not None and set(actuator_target) != set(accepted):
        raise ValueError("Actuator targets and accepted qpos must cover the same sides.")

    gripper_snapshot = snapshot.get("gripper_execution")
    gripper_measurement: dict[str, object] | None = None
    if isinstance(gripper_snapshot, Mapping):
        gripper_accepted = _finite_gripper_mapping(
            gripper_snapshot.get("accepted_close_by_gripper", {}),
            field_name="accepted_close_by_gripper",
        )
        gripper_target = _finite_gripper_mapping(
            gripper_snapshot.get("actuator_target_close_by_gripper", {}),
            field_name="actuator_target_close_by_gripper",
        )
        gripper_measured = _finite_gripper_mapping(
            gripper_snapshot.get("measured_close_by_gripper", {}),
            field_name="measured_close_by_gripper",
        )
        if not (
            set(gripper_accepted) == set(gripper_target) == set(gripper_measured)
        ):
            raise ValueError(
                "Gripper accepted, actuator-target, and measured state must cover "
                "the same grippers."
            )
        gripper_measurement = {
            "accepted_close_by_gripper": gripper_accepted,
            "actuator_target_close_by_gripper": gripper_target,
            "measured_close_by_gripper": gripper_measured,
        }

    execution_mode = str(snapshot.get("execution_mode", "unknown"))
    robot_dynamics = str(snapshot.get("robot_dynamics", "unknown"))
    return {
        "schema_version": MUJOCO_BACKEND_MEASUREMENT_VERSION,
        # Compatibility alias retained for existing formal-trace consumers.
        # Its semantics are now measured qpos, never backend-accepted qcmd.
        "arm_qpos_by_side": measured,
        "accepted_qcmd_by_side": accepted,
        "actuator_target_qpos_by_side": actuator_target,
        "measured_qpos_by_side": measured,
        "gripper": gripper_measurement,
        "execution_mode": execution_mode,
        "robot_dynamics": robot_dynamics,
        "physics_tick": physics_tick,
        "measurement_semantics": {
            "accepted_qcmd_by_side": "backend_accepted_command_not_measurement",
            "actuator_target_qpos_by_side": (
                "delayed_servo_target_or_null_when_no_actuator_model"
            ),
            "measured_qpos_by_side": (
                "mujoco_plant_qpos_or_prescribed_qpos_in_debug_mode"
            ),
        },
        "physics_ticks_after_dispatch": ticks,
    }


__all__ = [
    "MUJOCO_BACKEND_MEASUREMENT_VERSION",
    "MUJOCO_QPOS_MEASUREMENT_HASH_DOMAIN_V1",
    "MujocoQposMeasurement",
    "build_mujoco_backend_measurement",
    "capture_mujoco_qpos_measurement",
]
