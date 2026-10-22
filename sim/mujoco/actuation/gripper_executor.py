"""Physics-rate DM-Tac gripper transport and servo approximation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from sim.rm65.actuation import DmtacPositionActuator

from .gripper_profile import GripperActuationProfile


@dataclass(frozen=True)
class GripperControlTick:
    physics_time_sec: float
    accepted_by_gripper: Mapping[str, bool]
    accepted_close_by_gripper: Mapping[str, float]
    servo_goal_close_by_gripper: Mapping[str, float]
    actuator_target_close_by_gripper: Mapping[str, float]
    actuator_target_rate_by_gripper: Mapping[str, float]
    measured_close_by_gripper: Mapping[str, float]


class MujocoDmtacGripperExecutor:
    """Bind the reusable 20 Hz RM65 gripper core to MuJoCo finger servos.

    This component never advances MuJoCo and never writes qpos/qvel. The parent
    MuJoCo rig remains the sole ``MjData`` owner and calls
    :meth:`before_physics_step` immediately before its single ``mj_step``.
    """

    execution_mode = "dmtac_fitted_position_servo_v2"

    def __init__(self, scene, profile: GripperActuationProfile) -> None:
        self.scene = scene
        self.profile = profile
        missing = set(profile.joint_names_by_gripper) - set(scene.gripper_specs)
        if missing:
            raise KeyError(f"scene is missing configured grippers: {sorted(missing)}")
        self._actuator_ids: dict[str, np.ndarray] = {}
        for gripper_name, joint_names in profile.joint_names_by_gripper.items():
            spec = scene.gripper_specs[gripper_name]
            if tuple(spec.joint_names) != tuple(joint_names):
                raise ValueError(
                    f"{gripper_name} scene/profile joint mismatch: "
                    f"scene={tuple(spec.joint_names)} profile={tuple(joint_names)}"
                )
            if (
                tuple(spec.open_qpos) != profile.open_qpos_by_gripper[gripper_name]
                or tuple(spec.closed_qpos) != profile.closed_qpos_by_gripper[gripper_name]
            ):
                raise ValueError(f"{gripper_name} scene/profile operating range mismatch.")
            ids: list[int] = []
            for index in range(len(joint_names)):
                actuator_name = profile.actuator_name(gripper_name, index)
                actuator_id = scene.mujoco.mj_name2id(
                    scene.model,
                    scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
                    actuator_name,
                )
                if actuator_id < 0:
                    raise KeyError(
                        f"scene has no gripper actuator {actuator_name!r}; compile "
                        "with the same GripperActuationProfile."
                    )
                ids.append(int(actuator_id))
            self._actuator_ids[gripper_name] = np.asarray(ids, dtype=np.int32)
        self._gripper_by_side = {
            side: name for name, side in profile.side_by_gripper.items()
        }
        self._position_actuator = DmtacPositionActuator(
            profile.normalized_position_profile()
        )
        self._accepted_close_by_gripper: dict[str, float] = {}
        self._servo_goal_close_by_gripper: dict[str, float] = {}
        self._actuator_target_close_by_gripper: dict[str, float] = {}
        self._actuator_target_rate_by_gripper: dict[str, float] = {}
        self._last_status_by_gripper: dict[str, str] = {}
        self._latest_command_by_gripper: dict[str, tuple[int, float | None]] = {}
        self._last_physics_time_sec = 0.0
        self._reset = False

    @property
    def execution_metadata(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode,
            "command_hz": self.profile.command_hz,
            "command_hold": "hold_last_target",
            "profile": self.profile.to_metadata(),
        }

    def reset(
        self,
        initial_close_by_gripper: Mapping[str, float],
        *,
        initial_physics_time_sec: float = 0.0,
        initial_state: Mapping[str, object] | None = None,
    ) -> None:
        expected = set(self.profile.joint_names_by_gripper)
        if set(initial_close_by_gripper) != expected:
            raise ValueError(
                "initial gripper close mapping must exactly match profile: "
                f"expected={sorted(expected)} got={sorted(initial_close_by_gripper)}"
            )
        initial = {
            name: self._validate_close(value, name=name)
            for name, value in initial_close_by_gripper.items()
        }
        core_state = initial_state
        if isinstance(initial_state, Mapping) and "core" in initial_state:
            required_wrapper_fields = (
                "physics_time_sec",
                "profile",
                "actuator_target_close_by_gripper",
            )
            missing = [
                name for name in required_wrapper_fields if name not in initial_state
            ]
            if missing:
                raise ValueError(
                    "initial_state is missing causal gripper wrapper fields: "
                    f"{missing}"
                )
            raw_core = initial_state["core"]
            if not isinstance(raw_core, Mapping):
                raise TypeError("initial_state core must be a mapping.")
            if initial_state["profile"] != self.profile.to_metadata():
                raise ValueError(
                    "initial_state gripper profile does not match the "
                    "configured MuJoCo gripper profile."
                )
            raw_targets = initial_state["actuator_target_close_by_gripper"]
            if not isinstance(raw_targets, Mapping) or set(raw_targets) != expected:
                raise ValueError(
                    "initial_state actuator target mapping must exactly match "
                    "the configured grippers."
                )
            for name in expected:
                try:
                    saved_target = float(raw_targets[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"initial_state actuator target for {name!r} is invalid."
                    ) from exc
                if not math.isfinite(saved_target) or not math.isclose(
                    saved_target,
                    initial[name],
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        "initial_state actuator target does not match "
                        f"initial_close_by_gripper for {name!r}."
                    )
            raw_time = initial_state.get("physics_time_sec")
            try:
                saved_time = float(raw_time)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "initial_state physics_time_sec must be finite."
                ) from exc
            if not math.isfinite(saved_time) or not math.isclose(
                saved_time,
                float(initial_physics_time_sec),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "initial_state physics_time_sec does not match the "
                    "requested initial_physics_time_sec."
                )
            core_state = raw_core
        self._position_actuator.reset(
            {
                self.profile.side_by_gripper[name]: value
                for name, value in initial.items()
            },
            initial_physics_time_sec=initial_physics_time_sec,
            initial_state=core_state,
        )
        # A resumed executor may restore queued/delayed targets and a
        # non-zero servo state.  Do not overwrite that causal state with the
        # bootstrap values after the core reset; derive every public cache
        # from the restored core instead.  This also keeps ``ctrl`` and the
        # snapshot metadata consistent when the saved actuator target differs
        # from the measured close position.
        core_snapshot = self._position_actuator.snapshot()
        self._accepted_close_by_gripper = self._map_by_gripper(
            core_snapshot["accepted_target_by_side"]
        )
        self._servo_goal_close_by_gripper = self._map_by_gripper(
            core_snapshot["servo_goal_by_side"]
        )
        self._actuator_target_close_by_gripper = self._map_by_gripper(
            core_snapshot["actuator_position_by_side"]
        )
        self._actuator_target_rate_by_gripper = self._map_by_gripper(
            core_snapshot["actuator_rate_by_side"]
        )
        self._last_status_by_gripper = {name: "reset" for name in expected}
        self._latest_command_by_gripper = {name: (0, None) for name in expected}
        self._last_physics_time_sec = float(initial_physics_time_sec)
        self._reset = True
        self._write_ctrl()

    def latest_command(self, gripper_name: str) -> tuple[int, float | None]:
        """Return queued intent, not an internal HOLD target; the rig owns locking."""
        return self._latest_command_by_gripper[gripper_name]

    def queue_close_targets(self, close_by_gripper: Mapping[str, float]) -> int:
        if not self._reset:
            raise RuntimeError("reset(initial_close_by_gripper) is required first.")
        unknown = set(close_by_gripper) - set(self.profile.joint_names_by_gripper)
        if unknown:
            raise KeyError(f"unknown gripper target(s): {sorted(unknown)}")
        validated = {
            name: self._validate_close(value, name=name)
            for name, value in close_by_gripper.items()
        }
        targets = {self.profile.side_by_gripper[name]: value for name, value in validated.items()}
        sequence = self._position_actuator.queue_targets(targets)
        for name, target in validated.items():
            self._latest_command_by_gripper[name] = (sequence, target)
            self._last_status_by_gripper[name] = "queued"
        return sequence

    def hold_targets(self) -> None:
        """Discard queued transport work and freeze the current actuator target."""

        self._position_actuator.hold_targets()
        snapshot = self._position_actuator.snapshot()
        self._accepted_close_by_gripper = self._map_by_gripper(
            snapshot["accepted_target_by_side"]
        )
        self._servo_goal_close_by_gripper = self._map_by_gripper(
            snapshot["servo_goal_by_side"]
        )
        self._actuator_target_close_by_gripper = self._map_by_gripper(
            snapshot["actuator_position_by_side"]
        )
        self._actuator_target_rate_by_gripper = self._map_by_gripper(
            snapshot["actuator_rate_by_side"]
        )
        for name in self._last_status_by_gripper:
            self._last_status_by_gripper[name] = "held"
        self._write_ctrl()

    def before_physics_step(
        self,
        *,
        physics_time_sec: float,
        timestep_sec: float,
    ) -> GripperControlTick:
        if not self._reset:
            raise RuntimeError("reset(initial_close_by_gripper) is required first.")
        now = float(physics_time_sec)
        dt = float(timestep_sec)
        if not math.isfinite(now) or now < 0.0:
            raise ValueError("physics_time_sec must be finite and non-negative.")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("timestep_sec must be finite and positive.")
        core_tick = self._position_actuator.step(
            physics_time_sec=now,
            timestep_sec=dt,
        )
        # ``before_physics_step`` integrates the complete servo state for the
        # interval [now, now + dt).  Bind the resumable snapshot to the end of
        # that interval, which is the time MuJoCo reaches after ``mj_step``.
        self._last_physics_time_sec = now + dt
        accepted = self._map_by_gripper(core_tick.accepted_by_side)
        self._accepted_close_by_gripper = self._map_by_gripper(
            core_tick.accepted_target_by_side
        )
        self._servo_goal_close_by_gripper = self._map_by_gripper(
            core_tick.servo_goal_by_side
        )
        self._actuator_target_close_by_gripper = self._map_by_gripper(
            core_tick.actuator_position_by_side
        )
        self._actuator_target_rate_by_gripper = self._map_by_gripper(
            core_tick.actuator_rate_by_side
        )
        for name, was_accepted in accepted.items():
            if was_accepted:
                self._last_status_by_gripper[name] = "accepted_delayed"
        self._write_ctrl()
        return GripperControlTick(
            physics_time_sec=now,
            accepted_by_gripper=accepted,
            accepted_close_by_gripper=dict(self._accepted_close_by_gripper),
            servo_goal_close_by_gripper=dict(self._servo_goal_close_by_gripper),
            actuator_target_close_by_gripper=dict(
                self._actuator_target_close_by_gripper
            ),
            actuator_target_rate_by_gripper=dict(
                self._actuator_target_rate_by_gripper
            ),
            measured_close_by_gripper=self.measured_close_by_gripper(),
        )

    def accepted_close(self, gripper_name: str) -> float:
        return float(self._accepted_close_by_gripper[str(gripper_name)])

    def measured_close_by_gripper(self) -> dict[str, float]:
        return {
            name: float(self.scene.get_gripper_close(name))
            for name in self.profile.joint_names_by_gripper
        }

    def planned_close_by_gripper(self) -> dict[str, float]:
        """Return the latest normalized close target for collision preview.

        The arm executor performs its continuous full-scene projection before
        the next MuJoCo step.  A gripper command is transported on an
        independent 20 Hz clock, so the arm side needs a read-only view of
        the target that is pending or already accepted by that transport.  We
        deliberately expose no mutable core state and never advance the
        gripper here.
        """

        core = self._position_actuator.snapshot()
        accepted_by_side = dict(core["accepted_target_by_side"])
        pending_by_side = dict(core["pending"])
        by_gripper: dict[str, float] = {}
        for name, side in self.profile.side_by_gripper.items():
            pending = pending_by_side.get(side)
            value = (
                pending["target"]
                if isinstance(pending, Mapping)
                else accepted_by_side[side]
            )
            by_gripper[name] = float(value)
        return by_gripper

    def snapshot(self) -> dict[str, object]:
        core = self._position_actuator.snapshot()
        return {
            **self.execution_metadata,
            "physics_time_sec": float(self._last_physics_time_sec),
            "core": core,
            "pending_close_by_gripper": self._map_by_gripper(
                {
                    side: values["target"]
                    for side, values in core["pending"].items()
                }
            ),
            "accepted_close_by_gripper": dict(self._accepted_close_by_gripper),
            "servo_goal_close_by_gripper": dict(self._servo_goal_close_by_gripper),
            "actuator_target_close_by_gripper": dict(
                self._actuator_target_close_by_gripper
            ),
            "actuator_target_rate_by_gripper": dict(
                self._actuator_target_rate_by_gripper
            ),
            "measured_close_by_gripper": self.measured_close_by_gripper(),
            "pending_delayed_targets": core["pending_delayed_targets"],
            "measured_max_joint_speed_rad_sec_by_gripper": {
                name: float(np.max(np.abs(self.scene.data.qvel[
                    [self.scene.joint_dof_addr(joint) for joint in joints]])))
                for name, joints in self.profile.joint_names_by_gripper.items()
            },
            "last_status_by_gripper": dict(self._last_status_by_gripper),
        }

    def _map_by_gripper(self, values_by_side: Mapping[str, object]) -> dict:
        return {
            self._gripper_by_side[side]: value
            for side, value in values_by_side.items()
        }

    @staticmethod
    def _validate_close(value: float, *, name: str) -> float:
        close = float(value)
        if not math.isfinite(close) or not 0.0 <= close <= 1.0:
            raise ValueError(f"{name} close target must be finite and in [0, 1].")
        return close

    def _write_ctrl(self) -> None:
        for name, actuator_ids in self._actuator_ids.items():
            self.scene.data.ctrl[actuator_ids] = self.profile.qpos_from_close(
                name,
                self._actuator_target_close_by_gripper[name],
            )


__all__ = ["GripperControlTick", "MujocoDmtacGripperExecutor"]
