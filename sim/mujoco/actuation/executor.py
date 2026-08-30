"""MuJoCo adapter for the backend-independent RM65 position actuator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from sim.contracts import TimingSpec
from sim.rm65.actuation import PositionActuationProfile, Rm65PositionActuator

from ..collision_safety import CollisionProjectionResult
from ..control import DirectQposCommand


COLLISION_PROJECTION_MODE_EXECUTOR = "executor"
"""The physical executor performs the full-scene qcmd projection."""

COLLISION_PROJECTION_MODE_UPSTREAM_ADMITTED = "upstream_admitted"
"""The caller already performed the authoritative compound projection."""

_COLLISION_PROJECTION_MODES = frozenset(
    {
        COLLISION_PROJECTION_MODE_EXECUTOR,
        COLLISION_PROJECTION_MODE_UPSTREAM_ADMITTED,
    }
)


def normalize_collision_projection_mode(value: str) -> str:
    """Validate the owner of continuous arm-command collision projection.

    ``executor`` is the compatibility-safe default.  ``upstream_admitted`` is
    deliberately explicit: it is only for callers such as a task kernel that
    has already projected the complete gripper/arm transaction.  Even in that
    mode this adapter keeps the active-contact precheck and post-step hard
    contact latch, so omitting the duplicate projection never disables the
    physical guard.
    """

    mode = str(value).strip().lower()
    if mode not in _COLLISION_PROJECTION_MODES:
        raise ValueError(
            "collision_projection_mode must be one of: "
            f"{', '.join(sorted(_COLLISION_PROJECTION_MODES))}."
        )
    return mode


@dataclass(frozen=True)
class ActuatedPositionTick:
    """One MuJoCo physics tick after the optional 120 Hz transport send."""

    physics_tick: int
    physics_time_sec: float
    command_due: bool
    scheduled_command_time_sec: float | None
    accepted_command_sequence: int
    command_applied: bool
    actuator_target_by_side: Mapping[str, np.ndarray]
    measured_qpos_by_side: Mapping[str, np.ndarray]
    # Safety fields are deliberately additive.  Existing callers that only
    # consume transport/measurement fields therefore keep the same contract,
    # while the physical path now exposes whether the full CAD scene accepted
    # or shortened the command.
    safety_status: str = "unavailable"
    target_modified: bool = False
    projection_alpha: float = 1.0
    minimum_distance_m: float = float("inf")
    collision_pair: tuple[str | None, str | None] = (None, None)
    hard_contact: bool = False


class MujocoActuatedPositionExecutor:
    """Bind reusable RM65 target dynamics to MuJoCo position actuators.

    The accepted-qcmd/delay/filter state machine lives in
    :class:`sim.rm65.actuation.Rm65PositionActuator`.  This adapter only maps
    target vectors to MuJoCo actuator IDs, advances ``mj_step``, and exposes
    measured qpos.
    """

    execution_mode = "actuated_position_servo"
    robot_dynamics = "mujoco_position_actuator"

    def __init__(
        self,
        scene,
        timing: TimingSpec,
        profile: PositionActuationProfile,
        *,
        active_sides: Sequence[str] | None = None,
        collision_projection_mode: str = COLLISION_PROJECTION_MODE_EXECUTOR,
    ) -> None:
        if not isinstance(timing, TimingSpec):
            raise TypeError("timing must be a TimingSpec.")
        self.scene = scene
        self.timing = timing
        self.profile = profile
        self.collision_projection_mode = normalize_collision_projection_mode(
            collision_projection_mode
        )
        model_option = getattr(getattr(scene, "model", None), "opt", None)
        if model_option is not None and not math.isclose(
            float(model_option.timestep),
            timing.physics_timestep_sec,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("scene model timestep does not match TimingSpec.physics_hz.")
        sides = tuple(
            str(side)
            for side in (
                active_sides
                if active_sides is not None
                else tuple(profile.joint_names_by_side)
            )
        )
        if not sides or len(set(sides)) != len(sides):
            raise ValueError("active_sides must be a non-empty unique sequence.")
        unknown = set(sides) - set(profile.joint_names_by_side)
        if unknown:
            raise ValueError(f"active_sides are absent from profile: {sorted(unknown)}")
        missing = set(sides) - set(scene.arm_specs)
        if missing:
            raise KeyError(f"scene is missing actuator side(s): {sorted(missing)}")
        self.active_sides = sides
        self._actuator_ids: dict[str, np.ndarray] = {}
        for side in sides:
            joints = profile.joint_names_by_side[side]
            expected = tuple(scene.arm_specs[side].joint_names)
            if tuple(joints) != expected:
                raise ValueError(
                    f"{side} profile joints {tuple(joints)} do not match "
                    f"scene arm joints {expected}."
                )
            ids: list[int] = []
            for index in range(6):
                name = profile.actuator_name(side, index)
                actuator_id = scene.mujoco.mj_name2id(
                    scene.model,
                    scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
                    name,
                )
                if actuator_id < 0:
                    raise KeyError(
                        f"scene has no position actuator {name!r}; compile it "
                        "with the same PositionActuationProfile."
                    )
                ids.append(int(actuator_id))
            self._actuator_ids[side] = np.asarray(ids, dtype=np.int32)
        self._actuator = Rm65PositionActuator(
            profile,
            command_hz=timing.command_hz,
            physics_hz=timing.physics_hz,
            active_sides=sides,
        )
        self._held_command: DirectQposCommand | None = None
        self._last_safety_result: CollisionProjectionResult | None = None
        self._last_active_collision_check = None
        self._last_admitted_target_check = None
        self._last_safety_status = "unavailable"
        self._hard_contact_latched = False
        self._last_requested_qpos_by_side: dict[str, np.ndarray] = {}

    @property
    def execution_metadata(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode,
            "robot_dynamics": self.robot_dynamics,
            "command_hz": float(self.timing.command_hz),
            "physics_hz": float(self.timing.physics_hz),
            "command_hold": "zero_order_hold_between_command_ticks",
            "transport_acceptance": "immediate_on_120hz_sender_call",
            "target_dynamics_owner": "sim.rm65.actuation.Rm65PositionActuator",
            "collision_projection_mode": self.collision_projection_mode,
            "collision_safety": (
                (
                    "upstream_admitted_full_scene_post_step_hard_contact"
                    if self.collision_projection_mode
                    == COLLISION_PROJECTION_MODE_UPSTREAM_ADMITTED
                    else "full_scene_continuous_projection_and_post_step_hard_contact"
                )
                if bool(getattr(self.scene, "collision_safety_available", False))
                else "unavailable"
            ),
            "profile": self.profile.to_metadata(),
        }

    @property
    def physics_time_sec(self) -> float:
        return self._actuator.physics_time_sec

    @property
    def command_due(self) -> bool:
        return self._actuator.command_due

    def accepted_arm_qpos(self, side: str) -> np.ndarray:
        return self._actuator.accepted_qpos(side)

    def measured_arm_qpos(self, side: str) -> np.ndarray:
        return (
            np.asarray(self.scene.get_arm_qpos(str(side)), dtype=float)
            .reshape(6)
            .copy()
        )

    @property
    def last_safety_result(self) -> CollisionProjectionResult | None:
        return self._last_safety_result

    @property
    def last_active_collision_check(self):
        return self._last_active_collision_check

    @property
    def hard_contact_latched(self) -> bool:
        return bool(self._hard_contact_latched)

    def reset(
        self,
        initial_command: DirectQposCommand,
        *,
        initial_physics_time_sec: float = 0.0,
        initial_state: Mapping[str, object] | None = None,
    ) -> None:
        self._validate_command(initial_command, require_newer_than=None)
        wrapper_identity_fields = {
            "execution_mode",
            "profile",
            "command_hz",
            "physics_hz",
            "collision_projection_mode",
        }
        if isinstance(initial_state, Mapping) and (
            wrapper_identity_fields.intersection(initial_state)
        ):
            missing_identity = sorted(wrapper_identity_fields - set(initial_state))
            if missing_identity:
                raise ValueError(
                    "initial_state is missing MuJoCo executor identity fields: "
                    f"{missing_identity}"
                )
            if initial_state["profile"] != self.profile.to_metadata():
                raise ValueError(
                    "initial_state actuation profile does not match the "
                    "configured MuJoCo executor profile."
                )
            expected_identity = {
                "execution_mode": self.execution_mode,
                "collision_projection_mode": self.collision_projection_mode,
            }
            for name, expected in expected_identity.items():
                if name in initial_state and initial_state[name] != expected:
                    raise ValueError(
                        f"initial_state {name} does not match the configured "
                        "MuJoCo executor."
                    )
            for name, expected in (
                ("command_hz", float(self.timing.command_hz)),
                ("physics_hz", float(self.timing.physics_hz)),
            ):
                if name not in initial_state:
                    continue
                try:
                    observed = float(initial_state[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"initial_state {name} must be finite."
                    ) from exc
                if not math.isfinite(observed) or not math.isclose(
                    observed,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"initial_state {name} does not match the configured "
                        "MuJoCo executor."
                    )
        self._held_command = initial_command
        self._actuator.reset(
            command_sequence=int(initial_command.sequence),
            qpos_by_side=initial_command.qpos_by_side,
            initial_physics_time_sec=initial_physics_time_sec,
            initial_state=initial_state,
        )
        self._last_safety_result = None
        self._last_active_collision_check = None
        self._last_admitted_target_check = None
        self._hard_contact_latched = False
        self._last_requested_qpos_by_side = {
            side: np.asarray(values, dtype=float).reshape(6).copy()
            for side, values in initial_command.qpos_by_side.items()
        }
        projector = self._collision_projector()
        self._last_safety_status = (
            self.collision_projection_mode
            if projector is not None
            else "unavailable"
        )
        if projector is not None:
            self._check_active_contact_or_latch(projector, phase="reset")
            initial_qpos = self._initial_controlled_qpos(initial_command, projector)
            initial_check = projector.check_qpos(self.scene.data, initial_qpos)
            self._last_active_collision_check = initial_check
            if not initial_check.safe:
                # A prepared simulation/HIL scene may intentionally be
                # touching an allowed task surface (distance ~= 0) while it
                # is still outside the *preventive* clearance margin.  That
                # is not a hard penetration and must not make the executor
                # impossible to start; the next command is projected through
                # the explicit soft-clearance recovery path.  Only a NaN or a
                # penetration beyond the hard bound latches the lifecycle.
                distance = float(initial_check.minimum_distance_m)
                hard_limit = float(projector.policy.hard_penetration_m)
                hard_violation = bool(
                    math.isnan(distance) or distance < -hard_limit
                )
                if hard_violation:
                    self._hard_contact_latched = True
                    self._last_safety_status = "invalid_safe_state"
                    raise RuntimeError(
                        "Initial MuJoCo physical command exceeds the hard "
                        "collision bound: "
                        f"{initial_check.to_dict()}"
                    )
                self._last_safety_status = "initial_clearance_violation"
        self._write_ctrl(self._actuator.actuator_target_by_side)

    def step_physics(
        self,
        command: DirectQposCommand | None = None,
    ) -> ActuatedPositionTick:
        """Optionally accept one due 120 Hz qcmd, then advance MuJoCo once."""

        if self._held_command is None:
            raise RuntimeError("reset(initial_command) is required before stepping physics.")
        if self._hard_contact_latched:
            raise RuntimeError(
                "MuJoCo physical execution is latched after forbidden hard "
                "contact; reset/rearm is required before another physics step."
            )
        projector = self._collision_projector()
        if projector is not None:
            self._check_active_contact_or_latch(projector, phase="pre_step")
        due = self.command_due
        # Tick metadata describes this call, while the snapshot below keeps
        # the last accepted projection for post-mortem diagnostics.  Do not
        # leak a previous command's ``projected`` flag into every held
        # physics tick; that made a stable zero-order hold look like a fresh
        # safety intervention to visualizers and downstream metrics.
        tick_safety_result: CollisionProjectionResult | None = None
        tick_safety_status = "held" if projector is not None else "unavailable"
        if due:
            if command is None:
                raise RuntimeError(
                    "A DirectQposCommand is required at the current 120 Hz deadline."
                )
            self._validate_command(command, require_newer_than=self._held_command)
            effective_command = command
            self._last_requested_qpos_by_side = {
                side: np.asarray(values, dtype=float).reshape(6).copy()
                for side, values in command.qpos_by_side.items()
            }
            if (
                projector is not None
                and self.collision_projection_mode
                == COLLISION_PROJECTION_MODE_EXECUTOR
            ):
                self._last_safety_result = None
                self._last_safety_status = "checking"
                effective_command, projection = (
                    self._project_arm_command(command, projector)
                )
                self._last_safety_result = projection
                self._last_admitted_target_check = (
                    None if projection is None else projection.check
                )
                self._last_safety_status = (
                    "unavailable" if projection is None else projection.status
                )
                tick_safety_result = projection
                tick_safety_status = self._last_safety_status
            elif projector is not None:
                # The task kernel has already performed the complete staged
                # projection (including any coupled gripper target).  Do not
                # project the delayed arm transport a second time: that would
                # turn a valid admitted qcmd into a different transaction.
                # Active-contact checks below remain authoritative.
                self._last_safety_result = None
                admitted_target = self._initial_controlled_qpos(command, projector)
                admitted_check = projector.check_qpos(
                    self.scene.data,
                    admitted_target,
                    minimum_clearance_m=-float(
                        projector.policy.hard_penetration_m
                    ),
                )
                self._last_admitted_target_check = admitted_check
                if not admitted_check.safe:
                    self._last_safety_status = "unsafe_upstream_admitted"
                    raise RuntimeError(
                        "Upstream-admitted MuJoCo arm qcmd exceeds the hard "
                        "collision bound: "
                        f"{admitted_check.to_dict()}"
                    )
                self._last_safety_status = self.collision_projection_mode
                tick_safety_status = self.collision_projection_mode
            core_tick = self._actuator.step(
                command_sequence=int(effective_command.sequence),
                qpos_by_side=effective_command.qpos_by_side,
            )
            self._held_command = effective_command
        else:
            if command is not None:
                raise RuntimeError(
                    "DirectQposCommand arrived before the next 120 Hz deadline."
                )
            core_tick = self._actuator.step()
        self._write_ctrl(core_tick.actuator_target_by_side)
        self.scene.mujoco.mj_step(self.scene.model, self.scene.data)
        hard_contact = False
        if projector is not None:
            post_check = projector.oracle.check_contacts(self.scene.data)
            self._last_active_collision_check = post_check
            if not post_check.safe:
                hard_contact = True
                self._hard_contact_latched = True
                self._last_safety_status = "hard_contact"
                raise RuntimeError(
                    "MuJoCo physical execution entered forbidden hard contact "
                    f"after physics step: {post_check.to_dict()}"
                )
            if not due:
                self._last_safety_status = "held"
        else:
            self._last_safety_status = "unavailable"
        check = self._last_active_collision_check
        if check is None or not np.isfinite(float(check.minimum_distance_m)):
            check = self._last_admitted_target_check
        return ActuatedPositionTick(
            physics_tick=core_tick.physics_tick,
            physics_time_sec=core_tick.physics_time_sec,
            command_due=core_tick.command_due,
            scheduled_command_time_sec=core_tick.scheduled_command_time_sec,
            accepted_command_sequence=core_tick.accepted_command_sequence,
            command_applied=core_tick.command_applied,
            actuator_target_by_side={
                side: values.copy()
                for side, values in core_tick.actuator_target_by_side.items()
            },
            measured_qpos_by_side={
                side: self.measured_arm_qpos(side) for side in self.active_sides
            },
            safety_status=tick_safety_status,
            target_modified=(
                False
                if tick_safety_result is None
                else bool(tick_safety_result.modified)
            ),
            projection_alpha=(
                1.0
                if tick_safety_result is None
                else float(tick_safety_result.projection_alpha)
            ),
            minimum_distance_m=(
                float("inf") if check is None else float(check.minimum_distance_m)
            ),
            collision_pair=(
                (None, None)
                if check is None
                else (check.body1_name, check.body2_name)
            ),
            hard_contact=hard_contact,
        )

    def snapshot(self) -> dict[str, object]:
        accepted = {
            side: self._actuator.accepted_qpos(side) for side in self.active_sides
        }
        measured = {
            side: self.measured_arm_qpos(side) for side in self.active_sides
        }
        return {
            **self.execution_metadata,
            **self._actuator.snapshot(),
            "measured_qpos_by_side": {side: values.tolist() for side, values in measured.items()},
            "accepted_minus_measured_qpos_by_side": {
                side: (accepted[side] - measured[side]).tolist()
                for side in self.active_sides
            },
            "last_requested_qcmd_by_side": {
                side: values.tolist()
                for side, values in self._last_requested_qpos_by_side.items()
            },
            "collision_safety_state": {
                "status": self._last_safety_status,
                "hard_contact_latched": self._hard_contact_latched,
                "last_projection": (
                    None
                    if self._last_safety_result is None
                    else self._last_safety_result.to_dict()
                ),
                "last_active_contact": (
                    None
                    if self._last_active_collision_check is None
                    else self._last_active_collision_check.to_dict()
                ),
                "last_admitted_target": (
                    None
                    if self._last_admitted_target_check is None
                    else self._last_admitted_target_check.to_dict()
                ),
            },
        }

    def _collision_projector(self):
        factory = getattr(self.scene, "collision_projector", None)
        if not callable(factory):
            return None
        projector = factory()
        if projector is None:
            return None
        oracle = getattr(projector, "oracle", None)
        if not callable(getattr(oracle, "check_contacts", None)):
            raise TypeError(
                "scene collision projector must expose an oracle.check_contacts "
                "method for actuated execution"
            )
        required_methods = ["check_qpos"]
        if self.collision_projection_mode == COLLISION_PROJECTION_MODE_EXECUTOR:
            required_methods.append("project")
        for method_name in required_methods:
            if not callable(getattr(projector, method_name, None)):
                raise TypeError(
                    "scene collision projector must expose "
                    f"{method_name}() for actuated execution"
                )
        indices = np.asarray(getattr(projector, "controlled_qpos_indices", ()))
        if indices.ndim != 1 or indices.size == 0:
            raise RuntimeError(
                "scene collision projector must expose a non-empty "
                "controlled_qpos_indices layout"
            )
        return projector

    def _initial_controlled_qpos(self, command: DirectQposCommand, projector) -> np.ndarray:
        """Build the full controlled vector used for reset validation."""

        required = (
            "kinematic_controlled_qpos",
            "kinematic_target_vector",
        )
        if any(not callable(getattr(self.scene, name, None)) for name in required):
            if bool(getattr(self.scene, "collision_safety_available", False)):
                raise RuntimeError(
                    "collision-enabled actuated execution requires the scene "
                    "kinematic qpos layout for reset validation"
                )
            expected = int(
                np.asarray(getattr(projector, "controlled_qpos_indices", ())).size
            )
            if expected != 6 * len(self.active_sides):
                raise RuntimeError(
                    "scene/projector controlled-qpos layout is unavailable for "
                    "the configured active sides"
                )
            return np.asarray(
                np.concatenate(
                    [
                        np.asarray(command.qpos_by_side[side], dtype=float)
                        for side in self.active_sides
                    ]
                ),
                dtype=float,
            )
        current = np.asarray(self.scene.kinematic_controlled_qpos(), dtype=float).reshape(-1)
        expected = int(np.asarray(projector.controlled_qpos_indices).size)
        if current.size != expected:
            raise RuntimeError("scene/projector controlled-qpos layout is inconsistent")
        return self.scene.kinematic_target_vector(
            current,
            command.qpos_by_side,
            gripper_close_by_name=command.gripper_close_by_name,
        )

    def _check_active_contact_or_latch(self, projector, *, phase: str) -> None:
        check = projector.oracle.check_contacts(self.scene.data)
        self._last_active_collision_check = check
        if not check.safe:
            self._hard_contact_latched = True
            self._last_safety_status = "hard_contact"
            raise RuntimeError(
                "MuJoCo physical execution found forbidden hard contact "
                f"{phase}: {check.to_dict()}"
            )

    def _project_arm_command(self, command: DirectQposCommand, projector):
        """Project an arm qcmd through the complete CAD scene.

        The position actuator owns accepted qcmd, while the scene owns the
        current measured gripper/object state.  We therefore start the sweep
        at the last accepted arm command but preserve all other controlled
        coordinates from the live scene.  This keeps planner transaction
        bounds valid and still checks the actual CAD geometry continuously.
        """

        required = (
            "kinematic_controlled_qpos",
            "kinematic_target_vector",
            "arm_qpos_from_kinematic_vector",
        )
        if any(not callable(getattr(self.scene, name, None)) for name in required):
            if bool(getattr(self.scene, "collision_safety_available", False)):
                raise RuntimeError(
                    "collision-enabled actuated execution requires the scene "
                    "kinematic qpos layout for continuous arm projection"
                )
            # Small protocol fakes and explicitly collision-disabled legacy
            # callers do not expose the full scene layout.
            return command, None

        current = np.asarray(self.scene.kinematic_controlled_qpos(), dtype=float).reshape(-1)
        if current.size != int(np.asarray(projector.controlled_qpos_indices).size):
            raise RuntimeError("scene/projector controlled-qpos layout is inconsistent")
        accepted_by_side = {
            side: self._actuator.accepted_qpos(side) for side in self.active_sides
        }
        start = self.scene.kinematic_target_vector(current, accepted_by_side)
        # The physical arm path and DM-Tac hand path are separate transports.
        # Keep the gripper coordinates at the live scene state while sweeping
        # the arm command.  Compound task code performs the explicit staged
        # gripper-then-arm projection before publication; merging the delayed
        # hand target again here would turn normal fingertip closure into a
        # second projection and can reject valid arm motion while the hand
        # servo is still catching up.
        requested = self.scene.kinematic_target_vector(
            start,
            command.qpos_by_side,
        )
        start_check = projector.check_qpos(self.scene.data, start)
        if start_check.safe:
            result = projector.project(
                self.scene.data,
                start_qpos=start,
                requested_qpos=requested,
            )
        else:
            # Servo lag can put a measured geometry inside the preventive
            # margin while remaining within the hard penetration bound.  Use
            # the explicit hard-bound relaxation for this one transaction so a
            # valid retreat is not deadlocked by the preventive margin.
            result = projector.project(
                self.scene.data,
                start_qpos=start,
                requested_qpos=requested,
                minimum_clearance_m=-projector.policy.hard_penetration_m,
            )
            if result.status == "invalid_safe_state":
                result = projector.recover_from_clearance_violation(
                    self.scene.data,
                    start_qpos=start,
                    requested_qpos=requested,
                )
        if result.status == "invalid_safe_state":
            self._last_safety_status = "invalid_safe_state"
            raise RuntimeError(
                "MuJoCo physical arm command has no collision-safe segment: "
                f"{result.to_dict()}"
            )
        effective_qpos = {
            side: self.scene.arm_qpos_from_kinematic_vector(result.accepted_qpos, side)
            for side in self.active_sides
        }
        effective = DirectQposCommand(
            sequence=command.sequence,
            qpos_by_side=effective_qpos,
            gripper_close_by_name=command.gripper_close_by_name,
            source=command.source,
        )
        return effective, result

    def _validate_command(
        self,
        command: DirectQposCommand,
        *,
        require_newer_than: DirectQposCommand | None,
    ) -> None:
        if not isinstance(command, DirectQposCommand):
            raise TypeError("command must be a DirectQposCommand.")
        if set(command.qpos_by_side) != set(self.active_sides):
            raise ValueError(
                "qcmd sides must exactly match active_sides: "
                f"expected={sorted(self.active_sides)} "
                f"got={sorted(command.qpos_by_side)}"
            )
        known_grippers = set(getattr(self.scene, "gripper_specs", {}))
        unknown_grippers = sorted(
            set(command.gripper_close_by_name) - known_grippers
        )
        if unknown_grippers:
            raise KeyError(f"command references unknown gripper(s): {unknown_grippers}")
        if (
            require_newer_than is not None
            and command.sequence <= require_newer_than.sequence
        ):
            raise ValueError("qcmd sequence must increase at each 120 Hz send.")

    def _write_ctrl(self, target_by_side: Mapping[str, np.ndarray]) -> None:
        for side, actuator_ids in self._actuator_ids.items():
            self.scene.data.ctrl[actuator_ids] = target_by_side[side]


__all__ = [
    "COLLISION_PROJECTION_MODE_EXECUTOR",
    "COLLISION_PROJECTION_MODE_UPSTREAM_ADMITTED",
    "ActuatedPositionTick",
    "MujocoActuatedPositionExecutor",
    "normalize_collision_projection_mode",
]
