"""Ideal generalized-effort backend for pre-hardware B601 MIT validation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from hashlib import sha256

import numpy as np

from robot_motion.rebot_b601_rs.mit import (
    B601_RS_JOINT_MIT_FRAME,
    B601MitControlProfile,
    B601MitReference,
    B601MitTorqueBreakdown,
    compute_b601_mit_torque,
)
from robot_motion.rebot_b601_rs.model import (
    B601_RS_ARM_JOINT_NAMES,
    B601PinocchioModel,
)
from sim.contracts import TimingSpec
from sim.mujoco.scene import MujocoScene

from .real2sim import B601Real2SimControllerProfile, B601SimStartupProfile
from .scene import (
    _B601_RS_GRIPPER_LOCK_SOLREF,
    B601_RS_COMMAND_HZ,
    B601_RS_DEFAULT_ARM_QPOS,
    B601_RS_GRIPPER_OPEN_QPOS,
    B601_RS_PHYSICS_HZ,
    B601_RS_SIM_ARM_KEY,
    b601_effort_actuator_name,
    b601_position_actuator_name,
)

_ZERO_SIX = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_DEFAULT_TORQUE_RISE_TIME_SEC = 0.05
_DEFAULT_COMMAND_TIMEOUT_SEC = 0.1
_AUXILIARY_POSITION_TOLERANCE_RAD = 1e-4
_AUXILIARY_VELOCITY_TOLERANCE_RAD_S = 1e-2


def default_b601_mit_timing() -> TimingSpec:
    """Return the 500 Hz reference / 1 kHz effort-servo clock contract."""

    return TimingSpec(
        command_hz=B601_RS_COMMAND_HZ,
        physics_hz=B601_RS_PHYSICS_HZ,
        parameter_source="rebot_b601_rs_ideal_effort_simulation_only",
    )


def _effort_actuator_ids(scene: MujocoScene) -> np.ndarray:
    ids = np.asarray(
        [
            scene.mujoco.mj_name2id(
                scene.model,
                scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
                b601_effort_actuator_name(name),
            )
            for name in B601_RS_ARM_JOINT_NAMES
        ],
        dtype=np.int32,
    )
    if np.any(ids < 0):
        raise ValueError("scene does not contain all six B601 effort actuators")
    return ids


def default_b601_mit_profile(
    scene: MujocoScene,
    *,
    torque_rise_time_sec: float = _DEFAULT_TORQUE_RISE_TIME_SEC,
) -> B601MitControlProfile:
    """Build the conservative, simulation-only gravity/damping profile.

    The hard torque ceiling is copied from the pinned URDF effort fields.  The
    rate bound is explicitly a simulation startup profile, not a hardware
    safety limit or an identified motor capability.
    """

    if not isinstance(scene, MujocoScene):
        raise TypeError("scene must be a MujocoScene")
    rise_time = float(torque_rise_time_sec)
    if not math.isfinite(rise_time) or rise_time <= 0.0:
        raise ValueError("torque_rise_time_sec must be finite and positive")
    actuator_ids = _effort_actuator_ids(scene)
    ranges = np.asarray(scene.model.actuator_ctrlrange[actuator_ids], dtype=float)
    if (
        ranges.shape != (6, 2)
        or not np.all(np.isfinite(ranges))
        or not np.allclose(ranges[:, 0], -ranges[:, 1], atol=1e-12, rtol=0.0)
        or np.any(ranges[:, 1] <= 0.0)
    ):
        raise ValueError("B601 effort actuator ranges must be finite and symmetric")
    torque_limit = ranges[:, 1]
    return B601MitControlProfile(
        name="b601_rs_gravity_damping_simulation_only",
        position_kp_nm_per_rad=_ZERO_SIX,
        damping_kd_nm_s_per_rad=(1.0,) * 6,
        max_abs_torque_nm=tuple(float(value) for value in torque_limit),
        max_torque_rate_nm_s=tuple(float(value / rise_time) for value in torque_limit),
        max_abs_overlay_torque_nm=_ZERO_SIX,
        max_abs_reference_velocity_rad_s=(2.0,) * 6,
        max_abs_state_velocity_rad_s=(2.0,) * 6,
        max_abs_state_acceleration_rad_s2=(100.0,) * 6,
        metadata={
            "parameter_source": (
                "pinned_urdf_effort_ceiling_plus_conservative_pre_hardware_damping_seed"
            ),
            "torque_rise_time_sec": rise_time,
            "hardware_safety_claim": False,
        },
    )


@dataclass(frozen=True)
class B601EffortTick:
    """Audit record for one attempted ideal-effort MuJoCo step.

    ``command_applied`` means the reference passed admission.  Consumers must
    also inspect ``plant_advanced`` and ``fault_reason`` before treating that
    reference as physically simulated.
    """

    physics_tick: int
    physics_time_sec: float
    command_due: bool
    scheduled_command_time_sec: float | None
    command_received: bool
    command_applied: bool
    rejection_reason: str | None
    active_sequence: int | None
    active_source_timestamp_sec: float | None
    active_receive_timestamp_sec: float | None
    active_frame_id: str | None
    active_reference_age_sec: float | None
    reference_timed_out: bool
    control_qpos: tuple[float, ...]
    control_qvel: tuple[float, ...]
    control_qacc: tuple[float, ...]
    measured_qpos: tuple[float, ...]
    measured_qvel: tuple[float, ...]
    measured_qacc: tuple[float, ...]
    measured_time_sec: float
    torque_breakdown: B601MitTorqueBreakdown | None
    clipped_torque_nm: tuple[float, ...]
    applied_torque_nm: tuple[float, ...]
    torque_limited: bool
    slew_limited: bool
    state_valid: bool
    state_within_limits: bool
    driver_faulted: bool
    fault_reason: str | None
    plant_advanced: bool
    host_torque_updated: bool = False
    driver_damping_torque_nm: tuple[float, ...] = _ZERO_SIX
    host_command_torque_nm: tuple[float, ...] = _ZERO_SIX
    driver_position_torque_nm: tuple[float, ...] = _ZERO_SIX
    control_phase: str = "ACTIVE"
    phase_reason: str | None = None


class B601MujocoEffortDriver:
    """Run a software MIT law against six ideal generalized-effort actuators.

    This backend is intentionally not named as a RobStride driver: it validates
    the joint-space law, gravity sign, damping ownership and final effort
    conditioning, but makes no claim about a real motor's internal MIT loop.
    Command admission models one local, wired producer rather than a remote or
    network-synchronized transport.
    """

    execution_mode = "rebot_b601_rs_mujoco_ideal_generalized_effort"
    robot_dynamics = "mujoco_pre_hardware_effort_validation"

    def __init__(
        self,
        scene: MujocoScene,
        pinocchio_model: B601PinocchioModel,
        timing: TimingSpec | None = None,
        *,
        profile: B601MitControlProfile | None = None,
        enable_gravity_compensation: bool = False,
        command_timeout_sec: float = _DEFAULT_COMMAND_TIMEOUT_SEC,
        state_limit_tolerance_rad: float = 1e-4,
        gravity_parity_tolerance_nm: float = 1e-9,
        real2sim_profile: B601Real2SimControllerProfile | None = None,
        startup_profile: B601SimStartupProfile | None = None,
    ) -> None:
        if not isinstance(scene, MujocoScene):
            raise TypeError("scene must be a MujocoScene")
        if not isinstance(pinocchio_model, B601PinocchioModel):
            raise TypeError("pinocchio_model must be a B601PinocchioModel")
        if not isinstance(enable_gravity_compensation, bool):
            raise TypeError("enable_gravity_compensation must be a bool")
        if real2sim_profile is not None:
            if not isinstance(real2sim_profile, B601Real2SimControllerProfile):
                raise TypeError(
                    "real2sim_profile must be a B601Real2SimControllerProfile"
                )
            if not enable_gravity_compensation:
                raise ValueError(
                    "real2sim_profile requires explicit gravity compensation"
                )
        self.real2sim_profile = real2sim_profile
        if startup_profile is not None:
            if not isinstance(startup_profile, B601SimStartupProfile):
                raise TypeError("startup_profile must be a B601SimStartupProfile")
            if real2sim_profile is None:
                raise ValueError("startup_profile requires real2sim_profile")
        self.startup_profile = startup_profile
        if real2sim_profile is not None:
            self.execution_mode = "rebot_b601_rs_mujoco_host_zoh_local_damping"
            self.robot_dynamics = "unchanged_urdf_unidentified_motor_plant"
        self._host_breakdown: B601MitTorqueBreakdown | None = None
        resolved_timing = timing or default_b601_mit_timing()
        if not isinstance(resolved_timing, TimingSpec):
            raise TypeError("timing must be a TimingSpec")
        if not math.isclose(
            float(scene.model.opt.timestep),
            resolved_timing.physics_timestep_sec,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("scene timestep does not match TimingSpec.physics_hz")
        if pinocchio_model.arm_joint_names != B601_RS_ARM_JOINT_NAMES:
            raise ValueError("Pinocchio arm joint order does not match B601-RS")
        try:
            scene_urdf_path = scene.conversion.urdf_path.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise ValueError(
                "scene does not retain a resolvable B601 URDF source"
            ) from exc
        if scene_urdf_path != pinocchio_model.urdf_path:
            raise ValueError(
                "MuJoCo scene and Pinocchio model must use the same B601 URDF"
            )
        try:
            arm_spec = scene.arm_specs[B601_RS_SIM_ARM_KEY]
        except KeyError as exc:
            raise KeyError("scene has no explicit B601-RS arm spec") from exc
        if tuple(arm_spec.joint_names) != B601_RS_ARM_JOINT_NAMES:
            raise ValueError("scene B601-RS arm joint order does not match the model")

        self.scene = scene
        self.pinocchio_model = pinocchio_model
        self._urdf_sha256 = sha256(pinocchio_model.urdf_path.read_bytes()).hexdigest()
        self.timing = resolved_timing
        self.enable_gravity_compensation = enable_gravity_compensation
        self.command_timeout_sec = self._positive(
            command_timeout_sec,
            name="command_timeout_sec",
        )
        self.state_limit_tolerance_rad = self._non_negative(
            state_limit_tolerance_rad,
            name="state_limit_tolerance_rad",
        )
        self.gravity_parity_tolerance_nm = self._positive(
            gravity_parity_tolerance_nm,
            name="gravity_parity_tolerance_nm",
        )
        self._qpos_addresses = np.asarray(
            [scene.joint_qpos_addr(name) for name in B601_RS_ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self._qvel_addresses = np.asarray(
            [scene.joint_dof_addr(name) for name in B601_RS_ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self._joint_ids = np.asarray(
            [scene.joint_id(name) for name in B601_RS_ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self._auxiliary_joint_ids = np.asarray(
            [scene.joint_id(name) for name in B601_RS_GRIPPER_OPEN_QPOS],
            dtype=np.int32,
        )
        self._auxiliary_qpos_addresses = np.asarray(
            [scene.joint_qpos_addr(name) for name in B601_RS_GRIPPER_OPEN_QPOS],
            dtype=np.int32,
        )
        self._auxiliary_qvel_addresses = np.asarray(
            [scene.joint_dof_addr(name) for name in B601_RS_GRIPPER_OPEN_QPOS],
            dtype=np.int32,
        )
        self._auxiliary_open_qpos = np.asarray(
            tuple(B601_RS_GRIPPER_OPEN_QPOS.values()),
            dtype=float,
        )
        self._actuator_ids = _effort_actuator_ids(scene)
        self._gripper_equality_ids = self._verify_effort_scene()
        resolved_profile = profile or default_b601_mit_profile(scene)
        if real2sim_profile is not None and profile is None:
            resolved_profile = replace(
                resolved_profile, damping_kd_nm_s_per_rad=_ZERO_SIX
            )
        if not isinstance(resolved_profile, B601MitControlProfile):
            raise TypeError("profile must be a B601MitControlProfile")
        self._verify_profile(resolved_profile)
        self.profile = resolved_profile
        self._lower = np.asarray(
            scene.model.jnt_range[self._joint_ids, 0],
            dtype=float,
        )
        self._upper = np.asarray(
            scene.model.jnt_range[self._joint_ids, 1],
            dtype=float,
        )
        self._physics_tick = 0
        self._next_command_time_sec = 0.0
        self._hold_qpos = np.zeros(6, dtype=float)
        self._active_reference: B601MitReference | None = None
        self._accepted_at_physics_time_sec: float | None = None
        self._last_accepted_sequence: int | None = None
        self._last_source_timestamp_sec: float | None = None
        self._last_receive_timestamp_sec: float | None = None
        self._last_applied_torque = np.zeros(6, dtype=float)
        self._fault_reason: str | None = None
        self._verified_gravity_mps2 = np.asarray(
            scene.model.opt.gravity,
            dtype=float,
        ).copy()
        self.reset(B601_RS_DEFAULT_ARM_QPOS)

    @staticmethod
    def _positive(value: float, *, name: str) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be finite and positive")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return normalized

    @staticmethod
    def _non_negative(value: float, *, name: str) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be finite and non-negative")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return normalized

    @property
    def physics_time_sec(self) -> float:
        return self._physics_tick / float(self.timing.physics_hz)

    @property
    def command_due(self) -> bool:
        return self.physics_time_sec + 1e-12 >= self._next_command_time_sec

    @property
    def execution_metadata(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode,
            "robot_dynamics": self.robot_dynamics,
            "command_hz": float(self.timing.command_hz),
            "physics_hz": float(self.timing.physics_hz),
            "command_hold": "zero_order_hold_between_reference_ticks",
            "command_timeout_sec": self.command_timeout_sec,
            "transport_scope": "single_host_local_wired_control",
            "network_transport": "out_of_scope",
            "timestamp_contract": "host_local_monotonic_observability",
            "watchdog_clock": "driver_owned_local_acceptance_time",
            "stale_sample_action": ("reject_without_preemption_until_local_watchdog"),
            "unexpected_frame_action": (
                "deactivate_reference_keep_local_gravity_and_damping"
            ),
            "plant_step_owner": "exclusive_driver_mj_step",
            "actuator_semantics": "ideal_generalized_joint_effort_nm",
            "local_feedback_integration": (
                "none"
                if self.real2sim_profile is None
                else "mujoco_native_affine_bias_with_total_force_limit"
            ),
            "robstride_internal_mit_fidelity": "not_claimed",
            "gravity_mps2": tuple(
                float(value) for value in self.scene.model.opt.gravity
            ),
            "gravity_compensation": self.enable_gravity_compensation,
            "gravity_source": "B601PinocchioModel.arm_generalized_gravity",
            "gravity_auxiliary_joint_positions": dict(B601_RS_GRIPPER_OPEN_QPOS),
            "gravity_auxiliary_constraint": "continuous_mujoco_joint_equality",
            "gravity_auxiliary_position_tolerance_rad": (
                _AUXILIARY_POSITION_TOLERANCE_RAD
            ),
            "gravity_auxiliary_velocity_tolerance_rad_s": (
                _AUXILIARY_VELOCITY_TOLERANCE_RAD_S
            ),
            "gravity_parity_tolerance_nm": self.gravity_parity_tolerance_nm,
            "damping_owner": (
                "software_mit_profile_only"
                if self.real2sim_profile is None
                else "host_software_pd_plus_physics_rate_driver_kd_no_passive_damping"
            ),
            "gripper": "fixed_open_continuous_joint_equality",
            "profile": {
                "name": self.profile.name,
                "position_kp_nm_per_rad": self.profile.position_kp_nm_per_rad,
                "damping_kd_nm_s_per_rad": (self.profile.damping_kd_nm_s_per_rad),
                "max_abs_torque_nm": self.profile.max_abs_torque_nm,
                "max_torque_rate_nm_s": self.profile.max_torque_rate_nm_s,
                "max_abs_overlay_torque_nm": (self.profile.max_abs_overlay_torque_nm),
                "haptic_overlay_enabled": any(
                    value > 0.0 for value in self.profile.max_abs_overlay_torque_nm
                ),
                "max_abs_reference_velocity_rad_s": (
                    self.profile.max_abs_reference_velocity_rad_s
                ),
                "max_abs_state_velocity_rad_s": (
                    self.profile.max_abs_state_velocity_rad_s
                ),
                "max_abs_state_acceleration_rad_s2": (
                    self.profile.max_abs_state_acceleration_rad_s2
                ),
                "metadata": dict(self.profile.metadata),
            },
            "final_effort_conditioning": (
                "clip_then_slew_then_clip"
                if self.real2sim_profile is None
                else "host_clip_slew_clip_then_local_feedback_then_total_clip"
            ),
            "sim_control_phase": self._control_phase,
            "startup_profile": None
            if self.startup_profile is None
            else asdict(self.startup_profile),
            "startup_anchor_qpos_rad": tuple(float(v) for v in self._lock_anchor),
            "startup_model": (
                "ideal_complete_send_live_state_only"
                if self.startup_profile is not None
                else None
            ),
            "protective_hold_model": "live_feedback_only_no_cached_transport_recovery",
            "startup_torque_seed": (
                "controller_gravity_before_first_step"
                if self.real2sim_profile is not None
                else "gravity_equilibrium_before_first_step"
                if self.enable_gravity_compensation
                else "zero"
            ),
            "state_guard": (
                "latched_position_velocity_acceleration_and_finite_state_guard"
            ),
            "tick_state_semantics": (
                "control_qpos_qvel_qacc_are_pre_step;"
                "measured_qpos_qvel_qacc_are_post_step"
            ),
            "hardware_safety_claim": False,
            "real2sim_controller": (
                None
                if self.real2sim_profile is None
                else self.real2sim_profile.metadata()
            ),
            "host_torque_hz": (
                float(self.timing.physics_hz)
                if self.real2sim_profile is None
                else float(self.timing.command_hz)
            ),
            "local_damping_hz": float(self.timing.physics_hz),
            "urdf_sha256": self._urdf_sha256,
            "local_damping_fidelity": "physics_cadence_approximation_not_identified_motor_loop",
        }

    def reset(
        self,
        initial_qpos: Sequence[float] = B601_RS_DEFAULT_ARM_QPOS,
    ) -> None:
        """Reset and arm the ideal-effort simulation without advancing it."""

        qpos = np.asarray(initial_qpos, dtype=float)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise ValueError("initial_qpos must contain six finite values")
        if np.any(qpos < self._lower) or np.any(qpos > self._upper):
            raise ValueError("initial_qpos exceeds B601-RS arm limits")
        self.scene.mujoco.mj_resetData(self.scene.model, self.scene.data)
        self.scene.data.qpos[self._qpos_addresses] = qpos
        self.scene.data.qvel[self._qvel_addresses] = 0.0
        self.scene.data.ctrl[:] = 0.0
        self._clear_actuator_effort()
        self._initialize_gripper_open()
        self.scene.forward()
        self._physics_tick = 0
        self._next_command_time_sec = 0.0
        self._hold_qpos = qpos.copy()
        self._active_reference = None
        self._accepted_at_physics_time_sec = None
        self._last_accepted_sequence = None
        self._last_source_timestamp_sec = None
        self._last_receive_timestamp_sec = None
        self._fault_reason = None
        self._host_breakdown = None
        self._control_phase = "ENABLED_UNARMED" if self.startup_profile else "ACTIVE"
        self._phase_reason = None
        self._lock_anchor = qpos.copy()
        self._lock_fraction = 1.0 if self.startup_profile else 0.0
        self._lock_started_at = None
        self._ramp_started_at = None
        self._verify_gravity_configuration(qpos)
        self._verified_gravity_mps2 = np.asarray(
            self.scene.model.opt.gravity,
            dtype=float,
        ).copy()
        initial_torque = self._controller_gravity(qpos, np.zeros(6))
        torque_limit = np.asarray(self.profile.max_abs_torque_nm, dtype=float)
        if np.any(np.abs(initial_torque) > torque_limit + 1e-12):
            raise ValueError("gravity torque exceeds the configured effort limit")
        self._last_applied_torque = initial_torque.copy()
        self._host_command_torque = initial_torque.copy()
        self._host_command_time_sec = 0.0
        self.scene.data.ctrl[self._actuator_ids] = initial_torque
        self.scene.forward()

    def step_physics(
        self,
        reference: B601MitReference | None = None,
    ) -> B601EffortTick:
        """Admit one due reference and attempt one 1 kHz effort step."""

        due = self.command_due
        scheduled_time = self._next_command_time_sec if due else None
        applied = False
        rejection_reason = None
        reference_timed_out = False
        self._latch_state_fault_if_needed()
        qpos = self._measured_qpos()
        qvel = self._measured_qvel()
        qacc = self._measured_qacc()
        if self.startup_profile is not None and due and self._fault_reason is None:
            self._advance_startup(qpos, qvel)

        if reference is not None:
            if not isinstance(reference, B601MitReference):
                raise TypeError("reference must be a B601MitReference")
            if self._fault_reason is not None:
                rejection_reason = "driver_fault_latched"
            elif self._control_phase in {"PROTECTIVE_HOLD", "DISABLED"}:
                rejection_reason = "lifecycle_latched"
            elif not reference.valid:
                rejection_reason = "invalid_sample"
                self._deactivate_reference(qpos)
                if self.startup_profile is not None:
                    self.request_protective_hold("invalid_sample")
            elif reference.frame_id != B601_RS_JOINT_MIT_FRAME:
                # A frame mismatch on the intended wired path is a local
                # configuration/command-contract error, not packet jitter.
                # Relinquish the active reference immediately while retaining
                # the backend-owned gravity and zero-velocity damping fallback.
                rejection_reason = "unexpected_frame"
                self._deactivate_reference(qpos)
                if self.startup_profile is not None:
                    self.request_protective_hold("unexpected_frame")
            elif not due:
                rejection_reason = "command_before_deadline"
            else:
                rejection_reason = self._rejection_reason(reference)
                if rejection_reason is None:
                    self._active_reference = reference
                    self._accepted_at_physics_time_sec = self.physics_time_sec
                    self._last_accepted_sequence = int(reference.sequence)
                    self._last_source_timestamp_sec = float(
                        reference.source_timestamp_sec
                    )
                    self._last_receive_timestamp_sec = float(
                        reference.receive_timestamp_sec
                    )
                    applied = True

        active_age = self._active_reference_age_sec()
        if (
            self._active_reference is not None
            and active_age is not None
            and active_age + 1e-12 >= self.command_timeout_sec
        ):
            self._deactivate_reference(qpos)
            if self.startup_profile is not None:
                self.request_protective_hold("reference_timeout")
            reference_timed_out = True
            active_age = None

        if due and self._fault_reason is None:
            period = 1.0 / float(self.timing.command_hz)
            while self._next_command_time_sec <= self.physics_time_sec + 1e-12:
                self._next_command_time_sec += period

        tick_index = self._physics_tick
        tick_time = self.physics_time_sec
        breakdown: B601MitTorqueBreakdown | None = None
        clipped = np.zeros(6, dtype=float)
        applied_torque = np.zeros(6, dtype=float)
        torque_limited = False
        slew_limited = False
        plant_advanced = False
        host_torque_updated = False
        driver_damping = np.zeros(6)
        driver_position = np.zeros(6)
        local_kp = np.zeros(6)
        local_kd = np.zeros(6)
        active_reference = self._active_reference

        if self._fault_reason is None and self._control_phase != "DISABLED":
            control_reference = active_reference or self._hold_reference()
            try:
                # A revoked reference invalidates the host cache even off cadence.
                # Ideal mode keeps its original physics-rate software controller.
                if self.real2sim_profile is None or due or self._host_breakdown is None:
                    compute_profile = self.profile
                    if self.real2sim_profile is not None:
                        if self._control_phase == "PROTECTIVE_HOLD":
                            compute_profile = replace(
                                self.profile,
                                position_kp_nm_per_rad=_ZERO_SIX,
                                damping_kd_nm_s_per_rad=tuple(
                                    0.0 if kd > 0.0 else 1.0
                                    for kd in self.real2sim_profile.driver_kd_nm_s_per_rad
                                ),
                            )
                        elif self._lock_fraction > 0.0:
                            active_weight = 1.0 - self._lock_fraction
                            compute_profile = replace(
                                self.profile,
                                position_kp_nm_per_rad=tuple(
                                    active_weight * v
                                    for v in self.profile.position_kp_nm_per_rad
                                ),
                                damping_kd_nm_s_per_rad=tuple(
                                    active_weight * v
                                    for v in self.profile.damping_kd_nm_s_per_rad
                                ),
                            )
                            control_reference = replace(
                                control_reference,
                                joint_torque_overlay_nm=tuple(
                                    active_weight * v
                                    for v in control_reference.joint_torque_overlay_nm
                                ),
                            )
                    self._host_breakdown = compute_b601_mit_torque(
                        control_reference,
                        compute_profile,
                        measured_qpos_rad=qpos,
                        measured_qvel_rad_s=qvel,
                        gravity_torque_nm=self._controller_gravity(qpos, qvel),
                    )
                    host_torque_updated = True
                breakdown = self._host_breakdown
                requested = np.asarray(breakdown.requested_torque_nm, dtype=float)
                torque_limit = np.asarray(self.profile.max_abs_torque_nm, dtype=float)
                if self.real2sim_profile is not None:
                    if host_torque_updated:
                        host_clipped = np.clip(requested, -torque_limit, torque_limit)
                        torque_limited = not np.array_equal(requested, host_clipped)
                        dt = self.physics_time_sec - self._host_command_time_sec
                        maximum_delta = (
                            np.asarray(self.profile.max_torque_rate_nm_s) * dt
                        )
                        self._host_command_torque = np.clip(
                            self._host_command_torque
                            + np.clip(
                                host_clipped - self._host_command_torque,
                                -maximum_delta,
                                maximum_delta,
                            ),
                            -torque_limit,
                            torque_limit,
                        )
                        self._host_command_time_sec = self.physics_time_sec
                        slew_limited = not np.array_equal(
                            host_clipped, self._host_command_torque
                        )
                    local_kd = np.asarray(
                        self.real2sim_profile.driver_kd_nm_s_per_rad
                    )
                    if self.startup_profile is not None and self._lock_fraction > 0.0:
                        weight = self._lock_fraction
                        local_kd = local_kd + weight * (
                            np.asarray(self.startup_profile.lock_kd_nm_s_per_rad)
                            - local_kd
                        )
                        local_kp = weight * np.asarray(
                            self.startup_profile.lock_kp_nm_per_rad
                        )
                    driver_position = local_kp * (self._lock_anchor - qpos)
                    driver_damping = -local_kd * qvel
                    # Local motor feedback bypasses the host feedforward slew.
                    requested = (
                        self._host_command_torque + driver_damping + driver_position
                    )
                clipped = np.clip(requested, -torque_limit, torque_limit)
                torque_limited |= not np.array_equal(requested, clipped)
                applied_torque = clipped
                if self.real2sim_profile is None:
                    maximum_delta = (
                        np.asarray(self.profile.max_torque_rate_nm_s, dtype=float)
                        * self.timing.physics_timestep_sec
                    )
                    delta = np.clip(
                        clipped - self._last_applied_torque,
                        -maximum_delta,
                        maximum_delta,
                    )
                    applied_torque = np.clip(
                        self._last_applied_torque + delta,
                        -torque_limit,
                        torque_limit,
                    )
                    slew_limited = not np.array_equal(clipped, applied_torque)
                if not np.all(np.isfinite(applied_torque)):
                    self._fault_reason = "non_finite_computed_torque"
            except (ArithmeticError, RuntimeError, ValueError) as exc:
                if (
                    isinstance(exc, ValueError)
                    and str(exc) == "computed MIT torque contains non-finite values"
                ):
                    self._fault_reason = "non_finite_computed_torque"
                else:
                    self._fault_reason = (
                        f"torque_computation_error:{type(exc).__name__}"
                    )

        if self._fault_reason is None:
            self._last_applied_torque = applied_torque.copy()
            self.scene.data.ctrl[self._actuator_ids] = applied_torque
            if self.real2sim_profile is not None:
                # Native affine feedback lets implicitfast account for the
                # motor's velocity derivative; explicit ctrl PD does not.
                model = self.scene.model
                ids = self._actuator_ids
                model.actuator_biastype[ids] = self.scene.mujoco.mjtBias.mjBIAS_AFFINE
                model.actuator_biasprm[ids, :] = 0.0
                model.actuator_biasprm[ids, 0] = local_kp * self._lock_anchor
                model.actuator_biasprm[ids, 1] = -local_kp
                model.actuator_biasprm[ids, 2] = -local_kd
                force_limit = np.minimum(
                    self.profile.max_abs_torque_nm,
                    model.actuator_ctrlrange[ids, 1],
                )
                model.actuator_forcerange[ids, 0] = -force_limit
                model.actuator_forcerange[ids, 1] = force_limit
                self.scene.data.ctrl[ids] = (
                    self._host_command_torque
                    if self._control_phase != "DISABLED"
                    else 0.0
                )
            self.scene.mujoco.mj_step(self.scene.model, self.scene.data)
            if self.real2sim_profile is not None:
                applied_torque = self.scene.data.actuator_force[
                    self._actuator_ids
                ].copy()
                self._last_applied_torque = applied_torque.copy()
            self._physics_tick += 1
            plant_advanced = True
            self._latch_state_fault_if_needed()
            if self._fault_reason is not None:
                self._clear_actuator_effort()
        else:
            self._last_applied_torque = np.zeros(6, dtype=float)
            self._clear_actuator_effort()

        measured_qpos = self._measured_qpos()
        measured_qvel = self._measured_qvel()
        measured_qacc = self._measured_qacc()
        state_finite, state_within_limits, _reason = self._state_status()
        return B601EffortTick(
            physics_tick=tick_index,
            physics_time_sec=tick_time,
            command_due=due,
            scheduled_command_time_sec=scheduled_time,
            command_received=reference is not None,
            command_applied=applied,
            rejection_reason=rejection_reason,
            active_sequence=(
                None if active_reference is None else active_reference.sequence
            ),
            active_source_timestamp_sec=(
                None
                if active_reference is None
                else active_reference.source_timestamp_sec
            ),
            active_receive_timestamp_sec=(
                None
                if active_reference is None
                else active_reference.receive_timestamp_sec
            ),
            active_frame_id=(
                None if active_reference is None else active_reference.frame_id
            ),
            active_reference_age_sec=active_age,
            reference_timed_out=reference_timed_out,
            control_qpos=tuple(float(value) for value in qpos),
            control_qvel=tuple(float(value) for value in qvel),
            control_qacc=tuple(float(value) for value in qacc),
            measured_qpos=tuple(float(value) for value in measured_qpos),
            measured_qvel=tuple(float(value) for value in measured_qvel),
            measured_qacc=tuple(float(value) for value in measured_qacc),
            measured_time_sec=self.physics_time_sec,
            torque_breakdown=breakdown,
            clipped_torque_nm=tuple(float(value) for value in clipped),
            applied_torque_nm=tuple(float(value) for value in applied_torque),
            torque_limited=torque_limited,
            slew_limited=slew_limited,
            state_valid=state_finite,
            state_within_limits=state_within_limits,
            driver_faulted=self._fault_reason is not None,
            fault_reason=self._fault_reason,
            plant_advanced=plant_advanced,
            host_torque_updated=host_torque_updated,
            driver_damping_torque_nm=tuple(float(v) for v in driver_damping),
            host_command_torque_nm=tuple(
                float(v)
                for v in (
                    self._host_command_torque
                    if self.real2sim_profile is not None
                    else applied_torque
                )
            ),
            driver_position_torque_nm=tuple(float(v) for v in driver_position),
            control_phase=self._control_phase,
            phase_reason=self._phase_reason,
        )

    def request_protective_hold(self, reason: str = "operator_stop") -> None:
        """Latch a live-feedback, six-axis draggable simulation hold.

        Non-finite plant faults still freeze the existing simulator; no cached
        sensor/transport recovery is claimed here. Reset explicitly rearms.
        """
        if self.real2sim_profile is None:
            raise ValueError("protective hold requires real2sim_profile")
        if self._control_phase == "DISABLED":
            return
        self._control_phase = "PROTECTIVE_HOLD"
        self._phase_reason = reason
        self._lock_fraction = 0.0
        self._deactivate_reference(self._measured_qpos())

    def request_disable(self) -> None:
        """Explicitly remove all six simulated actuator efforts until reset."""
        if self.real2sim_profile is None:
            raise ValueError("disable lifecycle requires real2sim_profile")
        self._control_phase = "DISABLED"
        self._phase_reason = "explicit_disable"
        self._lock_fraction = 0.0
        self._deactivate_reference(self._measured_qpos())
        self._host_command_torque[:] = 0.0
        self._last_applied_torque[:] = 0.0
        self._clear_actuator_effort()

    def _clear_actuator_effort(self) -> None:
        """Zero control AND native feedback; ctrl=0 alone is not disabled."""
        self.scene.data.ctrl[self._actuator_ids] = 0.0
        if self.real2sim_profile is not None:
            self.scene.model.actuator_biasprm[self._actuator_ids, :] = 0.0
            self.scene.model.actuator_biastype[self._actuator_ids] = (
                self.scene.mujoco.mjtBias.mjBIAS_NONE
            )

    def _advance_startup(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        profile = self.startup_profile
        if self._control_phase in {"ACTIVE", "PROTECTIVE_HOLD", "DISABLED"}:
            return
        if self._control_phase == "ENABLED_UNARMED":
            self._control_phase = "LOCKING"
            self._lock_started_at = self.physics_time_sec
            return
        if np.any(np.abs(qpos - self._lock_anchor) > profile.max_lock_error_rad):
            self.request_protective_hold("startup_lock_displacement")
            return
        if self._control_phase == "LOCKING":
            self._control_phase = "LOCKED"
            return
        if self._control_phase == "LOCKED":
            if (
                self.physics_time_sec + 1e-12
                < self._lock_started_at + profile.freeze_sec
            ):
                return
            if np.any(np.abs(qvel) > profile.settled_velocity_rad_s):
                self.request_protective_hold("startup_not_settled")
                return
            self._control_phase = "GRAVITY_RAMP"
            self._ramp_started_at = self.physics_time_sec
            return
        elapsed = self.physics_time_sec - self._ramp_started_at
        # Use the same clock tolerance as command admission. Subtracting decimal
        # durations can leave a nominal deadline one ULP short and delay the
        # handoff by a whole host tick. State guards above still apply first.
        complete = elapsed + 1e-12 >= 2.0 * profile.rise_time_sec
        fraction = (
            1.0
            if complete
            else max(
                0.0, min(1.0, (elapsed - profile.rise_time_sec) / profile.rise_time_sec)
            )
        )
        self._lock_fraction = 1.0 - fraction
        self._control_phase = (
            "ACTIVE"
            if fraction >= 1.0
            else (
                "HANDOFF"
                if elapsed + 1e-12 >= profile.rise_time_sec
                else "GRAVITY_RAMP"
            )
        )

    def _verify_effort_scene(self) -> np.ndarray:
        if int(self.scene.model.nu) != 6:
            raise ValueError("B601 effort scene must contain exactly six actuators")
        for name in B601_RS_ARM_JOINT_NAMES:
            position_id = self.scene.mujoco.mj_name2id(
                self.scene.model,
                self.scene.mujoco.mjtObj.mjOBJ_ACTUATOR,
                b601_position_actuator_name(name),
            )
            if position_id >= 0:
                raise ValueError("effort scene must not contain position actuators")
        if not np.all(self.scene.model.jnt_limited[self._joint_ids]):
            raise ValueError("all B601-RS arm joints must be position limited")
        if not np.array_equal(
            self.scene.model.actuator_trnid[self._actuator_ids, 0],
            self._joint_ids,
        ):
            raise ValueError(
                "B601 effort actuators do not map one-to-one to arm joints"
            )
        if not np.all(
            self.scene.model.actuator_trntype[self._actuator_ids]
            == self.scene.mujoco.mjtTrn.mjTRN_JOINT
        ):
            raise ValueError("B601 effort actuators must use joint transmission")
        if not np.all(
            self.scene.model.actuator_dyntype[self._actuator_ids]
            == self.scene.mujoco.mjtDyn.mjDYN_NONE
        ):
            raise ValueError("B601 effort actuators must not have activation dynamics")
        if not np.all(
            self.scene.model.actuator_gaintype[self._actuator_ids]
            == self.scene.mujoco.mjtGain.mjGAIN_FIXED
        ):
            raise ValueError("B601 effort actuators must use fixed gain")
        if not np.all(
            self.scene.model.actuator_biastype[self._actuator_ids]
            == self.scene.mujoco.mjtBias.mjBIAS_NONE
        ):
            raise ValueError("B601 effort actuators must not have actuator bias")
        gain_parameters = np.asarray(
            self.scene.model.actuator_gainprm[self._actuator_ids],
            dtype=float,
        )
        expected_gain_parameters = np.zeros_like(gain_parameters)
        expected_gain_parameters[:, 0] = 1.0
        if not np.allclose(
            gain_parameters,
            expected_gain_parameters,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("B601 effort actuators must have unit fixed gain")
        gear = np.asarray(
            self.scene.model.actuator_gear[self._actuator_ids],
            dtype=float,
        )
        expected_gear = np.zeros_like(gear)
        expected_gear[:, 0] = 1.0
        if not np.allclose(gear, expected_gear, atol=1e-12, rtol=0.0):
            raise ValueError("B601 effort actuators must have unit joint gear")
        if not np.all(self.scene.model.actuator_ctrllimited[self._actuator_ids]):
            raise ValueError("all B601 effort actuators must be control limited")
        if not np.all(self.scene.model.actuator_forcelimited[self._actuator_ids]):
            raise ValueError("all B601 effort actuators must be force limited")
        control_ranges = np.asarray(
            self.scene.model.actuator_ctrlrange[self._actuator_ids],
            dtype=float,
        )
        force_ranges = np.asarray(
            self.scene.model.actuator_forcerange[self._actuator_ids],
            dtype=float,
        )
        for ranges, name in (
            (control_ranges, "control"),
            (force_ranges, "force"),
        ):
            if (
                ranges.shape != (6, 2)
                or not np.all(np.isfinite(ranges))
                or not np.allclose(
                    ranges[:, 0],
                    -ranges[:, 1],
                    atol=1e-12,
                    rtol=0.0,
                )
                or np.any(ranges[:, 1] <= 0.0)
            ):
                raise ValueError(
                    f"B601 effort actuator {name} ranges must be finite and symmetric"
                )
        if not np.allclose(
            control_ranges,
            force_ranges,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "B601 effort actuator force ranges must equal control ranges"
            )
        joint_stiffness = np.asarray(
            self.scene.model.jnt_stiffness[self._joint_ids],
            dtype=float,
        )
        if not np.allclose(joint_stiffness, 0.0, atol=1e-15, rtol=0.0):
            raise ValueError("B601 software MIT requires zero passive joint stiffness")
        for attribute in ("dof_damping", "dof_frictionloss", "dof_armature"):
            values = np.asarray(
                getattr(self.scene.model, attribute)[self._qvel_addresses],
                dtype=float,
            )
            if not np.allclose(values, 0.0, atol=1e-15, rtol=0.0):
                raise ValueError(
                    f"B601 software MIT damping requires zero passive {attribute}"
                )
        equality_type = np.asarray(self.scene.model.eq_type, dtype=np.int32)
        equality_joint = np.asarray(self.scene.model.eq_obj1id, dtype=np.int32)
        equality_other = np.asarray(self.scene.model.eq_obj2id, dtype=np.int32)
        equality_active = np.asarray(self.scene.model.eq_active0, dtype=bool)
        gripper_equality_ids: list[int] = []
        for joint_id, (name, open_qpos) in zip(
            self._auxiliary_joint_ids,
            B601_RS_GRIPPER_OPEN_QPOS.items(),
            strict=True,
        ):
            matches = np.flatnonzero(
                (equality_type == self.scene.mujoco.mjtEq.mjEQ_JOINT)
                & (equality_joint == joint_id)
                & (equality_other == -1)
                & equality_active
            )
            if matches.shape != (1,):
                raise ValueError(f"effort scene must fix auxiliary joint {name!r} open")
            equality_id = int(matches[0])
            gripper_equality_ids.append(equality_id)
            expected_polycoef = np.asarray(
                [open_qpos, 0.0, 0.0, 0.0, 0.0],
                dtype=float,
            )
            if not np.allclose(
                self.scene.model.eq_data[equality_id, :5],
                expected_polycoef,
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    f"auxiliary joint {name!r} equality has the wrong open position"
                )
            if not np.allclose(
                self.scene.model.eq_solref[equality_id],
                _B601_RS_GRIPPER_LOCK_SOLREF,
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    f"auxiliary joint {name!r} equality has an unsafe solver reference"
                )
        return np.asarray(gripper_equality_ids, dtype=np.int32)

    def _verify_profile(self, profile: B601MitControlProfile) -> None:
        actuator_limit = np.minimum(
            np.asarray(
                self.scene.model.actuator_ctrlrange[self._actuator_ids, 1],
                dtype=float,
            ),
            np.asarray(
                self.scene.model.actuator_forcerange[self._actuator_ids, 1],
                dtype=float,
            ),
        )
        profile_limit = np.asarray(profile.max_abs_torque_nm, dtype=float)
        if np.any(profile_limit > actuator_limit + 1e-12):
            raise ValueError("profile torque limit exceeds the effort actuator range")

    def _verify_gravity_configuration(self, qpos: np.ndarray) -> None:
        if not self.enable_gravity_compensation:
            return
        scene_gravity = np.asarray(self.scene.model.opt.gravity, dtype=float)
        model_gravity = np.asarray(self.pinocchio_model.gravity_mps2, dtype=float)
        if not np.allclose(
            scene_gravity,
            model_gravity,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                "MuJoCo and Pinocchio gravity vectors do not match; "
                "gravity compensation is not armed"
            )
        expected = self._gravity_torque(qpos)
        observed = np.asarray(
            self.scene.data.qfrc_bias[self._qvel_addresses],
            dtype=float,
        )
        if not np.allclose(
            expected,
            observed,
            atol=self.gravity_parity_tolerance_nm,
            rtol=1e-9,
        ):
            maximum_error = float(np.max(np.abs(expected - observed)))
            raise ValueError(
                "Pinocchio and MuJoCo generalized gravity do not match: "
                f"max_error_nm={maximum_error:.9g}"
            )

    def _gravity_torque(self, qpos: np.ndarray) -> np.ndarray:
        if not self.enable_gravity_compensation:
            return np.zeros(6, dtype=float)
        return self.pinocchio_model.arm_generalized_gravity(
            qpos,
            auxiliary_joint_positions=B601_RS_GRIPPER_OPEN_QPOS,
        )

    def _controller_gravity(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        if self.real2sim_profile is None:
            return self._gravity_torque(qpos)
        profile = self.real2sim_profile
        gravity = self._gravity_torque(
            qpos + np.asarray(profile.gravity_joint_offsets_rad)
        )
        return profile.compensate(gravity, qvel)

    def _rejection_reason(self, reference: B601MitReference) -> str | None:
        if (
            self._last_accepted_sequence is not None
            and reference.sequence <= self._last_accepted_sequence
        ):
            return "stale_sequence"
        if (
            self._last_source_timestamp_sec is not None
            and reference.source_timestamp_sec < self._last_source_timestamp_sec
        ):
            return "stale_source_timestamp"
        if (
            self._last_receive_timestamp_sec is not None
            and reference.receive_timestamp_sec < self._last_receive_timestamp_sec
        ):
            return "stale_receive_timestamp"
        qpos = np.asarray(reference.qpos_rad, dtype=float)
        if np.any(qpos < self._lower) or np.any(qpos > self._upper):
            return "joint_limit_violation"
        qvel = np.asarray(reference.qvel_rad_s, dtype=float)
        velocity_limit = np.asarray(
            self.profile.max_abs_reference_velocity_rad_s,
            dtype=float,
        )
        if np.any(np.abs(qvel) > velocity_limit):
            return "reference_velocity_limit_violation"
        return None

    def _hold_reference(self) -> B601MitReference:
        return B601MitReference(
            sequence=0,
            source_timestamp_sec=0.0,
            receive_timestamp_sec=0.0,
            qpos_rad=tuple(float(value) for value in self._hold_qpos),
        )

    def _deactivate_reference(self, qpos: np.ndarray) -> None:
        self._host_breakdown = None
        self._hold_qpos = qpos.copy()
        self._active_reference = None
        self._accepted_at_physics_time_sec = None

    def _active_reference_age_sec(self) -> float | None:
        if self._accepted_at_physics_time_sec is None:
            return None
        return max(
            0.0,
            self.physics_time_sec - self._accepted_at_physics_time_sec,
        )

    def _measured_qpos(self) -> np.ndarray:
        return np.asarray(
            self.scene.data.qpos[self._qpos_addresses],
            dtype=float,
        ).copy()

    def _measured_qvel(self) -> np.ndarray:
        return np.asarray(
            self.scene.data.qvel[self._qvel_addresses],
            dtype=float,
        ).copy()

    def _measured_qacc(self) -> np.ndarray:
        return np.asarray(
            self.scene.data.qacc[self._qvel_addresses],
            dtype=float,
        ).copy()

    def _state_status(self) -> tuple[bool, bool, str | None]:
        qpos = self._measured_qpos()
        qvel = self._measured_qvel()
        qacc = self._measured_qacc()
        finite = bool(
            self.scene.state_is_finite()
            and np.all(np.isfinite(qpos))
            and np.all(np.isfinite(qvel))
            and np.all(np.isfinite(qacc))
            and np.all(np.isfinite(self.scene.data.qacc_warmstart))
            and np.all(np.isfinite(self.scene.data.qfrc_applied))
            and np.all(np.isfinite(self.scene.data.xfrc_applied))
        )
        if not finite:
            return False, False, "non_finite_simulated_state"
        lower_excursion = self._lower - qpos
        upper_excursion = qpos - self._upper
        excursion = np.maximum(lower_excursion, upper_excursion)
        limit_index = int(np.argmax(excursion))
        maximum_excursion = float(excursion[limit_index])
        if maximum_excursion > self.state_limit_tolerance_rad:
            direction = (
                "below_lower"
                if lower_excursion[limit_index] >= upper_excursion[limit_index]
                else "above_upper"
            )
            return (
                True,
                False,
                (
                    f"joint_state_limit_violation:"
                    f"{B601_RS_ARM_JOINT_NAMES[limit_index]}:{direction}:"
                    f"excursion_rad={maximum_excursion:.9g}"
                ),
            )
        velocity_limit = np.asarray(
            self.profile.max_abs_state_velocity_rad_s,
            dtype=float,
        )
        velocity_excursion = np.abs(qvel) - velocity_limit
        velocity_index = int(np.argmax(velocity_excursion))
        maximum_velocity_excursion = float(velocity_excursion[velocity_index])
        if maximum_velocity_excursion > 0.0:
            return (
                True,
                False,
                (
                    f"joint_velocity_limit_violation:"
                    f"{B601_RS_ARM_JOINT_NAMES[velocity_index]}:"
                    f"velocity_rad_s={abs(float(qvel[velocity_index])):.9g}"
                ),
            )
        acceleration_limit = np.asarray(
            self.profile.max_abs_state_acceleration_rad_s2,
            dtype=float,
        )
        acceleration_excursion = np.abs(qacc) - acceleration_limit
        acceleration_index = int(np.argmax(acceleration_excursion))
        maximum_acceleration_excursion = float(
            acceleration_excursion[acceleration_index]
        )
        if maximum_acceleration_excursion > 0.0:
            return (
                True,
                False,
                (
                    f"joint_acceleration_limit_violation:"
                    f"{B601_RS_ARM_JOINT_NAMES[acceleration_index]}:"
                    f"acceleration_rad_s2={abs(float(qacc[acceleration_index])):.9g}"
                ),
            )
        return True, True, None

    def _latch_state_fault_if_needed(self) -> bool:
        expected_time_sec = self.physics_time_sec
        # MuJoCo repeatedly adds dt, whereas the scheduler multiplies tick*dt.
        # Bound summation roundoff (gamma_n * sum(abs(dt))) rather than treating
        # accumulated rounding as a foreign step. Never tolerate a full tick.
        n_eps = self._physics_tick * np.finfo(float).eps
        roundoff = (
            n_eps / (1.0 - n_eps) * abs(expected_time_sec)
            if n_eps < 1.0 else math.inf
        )
        time_tolerance_sec = min(
            self.timing.physics_timestep_sec / 4.0,
            max(1e-12, roundoff),
        )
        if self._fault_reason is None and not math.isclose(
            float(self.scene.data.time),
            expected_time_sec,
            rel_tol=0.0,
            abs_tol=time_tolerance_sec,
        ):
            self._fault_reason = "physics_clock_desynchronized"
        if (
            self._fault_reason is None
            and self.enable_gravity_compensation
            and (
                not np.array_equal(
                    np.asarray(self.scene.model.opt.gravity, dtype=float),
                    self._verified_gravity_mps2,
                )
                or not np.array_equal(
                    np.asarray(self.pinocchio_model.gravity_mps2, dtype=float),
                    self._verified_gravity_mps2,
                )
            )
        ):
            self._fault_reason = "gravity_configuration_changed"
        finite, _within_limits, reason = self._state_status()
        if not finite:
            self._fault_reason = "non_finite_simulated_state"
        elif self._fault_reason is None:
            auxiliary_reason = self._auxiliary_constraint_fault_reason()
            self._fault_reason = auxiliary_reason or reason
        return finite

    def _auxiliary_constraint_fault_reason(self) -> str | None:
        if not np.all(self.scene.data.eq_active[self._gripper_equality_ids]):
            return "auxiliary_constraint_changed"
        auxiliary_qpos = np.asarray(
            self.scene.data.qpos[self._auxiliary_qpos_addresses],
            dtype=float,
        )
        auxiliary_qvel = np.asarray(
            self.scene.data.qvel[self._auxiliary_qvel_addresses],
            dtype=float,
        )
        if np.any(
            np.abs(auxiliary_qpos - self._auxiliary_open_qpos)
            > _AUXILIARY_POSITION_TOLERANCE_RAD
        ) or np.any(np.abs(auxiliary_qvel) > _AUXILIARY_VELOCITY_TOLERANCE_RAD_S):
            return "auxiliary_constraint_violation"
        return None

    def _initialize_gripper_open(self) -> None:
        for name, value in B601_RS_GRIPPER_OPEN_QPOS.items():
            self.scene.data.qpos[self.scene.joint_qpos_addr(name)] = float(value)
            self.scene.data.qvel[self.scene.joint_dof_addr(name)] = 0.0


__all__ = [
    "B601EffortTick",
    "B601MujocoEffortDriver",
    "default_b601_mit_profile",
    "default_b601_mit_timing",
]
