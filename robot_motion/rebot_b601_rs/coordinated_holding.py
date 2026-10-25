"""Receipt-driven J2/J3 pose holding; no I/O or hardware authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .calibration import finite, six
from .command_field_source import CommandFieldSource
from .joint_motion import (
    B601JointMotionTracker,
    B601WindowStopEvidence,
    B601WindowVelocityEstimator,
    window_stationarity_gate_reason,
)

ZERO = (0.0,) * 6
AXES = (1, 2)


@dataclass(frozen=True)
class CoordinatedHoldingConfig:
    """Explicit acquisition parameters, not shared-control defaults."""

    total_assist_cap_nm: float = 1.0
    bias_cap_nm: float = 1.0
    bias_gain_nm_per_rad_sec: float = 20.0
    rise_slew_nm_s: float = 1.0
    fall_slew_nm_s: float = 0.5
    supported_soft_recovery: bool = False
    recovery_kd_nm_s_per_rad: float = 6.0
    cruise_velocity_kp_nm_s_per_rad: float = 0.0
    cruise_velocity_ki_nm_per_rad: float = 0.0
    cruise_velocity_window_sec: float = .2
    cruise_arrival_feedback: bool = False
    command_field_mode: str = "off"

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "command_field_mode":
                if not isinstance(value, str) or value not in {"off", "shadow", "candidate"}:
                    raise ValueError("command field mode must be off, shadow or candidate")
                continue
            if name in {"supported_soft_recovery", "cruise_arrival_feedback"}:
                if type(value) is not bool:
                    raise TypeError("supported soft recovery must be boolean")
                continue
            if name in {"recovery_kd_nm_s_per_rad", "cruise_velocity_kp_nm_s_per_rad", "cruise_velocity_ki_nm_per_rad"}:
                if finite(value, name) < 0:
                    raise ValueError(f"coordinated {name} must be nonnegative")
                continue
            if finite(value, name) <= 0:
                raise ValueError(f"coordinated {name} must be positive")
        if self.bias_cap_nm > self.total_assist_cap_nm:
            raise ValueError("holding bias must fit the combined assist envelope")
        if self.cruise_enabled and not self.motor_local_recovery:
            raise ValueError("cruise velocity feedback requires the explicit motor-local recovery profile")
        if self.cruise_arrival_feedback and not self.cruise_enabled:
            raise ValueError("arrival velocity feedback requires explicit cruise feedback")
        if self.command_field_mode != "off" and not self.cruise_enabled:
            raise ValueError("command field requires explicit cruise feedback")

    @property
    def motor_local_recovery(self):
        return self.supported_soft_recovery and self.recovery_kd_nm_s_per_rad == 0

    @property
    def cruise_enabled(self):
        return self.cruise_velocity_kp_nm_s_per_rad > 0 or self.cruise_velocity_ki_nm_per_rad > 0


class CoordinatedPoseHolding:
    """Compose two existing motion trackers under ONE combined L1 envelope.

    References and bias only commit on an admitted complete owner receipt.
    A probe replaces one axis' entire active demand, not its gravity baseline.
    The other axis keeps its pose feedback and last learned bias.
    """

    def __init__(self, settings, config: CoordinatedHoldingConfig, *, arrival_tolerance_rad=None, command_field_source=None):
        self.settings, self.config = settings, config
        self.motion = settings.staged_assist.motion_control
        if (self.motion is None or not self.motion.responsive_tracking_brake
            or self.motion.continuous_speed_limit_rad_s is None
            or config.total_assist_cap_nm > settings.limits.max_abs_assist_nm
            or max(config.rise_slew_nm_s, config.fall_slew_nm_s) > self.motion.braking_slew_nm_s):
            raise ValueError("coordinated holding must fit resolved native motion and assist limits")
        self.trackers = tuple(B601JointMotionTracker(self.motion) for _ in AXES)
        self.estimators = tuple(B601WindowVelocityEstimator(self.motion.velocity_window_sec) for _ in range(6))
        if config.cruise_enabled and not (self.motion.velocity_window_sec <= config.cruise_velocity_window_sec
            < 255*settings.nominal_control_period_sec):
            raise ValueError("cruise window must fit the native velocity window and bounded observer history")
        self.cruise_estimators = (tuple(B601WindowVelocityEstimator(config.cruise_velocity_window_sec) for _ in AXES)
                                  if config.cruise_enabled else ())
        self.cruise_error_estimators = (tuple(B601WindowVelocityEstimator(config.cruise_velocity_window_sec) for _ in AXES)
                                        if config.cruise_enabled else ())
        self.cruise_velocity = ZERO
        self.cruise_error_velocity = ZERO
        self.q = self.raw = self.velocity = ZERO
        self.goal = None
        self.bias = ZERO
        self._departure_bias = ZERO
        self.last_delta = ZERO
        self.last_time = None
        self.pending = None
        self.frames = ()
        self.fault_frame = None
        self.braking = False
        self.arrival_tolerance_rad = (settings.staged_assist.settle_error_rad
            if arrival_tolerance_rad is None else finite(arrival_tolerance_rad, "coordinated arrival tolerance"))
        if not 0 < self.arrival_tolerance_rad <= settings.staged_assist.settle_error_rad:
            raise ValueError("coordinated arrival tolerance must fit the native arrival envelope")
        self._approach_anchor = ZERO
        self._approach_moving = (False,) * 6
        self._brake_seed = None
        self._support_anchor = None
        self._support_receipt = False
        if (config.command_field_mode == "off") != (command_field_source is None):
            raise ValueError("command field mode and frozen source must be configured together")
        if command_field_source is not None and not isinstance(command_field_source, CommandFieldSource):
            raise TypeError("command field requires the validated source loader")
        if command_field_source is not None and command_field_source.total_assist_cap_nm > config.total_assist_cap_nm:
            raise ValueError("command field source must fit the existing combined assist envelope")
        self.command_field_source = command_field_source
        self.field_component = ZERO
        self._field_joint = None

    def observe(self, now, q, raw, *, idle=False):
        q, raw = six(q, "coordinated position"), six(raw, "coordinated velocity")
        now = finite(now, "coordinated feedback time")
        if self.pending is not None:
            raise ValueError("coordinated previous dispatch not confirmed")
        if type(idle) is not bool or idle and (self.goal is not None or self.bias != ZERO or self.last_delta != ZERO):
            raise ValueError("idle reentry requires an unstarted zero-command controller")
        if self.last_time is not None:
            elapsed = now-self.last_time
            if elapsed <= 0 or elapsed > self.motion.max_control_gap_sec and not idle:
                raise ValueError("coordinated feedback clock or watchdog gap")
            if elapsed > self.motion.max_control_gap_sec:
                # GUI modes can be visited again before starting a batch.
                # Never reuse their old stationary evidence for authorization.
                for estimator in self.estimators:
                    estimator.reset()
                self._reset_cruise_observers()
                self.last_time = None
        dt = self.settings.nominal_control_period_sec if self.last_time is None else now - self.last_time
        self.last_time, self.q, self.raw = now, q, raw
        self.velocity = tuple(e.update(timestamp_sec=now, qpos_rad=p, fallback_rad_s=v)
                              for e, p, v in zip(self.estimators, q, raw, strict=True))
        return dt

    def _reset_cruise_observers(self):
        for estimator in (*self.cruise_estimators, *self.cruise_error_estimators):
            estimator.reset()
        self.cruise_velocity = self.cruise_error_velocity = ZERO

    def quiet(self):
        return window_stationarity_gate_reason(self.estimators, self.raw, self.velocity,
            soft_limit_rad_s=self.settings.staged_assist.settle_velocity_rad_s,
            hard_limit_rad_s=self.settings.limits.max_joint_velocity_rad_s) == "ready"

    def set_goal(self, goal, *, max_reference_velocity_rad_s=None, support_soft_speed_recovery=None):
        goal = six(goal, "coordinated goal")
        if self.pending is not None or self.last_time is None:
            raise ValueError("goal changes require consumed feedback/dispatch")
        if any(abs(goal[i] - self.q[i]) > self.settings.maximum_target_offset_rad for i in AXES):
            raise ValueError("coordinated goal exceeds existing local target envelope")
        speed = (self.motion.max_reference_velocity_rad_s if max_reference_velocity_rad_s is None
                 else finite(max_reference_velocity_rad_s, "coordinated reference speed"))
        if not 0 < speed <= self.motion.max_reference_velocity_rad_s:
            raise ValueError("coordinated reference speed exceeds the resolved ceiling")
        recovery = (self.motion.support_soft_speed_recovery if support_soft_speed_recovery is None
                    else support_soft_speed_recovery)
        if type(recovery) is not bool:
            raise TypeError("coordinated support recovery must be boolean")
        recovery = recovery or self.config.supported_soft_recovery
        if any(tracker.config.max_reference_velocity_rad_s != speed
               or tracker.config.support_soft_speed_recovery != recovery for tracker in self.trackers):
            if not self.quiet():
                raise ValueError("reference parameter changes require fresh six-axis stillness")
            # Only stage-owned parameters change at a consumed, quiet handoff.
            # Keep the actual command and learned support bias continuously.
            self.trackers = tuple(B601JointMotionTracker(replace(self.motion,
                max_reference_velocity_rad_s=speed, support_soft_speed_recovery=recovery)) for _ in AXES)
        self.goal = goal
        # A new leg must resolve motion after this handoff. Do not reuse the
        # preceding leg's direction or deceleration tail for cruise control.
        self._reset_cruise_observers()
        # Retained total support is unchanged; only its model allocation ends.
        self.field_component = ZERO
        self._field_joint = None
        # A consumed command at departure is a handoff prior, not an estimate
        # of the holding force at the destination or a gravity correction.
        self._departure_bias = self.bias
        self.fault_frame = None
        # Proposals are local to this exact goal/pose context, never a model.
        self._approach_anchor = self.q
        self._approach_moving = (False,) * 6
        self._brake_seed = None
        self._support_anchor = None
        self._support_receipt = False
        for tracker, i in zip(self.trackers, AXES, strict=True):
            tracker.start(now=self.last_time, q=self.q[i], target=goal[i])

    def adopt_hold(self, joint, command_nm):
        if type(joint) is not int or joint not in AXES or self.pending is not None:
            raise ValueError("hold transfer requires a consumed J2/J3 receipt")
        value = finite(command_nm, "transferred hold")
        if abs(value) > self.config.bias_cap_nm:
            raise ValueError("transferred hold exceeds the explicit bias envelope")
        # Keep the observed command as an explicit bias, never as gravity.
        bias = list(self.bias)
        bias[joint] = value
        self.bias = tuple(bias)
        departure = list(self._departure_bias)
        departure[joint] = value
        self._departure_bias = tuple(departure)

    def _approach_bias(self, i, frame, dt, moving, anchor, *, point_search=True, point_reacquire=False):
        """Separate resolved motion from optional point-local exploration.

        Half is an exploration proposal toward zero, not a measured equilibrium
        or a signed force bound. Motor-local arrival unloads gradually while
        resolved motion is still assisted, never to a presumed support value.
        Native braking always overrides this path.
        """
        resolution = self.motion.position_resolution_rad or 0.0
        onset = max(2 * resolution, self.settings.staged_assist.motion_onset_confirm_displacement_rad)
        velocity = self.velocity[i]
        quiet = (abs(velocity) <= self.settings.staged_assist.settle_velocity_rad_s
                 and self.estimators[i].position_span_rad <= resolution)
        if quiet:
            moving, anchor = False, self.q[i]
        progress = self.q[i] - anchor
        onset_confirmed = (not moving and abs(progress) >= onset
            and abs(velocity) > self.settings.staged_assist.settle_velocity_rad_s
            and progress * velocity > 0 and self.last_delta[i] * velocity > 0)
        bias, reason = self.bias[i], "retain_bias_during_motion"
        target_error = self.goal[i]-self.q[i] if self.goal is not None else frame.q_reference_rad-self.q[i]
        arrival = (self.config.motor_local_recovery
                   and getattr(frame, "reason", None) in {"target_reached_braking", "calibration_target_tracking"})
        continuous_return = not point_search or point_reacquire
        if (arrival and continuous_return and (moving or onset_confirmed)
            and abs(velocity) > self.settings.staged_assist.settle_velocity_rad_s
            and self.estimators[i].position_span_rad > resolution
            and bias*velocity > 0 and self.last_delta[i]*velocity > 0
            and (frame.requested_torque_nm+bias)*velocity > 0):
            # Support at the destination is unknown. Before crossing it, use
            # the normal fall rate only while assisted motion persists; never
            # substitute a departure value or a fixed half.
            # Outside the arrival band AND above the existing resolved speed
            # gate, old drive continuing away from the target is not a slow
            # support adjustment. Remove it at the existing brake rate. Slow
            # load support keeps its fall rate; opposite drive keeps its rise.
            passed_target = (abs(target_error) > self.arrival_tolerance_rad and target_error*velocity < 0
                and abs(velocity) > self.motion.resolved_soft_velocity_rad_s)
            rate = self.motion.braking_slew_nm_s if passed_target else self.config.fall_slew_nm_s
            reduction = min(abs(bias), rate*dt)
            bias -= reduction*(1 if bias > 0 else -1)
            moving, reason = True, ("arrival_overshoot_drive_unloading" if passed_target else "arrival_feedback_unloading")
        elif onset_confirmed and point_search and not point_reacquire:
            half = self.last_delta[i] / 2
            original = frame.requested_torque_nm + self.bias[i]
            # A point-local half is a proposal, not a measured holding force.
            target = min(half, original) if velocity > 0 else max(half, original)
            bias = target - frame.requested_torque_nm
            moving, reason = True, "resolved_motion_half_command_proposal"
            if target != half:
                reason = "motion_unload_keeps_stronger_native_braking"
        elif (self.config.motor_local_recovery and continuous_return
              and abs(target_error) > resolution and bias*target_error < 0
              and bias*velocity > 0 and abs(velocity) > self.settings.staged_assist.settle_velocity_rad_s
              and self.estimators[i].position_span_rad > resolution):
            # Error reversal may unwind an obsolete drive while moving. It
            # cannot integrate a new opposite drive or claim a measured hold.
            reduction = min(abs(bias), self.config.bias_gain_nm_per_rad_sec*abs(target_error)*dt)
            bias -= reduction*(1 if bias > 0 else -1)
            reason = "moving_target_error_unwinds_bias"
        elif onset_confirmed:
            moving, reason = True, "resolved_motion_continuous_tracking"
        elif quiet and arrival and not point_search and abs(target_error) <= self.arrival_tolerance_rad:
            reason = "arrival_retains_support_without_extra_integration"
        elif quiet:
            error = frame.q_reference_rad - self.q[i]
            if abs(error) > resolution:
                bias += self.config.bias_gain_nm_per_rad_sec * error * dt
            reason = "stationary_bias_refinement"
        return (max(-self.config.bias_cap_nm, min(self.config.bias_cap_nm, bias)),
                moving, anchor, reason)

    def _slew_axis(self, i, target, dt, resolved_motion, *, arrival_unload=False):
        previous = self.last_delta[i]
        rate = (self.motion.braking_slew_nm_s if self.braking else
                self.config.rise_slew_nm_s if abs(target) > abs(previous) else self.config.fall_slew_nm_s)
        reason = "native_braking" if self.braking else "normal_command_slew"
        arrival_reduction = arrival_unload and previous*(previous-target) > 0
        resolved_reduction = (resolved_motion and previous*self.velocity[i] > 0
                              and previous*self.raw[i] > 0 and (target-previous)*self.velocity[i] < 0)
        if not self.braking and (arrival_reduction or resolved_reduction):
            # Fast removal of existing positive work is not permission for
            # fast opposite drive. Cross zero on a later normal-rise frame.
            if target * previous < 0:
                target = 0.0
            rate = self.motion.braking_slew_nm_s
            reason = "arrival_drive_tail_unload" if arrival_reduction else "resolved_working_tail_unload"
        return previous + max(-rate * dt, min(rate * dt, target - previous)), reason, rate

    def _composed_speed_scale(self, i, frame, demand):
        # The planner's factor follows the goal direction. An additive bias
        # can drive AWAY from that goal, including during normal arrival or a
        # supported stop. Govern the complete demand in its own direction,
        # using the same resolved windows and speed envelope, not raw jitter.
        if demand == 0:
            return 1.0
        direction = 1 if demand > 0 else -1
        resolution = self.motion.position_resolution_rad or 0.0
        speed = (max(0.0, direction*self.velocity[i])
                 if self.estimators[i].position_span_rad > resolution else 0.0)
        if (frame.tracking_recent_velocity_rad_s is not None
            and frame.tracking_recent_position_span_rad is not None
            and frame.tracking_recent_position_span_rad > resolution):
            speed = max(speed, direction*frame.tracking_recent_velocity_rad_s)
        limit = self.motion.continuous_speed_limit_rad_s
        scale = max(0.0, min(1.0,
            (limit-speed)/(limit-self.motion.resolved_soft_velocity_rad_s)))
        if frame.continuous_speed_scale is not None and demand*self.velocity[i] > 0:
            scale = min(scale, frame.continuous_speed_scale)
        return scale

    def _cruise_feedback(self, i, frame, dt):
        audit = {"enabled": self.config.cruise_enabled, "active": False, "reason": "disabled"}
        if not self.config.cruise_enabled:
            return 0.0, self.bias[i], audit
        audit["reason"] = "native_transition_owns_command"
        arrival = self.config.cruise_arrival_feedback and frame.reason in {
            "target_reached_braking", "calibration_target_tracking"}
        audit["arrival"] = arrival
        if not arrival and (frame.phase != "track" or frame.reason != "bounded_reference_tracking"):
            return 0.0, self.bias[i], audit
        estimator = self.cruise_estimators[AXES.index(i)]
        relative = self.cruise_error_estimators[AXES.index(i)]
        latest = relative.latest_observation
        reference, velocity = (0.0 if arrival else frame.qvel_reference_rad_s), self.cruise_velocity[i]
        audit.update(reference_velocity_rad_s=reference, measured_velocity_rad_s=velocity,
            aligned_reference_velocity_rad_s=0.0 if arrival else velocity+self.cruise_error_velocity[i],
            observed_at_sec=None if latest is None else latest[0],
            window_span_sec=estimator.span_sec, position_span_rad=estimator.position_span_rad)
        if (not estimator.ready or not relative.ready or latest is None
            or not 0 < self.last_time-latest[0] <= self.motion.max_control_gap_sec
            or estimator.position_span_rad < 2*(self.motion.position_resolution_rad or 0)):
            audit["reason"] = "waiting_for_resolved_cruise_motion"
            return 0.0, self.bias[i], audit
        if not arrival and (reference*velocity <= 0 or reference*(self.goal[i]-self.q[i]) <= 0):
            audit["reason"] = "motion_not_aligned_with_reference"
            return 0.0, self.bias[i], audit
        # Both slopes use the same acknowledged frames/timestamps. Comparing
        # an instantaneous reference with a filtered measurement would charge
        # normal reference acceleration/deceleration into the retained bias.
        # Arrival owns a zero velocity objective. Do not differentiate the
        # native reference jump from the paced trajectory to the endpoint.
        error = -velocity if arrival else self.cruise_error_velocity[i]
        proportional = self.config.cruise_velocity_kp_nm_s_per_rad*error
        increment = self.config.cruise_velocity_ki_nm_per_rad*error*dt
        bias = max(-self.config.bias_cap_nm, min(self.config.bias_cap_nm, self.bias[i]+increment))
        audit.update(active=True, reason="resolved_arrival_velocity_feedback" if arrival else "resolved_cruise_velocity_feedback", error_rad_s=error,
            proportional_request_nm=proportional, requested_bias_increment_nm=increment)
        return proportional, bias, audit

    def step(self, *, sequence, dt, permitted, probe_joint=None, probe_nm=None, point_search=True, point_reacquire=False,
             cruise_joint=None, field_pose_index=None, field_driver_kd=None):
        if type(sequence) is not int or sequence < 0 or type(permitted) is not bool or not 0 < finite(dt, "control interval") <= self.motion.max_control_gap_sec:
            raise ValueError("coordinated step requires valid sequence, permit and control interval")
        if self.pending is not None:
            raise ValueError("coordinated previous dispatch not confirmed")
        if type(point_search) is not bool:
            raise TypeError("coordinated point search must be boolean")
        if type(point_reacquire) is not bool:
            raise TypeError("coordinated point reacquire must be boolean")
        if point_reacquire and (not point_search or not self.config.motor_local_recovery or probe_nm is not None):
            raise ValueError("point reacquire requires motor-local frozen-point return, not a fixed probe")
        if probe_joint is not None and (type(probe_joint) is not int or probe_joint not in AXES):
            raise ValueError("coordinated probes support J2/J3 only")
        if probe_nm is not None and (probe_joint is None or abs(finite(probe_nm, "probe")) > self.config.total_assist_cap_nm):
            raise ValueError("probe must fit the combined assist envelope")
        if cruise_joint is not None and (type(cruise_joint) is not int or cruise_joint not in AXES
            or point_search or point_reacquire or probe_nm is not None):
            raise ValueError("cruise feedback requires continuous J2/J3 tracking without point probes")
        self.fault_frame = None
        proposed_bias, demand = list(self.bias), [0.0] * 6
        anchors, moving = list(self._approach_anchor), list(self._approach_moving)
        brake_seed = self._brake_seed
        approach_reasons = [None] * 6
        support_anchor = self._support_anchor
        supported_braking = False
        support_damping_velocity = [None] * 6
        composed_speed_scales = [1.0] * 6
        cruise_p = [0.0]*6
        cruise_audit = {"enabled": self.config.cruise_enabled, "active": False,
            "joint_name": None if cruise_joint is None else f"joint{cruise_joint+1}",
            "reason": "outside_identification_sweep"}
        field_proposal = [0.0]*6
        field_joint = None
        field_audit = {"mode": self.config.command_field_mode, "predicted": False,
            "reason": "outside_resolved_native_cruise", "prediction_nm": None,
            "requested_increment_nm": 0.0, "previous_component_nm": self.field_component}
        frames = []
        fixed_probes = set()
        if permitted:
            if self.goal is None:
                raise ValueError("coordinated drive requires an admitted goal")
            if max(abs(v) for v in self.raw) > self.settings.limits.max_joint_velocity_rad_s:
                raise ValueError("coordinated raw joint velocity exceeded")
            for tracker, i in zip(self.trackers, AXES, strict=True):
                evidence = B601WindowStopEvidence(timestamp_sec=self.last_time, qpos_rad=self.q,
                    raw_qvel_rad_s=self.raw, window_qvel_rad_s=self.velocity,
                    position_span_rad=tuple(e.position_span_rad for e in self.estimators),
                    window_span_sec=min(e.span_sec for e in self.estimators),
                    raw_velocity_limit_rad_s=self.settings.limits.max_joint_velocity_rad_s, joint_index=i)
                frame = tracker.update(now=self.last_time, q=self.q[i], raw_v=self.raw[i], control_v=self.velocity[i],
                    control_sequence=sequence, authority=1.0,
                    kp=self.motion.tracking_kp_nm_per_rad, kd=self.motion.tracking_kd_nm_s_per_rad,
                    stop_error_rad=self.arrival_tolerance_rad,
                    stop_velocity_rad_s=self.settings.staged_assist.settle_velocity_rad_s,
                    stop_dwell_sec=self.settings.staged_assist.settle_dwell_sec,
                    support_max_velocity_rad_s=max(abs(v) for j, v in enumerate(self.velocity) if j != i),
                    support_raw_max_velocity_rad_s=max(abs(v) for j, v in enumerate(self.raw) if j != i),
                    stop_evidence=evidence, speed_governor_permitted=True,
                    zero_dispatch_confirmed=max(abs(v) for v in self.last_delta) <= 1e-9,
                    calibration_hold_enabled=True,
                    calibration_target_hold=True,
                    calibration_support_dispatch_confirmed=(self._support_receipt
                        if self.config.supported_soft_recovery else None),
                    calibration_probe_nm=probe_nm if i == probe_joint else None)
                frames.append(frame)
                if frame.phase == "authority_exhausted":
                    self.fault_frame = {"joint_name": f"joint{i+1}", **asdict(frame)}
                    raise ValueError(frame.reason)
                if (frame.phase == "soft_hold" and frame.brake_trigger_lag_rad is not None
                    and abs(frame.brake_trigger_lag_rad) > self.motion.max_reference_lag_rad):
                    self.fault_frame = {"joint_name": f"joint{i+1}", **asdict(frame)}
                    raise ValueError("coordinated_reference_lag_stop")
                if (frame.phase == "soft_hold"
                    and abs(frame.q_reference_rad-self.goal[i]) > (self.motion.position_resolution_rad or 0)):
                    self.fault_frame = {"joint_name": f"joint{i+1}", **asdict(frame)}
                    raise ValueError("coordinated_target_stopped_before_goal")
                is_probe = i == probe_joint and probe_nm is not None and frame.reason == "calibration_fixed_command_observation"
                if not is_probe and brake_seed is None:
                    proposed_bias[i], moving[i], anchors[i], approach_reasons[i] = self._approach_bias(
                        i, frame, dt, moving[i], anchors[i], point_search=point_search,
                        point_reacquire=point_reacquire)
                elif is_probe:
                    moving[i], anchors[i] = False, self.q[i]
                    fixed_probes.add(i)
                if i == cruise_joint and not is_probe and brake_seed is None:
                    proportional, cruise_bias, status = self._cruise_feedback(i, frame, dt)
                    cruise_audit.update(status)
                    if status["active"]:
                        # Replace, rather than double-count, a short-window
                        # stationary proposal during resolved slow motion.
                        proposed_bias[i], moving[i] = cruise_bias, True
                        cruise_p[i] = proportional
                        approach_reasons[i] = "cruise_velocity_bias_refinement"
                    elif status.get("arrival") and approach_reasons[i] in {
                        "arrival_feedback_unloading", "arrival_overshoot_drive_unloading", "moving_target_error_unwinds_bias"}:
                        # Keep support while the committed velocity window is
                        # warming. Native brakes and the composed speed cap
                        # still own actual excessive motion.
                        proposed_bias[i] = self.bias[i]
                        approach_reasons[i] = "arrival_observer_warming_retains_support"
                demand[i] = frame.requested_torque_nm + cruise_p[i] + (0.0 if is_probe else proposed_bias[i])
        self.braking = any(f.phase == "brake" and f.reason != "target_reached_braking" for f in frames)
        if (self.command_field_source is not None and permitted and not self.braking
            and cruise_joint is not None and field_pose_index is not None and field_driver_kd is not None):
            i = cruise_joint
            frame = frames[AXES.index(i)]
            if cruise_audit["active"] and frame.phase == "track" and frame.reason == "bounded_reference_tracking":
                # Predict the command needed at the causal reference-window
                # speed. Chasing measured overspeed with the model would add
                # positive velocity feedback; the existing PI owns that error.
                reference_velocity = cruise_audit["aligned_reference_velocity_rad_s"]
                field_audit["query_reference_velocity_rad_s"] = reference_velocity
                field_audit["observed_velocity_rad_s"] = self.cruise_velocity[i]
                prediction = self.command_field_source.predict(pose_index=field_pose_index,
                    joint_name=f"joint{i+1}", qpos_rad=self.q, velocity_rad_s=reference_velocity,
                    active_driver_kd_nm_s_per_rad=field_driver_kd) if reference_velocity*self.cruise_velocity[i] > 0 else {
                        "predicted": False, "reason": "reference_window_direction_unresolved"}
                field_audit.update(predicted=prediction["predicted"], reason=prediction["reason"])
                if prediction["predicted"]:
                    target = prediction["predicted_command_residual_nm"]
                    field_audit["prediction_nm"] = target
                    if self.config.command_field_mode == "candidate":
                        # First engagement transfers the existing support into
                        # model + residual without adding the model twice.
                        previous = self.field_component[i]
                        value = target if self._field_joint != i else max(previous-self.config.rise_slew_nm_s*dt,
                            min(previous+self.config.rise_slew_nm_s*dt, target))
                        increment = value-previous if self._field_joint == i else 0.0
                        field_proposal[i] = value
                        field_joint = i
                        updated = max(-self.config.bias_cap_nm,
                            min(self.config.bias_cap_nm, proposed_bias[i]+increment))
                        field_audit["requested_increment_nm"] = updated-proposed_bias[i]
                        demand[i] += updated-proposed_bias[i]
                        proposed_bias[i] = updated
        if self.braking:
            cruise_p = [0.0]*6
            cruise_audit.update(active=False, reason="native_braking_owns_command",
                                proportional_request_nm=0.0, requested_bias_increment_nm=0.0)
            supported_braking = self.config.supported_soft_recovery and all(
                f.phase != "brake" or f.reason == "target_reached_braking" or f.speed_governor_active
                for f in frames)
            if brake_seed is None:
                # Remember the preceding complete dispatch, not the new brake
                # frame. Resume requires the selected stop receipt and dwell.
                seed = list(self.last_delta)
                for i, frame in zip(AXES, frames, strict=True):
                    if supported_braking:
                        seed[i] = (min(abs(self.bias[i]), abs(self.last_delta[i]))
                                   * (1 if self.bias[i] >= 0 else -1)
                                   if self.bias[i]*self.last_delta[i] > 0 else 0.0)
                    elif (frame.reason in {"resolved_tracking_speed_braking", "soft_velocity_braking"}
                        and self.last_delta[i] * self.velocity[i] > 0
                        and self.estimators[i].position_span_rad >= max(
                            2 * (self.motion.position_resolution_rad or 0.0),
                            self.settings.staged_assist.motion_onset_confirm_displacement_rad)):
                        seed[i] /= 2
                brake_seed = tuple(seed)
            proposed_bias = list(self.bias) if supported_braking else [0.0] * 6
            demand = [0.0] * 6
            if supported_braking and support_anchor is None:
                support_anchor = self.q
            for i, frame in zip(AXES, frames, strict=True):
                # Normal arrival can still request position effort while a
                # peer needs an all-axis zero receipt. Drain that effort too;
                # otherwise the peer cannot leave its genuine soft brake.
                demand[i] = (frame.requested_torque_nm
                             if frame.phase == "brake" and frame.reason != "target_reached_braking"
                             else -self.motion.braking_kd_nm_s_per_rad * self.velocity[i])
                if supported_braking:
                    # Latch support once at entry. Recomputing it from every
                    # composite would repeatedly debit the same spring effort.
                    seed = brake_seed[i]
                    resolved = (not self.config.motor_local_recovery
                                and self.estimators[i].position_span_rad > (self.motion.position_resolution_rad or 0)
                                and abs(self.velocity[i]) > self.settings.staged_assist.settle_velocity_rad_s)
                    restoring = self.motion.tracking_kp_nm_per_rad*(support_anchor[i]-self.q[i])
                    if resolved:
                        if seed*self.velocity[i] > 0:
                            seed = 0.0
                        if restoring*self.velocity[i] > 0:
                            restoring = 0.0
                    # The full window remains authoritative for stop capture.
                    # Its deceleration tail must not keep requesting reverse
                    # effort after the existing recent window has slowed down.
                    recent = frame.tracking_recent_velocity_rad_s
                    damping_v = 0.0 if self.config.motor_local_recovery else self.velocity[i]
                    if not self.config.motor_local_recovery and recent is not None:
                        recent = finite(recent, "recent recovery velocity")
                        damping_v = (min(abs(damping_v), abs(recent)) * (1 if damping_v >= 0 else -1)
                                     if damping_v*recent > 0 else 0.0)
                    support_damping_velocity[i] = damping_v
                    demand[i] = seed + restoring - self.config.recovery_kd_nm_s_per_rad*damping_v
                    approach_reasons[i] = "supported_soft_stop_not_gravity_only"
            if not supported_braking:
                support_anchor = None
        elif (permitted and brake_seed is not None
              and (self.config.supported_soft_recovery
                   or self.quiet() and max(abs(v) for v in self.last_delta) <= 1e-9)):
            # With supported recovery, every native tracker has now left its
            # genuine brake using the same-tick resolution-aware dwell and an
            # exact support receipt. Do not re-gate that transition with the
            # preparation-only quiet test: one-count slope can fail it on the
            # resume tick, then the following tracking receipt clears the
            # support flag and strands the seed forever. Rebase from the last
            # actual command; a conditioned handoff retries this proposal on
            # the next receipt without reusing an expired support flag.
            for i, frame in zip(AXES, frames, strict=True):
                proposed_bias[i] = max(-self.config.bias_cap_nm, min(self.config.bias_cap_nm,
                    (self.last_delta[i] if self.config.supported_soft_recovery else brake_seed[i])
                    - frame.requested_torque_nm))
                demand[i] = frame.requested_torque_nm + proposed_bias[i]
                moving[i], anchors[i] = False, self.q[i]
                approach_reasons[i] = ("resume_confirmed_support" if self.config.supported_soft_recovery
                                       else "post_brake_reduced_command_proposal")
            brake_seed = None
            support_anchor = None
        for i, frame in zip(AXES if permitted else (), frames, strict=True):
            # Fixed-command observations retain their admitted exact plateau.
            # A genuine brake censors the observation and owns the demand.
            if i not in fixed_probes or self.braking:
                composed_speed_scales[i] = self._composed_speed_scale(i, frame, demand[i])
                demand[i] *= composed_speed_scales[i]
                if composed_speed_scales[i] < 1:
                    proposed_bias[i] = self._unwind_only(i, proposed_bias[i])
        norm = sum(abs(v) for v in demand)
        scale = min(1.0, self.config.total_assist_cap_nm / norm) if norm else 1.0
        bias_before_cap = tuple(proposed_bias)
        cap_feedback = (scale < 1 and permitted and not self.braking
            and self.config.motor_local_recovery and not fixed_probes
            and (not point_search or point_reacquire))
        if scale < 1 and not self.braking:
            proposed_bias = [self._unwind_only(i, value) for i, value in enumerate(proposed_bias)]
        command = [0.0] * 6
        uncapped_command = [0.0] * 6
        cap_shortfall = [0.0] * 6
        transition_reasons, transition_rates = [None] * 6, [None] * 6
        if permitted:
            for i, frame in zip(AXES, frames, strict=True):
                target = demand[i] * scale
                # An arrival reduction's actual command tail must keep
                # unloading even across zero-count feedback.
                # The fast path can only reduce toward zero, never reverse.
                arrival_unload = (self.config.motor_local_recovery
                                  and (not point_search or point_reacquire)
                                  and frame.reason in {"target_reached_braking", "calibration_target_tracking"})
                command[i], transition_reasons[i], transition_rates[i] = self._slew_axis(
                    i, target, dt, moving[i], arrival_unload=arrival_unload)
                if cap_feedback:
                    uncapped_command[i] = self._slew_axis(
                        i, demand[i], dt, moving[i], arrival_unload=arrival_unload)[0]
            # The final sum, including a preceding axis' tail, owns one cap.
            norm = sum(abs(v) for v in command)
            if norm > self.config.total_assist_cap_nm:
                # Retain all requested reductions, then share only the freed
                # magnitude. Radial projection could violate a slow release
                # rate on the outgoing support axis during a handoff.
                increases = [i for i in AXES if abs(command[i]) > abs(self.last_delta[i])]
                base = [command[i] if i not in increases else self.last_delta[i] for i in range(6)]
                headroom = max(0.0, self.config.total_assist_cap_nm-sum(abs(v) for v in base))
                requested = sum(abs(command[i])-abs(self.last_delta[i]) for i in increases)
                for i in increases:
                    magnitude = abs(self.last_delta[i]) + (abs(command[i])-abs(self.last_delta[i])) * headroom/requested
                    command[i] = magnitude if command[i] > 0 else -magnitude
                if not self.braking:
                    proposed_bias = [self._unwind_only(i, value) for i, value in enumerate(proposed_bias)]
            if cap_feedback:
                for i, frame in zip(AXES, frames, strict=True):
                    shortfall = command[i]-uncapped_command[i]
                    if (composed_speed_scales[i] == 1 and shortfall*uncapped_command[i] < 0
                        and shortfall*bias_before_cap[i] < 0):
                        # Back-calculate only the final L1 clip, relative to
                        # the SAME native demand and slew rule without it.
                        # Ordinary slowing/unloading is not debited again.
                        # Include the pre-cap proposal so blocked integration
                        # is not also subtracted from the retained old bias.
                        cap_shortfall[i] = shortfall
                        value = bias_before_cap[i]+shortfall
                        if value*bias_before_cap[i] <= 0 or value*self.bias[i] < 0:
                            value = 0.0
                        # Conditional integration can retain growth already
                        # realized inside the cap, not unexecuted slew debt.
                        # Otherwise a roundoff-sized clip freezes growth again.
                        realized = command[i]-frame.requested_torque_nm-cruise_p[i]
                        if abs(value) > abs(self.bias[i]) and not (
                            value*realized >= 0 and abs(value) <= abs(realized)+1e-12
                        ):
                            value = self._unwind_only(i, value)
                        proposed_bias[i] = value
            if cruise_audit["active"]:
                i = cruise_joint
                # PI growth blocked by slew/cap is not an executed bias.
                # Existing-bias reductions and native cap back-calculation
                # remain allowed; they never grant a new opposite drive.
                blocked = (abs(proposed_bias[i]) > abs(self.bias[i])
                    and (proposed_bias[i]-self.bias[i])*(demand[i]-command[i]) > 1e-12)
                if blocked:
                    proposed_bias[i] = self._unwind_only(i, proposed_bias[i])
                cruise_audit.update(integral_growth_blocked=blocked,
                    proposed_bias_after_limits_nm=proposed_bias[i])
        else:
            proposed_bias = [0.0] * 6
            # Revocation invalidates a proposal immediately, even if a final
            # conditioner tail prevents matching this requested zero frame.
            self._brake_seed = None
            self._support_anchor = None
            self._support_receipt = False
            self._departure_bias = ZERO
            self.field_component = ZERO
            self._field_joint = None
            brake_seed = None
            support_anchor = None
            moving, anchors = [False] * 6, list(self.q)
        self.frames = tuple(frames)
        cruise_sample = None
        if self.config.cruise_enabled and permitted and not self.braking and cruise_joint is not None:
            frame = frames[AXES.index(cruise_joint)]
            if (frame.phase == "track" and frame.reason == "bounded_reference_tracking"
                or self.config.cruise_arrival_feedback and frame.reason in {"target_reached_braking", "calibration_target_tracking"}):
                cruise_sample = (cruise_joint, self.last_time, self.q[cruise_joint], frame.q_reference_rad)
        self.pending = (sequence, tuple(command), tuple(proposed_bias), tuple(anchors), tuple(moving), brake_seed,
                        support_anchor, supported_braking, cruise_sample, tuple(field_proposal), field_joint)
        return tuple(command), {"target_qpos_rad": self.goal, "bias_nm": self.bias,
            "max_reference_velocity_rad_s": self.trackers[0].config.max_reference_velocity_rad_s,
            "support_soft_speed_recovery": self.trackers[0].config.support_soft_speed_recovery,
            "proposed_bias_nm": tuple(proposed_bias), "combined_command_nm": tuple(command),
            "departure_bias_nm": self._departure_bias,
            "departure_bias_semantics": "consumed_handoff_prior_not_destination_holding_force",
            "combined_l1_cap_nm": self.config.total_assist_cap_nm,
            "combined_cap_shortfall_nm": tuple(cap_shortfall),
            "saturated": scale < 1, "braking": self.braking,
            "arrival_tolerance_rad": self.arrival_tolerance_rad,
            "point_search_enabled": point_search,
            "point_reacquire_enabled": point_reacquire,
            "cruise_velocity_control": cruise_audit,
            "supported_soft_stop": supported_braking,
            "support_anchor_rad": support_anchor,
            "support_dispatch_confirmed": self._support_receipt,
            "support_damping_owner": "motor_local_profile" if self.config.motor_local_recovery else "software_window",
            "support_damping_velocity_rad_s": support_damping_velocity if supported_braking else None,
            "approach_reasons": approach_reasons, "brake_recovery_proposal_nm": brake_seed,
            "composed_speed_scales": composed_speed_scales,
            "torque_transition_reasons": transition_reasons, "torque_transition_rates_nm_s": transition_rates,
            "axes": [asdict(f) for f in frames], "auto_apply": False,
            **({"command_field": {**field_audit, "proposed_component_nm": tuple(field_proposal),
                "proposed_feedback_bias_nm": tuple(b-f for b, f in zip(proposed_bias, field_proposal))}}
               if self.config.command_field_mode != "off" else {}),
            "quantity": "software_command_not_measured_force"}

    def _unwind_only(self, i, proposed):
        # Saturation blocks accumulation, not a reduction already requested by
        # the position controller. Crossing zero may only unload to zero here.
        previous = self.bias[i]
        if previous*proposed < 0:
            return 0.0
        return proposed if abs(proposed) <= abs(previous) else previous

    def record_dispatch(self, sequence, actual_delta):
        if self.pending is None or sequence != self.pending[0]:
            raise ValueError("coordinated receipt does not match pending frame")
        actual = six(actual_delta, "actual coordinated delta")
        if any(abs(actual[i]) > 1e-9 for i in (0, 3, 4, 5)) or sum(abs(v) for v in actual) > self.config.total_assist_cap_nm + 1e-9:
            raise ValueError("actual coordinated command exceeds combined envelope")
        matching = max(abs(a-b) for a, b in zip(actual, self.pending[1])) <= 1e-9
        if matching:
            self.bias = self.pending[2]
            self.field_component = self.pending[9]
            self._field_joint = self.pending[10]
            self._approach_anchor, self._approach_moving, self._brake_seed = self.pending[3:6]
            self._support_anchor, self._support_receipt = self.pending[6:8]
        else:
            self._support_receipt = False
            self.field_component = ZERO
            self._field_joint = None
        for tracker in self.trackers:
            tracker.record_dispatch(sequence)
        if self.config.cruise_enabled:
            sample = self.pending[8] if matching else None
            if sample is None:
                self._reset_cruise_observers()
            else:
                i, stamp, q, reference = sample
                k = AXES.index(i)
                latest = self.cruise_error_estimators[k].latest_observation
                if latest is not None and stamp <= latest[0]:
                    self._reset_cruise_observers()
                else:
                    for other in range(len(AXES)):
                        if other != k:
                            self.cruise_estimators[other].reset()
                            self.cruise_error_estimators[other].reset()
                    values, errors = [0.0]*6, [0.0]*6
                    values[i] = self.cruise_estimators[k].update(timestamp_sec=stamp, qpos_rad=q, fallback_rad_s=0.0)
                    errors[i] = self.cruise_error_estimators[k].update(timestamp_sec=stamp,
                        qpos_rad=reference-q, fallback_rad_s=0.0)
                    self.cruise_velocity, self.cruise_error_velocity = tuple(values), tuple(errors)
        self.last_delta, self.pending = actual, None
