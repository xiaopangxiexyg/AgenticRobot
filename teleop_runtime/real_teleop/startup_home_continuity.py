"""Fail-closed admission for the first real position command after runtime Home.

The gate observes the final admitted joint command after retargeting and the
ordinary command filters/guards.  It never changes IK, candidate certificates,
or qpos values.  Until a short run of candidates remains close to the atomic
runtime-Home snapshot, dispatch is simply withheld and the session remains
ARMED.  This keeps the 120 Hz position worker anchored at the qpos installed by
``reset_motion_state`` instead of introducing a synthetic qref with stale
candidate metadata.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_motion.pose import pose_delta_metrics
from robot_motion.runtime import TeleopCommand, TeleopObservation


DEFAULT_STARTUP_HOME_MAX_JOINT_DELTA_DEG = 0.75
DEFAULT_STARTUP_HOME_MAX_TCP_TRANSLATION_M = 0.003
DEFAULT_STARTUP_HOME_MAX_TCP_ROTATION_DEG = 2.0
DEFAULT_STARTUP_HOME_REQUIRED_CONSECUTIVE_FRAMES = 3


@dataclass(frozen=True)
class StartupHomeContinuityConfig:
    active_sides: tuple[str, ...] | Sequence[str]
    max_joint_delta_deg: float = DEFAULT_STARTUP_HOME_MAX_JOINT_DELTA_DEG
    max_tcp_translation_m: float = DEFAULT_STARTUP_HOME_MAX_TCP_TRANSLATION_M
    max_tcp_rotation_deg: float = DEFAULT_STARTUP_HOME_MAX_TCP_ROTATION_DEG
    required_consecutive_frames: int = (
        DEFAULT_STARTUP_HOME_REQUIRED_CONSECUTIVE_FRAMES
    )

    def __post_init__(self) -> None:
        sides = tuple(str(side).strip() for side in self.active_sides)
        if not sides or any(side not in {"left", "right"} for side in sides):
            raise ValueError("active_sides must contain left and/or right.")
        if len(set(sides)) != len(sides):
            raise ValueError("active_sides must not contain duplicates.")
        finite_positive = {
            "max_joint_delta_deg": self.max_joint_delta_deg,
            "max_tcp_translation_m": self.max_tcp_translation_m,
            "max_tcp_rotation_deg": self.max_tcp_rotation_deg,
        }
        for name, value in finite_positive.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and > 0.")
        if int(self.required_consecutive_frames) <= 0:
            raise ValueError("required_consecutive_frames must be positive.")
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(
            self,
            "required_consecutive_frames",
            int(self.required_consecutive_frames),
        )


class StartupHomeContinuityGate:
    """One-shot Home-to-first-qref continuity admission.

    ``arm_from_home`` starts a new epoch.  ``admit`` returns ``False`` without
    faulting or dispatching while the final qref is outside the declared Home
    neighbourhood.  Once the configured consecutive-frame requirement is met,
    the gate releases for the rest of that epoch.  A later runtime Home arms a
    fresh epoch again.
    """

    def __init__(
        self,
        *,
        config: StartupHomeContinuityConfig,
        fk_by_side: Mapping[str, Callable[[np.ndarray], np.ndarray]],
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self._fk_by_side = dict(fk_by_side)
        missing_fk = set(config.active_sides) - set(self._fk_by_side)
        if missing_fk:
            raise ValueError(
                f"Missing startup Home FK providers for sides: {sorted(missing_fk)}"
            )
        self._logger = logger
        self._home_qpos_by_side: dict[str, np.ndarray] = {}
        self._home_tcp_by_side: dict[str, np.ndarray] = {}
        self._epoch = 0
        self._stable_frames = 0
        self._released = False
        self._armed = False
        self._logged_statuses: set[str] = set()
        self._last_evaluation: dict[str, Any] = {
            "status": "unarmed",
            "dispatch_allowed": True,
        }

    def _emit_transition(self, status: str, message: str) -> None:
        if status in self._logged_statuses:
            return
        self._logged_statuses.add(status)
        if self._logger is not None:
            self._logger(message)

    def arm_from_home(self, qpos_by_side: Mapping[str, Any]) -> dict[str, Any]:
        home_qpos: dict[str, np.ndarray] = {}
        home_tcp: dict[str, np.ndarray] = {}
        for side in self.config.active_sides:
            raw_qpos = qpos_by_side.get(side)
            try:
                qpos = np.asarray(raw_qpos, dtype=float).reshape(6)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{side} runtime Home qpos must be a finite 6-vector."
                ) from exc
            if not np.all(np.isfinite(qpos)):
                raise ValueError(f"{side} runtime Home qpos is non-finite.")
            pose = np.asarray(self._fk_by_side[side](qpos.copy()), dtype=float)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                raise ValueError(f"{side} runtime Home FK must be a finite 4x4 pose.")
            home_qpos[side] = qpos.copy()
            home_tcp[side] = pose.copy()

        self._epoch += 1
        self._home_qpos_by_side = home_qpos
        self._home_tcp_by_side = home_tcp
        self._stable_frames = 0
        self._released = False
        self._armed = True
        self._logged_statuses.clear()
        self._last_evaluation = self._base_payload(
            status="waiting_for_candidate",
            dispatch_allowed=False,
        )
        self._emit_transition(
            "waiting_for_candidate",
            "armed from atomic runtime Home; final qref dispatch remains withheld "
            f"until {self.config.required_consecutive_frames} consecutive continuity frames pass",
        )
        return self.snapshot()

    def _base_payload(self, *, status: str, dispatch_allowed: bool) -> dict[str, Any]:
        return {
            "schema": "teleop_runtime.startup_home_continuity/v1",
            "epoch": int(self._epoch),
            "status": str(status),
            "armed": bool(self._armed),
            "released": bool(self._released),
            "dispatch_allowed": bool(dispatch_allowed),
            "stable_frames": int(self._stable_frames),
            "required_consecutive_frames": int(
                self.config.required_consecutive_frames
            ),
            "limits": {
                "max_joint_delta_deg": float(self.config.max_joint_delta_deg),
                "max_tcp_translation_m": float(
                    self.config.max_tcp_translation_m
                ),
                "max_tcp_rotation_deg": float(
                    self.config.max_tcp_rotation_deg
                ),
            },
        }

    def _reject(
        self,
        command: TeleopCommand,
        *,
        status: str,
        reason: str,
        by_side: Mapping[str, Any] | None = None,
    ) -> bool:
        self._stable_frames = 0
        payload = self._base_payload(status=status, dispatch_allowed=False)
        payload["reason"] = str(reason)
        if by_side is not None:
            payload["by_side"] = dict(by_side)
        self._last_evaluation = payload
        command.metadata["startup_home_continuity"] = dict(payload)
        command.metadata["dataset_recording_eligible"] = False
        self._emit_transition(
            status,
            "dispatch withheld in ARMED state: " + str(reason),
        )
        return False

    def admit(
        self,
        command: TeleopCommand,
        _observation: TeleopObservation,
        _now: float,
    ) -> bool:
        if not self._armed or self._released:
            return True

        # Idle application controls such as a dedicated gripper reset do not
        # move either arm and therefore cannot invalidate Home-to-arm-qref
        # continuity.  They still pass through this admission callback so the
        # Session does not need a second dispatch path, but must carry an
        # explicit non-motion marker; an unmarked arm-less command remains a
        # fail-closed missing-qref error.
        command_effectors = set(command.commands)
        arm_effectors = {f"{side}_arm" for side in self.config.active_sides}
        if not command_effectors & arm_effectors:
            if command_effectors and all(
                name in {"left_hand", "right_hand"} for name in command_effectors
            ) and bool(command.metadata.get("external_non_motion", False)):
                payload = self._base_payload(
                    status="non_motion_passthrough",
                    dispatch_allowed=True,
                )
                payload["effectors"] = sorted(command_effectors)
                self._last_evaluation = payload
                command.metadata["startup_home_continuity"] = dict(payload)
                command.metadata["dataset_recording_eligible"] = False
                return True
            return self._reject(
                command,
                status="missing_arm_qref",
                reason="final command is missing an active arm qref",
            )

        metrics_by_side: dict[str, dict[str, float]] = {}
        try:
            for side in self.config.active_sides:
                effector = command.commands.get(f"{side}_arm")
                if effector is None:
                    return self._reject(
                        command,
                        status="missing_arm_qref",
                        reason=f"final command is missing {side}_arm",
                    )
                if str(effector.command_type) != "joint_vector":
                    return self._reject(
                        command,
                        status="invalid_arm_qref",
                        reason=(
                            f"{side}_arm final command type is "
                            f"{effector.command_type!r}, expected 'joint_vector'"
                        ),
                    )
                qpos = np.asarray(effector.value, dtype=float).reshape(6)
                if not np.all(np.isfinite(qpos)):
                    return self._reject(
                        command,
                        status="invalid_arm_qref",
                        reason=f"{side}_arm final qref is non-finite",
                    )
                candidate_tcp = np.asarray(
                    self._fk_by_side[side](qpos.copy()), dtype=float
                )
                translation_m, rotation_deg = pose_delta_metrics(
                    self._home_tcp_by_side[side],
                    candidate_tcp,
                )
                max_joint_delta_deg = float(
                    np.max(
                        np.abs(
                            np.rad2deg(qpos - self._home_qpos_by_side[side])
                        )
                    )
                )
                metrics_by_side[side] = {
                    "max_joint_delta_deg": max_joint_delta_deg,
                    "tcp_translation_m": float(translation_m),
                    "tcp_rotation_deg": float(rotation_deg),
                }
        except Exception as exc:  # noqa: BLE001 - admission must fail closed.
            return self._reject(
                command,
                status="evaluation_error",
                reason=f"{type(exc).__name__}: {exc}",
                by_side=metrics_by_side,
            )

        outside = {
            side: metrics
            for side, metrics in metrics_by_side.items()
            if (
                metrics["max_joint_delta_deg"]
                > float(self.config.max_joint_delta_deg)
                or metrics["tcp_translation_m"]
                > float(self.config.max_tcp_translation_m)
                or metrics["tcp_rotation_deg"]
                > float(self.config.max_tcp_rotation_deg)
            )
        }
        if outside:
            return self._reject(
                command,
                status="outside_home_envelope",
                reason="final qref is not continuous with the atomic runtime Home snapshot",
                by_side=metrics_by_side,
            )

        self._stable_frames += 1
        if self._stable_frames < self.config.required_consecutive_frames:
            payload = self._base_payload(
                status="continuity_dwell",
                dispatch_allowed=False,
            )
            payload["by_side"] = metrics_by_side
            self._last_evaluation = payload
            command.metadata["startup_home_continuity"] = dict(payload)
            command.metadata["dataset_recording_eligible"] = False
            self._emit_transition(
                "continuity_dwell",
                "final qref is inside the Home envelope; dispatch remains withheld "
                f"during continuity dwell {self._stable_frames}/"
                f"{self.config.required_consecutive_frames}",
            )
            return False

        self._released = True
        payload = self._base_payload(status="released", dispatch_allowed=True)
        payload["by_side"] = metrics_by_side
        self._last_evaluation = payload
        command.metadata["startup_home_continuity"] = dict(payload)
        self._emit_transition(
            "released",
            "Home continuity passed; releasing the first real qref to the existing "
            "position planner",
        )
        return True

    def snapshot(self) -> dict[str, Any]:
        return dict(self._last_evaluation)

    def home_qpos_snapshot(self) -> dict[str, np.ndarray] | None:
        """Return the atomic runtime-Home qpos used by this gate.

        The initial-alignment planner must use the same qpos snapshot as the
        continuity gate.  Returning defensive copies keeps the gate as the
        owner of that safety anchor while allowing another component to build
        a plan from it without sharing mutable arrays.
        """

        if not self._armed or not self._home_qpos_by_side:
            return None
        return {
            side: qpos.copy()
            for side, qpos in self._home_qpos_by_side.items()
        }


__all__ = [
    "DEFAULT_STARTUP_HOME_MAX_JOINT_DELTA_DEG",
    "DEFAULT_STARTUP_HOME_MAX_TCP_ROTATION_DEG",
    "DEFAULT_STARTUP_HOME_MAX_TCP_TRANSLATION_M",
    "DEFAULT_STARTUP_HOME_REQUIRED_CONSECUTIVE_FRAMES",
    "StartupHomeContinuityConfig",
    "StartupHomeContinuityGate",
]
