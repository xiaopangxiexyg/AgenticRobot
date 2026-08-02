"""Task-owned CubeLift policy-action conditioning shared by train and deploy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import CubeLiftTaskConfig


@dataclass(frozen=True)
class CubeLiftActionConditioningResult:
    """One conditioned action plus the task phase state that produced it."""

    values: np.ndarray
    scripted_contact_modified: bool
    scripted_pregrasp_modified: bool
    post_contact_modified: bool
    post_contact_phase_active: bool
    post_contact_phase_contact_steps: int
    post_contact_phase_loss_steps: int

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32).reshape(-1)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("Conditioned CubeLift action must be finite 7-D.")
        object.__setattr__(self, "values", values.copy())


class CubeLiftActionConditioner:
    """Own the task's reset and post-contact policy-action semantics."""

    def __init__(self, config: CubeLiftTaskConfig) -> None:
        if not isinstance(config, CubeLiftTaskConfig):
            raise TypeError("config must be a CubeLiftTaskConfig.")
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._post_contact_phase_active = False
        self._post_contact_phase_contact_steps = 0
        self._post_contact_phase_loss_steps = 0

    def restore_post_contact_phase(
        self,
        *,
        active: bool,
        contact_steps: int,
        loss_steps: int,
    ) -> None:
        """Restore task-owned phase state from an explicit lifecycle snapshot."""

        normalized_contact = int(contact_steps)
        normalized_loss = int(loss_steps)
        if normalized_contact < 0 or normalized_loss < 0:
            raise ValueError("Post-contact phase counters must be non-negative.")
        self._post_contact_phase_active = bool(active)
        self._post_contact_phase_contact_steps = normalized_contact
        self._post_contact_phase_loss_steps = normalized_loss

    def condition(
        self,
        action: Any,
        *,
        left_finger_contact: bool,
        right_finger_contact: bool,
        measured_gripper_close: float,
        scripted_pregrasp_reset_applied: bool,
        scripted_contact_reset_applied: bool,
    ) -> CubeLiftActionConditioningResult:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("CubeLift action must be a finite 7-D vector.")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("CubeLift action values must lie in [-1, 1].")
        close = float(measured_gripper_close)
        if not np.isfinite(close):
            raise ValueError("measured_gripper_close must be finite.")
        if scripted_pregrasp_reset_applied and scripted_contact_reset_applied:
            raise ValueError("Only one scripted reset mode may be active.")

        if self.config.post_contact_phase_enabled:
            contact_ready = (
                bool(left_finger_contact)
                and bool(right_finger_contact)
                and close >= self.config.post_contact_phase_min_measured_close
            )
            if contact_ready:
                self._post_contact_phase_contact_steps += 1
                self._post_contact_phase_loss_steps = 0
                if (
                    not self._post_contact_phase_active
                    and self._post_contact_phase_contact_steps
                    >= self.config.post_contact_phase_activation_steps
                ):
                    self._post_contact_phase_active = True
            else:
                self._post_contact_phase_contact_steps = 0
                if self._post_contact_phase_active:
                    self._post_contact_phase_loss_steps += 1
                    grace = self.config.post_contact_phase_loss_grace_steps
                    if (
                        grace == 0
                        or self._post_contact_phase_loss_steps >= grace
                    ):
                        self._post_contact_phase_active = False
                        self._post_contact_phase_loss_steps = 0

        conditioned = values.copy()
        scripted_contact_modified = False
        scripted_pregrasp_modified = False
        if scripted_contact_reset_applied or scripted_pregrasp_reset_applied:
            before = conditioned.copy()
            action_scale = (
                self.config.scripted_contact_action_scale
                if scripted_contact_reset_applied
                else self.config.scripted_pregrasp_action_scale
            )
            action_bias = (
                self.config.scripted_contact_action_bias
                if scripted_contact_reset_applied
                else self.config.scripted_pregrasp_action_bias
            )
            conditioned *= np.asarray(action_scale, dtype=np.float32)
            conditioned += np.asarray(action_bias, dtype=np.float32)
            if (
                scripted_contact_reset_applied
                and self.config.scripted_contact_nonnegative_lift_action
            ):
                conditioned[2] = max(float(conditioned[2]), 0.0)
            modified = not np.array_equal(conditioned, before)
            scripted_contact_modified = (
                modified if scripted_contact_reset_applied else False
            )
            scripted_pregrasp_modified = (
                modified if scripted_pregrasp_reset_applied else False
            )

        post_contact_modified = False
        if self._post_contact_phase_active:
            before = conditioned.copy()
            conditioned *= np.asarray(
                self.config.post_contact_action_scale,
                dtype=np.float32,
            )
            conditioned += np.asarray(
                self.config.post_contact_action_bias,
                dtype=np.float32,
            )
            if self.config.post_contact_nonnegative_lift_action:
                conditioned[2] = max(float(conditioned[2]), 0.0)
            post_contact_modified = not np.array_equal(conditioned, before)

        return CubeLiftActionConditioningResult(
            values=conditioned,
            scripted_contact_modified=scripted_contact_modified,
            scripted_pregrasp_modified=scripted_pregrasp_modified,
            post_contact_modified=post_contact_modified,
            post_contact_phase_active=self._post_contact_phase_active,
            post_contact_phase_contact_steps=(
                self._post_contact_phase_contact_steps
            ),
            post_contact_phase_loss_steps=self._post_contact_phase_loss_steps,
        )


class CubeLiftFingerContactObserver:
    """Read only the two task-declared finger/cube contact booleans."""

    def __init__(self, scene: Any, config: CubeLiftTaskConfig) -> None:
        self.model = scene.model
        self.data = scene.data
        self._cube_body_id = scene.body_id(config.object_body_name)
        self._left_finger_body_id = scene.body_id(
            config.grasp_latch_left_finger_body_name
        )
        self._right_finger_body_id = scene.body_id(
            config.grasp_latch_right_finger_body_name
        )

    def observe(self) -> tuple[bool, bool]:
        left_contact = False
        right_contact = False
        geom_body_id = self.model.geom_bodyid
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            body1 = int(geom_body_id[int(contact.geom1)])
            body2 = int(geom_body_id[int(contact.geom2)])
            if body1 != self._cube_body_id and body2 != self._cube_body_id:
                continue
            other = body2 if body1 == self._cube_body_id else body1
            left_contact = (
                left_contact or other == self._left_finger_body_id
            )
            right_contact = (
                right_contact or other == self._right_finger_body_id
            )
        return left_contact, right_contact


__all__ = [
    "CubeLiftActionConditioner",
    "CubeLiftActionConditioningResult",
    "CubeLiftFingerContactObserver",
]
