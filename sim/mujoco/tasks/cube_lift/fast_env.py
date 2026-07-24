"""Explicit single-environment Gymnasium adapter over CubeLiftKernel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import CubeLiftTaskConfig
from .env import DEFAULT_CUBE_LIFT_TASK_CONFIG_PATH
from .kernel import CubeLiftKernel, InfoMode, ScriptedContactResetError
from .spec import CubeLiftActionSpec, CubeLiftObservationSpec, require_gymnasium


try:
    _gymnasium, _spaces = require_gymnasium()
    _EnvBase = _gymnasium.Env
except ModuleNotFoundError:
    _gymnasium = None
    _spaces = None
    _EnvBase = object


class FastCubeLiftEnv(_EnvBase):
    """Fast training adapter; use CubeLiftEnv for detailed debugging/audit."""

    metadata = {"render_modes": [], "render_fps": 15}

    def __init__(
        self,
        config: CubeLiftTaskConfig | str | Path = DEFAULT_CUBE_LIFT_TASK_CONFIG_PATH,
        *,
        info_mode: InfoMode = "terminal",
        use_settled_snapshot: bool = True,
        use_model_cache: bool = True,
        model_cache_dir: str | Path | None = None,
        include_training_reward_state: bool = False,
    ) -> None:
        if _gymnasium is None:
            require_gymnasium()
        self.kernel = CubeLiftKernel(
            config,
            info_mode=info_mode,
            use_settled_snapshot=use_settled_snapshot,
            use_model_cache=use_model_cache,
            model_cache_dir=model_cache_dir,
            include_training_reward_state=include_training_reward_state,
        )
        self.config = self.kernel.config
        self.action_spec = CubeLiftActionSpec(layout_version=self.config.action_layout)
        self.observation_spec = CubeLiftObservationSpec(
            layout_version=self.config.observation_layout
        )
        self.action_space = self.action_spec.gym_space()
        self.observation_space = self.observation_spec.gym_space()
        self.last_physics_ticks = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if _gymnasium is None:
            require_gymnasium()
        super().reset(seed=seed)
        if options:
            raise ValueError("FastCubeLiftEnv does not support reset options yet.")
        reset_failures: list[str] = []
        max_attempts = (
            self.config.scripted_contact_reset_max_attempts
            if self.config.scripted_contact_reset
            else 1
        )
        for attempt in range(1, max_attempts + 1):
            try:
                observation, info = self.kernel.reset(seed=seed)
                break
            except ScriptedContactResetError as exc:
                reset_failures.append(str(exc))
                if attempt >= max_attempts:
                    raise
        info = {
            **info,
            "scripted_contact_reset_attempts": attempt,
            "scripted_contact_reset_failures": tuple(reset_failures),
        }
        self.last_physics_ticks = self.kernel.last_physics_ticks
        return observation, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = self.kernel.step(action)
        self.last_physics_ticks = self.kernel.last_physics_ticks
        return (
            result.observation,
            result.reward,
            result.terminated,
            result.truncated,
            result.info,
        )

    def close(self) -> None:
        self.kernel.close()


__all__ = ["FastCubeLiftEnv"]
