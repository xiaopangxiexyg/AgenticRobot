from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.writer import ACTION_DIM, ARM_DOF, HAND_DOF, HAND_VECTOR_ORDER
from .observation import DeploymentObservation
from .replay import load_replay_actions_from_parquet, resolve_replay_parquet_path


_ARM_SLICE = slice(0, ARM_DOF)
_HAND_SLICES = {
    "right": slice(ARM_DOF, ARM_DOF + HAND_DOF),
    "left": slice(ARM_DOF + HAND_DOF, ACTION_DIM),
}


def _action_vector(value: Any, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (ACTION_DIM,):
        raise ValueError(f"{label} must have shape ({ACTION_DIM},), got {vector.shape}.")
    return vector.astype(np.float32, copy=True)


def _policy_replay_first_action(policy: Any) -> np.ndarray | None:
    wrapped = getattr(policy, "policy", policy)
    actions = getattr(wrapped, "actions", None)
    if actions is None:
        return None
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM or array.shape[0] <= 0:
        raise ValueError(
            f"Replay startup init actions must have shape (N, {ACTION_DIM}), got {array.shape}."
        )
    return array[0].astype(np.float32, copy=True)


def _load_dataset_init_action(args: Any) -> tuple[np.ndarray, str]:
    dataset_root = getattr(args, "init_dataset", None)
    parquet_path = getattr(args, "init_parquet", None)
    if not dataset_root and not parquet_path:
        raise ValueError("--init-source dataset requires --init-dataset or --init-parquet.")
    frame_index = int(getattr(args, "init_frame", 0))
    path, metadata = resolve_replay_parquet_path(
        dataset_root=dataset_root,
        parquet_path=parquet_path,
        episode_index=int(getattr(args, "init_episode", 0)),
    )
    actions = load_replay_actions_from_parquet(
        path,
        action_column=str(getattr(args, "init_action_column", "action") or "action"),
        start_index=frame_index,
        end_index=frame_index + 1,
    )
    source = (
        f"dataset:{Path(path)}"
        f":episode={metadata.get('episode_index', getattr(args, 'init_episode', 0))}"
        f":frame={frame_index}"
    )
    return actions[0].astype(np.float32, copy=True), source


def resolve_startup_init_action(args: Any, policy: Any) -> tuple[np.ndarray | None, str]:
    source = str(getattr(args, "init_source", "auto") or "auto").strip().lower()
    if source in {"none", "off", "disabled"} or float(getattr(args, "init_time_s", 0.0)) <= 0.0:
        return None, "disabled"
    if source not in {"auto", "current", "dataset", "replay"}:
        raise ValueError("--init-source must be one of: auto, current, dataset, replay, none.")

    replay_first = _policy_replay_first_action(policy)
    has_dataset_init = bool(getattr(args, "init_dataset", None) or getattr(args, "init_parquet", None))
    if source == "auto":
        if replay_first is not None:
            return replay_first, "replay:first_frame"
        if has_dataset_init:
            return _load_dataset_init_action(args)
        return None, "current"
    if source == "current":
        return None, "current"
    if source == "replay":
        if replay_first is None:
            raise ValueError("--init-source replay requires --debug-replay-dataset or --debug-replay-parquet.")
        return replay_first, "replay:first_frame"
    return _load_dataset_init_action(args)


def _required_slices(*, active_sides: tuple[str, ...], arm_enabled: bool) -> list[tuple[str, slice]]:
    required: list[tuple[str, slice]] = []
    if arm_enabled:
        required.append(("arm", _ARM_SLICE))
    active = set(active_sides)
    for side in HAND_VECTOR_ORDER:
        if side in active:
            required.append((f"{side}_hand", _HAND_SLICES[side]))
    return required


def _missing_required(vector: np.ndarray, required: list[tuple[str, slice]]) -> list[str]:
    return [name for name, target_slice in required if not np.all(np.isfinite(vector[target_slice]))]


def _smoothstep(alpha: float) -> float:
    x = float(np.clip(alpha, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


@dataclass
class StartupInitializer:
    duration_s: float
    source: str
    target_action: np.ndarray | None
    active_sides: tuple[str, ...]
    arm_enabled: bool
    started_at: float | None = None
    start_action: np.ndarray | None = None
    resolved_target_action: np.ndarray | None = None
    completed: bool = False
    progress: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.source != "disabled" and float(self.duration_s) > 0.0

    def reset(self) -> None:
        self.started_at = None
        self.start_action = None
        self.resolved_target_action = None
        self.completed = not self.enabled
        self.progress = 1.0 if self.completed else 0.0

    def _start(self, observation: DeploymentObservation, now: float) -> list[str]:
        start = _action_vector(observation.hold_action(), label="startup init start action")
        if self.target_action is None:
            target = start.copy()
        else:
            target = _action_vector(self.target_action, label="startup init target action")
            missing_target = ~np.isfinite(target)
            target[missing_target] = start[missing_target]

        required = _required_slices(
            active_sides=self.active_sides,
            arm_enabled=bool(self.arm_enabled),
        )
        missing_start = _missing_required(start, required)
        missing_target = _missing_required(target, required)
        if missing_start or missing_target:
            missing = sorted(set(missing_start + missing_target))
            return ["startup_init_reference_unavailable:" + ",".join(missing)]

        self.started_at = float(now)
        self.start_action = start
        self.resolved_target_action = target
        self.progress = 0.0
        print(
            "[StartupInit] moving to initial target "
            f"source={self.source} duration={float(self.duration_s):.2f}s"
        )
        return []

    def next_action(
        self,
        observation: DeploymentObservation,
        *,
        now: float | None = None,
    ) -> tuple[np.ndarray | None, list[str], bool]:
        if not self.enabled or self.completed:
            return None, [], True
        current_time = time.time() if now is None else float(now)
        if self.started_at is None:
            faults = self._start(observation, current_time)
            if faults:
                return None, faults, False

        assert self.started_at is not None
        assert self.start_action is not None
        assert self.resolved_target_action is not None
        elapsed = max(0.0, current_time - float(self.started_at))
        raw_alpha = 1.0 if float(self.duration_s) <= 0.0 else min(1.0, elapsed / float(self.duration_s))
        alpha = _smoothstep(raw_alpha)
        self.progress = float(raw_alpha)
        if raw_alpha >= 1.0:
            return self.resolved_target_action.copy(), [], True
        action = self.start_action + (self.resolved_target_action - self.start_action) * alpha
        return action.astype(np.float32, copy=True), [], False

    def mark_complete(self) -> None:
        if not self.completed:
            print("[StartupInit] complete; policy/replay execution may start.")
        self.completed = True
        self.progress = 1.0

    @property
    def status_text(self) -> str:
        if not self.enabled:
            return ""
        state = "done" if self.completed else ("wait" if self.started_at is None else f"{100.0 * self.progress:.0f}%")
        return f"init={state}:{self.source}"
