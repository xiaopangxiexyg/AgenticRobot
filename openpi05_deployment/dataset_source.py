from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

from openpi05_deployment.action_adapter import compact_action_to_full_vector
from openpi05_deployment.errors import DatasetTrajectoryError
from openpi05_deployment.metadata import ServerContract
from teleop_runtime.deploy.schema import ACTION_DIM

_OPENPI_SRC = Path("/home/ai/zc/openpi/src")


def _require_openpi_modules() -> tuple[Any, Any]:
    if _OPENPI_SRC.exists():
        src = str(_OPENPI_SRC)
        if src not in sys.path:
            sys.path.insert(0, src)
    try:
        import openpi.training.local_robot_dataset as local_robot_dataset
    except ModuleNotFoundError as exc:
        raise DatasetTrajectoryError(
            "Dataset-backed OpenPI deployment requires the openpi source tree and its "
            f"dataset-reading dependencies to be importable (expected source at {_OPENPI_SRC})."
        ) from exc
    return local_robot_dataset, None


def _embodiment_spec_for_contract(local_robot_dataset: Any, contract: ServerContract) -> Any:
    if contract.embodiment.name == "realman_dual_arm_hands":
        return local_robot_dataset.REALMAN_DUAL_ARM_HANDS
    if contract.embodiment.name == "realman_right_arm_right_hand":
        return local_robot_dataset.REALMAN_RIGHT_ARM_RIGHT_HAND
    raise DatasetTrajectoryError(
        f"No dataset embodiment template mapping for {contract.embodiment.name!r}."
    )


@dataclass(frozen=True)
class DatasetCompareMetrics:
    mae: float
    rmse: float
    max_abs: float
    first_step_mae: float
    first_step_max_abs: float

    @property
    def status_text(self) -> str:
        return (
            f"mae={self.mae:.4f},rmse={self.rmse:.4f},"
            f"max={self.max_abs:.4f},first={self.first_step_mae:.4f}"
        )


@dataclass(frozen=True)
class DatasetTrajectoryStep:
    payload: dict[str, Any]
    compact_ground_truth_actions: np.ndarray
    frame_index: int
    episode_index: int
    prompt: str


class DatasetTrajectorySource:
    def __init__(
        self,
        contract: ServerContract,
        *,
        dataset_root: str,
        episode_index: int | None = None,
        episode_selection: str = "specified",
        start_frame: int = 0,
        camera: str | None = None,
        prompt: str | None = None,
        loop: bool = False,
        seed: int = 0,
        stride: int | None = None,
    ) -> None:
        if not str(dataset_root or "").strip():
            raise DatasetTrajectoryError("dataset_root is required for dataset-backed deployment.")
        local_robot_dataset, _ = _require_openpi_modules()
        embodiment = _embodiment_spec_for_contract(local_robot_dataset, contract)
        if contract.tactile_required and getattr(embodiment, "tactile_kind", "vector") != "vector":
            raise DatasetTrajectoryError(
                f"Embodiment {embodiment.name!r} does not support tactile vector deployment."
            )

        self.contract = contract
        self.dataset_root = str(Path(dataset_root).expanduser().resolve())
        self.episode_selection = str(episode_selection or "specified").strip().lower()
        self.start_frame = int(start_frame)
        self.loop = bool(loop)
        self.seed = int(seed)
        self.prompt_override = None if prompt in (None, "") else str(prompt)
        self.stride = int(contract.action_horizon if stride is None else stride)
        if self.start_frame < 0:
            raise DatasetTrajectoryError("dataset start_frame must be non-negative.")
        if self.stride <= 0:
            raise DatasetTrajectoryError("dataset stride must be positive.")
        if self.episode_selection not in {"specified", "random"}:
            raise DatasetTrajectoryError("dataset episode_selection must be 'specified' or 'random'.")

        self._dataset = local_robot_dataset.LocalRobotDataset(
            self.dataset_root,
            action_horizon=int(contract.action_horizon),
            embodiment=embodiment,
            camera=camera,
            load_images=True,
            load_tactile=bool(contract.tactile_required),
        )
        self._rng = np.random.default_rng(self.seed)
        self._episode_list_index, self._episode = self._select_episode(episode_index)
        self._cursor = int(self.start_frame)
        if self._cursor >= int(self._episode.frame_count):
            raise DatasetTrajectoryError(
                f"dataset start_frame {self._cursor} is out of range for episode "
                f"{self._episode.episode_index} with {self._episode.frame_count} frames."
            )
        self._episode_start = int(self._dataset._episode_offsets[self._episode_list_index])  # type: ignore[attr-defined]
        self._initial_state_action = self._build_initial_state_action()

    @property
    def selected_episode_index(self) -> int:
        return int(self._episode.episode_index)

    @property
    def current_frame_index(self) -> int:
        return int(self._cursor)

    @property
    def startup_init_source(self) -> str:
        return (
            f"dataset_state:{self.dataset_root}"
            f":episode={self.selected_episode_index}"
            f":frame={self.start_frame}"
        )

    def get_startup_init_action(self) -> np.ndarray:
        return self._initial_state_action.astype(np.float32, copy=True)

    def skip_startup_frames(self, count: int = 1) -> int:
        steps = max(0, int(count))
        if steps <= 0:
            return 0
        available = max(0, int(self._episode.frame_count) - int(self._cursor))
        skipped = min(steps, available)
        self._cursor += skipped
        return int(skipped)

    def next_step(self) -> DatasetTrajectoryStep:
        if self._cursor >= int(self._episode.frame_count):
            raise DatasetTrajectoryError(
                f"Dataset trajectory exhausted at episode={self.selected_episode_index}, "
                f"frame={self._cursor}."
            )
        global_index = int(self._episode_start + self._cursor)
        sample = dict(self._dataset[global_index])
        compact_actions = np.asarray(sample.pop("actions"), dtype=np.float32)
        if compact_actions.shape != (
            int(self.contract.action_horizon),
            int(self.contract.embodiment.output_action_dim),
        ):
            raise DatasetTrajectoryError(
                "Dataset action chunk shape mismatch: expected "
                f"{(self.contract.action_horizon, self.contract.embodiment.output_action_dim)}, "
                f"got {compact_actions.shape}."
            )
        prompt = str(self.prompt_override or sample.get("prompt") or self._episode.task)
        sample["prompt"] = prompt
        return DatasetTrajectoryStep(
            payload=sample,
            compact_ground_truth_actions=compact_actions.astype(np.float32, copy=True),
            frame_index=int(self._cursor),
            episode_index=self.selected_episode_index,
            prompt=prompt,
        )

    def advance(self, steps: int | None = None) -> None:
        step_count = self.stride if steps is None else int(steps)
        if step_count <= 0:
            raise DatasetTrajectoryError("advance step count must be positive.")
        next_cursor = int(self._cursor + step_count)
        if next_cursor < int(self._episode.frame_count):
            self._cursor = next_cursor
            return
        if not self.loop:
            self._cursor = next_cursor
            return
        valid_count = int(self._episode.frame_count - self.start_frame)
        if valid_count <= 0:
            raise DatasetTrajectoryError(
                f"Episode {self.selected_episode_index} has no frames available after start_frame={self.start_frame}."
            )
        self._cursor = int(self.start_frame + ((next_cursor - self.start_frame) % valid_count))

    def compact_actions_to_full_chunk(self, compact_actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(compact_actions, dtype=np.float32)
        if actions.ndim != 2:
            raise DatasetTrajectoryError(f"Expected rank-2 actions, got {actions.shape}.")
        chunk = np.full((actions.shape[0], ACTION_DIM), np.nan, dtype=np.float32)
        for index in range(actions.shape[0]):
            chunk[index] = compact_action_to_full_vector(
                actions[index],
                self.contract,
                seed_vector=np.full((ACTION_DIM,), np.nan, dtype=np.float32),
            )
        return chunk

    def compare_prediction(self, predicted_actions: Any, ground_truth_actions: np.ndarray) -> DatasetCompareMetrics:
        pred = np.asarray(predicted_actions, dtype=np.float32)
        gt = np.asarray(ground_truth_actions, dtype=np.float32)
        if pred.shape != gt.shape:
            raise DatasetTrajectoryError(
                f"Prediction/ground-truth shape mismatch: {pred.shape} vs {gt.shape}."
            )
        diff = pred - gt
        abs_diff = np.abs(diff)
        mae = float(np.mean(abs_diff))
        rmse = float(np.sqrt(np.mean(np.square(diff))))
        max_abs = float(np.max(abs_diff))
        first_abs = abs_diff[0]
        return DatasetCompareMetrics(
            mae=mae,
            rmse=rmse,
            max_abs=max_abs,
            first_step_mae=float(np.mean(first_abs)),
            first_step_max_abs=float(np.max(first_abs)),
        )

    def _build_initial_state_action(self) -> np.ndarray:
        step = self.next_step()
        compact_state = np.asarray(step.payload["observation/state"], dtype=np.float32).reshape(-1)
        if compact_state.shape != (int(self.contract.embodiment.state_dim),):
            raise DatasetTrajectoryError(
                "Dataset observation/state shape mismatch: expected "
                f"{(self.contract.embodiment.state_dim,)}, got {compact_state.shape}."
            )
        return compact_action_to_full_vector(
            compact_state,
            self.contract,
            seed_vector=np.full((ACTION_DIM,), np.nan, dtype=np.float32),
        )

    def _select_episode(self, requested_episode_index: int | None) -> tuple[int, Any]:
        candidates: list[tuple[int, Any]] = []
        for list_index, episode in enumerate(self._dataset.episodes):
            sides = tuple(getattr(episode, "active_sides", ()) or ())
            if sides and tuple(sides) != self.contract.embodiment.active_sides:
                continue
            if int(episode.frame_count) <= int(self.start_frame):
                continue
            candidates.append((int(list_index), episode))
        if not candidates:
            raise DatasetTrajectoryError(
                f"No usable episodes found in {self.dataset_root} for embodiment {self.contract.embodiment.name!r}."
            )

        if self.episode_selection == "random":
            choice = int(self._rng.integers(0, len(candidates)))
            return candidates[choice]

        target_episode = 0 if requested_episode_index is None else int(requested_episode_index)
        for list_index, episode in candidates:
            if int(episode.episode_index) == target_episode:
                return list_index, episode
        raise DatasetTrajectoryError(
            f"Episode {target_episode} not found in {self.dataset_root} for embodiment {self.contract.embodiment.name!r}."
        )
