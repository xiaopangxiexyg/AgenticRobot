from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Datasets.tactile_uv import InspireTactileUVConverter
from Training.engine import resolve_device
from Training.registry import get_experiment_spec


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {path} must contain a dict payload.")
    return payload


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _split_summary(data_summary: Mapping[str, Any]) -> Mapping[str, Any]:
    for split_name in ("train", "val"):
        split_summary = data_summary.get(split_name)
        if isinstance(split_summary, Mapping):
            return split_summary
    raise KeyError("Checkpoint data_summary must contain a train or val split summary.")


def _checkpoint_data_summary(checkpoint: Mapping[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    direct = checkpoint.get("data_summary")
    if isinstance(direct, dict):
        return direct
    summary = checkpoint.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("data_summary"), dict):
        return summary["data_summary"]
    sidecar = _load_json(checkpoint_path.parent / "data_summary.json")
    if sidecar is not None:
        return sidecar
    raise KeyError(
        f"Unable to find data_summary in {checkpoint_path}. "
        "Expected checkpoint['data_summary'], checkpoint['summary']['data_summary'], or sibling data_summary.json."
    )


def _checkpoint_config(checkpoint: Mapping[str, Any], checkpoint_path: Path) -> Any:
    config = checkpoint.get("config")
    if config is None:
        sidecar = checkpoint_path.parent / "resolved_config.yaml"
        if not sidecar.exists():
            raise KeyError(
                f"Unable to find config in {checkpoint_path}. "
                "Expected checkpoint['config'] or sibling resolved_config.yaml."
            )
        return OmegaConf.load(sidecar)
    return OmegaConf.create(config)


def _camera_key_from_video_key(video_key: str | None) -> str:
    text = str(video_key or "").strip()
    prefix = "observation.images."
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text or "front"


def _resize_image_if_available(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    if int(height) <= 0 or int(width) <= 0:
        return image
    if image.shape[0] == int(height) and image.shape[1] == int(width):
        return image
    try:
        import cv2

        return cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    except ModuleNotFoundError:
        return image


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", "none", "null"}:
            return False
    return bool(value)


def _float_or_default(value: Any, default: float) -> float:
    if value is None or str(value).strip().lower() in {"null", "none"}:
        return float(default)
    return float(value)


def _optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "all", "full", "none", "null"}:
            return None
        value = normalized
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be positive, 0/all, or None; got {value!r}.") from error
    if result == 0:
        return None
    if result < 0:
        raise ValueError(f"{field_name} must be positive, 0/all, or None; got {value!r}.")
    return result


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            continue
        return value
    return None


def _normalize_tactile_representation(value: Any, *, output_dim: int | None = None) -> str:
    if value in (None, "null", "None"):
        return "raw" if output_dim == 1062 else "uv"
    normalized = str(value).strip().lower()
    aliases = {
        "uv": "uv",
        "uv351": "uv",
        "converted": "uv",
        "compact": "uv",
        "raw": "raw",
        "raw1062": "raw",
        "1062": "raw",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported tactile representation={value!r}; expected one of {sorted(aliases)}.")
    return aliases[normalized]


def _normalize_tactile_normalization_mode(value: Any, *, has_minmax: bool = False) -> str:
    if value is None or str(value).strip().lower() in {"", "null", "none"}:
        return "minmax" if has_minmax else "offset_scale"
    normalized = str(value).strip().lower()
    aliases = {
        "minmax": "minmax",
        "min_max": "minmax",
        "dataset_minmax": "minmax",
        "offset_scale": "offset_scale",
        "scale_offset": "offset_scale",
        "legacy": "offset_scale",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported tactile normalization mode={value!r}; expected one of {sorted(aliases)}.")
    return aliases[normalized]


class ManipRepFR3Policy:
    """Deployment adapter for ManipRep real-robot VLA checkpoints.

    The adapter is loaded by ``FR3_deploy`` through its generic ``--policy``
    hook. It restores the training model from the checkpoint and rebuilds the
    exact lightweight inputs used by the real-robot dataloader:
    state [19], optional tactile [1, D], and vision token [RGB mean + t].
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        camera_key: str | None = None,
        tactile_side: str | None = None,
        require_image: bool = True,
        require_tactile: bool = True,
        strict_state_dim: bool = True,
        action_clip_nan_to_hold: bool = True,
        action_chunk_size: int | str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint).expanduser().resolve()
        if self.checkpoint_path.is_dir():
            self.checkpoint_path = self.checkpoint_path / "last.pt"
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"ManipRep checkpoint not found: {self.checkpoint_path}")

        resolved_device = "cuda" if str(device).lower() == "auto" else str(device)
        self.device = resolve_device(resolved_device)
        checkpoint_payload = _torch_load(self.checkpoint_path, self.device)
        self.config = _checkpoint_config(checkpoint_payload, self.checkpoint_path)
        self.data_summary = _checkpoint_data_summary(checkpoint_payload, self.checkpoint_path)
        self.split_summary = dict(_split_summary(self.data_summary))

        self.experiment_name = str(self.config.experiment.name)
        self.state_dim = int(self.split_summary["state_dim"])
        self.action_dim = int(self.split_summary["action_dim"])
        self.action_chunk = int(self.split_summary.get("action_chunk", 1))
        self.tactile_enabled = bool(self.split_summary.get("tactile_observation_enabled", False))
        self.camera_key = str(camera_key or _camera_key_from_video_key(self.split_summary.get("video_key")))
        self.resize_height = int(OmegaConf.select(self.config, "data.raw.video.resize_height", default=90) or 90)
        self.resize_width = int(OmegaConf.select(self.config, "data.raw.video.resize_width", default=160) or 160)
        self.require_image = bool(require_image)
        self.require_tactile = bool(require_tactile)
        self.strict_state_dim = bool(strict_state_dim)
        self.action_clip_nan_to_hold = bool(action_clip_nan_to_hold)
        self.action_chunk_size = _optional_positive_int(action_chunk_size, field_name="action_chunk_size")

        spec = get_experiment_spec(self.experiment_name)
        model, _ = spec.build_model_and_optimizer(self.config, self.data_summary)
        model.load_state_dict(checkpoint_payload["model"], strict=True)
        self.model = model.to(self.device)
        self.model.eval()

        self.tactile_converter: InspireTactileUVConverter | None = None
        self.tactile_side = str(tactile_side or "right").lower()
        self.tactile_representation = "uv"
        self.tactile_expected_dim = 0
        self.tactile_normalize = False
        self.tactile_normalization_mode = "minmax"
        self.tactile_normalization_min = 0.0
        self.tactile_normalization_max = 1.0
        self.tactile_normalization_range = 1.0
        self.tactile_normalization_offset = 0.0
        self.tactile_normalization_scale = 1.0
        self.tactile_normalization_clip = False
        if self.tactile_enabled:
            tactile_fields = self.split_summary.get("tactile_observation_fields") or {}
            tactile_fields = tactile_fields if isinstance(tactile_fields, Mapping) else {}
            tactile_normalization = self.split_summary.get("tactile_observation_normalization") or {}
            tactile_normalization = tactile_normalization if isinstance(tactile_normalization, Mapping) else {}
            self.tactile_expected_dim = int(
                tactile_fields.get("converted_dim") or self.split_summary["tactile_observation_dim"]
            )
            self.tactile_representation = _normalize_tactile_representation(
                tactile_fields.get("representation", self.split_summary.get("tactile_observation_representation")),
                output_dim=self.tactile_expected_dim,
            )
            self.tactile_side = str(tactile_side or tactile_fields.get("hand") or "right").lower()
            normalized_flag = self.split_summary.get("tactile_observation_normalized")
            if normalized_flag is None:
                normalized_flag = tactile_fields.get("normalize")
            self.tactile_normalize = _as_bool(normalized_flag, default=False)
            min_value = _first_present(tactile_fields.get("normalization_min"), tactile_normalization.get("min"))
            max_value = _first_present(tactile_fields.get("normalization_max"), tactile_normalization.get("max"))
            range_value = _first_present(tactile_fields.get("normalization_range"), tactile_normalization.get("range"))
            mode_value = _first_present(tactile_fields.get("normalization_mode"), tactile_normalization.get("mode"))
            self.tactile_normalization_mode = _normalize_tactile_normalization_mode(
                mode_value,
                has_minmax=min_value is not None and (max_value is not None or range_value is not None),
            )
            self.tactile_normalization_clip = _as_bool(
                _first_present(tactile_fields.get("normalization_clip"), tactile_normalization.get("clip")),
                default=False,
            )
            if self.tactile_normalization_mode == "minmax":
                if min_value is None:
                    legacy_offset = _first_present(tactile_fields.get("normalization_offset"), tactile_normalization.get("offset"))
                    min_value = legacy_offset
                if range_value is None:
                    legacy_scale = _first_present(tactile_fields.get("normalization_scale"), tactile_normalization.get("scale"))
                    range_value = legacy_scale
                if max_value is None and min_value is not None and range_value is not None:
                    max_value = float(min_value) + float(range_value)
                if min_value is None or max_value is None:
                    raise ValueError("Normalized min-max tactile checkpoint is missing normalization min/max.")
                self.tactile_normalization_min = float(min_value)
                self.tactile_normalization_max = float(max_value)
                self.tactile_normalization_range = float(range_value) if range_value is not None else (
                    self.tactile_normalization_max - self.tactile_normalization_min
                )
                if self.tactile_normalization_range <= 0.0:
                    raise ValueError("tactile min-max normalization range must be positive.")
                self.tactile_normalization_offset = self.tactile_normalization_min
                self.tactile_normalization_scale = self.tactile_normalization_range
            else:
                self.tactile_normalization_offset = _float_or_default(
                    _first_present(tactile_fields.get("normalization_offset"), tactile_normalization.get("offset")),
                    0.0,
                )
                self.tactile_normalization_scale = _float_or_default(
                    _first_present(tactile_fields.get("normalization_scale"), tactile_normalization.get("scale")),
                    1.0,
                )
                if self.tactile_normalization_scale == 0.0:
                    raise ValueError("tactile normalization scale must be non-zero.")

            if self.tactile_representation == "uv":
                uv_root = str(tactile_fields.get("uv_conversion_root") or "PolicyDeploy/UV_conversion")
                correct_order = _as_bool(tactile_fields.get("correct_order"), default=True)
                self.tactile_converter = InspireTactileUVConverter(
                    uv_conversion_root=uv_root,
                    hand=self.tactile_side,
                    expected_dim=self.tactile_expected_dim,
                    correct_order=correct_order,
                )

        self._last_metadata: dict[str, Any] = {}

    def reset(self) -> None:
        self._last_metadata = {}

    @property
    def status_text(self) -> str:
        mode = self._mode_label()
        chunk = "all" if self.action_chunk_size is None else f"first{self.action_chunk_size}"
        return f"maniprep={mode}:epoch_ckpt={self.checkpoint_path.name}:chunk={chunk}"

    def _mode_label(self) -> str:
        if not self.tactile_enabled:
            return "no_tactile"
        suffix = "_norm" if self.tactile_normalize else ""
        return f"tactile_{self.tactile_representation}{suffix}"

    def _prepare_state(self, observation: Any) -> torch.Tensor:
        state = np.asarray(getattr(observation, "state"), dtype=np.float32).reshape(-1)
        if self.strict_state_dim and state.shape != (self.state_dim,):
            raise ValueError(f"Expected deployment state dim={self.state_dim}, got {state.shape}.")
        if state.shape[0] < self.state_dim:
            padded = np.zeros((self.state_dim,), dtype=np.float32)
            padded[: state.shape[0]] = state
            state = padded
        elif state.shape[0] > self.state_dim:
            state = state[: self.state_dim]
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return torch.from_numpy(state).unsqueeze(0).to(device=self.device)

    def _prepare_vision(self, observation: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not bool(OmegaConf.select(self.config, "model.use_vision_tokens", default=True)):
            return None, None
        images = getattr(observation, "images", {}) or {}
        image = images.get(self.camera_key)
        if image is None and len(images) == 1:
            image = next(iter(images.values()))
        if image is None:
            if self.require_image:
                raise RuntimeError(f"Missing required camera frame '{self.camera_key}'.")
            tokens = np.zeros((1, 1, 4), dtype=np.float32)
            mask = np.zeros((1, 1), dtype=np.float32)
            return (
                torch.from_numpy(tokens).to(device=self.device),
                torch.from_numpy(mask).to(device=self.device),
            )

        frame = np.asarray(image)
        if frame.ndim != 3 or int(frame.shape[-1]) < 3:
            raise ValueError(f"Expected HxWx3 camera image, got shape={frame.shape}.")
        frame = frame[..., :3].astype(np.float32, copy=False)
        frame = _resize_image_if_available(frame, height=self.resize_height, width=self.resize_width)
        if frame.max(initial=0.0) > 1.5:
            frame = frame / 255.0
        rgb_mean = frame.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        token = np.concatenate([rgb_mean, np.array([0.0], dtype=np.float32)], axis=0)
        tokens = torch.from_numpy(token.reshape(1, 1, 4)).to(device=self.device)
        mask = torch.ones((1, 1), dtype=torch.float32, device=self.device)
        return tokens, mask

    def _prepare_tactile(self, observation: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.tactile_enabled:
            return None, None
        tactile_by_side = getattr(observation, "tactile_by_side", {}) or {}
        raw = tactile_by_side.get(self.tactile_side)
        if raw is None:
            if self.require_tactile:
                raise RuntimeError(f"Missing required tactile vector for side '{self.tactile_side}'.")
            raw = np.zeros((1062,), dtype=np.float32)
        raw_array = np.asarray(raw, dtype=np.float32).reshape(-1)
        if raw_array.shape != (1062,):
            raise ValueError(f"Expected raw tactile shape=(1062,) for {self.tactile_side}, got {raw_array.shape}.")
        if self.tactile_representation == "raw":
            tactile = raw_array.reshape(1, 1, -1).astype(np.float32, copy=False)
        else:
            if self.tactile_converter is None:
                raise RuntimeError("UV tactile checkpoint is missing its tactile converter.")
            tactile = self.tactile_converter.convert(raw_array).reshape(1, 1, -1).astype(np.float32, copy=False)
        if self.tactile_normalize:
            if self.tactile_normalization_mode == "minmax":
                tactile = (tactile - self.tactile_normalization_min) / self.tactile_normalization_range
                if self.tactile_normalization_clip:
                    tactile = np.clip(tactile, 0.0, 1.0)
            else:
                tactile = (tactile - self.tactile_normalization_offset) / self.tactile_normalization_scale
            tactile = tactile.astype(np.float32, copy=False)
        if self.tactile_expected_dim and int(tactile.shape[-1]) != self.tactile_expected_dim:
            raise ValueError(
                f"Expected tactile feature dim={self.tactile_expected_dim}, got shape={tactile.shape}."
            )
        tactile_tensor = torch.from_numpy(tactile).to(device=self.device)
        tactile_mask = torch.ones((1, 1), dtype=torch.float32, device=self.device)
        return tactile_tensor, tactile_mask

    def _model_inputs(self, observation: Any) -> dict[str, Any]:
        state = self._prepare_state(observation)
        vision_tokens, vision_mask = self._prepare_vision(observation)
        inputs: dict[str, Any] = {"state": state}
        if vision_tokens is not None:
            inputs["vision_tokens"] = vision_tokens
            inputs["vision_mask"] = vision_mask
        tactile, tactile_mask = self._prepare_tactile(observation)
        if tactile is not None:
            inputs["tactile"] = tactile
            inputs["tactile_mask"] = tactile_mask
        if bool(OmegaConf.select(self.config, "model.task.enabled", default=False)):
            metadata = getattr(observation, "metadata", {}) or {}
            task_text = metadata.get("task_text") or metadata.get("task") or self._default_task_text()
            inputs["task_text"] = [str(task_text)]
        return inputs

    def _default_task_text(self) -> str:
        task_names = self.split_summary.get("task_names")
        if isinstance(task_names, list) and task_names:
            return str(task_names[0])
        return "unknown_task"

    def predict(self, observation: Any) -> dict[str, Any]:
        inputs = self._model_inputs(observation)
        with torch.inference_mode():
            output = self.model(**inputs)
        action = output["action"].detach().to("cpu").numpy().astype(np.float32, copy=False)
        if action.ndim == 3:
            if int(action.shape[0]) != 1:
                raise ValueError(f"Expected batch size 1 action output, got {action.shape}.")
            action = action[0]
        elif action.ndim == 2:
            pass
        elif action.ndim == 1:
            action = action.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported action output shape {action.shape}.")
        if int(action.shape[-1]) != self.action_dim:
            raise ValueError(f"Expected action dim={self.action_dim}, got shape={action.shape}.")
        model_action_shape = list(action.shape)
        if self.action_chunk_size is not None:
            action = action[: self.action_chunk_size]
            if int(action.shape[0]) <= 0:
                raise ValueError(
                    f"action_chunk_size={self.action_chunk_size} produced an empty action chunk "
                    f"from model output shape {model_action_shape}."
                )
        if self.action_clip_nan_to_hold and not np.all(np.isfinite(action)):
            hold = np.asarray(observation.hold_action(), dtype=np.float32).reshape(1, -1)
            action = np.where(np.isfinite(action), action, hold)
        if not np.all(np.isfinite(action)):
            raise ValueError("Model produced non-finite action values.")
        self._last_metadata = {
            "checkpoint": str(self.checkpoint_path),
            "experiment": self.experiment_name,
            "mode": self._mode_label(),
            "action_shape": list(action.shape),
            "model_action_shape": model_action_shape,
            "action_chunk_size": self.action_chunk_size,
        }
        return {
            "action_chunk": action,
            "metadata": dict(self._last_metadata),
        }


Policy = ManipRepFR3Policy


def create_policy(**kwargs: Any) -> ManipRepFR3Policy:
    return ManipRepFR3Policy(**kwargs)
