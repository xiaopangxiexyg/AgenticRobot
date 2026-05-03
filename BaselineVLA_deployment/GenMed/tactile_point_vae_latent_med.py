from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import OmegaConf

from Models.PointCloud.tactile_point_vae import TactilePointVAE

from .base import BaseMedProvider
from utils.path_config import resolve_repo_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalize_coordinate_frame(value: Any) -> str:
    normalized = str(value or "camera").strip().lower()
    aliases = {
        "camera": "camera",
        "camera_frame": "camera",
        "first_person": "camera",
        "first_person_camera": "camera",
        "first_person_camera_opencv": "camera",
        "egocentric": "camera",
        "wrist": "wrist_local",
        "wrist_local": "wrist_local",
        "eef": "wrist_local",
        "eef_local": "wrist_local",
        "hand": "wrist_local",
        "hand_local": "wrist_local",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported tactile point VAE coordinate_frame={value!r}.")
    return aliases[normalized]


def _reference_summary(data_summary: Dict[str, Any]) -> Dict[str, Any]:
    for split_name in ("train", "val"):
        split_summary = data_summary.get(split_name)
        if split_summary is not None:
            return split_summary
    return data_summary


def _coordinate_frame_from_summary(data_summary: Dict[str, Any]) -> str | None:
    reference = _reference_summary(data_summary)
    selected_fields = reference.get("med_selected_fields") or {}
    coordinate_frame = selected_fields.get("coordinate_frame")
    if coordinate_frame is None:
        return None
    return _normalize_coordinate_frame(coordinate_frame)


def _coordinate_frame_from_config(config: Any) -> str | None:
    try:
        point_config = config.data.raw.sequence.med.point_cloud
    except Exception:
        return None
    coordinate_frame = getattr(point_config, "coordinate_frame", None)
    if coordinate_frame in (None, "null"):
        return None
    return _normalize_coordinate_frame(coordinate_frame)


def _resolve_checkpoint_path(checkpoint_ref: Any, checkpoint_kind: str) -> Path:
    if checkpoint_ref in (None, "null", ""):
        raise ValueError("data.med.checkpoint must point to a trained pcvae checkpoint, run directory, or run name.")
    candidate = resolve_repo_path(str(checkpoint_ref))
    if candidate.exists():
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
        preferred = resolved / f"{checkpoint_kind}.pt"
        if preferred.exists():
            return preferred
        for name in ("best.pt", "last.pt"):
            fallback = resolved / name
            if fallback.exists():
                return fallback
        raise FileNotFoundError(f"No best.pt or last.pt found under checkpoint directory: {resolved}")

    run_dir = PROJECT_ROOT / "Outputs" / "pcvae" / str(checkpoint_ref)
    if run_dir.exists():
        return _resolve_checkpoint_path(run_dir, checkpoint_kind=checkpoint_kind)
    raise FileNotFoundError(f"Unable to resolve tactile point VAE checkpoint: {checkpoint_ref}")


class TactilePointVAELatentMedProvider(BaseMedProvider):
    name = "TactilePointVAE_Latent_Med"

    def __init__(self, config: Any) -> None:
        self.config = config
        self.latent_dim = int(getattr(config, "latent_dim", 32))
        self.configured_coordinate_frame = _normalize_coordinate_frame(getattr(config, "coordinate_frame", "camera"))
        self.strict_coordinate_frame = bool(getattr(config, "strict_coordinate_frame", True))
        self.device = torch.device(str(getattr(config, "device", "cpu")))
        self._model: TactilePointVAE | None = None
        self._checkpoint_coordinate_frame: str | None = None
        self._checkpoint_path: Path | None = None

    def _resolve_source_key(self, sample: Any) -> str:
        configured_key = getattr(self.config, "source_intermediate_key", None)
        if configured_key is not None and str(configured_key).lower() not in {"auto", "null"}:
            return str(configured_key)
        metadata_key = sample.metadata.get("med_intermediate_key")
        if metadata_key is not None and str(metadata_key).lower() not in {"auto", "null"}:
            return str(metadata_key)
        return "med"

    def _assert_coordinate_frame(self, observed: str | None, source: str) -> None:
        if observed is None:
            return
        normalized = _normalize_coordinate_frame(observed)
        if normalized != self.configured_coordinate_frame and self.strict_coordinate_frame:
            raise ValueError(
                f"Tactile point VAE coordinate-frame mismatch from {source}: "
                f"provider={self.configured_coordinate_frame}, observed={normalized}. "
                "Use the matching camera or wrist-local data preset/checkpoint."
            )

    def _load_checkpoint_model(self) -> TactilePointVAE:
        if self._model is not None:
            return self._model
        checkpoint_path = _resolve_checkpoint_path(
            getattr(self.config, "checkpoint", None),
            checkpoint_kind=str(getattr(self.config, "checkpoint_kind", "best")),
        )
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "config" not in checkpoint or "model" not in checkpoint:
            raise KeyError(f"Invalid tactile point VAE checkpoint payload: {checkpoint_path}")
        checkpoint_config = OmegaConf.create(checkpoint["config"])
        checkpoint_summary = checkpoint.get("summary", {})
        checkpoint_data_summary = checkpoint_summary.get("data_summary", checkpoint_summary)
        checkpoint_frame = _coordinate_frame_from_config(checkpoint_config) or _coordinate_frame_from_summary(checkpoint_data_summary)
        self._assert_coordinate_frame(checkpoint_frame, source=str(checkpoint_path))
        model = TactilePointVAE.from_config(config=checkpoint_config, data_summary=checkpoint_data_summary)
        if int(model.latent_dim) != int(self.latent_dim):
            raise ValueError(
                f"Configured latent_dim={self.latent_dim} does not match checkpoint latent_dim={model.latent_dim}: "
                f"{checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model"])
        model.to(self.device)
        model.eval()
        self._checkpoint_coordinate_frame = checkpoint_frame
        self._checkpoint_path = checkpoint_path
        self._model = model
        return model

    def build_med(self, sample: Any) -> torch.Tensor:
        source_key = self._resolve_source_key(sample)
        if source_key not in sample.intermediates:
            raise KeyError(f"Sample intermediates missing VAE point-cloud source key: {source_key}")
        selected_fields = sample.metadata.get("med_selected_fields") or {}
        self._assert_coordinate_frame(selected_fields.get("coordinate_frame"), source="sample metadata")
        model = self._load_checkpoint_model()
        point_cloud = sample.intermediates[source_key].to(dtype=torch.float32)
        if point_cloud.dim() != 3:
            raise ValueError(f"TactilePointVAE_Latent_Med expects source shape [T,N,C], got {tuple(point_cloud.shape)}")
        if int(point_cloud.shape[-1]) != int(model.input_dim):
            raise ValueError(
                f"Tactile point VAE checkpoint expects per-point feature dim={model.input_dim}, "
                f"but source has dim={point_cloud.shape[-1]}. "
                "Use a VAE data preset with point_cloud.include_valid_feature=true."
            )
        with torch.no_grad():
            latent = model.encode_mu(point_cloud.unsqueeze(0).to(device=self.device))[0]
        return latent.detach().to(dtype=torch.float32, device=torch.device("cpu"))

    def infer_output_spec(self, data_summary: Dict[str, Any]) -> Dict[str, Any] | None:
        observed_frame = _coordinate_frame_from_summary(data_summary)
        self._assert_coordinate_frame(observed_frame, source="data summary")
        reference = _reference_summary(data_summary)
        med_steps = reference.get("med_steps", reference.get("future_tactile_steps"))
        if med_steps is None:
            source_shape = reference.get("med_source_shape")
            if source_shape is not None and len(source_shape) > 0:
                med_steps = int(source_shape[0])
        if med_steps is None:
            return None
        shape = [int(med_steps), int(self.latent_dim)]
        return {
            "kind": "vector",
            "shape": shape,
            "flat_dim": int(shape[0] * shape[1]),
            "provider": self.name,
            "semantic_kind": "tactile_point_vae_latent",
            "latent_dim": int(self.latent_dim),
            "coordinate_frame": self.configured_coordinate_frame,
            "source_kind": "point_cloud",
        }
