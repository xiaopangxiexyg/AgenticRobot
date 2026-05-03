from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from Models.med_support import normalize_med_spec

from .pointnext_style import PointNeXtStyleEncoder, PointNeXtStyleQueryDecoder


def _resolve_reference_summary(data_summary: Dict[str, Any]) -> Dict[str, Any]:
    for split_name in ("train", "val"):
        split_summary = data_summary.get(split_name)
        if split_summary is not None:
            return split_summary
    return data_summary


def _resolve_med_spec_from_summary(data_summary: Dict[str, Any]) -> Dict[str, Any]:
    reference = _resolve_reference_summary(data_summary)
    med_spec = data_summary.get("med_spec") or reference.get("med_spec")
    if med_spec is None:
        raise ValueError("TactilePointVAE requires data_summary.med_spec from a point-cloud med provider.")
    return normalize_med_spec(med_spec)


def _config_get(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default) if config is not None else default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_float_list(value: Any, *, length: int, default: list[float]) -> list[float]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if value.strip().lower() in {"auto", "null"}:
            return list(default)
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    elif isinstance(value, (int, float)):
        values = [float(value)]
    else:
        values = [float(item) for item in value]
    if len(values) == 1:
        values = values * int(length)
    if len(values) != int(length):
        raise ValueError(f"Expected {length} values, got {values}.")
    return values


def _resolve_selected_fields(data_summary: Dict[str, Any]) -> Dict[str, Any]:
    reference = _resolve_reference_summary(data_summary)
    return dict(reference.get("med_selected_fields") or {})


def _resolve_coordinate_frame(data_summary: Dict[str, Any]) -> str:
    selected_fields = _resolve_selected_fields(data_summary)
    return str(selected_fields.get("coordinate_frame", "camera")).strip().lower()


def _default_xyz_center_for_frame(coordinate_frame: str) -> list[float]:
    if coordinate_frame in {"wrist", "wrist_local", "eef", "eef_local", "hand", "hand_local"}:
        return [0.0, 0.0, 0.0]
    # Raw force-point xyz is first-person camera coordinates; z is centered
    # around the workspace depth so normalized geometry does not inherit a
    # large constant offset.
    return [0.0, 0.0, 0.5]


def _default_xyz_scale_for_frame(coordinate_frame: str) -> list[float]:
    if coordinate_frame in {"wrist", "wrist_local", "eef", "eef_local", "hand", "hand_local"}:
        return [0.1, 0.1, 0.1]
    return [0.25, 0.25, 0.25]


class TactilePointVAE(nn.Module):
    def __init__(
        self,
        *,
        point_count: int,
        input_dim: int,
        target_dim: int,
        latent_dim: int,
        valid_feature_index: int | None,
        encoder_width: int,
        encoder_depth: int,
        encoder_k_neighbors: int,
        decoder_query_dim: int,
        decoder_hidden_dims: list[int],
        decoder_refinement_blocks: int,
        decoder_k_neighbors: int,
        activation: str,
        dropout: float,
        beta: float,
        xyz_loss_weight: float,
        value_loss_weight: float,
        normalization_enabled: bool,
        xyz_center: list[float],
        xyz_scale: list[float],
        value_offset: float,
        value_scale: float,
    ) -> None:
        super().__init__()
        self.point_count = int(point_count)
        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)
        self.latent_dim = int(latent_dim)
        self.valid_feature_index = None if valid_feature_index is None else int(valid_feature_index)
        self.beta = float(beta)
        self.xyz_loss_weight = float(xyz_loss_weight)
        self.value_loss_weight = float(value_loss_weight)
        self.normalization_enabled = bool(normalization_enabled)
        xyz_scale_tensor = torch.tensor(xyz_scale, dtype=torch.float32).view(1, 1, 3).clamp_min(1.0e-8)
        value_scale_tensor = torch.tensor(float(value_scale), dtype=torch.float32).view(1, 1, 1).clamp_min(1.0e-8)
        self.register_buffer("xyz_center", torch.tensor(xyz_center, dtype=torch.float32).view(1, 1, 3), persistent=False)
        self.register_buffer("xyz_scale", xyz_scale_tensor, persistent=False)
        self.register_buffer("value_offset", torch.tensor(float(value_offset), dtype=torch.float32).view(1, 1, 1), persistent=False)
        self.register_buffer("value_scale", value_scale_tensor, persistent=False)
        self.encoder = PointNeXtStyleEncoder(
            input_dim=self.input_dim,
            width=int(encoder_width),
            depth=int(encoder_depth),
            k_neighbors=int(encoder_k_neighbors),
            output_dim=int(encoder_width),
            activation=activation,
            dropout=dropout,
        )
        self.to_mu = nn.Linear(int(encoder_width), self.latent_dim)
        self.to_logvar = nn.Linear(int(encoder_width), self.latent_dim)
        self.decoder = PointNeXtStyleQueryDecoder(
            point_count=self.point_count,
            latent_dim=self.latent_dim,
            query_dim=int(decoder_query_dim),
            hidden_dims=decoder_hidden_dims,
            output_dim=self.target_dim,
            refinement_blocks=int(decoder_refinement_blocks),
            k_neighbors=int(decoder_k_neighbors),
            activation=activation,
            dropout=dropout,
        )

    @classmethod
    def from_config(cls, config: Any, data_summary: Dict[str, Any]) -> "TactilePointVAE":
        model_config = config.model
        med_spec = _resolve_med_spec_from_summary(data_summary)
        if med_spec["kind"] != "point_cloud":
            raise ValueError(f"TactilePointVAE expects point-cloud med, got {med_spec['kind']!r}.")
        med_shape = [int(value) for value in med_spec["shape"]]
        if len(med_shape) < 2:
            raise ValueError(f"TactilePointVAE expects point-cloud med shape [N,C] or [T,N,C], got {med_shape}.")
        point_count = int(med_shape[-2])
        input_dim = int(med_shape[-1])
        target_dim = int(_config_get(model_config, "target_feature_dim", 4))
        if target_dim > input_dim:
            raise ValueError(f"target_feature_dim={target_dim} cannot exceed input point feature dim={input_dim}.")
        configured_valid_index = _config_get(model_config, "valid_feature_index", None)
        valid_feature_index = None
        if configured_valid_index not in (None, "null"):
            valid_feature_index = int(configured_valid_index)
            if valid_feature_index >= input_dim:
                valid_feature_index = None

        encoder_config = _config_get(model_config, "encoder", None)
        decoder_config = _config_get(model_config, "decoder", None)
        loss_config = _config_get(model_config, "loss", None)
        normalization_config = _config_get(model_config, "normalization", None)
        latent_dim = int(_config_get(model_config, "latent_dim", 32))
        activation = str(_config_get(model_config, "activation", "gelu"))
        dropout = float(_config_get(model_config, "dropout", 0.0))
        coordinate_frame = _resolve_coordinate_frame(data_summary)
        default_xyz_center = _default_xyz_center_for_frame(coordinate_frame)
        default_xyz_scale = _default_xyz_scale_for_frame(coordinate_frame)
        normalization_enabled = _as_bool(_config_get(normalization_config, "enabled", True), default=True)
        xyz_center = _as_float_list(
            _config_get(normalization_config, "xyz_center", "auto"),
            length=3,
            default=default_xyz_center,
        )
        xyz_scale = _as_float_list(
            _config_get(normalization_config, "xyz_scale", "auto"),
            length=3,
            default=default_xyz_scale,
        )
        return cls(
            point_count=point_count,
            input_dim=input_dim,
            target_dim=target_dim,
            latent_dim=latent_dim,
            valid_feature_index=valid_feature_index,
            encoder_width=int(_config_get(encoder_config, "width", 64)),
            encoder_depth=int(_config_get(encoder_config, "depth", 3)),
            encoder_k_neighbors=int(_config_get(encoder_config, "k_neighbors", 16)),
            decoder_query_dim=int(_config_get(decoder_config, "query_dim", 64)),
            decoder_hidden_dims=[int(dim) for dim in _config_get(decoder_config, "hidden_dims", [128, 128])],
            decoder_refinement_blocks=int(_config_get(decoder_config, "refinement_blocks", 2)),
            decoder_k_neighbors=int(_config_get(decoder_config, "k_neighbors", 16)),
            activation=activation,
            dropout=dropout,
            beta=float(_config_get(loss_config, "beta", 1.0e-3)),
            xyz_loss_weight=float(_config_get(loss_config, "xyz_weight", 1.0)),
            value_loss_weight=float(_config_get(loss_config, "value_weight", 1.0)),
            normalization_enabled=normalization_enabled,
            xyz_center=xyz_center,
            xyz_scale=xyz_scale,
            value_offset=float(_config_get(normalization_config, "value_offset", 0.0)),
            value_scale=float(_config_get(normalization_config, "value_scale", 32.0)),
        )

    def _valid_mask(self, points: torch.Tensor) -> torch.Tensor:
        if self.valid_feature_index is None:
            return torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        return points[..., self.valid_feature_index] > 0.5

    def _flatten_frames(self, points: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...] | None]:
        if points.dim() == 4:
            batch_size, time_steps, point_count, feature_dim = points.shape
            if int(point_count) != self.point_count:
                raise ValueError(f"Expected point_count={self.point_count}, got {point_count}.")
            return points.reshape(batch_size * time_steps, point_count, feature_dim), (batch_size, time_steps)
        if points.dim() == 3:
            if int(points.shape[1]) != self.point_count:
                raise ValueError(f"Expected point_count={self.point_count}, got {points.shape[1]}.")
            return points, None
        raise ValueError(f"Expected point tensor shape [B,N,C] or [B,T,N,C], got {tuple(points.shape)}.")

    def _normalize_target(self, target: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self.normalization_enabled:
            return target
        normalized = target.clone()
        normalized[..., :3] = (normalized[..., :3] - self.xyz_center.to(normalized.device)) / self.xyz_scale.to(normalized.device)
        if self.target_dim > 3:
            normalized[..., 3:self.target_dim] = (
                normalized[..., 3:self.target_dim] - self.value_offset.to(normalized.device)
            ) / self.value_scale.to(normalized.device)
        if valid_mask is not None:
            normalized = normalized * valid_mask.to(dtype=normalized.dtype, device=normalized.device).unsqueeze(-1)
        return normalized

    def _denormalize_target(self, target: torch.Tensor) -> torch.Tensor:
        if not self.normalization_enabled:
            return target
        denormalized = target.clone()
        denormalized[..., :3] = denormalized[..., :3] * self.xyz_scale.to(denormalized.device) + self.xyz_center.to(denormalized.device)
        if self.target_dim > 3:
            denormalized[..., 3:self.target_dim] = (
                denormalized[..., 3:self.target_dim] * self.value_scale.to(denormalized.device)
                + self.value_offset.to(denormalized.device)
            )
        return denormalized

    def _normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        if not self.normalization_enabled:
            return points
        normalized = points.clone()
        valid_mask = self._valid_mask(points)
        normalized[..., : self.target_dim] = self._normalize_target(
            normalized[..., : self.target_dim],
            valid_mask=valid_mask,
        )
        return normalized

    def _encode_normalized(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if int(points.shape[-1]) != self.input_dim:
            raise ValueError(f"Expected point feature dim={self.input_dim}, got {points.shape[-1]}.")
        valid_mask = self._valid_mask(points)
        encoded = self.encoder(points, valid_mask=valid_mask)
        return self.to_mu(encoded), self.to_logvar(encoded)

    def encode(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_normalized(self._normalize_points(points))

    def encode_mu(self, points: torch.Tensor) -> torch.Tensor:
        flattened, original_shape = self._flatten_frames(points)
        mu, _ = self.encode(flattened)
        if original_shape is None:
            return mu
        batch_size, time_steps = original_shape
        return mu.reshape(batch_size, time_steps, self.latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        normalized_points = self._normalize_points(points)
        mu, logvar = self._encode_normalized(normalized_points)
        latent = self.reparameterize(mu, logvar)
        reconstruction_normalized = self.decode(latent)
        reconstruction = self._denormalize_target(reconstruction_normalized)
        return {
            "reconstruction": reconstruction,
            "reconstruction_normalized": reconstruction_normalized,
            "mu": mu,
            "logvar": logvar,
            "latent": latent,
        }

    def compute_custom_loss(self, model_inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]):
        del model_inputs
        raw_points = targets["med"].to(dtype=torch.float32)
        points, _ = self._flatten_frames(raw_points)
        outputs = self(points)
        target = points[..., : self.target_dim]
        valid_mask = self._valid_mask(points).to(dtype=target.dtype)
        valid_count = valid_mask.sum().clamp_min(1.0)

        normalized_target = self._normalize_target(target, valid_mask=valid_mask)
        reconstruction = outputs["reconstruction_normalized"]
        xyz_error = (reconstruction[..., :3] - normalized_target[..., :3]).pow(2) * valid_mask.unsqueeze(-1)
        xyz_loss = xyz_error.sum() / (valid_count * 3.0)
        if self.target_dim > 3:
            value_error = (reconstruction[..., 3:self.target_dim] - normalized_target[..., 3:self.target_dim]).pow(2)
            value_error = value_error * valid_mask.unsqueeze(-1)
            value_loss = value_error.sum() / (valid_count * float(self.target_dim - 3))
        else:
            value_loss = torch.zeros((), dtype=target.dtype, device=target.device)
        kl_loss = -0.5 * torch.mean(1.0 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp())
        reconstruction_loss = self.xyz_loss_weight * xyz_loss + self.value_loss_weight * value_loss
        total_loss = reconstruction_loss + self.beta * kl_loss

        raw_reconstruction = outputs["reconstruction"]
        xyz_raw_error = (raw_reconstruction[..., :3] - target[..., :3]).pow(2) * valid_mask.unsqueeze(-1)
        xyz_raw_loss = xyz_raw_error.sum() / (valid_count * 3.0)
        if self.target_dim > 3:
            value_raw_error = (raw_reconstruction[..., 3:self.target_dim] - target[..., 3:self.target_dim]).pow(2)
            value_raw_error = value_raw_error * valid_mask.unsqueeze(-1)
            value_raw_loss = value_raw_error.sum() / (valid_count * float(self.target_dim - 3))
        else:
            value_raw_loss = torch.zeros((), dtype=target.dtype, device=target.device)
        return outputs, total_loss, {
            "reconstruction": reconstruction_loss,
            "xyz": xyz_loss,
            "value": value_loss,
            "xyz_raw": xyz_raw_loss,
            "value_raw": value_raw_loss,
            "kl": kl_loss,
        }
