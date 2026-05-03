from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from Models.Decoders.med_head import MedMLPHead
from Models.Decoders.point_cloud_head import PointCloudQueryHead
from Models.Encoders.med_projector import MedMLPProjector
from Models.Encoders.pointnet import PointNetEncoder
from Models.Policies.policy_support import resolve_hidden_dims


def _product(values: Sequence[int]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def normalize_med_spec(med_spec: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(med_spec)
    normalized["kind"] = str(normalized.get("kind", "vector")).lower()
    shape = normalized.get("shape", [])
    normalized["shape"] = [int(value) for value in shape]
    flat_dim = normalized.get("flat_dim")
    if flat_dim is None:
        if not normalized["shape"]:
            raise ValueError("med_spec must define either flat_dim or shape.")
        flat_dim = _product(normalized["shape"])
    normalized["flat_dim"] = int(flat_dim)
    if normalized["kind"] == "point_cloud":
        if len(normalized["shape"]) < 2:
            raise ValueError(f"Point-cloud med_spec requires shape with at least 2 dims, got {normalized['shape']}")
        normalized["point_feature_dim"] = int(normalized.get("point_feature_dim", normalized["shape"][-1]))
        if len(normalized["shape"]) >= 3:
            normalized["point_count"] = int(normalized["shape"][-2])
            normalized["sequence_length"] = int(normalized.get("sequence_length", _product(normalized["shape"][:-2])))
            normalized["sequence_point_count"] = int(_product(normalized["shape"][:-1]))
        elif "point_count" in normalized:
            normalized["point_count"] = int(normalized["point_count"])
        else:
            normalized["point_count"] = int(_product(normalized["shape"][:-1]))
    return normalized


def resolve_med_output_shape(med_spec: Dict[str, Any]) -> list[int] | None:
    normalized_spec = normalize_med_spec(med_spec)
    if len(normalized_spec["shape"]) <= 1:
        return None
    return [int(value) for value in normalized_spec["shape"]]


def resolve_med_sequence_shape(med_spec: Dict[str, Any]) -> list[int]:
    normalized_spec = normalize_med_spec(med_spec)
    shape = [int(value) for value in normalized_spec["shape"]]
    if not shape:
        return [1, int(normalized_spec["flat_dim"])]
    if len(shape) == 1:
        return [1, int(shape[0])]
    return [_product(shape[:-1]), int(shape[-1])]


def flatten_med_to_sequence(med: torch.Tensor, med_spec: Dict[str, Any]) -> torch.Tensor:
    sequence_length, feature_dim = resolve_med_sequence_shape(med_spec)
    if med.dim() == 1:
        raise ValueError(f"MED tensor is expected to include a batch dimension, got shape={tuple(med.shape)}")
    return med.reshape(med.shape[0], int(sequence_length), int(feature_dim))


def restore_sequence_to_med(sequence: torch.Tensor, med_spec: Dict[str, Any]) -> torch.Tensor:
    normalized_spec = normalize_med_spec(med_spec)
    if sequence.dim() != 3:
        raise ValueError(f"Expected MED diffusion output shape [B, L, C], got {tuple(sequence.shape)}")
    if len(normalized_spec["shape"]) <= 1:
        return sequence.reshape(sequence.shape[0], int(normalized_spec["flat_dim"]))
    return sequence.reshape(sequence.shape[0], *normalized_spec["shape"])


def _med_branch_config(config: Any) -> Any:
    return getattr(config.model, "med_branch", None)


def _resolve_requested_branch_type(config: Any, branch_name: str) -> str:
    med_branch_config = _med_branch_config(config)
    if med_branch_config is None:
        return "auto"
    branch_config = getattr(med_branch_config, branch_name, None)
    if branch_config is not None and getattr(branch_config, "type", None) is not None:
        return str(branch_config.type).lower()
    legacy_type = getattr(med_branch_config, "type", None)
    if legacy_type is not None:
        normalized_legacy_type = str(legacy_type).lower()
        if branch_name == "predictor" and normalized_legacy_type == "pointnet":
            return "point_query"
        return normalized_legacy_type
    return "auto"


def resolve_med_projector_type(config: Any, med_spec: Dict[str, Any]) -> str:
    normalized_spec = normalize_med_spec(med_spec)
    requested_type = _resolve_requested_branch_type(config, "projector")
    if normalized_spec["kind"] == "vector":
        if requested_type not in {"auto", "mlp"}:
            raise ValueError(
                f"Configured model.med_branch.projector.type={requested_type} requires point-cloud med, "
                f"but current med kind is {normalized_spec['kind']}."
            )
        return "mlp"
    if normalized_spec["kind"] != "point_cloud":
        raise ValueError(f"Unsupported med kind: {normalized_spec['kind']}")
    if requested_type == "auto":
        return "mlp"
    if requested_type not in {"mlp", "pointnet"}:
        raise ValueError(
            f"Unsupported model.med_branch.projector.type={requested_type}. "
            "Expected one of: auto, mlp, pointnet."
        )
    return requested_type


def resolve_med_predictor_type(config: Any, med_spec: Dict[str, Any]) -> str:
    normalized_spec = normalize_med_spec(med_spec)
    requested_type = _resolve_requested_branch_type(config, "predictor")
    if normalized_spec["kind"] == "vector":
        if requested_type not in {"auto", "mlp"}:
            raise ValueError(
                f"Configured model.med_branch.predictor.type={requested_type} requires point-cloud med, "
                f"but current med kind is {normalized_spec['kind']}."
            )
        return "mlp"
    if normalized_spec["kind"] != "point_cloud":
        raise ValueError(f"Unsupported med kind: {normalized_spec['kind']}")
    if requested_type == "auto":
        return "mlp"
    if requested_type not in {"mlp", "point_query"}:
        raise ValueError(
            f"Unsupported model.med_branch.predictor.type={requested_type}. "
            "Expected one of: auto, mlp, point_query."
        )
    return requested_type


def _resolve_pointnet_hidden_dims(config: Any, fallback_dim: int) -> list[int]:
    med_branch_config = _med_branch_config(config)
    projector_config = None if med_branch_config is None else getattr(med_branch_config, "projector", None)
    hidden_dims = None
    if projector_config is not None:
        hidden_dims = getattr(projector_config, "pointnet_hidden_dims", None)
    if hidden_dims is None and med_branch_config is not None:
        hidden_dims = getattr(med_branch_config, "pointnet_hidden_dims", None)
    if hidden_dims is None:
        return [max(64, int(fallback_dim) // 2), int(fallback_dim)]
    return [int(dim) for dim in hidden_dims]


def _resolve_point_query_dim(config: Any, fallback_dim: int) -> int:
    med_branch_config = _med_branch_config(config)
    predictor_config = None if med_branch_config is None else getattr(med_branch_config, "predictor", None)
    query_dim = None
    if predictor_config is not None:
        query_dim = getattr(predictor_config, "point_query_dim", None)
    if query_dim is None and med_branch_config is not None:
        query_dim = getattr(med_branch_config, "point_query_dim", None)
    if query_dim is None and med_branch_config is not None:
        query_dim = getattr(med_branch_config, "pointnet_query_dim", None)
    if query_dim is None:
        return int(fallback_dim)
    return int(query_dim)


def build_med_projector(config: Any, med_spec: Dict[str, Any], feature_dim: int) -> nn.Module:
    normalized_spec = normalize_med_spec(med_spec)
    branch_type = resolve_med_projector_type(config, normalized_spec)
    if branch_type == "mlp":
        return MedMLPProjector(
            input_dim=int(normalized_spec["flat_dim"]),
            hidden_dims=resolve_hidden_dims(config.model, "med_projector_hidden_dims", default=[]),
            feature_dim=int(feature_dim),
            activation=str(config.model.activation),
            dropout=float(config.model.dropout),
        )
    return PointNetEncoder(
        input_dim=int(normalized_spec["point_feature_dim"]),
        hidden_dims=_resolve_pointnet_hidden_dims(config, fallback_dim=feature_dim),
        output_dim=int(feature_dim),
        activation=str(config.model.activation),
        dropout=float(config.model.dropout),
    )


def build_med_predictor(config: Any, med_spec: Dict[str, Any], input_dim: int) -> nn.Module:
    normalized_spec = normalize_med_spec(med_spec)
    branch_type = resolve_med_predictor_type(config, normalized_spec)
    if branch_type == "mlp":
        return MedMLPHead(
            input_dim=int(input_dim),
            hidden_dims=resolve_hidden_dims(config.model, "med_predictor_hidden_dims", default=[input_dim]),
            med_dim=int(normalized_spec["flat_dim"]),
            activation=str(config.model.activation),
            dropout=float(config.model.dropout),
            output_shape=resolve_med_output_shape(normalized_spec),
        )
    if branch_type != "point_query":
        raise ValueError(f"Unsupported point-cloud predictor type: {branch_type}")
    return PointCloudQueryHead(
        input_dim=int(input_dim),
        output_shape=normalized_spec["shape"],
        hidden_dims=resolve_hidden_dims(config.model, "med_predictor_hidden_dims", default=[input_dim]),
        query_dim=_resolve_point_query_dim(config, fallback_dim=input_dim),
        activation=str(config.model.activation),
        dropout=float(config.model.dropout),
    )
