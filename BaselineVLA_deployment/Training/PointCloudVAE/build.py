from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from Models.PointCloud.tactile_point_vae import TactilePointVAE
from Training.build_utils import build_adam_optimizer


def build_pcvae_model(config: Any, data_summary: Dict[str, Any]) -> nn.Module:
    return TactilePointVAE.from_config(config=config, data_summary=data_summary)


def build_model_and_optimizer(config: Any, data_summary: Dict[str, Any]) -> tuple[nn.Module, torch.optim.Optimizer]:
    model = build_pcvae_model(config=config, data_summary=data_summary)
    optimizer = build_adam_optimizer(config, model)
    return model, optimizer
