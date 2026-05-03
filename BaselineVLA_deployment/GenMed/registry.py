from __future__ import annotations

from typing import Any

from .base import MedAugmentedDataset
from .future_point_cloud_med import FutureRightPointCloudMedProvider
from .future_sequence_med import FutureSequenceMedProvider
from .future_tactile_med import FutureRightTactileMedProvider
from .mock_med import MockMedProvider
from .tactile_point_vae_latent_med import TactilePointVAELatentMedProvider


MED_PROVIDERS = {
    "FutureSequence_Med": FutureSequenceMedProvider,
    "FutureRightPointCloud_Med": FutureRightPointCloudMedProvider,
    "FutureRightTactile_Med": FutureRightTactileMedProvider,
    "Mock_Med": MockMedProvider,
    "TactilePointVAE_Latent_Med": TactilePointVAELatentMedProvider,
}


def build_med_provider(config: Any):
    provider_type = str(config.provider_type)
    if provider_type not in MED_PROVIDERS:
        raise KeyError(f"Unknown med provider: {provider_type}")
    return MED_PROVIDERS[provider_type](config)


def med_enabled_for_experiment(med_config: Any, experiment_name: str) -> bool:
    return bool(med_config.enabled_for_experiments.get(str(experiment_name), False))


def attach_med_provider(dataset: Any, med_provider: Any):
    if med_provider is None:
        return dataset
    return MedAugmentedDataset(dataset, med_provider)
