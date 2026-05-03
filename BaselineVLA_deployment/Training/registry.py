from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Type

from Training.FullTrain.build import build_model_and_optimizer as build_fulltrain_model_and_optimizer
from Training.FullTrain.trainer import FullTrainTrainer
from Training.GenMed.build import build_model_and_optimizer as build_genmed_model_and_optimizer
from Training.GenMed.trainer import GenMedTrainer
from Training.GTMed.build import build_model_and_optimizer as build_gtmed_model_and_optimizer
from Training.GTMed.trainer import GTMedTrainer
from Training.PointCloudVAE.build import build_model_and_optimizer as build_pcvae_model_and_optimizer
from Training.PointCloudVAE.trainer import PointCloudVAETrainer
from Training.VanillaTrain.build import (
    build_model_and_optimizer as build_vanilla_model_and_optimizer,
    build_tactile_model_and_optimizer as build_vanilla_tactile_model_and_optimizer,
)
from Training.VanillaTrain.trainer import VanillaTrainer


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    default_config_name: str
    build_model_and_optimizer: Callable[[Any, Dict[str, Any]], Any]
    trainer_cls: Type[Any]


EXPERIMENT_SPECS: Dict[str, ExperimentSpec] = {
    "vanilla": ExperimentSpec(
        name="vanilla",
        default_config_name="vanilla",
        build_model_and_optimizer=build_vanilla_model_and_optimizer,
        trainer_cls=VanillaTrainer,
    ),
    "vanilla-tactile": ExperimentSpec(
        name="vanilla-tactile",
        default_config_name="vanilla_tactile",
        build_model_and_optimizer=build_vanilla_tactile_model_and_optimizer,
        trainer_cls=VanillaTrainer,
    ),
    "gtmed": ExperimentSpec(
        name="gtmed",
        default_config_name="gtmed",
        build_model_and_optimizer=build_gtmed_model_and_optimizer,
        trainer_cls=GTMedTrainer,
    ),
    "fulltrain": ExperimentSpec(
        name="fulltrain",
        default_config_name="fulltrain",
        build_model_and_optimizer=build_fulltrain_model_and_optimizer,
        trainer_cls=FullTrainTrainer,
    ),
    "genmed": ExperimentSpec(
        name="genmed",
        default_config_name="genmed",
        build_model_and_optimizer=build_genmed_model_and_optimizer,
        trainer_cls=GenMedTrainer,
    ),
    "pcvae": ExperimentSpec(
        name="pcvae",
        default_config_name="pcvae",
        build_model_and_optimizer=build_pcvae_model_and_optimizer,
        trainer_cls=PointCloudVAETrainer,
    ),
}


def list_experiment_names() -> list[str]:
    return sorted(EXPERIMENT_SPECS.keys())


def get_experiment_spec(experiment_name: str) -> ExperimentSpec:
    normalized = str(experiment_name).lower()
    if normalized not in EXPERIMENT_SPECS:
        raise KeyError(
            f"Unknown experiment '{experiment_name}'. Available experiments: {', '.join(list_experiment_names())}"
        )
    return EXPERIMENT_SPECS[normalized]
