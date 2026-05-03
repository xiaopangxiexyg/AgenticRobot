"""Backward-compatible facade for shared training runtime symbols."""

from Training.bootstrap import bootstrap_project_root
from Training.engine import SupervisedExperimentTrainer, resolve_device

__all__ = ["bootstrap_project_root", "resolve_device", "SupervisedExperimentTrainer"]
