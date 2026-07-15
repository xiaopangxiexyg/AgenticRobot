"""Native DexTeleop v2.2 dataset export for MuJoCo.

This package deliberately does not depend on the legacy replay dataset writer.
It produces the model-facing v2.2 policy I/O fields directly from the MuJoCo
scene and records sim-only provenance separately from policy data.
"""

from .layout import (
    DEFAULT_RM65_BIMANUAL_DMTAC_LAYOUT_PATH,
    load_rm65_bimanual_dmtac_layout,
)
from .writer import MujocoV22DatasetWriter, MujocoV22DatasetWriterConfig

__all__ = [
    "DEFAULT_RM65_BIMANUAL_DMTAC_LAYOUT_PATH",
    "MujocoV22DatasetWriter",
    "MujocoV22DatasetWriterConfig",
    "load_rm65_bimanual_dmtac_layout",
]
