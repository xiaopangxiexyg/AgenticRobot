from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .action import ActionSpec
from .observation import ObservationSpec
from .scene import SceneSpec
from .timing import TimingSpec
from .versions import SIMULATION_CONTRACT_VERSION, require_version


@dataclass(frozen=True)
class SimulationContract:
    schema_version: str = SIMULATION_CONTRACT_VERSION
    name: str = "rm65_bimanual_physics_v1"
    action: ActionSpec = field(default_factory=ActionSpec)
    observation: ObservationSpec = field(default_factory=ObservationSpec)
    timing: TimingSpec = field(default_factory=TimingSpec)
    scene: SceneSpec = field(default_factory=SceneSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_version(
            self.schema_version,
            SIMULATION_CONTRACT_VERSION,
            field="simulation contract version",
        )
        if not str(self.name).strip():
            raise ValueError("SimulationContract.name must be non-empty.")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "action": self.action.to_dict(),
            "observation": self.observation.to_dict(),
            "timing": self.timing.to_dict(),
            "scene": self.scene.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SimulationContract":
        return cls(
            schema_version=str(
                payload.get("schema_version", SIMULATION_CONTRACT_VERSION)
            ),
            name=str(payload.get("name", "rm65_bimanual_physics_v1")),
            action=ActionSpec.from_dict(dict(payload.get("action", {}))),
            observation=ObservationSpec.from_dict(dict(payload.get("observation", {}))),
            timing=TimingSpec.from_dict(dict(payload.get("timing", {}))),
            scene=SceneSpec.from_dict(dict(payload.get("scene", {}))),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SimulationContract":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Simulation contract {path} must contain a JSON object.")
        return cls.from_dict(payload)


__all__ = ["SimulationContract"]
