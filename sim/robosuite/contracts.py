"""Content-addressed reset/replay contracts for the robosuite A0 seam."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash

ROBOSUITE_RESET_MANIFEST_SCHEMA_A0 = "dexteleop.robosuite.reset_manifest.a0"
ROBOSUITE_RESET_MANIFEST_HASH_DOMAIN_A0 = "dexteleop.robosuite.reset_manifest.a0"
ROBOSUITE_RESET_RECEIPT_HASH_DOMAIN_A0 = "dexteleop.robosuite.reset_receipt.a0"


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _freeze_primitive(value: Any, *, name: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite float.")
        return float(value)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_primitive(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _required(key, name=f"{name}.key")
            if normalized_key in normalized:
                raise ValueError(f"{name} contains duplicate normalized keys.")
            normalized[normalized_key] = _freeze_primitive(
                item,
                name=f"{name}[{normalized_key!r}]",
            )
        return MappingProxyType(dict(sorted(normalized.items())))
    raise TypeError(
        f"{name} accepts only explicit primitive reset options; "
        f"got {type(value).__name__}."
    )


@dataclass(frozen=True)
class RobosuiteResetManifest:
    """One deterministic reset request that can be replayed by exact hash.

    The manifest does not claim that a real robosuite environment was started.
    A backend receipt is required to prove that a reset was actually applied.
    """

    environment_id: str
    environment_contract_hash: str
    task_id: str
    episode_id: str
    reset_sequence: int
    seed: int
    action_dimension: int
    observation_keys: tuple[str, ...]
    max_episode_steps: int
    options: Mapping[str, Any] = field(default_factory=dict)
    schema_id: str = ROBOSUITE_RESET_MANIFEST_SCHEMA_A0
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        schema_id = _required(self.schema_id, name="schema_id")
        if schema_id != ROBOSUITE_RESET_MANIFEST_SCHEMA_A0:
            raise ValueError(f"Unsupported robosuite reset schema {schema_id!r}.")
        environment_id = _required(self.environment_id, name="environment_id")
        environment_hash = _required(
            self.environment_contract_hash,
            name="environment_contract_hash",
        )
        task_id = _required(self.task_id, name="task_id")
        episode_id = _required(self.episode_id, name="episode_id")
        reset_sequence = _nonnegative_int(
            self.reset_sequence,
            name="reset_sequence",
        )
        seed = _nonnegative_int(self.seed, name="seed")
        action_dimension = _positive_int(
            self.action_dimension,
            name="action_dimension",
        )
        max_episode_steps = _positive_int(
            self.max_episode_steps,
            name="max_episode_steps",
        )
        observation_keys = tuple(
            _required(key, name="observation_keys.item")
            for key in self.observation_keys
        )
        if not observation_keys:
            raise ValueError("observation_keys must not be empty.")
        if len(observation_keys) != len(set(observation_keys)):
            raise ValueError("observation_keys must be unique.")
        options = _freeze_primitive(dict(self.options), name="options")
        payload = {
            "schema_id": schema_id,
            "environment_id": environment_id,
            "environment_contract_hash": environment_hash,
            "task_id": task_id,
            "episode_id": episode_id,
            "reset_sequence": reset_sequence,
            "seed": seed,
            "action_dimension": action_dimension,
            "observation_keys": observation_keys,
            "max_episode_steps": max_episode_steps,
            "options": options,
            "reset_semantics": "seeded_full_environment_reset",
            "replay_binding": "exact_manifest_hash",
        }
        for name, value in payload.items():
            if name not in {"reset_semantics", "replay_binding"}:
                object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=ROBOSUITE_RESET_MANIFEST_HASH_DOMAIN_A0,
            ),
        )

    @property
    def manifest_id(self) -> str:
        return f"robosuite-reset-manifest:{self.semantic_content_hash}"

    def to_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_id": self.schema_id,
                "environment_id": self.environment_id,
                "environment_contract_hash": self.environment_contract_hash,
                "task_id": self.task_id,
                "episode_id": self.episode_id,
                "reset_sequence": self.reset_sequence,
                "seed": self.seed,
                "action_dimension": self.action_dimension,
                "observation_keys": self.observation_keys,
                "max_episode_steps": self.max_episode_steps,
                "options": self.options,
            }
        )


@dataclass(frozen=True)
class RobosuiteResetReceipt:
    """Deterministic proof that one backend reset used one exact manifest."""

    backend_id: str
    manifest_id: str
    manifest_hash: str
    initial_observation_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "backend_id": _required(self.backend_id, name="backend_id"),
            "manifest_id": _required(self.manifest_id, name="manifest_id"),
            "manifest_hash": _required(self.manifest_hash, name="manifest_hash"),
            "initial_observation_hash": _required(
                self.initial_observation_hash,
                name="initial_observation_hash",
            ),
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=ROBOSUITE_RESET_RECEIPT_HASH_DOMAIN_A0,
            ),
        )

    @property
    def receipt_id(self) -> str:
        return f"robosuite-reset-receipt:{self.semantic_content_hash}"


__all__ = [
    "ROBOSUITE_RESET_MANIFEST_HASH_DOMAIN_A0",
    "ROBOSUITE_RESET_MANIFEST_SCHEMA_A0",
    "ROBOSUITE_RESET_RECEIPT_HASH_DOMAIN_A0",
    "RobosuiteResetManifest",
    "RobosuiteResetReceipt",
]
