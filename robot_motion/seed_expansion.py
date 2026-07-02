from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SeedPerturbation:
    label: str
    offset_deg: tuple[float, ...]
    scale: tuple[float, ...] = ()
    source_label_allowlist: tuple[str, ...] = ()

    def expanded(self, seed_qpos: np.ndarray) -> np.ndarray:
        seed = np.asarray(seed_qpos, dtype=float).reshape(-1)
        scale = (
            np.ones(seed.shape, dtype=float)
            if not self.scale
            else np.asarray(self.scale, dtype=float).reshape(seed.shape)
        )
        offset = np.deg2rad(np.asarray(self.offset_deg, dtype=float).reshape(seed.shape))
        return seed * scale + offset


@dataclass(frozen=True)
class SeedExpansionProfile:
    name: str
    dof: int
    perturbations: tuple[SeedPerturbation, ...]
    enabled: bool = True
    max_extra_seeds: int = 16
    joint_names: tuple[str, ...] = ()
    source_label_allowlist: tuple[str, ...] = ()
    dedupe_atol_rad: float = 1e-7
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.dof) <= 0:
            raise ValueError("seed expansion dof must be positive")
        if int(self.max_extra_seeds) < 0:
            raise ValueError("max_extra_seeds must be >= 0")
        if self.joint_names and len(self.joint_names) != int(self.dof):
            raise ValueError("joint_names must match dof")
        if float(self.dedupe_atol_rad) < 0.0:
            raise ValueError("dedupe_atol_rad must be >= 0")
        for perturbation in self.perturbations:
            if len(perturbation.offset_deg) != int(self.dof):
                raise ValueError(f"perturbation {perturbation.label!r} offset_deg must match dof")
            if perturbation.scale and len(perturbation.scale) != int(self.dof):
                raise ValueError(f"perturbation {perturbation.label!r} scale must match dof")


@dataclass(frozen=True)
class SeedExpansionResult:
    seeds: tuple[np.ndarray, ...]
    labels: tuple[str, ...]
    metadata: dict[str, Any]


def default_seed_expansion_profiles_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "ik_seed_expansion_profiles.json"


def seed_expansion_profile_from_dict(name: str, payload: dict[str, Any]) -> SeedExpansionProfile:
    dof = int(payload.get("dof", len(payload.get("joint_names", ())) or 0))
    perturbations = tuple(
        SeedPerturbation(
            label=str(item["label"]),
            offset_deg=tuple(float(value) for value in item.get("offset_deg", ())),
            scale=tuple(float(value) for value in item.get("scale", ())),
            source_label_allowlist=tuple(
                str(value) for value in item.get("source_label_allowlist", ())
            ),
        )
        for item in payload.get("perturbations", ())
    )
    if dof <= 0 and perturbations:
        dof = len(perturbations[0].offset_deg)
    return SeedExpansionProfile(
        name=name,
        dof=dof,
        perturbations=perturbations,
        enabled=bool(payload.get("enabled", True)),
        max_extra_seeds=int(payload.get("max_extra_seeds", 16)),
        joint_names=tuple(str(value) for value in payload.get("joint_names", ())),
        source_label_allowlist=tuple(str(value) for value in payload.get("source_label_allowlist", ())),
        dedupe_atol_rad=float(payload.get("dedupe_atol_rad", 1e-7)),
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
    )


def load_seed_expansion_profiles(path: str | Path | None = None) -> dict[str, SeedExpansionProfile]:
    resolved = default_seed_expansion_profiles_path() if path in (None, "") else Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", payload)
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"{resolved} does not contain any seed expansion profiles")
    return {
        str(name): seed_expansion_profile_from_dict(str(name), dict(profile_payload))
        for name, profile_payload in profiles.items()
    }


def expand_seed_qpos(
    *,
    seeds: tuple[np.ndarray, ...] | list[np.ndarray],
    labels: tuple[str, ...] | list[str],
    profile: SeedExpansionProfile | None,
) -> SeedExpansionResult:
    base_seeds = [np.asarray(seed, dtype=float).reshape(-1) for seed in seeds]
    base_labels = [str(label) for label in labels]
    if profile is None or not bool(profile.enabled) or int(profile.max_extra_seeds) <= 0:
        return SeedExpansionResult(
            seeds=tuple(seed.copy() for seed in base_seeds),
            labels=tuple(base_labels),
            metadata=_seed_expansion_metadata(profile=profile, generated=[]),
        )
    if not base_seeds:
        return SeedExpansionResult(
            seeds=(),
            labels=(),
            metadata=_seed_expansion_metadata(profile=profile, generated=[]),
        )
    dof = int(profile.dof)
    allowed_sources = {str(value) for value in profile.source_label_allowlist}
    output_seeds: list[np.ndarray] = []
    output_labels: list[str] = []
    generated: list[dict[str, Any]] = []

    def append_unique(seed: np.ndarray, label: str) -> bool:
        candidate = np.asarray(seed, dtype=float).reshape(-1)
        if candidate.shape != (dof,) or not np.all(np.isfinite(candidate)):
            return False
        if any(np.allclose(candidate, existing, atol=float(profile.dedupe_atol_rad)) for existing in output_seeds):
            return False
        output_seeds.append(candidate.copy())
        output_labels.append(str(label))
        return True

    for base_seed, base_label in zip(base_seeds, base_labels):
        append_unique(base_seed, base_label)
        if len(generated) >= int(profile.max_extra_seeds):
            continue
        if allowed_sources and str(base_label) not in allowed_sources:
            continue
        for perturbation in profile.perturbations:
            if len(generated) >= int(profile.max_extra_seeds):
                break
            perturb_sources = {str(value) for value in perturbation.source_label_allowlist}
            if perturb_sources and str(base_label) not in perturb_sources:
                continue
            candidate = perturbation.expanded(base_seed)
            expanded_label = f"{base_label}:{perturbation.label}"
            if not append_unique(candidate, expanded_label):
                continue
            generated.append(
                {
                    "label": expanded_label,
                    "base_label": str(base_label),
                    "perturbation": perturbation.label,
                    "offset_deg": list(perturbation.offset_deg),
                    "scale": list(perturbation.scale),
                }
            )

    return SeedExpansionResult(
        seeds=tuple(seed.copy() for seed in output_seeds),
        labels=tuple(output_labels),
        metadata=_seed_expansion_metadata(profile=profile, generated=generated),
    )


def _seed_expansion_metadata(
    *,
    profile: SeedExpansionProfile | None,
    generated: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "enabled": bool(profile is not None and profile.enabled),
        "profile": "" if profile is None else profile.name,
        "max_extra_seeds": 0 if profile is None else int(profile.max_extra_seeds),
        "generated_count": len(generated),
        "generated_labels": [str(item["label"]) for item in generated],
        "generated": generated,
    }
