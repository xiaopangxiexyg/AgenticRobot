"""Selection of complete embodiment-specific application implementations.

An application suite is a vertical integration boundary, not a request to
split embodiment-specific teleoperation, control, reading, dispatch, or asset
logic into common algorithms.  The portable spec pins those complete surfaces
by semantic identity.  An application composition root later binds each
surface to one explicit lazy loader.

Catalog listing and selection never load an implementation.  Loading one
selected surface returns an opaque implementation envelope; this module does
not call it, construct a target handle, start a device, or own its lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash


EMBODIMENT_APPLICATION_SURFACE_SPEC_SCHEMA_V1 = (
    "dexteleop.embodiment_application_surface_spec.v1"
)
EMBODIMENT_APPLICATION_SUITE_SPEC_SCHEMA_V1 = (
    "dexteleop.embodiment_application_suite_spec.v1"
)
EMBODIMENT_APPLICATION_SURFACE_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.embodiment_application_surface_spec.v1"
)
EMBODIMENT_APPLICATION_SUITE_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.embodiment_application_suite_spec.v1"
)


def _required(value: object, *, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must be non-empty.")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _require_exact_keys(
    payload: Mapping[str, Any],
    *,
    expected: set[str],
    name: str,
) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{name} fields differ from schema: "
            f"missing={sorted(expected - actual)!r} "
            f"extra={sorted(actual - expected)!r}."
        )


def _identifiers(
    values: Iterable[object],
    *,
    name: str,
    sort_values: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} values must be an iterable of identifiers.")
    normalized = tuple(_required(value, name=name) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique.")
    return tuple(sorted(normalized)) if sort_values else normalized


def _hashes_by_id(
    values: Mapping[str, str],
    *,
    name: str,
) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} hashes must be a mapping.")
    normalized_items = tuple(
        (
            _required(identifier, name=f"{name}_id"),
            _required(content_hash, name=f"{name}_hash"),
        )
        for identifier, content_hash in dict(values).items()
    )
    normalized = dict(normalized_items)
    if len(normalized) != len(normalized_items):
        raise ValueError(f"{name}_id must be unique after normalization.")
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class EmbodimentApplicationSurfaceSpec:
    """Portable identity for one complete embodiment application surface."""

    surface_id: str
    surface_kind: str
    application_contract_id: str
    application_contract_hash: str
    target_runtime_profile_id: str
    target_runtime_profile_hash: str
    required_asset_bundle_ids: tuple[str, ...] = ()
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=EMBODIMENT_APPLICATION_SURFACE_SPEC_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "surface_id",
            "surface_kind",
            "application_contract_id",
            "application_contract_hash",
            "target_runtime_profile_id",
            "target_runtime_profile_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "required_asset_bundle_ids",
            _identifiers(
                self.required_asset_bundle_ids,
                name="required_asset_bundle_id",
                sort_values=True,
            ),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=EMBODIMENT_APPLICATION_SURFACE_SPEC_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "surface_id": self.surface_id,
            "surface_kind": self.surface_kind,
            "application_contract_id": self.application_contract_id,
            "application_contract_hash": self.application_contract_hash,
            "target_runtime_profile_id": self.target_runtime_profile_id,
            "target_runtime_profile_hash": self.target_runtime_profile_hash,
            "required_asset_bundle_ids": self.required_asset_bundle_ids,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "EmbodimentApplicationSurfaceSpec":
        if not isinstance(payload, Mapping):
            raise TypeError("surface payload must be a mapping.")
        if payload.get("schema") != EMBODIMENT_APPLICATION_SURFACE_SPEC_SCHEMA_V1:
            raise ValueError("Unsupported embodiment application surface schema.")
        _require_exact_keys(
            payload,
            expected={
                "schema",
                "surface_id",
                "surface_kind",
                "application_contract_id",
                "application_contract_hash",
                "target_runtime_profile_id",
                "target_runtime_profile_hash",
                "required_asset_bundle_ids",
                "semantic_content_hash",
            },
            name="surface payload",
        )
        expected_hash = _required(
            payload.get("semantic_content_hash"),
            name="semantic_content_hash",
        )
        required_assets = payload.get("required_asset_bundle_ids")
        if not isinstance(required_assets, (tuple, list)):
            raise TypeError("required_asset_bundle_ids must be a sequence.")
        spec = cls(
            surface_id=payload.get("surface_id", ""),
            surface_kind=payload.get("surface_kind", ""),
            application_contract_id=payload.get("application_contract_id", ""),
            application_contract_hash=payload.get(
                "application_contract_hash",
                "",
            ),
            target_runtime_profile_id=payload.get(
                "target_runtime_profile_id",
                "",
            ),
            target_runtime_profile_hash=payload.get(
                "target_runtime_profile_hash",
                "",
            ),
            required_asset_bundle_ids=tuple(required_assets),
        )
        if spec.semantic_content_hash != expected_hash:
            raise ValueError(
                "Embodiment application surface semantic hash mismatch."
            )
        return spec


@dataclass(frozen=True)
class EmbodimentApplicationSuiteSpec:
    """Portable allowlist for one complete embodiment integration suite."""

    suite_id: str
    suite_version: str
    embodiment_definition_id: str
    embodiment_definition_hash: str
    integration_package_name: str
    integration_package_version: str
    suite_contract_hash: str
    surfaces: tuple[EmbodimentApplicationSurfaceSpec, ...]
    asset_bundle_hash_by_id: Mapping[str, str] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=EMBODIMENT_APPLICATION_SUITE_SPEC_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "suite_id",
            "suite_version",
            "embodiment_definition_id",
            "embodiment_definition_hash",
            "integration_package_name",
            "integration_package_version",
            "suite_contract_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        surfaces = tuple(self.surfaces)
        if not surfaces:
            raise ValueError("An embodiment application suite requires a surface.")
        if any(
            not isinstance(surface, EmbodimentApplicationSurfaceSpec)
            for surface in surfaces
        ):
            raise TypeError(
                "surfaces must contain EmbodimentApplicationSurfaceSpec values."
            )
        surface_by_id = {surface.surface_id: surface for surface in surfaces}
        if len(surface_by_id) != len(surfaces):
            raise ValueError("Embodiment application surface IDs must be unique.")
        canonical_surfaces = tuple(
            surface_by_id[surface_id] for surface_id in sorted(surface_by_id)
        )
        asset_hashes = _hashes_by_id(
            self.asset_bundle_hash_by_id,
            name="asset_bundle",
        )
        unknown_assets = {
            asset_id
            for surface in canonical_surfaces
            for asset_id in surface.required_asset_bundle_ids
            if asset_id not in asset_hashes
        }
        if unknown_assets:
            raise ValueError(
                "Application surfaces require undeclared asset bundles: "
                f"{sorted(unknown_assets)!r}."
            )
        object.__setattr__(self, "surfaces", canonical_surfaces)
        object.__setattr__(self, "asset_bundle_hash_by_id", asset_hashes)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=EMBODIMENT_APPLICATION_SUITE_SPEC_HASH_DOMAIN_V1,
            ),
        )

    @property
    def surface_by_id(self) -> Mapping[str, EmbodimentApplicationSurfaceSpec]:
        return MappingProxyType(
            {surface.surface_id: surface for surface in self.surfaces}
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "embodiment_definition_id": self.embodiment_definition_id,
            "embodiment_definition_hash": self.embodiment_definition_hash,
            "integration_package_name": self.integration_package_name,
            "integration_package_version": self.integration_package_version,
            "suite_contract_hash": self.suite_contract_hash,
            "surfaces": tuple(surface.to_payload() for surface in self.surfaces),
            "asset_bundle_hash_by_id": dict(self.asset_bundle_hash_by_id),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "EmbodimentApplicationSuiteSpec":
        if not isinstance(payload, Mapping):
            raise TypeError("suite payload must be a mapping.")
        if payload.get("schema") != EMBODIMENT_APPLICATION_SUITE_SPEC_SCHEMA_V1:
            raise ValueError("Unsupported embodiment application suite schema.")
        _require_exact_keys(
            payload,
            expected={
                "schema",
                "suite_id",
                "suite_version",
                "embodiment_definition_id",
                "embodiment_definition_hash",
                "integration_package_name",
                "integration_package_version",
                "suite_contract_hash",
                "surfaces",
                "asset_bundle_hash_by_id",
                "semantic_content_hash",
            },
            name="suite payload",
        )
        expected_hash = _required(
            payload.get("semantic_content_hash"),
            name="semantic_content_hash",
        )
        surface_payloads = payload.get("surfaces", ())
        if not isinstance(surface_payloads, (tuple, list)):
            raise TypeError("suite surfaces payload must be a sequence.")
        assets = payload.get("asset_bundle_hash_by_id", {})
        if not isinstance(assets, Mapping):
            raise TypeError("asset_bundle_hash_by_id must be a mapping.")
        spec = cls(
            suite_id=payload.get("suite_id", ""),
            suite_version=payload.get("suite_version", ""),
            embodiment_definition_id=payload.get(
                "embodiment_definition_id",
                "",
            ),
            embodiment_definition_hash=payload.get(
                "embodiment_definition_hash",
                "",
            ),
            integration_package_name=payload.get(
                "integration_package_name",
                "",
            ),
            integration_package_version=payload.get(
                "integration_package_version",
                "",
            ),
            suite_contract_hash=payload.get("suite_contract_hash", ""),
            surfaces=tuple(
                EmbodimentApplicationSurfaceSpec.from_payload(surface_payload)
                for surface_payload in surface_payloads
            ),
            asset_bundle_hash_by_id=dict(assets),
        )
        if spec.semantic_content_hash != expected_hash:
            raise ValueError("Embodiment application suite semantic hash mismatch.")
        return spec


@dataclass(frozen=True)
class LoadedEmbodimentApplicationSurface:
    """Identity-checked envelope around one opaque application implementation."""

    suite_id: str
    suite_spec_hash: str
    surface_id: str
    surface_spec_hash: str
    application_contract_hash: str
    implementation: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "suite_id",
            "suite_spec_hash",
            "surface_id",
            "surface_spec_hash",
            "application_contract_hash",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        if self.implementation is None:
            raise ValueError("Loaded application implementation cannot be None.")


EmbodimentApplicationSurfaceLoader = Callable[
    [],
    LoadedEmbodimentApplicationSurface,
]


@dataclass(frozen=True)
class EmbodimentApplicationSuiteBinding:
    """Pair one portable suite with explicit, non-discovered surface loaders."""

    suite_spec: EmbodimentApplicationSuiteSpec
    surface_loader_by_id: Mapping[str, EmbodimentApplicationSurfaceLoader] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.suite_spec, EmbodimentApplicationSuiteSpec):
            raise TypeError("suite_spec must be EmbodimentApplicationSuiteSpec.")
        if not isinstance(self.surface_loader_by_id, Mapping):
            raise TypeError("surface_loader_by_id must be a mapping.")
        loader_items = tuple(
            (
                _required(surface_id, name="surface_loader_id"),
                loader,
            )
            for surface_id, loader in dict(self.surface_loader_by_id).items()
        )
        loaders = dict(loader_items)
        if len(loaders) != len(loader_items):
            raise ValueError(
                "Surface loader IDs must be unique after normalization."
            )
        expected_ids = set(self.suite_spec.surface_by_id)
        if set(loaders) != expected_ids:
            raise ValueError(
                "Surface loaders must exactly match the portable suite: "
                f"expected={sorted(expected_ids)!r} actual={sorted(loaders)!r}."
            )
        if any(not callable(loader) for loader in loaders.values()):
            raise TypeError("Every embodiment application surface loader is callable.")
        object.__setattr__(
            self,
            "surface_loader_by_id",
            MappingProxyType(dict(sorted(loaders.items()))),
        )


class EmbodimentApplicationSuiteCatalog:
    """Exact allowlist with lazy loading of one selected vertical surface."""

    def __init__(
        self,
        bindings: Iterable[EmbodimentApplicationSuiteBinding] = (),
    ) -> None:
        self._binding_by_id: dict[str, EmbodimentApplicationSuiteBinding] = {}
        for binding in bindings:
            self.register(binding)

    def register(self, binding: EmbodimentApplicationSuiteBinding) -> None:
        if not isinstance(binding, EmbodimentApplicationSuiteBinding):
            raise TypeError(
                "binding must be EmbodimentApplicationSuiteBinding."
            )
        suite_id = binding.suite_spec.suite_id
        if suite_id in self._binding_by_id:
            raise KeyError(
                f"Embodiment application suite {suite_id!r} is already registered."
            )
        self._binding_by_id[suite_id] = binding

    def suite_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._binding_by_id))

    def suite_specs(self) -> tuple[EmbodimentApplicationSuiteSpec, ...]:
        return tuple(
            self._binding_by_id[suite_id].suite_spec
            for suite_id in self.suite_ids()
        )

    def select(self, suite_id: str) -> EmbodimentApplicationSuiteBinding:
        normalized = _required(suite_id, name="suite_id")
        try:
            return self._binding_by_id[normalized]
        except KeyError as exc:
            raise KeyError(
                f"Unknown embodiment application suite {normalized!r}. "
                f"Registered: {list(self.suite_ids())!r}."
            ) from exc

    def load_surface(
        self,
        suite_id: str,
        surface_id: str,
    ) -> LoadedEmbodimentApplicationSurface:
        binding = self.select(suite_id)
        normalized_surface_id = _required(surface_id, name="surface_id")
        surface = binding.suite_spec.surface_by_id.get(normalized_surface_id)
        if surface is None:
            raise KeyError(
                f"Suite {binding.suite_spec.suite_id!r} has no surface "
                f"{normalized_surface_id!r}. Registered: "
                f"{list(binding.suite_spec.surface_by_id)!r}."
            )
        loaded = binding.surface_loader_by_id[normalized_surface_id]()
        if not isinstance(loaded, LoadedEmbodimentApplicationSurface):
            raise TypeError(
                "An embodiment application surface loader must return "
                "LoadedEmbodimentApplicationSurface."
            )
        expected_values = (
            ("suite_id", loaded.suite_id, binding.suite_spec.suite_id),
            (
                "suite_spec_hash",
                loaded.suite_spec_hash,
                binding.suite_spec.semantic_content_hash,
            ),
            ("surface_id", loaded.surface_id, surface.surface_id),
            (
                "surface_spec_hash",
                loaded.surface_spec_hash,
                surface.semantic_content_hash,
            ),
            (
                "application_contract_hash",
                loaded.application_contract_hash,
                surface.application_contract_hash,
            ),
        )
        mismatches = [
            name for name, actual, expected in expected_values if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "Loaded embodiment application surface differs from its "
                f"allowlisted spec: {','.join(mismatches)}."
            )
        return loaded


__all__ = [
    "EMBODIMENT_APPLICATION_SUITE_SPEC_HASH_DOMAIN_V1",
    "EMBODIMENT_APPLICATION_SUITE_SPEC_SCHEMA_V1",
    "EMBODIMENT_APPLICATION_SURFACE_SPEC_HASH_DOMAIN_V1",
    "EMBODIMENT_APPLICATION_SURFACE_SPEC_SCHEMA_V1",
    "EmbodimentApplicationSurfaceLoader",
    "EmbodimentApplicationSurfaceSpec",
    "EmbodimentApplicationSuiteBinding",
    "EmbodimentApplicationSuiteCatalog",
    "EmbodimentApplicationSuiteSpec",
    "LoadedEmbodimentApplicationSurface",
]
