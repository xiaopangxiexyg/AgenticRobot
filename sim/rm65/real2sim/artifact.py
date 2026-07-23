from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from sim.contracts import SimulationContract, TimingSpec


REAL2SIM_ARTIFACT_VERSION = "dexteleop.sim.real2sim_artifact.v1"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FIDELITY_STATUSES = frozenset(
    {"identified", "provisional", "debug_only", "unmodeled", "unavailable"}
)
_VALIDATION_SPLITS = frozenset({"fit", "validation", "held_out"})
_VALIDATION_STATUSES = frozenset({"passed", "failed", "not_run"})


def _required_text(value: object, *, field_name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field_name} must be non-empty.")
    return result


def _sha256(value: object, *, field_name: str) -> str:
    result = str(value).strip()
    if not _SHA256_PATTERN.fullmatch(result):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest.")
    return result


def _unique_text(values: Sequence[object], *, field_name: str) -> tuple[str, ...]:
    result = tuple(_required_text(value, field_name=field_name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return result


def _freeze_json(value: Any, *, field_name: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _required_text(raw_key, field_name=f"{field_name} key")
            if key in frozen:
                raise ValueError(f"{field_name} contains duplicate key {key!r}.")
            frozen[key] = _freeze_json(raw_value, field_name=f"{field_name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, field_name=f"{field_name}[]") for item in value
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite floats.")
        return value
    raise TypeError(
        f"{field_name} must contain JSON-compatible values; got {type(value).__name__}."
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    """Content-addressed input used to justify or reproduce a sim claim."""

    artifact_id: str
    artifact_kind: str
    sha256: str
    locator: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            _required_text(self.artifact_id, field_name="artifact_id"),
        )
        object.__setattr__(
            self,
            "artifact_kind",
            _required_text(self.artifact_kind, field_name="artifact_kind"),
        )
        object.__setattr__(self, "sha256", _sha256(self.sha256, field_name="sha256"))
        object.__setattr__(self, "locator", str(self.locator).strip())

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        artifact_id: str,
        artifact_kind: str,
        locator: str | None = None,
    ) -> "ArtifactRef":
        source = Path(path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        return cls(
            artifact_id=artifact_id,
            artifact_kind=artifact_kind,
            sha256=digest,
            locator=str(source) if locator is None else locator,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "sha256": self.sha256,
            "locator": self.locator,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            artifact_id=str(payload["artifact_id"]),
            artifact_kind=str(payload["artifact_kind"]),
            sha256=str(payload["sha256"]),
            locator=str(payload.get("locator", "")),
        )


@dataclass(frozen=True)
class ResolvedTiming:
    """All clocks after defaults have been resolved from a SimulationContract."""

    sample_hz: float
    policy_hz: float
    reference_hz: float
    ik_hz: float
    command_hz: float
    physics_hz: float
    camera_hz: float
    interpolation: str
    command_hold: str
    parameter_source: str

    def __post_init__(self) -> None:
        validated = TimingSpec(
            sample_hz=self.sample_hz,
            policy_hz=self.policy_hz,
            reference_hz=self.reference_hz,
            ik_hz=self.ik_hz,
            command_hz=self.command_hz,
            physics_hz=self.physics_hz,
            camera_hz=self.camera_hz,
            interpolation=self.interpolation,
            command_hold=self.command_hold,
            parameter_source=self.parameter_source,
        )
        for name in (
            "sample_hz",
            "policy_hz",
            "reference_hz",
            "ik_hz",
            "command_hz",
            "physics_hz",
            "camera_hz",
        ):
            object.__setattr__(self, name, float(getattr(validated, name)))
        object.__setattr__(self, "interpolation", validated.interpolation)
        object.__setattr__(self, "command_hold", validated.command_hold)
        object.__setattr__(
            self,
            "parameter_source",
            _required_text(validated.parameter_source, field_name="parameter_source"),
        )

    @classmethod
    def from_timing_spec(cls, timing: TimingSpec) -> "ResolvedTiming":
        return cls(
            sample_hz=timing.sample_hz,
            policy_hz=timing.policy_hz,
            reference_hz=timing.reference_hz,
            ik_hz=timing.ik_hz,
            command_hz=timing.command_hz,
            physics_hz=timing.physics_hz,
            camera_hz=timing.camera_hz,
            interpolation=timing.interpolation,
            command_hold=timing.command_hold,
            parameter_source=timing.parameter_source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_hz": self.sample_hz,
            "policy_hz": self.policy_hz,
            "reference_hz": self.reference_hz,
            "ik_hz": self.ik_hz,
            "command_hz": self.command_hz,
            "physics_hz": self.physics_hz,
            "camera_hz": self.camera_hz,
            "interpolation": self.interpolation,
            "command_hold": self.command_hold,
            "parameter_source": self.parameter_source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResolvedTiming":
        return cls(
            sample_hz=float(payload["sample_hz"]),
            policy_hz=float(payload["policy_hz"]),
            reference_hz=float(payload["reference_hz"]),
            ik_hz=float(payload["ik_hz"]),
            command_hz=float(payload["command_hz"]),
            physics_hz=float(payload["physics_hz"]),
            camera_hz=float(payload["camera_hz"]),
            interpolation=str(payload["interpolation"]),
            command_hold=str(payload["command_hold"]),
            parameter_source=str(payload["parameter_source"]),
        )


@dataclass(frozen=True)
class FidelityClaim:
    """An explicit fidelity assertion; non-identified states must disclose gaps."""

    component: str
    status: str
    evidence_artifact_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component",
            _required_text(self.component, field_name="fidelity component"),
        )
        status = str(self.status).strip()
        if status not in _FIDELITY_STATUSES:
            raise ValueError(
                f"Unsupported fidelity status {status!r}; "
                f"expected one of {sorted(_FIDELITY_STATUSES)}."
            )
        object.__setattr__(self, "status", status)
        evidence = _unique_text(
            self.evidence_artifact_ids,
            field_name="evidence_artifact_ids",
        )
        limitations = _unique_text(self.limitations, field_name="limitations")
        if status == "identified" and not evidence:
            raise ValueError("identified fidelity claims require evidence_artifact_ids.")
        if status != "identified" and not limitations:
            raise ValueError(
                f"{status} fidelity claims must state at least one limitation."
            )
        object.__setattr__(self, "evidence_artifact_ids", evidence)
        object.__setattr__(self, "limitations", limitations)
        object.__setattr__(self, "notes", str(self.notes).strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "status": self.status,
            "evidence_artifact_ids": list(self.evidence_artifact_ids),
            "limitations": list(self.limitations),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FidelityClaim":
        return cls(
            component=str(payload["component"]),
            status=str(payload["status"]),
            evidence_artifact_ids=tuple(payload.get("evidence_artifact_ids", ())),
            limitations=tuple(payload.get("limitations", ())),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class ValidationEvidence:
    """Metrics from fit, validation, or held-out traces."""

    validation_id: str
    split: str
    status: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    evidence_artifact_ids: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "validation_id",
            _required_text(self.validation_id, field_name="validation_id"),
        )
        split = str(self.split).strip()
        if split not in _VALIDATION_SPLITS:
            raise ValueError(
                f"Unsupported validation split {split!r}; "
                f"expected one of {sorted(_VALIDATION_SPLITS)}."
            )
        status = str(self.status).strip()
        if status not in _VALIDATION_STATUSES:
            raise ValueError(
                f"Unsupported validation status {status!r}; "
                f"expected one of {sorted(_VALIDATION_STATUSES)}."
            )
        metrics: dict[str, float] = {}
        for raw_name, raw_value in self.metrics.items():
            name = _required_text(raw_name, field_name="metric name")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"Metric {name!r} must be finite.")
            metrics[name] = value
        evidence = _unique_text(
            self.evidence_artifact_ids,
            field_name="validation evidence_artifact_ids",
        )
        if status != "not_run" and (not metrics or not evidence):
            raise ValueError(
                "passed/failed validation requires metrics and evidence_artifact_ids."
            )
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "evidence_artifact_ids", evidence)
        object.__setattr__(self, "notes", str(self.notes).strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "split": self.split,
            "status": self.status,
            "metrics": dict(self.metrics),
            "evidence_artifact_ids": list(self.evidence_artifact_ids),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationEvidence":
        return cls(
            validation_id=str(payload["validation_id"]),
            split=str(payload["split"]),
            status=str(payload["status"]),
            metrics=dict(payload.get("metrics", {})),
            evidence_artifact_ids=tuple(payload.get("evidence_artifact_ids", ())),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class Real2SimArtifactManifest:
    """Auditable boundary between a sim configuration and its real evidence."""

    name: str
    simulation_contract: ArtifactRef
    resolved_timing: ResolvedTiming
    dependency_artifacts: tuple[ArtifactRef, ...]
    fidelity_claims: tuple[FidelityClaim, ...]
    validation_evidence: tuple[ValidationEvidence, ...]
    code_revision: str
    environment_fingerprint: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = REAL2SIM_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REAL2SIM_ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported real2sim artifact version {self.schema_version!r}; "
                f"expected {REAL2SIM_ARTIFACT_VERSION!r}."
            )
        object.__setattr__(self, "name", _required_text(self.name, field_name="name"))
        if self.simulation_contract.artifact_kind != "simulation_contract":
            raise ValueError(
                "simulation_contract artifact_kind must be 'simulation_contract'."
            )
        dependencies = tuple(self.dependency_artifacts)
        claims = tuple(self.fidelity_claims)
        validations = tuple(self.validation_evidence)
        if not claims:
            raise ValueError("A Real2Sim artifact must contain a fidelity claim.")
        artifacts = (self.simulation_contract, *dependencies)
        artifact_ids = [artifact.artifact_id for artifact in artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("Artifact IDs must be unique across the manifest.")
        claim_components = [claim.component for claim in claims]
        if len(set(claim_components)) != len(claim_components):
            raise ValueError("Fidelity claim components must be unique.")
        validation_ids = [item.validation_id for item in validations]
        if len(set(validation_ids)) != len(validation_ids):
            raise ValueError("Validation IDs must be unique.")
        known_ids = set(artifact_ids)
        referenced_ids = {
            artifact_id
            for claim in claims
            for artifact_id in claim.evidence_artifact_ids
        }
        referenced_ids.update(
            artifact_id
            for validation in validations
            for artifact_id in validation.evidence_artifact_ids
        )
        unknown_ids = sorted(referenced_ids - known_ids)
        if unknown_ids:
            raise ValueError(
                "Evidence references unknown artifact IDs: " + ", ".join(unknown_ids)
            )
        artifact_kind_by_id = {
            artifact.artifact_id: artifact.artifact_kind for artifact in artifacts
        }
        fit_real_traces = {
            artifact_id
            for validation in validations
            if validation.split == "fit"
            for artifact_id in validation.evidence_artifact_ids
            if artifact_kind_by_id[artifact_id] == "real_trace"
        }
        held_out_real_traces = {
            artifact_id
            for validation in validations
            if validation.split == "held_out"
            for artifact_id in validation.evidence_artifact_ids
            if artifact_kind_by_id[artifact_id] == "real_trace"
        }
        leaked_traces = sorted(fit_real_traces & held_out_real_traces)
        if leaked_traces:
            raise ValueError(
                "Fit and held-out validation must use different real_trace artifacts: "
                + ", ".join(leaked_traces)
            )
        environment: dict[str, str] = {}
        for raw_key, raw_value in self.environment_fingerprint.items():
            key = _required_text(raw_key, field_name="environment_fingerprint key")
            if key in environment:
                raise ValueError(
                    f"environment_fingerprint contains duplicate key {key!r}."
                )
            environment[key] = _required_text(
                raw_value,
                field_name=f"environment_fingerprint[{key!r}]",
            )
        if not environment:
            raise ValueError("environment_fingerprint must be non-empty.")
        object.__setattr__(self, "dependency_artifacts", dependencies)
        object.__setattr__(self, "fidelity_claims", claims)
        object.__setattr__(self, "validation_evidence", validations)
        object.__setattr__(
            self,
            "code_revision",
            _required_text(self.code_revision, field_name="code_revision"),
        )
        object.__setattr__(
            self,
            "environment_fingerprint",
            MappingProxyType(environment),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_json(dict(self.metadata), field_name="metadata"),
        )

    @classmethod
    def resolved_timing_from_contract_file(
        cls,
        path: str | Path,
    ) -> ResolvedTiming:
        contract = SimulationContract.load(path)
        return ResolvedTiming.from_timing_spec(contract.timing)

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "simulation_contract": self.simulation_contract.to_dict(),
            "resolved_timing": self.resolved_timing.to_dict(),
            "dependency_artifacts": [
                artifact.to_dict() for artifact in self.dependency_artifacts
            ],
            "fidelity_claims": [claim.to_dict() for claim in self.fidelity_claims],
            "validation_evidence": [
                item.to_dict() for item in self.validation_evidence
            ],
            "code_revision": self.code_revision,
            "environment_fingerprint": dict(self.environment_fingerprint),
            "metadata": _thaw_json(self.metadata),
        }

    @property
    def artifact_sha256(self) -> str:
        return _canonical_sha256(self._payload_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["artifact_sha256"] = self.artifact_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Real2SimArtifactManifest":
        manifest = cls(
            schema_version=str(
                payload.get("schema_version", REAL2SIM_ARTIFACT_VERSION)
            ),
            name=str(payload["name"]),
            simulation_contract=ArtifactRef.from_dict(payload["simulation_contract"]),
            resolved_timing=ResolvedTiming.from_dict(payload["resolved_timing"]),
            dependency_artifacts=tuple(
                ArtifactRef.from_dict(item)
                for item in payload.get("dependency_artifacts", ())
            ),
            fidelity_claims=tuple(
                FidelityClaim.from_dict(item)
                for item in payload.get("fidelity_claims", ())
            ),
            validation_evidence=tuple(
                ValidationEvidence.from_dict(item)
                for item in payload.get("validation_evidence", ())
            ),
            code_revision=str(payload["code_revision"]),
            environment_fingerprint=dict(payload["environment_fingerprint"]),
            metadata=dict(payload.get("metadata", {})),
        )
        expected_sha256 = payload.get("artifact_sha256")
        if expected_sha256 is not None:
            expected = _sha256(
                expected_sha256,
                field_name="artifact_sha256",
            )
            if expected != manifest.artifact_sha256:
                raise ValueError(
                    "Real2Sim artifact SHA-256 mismatch: "
                    f"expected={expected}, actual={manifest.artifact_sha256}."
                )
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "Real2SimArtifactManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Real2Sim artifact {path} must contain a JSON object.")
        return cls.from_dict(payload)


__all__ = [
    "REAL2SIM_ARTIFACT_VERSION",
    "ArtifactRef",
    "FidelityClaim",
    "Real2SimArtifactManifest",
    "ResolvedTiming",
    "ValidationEvidence",
]
