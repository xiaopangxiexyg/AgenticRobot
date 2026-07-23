"""Stable, backend-neutral contracts for model observations and actions.

These types are intentionally independent of Cube-Lift and of a particular
policy library.  A simulator adapter and a real deployment adapter can both
project their native snapshots into :class:`ObservationContract` fields,
after which the same :class:`ModelInputBinding` and
:class:`PolicyActionBinding` are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash


LEARNING_CONTRACT_VERSION = "dexteleop.learning_contract.v1"
_OBSERVATION_CONTRACT_HASH_DOMAIN = "dexteleop.learning.observation_contract.v1"
_MODEL_INPUT_BINDING_HASH_DOMAIN = "dexteleop.learning.model_input_binding.v1"
_ACTION_BINDING_HASH_DOMAIN = "dexteleop.learning.action_binding.v1"


class ObservationRole(str, Enum):
    """Who may consume a field.

    ``ACTOR`` is the only role admitted by a normal deployable policy.  The
    other roles are useful for an expert critic or reward calculation, but are
    rejected by a normal :class:`ModelInputBinding`.
    """

    ACTOR = "actor"
    PRIVILEGED = "privileged"
    REWARD_ONLY = "reward_only"


def _normalise_shape(shape: tuple[int | None, ...] | None) -> tuple[int | None, ...] | None:
    if shape is None:
        return None
    result = tuple(None if size is None else int(size) for size in shape)
    if any(size is not None and size < 0 for size in result):
        raise ValueError("shape dimensions must be non-negative or None")
    return result


@dataclass(frozen=True)
class ObservationFieldSpec:
    """Description of one causal observation field.

    ``shape`` describes one sample; a leading batch dimension is accepted by
    the encoder.  Images are explicitly CHW so a real camera adapter cannot
    silently swap HWC/CHW at the deployment boundary.
    """

    name: str
    shape: tuple[int | None, ...] | None = None
    dtype: str = "float32"
    role: ObservationRole = ObservationRole.ACTOR
    encoding: str = "numeric"
    unit: str = "unitless"
    source: str = "backend"
    normalization: str = "identity"
    required: bool = True
    missing_policy: str = "error"
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("ObservationFieldSpec.name must be non-empty")
        object.__setattr__(self, "name", name)
        try:
            dtype = str(self.dtype)
            # Validate early while preserving the serializable spelling.
            import numpy as np

            np.dtype(dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported observation dtype: {self.dtype!r}") from exc
        object.__setattr__(self, "dtype", dtype)
        try:
            role = ObservationRole(self.role)
        except ValueError as exc:
            raise ValueError(f"Unsupported observation role: {self.role!r}") from exc
        object.__setattr__(self, "role", role)
        encoding = str(self.encoding).strip().lower()
        if encoding not in {"numeric", "image_chw"}:
            raise ValueError("encoding must be 'numeric' or 'image_chw'")
        if encoding == "image_chw" and self.shape is not None and len(self.shape) != 3:
            raise ValueError("image_chw fields must have a three-dimensional CHW shape")
        object.__setattr__(self, "encoding", encoding)
        normalization = str(self.normalization).strip().lower()
        if normalization not in {
            "identity",
            "unit_interval",
            "symmetric_unit",
            "scale_offset",
        }:
            raise ValueError(
                "normalization must be identity, unit_interval, symmetric_unit, or scale_offset"
            )
        object.__setattr__(self, "normalization", normalization)
        if normalization == "scale_offset" and float(self.scale) == 0.0:
            raise ValueError("scale_offset normalization requires a non-zero scale")
        if not str(self.unit).strip() or not str(self.source).strip():
            raise ValueError("unit and source must be non-empty")
        if self.missing_policy not in {"error", "omit"}:
            raise ValueError("missing_policy must be 'error' or 'omit'")
        object.__setattr__(self, "shape", _normalise_shape(self.shape))
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "offset", float(self.offset))

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
            "role": self.role.value,
            "encoding": self.encoding,
            "unit": self.unit,
            "source": self.source,
            "normalization": self.normalization,
            "required": self.required,
            "missing_policy": self.missing_policy,
            "scale": self.scale,
            "offset": self.offset,
        }


@dataclass(frozen=True)
class ObservationContract:
    """Named field contract shared by a simulator and a real adapter."""

    contract_id: str
    fields: tuple[ObservationFieldSpec, ...]
    version: str = LEARNING_CONTRACT_VERSION

    def __post_init__(self) -> None:
        contract_id = str(self.contract_id).strip()
        if not contract_id:
            raise ValueError("ObservationContract.contract_id must be non-empty")
        if self.version != LEARNING_CONTRACT_VERSION:
            raise ValueError(f"Unsupported learning contract version: {self.version!r}")
        fields = tuple(self.fields)
        if not fields:
            raise ValueError("ObservationContract requires at least one field")
        names = [field.name for field in fields]
        if len(names) != len(set(names)):
            raise ValueError("ObservationContract fields must be unique")
        object.__setattr__(self, "contract_id", contract_id)
        object.__setattr__(self, "fields", fields)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def actor_fields(self) -> tuple[ObservationFieldSpec, ...]:
        return tuple(field for field in self.fields if field.role is ObservationRole.ACTOR)

    @property
    def privileged_fields(self) -> tuple[ObservationFieldSpec, ...]:
        return tuple(
            field for field in self.fields if field.role is ObservationRole.PRIVILEGED
        )

    @property
    def reward_only_fields(self) -> tuple[ObservationFieldSpec, ...]:
        return tuple(
            field for field in self.fields if field.role is ObservationRole.REWARD_ONLY
        )

    def field(self, name: str) -> ObservationFieldSpec:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(f"Unknown observation field: {name!r}")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "version": self.version,
                "contract_id": self.contract_id,
                "fields": tuple(field.semantic_payload() for field in self.fields),
            },
            domain=_OBSERVATION_CONTRACT_HASH_DOMAIN,
        )


@dataclass(frozen=True)
class ModelInputBinding:
    """Ordered model input projection from an observation contract."""

    binding_id: str
    observation_contract_hash: str
    input_fields: tuple[str, ...]
    encoder_id: str = "ordered_fields_flat_v1"
    allow_privileged: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        binding_id = str(self.binding_id).strip()
        contract_hash = str(self.observation_contract_hash).strip()
        if not binding_id or not contract_hash:
            raise ValueError("ModelInputBinding identity and contract hash are required")
        fields = tuple(str(name).strip() for name in self.input_fields)
        if not fields or any(not name for name in fields):
            raise ValueError("ModelInputBinding.input_fields must be non-empty")
        if len(fields) != len(set(fields)):
            raise ValueError("ModelInputBinding.input_fields must be unique")
        if not str(self.encoder_id).strip():
            raise ValueError("ModelInputBinding.encoder_id must be non-empty")
        object.__setattr__(self, "binding_id", binding_id)
        object.__setattr__(self, "observation_contract_hash", contract_hash)
        object.__setattr__(self, "input_fields", fields)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def validate(self, contract: ObservationContract) -> None:
        if contract.semantic_content_hash != self.observation_contract_hash:
            raise ValueError(
                "ModelInputBinding observation contract hash mismatch: "
                f"expected {self.observation_contract_hash}, got {contract.semantic_content_hash}"
            )
        for name in self.input_fields:
            field_spec = contract.field(name)
            if not self.allow_privileged and field_spec.role is not ObservationRole.ACTOR:
                raise ValueError(
                    f"Normal model input cannot consume {field_spec.role.value} field {name!r}"
                )

    def expected_flat_dim(self, contract: ObservationContract) -> int:
        self.validate(contract)
        dimensions = 0
        for name in self.input_fields:
            shape = contract.field(name).shape
            if shape is None or any(size is None for size in shape):
                raise ValueError(f"Field {name!r} needs a fully known shape for flat encoding")
            size = 1
            for dimension in shape:
                size *= int(dimension)
            dimensions += size
        return dimensions

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "version": LEARNING_CONTRACT_VERSION,
                "binding_id": self.binding_id,
                "observation_contract_hash": self.observation_contract_hash,
                "input_fields": self.input_fields,
                "encoder_id": self.encoder_id,
                "allow_privileged": self.allow_privileged,
                "metadata": self.metadata,
            },
            domain=_MODEL_INPUT_BINDING_HASH_DOMAIN,
        )


@dataclass(frozen=True)
class PolicyActionBinding:
    """Model output semantics shared by training and deployment."""

    binding_id: str
    action_space: str
    output_dim: int
    frame: str
    mode: str
    normalized: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = {
            "binding_id": str(self.binding_id).strip(),
            "action_space": str(self.action_space).strip(),
            "frame": str(self.frame).strip(),
            "mode": str(self.mode).strip(),
        }
        if any(not value for value in values.values()):
            raise ValueError("PolicyActionBinding string fields must be non-empty")
        if int(self.output_dim) <= 0:
            raise ValueError("PolicyActionBinding.output_dim must be positive")
        if self.mode not in {"absolute", "relative"}:
            raise ValueError("PolicyActionBinding.mode must be 'absolute' or 'relative'")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "output_dim", int(self.output_dim))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "version": LEARNING_CONTRACT_VERSION,
                "binding_id": self.binding_id,
                "action_space": self.action_space,
                "output_dim": self.output_dim,
                "frame": self.frame,
                "mode": self.mode,
                "normalized": self.normalized,
                "metadata": self.metadata,
            },
            domain=_ACTION_BINDING_HASH_DOMAIN,
        )


__all__ = [
    "LEARNING_CONTRACT_VERSION",
    "ModelInputBinding",
    "ObservationContract",
    "ObservationFieldSpec",
    "ObservationRole",
    "PolicyActionBinding",
]
