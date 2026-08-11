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
_ACTION_TRANSFORM_HASH_DOMAIN = "dexteleop.learning.action_transform.v1"


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


@dataclass(frozen=True)
class PolicyActionTransform:
    """Deterministic model-output to task-action mapping.

    PPO owns the normalized values described by ``policy_components``.  The
    task owns this transform and always emits the complete normalized action
    described by ``task_components``.  Safety projection and plant response
    are deliberately downstream and must never be inverted through this
    object to manufacture a replacement PPO sample.
    """

    transform_id: str
    policy_action_binding_hash: str
    task_action_binding_hash: str
    policy_components: tuple[str, ...]
    task_components: tuple[str, ...]
    task_defaults: tuple[float, ...]
    policy_scales: tuple[float, ...] = ()
    policy_biases: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        transform_id = str(self.transform_id).strip()
        policy_hash = str(self.policy_action_binding_hash).strip()
        task_hash = str(self.task_action_binding_hash).strip()
        if not transform_id or not policy_hash or not task_hash:
            raise ValueError(
                "PolicyActionTransform identity and binding hashes are required"
            )
        policy_components = tuple(
            str(component).strip() for component in self.policy_components
        )
        task_components = tuple(
            str(component).strip() for component in self.task_components
        )
        if not policy_components or any(not value for value in policy_components):
            raise ValueError("policy_components must be non-empty")
        if not task_components or any(not value for value in task_components):
            raise ValueError("task_components must be non-empty")
        if len(policy_components) != len(set(policy_components)):
            raise ValueError("policy_components must be unique")
        if len(task_components) != len(set(task_components)):
            raise ValueError("task_components must be unique")
        unknown = tuple(
            component
            for component in policy_components
            if component not in task_components
        )
        if unknown:
            raise ValueError(
                f"Policy action components are absent from the task action: {unknown}"
            )
        defaults = tuple(float(value) for value in self.task_defaults)
        if len(defaults) != len(task_components):
            raise ValueError(
                "task_defaults must contain one value per task action component"
            )
        import math

        if any(not math.isfinite(value) or abs(value) > 1.0 for value in defaults):
            raise ValueError("task_defaults must be finite normalized values in [-1, 1]")
        scales = (
            (1.0,) * len(policy_components)
            if not self.policy_scales
            else tuple(float(value) for value in self.policy_scales)
        )
        biases = (
            (0.0,) * len(policy_components)
            if not self.policy_biases
            else tuple(float(value) for value in self.policy_biases)
        )
        if len(scales) != len(policy_components) or len(biases) != len(
            policy_components
        ):
            raise ValueError(
                "policy_scales and policy_biases must match policy_components"
            )
        if any(
            not math.isfinite(scale) or not 0.0 < scale <= 1.0
            for scale in scales
        ):
            raise ValueError("policy_scales must be finite values in (0, 1]")
        if any(
            not math.isfinite(bias) or abs(bias) + scale > 1.0
            for scale, bias in zip(scales, biases)
        ):
            raise ValueError(
                "policy_biases plus policy_scales must keep mapped values in [-1, 1]"
            )
        object.__setattr__(self, "transform_id", transform_id)
        object.__setattr__(self, "policy_action_binding_hash", policy_hash)
        object.__setattr__(self, "task_action_binding_hash", task_hash)
        object.__setattr__(self, "policy_components", policy_components)
        object.__setattr__(self, "task_components", task_components)
        object.__setattr__(self, "task_defaults", defaults)
        object.__setattr__(self, "policy_scales", scales)
        object.__setattr__(self, "policy_biases", biases)

    @property
    def policy_dim(self) -> int:
        return len(self.policy_components)

    @property
    def task_dim(self) -> int:
        return len(self.task_components)

    @property
    def task_indices(self) -> tuple[int, ...]:
        return tuple(
            self.task_components.index(component)
            for component in self.policy_components
        )

    def validate_bindings(
        self,
        policy_binding: PolicyActionBinding,
        task_binding: PolicyActionBinding,
    ) -> None:
        """Fail closed when either side of the declared transform drifts."""

        if policy_binding.semantic_content_hash != self.policy_action_binding_hash:
            raise ValueError("Policy action binding does not match action transform")
        if task_binding.semantic_content_hash != self.task_action_binding_hash:
            raise ValueError("Task action binding does not match action transform")
        if policy_binding.output_dim != self.policy_dim:
            raise ValueError("Policy action binding dimension does not match transform")
        if task_binding.output_dim != self.task_dim:
            raise ValueError("Task action binding dimension does not match transform")

    def transform(self, values: Any) -> "Any":
        """Map one or a batch of normalized latent actions to full task actions."""

        import numpy as np

        policy_values = np.asarray(values, dtype=np.float32)
        if policy_values.ndim == 0 or policy_values.shape[-1] != self.policy_dim:
            raise ValueError(
                "Policy action must end in "
                f"{self.policy_dim} values; got {policy_values.shape}"
            )
        if not np.all(np.isfinite(policy_values)):
            raise ValueError("Policy action must be finite")
        if np.any(policy_values < -1.0) or np.any(policy_values > 1.0):
            raise ValueError("Policy action values must lie in [-1, 1]")
        output = np.broadcast_to(
            np.asarray(self.task_defaults, dtype=np.float32),
            (*policy_values.shape[:-1], self.task_dim),
        ).copy()
        mapped = (
            policy_values
            * np.asarray(self.policy_scales, dtype=np.float32)
            + np.asarray(self.policy_biases, dtype=np.float32)
        )
        output[..., list(self.task_indices)] = mapped
        return output

    def semantic_payload(self) -> dict[str, Any]:
        payload = {
            "transform_id": self.transform_id,
            "policy_action_binding_hash": self.policy_action_binding_hash,
            "task_action_binding_hash": self.task_action_binding_hash,
            "policy_components": self.policy_components,
            "task_components": self.task_components,
            "task_defaults": self.task_defaults,
        }
        if self.policy_scales != (1.0,) * self.policy_dim or self.policy_biases != (
            0.0,
        ) * self.policy_dim:
            payload["policy_scales"] = self.policy_scales
            payload["policy_biases"] = self.policy_biases
        return payload

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "version": LEARNING_CONTRACT_VERSION,
                **self.semantic_payload(),
            },
            domain=_ACTION_TRANSFORM_HASH_DOMAIN,
        )


__all__ = [
    "LEARNING_CONTRACT_VERSION",
    "ModelInputBinding",
    "ObservationContract",
    "ObservationFieldSpec",
    "ObservationRole",
    "PolicyActionBinding",
    "PolicyActionTransform",
]
