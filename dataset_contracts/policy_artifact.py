"""Versioned policy artifact metadata bound to a v2.2 policy-I/O layout.

The artifact payload is derived from :class:`PolicyIOLayout`.  It is suitable
for a checkpoint sidecar or model-server metadata, but it is not another
editable source of tensor semantics: every declared dimension is validated
against the embedded layout and its canonical hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .v22 import (
    DEXTELEOP_DATASET_SCHEMA_V22,
    DatasetLayoutBindingError,
    PolicyIOLayout,
)
from .v23 import DEXTELEOP_DATASET_SCHEMA_V23
from .state_input import (
    STATE_INPUT_MODE_RAW,
    StateInputMode,
    normalize_state_input_mode,
    state_input_dim,
)


POLICY_ARTIFACT_SCHEMA_V1 = "dexteleop_policy_artifact_v1"
POLICY_ARTIFACT_METADATA_KEY = "policy_artifact"
DEPLOY_SCHEMA_V2 = "dexteleop_deploy_v2"
POLICY_ARTIFACT_DATASET_SCHEMAS = (
    DEXTELEOP_DATASET_SCHEMA_V22,
    DEXTELEOP_DATASET_SCHEMA_V23,
)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetLayoutBindingError(f"{name} must be a mapping.")
    return value


@dataclass(frozen=True)
class PolicyTensorField:
    """One named dense or token field in a model artifact contract."""

    key: str
    dim: int
    dtype: str

    def __post_init__(self) -> None:
        key = str(self.key).strip()
        dtype = str(self.dtype).strip().lower()
        dim = int(self.dim)
        if not key:
            raise DatasetLayoutBindingError("Policy tensor field key must be non-empty.")
        if dim < 0:
            raise DatasetLayoutBindingError("Policy tensor field dim must be non-negative.")
        if dtype not in {"float32", "int64"}:
            raise DatasetLayoutBindingError(
                f"Unsupported policy tensor dtype {dtype!r}; expected float32 or int64."
            )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "dtype", dtype)

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "dim": int(self.dim), "dtype": self.dtype}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyTensorField":
        return cls(
            key=str(payload["key"]),
            dim=int(payload["dim"]),
            dtype=str(payload["dtype"]),
        )


@dataclass(frozen=True)
class PolicyCameraInput:
    """One ordered image input expected by a visual policy.

    ``source_key`` is the stable DexTeleop logical camera key (for example
    ``front`` or ``left_wrist``).  ``model_input_key`` is the exact key sent
    to the policy server, while ``server_camera_slot`` is the corresponding
    slot advertised in the server capability handshake.
    """

    source_key: str
    model_input_key: str
    server_camera_slot: str

    def __post_init__(self) -> None:
        for name in ("source_key", "model_input_key", "server_camera_slot"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise DatasetLayoutBindingError(f"Policy camera {name} must be non-empty.")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, str]:
        return {
            "source_key": self.source_key,
            "model_input_key": self.model_input_key,
            "server_camera_slot": self.server_camera_slot,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyCameraInput":
        return cls(
            source_key=str(payload["source_key"]),
            model_input_key=str(payload["model_input_key"]),
            server_camera_slot=str(payload["server_camera_slot"]),
        )


@dataclass(frozen=True)
class PolicyCameraInputContract:
    """Formal visual input contract bound to a checkpoint artifact.

    The ordered ``inputs`` sequence is deliberately explicit: a policy cannot
    silently consume a different camera or rename a view by position.  It is
    additive to artifact v1 so older visual artifacts (which have no camera
    contract) retain their historical adapter defaults.
    """

    inputs: tuple[PolicyCameraInput, ...]
    required: bool = True
    color_space: str = "rgb"
    layout: str = "HWC"
    dtype: str = "uint8"

    def __post_init__(self) -> None:
        inputs = tuple(self.inputs)
        if any(not isinstance(item, PolicyCameraInput) for item in inputs):
            raise TypeError("Policy camera inputs must be PolicyCameraInput entries.")
        for name, values in (
            ("source_key", tuple(item.source_key for item in inputs)),
            ("model_input_key", tuple(item.model_input_key for item in inputs)),
            ("server_camera_slot", tuple(item.server_camera_slot for item in inputs)),
        ):
            if len(set(values)) != len(values):
                raise DatasetLayoutBindingError(
                    f"Policy camera input {name} values must be unique."
                )
        if not isinstance(self.required, bool):
            raise DatasetLayoutBindingError("Policy camera required must be a boolean.")
        required = self.required
        if required and not inputs:
            raise DatasetLayoutBindingError(
                "A required policy camera contract must declare at least one input."
            )
        color_space = str(self.color_space).strip().lower()
        layout = str(self.layout).strip().upper()
        dtype = str(self.dtype).strip().lower()
        if color_space != "rgb":
            raise DatasetLayoutBindingError(
                f"Unsupported policy camera color_space {color_space!r}; expected 'rgb'."
            )
        if layout != "HWC":
            raise DatasetLayoutBindingError(
                f"Unsupported policy camera layout {layout!r}; expected 'HWC'."
            )
        if dtype != "uint8":
            raise DatasetLayoutBindingError(
                f"Unsupported policy camera dtype {dtype!r}; expected 'uint8'."
            )
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "color_space", color_space)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "dtype", dtype)

    @property
    def source_keys(self) -> tuple[str, ...]:
        return tuple(item.source_key for item in self.inputs)

    @property
    def image_key_map(self) -> dict[str, str]:
        return {item.source_key: item.model_input_key for item in self.inputs}

    @property
    def server_camera_slots(self) -> tuple[str, ...]:
        return tuple(item.server_camera_slot for item in self.inputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": [item.to_dict() for item in self.inputs],
            "required": self.required,
            "color_space": self.color_space,
            "layout": self.layout,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyCameraInputContract":
        raw_inputs = payload.get("inputs")
        if not isinstance(raw_inputs, Sequence) or isinstance(raw_inputs, (str, bytes)):
            raise DatasetLayoutBindingError("policy camera inputs must be a sequence.")
        required = payload.get("required", True)
        if not isinstance(required, bool):
            raise DatasetLayoutBindingError("policy camera required must be a boolean.")
        return cls(
            inputs=tuple(
                PolicyCameraInput.from_dict(_mapping(value, name="policy camera input"))
                for value in raw_inputs
            ),
            required=required,
            color_space=str(payload.get("color_space", "rgb")),
            layout=str(payload.get("layout", "HWC")),
            dtype=str(payload.get("dtype", "uint8")),
        )


@dataclass(frozen=True)
class PolicyArtifactContract:
    """Checkpoint/server sidecar whose tensor contract is derived from layout."""

    policy_io_layout: PolicyIOLayout
    model_id: str | None = None
    checkpoint_step: int | str | None = None
    source_datasets: tuple[str, ...] = ()
    state_key: str = "state"
    state_discrete_key: str = "state_discrete"
    state_input_mode: StateInputMode | str = STATE_INPUT_MODE_RAW
    action_key: str = "actions"
    action_discrete_key: str = "actions_discrete"
    schema: str = POLICY_ARTIFACT_SCHEMA_V1
    dataset_schema: str = DEXTELEOP_DATASET_SCHEMA_V22
    deploy_schema_version: str = DEPLOY_SCHEMA_V2
    camera_input: PolicyCameraInputContract | Mapping[str, Any] | None = None
    state_augmentation: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.policy_io_layout, PolicyIOLayout):
            raise TypeError("policy_io_layout must be a PolicyIOLayout.")
        if str(self.schema) != POLICY_ARTIFACT_SCHEMA_V1:
            raise DatasetLayoutBindingError(
                f"Unsupported policy artifact schema {self.schema!r}."
            )
        dataset_schema = str(self.dataset_schema).strip()
        if dataset_schema not in POLICY_ARTIFACT_DATASET_SCHEMAS:
            raise DatasetLayoutBindingError(
                "Policy artifact v1 requires dexteleop_dataset_v2.2 or "
                "dexteleop_dataset_v2.3 semantics."
            )
        if str(self.deploy_schema_version) != DEPLOY_SCHEMA_V2:
            raise DatasetLayoutBindingError(
                f"Policy artifact v1 requires {DEPLOY_SCHEMA_V2!r}."
            )
        if not isinstance(self.state_augmentation, Mapping):
            raise DatasetLayoutBindingError("state_augmentation must be a mapping.")
        camera_input = self.camera_input
        if isinstance(camera_input, Mapping):
            camera_input = PolicyCameraInputContract.from_dict(camera_input)
        elif camera_input is not None and not isinstance(camera_input, PolicyCameraInputContract):
            raise TypeError(
                "camera_input must be a PolicyCameraInputContract, mapping, or None."
            )
        for name in ("state_key", "state_discrete_key", "action_key", "action_discrete_key"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise DatasetLayoutBindingError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "state_input_mode", normalize_state_input_mode(self.state_input_mode))
        object.__setattr__(self, "model_id", _optional_text(self.model_id))
        object.__setattr__(
            self,
            "source_datasets",
            tuple(dict.fromkeys(str(value) for value in self.source_datasets if str(value))),
        )
        object.__setattr__(self, "schema", POLICY_ARTIFACT_SCHEMA_V1)
        object.__setattr__(self, "dataset_schema", dataset_schema)
        object.__setattr__(self, "deploy_schema_version", DEPLOY_SCHEMA_V2)
        object.__setattr__(self, "camera_input", camera_input)
        object.__setattr__(self, "state_augmentation", dict(self.state_augmentation))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def layout_hash(self) -> str:
        return self.policy_io_layout.layout_hash

    @property
    def state_input_dim(self) -> int:
        """Float state dimension expected by this model, not the on-disk base dimension."""

        return state_input_dim(self.policy_io_layout.state_dim, self.state_input_mode)

    @property
    def inputs(self) -> tuple[PolicyTensorField, PolicyTensorField]:
        return (
            PolicyTensorField(self.state_key, self.state_input_dim, "float32"),
            PolicyTensorField(
                self.state_discrete_key,
                self.policy_io_layout.state_discrete_dim,
                "int64",
            ),
        )

    @property
    def outputs(self) -> tuple[PolicyTensorField, PolicyTensorField]:
        return (
            PolicyTensorField(self.action_key, self.policy_io_layout.action_dim, "float32"),
            PolicyTensorField(
                self.action_discrete_key,
                self.policy_io_layout.action_discrete_dim,
                "int64",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POLICY_ARTIFACT_SCHEMA_V1,
            "dataset_schema": self.dataset_schema,
            "deploy_schema_version": DEPLOY_SCHEMA_V2,
            "model_id": self.model_id,
            "checkpoint_step": self.checkpoint_step,
            "source_datasets": list(self.source_datasets),
            "policy_io_layout": self.policy_io_layout.to_dict(),
            "policy_io_layout_hash": self.layout_hash,
            "state_input_mode": self.state_input_mode,
            "state_augmentation": dict(self.state_augmentation),
            "camera_input": None if self.camera_input is None else self.camera_input.to_dict(),
            "inputs": {field.key: field.to_dict() for field in self.inputs},
            "outputs": {field.key: field.to_dict() for field in self.outputs},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyArtifactContract":
        raw_layout = _mapping(payload.get("policy_io_layout"), name="policy_io_layout")
        layout = PolicyIOLayout.from_dict(raw_layout)
        declared_hash = payload.get("policy_io_layout_hash")
        if str(declared_hash or "") != layout.layout_hash:
            raise DatasetLayoutBindingError(
                "Policy artifact policy_io_layout_hash does not match its embedded layout."
            )
        inputs = _mapping(payload.get("inputs"), name="policy artifact inputs")
        outputs = _mapping(payload.get("outputs"), name="policy artifact outputs")
        parsed_inputs = tuple(
            PolicyTensorField.from_dict(_mapping(value, name=f"inputs[{key!r}]"))
            for key, value in inputs.items()
        )
        parsed_outputs = tuple(
            PolicyTensorField.from_dict(_mapping(value, name=f"outputs[{key!r}]"))
            for key, value in outputs.items()
        )
        for field_key, field_value in zip(inputs, parsed_inputs, strict=True):
            if str(field_key) != field_value.key:
                raise DatasetLayoutBindingError(
                    f"Policy artifact input key {field_key!r} disagrees with "
                    f"embedded field key {field_value.key!r}."
                )
        for field_key, field_value in zip(outputs, parsed_outputs, strict=True):
            if str(field_key) != field_value.key:
                raise DatasetLayoutBindingError(
                    f"Policy artifact output key {field_key!r} disagrees with "
                    f"embedded field key {field_value.key!r}."
                )
        float_inputs = [field for field in parsed_inputs if field.dtype == "float32"]
        token_inputs = [field for field in parsed_inputs if field.dtype == "int64"]
        float_outputs = [field for field in parsed_outputs if field.dtype == "float32"]
        token_outputs = [field for field in parsed_outputs if field.dtype == "int64"]
        if not (len(float_inputs) == len(token_inputs) == len(float_outputs) == len(token_outputs) == 1):
            raise DatasetLayoutBindingError(
                "Policy artifact v1 requires one float32 and one int64 input/output field."
            )
        expected = (
            (
                float_inputs[0],
                state_input_dim(layout.state_dim, payload.get("state_input_mode", STATE_INPUT_MODE_RAW)),
            ),
            (token_inputs[0], layout.state_discrete_dim),
            (float_outputs[0], layout.action_dim),
            (token_outputs[0], layout.action_discrete_dim),
        )
        for field_value, expected_dim in expected:
            if field_value.dim != expected_dim:
                raise DatasetLayoutBindingError(
                    f"Policy artifact field {field_value.key!r} dim {field_value.dim} "
                    f"differs from layout dim {expected_dim}."
                )
        source_datasets = payload.get("source_datasets", ())
        if not isinstance(source_datasets, Sequence) or isinstance(source_datasets, (str, bytes)):
            raise DatasetLayoutBindingError("source_datasets must be a sequence.")
        return cls(
            policy_io_layout=layout,
            model_id=payload.get("model_id"),
            checkpoint_step=payload.get("checkpoint_step"),
            source_datasets=tuple(str(value) for value in source_datasets),
            state_key=float_inputs[0].key,
            state_discrete_key=token_inputs[0].key,
            state_input_mode=payload.get("state_input_mode", STATE_INPUT_MODE_RAW),
            action_key=float_outputs[0].key,
            action_discrete_key=token_outputs[0].key,
            schema=str(payload.get("schema", "")),
            dataset_schema=str(payload.get("dataset_schema", "")),
            deploy_schema_version=str(payload.get("deploy_schema_version", "")),
            camera_input=(
                None
                if payload.get("camera_input") is None
                else PolicyCameraInputContract.from_dict(
                    _mapping(payload.get("camera_input"), name="policy camera_input")
                )
            ),
            state_augmentation=dict(_mapping(payload.get("state_augmentation", {}), name="state_augmentation")),
            metadata=dict(_mapping(payload.get("metadata", {}), name="policy artifact metadata")),
        )


def policy_artifact_from_metadata(metadata: Mapping[str, Any]) -> PolicyArtifactContract:
    """Read a top-level artifact payload or a ``policy_artifact`` wrapper."""

    raw = metadata.get(POLICY_ARTIFACT_METADATA_KEY, metadata)
    return PolicyArtifactContract.from_dict(_mapping(raw, name=POLICY_ARTIFACT_METADATA_KEY))


def bind_policy_artifact(
    metadata: Mapping[str, Any] | None,
    artifact: PolicyArtifactContract,
) -> dict[str, Any]:
    bound = {} if metadata is None else dict(metadata)
    bound[POLICY_ARTIFACT_METADATA_KEY] = artifact.to_dict()
    bound["policy_io_layout_hash"] = artifact.layout_hash
    return bound


__all__ = [
    "DEPLOY_SCHEMA_V2",
    "POLICY_ARTIFACT_DATASET_SCHEMAS",
    "POLICY_ARTIFACT_METADATA_KEY",
    "POLICY_ARTIFACT_SCHEMA_V1",
    "PolicyArtifactContract",
    "PolicyCameraInput",
    "PolicyCameraInputContract",
    "PolicyTensorField",
    "bind_policy_artifact",
    "policy_artifact_from_metadata",
]
