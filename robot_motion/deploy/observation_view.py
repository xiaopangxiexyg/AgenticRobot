"""Artifact-bound actor observation allowlist and compatibility firewall."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.immutable import deep_freeze, deep_thaw

from .observation import DeploymentObservation
from .schema import ACTION_DIM, OBSERVATION_STATE_DIM, nan_vector


POLICY_OBSERVATION_CONTRACT_HASH_DOMAIN_V1 = "dexteleop.policy_observation_contract.v1"
POLICY_OBSERVATION_COMPATIBILITY_ADAPTER_V1 = "sanitized_deployment_observation_v1"

DEFAULT_COMPATIBILITY_HAND_STATE_KEYS_V1 = frozenset(
    {
        "joint_pos",
        "tactile",
        "gripper_state",
        "gripper_state_scalar",
        "state",
        "mode",
        "source_timestamp",
        "available_timestamp",
        "timestamp",
        "valid",
    }
)

_COMPATIBILITY_ARM_STATE_KEYS_V1 = frozenset(
    {
        "current_qpos",
        "actual_qpos",
        "current_tcp_pose",
        "actual_tcp_pose",
        "target_pose_base",
        "runtime_raw_current_tcp_pose",
        "runtime_raw_actual_tcp_pose",
        "source_timestamp",
        "available_timestamp",
        "timestamp",
        "valid",
    }
)

_PUBLIC_OBSERVATION_METADATA_KEYS_V1 = frozenset(
    {"prompt", "instruction", "task_instruction", "pose_frame_contract"}
)

ACTOR_SAFE_OBSERVATION_FIELDS_V1 = frozenset(
    {
        "image",
        "images",
        "image_timestamps",
        "image_ages_sec",
        "state",
        "state_dict",
        "state_valid",
        "tactile",
        "tactile_by_side",
        "tactile_valid_by_side",
        "last_action",
        "timestamp",
        "step_index",
        "active_sides",
        "arms_enabled",
        "hands_enabled",
        "end_effector_sides",
    }
)

CONTROL_PRIVATE_OBSERVATION_FIELDS_V1 = frozenset(
    {
        "faults",
        "metadata",
        "selected_owner",
        "authority_epoch",
        "proposal_id",
        "decision_id",
        "safety",
        "ik_status",
        "planner_status",
        "dispatch_receipt",
        "reward",
        "trace",
        "writer_status",
        "backend_status",
    }
)


@dataclass(frozen=True)
class LastActionContract:
    stage: str
    producer_scope: str
    pose_mode: str
    anchor_contract_hash: str | None = None

    def __post_init__(self) -> None:
        for name in ("stage", "producer_scope", "pose_mode"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"LastActionContract.{name} must be non-empty.")


@dataclass(frozen=True)
class PolicyObservationFieldSpec:
    field_name: str
    required: bool = True
    dtype: str | None = None
    shape: tuple[int | None, ...] | None = None
    missing_policy: str = "error"
    access_role: str = "actor"

    def __post_init__(self) -> None:
        field_name = str(self.field_name).strip()
        if not field_name:
            raise ValueError("field_name must be non-empty.")
        object.__setattr__(self, "field_name", field_name)
        if self.access_role != "actor":
            raise ValueError("PolicyObservationView v1 only supports the actor access role.")
        if self.missing_policy not in {"error", "omit", "include_none"}:
            raise ValueError("missing_policy must be 'error', 'omit', or 'include_none'.")
        if self.shape is not None:
            normalized_shape = tuple(None if size is None else int(size) for size in self.shape)
            if any(size is not None and size < 0 for size in normalized_shape):
                raise ValueError("shape dimensions must be non-negative or None.")
            object.__setattr__(self, "shape", normalized_shape)


@dataclass(frozen=True)
class PolicyObservationContract:
    contract_id: str
    fields: tuple[PolicyObservationFieldSpec, ...]
    last_action_contract: LastActionContract | None = None
    compatibility_adapter_id: str = POLICY_OBSERVATION_COMPATIBILITY_ADAPTER_V1
    compatibility_hand_state_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        contract_id = str(self.contract_id).strip()
        if not contract_id:
            raise ValueError("contract_id must be non-empty.")
        object.__setattr__(self, "contract_id", contract_id)
        object.__setattr__(self, "fields", tuple(self.fields))
        adapter_id = str(self.compatibility_adapter_id).strip()
        if adapter_id != POLICY_OBSERVATION_COMPATIBILITY_ADAPTER_V1:
            raise ValueError(f"Unsupported policy observation compatibility adapter: {adapter_id!r}.")
        object.__setattr__(self, "compatibility_adapter_id", adapter_id)
        hand_keys = tuple(
            sorted(
                dict.fromkeys(
                    str(value).strip()
                    for value in (
                        *DEFAULT_COMPATIBILITY_HAND_STATE_KEYS_V1,
                        *self.compatibility_hand_state_keys,
                    )
                    if str(value).strip()
                )
            )
        )
        object.__setattr__(self, "compatibility_hand_state_keys", hand_keys)
        names = [field.field_name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("PolicyObservationContract fields must be unique.")
        forbidden = set(names) & CONTROL_PRIVATE_OBSERVATION_FIELDS_V1
        unknown = set(names) - ACTOR_SAFE_OBSERVATION_FIELDS_V1
        if forbidden:
            raise ValueError(f"Control-private actor fields are forbidden: {sorted(forbidden)}.")
        if unknown:
            raise ValueError(f"Unknown actor observation fields: {sorted(unknown)}.")
        if "last_action" in names and self.last_action_contract is None:
            raise ValueError("last_action requires an explicit LastActionContract.")

    @property
    def semantic_content_hash(self) -> str:
        payload = {
            "contract_id": self.contract_id,
            "fields": tuple(
                {
                    "field_name": field.field_name,
                    "required": field.required,
                    "dtype": field.dtype,
                    "shape": field.shape,
                    "missing_policy": field.missing_policy,
                    "access_role": field.access_role,
                }
                for field in self.fields
            ),
            "last_action_contract": (
                None
                if self.last_action_contract is None
                else {
                    "stage": self.last_action_contract.stage,
                    "producer_scope": self.last_action_contract.producer_scope,
                    "pose_mode": self.last_action_contract.pose_mode,
                    "anchor_contract_hash": self.last_action_contract.anchor_contract_hash,
                }
            ),
            "compatibility_adapter_id": self.compatibility_adapter_id,
            "compatibility_arm_state_keys": tuple(
                sorted(_COMPATIBILITY_ARM_STATE_KEYS_V1)
            ),
            "compatibility_hand_state_keys": self.compatibility_hand_state_keys,
            "public_observation_metadata_keys": tuple(
                sorted(_PUBLIC_OBSERVATION_METADATA_KEYS_V1)
            ),
        }
        return canonical_content_hash(
            payload,
            domain=POLICY_OBSERVATION_CONTRACT_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class PolicyObservationView:
    contract_id: str
    contract_hash: str
    payload: Mapping[str, Any]
    compatibility_payload: Mapping[str, Any] = field(default_factory=dict)
    compatibility_adapter_id: str = POLICY_OBSERVATION_COMPATIBILITY_ADAPTER_V1
    compatibility_hand_state_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.contract_id).strip() or not str(self.contract_hash).strip():
            raise ValueError("PolicyObservationView contract identity/hash must be non-empty.")
        source_payload = dict(self.payload)
        frozen_images = (
            deep_freeze(source_payload["images"])
            if "images" in source_payload
            else None
        )
        image_alias = source_payload.get("image")
        source_images = source_payload.get("images")
        if image_alias is source_images:
            frozen_image_alias = frozen_images
        elif (
            isinstance(source_images, Mapping)
            and len(source_images) == 1
            and image_alias is next(iter(source_images.values()))
            and isinstance(frozen_images, Mapping)
        ):
            frozen_image_alias = next(iter(frozen_images.values()))
        else:
            frozen_image_alias = None

        frozen_payload: dict[str, Any] = {}
        for key, value in source_payload.items():
            if key == "images" and frozen_images is not None:
                frozen_payload[str(key)] = frozen_images
            elif key == "image" and frozen_image_alias is not None:
                frozen_payload[str(key)] = frozen_image_alias
            else:
                frozen_payload[str(key)] = deep_freeze(value)
        object.__setattr__(
            self,
            "payload",
            MappingProxyType(frozen_payload),
        )
        object.__setattr__(
            self,
            "compatibility_payload",
            MappingProxyType(
                {
                    str(key): deep_freeze(value)
                    for key, value in self.compatibility_payload.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "compatibility_hand_state_keys",
            tuple(
                sorted(
                    dict.fromkeys(
                        (
                            *DEFAULT_COMPATIBILITY_HAND_STATE_KEYS_V1,
                            *self.compatibility_hand_state_keys,
                        )
                    )
                )
            ),
        )
        if self.compatibility_adapter_id != POLICY_OBSERVATION_COMPATIBILITY_ADAPTER_V1:
            raise ValueError("Unsupported PolicyObservationView compatibility adapter.")

    def model_input(self) -> dict[str, Any]:
        thawed_images = (
            deep_thaw(self.payload["images"])
            if "images" in self.payload
            else None
        )
        frozen_image_alias = self.payload.get("image")
        frozen_images = self.payload.get("images")
        if frozen_image_alias is frozen_images:
            thawed_image_alias = thawed_images
        elif (
            isinstance(frozen_images, Mapping)
            and len(frozen_images) == 1
            and frozen_image_alias is next(iter(frozen_images.values()))
            and isinstance(thawed_images, Mapping)
        ):
            thawed_image_alias = next(iter(thawed_images.values()))
        else:
            thawed_image_alias = None

        result: dict[str, Any] = {}
        for key, value in self.payload.items():
            if key == "images" and thawed_images is not None:
                result[key] = thawed_images
            elif key == "image" and thawed_image_alias is not None:
                result[key] = thawed_image_alias
            else:
                result[key] = deep_thaw(value)
        return result

    def __getattr__(self, name: str) -> Any:
        payload = object.__getattribute__(self, "payload")
        if name in payload:
            return deep_thaw(payload[name])
        compatibility = object.__getattribute__(self, "compatibility_payload")
        if name in compatibility:
            return deep_thaw(compatibility[name])
        raise AttributeError(name)

    def hold_action(self) -> np.ndarray:
        return DeploymentObservation.hold_action(self)

    def to_deployment_observation(self) -> DeploymentObservation:
        """Build a legacy-shape adapter containing no private control fields."""

        payload = self.model_input()
        compatibility = {
            key: deep_thaw(value) for key, value in self.compatibility_payload.items()
        }
        return DeploymentObservation(
            timestamp=float(payload.get("timestamp", 0.0)),
            step_index=int(payload.get("step_index", 0)),
            active_sides=tuple(payload.get("active_sides", ())),
            arms_enabled=bool(payload.get("arms_enabled", False)),
            hands_enabled=bool(payload.get("hands_enabled", False)),
            end_effector_sides=tuple(payload.get("end_effector_sides", ())),
            images=dict(payload.get("images", {})),
            image_timestamps=dict(payload.get("image_timestamps", {})),
            image_ages_sec=dict(payload.get("image_ages_sec", {})),
            state=np.asarray(
                payload.get("state", nan_vector(OBSERVATION_STATE_DIM)), dtype=np.float32
            ).copy(),
            state_dict=dict(payload.get("state_dict", {})),
            state_valid=np.asarray(
                payload.get("state_valid", np.zeros(OBSERVATION_STATE_DIM, dtype=bool)),
                dtype=bool,
            ).copy(),
            tactile=np.asarray(
                payload.get("tactile", np.zeros((0,), dtype=np.float32))
            ).copy(),
            tactile_by_side=dict(payload.get("tactile_by_side", {})),
            tactile_valid_by_side=dict(payload.get("tactile_valid_by_side", {})),
            last_action=np.asarray(
                payload.get("last_action", nan_vector(ACTION_DIM)), dtype=np.float32
            ).copy(),
            arm_states=dict(compatibility.get("arm_states", {})),
            hand_states=dict(compatibility.get("hand_states", {})),
            faults=[],
            metadata=dict(compatibility.get("public_metadata", {})),
        )


@dataclass(frozen=True)
class PolicyObservationParityReceipt:
    contract_id: str
    contract_hash: str
    matched: bool
    checked_fields: tuple[str, ...]
    difference_paths: tuple[str, ...]
    excluded_legacy_private_fields: tuple[str, ...]


def default_deploy_policy_observation_contract(
    *,
    binding_id: str,
    last_action_pose_mode: str,
    compatibility_hand_state_keys: Sequence[str] = (),
) -> PolicyObservationContract:
    binding = str(binding_id).strip()
    if not binding:
        raise ValueError("Policy observation binding_id must be non-empty.")
    return PolicyObservationContract(
        contract_id=f"dexteleop.deploy.actor.v1:{binding}",
        fields=(
            PolicyObservationFieldSpec("image", required=False, missing_policy="include_none"),
            PolicyObservationFieldSpec("images"),
            PolicyObservationFieldSpec("image_timestamps"),
            PolicyObservationFieldSpec("image_ages_sec"),
            PolicyObservationFieldSpec(
                "state", dtype="float32", shape=(OBSERVATION_STATE_DIM,)
            ),
            PolicyObservationFieldSpec("state_dict"),
            PolicyObservationFieldSpec(
                "state_valid", dtype="bool", shape=(OBSERVATION_STATE_DIM,)
            ),
            PolicyObservationFieldSpec("tactile", dtype="float32", shape=(None,)),
            PolicyObservationFieldSpec("tactile_by_side"),
            PolicyObservationFieldSpec("tactile_valid_by_side"),
            PolicyObservationFieldSpec(
                "last_action", dtype="float32", shape=(ACTION_DIM,)
            ),
            PolicyObservationFieldSpec("timestamp"),
            PolicyObservationFieldSpec("step_index"),
            PolicyObservationFieldSpec("active_sides"),
            PolicyObservationFieldSpec("arms_enabled"),
            PolicyObservationFieldSpec("hands_enabled"),
            PolicyObservationFieldSpec("end_effector_sides"),
        ),
        last_action_contract=LastActionContract(
            stage="previous_safe_action",
            producer_scope="policy_instance",
            pose_mode=str(last_action_pose_mode),
        ),
        compatibility_hand_state_keys=tuple(compatibility_hand_state_keys),
    )


def _observation_field(observation: DeploymentObservation, field_name: str) -> Any:
    if field_name == "image":
        return observation.image
    if field_name == "end_effector_sides":
        return tuple(observation.end_effector_sides or ())
    return getattr(observation, field_name)


def _validate_field(value: Any, spec: PolicyObservationFieldSpec) -> None:
    if spec.dtype is None and spec.shape is None:
        return
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {spec.field_name!r} is not array-like.") from exc
    if spec.dtype is not None and array.dtype != np.dtype(spec.dtype):
        raise ValueError(
            f"Field {spec.field_name!r} dtype mismatch: expected {spec.dtype}, got {array.dtype}."
        )
    if spec.shape is not None:
        if len(array.shape) != len(spec.shape) or any(
            expected is not None and actual != expected
            for actual, expected in zip(array.shape, spec.shape)
        ):
            raise ValueError(
                f"Field {spec.field_name!r} shape mismatch: expected {spec.shape}, got {array.shape}."
            )


def _sanitize_state_mapping(
    states: Mapping[str, Mapping[str, Any]],
    *,
    allowed_keys: frozenset[str] | set[str],
) -> dict[str, dict[str, Any]]:
    return {
        str(side): {
            str(key): value
            for key, value in dict(state or {}).items()
            if str(key) in allowed_keys
        }
        for side, state in states.items()
    }


def _public_metadata(observation: DeploymentObservation) -> dict[str, Any]:
    metadata = dict(observation.metadata)
    public = {
        key: metadata[key]
        for key in _PUBLIC_OBSERVATION_METADATA_KEYS_V1
        if key in metadata
    }
    task = metadata.get("task")
    if isinstance(task, Mapping) and task.get("instruction"):
        public["task_instruction"] = str(task["instruction"])
    return public


def _value_difference_paths(expected: Any, actual: Any, *, path: str) -> list[str]:
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        try:
            expected_array = np.asarray(expected)
            actual_array = np.asarray(actual)
        except (TypeError, ValueError):
            return [path]
        if expected_array.dtype != actual_array.dtype or expected_array.shape != actual_array.shape:
            return [path]
        if expected_array.dtype.kind in {"f", "c"}:
            equal = np.array_equal(expected_array, actual_array, equal_nan=True)
        else:
            equal = np.array_equal(expected_array, actual_array)
        return [] if equal else [path]
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        keys = set(expected) | set(actual)
        for key in sorted(str(value) for value in keys):
            if key not in expected or key not in actual:
                differences.append(f"{path}.{key}")
            else:
                differences.extend(
                    _value_difference_paths(expected[key], actual[key], path=f"{path}.{key}")
                )
        return differences
    if isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        if len(expected) != len(actual):
            return [path]
        differences = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            differences.extend(
                _value_difference_paths(left, right, path=f"{path}[{index}]")
            )
        return differences
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            if np.isnan(expected) and np.isnan(actual):
                return []
        except TypeError:
            pass
    return [] if expected == actual else [path]


def compare_policy_observation_view(
    legacy_observation: DeploymentObservation | Mapping[str, Any],
    view: PolicyObservationView,
) -> PolicyObservationParityReceipt:
    if isinstance(legacy_observation, DeploymentObservation):
        legacy_model_input = legacy_observation.model_input()
    else:
        legacy_model_input = legacy_observation
    strict_input = view.payload
    differences: list[str] = []
    checked_fields: list[str] = list(strict_input)
    compared_identity_pairs: set[tuple[int, int]] = set()
    for field_name, strict_value in strict_input.items():
        if field_name not in legacy_model_input:
            differences.append(field_name)
            continue
        identity_pair = (id(legacy_model_input[field_name]), id(strict_value))
        if identity_pair in compared_identity_pairs:
            continue
        compared_identity_pairs.add(identity_pair)
        differences.extend(
            _value_difference_paths(
                legacy_model_input[field_name], strict_value, path=field_name
            )
        )
    if isinstance(legacy_observation, DeploymentObservation):
        expected_compatibility = {
            "compatibility.arm_states": _sanitize_state_mapping(
                legacy_observation.arm_states,
                allowed_keys=_COMPATIBILITY_ARM_STATE_KEYS_V1,
            ),
            "compatibility.hand_states": _sanitize_state_mapping(
                legacy_observation.hand_states,
                allowed_keys=set(view.compatibility_hand_state_keys),
            ),
            "compatibility.public_metadata": _public_metadata(legacy_observation),
        }
        actual_compatibility = {
            "compatibility.arm_states": view.compatibility_payload.get(
                "arm_states", {}
            ),
            "compatibility.hand_states": view.compatibility_payload.get(
                "hand_states", {}
            ),
            "compatibility.public_metadata": view.compatibility_payload.get(
                "public_metadata", {}
            ),
        }
        for field_name, expected in expected_compatibility.items():
            checked_fields.append(field_name)
            differences.extend(
                _value_difference_paths(
                    expected,
                    actual_compatibility[field_name],
                    path=field_name,
                )
            )
    private_fields = tuple(
        sorted(set(legacy_model_input) & CONTROL_PRIVATE_OBSERVATION_FIELDS_V1)
    )
    return PolicyObservationParityReceipt(
        contract_id=view.contract_id,
        contract_hash=view.contract_hash,
        matched=not differences,
        checked_fields=tuple(checked_fields),
        difference_paths=tuple(differences),
        excluded_legacy_private_fields=private_fields,
    )


def build_policy_observation_view(
    observation: DeploymentObservation,
    *,
    contract: PolicyObservationContract,
) -> PolicyObservationView:
    projected: dict[str, Any] = {}
    for spec in contract.fields:
        value = _observation_field(observation, spec.field_name)
        if value is None and not spec.required and spec.missing_policy == "omit":
            continue
        if value is None and spec.required:
            raise ValueError(f"Required observation field {spec.field_name!r} is missing.")
        _validate_field(value, spec)
        projected[spec.field_name] = value
    compatibility_payload = {
        "arm_states": _sanitize_state_mapping(
            observation.arm_states,
            allowed_keys=_COMPATIBILITY_ARM_STATE_KEYS_V1,
        ),
        "hand_states": _sanitize_state_mapping(
            observation.hand_states,
            allowed_keys=set(contract.compatibility_hand_state_keys),
        ),
        "public_metadata": _public_metadata(observation),
    }
    return PolicyObservationView(
        contract_id=contract.contract_id,
        contract_hash=contract.semantic_content_hash,
        payload=projected,
        compatibility_payload=compatibility_payload,
        compatibility_adapter_id=contract.compatibility_adapter_id,
        compatibility_hand_state_keys=contract.compatibility_hand_state_keys,
    )


__all__ = [
    "ACTOR_SAFE_OBSERVATION_FIELDS_V1",
    "CONTROL_PRIVATE_OBSERVATION_FIELDS_V1",
    "DEFAULT_COMPATIBILITY_HAND_STATE_KEYS_V1",
    "LastActionContract",
    "POLICY_OBSERVATION_COMPATIBILITY_ADAPTER_V1",
    "PolicyObservationContract",
    "PolicyObservationFieldSpec",
    "PolicyObservationParityReceipt",
    "PolicyObservationView",
    "build_policy_observation_view",
    "compare_policy_observation_view",
    "default_deploy_policy_observation_contract",
]
