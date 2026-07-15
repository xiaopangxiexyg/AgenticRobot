from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from dataset_contracts.state_input import (
    STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT,
    normalize_state_input_mode,
)


DEXTELEOP_DEPLOY_SCHEMA_VERSION = "dexteleop_deploy_v1"
DEXTELEOP_DEPLOY_SCHEMA_V2 = "dexteleop_deploy_v2"
DEFAULT_ACTION_KEY = "actions"


class ServerMetadataError(ValueError):
    """Raised when a remote model server is incompatible with this client."""


@dataclass(frozen=True)
class DeployPolicyContract:
    """Local expectations for a remote model server.

    This contract is intentionally model-family agnostic. OpenPI, ACT, Diffusion
    Policy, or another server can set protocol-specific expectations while
    reusing the same validation and unified-action output checks.
    """

    protocol_version: str | None = None
    deploy_schema_version: str | None = None
    model_id: str | None = None
    checkpoint_step: int | str | None = None
    state_key: str | None = None
    state_dim: int | None = None
    state_input_mode: str | None = None
    state_discrete_key: str | None = None
    state_discrete_dim: int | None = None
    action_key: str = DEFAULT_ACTION_KEY
    action_dim: int = 26
    action_discrete_dim: int | None = None
    policy_io_layout_hash: str | None = None
    min_action_horizon: int = 1
    active_sides: tuple[str, ...] = ("left", "right")
    require_active_sides_match: bool = True
    required_camera_slots: tuple[str, ...] = ()
    hand_layout: str | None = "q6_slot0_gripper"
    metadata_required: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_version", _optional_str(self.protocol_version))
        object.__setattr__(self, "deploy_schema_version", _optional_str(self.deploy_schema_version))
        object.__setattr__(self, "model_id", _optional_str(self.model_id))
        object.__setattr__(self, "state_key", _optional_str(self.state_key))
        state_input_mode = self.state_input_mode
        object.__setattr__(
            self,
            "state_input_mode",
            None if state_input_mode is None else normalize_state_input_mode(state_input_mode),
        )
        object.__setattr__(self, "state_discrete_key", _optional_str(self.state_discrete_key))
        for name in ("state_dim", "state_discrete_dim"):
            value = getattr(self, name)
            if value is not None:
                value = int(value)
                if value < 0:
                    raise ValueError(f"{name} must be non-negative when declared.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "action_key", str(self.action_key or DEFAULT_ACTION_KEY))
        object.__setattr__(self, "action_dim", int(self.action_dim))
        discrete_dim = self.action_discrete_dim
        if discrete_dim is not None:
            discrete_dim = int(discrete_dim)
            if discrete_dim < 0:
                raise ValueError("action_discrete_dim must be non-negative when declared.")
        object.__setattr__(self, "action_discrete_dim", discrete_dim)
        object.__setattr__(self, "policy_io_layout_hash", _optional_str(self.policy_io_layout_hash))
        object.__setattr__(self, "min_action_horizon", int(self.min_action_horizon))
        object.__setattr__(self, "active_sides", _normalize_sides(self.active_sides))
        object.__setattr__(self, "required_camera_slots", tuple(str(v) for v in self.required_camera_slots))
        object.__setattr__(self, "hand_layout", _optional_str(self.hand_layout))
        object.__setattr__(self, "extra", dict(self.extra))


@dataclass(frozen=True)
class MetadataValidationResult:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    server_summary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "errors", tuple(str(v) for v in self.errors))
        object.__setattr__(self, "warnings", tuple(str(v) for v in self.warnings))
        object.__setattr__(self, "server_summary", dict(self.server_summary))


def validate_server_metadata(
    metadata: Mapping[str, Any] | None,
    contract: DeployPolicyContract | None = None,
    *,
    raise_on_error: bool = True,
) -> MetadataValidationResult:
    contract = DeployPolicyContract() if contract is None else contract
    payload = {} if metadata is None else dict(metadata)
    errors: list[str] = []
    warnings: list[str] = []

    if not payload and contract.metadata_required:
        errors.append("server_metadata_missing")

    protocol = _optional_str(_nested_get(payload, "protocol_version"))
    if contract.protocol_version is not None and protocol != contract.protocol_version:
        errors.append(
            "protocol_version_mismatch:"
            f"expected={contract.protocol_version!r}, actual={protocol!r}"
        )
    elif contract.protocol_version is None and protocol is None:
        warnings.append("protocol_version_not_declared")

    if contract.deploy_schema_version is not None:
        schema = _optional_str(
            _first_present(
                payload,
                ("deploy_schema_version", "schema_version", "deployment.deploy_schema_version"),
            )
        )
        if schema != contract.deploy_schema_version:
            errors.append(
                "deploy_schema_version_mismatch:"
                f"expected={contract.deploy_schema_version!r}, actual={schema!r}"
            )
    elif _first_present(payload, ("deploy_schema_version", "schema_version")) is None:
        warnings.append("deploy_schema_version_not_declared")

    server_model_id = _server_model_id(payload)
    if contract.model_id is not None and server_model_id != contract.model_id:
        errors.append(f"model_id_mismatch:expected={contract.model_id!r}, actual={server_model_id!r}")

    checkpoint_step = _optional_str(_nested_get(payload, "deployment.checkpoint_step"))
    if contract.checkpoint_step is not None and checkpoint_step != str(contract.checkpoint_step):
        errors.append(
            "checkpoint_step_mismatch:"
            f"expected={str(contract.checkpoint_step)!r}, actual={checkpoint_step!r}"
        )

    state_dim = _server_state_dim(payload)
    if contract.state_dim is not None:
        if state_dim is None:
            errors.append("input_state_dim_missing")
        elif state_dim != contract.state_dim:
            errors.append(
                f"input_state_dim_mismatch:expected={contract.state_dim}, actual={state_dim}"
            )
    state_discrete_dim = _server_state_discrete_dim(payload)
    if contract.state_discrete_dim is not None:
        if state_discrete_dim is None:
            errors.append("input_state_discrete_dim_missing")
        elif state_discrete_dim != contract.state_discrete_dim:
            errors.append(
                "input_state_discrete_dim_mismatch:"
                f"expected={contract.state_discrete_dim}, actual={state_discrete_dim}"
            )
    state_key = _optional_str(
        _first_present(payload, ("local_robot.input_state_key", "model.state_key", "state_key"))
    )
    if contract.state_key is not None and state_key != contract.state_key:
        errors.append(
            f"input_state_key_mismatch:expected={contract.state_key!r}, actual={state_key!r}"
        )
    server_state_input_mode = _optional_str(
        _first_present(
            payload,
            (
                "local_robot.input_state_mode",
                "dataset_adapter.state_input_mode",
                "model.state_input_mode",
                "state_input_mode",
            ),
        )
    )
    if contract.state_input_mode == STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT:
        if server_state_input_mode != contract.state_input_mode:
            errors.append(
                "input_state_mode_mismatch:"
                f"expected={contract.state_input_mode!r}, actual={server_state_input_mode!r}"
            )
    elif contract.state_input_mode is not None and server_state_input_mode not in (None, contract.state_input_mode):
        errors.append(
            "input_state_mode_mismatch:"
            f"expected={contract.state_input_mode!r}, actual={server_state_input_mode!r}"
        )
    state_discrete_key = _optional_str(
        _first_present(
            payload,
            (
                "local_robot.input_state_discrete_key",
                "model.state_discrete_key",
                "state_discrete_key",
            ),
        )
    )
    if (
        contract.state_discrete_key is not None
        and state_discrete_key != contract.state_discrete_key
    ):
        errors.append(
            "input_state_discrete_key_mismatch:"
            f"expected={contract.state_discrete_key!r}, actual={state_discrete_key!r}"
        )

    action_dim = _server_action_dim(payload)
    if action_dim is None:
        errors.append("output_action_dim_missing")
    elif int(action_dim) != contract.action_dim:
        errors.append(f"output_action_dim_mismatch:expected={contract.action_dim}, actual={int(action_dim)}")

    if contract.action_discrete_dim is not None:
        server_discrete_dim = _server_action_discrete_dim(payload)
        if server_discrete_dim is None:
            errors.append("output_action_discrete_dim_missing")
        elif int(server_discrete_dim) != int(contract.action_discrete_dim):
            errors.append(
                "output_action_discrete_dim_mismatch:"
                f"expected={int(contract.action_discrete_dim)}, actual={int(server_discrete_dim)}"
            )
    else:
        server_discrete_dim = _server_action_discrete_dim(payload)

    if contract.policy_io_layout_hash is not None:
        server_layout_hash = _optional_str(
            _first_present(
                payload,
                (
                    "policy_io_layout_hash",
                    "dataset_adapter.policy_io_layout_hash",
                    "model.policy_io_layout_hash",
                    "deployment.policy_io_layout_hash",
                ),
            )
        )
        if server_layout_hash != contract.policy_io_layout_hash:
            errors.append(
                "policy_io_layout_hash_mismatch:"
                f"expected={contract.policy_io_layout_hash!r}, actual={server_layout_hash!r}"
            )
    else:
        server_layout_hash = _optional_str(
            _first_present(
                payload,
                (
                    "policy_io_layout_hash",
                    "dataset_adapter.policy_io_layout_hash",
                    "model.policy_io_layout_hash",
                    "deployment.policy_io_layout_hash",
                ),
            )
        )

    horizon = _int_or_none(_nested_get(payload, "model.action_horizon"))
    if horizon is None:
        warnings.append("action_horizon_missing")
    elif horizon < contract.min_action_horizon:
        errors.append(
            f"action_horizon_too_short:expected_min={contract.min_action_horizon}, actual={horizon}"
        )

    server_sides = _normalize_sides(_nested_get(payload, "local_robot.active_sides") or ())
    if contract.require_active_sides_match and server_sides and server_sides != contract.active_sides:
        errors.append(
            "active_sides_mismatch:"
            f"expected={list(contract.active_sides)!r}, actual={list(server_sides)!r}"
        )

    if contract.required_camera_slots:
        server_slots = set(_server_camera_slots(payload))
        missing_slots = [slot for slot in contract.required_camera_slots if slot not in server_slots]
        if missing_slots:
            errors.append(f"camera_slots_missing:{','.join(missing_slots)}")

    if contract.hand_layout is not None and not _metadata_supports_hand_layout(payload, contract.hand_layout):
        errors.append(f"hand_layout_mismatch:expected={contract.hand_layout!r}")

    summary = {
        "protocol_version": protocol,
        "deploy_schema_version": _first_present(payload, ("deploy_schema_version", "schema_version")),
        "model_id": server_model_id,
        "checkpoint_step": checkpoint_step,
        "state_key": state_key,
        "state_dim": state_dim,
        "state_input_mode": server_state_input_mode,
        "state_discrete_key": state_discrete_key,
        "state_discrete_dim": state_discrete_dim,
        "action_dim": action_dim,
        "action_discrete_dim": server_discrete_dim,
        "policy_io_layout_hash": server_layout_hash,
        "action_horizon": horizon,
        "active_sides": list(server_sides),
        "camera_slots": _server_camera_slots(payload),
    }
    result = MetadataValidationResult(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        server_summary=summary,
    )
    if errors and raise_on_error:
        raise ServerMetadataError("; ".join(errors))
    return result


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_sides(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw = (values,)
    else:
        try:
            raw = tuple(values)
        except TypeError:
            raw = (values,)
    requested = {str(side).strip().lower() for side in raw if str(side).strip()}
    return tuple(side for side in ("left", "right") if side in requested)


def _nested_get(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in str(key).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = _nested_get(mapping, key)
        if value is not None:
            return value
    return None


def _server_model_id(metadata: Mapping[str, Any]) -> str | None:
    return _optional_str(
        _first_present(
            metadata,
            (
                "model_id",
                "deployment.model_id",
                "deployment.config_name",
                "config_name",
            ),
        )
    )


def _server_action_dim(metadata: Mapping[str, Any]) -> int | None:
    return _int_or_none(
        _first_present(
            metadata,
            (
                "local_robot.output_action_dim",
                "dataset_adapter.deployment_action_dim",
                "output_action_dim",
                "action_dim",
            ),
        )
    )


def _server_state_dim(metadata: Mapping[str, Any]) -> int | None:
    return _int_or_none(
        _first_present(
            metadata,
            (
                "local_robot.input_state_dim",
                "dataset_adapter.state_dim",
                "dataset_adapter.model_state_dim",
                "model.state_dim",
                "input_state_dim",
                "state_dim",
            ),
        )
    )


def _server_state_discrete_dim(metadata: Mapping[str, Any]) -> int | None:
    return _int_or_none(
        _first_present(
            metadata,
            (
                "local_robot.input_state_discrete_dim",
                "dataset_adapter.state_discrete_dim",
                "dataset_adapter.model_state_discrete_dim",
                "model.state_discrete_dim",
                "input_state_discrete_dim",
                "state_discrete_dim",
            ),
        )
    )


def _server_action_discrete_dim(metadata: Mapping[str, Any]) -> int | None:
    return _int_or_none(
        _first_present(
            metadata,
            (
                "local_robot.output_action_discrete_dim",
                "dataset_adapter.action_discrete_dim",
                "dataset_adapter.deployment_action_discrete_dim",
                "model.action_discrete_dim",
                "action_discrete_dim",
            ),
        )
    )


def _server_camera_slots(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    value = _first_present(
        metadata,
        (
            "dataset_adapter.camera_slots",
            "camera_contract.selected_keys",
            "local_robot.camera_slots",
        ),
    )
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return (str(value),)


def _metadata_supports_hand_layout(metadata: Mapping[str, Any], expected: str) -> bool:
    expected = str(expected)
    declared = _optional_str(
        _first_present(
            metadata,
            (
                "hand_layout",
                "local_robot.hand_layout",
                "dataset_adapter.hand_layout",
            ),
        )
    )
    if declared is not None:
        return declared == expected
    order = _first_present(metadata, ("dataset_adapter.deployment_action_order",))
    if isinstance(order, Sequence) and not isinstance(order, (str, bytes)):
        return any(expected in str(item) for item in order)
    return False


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
