"""Pure numeric-normalization contracts for policy deployment.

This module describes *numeric* transforms only.  It deliberately does not
encode pose frames, relative/absolute semantics, tensor layout, image resize,
or action chunk timing; those are separate contracts.  Keeping those concepts
separate prevents a field called ``normalized`` from silently changing units or
applying an inverse transform twice.

Each domain names the owner of a transform:

``none``
    The values are already in the declared output space.  No transform is
    allowed or implied.
``client``
    The local policy adapter must apply the declared transform before sending
    values to the model/runtime boundary.
``server``
    The remote model server owns the transform.  The local adapter must pass
    values through and bind the server's contract/hash during preflight.

The current deployment runtime only performs the representation conversions
that were already present before this contract was introduced.  This module is
therefore intentionally a strict declaration/validation boundary first; a
non-identity client-owned transform must not be guessed by a generic adapter.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash


NORMALIZATION_CONTRACT_SCHEMA_V1 = "dexteleop_normalization_contract_v1"
NORMALIZATION_CONTRACT_HASH_DOMAIN_V1 = "dexteleop.normalization_contract.v1"
NORMALIZATION_DOMAINS = ("state", "action", "image")
NORMALIZATION_OWNERS = ("none", "client", "server")
NORMALIZATION_METHODS = ("identity", "zscore", "quantile_range", "affine", "range")
NORMALIZATION_OPERATIONS = ("identity", "normalize", "unnormalize")

# These schemas predate an executable owner/method contract.  They are
# accepted only as opaque compatibility payloads; no code may infer a numeric
# transform from them.
LEGACY_NORMALIZATION_SCHEMAS = frozenset({"legacy_normalized_v1", "normalized_v1"})


class NormalizationContractError(ValueError):
    """Raised when a normalization declaration is incomplete or ambiguous."""


def _text(value: Any, *, name: str) -> str:
    result = str(value or "").strip().lower()
    if not result:
        raise NormalizationContractError(f"{name} must be non-empty.")
    return result


def _mapping(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise NormalizationContractError(f"{name} must be a mapping.")
    return {str(key): value[key] for key in value}


def _finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NormalizationContractError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise NormalizationContractError(f"{name} must be a finite number.")
    return result


def _validate_numeric_parameter(value: Any, *, name: str) -> Any:
    """Validate scalar/list numeric parameters without changing their shape."""

    if isinstance(value, Mapping):
        return {
            str(key): _validate_numeric_parameter(item, name=f"{name}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _validate_numeric_parameter(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    return _finite(value, name=name)


@dataclass(frozen=True)
class NormalizationDomainSpec:
    """One numeric transform declaration for state, action, or image values."""

    owner: str = "none"
    method: str = "identity"
    input_space: str = "raw"
    output_space: str = "raw"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    stats_ref: str | None = None
    stats_hash: str | None = None
    # The direction is part of the contract because an action output usually
    # performs the inverse of the state input transform.  ``None`` preserves
    # the constructor shorthand and is materialized explicitly in ``to_dict``.
    operation: str | None = None

    def __post_init__(self) -> None:
        owner = _text(self.owner, name="normalization owner")
        method = _text(self.method, name="normalization method")
        input_space = _text(self.input_space, name="normalization input_space")
        output_space = _text(self.output_space, name="normalization output_space")
        operation = (
            None
            if self.operation is None
            else _text(self.operation, name="normalization operation")
        )
        if owner not in NORMALIZATION_OWNERS:
            raise NormalizationContractError(
                f"normalization owner must be one of {NORMALIZATION_OWNERS!r}, got {owner!r}."
            )
        if method not in NORMALIZATION_METHODS:
            raise NormalizationContractError(
                f"normalization method must be one of {NORMALIZATION_METHODS!r}, got {method!r}."
            )
        if operation is None:
            operation = "identity" if method == "identity" else "normalize"
        if operation not in NORMALIZATION_OPERATIONS:
            raise NormalizationContractError(
                "normalization operation must be one of "
                f"{NORMALIZATION_OPERATIONS!r}, got {operation!r}."
            )
        parameters = _mapping(self.parameters, name="normalization parameters")
        if method == "identity":
            if owner != "none" or operation != "identity":
                raise NormalizationContractError(
                    "identity normalization must use owner='none' and "
                    "operation='identity'; an identity transform has no executor."
                )
            if input_space != output_space:
                raise NormalizationContractError(
                    "identity normalization requires input_space == output_space."
                )
            if parameters or self.stats_ref is not None or self.stats_hash is not None:
                raise NormalizationContractError(
                    "identity normalization must not carry parameters or statistics."
                )
        else:
            if operation == "identity":
                raise NormalizationContractError(
                    f"{method} normalization must use operation='normalize' or "
                    "operation='unnormalize'."
                )
            if owner == "none":
                raise NormalizationContractError(
                    f"{method} normalization requires owner='client' or owner='server'."
                )
            if input_space == output_space:
                raise NormalizationContractError(
                    f"{method} normalization must declare distinct input_space and output_space."
                )
            if method in {"zscore", "quantile_range"}:
                if not str(self.stats_ref or "").strip():
                    raise NormalizationContractError(
                        f"{method} normalization requires stats_ref."
                    )
                if not str(self.stats_hash or "").strip():
                    raise NormalizationContractError(
                        f"{method} normalization requires stats_hash."
                    )
                if parameters:
                    raise NormalizationContractError(
                        f"{method} parameters must be stored in the referenced "
                        "statistics artifact."
                    )
            elif method == "affine":
                if self.stats_ref is not None or self.stats_hash is not None:
                    raise NormalizationContractError(
                        "affine normalization must keep parameters inline; "
                        "stats_ref/stats_hash are not allowed."
                    )
                if "scale" not in parameters or "offset" not in parameters:
                    raise NormalizationContractError(
                        "affine normalization requires numeric scale and offset parameters."
                    )
                scale = _validate_numeric_parameter(parameters["scale"], name="scale")
                offset = _validate_numeric_parameter(parameters["offset"], name="offset")
                # A zero scale is never a reversible normalization and is
                # almost always an accidental placeholder.
                if isinstance(scale, (int, float)) and scale == 0.0:
                    raise NormalizationContractError("affine normalization scale must be non-zero.")
                parameters = {**parameters, "scale": scale, "offset": offset}
            elif method == "range":
                if self.stats_ref is not None or self.stats_hash is not None:
                    raise NormalizationContractError(
                        "range normalization must keep bounds inline; "
                        "stats_ref/stats_hash are not allowed."
                    )
                required = ("input_min", "input_max", "output_min", "output_max")
                missing = [key for key in required if key not in parameters]
                if missing:
                    raise NormalizationContractError(
                        "range normalization is missing parameters: " + ", ".join(missing)
                    )
                input_min = _finite(parameters["input_min"], name="input_min")
                input_max = _finite(parameters["input_max"], name="input_max")
                output_min = _finite(parameters["output_min"], name="output_min")
                output_max = _finite(parameters["output_max"], name="output_max")
                if input_max <= input_min or output_max <= output_min:
                    raise NormalizationContractError(
                        "range normalization requires max > min for both spaces."
                    )
                parameters = {
                    **parameters,
                    "input_min": input_min,
                    "input_max": input_max,
                    "output_min": output_min,
                    "output_max": output_max,
                }
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "input_space", input_space)
        object.__setattr__(self, "output_space", output_space)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "stats_ref", None if self.stats_ref is None else str(self.stats_ref).strip())
        object.__setattr__(self, "stats_hash", None if self.stats_hash is None else str(self.stats_hash).strip())
        object.__setattr__(self, "operation", operation)

    @property
    def is_identity(self) -> bool:
        return self.method == "identity"

    @property
    def is_client_owned(self) -> bool:
        return self.owner == "client" and not self.is_identity

    @property
    def is_server_owned(self) -> bool:
        return self.owner == "server" and not self.is_identity

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "owner": self.owner,
            "method": self.method,
            "input_space": self.input_space,
            "output_space": self.output_space,
            "operation": self.operation,
        }
        if self.parameters:
            payload["parameters"] = dict(self.parameters)
        if self.stats_ref is not None:
            payload["stats_ref"] = self.stats_ref
        if self.stats_hash is not None:
            payload["stats_hash"] = self.stats_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "NormalizationDomainSpec":
        values = _mapping(payload, name="normalization domain")
        unknown = sorted(
            set(values).difference(
                {
                    "owner",
                    "method",
                    "input_space",
                    "output_space",
                    "operation",
                    "parameters",
                    "stats_ref",
                    "stats_hash",
                }
            )
        )
        if unknown:
            raise NormalizationContractError(
                "normalization domain contains unknown entries: " + ", ".join(unknown)
            )
        return cls(
            owner=values.get("owner", "none"),
            method=values.get("method", "identity"),
            input_space=values.get("input_space", "raw"),
            output_space=values.get("output_space", "raw"),
            operation=values.get("operation"),
            parameters=_mapping(values.get("parameters"), name="normalization parameters"),
            stats_ref=values.get("stats_ref"),
            stats_hash=values.get("stats_hash"),
        )


@dataclass(frozen=True)
class NormalizationContract:
    """Complete, hashable numeric contract for one policy artifact/boundary."""

    state: NormalizationDomainSpec = field(default_factory=NormalizationDomainSpec)
    action: NormalizationDomainSpec = field(default_factory=NormalizationDomainSpec)
    image: NormalizationDomainSpec = field(default_factory=NormalizationDomainSpec)

    def __post_init__(self) -> None:
        for name in NORMALIZATION_DOMAINS:
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = NormalizationDomainSpec.from_dict(value)
            elif not isinstance(value, NormalizationDomainSpec):
                raise TypeError(f"normalization {name} must be a NormalizationDomainSpec or mapping.")
            object.__setattr__(self, name, value)

    @classmethod
    def identity(cls) -> "NormalizationContract":
        """Return an explicit no-op contract for already-runtime values."""

        return cls()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NormalizationContract":
        values = _mapping(payload, name="normalization_contract")
        unknown = sorted(set(values).difference({"schema", "domains"}))
        if unknown:
            raise NormalizationContractError(
                "normalization_contract contains unknown entries: " + ", ".join(unknown)
            )
        schema = str(values.get("schema", "")).strip()
        if schema != NORMALIZATION_CONTRACT_SCHEMA_V1:
            raise NormalizationContractError(
                f"Unsupported normalization contract schema {schema!r}; expected "
                f"{NORMALIZATION_CONTRACT_SCHEMA_V1!r}."
            )
        domains = _mapping(values.get("domains"), name="normalization domains")
        missing = [name for name in NORMALIZATION_DOMAINS if name not in domains]
        if missing:
            raise NormalizationContractError(
                "normalization_contract.domains is missing: " + ", ".join(missing)
            )
        unknown = sorted(set(domains).difference(NORMALIZATION_DOMAINS))
        if unknown:
            raise NormalizationContractError(
                "normalization_contract.domains contains unknown entries: " + ", ".join(unknown)
            )
        return cls(
            state=NormalizationDomainSpec.from_dict(domains["state"]),
            action=NormalizationDomainSpec.from_dict(domains["action"]),
            image=NormalizationDomainSpec.from_dict(domains["image"]),
        )

    @classmethod
    def from_optional_mapping(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        allow_legacy: bool = True,
    ) -> "NormalizationContract | None":
        """Parse an explicit contract, returning ``None`` for opaque legacy data.

        Legacy payloads are intentionally not interpreted.  Callers that need
        an executable numeric transform must pass ``allow_legacy=False`` and
        fail closed instead of guessing what ``normalized`` meant.
        """

        if payload is None:
            return NormalizationContract.identity()
        values = _mapping(payload, name="normalization_contract")
        schema = str(values.get("schema", "")).strip()
        if schema == NORMALIZATION_CONTRACT_SCHEMA_V1:
            return cls.from_dict(values)
        if allow_legacy and schema in LEGACY_NORMALIZATION_SCHEMAS:
            return None
        raise NormalizationContractError(
            f"Normalization contract must use {NORMALIZATION_CONTRACT_SCHEMA_V1!r}; "
            f"legacy/unknown schema {schema!r} is not executable."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": NORMALIZATION_CONTRACT_SCHEMA_V1,
            "domains": {
                "state": self.state.to_dict(),
                "action": self.action.to_dict(),
                "image": self.image.to_dict(),
            },
        }

    @property
    def contract_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=NORMALIZATION_CONTRACT_HASH_DOMAIN_V1,
        )

    @property
    def has_client_owned_transform(self) -> bool:
        return any(getattr(self, name).is_client_owned for name in NORMALIZATION_DOMAINS)

    @property
    def has_server_owned_transform(self) -> bool:
        return any(getattr(self, name).is_server_owned for name in NORMALIZATION_DOMAINS)


def explicit_identity_normalization_contract() -> dict[str, Any]:
    """Return a fresh JSON-safe identity contract mapping."""

    return NormalizationContract.identity().to_dict()


def normalization_stats_hash(data: bytes | bytearray | memoryview) -> str:
    """Return the content identity used by a statistics artifact reference.

    The hash is over the exact artifact bytes, not a re-serialized JSON value.
    This keeps formatting changes visible and lets a server bind the same
    checkpoint asset without assuming a particular JSON canonicalization.
    """

    return "sha256:" + hashlib.sha256(bytes(data)).hexdigest()


def normalization_contract_hash(payload: Mapping[str, Any]) -> str:
    """Hash an explicit contract or an opaque legacy payload without inference."""

    parsed = NormalizationContract.from_optional_mapping(payload, allow_legacy=True)
    if parsed is not None:
        return parsed.contract_hash
    return canonical_content_hash(payload, domain=NORMALIZATION_CONTRACT_HASH_DOMAIN_V1)


__all__ = [
    "LEGACY_NORMALIZATION_SCHEMAS",
    "NORMALIZATION_CONTRACT_HASH_DOMAIN_V1",
    "NORMALIZATION_CONTRACT_SCHEMA_V1",
    "NORMALIZATION_DOMAINS",
    "NORMALIZATION_METHODS",
    "NORMALIZATION_OPERATIONS",
    "NORMALIZATION_OWNERS",
    "NormalizationContract",
    "NormalizationContractError",
    "NormalizationDomainSpec",
    "explicit_identity_normalization_contract",
    "normalization_stats_hash",
    "normalization_contract_hash",
]
