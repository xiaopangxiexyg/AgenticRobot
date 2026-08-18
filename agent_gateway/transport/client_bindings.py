"""Validated local identities for one full and optional read-only clients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType


def _client_id(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty trimmed text.")
    if len(value) > 128:
        raise ValueError(f"{field_name} exceeds 128 characters.")
    return value


def _uid(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a non-negative integer.") from error
    if parsed < 0 or str(parsed) != str(value).strip():
        raise ValueError(f"{field_name} must be a canonical non-negative integer.")
    return parsed


@dataclass(frozen=True)
class AgentGatewayUnixClientBindings:
    """One trusted local client and distinct SSH read-only identities."""

    client_id_by_uid: Mapping[int, str]
    read_only_client_ids: frozenset[str]
    socket_mode: int

    def __post_init__(self) -> None:
        values = dict(self.client_id_by_uid)
        if not values:
            raise ValueError("client_id_by_uid must not be empty.")
        if len(set(values.values())) != len(values):
            raise ValueError("Each client_id must map to one distinct UID.")
        if not self.read_only_client_ids < frozenset(values.values()):
            raise ValueError(
                "read_only_client_ids must be a strict subset of bound clients."
            )
        expected_mode = 0o660 if self.read_only_client_ids else 0o600
        if self.socket_mode != expected_mode:
            raise ValueError("socket_mode does not match the client binding set.")
        object.__setattr__(self, "client_id_by_uid", MappingProxyType(values))

    @property
    def allowed_client_ids(self) -> tuple[str, ...]:
        return tuple(self.client_id_by_uid.values())


def compile_agent_gateway_unix_client_bindings(
    *,
    primary_client_id: str,
    primary_client_uid: int,
    ssh_read_only_clients: Sequence[Sequence[object]] = (),
) -> AgentGatewayUnixClientBindings:
    """Compile CLI pairs without changing the primary client's authority."""

    primary_id = _client_id(primary_client_id, field_name="primary_client_id")
    primary_uid = _uid(primary_client_uid, field_name="primary_client_uid")
    if isinstance(ssh_read_only_clients, (str, bytes)) or not isinstance(
        ssh_read_only_clients,
        Sequence,
    ):
        raise TypeError("ssh_read_only_clients must be a sequence of pairs.")
    client_id_by_uid = {primary_uid: primary_id}
    read_only_ids: set[str] = set()
    for index, pair in enumerate(ssh_read_only_clients):
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence):
            raise ValueError(f"ssh_read_only_clients[{index}] must be a pair.")
        values = tuple(pair)
        if len(values) != 2:
            raise ValueError(f"ssh_read_only_clients[{index}] must be CLIENT_ID UID.")
        client_id = _client_id(
            values[0],
            field_name=f"ssh_read_only_clients[{index}].client_id",
        )
        uid = _uid(
            values[1],
            field_name=f"ssh_read_only_clients[{index}].uid",
        )
        if client_id in client_id_by_uid.values():
            raise ValueError("Client IDs must be unique across all bindings.")
        if uid in client_id_by_uid:
            raise ValueError("Unix UIDs must be unique across all bindings.")
        client_id_by_uid[uid] = client_id
        read_only_ids.add(client_id)
    return AgentGatewayUnixClientBindings(
        client_id_by_uid=client_id_by_uid,
        read_only_client_ids=frozenset(read_only_ids),
        socket_mode=0o660 if read_only_ids else 0o600,
    )


__all__ = [
    "AgentGatewayUnixClientBindings",
    "compile_agent_gateway_unix_client_bindings",
]
