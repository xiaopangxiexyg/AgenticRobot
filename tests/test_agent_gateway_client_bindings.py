from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_gateway.profiles.mujoco import motion_gateway_session as mujoco_session
from agent_gateway.profiles.real import motion_gateway_session as real_session
from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)


def test_primary_client_keeps_private_socket_mode() -> None:
    bindings = compile_agent_gateway_unix_client_bindings(
        primary_client_id="rpent.dexteleop",
        primary_client_uid=1000,
    )

    assert dict(bindings.client_id_by_uid) == {1000: "rpent.dexteleop"}
    assert bindings.allowed_client_ids == ("rpent.dexteleop",)
    assert bindings.read_only_client_ids == frozenset()
    assert bindings.socket_mode == 0o600


def test_distinct_ssh_clients_receive_distinct_lower_identities() -> None:
    bindings = compile_agent_gateway_unix_client_bindings(
        primary_client_id="rpent.dexteleop",
        primary_client_uid=1000,
        ssh_read_only_clients=(
            ("rpent.remote.alice", "1101"),
            ("rpent.remote.bob", "1102"),
        ),
    )

    assert dict(bindings.client_id_by_uid) == {
        1000: "rpent.dexteleop",
        1101: "rpent.remote.alice",
        1102: "rpent.remote.bob",
    }
    assert bindings.read_only_client_ids == frozenset(
        {"rpent.remote.alice", "rpent.remote.bob"}
    )
    assert bindings.socket_mode == 0o660


@pytest.mark.parametrize(
    "read_only_clients",
    (
        (("rpent.dexteleop", "1101"),),
        (("rpent.remote.alice", "1000"),),
        (("rpent.remote.alice", "1101"), ("rpent.remote.alice", "1102")),
        (("rpent.remote.alice", "1101"), ("rpent.remote.bob", "1101")),
        (("rpent.remote.alice", "01"),),
    ),
)
def test_client_binding_aliases_and_ambiguous_uids_fail_closed(
    read_only_clients,
) -> None:
    with pytest.raises(ValueError):
        compile_agent_gateway_unix_client_bindings(
            primary_client_id="rpent.dexteleop",
            primary_client_uid=1000,
            ssh_read_only_clients=read_only_clients,
        )


@pytest.mark.parametrize(
    ("session_module", "session_type"),
    (
        (mujoco_session, mujoco_session.MujocoAgentTcpPoseGatewaySession),
        (real_session, real_session.RealAgentPrimitiveGatewaySession),
    ),
)
def test_profile_server_binds_read_only_access_to_authenticated_identity(
    monkeypatch,
    session_module,
    session_type,
) -> None:
    session = object.__new__(session_type)
    session.service = SimpleNamespace(
        access_policy=SimpleNamespace(
            allowed_client_ids=("rpent.primary", "rpent.remote.alice")
        )
    )
    captured: dict[str, object] = {}

    def fake_server(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    def fake_adapter(*, authenticated_client_id, read_only_access, **_kwargs):
        return SimpleNamespace(
            authenticated_client_id=authenticated_client_id,
            read_only_access=read_only_access,
        )

    monkeypatch.setattr(session_module, "AgentGatewayUnixJsonServer", fake_server)
    session.wire_adapter = fake_adapter

    server = session.build_unix_json_server(
        socket_path="/short/g.sock",
        client_id_by_uid={1000: "rpent.primary", 1101: "rpent.remote.alice"},
        read_only_client_ids=("rpent.remote.alice",),
        socket_mode=0o660,
    )

    primary = server.facade_factory("rpent.primary")
    remote = server.facade_factory("rpent.remote.alice")
    assert primary.read_only_access is False
    assert remote.read_only_access is True
    assert server.socket_mode == 0o660


@pytest.mark.parametrize(
    ("session_module", "session_type"),
    (
        (mujoco_session, mujoco_session.MujocoAgentTcpPoseGatewaySession),
        (real_session, real_session.RealAgentPrimitiveGatewaySession),
    ),
)
def test_profile_server_rejects_unbound_read_only_identity(
    session_module,
    session_type,
) -> None:
    del session_module
    session = object.__new__(session_type)
    session.service = SimpleNamespace(
        access_policy=SimpleNamespace(allowed_client_ids=("rpent.primary",))
    )

    with pytest.raises(ValueError, match="must exist in the Unix peer map"):
        session.build_unix_json_server(
            socket_path="/short/g.sock",
            client_id_by_uid={1000: "rpent.primary"},
            read_only_client_ids=("rpent.remote.alice",),
        )
