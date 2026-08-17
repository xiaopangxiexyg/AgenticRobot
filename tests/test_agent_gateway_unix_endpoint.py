from __future__ import annotations

import json
import os
import stat

import pytest

from agent_gateway import __main__ as gateway_main
from agent_gateway.transport import endpoint_cli
from agent_gateway.transport.unix_endpoint import (
    allocate_agent_gateway_unix_endpoint,
)


def test_allocator_returns_unique_private_absent_endpoints_within_budget(
    short_unix_socket_path_factory,
) -> None:
    root = short_unix_socket_path_factory("root").parent / "ag"

    first = allocate_agent_gateway_unix_endpoint(endpoint_root=root)
    second = allocate_agent_gateway_unix_endpoint(endpoint_root=root)

    assert first.socket_path != second.socket_path
    for allocation in (first, second):
        assert allocation.endpoint_dir.parent == root
        assert allocation.socket_path.parent == allocation.endpoint_dir
        assert allocation.socket_path.name == "g.sock"
        assert not os.path.lexists(allocation.socket_path)
        assert stat.S_IMODE(os.lstat(allocation.endpoint_dir).st_mode) == 0o700
        assert allocation.encoded_path_bytes <= 107


def test_allocator_rejects_symlink_or_world_writable_root(
    short_unix_socket_path_factory,
) -> None:
    parent = short_unix_socket_path_factory("root").parent
    real_root = parent / "real"
    real_root.mkdir(mode=0o700)
    link = parent / "link"
    link.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="real directory"):
        allocate_agent_gateway_unix_endpoint(endpoint_root=link)

    os.chmod(real_root, 0o707)
    with pytest.raises(ValueError, match="world-writable"):
        allocate_agent_gateway_unix_endpoint(endpoint_root=real_root)


def test_endpoint_cli_supports_shell_path_and_auditable_json(
    short_unix_socket_path_factory,
    capsys,
) -> None:
    root = short_unix_socket_path_factory("root").parent / "ag"

    assert endpoint_cli.main(["--root", str(root)]) == 0
    path_output = capsys.readouterr().out.strip()
    assert path_output.endswith("/g.sock")
    assert not os.path.lexists(path_output)

    assert endpoint_cli.main(
        ["--root", str(root), "--format", "json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["event"] == "agent_gateway_unix_endpoint_allocated"
    assert payload["socket_path"].endswith("/g.sock")
    assert payload["encoded_path_bytes"] <= payload["max_path_bytes"] == 107


def test_package_dispatches_endpoint_allocation_utility(
    short_unix_socket_path_factory,
    capsys,
) -> None:
    root = short_unix_socket_path_factory("root").parent / "ag"

    assert gateway_main.main(
        ["allocate-unix-endpoint", "--root", str(root)]
    ) == 0

    socket_path = capsys.readouterr().out.strip()
    assert socket_path.endswith("/g.sock")
    assert not os.path.lexists(socket_path)
