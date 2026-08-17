from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.mujoco import gateway_cli
from agent_gateway.profiles.mujoco.gateway_cli import (
    _absolute_managed_directory,
    _resolved_runtime_session_id,
    _stopped_payload,
    build_parser,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _server(*, receipt=None):
    return SimpleNamespace(
        gateway_close_outcome=SimpleNamespace(
            accepted=receipt is not None,
            reason_code=(
                "station_hold_confirmed_no_active_operation"
                if receipt is not None
                else "station_hold_not_confirmed_no_active_operation"
            ),
            port_receipt=receipt,
        ),
        gateway_close_error=None,
    )


def _session(*, started: bool):
    return SimpleNamespace(application=SimpleNamespace(started=started))


def test_never_activated_gateway_closes_safely_without_hold_claim() -> None:
    payload = _stopped_payload(
        server=_server(),
        session=_session(started=False),
    )

    assert payload["safe_shutdown"] is True
    assert payload["safe_shutdown_basis"] == "motion_never_activated"
    assert payload["lower_hold_proved"] is False


def test_activated_gateway_requires_lower_hold_proof() -> None:
    missing = _stopped_payload(
        server=_server(),
        session=_session(started=True),
    )
    proved = _stopped_payload(
        server=_server(
            receipt=SimpleNamespace(
                succeeded=True,
                proves_motion_or_hold=True,
            )
        ),
        session=_session(started=True),
    )

    assert missing["safe_shutdown"] is False
    assert missing["safe_shutdown_basis"] == "missing_lower_hold_proof"
    assert proved["safe_shutdown"] is True
    assert proved["safe_shutdown_basis"] == "lower_hold_proved"


def test_runtime_session_identity_is_unique_by_default_and_overridable() -> None:
    parser = build_parser()
    common = [
        "--socket-path",
        "/absolute/gateway.sock",
        "--task-config",
        "/absolute/task.json",
    ]

    assert parser.parse_args(common).runtime_session_id is None
    assert parser.parse_args(common).artifact_root is None
    assert (
        parser.parse_args(
            [*common, "--runtime-session-id", "runtime:operator-selected"]
        ).runtime_session_id
        == "runtime:operator-selected"
    )
    first = _resolved_runtime_session_id(None)
    second = _resolved_runtime_session_id(None)
    assert first.startswith("dexteleop-agent-mujoco-tcp-pose:")
    assert second.startswith("dexteleop-agent-mujoco-tcp-pose:")
    assert first != second
    assert (
        _resolved_runtime_session_id("runtime:operator-selected")
        == "runtime:operator-selected"
    )


def test_artifact_root_preflight_rejects_symlink_and_world_writable(tmp_path) -> None:
    parser = argparse.ArgumentParser()
    managed = tmp_path / "managed"
    managed.mkdir(mode=0o700)
    assert (
        _absolute_managed_directory(
            parser,
            managed,
            flag="--artifact-root",
        )
        == managed
    )

    link = tmp_path / "managed-link"
    link.symlink_to(managed, target_is_directory=True)
    with pytest.raises(SystemExit):
        _absolute_managed_directory(
            parser,
            link,
            flag="--artifact-root",
        )

    os.chmod(managed, 0o707)
    with pytest.raises(SystemExit):
        _absolute_managed_directory(
            parser,
            managed,
            flag="--artifact-root",
        )


def test_cli_rejects_overlong_socket_before_simulator_assembly(
    monkeypatch,
    capsys,
) -> None:
    assembled = False

    def should_not_assemble(**_kwargs):
        nonlocal assembled
        assembled = True
        raise AssertionError("socket preflight must precede simulator assembly")

    monkeypatch.setattr(
        gateway_cli,
        "build_mujoco_agent_tcp_pose_gateway_session",
        should_not_assemble,
    )

    with pytest.raises(SystemExit) as captured:
        gateway_cli.main(
            [
                "--socket-path",
                "/" + "a" * 108,
                "--task-config",
                str(gateway_cli.repo_root() / "pyproject.toml"),
            ]
        )

    assert captured.value.code == 2
    assert assembled is False
    error = capsys.readouterr().err
    assert "--socket-path preflight failed" in error
    assert "uses 109 encoded bytes" in error
    assert "will not relocate it automatically" in error
    assert "Traceback" not in error
