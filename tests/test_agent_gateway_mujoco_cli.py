from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_gateway.profiles.mujoco.gateway_cli import (
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
