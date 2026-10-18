from __future__ import annotations

import argparse
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.mujoco import gateway_cli
from agent_gateway.profiles.mujoco.gateway_cli import (
    _absolute_managed_directory,
    _close_viewer_for_stopped_payload,
    _prepare_gateway_before_ready,
    _resolved_runtime_session_id,
    _stopped_payload,
    build_parser,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_bilateral_grippers_refuse_realtime_physics_before_session_construction(monkeypatch, tmp_path, capsys):
    from pathlib import Path

    def forbidden(**kwargs):
        pytest.fail("Conflicting physics ownership must fail before session construction")

    monkeypatch.setattr(gateway_cli, "build_mujoco_agent_tcp_pose_gateway_session", forbidden)
    task = Path(__file__).resolve().parents[1] / "sim/configs/tasks/rm65_right_cube_reach_fixed_pose_xyz_proprio_v1.json"
    args = ["--socket-path", str(tmp_path / "gateway.sock"), "--task-config", str(task)]
    assert not build_parser().parse_args(args).enable_grippers
    with pytest.raises(SystemExit) as captured:
        gateway_cli.main([*args, "--enable-grippers"])
    assert captured.value.code == 2
    assert "--enable-grippers requires --no-realtime-worker" in capsys.readouterr().err


def test_cli_import_does_not_load_optional_perception_models() -> None:
    script = r"""
import importlib.abc
import sys

class BlockPerceptionModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "deployments.perception" or fullname.startswith(
            "deployments.perception."
        ):
            raise AssertionError(f"optional perception model imported: {fullname}")
        return None

sys.meta_path.insert(0, BlockPerceptionModels())
import agent_gateway.profiles.mujoco.gateway_cli
assert "deployments.perception" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=os.getcwd(),
    )


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


def test_spatial_goal_cli_is_explicitly_optional():
    parser = build_parser()
    assert parser.get_default("enable_spatial_goals") is False
    action = next(item for item in parser._actions if item.dest == "enable_spatial_goals")
    assert action.option_strings == ["--enable-spatial-goals"]
    assert action.const is True


def test_never_activated_gateway_closes_safely_without_hold_claim() -> None:
    payload = _stopped_payload(
        server=_server(),
        session=_session(started=False),
    )

    assert payload["safe_shutdown"] is True
    assert payload["safe_shutdown_basis"] == "motion_never_activated"
    assert payload["lower_hold_proved"] is False
    assert payload["stop_reason"] == "server_loop_ended"
    assert payload["viewer"] == {
        "enabled": False,
        "state": "disabled",
        "running": False,
    }


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


def test_stopped_payload_preserves_bounded_viewer_exit_reason() -> None:
    payload = _stopped_payload(
        server=_server(),
        session=_session(started=False),
        stop_reason="viewer:faulted",
        viewer_status={
            "enabled": True,
            "state": "faulted",
            "running": False,
            "last_error": "RuntimeError:sync failed",
        },
    )

    assert payload["stop_reason"] == "viewer:faulted"
    assert payload["viewer"]["state"] == "faulted"
    assert payload["viewer"]["last_error"] == "RuntimeError:sync failed"


def test_viewer_is_closed_before_its_stopped_payload_status_is_captured() -> None:
    calls: list[str] = []

    class _Viewer:
        def close(self) -> None:
            calls.append("close")

        def status(self) -> dict[str, object]:
            calls.append("status")
            return {
                "enabled": True,
                "state": "closed",
                "running": False,
                "teardown_confirmed": True,
            }

    status, error = _close_viewer_for_stopped_payload(_Viewer())
    payload = _stopped_payload(
        server=_server(),
        session=_session(started=False),
        viewer_status=status,
    )

    assert calls == ["close", "status"]
    assert error is None
    assert payload["viewer"] == {
        "enabled": True,
        "state": "closed",
        "running": False,
        "teardown_confirmed": True,
    }


def test_viewer_close_failure_is_reported_before_it_is_reraised() -> None:
    failure = RuntimeError("native close failed")

    class _Viewer:
        def close(self) -> None:
            raise failure

        def status(self) -> dict[str, object]:
            return {
                "enabled": True,
                "state": "close_faulted",
                "running": False,
                "last_error": "RuntimeError:native close failed",
                "teardown_confirmed": False,
            }

    status, error = _close_viewer_for_stopped_payload(_Viewer())

    assert error is failure
    assert status is not None
    assert status["state"] == "close_faulted"
    assert status["teardown_confirmed"] is False


def test_gateway_warms_rgb_backend_in_lower_hold_before_history() -> None:
    calls: list[object] = []

    class _Application:
        started = False
        held = False
        active_operation_id = None

        def start_observation_ready(self) -> None:
            calls.append("observation_ready")
            self.started = True
            self.held = True

        def visual_source_catalog(self):
            calls.append("visual_catalog")
            return ({"source_id": "front"},)

        def capture_visual_jpeg_snapshot(self, **request):
            calls.append(("rgb_warmup", request))
            return SimpleNamespace(
                frames={
                    "front": SimpleNamespace(
                        width=16,
                        height=16,
                        data=b"warmup-jpeg",
                    )
                }
            )

    viewer = SimpleNamespace(start=lambda: calls.append("viewer"))
    history = SimpleNamespace(
        capture_once=lambda: calls.append("history_capture"),
        start=lambda: calls.append("history_start"),
    )
    session = SimpleNamespace(
        application=_Application(),
        rgb_executor=object(),
        history_recorder=history,
    )

    _prepare_gateway_before_ready(session=session, viewer=viewer)

    assert calls == [
        "viewer",
        "observation_ready",
        "visual_catalog",
        (
            "rgb_warmup",
            {"source_ids": ("front",), "width": 16, "height": 16},
        ),
        "history_capture",
        "history_start",
    ]


def test_gateway_skips_visual_warmup_when_rgb_route_is_not_advertised() -> None:
    calls: list[str] = []
    application = SimpleNamespace(
        start_observation_ready=lambda: calls.append("observation_ready")
    )
    session = SimpleNamespace(
        application=application,
        rgb_executor=None,
        history_recorder=None,
    )

    _prepare_gateway_before_ready(session=session, viewer=None)

    assert calls == ["observation_ready"]


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
    assert parser.parse_args(common).viewer is False
    assert parser.parse_args([*common, "--viewer"]).viewer is True
    assert parser.parse_args(common).max_session_admissions_per_route is None
    assert (
        parser.parse_args(
            [*common, "--max-session-admissions-per-route", "23"]
        ).max_session_admissions_per_route
        == 23
    )
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


def test_cli_accepts_distinct_repeatable_ssh_read_only_identities() -> None:
    parsed = build_parser().parse_args(
        [
            "--socket-path",
            "/absolute/gateway.sock",
            "--task-config",
            "/absolute/task.json",
            "--ssh-read-only-client",
            "rpent.remote.alice",
            "1101",
            "--ssh-read-only-client",
            "rpent.remote.bob",
            "1102",
        ]
    )

    assert parsed.ssh_read_only_client == [
        ["rpent.remote.alice", "1101"],
        ["rpent.remote.bob", "1102"],
    ]


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
