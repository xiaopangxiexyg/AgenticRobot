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


def test_motion_side_is_explicit_fixed_and_defaults_to_right():
    parser = build_parser()
    required = ["--socket-path", "/unused/socket", "--task-config", "/unused/task.json"]
    assert parser.parse_args(required).motion_side == "right"
    assert parser.parse_args([*required, "--motion-side", "left"]).motion_side == "left"
    assert not parser.parse_args(required).enable_object_follow_observation
    assert not parser.parse_args(required).enable_rgbd_capture_v3
    assert parser.parse_args([*required, "--enable-rgbd-capture-v3"]).enable_rgbd_capture_v3
    assert parser.parse_args([*required, "--enable-object-follow-observation"]).enable_object_follow_observation
    with pytest.raises(SystemExit) as captured:
        parser.parse_args([*required, "--motion-side", "both"])
    assert captured.value.code == 2


@pytest.mark.parametrize("unsupported", [
    ["--enable-grippers"],
    ["--enable-fast-grasp"],
    ["--enable-grasp-approach"],
    ["--enable-spatial-goals"],
    ["--enable-rgbd-capture-v3"],
    ["--enable-segmentation-batch-v2"],
    ["--enable-segmentation-geometry-v3"],
    ["--history-root", "/unused/history"],
    ["--segmentation-asset-bundle", "/unused/segmentation"],
    ["--grasp-contact-target-body", "cube"],
])
def test_left_unsupported_combinations_fail_before_resources(monkeypatch, capsys, unsupported):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported left combination must fail before path or resource work")

    monkeypatch.setattr(gateway_cli, "_absolute_existing_file", forbidden)
    monkeypatch.setattr(gateway_cli, "build_mujoco_agent_tcp_pose_gateway_session", forbidden)
    with pytest.raises(SystemExit) as captured:
        gateway_cli.main([
            "--socket-path", "/unused/socket", "--task-config", "/unused/task.json",
            "--motion-side", "left", *unsupported,
        ])
    assert captured.value.code == 2
    assert "baseline TCP/state Routes only" in capsys.readouterr().err


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


@pytest.mark.parametrize("flags", [
    [], ["--enable-fast-grasp"],
    ["--graspgenx-source-root", "/unused/graspgenx"],
])
def test_fast_grasp_is_explicit_and_requires_its_complete_resource_graph(flags):
    parser = build_parser()
    args = parser.parse_args([
        "--socket-path", "/unused/socket", "--task-config", "/unused/task.json", *flags,
    ])
    if not flags:
        assert gateway_cli._fast_grasp_model_from_args(parser, args) is None
    else:
        with pytest.raises(SystemExit) as captured:
            gateway_cli._fast_grasp_model_from_args(parser, args)
        assert captured.value.code == 2


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


def test_tcp_pose_comparison_cli_is_explicitly_optional():
    parser = build_parser()
    assert parser.get_default("enable_tcp_pose_comparison") is False
    action = next(item for item in parser._actions if item.dest == "enable_tcp_pose_comparison")
    assert action.option_strings == ["--enable-tcp-pose-comparison"]
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
    assert parser.parse_args(common).enable_grasp_approach is False
    assert parser.parse_args(common).grasp_contact_target_body is None
    assert parser.parse_args([*common, "--grasp-contact-target-body", "debug_cube"]).grasp_contact_target_body == "debug_cube"
    assert parser.parse_args([*common, "--enable-grasp-approach"]).enable_grasp_approach is True
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


def test_safe_motion_profile_is_explicit_and_rejected_before_startup(capsys):
    default = build_parser().get_default("safe_tcp_motion_profile")
    assert default == "local_step"
    with pytest.raises(SystemExit) as error:
        gateway_cli.main(["--socket-path", "/not-used/gateway.sock", "--task-config", "/not-used/task.json",
                          "--client-id", "test", "--client-uid", str(os.getuid()),
                          "--safe-tcp-motion-profile", "object_relation"])
    assert error.value.code == 2
    assert "--safe-tcp-motion-profile requires --enable-safe-tcp-pose" in capsys.readouterr().err


def test_object_query_is_default_off_and_requires_safe_geometry(capsys):
    assert build_parser().get_default("enable_object_goal_queries") is False
    with pytest.raises(SystemExit) as error:
        gateway_cli.main(["--socket-path", "/not-used/gateway.sock", "--task-config", "/not-used/task.json",
                          "--client-id", "test", "--client-uid", str(os.getuid()), "--enable-object-goal-queries"])
    assert error.value.code == 2
    assert "requires --enable-safe-tcp-pose" in capsys.readouterr().err


def test_surface_anchor_v2_is_opt_in_and_rejected_before_resource_startup(capsys):
    assert build_parser().get_default("enable_surface_anchor_v2") is False
    with pytest.raises(SystemExit) as error:
        gateway_cli.main(["--socket-path", "/not-used/gateway.sock", "--task-config", "/not-used/task.json",
                          "--client-id", "test", "--client-uid", str(os.getuid()), "--enable-surface-anchor-v2"])
    assert error.value.code == 2
    assert "--enable-surface-anchor-v2 requires" in capsys.readouterr().err


def test_surface_selection_profile_is_default_preserving_and_requires_v2_before_startup(capsys):
    assert build_parser().get_default("surface_anchor_selection_policy") == "prompt_nearest"
    with pytest.raises(SystemExit) as error:
        gateway_cli.main(["--socket-path", "/not-used/gateway.sock", "--task-config", "/not-used/task.json",
                          "--client-id", "test", "--client-uid", str(os.getuid()),
                          "--surface-anchor-selection-policy", "max_clearance_medoid"])
    assert error.value.code == 2
    assert "requires --enable-surface-anchor-v2" in capsys.readouterr().err


@pytest.mark.parametrize("policy", ["max_clearance_medoid", "unknown", True, None, []])
def test_surface_selection_policy_is_checked_at_both_direct_assembly_boundaries(policy):
    from agent_gateway.profiles.mujoco.motion_deployment import (
        build_mujoco_agent_tcp_pose_contract_bundle,
    )
    from agent_gateway.profiles.mujoco.motion_factory import (
        build_mujoco_agent_tcp_pose_gateway_session,
    )

    with pytest.raises(ValueError, match="selection policy"):
        build_mujoco_agent_tcp_pose_contract_bundle(application=None,
            surface_anchor_selection_policy=policy)
    with pytest.raises(ValueError, match="selection policy"):
        build_mujoco_agent_tcp_pose_gateway_session(target_config=None,
            allowed_client_ids=("test",), surface_anchor_selection_policy=policy)


def test_relation_query_is_default_off_and_requires_authenticated_surface_anchors(capsys):
    assert build_parser().get_default("enable_relation_goal_queries") is False
    with pytest.raises(SystemExit) as error:
        gateway_cli.main(["--socket-path", "/not-used/gateway.sock", "--task-config", "/not-used/task.json",
                          "--client-id", "test", "--client-uid", str(os.getuid()),
                          "--enable-relation-goal-queries"])
    assert error.value.code == 2
    assert "--enable-relation-goal-queries requires --enable-surface-anchors" in capsys.readouterr().err


@pytest.mark.parametrize("enabled", [True, "yes", 1])
def test_relation_query_factory_rejects_invalid_dependency_before_resources(enabled):
    from agent_gateway.profiles.mujoco.motion_factory import (
        build_mujoco_agent_tcp_pose_gateway_session,
    )

    with pytest.raises(ValueError, match="Relation goal queries require"):
        build_mujoco_agent_tcp_pose_gateway_session(target_config=None,
            allowed_client_ids=("test",), enable_relation_goal_queries=enabled)


@pytest.mark.parametrize("paired_flags,retention,message", [
    ([], 3, "requires v2 anchors, RGB-D v3 and segmentation v3"),
    (["--enable-rgbd-capture-v3", "--enable-segmentation-geometry-v3"], 2,
     "requires capture, segmentation and anchor retention"),
])
def test_surface_anchor_v3_dependencies_fail_before_resource_startup(paired_flags, retention, message, capsys):
    with pytest.raises(SystemExit) as error:
        gateway_cli.main(["--socket-path", "/not-used/gateway.sock", "--task-config", "/not-used/task.json",
            "--enable-safe-tcp-pose", "--enable-object-goal-queries", "--enable-surface-anchors",
            "--enable-surface-anchor-v2", "--enable-surface-anchor-v3",
            "--artifact-root", "/not-used/artifacts", "--segmentation-asset-bundle", "/not-used/model",
            "--max-retained-operations", str(retention), *paired_flags])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
