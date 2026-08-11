from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from runtime_artifact_paths import (
    LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
    RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
)
from teleop_runtime.real_teleop.artifact_context import (
    allocate_real_teleop_artifact_context,
    configure_real_teleop_artifact_paths,
    materialize_real_teleop_artifact_context,
)
from teleop_runtime.real_teleop.cli import build_real_teleop_argument_parser
from teleop_runtime.real_teleop.episode_runtime import _episode_root_dir
from teleop_runtime.real_teleop.session_recording import _session_record_root_dir


def _namespace(*, layout: str) -> argparse.Namespace:
    return argparse.Namespace(
        artifact_layout=layout,
        session_record_root="",
        control_debug_root="teleop_records/control_debug",
        actuator_trace_root="teleop_records/actuator_identification",
        debug_action_log_root="teleop_records/action_debug",
        episode_root="",
        dataset_root="/formal/dataset/root",
    )


def test_real_teleop_parser_keeps_legacy_layout_as_default() -> None:
    parser = build_real_teleop_argument_parser()

    legacy = parser.parse_args([])
    canonical = parser.parse_args(
        ["--artifact-layout", RUNTIME_ARTIFACT_LAYOUT_CLI_NAME]
    )

    assert legacy.artifact_layout == LEGACY_ARTIFACT_LAYOUT_CLI_NAME
    assert canonical.artifact_layout == RUNTIME_ARTIFACT_LAYOUT_CLI_NAME


def test_real_teleop_parser_uses_local_cutover_but_cli_can_roll_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from teleop_runtime.real_teleop import cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "configured_runtime_artifact_layout",
        lambda: RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
    )
    parser = cli_module.build_real_teleop_argument_parser()

    assert parser.parse_args([]).artifact_layout == RUNTIME_ARTIFACT_LAYOUT_CLI_NAME
    assert (
        parser.parse_args(
            ["--artifact-layout", LEGACY_ARTIFACT_LAYOUT_CLI_NAME]
        ).artifact_layout
        == LEGACY_ARTIFACT_LAYOUT_CLI_NAME
    )


def test_legacy_context_does_not_rewrite_or_create_any_root(tmp_path: Path) -> None:
    args = _namespace(layout=LEGACY_ARTIFACT_LAYOUT_CLI_NAME)
    before = dict(vars(args))

    context = configure_real_teleop_artifact_paths(
        args,
        raw_argv=[],
        now=1_786_234_923.456789,
    )

    assert context.canonical is False
    assert context.run_root is None
    assert args.control_debug_root == before["control_debug_root"]
    assert args.actuator_trace_root == before["actuator_trace_root"]
    assert args.debug_action_log_root == before["debug_action_log_root"]
    assert args.dataset_root == before["dataset_root"]
    assert list(tmp_path.iterdir()) == []


def test_canonical_context_groups_debug_sinks_without_redirecting_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from teleop_runtime.real_teleop import artifact_context as module

    configured = module.RuntimeArtifactLayout(root=tmp_path / "canonical")
    monkeypatch.setattr(
        module.RuntimeArtifactLayout,
        "configured",
        classmethod(lambda cls: configured),
    )
    now = 1_786_234_923.456789
    local_time = time.localtime(now)
    date_partition = time.strftime("%Y-%m-%d", local_time)
    stamp = time.strftime("%Y%m%d_%H%M%S", local_time)
    micros = int((now % 1.0) * 1_000_000.0)
    args = _namespace(layout=RUNTIME_ARTIFACT_LAYOUT_CLI_NAME)

    context = configure_real_teleop_artifact_paths(
        args,
        raw_argv=["--artifact-layout", RUNTIME_ARTIFACT_LAYOUT_CLI_NAME],
        now=now,
    )

    expected_run = (
        configured.runs_root
        / "real_teleop"
        / date_partition
        / f"{stamp}_{micros:06d}_vive_rm65_real_teleop"
    )
    assert context.run_root == expected_run
    assert _session_record_root_dir(args) == expected_run / "session"
    assert Path(args.control_debug_root).is_relative_to(expected_run)
    assert Path(args.actuator_trace_root).is_relative_to(expected_run)
    assert Path(args.debug_action_log_root).is_relative_to(expected_run)
    assert Path(args._runtime_artifact_debug_timing_root).is_relative_to(
        expected_run
    )
    assert Path(args._runtime_artifact_ik_debug_root).is_relative_to(expected_run)
    assert Path(args._runtime_artifact_trajectory_root) == (
        expected_run / "trajectories"
    )
    assert _episode_root_dir(
        args,
        SimpleNamespace(session=SimpleNamespace(recorder=None)),
    ) == (expected_run / "trajectories")
    assert Path(args._runtime_artifact_fix_arm_snapshot_root).is_relative_to(
        expected_run
    )
    assert args.dataset_root == "/formal/dataset/root"
    assert args._runtime_artifact_context["formal_dataset_redirected"] is False
    assert (
        args._runtime_artifact_context["formal_dataset_root_unchanged"]
        == "/formal/dataset/root"
    )
    assert not configured.root.exists()


def test_explicit_sink_roots_override_canonical_defaults(tmp_path: Path) -> None:
    args = _namespace(layout=RUNTIME_ARTIFACT_LAYOUT_CLI_NAME)
    args.session_record_root = str(tmp_path / "session-explicit")
    args.control_debug_root = str(tmp_path / "control-explicit")
    args.actuator_trace_root = str(tmp_path / "actuator-explicit")
    args.debug_action_log_root = str(tmp_path / "action-explicit")
    args.episode_root = str(tmp_path / "trajectory-explicit")
    raw_argv = [
        "--artifact-layout",
        RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
        "--session-record-root",
        args.session_record_root,
        "--control-debug-root",
        args.control_debug_root,
        "--actuator-trace-root",
        args.actuator_trace_root,
        "--debug-action-log-root",
        args.debug_action_log_root,
        "--episode-root",
        args.episode_root,
    ]

    context = configure_real_teleop_artifact_paths(
        args,
        raw_argv=raw_argv,
        now=1_786_234_923.456789,
    )

    assert args.control_debug_root == str(tmp_path / "control-explicit")
    assert args.actuator_trace_root == str(tmp_path / "actuator-explicit")
    assert args.debug_action_log_root == str(tmp_path / "action-explicit")
    assert args._runtime_artifact_session_dir == ""
    assert args._runtime_artifact_trajectory_root == ""
    assert set(context.explicit_root_options) == {
        "--session-record-root",
        "--control-debug-root",
        "--actuator-trace-root",
        "--debug-action-log-root",
        "--episode-root",
    }
    assert _episode_root_dir(
        args,
        SimpleNamespace(session=SimpleNamespace(recorder=None)),
    ) == (tmp_path / "trajectory-explicit")
    effective = args._runtime_artifact_context["effective_root_settings"]
    assert effective["control_debug_root"] == str(
        tmp_path / "control-explicit"
    )
    assert effective["runtime_artifact_session_dir"] == ""


def test_context_allocator_rejects_unknown_layout() -> None:
    with pytest.raises(ValueError, match="unsupported artifact layout"):
        allocate_real_teleop_artifact_context(layout_name="future-layout")


def test_materialize_canonical_context_writes_one_idempotent_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from teleop_runtime.real_teleop import artifact_context as module

    configured = module.RuntimeArtifactLayout(root=tmp_path / "canonical")
    monkeypatch.setattr(
        module.RuntimeArtifactLayout,
        "configured",
        classmethod(lambda cls: configured),
    )
    args = _namespace(layout=RUNTIME_ARTIFACT_LAYOUT_CLI_NAME)
    context = configure_real_teleop_artifact_paths(
        args,
        raw_argv=["--artifact-layout", RUNTIME_ARTIFACT_LAYOUT_CLI_NAME],
        now=1_786_234_923.456789,
    )

    manifest = materialize_real_teleop_artifact_context(args, now=123.5)
    repeated = materialize_real_teleop_artifact_context(args, now=999.0)

    assert manifest is not None
    assert repeated == manifest
    assert manifest.parent == context.run_root
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema"] == "dexteleop.runtime_artifact_context.v1"
    assert payload["created_wall_time"] == 123.5
    assert payload["run_root"] == str(context.run_root)
    assert payload["context"]["formal_dataset_redirected"] is False


def test_materialize_refuses_preexisting_unowned_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from teleop_runtime.real_teleop import artifact_context as module

    configured = module.RuntimeArtifactLayout(root=tmp_path / "canonical")
    monkeypatch.setattr(
        module.RuntimeArtifactLayout,
        "configured",
        classmethod(lambda cls: configured),
    )
    args = _namespace(layout=RUNTIME_ARTIFACT_LAYOUT_CLI_NAME)
    context = configure_real_teleop_artifact_paths(
        args,
        raw_argv=["--artifact-layout", RUNTIME_ARTIFACT_LAYOUT_CLI_NAME],
        now=1_786_234_923.456789,
    )
    assert context.run_root is not None
    context.run_root.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="without a manifest"):
        materialize_real_teleop_artifact_context(args)


def test_no_debug_disk_write_does_not_materialize_canonical_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from teleop_runtime.real_teleop import artifact_context as module

    configured = module.RuntimeArtifactLayout(root=tmp_path / "canonical")
    monkeypatch.setattr(
        module.RuntimeArtifactLayout,
        "configured",
        classmethod(lambda cls: configured),
    )
    args = _namespace(layout=RUNTIME_ARTIFACT_LAYOUT_CLI_NAME)
    args.no_debug_disk_write = True
    configure_real_teleop_artifact_paths(
        args,
        raw_argv=[
            "--artifact-layout",
            RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
            "--no-debug-disk-write",
        ],
        now=1_786_234_923.456789,
    )

    assert materialize_real_teleop_artifact_context(args) is None
    assert not configured.root.exists()
