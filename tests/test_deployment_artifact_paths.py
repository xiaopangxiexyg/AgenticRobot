from __future__ import annotations

import argparse
from pathlib import Path
import time

import pytest

from project_paths import repo_path
from teleop_runtime.deploy import run_policy


def _args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    return parser.parse_args(argv)


def test_deploy_tcp_pose_envelope_is_default_off_and_explicitly_activated() -> None:
    defaults = _args([])
    assert defaults.tcp_pose_envelope_profile is None
    assert defaults.tcp_pose_envelope_active is False

    shadow = _args(
        ["--tcp-pose-envelope-profile", "/tmp/tcp-envelope.json"]
    )
    assert shadow.tcp_pose_envelope_profile == Path("/tmp/tcp-envelope.json")
    assert shadow.tcp_pose_envelope_active is False

    active = _args(
        [
            "--tcp-pose-envelope-profile",
            "/tmp/tcp-envelope.json",
            "--tcp-pose-envelope-active",
        ]
    )
    assert active.tcp_pose_envelope_active is True


@pytest.mark.parametrize(
    "argv, message",
    [
        (
            ["--tcp-pose-envelope-active"],
            "requires --tcp-pose-envelope-profile",
        ),
        (
            [
                "--no-arms",
                "--tcp-pose-envelope-profile",
                "/tmp/tcp-envelope.json",
            ],
            "require enabled arms in position or pose-impedance mode",
        ),
    ],
)
def test_deploy_tcp_pose_envelope_rejects_unsafe_cli_combinations(
    argv: list[str],
    message: str,
) -> None:
    args = _args(argv)
    with pytest.raises(ValueError, match=message):
        run_policy.validate_args(args, raw_argv=argv)


def test_deploy_tcp_pose_envelope_accepts_pose_impedance_mode() -> None:
    argv = [
        "--arm-control-mode",
        "impedance",
        "--tcp-pose-envelope-profile",
        "/tmp/tcp-envelope.json",
    ]

    run_policy.validate_args(_args(argv), raw_argv=argv)


def test_deployment_debug_defaults_to_legacy_root() -> None:
    resolution = run_policy.resolve_deployment_debug_root(
        _args([]),
        now=1_786_234_923.456,
    )

    assert resolution.path == repo_path("teleop_records", "deployment_debug")
    assert resolution.source == "legacy"


def test_deployment_debug_canonical_layout_allocates_one_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = run_policy.RuntimeArtifactLayout(root=tmp_path / "canonical")
    monkeypatch.setattr(
        run_policy.RuntimeArtifactLayout,
        "configured",
        classmethod(lambda cls: configured),
    )

    now = 1_786_234_923.456
    local_time = time.localtime(now)
    date_partition = time.strftime("%Y-%m-%d", local_time)
    stamp = time.strftime("%Y%m%d_%H%M%S", local_time)
    path = run_policy.canonical_deployment_debug_root(now=now)

    assert path == (
        configured.runs_root
        / "policy_deploy"
        / date_partition
        / f"deploy_{stamp}_456"
    )
    assert not configured.root.exists()


def test_deployment_debug_canonical_layout_is_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical-run"
    monkeypatch.setattr(
        run_policy,
        "canonical_deployment_debug_root",
        lambda *, now=None: canonical,
    )

    resolution = run_policy.resolve_deployment_debug_root(
        _args(
            [
                "--artifact-layout",
                run_policy.RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
            ]
        ),
        now=1.0,
    )

    assert resolution.path == canonical
    assert resolution.source == "canonical"
    assert not canonical.exists()


def test_explicit_deployment_debug_dir_wins_over_layout(tmp_path: Path) -> None:
    explicit = tmp_path / "exact"

    resolution = run_policy.resolve_deployment_debug_root(
        _args(
            [
                "--artifact-layout",
                run_policy.RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
                "--debug-dir",
                str(explicit),
            ]
        ),
        now=1.0,
    )

    assert resolution.path == explicit
    assert resolution.source == "explicit"
