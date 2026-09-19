"""Run the B601 -> RM65 MuJoCo read-only follower in the foreground.

This is an operator convenience launcher, not a second control implementation.
It delegates the lifecycle to ``python -m control_station b601-rm65`` while
providing workspace-local defaults, a fresh evidence directory, and a live
terminal log.  The child inherits the operator's stdin, so the existing
single-key GUI controls remain available without a shell prompt or a background
process.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REBOT_SOURCE_ROOT = REPO_ROOT.parent / "reBotArm_control_py"
DEFAULT_CONFIG = DEFAULT_REBOT_SOURCE_ROOT / "config" / "rebotarm_rs.yaml"
DEFAULT_TASK_CONFIG = (
    REPO_ROOT / "sim" / "configs" / "tasks" / "rm65_right_cube_lift_rl_normal_v1.json"
)
DEFAULT_MAPPING_PROFILE = (
    REPO_ROOT / "docs" / "hardware" / "profiles" / "b601_rm65_mapping_150_v2.json"
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/b601-rs-commissioning")


@dataclass(frozen=True)
class LauncherPaths:
    """Resolved local paths used by one foreground run."""

    repo_root: Path
    config: Path
    rebot_source_root: Path
    task_config: Path
    mapping_profile: Path
    output_root: Path


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


def resolve_paths(args: argparse.Namespace) -> LauncherPaths:
    """Resolve defaults without importing hardware, MuJoCo, or MotorBridge."""

    return LauncherPaths(
        repo_root=REPO_ROOT,
        config=Path(args.config).expanduser(),
        rebot_source_root=Path(args.rebot_source_root).expanduser(),
        task_config=Path(args.task_config).expanduser(),
        mapping_profile=Path(args.mapping_profile).expanduser(),
        output_root=Path(args.output_root).expanduser(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Foreground, receive-only B601 master to RM65 MuJoCo following. "
            "No real motor/report command is issued by this launcher."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_env_path("REBOT_RS_CONFIG", DEFAULT_CONFIG),
        help="B601 RS YAML (default: REBOT_RS_CONFIG or the sibling reBotArm checkout)",
    )
    parser.add_argument(
        "--rebot-source-root",
        type=Path,
        default=_env_path("REBOT_SOURCE_ROOT", DEFAULT_REBOT_SOURCE_ROOT),
        help="reBotArm_control_py root (default: sibling checkout)",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=Path(
            os.environ.get("B601_RM65_TASK_CONFIG", DEFAULT_TASK_CONFIG)
        ),
        help="RM65 MuJoCo task config",
    )
    parser.add_argument(
        "--mapping-profile",
        type=Path,
        default=Path(
            os.environ.get("B601_RM65_MAPPING_PROFILE", DEFAULT_MAPPING_PROFILE)
        ),
        help="B601 -> RM65 canonical mapping profile",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get("B601_RM65_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
        ),
        help="root for fresh run directories and logs",
    )
    activation = parser.add_mutually_exclusive_group()
    activation.add_argument(
        "--auto-activate",
        dest="auto_activate",
        action="store_true",
        help="capture a fresh anchor and start following without pressing a",
    )
    activation.add_argument(
        "--manual-activate",
        dest="auto_activate",
        action="store_false",
        help="start inactive and require the existing single-key a command",
    )
    parser.set_defaults(auto_activate=True)
    parser.add_argument(
        "--keep-residual-envelope",
        action="store_true",
        help="retain the application residual translation/rotation pre-check",
    )
    parser.add_argument("--maximum-source-age-sec", type=float, default=0.25)
    parser.add_argument("--maximum-measurement-age-sec", type=float, default=0.20)
    parser.add_argument("--simulation-feedback-timeout-sec", type=float, default=0.20)
    reference_rotation = parser.add_mutually_exclusive_group()
    reference_rotation.add_argument(
        "--mapping-reference-rotation-rpy-deg",
        nargs=3,
        type=float,
        metavar=("ROLL", "PITCH", "YAW"),
        help="temporary source-base rotation override in degrees",
    )
    reference_rotation.add_argument(
        "--mapping-reference-rotation-matrix",
        nargs=9,
        type=float,
        metavar="R",
        help="temporary source-base 3x3 rotation override, row-major",
    )
    controlled_rotation = parser.add_mutually_exclusive_group()
    controlled_rotation.add_argument(
        "--mapping-controlled-rotation-rpy-deg",
        nargs=3,
        type=float,
        metavar=("ROLL", "PITCH", "YAW"),
        help="temporary controlled-tool rotation override in degrees",
    )
    controlled_rotation.add_argument(
        "--mapping-controlled-rotation-matrix",
        nargs=9,
        type=float,
        metavar="R",
        help="temporary controlled-tool 3x3 rotation override, row-major",
    )
    parser.add_argument(
        "--pause-on-exit",
        action="store_true",
        help="wait for Enter after the foreground child exits (useful for a throw-away terminal)",
    )
    return parser


def _append_option(command: list[str], name: str, value: object | None) -> None:
    if value is None:
        return
    command.append(name)
    if isinstance(value, (tuple, list)):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def build_control_station_command(
    args: argparse.Namespace,
    *,
    paths: LauncherPaths,
    run_dir: Path,
) -> list[str]:
    """Build the canonical application argv without invoking a shell."""

    command = [
        sys.executable,
        "-m",
        "control_station",
        "b601-rm65",
        "--connect-read-only",
        "--confirmation",
        "B601_RM65_READ_ONLY_CAN_SIM_TARGET",
        "--task-config",
        str(paths.task_config),
        "--mapping-profile",
        str(paths.mapping_profile),
        "--config",
        str(paths.config),
        "--rebot-source-root",
        str(paths.rebot_source_root),
        "--maximum-source-age-sec",
        str(args.maximum_source_age_sec),
        "--maximum-measurement-age-sec",
        str(args.maximum_measurement_age_sec),
        "--simulation-feedback-timeout-sec",
        str(args.simulation_feedback_timeout_sec),
        "--log",
        str(run_dir / "gui.log"),
        "--output",
        str(run_dir / "gui_result.json"),
    ]
    if args.auto_activate:
        command.append("--auto-activate")
    if not args.keep_residual_envelope:
        command.append("--disable-teleop-residual-envelope")
    _append_option(
        command,
        "--mapping-reference-rotation-rpy-deg",
        args.mapping_reference_rotation_rpy_deg,
    )
    _append_option(
        command,
        "--mapping-reference-rotation-matrix",
        args.mapping_reference_rotation_matrix,
    )
    _append_option(
        command,
        "--mapping-controlled-rotation-rpy-deg",
        args.mapping_controlled_rotation_rpy_deg,
    )
    _append_option(
        command,
        "--mapping-controlled-rotation-matrix",
        args.mapping_controlled_rotation_matrix,
    )
    return command


def _write_log(log: TextIO, text: str) -> None:
    log.write(text)
    if not text.endswith("\n"):
        log.write("\n")
    log.flush()


def _run_foreground(command: list[str], *, cwd: Path, terminal_log: Path) -> int:
    """Run one child in the foreground while mirroring its output to a log."""

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(cwd)
        if not current_pythonpath
        else f"{cwd}{os.pathsep}{current_pythonpath}"
    )
    with terminal_log.open("x", encoding="utf-8") as log:
        _write_log(log, "command: " + " ".join(command))
        _write_log(log, f"cwd: {cwd}")
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as error:
            _write_log(log, f"launcher error: {type(error).__name__}: {error}")
            print(f"[RM65] ERROR: cannot start control station: {error}", flush=True)
            return 127

        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                _write_log(log, line)
        except KeyboardInterrupt:
            _write_log(log, "launcher: operator interrupt; forwarding SIGINT")
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
        finally:
            process.stdout.close()
        return process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    paths = resolve_paths(args)
    required_paths = {
        "config": paths.config,
        "rebot-source-root": paths.rebot_source_root,
        "task-config": paths.task_config,
        "mapping-profile": paths.mapping_profile,
    }
    missing = [f"{name}={path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        parser.error("missing required local resource(s): " + ", ".join(missing))
    paths.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(prefix="rm65-mujoco-follow-", dir=paths.output_root)
    )
    terminal_log = run_dir / "terminal.log"
    command = build_control_station_command(args, paths=paths, run_dir=run_dir)

    print("[RM65] foreground receive-only MuJoCo following starts", flush=True)
    print(f"[RM65] run directory: {run_dir}", flush=True)
    print(f"[RM65] terminal log: {terminal_log}", flush=True)
    print(
        "[RM65] auto activation is enabled; use --manual-activate to start inactive",
        flush=True,
    )
    print(
        "[RM65] close the MuJoCo window or press q to return; this launcher never starts a background process",
        flush=True,
    )
    status = _run_foreground(command, cwd=paths.repo_root, terminal_log=terminal_log)
    print(f"[RM65] launcher exit code: {status}", flush=True)
    print(f"[RM65] result: {run_dir / 'gui_result.json'}", flush=True)
    print(f"[RM65] log: {run_dir / 'gui.log'}", flush=True)
    print(f"[RM65] terminal log: {terminal_log}", flush=True)
    print("[RM65] parent shell remains active", flush=True)
    if args.pause_on_exit and sys.stdin.isatty():
        try:
            input("[RM65] press Enter to return to the shell...")
        except EOFError:
            pass
    return status


if __name__ == "__main__":
    raise SystemExit(main())
