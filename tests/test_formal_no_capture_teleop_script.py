from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probes/run_formal_no_capture_teleop.sh"


def _fixture_checkout(tmp_path: Path, *, include_rate_fault: bool = True) -> tuple[Path, str]:
    root = tmp_path / "formal"
    args = root / "docs/real/teleop_args.sh"
    entrypoint = root / "teleop_runtime/debug/run_vive_rm65_real_teleop.py"
    tool_dir = root / "hardware/rm65/utils/configs/calibrations"
    args.parent.mkdir(parents=True)
    entrypoint.parent.mkdir(parents=True)
    tool_dir.mkdir(parents=True)
    entrypoint.write_text("# Dry-run fixture only.\n", encoding="utf-8")
    (tool_dir / "rm65_left_DMTAC_M.json").write_text("{}\n", encoding="utf-8")
    (tool_dir / "rm65_right_DMTAC_M.json").write_text("{}\n", encoding="utf-8")
    switches = (
        "--no-position-execution-qvel-feedback --minimum-control-hz 30 "
        "--control-rate-warmup-sec 1.0 --freeze-on-low-control-rate "
        "--arm-canfd-low-rate-fault-ratio 0.60 --arm-canfd-low-rate-fault-window 121"
    )
    if not include_rate_fault:
        switches = switches.replace("--arm-canfd-low-rate-fault-window 121", "")
    args.write_text(
        "NO_GRIPPER_COMMON_ARGS=(--active-sides both --hand-backend none "
        "--no-camera --no-collection --no-episode-recording)\n"
        "POSITION_FORMAL_V2_BALANCED_60HZ_JERK_V2_ARGS=("
        "--position-control-profile rm65_bimanual_position_v2_balanced_60hz_jerk_v2 "
        f"{switches})\n"
        "POSITION_COLLECTION_PRODUCTION_ARGS=(--actuator-trace --control-debug off)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(root), "-c", "user.name=Test", "-c",
            "user.email=test@example.invalid", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return root, head


def _run(root: Path, head: str, log_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash", str(SCRIPT), "--formal-root", str(root),
            "--expected-head", head, "--python", sys.executable,
            "--log-dir", str(log_dir), "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_no_capture_dry_run_uses_formal_controls_without_opening_devices(tmp_path: Path) -> None:
    root, head = _fixture_checkout(tmp_path)
    log_dir = tmp_path / "logs"

    result = _run(root, head, log_dir)

    assert result.returncode == 0, result.stderr
    assert "taskset -c 0-7" in result.stdout
    assert "--no-camera --no-collection --no-episode-recording" in result.stdout
    assert "--hand-backend none" in result.stdout
    assert "--freeze-on-low-control-rate" in result.stdout
    assert "--arm-canfd-low-rate-fault-window 121" in result.stdout
    assert "--actuator-trace" in result.stdout
    assert "tool_profile_sha256" in result.stdout
    assert "dry-run only" in result.stdout
    assert not log_dir.exists()


def test_no_capture_rejects_wrong_formal_commit(tmp_path: Path) -> None:
    root, _ = _fixture_checkout(tmp_path)

    result = _run(root, "0" * 40, tmp_path / "logs")

    assert result.returncode == 2
    assert "formal HEAD differs" in result.stderr


def test_no_capture_rejects_missing_formal_rate_guard(tmp_path: Path) -> None:
    root, head = _fixture_checkout(tmp_path, include_rate_fault=False)

    result = _run(root, head, tmp_path / "logs")

    assert result.returncode == 2
    assert "missing --arm-canfd-low-rate-fault-window" in result.stderr


def test_no_capture_rejects_staged_tool_profile_change(tmp_path: Path) -> None:
    root, head = _fixture_checkout(tmp_path)
    tool_file = root / "hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json"
    tool_file.write_text('{"changed": true}\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", str(tool_file)], check=True)

    result = _run(root, head, tmp_path / "logs")

    assert result.returncode == 2
    assert "staged changes" in result.stderr
