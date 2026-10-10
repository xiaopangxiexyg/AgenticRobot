"""Run four ordinary state-only writing policies in verified order."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from deployments.openpi.cpu_resources import (
    add_cpu_resource_arguments,
    bootstrap_training_cpu_resources,
    require_cpu_resource_cli,
)

if __name__ == "__main__":
    bootstrap_training_cpu_resources("deployments.openpi.training.prepared_writing_state_sequence")

from deployments.openpi.state_only import (
    OPENPI_STATE_ONLY_CAPABILITY_KEY_V1,
    OpenPIStateOnlyInputBridge,
)

from .prepared_skill_launch import (
    _load_profile,
    _numbered_checkpoint_steps,
    _openpi_cwd,
    _resolve_train_checkpoint_policy,
)


_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "training"
_MATERIAL_GPU_PROCESS_MIB = 2_048


@dataclass(frozen=True)
class WritingStateStage:
    number: int
    profile: Path


@dataclass(frozen=True)
class WritingStateStageState:
    stage: WritingStateStage
    checkpoint_root: Path
    target_step: int
    latest_checkpoint_step: int | None
    action: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.number,
            "action": self.action,
            "target_step": self.target_step,
            "latest_checkpoint_step": self.latest_checkpoint_step,
            "checkpoint_root": str(self.checkpoint_root),
            "profile": str(self.stage.profile),
        }


@dataclass(frozen=True)
class GpuComputeProcess:
    """One material NVIDIA compute allocation visible before training."""

    pid: int
    process_name: str
    used_memory_mib: int
    command: str

    def describe(self) -> str:
        return (
            f"pid={self.pid}, used_memory_mib={self.used_memory_mib}, "
            f"command={self.command!r}"
        )


DEFAULT_STAGES = tuple(
    WritingStateStage(
        number=stage,
        profile=(
            _CONFIG_ROOT
            / f"writing_stage{stage}_0804_0806_state_h16_absolute_60k_v1.json"
        ),
    )
    for stage in range(1, 5)
)


def _process_command(pid: int, fallback: str) -> str:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return fallback
    command = " ".join(
        part.decode("utf-8", errors="replace")
        for part in raw.split(b"\0")
        if part
    )
    return command or fallback


def _material_gpu_processes() -> tuple[GpuComputeProcess, ...]:
    """Return GPU users large enough to conflict with this 4090 training run."""

    argv = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "NVIDIA GPU preflight requires nvidia-smi before starting training."
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"NVIDIA GPU preflight failed: {detail}")

    current_pid = os.getpid()
    processes: list[GpuComputeProcess] = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.rsplit(",", maxsplit=2)]
        if len(values) != 3:
            raise RuntimeError(f"Unexpected nvidia-smi process row: {line!r}")
        try:
            pid = int(values[0])
            used_memory_mib = int(values[2])
        except ValueError as exc:
            raise RuntimeError(
                f"Unexpected nvidia-smi process row: {line!r}"
            ) from exc
        if pid == current_pid or used_memory_mib < _MATERIAL_GPU_PROCESS_MIB:
            continue
        process_name = values[1]
        processes.append(
            GpuComputeProcess(
                pid=pid,
                process_name=process_name,
                used_memory_mib=used_memory_mib,
                command=_process_command(pid, process_name),
            )
        )
    return tuple(processes)


def _require_training_gpu_available() -> None:
    """Fail before expensive preparation when another workload owns the GPU."""

    conflicts = _material_gpu_processes()
    if not conflicts:
        return
    details = "\n".join(f"  - {process.describe()}" for process in conflicts)
    raise RuntimeError(
        "The four-stage Pi0.5 training run requires an otherwise idle GPU, but "
        "material compute allocations are active:\n"
        f"{details}\n"
        "Stop or relocate only the workload you own, verify nvidia-smi, then "
        "rerun the same sequence command."
    )


def _checkpoint_root(profile: Mapping[str, Any]) -> Path:
    checkout = Path(str(profile["openpi_checkout"])).expanduser().resolve()
    if not (checkout / "src/openpi").is_dir():
        raise FileNotFoundError(f"OpenPI checkout is missing: {checkout}.")
    import openpi.training.config as training_config

    with _openpi_cwd(checkout):
        base = training_config.get_config(str(profile["openpi_base_config"]))
        checkpoint_base = Path(str(base.checkpoint_base_dir)).expanduser().resolve()
    return (
        checkpoint_base
        / str(profile["openpi_config_name"])
        / str(profile["default_exp_name"])
    )


def _verify_checkpoint(
    checkpoint_dir: Path,
    *,
    profile: Mapping[str, Any],
    expected_step: int,
) -> None:
    artifact_path = checkpoint_dir / "policy_artifact.json"
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"State-only checkpoint has no policy_artifact.json: {artifact_path}."
        )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid policy artifact: {artifact_path}.")
    if str(payload.get("model_id")) != str(profile["openpi_config_name"]):
        raise ValueError("State-only checkpoint model_id differs from its profile.")
    if int(payload.get("checkpoint_step", -1)) != int(expected_step):
        raise ValueError("State-only checkpoint step differs from its directory.")
    bridge = OpenPIStateOnlyInputBridge.from_profile_value(
        profile.get("openpi_observation_mode"),
        image_keys=tuple(
            str(value) for value in profile["camera_slot_to_dataset_key"].keys()
        ),
    )
    if bridge is None:
        raise ValueError("Writing state sequence requires a state-only profile.")
    actual_bridge = payload.get(OPENPI_STATE_ONLY_CAPABILITY_KEY_V1)
    if not isinstance(actual_bridge, Mapping) or dict(actual_bridge) != bridge.contract:
        raise ValueError(
            "State-only checkpoint input contract differs from its profile."
        )
    stats_path = (
        checkpoint_dir
        / "assets"
        / str(profile["openpi_asset_id"])
        / "norm_stats.json"
    )
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"State-only checkpoint normalization stats are missing: {stats_path}."
        )


def inspect_stage(stage: WritingStateStage) -> WritingStateStageState:
    profile = _load_profile(stage.profile)
    target_step = int(profile.get("openpi_num_train_steps", 0))
    if target_step != 60_000:
        raise ValueError(
            f"Writing state stage {stage.number} must target exactly 60000 steps."
        )
    checkpoint_root = _checkpoint_root(profile)
    positive_steps = tuple(
        step for step in _numbered_checkpoint_steps(checkpoint_root) if step > 0
    )
    if not positive_steps:
        _resolve_train_checkpoint_policy(
            checkpoint_root,
            resume=False,
            overwrite=False,
        )
        return WritingStateStageState(
            stage=stage,
            checkpoint_root=checkpoint_root,
            target_step=target_step,
            latest_checkpoint_step=None,
            action="fresh",
        )
    latest_step = positive_steps[-1]
    _verify_checkpoint(
        checkpoint_root / str(latest_step),
        profile=profile,
        expected_step=latest_step,
    )
    return WritingStateStageState(
        stage=stage,
        checkpoint_root=checkpoint_root,
        target_step=target_step,
        latest_checkpoint_step=latest_step,
        action="complete" if latest_step >= target_step else "resume",
    )


def _launcher_argv(state: WritingStateStageState) -> list[str]:
    argv = [
        sys.executable,
        "-u",
        "-m",
        "deployments.openpi.training.prepared_skill_launch",
        "train",
        "--profile",
        str(state.stage.profile),
    ]
    if state.action == "resume":
        argv.append("--resume")
    return argv


def _run_stage_process(argv: Sequence[str]) -> int:
    process = subprocess.Popen(list(argv))
    previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return process.wait()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


def run_sequence(
    stages: Sequence[WritingStateStage] = DEFAULT_STAGES,
    *,
    dry_run: bool = False,
) -> int:
    numbers = [stage.number for stage in stages]
    if numbers != list(range(1, len(stages) + 1)):
        raise ValueError("Writing state stages must be ordered and contiguous.")
    if not sys.stdin.isatty() and not dry_run:
        print(
            "warning: stdin is not a TTY; OpenPI's S save shortcut is unavailable",
            file=sys.stderr,
            flush=True,
        )
    for stage in stages:
        before = inspect_stage(stage)
        print(json.dumps(before.to_dict(), indent=2, sort_keys=True), flush=True)
        if dry_run or before.action == "complete":
            continue
        _require_training_gpu_available()
        return_code = _run_stage_process(_launcher_argv(before))
        if return_code != 0:
            print(
                f"stage {stage.number} exited with code {return_code}; rerun the "
                "same sequence command after resolving the failure",
                file=sys.stderr,
                flush=True,
            )
            return return_code
        after = inspect_stage(stage)
        print(json.dumps(after.to_dict(), indent=2, sort_keys=True), flush=True)
        if after.action != "complete":
            print(
                f"stage {stage.number} stopped at {after.latest_checkpoint_step}; "
                "rerun the same command to resume",
                flush=True,
            )
            return 0
    print(
        "dry-run complete; no training process was started"
        if dry_run
        else "all four ordinary state-only writing stages are complete",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run writing stages 1-4 as independent ordinary state-only OpenPI "
            "policies, each ending at step 60000."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect checkpoint decisions without starting training.",
    )
    add_cpu_resource_arguments(parser, check_only=True)
    args = parser.parse_args(argv)
    require_cpu_resource_cli(args, parser)
    return run_sequence(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_STAGES",
    "GpuComputeProcess",
    "WritingStateStage",
    "WritingStateStageState",
    "inspect_stage",
    "main",
    "run_sequence",
]
