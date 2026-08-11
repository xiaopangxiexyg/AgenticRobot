"""Run the four Prepared writing model-RTC stages in verified order.

This module is deliberately a thin lifecycle wrapper around
``prepared_skill_launch``.  Each stage remains an independent OpenPI process,
so OpenPI owns training, checkpoints, and its terminal ``S`` save shortcut.
The wrapper only decides whether a stage is fresh, resumable, or complete and
will advance only after verifying a checkpoint at the configured target step.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import signal
import subprocess
import sys
from typing import Sequence

from deployments.openpi.model_rtc_profile import (
    load_prepared_model_rtc_profile,
)

from .evidence import (
    OpenPIModelRtcCheckpointEvidence,
    verify_openpi_model_rtc_checkpoint,
)
from .prepared_skill_launch import (
    _load_checkpoint_bound_prepared_resume,
    _load_profile,
    _model_rtc_checkpoint_root,
    _numbered_checkpoint_steps,
    _openpi_cwd,
    _resolve_train_checkpoint_policy,
)


_OPENPI_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _OPENPI_INTEGRATION_ROOT / "configs"


@dataclass(frozen=True)
class WritingStage:
    number: int
    prepared_profile: Path
    model_rtc_profile: Path


@dataclass(frozen=True)
class WritingStageState:
    stage: WritingStage
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
            "prepared_profile": str(self.stage.prepared_profile),
            "model_rtc_profile": str(self.stage.model_rtc_profile),
        }


def _default_stages() -> tuple[WritingStage, ...]:
    return tuple(
        WritingStage(
            number=stage,
            prepared_profile=(
                _CONFIG_ROOT
                / "training"
                / (
                    f"writing_stage{stage}_0804_0806_front_h16_"
                    "absolute_v1.json"
                )
            ),
            model_rtc_profile=(
                _CONFIG_ROOT
                / "model_rtc"
                / (
                    f"writing_stage{stage}_0804_0806_front_h16_"
                    "absolute_prepared_model_rtc_v1.json"
                )
            ),
        )
        for stage in range(1, 5)
    )


DEFAULT_STAGES = _default_stages()


def _target_train_step(prepared_profile: dict[str, object]) -> int:
    checkout = Path(
        str(prepared_profile["openpi_checkout"])
    ).expanduser().resolve()
    if not (checkout / "src/openpi").is_dir():
        raise FileNotFoundError(f"OpenPI checkout is missing: {checkout}.")
    import openpi.training.config as training_config

    with _openpi_cwd(checkout):
        base = training_config.get_config(
            str(prepared_profile["openpi_base_config"])
        )
    target = int(base.num_train_steps)
    if target <= 0:
        raise ValueError("OpenPI num_train_steps must be positive.")
    return target


def _verify_stage_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_model_id: str,
    expected_source_id: str,
) -> OpenPIModelRtcCheckpointEvidence:
    evidence = verify_openpi_model_rtc_checkpoint(checkpoint_dir)
    if evidence.training_run.output_model_id != expected_model_id:
        raise ValueError(
            "Checkpoint output model differs from the stage profile: "
            f"{evidence.training_run.output_model_id!r} != "
            f"{expected_model_id!r}."
        )
    source_bindings = evidence.training_run.source_bindings
    if len(source_bindings) != 1 or source_bindings[0].source_id != expected_source_id:
        actual = [binding.source_id for binding in source_bindings]
        raise ValueError(
            "Checkpoint training source differs from the stage profile: "
            f"actual={actual!r} expected={[expected_source_id]!r}."
        )
    return evidence


def inspect_stage(stage: WritingStage) -> WritingStageState:
    prepared_profile = _load_profile(stage.prepared_profile)
    model_rtc_profile = load_prepared_model_rtc_profile(stage.model_rtc_profile)
    if model_rtc_profile["prepared_skill_profile_id"] != prepared_profile["profile_id"]:
        raise ValueError(
            f"Stage {stage.number} Prepared and model-RTC profiles do not match."
        )
    checkpoint_root = _model_rtc_checkpoint_root(
        prepared_profile,
        model_rtc_profile,
        exp_name=None,
    )
    target_step = _target_train_step(prepared_profile)
    positive_steps = tuple(
        step for step in _numbered_checkpoint_steps(checkpoint_root) if step > 0
    )
    if not positive_steps:
        _resolve_train_checkpoint_policy(
            checkpoint_root,
            resume=False,
            overwrite=False,
        )
        return WritingStageState(
            stage=stage,
            checkpoint_root=checkpoint_root,
            target_step=target_step,
            latest_checkpoint_step=None,
            action="fresh",
        )

    latest_step = positive_steps[-1]
    _verify_stage_checkpoint(
        checkpoint_root / str(latest_step),
        expected_model_id=str(model_rtc_profile["output_config_name"]),
        expected_source_id=str(model_rtc_profile["profile_id"]),
    )
    if latest_step < target_step:
        _load_checkpoint_bound_prepared_resume(
            prepared_profile,
            model_rtc_profile,
            checkpoint_root=checkpoint_root,
            # The child launcher repeats the full source-file verification
            # immediately before training.  Sequence inspection only needs to
            # prove that the frozen binding can be reconstructed exactly.
            verify_source_files=False,
        )
    return WritingStageState(
        stage=stage,
        checkpoint_root=checkpoint_root,
        target_step=target_step,
        latest_checkpoint_step=latest_step,
        action="complete" if latest_step >= target_step else "resume",
    )


def _launcher_argv(state: WritingStageState) -> list[str]:
    argv = [
        sys.executable,
        "-u",
        "-m",
        "deployments.openpi.training.prepared_skill_launch",
        "train",
        "--profile",
        str(state.stage.prepared_profile),
        "--model-rtc-profile",
        str(state.stage.model_rtc_profile),
        "--no-motion",
    ]
    if state.action == "resume":
        argv.append("--resume")
    return argv


def _run_stage_process(argv: Sequence[str]) -> int:
    """Run one child with terminal stdin while the parent ignores Ctrl+C.

    The child inherits the parent's original handlers at spawn time.  Ignoring
    SIGINT in the parent only while waiting lets OpenPI perform its own
    save-and-exit handling without the sequence wrapper exiting first.
    """

    process = subprocess.Popen(list(argv))
    previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return process.wait()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


def run_sequence(
    stages: Sequence[WritingStage] = DEFAULT_STAGES,
    *,
    dry_run: bool = False,
) -> int:
    if not stages:
        raise ValueError("Writing stage sequence must not be empty.")
    expected_numbers = list(range(1, len(stages) + 1))
    actual_numbers = [stage.number for stage in stages]
    if actual_numbers != expected_numbers:
        raise ValueError(
            "Writing stages must be ordered and contiguous: "
            f"actual={actual_numbers!r}."
        )

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

        return_code = _run_stage_process(_launcher_argv(before))
        if return_code != 0:
            print(
                f"stage {stage.number} training exited with code {return_code}; "
                "rerun the same sequence command after resolving the failure",
                file=sys.stderr,
                flush=True,
            )
            return return_code

        after = inspect_stage(stage)
        print(json.dumps(after.to_dict(), indent=2, sort_keys=True), flush=True)
        if after.action != "complete":
            print(
                f"stage {stage.number} stopped at checkpoint "
                f"{after.latest_checkpoint_step}; rerun the same command to resume",
                flush=True,
            )
            return 0

    if dry_run:
        print(
            "dry-run complete; no training process was started",
            flush=True,
        )
    else:
        print("all four writing model-RTC stages are complete", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resume and run writing model-RTC stages 1-4 in verified order. "
            "During training, press S to save and continue; press Ctrl+C once "
            "to use OpenPI's save-and-exit behavior, then rerun this command."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and verify stage decisions without starting training.",
    )
    args = parser.parse_args(argv)
    return run_sequence(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_STAGES",
    "WritingStage",
    "WritingStageState",
    "inspect_stage",
    "main",
    "run_sequence",
]
