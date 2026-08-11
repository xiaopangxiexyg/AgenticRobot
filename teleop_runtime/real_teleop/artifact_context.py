"""Side-effect-free runtime-artifact context for one real-teleop process.

Formal collection datasets remain owned by the dataset writer and are never
redirected here.  This context only groups optional observational/debug output
that otherwise lands in several unrelated legacy directories.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Sequence

from runtime_artifact_paths import (
    LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
    RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
    RuntimeArtifactLayout,
)


REAL_TELEOP_ARTIFACT_KIND = "real_teleop"
REAL_TELEOP_ARTIFACT_CONTEXT_SCHEMA = "dexteleop.runtime_artifact_context.v1"
REAL_TELEOP_ARTIFACT_CONTEXT_FILENAME = "artifact_context.json"


def _argv_has_option(argv: Sequence[str], option: str) -> bool:
    return any(token == option or token.startswith(option + "=") for token in argv)


@dataclass(frozen=True)
class RealTeleopArtifactContext:
    layout_name: str
    run_root: Path | None
    session_dir: Path | None
    control_debug_root: Path | None
    actuator_trace_root: Path | None
    debug_timing_root: Path | None
    action_debug_root: Path | None
    ik_debug_root: Path | None
    trajectory_root: Path | None
    fix_arm_snapshot_root: Path | None
    explicit_root_options: tuple[str, ...] = ()

    @property
    def canonical(self) -> bool:
        return self.layout_name == RUNTIME_ARTIFACT_LAYOUT_CLI_NAME

    def to_metadata(self) -> dict[str, object]:
        return {
            "layout_name": self.layout_name,
            "canonical": self.canonical,
            "run_root": None if self.run_root is None else str(self.run_root),
            "session_dir": (
                None if self.session_dir is None else str(self.session_dir)
            ),
            "control_debug_root": (
                None
                if self.control_debug_root is None
                else str(self.control_debug_root)
            ),
            "actuator_trace_root": (
                None
                if self.actuator_trace_root is None
                else str(self.actuator_trace_root)
            ),
            "debug_timing_root": (
                None
                if self.debug_timing_root is None
                else str(self.debug_timing_root)
            ),
            "action_debug_root": (
                None
                if self.action_debug_root is None
                else str(self.action_debug_root)
            ),
            "ik_debug_root": (
                None if self.ik_debug_root is None else str(self.ik_debug_root)
            ),
            "trajectory_root": (
                None if self.trajectory_root is None else str(self.trajectory_root)
            ),
            "fix_arm_snapshot_root": (
                None
                if self.fix_arm_snapshot_root is None
                else str(self.fix_arm_snapshot_root)
            ),
            "explicit_root_options": list(self.explicit_root_options),
            "formal_dataset_redirected": False,
        }


def allocate_real_teleop_artifact_context(
    *,
    layout_name: str,
    now: float | None = None,
    explicit_root_options: Sequence[str] = (),
) -> RealTeleopArtifactContext:
    selected = str(layout_name or LEGACY_ARTIFACT_LAYOUT_CLI_NAME).strip()
    explicit = tuple(sorted({str(option) for option in explicit_root_options}))
    if selected == LEGACY_ARTIFACT_LAYOUT_CLI_NAME:
        return RealTeleopArtifactContext(
            layout_name=selected,
            run_root=None,
            session_dir=None,
            control_debug_root=None,
            actuator_trace_root=None,
            debug_timing_root=None,
            action_debug_root=None,
            ik_debug_root=None,
            trajectory_root=None,
            fix_arm_snapshot_root=None,
            explicit_root_options=explicit,
        )
    if selected != RUNTIME_ARTIFACT_LAYOUT_CLI_NAME:
        raise ValueError(f"unsupported artifact layout: {selected!r}")

    timestamp_time = time.time() if now is None else float(now)
    local_time = time.localtime(timestamp_time)
    date_partition = time.strftime("%Y-%m-%d", local_time)
    stamp = time.strftime("%Y%m%d_%H%M%S", local_time)
    micros = int((timestamp_time % 1.0) * 1_000_000.0)
    run_root = RuntimeArtifactLayout.configured().run_dir(
        kind=REAL_TELEOP_ARTIFACT_KIND,
        date_partition=date_partition,
        run_id=f"{stamp}_{micros:06d}_vive_rm65_real_teleop",
    )
    diagnostics = run_root / "diagnostics"
    return RealTeleopArtifactContext(
        layout_name=selected,
        run_root=run_root,
        session_dir=run_root / "session",
        control_debug_root=diagnostics / "control_debug",
        actuator_trace_root=diagnostics / "actuator_trace",
        debug_timing_root=diagnostics / "timing",
        action_debug_root=diagnostics / "action",
        ik_debug_root=diagnostics / "ik",
        trajectory_root=run_root / "trajectories",
        fix_arm_snapshot_root=run_root / "snapshots" / "fix_arm_pose",
        explicit_root_options=explicit,
    )


def configure_real_teleop_artifact_paths(
    args: argparse.Namespace,
    *,
    raw_argv: Sequence[str],
    now: float | None = None,
) -> RealTeleopArtifactContext:
    """Apply canonical defaults while preserving every explicit sink root."""

    root_options = (
        "--session-record-root",
        "--control-debug-root",
        "--actuator-trace-root",
        "--debug-action-log-root",
        "--episode-root",
    )
    explicit = tuple(
        option for option in root_options if _argv_has_option(raw_argv, option)
    )
    context = allocate_real_teleop_artifact_context(
        layout_name=str(
            getattr(args, "artifact_layout", LEGACY_ARTIFACT_LAYOUT_CLI_NAME)
        ),
        now=now,
        explicit_root_options=explicit,
    )

    for attribute in (
        "_runtime_artifact_session_dir",
        "_runtime_artifact_debug_timing_root",
        "_runtime_artifact_ik_debug_root",
        "_runtime_artifact_trajectory_root",
        "_runtime_artifact_fix_arm_snapshot_root",
    ):
        setattr(args, attribute, "")

    if context.canonical:
        assert context.session_dir is not None
        assert context.control_debug_root is not None
        assert context.actuator_trace_root is not None
        assert context.debug_timing_root is not None
        assert context.action_debug_root is not None
        assert context.ik_debug_root is not None
        assert context.trajectory_root is not None
        assert context.fix_arm_snapshot_root is not None
        if "--session-record-root" not in explicit:
            args._runtime_artifact_session_dir = str(context.session_dir)
        if "--control-debug-root" not in explicit:
            args.control_debug_root = str(context.control_debug_root)
        if "--actuator-trace-root" not in explicit:
            args.actuator_trace_root = str(context.actuator_trace_root)
        if "--debug-action-log-root" not in explicit:
            args.debug_action_log_root = str(context.action_debug_root)
        args._runtime_artifact_debug_timing_root = str(
            context.debug_timing_root
        )
        args._runtime_artifact_ik_debug_root = str(context.ik_debug_root)
        if not {
            "--episode-root",
            "--session-record-root",
        }.intersection(explicit):
            args._runtime_artifact_trajectory_root = str(context.trajectory_root)
        args._runtime_artifact_fix_arm_snapshot_root = str(
            context.fix_arm_snapshot_root
        )

    context_metadata = context.to_metadata()
    context_metadata["effective_root_settings"] = {
        "session_record_root": str(
            getattr(args, "session_record_root", "") or ""
        ),
        "runtime_artifact_session_dir": str(
            getattr(args, "_runtime_artifact_session_dir", "") or ""
        ),
        "control_debug_root": str(
            getattr(args, "control_debug_root", "") or ""
        ),
        "actuator_trace_root": str(
            getattr(args, "actuator_trace_root", "") or ""
        ),
        "debug_action_log_root": str(
            getattr(args, "debug_action_log_root", "") or ""
        ),
        "debug_timing_root": str(
            getattr(args, "_runtime_artifact_debug_timing_root", "") or ""
        ),
        "ik_debug_root": str(
            getattr(args, "_runtime_artifact_ik_debug_root", "") or ""
        ),
        "trajectory_root": str(
            getattr(args, "_runtime_artifact_trajectory_root", "") or ""
        ),
        "fix_arm_snapshot_root": str(
            getattr(args, "_runtime_artifact_fix_arm_snapshot_root", "") or ""
        ),
        "episode_root": str(getattr(args, "episode_root", "") or ""),
    }
    context_metadata["formal_dataset_root_unchanged"] = str(
        getattr(args, "dataset_root", "") or ""
    )
    args._runtime_artifact_context = context_metadata
    return context


def materialize_real_teleop_artifact_context(
    args: argparse.Namespace,
    *,
    now: float | None = None,
) -> Path | None:
    """Create the opted-in run anchor before any device or recorder starts."""

    selected = str(
        getattr(args, "artifact_layout", LEGACY_ARTIFACT_LAYOUT_CLI_NAME)
    ).strip()
    if selected == LEGACY_ARTIFACT_LAYOUT_CLI_NAME:
        return None
    if selected != RUNTIME_ARTIFACT_LAYOUT_CLI_NAME:
        raise ValueError(f"unsupported artifact layout: {selected!r}")
    if bool(getattr(args, "no_debug_disk_write", False)):
        return None
    metadata = getattr(args, "_runtime_artifact_context", None)
    if not isinstance(metadata, dict) or not bool(metadata.get("canonical")):
        raise RuntimeError(
            "canonical real-teleop artifact paths were not configured before runtime"
        )
    run_root_text = str(metadata.get("run_root", "") or "").strip()
    if not run_root_text:
        raise RuntimeError("canonical real-teleop artifact context has no run root")
    run_root = Path(run_root_text).expanduser().resolve()
    family_root = RuntimeArtifactLayout.configured().run_family_dir(
        kind=REAL_TELEOP_ARTIFACT_KIND
    ).resolve()
    if not run_root.is_relative_to(family_root):
        raise RuntimeError(
            f"canonical real-teleop run root escapes configured family: {run_root}"
        )
    manifest_path = run_root / REAL_TELEOP_ARTIFACT_CONTEXT_FILENAME
    if run_root.exists():
        if not manifest_path.is_file():
            raise FileExistsError(
                f"real-teleop artifact run root already exists without a manifest: {run_root}"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise RuntimeError(f"invalid artifact context manifest: {manifest_path}")
        if (
            existing.get("schema") != REAL_TELEOP_ARTIFACT_CONTEXT_SCHEMA
            or existing.get("run_root") != str(run_root)
            or int(existing.get("creator_pid", -1)) != os.getpid()
        ):
            raise RuntimeError(
                f"artifact context manifest is owned by another run: {manifest_path}"
            )
        args._runtime_artifact_manifest_path = str(manifest_path)
        return manifest_path

    run_root.mkdir(parents=True, exist_ok=False)
    created_at = time.time() if now is None else float(now)
    payload = {
        "schema": REAL_TELEOP_ARTIFACT_CONTEXT_SCHEMA,
        "created_wall_time": created_at,
        "creator_pid": os.getpid(),
        "source_entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
        "run_root": str(run_root),
        "context": metadata,
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, manifest_path)
    directory_fd = os.open(run_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    args._runtime_artifact_manifest_path = str(manifest_path)
    return manifest_path


__all__ = [
    "REAL_TELEOP_ARTIFACT_KIND",
    "REAL_TELEOP_ARTIFACT_CONTEXT_FILENAME",
    "REAL_TELEOP_ARTIFACT_CONTEXT_SCHEMA",
    "RealTeleopArtifactContext",
    "allocate_real_teleop_artifact_context",
    "configure_real_teleop_artifact_paths",
    "materialize_real_teleop_artifact_context",
]
