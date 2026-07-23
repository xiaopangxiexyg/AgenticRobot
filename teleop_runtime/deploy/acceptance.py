"""Read-only A/B acceptance for absolute and relative real-deploy debug logs.

This module never constructs a backend or issues a robot command.  It aligns
executed debug-replay actions by their immutable dataset source row and checks
that the relative path decoded to the same runtime-absolute target as the
absolute path.  Physical tracking metrics are reported separately because two
real executions are not bitwise deterministic.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_SIDES = ("left", "right")
_FATAL_EVENTS = {
    "policy_error",
    "cleanup_error",
    "position_executor_close_error",
    "position_executor_fault",
}


@dataclass(frozen=True)
class RealDeployAcceptanceThresholds:
    decoded_translation_m: float = 2e-6
    decoded_rotation_deg: float = 2e-3
    command_translation_m: float = 2e-3
    command_rotation_deg: float = 0.5
    measured_translation_m: float = 3e-2
    measured_rotation_deg: float = 5.0
    measured_qpos_rad: float = 0.15

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, number)


def _resolve_debug_log(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        return resolved
    if not resolved.is_dir():
        raise ValueError(f"Deploy debug path does not exist: {resolved}")
    candidates = sorted(resolved.glob("deploy_debug_*.jsonl"))
    if not candidates:
        raise ValueError(f"No deploy_debug_*.jsonl under {resolved}")
    return candidates[-1]


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"Debug event must be an object: {path}:{line_number}")
            events.append(dict(value))
    if not events:
        raise ValueError(f"Deploy debug log is empty: {path}")
    return events


def _startup(events: Sequence[Mapping[str, Any]], *, path: Path) -> Mapping[str, Any]:
    values = [event for event in events if event.get("event") == "startup"]
    if len(values) != 1:
        raise ValueError(f"Expected exactly one startup event in {path}, got {len(values)}")
    return values[0]


def _action_source_index(event: Mapping[str, Any]) -> int:
    raw_action = event.get("raw_action")
    metadata = raw_action.get("metadata") if isinstance(raw_action, Mapping) else None
    if not isinstance(metadata, Mapping):
        raise ValueError("action_step raw_action metadata is missing")
    value = metadata.get("debug_replay_source_action_index")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "action_step is missing debug_replay_source_action_index; regenerate the "
            "log with the current replay runtime"
        ) from exc


def _executed_actions(
    events: Sequence[Mapping[str, Any]], *, path: Path
) -> dict[int, Mapping[str, Any]]:
    output: dict[int, Mapping[str, Any]] = {}
    for event in events:
        if event.get("event") != "action_step":
            continue
        source_index = _action_source_index(event)
        if source_index in output:
            raise ValueError(f"Duplicate executed replay source index {source_index} in {path}")
        output[source_index] = event
    if not output:
        raise ValueError(f"No executed action_step events in {path}")
    return output


def _vector(value: Any, *, size: int) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _pose_delta(first: Any, second: Any) -> tuple[float, float] | None:
    lhs = _vector(first, size=7)
    rhs = _vector(second, size=7)
    if lhs is None or rhs is None:
        return None
    translation = float(np.linalg.norm(rhs[:3] - lhs[:3]))
    lhs_q = lhs[3:]
    rhs_q = rhs[3:]
    lhs_norm = float(np.linalg.norm(lhs_q))
    rhs_norm = float(np.linalg.norm(rhs_q))
    if lhs_norm <= 1e-12 or rhs_norm <= 1e-12:
        return None
    dot = float(np.clip(abs(np.dot(lhs_q / lhs_norm, rhs_q / rhs_norm)), 0.0, 1.0))
    rotation_deg = float(np.degrees(2.0 * np.arccos(dot)))
    return translation, rotation_deg


def _nested_pose(event: Mapping[str, Any], section: str, side: str) -> Any:
    value = event.get(section)
    if not isinstance(value, Mapping):
        return None
    poses = value.get("arm_pose_by_side")
    return poses.get(side) if isinstance(poses, Mapping) else None


def _measured_pose(event: Mapping[str, Any], side: str) -> Any:
    states = event.get("pre_arm_state_by_side")
    state = states.get(side) if isinstance(states, Mapping) else None
    return state.get("current_tcp_xyzquat") if isinstance(state, Mapping) else None


def _measured_qpos(event: Mapping[str, Any], side: str) -> Any:
    states = event.get("pre_arm_state_by_side")
    state = states.get(side) if isinstance(states, Mapping) else None
    if not isinstance(state, Mapping):
        return None
    return state.get("hardware_qpos", state.get("current_qpos"))


def _max_pose_metric(
    absolute: Mapping[int, Mapping[str, Any]],
    relative: Mapping[int, Mapping[str, Any]],
    *,
    extractor: Any,
) -> dict[str, Any]:
    by_side: dict[str, dict[str, Any]] = {}
    for side in _SIDES:
        translations: list[float] = []
        rotations: list[float] = []
        for source_index in sorted(absolute):
            delta = _pose_delta(
                extractor(absolute[source_index], side),
                extractor(relative[source_index], side),
            )
            if delta is None:
                continue
            translations.append(delta[0])
            rotations.append(delta[1])
        by_side[side] = {
            "compared_count": len(translations),
            "max_translation_m": max(translations, default=None),
            "max_rotation_deg": max(rotations, default=None),
        }
    return by_side


def _max_qpos_metric(
    absolute: Mapping[int, Mapping[str, Any]],
    relative: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    by_side: dict[str, dict[str, Any]] = {}
    for side in _SIDES:
        maxima: list[float] = []
        for source_index in sorted(absolute):
            lhs = _vector(_measured_qpos(absolute[source_index], side), size=6)
            rhs = _vector(_measured_qpos(relative[source_index], side), size=6)
            if lhs is not None and rhs is not None:
                maxima.append(float(np.max(np.abs(rhs - lhs))))
        by_side[side] = {
            "compared_count": len(maxima),
            "max_abs_joint_delta_rad": max(maxima, default=None),
        }
    return by_side


def _startup_args(startup: Mapping[str, Any]) -> Mapping[str, Any]:
    args = startup.get("args")
    return args if isinstance(args, Mapping) else {}


def _fatal_diagnostics(events: Sequence[Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for event in events:
        name = str(event.get("event") or "")
        if name in _FATAL_EVENTS or name.endswith("_fault"):
            failures.append(name)
        if name == "action_step":
            if event.get("dispatch_error") is not None:
                failures.append("action_step:dispatch_error")
            faults = event.get("faults")
            if isinstance(faults, Sequence) and not isinstance(faults, (str, bytes)):
                failures.extend(f"action_step:fault:{fault}" for fault in faults if str(fault))
        if name == "position_executor_collision_plan":
            status = str(event.get("status") or "").lower()
            if bool(event.get("hold")) or "hold" in status or "fail" in status:
                failures.append(f"{name}:{status or 'hold'}")
        if name == "position_executor_final_snapshot":
            for key in (
                "debug_event_drop_count",
                "arm_dispatch_failed_count",
                "collision_hold_count",
            ):
                try:
                    count = int(event.get(key, 0))
                except (TypeError, ValueError):
                    count = 1
                if count:
                    failures.append(f"{name}:{key}={count}")
            if event.get("fault"):
                failures.append(f"{name}:fault={event.get('fault')}")
    return failures


def _check_pose_threshold(
    checks: list[dict[str, Any]],
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    name: str,
    translation_limit: float,
    rotation_limit: float,
    required_count: int,
) -> None:
    for side in _SIDES:
        values = metrics[side]
        count = int(values["compared_count"])
        translation = values["max_translation_m"]
        rotation = values["max_rotation_deg"]
        passed = (
            count == required_count
            and translation is not None
            and rotation is not None
            and float(translation) <= translation_limit
            and float(rotation) <= rotation_limit
        )
        checks.append(
            {
                "name": f"{name}:{side}",
                "passed": bool(passed),
                "compared_count": count,
                "required_count": required_count,
                "max_translation_m": translation,
                "translation_limit_m": translation_limit,
                "max_rotation_deg": rotation,
                "rotation_limit_deg": rotation_limit,
            }
        )


def compare_real_deploy_debug_logs(
    absolute_log: str | Path,
    relative_log: str | Path,
    *,
    thresholds: RealDeployAcceptanceThresholds | None = None,
    expected_actions: int | None = None,
    require_hardware_dispatch: bool = True,
) -> dict[str, Any]:
    """Compare two completed real-deploy logs without touching hardware."""

    limits = thresholds or RealDeployAcceptanceThresholds()
    absolute_path = _resolve_debug_log(absolute_log)
    relative_path = _resolve_debug_log(relative_log)
    absolute_events = _read_events(absolute_path)
    relative_events = _read_events(relative_path)
    absolute_startup = _startup(absolute_events, path=absolute_path)
    relative_startup = _startup(relative_events, path=relative_path)
    absolute_args = _startup_args(absolute_startup)
    relative_args = _startup_args(relative_startup)
    absolute_actions = _executed_actions(absolute_events, path=absolute_path)
    relative_actions = _executed_actions(relative_events, path=relative_path)
    absolute_indices = tuple(sorted(absolute_actions))
    relative_indices = tuple(sorted(relative_actions))
    common_indices = tuple(sorted(set(absolute_indices) & set(relative_indices)))

    comparable_keys = (
        "debug_replay_dataset",
        "debug_replay_parquet",
        "debug_replay_episode",
        "debug_replay_start_index",
        "debug_replay_end_index",
        "debug_replay_chunk_size",
        "arm_control_mode",
        "active_sides",
        "end_effector_sides",
    )
    startup_mismatches = {
        key: {"absolute": absolute_args.get(key), "relative": relative_args.get(key)}
        for key in comparable_keys
        if absolute_args.get(key) != relative_args.get(key)
    }
    checks: list[dict[str, Any]] = [
        {
            "name": "startup_modes",
            "passed": (
                absolute_args.get("policy_action_pose_mode") == "absolute"
                and absolute_args.get("debug_replay_output_pose_mode") == "absolute"
                and relative_args.get("policy_action_pose_mode") == "relative"
                and relative_args.get("debug_replay_output_pose_mode") == "relative"
            ),
        },
        {
            "name": "startup_inputs_match",
            "passed": not startup_mismatches,
            "mismatches": startup_mismatches,
        },
        {
            "name": "hardware_dispatch_enabled",
            "passed": (
                True
                if not require_hardware_dispatch
                else bool(absolute_startup.get("send_actions"))
                and bool(relative_startup.get("send_actions"))
            ),
        },
        {
            "name": "source_action_indices_match",
            "passed": absolute_indices == relative_indices,
            "absolute_indices": absolute_indices,
            "relative_indices": relative_indices,
        },
        {
            "name": "expected_action_count",
            "passed": (
                len(common_indices) > 0
                and (expected_actions is None or len(common_indices) == int(expected_actions))
            ),
            "actual": len(common_indices),
            "expected": expected_actions,
        },
    ]
    if absolute_indices == relative_indices and common_indices:
        decoded = _max_pose_metric(
            absolute_actions,
            relative_actions,
            extractor=lambda event, side: _nested_pose(event, "raw_action", side),
        )
        command = _max_pose_metric(
            absolute_actions,
            relative_actions,
            extractor=lambda event, side: _nested_pose(event, "command", side),
        )
        measured = _max_pose_metric(
            absolute_actions,
            relative_actions,
            extractor=_measured_pose,
        )
        measured_qpos = _max_qpos_metric(absolute_actions, relative_actions)
        required_count = len(common_indices)
        _check_pose_threshold(
            checks,
            decoded,
            name="decoded_runtime_absolute_target",
            translation_limit=limits.decoded_translation_m,
            rotation_limit=limits.decoded_rotation_deg,
            required_count=required_count,
        )
        _check_pose_threshold(
            checks,
            command,
            name="canonical_command_target",
            translation_limit=limits.command_translation_m,
            rotation_limit=limits.command_rotation_deg,
            required_count=required_count,
        )
        _check_pose_threshold(
            checks,
            measured,
            name="measured_tcp_trajectory",
            translation_limit=limits.measured_translation_m,
            rotation_limit=limits.measured_rotation_deg,
            required_count=required_count,
        )
        for side in _SIDES:
            values = measured_qpos[side]
            checks.append(
                {
                    "name": f"measured_qpos_trajectory:{side}",
                    "passed": (
                        int(values["compared_count"]) == required_count
                        and values["max_abs_joint_delta_rad"] is not None
                        and float(values["max_abs_joint_delta_rad"])
                        <= limits.measured_qpos_rad
                    ),
                    **values,
                    "limit_rad": limits.measured_qpos_rad,
                    "required_count": required_count,
                }
            )
    else:
        decoded = command = measured = measured_qpos = {}

    absolute_failures = _fatal_diagnostics(absolute_events)
    relative_failures = _fatal_diagnostics(relative_events)
    checks.append(
        {
            "name": "no_runtime_or_dispatch_errors",
            "passed": not absolute_failures and not relative_failures,
            "absolute": absolute_failures,
            "relative": relative_failures,
        }
    )
    passed = all(bool(check.get("passed")) for check in checks)
    return {
        "schema": "dexteleop_v23_real_deploy_ab_acceptance_v1",
        "passed": passed,
        "absolute_log": str(absolute_path),
        "relative_log": str(relative_path),
        "thresholds": asdict(limits),
        "source_action_indices": common_indices,
        "checks": checks,
        "metrics": {
            "decoded_runtime_absolute_target": decoded,
            "canonical_command_target": command,
            "measured_tcp_trajectory": measured,
            "measured_qpos_trajectory": measured_qpos,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only A/B acceptance for absolute/relative deploy debug logs."
    )
    parser.add_argument("absolute_log", type=Path)
    parser.add_argument("relative_log", type=Path)
    parser.add_argument("--expected-actions", type=int, default=None)
    parser.add_argument("--allow-dry-run", action="store_true")
    parser.add_argument("--decoded-translation-m", type=float, default=2e-6)
    parser.add_argument("--decoded-rotation-deg", type=float, default=2e-3)
    parser.add_argument("--command-translation-m", type=float, default=2e-3)
    parser.add_argument("--command-rotation-deg", type=float, default=0.5)
    parser.add_argument("--measured-translation-m", type=float, default=3e-2)
    parser.add_argument("--measured-rotation-deg", type=float, default=5.0)
    parser.add_argument("--measured-qpos-rad", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = compare_real_deploy_debug_logs(
            args.absolute_log,
            args.relative_log,
            thresholds=RealDeployAcceptanceThresholds(
                decoded_translation_m=args.decoded_translation_m,
                decoded_rotation_deg=args.decoded_rotation_deg,
                command_translation_m=args.command_translation_m,
                command_rotation_deg=args.command_rotation_deg,
                measured_translation_m=args.measured_translation_m,
                measured_rotation_deg=args.measured_rotation_deg,
                measured_qpos_rad=args.measured_qpos_rad,
            ),
            expected_actions=args.expected_actions,
            require_hardware_dispatch=not bool(args.allow_dry_run),
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RealDeployAcceptanceThresholds",
    "compare_real_deploy_debug_logs",
    "main",
]
