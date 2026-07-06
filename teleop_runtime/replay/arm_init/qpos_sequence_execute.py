from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robot_motion.arm_init import load_qpos_by_side_file
from robot_motion.rm65.arm_init.qpos_sequence import (
    QposTestSequence,
    load_qpos_test_sequence,
)
from robot_motion.rm65.qpos_layout import normalize_sides, validate_qpos_by_side

from .qpos_execute import (
    DEFAULT_QPOS_INIT_COLLISION_HOLD_DISTANCE_M,
    DEFAULT_QPOS_INIT_COLLISION_SCENE,
    DEFAULT_QPOS_INIT_COLLISION_SLOW_DISTANCE_M,
    DEFAULT_QPOS_INIT_COLLISION_STOP_DISTANCE_M,
    DEFAULT_QPOS_INIT_JOINT_HOLD_MARGIN_DEG,
    DEFAULT_QPOS_INIT_JOINT_SLOW_MARGIN_DEG,
    DEFAULT_QPOS_INIT_MAX_JOINT_ABS_DEG,
    DEFAULT_QPOS_INIT_BRANCH_SEARCH,
    DEFAULT_QPOS_INIT_BRANCH_SEARCH_MAX_BRANCHES,
    DEFAULT_SINGLE_ARM_MAX_QPOS_DELTA_DEG,
    DEFAULT_QPOS_TARGET_PATH,
    build_rm65_qpos_init_collision_model,
    build_rm65_qpos_init_plan,
    close_rm65_arms,
    connect_rm65_arms,
    _load_branch_neutral_qpos,
    rm65_qpos_init_plan_to_dict,
    read_current_qpos_by_side,
    send_rm65_qpos_plan,
    stop_rm65_arms,
    _plan_collision_scene,
    _plan_planner_name,
    _require_plan_completed,
)
from .safety import RM65ForceGuardConfig, RM65ForceLimitExceeded


CONFIRM_TEXT = "INIT_RM65_QPOS_SEQUENCE_EXECUTE"


@dataclass(frozen=True)
class RM65QposSequenceExecuteConfig:
    sequence_path: str | Path
    active_sides: tuple[str, ...] = ("left", "right")
    current_qpos_path: str | Path | None = None
    execute: bool = False
    confirm: str = ""
    start_index: int = 0
    max_targets: int = 0
    hold_sec_between_targets: float = 0.25
    qpos_dt_sec: float = 0.05
    joint_velocity_profile: str = "rm65_init_slow"
    max_steps_per_side: int = 10000
    collision_check: bool = True
    collision_scene_profile: str = DEFAULT_QPOS_INIT_COLLISION_SCENE
    collision_slow_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_SLOW_DISTANCE_M
    collision_hold_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_HOLD_DISTANCE_M
    collision_stop_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_STOP_DISTANCE_M
    max_joint_abs_deg: float = DEFAULT_QPOS_INIT_MAX_JOINT_ABS_DEG
    joint_slow_margin_deg: float = DEFAULT_QPOS_INIT_JOINT_SLOW_MARGIN_DEG
    joint_hold_margin_deg: float = DEFAULT_QPOS_INIT_JOINT_HOLD_MARGIN_DEG
    single_arm_max_qpos_delta_deg: float = DEFAULT_SINGLE_ARM_MAX_QPOS_DELTA_DEG
    branch_search: bool = DEFAULT_QPOS_INIT_BRANCH_SEARCH
    branch_search_max_branches: int = DEFAULT_QPOS_INIT_BRANCH_SEARCH_MAX_BRANCHES
    branch_neutral_qpos_path: str | Path | None = DEFAULT_QPOS_TARGET_PATH
    trajectory_mode: int = 0
    radio: int = 0
    state_read_retries: int = 2
    force_guard: RM65ForceGuardConfig = field(default_factory=RM65ForceGuardConfig)
    tracking_check_every: int = 10
    max_tracking_error_deg: float = 30.0
    progress_every: int = 20
    output_path: str | Path | None = None
    include_per_target_plans: bool = False
    max_steps_json: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_sides", normalize_sides(tuple(self.active_sides)))
        if bool(self.execute) and str(self.confirm) != CONFIRM_TEXT:
            raise ValueError(f"real sequence execution requires --confirm {CONFIRM_TEXT}")
        if int(self.start_index) < 0:
            raise ValueError("start_index must be >= 0.")
        if int(self.max_targets) < 0:
            raise ValueError("max_targets must be >= 0.")
        if float(self.hold_sec_between_targets) < 0.0:
            raise ValueError("hold_sec_between_targets must be >= 0.")
        if float(self.qpos_dt_sec) <= 0.0 or not math.isfinite(float(self.qpos_dt_sec)):
            raise ValueError("qpos_dt_sec must be positive and finite.")
        if int(self.max_steps_per_side) < 1:
            raise ValueError("max_steps_per_side must be >= 1.")
        if float(self.collision_slow_distance_m) <= 0.0:
            raise ValueError("collision_slow_distance_m must be positive.")
        if float(self.collision_hold_distance_m) < 0.0:
            raise ValueError("collision_hold_distance_m must be >= 0.")
        if float(self.collision_stop_distance_m) < 0.0:
            raise ValueError("collision_stop_distance_m must be >= 0.")
        if float(self.collision_stop_distance_m) > float(self.collision_hold_distance_m):
            raise ValueError("collision_stop_distance_m must be <= collision_hold_distance_m.")
        if float(self.collision_hold_distance_m) > float(self.collision_slow_distance_m):
            raise ValueError("collision_hold_distance_m must be <= collision_slow_distance_m.")
        if float(self.max_joint_abs_deg) <= 0.0:
            raise ValueError("max_joint_abs_deg must be positive.")
        if float(self.joint_slow_margin_deg) < 0.0:
            raise ValueError("joint_slow_margin_deg must be >= 0.")
        if float(self.joint_hold_margin_deg) < 0.0:
            raise ValueError("joint_hold_margin_deg must be >= 0.")
        if float(self.joint_hold_margin_deg) > float(self.max_joint_abs_deg):
            raise ValueError("joint_hold_margin_deg must be <= max_joint_abs_deg.")
        if float(self.joint_slow_margin_deg) > float(self.max_joint_abs_deg):
            raise ValueError("joint_slow_margin_deg must be <= max_joint_abs_deg.")
        if float(self.single_arm_max_qpos_delta_deg) <= 0.0:
            raise ValueError("single_arm_max_qpos_delta_deg must be positive.")
        if int(self.branch_search_max_branches) < 0:
            raise ValueError("branch_search_max_branches must be >= 0.")
        if int(self.tracking_check_every) < 0:
            raise ValueError("tracking_check_every must be >= 0.")
        if float(self.max_tracking_error_deg) <= 0.0:
            raise ValueError("max_tracking_error_deg must be positive.")
        if int(self.progress_every) < 0:
            raise ValueError("progress_every must be >= 0.")


def execute_rm65_qpos_test_sequence(
    config: RM65QposSequenceExecuteConfig,
    *,
    robots_by_side: Mapping[str, object] | None = None,
    arm_ip_by_side: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    sequence = load_qpos_test_sequence(config.sequence_path)
    entries = _selected_entries(
        sequence,
        start_index=int(config.start_index),
        max_targets=int(config.max_targets),
    )
    sides = tuple(config.active_sides)
    robots: dict[str, object] = {}
    owned_robots = False
    started_at = time.time()
    results: list[dict[str, Any]] = []
    total_plan_steps = 0
    total_plan_duration = 0.0
    active_target_context: dict[str, Any] | None = None

    try:
        if config.current_qpos_path is not None:
            current_qpos = validate_qpos_by_side(load_qpos_by_side_file(config.current_qpos_path))
        else:
            if robots_by_side is None:
                robots = connect_rm65_arms(arm_ip_by_side or {}, active_sides=sides)
                owned_robots = True
            else:
                robots = {side: robots_by_side[side] for side in sides}
            current_qpos = read_current_qpos_by_side(
                robots,
                active_sides=sides,
                retry=int(config.state_read_retries),
            )

        if config.execute and not robots:
            if robots_by_side is None:
                robots = connect_rm65_arms(arm_ip_by_side or {}, active_sides=sides)
                owned_robots = True
            else:
                robots = {side: robots_by_side[side] for side in sides}

        current_qpos = _filter_sides(current_qpos, sides)
        neutral_qpos = (
            _load_branch_neutral_qpos(config.branch_neutral_qpos_path, sides=sides)
            if bool(config.branch_search) and sides == ("left", "right")
            else None
        )
        collision_model = None
        if bool(config.collision_check) and sides == ("left", "right"):
            if int(config.progress_every) > 0:
                print(
                    "rm65-qpos-sequence-execute: building collision model "
                    f"scene={config.collision_scene_profile}",
                    flush=True,
                )
            collision_model = build_rm65_qpos_init_collision_model(
                reference_qpos_by_side=current_qpos,
                collision_scene_profile=str(config.collision_scene_profile),
            )
        for sequence_position, entry in enumerate(entries):
            target = _filter_sides(entry.qpos_by_side, sides)
            active_target_context = {
                "sequence_position": int(sequence_position),
                "target_index": int(entry.index),
                "target_name": entry.name,
                "source": _json_safe(entry.source),
            }
            if int(config.progress_every) > 0:
                print(
                    "rm65-qpos-sequence-execute: planning "
                    f"target {sequence_position + 1}/{len(entries)} "
                    f"index={entry.index} name={entry.name}",
                    flush=True,
                )
            plan = build_rm65_qpos_init_plan(
                start_qpos_by_side=current_qpos,
                target_qpos_by_side=target,
                active_sides=sides,
                qpos_dt_sec=float(config.qpos_dt_sec),
                joint_velocity_profile=str(config.joint_velocity_profile),
                max_steps_per_side=int(config.max_steps_per_side),
                collision_check=bool(config.collision_check),
                collision_scene_profile=str(config.collision_scene_profile),
                collision_slow_distance_m=float(config.collision_slow_distance_m),
                collision_hold_distance_m=float(config.collision_hold_distance_m),
                collision_stop_distance_m=float(config.collision_stop_distance_m),
                max_joint_abs_deg=float(config.max_joint_abs_deg),
                joint_slow_margin_deg=float(config.joint_slow_margin_deg),
                joint_hold_margin_deg=float(config.joint_hold_margin_deg),
                single_arm_max_qpos_delta_deg=float(config.single_arm_max_qpos_delta_deg),
                branch_search=bool(config.branch_search),
                branch_search_max_branches=int(config.branch_search_max_branches),
                branch_neutral_qpos_by_side=neutral_qpos,
                collision_model=collision_model,
            )
            _require_plan_completed(plan)
            plan_summary = plan.summary()
            total_plan_steps += len(plan.steps)
            total_plan_duration += float(plan_summary["duration_sec"])
            if int(config.progress_every) > 0:
                print(
                    "rm65-qpos-sequence-execute: plan "
                    f"target {sequence_position + 1}/{len(entries)} "
                    f"planner={_plan_planner_name(plan)} "
                    f"steps={plan_summary['step_count']} "
                    f"duration_sec={float(plan_summary['duration_sec']):.3f} "
                    f"collision_scene={_plan_collision_scene(plan)}",
                    flush=True,
                )

            execution_report: dict[str, Any] = {
                "executed": False,
                "sent_step_count": 0,
                "last_sent_step_index": None,
                "tracking_checks": [],
                "stop_results": [],
            }
            if config.execute:
                execution_report = send_rm65_qpos_plan(
                    robots_by_side=robots,
                    plan=plan,
                    force_guard=config.force_guard,
                    dt_sec=float(config.qpos_dt_sec),
                    trajectory_mode=int(config.trajectory_mode),
                    radio=int(config.radio),
                    tracking_check_every=int(config.tracking_check_every),
                    max_tracking_error_deg=float(config.max_tracking_error_deg),
                    state_read_retries=int(config.state_read_retries),
                    progress_every=int(config.progress_every),
                    progress_label=(
                        "rm65-qpos-sequence-execute "
                        f"target {sequence_position + 1}/{len(entries)}"
                    ),
                )
                current_qpos = read_current_qpos_by_side(
                    robots,
                    active_sides=sides,
                    retry=int(config.state_read_retries),
                )
                if float(config.hold_sec_between_targets) > 0.0 and sequence_position + 1 < len(entries):
                    time.sleep(float(config.hold_sec_between_targets))
            else:
                current_qpos = plan.final_qpos_by_side

            target_result: dict[str, Any] = {
                "sequence_position": int(sequence_position),
                "target_index": int(entry.index),
                "target_name": entry.name,
                "source": _json_safe(entry.source),
                "plan_summary": _json_safe(plan_summary),
                "execution": execution_report,
            }
            if config.include_per_target_plans:
                target_result["plan"] = rm65_qpos_init_plan_to_dict(
                    plan,
                    include_steps=True,
                    max_steps=int(config.max_steps_json),
                )
            results.append(target_result)
            active_target_context = None

        report = _build_report(
            config=config,
            sequence=sequence,
            entries=entries,
            started_at=started_at,
            results=results,
            total_plan_steps=total_plan_steps,
            total_plan_duration=total_plan_duration,
            final_qpos_by_side=current_qpos,
            status="completed",
        )
        _write_report_if_requested(report, config.output_path)
        return report
    except BaseException as exc:
        stop_results = []
        if config.execute and robots:
            stop_results = stop_rm65_arms(robots, reason="qpos_sequence_exception")
        report = {
            "data_type": "teleop_runtime_rm65_qpos_sequence_execute_error",
            "schema_version": 1,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_count": len(results),
            "failed_target": active_target_context,
            "results": results,
            "stop_results": [result.to_dict() for result in stop_results],
        }
        _write_report_if_requested(report, config.output_path)
        raise
    finally:
        if owned_robots:
            close_rm65_arms(robots)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially execute an RM65 qpos test sequence on real hardware. "
            "Without --execute this only reads live qpos and dry-runs the sequence."
        )
    )
    parser.add_argument("--sequence", required=True, help="Qpos test sequence JSON.")
    parser.add_argument("--current-qpos", default="", help="Optional JSON current qpos; skips live state read.")
    parser.add_argument("--arm", default="both", help="left, right, or both.")
    parser.add_argument("--left-ip", default=os.environ.get("LEFT_ARM_IP", ""))
    parser.add_argument("--right-ip", default=os.environ.get("RIGHT_ARM_IP", ""))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=0, help="0 means all selected targets.")
    parser.add_argument("--hold-sec-between-targets", type=float, default=0.25)
    parser.add_argument("--qpos-dt-sec", type=float, default=0.05)
    parser.add_argument("--joint-velocity-profile", default="rm65_init_slow")
    parser.add_argument("--max-steps-per-side", type=int, default=10000)
    parser.add_argument(
        "--collision-scene-profile",
        default=DEFAULT_QPOS_INIT_COLLISION_SCENE,
        help="Dual-arm QPOS init collision scene profile.",
    )
    parser.add_argument("--no-collision-check", action="store_true", help="Disable dual-arm constrained collision planning.")
    parser.add_argument("--collision-slow-distance-m", type=float, default=DEFAULT_QPOS_INIT_COLLISION_SLOW_DISTANCE_M)
    parser.add_argument("--collision-hold-distance-m", type=float, default=DEFAULT_QPOS_INIT_COLLISION_HOLD_DISTANCE_M)
    parser.add_argument("--collision-stop-distance-m", type=float, default=DEFAULT_QPOS_INIT_COLLISION_STOP_DISTANCE_M)
    parser.add_argument("--max-joint-abs-deg", type=float, default=DEFAULT_QPOS_INIT_MAX_JOINT_ABS_DEG)
    parser.add_argument("--joint-slow-margin-deg", type=float, default=DEFAULT_QPOS_INIT_JOINT_SLOW_MARGIN_DEG)
    parser.add_argument("--joint-hold-margin-deg", type=float, default=DEFAULT_QPOS_INIT_JOINT_HOLD_MARGIN_DEG)
    parser.add_argument("--single-arm-max-qpos-delta-deg", type=float, default=DEFAULT_SINGLE_ARM_MAX_QPOS_DELTA_DEG)
    parser.add_argument("--no-branch-search", action="store_true", help="Disable dual-arm staged joint-order fallback planning.")
    parser.add_argument("--branch-search-max-branches", type=int, default=DEFAULT_QPOS_INIT_BRANCH_SEARCH_MAX_BRANCHES)
    parser.add_argument(
        "--branch-neutral-qpos",
        default=str(DEFAULT_QPOS_TARGET_PATH),
        help="Optional neutral qpos target JSON used as the first dual-arm branch-search waypoint; empty disables.",
    )
    parser.add_argument("--trajectory-mode", type=int, default=0)
    parser.add_argument("--radio", type=int, default=0)
    parser.add_argument("--state-read-retries", type=int, default=2)
    parser.add_argument("--tracking-check-every", type=int, default=10)
    parser.add_argument("--max-tracking-error-deg", type=float, default=30.0)
    parser.add_argument("--disable-force-guard", action="store_true")
    parser.add_argument("--force-source", default="tool")
    parser.add_argument("--max-force-n", type=float, default=15.0)
    parser.add_argument("--max-torque-nm", type=float, default=2.0)
    parser.add_argument("--progress-every", type=int, default=20, help="Print execution progress every N steps; 0 disables.")
    parser.add_argument("--execute", action="store_true", help="Actually send Movej_CANFD commands.")
    parser.add_argument("--confirm", default="", help=f"Required for --execute: {CONFIRM_TEXT}")
    parser.add_argument("--output", default="", help="Optional report JSON path.")
    parser.add_argument("--include-per-target-plans", action="store_true")
    parser.add_argument("--max-steps-json", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(_normalize_cli_argv(argv))
    active_sides = normalize_sides(str(args.arm))
    arm_ip_by_side = {
        side: value
        for side, value in {
            "left": str(args.left_ip).strip(),
            "right": str(args.right_ip).strip(),
        }.items()
        if value
    }
    config = RM65QposSequenceExecuteConfig(
        sequence_path=args.sequence,
        active_sides=active_sides,
        current_qpos_path=(str(args.current_qpos).strip() or None),
        execute=bool(args.execute),
        confirm=str(args.confirm),
        start_index=int(args.start_index),
        max_targets=int(args.max_targets),
        hold_sec_between_targets=float(args.hold_sec_between_targets),
        qpos_dt_sec=float(args.qpos_dt_sec),
        joint_velocity_profile=str(args.joint_velocity_profile),
        max_steps_per_side=int(args.max_steps_per_side),
        collision_check=not bool(args.no_collision_check),
        collision_scene_profile=str(args.collision_scene_profile),
        collision_slow_distance_m=float(args.collision_slow_distance_m),
        collision_hold_distance_m=float(args.collision_hold_distance_m),
        collision_stop_distance_m=float(args.collision_stop_distance_m),
        max_joint_abs_deg=float(args.max_joint_abs_deg),
        joint_slow_margin_deg=float(args.joint_slow_margin_deg),
        joint_hold_margin_deg=float(args.joint_hold_margin_deg),
        single_arm_max_qpos_delta_deg=float(args.single_arm_max_qpos_delta_deg),
        branch_search=not bool(args.no_branch_search),
        branch_search_max_branches=int(args.branch_search_max_branches),
        branch_neutral_qpos_path=(str(args.branch_neutral_qpos).strip() or None),
        trajectory_mode=int(args.trajectory_mode),
        radio=int(args.radio),
        state_read_retries=int(args.state_read_retries),
        force_guard=RM65ForceGuardConfig(
            enabled=not bool(args.disable_force_guard),
            source=str(args.force_source),
            max_force_n=float(args.max_force_n),
            max_torque_nm=float(args.max_torque_nm),
            emergency_stop_on_limit=True,
        ),
        tracking_check_every=int(args.tracking_check_every),
        max_tracking_error_deg=float(args.max_tracking_error_deg),
        progress_every=int(args.progress_every),
        output_path=(str(args.output).strip() or None),
        include_per_target_plans=bool(args.include_per_target_plans),
        max_steps_json=int(args.max_steps_json),
    )
    report = execute_rm65_qpos_test_sequence(config, arm_ip_by_side=arm_ip_by_side)
    print(
        "rm65-qpos-sequence-execute "
        f"sequence={Path(args.sequence).name} "
        f"targets={report['target_count']} "
        f"completed={report['completed_count']} "
        f"execute={report['execute_requested']} "
        f"sent_steps={report['total_sent_step_count']} "
        f"planned_steps={report['total_plan_step_count']}"
    )
    if not str(args.output).strip():
        print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    return 0


def _selected_entries(
    sequence: QposTestSequence,
    *,
    start_index: int,
    max_targets: int,
) -> tuple[Any, ...]:
    if int(start_index) >= len(sequence.entries):
        raise ValueError(
            f"start_index={start_index} is outside sequence with {len(sequence.entries)} entries."
        )
    entries = sequence.entries[int(start_index) :]
    if int(max_targets) > 0:
        entries = entries[: int(max_targets)]
    if not entries:
        raise ValueError("selected qpos test sequence target list is empty.")
    return tuple(entries)


def _normalize_cli_argv(argv: Sequence[str] | None) -> list[str]:
    raw = sys.argv[1:] if argv is None else list(argv)
    return [str(item).strip() for item in raw if str(item).strip()]


def _filter_sides(qpos_by_side: Mapping[str, Any], sides: Sequence[str]) -> dict[str, np.ndarray]:
    qpos = validate_qpos_by_side(qpos_by_side)
    missing = [side for side in sides if side not in qpos]
    if missing:
        raise ValueError(f"qpos mapping missing sides {missing}.")
    return {side: qpos[side].copy() for side in sides}


def _build_report(
    *,
    config: RM65QposSequenceExecuteConfig,
    sequence: QposTestSequence,
    entries: Sequence[Any],
    started_at: float,
    results: Sequence[Mapping[str, Any]],
    total_plan_steps: int,
    total_plan_duration: float,
    final_qpos_by_side: Mapping[str, np.ndarray],
    status: str,
) -> dict[str, Any]:
    return {
        "data_type": "teleop_runtime_rm65_qpos_sequence_execute_report",
        "schema_version": 1,
        "status": status,
        "generated_at_unix_sec": started_at,
        "finished_at_unix_sec": time.time(),
        "sequence_path": str(config.sequence_path),
        "sequence_name": sequence.name,
        "sequence_entry_count": len(sequence.entries),
        "target_count": len(entries),
        "completed_count": len(results),
        "execute_requested": bool(config.execute),
        "active_sides": list(config.active_sides),
        "start_index": int(config.start_index),
        "max_targets": int(config.max_targets),
        "total_plan_step_count": int(total_plan_steps),
        "total_plan_duration_sec": float(total_plan_duration),
        "total_sent_step_count": int(
            sum(int(result["execution"].get("sent_step_count", 0)) for result in results)
        ),
        "final_qpos_by_side": {
            side: np.asarray(qpos, dtype=float).reshape(-1).astype(float).tolist()
            for side, qpos in final_qpos_by_side.items()
        },
        "config": {
            "qpos_dt_sec": float(config.qpos_dt_sec),
            "joint_velocity_profile": str(config.joint_velocity_profile),
            "max_steps_per_side": int(config.max_steps_per_side),
            "collision_check": bool(config.collision_check),
            "collision_scene_profile": str(config.collision_scene_profile),
            "collision_slow_distance_m": float(config.collision_slow_distance_m),
            "collision_hold_distance_m": float(config.collision_hold_distance_m),
            "collision_stop_distance_m": float(config.collision_stop_distance_m),
            "max_joint_abs_deg": float(config.max_joint_abs_deg),
            "joint_slow_margin_deg": float(config.joint_slow_margin_deg),
            "joint_hold_margin_deg": float(config.joint_hold_margin_deg),
            "single_arm_max_qpos_delta_deg": float(config.single_arm_max_qpos_delta_deg),
            "branch_search": bool(config.branch_search),
            "branch_search_max_branches": int(config.branch_search_max_branches),
            "branch_neutral_qpos_path": None if config.branch_neutral_qpos_path is None else str(config.branch_neutral_qpos_path),
            "trajectory_mode": int(config.trajectory_mode),
            "radio": int(config.radio),
            "tracking_check_every": int(config.tracking_check_every),
            "max_tracking_error_deg": float(config.max_tracking_error_deg),
            "progress_every": int(config.progress_every),
            "hold_sec_between_targets": float(config.hold_sec_between_targets),
            "force_guard": config.force_guard.to_dict(),
        },
        "results": [_json_safe(result) for result in results],
    }


def _write_report_if_requested(report: Mapping[str, Any], output_path: str | Path | None) -> None:
    if output_path is None:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, RM65ForceLimitExceeded):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
