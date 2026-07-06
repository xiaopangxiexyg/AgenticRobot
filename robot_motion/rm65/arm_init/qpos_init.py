from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robot_motion.arm_init import (
    ArmQposTransitionConfig,
    load_qpos_by_side_file,
    plan_slow_qpos_transition,
    qpos_transition_plan_to_dict,
)
from robot_motion.rm65.qpos_layout import normalize_sides, validate_qpos_by_side

from .schema import load_qpos_target


DEFAULT_QPOS_TARGET_PATH = Path("robot_motion/rm65/arm_init/qpos_targets/bench_init_pose.json")


def build_rm65_slow_qpos_init_plan_from_files(
    *,
    current_qpos_path: str | Path,
    qpos_target_path: str | Path = DEFAULT_QPOS_TARGET_PATH,
    active_sides: Sequence[str] | None = None,
    config: ArmQposTransitionConfig | None = None,
):
    current = validate_qpos_by_side(load_qpos_by_side_file(current_qpos_path))
    target = load_qpos_target(qpos_target_path).qpos_by_side()
    sides = tuple(target) if active_sides is None else normalize_sides(tuple(active_sides))
    current = {side: current[side] for side in sides}
    target = {side: target[side] for side in sides}
    return plan_slow_qpos_transition(
        start_qpos_by_side=current,
        target_qpos_by_side=target,
        config=config if config is not None else ArmQposTransitionConfig(),
        side_order=sides,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a slow RM65 qpos initialization plan from current qpos to a target qpos JSON. "
            "This command is read-only and does not connect to hardware."
        )
    )
    parser.add_argument("--current-qpos", required=True, help="JSON file with current qpos by side.")
    parser.add_argument("--qpos-target", default=str(DEFAULT_QPOS_TARGET_PATH), help="RM65 qpos target schema JSON.")
    parser.add_argument("--arm", default="", help="left, right, or both. Empty means target file sides.")
    parser.add_argument("--qpos-dt-sec", type=float, default=0.05)
    parser.add_argument("--joint-velocity-profile", default="rm65_init_slow")
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    parser.add_argument("--include-steps", action="store_true", help="Include qpos steps in output JSON.")
    parser.add_argument("--max-steps-json", type=int, default=0, help="0 means all serialized steps.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    active_sides = None if not str(args.arm).strip() else normalize_sides(str(args.arm))
    plan = build_rm65_slow_qpos_init_plan_from_files(
        current_qpos_path=args.current_qpos,
        qpos_target_path=args.qpos_target,
        active_sides=active_sides,
        config=ArmQposTransitionConfig(
            dt_sec=float(args.qpos_dt_sec),
            joint_velocity_profile_name=str(args.joint_velocity_profile),
        ),
    )
    payload = qpos_transition_plan_to_dict(
        plan,
        include_steps=bool(args.include_steps),
        max_steps=int(args.max_steps_json),
    )
    payload["data_type"] = "robot_motion_rm65_slow_qpos_init_plan"
    text = json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    if str(args.output).strip():
        Path(args.output).write_text(text, encoding="utf-8")
    print(
        "rm65-qpos-init "
        f"sides={','.join(plan.sides)} "
        f"steps={len(plan.steps)} "
        f"duration_sec={plan.summary()['duration_sec']:.3f} "
        f"profile={plan.metadata.get('joint_velocity_profile')}"
    )
    if not str(args.output).strip():
        print(text, end="")
    return 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
