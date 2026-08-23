"""Headless-first B601-RS IK-to-MuJoCo position-servo demonstration."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from robot_motion.rebot_b601_rs import (
    B601DampedLeastSquaresIK,
    B601PinocchioModel,
)
from sim.mujoco.viewer import MujocoPassiveViewerSession

from .driver import B601JointPositionCommand, B601PositionDriver
from .scene import (
    B601_RS_DEFAULT_ARM_QPOS,
    B601_RS_PHYSICS_HZ,
    build_b601_mujoco_scene,
)

DEFAULT_TARGET_QPOS = (0.15, 1.1, 1.15, 0.1, -0.1, 0.2)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve one continuous-seed B601 pose target and track the result "
            "with the provisional 500 Hz MuJoCo position servo."
        )
    )
    parser.add_argument(
        "--rebot-source-root",
        type=Path,
        required=True,
        help="Explicit root of the pinned reBotArm_control_py source snapshot.",
    )
    parser.add_argument(
        "--target-qpos",
        type=float,
        nargs=6,
        default=DEFAULT_TARGET_QPOS,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Reachable q used only to generate the demonstration target pose.",
    )
    parser.add_argument(
        "--seed-qpos",
        type=float,
        nargs=6,
        default=B601_RS_DEFAULT_ARM_QPOS,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Explicit IK seed and initial simulated arm position.",
    )
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the existing passive MuJoCo viewer and run in real time.",
    )
    return parser


def run_demo(
    *,
    source_root: str | Path,
    target_qpos: Sequence[float],
    seed_qpos: Sequence[float],
    duration_sec: float,
    viewer: bool,
) -> tuple[int, dict[str, object]]:
    """Run one deterministic demo and return an exit code plus JSON payload."""

    duration = float(duration_sec)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_sec must be finite and positive")
    seed = np.asarray(seed_qpos, dtype=float)
    target_configuration = np.asarray(target_qpos, dtype=float)
    model = B601PinocchioModel.from_rebot_source_root(source_root)
    target_pose = model.forward_kinematics(target_configuration)
    solution = B601DampedLeastSquaresIK(model).solve(
        target_pose,
        seed_qpos=seed,
    )
    payload: dict[str, object] = {
        "ik": {
            "success": solution.success,
            "status": solution.status,
            "iterations": solution.iterations,
            "position_error_m": solution.position_error_m,
            "orientation_error_rad": solution.orientation_error_rad,
            "seed_distance_rad": solution.seed_distance_rad,
            "qpos": list(solution.qpos),
        },
        "fidelity": (
            "visualization_only_provisional; no hardware, gravity, collision, "
            "gripper-actuation, target-slew/state-limit safety, force, or "
            "haptic validation; modest targets only"
        ),
    }
    if not solution.success:
        return 2, payload

    scene = build_b601_mujoco_scene(
        source_root,
        initial_arm_qpos=seed,
    )
    driver = B601PositionDriver(scene)
    driver.reset(seed)
    command = B601JointPositionCommand(
        sequence=0,
        source_timestamp_sec=0.0,
        receive_timestamp_sec=0.0,
        qpos=solution.qpos,
    )
    viewer_session = MujocoPassiveViewerSession(scene) if viewer else None
    if viewer_session is not None:
        viewer_session.start()
    final_tick = None
    steps = math.ceil(duration * B601_RS_PHYSICS_HZ)
    try:
        for step in range(steps):
            final_tick = driver.step_physics(command if step == 0 else None)
            if viewer_session is not None:
                if step % 16 == 0 and not viewer_session.sync():
                    break
                time.sleep(1.0 / B601_RS_PHYSICS_HZ)
    finally:
        if viewer_session is not None:
            viewer_session.close()

    assert final_tick is not None
    tracking_error = float(
        np.max(
            np.abs(
                np.asarray(final_tick.measured_qpos, dtype=float)
                - np.asarray(solution.qpos, dtype=float)
            )
        )
    )
    payload["simulation"] = {
        **driver.execution_metadata,
        "physics_ticks": final_tick.physics_tick + 1,
        "accepted_sequence": final_tick.accepted_sequence,
        "state_valid": final_tick.state_valid,
        "max_joint_tracking_error_rad": tracking_error,
        "measured_qpos": list(final_tick.measured_qpos),
    }
    if not final_tick.state_valid:
        return 3, payload
    if tracking_error > 2e-3:
        return 4, payload
    return 0, payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    code, payload = run_demo(
        source_root=args.rebot_source_root,
        target_qpos=args.target_qpos,
        seed_qpos=args.seed_qpos,
        duration_sec=args.duration_sec,
        viewer=bool(args.viewer),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
