from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from live_source import HardwareActionExecutor, profile_max_arm_joint_step_rad
from paths import ensure_import_paths
from profiles import DeploymentProfile


ensure_import_paths()


FR3_ACTION_DIM = 19
FR3_ARM_DOF = 7
FR3_HAND_DOF = 6
FR3_ARM_SLICE = slice(0, FR3_ARM_DOF)
FR3_RIGHT_HAND_SLICE = slice(FR3_ARM_DOF, FR3_ARM_DOF + FR3_HAND_DOF)
FR3_LEFT_HAND_SLICE = slice(FR3_ARM_DOF + FR3_HAND_DOF, FR3_ACTION_DIM)


def add_startup_replay_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-startup-key-wait", dest="startup_key_wait", action="store_false", default=True)
    parser.add_argument("--startup-init-sleep-s", type=float, default=10.0)
    parser.add_argument("--startup-replay-action-column", type=str, default="action")
    parser.add_argument("--startup-replay-start-frame", type=int, default=0)
    parser.add_argument("--startup-replay-seed", type=int, default=None)
    parser.add_argument("--which-episode", type=int, default=None)
    parser.add_argument("--startup-init-steps", type=int, default=24)
    parser.add_argument("--startup-init-hz", type=float, default=12.0)
    parser.add_argument("--startup-init-tolerance-rad", type=float, default=0.02)
    parser.add_argument("--startup-init-settle-timeout-s", type=float, default=10.0)
    parser.add_argument("--startup-init-settle-log-s", type=float, default=0.5)


def _is_fr3_profile(profile: DeploymentProfile) -> bool:
    return str(getattr(profile, "arm_backend", "rm65")).lower() == "fr3"


def _uses_dmtac_gripper(profile: DeploymentProfile) -> bool:
    return str(getattr(profile, "hand_backend", "")).lower() == "dmtac_gripper"


def _startup_hardware_args(args: argparse.Namespace, profile: DeploymentProfile) -> argparse.Namespace:
    if not _uses_dmtac_gripper(profile):
        return args
    payload = vars(args).copy()
    if bool(payload.get("gripper_connect_tactile", True)):
        payload["gripper_connect_tactile"] = False
        payload["gripper_tactile_warmup_frames"] = 0
        print("[Startup] DM-Tac tactile disabled for temporary startup motion; live inference will reconnect it.")
    return argparse.Namespace(**payload)


def _parquet_reader_pythons() -> list[str]:
    helpers: list[str] = []
    override = os.environ.get("PARQUET_READER_PYTHON", "").strip()
    if override:
        helpers.append(override)
    helpers.extend(
        [
            "/home/user/anaconda3/envs/lerobot/bin/python",
            "/home/user/anaconda3/envs/beingd1/bin/python",
            "/home/user/anaconda3/envs/lerobot032/bin/python",
        ]
    )
    return helpers


def _run_parquet_helper(script: str, *args: str) -> Any:
    last_error: Exception | None = None
    for python_bin in _parquet_reader_pythons():
        if not Path(python_bin).exists():
            continue
        try:
            result = subprocess.run(
                [python_bin, "-c", script, *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(result.stdout)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "Failed to read Franka startup parquet with helper Python envs. "
        "Set PARQUET_READER_PYTHON to a Python executable with working pandas/pyarrow."
    ) from last_error


def _load_fr3_replay_actions_via_helper(
    path: Path,
    *,
    action_column: str,
    start_index: int,
    end_index: int | None,
) -> np.ndarray:
    script = r"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

path = Path(sys.argv[1])
action_column = sys.argv[2]
start = max(0, int(sys.argv[3]))
end_arg = sys.argv[4]
end = None if end_arg == "" else int(end_arg)
df = pd.read_parquet(path)
if len(df) == 0:
    raise ValueError(f"Replay parquet contains no rows: {path}")
stop = len(df) if end is None else min(len(df), end)
if stop <= start:
    raise ValueError(f"Replay frame range is empty: start_index={start}, end_index={end}, row_count={len(df)}.")

def vector(value, dim, row_index, column):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (dim,):
        raise ValueError(f"Replay row {row_index} column {column!r} must have shape ({dim},), got {arr.shape}.")
    return arr

actions = []
for row_index in range(start, stop):
    row = df.iloc[row_index]
    if action_column in df.columns and row.get(action_column) is not None:
        action = vector(row.get(action_column), 19, row_index, action_column)
    else:
        action = np.full((19,), np.nan, dtype=np.float32)
        arm_value = row.get("action.arm.joint_target") if "action.arm.joint_target" in df.columns else None
        right_value = row.get("action.right_hand.joint_target") if "action.right_hand.joint_target" in df.columns else None
        left_value = row.get("action.left_hand.joint_target") if "action.left_hand.joint_target" in df.columns else None
        if arm_value is not None:
            action[:7] = vector(arm_value, 7, row_index, "action.arm.joint_target")
        if right_value is not None:
            action[7:13] = vector(right_value, 6, row_index, "action.right_hand.joint_target")
        if left_value is not None:
            action[13:19] = vector(left_value, 6, row_index, "action.left_hand.joint_target")
        if not np.any(np.isfinite(action)):
            raise ValueError(
                "Replay parquet does not contain an action column or supported action.*.joint_target columns."
            )
    actions.append(action.astype(np.float32).tolist())
print(json.dumps(actions))
"""
    data = _run_parquet_helper(
        script,
        str(path),
        str(action_column or "action"),
        str(max(0, int(start_index))),
        "" if end_index is None else str(int(end_index)),
    )
    return np.asarray(data, dtype=np.float32)


def _load_replay_actions_from_parquet(profile: DeploymentProfile, path: Path, args: argparse.Namespace, mode: str) -> np.ndarray:
    start = max(0, int(getattr(args, "startup_replay_start_frame", 0)))
    end = start + 1 if mode == "i" else None
    action_column = str(getattr(args, "startup_replay_action_column", "action") or "action")
    if _is_fr3_profile(profile):
        actions = _load_fr3_replay_actions_via_helper(
            path,
            action_column=action_column,
            start_index=start,
            end_index=end,
        )
    else:
        from teleop_runtime.deploy.replay import load_replay_actions_from_parquet

        actions = load_replay_actions_from_parquet(
            path,
            action_column=action_column,
            start_index=start,
            end_index=end,
        )
    if mode == "i":
        actions = actions[:1]
    return np.asarray(actions, dtype=np.float32)


def wait_for_startup_key(args: argparse.Namespace) -> str | None:
    if not bool(getattr(args, "startup_key_wait", True)):
        return None
    print("[Startup] waiting for key: press r to replay a random dataset episode, i to send its first action.")
    if not sys.stdin.isatty():
        line = sys.stdin.readline().strip().lower()
        return line[:1] if line[:1] in {"r", "i"} else None

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1).lower()
            if ch in {"r", "i"}:
                print("")
                return ch
            if ch in {"q", "\x03"}:
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _episode_candidates(args: argparse.Namespace) -> list[Path]:
    if str(getattr(args, "parquet_path", "") or ""):
        path = Path(args.parquet_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--parquet-path does not exist: {path}")
        return [path]

    dataset_path = str(getattr(args, "dataset_path", "") or "")
    if not dataset_path:
        raise ValueError("startup replay requires --dataset-path or --parquet-path.")
    root = Path(dataset_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"--dataset-path does not exist: {root}")
    candidates = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'data'}")
    return candidates


def choose_episode(args: argparse.Namespace) -> Path:
    candidates = _episode_candidates(args)
    which_episode = getattr(args, "which_episode", None)
    if which_episode is not None:
        index = int(which_episode)
        if index < 0 or index >= len(candidates):
            raise IndexError(f"--which-episode={index} is out of range for {len(candidates)} episode(s).")
        return Path(candidates[index]).expanduser().resolve()
    seed = getattr(args, "startup_replay_seed", None)
    rng = random.Random(seed) if seed is not None else random
    return Path(rng.choice(candidates)).expanduser().resolve()


def bind_selected_episode(args: argparse.Namespace, path: Path) -> None:
    setattr(args, "parquet_path", str(Path(path).expanduser().resolve()))


def load_startup_actions(args: argparse.Namespace, profile: DeploymentProfile, mode: str) -> tuple[Path, np.ndarray]:
    path = choose_episode(args)
    bind_selected_episode(args, path)
    return path, _load_replay_actions_from_parquet(profile, path, args, mode)


def _startup_direct_columns(profile: DeploymentProfile) -> tuple[list[str], dict[str, list[str]]]:
    columns = []
    arm_columns_by_side: dict[str, list[str]] = {}
    for side in profile.required_arms:
        candidates = [f"action.replay.{side}_arm_joint_target"]
        if _is_fr3_profile(profile):
            candidates.insert(0, "action.replay.arm_joint_target")
        arm_columns_by_side[side] = candidates
        columns.extend(candidates)
    for side in profile.required_hands:
        columns.append(f"action.replay.{side}_hand_joint_target")
    return columns, arm_columns_by_side


def _load_fr3_direct_target_row_via_helper(path: Path, columns: list[str], frame_index: int) -> dict[str, Any]:
    script = r"""
import json
import sys
from pathlib import Path
import pandas as pd

path = Path(sys.argv[1])
columns = json.loads(sys.argv[2])
frame_index = max(0, int(sys.argv[3]))
df = pd.read_parquet(path)
if len(df) == 0:
    raise ValueError(f"Replay parquet contains no rows: {path}")
row = df.iloc[min(frame_index, len(df) - 1)]
available = set(df.columns)
item = {}
for key in columns:
    if key not in available:
        continue
    value = row.get(key)
    if hasattr(value, "tolist"):
        value = value.tolist()
    item[key] = value
print(json.dumps(item))
"""
    data = _run_parquet_helper(script, str(path), json.dumps(columns), str(max(0, int(frame_index))))
    return dict(data)


def load_startup_direct_targets(args: argparse.Namespace, profile: DeploymentProfile) -> tuple[Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    path = choose_episode(args)
    bind_selected_episode(args, path)
    frame_index = max(0, int(getattr(args, "startup_replay_start_frame", 0)))

    columns, arm_columns_by_side = _startup_direct_columns(profile)
    if _is_fr3_profile(profile):
        row = _load_fr3_direct_target_row_via_helper(path, columns, frame_index)
        if not row:
            raise ValueError(f"{path} has no action.replay.* joint target columns for direct startup init.")
    else:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Direct startup init requires pyarrow in the active environment.") from exc

        schema = pq.read_schema(path)
        available = set(schema.names)
        columns = [column for column in columns if column in available]
        if not columns:
            raise ValueError(f"{path} has no action.replay.* joint target columns for direct startup init.")
        table = pq.read_table(path, columns=columns)
        rows = table.to_pylist()
        if not rows:
            raise ValueError(f"Replay parquet contains no rows: {path}")
        row = rows[min(frame_index, len(rows) - 1)]

    arm_targets: dict[str, np.ndarray] = {}
    hand_targets: dict[str, np.ndarray] = {}
    arm_width = 7 if _is_fr3_profile(profile) else 6
    for side in profile.required_arms:
        value = None
        for column in arm_columns_by_side.get(side, ()):
            value = row.get(column)
            if value is not None:
                break
        if value is not None:
            arm_targets[side] = np.asarray(value, dtype=np.float32).reshape(arm_width)
    for side in profile.required_hands:
        value = row.get(f"action.replay.{side}_hand_joint_target")
        if value is not None:
            hand_targets[side] = np.asarray(value, dtype=np.float32).reshape(6)
    if not arm_targets and not hand_targets:
        raise ValueError(f"{path} row {frame_index} has no finite direct startup targets.")
    return path, arm_targets, hand_targets


def _print_replay_progress(sent: int, total: int) -> None:
    total = max(1, int(total))
    sent = max(0, min(int(sent), total))
    width = 30
    filled = int(round(width * sent / total))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[Startup] replay progress [{bar}] {sent}/{total}", end="", flush=True)
    if sent >= total:
        print("")


def _sleep_with_countdown(duration_s: float) -> None:
    remaining = int(np.ceil(max(0.0, float(duration_s))))
    while remaining > 0:
        print(f"[Startup] inference starts in {remaining}s")
        time.sleep(1.0)
        remaining -= 1


def _finite_6(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        return None
    return arr.astype(np.float32, copy=True)


def _fixed_hand_from_args(args: argparse.Namespace) -> np.ndarray | None:
    return _finite_6(getattr(args, "fixed_hand", None))


def _apply_fixed_hand_targets(
    hand_targets: Mapping[str, np.ndarray],
    *,
    args: argparse.Namespace,
    profile: DeploymentProfile,
) -> dict[str, np.ndarray]:
    fixed_hand = _fixed_hand_from_args(args)
    if fixed_hand is None or not _is_fr3_profile(profile):
        return {str(side): np.asarray(value, dtype=np.float32).reshape(6) for side, value in hand_targets.items()}
    sides = tuple(str(side) for side in profile.required_hands)
    if not sides:
        return {str(side): np.asarray(value, dtype=np.float32).reshape(6) for side, value in hand_targets.items()}
    return {side: fixed_hand.copy() for side in sides}


def _finite_vector(value: Any, width: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (width,) or not np.all(np.isfinite(arr)):
        return None
    return arr.astype(np.float32, copy=True)


def _current_joint_targets(executor: HardwareActionExecutor) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    observation = executor._build_observation()
    arm_current: dict[str, np.ndarray] = {}
    hand_current: dict[str, np.ndarray] = {}
    arm_width = 7 if str(getattr(executor, "backend_kind", "rm65")).lower() == "fr3" else 6

    arm_states = getattr(observation, "arm_states", None)
    if isinstance(arm_states, Mapping):
        arm_items = arm_states.items()
    else:
        arm_state = getattr(observation, "arm_state", None)
        if isinstance(arm_state, Mapping):
            side = str(arm_state.get("arm_side") or "")
            if not side:
                active_sides = tuple(getattr(observation, "active_sides", ()) or ())
                side = str(active_sides[0]) if active_sides else "right"
            arm_items = ((side, arm_state),)
        else:
            arm_items = ()

    for side, state in arm_items:
        if not isinstance(state, Mapping):
            continue
        qpos = _finite_vector(state.get("current_qpos"), arm_width)
        if qpos is None:
            qpos = _finite_vector(state.get("qpos"), arm_width)
        if qpos is not None:
            arm_current[str(side)] = qpos

    hand_states = getattr(observation, "hand_states", {})
    if not isinstance(hand_states, Mapping):
        hand_states = {}
    for side, state in hand_states.items():
        if not isinstance(state, Mapping):
            continue
        qpos = _finite_6(state.get("joint_pos"))
        if qpos is None:
            qpos = _finite_6(state.get("last_sent_command"))
        if qpos is not None:
            hand_current[str(side)] = qpos
    return arm_current, hand_current


def _lerp_targets(
    current: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for side, target_value in target.items():
        target_arr = np.asarray(target_value, dtype=np.float32).reshape(-1)
        start_arr = current.get(side)
        if start_arr is None:
            out[side] = target_arr
        else:
            out[side] = (start_arr + (target_arr - start_arr) * float(alpha)).astype(np.float32)
    return out


def _format_target_vector(value: np.ndarray, precision: int = 4) -> str:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return np.array2string(arr, precision=precision, separator=", ")


def _arm_target_errors(
    current: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for side, target_value in target.items():
        current_value = current.get(side)
        if current_value is None:
            continue
        target_arr = np.asarray(target_value, dtype=np.float32).reshape(-1)
        current_arr = np.asarray(current_value, dtype=np.float32).reshape(-1)
        if current_arr.shape != target_arr.shape:
            continue
        if not np.all(np.isfinite(current_arr)) or not np.all(np.isfinite(target_arr)):
            continue
        errors[str(side)] = float(np.max(np.abs(current_arr - target_arr)))
    return errors


def _print_direct_init_targets(
    start_arms: Mapping[str, np.ndarray],
    arm_targets: Mapping[str, np.ndarray],
) -> None:
    for side, target_value in sorted(arm_targets.items()):
        start_value = start_arms.get(side)
        if start_value is None:
            print(f"[Startup] {side} arm target={_format_target_vector(target_value)} current=unavailable")
            continue
        delta = float(
            np.max(
                np.abs(
                    np.asarray(target_value, dtype=np.float32).reshape(-1)
                    - np.asarray(start_value, dtype=np.float32).reshape(-1)
                )
            )
        )
        print(
            f"[Startup] {side} arm current={_format_target_vector(start_value)} "
            f"target={_format_target_vector(target_value)} max_delta={delta:.4f}rad"
        )


def _wait_for_direct_init_convergence(
    executor: HardwareActionExecutor,
    arm_targets: Mapping[str, np.ndarray],
    hand_targets: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    profile: DeploymentProfile,
) -> None:
    if not _is_fr3_profile(profile) or not arm_targets:
        return

    tolerance = max(0.0, float(getattr(args, "startup_init_tolerance_rad", 0.02)))
    timeout_s = max(0.0, float(getattr(args, "startup_init_settle_timeout_s", 10.0)))
    if timeout_s <= 0.0:
        return

    hz = max(1e-6, float(getattr(args, "startup_init_hz", 12.0)))
    log_period_s = max(0.1, float(getattr(args, "startup_init_settle_log_s", 0.5)))
    deadline = time.time() + timeout_s
    last_log_at = 0.0
    last_errors: dict[str, float] = {}

    print(
        f"[Startup] holding final Franka target until measured qpos is within "
        f"{tolerance:.4f}rad, timeout={timeout_s:.1f}s"
    )
    while True:
        current_arms, _ = _current_joint_targets(executor)
        errors = _arm_target_errors(current_arms, arm_targets)
        if errors:
            last_errors = errors
            max_error = max(errors.values())
            if max_error <= tolerance:
                details = ", ".join(f"{side}={err:.4f}rad" for side, err in sorted(errors.items()))
                print(f"[Startup] final Franka target reached: {details}")
                return
        else:
            max_error = float("inf")

        now = time.time()
        if now >= deadline:
            if last_errors:
                details = ", ".join(f"{side}={err:.4f}rad" for side, err in sorted(last_errors.items()))
                print(
                    "WARNING: startup direct init settle timed out before Franka reached target: "
                    f"{details}, tolerance={tolerance:.4f}rad"
                )
            else:
                print("WARNING: startup direct init settle timed out without measured Franka qpos.")
            return

        statuses = executor.send_direct_joint_targets(arm_qpos=arm_targets, hand_qpos=hand_targets)
        if now - last_log_at >= log_period_s:
            if errors:
                details = ", ".join(f"{side}={err:.4f}rad" for side, err in sorted(errors.items()))
                print(f"[Startup] settling final Franka target: {details}")
            else:
                print(f"[Startup] settling final Franka target: waiting for measured qpos; status={';'.join(statuses[:4])}")
            last_log_at = now
        time.sleep(1.0 / hz)


def _print_init_progress(step: int, total: int) -> None:
    total = max(1, int(total))
    step = max(0, min(int(step), total))
    width = 30
    filled = int(round(width * step / total))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[Startup] init progress [{bar}] {step}/{total}", end="", flush=True)
    if step >= total:
        print("")


def _smoothstep(alpha: float) -> float:
    x = float(np.clip(alpha, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _direct_init_steps(
    start_arms: Mapping[str, np.ndarray],
    arm_targets: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    profile: DeploymentProfile,
) -> int:
    requested_steps = max(1, int(getattr(args, "startup_init_steps", 24)))
    max_step = profile_max_arm_joint_step_rad(profile, args)
    if max_step <= 0.0:
        return requested_steps
    max_delta = 0.0
    for side, target in arm_targets.items():
        start = start_arms.get(side)
        if start is None:
            continue
        delta = np.asarray(target, dtype=np.float32).reshape(-1) - np.asarray(start, dtype=np.float32).reshape(-1)
        if delta.size:
            max_delta = max(max_delta, float(np.max(np.abs(delta))))
    limited_steps = int(np.ceil(max_delta / max_step)) if max_delta > 0.0 else 1
    return max(requested_steps, limited_steps)


def run_startup_replay(profile: DeploymentProfile, args: argparse.Namespace, mode: str) -> int:
    if mode not in {"r", "i"}:
        return 0
    if mode == "i":
        path, arm_targets, hand_targets = load_startup_direct_targets(args, profile)
        hand_targets = _apply_fixed_hand_targets(hand_targets, args=args, profile=profile)
        print(
            f"[Startup] selected {path} | mode=first action direct init | "
            f"arms={sorted(arm_targets)} hands={sorted(hand_targets)}"
        )
        fixed_hand = _fixed_hand_from_args(args)
        if fixed_hand is not None and _is_fr3_profile(profile):
            print(f"[Startup] fixed hand override active: {_format_target_vector(fixed_hand)}")
        startup_args = _startup_hardware_args(args, profile)
        executor_cm = HardwareActionExecutor(profile, startup_args)
        executor_cm.start_dispatcher = False
        with executor_cm as executor:
            start_arms, start_hands = _current_joint_targets(executor)
            steps = _direct_init_steps(start_arms, arm_targets, startup_args, profile)
            hz = max(1e-6, float(getattr(startup_args, "startup_init_hz", 12.0)))
            _print_direct_init_targets(start_arms, arm_targets)
            print(
                f"[Startup] interpolating first action over {steps} step(s) at {hz:.3g}Hz "
                f"(max_arm_joint_step={profile_max_arm_joint_step_rad(profile, startup_args):.4f}rad)"
            )
            for step in range(1, steps + 1):
                alpha = _smoothstep(float(step) / float(steps))
                arm_waypoint = _lerp_targets(start_arms, arm_targets, alpha)
                hand_waypoint = (
                    {side: value.copy() for side, value in hand_targets.items()}
                    if fixed_hand is not None and _is_fr3_profile(profile)
                    else _lerp_targets(start_hands, hand_targets, alpha)
                )
                statuses = executor.send_direct_joint_targets(arm_qpos=arm_waypoint, hand_qpos=hand_waypoint)
                if step == 1 or step == steps:
                    print(f"[Startup] direct init status: {';'.join(statuses[:8])}")
                _print_init_progress(step, steps)
                if step < steps:
                    time.sleep(1.0 / hz)
            _wait_for_direct_init_convergence(executor, arm_targets, hand_targets, startup_args, profile)
        sleep_s = max(0.0, float(getattr(args, "startup_init_sleep_s", 10.0)))
        print(f"[Startup] first action sent directly; sleeping {sleep_s:.1f}s before server-client inference.")
        _sleep_with_countdown(sleep_s)
        return 0

    path, actions = load_startup_actions(args, profile, mode)
    label = "full replay" if mode == "r" else "first action"
    print(
        f"[Startup] selected {path} | mode={label} | "
        f"actions={actions.shape[0]} | action_hz={float(getattr(args, 'action_hz', 12.0)):.3g}"
    )

    startup_args = _startup_hardware_args(args, profile)
    with HardwareActionExecutor(profile, startup_args) as executor:
        faults = executor.consume_server_result({"action": actions})
        if faults:
            print(f"[Startup] replay status: {';'.join(faults[:6])}")
        timeout_s = max(5.0, float(actions.shape[0]) / max(1e-6, float(getattr(startup_args, "action_hz", 12.0))) + 5.0)
        progress = _print_replay_progress if mode == "r" else None
        if not executor.wait_until_sent(
            int(actions.shape[0]),
            timeout_s=timeout_s,
            progress=progress,
        ):
            print(f"WARNING: startup replay did not finish within {timeout_s:.1f}s; stopping dispatcher.")
        else:
            print(f"[Startup] replay dispatched {int(actions.shape[0])} action(s).")

    if mode == "i":
        sleep_s = max(0.0, float(getattr(args, "startup_init_sleep_s", 10.0)))
        print(f"[Startup] first action sent; sleeping {sleep_s:.1f}s before server-client inference.")
        _sleep_with_countdown(sleep_s)
    else:
        print("[Startup] full replay finished; exiting.")
    return 0
