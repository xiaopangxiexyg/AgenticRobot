from __future__ import annotations

import argparse
import random
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any

import numpy as np

from live_source import HardwareActionExecutor
from paths import ensure_import_paths
from profiles import DeploymentProfile


ensure_import_paths()

from teleop_runtime.deploy.replay import load_replay_actions_from_parquet  # noqa: E402


def add_startup_replay_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-startup-key-wait", dest="startup_key_wait", action="store_false", default=True)
    parser.add_argument("--startup-init-sleep-s", type=float, default=10.0)
    parser.add_argument("--startup-replay-action-column", type=str, default="action")
    parser.add_argument("--startup-replay-start-frame", type=int, default=0)
    parser.add_argument("--startup-replay-seed", type=int, default=None)
    parser.add_argument("--which-episode", type=int, default=None)
    parser.add_argument("--startup-init-steps", type=int, default=24)
    parser.add_argument("--startup-init-hz", type=float, default=12.0)


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


def load_startup_actions(args: argparse.Namespace, mode: str) -> tuple[Path, np.ndarray]:
    path = choose_episode(args)
    bind_selected_episode(args, path)
    start = max(0, int(getattr(args, "startup_replay_start_frame", 0)))
    end = start + 1 if mode == "i" else None
    actions = load_replay_actions_from_parquet(
        path,
        action_column=str(getattr(args, "startup_replay_action_column", "action") or "action"),
        start_index=start,
        end_index=end,
    )
    if mode == "i":
        actions = actions[:1]
    return path, np.asarray(actions, dtype=np.float32)


def load_startup_direct_targets(args: argparse.Namespace, profile: DeploymentProfile) -> tuple[Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    path = choose_episode(args)
    bind_selected_episode(args, path)
    frame_index = max(0, int(getattr(args, "startup_replay_start_frame", 0)))
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Direct startup init requires pyarrow in the active environment.") from exc

    columns = []
    for side in profile.required_arms:
        columns.append(f"action.replay.{side}_arm_joint_target")
    for side in profile.required_hands:
        columns.append(f"action.replay.{side}_hand_joint_target")
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
    for side in profile.required_arms:
        value = row.get(f"action.replay.{side}_arm_joint_target")
        if value is not None:
            arm_targets[side] = np.asarray(value, dtype=np.float32).reshape(6)
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


def _current_joint_targets(executor: HardwareActionExecutor) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    observation = executor._build_observation()
    arm_current: dict[str, np.ndarray] = {}
    hand_current: dict[str, np.ndarray] = {}
    for side, state in observation.arm_states.items():
        qpos = _finite_6(state.get("current_qpos"))
        if qpos is not None:
            arm_current[side] = qpos
    for side, state in observation.hand_states.items():
        qpos = _finite_6(state.get("joint_pos"))
        if qpos is None:
            qpos = _finite_6(state.get("last_sent_command"))
        if qpos is not None:
            hand_current[side] = qpos
    return arm_current, hand_current


def _lerp_targets(
    current: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for side, target_value in target.items():
        target_arr = np.asarray(target_value, dtype=np.float32).reshape(6)
        start_arr = current.get(side)
        if start_arr is None:
            out[side] = target_arr
        else:
            out[side] = (start_arr + (target_arr - start_arr) * float(alpha)).astype(np.float32)
    return out


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


def run_startup_replay(profile: DeploymentProfile, args: argparse.Namespace, mode: str) -> int:
    if mode not in {"r", "i"}:
        return 0
    if not profile.supports_live_rm65:
        raise ValueError(f"{profile.data_config_name} startup replay is not supported by DexTeleop RM65 backends.")

    if mode == "i":
        path, arm_targets, hand_targets = load_startup_direct_targets(args, profile)
        print(
            f"[Startup] selected {path} | mode=first action direct init | "
            f"arms={sorted(arm_targets)} hands={sorted(hand_targets)}"
        )
        with HardwareActionExecutor(profile, args) as executor:
            start_arms, start_hands = _current_joint_targets(executor)
            steps = max(1, int(getattr(args, "startup_init_steps", 24)))
            hz = max(1e-6, float(getattr(args, "startup_init_hz", 12.0)))
            print(f"[Startup] interpolating first action over {steps} step(s) at {hz:.3g}Hz")
            for step in range(1, steps + 1):
                alpha = _smoothstep(float(step) / float(steps))
                arm_waypoint = _lerp_targets(start_arms, arm_targets, alpha)
                hand_waypoint = _lerp_targets(start_hands, hand_targets, alpha)
                statuses = executor.send_direct_joint_targets(arm_qpos=arm_waypoint, hand_qpos=hand_waypoint)
                if step == 1 or step == steps:
                    print(f"[Startup] direct init status: {';'.join(statuses[:8])}")
                _print_init_progress(step, steps)
                if step < steps:
                    time.sleep(1.0 / hz)
        sleep_s = max(0.0, float(getattr(args, "startup_init_sleep_s", 10.0)))
        print(f"[Startup] first action sent directly; sleeping {sleep_s:.1f}s before server-client inference.")
        _sleep_with_countdown(sleep_s)
        return 0

    path, actions = load_startup_actions(args, mode)
    label = "full replay" if mode == "r" else "first action"
    print(
        f"[Startup] selected {path} | mode={label} | "
        f"actions={actions.shape[0]} | action_hz={float(getattr(args, 'action_hz', 12.0)):.3g}"
    )

    with HardwareActionExecutor(profile, args) as executor:
        faults = executor.consume_server_result({"action": actions})
        if faults:
            print(f"[Startup] replay status: {';'.join(faults[:6])}")
        timeout_s = max(5.0, float(actions.shape[0]) / max(1e-6, float(getattr(args, "action_hz", 12.0))) + 5.0)
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
