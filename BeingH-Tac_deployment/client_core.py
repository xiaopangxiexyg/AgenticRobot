from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import termios
import threading
import ast
import signal
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dataset_source import DatasetObservationSource
from image_transport import maybe_resize_payload_image
from live_source import (
    HardwareActionExecutor,
    LiveObservationSource,
    LiveSnapshot,
    add_live_args,
    format_fr3_dataset_gt_debug,
    format_fr3_policy_debug,
)
from paths import ensure_import_paths
from profiles import DeploymentProfile
from startup_replay import (
    _current_joint_targets,
    _lerp_targets,
    _smoothstep,
    add_startup_replay_args,
    bind_selected_episode,
    choose_episode,
    load_startup_actions,
    load_startup_direct_targets,
    run_startup_replay,
    wait_for_startup_key,
)


ensure_import_paths()

HISTORY_SIZE = 4
HISTORY_STRIDE = 4

REALMAN20_A_RECOVERY_ARM_TARGET = np.asarray(
    [-0.3268, 0.9280, 1.5185, -1.8084, 1.4316, 5.5148],
    dtype=np.float32,
)
REALMAN20_A_RECOVERY_HAND_TARGET = np.ones((6,), dtype=np.float32)
REALMAN20_A_RECOVERY_HAND_CLOSE_TARGET = np.asarray([0.1, 0.1, 0.1, 0.1, 1.0, 1.0], dtype=np.float32)
REALMAN20_A_RECOVERY_STEPS = 100
REALMAN20_A_RECOVERY_HZ = 12.0
REALMAN20_A_RECOVERY_TARGET_HOLD_S = 10.0
REALMAN20_A_RECOVERY_TARGET_TOLERANCE_RAD = 0.04
REALMAN20_H_INIT_DATASET = "/home/ai/Desktop/DexTeleop/datasets/2026-05-06_right_fixed_flip_hand"
REALMAN20_H_INIT_EPISODE = 0
REALMAN20_H_INIT_START_FRAME = 0
REALMAN20_H_INIT_MOVE_TIME_S = 3.0
REALMAN20_H_INIT_RIGHT_ARM_SLICE = slice(7, 14)
REALMAN20_H_INIT_RIGHT_HAND_SLICE = slice(20, 26)


def add_common_args(parser: argparse.ArgumentParser, profile: DeploymentProfile) -> None:
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=profile.default_port)
    parser.add_argument("--api-token", type=str, default=None)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--preflight-timeout-ms", type=int, default=5000)
    parser.add_argument("--no-server-preflight", dest="server_preflight", action="store_false", default=True)
    parser.add_argument("--source", choices=["dataset", "live"], default="dataset")
    parser.add_argument("--dataset-path", type=str, default=profile.default_dataset_path)
    parser.add_argument("--parquet-path", type=str, default="")
    parser.add_argument("--episode-index", type=int, default=-1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--dataset-frame-stride", type=int, default=int(getattr(profile, "default_dataset_frame_stride", 1)))
    parser.add_argument("--dataset-loop", action="store_true")
    parser.add_argument("--instruction", type=str, default=profile.default_instruction)
    parser.add_argument("--history-size", type=int, default=HISTORY_SIZE)
    parser.add_argument("--history-stride", type=int, default=HISTORY_STRIDE)
    parser.add_argument("--inference-delay", type=int, default=0)
    parser.add_argument("--max-requests", type=int, default=-1)
    parser.add_argument("--loop-sleep-s", type=float, default=0.1)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--fallback-image-height", type=int, default=480)
    parser.add_argument("--fallback-image-width", type=int, default=640)
    parser.add_argument("--transport-image-size", type=int, default=224)
    parser.add_argument("--no-rtc", dest="use_rtc", action="store_false", default=True)
    add_live_args(parser)
    if profile.data_config_name == "franka":
        parser.add_argument(
            "--fixed-hand",
            type=parse_fixed_hand,
            default=None,
            help=(
                "Override the Franka hand action sent to hardware with a fixed 6-D vector. "
                "Accepts comma-separated values or list/tuple/array text."
            ),
        )
    add_startup_replay_args(parser)


def parse_fixed_hand(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None

    literal_text = text
    for prefix in ("np.array(", "numpy.array(", "array("):
        if literal_text.startswith(prefix) and literal_text.endswith(")"):
            literal_text = literal_text[len(prefix) : -1].strip()
            break

    try:
        parsed = ast.literal_eval(literal_text)
    except (SyntaxError, ValueError):
        parsed = [piece.strip() for piece in text.replace(";", ",").split(",") if piece.strip()]

    arr = np.asarray(parsed, dtype=np.float32).reshape(-1)
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        raise argparse.ArgumentTypeError(
            "--fixed-hand must be a finite 6-D vector, for example '0,0,0,0,0,0' or '[0,0,0,0,0,0]'."
        )
    return tuple(float(x) for x in arr)


def make_client(args: argparse.Namespace):
    from BeingH.inference.beingh_service import BeingHInferenceClient

    client = BeingHInferenceClient(host=args.host, port=int(args.port), api_token=args.api_token)
    try:
        import zmq

        client.socket.setsockopt(zmq.RCVTIMEO, int(args.timeout_ms))
        client.socket.setsockopt(zmq.SNDTIMEO, int(args.timeout_ms))
        client.socket.setsockopt(zmq.LINGER, 0)
    except Exception as exc:
        print(f"WARNING: could not set ZMQ timeout options: {exc}")
    return client


def close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
            return
        except Exception:
            pass
    socket = getattr(client, "socket", None)
    if socket is not None:
        try:
            socket.close(0)
        except Exception:
            pass


class RequestDeadlineExceeded(TimeoutError):
    pass


def _request_alarm_handler(signum, frame) -> None:
    raise RequestDeadlineExceeded("client request deadline exceeded")


def get_action_with_deadline(client: Any, payload: Mapping[str, Any], args: argparse.Namespace) -> Any:
    timeout_s = max(0.0, float(getattr(args, "timeout_ms", 0)) / 1000.0)
    if timeout_s <= 0.0:
        return client.get_action(payload)

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _request_alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return client.get_action(payload)
    except RequestDeadlineExceeded as exc:
        raise TimeoutError(
            f"server {args.host}:{args.port} did not reply within {int(args.timeout_ms)} ms; "
            "check the server logs and SSH tunnel."
        ) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def call_endpoint_with_deadline(
    client: Any,
    endpoint: str,
    args: argparse.Namespace,
    *,
    data: Mapping[str, Any] | None = None,
    requires_input: bool = True,
    timeout_ms: int | None = None,
) -> Any:
    timeout_value = int(getattr(args, "timeout_ms", 0) if timeout_ms is None else timeout_ms)
    timeout_s = max(0.0, float(timeout_value) / 1000.0)
    if timeout_s <= 0.0:
        return client.call_endpoint(endpoint, data, requires_input=requires_input)

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _request_alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return client.call_endpoint(endpoint, data, requires_input=requires_input)
    except RequestDeadlineExceeded as exc:
        raise TimeoutError(
            f"server {args.host}:{args.port} did not reply to {endpoint!r} within {timeout_value} ms; "
            "check the SSH tunnel and server process."
        ) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def preflight_server(client: Any, args: argparse.Namespace) -> None:
    timeout_ms = int(getattr(args, "preflight_timeout_ms", 5000))
    response = call_endpoint_with_deadline(
        client,
        "ping",
        args,
        requires_input=False,
        timeout_ms=timeout_ms,
    )
    status = response.get("status") if isinstance(response, Mapping) else None
    if status != "ok":
        raise RuntimeError(f"Unexpected ping response from server: {response!r}")
    print(f"[Client] server preflight ok: ping replied within {timeout_ms} ms")


def maybe_preflight_server(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "server_preflight", True)):
        return
    client = make_client(args)
    try:
        preflight_server(client, args)
    finally:
        close_client(client)


def set_policy_request_in_flight(target: Any, in_flight: bool) -> None:
    setter = getattr(target, "set_policy_request_in_flight", None)
    if callable(setter):
        try:
            setter(bool(in_flight))
        except Exception:
            pass


def _sent_action_count(target: Any) -> int | None:
    value = getattr(target, "sent_actions", None)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _accepted_action_chunk_len(status: list[str]) -> int:
    for item in status:
        text = str(item)
        if not text.startswith("chunk="):
            continue
        try:
            return max(0, int(text.split("=", 1)[1]))
        except ValueError:
            return 0
    return 0


def wait_for_submitted_actions(target: Any, sent_before: int | None, status: list[str], args: argparse.Namespace) -> None:
    chunk_len = _accepted_action_chunk_len(status)
    if chunk_len <= 0:
        return
    if not bool(getattr(args, "auto_start", False)):
        return

    wait_until_sent = getattr(target, "wait_until_sent", None)
    if callable(wait_until_sent) and sent_before is not None:
        target_sent = sent_before + chunk_len
        timeout_s = (chunk_len / max(1e-6, float(getattr(args, "action_hz", 12.0)))) + max(
            2.0,
            float(getattr(args, "action_underrun_log_sec", 2.0)),
        )
        ok = wait_until_sent(target_sent, timeout_s=timeout_s)
        if not ok:
            print(
                "[Action] timed out waiting for submitted chunk to finish "
                f"(target_sent={target_sent}, chunk={chunk_len})"
            )
        return

    wait_until_idle = getattr(target, "wait_until_idle", None)
    if callable(wait_until_idle):
        timeout_s = (chunk_len / max(1e-6, float(getattr(args, "action_hz", 12.0)))) + max(
            2.0,
            float(getattr(args, "action_underrun_log_sec", 2.0)),
        )
        ok = wait_until_idle(timeout_s=timeout_s)
        if not ok:
            print(f"[Action] timed out waiting for submitted chunk queue to drain (chunk={chunk_len})")


def compact_shape(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"shape={value.shape}, dtype={value.dtype}"
    if hasattr(value, "shape"):
        return f"shape={tuple(value.shape)}"
    if isinstance(value, (list, tuple)):
        return f"len={len(value)}"
    return type(value).__name__


def print_payload(payload: Mapping[str, Any], prefix: str = "") -> None:
    for key, value in payload.items():
        if isinstance(value, dict):
            print(f"[Payload] {prefix}{key}: dict")
            print_payload(value, prefix=f"{prefix}{key}.")
        else:
            print(f"[Payload] {prefix}{key}: {compact_shape(value)}")


def build_source(profile: DeploymentProfile, args: argparse.Namespace):
    if args.source == "dataset":
        if not args.dataset_path:
            raise ValueError(f"{profile.data_config_name} has no default dataset; pass --dataset-path.")
        return DatasetObservationSource(
            profile=profile,
            dataset_path=args.dataset_path,
            parquet_path=args.parquet_path,
            episode_index=args.episode_index,
            start_frame=args.start_frame,
            history_size=args.history_size,
            history_stride=args.history_stride,
            fallback_image_height=args.fallback_image_height,
            fallback_image_width=args.fallback_image_width,
            frame_stride=args.dataset_frame_stride,
            loop=args.dataset_loop,
        )
    return LiveObservationSource(profile, args, args.history_size, args.history_stride)


def action_summary(result: Mapping[str, Any]) -> str:
    raw = np.asarray(result.get("action_raw", []), dtype=np.float32)
    unified = np.asarray(result.get("action_unified", []), dtype=np.float32)
    tactile = np.asarray(result.get("tactile_pred", []), dtype=np.float32)
    return f"action_raw={raw.shape} action_unified={unified.shape} tactile_pred={tactile.shape}"


def timing_summary(result: Mapping[str, Any], round_trip_s: float) -> str:
    pieces = [f"round_trip_ms={round_trip_s * 1000.0:.1f}"]
    timing = result.get("__timing")
    if isinstance(timing, Mapping):
        policy_s = timing.get("policy_inference_s")
        server_s = timing.get("server_process_s")
        decode_s = timing.get("server_decode_request_s")
        if isinstance(policy_s, (int, float)):
            pieces.append(f"policy_ms={float(policy_s) * 1000.0:.1f}")
        if isinstance(server_s, (int, float)):
            pieces.append(f"server_ms={float(server_s) * 1000.0:.1f}")
        if isinstance(decode_s, (int, float)):
            pieces.append(f"decode_ms={float(decode_s) * 1000.0:.1f}")
    return " ".join(pieces)


def format_request_error(exc: Exception, args: argparse.Namespace) -> str:
    try:
        import zmq

        if isinstance(exc, zmq.error.Again):
            return (
                f"server {args.host}:{args.port} did not reply within {int(args.timeout_ms)} ms; "
                "check that the Being-H-Tac server is running and reachable."
            )
    except Exception:
        pass
    text = f"{type(exc).__name__}: {exc}"
    if "Input type (float) and bias type (c10::BFloat16)" in text:
        return (
            text
            + " | server-side dtype mismatch in gripper_DM vision-tactile model. "
            "Restart run_server_gripper_DM.py after patching the model path that feeds Conv2d "
            "so image tensors are cast to the layer weight dtype before convolution."
        )
    return text


def _print_recovery_progress(step: int, total: int) -> None:
    total = max(1, int(total))
    step = max(0, min(int(step), total))
    width = 30
    filled = int(round(width * step / total))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[Action] realman20 recovery progress [{bar}] {step}/{total}", end="", flush=True)
    if step >= total:
        print("")


def _recovery_countdown(label: str, seconds: float, stop_event: threading.Event) -> None:
    remaining = int(np.ceil(max(0.0, float(seconds))))
    while remaining > 0 and not stop_event.is_set():
        print(f"[Action] realman20 recovery {label}: {remaining}s")
        stop_event.wait(1.0)
        remaining -= 1


def _action_countdown(label: str, seconds: float, stop_event: threading.Event) -> None:
    remaining = int(np.ceil(max(0.0, float(seconds))))
    while remaining > 0 and not stop_event.is_set():
        print(f"[Action] realman20 {label}: {remaining}s")
        stop_event.wait(1.0)
        remaining -= 1


def _finite_vector(value: Any, width: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (width,) or not np.all(np.isfinite(arr)):
        return None
    return arr.astype(np.float32, copy=True)


def _sync_action_chunks(profile: DeploymentProfile, action_target: Any | None) -> bool:
    return bool(
        profile.data_config_name in {"realman20", "realman22"}
        and action_target is not None
        and hasattr(action_target, "sent_actions")
        and hasattr(action_target, "wait_until_sent")
    )


def _chunk_len_from_faults(faults: list[str]) -> int | None:
    for item in faults:
        text = str(item)
        if not text.startswith("chunk="):
            continue
        try:
            value = int(text.split("=", 1)[1])
        except ValueError:
            return None
        return value if value > 0 else None
    return None


def _wait_for_submitted_chunk(
    profile: DeploymentProfile,
    action_target: Any | None,
    faults: list[str],
    sent_before: int | None,
    *,
    should_interrupt: Any | None = None,
) -> None:
    if not _sync_action_chunks(profile, action_target) or sent_before is None:
        return
    chunk_len = _chunk_len_from_faults(faults)
    if chunk_len is None:
        return
    action_hz = max(1e-6, float(getattr(getattr(action_target, "args", None), "action_hz", 12.0)))
    timeout_s = max(5.0, float(chunk_len) / action_hz + 5.0)
    target_sent = int(sent_before) + int(chunk_len)
    deadline = time.monotonic() + timeout_s
    while int(getattr(action_target, "sent_actions", sent_before)) < target_sent:
        if callable(should_interrupt) and should_interrupt():
            return
        if time.monotonic() >= deadline:
            current = int(getattr(action_target, "sent_actions", sent_before))
            print(
                "WARNING: action chunk did not finish before next inference wait timeout: "
                f"sent={current}/{target_sent} timeout={timeout_s:.1f}s"
            )
            return
        time.sleep(0.01)


@dataclass(frozen=True)
class Realman20HInitSequence:
    source_path: Path
    actions: np.ndarray
    right_arm_joint_targets: np.ndarray
    right_hand_joint_targets: np.ndarray


def _parquet_reader_candidates() -> list[str]:
    candidates = [
        os.environ.get("PARQUET_READER_PYTHON", "").strip(),
        sys.executable,
        "/home/ai/miniconda3/envs/grasp310_new/bin/python",
        "/home/ai/miniconda3/envs/beingh/bin/python",
        "/home/ai/zc/openpi/.venv/bin/python",
        "/home/ai/miniconda3/envs/ManipRepDeploy/bin/python",
        "/home/user/anaconda3/envs/lerobot/bin/python",
        "/home/user/anaconda3/envs/beingd1/bin/python",
        "/home/user/anaconda3/envs/lerobot032/bin/python",
    ]
    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).exists():
            out.append(candidate)
    return out


def _load_realman20_h_rows_with_helper(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    script = r"""
import json
import sys
from pathlib import Path

import pandas as pd

path = Path(sys.argv[1])
requested = json.loads(sys.argv[2])
df0 = pd.read_parquet(path)
columns = [column for column in requested if column in df0.columns]
if not columns:
    raise ValueError(f"None of the requested columns exist in {path}: {requested}")
df = df0[columns]
rows = []
for _, row in df.iterrows():
    item = {}
    for column in columns:
        value = row[column]
        if hasattr(value, "tolist"):
            value = value.tolist()
        item[column] = value
    rows.append(item)
print(json.dumps(rows))
"""
    last_error: Exception | None = None
    for python_bin in _parquet_reader_candidates():
        try:
            completed = subprocess.run(
                [python_bin, "-c", script, str(path), json.dumps(columns)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60.0,
            )
            return list(json.loads(completed.stdout))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "Failed to read realman20 H init parquet. Install pyarrow in the active "
        "environment or set PARQUET_READER_PYTHON to a Python with pandas/pyarrow."
    ) from last_error


def _load_realman20_h_rows(path: Path, columns: list[str]) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return _load_realman20_h_rows_with_helper(path, columns)

    schema = pq.read_schema(path)
    available = set(schema.names)
    selected = [column for column in columns if column in available]
    if not selected:
        raise ValueError(f"None of the requested columns exist in {path}: {columns}")
    return list(pq.read_table(path, columns=selected).to_pylist())


def _vector_or_raise(value: Any, width: int, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (int(width),) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite shape ({int(width)},), got {arr.shape}.")
    return arr.astype(np.float32, copy=True)


def _realman20_action_or_raise(value: Any, row_index: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (26,):
        raise ValueError(f"H init row {row_index} action must have shape (26,), got {arr.shape}.")
    required = np.concatenate(
        [
            arr[REALMAN20_H_INIT_RIGHT_ARM_SLICE],
            arr[REALMAN20_H_INIT_RIGHT_HAND_SLICE],
        ],
        axis=0,
    )
    if not np.all(np.isfinite(required)):
        raise ValueError(f"H init row {row_index} action has non-finite right arm/hand targets.")
    return arr.astype(np.float32, copy=True)


def _action_from_realman20_h_row(row: Mapping[str, Any], row_index: int) -> np.ndarray:
    if row.get("action") is not None:
        return _realman20_action_or_raise(row.get("action"), row_index)
    action = np.full((26,), np.nan, dtype=np.float32)
    if row.get("action.right_arm.pose") is not None:
        action[REALMAN20_H_INIT_RIGHT_ARM_SLICE] = _vector_or_raise(
            row.get("action.right_arm.pose"),
            7,
            f"H init row {row_index} action.right_arm.pose",
        )
    if row.get("action.right_hand.joint_target") is not None:
        action[REALMAN20_H_INIT_RIGHT_HAND_SLICE] = _vector_or_raise(
            row.get("action.right_hand.joint_target"),
            6,
            f"H init row {row_index} action.right_hand.joint_target",
        )
    if not np.any(np.isfinite(action)):
        raise ValueError(f"H init row {row_index} has no supported action columns.")
    return action


def load_realman20_h_init_sequence() -> Realman20HInitSequence:
    from teleop_runtime.deploy.replay import resolve_replay_parquet_path

    source_path, _ = resolve_replay_parquet_path(
        dataset_root=REALMAN20_H_INIT_DATASET,
        parquet_path=None,
        episode_index=REALMAN20_H_INIT_EPISODE,
    )
    rows = _load_realman20_h_rows(
        source_path,
        [
            "action",
            "action.right_arm.pose",
            "action.right_hand.joint_target",
            "action.replay.right_arm_joint_target",
            "action.replay.right_hand_joint_target",
        ],
    )
    if not rows:
        raise ValueError(f"realman20 H init episode contains no rows: {source_path}")

    actions: list[np.ndarray] = []
    arm_targets: list[np.ndarray] = []
    hand_targets: list[np.ndarray] = []
    start = max(0, int(REALMAN20_H_INIT_START_FRAME))
    for row_index, row in enumerate(rows[start:], start=start):
        actions.append(_action_from_realman20_h_row(row, row_index))
        arm_targets.append(
            _vector_or_raise(
                row.get("action.replay.right_arm_joint_target"),
                6,
                f"H init row {row_index} action.replay.right_arm_joint_target",
            )
        )
        hand_value = row.get("action.replay.right_hand_joint_target")
        if hand_value is None:
            hand_value = row.get("action.right_hand.joint_target")
        hand_targets.append(
            _vector_or_raise(
                hand_value,
                6,
                f"H init row {row_index} right hand target",
            )
        )

    return Realman20HInitSequence(
        source_path=source_path,
        actions=np.stack(actions, axis=0).astype(np.float32, copy=False),
        right_arm_joint_targets=np.stack(arm_targets, axis=0).astype(np.float32, copy=False),
        right_hand_joint_targets=np.stack(hand_targets, axis=0).astype(np.float32, copy=False),
    )


def _print_motion_progress(prefix: str, step: int, total: int) -> None:
    total = max(1, int(total))
    step = max(0, min(int(step), total))
    width = 30
    filled = int(round(width * step / total))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix} [{bar}] {step}/{total}", end="", flush=True)
    if step >= total:
        print("")


def _call_startup_loader_allowing_bound_parquet(args: argparse.Namespace, loader: Any, *loader_args: Any) -> Any:
    original_which_episode = getattr(args, "which_episode", None)
    has_bound_parquet = bool(str(getattr(args, "parquet_path", "") or "").strip())
    if has_bound_parquet and original_which_episode is not None:
        setattr(args, "which_episode", None)
    try:
        return loader(args, *loader_args)
    finally:
        if has_bound_parquet and original_which_episode is not None:
            setattr(args, "which_episode", original_which_episode)


def _realman20_h_gate_enabled(profile: DeploymentProfile, args: argparse.Namespace) -> bool:
    return bool(
        profile.data_config_name == "realman20"
        and (
            str(getattr(args, "source", "")) == "live"
            or bool(getattr(args, "send_actions", False))
        )
    )


def run_profile(profile: DeploymentProfile, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Being-H-Tac client for {profile.data_config_name}.")
    add_common_args(parser, profile)
    args = parser.parse_args(argv)
    if profile.data_config_name == "franka" and args.source == "dataset" and args.send_actions:
        setattr(args, "debug_fr3_actions", True)
    else:
        setattr(args, "debug_fr3_actions", False)

    previous_result: dict[str, Any] | None = None

    print(f"[Client] profile={profile.data_config_name} source={args.source} target={args.host}:{args.port}")
    print(f"[Client] HISTORY_SIZE={args.history_size} HISTORY_STRIDE={args.history_stride} inference_delay={args.inference_delay}")

    startup_mode: str | None = None
    if not _realman20_h_gate_enabled(profile, args):
        startup_mode = wait_for_startup_key(args)
        if startup_mode is not None:
            if startup_mode == "i":
                maybe_preflight_server(args)
            replay_status = run_startup_replay(profile, args, startup_mode)
            if startup_mode == "r":
                return replay_status
    elif bool(getattr(args, "startup_key_wait", True)):
        print("[Startup] realman20 live H gate enabled: press H -> A, H -> B, then i to run dataset init and start inference.")

    if startup_mode is None and args.source == "dataset" and args.which_episode is not None:
        bind_selected_episode(args, choose_episode(args))
        print(f"[Dataset] selected episode from --which-episode: {args.parquet_path}")

    if startup_mode is None:
        maybe_preflight_server(args)

    client = make_client(args)
    request_id = 0

    action_executor = HardwareActionExecutor(profile, args) if args.source == "dataset" and args.send_actions else None

    with build_source(profile, args) as source:
        executor_cm = action_executor if action_executor is not None else _NullContext()
        with executor_cm as executor:
            action_target = action_executor if action_executor is not None else (source if args.source == "live" else None)
            with Realman20ARecoveryController(profile, args, action_target) as recovery_controller:
                return _run_loop(
                    args=args,
                    client=client,
                    executor=executor,
                    action_target=action_target,
                    previous_result=previous_result,
                    recovery_controller=recovery_controller,
                    request_id=request_id,
                    source=source,
                )
    return 0


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class Realman20ARecoveryController:
    def __init__(self, profile: DeploymentProfile, args: argparse.Namespace, action_target: Any):
        self.profile = profile
        self.args = args
        self.executor = action_target
        self.enabled = bool(
            profile.data_config_name == "realman20"
            and action_target is not None
            and hasattr(action_target, "send_direct_joint_targets")
            and hasattr(action_target, "consume_server_result")
            and hasattr(action_target, "dispatcher")
        )
        self.h_gate_enabled = bool(self.enabled and _realman20_h_gate_enabled(profile, args))
        self._fd: int | None = None
        self._owns_fd = False
        self._old_settings: list[Any] | None = None
        self._stop_event = threading.Event()
        self._recovering_event = threading.Event()
        self._gate_ready_event = threading.Event()
        if not (self.h_gate_enabled and bool(getattr(args, "startup_key_wait", True))):
            self._gate_ready_event.set()
        self._recovery_lock = threading.Lock()
        self._operation_lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._h_init_sequence: Realman20HInitSequence | None = None
        self._h_press_count = 0
        self._h_state_b_ready = False
        self._startup_i_active = False
        self._exit_requested = False

    def __enter__(self) -> "Realman20ARecoveryController":
        if not self.enabled:
            return self
        self._open_keyboard()
        if self._fd is None:
            print("WARNING: realman20 keyboard controls disabled because no interactive terminal is available.")
            self.enabled = False
            self.h_gate_enabled = False
            self._gate_ready_event.set()
            return self
        self._thread = threading.Thread(target=self._keyboard_loop, name="realman20_runtime_keys", daemon=True)
        self._thread.start()
        if self.h_gate_enabled:
            if bool(getattr(self.args, "startup_key_wait", True)):
                self._gate_ready_event.clear()
            print("[Action] realman20 keys enabled: H -> state A, H -> state B, i -> dataset init/resume, a -> recovery.")
        else:
            print("[Action] realman20 a-key recovery enabled: press a to interrupt actions and move to recovery pose.")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def generation(self) -> int:
        with self._operation_lock:
            return int(self._generation)

    def is_recovering(self) -> bool:
        return self._recovering_event.is_set()

    def is_gate_ready(self) -> bool:
        return self._gate_ready_event.is_set()

    def should_pause_inference(self) -> bool:
        return bool(self.is_recovering() or not self.is_gate_ready())

    def is_startup_i_active(self) -> bool:
        return bool(self._startup_i_active)

    def exit_requested(self) -> bool:
        return bool(self._exit_requested)

    def wait_until_idle(self) -> None:
        while self.is_recovering():
            time.sleep(0.02)

    def wait_until_ready_for_inference(self) -> bool:
        while not self._stop_event.is_set() and not self._exit_requested:
            if not self.should_pause_inference():
                return True
            time.sleep(0.02)
        return False

    def submit_server_result(self, executor: Any, result: Mapping[str, Any], request_generation: int) -> tuple[bool, list[str]]:
        if not self.enabled:
            return True, executor.consume_server_result(result)
        with self._operation_lock:
            if self._generation != int(request_generation):
                return False, ["stale_after_recovery"]
            return True, executor.consume_server_result(result)

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close_keyboard()

    def _open_keyboard(self) -> None:
        try:
            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
                self._owns_fd = False
            else:
                self._fd = os.open("/dev/tty", os.O_RDONLY)
                self._owns_fd = True
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except (OSError, termios.error):
            if self._owns_fd and self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
            self._fd = None
            self._owns_fd = False
            self._old_settings = None

    def _close_keyboard(self) -> None:
        if self._fd is not None and self._old_settings is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            except (OSError, termios.error):
                pass
        if self._owns_fd and self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self._fd = None
        self._owns_fd = False
        self._old_settings = None

    def _keyboard_loop(self) -> None:
        assert self._fd is not None
        while not self._stop_event.is_set():
            try:
                readable, _, _ = select.select([self._fd], [], [], 0.05)
            except (OSError, ValueError):
                return
            if not readable:
                continue
            try:
                ch = os.read(self._fd, 1).decode(errors="ignore").lower()
            except OSError:
                return
            if ch == "a":
                self._start_recovery()
            elif ch == "h":
                self._start_h_gate_motion()
            elif ch == "i":
                self._start_startup_i_resume()
            elif ch == "r":
                self._start_startup_r_replay()
            elif ch in {"q", "\x03"}:
                print("\n[Action] realman20 key controller requested exit.")
                self._exit_requested = True
                self._gate_ready_event.set()

    def _start_recovery(self) -> None:
        if not self._recovery_lock.acquire(blocking=False):
            print("[Action] realman20 a-key recovery is already running; ignoring duplicate a.")
            return
        try:
            with self._operation_lock:
                self._generation += 1
                generation = self._generation
                self._recovering_event.set()
                print(f"\n[Action] realman20 a-key recovery triggered (generation={generation}); interrupting actions.")
                completed = False
                try:
                    completed = self._run_recovery_motion()
                except Exception as exc:
                    print(f"WARNING: realman20 a-key recovery failed: {type(exc).__name__}: {exc}")
                finally:
                    if not completed:
                        self._reset_last_action()
                    if not self._stop_event.is_set():
                        self.executor.dispatcher.start()
                    self._recovering_event.clear()
                    print("[Action] realman20 a-key recovery finished; inference may resume.")
        finally:
            self._recovery_lock.release()

    def _start_h_gate_motion(self) -> None:
        if not self.h_gate_enabled:
            return
        if not self._recovery_lock.acquire(blocking=False):
            print("[Action] realman20 H gate is already moving; ignoring duplicate H.")
            return
        try:
            with self._operation_lock:
                self._generation += 1
                generation = self._generation
                self._recovering_event.set()
                self._gate_ready_event.clear()
                self._h_press_count += 1
                self._h_state_b_ready = False
                target_label = "A" if self._h_press_count % 2 == 1 else "B"
                print(
                    f"\n[Action] realman20 H gate triggered: moving to state {target_label} "
                    f"(H count={self._h_press_count}, generation={generation})."
                )
                completed = False
                try:
                    if target_label == "A":
                        completed = self._run_h_state_a_motion()
                    else:
                        completed = self._run_h_state_b_motion()
                except Exception as exc:
                    print(f"WARNING: realman20 H gate failed: {type(exc).__name__}: {exc}")
                finally:
                    if not completed:
                        self._reset_last_action()
                    self._recovering_event.clear()
                    if completed:
                        if target_label == "B":
                            self._h_state_b_ready = True
                        next_hint = (
                            "press i to run dataset init and resume inference."
                            if target_label == "B"
                            else "press H again to reach state B."
                        )
                        print(
                            f"[Action] realman20 H gate reached state {target_label}; {next_hint}"
                        )
                    else:
                        print("[Action] realman20 H gate stopped before completion; press H or i when ready.")
        finally:
            self._recovery_lock.release()

    def _start_startup_i_resume(self) -> None:
        if not self.h_gate_enabled:
            return
        if self.is_gate_ready():
            print("[Action] realman20 i ignored: inference is already running.")
            return
        if not self._h_state_b_ready:
            print("[Action] realman20 i ignored: press H twice first so the robot reaches state B.")
            return
        if not self._recovery_lock.acquire(blocking=False):
            print("[Action] realman20 i ignored: H gate motion is still active.")
            return
        try:
            with self._operation_lock:
                self._generation += 1
                generation = self._generation
                self._recovering_event.set()
                self._startup_i_active = True
                print(f"\n[Action] realman20 i pressed: running dataset first-frame init (generation={generation}).")
                completed = False
                try:
                    completed = self._run_startup_i_motion()
                except Exception as exc:
                    print(f"WARNING: realman20 startup i init failed: {type(exc).__name__}: {exc}")
                finally:
                    if completed:
                        if not self._stop_event.is_set():
                            self.executor.dispatcher.start()
                        self._gate_ready_event.set()
                        self._h_state_b_ready = False
                        print("[Action] realman20 startup i init finished; inference may run.")
                    else:
                        self._reset_last_action()
                        print("[Action] realman20 startup i init did not finish; inference remains paused.")
                    self._startup_i_active = False
                    self._recovering_event.clear()
        finally:
            self._recovery_lock.release()

    def _start_startup_r_replay(self) -> None:
        if not self.h_gate_enabled:
            return
        if self.is_gate_ready():
            print("[Action] realman20 r ignored: inference is already running.")
            return
        if not self._h_state_b_ready:
            print("[Action] realman20 r ignored: press H twice first so the robot reaches state B.")
            return
        if not self._recovery_lock.acquire(blocking=False):
            print("[Action] realman20 r ignored: H gate motion is still active.")
            return
        try:
            with self._operation_lock:
                self._generation += 1
                generation = self._generation
                self._recovering_event.set()
                print(f"\n[Action] realman20 r pressed: running startup replay then exit (generation={generation}).")
                completed = False
                try:
                    completed = self._run_startup_r_motion()
                except Exception as exc:
                    print(f"WARNING: realman20 startup r replay failed: {type(exc).__name__}: {exc}")
                    self._recovering_event.clear()
                    return
                if completed:
                    self._exit_requested = True
                    self._gate_ready_event.set()
                self._recovering_event.clear()
        finally:
            self._recovery_lock.release()

    def _run_recovery_motion(self) -> bool:
        self.executor.dispatcher.stop()
        arm_targets = {"right": REALMAN20_A_RECOVERY_ARM_TARGET}
        hand_targets = {"right": REALMAN20_A_RECOVERY_HAND_TARGET}
        start_arms, start_hands = self._current_joint_targets()
        start_arm = start_arms.get("right")
        if start_arm is not None:
            delta = float(np.max(np.abs(np.asarray(start_arm, dtype=np.float32) - REALMAN20_A_RECOVERY_ARM_TARGET)))
            print(
                "[Action] realman20 recovery right arm "
                f"current={np.array2string(start_arm, precision=4, separator=', ')} "
                f"target={np.array2string(REALMAN20_A_RECOVERY_ARM_TARGET, precision=4, separator=', ')} "
                f"max_delta={delta:.4f}rad"
            )
        else:
            print(
                "[Action] realman20 recovery right arm "
                f"target={np.array2string(REALMAN20_A_RECOVERY_ARM_TARGET, precision=4, separator=', ')} "
                "current=unavailable"
            )
        print(
            f"[Action] realman20 recovery interpolating over {REALMAN20_A_RECOVERY_STEPS} step(s) "
            f"at {REALMAN20_A_RECOVERY_HZ:.3g}Hz (~{REALMAN20_A_RECOVERY_STEPS / REALMAN20_A_RECOVERY_HZ:.1f}s)"
        )
        self._send_recovery_interpolation(
            start_arms=start_arms,
            start_hands=start_hands,
            arm_targets=arm_targets,
            hand_targets=hand_targets,
        )
        if self._stop_event.is_set():
            return False

        self._wait_for_recovery_target(
            label="settling first target",
            arm_targets=arm_targets,
            hand_targets=hand_targets,
            tolerance_rad=REALMAN20_A_RECOVERY_TARGET_TOLERANCE_RAD,
        )
        if self._stop_event.is_set():
            return False

        closed_hand_targets = {"right": REALMAN20_A_RECOVERY_HAND_CLOSE_TARGET}
        # _ = input("按下回车键继续，关闭手指并进入下一轮推理...")
        statuses = self.executor.send_direct_joint_targets(arm_qpos=arm_targets, hand_qpos=closed_hand_targets)
        print(f"[Action] realman20 recovery close hand status: {';'.join(statuses[:8])}")
        if self._stop_event.is_set():
            return False

        _recovery_countdown(
            "holding first target with closed hand",
            REALMAN20_A_RECOVERY_TARGET_HOLD_S,
            self._stop_event,
        )
        if self._stop_event.is_set():
            return False

        self._set_recovery_hand_last_action(hand_targets=closed_hand_targets)
        return True

    def _h_sequence(self) -> Realman20HInitSequence:
        if self._h_init_sequence is None:
            self._h_init_sequence = load_realman20_h_init_sequence()
            print(
                f"[Action] realman20 H qpos replay loaded {self._h_init_sequence.actions.shape[0]} frame(s) "
                f"from {self._h_init_sequence.source_path}"
            )
        return self._h_init_sequence

    def _run_h_state_a_motion(self) -> bool:
        self.executor.dispatcher.stop()
        sequence = self._h_sequence()
        arm_targets = {"right": sequence.right_arm_joint_targets[0]}
        hand_targets = {"right": sequence.right_hand_joint_targets[0]}
        steps = max(1, int(round(float(REALMAN20_H_INIT_MOVE_TIME_S) * float(REALMAN20_A_RECOVERY_HZ))))
        start_arms, start_hands = self._current_joint_targets()
        self._print_direct_arm_target("H state A", start_arms, arm_targets)
        print(
            f"[Action] realman20 H qpos state A interpolating over {steps} step(s) "
            f"at {REALMAN20_A_RECOVERY_HZ:.3g}Hz (~{steps / REALMAN20_A_RECOVERY_HZ:.1f}s)"
        )
        self._send_direct_interpolation(
            label="H state A",
            start_arms=start_arms,
            start_hands=start_hands,
            arm_targets=arm_targets,
            hand_targets=hand_targets,
            steps=steps,
            hz=REALMAN20_A_RECOVERY_HZ,
        )
        if self._stop_event.is_set():
            return False
        self._wait_for_recovery_target(
            label="H state A settling",
            arm_targets=arm_targets,
            hand_targets=hand_targets,
            tolerance_rad=REALMAN20_A_RECOVERY_TARGET_TOLERANCE_RAD,
        )
        self._set_h_last_action(sequence.actions[0])
        return not self._stop_event.is_set()

    def _run_h_state_b_motion(self) -> bool:
        self.executor.dispatcher.stop()
        sequence = self._h_sequence()
        total = int(sequence.actions.shape[0])
        replay_hz = max(1e-6, float(getattr(self.args, "hz", 30.0)))
        print(
            f"[Action] realman20 H qpos state B replaying {total} frame(s) "
            f"at {replay_hz:.3g}Hz (~{total / replay_hz:.1f}s)"
        )
        for index in range(total):
            if self._stop_event.is_set():
                return False
            arm_targets = {"right": sequence.right_arm_joint_targets[index]}
            hand_targets = {"right": sequence.right_hand_joint_targets[index]}
            statuses = self.executor.send_direct_joint_targets(arm_qpos=arm_targets, hand_qpos=hand_targets)
            if index in {0, total - 1}:
                print(f"[Action] realman20 H qpos state B direct status: {';'.join(statuses[:8])}")
            _print_motion_progress("[Action] realman20 H qpos state B progress", index + 1, total)
            if index + 1 < total:
                self._stop_event.wait(1.0 / replay_hz)
        self._wait_for_recovery_target(
            label="H state B settling",
            arm_targets={"right": sequence.right_arm_joint_targets[-1]},
            hand_targets={"right": sequence.right_hand_joint_targets[-1]},
            tolerance_rad=REALMAN20_A_RECOVERY_TARGET_TOLERANCE_RAD,
        )
        self._set_h_last_action(sequence.actions[-1])
        return not self._stop_event.is_set()

    def _run_startup_i_motion(self) -> bool:
        self.executor.dispatcher.stop()
        path, arm_targets, hand_targets = _call_startup_loader_allowing_bound_parquet(
            self.args,
            load_startup_direct_targets,
            self.profile,
        )
        print(
            f"[Action] realman20 startup i selected {path} | "
            f"arms={sorted(arm_targets)} hands={sorted(hand_targets)}"
        )
        start_arms, start_hands = self._current_joint_targets()
        steps = max(1, int(getattr(self.args, "startup_init_steps", 60)))
        hz = max(1e-6, float(getattr(self.args, "startup_init_hz", 12.0)))
        self._print_direct_arm_target("startup i", start_arms, arm_targets)
        print(
            f"[Action] realman20 startup i interpolating first action over {steps} step(s) "
            f"at {hz:.3g}Hz (~{steps / hz:.1f}s)"
        )
        self._send_direct_interpolation(
            label="startup i",
            start_arms=start_arms,
            start_hands=start_hands,
            arm_targets=arm_targets,
            hand_targets=hand_targets,
            steps=steps,
            hz=hz,
        )
        if self._stop_event.is_set():
            return False
        self._set_recovery_hand_last_action(hand_targets=hand_targets)
        sleep_s = max(0.0, float(getattr(self.args, "startup_init_sleep_s", 10.0)))
        if sleep_s > 0.0:
            _action_countdown("startup i hold before inference", sleep_s, self._stop_event)
        return True

    def _run_startup_r_motion(self) -> bool:
        if not self._run_startup_i_motion():
            return False
        path, actions = _call_startup_loader_allowing_bound_parquet(
            self.args,
            load_startup_actions,
            self.profile,
            "r",
        )
        print(
            f"[Action] realman20 startup r replay selected {path} | "
            f"actions={actions.shape[0]} | action_hz={float(getattr(self.args, 'action_hz', 12.0)):.3g}"
        )
        self.executor.dispatcher.start()
        faults = self.executor.consume_server_result({"action": actions})
        if faults:
            print(f"[Action] realman20 startup r replay status: {';'.join(faults[:6])}")
        timeout_s = max(5.0, float(actions.shape[0]) / max(1e-6, float(getattr(self.args, "action_hz", 12.0))) + 5.0)
        sent_before = int(getattr(self.executor, "sent_actions", 0))
        return self.executor.wait_until_sent(
            sent_before + int(actions.shape[0]),
            timeout_s=timeout_s,
            progress=lambda sent, total: _print_motion_progress(
                "[Action] realman20 startup r replay progress",
                max(0, int(sent) - sent_before),
                int(actions.shape[0]),
            ),
        )

    def _print_direct_arm_target(
        self,
        label: str,
        start_arms: Mapping[str, np.ndarray],
        arm_targets: Mapping[str, np.ndarray],
    ) -> None:
        for side, target in sorted(arm_targets.items()):
            start = start_arms.get(side)
            target_arr = np.asarray(target, dtype=np.float32).reshape(-1)
            if start is None:
                print(
                    f"[Action] realman20 {label} {side} arm "
                    f"target={np.array2string(target_arr, precision=4, separator=', ')} current=unavailable"
                )
                continue
            start_arr = np.asarray(start, dtype=np.float32).reshape(-1)
            delta = float(np.max(np.abs(start_arr - target_arr)))
            print(
                f"[Action] realman20 {label} {side} arm "
                f"current={np.array2string(start_arr, precision=4, separator=', ')} "
                f"target={np.array2string(target_arr, precision=4, separator=', ')} "
                f"max_delta={delta:.4f}rad"
            )

    def _send_direct_interpolation(
        self,
        *,
        label: str,
        start_arms: Mapping[str, np.ndarray],
        start_hands: Mapping[str, np.ndarray],
        arm_targets: Mapping[str, np.ndarray],
        hand_targets: Mapping[str, np.ndarray],
        steps: int,
        hz: float,
    ) -> None:
        total = max(1, int(steps))
        wait_s = 1.0 / max(1e-6, float(hz))
        for step in range(1, total + 1):
            if self._stop_event.is_set():
                return
            alpha = _smoothstep(float(step) / float(total))
            arm_waypoint = _lerp_targets(dict(start_arms), dict(arm_targets), alpha)
            hand_waypoint = _lerp_targets(dict(start_hands), dict(hand_targets), alpha)
            statuses = self.executor.send_direct_joint_targets(arm_qpos=arm_waypoint, hand_qpos=hand_waypoint)
            if step == 1 or step == total:
                print(f"[Action] realman20 {label} direct status: {';'.join(statuses[:8])}")
            _print_motion_progress(f"[Action] realman20 {label} progress", step, total)
            if step < total:
                self._stop_event.wait(wait_s)

    def _set_h_last_action(self, action_value: np.ndarray) -> None:
        action = np.asarray(action_value, dtype=np.float32).reshape(26).copy()
        setter = getattr(self.executor, "_set_last_action", None)
        if callable(setter):
            setter(action)
        elif hasattr(self.executor, "last_action"):
            self.executor.last_action = action

    def _send_recovery_interpolation(
        self,
        *,
        start_arms: dict[str, np.ndarray],
        start_hands: dict[str, np.ndarray],
        arm_targets: dict[str, np.ndarray],
        hand_targets: dict[str, np.ndarray],
    ) -> None:
        for step in range(1, REALMAN20_A_RECOVERY_STEPS + 1):
            if self._stop_event.is_set():
                return
            alpha = _smoothstep(float(step) / float(REALMAN20_A_RECOVERY_STEPS))
            arm_waypoint = _lerp_targets(start_arms, arm_targets, alpha)
            hand_waypoint = _lerp_targets(start_hands, hand_targets, alpha)
            statuses = self.executor.send_direct_joint_targets(arm_qpos=arm_waypoint, hand_qpos=hand_waypoint)
            if step == 1 or step == REALMAN20_A_RECOVERY_STEPS:
                print(f"[Action] realman20 recovery direct status: {';'.join(statuses[:8])}")
            _print_recovery_progress(step, REALMAN20_A_RECOVERY_STEPS)
            if step < REALMAN20_A_RECOVERY_STEPS:
                self._stop_event.wait(1.0 / REALMAN20_A_RECOVERY_HZ)

    def _wait_for_recovery_target(
        self,
        *,
        label: str,
        arm_targets: dict[str, np.ndarray],
        hand_targets: dict[str, np.ndarray],
        tolerance_rad: float,
    ) -> None:
        wait_s = 1.0 / REALMAN20_A_RECOVERY_HZ
        target = arm_targets.get("right")
        if target is None:
            return
        target_arr = np.asarray(target, dtype=np.float32).reshape(-1)
        while not self._stop_event.is_set():
            current_arms, _ = self._current_joint_targets()
            current = current_arms.get("right")
            if current is None:
                print(
                    f"[Action] realman20 recovery {label}: current=unavailable "
                    f"target={np.array2string(target_arr, precision=4, separator=', ')}"
                )
            else:
                current_arr = np.asarray(current, dtype=np.float32).reshape(-1)
                diff = current_arr - target_arr
                max_delta = float(np.max(np.abs(diff)))
                print(
                    f"[Action] realman20 recovery {label}: "
                    f"current={np.array2string(current_arr, precision=4, separator=', ')} "
                    f"target={np.array2string(target_arr, precision=4, separator=', ')} "
                    f"diff={np.array2string(diff, precision=4, separator=', ')} "
                    f"max_delta={max_delta:.4f}rad"
                )
                if max_delta <= float(tolerance_rad):
                    print(
                        f"[Action] realman20 recovery {label}: reached within "
                        f"{float(tolerance_rad):.4f}rad"
                    )
                    return
            statuses = self.executor.send_direct_joint_targets(arm_qpos=arm_targets, hand_qpos=hand_targets)
            print(f"[Action] realman20 recovery {label} hold status: {';'.join(statuses[:8])}")
            self._stop_event.wait(wait_s)

    def _reset_last_action(self) -> None:
        reset = np.full((26,), np.nan, dtype=np.float32)
        setter = getattr(self.executor, "_set_last_action", None)
        if callable(setter):
            setter(reset)
        elif hasattr(self.executor, "last_action"):
            self.executor.last_action = reset

    def _set_recovery_hand_last_action(self, *, hand_targets: Mapping[str, np.ndarray]) -> None:
        action = np.full((26,), np.nan, dtype=np.float32)
        if "left" in hand_targets:
            action[14:20] = np.asarray(hand_targets["left"], dtype=np.float32).reshape(6)
        if "right" in hand_targets:
            action[20:26] = np.asarray(hand_targets["right"], dtype=np.float32).reshape(6)
        setter = getattr(self.executor, "_set_last_action", None)
        if callable(setter):
            setter(action)
        elif hasattr(self.executor, "last_action"):
            self.executor.last_action = action

    def _current_joint_targets(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if isinstance(self.executor, LiveObservationSource):
            build_observation = self.executor._build_observation
            try:
                observation = build_observation(time.time())
            except TypeError:
                observation = build_observation()
            arm_current: dict[str, np.ndarray] = {}
            hand_current: dict[str, np.ndarray] = {}
            for side, state in observation.arm_states.items():
                if not isinstance(state, Mapping):
                    continue
                qpos = state.get("current_qpos")
                if qpos is None:
                    qpos = state.get("qpos")
                arr = _finite_vector(qpos, 6)
                if arr is not None:
                    arm_current[str(side)] = arr
            for side, state in observation.hand_states.items():
                if not isinstance(state, Mapping):
                    continue
                qpos = state.get("joint_pos")
                if qpos is None:
                    qpos = state.get("last_sent_command")
                arr = _finite_vector(qpos, 6)
                if arr is not None:
                    hand_current[str(side)] = arr
            return arm_current, hand_current
        return _current_joint_targets(self.executor)


def _run_loop(
    *,
    args: argparse.Namespace,
    client: Any,
    executor: Any,
    action_target: Any | None,
    previous_result: dict[str, Any] | None,
    recovery_controller: Realman20ARecoveryController,
    request_id: int,
    source: Any,
) -> int:
    observed_recovery_generation = recovery_controller.generation
    while args.max_requests < 0 or request_id < args.max_requests:
        if not recovery_controller.wait_until_ready_for_inference():
            break
        if recovery_controller.generation != observed_recovery_generation:
            previous_result = None
            observed_recovery_generation = recovery_controller.generation
        request_recovery_generation = recovery_controller.generation
        try:
            item = source.next_payload_base()
        except StopIteration as exc:
            print(f"[Client] stopping: {exc}")
            break
        observation = None
        if isinstance(item, LiveSnapshot):
            payload = item.payload_base
            observation = item.observation
            if not recovery_controller.is_startup_i_active():
                source.remember_observation(observation)
        else:
            payload = item
        dataset_row_index = payload.pop("__dataset_row_index", None)
        dataset_gt_action = payload.pop("__dataset_gt_action", None)
        payload["language"] = args.instruction
        payload["inference_delay"] = int(args.inference_delay)
        payload["__client_send_ts"] = time.time()
        if args.use_rtc and previous_result and previous_result.get("action_unified") is not None:
            payload["prev_chunk"] = np.asarray(previous_result["action_unified"], dtype=np.float32)
        if observation is not None and np.all(np.isfinite(observation.last_action)):
            payload["last_executed_action"] = observation.last_action.astype(np.float32)
        maybe_resize_payload_image(payload, int(args.transport_image_size))

        if args.print_every > 0 and request_id % args.print_every == 0:
            if dataset_row_index is None:
                print(f"[Client] request={request_id}")
            else:
                print(f"[Client] request={request_id} dataset_row={dataset_row_index}")
            print_payload(payload)

        try:
            if (
                recovery_controller.generation != request_recovery_generation
                or recovery_controller.should_pause_inference()
            ):
                print("[Action] skipped inference request built before realman20 a-key recovery finished.")
                previous_result = None
                observed_recovery_generation = recovery_controller.generation
                continue
            request_started_at = time.time()
            if executor is not None:
                set_policy_request_in_flight(executor, True)
            previous_result = get_action_with_deadline(client, payload, args)
            round_trip_s = time.time() - request_started_at
            if executor is not None:
                set_policy_request_in_flight(executor, False)
            request_label = f"request={request_id}"
            if dataset_row_index is not None:
                request_label += f" dataset_row={dataset_row_index}"
            print(f"[Client] {request_label} ok | {action_summary(previous_result)} | {timing_summary(previous_result, round_trip_s)}")
            if bool(getattr(args, "debug_fr3_actions", False)):
                for line in format_fr3_policy_debug(previous_result):
                    print(line)
                for line in format_fr3_dataset_gt_debug(dataset_gt_action):
                    print(line)
            if action_target is not None:
                sent_before = _sent_action_count(action_target)
                if recovery_controller.enabled:
                    submitted, faults = recovery_controller.submit_server_result(
                        action_target,
                        previous_result,
                        request_recovery_generation,
                    )
                    if not submitted:
                        print("[Action] ignored stale inference result produced before realman20 a-key recovery finished.")
                        previous_result = None
                        observed_recovery_generation = recovery_controller.generation
                        continue
                else:
                    faults = action_target.consume_server_result(previous_result)
                if faults:
                    prefix = "[Client]" if args.source == "live" else "[Action]"
                    print(f"{prefix} action faults/status: {';'.join(faults[:6])}")
                if recovery_controller.enabled:
                    _wait_for_submitted_chunk(
                        recovery_controller.profile,
                        action_target,
                        faults,
                        sent_before,
                        should_interrupt=recovery_controller.should_pause_inference,
                    )
                else:
                    wait_for_submitted_actions(action_target, sent_before, faults, args)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if executor is not None:
                set_policy_request_in_flight(executor, False)
            close_client(client)
            print(f"WARNING: request {request_id} failed: {format_request_error(exc, args)}")
            if executor is not None:
                print("[Client] stopping because --send-actions is active and no server action was received.")
                break
            previous_result = None
            client = make_client(args)

        request_id += 1
        if args.loop_sleep_s > 0:
            time.sleep(float(args.loop_sleep_s))
    if recovery_controller.exit_requested():
        print("\n正在退出 BeingH-Tac client Quitting the BeingH-Tac client...\n", flush=True)
    return 0
