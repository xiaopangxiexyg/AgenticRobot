from __future__ import annotations

import argparse
import time
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
from startup_replay import add_startup_replay_args, bind_selected_episode, choose_episode, run_startup_replay, wait_for_startup_key


ensure_import_paths()

HISTORY_SIZE = 4
HISTORY_STRIDE = 4


def add_common_args(parser: argparse.ArgumentParser, profile: DeploymentProfile) -> None:
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=profile.default_port)
    parser.add_argument("--api-token", type=str, default=None)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--source", choices=["dataset", "live"], default="dataset")
    parser.add_argument("--dataset-path", type=str, default=profile.default_dataset_path)
    parser.add_argument("--parquet-path", type=str, default="")
    parser.add_argument("--episode-index", type=int, default=-1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--dataset-frame-stride", type=int, default=1)
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
    add_startup_replay_args(parser)


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
    return f"{type(exc).__name__}: {exc}"


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

    startup_mode = wait_for_startup_key(args)
    if startup_mode is not None:
        replay_status = run_startup_replay(profile, args, startup_mode)
        if startup_mode == "r":
            return replay_status
    elif args.source == "dataset" and args.which_episode is not None:
        bind_selected_episode(args, choose_episode(args))
        print(f"[Dataset] selected episode from --which-episode: {args.parquet_path}")

    client = make_client(args)
    request_id = 0

    action_executor = HardwareActionExecutor(profile, args) if args.source == "dataset" and args.send_actions else None

    with build_source(profile, args) as source:
        executor_cm = action_executor if action_executor is not None else _NullContext()
        with executor_cm as executor:
            return _run_loop(
                args=args,
                client=client,
                executor=executor,
                previous_result=previous_result,
                request_id=request_id,
                source=source,
            )
    return 0


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _run_loop(
    *,
    args: argparse.Namespace,
    client: Any,
    executor: Any,
    previous_result: dict[str, Any] | None,
    request_id: int,
    source: Any,
) -> int:
    while args.max_requests < 0 or request_id < args.max_requests:
        try:
            item = source.next_payload_base()
        except StopIteration as exc:
            print(f"[Client] stopping: {exc}")
            break
        observation = None
        if isinstance(item, LiveSnapshot):
            payload = item.payload_base
            observation = item.observation
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
            request_started_at = time.time()
            previous_result = client.get_action(payload)
            round_trip_s = time.time() - request_started_at
            request_label = f"request={request_id}"
            if dataset_row_index is not None:
                request_label += f" dataset_row={dataset_row_index}"
            print(f"[Client] {request_label} ok | {action_summary(previous_result)} | {timing_summary(previous_result, round_trip_s)}")
            if bool(getattr(args, "debug_fr3_actions", False)):
                for line in format_fr3_policy_debug(previous_result):
                    print(line)
                for line in format_fr3_dataset_gt_debug(dataset_gt_action):
                    print(line)
            if args.source == "live":
                faults = source.consume_server_result(previous_result)
                if faults:
                    print(f"[Client] action faults/status: {';'.join(faults[:6])}")
            elif executor is not None:
                faults = executor.consume_server_result(previous_result)
                if faults:
                    print(f"[Action] faults/status: {';'.join(faults[:6])}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"WARNING: request {request_id} failed: {format_request_error(exc, args)}")
            previous_result = None
            client = make_client(args)

        request_id += 1
        if args.loop_sleep_s > 0:
            time.sleep(float(args.loop_sleep_s))
    return 0
