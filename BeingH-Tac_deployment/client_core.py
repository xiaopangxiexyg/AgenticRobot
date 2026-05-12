from __future__ import annotations

import argparse
import ast
import signal
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
    parser.add_argument("--preflight-timeout-ms", type=int, default=5000)
    parser.add_argument("--no-server-preflight", dest="server_preflight", action="store_false", default=True)
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

    client = make_client(args)
    if bool(getattr(args, "server_preflight", True)):
        try:
            preflight_server(client, args)
        except Exception:
            close_client(client)
            raise
        close_client(client)

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
            sent_before = _sent_action_count(source if args.source == "live" else executor)
            if args.source == "live":
                faults = source.consume_server_result(previous_result)
                if faults:
                    print(f"[Client] action faults/status: {';'.join(faults[:6])}")
                wait_for_submitted_actions(source, sent_before, faults, args)
            elif executor is not None:
                faults = executor.consume_server_result(previous_result)
                if faults:
                    print(f"[Action] faults/status: {';'.join(faults[:6])}")
                wait_for_submitted_actions(executor, sent_before, faults, args)
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
    return 0
