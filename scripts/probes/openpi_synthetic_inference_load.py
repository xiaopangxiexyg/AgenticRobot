"""Bounded synthetic OpenPI inference load; never dispatch model actions.

The caller starts an existing OpenPI server separately. This client reads only
the checkpoint artifact, generates black images and a finite state vector, and
prints one JSON record per request plus a final summary. It opens no robot,
camera, sensor, or teleop component.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from deployments.openpi.client.transport import OpenPIWebsocketTransport
from deployments.client.rtc_transport import RtcTransportDeadlineSpec


def build_observation(artifact: dict, *, prompt: str) -> dict[str, object]:
    state = artifact["inputs"]["state"]
    if state["dtype"] != "float32" or not 1 <= int(state["dim"]) <= 4096:
        raise ValueError("Unsupported checkpoint state contract")
    camera = artifact["camera_input"]
    if camera["dtype"] != "uint8" or camera["layout"] != "HWC":
        raise ValueError("Unsupported checkpoint camera contract")
    image_inputs = camera["inputs"]
    if not image_inputs or len(image_inputs) > 8:
        raise ValueError("Expected one to eight camera inputs")
    state_dim = int(state["dim"])
    synthetic_state = np.zeros((state_dim,), dtype=np.float32)
    if artifact.get("state_input_mode") == "masked_normalized_concat_v1":
        if state_dim % 2:
            raise ValueError("Masked state contract requires even state dimension")
        # The second half is the availability mask, not a state value.
        synthetic_state[state_dim // 2 :] = 1.0
    observation: dict[str, object] = {
        "observation.state": synthetic_state,
        "prompt": prompt,
    }
    for item in image_inputs:
        key = str(item["model_input_key"])
        if not key.startswith("observation.images.") or key in observation:
            raise ValueError(f"Invalid or duplicate camera input key: {key}")
        observation[key] = np.zeros((224, 224, 3), dtype=np.uint8)
    return observation


def run(args: argparse.Namespace) -> dict[str, object]:
    artifact_path = args.checkpoint_dir / "policy_artifact.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    observation = build_observation(artifact, prompt=args.prompt)
    transport = OpenPIWebsocketTransport(
        host=args.host,
        port=args.port,
        retry_interval_s=0.1,
        deadline_spec=RtcTransportDeadlineSpec(
            connect_timeout_sec=10.0,
            metadata_timeout_sec=10.0,
            inference_timeout_sec=args.request_timeout_sec,
            close_timeout_sec=5.0,
        ),
    )
    durations: list[float] = []
    try:
        metadata = transport.metadata
        server_model_id = metadata.get("model_id")
        expected_model_id = artifact.get("model_id")
        if server_model_id is not None and server_model_id != expected_model_id:
            raise ValueError(
                f"Server model {server_model_id!r} differs from checkpoint "
                f"model {expected_model_id!r}"
            )
        start = time.monotonic()
        deadline = start + args.duration_sec
        next_request_at = start
        print(json.dumps({
            "event": "start",
            "checkpoint_dir": str(args.checkpoint_dir),
            "model_id": expected_model_id,
            "camera_keys": sorted(k for k in observation if k.startswith("observation.images.")),
            "state_dim": artifact["inputs"]["state"]["dim"],
            "duration_sec": args.duration_sec,
            "target_hz": args.target_hz,
            "server_model_id": server_model_id,
        }), flush=True)
        while time.monotonic() < deadline:
            started = time.monotonic()
            response = transport.infer(observation)
            ended = time.monotonic()
            actions = np.asarray(response["actions"])
            if actions.size == 0 or not np.all(np.isfinite(actions)):
                raise ValueError("OpenPI returned empty or non-finite actions")
            latency = ended - started
            durations.append(latency)
            print(json.dumps({
                "event": "request",
                "index": len(durations),
                "start_monotonic_ns": int(started * 1e9),
                "end_monotonic_ns": int(ended * 1e9),
                "latency_sec": latency,
                "action_shape": list(actions.shape),
            }), flush=True)
            # A slow inference must not create a burst of overdue requests.
            next_request_at = max(next_request_at + 1.0 / args.target_hz, ended)
            time.sleep(max(0.0, min(next_request_at - time.monotonic(), deadline - time.monotonic())))
        elapsed = time.monotonic() - start
        return {
            "event": "summary",
            "requests": len(durations),
            "elapsed_sec": elapsed,
            "achieved_hz": len(durations) / elapsed,
            "latency_p50_sec": statistics.median(durations) if durations else None,
            "latency_p95_sec": float(np.percentile(durations, 95)) if durations else None,
            "latency_p99_sec": float(np.percentile(durations, 99)) if durations else None,
            "latency_max_sec": max(durations) if durations else None,
        }
    finally:
        transport.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8078)
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--target-hz", type=float, default=5.0)
    parser.add_argument("--request-timeout-sec", type=float, default=30.0)
    parser.add_argument("--prompt", default="bimanual writing stage one")
    args = parser.parse_args()
    if args.duration_sec <= 0 or args.target_hz <= 0 or args.request_timeout_sec <= 0:
        parser.error("duration, target rate, and request timeout must be positive")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("synthetic load client only connects to a local OpenPI server")
    try:
        print(json.dumps(run(args)), flush=True)
    except Exception as exc:
        print(json.dumps({"event": "error", "type": type(exc).__name__, "detail": str(exc)}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
