"""No-motion front RealSense to OpenPI websocket loopback probe.

The probe reuses the formal real-teleop camera startup lifecycle, captures one
real frame, adapts it through the OpenPI observation boundary, and sends it to
a synthetic localhost websocket receiver.  It never creates arm, gripper, or
model backends and never dispatches an action.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployments.client.rtc_transport import RtcTransportDeadlineSpec
from deployments.openpi.client.observation_adapter import (
    OpenPIObservationAdapterConfig,
    deployment_observation_to_openpi,
)
from deployments.openpi.client.transport import (
    OpenPIWebsocketTransport,
    packb,
    unpackb,
)
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.schema import OBSERVATION_STATE_DIM
from teleop_runtime.deploy.camera_runtime import run_camera_startup_self_check
from teleop_runtime.real_teleop.camera_runtime import (
    _close_camera_sources,
    _start_camera_sources_best_effort,
    apply_real_teleop_camera_startup_defaults,
)
from teleop_runtime.sources.camera import (
    RealSenseRGBSource,
    load_realsense_camera_config_map,
)


@contextmanager
def _localhost_receiver(
    receipt: dict[str, Any],
) -> Iterator[int]:
    from websockets.asyncio.server import serve

    started = threading.Event()
    stop = threading.Event()
    port_box: list[int] = []
    failures: list[BaseException] = []

    async def handler(connection: Any) -> None:
        await connection.send(
            packb(
                {
                    "protocol_version": "openpi.local_policy.websocket.v1",
                    "runtime_build": "synthetic-camera-loopback-no-model",
                }
            )
        )
        encoded = await asyncio.wait_for(connection.recv(), timeout=3.0)
        if not isinstance(encoded, bytes):
            raise TypeError(f"expected bytes request, got {type(encoded).__name__}")
        request = unpackb(encoded)
        if not isinstance(request, dict):
            raise TypeError(f"expected dict request, got {type(request).__name__}")
        key = "observation.images.base_0_rgb"
        if key not in request:
            raise KeyError(f"missing OpenPI image key {key!r}")
        image = np.asarray(request[key])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"invalid received image shape/dtype: {image.shape}/{image.dtype}"
            )
        image_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
        receipt.update(
            {
                "received": True,
                "request_keys": sorted(str(name) for name in request),
                "image_key": key,
                "image_shape": list(image.shape),
                "image_dtype": str(image.dtype),
                "image_min": int(image.min()),
                "image_max": int(image.max()),
                "image_mean": float(image.mean()),
                "image_sha256": image_sha256,
                "state_shape": list(np.asarray(request.get("state")).shape),
            }
        )
        await connection.send(
            packb(
                {
                    "accepted": True,
                    "received_image_sha256": image_sha256,
                }
            )
        )

    async def run_server() -> None:
        try:
            async with serve(
                handler,
                "127.0.0.1",
                0,
                compression=None,
                max_size=None,
            ) as server:
                if not server.sockets:
                    raise RuntimeError("localhost receiver has no listening socket")
                port_box.append(int(server.sockets[0].getsockname()[1]))
                started.set()
                while not stop.is_set():
                    await asyncio.sleep(0.001)
        except BaseException as exc:  # noqa: BLE001 - surface thread evidence.
            failures.append(exc)
            started.set()

    thread = threading.Thread(
        target=lambda: asyncio.run(run_server()),
        name="openpi-camera-loopback-server",
        daemon=True,
    )
    thread.start()
    if not started.wait(timeout=3.0):
        raise TimeoutError("localhost receiver did not start within 3 seconds")
    if failures:
        raise failures[0]
    if not port_box:
        raise RuntimeError("localhost receiver did not publish a port")
    try:
        yield port_box[0]
    finally:
        stop.set()
        thread.join(timeout=3.0)
        if thread.is_alive():
            raise RuntimeError("localhost receiver thread did not stop")
        if failures:
            raise failures[0]


def _save_snapshot(path: Path, image: np.ndarray, *, color_space: str) -> None:
    import cv2

    output = np.asarray(image)
    if str(color_space).strip().lower() == "rgb":
        output = np.ascontiguousarray(output[..., ::-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"failed to save snapshot: {path}")


def _wait_for_transport_close(
    transport: OpenPIWebsocketTransport,
    *,
    timeout_sec: float = 1.0,
) -> None:
    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline:
        receipt = transport.last_close_receipt
        if receipt is not None:
            if not receipt.completed:
                raise RuntimeError(f"OpenPI transport close failed: {receipt}")
            return
        time.sleep(0.01)
    raise TimeoutError("OpenPI transport close did not complete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=Path("teleop_runtime/configs/devices/realsense.json"),
    )
    parser.add_argument("--camera-key", default="front")
    parser.add_argument("--capture-settle-sec", type=float, default=0.0)
    parser.add_argument("--snapshot-out", type=Path, required=True)
    args = parser.parse_args()

    configs = load_realsense_camera_config_map(args.camera_config)
    camera_key = str(args.camera_key)
    if camera_key not in configs:
        raise KeyError(
            f"camera {camera_key!r} missing from {args.camera_config}; "
            f"available={sorted(configs)}"
        )
    selected_configs = {camera_key: configs[camera_key]}
    apply_real_teleop_camera_startup_defaults(selected_configs)
    # Opening and streaming are authorized by this probe; USB hardware reset is
    # a separate device-state change and is intentionally excluded.
    for config in selected_configs.values():
        config.startup_hardware_reset = False
        config.startup_hardware_reset_on_failure = False
    sources = {
        key: RealSenseRGBSource(config)
        for key, config in selected_configs.items()
    }
    transport: OpenPIWebsocketTransport | None = None
    summary: dict[str, Any] = {
        "schema": "dexteleop_openpi_camera_loopback_probe_v1",
        "camera_key": camera_key,
        "no_motion": True,
        "model_loaded": False,
        "gpu_required": False,
    }
    try:
        startup_errors = _start_camera_sources_best_effort(sources)
        if startup_errors:
            raise RuntimeError(f"camera startup failed: {startup_errors}")
        run_camera_startup_self_check(sources)
        settle_sec = max(0.0, float(args.capture_settle_sec))
        if settle_sec:
            time.sleep(settle_sec)

        source = sources[camera_key]
        frame = source.get_latest_frame()
        if frame is None:
            raise RuntimeError("front camera passed self-check but has no latest frame")
        image = np.asarray(frame.image)
        local_image_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise RuntimeError(
                f"invalid local frame shape/dtype: {image.shape}/{image.dtype}"
            )
        _save_snapshot(
            args.snapshot_out,
            image,
            color_space=str(source.config.color_space),
        )

        observation = DeploymentObservation(
            timestamp=time.time(),
            step_index=0,
            active_sides=("left", "right"),
            arms_enabled=False,
            hands_enabled=False,
            images={camera_key: image},
            image_timestamps={camera_key: float(frame.source_timestamp)},
            image_ages_sec={camera_key: max(0.0, time.time() - float(frame.source_timestamp))},
            state=np.zeros(OBSERVATION_STATE_DIM, dtype=np.float32),
            metadata={"instruction": "camera loopback probe"},
        )
        adapter = OpenPIObservationAdapterConfig(
            image_key_map={camera_key: "observation.images.base_0_rgb"},
            required_image_keys=(camera_key,),
        )
        request = deployment_observation_to_openpi(observation, adapter)
        server_receipt: dict[str, Any] = {}
        with _localhost_receiver(server_receipt) as port:
            transport = OpenPIWebsocketTransport(
                host="127.0.0.1",
                port=port,
                retry_interval_s=0.05,
                deadline_spec=RtcTransportDeadlineSpec(
                    connect_timeout_sec=2.0,
                    metadata_timeout_sec=2.0,
                    inference_timeout_sec=3.0,
                    close_timeout_sec=1.0,
                ),
            )
            server_metadata = transport.metadata
            response = transport.infer(request)
            phase_receipts = [
                {
                    "phase": receipt.phase,
                    "status": receipt.status,
                    "elapsed_sec": receipt.completed_at - receipt.started_at,
                }
                for receipt in transport.phase_receipts_snapshot()
            ]
            transport.close()
            _wait_for_transport_close(transport)
            transport = None

        if not server_receipt.get("received"):
            raise RuntimeError("localhost server did not record the OpenPI request")
        if server_receipt.get("image_sha256") != local_image_sha256:
            raise RuntimeError(
                "camera image hash changed across the OpenPI websocket boundary"
            )
        if response.get("received_image_sha256") != local_image_sha256:
            raise RuntimeError("server acknowledgement image hash mismatch")

        info = source.device_info()
        summary.update(
            {
                "ok": True,
                "camera": {
                    "serial": info.get("serial_number") or source.config.serial,
                    "device_name": info.get("device_name"),
                    "frame_count": info.get("frame_count"),
                    "last_frame_age_sec": info.get("last_frame_age_sec"),
                    "startup_timing": source.startup_timing,
                },
                "local_frame": {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "min": int(image.min()),
                    "max": int(image.max()),
                    "mean": float(image.mean()),
                    "sha256": local_image_sha256,
                    "snapshot": str(args.snapshot_out),
                },
                "server_metadata": server_metadata,
                "server_receipt": server_receipt,
                "transport_phases": phase_receipts,
                "response": response,
            }
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        if transport is not None:
            transport.close()
        _close_camera_sources(sources)


if __name__ == "__main__":
    raise SystemExit(main())
