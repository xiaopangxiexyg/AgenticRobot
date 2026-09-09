"""Lightweight, model-free client for the local GraspGenX service.

The upstream client imports the complete ``graspgenx`` package and therefore
eagerly resolves model assets.  AgenticRobot keeps the GPU stack in a separate
asset-contained environment, so this module implements only the reviewed
msgpack-numpy/ZMQ wire contract.  It deliberately accepts loopback endpoints
only and never owns robot or motion resources.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from typing import Any

import msgpack
import numpy as np
import zmq

GRASPGENX_DEFAULT_HOST = "127.0.0.1"
GRASPGENX_DEFAULT_PORT = 5556
GRASPGENX_DEFAULT_TIMEOUT_MS = 120_000
GRASPGENX_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _encode_numpy(value: object) -> object:
    """Encode the non-object NumPy subset used by upstream msgpack-numpy."""

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise TypeError("Object and structured NumPy arrays are not supported.")
        contiguous = np.ascontiguousarray(value)
        return {
            b"nd": True,
            b"type": contiguous.dtype.str,
            b"kind": b"",
            b"shape": contiguous.shape,
            b"data": contiguous.tobytes(),
        }
    if isinstance(value, (np.bool_, np.number)):
        scalar = np.asarray(value)
        return {
            b"nd": False,
            b"type": scalar.dtype.str,
            b"data": scalar.tobytes(),
        }
    raise TypeError(f"Cannot encode {type(value).__name__} in GraspGenX request.")


def _decode_numpy(value: dict[object, object]) -> object:
    """Decode upstream msgpack-numpy arrays without its pickle fallback."""

    if b"nd" not in value:
        return value
    dtype = np.dtype(value[b"type"])
    if dtype.hasobject or dtype.fields is not None:
        raise ValueError("Object and structured arrays are forbidden on the wire.")
    data = value[b"data"]
    if not isinstance(data, bytes):
        raise TypeError("NumPy wire payload data must be bytes.")
    if value[b"nd"] is False:
        array = np.frombuffer(data, dtype=dtype)
        if array.size != 1:
            raise ValueError("NumPy scalar wire payload has an invalid size.")
        return array[0]
    shape = tuple(int(item) for item in value[b"shape"])
    if any(item < 0 for item in shape):
        raise ValueError("NumPy wire payload shape cannot be negative.")
    expected = math.prod(shape) * dtype.itemsize
    if expected != len(data) or expected > GRASPGENX_MAX_MESSAGE_BYTES:
        raise ValueError("NumPy wire payload shape and byte length differ.")
    return np.frombuffer(data, dtype=dtype).reshape(shape)


def pack_graspgenx_message(payload: Mapping[str, object]) -> bytes:
    encoded = msgpack.packb(dict(payload), default=_encode_numpy, use_bin_type=True)
    if len(encoded) > GRASPGENX_MAX_MESSAGE_BYTES:
        raise ValueError("GraspGenX request exceeds the 64 MiB wire limit.")
    return encoded


def unpack_graspgenx_message(payload: bytes) -> dict[str, object]:
    if len(payload) > GRASPGENX_MAX_MESSAGE_BYTES:
        raise ValueError("GraspGenX response exceeds the 64 MiB wire limit.")
    decoded = msgpack.unpackb(payload, raw=False, object_hook=_decode_numpy)
    if not isinstance(decoded, dict):
        raise TypeError("GraspGenX response must be a mapping.")
    return decoded


def _finite_rigid_matrices(value: object) -> np.ndarray:
    try:
        matrices = np.asarray(value, dtype=np.float64).reshape(-1, 4, 4)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("GraspGenX returned malformed grasp poses.") from exc
    if not np.all(np.isfinite(matrices)):
        raise RuntimeError("GraspGenX returned non-finite grasp poses.")
    for matrix in matrices:
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise RuntimeError("GraspGenX returned a non-homogeneous grasp pose.")
        rotation = matrix[:3, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=2e-3
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
            raise RuntimeError("GraspGenX returned a non-rigid grasp pose.")
    return matrices


class GraspGenXClient:
    """Synchronous loopback client for one GraspGenX inference server."""

    def __init__(
        self,
        *,
        host: str = GRASPGENX_DEFAULT_HOST,
        port: int = GRASPGENX_DEFAULT_PORT,
        timeout_ms: int = GRASPGENX_DEFAULT_TIMEOUT_MS,
        context: zmq.Context | None = None,
    ) -> None:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("GraspGenX inference must use a loopback endpoint.")
        if isinstance(port, bool) or not 1 <= int(port) <= 65_535:
            raise ValueError("GraspGenX port must be in [1, 65535].")
        if isinstance(timeout_ms, bool) or not 1_000 <= int(timeout_ms) <= 600_000:
            raise ValueError("GraspGenX timeout_ms must be in [1000, 600000].")
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self._context = context
        self._socket: Any = None
        self._lock = threading.RLock()
        self._closed = False

    @property
    def address(self) -> str:
        host = f"[{self.host}]" if self.host == "::1" else self.host
        return f"tcp://{host}:{self.port}"

    @property
    def semantic_config(self) -> dict[str, object]:
        return {
            "transport": "graspgenx-msgpack-numpy-zmq-req-v1",
            "host": self.host,
            "port": self.port,
            "timeout_ms": self.timeout_ms,
            "maximum_message_bytes": GRASPGENX_MAX_MESSAGE_BYTES,
        }

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Closed GraspGenX client cannot restart.")
            health = self._request({"action": "health"})
            if health.get("status") != "ok":
                raise RuntimeError("GraspGenX server health response is not ok.")
            metadata = self._request({"action": "metadata"})
            actions = metadata.get("actions")
            if not isinstance(actions, list) or "infer_object" not in actions:
                raise RuntimeError(
                    "GraspGenX server does not expose infer_object sweep-volume inference."
                )

    def close(self) -> None:
        with self._lock:
            if self._socket is not None:
                self._socket.close(linger=0)
                self._socket = None
            self._closed = True

    def _connect(self) -> None:
        if self._closed:
            raise RuntimeError("GraspGenX client is closed.")
        if self._socket is not None:
            return
        context = self._context or zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.MAXMSGSIZE, GRASPGENX_MAX_MESSAGE_BYTES)
        socket.connect(self.address)
        self._socket = socket

    def _request(self, payload: Mapping[str, object]) -> dict[str, object]:
        self._connect()
        try:
            self._socket.send(pack_graspgenx_message(payload))
            raw = self._socket.recv()
        except zmq.error.Again as exc:
            self._socket.close(linger=0)
            self._socket = None
            raise TimeoutError(
                f"GraspGenX server did not respond within {self.timeout_ms} ms."
            ) from exc
        response = unpack_graspgenx_message(raw)
        if "error" in response:
            raise RuntimeError(f"GraspGenX server error: {response['error']}")
        return response

    def infer_object(
        self,
        point_cloud: np.ndarray,
        *,
        sweep_volume_params: Mapping[str, object],
        planner: str = "graspmoe",
        num_grasps: int = 200,
        score_threshold: float = -1.0,
        top_k: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.ascontiguousarray(point_cloud, dtype=np.float32)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or len(points) < 16
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(
                "point_cloud must contain at least 16 finite XYZ meter points."
            )
        if planner not in {"graspmoe", "diffusion"}:
            raise ValueError("planner must be 'graspmoe' or 'diffusion'.")
        if isinstance(num_grasps, bool) or not 1 <= int(num_grasps) <= 1_000:
            raise ValueError("num_grasps must be in [1, 1000].")
        if isinstance(top_k, bool) or not 1 <= int(top_k) <= 1_000:
            raise ValueError("top_k must be in [1, 1000].")
        threshold = float(score_threshold)
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("score_threshold must be finite and in [-1, 1].")
        params = self._validated_sweep_volume(sweep_volume_params)
        with self._lock:
            response = self._request(
                {
                    "action": "infer_object",
                    "point_cloud": points,
                    "sweep_volume_params": params,
                    "planner": planner,
                    "num_grasps": int(num_grasps),
                }
            )
        grasps = _finite_rigid_matrices(response.get("grasps"))
        try:
            scores = np.asarray(response.get("confidences"), dtype=np.float64).reshape(
                -1
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "GraspGenX returned malformed confidence scores."
            ) from exc
        if len(scores) != len(grasps) or not np.all(np.isfinite(scores)):
            raise RuntimeError("GraspGenX grasp and confidence counts differ.")
        keep = np.ones(len(scores), dtype=bool)
        if threshold > 0.0:
            keep &= scores >= threshold
        grasps = grasps[keep]
        scores = scores[keep]
        order = np.argsort(-scores, kind="stable")[: int(top_k)]
        return (
            np.ascontiguousarray(grasps[order], dtype=np.float64),
            np.ascontiguousarray(scores[order], dtype=np.float64),
        )

    @staticmethod
    def _validated_sweep_volume(
        value: Mapping[str, object],
    ) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise TypeError("sweep_volume_params must be a mapping.")
        required = {"extents_open", "offset_open", "extents_mid", "offset_mid"}
        optional = {"gripper_type", "fingertip_depth"}
        if set(value) - required - optional or required - set(value):
            raise ValueError("sweep_volume_params fields are invalid.")
        result: dict[str, object] = {}
        for name in sorted(required):
            array = np.asarray(value[name], dtype=np.float32).reshape(-1)
            if array.shape != (3,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain three finite values.")
            if name.startswith("extents_") and not np.all(array > 0.0):
                raise ValueError(f"{name} values must be positive.")
            result[name] = array
        gripper_type = int(value.get("gripper_type", 0))
        if gripper_type not in {0, 1, 2}:
            raise ValueError("gripper_type must be 0, 1, or 2.")
        result["gripper_type"] = gripper_type
        if "fingertip_depth" in value:
            fingertip_depth = float(value["fingertip_depth"])
            if not math.isfinite(fingertip_depth) or fingertip_depth <= 0.0:
                raise ValueError("fingertip_depth must be positive and finite.")
            result["fingertip_depth"] = fingertip_depth
        return result


__all__ = [
    "GRASPGENX_DEFAULT_HOST",
    "GRASPGENX_DEFAULT_PORT",
    "GRASPGENX_DEFAULT_TIMEOUT_MS",
    "GraspGenXClient",
    "pack_graspgenx_message",
    "unpack_graspgenx_message",
]
