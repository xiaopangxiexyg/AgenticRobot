"""GraspGenX inference backends for the point-grasp planner.

The loopback client keeps the GPU stack in a separate asset-contained
environment, while the optional in-process adapter imports the official
package lazily into a compatible Gateway environment. Neither backend owns
robot or motion resources.
"""

from __future__ import annotations

import importlib
import math
import os
import re
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import msgpack
import numpy as np
import zmq

GRASPGENX_DEFAULT_HOST = "127.0.0.1"
GRASPGENX_DEFAULT_PORT = 5556
GRASPGENX_DEFAULT_TIMEOUT_MS = 120_000
GRASPGENX_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
GRASPGENX_INPROCESS_BACKEND_ID = "graspgenx-inprocess-v1"
_GRASPGENX_IMPORT_LOCK = threading.RLock()


@runtime_checkable
class GraspGenXInferenceBackend(Protocol):
    """Common lifecycle and inference contract for one GraspGenX backend.

    The point-grasp planner owns frame conversion and candidate admission. A
    backend only owns model execution and returns poses in the input point
    cloud frame. Both the loopback client and the direct in-process adapter
    implement this contract.
    """

    backend_id: str

    @property
    def semantic_config(self) -> Mapping[str, object]:
        ...

    def start(self) -> None:
        ...

    def close(self) -> None:
        ...

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
        ...


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

    backend_id = "graspgenx-service-v1"

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


class GraspGenXInProcessBackend:
    """Direct GraspGenX model adapter for a compatible Gateway process.

    The official GraspGenX source is imported lazily from ``source_root`` and
    the reviewed checkpoint bundle is loaded once during :meth:`start`. The
    model is shared by all sweep-volume samplers, matching the existing ZMQ
    server. Inference is serialized because the upstream sampler mutates the
    model's sampling count and is not a documented re-entrant API.

    This adapter intentionally supports CUDA only. The reviewed upstream
    loader calls ``model.cuda()`` and the sampler's gripper conditioning also
    uses ``.cuda()``; silently accepting a CPU device would therefore produce
    a misleading half-working backend.
    """

    backend_id = GRASPGENX_INPROCESS_BACKEND_ID

    def __init__(
        self,
        *,
        model_spec: object,
        source_root: str | Path | None = None,
        device: str = "cuda",
        use_tensorrt: bool = False,
        tensorrt_precision: str = "fp32",
    ) -> None:
        if not isinstance(
            getattr(model_spec, "semantic_content_hash", None), str
        ):
            raise TypeError("model_spec must expose semantic_content_hash.")
        try:
            bundle_root = Path(getattr(model_spec, "bundle_root")).resolve(strict=True)
            checkpoint_root = Path(
                getattr(model_spec, "checkpoint_root")
            ).resolve(strict=True)
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise TypeError(
                "model_spec must expose existing bundle_root and checkpoint_root."
            ) from exc
        if not bundle_root.is_dir() or not checkpoint_root.is_dir():
            raise ValueError("model_spec asset roots must be existing directories.")
        normalized_device = str(device).strip()
        if not re.fullmatch(r"cuda(?::[0-9]+)?", normalized_device):
            raise ValueError(
                "GraspGenX in-process backend requires a CUDA device such as 'cuda'."
            )
        if not isinstance(use_tensorrt, bool):
            raise TypeError("use_tensorrt must be boolean.")
        if tensorrt_precision not in {"fp32", "fp16"}:
            raise ValueError("tensorrt_precision must be 'fp32' or 'fp16'.")
        if source_root is None:
            source_path = (
                Path(__file__).resolve().parents[2]
                / "model"
                / "perception"
                / "graspgenx"
            )
        else:
            source_path = Path(source_root).expanduser()
        self.model_spec = model_spec
        self.bundle_root = bundle_root
        self.checkpoint_root = checkpoint_root
        self.source_root = source_path
        self.device = normalized_device
        self.use_tensorrt = use_tensorrt
        self.tensorrt_precision = tensorrt_precision
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._torch: Any = None
        self._device_obj: Any = None
        self._cfg: Any = None
        self._model: Any = None
        self._sampler_cls: Any = None
        self._sweep_volume_params_cls: Any = None
        self._run_planner_on_object: Any = None
        self._samplers: dict[str, Any] = {}
        self._tensorrt_applied = False

    @property
    def semantic_config(self) -> Mapping[str, object]:
        return {
            "backend": self.backend_id,
            "model_spec_hash": str(self.model_spec.semantic_content_hash),
            "device": self.device,
            "use_tensorrt": self.use_tensorrt,
            "tensorrt_precision": (
                self.tensorrt_precision if self.use_tensorrt else None
            ),
        }

    @staticmethod
    def _temporary_environment(
        *, bundle_root: Path
    ) -> dict[str, str | None]:
        values = {
            "GRASPGENX_CHECKPOINT_DIR": str(bundle_root / "checkpoints"),
            "GRASPGENX_GRIPPER_CFG_DIR": str(bundle_root / "gripper_descriptions"),
        }
        previous = {name: os.environ.get(name) for name in values}
        for name, value in values.items():
            os.environ[name] = value
        return previous

    @staticmethod
    def _restore_environment(previous: Mapping[str, str | None]) -> None:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _load_runtime(self) -> tuple[Any, Any, Any, Any, Any, Any, Any, bool]:
        try:
            source_root = self.source_root.resolve(strict=True)
        except OSError as exc:
            raise FileNotFoundError(
                f"GraspGenX source_root is not an existing directory: "
                f"{self.source_root}"
            ) from exc
        if not source_root.is_dir():
            raise FileNotFoundError(
                f"GraspGenX source_root is not an existing directory: {source_root}"
            )
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        with _GRASPGENX_IMPORT_LOCK:
            previous = self._temporary_environment(bundle_root=self.bundle_root)
            try:
                package = importlib.import_module("graspgenx")
                package_file = getattr(package, "__file__", None)
                if (
                    package_file is None
                    or source_root not in Path(package_file).resolve().parents
                ):
                    raise RuntimeError(
                        "GraspGenX package was not imported from the requested source_root."
                    )
                torch = importlib.import_module("torch")
                torch_version = re.match(
                    r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)",
                    str(torch.__version__),
                )
                if (
                    torch_version is None
                    or int(torch_version.group("major")) != 2
                    or not 1 <= int(torch_version.group("minor")) < 7
                ):
                    raise RuntimeError(
                        "Official GraspGenX in-process inference requires "
                        "torch>=2.1,<2.7; found "
                        f"{torch.__version__}."
                    )
                grasp_server = importlib.import_module("graspgenx.grasp_server")
                checkpoint_io = importlib.import_module(
                    "graspgenx.utils.checkpoint_io"
                )
                samplers = importlib.import_module("graspgenx.samplers")
                serving_types = importlib.import_module("graspgenx.serving.types")
                device_obj = torch.device(self.device)
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "GraspGenX in-process backend requires torch.cuda.is_available()."
                    )
                gen_dir = self.checkpoint_root / "gen"
                dis_dir = self.checkpoint_root / "dis"
                cfg = checkpoint_io.load_model_cfg(str(gen_dir), str(dis_dir))
                with torch.cuda.device(device_obj):
                    model = grasp_server.load_grasp_gen_model(cfg)
                    tensorrt_applied = False
                    if self.use_tensorrt:
                        from types import SimpleNamespace

                        accelerate_sampler = importlib.import_module(
                            "graspgenx.models.tensorrt_utils"
                        ).accelerate_sampler
                        set_gpu_obb = importlib.import_module(
                            "graspgenx.samplers.graspmoe"
                        ).set_gpu_obb
                        tensorrt_applied = bool(
                            accelerate_sampler(
                                SimpleNamespace(model=model),
                                precision=self.tensorrt_precision,
                            )
                        )
                        if tensorrt_applied:
                            set_gpu_obb(True)
                return (
                    torch,
                    device_obj,
                    cfg,
                    model,
                    grasp_server.GraspGenXSampler,
                    serving_types.SweepVolumeParams,
                    samplers.run_planner_on_object,
                    tensorrt_applied,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "In-process GraspGenX requires the compatible official runtime "
                    "dependencies in the Gateway environment."
                ) from exc
            finally:
                self._restore_environment(previous)

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Closed GraspGenX backend cannot restart.")
            if self._started:
                return
            runtime = self._load_runtime()
            (
                self._torch,
                self._device_obj,
                self._cfg,
                self._model,
                self._sampler_cls,
                self._sweep_volume_params_cls,
                self._run_planner_on_object,
                self._tensorrt_applied,
            ) = runtime
            self._samplers.clear()
            self._started = True

    def close(self) -> None:
        with self._lock:
            self._samplers.clear()
            self._model = None
            self._cfg = None
            self._sampler_cls = None
            self._sweep_volume_params_cls = None
            self._run_planner_on_object = None
            self._torch = None
            self._device_obj = None
            self._started = False
            self._closed = True

    def _get_sampler(self, params: Any) -> Any:
        key = "sweep:" + params.cache_key()
        sampler = self._samplers.get(key)
        if sampler is None:
            sampler = self._sampler_cls.from_sweep_volume(
                self._cfg,
                params,
                model=self._model,
            )
            self._samplers[key] = sampler
        return sampler

    @staticmethod
    def _to_numpy(value: object) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

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
        params_dict = GraspGenXClient._validated_sweep_volume(sweep_volume_params)
        with self._lock:
            if not self._started or self._closed:
                raise RuntimeError("GraspGenX in-process backend is not running.")
            params = self._sweep_volume_params_cls.coerce(params_dict)
            sampler = self._get_sampler(params)
            backend_top_k = -1 if planner == "graspmoe" else int(num_grasps)
            with self._torch.cuda.device(self._device_obj):
                raw_grasps, raw_scores, _branch_tags, _obb = (
                    self._run_planner_on_object(
                        points,
                        sampler,
                        planner=planner,
                        grasp_threshold=-1.0,
                        num_grasps=int(num_grasps),
                        topk_num_grasps=backend_top_k,
                    )
                )
        grasps = _finite_rigid_matrices(self._to_numpy(raw_grasps))
        try:
            scores = self._to_numpy(raw_scores).astype(np.float64).reshape(-1)
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


__all__ = [
    "GRASPGENX_DEFAULT_HOST",
    "GRASPGENX_DEFAULT_PORT",
    "GRASPGENX_DEFAULT_TIMEOUT_MS",
    "GRASPGENX_INPROCESS_BACKEND_ID",
    "GraspGenXClient",
    "GraspGenXInferenceBackend",
    "GraspGenXInProcessBackend",
    "pack_graspgenx_message",
    "unpack_graspgenx_message",
]
