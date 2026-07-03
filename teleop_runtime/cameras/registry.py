from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..recording import _to_serializable


CAMERA_REGISTRY_SCHEMA_VERSION = "dexteleop.camera_registry.v1"


def load_camera_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = normalize_camera_registry(payload)
    registry["source_path"] = str(registry_path)
    return registry


def normalize_camera_registry(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("camera registry must be a JSON object.")
    raw_cameras = payload.get("cameras")
    if not isinstance(raw_cameras, Mapping):
        raise ValueError("camera registry must contain a 'cameras' object.")

    registry = {
        key: _jsonable(value)
        for key, value in payload.items()
        if key not in {"cameras", "schema_version"}
    }
    registry["schema_version"] = str(
        payload.get("schema_version") or CAMERA_REGISTRY_SCHEMA_VERSION
    )
    registry["cameras"] = {
        str(camera_key): _normalize_camera_spec(str(camera_key), raw_camera)
        for camera_key, raw_camera in raw_cameras.items()
    }
    return registry


def build_camera_registry_snapshot(
    registry: Mapping[str, Any],
    *,
    device_info_by_key: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    snapshot = normalize_camera_registry(registry)
    devices = dict(device_info_by_key or {})
    cameras = dict(snapshot.get("cameras") or {})

    for camera_key in sorted(set(cameras) | {str(key) for key in devices}):
        spec = dict(cameras.get(camera_key) or {"camera_key": camera_key})
        device_info = devices.get(camera_key)
        if isinstance(device_info, Mapping):
            spec = _merge_runtime_device_info(spec, device_info)
        cameras[camera_key] = spec

    snapshot["cameras"] = cameras
    return _jsonable(snapshot)


def camera_matrix_from_intrinsics(intrinsics: Mapping[str, Any]) -> list[list[float]]:
    intr = _normalize_intrinsics(intrinsics)
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def pinhole_fov_from_intrinsics(intrinsics: Mapping[str, Any]) -> dict[str, float]:
    intr = _normalize_intrinsics(intrinsics)
    width = float(intr["width"])
    height = float(intr["height"])
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    return {
        "fovx_rad": float(2.0 * math.atan(width / (2.0 * fx))),
        "fovy_rad": float(2.0 * math.atan(height / (2.0 * fy))),
    }


def _normalize_camera_spec(camera_key: str, raw_camera: Any) -> dict[str, Any]:
    if not isinstance(raw_camera, Mapping):
        raise ValueError(f"camera registry entry '{camera_key}' must be an object.")

    spec = {
        key: _jsonable(value)
        for key, value in raw_camera.items()
        if key not in {"profile", "intrinsics", "extrinsics"}
    }
    spec["camera_key"] = str(raw_camera.get("camera_key") or camera_key)
    if "serial" in raw_camera:
        spec["serial"] = str(raw_camera["serial"])
    spec["profile"] = _normalize_profile(raw_camera.get("profile"), raw_camera)

    intrinsics = raw_camera.get("intrinsics")
    spec["intrinsics"] = (
        None if intrinsics in (None, {}) else _normalize_intrinsics(intrinsics)
    )
    spec["extrinsics"] = _normalize_extrinsics(raw_camera.get("extrinsics"))
    return spec


def _normalize_profile(raw_profile: Any, fallback: Mapping[str, Any]) -> dict[str, Any]:
    profile = dict(raw_profile) if isinstance(raw_profile, Mapping) else {}
    for key in ("width", "height", "fps", "format", "stream"):
        if key not in profile and key in fallback:
            profile[key] = fallback[key]
    if "stream" not in profile:
        profile["stream"] = "color"
    return _jsonable(profile)


def _normalize_intrinsics(raw_intrinsics: Any) -> dict[str, Any]:
    if not isinstance(raw_intrinsics, Mapping):
        raise ValueError("camera intrinsics must be an object.")
    intr = dict(raw_intrinsics)
    if "cx" not in intr and "ppx" in intr:
        intr["cx"] = intr["ppx"]
    if "cy" not in intr and "ppy" in intr:
        intr["cy"] = intr["ppy"]
    if "ppx" not in intr and "cx" in intr:
        intr["ppx"] = intr["cx"]
    if "ppy" not in intr and "cy" in intr:
        intr["ppy"] = intr["cy"]
    missing = [
        key
        for key in ("width", "height", "fx", "fy", "cx", "cy")
        if key not in intr or intr[key] is None
    ]
    if missing:
        raise ValueError(f"camera intrinsics missing required fields: {missing}")

    normalized = dict(intr)
    normalized["width"] = int(intr["width"])
    normalized["height"] = int(intr["height"])
    for key in ("fx", "fy", "cx", "cy", "ppx", "ppy"):
        if key in normalized and normalized[key] is not None:
            normalized[key] = float(normalized[key])
    if "coeffs" in normalized and normalized["coeffs"] is not None:
        normalized["coeffs"] = [float(value) for value in normalized["coeffs"]]
    normalized.setdefault("coeffs", [])
    normalized.setdefault("distortion_model", "unknown")
    return _jsonable(normalized)


def _normalize_extrinsics(raw_extrinsics: Any) -> dict[str, Any]:
    if raw_extrinsics in (None, {}):
        return {"frames": {}}
    if not isinstance(raw_extrinsics, Mapping):
        raise ValueError("camera extrinsics must be an object.")

    extrinsics = {
        key: _jsonable(value)
        for key, value in raw_extrinsics.items()
        if key not in {"frames", "urdf", "urdfs", "placements"}
    }
    raw_frames = raw_extrinsics.get("frames")
    frames: dict[str, Any] = {}
    if isinstance(raw_frames, Mapping):
        for frame_name, raw_frame in raw_frames.items():
            frames[str(frame_name)] = _normalize_extrinsic_frame(str(frame_name), raw_frame)
    else:
        frame_name = str(
            raw_extrinsics.get("camera_frame")
            or raw_extrinsics.get("frame")
            or raw_extrinsics.get("frame_name")
            or ""
        ).strip()
        if frame_name:
            frames[frame_name] = _normalize_extrinsic_frame(frame_name, raw_extrinsics)
    extrinsics["frames"] = frames
    return extrinsics


def _normalize_extrinsic_frame(frame_name: str, raw_frame: Any) -> dict[str, Any]:
    if raw_frame in (None, {}):
        raw_frame = {}
    if not isinstance(raw_frame, Mapping):
        raise ValueError(f"extrinsic frame '{frame_name}' must be an object.")

    frame = {
        key: _jsonable(value)
        for key, value in raw_frame.items()
        if key not in {"placements", "urdf", "urdfs", "frame"}
    }
    frame["frame"] = str(raw_frame.get("frame") or frame_name)
    frame["placements"] = _normalize_placements(frame_name, raw_frame)
    return frame


def _normalize_placements(frame_name: str, raw_frame: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_placements = raw_frame.get("placements")
    if raw_placements is None:
        raw_placements = raw_frame.get("urdfs")
    if raw_placements is None and raw_frame.get("urdf") is not None:
        raw_placements = [{"urdf": raw_frame.get("urdf")}]
    if raw_placements is None:
        return []
    if not isinstance(raw_placements, list):
        raise ValueError(f"placements for frame '{frame_name}' must be a list.")

    placements: list[dict[str, Any]] = []
    for index, raw_placement in enumerate(raw_placements):
        if isinstance(raw_placement, str):
            placement = {"urdf": raw_placement}
        elif isinstance(raw_placement, Mapping):
            placement = dict(raw_placement)
        else:
            raise ValueError(
                f"placement {index} for frame '{frame_name}' must be an object or string."
            )
        if not str(placement.get("urdf") or "").strip():
            raise ValueError(f"placement {index} for frame '{frame_name}' is missing 'urdf'.")
        placement["urdf"] = str(placement["urdf"])
        placement["frame"] = str(placement.get("frame") or frame_name)
        placements.append(_jsonable(placement))
    return placements


def _merge_runtime_device_info(
    spec: Mapping[str, Any],
    device_info: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(spec)
    runtime = _stable_runtime_device_info(device_info)
    merged["runtime_device"] = runtime

    runtime_intrinsics = _runtime_intrinsics(device_info)
    if runtime_intrinsics is not None:
        previous = merged.get("intrinsics")
        intrinsics = dict(previous) if isinstance(previous, Mapping) else {}
        intrinsics.update(runtime_intrinsics)
        intrinsics["source"] = "runtime_device_info"
        merged["intrinsics"] = _normalize_intrinsics(intrinsics)

    profile = dict(merged.get("profile") or {})
    for key in ("width", "height", "fps", "color_space"):
        if key in device_info and device_info[key] is not None:
            profile[key] = _jsonable(device_info[key])
    merged["profile"] = profile
    if "serial_number" in runtime:
        merged["runtime_serial"] = runtime["serial_number"]
    return _jsonable(merged)


def _runtime_intrinsics(device_info: Mapping[str, Any]) -> dict[str, Any] | None:
    intrinsics = device_info.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        return None
    payload = dict(intrinsics)
    for key in ("width", "height"):
        if key not in payload and key in device_info:
            payload[key] = device_info[key]
    payload.setdefault("distortion_model", device_info.get("distortion_model", "unknown"))
    return _normalize_intrinsics(payload)


def _stable_runtime_device_info(device_info: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "camera_key",
        "name",
        "serial",
        "requested_serial",
        "serial_number",
        "serial_match",
        "device_name",
        "product_line",
        "firmware_version",
        "width",
        "height",
        "fps",
        "color_space",
        "metadata",
    )
    return {
        key: _jsonable(device_info[key])
        for key in keys
        if key in device_info and device_info[key] is not None
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return _to_serializable(value)
