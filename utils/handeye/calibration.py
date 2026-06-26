from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .transforms import (
    as_transform,
    average_transforms,
    invert_transform,
    make_transform,
    transform_distance,
    transform_to_list,
)


SCHEMA_VERSION = "dexteleop.handeye.v1"


@dataclass
class CalibrationSample:
    """One synchronized robot/end-effector and target/camera observation."""

    T_gripper_to_base: np.ndarray
    T_target_to_camera: np.ndarray
    robot_pose_xyz_rpy: list[float] | None = None
    robot_joint_deg: list[float] | None = None
    target_reprojection_rmse_px: float | None = None
    target_corner_count: int | None = None
    timestamp_utc: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.T_gripper_to_base = as_transform(
            self.T_gripper_to_base,
            name="T_gripper_to_base",
        )
        self.T_target_to_camera = as_transform(
            self.T_target_to_camera,
            name="T_target_to_camera",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "T_gripper_to_base": transform_to_list(self.T_gripper_to_base),
            "T_target_to_camera": transform_to_list(self.T_target_to_camera),
            "robot_pose_xyz_rpy": _list_or_none(self.robot_pose_xyz_rpy),
            "robot_joint_deg": _list_or_none(self.robot_joint_deg),
            "target_reprojection_rmse_px": _float_or_none(self.target_reprojection_rmse_px),
            "target_corner_count": (
                int(self.target_corner_count) if self.target_corner_count is not None else None
            ),
            "timestamp_utc": self.timestamp_utc,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationSample":
        return cls(
            T_gripper_to_base=np.asarray(payload["T_gripper_to_base"], dtype=np.float64),
            T_target_to_camera=np.asarray(payload["T_target_to_camera"], dtype=np.float64),
            robot_pose_xyz_rpy=_list_or_none(payload.get("robot_pose_xyz_rpy")),
            robot_joint_deg=_list_or_none(payload.get("robot_joint_deg")),
            target_reprojection_rmse_px=_float_or_none(
                payload.get("target_reprojection_rmse_px")
            ),
            target_corner_count=payload.get("target_corner_count"),
            timestamp_utc=payload.get("timestamp_utc"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class CalibrationResult:
    T_camera_to_base: np.ndarray
    T_target_to_gripper: np.ndarray
    method: str
    metrics: dict[str, Any]
    samples: list[CalibrationSample]
    created_at_utc: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.T_camera_to_base = as_transform(self.T_camera_to_base, name="T_camera_to_base")
        self.T_target_to_gripper = as_transform(
            self.T_target_to_gripper,
            name="T_target_to_gripper",
        )

    @property
    def T_base_to_camera(self) -> np.ndarray:
        return invert_transform(self.T_camera_to_base)

    def to_dict(self, *, include_samples: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": self.created_at_utc,
            "matrix_convention": {
                "T_camera_to_base": "p_base = T_camera_to_base @ p_camera; translation is meters",
                "T_base_to_camera": "p_camera = T_base_to_camera @ p_base; translation is meters",
                "T_target_to_gripper": "p_gripper = T_target_to_gripper @ p_target",
            },
            "method": self.method,
            "sample_count": len(self.samples),
            "T_camera_to_base": transform_to_list(self.T_camera_to_base),
            "T_base_to_camera": transform_to_list(self.T_base_to_camera),
            "T_target_to_gripper": transform_to_list(self.T_target_to_gripper),
            "metrics": _jsonable(self.metrics),
            "metadata": _jsonable(self.metadata),
        }
        if include_samples:
            payload["samples"] = [sample.to_dict() for sample in self.samples]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationResult":
        samples = [CalibrationSample.from_dict(item) for item in payload.get("samples", [])]
        return cls(
            T_camera_to_base=np.asarray(payload["T_camera_to_base"], dtype=np.float64),
            T_target_to_gripper=np.asarray(
                payload.get("T_target_to_gripper", np.eye(4)),
                dtype=np.float64,
            ),
            method=str(payload.get("method", "unknown")),
            metrics=dict(payload.get("metrics") or {}),
            samples=samples,
            created_at_utc=str(payload.get("created_at_utc") or _now_utc()),
            metadata=dict(payload.get("metadata") or {}),
        )


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local calibration env
        raise ImportError(
            "OpenCV with cv2 is required for solving hand-eye calibration. "
            "Install opencv-contrib-python in the calibration environment."
        ) from exc
    return cv2


def _hand_eye_methods(cv2: Any) -> dict[str, int]:
    names = {
        "TSAI": "CALIB_HAND_EYE_TSAI",
        "PARK": "CALIB_HAND_EYE_PARK",
        "HORAUD": "CALIB_HAND_EYE_HORAUD",
        "ANDREFF": "CALIB_HAND_EYE_ANDREFF",
        "DANIILIDIS": "CALIB_HAND_EYE_DANIILIDIS",
    }
    return {name: getattr(cv2, attr) for name, attr in names.items() if hasattr(cv2, attr)}


def solve_eye_to_hand(
    samples: Sequence[CalibrationSample],
    *,
    method: str = "AUTO",
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationResult:
    """Solve fixed-camera RM65 base calibration.

    Inputs follow the physical convention:
    - ``T_gripper_to_base`` maps RM65 end-effector coordinates into base.
    - ``T_target_to_camera`` maps the calibration target into camera.
    - Returned ``T_camera_to_base`` maps camera coordinates into RM65 base.
    """

    if len(samples) < 3:
        raise ValueError("at least 3 samples are required; 15+ diverse samples are recommended.")
    cv2 = _require_cv2()
    methods = _hand_eye_methods(cv2)
    if not methods:
        raise RuntimeError("this OpenCV build does not expose hand-eye calibration methods.")
    requested = str(method or "AUTO").upper()
    if requested == "AUTO":
        candidates = list(methods)
    elif requested in methods:
        candidates = [requested]
    else:
        raise ValueError(f"unknown hand-eye method {method!r}; available: {sorted(methods)}")

    attempts: list[tuple[float, CalibrationResult]] = []
    failures: dict[str, str] = {}
    for name in candidates:
        try:
            T_camera_to_base = _solve_eye_to_hand_with_cv2(samples, cv2, methods[name])
            metrics = evaluate_eye_to_hand(samples, T_camera_to_base)
            T_target_to_gripper = _mean_target_to_gripper(samples, T_camera_to_base)
            result = CalibrationResult(
                T_camera_to_base=T_camera_to_base,
                T_target_to_gripper=T_target_to_gripper,
                method=name,
                metrics=metrics,
                samples=list(samples),
                created_at_utc=_now_utc(),
                metadata=dict(metadata or {}),
            )
            score = _method_score(metrics)
            attempts.append((score, result))
        except Exception as exc:
            failures[name] = str(exc)

    if not attempts:
        raise RuntimeError(f"all hand-eye methods failed: {failures}")

    attempts.sort(key=lambda item: item[0])
    best = attempts[0][1]
    all_method_metrics: dict[str, Any] = {}
    for _, result in attempts:
        all_method_metrics[result.method] = {
            "score": _method_score(result.metrics),
            "translation_rms_m": result.metrics["target_in_gripper_translation_rms_m"],
            "rotation_rms_deg": result.metrics["target_in_gripper_rotation_rms_deg"],
            "translation_max_m": result.metrics["target_in_gripper_translation_max_m"],
            "rotation_max_deg": result.metrics["target_in_gripper_rotation_max_deg"],
        }
    for name, reason in failures.items():
        all_method_metrics[name] = {"failed": reason}
    best.metrics["all_method_metrics"] = all_method_metrics
    best.metrics["quality_warnings"] = quality_warnings(best.metrics)
    return best


def _solve_eye_to_hand_with_cv2(
    samples: Sequence[CalibrationSample],
    cv2: Any,
    method_flag: int,
) -> np.ndarray:
    R_base_to_gripper: list[np.ndarray] = []
    t_base_to_gripper: list[np.ndarray] = []
    R_target_to_camera: list[np.ndarray] = []
    t_target_to_camera: list[np.ndarray] = []
    for sample in samples:
        T_base_to_gripper = invert_transform(sample.T_gripper_to_base)
        R_base_to_gripper.append(T_base_to_gripper[:3, :3])
        t_base_to_gripper.append(T_base_to_gripper[:3, 3].reshape(3, 1))
        R_target_to_camera.append(sample.T_target_to_camera[:3, :3])
        t_target_to_camera.append(sample.T_target_to_camera[:3, 3].reshape(3, 1))

    R_camera_to_base, t_camera_to_base = cv2.calibrateHandEye(
        R_base_to_gripper,
        t_base_to_gripper,
        R_target_to_camera,
        t_target_to_camera,
        method=method_flag,
    )
    return make_transform(R_camera_to_base, np.asarray(t_camera_to_base).reshape(3))


def evaluate_eye_to_hand(
    samples: Sequence[CalibrationSample],
    T_camera_to_base: object,
) -> dict[str, Any]:
    T_camera_to_base = as_transform(T_camera_to_base, name="T_camera_to_base")
    target_to_gripper = [
        invert_transform(sample.T_gripper_to_base)
        @ T_camera_to_base
        @ sample.T_target_to_camera
        for sample in samples
    ]
    mean_target = average_transforms(target_to_gripper)

    target_errors = [transform_distance(mean_target, value) for value in target_to_gripper]
    predicted_errors: list[tuple[float, float]] = []
    for sample in samples:
        predicted_target_to_camera = (
            invert_transform(T_camera_to_base)
            @ sample.T_gripper_to_base
            @ mean_target
        )
        predicted_errors.append(
            transform_distance(predicted_target_to_camera, sample.T_target_to_camera)
        )
    per_sample_errors = []
    for index, (sample, target_error, predicted_error) in enumerate(
        zip(samples, target_errors, predicted_errors),
        start=1,
    ):
        sample_index = sample.metadata.get("sample_index", index)
        per_sample_errors.append(
            {
                "sample_index": int(sample_index),
                "target_in_gripper_translation_m": float(target_error[0]),
                "target_in_gripper_rotation_deg": float(target_error[1]),
                "prediction_translation_m": float(predicted_error[0]),
                "prediction_rotation_deg": float(predicted_error[1]),
                "target_reprojection_rmse_px": sample.target_reprojection_rmse_px,
            }
        )

    robot_span = _pose_span([sample.T_gripper_to_base for sample in samples])
    target_span = _pose_span([sample.T_target_to_camera for sample in samples])
    reprojection_values = [
        float(sample.target_reprojection_rmse_px)
        for sample in samples
        if sample.target_reprojection_rmse_px is not None
    ]

    metrics = {
        "sample_count": len(samples),
        "target_in_gripper_translation_rms_m": _rms([value[0] for value in target_errors]),
        "target_in_gripper_translation_max_m": _max_or_zero([value[0] for value in target_errors]),
        "target_in_gripper_rotation_rms_deg": _rms([value[1] for value in target_errors]),
        "target_in_gripper_rotation_max_deg": _max_or_zero([value[1] for value in target_errors]),
        "target_to_camera_prediction_translation_rms_m": _rms(
            [value[0] for value in predicted_errors]
        ),
        "target_to_camera_prediction_translation_max_m": _max_or_zero(
            [value[0] for value in predicted_errors]
        ),
        "target_to_camera_prediction_rotation_rms_deg": _rms(
            [value[1] for value in predicted_errors]
        ),
        "target_to_camera_prediction_rotation_max_deg": _max_or_zero(
            [value[1] for value in predicted_errors]
        ),
        "robot_translation_span_m": robot_span[0],
        "robot_rotation_span_deg": robot_span[1],
        "target_translation_span_m": target_span[0],
        "target_rotation_span_deg": target_span[1],
        "per_sample_errors": per_sample_errors,
    }
    if reprojection_values:
        metrics["target_reprojection_rmse_px_mean"] = float(np.mean(reprojection_values))
        metrics["target_reprojection_rmse_px_max"] = float(np.max(reprojection_values))
    return metrics


def quality_warnings(
    metrics: Mapping[str, Any],
    *,
    min_samples: int = 15,
    min_robot_translation_span_m: float = 0.08,
    min_robot_rotation_span_deg: float = 25.0,
    max_translation_rms_m: float = 0.003,
    max_rotation_rms_deg: float = 1.0,
    max_reprojection_rmse_px: float = 0.8,
) -> list[str]:
    warnings: list[str] = []
    if int(metrics.get("sample_count", 0)) < min_samples:
        warnings.append(f"sample_count below {min_samples}; collect more diverse poses.")
    if float(metrics.get("robot_translation_span_m", 0.0)) < min_robot_translation_span_m:
        warnings.append("robot translation span is small; move the target across more of the image.")
    if float(metrics.get("robot_rotation_span_deg", 0.0)) < min_robot_rotation_span_deg:
        warnings.append("robot rotation span is small; add wrist rotations around multiple axes.")
    if float(metrics.get("target_in_gripper_translation_rms_m", 0.0)) > max_translation_rms_m:
        warnings.append("translation residual is high; reject blurry/outlier samples and recalibrate.")
    if float(metrics.get("target_in_gripper_rotation_rms_deg", 0.0)) > max_rotation_rms_deg:
        warnings.append("rotation residual is high; use clearer target views and more tilted poses.")
    reproj = metrics.get("target_reprojection_rmse_px_max")
    if reproj is not None and float(reproj) > max_reprojection_rmse_px:
        warnings.append("marker reprojection error is high; improve focus, lighting, or target geometry.")
    return warnings


def save_samples(
    path: str | Path,
    samples: Sequence[CalibrationSample],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now_utc(),
        "metadata": _jsonable(dict(metadata or {})),
        "samples": [sample.to_dict() for sample in samples],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def load_samples(path: str | Path) -> list[CalibrationSample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("samples", payload if isinstance(payload, list) else [])
    return [CalibrationSample.from_dict(item) for item in items]


def save_result(
    result: CalibrationResult,
    output_dir: str | Path,
    *,
    name: str | None = None,
    save_legacy_npy: bool = True,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = name or f"{_timestamp_for_filename()}_rm65_cam2base"
    paths = {
        "json": output / f"{stem}.json",
        "npz": output / f"{stem}.npz",
    }
    paths["json"].write_text(
        json.dumps(result.to_dict(include_samples=True), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez(
        paths["npz"],
        T_camera_to_base=result.T_camera_to_base,
        T_base_to_camera=result.T_base_to_camera,
        T_target_to_gripper=result.T_target_to_gripper,
        metrics_json=json.dumps(_jsonable(result.metrics), sort_keys=True),
    )
    if save_legacy_npy:
        paths["legacy_npy"] = output / f"{stem}_cam2base.npy"
        np.save(paths["legacy_npy"], result.T_camera_to_base)
    return paths


def load_result(path: str | Path) -> CalibrationResult:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return CalibrationResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        metrics = {}
        if "metrics_json" in data.files:
            raw_metrics = data["metrics_json"]
            if hasattr(raw_metrics, "item"):
                raw_metrics = raw_metrics.item()
            metrics = json.loads(str(raw_metrics))
        T_target_to_gripper = (
            np.asarray(data["T_target_to_gripper"], dtype=np.float64)
            if "T_target_to_gripper" in data.files
            else np.eye(4)
        )
        return CalibrationResult(
            T_camera_to_base=np.asarray(data["T_camera_to_base"], dtype=np.float64),
            T_target_to_gripper=T_target_to_gripper,
            method="loaded_npz",
            metrics=metrics,
            samples=[],
            created_at_utc=_now_utc(),
            metadata={"source": str(path)},
        )
    if path.suffix.lower() == ".npy":
        return CalibrationResult(
            T_camera_to_base=np.load(path),
            T_target_to_gripper=np.eye(4),
            method="loaded_npy",
            metrics={},
            samples=[],
            created_at_utc=_now_utc(),
            metadata={"source": str(path), "legacy": True},
        )
    raise ValueError(f"unsupported calibration file: {path}")


def _mean_target_to_gripper(
    samples: Sequence[CalibrationSample],
    T_camera_to_base: object,
) -> np.ndarray:
    T_camera_to_base = as_transform(T_camera_to_base, name="T_camera_to_base")
    return average_transforms(
        invert_transform(sample.T_gripper_to_base)
        @ T_camera_to_base
        @ sample.T_target_to_camera
        for sample in samples
    )


def _pose_span(transforms: Sequence[object]) -> tuple[float, float]:
    matrices = [as_transform(value) for value in transforms]
    if len(matrices) < 2:
        return 0.0, 0.0
    translation_span = 0.0
    rotation_span = 0.0
    for i, first in enumerate(matrices):
        for second in matrices[i + 1 :]:
            trans, rot = transform_distance(first, second)
            translation_span = max(translation_span, trans)
            rotation_span = max(rotation_span, rot)
    return float(translation_span), float(rotation_span)


def _method_score(metrics: Mapping[str, Any]) -> float:
    translation_mm = float(metrics.get("target_in_gripper_translation_rms_m", 0.0)) * 1000.0
    rotation_deg = float(metrics.get("target_in_gripper_rotation_rms_deg", 0.0))
    return translation_mm + rotation_deg


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def _max_or_zero(values: Sequence[float]) -> float:
    return float(max(values)) if values else 0.0


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _list_or_none(values: Any) -> list[float] | None:
    if values is None:
        return None
    return [float(value) for value in list(values)]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
