from __future__ import annotations

import ast
import json
import math
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
    pose6_xyz_rpy_to_matrix,
    transform_distance,
    transform_to_list,
)


SCHEMA_VERSION = "dexteleop.hand_eye.eye_in_hand.v1"
_GRID_CELL_NAMES = (
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)


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
    T_camera_to_gripper: np.ndarray
    T_target_to_base: np.ndarray
    method: str
    metrics: dict[str, Any]
    samples: list[CalibrationSample]
    created_at_utc: str
    T_active_tool_to_arm_end: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.T_camera_to_gripper = as_transform(
            self.T_camera_to_gripper,
            name="T_camera_to_gripper",
        )
        self.T_target_to_base = as_transform(
            self.T_target_to_base,
            name="T_target_to_base",
        )
        if self.T_active_tool_to_arm_end is not None:
            self.T_active_tool_to_arm_end = as_transform(
                self.T_active_tool_to_arm_end,
                name="T_active_tool_to_arm_end",
            )

    @property
    def T_gripper_to_camera(self) -> np.ndarray:
        return invert_transform(self.T_camera_to_gripper)

    @property
    def T_camera_to_active_tool(self) -> np.ndarray:
        return self.T_camera_to_gripper

    @property
    def T_active_tool_to_camera(self) -> np.ndarray:
        return self.T_gripper_to_camera

    @property
    def has_arm_end_transform(self) -> bool:
        return self.T_active_tool_to_arm_end is not None

    @property
    def T_arm_end_to_active_tool(self) -> np.ndarray | None:
        if self.T_active_tool_to_arm_end is None:
            return None
        return invert_transform(self.T_active_tool_to_arm_end)

    @property
    def T_camera_to_arm_end(self) -> np.ndarray:
        if self.T_active_tool_to_arm_end is None:
            return self.T_camera_to_gripper
        return self.T_active_tool_to_arm_end @ self.T_camera_to_gripper

    @property
    def T_arm_end_to_camera(self) -> np.ndarray:
        return invert_transform(self.T_camera_to_arm_end)

    @property
    def T_end_effector_to_camera(self) -> np.ndarray:
        return self.T_arm_end_to_camera

    @property
    def T_camera_to_end_effector(self) -> np.ndarray:
        return self.T_camera_to_arm_end

    def camera_to_base_for_pose(self, T_gripper_to_base: object) -> np.ndarray:
        return as_transform(T_gripper_to_base, name="T_gripper_to_base") @ self.T_camera_to_gripper

    def to_dict(self, *, include_samples: bool = True) -> dict[str, Any]:
        metadata = _result_metadata_with_frame_compatibility(self)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": self.created_at_utc,
            "matrix_convention": {
                "T_camera_to_gripper": (
                    "legacy name: p_active_tool = T_camera_to_gripper @ p_camera; "
                    "the gripper frame is the RM65 active tool/TCP returned by Get_Current_Arm_State"
                ),
                "T_gripper_to_camera": "legacy name: p_camera = T_gripper_to_camera @ p_active_tool",
                "T_camera_to_active_tool": "same matrix as T_camera_to_gripper, with explicit frame name",
                "T_active_tool_to_camera": "same matrix as T_gripper_to_camera, with explicit frame name",
                "T_active_tool_to_arm_end": (
                    "p_arm_end = T_active_tool_to_arm_end @ p_active_tool; "
                    "derived from the active tool pose in the RM65 flange/link6 frame"
                ),
                "T_camera_to_arm_end": (
                    "p_arm_end = T_camera_to_arm_end @ p_camera; "
                    "mechanical arm endpoint/flange/link6 camera extrinsic"
                ),
                "T_arm_end_to_camera": "p_camera = T_arm_end_to_camera @ p_arm_end",
                "T_camera_to_end_effector": (
                    "alias of T_camera_to_arm_end for new files; older files may have used this "
                    "name for the active tool/TCP, so check metadata.frame_compatibility"
                ),
                "T_end_effector_to_camera": "alias of T_arm_end_to_camera for new files",
                "T_target_to_base": "p_base = T_target_to_base @ p_target; fixed scene target estimate",
                "translation_unit": "meters",
            },
            "method": self.method,
            "sample_count": len(self.samples),
            "T_camera_to_gripper": transform_to_list(self.T_camera_to_gripper),
            "T_gripper_to_camera": transform_to_list(self.T_gripper_to_camera),
            "T_camera_to_active_tool": transform_to_list(self.T_camera_to_active_tool),
            "T_active_tool_to_camera": transform_to_list(self.T_active_tool_to_camera),
            "T_end_effector_to_camera": transform_to_list(self.T_gripper_to_camera),
            "T_camera_to_end_effector": transform_to_list(self.T_camera_to_gripper),
            "T_target_to_base": transform_to_list(self.T_target_to_base),
            "metrics": _jsonable(self.metrics),
            "metadata": _jsonable(metadata),
        }
        if self.T_active_tool_to_arm_end is not None:
            T_arm_end_to_active_tool = invert_transform(self.T_active_tool_to_arm_end)
            payload.update(
                {
                    "T_active_tool_to_arm_end": transform_to_list(self.T_active_tool_to_arm_end),
                    "T_arm_end_to_active_tool": transform_to_list(T_arm_end_to_active_tool),
                    "T_camera_to_arm_end": transform_to_list(self.T_camera_to_arm_end),
                    "T_arm_end_to_camera": transform_to_list(self.T_arm_end_to_camera),
                    "T_camera_to_flange": transform_to_list(self.T_camera_to_arm_end),
                    "T_flange_to_camera": transform_to_list(self.T_arm_end_to_camera),
                    "T_camera_to_end_effector": transform_to_list(self.T_camera_to_end_effector),
                    "T_end_effector_to_camera": transform_to_list(self.T_end_effector_to_camera),
                }
            )
        if include_samples:
            payload["samples"] = [sample.to_dict() for sample in self.samples]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationResult":
        samples = [CalibrationSample.from_dict(item) for item in payload.get("samples", [])]
        if "T_camera_to_gripper" in payload:
            T_camera_to_gripper = np.asarray(payload["T_camera_to_gripper"], dtype=np.float64)
        elif "T_gripper_to_camera" in payload:
            T_camera_to_gripper = invert_transform(
                np.asarray(payload["T_gripper_to_camera"], dtype=np.float64)
            )
        else:
            raise ValueError("calibration JSON must contain T_camera_to_gripper or T_gripper_to_camera.")
        metadata = dict(payload.get("metadata") or {})
        frame_info = _infer_active_tool_frame_info(samples, metadata)
        T_active_tool_to_arm_end = _active_tool_to_arm_end_from_payload(payload, samples)
        if T_active_tool_to_arm_end is not None:
            metadata = _metadata_with_active_tool_frame_info(metadata, frame_info)
        return cls(
            T_camera_to_gripper=T_camera_to_gripper,
            T_target_to_base=np.asarray(
                payload.get("T_target_to_base", np.eye(4)),
                dtype=np.float64,
            ),
            method=str(payload.get("method", "unknown")),
            metrics=dict(payload.get("metrics") or {}),
            samples=samples,
            created_at_utc=str(payload.get("created_at_utc") or _now_utc()),
            T_active_tool_to_arm_end=T_active_tool_to_arm_end,
            metadata=metadata,
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


def _tool_frame_record(value: object) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return None


def _finite_pose6(value: object) -> list[float] | None:
    try:
        pose = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        return None
    return [float(item) for item in pose]


def _infer_active_tool_frame_info(
    samples: Sequence[CalibrationSample],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(metadata or {})
    metadata_pose = _finite_pose6(metadata.get("active_tool_pose_flange_m_rad"))
    if metadata_pose is not None:
        return {
            "status": "ok",
            "source": "metadata.active_tool_pose_flange_m_rad",
            "name": metadata.get("active_tool_frame_name", ""),
            "pose_flange_m_rad": metadata_pose,
            "T_active_tool_to_arm_end": pose6_xyz_rpy_to_matrix(metadata_pose),
        }

    records: list[dict[str, Any]] = []
    for sample in samples:
        record = _tool_frame_record(sample.metadata.get("robot_tool_frame"))
        pose = _finite_pose6(record.get("pose") if record is not None else None)
        if record is not None and pose is not None:
            record = dict(record)
            record["pose"] = pose
            records.append(record)
    if not records:
        return {
            "status": "unavailable",
            "reason": "sample metadata does not contain a parseable robot_tool_frame.pose",
        }

    poses = [np.asarray(record["pose"], dtype=np.float64).reshape(6) for record in records]
    first_pose = poses[0]
    if any(not np.allclose(pose, first_pose, atol=1e-9, rtol=0.0) for pose in poses[1:]):
        return {
            "status": "mixed_tool_poses",
            "reason": "samples were captured with different active tool frame poses",
            "sample_count": len(records),
        }

    names = [str(record.get("name", "")).strip() for record in records if record.get("name")]
    pose = [float(item) for item in first_pose]
    return {
        "status": "ok",
        "source": "sample.metadata.robot_tool_frame.pose",
        "name": _most_common_text(names),
        "pose_flange_m_rad": pose,
        "record": records[0],
        "T_active_tool_to_arm_end": pose6_xyz_rpy_to_matrix(pose),
    }


def _active_tool_to_arm_end_from_payload(
    payload: Mapping[str, Any],
    samples: Sequence[CalibrationSample],
) -> np.ndarray | None:
    if payload.get("T_active_tool_to_arm_end") is not None:
        return as_transform(payload["T_active_tool_to_arm_end"], name="T_active_tool_to_arm_end")
    metadata = dict(payload.get("metadata") or {})
    info = _infer_active_tool_frame_info(samples, metadata)
    transform = info.get("T_active_tool_to_arm_end")
    if transform is None:
        return None
    return as_transform(transform, name="inferred_T_active_tool_to_arm_end")


def _metadata_with_active_tool_frame_info(
    metadata: Mapping[str, Any] | None,
    frame_info: Mapping[str, Any],
) -> dict[str, Any]:
    out = dict(metadata or {})
    status = str(frame_info.get("status") or "unknown")
    out.setdefault("robot_pose_frame", "active_tool_tcp")
    out.setdefault("mechanical_arm_end_frame", "arm_link6/flange")
    out["active_tool_frame_inference"] = {
        "status": status,
        "source": frame_info.get("source", ""),
        "reason": frame_info.get("reason", ""),
    }
    if status == "ok":
        out["active_tool_frame_name"] = str(frame_info.get("name", "") or "")
        out["active_tool_pose_flange_m_rad"] = _jsonable(frame_info.get("pose_flange_m_rad"))
        record = frame_info.get("record")
        if record is not None:
            out["active_tool_frame_record"] = _jsonable(record)
    return out


def _result_metadata_with_frame_compatibility(result: CalibrationResult) -> dict[str, Any]:
    metadata = dict(result.metadata)
    metadata.setdefault("robot_pose_frame", "active_tool_tcp")
    metadata.setdefault("mechanical_arm_end_frame", "arm_link6/flange")
    metadata["frame_compatibility"] = {
        "legacy_gripper_fields_frame": "active_tool_tcp",
        "legacy_end_effector_fields_before_2026_07_04": "active_tool_tcp",
        "new_end_effector_fields_frame": (
            "mechanical_arm_end_flange_link6"
            if result.has_arm_end_transform
            else "same_as_legacy_gripper_frame_transform_unavailable"
        ),
        "preferred_mechanical_arm_end_fields": [
            "T_camera_to_arm_end",
            "T_arm_end_to_camera",
            "T_camera_to_end_effector",
            "T_end_effector_to_camera",
        ],
        "preferred_active_tool_tcp_fields": [
            "T_camera_to_active_tool",
            "T_active_tool_to_camera",
            "T_camera_to_gripper",
            "T_gripper_to_camera",
        ],
    }
    return metadata


def _most_common_text(values: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: item[1])[0]


def _hand_eye_methods(cv2: Any) -> dict[str, int]:
    names = {
        "TSAI": "CALIB_HAND_EYE_TSAI",
        "PARK": "CALIB_HAND_EYE_PARK",
        "HORAUD": "CALIB_HAND_EYE_HORAUD",
        "ANDREFF": "CALIB_HAND_EYE_ANDREFF",
        "DANIILIDIS": "CALIB_HAND_EYE_DANIILIDIS",
    }
    return {name: getattr(cv2, attr) for name, attr in names.items() if hasattr(cv2, attr)}


def solve_eye_in_hand(
    samples: Sequence[CalibrationSample],
    *,
    method: str = "AUTO",
    metadata: Mapping[str, Any] | None = None,
    refine: bool = True,
    refine_loss: str = "huber",
    refine_max_nfev: int = 200,
    auto_reject_outliers: bool = False,
    outlier_reject_rounds: int = 2,
    outlier_reject_fraction: float = 0.10,
    outlier_min_samples: int = 20,
) -> CalibrationResult:
    """Solve wrist-mounted camera calibration against a fixed scene target.

    Inputs follow the physical convention:
    - ``T_gripper_to_base`` maps RM65 end-effector coordinates into base.
    - ``T_target_to_camera`` maps the fixed calibration target into camera.
    - Returned ``T_camera_to_gripper`` maps camera coordinates into end-effector.
    - ``T_gripper_to_camera`` is the inverse, i.e. end-effector to camera.
    """

    active_samples = list(samples)
    if len(active_samples) < 3:
        raise ValueError("at least 3 samples are required; 15+ diverse samples are recommended.")
    rejection_history: list[dict[str, Any]] = []
    if auto_reject_outliers:
        rounds = max(0, int(outlier_reject_rounds))
        for round_index in range(rounds):
            if len(active_samples) <= int(outlier_min_samples):
                break
            trial = _solve_eye_in_hand_once(
                active_samples,
                method=method,
                metadata=metadata,
                refine=False,
                refine_loss=refine_loss,
                refine_max_nfev=refine_max_nfev,
            )
            kept, rejected = _reject_worst_samples(
                active_samples,
                trial,
                reject_fraction=float(outlier_reject_fraction),
                min_samples=int(outlier_min_samples),
            )
            if not rejected:
                break
            rejection_history.append(
                {
                    "round": round_index + 1,
                    "input_samples": len(active_samples),
                    "kept_samples": len(kept),
                    "rejected_samples": rejected,
                }
            )
            active_samples = kept

    result = _solve_eye_in_hand_once(
        active_samples,
        method=method,
        metadata=metadata,
        refine=refine,
        refine_loss=refine_loss,
        refine_max_nfev=refine_max_nfev,
    )
    if rejection_history:
        result.metrics["outlier_rejection"] = rejection_history
        result.metadata["outlier_rejection_enabled"] = True
        result.metadata["outlier_rejection_original_sample_count"] = len(samples)
        result.metadata["outlier_rejection_kept_sample_count"] = len(active_samples)
        result.metadata["outlier_rejection_fraction"] = float(outlier_reject_fraction)
        result.metadata["outlier_rejection_min_samples"] = int(outlier_min_samples)
    return result


def _solve_eye_in_hand_once(
    samples: Sequence[CalibrationSample],
    *,
    method: str = "AUTO",
    metadata: Mapping[str, Any] | None = None,
    refine: bool = True,
    refine_loss: str = "huber",
    refine_max_nfev: int = 200,
) -> CalibrationResult:
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

    frame_info = _infer_active_tool_frame_info(samples, metadata)
    result_metadata = _metadata_with_active_tool_frame_info(metadata, frame_info)
    T_active_tool_to_arm_end = frame_info.get("T_active_tool_to_arm_end")
    if T_active_tool_to_arm_end is not None:
        T_active_tool_to_arm_end = as_transform(
            T_active_tool_to_arm_end,
            name="T_active_tool_to_arm_end",
        )

    attempts: list[tuple[float, CalibrationResult]] = []
    failures: dict[str, str] = {}
    for name in candidates:
        try:
            T_camera_to_gripper = _solve_eye_in_hand_with_cv2(samples, cv2, methods[name])
            metrics = evaluate_eye_in_hand(samples, T_camera_to_gripper)
            T_target_to_base = _mean_target_to_base(samples, T_camera_to_gripper)
            if refine:
                try:
                    refined = refine_eye_in_hand(
                        samples,
                        T_camera_to_gripper,
                        initial_T_target_to_base=T_target_to_base,
                        loss=refine_loss,
                        max_nfev=refine_max_nfev,
                    )
                except Exception as exc:
                    refined = None
                    metrics["nonlinear_refinement"] = {"failed": str(exc)}
                if refined is not None:
                    T_camera_to_gripper, T_target_to_base, refine_metrics = refined
                    metrics = evaluate_eye_in_hand(samples, T_camera_to_gripper)
                    metrics["nonlinear_refinement"] = refine_metrics
                elif "nonlinear_refinement" not in metrics:
                    metrics["nonlinear_refinement"] = {"skipped": "scipy.optimize is not available"}
            result = CalibrationResult(
                T_camera_to_gripper=T_camera_to_gripper,
                T_target_to_base=T_target_to_base,
                method=f"{name}+NONLINEAR"
                if bool(metrics.get("nonlinear_refinement", {}).get("success"))
                else name,
                metrics=metrics,
                samples=list(samples),
                created_at_utc=_now_utc(),
                T_active_tool_to_arm_end=T_active_tool_to_arm_end,
                metadata=dict(result_metadata),
            )
            attempts.append((_method_score(metrics), result))
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
            "translation_rms_m": result.metrics["target_in_base_translation_rms_m"],
            "rotation_rms_deg": result.metrics["target_in_base_rotation_rms_deg"],
            "translation_max_m": result.metrics["target_in_base_translation_max_m"],
            "rotation_max_deg": result.metrics["target_in_base_rotation_max_deg"],
        }
    for name, reason in failures.items():
        all_method_metrics[name] = {"failed": reason}
    best.metrics["all_method_metrics"] = all_method_metrics
    best.metrics["quality_warnings"] = quality_warnings(best.metrics)
    return best


def solve_hand_eye(
    samples: Sequence[CalibrationSample],
    *,
    method: str = "AUTO",
    metadata: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> CalibrationResult:
    return solve_eye_in_hand(samples, method=method, metadata=metadata, **kwargs)


def _solve_eye_in_hand_with_cv2(
    samples: Sequence[CalibrationSample],
    cv2: Any,
    method_flag: int,
) -> np.ndarray:
    R_gripper_to_base: list[np.ndarray] = []
    t_gripper_to_base: list[np.ndarray] = []
    R_target_to_camera: list[np.ndarray] = []
    t_target_to_camera: list[np.ndarray] = []
    for sample in samples:
        R_gripper_to_base.append(sample.T_gripper_to_base[:3, :3])
        t_gripper_to_base.append(sample.T_gripper_to_base[:3, 3].reshape(3, 1))
        R_target_to_camera.append(sample.T_target_to_camera[:3, :3])
        t_target_to_camera.append(sample.T_target_to_camera[:3, 3].reshape(3, 1))

    R_camera_to_gripper, t_camera_to_gripper = cv2.calibrateHandEye(
        R_gripper_to_base,
        t_gripper_to_base,
        R_target_to_camera,
        t_target_to_camera,
        method=method_flag,
    )
    return make_transform(R_camera_to_gripper, np.asarray(t_camera_to_gripper).reshape(3))


def refine_eye_in_hand(
    samples: Sequence[CalibrationSample],
    initial_T_camera_to_gripper: object,
    *,
    initial_T_target_to_base: object | None = None,
    loss: str = "huber",
    max_nfev: int = 200,
    translation_sigma_m: float = 0.003,
    rotation_sigma_deg: float = 1.0,
    reprojection_sigma_px: float = 0.8,
    reprojection_weight: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    try:
        from scipy.optimize import least_squares  # type: ignore
    except Exception:
        return None

    cv2 = _require_cv2()
    T_camera_to_gripper = as_transform(
        initial_T_camera_to_gripper,
        name="initial_T_camera_to_gripper",
    )
    T_target_to_base = (
        as_transform(initial_T_target_to_base, name="initial_T_target_to_base")
        if initial_T_target_to_base is not None
        else _mean_target_to_base(samples, T_camera_to_gripper)
    )
    x0 = np.concatenate(
        (
            _transform_to_params(T_camera_to_gripper, cv2),
            _transform_to_params(T_target_to_base, cv2),
        )
    )
    initial_residuals = _nonlinear_residuals(
        x0,
        samples,
        cv2,
        translation_sigma_m=translation_sigma_m,
        rotation_sigma_rad=math.radians(rotation_sigma_deg),
        reprojection_sigma_px=reprojection_sigma_px,
        reprojection_weight=reprojection_weight,
    )
    result = least_squares(
        _nonlinear_residuals,
        x0,
        args=(samples, cv2),
        kwargs={
            "translation_sigma_m": translation_sigma_m,
            "rotation_sigma_rad": math.radians(rotation_sigma_deg),
            "reprojection_sigma_px": reprojection_sigma_px,
            "reprojection_weight": reprojection_weight,
        },
        loss=str(loss),
        f_scale=1.0,
        max_nfev=max(1, int(max_nfev)),
    )
    T_camera_to_gripper_refined = _params_to_transform(result.x[:6], cv2)
    T_target_to_base_refined = _params_to_transform(result.x[6:12], cv2)
    final_residuals = _nonlinear_residuals(
        result.x,
        samples,
        cv2,
        translation_sigma_m=translation_sigma_m,
        rotation_sigma_rad=math.radians(rotation_sigma_deg),
        reprojection_sigma_px=reprojection_sigma_px,
        reprojection_weight=reprojection_weight,
    )
    metrics = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "loss": str(loss),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "initial_residual_rms": _rms(initial_residuals.tolist()),
        "final_residual_rms": _rms(final_residuals.tolist()),
        "translation_sigma_m": float(translation_sigma_m),
        "rotation_sigma_deg": float(rotation_sigma_deg),
        "reprojection_sigma_px": float(reprojection_sigma_px),
        "reprojection_weight": float(reprojection_weight),
        "reprojection_sample_count": _reprojection_sample_count(samples),
    }
    return T_camera_to_gripper_refined, T_target_to_base_refined, metrics


def _nonlinear_residuals(
    params: np.ndarray,
    samples: Sequence[CalibrationSample],
    cv2: Any,
    *,
    translation_sigma_m: float,
    rotation_sigma_rad: float,
    reprojection_sigma_px: float,
    reprojection_weight: float,
) -> np.ndarray:
    T_camera_to_gripper = _params_to_transform(params[:6], cv2)
    T_target_to_base = _params_to_transform(params[6:12], cv2)
    residuals: list[float] = []
    for sample in samples:
        predicted_target_to_base = (
            sample.T_gripper_to_base @ T_camera_to_gripper @ sample.T_target_to_camera
        )
        delta = invert_transform(T_target_to_base) @ predicted_target_to_base
        residuals.extend((delta[:3, 3] / max(1e-9, float(translation_sigma_m))).tolist())
        residuals.extend(
            (
                _rotation_vector_from_matrix(delta[:3, :3], cv2)
                / max(1e-9, float(rotation_sigma_rad))
            ).tolist()
        )
        residuals.extend(
            _reprojection_residuals(
                sample,
                T_camera_to_gripper,
                T_target_to_base,
                cv2,
                reprojection_sigma_px=max(1e-9, float(reprojection_sigma_px)),
                reprojection_weight=float(reprojection_weight),
            )
        )
    return np.asarray(residuals, dtype=np.float64)


def _reprojection_residuals(
    sample: CalibrationSample,
    T_camera_to_gripper: np.ndarray,
    T_target_to_base: np.ndarray,
    cv2: Any,
    *,
    reprojection_sigma_px: float,
    reprojection_weight: float,
) -> list[float]:
    object_points = sample.metadata.get("target_object_points")
    image_points = sample.metadata.get("target_image_points")
    camera_matrix = sample.metadata.get("camera_matrix")
    distortion_coeffs = sample.metadata.get("distortion_coeffs")
    if (
        object_points is None
        or image_points is None
        or camera_matrix is None
        or distortion_coeffs is None
    ):
        return []
    object_array = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_array = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(object_array) == 0 or len(object_array) != len(image_array):
        return []
    T_gripper_to_base = np.asarray(
        sample.metadata.get("reprojection_T_gripper_to_base", sample.T_gripper_to_base),
        dtype=np.float64,
    )
    T_target_to_camera = (
        invert_transform(T_camera_to_gripper)
        @ invert_transform(T_gripper_to_base)
        @ T_target_to_base
    )
    rvec = _rotation_vector_from_matrix(T_target_to_camera[:3, :3], cv2).reshape(3, 1)
    projected, _ = cv2.projectPoints(
        object_array,
        rvec,
        T_target_to_camera[:3, 3].reshape(3, 1),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion_coeffs, dtype=np.float64).reshape(-1),
    )
    diff = projected.reshape(-1, 2) - image_array
    scale = float(reprojection_weight) / math.sqrt(max(1, len(object_array)))
    return (diff.reshape(-1) / float(reprojection_sigma_px) * scale).tolist()


def _transform_to_params(transform: object, cv2: Any) -> np.ndarray:
    matrix = as_transform(transform)
    return np.concatenate((_rotation_vector_from_matrix(matrix[:3, :3], cv2), matrix[:3, 3]))


def _params_to_transform(params: Sequence[float], cv2: Any) -> np.ndarray:
    values = np.asarray(params, dtype=np.float64).reshape(6)
    rotation, _ = cv2.Rodrigues(values[:3].reshape(3, 1))
    return make_transform(rotation, values[3:6])


def _rotation_vector_from_matrix(rotation: object, cv2: Any) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64).reshape(3, 3))
    return np.asarray(rvec, dtype=np.float64).reshape(3)


def _reprojection_sample_count(samples: Sequence[CalibrationSample]) -> int:
    count = 0
    for sample in samples:
        if (
            sample.metadata.get("target_object_points") is not None
            and sample.metadata.get("target_image_points") is not None
            and sample.metadata.get("camera_matrix") is not None
        ):
            count += 1
    return count


def evaluate_eye_in_hand(
    samples: Sequence[CalibrationSample],
    T_camera_to_gripper: object,
) -> dict[str, Any]:
    T_camera_to_gripper = as_transform(T_camera_to_gripper, name="T_camera_to_gripper")
    T_gripper_to_camera = invert_transform(T_camera_to_gripper)
    target_to_base = [
        sample.T_gripper_to_base @ T_camera_to_gripper @ sample.T_target_to_camera
        for sample in samples
    ]
    mean_target = average_transforms(target_to_base)

    target_errors = [transform_distance(mean_target, value) for value in target_to_base]
    predicted_errors: list[tuple[float, float]] = []
    for sample in samples:
        predicted_target_to_camera = (
            T_gripper_to_camera
            @ invert_transform(sample.T_gripper_to_base)
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
                "target_in_base_translation_m": float(target_error[0]),
                "target_in_base_rotation_deg": float(target_error[1]),
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
        "target_in_base_translation_rms_m": _rms([value[0] for value in target_errors]),
        "target_in_base_translation_max_m": _max_or_zero([value[0] for value in target_errors]),
        "target_in_base_rotation_rms_deg": _rms([value[1] for value in target_errors]),
        "target_in_base_rotation_max_deg": _max_or_zero([value[1] for value in target_errors]),
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
        "sample_plan": sample_plan_summary(samples),
    }
    if reprojection_values:
        metrics["target_reprojection_rmse_px_mean"] = float(np.mean(reprojection_values))
        metrics["target_reprojection_rmse_px_max"] = float(np.max(reprojection_values))
    return metrics


def evaluate_hand_eye(
    samples: Sequence[CalibrationSample],
    T_camera_to_gripper: object,
) -> dict[str, Any]:
    return evaluate_eye_in_hand(samples, T_camera_to_gripper)


def sample_plan_summary(samples: Sequence[CalibrationSample]) -> dict[str, Any]:
    distances = [
        float(sample.metadata["target_distance_m"])
        for sample in samples
        if sample.metadata.get("target_distance_m") is not None
    ]
    centers = [
        [float(value) for value in sample.metadata["target_center_norm"]]
        for sample in samples
        if sample.metadata.get("target_center_norm") is not None
    ]
    pose_values = [
        sample.robot_pose_xyz_rpy
        for sample in samples
        if sample.robot_pose_xyz_rpy is not None and len(sample.robot_pose_xyz_rpy) >= 6
    ]

    distance_bins = {"near": 0, "middle": 0, "far": 0}
    for distance in distances:
        if distance < 0.45:
            distance_bins["near"] += 1
        elif distance < 0.75:
            distance_bins["middle"] += 1
        else:
            distance_bins["far"] += 1

    grid_counts = {name: 0 for name in _GRID_CELL_NAMES}
    for center in centers:
        grid_counts[_grid_cell(center[0], center[1])] += 1

    rpy_span_deg = [0.0, 0.0, 0.0]
    if pose_values:
        rpy = np.asarray([pose[3:6] for pose in pose_values], dtype=np.float64)
        rpy_span_deg = np.degrees(np.max(rpy, axis=0) - np.min(rpy, axis=0)).tolist()

    warnings: list[str] = []
    if len(samples) < 25:
        warnings.append("collect at least 25 samples for high-precision calibration; 40 is better.")
    if distances and min(distance_bins.values()) == 0:
        warnings.append("distance coverage is incomplete; collect near, middle, and far target views.")
    if centers:
        filled = sum(1 for value in grid_counts.values() if value > 0)
        if grid_counts["center"] == 0:
            warnings.append("image center has no samples.")
        if filled < 6:
            warnings.append("image-plane coverage is sparse; visit center, corners, and edges.")
    if pose_values and min(rpy_span_deg) < 10.0:
        warnings.append("one or more wrist Euler axes have less than 10 deg span.")

    return {
        "recommended_sample_count": "25-40",
        "distance_bins_m": {"near": "<0.45", "middle": "0.45-0.75", "far": ">=0.75"},
        "distance_bin_counts": distance_bins,
        "distance_range_m": [
            float(min(distances)) if distances else 0.0,
            float(max(distances)) if distances else 0.0,
        ],
        "image_grid_counts": grid_counts,
        "image_grid_filled_cells": sum(1 for value in grid_counts.values() if value > 0),
        "wrist_rpy_span_deg": [float(value) for value in rpy_span_deg],
        "warnings": warnings,
    }


def _grid_cell(x_norm: float, y_norm: float) -> str:
    x = max(0.0, min(0.999999, float(x_norm)))
    y = max(0.0, min(0.999999, float(y_norm)))
    col = int(x * 3.0)
    row = int(y * 3.0)
    return _GRID_CELL_NAMES[row * 3 + col]


def quality_warnings(
    metrics: Mapping[str, Any],
    *,
    min_samples: int = 25,
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
        warnings.append("robot translation span is small; move the wrist camera through more positions.")
    if float(metrics.get("robot_rotation_span_deg", 0.0)) < min_robot_rotation_span_deg:
        warnings.append("robot rotation span is small; add wrist rotations around multiple axes.")
    if float(metrics.get("target_in_base_translation_rms_m", 0.0)) > max_translation_rms_m:
        warnings.append("translation residual is high; reject blurry/outlier samples and recalibrate.")
    if float(metrics.get("target_in_base_rotation_rms_deg", 0.0)) > max_rotation_rms_deg:
        warnings.append("rotation residual is high; use clearer target views and more tilted poses.")
    reproj = metrics.get("target_reprojection_rmse_px_max")
    if reproj is not None and float(reproj) > max_reprojection_rmse_px:
        warnings.append("marker reprojection error is high; improve focus, lighting, or target geometry.")
    sample_plan = metrics.get("sample_plan")
    if isinstance(sample_plan, Mapping):
        warnings.extend(str(item) for item in sample_plan.get("warnings", []))
    return warnings


def _reject_worst_samples(
    samples: Sequence[CalibrationSample],
    result: CalibrationResult,
    *,
    reject_fraction: float,
    min_samples: int,
) -> tuple[list[CalibrationSample], list[dict[str, Any]]]:
    removable = max(0, len(samples) - int(min_samples))
    reject_count = min(removable, max(1, int(math.ceil(len(samples) * float(reject_fraction)))))
    if reject_count <= 0:
        return list(samples), []
    errors = list(result.metrics.get("per_sample_errors") or [])
    if not errors:
        return list(samples), []
    ranked = sorted(
        errors,
        key=lambda item: (
            float(item.get("target_in_base_translation_m", 0.0)) * 1000.0
            + float(item.get("target_in_base_rotation_deg", 0.0))
        ),
        reverse=True,
    )
    rejected_indices = {int(item["sample_index"]) for item in ranked[:reject_count]}
    kept: list[CalibrationSample] = []
    rejected: list[dict[str, Any]] = []
    for position, sample in enumerate(samples, start=1):
        sample_index = int(sample.metadata.get("sample_index", position))
        if sample_index in rejected_indices:
            match = next(
                (item for item in ranked if int(item.get("sample_index", -1)) == sample_index),
                {},
            )
            rejected.append(
                {
                    "sample_index": sample_index,
                    "target_in_base_translation_m": match.get("target_in_base_translation_m"),
                    "target_in_base_rotation_deg": match.get("target_in_base_rotation_deg"),
                    "target_reprojection_rmse_px": match.get("target_reprojection_rmse_px"),
                }
            )
        else:
            kept.append(sample)
    return kept, rejected


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
    stem = name or f"{_timestamp_for_filename()}_rm65_gripper2camera"
    paths = {
        "json": output / f"{stem}.json",
        "npz": output / f"{stem}.npz",
    }
    paths["json"].write_text(
        json.dumps(result.to_dict(include_samples=True), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    npz_payload: dict[str, Any] = {
        "T_camera_to_gripper": result.T_camera_to_gripper,
        "T_gripper_to_camera": result.T_gripper_to_camera,
        "T_camera_to_active_tool": result.T_camera_to_active_tool,
        "T_active_tool_to_camera": result.T_active_tool_to_camera,
        "T_target_to_base": result.T_target_to_base,
        "metrics_json": json.dumps(_jsonable(result.metrics), sort_keys=True),
        "metadata_json": json.dumps(
            _jsonable(_result_metadata_with_frame_compatibility(result)),
            sort_keys=True,
        ),
    }
    if result.T_active_tool_to_arm_end is not None:
        npz_payload.update(
            {
                "T_active_tool_to_arm_end": result.T_active_tool_to_arm_end,
                "T_arm_end_to_active_tool": result.T_arm_end_to_active_tool,
                "T_camera_to_arm_end": result.T_camera_to_arm_end,
                "T_arm_end_to_camera": result.T_arm_end_to_camera,
                "T_camera_to_end_effector": result.T_camera_to_end_effector,
                "T_end_effector_to_camera": result.T_end_effector_to_camera,
            }
        )
    np.savez(paths["npz"], **npz_payload)
    if save_legacy_npy:
        paths["legacy_npy"] = output / f"{stem}_gripper2camera.npy"
        np.save(paths["legacy_npy"], result.T_gripper_to_camera)
        paths["camera_to_gripper_npy"] = output / f"{stem}_cam2gripper.npy"
        np.save(paths["camera_to_gripper_npy"], result.T_camera_to_gripper)
        if result.T_active_tool_to_arm_end is not None:
            paths["arm_end_to_camera_npy"] = output / f"{stem}_arm_end2camera.npy"
            np.save(paths["arm_end_to_camera_npy"], result.T_arm_end_to_camera)
            paths["camera_to_arm_end_npy"] = output / f"{stem}_cam2arm_end.npy"
            np.save(paths["camera_to_arm_end_npy"], result.T_camera_to_arm_end)
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
        metadata = {"source": str(path)}
        if "metadata_json" in data.files:
            raw_metadata = data["metadata_json"]
            if hasattr(raw_metadata, "item"):
                raw_metadata = raw_metadata.item()
            metadata.update(json.loads(str(raw_metadata)))
        if "T_camera_to_gripper" in data.files:
            T_camera_to_gripper = np.asarray(data["T_camera_to_gripper"], dtype=np.float64)
        elif "T_gripper_to_camera" in data.files:
            T_camera_to_gripper = invert_transform(
                np.asarray(data["T_gripper_to_camera"], dtype=np.float64)
            )
        else:
            raise ValueError(f"npz calibration missing hand-eye matrix: {path}")
        T_target_to_base = (
            np.asarray(data["T_target_to_base"], dtype=np.float64)
            if "T_target_to_base" in data.files
            else np.eye(4)
        )
        T_active_tool_to_arm_end = (
            np.asarray(data["T_active_tool_to_arm_end"], dtype=np.float64)
            if "T_active_tool_to_arm_end" in data.files
            else None
        )
        return CalibrationResult(
            T_camera_to_gripper=T_camera_to_gripper,
            T_target_to_base=T_target_to_base,
            method="loaded_npz",
            metrics=metrics,
            samples=[],
            created_at_utc=_now_utc(),
            T_active_tool_to_arm_end=T_active_tool_to_arm_end,
            metadata=metadata,
        )
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
        lower_name = path.stem.lower()
        is_arm_end_matrix = (
            "arm_end" in lower_name
            or "flange" in lower_name
            or "end_effector" in lower_name
        )
        if (
            "cam2gripper" in lower_name
            or "camera_to_gripper" in lower_name
            or "cam2arm_end" in lower_name
            or "camera_to_arm_end" in lower_name
            or "cam2flange" in lower_name
            or "camera_to_flange" in lower_name
        ):
            T_camera_to_gripper = matrix
        else:
            T_camera_to_gripper = invert_transform(matrix)
        return CalibrationResult(
            T_camera_to_gripper=T_camera_to_gripper,
            T_target_to_base=np.eye(4),
            method="loaded_npy",
            metrics={},
            samples=[],
            created_at_utc=_now_utc(),
            T_active_tool_to_arm_end=np.eye(4) if is_arm_end_matrix else None,
            metadata={
                "source": str(path),
                "legacy": not is_arm_end_matrix,
                "robot_pose_frame": "arm_link6/flange" if is_arm_end_matrix else "active_tool_tcp",
                "npy_frame": "mechanical_arm_end" if is_arm_end_matrix else "active_tool_tcp",
            },
        )
    raise ValueError(f"unsupported calibration file: {path}")


def _mean_target_to_base(
    samples: Sequence[CalibrationSample],
    T_camera_to_gripper: object,
) -> np.ndarray:
    T_camera_to_gripper = as_transform(T_camera_to_gripper, name="T_camera_to_gripper")
    return average_transforms(
        sample.T_gripper_to_base @ T_camera_to_gripper @ sample.T_target_to_camera
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
    translation_mm = float(metrics.get("target_in_base_translation_rms_m", 0.0)) * 1000.0
    rotation_deg = float(metrics.get("target_in_base_rotation_rms_deg", 0.0))
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
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
