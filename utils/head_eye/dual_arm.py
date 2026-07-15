from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from typing import Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    __package__ = "utils.head_eye"

from .calibration import CalibrationResult, load_result, save_result
from .transforms import (
    invert_transform,
    make_transform,
    rotation_matrix_from_euler_xyz,
    transform_to_list,
)


DEFAULT_DUAL_ARM_URDF = "assets/rm65_inspire_tac/rm65_dual_arm.urdf"
LEFT_ARM_JOINT = "left_arm"
RIGHT_ARM_JOINT = "right_arm"


@dataclass(frozen=True)
class DualArmBaseTransforms:
    urdf_path: Path
    left_base_link: str
    right_base_link: str
    T_left_base_to_world: np.ndarray
    T_right_base_to_world: np.ndarray

    @property
    def T_world_to_left_base(self) -> np.ndarray:
        return invert_transform(self.T_left_base_to_world)

    @property
    def T_world_to_right_base(self) -> np.ndarray:
        return invert_transform(self.T_right_base_to_world)

    @property
    def T_right_base_to_left_base(self) -> np.ndarray:
        return self.T_world_to_left_base @ self.T_right_base_to_world

    @property
    def T_left_base_to_right_base(self) -> np.ndarray:
        return self.T_world_to_right_base @ self.T_left_base_to_world

    def source_to_target(self, source_arm: str, target_arm: str) -> np.ndarray:
        source = _normalize_arm(source_arm, name="source_arm")
        target = _normalize_arm(target_arm, name="target_arm")
        if source == target:
            return np.eye(4, dtype=np.float64)
        if source == "right" and target == "left":
            return self.T_right_base_to_left_base
        return self.T_left_base_to_right_base


def load_dual_arm_base_transforms(
    urdf_path: str | Path = DEFAULT_DUAL_ARM_URDF,
) -> DualArmBaseTransforms:
    path = Path(urdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"dual-arm URDF not found: {path}")

    root = ET.parse(path).getroot()
    left_child, T_left = _fixed_joint_transform(root, LEFT_ARM_JOINT)
    right_child, T_right = _fixed_joint_transform(root, RIGHT_ARM_JOINT)
    return DualArmBaseTransforms(
        urdf_path=path,
        left_base_link=left_child,
        right_base_link=right_child,
        T_left_base_to_world=T_left,
        T_right_base_to_world=T_right,
    )


def derive_camera_to_arm_base(
    T_camera_to_source_base: np.ndarray,
    *,
    source_arm: str,
    target_arm: str,
    dual_arm: DualArmBaseTransforms,
) -> np.ndarray:
    T_source_base_to_target_base = dual_arm.source_to_target(source_arm, target_arm)
    return T_source_base_to_target_base @ T_camera_to_source_base


def derive_calibration_result(
    source_result: CalibrationResult,
    *,
    source_arm: str,
    target_arm: str,
    dual_arm: DualArmBaseTransforms,
    source_path: str | Path | None = None,
) -> CalibrationResult:
    source = _normalize_arm(source_arm, name="source_arm")
    target = _normalize_arm(target_arm, name="target_arm")
    T_source_base_to_target_base = dual_arm.source_to_target(source, target)
    metadata = dict(source_result.metadata)
    metadata.update(
        {
            "derived_from_dual_arm_urdf": True,
            "source_calibration": str(source_path) if source_path is not None else "",
            "source_arm": source,
            "target_arm": target,
            "dual_arm_urdf": str(dual_arm.urdf_path),
            "left_base_link": dual_arm.left_base_link,
            "right_base_link": dual_arm.right_base_link,
            "T_source_base_to_target_base": transform_to_list(T_source_base_to_target_base),
            "T_target_base_to_source_base": transform_to_list(
                invert_transform(T_source_base_to_target_base)
            ),
        }
    )
    metrics = dict(source_result.metrics)
    metrics["derived_dual_arm_urdf"] = {
        "source_arm": source,
        "target_arm": target,
        "dual_arm_urdf": str(dual_arm.urdf_path),
    }
    return CalibrationResult(
        T_camera_to_base=T_source_base_to_target_base @ source_result.T_camera_to_base,
        T_target_to_gripper=np.eye(4, dtype=np.float64),
        method=f"derived_dual_arm_urdf_from_{source}_to_{target}",
        metrics=metrics,
        samples=[],
        created_at_utc=_now_utc(),
        metadata=metadata,
    )


def describe_dual_arm_transforms(dual_arm: DualArmBaseTransforms) -> str:
    lines = [
        f"dual_arm_urdf={dual_arm.urdf_path}",
        f"left_base_link={dual_arm.left_base_link}",
        f"right_base_link={dual_arm.right_base_link}",
        "T_left_base_to_world:",
        _matrix_string(dual_arm.T_left_base_to_world),
        "T_right_base_to_world:",
        _matrix_string(dual_arm.T_right_base_to_world),
        "T_right_base_to_left_base:",
        _matrix_string(dual_arm.T_right_base_to_left_base),
        "T_left_base_to_right_base:",
        _matrix_string(dual_arm.T_left_base_to_right_base),
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive left/right RM65 hand-eye calibration from the fixed dual-arm URDF transform."
    )
    parser.add_argument("--calibration", required=True, help="Source hand-eye JSON/NPZ/Npy file.")
    parser.add_argument("--source-arm", choices=("left", "right"), required=True)
    parser.add_argument("--target-arm", choices=("left", "right"), required=True)
    parser.add_argument("--dual-arm-urdf", default=DEFAULT_DUAL_ARM_URDF)
    parser.add_argument("--output-dir", default="teleop_records/handeye")
    parser.add_argument("--name", default="")
    parser.add_argument("--no-save", action="store_true", help="Only print transforms.")
    parser.add_argument("--save-legacy-npy", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.source_arm == args.target_arm:
        print("source and target arm are the same; output equals input calibration.")
    dual_arm = load_dual_arm_base_transforms(args.dual_arm_urdf)
    source_result = load_result(args.calibration)
    derived = derive_calibration_result(
        source_result,
        source_arm=args.source_arm,
        target_arm=args.target_arm,
        dual_arm=dual_arm,
        source_path=args.calibration,
    )
    print(describe_dual_arm_transforms(dual_arm))
    print(f"T_camera_to_{args.target_arm}_base:")
    print(_matrix_string(derived.T_camera_to_base))
    if args.no_save:
        return 0

    source_stem = Path(args.calibration).stem
    name = args.name or f"{source_stem}_derived_{args.target_arm}_base"
    paths = save_result(
        derived,
        args.output_dir,
        name=name,
        save_legacy_npy=bool(args.save_legacy_npy),
    )
    print("Saved:")
    for label, path in paths.items():
        print(f"  {label}: {path.resolve()}")
    return 0


def _fixed_joint_transform(root: ET.Element, joint_name: str) -> tuple[str, np.ndarray]:
    joint = root.find(f"./joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"URDF joint '{joint_name}' not found.")
    if joint.attrib.get("type") != "fixed":
        raise ValueError(f"URDF joint '{joint_name}' must be fixed.")
    child = joint.find("child")
    origin = joint.find("origin")
    if child is None or not child.attrib.get("link"):
        raise ValueError(f"URDF joint '{joint_name}' has no child link.")
    xyz, rpy = _origin_xyz_rpy(origin)
    return child.attrib["link"], make_transform(rotation_matrix_from_euler_xyz(*rpy), xyz)


def _origin_xyz_rpy(origin: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if origin is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    xyz = _parse_vector(origin.attrib.get("xyz", "0 0 0"))
    rpy = _parse_vector(origin.attrib.get("rpy", "0 0 0"))
    return xyz, rpy


def _parse_vector(value: str) -> np.ndarray:
    parts = [float(part) for part in str(value).replace(",", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"expected 3-vector, got: {value!r}")
    return np.asarray(parts, dtype=np.float64)


def _normalize_arm(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"{name} must be 'left' or 'right', got {value!r}")
    return normalized


def _matrix_string(matrix: np.ndarray) -> str:
    return np.array2string(np.asarray(matrix, dtype=np.float64), precision=8, suppress_small=False)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
