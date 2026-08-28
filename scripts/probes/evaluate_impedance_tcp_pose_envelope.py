"""Offline evidence for final composed TCP envelopes on impedance traces.

This probe is deliberately non-production.  It reads historical JSON/JSONL
records, exercises the same pure single-side projector used by real runtimes,
and replays native admittance offset increments through the controller's
anti-windup seam without constructing a robot or invoking IK.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from hardware.rm65.force_compensation import PoseAdmittance, apply_pose_offset
from hardware.rm65.impedance import RM65ImpedanceConfig, RM65PoseImpedanceController
from robot_motion.pose import matrix_to_pose6, pose6_to_matrix
from robot_motion.target_safety.tcp_pose_envelope import (
    TCP_CONTROLLED_FRAME_ID,
    TCPPoseEnvelopeAnchorReceipt,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeSideConfig,
    project_tcp_pose_envelope_side,
)


@dataclass(frozen=True)
class TraceSample:
    side: str
    reference_pose: np.ndarray
    composed_pose: np.ndarray
    offset: np.ndarray
    active_wrench: np.ndarray
    target_offset: np.ndarray


def _matrix(value: object, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _rotation(matrix: np.ndarray) -> Rotation:
    return Rotation.from_matrix(np.array(matrix[:3, :3], dtype=float, copy=True))


def _reference_from_command_and_offset(
    command_pose6: Iterable[float],
    offset: Iterable[float],
) -> np.ndarray:
    command = pose6_to_matrix(command_pose6)
    values = np.asarray(offset, dtype=float).reshape(6)
    reference = np.eye(4, dtype=float)
    reference[:3, 3] = command[:3, 3] - values[:3]
    reference[:3, :3] = (
        Rotation.from_rotvec(values[3:]).inv()
        * _rotation(command)
    ).as_matrix()
    return reference


def _offset_between(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    linear = target[:3, 3] - reference[:3, 3]
    angular = (
        _rotation(target)
        * _rotation(reference).inv()
    ).as_rotvec()
    return np.concatenate((linear, angular)).astype(float)


def _native_samples(path: Path) -> list[TraceSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} has no records")
    side = str(payload.get("arm") or payload.get("side") or "").strip().lower()
    if side not in {"left", "right"}:
        raise ValueError(f"{path} has no valid arm side")
    samples: list[TraceSample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path} record {index} is not an object")
        offset = np.asarray(row.get("offset", [0.0] * 6), dtype=float).reshape(6)
        command_pose6 = row.get("pose_cmd", row.get("command_pose"))
        if command_pose6 is None:
            raise ValueError(f"{path} record {index} has no command pose")
        composed = pose6_to_matrix(command_pose6)
        reference_raw = row.get("reference")
        reference = (
            _reference_from_command_and_offset(command_pose6, offset)
            if reference_raw is None
            else pose6_to_matrix(reference_raw)
        )
        samples.append(
            TraceSample(
                side=side,
                reference_pose=reference,
                composed_pose=composed,
                offset=offset,
                active_wrench=np.asarray(
                    row.get("active_wrench", [0.0] * 6), dtype=float
                ).reshape(6),
                target_offset=np.asarray(
                    row.get("target_offset", offset), dtype=float
                ).reshape(6),
            )
        )
    return samples


def _action_debug_samples(path: Path) -> dict[str, list[TraceSample]]:
    traces: dict[str, list[TraceSample]] = {"left": [], "right": []}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if row.get("event") != "frame":
                continue
            sides = (row.get("position_admittance") or {}).get("sides") or {}
            if not isinstance(sides, dict):
                continue
            for side in ("left", "right"):
                record = sides.get(side)
                if not isinstance(record, dict):
                    continue
                raw = record.get("raw_target_pose_solver_world")
                applied = record.get("applied_target_pose_solver_world")
                if raw is None or applied is None:
                    continue
                reference = _matrix(raw, name=f"{path}:{line_number}:{side}:raw")
                composed = _matrix(
                    applied,
                    name=f"{path}:{line_number}:{side}:applied",
                )
                traces[side].append(
                    TraceSample(
                        side=side,
                        reference_pose=reference,
                        composed_pose=composed,
                        offset=_offset_between(reference, composed),
                        active_wrench=np.asarray(
                            record.get("tared_wrench_n_nm", [0.0] * 6),
                            dtype=float,
                        ).reshape(6),
                        target_offset=np.concatenate(
                            (
                                np.asarray(
                                    record.get("requested_offset_m", [0.0] * 3),
                                    dtype=float,
                                ).reshape(3),
                                np.asarray(
                                    record.get(
                                        "rotation_requested_rotvec_rad",
                                        [0.0] * 3,
                                    ),
                                    dtype=float,
                                ).reshape(3),
                            )
                        ),
                    )
                )
    return {side: rows for side, rows in traces.items() if rows}


def load_traces(paths: Iterable[Path]) -> dict[str, list[TraceSample]]:
    traces: dict[str, list[TraceSample]] = {}
    for path in paths:
        if path.suffix == ".jsonl":
            loaded = _action_debug_samples(path)
            for side, samples in loaded.items():
                traces[f"{path.name}:{side}"] = samples
        else:
            samples = _native_samples(path)
            traces[path.name] = samples
    if not traces:
        raise ValueError("no usable impedance traces were loaded")
    return traces


def _profile_and_anchor(
    samples: list[TraceSample],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    max_rotation_rad: float,
) -> tuple[TCPPoseEnvelopeProfile, TCPPoseEnvelopeAnchorReceipt]:
    side = samples[0].side
    target_hash = f"offline-target:{side}"
    tool_hash = f"offline-tool:{side}"
    profile = TCPPoseEnvelopeProfile(
        profile_id=f"offline-impedance-envelope:{side}",
        target_frame_binding_hash=target_hash,
        tool_frame_binding_hash_by_side={side: tool_hash},
        reference_frame_by_side={side: "offline_arm_base"},
        side_config_by_side={
            side: TCPPoseEnvelopeSideConfig(
                relative_min_xyz_m=tuple(float(value) for value in lower),
                relative_max_xyz_m=tuple(float(value) for value in upper),
                relative_axes="canonical_axes",
                max_rotation_rad=float(max_rotation_rad),
            )
        },
    )
    anchor = TCPPoseEnvelopeAnchorReceipt.capture(
        anchor_id=f"offline-first-reference:{side}",
        profile=profile,
        execution_epoch=1,
        clock_domain_id="offline_trace_index",
        tcp_pose_by_side={side: samples[0].reference_pose},
        captured_at_by_side={side: 0.0},
    )
    return profile, anchor


def _project(
    sample: TraceSample,
    *,
    profile: TCPPoseEnvelopeProfile,
    anchor: TCPPoseEnvelopeAnchorReceipt,
) -> tuple[np.ndarray, bool, float, float]:
    side = sample.side
    result = project_tcp_pose_envelope_side(
        side,
        sample.composed_pose,
        profile=profile,
        target_frame_binding_hash=profile.target_frame_binding_hash,
        tool_frame_binding_hash_by_side=profile.tool_frame_binding_hash_by_side,
        controlled_frame_id=TCP_CONTROLLED_FRAME_ID,
        reference_frame_by_side=profile.reference_frame_by_side,
        execution_epoch=1,
        anchor_receipt=anchor,
    )
    diagnostic = result.side_projection_by_side[side]
    return (
        np.asarray(result.effective_tcp_pose_by_side[side], dtype=float),
        bool(diagnostic.modified),
        float(diagnostic.translation_adjustment_m),
        float(diagnostic.rotation_adjustment_rad),
    )


def evaluate_trace(
    samples: list[TraceSample],
    *,
    tight_translation_m: float,
    tight_rotation_rad: float,
) -> dict[str, object]:
    anchor_pose = samples[0].reference_pose
    positions = np.asarray(
        [sample.composed_pose[:3, 3] - anchor_pose[:3, 3] for sample in samples]
    )
    rotations = np.asarray(
        [
            (
                _rotation(sample.composed_pose)
                * _rotation(anchor_pose).inv()
            ).magnitude()
            for sample in samples
        ],
        dtype=float,
    )
    wide_lower = np.min(positions, axis=0) - 1e-6
    wide_upper = np.max(positions, axis=0) + 1e-6
    wide_rotation = min(math.pi, float(np.max(rotations)) + 1e-6)
    wide_profile, wide_anchor = _profile_and_anchor(
        samples,
        lower=wide_lower,
        upper=wide_upper,
        max_rotation_rad=wide_rotation,
    )
    wide_modified = 0
    wide_max_linear_delta = 0.0
    wide_max_angular_delta = 0.0
    for sample in samples:
        _, modified, linear_delta, angular_delta = _project(
            sample,
            profile=wide_profile,
            anchor=wide_anchor,
        )
        wide_modified += int(modified)
        wide_max_linear_delta = max(wide_max_linear_delta, linear_delta)
        wide_max_angular_delta = max(wide_max_angular_delta, angular_delta)

    tight = float(tight_translation_m)
    tight_profile, tight_anchor = _profile_and_anchor(
        samples,
        lower=np.full(3, -tight),
        upper=np.full(3, tight),
        max_rotation_rad=float(tight_rotation_rad),
    )
    tight_modified = 0
    tight_max_linear_adjustment = 0.0
    tight_max_angular_adjustment = 0.0
    tight_max_translation_violation = 0.0
    tight_max_rotation_violation = 0.0
    for sample in samples:
        effective, modified, linear_delta, angular_delta = _project(
            sample,
            profile=tight_profile,
            anchor=tight_anchor,
        )
        tight_modified += int(modified)
        tight_max_linear_adjustment = max(tight_max_linear_adjustment, linear_delta)
        tight_max_angular_adjustment = max(tight_max_angular_adjustment, angular_delta)
        relative = effective[:3, 3] - anchor_pose[:3, 3]
        tight_max_translation_violation = max(
            tight_max_translation_violation,
            float(np.max(np.maximum(np.abs(relative) - tight, 0.0))),
        )
        angle = float(
            (
                _rotation(effective)
                * _rotation(anchor_pose).inv()
            ).magnitude()
        )
        tight_max_rotation_violation = max(
            tight_max_rotation_violation,
            max(0.0, angle - tight_rotation_rad),
        )

    max_recorded_linear_offset = max(
        float(np.linalg.norm(sample.offset[:3])) for sample in samples
    )
    max_recorded_angular_offset = max(
        float(np.linalg.norm(sample.offset[3:])) for sample in samples
    )
    return {
        "side": samples[0].side,
        "samples": len(samples),
        "nonzero_offset_samples": sum(
            int(np.linalg.norm(sample.offset) > 1e-9) for sample in samples
        ),
        "max_recorded_linear_offset_m": max_recorded_linear_offset,
        "max_recorded_angular_offset_rad": max_recorded_angular_offset,
        "wide_envelope": {
            "modified_samples": wide_modified,
            "max_linear_delta_m": wide_max_linear_delta,
            "max_angular_delta_rad": wide_max_angular_delta,
        },
        "tight_envelope": {
            "translation_half_extent_m": tight,
            "max_rotation_rad": float(tight_rotation_rad),
            "modified_samples": tight_modified,
            "max_linear_adjustment_m": tight_max_linear_adjustment,
            "max_angular_adjustment_rad": tight_max_angular_adjustment,
            "max_translation_violation_m": tight_max_translation_violation,
            "max_rotation_violation_rad": tight_max_rotation_violation,
        },
    }


def evaluate_controller_antiwindup(
    samples: list[TraceSample],
    *,
    tight_translation_m: float,
    tight_rotation_rad: float,
) -> dict[str, object]:
    side = samples[0].side
    profile, anchor = _profile_and_anchor(
        samples,
        lower=np.full(3, -tight_translation_m),
        upper=np.full(3, tight_translation_m),
        max_rotation_rad=tight_rotation_rad,
    )

    def projector(project_side: str, raw_pose: np.ndarray) -> np.ndarray:
        result = project_tcp_pose_envelope_side(
            project_side,
            raw_pose,
            profile=profile,
            target_frame_binding_hash=profile.target_frame_binding_hash,
            tool_frame_binding_hash_by_side=profile.tool_frame_binding_hash_by_side,
            controlled_frame_id=TCP_CONTROLLED_FRAME_ID,
            reference_frame_by_side=profile.reference_frame_by_side,
            execution_epoch=1,
            anchor_receipt=anchor,
        )
        return np.asarray(result.effective_tcp_pose_by_side[project_side], dtype=float)

    controller = RM65PoseImpedanceController(
        arm=side,
        config=RM65ImpedanceConfig(
            tool_name="offline_trace",
            tare_sec=0.0,
            max_linear_offset_m=0.5,
            max_angular_offset_rad=math.pi,
            joint_rate_limiter_enabled=False,
        ),
        robot=None,
    )
    controller.set_final_tcp_pose_projector(projector)
    controller._admittance = PoseAdmittance(  # noqa: SLF001 - deliberate probe.
        enabled_axes=[True] * 6,
        max_linear_offset_m=0.5,
        max_angular_offset_rad=math.pi,
    )
    previous_recorded = np.zeros(6, dtype=float)
    state = np.zeros(6, dtype=float)
    projection_count = 0
    antiwindup_count = 0
    max_final_translation_violation = 0.0
    max_final_rotation_violation = 0.0
    anchor_pose = samples[0].reference_pose
    for sample in samples:
        state = state + (sample.offset - previous_recorded)
        previous_recorded = sample.offset.copy()
        controller._admittance.offset = state.astype(float).tolist()  # noqa: SLF001
        controller._latest_wrench_context = {  # noqa: SLF001
            "offset": state.astype(float).tolist(),
            "active_wrench": sample.active_wrench.astype(float).tolist(),
            "target_offset_after_offset_limit": sample.target_offset.astype(float).tolist(),
        }
        reference_pose6 = matrix_to_pose6(sample.reference_pose)
        effective_offset, modified = controller._project_final_composed_tcp_pose(  # noqa: SLF001
            reference_pose=reference_pose6,
            requested_offset=state.astype(float).tolist(),
        )
        projection_count += int(modified)
        antiwindup_count += int(
            bool(
                controller._latest_wrench_context.get(  # noqa: SLF001
                    "admittance_offset_antiwindup_applied",
                    False,
                )
            )
        )
        state = np.asarray(controller._admittance.offset, dtype=float)  # noqa: SLF001
        final_pose = pose6_to_matrix(
            apply_pose_offset(reference_pose6, effective_offset)
        )
        relative = final_pose[:3, 3] - anchor_pose[:3, 3]
        max_final_translation_violation = max(
            max_final_translation_violation,
            float(
                np.max(
                    np.maximum(
                        np.abs(relative) - tight_translation_m,
                        0.0,
                    )
                )
            ),
        )
        angle = float(
            (
                _rotation(final_pose)
                * _rotation(anchor_pose).inv()
            ).magnitude()
        )
        max_final_rotation_violation = max(
            max_final_rotation_violation,
            max(0.0, angle - tight_rotation_rad),
        )
    return {
        "samples": len(samples),
        "projected_samples": projection_count,
        "antiwindup_applied_samples": antiwindup_count,
        "final_state_linear_norm_m": float(np.linalg.norm(state[:3])),
        "final_state_angular_norm_rad": float(np.linalg.norm(state[3:])),
        "max_final_translation_violation_m": max_final_translation_violation,
        "max_final_rotation_violation_rad": max_final_rotation_violation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--tight-translation-m", type=float, default=0.005)
    parser.add_argument("--tight-rotation-deg", type=float, default=5.0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tight_rotation_rad = math.radians(float(args.tight_rotation_deg))
    traces = load_traces(args.inputs)
    trace_results = {
        name: evaluate_trace(
            samples,
            tight_translation_m=float(args.tight_translation_m),
            tight_rotation_rad=tight_rotation_rad,
        )
        for name, samples in traces.items()
    }
    native_nonzero = [
        (name, samples)
        for name, samples in traces.items()
        if name.endswith(".json")
        and any(np.linalg.norm(sample.offset) > 1e-9 for sample in samples)
    ]
    antiwindup = {
        name: evaluate_controller_antiwindup(
            samples,
            tight_translation_m=float(args.tight_translation_m),
            tight_rotation_rad=tight_rotation_rad,
        )
        for name, samples in native_nonzero
    }
    result = {
        "schema_version": "offline-impedance-tcp-envelope-evidence-v1",
        "inputs": [str(path) for path in args.inputs],
        "trace_results": trace_results,
        "controller_antiwindup": antiwindup,
    }

    failures: list[str] = []
    for name, metrics in trace_results.items():
        wide = metrics["wide_envelope"]
        tight = metrics["tight_envelope"]
        if wide["modified_samples"] != 0:
            failures.append(f"{name}: wide envelope changed recorded commands")
        if tight["max_translation_violation_m"] > 1e-9:
            failures.append(f"{name}: tight translation bound was violated")
        if tight["max_rotation_violation_rad"] > 1e-9:
            failures.append(f"{name}: tight rotation bound was violated")
    if not any(
        metrics["tight_envelope"]["modified_samples"] > 0
        for metrics in trace_results.values()
    ):
        failures.append("tight envelope never exercised the Active projection")
    if native_nonzero and not any(
        metrics["antiwindup_applied_samples"] > 0
        for metrics in antiwindup.values()
    ):
        failures.append("nonzero native trace never exercised anti-windup")
    for name, metrics in antiwindup.items():
        if metrics["max_final_translation_violation_m"] > 1e-9:
            failures.append(f"{name}: closed-loop translation escaped envelope")
        if metrics["max_final_rotation_violation_rad"] > 1e-9:
            failures.append(f"{name}: closed-loop rotation escaped envelope")
    result["failures"] = failures
    result["passed"] = not failures
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
