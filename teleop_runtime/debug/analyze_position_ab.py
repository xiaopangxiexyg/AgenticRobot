"""Offline first/second-layer analysis for a position-control debug capture.

This tool is intentionally read-only with respect to the runtime/control path.  It
consumes the full ``action_debug`` trace and the position-control sidecars, then:

* attributes target/qref/qcmd and post-IK/planner errors (layer 1); and
* replays the recorded desk-frame targets through the current dual-arm IK with a
  small set of diagnostic variants (layer 2); and
* splits the recorded FK error by tracker state/speed and runs unilateral
  teacher-forced replays (layer 3); and
* compares one, two, and four solver iterations plus a collision-barrier-off
  diagnostic (layer 4).

The replay is not a hardware replay.  It only constructs the existing
``DualArmCollisionAwareRetargeter`` in-process and never dispatches commands.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_motion.frames import TOOL_TCP_FRAME
from robot_motion.pose_smoothing import build_bimanual_pose_smoother
from robot_motion.runtime import EffectorCommand, TeleopCommand, TeleopObservation
from robot_motion.transforms import as_pose_matrix
from robot_motion.rm65.ik_tuning import IKObjectiveWeights, load_ik_tuning_config
from teleop_runtime.configs import load_default_pose_smoothing_config
from teleop_runtime.retargeters.dual_arm_collision_aware import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    DualArmTargetContext,
)


SIDES = ("left", "right")
ARM_MAX_ANGVEL_RAD_S = np.asarray((0.7, 0.7, 0.9, 1.5, 1.5, 2.0), dtype=float)
SOLVER_HZ = 60.0
TOOL_TCP_Z_M = 0.14


def _rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _matrix(value: Any) -> np.ndarray | None:
    if isinstance(value, Mapping) and "xyz_m" in value and "rpy_deg" in value:
        try:
            out = np.eye(4, dtype=float)
            out[:3, 3] = np.asarray(value["xyz_m"], dtype=float).reshape(3)
            out[:3, :3] = R.from_euler(
                "xyz", np.asarray(value["rpy_deg"], dtype=float).reshape(3), degrees=True
            ).as_matrix()
            return out
        except (TypeError, ValueError):
            return None
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if out.shape != (4, 4) or not np.all(np.isfinite(out)):
        return None
    return out


def _qpos(value: Any) -> np.ndarray | None:
    if isinstance(value, Mapping):
        value = value.get("deg")
        if value is None:
            return None
        try:
            return np.deg2rad(np.asarray(value, dtype=float).reshape(6))
        except (TypeError, ValueError):
            return None
    try:
        out = np.asarray(value, dtype=float).reshape(6)
    except (TypeError, ValueError):
        return None
    return out if np.all(np.isfinite(out)) else None


def _pose_error(a: np.ndarray | None, b: np.ndarray | None) -> tuple[float, float] | None:
    if a is None or b is None:
        return None
    translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rotation = float(np.rad2deg((R.from_matrix(a[:3, :3].T @ b[:3, :3])).magnitude()))
    return translation, rotation


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"n": 0, "p50": None, "p95": None, "max": None, "mean": None}
    ordered = np.asarray(finite, dtype=float)
    return {
        "n": int(ordered.size),
        "p50": float(np.percentile(ordered, 50)),
        "p95": float(np.percentile(ordered, 95)),
        "max": float(np.max(ordered)),
        "mean": float(np.mean(ordered)),
    }


def _abs_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    return _stats([abs(float(value)) for value in values])


def _pose_metric_stats(samples: Sequence[tuple[float, float]]) -> dict[str, Any]:
    return {
        "translation_m": _stats([item[0] for item in samples]),
        "rotation_deg": _stats([item[1] for item in samples]),
    }


def _tool_matrix() -> np.ndarray:
    tool = np.eye(4, dtype=float)
    tool[2, 3] = TOOL_TCP_Z_M
    return tool


def _load_capture(capture_dir: Path) -> tuple[list[Mapping[str, Any]], dict[tuple[int, str], Mapping[str, Any]], list[Mapping[str, Any]]]:
    action_candidates = sorted(capture_dir.glob("action_debug_*.jsonl"))
    if not action_candidates:
        raise FileNotFoundError(f"Missing action_debug_*.jsonl in {capture_dir}")
    action_path = action_candidates[-1]
    reference_path = capture_dir / "position_reference.jsonl"
    command_path = capture_dir / "position_command_150.jsonl"
    action_frames = [row for row in _rows(action_path) if row.get("event") == "frame"]
    action_frames.sort(key=lambda row: int(row.get("step_index", 0)))
    references: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in _rows(reference_path):
        sequence = int(row.get("reference_sequence", -1))
        for side in SIDES:
            reference = _mapping(_mapping(row.get("references")).get(side))
            if reference:
                references[(sequence, side)] = reference
    commands = [
        row
        for row in _rows(command_path)
        if bool(row.get("sdk_accepted"))
    ]
    return action_frames, references, commands


def _layer_one(
    capture_dir: Path,
    action_frames: Sequence[Mapping[str, Any]],
    references: Mapping[tuple[int, str], Mapping[str, Any]],
    commands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "capture_dir": str(capture_dir),
        "action_frames": len(action_frames),
        "reference_rows": len(references),
        "accepted_command_rows": len(commands),
        "stage_errors": {},
        "qpos_raw_to_qref_deg": {},
        "by_state": {},
        "candidate_selection": {},
        "output_smoothing": {},
        "execution_step": {},
        "planner": {},
    }

    qcmd_by_ref: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    qcmd_pose_by_ref: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    planner_rows: list[Mapping[str, Any]] = []
    for row in commands:
        raw_sequence = row.get("reference_sequence")
        if raw_sequence is None:
            continue
        sequence = int(raw_sequence)
        side = str(row.get("side", ""))
        if side not in SIDES:
            continue
        q_cmd = row.get("q_cmd")
        if q_cmd is not None:
            try:
                qcmd_by_ref[(sequence, side)].append(np.asarray(q_cmd, dtype=float).reshape(6))
            except (TypeError, ValueError):
                pass
        metadata = _mapping(row.get("metadata"))
        pose = _matrix(metadata.get("sdk_accepted_qcmd_pose_urdf_root"))
        if pose is not None:
            qcmd_pose_by_ref[(sequence, side)].append(pose)
        planner_rows.append(row)

    stage: dict[str, dict[str, list[tuple[float, float]]]] = {
        side: {"target_to_qref": [], "qref_to_qcmd": [], "target_to_qcmd": []}
        for side in SIDES
    }
    qpos_delta: dict[str, list[float]] = {side: [] for side in SIDES}
    by_state: dict[str, dict[str, dict[str, list[tuple[float, float]]]]] = defaultdict(
        lambda: {side: {"target_to_qref": [], "qref_to_qcmd": [], "target_to_qcmd": []} for side in SIDES}
    )

    for (sequence, side), reference in references.items():
        metadata = _mapping(reference.get("frame_metadata"))
        target = _matrix(metadata.get("source_target_pose_urdf_root"))
        qref = _matrix(metadata.get("q_ref_pose_urdf_root"))
        qref_q = _qpos(reference.get("q_ref"))
        raw_q = _qpos(reference.get("q_ik_raw"))
        if target is None or qref is None:
            continue
        state = str(_mapping(metadata.get("target_activity")).get("state") or "UNKNOWN")
        qcmd_poses = qcmd_pose_by_ref.get((sequence, side), [])
        pair = _pose_error(target, qref)
        if pair is not None:
            stage[side]["target_to_qref"].append(pair)
            by_state[state][side]["target_to_qref"].append(pair)
        for qcmd in qcmd_poses:
            qref_cmd = _pose_error(qref, qcmd)
            target_cmd = _pose_error(target, qcmd)
            if qref_cmd is not None:
                stage[side]["qref_to_qcmd"].append(qref_cmd)
                by_state[state][side]["qref_to_qcmd"].append(qref_cmd)
            if target_cmd is not None:
                stage[side]["target_to_qcmd"].append(target_cmd)
                by_state[state][side]["target_to_qcmd"].append(target_cmd)
        if raw_q is not None and qref_q is not None:
            qpos_delta[side].append(float(np.rad2deg(np.max(np.abs(raw_q - qref_q)))))

    for side in SIDES:
        result["stage_errors"][side] = {
            name: _pose_metric_stats(values) for name, values in stage[side].items()
        }
        result["qpos_raw_to_qref_deg"][side] = _stats(qpos_delta[side])
    for state, state_values in by_state.items():
        result["by_state"][state] = {
            side: {name: _pose_metric_stats(values) for name, values in side_values.items()}
            for side, side_values in state_values.items()
        }

    branches: dict[str, Counter[str]] = {side: Counter() for side in SIDES}
    candidate_counts: dict[str, Counter[str]] = {side: Counter() for side in SIDES}
    position_worsen: dict[str, list[float]] = {side: [] for side in SIDES}
    rotation_improve: dict[str, list[float]] = {side: [] for side in SIDES}
    worsen_and_improve: dict[str, list[float]] = {side: [] for side in SIDES}
    smoothing_delta: dict[str, list[float]] = {side: [] for side in SIDES}
    smoothing_applied: Counter[str] = Counter()
    step_ratios: list[float] = []
    step_limited = 0
    for frame in action_frames:
        for side in SIDES:
            metadata = _mapping(_mapping(frame.get("command_by_side")).get(side)).get("metadata")
            metadata = _mapping(metadata)
            branches[side][str(metadata.get("ik_branch_candidate", "unknown"))] += 1
            selection = _mapping(metadata.get("ik_candidate_selection"))
            candidate_counts[side][str(selection.get("reason") or selection.get("status") or "none")] += 1
            selected = _mapping(selection.get("selected_metadata"))
            before = _mapping(selected.get("dual_pose_before")).get(side)
            after = _mapping(selected.get("dual_pose_after")).get(side)
            before = _mapping(before)
            after = _mapping(after)
            if before and after:
                dp = float(after.get("translation_m", 0.0)) - float(before.get("translation_m", 0.0))
                dr = float(after.get("rotation_deg", 0.0)) - float(before.get("rotation_deg", 0.0))
                if dp > 1e-8:
                    position_worsen[side].append(dp)
                if dr < -1e-8:
                    rotation_improve[side].append(-dr)
                if dp > 1e-8 and dr < -1e-8:
                    worsen_and_improve[side].append(dp)
            output = _mapping(metadata.get("output_qpos_smoothing"))
            delta = output.get("raw_to_output_max_delta_deg")
            if delta is not None:
                smoothing_delta[side].append(float(delta))
            if bool(output.get("applied")):
                smoothing_applied[side] += 1
            execution = _mapping(metadata.get("execution_step_metadata"))
            ratio = execution.get("joint_velocity_limit_ratio")
            if ratio is not None:
                step_ratios.append(float(ratio))
            if bool(execution.get("step_limited")):
                step_limited += 1

    result["candidate_selection"] = {
        side: {
            "branch_counts": dict(branches[side]),
            "reason_counts": dict(candidate_counts[side]),
            "position_worsen_delta_m": _stats(position_worsen[side]),
            "rotation_improve_delta_deg": _stats(rotation_improve[side]),
            "position_worsen_and_rotation_improve_count": len(worsen_and_improve[side]),
        }
        for side in SIDES
    }
    result["output_smoothing"] = {
        side: {
            "applied_count": int(smoothing_applied[side]),
            "raw_to_output_max_delta_deg": _stats(smoothing_delta[side]),
        }
        for side in SIDES
    }
    result["execution_step"] = {
        "step_limited_frame_count": step_limited,
        "joint_velocity_limit_ratio": _stats(step_ratios),
    }

    planner_bool_fields = (
        "trajectory_endpoint_velocity_limited",
        "trajectory_velocity_limited",
        "trajectory_acceleration_limited",
        "trajectory_safety_envelope_limited",
        "trajectory_command_step_limited",
        "trajectory_target_preempted",
    )
    planner: dict[str, Any] = {field: 0 for field in planner_bool_fields}
    ages: list[float] = []
    for row in planner_rows:
        metadata = _mapping(row.get("metadata"))
        for field in planner_bool_fields:
            if bool(metadata.get(field)):
                planner[field] += 1
        age = row.get("target_age_sec")
        if age is not None and np.isfinite(float(age)):
            ages.append(float(age))
    planner["accepted_rows"] = len(planner_rows)
    planner["target_age_sec"] = _stats(ages)
    result["planner"] = planner
    return result


class _RecordedBasePoseSource:
    def __init__(self, *, side: str, frame: Mapping[str, Any]):
        self.side = side
        command = _mapping(_mapping(frame.get("command_by_side")).get(side))
        self.metadata = _mapping(command.get("metadata"))

    def retarget(self, observation: TeleopObservation, calibration: Any = None) -> TeleopCommand:
        source_base = _matrix(self.metadata.get("source_target_pose_base"))
        source_desk = _matrix(self.metadata.get("source_target_pose_desk"))
        if source_base is None or source_desk is None:
            raise ValueError(f"Missing recorded source target for {self.side}")
        command = TeleopCommand(timestamp=observation.timestamp)
        command.add_command(
            EffectorCommand(
                effector_name=f"{self.side}_arm",
                side=self.side,
                command_type="cartesian_pose",
                value=source_base,
                frame_name="arm_base",
                metadata={
                    "target_pose_desk": source_desk,
                    "target_pose_tcp_desk": source_desk,
                    "target_pose_base": source_base,
                    "target_pose_tcp_base": source_base,
                    "arm_controlled_frame": TOOL_TCP_FRAME,
                    "target_pose_controlled_frame": TOOL_TCP_FRAME,
                    "tracker_motion_intent": dict(_mapping(self.metadata.get("target_activity"))),
                },
            )
        )
        return command


def _make_replay(
    *,
    first_frame: Mapping[str, Any],
    tuning: Any,
    post_smoothing: bool,
    execution_limit: bool,
    candidate_selection: bool,
    solve_max_iterations: int = 1,
    collision_barrier_gain: float | None = None,
    collision_min_distance_m: float | None = None,
) -> DualArmCollisionAwareRetargeter:
    current_q: dict[str, np.ndarray] = {}
    for side in SIDES:
        backend = _mapping(_mapping(first_frame.get("observation_backend_by_side")).get(side))
        value = _qpos(backend.get("current_qpos"))
        if value is None:
            raise ValueError(f"Missing initial qpos for {side}")
        current_q[side] = value
    step = ARM_MAX_ANGVEL_RAD_S / SOLVER_HZ
    config_kwargs: dict[str, Any] = {
        "control_mode": "absolute",
        "orientation_mode": "full",
        "translation_scale": 1.0,
        "scene_profile": "dual_arm_hand_proxy_open",
        "contact_mode": "boundary_slide",
        "solver_frequency": SOLVER_HZ,
        "solve_max_iterations": int(solve_max_iterations),
        "enforce_realistic_joint_step": bool(execution_limit),
        "left_execution_max_joint_step_rad": step if execution_limit else None,
        "right_execution_max_joint_step_rad": step if execution_limit else None,
        "left_tool_tcp_pose_flange_m_rad": (0.0, 0.0, TOOL_TCP_Z_M, 0.0, 0.0, 0.0),
        "right_tool_tcp_pose_flange_m_rad": (0.0, 0.0, TOOL_TCP_Z_M, 0.0, 0.0, 0.0),
        "left_initial_qpos": current_q["left"],
        "right_initial_qpos": current_q["right"],
        "ik_tuning": tuning,
        "output_command_type": "joint_vector",
        "output_qpos_smoothing_alpha": 0.17 if post_smoothing else 1.0,
        "output_qpos_smoothing_adaptive_enabled": bool(post_smoothing),
        "output_qpos_smoothing_fast_alpha": 0.5,
        "current_target_hold_enabled": True,
        "local_step_candidate_selection_enabled": bool(candidate_selection),
        "inactive_side_hold_enabled": True,
        "inactive_side_hold_execution_cap_enabled": True,
        "inactive_side_hold_execution_release_enabled": True,
        "target_pose_smoother": None,
    }
    if collision_barrier_gain is not None:
        config_kwargs["collision_barrier_gain"] = float(collision_barrier_gain)
    if collision_min_distance_m is not None:
        config_kwargs["collision_min_distance_m"] = float(collision_min_distance_m)
    return DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            **config_kwargs,
        ),
        left_pose_retargeter=_RecordedBasePoseSource(side="left", frame=first_frame),
        right_pose_retargeter=_RecordedBasePoseSource(side="right", frame=first_frame),
    )


def _build_smoothed_targets(action_frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    smoother = build_bimanual_pose_smoother(load_default_pose_smoothing_config())
    tool_inverse = np.linalg.inv(_tool_matrix())
    output: list[dict[str, Any]] = []
    for frame in action_frames:
        raw: dict[str, np.ndarray] = {}
        for side in SIDES:
            metadata = _mapping(_mapping(frame.get("command_by_side")).get(side)).get("metadata")
            desk = _matrix(_mapping(metadata).get("source_target_pose_desk"))
            if desk is None:
                raise ValueError(f"Missing source_target_pose_desk at step {frame.get('step_index')}")
            raw[side] = desk @ tool_inverse
        smoothed, smoothing_metadata = smoother.smooth_pair(
            left_target=raw["left"],
            right_target=raw["right"],
            left_active=True,
            right_active=True,
            timestamp=float(frame.get("sample_timestamp")),
        )
        output.append(
            {
                "timestamp": float(
                    frame.get("teleop_command_timestamp")
                    or frame.get("sample_timestamp")
                ),
                "targets": {side: as_pose_matrix(smoothed[side]).copy() for side in SIDES},
                "smoothing_metadata": dict(smoothing_metadata),
            }
        )
    return output


def _layer_two(
    action_frames: Sequence[Mapping[str, Any]],
    references: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    tuning = load_ik_tuning_config()
    position_weights = tuning.objective_modes.get("position_priority")
    # Layer 1 already measures the post-IK smoother directly from raw q_ik/qref.
    # Layer 2 therefore uses teacher-forced recorded qref as the per-frame IK
    # seed and keeps output smoothing disabled.  This isolates the dual-IK
    # objective/candidate/step behavior instead of accumulating replay drift.
    variants: list[tuple[str, Any, bool, bool, bool]] = [
        ("baseline_ik_only", tuning, False, True, True),
        ("no_execution_limit_ik_only", tuning, False, False, False),
        ("candidate_selection_disabled_ik_only", tuning, False, True, False),
    ]
    if position_weights is not None:
        variants.append(
            (
                "position_priority_weights_ik_only",
                replace(tuning, ik_objective_mode="position_priority"),
                False,
                True,
                True,
            )
        )
    replay_targets = _build_smoothed_targets(action_frames)
    output: dict[str, Any] = {
        "frames": len(action_frames),
        "tool_tcp_pose_flange_m_rad": [0.0, 0.0, TOOL_TCP_Z_M, 0.0, 0.0, 0.0],
        "variants": {},
        "target_smoothing_replay": {},
    }
    recorded_smoothing_alpha: list[float] = []
    replay_smoothing_alpha: list[float] = []
    for frame, target in zip(action_frames, replay_targets):
        for side in SIDES:
            metadata = _mapping(_mapping(frame.get("command_by_side")).get(side)).get("metadata")
            alpha = _mapping(_mapping(metadata).get("target_pose_smoothing")).get("alpha")
            if alpha is not None:
                recorded_smoothing_alpha.append(float(alpha))
            alpha = _mapping(_mapping(target.get("smoothing_metadata")).get(side)).get("alpha")
            if alpha is not None:
                replay_smoothing_alpha.append(float(alpha))
    output["target_smoothing_replay"] = {
        "recorded_alpha": _stats(recorded_smoothing_alpha),
        "replayed_alpha": _stats(replay_smoothing_alpha),
    }

    for name, variant_tuning, post_smoothing, execution_limit, candidate_selection in variants:
        retargeter = _make_replay(
            first_frame=action_frames[0],
            tuning=variant_tuning,
            post_smoothing=post_smoothing,
            execution_limit=execution_limit,
            candidate_selection=candidate_selection,
        )
        target_to_raw: dict[str, list[tuple[float, float]]] = {side: [] for side in SIDES}
        target_to_output: dict[str, list[tuple[float, float]]] = {side: [] for side in SIDES}
        raw_qref_error: dict[str, list[float]] = {side: [] for side in SIDES}
        output_qref_error: dict[str, list[float]] = {side: [] for side in SIDES}
        raw_to_output: dict[str, list[float]] = {side: [] for side in SIDES}
        branch_counts: dict[str, Counter[str]] = {side: Counter() for side in SIDES}
        failures: list[str] = []
        for frame, target in zip(action_frames, replay_targets):
            # Seed the solver from the recorded qref for this frame.  This is
            # deliberate teacher forcing: it makes every variant answer the
            # same one-step question and prevents a small branch mismatch in
            # frame N from contaminating all later frames.
            seed_q = {}
            for side in SIDES:
                reference = references.get((int(frame.get("step_index", -1)), side), {})
                seed_q[side] = _qpos(reference.get("q_ref"))
            if seed_q["left"] is not None and seed_q["right"] is not None:
                retargeter.set_current_qpos_pair(
                    left_qpos=seed_q["left"],
                    right_qpos=seed_q["right"],
                    reset_tracking=False,
                )
            # The custom source retargeters capture frame 0; swap them for this
            # frame before constructing the context.  This keeps the existing
            # DualArm retargeter untouched.
            retargeter.left_pose_retargeter = _RecordedBasePoseSource(side="left", frame=frame)
            retargeter.right_pose_retargeter = _RecordedBasePoseSource(side="right", frame=frame)
            left_command = retargeter.left_pose_retargeter.retarget(
                TeleopObservation(timestamp=target["timestamp"])
            ).commands["left_arm"]
            right_command = retargeter.right_pose_retargeter.retarget(
                TeleopObservation(timestamp=target["timestamp"])
            ).commands["right_arm"]
            context = DualArmTargetContext(
                timestamp=target["timestamp"],
                left_target=target["targets"]["left"].copy(),
                right_target=target["targets"]["right"].copy(),
                left_world_to_base=None,
                right_world_to_base=None,
                left_cartesian=left_command,
                right_cartesian=right_command,
                left_input_skipped=None,
                right_input_skipped=None,
                target_smoothing_metadata=dict(target["smoothing_metadata"]),
            )
            try:
                command = retargeter.build_command_for_context(context)
            except Exception as exc:  # pragma: no cover - diagnostic report path
                failures.append(f"step={frame.get('step_index')}:{type(exc).__name__}:{exc}")
                continue
            for side in SIDES:
                effector = _mapping(command.commands).get(f"{side}_arm")
                if effector is None:
                    continue
                metadata = _mapping(effector.metadata)
                branch_counts[side][str(metadata.get("ik_branch_candidate", "unknown"))] += 1
                target_pose = target["targets"][side] @ _tool_matrix()
                raw_q = _qpos(metadata.get("q_ik_raw"))
                output_q = np.asarray(effector.value, dtype=float).reshape(6)
                if raw_q is None:
                    raw_q = output_q.copy()
                raw_endpoint = retargeter.ik_solver.compute_frame_pose(
                    retargeter.profile.left_frame_name if side == "left" else retargeter.profile.right_frame_name,
                    retargeter._embed_arm_qpos(
                        current_full_q=retargeter.ik_solver.current_q,
                        left_qpos=raw_q if side == "left" else retargeter._split_solution(retargeter.ik_solver.current_q)[0],
                        right_qpos=raw_q if side == "right" else retargeter._split_solution(retargeter.ik_solver.current_q)[1],
                    ),
                )
                raw_pose = raw_endpoint @ _tool_matrix()
                output_pose = _matrix(metadata.get("target_pose_base"))
                if output_pose is None:
                    output_endpoint = retargeter.ik_solver.compute_frame_pose(
                        retargeter.profile.left_frame_name if side == "left" else retargeter.profile.right_frame_name,
                        retargeter.ik_solver.current_q,
                    )
                    output_pose = output_endpoint @ _tool_matrix()
                raw_error = _pose_error(target_pose, raw_pose)
                output_error = _pose_error(target_pose, output_pose)
                if raw_error is not None:
                    target_to_raw[side].append(raw_error)
                if output_error is not None:
                    target_to_output[side].append(output_error)
                recorded = _mapping(_mapping(frame.get("command_by_side")).get(side))
                recorded_q = _qpos(recorded.get("value_qpos"))
                recorded_reference = references.get((int(frame.get("step_index", -1)), side), {})
                recorded_raw = _qpos(recorded_reference.get("q_ik_raw"))
                if recorded_q is not None:
                    output_qref_error[side].append(float(np.rad2deg(np.max(np.abs(output_q - recorded_q)))))
                if recorded_raw is not None:
                    raw_qref_error[side].append(float(np.rad2deg(np.max(np.abs(raw_q - recorded_raw)))))
                raw_to_output[side].append(float(np.rad2deg(np.max(np.abs(raw_q - output_q)))))

        output["variants"][name] = {
            "frames_replayed": len(action_frames) - len(failures),
            "failures": failures[:10],
            "failure_count": len(failures),
            "target_to_raw_ik_fk": {side: _pose_metric_stats(target_to_raw[side]) for side in SIDES},
            "target_to_output_qref_fk": {side: _pose_metric_stats(target_to_output[side]) for side in SIDES},
            "replay_raw_q_vs_recorded_raw_q_max_deg": {side: _stats(raw_qref_error[side]) for side in SIDES},
            "replay_qref_vs_recorded_qref_max_deg": {side: _stats(output_qref_error[side]) for side in SIDES},
            "raw_to_output_qpos_max_deg": {side: _stats(raw_to_output[side]) for side in SIDES},
            "branch_counts": {side: dict(branch_counts[side]) for side in SIDES},
        }
    return output


def _layer_three_dynamics(
    action_frames: Sequence[Mapping[str, Any]],
    references: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Split recorded FK tracking error by unilateral motion and target speed."""
    entries: dict[str, list[dict[str, Any]]] = {side: [] for side in SIDES}
    previous: dict[str, tuple[float, np.ndarray]] = {}
    state_pairs: Counter[str] = Counter()
    state_pair_items: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {side: [] for side in SIDES}
    )
    for frame in action_frames:
        step = int(frame.get("step_index", -1))
        frame_values: dict[str, dict[str, Any]] = {}
        for side in SIDES:
            reference = _mapping(references.get((step, side)))
            metadata = _mapping(reference.get("frame_metadata"))
            source = _matrix(metadata.get("source_target_pose_urdf_root"))
            qref = _matrix(metadata.get("q_ref_pose_urdf_root"))
            activity = _mapping(metadata.get("target_activity"))
            if source is None or qref is None:
                continue
            delta = _pose_error(source, qref)
            if delta is None:
                continue
            timestamp = float(
                reference.get("published_wall_time")
                or reference.get("generated_wall_time")
                or frame.get("teleop_command_timestamp")
                or frame.get("sample_timestamp")
            )
            linear_speed = 0.0
            angular_speed = 0.0
            if side in previous:
                prev_t, prev_pose = previous[side]
                dt = max(1e-6, timestamp - prev_t)
                linear_speed = float(np.linalg.norm(source[:3, 3] - prev_pose[:3, 3]) / dt)
                angular_speed = float(
                    (R.from_matrix(prev_pose[:3, :3].T @ source[:3, :3]).magnitude()) / dt
                )
            previous[side] = (timestamp, source)
            item = {
                "translation_error_m": float(delta[0]),
                "rotation_error_deg": float(delta[1]),
                "linear_speed_m_s": linear_speed,
                "angular_speed_rad_s": angular_speed,
                "state": str(activity.get("state") or "UNKNOWN"),
                "joint_margin_deg": float(
                    _mapping(metadata.get("joint_limit_margin")).get(
                        "nearest_margin_deg", math.nan
                    )
                    or math.nan
                ),
                "collision_distance_m": metadata.get("collision_distance_m"),
            }
            entries[side].append(item)
            frame_values[side] = item
        if len(frame_values) == 2:
            pair = f"{frame_values['left']['state']}+{frame_values['right']['state']}"
            state_pairs[pair] += 1
            for side in SIDES:
                state_pair_items[pair][side].append(frame_values[side])

    def grouped(items: Sequence[Mapping[str, Any]], key: str, bins: Sequence[float]) -> dict[str, Any]:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in items:
            value = float(item.get(key, 0.0))
            lower = 0.0
            label = f"{bins[-1]}+"
            for upper in bins:
                if value < upper:
                    label = f"{lower}-{upper}"
                    break
                lower = upper
            groups[label].append(item)
        return {
            label: {
                "count": len(values),
                "translation_error_m": _stats([float(v["translation_error_m"]) for v in values]),
                "rotation_error_deg": _stats([float(v["rotation_error_deg"]) for v in values]),
            }
            for label, values in groups.items()
        }

    by_state: dict[str, Any] = {}
    for side in SIDES:
        state_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in entries[side]:
            state_groups[str(item["state"])].append(item)
        by_state[side] = {
            state: {
                "count": len(values),
                "translation_error_m": _stats([float(v["translation_error_m"]) for v in values]),
                "rotation_error_deg": _stats([float(v["rotation_error_deg"]) for v in values]),
                "linear_speed_m_s": _stats([float(v["linear_speed_m_s"]) for v in values]),
                "angular_speed_rad_s": _stats([float(v["angular_speed_rad_s"]) for v in values]),
            }
            for state, values in state_groups.items()
        }

    state_pair_errors: dict[str, Any] = {}
    for pair in state_pairs:
        left_state, right_state = pair.split("+", 1)
        state_pair_errors[pair] = {"count": int(state_pairs[pair]), "left": {}, "right": {}}
        for side, state in (("left", left_state), ("right", right_state)):
            values = state_pair_items[pair][side]
            state_pair_errors[pair][side] = {
                "translation_error_m": _stats([float(v["translation_error_m"]) for v in values]),
                "rotation_error_deg": _stats([float(v["rotation_error_deg"]) for v in values]),
            }

    return {
        "by_state": by_state,
        "state_pair_counts": dict(state_pairs),
        "state_pair_errors": state_pair_errors,
        "by_linear_speed_m_s": {
            side: grouped(entries[side], "linear_speed_m_s", (0.02, 0.05, 0.10, 0.20, 0.40))
            for side in SIDES
        },
        "by_angular_speed_rad_s": {
            side: grouped(entries[side], "angular_speed_rad_s", (0.10, 0.30, 0.60, 1.20, 2.00))
            for side in SIDES
        },
    }


def _layer_three_unilateral_replay(
    action_frames: Sequence[Mapping[str, Any]],
    references: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Teacher-forced replay with the opposite-arm target held at qref FK."""
    replay_targets = _build_smoothed_targets(action_frames)
    output: dict[str, Any] = {"modes": {}}
    for mode in ("both", "left_only", "right_only"):
        retargeter = _make_replay(
            first_frame=action_frames[0],
            tuning=load_ik_tuning_config(),
            post_smoothing=False,
            execution_limit=True,
            candidate_selection=True,
        )
        errors: dict[str, list[tuple[float, float]]] = {side: [] for side in SIDES}
        raw_q_vs_recorded: dict[str, list[float]] = {side: [] for side in SIDES}
        branch_counts: dict[str, Counter[str]] = {side: Counter() for side in SIDES}
        failures = 0
        for frame, target in zip(action_frames, replay_targets):
            step = int(frame.get("step_index", -1))
            seed = {side: _qpos(references.get((step, side), {}).get("q_ref")) for side in SIDES}
            if seed["left"] is None or seed["right"] is None:
                failures += 1
                continue
            retargeter.set_current_qpos_pair(
                left_qpos=seed["left"], right_qpos=seed["right"], reset_tracking=False
            )
            retargeter.left_pose_retargeter = _RecordedBasePoseSource(side="left", frame=frame)
            retargeter.right_pose_retargeter = _RecordedBasePoseSource(side="right", frame=frame)
            left_command = retargeter.left_pose_retargeter.retarget(
                TeleopObservation(timestamp=target["timestamp"])
            ).commands["left_arm"]
            right_command = retargeter.right_pose_retargeter.retarget(
                TeleopObservation(timestamp=target["timestamp"])
            ).commands["right_arm"]
            current_targets = retargeter.current_world_targets()
            targets = {side: target["targets"][side].copy() for side in SIDES}
            if mode == "left_only":
                targets["right"] = current_targets["right"].copy()
            elif mode == "right_only":
                targets["left"] = current_targets["left"].copy()
            context = DualArmTargetContext(
                timestamp=target["timestamp"],
                left_target=targets["left"],
                right_target=targets["right"],
                left_world_to_base=None,
                right_world_to_base=None,
                left_cartesian=left_command,
                right_cartesian=right_command,
                left_input_skipped=None,
                right_input_skipped=None,
                target_smoothing_metadata=dict(target["smoothing_metadata"]),
            )
            try:
                command = retargeter.build_command_for_context(context)
            except Exception:
                failures += 1
                continue
            for side in SIDES:
                if mode == "left_only" and side == "right":
                    continue
                if mode == "right_only" and side == "left":
                    continue
                effector = command.commands.get(f"{side}_arm")
                if effector is None:
                    continue
                metadata = _mapping(effector.metadata)
                branch_counts[side][str(metadata.get("ik_branch_candidate", "unknown"))] += 1
                raw_q = _qpos(metadata.get("q_ik_raw"))
                if raw_q is None:
                    continue
                raw_full = retargeter._embed_arm_qpos(
                    current_full_q=retargeter.ik_solver.current_q,
                    left_qpos=raw_q if side == "left" else seed["left"],
                    right_qpos=raw_q if side == "right" else seed["right"],
                )
                raw_endpoint = retargeter.ik_solver.compute_frame_pose(
                    retargeter.profile.left_frame_name if side == "left" else retargeter.profile.right_frame_name,
                    raw_full,
                )
                raw_pose = raw_endpoint @ _tool_matrix()
                error = _pose_error(target["targets"][side] @ _tool_matrix(), raw_pose)
                if error is not None:
                    errors[side].append(error)
                recorded_raw = _qpos(references.get((step, side), {}).get("q_ik_raw"))
                if recorded_raw is not None:
                    raw_q_vs_recorded[side].append(float(np.rad2deg(np.max(np.abs(raw_q - recorded_raw)))))
        output["modes"][mode] = {
            "failures": failures,
            "target_to_raw_ik_fk": {side: _pose_metric_stats(errors[side]) for side in SIDES},
            "raw_q_vs_recorded_raw_q_max_deg": {side: _stats(raw_q_vs_recorded[side]) for side in SIDES},
            "branch_counts": {side: dict(branch_counts[side]) for side in SIDES},
        }
    return output


def _layer_four_solver_convergence(
    action_frames: Sequence[Mapping[str, Any]],
    references: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Teacher-forced solver A/B for iteration count and collision barrier.

    Every frame is seeded from its recorded qref and replayed with execution
    limits, candidate selection, and output smoothing disabled.  This keeps
    the comparison focused on the Pink/MultiTask IK solve itself: one versus
    several solver iterations, plus a diagnostic with the collision barrier
    disabled.  No command is dispatched and this function is never used by
    the runtime control path.
    """
    replay_targets = _build_smoothed_targets(action_frames)
    tuning = load_ik_tuning_config()
    variants = (
        ("iterations_1", 1, None, None),
        ("iterations_2", 2, None, None),
        ("iterations_4", 4, None, None),
        ("collision_barrier_disabled", 1, 0.0, 0.0),
    )
    output: dict[str, Any] = {"variants": {}}
    for name, iterations, barrier_gain, min_distance in variants:
        retargeter = _make_replay(
            first_frame=action_frames[0],
            tuning=tuning,
            post_smoothing=False,
            execution_limit=False,
            candidate_selection=False,
            solve_max_iterations=iterations,
            collision_barrier_gain=barrier_gain,
            collision_min_distance_m=min_distance,
        )
        target_to_raw: dict[str, list[tuple[float, float]]] = {side: [] for side in SIDES}
        raw_q_vs_recorded: dict[str, list[float]] = {side: [] for side in SIDES}
        raw_q_vs_qref: dict[str, list[float]] = {side: [] for side in SIDES}
        failures: list[str] = []
        iteration_counts: list[int] = []
        for frame, target in zip(action_frames, replay_targets):
            step = int(frame.get("step_index", -1))
            seed = {
                side: _qpos(references.get((step, side), {}).get("q_ref"))
                for side in SIDES
            }
            if seed["left"] is None or seed["right"] is None:
                failures.append(f"step={step}:missing_qref")
                continue
            retargeter.set_current_qpos_pair(
                left_qpos=seed["left"],
                right_qpos=seed["right"],
                reset_tracking=False,
            )
            retargeter.left_pose_retargeter = _RecordedBasePoseSource(side="left", frame=frame)
            retargeter.right_pose_retargeter = _RecordedBasePoseSource(side="right", frame=frame)
            left_command = retargeter.left_pose_retargeter.retarget(
                TeleopObservation(timestamp=target["timestamp"])
            ).commands["left_arm"]
            right_command = retargeter.right_pose_retargeter.retarget(
                TeleopObservation(timestamp=target["timestamp"])
            ).commands["right_arm"]
            context = DualArmTargetContext(
                timestamp=target["timestamp"],
                left_target=target["targets"]["left"].copy(),
                right_target=target["targets"]["right"].copy(),
                left_world_to_base=None,
                right_world_to_base=None,
                left_cartesian=left_command,
                right_cartesian=right_command,
                left_input_skipped=None,
                right_input_skipped=None,
                target_smoothing_metadata=dict(target["smoothing_metadata"]),
            )
            try:
                command = retargeter.build_command_for_context(context)
            except Exception as exc:  # pragma: no cover - diagnostic report path
                failures.append(f"step={step}:{type(exc).__name__}:{exc}")
                continue
            metadata_iterations = _mapping(command.metadata).get("ik_iterations")
            if metadata_iterations is not None:
                try:
                    iteration_counts.append(int(metadata_iterations))
                except (TypeError, ValueError):
                    pass
            for side in SIDES:
                effector = _mapping(command.commands).get(f"{side}_arm")
                if effector is None:
                    continue
                metadata = _mapping(effector.metadata)
                raw_q = _qpos(metadata.get("q_ik_raw"))
                if raw_q is None:
                    failures.append(f"step={step}:{side}:missing_raw_q")
                    continue
                raw_full = retargeter._embed_arm_qpos(
                    current_full_q=retargeter.ik_solver.current_q,
                    left_qpos=raw_q if side == "left" else seed["left"],
                    right_qpos=raw_q if side == "right" else seed["right"],
                )
                endpoint = retargeter.ik_solver.compute_frame_pose(
                    retargeter.profile.left_frame_name
                    if side == "left"
                    else retargeter.profile.right_frame_name,
                    raw_full,
                )
                raw_pose = endpoint @ _tool_matrix()
                target_pose = target["targets"][side] @ _tool_matrix()
                error = _pose_error(target_pose, raw_pose)
                if error is not None:
                    target_to_raw[side].append(error)
                recorded_raw = _qpos(references.get((step, side), {}).get("q_ik_raw"))
                recorded_qref = seed[side]
                if recorded_raw is not None:
                    raw_q_vs_recorded[side].append(
                        float(np.rad2deg(np.max(np.abs(raw_q - recorded_raw))))
                    )
                raw_q_vs_qref[side].append(
                    float(np.rad2deg(np.max(np.abs(raw_q - recorded_qref))))
                )
        output["variants"][name] = {
            "solve_max_iterations": iterations,
            "collision_barrier_gain": barrier_gain,
            "collision_min_distance_m": min_distance,
            "frames_replayed": len(action_frames) - len(failures),
            "failures": failures[:10],
            "failure_count": len(failures),
            "target_to_raw_ik_fk": {
                side: _pose_metric_stats(target_to_raw[side]) for side in SIDES
            },
            "raw_q_vs_recorded_raw_q_max_deg": {
                side: _stats(raw_q_vs_recorded[side]) for side in SIDES
            },
            "raw_q_vs_recorded_qref_max_deg": {
                side: _stats(raw_q_vs_qref[side]) for side in SIDES
            },
            "reported_iteration_count": _stats(iteration_counts),
        }
    return output


def _layer_three(
    action_frames: Sequence[Mapping[str, Any]],
    references: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "dynamics": _layer_three_dynamics(action_frames, references),
        "unilateral_replay": _layer_three_unilateral_replay(action_frames, references),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    layer_one = _mapping(report.get("layer_one"))
    lines = [
        "# Position-control offline A/B report",
        "",
        f"Capture: `{report.get('capture_dir')}`",
        "",
        "## Layer 1: recorded trace attribution",
        "",
    ]
    for side in SIDES:
        lines.append(f"### {side}")
        for stage, values in _mapping(_mapping(layer_one.get("stage_errors")).get(side)).items():
            metrics = _mapping(values)
            trans = _mapping(metrics.get("translation_m"))
            rot = _mapping(metrics.get("rotation_deg"))
            lines.append(
                f"- {stage}: translation p95={trans.get('p95')}m, rotation p95={rot.get('p95')}deg, n={trans.get('n')}"
            )
        lines.append(
            f"- raw q_ik→qref max joint delta p95={_mapping(_mapping(layer_one.get('qpos_raw_to_qref_deg')).get(side)).get('p95')}deg"
        )
        candidate = _mapping(_mapping(layer_one.get("candidate_selection")).get(side))
        lines.append(f"- branches: {candidate.get('branch_counts')}")
        lines.append(
            f"- position-worsen/rotation-improve count: {candidate.get('position_worsen_and_rotation_improve_count')}"
        )
    lines.extend(["", "## Layer 2: offline dual-IK replay", ""])
    for name, values in _mapping(report.get("layer_two")).get("variants", {}).items():
        lines.append(f"### {name}")
        lines.append(f"- replayed={values.get('frames_replayed')} failures={values.get('failure_count')}")
        for side in SIDES:
            raw = _mapping(_mapping(values.get("target_to_raw_ik_fk")).get(side))
            out = _mapping(_mapping(values.get("target_to_output_qref_fk")).get(side))
            raw_t = _mapping(_mapping(raw.get("translation_m")))
            out_t = _mapping(_mapping(out.get("translation_m")))
            lines.append(
                f"- {side}: target→raw p95={raw_t.get('p95')}m, target→qref p95={out_t.get('p95')}m"
            )
    layer_three = _mapping(report.get("layer_three"))
    lines.extend(["", "## Layer 3: unilateral and speed-conditioned analysis", ""])
    dynamics = _mapping(layer_three.get("dynamics"))
    lines.append(f"- state-pair counts: {dynamics.get('state_pair_counts')}")
    for side, groups in _mapping(dynamics.get("by_linear_speed_m_s")).items():
        lines.append(f"### {side} target linear speed")
        for label, values in _mapping(groups).items():
            trans = _mapping(_mapping(values).get("translation_error_m"))
            rot = _mapping(_mapping(values).get("rotation_error_deg"))
            lines.append(
                f"- {label}m/s: n={values.get('count')}, translation p95={trans.get('p95')}m, rotation p95={rot.get('p95')}deg"
            )
    for mode, values in _mapping(_mapping(layer_three.get("unilateral_replay")).get("modes")).items():
        lines.append(f"### unilateral replay: {mode}")
        lines.append(f"- failures={values.get('failures')}")
        for side in SIDES:
            metrics = _mapping(_mapping(values.get("target_to_raw_ik_fk")).get(side))
            trans = _mapping(metrics.get("translation_m"))
            rot = _mapping(metrics.get("rotation_deg"))
            if trans.get("n"):
                lines.append(
                    f"- {side}: target→raw p95={trans.get('p95')}m, rotation p95={rot.get('p95')}deg"
                )
    lines.extend(["", "## Layer 4: solver convergence and collision-barrier A/B", ""])
    for name, values in _mapping(report.get("layer_four")).get("variants", {}).items():
        lines.append(f"### {name}")
        min_distance = values.get("collision_min_distance_m")
        min_distance_text = "default" if min_distance is None else f"{min_distance}m"
        lines.append(
            f"- iterations={values.get('solve_max_iterations')}, "
            f"barrier_gain={values.get('collision_barrier_gain')}, "
            f"min_distance={min_distance_text}, "
            f"replayed={values.get('frames_replayed')} failures={values.get('failure_count')}"
        )
        for side in SIDES:
            metrics = _mapping(_mapping(values.get("target_to_raw_ik_fk")).get(side))
            trans = _mapping(metrics.get("translation_m"))
            rot = _mapping(metrics.get("rotation_deg"))
            lines.append(
                f"- {side}: target→raw p95={trans.get('p95')}m, rotation p95={rot.get('p95')}deg"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    action_frames, references, commands = _load_capture(capture_dir)
    report = {
        "capture_dir": str(capture_dir),
        "layer_one": _layer_one(capture_dir, action_frames, references, commands),
        "layer_two": _layer_two(action_frames, references),
        "layer_three": _layer_three(action_frames, references),
        "layer_four": _layer_four_solver_convergence(action_frames, references),
    }
    prefix = Path(args.output_prefix or (capture_dir / "position_ab_report"))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    prefix.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))
    print(f"JSON report: {prefix.with_suffix('.json')}")
    print(f"Markdown report: {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
