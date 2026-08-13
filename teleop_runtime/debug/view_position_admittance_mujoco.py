"""Visualize saved 60 Hz force-control replay results in MuJoCo.

The expensive control result is computed first by the validated offline
baseline/Active evaluator.  The viewer then performs read-only playback of the
already simulated measured qpos; it does not solve IK, advance force-control
state, run physics, or construct hardware while the window is open.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from robot_motion.rm65.position_profile import (
    RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
    RM65_POSITION_CONTROL_PROFILE_CHOICES,
)
from sim.contracts import SimulationContract
from sim.mujoco.control.desk_tcp_position import (
    DEFAULT_SIMULATION_CONTRACT_PATH,
    position_15_60_120_timing,
)
from sim.mujoco.replay.tcp_target_visualization import (
    TcpTargetVisualizationConfig,
    TcpTargetVisualizer,
)
from sim.mujoco.scene import MujocoScene
from sim.mujoco.viewer import passive_viewer_handle

from .replay_position_admittance_ik_trace import (
    DEFAULT_LEFT_TOOL_PROFILE,
    DEFAULT_REPLAY_HZ,
    DEFAULT_RIGHT_TOOL_PROFILE,
    SIDES,
)
from .replay_position_admittance_mujoco import (
    ADMITTANCE_ENVELOPES,
    OFFLINE_NORMAL_PARITY_ENVELOPE,
    OFFLINE_NORMAL_PARITY_MAX_TOOL_EXCURSION_M,
    _PhysicalRollout,
    _replay_recorded_wrench_through_mujoco_evaluation,
)
from .replay_position_rotational_admittance_mujoco import (
    ROTATIONAL_ADMITTANCE_ENVELOPES,
    rotational_admittance_config_for_envelope,
)


_BASELINE_RGBA = np.array([0.10, 0.90, 1.00, 0.95], dtype=np.float32)
_ACTIVE_RGBA = np.array([1.00, 0.35, 0.05, 0.95], dtype=np.float32)
_APPLIED_TARGET_RGBA = np.array([1.00, 0.90, 0.05, 0.95], dtype=np.float32)
_FORCE_RGBA = np.array([0.20, 1.00, 0.20, 0.95], dtype=np.float32)
_MODIFIER_RGBA = np.array([1.00, 0.75, 0.05, 0.95], dtype=np.float32)
_PHYSICAL_DELTA_RGBA = np.array([0.95, 0.15, 1.00, 0.95], dtype=np.float32)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


@dataclass(frozen=True)
class ForceControlVisualizationConfig:
    endpoint_radius_m: float = 0.008
    applied_target_radius_m: float = 0.006
    arrow_width_m: float = 0.004
    force_arrow_scale_m_per_n: float = 0.03
    force_arrow_max_length_m: float = 0.15
    modifier_arrow_scale: float = 30.0
    physical_delta_arrow_scale: float = 30.0

    def __post_init__(self) -> None:
        for name in (
            "endpoint_radius_m",
            "applied_target_radius_m",
            "arrow_width_m",
            "force_arrow_scale_m_per_n",
            "force_arrow_max_length_m",
            "modifier_arrow_scale",
            "physical_delta_arrow_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)


class ForceControlResultVisualizer:
    """Draw non-authoritative force-control comparison decorations."""

    _GEOM_COUNT = 6

    def __init__(
        self,
        config: ForceControlVisualizationConfig | None = None,
    ) -> None:
        self.config = config or ForceControlVisualizationConfig()
        self._base_geom_index: int | None = None

    @property
    def geom_count(self) -> int:
        return self._GEOM_COUNT

    def update(
        self,
        viewer_handle: Any,
        *,
        baseline_endpoint_pose: np.ndarray,
        active_endpoint_pose: np.ndarray,
        raw_target_pose: np.ndarray,
        applied_target_pose: np.ndarray,
        filtered_force_control_n: Sequence[float],
        active_offset_m: Sequence[float],
        selected_side: str,
    ) -> bool:
        if viewer_handle is None:
            return False
        side = str(selected_side).strip().lower()
        if side not in SIDES:
            raise ValueError("selected_side must be left or right")
        poses = {
            "baseline": self._pose(baseline_endpoint_pose, name="baseline_endpoint_pose"),
            "active": self._pose(active_endpoint_pose, name="active_endpoint_pose"),
            "raw": self._pose(raw_target_pose, name="raw_target_pose"),
            "applied": self._pose(applied_target_pose, name="applied_target_pose"),
        }
        force = self._vector3(
            filtered_force_control_n,
            name="filtered_force_control_n",
        )
        offset = self._vector3(active_offset_m, name="active_offset_m")
        lock = getattr(viewer_handle, "lock", None)
        if not callable(lock):
            raise TypeError("viewer_handle must expose lock()")
        with lock():
            user_scene = getattr(viewer_handle, "user_scn", None)
            if user_scene is None:
                raise TypeError("viewer_handle must expose user_scn")
            if self._base_geom_index is None:
                self._base_geom_index = int(user_scene.ngeom)
            stop = self._base_geom_index + self.geom_count
            if stop > int(user_scene.maxgeom):
                raise RuntimeError(
                    "MuJoCo viewer user_scn has insufficient geometry capacity: "
                    f"need={stop} max={user_scene.maxgeom}"
                )
            geoms = user_scene.geoms
            base = self._base_geom_index
            self._write_sphere(
                geoms[base],
                position=poses["baseline"][:3, 3],
                radius_m=self.config.endpoint_radius_m,
                rgba=_BASELINE_RGBA,
                label=f"{side}_baseline_measured_endpoint",
            )
            self._write_sphere(
                geoms[base + 1],
                position=poses["active"][:3, 3],
                radius_m=self.config.endpoint_radius_m,
                rgba=_ACTIVE_RGBA,
                label=f"{side}_active_measured_endpoint",
            )
            self._write_sphere(
                geoms[base + 2],
                position=poses["applied"][:3, 3],
                radius_m=self.config.applied_target_radius_m,
                rgba=_APPLIED_TARGET_RGBA,
                label=f"{side}_active_applied_target",
            )
            raw_origin = poses["raw"][:3, 3]
            force_length = min(
                float(np.linalg.norm(force))
                * self.config.force_arrow_scale_m_per_n,
                self.config.force_arrow_max_length_m,
            )
            force_endpoint = (
                raw_origin
                if force_length <= 1e-12
                else raw_origin + force / float(np.linalg.norm(force)) * force_length
            )
            self._write_arrow(
                geoms[base + 3],
                start=raw_origin,
                end=force_endpoint,
                rgba=_FORCE_RGBA,
                label=f"{side}_filtered_force_solver_world",
                visible=force_length > 1e-12,
            )
            self._write_arrow(
                geoms[base + 4],
                start=raw_origin,
                end=raw_origin + offset * self.config.modifier_arrow_scale,
                rgba=_MODIFIER_RGBA,
                label=(
                    f"{side}_modifier_x{self.config.modifier_arrow_scale:g}"
                ),
                visible=float(np.linalg.norm(offset)) > 1e-12,
            )
            physical_delta = (
                poses["active"][:3, 3] - poses["baseline"][:3, 3]
            )
            self._write_arrow(
                geoms[base + 5],
                start=poses["baseline"][:3, 3],
                end=(
                    poses["baseline"][:3, 3]
                    + physical_delta * self.config.physical_delta_arrow_scale
                ),
                rgba=_PHYSICAL_DELTA_RGBA,
                label=(
                    f"{side}_physical_delta_x"
                    f"{self.config.physical_delta_arrow_scale:g}"
                ),
                visible=float(np.linalg.norm(physical_delta)) > 1e-12,
            )
            user_scene.ngeom = max(int(user_scene.ngeom), stop)
        return True

    @staticmethod
    def _pose(value: Any, *, name: str) -> np.ndarray:
        pose = np.asarray(value, dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError(f"{name} must be a finite 4x4 pose")
        return pose.copy()

    @staticmethod
    def _vector3(value: Any, *, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float).reshape(-1)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain three finite values")
        return vector.copy()

    @staticmethod
    def _write_sphere(
        geom: Any,
        *,
        position: np.ndarray,
        radius_m: float,
        rgba: np.ndarray,
        label: str,
    ) -> None:
        import mujoco

        radius = float(radius_m)
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius], dtype=float),
            np.asarray(position, dtype=float),
            np.eye(3, dtype=float).reshape(-1),
            rgba,
        )
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.label = str(label)

    def _write_arrow(
        self,
        geom: Any,
        *,
        start: np.ndarray,
        end: np.ndarray,
        rgba: np.ndarray,
        label: str,
        visible: bool,
    ) -> None:
        import mujoco

        resolved_rgba = rgba.copy()
        resolved_end = np.asarray(end, dtype=float)
        if not visible:
            resolved_rgba[3] = 0.0
            resolved_end = np.asarray(start, dtype=float) + np.array(
                [0.0, 0.0, 1e-9], dtype=float
            )
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            np.eye(3, dtype=float).reshape(-1),
            resolved_rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            self.config.arrow_width_m,
            np.asarray(start, dtype=float),
            resolved_end,
        )
        geom.rgba = resolved_rgba
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.label = str(label)


def _frame_status(
    *,
    index: int,
    baseline: _PhysicalRollout,
    active: _PhysicalRollout,
    selected_side: str,
) -> dict[str, Any]:
    side = str(selected_side)
    baseline_pose = baseline.measured_pose[side][index]
    active_pose = active.measured_pose[side][index]
    torque = (
        active.filtered_torque_control_nm[index]
        if active.filtered_torque_control_nm
        else np.zeros(3, dtype=float)
    )
    rotation_offset = (
        active.active_rotation_rotvec_rad[index]
        if active.active_rotation_rotvec_rad
        else np.zeros(3, dtype=float)
    )
    return {
        "time_sec": float(
            active.sample_times_sec[index] - active.sample_times_sec[0]
        ),
        "filtered_force_control_n": active.filtered_force_control_n[index]
        .astype(float)
        .tolist(),
        "force_norm_n": float(
            np.linalg.norm(active.filtered_force_control_n[index])
        ),
        "active_offset_m": active.active_offset_m[index].astype(float).tolist(),
        "active_offset_norm_m": float(
            np.linalg.norm(active.active_offset_m[index])
        ),
        "filtered_torque_control_nm": np.asarray(torque, dtype=float).tolist(),
        "torque_norm_nm": float(np.linalg.norm(torque)),
        "active_rotation_rotvec_deg": np.rad2deg(rotation_offset).tolist(),
        "active_rotation_norm_deg": float(
            np.degrees(np.linalg.norm(rotation_offset))
        ),
        "qref_delta_max_deg": float(
            np.rad2deg(
                np.max(
                    np.abs(
                        active.qref[side][index] - baseline.qref[side][index]
                    )
                )
            )
        ),
        "qcmd_delta_max_deg": float(
            np.rad2deg(
                np.max(
                    np.abs(
                        active.qcmd[side][index] - baseline.qcmd[side][index]
                    )
                )
            )
        ),
        "measured_qpos_delta_max_deg": float(
            np.rad2deg(
                np.max(
                    np.abs(
                        active.measured_qpos[side][index]
                        - baseline.measured_qpos[side][index]
                    )
                )
            )
        ),
        "measured_endpoint_delta_m": float(
            np.linalg.norm(active_pose[:3, 3] - baseline_pose[:3, 3])
        ),
    }


def _build_visualization_scene(path: Path) -> MujocoScene:
    base_contract = SimulationContract.load(path)
    timing = position_15_60_120_timing(
        base_contract.timing,
        parameter_source=RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
    )
    return MujocoScene.from_contract(replace(base_contract, timing=timing))


def _sleep_interruptibly(
    duration_sec: float,
    *,
    viewer_handle: Any,
) -> bool:
    deadline = time.monotonic() + max(0.0, float(duration_sec))
    while time.monotonic() < deadline:
        if not viewer_handle.is_running():
            return False
        viewer_handle.sync()
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
    return bool(viewer_handle.is_running())


def _playback_indices(
    rollout: _PhysicalRollout,
    *,
    start_sec: float,
    end_sec: float,
) -> list[int]:
    origin = float(rollout.sample_times_sec[0])
    return [
        index
        for index, sample_time in enumerate(rollout.sample_times_sec)
        if start_sec - 1e-12 <= float(sample_time) - origin <= end_sec + 1e-12
    ]


def _compact_report(report: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    comparison = report["comparisons"][mode]
    active = report["rollouts"][f"{mode}_active"]
    selected_translation = comparison.get("selected_measured_translation_m") or {}
    selected_rotation = comparison.get("selected_measured_rotation_deg") or {}
    opposite_qpos = comparison.get("opposite_measured_qpos_delta_rad") or {}
    return {
        "path": report["path"],
        "mode": mode,
        "admittance_envelope": report["admittance_envelope"],
        "hardware_authorization": report["hardware_authorization"],
        "safety_passed": report["safety_passed"],
        "coverage_passed": report["coverage_passed"],
        "reference_ticks": active["completed_reference_ticks"],
        "physics_ticks": active["physics_ticks"],
        "active_offset_max_m": active["active_offset_norm_m"]["max"],
        "active_offset_final_m": active["final_active_offset_m"],
        "active_rotation_offset_max_deg": (
            None
            if active.get("active_rotation_offset_norm_rad", {}).get("max")
            is None
            else float(
                np.degrees(
                    active["active_rotation_offset_norm_rad"]["max"]
                )
            )
        ),
        "active_rotation_offset_final_deg": (
            None
            if active.get("final_active_rotation_offset_rad") is None
            else float(
                np.degrees(active["final_active_rotation_offset_rad"])
            )
        ),
        "active_qref_motion_ticks": active["active_qref_motion_ticks"],
        "comparison_error": comparison.get("error"),
        "measured_endpoint_delta_max_m": selected_translation.get("max"),
        "measured_endpoint_rotation_max_deg": selected_rotation.get("max"),
        "opposite_measured_qpos_delta_max_rad": opposite_qpos.get("max"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--side", choices=SIDES, default="left")
    parser.add_argument(
        "--admittance-envelope",
        choices=ADMITTANCE_ENVELOPES,
        default="cautious-active",
        help=(
            "cautious-active keeps the production 2 mm / 5 mm/s envelope; "
            "recorded-shadow uses the exact saved shadow config; "
            "normal-parity-offline reconstructs 8 cm / 60 mm/s in sim only."
        ),
    )
    parser.add_argument("--interpolation", choices=("zoh", "linear"), default="zoh")
    parser.add_argument(
        "--rotation-envelope",
        choices=("disabled", *ROTATIONAL_ADMITTANCE_ENVELOPES),
        default="disabled",
        help=(
            "Optional offline-only rotational profile. Translation is disabled "
            "unless --admittance-envelope normal-parity-offline is selected; "
            "production Active authorization always remains false."
        ),
    )
    parser.add_argument("--replay-hz", type=_positive_float, default=DEFAULT_REPLAY_HZ)
    parser.add_argument("--release-tail-sec", type=_positive_float, default=10.0)
    parser.add_argument(
        "--translation-output-follow-tau-sec",
        type=_nonnegative_float,
        default=None,
        help="Optional 60 Hz translation modifier output follower; 0 is rollback.",
    )
    parser.add_argument(
        "--rotation-output-follow-tau-sec",
        type=_nonnegative_float,
        default=None,
        help="Optional 60 Hz rotation modifier output follower; 0 is rollback.",
    )
    parser.add_argument(
        "--translation-max-output-acceleration-m-s2",
        type=_nonnegative_float,
        default=None,
        help="Optional translation modifier acceleration limit; 0 is rollback.",
    )
    parser.add_argument(
        "--rotation-max-output-angular-acceleration-rad-s2",
        type=_nonnegative_float,
        default=None,
        help="Optional rotation modifier angular-acceleration limit; 0 is rollback.",
    )
    parser.add_argument(
        "--visible-release-tail-sec",
        type=_nonnegative_float,
        default=2.0,
        help="Release tail shown after the recorded interval; full release-tail-sec is still simulated.",
    )
    parser.add_argument("--start-sec", type=_nonnegative_float, default=0.0)
    parser.add_argument("--speed", type=_positive_float, default=1.0)
    parser.add_argument("--repeat", type=_positive_int, default=1)
    parser.add_argument("--branch", choices=("active", "baseline", "both"), default="active")
    parser.add_argument("--status-hz", type=_positive_float, default=5.0)
    parser.add_argument("--force-arrow-scale-m-per-n", type=_positive_float, default=0.03)
    parser.add_argument("--modifier-arrow-scale", type=_positive_float, default=30.0)
    parser.add_argument("--physical-delta-arrow-scale", type=_positive_float, default=30.0)
    parser.add_argument("--no-target-visualization", action="store_true")
    parser.add_argument("--hide-left-ui", action="store_true")
    parser.add_argument("--hide-right-ui", action="store_true")
    parser.add_argument("--hold-viewer-at-end", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run and print the offline evaluation without opening a GUI.",
    )
    parser.add_argument(
        "--left-tool-profile",
        type=Path,
        default=Path(DEFAULT_LEFT_TOOL_PROFILE),
    )
    parser.add_argument(
        "--right-tool-profile",
        type=Path,
        default=Path(DEFAULT_RIGHT_TOOL_PROFILE),
    )
    parser.add_argument(
        "--simulation-contract",
        type=Path,
        default=DEFAULT_SIMULATION_CONTRACT_PATH,
    )
    parser.add_argument(
        "--position-execution-profile",
        choices=tuple(
            value
            for value in RM65_POSITION_CONTROL_PROFILE_CHOICES
            if value != "legacy"
        ),
        default=RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    mode = str(args.interpolation)
    rotation_envelope = str(args.rotation_envelope)
    rotational_config = (
        None
        if rotation_envelope == "disabled"
        else rotational_admittance_config_for_envelope(rotation_envelope)
    )
    max_rotation_excursion_deg = (
        None
        if rotational_config is None
        else float(np.degrees(max(rotational_config.max_offset_rad))) + 0.5
    )
    print(
        "[force-mujoco-viz] preparing deterministic baseline/Active physical replay; "
        "no hardware or device resource is constructed",
        flush=True,
    )
    report, rollouts = _replay_recorded_wrench_through_mujoco_evaluation(
        Path(args.log),
        side=str(args.side),
        admittance_envelope=str(args.admittance_envelope),
        replay_hz=float(args.replay_hz),
        release_tail_sec=float(args.release_tail_sec),
        interpolation_modes=(mode,),
        left_tool_profile=Path(args.left_tool_profile),
        right_tool_profile=Path(args.right_tool_profile),
        simulation_contract_path=Path(args.simulation_contract),
        max_tool_excursion_m=(
            OFFLINE_NORMAL_PARITY_MAX_TOOL_EXCURSION_M
            if str(args.admittance_envelope)
            == OFFLINE_NORMAL_PARITY_ENVELOPE
            else None
        ),
        rotational_config=rotational_config,
        max_tool_rotation_excursion_deg=max_rotation_excursion_deg,
        position_execution_profile_name=str(args.position_execution_profile),
        translation_output_follow_tau_sec=(
            args.translation_output_follow_tau_sec
        ),
        rotation_output_follow_tau_sec=args.rotation_output_follow_tau_sec,
        translation_max_output_acceleration_m_s2=(
            args.translation_max_output_acceleration_m_s2
        ),
        rotation_max_output_angular_acceleration_rad_s2=(
            args.rotation_max_output_angular_acceleration_rad_s2
        ),
    )
    summary = _compact_report(report, mode=mode)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not bool(report["safety_passed"]):
        raise RuntimeError("offline MuJoCo safety gates failed; viewer playback refused")
    if not bool(report["coverage_passed"]):
        print(
            "[force-mujoco-viz] coverage=insufficient: this trace is safe to inspect "
            "but contains no admitted Active physical response",
            flush=True,
        )
    if str(report["admittance_envelope"]) in {
        "recorded-shadow",
        OFFLINE_NORMAL_PARITY_ENVELOPE,
    }:
        print(
            "[force-mujoco-viz] envelope="
            f"{report['admittance_envelope']} is an offline-only "
            "counterfactual; it does not satisfy or change production Active admission",
            flush=True,
        )
    if rotational_config is not None:
        translation_note = (
            "normal-parity translation is enabled"
            if str(report["admittance_envelope"])
            == OFFLINE_NORMAL_PARITY_ENVELOPE
            else "translation modifier is disabled"
        )
        print(
            "[force-mujoco-viz] rotation_envelope="
            f"{rotation_envelope} is an offline-only SO(3) counterfactual; "
            f"{translation_note}",
            flush=True,
        )
    if bool(args.prepare_only):
        return dict(report)

    baseline = rollouts[f"{mode}_baseline"]
    active = rollouts[f"{mode}_active"]
    recorded_duration = float(report["input"]["recorded_duration_sec"])
    end_sec = recorded_duration + float(args.visible_release_tail_sec)
    indices = _playback_indices(
        active,
        start_sec=float(args.start_sec),
        end_sec=end_sec,
    )
    if not indices:
        raise ValueError(
            "selected playback interval contains no replay samples: "
            f"start={args.start_sec:.3f}s end={end_sec:.3f}s"
        )
    scene = _build_visualization_scene(Path(args.simulation_contract))
    target_visualizer = TcpTargetVisualizer(
        TcpTargetVisualizationConfig(
            enabled=not bool(args.no_target_visualization)
        )
    )
    result_visualizer = ForceControlResultVisualizer(
        ForceControlVisualizationConfig(
            force_arrow_scale_m_per_n=float(args.force_arrow_scale_m_per_n),
            modifier_arrow_scale=float(args.modifier_arrow_scale),
            physical_delta_arrow_scale=float(args.physical_delta_arrow_scale),
        )
    )
    branch_order = (
        ("active", "baseline")
        if args.branch == "both"
        else (str(args.branch),)
    )
    print(
        "[force-mujoco-viz] legend: robot=selected branch, cyan=baseline measured, "
        "orange=Active measured, yellow sphere=Active applied target, "
        "green arrow=filtered force, yellow arrow=modifier (display-scaled), "
        "magenta arrow=physical Active-baseline delta (display-scaled)",
        flush=True,
    )
    print(
        "[force-mujoco-viz] viewer is qpos-only playback of precomputed physical results; "
        "decoration scales never alter replay state",
        flush=True,
    )
    with passive_viewer_handle(
        scene,
        show_left_ui=not bool(args.hide_left_ui),
        show_right_ui=not bool(args.hide_right_ui),
    ) as viewer_handle:
        keep_running = True
        for repeat_index in range(int(args.repeat)):
            for branch_name in branch_order:
                if not keep_running or not viewer_handle.is_running():
                    keep_running = False
                    break
                branch = active if branch_name == "active" else baseline
                print(
                    f"[force-mujoco-viz] play branch={branch_name} "
                    f"repeat={repeat_index + 1}/{int(args.repeat)} "
                    f"samples={len(indices)} speed={float(args.speed):g}x",
                    flush=True,
                )
                next_status_time = -float("inf")
                previous_sample_time: float | None = None
                for index in indices:
                    if not viewer_handle.is_running():
                        keep_running = False
                        break
                    scene.set_qpos_by_side(
                        {
                            side: branch.measured_qpos[side][index]
                            for side in SIDES
                        }
                    )
                    if target_visualizer.config.enabled:
                        target_visualizer.update(
                            viewer_handle,
                            {
                                side: active.raw_target_pose[side][index]
                                for side in SIDES
                            },
                        )
                    result_visualizer.update(
                        viewer_handle,
                        baseline_endpoint_pose=baseline.measured_pose[
                            str(args.side)
                        ][index],
                        active_endpoint_pose=active.measured_pose[str(args.side)][index],
                        raw_target_pose=active.raw_target_pose[str(args.side)][index],
                        applied_target_pose=active.applied_target_pose[
                            str(args.side)
                        ][index],
                        filtered_force_control_n=active.filtered_force_control_n[index],
                        active_offset_m=active.active_offset_m[index],
                        selected_side=str(args.side),
                    )
                    viewer_handle.sync()
                    status = _frame_status(
                        index=index,
                        baseline=baseline,
                        active=active,
                        selected_side=str(args.side),
                    )
                    if status["time_sec"] + 1e-12 >= next_status_time:
                        force = status["filtered_force_control_n"]
                        torque = status["filtered_torque_control_nm"]
                        wrench_text = (
                            f"torque=[{torque[0]:+.2f},{torque[1]:+.2f},"
                            f"{torque[2]:+.2f}]Nm "
                            f"|dR|={status['active_rotation_norm_deg']:.3f}deg "
                            if rotational_config is not None
                            else f"force=[{force[0]:+.2f},{force[1]:+.2f},"
                            f"{force[2]:+.2f}]N "
                            f"|offset|={status['active_offset_norm_m'] * 1000.0:.3f}mm "
                        )
                        print(
                            "[force-mujoco-viz] "
                            f"branch={branch_name} t={status['time_sec']:.3f}s "
                            f"{wrench_text}"
                            f"dqref={status['qref_delta_max_deg']:.3f}deg "
                            f"dqcmd={status['qcmd_delta_max_deg']:.3f}deg "
                            f"dqmeas={status['measured_qpos_delta_max_deg']:.3f}deg "
                            f"dxmeas={status['measured_endpoint_delta_m'] * 1000.0:.3f}mm",
                            flush=True,
                        )
                        next_status_time = status["time_sec"] + 1.0 / float(
                            args.status_hz
                        )
                    sample_time = float(active.sample_times_sec[index])
                    if previous_sample_time is not None:
                        dt = max(0.0, sample_time - previous_sample_time)
                        if not _sleep_interruptibly(
                            dt / float(args.speed),
                            viewer_handle=viewer_handle,
                        ):
                            keep_running = False
                            break
                    previous_sample_time = sample_time
        if keep_running and bool(args.hold_viewer_at_end):
            print(
                "[force-mujoco-viz] viewer_hold=close the MuJoCo window to exit",
                flush=True,
            )
            while viewer_handle.is_running():
                viewer_handle.sync()
                time.sleep(0.02)
    return dict(report)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\n[force-mujoco-viz] stopped", flush=True)


if __name__ == "__main__":
    main()
