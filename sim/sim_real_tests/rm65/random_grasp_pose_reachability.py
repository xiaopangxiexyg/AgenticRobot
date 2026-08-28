"""Deterministic model-only probes for arbitrary RM65 grasp poses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.rm65.grasp_reachability import (
    ConservativeWorkspaceEnvelope,
    GraspReachabilityReceipt,
    GraspReachabilityStatus,
    IKCandidate,
    evaluate_grasp_pose_reachability,
)
from robot_motion.rm65.ik import RM65SingleArmIKConfig, build_rm65_single_arm_ik_solver

from ..core import SimRealScenarioResult


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "rm65_random_grasp_pose_reachability_v1.json"
)


@dataclass(frozen=True)
class RM65RandomGraspPoseReachabilityConfig:
    seed: int = 20260829
    sample_count: int = 12
    position_min_m: tuple[float, float, float] = (-0.70, -0.80, -0.10)
    position_max_m: tuple[float, float, float] = (0.70, 0.80, 1.20)
    orientation_sampler: str = "shoemake_uniform_so3"
    controlled_frame: str = "tool_tcp"
    max_ik_candidates: int = 6
    ik_max_iterations: int = 250
    ik_max_joint_step_rad: float = 0.10
    ik_damping: float = 0.03
    maximum_position_residual_m: float = 0.003
    maximum_orientation_residual_deg: float = 1.0
    conservative_workspace_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    conservative_workspace_maximum_radius_m: float = 1.25

    def __post_init__(self) -> None:
        minimum = tuple(float(value) for value in self.position_min_m)
        maximum = tuple(float(value) for value in self.position_max_m)
        origin = tuple(float(value) for value in self.conservative_workspace_origin_m)
        if len(minimum) != 3 or len(maximum) != 3 or len(origin) != 3:
            raise ValueError("Cartesian bounds and workspace origin must have 3 values.")
        if not np.all(np.isfinite(minimum + maximum + origin)):
            raise ValueError("Cartesian bounds and workspace origin must be finite.")
        if any(low >= high for low, high in zip(minimum, maximum)):
            raise ValueError("Every position_min_m value must be below position_max_m.")
        if str(self.orientation_sampler) != "shoemake_uniform_so3":
            raise ValueError("orientation_sampler must be 'shoemake_uniform_so3'.")
        positive_fields = (
            "sample_count",
            "max_ik_candidates",
            "ik_max_iterations",
            "ik_max_joint_step_rad",
            "ik_damping",
            "conservative_workspace_maximum_radius_m",
        )
        if any(
            not np.isfinite(float(getattr(self, name)))
            or float(getattr(self, name)) <= 0.0
            for name in positive_fields
        ):
            raise ValueError(f"{positive_fields} must all be positive.")
        for name in ("maximum_position_residual_m", "maximum_orientation_residual_deg"):
            if not np.isfinite(float(getattr(self, name))) or float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "max_ik_candidates", int(self.max_ik_candidates))
        object.__setattr__(self, "ik_max_iterations", int(self.ik_max_iterations))
        object.__setattr__(self, "ik_max_joint_step_rad", float(self.ik_max_joint_step_rad))
        object.__setattr__(self, "ik_damping", float(self.ik_damping))
        object.__setattr__(
            self, "maximum_position_residual_m", float(self.maximum_position_residual_m)
        )
        object.__setattr__(
            self, "maximum_orientation_residual_deg", float(self.maximum_orientation_residual_deg)
        )
        object.__setattr__(
            self,
            "conservative_workspace_maximum_radius_m",
            float(self.conservative_workspace_maximum_radius_m),
        )
        object.__setattr__(self, "position_min_m", minimum)
        object.__setattr__(self, "position_max_m", maximum)
        object.__setattr__(self, "conservative_workspace_origin_m", origin)
        object.__setattr__(self, "controlled_frame", str(self.controlled_frame))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RM65RandomGraspPoseReachabilityConfig":
        values = dict(payload)
        values.pop("schema_version", None)
        values.pop("scenario", None)
        return cls(**values)


@dataclass(frozen=True)
class RandomGraspPoseSample:
    sample_index: int
    target_pose: np.ndarray

    def __post_init__(self) -> None:
        pose = np.asarray(self.target_pose, dtype=float).reshape(4, 4).copy()
        pose.setflags(write=False)
        object.__setattr__(self, "sample_index", int(self.sample_index))
        object.__setattr__(self, "target_pose", pose)


def load_random_grasp_pose_reachability_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> RM65RandomGraspPoseReachabilityConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return RM65RandomGraspPoseReachabilityConfig.from_mapping(payload)


def generate_random_grasp_pose_samples(
    config: RM65RandomGraspPoseReachabilityConfig,
) -> tuple[RandomGraspPoseSample, ...]:
    """Generate every requested pose once using PCG64 and Shoemake SO(3).

    There is intentionally no rejection or retry loop: every sample is emitted
    and must receive one typed terminal classification from the probe.
    """

    rng = np.random.Generator(np.random.PCG64(int(config.seed)))
    lower = np.asarray(config.position_min_m, dtype=float)
    upper = np.asarray(config.position_max_m, dtype=float)
    samples: list[RandomGraspPoseSample] = []
    for sample_index in range(int(config.sample_count)):
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = rng.uniform(lower, upper)
        pose[:3, :3] = _sample_shoemake_rotation_matrix(rng)
        samples.append(RandomGraspPoseSample(sample_index=sample_index, target_pose=pose))
    return tuple(samples)


def build_rm65_multistart_candidate_provider(
    solver: Any,
    *,
    qpos_seeds: Sequence[Sequence[float] | np.ndarray],
    max_iterations: int,
    max_joint_step_rad: float,
    damping: float,
) -> Callable[[int, np.ndarray], IKCandidate]:
    seeds = tuple(np.asarray(seed, dtype=float).reshape(-1).copy() for seed in qpos_seeds)
    if not seeds:
        raise ValueError("qpos_seeds must not be empty.")
    if any(seed.shape != (6,) or not np.all(np.isfinite(seed)) for seed in seeds):
        raise ValueError("Every qpos seed must contain 6 finite RM65 joint values.")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive.")
    if not np.isfinite(float(max_joint_step_rad)) or float(max_joint_step_rad) <= 0.0:
        raise ValueError("max_joint_step_rad must be finite and positive.")
    if not np.isfinite(float(damping)) or float(damping) <= 0.0:
        raise ValueError("damping must be finite and positive.")

    def provide(candidate_index: int, target_pose: np.ndarray) -> IKCandidate:
        seed = seeds[int(candidate_index)]
        qpos = np.asarray(
            solver.solve_iterative(
                target_pose=target_pose,
                q_init=seed,
                max_iterations=int(max_iterations),
                max_joint_step=float(max_joint_step_rad),
                damping=float(damping),
                tolerance=1e-6,
            ),
            dtype=float,
        ).reshape(-1)
        solved_pose = np.asarray(solver.compute_pose(qpos), dtype=float).reshape(4, 4)
        position_residual = float(np.linalg.norm(solved_pose[:3, 3] - target_pose[:3, 3]))
        orientation_residual = float(
            Rotation.from_matrix(target_pose[:3, :3] @ solved_pose[:3, :3].T).magnitude()
        )
        return IKCandidate(
            candidate_index=int(candidate_index),
            qpos_rad=tuple(float(value) for value in qpos),
            converged=bool(getattr(solver, "last_solve_converged", False)),
            position_residual_m=position_residual,
            orientation_residual_rad=orientation_residual,
            iterations=int(getattr(solver, "last_solve_iterations", 0)),
            metadata={
                "seed_index": int(candidate_index),
                "deadline_exceeded": bool(
                    getattr(solver, "last_solve_deadline_exceeded", False)
                ),
            },
        )

    return provide


def make_default_rm65_grasp_pose_evaluator(
    config: RM65RandomGraspPoseReachabilityConfig,
) -> Callable[[RandomGraspPoseSample], GraspReachabilityReceipt]:
    solver = build_rm65_single_arm_ik_solver(
        RM65SingleArmIKConfig(controlled_frame=config.controlled_frame)
    )
    seed_rng = np.random.Generator(np.random.PCG64(int(config.seed) ^ 0x524D3635))
    lower = np.asarray(solver.q_min, dtype=float)
    upper = np.asarray(solver.q_max, dtype=float)
    seeds = [np.clip(np.zeros_like(lower), lower, upper)]
    while len(seeds) < int(config.max_ik_candidates):
        seeds.append(seed_rng.uniform(lower, upper))
    provider = build_rm65_multistart_candidate_provider(
        solver,
        qpos_seeds=seeds,
        max_iterations=config.ik_max_iterations,
        max_joint_step_rad=config.ik_max_joint_step_rad,
        damping=config.ik_damping,
    )
    envelope = ConservativeWorkspaceEnvelope(
        origin_xyz_m=config.conservative_workspace_origin_m,
        maximum_radius_m=config.conservative_workspace_maximum_radius_m,
    )

    def evaluate(sample: RandomGraspPoseSample) -> GraspReachabilityReceipt:
        return evaluate_grasp_pose_reachability(
            sample.target_pose,
            candidate_provider=provider,
            max_candidates=config.max_ik_candidates,
            maximum_position_residual_m=config.maximum_position_residual_m,
            maximum_orientation_residual_rad=np.deg2rad(
                config.maximum_orientation_residual_deg
            ),
            workspace_envelope=envelope,
        )

    return evaluate


def run_random_grasp_pose_reachability(
    config: RM65RandomGraspPoseReachabilityConfig | None = None,
    *,
    evaluator: Callable[[RandomGraspPoseSample], GraspReachabilityReceipt] | None = None,
) -> SimRealScenarioResult:
    cfg = config if config is not None else load_random_grasp_pose_reachability_config()
    samples = generate_random_grasp_pose_samples(cfg)
    resolved_evaluator = (
        evaluator if evaluator is not None else make_default_rm65_grasp_pose_evaluator(cfg)
    )
    records: list[dict[str, Any]] = []
    counts = {status.value: 0 for status in GraspReachabilityStatus}
    for sample in samples:
        try:
            receipt = resolved_evaluator(sample)
            if not isinstance(receipt, GraspReachabilityReceipt):
                raise TypeError("evaluator must return GraspReachabilityReceipt")
        except Exception as exc:
            receipt = GraspReachabilityReceipt(
                status=GraspReachabilityStatus.PLANNER_ERROR,
                reason_code="scenario_evaluator_error",
                proof_scope="scenario_adapter_failure",
                target_pose=tuple(
                    tuple(float(value) for value in row) for row in sample.target_pose
                ),
                metadata={"error_type": type(exc).__name__, "error": str(exc)},
            )
        record = receipt.to_dict()
        record["sample_index"] = sample.sample_index
        records.append(record)
        counts[receipt.status.value] += 1

    system_failure_count = (
        counts[GraspReachabilityStatus.INVALID_TARGET.value]
        + counts[GraspReachabilityStatus.PLANNER_ERROR.value]
    )
    return SimRealScenarioResult(
        scenario="rm65_random_grasp_pose_reachability",
        passed=len(records) == int(cfg.sample_count) and system_failure_count == 0,
        metrics={
            "seed": int(cfg.seed),
            "requested_samples": int(cfg.sample_count),
            "terminal_records": len(records),
            **{f"status_{name}": count for name, count in counts.items()},
        },
        records=tuple(records),
        failure=(
            ""
            if system_failure_count == 0
            else f"{system_failure_count} sample(s) ended in invalid-target/planner error"
        ),
    )


def _sample_shoemake_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = (float(value) for value in rng.random(3))
    quaternion_xyzw = np.asarray(
        (
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
        ),
        dtype=float,
    )
    if quaternion_xyzw[3] < 0.0:
        quaternion_xyzw *= -1.0
    return Rotation.from_quat(quaternion_xyzw).as_matrix()


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "RM65RandomGraspPoseReachabilityConfig",
    "RandomGraspPoseSample",
    "build_rm65_multistart_candidate_provider",
    "generate_random_grasp_pose_samples",
    "load_random_grasp_pose_reachability_config",
    "make_default_rm65_grasp_pose_evaluator",
    "run_random_grasp_pose_reachability",
]
