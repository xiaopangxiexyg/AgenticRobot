"""Replay-based local executable-action probe for temporary PPO policies.

The probe is observational: it never changes PPO likelihood accounting,
reward, actor observations, IK defaults, or the authoritative collision
projector.  One scratch environment repeatedly replays the exact deterministic
action prefix so candidate evaluation has constant memory use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from sim.mujoco.diagnostics.local_executability import (
    LOCAL_EXECUTABILITY_SCHEMA,
    LocalActionCandidate,
    LocalActionOutcome,
    generate_local_action_candidates,
    summarize_local_outcomes,
)
from sim.mujoco.tasks.cube_lift import CubeLiftTaskConfig, FastCubeLiftEnv
from sim.mujoco.tasks.cube_lift.learning import (
    build_policy_action_transform,
    encode_policy_observation,
)

from .checkpoint import (
    build_policy_from_checkpoint,
    check_task_config_compatibility,
    file_sha256,
)
from .policy import resolve_device
from .train import DEFAULT_CHECKPOINT, DEFAULT_TASK_CONFIG


_TRANSLATION_COMPONENTS = (
    "right_tcp_delta_x",
    "right_tcp_delta_y",
    "right_tcp_delta_z",
)


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated finite floats"
        ) from exc
    if not parsed or not all(np.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError(
            "expected comma-separated finite floats"
        )
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the local executable policy-action set at deterministic "
            "collision-projector onset states."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--max-onsets", type=int, default=1)
    parser.add_argument(
        "--onset-category",
        action="append",
        default=[],
        help="Optional blocking category filter; repeat to accept multiple categories.",
    )
    parser.add_argument(
        "--onset-phase",
        action="append",
        choices=("pre_contact", "bilateral_inner_contact", "post_contact"),
        default=[],
        help="Optional manipulation-phase filter; repeat to accept multiple phases.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--radial-scales",
        type=_parse_float_tuple,
        default=(0.0, 0.25, 0.5, 0.75),
        help="Translation-only scales of the raw policy action.",
    )
    parser.add_argument(
        "--repair-radii",
        type=_parse_float_tuple,
        default=(0.25, 0.5),
        help="Normalized policy-space radii around the raw xyz proposal.",
    )
    parser.add_argument(
        "--sphere-directions",
        type=int,
        default=8,
        help="Additional deterministic sphere directions beyond the six axes.",
    )
    parser.add_argument(
        "--projection-identity-threshold",
        type=float,
        default=0.999,
        help="Minimum projection alpha considered unchanged; intervention must also be false.",
    )
    parser.add_argument(
        "--replay-observation-atol",
        type=float,
        default=1e-6,
        help="Fail if deterministic prefix replay does not reproduce the onset observation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path; stdout is always printed.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if int(args.episodes) <= 0 or int(args.max_onsets) <= 0:
        raise ValueError("episodes and max-onsets must be positive.")
    threshold = float(args.projection_identity_threshold)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("projection-identity-threshold must lie in (0, 1].")
    replay_atol = float(args.replay_observation_atol)
    if not np.isfinite(replay_atol) or replay_atol < 0.0:
        raise ValueError("replay-observation-atol must be finite and non-negative.")

    device = resolve_device(args.device)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    policy, normalizer, payload = build_policy_from_checkpoint(
        args.checkpoint,
        device=device,
    )
    compatibility = check_task_config_compatibility(
        payload,
        args.task_config,
        allow_mismatch=False,
    )
    config = CubeLiftTaskConfig.load(args.task_config)
    transform = build_policy_action_transform(config)
    if (
        policy.action_dim != transform.policy_dim
        or policy.environment_action_dim != transform.task_dim
        or tuple(policy.active_action_indices) != transform.task_indices
    ):
        raise ValueError(
            "Checkpoint policy action dimensions do not match the exact task transform."
        )
    try:
        translation_indices = tuple(
            transform.policy_components.index(component)
            for component in _TRANSLATION_COMPONENTS
        )
    except ValueError as exc:
        raise ValueError(
            "Local xyz executability probe requires all three TCP translation components."
        ) from exc

    primary = FastCubeLiftEnv(
        config,
        info_mode="compact",
        use_model_cache=True,
        model_cache_dir=args.model_cache_dir,
    )
    scratch = FastCubeLiftEnv(
        config,
        info_mode="compact",
        use_model_cache=True,
        model_cache_dir=args.model_cache_dir,
    )
    onset_reports: list[dict[str, object]] = []
    evaluated_policy_steps = 0
    try:
        for episode_index in range(int(args.episodes)):
            episode_seed = int(args.seed) + episode_index
            observation, pre_info = primary.reset(seed=episode_seed)
            action_prefix: list[np.ndarray] = []
            previous_intervention = False
            for episode_step in range(config.max_episode_steps):
                latent_action = _predict_policy_action(
                    policy,
                    normalizer,
                    config,
                    observation,
                    device=device,
                )
                task_action = np.asarray(
                    transform.transform(latent_action, observation=observation),
                    dtype=np.float32,
                )
                (
                    next_observation,
                    _reward,
                    terminated,
                    truncated,
                    info,
                ) = primary.step(task_action)
                evaluated_policy_steps += 1
                intervention = bool(
                    info.get("safety_intervention", info.get("safety_modified", False))
                )
                onset = intervention and not previous_intervention
                onset_category = info.get("safety_blocking_category")
                onset_phase = _manipulation_phase(info)
                category_matches = bool(
                    not args.onset_category
                    or onset_category in set(args.onset_category)
                )
                phase_matches = bool(
                    not args.onset_phase or onset_phase in set(args.onset_phase)
                )
                if onset and category_matches and phase_matches:
                    candidates = generate_local_action_candidates(
                        latent_action,
                        translation_indices=translation_indices,
                        radial_scales=args.radial_scales,
                        repair_radii=args.repair_radii,
                        sphere_directions=int(args.sphere_directions),
                    )
                    outcomes = tuple(
                        _evaluate_candidate(
                            scratch,
                            candidate=candidate,
                            transform=transform,
                            episode_seed=episode_seed,
                            action_prefix=action_prefix,
                            expected_observation=observation,
                            expected_pre_info=pre_info,
                            replay_atol=replay_atol,
                            projection_identity_threshold=threshold,
                            success_termination_reason=(
                                config.success_termination_reason
                            ),
                        )
                        for candidate in candidates
                    )
                    zero_policy_action = np.zeros(
                        transform.policy_dim,
                        dtype=np.float32,
                    )
                    diagnostic_controls = tuple(
                        _evaluate_candidate(
                            scratch,
                            candidate=LocalActionCandidate(
                                label=label,
                                family="diagnostic_measured_pose_reanchor",
                                policy_action=tuple(float(value) for value in values),
                            ),
                            transform=transform,
                            episode_seed=episode_seed,
                            action_prefix=action_prefix,
                            expected_observation=observation,
                            expected_pre_info=pre_info,
                            replay_atol=replay_atol,
                            projection_identity_threshold=threshold,
                            success_termination_reason=(
                                config.success_termination_reason
                            ),
                            reanchor_pose_target=True,
                        )
                        for label, values in (
                            ("measured_pose_reanchor:zero", zero_policy_action),
                            ("measured_pose_reanchor:raw", latent_action),
                        )
                    )
                    reanchored_outcomes = tuple(
                        _evaluate_candidate(
                            scratch,
                            candidate=candidate,
                            transform=transform,
                            episode_seed=episode_seed,
                            action_prefix=action_prefix,
                            expected_observation=observation,
                            expected_pre_info=pre_info,
                            replay_atol=replay_atol,
                            projection_identity_threshold=threshold,
                            success_termination_reason=(
                                config.success_termination_reason
                            ),
                            reanchor_pose_target=True,
                        )
                        for candidate in candidates
                    )
                    raw_outcome = next(
                        outcome
                        for outcome in outcomes
                        if outcome.candidate.family == "raw"
                    )
                    primary_alpha = float(info.get("safety_projection_alpha", 1.0))
                    if not np.isclose(
                        raw_outcome.projection_alpha,
                        primary_alpha,
                        rtol=0.0,
                        atol=1e-9,
                    ):
                        raise RuntimeError(
                            "Scratch raw-action replay changed the projector outcome: "
                            f"primary_alpha={primary_alpha}, "
                            f"scratch_alpha={raw_outcome.projection_alpha}."
                        )
                    onset_reports.append(
                        {
                            "episode_index": episode_index,
                            "episode_seed": episode_seed,
                            "episode_step": episode_step,
                            "observation_sha256": _array_sha256(observation),
                            "raw_policy_action": latent_action.astype(float).tolist(),
                            "raw_task_action": task_action.astype(float).tolist(),
                            "raw_blocking_category": info.get(
                                "safety_blocking_category"
                            ),
                            "raw_blocking_body_pair": _blocking_body_pair(info),
                            "phase": onset_phase,
                            "pre_action_pose_target_translation_error_m": (
                                info.get("pose_target_pre_action_translation_error_m")
                            ),
                            "pre_action_pose_target_rotation_error_deg": (
                                info.get("pose_target_pre_action_rotation_error_deg")
                            ),
                            "pre_action_arm_tracking_error_rad_mean": (
                                pre_info.get("arm_tracking_error_rad_mean")
                            ),
                            "pre_action_arm_tracking_error_rad_max": (
                                pre_info.get("arm_tracking_error_rad_max")
                            ),
                            "summary": summarize_local_outcomes(
                                outcomes,
                                translation_indices=translation_indices,
                            ),
                            "diagnostic_controls": [
                                outcome.to_dict()
                                for outcome in diagnostic_controls
                            ],
                            "measured_pose_reanchored_summary": (
                                summarize_local_outcomes(
                                    reanchored_outcomes,
                                    translation_indices=translation_indices,
                                )
                            ),
                            "measured_pose_reanchored_candidates": [
                                outcome.to_dict()
                                for outcome in reanchored_outcomes
                            ],
                            "candidates": [outcome.to_dict() for outcome in outcomes],
                        }
                    )
                    if len(onset_reports) >= int(args.max_onsets):
                        break

                action_prefix.append(task_action.copy())
                observation = next_observation
                pre_info = info
                previous_intervention = intervention
                if terminated or truncated:
                    break
            if len(onset_reports) >= int(args.max_onsets):
                break
    finally:
        primary.close()
        scratch.close()

    summaries = [report["summary"] for report in onset_reports]
    report: dict[str, object] = {
        "schema": LOCAL_EXECUTABILITY_SCHEMA,
        "task_config": str(Path(args.task_config).resolve()),
        "task_config_sha256": config.source_sha256,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "compatibility": compatibility,
        "device": str(device),
        "seed": int(args.seed),
        "episodes_requested": int(args.episodes),
        "evaluated_policy_steps": evaluated_policy_steps,
        "onset_count": len(onset_reports),
        "candidate_settings": {
            "translation_policy_indices": list(translation_indices),
            "radial_scales": [float(value) for value in args.radial_scales],
            "repair_radii": [float(value) for value in args.repair_radii],
            "sphere_directions": int(args.sphere_directions),
            "projection_identity_threshold": threshold,
            "replay_observation_atol": replay_atol,
            "onset_categories": list(args.onset_category),
            "onset_phases": list(args.onset_phase),
        },
        "aggregate": {
            "raw_executable_rate": (
                None
                if not summaries
                else sum(bool(summary["raw_executable"]) for summary in summaries)
                / len(summaries)
            ),
            "executable_ratio_mean": (
                None
                if not summaries
                else float(
                    np.mean(
                        [float(summary["executable_ratio"]) for summary in summaries]
                    )
                )
            ),
            "nearest_executable_distance_mean": _optional_mean(
                summary["nearest_executable_distance"] for summary in summaries
            ),
            "maximum_executable_radial_scale_mean": _optional_mean(
                summary["maximum_executable_radial_scale"] for summary in summaries
            ),
        },
        "onsets": onset_reports,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return report


def _predict_policy_action(
    policy: Any,
    normalizer: Any,
    config: CubeLiftTaskConfig,
    observation: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    actor_observation = encode_policy_observation(config, observation[None, :])
    with torch.no_grad():
        action, _log_prob = policy.predict_action(
            torch.as_tensor(
                normalizer.normalize(actor_observation),
                device=device,
            ),
            deterministic=True,
        )
    return action.cpu().numpy()[0].astype(np.float32, copy=False)


def _evaluate_candidate(
    scratch: FastCubeLiftEnv,
    *,
    candidate: LocalActionCandidate,
    transform: Any,
    episode_seed: int,
    action_prefix: Sequence[np.ndarray],
    expected_observation: np.ndarray,
    expected_pre_info: dict[str, Any],
    replay_atol: float,
    projection_identity_threshold: float,
    success_termination_reason: str,
    reanchor_pose_target: bool = False,
) -> LocalActionOutcome:
    observation, pre_info = scratch.reset(seed=episode_seed)
    for prefix_index, action in enumerate(action_prefix):
        observation, _reward, terminated, truncated, pre_info = scratch.step(action)
        if terminated or truncated:
            raise RuntimeError(
                "Deterministic scratch replay terminated before the probed state: "
                f"prefix_index={prefix_index}, "
                f"reason={pre_info.get('termination_reason')}."
            )
    if observation.shape != expected_observation.shape:
        raise RuntimeError(
            "Scratch replay observation shape changed: "
            f"expected={expected_observation.shape}, got={observation.shape}."
        )
    replay_error = float(np.max(np.abs(observation - expected_observation)))
    if replay_error > replay_atol:
        raise RuntimeError(
            "Scratch replay did not reproduce the probed observation: "
            f"max_abs_error={replay_error:.9g}, atol={replay_atol:.9g}."
        )
    expected_progress = float(expected_pre_info.get("objective_progress", 0.0))
    replay_progress = float(pre_info.get("objective_progress", 0.0))
    if not np.isclose(expected_progress, replay_progress, rtol=0.0, atol=1e-8):
        raise RuntimeError(
            "Scratch replay changed pre-action objective progress: "
            f"expected={expected_progress}, got={replay_progress}."
        )

    if reanchor_pose_target:
        controller = scratch.kernel._pose_action_controller
        if controller is None:
            raise RuntimeError(
                "Measured-pose reanchor diagnostic requires a pose-action controller."
            )
        controller.reanchor_target_from_scene()

    policy_action = np.asarray(candidate.policy_action, dtype=np.float32)
    task_action = np.asarray(
        transform.transform(policy_action, observation=observation),
        dtype=np.float32,
    )
    (
        _next_observation,
        _reward,
        terminated,
        _truncated,
        info,
    ) = scratch.step(task_action)
    intervention = bool(
        info.get("safety_intervention", info.get("safety_modified", False))
    )
    alpha = float(info.get("safety_projection_alpha", 1.0))
    pose_ik_success = info.get("pose_ik_success")
    pose_ik_hold = info.get("pose_ik_hold")
    ik_feasible = bool(
        (True if pose_ik_success is None else pose_ik_success)
        and not bool(pose_ik_hold)
    )
    projector_identity = bool(
        not intervention and alpha >= projection_identity_threshold
    )
    actual_forbidden_contact = bool(
        info.get("safety_actual_forbidden_contact", False)
    )
    hard_violation = bool(info.get("safety_actual_hard_violation", False))
    termination_reason = info.get("termination_reason")
    unexpected_termination = bool(
        terminated and termination_reason != success_termination_reason
    )
    safety_status = info.get("safety_status")
    fully_rejected = bool(
        (intervention and alpha <= 1e-9)
        or safety_status in {"rejected", "invalid_safe_state"}
    )
    progress_delta = float(info.get("objective_progress", 0.0)) - replay_progress
    pre_reach_distance = expected_pre_info.get("reach_distance_m")
    post_reach_distance = info.get("reach_distance_m")
    goal_distance_progress_m = (
        None
        if pre_reach_distance is None or post_reach_distance is None
        else float(pre_reach_distance) - float(post_reach_distance)
    )
    return LocalActionOutcome(
        candidate=candidate,
        task_action=tuple(float(value) for value in task_action),
        ik_feasible=ik_feasible,
        projector_identity=projector_identity,
        executable=bool(
            ik_feasible
            and projector_identity
            and not actual_forbidden_contact
            and not hard_violation
            and not unexpected_termination
        ),
        projection_alpha=alpha,
        safety_status=None if safety_status is None else str(safety_status),
        fully_rejected=fully_rejected,
        actual_forbidden_contact=actual_forbidden_contact,
        hard_violation=hard_violation,
        objective_progress_delta=progress_delta,
        goal_distance_progress_m=goal_distance_progress_m,
        blocking_category=(
            None
            if info.get("safety_blocking_category") is None
            else str(info["safety_blocking_category"])
        ),
        blocking_body_pair=_blocking_body_pair(info),
        termination_reason=(
            None if termination_reason is None else str(termination_reason)
        ),
    )


def _blocking_body_pair(info: dict[str, Any]) -> str | None:
    body1 = info.get("safety_blocking_body1")
    body2 = info.get("safety_blocking_body2")
    if body1 is None or body2 is None:
        return None
    return f"{body1}/{body2}"


def _manipulation_phase(info: dict[str, Any]) -> str:
    if bool(info.get("post_contact_phase_active", False)):
        return "post_contact"
    if float(info.get("gripper_bilateral_inner_contact_score", 0.0)) >= 0.5:
        return "bilateral_inner_contact"
    return "pre_contact"


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _optional_mean(values: Any) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.mean(finite))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_arg_parser", "main", "run"]
