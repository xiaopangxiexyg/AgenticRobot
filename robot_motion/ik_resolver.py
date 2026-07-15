from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .interfaces import CollisionChecker, KinematicsProvider
from .joint_limits import JointVelocityLimitProfile, limit_joint_step
from .pose import pose_delta_metrics, relax_pose_toward_reference
from .posture_preferences import JointPosturePreference, posture_preference_from_metadata
from .profiles import IKSafetyProfile
from .seed_expansion import SeedExpansionProfile, expand_seed_qpos
from .types import IKCandidate, IKResolveRequest, IKResolveResult, IKResolveStatus, SingularityReport


@dataclass(frozen=True)
class _EvaluatedCandidate:
    candidate: IKCandidate
    pose_error_m: float | None
    pose_error_deg: float | None
    score: float
    metadata: dict[str, Any]


def _hard_deadline_monotonic(request: IKResolveRequest) -> float | None:
    if not bool(request.metadata.get("hard_deadline_enabled", False)):
        return None
    return _finite_float(request.metadata.get("hard_deadline_monotonic"), default=None)


def _hard_deadline_expired(deadline_monotonic: float | None) -> bool:
    return bool(
        deadline_monotonic is not None
        and time.perf_counter() >= float(deadline_monotonic)
    )


def _hard_deadline_metadata(
    request: IKResolveRequest,
    *,
    deadline_monotonic: float | None,
    expired: bool,
    phase: str,
) -> dict[str, Any]:
    if deadline_monotonic is None:
        return {"hard_deadline_enabled": False}
    remaining_ms = max(
        0.0,
        (float(deadline_monotonic) - time.perf_counter()) * 1000.0,
    )
    return {
        "hard_deadline_enabled": True,
        "hard_deadline_budget_ms": _finite_float(
            request.metadata.get("hard_deadline_budget_ms"),
            default=None,
        ),
        "hard_deadline_expired": bool(expired),
        "hard_deadline_phase": str(phase),
        "hard_deadline_remaining_ms": float(remaining_ms),
    }


def _provider_options_with_hard_deadline(
    options: dict[str, Any],
    *,
    hard_deadline: float | None,
) -> dict[str, Any]:
    """Propagate the earliest resolver deadline to cooperative providers.

    A provider cannot preempt an SDK call already in progress, but it must not
    start another solve/seed-expansion stage after the resolver deadline.  Keep
    an upstream provider deadline when it is earlier than the force-control
    deadline.
    """

    out = dict(options)
    provider_deadline = _finite_float(
        out.get("provider_deadline_monotonic"),
        default=None,
    )
    deadlines = [
        value
        for value in (provider_deadline, hard_deadline)
        if value is not None and float(value) > 0.0
    ]
    if not deadlines:
        return out
    effective_deadline = min(float(value) for value in deadlines)
    out["provider_deadline_monotonic"] = effective_deadline
    out["provider_remaining_time_budget_ms"] = max(
        0.0,
        (effective_deadline - time.perf_counter()) * 1000.0,
    )
    return out


class IKSafetyResolver:
    def __init__(
        self,
        *,
        profile: IKSafetyProfile,
        provider: KinematicsProvider,
        collision_checker: CollisionChecker | None = None,
        seed_expansion_profile: SeedExpansionProfile | None = None,
    ):
        self.profile = profile
        self.provider = provider
        self.collision_checker = collision_checker
        self.seed_expansion_profile = seed_expansion_profile
        self._resolve_step_counter = 0
        self.posture_preference: JointPosturePreference | None = posture_preference_from_metadata(
            profile.metadata
        )

    def resolve(
        self,
        request: IKResolveRequest,
        *,
        timing_profile: dict[str, Any] | None = None,
    ) -> IKResolveResult:
        started_at = time.perf_counter()
        own_timing_profile = timing_profile is None
        if timing_profile is None:
            timing_profile = _new_timing_profile("resolve")
        hard_deadline = _hard_deadline_monotonic(request)
        hard_deadline_expired = False
        if hard_deadline is not None:
            timing_profile["hard_deadline_enabled"] = True
            timing_profile["hard_deadline_budget_ms"] = _finite_float(
                request.metadata.get("hard_deadline_budget_ms"),
                default=None,
            )
            hard_budget_ms = _finite_float(
                request.metadata.get("hard_deadline_budget_ms"),
                default=None,
            )
            existing_budget_ms = _finite_float(timing_profile.get("time_budget_ms"), default=None)
            if hard_budget_ms is not None:
                timing_profile["time_budget_ms"] = (
                    hard_budget_ms
                    if existing_budget_ms is None
                    else min(existing_budget_ms, hard_budget_ms)
                )
            if _hard_deadline_expired(hard_deadline):
                result = self._hard_deadline_hold_result(
                    request=request,
                    phase="resolve_entry",
                )
                return (
                    self._with_timing_metadata(result, timing_profile, started_at)
                    if own_timing_profile
                    else result
                )
        target_pose = np.asarray(request.target_pose, dtype=float).reshape(4, 4)
        reference_qpos = np.asarray(request.reference_qpos, dtype=float).reshape(-1)
        if request.reference_pose is not None:
            reference_pose = np.asarray(request.reference_pose, dtype=float).reshape(4, 4)
        else:
            reference_fk_started = time.perf_counter()
            reference_pose = self.provider.forward_kinematics(reference_qpos)
            _timing_add(timing_profile, "reference_fk_ms", _elapsed_ms(reference_fk_started))
        if _hard_deadline_expired(hard_deadline):
            result = self._hard_deadline_hold_result(
                request=request,
                phase="reference_pose",
            )
            return (
                self._with_timing_metadata(result, timing_profile, started_at)
                if own_timing_profile
                else result
            )
        if reference_pose is None:
            result = self._failed_result(
                request,
                target_pose,
                target_pose,
                "reference_pose_unavailable",
                {"reason": "provider did not provide FK for reference_qpos"},
            )
            return (
                self._with_timing_metadata(result, timing_profile, started_at)
                if own_timing_profile
                else result
            )

        seed_started = time.perf_counter()
        seeds, seed_labels, seed_expansion_metadata = self._seed_tuple(request, reference_qpos)
        _timing_add(timing_profile, "seed_ms", _elapsed_ms(seed_started))
        if _hard_deadline_expired(hard_deadline):
            result = self._hard_deadline_hold_result(
                request=request,
                phase="seed_expansion",
            )
            return (
                self._with_timing_metadata(result, timing_profile, started_at)
                if own_timing_profile
                else result
            )
        rejection_log: list[dict[str, Any]] = []
        best_rejected: dict[str, Any] | None = None

        for raw_fraction in self.profile.relaxation_fractions:
            if _hard_deadline_expired(hard_deadline):
                hard_deadline_expired = True
                break
            fraction = max(0.0, min(1.0, float(raw_fraction)))
            _timing_inc(timing_profile, "full_relaxation_attempts")
            tolerance = self.profile.target_tolerance.scaled(fraction)
            effective_target = (
                target_pose
                if fraction <= 0.0
                else relax_pose_toward_reference(
                    target_pose,
                    reference_pose,
                    max_position_m=tolerance.position_m,
                    max_orientation_deg=tolerance.orientation_deg,
                )
            )
            provider_started = time.perf_counter()
            candidates = self.provider.iter_ik_candidates(
                target_pose=effective_target,
                seeds=seeds,
                seed_labels=seed_labels,
                options=_provider_options_with_hard_deadline(
                    {
                        "enable_expensive_seed_expansion": bool(self.profile.enable_expensive_seed_expansion),
                        "allow_parallel": bool(self.profile.allow_parallel_candidate_generation),
                        "max_candidate_count": int(self.profile.max_candidate_count),
                        "provider_time_budget_ms": self.profile.provider_time_budget_ms,
                        "tolerance_fraction": fraction,
                        **request.metadata,
                    },
                    hard_deadline=hard_deadline,
                ),
            )
            provider_elapsed_ms = _elapsed_ms(provider_started)
            _record_provider_call(
                timing_profile,
                phase="full",
                provider_metadata=getattr(self.provider, "last_metadata", {}),
                candidate_count=len(candidates),
                measured_elapsed_ms=provider_elapsed_ms,
                extra={"tolerance_fraction": fraction},
            )
            evaluated: list[_EvaluatedCandidate] = []
            evaluate_started = time.perf_counter()
            for candidate in candidates:
                if _hard_deadline_expired(hard_deadline):
                    hard_deadline_expired = True
                    break
                accepted, metadata = self._evaluate_candidate(
                    candidate=candidate,
                    reference_qpos=reference_qpos,
                    effective_target=effective_target,
                    reference_pose=reference_pose,
                    request_metadata=request.metadata,
                    timing_profile=timing_profile,
                    timing_phase="full",
                )
                if _hard_deadline_expired(hard_deadline):
                    hard_deadline_expired = True
                    break
                if accepted:
                    evaluated.append(
                        _EvaluatedCandidate(
                            candidate=candidate,
                            pose_error_m=metadata.get("pose_error_m"),
                            pose_error_deg=metadata.get("pose_error_deg"),
                            score=float(metadata.get("score", 0.0)),
                            metadata=metadata,
                        )
                    )
                else:
                    rejection_log.append(metadata)
                    if best_rejected is None or float(metadata.get("score", 1e18)) < float(best_rejected.get("score", 1e18)):
                        best_rejected = metadata
            _timing_add(timing_profile, "full_evaluate_ms", _elapsed_ms(evaluate_started))
            _timing_inc(timing_profile, "full_candidate_count", len(candidates))
            _timing_inc(timing_profile, "full_accepted_candidate_count", len(evaluated))
            if evaluated:
                selected = min(evaluated, key=lambda item: item.score)
                pose_slack_tier = float(selected.metadata.get("pose_slack_tier", 0.0) or 0.0)
                coarse_fraction = (
                    float(selected.metadata.get("coarse_tracking_pose_fraction", 0.0) or 0.0)
                    if bool(selected.metadata.get("coarse_tracking_accepted", False))
                    else 0.0
                )
                result_fraction = max(
                    fraction,
                    pose_slack_tier if fraction <= 0.0 else fraction,
                    coarse_fraction,
                )
                status = IKResolveStatus.PASS if result_fraction <= 0.0 else IKResolveStatus.RELAXED
                message = "ik_solution_accepted" if status == IKResolveStatus.PASS else "ik_solution_relaxed"
                if bool(selected.metadata.get("coarse_tracking_accepted", False)):
                    message = "ik_solution_coarse_tracking"
                result = IKResolveResult(
                    status=status,
                    qpos=selected.candidate.qpos.copy(),
                    requested_target_pose=target_pose,
                    effective_target_pose=effective_target,
                    tolerance_fraction=result_fraction,
                    pose_error_m=selected.pose_error_m,
                    pose_error_deg=selected.pose_error_deg,
                    candidate_source=selected.candidate.source,
                    message=message,
                    metadata={
                        "profile": self.profile.name,
                        "status": status.value,
                        "candidate": dict(selected.candidate.metadata),
                        "candidate_source": selected.candidate.source,
                        "seed_source": selected.candidate.seed_source,
                        "method": selected.candidate.method,
                        "tolerance_fraction": result_fraction,
                        "target_relaxation_fraction": fraction,
                        "rejection_count": len(rejection_log),
                        "seed_expansion": seed_expansion_metadata,
                        **_hard_deadline_metadata(
                            request,
                            deadline_monotonic=hard_deadline,
                            expired=hard_deadline_expired,
                            phase="full_candidate_evaluation",
                        ),
                        **selected.metadata,
                    },
                )
                return (
                    self._with_timing_metadata(result, timing_profile, started_at)
                    if own_timing_profile
                    else result
                )
            if hard_deadline_expired:
                break

        metadata = {
            "profile": self.profile.name,
            "status": IKResolveStatus.SMOOTHING_REQUIRED.value
            if self.profile.allow_smoothing_recovery
            else IKResolveStatus.FAILED.value,
            "rejection_count": len(rejection_log),
            "last_rejections": rejection_log[-8:],
            "best_rejected": best_rejected,
            "seed_expansion": seed_expansion_metadata,
            "provider": dict(getattr(self.provider, "last_metadata", {})),
            **_hard_deadline_metadata(
                request,
                deadline_monotonic=hard_deadline,
                expired=hard_deadline_expired,
                phase="full_candidate_evaluation",
            ),
        }
        status = (
            IKResolveStatus.SMOOTHING_REQUIRED
            if self.profile.allow_smoothing_recovery
            else IKResolveStatus.FAILED
        )
        result = IKResolveResult(
            status=status,
            qpos=None,
            requested_target_pose=target_pose,
            effective_target_pose=target_pose,
            tolerance_fraction=1.0,
            message=(
                "no_safe_ik_solution_within_tolerance; smoothing recovery required"
                if status == IKResolveStatus.SMOOTHING_REQUIRED
                else "no_safe_ik_solution_within_tolerance"
            ),
            metadata=metadata,
        )
        return (
            self._with_timing_metadata(result, timing_profile, started_at)
            if own_timing_profile
            else result
        )

    def resolve_step(
        self,
        request: IKResolveRequest,
        *,
        joint_velocity_profile: JointVelocityLimitProfile | None,
        dt_sec: float,
        local_provider_names: tuple[str, ...] | list[str] = (),
        local_options: dict[str, Any] | None = None,
        allow_full_solve: bool = True,
    ) -> IKResolveResult:
        """Resolve IK for a realtime command step.

        ``resolve`` returns a complete IK target. For high-frequency control, a
        complete target may be many velocity-limited frames away. Profiles can
        therefore enable a local-first receding-horizon path: try one bounded
        local step toward the latest pose target, and only fall back to the full
        multi-provider search when the local step cannot make safe progress or
        when a periodic full solve is due. ``allow_full_solve=False`` is for a
        caller that owns full solves on a separate scheduler: this method then
        performs only the bounded local attempt and returns a safe hold rather
        than unexpectedly starting an expensive search on the realtime thread.
        """
        started_at = time.perf_counter()
        timing_profile = _new_timing_profile("resolve_step")
        self._resolve_step_counter += 1
        local_provider_names_tuple = tuple(str(name) for name in local_provider_names if str(name).strip())
        local_options_dict = dict(local_options or {})
        local_step_possible = (
            joint_velocity_profile is not None
            and bool(self.profile.local_step_enabled)
            and bool(local_provider_names_tuple)
        )
        full_solve_interval = max(0, int(getattr(self.profile, "local_step_first_full_solve_interval", 0) or 0))
        periodic_full_solve = bool(allow_full_solve) and (
            full_solve_interval > 0
            and self._resolve_step_counter % full_solve_interval == 0
        )
        hard_deadline = _hard_deadline_monotonic(request)
        frame_budget_ms = _finite_float(self.profile.provider_time_budget_ms, default=None)
        if hard_deadline is not None:
            frame_deadline = float(hard_deadline)
            frame_budget_ms = _finite_float(
                request.metadata.get("hard_deadline_budget_ms"),
                default=frame_budget_ms,
            )
            timing_profile["hard_deadline_enabled"] = True
            timing_profile["hard_deadline_budget_ms"] = frame_budget_ms
        else:
            frame_deadline = (
                None
                if frame_budget_ms is None
                else started_at + max(0.0, float(frame_budget_ms)) / 1000.0
            )
        if frame_budget_ms is not None:
            timing_profile["time_budget_ms"] = float(frame_budget_ms)
        if _hard_deadline_expired(hard_deadline):
            return self._with_timing_metadata(
                self._hard_deadline_hold_result(
                    request=request,
                    phase="resolve_step_entry",
                ),
                timing_profile,
                started_at,
            )
        local_first_metadata: dict[str, Any] = {
            "local_step_first_enabled": bool(getattr(self.profile, "local_step_first_enabled", False)),
            "local_step_first_full_solve_interval": full_solve_interval,
            "local_step_first_step_index": self._resolve_step_counter,
            "local_step_frame_budget_ms": frame_budget_ms,
        }
        local_fallback_result: IKResolveResult | None = None
        if (
            local_step_possible
            and bool(getattr(self.profile, "local_step_first_enabled", False))
        ):
            rate_metadata = {
                **local_first_metadata,
                "local_step_enabled": True,
                "local_step_trigger_ratio": float(self.profile.local_step_trigger_ratio),
                "local_step_provider_names": [str(name) for name in local_provider_names_tuple],
                "local_step_first_attempted": True,
                "local_step_first_periodic_full_solve": bool(periodic_full_solve),
                "local_step_full_target_limited": None,
                "local_step_full_target_ratio": None,
            }
            local_result = self._resolve_local_step(
                request=request,
                full_result=None,
                joint_velocity_profile=joint_velocity_profile,
                dt_sec=dt_sec,
                local_provider_names=local_provider_names_tuple,
                local_options={**local_options_dict, "local_step_enable_seed_expansion": False},
                frame_deadline=frame_deadline,
                rate_metadata=rate_metadata,
                timing_profile=timing_profile,
                timing_phase="local_first",
            )
            if _hard_deadline_expired(hard_deadline):
                if local_result is not None:
                    return self._with_timing_metadata(
                        self._with_step_metadata(
                            local_result,
                            _hard_deadline_metadata(
                                request,
                                deadline_monotonic=hard_deadline,
                                expired=True,
                                phase="local_first",
                            ),
                        ),
                        timing_profile,
                        started_at,
                    )
                return self._with_timing_metadata(
                    self._hard_deadline_hold_result(
                        request=request,
                        phase="local_first",
                    ),
                    timing_profile,
                    started_at,
                )
            if local_result is not None:
                skip_reason = self._local_first_full_solve_skip_reason(
                    local_result=local_result,
                    started_at=started_at,
                    frame_deadline=frame_deadline,
                )
                if periodic_full_solve and skip_reason is not None:
                    return self._with_timing_metadata(
                        self._with_step_metadata(
                            local_result,
                            {
                                "local_step_first_status": "accepted",
                                "local_step_first_periodic_full_solve": True,
                                "local_step_periodic_full_solve_skipped": True,
                                "local_step_periodic_full_solve_skip_reason": skip_reason,
                            },
                        ),
                        timing_profile,
                        started_at,
                    )
                if not periodic_full_solve:
                    return self._with_timing_metadata(local_result, timing_profile, started_at)
                local_fallback_result = self._with_step_metadata(
                    local_result,
                    {
                        "local_step_first_status": "accepted",
                        "local_step_first_periodic_full_solve": True,
                        "local_step_periodic_full_solve_fallback_available": True,
                    },
                )
                local_first_metadata = {
                    **rate_metadata,
                    "local_step_first_status": "accepted",
                    "local_step_first_periodic_full_solve": True,
                    "local_step_periodic_full_solve_fallback_available": True,
                    "local_step_periodic_full_solve_fallback_source": local_result.candidate_source,
                    "local_step_periodic_full_solve_fallback_method": str(
                        local_result.metadata.get("method", "")
                    ),
                }
            else:
                local_first_metadata = {
                    **rate_metadata,
                    "local_step_first_status": "failed_falling_back_to_full_resolve",
                }
        elif bool(getattr(self.profile, "local_step_first_enabled", False)):
            local_first_metadata = {
                **local_first_metadata,
                "local_step_first_attempted": False,
                "local_step_first_periodic_full_solve": bool(periodic_full_solve),
            }

        if not bool(allow_full_solve):
            hold = self._hard_deadline_hold_result(
                request=request,
                phase="local_only_no_safe_candidate",
            )
            return self._with_timing_metadata(
                self._with_step_metadata(
                    hold,
                    {
                        **local_first_metadata,
                        "local_step_status": "no_safe_local_candidate_hold_reference",
                        "full_solve_deferred_to_external_scheduler": True,
                    },
                ),
                timing_profile,
                started_at,
            )

        if _hard_deadline_expired(hard_deadline):
            if local_fallback_result is not None:
                return self._with_timing_metadata(
                    self._with_step_metadata(
                        local_fallback_result,
                        _hard_deadline_metadata(
                            request,
                            deadline_monotonic=hard_deadline,
                            expired=True,
                            phase="before_full_resolve",
                        ),
                    ),
                    timing_profile,
                    started_at,
                )
            return self._with_timing_metadata(
                self._hard_deadline_hold_result(
                    request=request,
                    phase="before_full_resolve",
                ),
                timing_profile,
                started_at,
            )

        full_started = time.perf_counter()
        full_request = request
        full_provider_budget_ms = frame_budget_ms
        if full_provider_budget_ms is not None:
            full_deadline = (
                frame_deadline
                if frame_deadline is not None
                else time.perf_counter() + max(0.0, float(full_provider_budget_ms)) / 1000.0
            )
            remaining_ms = max(0.0, (float(full_deadline) - time.perf_counter()) * 1000.0)
            full_request = IKResolveRequest(
                target_pose=request.target_pose,
                reference_qpos=request.reference_qpos,
                reference_pose=request.reference_pose,
                seed_qpos=request.seed_qpos,
                seed_labels=request.seed_labels,
                previous_safe_qpos=request.previous_safe_qpos,
                metadata={
                    **dict(request.metadata),
                    "provider_deadline_monotonic": float(full_deadline),
                    "provider_global_time_budget_ms": float(full_provider_budget_ms),
                    "provider_remaining_time_budget_ms": float(remaining_ms),
                    "provider_global_time_budget_scope": "resolve_step_full_solve",
                },
            )
            timing_profile["full_provider_global_time_budget_ms"] = float(full_provider_budget_ms)
            timing_profile["full_provider_remaining_time_budget_ms"] = float(remaining_ms)
        full_result = self.resolve(full_request, timing_profile=timing_profile)
        _timing_add(timing_profile, "full_solve_wall_ms", _elapsed_ms(full_started))
        if _hard_deadline_expired(hard_deadline):
            if full_result.accepted and full_result.qpos is not None:
                return self._with_timing_metadata(
                    self._with_step_metadata(
                        full_result,
                        _hard_deadline_metadata(
                            request,
                            deadline_monotonic=hard_deadline,
                            expired=True,
                            phase="full_resolve",
                        ),
                    ),
                    timing_profile,
                    started_at,
                )
            if local_fallback_result is not None:
                return self._with_timing_metadata(
                    self._with_step_metadata(
                        local_fallback_result,
                        _hard_deadline_metadata(
                            request,
                            deadline_monotonic=hard_deadline,
                            expired=True,
                            phase="full_resolve_fallback",
                        ),
                    ),
                    timing_profile,
                    started_at,
                )
            return self._with_timing_metadata(
                self._hard_deadline_hold_result(
                    request=request,
                    phase="full_resolve",
                ),
                timing_profile,
                started_at,
            )
        if not full_result.accepted or full_result.qpos is None:
            if local_fallback_result is not None:
                return self._with_timing_metadata(
                    self._with_step_metadata(
                        local_fallback_result,
                        {
                            "local_step_full_solve_fallback_used": True,
                            "local_step_full_solve_status": full_result.status.value,
                            "local_step_full_solve_message": full_result.message,
                            "local_step_full_solve_candidate_source": full_result.candidate_source,
                            "full_target_result_message": full_result.message,
                            "full_target_failure_status": full_result.status.value,
                            "full_target_failure_best_rejected": full_result.metadata.get("best_rejected"),
                            "full_target_failure_last_rejections": full_result.metadata.get("last_rejections"),
                            "full_target_failure_rejection_count": full_result.metadata.get("rejection_count"),
                            "full_target_failure_provider": full_result.metadata.get("provider"),
                        },
                    ),
                    timing_profile,
                    started_at,
                )
            return self._with_timing_metadata(
                self._with_step_metadata(full_result, local_first_metadata),
                timing_profile,
                started_at,
            )
        if (
            joint_velocity_profile is None
            or not bool(self.profile.local_step_enabled)
            or not bool(local_provider_names_tuple)
        ):
            return self._with_timing_metadata(
                self._with_step_metadata(
                    full_result,
                    {
                        **local_first_metadata,
                        "local_step_enabled": bool(self.profile.local_step_enabled),
                        "local_step_status": "disabled_or_no_local_provider",
                    },
                ),
                timing_profile,
                started_at,
            )

        try:
            rate_step = limit_joint_step(
                reference_qpos=request.reference_qpos,
                target_qpos=full_result.qpos,
                profile=joint_velocity_profile,
                dt_sec=dt_sec,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics should not mask a valid IK target.
            return self._with_timing_metadata(
                self._with_step_metadata(
                    full_result,
                    {
                        "local_step_enabled": True,
                        "local_step_status": "rate_limit_metadata_unavailable",
                        "local_step_error": f"{type(exc).__name__}: {exc}",
                    },
                ),
                timing_profile,
                started_at,
            )

        requested_delta_deg = np.rad2deg(full_result.qpos - np.asarray(request.reference_qpos, dtype=float).reshape(-1))
        rate_metadata = {
            **local_first_metadata,
            "local_step_enabled": True,
            "local_step_trigger_ratio": float(self.profile.local_step_trigger_ratio),
            "local_step_full_target_limited": bool(rate_step.limited),
            "local_step_full_target_ratio": float(rate_step.limit_ratio),
            "local_step_full_target_requested_delta_deg": requested_delta_deg.tolist(),
            "local_step_full_target_max_step_deg": rate_step.max_step_deg.tolist(),
            "local_step_full_target_joint_deg": np.rad2deg(full_result.qpos).astype(float).tolist(),
            "local_step_full_target_status": full_result.status.value,
            "local_step_full_target_source": full_result.candidate_source,
            "local_step_provider_names": [str(name) for name in local_provider_names_tuple],
        }
        if (not rate_step.limited) or float(rate_step.limit_ratio) < float(self.profile.local_step_trigger_ratio):
            return self._with_timing_metadata(
                self._with_step_metadata(full_result, {**rate_metadata, "local_step_status": "not_triggered"}),
                timing_profile,
                started_at,
            )

        local_result = self._resolve_local_step(
            request=request,
            full_result=full_result,
            joint_velocity_profile=joint_velocity_profile,
            dt_sec=dt_sec,
            local_provider_names=local_provider_names_tuple,
            local_options=local_options_dict,
            frame_deadline=frame_deadline,
            rate_metadata=rate_metadata,
            timing_profile=timing_profile,
            timing_phase="local_recovery",
        )
        if _hard_deadline_expired(hard_deadline):
            selected = full_result if local_result is None else local_result
            return self._with_timing_metadata(
                self._with_step_metadata(
                    selected,
                    _hard_deadline_metadata(
                        request,
                        deadline_monotonic=hard_deadline,
                        expired=True,
                        phase="local_recovery",
                    ),
                ),
                timing_profile,
                started_at,
            )
        if local_result is not None:
            return self._with_timing_metadata(local_result, timing_profile, started_at)
        if local_fallback_result is not None:
            return self._with_timing_metadata(
                self._with_step_metadata(
                    local_fallback_result,
                    {
                        "local_step_full_solve_fallback_used": True,
                        "local_step_full_solve_status": full_result.status.value,
                        "local_step_full_solve_message": full_result.message,
                        "local_step_recovery_failure_fallback_used": True,
                        "full_target_result_message": full_result.message,
                    },
                ),
                timing_profile,
                started_at,
            )
        if bool(self.profile.local_step_hold_on_failure):
            return self._with_timing_metadata(
                self._hold_local_step_result(
                    request=request,
                    full_result=full_result,
                    rate_metadata=rate_metadata,
                    message="ik_local_step_no_progress_hold_reference",
                ),
                timing_profile,
                started_at,
            )
        return self._with_timing_metadata(
            self._with_step_metadata(
                full_result,
                {**rate_metadata, "local_step_status": "failed_falling_back_to_full_target"},
            ),
            timing_profile,
            started_at,
        )

    def _hard_deadline_hold_result(
        self,
        *,
        request: IKResolveRequest,
        phase: str,
    ) -> IKResolveResult:
        """Return the last known-safe qpos when no candidate was ready in time."""

        reference_qpos = np.asarray(request.reference_qpos, dtype=float).reshape(-1)
        previous_safe_qpos = (
            None
            if request.previous_safe_qpos is None
            else np.asarray(request.previous_safe_qpos, dtype=float).reshape(-1)
        )
        use_previous_safe = bool(
            previous_safe_qpos is not None
            and previous_safe_qpos.shape == reference_qpos.shape
            and np.all(np.isfinite(previous_safe_qpos))
        )
        hold_qpos = previous_safe_qpos if use_previous_safe else reference_qpos
        reference_pose = (
            np.asarray(request.reference_pose, dtype=float).reshape(4, 4)
            if request.reference_pose is not None
            else np.asarray(request.target_pose, dtype=float).reshape(4, 4)
        )
        return IKResolveResult(
            status=IKResolveStatus.RELAXED,
            qpos=np.asarray(hold_qpos, dtype=float).copy(),
            requested_target_pose=np.asarray(request.target_pose, dtype=float).reshape(4, 4),
            effective_target_pose=reference_pose.copy(),
            tolerance_fraction=1.0,
            candidate_source="hard_deadline_hold_reference",
            message="ik_hard_deadline_hold_reference",
            metadata={
                "profile": self.profile.name,
                "status": IKResolveStatus.RELAXED.value,
                "candidate_source": "hard_deadline_hold_reference",
                "seed_source": "previous_safe" if use_previous_safe else "reference",
                "method": "hold_previous_safe" if use_previous_safe else "hold_reference",
                "hard_deadline_hold_qpos_source": "previous_safe" if use_previous_safe else "reference",
                "local_step_status": "hard_deadline_hold_reference",
                **_hard_deadline_metadata(
                    request,
                    deadline_monotonic=_hard_deadline_monotonic(request),
                    expired=True,
                    phase=phase,
                ),
            },
        )

    def _seed_tuple(
        self,
        request: IKResolveRequest,
        reference_qpos: np.ndarray,
        *,
        enable_expansion: bool = True,
    ) -> tuple[tuple[np.ndarray, ...], tuple[str, ...], dict[str, Any]]:
        raw_seeds = [reference_qpos]
        raw_labels = ["reference"]
        if request.previous_safe_qpos is not None:
            raw_seeds.append(np.asarray(request.previous_safe_qpos, dtype=float).reshape(-1))
            raw_labels.append("previous_safe")
        for index, seed in enumerate(request.seed_qpos):
            raw_seeds.append(np.asarray(seed, dtype=float).reshape(-1))
            if index < len(request.seed_labels):
                raw_labels.append(str(request.seed_labels[index]))
            else:
                raw_labels.append(f"seed_{index}")

        seeds: list[np.ndarray] = []
        labels: list[str] = []
        for seed, label in zip(raw_seeds, raw_labels):
            if seed.shape != reference_qpos.shape or not np.all(np.isfinite(seed)):
                continue
            if any(np.allclose(seed, existing, atol=1e-9) for existing in seeds):
                continue
            seeds.append(seed.copy())
            labels.append(label)
        expanded = expand_seed_qpos(
            seeds=tuple(seeds),
            labels=tuple(labels),
            profile=self.seed_expansion_profile if bool(enable_expansion) else None,
        )
        return expanded.seeds, expanded.labels, expanded.metadata

    def _evaluate_candidate(
        self,
        *,
        candidate: IKCandidate,
        reference_qpos: np.ndarray,
        effective_target: np.ndarray,
        reference_pose: np.ndarray | None,
        request_metadata: dict[str, Any],
        timing_profile: dict[str, Any] | None = None,
        timing_phase: str = "full",
    ) -> tuple[bool, dict[str, Any]]:
        evaluate_started = time.perf_counter()
        qpos = np.asarray(candidate.qpos, dtype=float).reshape(-1)
        metadata: dict[str, Any] = {
            "candidate_source": candidate.source,
            "seed_source": candidate.seed_source,
            "method": candidate.method,
        }
        if qpos.shape != reference_qpos.shape or not np.all(np.isfinite(qpos)):
            metadata["reject_reason"] = "candidate_qpos_invalid"
            metadata["score"] = 1e18
            _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
            return False, metadata

        joint_delta_deg = np.rad2deg(np.abs(qpos - reference_qpos))
        metadata["joint_delta_deg"] = joint_delta_deg.tolist()
        metadata["candidate_joint_deg"] = np.rad2deg(qpos).astype(float).tolist()
        joint_delta_max_deg = float(np.max(joint_delta_deg)) if joint_delta_deg.size else 0.0
        joint_delta_norm_deg = float(np.linalg.norm(joint_delta_deg))
        metadata["joint_delta_max_deg"] = joint_delta_max_deg
        metadata["joint_delta_norm_deg"] = joint_delta_norm_deg
        metadata["score_policy"] = (
            "minimize_penalized_max_joint_delta_then_norm;"
            "soft_singularity_and_pose_slack_are_penalties"
        )
        score_penalty_deg = 0.0
        posture_comparison = None
        if self.posture_preference is not None:
            posture_comparison = self.posture_preference.compare(reference_qpos, qpos)
            posture_penalty_deg = max(0.0, float(posture_comparison.candidate.score_deg))
            score_penalty_deg += posture_penalty_deg
            metadata["posture_preference"] = {
                "name": self.posture_preference.name,
                "score_penalty_deg": float(posture_penalty_deg),
                **posture_comparison.metadata(),
            }

        branch_indices = tuple(int(index) for index in self.profile.hard_branch_joint_indices)
        branch_limits = self.profile.hard_branch_limits_array()
        if branch_indices:
            branch_delta = joint_delta_deg[list(branch_indices)]
            metadata["hard_branch_joint_delta_deg"] = branch_delta.tolist()
            if np.any(branch_delta > branch_limits):
                exceeded = tuple(
                    int(branch_indices[index])
                    for index, exceeded in enumerate((branch_delta > branch_limits).tolist())
                    if bool(exceeded)
                )
                escape_metadata = (
                    None
                    if self.posture_preference is None
                    else self.posture_preference.hard_branch_escape_metadata(
                        reference_qpos=reference_qpos,
                        candidate_qpos=qpos,
                        exceeded_joint_indices=exceeded,
                    )
                )
                if escape_metadata is not None:
                    metadata["posture_hard_branch_escape"] = escape_metadata
                if escape_metadata is None or not bool(escape_metadata.get("allowed", False)):
                    metadata["reject_reason"] = "hard_branch_jump"
                    metadata["score"] = _candidate_score(
                        joint_delta_max_deg=joint_delta_max_deg,
                        joint_delta_norm_deg=joint_delta_norm_deg,
                        penalty_deg=score_penalty_deg,
                        pose_fraction=0.0,
                    )
                    metadata["hard_branch_max_delta_deg"] = branch_limits.tolist()
                    _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
                    return False, metadata
                metadata["hard_branch_escape_allowed"] = True
                metadata["hard_branch_escape_reason"] = "posture_bad_to_good"
                metadata["hard_branch_max_delta_deg"] = branch_limits.tolist()

        if self.profile.per_step_joint_limit_deg is not None:
            limits = np.asarray(self.profile.per_step_joint_limit_deg, dtype=float).reshape(-1)
            if limits.shape == qpos.shape and np.any(joint_delta_deg > limits):
                metadata["reject_reason"] = "per_step_joint_jump"
                metadata["score"] = _candidate_score(
                    joint_delta_max_deg=joint_delta_max_deg,
                    joint_delta_norm_deg=joint_delta_norm_deg,
                    penalty_deg=0.0,
                    pose_fraction=0.0,
                )
                metadata["per_step_joint_limit_deg"] = limits.tolist()
                _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
                return False, metadata

        singularity_started = time.perf_counter()
        singularity = self.provider.analyze_singularity(qpos)
        singularity_elapsed_ms = _elapsed_ms(singularity_started)
        _timing_add(timing_profile, "singularity_ms", singularity_elapsed_ms)
        _timing_add(timing_profile, f"{timing_phase}_singularity_ms", singularity_elapsed_ms)
        metadata["singularity"] = _singularity_metadata(singularity)
        if self.profile.reject_singular and singularity.is_singular:
            soft_labels = {str(label) for label in self.profile.soft_singularity_labels}
            if singularity.label in soft_labels:
                score_penalty_deg += max(0.0, float(self.profile.soft_singularity_penalty_deg))
                metadata["soft_singularity_candidate"] = True
                metadata["soft_singularity_penalty_deg"] = float(
                    max(0.0, float(self.profile.soft_singularity_penalty_deg))
                )
            else:
                singular_escape_metadata = (
                    None
                    if self.posture_preference is None
                    else self.posture_preference.singular_escape_metadata(
                        reference_qpos=reference_qpos,
                        candidate_qpos=qpos,
                        singularity_label=str(singularity.label),
                    )
                )
                if singular_escape_metadata is not None:
                    metadata["posture_singular_escape"] = singular_escape_metadata
                if singular_escape_metadata is not None and bool(singular_escape_metadata.get("allowed", False)):
                    metadata["singular_escape_allowed"] = True
                    metadata["singular_escape_reason"] = "posture_bad_to_good_crossing"
                    score_penalty_deg += max(0.0, float(self.profile.soft_singularity_penalty_deg))
                else:
                    metadata["reject_reason"] = "singular_candidate"
                    metadata["score"] = _candidate_score(
                        joint_delta_max_deg=joint_delta_max_deg,
                        joint_delta_norm_deg=joint_delta_norm_deg,
                        penalty_deg=score_penalty_deg,
                        pose_fraction=0.0,
                    )
                    _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
                    return False, metadata

        pose_fraction = 0.0
        fk_started = time.perf_counter()
        fk_pose = self.provider.forward_kinematics(qpos)
        fk_elapsed_ms = _elapsed_ms(fk_started)
        _timing_add(timing_profile, "fk_ms", fk_elapsed_ms)
        _timing_add(timing_profile, f"{timing_phase}_fk_ms", fk_elapsed_ms)
        if fk_pose is not None:
            pose_error_m, pose_error_deg = pose_delta_metrics(fk_pose, effective_target)
            metadata["pose_error_m"] = pose_error_m
            metadata["pose_error_deg"] = pose_error_deg
            pose_fraction = _pose_tolerance_fraction(
                pose_error_m=pose_error_m,
                pose_error_deg=pose_error_deg,
                position_tolerance_m=float(self.profile.target_tolerance.position_m),
                orientation_tolerance_deg=float(self.profile.target_tolerance.orientation_deg),
            )
            pose_slack_tier = _pose_slack_tier(pose_fraction)
            metadata["pose_tolerance_fraction"] = pose_fraction
            metadata["pose_slack_tier"] = pose_slack_tier
            if pose_slack_tier > 0.0:
                penalty = pose_slack_tier * max(0.0, float(self.profile.pose_slack_penalty_deg))
                score_penalty_deg += penalty
                metadata["pose_slack_penalty_deg"] = penalty
            if (
                pose_error_m
                > max(float(self.profile.fk_error_tolerance_m), float(self.profile.target_tolerance.position_m))
                or pose_error_deg
                > max(float(self.profile.fk_error_tolerance_deg), float(self.profile.target_tolerance.orientation_deg))
            ):
                coarse_metadata = _coarse_tracking_acceptance(
                    profile=self.profile,
                    reference_pose=reference_pose,
                    target_pose=effective_target,
                    pose_error_m=pose_error_m,
                    pose_error_deg=pose_error_deg,
                )
                metadata["coarse_tracking"] = coarse_metadata
                if not bool(coarse_metadata.get("accepted", False)):
                    metadata["reject_reason"] = "fk_pose_error"
                    metadata["score"] = _candidate_score(
                        joint_delta_max_deg=joint_delta_max_deg,
                        joint_delta_norm_deg=joint_delta_norm_deg,
                        penalty_deg=score_penalty_deg,
                        pose_fraction=pose_fraction,
                    )
                    _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
                    return False, metadata
                metadata["coarse_tracking_accepted"] = True
                metadata["coarse_tracking_accept_reason"] = coarse_metadata.get("accept_reason")
                metadata["coarse_tracking_pose_fraction"] = float(coarse_metadata.get("after_fraction", pose_fraction))
                score_penalty_deg += max(0.0, float(pose_fraction) - 1.0) * max(
                    0.0,
                    float(self.profile.pose_slack_penalty_deg),
                )

        if self.collision_checker is not None:
            collision_started = time.perf_counter()
            ok, collision_metadata = self.collision_checker.check_candidate(
                qpos,
                metadata={**request_metadata, **metadata},
            )
            collision_elapsed_ms = _elapsed_ms(collision_started)
            _timing_add(timing_profile, "collision_ms", collision_elapsed_ms)
            _timing_add(timing_profile, f"{timing_phase}_collision_ms", collision_elapsed_ms)
            metadata["collision"] = dict(collision_metadata)
            if not ok:
                metadata["reject_reason"] = "collision"
                metadata["score"] = _candidate_score(
                    joint_delta_max_deg=joint_delta_max_deg,
                    joint_delta_norm_deg=joint_delta_norm_deg,
                    penalty_deg=score_penalty_deg,
                    pose_fraction=pose_fraction,
                )
                _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
                return False, metadata

        metadata["score_penalty_deg"] = float(score_penalty_deg)
        metadata["penalized_joint_delta_max_deg"] = float(joint_delta_max_deg + score_penalty_deg)
        metadata["score"] = _candidate_score(
            joint_delta_max_deg=joint_delta_max_deg,
            joint_delta_norm_deg=joint_delta_norm_deg,
            penalty_deg=score_penalty_deg,
            pose_fraction=pose_fraction,
        )
        metadata["reject_reason"] = None
        _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
        return True, metadata

    def _resolve_local_step(
        self,
        *,
        request: IKResolveRequest,
        full_result: IKResolveResult | None,
        joint_velocity_profile: JointVelocityLimitProfile,
        dt_sec: float,
        local_provider_names: tuple[str, ...],
        local_options: dict[str, Any],
        frame_deadline: float | None,
        rate_metadata: dict[str, Any],
        timing_profile: dict[str, Any] | None = None,
        timing_phase: str = "local_step",
    ) -> IKResolveResult | None:
        target_pose = np.asarray(request.target_pose, dtype=float).reshape(4, 4)
        reference_qpos = np.asarray(request.reference_qpos, dtype=float).reshape(-1)
        hard_deadline = _hard_deadline_monotonic(request)
        hard_deadline_expired = False
        if request.reference_pose is not None:
            reference_pose = np.asarray(request.reference_pose, dtype=float).reshape(4, 4)
        else:
            reference_fk_started = time.perf_counter()
            reference_pose = self.provider.forward_kinematics(reference_qpos)
            reference_fk_elapsed_ms = _elapsed_ms(reference_fk_started)
            _timing_add(timing_profile, "reference_fk_ms", reference_fk_elapsed_ms)
            _timing_add(timing_profile, f"{timing_phase}_reference_fk_ms", reference_fk_elapsed_ms)
        if reference_pose is None:
            return None
        enable_local_seed_expansion = bool(local_options.pop("local_step_enable_seed_expansion", False))
        seed_started = time.perf_counter()
        seeds, seed_labels, seed_expansion_metadata = self._seed_tuple(
            request,
            reference_qpos,
            enable_expansion=enable_local_seed_expansion,
        )
        seed_elapsed_ms = _elapsed_ms(seed_started)
        _timing_add(timing_profile, "seed_ms", seed_elapsed_ms)
        _timing_add(timing_profile, f"{timing_phase}_seed_ms", seed_elapsed_ms)
        local_step_rad = np.deg2rad(joint_velocity_profile.max_step_deg(dt_sec))
        max_joint_step_rad = float(np.max(local_step_rad)) if local_step_rad.size else 0.0
        base_options = {
            **request.metadata,
            "enable_expensive_seed_expansion": False,
            "allow_parallel": bool(self.profile.allow_parallel_candidate_generation),
            "max_candidate_count": min(max(1, int(self.profile.max_candidate_count)), 8),
            "provider_time_budget_ms": self.profile.provider_time_budget_ms,
            "tolerance_fraction": 0.0,
            "local_step_mode": True,
            "provider_name_allowlist": list(local_provider_names),
            "pinocchio_max_joint_step_rad": max(1e-6, max_joint_step_rad),
            "seed_expansion": seed_expansion_metadata,
            **local_options,
        }
        if frame_deadline is not None:
            base_options["provider_deadline_monotonic"] = float(frame_deadline)
            base_options["provider_remaining_time_budget_ms"] = max(
                0.0,
                (float(frame_deadline) - time.perf_counter()) * 1000.0,
            )
        candidates: list[IKCandidate] = []
        provider_metadata_by_horizon: list[dict[str, Any]] = []
        for iteration_count in _local_step_iteration_counts(self.profile):
            if _hard_deadline_expired(hard_deadline):
                hard_deadline_expired = True
                break
            options = {
                **base_options,
                "pinocchio_max_iterations": int(iteration_count),
                "local_step_iteration_count": int(iteration_count),
            }
            provider_started = time.perf_counter()
            horizon_candidates = self.provider.iter_ik_candidates(
                target_pose=target_pose,
                seeds=seeds,
                seed_labels=seed_labels,
                options=options,
            )
            provider_elapsed_ms = _elapsed_ms(provider_started)
            provider_metadata = (
                dict(getattr(self.provider, "last_metadata", {}))
                if hasattr(self.provider, "last_metadata")
                else {}
            )
            _record_provider_call(
                timing_profile,
                phase=timing_phase,
                provider_metadata=provider_metadata,
                candidate_count=len(horizon_candidates),
                measured_elapsed_ms=provider_elapsed_ms,
                extra={"local_step_iteration_count": int(iteration_count)},
            )
            if hasattr(self.provider, "last_metadata"):
                provider_metadata_by_horizon.append(
                    {
                        "local_step_iteration_count": int(iteration_count),
                        **dict(getattr(self.provider, "last_metadata", {})),
                    }
                )
            for candidate in horizon_candidates:
                if _hard_deadline_expired(hard_deadline):
                    hard_deadline_expired = True
                    break
                candidates.append(
                    IKCandidate(
                        qpos=np.asarray(candidate.qpos, dtype=float).reshape(reference_qpos.shape),
                        source=candidate.source,
                        seed_source=candidate.seed_source,
                        method=candidate.method,
                        metadata={
                            **dict(candidate.metadata),
                            "local_step_iteration_count": int(iteration_count),
                        },
                    )
                )
            if hard_deadline_expired:
                break
        if (
            not hard_deadline_expired
            and full_result is not None
            and bool(self.profile.local_step_full_target_step_enabled)
            and full_result.qpos is not None
        ):
            candidates.append(
                IKCandidate(
                    qpos=np.asarray(full_result.qpos, dtype=float).reshape(reference_qpos.shape),
                    source="rate_limited_full_target",
                    seed_source=str(full_result.metadata.get("seed_source", "")),
                    method="rate_limited_full_target",
                    metadata={
                        "local_step_iteration_count": 0,
                        "full_target_source": full_result.candidate_source,
                        "full_target_method": str(full_result.metadata.get("method", "")),
                        "full_target_status": full_result.status.value,
                    },
                )
            )
        if hard_deadline is not None:
            return self._resolve_local_step_with_hard_deadline(
                request=request,
                full_result=full_result,
                reference_qpos=reference_qpos,
                target_pose=target_pose,
                reference_pose=reference_pose,
                joint_velocity_profile=joint_velocity_profile,
                dt_sec=dt_sec,
                base_candidates=candidates,
                seed_expansion_metadata=seed_expansion_metadata,
                provider_metadata_by_horizon=provider_metadata_by_horizon,
                rate_metadata=rate_metadata,
                hard_deadline=hard_deadline,
                already_expired=hard_deadline_expired,
                timing_profile=timing_profile,
                timing_phase=timing_phase,
            )
        if _hard_deadline_expired(hard_deadline):
            hard_deadline_expired = True
        if hard_deadline_expired:
            skipped_auxiliary = {
                "enabled": False,
                "status": "skipped_hard_deadline",
                "candidate_count": 0,
            }
            limit_escape_candidates, limit_escape_metadata = [], dict(skipped_auxiliary)
            singularity_escape_candidates, singularity_escape_metadata = [], dict(skipped_auxiliary)
            posture_escape_candidates, posture_escape_metadata = [], dict(skipped_auxiliary)
            wrist_unload_candidates, wrist_unload_metadata = [], dict(skipped_auxiliary)
        else:
            limit_escape_candidates, limit_escape_metadata = self._local_step_limit_escape_candidates(
                reference_qpos=reference_qpos,
                target_pose=target_pose,
                reference_pose=reference_pose,
                joint_velocity_profile=joint_velocity_profile,
                dt_sec=dt_sec,
                timing_profile=timing_profile,
                timing_phase=timing_phase,
            )
            candidates.extend(limit_escape_candidates)
            if _hard_deadline_expired(hard_deadline):
                hard_deadline_expired = True
                skipped_auxiliary = {
                    "enabled": False,
                    "status": "skipped_hard_deadline",
                    "candidate_count": 0,
                }
                singularity_escape_candidates, singularity_escape_metadata = [], dict(skipped_auxiliary)
                posture_escape_candidates, posture_escape_metadata = [], dict(skipped_auxiliary)
                wrist_unload_candidates, wrist_unload_metadata = [], dict(skipped_auxiliary)
            else:
                singularity_escape_candidates, singularity_escape_metadata = self._local_step_singularity_escape_candidates(
                    reference_qpos=reference_qpos,
                    target_pose=target_pose,
                    reference_pose=reference_pose,
                    joint_velocity_profile=joint_velocity_profile,
                    dt_sec=dt_sec,
                    timing_profile=timing_profile,
                    timing_phase=timing_phase,
                )
                candidates.extend(singularity_escape_candidates)
                if _hard_deadline_expired(hard_deadline):
                    hard_deadline_expired = True
                    skipped_auxiliary = {
                        "enabled": False,
                        "status": "skipped_hard_deadline",
                        "candidate_count": 0,
                    }
                    posture_escape_candidates, posture_escape_metadata = [], dict(skipped_auxiliary)
                    wrist_unload_candidates, wrist_unload_metadata = [], dict(skipped_auxiliary)
                else:
                    posture_escape_candidates, posture_escape_metadata = self._local_step_posture_escape_candidates(
                        reference_qpos=reference_qpos,
                        joint_velocity_profile=joint_velocity_profile,
                        dt_sec=dt_sec,
                    )
                    candidates.extend(posture_escape_candidates)
                    if _hard_deadline_expired(hard_deadline):
                        hard_deadline_expired = True
                        wrist_unload_candidates, wrist_unload_metadata = [], {
                            "enabled": False,
                            "status": "skipped_hard_deadline",
                            "candidate_count": 0,
                        }
                    else:
                        wrist_unload_candidates, wrist_unload_metadata = self._local_step_wrist_unload_candidates(
                            reference_qpos=reference_qpos,
                            joint_velocity_profile=joint_velocity_profile,
                            dt_sec=dt_sec,
                        )
                        candidates.extend(wrist_unload_candidates)
        evaluated: list[_EvaluatedCandidate] = []
        rejection_log: list[dict[str, Any]] = []
        step_scales = _local_step_scale_factors(self.profile)
        primary_step_scales = step_scales[:1]
        fallback_step_scales = step_scales[1:]
        limit_escape_active = str(limit_escape_metadata.get("status", "")) == "active"
        singularity_escape_active = str(singularity_escape_metadata.get("status", "")) == "active"
        posture_escape_active = str(posture_escape_metadata.get("status", "")) == "active"
        wrist_unload_active = str(wrist_unload_metadata.get("status", "")) == "active"
        scale_passes: list[dict[str, Any]] = []
        eval_count = 0
        evaluate_started = time.perf_counter()

        def evaluate_scale_pass(
            *,
            name: str,
            scales: tuple[float, ...],
            indexed_candidates: list[tuple[int, IKCandidate]],
            fallback_reason: str | None = None,
            allow_pose_nonprogress_transition: bool = False,
        ) -> None:
            nonlocal eval_count, hard_deadline_expired
            if not scales or not indexed_candidates:
                return
            pass_started = time.perf_counter()
            pass_eval_before = eval_count
            pass_accept_before = len(evaluated)
            for candidate_index, candidate in indexed_candidates:
                for step_scale in scales:
                    if _hard_deadline_expired(hard_deadline):
                        hard_deadline_expired = True
                        break
                    eval_count += 1
                    accepted, metadata = self._evaluate_local_step_candidate(
                        candidate=candidate,
                        reference_qpos=reference_qpos,
                        target_pose=target_pose,
                        reference_pose=reference_pose,
                        request_metadata=request.metadata,
                        joint_velocity_profile=joint_velocity_profile,
                        dt_sec=dt_sec,
                        step_scale=step_scale,
                        allow_pose_nonprogress_transition=allow_pose_nonprogress_transition,
                        timing_profile=timing_profile,
                        timing_phase=timing_phase,
                    )
                    metadata["local_step_scale_pass"] = name
                    metadata["local_step_candidate_index"] = int(candidate_index)
                    if fallback_reason:
                        metadata["local_step_scale_fallback_reason"] = fallback_reason
                    if accepted:
                        evaluated.append(
                            _EvaluatedCandidate(
                                candidate=IKCandidate(
                                    qpos=np.asarray(metadata["local_step_output_qpos"], dtype=float).reshape(reference_qpos.shape),
                                    source=candidate.source,
                                    seed_source=candidate.seed_source,
                                    method=candidate.method,
                                    metadata=dict(candidate.metadata),
                                ),
                                pose_error_m=metadata.get("pose_error_m"),
                                pose_error_deg=metadata.get("pose_error_deg"),
                                score=float(metadata.get("score", 0.0)),
                                metadata=metadata,
                            )
                        )
                    else:
                        rejection_log.append(metadata)
                if hard_deadline_expired:
                    break
            pass_elapsed_ms = _elapsed_ms(pass_started)
            _timing_add(timing_profile, f"{timing_phase}_{name}_evaluate_ms", pass_elapsed_ms)
            _timing_inc(timing_profile, f"{timing_phase}_{name}_candidate_eval_count", eval_count - pass_eval_before)
            _timing_inc(timing_profile, f"{timing_phase}_{name}_accepted_candidate_count", len(evaluated) - pass_accept_before)
            scale_passes.append(
                {
                    "name": name,
                    "scales": [float(scale) for scale in scales],
                    "candidate_count": len(indexed_candidates),
                    "eval_count": eval_count - pass_eval_before,
                    "accepted_count": len(evaluated) - pass_accept_before,
                    **({"fallback_reason": fallback_reason} if fallback_reason else {}),
                }
            )

        indexed_candidates = list(enumerate(candidates))
        evaluate_scale_pass(
            name="primary",
            scales=primary_step_scales,
            indexed_candidates=indexed_candidates,
        )
        fallback_reason = None
        fallback_candidates = indexed_candidates
        if not hard_deadline_expired and fallback_step_scales:
            if not evaluated:
                fallback_reason = "primary_failed"
            elif limit_escape_active:
                fallback_reason = "joint_limit_escape_active"
            elif singularity_escape_active:
                fallback_reason = "singularity_escape_active"
                fallback_candidates = [
                    (index, candidate)
                    for index, candidate in indexed_candidates
                    if bool(candidate.metadata.get("local_step_singularity_escape_candidate", False))
                ]
                if not fallback_candidates:
                    fallback_candidates = indexed_candidates
            elif posture_escape_active:
                fallback_reason = "posture_escape_active"
                fallback_candidates = [
                    (index, candidate)
                    for index, candidate in indexed_candidates
                    if bool(candidate.metadata.get("local_step_posture_escape_candidate", False))
                ]
                if not fallback_candidates:
                    fallback_candidates = indexed_candidates
            elif wrist_unload_active:
                fallback_reason = "wrist_unload_active"
                fallback_candidates = [
                    (index, candidate)
                    for index, candidate in indexed_candidates
                    if bool(candidate.metadata.get("local_step_wrist_unload_candidate", False))
                ]
                if not fallback_candidates:
                    fallback_candidates = indexed_candidates
            if fallback_reason == "joint_limit_escape_active":
                selected_primary = min(evaluated, key=lambda item: item.score)
                selected_primary_index = int(selected_primary.metadata.get("local_step_candidate_index", -1))
                accepted_candidate_indices = (
                    {selected_primary_index}
                    if selected_primary_index >= 0
                    else set()
                )
                fallback_candidates = [
                    (index, candidate)
                    for index, candidate in indexed_candidates
                    if int(index) in accepted_candidate_indices
                ]
        if not hard_deadline_expired and fallback_reason and fallback_candidates:
            for fallback_scale in fallback_step_scales:
                fallback_accept_before = len(evaluated)
                evaluate_scale_pass(
                    name="fallback",
                    scales=(fallback_scale,),
                    indexed_candidates=fallback_candidates,
                    fallback_reason=fallback_reason,
                )
                if len(evaluated) > fallback_accept_before:
                    break
        if (
            not hard_deadline_expired
            and not evaluated
            and full_result is not None
            and full_result.accepted
            and full_result.qpos is not None
        ):
            transition_candidates = [
                (index, candidate)
                for index, candidate in indexed_candidates
                if candidate.source == "rate_limited_full_target"
            ]
            evaluate_scale_pass(
                name="transition",
                scales=(1.0,),
                indexed_candidates=transition_candidates,
                fallback_reason="no_pose_progress_but_full_target_is_safe",
                allow_pose_nonprogress_transition=True,
            )
        evaluate_elapsed_ms = _elapsed_ms(evaluate_started)
        _timing_add(timing_profile, "local_evaluate_ms", evaluate_elapsed_ms)
        _timing_add(timing_profile, f"{timing_phase}_evaluate_ms", evaluate_elapsed_ms)
        _timing_inc(timing_profile, "local_candidate_count", len(candidates))
        _timing_inc(timing_profile, f"{timing_phase}_candidate_count", len(candidates))
        _timing_inc(timing_profile, "local_candidate_eval_count", eval_count)
        _timing_inc(timing_profile, "local_accepted_candidate_count", len(evaluated))
        _timing_inc(timing_profile, f"{timing_phase}_accepted_candidate_count", len(evaluated))
        if hard_deadline is not None and timing_profile is not None:
            timing_profile["hard_deadline_enabled"] = True
            timing_profile["hard_deadline_expired"] = bool(hard_deadline_expired)
        if not evaluated:
            rate_metadata.update(
                _hard_deadline_metadata(
                    request,
                    deadline_monotonic=hard_deadline,
                    expired=hard_deadline_expired,
                    phase=timing_phase,
                )
            )
            rate_metadata["local_step_seed_expansion"] = seed_expansion_metadata
            rate_metadata["local_step_limit_escape"] = limit_escape_metadata
            rate_metadata["local_step_singularity_escape"] = singularity_escape_metadata
            rate_metadata["local_step_posture_escape"] = posture_escape_metadata
            rate_metadata["local_step_wrist_unload"] = wrist_unload_metadata
            rate_metadata["local_step_scale_strategy"] = {
                "configured_scales": [float(scale) for scale in step_scales],
                "primary_scales": [float(scale) for scale in primary_step_scales],
                "fallback_scales": [float(scale) for scale in fallback_step_scales],
                "fallback_used": any(pass_info.get("name") == "fallback" for pass_info in scale_passes),
                "fallback_reason": fallback_reason,
                "limit_escape_active": bool(limit_escape_active),
                "singularity_escape_active": bool(singularity_escape_active),
                "posture_escape_active": bool(posture_escape_active),
                "wrist_unload_active": bool(wrist_unload_active),
                "passes": list(scale_passes),
            }
            if provider_metadata_by_horizon:
                rate_metadata["local_step_provider_horizons"] = provider_metadata_by_horizon
            elif hasattr(self.provider, "last_metadata"):
                rate_metadata["local_step_provider"] = dict(getattr(self.provider, "last_metadata", {}))
            rate_metadata["local_step_rejection_count"] = len(rejection_log)
            rate_metadata["local_step_last_rejections"] = rejection_log[-8:]
            return None
        selection_pool, selection_policy = _local_step_stable_selection_pool(evaluated)
        selected = min(selection_pool, key=lambda item: item.score)
        full_metadata = dict(full_result.metadata) if full_result is not None else {}
        metadata = {
            **full_metadata,
            **rate_metadata,
            **selected.metadata,
            "status": IKResolveStatus.RELAXED.value,
            "source": "generic_ik_resolver",
            "result_message": "ik_local_step_first" if full_result is None else "ik_local_step_recovery",
            "local_step_status": "active",
            "local_step_recovery": True,
            "local_step_seed_expansion": seed_expansion_metadata,
            "local_step_limit_escape": limit_escape_metadata,
            "local_step_singularity_escape": singularity_escape_metadata,
            "local_step_posture_escape": posture_escape_metadata,
            "local_step_wrist_unload": wrist_unload_metadata,
            "local_step_scale_strategy": {
                "configured_scales": [float(scale) for scale in step_scales],
                "primary_scales": [float(scale) for scale in primary_step_scales],
                "fallback_scales": [float(scale) for scale in fallback_step_scales],
                "fallback_used": any(pass_info.get("name") == "fallback" for pass_info in scale_passes),
                "fallback_reason": fallback_reason,
                "limit_escape_active": bool(limit_escape_active),
                "singularity_escape_active": bool(singularity_escape_active),
                "posture_escape_active": bool(posture_escape_active),
                "wrist_unload_active": bool(wrist_unload_active),
                "passes": list(scale_passes),
            },
            "local_step_selection_policy": selection_policy,
            "local_step_rejection_count": len(rejection_log),
            "local_step_first_status": "accepted" if full_result is None else rate_metadata.get("local_step_first_status"),
            **_hard_deadline_metadata(
                request,
                deadline_monotonic=hard_deadline,
                expired=hard_deadline_expired,
                phase=timing_phase,
            ),
        }
        metadata["local_step_recovery_state"] = _local_step_recovery_state(selected.metadata)
        if full_result is not None:
            metadata["full_target_result_message"] = full_result.message
        if provider_metadata_by_horizon:
            metadata["local_step_provider_horizons"] = provider_metadata_by_horizon
        elif hasattr(self.provider, "last_metadata"):
            metadata["local_step_provider"] = dict(getattr(self.provider, "last_metadata", {}))
        return IKResolveResult(
            status=IKResolveStatus.RELAXED,
            qpos=selected.candidate.qpos.copy(),
            requested_target_pose=target_pose,
            effective_target_pose=np.asarray(selected.metadata.get("local_step_output_pose"), dtype=float).reshape(4, 4),
            tolerance_fraction=max(
                0.0 if full_result is None else float(full_result.tolerance_fraction),
                float(selected.metadata.get("pose_slack_tier", 1.0)),
            ),
            pose_error_m=selected.pose_error_m,
            pose_error_deg=selected.pose_error_deg,
            candidate_source=selected.candidate.source,
            message="ik_local_step_first" if full_result is None else "ik_local_step_recovery",
            metadata=metadata,
        )

    def _resolve_local_step_with_hard_deadline(
        self,
        *,
        request: IKResolveRequest,
        full_result: IKResolveResult | None,
        reference_qpos: np.ndarray,
        target_pose: np.ndarray,
        reference_pose: np.ndarray,
        joint_velocity_profile: JointVelocityLimitProfile,
        dt_sec: float,
        base_candidates: list[IKCandidate],
        seed_expansion_metadata: dict[str, Any],
        provider_metadata_by_horizon: list[dict[str, Any]],
        rate_metadata: dict[str, Any],
        hard_deadline: float,
        already_expired: bool,
        timing_profile: dict[str, Any] | None,
        timing_phase: str,
    ) -> IKResolveResult | None:
        """Evaluate base candidates before spending deadline on recoveries.

        The regular local-step path collects all recovery branches before it
        evaluates any candidate.  That is useful for offline-quality search,
        but in a force-control frame it can spend the whole period generating
        auxiliary candidates and lose an already-safe local result.  This
        deadline-specific path is deliberately staged: base candidates first,
        then one recovery family at a time only if no safe result exists.
        """

        deadline_expired = bool(already_expired or _hard_deadline_expired(hard_deadline))
        evaluated: list[_EvaluatedCandidate] = []
        rejection_log: list[dict[str, Any]] = []
        stage_records: list[dict[str, Any]] = []
        step_scales = _local_step_scale_factors(self.profile)
        primary_scales = step_scales[:1]
        fallback_scales = step_scales[1:]
        rate_metadata = dict(rate_metadata)

        def skipped_auxiliary(status: str) -> dict[str, Any]:
            return {
                "enabled": False,
                "status": status,
                "candidate_count": 0,
            }

        limit_escape_metadata: dict[str, Any] = skipped_auxiliary("not_attempted")
        singularity_escape_metadata: dict[str, Any] = skipped_auxiliary("not_attempted")
        posture_escape_metadata: dict[str, Any] = skipped_auxiliary("not_attempted")
        wrist_unload_metadata: dict[str, Any] = skipped_auxiliary("not_attempted")

        def evaluate_stage(name: str, candidates: list[IKCandidate]) -> None:
            nonlocal deadline_expired
            if deadline_expired:
                return
            stage_started = time.perf_counter()
            stage_evaluations = 0
            stage_accepted_before = len(evaluated)
            for pass_name, scales in (("primary", primary_scales), ("fallback", fallback_scales)):
                if deadline_expired or not scales:
                    break
                if pass_name == "fallback" and len(evaluated) > stage_accepted_before:
                    break
                for candidate in candidates:
                    for step_scale in scales:
                        if _hard_deadline_expired(hard_deadline):
                            deadline_expired = True
                            break
                        stage_evaluations += 1
                        accepted, metadata = self._evaluate_local_step_candidate(
                            candidate=candidate,
                            reference_qpos=reference_qpos,
                            target_pose=target_pose,
                            reference_pose=reference_pose,
                            request_metadata=request.metadata,
                            joint_velocity_profile=joint_velocity_profile,
                            dt_sec=dt_sec,
                            step_scale=step_scale,
                            allow_pose_nonprogress_transition=(
                                candidate.source == "rate_limited_full_target"
                            ),
                            timing_profile=timing_profile,
                            timing_phase=timing_phase,
                        )
                        metadata["local_step_scale_pass"] = f"hard_deadline_{pass_name}"
                        if _hard_deadline_expired(hard_deadline):
                            # A candidate that only completed after the cap was
                            # crossed was not known-safe at the deadline.
                            deadline_expired = True
                            break
                        if accepted:
                            evaluated.append(
                                _EvaluatedCandidate(
                                    candidate=IKCandidate(
                                        qpos=np.asarray(
                                            metadata["local_step_output_qpos"],
                                            dtype=float,
                                        ).reshape(reference_qpos.shape),
                                        source=candidate.source,
                                        seed_source=candidate.seed_source,
                                        method=candidate.method,
                                        metadata=dict(candidate.metadata),
                                    ),
                                    pose_error_m=metadata.get("pose_error_m"),
                                    pose_error_deg=metadata.get("pose_error_deg"),
                                    score=float(metadata.get("score", 0.0)),
                                    metadata=metadata,
                                )
                            )
                        else:
                            rejection_log.append(metadata)
                    if deadline_expired:
                        break
                if deadline_expired:
                    break
            stage_records.append(
                {
                    "name": name,
                    "candidate_count": len(candidates),
                    "evaluation_count": stage_evaluations,
                    "accepted_count": len(evaluated) - stage_accepted_before,
                    "elapsed_ms": _elapsed_ms(stage_started),
                    "deadline_expired": bool(deadline_expired),
                }
            )

        def selected_result() -> IKResolveResult | None:
            if not evaluated:
                return None
            selection_pool, selection_policy = _local_step_stable_selection_pool(evaluated)
            selected = min(selection_pool, key=lambda item: item.score)
            full_metadata = dict(full_result.metadata) if full_result is not None else {}
            metadata = {
                **full_metadata,
                **rate_metadata,
                **selected.metadata,
                "status": IKResolveStatus.RELAXED.value,
                "source": "generic_ik_resolver",
                "result_message": "ik_local_step_hard_deadline",
                "local_step_status": "active",
                "local_step_recovery": True,
                "local_step_seed_expansion": seed_expansion_metadata,
                "local_step_limit_escape": limit_escape_metadata,
                "local_step_singularity_escape": singularity_escape_metadata,
                "local_step_posture_escape": posture_escape_metadata,
                "local_step_wrist_unload": wrist_unload_metadata,
                "local_step_scale_strategy": {
                    "hard_deadline_staged": True,
                    "configured_scales": [float(scale) for scale in step_scales],
                    "primary_scales": [float(scale) for scale in primary_scales],
                    "fallback_scales": [float(scale) for scale in fallback_scales],
                    "stages": list(stage_records),
                },
                "local_step_selection_policy": selection_policy,
                "local_step_rejection_count": len(rejection_log),
                "local_step_last_rejections": rejection_log[-8:],
                **_hard_deadline_metadata(
                    request,
                    deadline_monotonic=hard_deadline,
                    expired=deadline_expired,
                    phase=timing_phase,
                ),
            }
            if provider_metadata_by_horizon:
                metadata["local_step_provider_horizons"] = list(provider_metadata_by_horizon)
            elif hasattr(self.provider, "last_metadata"):
                metadata["local_step_provider"] = dict(getattr(self.provider, "last_metadata", {}))
            metadata["local_step_recovery_state"] = _local_step_recovery_state(selected.metadata)
            return IKResolveResult(
                status=IKResolveStatus.RELAXED,
                qpos=selected.candidate.qpos.copy(),
                requested_target_pose=target_pose,
                effective_target_pose=np.asarray(
                    selected.metadata.get("local_step_output_pose"),
                    dtype=float,
                ).reshape(4, 4),
                tolerance_fraction=max(
                    0.0 if full_result is None else float(full_result.tolerance_fraction),
                    float(selected.metadata.get("pose_slack_tier", 1.0)),
                ),
                pose_error_m=selected.pose_error_m,
                pose_error_deg=selected.pose_error_deg,
                candidate_source=selected.candidate.source,
                message="ik_local_step_hard_deadline",
                metadata=metadata,
            )

        # Base candidates are already generated by the local provider.  Evaluate
        # them before any escape expansion so a deadline never discards a safe
        # candidate that was available in the current frame.
        evaluate_stage("base", list(base_candidates))
        result = selected_result()
        if result is not None:
            return result
        if deadline_expired:
            return None

        limit_candidates, limit_escape_metadata = self._local_step_limit_escape_candidates(
            reference_qpos=reference_qpos,
            target_pose=target_pose,
            reference_pose=reference_pose,
            joint_velocity_profile=joint_velocity_profile,
            dt_sec=dt_sec,
            deadline_monotonic=hard_deadline,
            timing_profile=timing_profile,
            timing_phase=timing_phase,
        )
        if _hard_deadline_expired(hard_deadline):
            deadline_expired = True
            singularity_escape_metadata = skipped_auxiliary("skipped_hard_deadline")
            posture_escape_metadata = skipped_auxiliary("skipped_hard_deadline")
            wrist_unload_metadata = skipped_auxiliary("skipped_hard_deadline")
            return selected_result()
        evaluate_stage("joint_limit_escape", limit_candidates)
        result = selected_result()
        if result is not None:
            return result
        if deadline_expired:
            return None

        singularity_candidates, singularity_escape_metadata = self._local_step_singularity_escape_candidates(
            reference_qpos=reference_qpos,
            target_pose=target_pose,
            reference_pose=reference_pose,
            joint_velocity_profile=joint_velocity_profile,
            dt_sec=dt_sec,
            deadline_monotonic=hard_deadline,
            timing_profile=timing_profile,
            timing_phase=timing_phase,
        )
        if _hard_deadline_expired(hard_deadline):
            deadline_expired = True
            posture_escape_metadata = skipped_auxiliary("skipped_hard_deadline")
            wrist_unload_metadata = skipped_auxiliary("skipped_hard_deadline")
            return selected_result()
        evaluate_stage("singularity_escape", singularity_candidates)
        result = selected_result()
        if result is not None:
            return result
        if deadline_expired:
            return None

        posture_candidates, posture_escape_metadata = self._local_step_posture_escape_candidates(
            reference_qpos=reference_qpos,
            joint_velocity_profile=joint_velocity_profile,
            dt_sec=dt_sec,
            deadline_monotonic=hard_deadline,
        )
        if _hard_deadline_expired(hard_deadline):
            deadline_expired = True
            wrist_unload_metadata = skipped_auxiliary("skipped_hard_deadline")
            return selected_result()
        evaluate_stage("posture_escape", posture_candidates)
        result = selected_result()
        if result is not None:
            return result
        if deadline_expired:
            return None

        wrist_candidates, wrist_unload_metadata = self._local_step_wrist_unload_candidates(
            reference_qpos=reference_qpos,
            joint_velocity_profile=joint_velocity_profile,
            dt_sec=dt_sec,
            deadline_monotonic=hard_deadline,
        )
        if _hard_deadline_expired(hard_deadline):
            deadline_expired = True
            return selected_result()
        evaluate_stage("wrist_unload", wrist_candidates)
        return selected_result()

    def _local_step_limit_escape_candidates(
        self,
        *,
        reference_qpos: np.ndarray,
        target_pose: np.ndarray,
        reference_pose: np.ndarray,
        joint_velocity_profile: JointVelocityLimitProfile,
        dt_sec: float,
        deadline_monotonic: float | None = None,
        timing_profile: dict[str, Any] | None = None,
        timing_phase: str = "local_step",
    ) -> tuple[list[IKCandidate], dict[str, Any]]:
        metadata: dict[str, Any] = {
            "enabled": bool(getattr(self.profile, "local_step_limit_escape_enabled", False)),
            "candidate_count": 0,
        }
        if not bool(getattr(self.profile, "local_step_limit_escape_enabled", False)):
            metadata["status"] = "disabled"
            return [], metadata
        if _hard_deadline_expired(deadline_monotonic):
            metadata["status"] = "skipped_hard_deadline"
            metadata["hard_deadline_expired"] = True
            return [], metadata
        limits = _joint_position_limits_from_provider(self.provider, reference_qpos.shape)
        if limits is None:
            metadata["status"] = "joint_position_limits_unavailable"
            return [], metadata
        q_min, q_max = limits
        q = np.asarray(reference_qpos, dtype=float).reshape(-1)
        q_min = np.asarray(q_min, dtype=float).reshape(q.shape)
        q_max = np.asarray(q_max, dtype=float).reshape(q.shape)
        if np.any(q_max <= q_min):
            metadata["status"] = "invalid_joint_position_limits"
            return [], metadata

        lower_margin = q - q_min
        upper_margin = q_max - q
        min_margin = np.minimum(lower_margin, upper_margin)
        activation_margin_rad = np.deg2rad(
            max(0.0, float(getattr(self.profile, "local_step_limit_escape_activation_margin_deg", 2.0)))
        )
        active_indices = np.flatnonzero(min_margin <= activation_margin_rad)
        metadata.update(
            {
                "status": "inactive" if active_indices.size == 0 else "active",
                "activation_margin_deg": float(np.rad2deg(activation_margin_rad)),
                "min_margin_deg": np.rad2deg(min_margin).astype(float).tolist(),
                "lower_margin_deg": np.rad2deg(lower_margin).astype(float).tolist(),
                "upper_margin_deg": np.rad2deg(upper_margin).astype(float).tolist(),
                "active_joint_indices": [int(index) for index in active_indices.tolist()],
            }
        )
        if active_indices.size == 0:
            return [], metadata

        max_candidates = max(0, int(getattr(self.profile, "local_step_limit_escape_max_candidates", 12)))
        if max_candidates <= 0:
            metadata["status"] = "max_candidates_zero"
            return [], metadata
        step_fraction = max(0.0, min(1.0, float(getattr(self.profile, "local_step_limit_escape_step_fraction", 1.0))))
        if step_fraction <= 0.0:
            metadata["status"] = "step_fraction_zero"
            return [], metadata
        max_step_rad = np.deg2rad(joint_velocity_profile.max_step_deg(dt_sec)) * step_fraction
        before_m, before_deg = pose_delta_metrics(reference_pose, target_pose)
        before_score = _pose_progress_score(
            pose_error_m=before_m,
            pose_error_deg=before_deg,
            position_tolerance_m=float(self.profile.target_tolerance.position_m),
            orientation_tolerance_deg=float(self.profile.target_tolerance.orientation_deg),
        )
        active_set = {int(index) for index in active_indices.tolist()}
        scored: list[tuple[float, float, float, int, int, np.ndarray, dict[str, Any]]] = []
        eps = 1e-9
        for joint_index in range(q.size):
            if _hard_deadline_expired(deadline_monotonic):
                metadata["status"] = "interrupted_hard_deadline"
                metadata["hard_deadline_expired"] = True
                return [], metadata
            for direction in (-1, 1):
                if _hard_deadline_expired(deadline_monotonic):
                    metadata["status"] = "interrupted_hard_deadline"
                    metadata["hard_deadline_expired"] = True
                    return [], metadata
                if direction < 0 and q[joint_index] <= q_min[joint_index] + eps:
                    continue
                if direction > 0 and q[joint_index] >= q_max[joint_index] - eps:
                    continue
                if joint_index in active_set:
                    near_upper = upper_margin[joint_index] <= lower_margin[joint_index]
                    if near_upper and direction > 0:
                        continue
                    if (not near_upper) and direction < 0:
                        continue
                step = min(float(max_step_rad[joint_index]), float(upper_margin[joint_index] if direction > 0 else lower_margin[joint_index]))
                if step <= 1e-12:
                    continue
                candidate_qpos = q.copy()
                candidate_qpos[joint_index] += float(direction) * step
                fk_started = time.perf_counter()
                candidate_pose = self.provider.forward_kinematics(candidate_qpos)
                fk_elapsed_ms = _elapsed_ms(fk_started)
                _timing_add(timing_profile, "fk_ms", fk_elapsed_ms)
                _timing_add(timing_profile, f"{timing_phase}_fk_ms", fk_elapsed_ms)
                _timing_add(timing_profile, "local_limit_escape_fk_ms", fk_elapsed_ms)
                if _hard_deadline_expired(deadline_monotonic):
                    metadata["status"] = "interrupted_hard_deadline"
                    metadata["hard_deadline_expired"] = True
                    return [], metadata
                if candidate_pose is None:
                    pose_progress = -1e9
                    after_m = None
                    after_deg = None
                    after_score = 1e9
                else:
                    after_m, after_deg = pose_delta_metrics(candidate_pose, target_pose)
                    after_score = _pose_progress_score(
                        pose_error_m=after_m,
                        pose_error_deg=after_deg,
                        position_tolerance_m=float(self.profile.target_tolerance.position_m),
                        orientation_tolerance_deg=float(self.profile.target_tolerance.orientation_deg),
                    )
                    pose_progress = before_score - after_score
                candidate_lower_margin = candidate_qpos - q_min
                candidate_upper_margin = q_max - candidate_qpos
                margin_before = float(np.min(min_margin))
                margin_after = float(np.min(np.minimum(candidate_lower_margin, candidate_upper_margin)))
                margin_progress = margin_after - margin_before
                candidate_metadata = {
                    "local_step_limit_escape_candidate": True,
                    "local_step_limit_escape_joint_index": int(joint_index),
                    "local_step_limit_escape_direction": int(direction),
                    "local_step_limit_escape_step_deg": float(np.rad2deg(step)),
                    "local_step_limit_escape_step_fraction": float(step_fraction),
                    "local_step_limit_escape_active_joint_indices": [int(index) for index in active_indices.tolist()],
                    "local_step_limit_escape_min_margin_before_deg": float(np.rad2deg(margin_before)),
                    "local_step_limit_escape_min_margin_after_deg": float(np.rad2deg(margin_after)),
                    "local_step_limit_escape_margin_progress_deg": float(np.rad2deg(margin_progress)),
                    "local_step_limit_escape_predicted_pose_error_m": after_m,
                    "local_step_limit_escape_predicted_pose_error_deg": after_deg,
                    "local_step_limit_escape_predicted_pose_progress": float(pose_progress),
                }
                scored.append(
                    (
                        float(joint_index not in active_set),
                        -float(margin_progress),
                        -float(pose_progress),
                        int(joint_index),
                        int(direction),
                        candidate_qpos,
                        candidate_metadata,
                    )
                )
        scored.sort(key=lambda item: item[:5])
        candidates: list[IKCandidate] = []
        seen: list[np.ndarray] = []
        for _non_active, _margin, _score, joint_index, direction, candidate_qpos, candidate_metadata in scored:
            if _hard_deadline_expired(deadline_monotonic):
                metadata["status"] = "interrupted_hard_deadline"
                metadata["hard_deadline_expired"] = True
                return [], metadata
            if len(candidates) >= max_candidates:
                break
            if any(np.allclose(candidate_qpos, existing, atol=1e-9) for existing in seen):
                continue
            seen.append(candidate_qpos.copy())
            candidates.append(
                IKCandidate(
                    qpos=candidate_qpos.copy(),
                    source="joint_limit_escape",
                    seed_source="reference",
                    method=f"joint_limit_escape_j{int(joint_index) + 1}_{'pos' if int(direction) > 0 else 'neg'}",
                    metadata=candidate_metadata,
                )
            )
        metadata.update(
            {
                "candidate_count": len(candidates),
                "max_candidates": int(max_candidates),
                "step_fraction": float(step_fraction),
                "emitted_methods": [candidate.method for candidate in candidates],
            }
        )
        return candidates, metadata

    def _local_step_singularity_escape_candidates(
        self,
        *,
        reference_qpos: np.ndarray,
        target_pose: np.ndarray,
        reference_pose: np.ndarray,
        joint_velocity_profile: JointVelocityLimitProfile,
        dt_sec: float,
        deadline_monotonic: float | None = None,
        timing_profile: dict[str, Any] | None = None,
        timing_phase: str = "local_step",
    ) -> tuple[list[IKCandidate], dict[str, Any]]:
        metadata: dict[str, Any] = {
            "enabled": bool(getattr(self.profile, "local_step_singularity_escape_enabled", False)),
            "candidate_count": 0,
        }
        if not bool(getattr(self.profile, "local_step_singularity_escape_enabled", False)):
            metadata["status"] = "disabled"
            return [], metadata
        if _hard_deadline_expired(deadline_monotonic):
            metadata["status"] = "skipped_hard_deadline"
            metadata["hard_deadline_expired"] = True
            return [], metadata

        q = np.asarray(reference_qpos, dtype=float).reshape(-1)
        reference_report = self.provider.analyze_singularity(q)
        if _hard_deadline_expired(deadline_monotonic):
            metadata["status"] = "interrupted_hard_deadline"
            metadata["hard_deadline_expired"] = True
            return [], metadata
        metadata["reference_singularity"] = _singularity_metadata(reference_report)
        if not reference_report.is_singular:
            metadata["status"] = "inactive_reference_not_singular"
            return [], metadata

        max_candidates = max(0, int(getattr(self.profile, "local_step_singularity_escape_max_candidates", 6)))
        if max_candidates <= 0:
            metadata["status"] = "max_candidates_zero"
            return [], metadata
        step_fraction = max(
            0.0,
            min(1.0, float(getattr(self.profile, "local_step_singularity_escape_step_fraction", 1.0))),
        )
        if step_fraction <= 0.0:
            metadata["status"] = "step_fraction_zero"
            return [], metadata

        limits = _joint_position_limits_from_provider(self.provider, q.shape)
        q_min: np.ndarray | None = None
        q_max: np.ndarray | None = None
        if limits is not None:
            q_min, q_max = limits
        max_step_rad = np.deg2rad(joint_velocity_profile.max_step_deg(dt_sec)) * step_fraction
        before_m, before_deg = pose_delta_metrics(reference_pose, target_pose)
        before_score = _pose_progress_score(
            pose_error_m=before_m,
            pose_error_deg=before_deg,
            position_tolerance_m=float(self.profile.target_tolerance.position_m),
            orientation_tolerance_deg=float(self.profile.target_tolerance.orientation_deg),
        )
        scored: list[tuple[int, float, float, int, int, np.ndarray, dict[str, Any]]] = []
        for joint_index in range(q.size):
            if _hard_deadline_expired(deadline_monotonic):
                metadata["status"] = "interrupted_hard_deadline"
                metadata["hard_deadline_expired"] = True
                return [], metadata
            step = float(max_step_rad[joint_index])
            if step <= 1e-12:
                continue
            for direction in (-1, 1):
                if _hard_deadline_expired(deadline_monotonic):
                    metadata["status"] = "interrupted_hard_deadline"
                    metadata["hard_deadline_expired"] = True
                    return [], metadata
                candidate_qpos = q.copy()
                candidate_qpos[joint_index] += float(direction) * step
                if q_min is not None and q_max is not None:
                    if candidate_qpos[joint_index] < q_min[joint_index] - 1e-9:
                        continue
                    if candidate_qpos[joint_index] > q_max[joint_index] + 1e-9:
                        continue
                candidate_report = self.provider.analyze_singularity(candidate_qpos)
                if _hard_deadline_expired(deadline_monotonic):
                    metadata["status"] = "interrupted_hard_deadline"
                    metadata["hard_deadline_expired"] = True
                    return [], metadata
                candidate_pose = self.provider.forward_kinematics(candidate_qpos)
                if _hard_deadline_expired(deadline_monotonic):
                    metadata["status"] = "interrupted_hard_deadline"
                    metadata["hard_deadline_expired"] = True
                    return [], metadata
                if candidate_pose is None:
                    continue
                after_m, after_deg = pose_delta_metrics(candidate_pose, target_pose)
                after_score = _pose_progress_score(
                    pose_error_m=after_m,
                    pose_error_deg=after_deg,
                    position_tolerance_m=float(self.profile.target_tolerance.position_m),
                    orientation_tolerance_deg=float(self.profile.target_tolerance.orientation_deg),
                )
                pose_progress = before_score - after_score
                normal_rank = 0 if not candidate_report.is_singular else 1
                metric_delta = _singularity_metric_delta(reference_report, candidate_report)
                candidate_metadata = {
                    "local_step_singularity_escape_candidate": True,
                    "local_step_singularity_escape_joint_index": int(joint_index),
                    "local_step_singularity_escape_direction": int(direction),
                    "local_step_singularity_escape_step_deg": float(np.rad2deg(step)),
                    "local_step_singularity_escape_step_fraction": float(step_fraction),
                    "local_step_singularity_escape_reference": _singularity_metadata(reference_report),
                    "local_step_singularity_escape_candidate_report": _singularity_metadata(candidate_report),
                    "local_step_singularity_escape_metric_delta": metric_delta,
                    "local_step_singularity_escape_predicted_pose_error_m": float(after_m),
                    "local_step_singularity_escape_predicted_pose_error_deg": float(after_deg),
                    "local_step_singularity_escape_predicted_pose_progress": float(pose_progress),
                }
                scored.append(
                    (
                        normal_rank,
                        -float(metric_delta if metric_delta is not None else 0.0),
                        -float(pose_progress),
                        int(joint_index),
                        int(direction),
                        candidate_qpos,
                        candidate_metadata,
                    )
                )
        scored.sort(key=lambda item: item[:5])
        candidates: list[IKCandidate] = []
        seen: list[np.ndarray] = []
        for _rank, _metric, _pose, joint_index, direction, candidate_qpos, candidate_metadata in scored:
            if _hard_deadline_expired(deadline_monotonic):
                metadata["status"] = "interrupted_hard_deadline"
                metadata["hard_deadline_expired"] = True
                return [], metadata
            if len(candidates) >= max_candidates:
                break
            if any(np.allclose(candidate_qpos, existing, atol=1e-9) for existing in seen):
                continue
            seen.append(candidate_qpos.copy())
            candidates.append(
                IKCandidate(
                    qpos=candidate_qpos.copy(),
                    source="singularity_escape",
                    seed_source="reference",
                    method=f"singularity_escape_j{int(joint_index) + 1}_{'pos' if int(direction) > 0 else 'neg'}",
                    metadata=candidate_metadata,
                )
            )
        metadata.update(
            {
                "status": "active" if candidates else "no_candidates",
                "candidate_count": len(candidates),
                "max_candidates": int(max_candidates),
                "step_fraction": float(step_fraction),
                "emitted_methods": [candidate.method for candidate in candidates],
            }
        )
        return candidates, metadata

    def _local_step_posture_escape_candidates(
        self,
        *,
        reference_qpos: np.ndarray,
        joint_velocity_profile: JointVelocityLimitProfile,
        dt_sec: float,
        deadline_monotonic: float | None = None,
    ) -> tuple[list[IKCandidate], dict[str, Any]]:
        metadata: dict[str, Any] = {
            "enabled": bool(
                self.posture_preference is not None
                and self.posture_preference.enabled
                and self.posture_preference.local_escape_enabled
            ),
            "candidate_count": 0,
        }
        if self.posture_preference is None or not self.posture_preference.local_escape_enabled:
            metadata["status"] = "disabled"
            return [], metadata
        if _hard_deadline_expired(deadline_monotonic):
            metadata["status"] = "skipped_hard_deadline"
            metadata["hard_deadline_expired"] = True
            return [], metadata
        q = np.asarray(reference_qpos, dtype=float).reshape(-1)
        limits = _joint_position_limits_from_provider(self.provider, q.shape)
        q_min = q_max = None
        if limits is not None:
            q_min, q_max = limits
        max_step_rad = np.deg2rad(joint_velocity_profile.max_step_deg(dt_sec))
        targets, preference_metadata = self.posture_preference.local_escape_targets(
            reference_qpos=q,
            max_step_rad=max_step_rad,
            q_min=q_min,
            q_max=q_max,
            deadline_monotonic=deadline_monotonic,
        )
        metadata.update(preference_metadata)
        candidates: list[IKCandidate] = []
        for label, candidate_qpos, candidate_metadata in targets:
            if _hard_deadline_expired(deadline_monotonic):
                metadata["status"] = "interrupted_hard_deadline"
                metadata["hard_deadline_expired"] = True
                return [], metadata
            candidates.append(
                IKCandidate(
                    qpos=np.asarray(candidate_qpos, dtype=float).reshape(q.shape),
                    source="posture_escape",
                    seed_source="reference",
                    method=f"posture_escape_{label}",
                    metadata=candidate_metadata,
                )
            )
        metadata["candidate_count"] = len(candidates)
        return candidates, metadata

    def _local_step_wrist_unload_candidates(
        self,
        *,
        reference_qpos: np.ndarray,
        joint_velocity_profile: JointVelocityLimitProfile,
        dt_sec: float,
        deadline_monotonic: float | None = None,
    ) -> tuple[list[IKCandidate], dict[str, Any]]:
        metadata: dict[str, Any] = {
            "enabled": bool(
                self.posture_preference is not None
                and self.posture_preference.enabled
                and self.posture_preference.wrist_unload_enabled
            ),
            "candidate_count": 0,
        }
        if self.posture_preference is None or not self.posture_preference.wrist_unload_enabled:
            metadata["status"] = "disabled"
            return [], metadata
        if _hard_deadline_expired(deadline_monotonic):
            metadata["status"] = "skipped_hard_deadline"
            metadata["hard_deadline_expired"] = True
            return [], metadata
        q = np.asarray(reference_qpos, dtype=float).reshape(-1)
        limits = _joint_position_limits_from_provider(self.provider, q.shape)
        q_min = q_max = None
        if limits is not None:
            q_min, q_max = limits
        max_step_rad = np.deg2rad(joint_velocity_profile.max_step_deg(dt_sec))
        targets, preference_metadata = self.posture_preference.wrist_unload_targets(
            reference_qpos=q,
            max_step_rad=max_step_rad,
            q_min=q_min,
            q_max=q_max,
            deadline_monotonic=deadline_monotonic,
        )
        metadata.update(preference_metadata)
        candidates: list[IKCandidate] = []
        for label, candidate_qpos, candidate_metadata in targets:
            if _hard_deadline_expired(deadline_monotonic):
                metadata["status"] = "interrupted_hard_deadline"
                metadata["hard_deadline_expired"] = True
                return [], metadata
            candidates.append(
                IKCandidate(
                    qpos=np.asarray(candidate_qpos, dtype=float).reshape(q.shape),
                    source="wrist_unload",
                    seed_source="reference",
                    method=str(label),
                    metadata=candidate_metadata,
                )
            )
        metadata["candidate_count"] = len(candidates)
        return candidates, metadata

    def _local_first_full_solve_skip_reason(
        self,
        *,
        local_result: IKResolveResult,
        started_at: float,
        frame_deadline: float | None,
    ) -> str | None:
        if bool(getattr(self.profile, "local_step_first_skip_full_when_far", False)):
            fraction = _finite_float(local_result.metadata.get("pose_tolerance_fraction"), default=None)
            threshold = max(
                0.0,
                float(getattr(self.profile, "local_step_first_skip_full_pose_fraction", 1.0)),
            )
            if fraction is not None and fraction >= threshold:
                return "far_tracking_local_step_accepted"
        if bool(getattr(self.profile, "local_step_first_skip_full_if_budget_low", False)):
            if frame_deadline is None:
                return None
            remaining_ms = (float(frame_deadline) - time.perf_counter()) * 1000.0
            min_remaining_ms = max(
                0.0,
                float(getattr(self.profile, "local_step_first_full_solve_min_remaining_ms", 5.0)),
            )
            if remaining_ms <= min_remaining_ms:
                return "frame_budget_remaining_low"
            if _elapsed_ms(started_at) >= max(
                0.0,
                (float(frame_deadline) - float(started_at)) * 1000.0 - min_remaining_ms,
            ):
                return "frame_budget_elapsed"
        return None

    def _evaluate_local_step_candidate(
        self,
        *,
        candidate: IKCandidate,
        reference_qpos: np.ndarray,
        target_pose: np.ndarray,
        reference_pose: np.ndarray,
        request_metadata: dict[str, Any],
        joint_velocity_profile: JointVelocityLimitProfile,
        dt_sec: float,
        step_scale: float,
        allow_pose_nonprogress_transition: bool = False,
        timing_profile: dict[str, Any] | None = None,
        timing_phase: str = "local_step",
    ) -> tuple[bool, dict[str, Any]]:
        evaluate_started = time.perf_counter()
        raw_qpos = np.asarray(candidate.qpos, dtype=float).reshape(-1)
        scale = max(1e-6, min(1.0, float(step_scale)))
        metadata: dict[str, Any] = {
            "candidate_source": candidate.source,
            "seed_source": candidate.seed_source,
            "method": candidate.method,
            "local_step_candidate_metadata": dict(candidate.metadata),
            "local_step_iteration_count": int(candidate.metadata.get("local_step_iteration_count", 0) or 0),
            "local_step_candidate": True,
            "local_step_scale_factor": scale,
            "local_step_limit_escape_candidate": bool(candidate.metadata.get("local_step_limit_escape_candidate", False)),
            "local_step_singularity_escape_candidate": bool(
                candidate.metadata.get("local_step_singularity_escape_candidate", False)
            ),
            "local_step_posture_escape_candidate": bool(
                candidate.metadata.get("local_step_posture_escape_candidate", False)
            ),
            "local_step_wrist_unload_candidate": bool(
                candidate.metadata.get("local_step_wrist_unload_candidate", False)
            ),
        }
        if raw_qpos.shape != reference_qpos.shape or not np.all(np.isfinite(raw_qpos)):
            metadata["reject_reason"] = "candidate_qpos_invalid"
            metadata["score"] = 1e18
            _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
            return False, metadata

        raw_joint_delta_deg = np.rad2deg(np.abs(raw_qpos - reference_qpos))
        metadata["local_step_raw_joint_delta_deg"] = raw_joint_delta_deg.tolist()
        score_penalty_deg = 0.0
        raw_posture_comparison = None
        if self.posture_preference is not None:
            raw_posture_comparison = self.posture_preference.compare(reference_qpos, raw_qpos)
            metadata["raw_posture_preference"] = {
                "name": self.posture_preference.name,
                **raw_posture_comparison.metadata(),
            }
        branch_indices = tuple(int(index) for index in self.profile.hard_branch_joint_indices)
        branch_limits = self.profile.hard_branch_limits_array()
        if branch_indices:
            branch_delta = raw_joint_delta_deg[list(branch_indices)]
            metadata["hard_branch_joint_delta_deg"] = branch_delta.tolist()
            if np.any(branch_delta > branch_limits):
                exceeded = tuple(
                    int(branch_indices[index])
                    for index, exceeded in enumerate((branch_delta > branch_limits).tolist())
                    if bool(exceeded)
                )
                escape_metadata = (
                    None
                    if self.posture_preference is None
                    else self.posture_preference.hard_branch_escape_metadata(
                        reference_qpos=reference_qpos,
                        candidate_qpos=raw_qpos,
                        exceeded_joint_indices=exceeded,
                    )
                )
                if escape_metadata is not None:
                    metadata["posture_hard_branch_escape"] = escape_metadata
                if escape_metadata is None or not bool(escape_metadata.get("allowed", False)):
                    metadata["reject_reason"] = "hard_branch_jump"
                    metadata["hard_branch_max_delta_deg"] = branch_limits.tolist()
                    metadata["score"] = 1e18
                    _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
                    return False, metadata
                metadata["hard_branch_escape_allowed"] = True
                metadata["hard_branch_escape_reason"] = "posture_bad_to_good"
                metadata["hard_branch_max_delta_deg"] = branch_limits.tolist()

        full_limited = limit_joint_step(
            reference_qpos=reference_qpos,
            target_qpos=raw_qpos,
            profile=joint_velocity_profile,
            dt_sec=float(dt_sec),
        )
        full_delta = full_limited.qpos - reference_qpos
        qpos = reference_qpos + full_delta * scale
        local_delta_deg = np.rad2deg(qpos - reference_qpos)
        full_delta_deg = np.rad2deg(full_delta)
        scaled_max_step_deg = np.asarray(full_limited.max_step_deg, dtype=float).reshape(-1) * scale
        metadata.update(
            {
                "local_step_requested_dt_sec": float(dt_sec),
                "local_step_scaled_dt_sec": float(dt_sec) * scale,
                "local_step_raw_joint_deg": np.rad2deg(raw_qpos).astype(float).tolist(),
                "local_step_unscaled_output_qpos": full_limited.qpos.astype(float).tolist(),
                "local_step_unscaled_output_joint_deg": np.rad2deg(full_limited.qpos).astype(float).tolist(),
                "local_step_unscaled_delta_deg": full_delta_deg.astype(float).tolist(),
                "local_step_output_qpos": qpos.astype(float).tolist(),
                "local_step_output_joint_deg": np.rad2deg(qpos).astype(float).tolist(),
                "local_step_delta_deg": local_delta_deg.astype(float).tolist(),
                "local_step_full_max_step_deg": full_limited.max_step_deg.astype(float).tolist(),
                "local_step_max_step_deg": scaled_max_step_deg.astype(float).tolist(),
                "local_step_raw_limited": bool(full_limited.limited),
                "local_step_raw_limit_ratio": float(full_limited.limit_ratio),
                "local_step_scaled_limit_ratio": float(full_limited.limit_ratio) / scale,
            }
        )
        output_posture_comparison = None
        if self.posture_preference is not None:
            output_posture_comparison = self.posture_preference.compare(reference_qpos, qpos)
            posture_penalty_deg = max(0.0, float(output_posture_comparison.candidate.score_deg))
            score_penalty_deg += posture_penalty_deg
            metadata["posture_preference"] = {
                "name": self.posture_preference.name,
                "score_penalty_deg": float(posture_penalty_deg),
                **output_posture_comparison.metadata(),
            }

        fk_started = time.perf_counter()
        output_pose = self.provider.forward_kinematics(qpos)
        fk_elapsed_ms = _elapsed_ms(fk_started)
        _timing_add(timing_profile, "fk_ms", fk_elapsed_ms)
        _timing_add(timing_profile, f"{timing_phase}_fk_ms", fk_elapsed_ms)
        if output_pose is None:
            metadata["reject_reason"] = "fk_pose_unavailable"
            metadata["score"] = 1e18
            _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
            return False, metadata
        before_m, before_deg = pose_delta_metrics(reference_pose, target_pose)
        after_m, after_deg = pose_delta_metrics(output_pose, target_pose)
        position_progress_m = before_m - after_m
        orientation_progress_deg = before_deg - after_deg
        before_score = _pose_progress_score(
            pose_error_m=before_m,
            pose_error_deg=before_deg,
            position_tolerance_m=float(self.profile.target_tolerance.position_m),
            orientation_tolerance_deg=float(self.profile.target_tolerance.orientation_deg),
        )
        after_score = _pose_progress_score(
            pose_error_m=after_m,
            pose_error_deg=after_deg,
            position_tolerance_m=float(self.profile.target_tolerance.position_m),
            orientation_tolerance_deg=float(self.profile.target_tolerance.orientation_deg),
        )
        before_balanced_score = _pose_balanced_progress_score(
            pose_error_m=before_m,
            pose_error_deg=before_deg,
            position_tolerance_m=float(self.profile.target_tolerance.position_m),
            orientation_tolerance_deg=float(self.profile.target_tolerance.orientation_deg),
        )
        after_balanced_score = _pose_balanced_progress_score(
            pose_error_m=after_m,
            pose_error_deg=after_deg,
            position_tolerance_m=float(self.profile.target_tolerance.position_m),
            orientation_tolerance_deg=float(self.profile.target_tolerance.orientation_deg),
        )
        progress = before_score - after_score
        balanced_progress = before_balanced_score - after_balanced_score
        before_pose_fraction = _pose_tolerance_fraction(
            pose_error_m=before_m,
            pose_error_deg=before_deg,
            position_tolerance_m=float(self.profile.target_tolerance.position_m),
            orientation_tolerance_deg=float(self.profile.target_tolerance.orientation_deg),
        )
        pose_fraction = _pose_tolerance_fraction(
            pose_error_m=after_m,
            pose_error_deg=after_deg,
            position_tolerance_m=float(self.profile.target_tolerance.position_m),
            orientation_tolerance_deg=float(self.profile.target_tolerance.orientation_deg),
        )
        pose_slack_tier = _pose_slack_tier(pose_fraction)
        metadata.update(
            {
                "local_step_output_pose": output_pose.astype(float).tolist(),
                "local_step_pose_error_before_m": before_m,
                "local_step_pose_error_before_deg": before_deg,
                "pose_error_m": after_m,
                "pose_error_deg": after_deg,
                "pose_tolerance_fraction": pose_fraction,
                "pose_slack_tier": pose_slack_tier,
                "local_step_pose_score_before": before_score,
                "local_step_pose_score_after": after_score,
                "local_step_pose_tolerance_fraction_before": before_pose_fraction,
                "local_step_pose_progress": progress,
                "local_step_balanced_pose_score_before": before_balanced_score,
                "local_step_balanced_pose_score_after": after_balanced_score,
                "local_step_balanced_pose_progress": balanced_progress,
                "local_step_position_progress_m": position_progress_m,
                "local_step_orientation_progress_deg": orientation_progress_deg,
            }
        )
        min_progress = max(0.0, float(self.profile.local_step_min_progress_fraction))
        position_progress_threshold = max(0.0, float(self.profile.local_step_accept_position_progress_m))
        orientation_progress_threshold = max(0.0, float(self.profile.local_step_accept_orientation_progress_deg))
        worsen_limits = _local_step_worsen_limits(
            profile=self.profile,
            before_m=before_m,
            before_deg=before_deg,
        )
        max_position_worsen_m = worsen_limits["position_m"]
        max_orientation_worsen_deg = worsen_limits["orientation_deg"]
        position_progress_ok = (
            position_progress_m >= position_progress_threshold
            and orientation_progress_deg >= -max_orientation_worsen_deg
        )
        orientation_progress_ok = (
            orientation_progress_deg >= orientation_progress_threshold
            and position_progress_m >= -max_position_worsen_m
        )
        balanced_progress_ok = (
            bool(worsen_limits["active"])
            and balanced_progress >= min_progress
            and position_progress_m >= -max_position_worsen_m
            and orientation_progress_deg >= -max_orientation_worsen_deg
        )
        limit_escape_candidate = bool(candidate.metadata.get("local_step_limit_escape_candidate", False))
        limit_escape_margin_progress_deg = float(
            candidate.metadata.get("local_step_limit_escape_margin_progress_deg", 0.0) or 0.0
        )
        limit_escape_min_margin_before_deg = _finite_float(
            candidate.metadata.get("local_step_limit_escape_min_margin_before_deg"),
            default=None,
        )
        singularity_escape_candidate = bool(
            candidate.metadata.get("local_step_singularity_escape_candidate", False)
        )
        posture_escape_candidate = bool(candidate.metadata.get("local_step_posture_escape_candidate", False))
        wrist_unload_candidate = bool(candidate.metadata.get("local_step_wrist_unload_candidate", False))
        wrist_unload_improvement_deg = float(
            candidate.metadata.get("local_step_wrist_unload_cost_improvement_deg", 0.0) or 0.0
        )
        posture_improvement_deg = (
            0.0
            if output_posture_comparison is None
            else float(output_posture_comparison.improvement_deg)
        )
        posture_escape_active = bool(
            self.posture_preference is not None
            and posture_improvement_deg
            >= max(0.0, float(self.posture_preference.local_escape_min_improvement_deg))
            and (
                posture_escape_candidate
                or bool(output_posture_comparison.reference.bad if output_posture_comparison else False)
            )
        )
        orientation_recovery_max_position_worsen_m = max(
            0.0,
            float(getattr(self.profile, "local_step_orientation_recovery_max_position_worsen_m", 0.08)),
        )
        orientation_recovery_max_after_position_error_m = max(
            0.0,
            float(getattr(self.profile, "local_step_orientation_recovery_max_after_position_error_m", 0.30)),
        )
        orientation_recovery_ok = (
            bool(getattr(self.profile, "local_step_orientation_recovery_enabled", False))
            and orientation_progress_deg
            >= max(0.0, float(getattr(self.profile, "local_step_orientation_recovery_min_progress_deg", 5.0)))
            and position_progress_m >= -orientation_recovery_max_position_worsen_m
            and (
                orientation_recovery_max_after_position_error_m <= 0.0
                or after_m <= orientation_recovery_max_after_position_error_m
            )
        )
        limit_escape_max_position_worsen_m = max(
            max_position_worsen_m,
            max(0.0, float(getattr(self.profile, "local_step_limit_escape_max_position_worsen_m", 0.03))),
        )
        limit_escape_max_orientation_worsen_deg = max(
            max_orientation_worsen_deg,
            max(0.0, float(getattr(self.profile, "local_step_limit_escape_max_orientation_worsen_deg", 12.0))),
        )
        limit_escape_bounded_worse_ok = (
            limit_escape_candidate
            and bool(getattr(self.profile, "local_step_limit_escape_accept_bounded_worse", True))
            and limit_escape_margin_progress_deg > 0.0
            and position_progress_m >= -limit_escape_max_position_worsen_m
            and orientation_progress_deg >= -limit_escape_max_orientation_worsen_deg
        )
        singularity_escape_max_position_worsen_m = max(
            max_position_worsen_m,
            max(0.0, float(getattr(self.profile, "local_step_singularity_escape_max_position_worsen_m", 0.02))),
        )
        singularity_escape_max_orientation_worsen_deg = max(
            max_orientation_worsen_deg,
            max(0.0, float(getattr(self.profile, "local_step_singularity_escape_max_orientation_worsen_deg", 8.0))),
        )
        singularity_escape_bounded_worse_ok = (
            singularity_escape_candidate
            and position_progress_m >= -singularity_escape_max_position_worsen_m
            and orientation_progress_deg >= -singularity_escape_max_orientation_worsen_deg
        )
        posture_escape_max_position_worsen_m = max_position_worsen_m
        posture_escape_max_orientation_worsen_deg = max_orientation_worsen_deg
        if self.posture_preference is not None:
            posture_escape_max_position_worsen_m = max(
                posture_escape_max_position_worsen_m,
                max(0.0, float(self.posture_preference.local_escape_max_position_worsen_m)),
            )
            posture_escape_max_orientation_worsen_deg = max(
                posture_escape_max_orientation_worsen_deg,
                max(0.0, float(self.posture_preference.local_escape_max_orientation_worsen_deg)),
            )
        posture_escape_bounded_worse_ok = (
            posture_escape_active
            and position_progress_m >= -posture_escape_max_position_worsen_m
            and orientation_progress_deg >= -posture_escape_max_orientation_worsen_deg
        )
        wrist_unload_max_position_worsen_m = max_position_worsen_m
        wrist_unload_max_orientation_worsen_deg = max_orientation_worsen_deg
        wrist_unload_active = False
        if self.posture_preference is not None:
            wrist_unload_max_position_worsen_m = max(
                wrist_unload_max_position_worsen_m,
                max(0.0, float(self.posture_preference.wrist_unload_max_position_worsen_m)),
            )
            wrist_unload_max_orientation_worsen_deg = max(
                wrist_unload_max_orientation_worsen_deg,
                max(0.0, float(self.posture_preference.wrist_unload_max_orientation_worsen_deg)),
            )
            wrist_unload_active = bool(
                wrist_unload_candidate
                and (
                    wrist_unload_improvement_deg
                    >= max(0.0, float(self.posture_preference.wrist_unload_min_cost_improvement_deg))
                    or position_progress_ok
                    or orientation_progress_ok
                    or balanced_progress_ok
                )
            )
        wrist_unload_bounded_worse_ok = (
            wrist_unload_active
            and position_progress_m >= -wrist_unload_max_position_worsen_m
            and orientation_progress_deg >= -wrist_unload_max_orientation_worsen_deg
        )
        limit_escape_within_tolerance_ok = (
            limit_escape_candidate
            and bool(getattr(self.profile, "local_step_limit_escape_accept_worse_within_tolerance", True))
            and limit_escape_margin_progress_deg > 0.0
            and position_progress_m >= -max_position_worsen_m
            and orientation_progress_deg >= -max_orientation_worsen_deg
        )
        metadata.update(
            {
                "local_step_accept_position_progress_m": position_progress_threshold,
                "local_step_accept_orientation_progress_deg": orientation_progress_threshold,
                "local_step_max_position_worsen_m": max_position_worsen_m,
                "local_step_max_orientation_worsen_deg": max_orientation_worsen_deg,
                "local_step_base_max_position_worsen_m": worsen_limits["base_position_m"],
                "local_step_base_max_orientation_worsen_deg": worsen_limits["base_orientation_deg"],
                "local_step_dynamic_worsen_enabled": worsen_limits["enabled"],
                "local_step_dynamic_worsen_active": worsen_limits["active"],
                "local_step_dynamic_worsen_position_fraction": worsen_limits["position_fraction"],
                "local_step_dynamic_worsen_orientation_fraction": worsen_limits["orientation_fraction"],
                "local_step_position_progress_ok": bool(position_progress_ok),
                "local_step_orientation_progress_ok": bool(orientation_progress_ok),
                "local_step_balanced_progress_ok": bool(balanced_progress_ok),
                "local_step_orientation_recovery_ok": bool(orientation_recovery_ok),
                "local_step_orientation_recovery_max_position_worsen_m": orientation_recovery_max_position_worsen_m,
                "local_step_orientation_recovery_max_after_position_error_m": orientation_recovery_max_after_position_error_m,
                "local_step_limit_escape_margin_progress_deg": limit_escape_margin_progress_deg,
                "local_step_limit_escape_min_margin_before_deg": limit_escape_min_margin_before_deg,
                "local_step_limit_escape_within_tolerance_ok": bool(limit_escape_within_tolerance_ok),
                "local_step_limit_escape_bounded_worse_ok": bool(limit_escape_bounded_worse_ok),
                "local_step_limit_escape_max_position_worsen_m": limit_escape_max_position_worsen_m,
                "local_step_limit_escape_max_orientation_worsen_deg": limit_escape_max_orientation_worsen_deg,
                "local_step_singularity_escape_bounded_worse_ok": bool(singularity_escape_bounded_worse_ok),
                "local_step_singularity_escape_max_position_worsen_m": singularity_escape_max_position_worsen_m,
                "local_step_singularity_escape_max_orientation_worsen_deg": singularity_escape_max_orientation_worsen_deg,
                "local_step_posture_escape_active": bool(posture_escape_active),
                "local_step_posture_escape_improvement_deg": float(posture_improvement_deg),
                "local_step_posture_escape_bounded_worse_ok": bool(posture_escape_bounded_worse_ok),
                "local_step_posture_escape_max_position_worsen_m": posture_escape_max_position_worsen_m,
                "local_step_posture_escape_max_orientation_worsen_deg": posture_escape_max_orientation_worsen_deg,
                "local_step_wrist_unload_active": bool(wrist_unload_active),
                "local_step_wrist_unload_improvement_deg": float(wrist_unload_improvement_deg),
                "local_step_wrist_unload_bounded_worse_ok": bool(wrist_unload_bounded_worse_ok),
                "local_step_wrist_unload_max_position_worsen_m": wrist_unload_max_position_worsen_m,
                "local_step_wrist_unload_max_orientation_worsen_deg": wrist_unload_max_orientation_worsen_deg,
            }
        )
        progress_accept_reason = ""
        if pose_fraction <= 1.0:
            if bool(getattr(self.profile, "local_step_accept_worse_within_tolerance", True)):
                progress_accept_reason = "within_pose_tolerance"
            elif progress >= -min_progress:
                progress_accept_reason = "within_pose_tolerance_nonregressing"
            elif limit_escape_within_tolerance_ok:
                progress_accept_reason = "within_pose_tolerance_joint_limit_escape"
        elif progress >= min_progress:
            progress_accept_reason = "combined_pose_score_progress"
        else:
            if balanced_progress_ok:
                progress_accept_reason = "balanced_pose_score_progress"
            elif position_progress_ok:
                progress_accept_reason = "position_progress_with_bounded_orientation_worsening"
            elif orientation_progress_ok:
                progress_accept_reason = "orientation_progress_with_bounded_position_worsening"
            elif orientation_recovery_ok:
                progress_accept_reason = "orientation_recovery_bounded_position_worsening"
            elif limit_escape_bounded_worse_ok:
                progress_accept_reason = "joint_limit_escape_bounded_pose_worsening"
            elif singularity_escape_bounded_worse_ok:
                progress_accept_reason = "singularity_escape_bounded_pose_worsening"
            elif posture_escape_bounded_worse_ok:
                progress_accept_reason = "posture_escape_bounded_pose_worsening"
            elif wrist_unload_bounded_worse_ok:
                progress_accept_reason = "wrist_unload_bounded_pose_worsening"
        if not progress_accept_reason:
            if bool(allow_pose_nonprogress_transition):
                progress_accept_reason = "safe_rate_limited_full_target_transition"
                metadata["local_step_nonprogress_transition_candidate"] = True
                metadata["local_step_nonprogress_transition_reason"] = (
                    "full_target_was_accepted_and_rate_limited_step_requires_nonmonotonic_pose_progress"
                )
            else:
                metadata["reject_reason"] = "local_step_no_pose_progress"
                metadata["score"] = _candidate_score(
                    joint_delta_max_deg=float(np.max(np.abs(local_delta_deg))) if local_delta_deg.size else 0.0,
                    joint_delta_norm_deg=float(np.linalg.norm(local_delta_deg)),
                    penalty_deg=score_penalty_deg,
                    pose_fraction=pose_fraction,
                )
                _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
                return False, metadata
        metadata["local_step_progress_accept_reason"] = progress_accept_reason

        singularity_started = time.perf_counter()
        singularity = self.provider.analyze_singularity(qpos)
        singularity_elapsed_ms = _elapsed_ms(singularity_started)
        _timing_add(timing_profile, "singularity_ms", singularity_elapsed_ms)
        _timing_add(timing_profile, f"{timing_phase}_singularity_ms", singularity_elapsed_ms)
        metadata["singularity"] = _singularity_metadata(singularity)
        if self.profile.reject_singular and singularity.is_singular:
            soft_labels = {str(label) for label in self.profile.soft_singularity_labels}
            if singularity.label in soft_labels:
                score_penalty_deg += max(0.0, float(self.profile.soft_singularity_penalty_deg))
                metadata["soft_singularity_candidate"] = True
                metadata["soft_singularity_penalty_deg"] = float(
                    max(0.0, float(self.profile.soft_singularity_penalty_deg))
                )
            else:
                singular_escape_metadata = (
                    None
                    if self.posture_preference is None
                    else self.posture_preference.singular_escape_metadata(
                        reference_qpos=reference_qpos,
                        candidate_qpos=qpos,
                        singularity_label=str(singularity.label),
                    )
                )
                if singular_escape_metadata is not None:
                    metadata["posture_singular_escape"] = singular_escape_metadata
                if singular_escape_metadata is not None and bool(singular_escape_metadata.get("allowed", False)):
                    metadata["singular_escape_allowed"] = True
                    metadata["singular_escape_reason"] = "posture_bad_to_good_crossing"
                    score_penalty_deg += max(0.0, float(self.profile.soft_singularity_penalty_deg))
                else:
                    metadata["reject_reason"] = "singular_candidate"
                    metadata["score"] = 1e18
                    _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
                    return False, metadata

        if self.collision_checker is not None:
            collision_started = time.perf_counter()
            ok, collision_metadata = self.collision_checker.check_candidate(
                qpos,
                metadata={**request_metadata, **metadata},
            )
            collision_elapsed_ms = _elapsed_ms(collision_started)
            _timing_add(timing_profile, "collision_ms", collision_elapsed_ms)
            _timing_add(timing_profile, f"{timing_phase}_collision_ms", collision_elapsed_ms)
            metadata["collision"] = dict(collision_metadata)
            if not ok:
                metadata["reject_reason"] = "collision"
                metadata["score"] = 1e18
                _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
                return False, metadata

        joint_delta_max_deg = float(np.max(np.abs(local_delta_deg))) if local_delta_deg.size else 0.0
        joint_delta_norm_deg = float(np.linalg.norm(local_delta_deg))
        metadata["joint_delta_deg"] = np.abs(local_delta_deg).astype(float).tolist()
        metadata["joint_delta_max_deg"] = joint_delta_max_deg
        metadata["joint_delta_norm_deg"] = joint_delta_norm_deg
        metadata["score_penalty_deg"] = float(score_penalty_deg)
        if bool(metadata.get("local_step_nonprogress_transition_candidate", False)):
            metadata["local_step_nonprogress_transition_used"] = True
        metadata["score_policy"] = "local_step_minimize_pose_error_progress_then_joint_delta"
        recovery_priority_bonus = 0.0
        if progress_accept_reason == "joint_limit_escape_bounded_pose_worsening":
            recovery_priority_bonus = min(
                10_000_000.0,
                5_000_000.0 + max(0.0, limit_escape_margin_progress_deg) * 1_000_000.0,
            )
        elif progress_accept_reason == "singularity_escape_bounded_pose_worsening":
            recovery_priority_bonus = 10_000.0
        elif progress_accept_reason == "posture_escape_bounded_pose_worsening":
            recovery_priority_bonus = min(
                5_000_000.0,
                100_000.0 + max(0.0, posture_improvement_deg) * 100_000.0,
            )
        elif progress_accept_reason == "wrist_unload_bounded_pose_worsening":
            recovery_priority_bonus = min(
                1_000_000.0,
                50_000.0 + max(0.0, wrist_unload_improvement_deg) * 50_000.0,
            )
        elif progress_accept_reason == "orientation_recovery_bounded_position_worsening":
            recovery_priority_bonus = min(20_000.0, max(0.0, orientation_progress_deg) * 100.0)
        metadata["local_step_recovery_priority_bonus"] = float(recovery_priority_bonus)
        metadata["score"] = float(
            after_score * 1_000_000.0
            + after_balanced_score * 10_000.0
            + score_penalty_deg * 10_000.0
            + joint_delta_norm_deg
            + (1.0 - scale) * 10.0
            - recovery_priority_bonus
        )
        metadata["reject_reason"] = None
        _record_candidate_evaluation(timing_profile, timing_phase, evaluate_started)
        return True, metadata

    def _hold_local_step_result(
        self,
        *,
        request: IKResolveRequest,
        full_result: IKResolveResult,
        rate_metadata: dict[str, Any],
        message: str,
    ) -> IKResolveResult:
        reference_qpos = np.asarray(request.reference_qpos, dtype=float).reshape(-1)
        reference_pose = (
            np.asarray(request.reference_pose, dtype=float).reshape(4, 4)
            if request.reference_pose is not None
            else self.provider.forward_kinematics(reference_qpos)
        )
        if reference_pose is None:
            reference_pose = np.asarray(request.target_pose, dtype=float).reshape(4, 4)
        pose_error_m, pose_error_deg = pose_delta_metrics(reference_pose, request.target_pose)
        pose_fraction = _pose_tolerance_fraction(
            pose_error_m=pose_error_m,
            pose_error_deg=pose_error_deg,
            position_tolerance_m=float(self.profile.target_tolerance.position_m),
            orientation_tolerance_deg=float(self.profile.target_tolerance.orientation_deg),
        )
        pose_slack_tier = _pose_slack_tier(pose_fraction)
        metadata = {
            **dict(full_result.metadata),
            **rate_metadata,
            "status": IKResolveStatus.RELAXED.value,
            "source": "generic_ik_resolver",
            "candidate_source": "local_step_hold",
            "seed_source": "reference",
            "method": "hold_reference",
            "result_message": message,
            "local_step_status": "hold_reference",
            "local_step_recovery": True,
            "local_step_recovery_state": "hold",
            "local_step_hold_reason": "no_safe_local_step",
            "local_step_hold_full_target_source": full_result.candidate_source,
            "local_step_hold_full_target_method": str(full_result.metadata.get("method", "")),
            "local_step_output_joint_deg": np.rad2deg(reference_qpos).astype(float).tolist(),
            "pose_error_m": pose_error_m,
            "pose_error_deg": pose_error_deg,
            "pose_tolerance_fraction": pose_fraction,
            "pose_slack_tier": pose_slack_tier,
        }
        return IKResolveResult(
            status=IKResolveStatus.RELAXED,
            qpos=reference_qpos.copy(),
            requested_target_pose=np.asarray(request.target_pose, dtype=float).reshape(4, 4),
            effective_target_pose=reference_pose.copy(),
            tolerance_fraction=max(1.0, float(pose_slack_tier)),
            pose_error_m=pose_error_m,
            pose_error_deg=pose_error_deg,
            candidate_source="local_step_hold",
            message=message,
            metadata=metadata,
        )

    def _with_step_metadata(self, result: IKResolveResult, metadata: dict[str, Any]) -> IKResolveResult:
        return IKResolveResult(
            status=result.status,
            qpos=None if result.qpos is None else result.qpos.copy(),
            requested_target_pose=result.requested_target_pose.copy(),
            effective_target_pose=result.effective_target_pose.copy(),
            tolerance_fraction=result.tolerance_fraction,
            pose_error_m=result.pose_error_m,
            pose_error_deg=result.pose_error_deg,
            candidate_source=result.candidate_source,
            message=result.message,
            metadata={**dict(result.metadata), **metadata},
        )

    def _with_timing_metadata(
        self,
        result: IKResolveResult,
        timing_profile: dict[str, Any] | None,
        started_at: float,
    ) -> IKResolveResult:
        profile = _finalize_timing_profile(timing_profile, started_at)
        if not profile:
            return result
        metadata = dict(result.metadata)
        existing = metadata.get("ik_debug_profile")
        merged_profile = dict(existing) if isinstance(existing, dict) else {}
        merged_profile.update(profile)
        metadata["ik_debug_profile"] = merged_profile
        return IKResolveResult(
            status=result.status,
            qpos=None if result.qpos is None else result.qpos.copy(),
            requested_target_pose=result.requested_target_pose.copy(),
            effective_target_pose=result.effective_target_pose.copy(),
            tolerance_fraction=result.tolerance_fraction,
            pose_error_m=result.pose_error_m,
            pose_error_deg=result.pose_error_deg,
            candidate_source=result.candidate_source,
            message=result.message,
            metadata=metadata,
        )

    def _failed_result(
        self,
        request: IKResolveRequest,
        target_pose: np.ndarray,
        effective_pose: np.ndarray,
        message: str,
        metadata: dict[str, Any],
    ) -> IKResolveResult:
        return IKResolveResult(
            status=IKResolveStatus.FAILED,
            qpos=None,
            requested_target_pose=target_pose,
            effective_target_pose=effective_pose,
            tolerance_fraction=0.0,
            message=message,
            metadata={"profile": self.profile.name, **metadata, **request.metadata},
        )


def _singularity_metadata(report: SingularityReport) -> dict[str, Any]:
    return {
        "status_code": report.status_code,
        "label": report.label,
        "metric": report.metric,
        **dict(report.metadata),
    }


def _singularity_metric_delta(
    reference_report: SingularityReport,
    candidate_report: SingularityReport,
) -> float | None:
    if reference_report.metric is None or candidate_report.metric is None:
        return None
    try:
        reference = float(reference_report.metric)
        candidate = float(candidate_report.metric)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(reference) or not np.isfinite(candidate):
        return None
    return float(candidate - reference)


def _local_step_recovery_state(metadata: dict[str, Any]) -> str:
    reason = str(metadata.get("local_step_progress_accept_reason", ""))
    if reason.startswith("joint_limit_escape"):
        return "joint_limit_escape"
    if reason.startswith("singularity_escape"):
        return "singularity_escape"
    if reason.startswith("posture_escape"):
        return "posture_escape"
    if reason.startswith("wrist_unload"):
        return "wrist_unload"
    if reason.startswith("orientation_recovery"):
        return "orientation_recovery"
    if bool(metadata.get("local_step_nonprogress_transition_used", False)):
        return "rate_limited_transition"
    if bool(metadata.get("local_step_limit_escape_candidate", False)):
        return "joint_limit_escape"
    if bool(metadata.get("local_step_singularity_escape_candidate", False)):
        return "singularity_escape"
    if bool(metadata.get("local_step_posture_escape_candidate", False)):
        return "posture_escape"
    if bool(metadata.get("local_step_wrist_unload_candidate", False)):
        return "wrist_unload"
    return "tracking"


def _local_step_is_auxiliary_recovery_without_pose_progress(metadata: dict[str, Any]) -> bool:
    state = _local_step_recovery_state(metadata)
    if state not in {"posture_escape", "wrist_unload"}:
        return False
    reason = str(metadata.get("local_step_progress_accept_reason", ""))
    if reason in {
        "combined_pose_score_progress",
        "balanced_pose_score_progress",
        "position_progress_with_bounded_orientation_worsening",
        "orientation_progress_with_bounded_position_worsening",
        "orientation_recovery_bounded_position_worsening",
    }:
        return False
    return True


def _local_step_stable_selection_pool(
    evaluated: list[_EvaluatedCandidate],
) -> tuple[list[_EvaluatedCandidate], dict[str, Any]]:
    preferred = [
        item
        for item in evaluated
        if not _local_step_is_auxiliary_recovery_without_pose_progress(item.metadata)
    ]
    if preferred:
        return preferred, {
            "name": "prefer_tracking_over_auxiliary_posture_recovery",
            "accepted_count": len(evaluated),
            "selection_count": len(preferred),
            "posture_wrist_auxiliary_deprioritized_count": len(evaluated) - len(preferred),
            "fallback_to_auxiliary_recovery": False,
        }
    return evaluated, {
        "name": "prefer_tracking_over_auxiliary_posture_recovery",
        "accepted_count": len(evaluated),
        "selection_count": len(evaluated),
        "posture_wrist_auxiliary_deprioritized_count": 0,
        "fallback_to_auxiliary_recovery": True,
    }


def _elapsed_ms(started_at: float) -> float:
    return float((time.perf_counter() - float(started_at)) * 1000.0)


def _new_timing_profile(mode: str) -> dict[str, Any]:
    return {
        "mode": str(mode),
        "provider_calls": [],
        "provider_total_ms": 0.0,
        "provider_call_count": 0,
        "candidate_eval_ms": 0.0,
        "candidate_eval_count": 0,
        "fk_ms": 0.0,
        "singularity_ms": 0.0,
        "collision_ms": 0.0,
    }


def _timing_add(profile: dict[str, Any] | None, key: str, value: float | int | None) -> None:
    if profile is None or value is None:
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return
    if not np.isfinite(numeric):
        return
    profile[str(key)] = float(profile.get(str(key), 0.0) or 0.0) + numeric


def _timing_inc(profile: dict[str, Any] | None, key: str, amount: int = 1) -> None:
    if profile is None:
        return
    profile[str(key)] = int(profile.get(str(key), 0) or 0) + int(amount)


def _record_candidate_evaluation(
    profile: dict[str, Any] | None,
    phase: str,
    started_at: float,
) -> None:
    elapsed_ms = _elapsed_ms(started_at)
    _timing_add(profile, "candidate_eval_ms", elapsed_ms)
    _timing_add(profile, f"{phase}_candidate_eval_ms", elapsed_ms)
    _timing_inc(profile, "candidate_eval_count")
    _timing_inc(profile, f"{phase}_candidate_eval_count")


def _record_provider_call(
    profile: dict[str, Any] | None,
    *,
    phase: str,
    provider_metadata: Any,
    candidate_count: int,
    measured_elapsed_ms: float,
    extra: dict[str, Any] | None = None,
) -> None:
    if profile is None:
        return
    metadata = dict(provider_metadata) if isinstance(provider_metadata, dict) else {}
    elapsed_ms = _finite_float(metadata.get("provider_elapsed_ms"), default=measured_elapsed_ms)
    _timing_add(profile, "provider_total_ms", elapsed_ms)
    _timing_add(profile, f"{phase}_provider_ms", elapsed_ms)
    _timing_inc(profile, "provider_call_count")
    _timing_inc(profile, f"{phase}_provider_call_count")
    _timing_inc(profile, "provider_candidate_count", int(candidate_count))
    _timing_inc(profile, f"{phase}_provider_candidate_count", int(candidate_count))
    provider_time_budget_ms = _finite_float(metadata.get("provider_time_budget_ms"), default=None)
    configured_budget_ms = _finite_float(profile.get("time_budget_ms"), default=None)
    if provider_time_budget_ms is not None:
        profile["provider_time_budget_ms"] = provider_time_budget_ms
        profile["time_budget_ms"] = (
            provider_time_budget_ms
            if configured_budget_ms is None
            else min(configured_budget_ms, provider_time_budget_ms)
        )
    time_budget_ms = _finite_float(profile.get("time_budget_ms"), default=provider_time_budget_ms)
    exceeded = bool(metadata.get("provider_time_budget_exceeded", False))
    profile["time_budget_exceeded"] = bool(profile.get("time_budget_exceeded", False) or exceeded)
    if exceeded:
        _timing_inc(profile, "provider_time_budget_exceeded_count")
        _timing_inc(profile, f"{phase}_provider_time_budget_exceeded_count")
    call: dict[str, Any] = {
        "phase": str(phase),
        "elapsed_ms": float(elapsed_ms),
        "candidate_count": int(candidate_count),
        "requested_provider_names": list(metadata.get("requested_provider_names") or []),
        "provider_budget_skipped_names": list(metadata.get("provider_budget_skipped_names") or []),
        "time_budget_ms": time_budget_ms,
        "provider_time_budget_ms": provider_time_budget_ms,
        "time_budget_exceeded": exceeded,
    }
    if extra:
        call.update(dict(extra))
    calls = profile.setdefault("provider_calls", [])
    if isinstance(calls, list):
        calls.append(call)


def _finalize_timing_profile(
    profile: dict[str, Any] | None,
    started_at: float,
) -> dict[str, Any]:
    if profile is None:
        return {}
    out = dict(profile)
    frame_total_ms = _elapsed_ms(started_at)
    out["frame_total_ms"] = float(frame_total_ms)
    out["solve_total_ms"] = float(frame_total_ms)
    out["provider_total_ms"] = float(out.get("provider_total_ms", 0.0) or 0.0)
    out["non_provider_ms"] = max(0.0, float(frame_total_ms) - float(out["provider_total_ms"]))
    out["fk_ms"] = float(out.get("fk_ms", 0.0) or 0.0)
    out["singularity_ms"] = float(out.get("singularity_ms", 0.0) or 0.0)
    out["collision_ms"] = float(out.get("collision_ms", 0.0) or 0.0)
    out["candidate_eval_ms"] = float(out.get("candidate_eval_ms", 0.0) or 0.0)
    out["provider_call_count"] = int(out.get("provider_call_count", 0) or 0)
    out["candidate_eval_count"] = int(out.get("candidate_eval_count", 0) or 0)
    time_budget_ms = _finite_float(out.get("time_budget_ms"), default=None)
    frame_time_budget_exceeded = bool(
        time_budget_ms is not None
        and float(time_budget_ms) > 0.0
        and float(frame_total_ms) > float(time_budget_ms)
    )
    out["frame_time_budget_exceeded"] = bool(frame_time_budget_exceeded)
    out["time_budget_exceeded"] = bool(out.get("time_budget_exceeded", False) or frame_time_budget_exceeded)
    return out


def _finite_float(value: Any, *, default: float | None) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return float(numeric)


def _local_step_iteration_counts(profile: IKSafetyProfile) -> tuple[int, ...]:
    raw_counts = tuple(int(value) for value in getattr(profile, "local_step_iteration_counts", ()) or ())
    if not raw_counts:
        raw_counts = (int(profile.local_step_max_iterations),)
    counts: list[int] = []
    for value in raw_counts:
        count = max(1, int(value))
        if count in counts:
            continue
        counts.append(count)
    counts.sort()
    return tuple(counts)


def _local_step_scale_factors(profile: IKSafetyProfile) -> tuple[float, ...]:
    raw_scales = tuple(float(value) for value in getattr(profile, "local_step_scale_factors", ()) or ())
    if not raw_scales:
        raw_scales = (1.0,)
    scales: list[float] = []
    for value in raw_scales:
        if not np.isfinite(value) or value <= 0.0:
            continue
        scale = max(1e-6, min(1.0, float(value)))
        if any(abs(scale - existing) < 1e-9 for existing in scales):
            continue
        scales.append(scale)
    if not scales:
        scales = [1.0]
    scales.sort(reverse=True)
    return tuple(scales)


def _local_step_worsen_limits(
    *,
    profile: IKSafetyProfile,
    before_m: float,
    before_deg: float,
) -> dict[str, Any]:
    base_position_m = max(0.0, float(profile.local_step_max_position_worsen_m))
    base_orientation_deg = max(0.0, float(profile.local_step_max_orientation_worsen_deg))
    enabled = bool(getattr(profile, "local_step_dynamic_worsen_enabled", False))
    start_m = max(0.0, float(getattr(profile, "local_step_dynamic_worsen_error_start_m", 0.05)))
    start_deg = max(0.0, float(getattr(profile, "local_step_dynamic_worsen_error_start_deg", 15.0)))
    active = bool(enabled and (float(before_m) >= start_m or float(before_deg) >= start_deg))
    position_m = base_position_m
    orientation_deg = base_orientation_deg
    position_fraction = max(0.0, float(getattr(profile, "local_step_dynamic_worsen_position_fraction", 0.10)))
    orientation_fraction = max(0.0, float(getattr(profile, "local_step_dynamic_worsen_orientation_fraction", 0.10)))
    max_position_m = max(0.0, float(getattr(profile, "local_step_dynamic_worsen_max_position_m", 0.02)))
    max_orientation_deg = max(0.0, float(getattr(profile, "local_step_dynamic_worsen_max_orientation_deg", 8.0)))
    if active:
        position_m = max(base_position_m, max(0.0, float(before_m)) * position_fraction)
        orientation_deg = max(base_orientation_deg, max(0.0, float(before_deg)) * orientation_fraction)
        if max_position_m > 0.0:
            position_m = min(position_m, max_position_m)
        if max_orientation_deg > 0.0:
            orientation_deg = min(orientation_deg, max_orientation_deg)
    return {
        "position_m": float(position_m),
        "orientation_deg": float(orientation_deg),
        "base_position_m": float(base_position_m),
        "base_orientation_deg": float(base_orientation_deg),
        "enabled": bool(enabled),
        "active": bool(active),
        "position_fraction": float(position_fraction),
        "orientation_fraction": float(orientation_fraction),
    }


def _coarse_tracking_acceptance(
    *,
    profile: IKSafetyProfile,
    reference_pose: np.ndarray | None,
    target_pose: np.ndarray,
    pose_error_m: float,
    pose_error_deg: float,
) -> dict[str, Any]:
    enabled = bool(getattr(profile, "coarse_tracking_enabled", False))
    base_position_tolerance_m = max(0.0, float(profile.target_tolerance.position_m))
    base_orientation_tolerance_deg = max(0.0, float(profile.target_tolerance.orientation_deg))
    max_fraction = max(1.0, float(getattr(profile, "coarse_tracking_max_tolerance_fraction", 4.0)))
    configured_max_position_m = max(0.0, float(getattr(profile, "coarse_tracking_max_position_error_m", 0.0)))
    configured_max_orientation_deg = max(
        0.0,
        float(getattr(profile, "coarse_tracking_max_orientation_error_deg", 0.0)),
    )
    max_position_m = (
        configured_max_position_m
        if configured_max_position_m > 0.0
        else base_position_tolerance_m * max_fraction
    )
    max_orientation_deg = (
        configured_max_orientation_deg
        if configured_max_orientation_deg > 0.0
        else base_orientation_tolerance_deg * max_fraction
    )
    after_fraction = _pose_tolerance_fraction(
        pose_error_m=pose_error_m,
        pose_error_deg=pose_error_deg,
        position_tolerance_m=base_position_tolerance_m,
        orientation_tolerance_deg=base_orientation_tolerance_deg,
    )
    metadata: dict[str, Any] = {
        "enabled": bool(enabled),
        "accepted": False,
        "after_position_error_m": float(pose_error_m),
        "after_orientation_error_deg": float(pose_error_deg),
        "after_fraction": float(after_fraction),
        "start_fraction": float(getattr(profile, "coarse_tracking_start_fraction", 1.0)),
        "max_tolerance_fraction": float(max_fraction),
        "max_position_error_m": float(max_position_m),
        "max_orientation_error_deg": float(max_orientation_deg),
        "min_fraction_progress": float(getattr(profile, "coarse_tracking_min_fraction_progress", 0.05)),
    }
    if not enabled:
        metadata["reject_reason"] = "disabled"
        return metadata
    if reference_pose is None:
        metadata["reject_reason"] = "reference_pose_unavailable"
        return metadata
    before_m, before_deg = pose_delta_metrics(reference_pose, target_pose)
    before_fraction = _pose_tolerance_fraction(
        pose_error_m=before_m,
        pose_error_deg=before_deg,
        position_tolerance_m=base_position_tolerance_m,
        orientation_tolerance_deg=base_orientation_tolerance_deg,
    )
    before_balanced = _pose_balanced_progress_score(
        pose_error_m=before_m,
        pose_error_deg=before_deg,
        position_tolerance_m=base_position_tolerance_m,
        orientation_tolerance_deg=base_orientation_tolerance_deg,
    )
    after_balanced = _pose_balanced_progress_score(
        pose_error_m=pose_error_m,
        pose_error_deg=pose_error_deg,
        position_tolerance_m=base_position_tolerance_m,
        orientation_tolerance_deg=base_orientation_tolerance_deg,
    )
    progress_fraction = float(before_fraction) - float(after_fraction)
    balanced_progress = float(before_balanced) - float(after_balanced)
    start_fraction = max(0.0, float(metadata["start_fraction"]))
    min_progress = max(0.0, float(metadata["min_fraction_progress"]))
    worsen_limits = _local_step_worsen_limits(
        profile=profile,
        before_m=before_m,
        before_deg=before_deg,
    )
    position_progress_m = float(before_m) - float(pose_error_m)
    orientation_progress_deg = float(before_deg) - float(pose_error_deg)
    component_worsen_ok = (
        position_progress_m >= -float(worsen_limits["position_m"])
        and orientation_progress_deg >= -float(worsen_limits["orientation_deg"])
    )
    orientation_recovery_max_position_worsen_m = max(
        0.0,
        float(getattr(profile, "coarse_tracking_orientation_recovery_max_position_worsen_m", 0.08)),
    )
    orientation_recovery_ok = (
        bool(getattr(profile, "coarse_tracking_orientation_recovery_enabled", False))
        and orientation_progress_deg
        >= max(0.0, float(getattr(profile, "coarse_tracking_orientation_recovery_min_progress_deg", 15.0)))
        and position_progress_m >= -orientation_recovery_max_position_worsen_m
    )
    error_cap_ok = (
        (max_position_m <= 0.0 or float(pose_error_m) <= max_position_m)
        and (max_orientation_deg <= 0.0 or float(pose_error_deg) <= max_orientation_deg)
        and float(after_fraction) <= max_fraction
    )
    progress_ok = progress_fraction >= min_progress or balanced_progress >= min_progress
    far_enough = float(before_fraction) >= start_fraction
    metadata.update(
        {
            "before_position_error_m": float(before_m),
            "before_orientation_error_deg": float(before_deg),
            "before_fraction": float(before_fraction),
            "before_balanced_fraction": float(before_balanced),
            "after_balanced_fraction": float(after_balanced),
            "progress_fraction": float(progress_fraction),
            "balanced_progress_fraction": float(balanced_progress),
            "position_progress_m": float(position_progress_m),
            "orientation_progress_deg": float(orientation_progress_deg),
            "component_worsen_ok": bool(component_worsen_ok),
            "orientation_recovery_ok": bool(orientation_recovery_ok),
            "orientation_recovery_max_position_worsen_m": float(orientation_recovery_max_position_worsen_m),
            "error_cap_ok": bool(error_cap_ok),
            "progress_ok": bool(progress_ok),
            "far_enough": bool(far_enough),
        }
    )
    if not far_enough:
        metadata["reject_reason"] = "target_gap_below_start_fraction"
        return metadata
    if not error_cap_ok:
        metadata["reject_reason"] = "coarse_error_cap_exceeded"
        return metadata
    if not component_worsen_ok and not orientation_recovery_ok:
        metadata["reject_reason"] = "component_worsen_exceeded"
        return metadata
    if not progress_ok:
        metadata["reject_reason"] = "insufficient_progress"
        return metadata
    metadata["accepted"] = True
    metadata["accept_reason"] = (
        "coarse_orientation_recovery"
        if orientation_recovery_ok and not component_worsen_ok
        else "coarse_tracking_progress"
    )
    return metadata


def _joint_position_limits_from_provider(
    provider: KinematicsProvider,
    qpos_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    getter = getattr(provider, "joint_position_limits", None)
    if getter is None:
        return None
    try:
        limits = getter()
    except Exception:
        return None
    if limits is None:
        return None
    try:
        q_min = np.asarray(limits[0], dtype=float).reshape(qpos_shape)
        q_max = np.asarray(limits[1], dtype=float).reshape(qpos_shape)
    except Exception:
        return None
    if not np.all(np.isfinite(q_min)) or not np.all(np.isfinite(q_max)):
        return None
    if np.any(q_max <= q_min):
        return None
    return q_min.copy(), q_max.copy()


def _candidate_score(
    *,
    joint_delta_max_deg: float,
    joint_delta_norm_deg: float,
    penalty_deg: float,
    pose_fraction: float,
) -> float:
    return float(
        (float(joint_delta_max_deg) + max(0.0, float(penalty_deg))) * 1_000_000.0
        + float(joint_delta_norm_deg)
        + max(0.0, float(pose_fraction)) * 1_000.0
    )


def _pose_tolerance_fraction(
    *,
    pose_error_m: float,
    pose_error_deg: float,
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
) -> float:
    fractions: list[float] = []
    if position_tolerance_m > 0.0:
        fractions.append(max(0.0, float(pose_error_m)) / float(position_tolerance_m))
    if orientation_tolerance_deg > 0.0:
        fractions.append(max(0.0, float(pose_error_deg)) / float(orientation_tolerance_deg))
    return max(fractions) if fractions else 0.0


def _pose_progress_score(
    *,
    pose_error_m: float,
    pose_error_deg: float,
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
) -> float:
    values: list[float] = []
    if position_tolerance_m > 0.0:
        values.append(max(0.0, float(pose_error_m)) / float(position_tolerance_m))
    else:
        values.append(max(0.0, float(pose_error_m)))
    if orientation_tolerance_deg > 0.0:
        values.append(max(0.0, float(pose_error_deg)) / float(orientation_tolerance_deg))
    else:
        values.append(max(0.0, float(pose_error_deg)))
    return max(values) if values else 0.0


def _pose_balanced_progress_score(
    *,
    pose_error_m: float,
    pose_error_deg: float,
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
) -> float:
    values: list[float] = []
    if position_tolerance_m > 0.0:
        values.append(max(0.0, float(pose_error_m)) / float(position_tolerance_m))
    else:
        values.append(max(0.0, float(pose_error_m)))
    if orientation_tolerance_deg > 0.0:
        values.append(max(0.0, float(pose_error_deg)) / float(orientation_tolerance_deg))
    else:
        values.append(max(0.0, float(pose_error_deg)))
    return float(sum(values))


def _pose_slack_tier(fraction: float) -> float:
    value = max(0.0, float(fraction))
    if value <= 1e-4:
        return 0.0
    if value <= 0.25:
        return 0.25
    if value <= 0.5:
        return 0.5
    return 1.0
