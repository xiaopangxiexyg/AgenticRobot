from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import robot_motion.ik_resolver as ik_resolver_module
from robot_motion.ik_resolver import IKSafetyResolver
from robot_motion.joint_limits import JointVelocityLimitProfile
from robot_motion.profiles import IKSafetyProfile
from robot_motion.seed_expansion import SeedExpansionProfile, SeedPerturbation
from robot_motion.types import IKCandidate, IKResolveRequest, IKResolveStatus, PoseTolerance, SingularityReport


class FakeProvider:
    def __init__(self):
        self.calls: list[float] = []
        self.soft_singularity_case = False

    def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
        fraction = float(options["tolerance_fraction"])
        self.calls.append(fraction)
        self.soft_singularity_case = options.get("case") in {
            "soft_singularity",
            "soft_singularity_only",
        }
        if options.get("case") == "hard_branch":
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
                    source="fake",
                    seed_source="reference",
                    method="bad_branch",
                ),
                IKCandidate(
                    qpos=np.deg2rad(np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
                    source="fake",
                    seed_source="reference",
                    method="safe_branch",
                ),
            ]
        if options.get("case") == "soft_singularity":
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
                    source="fake",
                    seed_source="reference",
                    method="wrist_singular_but_close",
                ),
                IKCandidate(
                    qpos=np.deg2rad(np.array([20.0, 0.0, 0.0, 0.0, 15.0, 0.0])),
                    source="fake",
                    seed_source="reference",
                    method="normal_farther",
                ),
            ]
        if options.get("case") == "soft_singularity_only":
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
                    source="fake",
                    seed_source="reference",
                    method="wrist_singular_fallback",
                )
            ]
        if options.get("case") == "pose_slack":
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([2.0, 0.0, 0.0, 0.0, 10.0, 0.0])),
                    source="fake",
                    seed_source="reference",
                    method="soft_pose",
                )
            ]
        if options.get("case") == "relax" and fraction >= 0.5:
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([4.0, 3.0, 2.0, 0.0, 0.0, 0.0])),
                    source="fake",
                    seed_source="reference",
                    method="relaxed",
                )
            ]
        return [
            IKCandidate(
                qpos=np.deg2rad(np.array([120.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
                source="fake",
                seed_source="reference",
                method="unsafe",
            )
        ]

    def forward_kinematics(self, qpos):
        return np.eye(4)

    def analyze_singularity(self, qpos):
        if self.soft_singularity_case and abs(float(np.rad2deg(qpos[4]))) < 1.0:
            return SingularityReport(status_code=-3, label="wrist_singularity", metric=0.0)
        return SingularityReport(status_code=0, label="normal", metric=1.0)


class FullTargetProvider:
    def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
        if options.get("local_step_mode"):
            return []
        return [
            IKCandidate(
                qpos=np.deg2rad(np.array([0.0, 0.0, 0.0, 80.0, 0.0, 0.0])),
                source="full",
                seed_source=seed_labels[0],
                method="full_target",
            )
        ]

    def forward_kinematics(self, qpos):
        pose = np.eye(4)
        pose[0, 3] = float(np.rad2deg(qpos[3])) * 0.01
        return pose

    def analyze_singularity(self, qpos):
        return SingularityReport(status_code=0, label="normal", metric=1.0)


class LocalStepProvider(FullTargetProvider):
    def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
        if not options.get("local_step_mode"):
            return []
        return [
            IKCandidate(
                qpos=np.deg2rad(np.array([0.0, 0.0, 0.0, 6.0, 0.0, 0.0])),
                source="local",
                seed_source=seed_labels[0],
                method="local_step",
            )
        ]


class CoarsePoseProvider:
    def __init__(self, *, candidate_x_m: float):
        self.candidate_x_m = float(candidate_x_m)

    def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
        del target_pose, options
        return [
            IKCandidate(
                qpos=np.deg2rad(np.array([0.0, 0.0, 0.0, self.candidate_x_m * 100.0, 0.0, 0.0])),
                source="coarse",
                seed_source=seed_labels[0],
                method="coarse_candidate",
            )
        ]

    def forward_kinematics(self, qpos):
        pose = np.eye(4)
        pose[0, 3] = float(np.rad2deg(np.asarray(qpos)[3])) * 0.01
        return pose

    def analyze_singularity(self, qpos):
        return SingularityReport(status_code=0, label="normal", metric=1.0)


class NoFullCandidateLocalStepProvider(FullTargetProvider):
    def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
        if not options.get("local_step_mode"):
            return []
        return [
            IKCandidate(
                qpos=np.deg2rad(np.array([0.0, 0.0, 0.0, 6.0, 0.0, 0.0])),
                source="local",
                seed_source=seed_labels[0],
                method="local_step",
            )
        ]


class CompositeLikeProvider:
    def __init__(self, providers, names):
        self.providers = tuple(providers)
        self.names = tuple(names)
        self.last_metadata = {}

    def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
        allow = options.get("provider_name_allowlist")
        allow = set(allow or self.names)
        out = []
        requested = []
        for name, provider in zip(self.names, self.providers):
            if name not in allow:
                continue
            requested.append(name)
            out.extend(provider.iter_ik_candidates(target_pose=target_pose, seeds=seeds, seed_labels=seed_labels, options=options))
        self.last_metadata = {"requested_provider_names": requested}
        return out

    def forward_kinematics(self, qpos):
        return self.providers[0].forward_kinematics(qpos)

    def analyze_singularity(self, qpos):
        return self.providers[0].analyze_singularity(qpos)


def _request(case: str) -> IKResolveRequest:
    target = np.eye(4)
    target[0, 3] = 0.05
    return IKResolveRequest(
        target_pose=target,
        reference_qpos=np.zeros(6),
        reference_pose=np.eye(4),
        metadata={"case": case},
    )


def _rm65_posture_metadata() -> dict[str, object]:
    return {
        "joint_posture_preference": {
            "enabled": True,
            "name": "rm65_elbow_good_branch",
            "j2_index": 1,
            "j3_index": 2,
            "j2_high_deg": 70.0,
            "j2_preferred_min_deg": -20.0,
            "j2_preferred_max_deg": 60.0,
            "j3_zero_margin_deg": 10.0,
            "j3_low_deg": 30.0,
            "j3_bad_max_deg": 60.0,
            "j3_preferred_min_deg": 80.0,
            "j3_preferred_max_deg": 130.0,
            "bad_branch_penalty_deg": 180.0,
            "j3_negative_penalty_deg": 160.0,
            "high_j2_low_j3_penalty_deg": 120.0,
            "hard_branch_escape_enabled": True,
            "hard_branch_escape_allowed_joint_indices": [1, 2],
            "hard_branch_escape_min_improvement_deg": 40.0,
            "singular_escape_enabled": True,
            "singular_escape_labels": ["elbow_singularity"],
            "local_escape_enabled": True,
            "local_escape_max_candidates": 6,
            "local_escape_max_position_worsen_m": 0.04,
            "local_escape_max_orientation_worsen_deg": 15.0,
            "local_escape_min_improvement_deg": 1.0,
            "wrist_unload_enabled": True,
            "wrist_unload_j1_index": 0,
            "wrist_unload_j4_index": 3,
            "wrist_unload_j5_index": 4,
            "wrist_unload_j6_index": 5,
            "wrist_unload_j4_abs_activation_deg": 160.0,
            "wrist_unload_j5_abs_activation_deg": 75.0,
            "wrist_unload_j6_wrap_activation_deg": 165.0,
            "wrist_unload_step_fraction": 1.0,
            "wrist_unload_max_candidates": 8,
            "wrist_unload_max_position_worsen_m": 0.05,
            "wrist_unload_max_orientation_worsen_deg": 20.0,
            "wrist_unload_min_cost_improvement_deg": 0.5,
        }
    }


def test_hard_branch_jump_rejected_and_safe_candidate_selected():
    provider = FakeProvider()
    profile = IKSafetyProfile(
        name="test",
        hard_branch_joint_indices=(0, 1, 2),
        hard_branch_max_delta_deg=(90.0, 90.0, 90.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
    )
    result = IKSafetyResolver(profile=profile, provider=provider).resolve(_request("hard_branch"))

    assert result.status is IKResolveStatus.PASS
    assert result.qpos is not None
    assert np.allclose(np.rad2deg(result.qpos), [5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert result.metadata["rejection_count"] == 1


def test_rm65_posture_preference_prioritizes_good_branch_and_allows_j23_escape():
    class RM65BranchProvider:
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            del target_pose, seeds, seed_labels, options
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([0.0, 100.0, -80.0, 0.0, 0.0, 0.0])),
                    source="fake",
                    seed_source="reference",
                    method="bad_branch_close",
                ),
                IKCandidate(
                    qpos=np.deg2rad(np.array([0.0, 20.0, 130.0, 0.0, 0.0, 0.0])),
                    source="fake",
                    seed_source="reference",
                    method="good_branch_far",
                ),
            ]

        def forward_kinematics(self, qpos):
            return np.eye(4)

        def analyze_singularity(self, qpos):
            return SingularityReport(status_code=0, label="normal", metric=1.0)

    reference = np.deg2rad(np.array([0.0, 100.0, -90.0, 0.0, 0.0, 0.0]))
    profile = IKSafetyProfile(
        name="rm65_test",
        hard_branch_joint_indices=(0, 1, 2),
        hard_branch_max_delta_deg=(90.0, 90.0, 90.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        metadata=_rm65_posture_metadata(),
    )

    result = IKSafetyResolver(profile=profile, provider=RM65BranchProvider()).resolve(
        IKResolveRequest(
            target_pose=np.eye(4),
            reference_qpos=reference,
            reference_pose=np.eye(4),
        )
    )

    assert result.status is IKResolveStatus.PASS
    assert result.qpos is not None
    assert np.allclose(np.rad2deg(result.qpos), [0.0, 20.0, 130.0, 0.0, 0.0, 0.0])
    assert result.metadata["method"] == "good_branch_far"
    assert result.metadata["hard_branch_escape_allowed"] is True
    assert result.metadata["posture_preference"]["candidate"]["preferred"] is True


def test_resolver_timing_profile_records_provider_allowlist():
    provider = CompositeLikeProvider(
        providers=(FakeProvider(), FakeProvider()),
        names=("official", "external"),
    )
    profile = IKSafetyProfile(
        name="test",
        hard_branch_joint_indices=(0, 1, 2),
        hard_branch_max_delta_deg=(90.0, 90.0, 90.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
    )
    request = _request("hard_branch")
    request.metadata["provider_name_allowlist"] = ["external"]

    result = IKSafetyResolver(profile=profile, provider=provider).resolve(request)

    timing = result.metadata["ik_debug_profile"]
    assert timing["mode"] == "resolve"
    assert timing["provider_call_count"] == 1
    assert timing["provider_calls"][0]["requested_provider_names"] == ["external"]
    assert timing["full_candidate_count"] == 2
    assert timing["candidate_eval_count"] == 2
    assert timing["provider_total_ms"] >= 0.0
    assert timing["non_provider_ms"] >= 0.0


def test_soft_singularity_is_penalized_instead_of_rejected():
    provider = FakeProvider()
    profile = IKSafetyProfile(
        name="test",
        soft_singularity_labels=("wrist_singularity",),
        soft_singularity_penalty_deg=25.0,
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
    )
    result = IKSafetyResolver(profile=profile, provider=provider).resolve(_request("soft_singularity"))

    assert result.status is IKResolveStatus.PASS
    assert result.qpos is not None
    assert np.allclose(np.rad2deg(result.qpos), [20.0, 0.0, 0.0, 0.0, 15.0, 0.0])
    assert result.metadata["method"] == "normal_farther"
    assert result.metadata["rejection_count"] == 0


def test_soft_singularity_can_be_used_as_fallback_candidate():
    provider = FakeProvider()
    profile = IKSafetyProfile(
        name="test",
        soft_singularity_labels=("wrist_singularity",),
        soft_singularity_penalty_deg=25.0,
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
    )
    result = IKSafetyResolver(profile=profile, provider=provider).resolve(_request("soft_singularity_only"))

    assert result.status is IKResolveStatus.PASS
    assert result.qpos is not None
    assert np.allclose(np.rad2deg(result.qpos), [2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert result.metadata["soft_singularity_candidate"] is True


def test_relaxation_is_staged_to_half_limit_before_acceptance():
    provider = FakeProvider()
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.04, orientation_deg=8.0),
        relaxation_fractions=(0.0, 0.25, 0.5, 1.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
    )
    result = IKSafetyResolver(profile=profile, provider=provider).resolve(_request("relax"))

    assert result.status is IKResolveStatus.RELAXED
    assert result.tolerance_fraction == 0.5
    assert provider.calls == [0.0, 0.25, 0.5]


def test_pose_slack_can_accept_candidate_without_staged_resolve_calls():
    provider = FakeProvider()
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.10, orientation_deg=8.0),
        relaxation_fractions=(0.0, 0.25, 0.5, 1.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
    )
    result = IKSafetyResolver(profile=profile, provider=provider).resolve(_request("pose_slack"))

    assert result.status is IKResolveStatus.RELAXED
    assert result.tolerance_fraction == 0.5
    assert result.metadata["pose_slack_tier"] == 0.5
    assert provider.calls == [0.0]


def test_resolver_expands_seed_labels_for_candidate_providers():
    class EchoSeedProvider:
        def __init__(self):
            self.seen_labels: tuple[str, ...] = ()

        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            self.seen_labels = tuple(seed_labels)
            return [
                IKCandidate(
                    qpos=np.asarray(seed, dtype=float).reshape(6),
                    source="echo",
                    seed_source=str(label),
                    method="seed_echo",
                )
                for seed, label in zip(seeds, seed_labels)
            ]

        def forward_kinematics(self, qpos):
            return np.eye(4)

        def analyze_singularity(self, qpos):
            return SingularityReport(status_code=0, label="normal", metric=1.0)

    provider = EchoSeedProvider()
    seed_profile = SeedExpansionProfile(
        name="test",
        dof=6,
        max_extra_seeds=1,
        source_label_allowlist=("reference",),
        perturbations=(
            SeedPerturbation(label="small_j1_pos", offset_deg=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ),
    )
    profile = IKSafetyProfile(
        name="test",
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
    )

    result = IKSafetyResolver(
        profile=profile,
        provider=provider,
        seed_expansion_profile=seed_profile,
    ).resolve(_request("none"))

    assert result.status is IKResolveStatus.PASS
    assert "reference:small_j1_pos" in provider.seen_labels
    assert result.metadata["seed_expansion"]["generated_labels"] == ["reference:small_j1_pos"]


def test_nearly_exact_pose_error_does_not_mark_relaxed_slack():
    provider = FakeProvider()
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.10, orientation_deg=8.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
    )
    result = IKSafetyResolver(profile=profile, provider=provider).resolve(
        IKResolveRequest(
            target_pose=np.eye(4),
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
            metadata={"case": "hard_branch"},
        )
    )

    assert result.status is IKResolveStatus.PASS
    assert result.metadata["pose_slack_tier"] == 0.0


def test_resolve_step_uses_local_provider_for_large_rate_limited_target():
    provider = CompositeLikeProvider(
        providers=(FullTargetProvider(), LocalStepProvider()),
        names=("full_provider", "local_provider"),
    )
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.05, orientation_deg=10.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_trigger_ratio=4.0,
        local_step_max_iterations=1,
    )
    target = np.eye(4)
    target[0, 3] = 0.8
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(1000.0, 1000.0, 1000.0, 10.0, 1000.0, 1000.0),
    )

    result = IKSafetyResolver(profile=profile, provider=provider).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local_provider",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.message == "ik_local_step_recovery"
    assert result.qpos is not None
    assert np.allclose(np.rad2deg(result.qpos), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert result.metadata["local_step_status"] == "active"
    assert result.metadata["local_step_full_target_ratio"] == 80.0
    assert result.metadata["local_step_pose_error_before_m"] == 0.8
    assert result.metadata["pose_error_m"] == 0.79


def test_resolve_step_local_first_skips_full_resolve_when_progress_is_safe():
    provider = CompositeLikeProvider(
        providers=(FullTargetProvider(), LocalStepProvider()),
        names=("full_provider", "local_provider"),
    )
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.05, orientation_deg=10.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_trigger_ratio=4.0,
        local_step_max_iterations=1,
    )
    target = np.eye(4)
    target[0, 3] = 0.8
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(1000.0, 1000.0, 1000.0, 10.0, 1000.0, 1000.0),
    )

    result = IKSafetyResolver(profile=profile, provider=provider).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local_provider",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.message == "ik_local_step_first"
    assert result.qpos is not None
    assert np.allclose(np.rad2deg(result.qpos), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert result.metadata["local_step_first_status"] == "accepted"
    assert result.metadata["local_step_seed_expansion"]["generated_count"] == 0
    assert provider.last_metadata["requested_provider_names"] == ["local_provider"]


def test_resolve_step_periodic_full_solve_skips_when_far_tracking_is_safe():
    provider = CompositeLikeProvider(
        providers=(FullTargetProvider(), LocalStepProvider()),
        names=("full_provider", "local_provider"),
    )
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.05, orientation_deg=10.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        provider_time_budget_ms=35.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_first_full_solve_interval=1,
        local_step_first_skip_full_when_far=True,
        local_step_first_skip_full_pose_fraction=1.0,
        local_step_trigger_ratio=4.0,
        local_step_max_iterations=1,
    )
    target = np.eye(4)
    target[0, 3] = 0.8
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(1000.0, 1000.0, 1000.0, 10.0, 1000.0, 1000.0),
    )

    result = IKSafetyResolver(profile=profile, provider=provider).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local_provider",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.message == "ik_local_step_first"
    assert result.metadata["local_step_periodic_full_solve_skipped"] is True
    assert result.metadata["local_step_periodic_full_solve_skip_reason"] == "far_tracking_local_step_accepted"
    assert provider.last_metadata["requested_provider_names"] == ["local_provider"]


def test_resolve_step_local_only_never_falls_back_to_full_search():
    class RecordingProvider(CompositeLikeProvider):
        def __init__(self):
            super().__init__(
                providers=(FullTargetProvider(), FullTargetProvider()),
                names=("full_provider", "local_provider"),
            )
            self.requested: list[tuple[bool, tuple[str, ...]]] = []

        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            allow = tuple(options.get("provider_name_allowlist") or ())
            self.requested.append((bool(options.get("local_step_mode")), allow))
            return super().iter_ik_candidates(
                target_pose=target_pose,
                seeds=seeds,
                seed_labels=seed_labels,
                options=options,
            )

    provider = RecordingProvider()
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.05, orientation_deg=10.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_first_full_solve_interval=1,
        local_step_trigger_ratio=4.0,
        local_step_max_iterations=1,
    )
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(1000.0,) * 6,
    )

    result = IKSafetyResolver(profile=profile, provider=provider).resolve_step(
        _request("local_only"),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local_provider",),
        allow_full_solve=False,
    )

    assert result.metadata["full_solve_deferred_to_external_scheduler"] is True
    assert result.metadata["local_step_status"] == "no_safe_local_candidate_hold_reference"
    assert provider.requested == [(True, ("local_provider",))]


def test_resolve_step_generates_rm65_posture_escape_candidate_from_bad_branch():
    class NoIKPostureProvider:
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            del target_pose, seeds, seed_labels, options
            return []

        def forward_kinematics(self, qpos):
            return np.eye(4)

        def analyze_singularity(self, qpos):
            return SingularityReport(status_code=0, label="normal", metric=1.0)

    reference = np.deg2rad(np.array([0.0, 100.0, -90.0, 0.0, 0.0, 0.0]))
    target = np.eye(4)
    target[0, 3] = 0.5
    profile = IKSafetyProfile(
        name="rm65_local_test",
        target_tolerance=PoseTolerance(position_m=0.02, orientation_deg=5.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_accept_worse_within_tolerance=False,
        local_step_hold_on_failure=True,
        metadata=_rm65_posture_metadata(),
    )
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(1000.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0),
    )

    result = IKSafetyResolver(profile=profile, provider=NoIKPostureProvider()).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=reference,
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local_provider",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.qpos is not None
    q_deg = np.rad2deg(result.qpos)
    assert q_deg[2] > -90.0
    assert result.candidate_source == "posture_escape"
    assert result.metadata["local_step_recovery_state"] == "posture_escape"
    assert result.metadata["local_step_progress_accept_reason"] == "posture_escape_bounded_pose_worsening"


def test_resolve_step_generates_rm65_wrist_unload_candidate_near_wrist_boundary():
    class NoIKProvider:
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            del target_pose, seeds, seed_labels, options
            return []

        def forward_kinematics(self, qpos):
            return np.eye(4)

        def analyze_singularity(self, qpos):
            return SingularityReport(status_code=0, label="normal", metric=1.0)

        def joint_position_limits(self):
            return np.deg2rad(np.array([-178.0, -130.0, -135.0, -178.0, -128.0, -360.0])), np.deg2rad(
                np.array([178.0, 130.0, 135.0, 178.0, 128.0, 360.0])
            )

    reference = np.deg2rad(np.array([-96.0, 50.0, 120.0, 177.0, 88.0, -170.0]))
    target = np.eye(4)
    target[0, 3] = 0.5
    profile = IKSafetyProfile(
        name="rm65_wrist_test",
        target_tolerance=PoseTolerance(position_m=0.02, orientation_deg=5.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_accept_worse_within_tolerance=False,
        local_step_hold_on_failure=True,
        metadata=_rm65_posture_metadata(),
    )
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(20.0, 10.0, 10.0, 20.0, 20.0, 20.0),
    )

    result = IKSafetyResolver(profile=profile, provider=NoIKProvider()).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=reference,
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local_provider",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.qpos is not None
    assert result.candidate_source == "wrist_unload"
    assert result.metadata["local_step_recovery_state"] == "wrist_unload"
    assert result.metadata["local_step_progress_accept_reason"] == "wrist_unload_bounded_pose_worsening"


def test_resolve_step_prefers_tracking_candidate_over_wrist_unload_auxiliary_recovery():
    class TrackingProvider:
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            del target_pose, seeds
            if not options.get("local_step_mode"):
                return []
            return [
                IKCandidate(
                    qpos=reference.copy(),
                    source="local_tracking",
                    seed_source=seed_labels[0],
                    method="hold_tracking",
                )
            ]

        def forward_kinematics(self, qpos):
            return np.eye(4)

        def analyze_singularity(self, qpos):
            return SingularityReport(status_code=0, label="normal", metric=1.0)

        def joint_position_limits(self):
            return np.deg2rad(np.array([-178.0, -130.0, -135.0, -178.0, -128.0, -360.0])), np.deg2rad(
                np.array([178.0, 130.0, 135.0, 178.0, 128.0, 360.0])
            )

    reference = np.deg2rad(np.array([-96.0, 50.0, 120.0, 177.0, 88.0, -170.0]))
    profile = IKSafetyProfile(
        name="rm65_wrist_stable_selection_test",
        target_tolerance=PoseTolerance(position_m=0.02, orientation_deg=5.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_accept_worse_within_tolerance=False,
        local_step_scale_factors=(1.0,),
        metadata=_rm65_posture_metadata(),
    )
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(20.0, 10.0, 10.0, 20.0, 20.0, 20.0),
    )

    result = IKSafetyResolver(profile=profile, provider=TrackingProvider()).resolve_step(
        IKResolveRequest(
            target_pose=np.eye(4),
            reference_qpos=reference,
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local_provider",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.candidate_source == "local_tracking"
    assert result.metadata["local_step_recovery_state"] == "tracking"
    assert (
        result.metadata["local_step_selection_policy"]["posture_wrist_auxiliary_deprioritized_count"]
        > 0
    )


def test_full_resolve_accepts_coarse_tracking_candidate_when_far_and_improving():
    provider = CoarsePoseProvider(candidate_x_m=0.11)
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.05, orientation_deg=10.0),
        fk_error_tolerance_m=0.01,
        fk_error_tolerance_deg=1.0,
        coarse_tracking_enabled=True,
        coarse_tracking_start_fraction=1.0,
        coarse_tracking_max_tolerance_fraction=3.0,
        coarse_tracking_min_fraction_progress=0.05,
    )
    target = np.eye(4)
    target[0, 3] = 0.20

    result = IKSafetyResolver(profile=profile, provider=provider).resolve(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        )
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.message == "ik_solution_coarse_tracking"
    assert result.qpos is not None
    assert result.metadata["coarse_tracking_accepted"] is True
    assert np.isclose(result.metadata["coarse_tracking"]["before_fraction"], 4.0)
    assert np.isclose(result.metadata["coarse_tracking"]["after_fraction"], 1.8)
    assert result.metadata["coarse_tracking"]["progress_ok"] is True


def test_full_resolve_rejects_coarse_tracking_candidate_without_progress():
    provider = CoarsePoseProvider(candidate_x_m=0.0)
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.05, orientation_deg=10.0),
        fk_error_tolerance_m=0.01,
        fk_error_tolerance_deg=1.0,
        coarse_tracking_enabled=True,
        coarse_tracking_start_fraction=1.0,
        coarse_tracking_max_tolerance_fraction=6.0,
        coarse_tracking_max_position_error_m=0.30,
        coarse_tracking_min_fraction_progress=0.05,
    )
    target = np.eye(4)
    target[0, 3] = 0.20

    result = IKSafetyResolver(profile=profile, provider=provider).resolve(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        )
    )

    assert result.status is IKResolveStatus.SMOOTHING_REQUIRED
    assert result.metadata["best_rejected"]["reject_reason"] == "fk_pose_error"
    assert result.metadata["best_rejected"]["coarse_tracking"]["reject_reason"] == "insufficient_progress"


def test_full_resolve_accepts_coarse_orientation_recovery_with_bounded_position_worsening():
    class CoarseOrientationRecoveryProvider:
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            del target_pose, seeds, options
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])),
                    source="coarse",
                    seed_source=seed_labels[0],
                    method="orientation_recovery",
                )
            ]

        def forward_kinematics(self, qpos):
            pose = np.eye(4)
            pose[0, 3] = 0.05
            return pose

        def analyze_singularity(self, qpos):
            return SingularityReport(status_code=0, label="normal", metric=1.0)

    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.02, orientation_deg=10.0),
        fk_error_tolerance_m=0.001,
        fk_error_tolerance_deg=1.0,
        coarse_tracking_enabled=True,
        coarse_tracking_max_position_error_m=0.10,
        coarse_tracking_max_orientation_error_deg=90.0,
        coarse_tracking_orientation_recovery_enabled=True,
        coarse_tracking_orientation_recovery_min_progress_deg=15.0,
        coarse_tracking_orientation_recovery_max_position_worsen_m=0.06,
    )
    target = np.eye(4)
    reference_pose = np.eye(4)
    reference_pose[:3, :3] = R.from_euler("z", 60.0, degrees=True).as_matrix()

    result = IKSafetyResolver(profile=profile, provider=CoarseOrientationRecoveryProvider()).resolve(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=reference_pose,
        )
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.metadata["coarse_tracking_accepted"] is True
    assert result.metadata["coarse_tracking"]["accept_reason"] == "coarse_orientation_recovery"
    assert result.metadata["coarse_tracking"]["component_worsen_ok"] is False
    assert result.metadata["coarse_tracking"]["orientation_recovery_ok"] is True


def test_periodic_full_solve_failure_falls_back_to_safe_local_step():
    provider = CompositeLikeProvider(
        providers=(NoFullCandidateLocalStepProvider(), NoFullCandidateLocalStepProvider()),
        names=("full_provider", "local_provider"),
    )
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.05, orientation_deg=10.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_first_full_solve_interval=1,
        local_step_trigger_ratio=4.0,
        local_step_max_iterations=1,
        allow_smoothing_recovery=True,
    )
    target = np.eye(4)
    target[0, 3] = 0.8
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(1000.0, 1000.0, 1000.0, 10.0, 1000.0, 1000.0),
    )

    result = IKSafetyResolver(profile=profile, provider=provider).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local_provider",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.qpos is not None
    assert np.allclose(np.rad2deg(result.qpos), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert result.metadata["local_step_status"] == "active"
    assert result.metadata["local_step_first_periodic_full_solve"] is True
    assert result.metadata["local_step_full_solve_fallback_used"] is True
    assert result.metadata["local_step_full_solve_status"] == "smoothing_required"


def test_resolve_step_uses_primary_scale_without_evaluating_fallback_scales():
    provider = CompositeLikeProvider(
        providers=(FullTargetProvider(), LocalStepProvider()),
        names=("full_provider", "local_provider"),
    )
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.05, orientation_deg=10.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_scale_factors=(1.0, 0.5, 0.25),
    )
    target = np.eye(4)
    target[0, 3] = 0.8
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(1000.0, 1000.0, 1000.0, 10.0, 1000.0, 1000.0),
    )

    result = IKSafetyResolver(profile=profile, provider=provider).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local_provider",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.qpos is not None
    assert result.metadata["local_step_scale_factor"] == 1.0
    assert result.metadata["local_step_scale_strategy"]["fallback_used"] is False
    assert result.metadata["ik_debug_profile"]["candidate_eval_count"] == 1
    assert result.metadata["ik_debug_profile"]["local_candidate_eval_count"] == 1


def test_resolve_step_retries_smaller_local_step_scale_when_full_step_overshoots():
    class OvershootProvider:
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            if not options.get("local_step_mode"):
                return []
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0])),
                    source="local",
                    seed_source=seed_labels[0],
                    method="overshoot_step",
                )
            ]

        def forward_kinematics(self, qpos):
            pose = np.eye(4)
            pose[0, 3] = float(np.rad2deg(qpos[3])) * 0.10
            return pose

        def analyze_singularity(self, qpos):
            return SingularityReport(status_code=0, label="normal", metric=1.0)

    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.001, orientation_deg=1.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_scale_factors=(1.0, 0.5, 0.25),
    )
    target = np.eye(4)
    target[0, 3] = 0.05
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(1000.0, 1000.0, 1000.0, 10.0, 1000.0, 1000.0),
    )

    result = IKSafetyResolver(profile=profile, provider=OvershootProvider()).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.qpos is not None
    assert np.allclose(np.rad2deg(result.qpos), [0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    assert result.metadata["local_step_scale_factor"] == 0.5
    assert result.metadata["local_step_scale_strategy"]["fallback_used"] is True
    assert result.metadata["local_step_scale_strategy"]["fallback_reason"] == "primary_failed"
    assert result.metadata["ik_debug_profile"]["candidate_eval_count"] == 2
    assert [
        entry["scales"]
        for entry in result.metadata["local_step_scale_strategy"]["passes"]
    ] == [[1.0], [0.5]]
    assert result.metadata["local_step_progress_accept_reason"] == "within_pose_tolerance"


def test_resolve_step_allows_safe_rate_limited_full_target_transition_without_pose_progress():
    class NonmonotonicTransitionProvider:
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            if options.get("local_step_mode"):
                return []
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([0.0, 0.0, 0.0, 80.0, 0.0, 0.0])),
                    source="full",
                    seed_source=seed_labels[0],
                    method="full_target",
                )
            ]

        def forward_kinematics(self, qpos):
            q4_deg = float(np.rad2deg(np.asarray(qpos)[3]))
            pose = np.eye(4)
            if q4_deg > 70.0:
                pose[0, 3] = 1.0
            elif q4_deg > 0.0:
                pose[0, 3] = -0.1
            return pose

        def analyze_singularity(self, qpos):
            return SingularityReport(status_code=0, label="normal", metric=1.0)

    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.01, orientation_deg=1.0),
        fk_error_tolerance_m=0.01,
        fk_error_tolerance_deg=1.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_scale_factors=(1.0,),
        local_step_hold_on_failure=True,
    )
    target = np.eye(4)
    target[0, 3] = 1.0
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(1000.0, 1000.0, 1000.0, 10.0, 1000.0, 1000.0),
    )

    result = IKSafetyResolver(profile=profile, provider=NonmonotonicTransitionProvider()).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.qpos is not None
    assert np.allclose(np.rad2deg(result.qpos), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert result.candidate_source == "rate_limited_full_target"
    assert result.metadata["local_step_scale_pass"] == "transition"
    assert result.metadata["local_step_nonprogress_transition_used"] is True
    assert result.metadata["local_step_progress_accept_reason"] == "safe_rate_limited_full_target_transition"
    assert result.metadata["method"] == "rate_limited_full_target"


def test_resolve_step_dynamic_worsen_allows_large_error_orientation_recovery():
    class LargeErrorTradeoffProvider:
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            if not options.get("local_step_mode"):
                return []
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0])),
                    source="local",
                    seed_source=seed_labels[0],
                    method="orientation_tradeoff",
                )
            ]

        def forward_kinematics(self, qpos):
            q4_deg = float(np.rad2deg(qpos[3]))
            pose = np.eye(4)
            pose[0, 3] = 0.10 + 0.005 * q4_deg
            pose[:3, :3] = R.from_euler("z", q4_deg, degrees=True).as_matrix()
            return pose

        def analyze_singularity(self, qpos):
            return SingularityReport(status_code=0, label="normal", metric=1.0)

    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.01, orientation_deg=10.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_scale_factors=(1.0,),
        local_step_dynamic_worsen_enabled=True,
        local_step_dynamic_worsen_error_start_m=0.05,
        local_step_dynamic_worsen_error_start_deg=15.0,
        local_step_dynamic_worsen_position_fraction=0.10,
        local_step_dynamic_worsen_max_position_m=0.02,
        local_step_accept_orientation_progress_deg=0.25,
        local_step_max_position_worsen_m=0.003,
    )
    target = np.eye(4)
    target[:3, :3] = R.from_euler("z", 40.0, degrees=True).as_matrix()
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(1000.0, 1000.0, 1000.0, 10.0, 1000.0, 1000.0),
    )
    reference_pose = np.eye(4)
    reference_pose[0, 3] = 0.10

    result = IKSafetyResolver(profile=profile, provider=LargeErrorTradeoffProvider()).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=reference_pose,
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.qpos is not None
    assert np.allclose(np.rad2deg(result.qpos), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert result.metadata["local_step_dynamic_worsen_active"] is True
    assert result.metadata["local_step_max_position_worsen_m"] > 0.003
    assert result.metadata["local_step_progress_accept_reason"] == (
        "orientation_progress_with_bounded_position_worsening"
    )


def test_resolve_step_generates_generic_joint_limit_escape_candidate():
    class LimitEscapeProvider:
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            return []

        def forward_kinematics(self, qpos):
            q = np.asarray(qpos, dtype=float).reshape(6)
            pose = np.eye(4)
            pose[0, 3] = -float(np.rad2deg(q[0])) * 0.01
            return pose

        def analyze_singularity(self, qpos):
            return SingularityReport(status_code=0, label="normal", metric=1.0)

        def joint_position_limits(self):
            return np.deg2rad(np.array([-178.0] * 6)), np.deg2rad(np.array([178.0] * 6))

    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.01, orientation_deg=1.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_scale_factors=(1.0,),
        local_step_limit_escape_enabled=True,
        local_step_limit_escape_activation_margin_deg=2.0,
        local_step_limit_escape_max_candidates=12,
    )
    target = np.eye(4)
    target[0, 3] = 0.10
    reference_qpos = np.deg2rad(np.array([0.0, 0.0, 0.0, 177.5, 0.0, 0.0]))
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
    )

    result = IKSafetyResolver(profile=profile, provider=LimitEscapeProvider()).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=reference_qpos,
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.qpos is not None
    assert result.candidate_source == "joint_limit_escape"
    assert result.metadata["method"] == "joint_limit_escape_j4_neg"
    assert result.metadata["local_step_limit_escape"]["status"] == "active"
    assert result.metadata["local_step_limit_escape"]["active_joint_indices"] == [3]
    assert np.allclose(np.rad2deg(result.qpos), [0.0, 0.0, 0.0, 176.5, 0.0, 0.0])
    assert result.metadata["local_step_progress_accept_reason"] == "joint_limit_escape_bounded_pose_worsening"
    assert result.metadata["local_step_recovery_state"] == "joint_limit_escape"


def test_resolve_step_generates_singularity_escape_candidate_from_singular_reference():
    class SingularityEscapeProvider:
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            return []

        def forward_kinematics(self, qpos):
            return np.eye(4)

        def analyze_singularity(self, qpos):
            q = np.asarray(qpos, dtype=float).reshape(6)
            if abs(float(np.rad2deg(q[4]))) < 0.5:
                return SingularityReport(status_code=-2, label="elbow_singularity", metric=0.1)
            return SingularityReport(status_code=0, label="normal", metric=1.0)

        def joint_position_limits(self):
            return np.deg2rad(np.array([-178.0] * 6)), np.deg2rad(np.array([178.0] * 6))

    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.01, orientation_deg=1.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_scale_factors=(1.0,),
        local_step_singularity_escape_enabled=True,
        local_step_singularity_escape_max_candidates=4,
    )
    target = np.eye(4)
    target[0, 3] = 0.10
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
    )

    result = IKSafetyResolver(profile=profile, provider=SingularityEscapeProvider()).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.candidate_source == "singularity_escape"
    assert result.metadata["local_step_progress_accept_reason"] == "singularity_escape_bounded_pose_worsening"
    assert result.metadata["local_step_recovery_state"] == "singularity_escape"
    assert result.metadata["singularity"]["label"] == "normal"


def test_resolve_step_can_reject_worse_step_inside_pose_tolerance():
    class WorseWithinToleranceProvider:
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            if not options.get("local_step_mode"):
                return []
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
                    source="local",
                    seed_source=seed_labels[0],
                    method="worse_inside_tolerance",
                )
            ]

        def forward_kinematics(self, qpos):
            pose = np.eye(4)
            pose[0, 3] = float(np.rad2deg(np.asarray(qpos)[0])) * 0.001
            return pose

        def analyze_singularity(self, qpos):
            return SingularityReport(status_code=0, label="normal", metric=1.0)

    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.01, orientation_deg=1.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_accept_worse_within_tolerance=False,
        local_step_hold_on_failure=True,
    )
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
    )

    result = IKSafetyResolver(profile=profile, provider=WorseWithinToleranceProvider()).resolve_step(
        IKResolveRequest(
            target_pose=np.eye(4),
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local",),
    )

    assert not result.accepted or result.candidate_source == "local_step_hold"
    assert result.qpos is None or np.allclose(result.qpos, np.zeros(6))


def test_resolve_step_accepts_orientation_progress_with_bounded_position_worsening():
    class OrientationProgressProvider:
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            target_deg = 6.0 if options.get("local_step_mode") else 80.0
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([0.0, 0.0, 0.0, target_deg, 0.0, 0.0])),
                    source="local" if options.get("local_step_mode") else "full",
                    seed_source=seed_labels[0],
                    method="orientation_step" if options.get("local_step_mode") else "full_target",
                )
            ]

        def forward_kinematics(self, qpos):
            q4_deg = float(np.rad2deg(qpos[3]))
            pose = np.eye(4)
            pose[0, 3] = -0.001 * q4_deg
            pose[:3, :3] = R.from_euler("z", q4_deg, degrees=True).as_matrix()
            return pose

        def analyze_singularity(self, qpos):
            return SingularityReport(status_code=0, label="normal", metric=1.0)

    provider = OrientationProgressProvider()
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.0001, orientation_deg=1.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_trigger_ratio=4.0,
        local_step_accept_orientation_progress_deg=0.25,
        local_step_max_position_worsen_m=0.003,
    )
    target = np.eye(4)
    target[:3, :3] = R.from_euler("z", 10.0, degrees=True).as_matrix()
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(1000.0, 1000.0, 1000.0, 10.0, 1000.0, 1000.0),
    )

    result = IKSafetyResolver(profile=profile, provider=provider).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.qpos is not None
    assert np.allclose(np.rad2deg(result.qpos), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert result.metadata["local_step_progress_accept_reason"] == (
        "orientation_progress_with_bounded_position_worsening"
    )
    assert result.metadata["local_step_orientation_progress_deg"] > 0.25
    assert result.metadata["local_step_position_progress_m"] < 0.0


def test_resolve_step_uses_rate_limited_full_target_when_local_step_has_no_progress():
    class NoProgressProvider(LocalStepProvider):
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            if not options.get("local_step_mode"):
                return []
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([0.0, 0.0, 0.0, -6.0, 0.0, 0.0])),
                    source="local",
                    seed_source=seed_labels[0],
                    method="local_step_wrong_way",
                )
            ]

    provider = CompositeLikeProvider(
        providers=(FullTargetProvider(), NoProgressProvider()),
        names=("full_provider", "local_provider"),
    )
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.05, orientation_deg=10.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_trigger_ratio=4.0,
        local_step_hold_on_failure=True,
    )
    target = np.eye(4)
    target[0, 3] = 0.8
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(1000.0, 1000.0, 1000.0, 10.0, 1000.0, 1000.0),
    )

    result = IKSafetyResolver(profile=profile, provider=provider).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local_provider",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.qpos is not None
    assert np.allclose(np.rad2deg(result.qpos), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert result.candidate_source == "rate_limited_full_target"
    assert result.metadata["local_step_status"] == "active"
    assert result.metadata["local_step_progress_accept_reason"] == "combined_pose_score_progress"


def test_resolve_step_holds_reference_when_no_safe_local_step_exists():
    class NoProgressProvider(LocalStepProvider):
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            if not options.get("local_step_mode"):
                return []
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([0.0, 0.0, 0.0, -6.0, 0.0, 0.0])),
                    source="local",
                    seed_source=seed_labels[0],
                    method="local_step_wrong_way",
                )
            ]

    provider = CompositeLikeProvider(
        providers=(FullTargetProvider(), NoProgressProvider()),
        names=("full_provider", "local_provider"),
    )
    profile = IKSafetyProfile(
        name="test",
        target_tolerance=PoseTolerance(position_m=0.05, orientation_deg=10.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_trigger_ratio=4.0,
        local_step_full_target_step_enabled=False,
        local_step_hold_on_failure=True,
    )
    target = np.eye(4)
    target[0, 3] = 0.8
    velocity = JointVelocityLimitProfile(
        name="test",
        max_velocity_deg_s=(1000.0, 1000.0, 1000.0, 10.0, 1000.0, 1000.0),
    )

    result = IKSafetyResolver(profile=profile, provider=provider).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        ),
        joint_velocity_profile=velocity,
        dt_sec=0.1,
        local_provider_names=("local_provider",),
    )

    assert result.status is IKResolveStatus.RELAXED
    assert result.qpos is not None
    assert np.allclose(result.qpos, np.zeros(6))
    assert result.candidate_source == "local_step_hold"
    assert result.metadata["candidate_source"] == "local_step_hold"
    assert result.metadata["method"] == "hold_reference"
    assert result.metadata["local_step_status"] == "hold_reference"
    assert result.metadata["local_step_hold_reason"] == "no_safe_local_step"


def test_smoothing_required_when_all_candidates_are_unsafe():
    provider = FakeProvider()
    profile = IKSafetyProfile(
        name="test",
        hard_branch_joint_indices=(0, 1, 2),
        hard_branch_max_delta_deg=(90.0, 90.0, 90.0),
        allow_smoothing_recovery=True,
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
    )
    result = IKSafetyResolver(profile=profile, provider=provider).resolve(_request("none"))

    assert result.status is IKResolveStatus.SMOOTHING_REQUIRED
    assert not result.accepted
    assert result.metadata["best_rejected"]["reject_reason"] == "hard_branch_jump"


def test_hard_deadline_at_resolve_step_entry_holds_previous_safe_without_provider_work():
    class CountingProvider:
        def __init__(self):
            self.calls = 0

        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            del target_pose, seeds, seed_labels, options
            self.calls += 1
            return []

        def forward_kinematics(self, qpos):
            del qpos
            return np.eye(4)

        def analyze_singularity(self, qpos):
            del qpos
            return SingularityReport(status_code=0, label="normal", metric=1.0)

    provider = CountingProvider()
    previous_safe = np.deg2rad(np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0]))
    result = IKSafetyResolver(
        profile=IKSafetyProfile(name="hard_deadline"),
        provider=provider,
    ).resolve_step(
        IKResolveRequest(
            target_pose=np.eye(4),
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
            previous_safe_qpos=previous_safe,
            metadata={
                "hard_deadline_enabled": True,
                "hard_deadline_budget_ms": 25.0,
                "hard_deadline_monotonic": time.perf_counter() - 1.0,
            },
        ),
        joint_velocity_profile=JointVelocityLimitProfile(
            name="test",
            max_velocity_deg_s=(100.0,) * 6,
        ),
        dt_sec=0.025,
        local_provider_names=("local",),
    )

    assert provider.calls == 0
    assert result.candidate_source == "hard_deadline_hold_reference"
    assert result.qpos is not None
    assert np.allclose(result.qpos, previous_safe)
    assert result.metadata["hard_deadline_hold_qpos_source"] == "previous_safe"
    assert result.metadata["hard_deadline_expired"] is True


def test_resolve_propagates_hard_deadline_to_provider_options():
    class DeadlineAwareProvider:
        def __init__(self):
            self.options: dict[str, object] | None = None

        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            del target_pose
            self.options = dict(options)
            return [
                IKCandidate(
                    qpos=np.asarray(seeds[0], dtype=float),
                    source="deadline_provider",
                    seed_source=seed_labels[0],
                    method="identity",
                )
            ]

        def forward_kinematics(self, qpos):
            del qpos
            return np.eye(4)

        def analyze_singularity(self, qpos):
            del qpos
            return SingularityReport(status_code=0, label="normal", metric=1.0)

    provider = DeadlineAwareProvider()
    deadline = time.perf_counter() + 1.0
    result = IKSafetyResolver(
        profile=IKSafetyProfile(
            name="hard_deadline",
            fk_error_tolerance_m=1.0,
            fk_error_tolerance_deg=180.0,
        ),
        provider=provider,
    ).resolve(
        IKResolveRequest(
            target_pose=np.eye(4),
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
            metadata={
                "hard_deadline_enabled": True,
                "hard_deadline_budget_ms": 25.0,
                "hard_deadline_monotonic": deadline,
            },
        )
    )

    assert result.accepted
    assert provider.options is not None
    assert provider.options["provider_deadline_monotonic"] == deadline
    assert float(provider.options["provider_remaining_time_budget_ms"]) > 0.0


def test_hard_deadline_local_step_evaluates_safe_base_before_escape_expansion():
    class BaseCandidateProvider:
        def __init__(self):
            self.fk_calls = 0

        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            del target_pose
            if not options.get("local_step_mode"):
                return []
            return [
                IKCandidate(
                    qpos=np.deg2rad(np.array([0.0, 0.0, 133.9, 1.0, 0.0, 0.0])),
                    source="base_local",
                    seed_source=seed_labels[0],
                    method="base_local",
                )
            ]

        def forward_kinematics(self, qpos):
            self.fk_calls += 1
            pose = np.eye(4)
            pose[0, 3] = float(np.rad2deg(np.asarray(qpos)[3])) * 0.01
            return pose

        def analyze_singularity(self, qpos):
            del qpos
            return SingularityReport(status_code=0, label="normal", metric=1.0)

        def joint_position_limits(self):
            return (
                np.deg2rad(np.array([-180.0, -180.0, -135.0, -180.0, -180.0, -180.0])),
                np.deg2rad(np.array([180.0, 180.0, 135.0, 180.0, 180.0, 180.0])),
            )

    provider = BaseCandidateProvider()
    target = np.eye(4)
    target[0, 3] = 0.10
    profile = IKSafetyProfile(
        name="hard_deadline_local",
        target_tolerance=PoseTolerance(position_m=0.05, orientation_deg=10.0),
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
        local_step_enabled=True,
        local_step_first_enabled=True,
        local_step_limit_escape_enabled=True,
        local_step_limit_escape_activation_margin_deg=2.0,
    )
    result = IKSafetyResolver(profile=profile, provider=provider).resolve_step(
        IKResolveRequest(
            target_pose=target,
            reference_qpos=np.deg2rad(np.array([0.0, 0.0, 133.9, 0.0, 0.0, 0.0])),
            reference_pose=np.eye(4),
            metadata={
                "hard_deadline_enabled": True,
                "hard_deadline_budget_ms": 1000.0,
                "hard_deadline_monotonic": time.perf_counter() + 1.0,
            },
        ),
        joint_velocity_profile=JointVelocityLimitProfile(
            name="test",
            max_velocity_deg_s=(100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
        ),
        dt_sec=0.025,
        local_provider_names=("local",),
    )

    assert result.accepted
    assert result.candidate_source == "base_local"
    assert result.metadata["local_step_scale_strategy"]["hard_deadline_staged"] is True
    # Only the accepted base candidate needs FK. A limit-escape expansion
    # would add up to twelve more FK calls before evaluating this candidate.
    assert provider.fk_calls == 1


def test_limit_escape_stops_after_the_inflight_fk_crosses_hard_deadline(monkeypatch):
    class Clock:
        def __init__(self):
            self.value = 0.0

        def __call__(self):
            return self.value

    class SlowFkProvider:
        def __init__(self, clock):
            self.clock = clock
            self.fk_calls = 0

        def forward_kinematics(self, qpos):
            del qpos
            self.fk_calls += 1
            # An SDK/FK call cannot be preempted, but once it returns the
            # resolver must not launch the next auxiliary candidate.
            self.clock.value = 0.010
            return np.eye(4)

        def joint_position_limits(self):
            return (
                np.deg2rad(np.array([-180.0, -180.0, -135.0, -180.0, -180.0, -180.0])),
                np.deg2rad(np.array([180.0, 180.0, 135.0, 180.0, 180.0, 180.0])),
            )

    clock = Clock()
    monkeypatch.setattr(ik_resolver_module.time, "perf_counter", clock)
    provider = SlowFkProvider(clock)
    resolver = IKSafetyResolver(
        profile=IKSafetyProfile(
            name="deadline_escape",
            local_step_limit_escape_enabled=True,
            local_step_limit_escape_activation_margin_deg=2.0,
        ),
        provider=provider,
    )

    candidates, metadata = resolver._local_step_limit_escape_candidates(
        reference_qpos=np.deg2rad(np.array([0.0, 0.0, 133.9, 0.0, 0.0, 0.0])),
        target_pose=np.eye(4),
        reference_pose=np.eye(4),
        joint_velocity_profile=JointVelocityLimitProfile(
            name="test",
            max_velocity_deg_s=(100.0,) * 6,
        ),
        dt_sec=0.025,
        deadline_monotonic=0.005,
    )

    assert candidates == []
    assert metadata["status"] == "interrupted_hard_deadline"
    assert metadata["hard_deadline_expired"] is True
    assert provider.fk_calls == 1
