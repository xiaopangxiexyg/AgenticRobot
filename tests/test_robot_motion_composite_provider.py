from __future__ import annotations

import time

import numpy as np

from robot_motion.composite_provider import CompositeKinematicsProvider
from robot_motion.ik_resolver import IKSafetyResolver
from robot_motion.profiles import IKSafetyProfile
from robot_motion.types import IKCandidate, IKResolveRequest, IKResolveStatus, SingularityReport


class ConstantProvider:
    def __init__(self, *, source: str, qpos_deg: list[list[float]]):
        self.source = source
        self.qpos_deg = qpos_deg

    def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
        return [
            IKCandidate(
                qpos=np.deg2rad(np.asarray(qpos_deg, dtype=float)),
                source=self.source,
                seed_source=seed_labels[0],
                method="constant",
            )
            for qpos_deg in self.qpos_deg
        ]

    def forward_kinematics(self, qpos):
        return np.eye(4)

    def analyze_singularity(self, qpos):
        return SingularityReport(status_code=0, label="normal", metric=1.0)


def test_composite_provider_merges_dedupes_and_marks_time_budget():
    provider = CompositeKinematicsProvider(
        (
            ConstantProvider(source="a", qpos_deg=[[1, 2, 3, 4, 5, 6]]),
            ConstantProvider(source="b", qpos_deg=[[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]),
        ),
        names=("provider_a", "provider_b"),
    )

    candidates = provider.iter_ik_candidates(
        target_pose=np.eye(4),
        seeds=(np.zeros(6),),
        seed_labels=("reference",),
        options={"provider_time_budget_ms": 1000.0, "max_candidate_count": 8},
    )

    assert len(candidates) == 2
    assert candidates[0].metadata["composite_provider_name"] == "provider_a"
    assert candidates[1].metadata["composite_provider_name"] == "provider_b"
    assert candidates[0].metadata["provider_time_budget_exceeded"] is False
    assert provider.last_metadata["provider_count"] == 2


def test_composite_provider_stops_serial_collection_after_budget_is_spent():
    class SlowProvider(ConstantProvider):
        def iter_ik_candidates(self, *, target_pose, seeds, seed_labels, options):
            time.sleep(0.002)
            return super().iter_ik_candidates(
                target_pose=target_pose,
                seeds=seeds,
                seed_labels=seed_labels,
                options=options,
            )

    provider = CompositeKinematicsProvider(
        (
            SlowProvider(source="a", qpos_deg=[[1, 0, 0, 0, 0, 0]]),
            SlowProvider(source="b", qpos_deg=[[2, 0, 0, 0, 0, 0]]),
        ),
        names=("provider_a", "provider_b"),
    )

    candidates = provider.iter_ik_candidates(
        target_pose=np.eye(4),
        seeds=(np.zeros(6),),
        seed_labels=("reference",),
        options={"provider_time_budget_ms": 0.5, "max_candidate_count": 8},
    )

    assert len(candidates) == 1
    assert candidates[0].source == "a"
    assert provider.last_metadata["provider_time_budget_exceeded"] is True
    assert provider.last_metadata["provider_budget_skipped_names"] == ["provider_b"]

    candidates = provider.iter_ik_candidates(
        target_pose=np.eye(4),
        seeds=(np.zeros(6),),
        seed_labels=("reference",),
        options={"provider_time_budget_ms": 0.5, "max_candidate_count": 8},
    )

    assert len(candidates) == 1
    assert candidates[0].source == "b"
    assert provider.last_metadata["provider_budget_skipped_names"] == ["provider_a"]


def test_composite_provider_respects_external_deadline_before_provider_call():
    provider = CompositeKinematicsProvider(
        (
            ConstantProvider(source="a", qpos_deg=[[1, 0, 0, 0, 0, 0]]),
            ConstantProvider(source="b", qpos_deg=[[2, 0, 0, 0, 0, 0]]),
        ),
        names=("provider_a", "provider_b"),
    )

    candidates = provider.iter_ik_candidates(
        target_pose=np.eye(4),
        seeds=(np.zeros(6),),
        seed_labels=("reference",),
        options={
            "provider_time_budget_ms": 1000.0,
            "provider_deadline_monotonic": time.perf_counter() - 0.001,
            "max_candidate_count": 8,
        },
    )

    assert candidates == []
    assert provider.last_metadata["provider_external_deadline_active"] is True
    assert provider.last_metadata["provider_budget_skipped_names"] == [
        "provider_a",
        "provider_b",
    ]


def test_composite_provider_can_allowlist_provider_names():
    provider = CompositeKinematicsProvider(
        (
            ConstantProvider(source="a", qpos_deg=[[1, 0, 0, 0, 0, 0]]),
            ConstantProvider(source="b", qpos_deg=[[2, 0, 0, 0, 0, 0]]),
        ),
        names=("provider_a", "provider_b"),
    )

    candidates = provider.iter_ik_candidates(
        target_pose=np.eye(4),
        seeds=(np.zeros(6),),
        seed_labels=("reference",),
        options={"provider_name_allowlist": ["provider_b"], "max_candidate_count": 8},
    )

    assert len(candidates) == 1
    assert candidates[0].source == "b"
    assert candidates[0].metadata["composite_provider_name"] == "provider_b"
    assert provider.last_metadata["requested_provider_names"] == ["provider_b"]


def test_composite_provider_round_robins_candidates_to_avoid_provider_starvation():
    provider = CompositeKinematicsProvider(
        (
            ConstantProvider(
                source="sdk_like",
                qpos_deg=[
                    [10, 0, 0, 0, 0, 0],
                    [11, 0, 0, 0, 0, 0],
                    [12, 0, 0, 0, 0, 0],
                    [13, 0, 0, 0, 0, 0],
                ],
            ),
            ConstantProvider(source="pinocchio_like", qpos_deg=[[1, 0, 0, 0, 0, 0]]),
        ),
        names=("sdk_like", "pinocchio_like"),
    )

    candidates = provider.iter_ik_candidates(
        target_pose=np.eye(4),
        seeds=(np.zeros(6),),
        seed_labels=("reference",),
        options={"max_candidate_count": 3},
    )

    assert [candidate.source for candidate in candidates] == [
        "sdk_like",
        "pinocchio_like",
        "sdk_like",
    ]


def test_resolver_prefers_smallest_max_joint_delta_across_providers():
    provider = CompositeKinematicsProvider(
        (
            ConstantProvider(source="single_big_joint", qpos_deg=[[6, 0, 0, 0, 0, 0]]),
            ConstantProvider(source="spread_delta", qpos_deg=[[4, 4, 4, 4, 4, 4]]),
        ),
        names=("single_big_joint", "spread_delta"),
    )
    profile = IKSafetyProfile(
        name="test",
        reject_singular=True,
        fk_error_tolerance_m=1.0,
        fk_error_tolerance_deg=180.0,
    )

    result = IKSafetyResolver(profile=profile, provider=provider).resolve(
        IKResolveRequest(
            target_pose=np.eye(4),
            reference_qpos=np.zeros(6),
            reference_pose=np.eye(4),
        )
    )

    assert result.status is IKResolveStatus.PASS
    assert result.candidate_source == "spread_delta"
    assert result.metadata["joint_delta_max_deg"] == 4.0
    assert result.metadata["score_policy"].startswith("minimize_penalized_max_joint_delta_then_norm")
