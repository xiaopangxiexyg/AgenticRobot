from __future__ import annotations

import numpy as np

from robot_motion.seed_expansion import (
    SeedExpansionProfile,
    SeedPerturbation,
    expand_seed_qpos,
    load_seed_expansion_profiles,
)


def test_default_rm65_seed_expansion_profile_loads_and_labels_key_perturbations():
    profile = load_seed_expansion_profiles()["rm65"]

    result = expand_seed_qpos(
        seeds=(np.deg2rad(np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])),),
        labels=("reference",),
        profile=profile,
    )

    assert result.metadata["profile"] == "rm65"
    assert result.metadata["generated_count"] == profile.max_extra_seeds
    assert result.metadata["generated_count"] == 8
    labels = set(result.labels)
    assert "reference:j2_j3_coord_pos_5deg" in labels
    assert "reference:wrist_flip_j4p_j6p" in labels
    assert "reference:small_j4_neg_2deg" in labels
    assert "reference:small_j6_neg_2deg" not in labels


def test_seed_expansion_applies_wrist_flip_scale_and_offset():
    profile = SeedExpansionProfile(
        name="test",
        dof=6,
        max_extra_seeds=1,
        source_label_allowlist=("reference",),
        perturbations=(
            SeedPerturbation(
                label="wrist_flip",
                scale=(1.0, 1.0, 1.0, 1.0, -1.0, 1.0),
                offset_deg=(0.0, 0.0, 0.0, 180.0, 0.0, -180.0),
            ),
        ),
    )

    result = expand_seed_qpos(
        seeds=(np.deg2rad(np.array([1.0, 2.0, 3.0, 30.0, 40.0, 50.0])),),
        labels=("reference",),
        profile=profile,
    )

    assert result.labels == ("reference", "reference:wrist_flip")
    assert np.allclose(np.rad2deg(result.seeds[1]), [1.0, 2.0, 3.0, 210.0, -40.0, -130.0])


def test_seed_expansion_inserts_expanded_seeds_after_allowed_base_seed():
    profile = SeedExpansionProfile(
        name="test",
        dof=2,
        max_extra_seeds=2,
        source_label_allowlist=("reference",),
        perturbations=(
            SeedPerturbation(label="pos", offset_deg=(1.0, 0.0)),
            SeedPerturbation(label="neg", offset_deg=(-1.0, 0.0)),
        ),
    )

    result = expand_seed_qpos(
        seeds=(np.zeros(2), np.deg2rad(np.array([10.0, 0.0]))),
        labels=("reference", "measured_current"),
        profile=profile,
    )

    assert result.labels == (
        "reference",
        "reference:pos",
        "reference:neg",
        "measured_current",
    )
