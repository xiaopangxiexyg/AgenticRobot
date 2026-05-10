from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeploymentProfile:
    name: str
    data_config_name: str
    default_port: int
    default_dataset_path: str
    default_instruction: str
    required_arms: tuple[str, ...]
    required_hands: tuple[str, ...]
    arm_backend: str = "rm65"
    supports_live_rm65: bool = True


FRANKA_PROFILE = DeploymentProfile(
    name="chips",
    data_config_name="franka",
    default_port=5555,
    default_dataset_path="/home/user/zc/dex_inspire_teleop_new/real_dataset/2026-05-07_peeling_white_carrot_converted",
    default_instruction="Pick and place the potato chips.",
    required_arms=("right",),
    required_hands=("right",),
    arm_backend="fr3",
    supports_live_rm65=False,
)

REALMAN20_PROFILE = DeploymentProfile(
    name="heavy_light",
    data_config_name="realman20",
    default_port=5557,
    default_dataset_path="/home/ai/Desktop/Being-H-series/2026-04-30_heavy_light_converted",
    default_instruction="Sort the heavy and light objects.",
    required_arms=("right",),
    required_hands=("right",),
    arm_backend="rm65",
    supports_live_rm65=True,
)

REALMAN21_PROFILE = DeploymentProfile(
    name="realman21",
    data_config_name="realman21",
    default_port=5558,
    default_dataset_path="",
    default_instruction="Run the Realman right-arm dual-hand task.",
    required_arms=("right",),
    required_hands=("left", "right"),
    arm_backend="rm65",
    supports_live_rm65=True,
)

REALMAN22_PROFILE = DeploymentProfile(
    name="paper_folding",
    data_config_name="realman22",
    default_port=5556,
    default_dataset_path="/home/ai/Desktop/Being-H-series/2026-04-28_large_brown_paper_fold_conerted",
    default_instruction="Fold the large brown paper.",
    required_arms=("left", "right"),
    required_hands=("left", "right"),
    arm_backend="rm65",
    supports_live_rm65=True,
)


PROFILES = {
    "franka": FRANKA_PROFILE,
    "realman20": REALMAN20_PROFILE,
    "realman21": REALMAN21_PROFILE,
    "realman22": REALMAN22_PROFILE,
}
