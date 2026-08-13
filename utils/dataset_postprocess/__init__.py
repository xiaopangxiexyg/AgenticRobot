from .causal_align import CausalAlignPlan, causal_align_dataset, plan_causal_align
from .camera_semantic_materialize import (
    CameraSemanticMaterializationError,
    CameraSemanticMaterializationPlan,
    materialize_camera_semantics,
    plan_camera_semantic_materialization,
)
from .delete_episodes import EpisodeDeletionPlan, delete_episodes, plan_episode_deletion
from .fault_nan_tail import FaultNanTail, detect_fault_nan_tail
from .force_collision_labels import (
    ForceCollisionEpisodeLabel,
    ForceCollisionLabelPlan,
    label_force_collisions,
    plan_force_collision_labels,
)
from .selection import parse_episode_selection
from .split_episodes import EpisodeSplitPlan, plan_episode_split, split_episodes
from .stutter_segments import (
    StutterCleanPlan,
    StutterCleanSegment,
    StutterSegment,
    clean_stutters,
    detect_stutter_segments,
    plan_stutter_clean,
)
from .tactile_compact import (
    TactileCompactionPlan,
    TactileCompactionResult,
    compact_tactile_dataset_in_place,
    plan_tactile_compaction,
)
from .trim_nan_tails import TailTrim, TailTrimPlan, plan_nan_tail_trim, trim_nan_tails
__all__ = [
    "EpisodeDeletionPlan",
    "EpisodeSplitPlan",
    "CausalAlignPlan",
    "CameraSemanticMaterializationError",
    "CameraSemanticMaterializationPlan",
    "ForceCollisionEpisodeLabel",
    "ForceCollisionLabelPlan",
    "FaultNanTail",
    "StutterSegment",
    "StutterCleanPlan",
    "StutterCleanSegment",
    "TailTrim",
    "TailTrimPlan",
    "TactileCompactionPlan",
    "TactileCompactionResult",
    "UmiArmAlignmentPlan",
    "V22ToV23DeskPlan",
    "causal_align_dataset",
    "clean_stutters",
    "compact_tactile_dataset_in_place",
    "convert_v22_dataset_to_v23_desk",
    "delete_episodes",
    "detect_fault_nan_tail",
    "detect_stutter_segments",
    "label_force_collisions",
    "materialize_camera_semantics",
    "migrate_umi_workspace_near_boundary",
    "migrate_umi_arm_action_to_next_observation",
    "parse_episode_selection",
    "plan_force_collision_labels",
    "plan_causal_align",
    "plan_camera_semantic_materialization",
    "plan_episode_split",
    "plan_nan_tail_trim",
    "plan_episode_deletion",
    "plan_stutter_clean",
    "plan_tactile_compaction",
    "plan_v22_to_v23_desk_conversion",
    "split_episodes",
    "trim_nan_tails",
]


_LAZY_EXPORTS = {
    "UmiArmAlignmentPlan": (
        ".umi_next_state_arm_action",
        "UmiArmAlignmentPlan",
    ),
    "V22ToV23DeskPlan": (".v22_to_v23_desk", "V22ToV23DeskPlan"),
    "convert_v22_dataset_to_v23_desk": (
        ".v22_to_v23_desk",
        "convert_v22_dataset_to_v23_desk",
    ),
    "migrate_umi_arm_action_to_next_observation": (
        ".umi_next_state_arm_action",
        "migrate_umi_arm_action_to_next_observation",
    ),
    "migrate_umi_workspace_near_boundary": (
        ".workspace_near_boundary",
        "migrate_umi_workspace_near_boundary",
    ),
    "plan_v22_to_v23_desk_conversion": (
        ".v22_to_v23_desk",
        "plan_v22_to_v23_desk_conversion",
    ),
}


def __getattr__(name: str):
    """Preserve public imports without loading compatibility tools eagerly."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
