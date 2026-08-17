"""Resolve strict v2.2 policy-I/O layouts from live collection components.

The collection CLI must never guess a model contract from a legacy vector
width.  This module derives the end-effector portion from the instantiated
backend/retargeter pair, then optionally selects a semantically identical
checked-in preset.  Unknown pairs fail closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_contracts.v22 import EndEffectorSpec, PolicyIOLayout
from dataset_interface.policy_io_layout import load_policy_io_layout

from ..backends.grippers import DMTacGripperBackend, GRIPPER_ACTION_MAX
from ..backends.hands import InspireTacHandBackend
from ..retargeters import JointsToDexHandRetargeter, ScalarToGripperRetargeter


AUTO_POLICY_IO_LAYOUT = "auto"
DEFAULT_POLICY_IO_LAYOUT_PRESET_DIR = (
    Path(__file__).resolve().parent.parent / "configs" / "policy_io_layouts"
)
_SIDES = ("left", "right")


@dataclass(frozen=True)
class ResolvedCollectionPolicyIOLayout:
    """A layout plus non-semantic provenance for collection metadata."""

    layout: PolicyIOLayout | None
    provenance: Mapping[str, Any]


def _normalized_sides(active_sides: Sequence[str]) -> tuple[str, ...]:
    sides = tuple(str(side).strip().lower() for side in active_sides)
    if set(sides) != set(_SIDES) or len(sides) != len(_SIDES):
        raise ValueError(
            "DexTeleop v2.2 collection requires active_sides=('left', 'right') "
            "because its arm pose core is fixed at left_pose7 + right_pose7."
        )
    return _SIDES


def _side_component(
    values: Mapping[str, Any] | None,
    side: str,
) -> Any | None:
    return None if values is None else values.get(side)


def _dmtac_spec(
    *,
    side: str,
    backend: DMTacGripperBackend,
    retargeter: ScalarToGripperRetargeter,
) -> EndEffectorSpec:
    backend_config = backend.config
    retargeter_config = retargeter.config
    if str(retargeter_config.side) != side:
        raise ValueError(
            f"{side} ScalarToGripperRetargeter declares side={retargeter_config.side!r}."
        )
    if str(retargeter_config.effector_name) != str(backend.name):
        raise ValueError(
            f"{side} gripper backend/retargeter effector mismatch: "
            f"{backend.name!r} != {retargeter_config.effector_name!r}."
        )
    if int(backend_config.command_dim) != 0 or int(retargeter_config.output_dim) != 0:
        raise ValueError("DM-Tac v2.2 collection requires its only actuator in slot 0.")
    if abs(float(backend_config.max_action) - float(GRIPPER_ACTION_MAX)) > 1e-6:
        raise ValueError(
            "DM-Tac v2.2 collection requires max_action=1.0; "
            f"got {backend_config.max_action!r}."
        )
    if abs(float(retargeter_config.output_max) - float(GRIPPER_ACTION_MAX)) > 1e-6:
        raise ValueError(
            "DM-Tac retargeter output_max must be 1.0 for v2.2 collection; "
            f"got {retargeter_config.output_max!r}."
        )
    return EndEffectorSpec.continuous(
        side,
        adapter="dmtac_gripper_v1",
        target_dim=1,
        target_names=("gripper_close",),
        state_names=("gripper_close",),
        target_source_indices=(0,),
        state_source_indices=(0,),
        target_legacy_q6_slot_indices=(0,),
        state_source_field="gripper_measured_scalar",
        units="normalized_0_1",
        normalization={"minimum": 0.0, "maximum": float(GRIPPER_ACTION_MAX)},
        metadata={
            "collection_action_source": "final_hand_command.slot_0",
            "collection_state_source": "gripper_measured_scalar",
            "collection_state_timestamp_field": "joint_timestamp",
            "collection_state_requires_measured_feedback": True,
            "collection_state_max_age_sec": 0.5,
        },
    )


def _inspire_spec(
    *,
    side: str,
    backend: InspireTacHandBackend,
    retargeter: JointsToDexHandRetargeter,
) -> EndEffectorSpec:
    retargeter_config = retargeter.config
    if str(retargeter_config.side) != side:
        raise ValueError(
            f"{side} JointsToDexHandRetargeter declares side={retargeter_config.side!r}."
        )
    if str(retargeter_config.effector_name) != str(backend.name):
        raise ValueError(
            f"{side} Inspire backend/retargeter effector mismatch: "
            f"{backend.name!r} != {retargeter_config.effector_name!r}."
        )
    names = ("pinky", "ring", "middle", "index", "thumb_yaw", "thumb_flex")
    return EndEffectorSpec.continuous(
        side,
        adapter="inspire_tac_v1",
        target_dim=6,
        target_names=names,
        state_names=names,
        target_source_indices=(0, 1, 2, 3, 4, 5),
        state_source_indices=(0, 1, 2, 3, 4, 5),
        target_legacy_q6_slot_indices=(0, 1, 2, 3, 4, 5),
        state_source_field="measured_joint_pos",
        units="normalized_0_1",
        normalization={"minimum": 0.0, "maximum": 1.0},
        metadata={
            "collection_action_source": "final_hand_command.q6",
            "collection_state_source": "measured_joint_pos",
            "collection_state_timestamp_field": "joint_timestamp",
            "collection_state_requires_measured_feedback": True,
            "collection_state_max_age_sec": 0.5,
        },
    )


def derive_collection_policy_io_layout(
    *,
    active_sides: Sequence[str],
    hand_backends_by_side: Mapping[str, Any] | None,
    hand_retargeters_by_side: Mapping[str, Any] | None,
    arm_controlled_frame_by_side: Mapping[str, str] | None = None,
    arm_reference_frame_by_side: Mapping[str, str] | None = None,
) -> PolicyIOLayout:
    """Create a v2.2 layout from actual runtime component instances.

    A configured end effector needs both a target-producing retargeter and a
    feedback-producing backend.  This intentionally does not infer a discrete
    contract: a future discrete pair must register its vocabulary explicitly.
    """

    _normalized_sides(active_sides)
    effectors: dict[str, EndEffectorSpec] = {}
    for side in _SIDES:
        backend = _side_component(hand_backends_by_side, side)
        retargeter = _side_component(hand_retargeters_by_side, side)
        if backend is None and retargeter is None:
            effectors[side] = EndEffectorSpec.none(side)
        elif isinstance(backend, DMTacGripperBackend) and isinstance(
            retargeter, ScalarToGripperRetargeter
        ):
            effectors[side] = _dmtac_spec(
                side=side,
                backend=backend,
                retargeter=retargeter,
            )
        elif isinstance(backend, InspireTacHandBackend) and isinstance(
            retargeter, JointsToDexHandRetargeter
        ):
            effectors[side] = _inspire_spec(
                side=side,
                backend=backend,
                retargeter=retargeter,
            )
        else:
            raise ValueError(
                f"Unsupported or incomplete v2.2 end-effector pair for {side}: "
                f"backend={type(backend).__name__ if backend is not None else None}, "
                f"retargeter={type(retargeter).__name__ if retargeter is not None else None}. "
                "A v2.2 collector requires a registered command and measured-state pair."
            )
    return PolicyIOLayout(
        name="runtime_auto_" + "_".join(
            f"{side}-{effectors[side].adapter.removesuffix('_v1')}" for side in _SIDES
        ),
        arm_controlled_frame_by_side=(
            {side: str(arm_controlled_frame_by_side.get(side, "tool_tcp")) for side in _SIDES}
            if arm_controlled_frame_by_side is not None
            else {side: "tool_tcp" for side in _SIDES}
        ),
        arm_reference_frame_by_side=(
            {side: str(arm_reference_frame_by_side.get(side, f"{side}_arm_base")) for side in _SIDES}
            if arm_reference_frame_by_side is not None
            else {side: f"{side}_arm_base" for side in _SIDES}
        ),
        end_effectors=effectors,
        metadata={"collection_layout_origin": "runtime_auto_v1"},
    )


def _semantic_layout_signature(layout: PolicyIOLayout) -> str:
    payload = layout.to_dict()
    payload.pop("name", None)
    payload.pop("metadata", None)
    # ``legacy_q6_slot_indices`` only projects a policy *action* into old
    # dispatch slots.  It has no state-source meaning, so older presets that
    # inherit it into ``state`` remain semantically equivalent to a runtime
    # layout that declares measured state explicitly.
    for spec in payload.get("end_effectors", {}).values():
        if isinstance(spec, dict) and isinstance(spec.get("state"), dict):
            spec["state"].pop("legacy_q6_slot_indices", None)
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _matching_preset(
    layout: PolicyIOLayout,
    *,
    preset_dir: Path,
) -> tuple[PolicyIOLayout, Path] | None:
    if not preset_dir.is_dir():
        return None
    signature = _semantic_layout_signature(layout)
    matches: list[tuple[PolicyIOLayout, Path]] = []
    for path in sorted(preset_dir.glob("*.json")):
        candidate = load_policy_io_layout(path)
        if candidate is not None and _semantic_layout_signature(candidate) == signature:
            matches.append((candidate, path))
    if len(matches) > 1:
        raise ValueError(
            "Multiple policy-I/O presets have identical runtime semantics: "
            + ", ".join(str(path) for _layout, path in matches)
        )
    return matches[0] if matches else None


def resolve_collection_policy_io_layout(
    requested: str | Path | None,
    *,
    active_sides: Sequence[str],
    hand_backends_by_side: Mapping[str, Any] | None,
    hand_retargeters_by_side: Mapping[str, Any] | None,
    arm_controlled_frame_by_side: Mapping[str, str] | None = None,
    arm_reference_frame_by_side: Mapping[str, str] | None = None,
    preset_dir: Path = DEFAULT_POLICY_IO_LAYOUT_PRESET_DIR,
) -> ResolvedCollectionPolicyIOLayout:
    """Resolve legacy/off, automatic, or explicit collection layout selection."""

    request = "" if requested is None else str(requested).strip()
    if not request:
        return ResolvedCollectionPolicyIOLayout(
            layout=None,
            provenance={"mode": "legacy_v21", "strict": False},
        )
    derived = derive_collection_policy_io_layout(
        active_sides=active_sides,
        hand_backends_by_side=hand_backends_by_side,
        hand_retargeters_by_side=hand_retargeters_by_side,
        arm_controlled_frame_by_side=arm_controlled_frame_by_side,
        arm_reference_frame_by_side=arm_reference_frame_by_side,
    )
    if request.lower() == AUTO_POLICY_IO_LAYOUT:
        preset = _matching_preset(derived, preset_dir=Path(preset_dir))
        if preset is None:
            return ResolvedCollectionPolicyIOLayout(
                layout=derived,
                provenance={
                    "mode": "auto_generated",
                    "strict": True,
                    "preset": None,
                    "derived_layout_hash": derived.layout_hash,
                },
            )
        layout, path = preset
        return ResolvedCollectionPolicyIOLayout(
            layout=layout,
            provenance={
                "mode": "auto_preset",
                "strict": True,
                "preset": path.name,
                "derived_layout_hash": derived.layout_hash,
            },
        )
    explicit = load_policy_io_layout(request)
    assert explicit is not None
    if _semantic_layout_signature(explicit) != _semantic_layout_signature(derived):
        raise ValueError(
            "--policy-io-layout does not match the instantiated backend/retargeter "
            "contract. Use --policy-io-layout auto or provide a semantically identical layout."
        )
    return ResolvedCollectionPolicyIOLayout(
        layout=explicit,
        provenance={
            "mode": "explicit",
            "strict": True,
            "preset": str(Path(request)),
            "derived_layout_hash": derived.layout_hash,
        },
    )


__all__ = [
    "AUTO_POLICY_IO_LAYOUT",
    "DEFAULT_POLICY_IO_LAYOUT_PRESET_DIR",
    "ResolvedCollectionPolicyIOLayout",
    "derive_collection_policy_io_layout",
    "resolve_collection_policy_io_layout",
]
