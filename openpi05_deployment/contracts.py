from __future__ import annotations

from dataclasses import dataclass

from openpi05_deployment.errors import ServerMetadataError


OPENPI_LOCAL_PROTOCOL_VERSION = "openpi.local_policy.websocket.v1"
TACTILE_KIND_NONE = "none"
TACTILE_KIND_VECTOR = "vector"


@dataclass(frozen=True)
class EmbodimentContract:
    name: str
    aliases: tuple[str, ...]
    active_sides: tuple[str, ...]
    state_dim: int
    output_action_dim: int
    tactile_token_order: tuple[str, ...]
    action_semantics: str


_SUPPORTED_EMBODIMENTS = (
    EmbodimentContract(
        name="realman_dual_arm_hands",
        aliases=("realman_dual_arm_hand",),
        active_sides=("left", "right"),
        state_dim=26,
        output_action_dim=26,
        tactile_token_order=("left", "right"),
        action_semantics=(
            "left arm end-effector pose (7) + right arm end-effector pose (7) + "
            "left hand joints (6) + right hand joints (6)"
        ),
    ),
    EmbodimentContract(
        name="realman_right_arm_right_hand",
        aliases=("realman_right_arm_hand",),
        active_sides=("right",),
        state_dim=13,
        output_action_dim=13,
        tactile_token_order=("right",),
        action_semantics="right arm end-effector pose (7) + right hand joints (6)",
    ),
)

_EMBODIMENT_INDEX: dict[str, EmbodimentContract] = {}
for _contract in _SUPPORTED_EMBODIMENTS:
    _EMBODIMENT_INDEX[_contract.name] = _contract
    for _alias in _contract.aliases:
        _EMBODIMENT_INDEX[_alias] = _contract


def normalize_embodiment_name(value: str) -> str:
    return str(value or "").strip().lower()


def resolve_embodiment_contract(value: str) -> EmbodimentContract:
    name = normalize_embodiment_name(value)
    if not name:
        raise ServerMetadataError("Embodiment name is required.")
    try:
        return _EMBODIMENT_INDEX[name]
    except KeyError as exc:
        supported = sorted(contract.name for contract in _SUPPORTED_EMBODIMENTS)
        raise ServerMetadataError(
            f"Unsupported embodiment {value!r}. Supported embodiments: {supported}."
        ) from exc
