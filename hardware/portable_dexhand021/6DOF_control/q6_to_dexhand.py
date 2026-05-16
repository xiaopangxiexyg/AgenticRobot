"""Map a 6D hand command onto the 12-DoF DexHand joint interface.

Internal q6 convention used by this wrapper:

    [thumb_flex, thumb_rot, index, middle, ring, little]

Each flexion dimension is normalized to 0..1, where 0 means open and 1 means
closed. The spread joint is intentionally not controlled and is always sent as
``ff_spr = spread_angle_deg``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portable_dexhand021 import DexHandController


INTERNAL_Q6_ORDER = (
    "thumb_flex",
    "thumb_rot",
    "index",
    "middle",
    "ring",
    "little",
)

TELEOP_READER_Q6_ORDER = (
    "thumb_rot",
    "thumb_flex",
    "index",
    "middle",
    "ring",
    "little",
)

DEXHAND_JOINT_ORDER = (
    "th_dip",
    "th_mcp",
    "th_rot",
    "ff_spr",
    "ff_dip",
    "ff_mcp",
    "mf_dip",
    "mf_mcp",
    "rf_dip",
    "rf_mcp",
    "lf_dip",
    "lf_mcp",
)

DEFAULT_MIN_DEG = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DEFAULT_MAX_DEG = (90.0, 150.0, 90.0, 90.0, 90.0, 90.0)
DEFAULT_FINGER_PROFILE_PATH = Path(__file__).resolve().with_name("current_joint_angles_lh2tsvng.json")
FOUR_FINGER_NAMES = ("index", "middle", "ring", "little")
FOUR_FINGER_JOINTS = {
    "index": ("ff_dip", "ff_mcp"),
    "middle": ("mf_dip", "mf_mcp"),
    "ring": ("rf_dip", "rf_mcp"),
    "little": ("lf_dip", "lf_mcp"),
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def parse_float_vector(text: str, length: int, name: str) -> Tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) == 1:
        return tuple([float(parts[0])] * length)
    if len(parts) != length:
        raise ValueError(f"{name} must contain 1 or {length} comma-separated floats")
    return tuple(float(part) for part in parts)


def reorder_q6(q6: Sequence[float], source_order: str = "internal") -> Tuple[float, ...]:
    """Return q6 in the internal order.

    ``teleop`` means the order produced by ``teleop_hand/hand_q6_reader.py``:
    ``[thumb_rot, thumb_flex, index, middle, ring, little]``.
    """
    values = tuple(float(v) for v in q6)
    if len(values) != 6:
        raise ValueError(f"q6 must contain 6 values, got {len(values)}")

    if source_order == "internal":
        return values
    if source_order == "teleop":
        return (values[1], values[0], values[2], values[3], values[4], values[5])
    raise ValueError("source_order must be 'internal' or 'teleop'")


def load_four_finger_mcp_to_dip_ratios(path: str | Path = DEFAULT_FINGER_PROFILE_PATH) -> Dict[str, float]:
    """Load four-finger MCP/DIP angle ratios from a saved joint-angle JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    angles = payload.get("angles_deg")
    if not isinstance(angles, dict):
        raise ValueError(f"{path} does not contain an angles_deg object")

    ratios = {}
    for finger, (dip_joint, mcp_joint) in FOUR_FINGER_JOINTS.items():
        dip = float(angles[dip_joint])
        mcp = float(angles[mcp_joint])
        if abs(dip) < 1e-6:
            raise ValueError(f"{path} has near-zero {dip_joint}; cannot compute {mcp_joint}/{dip_joint}")
        ratios[finger] = mcp / dip
    return ratios


@dataclass
class DexHandQ6Mapper:
    """Pure 6D-to-DexHand mapper."""

    min_deg: Sequence[float] = DEFAULT_MIN_DEG
    max_deg: Sequence[float] = DEFAULT_MAX_DEG
    spread_angle_deg: float = 0.0
    invert: bool = False
    gain: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    bias: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    four_finger_mode: str = "profile"
    four_finger_profile_path: str | Path = DEFAULT_FINGER_PROFILE_PATH

    def __post_init__(self) -> None:
        self.min_deg = tuple(float(v) for v in self.min_deg)
        self.max_deg = tuple(float(v) for v in self.max_deg)
        self.gain = tuple(float(v) for v in self.gain)
        self.bias = tuple(float(v) for v in self.bias)
        for name, values in (
            ("min_deg", self.min_deg),
            ("max_deg", self.max_deg),
            ("gain", self.gain),
            ("bias", self.bias),
        ):
            if len(values) != 6:
                raise ValueError(f"{name} must contain 6 values")
        if self.four_finger_mode not in {"same", "profile"}:
            raise ValueError("four_finger_mode must be 'same' or 'profile'")
        self.four_finger_ratios = (
            load_four_finger_mcp_to_dip_ratios(self.four_finger_profile_path)
            if self.four_finger_mode == "profile"
            else {finger: 1.0 for finger in FOUR_FINGER_NAMES}
        )

    def normalize_q6(self, q6: Sequence[float], source_order: str = "internal") -> Tuple[float, ...]:
        internal = reorder_q6(q6, source_order=source_order)
        out = []
        for idx, value in enumerate(internal):
            x = 1.0 - float(value) if self.invert else float(value)
            x = x * self.gain[idx] + self.bias[idx]
            out.append(clamp(x))
        return tuple(out)

    def normalized_q6_to_angles(self, q6_internal: Sequence[float]) -> Dict[str, float]:
        q = tuple(clamp(value) for value in q6_internal)
        if len(q) != 6:
            raise ValueError(f"q6 must contain 6 values, got {len(q)}")
        return {
            name: self.min_deg[idx] + q[idx] * (self.max_deg[idx] - self.min_deg[idx])
            for idx, name in enumerate(INTERNAL_Q6_ORDER)
        }

    def q6_to_angles(self, q6: Sequence[float], source_order: str = "internal") -> Dict[str, float]:
        return self.normalized_q6_to_angles(self.normalize_q6(q6, source_order=source_order))

    def normalized_q6_to_dexhand_action(self, q6_internal: Sequence[float]) -> Dict[str, float]:
        angles = self.normalized_q6_to_angles(q6_internal)
        thumb = angles["thumb_flex"]
        four_finger_targets = self._four_finger_targets(angles)

        return {
            "th_dip": thumb,
            "th_mcp": thumb,
            "th_rot": angles["thumb_rot"],
            "ff_spr": float(self.spread_angle_deg),
            **four_finger_targets,
        }

    def q6_to_dexhand_action(self, q6: Sequence[float], source_order: str = "internal") -> Dict[str, float]:
        return self.normalized_q6_to_dexhand_action(self.normalize_q6(q6, source_order=source_order))

    def _four_finger_targets(self, angles: Dict[str, float]) -> Dict[str, float]:
        targets = {}
        for finger, (dip_joint, mcp_joint) in FOUR_FINGER_JOINTS.items():
            dip = clamp(angles[finger], 0.0, 90.0)
            mcp = clamp(dip * self.four_finger_ratios[finger], 0.0, 90.0)
            targets[dip_joint] = dip
            targets[mcp_joint] = mcp
        return targets


class DexHandQ6Controller:
    """Thin hardware wrapper around :class:`DexHandController` plus q6 mapping."""

    def __init__(
        self,
        *,
        mapper: Optional[DexHandQ6Mapper] = None,
        controller: Optional[DexHandController] = None,
        **controller_kwargs,
    ) -> None:
        self.q6_reader = controller_kwargs.pop("q6_reader", None)
        self.mapper = mapper or DexHandQ6Mapper()
        self.controller = controller or DexHandController(**controller_kwargs)
        self.last_source_q6: Optional[Tuple[float, ...]] = None
        self.last_q6: Optional[Tuple[float, ...]] = None
        self.last_action: Optional[Dict[str, float]] = None

    def connect(self) -> "DexHandQ6Controller":
        self.controller.connect()
        return self

    def send_q6(
        self,
        q6: Sequence[float],
        *,
        source_order: str = "internal",
        **send_kwargs,
    ) -> Dict[str, float]:
        self.last_source_q6 = tuple(float(v) for v in q6)
        self.last_q6 = self.mapper.normalize_q6(q6, source_order=source_order)
        action = self.mapper.normalized_q6_to_dexhand_action(self.last_q6)
        self.controller.send_action(action, **send_kwargs)
        self.last_action = action
        return action

    def read_q6(self, *, source_order: str = "internal") -> Tuple[float, ...]:
        """Read one 6D action from the configured q6_reader and normalize it."""
        if self.q6_reader is None:
            raise RuntimeError("DexHandQ6Controller was created without a q6_reader")
        raw_q6 = self.q6_reader.read_q6()
        self.last_source_q6 = tuple(float(v) for v in raw_q6)
        self.last_q6 = self.mapper.normalize_q6(raw_q6, source_order=source_order)
        return self.last_q6

    def reproduce_q6(
        self,
        q6: Optional[Sequence[float]] = None,
        *,
        source_order: str = "internal",
        **send_kwargs,
    ) -> Dict[str, float]:
        """Send ``q6`` or replay the most recently read normalized 6D action."""
        if q6 is not None:
            return self.send_q6(q6, source_order=source_order, **send_kwargs)
        if self.last_q6 is None:
            self.read_q6(source_order=source_order)
        assert self.last_q6 is not None
        action = self.mapper.normalized_q6_to_dexhand_action(self.last_q6)
        self.controller.send_action(action, **send_kwargs)
        self.last_action = action
        return action

    def read_state(self) -> Dict[str, object]:
        return self.controller.read_state()

    def close(self) -> None:
        self.controller.close()

    def __enter__(self) -> "DexHandQ6Controller":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
