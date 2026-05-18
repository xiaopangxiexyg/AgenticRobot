from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from paths import UV_CONVERSION_ROOT, ensure_import_paths


ensure_import_paths()

_REGION_NAMES = [
    "little_1",
    "little_2",
    "little_3",
    "ring_1",
    "ring_2",
    "ring_3",
    "middle_1",
    "middle_2",
    "middle_3",
    "index_1",
    "index_2",
    "index_3",
    "thumb_1",
    "thumb_2",
    "thumb_3",
    "thumb_4",
    "palm",
]
_REGION_SIZES = [
    (3, 3),
    (12, 8),
    (10, 8),
    (3, 3),
    (12, 8),
    (10, 8),
    (3, 3),
    (12, 8),
    (10, 8),
    (3, 3),
    (12, 8),
    (10, 8),
    (3, 3),
    (12, 8),
    (3, 3),
    (12, 8),
    (8, 14),
]
_REGION_STARTS = np.asarray(
    [0] + [sum(rows * cols for rows, cols in _REGION_SIZES[:index]) for index in range(1, len(_REGION_SIZES) + 1)],
    dtype=np.int64,
)
_REPEAT_PAIRS_UV891: list[tuple[int, int]] = [
    (5, 6),
    (7, 8),
    (12, 13),
    (14, 15),
    (17, 18),
    (34, 35),
    (36, 37),
    (43, 44),
    (45, 46),
    (49, 50),
    (67, 68),
    (69, 70),
    (116, 117),
    (118, 119),
    (136, 137),
    (142, 143),
    (150, 151),
    (152, 153),
    (179, 180),
    (181, 182),
    (194, 195),
    (198, 199),
    (207, 208),
    (244, 245),
    (246, 247),
    (250, 251),
    (272, 273),
    (274, 275),
    (276, 277),
    (278, 279),
    (288, 289),
    (293, 294),
    (295, 296),
    (306, 307),
    (311, 312),
    (328, 329),
    (333, 334),
    (336, 337),
    (340, 341),
    (360, 361),
    (362, 363),
    (364, 365),
    (366, 367),
    (376, 377),
    (380, 381),
    (382, 383),
    (398, 399),
    (408, 409),
    (410, 411),
    (416, 417),
    (418, 419),
    (432, 433),
    (436, 437),
    (443, 444),
    (445, 446),
    (449, 450),
    (457, 458),
    (466, 467),
    (469, 470),
    (473, 474),
    (491, 492),
    (493, 494),
    (495, 496),
    (497, 498),
    (508, 509),
    (513, 514),
    (515, 516),
    (531, 532),
    (541, 542),
    (543, 544),
    (549, 550),
    (551, 552),
    (563, 564),
    (567, 568),
    (574, 575),
    (576, 577),
    (580, 581),
    (589, 590),
    (598, 599),
    (601, 602),
    (605, 606),
    (623, 624),
    (625, 626),
    (627, 628),
    (629, 630),
    (640, 641),
    (645, 646),
    (647, 648),
    (663, 664),
    (674, 675),
    (676, 677),
    (682, 683),
    (684, 685),
    (722, 723),
    (731, 732),
    (754, 755),
    (756, 757),
    (758, 759),
    (760, 761),
    (771, 772),
    (797, 798),
    (804, 805),
    (812, 813),
    (815, 816),
    (819, 820),
    (837, 838),
    (839, 840),
    (841, 842),
    (843, 844),
    (854, 855),
    (859, 860),
    (861, 862),
    (877, 878),
]


def _assets_dir() -> Path:
    root = Path(UV_CONVERSION_ROOT).expanduser().resolve()
    assets = root / "uv_conversion" / "assets"
    if not assets.exists():
        raise FileNotFoundError(f"UV_conversion assets not found: {assets}")
    return assets


def _original_inspire_tac_vec_to_dict(tac_vec: np.ndarray, side: str) -> dict[str, np.ndarray]:
    tac_dict: dict[str, np.ndarray] = {}
    is_left = str(side).lower() == "left"
    for index, region_name in enumerate(_REGION_NAMES):
        region_size = _REGION_SIZES[index]
        start = int(_REGION_STARTS[index])
        end = int(_REGION_STARTS[index + 1])
        region_data = np.asarray(tac_vec[start:end]).reshape(region_size)
        if region_name.endswith("_1") or region_name == "thumb_3" or region_name == "palm":
            region_data = region_data.reshape(region_size[1], region_size[0]).T
        elif region_name == "thumb_4":
            region_data = region_data[::-1, ::-1]
        if is_left:
            region_data = np.flip(region_data, axis=1)
        tac_dict[region_name] = region_data.reshape(-1)
    return tac_dict


def _seq_inspire_vec_from_dict(tac_dict: dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros((1062,), dtype=np.float32)
    for index, region_name in enumerate(_REGION_NAMES):
        start = int(_REGION_STARTS[index])
        end = int(_REGION_STARTS[index + 1])
        out[start:end] = np.asarray(tac_dict[region_name]).reshape(-1)
    return out


@lru_cache(maxsize=2)
def _correct_order(side: str) -> np.ndarray:
    probe = np.arange(1062, dtype=np.float32)
    return _seq_inspire_vec_from_dict(_original_inspire_tac_vec_to_dict(probe, side)).astype(np.int64)


def _repeat_pairs_for_side(side: str) -> list[tuple[int, int]]:
    return list(_REPEAT_PAIRS_UV891)


def _mano778_to_uv891(values: np.ndarray, side: str, mappings_dir: Path) -> np.ndarray:
    hand = side.upper()
    uv891_to_uv778 = np.asarray(json.loads((mappings_dir / f"inverse_indices_{hand}.json").read_text()), dtype=np.int64)
    mappings = json.loads((mappings_dir / f"mappings_{hand}.json").read_text())
    mano778_to_uv778 = np.asarray(mappings["geom_to_template_uv_map"], dtype=np.int64)
    uv778 = values[:, mano778_to_uv778]
    return uv778[:, uv891_to_uv778]


def _apply_index_convert(data: np.ndarray, convert_np: np.ndarray) -> np.ndarray:
    if convert_np.size == 0:
        return data
    out = data.copy()
    out[..., convert_np[:, 1]] = out[..., convert_np[:, 0]]
    return out


class _LightweightInspireTactileConverter:
    def __init__(self) -> None:
        assets = _assets_dir()
        mappings_dir = assets / "mappings"
        self._mappings_dir = mappings_dir
        self._inspire_weights = np.asarray(np.load(assets / "all_inspire_mappings.npz")["arr"], dtype=np.float32)
        self._glove_weights = np.asarray(np.load(mappings_dir / "tactile_to_vertex_matrix.npz")["arr_0"], dtype=np.float32)
        if self._inspire_weights.shape != (1062, 778):
            raise ValueError(f"Expected Inspire mapping shape (1062, 778), got {self._inspire_weights.shape}.")
        if self._glove_weights.shape != (137, 891):
            raise ValueError(f"Expected glove mapping shape (137, 891), got {self._glove_weights.shape}.")

        self.repeat_single_indices: dict[str, list[int]] = {"left": [], "right": []}
        self.zero_indices: dict[str, list[int]] = {"left": [], "right": []}
        self.inspire_converts_np: dict[str, np.ndarray] = {
            "left": np.empty((0, 2), dtype=np.int32),
            "right": np.empty((0, 2), dtype=np.int32),
        }
        self.glove_converts_np: dict[str, np.ndarray] = {
            "left": np.empty((0, 2), dtype=np.int32),
            "right": np.empty((0, 2), dtype=np.int32),
        }
        self._init_masks()
        valid = []
        for side in ("right", "left"):
            glove = self._glove137_to_uv891(np.ones((1, 137), dtype=np.float32), side, mask=True)[0]
            valid.append(np.where(glove != -1.0)[0])
        self.union_idx = np.union1d(valid[0], valid[1]).astype(np.int64)
        if self.union_idx.shape != (351,):
            raise RuntimeError(f"Expected 351 UV-union indices, got {self.union_idx.shape}.")

    def _glove137_to_uv891(self, values: np.ndarray, side: str, *, mask: bool) -> np.ndarray:
        uv891 = np.clip(np.asarray(values, dtype=np.float32) @ self._glove_weights, 0.0, 1.0)
        uv891 = _apply_index_convert(uv891, self.glove_converts_np[side])
        if mask:
            uv891[..., self.repeat_single_indices[side]] = -1.0
            uv891[..., self.zero_indices[side]] = -1.0
        return uv891

    def _inspire1062_to_uv891(self, values: np.ndarray, side: str, *, correct: bool, mask: bool) -> np.ndarray:
        safe = np.asarray(values, dtype=np.float32)
        if correct:
            safe = safe[..., _correct_order(side)]
        mano778 = safe @ self._inspire_weights
        uv891 = _mano778_to_uv891(mano778, side, self._mappings_dir)
        uv891 = _apply_index_convert(uv891, self.inspire_converts_np[side])
        uv891 = np.clip(uv891, 0.0, 1.0)
        if mask:
            uv891[..., self.repeat_single_indices[side]] = -1.0
            uv891[..., self.zero_indices[side]] = -1.0
        return uv891

    def _init_masks(self) -> None:
        glove_probe = np.ones((1, 137), dtype=np.float32)
        inspire_probe = np.ones((1, 1062), dtype=np.float32)
        for side in ("left", "right"):
            glove_uv891 = self._glove137_to_uv891(glove_probe, side, mask=False)[0]
            inspire_uv891 = self._inspire1062_to_uv891(inspire_probe, side, correct=False, mask=False)[0]
            glove_zero = glove_uv891 == 0
            inspire_zero = inspire_uv891 == 0
            zero_indices = set(np.where(glove_zero & inspire_zero)[0].tolist())
            non_zero_indices = set(np.where(~glove_zero & ~inspire_zero)[0].tolist())
            glove_zero_indices = set(np.where(glove_zero)[0].tolist())
            inspire_zero_indices = set(np.where(inspire_zero)[0].tolist())

            glove_converts: list[list[int]] = []
            inspire_converts: list[list[int]] = []
            repeat_single: list[int] = []
            for source_index, target_index in _repeat_pairs_for_side(side):
                repeat_single.append(source_index)
                if source_index in zero_indices:
                    continue
                if target_index in zero_indices:
                    glove_converts.append([source_index, target_index])
                    inspire_converts.append([source_index, target_index])
                elif target_index in non_zero_indices:
                    continue
                elif source_index in non_zero_indices:
                    glove_converts.append([source_index, target_index])
                    inspire_converts.append([source_index, target_index])
                elif source_index in inspire_zero_indices:
                    glove_converts.append([source_index, target_index])
                else:
                    inspire_converts.append([source_index, target_index])

            self.repeat_single_indices[side] = repeat_single
            self.glove_converts_np[side] = np.asarray(glove_converts, dtype=np.int32)
            self.inspire_converts_np[side] = np.asarray(inspire_converts, dtype=np.int32)
            glove_uv891 = _apply_index_convert(glove_uv891, self.glove_converts_np[side])
            inspire_uv891 = _apply_index_convert(inspire_uv891, self.inspire_converts_np[side])
            self.zero_indices[side] = np.where((glove_uv891 == 0) & (inspire_uv891 == 0))[0].tolist()

    def to_351(self, values: np.ndarray, side: str) -> np.ndarray:
        hand_side = "left" if str(side).lower() == "left" else "right"
        uv891 = self._inspire1062_to_uv891(values, hand_side, correct=True, mask=True)
        raw = uv891[..., self.union_idx]
        valid = np.isfinite(raw) & (raw != -1.0)
        return np.where(valid, np.clip(raw, 0.0, 1.0), 0.0).astype(np.float32)


@lru_cache(maxsize=1)
def _converter() -> _LightweightInspireTactileConverter:
    return _LightweightInspireTactileConverter()


def tactile_to_351(value: Any, side: str) -> np.ndarray:
    if value is None:
        return np.zeros((351,), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0 or arr.size == 0:
        return np.zeros((351,), dtype=np.float32)
    if arr.ndim == 1:
        arr2 = arr[None, :]
        squeeze = True
    else:
        arr2 = arr.reshape((-1, arr.shape[-1]))
        squeeze = False

    if arr2.shape[-1] == 351:
        valid = np.isfinite(arr2) & (arr2 != -1.0)
        out = np.where(valid, np.clip(arr2, 0.0, 1.0), 0.0).astype(np.float32)
    elif arr2.shape[-1] == 1062:
        safe = np.nan_to_num(arr2, nan=0.0, posinf=0.0, neginf=0.0)
        out = _converter().to_351(safe, side)
    else:
        raise ValueError(f"{side} tactile expected 351D or 1062D, got {arr2.shape[-1]}D.")

    return out[0] if squeeze else out.reshape(arr.shape[:-1] + (351,))


def tactile_to_dmtac_history_frame(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((2, 3, 144, 192), dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] in (
        144 * 192 * 3,
        2 * 144 * 192 * 3,
        288 * 384 * 3,
        2 * 288 * 384 * 3,
    ):
        if arr.shape[0] % (288 * 384 * 3) == 0:
            sensors = arr.shape[0] // (288 * 384 * 3)
            frame = _dmtac_pool_2x2(arr.reshape(sensors, 288, 384, 3))
        else:
            sensors = arr.shape[0] // (144 * 192 * 3)
            frame = arr.reshape(sensors, 144, 192, 3)
        return np.moveaxis(frame, -1, -3).astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape == (144, 192, 3):
        return np.moveaxis(arr[None, ...], -1, -3).astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape == (288, 384, 3):
        return np.moveaxis(_dmtac_pool_2x2(arr)[None, ...], -1, -3).astype(np.float32, copy=False)
    if arr.ndim == 4 and arr.shape[-3:] == (144, 192, 3):
        return np.moveaxis(arr, -1, -3).astype(np.float32, copy=False)
    if arr.ndim == 4 and arr.shape[-3:] == (288, 384, 3):
        return np.moveaxis(_dmtac_pool_2x2(arr), -1, -3).astype(np.float32, copy=False)
    if arr.ndim == 4 and arr.shape[-3:] == (3, 144, 192):
        return arr.astype(np.float32, copy=True)
    if arr.ndim == 2 and arr.shape[-1] in (
        144 * 192 * 3,
        2 * 144 * 192 * 3,
        288 * 384 * 3,
        2 * 288 * 384 * 3,
    ):
        return tactile_to_dmtac_history_frame(arr[-1])
    if arr.ndim == 5 and arr.shape[-3:] in ((144, 192, 3), (3, 144, 192)):
        return tactile_to_dmtac_history_frame(arr[-1])
    if arr.ndim == 5 and arr.shape[-3:] == (288, 384, 3):
        return tactile_to_dmtac_history_frame(arr[-1])
    raise ValueError(f"DM-Tac tactile expected image-like frame/history, got {arr.shape}.")


def _dmtac_pool_2x2(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[-3:-1] != (288, 384):
        return arr.astype(np.float32, copy=True)
    new_shape = (*arr.shape[:-3], 144, 2, 192, 2, arr.shape[-1])
    return arr.reshape(new_shape).mean(axis=(-4, -2), dtype=np.float32).astype(np.float32, copy=False)


class TactileHistory:
    def __init__(self, history_size: int, history_stride: int):
        self.history_size = max(1, int(history_size))
        self.history_stride = max(1, int(history_stride))
        self._buffers: dict[str, list[np.ndarray]] = {"left": [], "right": []}

    def append(self, side: str, tactile: Any) -> None:
        side = str(side).lower()
        vec = tactile_to_351(tactile, side)
        buf = self._buffers.setdefault(side, [])
        buf.append(vec.astype(np.float32, copy=True))
        keep = self.history_size * self.history_stride + 1
        if len(buf) > keep:
            del buf[: len(buf) - keep]

    def stack(self, side: str) -> np.ndarray:
        side = str(side).lower()
        buf = self._buffers.setdefault(side, [])
        if not buf:
            return np.zeros((self.history_size, 351), dtype=np.float32)
        samples = []
        for offset in range(self.history_size - 1, -1, -1):
            idx = len(buf) - 1 - offset * self.history_stride
            idx = max(0, min(idx, len(buf) - 1))
            samples.append(buf[idx])
        return np.stack(samples, axis=0).astype(np.float32)


class DMTacTactileHistory:
    def __init__(self, history_size: int, history_stride: int, default_sensors: int = 2):
        self.history_size = max(1, int(history_size))
        self.history_stride = max(1, int(history_stride))
        self.default_sensors = max(1, int(default_sensors))
        self._buffers: dict[str, list[np.ndarray]] = {"left": [], "right": []}

    def append(self, side: str, tactile: Any) -> None:
        side = str(side).lower()
        frame = tactile_to_dmtac_history_frame(tactile)
        buf = self._buffers.setdefault(side, [])
        buf.append(frame.astype(np.float32, copy=True))
        keep = self.history_size * self.history_stride + 1
        if len(buf) > keep:
            del buf[: len(buf) - keep]

    def stack(self, side: str) -> np.ndarray:
        side = str(side).lower()
        buf = self._buffers.setdefault(side, [])
        if not buf:
            return np.zeros((self.history_size, self.default_sensors, 3, 144, 192), dtype=np.float32)
        samples = []
        for offset in range(self.history_size - 1, -1, -1):
            idx = len(buf) - 1 - offset * self.history_stride
            idx = max(0, min(idx, len(buf) - 1))
            samples.append(buf[idx])
        return np.stack(samples, axis=0).astype(np.float32)
