from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _install_numpy_legacy_aliases() -> None:
    # chumpy in the bundled UV converter still imports NumPy aliases removed in
    # newer NumPy versions. Keep the workaround local to the conversion import.
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _resolve_uv_conversion_root(root: str | Path) -> Path:
    path = Path(root).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path.resolve()


class InspireTactileUVConverter:
    """Convert Inspire 1062 tactile vectors to compact valid UV features."""

    def __init__(
        self,
        uv_conversion_root: str | Path = "PolicyDeploy/UV_conversion",
        hand: str = "right",
        expected_dim: int | None = 351,
        correct_order: bool = True,
    ) -> None:
        self.uv_conversion_root = _resolve_uv_conversion_root(uv_conversion_root)
        self.hand = str(hand).lower()
        if self.hand not in {"left", "right"}:
            raise ValueError(f"Unsupported hand={hand!r}; expected 'left' or 'right'.")
        self.expected_dim = None if expected_dim is None else int(expected_dim)
        self.correct_order = bool(correct_order)

        os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "maniprep_matplotlib"))
        _install_numpy_legacy_aliases()
        if str(self.uv_conversion_root) not in sys.path:
            sys.path.insert(0, str(self.uv_conversion_root))

        from uv_conversion.universal_UV_converter import (  # type: ignore
            ASSETS_DIR,
            MAPPINGS_DIR,
            UV_MAPPING_CONFIG,
            HandType,
            InspireMANOConverter,
            TactileGloveUVMapper,
            VertexConverter,
            correct_inspire_vec_from_1d,
            load_mano_data,
        )

        self._hand_types = {"left": HandType.LEFT, "right": HandType.RIGHT}
        self._hand_type = self._hand_types[self.hand]
        self._assets_dir = ASSETS_DIR
        self._load_mano_data = load_mano_data
        self._correct_inspire_vec_from_1d = correct_inspire_vec_from_1d
        self.glove_mapper = TactileGloveUVMapper(
            UV_MAPPING_CONFIG=UV_MAPPING_CONFIG,
            mano_obj_model_dir=ASSETS_DIR,
        )
        self.inspire_mapper = InspireMANOConverter(os.path.join(ASSETS_DIR, "all_inspire_mappings.npz"))
        self.mano_mapper = VertexConverter(mapping_dir=MAPPINGS_DIR)
        self.repeat_single_indices: dict[str, list[int]] = {"left": [], "right": []}
        self.zero_indices: dict[str, list[int]] = {"left": [], "right": []}
        self.non_zero_indices: dict[str, list[int]] = {"left": [], "right": []}
        self.glove_converts_np: dict[str, np.ndarray] = {
            "left": np.empty((0, 2), dtype=np.int32),
            "right": np.empty((0, 2), dtype=np.int32),
        }
        self.inspire_converts_np: dict[str, np.ndarray] = {
            "left": np.empty((0, 2), dtype=np.int32),
            "right": np.empty((0, 2), dtype=np.int32),
        }
        self._correct_orders: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._mask_indices_inspire_glove()
        self.hand_valid_indices: dict[str, np.ndarray] = {}
        for hand_key, hand_type in self._hand_types.items():
            masked_probe = self._inspire1062_to_uv891(
                np.ones((1062,), dtype=np.float32),
                correct=False,
                mask=True,
                hand_type=hand_type,
            )
            self.hand_valid_indices[hand_key] = np.where(masked_probe != -1.0)[0].astype(np.int64)
        self.valid_indices = np.union1d(self.hand_valid_indices["left"], self.hand_valid_indices["right"]).astype(np.int64)
        self.actual_valid_dim = int(self.hand_valid_indices[self.hand].size)
        self.union_valid_dim = int(self.valid_indices.size)
        if self.expected_dim is not None and self.union_valid_dim != self.expected_dim:
            raise ValueError(
                f"UV converter produced {self.union_valid_dim} union tactile dims from left/right valid UV indices, "
                f"expected {self.expected_dim}."
            )
        self.target_dim = self.union_valid_dim

    @property
    def output_dim(self) -> int:
        return int(self.target_dim)

    def convert(self, tactile1062: Any) -> np.ndarray:
        values = np.asarray(tactile1062, dtype=np.float32)
        if values.shape[-1] != 1062:
            raise ValueError(f"Expected Inspire tactile dim=1062, got shape={values.shape}.")
        original_prefix = values.shape[:-1]
        flat = values.reshape(-1, 1062)
        uv891 = self._inspire1062_to_uv891(flat, correct=self.correct_order, mask=True, hand_type=self._hand_type)
        compact = uv891[:, self.valid_indices].astype(np.float32, copy=False)
        compact = np.where(compact < 0.0, 0.0, compact)
        return compact.reshape(*original_prefix, self.output_dim)

    def _build_correct_order(self, hand_type: Any) -> np.ndarray:
        probe = np.arange(1062, dtype=np.float32)
        order = self._correct_inspire_vec_from_1d(probe, hand_type=hand_type)
        return np.asarray(order, dtype=np.int64)

    def _correct_order_for(self, hand_type: Any) -> np.ndarray:
        hand_key = hand_type.lower()
        correct_order = self._correct_orders[hand_key]
        if correct_order is None:
            correct_order = self._build_correct_order(hand_type)
            self._correct_orders[hand_key] = correct_order
        return correct_order

    def _apply_indice_convert(self, data: np.ndarray, hand_type: Any, data_type: str) -> np.ndarray:
        if data_type == "glove":
            convert_np = self.glove_converts_np[hand_type.lower()]
        elif data_type == "inspire":
            convert_np = self.inspire_converts_np[hand_type.lower()]
        else:
            raise ValueError(f"Unknown data_type={data_type!r}")
        if len(convert_np) == 0:
            return data
        data = data.copy()
        data[..., convert_np[:, 1]] = data[..., convert_np[:, 0]]
        return data

    def _glove137_to_uv891(self, glove137_data: np.ndarray, hand_type: Any, mask: bool) -> np.ndarray:
        flat_input = glove137_data.ndim == 1
        values = glove137_data[np.newaxis, :] if flat_input else glove137_data
        _, uv891 = self.glove_mapper.map_all_sensors(
            values,
            hand_type=hand_type,
            uv_map_shape=(512, 512),
            return_img=False,
            with_bend=False,
        )
        uv891 = np.clip(uv891, 0.0, 1.0)
        uv891 = self._apply_indice_convert(uv891, hand_type=hand_type, data_type="glove")
        if mask:
            hand_key = hand_type.lower()
            uv891[..., self.repeat_single_indices[hand_key]] = -1.0
            uv891[..., self.zero_indices[hand_key]] = -1.0
        return uv891[0] if flat_input else uv891

    def _inspire1062_to_uv891(
        self,
        inspire1062_data: np.ndarray,
        correct: bool,
        mask: bool,
        hand_type: Any | None = None,
    ) -> np.ndarray:
        resolved_hand_type = self._hand_type if hand_type is None else hand_type
        flat_input = inspire1062_data.ndim == 1
        values = inspire1062_data[np.newaxis, :] if flat_input else inspire1062_data
        if correct:
            values = values[..., self._correct_order_for(resolved_hand_type)]
        vertice_pressure = self.inspire_mapper.inspire_tac_to_mano_vertices(values)
        uv891 = self.mano_mapper.convert_values(
            vertice_pressure,
            from_space="mano_778",
            to_space="uvmap_891",
            hand_type=resolved_hand_type,
        )
        uv891 = self._apply_indice_convert(uv891, hand_type=resolved_hand_type, data_type="inspire")
        uv891 = np.clip(uv891, 0.0, 1.0)
        if mask:
            hand_key = resolved_hand_type.lower()
            uv891[..., self.repeat_single_indices[hand_key]] = -1.0
            uv891[..., self.zero_indices[hand_key]] = -1.0
        return uv891[0] if flat_input else uv891

    def _mask_indices_inspire_glove(self) -> None:
        glove137_data = np.ones((137,), dtype=np.float32)
        inspire1062_data = np.ones((1062,), dtype=np.float32)
        for hand_type in self._hand_types.values():
            hand_key = hand_type.lower()
            glove_uv891 = self._glove137_to_uv891(glove137_data, hand_type=hand_type, mask=False)
            inspire_uv891 = self._inspire1062_to_uv891(
                inspire1062_data,
                correct=False,
                mask=False,
                hand_type=hand_type,
            )

            glove_zero = glove_uv891 == 0
            inspire_zero = inspire_uv891 == 0
            self.zero_indices[hand_key] = np.where(glove_zero & inspire_zero)[0].tolist()
            self.non_zero_indices[hand_key] = np.where(~glove_zero & ~inspire_zero)[0].tolist()

            glove_zero_indices = set(np.where(glove_zero)[0].tolist())
            inspire_zero_indices = set(np.where(inspire_zero)[0].tolist())

            _, mesh = self._load_mano_data(os.path.join(self._assets_dir, f"MANO_UV_{hand_key}.obj"))
            vertices = mesh.vertices
            repeat_pairs: list[tuple[int, int]] = []
            for left_index in range(890):
                for right_index in range(left_index + 1, 891):
                    if (
                        vertices[left_index][0] == vertices[right_index][0]
                        and vertices[left_index][1] == vertices[right_index][1]
                        and vertices[left_index][2] == vertices[right_index][2]
                    ):
                        repeat_pairs.append((left_index, right_index))

            glove_converts: list[list[int]] = []
            inspire_converts: list[list[int]] = []
            self.repeat_single_indices[hand_key] = []
            for source_index, target_index in repeat_pairs:
                self.repeat_single_indices[hand_key].append(source_index)
                if source_index in self.zero_indices[hand_key]:
                    continue
                if target_index in self.zero_indices[hand_key]:
                    glove_converts.append([source_index, target_index])
                    inspire_converts.append([source_index, target_index])
                    continue
                if target_index in self.non_zero_indices[hand_key]:
                    continue
                if source_index in self.non_zero_indices[hand_key]:
                    glove_converts.append([source_index, target_index])
                    inspire_converts.append([source_index, target_index])
                elif source_index in inspire_zero_indices:
                    glove_converts.append([source_index, target_index])
                else:
                    inspire_converts.append([source_index, target_index])

            self.glove_converts_np[hand_key] = np.asarray(glove_converts, dtype=np.int32)
            if len(self.glove_converts_np[hand_key]) > 0:
                src = self.glove_converts_np[hand_key][:, 0]
                dst = self.glove_converts_np[hand_key][:, 1]
                glove_uv891[dst] = glove_uv891[src]

            self.inspire_converts_np[hand_key] = np.asarray(inspire_converts, dtype=np.int32)
            if len(self.inspire_converts_np[hand_key]) > 0:
                src = self.inspire_converts_np[hand_key][:, 0]
                dst = self.inspire_converts_np[hand_key][:, 1]
                inspire_uv891[dst] = inspire_uv891[src]

            glove_zero = glove_uv891 == 0
            inspire_zero = inspire_uv891 == 0
            self.zero_indices[hand_key] = np.where(glove_zero & inspire_zero)[0].tolist()
            self.non_zero_indices[hand_key] = np.where(~glove_zero & ~inspire_zero)[0].tolist()
