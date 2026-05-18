from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

from inspire_data_collect.hand.dexhand_q6_backend import DexHandQ6Backend


CONTROL_DIR = Path(__file__).resolve().parent / "hardware" / "portable_dexhand021" / "6DOF_control"
if str(CONTROL_DIR) not in sys.path:
    sys.path.insert(0, str(CONTROL_DIR))

from q6_to_dexhand import DexHandQ6Mapper, reorder_q6  # noqa: E402


class DexHandQ6ThumbOrderTests(unittest.TestCase):
    def test_collect_backend_keeps_thumb_dimensions_in_dexhand_order(self) -> None:
        q6 = np.asarray([0.2, 0.7, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)

        mapped = DexHandQ6Backend._teleop_q6_to_internal_q6(q6)

        np.testing.assert_allclose(mapped, q6)

    def test_standalone_mapper_keeps_teleop_thumb_dimensions_in_dexhand_order(self) -> None:
        q6 = (0.2, 0.7, 0.3, 0.4, 0.5, 0.6)

        self.assertEqual(reorder_q6(q6, source_order="teleop"), q6)

    def test_first_thumb_dim_controls_flex_and_second_controls_rotation(self) -> None:
        mapper = DexHandQ6Mapper()

        action = mapper.q6_to_dexhand_action(
            [0.2, 0.7, 0.0, 0.0, 0.0, 0.0],
            source_order="teleop",
        )

        self.assertAlmostEqual(action["th_dip"], 18.0)
        self.assertAlmostEqual(action["th_mcp"], 18.0)
        self.assertAlmostEqual(action["th_rot"], 105.0)


if __name__ == "__main__":
    unittest.main()
