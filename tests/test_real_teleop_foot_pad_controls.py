from __future__ import annotations

import unittest

from teleop_runtime.real_teleop.foot_pad_controls import (
    FOOT_PAD_EVENT_BY_KEY_CODE,
    SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
    SPECIAL_ARM_CONTROL_MODE_MANUAL,
    resolve_special_arm_control_mode,
)
from teleop_runtime.real_teleop.gui_controls import (
    GuiControlEvents,
    GuiControlState,
    current_active_controls,
)


class FootPadControlModeTests(unittest.TestCase):
    def test_both_special_arm_features_default_to_foot_pad(self) -> None:
        self.assertEqual(
            resolve_special_arm_control_mode(
                None,
                fix_arm_enabled=True,
                init_arm_enabled=True,
            ),
            SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
        )

    def test_manual_remains_default_without_both_special_arm_features(self) -> None:
        self.assertEqual(
            resolve_special_arm_control_mode(
                None,
                fix_arm_enabled=True,
                init_arm_enabled=False,
            ),
            SPECIAL_ARM_CONTROL_MODE_MANUAL,
        )

    def test_foot_pad_requires_both_special_arm_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires both"):
            resolve_special_arm_control_mode(
                SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
                fix_arm_enabled=True,
                init_arm_enabled=False,
            )

    def test_foot_pad_mapping_matches_pedal_order(self) -> None:
        self.assertEqual(
            FOOT_PAD_EVENT_BY_KEY_CODE,
            {
                183: "fix_arm_left",
                184: "fix_arm_right",
                185: "init_arm_left",
                186: "init_arm_right",
            },
        )

    def test_foot_pad_mode_hides_manual_special_arm_events(self) -> None:
        controls = current_active_controls(
            GuiControlState(
                teleop_started=True,
                fix_arm_available=True,
                init_arm_available=True,
                special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
            ),
            GuiControlEvents(),
        )
        self.assertNotIn("c", controls)
        self.assertNotIn("g", controls)
        self.assertNotIn("fix_arm_toggle", controls)
        self.assertNotIn("init_arm_toggle", controls)
        self.assertNotIn("fix_arm_select", controls)
        self.assertNotIn("init_arm_select", controls)


if __name__ == "__main__":
    unittest.main()
