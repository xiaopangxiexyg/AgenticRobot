import argparse
import unittest

from teleop_runtime.collection.modes import resolve_collection_mode


def _args(
    *,
    left_arm_fixed: bool = False,
    left_arm_jitter: bool = False,
    disable_dual_arm_collision_avoidance: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        left_arm_fixed=left_arm_fixed,
        left_arm_jitter=left_arm_jitter,
        disable_dual_arm_collision_avoidance=disable_dual_arm_collision_avoidance,
    )


class TeleopCollectionModesTest(unittest.TestCase):
    def test_right_arm_right_hand_uses_package_entrypoint(self) -> None:
        spec = resolve_collection_mode(_args(), ("right",))

        self.assertEqual(spec.name, "right_arm_collect")
        self.assertEqual(spec.fixed_arm_sides, ())
        self.assertEqual(spec.jitter_arm_sides, ())
        self.assertEqual(spec.moving_arm_sides, ("right",))
        self.assertEqual(spec.source_entrypoint, "teleop_runtime.collection")
        self.assertFalse(spec.experimental)

    def test_dual_arm_dual_hand_uses_bimanual_mode(self) -> None:
        spec = resolve_collection_mode(_args(), ("left", "right"))

        self.assertEqual(spec.name, "bimanual_collect")
        self.assertEqual(spec.moving_arm_sides, ("left", "right"))
        self.assertEqual(spec.source_entrypoint, "teleop_runtime.collection")
        self.assertFalse(spec.experimental)

    def test_left_arm_fixed_marks_left_non_operator_side(self) -> None:
        spec = resolve_collection_mode(
            _args(left_arm_fixed=True),
            ("left", "right"),
        )

        self.assertEqual(spec.name, "left_arm_fixed")
        self.assertEqual(spec.fixed_arm_sides, ("left",))
        self.assertEqual(spec.jitter_arm_sides, ())
        self.assertEqual(spec.moving_arm_sides, ("right",))
        self.assertEqual(spec.source_entrypoint, "teleop_runtime.collection")
        self.assertTrue(spec.experimental)

    def test_left_arm_jitter_requires_dual_arm_ik(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires dual-arm IK"):
            resolve_collection_mode(
                _args(left_arm_jitter=True, disable_dual_arm_collision_avoidance=True),
                ("left", "right"),
            )

    def test_left_arm_jitter_uses_experimental_entrypoint(self) -> None:
        spec = resolve_collection_mode(
            _args(left_arm_jitter=True),
            ("left", "right"),
        )

        self.assertEqual(spec.name, "left_arm_jitter")
        self.assertEqual(spec.fixed_arm_sides, ())
        self.assertEqual(spec.jitter_arm_sides, ("left",))
        self.assertEqual(spec.moving_arm_sides, ("right",))
        self.assertEqual(spec.source_entrypoint, "teleop_runtime.collection")
        self.assertTrue(spec.experimental)

    def test_left_arm_experimental_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            resolve_collection_mode(
                _args(left_arm_fixed=True, left_arm_jitter=True),
                ("left", "right"),
            )


if __name__ == "__main__":
    unittest.main()
