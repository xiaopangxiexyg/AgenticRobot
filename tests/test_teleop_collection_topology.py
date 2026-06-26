import unittest

from teleop_runtime.collection.topology import (
    CollectionTopology,
    default_side_topology,
    normalize_active_sides,
    parse_active_sides,
)


class TeleopCollectionTopologyTest(unittest.TestCase):
    def test_parse_active_sides_preserves_existing_cli_values(self) -> None:
        self.assertEqual(parse_active_sides("left"), ("left",))
        self.assertEqual(parse_active_sides("right"), ("right",))
        self.assertEqual(parse_active_sides("both"), ("left", "right"))

    def test_parse_active_sides_accepts_descriptive_aliases(self) -> None:
        self.assertEqual(parse_active_sides("dual"), ("left", "right"))
        self.assertEqual(parse_active_sides("bimanual"), ("left", "right"))

    def test_normalize_active_sides_deduplicates_and_orders(self) -> None:
        self.assertEqual(normalize_active_sides(("right", "left", "right")), ("left", "right"))

    def test_default_side_topology_names_existing_channels_and_effectors(self) -> None:
        left = default_side_topology("left")

        self.assertEqual(left.arm_effector, "left_arm")
        self.assertEqual(left.hand_effector, "left_hand")
        self.assertEqual(left.wrist_pose_channel, "left_wrist_pose")
        self.assertEqual(left.hand_q6_channel, "left_hand_q6")
        self.assertEqual(left.clutch_button_channel, "left_clutch_button")
        self.assertEqual(left.wrist_tracker_role, "left_wrist")

    def test_collection_topology_exposes_side_lookups(self) -> None:
        topology = CollectionTopology.for_active_sides(("left", "right"))

        self.assertEqual(topology.active_sides, ("left", "right"))
        self.assertEqual(topology.arm_effector("right"), "right_arm")
        self.assertEqual(topology.hand_effector("right"), "right_hand")
        self.assertEqual(topology.wrist_pose_channel("right"), "right_wrist_pose")
        self.assertEqual(topology.hand_q6_channel("right"), "right_hand_q6")
        self.assertEqual(topology.arm_effectors(), ("left_arm", "right_arm"))
        self.assertEqual(topology.hand_effectors(), ("left_hand", "right_hand"))

    def test_collection_topology_rejects_inactive_side_lookup(self) -> None:
        topology = CollectionTopology.for_active_sides(("right",))

        with self.assertRaises(KeyError):
            topology.arm_effector("left")


if __name__ == "__main__":
    unittest.main()
