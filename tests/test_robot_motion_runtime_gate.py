from __future__ import annotations

import unittest

from robot_motion.runtime_gate import IKActionChunkGate


class IKActionChunkGateTest(unittest.TestCase):
    def test_discards_until_fresh_post_recovery_policy_output(self) -> None:
        gate = IKActionChunkGate(stable_frames_required=2)

        self.assertEqual(gate.update({"right": {"ik_metadata": {"status": "pass"}}}), "")
        self.assertFalse(gate.blocks_action_output)

        self.assertEqual(
            gate.update({"right": {"ik_metadata": {"status": "smoothing_required"}}}),
            "ik_recovery_started",
        )
        self.assertTrue(gate.discards_policy_output)
        self.assertTrue(gate.blocks_action_output)

        self.assertEqual(gate.update({"right": {"ik_metadata": {"status": "pass"}}}), "")
        self.assertTrue(gate.blocks_action_output)

        self.assertEqual(
            gate.update({"right": {"ik_metadata": {"status": "pass"}}}),
            "ik_recovery_stable_waiting_fresh_policy",
        )
        self.assertTrue(gate.blocks_action_output)
        self.assertFalse(gate.discards_policy_output)

        gate.mark_policy_output_accepted()
        self.assertFalse(gate.blocks_action_output)

    def test_legacy_jump_gate_hold_metadata_triggers_recovery_gate(self) -> None:
        gate = IKActionChunkGate(stable_frames_required=1)

        self.assertEqual(
            gate.update({"left": {"ik_metadata": {"ik_jump_gate_status": "holding"}}}),
            "ik_recovery_started",
        )
        self.assertTrue(gate.blocks_action_output)

    def test_legacy_singularity_hold_metadata_triggers_recovery_gate(self) -> None:
        gate = IKActionChunkGate(stable_frames_required=1)

        self.assertEqual(
            gate.update({"left": {"ik_metadata": {"singularity_gate_status": "hold_entry"}}}),
            "ik_recovery_started",
        )
        self.assertTrue(gate.blocks_action_output)


if __name__ == "__main__":
    unittest.main()
