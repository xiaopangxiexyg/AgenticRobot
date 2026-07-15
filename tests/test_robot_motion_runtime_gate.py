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

    def test_status_text_identifies_recovering_sides(self) -> None:
        gate = IKActionChunkGate()
        gate.update(
            {
                "left": {"ik_metadata": {"status": "pass"}},
                "right": {"ik_metadata": {"status": "smoothing_required"}},
            }
        )
        self.assertEqual(
            gate.status_text(),
            "ik_recovery=active[right:smoothing_required](discard_policy_chunks)",
        )

    def test_dual_rate_worker_busy_hold_is_transient(self) -> None:
        gate = IKActionChunkGate()
        self.assertEqual(
            gate.update(
                {
                    "right": {
                        "ik_metadata": {
                            "status": "hold",
                            "hold_reason": "full_ik_worker_in_flight",
                        }
                    }
                }
            ),
            "",
        )
        self.assertFalse(gate.blocks_action_output)

    def test_dual_rate_local_no_safe_candidate_hold_is_backend_owned(self) -> None:
        gate = IKActionChunkGate()
        self.assertEqual(
            gate.update(
                {
                    "right": {
                        "ik_metadata": {
                            "status": "hold",
                            "source": "deployment_dual_rate_ik",
                            "method": "deployment_dual_rate_hold_last_safe",
                            "hold_reason": "local_ik_no_safe_candidate",
                        }
                    }
                }
            ),
            "",
        )
        self.assertFalse(gate.blocks_action_output)

    def test_status_text_includes_blocking_hold_reason(self) -> None:
        gate = IKActionChunkGate()
        gate.update(
            {
                "right": {
                    "ik_metadata": {
                        "status": "hold",
                        "source": "legacy_ik",
                        "hold_reason": "legacy_failure",
                    }
                }
            }
        )
        self.assertEqual(
            gate.status_text(),
            "ik_recovery=active[right:hold(legacy_failure)](discard_policy_chunks)",
        )


if __name__ == "__main__":
    unittest.main()
