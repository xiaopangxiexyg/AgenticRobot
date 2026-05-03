from __future__ import annotations

import argparse
import unittest

from teleop_runtime.deploy.run_policy import _resolve_action_queue_policy


class TeleopDeployQueuePolicyTest(unittest.TestCase):
    def test_auto_keeps_replace_for_normal_policy(self) -> None:
        args = argparse.Namespace(action_queue_policy="auto", append_action_chunks=False)
        self.assertEqual(_resolve_action_queue_policy(args, debug_replay=False), "replace")

    def test_auto_uses_append_for_debug_replay(self) -> None:
        args = argparse.Namespace(action_queue_policy="auto", append_action_chunks=False)
        self.assertEqual(_resolve_action_queue_policy(args, debug_replay=True), "append")

    def test_legacy_append_flag_takes_precedence(self) -> None:
        args = argparse.Namespace(action_queue_policy="replace", append_action_chunks=True)
        self.assertEqual(_resolve_action_queue_policy(args, debug_replay=False), "append")


if __name__ == "__main__":
    unittest.main()
