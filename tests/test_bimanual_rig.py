from __future__ import annotations

import threading
import unittest

from robot_motion.runtime import (
    EffectorCommand,
    TeleopCommand,
    readiness_report_blockers,
)
from teleop_runtime.rigs.bimanual import BimanualRig


class _FakeBackend:
    def __init__(
        self,
        name: str,
        *,
        fail_hold: bool = False,
        fail_send: bool = False,
    ) -> None:
        self._name = name
        self.fail_hold = fail_hold
        self.fail_send = fail_send
        self.holds: list[str] = []
        self.send_count = 0
        self.commands = []
        self.start_count = 0
        self.close_count = 0

    @property
    def name(self) -> str:
        return self._name

    def send_command(self, command) -> None:
        self.commands.append(command)
        self.send_count += 1
        if self.fail_send:
            raise RuntimeError(f"{self.name} send failed")

    def start(self) -> None:
        self.start_count += 1

    def is_ready(self) -> bool:
        return True

    def close(self) -> None:
        self.close_count += 1

    def hold_motion(self, reason: str = "manual hold") -> None:
        self.holds.append(reason)
        if self.fail_hold:
            raise RuntimeError(f"{self.name} hold failed")


class BimanualRigHoldMotionTest(unittest.TestCase):
    def test_readiness_report_identifies_one_unready_effector(self) -> None:
        class FaultBackend(_FakeBackend):
            def is_ready(self) -> bool:
                return False

            def readiness_report(self):
                return {
                    "ready": False,
                    "backend": "FaultBackend",
                    "reason": "error_code=16 status_code=1",
                    "details": {
                        "gripper_port": "/dev/ttyUSB1",
                        "gripper_id": 1,
                    },
                }

        rig = BimanualRig(
            {
                "left_arm": _FakeBackend("left"),
                "right_hand": FaultBackend("right_hand"),
            }
        )

        report = rig.readiness_report()

        self.assertFalse(report["ready"])
        self.assertEqual(report["not_ready_effectors"], ("right_hand",))
        self.assertEqual(
            readiness_report_blockers(report),
            {"right_hand": "error_code=16 status_code=1"},
        )

    def test_dispatch_binds_one_atomic_parent_timestamp_to_both_effectors(self) -> None:
        left = _FakeBackend("left")
        right = _FakeBackend("right")
        rig = BimanualRig({"left_arm": left, "right_arm": right})
        command = TeleopCommand(timestamp=123.5)
        command.add_command(
            EffectorCommand("left_arm", "left", "joint_vector", [0.0] * 6)
        )
        command.add_command(
            EffectorCommand("right_arm", "right", "joint_vector", [0.0] * 6)
        )

        rig.send_commands(command)

        self.assertEqual(
            left.commands[0].metadata["_parent_command_timestamp"],
            123.5,
        )
        self.assertEqual(
            right.commands[0].metadata["_parent_command_timestamp"],
            123.5,
        )

    def test_hold_motion_attempts_all_backends(self) -> None:
        left = _FakeBackend("left", fail_hold=True)
        right = _FakeBackend("right")
        rig = BimanualRig({"left_arm": left, "right_arm": right})

        with self.assertRaisesRegex(RuntimeError, "left_arm"):
            rig.hold_motion("unit test hold")

        self.assertEqual(left.holds, ["unit test hold"])
        self.assertEqual(right.holds, ["unit test hold"])

    def test_hold_motion_reports_all_failures(self) -> None:
        left = _FakeBackend("left", fail_hold=True)
        right = _FakeBackend("right", fail_hold=True)
        rig = BimanualRig({"left_arm": left, "right_arm": right})

        with self.assertRaisesRegex(RuntimeError, "left_arm.*right_arm"):
            rig.hold_motion("unit test hold")

    def test_parallel_rig_submits_both_holds_before_waiting(self) -> None:
        left_entered = threading.Event()
        right_entered = threading.Event()

        class CoordinatedBackend(_FakeBackend):
            def __init__(
                self, name: str, own: threading.Event, peer: threading.Event
            ) -> None:
                super().__init__(name)
                self.own = own
                self.peer = peer

            def hold_motion(self, reason: str = "manual hold") -> None:
                self.holds.append(reason)
                self.own.set()
                if not self.peer.wait(timeout=1.0):
                    raise RuntimeError(f"{self.name} peer hold was not submitted")

        left = CoordinatedBackend("left", left_entered, right_entered)
        right = CoordinatedBackend("right", right_entered, left_entered)
        rig = BimanualRig(
            {"left_arm": left, "right_arm": right},
            parallel_dispatch=True,
        )
        try:
            rig.hold_motion("unit test hold")
        finally:
            rig.close()

        self.assertTrue(left_entered.is_set())
        self.assertTrue(right_entered.is_set())

    def test_closed_rig_cannot_restart_terminal_backends(self) -> None:
        backend = _FakeBackend("left")
        rig = BimanualRig({"left_arm": backend})
        rig.start()
        rig.close()

        with self.assertRaisesRegex(RuntimeError, "construct a fresh rig"):
            rig.start()

        self.assertEqual(backend.start_count, 1)

    def test_capability_start_keeps_unready_effector_in_command_topology(self) -> None:
        class UnreadyBackend(_FakeBackend):
            def is_ready(self) -> bool:
                return False

            def readiness_report(self):
                return {
                    "ready": False,
                    "reason": "not connected",
                    "details": {},
                }

        right_hand = UnreadyBackend("right_hand")
        rig = BimanualRig(
            {
                "left_arm": _FakeBackend("left_arm"),
                "right_hand": right_hand,
            },
            allow_partial_start=True,
        )

        rig.start()
        report = rig.readiness_report()

        self.assertIn("right_hand", rig.backends)
        self.assertFalse(report["ready"])
        self.assertEqual(report["not_ready_effectors"], ("right_hand",))
        self.assertEqual(
            readiness_report_blockers(report),
            {"right_hand": "not connected"},
        )

    def test_position_group_dispatch_failure_holds_every_backend(self) -> None:
        left = _FakeBackend("left", fail_send=True)
        right = _FakeBackend("right")
        rig = BimanualRig(
            {"left_arm": left, "right_arm": right},
            parallel_dispatch=True,
            fail_stop_on_dispatch_error=True,
        )
        command = type(
            "Command", (), {"commands": {"left_arm": object(), "right_arm": object()}}
        )()

        with self.assertRaisesRegex(RuntimeError, "group fail-stop"):
            rig.send_commands(command)

        self.assertEqual(left.send_count, 1)
        self.assertEqual(right.send_count, 1)
        self.assertEqual(len(left.holds), 1)
        self.assertEqual(len(right.holds), 1)

    def test_default_dispatch_failure_keeps_legacy_non_fail_stop_behavior(self) -> None:
        left = _FakeBackend("left", fail_send=True)
        right = _FakeBackend("right")
        rig = BimanualRig({"left_arm": left, "right_arm": right})
        command = type(
            "Command", (), {"commands": {"left_arm": object(), "right_arm": object()}}
        )()

        with self.assertRaisesRegex(RuntimeError, "left send failed"):
            rig.send_commands(command)

        self.assertEqual(left.holds, [])
        self.assertEqual(right.holds, [])


if __name__ == "__main__":
    unittest.main()
