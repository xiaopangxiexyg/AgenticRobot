from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from teleop_runtime.real_teleop.foot_pad_controls import (
    FOOT_PAD_EVENT_BY_KEY_CODE,
    FOOT_PAD_EVENT_BY_KEY_CODE_BY_COUNT,
    FootPadSpecialArmInput,
    FootPadUnavailableError,
    SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
    SPECIAL_ARM_CONTROL_MODE_MANUAL,
    foot_pad_count_from_key_codes,
    resolve_special_arm_control_mode,
)
from teleop_runtime.gui.real_teleop_control import RealTeleopControlInput
from teleop_runtime.real_teleop.gui_controls import (
    GuiControlEvents,
    GuiControlState,
    current_active_controls,
)


class FootPadControlModeTests(unittest.TestCase):
    def test_gui_status_distinguishes_unavailable_two_and_four_pedals(self) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.args = SimpleNamespace(
            foot_pad_requested=True,
            foot_pad_startup_error="FootPadUnavailableError: virtual keyboard missing",
        )
        control._foot_pad_test_active = False
        control._foot_pad_test_completed = False
        control._foot_pad_test_seen_events = set()
        control._foot_pad_test_last_event = ""
        control._foot_pad_test_status = ""

        control.foot_pad_input = None
        unavailable = control._foot_pad_test_ui_state()
        self.assertTrue(unavailable["foot_pad_requested"])
        self.assertFalse(unavailable["foot_pad_available"])
        self.assertEqual(unavailable["foot_pad_status"], "unavailable")
        self.assertEqual(unavailable["foot_pad_status_label"], "未连接 / 未就绪")
        self.assertIn("virtual keyboard missing", unavailable["foot_pad_status_message"])
        control.foot_pad_input = SimpleNamespace(
            foot_pad_count=4,
            is_online=False,
        )
        waiting = control._foot_pad_test_ui_state()
        self.assertIn("先进入 Hold", waiting["foot_pad_status_message"])
        self.assertIn("重连失败设备", waiting["foot_pad_status_message"])

        for count in (2, 4):
            control.foot_pad_input = SimpleNamespace(
                foot_pad_count=count,
                is_online=True,
            )
            online = control._foot_pad_test_ui_state()
            self.assertTrue(online["foot_pad_available"])
            self.assertEqual(online["foot_pad_count"], count)
            self.assertEqual(online["foot_pad_status"], "online")
            self.assertEqual(online["foot_pad_status_label"], f"{count} 个已连接")

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

    def test_two_pedals_default_to_left_and_right_fix(self) -> None:
        self.assertEqual(
            resolve_special_arm_control_mode(
                None,
                fix_arm_enabled=True,
                init_arm_enabled=False,
                foot_pad_count=2,
            ),
            SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
        )

    def test_unsupported_pedal_count_falls_back_to_manual(self) -> None:
        self.assertEqual(
            resolve_special_arm_control_mode(
                None,
                fix_arm_enabled=True,
                init_arm_enabled=True,
                foot_pad_count=3,
            ),
            SPECIAL_ARM_CONTROL_MODE_MANUAL,
        )

    def test_two_pedal_explicit_mode_only_requires_fix_arm(self) -> None:
        self.assertEqual(
            resolve_special_arm_control_mode(
                SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
                fix_arm_enabled=True,
                init_arm_enabled=False,
                foot_pad_count=2,
            ),
            SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
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

    def test_two_pedal_mapping_only_emits_fix_events(self) -> None:
        self.assertEqual(
            FOOT_PAD_EVENT_BY_KEY_CODE_BY_COUNT[2],
            {183: "fix_arm_left", 184: "fix_arm_right"},
        )

        class FakeEvent:
            type = 1
            value = 1

        class FakeCodes:
            EV_KEY = 1

        class FakeEvdev:
            ecodes = FakeCodes()

        class Device:
            def read(self):
                left = FakeEvent()
                left.code = 183
                right = FakeEvent()
                right.code = 185
                return [left, right]

            def close(self):
                return None

        source = FootPadSpecialArmInput(Device(), FakeEvdev(), foot_pad_count=2)
        self.assertEqual(source.consume_events(), ["fix_arm_left"])
        self.assertEqual(
            source.consume_test_events(),
            ["fix_arm_left", "init_arm_left"],
        )
        source.close()

    def test_lazy_evdev_eagain_is_a_normal_empty_poll(self) -> None:
        class FakeCodes:
            EV_KEY = 1

        class FakeEvdev:
            ecodes = FakeCodes()

        class Device:
            def read(self):
                def events():
                    raise BlockingIOError(11, "Resource temporarily unavailable")
                    yield  # pragma: no cover

                return events()

            def close(self):
                return None

        source = FootPadSpecialArmInput(Device(), FakeEvdev(), foot_pad_count=2)
        self.assertEqual(source.consume_events(), [])
        self.assertTrue(source.is_online)
        source.close()

    def test_live_keyboard_capabilities_identify_two_pedal_topology(self) -> None:
        self.assertEqual(foot_pad_count_from_key_codes({183, 184}), 2)
        self.assertEqual(foot_pad_count_from_key_codes({183, 184, 185, 186}), 4)
        self.assertIsNone(foot_pad_count_from_key_codes({183, 185}))

    def test_open_online_accepts_two_pedal_keyboard_without_saved_config(self) -> None:
        class FakeCodes:
            EV_KEY = 1

        class FakeDevice:
            name = "DexTeleop foot pads"

            def capabilities(self, *, absinfo=False):
                del absinfo
                return {FakeCodes.EV_KEY: [183, 184]}

            def close(self):
                return None

        class FakeEvdev:
            ecodes = FakeCodes()

            @staticmethod
            def list_devices():
                return ["/dev/input/event-two-pads"]

            @staticmethod
            def InputDevice(_path):
                return FakeDevice()

        for saved_count in (None, 4):
            with self.subTest(saved_count=saved_count), mock.patch(
                "teleop_runtime.real_teleop.foot_pad_controls.configured_foot_pad_count",
                return_value=saved_count,
            ), mock.patch(
                "teleop_runtime.real_teleop.foot_pad_controls._require_evdev",
                return_value=FakeEvdev,
            ):
                source = FootPadSpecialArmInput.open_online()
            self.assertEqual(source.foot_pad_count, 2)
            source.close()

    def test_open_online_reports_inaccessible_named_virtual_keyboard(self) -> None:
        class FakeEvdev:
            @staticmethod
            def list_devices():
                return ["/dev/input/event31"]

            @staticmethod
            def InputDevice(path):
                raise PermissionError(13, "Permission denied", path)

        with mock.patch(
            "teleop_runtime.real_teleop.foot_pad_controls.configured_foot_pad_count",
            return_value=4,
        ), mock.patch(
            "teleop_runtime.real_teleop.foot_pad_controls._require_evdev",
            return_value=FakeEvdev,
        ), mock.patch(
            "teleop_runtime.real_teleop.foot_pad_controls._sysfs_input_device_name",
            return_value="DexTeleop foot pads",
        ):
            with self.assertRaisesRegex(
                FootPadUnavailableError,
                r"event31.*cannot read.*--fix-permissions",
            ):
                FootPadSpecialArmInput.open_online()

    def test_open_online_skips_non_foot_pad_nodes_before_evdev_open(self) -> None:
        opened_paths: list[str] = []

        class FakeEvdev:
            @staticmethod
            def list_devices():
                return ["/dev/input/event8", "/dev/input/event9"]

            @staticmethod
            def InputDevice(path):
                opened_paths.append(path)
                raise AssertionError("non-foot-pad event node must not be opened")

        with mock.patch(
            "teleop_runtime.real_teleop.foot_pad_controls.configured_foot_pad_count",
            return_value=4,
        ), mock.patch(
            "teleop_runtime.real_teleop.foot_pad_controls._require_evdev",
            return_value=FakeEvdev,
        ), mock.patch(
            "teleop_runtime.real_teleop.foot_pad_controls._sysfs_input_device_name",
            side_effect=["USB keyboard", "USB mouse"],
        ), mock.patch(
            "teleop_runtime.real_teleop.foot_pad_controls._input_permission_hint",
            return_value="",
        ):
            with self.assertRaises(FootPadUnavailableError):
                FootPadSpecialArmInput.open_online()

        self.assertEqual(opened_paths, [])

    def test_four_pedal_mode_keeps_all_gui_special_arm_events(self) -> None:
        controls = current_active_controls(
            GuiControlState(
                teleop_started=True,
                fix_arm_available=True,
                init_arm_available=True,
                special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
                foot_pad_available=True,
                foot_pad_count=4,
            ),
            GuiControlEvents(),
        )
        self.assertIn("c", controls)
        self.assertIn("g", controls)
        self.assertIn("fix_arm_toggle", controls)
        self.assertIn("init_arm_toggle", controls)
        self.assertIn("fix_arm_select", controls)
        self.assertIn("init_arm_select", controls)

    def test_two_pedal_mode_keeps_all_gui_special_arm_controls(self) -> None:
        controls = current_active_controls(
            GuiControlState(
                teleop_started=True,
                fix_arm_available=True,
                init_arm_available=True,
                special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
                foot_pad_available=True,
                foot_pad_count=2,
            ),
            GuiControlEvents(),
        )
        self.assertIn("c", controls)
        self.assertIn("fix_arm_toggle", controls)
        self.assertIn("fix_arm_select", controls)
        self.assertIn("g", controls)
        self.assertIn("init_arm_toggle", controls)
        self.assertIn("init_arm_select", controls)
        self.assertIn("init_arm_move_to_qpos", controls)
        self.assertIn("init_arm_move_both_to_qpos", controls)

    def test_four_pedal_mode_exposes_keyboard_and_explicit_gui_init_move(self) -> None:
        controls = current_active_controls(
            GuiControlState(
                teleop_started=True,
                fix_arm_available=True,
                init_arm_available=True,
                special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
                foot_pad_available=True,
                foot_pad_count=4,
            ),
            GuiControlEvents(),
        )

        self.assertIn("g", controls)
        self.assertIn("init_arm_toggle", controls)
        self.assertIn("init_arm_select", controls)
        self.assertIn("init_arm_move_to_qpos", controls)
        self.assertIn("init_arm_move_both_to_qpos", controls)

    def test_unavailable_foot_pad_falls_back_to_manual_arm_maintenance(self) -> None:
        controls = current_active_controls(
            GuiControlState(
                teleop_started=True,
                fix_arm_available=True,
                init_arm_available=True,
                init_arm_qpos_move_available=True,
                special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
                foot_pad_available=False,
                foot_pad_count=4,
            ),
            GuiControlEvents(),
        )

        self.assertIn("fix_arm_select", controls)
        self.assertIn("init_arm_select", controls)
        self.assertIn("init_arm_move_to_qpos", controls)
        self.assertIn("init_arm_move_both_to_qpos", controls)

    def test_unsupported_online_count_falls_back_to_manual_controls(self) -> None:
        controls = current_active_controls(
            GuiControlState(
                teleop_started=True,
                fix_arm_available=True,
                init_arm_available=True,
                special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_MANUAL,
                foot_pad_available=False,
                foot_pad_count=3,
            ),
            GuiControlEvents(),
        )
        self.assertIn("c", controls)
        self.assertIn("g", controls)

    def test_foot_pad_test_control_is_additive_while_active(self) -> None:
        ready = current_active_controls(
            GuiControlState(
                foot_pad_available=True,
                fix_arm_available=True,
                init_arm_available=True,
                startup_qpos_maintenance_enabled=True,
            ),
            GuiControlEvents(),
        )
        self.assertIn("foot_pad_test_start", ready)
        active = current_active_controls(
            GuiControlState(
                foot_pad_available=True,
                foot_pad_test_active=True,
                fix_arm_available=True,
                init_arm_available=True,
                startup_qpos_maintenance_enabled=True,
            ),
            GuiControlEvents(),
        )
        self.assertIn("foot_pad_test_stop", active)
        self.assertIn("fix_arm_select", active)
        self.assertIn("init_arm_select", active)
        self.assertIn("init_arm_move_to_qpos", active)
        self.assertIn("device_retry", active)
        self.assertEqual(
            set(active) - {"foot_pad_test_stop"},
            set(ready) - {"foot_pad_test_start"},
        )

    def test_disconnected_virtual_keyboard_does_not_scan_during_poll(self) -> None:
        class FakeCodes:
            EV_KEY = 1

        class FakeEvdev:
            ecodes = FakeCodes()

        class DisconnectedDevice:
            def read(self):
                raise OSError("disconnected")

            def close(self):
                return None

        source = FootPadSpecialArmInput(DisconnectedDevice(), FakeEvdev())
        self.assertTrue(source.is_online)
        self.assertEqual(source.consume_events(), [])
        self.assertFalse(source.is_online)
        with mock.patch.object(
            FootPadSpecialArmInput,
            "open_online",
        ) as open_online:
            self.assertEqual(source.consume_events(), [])
            self.assertEqual(source.consume_events(), [])
        self.assertFalse(source.is_online)
        open_online.assert_not_called()
        source.close()

    def test_startup_waiting_adapter_scans_only_after_manual_retry(self) -> None:
        source = FootPadSpecialArmInput.waiting_for_service(foot_pad_count=4)
        self.assertFalse(source.is_online)
        with mock.patch.object(
            FootPadSpecialArmInput,
            "open_online",
        ) as open_online:
            self.assertEqual(source.consume_events(), [])
            self.assertEqual(source.consume_events(), [])

        self.assertFalse(source.is_online)
        self.assertEqual(source.foot_pad_count, 4)
        open_online.assert_not_called()
        source.close()

    def test_gui_test_mode_records_mapping_without_routing_an_arm_action(self) -> None:
        class FakeUi:
            def __init__(self) -> None:
                self.published = []
                self.messages = []

            def publish(self, **updates):
                self.published.append(updates)

            def append_message(self, tag, message):
                self.messages.append((tag, message))

        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.ui = FakeUi()
        control.foot_pad_input = object()
        control._foot_pad_test_active = False
        control._foot_pad_test_completed = False
        control._foot_pad_test_seen_events = set()
        control._foot_pad_test_last_event = ""
        control._foot_pad_test_status = ""
        control._teleop_started = False
        control._hold_active = False
        control._current_active_controls = lambda: ["q", "foot_pad_test_stop"]
        control._control_hint = lambda: "normal"
        control._episode_busy = lambda: False

        control._start_foot_pad_test()
        self.assertTrue(control._foot_pad_test_active)
        control._record_foot_pad_test_event("init_arm_right", now=1.0)
        self.assertEqual(
            control._foot_pad_test_last_event,
            "踏板 4 / F16 → 右臂精确 Init qpos",
        )
        self.assertIn("未执行实际操作", control.ui.messages[-1][1])
        control._stop_foot_pad_test()
        self.assertFalse(control._foot_pad_test_active)
        self.assertTrue(control._foot_pad_test_completed)
        self.assertIn("本次检测到 1/4", control._foot_pad_test_status)

    def test_gui_test_mode_two_pedals_auto_completes_after_both_pedals(self) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.ui = SimpleNamespace(
            published=[],
            messages=[],
            publish=lambda **updates: control.ui.published.append(updates),
            append_message=lambda tag, message: control.ui.messages.append(
                (tag, message)
            ),
        )
        control.foot_pad_input = SimpleNamespace(
            foot_pad_count=2,
            is_online=True,
        )
        control._foot_pad_test_active = True
        control._foot_pad_test_completed = False
        control._foot_pad_test_seen_events = set()
        control._foot_pad_test_last_event = ""
        control._foot_pad_test_status = ""
        control._current_active_controls = lambda: ["q", "two_pedal_b", "two_pedal_c"]
        control._control_hint = lambda: "normal"
        control.args = SimpleNamespace(foot_pad_requested=True)

        for index, event in enumerate(("fix_arm_left", "fix_arm_right"), start=1):
            control._record_foot_pad_test_event(event, now=float(index))

        self.assertFalse(control._foot_pad_test_active)
        self.assertTrue(control._foot_pad_test_completed)
        self.assertEqual(len(control._foot_pad_test_seen_events), 2)
        self.assertIn("自动完成", control._foot_pad_test_status)
        self.assertTrue(control.ui.published[-1]["foot_pad_test_completed"])
        self.assertEqual(control.ui.published[-1]["foot_pad_test_seen_count"], 2)

    def test_gui_test_mode_two_pedal_ignores_nonexistent_init_keys(self) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.ui = SimpleNamespace(
            published=[],
            messages=[],
            publish=lambda **updates: control.ui.published.append(updates),
            append_message=lambda tag, message: control.ui.messages.append(
                (tag, message)
            ),
        )
        control.foot_pad_input = SimpleNamespace(foot_pad_count=2, is_online=True)
        control._foot_pad_test_active = True
        control._foot_pad_test_completed = False
        control._foot_pad_test_seen_events = set()
        control._foot_pad_test_last_event = ""
        control._foot_pad_test_status = ""
        control._current_active_controls = lambda: ["q"]
        control._control_hint = lambda: "normal"
        control.args = SimpleNamespace(foot_pad_requested=True)

        control._record_foot_pad_test_event("init_arm_left", now=1.0)
        self.assertTrue(control._foot_pad_test_active)
        self.assertEqual(control._foot_pad_test_seen_events, set())

        control._record_foot_pad_test_event("fix_arm_left", now=2.0)
        self.assertTrue(control._foot_pad_test_active)
        control._record_foot_pad_test_event("fix_arm_right", now=3.0)
        self.assertFalse(control._foot_pad_test_active)
        self.assertTrue(control._foot_pad_test_completed)
        self.assertEqual(len(control._foot_pad_test_seen_events), 2)

    def test_gui_test_mode_four_pedals_keeps_four_key_completion(self) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.ui = SimpleNamespace(
            published=[],
            messages=[],
            publish=lambda **updates: control.ui.published.append(updates),
            append_message=lambda tag, message: control.ui.messages.append(
                (tag, message)
            ),
        )
        control.foot_pad_input = SimpleNamespace(foot_pad_count=4, is_online=True)
        control._foot_pad_test_active = True
        control._foot_pad_test_completed = False
        control._foot_pad_test_seen_events = set()
        control._foot_pad_test_last_event = ""
        control._foot_pad_test_status = ""
        control._current_active_controls = lambda: ["q"]
        control._control_hint = lambda: "normal"
        control.args = SimpleNamespace(foot_pad_requested=True)

        for index, event in enumerate(
            ("fix_arm_left", "fix_arm_right", "init_arm_left", "init_arm_right"),
            start=1,
        ):
            control._record_foot_pad_test_event(event, now=float(index))
            if index < 4:
                self.assertTrue(control._foot_pad_test_active)
        self.assertFalse(control._foot_pad_test_active)
        self.assertTrue(control._foot_pad_test_completed)
        self.assertEqual(len(control._foot_pad_test_seen_events), 4)
        self.assertIn("自动完成", control._foot_pad_test_status)

    def test_fourth_pedal_routes_to_exact_qpos_not_legacy_init_pose(self) -> None:
        class InitController:
            active = False
            active_side = ""

            def __init__(self) -> None:
                self.selected: list[str] = []

            def select_side(self, side: str):
                self.selected.append(side)
                return SimpleNamespace(ok=True, message="selected")

        controller = InitController()
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.components = SimpleNamespace(init_arm_controller=controller)
        control._foot_pad_mode_enabled = lambda: True
        control._foot_pad_count = lambda: 4
        control._fix_init_allowed_in_controls = lambda: True
        exact_calls: list[float] = []
        control._handle_init_arm_move_to_qpos_event = (
            lambda *, now: exact_calls.append(float(now))
        )
        control._handle_init_arm_g_event = lambda **_kwargs: self.fail(
            "four-pedal Init must not enter the legacy pose/Home path"
        )

        control._handle_foot_pad_special_arm_event("init_arm_right", now=12.5)

        self.assertEqual(controller.selected, ["right"])
        self.assertEqual(exact_calls, [12.5])

    def test_two_pedal_routes_mean_moving_side_and_never_init(self) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control._foot_pad_mode_enabled = lambda: True
        control._foot_pad_count = lambda: 2
        control._two_pedal_mode_active = lambda: True
        requests = []
        control._request_two_pedal_fixed_side = (
            lambda side, *, now, source: requests.append((side, now, source))
        )

        control._handle_foot_pad_special_arm_event("fix_arm_left", now=10.0)
        control._handle_foot_pad_special_arm_event("fix_arm_right", now=11.0)
        control._handle_foot_pad_special_arm_event("init_arm_left", now=12.0)

        self.assertEqual(
            requests,
            [
                ("right", 10.0, "pedal_1"),
                ("left", 11.0, "pedal_2"),
            ],
        )

    def test_two_pedal_alerts_replace_each_other_and_red_expires(self) -> None:
        published = []
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.ui = SimpleNamespace(
            publish=lambda **updates: published.append(updates)
        )
        control._control_hint = lambda: "normal"

        control._publish_two_pedal_alert(
            level="red",
            message="failed",
            now=10.0,
            duration_sec=3.0,
        )
        control._publish_two_pedal_alert(
            level="orange",
            message="waiting",
            now=11.0,
        )
        self.assertEqual(published[-1]["two_pedal_alert"]["level"], "orange")
        self.assertEqual(control._two_pedal_alert_expires_at, 0.0)

        control._publish_two_pedal_alert(
            level="red",
            message="failed again",
            now=12.0,
            duration_sec=3.0,
        )
        control._poll_two_pedal_alert_timeout(now=14.9)
        self.assertTrue(published[-1]["two_pedal_alert"]["active"])
        control._poll_two_pedal_alert_timeout(now=15.0)
        self.assertEqual(published[-1]["two_pedal_alert"], {})
        self.assertEqual(published[-1]["control_hint"], "normal")

    def test_matching_init_pedal_during_qpos_pauses_and_starts_home_recapture(self) -> None:
        class InitController:
            active = True
            active_side = "right"
            recapturing = False

            def __init__(self) -> None:
                self.calls = []

            def start_recalibration(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(ok=True, message="paused", side="right")

        controller = InitController()
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.args = SimpleNamespace(calibration_timeout=7.0, calibration_samples=12)
        control.components = SimpleNamespace(init_arm_controller=controller)
        control._foot_pad_mode_enabled = lambda: True
        control._foot_pad_count = lambda: 4
        control._fix_init_allowed_in_controls = lambda: True
        messages = []
        control._publish_init_arm_state = lambda **kwargs: messages.append(kwargs)

        control._handle_foot_pad_special_arm_event("init_arm_right", now=12.5)

        self.assertEqual(
            controller.calls,
            [{"now": 12.5, "timeout_sec": 7.0, "min_samples": 12}],
        )
        self.assertIn("Init 数采标签已结束", messages[-1]["message"])


if __name__ == "__main__":
    unittest.main()
