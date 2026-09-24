from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from teleop_runtime.real_teleop.foot_pad_controls import (
    FOOT_PAD_EVENT_BY_KEY_CODE,
    FOOT_PAD_EVENT_BY_KEY_CODE_BY_COUNT,
    FOOT_PAD_FIVE_EVENT_BY_KEY_CODE,
    FOOT_PAD_TWO_PEDAL_STYLE_COUNTS,
    FootPadSpecialArmInput,
    FootPadUnavailableError,
    SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
    SPECIAL_ARM_CONTROL_MODE_MANUAL,
    apply_live_foot_pad_topology,
    foot_pad_count_from_key_codes,
    foot_pad_progress_topology_guidance,
    initialize_startup_foot_pad_input,
    persist_five_pedal_test_receipt,
    persisted_five_pedal_test_verified,
    resolve_special_arm_control_mode,
)
from teleop_runtime.gui.real_teleop_control import RealTeleopControlInput
from control_station.real_teleop.control_availability import (
    GuiControlEvents,
    GuiControlState,
    current_active_controls,
)
from control_station.real_teleop_application import (
    build_real_teleop_progress_application,
)
from tests.support.real_teleop_control import bind_control_application


class FootPadControlModeTests(unittest.TestCase):
    @staticmethod
    def _attach_progress_application(control: RealTeleopControlInput) -> None:
        control._progress_application = build_real_teleop_progress_application(
            control.args
        )

    @staticmethod
    def _five_pedal_assignment() -> dict[str, object]:
        return {
            "schema_version": "foot_pad_assignment.v1",
            "created_at": "2026-08-02T00:00:00+08:00",
            "count": 5,
            "pedals": [
                {
                    "slot": slot,
                    "source": {
                        "id_path": f"pci-usb-pedal-{slot}",
                        "vendor_id": "0483",
                        "model_id": "5750",
                        "name": "HID 0483:5750",
                        "key_code": 98,
                        "key_name": "KEY_KPSLASH",
                    },
                    "output": {
                        "key_name": f"F{12 + slot}",
                        "key_code": 182 + slot,
                    },
                }
                for slot in range(1, 6)
            ],
        }

    def test_five_pedal_test_receipt_is_shared_until_mapping_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            config_path = Path(temporary_dir) / "latest.json"
            config = self._five_pedal_assignment()
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with mock.patch(
                "teleop_runtime.real_teleop.foot_pad_controls._load_foot_pad_config",
                return_value=(config_path, 5),
            ):
                receipt_path = persist_five_pedal_test_receipt()
                self.assertTrue(receipt_path.is_file())
                self.assertTrue(persisted_five_pedal_test_verified())

                config["created_at"] = "2026-08-02T01:00:00+08:00"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                self.assertTrue(persisted_five_pedal_test_verified())

                config["pedals"][0]["source"]["id_path"] = "remapped-pedal"
                config_path.write_text(json.dumps(config), encoding="utf-8")
                self.assertFalse(persisted_five_pedal_test_verified())

    def test_five_pedal_control_restores_persisted_test_across_modes(self) -> None:
        for arm_control_mode in ("position", "impedance"):
            with self.subTest(arm_control_mode=arm_control_mode), mock.patch(
                "teleop_runtime.real_teleop.operator_control_runtime."
                "persisted_five_pedal_test_verified",
                return_value=True,
            ):
                control = RealTeleopControlInput.__new__(RealTeleopControlInput)
                control.args = SimpleNamespace(arm_control_mode=arm_control_mode)
                bind_control_application(control)
                control.foot_pad_input = SimpleNamespace(
                    foot_pad_count=5,
                    is_online=True,
                )
                control._foot_pad_test_completed = False
                control._foot_pad_test_seen_events = {"stale"}
                control._foot_pad_test_last_event = "stale"
                control._foot_pad_test_status = ""

                self.assertTrue(control._restore_persisted_five_pedal_test_state())
                self.assertTrue(control._foot_pad_test_completed)
                self.assertEqual(control._foot_pad_test_seen_events, set())
                self.assertIn("力控/位控共用", control._foot_pad_test_status)

    def test_gui_status_distinguishes_supported_pedal_topologies(self) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.args = SimpleNamespace(
            foot_pad_requested=True,
            foot_pad_startup_error="FootPadUnavailableError: virtual keyboard missing",
        )
        bind_control_application(control)
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

        for count in (2, 4, 5):
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

    def test_rule_task_chain_mode_admits_only_two_pedals_without_fix_init_gate(
        self,
    ) -> None:
        self.assertEqual(
            resolve_special_arm_control_mode(
                SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
                fix_arm_enabled=False,
                init_arm_enabled=False,
                foot_pad_count=2,
                rule_task_chain_foot_pad_enabled=True,
            ),
            SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
        )
        with self.assertRaisesRegex(ValueError, "exactly 2 pedals"):
            resolve_special_arm_control_mode(
                SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
                fix_arm_enabled=False,
                init_arm_enabled=False,
                foot_pad_count=4,
                rule_task_chain_foot_pad_enabled=True,
            )

    def test_five_pedal_topology_is_admitted_before_progress_profile_selection(self) -> None:
        self.assertEqual(
            resolve_special_arm_control_mode(
                None,
                fix_arm_enabled=True,
                init_arm_enabled=False,
                foot_pad_count=5,
                progress_enabled=True,
            ),
            SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
        )
        self.assertEqual(
            resolve_special_arm_control_mode(
                None,
                fix_arm_enabled=True,
                init_arm_enabled=True,
                foot_pad_count=5,
                progress_enabled=False,
            ),
            SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
        )

    def test_live_five_pedal_topology_replaces_stale_startup_count(self) -> None:
        args = SimpleNamespace(
            special_arm_control_mode_requested=None,
            special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
            fix_arm=True,
            init_arm_available=True,
            collection_progress_profile=None,
            collection_progress_gui=True,
            foot_pad_config_count=2,
        )

        mode = apply_live_foot_pad_topology(args, 5)

        self.assertEqual(mode, SPECIAL_ARM_CONTROL_MODE_FOOT_PAD)
        self.assertEqual(args.foot_pad_config_count, 5)
        self.assertEqual(
            args.special_arm_control_mode,
            SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
        )

    def test_live_five_pedal_topology_without_progress_keeps_fix_prefix(self) -> None:
        args = SimpleNamespace(
            special_arm_control_mode_requested=None,
            special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
            fix_arm=True,
            init_arm_available=True,
            collection_progress_profile=None,
            collection_progress_gui=False,
            foot_pad_config_count=2,
        )

        self.assertEqual(
            apply_live_foot_pad_topology(args, 5),
            SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
        )
        self.assertEqual(args.foot_pad_config_count, 5)
        self.assertEqual(
            args.special_arm_control_mode,
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

    def test_five_pedal_mapping_adds_only_progress_undo(self) -> None:
        self.assertEqual(
            FOOT_PAD_FIVE_EVENT_BY_KEY_CODE,
            {
                183: "fix_arm_left",
                184: "fix_arm_right",
                185: "init_arm_left",
                186: "init_arm_right",
                187: "progress_undo",
            },
        )

        class FakeEvent:
            type = 1
            value = 1
            code = 187

        class FakeCodes:
            EV_KEY = 1

        class FakeEvdev:
            ecodes = FakeCodes()

        class Device:
            def read(self):
                return [FakeEvent()]

            def close(self):
                return None

        source = FootPadSpecialArmInput(Device(), FakeEvdev(), foot_pad_count=5)
        self.assertEqual(source.consume_events(), ["progress_undo"])
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
        self.assertEqual(
            foot_pad_count_from_key_codes({183, 184, 185, 186, 187}),
            5,
        )
        self.assertIsNone(foot_pad_count_from_key_codes({183, 185}))

    def test_progress_guidance_explains_replacing_legacy_live_topology(self) -> None:
        guidance = foot_pad_progress_topology_guidance(
            2,
            progress_enabled=True,
        )
        self.assertIn("当前 live 虚拟键盘为 2 踏板", guidance)
        self.assertIn("hardware.foot_pad --count 5", guidance)
        self.assertIn("不会把旧虚拟键盘热升级", guidance)
        self.assertEqual(
            foot_pad_progress_topology_guidance(5, progress_enabled=True),
            "",
        )

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

    def test_open_online_accepts_five_pedal_keyboard_without_saved_config(self) -> None:
        class FakeCodes:
            EV_KEY = 1

        class FakeDevice:
            name = "DexTeleop foot pads"

            def capabilities(self, *, absinfo=False):
                del absinfo
                return {FakeCodes.EV_KEY: [183, 184, 185, 186, 187]}

            def close(self):
                return None

        class FakeEvdev:
            ecodes = FakeCodes()

            @staticmethod
            def list_devices():
                return ["/dev/input/event-five-pads"]

            @staticmethod
            def InputDevice(_path):
                return FakeDevice()

        with mock.patch(
            "teleop_runtime.real_teleop.foot_pad_controls.configured_foot_pad_count",
            return_value=2,
        ), mock.patch(
            "teleop_runtime.real_teleop.foot_pad_controls._require_evdev",
            return_value=FakeEvdev,
        ):
            source = FootPadSpecialArmInput.open_online()
        self.assertEqual(source.foot_pad_count, 5)
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

    def test_missing_service_mentions_five_pedal_progress_command(self) -> None:
        class FakeEvdev:
            @staticmethod
            def list_devices():
                return []

        with mock.patch(
            "teleop_runtime.real_teleop.foot_pad_controls.configured_foot_pad_count",
            return_value=None,
        ), mock.patch(
            "teleop_runtime.real_teleop.foot_pad_controls._require_evdev",
            return_value=FakeEvdev,
        ), mock.patch(
            "teleop_runtime.real_teleop.foot_pad_controls._input_permission_hint",
            return_value="",
        ):
            with self.assertRaisesRegex(
                FootPadUnavailableError,
                r"--count 5 --recalibrate.*five-pedal",
            ):
                FootPadSpecialArmInput.open_online()

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

    def test_five_pedal_mode_uses_the_two_pedal_abc_dispatch_contract(self) -> None:
        controls = current_active_controls(
            GuiControlState(
                teleop_started=True,
                episode_recording_available=True,
                fix_arm_available=True,
                init_arm_available=True,
                special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
                foot_pad_available=True,
                foot_pad_count=5,
            ),
            GuiControlEvents(),
        )

        # The shared GUI projection still exposes the common fallback controls;
        # the entity-side adapter adds the dedicated B/C routes below.
        self.assertIn("episode_start", controls)
        self.assertIn("h", controls)
        self.assertIn("c", controls)
        self.assertIn("fix_arm_toggle", controls)

        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.args = SimpleNamespace(
            special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
        )
        bind_control_application(control)
        control.foot_pad_input = SimpleNamespace(foot_pad_count=5, is_online=True)
        control._foot_pad_mode_enabled = lambda: True
        control._gui_control_state = lambda: GuiControlState(
            teleop_started=True,
            episode_recording_available=True,
            fix_arm_available=True,
            init_arm_available=True,
            special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
            foot_pad_available=True,
            foot_pad_count=5,
        )
        control._gui_events = lambda: GuiControlEvents()
        control._progress_ui_controls = ()
        control._plus_curation_save_in_flight = lambda: False
        control.plus_curation = None
        control._teleop_started = True
        control._two_pedal_switch_blocked_by_init = lambda: False
        control._hold_requested = False

        self.assertIn(5, FOOT_PAD_TWO_PEDAL_STYLE_COUNTS)
        with mock.patch(
            "control_station.real_teleop.control_availability.current_active_controls",
            return_value=controls,
            create=True,
        ):
            active_controls = control._current_active_controls()
        self.assertIn("two_pedal_b", active_controls)
        self.assertIn("two_pedal_c", active_controls)

    def test_five_pedal_disconnect_preserves_abc_dispatch_contract(self) -> None:
        controls = ["episode_start", "h", "c", "fix_arm_toggle"]
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.args = SimpleNamespace(
            special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
        )
        bind_control_application(control)
        control.foot_pad_input = SimpleNamespace(foot_pad_count=5, is_online=False)
        control.components = SimpleNamespace(
            session=None,
            init_arm_controller=None,
            fix_arm_controller=None,
            active_sides=("left", "right"),
        )
        control._gui_control_state = lambda: GuiControlState(
            teleop_started=True,
            episode_recording_available=True,
            fix_arm_available=True,
            init_arm_available=True,
            special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
            foot_pad_available=False,
            foot_pad_count=5,
        )
        control._gui_events = lambda: GuiControlEvents()
        control._progress_ui_controls = ()
        control._plus_curation_save_in_flight = lambda: False
        control.plus_curation = None
        control._teleop_started = True
        control._two_pedal_switch_blocked_by_init = lambda: False
        control._hold_requested = False

        self.assertTrue(control._foot_pad_mode_configured())
        self.assertFalse(control._foot_pad_mode_enabled())
        self.assertTrue(control._two_pedal_mode_active())
        with mock.patch(
            "control_station.real_teleop.control_availability.current_active_controls",
            return_value=controls,
        ):
            active_controls = control._current_active_controls()
        self.assertIn("two_pedal_b", active_controls)
        self.assertIn("two_pedal_c", active_controls)

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

    def test_five_pedal_test_remains_available_during_regular_hold(self) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        bind_control_application(control)
        control._gui_control_state = lambda: GuiControlState(
            teleop_started=True,
            hold_active=True,
            special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
            foot_pad_available=True,
            foot_pad_count=5,
        )
        control._gui_events = GuiControlEvents
        control._progress_ui_controls = ()
        control._two_pedal_mode_active = lambda: False
        control._teleop_started = True
        control._hold_requested = True

        with mock.patch(
            "control_station.real_teleop.control_availability.current_active_controls",
            current_active_controls,
        ):
            self.assertIn(
                "foot_pad_test_start",
                control._current_active_controls(),
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
        bind_control_application(control)
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
        bind_control_application(control)

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
        bind_control_application(control)

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
        bind_control_application(control)

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

    def test_gui_test_mode_five_pedals_auto_completes_after_all_keys(self) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.ui = SimpleNamespace(
            published=[],
            messages=[],
            publish=lambda **updates: control.ui.published.append(updates),
            append_message=lambda tag, message: control.ui.messages.append(
                (tag, message)
            ),
        )
        control.foot_pad_input = SimpleNamespace(foot_pad_count=5, is_online=True)
        control._foot_pad_test_active = True
        control._foot_pad_test_completed = False
        control._foot_pad_test_seen_events = set()
        control._foot_pad_test_last_event = ""
        control._foot_pad_test_status = ""
        control._current_active_controls = lambda: ["q"]
        control._control_hint = lambda: "normal"
        control.args = SimpleNamespace(foot_pad_requested=True)
        bind_control_application(control)

        events = (
            "fix_arm_left",
            "fix_arm_right",
            "init_arm_left",
            "init_arm_right",
            "progress_undo",
        )
        for index, event in enumerate(events, start=1):
            control._record_foot_pad_test_event(event, now=float(index))
            if index < 5:
                self.assertTrue(control._foot_pad_test_active)

        self.assertFalse(control._foot_pad_test_active)
        self.assertTrue(control._foot_pad_test_completed)
        self.assertEqual(len(control._foot_pad_test_seen_events), 5)
        self.assertEqual(
            control._foot_pad_test_last_event,
            "踏板 5 / F17 → 撤销上一个子任务进度标注",
        )
        self.assertIn("踏板 1–5", control._foot_pad_test_status)
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
        bind_control_application(control)
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

    def test_four_pedal_progress_routes_pedals_three_and_four_to_existing_intents(
        self,
    ) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.args = SimpleNamespace(
            collection_progress_profile=object(),
        )
        bind_control_application(control)
        self._attach_progress_application(control)
        control.ui = SimpleNamespace(
            messages=[],
            append_message=lambda tag, message: control.ui.messages.append(
                (tag, message)
            ),
        )
        control._progress_operator_intents = __import__("collections").deque()
        control._progress_operator_intent_sequence = 0
        control._progress_ui_controls = ("progress_advance", "progress_failure")
        control._progress_ui_payload = {"enabled": True}
        control._progress_ui_transport_revision = 0
        control._foot_pad_mode_enabled = lambda: True
        control._foot_pad_count = lambda: 4
        control._fix_init_allowed_in_controls = lambda: True
        control.components = SimpleNamespace(
            init_arm_controller=SimpleNamespace(
                active=False,
                selected=[],
                select_side=lambda side: control.components.init_arm_controller.selected.append(
                    side
                ),
            )
        )
        exact_calls: list[float] = []
        control._handle_init_arm_move_to_qpos_event = (
            lambda *, now: exact_calls.append(float(now))
        )

        control._handle_foot_pad_special_arm_event(
            "init_arm_left",
            now=20.0,
        )
        advance = control.consume_progress_operator_intents()
        self.assertEqual(len(advance), 1)
        self.assertEqual(advance[0].kind, "ADVANCE_PROGRESS")
        self.assertTrue(advance[0].intent_id.startswith(
            "real-teleop-progress-foot-pad:"
        ))

        control._progress_ui_controls = ("progress_failure",)
        control._handle_foot_pad_special_arm_event(
            "init_arm_right",
            now=21.0,
        )
        failure = control.consume_progress_operator_intents()
        self.assertEqual(len(failure), 1)
        self.assertEqual(failure[0].kind, "REPORT_FAILURE")
        self.assertEqual(exact_calls, [])
        self.assertEqual(
            control.components.init_arm_controller.selected,
            [],
        )

    def test_four_pedal_progress_unavailable_event_never_falls_back_to_init(
        self,
    ) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.args = SimpleNamespace(
            collection_progress_profile=object(),
        )
        bind_control_application(control)
        self._attach_progress_application(control)
        control.ui = None
        control._progress_operator_intents = __import__("collections").deque()
        control._progress_operator_intent_sequence = 0
        control._progress_ui_controls = ()
        control._foot_pad_mode_enabled = lambda: True
        control._foot_pad_count = lambda: 4
        control._fix_init_allowed_in_controls = lambda: True
        selected: list[str] = []
        control.components = SimpleNamespace(
            init_arm_controller=SimpleNamespace(
                active=False,
                select_side=lambda side: selected.append(side),
            )
        )
        exact_calls: list[float] = []
        control._handle_init_arm_move_to_qpos_event = (
            lambda *, now: exact_calls.append(float(now))
        )

        control._handle_foot_pad_special_arm_event(
            "init_arm_left",
            now=22.0,
        )
        control._handle_foot_pad_special_arm_event(
            "init_arm_right",
            now=23.0,
        )

        self.assertEqual(control.consume_progress_operator_intents(), ())
        self.assertEqual(selected, [])
        self.assertEqual(exact_calls, [])

    def test_progress_pedal_admission_is_independent_of_hidden_gui(self) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.args = SimpleNamespace(collection_progress_profile=object())
        bind_control_application(control)
        self._attach_progress_application(control)
        control.ui = None
        control._progress_operator_intents = __import__("collections").deque()
        control._progress_operator_intent_sequence = 0
        control._progress_ui_controls = ()
        control._progress_ui_payload = {"enabled": False}
        control._progress_ui_transport_revision = 0
        control._foot_pad_mode_enabled = lambda: True
        control._foot_pad_count = lambda: 4
        control.publish_progress_control_availability(
            configured=True,
            active_controls=["progress_failure", "progress_advance"],
            recording_active=True,
            phase="ACTIVE_ATTEMPT",
            application_held=False,
            allowed_next_step_ids=["next"],
        )

        control._handle_foot_pad_special_arm_event("init_arm_left", now=24.0)

        intents = control.consume_progress_operator_intents()
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].kind, "ADVANCE_PROGRESS")

    def test_progress_pedal_rejection_reports_idle_episode_reason(self) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.args = SimpleNamespace(collection_progress_profile=object())
        bind_control_application(control)
        self._attach_progress_application(control)
        control.ui = None
        control._progress_operator_intents = __import__("collections").deque()
        control._progress_operator_intent_sequence = 0
        control._progress_ui_controls = ()
        control._foot_pad_mode_enabled = lambda: True
        control._foot_pad_count = lambda: 4
        messages: list[str] = []
        control._publish_progress_pedal_message = (
            lambda message, *, now: messages.append(str(message))
        )
        control.publish_progress_control_availability(
            configured=True,
            active_controls=[],
            recording_active=False,
            phase="IDLE",
            application_held=False,
            allowed_next_step_ids=["next"],
        )

        control._handle_foot_pad_special_arm_event("init_arm_left", now=25.0)

        self.assertEqual(control.consume_progress_operator_intents(), ())
        self.assertEqual(len(messages), 1)
        self.assertIn("当前没有活动的正式 Episode", messages[0])
        self.assertIn("未执行 Init", messages[0])

    def test_five_pedal_routes_fifth_pedal_to_existing_undo_intent(self) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.args = SimpleNamespace(collection_progress_profile=object())
        bind_control_application(control)
        self._attach_progress_application(control)
        control.ui = None
        control._progress_operator_intents = __import__("collections").deque()
        control._progress_operator_intent_sequence = 0
        control._progress_ui_controls = ("progress_undo",)
        control._progress_ui_payload = {"enabled": True}
        control._progress_ui_transport_revision = 0
        control._foot_pad_mode_enabled = lambda: True
        control._foot_pad_count = lambda: 5

        control._handle_foot_pad_special_arm_event("progress_undo", now=24.0)

        intents = control.consume_progress_operator_intents()
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].kind, "UNDO_LAST_ADVANCE")
        self.assertTrue(
            intents[0].intent_id.startswith("real-teleop-progress-foot-pad:")
        )

    def test_five_pedal_undo_is_not_queued_when_current_state_disallows_it(
        self,
    ) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.args = SimpleNamespace(collection_progress_profile=object())
        bind_control_application(control)
        self._attach_progress_application(control)
        control.ui = None
        control._progress_operator_intents = __import__("collections").deque()
        control._progress_operator_intent_sequence = 0
        control._progress_ui_controls = ()
        control._foot_pad_mode_enabled = lambda: True
        control._foot_pad_count = lambda: 5

        control._handle_foot_pad_special_arm_event("progress_undo", now=24.5)
        self.assertEqual(control.consume_progress_operator_intents(), ())

        control._progress_ui_controls = ("progress_undo",)
        self.assertEqual(control.consume_progress_operator_intents(), ())

    def test_five_pedal_fix_prefix_matches_two_pedal_side_switch_without_profile(
        self,
    ) -> None:
        for arm_control_mode in ("position", "impedance"):
            control = RealTeleopControlInput.__new__(RealTeleopControlInput)
            control.args = SimpleNamespace(
                arm_control_mode=arm_control_mode,
                collection_progress_profile=None,
            )
            bind_control_application(control)
            control.ui = None
            control._foot_pad_mode_enabled = lambda: True
            control._foot_pad_count = lambda: 5
            control._two_pedal_mode_active = lambda: True
            requests: list[tuple[str | None, float, str]] = []
            control._request_two_pedal_fixed_side = (
                lambda side, *, now, source: requests.append((side, now, source))
            )

            control._handle_foot_pad_special_arm_event(
                "fix_arm_left",
                now=25.0,
            )
            control._handle_foot_pad_special_arm_event(
                "fix_arm_right",
                now=26.0,
            )
            self.assertEqual(
                requests,
                [
                    ("right", 25.0, "pedal_1"),
                    ("left", 26.0, "pedal_2"),
                ],
            )

    def test_five_pedal_progress_prefix_is_shared_by_position_and_impedance(self) -> None:
        for arm_control_mode in ("position", "impedance"):
            control = RealTeleopControlInput.__new__(RealTeleopControlInput)
            control.args = SimpleNamespace(
                arm_control_mode=arm_control_mode,
                collection_progress_profile=object(),
            )
            bind_control_application(control)
            self._attach_progress_application(control)
            control.ui = None
            control._progress_operator_intents = __import__("collections").deque()
            control._progress_operator_intent_sequence = 0
            control._progress_ui_controls = (
                "progress_advance",
                "progress_failure",
                "progress_undo",
            )
            control._progress_ui_payload = {"enabled": True}
            control._progress_ui_transport_revision = 0
            control._foot_pad_mode_enabled = lambda: True
            control._foot_pad_count = lambda: 5

            observed_kinds: list[str] = []
            for event in (
                "init_arm_left",
                "init_arm_right",
                "progress_undo",
            ):
                control._handle_foot_pad_special_arm_event(event, now=28.0)
                observed_kinds.extend(
                    intent.kind
                    for intent in control.consume_progress_operator_intents()
                )
                # submit_progress_operator_intent clears one-shot progress
                # controls until the canonical state is rebuilt. Simulate that
                # rebuild so the three pedal mappings can be checked in order.
                control._progress_ui_controls = (
                    "progress_advance",
                    "progress_failure",
                    "progress_undo",
                )

            self.assertEqual(
                observed_kinds,
                ["ADVANCE_PROGRESS", "REPORT_FAILURE", "UNDO_LAST_ADVANCE"],
            )

    def test_four_pedal_without_progress_keeps_existing_init_mapping(self) -> None:
        class InitController:
            active = False

            def __init__(self) -> None:
                self.selected: list[str] = []

            def select_side(self, side: str):
                self.selected.append(side)
                return SimpleNamespace(ok=True, message="selected")

        controller = InitController()
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.args = SimpleNamespace(collection_progress_profile=None)
        bind_control_application(control)
        control.components = SimpleNamespace(init_arm_controller=controller)
        control._foot_pad_mode_enabled = lambda: True
        control._foot_pad_count = lambda: 4
        control._fix_init_allowed_in_controls = lambda: True
        exact_calls: list[float] = []
        control._handle_init_arm_move_to_qpos_event = (
            lambda *, now: exact_calls.append(float(now))
        )

        control._handle_foot_pad_special_arm_event(
            "init_arm_left",
            now=24.0,
        )

        self.assertEqual(controller.selected, ["left"])
        self.assertEqual(exact_calls, [24.0])

    def test_two_pedal_routes_mean_moving_side_and_never_init(self) -> None:
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        bind_control_application(control)
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

    def test_rule_task_chain_two_pedals_use_dedicated_application_ports(self) -> None:
        calls: list[str] = []
        messages: list[tuple[str, str]] = []
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        control.args = SimpleNamespace(rule_task_chain_foot_pad=True)
        bind_control_application(control)
        control.ui = SimpleNamespace(
            submit_task_chain_advance_from_foot_pad=lambda: (
                calls.append("advance") or True
            ),
            task_chain_foot_pad_action_label=lambda: "正在激活随机配置",
            submit_emergency_stop_from_foot_pad=lambda: (
                calls.append("stop") or True
            ),
            append_message=lambda tag, message: messages.append((tag, message)),
        )
        control._foot_pad_mode_enabled = lambda: True
        control._foot_pad_count = lambda: 2

        control._handle_foot_pad_special_arm_event("fix_arm_left", now=10.0)
        control._handle_foot_pad_special_arm_event("fix_arm_right", now=11.0)

        self.assertEqual(calls, ["advance", "stop"])
        self.assertEqual([tag for tag, _message in messages], ["FootPad", "FootPad"])
        self.assertIn("正在激活随机配置", messages[0][1])
        self.assertIn("停止并 Hold", messages[1][1])

    def test_two_pedal_alerts_replace_each_other_and_red_expires(self) -> None:
        published = []
        control = RealTeleopControlInput.__new__(RealTeleopControlInput)
        bind_control_application(control)
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
        bind_control_application(control)
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

    def test_startup_foot_pad_transaction_registers_live_input_and_guidance(self) -> None:
        events = []

        class Input:
            is_online = True
            foot_pad_count = 5

            @classmethod
            def open_online(cls):
                events.append("open")
                return cls()

            @classmethod
            def waiting_for_service(cls):
                raise AssertionError("waiting fallback should not be used")

            def close(self):
                events.append("close")

        progress = SimpleNamespace(
            begin=lambda key, **details: events.append(("begin", key, details)),
            succeed=lambda key, **details: events.append(("succeed", key, details)),
            fail=lambda *_args, **_kwargs: self.fail("unexpected failure receipt"),
        )
        cleanup = SimpleNamespace(
            add=lambda label, callback: events.append(("cleanup", label, callback))
        )
        ui = SimpleNamespace(
            append_message=lambda tag, message: events.append(("ui", tag, message))
        )
        args = SimpleNamespace(
            special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
            collection_progress_profile="linear",
            collection_progress_gui=True,
        )

        source = initialize_startup_foot_pad_input(
            args,
            foot_pad_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
            input_type=Input,
            unavailable_error_type=FootPadUnavailableError,
            startup_cleanup=cleanup,
            startup_progress=progress,
            control_ui=ui,
            apply_live_topology=lambda _args, count: (
                events.append(("topology", count)) or SPECIAL_ARM_CONTROL_MODE_FOOT_PAD
            ),
            topology_guidance=lambda count, **_kwargs: f"guidance:{count}",
            restore_init_after_fallback=lambda _args, **kwargs: (
                events.append(("restore", kwargs["live_foot_pad_count"])) or (True, "restored")
            ),
            log=lambda tag, message: events.append(("log", tag, message)),
        )

        self.assertIsInstance(source, Input)
        self.assertTrue(args.foot_pad_requested)
        self.assertEqual(args.foot_pad_startup_error, "")
        self.assertIn(("topology", 5), events)
        self.assertIn(("restore", 5), events)
        self.assertTrue(any(event[:2] == ("cleanup", "foot_pad") for event in events))
        self.assertIn(("ui", "FootPad", "guidance:5"), events)
        self.assertIn(("ui", "InitArm", "restored"), events)

    def test_startup_foot_pad_transaction_installs_retryable_waiting_fallback(self) -> None:
        events = []

        class Input:
            is_online = True
            foot_pad_count = 4

            def __init__(self, *, waiting: bool = False) -> None:
                self.waiting = waiting

            @classmethod
            def open_online(cls):
                source = cls()
                events.append(("candidate", source))
                return source

            @classmethod
            def waiting_for_service(cls):
                source = cls(waiting=True)
                source.is_online = False
                events.append(("waiting", source))
                return source

            def close(self):
                events.append(("close", self))

        args = SimpleNamespace(
            special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
            collection_progress_profile=None,
            collection_progress_gui=False,
        )
        progress = SimpleNamespace(
            begin=lambda *_args, **_kwargs: None,
            succeed=lambda *_args, **_kwargs: self.fail("unexpected success receipt"),
            fail=lambda key, detail, **kwargs: events.append(
                ("fail", key, detail, kwargs)
            ),
        )

        source = initialize_startup_foot_pad_input(
            args,
            foot_pad_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
            input_type=Input,
            unavailable_error_type=FootPadUnavailableError,
            startup_cleanup=SimpleNamespace(
                add=lambda label, callback: events.append(("cleanup", label, callback))
            ),
            startup_progress=progress,
            control_ui=None,
            apply_live_topology=lambda *_args: (_ for _ in ()).throw(
                ValueError("unsupported live topology")
            ),
            topology_guidance=lambda *_args, **_kwargs: "",
            restore_init_after_fallback=lambda _args, **kwargs: (
                events.append(("restore", kwargs["live_foot_pad_count"])) or (False, "")
            ),
            log=lambda tag, message: events.append(("log", tag, message)),
        )

        candidate = next(item[1] for item in events if item[0] == "candidate")
        self.assertTrue(candidate in [item[1] for item in events if item[0] == "close"])
        self.assertTrue(source.waiting)
        self.assertFalse(source.is_online)
        self.assertEqual(args.foot_pad_startup_error, "ValueError: unsupported live topology")
        self.assertIn(("restore", None), events)
        self.assertTrue(any(event[:2] == ("cleanup", "foot_pad") for event in events))

    def test_startup_foot_pad_transaction_keeps_manual_mode_device_free(self) -> None:
        restored = []

        class Input:
            @classmethod
            def open_online(cls):
                self.fail("manual startup must not open the foot-pad device")

        args = SimpleNamespace(special_arm_control_mode=SPECIAL_ARM_CONTROL_MODE_MANUAL)
        source = initialize_startup_foot_pad_input(
            args,
            foot_pad_control_mode=SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
            input_type=Input,
            unavailable_error_type=FootPadUnavailableError,
            startup_cleanup=SimpleNamespace(add=lambda *_args: None),
            startup_progress=None,
            control_ui=None,
            apply_live_topology=lambda *_args: SPECIAL_ARM_CONTROL_MODE_MANUAL,
            topology_guidance=lambda *_args, **_kwargs: "",
            restore_init_after_fallback=lambda _args, **kwargs: (
                restored.append(kwargs["live_foot_pad_count"]) or (False, "")
            ),
            log=lambda *_args: None,
        )

        self.assertIsNone(source)
        self.assertFalse(args.foot_pad_requested)
        self.assertEqual(args.foot_pad_startup_error, "")
        self.assertEqual(restored, [None])


if __name__ == "__main__":
    unittest.main()
