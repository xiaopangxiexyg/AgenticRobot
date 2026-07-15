from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from teleop_runtime.configs import rm65_arm_ips
from teleop_runtime import device_preflight
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop


def _dmtac_scan() -> dict:
    return {
        "schema_version": "dmtac_usb_scan.v1",
        "hostname": "unit-host",
        "grippers": [
            {
                "index": 0,
                "port": "/dev/serial/by-id/left-gripper",
                "preferred_port": "/dev/serial/by-id/left-gripper",
                "serial_number": "LG",
                "hwid": "USB VID:PID=1A86:7523 SER=LG",
                "probe": {"reachable": True},
            },
            {
                "index": 1,
                "port": "/dev/serial/by-id/right-gripper",
                "preferred_port": "/dev/serial/by-id/right-gripper",
                "serial_number": "RG",
                "hwid": "USB VID:PID=1A86:7523 SER=RG",
                "probe": {"reachable": True},
            },
        ],
        "tactile": [
            {"index": 0, "dev_id": "L_IN"},
            {"index": 1, "dev_id": "L_OUT"},
            {"index": 2, "dev_id": "R_IN"},
            {"index": 3, "dev_id": "R_OUT"},
        ],
    }


def _q6_scan() -> dict:
    return {
        "schema_version": "hand_q6_scan.v1",
        "hostname": "unit-host",
        "baudrate": 115200,
        "timeout": 0.05,
        "q6_devices": [
            {
                "q6_index": 0,
                "index": 34,
                "port": "/dev/ttyACM0",
                "device": "/dev/ttyACM0",
                "preferred_port": "/dev/serial/by-id/q6-left",
                "port_aliases": ["/dev/ttyACM0", "/dev/serial/by-id/q6-left"],
                "probe": {"reachable": True, "valid_frames": 1},
            },
            {
                "q6_index": 1,
                "index": 35,
                "port": "/dev/ttyACM1",
                "device": "/dev/ttyACM1",
                "preferred_port": "/dev/serial/by-id/q6-right",
                "port_aliases": ["/dev/ttyACM1", "/dev/serial/by-id/q6-right"],
                "probe": {"reachable": True, "valid_frames": 1},
            },
        ],
        "serial_ports": [],
    }


class DevicePreflightTests(unittest.TestCase):
    def test_realsense_compare_reports_missing_configured_serials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "realsense.json"
            path.write_text(
                json.dumps(
                    {
                        "front": {"serial": "A", "width": 640, "height": 480, "fps": 30},
                        "left_wrist": {"serial": "B", "width": 640, "height": 480, "fps": 30},
                    }
                ),
                encoding="utf-8",
            )

            comparison = device_preflight.compare_realsense_config(
                path,
                [{"index": 0, "serial": "A"}, {"index": 1, "serial": "C"}],
            )

        self.assertEqual(comparison["missing"], [{"camera_key": "left_wrist", "serial": "B"}])
        self.assertEqual(len(comparison["matched"]), 1)
        self.assertEqual(comparison["extra"][0]["serial"], "C")

    def test_permission_report_for_unwritable_path_includes_repair_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "devnode"
            path.write_text("", encoding="utf-8")
            path.chmod(0o000)
            try:
                reports = device_preflight.repair_device_permissions([path], apply=False, user="tester")
            finally:
                path.chmod(0o600)

        self.assertEqual(reports[0]["command"], f"chmod 666 {path}")
        text = device_preflight._permission_error_text(reports)
        self.assertIn("sudo chmod 666", text)

    def test_fix_permissions_chmods_even_when_current_process_can_access_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "devnode"
            path.write_text("", encoding="utf-8")
            path.chmod(0o660)
            with patch.object(device_preflight.os, "geteuid", return_value=0):
                reports = device_preflight.repair_device_permissions([path], apply=True)

            mode = path.stat().st_mode & 0o777

        self.assertEqual(mode, 0o666)
        self.assertTrue(reports[0]["changed"])
        self.assertEqual(reports[0]["target_mode"], "0o666")

    def test_cli_fix_permissions_does_not_validate_device_counts(self) -> None:
        permissions_result = device_preflight.DevicePreflightResult(
            ok=True,
            messages=["permissions checked"],
        )
        with (
            patch.object(device_preflight, "prepare_device_permissions", return_value=permissions_result),
            patch.object(
                device_preflight,
                "prepare_devices_for_teleop",
                side_effect=AssertionError("device validation should not run"),
            ),
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = device_preflight.main(["--active-sides", "both", "--fix-permissions"])

        self.assertEqual(code, 0)
        self.assertIn("[DevicePermissions] OK", stdout.getvalue())

    def test_cli_permissions_only_warns_when_permissions_need_attention(self) -> None:
        permissions_result = device_preflight.DevicePreflightResult(
            ok=True,
            messages=["needs sudo"],
            permission_fixes=[{"after": {"readable": False, "writable": False}}],
        )
        with patch.object(device_preflight, "prepare_device_permissions", return_value=permissions_result):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = device_preflight.main(["--permissions-only"])

        self.assertEqual(code, 0)
        self.assertIn("needs sudo", stdout.getvalue())
        self.assertIn("[DevicePermissions] WARNING", stdout.getvalue())

    def test_cli_fix_permissions_fails_when_chmod_does_not_take_effect(self) -> None:
        permissions_result = device_preflight.DevicePreflightResult(
            ok=False,
            messages=["root required"],
        )
        with patch.object(device_preflight, "prepare_device_permissions", return_value=permissions_result):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = device_preflight.main(["--fix-permissions"])

        self.assertEqual(code, 2)
        self.assertIn("root required", stdout.getvalue())

    def test_serial_conflict_text_includes_brltty_standard_operation(self) -> None:
        text = device_preflight._serial_conflict_text(
            {
                "risk": True,
                "rule_paths": ["/usr/lib/udev/rules.d/85-brltty.rules"],
                "active_units": ["brltty.service"],
                "enabled_units": [],
                "processes": [],
            }
        )

        self.assertIn("brltty", text)
        self.assertIn("--fix-serial-conflicts", text)
        self.assertIn("Replug the gripper USB/HUB", text)

    def test_repair_serial_conflicts_requires_root_when_applying(self) -> None:
        with (
            patch.object(
                device_preflight,
                "detect_brltty_serial_conflict",
                return_value={"risk": True, "rule_paths": ["/usr/lib/udev/rules.d/85-brltty.rules"]},
            ),
            patch.object(device_preflight.os, "geteuid", return_value=1000),
        ):
            report = device_preflight.repair_serial_conflicts(apply=True)

        self.assertFalse(report["ok"])
        self.assertIn("root_required", report["error"])

    def test_repair_serial_conflicts_stops_and_masks_brltty_when_root(self) -> None:
        calls: list[tuple[str, ...]] = []

        def fake_run(command, **_kwargs):
            calls.append(tuple(command))
            return {
                "command": " ".join(command),
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            }

        with (
            patch.object(
                device_preflight,
                "detect_brltty_serial_conflict",
                return_value={"risk": True, "rule_paths": ["/usr/lib/udev/rules.d/85-brltty.rules"]},
            ),
            patch.object(device_preflight.os, "geteuid", return_value=0),
            patch.object(device_preflight, "_run_command", fake_run),
        ):
            report = device_preflight.repair_serial_conflicts(apply=True)

        self.assertTrue(report["ok"])
        self.assertTrue(report["changed"])
        self.assertIn(("systemctl", "stop", "brltty.service"), calls)
        self.assertIn(("systemctl", "disable", "brltty.service"), calls)
        self.assertIn(("systemctl", "mask", "brltty.service"), calls)
        self.assertIn(("udevadm", "control", "--reload-rules"), calls)
        self.assertIn(("modprobe", "ch341"), calls)

    def test_cli_fix_serial_conflicts_does_not_validate_device_counts(self) -> None:
        permissions_result = device_preflight.DevicePreflightResult(
            ok=True,
            messages=["serial conflict fixed"],
            serial_conflict_report={"risk": True, "changed": True, "ok": True},
        )
        with (
            patch.object(device_preflight, "prepare_device_permissions", return_value=permissions_result),
            patch.object(
                device_preflight,
                "prepare_devices_for_teleop",
                side_effect=AssertionError("device validation should not run"),
            ),
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = device_preflight.main(["--fix-serial-conflicts"])

        self.assertEqual(code, 0)
        self.assertIn("serial conflict fixed", stdout.getvalue())
        self.assertIn("[DevicePermissions] OK", stdout.getvalue())

    def test_cli_check_devices_explicitly_runs_device_preflight(self) -> None:
        preflight_result = device_preflight.DevicePreflightResult(
            ok=True,
            messages=["device preflight checked"],
        )
        with patch.object(device_preflight, "prepare_devices_for_teleop", return_value=preflight_result) as mocked:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = device_preflight.main(["--check-devices", "--check-only"])

        self.assertEqual(code, 0)
        self.assertEqual(mocked.call_count, 1)
        self.assertFalse(mocked.call_args.args[0].auto_configure)
        self.assertIn("[DevicePreflight] OK", stdout.getvalue())

    def test_q6_preflight_uses_preview_confirmation_for_new_devices(self) -> None:
        q6 = device_preflight._load_q6_module()
        preview_calls: list[list[dict]] = []

        def fake_preview(devices, **_kwargs):
            preview_calls.append([dict(item) for item in devices])

        result = device_preflight.DevicePreflightResult(ok=True)
        options = device_preflight.DevicePreflightOptions(
            active_sides=("left", "right"),
            assume_yes=False,
        )
        with (
            patch.object(q6, "scan_q6_devices", return_value=_q6_scan()),
            patch.object(q6, "load_assignment", side_effect=FileNotFoundError),
            patch.object(q6, "save_assignment", return_value=Path("/tmp/q6.json")),
            patch.object(q6, "visualize_q6_devices", fake_preview),
            patch.object(device_preflight.sys, "stdin", SimpleNamespace(isatty=lambda: True)),
            patch("builtins.input", side_effect=["1", "0"]),
        ):
            device_preflight.prepare_q6(options, result)

        self.assertEqual(len(preview_calls), 1)
        self.assertEqual(
            result.q6_ports,
            {
                "left": "/dev/serial/by-id/q6-right",
                "right": "/dev/serial/by-id/q6-left",
            },
        )

    def test_q6_preflight_retries_all_serial_before_reassigning(self) -> None:
        q6 = device_preflight._load_q6_module()
        first_scan = _q6_scan()
        first_scan["q6_devices"] = []
        retry_scan = _q6_scan()
        result = device_preflight.DevicePreflightResult(ok=True)
        options = device_preflight.DevicePreflightOptions(
            active_sides=("left", "right"),
            assume_yes=True,
            auto_configure=True,
        )
        with (
            patch.object(q6, "scan_q6_devices", side_effect=[first_scan, retry_scan]) as scan_mock,
            patch.object(q6, "load_assignment", side_effect=FileNotFoundError),
            patch.object(q6, "save_assignment", return_value=Path("/tmp/q6.json")),
        ):
            device_preflight.prepare_q6(options, result)

        self.assertEqual(scan_mock.call_args_list[0].kwargs["probe_all_serial"], False)
        self.assertEqual(scan_mock.call_args_list[1].kwargs["probe_all_serial"], True)
        self.assertEqual(
            result.q6_ports,
            {
                "left": "/dev/serial/by-id/q6-left",
                "right": "/dev/serial/by-id/q6-right",
            },
        )
        self.assertTrue(any("retrying with all serial ports" in message for message in result.messages))

    def test_q6_force_reconfigure_skips_valid_latest_and_saves(self) -> None:
        q6 = device_preflight._load_q6_module()
        scan = _q6_scan()
        result = device_preflight.DevicePreflightResult(ok=True)
        options = device_preflight.DevicePreflightOptions(
            active_sides=("left", "right"),
            assume_yes=True,
            auto_configure=True,
            force_reconfigure=True,
        )
        with (
            patch.object(q6, "scan_q6_devices", return_value=scan),
            patch.object(q6, "load_assignment", side_effect=AssertionError("should skip latest")) as load_mock,
            patch.object(q6, "save_assignment", return_value=Path("/tmp/q6.json")) as save_mock,
        ):
            device_preflight.prepare_q6(options, result)

        self.assertEqual(load_mock.call_count, 0)
        self.assertEqual(save_mock.call_count, 1)
        self.assertEqual(
            result.q6_ports,
            {
                "left": "/dev/serial/by-id/q6-left",
                "right": "/dev/serial/by-id/q6-right",
            },
        )
        self.assertTrue(any("force reconfigure requested" in message for message in result.messages))

    def test_dmtac_partial_config_infers_single_missing_gripper(self) -> None:
        usb_assignment, *_ = device_preflight._load_dmtac_modules()
        scan = _dmtac_scan()
        assignment = usb_assignment.build_assignment_from_tokens(scan, left_gripper="0")
        comparison = usb_assignment.compare_assignment_to_scan(assignment, scan)

        inferred = device_preflight._infer_dmtac_assignment_from_partial_config(
            usb_assignment,
            scan,
            assignment,
            comparison,
            active_sides=("left", "right"),
            require_tactile=False,
            assume_yes=True,
        )

        self.assertIsNotNone(inferred)
        assert inferred is not None
        self.assertEqual(
            inferred["assignments"]["left"]["gripper"]["port"],
            "/dev/serial/by-id/left-gripper",
        )
        self.assertEqual(
            inferred["assignments"]["right"]["gripper"]["port"],
            "/dev/serial/by-id/right-gripper",
        )

    def test_dmtac_ambiguous_grippers_require_motion_confirmation(self) -> None:
        usb_assignment, *_ = device_preflight._load_dmtac_modules()
        scan = _dmtac_scan()
        assignment = usb_assignment.build_assignment_from_tokens(scan)
        comparison = usb_assignment.compare_assignment_to_scan(assignment, scan)
        inferred = device_preflight._infer_dmtac_assignment_from_partial_config(
            usb_assignment,
            scan,
            assignment,
            comparison,
            active_sides=("left", "right"),
            require_tactile=False,
            assume_yes=True,
        )
        calls: list[str] = []
        answers = iter(["", "left", "", "right"])

        tokens = device_preflight._motion_confirm_dmtac_gripper_tokens(
            usb_assignment,
            scan,
            active_sides=("left", "right"),
            input_func=lambda _prompt: next(answers),
            motion_func=lambda port, **_kwargs: calls.append(port) or {"ok": True},
        )

        self.assertIsNone(inferred)
        self.assertEqual(tokens, {"left": "0", "right": "1"})
        self.assertEqual(calls, ["/dev/serial/by-id/left-gripper", "/dev/serial/by-id/right-gripper"])

    def test_dmtac_preflight_retries_all_serial_before_reassigning(self) -> None:
        usb_assignment, *_ = device_preflight._load_dmtac_modules()
        first_scan = _dmtac_scan()
        for item in first_scan["grippers"]:
            item["probe"] = {"reachable": False, "error": "candidate filter missed device"}
        retry_scan = _dmtac_scan()
        result = device_preflight.DevicePreflightResult(ok=True)
        options = device_preflight.DevicePreflightOptions(
            active_sides=("left", "right"),
            require_dmtac_tactile=False,
            assume_yes=True,
            auto_configure=True,
        )
        with (
            patch.object(usb_assignment, "scan_usb_devices", side_effect=[first_scan, retry_scan]) as scan_mock,
            patch.object(usb_assignment, "load_assignment", side_effect=FileNotFoundError),
            patch.object(usb_assignment, "save_assignment", return_value=Path("/tmp/dmtac.json")),
        ):
            device_preflight.prepare_dmtac(options, result)

        self.assertEqual(scan_mock.call_args_list[0].kwargs["probe_all_serial"], False)
        self.assertEqual(scan_mock.call_args_list[1].kwargs["probe_all_serial"], True)
        self.assertEqual(
            result.dmtac_assignment["assignments"]["left"]["gripper"]["port"],
            "/dev/serial/by-id/left-gripper",
        )
        self.assertTrue(any("retrying with all serial ports" in message for message in result.messages))

    def test_dmtac_force_reconfigure_skips_valid_latest_and_saves(self) -> None:
        usb_assignment, *_ = device_preflight._load_dmtac_modules()
        scan = _dmtac_scan()
        result = device_preflight.DevicePreflightResult(ok=True)
        options = device_preflight.DevicePreflightOptions(
            active_sides=("left", "right"),
            require_dmtac_tactile=False,
            assume_yes=True,
            auto_configure=True,
            force_reconfigure=True,
        )
        with (
            patch.object(usb_assignment, "scan_usb_devices", return_value=scan),
            patch.object(
                usb_assignment,
                "load_assignment",
                side_effect=AssertionError("should skip latest"),
            ) as load_mock,
            patch.object(usb_assignment, "save_assignment", return_value=Path("/tmp/dmtac.json")) as save_mock,
        ):
            device_preflight.prepare_dmtac(options, result)

        self.assertEqual(load_mock.call_count, 0)
        self.assertEqual(save_mock.call_count, 1)
        self.assertEqual(
            result.dmtac_assignment["assignments"]["left"]["gripper"]["port"],
            "/dev/serial/by-id/left-gripper",
        )
        self.assertTrue(any("force reconfigure requested" in message for message in result.messages))

    def test_dmtac_preflight_keeps_complete_config_when_gripper_probe_fails(self) -> None:
        usb_assignment, *_ = device_preflight._load_dmtac_modules()
        base_scan = _dmtac_scan()
        assignment = usb_assignment.build_assignment_from_tokens(
            base_scan,
            left_gripper="0",
            right_gripper="1",
        )
        probe_failed_scan = _dmtac_scan()
        for item in probe_failed_scan["grippers"]:
            item["probe"] = {"reachable": False, "error": "serial port busy"}

        result = device_preflight.DevicePreflightResult(ok=True)
        options = device_preflight.DevicePreflightOptions(
            active_sides=("left", "right"),
            require_dmtac_tactile=False,
            auto_configure=True,
        )
        with (
            patch.object(usb_assignment, "scan_usb_devices", return_value=probe_failed_scan),
            patch.object(usb_assignment, "load_assignment", return_value=assignment),
        ):
            device_preflight.prepare_dmtac(options, result)

        self.assertIsNotNone(result.dmtac_assignment)
        self.assertEqual(
            result.dmtac_assignment["assignments"]["left"]["gripper"]["port"],
            "/dev/serial/by-id/left-gripper",
        )
        self.assertTrue(any("startup probe failed" in message for message in result.messages))
        self.assertFalse(any("[DMTAC CONFIG ERROR]" in message for message in result.messages))

    def test_dmtac_preflight_still_fails_without_config_when_no_gripper_reachable(self) -> None:
        usb_assignment, *_ = device_preflight._load_dmtac_modules()
        scan = _dmtac_scan()
        for item in scan["grippers"]:
            item["probe"] = {"reachable": False, "error": "serial port busy"}

        result = device_preflight.DevicePreflightResult(ok=True)
        options = device_preflight.DevicePreflightOptions(
            active_sides=("left", "right"),
            require_dmtac_tactile=False,
            auto_configure=True,
        )
        with (
            patch.object(usb_assignment, "scan_usb_devices", return_value=scan),
            patch.object(usb_assignment, "load_assignment", side_effect=FileNotFoundError),
        ):
            with self.assertRaisesRegex(device_preflight.DevicePreflightError, "reachable 0, expected 2"):
                device_preflight.prepare_dmtac(options, result)

    def test_rm65_arm_ip_assignment_saves_and_loads_latest(self) -> None:
        assignment = rm65_arm_ips.build_assignment(
            left_arm_ip="192.168.10.21",
            right_arm_ip="192.168.10.22",
        )
        with tempfile.TemporaryDirectory() as tmp:
            saved = rm65_arm_ips.save_assignment(assignment, config_dir=tmp)
            loaded = rm65_arm_ips.load_assignment("latest", config_dir=tmp)

        self.assertTrue(saved.name.endswith("_rm65_arm_ip_assignment.json"))
        self.assertEqual(
            rm65_arm_ips.arm_ips_for_sides(loaded),
            {"left": "192.168.10.21", "right": "192.168.10.22"},
        )

    def test_rm65_arm_ip_default_config_dir_lives_under_teleop_runtime_configs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            rm65_arm_ips.default_config_dir(),
            root / "teleop_runtime" / "configs" / "hardware" / "rm65_arm_ips",
        )

    def test_rm65_arm_ip_latest_falls_back_to_legacy_config_dir_but_saves_new_default(self) -> None:
        assignment = rm65_arm_ips.build_assignment(
            left_arm_ip="192.168.10.21",
            right_arm_ip="192.168.10.22",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            new_dir = root / "teleop_runtime" / "configs" / "hardware" / "rm65_arm_ips"
            legacy_dir = root / "teleop_records" / "rm65_arm_ip_configs"
            rm65_arm_ips.save_assignment(assignment, config_dir=legacy_dir)

            with (
                patch.object(rm65_arm_ips, "default_config_dir", return_value=new_dir),
                patch.object(rm65_arm_ips, "legacy_config_dirs", return_value=(legacy_dir,)),
            ):
                loaded = rm65_arm_ips.load_assignment("latest")
                saved = rm65_arm_ips.save_assignment(assignment)

        self.assertEqual(
            rm65_arm_ips.arm_ips_for_sides(loaded),
            {"left": "192.168.10.21", "right": "192.168.10.22"},
        )
        self.assertEqual(saved.parent, new_dir)

    def test_force_reconfigure_does_not_rewrite_matching_realsense_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "realsense.json"
            path.write_text(
                json.dumps(
                    {
                        "front": {"serial": "OLD_FRONT", "width": 640, "height": 480, "fps": 30},
                        "left_wrist": {"serial": "OLD_LEFT", "width": 640, "height": 480, "fps": 30},
                    }
                ),
                encoding="utf-8",
            )
            result = device_preflight.DevicePreflightResult(ok=True)
            options = device_preflight.DevicePreflightOptions(
                active_sides=("left", "right"),
                camera_config=path,
                assume_yes=True,
                force_reconfigure=True,
            )

            device_preflight.prepare_cameras(
                options,
                result,
                discover=lambda: [
                    {"index": 0, "serial": "OLD_FRONT", "name": "front"},
                    {"index": 1, "serial": "OLD_LEFT", "name": "left"},
                ],
            )

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["front"]["serial"], "OLD_FRONT")
        self.assertEqual(payload["left_wrist"]["serial"], "OLD_LEFT")
        self.assertTrue(any("RealSense config ok" in message for message in result.messages))

    def test_real_teleop_parse_loads_missing_arm_ips_from_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "latest.json"
            rm65_arm_ips.save_assignment(
                rm65_arm_ips.build_assignment(
                    left_arm_ip="192.168.10.21",
                    right_arm_ip="192.168.10.22",
                ),
                config_dir=tmp,
            )
            argv = [
                "prog",
                "--no-gui",
                "--active-sides",
                "both",
                "--arm-ip-config",
                str(latest),
            ]
            with patch.object(sys, "argv", argv):
                args = real_teleop.parse_args()

        self.assertEqual(args.left_arm_ip, "192.168.10.21")
        self.assertEqual(args.right_arm_ip, "192.168.10.22")
        self.assertEqual(args.arm_ip_config_loaded_sides, ["left", "right"])

    def test_real_teleop_parse_saves_explicit_arm_ips_to_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "latest.json"
            argv = [
                "prog",
                "--no-gui",
                "--active-sides",
                "both",
                "--arm-ip-config",
                str(latest),
                "--left-arm-ip",
                "192.168.20.21",
                "--right-arm-ip",
                "192.168.20.22",
            ]
            with patch.object(sys, "argv", argv):
                args = real_teleop.parse_args()
            loaded = rm65_arm_ips.load_assignment(str(latest))

        self.assertTrue(str(args.arm_ip_config_saved_path).endswith("_rm65_arm_ip_assignment.json"))
        self.assertEqual(
            rm65_arm_ips.arm_ips_for_sides(loaded),
            {"left": "192.168.20.21", "right": "192.168.20.22"},
        )

    def test_real_teleop_parse_auto_tunes_real_loop_timing_defaults(self) -> None:
        argv = [
            "prog",
            "--no-gui",
            "--active-sides",
            "right",
            "--right-arm-ip",
            "192.168.0.11",
            "--no-arm-ip-save-latest",
            "--hz",
            "15",
            "--arm-control-hz",
            "50",
        ]
        with patch.object(sys, "argv", argv):
            args = real_teleop.parse_args()

        # Impedance outer control runs independently from the 15 Hz sample
        # cadence, so the automatic watchdog threshold reaches its 15 Hz cap.
        self.assertAlmostEqual(args.minimum_control_hz, 15.0)
        self.assertTrue(args.minimum_control_hz_auto)
        # The computed 50/30 margin is below the configured safety floor.
        self.assertAlmostEqual(args.arm_step_dt_max_scale, 3.0)
        self.assertTrue(args.arm_step_dt_max_scale_auto)
        self.assertAlmostEqual(args.dual_arm_temp_max_xyz_step_m, 0.03)
        self.assertAlmostEqual(args.dual_arm_temp_max_rot_step_deg, 8.0)
        self.assertAlmostEqual(args.dual_arm_temp_max_joint_step_deg, 6.0)

    def test_real_teleop_parse_keeps_explicit_real_loop_timing_limits(self) -> None:
        argv = [
            "prog",
            "--no-gui",
            "--active-sides",
            "right",
            "--right-arm-ip",
            "192.168.0.11",
            "--no-arm-ip-save-latest",
            "--hz",
            "15",
            "--arm-control-hz",
            "50",
            "--minimum-control-hz",
            "15",
            "--arm-step-dt-max-scale",
            "2.5",
        ]
        with patch.object(sys, "argv", argv):
            args = real_teleop.parse_args()

        self.assertAlmostEqual(args.minimum_control_hz, 15.0)
        self.assertFalse(args.minimum_control_hz_auto)
        self.assertAlmostEqual(args.arm_step_dt_max_scale, 2.5)
        self.assertFalse(args.arm_step_dt_max_scale_auto)

    def test_real_teleop_gui_parse_runs_preflight_and_loads_ports(self) -> None:
        seen_options: list[device_preflight.DevicePreflightOptions] = []

        def fake_prepare(options: device_preflight.DevicePreflightOptions):
            seen_options.append(options)
            return SimpleNamespace(
                messages=["ok"],
                q6_ports={"left": "/dev/q6-left", "right": "/dev/q6-right"},
                dmtac_assignment={"assignments": {"left": {"gripper": {"port": "/dev/gripper-left"}}}},
                permission_fixes=[],
            )

        argv = [
            "prog",
            "--gui",
            "--active-sides",
            "both",
            "--left-arm-ip",
            "192.168.0.10",
            "--right-arm-ip",
            "192.168.0.11",
            "--no-arm-ip-save-latest",
            "--hand-backend",
            "dmtac-gripper",
            "--no-camera",
            "--device-preflight-yes",
            "--device-preflight-probe-all-serial",
            "--device-preflight-force-reconfigure",
        ]
        with patch.object(sys, "argv", argv), patch.object(real_teleop, "prepare_devices_for_teleop", fake_prepare):
            args = real_teleop.parse_args()

        self.assertEqual(args.left_hand_port, "/dev/q6-left")
        self.assertEqual(args.right_hand_port, "/dev/q6-right")
        self.assertIsNone(args.dmtac_tactile)
        self.assertEqual(len(seen_options), 1)
        self.assertEqual(args.gripper_input_source, "angle_trigger")
        self.assertFalse(seen_options[0].require_q6)
        self.assertTrue(seen_options[0].require_dmtac_gripper)
        self.assertFalse(seen_options[0].require_dmtac_tactile)
        self.assertTrue(seen_options[0].probe_all_serial)
        self.assertTrue(seen_options[0].force_reconfigure)
        self.assertEqual(
            args._dmtac_preflight_assignment["assignments"]["left"]["gripper"]["port"],
            "/dev/gripper-left",
        )

    def test_real_teleop_gui_parse_requires_dmtac_tactile_when_explicitly_enabled(self) -> None:
        seen_options: list[device_preflight.DevicePreflightOptions] = []

        def fake_prepare(options: device_preflight.DevicePreflightOptions):
            seen_options.append(options)
            return SimpleNamespace(
                messages=["ok"],
                q6_ports={"left": "/dev/q6-left", "right": "/dev/q6-right"},
                permission_fixes=[],
            )

        argv = [
            "prog",
            "--gui",
            "--active-sides",
            "both",
            "--left-arm-ip",
            "192.168.0.10",
            "--right-arm-ip",
            "192.168.0.11",
            "--no-arm-ip-save-latest",
            "--hand-backend",
            "dmtac-gripper",
            "--dmtac-with-tactile",
            "--no-camera",
            "--device-preflight-yes",
        ]
        with patch.object(sys, "argv", argv), patch.object(real_teleop, "prepare_devices_for_teleop", fake_prepare):
            args = real_teleop.parse_args()

        self.assertTrue(args.dmtac_tactile)
        self.assertEqual(len(seen_options), 1)
        self.assertTrue(seen_options[0].require_dmtac_tactile)

    def test_real_teleop_missing_arm_ip_fails_before_device_preflight(self) -> None:
        argv = [
            "prog",
            "--gui",
            "--active-sides",
            "both",
            "--no-arm-ip-default-config",
            "--hand-backend",
            "dmtac-gripper",
            "--no-camera",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(real_teleop, "prepare_devices_for_teleop") as mocked_prepare,
            self.assertRaises(SystemExit) as raised,
        ):
            real_teleop.parse_args()

        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(mocked_prepare.call_count, 0)


if __name__ == "__main__":
    unittest.main()
