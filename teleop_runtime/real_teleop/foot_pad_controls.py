"""GUI-side input adapter for the virtual foot-pad keyboard.

The hardware service remains an independently managed process.  It exposes a
virtual ``DexTeleop foot pads`` keyboard that advertises F13/F14 for two pads
or F13--F16/F17 for four/five pads; this module only checks that keyboard and
translates press edges into existing GUI actions.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


SPECIAL_ARM_CONTROL_MODE_MANUAL = "manual"
SPECIAL_ARM_CONTROL_MODE_FOOT_PAD = "foot-pad"
SPECIAL_ARM_CONTROL_MODE_CHOICES = (
    SPECIAL_ARM_CONTROL_MODE_MANUAL,
    SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
)

FOOT_PAD_DEVICE_NAME = "DexTeleop foot pads"
FOOT_PAD_START_COMMAND = "python -m hardware.foot_pad --count 4"
FOOT_PAD_FIVE_START_COMMAND = "python -m hardware.foot_pad --count 5"
FOOT_PAD_TEST_RECEIPT_SCHEMA = "dexteleop.foot_pad_test_receipt.v1"
FOOT_PAD_TEST_MAPPING_REVISION = "real_teleop_five_pedal_mapping.v1"
FOOT_PAD_TEST_RECEIPT_NAME = "five_pedal_test_receipt.json"
FOOT_PAD_SUPPORTED_COUNTS = frozenset((2, 4, 5))
# The five-pedal layout keeps the two-pedal side-switch contract for its
# first two pedals and adds progress actions on pedals 3/4/5.
FOOT_PAD_TWO_PEDAL_STYLE_COUNTS = frozenset((2, 5))
FOOT_PAD_REQUIRED_KEY_CODES = frozenset((183, 184, 185, 186))  # F13--F16
FOOT_PAD_FIVE_REQUIRED_KEY_CODES = frozenset(
    (*FOOT_PAD_REQUIRED_KEY_CODES, 187)
)  # F13--F17
FOOT_PAD_REQUIRED_KEY_CODES_BY_COUNT = {
    2: frozenset((183, 184)),
    4: FOOT_PAD_REQUIRED_KEY_CODES,
    5: FOOT_PAD_FIVE_REQUIRED_KEY_CODES,
}
FOOT_PAD_EVENT_BY_KEY_CODE = {
    183: "fix_arm_left",
    184: "fix_arm_right",
    185: "init_arm_left",
    186: "init_arm_right",
}
FOOT_PAD_FIVE_EVENT_BY_KEY_CODE = {
    **FOOT_PAD_EVENT_BY_KEY_CODE,
    187: "progress_undo",
}
FOOT_PAD_FIVE_PROGRESS_EVENTS = frozenset(
    ("init_arm_left", "init_arm_right", "progress_undo")
)
FOOT_PAD_EVENT_BY_KEY_CODE_BY_COUNT = {
    2: {183: "fix_arm_left", 184: "fix_arm_right"},
    4: FOOT_PAD_EVENT_BY_KEY_CODE,
    5: FOOT_PAD_FIVE_EVENT_BY_KEY_CODE,
}


class FootPadUnavailableError(RuntimeError):
    """The independently managed virtual foot-pad keyboard is absent."""


def _input_permission_hint() -> str:
    event_nodes = sorted(Path("/dev/input").glob("event*"))
    uinput = Path("/dev/uinput")
    event_unreadable = bool(event_nodes) and not any(
        os.access(path, os.R_OK) for path in event_nodes
    )
    uinput_unwritable = uinput.exists() and not os.access(uinput, os.W_OK)
    if not (event_unreadable or uinput_unwritable):
        return ""
    python = sys.executable or "python"
    return (
        "physical input/uinput nodes exist but are not accessible to this user; "
        "add the user to the input group or run: "
        f"sudo -E {python} -m teleop_runtime.configure_devices --fix-permissions"
    )


def foot_pad_count_from_key_codes(key_codes: Any) -> int | None:
    """Infer the active topology from the daemon's virtual-key capabilities.

    The service creates a virtual keyboard whose advertised F13--F17 set is
    the authoritative *live* topology.  Looking at the capabilities instead
    of only the saved calibration file lets teleop recover when a two-pad
    daemon was started after argument parsing, or when the saved file is
    stale/missing.  Larger supported layouts are checked first because their
    key sets contain the smaller subsets.
    """

    try:
        available = {int(code) for code in key_codes}
    except (TypeError, ValueError):
        return None
    for count in (5, 4, 2):
        if FOOT_PAD_REQUIRED_KEY_CODES_BY_COUNT[count].issubset(available):
            return count
    return None


def resolve_special_arm_control_mode(
    requested_mode: str | None,
    *,
    fix_arm_enabled: bool,
    init_arm_enabled: bool,
    foot_pad_count: int | None = None,
    progress_enabled: bool = False,
    rule_task_chain_foot_pad_enabled: bool = False,
) -> str:
    """Resolve the mode for a configured 2-, 4-, or 5-pedal topology.

    Four pedals retain the original fix-left/fix-right/init-left/init-right
    mapping. With exactly two configured pedals, the pedals emit only
    fix-left/fix-right. Five pedals use the same two-pedal side-switch contract
    for F13/F14 and add F15--F17 progress events. Their progress events are
    admitted or rejected later by the progress state machine; the pedal
    topology itself must not depend on whether an operator has selected a
    profile yet. The rule task-chain mode is an explicit narrow exception:
    it only admits a live two-pedal topology and leaves the actual task-chain
    routing to the dedicated rule UI. This function selects the input adapter;
    it does not enable or disable the Fix/Init controllers or their GUI and
    keyboard controls. Any other known count selects manual mode so the caller
    can continue without foot-pad hardware.
    """

    requested = str(requested_mode or "").strip()
    if requested and requested not in SPECIAL_ARM_CONTROL_MODE_CHOICES:
        raise ValueError(f"Unsupported special-arm control mode {requested!r}.")
    count = None if foot_pad_count is None else int(foot_pad_count)
    if count is not None and count not in FOOT_PAD_SUPPORTED_COUNTS:
        return SPECIAL_ARM_CONTROL_MODE_MANUAL
    # Retain this compatibility argument for callers that use a progress gate
    # during argument resolution. A five-pedal device is still a valid input
    # before profile selection: F13/F14 are ordinary Fix actions, while
    # F15--F17 are gated by the runtime progress state below.
    del progress_enabled
    if requested:
        mode = requested
    elif rule_task_chain_foot_pad_enabled:
        mode = SPECIAL_ARM_CONTROL_MODE_FOOT_PAD
    elif count in FOOT_PAD_TWO_PEDAL_STYLE_COUNTS:
        mode = (
            SPECIAL_ARM_CONTROL_MODE_FOOT_PAD
            if fix_arm_enabled
            else SPECIAL_ARM_CONTROL_MODE_MANUAL
        )
    else:
        mode = (
            SPECIAL_ARM_CONTROL_MODE_FOOT_PAD
            if fix_arm_enabled and init_arm_enabled
            else SPECIAL_ARM_CONTROL_MODE_MANUAL
        )
    if mode == SPECIAL_ARM_CONTROL_MODE_FOOT_PAD:
        if rule_task_chain_foot_pad_enabled:
            if count not in (None, 2):
                raise ValueError(
                    "rule task-chain foot-pad mode requires exactly 2 pedals."
                )
        elif count in FOOT_PAD_TWO_PEDAL_STYLE_COUNTS:
            if not fix_arm_enabled:
                count_label = "two" if count == 2 else "five"
                raise ValueError(
                    f"--special-arm-control-mode foot-pad with {count_label} pedals "
                    "requires --fix-arm."
                )
        elif not (fix_arm_enabled and init_arm_enabled):
            raise ValueError(
                "--special-arm-control-mode foot-pad requires both --fix-arm "
                "and --init-arm-available."
            )
    return mode


def apply_live_foot_pad_topology(args: Any, foot_pad_count: int) -> str:
    """Apply the topology discovered from the live virtual keyboard.

    CLI resolution necessarily runs before the independently managed foot-pad
    daemon is opened, so its saved calibration count is only a provisional
    hint.  Once evdev reports the live key set, keep the runtime arguments and
    the input adapter on the same topology.  A missing
    ``special_arm_control_mode_requested`` attribute is supported for the
    small compatibility/test namespaces that construct arguments directly.
    """

    count = int(foot_pad_count)
    if count not in FOOT_PAD_SUPPORTED_COUNTS:
        raise ValueError(
            f"live foot-pad topology must be 2, 4, or 5 pedals, got {count}"
        )
    if hasattr(args, "special_arm_control_mode_requested"):
        requested_mode = getattr(
            args,
            "special_arm_control_mode_requested",
            None,
        )
    else:
        requested_mode = getattr(
            args,
            "special_arm_control_mode",
            None,
        )
    mode = resolve_special_arm_control_mode(
        requested_mode,
        fix_arm_enabled=bool(getattr(args, "fix_arm", False)),
        init_arm_enabled=bool(getattr(args, "init_arm_available", False)),
        foot_pad_count=count,
        progress_enabled=bool(
            getattr(args, "collection_progress_profile", None) is not None
            or getattr(args, "collection_progress_gui", False)
        ),
        rule_task_chain_foot_pad_enabled=bool(
            getattr(args, "rule_task_chain_foot_pad", False)
        ),
    )
    args.foot_pad_config_count = count
    args.special_arm_control_mode = mode
    return mode


def foot_pad_progress_topology_guidance(
    foot_pad_count: int,
    *,
    progress_enabled: bool,
) -> str:
    """Explain how to replace a live legacy topology for progress capture."""

    count = int(foot_pad_count)
    if not bool(progress_enabled) or count == 5:
        return ""
    if count not in FOOT_PAD_SUPPORTED_COUNTS:
        return ""
    return (
        f"当前 live 虚拟键盘为 {count} 踏板；如需五踏板长程进度，请先停止旧脚踏服务，"
        f"执行 {FOOT_PAD_FIVE_START_COMMAND}，再点击“重连失败设备”。"
        "运行时不会把旧虚拟键盘热升级为五踏板。"
    )


def initialize_startup_foot_pad_input(
    args: Any,
    *,
    foot_pad_control_mode: str,
    input_type: type,
    unavailable_error_type: type[Exception],
    startup_cleanup: Any,
    startup_progress: Any | None,
    control_ui: Any | None,
    apply_live_topology: Callable[[Any, int], str],
    topology_guidance: Callable[..., str],
    restore_init_after_fallback: Callable[..., tuple[bool, str]],
    log: Callable[..., None],
) -> Any | None:
    """Open the optional foot-pad input or install its retryable fallback.

    This is the sole initial-startup transaction for the independently
    managed virtual keyboard.  Manual retry remains owned by
    ``device_retry_runtime``.
    """

    requested = (
        str(getattr(args, "special_arm_control_mode", ""))
        == str(foot_pad_control_mode)
    )
    args.foot_pad_requested = bool(requested)
    args.foot_pad_startup_error = ""
    foot_pad_input = None
    if requested:
        if startup_progress is not None:
            startup_progress.begin(
                "foot_pad",
                label="脚踏输入",
                message="正在查找虚拟脚踏设备",
                requirement="optional_input",
            )
        candidate = None
        try:
            candidate = input_type.open_online()
            live_count = int(getattr(candidate, "foot_pad_count", 0) or 0)
            live_mode = apply_live_topology(args, live_count)
            foot_pad_input = candidate
            log(
                "FootPad",
                "live virtual keyboard topology resolved: "
                f"count={live_count} mode={live_mode}",
            )
            guidance = topology_guidance(
                live_count,
                progress_enabled=bool(
                    getattr(args, "collection_progress_profile", None) is not None
                    or getattr(args, "collection_progress_gui", False)
                ),
            )
            if guidance:
                log("FootPad", guidance)
                if control_ui is not None:
                    control_ui.append_message("FootPad", guidance)
        except (unavailable_error_type, OSError, RuntimeError, ValueError) as exc:
            if candidate is not None:
                try:
                    candidate.close()
                except Exception:
                    pass
            args.foot_pad_startup_error = f"{type(exc).__name__}: {exc}"
            foot_pad_input = input_type.waiting_for_service()
            startup_cleanup.add("foot_pad", foot_pad_input.close)
            if startup_progress is not None:
                startup_progress.fail(
                    "foot_pad",
                    args.foot_pad_startup_error,
                    message="未就绪；点击“重连失败设备”后才会重新扫描",
                )
            log(
                "FootPad",
                "foot-pad input is unavailable; GUI/keyboard special-arm controls "
                "remain available; press 'retry failed devices' in the GUI to "
                "scan for the virtual keyboard again: "
                f"{exc}",
            )
        else:
            startup_cleanup.add("foot_pad", foot_pad_input.close)
            if startup_progress is not None:
                startup_progress.succeed("foot_pad", message="脚踏输入已连接")

    _init_restored, init_restore_message = restore_init_after_fallback(
        args,
        live_foot_pad_count=(
            None
            if foot_pad_input is None
            or not bool(getattr(foot_pad_input, "is_online", False))
            else int(getattr(foot_pad_input, "foot_pad_count", 0) or 0)
        ),
    )
    if init_restore_message:
        log("InitArm", init_restore_message)
        if control_ui is not None:
            control_ui.append_message("InitArm", init_restore_message)
    return foot_pad_input


def _require_evdev() -> Any:
    try:
        import evdev
    except ImportError as exc:  # pragma: no cover - depends on the host environment.
        raise FootPadUnavailableError(
            "Python package 'evdev' is unavailable; install "
            "hardware/foot_pad/requirements.txt before starting the foot-pad service."
        ) from exc
    return evdev


def _load_foot_pad_config() -> tuple[Any, int]:
    """Load the service config and return its path and configured count."""

    try:
        from hardware.foot_pad.service import default_config_path, load_config

        config_path = default_config_path()
        config = load_config(config_path)
    except Exception as exc:
        raise FootPadUnavailableError("a valid foot-pad config is unavailable") from exc
    try:
        count = int(config.get("count", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise FootPadUnavailableError("foot-pad config has an invalid pedal count") from exc
    return config_path, count


def configured_foot_pad_count() -> int | None:
    """Return the saved count without touching evdev or starting hardware."""

    try:
        _config_path, count = _load_foot_pad_config()
    except FootPadUnavailableError:
        return None
    return count


def _foot_pad_assignment_fingerprint(config: Any) -> str:
    """Return a stable identity for the tested physical-to-virtual mapping.

    ``created_at`` is intentionally excluded: rewriting an identical assignment
    must not make a successfully tested five-pedal mapping require another
    operator test.  Device identity, source key, slot order, virtual output, and
    the RealTeleop semantic mapping revision are all included, so any meaningful
    remap or future semantic change invalidates the receipt fail-closed.
    """

    if not isinstance(config, dict):
        raise ValueError("foot-pad assignment must be a JSON object")
    pedals = config.get("pedals")
    if not isinstance(pedals, list):
        raise ValueError("foot-pad assignment pedals must be a list")
    canonical_pedals: list[dict[str, object]] = []
    for raw in pedals:
        if not isinstance(raw, dict):
            raise ValueError("foot-pad assignment pedal must be an object")
        source = raw.get("source")
        output = raw.get("output")
        if not isinstance(source, dict) or not isinstance(output, dict):
            raise ValueError("foot-pad assignment source/output must be objects")
        canonical_pedals.append(
            {
                "slot": int(raw.get("slot", 0) or 0),
                "source": {
                    "id_path": str(source.get("id_path", "")),
                    "vendor_id": str(source.get("vendor_id", "")),
                    "model_id": str(source.get("model_id", "")),
                    "name": str(source.get("name", "")),
                    "key_code": int(source.get("key_code", 0) or 0),
                },
                "output": {
                    "key_name": str(output.get("key_name", "")),
                    "key_code": int(output.get("key_code", 0) or 0),
                },
            }
        )
    payload = {
        "assignment_schema_version": str(config.get("schema_version", "")),
        "count": int(config.get("count", 0) or 0),
        "mapping_revision": FOOT_PAD_TEST_MAPPING_REVISION,
        "pedals": sorted(canonical_pedals, key=lambda item: int(item["slot"])),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _foot_pad_test_receipt_path(config_path: Path) -> Path:
    return Path(config_path).with_name(FOOT_PAD_TEST_RECEIPT_NAME)


def persisted_five_pedal_test_verified() -> bool:
    """Whether the current saved five-pedal assignment has a valid test receipt."""

    try:
        config_path, count = _load_foot_pad_config()
        if int(count) != 5:
            return False
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        expected_fingerprint = _foot_pad_assignment_fingerprint(config)
        receipt = json.loads(
            _foot_pad_test_receipt_path(Path(config_path)).read_text(encoding="utf-8")
        )
        return bool(
            isinstance(receipt, dict)
            and receipt.get("schema_version") == FOOT_PAD_TEST_RECEIPT_SCHEMA
            and receipt.get("mapping_revision") == FOOT_PAD_TEST_MAPPING_REVISION
            and int(receipt.get("foot_pad_count", 0) or 0) == 5
            and receipt.get("assignment_fingerprint") == expected_fingerprint
            and tuple(receipt.get("verified_output_key_codes", ()))
            == tuple(sorted(FOOT_PAD_FIVE_REQUIRED_KEY_CODES))
        )
    except (FootPadUnavailableError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def persist_five_pedal_test_receipt() -> Path:
    """Atomically persist successful verification of the current five-pad map."""

    config_path, count = _load_foot_pad_config()
    if int(count) != 5:
        raise ValueError(
            "five-pedal test receipt requires a saved five-pedal assignment"
        )
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    receipt_path = _foot_pad_test_receipt_path(config_path)
    receipt = {
        "schema_version": FOOT_PAD_TEST_RECEIPT_SCHEMA,
        "mapping_revision": FOOT_PAD_TEST_MAPPING_REVISION,
        "foot_pad_count": 5,
        "assignment_fingerprint": _foot_pad_assignment_fingerprint(config),
        "verified_output_key_codes": sorted(FOOT_PAD_FIVE_REQUIRED_KEY_CODES),
        "verified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=receipt_path.parent,
            delete=False,
        ) as handle:
            json.dump(receipt, handle, indent=2, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, receipt_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return receipt_path


def _sysfs_input_device_name(event_path: str | os.PathLike[str]) -> str:
    """Read an event node's kernel name without opening the protected node."""

    event_name = Path(event_path).name
    try:
        return (
            Path("/sys/class/input")
            .joinpath(event_name, "device", "name")
            .read_text(encoding="utf-8")
            .strip()
        )
    except OSError:
        return ""


class FootPadSpecialArmInput:
    """Read F13--F17 press edges from the foot-pad service's virtual device."""

    def __init__(self, device: Any, evdev: Any, *, foot_pad_count: int = 4) -> None:
        count = int(foot_pad_count)
        if count not in FOOT_PAD_SUPPORTED_COUNTS:
            raise ValueError(f"foot_pad_count must be 2, 4, or 5, got {count}")
        self._device = device
        self._evdev = evdev
        self.foot_pad_count = count
        self._event_by_key_code = FOOT_PAD_EVENT_BY_KEY_CODE_BY_COUNT[count]
        self._closed = False

    @classmethod
    def waiting_for_service(
        cls,
        *,
        foot_pad_count: int | None = None,
    ) -> "FootPadSpecialArmInput":
        """Create an offline adapter that never scans from the poll path.

        The GUI keeps this object so foot-pad status stays visible.  Rebinding
        is intentionally explicit: only the user's "retry failed devices"
        action calls :meth:`open_online`.
        """

        count = (
            configured_foot_pad_count()
            if foot_pad_count is None
            else int(foot_pad_count)
        )
        if count not in FOOT_PAD_SUPPORTED_COUNTS:
            count = 4
        return cls(None, None, foot_pad_count=int(count))

    @classmethod
    def open_online(cls) -> "FootPadSpecialArmInput":
        # A saved assignment is useful for diagnostics and for starting the
        # forwarding daemon, but the live virtual keyboard is authoritative
        # once it exists.  In particular, a two-pad daemon must be accepted
        # even when parsing happened before its config was written or while an
        # old four-pad config is still present.
        configured_count = configured_foot_pad_count()
        evdev = _require_evdev()
        matches: list[tuple[Any, int]] = []
        inaccessible_matches: list[tuple[str, BaseException]] = []
        event_paths = list(evdev.list_devices())
        for event_path in event_paths:
            # Most hosts expose many event nodes, and constructing an evdev
            # InputDevice may perform several blocking ioctls.  The kernel
            # name is available through sysfs without opening the device, so
            # reject known non-foot-pad nodes before the expensive probe.
            sysfs_name = _sysfs_input_device_name(event_path)
            if sysfs_name and sysfs_name != FOOT_PAD_DEVICE_NAME:
                continue
            try:
                device = evdev.InputDevice(event_path)
                if str(device.name or "") != FOOT_PAD_DEVICE_NAME:
                    device.close()
                    continue
                capabilities = device.capabilities(absinfo=False)
                key_codes = {
                    int(code)
                    for code in capabilities.get(evdev.ecodes.EV_KEY, ())
                }
                live_count = foot_pad_count_from_key_codes(key_codes)
                if live_count is not None:
                    matches.append((device, live_count))
                else:
                    device.close()
            except PermissionError as exc:
                if _sysfs_input_device_name(event_path) == FOOT_PAD_DEVICE_NAME:
                    inaccessible_matches.append((str(event_path), exc))
                continue
            except (OSError, ValueError):
                continue
        if not matches:
            if inaccessible_matches:
                blocked = ", ".join(path for path, _exc in inaccessible_matches)
                python = sys.executable or "python"
                raise FootPadUnavailableError(
                    f"virtual keyboard {FOOT_PAD_DEVICE_NAME!r} exists at {blocked}, "
                    "but the current user cannot read its event node; run while the "
                    "foot-pad service is active: "
                    f"sudo -E {python} -m teleop_runtime.configure_devices "
                    "--fix-permissions"
                ) from inaccessible_matches[0][1]
            permission_hint = _input_permission_hint()
            if permission_hint:
                raise FootPadUnavailableError(permission_hint)
            if configured_count in FOOT_PAD_SUPPORTED_COUNTS:
                configured_hint = f"configured {configured_count}-pedal"
                remediation = f"python -m hardware.foot_pad --count {configured_count}"
                if configured_count != 5:
                    remediation += (
                        f"; five-pedal progress: {FOOT_PAD_FIVE_START_COMMAND}"
                    )
            else:
                configured_hint = "no saved 2/4/5-pedal config"
                remediation = (
                    f"{FOOT_PAD_FIVE_START_COMMAND} --recalibrate for five-pedal "
                    "progress, or choose --count 2 or --count 4 with "
                    "--recalibrate for a legacy topology"
                )
            raise FootPadUnavailableError(
                f"virtual keyboard {FOOT_PAD_DEVICE_NAME!r} with a supported "
                f"2/4/5-pedal key set was not found ({configured_hint}); "
                f"start/calibrate it with: {remediation}"
            )
        # Prefer the largest supported match if a stale duplicate UInput device is
        # present.  ``matches`` is non-empty and every entry has a supported
        # live count, so this also handles a missing/stale saved config.
        selected_device, detected_count = max(matches, key=lambda item: item[1])
        for device, _count in matches:
            if device is not selected_device:
                device.close()
        return cls(selected_device, evdev, foot_pad_count=detected_count)

    @classmethod
    def wait_until_online(
        cls,
        *,
        retry_sec: float = 1.0,
        output: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "FootPadSpecialArmInput":
        """Block before real-teleop construction until the service is healthy."""

        stream = sys.stderr if output is None else output
        last_message = ""
        while True:
            try:
                source = cls.open_online()
            except FootPadUnavailableError as exc:
                message = str(exc)
                if message != last_message:
                    print("[FootPad] foot-pad mode is waiting for its virtual keyboard.", file=stream)
                    print(f"[FootPad] {message}", file=stream)
                    configured_count = configured_foot_pad_count()
                    start_command = (
                        f"python -m hardware.foot_pad --count {configured_count}"
                        if configured_count in FOOT_PAD_SUPPORTED_COUNTS
                        else FOOT_PAD_START_COMMAND
                    )
                    print(f"[FootPad] Start it in another terminal: {start_command}", file=stream)
                    stream.flush()
                    last_message = message
                sleep(max(0.1, float(retry_sec)))
                continue
            if source.foot_pad_count == 2:
                mapping = (
                    "F13=left-arm-only (fix right), "
                    "F14=right-arm-only (fix left)"
                )
            elif source.foot_pad_count == 5:
                mapping = (
                    "five-pedal layout: F13=left-arm-only (fix right), "
                    "F14=right-arm-only (fix left); "
                    "F15=advance, F16=failure, F17=undo when progress is ready"
                )
            else:
                mapping = (
                    "F13=fix-left, F14=fix-right, "
                    "F15=init-left, F16=init-right"
                )
            print(f"[FootPad] online: {FOOT_PAD_DEVICE_NAME} ({mapping}).", file=stream)
            stream.flush()
            return source

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._drop_device()

    @property
    def is_online(self) -> bool:
        """Whether a readable virtual keyboard is currently bound.

        The offline adapter intentionally survives a device disconnect so the
        GUI can keep showing its failed status until a manual retry replaces
        it. Availability must therefore use this property rather than checking
        whether the adapter object itself is non-null.
        """

        return bool(not self._closed and self._device is not None)

    def _drop_device(self, *, expected_device: Any = None) -> None:
        if expected_device is not None and self._device is not expected_device:
            return
        device = self._device
        self._device = None
        if device is None:
            return
        try:
            device.close()
        except OSError:
            pass

    def consume_events(self) -> list[str]:
        """Return new pedal press edges; releases and key-repeat events are ignored."""

        return self._consume_events(self._event_by_key_code)

    def consume_test_events(self) -> list[str]:
        """Return all F13--F17 edges for the diagnostic pedal test only.

        This does not alter the configured two-pedal control map.  In
        particular, F15/F16 observed during the test are labels only and can
        never reach the real Fix/Init routing path.
        """

        return self._consume_events(FOOT_PAD_FIVE_EVENT_BY_KEY_CODE)

    def _consume_events(self, event_by_key_code: dict[int, str]) -> list[str]:
        if self._closed or self._device is None or self._evdev is None:
            return []
        device = self._device
        evdev = self._evdev
        try:
            # evdev.read() returns a lazy iterator on some versions.  EAGAIN
            # may therefore be raised by iteration rather than by read()
            # itself; materialize it inside the protected region.
            events = list(device.read())
        except BlockingIOError:
            return []
        except OSError:
            # The daemon recreates its UInput device after a physical pad is
            # unplugged. Drop only this stale fd and remain offline; the GUI's
            # explicit retry action is the only path that scans for a replacement.
            self._drop_device(expected_device=device)
            return []
        event_type = int(evdev.ecodes.EV_KEY)
        mapped: list[str] = []
        for event in events:
            if int(getattr(event, "type", -1)) != event_type:
                continue
            if int(getattr(event, "value", 0)) != 1:
                continue
            action = event_by_key_code.get(int(getattr(event, "code", -1)))
            if action:
                mapped.append(action)
        return mapped


__all__ = [
    "FOOT_PAD_DEVICE_NAME",
    "FOOT_PAD_EVENT_BY_KEY_CODE",
    "FOOT_PAD_EVENT_BY_KEY_CODE_BY_COUNT",
    "FOOT_PAD_FIVE_EVENT_BY_KEY_CODE",
    "FOOT_PAD_FIVE_PROGRESS_EVENTS",
    "FOOT_PAD_FIVE_REQUIRED_KEY_CODES",
    "FOOT_PAD_FIVE_START_COMMAND",
    "FOOT_PAD_REQUIRED_KEY_CODES",
    "FOOT_PAD_REQUIRED_KEY_CODES_BY_COUNT",
    "FOOT_PAD_START_COMMAND",
    "FOOT_PAD_SUPPORTED_COUNTS",
    "FOOT_PAD_TEST_MAPPING_REVISION",
    "FOOT_PAD_TEST_RECEIPT_NAME",
    "FOOT_PAD_TEST_RECEIPT_SCHEMA",
    "FOOT_PAD_TWO_PEDAL_STYLE_COUNTS",
    "FootPadSpecialArmInput",
    "FootPadUnavailableError",
    "SPECIAL_ARM_CONTROL_MODE_CHOICES",
    "SPECIAL_ARM_CONTROL_MODE_FOOT_PAD",
    "SPECIAL_ARM_CONTROL_MODE_MANUAL",
    "configured_foot_pad_count",
    "foot_pad_count_from_key_codes",
    "foot_pad_progress_topology_guidance",
    "initialize_startup_foot_pad_input",
    "persist_five_pedal_test_receipt",
    "persisted_five_pedal_test_verified",
    "apply_live_foot_pad_topology",
    "resolve_special_arm_control_mode",
]
