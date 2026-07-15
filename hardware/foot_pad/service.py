from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "foot_pad_assignment.v1"
LATEST_CONFIG_NAME = "latest.json"
MAX_PEDALS = 4
# Linux input-event key codes for F13 through F16.  They remain available for
# configuration validation without importing the optional evdev dependency.
SPECIAL_KEYS: tuple[tuple[str, int], ...] = (
    ("F13", 183),
    ("F14", 184),
    ("F15", 185),
    ("F16", 186),
)
IDENTITY_FIELDS = ("id_path", "vendor_id", "model_id", "name")


class FootPadError(RuntimeError):
    """A recoverable configuration or input-device error."""


@dataclass(frozen=True)
class CandidateDevice:
    event_path: str
    id_path: str
    name: str
    vendor_id: str
    model_id: str
    key_codes: frozenset[int]

    def identity(self) -> dict[str, str]:
        return {
            "id_path": self.id_path,
            "name": self.name,
            "vendor_id": self.vendor_id,
            "model_id": self.model_id,
        }


@dataclass(frozen=True)
class CalibrationHit:
    device: CandidateDevice
    source_key_code: int
    source_key_name: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    try:
        root = repo_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from project_paths import hardware_interface_configs_root

        return hardware_interface_configs_root() / "foot_pad" / LATEST_CONFIG_NAME
    except Exception:
        return repo_root() / "teleop_runtime" / "configs" / "hardware" / "foot_pad" / LATEST_CONFIG_NAME


def iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _require_evdev() -> Any:
    try:
        import evdev
    except ImportError as exc:
        raise FootPadError(
            "Missing dependency 'evdev'. Install it with "
            "python -m pip install -r hardware/foot_pad/requirements.txt."
        ) from exc
    return evdev


def _udev_properties(event_path: str) -> dict[str, str]:
    command = ["udevadm", "info", "--query=property", f"--name={event_path}"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        return {}
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return properties


def discover_key_devices() -> list[CandidateDevice]:
    """Return event devices capable of emitting key events.

    The event node is intentionally not treated as device identity because its
    number changes across boots and USB reconnections.  ID_PATH includes the
    USB topology and interface, which separates otherwise identical foot pads.
    """

    evdev = _require_evdev()
    candidates: list[CandidateDevice] = []
    for event_path in sorted(evdev.list_devices()):
        try:
            device = evdev.InputDevice(event_path)
            capabilities = device.capabilities(absinfo=False)
            key_codes = frozenset(int(code) for code in capabilities.get(evdev.ecodes.EV_KEY, []))
            if not key_codes:
                device.close()
                continue
            properties = _udev_properties(event_path)
            id_path = properties.get("ID_PATH", "").strip()
            if not id_path:
                device.close()
                continue
            candidates.append(
                CandidateDevice(
                    event_path=event_path,
                    id_path=id_path,
                    name=str(device.name or properties.get("NAME", "")).strip(),
                    vendor_id=properties.get("ID_VENDOR_ID", "").strip().lower(),
                    model_id=properties.get("ID_MODEL_ID", "").strip().lower(),
                    key_codes=key_codes,
                )
            )
            device.close()
        except (OSError, ValueError):
            continue
    return candidates


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(dict(payload), handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FootPadError(f"Cannot read foot-pad config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FootPadError(f"Foot-pad config {path} must contain a JSON object.")
    return payload


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise FootPadError(f"{field} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FootPadError(f"{field} must be an integer.") from exc


def normalize_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != SCHEMA_VERSION:
        raise FootPadError(
            f"Unsupported foot-pad config schema {schema_version!r}; expected {SCHEMA_VERSION!r}."
        )
    count = _integer(payload.get("count"), "count")
    if not 1 <= count <= MAX_PEDALS:
        raise FootPadError(f"count must be between 1 and {MAX_PEDALS}.")
    pedals = payload.get("pedals")
    if not isinstance(pedals, list) or len(pedals) != count:
        raise FootPadError("pedals must contain exactly count entries.")

    normalized_pedals: list[dict[str, Any]] = []
    seen_slots: set[int] = set()
    seen_paths: set[str] = set()
    for raw_pedal in pedals:
        if not isinstance(raw_pedal, Mapping):
            raise FootPadError("Each pedal entry must be an object.")
        slot = _integer(raw_pedal.get("slot"), "pedal slot")
        if slot not in range(1, count + 1) or slot in seen_slots:
            raise FootPadError("Pedal slots must be unique and sequential from 1.")
        source = raw_pedal.get("source")
        output = raw_pedal.get("output")
        if not isinstance(source, Mapping) or not isinstance(output, Mapping):
            raise FootPadError("Each pedal must have source and output objects.")
        identity = {field: str(source.get(field, "")).strip() for field in IDENTITY_FIELDS}
        if not all(identity.values()):
            raise FootPadError(f"Pedal {slot} has incomplete source identity.")
        if identity["id_path"] in seen_paths:
            raise FootPadError("Each pedal must refer to a different input device.")
        source_key_code = _integer(source.get("key_code"), f"pedal {slot} source key_code")
        expected_name, expected_code = SPECIAL_KEYS[slot - 1]
        output_key_code = _integer(output.get("key_code"), f"pedal {slot} output key_code")
        if output_key_code != expected_code or str(output.get("key_name", "")).upper() != expected_name:
            raise FootPadError(f"Pedal {slot} must output {expected_name}.")
        normalized_pedals.append(
            {
                "slot": slot,
                "source": {
                    **identity,
                    "key_code": source_key_code,
                    "key_name": str(source.get("key_name", "")).strip() or str(source_key_code),
                },
                "output": {"key_name": expected_name, "key_code": expected_code},
            }
        )
        seen_slots.add(slot)
        seen_paths.add(identity["id_path"])

    normalized_pedals.sort(key=lambda pedal: pedal["slot"])
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": str(payload.get("created_at") or iso_timestamp()),
        "count": count,
        "pedals": normalized_pedals,
    }


def build_config(hits: Sequence[CalibrationHit]) -> dict[str, Any]:
    if not 1 <= len(hits) <= MAX_PEDALS:
        raise FootPadError(f"The number of calibration hits must be between 1 and {MAX_PEDALS}.")
    pedals: list[dict[str, Any]] = []
    for slot, hit in enumerate(hits, start=1):
        key_name, key_code = SPECIAL_KEYS[slot - 1]
        pedals.append(
            {
                "slot": slot,
                "source": {
                    **hit.device.identity(),
                    "key_code": hit.source_key_code,
                    "key_name": hit.source_key_name,
                },
                "output": {"key_name": key_name, "key_code": key_code},
            }
        )
    return normalize_config(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": iso_timestamp(),
            "count": len(hits),
            "pedals": pedals,
        }
    )


def validate_config(config: Mapping[str, Any], candidates: Iterable[CandidateDevice]) -> list[str]:
    """Return validation failures; an empty list means the config is usable."""

    try:
        normalized = normalize_config(config)
    except FootPadError as exc:
        return [str(exc)]
    candidates_by_path = {candidate.id_path: candidate for candidate in candidates}
    errors: list[str] = []
    for pedal in normalized["pedals"]:
        slot = pedal["slot"]
        source = pedal["source"]
        candidate = candidates_by_path.get(source["id_path"])
        if candidate is None:
            errors.append(f"Pedal {slot} is not connected at {source['id_path']}.")
            continue
        for field in ("name", "vendor_id", "model_id"):
            expected = source[field]
            actual = getattr(candidate, field)
            if actual != expected:
                errors.append(
                    f"Pedal {slot} identity mismatch for {field}: expected {expected!r}, got {actual!r}."
                )
        if source["key_code"] not in candidate.key_codes:
            errors.append(f"Pedal {slot} no longer exposes key code {source['key_code']}.")
    return errors


def load_config(path: Path) -> dict[str, Any]:
    return normalize_config(_read_json(path))


def save_config(path: Path, config: Mapping[str, Any]) -> None:
    _write_json(path, normalize_config(config))


def _key_name(evdev: Any, key_code: int) -> str:
    name = evdev.ecodes.KEY.get(key_code)
    return str(name or key_code)


def _open_devices(candidates: Sequence[CandidateDevice], evdev: Any) -> dict[str, Any]:
    opened: dict[str, Any] = {}
    try:
        for candidate in candidates:
            opened[candidate.id_path] = evdev.InputDevice(candidate.event_path)
        return opened
    except Exception:
        for device in opened.values():
            device.close()
        raise


def calibrate(count: int) -> dict[str, Any]:
    evdev = _require_evdev()
    candidates = discover_key_devices()
    if len(candidates) < count:
        raise FootPadError(
            f"Only {len(candidates)} key-capable input device(s) are visible; {count} foot pads are required."
        )
    print(f"Detected {len(candidates)} key-capable input device(s).")
    print("Do not press the regular keyboard during calibration.")
    opened = _open_devices(candidates, evdev)
    hits: list[CalibrationHit] = []
    used_paths: set[str] = set()
    try:
        while len(hits) < count:
            slot = len(hits) + 1
            key_name, _ = SPECIAL_KEYS[slot - 1]
            print(f"Step on foot pad {slot}/{count} now. It will be mapped to {key_name}.", flush=True)
            while True:
                ready, _, _ = select.select(list(opened.values()), [], [])
                duplicate = False
                for device in ready:
                    for event in device.read():
                        if event.type != evdev.ecodes.EV_KEY or event.value != 1:
                            continue
                        candidate = next(item for item in candidates if item.id_path in opened and opened[item.id_path] is device)
                        if candidate.id_path in used_paths:
                            print("That foot pad is already assigned; step on the next foot pad.", flush=True)
                            duplicate = True
                            break
                        hit = CalibrationHit(candidate, int(event.code), _key_name(evdev, int(event.code)))
                        hits.append(hit)
                        used_paths.add(candidate.id_path)
                        print(
                            f"Foot pad {slot} recorded: {candidate.id_path}, "
                            f"source {hit.source_key_name} -> {key_name}.",
                            flush=True,
                        )
                        break
                    if duplicate or len(hits) == slot:
                        break
                if duplicate or len(hits) == slot:
                    break
    except KeyboardInterrupt as exc:
        raise FootPadError("Calibration cancelled; no configuration was saved.") from exc
    finally:
        for device in opened.values():
            device.close()
    return build_config(hits)


def _configured_candidates(config: Mapping[str, Any], candidates: Sequence[CandidateDevice]) -> list[CandidateDevice]:
    by_path = {candidate.id_path: candidate for candidate in candidates}
    return [by_path[pedal["source"]["id_path"]] for pedal in config["pedals"]]


def _release_pressed(ui: Any, evdev: Any, pressed_keys: set[int]) -> None:
    for key_code in pressed_keys:
        ui.write(evdev.ecodes.EV_KEY, key_code, 0)
    if pressed_keys:
        ui.syn()
    pressed_keys.clear()


def _forward_events(config: Mapping[str, Any], devices: Mapping[str, Any], ui: Any, evdev: Any) -> None:
    source_to_output = {
        pedal["source"]["id_path"]: (pedal["source"]["key_code"], pedal["output"]["key_code"])
        for pedal in config["pedals"]
    }
    device_to_path = {device: id_path for id_path, device in devices.items()}
    pressed_keys: set[int] = set()
    try:
        while True:
            ready, _, _ = select.select(list(device_to_path), [], [])
            for device in ready:
                id_path = device_to_path[device]
                source_key, output_key = source_to_output[id_path]
                for event in device.read():
                    if event.type != evdev.ecodes.EV_KEY or event.code != source_key or event.value not in (0, 1, 2):
                        continue
                    ui.write(evdev.ecodes.EV_KEY, output_key, event.value)
                    ui.syn()
                    if event.value == 1:
                        pressed_keys.add(output_key)
                    elif event.value == 0:
                        pressed_keys.discard(output_key)
    finally:
        _release_pressed(ui, evdev, pressed_keys)


def run_daemon(config: Mapping[str, Any]) -> None:
    evdev = _require_evdev()
    print("Foot-pad forwarding started. Press Ctrl-C to stop.")
    while True:
        candidates = discover_key_devices()
        errors = validate_config(config, candidates)
        if errors:
            print("Configured foot pads are unavailable; retrying in 1 second:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            time.sleep(1.0)
            continue
        devices = _open_devices(_configured_candidates(config, candidates), evdev)
        ui = evdev.UInput(
            {evdev.ecodes.EV_KEY: [key_code for _, key_code in SPECIAL_KEYS]},
            name="DexTeleop foot pads",
        )
        grabbed: list[Any] = []
        try:
            for device in devices.values():
                device.grab()
                grabbed.append(device)
            print("All configured foot pads are connected and exclusively grabbed.")
            _forward_events(config, devices, ui, evdev)
        except KeyboardInterrupt:
            return
        except OSError as exc:
            print(f"Foot-pad input disconnected or unavailable: {exc}; reconnecting.", file=sys.stderr)
        finally:
            for device in reversed(grabbed):
                try:
                    device.ungrab()
                except OSError:
                    pass
            for device in devices.values():
                device.close()
            ui.close()
        time.sleep(1.0)


def _prompt_count() -> int:
    if not sys.stdin.isatty():
        raise FootPadError("No valid config exists; pass --count N to calibrate non-interactively.")
    while True:
        answer = input(f"How many foot pads should be used (1-{MAX_PEDALS})? ").strip()
        try:
            count = int(answer)
        except ValueError:
            print("Enter an integer.")
            continue
        if 1 <= count <= MAX_PEDALS:
            return count
        print(f"Enter a value from 1 to {MAX_PEDALS}.")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate USB foot pads and forward them as virtual F13-F16 keys."
    )
    parser.add_argument("--count", type=int, choices=range(1, MAX_PEDALS + 1), help="Number of foot pads to use.")
    parser.add_argument("--recalibrate", action="store_true", help="Ignore the stored mapping and calibrate again.")
    parser.add_argument("--config", type=Path, help="Config file path (default: project hardware config).")
    parser.add_argument("--check", action="store_true", help="Validate the stored config and exit.")
    parser.add_argument("--list-devices", action="store_true", help="List key-capable input devices and exit.")
    return parser.parse_args(argv)


def _print_devices(candidates: Sequence[CandidateDevice]) -> None:
    if not candidates:
        print("No key-capable input devices found.")
        return
    for candidate in candidates:
        print(
            f"{candidate.event_path}: name={candidate.name!r}, id_path={candidate.id_path!r}, "
            f"vid:pid={candidate.vendor_id}:{candidate.model_id}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = (args.config or default_config_path()).expanduser().resolve()
    try:
        candidates = discover_key_devices()
        if args.list_devices:
            _print_devices(candidates)
            return 0

        stored_config: dict[str, Any] | None = None
        config_errors: list[str] = []
        if config_path.exists():
            try:
                stored_config = load_config(config_path)
                if not args.recalibrate:
                    if args.count is not None and stored_config["count"] != args.count:
                        config_errors = [
                            f"Stored config uses {stored_config['count']} foot pads, but --count {args.count} was requested."
                        ]
                    else:
                        config_errors = validate_config(stored_config, candidates)
            except FootPadError as exc:
                config_errors = [str(exc)]
        elif not config_path.exists():
            config_errors = [f"No config exists at {config_path}."]

        if args.check:
            if stored_config is not None and not config_errors:
                print(f"Foot-pad config is valid: {config_path}")
                return 0
            for error in config_errors or ["No valid stored config."]:
                print(error, file=sys.stderr)
            return 1

        if stored_config is not None and not config_errors and not args.recalibrate:
            print(f"Loaded valid foot-pad config: {config_path}")
            config = stored_config
        else:
            if config_errors:
                print("Foot-pad calibration is required:", file=sys.stderr)
                for error in config_errors:
                    print(f"  - {error}", file=sys.stderr)
            count = args.count
            if count is None and stored_config is not None:
                count = stored_config["count"]
            if count is None:
                count = _prompt_count()
            config = calibrate(count)
            save_config(config_path, config)
            print(f"Saved foot-pad config: {config_path}")
        run_daemon(config)
        return 0
    except FootPadError as exc:
        print(f"foot_pad: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
