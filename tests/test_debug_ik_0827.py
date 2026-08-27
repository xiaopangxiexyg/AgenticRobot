from __future__ import annotations

import json

from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint
from teleop_runtime.real_teleop.cli import build_real_teleop_argument_parser
from teleop_runtime.real_teleop.debug_recorders import IkDebugLogRecorder
from teleop_runtime.real_teleop.debug_timing import debug_timing_modules


def test_debug_ik_0827_is_explicitly_opt_in_without_changing_timing_modules() -> None:
    parser = build_real_teleop_argument_parser()

    default_args = parser.parse_args([])
    investigation_args = parser.parse_args(["--debug-ik-0827"])

    assert default_args.debug_ik_0827 is False
    assert investigation_args.debug_ik_0827 is True
    assert debug_timing_modules(default_args) == debug_timing_modules(
        investigation_args
    )


def test_debug_ik_0827_recorder_uses_a_distinct_opt_in_filename(tmp_path) -> None:
    default_recorder = IkDebugLogRecorder(root_dir=tmp_path / "default")
    default_path = default_recorder.start(now=1_700_000_000.125)
    default_recorder.stop(reason="test", now=1_700_000_000.5)

    investigation_recorder = IkDebugLogRecorder(
        root_dir=tmp_path / "investigation",
        filename_prefix="ik_debug_0827",
    )
    investigation_path = investigation_recorder.start(now=1_700_000_000.125)
    investigation_recorder.write_event(
        "config",
        {
            "recording_profile": "ik-0827",
            "source": "--debug-ik-0827",
            "formal_dataset_debug_metadata": False,
        },
        now=1_700_000_000.2,
    )
    investigation_recorder.record_frame(
        {
            "schema": "dexteleop_ik_debug_v1",
            "event": "frame",
            "recording_profile": "ik-0827",
            "payload": {"loop_elapsed_ms": 8.0},
        }
    )
    investigation_recorder.stop(reason="test", now=1_700_000_000.5)

    assert default_path.name.startswith("ik_debug_")
    assert not default_path.name.startswith("ik_debug_0827_")
    assert investigation_path.name.startswith("ik_debug_0827_")
    rows = [
        json.loads(line)
        for line in investigation_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == ["start", "config", "frame", "stop"]
    assert rows[1]["payload"]["recording_profile"] == "ik-0827"
    assert rows[2]["recording_profile"] == "ik-0827"
    assert rows[-1]["payload"]["reason"] == "test"


def test_legacy_recorder_wrapper_accepts_the_runtime_prefix_assignment(tmp_path) -> None:
    recorder = legacy_entrypoint.IkDebugLogRecorder(root_dir=tmp_path)
    recorder.filename_prefix = "ik_debug_0827"
    path = recorder.start(now=1_700_000_000.125)
    recorder.stop(reason="test", now=1_700_000_000.5)

    assert path.name.startswith("ik_debug_0827_")


def test_no_debug_disk_write_can_coexist_with_the_opt_in_flag() -> None:
    args = build_real_teleop_argument_parser().parse_args(
        ["--debug-ik-0827", "--no-debug-disk-write"]
    )

    assert args.debug_ik_0827 is True
    assert args.no_debug_disk_write is True
