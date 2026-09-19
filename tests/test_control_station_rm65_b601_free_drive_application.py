from __future__ import annotations

import pytest

from control_station.sim_hil_profile.rm65_b601_free_drive_application import (
    READ_ONLY_CONFIRMATION,
    READ_ONLY_TELEMETRY_CONFIG_CONFIRMATION,
    build_observation_only_rm65_backend,
    main,
)


def test_observation_only_backend_builder_disables_command_paths(tmp_path) -> None:
    urdf = tmp_path / "rm65_arm.urdf"
    urdf.write_text("<robot name='test'/>", encoding="utf-8")

    backend = build_observation_only_rm65_backend(
        arm_ip="192.0.2.10",
        arm_urdf_path=urdf,
        connect_hardware=False,
    )
    try:
        assert backend.config.observation_only is True
        assert backend.config.connect_hardware is False
        assert backend.config.background_streaming_enabled is False
        assert backend.config.force_position_control is False
        assert backend.config.canfd_follow is False
    finally:
        backend.close()


def test_read_only_mode_requires_explicit_confirmation(tmp_path) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--connect-read-only",
                "--confirmation",
                "WRONG",
                "--arm-ip",
                "192.0.2.10",
                "--arm-urdf-path",
                str(tmp_path / "rm65_arm.urdf"),
                "--task-config",
                str(tmp_path / "task.json"),
                "--mapping-profile",
                str(tmp_path / "mapping.json"),
                "--rebot-source-root",
                str(tmp_path / "rebot"),
                "--log",
                str(tmp_path / "run.log"),
                "--output",
                str(tmp_path / "result.json"),
            ]
    )
    assert error.value.code == 2
    assert READ_ONLY_CONFIRMATION == "RM65_B601_READ_ONLY_FREE_DRIVE"


def test_realtime_push_configuration_requires_explicit_telemetry_confirmation(tmp_path) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--connect-read-only",
                "--source-transport",
                "realtime-push",
                "--configure-realtime-push",
                "--confirmation",
                READ_ONLY_CONFIRMATION,
                "--realtime-push-ip",
                "192.0.2.20",
                "--arm-ip",
                "192.0.2.10",
                "--arm-urdf-path",
                str(tmp_path / "rm65_arm.urdf"),
                "--task-config",
                str(tmp_path / "task.json"),
                "--mapping-profile",
                str(tmp_path / "mapping.json"),
                "--rebot-source-root",
                str(tmp_path / "rebot"),
                "--log",
                str(tmp_path / "run.log"),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
    assert error.value.code == 2
    assert READ_ONLY_TELEMETRY_CONFIG_CONFIRMATION == (
        "RM65_B601_READ_ONLY_CONFIGURE_TELEMETRY"
    )
