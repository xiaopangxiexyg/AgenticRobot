from __future__ import annotations

from types import SimpleNamespace

from hardware.rebot_b601_rs import (
    B601RSGripperControlConfig,
    B601RSGripperController,
    B601RSGripperReaderConfig,
)


def _config() -> B601RSGripperControlConfig:
    return B601RSGripperControlConfig(
        reader=B601RSGripperReaderConfig(),
        kp=50.0,
        kd=1.25,
        rate_hz=10.0,
        duration_sec=0.2,
        max_position_error_rad=0.25,
    )


def test_gripper_yaml_reads_mit_defaults_without_changing_reader_scope(tmp_path) -> None:
    path = tmp_path / "rebotarm_rs.yaml"
    path.write_text(
        """
channel: can0
groups:
  gripper:
    joints: [gripper]
joints:
  - name: gripper
    motor_id: 0x07
    feedback_id: 0xFD
    model: rs-00
    vendor: robstride
    MIT:
      kp: 50.0
      kd: 1.5
""",
        encoding="utf-8",
    )
    config = B601RSGripperControlConfig.from_rs_yaml(path, kd=0.75)

    assert config.reader.motor_id == 7
    assert config.reader.mit_kp == 50.0
    assert config.reader.mit_kd == 1.5
    assert config.kp == 50.0
    assert config.kd == 0.75


def test_j7_controller_only_enables_and_disables_registered_motor() -> None:
    class Clock:
        value = 0.0

        def __call__(self) -> float:
            return self.value

        def sleep(self, seconds: float) -> None:
            self.value += seconds

    class Motor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[float, ...]]] = []

        def robstride_get_param_f32_host_id(self, *_args):
            return 1.0

        def ensure_mode(self, mode, timeout_ms):
            self.calls.append(("ensure_mode", (float(mode), float(timeout_ms))))

        def enable(self):
            self.calls.append(("enable", ()))

        def disable(self):
            self.calls.append(("disable", ()))

        def send_mit(self, pos, vel, kp, kd, tau):
            self.calls.append(("send_mit", (pos, vel, kp, kd, tau)))

        def close(self):
            self.calls.append(("motor_close", ()))

    class Controller:
        def __init__(self):
            self.motor = Motor()
            self.added: tuple[int, int, str] | None = None
            self.closed = False

        def add_robstride_motor(self, motor_id, feedback_id, model):
            self.added = (motor_id, feedback_id, model)
            return self.motor

        def close(self):
            self.closed = True

    clock = Clock()
    controller = Controller()
    module = SimpleNamespace(Mode=SimpleNamespace(MIT=7))
    runtime = B601RSGripperController(
        _config(),
        controller_factory=lambda _channel: controller,
        module_loader=lambda: module,
        clock=clock,
        sleeper=clock.sleep,
    )

    runtime.connect()
    target = runtime.prepare_hold()
    commands = runtime.run_hold(target)
    runtime.close()

    assert target == 1.0
    assert commands == 2
    assert controller.added == (7, 0xFD, "rs-00")
    assert [name for name, _ in controller.motor.calls] == [
        "ensure_mode",
        "enable",
        "send_mit",
        "send_mit",
        "disable",
        "motor_close",
    ]
    assert controller.motor.calls[2][1] == (1.0, 0.0, 50.0, 1.25, 0.0)
    assert controller.closed is True
