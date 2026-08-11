from dataclasses import dataclass, field
import re
import time
from typing import Callable, List, Optional, Sequence, Tuple
import serial



@dataclass
class ExoDriverCfg:
    # port: str = "/dev/serial/by-id/usb-1a86_USB2.0-Serial-if00-port0"
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    # servo_ids: List[int] = field(default_factory=lambda: list((1, 2, 3, 4, 5, 6)))
    servo_ids: List[int] = field(default_factory=lambda: [1,2])
    timeout_s: float = 0.2
    command_delay_s: float = 0.008
    read_hz: float = 10.0
    clear_input_before_command: bool = True


@dataclass
class ExoRead:
    servo_id: int
    raw_pos: Optional[int]
    angle_deg: Optional[float]
    response: str
    ok: bool
    timestamp_s: float


@dataclass
class ExoFrame:
    reads: List[ExoRead]
    timestamp_s: float
    elapsed_s: float
    period_s: float
    overrun_s: float

    @property
    def ok(self) -> bool:
        return all(read.ok for read in self.reads)

    @property
    def angles_deg(self) -> List[Optional[float]]:
        return [read.angle_deg for read in self.reads]

    @property
    def raw_positions(self) -> List[Optional[int]]:
        return [read.raw_pos for read in self.reads]


@dataclass
class ExoChangeIdResult:
    current_id: int
    new_id: int
    read_current_cmd: str
    read_current_response: str
    change_id_cmd: str
    change_id_response: str
    verify_cmd: Optional[str] = None
    verify_response: Optional[str] = None

    @property
    def verified(self) -> bool:
        return self.verify_response is not None and f"#{self.new_id:03d}" in self.verify_response


def require_serial() -> None:
    if serial is None:
        raise RuntimeError("pyserial is not installed. Install it with: pip install pyserial")


def validate_servo_id(servo_id: int) -> None:
    if servo_id < 0 or servo_id > 999:
        raise ValueError("servo ID must be in range 0..999")


def build_cmd(servo_id: int, body: str) -> str:
    validate_servo_id(servo_id)
    return f"#{servo_id:03d}{body}!"


def send_command(
    ser: serial.Serial,
    cmd: str,
    delay_s: float = 0.01,
    clear_input: bool = False,
    *,
    response_pattern: re.Pattern[str] | None = None,
    timeout_s: float | None = None,
    poll_interval_s: float = 0.001,
) -> str:
    """Send one command and optionally wait for its protocol response.

    ``serial.read_all()`` is non-blocking even when the port has a configured
    timeout.  A single read after ``delay_s`` therefore drops otherwise valid
    replies that arrive a few milliseconds later.  Read-only queries pass an
    exact response pattern and use this bounded polling path; write commands
    retain the historical one-shot behavior.
    """

    if clear_input:
        ser.reset_input_buffer()
    ser.write(cmd.encode("ascii"))
    ser.flush()
    time.sleep(delay_s)
    response = bytearray(ser.read_all())
    if response_pattern is None:
        return response.decode("ascii", errors="ignore")
    if timeout_s is None or timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive when response_pattern is set")
    if poll_interval_s <= 0.0:
        raise ValueError("poll_interval_s must be positive")
    deadline = time.monotonic() + float(timeout_s)
    while True:
        decoded = response.decode("ascii", errors="ignore")
        if response_pattern.search(decoded) is not None:
            return decoded
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return decoded
        time.sleep(min(float(poll_interval_s), remaining))
        response.extend(ser.read_all())


def parse_position(
    response: str,
    *,
    servo_id: int | None = None,
) -> Tuple[Optional[int], Optional[float]]:
    prefix = r"\d{3}" if servo_id is None else f"{int(servo_id):03d}"
    match = re.search(rf"#({prefix})P(\d{{4}})", response)
    if not match:
        return None, None

    raw_pos = int(match.group(2))
    return raw_pos, None


def format_servo_ids(servo_ids: Sequence[int]) -> str:
    return ", ".join(f"{servo_id:03d}" for servo_id in servo_ids)


class ExoDriver:
    def __init__(self, cfg: Optional[ExoDriverCfg] = None) -> None:
        self.pwm_min = 500
        self.pwm_max = 2500
        self.angle_range_deg = 270.0
        self.cfg = cfg or ExoDriverCfg()
        self.port = self.cfg.port
        self.ser: Optional[serial.Serial] = None
        self._validate_cfg()

    @property
    def period_s(self) -> float:
        return 1.0 / self.cfg.read_hz

    @property
    def estimated_min_cycle_s(self) -> float:
        return len(self.cfg.servo_ids) * self.cfg.command_delay_s

    @property
    def estimated_max_hz(self) -> float:
        if self.estimated_min_cycle_s <= 0.0:
            return float("inf")
        return 1.0 / self.estimated_min_cycle_s

    def _validate_cfg(self) -> None:
        if not self.cfg.servo_ids:
            raise ValueError("servo_ids must not be empty")
        for servo_id in self.cfg.servo_ids:
            validate_servo_id(servo_id)
        if self.cfg.command_delay_s <= 0.0:
            raise ValueError("command_delay_s must be positive")
        if self.cfg.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if self.cfg.read_hz <= 0.0:
            raise ValueError("read_hz must be positive")
        if self.estimated_min_cycle_s > self.period_s:
            raise ValueError(
                f"read_hz={self.cfg.read_hz:.3f} is too high for "
                f"{len(self.cfg.servo_ids)} servos with command_delay_s={self.cfg.command_delay_s:.4f}; "
                f"estimated max is {self.estimated_max_hz:.2f}Hz before serial/CPU overhead"
            )

    def open(self) -> None:
        require_serial()
        try:
            self.ser = serial.Serial(self.port, self.cfg.baudrate, timeout=self.cfg.timeout_s)
        except serial.SerialException as exc:
            raise RuntimeError(
                f"Failed to open serial port {self.port!r}: {exc}. "
                "Check the USB path and Linux dialout permissions."
            ) from exc

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def __enter__(self) -> "ExoDriver":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _serial(self) -> serial.Serial:
        if self.ser is None:
            raise RuntimeError("ExoDriver is not open")
        return self.ser

    def command(
        self,
        servo_id: int,
        body: str,
        delay_s: Optional[float] = None,
        *,
        response_pattern: re.Pattern[str] | None = None,
    ) -> str:
        return send_command(
            self._serial(),
            build_cmd(servo_id, body),
            delay_s=self.cfg.command_delay_s if delay_s is None else delay_s,
            clear_input=self.cfg.clear_input_before_command,
            response_pattern=response_pattern,
            timeout_s=(
                self.cfg.timeout_s if response_pattern is not None else None
            ),
        )

    def read_servo_id(self, servo_id: int) -> str:
        return self.command(servo_id, "PID")

    def change_servo_id(self, current_id: int, new_id: int, verify: bool = True) -> ExoChangeIdResult:
        validate_servo_id(current_id)
        validate_servo_id(new_id)

        read_current_cmd = build_cmd(current_id, "PID")
        read_current_response = self.command(current_id, "PID")

        change_id_cmd = build_cmd(current_id, f"PID{new_id:03d}")
        change_id_response = self.command(current_id, f"PID{new_id:03d}", delay_s=0.2)
        time.sleep(0.2)

        verify_cmd = None
        verify_response = None
        if verify:
            verify_cmd = build_cmd(new_id, "PID")
            verify_response = self.command(new_id, "PID")

        return ExoChangeIdResult(
            current_id=current_id,
            new_id=new_id,
            read_current_cmd=read_current_cmd,
            read_current_response=read_current_response,
            change_id_cmd=change_id_cmd,
            change_id_response=change_id_response,
            verify_cmd=verify_cmd,
            verify_response=verify_response,
        )

    def raw_pos_to_angle_deg(self, raw_pos: int) -> float:
        return (raw_pos - self.pwm_min) / (self.pwm_max - self.pwm_min) * self.angle_range_deg

    def read_position(self, servo_id: int) -> ExoRead:
        timestamp_s = time.monotonic()
        response = self.command(
            servo_id,
            "PRAD",
            response_pattern=re.compile(rf"#{servo_id:03d}P\d{{4}}"),
        )
        raw_pos, _angle_deg = parse_position(response, servo_id=servo_id)
        angle_deg = self.raw_pos_to_angle_deg(raw_pos) if raw_pos is not None else None
        return ExoRead(
            servo_id=servo_id,
            raw_pos=raw_pos,
            angle_deg=angle_deg,
            response=response,
            ok=raw_pos is not None and angle_deg is not None,
            timestamp_s=timestamp_s,
        )

    def read_frame(self) -> ExoFrame:
        start_s = time.monotonic()
        reads = [self.read_position(servo_id) for servo_id in self.cfg.servo_ids]
        elapsed_s = time.monotonic() - start_s
        overrun_s = max(0.0, elapsed_s - self.period_s)
        return ExoFrame(
            reads=reads,
            timestamp_s=start_s,
            elapsed_s=elapsed_s,
            period_s=self.period_s,
            overrun_s=overrun_s,
        )

    def read_frame_at_rate(self, next_tick_s: float) -> Tuple[ExoFrame, float]:
        now_s = time.monotonic()
        if now_s < next_tick_s:
            time.sleep(next_tick_s - now_s)
        frame = self.read_frame()
        next_tick_s = max(next_tick_s + self.period_s, time.monotonic())
        return frame, next_tick_s

    def iter_frames(self, count: Optional[int] = None):
        next_tick_s = time.monotonic()
        idx = 0
        while count is None or idx < count:
            frame, next_tick_s = self.read_frame_at_rate(next_tick_s)
            yield frame
            idx += 1

    def run(
        self,
        callback: Callable[[ExoFrame], None],
        count: Optional[int] = None,
        stop_on_bad_frame: bool = False,
    ) -> None:
        for frame in self.iter_frames(count=count):
            if stop_on_bad_frame and not frame.ok:
                raise RuntimeError("Failed to read a complete exo frame")
            callback(frame)

    def release_torque(self) -> List[str]:
        return [self.command(servo_id, "PULK") for servo_id in self.cfg.servo_ids]

    def move_mid(self, pwm: int = 1500, move_time_ms: int = 1000) -> List[str]:
        delay_s = max(0.2, move_time_ms / 1000.0)
        return [
            self.command(servo_id, f"P{pwm:04d}T{move_time_ms:04d}", delay_s=delay_s)
            for servo_id in self.cfg.servo_ids
        ]


def _print_frame(frame: ExoFrame) -> None:
    parts = []
    for read in frame.reads:
        if read.ok:
            parts.append(f"id={read.servo_id:03d} raw={read.raw_pos} deg={read.angle_deg:.2f}")
        else:
            parts.append(f"id={read.servo_id:03d} raw=<bad> resp={read.response.strip() or '<empty>'}")

    print(
        f"ok={frame.ok} elapsed={frame.elapsed_s * 1000.0:.1f}ms "
        f"period={frame.period_s * 1000.0:.1f}ms overrun={frame.overrun_s * 1000.0:.1f}ms | "
        + " | ".join(parts)
    )


class ExoDialVisualizer:
    def __init__(self, servo_ids: Sequence[int], angle_range_deg: float) -> None:
        import matplotlib.pyplot as plt

        self.plt = plt
        self.angle_range_deg = angle_range_deg
        self.fig, axes = plt.subplots(2, 3, figsize=(9, 6), subplot_kw={"projection": "polar"})
        self.axes = list(axes.flat)
        self.lines = []
        self.labels = []

        for ax, servo_id in zip(self.axes, servo_ids):
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticklabels([])
            ax.set_thetamin(0.0)
            ax.set_thetamax(angle_range_deg)
            ax.set_title(f"ID {servo_id:03d}")
            line, = ax.plot([0.0, 0.0], [0.0, 0.9], linewidth=3)
            label = ax.text(0.0, 0.15, "--", ha="center", va="center")
            self.lines.append(line)
            self.labels.append(label)

        self.fig.tight_layout()
        plt.ion()
        plt.show(block=False)

    def update(self, frame: ExoFrame) -> bool:
        for line, label, read in zip(self.lines, self.labels, frame.reads):
            if read.angle_deg is None:
                label.set_text("bad")
                continue
            angle_deg = max(0.0, min(self.angle_range_deg, read.angle_deg))
            theta = angle_deg / self.angle_range_deg * (self.angle_range_deg * 3.141592653589793 / 180.0)
            line.set_data([theta, theta], [0.0, 0.9])
            label.set_position((theta, 0.15))
            label.set_text(f"{read.angle_deg:.1f} deg")

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)
        return self.plt.fignum_exists(self.fig.number)


if __name__ == "__main__":
    cfg = ExoDriverCfg()
    with ExoDriver(cfg) as exo:
        exo.release_torque()
        visualizer = ExoDialVisualizer(cfg.servo_ids, exo.angle_range_deg)
        for frame in exo.iter_frames():
            _print_frame(frame)
            if not visualizer.update(frame):
                break
