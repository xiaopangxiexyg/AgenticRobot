"""One-command operator trial for passive Quest/RM65 paired-pose capture.

This Control Station composition root deliberately exposes no arbitrary
``run_policy`` argument passthrough.  It selects the builtin hold policy,
disables end-effectors and cameras, omits ``--send-actions``, and injects one
Quest paired-pose factory into the existing policy-deploy lifecycle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from hardware.quest_dev.features.visualization.paired_pose_capture import (
    PairedPoseCaptureEvidence,
    StaticWindowCaptureProfile,
)
from hardware.quest_dev.server import load_config, validate_pairing_token
from runtime_artifact_paths import RuntimeArtifactLayout
from teleop_runtime.deploy.quest_paired_pose_capture import (
    QuestPairedPoseCaptureFactory,
    QuestPairedPoseCaptureLaunchConfig,
)
from teleop_runtime.deploy.quest_paired_pose_evidence_store import (
    QuestPairedPoseDiagnosticFit,
    fit_quest_paired_pose_evidence,
    publish_quest_paired_pose_diagnostic_fit,
    publish_sealed_quest_paired_pose_evidence,
)
from teleop_runtime.deploy.run_policy import main as run_policy_main

from .web import QuestCalibrationWebCoordinator

OPERATOR_TRIAL_PROFILE_ID_V1 = "quest-rm65-operator-trial-v1"
TRUSTED_LAN_EASY_PAIRING_CODE = "111111"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_QUEST_CONFIG = _PROJECT_ROOT / "hardware" / "quest_dev" / "config.yaml"


class _LaunchInfoPort(Protocol):
    base_url: str
    pairing_url: str
    pairing_code: str


class _FactoryPort(Protocol):
    launch_info: _LaunchInfoPort | None

    def wait_for_application(self, *, timeout_s: float) -> object: ...


class _CoordinatorPort(Protocol):
    @property
    def url(self) -> str: ...

    def start(self) -> None: ...

    def close(self) -> None: ...


def operator_trial_profile_v1() -> StaticWindowCaptureProfile:
    """Return an explicit candidate-capture profile, never an acceptance gate."""

    return StaticWindowCaptureProfile(
        profile_id=OPERATOR_TRIAL_PROFILE_ID_V1,
        min_window_duration_ns=1_000_000_000,
        max_window_duration_ns=5_000_000_000,
        min_source_span_ns=750_000_000,
        max_pair_skew_ns=100_000_000,
        min_robot_samples=5,
        max_robot_samples=128,
        min_quest_samples=10,
        max_quest_samples=256,
        max_robot_translation_excursion_m=0.003,
        max_robot_rotation_excursion_deg=0.75,
        max_quest_translation_excursion_m=0.008,
        max_quest_rotation_excursion_deg=1.5,
        max_quest_reorder_events=128,
        max_event_history=8192,
        max_completed_windows=32,
    )


def _positive_float(value: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return normalized


def _non_empty(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise argparse.ArgumentTypeError("value must be non-empty")
    return normalized


def _pairing_code(value: str) -> str:
    try:
        return validate_pairing_token(value, "--pairing-code")
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start the local Quest calibration operator UI and the existing "
            "RM65 no-action pose-observation composition."
        ),
        epilog=(
            "This launcher does not send arm actions, accept a calibration, "
            "install a frame, or render policy futures. Candidate publication "
            "requires an explicit --publish-* flag. Ctrl+C is the normal "
            "full-session shutdown."
        ),
    )
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument(
        "--controller-handedness",
        choices=("left", "right", "any"),
        default="left",
        help=(
            "Terminal capture controller filter. The default left keeps the "
            "selection deterministic for left-arm calibration; use any only "
            "when exactly one tracked controller is active."
        ),
    )
    parser.add_argument("--arm-ip", type=_non_empty, required=True)
    parser.add_argument(
        "--arm-urdf-path",
        type=_non_empty,
        default=None,
        help=(
            "Explicit six-axis RM65 arm-only URDF used for TCP FK; recommended "
            "for real calibration so the source model is unambiguous."
        ),
    )
    parser.add_argument(
        "--capture-id",
        type=_non_empty,
        default=None,
        help="Unique capture ID; omitted values use a UTC timestamp.",
    )
    parser.add_argument(
        "--capture-profile",
        choices=("operator-trial-v1",),
        default="operator-trial-v1",
        help=(
            "Explicit candidate-capture admission profile. It is not a "
            "calibration acceptance profile."
        ),
    )
    parser.add_argument(
        "--quest-config",
        type=Path,
        default=_DEFAULT_QUEST_CONFIG,
    )
    pairing = parser.add_mutually_exclusive_group()
    pairing.add_argument(
        "--trusted-lan",
        action="store_true",
        help=(
            "Use the easy fixed Quest pairing code 111111. Only use on an "
            "explicitly trusted isolated LAN."
        ),
    )
    pairing.add_argument(
        "--pairing-code",
        type=_pairing_code,
        help="Explicit six-digit Quest pairing code; otherwise use a random code.",
    )
    parser.add_argument("--operator-host", default="127.0.0.1")
    parser.add_argument("--operator-port", type=int, default=0)
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument(
        "--arm-state-refresh-hz",
        type=_positive_float,
        default=10.0,
    )
    parser.add_argument(
        "--quest-startup-timeout-s",
        type=_positive_float,
        default=30.0,
    )
    parser.add_argument(
        "--callback-drain-timeout-s",
        type=_positive_float,
        default=5.0,
    )
    parser.add_argument(
        "--host-join-timeout-s",
        type=_positive_float,
        default=5.0,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved no-action composition without starting devices.",
    )
    parser.add_argument(
        "--publish-sealed-evidence",
        action="store_true",
        help=(
            "After a clean run, publish the explicitly sealed candidate capture "
            "to the content-addressed runtime artifact layout."
        ),
    )
    parser.add_argument(
        "--publish-diagnostic-fit",
        action="store_true",
        help=(
            "Also publish the offline candidate diagnostic fit after sealed "
            "evidence publication; this never accepts or installs calibration."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "Explicit runtime-artifact root for --publish-*; otherwise use the "
            "configured local artifact layout."
        ),
    )
    return parser


def _capture_id(value: str | None) -> str:
    if value is not None:
        return value
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"quest-rm65-trial-{timestamp}"


def _pairing_token_env(config_path: Path) -> str:
    config = load_config(config_path)
    server = config.get("server", {})
    if not isinstance(server, Mapping):
        raise TypeError("Quest config server must be a mapping")
    pairing = server.get("pairing", {})
    if not isinstance(pairing, Mapping):
        raise TypeError("Quest config server.pairing must be a mapping")
    if not bool(pairing.get("enabled", True)):
        raise ValueError("Quest pairing must remain enabled")
    token_env = str(pairing.get("token_env", "QUEST_DEV_PAIRING_TOKEN")).strip()
    if not token_env:
        raise ValueError("Quest pairing token_env must be non-empty")
    return token_env


def _resolved_pairing_code(args: argparse.Namespace) -> str | None:
    if bool(args.trusted_lan):
        return TRUSTED_LAN_EASY_PAIRING_CODE
    return args.pairing_code


def build_deploy_argv(args: argparse.Namespace) -> list[str]:
    side = str(args.side)
    argv = [
        "--policy",
        "hold",
        "--rm65-pose-observation",
        "--active-sides",
        side,
        f"--{side}-arm-ip",
        str(args.arm_ip),
        "--end-effector-sides",
        "none",
        "--camera-mode",
        "none",
        "--arm-control-mode",
        "position",
        "--arm-state-refresh-hz",
        str(args.arm_state_refresh_hz),
    ]
    arm_urdf_path = getattr(args, "arm_urdf_path", None)
    if arm_urdf_path:
        argv.extend(["--arm-urdf-path", str(arm_urdf_path)])
    return argv


def _dry_run_payload(
    args: argparse.Namespace,
    *,
    capture_id: str,
    profile: StaticWindowCaptureProfile,
    pairing_token_env: str,
) -> dict[str, object]:
    pairing_code = _resolved_pairing_code(args)
    return {
        "mode": "quest_rm65_passive_calibration_trial",
        "capture_id": capture_id,
        "capture_profile": profile.as_dict(),
        "quest_config": str(args.quest_config),
        "arm": {
            "side": str(args.side),
            "ip": str(args.arm_ip),
            "urdf_path": getattr(args, "arm_urdf_path", None),
        },
        "quest_pairing": {
            "mode": (
                "trusted_lan_easy"
                if bool(args.trusted_lan)
                else "explicit"
                if pairing_code is not None
                else "random"
            ),
            "token_env": pairing_token_env,
            "fixed_code": pairing_code,
        },
        "operator_ui": {
            "host": args.operator_host,
            "port": args.operator_port,
            "open_browser": not bool(args.no_open_browser),
        },
        "terminal_capture": {
            "r": "start one fixed 3s static window; repeat after auto-completion",
            "c": "seal completed windows and run diagnostic AX=XB fit",
            "controller_handedness": str(args.controller_handedness),
            "preview_only": True,
        },
        "deploy_argv": build_deploy_argv(args),
        "safety": {
            "policy": "hold",
            "send_actions": False,
            "end_effectors": False,
            "cameras": False,
            "artifact_publication": bool(
                getattr(args, "publish_sealed_evidence", False)
                or getattr(args, "publish_diagnostic_fit", False)
            ),
            "calibration_acceptance": False,
        },
    }


def _announce_quest(
    factory: _FactoryPort,
    *,
    timeout_s: float,
    stop_event: threading.Event,
    emit: Callable[[str], None],
) -> None:
    try:
        factory.wait_for_application(timeout_s=timeout_s)
    except Exception as error:  # noqa: BLE001 - bounded type-only operator output.
        if not stop_event.is_set():
            emit(
                "[quest-calibration] Quest startup did not become ready: "
                f"{type(error).__name__}"
            )
        return
    if stop_event.is_set():
        return
    launch = factory.launch_info
    if launch is None:
        emit("[quest-calibration] Quest launch information is unavailable")
        return
    emit("[quest-calibration] ================================================")
    emit(f"[quest-calibration] Quest 打开: {launch.base_url}")
    emit(f"[quest-calibration] Quest 配对六码: {launch.pairing_code}")
    emit(f"[quest-calibration] 一次性完整网址: {launch.pairing_url}")
    emit(
        "[quest-calibration] 终端控制：r=自动记录3秒窗口（重复进入下一次），"
        "c=封存并解算"
    )
    emit("[quest-calibration] 解算结果仅作为 Quest 诊断预览，不安装正式标定")
    emit("[quest-calibration] 完成测试后回到终端按 Ctrl+C")
    emit("[quest-calibration] ================================================")


def _artifact_layout(args: argparse.Namespace) -> RuntimeArtifactLayout:
    root = getattr(args, "artifact_root", None)
    if root is None:
        return RuntimeArtifactLayout.configured()
    return RuntimeArtifactLayout(root=Path(root).expanduser())


def _publish_requested_artifacts(
    factory: object,
    args: argparse.Namespace,
    *,
    emit: Callable[[str], None],
) -> None:
    publish_evidence = bool(getattr(args, "publish_sealed_evidence", False))
    publish_fit = bool(getattr(args, "publish_diagnostic_fit", False))
    if not publish_evidence and not publish_fit:
        return
    application = getattr(factory, "application", None)
    evidence = getattr(application, "evidence", None)
    if not isinstance(evidence, PairedPoseCaptureEvidence):
        raise TypeError(
            "requested Quest artifact publication, but no sealed candidate "
            "evidence is available; click seal before stopping"
        )
    layout = _artifact_layout(args)
    evidence_commit = publish_sealed_quest_paired_pose_evidence(
        layout=layout,
        evidence=evidence,
    )
    emit(
        "[quest-calibration] published candidate evidence: "
        f"{evidence_commit.path} "
        f"evidence_hash={evidence_commit.evidence_semantic_content_hash}"
    )
    if not publish_fit:
        return
    diagnostic_fit = fit_quest_paired_pose_evidence(evidence)
    if not isinstance(diagnostic_fit, QuestPairedPoseDiagnosticFit):
        raise TypeError("Quest evidence store returned an invalid diagnostic fit")
    diagnostic_commit = publish_quest_paired_pose_diagnostic_fit(
        layout=layout,
        diagnostic_fit=diagnostic_fit,
    )
    emit(
        "[quest-calibration] published candidate diagnostic fit: "
        f"{diagnostic_commit.path} "
        f"fit_hash={diagnostic_commit.diagnostic_fit_semantic_content_hash}"
    )


def run(
    args: argparse.Namespace,
    *,
    factory_builder: Callable[
        [QuestPairedPoseCaptureLaunchConfig], _FactoryPort
    ] = QuestPairedPoseCaptureFactory,
    coordinator_builder: Callable[..., _CoordinatorPort] = (
        QuestCalibrationWebCoordinator
    ),
    policy_runner: Callable[..., int] = run_policy_main,
    emit: Callable[[str], None] = print,
) -> int:
    config_path = Path(args.quest_config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Quest config does not exist: {config_path}")
    args.quest_config = config_path
    pairing_token_env = _pairing_token_env(config_path)
    profile = operator_trial_profile_v1()
    capture_id = _capture_id(args.capture_id)
    if bool(args.dry_run):
        emit(
            json.dumps(
                _dry_run_payload(
                    args,
                    capture_id=capture_id,
                    profile=profile,
                    pairing_token_env=pairing_token_env,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    pairing_code = _resolved_pairing_code(args)
    previous_pairing_code = os.environ.get(pairing_token_env)
    if pairing_code is not None:
        os.environ[pairing_token_env] = pairing_code
    else:
        os.environ.pop(pairing_token_env, None)

    coordinator: _CoordinatorPort | None = None
    announcer: threading.Thread | None = None
    stop_event = threading.Event()
    try:
        factory = factory_builder(
            QuestPairedPoseCaptureLaunchConfig(
                config_path=config_path,
                force_http=False,
                profile=profile,
                capture_id=capture_id,
                startup_timeout_s=float(args.quest_startup_timeout_s),
                callback_drain_timeout_s=float(args.callback_drain_timeout_s),
                host_join_timeout_s=float(args.host_join_timeout_s),
                realman_left_ip=(
                    str(args.arm_ip)
                    if str(args.side) == "left" and args.arm_urdf_path
                    else None
                ),
                realman_left_urdf=(
                    Path(args.arm_urdf_path)
                    if str(args.side) == "left" and args.arm_urdf_path
                    else None
                ),
                controller_handedness=(
                    None
                    if str(args.controller_handedness) == "any"
                    else str(args.controller_handedness)
                ),
            )
        )
        coordinator = coordinator_builder(
            factory,
            host=str(args.operator_host),
            port=int(args.operator_port),
            open_browser=not bool(args.no_open_browser),
            application_timeout_s=float(args.quest_startup_timeout_s),
        )
        announcer = threading.Thread(
            target=_announce_quest,
            kwargs={
                "factory": factory,
                "timeout_s": float(args.quest_startup_timeout_s),
                "stop_event": stop_event,
                "emit": emit,
            },
            name="quest-calibration-launch-announcer",
            daemon=True,
        )
        coordinator.start()
        emit(f"[quest-calibration] 桌面操作页: {coordinator.url}")
        emit("[quest-calibration] 正在等待 RM65 观测链路与 Quest 页面...")
        announcer.start()
        result = int(
            policy_runner(
                argv=build_deploy_argv(args),
                quest_paired_pose_capture_factory=factory,
            )
        )
        if result == 0:
            _publish_requested_artifacts(factory, args, emit=emit)
        elif bool(
            getattr(args, "publish_sealed_evidence", False)
            or getattr(args, "publish_diagnostic_fit", False)
        ):
            emit(
                "[quest-calibration] skipped artifact publication because the "
                f"observation run returned status {result}"
            )
        return result
    finally:
        stop_event.set()
        if coordinator is not None:
            coordinator.close()
        if announcer is not None and announcer.is_alive():
            announcer.join(timeout=1.0)
        if previous_pairing_code is None:
            os.environ.pop(pairing_token_env, None)
        else:
            os.environ[pairing_token_env] = previous_pairing_code


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
