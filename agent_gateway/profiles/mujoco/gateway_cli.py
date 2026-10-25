"""DexTeleop-owned MuJoCo launcher for the generic Agent TCP-pose Gateway.

This application leaf owns simulator preparation and the local Gateway server.
It does not embed RPent, an Agent planner, or an upper-level ProfileRegistry.
The same Gateway protocol remains the attachment boundary for later real
deployments; only this lower-owned launcher and its published manifest change.
"""

from __future__ import annotations

from deployments.dmtac_graspgenx_alignment_ids import (
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
    DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_IDS,
)

import argparse
import json
import os
import signal
import stat
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)
from agent_gateway.transport.unix_transport import (
    validate_agent_gateway_unix_group_parent,
    validate_agent_gateway_unix_server_preflight,
)
from project_paths import repo_root
from sim.mujoco.deploy import MujocoSimHILTargetConfig

from .motion_factory import (
    MujocoAgentArtifactStorageConfig,
    MujocoAgentTcpPoseHistoryStorageConfig,
    build_mujoco_agent_tcp_pose_gateway_session,
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
)

_RGB_BACKEND_WARMUP_WIDTH = 16
_RGB_BACKEND_WARMUP_HEIGHT = 16


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent_gateway mujoco",
        description=(
            "Serve a fixed-single-arm TCP-pose Agent Gateway on MuJoCo (default: right). "
            "This starts no Agent planner and connects no real device."
        ),
    )
    parser.add_argument("--graspgenx-alignment-profile",
        choices=DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_IDS, default=DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
        help="Default v5 nominal alignment; select v4 explicitly for legacy geometry.")
    parser.add_argument(
        "--socket-path",
        type=Path,
        required=True,
        help=(
            "absolute Unix socket path; its real, non-world-writable parent "
            "must already exist, the encoded path must fit Linux's 107-byte "
            "limit, and an existing endpoint is never replaced"
        ),
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        required=True,
        help="absolute MuJoCo task-config JSON path",
    )
    parser.add_argument(
        "--motion-side",
        choices=("left", "right"),
        default="right",
        help="fixed movable arm for this session; not per-request switching or simultaneous arms",
    )
    parser.add_argument(
        "--client-id",
        default="rpent.dexteleop",
        help="exact authenticated client identity advertised by this deployment",
    )
    parser.add_argument(
        "--client-uid",
        type=int,
        default=os.getuid(),
        help="local Unix peer UID authorized for --client-id (default: current UID)",
    )
    parser.add_argument(
        "--ssh-read-only-client",
        action="append",
        nargs=2,
        default=[],
        metavar=("CLIENT_ID", "UID"),
        help=(
            "additional distinct client identity and Unix UID restricted by the "
            "Gateway to the SSH read-only method/Route surface; may be repeated"
        ),
    )
    parser.add_argument(
        "--runtime-session-id",
        default=None,
        help=(
            "optional explicit lower runtime session identity; omitting it "
            "creates a unique identity for this Gateway process"
        ),
    )
    parser.add_argument(
        "--control-domain-id",
        default="mujoco-agent-tcp-pose",
        help="lower-owned control-domain identity published in the manifest",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "optional absolute existing non-world-writable directory for the "
            "persistent Agent Artifact CAS; providing it enables the passive "
            "depth-capture Skill"
        ),
    )
    parser.add_argument(
        "--segmentation-asset-bundle",
        type=Path,
        help=(
            "optional absolute assets/perception/sam2_grounding_dino directory; "
            "enables the live MuJoCo perception.segment Route and requires "
            "--artifact-root"
        ),
    )
    parser.add_argument(
        "--segmentation-device",
        default="cuda",
        help="Torch device used by the optional SAM2/Grounding DINO segmenter",
    )
    parser.add_argument("--enable-rgbd-capture-v3", action="store_true",
                        help="Publish capture_rgb v3 with optional native-grid paired RGB-D Artifacts; default off.")
    parser.add_argument(
        "--history-root",
        type=Path,
        help=(
            "optional absolute disk-primary planning-history directory; "
            "omitting it removes the history Route"
        ),
    )
    parser.add_argument(
        "--history-filename",
        default="planning_history.sqlite3",
        help="SQLite history basename used only with --history-root",
    )
    parser.add_argument(
        "--max-retained-operations",
        type=int,
        default=16,
        help="bounded lower operation-retention capacity",
    )
    parser.add_argument(
        "--max-session-admissions-per-route",
        type=int,
        default=None,
        help=(
            "optional explicit lifetime admission budget for each Route in "
            "this Gateway session; omitted means no count-based Route limit"
        ),
    )
    parser.add_argument(
        "--peer-idle-timeout-sec",
        type=float,
        default=10.0,
        help="authenticated Unix connection idle timeout",
    )
    parser.add_argument(
        "--no-realtime-worker",
        action="store_true",
        help="run MuJoCo stepping without realtime pacing (simulation only)",
    )
    parser.add_argument(
        "--enable-spatial-goals",
        action="store_true",
        help="Expose read-only endpoint relation calculation; does not authorize motion.",
    )
    parser.add_argument(
        "--enable-tcp-pose-comparison",
        action="store_true",
        help="Expose explicit read_tcp_pose v2 comparison with a retained absolute safe target; no motion authority.",
    )
    parser.add_argument(
        "--enable-safe-tcp-pose",
        action="store_true",
        help="also publish collision-aware move_tcp_to_pose_safe using lower-owned joint paths",
    )
    parser.add_argument(
        "--enable-grippers",
        action="store_true",
        help="publish bilateral atomic gripper Skills on the same deterministic MuJoCo owner",
    )
    parser.add_argument(
        "--enable-grasp-approach",
        action="store_true",
        help="publish right-arm approach_grasp_pose with an absolute envelope and noncontact joint planning; requires grippers and no realtime worker",
    )
    parser.add_argument(
        "--grasp-contact-target-body",
        help="also publish bounded terminal contact for this task object and its profile-bound right sensors",
    )
    parser.add_argument(
        "--enable-object-state", action="store_true",
        help="publish passive native object pose/velocity/contact for the explicit target; no task-success inference",
    )
    parser.add_argument(
        "--enable-object-follow-observation", action="store_true",
        help="also observe frozen follow/release on existing physics ticks; requires object-state feedback; no motion authorization",
    )
    parser.add_argument(
        "--enable-object-motion", action="store_true",
        help="publish carried-object lift/carry/place on the existing planner; requires grippers, grasp approach, contact target and explicit follow observation",
    )
    parser.add_argument("--enable-fast-grasp", action="store_true",
        help="publish frozen observation, first-acceptable GraspGenX planning and paired one-use native execution")
    parser.add_argument("--sam2-grounding-dino-bundle", type=Path)
    parser.add_argument("--graspgenx-point-grasp-bundle", type=Path)
    parser.add_argument("--graspgenx-source-root", type=Path)
    parser.add_argument("--point-grasp-device", default="cuda")
    parser.add_argument(
        "--safe-tcp-planning-clearance-padding-m",
        type=float,
        default=0.002,
        help="extra planning reserve for simulated gripper tracking (metres); live collision margin stays unchanged",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help=(
            "open the lower-owned passive MuJoCo viewer; the default remains "
            "headless and a later viewer loss degrades to headless execution"
        ),
    )
    return parser


def _absolute_existing_file(parser: argparse.ArgumentParser, value: Path) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute():
        parser.error("--task-config must be an absolute path")
    resolved = expanded.resolve()
    if not resolved.is_file():
        parser.error(f"--task-config is not a file: {resolved}")
    return resolved


def _absolute_path(parser: argparse.ArgumentParser, value: Path, *, flag: str) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute():
        parser.error(f"{flag} must be an absolute path")
    # Normalize ``..`` without resolving symlinks. The transport must inspect
    # the caller-selected endpoint parent itself instead of receiving a path
    # whose symlink identity was erased by this CLI.
    return Path(os.path.abspath(expanded))


def _absolute_managed_directory(
    parser: argparse.ArgumentParser,
    value: Path,
    *,
    flag: str,
) -> Path:
    path = _absolute_path(parser, value, flag=flag)
    try:
        info = os.lstat(path)
    except OSError:
        parser.error(f"{flag} is not an existing directory: {path}")
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        parser.error(f"{flag} must be a real directory, not a symlink: {path}")
    if info.st_mode & stat.S_IWOTH:
        parser.error(f"{flag} cannot be world-writable: {path}")
    return path


def _resolved_runtime_session_id(explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    return "dexteleop-agent-mujoco-tcp-pose:" + uuid.uuid4().hex


def _ready_payload(
    *,
    session: Any,
    socket_path: Path,
    client_id: str,
    viewer: Any | None = None,
    read_only_client_ids: tuple[str, ...] = (),
    socket_mode: int = 0o600,
) -> dict[str, Any]:
    manifest = session.contract_bundle.route_manifest
    capabilities = session.service.protocol_runtime.session_capabilities
    return {
        "event": "gateway_ready",
        "socket_path": str(socket_path),
        "client_id": client_id,
        "read_only_client_ids": list(read_only_client_ids),
        "socket_mode": format(socket_mode, "04o"),
        "target_kind": manifest.target_kind,
        "current_mode": manifest.current_mode,
        "runtime_session_id": manifest.runtime_session_id,
        "control_domain_id": manifest.control_domain_id,
        "target_binding_id": manifest.target_binding_id,
        "route_manifest_hash": manifest.semantic_content_hash,
        "artifact_transfer_enabled": capabilities.artifact_transfer_enabled,
        "artifact_access_mechanisms": list(capabilities.artifact_access_mechanisms),
        "station_manifest_hash": manifest.station_manifest_hash,
        "execution_manifest_hash": manifest.execution_manifest_hash,
        "viewer": (
            {"enabled": False, "state": "disabled", "running": False}
            if viewer is None
            else dict(viewer.status())
        ),
        "routes": [
            {
                "skill_id": entry.route.skill_id,
                "skill_version": entry.route.skill_version,
                "route_id": entry.route.route_id,
                "route_version": entry.route.route_version,
                "route_manifest_entry_hash": entry.semantic_content_hash,
            }
            for entry in manifest.skill_routes
        ],
    }


def _stopped_payload(
    *,
    server: Any,
    session: Any,
    stop_reason: str | None = None,
    viewer_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    outcome = server.gateway_close_outcome
    receipt = None if outcome is None else outcome.port_receipt
    lower_hold_proved = bool(
        receipt is not None
        and getattr(receipt, "succeeded", False)
        and getattr(receipt, "proves_motion_or_hold", False)
    )
    motion_ever_activated = bool(session.application.started)
    safe_shutdown = not motion_ever_activated or lower_hold_proved
    return {
        "event": "gateway_stopped",
        "stop_reason": stop_reason or "server_loop_ended",
        "viewer": (
            {"enabled": False, "state": "disabled", "running": False}
            if viewer_status is None
            else dict(viewer_status)
        ),
        "gateway_close_error": (
            None
            if server.gateway_close_error is None
            else type(server.gateway_close_error).__name__
        ),
        "close_accepted": None if outcome is None else outcome.accepted,
        "close_reason_code": None if outcome is None else outcome.reason_code,
        "motion_ever_activated": motion_ever_activated,
        "lower_hold_proved": lower_hold_proved,
        "safe_shutdown": safe_shutdown,
        "safe_shutdown_basis": (
            "lower_hold_proved"
            if lower_hold_proved
            else "motion_never_activated"
            if not motion_ever_activated
            else "missing_lower_hold_proof"
        ),
    }


def _close_viewer_for_stopped_payload(
    viewer: Any | None,
) -> tuple[dict[str, Any] | None, BaseException | None]:
    """Close an optional viewer and retain its truthful final bounded status."""

    if viewer is None:
        return None, None
    close_error = None
    try:
        viewer.close()
    except BaseException as error:  # noqa: BLE001 - re-raised after reporting.
        close_error = error
    return dict(viewer.status()), close_error


def _prepare_gateway_before_ready(*, session: Any, viewer: Any | None) -> None:
    """Start lower resources and prove advertised passive reads are ready.

    MuJoCo's first offscreen renderer construction may include GLFW and shader
    initialization that exceeds the RGB Route's operation deadline when a
    passive viewer is already open.  Absorb that one-time cost before the
    socket is published instead of weakening the Skill deadline or letting the
    first admitted read expire.  The warmup is a discarded lower-local frame;
    it creates no Artifact, Agent observation, or operation.
    """

    start_perception_resources = getattr(session, "start_perception_resources", None)
    if callable(start_perception_resources):
        start_perception_resources()
    if viewer is not None:
        viewer.start()
    application = session.application
    application.start_observation_ready()
    if session.rgb_executor is not None:
        if (
            not application.started
            or not application.held
            or application.active_operation_id is not None
        ):
            raise RuntimeError(
                "RGB backend warmup requires a running lower application in HOLD"
            )
        sources = tuple(application.visual_source_catalog())
        if not sources:
            raise RuntimeError("Advertised RGB Route has no visual source")
        source_id = sources[0].get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise RuntimeError("Advertised RGB Route has an invalid visual source")
        snapshot = application.capture_visual_jpeg_snapshot(
            source_ids=(source_id,),
            width=_RGB_BACKEND_WARMUP_WIDTH,
            height=_RGB_BACKEND_WARMUP_HEIGHT,
        )
        frames = getattr(snapshot, "frames", None)
        frame = None if not isinstance(frames, Mapping) else frames.get(source_id)
        if (
            frame is None
            or getattr(frame, "width", None) != _RGB_BACKEND_WARMUP_WIDTH
            or getattr(frame, "height", None) != _RGB_BACKEND_WARMUP_HEIGHT
            or not isinstance(getattr(frame, "data", None), bytes)
            or not frame.data
        ):
            raise RuntimeError("MuJoCo RGB backend warmup returned an invalid frame")
    if session.history_recorder is not None:
        session.history_recorder.capture_once()
        session.history_recorder.start()


def _fast_grasp_model_from_args(parser, args):
    paths = (args.sam2_grounding_dino_bundle, args.graspgenx_point_grasp_bundle, args.graspgenx_source_root)
    if not args.enable_fast_grasp:
        if any(path is not None for path in paths):
            parser.error("model/source bundle flags require --enable-fast-grasp")
        return None
    if (not all(path is not None for path in paths) or args.artifact_root is None
            or not args.enable_grasp_approach or not args.enable_grippers
            or args.grasp_contact_target_body is None or not args.no_realtime_worker):
        parser.error("fast grasp requires all three model/source paths, Artifact storage, grippers, target-bound grasp approach and no realtime worker")
    if any(not p.is_absolute() or not p.is_dir() for p in paths):
        parser.error("fast grasp model/source paths must be absolute existing directories")
    from deployments.perception.dmtac_grasp_overlay import reviewed_graspgenx_dmtac_solid_geometry
    from deployments.perception.graspgenx_client import GraspGenXInProcessBackend
    from deployments.perception.sam2_graspgenx_point_grasp import (
        GraspGenXPointGraspModelSpec, Sam2GraspGenXPointGraspPlanner,
        GRASPGENX_DMTAC_REACHABILITY_ORIENTATION_VARIANTS_LOCAL_AXIS_DEG,
    )
    from deployments.perception.sam2_grounding_dino import (
        Sam2GroundingDinoLocalizer, Sam2GroundingDinoModelSpec,
    )

    spec = GraspGenXPointGraspModelSpec.from_asset_bundle(args.graspgenx_point_grasp_bundle)
    localizer = Sam2GroundingDinoLocalizer(
        model_spec=Sam2GroundingDinoModelSpec.from_asset_bundle(args.sam2_grounding_dino_bundle),
        device=args.point_grasp_device)
    return Sam2GraspGenXPointGraspPlanner(model_spec=spec, point_segmenter=localizer,
        client=GraspGenXInProcessBackend(model_spec=spec, source_root=args.graspgenx_source_root,
            device=args.point_grasp_device), alignment_profile_id=getattr(args, "graspgenx_alignment_profile", DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID),
        orientation_variants_local_axis_degrees=(
            GRASPGENX_DMTAC_REACHABILITY_ORIENTATION_VARIANTS_LOCAL_AXIS_DEG),
        enable_finger_exchange=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.graspgenx_alignment_profile != DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID and not args.enable_fast_grasp:
        parser.error("--graspgenx-alignment-profile requires --enable-fast-grasp")
    if args.motion_side == "left" and (
        args.enable_fast_grasp
        or args.enable_grippers
        or args.enable_grasp_approach
        or args.grasp_contact_target_body is not None
        or args.enable_rgbd_capture_v3
        or args.enable_spatial_goals
        or args.history_root is not None
        or args.segmentation_asset_bundle is not None
    ):
        parser.error(
            "--motion-side left currently supports baseline TCP/state Routes only; "
            "gripper, grasp, spatial goal, history and segmentation combinations are not enabled"
        )
    task_config = _absolute_existing_file(parser, args.task_config)
    socket_path = _absolute_path(parser, args.socket_path, flag="--socket-path")
    if args.client_uid < 0:
        parser.error("--client-uid must be non-negative")
    try:
        client_bindings = compile_agent_gateway_unix_client_bindings(
            primary_client_id=args.client_id,
            primary_client_uid=args.client_uid,
            ssh_read_only_clients=args.ssh_read_only_client,
        )
    except (TypeError, ValueError) as error:
        parser.error(f"invalid Agent Gateway client bindings: {error}")
    if client_bindings.read_only_client_ids:
        try:
            validate_agent_gateway_unix_group_parent(socket_path=socket_path)
        except (OSError, TypeError, ValueError) as error:
            parser.error(f"invalid read-only socket parent: {error}")
    if args.max_retained_operations <= 0:
        parser.error("--max-retained-operations must be positive")
    try:
        validate_agent_gateway_unix_server_preflight(
            socket_path=socket_path,
            peer_idle_timeout_sec=args.peer_idle_timeout_sec,
        )
    except (FileExistsError, TypeError, ValueError) as error:
        parser.error(f"--socket-path preflight failed: {error}")
    if (
        args.max_session_admissions_per_route is not None
        and args.max_session_admissions_per_route <= 0
    ):
        parser.error("--max-session-admissions-per-route must be positive")
    runtime_session_id = _resolved_runtime_session_id(args.runtime_session_id)
    history_storage = None
    if args.history_root is not None:
        history_storage = MujocoAgentTcpPoseHistoryStorageConfig(
            root=_absolute_path(parser, args.history_root, flag="--history-root"),
            filename=args.history_filename,
        )
    artifact_storage = None
    if args.artifact_root is not None:
        artifact_root = _absolute_managed_directory(
            parser,
            args.artifact_root,
            flag="--artifact-root",
        )
        artifact_storage = MujocoAgentArtifactStorageConfig(root=artifact_root)
    segmentation_localizer = None
    if args.enable_rgbd_capture_v3 and artifact_storage is None:
        parser.error("--enable-rgbd-capture-v3 requires --artifact-root")
    if args.segmentation_asset_bundle is not None:
        from deployments.perception.sam2_grounding_dino import (
            Sam2GroundingDinoLocalizer,
            Sam2GroundingDinoModelSpec,
        )

        if artifact_storage is None:
            parser.error("--segmentation-asset-bundle requires --artifact-root")
        segmentation_bundle = _absolute_managed_directory(
            parser,
            args.segmentation_asset_bundle,
            flag="--segmentation-asset-bundle",
        )
        try:
            segmentation_spec = Sam2GroundingDinoModelSpec.from_asset_bundle(
                segmentation_bundle
            )
            segmentation_localizer = Sam2GroundingDinoLocalizer(
                model_spec=segmentation_spec,
                device=args.segmentation_device,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            parser.error(f"invalid segmentation model bundle: {error}")

    # Task configs may contain repository-relative calibration/profile paths.
    # This dedicated process owns its working directory, so normalize it once
    # instead of making an RPent caller know DexTeleop's repository layout.
    os.chdir(repo_root())

    if not 0 <= args.safe_tcp_planning_clearance_padding_m < float("inf"):
        parser.error("--safe-tcp-planning-clearance-padding-m must be finite and non-negative")
    if args.enable_grippers and not args.no_realtime_worker:
        parser.error("--enable-grippers requires --no-realtime-worker for one physics owner")
    target_config = MujocoSimHILTargetConfig(
        task_config_path=task_config,
        realtime_worker=not args.no_realtime_worker,
        runtime_session_id=runtime_session_id,
        control_domain_id=args.control_domain_id,
        execution_observation_plan=(
            mujoco_agent_tcp_pose_execution_observation_plan_v1(
                motion_side=args.motion_side
            )
        ),
        exposed_pose_sides=(args.motion_side,),
        allow_scripted_task_reset=False,
    )
    fast_grasp_planner = _fast_grasp_model_from_args(parser, args)
    session = build_mujoco_agent_tcp_pose_gateway_session(
        motion_side=args.motion_side,
        enable_grippers=args.enable_grippers,
        target_config=target_config,
        allowed_client_ids=client_bindings.allowed_client_ids,
        max_retained_operations=args.max_retained_operations,
        max_session_admissions_per_route=(args.max_session_admissions_per_route),
        history_storage=history_storage,
        artifact_storage=artifact_storage,
        enable_rgbd_capture_v3=args.enable_rgbd_capture_v3,
        segmentation_localizer=segmentation_localizer,
        enable_safe_tcp_pose=args.enable_safe_tcp_pose,
        enable_spatial_goals=args.enable_spatial_goals,
        enable_tcp_pose_comparison=args.enable_tcp_pose_comparison,
        enable_object_state=args.enable_object_state,
        enable_object_follow_observation=args.enable_object_follow_observation,
        enable_object_motion=args.enable_object_motion,
        enable_grasp_approach=args.enable_grasp_approach,
        grasp_contact_target_body=args.grasp_contact_target_body,
        fast_grasp_planner=fast_grasp_planner,
        safe_tcp_planning_clearance_padding_m=args.safe_tcp_planning_clearance_padding_m,
    )
    viewer = None
    if args.viewer:
        from sim.mujoco.viewer import MujocoPassiveViewerSession

        viewer = MujocoPassiveViewerSession(
            session.application.scene,
            sync_callback=session.application.runtime.rig.sync_viewer,
        )
    server = session.build_unix_json_server(
        socket_path=str(socket_path),
        client_id_by_uid=client_bindings.client_id_by_uid,
        read_only_client_ids=client_bindings.read_only_client_ids,
        socket_mode=client_bindings.socket_mode,
        peer_idle_timeout_sec=args.peer_idle_timeout_sec,
    )
    stop_requested = threading.Event()
    server_error: list[BaseException] = []
    stopped_payload: dict[str, Any] | None = None
    viewer_close_error: BaseException | None = None
    stop_reason: str | None = None

    def serve() -> None:
        nonlocal stop_reason
        try:
            server.serve_forever(poll_interval=0.1)
        except BaseException as error:  # noqa: BLE001 - surfaced to main.
            server_error.append(error)
            stop_reason = "server_fault"
            stop_requested.set()

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_reason
        stop_reason = f"signal:{signal.Signals(signum).name}"
        stop_requested.set()

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, request_stop)
    server_thread = threading.Thread(
        target=serve,
        name="dexteleop-agent-gateway-mujoco",
        daemon=True,
    )
    server_thread_started = False
    viewer_sync_enabled = viewer is not None
    try:
        # Construction is deliberately default-off.  This explicit lower-owned
        # boundary starts MuJoCo and proves HOLD before any client can observe a
        # ready endpoint or before history capture touches the runtime.
        # The passive viewer must open first: its native window and the realtime
        # physics worker share MjModel/MjData, and concurrent startup has caused
        # native GLFW/GLX crashes on some systems.
        _prepare_gateway_before_ready(session=session, viewer=viewer)
        server_thread.start()
        server_thread_started = True
        print(
            json.dumps(
                _ready_payload(
                    session=session,
                    socket_path=socket_path,
                    client_id=args.client_id,
                    viewer=viewer,
                    read_only_client_ids=tuple(
                        sorted(client_bindings.read_only_client_ids)
                    ),
                    socket_mode=client_bindings.socket_mode,
                ),
                sort_keys=True,
            ),
            flush=True,
        )
        while server_thread.is_alive() and not stop_requested.wait(0.25):
            if viewer_sync_enabled and viewer is not None and not viewer.sync():
                viewer_state = str(viewer.status().get("state") or "unknown")
                if viewer_state in {"closed_by_user", "faulted"}:
                    viewer_sync_enabled = False
                    print(
                        json.dumps(
                            {
                                "event": "viewer_unavailable",
                                "gateway_continues_headless": True,
                                "viewer": dict(viewer.status()),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if (
                session.history_recorder is not None
                and session.history_recorder.last_error is not None
            ):
                raise RuntimeError("planning history recorder failed") from (
                    session.history_recorder.last_error
                )
        if server_error:
            raise RuntimeError("Agent Gateway server failed") from server_error[0]
    finally:
        if server_thread_started:
            server.shutdown()
        server.server_close()
        if server_thread_started:
            server_thread.join(timeout=2.0)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        # Gateway close above joins/fences the lower runtime first.  Release the
        # native viewer next so gateway_stopped describes its final state rather
        # than a pre-teardown snapshot that still claims it is running.
        viewer_status, viewer_close_error = _close_viewer_for_stopped_payload(viewer)
        stopped_payload = _stopped_payload(
            server=server,
            session=session,
            stop_reason=stop_reason,
            viewer_status=viewer_status,
        )
        print(json.dumps(stopped_payload, sort_keys=True), flush=True)
    if viewer_close_error is not None:
        raise RuntimeError("MuJoCo viewer close failed") from viewer_close_error
    if server.gateway_close_error is not None:
        raise RuntimeError("Agent Gateway close failed") from server.gateway_close_error
    if stopped_payload is None or not stopped_payload["safe_shutdown"]:
        raise RuntimeError("Agent Gateway stopped without a safe lower boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
