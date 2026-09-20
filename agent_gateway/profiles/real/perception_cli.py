"""Camera-only real RGB/depth Skill host; no RM65 backend is constructed."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import sys
import threading
import uuid
from pathlib import Path

from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)
from agent_gateway.transport.unix_transport import (
    validate_agent_gateway_unix_group_parent,
    validate_agent_gateway_unix_server_preflight,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from deployments.perception import (
    GRASPGENX_DEFAULT_MODE,
    GraspGenXClient,
    GraspGenXInProcessBackend,
    GraspGenXPointGraspModelSpec,
    GraspNetPointGraspModelSpec,
    Sam2GraspGenXPointGraspPlanner,
    Sam2GraspNetPointGraspPlanner,
    Sam2GroundingDinoLocalizer,
    Sam2GroundingDinoModelSpec,
    Sam2VgnPointGraspPlanner,
    VgnPointGraspModelSpec,
)
from deployments.perception.dmtac_grasp_overlay import (
    reviewed_graspgenx_dmtac_solid_geometry,
)
from robot_motion.rm65 import (
    build_rm65_dual_arm_scene_profile,
    load_world_from_arm_base_transforms,
    load_world_from_fixed_link_transform,
)

from .perception_application import (
    CameraOnlyRealPerceptionApplicationState,
    assemble_camera_only_real_perception_composition,
)
from .perception_frames import build_real_perception_station_frame_binding
from .point_grasp_deployment import (
    DEFAULT_POINT_GRASP_BACKEND_ID,
    SUPPORTED_POINT_GRASP_BACKEND_IDS,
)
from .realsense_perception import build_real_perception_source_binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent_gateway real-perception",
        description=(
            "This profile never constructs or connects an RM65 backend. "
            "It serves passive real RGB/depth plus optional supplied-image "
            "object localization and live Station-bound point-grasp Skills."
        ),
    )
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--camera-registry", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--collision-scene-profile",
        default="dual_arm_hand_proxy_open",
        help="trusted fixed arm-base geometry used only for static camera poses",
    )
    parser.add_argument("--client-id", default="rpent.dexteleop")
    parser.add_argument("--client-uid", type=int, default=os.getuid())
    parser.add_argument(
        "--ssh-read-only-client",
        action="append",
        nargs=2,
        default=[],
        metavar=("CLIENT_ID", "UID"),
    )
    parser.add_argument("--runtime-session-id", default=None)
    parser.add_argument(
        "--control-domain-id",
        default="real-perception-only",
    )
    parser.add_argument("--max-retained-operations", type=int, default=16)
    parser.add_argument("--peer-idle-timeout-sec", type=float, default=10.0)
    parser.add_argument(
        "--enable-object-localization",
        action="store_true",
        help="load the reviewed SAM2 + Grounding DINO model bundle before serving",
    )
    parser.add_argument(
        "--object-localization-asset-bundle",
        type=Path,
        default=None,
        help=(
            "absolute path to assets/perception/sam2_grounding_dino; required "
            "when object localization is enabled"
        ),
    )
    parser.add_argument("--object-localization-device", default="cuda")
    parser.add_argument("--object-localization-box-threshold", type=float, default=0.35)
    parser.add_argument(
        "--object-localization-text-threshold", type=float, default=0.25
    )
    parser.add_argument(
        "--object-localization-automatic-score-threshold",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--object-localization-nms-iou-threshold", type=float, default=0.70
    )
    parser.add_argument(
        "--enable-point-grasp",
        action="store_true",
        help=(
            "enable the proposal-only live RGB-D point-grasp Route; requires "
            "the shared object-localization model"
        ),
    )
    parser.add_argument(
        "--point-grasp-backend",
        choices=tuple(sorted(SUPPORTED_POINT_GRASP_BACKEND_IDS)),
        default=DEFAULT_POINT_GRASP_BACKEND_ID,
    )
    parser.add_argument(
        "--point-grasp-asset-bundle",
        type=Path,
        default=None,
        help="absolute path to the reviewed VGN, GraspNet, or GraspGenX asset bundle",
    )
    parser.add_argument("--point-grasp-device", default="cuda")
    parser.add_argument("--graspgenx-host", default="127.0.0.1")
    parser.add_argument("--graspgenx-port", type=int, default=5556)
    parser.add_argument("--graspgenx-timeout-ms", type=int, default=120_000)
    parser.add_argument(
        "--graspgenx-mode",
        choices=("service", "in-process"),
        default=GRASPGENX_DEFAULT_MODE,
        help=(
            "GraspGenX execution mode; in-process is the default and requires a "
            "compatible Gateway Python, while service explicitly uses loopback ZMQ"
        ),
    )
    parser.add_argument(
        "--graspgenx-source-root",
        type=Path,
        default=None,
        help=(
            "optional absolute GraspGenX source checkout for in-process mode; "
            "defaults to model/perception/graspgenx"
        ),
    )
    return parser


def _absolute_existing_file(
    parser: argparse.ArgumentParser,
    value: Path,
    *,
    flag: str,
) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        parser.error(f"{flag} must be an absolute path")
    resolved = path.resolve()
    if not resolved.is_file():
        parser.error(f"{flag} is not a file: {resolved}")
    return resolved


def _absolute_directory(
    parser: argparse.ArgumentParser,
    value: Path,
    *,
    flag: str,
) -> Path:
    path = Path(os.path.abspath(value.expanduser()))
    if not value.expanduser().is_absolute():
        parser.error(f"{flag} must be an absolute path")
    try:
        info = os.lstat(path)
    except OSError:
        parser.error(f"{flag} is not an existing directory: {path}")
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        parser.error(f"{flag} must be a real directory, not a symlink: {path}")
    if info.st_mode & stat.S_IWOTH:
        parser.error(f"{flag} cannot be world-writable: {path}")
    return path


def _runtime_session_id(value: str | None) -> str:
    return value or "dexteleop-real-perception:" + uuid.uuid4().hex


def _graspgenx_mode_was_explicit(argv: list[str] | None) -> bool:
    """Keep the in-process default inert unless GraspGenX is selected."""

    tokens = sys.argv[1:] if argv is None else argv
    return any(
        token == "--graspgenx-mode"
        or token.startswith("--graspgenx-mode=")
        for token in tokens
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    socket_path = Path(os.path.abspath(args.socket_path.expanduser()))
    if not args.socket_path.expanduser().is_absolute():
        parser.error("--socket-path must be an absolute path")
    try:
        endpoint_value, peer_timeout = validate_agent_gateway_unix_server_preflight(
            socket_path=socket_path,
            peer_idle_timeout_sec=args.peer_idle_timeout_sec,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(f"invalid Unix endpoint: {error}")
    camera_config = _absolute_existing_file(
        parser,
        args.camera_config,
        flag="--camera-config",
    )
    camera_registry = _absolute_existing_file(
        parser,
        args.camera_registry,
        flag="--camera-registry",
    )
    artifact_root = _absolute_directory(
        parser,
        args.artifact_root,
        flag="--artifact-root",
    )
    if (
        _graspgenx_mode_was_explicit(argv)
        and args.graspgenx_mode != "service"
        and (
            not args.enable_point_grasp
            or args.point_grasp_backend != "graspgenx"
        )
    ):
        parser.error(
            "--graspgenx-mode applies only to an enabled "
            "--point-grasp-backend graspgenx"
        )
    object_localizer = None
    if args.enable_object_localization:
        if args.object_localization_asset_bundle is None:
            parser.error(
                "--enable-object-localization requires "
                "--object-localization-asset-bundle"
            )
        model_bundle = _absolute_directory(
            parser,
            args.object_localization_asset_bundle,
            flag="--object-localization-asset-bundle",
        )
        try:
            model_spec = Sam2GroundingDinoModelSpec.from_asset_bundle(model_bundle)
            object_localizer = Sam2GroundingDinoLocalizer(
                model_spec=model_spec,
                device=args.object_localization_device,
                box_threshold=args.object_localization_box_threshold,
                text_threshold=args.object_localization_text_threshold,
                automatic_score_threshold=(
                    args.object_localization_automatic_score_threshold
                ),
                nms_iou_threshold=args.object_localization_nms_iou_threshold,
            )
        except (OSError, TypeError, ValueError) as error:
            parser.error(f"invalid object-localization model bundle: {error}")
    point_grasp_planner = None
    if args.enable_point_grasp:
        if not args.enable_object_localization or object_localizer is None:
            parser.error("--enable-point-grasp requires --enable-object-localization")
        if args.point_grasp_asset_bundle is None:
            parser.error("--enable-point-grasp requires --point-grasp-asset-bundle")
        point_grasp_bundle = _absolute_directory(
            parser,
            args.point_grasp_asset_bundle,
            flag="--point-grasp-asset-bundle",
        )
        try:
            if args.point_grasp_backend == "vgn":
                point_grasp_spec = VgnPointGraspModelSpec.from_asset_bundle(
                    point_grasp_bundle
                )
                point_grasp_planner = Sam2VgnPointGraspPlanner(
                    model_spec=point_grasp_spec,
                    point_segmenter=object_localizer,
                    device=args.point_grasp_device,
                )
            elif args.point_grasp_backend == "graspnet":
                point_grasp_spec = GraspNetPointGraspModelSpec.from_asset_bundle(
                    point_grasp_bundle
                )
                point_grasp_planner = Sam2GraspNetPointGraspPlanner(
                    model_spec=point_grasp_spec,
                    point_segmenter=object_localizer,
                    device=args.point_grasp_device,
                )
            else:
                point_grasp_spec = GraspGenXPointGraspModelSpec.from_asset_bundle(
                    point_grasp_bundle
                )
                graspgenx_source_root = (
                    None
                    if args.graspgenx_source_root is None
                    else _absolute_directory(
                        parser,
                        args.graspgenx_source_root,
                        flag="--graspgenx-source-root",
                    )
                )
                graspgenx_backend = (
                    GraspGenXClient(
                        host=args.graspgenx_host,
                        port=args.graspgenx_port,
                        timeout_ms=args.graspgenx_timeout_ms,
                    )
                    if args.graspgenx_mode == "service"
                    else GraspGenXInProcessBackend(
                        model_spec=point_grasp_spec,
                        source_root=graspgenx_source_root,
                        device=args.point_grasp_device,
                    )
                )
                point_grasp_planner = Sam2GraspGenXPointGraspPlanner(
                    model_spec=point_grasp_spec,
                    point_segmenter=object_localizer,
                    client=graspgenx_backend,
                    solid_geometry=reviewed_graspgenx_dmtac_solid_geometry(),
                )
        except (OSError, TypeError, ValueError) as error:
            parser.error(f"invalid point-grasp model bundle: {error}")
    read_only_clients = tuple(
        (str(client_id), int(uid_text))
        for client_id, uid_text in args.ssh_read_only_client
    )
    client_bindings = compile_agent_gateway_unix_client_bindings(
        primary_client_id=args.client_id,
        primary_client_uid=args.client_uid,
        ssh_read_only_clients=read_only_clients,
    )
    if client_bindings.socket_mode == 0o660:
        validate_agent_gateway_unix_group_parent(socket_path=endpoint_value)
    scene_profile = build_rm65_dual_arm_scene_profile(args.collision_scene_profile)
    world_from_base = load_world_from_arm_base_transforms(scene_profile)
    scene_mount_hash = canonical_content_hash(
        {side: world_from_base[side].tolist() for side in ("left", "right")},
        domain="dexteleop.real_skill.rm65_scene_mounts.v1",
    )
    frame_binding = build_real_perception_station_frame_binding(
        collision_scene_profile=args.collision_scene_profile,
        scene_mount_hash=scene_mount_hash,
        scene_world_from_canonical_station=load_world_from_fixed_link_transform(
            scene_profile,
            link_name="workspace_near_boundary_frame",
        ),
    )
    source_binding = build_real_perception_source_binding(
        camera_config_path=camera_config,
        camera_registry_path=camera_registry,
        scene_world_from_arm_base_by_side=world_from_base,
    )
    composition = assemble_camera_only_real_perception_composition(
        source_binding=source_binding,
        socket_path=endpoint_value,
        artifact_root=artifact_root,
        client_bindings=client_bindings,
        runtime_session_id=_runtime_session_id(args.runtime_session_id),
        control_domain_id=args.control_domain_id,
        frame_binding=frame_binding,
        object_localizer=object_localizer,
        point_grasp_planner=point_grasp_planner,
        max_retained_operations=args.max_retained_operations,
        peer_idle_timeout_sec=peer_timeout,
    )
    ready = threading.Event()
    stop_requested = threading.Event()
    errors: list[BaseException] = []

    def ready_callback() -> None:
        manifest = composition.contract_bundle.route_manifest
        ready_payload = {
            "event": "gateway_ready",
            "profile": "real-perception",
            "robot_backend": "absent",
            "socket_path": str(composition.socket_path),
            "runtime_session_id": manifest.runtime_session_id,
            "current_mode": manifest.current_mode,
            "route_manifest_hash": manifest.semantic_content_hash,
            "deployment_profile_hash": composition.deployment_profile_hash,
            "frame_graph_hash": frame_binding.frame_graph.semantic_content_hash,
            "reference_frame_id": frame_binding.reference_frame_id,
            "rgb_roles": list(composition.registration.profile.rgb_roles),
            "depth_roles": list(composition.registration.profile.depth_roles),
            "object_localization_enabled": (
                composition.localization_registration is not None
            ),
            "local_path_test_input_enabled": False,
        }
        if composition.station_point_grasp_registration is not None:
            ready_payload["point_grasp_enabled"] = True
            render_registration = composition.station_point_grasp_render_registration
            assert render_registration is not None
            ready_payload["point_grasp_candidate_render_enabled"] = True
            ready_payload["point_grasp_render_view_roles"] = list(
                render_registration.profile.source_by_role
            )
        print(
            json.dumps(
                ready_payload,
                sort_keys=True,
            ),
            flush=True,
        )
        ready.set()

    def serve() -> None:
        try:
            composition.application.serve_forever(ready_callback=ready_callback)
        except BaseException as error:  # noqa: BLE001 - surfaced to the main thread.
            errors.append(error)
        finally:
            stop_requested.set()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, request_stop)
    thread = threading.Thread(
        target=serve,
        name="dexteleop-agent-gateway-real-perception",
        daemon=False,
    )
    thread.start()
    try:
        while not ready.wait(0.1):
            if errors or not thread.is_alive():
                break
        if errors:
            raise RuntimeError("real perception Gateway failed to start") from errors[0]
        if not ready.is_set():
            raise RuntimeError("real perception Gateway stopped before ready")
        while thread.is_alive():
            if stop_requested.wait(0.25):
                composition.application.request_shutdown()
                break
        thread.join(timeout=3.0)
        if thread.is_alive():
            raise RuntimeError("real perception Gateway did not stop")
        if errors:
            raise RuntimeError("real perception Gateway failed") from errors[0]
    finally:
        if thread.is_alive():
            composition.application.request_shutdown()
            thread.join(timeout=3.0)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        print(
            json.dumps(
                {
                    "event": "gateway_stopped",
                    "profile": "real-perception",
                    "state": composition.application.state.value,
                    "robot_backend": "absent",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if composition.application.state != CameraOnlyRealPerceptionApplicationState.CLOSED:
        raise RuntimeError("real perception Gateway did not close cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
