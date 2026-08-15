"""Synthetic-observation OpenPI model-RTC wire/frontend probe.

This is a bounded no-action verification utility, not a production entrypoint.
It starts one localhost OpenPI server subprocess, warms the conditioned sampler,
runs the existing DexTeleop worker/timed-plan frontend, and terminates the exact
server subprocess in ``finally``.  It initializes no camera, robot, target,
safety, position, Control Station, teleop, replay, or dispatch component.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from deployments.client.artifact import load_policy_artifact
from deployments.client.rtc_frontend import (
    build_rtc_conditioned_model_shadow_frontend_v3,
)
from deployments.client.rtc_transport import RtcTransportDeadlineSpec
from deployments.openpi.client.rtc import (
    OPENPI_RTC_RESULT_TYPED_CANONICAL_V1,
    OpenPIFinalChunkRtcBackend,
    OpenPIFinalChunkRtcConfig,
)
from deployments.openpi.client.rtc_conditioned import (
    OPENPI_CONDITIONED_RTC_OPERATION_V1,
    OPENPI_CONDITIONED_RTC_RECEIPT_KEY_V1,
    OPENPI_CONDITIONED_RTC_WIRE_V1,
    OpenPIConditionedRtcBackendV3,
    OpenPIConditionedRtcConfig,
)
from deployments.openpi.client.transport import (
    OpenPIWebsocketTransport,
)
from robot_motion.deploy.observation import (
    OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
    DeploymentObservation,
)
from robot_motion.deploy.rtc import (
    RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1,
)
from robot_motion.deploy.runtime import (
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
)


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"OpenPI server exited before readiness: {process.returncode}."
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError("OpenPI server did not become ready within 60 seconds.")


def _synthetic_observation(*, timestamp: float, step_index: int) -> DeploymentObservation:
    state = np.zeros((OBSERVATION_STATE_DIM,), dtype=np.float32)
    for side in ("left", "right"):
        state[STATE_ARM_POSE_SLICES[side].start + 6] = 1.0
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    return DeploymentObservation(
        timestamp=timestamp,
        step_index=step_index,
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=True,
        end_effector_sides=("left", "right"),
        images={
            "left_wrist": image.copy(),
            "right_wrist": image.copy(),
        },
        image_timestamps={
            "left_wrist": timestamp,
            "right_wrist": timestamp,
        },
        state=state,
        state_valid=np.ones((OBSERVATION_STATE_DIM,), dtype=bool),
        hand_states={
            "left": {"gripper_state_scalar": 0.0},
            "right": {"gripper_state_scalar": 0.0},
        },
        metadata={"prompt": "bimanual cube manipulation"},
    )


def _warm_conditioned_wire(
    transport: OpenPIWebsocketTransport,
    *,
    horizon: int,
    action_dim: int,
) -> dict[str, object]:
    observation = {
        "observation.state": np.zeros((action_dim,), dtype=np.float32),
        "observation.images.left_wrist": np.zeros(
            (224, 224, 3),
            dtype=np.uint8,
        ),
        "observation.images.right_wrist": np.zeros(
            (224, 224, 3),
            dtype=np.uint8,
        ),
    }
    started = time.monotonic()
    response = transport.infer(
        {
            "__openpi_request__": {
                "wire_protocol": OPENPI_CONDITIONED_RTC_WIRE_V1,
                "operation": OPENPI_CONDITIONED_RTC_OPERATION_V1,
            },
            "observation": observation,
            "conditioning_actions": np.zeros(
                (horizon, action_dim),
                dtype=np.float32,
            ),
            "prefix_mask": np.zeros((horizon,), dtype=np.float32),
            "conditioning_mode": "unconditioned_bootstrap_v1",
            "receipt_context": {"probe": "mrtc-s1b2-warmup"},
        }
    )
    receipt = dict(response[OPENPI_CONDITIONED_RTC_RECEIPT_KEY_V1])
    actions = np.asarray(response["actions"], dtype=np.float32)
    return {
        "duration_sec": time.monotonic() - started,
        "shape": list(actions.shape),
        "finite": bool(np.all(np.isfinite(actions))),
        "conditioning_mode": receipt.get("conditioning_mode"),
        "consumed_generation_indices": receipt.get(
            "consumed_generation_indices"
        ),
        "provider_capability_hash": receipt.get(
            "provider_capability_hash"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    dexteleop_root = Path(args.dexteleop_root).resolve()
    openpi_root = Path(args.openpi_root).resolve()
    checkpoint_dir = (
        None
        if args.checkpoint_dir is None
        else Path(args.checkpoint_dir).resolve()
    )
    artifact_path = (
        checkpoint_dir / "policy_artifact.json"
        if checkpoint_dir is not None
        else (
            dexteleop_root
            / "deployments/openpi/configs/model_rtc"
            / "0712-wrist-cube-92599.policy_artifact.json"
        )
    )
    artifact = load_policy_artifact(artifact_path)
    continuation = artifact.model_continuation
    if continuation is None:
        raise ValueError("Probe artifact is missing model continuation.")
    runtime_contract = continuation.runtime
    port = _free_local_port()
    log_path = Path(f"/tmp/mrtc_s1b2_openpi_server_{port}.log")
    server_log = log_path.open("w", encoding="utf-8")
    if checkpoint_dir is None:
        server_command = [
            sys.executable,
            str(openpi_root / "scripts/serve_dexteleop_policy.py"),
            "--model",
            "0712-wrist-cube",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--enable-rtc-continuation",
        ]
    else:
        server_command = [
            sys.executable,
            "-m",
            "deployments.openpi.server",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--openpi-root",
            str(openpi_root),
            "--config",
            str(args.config),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--enable-rtc-continuation",
        ]
    server_env = dict(os.environ)
    existing_pythonpath = server_env.get("PYTHONPATH", "")
    server_env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(dexteleop_root), existing_pythonpath)
        if value
    )
    server = subprocess.Popen(
        server_command,
        cwd=openpi_root,
        env=server_env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    transport = OpenPIWebsocketTransport(
        host="127.0.0.1",
        port=port,
        retry_interval_s=0.1,
        deadline_spec=RtcTransportDeadlineSpec(
            connect_timeout_sec=60.0,
            metadata_timeout_sec=30.0,
            inference_timeout_sec=60.0,
            close_timeout_sec=5.0,
        ),
    )
    frontend = None
    try:
        _wait_for_server(server, port)
        base_backend = OpenPIFinalChunkRtcBackend(
            transport=transport,
            config=OpenPIFinalChunkRtcConfig(
                artifact=artifact,
                frame_contract_hash=runtime_contract.frame_contract_hash,
                normalization_hash=runtime_contract.normalization_hash,
                active_sides=("left", "right"),
                action_pose_mode="absolute",
                minimum_execution_horizon_steps=1,
                prompt="bimanual cube manipulation",
                result_action_representation=(
                    OPENPI_RTC_RESULT_TYPED_CANONICAL_V1
                ),
            ),
        )
        warmup = _warm_conditioned_wire(
            transport,
            horizon=runtime_contract.prediction_horizon_steps,
            action_dim=runtime_contract.model_action_dim,
        )
        conditioned = OpenPIConditionedRtcBackendV3(
            base_backend=base_backend,
            config=OpenPIConditionedRtcConfig.from_artifact(
                artifact,
                selected_prefix_schedule=(
                    RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1
                ),
            ),
        )
        deploy_runtime = PolicyDeployRuntime(
            PolicyDeployRuntimeConfig(
                active_sides=("left", "right"),
                arms_enabled=True,
                hands_enabled=True,
                action_keyframe_period_sec=1.0 / 30.0,
                action_maximum_hold_sec=1.0 / 30.0,
            )
        )
        observed = []
        base = time.monotonic()
        frontend = build_rtc_conditioned_model_shadow_frontend_v3(
            conditioned,
            runtime_session_id="session:mrtc-s1b2-openpi-gpu-shadow",
            control_domain_id="domain:mrtc-s1b2",
            producer_id="producer:mrtc-s1b2",
            policy_node_id="single_policy",
            deploy_runtime=deploy_runtime,
            preparer=object(),
            created_at=base,
            request_validity_steps=90,
            observe_bundle=observed.append,
        )
        preflight_now = time.monotonic()
        preflight_observation = _synthetic_observation(
            timestamp=preflight_now,
            step_index=0,
        )
        policy_observation = deploy_runtime.prepare_policy_observation(
            preflight_observation
        )
        actor_observation, _ = deploy_runtime.prepare_actor_observation(
            policy_observation
        )
        preflight_request, preflight_conditioning = (
            frontend._build_request_and_conditioning(
                actor_observation,
                now=preflight_now,
            )
        )
        if preflight_conditioning is None:
            raise RuntimeError("Model-RTC preflight did not build conditioning.")
        prepared = conditioned.prepare_worker_input(
            preflight_request,
            actor_observation,
        )
        preflight_started = time.monotonic()
        preflight_result = conditioned.create_client().infer(
            preflight_request,
            canonical_conditioning=preflight_conditioning,
            model_input=prepared.model_input,
            model_input_hash=prepared.model_input_hash,
        )
        exact_preflight = {
            "duration_sec": time.monotonic() - preflight_started,
            "conditioning_mode": (
                preflight_conditioning.conditioning_mode
            ),
            "response_accepted": (
                preflight_result.response_validation.accepted
            ),
            "output_horizon": (
                preflight_result.policy_chunk_result.execution_horizon_steps
            ),
        }
        frontend.start()
        emissions = 0
        max_step_duration = 0.0
        event_steps = []
        last_step = None
        deadline = time.monotonic() + 15.0
        step_index = 0
        while time.monotonic() < deadline:
            now = time.monotonic()
            started = time.monotonic()
            step = frontend.step(
                _synthetic_observation(
                    timestamp=now,
                    step_index=step_index,
                ),
                now=now,
            )
            last_step = step
            max_step_duration = max(
                max_step_duration,
                time.monotonic() - started,
            )
            if step.action_preparation.emission is not None:
                emissions += 1
            if (
                step.completion is not None
                or step.request_transition is not None
                or step.lifecycle_transition is not None
            ):
                event_steps.append(step.to_debug_payload())
            if len(observed) >= 2 and any(
                item.canonical_conditioning.selected_generation_indices
                for item in observed[1:]
            ):
                break
            step_index += 1
            time.sleep(max(0.0, 1.0 / 30.0 - (time.monotonic() - started)))

        bundles = [
            {
                "conditioning_mode": item.canonical_conditioning.conditioning_mode,
                "selected_generation_indices": list(
                    item.canonical_conditioning.selected_generation_indices
                ),
                "forecast_delay_steps": (
                    item.envelope.request.forecast_delay_steps
                ),
                "output_horizon": (
                    item.policy_chunk_result.execution_horizon_steps
                ),
                "response_accepted": item.response_validation.accepted,
                "wire_duration_sec": (
                    item.response.result_received_at
                    - item.response.inference_started_at
                ),
                "consumption_receipt_hash": (
                    item.response.consumption_receipt.semantic_content_hash
                ),
            }
            for item in observed
        ]
        if last_step is None:
            raise RuntimeError("Frontend loop did not execute.")
        return {
            "artifact_path": str(artifact_path),
            "checkpoint_dir": (
                None if checkpoint_dir is None else str(checkpoint_dir)
            ),
            "port": port,
            "server_pid": server.pid,
            "server_log": str(log_path),
            "warmup": warmup,
            "exact_client_preflight": exact_preflight,
            "frontend": {
                "profile": frontend.config.profile.profile_id,
                "send_actions": frontend.config.profile.send_actions,
                "position_lookahead_enabled": (
                    frontend.config.profile.position_lookahead_enabled
                ),
                "request_trigger_coverage_steps": (
                    frontend.config.request_trigger_coverage_steps
                ),
                "max_control_step_duration_sec": max_step_duration,
                "emission_count": emissions,
                "legacy_action_queue_pop_count": (
                    deploy_runtime.action_queue_pop_count()
                ),
                "bundles": bundles,
                "event_steps": event_steps,
                "final_step": last_step.to_debug_payload(),
                "final_transition_reason": (
                    last_step.state.last_transition_reason
                ),
            },
        }
    finally:
        if frontend is not None:
            frontend.close(now=time.monotonic())
        else:
            transport.close()
        server.terminate()
        try:
            server.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5.0)
        server_log.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dexteleop-root",
        default="/home/user/zc/DexTeleop",
    )
    parser.add_argument(
        "--openpi-root",
        default="/home/user/zc/openpi",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Optional checkpoint to test through deployments.openpi.server. "
            "When omitted, retain the historical 0712-wrist-cube probe."
        ),
    )
    parser.add_argument(
        "--config",
        default="0712-wrist-cube",
        help="Existing OpenPI base config used to restore a partial checkpoint.",
    )
    args = parser.parse_args()
    result = run(args)
    result["server_process_cleaned"] = True
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
