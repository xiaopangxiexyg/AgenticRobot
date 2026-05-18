from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = str(_PACKAGE_DIR.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from openpi05_deployment.action_adapter import adapt_openpi_result
from openpi05_deployment.client import OpenPIClient
from openpi05_deployment.dataset_source import DatasetTrajectoryError, DatasetTrajectorySource
from openpi05_deployment.errors import (
    InferenceResultError,
    ObservationAdapterError,
    ServerMetadataError,
)
from openpi05_deployment.metadata import ServerContract, parse_server_contract
from openpi05_deployment.observation_adapter import build_openpi_observation
from teleop_runtime.deploy.observation import DeploymentObservation


class OpenPIChunkPolicy:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        embodiment: str | None = None,
        camera_key: str = "front",
        prompt: str | None = None,
        api_key: str | None = None,
        wait_timeout_sec: float = 600.0,
        metadata_timeout_sec: float = 3.0,
        metadata_poll_interval_sec: float = 1.0,
        allow_missing_tactile_in_dry_run: bool = True,
        disable_deploy_rate_limits: bool = True,
        close_client_on_fatal_error: bool = True,
        input_source: str = "live",
        dataset_root: str | None = None,
        dataset_episode: int | None = None,
        dataset_episode_selection: str = "specified",
        dataset_start_frame: int = 0,
        dataset_camera: str | None = None,
        dataset_prompt: str | None = None,
        dataset_loop: bool = False,
        dataset_seed: int = 0,
        dataset_stride: int | None = None,
        use_dataset_initial_state: bool | None = True,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.expected_embodiment = embodiment
        self.camera_key = str(camera_key or "front")
        self.prompt = prompt
        self.api_key = api_key
        self.wait_timeout_sec = float(wait_timeout_sec)
        self.metadata_timeout_sec = float(metadata_timeout_sec)
        self.metadata_poll_interval_sec = float(metadata_poll_interval_sec)
        self.allow_missing_tactile_in_dry_run = bool(allow_missing_tactile_in_dry_run)
        self.disable_deploy_rate_limits = bool(disable_deploy_rate_limits)
        self.close_client_on_fatal_error = bool(close_client_on_fatal_error)
        self.input_source = str(input_source or "live").strip().lower()
        if self.input_source not in {"live", "dataset_compare", "dataset_execute"}:
            raise ValueError("input_source must be one of: live, dataset_compare, dataset_execute.")
        self.dataset_root = None if dataset_root in (None, "") else str(dataset_root)
        self.dataset_episode = None if dataset_episode is None else int(dataset_episode)
        self.dataset_episode_selection = str(dataset_episode_selection or "specified").strip().lower()
        self.dataset_start_frame = int(dataset_start_frame)
        self.dataset_camera = dataset_camera
        self.dataset_prompt = dataset_prompt
        self.dataset_loop = bool(dataset_loop)
        self.dataset_seed = int(dataset_seed)
        self.dataset_stride = None if dataset_stride is None else int(dataset_stride)
        self.use_dataset_initial_state = True if use_dataset_initial_state is None else bool(use_dataset_initial_state)
        if self.input_source != "live" and self.dataset_root is None:
            raise ValueError("dataset_root is required when input_source is dataset_compare or dataset_execute.")
        self._client: OpenPIClient | None = None
        self._contract: ServerContract | None = None
        self._dataset_source: DatasetTrajectorySource | None = None
        self._status_text = "openpi=init"
        self._last_warning_text = ""
        self._ensure_client()

    @property
    def status_text(self) -> str:
        return self._status_text

    def _ensure_client(self) -> tuple[OpenPIClient, ServerContract]:
        if self._client is None:
            self._client = OpenPIClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                wait_timeout_sec=self.wait_timeout_sec,
                metadata_timeout_sec=self.metadata_timeout_sec,
                metadata_poll_interval_sec=self.metadata_poll_interval_sec,
            )
        metadata = self._client.metadata
        self._contract = parse_server_contract(
            metadata,
            host=self.host,
            port=self.port,
            expected_embodiment=self.expected_embodiment,
        )
        if self._dataset_source is None and self.dataset_root and (
            self.input_source != "live" or self.use_dataset_initial_state
        ):
            self._dataset_source = DatasetTrajectorySource(
                self._contract,
                dataset_root=self.dataset_root,
                episode_index=self.dataset_episode,
                episode_selection=self.dataset_episode_selection,
                start_frame=self.dataset_start_frame,
                camera=self.dataset_camera,
                prompt=self.dataset_prompt,
                loop=self.dataset_loop,
                seed=self.dataset_seed,
                stride=self.dataset_stride,
            )
        return self._client, self._contract

    def _startup_init_enabled(self) -> bool:
        return bool(self.use_dataset_initial_state) and self.input_source != "dataset_compare"

    @property
    def disable_auto_startup_init(self) -> bool:
        return self.input_source == "dataset_compare"

    def _close_client(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        self._client = None
        self._contract = None

    def reset(self) -> None:
        self._status_text = "openpi=reset"
        self._last_warning_text = ""
        if self._client is not None:
            reset = getattr(self._client, "reset", None)
            if callable(reset):
                reset()

    def close(self) -> None:
        self._close_client()

    def get_startup_init_action(self) -> np.ndarray | None:
        if not self._startup_init_enabled():
            return None
        self._ensure_client()
        if self._dataset_source is None:
            return None
        return self._dataset_source.get_startup_init_action()

    @property
    def startup_init_source(self) -> str:
        if not self._startup_init_enabled():
            return "disabled"
        self._ensure_client()
        if self._dataset_source is None:
            return "disabled"
        return self._dataset_source.startup_init_source

    def skip_startup_frames(self, count: int = 1) -> int:
        if getattr(self, "input_source", "live") == "live" or self._dataset_source is None:
            return 0
        return self._dataset_source.skip_startup_frames(count)

    def _predict_from_dataset_source(
        self,
        observation: DeploymentObservation,
        *,
        client: OpenPIClient,
        contract: ServerContract,
    ) -> Any:
        if self._dataset_source is None:
            raise DatasetTrajectoryError("Dataset-backed policy mode requires a dataset source.")
        send_actions = bool(observation.metadata.get("send_actions"))
        if getattr(self, "input_source", "live") == "dataset_compare" and send_actions:
            raise ObservationAdapterError(
                "dataset_compare mode forbids send_actions=True; use dry-run only."
            )

        step = self._dataset_source.next_step()
        result = client.infer(step.payload)
        predicted_compact = np.asarray(result.get("actions"), dtype=np.float32)
        metrics = self._dataset_source.compare_prediction(
            predicted_compact,
            step.compact_ground_truth_actions,
        )
        chunk = adapt_openpi_result(
            result,
            observation,
            contract,
            disable_deploy_rate_limits=self.disable_deploy_rate_limits,
        )
        self._dataset_source.advance(len(chunk))
        timing = dict(result.get("server_timing") or {})
        infer_ms = float(timing.get("infer_ms", 0.0) or 0.0)
        mode = "cmp" if getattr(self, "input_source", "live") == "dataset_compare" else "exec"
        self._status_text = (
            f"openpi=ds_{mode}:ep={step.episode_index},frame={step.frame_index},"
            f"infer_ms={infer_ms:.1f},{metrics.status_text}"
        )
        return chunk

    def predict(self, observation: DeploymentObservation) -> Any:
        send_actions = bool(observation.metadata.get("send_actions"))
        allow_missing_tactile = bool((not send_actions) and self.allow_missing_tactile_in_dry_run)
        try:
            client, contract = self._ensure_client()
            if getattr(self, "input_source", "live") != "live":
                return self._predict_from_dataset_source(
                    observation,
                    client=client,
                    contract=contract,
                )
            adapted = build_openpi_observation(
                observation,
                contract,
                camera_key=self.camera_key,
                prompt=self.prompt,
                allow_missing_tactile=allow_missing_tactile,
            )
            result = client.infer(adapted.payload)
            chunk = adapt_openpi_result(
                result,
                observation,
                contract,
                disable_deploy_rate_limits=self.disable_deploy_rate_limits,
            )
            warning_text = ",".join(adapted.warnings)
            self._last_warning_text = warning_text
            timing = dict(result.get("server_timing") or {})
            infer_ms = float(timing.get("infer_ms", 0.0) or 0.0)
            warnings_suffix = "" if not warning_text else f",warn={warning_text}"
            self._status_text = (
                f"openpi=ok:{len(chunk)}/{contract.action_horizon},"
                f"infer_ms={infer_ms:.1f}{warnings_suffix}"
            )
            return chunk
        except (ServerMetadataError, ObservationAdapterError, InferenceResultError, DatasetTrajectoryError):
            self._status_text = "openpi=fault"
            if self.close_client_on_fatal_error:
                self._close_client()
            raise
        except Exception:
            self._status_text = "openpi=error"
            if self.close_client_on_fatal_error:
                self._close_client()
            raise


def create_policy(**kwargs: Any) -> OpenPIChunkPolicy:
    return OpenPIChunkPolicy(**kwargs)
