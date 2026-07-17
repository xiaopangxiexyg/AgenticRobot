"""Formal UMI collection application assembly."""

from __future__ import annotations

from robot_motion.runtime.lifecycle import run_best_effort_cleanup

from ...gui.calibration_web_ui import CalibrationWebUi
from ..operator_labels import OperatorLabelStateMachine
from ..policy_io import load_policy_io_layout
from .calibration import UmiCalibrationBinding
from .components import build_umi_components
from .config import UmiCollectionConfig
from .controller import UmiCollectionController
from .pose_adapter import UmiPoseAdapter
from .recorder import UmiDatasetRecorder
from .snapshot import UmiSnapshotAssembler


def run_umi_collection(config: UmiCollectionConfig) -> int:
    calibration = UmiCalibrationBinding.load(config.calibration_result)
    calibration.require_valid(config)
    layout = load_policy_io_layout(config.policy_io_layout)
    if layout is None:
        raise RuntimeError("UMI collection requires a policy I/O layout")
    pose_adapter = UmiPoseAdapter(
        T_vive_world_to_desk=calibration.T_vive_world_to_desk,
        T_tcp_to_tracker_by_side={
            side: config.side(side).tcp_transform
            for side in ("left", "right")
        },
    )
    labels = OperatorLabelStateMachine()
    ui = CalibrationWebUi(
        host=config.gui_host,
        port=config.gui_port,
        open_browser=config.open_browser,
    )
    recorder = UmiDatasetRecorder(
        config=config,
        calibration=calibration,
        policy_io_layout=layout,
    )
    components = None
    try:
        ui.publish(
            ui_mode="umi_collection",
            stage="UMI 设备初始化中",
            control_hint="正在初始化 Vive、双腕相机、AngleTrigger 与 DMTac。",
        )
        ui.start()
        components = build_umi_components(config)
        last_commands = {"left": 0.0, "right": 0.0}
        assembler = UmiSnapshotAssembler(
            config=config,
            pose_adapter=pose_adapter,
            vive_source=components.vive,
            camera_selectors=components.camera_selectors(),
            hand_state_poller=components.hand_state_poller,
            gripper_command_provider=lambda: dict(last_commands),
            labels=labels,
        )
        controller = UmiCollectionController(
            config=config,
            calibration=calibration,
            components=components,
            recorder=recorder,
            ui=ui,
            labels=labels,
            snapshot_assembler=assembler,
            last_commands=last_commands,
        )
        components.start()
        ui.publish(
            ui_mode="umi_collection",
            stage="UMI 就绪",
            control_hint=f"UMI GUI: {ui.url}",
        )
        controller.run()
        return 0
    finally:
        steps = []
        if components is not None:
            steps.append(("components", components.close))
        steps.append(("recorder", recorder.close))
        steps.append(("ui", ui.close))
        run_best_effort_cleanup("umi_collection_app.close", steps)


__all__ = ["run_umi_collection"]
