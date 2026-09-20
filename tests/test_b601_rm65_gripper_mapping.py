from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from control_station.sim_hil_profile.rebot_b601_rm65_gripper import (
    B601ToRM65GripperSimBridge,
)
from control_station.sim_hil_profile.rm65_b601_gripper import (
    RM65ToB601GripperSimBridge,
)
from hardware.rebot_b601_rs import (
    B601RSGripperReader,
    B601RSGripperReaderConfig,
)
from robot_motion.rebot_b601_rs import (
    B601_RS_GRIPPER_SOURCE_V1,
    B601_RS_URDF_RELATIVE_PATH,
    GRIPPER_CLOCK_DOMAIN_LOCAL_WALL_TIME,
    RM65_DMTAC_GRIPPER_SOURCE_V1,
    B601RM65GripperMappingProfile,
    GripperMappingCalibrationRequired,
    NormalizedGripperSample,
    load_b601_rm65_gripper_mapping_profile,
)
from sim.mujoco.config import GripperSpec
from sim.mujoco.rebot_b601_rs import (
    B601_RS_GRIPPER_NAME,
    B601GripperSimAdapter,
    build_b601_mujoco_scene,
)
from sim.mujoco.scene import MJCFConversion, MujocoScene, SceneLoadResult

REPO_ROOT = Path(__file__).resolve().parents[1]
REBOT_SOURCE_ROOT = REPO_ROOT / "hardware" / "reBotArm_control_py"
PROFILE_PATH = REPO_ROOT / "docs/hardware/profiles/b601_rm65_gripper_mapping_v1.json"
RM65_TASK_PATH = (
    REPO_ROOT
    / "sim/configs/tasks/rm65_right_cube_lift_rl_normal_full_grasp_stage1_v2.json"
)


def _mapping(**overrides) -> B601RM65GripperMappingProfile:
    payload = {
        "profile_id": "test",
        "b601_open_position_rad": 1.0,
        "b601_closed_position_rad": 0.0,
    }
    payload.update(overrides)
    return B601RM65GripperMappingProfile(**payload)


def _sample(
    close: float,
    sequence: int,
    *,
    source_id: str = B601_RS_GRIPPER_SOURCE_V1,
    timestamp_sec: float | None = None,
) -> NormalizedGripperSample:
    timestamp = time.time() if timestamp_sec is None else timestamp_sec
    return NormalizedGripperSample(
        close=close,
        sequence=sequence,
        source_timestamp_sec=timestamp,
        receive_timestamp_sec=timestamp,
        source_id=source_id,
    )


def _synthetic_scene(*, name: str) -> MujocoScene:
    mujoco = pytest.importorskip("mujoco")
    xml = f"""
<mujoco model="{name}">
  <option gravity="0 0 0" timestep="0.001"/>
  <worldbody>
    <body name="base">
      <body name="b601_gripper_end">
        <joint name="joint_left" type="slide" axis="1 0 0" range="0 0.05"/>
        <joint name="joint_right" type="slide" axis="1 0 0" range="0 0.0715"/>
        <geom name="finger" type="box" size="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    spec = GripperSpec(
        name=name,
        joint_names=("joint_left", "joint_right"),
        open_qpos=(0.05, 0.0715),
        closed_qpos=(0.0, 0.0),
        side="right",
    )
    return MujocoScene(
        mujoco=mujoco,
        model=model,
        data=mujoco.MjData(model),
        load_result=SceneLoadResult(
            conversion=MJCFConversion(
                urdf_path=Path("/synthetic/gripper.urdf"),
                mjcf_path=Path("/synthetic/gripper.xml"),
                camera_mounts=(),
                joint_names=("joint_left", "joint_right"),
                body_names=("base", "b601_gripper_end"),
                scene_profile_name="synthetic_gripper",
            ),
            cameras=(),
            arms={},
            grippers={name: spec},
        ),
    )


def test_profile_is_human_tunable_and_does_not_require_hash() -> None:
    profile = load_b601_rm65_gripper_mapping_profile(PROFILE_PATH)
    assert profile.profile_id.startswith("b601-rm65-gripper")
    assert profile.b601_calibrated is False
    assert profile.rm65_min_close == 0.0
    assert profile.rm65_max_close == 1.0
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    assert "sha256" not in payload
    assert "payload_sha256" not in payload


def test_b601_position_mapping_is_explicit_and_round_trips() -> None:
    profile = _mapping()
    assert profile.b601_position_to_canonical(1.0) == pytest.approx(0.0)
    assert profile.b601_position_to_canonical(0.5) == pytest.approx(0.5)
    assert profile.b601_position_to_canonical(0.0) == pytest.approx(1.0)
    assert profile.canonical_to_b601_position(0.25) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="outside the calibrated"):
        profile.b601_position_to_canonical(1.1)
    assert profile.b601_position_to_canonical(1.1, clamp=True) == pytest.approx(0.0)
    assert profile.b601_position_to_canonical(-0.1, clamp=True) == pytest.approx(1.0)


def test_uncalibrated_b601_position_fails_closed() -> None:
    profile = load_b601_rm65_gripper_mapping_profile(PROFILE_PATH)
    with pytest.raises(GripperMappingCalibrationRequired):
        profile.b601_position_to_canonical(0.0)


def test_normalized_sample_rejects_invalid_order_and_timestamps() -> None:
    with pytest.raises(ValueError, match="receive_timestamp"):
        NormalizedGripperSample(
            close=0.5,
            sequence=1,
            source_timestamp_sec=2.0,
            receive_timestamp_sec=1.0,
            source_id="test",
        )
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        _sample(1.1, 1)


def test_b601_j7_reader_is_parameter_only_and_never_commands_motor() -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    class Motor:
        def robstride_get_param_f32_host_id(self, *args):
            calls.append(("read", args))
            return -6.0

    class Controller:
        def __init__(self, channel):
            calls.append(("controller", (channel,)))

        def add_robstride_motor(self, *args):
            calls.append(("add", args))
            return Motor()

        def close(self):
            calls.append(("close", ()))

    config = B601RSGripperReaderConfig.from_rs_yaml(
        REBOT_SOURCE_ROOT / "config/rebotarm_rs.yaml"
    )
    reader = B601RSGripperReader(config, controller_factory=Controller)
    reader.connect()
    observation = reader.read_position()
    reader.close()

    assert observation.position_rad == pytest.approx(-6.0)
    assert calls[:3] == [
        ("controller", ("can0",)),
        ("add", (7, 0xFD, "rs-00")),
        ("read", (0x7019, 0xFD, 100)),
    ]
    assert calls[-1] == ("close", ())
    assert not {name for name, _ in calls} & {"enable", "disable", "mode"}


def test_forward_and_reverse_sim_bridges_use_same_normalized_contract() -> None:
    scene = _synthetic_scene(name="right_gripper")
    profile = _mapping()
    forward = B601ToRM65GripperSimBridge(scene, profile)
    reverse = RM65ToB601GripperSimBridge(scene, profile, target_gripper_name="right_gripper")

    forward_result = forward.apply_b601_sample(_sample(0.25, 1))
    assert forward_result.accepted is True
    assert scene.get_gripper_close("right_gripper") == pytest.approx(0.25)
    stale = forward.apply_b601_sample(_sample(0.20, 1))
    assert stale.accepted is False
    assert stale.rejection_reason == "stale_sequence"
    assert scene.get_gripper_close("right_gripper") == pytest.approx(0.25)

    reverse_result = reverse.apply_rm65_sample(
        _sample(0.75, 2, source_id=RM65_DMTAC_GRIPPER_SOURCE_V1)
    )
    assert reverse_result.accepted is True
    assert scene.get_gripper_close("right_gripper") == pytest.approx(0.75)

    wrong_source = forward.apply_b601_sample(
        _sample(0.50, 3, source_id=RM65_DMTAC_GRIPPER_SOURCE_V1)
    )
    assert wrong_source.accepted is False
    assert wrong_source.rejection_reason == "source_id_mismatch"
    assert scene.get_gripper_close("right_gripper") == pytest.approx(0.75)
    stale = forward.apply_b601_sample(
        _sample(0.5, 4, timestamp_sec=time.time() - 1.0)
    )
    assert stale.accepted is False
    assert stale.rejection_reason == "source_sample_stale"
    assert scene.get_gripper_close("right_gripper") == pytest.approx(0.75)
    wrong_clock = forward.apply_b601_sample(
        NormalizedGripperSample(
            close=0.5,
            sequence=5,
            source_timestamp_sec=time.time(),
            receive_timestamp_sec=time.time(),
            source_id=B601_RS_GRIPPER_SOURCE_V1,
            clock_domain_id="device_monotonic",
        )
    )
    assert wrong_clock.accepted is False
    assert wrong_clock.rejection_reason == "clock_domain_mismatch"
    assert GRIPPER_CLOCK_DOMAIN_LOCAL_WALL_TIME == "local_wall_time"


def test_b601_opt_in_scene_registers_gripper_without_changing_default() -> None:
    pytest.importorskip("mujoco")
    if not (REBOT_SOURCE_ROOT / B601_RS_URDF_RELATIVE_PATH).is_file():
        pytest.skip("initialize the pinned reBotArm_control_py gitlink")
    default = build_b601_mujoco_scene(REBOT_SOURCE_ROOT)
    assert default.gripper_specs == {}
    scene = build_b601_mujoco_scene(REBOT_SOURCE_ROOT, enable_gripper=True)
    assert set(scene.gripper_specs) == {B601_RS_GRIPPER_NAME}
    adapter = B601GripperSimAdapter(scene)
    adapter.set_close(0.5)
    assert adapter.measured_close == pytest.approx(0.5)


@pytest.mark.optional
def test_bridges_apply_to_existing_rm65_and_b601_sim_scene_assets() -> None:
    pytest.importorskip("mujoco")
    if not RM65_TASK_PATH.is_file():
        pytest.skip("the existing RM65 MuJoCo task asset is unavailable")
    from control_station.sim_hil_profile.rebot_b601_rm65_application import (
        build_b601_rm65_sim_target,
    )

    mapping = _mapping()
    runtime, _, _ = build_b601_rm65_sim_target(RM65_TASK_PATH)
    try:
        rm65_scene = runtime.rig.scene
        b601_scene = build_b601_mujoco_scene(
            REBOT_SOURCE_ROOT,
            enable_gripper=True,
        )
        forward = B601ToRM65GripperSimBridge(
            rm65_scene,
            mapping,
            target_gripper_name="right_gripper",
        )
        reverse = RM65ToB601GripperSimBridge(b601_scene, mapping)
        assert forward.apply_b601_sample(_sample(0.4, 1)).accepted is True
        assert rm65_scene.get_gripper_close("right_gripper") == pytest.approx(0.4)
        assert reverse.apply_rm65_sample(
            _sample(0.6, 1, source_id=RM65_DMTAC_GRIPPER_SOURCE_V1)
        ).accepted is True
        assert b601_scene.get_gripper_close(B601_RS_GRIPPER_NAME) == pytest.approx(0.6)
    finally:
        runtime.close()
