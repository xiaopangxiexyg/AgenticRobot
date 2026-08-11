from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim.contracts import SimulationContract
from sim.mujoco.diagnostics.cube_drop import (
    _require_glfw_connection,
    _require_interactive_display,
    build_arg_parser,
    build_isolated_drop_scene,
)


CONTRACT_PATH = Path("sim/configs/rm65_bimanual_physics_v1.json")


def test_cube_drop_cli_defaults_to_canonical_contract_without_loading_gui() -> None:
    args = build_arg_parser().parse_args([])

    assert args.contract.name == CONTRACT_PATH.name
    assert args.object is None
    assert args.start_delay == pytest.approx(1.0)
    assert args.repeat_every == pytest.approx(0.0)


def test_isolated_drop_keeps_only_contract_ground_table_and_object() -> None:
    contract = SimulationContract.load(CONTRACT_PATH)
    scene = build_isolated_drop_scene(contract)
    root = ET.fromstring(scene.xml)

    body_names = [body.attrib["name"] for body in root.findall(".//body")]
    assert body_names == ["table", "debug_cube"]
    assert root.find(".//body[@name='left_arm_base_link']") is None
    assert root.find(".//body[@name='right_arm_base_link']") is None
    assert root.find(".//geom[@name='ground_collision']") is not None
    assert root.find(".//geom[@name='table_collision_0']") is not None
    assert root.find(".//freejoint[@name='debug_cube_freejoint']") is not None
    assert root.find(".//geom[@name='debug_cube_collision']") is not None
    assert root.find("./contact") is None
    asset = root.find("./asset")
    assert asset is not None and len(asset) == 0

    option = root.find("./option")
    assert option is not None
    assert option.attrib["gravity"] == "0 0 -9.81"
    assert float(option.attrib["timestep"]) == pytest.approx(
        contract.timing.physics_timestep_sec
    )
    assert scene.object_spec == contract.scene.objects[0]


@pytest.mark.optional
def test_isolated_contract_cube_falls_and_settles_on_urdf_table() -> None:
    mujoco = pytest.importorskip("mujoco")
    contract = SimulationContract.load(CONTRACT_PATH)
    scene = build_isolated_drop_scene(contract)
    model = mujoco.MjModel.from_xml_string(scene.xml)
    data = mujoco.MjData(model)
    cube_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        scene.object_spec.name,
    )

    mujoco.mj_forward(model, data)
    initial_position = data.xpos[cube_id].copy()
    for _ in range(2000):
        mujoco.mj_step(model, data)

    assert model.nbody == 3  # world, fixed table, free cube
    assert model.njnt == 1
    assert data.xpos[cube_id, 2] < initial_position[2]
    # The canonical debug cube is 3 cm high, so it settles near z=0.025 m
    # on the table whose top surface is world z=0.01 m.
    assert data.xpos[cube_id, 2] == pytest.approx(0.025, abs=0.004)
    assert np.linalg.norm(data.qvel) < 0.05
    assert data.ncon >= 1


def test_gui_reports_missing_linux_display_before_importing_viewer() -> None:
    with pytest.raises(RuntimeError, match="neither DISPLAY nor WAYLAND_DISPLAY"):
        _require_interactive_display(environ={}, platform="linux")

    _require_interactive_display(environ={"DISPLAY": ":0"}, platform="linux")


def test_gui_reports_display_that_glfw_cannot_connect_to() -> None:
    unavailable = SimpleNamespace(init=lambda: False, terminate=lambda: None)
    with pytest.raises(RuntimeError, match="GLFW cannot connect"):
        _require_glfw_connection(unavailable)

    calls: list[str] = []
    available = SimpleNamespace(
        init=lambda: True,
        terminate=lambda: calls.append("terminate"),
    )
    _require_glfw_connection(available)
    assert calls == ["terminate"]
