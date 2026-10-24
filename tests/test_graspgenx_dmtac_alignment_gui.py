"""Offline regressions for the current DMTAC-only alignment window."""

from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.integration, pytest.mark.optional]


@pytest.mark.parametrize("side", ("left", "right"))
def test_current_dmtac_window_geometry_controls_and_legacy_json(side: str) -> None:
    # Qt may be installed in a separate desktop environment. Never install or
    # import that interpreter's binary extensions into the pytest environment.
    python = os.environ.get("DMTAC_QT_PYTHON")
    if not python:
        pytest.skip("set DMTAC_QT_PYTHON to a desktop interpreter with PyQt5")
    program = r'''
import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QCheckBox, QGroupBox, QLabel
from utils.graspgenx_dmtac_alignment_gui import (
    AlignmentModel, _state_from_payload, load_tool_tcp_pose,
)
from utils.graspgenx_dmtac_alignment_qt import (
    AlignmentWindow, ComponentSceneBuilder, current_dmtac_state,
)
from deployments.perception import dmtac_grasp_overlay as owner

side = sys.argv[1]
root = Path.cwd()
tcp = load_tool_tcp_pose(root / "hardware/rm65/utils/configs/calibrations" / f"rm65_{side}_DMTAC_M.json")
state = current_dmtac_state(side, tcp, [0.] * 6)
assert state.model_opening_m == owner.DMTAC_M_GRASPGENX_MAX_OPENING_M
assert state.model_finger_length_m == owner.DMTAC_M_GRASPGENX_FINGER_LENGTH_M
assert state.model_palm_z_m == owner.DMTAC_M_GRASPGENX_PALM_Z_M
assert state.model_finger_thickness_m == owner.DMTAC_M_GRASPGENX_FINGER_THICKNESS_M
assert state.model_pad_height_m == owner.DMTAC_M_GRASPGENX_PAD_HEIGHT_M
assert state.grasp_center_from_tactile_z_m == owner.DMTAC_M_GRASPGENX_GRASP_CENTER_FROM_TACTILE_XYZ_M[2]
model = AlignmentModel(root / "assets/realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf", state)
import numpy as np
frames, _ = model.frames()
actual = np.linalg.inv(frames["tool_tcp"]) @ np.asarray(frames["graspgenx_model"])
expected = owner.reviewed_graspgenx_dmtac_solid_geometry().solid_pose(np.eye(4))
# The CAD left/right attachment differs slightly; both must share this model
# convention without replacing either side's hardware TCP calibration.
np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-5 if side == "left" else 1e-12)
builder = ComponentSceneBuilder(model)
def scene(cad=False, tactile=False):
    triangles, lines, markers, _, _ = builder.build(
        show_cad=cad, show_dmtac=tactile,
        show_collision=False, show_graspgenx=True,
    )
    return triangles, lines, markers
before = scene()
assert any(t.color == "#c97b30" for t in before[0])
assert not any(t.color == "#f4cf38" for t in before[0])
assert not any("4B2" in item.label for items in before[1:] for item in items)
model.state.production_opening_m += .010
model.state.production_finger_length_m += .010
model.state.production_model_from_tcp_z_m += .010
assert scene() == before
model.state.model_opening_m += .010
assert scene()[0] != before[0]
assert len(scene(True, True)[0]) > len(scene()[0])
original = model.payload()["original_model"]
model.state = _state_from_payload(model.payload())
assert model.payload()["original_model"] == original

app = QApplication([])
window = AlignmentWindow(model)
assert not hasattr(window, "show_output")
assert not any(key.startswith("production_") for key in window.controls)
texts = [w.text() for kind in (QCheckBox, QLabel) for w in window.findChildren(kind)]
texts += [w.title() for w in window.findChildren(QGroupBox)]
assert not any("4B2" in text or "原始 /" in text for text in texts)
window.controls["opening_mm"].setValue(80.)
assert model.state.model_opening_m == .080
assert model.payload()["original_model"] == original
window.reset_defaults()
assert model.state.model_opening_m == owner.DMTAC_M_GRASPGENX_MAX_OPENING_M
assert window.controls["finger_length_mm"].value() == owner.DMTAC_M_GRASPGENX_FINGER_LENGTH_M * 1000
assert window.controls["grasp_center_z_mm"].value() == owner.DMTAC_M_GRASPGENX_GRASP_CENTER_FROM_TACTILE_XYZ_M[2] * 1000
assert not builder.warnings, builder.warnings
window.close()
print(side + ": current-only geometry, controls, defaults and legacy JSON passed")
'''
    result = subprocess.run(
        [python, "-c", program, side],
        cwd=ROOT,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("profile_name", ("v4", "v5"))
def test_actual_tk_load_save_keeps_complete_new_pose_and_selected_reset(tmp_path, monkeypatch, profile_name):
    from utils import graspgenx_dmtac_alignment_gui as gui
    from deployments.perception.dmtac_grasp_overlay import (
        DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID,
        DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID_V5,
        dmtac_graspgenx_alignment_profile,
    )

    class Variable:
        def __init__(self, value):
            self.value = value
        def get(self):
            return self.value
        def set(self, value):
            self.value = value

    profile_id = (DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID if profile_name == "v4"
        else DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID_V5)
    old_pose = [.01, -.02, .14, .03, -.04, .05]
    new_pose = [-.06, .07, .18, -.2, .15, -.1]
    urdf = ROOT / "assets/realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf"
    model = gui.AlignmentModel(urdf, gui.current_dmtac_state("right", old_pose, [0.] * 6,
        alignment_profile_id=profile_id))
    payload = model.payload()
    # The legacy scalar deliberately disagrees: a complete pose is primary.
    payload["profile_unchanged_fields"]["tool_tcp_pose_endpoint_m_rad"] = new_pose
    source, output = tmp_path/"loaded.json", tmp_path/"saved.json"
    source.write_text(json.dumps(payload))
    app = gui.AlignmentApp.__new__(gui.AlignmentApp)
    app.model, app.output, app.alignment_profile_id = model, output, profile_id
    app._alignment_profile = dmtac_graspgenx_alignment_profile(profile_id)
    names = ("side gripper_q tcp_z production_model_from_tcp_z production_opening production_palm_z "
        "production_finger_length production_finger_thickness production_pad_height opening palm_z "
        "finger_length finger_thickness pad_height grasp_center_x grasp_center_y grasp_center_z "
        "corr_x corr_y corr_z corr_roll corr_pitch corr_yaw").split()
    app.vars = {name: Variable("right" if name == "side" else 0.) for name in names}
    app.vars.update({f"q{i}": Variable(0.) for i in range(1, 7)})
    app.status = Variable("")
    refreshes = []
    app.refresh = lambda **kwargs: refreshes.append(kwargs)
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **_: str(source))
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **_: str(output))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args: pytest.fail(str(args)))
    app.load()
    assert model.state.tool_tcp_pose_endpoint_m_rad == new_pose
    assert model.state.tcp_z_m == new_pose[2]
    app.save()
    saved = json.loads(output.read_text())
    assert saved["profile_unchanged_fields"]["tool_tcp_pose_endpoint_m_rad"] == pytest.approx(new_pose)
    assert saved["profile_unchanged_fields"]["tool_tcp_from_endpoint_z_m"] == new_pose[2]
    assert refreshes == [{"fit": True}]
    model.state.model_finger_length_m = .123
    app.reset()
    assert gui._state_tool_tcp_pose(model.state) == pytest.approx(new_pose)
    assert model.state.model_finger_length_m == app._alignment_profile.nominal_geometry.finger_length_m


@pytest.mark.parametrize("pose", (None, [0.] * 5, [0., 0., float("nan"), 0., 0., 0.],
    [0., 0., .14, float("inf"), 0., 0.], [False, 0., .14, 0., 0., 0.]))
def test_complete_tcp_input_is_validated_instead_of_falling_back_to_legacy_z(pose):
    from utils.graspgenx_dmtac_alignment_gui import _state_from_payload
    with pytest.raises(ValueError, match="six finite"):
        _state_from_payload({"profile_unchanged_fields": {
            "tool_tcp_pose_endpoint_m_rad": pose, "tool_tcp_from_flange_z_m": .14}})


def test_legacy_scalar_tcp_still_loads_without_a_complete_pose():
    from utils.graspgenx_dmtac_alignment_gui import _state_from_payload, _state_tool_tcp_pose
    state = _state_from_payload({"profile_unchanged_fields": {"tool_tcp_from_endpoint_z_m": .16}})
    assert _state_tool_tcp_pose(state) == (0., 0., .16, 0., 0., 0.)


@pytest.mark.parametrize("entry", ("headless", "tk_startup"))
def test_v5_startup_rejects_loaded_left_origin_before_relabelling(tmp_path, monkeypatch, capsys, entry):
    import sys
    from utils import graspgenx_dmtac_alignment_gui as gui
    from deployments.perception.dmtac_grasp_overlay import DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID_V5
    source, output = tmp_path/"left.json", tmp_path/"blocked.json"
    content = json.dumps({"side": "left", "profile_unchanged_fields": {
        "tool_tcp_pose_endpoint_m_rad": [0., 0., .14, 0., 0., 0.]}})
    source.write_text(content)
    args = ["alignment", "--urdf", str(ROOT/"assets/realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf"),
        "--side", "right", "--alignment-profile", DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID_V5,
        "--load-alignment", str(source)]
    if entry == "headless":
        args.extend(["--headless-export", str(output)])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(gui.tk, "Tk", lambda: pytest.fail("Reject origin before opening Tk"))
    monkeypatch.setattr(gui, "AlignmentModel", lambda *args: pytest.fail("Reject origin before model assembly"))
    assert gui.main() == 2
    assert "one right-arm" in capsys.readouterr().out
    assert source.read_text() == content and not output.exists()


def test_headless_export_uses_the_explicit_selected_profile(tmp_path):
    from utils.graspgenx_dmtac_alignment_gui import export_payload
    from deployments.perception.dmtac_grasp_overlay import DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID_V5
    output = tmp_path/"right-v5.json"
    export_payload(ROOT/"assets/realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf",
        output, "right", [0.] * 6, None,
        ROOT/"hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json",
        DMTAC_M_GRASPGENX_ALIGNMENT_PROFILE_ID_V5)
    value = json.loads(output.read_text())
    assert value["side"] == "right"
    assert value["graspgenx_model"]["finger_length_m"] == .0867
    assert value["graspgenx_model"]["palm_z_m"] == pytest.approx(-.024)
    expected_tcp = json.loads((ROOT/"hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json").read_text())["tcp"]["pose_flange_m_rad"]
    assert value["profile_unchanged_fields"]["tool_tcp_pose_endpoint_m_rad"] == expected_tcp
