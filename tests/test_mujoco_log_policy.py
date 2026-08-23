from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.optional


def test_pytest_keeps_mujoco_default_log_out_of_current_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    config = mujoco.MjLogConfig.get()

    assert config.logto_console is True
    assert config.logto_file is False

    monkeypatch.chdir(tmp_path)
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <joint type="hinge"/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qpos[0] = float("nan")

    mujoco.mj_checkPos(model, data)

    assert not (tmp_path / "MUJOCO_LOG.TXT").exists()
