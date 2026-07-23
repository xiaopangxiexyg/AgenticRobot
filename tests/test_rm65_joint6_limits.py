from __future__ import annotations

import json
import math
from pathlib import Path
from xml.etree import ElementTree

import pytest

from robot_motion.rm65.constants import (
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
)


J6_LIMIT_RAD = math.radians(120.0)


def test_all_active_rm65_urdf_j6_limits_are_120_degrees() -> None:
    checked_files: list[Path] = []
    checked_joints = 0

    for path in sorted(Path("assets").rglob("*.urdf")):
        lowered_parts = {part.lower() for part in path.parts}
        if "backup" in lowered_parts or "sapien" in lowered_parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "arm_joint6" not in text:
            continue

        root = ElementTree.fromstring(text)
        joints = [
            joint
            for joint in root.findall("joint")
            if str(joint.attrib.get("name", "")).endswith("arm_joint6")
        ]
        if not joints:
            continue
        checked_files.append(path)
        checked_joints += len(joints)
        for joint in joints:
            limit = joint.find("limit")
            assert limit is not None, f"{path}: {joint.attrib['name']} has no limit"
            assert float(limit.attrib["lower"]) == pytest.approx(-J6_LIMIT_RAD)
            assert float(limit.attrib["upper"]) == pytest.approx(J6_LIMIT_RAD)

    assert len(checked_files) == 27
    assert checked_joints == 43


def test_shared_and_task_config_j6_limits_match_urdf() -> None:
    assert RM65_ABSOLUTE_Q_MIN_RAD[5] == pytest.approx(-J6_LIMIT_RAD)
    assert RM65_ABSOLUTE_Q_MAX_RAD[5] == pytest.approx(J6_LIMIT_RAD)

    checked_configs = 0
    for path in sorted(Path("sim/configs/tasks").glob("rm65*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "right_qpos_lower" not in payload or "right_qpos_upper" not in payload:
            continue
        checked_configs += 1
        assert float(payload["right_qpos_lower"][5]) == pytest.approx(-J6_LIMIT_RAD)
        assert float(payload["right_qpos_upper"][5]) == pytest.approx(J6_LIMIT_RAD)

    assert checked_configs == 3
