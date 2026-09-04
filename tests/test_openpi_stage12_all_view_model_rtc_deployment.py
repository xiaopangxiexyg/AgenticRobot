from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess


_REPOSITORY = Path(__file__).resolve().parents[1]
_COMMANDS = _REPOSITORY / "deployments/openpi/cmd_list.md"
_RTC_PROFILE = "model_rtc_flow_inpaint_shadow_15_30_60_120_v1"


def _section_23() -> str:
    document = _COMMANDS.read_text(encoding="utf-8")
    marker = "## 23. 当前 Stage1/Stage2 all-view h32 trained model-RTC 部署测试"
    assert marker in document
    return document.split(marker, maxsplit=1)[1]


def _bash_blocks(text: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL)


def test_stage12_profiles_keep_the_same_deployable_model_io_contract() -> None:
    training_root = _REPOSITORY / "deployments/openpi/configs/training"
    model_rtc_root = _REPOSITORY / "deployments/openpi/configs/model_rtc"
    expected_layout = "sha256:2567572f0f9170281b3e775ed13ecba903fbdeb199fd88a5bb1b7b1f4e96db16"

    for stage in (1, 2):
        prepared = json.loads(
            (
                training_root
                / f"writing_stage{stage}_all_valid_all_view_h32_absolute_non_rtc_100k_v1.json"
            ).read_text(encoding="utf-8")
        )
        model_rtc = json.loads(
            (
                model_rtc_root
                / f"writing_stage{stage}_all_valid_all_view_h32_absolute_prepared_model_rtc_v1.json"
            ).read_text(encoding="utf-8")
        )

        assert prepared["camera_selection"] == [
            "front",
            "left_wrist",
            "right_wrist",
        ]
        assert prepared["action_horizon"] == 32
        assert prepared["action_pose_mode"] == "absolute"
        assert prepared["expected_policy_io_layout_hash"] == expected_layout
        assert model_rtc["prepared_skill_profile_id"] == prepared["profile_id"]
        assert "checkpoint_dir" not in model_rtc
        assert "continuation_contract_hash" not in model_rtc


def test_stage12_remote_recipe_separates_server_and_client_checkpoint_identity() -> None:
    section = _section_23()

    assert "SERVER_CHECKPOINT_DIR" in section
    assert "CLIENT_CHECKPOINT_DIR" in section
    assert "SERVER_CKPT_STEP" in section
    assert "CLIENT_CKPT_STEP" in section
    assert "writing_stage1_all_valid_all_view_h32_absolute_prepared_model_rtc_v1.json" in section
    assert "writing_stage2_all_valid_all_view_h32_absolute_prepared_model_rtc_v1.json" in section
    assert "writing-stage1-all-valid-all-view-h32-absolute-model-rtc" in section
    assert "writing-stage2-all-valid-all-view-h32-absolute-model-rtc" in section


def test_stage12_real_recipe_is_model_rtc_no_action_only() -> None:
    section = _section_23()
    blocks = _bash_blocks(section)

    assert len(blocks) >= 5
    combined = "\n".join(blocks)
    assert "--enable-rtc-continuation" in combined
    assert "--model-rtc-profile" in combined
    assert f"--rtc-profile {_RTC_PROFILE}" in combined
    assert "--contract-check-only" in combined
    assert "--prepared-config-check-only" in combined
    assert "--rtc-real-shadow" in combined
    assert "--send-actions" not in combined
    assert "--strict-model-identity" not in combined


def test_stage12_remote_bash_blocks_are_syntactically_valid() -> None:
    for block in _bash_blocks(_section_23()):
        subprocess.run(
            ["bash", "-n"],
            input=block,
            text=True,
            check=True,
        )
