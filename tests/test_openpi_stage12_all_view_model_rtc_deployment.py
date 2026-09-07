from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMMANDS = _REPOSITORY / "deployments/openpi/cmd_list.md"
_RTC_PROFILE = "model_rtc_flow_inpaint_shadow_15_30_60_120_v1"


def _section_23() -> str:
    document = _COMMANDS.read_text(encoding="utf-8")
    marker = (
        "## 23. 当前 corrected Stage1 v2 / historical Stage2 v1 "
        "all-view h32 trained model-RTC 部署测试"
    )
    assert marker in document
    return document.split(marker, maxsplit=1)[1]


def _bash_blocks(text: str) -> list[str]:
    return re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL)


def _subsection(text: str, heading: str, next_heading: str | None = None) -> str:
    assert heading in text
    subsection = text.split(heading, maxsplit=1)[1]
    if next_heading is not None:
        assert next_heading in subsection
        subsection = subsection.split(next_heading, maxsplit=1)[0]
    return subsection


def test_stage12_profiles_keep_the_same_deployable_model_io_contract() -> None:
    training_root = _REPOSITORY / "deployments/openpi/configs/training"
    model_rtc_root = _REPOSITORY / "deployments/openpi/configs/model_rtc"
    expected_layout = "sha256:2567572f0f9170281b3e775ed13ecba903fbdeb199fd88a5bb1b7b1f4e96db16"
    model_rtc_profiles = {
        1: "writing_stage1_all_valid_all_view_h32_absolute_prepared_model_rtc_v2.json",
        2: "writing_stage2_all_valid_all_view_h32_absolute_prepared_model_rtc_v1.json",
    }

    for stage in (1, 2):
        prepared = json.loads(
            (
                training_root
                / f"writing_stage{stage}_all_valid_all_view_h32_absolute_non_rtc_100k_v1.json"
            ).read_text(encoding="utf-8")
        )
        model_rtc = json.loads(
            (model_rtc_root / model_rtc_profiles[stage]).read_text(encoding="utf-8")
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

    stage1_model_rtc = json.loads(
        (model_rtc_root / model_rtc_profiles[1]).read_text(encoding="utf-8")
    )
    assert stage1_model_rtc["schema"].endswith(".v2")
    assert stage1_model_rtc["training_implementation"] == (
        "mask_aware_hard_flow_prefix_v2"
    )
    assert stage1_model_rtc["rtc_prefix_mask_conditioning"] is True

    stage2_model_rtc = json.loads(
        (model_rtc_root / model_rtc_profiles[2]).read_text(encoding="utf-8")
    )
    assert stage2_model_rtc["schema"].endswith(".v1")
    assert "training_implementation" not in stage2_model_rtc
    assert "rtc_prefix_mask_conditioning" not in stage2_model_rtc


def test_stage12_remote_recipe_separates_server_and_client_checkpoint_identity() -> None:
    section = _section_23()

    assert "SERVER_CHECKPOINT_DIR" in section
    assert "CLIENT_CHECKPOINT_DIR" in section
    assert "SERVER_CKPT_STEP" in section
    assert "CLIENT_CKPT_STEP" in section
    assert "writing_stage1_all_valid_all_view_h32_absolute_prepared_model_rtc_v2.json" in section
    assert "writing_stage2_all_valid_all_view_h32_absolute_prepared_model_rtc_v1.json" in section
    assert "writing-stage1-all-valid-all-view-h32-absolute-model-rtc-systemic-v3" in section
    assert "writing-stage2-all-valid-all-view-h32-absolute-model-rtc" in section
    assert "DEFAULT_CKPT_STEP=25006" in section
    assert "DEFAULT_CKPT_STEP=100000" in section
    assert "EXPECTED_DEPLOY_COMMIT" in section
    assert "TRAINING_OPENPI_COMMIT" in section
    assert "RTC_RUNTIME_OPENPI_ROOT" in section

    stage1_case = re.search(
        r'case "\$STAGE" in\n  1\)(.*?)    ;;',
        section,
        flags=re.DOTALL,
    )
    assert stage1_case is not None
    assert "prepared_model_rtc_v2.json" in stage1_case.group(1)
    assert "prepared_model_rtc_v1.json" not in stage1_case.group(1)


def test_stage12_real_recipe_separates_no_action_and_bounded_active() -> None:
    section = _section_23()
    blocks = _bash_blocks(section)

    assert len(blocks) >= 6
    combined = "\n".join(blocks)
    assert "--enable-rtc-continuation" in combined
    assert "--model-rtc-profile" in combined
    assert f"--rtc-profile {_RTC_PROFILE}" in combined
    assert "--contract-check-only" in combined
    assert "--prepared-config-check-only" in combined
    assert "--rtc-real-shadow" in combined
    assert "--strict-model-identity" not in combined

    no_action = _subsection(
        section,
        "### 23.4 终端 B：100-step 真机 observation no-action shadow",
        "### 23.5 终端 B：Stage1 小步数 Active（必须另行获得现场运动授权）",
    )
    assert "--rtc-real-shadow" in no_action
    assert "--send-actions" not in no_action
    assert "--policy-qpos-init-dataset" in no_action
    assert "--policy-qpos-init-episode" in no_action
    assert "--policy-qpos-init-frame" in no_action
    assert "--policy-qpos-init-source" in no_action
    assert "--policy-qpos-init-collision-check" in no_action
    assert "plan-only" in no_action
    assert "不会声称机械臂已经到达数据集起点" in no_action
    assert "--no-init-qpos" not in "\n".join(_bash_blocks(no_action))

    active = _subsection(
        section,
        "### 23.5 终端 B：Stage1 小步数 Active（必须另行获得现场运动授权）",
    )
    active_blocks = "\n".join(_bash_blocks(active))
    assert 'if [ "$STAGE" != 1 ]' in active
    assert "--rtc-real-shadow" in active
    assert "--send-actions" in active
    assert "--wait-for-ready-confirm" in active
    assert "--policy-qpos-init-dataset" in active
    assert "--policy-qpos-init-episode" in active
    assert "--policy-qpos-init-frame" in active
    assert "--policy-qpos-init-source" in active
    assert "--policy-qpos-init-collision-check" in active
    assert "--max-steps 10" in active
    assert "--no-init-qpos" not in active_blocks
    assert "--no-wait-for-ready-confirm" not in active_blocks


def test_stage12_remote_bash_blocks_are_syntactically_valid() -> None:
    for block in _bash_blocks(_section_23()):
        subprocess.run(
            ["bash", "-n"],
            input=block,
            text=True,
            check=True,
        )
