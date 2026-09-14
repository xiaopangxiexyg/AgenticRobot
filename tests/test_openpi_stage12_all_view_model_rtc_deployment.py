from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parents[1]
_COMMANDS = _REPOSITORY / "deployments/openpi/cmd_list.md"
_RTC_PROFILE = "model_rtc_flow_inpaint_shadow_15_30_60_120_v1"


def _section_23() -> str:
    document = _COMMANDS.read_text(encoding="utf-8")
    marker = "## 23. 当前 Stage1–4 all-view h32 trained model-RTC 部署"
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


def _server_launcher_source() -> str:
    block = _bash_blocks(_section_23())[0]
    match = re.fullmatch(
        r"/home/user/zc/openpi/\.venv/bin/python -u -c '\n(.*)\n' 1 100000",
        block,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def test_stage12_checked_in_profiles_keep_the_same_model_io_contract() -> None:
    training_root = _REPOSITORY / "deployments/openpi/configs/training"
    model_rtc_root = _REPOSITORY / "deployments/openpi/configs/model_rtc"
    expected_layout = (
        "sha256:2567572f0f9170281b3e775ed13ecba903fbdeb199fd88a5bb1b7b1f4e96db16"
    )
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


def test_stage14_remote_recipe_uses_current_checkpoint_and_profile_roots() -> None:
    section = _section_23()

    expected_fragments = {
        1: (
            "writing-stage1-all-valid-all-view-h32-absolute-model-rtc-"
            "systemic-v3-359f1ced-r1-20260906"
        ),
        2: (
            "writing-stage2-all-valid-all-view-h32-absolute-model-rtc-"
            "systemic-v2-b1-nw2-359f1ced-20260909"
        ),
        3: (
            "writing-stage3-all-valid-all-view-h32-absolute-model-rtc-"
            "by-date-semantic-bounds-v2-b1-nw2-359f1ced-20260910"
        ),
        4: (
            "writing-stage4-all-valid-all-view-h32-absolute-model-rtc-"
            "by-date-semantic-bounds-v2-b1-nw2-20260913"
        ),
    }
    for stage, fragment in expected_fragments.items():
        assert f'"{stage}": {{' in section
        assert fragment in section

    assert (
        "20260909_systemic_v2/profiles/"
        "writing_stage2_model_rtc_systemic_v2_b1_nw2.json"
    ) in section
    assert (
        "writing_stage2_all_valid_all_view_h32_absolute_"
        "prepared_model_rtc_v2.json"
    ) in section
    assert "20260910_by_date_v2_semantic_bounds" in section
    assert "20260913_by_date_v2" in section
    assert 'allowed_steps = {"25000", "50000", "75000", "100000"}' in section
    assert "DEFAULT_CKPT_STEP=25006" not in section
    assert "20260904_snapshot_v5" not in section


def test_stage14_server_launcher_is_valid_python_and_fails_closed() -> None:
    source = _server_launcher_source()
    compile(source, "<documented-openpi-server-launcher>", "exec")

    invalid_stage = subprocess.run(
        [sys.executable, "-u", "-c", source, "0", "100000"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid_stage.returncode != 0
    assert "stage must be 1, 2, 3, or 4" in (
        invalid_stage.stdout + invalid_stage.stderr
    )

    invalid_step = subprocess.run(
        [sys.executable, "-u", "-c", source, "1", "100001"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert invalid_step.returncode != 0
    assert "checkpoint step must be 25000, 50000, 75000, or 100000" in (
        invalid_step.stdout + invalid_step.stderr
    )


def test_stage14_client_contract_check_is_server_bootstrapped() -> None:
    section = _section_23()
    contract_section = _subsection(
        section,
        "### 23.2 终端 B：无硬件 contract-check",
        "### 23.3 终端 B：真机 observation no-action shadow",
    )
    blocks = _bash_blocks(contract_section)
    assert len(blocks) == 1
    contract = blocks[0]

    assert "--host 127.0.0.1" in contract
    assert "--port 8078" in contract
    assert f"--rtc-profile {_RTC_PROFILE}" in contract
    assert "--rtc-inference-steps 10" in contract
    assert "--contract-check-only" in contract
    assert "--stage" not in contract
    assert "--checkpoint-step" not in contract
    assert "--checkpoint-dir" not in contract
    assert "--prepared-training-profile" not in contract
    assert "--model-rtc-profile" not in contract
    assert "不是 client 到某个 Stage、step 或 model id 的强绑定" in section
    assert "以新连接实际发布的 metadata 重新建立契约" in section


def test_stage14_real_recipe_separates_no_action_and_bounded_active() -> None:
    section = _section_23()
    blocks = _bash_blocks(section)

    assert len(blocks) == 4
    combined = "\n".join(blocks)
    assert "--enable-rtc-continuation" in combined
    assert "--model-rtc-profile" in combined
    assert f"--rtc-profile {_RTC_PROFILE}" in combined
    assert "--contract-check-only" in combined
    assert "--rtc-real-shadow" in combined
    assert "--strict-model-identity" not in combined
    assert "--rtc-inference-steps 8" not in combined

    no_action = _subsection(
        section,
        "### 23.3 终端 B：真机 observation no-action shadow",
        "### 23.4 终端 B：Stage1 限步 Active（需要单独的现场运动授权）",
    )
    no_action_blocks = _bash_blocks(no_action)
    assert len(no_action_blocks) == 1
    no_action_command = no_action_blocks[0]
    assert "--rtc-real-shadow" in no_action_command
    assert "--send-actions" not in no_action_command
    assert "--policy-qpos-init-dataset" in no_action_command
    assert "--policy-qpos-init-episode 127" in no_action_command
    assert "--policy-qpos-init-frame 0" in no_action_command
    assert "--policy-qpos-init-source auto" in no_action_command
    assert "--policy-qpos-init-collision-check" in no_action_command
    assert "--min-rtc-emissions-before-exit 1" in no_action_command
    assert "plan-only" in no_action
    assert "--no-init-qpos" not in no_action_command
    assert "--checkpoint-dir" not in no_action_command
    assert "--prepared-training-profile" not in no_action_command
    assert "--model-rtc-profile" not in no_action_command

    active = _subsection(
        section,
        "### 23.4 终端 B：Stage1 限步 Active（需要单独的现场运动授权）",
    )
    active_blocks = _bash_blocks(active)
    assert len(active_blocks) == 1
    active_command = active_blocks[0]
    assert "--rtc-real-shadow" in active_command
    assert "--send-actions" in active_command
    assert "--wait-for-ready-confirm" in active_command
    assert "--policy-qpos-init-episode 127" in active_command
    assert "--policy-qpos-init-frame 0" in active_command
    assert "--policy-qpos-init-collision-check" in active_command
    assert "--rtc-active-warmup-discard-completions 1" in active_command
    assert "--rtc-active-warmup-completions 3" in active_command
    assert "--rtc-active-max-inference-latency-sec 0.4" in active_command
    assert "--max-steps 10" in active_command
    assert "--min-rtc-emissions-before-exit 1" in active_command
    assert "--no-init-qpos" not in active_command
    assert "--no-wait-for-ready-confirm" not in active_command
    assert "--checkpoint-dir" not in active_command
    assert "--prepared-training-profile" not in active_command
    assert "--model-rtc-profile" not in active_command

    assert "分别为 `0/109/187/309`" in section
    assert "client 虽然无需知道 Stage/step" in section


def test_stage14_remote_bash_blocks_are_syntactically_valid() -> None:
    for block in _bash_blocks(_section_23()):
        subprocess.run(
            ["bash", "-n"],
            input=block,
            text=True,
            check=True,
        )
