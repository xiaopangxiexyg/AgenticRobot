from __future__ import annotations

from types import SimpleNamespace

from teleop_runtime.real_teleop.gui_special_arm_status import (
    fix_arm_debug_state,
    fix_arm_status_text,
    fix_arm_ui_state,
    init_arm_debug_state,
    init_arm_status_text,
    init_arm_ui_state,
)
from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint


def test_legacy_special_arm_status_helpers_are_direct_aliases() -> None:
    assert legacy_entrypoint._fix_arm_ui_state is fix_arm_ui_state
    assert legacy_entrypoint._fix_arm_debug_state is fix_arm_debug_state
    assert legacy_entrypoint._fix_arm_status_text is fix_arm_status_text
    assert legacy_entrypoint._init_arm_ui_state is init_arm_ui_state
    assert legacy_entrypoint._init_arm_debug_state is init_arm_debug_state
    assert legacy_entrypoint._init_arm_status_text is init_arm_status_text


class _Controller:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def status_payload(self) -> dict[str, object]:
        return dict(self.payload)


def test_extracted_special_arm_status_views_are_read_only_and_format_stably() -> None:
    components = SimpleNamespace(
        fix_arm_controller=_Controller(
            {
                "fix_arm_available": True,
                "fix_arm_selected_side": "left",
                "fix_arm_fixed_side": "right",
                "fix_arm_active": True,
                "fix_arm_recalibrating": False,
                "fix_arm_status": "fixed",
                "fix_arm_sample_count": 3,
                "fix_arm_min_samples": 5,
            }
        ),
        init_arm_controller=_Controller(
            {
                "init_arm_available": True,
                "init_arm_selected_side": "right",
                "init_arm_active_side": "left",
                "init_arm_active": False,
                "init_arm_recalibrating": True,
                "init_arm_status": "capturing",
                "init_arm_sample_count": 2,
                "init_arm_min_samples": 4,
                "init_arm_profile_path": "/tmp/init.json",
            }
        ),
    )

    fix_state = fix_arm_debug_state(components)
    init_state = init_arm_debug_state(components)
    assert fix_arm_status_text(fix_state) == " fixed(selected=left,fixed=right,samples=3/5)"
    assert init_arm_status_text(init_state) == " recapturing(selected=right,active=left,samples=2/4)"
    assert fix_state["status"] == "fixed"
    assert init_state["status"] == "capturing"
