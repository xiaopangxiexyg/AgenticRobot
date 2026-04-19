from __future__ import annotations

HAND_BASE_TCP_FRAME = "hand_base"
LEGACY_RM_EEF_TCP_MODE = "rm_eef"


def canonical_tcp_frame(tcp_mode: str | None = None) -> str:
    if tcp_mode is None:
        return HAND_BASE_TCP_FRAME
    text = str(tcp_mode).strip()
    if text in {"", HAND_BASE_TCP_FRAME, LEGACY_RM_EEF_TCP_MODE}:
        return HAND_BASE_TCP_FRAME
    return text


def legacy_tcp_mode_requested(tcp_mode: str | None = None) -> bool:
    if tcp_mode is None:
        return False
    return str(tcp_mode).strip() == LEGACY_RM_EEF_TCP_MODE
