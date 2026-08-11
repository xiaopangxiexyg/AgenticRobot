from __future__ import annotations

import numpy as np


def embed_arm_qpos(
    *,
    solution_layout: str,
    current_full_q: np.ndarray,
    left_qpos: np.ndarray,
    right_qpos: np.ndarray,
    owner_name: str,
) -> np.ndarray:
    """Embed two 6-DoF arm vectors without mutating the source vector."""

    full_q = np.asarray(current_full_q, dtype=float).copy()
    if solution_layout == "dual_arm_hand" and full_q.shape[0] >= 24:
        full_q[0:6] = left_qpos
        full_q[18:24] = right_qpos
        return full_q
    if full_q.shape[0] == 14:
        full_q[1:7] = left_qpos
        full_q[8:14] = right_qpos
        return full_q
    if full_q.shape[0] == 12:
        full_q[0:6] = left_qpos
        full_q[6:12] = right_qpos
        return full_q
    raise ValueError(
        f"{owner_name} cannot embed arm qpos into solver configuration shape "
        f"{full_q.shape}."
    )


def split_arm_solution(
    *,
    solution_layout: str,
    qpos: np.ndarray,
    owner_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return detached left/right arm views from a supported solver layout."""

    values = np.asarray(qpos, dtype=float)
    if solution_layout == "dual_arm_hand" and values.shape[0] >= 24:
        return values[0:6].copy(), values[18:24].copy()
    if values.shape[0] == 14:
        return values[1:7].copy(), values[8:14].copy()
    if values.shape[0] == 12:
        return values[0:6].copy(), values[6:12].copy()
    raise ValueError(
        f"{owner_name} received unsupported solver output shape: {values.shape}"
    )


def arm_joint_indices(
    *,
    solution_layout: str,
    vector_size: int,
    owner_name: str,
) -> dict[str, np.ndarray]:
    """Resolve arm joint indices for a supported coupled-solver layout."""

    if solution_layout == "dual_arm_hand" and vector_size >= 24:
        return {
            "left": np.arange(0, 6, dtype=int),
            "right": np.arange(18, 24, dtype=int),
        }
    if vector_size == 14:
        return {
            "left": np.arange(1, 7, dtype=int),
            "right": np.arange(8, 14, dtype=int),
        }
    if vector_size == 12:
        return {
            "left": np.arange(0, 6, dtype=int),
            "right": np.arange(6, 12, dtype=int),
        }
    raise ValueError(
        f"{owner_name} cannot resolve arm joint indices for vector size "
        f"{vector_size}."
    )


def arm_joint_entries(
    *,
    solution_layout: str,
    vector_size: int,
    owner_name: str,
) -> tuple[tuple[str, int, int], ...]:
    """Map each local arm joint index to its full solver-vector index."""

    try:
        indices_by_side = arm_joint_indices(
            solution_layout=solution_layout,
            vector_size=vector_size,
            owner_name=owner_name,
        )
    except ValueError as exc:
        raise ValueError(
            f"{owner_name} cannot map arm joint entries for solver shape "
            f"({vector_size},)."
        ) from exc
    return tuple(
        (side, local_index, int(full_index))
        for side in ("left", "right")
        for local_index, full_index in enumerate(indices_by_side[side])
    )
