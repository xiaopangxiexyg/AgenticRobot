from __future__ import annotations


def _np():
    import numpy as np

    return np


def interpolate_vectors_linear(start, end, max_distance):
    """Linear interpolation between start and end with per-axis max step length."""
    np = _np()
    start = np.asarray(start)
    end = np.asarray(end)
    max_distance = np.asarray(max_distance)
    diff = np.abs(end - start)
    num_steps = int(np.max(np.ceil(diff / max_distance)))
    if num_steps < 1:
        return []
    return [start + (end - start) * i / num_steps for i in range(1, num_steps + 1)]


def interpolate_vectors_constant_speed(start, end, max_distance):
    """Per-axis constant-speed interpolation with clipping at the target."""
    np = _np()
    start = np.asarray(start)
    end = np.asarray(end)
    diff = end - start
    max_distance = np.asarray(max_distance)
    if max_distance.shape == ():
        max_distance = np.ones_like(diff) * max_distance

    steps_per_dim = np.ceil(np.max(np.abs(diff)) / max_distance)
    max_steps = int(np.max(steps_per_dim))
    direction = np.where(diff != 0, diff / np.abs(diff), 0)
    interpolated_vectors = []
    for step in range(1, max_steps + 1):
        current = start + direction * np.minimum(step * max_distance, np.abs(diff))
        current = np.clip(current, np.minimum(start, end), np.maximum(start, end))
        interpolated_vectors.append(current.copy())
    return interpolated_vectors


def interpolate_6dpose(start, end, max_distance):
    """Interpolate a 6D pose using linear XYZ and Euler-angle SLERP."""
    np = _np()
    from scipy.spatial.transform import Rotation as R, Slerp

    start = np.asarray(start)
    end = np.asarray(end)
    total_distance = np.max(np.abs(end - start))
    num_steps = int(np.ceil(total_distance / max_distance))
    if num_steps < 1:
        return []

    position_a, euler_a = start[0:3], start[3:6]
    position_b, euler_b = end[0:3], end[3:6]
    t_values = np.linspace(0, 1, num=num_steps)[1:]
    interp_pos = [position_a * (1 - t) + position_b * t for t in t_values]
    slerp = Slerp([0, 1], R.from_euler("xyz", [euler_a, euler_b]))
    interp_euler = slerp(t_values).as_euler("xyz")
    return [np.concatenate([position, euler]) for position, euler in zip(interp_pos, interp_euler)]
