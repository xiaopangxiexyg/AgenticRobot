from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

Array = np.ndarray

@dataclass(frozen=True)
class AffineMap:
    q_m0: Array
    q_r0: Array
    sign: Array
    scale: Array

    @staticmethod
    def _as_vec(x: Sequence[float] | Array, n: int, name: str) -> Array:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 1 or len(arr) != n:
            raise ValueError(f"{name} 需要长度 {n}，得到形状 {arr.shape}")
        return arr

    @classmethod
    def from_params(
        cls,
        q_m0: Sequence[float] | Array,
        q_r0: Sequence[float] | Array,
        sign: Sequence[float] | Array | None = None,
        scale: Sequence[float] | Array | None = None,
    ) -> "AffineMap":
        n = len(q_m0)
        q_m0 = cls._as_vec(q_m0, n, "q_m0")
        q_r0 = cls._as_vec(q_r0, n, "q_r0")
        if sign is None:
            sign = np.ones(n, dtype=float)            # 方向未标定时默认 +1
        if scale is None:
            scale = np.ones(n, dtype=float)           # 比例未标定时默认 1
        sign  = cls._as_vec(sign,  n, "sign")
        scale = cls._as_vec(scale, n, "scale")
        return cls(q_m0=q_m0, q_r0=q_r0, sign=sign, scale=scale)

    def to_robot(self, q_m: Sequence[float] | Array) -> Array:
        q_m = np.asarray(q_m, dtype=float)
        if q_m.shape != self.q_m0.shape:
            raise ValueError(f"q_m 长度应为 {len(self.q_m0)}，得到 {q_m.shape}")
        return self.scale * self.sign * (q_m - self.q_m0) + self.q_r0

    def to_motor(self, q_r: Sequence[float] | Array) -> Array:
        q_r = np.asarray(q_r, dtype=float)
        if q_r.shape != self.q_r0.shape:
            raise ValueError(f"q_r 长度应为 {len(self.q_r0)}，得到 {q_r.shape}")
        denom = self.scale * self.sign
        if np.any(np.isclose(denom, 0.0)):
            raise ZeroDivisionError("scale 或 sign 含零，无法反解")
        return (q_r - self.q_r0) / denom + self.q_m0



# franka
JOINT_IDS = (21,22,23,24,25,26,27)
Q_M0 = np.array(np.deg2rad([180, 180, 180, 180, 180, 180, 180]), dtype=float)   # 电机零位(rad)
Q_R0 = np.array([0.0,  0.0,  0.0, -1.57,  0.0,  1.57,  0.0], dtype=float)   # 机器人零位(rad)
SIGN = np.array([1, 1, 1, -1, 1, -1, 1], dtype=float)   # 每关节 +1/-1
SCALE= np.array(np.ones(7), dtype=float)   # 每关节比例
_MAP = AffineMap.from_params(Q_M0, Q_R0, SIGN, SCALE)


def set_mapping(
    joint_ids: Iterable[int],
    q_m0: Sequence[float],
    q_r0: Sequence[float],
    sign: Sequence[float] | None = None,
    scale: Sequence[float] | None = None,
) -> None:
    """在程序启动时调用一次，设置映射参数。"""
    global JOINT_IDS, _MAP
    ids = tuple(int(i) for i in joint_ids)
    JOINT_IDS = ids
    _MAP = AffineMap.from_params(q_m0=q_m0, q_r0=q_r0, sign=sign, scale=scale)

def _require_map() -> AffineMap:
    if _MAP is None:
        raise RuntimeError("映射未设置。请先调用 set_mapping(...) 或在文件内填写 MAP")
    return _MAP

def dynamixel_to_robot_arm(q_m: Iterable[float]) -> list[float]:
    """电机读数(rad) → 机器人关节(rad)。映射需先通过 set_mapping 设置。"""
    amap = _require_map()
    q_m = np.asarray(list(q_m), dtype=float)
    return amap.to_robot(q_m).tolist()

def robot_to_dynamixel(q_r: Iterable[float]) -> list[float]:
    """机器人关节(rad) → 电机目标(rad)。"""
    amap = _require_map()
    q_r = np.asarray(list(q_r), dtype=float)
    return amap.to_motor(q_r).tolist()


if __name__ == "__main__":

    set_mapping(
        joint_ids=(0,1,2,3,4,5),
        q_m0=np.zeros(6),
        q_r0=np.zeros(6),
        sign=np.ones(6),
        scale=np.ones(6),
    )
    q_m = np.deg2rad([10,20,30,40,50,60])
    q_r = dynamixel_to_robot_arm(q_m)
    q_m_back = robot_to_dynamixel(q_r)
    assert np.allclose(q_m, q_m_back, atol=1e-9)
    print("motors_to_robot mapping OK")
