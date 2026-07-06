import argparse
# import pybullet as p
import numpy as np
import time
import sys
from pathlib import Path

# from chaoyi_sandbox.franka_sim.sim import FrankaDexUMI

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from hardware.exo_teleop.arm_exo.motors_config import make_default_arm_params
from hardware.exo_teleop.arm_exo.motors_to_robot import dynamixel_to_robot_arm
from hardware.exo_teleop.fr3_joint_control_client import FR3JointClient


def safe_clip(robot_q_rad, shrink_deg=10.0, print_if_clip=True):
    q = np.asarray(robot_q_rad, dtype=np.float64).reshape(-1)

    q_low = np.deg2rad(np.array([-166, -105, -166, -176, -165,  25, -175], dtype=np.float64))
    q_high = np.deg2rad(np.array([ 166,  105,  166,   -7,  165, 265,  175], dtype=np.float64))

    if q.shape[0] != q_low.shape[0]:
        raise ValueError(f"robot_q_rad dim mismatch: got {q.shape[0]}, expect {q_low.shape[0]}")

    margin = np.deg2rad(float(shrink_deg))
    q_low_safe = q_low + margin
    q_high_safe = q_high - margin

    # sanity check: safe interval must be valid
    bad = q_low_safe >= q_high_safe
    if np.any(bad):
        bad_ids = (np.where(bad)[0] + 1).tolist()
        raise ValueError(f"safe limits invalid after shrink_deg={shrink_deg} for joints {bad_ids}")

    q_clipped = np.clip(q, q_low_safe, q_high_safe)

    clipped_mask = np.abs(q_clipped - q) > 1e-12
    clipped = bool(np.any(clipped_mask))

    if print_if_clip and clipped:
        idx = np.where(clipped_mask)[0]
        parts = []
        for i in idx.tolist():
            parts.append(
                f"J{i+1}: {q[i]: .4f} -> {q_clipped[i]: .4f}  "
                f"limit[{q_low_safe[i]: .4f}, {q_high_safe[i]: .4f}]"
            )
        print("[safe_clip] clipped:", " | ".join(parts))

    return q_clipped, clipped, (q_low_safe, q_high_safe)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="192.168.50.2")
    ap.add_argument("--port", type=int, default=5558)
    ap.add_argument("--timeout_s", type=float, default=5.0)

    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--print_hz", type=float, default=5.0)

    args = ap.parse_args()

    dt = 1.0 / float(args.hz)

    cfg = make_default_arm_params()
    from hardware.exo_teleop.arm_exo.motors_driver import DynamixelDriver

    with FR3JointClient(args.host, args.port, timeout_s=args.timeout_s) as fr3, DynamixelDriver(cfg) as driver:
        # warmup
        driver.set_torque_mode(False)
        time.sleep(0.05)

        for _ in range(3):
            q_dev = driver.get_joints()
            print("[warmup] q_dev =", q_dev)
            time.sleep(0.02)

        # read initial states
        st0 = fr3.get_state()
        q_fr3_0 = np.array(st0["q"], dtype=np.float64)
        ndof = q_fr3_0.size

        # device baseline -> mapped baseline
        q_dev0 = driver.get_joints()
        q_map0 = np.array(dynamixel_to_robot_arm(q_dev0), dtype=np.float64).reshape(ndof)

        print("[teleop] ready.")
        print("  controls: (enter) q=quit, s=stop(hold), r=recalib baseline, p=pause/resume")
        print("  q_fr3_0 =", q_fr3_0.tolist())
        print("  q_map0  =", q_map0.tolist())

        next_print_t = time.time()
        print_dt = 1.0 / args.print_hz if args.print_hz > 0.0 else None

        while True:
            q = driver.get_joints()
            robot_arm_joint = dynamixel_to_robot_arm(q)

            robot_arm_joint_safe, clipped, _ = safe_clip(robot_arm_joint, shrink_deg=10.0, print_if_clip=True)

            fr3.send_q(robot_arm_joint_safe.tolist())

            if print_dt is not None and time.time() >= next_print_t:
                try:
                    st = fr3.get_state()
                    q_now = st.get("q", None)
                    dq_now = st.get("dq", None)
                    print(f"[state] t={st.get('t', None):.3f} q={q_now} dq={dq_now}")
                except Exception as e:
                    print(f"[warn] get_state failed: {e}")
                next_print_t += print_dt

            time.sleep(dt)


if __name__ == "__main__":
    main()
