# 仿真运行入口

MuJoCo 是 task-physics 主链路；SAPIEN 只做 preview。详细说明见 [MUJOCO](./MUJOCO.md)。

从仓库根目录、已激活 `dexteleop310` 的桌面会话执行。带 `--viewer` 的命令需要可用显示环境；
无显示服务器先使用第 1 节的只编译检查。

## 1. 编译 canonical profile（R1）

```bash
python -m sim.mujoco.run_control \
  --contract sim/configs/rm65_bimanual_physics_v1.json \
  --dump-mjcf-only \
  --mjcf-out /tmp/rm65_bimanual_physics_v1.xml
```

## 2. Real-like VIVE position teleop（R1）

先按 [真实系统入口](../real/README.md) 启动 VIVE sender，并让它发送到运行本命令的主机。

```bash
PYTHONPATH=. python -m teleop_runtime.debug.run_vive_mujoco_teleop \
  --left-tool-profile position_endpoint \
  --right-tool-profile position_endpoint \
  --viewer
```

## 3. Canonical dataset replay（R1）

```bash
python -m sim.mujoco.replay \
  --dataset-root "${DATASET_ROOT:?set DATASET_ROOT}" \
  --episode 0 \
  --viewer
```

需要验证 replay-authoritative 的 sim human-in-the-loop capture 时，使用独立的
[HIL 使用说明](../HIL/README.md)。该 profile 是显式、default-off 的 library/test
composition；它与临时 PPO pipeline 的 checkpoint 试用、RTC 和真机 deployment 分开。
