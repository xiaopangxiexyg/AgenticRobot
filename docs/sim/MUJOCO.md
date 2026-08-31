# MuJoCo 与仿真参数说明

MuJoCo 是 canonical task-physics backend；运行命令见 [sim README](./README.md)。
历史 SAPIEN operator preview 入口已退役，体系边界见
[`sim/README.md`](../../sim/README.md)。

以下命令按原样运行时均为 R1，不连接真实 RM65。若额外加入 real-arm read 选项，风险至少
上升为 H1，必须回到该入口的硬件说明核对，不能继续沿用本页的 R1 判断。

## 1. 编译 canonical MuJoCo profile

可直接复制的命令见 [sim README 的第 1 节](./README.md)。该模式只
生成/检查 MJCF，不推进物理。

## 2. Real-like VIVE position teleop

先启动 VIVE sender，再运行 [sim README 的第 2 节](./README.md)
中的命令。

该入口运行与真机一致的 15 Hz tracker、60 Hz IK/qref、120 Hz qcmd 语义，最终 plant
替换为 MuJoCo。它不是 force/impedance 仿真。

## 3. Canonical dataset replay

可直接复制的命令见 [sim README 的第 3 节](./README.md)。该入口消费
canonical desk tool-pose action，再运行共享 collision-aware IK 和 actuator path。
UMI 坐标与 overlay 说明见
[`sim/mujoco/replay/README_UMI.md`](../../sim/mujoco/replay/README_UMI.md)。

## 4. Debug-only direct qpos replay

```bash
python -m sim.mujoco.replay.runner \
  --dataset-root "${DATASET_ROOT:?set DATASET_ROOT}" \
  --episode 0 \
  --qpos-source auto \
  --no-render
```

仅用于相机、URDF/MJCF 和 joint recording 诊断。它跳过 canonical action→IK 语义，报告中
必须标记为 kinematic/debug replay。

## 5. Cube-Lift 与性能诊断

单次物理诊断：

```bash
python -m sim.mujoco.diagnostics.cube_lift --help
python -m sim.mujoco.diagnostics.cube_drop --help
```

Fast/vector runtime benchmark：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m sim.mujoco.benchmarks.benchmark_cube_lift \
  --runtime both \
  --vector-envs 1,4,8,16 \
  --steps 500 \
  --warmup-steps 50
```

训练/评估的实验入口仍位于 `sim.mujoco_temp_pipeline`，属于当前临时 pipeline：

```bash
python -m sim.mujoco_temp_pipeline.train --help
python -m sim.mujoco_temp_pipeline.evaluate --help
python -m sim.mujoco_temp_pipeline.play --help
```

不要把这些命令提升为仓库级 production entrypoint；参数与限制见
[`sim/mujoco_temp_pipeline/README.md`](../../sim/mujoco_temp_pipeline/README.md)。
